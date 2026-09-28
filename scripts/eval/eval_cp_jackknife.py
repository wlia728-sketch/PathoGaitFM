"""Experiment 1 — SimTK CP external: subject-level jackknife + subject-clustered bootstrap.

Shows the external CP vGRF result is not driven by a single subject, and gives an honest
subject-aware CI (cycles are nested within subjects, so a naive over-cycles CI overstates
precision). This is the subject-macro / jackknife form the paper (Table 1, Fig 3) reports
for Leuven CP, 0.847, with its subject-clustered interval.
Reuses the manuscript-method inference (tweedie_inpaint_drop: TSA + R/L merged, ALL-DATA 8ch
model). Reports:
  - per-subject vGRF PCC (subject-macro: per-cycle PCC averaged within subject)
  - leave-one-subject-out jackknife: cohort mean recomputed dropping each subject in turn
  - subject-clustered bootstrap 95% CI (resample SUBJECTS with replacement)

Run with Windows torch+CUDA python. Writes outputs/recomputed/_cp_jackknife.json. ASCII stdout.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from site_summary import load_site_summary, site_mass
from guarded_write import guarded_dump
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS
from drop_channels import DROP_16CH
from eval_common import load_model, safe_pcc, ZSTATS_PATH, SEED, build_known_mask, per_subject_transform
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

CKPT = ROOT / "checkpoints" / "v4_stage2_final8ch_ALLDATA" / "final.pt"
PROC = ROOT / "data" / "external" / "leuven_cp" / "processed"
R_VGRF, L_VGRF = 22, 25


_ZS = json.load(open(ZSTATS_PATH))


def denorm(x_z, ch40):
    """z-scored -> physical body weight, x_BW = x_z * (3 * std) + mean, using the frozen training stats."""
    c54 = str(KEEP_CHANNELS_54TO40[ch40])
    if c54 not in _ZS:
        return None
    return x_z * (3.0 * _ZS[c54]["std"]) + _ZS[c54]["mean"]


def rmse_nrmse_pct(gt_z, pred_z, ch40):
    """RMSE in %BW and range-normalised nRMSE in per cent, for one scored limb-cycle."""
    g = denorm(np.asarray(gt_z, float), ch40); p = denorm(np.asarray(pred_z, float), ch40)
    if g is None:
        return None, None
    rm = float(np.sqrt(np.mean((g - p) ** 2)))
    rng_gt = float(np.max(g) - np.min(g))
    return rm * 100.0, (100.0 * rm / rng_gt if rng_gt > 1e-9 else None)


ERR_CI_SEED = 20260726


def boot_ci(vals, nd=3, n_boot=5000):
    """Subject-clustered 95% CI: resample SUBJECTS with replacement (same rule as the PCC interval).
    Uses its OWN generator, seeded independently, so adding these intervals does not advance the shared
    stream the PCC bootstrap draws from and the published PCC intervals reproduce unchanged."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return None
    r = np.random.default_rng(ERR_CI_SEED)
    b = v[r.integers(0, v.size, (n_boot, v.size))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [round(float(lo), nd), round(float(hi), nd)]


def grp_cond(stem):
    s = stem.lower()
    return ("cp", 0, 0) if (s.startswith("hecp") or s.startswith("dicp")) else ("td", 1, 5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CP jackknife (vGRF, manuscript method) | device", device, flush=True)
    model, ckpt, _, imt = load_model(CKPT, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None: _i = ckpt.get("input_mask_token")
        if _i is not None: imt = _i
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)
    meta = V4SubjectMetadata()
    _sp = PROC / "_summary.json"
    for d in load_site_summary(_sp):
        if d.get("subj"):
            meta.bw_table[("ext_cp", d["subj"])] = site_mass(d, _sp)
    transform = per_subject_transform(meta, require_tables=False)

    skipped = []          # (file, reason) for every cycle file this run could not use
    rng = np.random.default_rng(SEED)
    # per-subject list of per-cycle vGRF PCCs, per group
    subj_pccs = {"cp": {}, "td": {}}
    # Error accumulators keyed identically to subj_pccs and filled inside the same validity guard, so the
    # RMSE cells are the same scored limb-cycles as the PCC cells and share one aggregation rule.
    subj_rmse = {"cp": {}, "td": {}}
    subj_nrmse = {"cp": {}, "td": {}}
    for f in sorted(PROC.glob("*.npy")):
        if f.name.endswith("_mask.npy"):
            continue
        stem = f.stem
        grp, cohort_id, severity_id = grp_cond(stem)
        raw = np.load(str(f))
        try:
            x40 = transform(raw.copy(), "ext_cp", "cohort", stem)[:, :100, :].astype(np.float32)
        except Exception as exc:
            skipped.append((f.name, "transform failed: %s" % type(exc).__name__))
            continue
        m = np.load(str(f).replace(".npy", "_mask.npy"))
        m40 = m[:, KEEP_CHANNELS_54TO40].astype(bool)
        keep = np.array([ci for ci in range(x40.shape[0]) if m40[ci, R_VGRF] or m40[ci, L_VGRF]])
        if len(keep) == 0:
            skipped.append((f.name, "no cycle has a valid vertical GRF on either side"))
            continue
        sel = rng.choice(keep, size=len(keep), replace=False)  # every valid cycle (size=len: none dropped)
        nb = len(sel)
        gt = torch.from_numpy(np.nan_to_num(x40[sel].transpose(0, 2, 1), nan=0.0)).to(device)
        valid = torch.from_numpy(m40[sel]).to(device)
        sid = torch.full((nb,), model.num_sources, dtype=torch.long, device=device)  # reserved unknown-external source id
        cid = torch.full((nb,), cohort_id, dtype=torch.long, device=device)
        svid = torch.full((nb,), severity_id, dtype=torch.long, device=device)
        gens = []
        for so in range(3):
            torch.manual_seed(SEED + so * 7919)
            with torch.no_grad():
                gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                                 50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
        g = torch.stack(gens).mean(0)
        km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
        g = g.cpu().numpy(); gtn = gt.cpu().numpy(); vm = m40[sel]
        rs = []; es = []; ns = []
        for ci in range(nb):
            for ch in (R_VGRF, L_VGRF):
                if vm[ci, ch]:
                    r = safe_pcc(gtn[ci, ch], g[ci, ch])
                    if np.isfinite(r):
                        rs.append(r)
                        rm, nrm = rmse_nrmse_pct(gtn[ci, ch], g[ci, ch], ch)
                        if rm is not None and np.isfinite(rm): es.append(rm)
                        if nrm is not None and np.isfinite(nrm): ns.append(nrm)
        if rs:
            subj_pccs[grp][stem] = rs
            if es: subj_rmse[grp][stem] = es
            if ns: subj_nrmse[grp][stem] = ns

    if skipped:
        print("  %d cycle file(s) skipped:" % len(skipped), flush=True)
        for name, why in skipped:
            print("    %-16s %s" % (name, why), flush=True)
    EXPECTED = {"cp": 7, "td": 5}          # the n this dataset contributes to Table 1
    got = {g: len(subj_pccs[g]) for g in EXPECTED}
    if got != EXPECTED:
        sys.exit("subject count mismatch: expected %s, scored %s; %d file(s) skipped"
                 % (EXPECTED, got, len(skipped)))
    print("  subject counts as expected: %s" % got, flush=True)

    out = {}
    for grp in ("cp", "td"):
        d = subj_pccs[grp]
        subjects = sorted(d.keys())
        subj_macro = {s: float(np.mean(d[s])) for s in subjects}   # subject-macro PCC
        macro_vals = np.array([subj_macro[s] for s in subjects])
        overall = float(np.mean(macro_vals))
        # leave-one-subject-out jackknife
        jack = {}
        for s in subjects:
            others = [subj_macro[x] for x in subjects if x != s]
            jack[s] = round(float(np.mean(others)), 3)
        jack_vals = np.array(list(jack.values()))
        # subject-clustered bootstrap (resample subjects with replacement)
        boot = []
        for _ in range(5000):
            idx = rng.choice(len(macro_vals), len(macro_vals), replace=True)
            boot.append(np.mean(macro_vals[idx]))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        # error, aggregated exactly like the PCC macro: per-subject mean, then mean over subjects
        subj_rm = {s: float(np.mean(v)) for s, v in subj_rmse[grp].items() if v}
        subj_nr = {s: float(np.mean(v)) for s, v in subj_nrmse[grp].items() if v}
        out[grp] = {
            "n_subj": len(subjects),
            "overall_subject_macro_pcc": round(overall, 3),
            "subject_clustered_ci95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
            "RMSE_pctBW_subject_macro": round(float(np.mean(list(subj_rm.values()))), 2) if subj_rm else None,
            "RMSE_pctBW_ci95": boot_ci(list(subj_rm.values()), nd=2),
            "nRMSE_pct_subject_macro": round(float(np.mean(list(subj_nr.values()))), 1) if subj_nr else None,
            "nRMSE_pct_ci95": boot_ci(list(subj_nr.values()), nd=1),
            "per_subject_rmse_pctBW": {s: round(v, 2) for s, v in subj_rm.items()},
            "per_subject_nrmse_pct": {s: round(v, 1) for s, v in subj_nr.items()},
            "per_subject_pcc": {s: round(v, 3) for s, v in subj_macro.items()},
            "jackknife_leave_one_subject_out_mean": jack,
            "jackknife_range": [round(float(jack_vals.min()), 3), round(float(jack_vals.max()), 3)],
            "jackknife_max_swing": round(float(jack_vals.max() - jack_vals.min()), 3),
        }
    saved_path = guarded_dump(out, ROOT / "outputs/evaluation/cp_jackknife.json", "eval_cp_jackknife.py")
    print("\n=== CP/TD subject-level jackknife (vGRF, manuscript method) ===", flush=True)
    for grp in ("cp", "td"):
        o = out[grp]
        print(f"\n[{grp.upper()}] n_subj={o['n_subj']}  subject-macro PCC={o['overall_subject_macro_pcc']}  "
              f"subject-clustered 95% CI {o['subject_clustered_ci95']}", flush=True)
        print(f"  per-subject: {o['per_subject_pcc']}", flush=True)
        print(f"  leave-one-subject-out mean range {o['jackknife_range']} "
              f"(max swing {o['jackknife_max_swing']}) -> NOT single-subject-driven if swing small", flush=True)
    print("\nsaved", saved_path, flush=True)


if __name__ == "__main__":
    main()
