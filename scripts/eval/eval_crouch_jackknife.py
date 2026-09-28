"""crouch external CP: subject-level jackknife + subject-clustered bootstrap CI for vGRF.
Mirrors eval_cp_jackknife.py EXACTLY (same TSA + R/L merged, ALL-DATA 8ch model, subject-macro),
adapted for crouchgait (all CP, cohort_id=0) and reporting BOTH PCC and RMSE(%BW) CIs since the
crouch result uses PCC + RMSE. vGRF-only (per the scope decision).

Reports:
  - per-subject vGRF PCC + RMSE(%BW)  (subject-macro: per-cycle averaged within subject)
  - leave-one-subject-out jackknife (cohort mean dropping each subject)
  - subject-clustered bootstrap 95% CI (resample SUBJECTS with replacement, 5000x)

Writes outputs/recomputed/_crouch_jackknife.json. Windows torch+CUDA. ASCII stdout.
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
PROC = ROOT / "data" / "external" / "gillette_crouch" / "processed"
R_VGRF, L_VGRF = 22, 25
KEEP = list(KEEP_CHANNELS_54TO40); INV = {i: c for i, c in enumerate(KEEP)}


def denorm(x, ch40, z):
    c = str(INV[ch40])
    if c not in z: return x
    return x * (3.0 * z[c]["std"]) + z[c]["mean"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("crouch jackknife (vGRF, manuscript method) | device", device, flush=True)
    z = json.load(open(ZSTATS_PATH))
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
            meta.bw_table[("ext_cp2", d["subj"])] = site_mass(d, _sp)
    transform = per_subject_transform(meta, require_tables=False)

    skipped = []          # (file, reason) for every cycle file this run could not use
    rng = np.random.default_rng(SEED)
    subj_pcc = {}   # stem -> list per-cycle vGRF PCC
    subj_rmse = {}  # stem -> list per-cycle vGRF RMSE (%BW)
    subj_nrmse = {}  # stem -> list per-cycle vGRF nRMSE (per cent of the cycle's ground-truth range)
    for f in sorted(PROC.glob("*.npy")):
        if f.name.endswith("_mask.npy"):
            continue
        stem = f.stem
        raw = np.load(str(f))
        try:
            x40 = transform(raw.copy(), "ext_cp2", "cohort", stem)[:, :100, :].astype(np.float32)
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
        cid = torch.zeros(nb, dtype=torch.long, device=device)   # all CP
        svid = torch.zeros(nb, dtype=torch.long, device=device)  # severity_id 0, the only bin cerebral palsy occurs with in training
        gens = []
        for so in range(3):
            torch.manual_seed(SEED + so * 7919)
            with torch.no_grad():
                gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                                 50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
        g = torch.stack(gens).mean(0)
        km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
        g = g.cpu().numpy(); gtn = gt.cpu().numpy(); vm = m40[sel]
        rs, rms, nrms = [], [], []
        for ci in range(nb):
            for ch in (R_VGRF, L_VGRF):
                if vm[ci, ch]:
                    r = safe_pcc(gtn[ci, ch], g[ci, ch])
                    # The error is accumulated INSIDE this guard, so that it
                    # covered cycles the correlation had rejected. Both now score the same limb-cycles.
                    if np.isfinite(r):
                        rs.append(r)
                        gw = denorm(gtn[ci, ch], ch, z); pw = denorm(g[ci, ch], ch, z)
                        rm = rmse(gw, pw)
                        rms.append(rm * 100)                                    # %BW
                        rng_gt = float(np.max(gw) - np.min(gw))
                        if rng_gt > 1e-9:
                            nrms.append(100.0 * rm / rng_gt)                    # per cent of GT range
        if rs:
            subj_pcc[stem] = rs
            subj_rmse[stem] = rms
            if nrms: subj_nrmse[stem] = nrms

    subjects = sorted(subj_pcc.keys())
    macro_pcc = np.array([np.mean(subj_pcc[s]) for s in subjects])
    macro_rmse = np.array([np.mean(subj_rmse[s]) for s in subjects])
    nrmse_subjects = [s for s in subjects if s in subj_nrmse]
    macro_nrmse = np.array([np.mean(subj_nrmse[s]) for s in nrmse_subjects])

    def boot_ci(vals):
        b = [np.mean(vals[rng.choice(len(vals), len(vals), replace=True)]) for _ in range(5000)]
        return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]

    if skipped:
        print("  %d cycle file(s) skipped:" % len(skipped), flush=True)
        for name, why in skipped:
            print("    %-16s %s" % (name, why), flush=True)
    EXPECTED_N = 10          # the n this dataset contributes to Table 1
    if len(subjects) != EXPECTED_N:
        sys.exit("subject count mismatch: expected %d, scored %d; %d file(s) skipped"
                 % (EXPECTED_N, len(subjects), len(skipped)))
    print("  subject count as expected: %d" % len(subjects), flush=True)

    # leave-one-subject-out jackknife (PCC)
    jack = {s: round(float(np.mean([np.mean(subj_pcc[x]) for x in subjects if x != s])), 3) for s in subjects}
    jv = np.array(list(jack.values()))

    out = {
        "n_subj": len(subjects), "axis": "vGRF", "method": "TSA R/L-merged, subject-macro, 5000x subject-clustered bootstrap",
        "PCC": {
            "subject_macro": round(float(macro_pcc.mean()), 3),
            "subject_clustered_ci95": boot_ci(macro_pcc),
        },
        "RMSE_pctBW": {
            "subject_macro": round(float(macro_rmse.mean()), 2),
            "subject_clustered_ci95": boot_ci(macro_rmse),
        },
        "nRMSE_pct": {
            "subject_macro": round(float(macro_nrmse.mean()), 1) if macro_nrmse.size else None,
            "subject_clustered_ci95": boot_ci(macro_nrmse) if macro_nrmse.size > 1 else None,
        },
        "per_subject_pcc": {s: round(float(np.mean(subj_pcc[s])), 3) for s in subjects},
        "per_subject_rmse_pctBW": {s: round(float(np.mean(subj_rmse[s])), 1) for s in subjects},
        "per_subject_nrmse_pct": {s: round(float(np.mean(subj_nrmse[s])), 1) for s in nrmse_subjects},
        "jackknife_leave_one_out_mean_pcc": jack,
        "jackknife_range": [round(float(jv.min()), 3), round(float(jv.max()), 3)],
        "jackknife_max_swing": round(float(jv.max() - jv.min()), 3),
    }
    saved_path = guarded_dump(out, ROOT / "outputs/evaluation/crouch_jackknife.json", "eval_crouch_jackknife.py")
    print("\n=== CROUCH CP subject-clustered CI (vGRF) ===", flush=True)
    print(f"  n_subj={out['n_subj']}", flush=True)
    print(f"  PCC  subject-macro {out['PCC']['subject_macro']}  95% CI {out['PCC']['subject_clustered_ci95']}", flush=True)
    print(f"  RMSE subject-macro {out['RMSE_pctBW']['subject_macro']}%BW  95% CI {out['RMSE_pctBW']['subject_clustered_ci95']}", flush=True)
    print(f"  leave-one-subject-out PCC range {out['jackknife_range']} (max swing {out['jackknife_max_swing']})", flush=True)
    print(f"  per-subject PCC: {out['per_subject_pcc']}", flush=True)
    print("\nsaved", saved_path, flush=True)


if __name__ == "__main__":
    main()
