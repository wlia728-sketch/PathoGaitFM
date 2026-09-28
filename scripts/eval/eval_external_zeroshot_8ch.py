"""External-site ZERO-SHOT validation of the ALL-DATA final-8ch model.
Two independent external datasets, never used in training:
  - PD-Nordic (data/external/pd_nordic/processed): PD patients -> cohort_id=3
  - Fukuchi healthy (data/external/fukuchi_healthy/processed): healthy adults -> cohort_id=1 (normal)
For each: zero-shot predict the 8 target channels (sagittal hip/knee/ankle moments + vGRF) from
16ch kinematic input, TSA inference, Method B per-limb filter, subject-level macro PCC.
PD moments masked (PD scores vGRF only). Model = checkpoints/v4_stage2_final8ch_ALLDATA/final.pt.
ASCII prints only; json dumped before prints. Writes outputs/evaluation/external_zeroshot_8ch.json.
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
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
MOM8 = {10: "Hip", 13: "Hip", 16: "Knee", 17: "Knee", 18: "Ankle", 19: "Ankle"}
NAMES = {10: "R_Hip", 13: "L_Hip", 16: "R_Knee", 17: "L_Knee", 18: "R_Ankle", 19: "L_Ankle", 22: "R_vGRF", 25: "L_vGRF"}

EXTERNAL = {
    "PD_Nordic": {"dir": ROOT / "data/external/pd_nordic/processed", "cohort_id": 3, "severity_id": 1, "pd": True},
    "Fukuchi_healthy": {"dir": ROOT / "data/external/fukuchi_healthy/processed", "cohort_id": 1, "severity_id": 5, "pd": False},
}


def load_npy_54(f, transform, sname):
    """CRITICAL: external data must pass through the SAME transform/zstats normalization the model
    was trained with (raw npy is physical units; model expects normalized): the transform is
    called as tf(raw, 'ext_pd'/'ext_healthy', 'cohort', stem)."""
    raw = np.load(str(f))  # (n,101,54) physical units, harmonized
    stem = f.stem
    a40 = transform(raw.copy(), sname, "cohort", stem)[:, :100, :].astype(np.float32)  # normalized 40ch
    mp = str(f).replace(".npy", "_mask.npy")
    m = np.load(mp)
    m40 = m[:, KEEP_CHANNELS_54TO40].astype(bool) if m.shape[-1] >= 54 else m.astype(bool)
    n = min(a40.shape[0], m40.shape[0])
    return a40[:n], m40[:n]


def gen_tsa(model, sched, gt, valid, sid, cid, svid, imt, device):
    gens = []
    for so in range(3):
        torch.manual_seed(SEED + so * 7919)
        with torch.no_grad():
            gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                             50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
    g = torch.stack(gens).mean(0)
    km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
    return g.cpu().numpy()


_ZS = json.load(open(ZSTATS_PATH))
ERR_CI_SEED = 20260726


def denorm(x_z, ch40):
    """z-scored -> physical units, x_phys = x_z * (3 * std) + mean, from the frozen training statistics."""
    c54 = str(KEEP_CHANNELS_54TO40[ch40])
    if c54 not in _ZS:
        return None
    return x_z * (3.0 * _ZS[c54]["std"]) + _ZS[c54]["mean"]


def rmse_nrmse_phys(gt_z, pred_z, ch40):
    """RMSE in physical units (%BW for vertical GRF, Nm/kg for moments) and nRMSE in per cent."""
    g = denorm(np.asarray(gt_z, float), ch40); p = denorm(np.asarray(pred_z, float), ch40)
    if g is None:
        return None, None
    rm = float(np.sqrt(np.mean((g - p) ** 2)))
    rng_gt = float(np.max(g) - np.min(g))
    scale = 100.0 if ch40 in (22, 25) else 1.0
    return rm * scale, (100.0 * rm / rng_gt if rng_gt > 1e-9 else None)


def boot_ci(vals, nd=3, n_boot=5000):
    """Subject-clustered 95% CI: resample SUBJECTS with replacement. Own generator, seeded independently."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return None
    r = np.random.default_rng(ERR_CI_SEED)
    b = v[r.integers(0, v.size, (n_boot, v.size))].mean(1)
    return [round(float(np.percentile(b, 2.5)), nd), round(float(np.percentile(b, 97.5)), nd)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--site", choices=list(EXTERNAL), action="append")
    parser.add_argument("--processed", type=Path, help="processed directory, requires exactly one --site")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/external_zeroshot_8ch.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    selected = {name: dict(EXTERNAL[name]) for name in (args.site or EXTERNAL)}
    if args.processed:
        if len(selected) != 1:
            parser.error("--processed requires exactly one --site")
        next(iter(selected.values()))["dir"] = args.processed
    for cfg in selected.values():
        directory = cfg["dir"]
        if not directory.is_dir() or not any(not f.name.endswith("_mask.npy") for f in directory.glob("*.npy")):
            raise FileNotFoundError(f"No processed cycles in {directory}; see data/external/ and --processed")
        if not (directory / "_harmonize_summary.json").is_file():
            raise FileNotFoundError(f"Missing body-mass metadata: {directory / '_harmonize_summary.json'}")
    device = torch.device(args.device)
    print("External zero-shot (ALL-DATA 8ch model) | device", device, flush=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"MODEL MISSING: {args.checkpoint}")
    model, ckpt, _, imt = load_model(args.checkpoint, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None: _i = ckpt.get("input_mask_token")
        if _i is not None: imt = _i
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)
    meta = V4SubjectMetadata()
    # inject external subjects' body weight (needed for vGRF BW-normalization in transform).
    # source_name used in load: 'ext_pd' / 'ext_healthy'; mass from each set's _harmonize_summary.json
    for name, cfg in selected.items():
        sname = "ext_pd" if cfg["pd"] else "ext_healthy"
        hsum = cfg["dir"] / "_harmonize_summary.json"
        if hsum.exists():
            entries = load_site_summary(hsum)
            for d in entries:
                sid = d.get("sid") or d.get("subject")
                if sid:
                    meta.bw_table[(sname, sid)] = site_mass(d, hsum)
    transform = per_subject_transform(meta, require_tables=False)
    # NOTE: external data is harmonized to the same 54ch->40ch convention; we trust their pipeline.
    # We do NOT re-apply the per-subject flip transform (external already harmonized).
    out = {}
    for name, cfg in selected.items():
        d = cfg["dir"]
        files = [f for f in sorted(d.glob("*.npy")) if not f.name.endswith("_mask.npy")]
        if not files:
            out[name] = {"note": "no npy files"}; continue
        is_pd = cfg["pd"]
        R = [22] + ([] if is_pd else [10, 16, 18]); L = [25] + ([] if is_pd else [13, 17, 19])
        subj = defaultdict(lambda: defaultdict(list))  # chan_name -> stem -> [pcc]
        # Admission bookkeeping. A limb-cycle slot is one cycle on one side, so a site with
        # n cycles offers 2n slots; a slot is admitted when its vertical GRF and every scored
        # channel on that side are valid. Recorded so the admitted count has a source.
        slots_total = 0
        slots_admitted = 0
        # Error accumulators keyed identically and filled inside the same validity guard, so the RMSE
        # cells are the same scored limb-cycles as the PCC cells and share one aggregation rule.
        subj_rm = defaultdict(lambda: defaultdict(list))
        subj_nr = defaultdict(lambda: defaultdict(list))
        for f in files:
            stem = f.stem
            try:
                x40, m40 = load_npy_54(f, transform, "ext_pd" if is_pd else "ext_healthy")
                n = min(x40.shape[0], m40.shape[0]); x40, m40 = x40[:n], m40[:n]
            except Exception as exc:
                raise RuntimeError(f"Cannot load {f}; refusing a partial evaluation") from exc
            slots_total += 2 * x40.shape[0]
            for ci in range(x40.shape[0]):
                for sc, vg in [(R, 22), (L, 25)]:
                    if m40[ci, vg] and all(m40[ci, c] for c in sc):
                        slots_admitted += 1
            keep = [ci for ci in range(x40.shape[0])
                    if (m40[ci, 22] and all(m40[ci, c] for c in R)) or (m40[ci, 25] and all(m40[ci, c] for c in L))]
            if not keep:
                continue
            keep = np.array(keep)
            for b0 in range(0, len(keep), 32):
                bidx = keep[b0:b0 + 32]; nb = len(bidx)
                gt = torch.from_numpy(np.nan_to_num(x40[bidx].transpose(0, 2, 1), nan=0.0)).to(device)
                valid = torch.from_numpy(m40[bidx]).to(device)
                sid = torch.full((nb,), model.num_sources, dtype=torch.long, device=device)  # reserved unknown-external source id  # source unknown -> 0
                cid = torch.full((nb,), cfg["cohort_id"], dtype=torch.long, device=device)
                svid = torch.full((nb,), cfg["severity_id"], dtype=torch.long, device=device)
                g = gen_tsa(model, sched, gt, valid, sid, cid, svid, imt, device)
                gtn = gt.cpu().numpy(); vm = m40[bidx]
                for ci in range(nb):
                    for sc, vg in [(R, 22), (L, 25)]:
                        if not vm[ci, vg] or not all(vm[ci, c] for c in sc):
                            continue
                        for ch in sc:
                            r = safe_pcc(gtn[ci, ch, :], g[ci, ch, :])
                            if np.isfinite(r):
                                subj[NAMES[ch]][stem].append(r)
                                rm, nrm = rmse_nrmse_phys(gtn[ci, ch, :], g[ci, ch, :], ch)
                                if rm is not None and np.isfinite(rm): subj_rm[NAMES[ch]][stem].append(rm)
                                if nrm is not None and np.isfinite(nrm): subj_nr[NAMES[ch]][stem].append(nrm)
        # subject-macro per channel, error aggregated the same way with a subject-clustered CI
        per = {}
        for ch, sd in subj.items():
            ps = [float(np.mean(v)) for v in sd.values() if v]
            if not ps:
                continue
            cell = {"pcc": round(float(np.mean(ps)), 3), "pcc_ci95": boot_ci(ps, nd=3), "n_subj": len(ps)}
            rs = [float(np.mean(v)) for v in subj_rm.get(ch, {}).values() if v]
            ns = [float(np.mean(v)) for v in subj_nr.get(ch, {}).values() if v]
            if rs:
                cell["RMSE"] = round(float(np.mean(rs)), 2); cell["RMSE_ci95"] = boot_ci(rs, nd=2)
                cell["unit_RMSE"] = "%BW" if ch.endswith("vGRF") else "Nm/kg"
            if ns:
                cell["nRMSE_pct"] = round(float(np.mean(ns)), 1); cell["nRMSE_pct_ci95"] = boot_ci(ns, nd=1)
            per[ch] = cell

        # limb-merged, the aggregation the two cerebral-palsy sites already report. A subject's
        # limb-cycles from both sides are pooled first, then averaged within subject, then over
        # subjects, so a subject scored on one side only still counts once. This is NOT the mean of
        # the two per-limb numbers, which would weight unequal subject sets equally.
        pairs = {}
        for ch in subj:
            if ch[:2] in ("R_", "L_"):
                pairs.setdefault(ch[2:], {})[ch[0]] = ch
        merged = {}
        for base, sides in sorted(pairs.items()):
            if set(sides) != {"R", "L"}:
                continue
            def pool(store):
                out_stems = defaultdict(list)
                for channel in sides.values():
                    for stem, vals in store.get(channel, {}).items():
                        out_stems[stem].extend(vals)
                return [float(np.mean(v)) for v in out_stems.values() if v]
            ps = pool(subj)
            if not ps:
                continue
            cell = {"pcc": round(float(np.mean(ps)), 3), "pcc_ci95": boot_ci(ps, nd=3),
                    "n_subj": len(ps)}
            # dispersion across subjects, the quantity the two cerebral-palsy sites already
            # report, so that every external site in Figure 3 carries the same statistic
            if len(ps) > 1:
                cell["sd_across_subjects"] = round(float(np.std(ps, ddof=1)), 4)
            rs, ns = pool(subj_rm), pool(subj_nr)
            if rs:
                cell["RMSE"] = round(float(np.mean(rs)), 2); cell["RMSE_ci95"] = boot_ci(rs, nd=2)
                cell["unit_RMSE"] = "%BW" if base.endswith("vGRF") else "Nm/kg"
            if ns:
                cell["nRMSE_pct"] = round(float(np.mean(ns)), 1); cell["nRMSE_pct_ci95"] = boot_ci(ns, nd=1)
            merged[base] = cell

        out[name] = {"cohort_id": cfg["cohort_id"],
                     "limb_cycle_slots_total": slots_total,
                     "limb_cycle_slots_admitted": slots_admitted,
                     "per_channel": per,
                     "per_channel_limb_merged": merged}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guarded_dump(out, args.out, "eval_external_zeroshot_8ch.py")
    print("\n=== EXTERNAL ZERO-SHOT (all-data 8ch model) ===", flush=True)
    for name, r in out.items():
        print(f"\n[{name}]", flush=True)
        for ch, m in r.get("per_channel", {}).items():
            print(f"  {ch:10s} PCC={m['pcc']} (n_subj={m['n_subj']})", flush=True)
    print("\nsaved", args.out, flush=True)


if __name__ == "__main__":
    main()
