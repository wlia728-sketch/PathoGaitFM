"""Evaluate Uhlrich/OpenCap reference joint moments without retraining.

The manuscript scores the fixed reference window [0:64) in each 100-sample
model cycle. It is defined by build_moment_eval_54ch.py: round(101 * 0.63),
not inferred from zero crossings or predicted values. The remaining samples
are layout padding and are excluded from the manuscript metrics.

The ALLDATA checkpoint, transform, DROP_16CH inputs and three-seed, 50-step
sampler retain the published inference settings. Each target is averaged over
eligible limb-cycles within participant, then equally over participants.
Only left-stance references occur in the reported retained dataset.

Outputs include per-participant PCC, RMSE (Nm/kg), nRMSE (%), cycle counts and
input/checkpoint hashes. --include-legacy-windows additionally reports the old
full and 0–60% windows for historical replay comparison; these are not manuscript
outcomes. Frozen result paths are protected.
"""
import argparse
import os, sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from site_summary import load_site_summary, site_mass
from guarded_write import guarded_dump
from evaluation_contract import safe_result_path, sha256
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS
from drop_channels import DROP_16CH
from eval_common import load_model, safe_pcc, ZSTATS_PATH, SEED, build_known_mask, per_subject_transform
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

CKPT = ROOT / "checkpoints" / "v4_stage2_final8ch_ALLDATA" / "final.pt"
# The site's processed arrays hold participant data and may be kept outside this tree.
# PATHOGAIT_OPENCAP_PROC points at them; without it the in-tree location is used.
PROC = ROOT / "data" / "external" / "uhlrich_opencap" / "processed"
PROC = Path(os.environ.get("PATHOGAIT_OPENCAP_PROC", PROC))

# 40-ch sagittal moment channels, grouped by joint (R,L)
MOM = {"Hip": (10, 13), "Knee": (16, 17), "Ankle": (18, 19)}
KEEP = list(KEEP_CHANNELS_54TO40)
INV40to54 = {i40: c54 for i40, c54 in enumerate(KEEP)}
FULL = slice(0, 100); STANCE = slice(0, 61)
# The reference moment occupies only the stance block of the 101-point layout. The builder
# (data/external/uhlrich_opencap/build_moment_eval_54ch.py) writes it into [0:stance_pts) with
# STANCE_FRAC = 0.63, so stance_pts = round(101 * 0.63) = 64 and samples 64 to 100 are zero.
# REFERENCE is therefore the only window whose every sample carries a measured reference value.
# FULL includes 36 zero-filled swing samples, and STANCE (0-60%) drops three measured samples,
# so neither of them scores exactly the reference coverage.
STANCE_FRAC = 0.63
REFERENCE = slice(0, max(2, int(round(101 * STANCE_FRAC))))
WINDOWS = (("full", FULL), ("stance", STANCE), ("reference", REFERENCE))
WNAMES = tuple(w for w, _ in WINDOWS)

SOURCE = "ext_opencap"
COHORT_ID = 1     # healthy/normal (matches eval_opencap_zeroshot.py)
SEVERITY_ID = 5   # severity_id 5, the only bin the healthy cohort occurs with in training


def denorm(x_z, ch40, z):
    c54 = str(INV40to54[ch40])
    if c54 not in z:
        return x_z
    return x_z * (3.0 * z[c54]["std"]) + z[c54]["mean"]


ERR_CI_SEED = 20260726


def boot_ci(vals, nd=3, n_boot=5000):
    """Subject-clustered 95% CI: resample SUBJECTS with replacement. Own generator, seeded independently."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return None
    r = np.random.default_rng(ERR_CI_SEED)
    b = v[r.integers(0, v.size, (n_boot, v.size))].mean(1)
    return [round(float(np.percentile(b, 2.5)), nd), round(float(np.percentile(b, 97.5)), nd)]


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def validate_inputs(proc):
    """Validate the published retained dataset before any expensive inference."""
    if not proc.is_dir():
        raise FileNotFoundError(f"Processed Uhlrich moment directory missing: {proc}")
    files = [f for f in sorted(proc.glob("*__momeval.npy")) if not f.name.endswith("_mask.npy")]
    if not files:
        raise FileNotFoundError(f"No *__momeval.npy arrays in {proc}")
    # These counts define the manuscript evaluation, not an arbitrary uploaded-data run.
    expected_subjects, expected_windows = 10, 59
    if len(files) != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} retained subjects; found {len(files)}")
    counts = {joint: 0 for joint in MOM}
    identities, per_subject = {}, {}
    for file in files:
        mask_file = file.with_name(file.stem + "_mask.npy")
        if not mask_file.is_file():
            raise FileNotFoundError(f"Validity mask missing: {mask_file}")
        raw = np.load(file, allow_pickle=False)
        mask = np.load(mask_file, allow_pickle=False)
        if raw.ndim != 3 or raw.shape[1:] != (101, 54) or mask.shape != (len(raw), 54):
            raise ValueError(f"Invalid array/mask layout: {file.name}: {raw.shape}, {mask.shape}")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Validity mask must contain only 0/1: {mask_file.name}")
        per_subject[file.stem] = {}
        for joint, (rch, lch) in MOM.items():
            r54, l54 = INV40to54[rch], INV40to54[lch]
            if mask[:, r54].any():
                raise ValueError(f"The retained manuscript dataset has left-stance references only: {file.name}/{joint}")
            valid = mask[:, l54].astype(bool)
            if not valid.any():
                raise ValueError(f"No valid {joint} reference for {file.name}")
            if not np.isfinite(raw[valid, REFERENCE, l54]).all():
                raise ValueError(f"Non-finite reference marked valid: {file.name}/{joint}")
            # Confirm the known padding layout without using it to choose the scored window.
            if not np.equal(raw[valid, REFERENCE.stop:, l54], 0).all():
                raise ValueError(f"Unexpected values outside the fixed reference block: {file.name}/{joint}")
            counts[joint] += int(valid.sum())
            per_subject[file.stem][joint] = int(valid.sum())
        for path in (file, mask_file):
            identities[path.name] = sha256(path)
    if any(value != expected_windows for value in counts.values()):
        raise ValueError(f"Expected {expected_windows} retained windows per joint; found {counts}")
    summary_path = proc / "_summary_momeval.json"
    # Missing or non-positive body mass is an input error, not a reason to skip a subject.
    masses = {row["subj"]: site_mass(row, summary_path)
              for row in load_site_summary(summary_path) if row.get("subj")}
    for file in files:
        key = file.stem.removesuffix("__momeval")
        if key not in masses:
            raise ValueError(f"Body mass metadata missing for {key}")
        if not np.isfinite(masses[key]) or masses[key] <= 0:
            raise ValueError(f"Body mass must be finite and positive for {key}")
    identities[summary_path.name] = sha256(summary_path)
    return files, {"expected_subjects": expected_subjects, "expected_windows_per_joint": expected_windows,
        "reference_samples": {"start": 0, "stop_exclusive": REFERENCE.stop},
        "counts_by_joint": counts, "counts_by_subject": per_subject, "input_sha256": identities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROC)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/opencap_moments.json")
    parser.add_argument("--include-legacy-windows", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Check all processed inputs and reference counts without loading the model")
    args = parser.parse_args()
    output_path = safe_result_path(args.out, ROOT)
    proc, checkpoint = args.processed.resolve(), args.checkpoint.resolve()
    windows = WINDOWS if args.include_legacy_windows else (("reference", REFERENCE),)
    wnames = tuple(name for name, _ in windows)
    files, input_report = validate_inputs(proc)
    if args.validate_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(input_report, indent=2, allow_nan=False), encoding="utf-8")
        print("Validated reference inputs:", output_path)
        return
    device = torch.device(args.device)
    print("OpenCap external moment zero-shot | device", device, flush=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    z = json.load(open(ZSTATS_PATH))
    model, ckpt, _, imt = load_model(checkpoint, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None:
            _i = ckpt.get("input_mask_token")
        if _i is not None:
            imt = _i
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

    meta = V4SubjectMetadata()
    _sp = proc / "_summary_momeval.json"
    for d in load_site_summary(_sp):
        if d.get("subj"):
            _m = site_mass(d, _sp)
            meta.bw_table[(SOURCE, d["subj"])] = _m
            meta.bw_table[(SOURCE, d["subj"] + "__momeval")] = _m
    transform = per_subject_transform(meta, require_tables=False)


    # per (joint, window): per-subject-mean PCC list, pooled rmse (denorm Nm/kg), n
    acc = {j: {w: {"subjpcc": [], "rmse": [], "nrmse": [], "n_cyc": 0} for w in wnames} for j in MOM}
    n_subj = 0
    persubj_detail = {}
    for f in files:
        stem = f.stem
        raw = np.load(str(f), allow_pickle=False)
        a40 = transform(raw.copy(), SOURCE, "cohort", stem)[:, :100, :].astype(np.float32)
        m = np.load(f.with_name(stem + "_mask.npy"), allow_pickle=False)
        m40 = m[:, KEEP_CHANNELS_54TO40].astype(bool)
        n = a40.shape[0]
        if m40.shape != (n, 40):
            raise ValueError(f"Transformed array/mask mismatch: {stem}")
        anymom = np.zeros(n, bool)
        for (rch, lch) in MOM.values():
            anymom |= (m40[:, rch] | m40[:, lch])
        keep = np.where(anymom)[0]
        if not len(keep):
            print(f"  [{stem}] no valid moment cycles"); continue
        gt = torch.from_numpy(np.nan_to_num(a40[keep].transpose(0, 2, 1), nan=0.0)).to(device)
        valid = torch.from_numpy(m40[keep]).to(device)
        nb = len(keep)
        sid_t = torch.full((nb,), model.num_sources, dtype=torch.long, device=device)  # reserved unknown-external source id
        cid_t = torch.full((nb,), COHORT_ID, dtype=torch.long, device=device)
        svid_t = torch.full((nb,), SEVERITY_ID, dtype=torch.long, device=device)
        gens = []
        for so in range(3):
            torch.manual_seed(SEED + so * 7919)
            with torch.no_grad():
                gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid_t, cid_t, svid_t,
                                                 50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
        g = torch.stack(gens).mean(0)
        km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
        gtn = gt.cpu().numpy(); prn = g.cpu().numpy(); vm = m40[keep]
        if not np.isfinite(prn).all():
            raise RuntimeError(f"Non-finite prediction: {stem}")

        subj_has = False
        pd = persubj_detail.setdefault(stem, {})
        for j, (rch, lch) in MOM.items():
            for wname, win in windows:
                cyc_pccs = []; cyc_rmse = []; cyc_nrmse = []; cycle_scores = []
                for ci in range(nb):
                    for ch in (rch, lch):
                        if not vm[ci, ch]:
                            continue
                        gw_z = gtn[ci, ch, win]; pw_z = prn[ci, ch, win]
                        r = safe_pcc(gw_z, pw_z)
                        # The error must not be accumulated over cycles and averaged flat, because
                        # the correlation was averaged within subject first. Both are now collected on the
                        # same scored cycles and aggregated subject-first, so one row shares one rule.
                        if np.isfinite(r):
                            cyc_pccs.append(r)
                            gw = denorm(gw_z, ch, z); pw = denorm(pw_z, ch, z)
                            rm = rmse(gw, pw)
                            cyc_rmse.append(rm)
                            gr = float(np.nanmax(gw) - np.nanmin(gw))
                            nr = 100.0 * rm / gr if gr > 1e-6 else None
                            if nr is not None:
                                cyc_nrmse.append(nr)
                            cycle_scores.append({"cycle_index": int(keep[ci]), "side": "R" if ch == rch else "L",
                                "pcc": float(r), "RMSE_Nm_per_kg": rm, "nRMSE_pct": nr})
                            acc[j][wname]["n_cyc"] += 1
                if cyc_pccs:
                    acc[j][wname]["subjpcc"].append(float(np.mean(cyc_pccs)))
                    if cyc_rmse: acc[j][wname]["rmse"].append(float(np.mean(cyc_rmse)))
                    if cyc_nrmse: acc[j][wname]["nrmse"].append(float(np.mean(cyc_nrmse)))
                    pd[f"{j}_{wname}"] = round(float(np.mean(cyc_pccs)), 3)
                    pd[f"{j}_{wname}_RMSE"] = round(float(np.mean(cyc_rmse)), 4) if cyc_rmse else None
                    pd[f"{j}_{wname}_nRMSE_pct"] = round(float(np.mean(cyc_nrmse)), 1) if cyc_nrmse else None
                    pd[f"{j}_{wname}_n_cyc"] = len(cycle_scores)
                    pd[f"{j}_{wname}_cycle_scores"] = cycle_scores
                    subj_has = True
        if subj_has:
            n_subj += 1

    out = {"site": "opencap", "cohort": "healthy_adult", "n_subj": n_subj,
           "units_RMSE": "Nm/kg (denormalized via v4 zstats x3sigma+mu)",
           "note": "release-provided OpenSim ID moments, quality-screened here, not recomputed (L stance only); GRF+moments MASKED, "
                   "predicted from visible kinematics; 3-seed tweedie_inpaint_drop; DROP_16CH input; "
                   "same pipeline/checkpoint as published crouch/opencap vGRF eval",
           "per_joint": {}}
    for j in MOM:
        out["per_joint"][j] = {}
        for w in wnames:
            d = acc[j][w]
            pcc = float(np.mean(d["subjpcc"])) if d["subjpcc"] else None
            rm = float(np.mean(d["rmse"])) if d["rmse"] else None
            nrmse = float(np.mean(d["nrmse"])) if d["nrmse"] else None
            out["per_joint"][j][w] = {
                "PCC_subject_macro": round(pcc, 3) if pcc is not None else None,
                "PCC_ci95": boot_ci(d["subjpcc"], nd=3),
                "RMSE_Nm_per_kg": round(rm, 4) if rm is not None else None,
                "RMSE_ci95": boot_ci(d["rmse"], nd=4),
                "nRMSE_pct": round(nrmse, 1) if nrmse is not None else None,
                "nRMSE_pct_ci95": boot_ci(d["nrmse"], nd=1),
                "n_subj_scored": len(d["subjpcc"]),
                "n_cyc": d["n_cyc"],
            }
    out["per_subject"] = persubj_detail
    out["evaluation_windows"] = {name: {"start": win.start, "stop_exclusive": win.stop,
        "manuscript_outcome": name == "reference"} for name, win in windows}
    out["reference_window_definition"] = "Fixed builder layout: round(101 * 0.63) = 64; never inferred from nonzero samples"
    out["input_validation"] = input_report
    out["provenance"] = {"checkpoint_sha256": sha256(checkpoint), "zstats_sha256": sha256(ZSTATS_PATH),
        "checkpoint_name": checkpoint.parent.name, "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "sampling_seeds": [SEED + i * 7919 for i in range(3)],
        "sampling_steps": 50, "guidance_scale": 1.0, "saved_steps": list(CHECKPOINT_STEPS),
        "drop_channels": list(DROP_16CH), "producer_sha256": sha256(Path(__file__))}
    if n_subj != input_report["expected_subjects"]:
        raise RuntimeError(f"Scored {n_subj} subjects, expected {input_report['expected_subjects']}")
    for joint in MOM:
        reference = out["per_joint"][joint]["reference"]
        if reference["n_cyc"] != input_report["expected_windows_per_joint"] or reference["n_subj_scored"] != n_subj:
            raise RuntimeError(f"Incomplete scored reference coverage: {joint}")
    guarded_dump(out, output_path, "eval_opencap_moments.py")

    print(f"\n=== OpenCap external moment zero-shot (n_subj={n_subj}) ===", flush=True)
    print("  joint  window | PCC(subj-macro) n | RMSE(Nm/kg) | nRMSE%", flush=True)
    for j in MOM:
        for w in wnames:
            m = out["per_joint"][j][w]
            print(f"  {j:6s} {w:6s} | {str(m['PCC_subject_macro']):>6s} "
                  f"(n={m['n_subj_scored']:>2d}) | {str(m['RMSE_Nm_per_kg']):>7s} | "
                  f"{str(m['nRMSE_pct']):>5s}", flush=True)
    print("\nsaved", output_path, flush=True)


if __name__ == "__main__":
    main()
