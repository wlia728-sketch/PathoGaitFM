"""FULL final-8ch evaluation â€” ALL-VALID variant (fixes sample-8-then-filter bias).
Instead of randomly picking 8 raw cycles then filtering, this evaluates EVERY valid
limb-cycle of every subject (pre-filter then score). No subject with valid data drops out;
no valid cycle is wasted. TSA inference is run in batches (<= BATCH cycles) to bound memory.

Same protocol otherwise: TSA (within-traj trimmed-mean [5,10,15,25,35,45] + 3-seed), DDIM 50,
CFG 1.0, Method B per-limb validity, R/L merged, 8ch target
(sagittal hip/knee/ankle moments + vGRF). grffix, quality-control exclusions applied, PD moments masked.
pretrain CV5 reuses expA. Writes complete selected variants to a separate --out path; archived results are protected.
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
from guarded_write import guarded_dump
from evaluation_contract import preflight_cv5, validate_coverage, safe_result_path
from patient_statistics import cluster_means, interval
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS, COH, EXCLUDE, GRFFIX, load_split
from drop_channels import DROP_16CH
from eval_common import (
    load_model, safe_pcc, get_val_pairs, load_valid_mask_40, source_type_and_name,
    ZSTATS_PATH, SOURCE_TO_ID, SEED, SEVERITY_JSON, build_known_mask, load_local_json, per_subject_transform,
)
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

MERGED8 = {10: "Hip_Mom", 13: "Hip_Mom", 16: "Knee_Mom", 17: "Knee_Mom",
           18: "Ankle_Mom", 19: "Ankle_Mom", 22: "GRF_V", 25: "GRF_V"}
CHANS_M = ["Hip_Mom", "Knee_Mom", "Ankle_Mom", "GRF_V"]
CV5 = [f"cv5_{i}" for i in range(5)]
COH_ORDER = ["normal", "cp", "vdk_stroke", "bmclab_pd"]
BATCH = 32  # cycles per inference batch (bound memory)
CHECKPOINT_ROOT = ROOT / "checkpoints"
SPLIT = None  # the five-fold split, loaded in main()
REFERENCE_MODE = "clipped"

# --- physical-unit error reporting ---------------------------------------------------------------
# The inverse restores the physical units of the transformed signal. The transform
# clips at +/-4.5 standard deviations, so clipped reference values do not recover
# their original amplitudes. Reported errors use this harmonised, clipped reference.
# Units: vertical GRF is body weight, reported as %BW; sagittal joint moments are Nm/kg.
_ZS = json.load(open(ZSTATS_PATH))
from pathogait.data.channels import KEEP_CHANNELS_54TO40  # noqa: E402
UNIT_SCALE = {"GRF_V": 100.0, "Hip_Mom": 1.0, "Knee_Mom": 1.0, "Ankle_Mom": 1.0}
UNIT_NAME = {"GRF_V": "%BW", "Hip_Mom": "Nm/kg", "Knee_Mom": "Nm/kg", "Ankle_Mom": "Nm/kg"}
N_BOOT_CI, CI_SEED = 5000, 20260726


def denorm(x_z, ch40):
    c54 = str(KEEP_CHANNELS_54TO40[ch40])
    if c54 not in _ZS:
        return None
    return x_z * (3.0 * _ZS[c54]["std"]) + _ZS[c54]["mean"]


def rmse_nrmse(gt_z, pred_z, ch40):
    """RMSE in physical units and range-normalised nRMSE for one limb-cycle. nRMSE is invariant to the
    linear de-normalisation, so it is directly comparable with the z-space definition used elsewhere."""
    g = denorm(np.asarray(gt_z, float), ch40); p = denorm(np.asarray(pred_z, float), ch40)
    if g is None:
        return None, None
    rm = float(np.sqrt(np.mean((g - p) ** 2)))
    rng = float(np.max(g) - np.min(g))
    return rm, (rm / rng if rng > 1e-9 else None)


def boot_ci(values_by_unit, cohort, seed=CI_SEED, n_boot=N_BOOT_CI, nd=3):
    """Keep PD medication-state records together in patient bootstrap draws."""
    return interval(cluster_means(cohort, values_by_unit, np.random.default_rng(seed), n_boot), nd)


def cohort_subject_means(scores):
    """Mean channels per unit, then units per cohort; round only for display."""
    return {cohort: float(np.mean([np.mean(channels) for channels in units.values()]))
            for cohort, units in scores.items()}


def cohort_error_means(scores):
    """Preserve the error summary's equal channel weights without rounding targets first."""
    channels = defaultdict(list)
    for (cohort, _channel), units in scores.items():
        unit_means = [float(np.mean(cycles)) for cycles in units.values() if cycles]
        if unit_means:
            channels[cohort].append(float(np.mean(unit_means)))
    return {cohort: float(np.mean(values)) for cohort, values in channels.items()}


def prefix_for(variant, fold):
    if variant == "pretrain" and fold.startswith("cv5_"):
        return f"v4_stage2_expA_dynpool_{fold}"
    return f"v4_stage2_final8ch_{variant}_{fold}"


def eval_fold(prefix, fold, transform, sev, sched, device, persubj, subj_scores,
              subj_rmse=None, subj_nrmse=None):
    cp = CHECKPOINT_ROOT / f"{prefix}" / "final.pt"
    if not cp.exists():
        raise FileNotFoundError(cp)
    model, ckpt, _, imt = load_model(cp, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None: _i = ckpt.get("input_mask_token")
        if _i is not None: imt = _i
    for src, stem in get_val_pairs(SPLIT, fold):
        if stem in EXCLUDE: continue
        rp = GRFFIX / src / f"{stem}.npy"
        if not rp.exists(): raise FileNotFoundError(rp)
        s = sev["mappings"].get(src, {}).get(stem)
        if s is None: raise ValueError(f"Missing severity: {src}/{stem}")
        try:
            raw = np.load(str(rp)); v40 = load_valid_mask_40(rp, raw.shape[0])
            st, sn = source_type_and_name(src)
            x40 = transform(raw, sn, st, stem)[:, :100, :].astype(np.float32)
            if v40.shape[0] != x40.shape[0]:
                raise ValueError("mask and waveform cycle counts differ")
        except Exception as e:
            raise RuntimeError(f"Failed loading/transforming {src}/{stem}") from e
        cohort = COH.get(int(s["cohort_id"]), "other")
        R_SIDE = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18])
        L_SIDE = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
        # PRE-FILTER: keep every cycle that has >=1 valid limb-side (pre-filter then score)
        keep = []
        for ci in range(x40.shape[0]):
            r_ok = v40[ci, 22] and all(v40[ci, c] for c in R_SIDE)
            l_ok = v40[ci, 25] and all(v40[ci, c] for c in L_SIDE)
            if r_ok or l_ok:
                keep.append(ci)
        if not keep:
            continue
        keep = np.array(keep)
        sid = torch.full((1,), SOURCE_TO_ID.get(src, 0), dtype=torch.long)
        cid = torch.full((1,), int(s["cohort_id"]), dtype=torch.long)
        svid = torch.full((1,), int(s["severity_id"]), dtype=torch.long)
        used = False
        # batch over ALL kept cycles
        for b0 in range(0, len(keep), BATCH):
            bidx = keep[b0:b0 + BATCH]
            nb = len(bidx)
            gt = torch.from_numpy(np.nan_to_num(x40[bidx].transpose(0, 2, 1), nan=0.0)).to(device)
            valid = torch.from_numpy(v40[bidx]).to(device)
            bsid = sid.expand(nb).to(device); bcid = cid.expand(nb).to(device); bsvid = svid.expand(nb).to(device)
            gens = []
            for so in range(3):
                torch.manual_seed(SEED + so * 7919)
                with torch.no_grad():
                    gens.append(tweedie_inpaint_drop(model, sched, gt, valid, bsid, bcid, bsvid,
                                                     50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
            gen = torch.stack(gens).mean(0)
            km = build_known_mask(gt, valid); gen = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * gen
            gtn = gt.cpu().numpy(); genn = gen.cpu().numpy(); vm = v40[bidx]
            for ci in range(gtn.shape[0]):
                for side_chs, vgrf_ch in [(R_SIDE, 22), (L_SIDE, 25)]:
                    if not vm[ci, vgrf_ch]: continue
                    if not all(vm[ci, c] for c in side_chs): continue
                    for ch in side_chs:
                        r = safe_pcc(gtn[ci, ch, :], genn[ci, ch, :])
                        if np.isfinite(r):
                            # per-subject: accumulate this limb-cycle PCC under (cohort, merged-channel, subject)
                            subj_scores[(cohort, MERGED8[ch])].setdefault(stem, []).append(r)
                            # Physical-unit error on the SAME scored limb-cycle, inside the same guards, so
                            # the error cells are identical to the PCC cells by construction.
                            if subj_rmse is not None:
                                rm, nrm = rmse_nrmse(gtn[ci, ch, :], genn[ci, ch, :], ch)
                                mc = MERGED8[ch]
                                if rm is not None and np.isfinite(rm):
                                    subj_rmse[(cohort, mc)].setdefault(stem, []).append(rm * UNIT_SCALE[mc])
                                if nrm is not None and np.isfinite(nrm):
                                    subj_nrmse[(cohort, mc)].setdefault(stem, []).append(nrm * 100.0)
                            used = True
        if used: persubj[cohort].add(stem)
    return True


def main(argv=None):
    global CHECKPOINT_ROOT, GRFFIX, SPLIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=["pretrain", "scratch"], default=["pretrain", "scratch"])
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=GRFFIX)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/kinetics_cv5.json")
    args = parser.parse_args(argv)
    CHECKPOINT_ROOT, GRFFIX = args.checkpoint_root, args.raw_root
    out_path = safe_result_path(args.out, ROOT)
    SPLIT = load_split()
    sev = load_local_json(SEVERITY_JSON, "severity table")
    checkpoints = [CHECKPOINT_ROOT / prefix_for(v, f) / "final.pt" for v in args.variants for f in CV5]
    expected, coverage = preflight_cv5(SPLIT, GRFFIX, sev, checkpoints, EXCLUDE)
    out = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ALL-VALID final-8ch eval | TSA + Method B (pre-filter then score) | 8ch | device {device}")
    meta = V4SubjectMetadata()
    transform = per_subject_transform(meta, require_tables=True)
    for source, stem in set.union(*expected.values()):
        meta.get_bw(source, stem)  # Missing anthropometry fails before model inference.
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)
    for variant in args.variants:
        for setting, folds in [("CV5", CV5)]:
            persubj = defaultdict(set); done = []
            subj_scores = defaultdict(dict)  # (cohort, merged_ch) -> {subject: [limb-cycle PCCs]}
            subj_rmse = defaultdict(dict)    # same keys -> {subject: [limb-cycle RMSE, physical units]}
            subj_nrmse = defaultdict(dict)   # same keys -> {subject: [limb-cycle nRMSE, per cent]}
            for f in folds:
                if eval_fold(prefix_for(variant, f), f, transform, sev, sched, device, persubj,
                             subj_scores, subj_rmse, subj_nrmse):
                    done.append(f)
            if done != CV5:
                raise RuntimeError(f"Incomplete CV5 evaluation: {done}")
            validate_coverage(set.union(*expected.values()),
                              {(c, st) for c, stems in persubj.items() for st in stems})

            nsub = {c: len(persubj[c]) for c in persubj}
            # --- subject-level macro ---
            # step1: per-subject per-channel = mean of that subject's limb-cycle PCCs
            # step2: cohort-channel macro = mean over subjects (equal subject weight)
            # step3a: per-cohort subject-score = for each subject, mean over ITS scored channels;
            #         cohort macro = mean over subjects. step3b: headline = mean over 4 cohorts.
            cohort_channel_macro = defaultdict(dict)  # cohort -> ch -> {pcc, n_subj}
            subj_overall = defaultdict(dict)          # cohort -> subject -> mean-over-its-channels
            for (coh, ch), sdict in subj_scores.items():
                per_subj_ch = {st: float(np.mean(v)) for st, v in sdict.items()}
                cohort_channel_macro[coh][ch] = {"pcc": round(float(np.mean(list(per_subj_ch.values()))), 3),
                                                 "n_subj": len(per_subj_ch)}
                for st, val in per_subj_ch.items():
                    subj_overall[coh].setdefault(st, []).append(val)
            cohort_subject_macro = {}  # cohort -> {pcc, n_subj} (subject = mean of its channels)
            cohort_raw_means = cohort_subject_means(subj_overall)
            for coh, sd in subj_overall.items():
                cohort_subject_macro[coh] = {"pcc": round(cohort_raw_means[coh], 3), "n_subj": len(sd)}
            headline_subject_macro = round(float(np.mean(list(cohort_raw_means.values()))), 3)
            key = f"{variant}/{setting}"
            # per-subject dump (for bootstrap CIs + paired tests; Move 1). subject overall =
            # mean over its scored channels; also keep per-channel per-subject.
            per_subject_overall = {c: {st: round(float(np.mean(chs)), 6) for st, chs in sd.items()}
                                   for c, sd in subj_overall.items()}
            per_subject_channel = defaultdict(lambda: defaultdict(dict))
            for (coh, ch), sdict in subj_scores.items():
                for st, v in sdict.items():
                    per_subject_channel[coh][st][ch] = round(float(np.mean(v)), 6)
            # --- physical-unit error, aggregated exactly like the PCC macro (per-subject mean, then mean
            #     over subjects) with a subject-clustered bootstrap 95% CI on the same per-subject values.
            error_macro = defaultdict(dict)
            for (coh, ch), sdict in subj_scores.items():
                cell = {"unit_RMSE": UNIT_NAME[ch], "n_subj": len(sdict)}
                for label, store, nd in (("RMSE", subj_rmse, 2), ("nRMSE_pct", subj_nrmse, 1)):
                    per_subj = {st: float(np.mean(v)) for st, v in store.get((coh, ch), {}).items() if v}
                    if per_subj:
                        cell[label] = round(float(np.mean(list(per_subj.values()))), nd)
                        cell[label + "_CI95"] = boot_ci(per_subj, coh, nd=nd)
                error_macro[coh][ch] = cell
            pooled_nrmse = {coh: round(value, 1) for coh, value in cohort_error_means(subj_nrmse).items()}

            # per_cohort_merged / per_cohort_sep (tsa_inference.summarize, a mean over limb CYCLES)
            # were REMOVED on 2026-07-26. They were never the reported quantity, but two consumers
            # read them by mistake and silently compared cycle-pooled against subject-macro values.
            # cohort_channel_macro below is the subject-macro form every reported number uses.
            out[key] = {"folds": done, "n_subj": nsub, "coverage": coverage,
                        "reference_mode": REFERENCE_MODE,
                        "cohort_channel_macro": {c: cohort_channel_macro[c] for c in cohort_channel_macro},
                        "cohort_subject_macro": cohort_subject_macro,
                        "headline_subject_macro": headline_subject_macro,
                        "cohort_channel_error": {c: error_macro[c] for c in error_macro},
                        "nRMSE_pct_mean_over_channels": pooled_nrmse,
                        "per_subject_overall": per_subject_overall,
                        "per_subject_channel": {c: dict(per_subject_channel[c]) for c in per_subject_channel},
                        "per_subject_RMSE": {c: {} for c in per_subject_overall},
                        "per_subject_nRMSE_pct": {c: {} for c in per_subject_overall}}
            for (coh, ch), sdict in subj_rmse.items():
                for st, v in sdict.items():
                    out[key]["per_subject_RMSE"].setdefault(coh, {}).setdefault(st, {})[ch] = round(float(np.mean(v)), 4)
            for (coh, ch), sdict in subj_nrmse.items():
                for st, v in sdict.items():
                    out[key]["per_subject_nRMSE_pct"].setdefault(coh, {}).setdefault(st, {})[ch] = round(float(np.mean(v)), 3)
            print(f"   >>> headline (subject-level macro, 4-cohort equal weight) = {headline_subject_macro}")
            print(f"\n[{key}] folds={len(done)} (R/L MERGED, ALL-VALID)")
            print(f"   {'cohort':12s}" + "".join(f"{m:>11s}" for m in CHANS_M) + "   nsubj")
            for c in COH_ORDER:
                if c in cohort_channel_macro:
                    row = f"   {c:12s}"
                    for m in CHANS_M:
                        cell = cohort_channel_macro[c].get(m)
                        row += f"{(str(cell['pcc']) if cell else '--'):>11s}"
                    row += f"   {nsub.get(c,0)}"
                    print(row)
            print(f"\n[{key}] physical-unit error, subject-macro (RMSE / nRMSE; RMSE in %BW for vGRF, Nm/kg for moments)")
            print(f"   {'cohort':12s}" + "".join(f"{m:>20s}" for m in CHANS_M))
            for c in COH_ORDER:
                if c not in error_macro:
                    continue
                row = f"   {c:12s}"
                for m in CHANS_M:
                    cell = error_macro[c].get(m)
                    if cell and "RMSE" in cell and "nRMSE_pct" in cell:
                        txt = "%.2f / %.1f%%" % (cell["RMSE"], cell["nRMSE_pct"])
                    else:
                        txt = "--"
                    row += f"{txt:>20s}"
                print(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    guarded_dump(out, out_path, "eval_kinetics_all_cycles.py")
    print(f"\nsaved {out_path} (complete selected CV5 variants)")
    print("   blocks in file: %s" % sorted(out))


if __name__ == "__main__":
    main()
