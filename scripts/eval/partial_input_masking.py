"""(b) PARTIAL-INPUT ROBUSTNESS, single joint-group removal and a cumulative ladder.

One factor at a time. From the full laboratory input we withdraw exactly one bilateral joint group
and measure the change in 8-channel reconstruction PCC, so any decrease is attributable to that
group alone. Bilateral throughout, no one-sided conditions. Two further conditions extend the
single removals into a cumulative ladder, withdrawing groups in the order their own removal costs,
least first, so the four rungs of 16, 14, 8 and 6 input channels show how far the input can be cut.

Conditions (each = DROP_16CH base + the withdrawn channels):
  full                    : full laboratory input, 16 channels  -- reference
  no_pelvis               : + mask the six pelvis channels
  no_hip_ang              : + mask every hip-angle channel
  no_knee_ang             : + mask every knee-angle channel
  no_ankle_ang            : + mask every ankle-angle channel
  ladder_no_ankle_pelvis  : + mask the ankle and the pelvis,    8 channels left
  ladder_hip_only         : + mask the ankle, pelvis and knee,  6 channels left
Eval on CV5 (pretrain expA); TSA 3-seed, Method B, subject-macro PCC, 8ch. Every fold checkpoint and
every admitted evaluation unit must be present: the run fails before inference when a weight or an
input is missing, and after scoring when a unit was not scored.
ASCII; json dumped before prints. Writes outputs/evaluation/partial_input_masking.json.
"""
import sys, argparse
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
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS, COH, EXCLUDE, GRFFIX, load_split
from drop_channels import DROP_16CH
from eval_common import (
    load_model, safe_pcc, get_val_pairs, load_valid_mask_40, source_type_and_name,
    SOURCE_TO_ID, SEED, SEVERITY_JSON, build_known_mask, load_local_json, per_subject_transform,
)
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

CV5 = [f"cv5_{i}" for i in range(5)]
TGT = [10, 13, 16, 17, 18, 19, 22, 25]
MERGED = {10: "Hip_Mom", 13: "Hip_Mom", 16: "Knee_Mom", 17: "Knee_Mom", 18: "Ankle_Mom", 19: "Ankle_Mom", 22: "GRF_V", 25: "GRF_V"}
# masking pools by joint (40ch idx). The per-joint lists below are kinematic channels.
HIP_ANG = [0, 1, 2, 3, 4, 5]; KNEE_ANG = [6, 7]; ANKLE_ANG = [8, 9]
# One joint group at a time, every channel that group has. HIP_ANG was [0, 3] until 2026-09-01,
# which withdrew only the sagittal hip and left the four non-sagittal hip channels in the input,
# so that arm was not a hip removal at all. Knee and ankle have no non-sagittal channels, so
# their lists are unchanged and their published values reproduce.
PELVIS = [34, 35, 36, 37, 38, 39]
BASE_DROP = list(DROP_16CH)
CONDITIONS = {
    "full": [],
    "no_pelvis": PELVIS,
    "no_hip_ang": HIP_ANG,
    "no_knee_ang": KNEE_ANG,
    "no_ankle_ang": ANKLE_ANG,
    # A cumulative ladder, withdrawing groups in the order their single-group removal costs, least
    # first: ankle 0.003, pelvis 0.004, knee 0.022, hip 0.041. Read together with `full` and
    # `no_ankle_ang` it gives four rungs of 16, 14, 8 and 6 input channels, ending on the hip alone.
    "ladder_no_ankle_pelvis": ANKLE_ANG + PELVIS,
    "ladder_hip_only": ANKLE_ANG + PELVIS + KNEE_ANG,
}
BATCH = 32


def gen_tsa(model, sched, gt, valid, sid, cid, svid, imt, device, drop_ch):
    gens = []
    for so in range(3):
        torch.manual_seed(SEED + so * 7919)
        with torch.no_grad():
            gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                             50, 1.0, device, imt, CHECKPOINT_STEPS, drop_ch))
    g = torch.stack(gens).mean(0)
    km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
    return g.cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--raw-root", type=Path, default=GRFFIX)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation/partial_input_masking.json")
    args = parser.parse_args(argv)
    out_path = safe_result_path(args.out, ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("(b) PARTIAL-INPUT MASKING robustness | device", device, flush=True)
    split = load_split()
    sev = load_local_json(SEVERITY_JSON, "severity table")
    checkpoints = [args.checkpoint_root / f"v4_stage2_expA_dynpool_{fold}" / "final.pt" for fold in CV5]
    expected, coverage = preflight_cv5(split, args.raw_root, sev, checkpoints, EXCLUDE)
    meta = V4SubjectMetadata()
    transform = per_subject_transform(meta, require_tables=True)
    for source, stem in set.union(*expected.values()):
        meta.get_bw(source, stem)  # missing anthropometry fails before model inference
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

    out = {}
    for cond, extra in CONDITIONS.items():
        drop_ch = sorted(set(BASE_DROP) | set(extra))
        subj_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        scored = set()
        for fold, cp in zip(CV5, checkpoints):
            model, ckpt, _, imt = load_model(cp, device); model.eval()
            if isinstance(ckpt, dict):
                _i = ckpt.get("input_mask_token_ema")
                if _i is None: _i = ckpt.get("input_mask_token")
                if _i is not None: imt = _i
            for src, stem in get_val_pairs(split, fold):
                if stem in EXCLUDE or (src, stem) not in expected[fold]:
                    continue  # declared exclusion, or no valid target cycle (listed in the coverage report)
                rp = args.raw_root / src / f"{stem}.npy"
                s = sev["mappings"][src][stem]
                try:
                    raw = np.load(str(rp)); v40 = load_valid_mask_40(rp, raw.shape[0])
                    st, sn = source_type_and_name(src)
                    x40 = transform(raw, sn, st, stem)[:, :100, :].astype(np.float32)
                    if v40.shape[0] != x40.shape[0]:
                        raise ValueError("mask and waveform cycle counts differ")
                except Exception as exc:
                    raise RuntimeError(f"Failed loading/transforming {src}/{stem}") from exc
                cohort = COH.get(int(s["cohort_id"]), "other")
                R = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18]); L = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
                keep = [ci for ci in range(x40.shape[0]) if (v40[ci, 22] and all(v40[ci, c] for c in R)) or (v40[ci, 25] and all(v40[ci, c] for c in L))]
                if not keep:
                    raise RuntimeError(f"{src}/{stem} has no scorable cycle although the preflight admitted it")
                keep = np.array(keep)
                sid = torch.full((1,), SOURCE_TO_ID.get(src, 0), dtype=torch.long)
                cid = torch.full((1,), int(s["cohort_id"]), dtype=torch.long); svid = torch.full((1,), int(s["severity_id"]), dtype=torch.long)
                for b0 in range(0, len(keep), BATCH):
                    bidx = keep[b0:b0 + BATCH]; nb = len(bidx)
                    gt = torch.from_numpy(np.nan_to_num(x40[bidx].transpose(0, 2, 1), nan=0.0)).to(device)
                    valid = torch.from_numpy(v40[bidx]).to(device)
                    g = gen_tsa(model, sched, gt, valid, sid.expand(nb).to(device), cid.expand(nb).to(device), svid.expand(nb).to(device), imt, device, drop_ch)
                    gtn = gt.cpu().numpy(); vm = v40[bidx]
                    for ci in range(nb):
                        for side, vg in [(R, 22), (L, 25)]:
                            if not vm[ci, vg] or not all(vm[ci, c] for c in side): continue
                            for ch in side:
                                r = safe_pcc(gtn[ci, ch, :], g[ci, ch, :])
                                if np.isfinite(r): subj_scores[cohort][MERGED[ch]][stem].append(r)
                scored.add((src, stem))
            del model
        validate_coverage(set.union(*expected.values()), scored)
        # subject-macro
        per_cohort_overall = {}
        n_subjects = {}
        for cohort in subj_scores:
            chm = {}
            for mc, sd in subj_scores[cohort].items():
                ps = [float(np.mean(v)) for v in sd.values() if v]
                if ps: chm[mc] = float(np.mean(ps))
            if chm: per_cohort_overall[cohort] = float(np.mean(list(chm.values())))
            n_subjects[cohort] = len({stem for sd in subj_scores[cohort].values() for stem in sd})
        macro = round(float(np.mean(list(per_cohort_overall.values()))), 3)
        out[cond] = {"n_dropped_extra": len(extra), "subject_macro_PCC": macro,
                     "per_cohort": {c: round(v, 3) for c, v in per_cohort_overall.items()},
                     "n_subjects": n_subjects}
        print(f"  {cond:14s} (+{len(extra)} masked) macro PCC = {macro}  n_subjects={n_subjects}", flush=True)
    # degradation vs full
    base = out["full"]["subject_macro_PCC"]
    for cond in out:
        out[cond]["degradation_vs_full"] = round(base - out[cond]["subject_macro_PCC"], 3)
    out["coverage"] = {"folds": coverage["folds"], "eligible_units": coverage["eligible_units"],
                       "excluded_no_valid_targets": coverage["excluded_no_valid_targets"]}
    saved_path = guarded_dump(out, out_path, "partial_input_masking.py")
    print("\n=== PARTIAL-INPUT MASKING (CV5 subject-macro PCC, degradation vs full) ===", flush=True)
    for cond in CONDITIONS:
        o = out[cond]
        print(f"  {cond:14s} PCC={o['subject_macro_PCC']}  deg={o.get('degradation_vs_full')}", flush=True)
    print("\nREAD: small degradation under masking = graceful handling of missing input (deployability).", flush=True)
    print("saved", saved_path, flush=True)


if __name__ == "__main__":
    main()
