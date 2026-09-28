# -*- coding: utf-8 -*-
"""Re-run the three published comparators to obtain normalised RMSE alongside PCC.

The comparator pipeline keeps only the scalar correlation of each scored cycle, so the predictions are
regenerated here with the same architectures, split, cycle gates, target masks, conditioning, three-seed
ensembling and subject-macro aggregation, and nRMSE is computed with rmse_nrmse from
eval_kinetics_all_cycles on the same arrays that feed safe_pcc: per limb-cycle, mean within subject and
channel, then mean over subjects. Only the information-matched arm is run. Writes
outputs/evaluation/comparators_nrmse.json.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PKG = Path(__file__).resolve().parents[3]
EVAL = PKG / "model_baseline"
sys.path.insert(0, str(EVAL))

import torch  # noqa: E402
import baseline_nondiffusion as B  # noqa: E402
from published_comparators import ARCHS  # noqa: E402
from eval_kinetics_all_cycles import rmse_nrmse, MERGED8, UNIT_SCALE, UNIT_NAME  # noqa: E402

OUT = PKG / "outputs" / "evaluation" / "comparators_nrmse.json"
ARM, IN_MASK = "matched", True


def eval_fold_both(models, va, seq, in_mask, acc_pcc, acc_nrm, acc_rmse):
    """baseline_nondiffusion.eval_fold with two extra accumulators on the identical cycle gate."""
    for m in models:
        m.eval()
    with torch.no_grad():
        for cohort, stem, xin, yout, v40, cond in va:
            n = xin.shape[0]
            xn = B.stack_input(xin, v40, seq, in_mask)
            xb = torch.tensor(xn, dtype=torch.float32).to(B.DEVICE)
            cb = torch.tensor(np.tile(cond, (n, 1)), dtype=torch.long).to(B.DEVICE)
            pred = np.mean([m(xb, cb).cpu().numpy() for m in models], axis=0)
            for ci in range(n):
                for vg in (22, 25):
                    side_t = B.scored_side_channels(cohort, vg)
                    if not all(v40[ci, c] for c in side_t):
                        continue
                    for c in side_t:
                        ti = B.TGT.index(c)
                        gt, pr = yout[ci, ti, :], pred[ci, ti, :]
                        r = B.safe_pcc(gt, pr)
                        if not np.isfinite(r):
                            continue
                        # error accumulates INSIDE the correlation's own guard, the way the diffusion
                        # evaluator does it, so the error cells are identical to the PCC cells by
                        # construction rather than by coincidence
                        acc_pcc[cohort][B.MERGED[c]][stem].append(r)
                        rm, nrm = rmse_nrmse(gt, pr, c)
                        mc = MERGED8[c]
                        if rm is not None and np.isfinite(rm):
                            acc_rmse[cohort][B.MERGED[c]][stem].append(rm * UNIT_SCALE[mc])
                        if nrm is not None and np.isfinite(nrm):
                            acc_nrm[cohort][B.MERGED[c]][stem].append(nrm * 100.0)


def macro(acc):
    """Per-subject mean, then mean over subjects. The aggregation subject_macro() applies to PCC."""
    per_cohort, per_subject = {}, {}
    for cohort in acc:
        ch_means, subj_ch = {}, defaultdict(dict)
        for chm, sd in acc[cohort].items():
            ps = []
            for stem, v in sd.items():
                if not v:
                    continue
                m = float(np.mean(v))
                ps.append(m)
                subj_ch[stem][chm] = m
            if ps:
                ch_means[chm] = {"value": float(np.mean(ps)), "n_subj": len(ps)}
        per_cohort[cohort] = ch_means
        per_subject[cohort] = {st: float(np.mean(list(cm.values()))) for st, cm in subj_ch.items() if cm}
    return per_cohort, per_subject


def run_one(build_model):
    B.preflight_inputs(SEV)
    acc_pcc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    acc_nrm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    acc_rms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nparam, t0 = None, time.time()
    for fold in B.CV5:
        tr_subj, va_subj = B.get_split_subjects(B.get_split(), fold)
        tr = B.load_cycles(TRANSFORM, SEV, tr_subj)
        va = B.load_cycles(TRANSFORM, SEV, va_subj)
        if not tr or not va:
            raise ValueError(f"Comparator {fold} has empty training or validation data")
        models = []
        for so in range(B.N_SEED):
            torch.manual_seed(so * 7919)
            np.random.seed(so * 7919)
            m = build_model()
            if nparam is None:
                nparam = sum(p.numel() for p in m.parameters())
            models.append(B.train_model(m, tr, seq=True, in_mask=IN_MASK))
        eval_fold_both(models, va, True, IN_MASK, acc_pcc, acc_nrm, acc_rms)
        print("    fold %s done (%.0f s)" % (fold, time.time() - t0), flush=True)
    pcc_c, pcc_s = macro(acc_pcc)
    nrm_c, _ = macro(acc_nrm)
    rms_c, _ = macro(acc_rms)
    cohort_overall = {c: float(np.mean([d["value"] for d in v.values()]))
                      for c, v in pcc_c.items() if v}
    return {
        "subject_macro_PCC": round(float(np.mean(list(cohort_overall.values()))), 4),
        "per_cohort_overall": {c: round(v, 4) for c, v in cohort_overall.items()},
        "per_cohort_PCC": {c: {ch: round(d["value"], 4) for ch, d in v.items()} for c, v in pcc_c.items()},
        "per_cohort_nRMSE_pct": {c: {ch: round(d["value"], 1) for ch, d in v.items()}
                                 for c, v in nrm_c.items()},
        "per_cohort_RMSE": {c: {ch: round(d["value"], 2) for ch, d in v.items()} for c, v in rms_c.items()},
        "RMSE_units": {B.MERGED[c]: UNIT_NAME[MERGED8[c]] for c in B.TGT},
        "n_subj_per_cell": {c: {ch: d["n_subj"] for ch, d in v.items()} for c, v in pcc_c.items()},
        "nRMSE_pct_mean_over_channels": {c: round(float(np.mean([d["value"] for d in v.values()])), 1)
                                         for c, v in nrm_c.items() if v},
        "per_subject_overall_PCC": {c: {st: round(v, 6) for st, v in sd.items()} for c, sd in pcc_s.items()},
        "n_params": int(nparam) if nparam else None,
        "n_seed_ensemble": B.N_SEED,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    global TRANSFORM, SEV
    TRANSFORM = B.build_transform()
    SEV = json.load(open(B.SEV, encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("comparator nRMSE re-run, %s arm, on %s" % (ARM, B.DEVICE), flush=True)
    print("channel-name map, comparator -> diffusion artifact: %s"
          % {B.MERGED[c]: MERGED8[c] for c in B.TGT}, flush=True)
    torch.manual_seed(0)
    np.random.seed(0)
    out = {"arm": ARM, "note": __doc__.strip().split("\n")[0],
           "nRMSE_definition": "range-normalised per limb-cycle, published rmse_nrmse(), per cent",
           "aggregation": "per cycle -> subject mean -> mean over subjects, as the PCC macro",
           "models": {}}
    for name, builder in ARCHS:
        print("\n[%s]" % name, flush=True)
        res = run_one(lambda b=builder: b(B.NIN * 2))
        out["models"][name] = res
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        print("  PCC %.4f" % res["subject_macro_PCC"], flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n%-18s %8s %8s %8s %8s %8s" % ("model / cohort", "vGRF", "Ankle", "Hip", "Knee", "mean"))
    for name in out["models"]:
        print("[%s] nRMSE %%" % name)
        for coh in ("normal", "cp", "vdk_stroke", "bmclab_pd"):
            cells = out["models"][name]["per_cohort_nRMSE_pct"].get(coh, {})
            row = "".join("%8s" % (cells.get(ch, "NA")) for ch in ("vGRF", "Ankle", "Hip", "Knee"))
            mean = out["models"][name]["nRMSE_pct_mean_over_channels"].get(coh, "NA")
            print("  %-16s %s %8s" % (coh, row, mean))
    print("wrote", OUT, flush=True)


TRANSFORM = None
SEV = None

if __name__ == "__main__":
    main()
