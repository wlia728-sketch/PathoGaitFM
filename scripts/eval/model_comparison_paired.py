"""Patient-clustered uncertainty and patient-level paired tests for archived models.
Point estimates retain recording-weighted PD cohorts. PD ON/OFF records stay
together when resampled. Paired tests average their differences within patient.
Inputs are the locally produced evaluation results.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from guarded_write import guarded_dump
from evaluation_contract import safe_result_path, sha256
from patient_statistics import resample_units, paired_test, groups
COHORTS = ["cp", "vdk_stroke", "bmclab_pd", "normal"]
N_BOOT, SEED = 5000, 20260726
# keep in step with published_comparators.ARCHS; the arm suffix selects the information-matched run
COMPARATORS = [("groundlink_cnn__matched", "GroundLinkNet"),
               ("sugai_lstm__matched", "Sugai LSTM"),
               ("ozates_cnn__matched", "Ozates CNN")]
# readout arms from head_vs_headless_ablation.py, scored on the same evaluation units
HEAD_ARMS = [("A2", "A2"), ("headless_3seed", "head-free 3-seed")]


def load_models(kinetics_path, comparators_path):
    """Per-subject scores written locally by eval_kinetics_all_cycles.py and published_comparators.py."""
    for path in (kinetics_path, comparators_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run the evaluation script that produces it first")
    D = json.load(open(kinetics_path, encoding="utf-8"))["pretrain/CV5"]
    S = json.load(open(comparators_path, encoding="utf-8"))
    missing = [k for k, _ in COMPARATORS if k not in S]
    if missing:
        raise KeyError("comparator artifact is missing %s; re-run published_comparators.py" % missing)
    # per-subject overall scores, keyed cohort -> subject
    per_subj = {"diffusion": D["per_subject_overall"]}
    per_subj.update({label: S[key]["per_subject_overall"] for key, label in COMPARATORS})
    # Per-cohort per-channel; the diffusion channel names differ from the baseline ones.
    # cohort_channel_macro, NOT per_cohort_merged: the latter is a mean over limb CYCLES while the
    # comparators' per_cohort is a mean over per-SUBJECT channel means, and cycle pooling overweights
    # bmclab_pd, which supplies 1,164 of 1,703 scored limb-cycles but 46 of 111 subjects. The
    # subject-macro form is the one the Methods declare.
    dch = {c: {k.replace("_Mom", "").replace("GRF_V", "vGRF"): v["pcc"]
               for k, v in D["cohort_channel_macro"][c].items()} for c in COHORTS}
    per_chan = {"diffusion": dch}
    per_chan.update({label: S[key]["per_cohort"] for key, label in COMPARATORS})
    published = {"diffusion": D["headline_subject_macro"]}
    published.update({label: S[key]["subject_macro_PCC"] for key, label in COMPARATORS})
    return per_subj, per_chan, published


def macro_from_subjects(per_subj_model, subjects_by_cohort):
    """cohort-equal-weight macro over a given subject selection."""
    means = [float(np.mean([per_subj_model[c][s] for s in subjects_by_cohort[c]])) for c in COHORTS]
    return float(np.mean(means))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kinetics',type=Path,default=ROOT/'outputs/evaluation/kinetics_cv5.json')
    parser.add_argument('--comparators',type=Path,default=ROOT/'outputs/evaluation/published_comparators.json')
    parser.add_argument('--head',type=Path,default=ROOT/'outputs/evaluation/head_vs_headless_ablation.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/evaluation/model_comparison.json')
    args=parser.parse_args()
    out_path=safe_result_path(args.out,ROOT)
    per_subj, per_chan, published = load_models(args.kinetics, args.comparators)
    models = ["diffusion"] + [label for _, label in COMPARATORS]

    # subjects present for EVERY model (they are the same 111 by construction; verified, not assumed)
    common = {}
    for c in COHORTS:
        sets = [set(per_subj[m][c]) for m in models]
        common[c] = sorted(set.intersection(*sets))
    n_common = {c: len(common[c]) for c in COHORTS}
    for m in models:
        for c in COHORTS:
            assert set(per_subj[m][c]) == set(common[c]), f"{m}/{c} subject set differs"

    if not args.head.is_file():
        raise FileNotFoundError(f"{args.head} not found; run head_vs_headless_ablation.py first")
    head = json.load(open(args.head, encoding="utf-8"))
    for key, label in HEAD_ARMS:
        if "per_subject_overall" not in head.get(key, {}):
            raise KeyError("%s lacks per_subject_overall; re-run head_vs_headless_ablation.py" % key)
        per_subj[label] = head[key]["per_subject_overall"]
        for c in COHORTS:
            assert set(per_subj[label][c]) == set(common[c]), f"{label}/{c} subject set differs"
    arms = models + [label for _, label in HEAD_ARMS]

    point = {m: round(macro_from_subjects(per_subj[m], common), 4) for m in arms}

    rng = np.random.default_rng(SEED)
    boot = {m: np.empty(N_BOOT) for m in arms}
    for b in range(N_BOOT):
        sel = {c: resample_units(c, common[c], rng) for c in COHORTS}
        for m in arms:
            boot[m][b] = macro_from_subjects(per_subj[m], sel)

    def compare(a, ref):
        differences = {c: {st: per_subj[a][c][st] - per_subj[ref][c][st]
                           for st in common[c]} for c in COHORTS}
        result = paired_test(differences)
        result["macro_difference"] = round(macro_from_subjects(per_subj[a], common)
                                           - macro_from_subjects(per_subj[ref], common), 4)
        result["CI95_patient_clustered_bootstrap"] = [
            round(float(x), 4) for x in np.percentile(boot[a] - boot[ref], [2.5, 97.5])]
        return result
    comparisons = {f"diffusion_minus_{m}": compare("diffusion", m) for m in models[1:]}
    head_comparisons = {f"{a}_minus_{ref}": compare(a, ref)
                       for a, ref in (("A2", "Sugai LSTM"), ("A2", "head-free 3-seed"))}
    # the ablation producer summarises channel-first and this file subject-first, so both are reported
    head_macro_check = {label: {"producer_channel_first": head[key]["subject_macro_PCC"],
                                "recomputed_subject_first": point[label],
                                "gap": round(point[label] - head[key]["subject_macro_PCC"], 4)}
                        for key, label in HEAD_ARMS}

    # channel-matched macro: vertical GRF only, the channel every cohort carries
    chan_matched = {m: round(float(np.mean([per_chan[m][c]["vGRF"] for c in COHORTS])), 4) for m in models}
    all_chan = {m: round(float(np.mean([float(np.mean(list(per_chan[m][c].values()))) for c in COHORTS])), 4)
                for m in models}

    out = {"n_subjects_per_cohort": n_common, "n_subjects_total": int(sum(n_common.values())),
           "n_bootstrap": N_BOOT, "seed": SEED,
           "macro_published_by_producer": {m: published[m] for m in models},
           "macro_recomputed_from_per_subject": point,
           "paired_comparisons_vs_diffusion": comparisons,
           "paired_comparisons_head_readout": head_comparisons,
           "head_readout_macro_check": head_macro_check,
           "channel_matched_macro_vGRF_only": chan_matched,
           "all_available_channel_macro": all_chan,
           "per_cohort_per_channel": per_chan}
    out['n_evaluation_units_per_cohort']=out.pop('n_subjects_per_cohort')
    out['n_evaluation_units_total']=out.pop('n_subjects_total')
    out['n_patients_per_cohort']={c:len(groups(c,common[c])) for c in COHORTS}
    out['bootstrap_unit']='patient, stratified by cohort; keep PD medication states together'
    out['point_weighting']='cohort equal; evaluation units equal within each cohort'
    out['input_sha256']={p.name:sha256(p) for p in (args.kinetics,args.comparators,args.head)}
    out_path.parent.mkdir(parents=True,exist_ok=True)
    guarded_dump(out,out_path,'model_comparison_paired.py')
    print('wrote',out_path)
    print(json.dumps(comparisons,indent=2))


if __name__ == "__main__":
    main()
