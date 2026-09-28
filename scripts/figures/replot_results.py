"""Numerical panel data and simple verification plots from locally produced evaluation results.

The inputs are the JSON results written by the evaluation scripts (scripts/eval/) and the nRMSE
captures written by scripts/eval/nrmse_rerun/, gathered in one directory given by --results-dir,
plus the per-participant external moment scores CSV given by --moment-scores. None of these are
distributed with the code; they are produced locally from your own copies of the source data. The
plots expose those results; they do not recreate the exact manuscript layout or rerun a model.
Uhlrich moments always use the measured reference window, never padded full-cycle scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
COHORTS = ("normal", "cp", "vdk_stroke", "bmclab_pd")
COHORT_LABELS = ("TD", "CP", "Stroke", "PD")
SOURCES: dict[str, str] = {}
MISSING: list[str] = []
RESULTS_DIR = Path()
MOMENT_SCORES = Path()


def source(path):
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; it is produced locally by the evaluation scripts (see the module docstring)")
    SOURCES[str(path).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load(name, capture=False):
    data = json.loads(source(RESULTS_DIR / name).read_text(encoding="utf-8"))
    return data["writes"][-1]["out"] if capture else data


def mean(values):
    values = list(values)
    if not values or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Expected nonempty finite archived values")
    return statistics.mean(values)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def internal_rows():
    kinetic = load("_kinetics_macro_eval.json")
    comparators = load("_published_comparators.json")
    comparator_error = load("_comparators_nrmse.json")["models"]
    head = load("_head_vs_headless_ablation.json")
    head_error = load("nrmse_runs/head_vs_headless_ablation_nrmse.json", capture=True)
    rows = []
    names = (("ozates_cnn", "Ozates CNN"), ("groundlink_cnn", "GroundLinkNet"),
             ("sugai_lstm", "Sugai LSTM"))
    for key, label in names:
        pcc = comparators[key + "__matched"]
        errors = comparator_error[key]["nRMSE_pct_mean_over_channels"]
        for cohort in COHORTS:
            rows.append(dict(model=label, cohort=cohort, PCC=pcc["per_cohort_overall"][cohort],
                nRMSE_percent=errors[cohort],
                source=f"_published_comparators.json/{key}__matched; _comparators_nrmse.json/models/{key}",
                note="nRMSE cohort means are rounded archived summaries; no per-unit error distribution is supplied"))
        rows.append(dict(model=label, cohort="overall", PCC=pcc["subject_macro_PCC"],
            nRMSE_percent=mean(errors[c] for c in COHORTS),
            source=f"_published_comparators.json/{key}__matched; _comparators_nrmse.json/models/{key}",
            note="nRMSE overall is the equal mean of stored rounded cohort summaries"))
    for key, label in (("scratch/CV5", "Without pretraining"), ("pretrain/CV5", "PathoGaitFM")):
        data = kinetic[key]
        # Average targets within evaluation unit and then units within cohort.
        errors = {c: mean(mean(v.values()) for v in data["per_subject_nRMSE_pct"][c].values()) for c in COHORTS}
        for cohort in COHORTS:
            rows.append(dict(model=label, cohort=cohort, PCC=data["cohort_subject_macro"][cohort]["pcc"],
                nRMSE_percent=errors[cohort], source=f"_kinetics_macro_eval.json/{key}",
                note="nRMSE recomputed from archived per-unit target scores without intermediate rounding"))
        rows.append(dict(model=label, cohort="overall", PCC=data["headline_subject_macro"],
            nRMSE_percent=mean(errors.values()), source=f"_kinetics_macro_eval.json/{key}",
            note="equal cohort mean; no intermediate rounding" +
                ("; TD nRMSE is 11.337 (historical display 11.4 uses double rounding)" if key == "pretrain/CV5" else "")))
    for arm, label in (("A1", "Direct regressor"), ("A2", "Supervised head")):
        for cohort in COHORTS:
            rows.append(dict(model=label, cohort=cohort, PCC=head[arm]["per_cohort"][cohort],
                nRMSE_percent=head_error[arm]["per_cohort"][cohort],
                source=f"_head_vs_headless_ablation.json/{arm}; nrmse_runs/head_vs_headless_ablation_nrmse.json/{arm}",
                note="nRMSE capture uses legacy PCC field names for captured errors"))
        rows.append(dict(model=label, cohort="overall", PCC=head[arm]["subject_macro_PCC"],
            nRMSE_percent=head_error[arm]["subject_macro_PCC"],
            source=f"_head_vs_headless_ablation.json/{arm}; nrmse_runs/head_vs_headless_ablation_nrmse.json/{arm}", note="equal cohort mean"))
    MISSING.append("Individual nRMSE distributions for the three published comparators are absent; only their archived cohort means are plotted.")
    return rows


def external_rows():
    cp, steele = load("_cp_jackknife.json"), load("_crouch_jackknife.json")
    force, error = load("_external_persubject_vgrf.json"), load("_external_persubject_vgrf_nrmse.json")
    rows = []
    for group, label in (("cp", "Meyns CP"), ("td", "Meyns TD")):
        data = cp[group]
        rows.append(dict(dataset=label, target="vGRF", window="full gait cycle", n=data["n_subj"],
            PCC=data["overall_subject_macro_pcc"], nRMSE_percent=data["nRMSE_pct_subject_macro"],
            RMSE=data["RMSE_pctBW_subject_macro"], RMSE_unit="%BW", source=f"_cp_jackknife.json/{group}"))
    rows.append(dict(dataset="Steele", target="vGRF", window="full gait cycle", n=steele["n_subj"],
        PCC=steele["PCC"]["subject_macro"], nRMSE_percent=steele["nRMSE_pct"]["subject_macro"],
        RMSE=steele["RMSE_pctBW"]["subject_macro"], RMSE_unit="%BW", source="_crouch_jackknife.json"))
    for key, label in (("Fukuchi_healthy", "Fukuchi"), ("PD_Nordic", "Viglialoro")):
        pcc = force[key]["per_channel_limb_merged"]["vGRF"]
        er = error[key]["per_channel_limb_merged"]["vGRF"]
        rows.append(dict(dataset=label, target="vGRF", window="full gait cycle", n=pcc["n_subj"],
            PCC=pcc["pcc"], nRMSE_percent=er["nRMSE_pct"], RMSE=er["RMSE"], RMSE_unit="%BW",
            source=f"_external_persubject_vgrf.json/{key}; _external_persubject_vgrf_nrmse.json/{key}"))
    fukuchi = load("_external_moments.json")["sites"]["fukuchi"]["per_joint"]
    uhlrich = load("_opencap_moments.json")["per_joint"]
    table = source(MOMENT_SCORES)
    with table.open(encoding="utf-8-sig", newline="") as stream:
        participants = list(csv.DictReader(stream))
    for label, data, window in (("Fukuchi", fukuchi, "full"), ("Uhlrich", uhlrich, "reference")):
        for joint in ("Hip", "Knee", "Ankle"):
            cell = data[joint][window]
            subset = [r for r in participants if r["dataset"] == label and r["joint"] == joint]
            if len(subset) != cell["n_subj_scored"] or len({r["subject"] for r in subset}) != len(subset):
                raise ValueError(f"Fig. 3e score population differs for {label}/{joint}")
            rmse = mean(float(r["RMSE_Nm_per_kg"]) for r in subset)
            if abs(rmse - cell["RMSE_Nm_per_kg"]) > 0.00011:
                raise ValueError(f"Fig. 3e mean differs from frozen moment result: {label}/{joint}")
            rows.append(dict(dataset=label, target=joint, window="full gait cycle" if window == "full" else "left stance, samples 0-63",
                n=len(subset), PCC=cell["PCC_subject_macro"], nRMSE_percent=cell["nRMSE_pct"],
                RMSE=rmse, RMSE_unit="N·m/kg", source=f"{'_external_moments.json/sites/fukuchi/per_joint' if window == 'full' else '_opencap_moments.json/per_joint'}/{joint}/{window}; {MOMENT_SCORES.name}"))
    return rows


def incomplete_rows():
    pcc = load("_partial_input_masking.json")
    error = load("nrmse_runs/partial_input_masking_nrmse.json", capture=True)
    labels = dict(full="All available", no_pelvis="No pelvis", no_hip_ang="No hip", no_knee_ang="No knee",
        no_ankle_ang="No ankle", ladder_no_ankle_pelvis="No ankle / pelvis", ladder_hip_only="Hip only")
    rows = []
    for key, label in labels.items():
        for cohort in (*COHORTS, "overall"):
            overall = cohort == "overall"
            rows.append(dict(configuration=label, configuration_key=key, cohort=cohort,
                PCC=pcc[key]["subject_macro_PCC"] if overall else pcc[key]["per_cohort"][cohort],
                nRMSE_percent=error[key]["subject_macro_PCC"] if overall else error[key]["per_cohort"][cohort],
                source=f"_partial_input_masking.json/{key}; nrmse_runs/partial_input_masking_nrmse.json/{key}"))
    MISSING.append("Reduced-input target-specific physical RMSE is not available in these frozen summaries; no RMSE bars or target-level deterioration values are invented.")
    return rows


def paired_rows():
    rows = []
    for dataset, prefix, alternative in (("Uhlrich", "opencap", "video"), ("Grouvel", "imu", "imu")):
        for branch in ("marker", alternative):
            name = f"_{prefix}_zeroshot_{branch}.json"
            capture = f"nrmse_runs/eval_{prefix}_zeroshot_{branch}_nrmse.json"
            pcc, error = load(name), load(capture, capture=True)
            p = pcc["vgrf_per_subject"]
            e = error["vgrf_per_subject"]
            if set(p) != set(e):
                raise ValueError(f"Paired score population differs for {dataset}/{branch}")
            for subject in sorted(p):
                rows.append(dict(dataset=dataset, modality=branch, subject=subject.split("__")[0],
                    PCC=p[subject], nRMSE_percent=e[subject], source=f"{name}; {capture}"))
    return rows


def plot_panels(out, internal, external, incomplete, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "axes.titleweight": "bold"})
    colours = ["#147D92", "#C68A20", "#4476AE", "#9464A2"]

    def export(fig, name):
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def mean_bars(rows, namekey, name):
        overall = [r for r in rows if r["cohort"] == "overall"]
        names = [r[namekey] for r in overall]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4), constrained_layout=True)
        for axis, metric in zip(axes, ("PCC", "nRMSE_percent")):
            y = np.arange(len(names))
            axis.barh(y, [r[metric] for r in overall], color="#DCE9EB", edgecolor="#8DA6AC", height=.65)
            for colour, cohort, cohort_label in zip(colours, COHORTS, COHORT_LABELS):
                values = [next(r[metric] for r in rows if r[namekey] == model and r["cohort"] == cohort) for model in names]
                axis.plot(values, y, "o", color=colour, ms=4, label=cohort_label)
            axis.set_yticks(y, names)
            axis.invert_yaxis()
            axis.set_xlabel("PCC" if metric == "PCC" else "nRMSE (%)")
            axis.set_xlim(left=0, right=1.03 if metric == "PCC" else None)
            axis.grid(axis="x", alpha=.15)
        axes[0].set_title("Waveform agreement", loc="left")
        axes[1].set_title("Normalised error", loc="left")
        axes[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.16))
        fig.suptitle("Numerical replot | bars: equal cohort mean; points: cohort means", fontsize=10)
        export(fig, name)

    mean_bars(internal, "model", "internal_model_comparison")
    mean_bars(incomplete, "configuration", "incomplete_inputs")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    for row_index, targets in enumerate(("vGRF", "moments")):
        selected = [r for r in external if (r["target"] == "vGRF") == (targets == "vGRF")]
        labels = [r["dataset"] if targets == "vGRF" else f"{r['dataset']} {r['target']}" for r in selected]
        for col, metric in enumerate(("PCC", "nRMSE_percent", "RMSE")):
            axis = axes[row_index, col]
            axis.barh(range(len(selected)), [r[metric] for r in selected], color="#DCE9EB", edgecolor="#147D92", height=.65)
            axis.set_yticks(range(len(selected)), labels)
            axis.invert_yaxis()
            axis.set_xlabel("PCC" if metric == "PCC" else "nRMSE (%)" if metric == "nRMSE_percent" else "RMSE (%BW)" if targets == "vGRF" else "RMSE (N·m/kg)")
            axis.set_xlim(left=0, right=1.03 if metric == "PCC" else None)
            axis.grid(axis="x", alpha=.15)
    fig.suptitle("External numerical replot | Uhlrich moments: measured left-stance window", fontsize=11)
    export(fig, "external_validation")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("PCC", "nRMSE_percent")):
        for dataset_index, (dataset, alternative) in enumerate((("Uhlrich", "video"), ("Grouvel", "imu"))):
            values = {branch: {r["subject"]: r[metric] for r in paired if r["dataset"] == dataset and r["modality"] == branch} for branch in ("marker", alternative)}
            if set(values["marker"]) != set(values[alternative]):
                raise ValueError(f"Unpaired participants in {dataset}")
            x = [dataset_index * 3, dataset_index * 3 + 1]
            for subject in values["marker"]:
                axis.plot(x, [values["marker"][subject], values[alternative][subject]], "o-", color=colours[dataset_index], alpha=.4, ms=3, lw=.7)
            axis.plot(x, [mean(values[branch].values()) for branch in ("marker", alternative)], "D", color="#172E38", ms=5)
        axis.set_xticks([0, 1, 3, 4], ["Uhlrich\nmarker", "Uhlrich\nvideo", "Grouvel\nmarker", "Grouvel\nIMU"])
        axis.set_ylabel("PCC" if metric == "PCC" else "nRMSE (%)")
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("Paired-modality numerical replot | diamonds: mean across participants", fontsize=10)
    export(fig, "paired_modalities")


def main():
    global RESULTS_DIR, MOMENT_SCORES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="directory holding the evaluation JSON results and the nrmse_runs/ captures")
    parser.add_argument("--moment-scores", type=Path, required=True,
                        help="per-participant external moment scores CSV (dataset, subject, joint, window, RMSE_Nm_per_kg, PCC, nRMSE_percent)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/numerical_replots")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR, MOMENT_SCORES = args.results_dir.resolve(), args.moment_scores.resolve()
    out = args.out.resolve()
    figures = ROOT / "figures"
    if out == figures or figures in out.parents:
        parser.error("Choose an output folder outside figures/")
    out.mkdir(parents=True, exist_ok=True)
    internal, external, incomplete, paired = internal_rows(), external_rows(), incomplete_rows(), paired_rows()
    for name, rows in (("internal_models", internal), ("external_targets", external), ("incomplete_inputs", incomplete), ("paired_modalities", paired)):
        save_csv(out / f"{name}.csv", rows)
    if not args.data_only:
        plot_panels(out, internal, external, incomplete, paired)
    report = {"scope": "Numerical replots from locally produced score summaries; not model reruns or exact manuscript layouts",
        "row_counts": {"internal_models": len(internal), "external_targets": len(external), "incomplete_inputs": len(incomplete), "paired_modalities": len(paired)},
        "sources_sha256": SOURCES, "unavailable": MISSING,
        "rounding_note": "Per-unit internal nRMSE aggregation gives TD 11.337%; the manuscript's historical 11.4% display uses intermediate rounding. Raw archived scores are unchanged."}
    (out / "replot_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["row_counts"]))
    print("Wrote numerical replots to", out)
    for item in MISSING:
        print("Not plotted:", item)


if __name__ == "__main__":
    main()
