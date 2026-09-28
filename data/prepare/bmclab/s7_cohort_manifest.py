"""Build cohort manifest summarizing all 5 (+ addbio reference) cohorts.

Read-only. Touches no .npy in data/cohorts_raw/{cp,normal,vdk_*,addbio}/.
Output:
  - data/cohorts_raw/cohort_manifest.csv
  - data/prepare/bmclab/work/step7/cohort_manifest.md (markdown rendering)
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
ED = PROJ / "data" / "cohorts_raw"
OUT_CSV = ED / "cohort_manifest.csv"
OUT_MD = PROJ / "data/prepare/bmclab/work/step7"
OUT_MD.mkdir(parents=True, exist_ok=True)

COHORTS = ["bmclab_pd", "cp", "normal", "vdk_healthy", "vdk_stroke", "addbio"]

# Read channel_names.txt to identify EMG / subtalar slots
with open(ED / "channel_names.txt") as f:
    names = [line.split("\t")[1].strip() for line in f if line.strip()]
EMG_IDX = list(range(40, 48))
SUBTALAR_IDX = [14, 17]
MOMENT_R_IDX = 26   # R_Ankle_Moment_X
GRF_V_R_IDX = 36    # R_GRF_V_Z (vertical post-swap convention)


def cohort_stats(cohort: str):
    cdir = ED / cohort
    if not cdir.is_dir():
        return {"group": cohort, "n_subjects_files": 0, "n_cycles": 0, "_missing": True}
    data_files = sorted(p for p in cdir.glob("*.npy") if not p.name.endswith("_mask.npy"))
    if not data_files:
        return {"group": cohort, "n_subjects_files": 0, "n_cycles": 0, "_missing": True}
    # Aggregate
    all_d = []; all_m = []
    for d in data_files:
        m = d.with_name(d.stem + "_mask.npy")
        if not m.is_file(): continue
        try:
            arr = np.load(d); mk = np.load(m)
        except Exception:
            continue
        if arr.shape[1:] != (101, 54) or mk.shape[1] != 54: continue
        all_d.append(arr); all_m.append(mk)
    D = np.concatenate(all_d, axis=0) if all_d else None
    M = np.concatenate(all_m, axis=0) if all_m else None
    n_cyc = 0 if D is None else int(D.shape[0])

    # Ankle PF peak (most negative ankle moment R per cycle, mask required)
    ankle_pf = []
    if D is not None and M is not None and M[:, MOMENT_R_IDX].any():
        for i in np.where(M[:, MOMENT_R_IDX])[0]:
            v = D[i, :, MOMENT_R_IDX]
            f = v[np.isfinite(v)]
            if f.size:
                ankle_pf.append(float(f.min()))
    ankle_pf = np.asarray(ankle_pf)
    ankle_pf_med = float(np.median(ankle_pf)) if ankle_pf.size else np.nan
    ankle_pf_iqr = (float(np.percentile(ankle_pf, 25)), float(np.percentile(ankle_pf, 75))) if ankle_pf.size else (np.nan, np.nan)

    # GRF vertical R-side availability
    if M is not None:
        grf_pass_pct = float(M[:, GRF_V_R_IDX].mean() * 100)
    else:
        grf_pass_pct = np.nan

    # EMG / subtalar availability
    if M is not None:
        emg_any = bool(M[:, EMG_IDX].any())
        subtalar_any = bool(M[:, SUBTALAR_IDX].any())
    else:
        emg_any = False; subtalar_any = False

    # Subject count: estimate from filename prefix (filenames look like cohort_NN.npy or SUB_state.npy)
    subj_ids = set()
    for d in data_files:
        stem = d.stem
        if cohort == "bmclab_pd":
            # SUB01_off → SUB01
            sub = stem.split("_")[0]
        elif cohort in ("cp", "normal"):
            # cp_01 → 01
            sub = "_".join(stem.split("_")[:2])
        elif cohort.startswith("vdk_"):
            # vdk_healthy_000 → 000
            sub = stem
        elif cohort == "addbio":
            # Camargo2021_AB06_split0 → AB06
            parts = stem.split("_")
            sub = parts[1] if len(parts) > 1 else stem
        else:
            sub = stem
        subj_ids.add(sub)
    n_subjects = len(subj_ids)

    return {
        "group": cohort,
        "n_subjects": n_subjects,
        "n_subject_state_files": len(data_files),
        "n_cycles": n_cyc,
        "ankle_pf_peak_median_per_kg": round(ankle_pf_med, 3) if np.isfinite(ankle_pf_med) else np.nan,
        "ankle_pf_peak_q25_per_kg": round(ankle_pf_iqr[0], 3) if np.isfinite(ankle_pf_iqr[0]) else np.nan,
        "ankle_pf_peak_q75_per_kg": round(ankle_pf_iqr[1], 3) if np.isfinite(ankle_pf_iqr[1]) else np.nan,
        "grf_R_vert_avail_pct": round(grf_pass_pct, 1) if np.isfinite(grf_pass_pct) else np.nan,
        "emg_available": emg_any,
        "subtalar_available": subtalar_any,
    }


# Per-cohort extra metadata (mass range, paper reference). Mass for bmclab_pd from PDGinfo.
mass_ranges = {}
pdg = pd.read_excel(PROJ / "data/prepare/bmclab/raw/PDGinfo.xlsx", sheet_name="PDGinfo", header=0)
mass_ranges["bmclab_pd"] = (float(pdg["Weight (kg)"].min()), float(pdg["Weight (kg)"].max()))
# Other cohorts: mass unknown without separate metadata file; leave NaN with "Wenqi to fill"

papers = {
    "bmclab_pd":   "Boari et al. 2021 (figshare 10.6084/m9.figshare.14896881)",
    "cp":          "(Wenqi to fill)",
    "normal":      "(Wenqi to fill)",
    "vdk_healthy": "van der Krogt et al. (Wenqi to fill exact ref)",
    "vdk_stroke":  "van der Krogt et al. (Wenqi to fill exact ref)",
    "addbio":      "Camargo et al. 2021 (AddBiomechanics)",
}

rows = []
for c in COHORTS:
    r = cohort_stats(c)
    if r.get("_missing"):
        print(f"  {c}: missing dir or empty"); continue
    if c in mass_ranges:
        r["mass_kg_min"] = round(mass_ranges[c][0], 1)
        r["mass_kg_max"] = round(mass_ranges[c][1], 1)
    else:
        r["mass_kg_min"] = np.nan
        r["mass_kg_max"] = np.nan
    r["paper_reference"] = papers.get(c, "(Wenqi to fill)")
    rows.append(r)

df = pd.DataFrame(rows)
df = df[["group", "n_subjects", "n_subject_state_files", "n_cycles",
         "mass_kg_min", "mass_kg_max",
         "ankle_pf_peak_median_per_kg", "ankle_pf_peak_q25_per_kg", "ankle_pf_peak_q75_per_kg",
         "grf_R_vert_avail_pct", "emg_available", "subtalar_available",
         "paper_reference"]]
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}")
print(df.to_string(index=False))

# Markdown rendering
md = ["# Cohort manifest — 5 pathological-gait cohorts + AddBio reference",
      "",
      "Generated by `data/prepare/bmclab/work/s7_cohort_manifest.py`. Read-only on `data/cohorts_raw/`.",
      "",
      df.to_markdown(index=False),
      "",
      "## Field definitions",
      "",
      "- `n_subjects`: unique subject identifier count (parsed from filename).",
      "- `n_subject_state_files`: number of `.npy` files (one per subject-state for bmclab_pd; one per subject for others).",
      "- `n_cycles`: total gait cycles across all subjects (one row per cycle in the .npy).",
      "- `mass_kg_min/max`: subject mass range from cohort metadata (BMClab from PDGinfo.xlsx; others to be filled by Wenqi from cohort references).",
      "- `ankle_pf_peak_median_per_kg`: per-cycle min(R_Ankle_Moment_X), median across cycles where channel mask is True. Sign convention varies across cohorts (see paper Methods).",
      "- `grf_R_vert_avail_pct`: percentage of cycles with R-side vertical GRF channel (ch 36) mask=True.",
      "- `emg_available`: any cycle has any EMG channel (ch 40-47) mask=True.",
      "- `subtalar_available`: any cycle has subtalar channel (ch 14 or 17) mask=True.",
      "- `paper_reference`: dataset citation; placeholders to be filled by Wenqi.",
      "",
      "## Cohort summary stats",
      "",
      f"- Total subjects across all 5 study cohorts: {int(df[df['group'] != 'addbio']['n_subjects'].sum())}",
      f"- Total cycles across all 5 study cohorts: {int(df[df['group'] != 'addbio']['n_cycles'].sum())}",
      f"- AddBio reference: {int(df[df['group']=='addbio']['n_subjects'].iloc[0])} subjects, {int(df[df['group']=='addbio']['n_cycles'].iloc[0])} cycles",
      ""]
(OUT_MD / "cohort_manifest.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nWrote {OUT_MD / 'cohort_manifest.md'}")
