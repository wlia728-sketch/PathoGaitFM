"""Batch GRF driver — write .mot + external_loads.xml + presence vector + mask.json for every walking trial.

Strategy:
  - Iterate over every walking TRC in data/prepare/bmclab/work/trc/ (per-medication-paired with the c3d in C3Dfiles.zip).
  - For each trial, extract the c3d + V3D _grf.csv from the zip on the fly.
  - Per-foot CoP (Fz-weighted across active plates) + V3D R/L force → .mot at 150 Hz.
  - Write per-frame R/L GRF presence (boolean) to _grf_presence/{trial}.csv.
  - Per-trial quality gates: peak Fz / stance duration / anomaly flag.
  - Aggregate trial-level mask categories to _grf_mask.json.
"""
import os
import sys
import csv
import json
import zipfile
import tempfile
import time
import traceback
import re
from pathlib import Path


import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s34_grf_transform import (
    load_c3d_baseline, per_foot_cop_at_pt_rate,
    load_v3d_grf_from_zip, build_grf_mot, build_external_loads_xml,
    linear_resample,
)

PROJ = Path(__file__).resolve().parents[3]
ZIP_PATH = PROJ / "data/prepare/bmclab/raw/C3Dfiles.zip"
TRC_DIR = PROJ / "data/prepare/bmclab/work/trc"
GRF_DIR = PROJ / "data/prepare/bmclab/work/grf"
PRESENCE_DIR = GRF_DIR / "_grf_presence"
LOG_PATH = GRF_DIR / "_grf_log.csv"
MASK_PATH = GRF_DIR / "_grf_mask.json"

# Subject×med where the data is missing per Step 1 Finding 7/8
SIDE_MISSING = {
    ("SUB05", "off"): "R",   # right side GRF missing per Step 1
    ("SUB08", "off"): "R",
    ("SUB15", "off"): "R",
    ("SUB09", "off"): "BOTH",
    ("SUB09", "on"):  "BOTH",
}
EXCLUDED_NO_DATA = {("SUB04", "off"), ("SUB23", "off"), ("SUB25", "off"), ("SUB26", "on")}

pdg = pd.read_excel(PROJ / "data/prepare/bmclab/raw/PDGinfo.xlsx", sheet_name="PDGinfo", header=0)
def mass_of(sid): return float(pdg[pdg["ID"] == sid]["Weight (kg)"].iloc[0])


def list_walking_trcs():
    """Return list of TRC stems for walking trials."""
    SKIP = {"SUB01_off_walk_12", "SUB01_off_walk_14"}  # Trimmed_ versions used instead
    pat = re.compile(r"^(Trimmed_)?SUB\d+_(off|on)_walk_\w+$")
    out = []
    for p in TRC_DIR.iterdir():
        if p.suffix != ".trc": continue
        if not pat.match(p.stem): continue
        if p.stem in SKIP: continue
        out.append(p.stem)
    return sorted(out)


def parse_subject_med(stem: str):
    """Extract (subject, med) from a trial stem (handles Trimmed_ prefix)."""
    toks = stem.split("_")
    if toks[0] == "Trimmed":
        return toks[1], toks[2]
    return toks[0], toks[1]


def derive_zip_entries(stem: str):
    """Build the C3Dfiles.zip entry paths for this trial's c3d + _grf.csv."""
    if stem.startswith("Trimmed_"):
        sub, med = stem.split("_")[1], stem.split("_")[2]
    else:
        sub, med = stem.split("_")[0], stem.split("_")[1]
    folder = f"C3Dfiles/{sub}_{med}/"
    return folder + stem + ".c3d", folder + stem + "_grf.csv"


def process_one(stem: str, zip_path: Path):
    """Process a single trial. Returns (row_for_log, mask_category)."""
    sub, med = parse_subject_med(stem)
    mass = mass_of(sub)
    BW = mass * 9.81

    row = {
        "trial": stem, "subject": sub, "med": med,
        "mass_kg": f"{mass:.2f}",
        "peak_fz_r_N": "", "peak_fz_l_N": "",
        "peak_fz_r_ratio_BW": "", "peak_fz_l_ratio_BW": "",
        "stance_r_s": "", "stance_l_s": "",
        "n_frames_pt": "", "duration_s": "",
        "anomaly": "", "status": "", "error_msg": "",
    }
    mask = "pass"

    # Excluded → skip entirely (no .mot)
    if (sub, med) in EXCLUDED_NO_DATA:
        row["status"] = "SKIP_EXCLUDED"
        return row, "excluded"

    c3d_entry, grf_entry = derive_zip_entries(stem)
    try:
        with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            # Extract c3d
            c3d_local = tmpdir / "x.c3d"
            try:
                with zf.open(c3d_entry) as src, open(c3d_local, "wb") as dst:
                    dst.write(src.read())
            except KeyError:
                row["status"] = "FAIL_NO_C3D"
                row["error_msg"] = f"missing {c3d_entry}"
                return row, "excluded"
            c, anal, a_rate = load_c3d_baseline(str(c3d_local))
            p_rate = int(c["header"]["points"]["frame_rate"])

            # V3D _grf.csv (may be empty/missing)
            try:
                v3d_t, v3d_cols = load_v3d_grf_from_zip(grf_entry)
            except KeyError:
                v3d_t, v3d_cols = None, None

        # Per-foot CoP from analog (at point rate)
        res = per_foot_cop_at_pt_rate(c, anal, mass, p_rate=p_rate)
        if res is None:
            row["status"] = "FAIL_NO_MARKERS"
            row["error_msg"] = "R.Heel/L.Heel missing"
            return row, "excluded"
        cop_r, cop_l, fz_r_analog, fz_l_analog = res
        n_p = cop_r.shape[1]

        # Decide mask category
        side_pref = SIDE_MISSING.get((sub, med), None)
        # V3D-empty trial detection: all V3D force columns ≈ zero
        r_force_max = 0.0
        l_force_max = 0.0
        if v3d_cols is not None:
            for n in ("R_AP","R_VERT","R_ML"):
                r_force_max = max(r_force_max, float(np.nanmax(np.abs(v3d_cols[n]))))
            for n in ("L_AP","L_VERT","L_ML"):
                l_force_max = max(l_force_max, float(np.nanmax(np.abs(v3d_cols[n]))))
        v3d_force_max = max(r_force_max, l_force_max)
        v3d_empty = v3d_cols is None or v3d_force_max < 0.5   # N/kg
        # Strike thresholds: 0.5 N/kg ≈ 0.05 BW peak — well above analog noise.
        r_strike = r_force_max >= 0.5
        l_strike = l_force_max >= 0.5
        if side_pref == "BOTH":
            mask = "both_missing"
        elif side_pref == "R":
            mask = "r_missing"
        elif side_pref == "L":
            mask = "l_missing"
        elif v3d_empty:
            mask = "grf_empty"
        elif r_strike and not l_strike:
            mask = "r_strike_only"
        elif l_strike and not r_strike:
            mask = "l_strike_only"

        # V3D values used for writing: zero out the missing sides
        if v3d_cols is None:
            # Fabricate zero V3D so .mot writes cleanly
            v3d_t = np.arange(n_p) / p_rate
            v3d_cols = {n: np.zeros(n_p) for n in ("R_AP","R_VERT","R_ML","L_AP","L_VERT","L_ML")}
        # Apply side mask
        if mask in ("r_missing", "both_missing"):
            for n in ("R_AP","R_VERT","R_ML"):
                v3d_cols[n] = np.zeros_like(v3d_cols[n])
        if mask in ("l_missing", "both_missing"):
            for n in ("L_AP","L_VERT","L_ML"):
                v3d_cols[n] = np.zeros_like(v3d_cols[n])

        # Build .mot
        mot_path = GRF_DIR / f"{stem}.mot"
        build_grf_mot(
            trial_name=stem,
            mass_kg=mass,
            out_mot=mot_path,
            p_rate=p_rate,
            cop_r_mm=cop_r,
            cop_l_mm=cop_l,
            v3d_t=v3d_t,
            v3d_R={"AP": v3d_cols["R_AP"], "VERT": v3d_cols["R_VERT"], "ML": v3d_cols["R_ML"]},
            v3d_L={"AP": v3d_cols["L_AP"], "VERT": v3d_cols["L_VERT"], "ML": v3d_cols["L_ML"]},
        )
        # Build XML
        xml_path = GRF_DIR / f"{stem}_external_loads.xml"
        build_external_loads_xml(
            trial_name=stem,
            mot_rel_path=f"{stem}.mot",
            out_xml=xml_path,
        )

        # Presence vector at point rate (150 Hz)
        t_pt = np.arange(n_p) / p_rate
        v3d_R_VERT_at_pt = linear_resample(v3d_t, v3d_cols["R_VERT"], t_pt) * mass
        v3d_L_VERT_at_pt = linear_resample(v3d_t, v3d_cols["L_VERT"], t_pt) * mass
        r_present = (np.abs(v3d_R_VERT_at_pt) > 20).astype(int)
        l_present = (np.abs(v3d_L_VERT_at_pt) > 20).astype(int)
        PRESENCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRESENCE_DIR / f"{stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "r_grf_present", "l_grf_present"])
            for i in range(n_p):
                w.writerow([i, f"{t_pt[i]:.4f}", int(r_present[i]), int(l_present[i])])

        # Quality gates
        peak_r = float(np.nanmax(np.abs(v3d_R_VERT_at_pt))) if v3d_R_VERT_at_pt.size else 0.0
        peak_l = float(np.nanmax(np.abs(v3d_L_VERT_at_pt))) if v3d_L_VERT_at_pt.size else 0.0

        # Per-stance-window mean duration (not cumulative across multiple stances in the trial).
        def mean_stance_window_dur(signal, threshold=20):
            mask = np.abs(signal) > threshold
            runs = []; in_run = False; start = 0
            for i in range(len(mask)):
                if mask[i] and not in_run: in_run = True; start = i
                elif not mask[i] and in_run:
                    in_run = False
                    if i - start > 0.05 * p_rate:   # discard runs < 50 ms (edge noise)
                        runs.append(i - start)
            if in_run and (len(mask) - start) > 0.05 * p_rate:
                runs.append(len(mask) - start)
            return float(np.mean(runs)) / p_rate if runs else 0.0
        stance_r = mean_stance_window_dur(v3d_R_VERT_at_pt)
        stance_l = mean_stance_window_dur(v3d_L_VERT_at_pt)

        row["peak_fz_r_N"] = f"{peak_r:.1f}"
        row["peak_fz_l_N"] = f"{peak_l:.1f}"
        row["peak_fz_r_ratio_BW"] = f"{peak_r/BW:.3f}"
        row["peak_fz_l_ratio_BW"] = f"{peak_l/BW:.3f}"
        row["stance_r_s"] = f"{stance_r:.3f}"
        row["stance_l_s"] = f"{stance_l:.3f}"
        row["n_frames_pt"] = n_p
        row["duration_s"] = f"{n_p/p_rate:.3f}"
        row["status"] = "OK"

        # Per-side anomaly: only check the side(s) that actually had a strike.
        # Skip the side when the data is genuinely missing or wasn't recorded.
        anomalies = []
        r_should_check = mask in ("pass", "r_strike_only")
        l_should_check = mask in ("pass", "l_strike_only")
        if r_should_check:
            if not (0.5 * BW <= peak_r <= 1.8 * BW):
                anomalies.append(f"r_fz_{peak_r/BW:.2f}BW")
            if not (0.3 <= stance_r <= 1.2):
                anomalies.append(f"r_stance_{stance_r:.2f}s")
        if l_should_check:
            if not (0.5 * BW <= peak_l <= 1.8 * BW):
                anomalies.append(f"l_fz_{peak_l/BW:.2f}BW")
            if not (0.3 <= stance_l <= 1.2):
                anomalies.append(f"l_stance_{stance_l:.2f}s")
        row["anomaly"] = ";".join(anomalies) if anomalies else ""
    except Exception as e:
        row["status"] = "FAIL_EXCEPTION"
        row["error_msg"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        mask = "excluded"
    return row, mask


def main():
    GRF_DIR.mkdir(parents=True, exist_ok=True)
    PRESENCE_DIR.mkdir(parents=True, exist_ok=True)

    trials = list_walking_trcs()
    print(f"Found {len(trials)} walking TRCs to process.")

    # Reset logs
    fields = ["trial","subject","med","mass_kg",
              "peak_fz_r_N","peak_fz_l_N","peak_fz_r_ratio_BW","peak_fz_l_ratio_BW",
              "stance_r_s","stance_l_s","n_frames_pt","duration_s",
              "anomaly","status","error_msg"]
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    mask_map = {}
    n_total = len(trials); n_done = 0; n_fail = 0
    t0 = time.time()
    for stem in trials:
        row, mask = process_one(stem, ZIP_PATH)
        mask_map[stem] = mask
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, "") for k in fields})
        n_done += 1
        if row["status"].startswith("FAIL"):
            n_fail += 1
        if n_done % 100 == 0 or n_done == n_total:
            elapsed = time.time() - t0
            print(f"  [{n_done:>4}/{n_total}]  status={row['status']:<10}  mask={mask:<13}  trial={stem}  elapsed={elapsed:.1f}s")

    with open(MASK_PATH, "w", encoding="utf-8") as f:
        json.dump(mask_map, f, indent=2, sort_keys=True)

    # Summary
    from collections import Counter
    mask_counts = Counter(mask_map.values())
    print(f"\nTotal: {n_total}")
    print(f"FAIL: {n_fail}")
    print("Mask category breakdown:")
    for k, v in mask_counts.most_common():
        print(f"  {k:<16} {v}")
    # Quality gate: anomaly rate
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid_masks = {"pass", "r_strike_only", "l_strike_only"}
    n_valid_trials = sum(1 for r in rows if mask_map.get(r["trial"]) in valid_masks)
    n_anomaly = sum(1 for r in rows if mask_map.get(r["trial"]) in valid_masks and r["anomaly"])
    rate = (n_anomaly / n_valid_trials) if n_valid_trials else 0
    print(f"\nAnomaly rate among 'pass'+single-side-strike trials: {n_anomaly}/{n_valid_trials} = {rate*100:.1f}%")
    if rate > 0.20:
        print("  ⚠ anomaly rate > 20% — stop and investigate")
        return 2
    print(f"\nLog:      {LOG_PATH}")
    print(f"Mask:     {MASK_PATH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
