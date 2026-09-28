"""Batch driver for Step 4 gait-cycle extraction.

Iterates all 885 trials in `_grf_mask.json`. Skips `excluded` trials.
For each trial, calls `s4_cycle_extraction.extract_trial_cycles`, saves per-cycle
.npy under data/prepare/bmclab/work/cycles/{SUB}/{med}_walk_{idx}/cycle_{i}_{side}.npy,
and accumulates metadata into a parquet table + outlier CSV.

Note: Trimmed_SUB01_off_walk_12b and Trimmed_SUB01_off_walk_14b are the only
two Trimmed_ trials, and their non-Trimmed bases are NOT in the dataset
(verified Step-1 finding). No de-dup needed.
"""
from __future__ import annotations

import os
import sys
import time
import csv
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4_cycle_extraction import (
    extract_trial_cycles, ANGLE_CHANNELS, MOMENT_CHANNELS, GRF_CHANNELS,
)

PROJ = Path(__file__).resolve().parents[3]
MASK_PATH = PROJ / "data/prepare/bmclab/work/grf/_grf_mask.json"
CYCLES_DIR = PROJ / "data/prepare/bmclab/work/cycles"
META_PARQUET = CYCLES_DIR / "_cycle_metadata.parquet"
OUTLIERS_CSV = CYCLES_DIR / "_cycle_outliers.csv"
LOG_CSV = CYCLES_DIR / "_batch_log.csv"
PDG = pd.read_excel(PROJ / "data/prepare/bmclab/raw/PDGinfo.xlsx", sheet_name="PDGinfo", header=0)

def mass_of(sid): return float(PDG[PDG["ID"] == sid]["Weight (kg)"].iloc[0])


def parse_trial(stem: str):
    """Return (subject_id, medication, walk_idx_str)."""
    if stem.startswith("Trimmed_"):
        parts = stem.split("_")
        sub, med = parts[1], parts[2]
        walk_idx = "_".join(parts[3:])  # walk_12b etc.
    else:
        parts = stem.split("_")
        sub, med = parts[0], parts[1]
        walk_idx = "_".join(parts[2:])
    return sub, med, walk_idx


# All channels saved to .npy in this fixed order
ALL_CHANNELS = ANGLE_CHANNELS + MOMENT_CHANNELS + GRF_CHANNELS


def cycle_to_array(cyc_data: dict) -> np.ndarray:
    """Stack channels in fixed order → (n_channels, 101) float32."""
    arr = np.stack([cyc_data[c] for c in ALL_CHANNELS], axis=0)
    return arr.astype(np.float32)


def main():
    CYCLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MASK_PATH, "r") as f:
        mask_map = json.load(f)
    trials = sorted([t for t, m in mask_map.items() if m != "excluded"])
    print(f"Trials to process: {len(trials)} (excluded skipped)")

    n_trials_done = 0; n_trials_failed = 0
    all_meta = []
    log_rows = []
    t0 = time.time()
    for stem in trials:
        sub, med, walk_idx = parse_trial(stem)
        try:
            mass = mass_of(sub)
        except (KeyError, IndexError):
            n_trials_failed += 1
            log_rows.append({"trial": stem, "status": "FAIL", "n_cycles": 0,
                             "reason": "mass lookup failed"})
            continue
        try:
            cycles, status = extract_trial_cycles(stem, mass, sub, med)
        except Exception as e:
            n_trials_failed += 1
            log_rows.append({"trial": stem, "status": "FAIL", "n_cycles": 0,
                             "reason": f"exception: {e}"})
            continue
        if status != "ok":
            log_rows.append({"trial": stem, "status": "SKIP", "n_cycles": 0,
                             "reason": status})
            continue
        # Save per-cycle .npy
        out_dir = CYCLES_DIR / sub / f"{med}_{walk_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        per_side_idx = {"r": 0, "l": 0}
        for c in cycles:
            side = c["meta"]["side"]
            idx = per_side_idx[side]
            per_side_idx[side] += 1
            fname = out_dir / f"cycle_{idx}_{side}.npy"
            np.save(fname, cycle_to_array(c["data"]))
            c["meta"]["cycle_npy_path"] = str(fname.relative_to(CYCLES_DIR))
            c["meta"]["cycle_idx_in_trial_side"] = idx
            all_meta.append(c["meta"])
        n_trials_done += 1
        log_rows.append({"trial": stem, "status": "OK", "n_cycles": len(cycles),
                         "reason": ""})
        if n_trials_done % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_trials_done:>4}/{len(trials)}] OK so far, elapsed={elapsed:.1f}s, cycles={len(all_meta)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  trials OK:     {n_trials_done}")
    print(f"  trials FAILED: {n_trials_failed}")
    print(f"  total cycles:  {len(all_meta)}")

    # Metadata parquet
    df = pd.DataFrame(all_meta)
    df.to_parquet(META_PARQUET, index=False)
    print(f"\nWrote {META_PARQUET}  ({len(df)} cycles)")

    # Outliers CSV
    out_df = df[df["is_outlier"] == True].copy()
    out_df.to_csv(OUTLIERS_CSV, index=False)
    print(f"Wrote {OUTLIERS_CSV}  ({len(out_df)} outliers)")

    # Log
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "status", "n_cycles", "reason"])
        w.writeheader()
        for r in log_rows: w.writerow(r)
    print(f"Wrote {LOG_CSV}")

    # Summary
    print("\n=== Step 4 quality gates ===")
    n_total = len(df)
    n_r = (df["side"] == "r").sum()
    n_l = (df["side"] == "l").sum()
    n_out = int(df["is_outlier"].sum())
    n_inlier = n_total - n_out
    pres_high = (df["grf_presence_in_cycle"] > 0.7).sum()
    median_dur = float(df["cycle_duration_sec"].median())
    print(f"  total cycles:     {n_total}  (R={n_r}, L={n_l}, asym={abs(n_r-n_l)/max(1,n_total):.1%})")
    print(f"  inlier cycles:    {n_inlier}  ({n_inlier/n_total:.1%})")
    print(f"  outlier cycles:   {n_out}  ({n_out/n_total:.1%})")
    print(f"  median dur (s):   {median_dur:.3f}")
    print(f"  cycles w/ presence>0.7:  {pres_high}  ({pres_high/n_total:.1%})")
    # Outlier breakdown
    if n_out:
        reason_counter = Counter()
        for r in out_df["outlier_reasons"]:
            for tag in r.split(";"):
                t_clean = re.split(r'[_\d]', tag, maxsplit=1)[0]
                reason_counter[t_clean] += 1
        print(f"  outlier reason tally: {dict(reason_counter)}")


if __name__ == "__main__":
    main()
