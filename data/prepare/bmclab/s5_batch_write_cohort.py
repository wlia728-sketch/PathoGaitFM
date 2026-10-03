"""Step 5 batch driver — per (subject, medication_state) output.

Reads Step 4 cycle metadata + per-cycle .npy, builds 54-channel
(n_cycles_subj_med, 101, 54) + mask (n_cycles_subj_med, 54) + meta CSV,
writes to `data/cohorts_raw/bmclab_pd/{SUB}_{med}.{npy,mask.npy,meta.csv}`.
"""
from __future__ import annotations

import os
import sys

import json
import time

from pathlib import Path


import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s5_channel_mapping import PELVIS_REPRESENTATION, build_54ch
from s5_joint_power import compute_powers

PROJ = Path(__file__).resolve().parents[3]
META_PARQUET = PROJ / "data/prepare/bmclab/work/cycles/_cycle_metadata.parquet"
CYCLES_DIR = PROJ / "data/prepare/bmclab/work/cycles"
GRF_MASK_JSON = PROJ / "data/prepare/bmclab/work/grf/_grf_mask.json"
OUT_DIR = PROJ / "data/cohorts_raw/bmclab_pd"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(META_PARQUET)
    with open(GRF_MASK_JSON, "r") as f:
        mask_map = json.load(f)
    meta["mask_cat"] = meta["trial"].map(mask_map).fillna("excluded")
    print(f"Loaded {len(meta)} cycle metadata rows. Building 54-channel mapping...")

    # Group by (subject_id, medication_state)
    groups = sorted(meta.groupby(["subject_id", "medication_state"]).groups.keys())
    print(f"Subject-state groups: {len(groups)}")

    t0 = time.time()
    n_total = 0
    n_subj_state_written = 0

    for (sub, med) in groups:
        sub_meta = meta[(meta["subject_id"] == sub) & (meta["medication_state"] == med)].sort_values(["trial", "side", "cycle_idx_in_trial_side"]).reset_index(drop=True)
        if len(sub_meta) == 0: continue
        n_cyc = len(sub_meta)
        data_arr = np.full((n_cyc, 101, 54), np.nan, dtype=np.float32)
        mask_arr = np.zeros((n_cyc, 54), dtype=bool)
        meta_rows = []

        for i, row in sub_meta.iterrows():
            trial = row["trial"]
            side = row["side"]
            cyc_idx = int(row["cycle_idx_in_trial_side"])
            walk_idx = trial.split("_", 2)[2] if not trial.startswith("Trimmed_") else "_".join(trial.split("_")[3:])
            cyc_npy = CYCLES_DIR / sub / f"{med}_{walk_idx}" / f"cycle_{cyc_idx}_{side}.npy"
            if not cyc_npy.is_file():
                # Should not happen — Step 4 wrote all cycles
                meta_rows.append({
                    "cycle_idx_in_trial": cyc_idx, "group": "pd", "n_valid_channels": 0,
                    "subject_id": f"bmclab_{sub.lower()}_{med}", "trial": trial, "side": side,
                    "pelvis_representation": PELVIS_REPRESENTATION,
                    "missing_cycle": True,
                })
                continue
            cyc_data = np.load(cyc_npy)   # (34, 101)
            # Powers
            duration = float(row["cycle_duration_sec"])
            powers = compute_powers(cyc_data, duration)
            # Map to 54-channel
            d54, m54 = build_54ch(
                cyc_data, cycle_side=side,
                grf_presence=float(row["grf_presence_in_cycle"]),
                grf_mask_category=row["mask_cat"],
                computed_powers=powers,
            )
            data_arr[i] = d54
            mask_arr[i] = m54
            meta_rows.append({
                "cycle_idx_in_trial": cyc_idx,
                "group": "pd",
                "n_valid_channels": int(m54.sum()),
                "subject_id": f"bmclab_{sub.lower()}_{med}",
                "trial": trial,
                "side": side,
                "pelvis_representation": PELVIS_REPRESENTATION,
                "cycle_duration_sec": duration,
                "grf_presence": float(row["grf_presence_in_cycle"]),
                "grf_mask_category": row["mask_cat"],
                "is_outlier": bool(row.get("is_outlier", False)),
            })
        # Write
        npy_name = f"{sub}_{med}"
        np.save(OUT_DIR / f"{npy_name}.npy", data_arr)
        np.save(OUT_DIR / f"{npy_name}_mask.npy", mask_arr)
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(OUT_DIR / f"{npy_name}_meta.csv", index=False)
        n_total += n_cyc
        n_subj_state_written += 1
        if n_subj_state_written % 10 == 0 or n_subj_state_written == len(groups):
            elapsed = time.time() - t0
            print(f"  [{n_subj_state_written:>3}/{len(groups)}]  {npy_name}: {n_cyc} cycles  total={n_total}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  subject-state files written: {n_subj_state_written}")
    print(f"  total cycles: {n_total}")
    print(f"  output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
