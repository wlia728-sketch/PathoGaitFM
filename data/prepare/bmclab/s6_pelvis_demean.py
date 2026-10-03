"""Pelvic_Lab angle centering (Step 6 sub-task 2).

Step 5 supplies OpenSim pelvis_tilt/list/rotation in degrees. Subtract
each angle's per-cycle mean from channels 48-53 to match the development
cohorts' centered Pelvic_Lab convention. Units remain degrees.

Modifies `data/cohorts_raw/bmclab_pd/{SUB}_{med}.npy` in place. Mask not touched.
"""
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
OUT = PROJ / "data/cohorts_raw/bmclab_pd"
PEL_CHS = list(range(48, 54))

def main():
    data_files = sorted(p for p in OUT.glob("*.npy") if not p.name.endswith("_mask.npy"))
    print(f"Harmonizing Pelvic_Lab (ch 48-53) — per-cycle demean — on {len(data_files)} subject-state files...")
    n_cyc_changed = 0
    verified_means = []
    for d_path in data_files:
        d = np.load(d_path)            # (N, 101, 54) float32
        if d.shape[1:] != (101, 54):
            print(f"  skip shape {d.shape}: {d_path.name}")
            continue
        # For ch 48-53: subtract per-cycle mean (axis=1, the time dimension)
        for ch in PEL_CHS:
            ch_data = d[:, :, ch]      # (N, 101)
            finite = np.isfinite(ch_data)
            # Compute per-cycle mean over finite timepoints only; if all-NaN cycle, leave as-is
            per_cycle_mean = np.where(finite.any(axis=1),
                                      np.where(finite, ch_data, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1),
                                      0.0)
            d[:, :, ch] = ch_data - per_cycle_mean[:, None]
        np.save(d_path, d)
        n_cyc_changed += d.shape[0]
        # Fully observed cycles only; missing angles stay unavailable.
        observed = np.isfinite(d[:, :, 48]).all(axis=1)
        if observed.any():
            verified_means.append(d[observed, :, 48].mean(axis=1))
    print(f"Done. Touched {n_cyc_changed} cycles across {len(data_files)} files.")

    if verified_means:
        means = np.concatenate(verified_means)
        print(f"Verification: ch 48 per-cycle means: min={means.min():+.6f} max={means.max():+.6f}  (should be ~0)")


if __name__ == "__main__":
    main()
