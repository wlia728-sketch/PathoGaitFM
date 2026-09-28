"""Issue B — flip mask=False on ch 14 (R_Ankle_Angle_Z) and ch 17 (L_Ankle_Angle_Z)
across all 48 bmclab_pd/*_mask.npy files. Data values (currently identically zero) are
not modified.

Reason: BMClab Rajagopal IK cannot recover ankle inversion/eversion (subtalar) because
(1) Step 3.2's LOWERBODY_NEUTRAL_OVERRIDES locks subtalar to 0 during MarkerPlacer,
and (2) BMClab Leardini 2007 marker set has no distal medial foot markers to drive
subtalar dynamics in walking IK. Subtalar is identically 0 across all 3915 cycles —
mask=False reflects this missing information truthfully (same convention as EMG).
"""
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
OUT = PROJ / "data/cohorts_raw/bmclab_pd"

SUBTALAR_CHANNELS = [14, 17]


def main():
    mask_files = sorted(OUT.glob("*_mask.npy"))
    print(f"Flipping subtalar mask=False on {len(mask_files)} files...")
    n_changed = 0
    for m_path in mask_files:
        m = np.load(m_path)
        if m.shape[1] != 54: continue
        before = int(m[:, SUBTALAR_CHANNELS].sum())
        m[:, SUBTALAR_CHANNELS] = False
        after = int(m[:, SUBTALAR_CHANNELS].sum())
        np.save(m_path, m)
        n_changed += before - after
    print(f"Done. Flipped {n_changed} mask positions to False (ch 14 + 17 combined).")


if __name__ == "__main__":
    main()
