"""V4 data-preparation constants shared by the transform pipeline.

Provides the fixed channel map and polarity-flip / study-exclusion tables the
V3 transforms build on:
  - KEEP_CHANNELS_54TO40: the 54 -> 40 channel reduction (drops POWER and the
    knee/ankle non-sagittal angle channels).
  - COHORT_FLIP_54CH / ADDBIO_FLIP_54CH: per-source sign-flip tables aligning
    every source to the vdk_healthy reference convention.
  - EXCLUDED_STUDIES + filter_addbio_files: AddBio studies dropped at file
    discovery (small-n / noisy / convention-different).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict


# Channel reduction: 54 -> 40 (drop ch7,8,10,11,13,14,16,17,28-33)
KEEP_CHANNELS_54TO40 = [
    0, 1, 2, 3, 4, 5,           # Hip XYZ R/L
    6, 9,                        # Knee X R/L (drop Y/Z = 7,8,10,11)
    12, 15,                      # Ankle X R/L (drop Y/Z = 13,14,16,17)
    18, 19, 20, 21, 22, 23,     # Hip Mom XYZ R/L
    24, 25, 26, 27,             # Knee + Ankle Mom X R/L
    34, 35, 36, 37, 38, 39,     # GRF XYZ R/L
    40, 41, 42, 43, 44, 45, 46, 47,  # EMG (8)
    48, 49, 50, 51, 52, 53,     # Pelvis XYZ R/L
]
assert len(KEEP_CHANNELS_54TO40) == 40

# 54-ch indices to flip per cohort. From v3 Step 1.B / V4.2 investigation.
COHORT_FLIP_54CH: Dict[str, list] = {
    "cp":          [12, 15, 35, 38],   # Ankle X R/L + GRF Y R/L
    "normal":      [12, 15, 35, 38],
    "vdk_healthy": [],
    "vdk_stroke":  [],
    "bmclab_pd":   [26, 50, 53],
}

# 54-ch indices to flip per AddBio study. Built from V4 systematic audit
# (dataset/v4_systematic_audit.json), 23 flipped (PCC < -0.5 vs vdk_healthy)
# pairs identified across 6 channel groups:
#   - ch12, 15: R/L_Ankle_X (54-ch)
#   - ch20, 23: R/L_Hip_Mom_Z (54-ch)
#   - ch48, 51: R/L_Pelvis_X (54-ch)
# Applied BEFORE per-study P1/P99 normalize.
ADDBIO_FLIP_54CH: Dict[str, list] = {
    "Camargo2021":   [12, 15, 20, 23],
    "Carter2023":    [20, 23, 48, 51],
    "Moore2015":     [48, 51],
    "Tan2021":       [20, 23, 48, 51],
    "Tan2022":       [12, 15, 20],
    "Wang2023":      [20, 23, 48, 51],
    "vanderZee2022": [12, 15],
}

EXCLUDED_STUDIES = (
    "Falisse2017",   # 6 cyc, too small
    "Tiziana2019",   # 22 cyc, GRF_Z buggy in raw
    "Han2023",       # 69 cyc, high per-file variance
    "Fregly2012",    # 156 cyc, mixed convention
    "Hamner2013",    # 144 cyc, small-n statistical noise
)


def filter_addbio_files(files):
    """Drop file paths whose study name is in EXCLUDED_STUDIES."""
    keep = []
    for f in files:
        name = Path(f).name
        if any(name.startswith(s + "_") for s in EXCLUDED_STUDIES):
            continue
        keep.append(f)
    return keep
