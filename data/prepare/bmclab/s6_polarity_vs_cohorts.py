"""Cohort polarity comparison via per-time-percent mean and peak.

For each cohort + channel, compute the mean waveform across all cycles
(only mask=True, finite values), then report:
  - min/max of the mean waveform
  - argmax / argmin (gait-cycle phase % where peak occurs)

This is what reveals POLARITY (sign of peak) — not the all-values mean.
"""
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
ED = PROJ / "data" / "cohorts_raw"
OUT_DIR = PROJ / "data/prepare/bmclab/work/step6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ED / "channel_names.txt") as f:
    NAMES = [line.split("\t")[1].strip() for line in f if line.strip()]

COHORTS = {
    "bmclab_pd":   ED / "bmclab_pd",
    "cp":          ED / "cp",
    "normal":      ED / "normal",
    "vdk_healthy": ED / "vdk_healthy",
    "vdk_stroke":  ED / "vdk_stroke",
    "addbio":      ED / "addbio",
}

# Load precomputed mean waveforms
wf_npz = np.load(OUT_DIR / "cohort_mean_waveforms.npz")

# Critical R-side sagittal channels for polarity check
CRIT_CH = [0, 6, 12, 18, 24, 26, 32]
crit_names = [NAMES[c] for c in CRIT_CH]

print(f"\n{'channel':>22} {'cohort':>12} {'min':>10} {'argmin%':>8} {'max':>10} {'argmax%':>8}")
print("-" * 80)
for ch in CRIT_CH:
    for coh in COHORTS:
        if coh not in wf_npz: continue
        wf = wf_npz[coh][ch]  # (101,)
        if not np.isfinite(wf).any():
            print(f"{NAMES[ch]:>22} {coh:>12}   (all-NaN)")
            continue
        finite_mask = np.isfinite(wf)
        mn = float(wf[finite_mask].min())
        mx = float(wf[finite_mask].max())
        amn = int(np.nanargmin(wf))
        amx = int(np.nanargmax(wf))
        print(f"{NAMES[ch]:>22} {coh:>12} {mn:>+10.3f} {amn:>7d}% {mx:>+10.3f} {amx:>7d}%")

# Also: look at the full waveform for the 7 critical channels (sample 5 timepoints)
print("\n=== Mean waveform values at gait-cycle phase 0% / 25% / 50% / 75% / 100% ===")
print("  (101 points → indices 0, 25, 50, 75, 100)")
for ch in CRIT_CH:
    print(f"\n  ch {ch} {NAMES[ch]}")
    print(f"    {'cohort':>12} {'@0%':>9} {'@25%':>9} {'@50%':>9} {'@75%':>9} {'@100%':>9}")
    for coh in COHORTS:
        if coh not in wf_npz: continue
        wf = wf_npz[coh][ch]
        if not np.isfinite(wf).any(): continue
        vals = [wf[i] for i in (0, 25, 50, 75, 100)]
        vals_str = "  ".join(f"{v:>+7.3f}" if np.isfinite(v) else "    NaN" for v in vals)
        print(f"    {coh:>12} {vals_str}")
