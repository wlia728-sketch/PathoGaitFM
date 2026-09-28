"""Build healthy-reference artifacts from the TRAINING 'normal' cohort, for harmonizing
the Fukuchi healthy external set (sagittal-axis/polarity decision + moment stance-peak validity).

Outputs:
  _train_normal_angle_ref.npz : keys '0','3','6','9','12','15' (54-ch sagittal kin idx) ->
       101-pt cohort-mean angle waveform from training 'normal' subjects (raw deg).
  _train_normal_peak_p5.json  : per sagittal-moment 54-ch ch (18,21,24,25,26,27) the 5th-pct of
       per-cycle STANCE peak |moment| across training normal -> validity threshold.
  _train_normal_grf_ref.npz   : keys '36','39' (R/L vertical GRF 54-ch) -> 101-pt cohort-mean,
       used only as an OPTIONAL shape sanity (NOT used to set polarity).

IRON RULE: uses ONLY training data. Nothing from Fukuchi touches these stats.

CONVENTION. These waveforms are written in the RAW cohort convention of data/cohorts_raw/normal,
that is, before COHORT_FLIP_54CH_RAW is applied. Channels 12 and 15 (R/L sagittal ankle) are in
the training 'normal' flip list [12, 15, 16, 35, 38], so they are stored with the sign OPPOSITE
to the one the model is trained on. The other four stored channels (0, 3, 6, 9) are not in that
list and are stored in the training sign. Any site harmonised against these references therefore
reaches the model in the raw convention and must carry [12, 15] in COHORT_FLIP_54CH_RAW. The
entries ext_healthy, fukuchi and ext_pd exist for exactly this reason. Do NOT apply the flip here
without simultaneously removing those entries and re-harmonising every site built from these
files, since the two corrections are alternatives and applying both would double-invert.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]            # fukuchi_healthy -> external -> data -> the package root
NORMAL_DIR = ROOT / "data" / "cohorts_raw" / "normal"

SAG6_54 = [0, 3, 6, 9, 12, 15]            # R/L hip flex, knee flex, ankle dorsi (sagittal/X in my convention)
SAGMOM_54 = [18, 21, 24, 25, 26, 27]      # R/L hip, knee, ankle sagittal moment
VGRF_54 = [36, 39]                        # R/L vertical GRF
STANCE = slice(0, 61)


def main():
    files = [f for f in sorted(glob.glob(str(NORMAL_DIR / "*.npy"))) if not f.endswith("_mask.npy")]
    angle_stacks = {ch: [] for ch in SAG6_54}
    grf_stacks = {ch: [] for ch in VGRF_54}
    mom_peaks = {ch: [] for ch in SAGMOM_54}
    n_subj = 0
    for f in files:
        raw = np.load(f)
        if raw.ndim != 3 or raw.shape[2] != 54:
            continue
        mpath = f.replace(".npy", "_mask.npy")
        mask = np.load(mpath) if Path(mpath).exists() else np.ones((raw.shape[0], 54), bool)
        n_subj += 1
        for ch in SAG6_54:
            col = raw[:, :, ch]; fin = np.isfinite(col).all(axis=1)
            if fin.any():
                angle_stacks[ch].append(np.nanmean(col[fin], axis=0))
        for ch in VGRF_54:
            col = raw[:, :, ch]; fin = np.isfinite(col).all(axis=1)
            if fin.any():
                grf_stacks[ch].append(np.nanmean(col[fin], axis=0))
        for ch in SAGMOM_54:
            for ci in range(raw.shape[0]):
                if ch < mask.shape[1] and not mask[ci, ch]:
                    continue
                seg = raw[ci, STANCE, ch]
                if np.isfinite(seg).any():
                    mom_peaks[ch].append(float(np.nanmax(np.abs(seg))))

    angle_ref = {str(ch): (np.nanmean(np.stack(v), axis=0).astype(np.float32) if v else np.full(101, np.nan, np.float32))
                 for ch, v in angle_stacks.items()}
    grf_ref = {str(ch): (np.nanmean(np.stack(v), axis=0).astype(np.float32) if v else np.full(101, np.nan, np.float32))
               for ch, v in grf_stacks.items()}
    np.savez(HERE / "_train_normal_angle_ref.npz", **angle_ref)
    np.savez(HERE / "_train_normal_grf_ref.npz", **grf_ref)
    peak_p5 = {str(ch): (round(float(np.percentile(mom_peaks[ch], 5)), 4) if mom_peaks[ch] else 0.0) for ch in SAGMOM_54}
    (HERE / "_train_normal_peak_p5.json").write_text(json.dumps(peak_p5, indent=2))

    summary = {"n_normal_subjects": n_subj, "peak_p5": peak_p5,
               "angle_ref_range": {k: [round(float(np.nanmin(v)), 1), round(float(np.nanmax(v)), 1)] for k, v in angle_ref.items()},
               "vgrf_ref_range": {k: [round(float(np.nanmin(v)), 2), round(float(np.nanmax(v)), 2)] for k, v in grf_ref.items()}}
    (HERE / "_train_healthy_ref_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
