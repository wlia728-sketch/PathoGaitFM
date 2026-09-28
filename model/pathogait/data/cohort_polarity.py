"""Cohort polarity-flip table for the raw cohort data (data/cohorts_raw/{cohort}/).

The V3 transform inherits per-source sign flips from channels.py and adds this
raw-cohort table, built from a PCC audit against the raw vdk_healthy reference
(channels with PCC < -0.5 are flipped to the reference convention).
"""
from __future__ import annotations
from typing import Dict

# 54-ch indices to flip per cohort for the RAW data at data/cohorts_raw/{cohort}/.
COHORT_FLIP_54CH_RAW: Dict[str, list] = {
    "cp":          [12, 15],                          # R/L Ankle_X
    "normal":      [12, 15, 16, 35, 38],               # + R_Ankle_Y, R/L GRF_Y
    "vdk_healthy": [],                                  # reference
    "vdk_stroke":  [],                                  # already aligned
    "bmclab_pd":   [12, 15, 18, 21, 22, 32, 50, 53],   # multiple flips

    # External sites. Each is harmonised outside this file, and its own polarity decision was taken
    # against a reference cohort-mean built from the RAW extracted_data arrays, before the flips
    # above were applied. That reference is therefore one convention behind the space the model
    # trains in, and it resolves every external subject to "no flip" on the two sagittal ankle
    # channels even though the training corpus has them negated. Measured against the never-flipped
    # vdk_healthy anchor over the stance window, the three sites below agree with training on hip
    # and knee and disagree on both ankle channels (Fukuchi -0.97/-0.97, PD-Nordic -0.63/-0.61,
    # Grouvel -0.52/-0.61), which is the signature of that one missing flip and not of a cohort
    # difference. Declaring it here is where every other source's convention is already declared.
    # Scope is set by which harmoniser actually consulted that reference, not by correlation alone.
    # Only harmonize_pd.py and the Fukuchi axis-mapping and sanity-gate steps load a
    # _train_*_angle_ref.npz, so only those two sites can carry the defect. Fukuchi appears twice
    # because the vGRF and the moment evaluator name the same site differently.
    "ext_healthy": [12, 15],                            # Fukuchi, vGRF and sparse-input evaluators
    "fukuchi":     [12, 15],                            # Fukuchi, moment evaluator
    "ext_pd":      [12, 15],                            # PD-Nordic
    # ext_imu is deliberately absent. Grouvel derives its angles by OpenSim inverse kinematics from
    # markers and from inertial orientations and never reads one of these references, so the defect
    # cannot reach it. Its low agreement with the healthy anchor comes from independent cycle
    # segmentation, and a lag scan on it separates no sign hypothesis from a phase one.
    # ext_opencap is absent for the same mechanical reason, and its ankle shows no inversion.
    # ext_cp and ext_cp2 are absent because both cerebral-palsy sites already agree with training.
}
