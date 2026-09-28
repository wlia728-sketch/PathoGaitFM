"""Programmatic Scale-Tool setup builder for the BMClab pipeline.

Rajagopal2015 ships **no MeasurementSet** (verified — no <Measurement> elements
in the .osim, no Setup_Scale.xml in the model directory). We define
anatomical marker pairs here; this is logged as the deliberate fallback in the
report (see step3_2_report.md).

Static trials are **T-pose** (arms abducted ~90°). The MarkerPlacer overrides
arm_add_r/l defaults so the model arms match. Lower-body coordinates default
to 0° = standing, which matches the static.

Reference for OpenSim ScaleTool API: opensim-org/opensim-core ScaleTool.h.
"""

from __future__ import annotations

import math
import os
import sys

from typing import List, Optional, Tuple

import opensim
opensim.Logger.setLevelString("Warn")  # quiet the .vtp-missing chatter


# ============================================================================
# Segment-scaling spec — pairs of MAPPED Rajagopal marker names
# Every name here must appear in marker_map.py BMCLAB_TO_RAJAGOPAL values.
# ============================================================================
#
# Convention: each Measurement scales one body segment along ALL axes equally
# using the inter-marker distance ratio (static vs model). Body names are the
# Rajagopal2015 body-set names confirmed in Step 3.1.
#
# We use single-pair measurements (1 distance per segment) for clarity.
# The pairs are chosen so distance is dominated by segment length, not by
# soft-tissue variation:
#   - pelvis_width: R.ASIS — L.ASIS  (frontal-plane width)
#   - femur: lateral knee marker — lateral pelvis marker  (close to greater trochanter)
#   - tibia: lateral malleolus — lateral knee marker
#   - foot: heel — toe-side marker
#   - torso: C7 — pelvis midpoint via CLAV (proxy)
# Upper body has only the acromion + elbow + ... but no shoulder cluster
# (Step 3.1 RAJAGOPAL_UNUSED), so torso & humerus scaling uses what we have.
SEGMENT_MEASUREMENTS: List[Tuple[str, List[str], List[Tuple[str, str]]]] = [
    # (name, body_scales [list of (body, axes)], marker_pairs [list of (m1, m2)])
    ("pelvis_width",
        [("pelvis", "X Y Z")],
        [("RASI", "LASI")]),

    ("femur_r_length",
        [("femur_r", "X Y Z")],
        [("RASI", "RLFC")]),   # hip-to-knee (proxy: ASIS, since HJC is virtual)
    ("femur_l_length",
        [("femur_l", "X Y Z")],
        [("LASI", "LLFC")]),

    ("tibia_r_length",
        [("tibia_r", "X Y Z"), ("talus_r", "X Y Z")],   # talus scales with tibia
        [("RLFC", "RLMAL")]),
    ("tibia_l_length",
        [("tibia_l", "X Y Z"), ("talus_l", "X Y Z")],
        [("LLFC", "LLMAL")]),

    ("foot_r_length",
        [("calcn_r", "X Y Z")],
        [("RCAL", "RTOE")]),    # toe marker is medial 1st-MT-mapped
    ("foot_l_length",
        [("calcn_l", "X Y Z")],
        [("LCAL", "LTOE")]),

    ("torso_length",
        [("torso", "X Y Z")],
        [("C7", "CLAV")]),

    ("humerus_r_length",
        [("humerus_r", "X Y Z")],
        [("RACR", "RLEL")]),
    ("humerus_l_length",
        [("humerus_l", "X Y Z")],
        [("LACR", "LLEL")]),
]

# Bodies that *no* measurement scales (fall back to uniform mass-based scale)
# In Rajagopal2015 these are typically the forearm/hand bones — we mapped 0 markers there.
BODIES_UNSCALED = ["radius_r", "ulna_r", "hand_r", "radius_l", "ulna_l", "hand_l"]


# ============================================================================
# T-pose arm coordinate overrides for MarkerPlacer
# ============================================================================
# Static trial = arms abducted ~90° laterally.
# Rajagopal2015 coordinate names (confirmed by inspecting model output):
#   arm_flex_r / arm_add_r / arm_rot_r  (and _l)
#   elbow_flex_r / pro_sup_r  (and _l)
#   wrist_flex_r / wrist_dev_r  (and _l)
# Coord conventions, EMPIRICALLY VERIFIED by _probe_arm_coords.py:
#   arm_add_r:  -90° puts the right elbow lateral (Z=+0.32m, right side, acromion height). T-pose.
#   arm_add_l:  -90° puts the left  elbow lateral (Z=-0.32m, left side,  acromion height). T-pose.
# Both negative — *not* mirrored. (The Rajagopal builder seems to use one signed convention
# where +arm_add adducts across the body on both sides; abduction on both sides is negative.)
TPOSE_COORD_OVERRIDES = {
    "arm_flex_r": 0.0,
    "arm_add_r":  -math.pi / 2,
    "arm_rot_r":  0.0,
    "elbow_flex_r": 0.0,
    "pro_sup_r":  0.0,
    "wrist_flex_r": 0.0,
    "wrist_dev_r":  0.0,

    "arm_flex_l": 0.0,
    "arm_add_l":  -math.pi / 2,   # was +π/2 — corrected after empirical probe
    "arm_rot_l":  0.0,
    "elbow_flex_l": 0.0,
    "pro_sup_l":  0.0,
    "wrist_flex_l": 0.0,
    "wrist_dev_l":  0.0,
}


# ============================================================================
# Lower-body neutral overrides (Step 3.3 phase-A fix)
# ============================================================================
# Force every lower-body + lumbar coordinate to anatomical neutral during
# MarkerPlacer. This decouples the model's stored marker positions from the
# subject's static-trial baseline posture — required because PD patients often
# stand stooped/crouched during calibration (SUB05 ON: pelvis_tilt = -6.4°,
# bilateral hip flex ~15°, bilateral knee flex ~16°), which without this fix
# bakes a non-anatomical pose into the model body frames and produces 25-degree
# offsets in walking-trial hip flexion vs Visual3D.
LOWERBODY_NEUTRAL_OVERRIDES = {
    "pelvis_tilt":    0.0,
    "pelvis_list":    0.0,
    "pelvis_rotation": 0.0,
    "hip_flexion_r":  0.0,  "hip_flexion_l":  0.0,
    "hip_adduction_r": 0.0, "hip_adduction_l": 0.0,
    "hip_rotation_r": 0.0,  "hip_rotation_l": 0.0,
    "knee_angle_r":   0.0,  "knee_angle_l":   0.0,
    "ankle_angle_r":  0.0,  "ankle_angle_l":  0.0,
    "subtalar_angle_r": 0.0, "subtalar_angle_l": 0.0,
    "mtp_angle_r":    0.0,  "mtp_angle_l":    0.0,
    "lumbar_extension": 0.0,
    "lumbar_bending":   0.0,
    "lumbar_rotation":  0.0,
}


# ============================================================================
# IK Task weights for MarkerPlacer (and reused later for batch IK in 3.3)
# ============================================================================
# Lower-body anatomical markers: weight 1
# Foot markers: weight 1 (mapped RTOE is medial — anatomical mismatch ~5 cm forward,
# so we down-weight slightly via 0.5 to avoid pulling the foot segment too aggressively)
# UNUSED Rajagopal markers (38 of them): weight 0 (they receive no input from BMClab)
IK_TASK_WEIGHTS = {
    # pelvis — drives the pelvis position+orientation (highest priority for ID)
    "RASI": 10, "LASI": 10, "RPSI": 10, "LPSI": 10,
    # knee — high weight on lateral, lower on medial (medial is static-only and noisier)
    "RLFC": 5, "LLFC": 5, "RMFC": 2, "LMFC": 2,
    # tibia (mid weight — Rajagopal's R_tibial_plateau is ~17 mm proximal to BMClab R.TT)
    "R_tibial_plateau": 1, "L_tibial_plateau": 1,
    # ankle
    "RLMAL": 5, "LLMAL": 5, "RMMAL": 2, "LMMAL": 2,
    # foot
    "RCAL": 3, "LCAL": 3, "RMT5": 2, "LMT5": 2,
    "RTOE": 0.5, "LTOE": 0.5,   # anatomical mismatch
    # trunk + upper body: low weight; only role is to seat the torso for inertia distribution.
    # Driving these markers tightly would fight the T-pose coordinate overrides.
    "C7": 0.5, "CLAV": 0.5,
    "RACR": 0.1, "LACR": 0.1,
    "RLEL": 0.05, "LLEL": 0.05, "RMEL": 0.05, "LMEL": 0.05,
}


# ============================================================================
def _rel(target_abs: str, anchor_dir: str) -> str:
    """Path of target relative to anchor_dir. OpenSim ScaleTool resolves
    marker/model file names relative to the *setup XML directory*, so every
    path stored in the XML must be relative to that directory."""
    return os.path.relpath(target_abs, anchor_dir)


def build_scale_setup(
    *,
    subject_id: str,
    mass_kg: float,
    height_cm: float,
    static_trc: str,
    model_file: str,
    output_model: str,
    output_setup_xml: str,
    time_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Build and write a Scale Tool setup XML for one subject.

    Parameters
    ----------
    subject_id : like "SUB01"
    mass_kg : subject mass from PDGinfo.xlsx
    height_cm : subject height (informational only)
    static_trc : absolute path to the static TRC
    model_file : absolute path to Rajagopal2015.osim
    output_model : where the scaled model should be written
    output_setup_xml : where to write this Setup XML file
    time_range : (t_start, t_end) over the static trial. If None, uses the
        first 0.5 s of the TRC (the static trial is ~1 s @ 150 Hz so this
        gives ~75 frames of averaging).
    """
    setup_dir = os.path.dirname(os.path.abspath(output_setup_xml))
    os.makedirs(setup_dir, exist_ok=True)
    # Path normalization: OpenSim resolves every file path relative to the setup XML's directory.
    model_rel        = _rel(os.path.abspath(model_file),  setup_dir)
    static_rel       = _rel(os.path.abspath(static_trc),  setup_dir)
    out_model_rel    = _rel(os.path.abspath(output_model), setup_dir)
    out_static_mot_rel = _rel(os.path.abspath(output_model.replace(".osim", "_static.mot")), setup_dir)

    st = opensim.ScaleTool()
    st.setSubjectMass(mass_kg)
    st.setSubjectHeight(height_cm)
    st.setName(subject_id)
    st.setPathToSubject(setup_dir + os.sep)  # tells the tool where relative paths anchor

    # -- GenericModelMaker -------------------------------------------------
    gen = st.getGenericModelMaker()
    gen.setModelFileName(model_rel)
    # No external markerset XML — markers come baked into Rajagopal2015.osim.

    # -- ModelScaler -------------------------------------------------------
    ms = st.getModelScaler()
    ms.setApply(True)
    ms.setPreserveMassDist(True)   # keep relative segment masses, scale total to mass_kg
    # scaling order: do measurement-based first, then any remaining bodies get mass-based
    order = opensim.ArrayStr()
    order.append("measurements")
    order.append("manualScale")
    ms.setScalingOrder(order)

    # Build MeasurementSet
    mset = ms.getMeasurementSet()
    mset.clearAndDestroy()
    for meas_name, body_scales, marker_pairs in SEGMENT_MEASUREMENTS:
        meas = opensim.Measurement()
        meas.setName(meas_name)
        # body scale axes
        bsset = meas.getBodyScaleSet()
        for body, axes in body_scales:
            bs = opensim.BodyScale()
            bs.setName(body)
            ax_arr = opensim.ArrayStr()
            for ax in axes.split():
                ax_arr.append(ax)
            bs.setAxisNames(ax_arr)
            bsset.cloneAndAppend(bs)
        # marker pairs
        mpset = meas.getMarkerPairSet()
        for m1, m2 in marker_pairs:
            mp = opensim.MarkerPair()
            mp.setMarkerName(0, m1)
            mp.setMarkerName(1, m2)
            mpset.cloneAndAppend(mp)
        mset.cloneAndAppend(meas)

    # Hand-off the static TRC for measurement averaging
    ms.setMarkerFileName(static_rel)

    # Time range: 0.0 .. 0.5 s by default. The static is 1.0 s so this gives
    # enough averaging without including any settling at the very start.
    if time_range is None:
        time_range = (0.0, 0.5)
    tr = opensim.ArrayDouble()
    tr.append(time_range[0])
    tr.append(time_range[1])
    ms.setTimeRange(tr)

    # ModelScaler manual ScaleSet: leave default (empty) — for any bodies not
    # covered by measurements, OpenSim falls back to mass-based uniform scale.

    # ms.setOutputScaleFileName(<...>)  -- skip; we don't need an intermediate.
    intermediate_model_rel = _rel(os.path.abspath(output_model.replace(".osim", "_ms.osim")), setup_dir)
    ms.setOutputModelFileName(intermediate_model_rel)

    # -- MarkerPlacer ------------------------------------------------------
    mp = st.getMarkerPlacer()
    mp.setApply(True)
    mp.setMarkerFileName(static_rel)
    mp.setTimeRange(tr)
    mp.setOutputModelFileName(out_model_rel)
    mp.setOutputMotionFileName(out_static_mot_rel)
    mp.setMaxMarkerMovement(-1)   # don't enforce a global movement limit

    # IK task set
    tset = mp.getIKTaskSet()
    tset.clearAndDestroy()
    for marker, weight in IK_TASK_WEIGHTS.items():
        t = opensim.IKMarkerTask()
        t.setName(marker)
        t.setApply(True)
        t.setWeight(weight)
        tset.cloneAndAppend(t)
    # T-pose arm coordinate overrides — these are CoordinateTasks (not marker tasks)
    for coord_name, value_rad in TPOSE_COORD_OVERRIDES.items():
        t = opensim.IKCoordinateTask()
        t.setName(coord_name)
        t.setApply(True)
        t.setValueType(opensim.IKCoordinateTask.ManualValue)
        t.setValue(value_rad)
        # Use weight 2.0 — strong enough to seed the T-pose, weak enough to let MarkerPlacer
        # refine to actual marker positions. Higher weights (e.g. 10) lock the arm coords and
        # force higher marker error on the elbow/acromion. Lower weights (e.g. 0.5) let the
        # solver drift away from T-pose into a local min where arms cross the body.
        t.setWeight(2.0)
        tset.cloneAndAppend(t)
    # Lower-body neutral overrides (Step 3.3 phase-A fix). Higher weight than the
    # arm overrides because we strictly REQUIRE the model to be in upright neutral —
    # marker placement on bodies should reflect a standard anatomical reference, not
    # the subject's stooped static pose.
    for coord_name, value_rad in LOWERBODY_NEUTRAL_OVERRIDES.items():
        t = opensim.IKCoordinateTask()
        t.setName(coord_name)
        t.setApply(True)
        t.setValueType(opensim.IKCoordinateTask.ManualValue)
        t.setValue(value_rad)
        t.setWeight(10.0)
        tset.cloneAndAppend(t)

    # -- Save XML and return -----------------------------------------------
    st.printToXML(output_setup_xml)


def run_scale_for_subject(setup_xml: str) -> bool:
    """Load the setup XML and run the Scale Tool. Returns True on success."""
    tool = opensim.ScaleTool(setup_xml)
    ok = tool.run()
    return bool(ok)


if __name__ == "__main__":
    # CLI smoke test
    if len(sys.argv) < 2:
        print("Usage: s32_scale_setup_builder.py <subject_id>   # e.g. SUB01", file=sys.stderr)
        sys.exit(1)
    subj = sys.argv[1]
    print(f"smoke test for {subj}")
