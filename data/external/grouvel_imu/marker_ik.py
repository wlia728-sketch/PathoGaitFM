"""Marker inverse kinematics (POSITIVE CONTROL) for the Grouvel dataset.

Writes a TRC from the C3D markers, scales Rajagopal_2015 to the subject with a static
frame, runs marker-IK -> the SAME 6 sagittal angles the IMU branch produces. On healthy
overground walking this MUST land ~0.85-0.95 vGRF (cf OpenCap marker 0.878); if far below,
the input convention/marker map is wrong -> debug before trusting the IMU number.

Runs inside the WSL conda env `bmclab_opensim` (opensim 4.5.2).

Grouvel marker set: 69 markers. The pelvis/thigh/shank/foot anatomical markers are mapped
to Rajagopal markers via GROUVEL_TO_RAJAGOPAL (filled after inspecting one real C3D's
POINT labels). Anchors used at weight 1.0; secondary at 0.1; trunk/arm ~0.01.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OSENSE = HERE / "opensense"
# Base Rajagopal_2015 ships WITHOUT a MarkerSet. The bmclab training IK model
# (SUB01_off_ik.osim) is the SAME Rajagopal 2016 convention WITH 28 markers already
# placed at the training marker names our GROUVEL_TO_RAJAGOPAL map targets -> use it as
# the marker-IK model so the positive control matches the training convention exactly.
# Subject scale differs but sagittal-angle SHAPE (what the vGRF PCC needs) is scale-robust.
BASE_MODEL = OSENSE / "Rajagopal_2015.osim"
TRAIN_IK_MODEL = HERE.parents[2] / "data" / "prepare" / "bmclab" / "work" / "scaled_models" / "SUB01_off_ik.osim"

# Grouvel(CGM/Plug-in-Gait-like) label -> Rajagopal model marker name.
# FILLED after inspecting one real C3D (imu_c3d_io print). Common Plug-in-Gait labels shown;
# adjust to the dataset's actual POINT labels. None => skip.
GROUVEL_TO_RAJAGOPAL = {
    # pelvis
    "RASI": "RASI", "LASI": "LASI", "RPSI": "RPSI", "LPSI": "LPSI", "SACR": None,
    # thigh / knee
    "RKNE": "RLFC", "LKNE": "LLFC", "RTHI": "RTH3", "LTHI": "LTH3",
    "RKNM": "RMFC", "LKNM": "LMFC",
    # shank / ankle
    "RANK": "RLMAL", "LANK": "LLMAL", "RMED": "RMMAL", "LMED": "LMMAL",
    "RTIB": "RSK3", "LTIB": "LSK3",
    # foot
    "RHEE": "RCAL", "LHEE": "LCAL", "RTOE": "RTOE", "LTOE": "LTOE",
    "RMT5": "RMT5", "LMT5": "LMT5",
}

WEIGHTS = {  # Rajagopal marker -> IK weight
    "RASI": 1.0, "LASI": 1.0, "RPSI": 1.0, "LPSI": 1.0,
    "RLFC": 1.0, "LLFC": 1.0, "RLMAL": 1.0, "LLMAL": 1.0, "RCAL": 1.0, "LCAL": 1.0,
    "RMFC": 0.1, "LMFC": 0.1, "RMMAL": 0.1, "LMMAL": 0.1, "RTOE": 0.1, "LTOE": 0.1,
    "RMT5": 0.1, "LMT5": 0.1, "RTH3": 0.1, "LTH3": 0.1, "RSK3": 0.1, "LSK3": 0.1,
}

IK_COORDS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
             "ankle_angle_r", "ankle_angle_l",
             "pelvis_tilt", "pelvis_list", "pelvis_rotation"]


def write_trc(path, markers, times, rate):
    """markers: {rajagopal_name:(n,3) m}. Writes an OpenSim TRC (mm)."""
    names = list(markers.keys())
    n = len(times)
    with open(path, "w") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t%s\n" % Path(path).name)
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("%.1f\t%.1f\t%d\t%d\tmm\t%.1f\t1\t%d\n" % (rate, rate, n, len(names), rate, n))
        hdr = "Frame#\tTime\t" + "\t".join("%s\t\t" % nm for nm in names) + "\n"
        f.write(hdr)
        sub = "\t\t" + "\t".join("X%d\tY%d\tZ%d" % (i+1, i+1, i+1) for i in range(len(names))) + "\n"
        f.write(sub)
        for i in range(n):
            row = ["%d" % (i+1), "%.6f" % times[i]]
            for nm in names:
                x, y, z = markers[nm][i] * 1000.0
                row += ["%.5f" % x, "%.5f" % y, "%.5f" % z]
            f.write("\t".join(row) + "\n")
    return path


# Marker pairs (model-marker A, model-marker B) whose static inter-marker distance scales
# the corresponding body segment. Both markers must exist in model + static TRC.
SCALE_PAIRS = {
    "pelvis": [("RASI", "LASI"), ("RPSI", "LPSI")],
    "femur_r": [("RLFC", "RASI")], "femur_l": [("LLFC", "LASI")],
    "tibia_r": [("RLMAL", "RLFC")], "tibia_l": [("LLMAL", "LLFC")],
    "calcn_r": [("RCAL", "RTOE")], "calcn_l": [("LCAL", "LTOE")],
}


def scale_model(trc_path, static_time, out_model, mass_kg=75.0):
    """PER-SUBJECT scale the training Rajagopal IK model from the static-trial marker span,
    matching the per-subject calibration the IMU branch gets (IMUPlacer static calibration).
    This removes the marker-branch handicap that made the generic-model marker IK a weaker
    positive control than IMU-IK.

    NOT OpenSim ModelScaler. Per-segment factors are computed here as the median ratio of
    subject to model inter-marker distance over SCALE_PAIRS, then applied with Body.scale().
    Factors outside (0.5, 2.0) are treated as implausible and skipped.

    Scaling failure RAISES. Until 2026-07-30 this fell back to the unscaled model and only
    printed a line, so a run could quietly produce a different and weaker positive control.
    The shipped Grouvel marker numbers predate that change and no run log survives, so if a
    re-run raises here, the original run took the fallback for that subject."""
    import opensim as osim
    import numpy as np
    osim.Logger.setLevelString("Error")
    # Fail closed rather than fall back. Rajagopal_2015.osim carries no MarkerSet, so marker IK on it
    # cannot reproduce this control; the silent fallback that used to sit here would have produced a
    # different and weaker positive control without saying so.
    if not Path(TRAIN_IK_MODEL).exists():
        raise FileNotFoundError(
            "%s not found. Build it by running the OpenSim ScaleTool on the SUB01 static trial of "
            "the BMClab release (doi:10.6084/m9.figshare.14896881) with the generic Rajagopal 2015 "
            "model and this site's 28 marker names. Rajagopal_2015.osim carries no MarkerSet and is "
            "not a substitute." % TRAIN_IK_MODEL)
    src = str(TRAIN_IK_MODEL)
    try:
        model = osim.Model(src); state = model.initSystem()
        # per-segment scale factor = (subject static inter-marker dist) / (model marker dist)
        md = osim.MarkerData(str(trc_path))
        _mn = md.getMarkerNames(); trc_names = [_mn.get(i) for i in range(_mn.getSize())]
        # average static marker positions (metres)
        md.averageFrames(0.0, md.getStartFrameTime(), md.getStartFrameTime() + 1e3)
        # read model marker positions in ground at default pose
        mset = model.getMarkerSet()
        modelpos = {}
        for i in range(mset.getSize()):
            mk = mset.get(i); modelpos[mk.getName()] = np.array(
                [mk.getLocationInGround(state).get(k) for k in range(3)])
        # subject static positions: use averaged frame
        avg = {}
        # average over all static frames from the raw TRC values
        arr = osim.MarkerData(str(trc_path))
        for i, nm in enumerate(trc_names):
            pts = []
            for f in range(arr.getNumFrames()):
                m = arr.getFrame(f).getMarker(i)
                pts.append([m.get(0), m.get(1), m.get(2)])
            avg[nm] = np.nanmean(np.array(pts), axis=0) / 1000.0  # mm->m
        def dist(d, a, b):
            return float(np.linalg.norm(d[a] - d[b])) if a in d and b in d else None
        # per-body scale from the SCALE_PAIRS
        BODY = {"pelvis": ["pelvis"], "femur_r": ["femur_r"], "femur_l": ["femur_l"],
                "tibia_r": ["tibia_r"], "tibia_l": ["tibia_l"], "calcn_r": ["calcn_r", "toes_r"],
                "calcn_l": ["calcn_l", "toes_l"]}
        scales = {}
        for seg, pairs in SCALE_PAIRS.items():
            rs = []
            for (a, b) in pairs:
                ds = dist(avg, a, b); dm = dist(modelpos, a, b)
                if ds and dm and dm > 1e-6:
                    rs.append(ds / dm)
            if rs:
                scales[seg] = float(np.median(rs))
        if not scales:
            raise RuntimeError("no scalable segments")
        # apply uniform per-body scale
        applied = 0
        for seg, bodies in BODY.items():
            s = scales.get(seg)
            if s is None or not (0.5 < s < 2.0):
                continue
            v = osim.Vec3(s, s, s)
            for bn in bodies:
                try:
                    body = model.getBodySet().get(bn); body.scale(v, True); applied += 1
                except Exception:
                    pass
        if applied == 0:
            raise RuntimeError("no bodies scaled")
        model.finalizeConnections()
        model.printToXML(str(out_model))
        return out_model
    except Exception as e:
        raise RuntimeError(
            "per-subject scaling failed for %s (%s: %s). Refusing the silent unscaled fallback, "
            "which would report a different and weaker marker positive control under the same "
            "name." % (trc_path, type(e).__name__, e))


def run_marker_ik(model_path, trc_path, results_mot, t0, t1):
    import opensim as osim
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "SIMTK_NUM_THREADS"):
        os.environ[k] = "1"
    ik = osim.InverseKinematicsTool()
    ik.set_model_file(str(model_path))
    ik.set_marker_file(str(trc_path))
    ik.set_output_motion_file(str(results_mot))
    ik.set_accuracy(1e-5)
    ik.set_constraint_weight(20.0)
    ik.set_time_range(0, float(t0))
    ik.set_time_range(1, float(t1))
    m = osim.Model(str(model_path)); m.initSystem()
    model_markers = [m.getMarkerSet().get(i).getName() for i in range(m.getMarkerSet().getSize())]
    ts = ik.get_IKTaskSet(); ts.clearAndDestroy()
    for nm in model_markers:
        task = osim.IKMarkerTask(); task.setName(nm)
        if nm in WEIGHTS:
            task.setApply(True); task.setWeight(WEIGHTS[nm])
        else:
            task.setApply(False); task.setWeight(0.0)
        ts.cloneAndAppend(task)
    ok = bool(ik.run())
    return results_mot if ok and Path(results_mot).exists() else None


def read_ik_mot(mot_path):
    lines = Path(mot_path).read_text().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[hi + 1].split()
    rows = [ln.split() for ln in lines[hi + 2:] if ln.strip()]
    dat = np.array([[float(x) for x in r] for r in rows if len(r) == len(cols)])
    t = dat[:, 0]
    return t, {c: dat[:, cols.index(c)] for c in IK_COORDS if c in cols}
