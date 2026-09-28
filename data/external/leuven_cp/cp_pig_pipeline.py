"""Step 2 — CP (Plug-in-Gait) -> Rajagopal2015 scale + Inverse Kinematics, in WSL OpenSim.

Runs in the WSL `bmclab_opensim` conda env (OpenSim 4.5.2). Reuses the project's
nimblephysics Rajagopal2015.osim base model (same one the BMClab pipeline scaled),
but with a Plug-in-Gait -> Rajagopal marker map and a single-lateral (no medial
knee/ankle) measurement set appropriate to the CP c3d.

PiG markers present in the CP trials (verified): pelvis RASI/LASI/(R/L)PSI/SACR,
thigh wand RTHI/LTHI, lateral knee RKNE/LKNE, shank wand RTIB/LTIB, lateral ankle
RANK/LANK, heel RHEE/LHEE, toe RTOE/LTOE, trunk C7/CLAV/T10/STRN, shoulders RSHO/LSHO.
No medial knee/ankle markers (PiG uses KAD/knee-width) -> we scale the shank from the
lateral knee->lateral malleolus distance and the femur from ASIS->lateral knee, exactly
as the BMClab fallback does, and rely on the static trial + MarkerPlacer for axis seating.
This is standard for sagittal-plane PiG IK, which is the only thing we consume (hip/knee/
ankle flexion + pelvis), so medial markers are not required.

Outputs (in models/ and ik/):
  models/<SUBJ>.osim          scaled model
  models/<SUBJ>_ik.osim       force-stripped IK model
  models/<SUBJ>_static.mot    marker-placer static pose
  ik/<TRIAL>_ik.mot           per-trial joint angles

Usage (inside WSL env):
  python cp_pig_pipeline.py scale <SUBJ>          # e.g. DiCP2 (needs trc/static/<SUBJ>.trc)
  python cp_pig_pipeline.py ik <SUBJ> <TRIAL>     # e.g. DiCP2 DiCP2  (gait trc)
  python cp_pig_pipeline.py smoke <SUBJ>          # scale + IK all that subject's gait trials
ASCII-only stdout.
"""
from __future__ import annotations
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "SIMTK_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import glob
from pathlib import Path

import opensim
opensim.Logger.setLevelString("Error")  # quiet vtp + info chatter

HERE = Path(__file__).resolve().parent
TRC = HERE / "trc"
MODELS = HERE / "models"
IK = HERE / "ik"
# Rajagopal2015 base model that ships with nimblephysics. Set RAJAGOPAL_OSIM to
# override if it lives elsewhere in your install.
def _find_rajagopal() -> Path:
    env = os.environ.get("RAJAGOPAL_OSIM")
    if env:
        return Path(env)
    try:
        import nimblephysics
        return (Path(nimblephysics.__file__).resolve().parent
                / "models" / "rajagopal_data" / "Rajagopal2015.osim")
    except Exception:
        return Path("nimblephysics/models/rajagopal_data/Rajagopal2015.osim")


BASE_MODEL = _find_rajagopal()

# group lookup for a subject id
def group_of(subj: str) -> str:
    s = subj.lower()
    if s.startswith("hecp"):
        return "hecp"
    if s.startswith("dicp"):
        return "dicp"
    return "td"

# ---- Plug-in-Gait -> Rajagopal2015 marker map (CP trials) -------------------
# Only markers that exist in BOTH the CP trc and the Rajagopal MarkerSet are used.
PIG_TO_RAJ = {
    "RASI": "RASI", "LASI": "LASI",
    "RPSI": "RPSI", "LPSI": "LPSI",
    "RKNE": "RLFC", "LKNE": "LLFC",       # lateral knee -> lateral femoral condyle
    "RANK": "RLMAL", "LANK": "LLMAL",     # lateral ankle -> lateral malleolus
    "RHEE": "RCAL", "LHEE": "LCAL",       # heel -> calcaneus
    "RTOE": "RTOE", "LTOE": "LTOE",
    "C7": "C7", "CLAV": "CLAV",
    "RSHO": "RACR", "LSHO": "LACR",
    # thigh/shank wands (RTHI/RTIB) and SACR/T10/STRN have no clean single Rajagopal
    # marker -> dropped (not needed for sagittal IK).
}

# Scale measurements using ONLY markers in PIG_TO_RAJ values (lateral chain, no medial).
SEGMENT_MEASUREMENTS = [
    ("pelvis_width", [("pelvis", "X Y Z")], [("RASI", "LASI")]),
    ("femur_r", [("femur_r", "X Y Z")], [("RASI", "RLFC")]),
    ("femur_l", [("femur_l", "X Y Z")], [("LASI", "LLFC")]),
    ("tibia_r", [("tibia_r", "X Y Z"), ("talus_r", "X Y Z")], [("RLFC", "RLMAL")]),
    ("tibia_l", [("tibia_l", "X Y Z"), ("talus_l", "X Y Z")], [("LLFC", "LLMAL")]),
    ("foot_r", [("calcn_r", "X Y Z")], [("RCAL", "RTOE")]),
    ("foot_l", [("calcn_l", "X Y Z")], [("LCAL", "LTOE")]),
    ("torso", [("torso", "X Y Z")], [("C7", "CLAV")]),
]

# IK marker weights (Rajagopal names). Lateral lower-body anchors high; trunk low.
IK_WEIGHTS = {
    "RASI": 10, "LASI": 10, "RPSI": 10, "LPSI": 10,
    "RLFC": 5, "LLFC": 5,
    "RLMAL": 5, "LLMAL": 5,
    "RCAL": 3, "LCAL": 3,
    "RTOE": 1, "LTOE": 1,
    "C7": 0.5, "CLAV": 0.5, "RACR": 0.1, "LACR": 0.1,
}

# Children stand arms-down (NOT T-pose). Lower-body neutral seating only.
LOWERBODY_NEUTRAL = {
    "pelvis_tilt": 0.0, "pelvis_list": 0.0, "pelvis_rotation": 0.0,
    "hip_flexion_r": 0.0, "hip_flexion_l": 0.0,
    "hip_adduction_r": 0.0, "hip_adduction_l": 0.0,
    "hip_rotation_r": 0.0, "hip_rotation_l": 0.0,
    "knee_angle_r": 0.0, "knee_angle_l": 0.0,
    "ankle_angle_r": 0.0, "ankle_angle_l": 0.0,
    "subtalar_angle_r": 0.0, "subtalar_angle_l": 0.0,
    "mtp_angle_r": 0.0, "mtp_angle_l": 0.0,
    "lumbar_extension": 0.0, "lumbar_bending": 0.0, "lumbar_rotation": 0.0,
}


def _load_subject_info():
    """Weights and heights from the de-identified masses.json, keyed by upper id.

    These two anthropometric values are all the model scaling needs. The site's
    participant_information.xlsx additionally carried age, sex and GMFCS for 14 minors, a quasi-identifier
    combination, so only mass and height are read from it; see PROVENANCE.md.
    """
    import json
    mj = HERE / "masses.json"
    if not mj.exists():
        raise FileNotFoundError(
            "masses.json not found. Build it from participant_information.xlsx in the SimTK "
            "cp-child-gait download as {ID: {mass_kg: float, height_cm: float}} and place it beside "
            "this script. See README, Dataset layout.")
    table = json.load(open(mj))
    return {str(k).strip().upper(): {"weight": float(v["mass_kg"]), "height": float(v["height_cm"])}
            for k, v in table.items()}


def _rename_trc_markers(in_trc: str, out_trc: str, require_complete: bool = True):
    """Rewrite a CP trc keeping only PIG_TO_RAJ markers, renamed to Rajagopal names.

    OpenSim's TimeSeriesTableVec3 reader (used inside IK) rejects empty/missing cells,
    so when require_complete=True we keep only the LONGEST CONTIGUOUS run of frames in
    which every kept marker is present (no gaps). For overground CP gait this run is the
    on-plate window where all lower-body markers are visible — exactly the cycles that
    also carry valid vGRF. The static trial (require_complete=False not needed; statics
    are full-coverage) is handled the same way safely.
    """
    lines = Path(in_trc).read_text().splitlines()
    name_row = lines[3].split("\t")
    src_names, col_of = [], {}
    for j in range(2, len(name_row), 3):
        nm = name_row[j].strip()
        if nm:
            src_names.append(nm)
            col_of[nm] = j
    keep = [(s, PIG_TO_RAJ[s]) for s in src_names if s in PIG_TO_RAJ]
    if not keep:
        raise RuntimeError(f"no mappable PiG markers in {in_trc}")
    meta = lines[2].split("\t")
    rate = meta[0]

    # parse data rows -> list of (frame, time, [vals per kept marker])
    data = []
    for ln in lines[6:]:
        if not ln.strip():
            continue
        cols = ln.split("\t")
        vals = []
        complete = True
        for src, _ in keep:
            c = col_of[src]
            xyz = [cols[c] if c < len(cols) else "",
                   cols[c + 1] if c + 1 < len(cols) else "",
                   cols[c + 2] if c + 2 < len(cols) else ""]
            if any(v.strip() == "" for v in xyz):
                complete = False
            vals.append(xyz)
        data.append((cols[0], cols[1], vals, complete))

    if require_complete:
        # longest contiguous run of complete frames
        best_lo = best_len = 0; cur_lo = None
        for i, (_, _, _, comp) in enumerate(data):
            if comp:
                if cur_lo is None:
                    cur_lo = i
                if i - cur_lo + 1 > best_len:
                    best_lo, best_len = cur_lo, i - cur_lo + 1
            else:
                cur_lo = None
        if best_len < 30:
            raise RuntimeError(f"no usable complete-marker window in {in_trc} "
                               f"(longest run {best_len} frames)")
        data = data[best_lo:best_lo + best_len]

    out = []
    out.append(f"PathFileType\t4\t(X/Y/Z)\t{Path(out_trc).name}")
    out.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
    out.append(f"{rate}\t{rate}\t{len(data)}\t{len(keep)}\tm\t{rate}\t1\t{len(data)}")
    hdr = ["Frame#", "Time"]
    for _, raj in keep:
        hdr += [raj, "", ""]
    out.append("\t".join(hdr))
    sub = ["", ""]
    for k in range(len(keep)):
        sub += [f"X{k+1}", f"Y{k+1}", f"Z{k+1}"]
    out.append("\t".join(sub))
    out.append("")
    for fno, (frame, time, vals, _) in enumerate(data, 1):
        row = [str(fno), time]
        for xyz in vals:
            row += xyz
        out.append("\t".join(row))
    Path(out_trc).parent.mkdir(parents=True, exist_ok=True)
    Path(out_trc).write_text("\n".join(out) + "\n")
    return [r for _, r in keep]


def trc_time_range(trc_path: str):
    md = opensim.MarkerData(trc_path)
    return float(md.getStartFrameTime()), float(md.getLastFrameTime())


def scale_subject(subj: str):
    info = _load_subject_info()
    sid_key = subj.upper().replace("HECP", "HECP").replace("DICP", "DICP")
    meta = info.get(sid_key, info.get(subj.upper(), {}))
    mass = meta.get("weight", 30.0)
    height_cm = meta.get("height", 130.0)

    MODELS.mkdir(parents=True, exist_ok=True)
    raw_static = TRC / "static" / f"{subj}.trc"
    if not raw_static.is_file():
        raise FileNotFoundError(f"no static trc for {subj}: {raw_static}")
    raj_static = MODELS / f"{subj}_static_raj.trc"
    kept = _rename_trc_markers(str(raw_static), str(raj_static))
    print(f"[{subj}] static markers mapped to Rajagopal: {kept}")

    out_model = MODELS / f"{subj}.osim"
    setup_xml = MODELS / f"_{subj}_scale_setup.xml"
    t0, t1 = trc_time_range(str(raj_static))
    tr_lo, tr_hi = t0, min(t0 + 0.5, t1)

    st = opensim.ScaleTool()
    st.setSubjectMass(mass)
    # ScaleTool records subject height in MILLIMETRES, masses.json carries centimetres. The field
    # is informational and never reaches ModelScaler, so correcting it leaves every published IK
    # result unchanged. It only stops the setup XML from recording a tenfold-wrong height.
    st.setSubjectHeight(height_cm * 10.0)
    st.setName(subj)
    # NOTE: do NOT call setPathToSubject — we use absolute paths everywhere, and
    # setPathToSubject would prepend models/ to those absolute paths (double path bug).

    # ScaleTool resolves ModelScaler/MarkerPlacer marker-file paths relative to its
    # internal subject dir, which prepends MODELS/. So we store paths as plain basenames
    # (relative to MODELS, where the setup XML and the renamed trc both live) and the
    # GenericModelMaker model path as absolute (it is resolved differently).
    raj_static_name = raj_static.name
    st.getGenericModelMaker().setModelFileName(str(BASE_MODEL))

    ms = st.getModelScaler()
    ms.setApply(True)
    ms.setPreserveMassDist(True)
    order = opensim.ArrayStr(); order.append("measurements"); order.append("manualScale")
    ms.setScalingOrder(order)
    mset = ms.getMeasurementSet(); mset.clearAndDestroy()
    for name, bodies, pairs in SEGMENT_MEASUREMENTS:
        meas = opensim.Measurement(); meas.setName(name)
        bss = meas.getBodyScaleSet()
        for body, axes in bodies:
            bs = opensim.BodyScale(); bs.setName(body)
            aa = opensim.ArrayStr()
            for ax in axes.split():
                aa.append(ax)
            bs.setAxisNames(aa); bss.cloneAndAppend(bs)
        mps = meas.getMarkerPairSet()
        for m1, m2 in pairs:
            mp = opensim.MarkerPair(); mp.setMarkerName(0, m1); mp.setMarkerName(1, m2)
            mps.cloneAndAppend(mp)
        mset.cloneAndAppend(meas)
    ms.setMarkerFileName(raj_static_name)
    tr = opensim.ArrayDouble(); tr.append(tr_lo); tr.append(tr_hi)
    ms.setTimeRange(tr)
    ms.setOutputModelFileName(f"{subj}_ms.osim")

    mp = st.getMarkerPlacer()
    mp.setApply(True)
    mp.setMarkerFileName(raj_static_name)
    mp.setTimeRange(tr)
    mp.setOutputModelFileName(f"{subj}.osim")
    mp.setOutputMotionFileName(f"{subj}_static.mot")
    mp.setMaxMarkerMovement(-1)
    tset = mp.getIKTaskSet(); tset.clearAndDestroy()
    for marker, w in IK_WEIGHTS.items():
        t = opensim.IKMarkerTask(); t.setName(marker); t.setApply(True); t.setWeight(w)
        tset.cloneAndAppend(t)
    for coord, val in LOWERBODY_NEUTRAL.items():
        t = opensim.IKCoordinateTask(); t.setName(coord); t.setApply(True)
        t.setValueType(opensim.IKCoordinateTask.ManualValue); t.setValue(val); t.setWeight(10.0)
        tset.cloneAndAppend(t)

    st.printToXML(str(setup_xml))
    # Run with CWD = MODELS so ScaleTool resolves the basename marker/model paths correctly.
    _cwd = os.getcwd()
    try:
        os.chdir(str(MODELS))
        ok = opensim.ScaleTool(setup_xml.name).run()
    finally:
        os.chdir(_cwd)
    print(f"[{subj}] scale {'OK' if ok else 'FAIL'} -> {out_model.name}")
    if not ok or not out_model.is_file():
        return False
    # strip forces for fast IK
    m = opensim.Model(str(out_model)); m.updForceSet().clearAndDestroy(); m.initSystem()
    m.printToXML(str(MODELS / f"{subj}_ik.osim"))
    return True


def run_ik(subj: str, trial: str):
    IK.mkdir(parents=True, exist_ok=True)
    grp = group_of(subj)
    raw_trc = TRC / "gait" / grp / f"{trial}.trc"
    if not raw_trc.is_file():
        raise FileNotFoundError(f"no gait trc: {raw_trc}")
    raj_trc = IK / f"{trial}_raj.trc"
    kept = _rename_trc_markers(str(raw_trc), str(raj_trc))
    ik_model = MODELS / f"{subj}_ik.osim"
    out_mot = IK / f"{trial}_ik.mot"
    setup_xml = IK / f"_{trial}_ik_setup.xml"

    ik_tool = opensim.InverseKinematicsTool()
    ik_tool.setName(trial)
    ik_tool.set_model_file(str(ik_model))
    ik_tool.set_marker_file(str(raj_trc))
    ik_tool.set_output_motion_file(str(out_mot))
    ik_tool.set_accuracy(1e-5)
    ik_tool.set_constraint_weight(20.0)
    t0, t1 = trc_time_range(str(raj_trc))
    ik_tool.set_time_range(0, t0); ik_tool.set_time_range(1, t1)
    tset = ik_tool.get_IKTaskSet(); tset.clearAndDestroy()
    # weight only the markers present
    for marker, w in IK_WEIGHTS.items():
        if marker in kept:
            t = opensim.IKMarkerTask(); t.setName(marker); t.setApply(True); t.setWeight(w)
            tset.cloneAndAppend(t)
    ik_tool.printToXML(str(setup_xml))
    ok = bool(opensim.InverseKinematicsTool(str(setup_xml)).run())
    print(f"[{subj}/{trial}] IK {'OK' if ok else 'FAIL'} -> {out_mot.name}")
    return ok and out_mot.is_file()


def smoke(subj: str):
    if not scale_subject(subj):
        print("scale failed; stop."); return
    grp = group_of(subj)
    trials = sorted(glob.glob(str(TRC / "gait" / grp / f"{subj}*.trc")))
    trials = [t for t in trials if "_raj" not in t]
    print(f"[{subj}] {len(trials)} gait trials")
    for t in trials:
        run_ik(subj, Path(t).stem)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "scale":
        scale_subject(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "ik":
        run_ik(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "smoke":
        smoke(sys.argv[2])
    else:
        print(__doc__); sys.exit(1)
