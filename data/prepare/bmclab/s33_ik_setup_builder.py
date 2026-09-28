"""Per-trial Inverse Kinematics setup builder.

Builds the IK setup XML programmatically from a scaled model, a walking TRC,
and the marker weights of the cross-protocol decision recorded below.

The marker-weight policy:
 - Mapped Rajagopal markers listed in IK_MARKER_WEIGHTS below get those weights.
 - Any other model marker (i.e. any name in the model's MarkerSet not listed
   here) is explicitly added with weight 0.01 — never silently weight=1.
 - Unmapped Rajagopal markers (RAJAGOPAL_UNUSED) get setApply(False) — they
   would have no TRC data anyway, so applying them with any weight crashes IK.

Time range = full TRC duration (no trimming; Step 4 handles cycle extraction).
"""

from __future__ import annotations

# *** Thread-count pinning must come BEFORE `import opensim` ***
# OpenSim 4.5 / SimTK auto-threads its Jacobian against all hardware cores per process.
# When we run K workers in parallel, K × cores threads contend for K × cores slots,
# producing 5× per-trial slowdown (see Step 3.3 CPU-bound diagnosis). Pinning every
# thread library to 1 thread per process forces a single-threaded IK solver, so the
# outer multiprocessing Pool can scale linearly.
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "SIMTK_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_k] = "1"

import os
import sys
from typing import Optional

import opensim
opensim.Logger.setLevelString("Warn")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marker_map as mm

# ---- the operative Option-A weight policy -----------------------------------
IK_MARKER_WEIGHTS = {
    # Primary lower-body anchors — full weight
    "RASI": 1.0, "LASI": 1.0,
    "RPSI": 1.0, "LPSI": 1.0,
    "RLFC": 1.0, "LLFC": 1.0,
    "RLMAL": 1.0, "LLMAL": 1.0,
    "RCAL": 1.0, "LCAL": 1.0,

    # Secondary lower-body — down-weighted
    "RMFC": 0.1, "LMFC": 0.1,
    "RMMAL": 0.1, "LMMAL": 0.1,
    "RTOE": 0.1, "LTOE": 0.1,
    "RMT5": 0.1, "LMT5": 0.1,

    # Tibial plateau — soft constraint (cross-protocol anatomical mismatch)
    "R_tibial_plateau": 0.1, "L_tibial_plateau": 0.1,

    # Upper body — effectively ignored
    "RACR": 0.01, "LACR": 0.01,
    "RLEL": 0.01, "LLEL": 0.01,
    "RMEL": 0.01, "LMEL": 0.01,
    "C7": 0.01,
    "CLAV": 0.01,
}

# Default for any mapped marker not in the explicit table.
IK_DEFAULT_WEIGHT_FOR_UNSPECIFIED_MAPPED = 0.01


def mapped_target_set() -> set:
    return {v for v in mm.BMCLAB_TO_RAJAGOPAL.values() if v is not None}


def list_model_markers(model_path: str) -> list:
    m = opensim.Model(model_path)
    m.initSystem()
    return [m.getMarkerSet().get(i).getName() for i in range(m.getMarkerSet().getSize())]


def trc_time_range(trc_path: str) -> tuple:
    """Return (t_start, t_end) in seconds from the TRC header + data."""
    md = opensim.MarkerData(trc_path)
    return float(md.getStartFrameTime()), float(md.getLastFrameTime())


def _rel(target: str, anchor: str) -> str:
    return os.path.relpath(target, anchor)


def build_ik_setup(
    *,
    scaled_model: str,
    marker_trc: str,
    output_mot: str,
    output_setup_xml: str,
    marker_weights: Optional[dict] = None,
    accuracy: float = 1e-5,
    constraint_weight: float = 20.0,
) -> dict:
    """Build IK setup XML on disk + return a dict of stats (n markers with
    weight=0.01 default, time range, etc.).

    All file paths inside the saved XML are relative to the XML's directory,
    matching the Step-3.2 ScaleTool path convention.
    """
    if marker_weights is None:
        marker_weights = IK_MARKER_WEIGHTS

    setup_dir = os.path.dirname(os.path.abspath(output_setup_xml))
    os.makedirs(setup_dir, exist_ok=True)
    # InverseKinematicsTool resolves paths relative to the current working directory at
    # run() time, not the setup XML's directory (unlike ScaleTool which honours
    # setPathToSubject). Use absolute paths so it's unambiguous either way.
    model_abs = os.path.abspath(scaled_model)
    trc_abs   = os.path.abspath(marker_trc)
    mot_abs   = os.path.abspath(output_mot)

    # Enumerate model markers so we can give every one an explicit weight or apply=False
    model_markers = list_model_markers(scaled_model)
    mapped = mapped_target_set()
    n_default_weight = 0
    n_unapplied = 0
    weights_used = []

    ik_tool = opensim.InverseKinematicsTool()
    ik_tool.setName(os.path.basename(output_mot).replace("_ik.mot", ""))
    ik_tool.set_model_file(model_abs)
    ik_tool.set_marker_file(trc_abs)
    ik_tool.set_output_motion_file(mot_abs)
    ik_tool.set_accuracy(accuracy)
    ik_tool.set_constraint_weight(constraint_weight)

    # Time range — use full TRC duration
    t0, t1 = trc_time_range(marker_trc)
    ik_tool.set_time_range(0, t0)
    ik_tool.set_time_range(1, t1)

    task_set = ik_tool.get_IKTaskSet()
    task_set.clearAndDestroy()
    for marker_name in model_markers:
        task = opensim.IKMarkerTask()
        task.setName(marker_name)
        if marker_name in marker_weights:
            w = marker_weights[marker_name]
            task.setApply(True)
            task.setWeight(w)
        elif marker_name in mapped:
            # mapped but not in explicit table — apply default and count for the log
            w = IK_DEFAULT_WEIGHT_FOR_UNSPECIFIED_MAPPED
            task.setApply(True)
            task.setWeight(w)
            n_default_weight += 1
        else:
            # unmapped — never fed by BMClab. Don't apply.
            task.setApply(False)
            task.setWeight(0.0)
            n_unapplied += 1
            w = 0.0
        weights_used.append((marker_name, w, task.getApply()))
        task_set.cloneAndAppend(task)

    ik_tool.printToXML(output_setup_xml)
    return {
        "n_markers_in_model": len(model_markers),
        "n_default_weight": n_default_weight,
        "n_unapplied": n_unapplied,
        "t_start": t0,
        "t_end": t1,
        "weights_used": weights_used,
    }


def run_ik(setup_xml: str) -> bool:
    tool = opensim.InverseKinematicsTool(setup_xml)
    return bool(tool.run())


if __name__ == "__main__":
    print("Library module. Import build_ik_setup() / run_ik().", file=sys.stderr)
