"""OpenSense IMU inverse kinematics: calibrate Rajagopal_2015 with a static frame, then
run IMUInverseKinematicsTool on the walking-bout quaternions -> sagittal joint angles.

Must run inside the WSL conda env `bmclab_opensim` (opensim 4.5.2). Consumes the
_orientations.sto written by imu_ahrs.write_orientations_sto (IMU-signal-only quats).

Returns the 6 sagittal joint-angle time series in the Rajagopal convention:
  hip_flexion_r/l, knee_angle_r/l, ankle_angle_r/l  (deg), plus pelvis_tilt/list/rotation.
Same convention as bmclab training marker-IK => IDENTITY sign into the 54-ch layout.
"""
from __future__ import annotations
import os
from math import pi
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OSENSE = HERE / "opensense"
BASE_MODEL = OSENSE / "Rajagopal_2015.osim"
# The generic Rajagopal 2015 full-body model, which ships with OpenSim at
# Models/RajagopalModel/Rajagopal_2015.osim and is also on SimTK (project full_body). Copy it into
# opensense/ from your OpenSim installation.

# Space-fixed XYZ Euler from IMU sensor space to OpenSim. Default OpenSense uses (-pi/2,0,0)
# for Xsens. For Grouvel (APDM/Xsens) we start from -pi/2 and VERIFY against the marker
# branch; if the marker/IMU angle traces are flipped, this is the knob to audit first.
SENSOR_TO_OSIM = (-pi/2, 0.0, 0.0)
BASE_IMU = "pelvis_imu"
BASE_HEADING_AXIS = "z"   # audited against marker branch; -z if heading reversed

IK_COORDS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
             "ankle_angle_r", "ankle_angle_l",
             "pelvis_tilt", "pelvis_list", "pelvis_rotation"]


def calibrate_model(orientations_sto, out_model, static_time=0.0):
    """IMUPlacer: attach IMU frames to Rajagopal bodies using a (near-)static frame."""
    import opensim as osim
    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            "%s not found. Copy Rajagopal_2015.osim from your OpenSim installation "
            "(Models/RajagopalModel/) or from SimTK project full_body into that directory." % BASE_MODEL)
    imuPlacer = osim.IMUPlacer()
    imuPlacer.set_model_file(str(BASE_MODEL))
    imuPlacer.set_orientation_file_for_calibration(str(orientations_sto))
    imuPlacer.set_sensor_to_opensim_rotations(osim.Vec3(*SENSOR_TO_OSIM))
    imuPlacer.set_base_imu_label(BASE_IMU)
    imuPlacer.set_base_heading_axis(BASE_HEADING_AXIS)
    imuPlacer.run(False)
    model = imuPlacer.getCalibratedModel()
    model.printToXML(str(out_model))
    return out_model


def run_imu_ik(calibrated_model, orientations_sto, results_dir, t0, t1):
    """IMUInverseKinematicsTool over [t0,t1] -> writes <results_dir>/*_orientations_ik.mot."""
    import opensim as osim
    os.makedirs(results_dir, exist_ok=True)
    imuIK = osim.IMUInverseKinematicsTool()
    imuIK.set_model_file(str(calibrated_model))
    imuIK.set_orientations_file(str(orientations_sto))
    imuIK.set_sensor_to_opensim_rotations(osim.Vec3(*SENSOR_TO_OSIM))
    imuIK.set_results_directory(str(results_dir))
    imuIK.set_time_range(0, float(t0))
    imuIK.set_time_range(1, float(t1))
    imuIK.run(False)
    # find the produced .mot
    mots = sorted(Path(results_dir).glob("*.mot"))
    return mots[-1] if mots else None


def read_ik_mot(mot_path):
    """Parse an OpenSim IK .mot -> (time, {coord: array}) for IK_COORDS present."""
    lines = Path(mot_path).read_text().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[hi + 1].split()
    rows = [ln.split() for ln in lines[hi + 2:] if ln.strip()]
    dat = np.array([[float(x) for x in r] for r in rows if len(r) == len(cols)])
    t = dat[:, 0]
    out = {}
    for c in IK_COORDS:
        if c in cols:
            out[c] = dat[:, cols.index(c)]
    return t, out
