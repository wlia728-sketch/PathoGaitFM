"""AHRS orientation from RAW IMU accel+gyro -> OpenSense _orientations.sto quaternions.

Leakage-clean: orientation is computed by a Madgwick filter from the IMU SIGNALS ONLY
(accel + gyro, NO magnetometer, NO mocap-referenced orientation). This is exactly what a
deployed wearable produces onboard, so the deployability claim holds.

Pipeline:
  1. Per IMU frame, normalise units (accel m/s^2, gyro rad/s), estimate sample rate.
  2. Madgwick(accel, gyro) -> per-sample quaternion (w,x,y,z) in the IMU/sensor world frame.
  3. Optional per-trial heading reset: rotate all frames so the pelvis IMU's initial
     heading is zeroed (short 10 m walkway => limited yaw drift, but we still de-mean the
     initial heading so the sensor world aligns predictably before OpenSense heading
     correction via base_imu_label).
  4. Resample all IMUs to a common OpenSim analysis rate and write a single .sto with one
     quaternion column per frame (OpenSense OrientationTable format).

"""
from __future__ import annotations
import numpy as np

G = 9.80665


def _normalise_units(acc, gyr):
    """Return accel in m/s^2 and gyro in rad/s, tolerant to g / deg-per-s inputs."""
    acc = np.asarray(acc, float).copy()
    gyr = np.asarray(gyr, float).copy()
    # accel: if resting magnitude ~1 -> it's in g
    finite = np.isfinite(acc).all(1)
    mag = np.linalg.norm(acc[finite], axis=1) if finite.any() else np.array([G])
    med = np.nanmedian(mag) if mag.size else G
    if 0.5 < med < 2.0:           # looks like g
        acc *= G
    # gyro: if 95th pct magnitude > ~40 -> deg/s (walking gyro peaks ~a few rad/s ~ >100 deg/s)
    gmag = np.linalg.norm(gyr[np.isfinite(gyr).all(1)], axis=1)
    if gmag.size and np.nanpercentile(gmag, 95) > 40:
        gyr = np.deg2rad(gyr)
    return acc, gyr


def madgwick_quat(acc, gyr, rate, gain=0.033):
    """Madgwick AHRS (IMU mode, accel+gyro only) -> (n,4) quat (w,x,y,z)."""
    from ahrs.filters import Madgwick
    acc, gyr = _normalise_units(acc, gyr)
    # ahrs wants gyro rad/s, acc m/s^2 (it internally normalises acc direction)
    n = len(acc)
    # fill any nan by nearest to keep the filter stable
    def _fill(a):
        a = a.copy()
        for c in range(a.shape[1]):
            col = a[:, c]; m = np.isfinite(col)
            if m.all():
                continue
            if not m.any():
                col[:] = 0.0
            else:
                col[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), col[m])
            a[:, c] = col
        return a
    acc = _fill(acc); gyr = _fill(gyr)
    mf = Madgwick(gyr=gyr, acc=acc, frequency=float(rate), gain=gain)
    Q = mf.Q  # (n,4) w,x,y,z
    if Q is None or len(Q) != n:
        # fallback: integrate as identity-start if ahrs returned unexpected shape
        Q = np.tile([1.0, 0, 0, 0], (n, 1))
    return Q


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def heading_reset(quats_by_frame, base_frame="pelvis_imu"):
    """Rotate all frames by the inverse of the base IMU's INITIAL orientation, so the
    trial starts from a consistent heading. Reduces cross-trial yaw ambiguity before
    OpenSense's own heading correction. Uses only IMU-derived quats (no leakage)."""
    if base_frame not in quats_by_frame:
        return quats_by_frame
    q0 = quats_by_frame[base_frame][0]
    q0_inv = _quat_conj(q0) / (np.dot(q0, q0) + 1e-12)
    out = {}
    for f, Q in quats_by_frame.items():
        out[f] = np.array([_quat_mul(q0_inv, q) for q in Q])
    return out


def resample_quats(Q, t_src, t_dst):
    """Resample a quaternion series by SLERP-ish nearest+normalise (short bouts; linear
    interp of quats then renormalise is adequate at these rates)."""
    from numpy import interp
    out = np.zeros((len(t_dst), 4))
    for c in range(4):
        out[:, c] = interp(t_dst, t_src, Q[:, c])
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    # sign-continuity
    for i in range(1, len(out)):
        if np.dot(out[i], out[i-1]) < 0:
            out[i] = -out[i]
    return out / nrm


def write_orientations_sto(path, frame_quats, times, frame_order):
    """Write an OpenSense OrientationTable .sto: DataRate, columns = frame names, cells =
    'w,x,y,z' quaternion strings.  This is the format IMUInverseKinematicsTool consumes."""
    n = len(times)
    rate = (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else 100.0
    with open(path, "w") as f:
        f.write("DataRate=%.6f\n" % rate)
        f.write("DataType=Quaternion\n")
        f.write("version=3\n")
        f.write("OpenSimVersion=4.5.2\n")
        f.write("endheader\n")
        f.write("time\t" + "\t".join(frame_order) + "\n")
        for i in range(n):
            cells = []
            for fr in frame_order:
                q = frame_quats[fr][i]
                cells.append("%.8f,%.8f,%.8f,%.8f" % (q[0], q[1], q[2], q[3]))
            f.write(("%.6f\t" % times[i]) + "\t".join(cells) + "\n")
    return path
