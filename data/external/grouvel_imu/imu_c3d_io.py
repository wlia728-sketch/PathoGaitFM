"""Data reader for the Grouvel et al. 2023 (Sci Data 10:180) dataset.

REAL FORMAT (verified 2026-07-09 on P01_S01_Gait_01):
  - RAW_DATA/<trial>.c3d : 77 markers @ 100 Hz (Plug-in-Gait labels) + 3 force plates
    @ 1000 Hz (Fx1/Fy1/Fz1/Mx1.. per plate) with FORCE_PLATFORM CORNERS/ORIGIN/CHANNEL.
    NO IMU channels in the C3D.
  - SYNC_DATA/<trial>.csv : wide table @ 100 Hz (405 rows for Gait_01), frame-aligned with
    the C3D points. 231 marker cols (<MARKER>_x/y/z) + 24 accel (P6_<SEG>_acc_x/y/z) +
    24 gyro (P6_<SEG>_gyro_x/y/z). 8 IMU segments: LF/RF foot, LS/RS shank, LT/RT thigh,
    SA sacrum, TR trunk. NO magnetometer (accel+gyro only -> leakage-clean AHRS native).

So: marker branch + force GT come from the C3D; IMU branch accel+gyro come from the paired
SYNC CSV. Both are 100 Hz and frame-synced.

ANTI-LEAKAGE: the IMU stream is raw accel+gyro only. The dataset ships NO mocap-referenced
IMU orientation, so there is nothing to leak; we compute orientation downstream via AHRS.

Run standalone to dump labels of a c3d (+ its sync csv):
    python imu_c3d_io.py raw/P01_S01/P01_S01_Gait_01.c3d
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np

# OpenSim Rajagopal IMU frame names (7 lower-limb frames; torso optional/unused for our 6 angles)
OSIM_IMU_FRAMES = ["pelvis_imu", "femur_r_imu", "femur_l_imu",
                   "tibia_r_imu", "tibia_l_imu", "calcn_r_imu", "calcn_l_imu"]

# Grouvel IMU segment token (in P6_<SEG>_acc_*) -> OpenSim frame.
IMU_SEG_TO_FRAME = {
    "SA": "pelvis_imu",     # sacrum
    "RT": "femur_r_imu", "LT": "femur_l_imu",   # thigh
    "RS": "tibia_r_imu", "LS": "tibia_l_imu",   # shank
    "RF": "calcn_r_imu", "LF": "calcn_l_imu",   # foot
    "TR": "torso_imu",      # trunk (not used for lower-limb IK)
}

# Grouvel marker (Plug-in-Gait) -> Rajagopal_2015 model marker name. None => skip.
GROUVEL_TO_RAJAGOPAL = {
    "RASI": "RASI", "LASI": "LASI", "RPSI": "RPSI", "LPSI": "LPSI",
    "RKNE": "RLFC", "LKNE": "LLFC",       # lateral femoral condyle
    "RANK": "RLMAL", "LANK": "LLMAL",     # lateral malleolus
    "RHEE": "RCAL", "LHEE": "LCAL",       # heel -> calcaneus
    "RTOE": "RTOE", "LTOE": "LTOE",
    "RTHI": "RTH3", "LTHI": "LTH3",       # thigh wand
    "RTIB": "RSK3", "LTIB": "LSK3",       # shank wand
}


def read_c3d(path):
    import ezc3d
    c = ezc3d.c3d(str(path))
    p = c["parameters"]
    point_rate = float(np.ravel(p["POINT"]["RATE"]["value"])[0])
    analog_rate = float(np.ravel(p["ANALOG"]["RATE"]["value"])[0])
    plabels = list(p["POINT"]["LABELS"]["value"])
    alabels = list(p["ANALOG"]["LABELS"]["value"])
    pts = c["data"]["points"]      # (4, n_pt, n_frames)
    ana = c["data"]["analogs"][0]  # (n_an_ch, n_an_frames)
    n_frames = pts.shape[2]
    t = np.arange(n_frames) / point_rate
    unit = "mm"
    try:
        u = p["POINT"]["UNITS"]["value"]
        if u: unit = str(u[0]).lower()
    except Exception:
        pass
    scale = 0.001 if unit.startswith("mm") else 1.0
    markers = {}
    for i, lab in enumerate(plabels):
        if lab.startswith("*"):
            continue
        markers[lab] = (pts[:3, i, :].T * scale).astype(np.float64)
    return {
        "raw": c, "params": p, "plabels": plabels, "alabels": alabels,
        "markers": markers, "analogs": ana, "analog_rate": analog_rate,
        "point_rate": point_rate, "times": t, "n_frames": n_frames, "point_unit": unit,
    }


def sync_csv_path(c3d_path):
    """RAW_DATA/<trial>.c3d -> SYNC_DATA/<trial>.csv (same subject dir here, flat)."""
    p = Path(c3d_path)
    cand = p.with_suffix(".csv")
    if cand.exists():
        return cand
    # try a SYNC_DATA sibling if the flat one is absent
    alt = p.parent / "SYNC_DATA" / (p.stem + ".csv")
    return alt if alt.exists() else cand


def read_sync_imu(csv_path):
    """Read SYNC_DATA CSV -> {osim_frame: {'acc':(n,3), 'gyr':(n,3)}} + n_rows.

    ANTI-LEAKAGE: only P6_<SEG>_acc_* and P6_<SEG>_gyro_* columns are read.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None, 0
    with open(csv_path, "r", encoding="latin-1") as f:
        hdr = f.readline().rstrip("\n\r").split(",")
    # map columns
    acc_idx = {}  # frame -> {axis: col}
    gyr_idx = {}
    for j, col in enumerate(hdr):
        m = re.match(r"P\d+_([A-Z]{2})_(acc|gyro)_([xyz])$", col)
        if not m:
            continue
        seg, kind, ax = m.group(1), m.group(2), m.group(3)
        frame = IMU_SEG_TO_FRAME.get(seg)
        if frame is None:
            continue
        d = (acc_idx if kind == "acc" else gyr_idx).setdefault(frame, {})
        d[ax] = j
    # load numeric body
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, encoding="latin-1")
    if data.ndim == 1:
        data = data[None, :]
    n = data.shape[0]
    out = {}
    for frame in set(list(acc_idx.keys()) + list(gyr_idx.keys())):
        acc = np.full((n, 3), np.nan)
        gyr = np.full((n, 3), np.nan)
        for a, ax in enumerate(("x", "y", "z")):
            if frame in acc_idx and ax in acc_idx[frame]:
                acc[:, a] = data[:, acc_idx[frame][ax]]
            if frame in gyr_idx and ax in gyr_idx[frame]:
                gyr[:, a] = data[:, gyr_idx[frame][ax]]
        out[frame] = {"acc": acc, "gyr": gyr}
    return out, n


def print_c3d_labels(path):
    d = read_c3d(path)
    print("point_rate", d["point_rate"], "analog_rate", d["analog_rate"], "unit", d["point_unit"])
    print("n markers", len(d["markers"]))
    print("MARKER labels:", sorted(d["markers"].keys()))
    print("ANALOG labels:", d["alabels"])
    mapped = {k: v for k, v in GROUVEL_TO_RAJAGOPAL.items() if v and k in d["markers"]}
    print("MARKER map hits (%d):" % len(mapped), mapped)
    csv = sync_csv_path(path)
    print("sync csv:", csv, "exists", Path(csv).exists())
    if Path(csv).exists():
        imu, n = read_sync_imu(csv)
        print("IMU frames from sync (%d rows):" % n, sorted(imu.keys()))
        missing = set(OSIM_IMU_FRAMES) - set(imu.keys())
        print("UNMAPPED required frames:", sorted(missing))
        # quick unit sanity on one frame
        if "pelvis_imu" in imu:
            a = imu["pelvis_imu"]["acc"]; g = imu["pelvis_imu"]["gyr"]
            print("  pelvis acc |mean mag| %.2f  gyr p95 |mag| %.2f" % (
                np.nanmean(np.linalg.norm(a, axis=1)),
                np.nanpercentile(np.linalg.norm(g, axis=1), 95)))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_c3d_labels(sys.argv[1])
    else:
        print("usage: python imu_c3d_io.py <trial.c3d>")
