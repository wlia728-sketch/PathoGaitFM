"""Grouvel 2023 -> 54-ch raw npy harmonizer, TWO branches sharing one force GT.

Output format IDENTICAL to opencap_harmonize.py so the frozen eval maps the same 40 chans.
Marker branch + force GT from the C3D; IMU branch accel+gyro from the paired SYNC CSV.
Both 100 Hz, frame-synced. Cycles segmented by force-plate events (SAME events both branches).

Usage (inside WSL conda env bmclab_opensim):
    python imu_harmonize.py marker [P01_S01 ...]
    python imu_harmonize.py imu    [P01_S01 ...]
ASCII stdout only.
"""
from __future__ import annotations
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "SIMTK_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_k] = "1"
import sys, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "processed"
OSENSE_WORK = HERE / "opensense" / "_work"
G = 9.80665

import imu_c3d_io as cio
import imu_ahrs as ahrs
import imu_opensense as osense
import marker_ik as mik

ANG54 = {"hip_flexion_r": 0, "hip_flexion_l": 3, "knee_angle_r": 6, "knee_angle_l": 9,
         "ankle_angle_r": 12, "ankle_angle_l": 15}
PEL54 = {"pelvis_tilt": (48, 51), "pelvis_list": (49, 52), "pelvis_rotation": (50, 53)}
GRF54 = {"R": (34, 35, 36), "L": (37, 38, 39)}   # AP(x), ML(y-lateral), vertical(z-up)
GRF_THRESH_N = 20.0
MIN_STANCE_FR = 8      # at 100 Hz point rate
STANCE_FRAC = 0.63
ANG_SIGN = {k: +1.0 for k in ANG54}   # Rajagopal knee +ve flexion => IDENTITY


def resample_n(x, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def _read_force_plates(data):
    """Per plate: F=(n,3) N at analog rate, corners XY (mm)."""
    p = data["params"]; al = data["alabels"]; ana = data["analogs"]
    corners = np.asarray(p["FORCE_PLATFORM"]["CORNERS"]["value"])  # (3,4,nplate) mm
    nplate = corners.shape[2]
    plates = []
    for pl in range(nplate):
        keys = [f"Fx{pl+1}", f"Fy{pl+1}", f"Fz{pl+1}"]
        if not all(k in al for k in keys):
            continue
        F = np.stack([ana[al.index(k)] for k in keys], axis=1)  # (n_an,3), Vicon: Fz down<0
        xy = corners[:2, :, pl].T  # (4,2)
        plates.append({"F": F, "corners_xy": xy,
                       "xmin": xy[:, 0].min(), "xmax": xy[:, 0].max(),
                       "ymin": xy[:, 1].min(), "ymax": xy[:, 1].max()})
    return {"plates": plates, "arate": data["analog_rate"], "prate": data["point_rate"]}


def assign_plates_to_feet(force, markers, point_rate):
    """Assign each plate's stance to R/L foot by which heel marker sits inside the plate
    footprint during the plate's loaded window. Returns {'R':{'vx','vy','vz'},'L':{...}} at
    POINT rate (downsampled from analog), with vy = vertical (up, >=0), vx=AP, vz=ML.

    Vicon axes: force plate X=lab-progression, Y=lab-lateral, Z=vertical(down<0).
    We map to training GRF (vertical up, clip>=0). Cycle segmentation uses these events.
    """
    plates = force["plates"]; arate = force["arate"]
    if not plates or "RHEE" not in markers or "LHEE" not in markers:
        return None
    rhee = markers["RHEE"] * 1000.0  # back to mm to match corner units
    lhee = markers["LHEE"] * 1000.0
    n_pt = len(rhee)
    step = int(round(arate / point_rate))  # analog per point frame (10)

    # anti-alias low-pass the analog force before decimation (force plates carry HF noise;
    # decimating raw 1000->100 Hz aliases it into the vGRF shape). 4th-order Butterworth 15 Hz.
    def lp(sig):
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 15.0 / (arate / 2.0), btype="low")
            return filtfilt(b, a, sig, axis=0)
        except Exception:
            return sig

    # accumulate EACH plate's 3-comp force to the foot it belongs to, SUMMED over time, so a
    # footfall spanning two adjacent plates is the sum of both (no false mid-stance dropout).
    feet = {"R": {"vx": np.zeros(n_pt), "vy": np.zeros(n_pt), "vz": np.zeros(n_pt)},
            "L": {"vx": np.zeros(n_pt), "vy": np.zeros(n_pt), "vz": np.zeros(n_pt)}}
    any_assigned = False
    for pl in plates:
        F = lp(pl["F"])                 # (n_an,3) filtered
        vmag = -F[:, 2]                 # up-positive vertical
        loaded = vmag > GRF_THRESH_N
        i, n = 0, len(loaded)
        while i < n:
            if loaded[i]:
                j = i
                while j < n and loaded[j]:
                    j += 1
                if (j - i) >= MIN_STANCE_FR * step:
                    mid_pt = min(n_pt - 1, ((i + j) // 2) // step)

                    def inside(h):
                        x, y = h[mid_pt, 0], h[mid_pt, 1]
                        return (pl["xmin"] <= x <= pl["xmax"]) and (pl["ymin"] <= y <= pl["ymax"])
                    rin, lin = inside(rhee), inside(lhee)
                    side = None
                    if rin and not lin:
                        side = "R"
                    elif lin and not rin:
                        side = "L"
                    elif rin and lin:
                        cx = (pl["xmin"] + pl["xmax"]) / 2; cy = (pl["ymin"] + pl["ymax"]) / 2
                        dr = (rhee[mid_pt, 0]-cx)**2 + (rhee[mid_pt, 1]-cy)**2
                        dl = (lhee[mid_pt, 0]-cx)**2 + (lhee[mid_pt, 1]-cy)**2
                        side = "R" if dr < dl else "L"
                    if side:
                        p0, p1 = i // step, min(n_pt, (j + step - 1) // step)
                        idx_an = np.clip(np.arange(p0, p1) * step, 0, n - 1)
                        feet[side]["vx"][p0:p1] += F[idx_an, 0]     # AP (sum overlapping plates)
                        feet[side]["vz"][p0:p1] += F[idx_an, 1]     # ML
                        feet[side]["vy"][p0:p1] += -F[idx_an, 2]    # vertical up
                        any_assigned = True
                i = j
            else:
                i += 1
    if not any_assigned:
        return None
    return feet


def _events_from_vgrf(vy, thr=GRF_THRESH_N):
    loaded = vy > thr
    runs, i, n = [], 0, len(loaded)
    while i < n:
        if loaded[i]:
            j = i
            while j < n and loaded[j]:
                j += 1
            if j - i >= MIN_STANCE_FR:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def build_cycle(side, i0, i1, feet, angles, ang_t, prate, mass):
    cyc = np.full((101, 54), np.nan, np.float32)
    stance_pts = max(2, int(round(101 * STANCE_FRAC)))
    ap_i, ml_i, vt_i = GRF54[side]
    blk = feet[side]
    for gi, comp, clip in ((vt_i, "vy", True), (ap_i, "vx", False), (ml_i, "vz", False)):
        seg = blk[comp][i0:i1] / (mass * G)
        if clip:
            seg = np.clip(seg, 0, None)
        full = np.zeros(101, np.float32)
        full[:stance_pts] = resample_n(seg, stance_pts)
        cyc[:, gi] = full
    t0 = i0 / prate; t1 = i1 / prate
    cyc_end = t0 + (t1 - t0) / STANCE_FRAC
    ct = np.clip(np.linspace(t0, cyc_end, 101), ang_t[0], ang_t[-1])
    for cn, ch in ANG54.items():
        if cn in angles:
            cyc[:, ch] = ANG_SIGN[cn] * np.interp(ct, ang_t, angles[cn])
    for cn, (rch, lch) in PEL54.items():
        if cn in angles:
            seg = np.interp(ct, ang_t, angles[cn]); cyc[:, rch] = seg; cyc[:, lch] = seg
    return cyc


def _estimate_mass(feet):
    peak = max(np.nanmax(feet[s]["vy"]) for s in feet)
    return float(peak / (1.05 * G))


def _complete_stance(vy_seg, mass):
    """Keep only a plate contact that is a COMPLETE single-foot stance (heel-strike to
    toe-off on ONE plate). In a 3-plate walkway many contacts are partial (foot spans two
    plates) -> those give a truncated vGRF that resamples to a single mid-cycle hump and
    poisons the shape PCC. Gate on physiological completeness:
      - peak >= 0.7 BW (full body-weight acceptance reached)
      - rises from and returns toward baseline at the ends (both tails < 0.5*peak)
      - unloaded (<5% peak) at the very first & last sample -> the whole footfall is captured
    Returns True if the stance segment is a clean full contact.
    """
    bw = mass * G
    v = vy_seg / bw
    if v.size < MIN_STANCE_FR:
        return False
    peak = v.max()
    if peak < 0.7:
        return False
    # both ends must be near-unloaded (complete strike->off captured on this plate)
    edge = 0.15 * peak
    if v[0] > edge or v[-1] > edge:
        return False
    # argmax not pinned to an end (a truncated stance humps at one edge)
    amax = v.argmax() / (v.size - 1)
    if amax < 0.12 or amax > 0.88:
        return False
    return True


def harmonize_subject(subj, branch, masses):
    OUT.mkdir(parents=True, exist_ok=True)
    OSENSE_WORK.mkdir(parents=True, exist_ok=True)
    sd = RAW / subj
    c3ds = sorted(p for p in sd.glob("*_Gait_*.c3d"))
    static = sorted(sd.glob("*_Static_*.c3d"))
    if not c3ds:
        print(f"[{subj}] no gait c3d"); return None
    cycles, meta = [], []
    mass = masses.get(subj)
    # pre-estimate mass from the first gait trial's vGRF (needed to scale the marker model
    # per-subject BEFORE the loop, matching the IMU branch's per-subject static calibration).
    if mass is None:
        for c3d in c3ds:
            data0 = cio.read_c3d(c3d)
            feet0 = assign_plates_to_feet(_read_force_plates(data0), data0["markers"], data0["point_rate"])
            if feet0 is not None:
                mass = _estimate_mass(feet0); break
    # per-subject models built ONCE from the static trial (both branches now per-subject-calibrated)
    cal_model = None
    marker_model = None
    if branch == "imu" and static:
        cal_model = _calibrate_from_static(subj, static[0])
    if branch == "marker" and static:
        marker_model = _scale_marker_from_static(subj, static[0], mass or 75.0)

    for c3d in c3ds:
        data = cio.read_c3d(c3d)
        prate = data["point_rate"]
        force = _read_force_plates(data)
        feet = assign_plates_to_feet(force, data["markers"], prate)
        if feet is None:
            meta.append({"trial": c3d.stem, "status": "NO_FOOT_PLATE"}); continue
        if mass is None:
            mass = _estimate_mass(feet)
        if branch == "marker":
            ang_t, angles = _marker_branch(subj, c3d, data, marker_model)
        else:
            ang_t, angles = _imu_branch(subj, c3d, data, cal_model)
        if angles is None:
            meta.append({"trial": c3d.stem, "status": "IK_FAIL"}); continue
        for side in ("R", "L"):
            for (i0, i1) in _events_from_vgrf(feet[side]["vy"]):
                t0, t1 = i0 / prate, i1 / prate
                if t1 <= ang_t[0] or t0 >= ang_t[-1]:
                    continue
                if not _complete_stance(feet[side]["vy"][i0:i1], mass):
                    meta.append({"trial": c3d.stem, "side": side, "status": "PARTIAL_STANCE", "stance_fr": i1 - i0})
                    continue
                cyc = build_cycle(side, i0, i1, feet, angles, ang_t, prate, mass)
                cycles.append(cyc)
                meta.append({"trial": c3d.stem, "side": side, "status": "OK", "stance_fr": i1 - i0})

    if not cycles:
        print(f"[{subj}/{branch}] no cycles"); return None
    stack = np.stack(cycles)
    mask = np.isfinite(stack).all(axis=1)
    tag = f"{subj}__{branch}"
    np.save(OUT / f"{tag}.npy", stack.astype(np.float32))
    np.save(OUT / f"{tag}_mask.npy", mask)
    with open(OUT / f"{tag}_meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "trial", "side", "stance_fr", "status"])
        for i, m in enumerate(meta):
            w.writerow([i, m.get("trial", ""), m.get("side", ""), m.get("stance_fr", ""), m["status"]])
    nvgrf = int(mask[:, [36, 39]].sum())
    print(f"[{tag}] {stack.shape[0]} cyc, mass~{mass:.1f}kg, vGRF-valid {nvgrf}")
    return {"subj": subj, "branch": branch, "n_cyc": int(stack.shape[0]), "mass": float(mass)}


def _scale_marker_from_static(subj, static_c3d, mass):
    """Per-subject scale the training Rajagopal marker-IK model from the STATIC trial's
    markers (mirrors the IMU branch's per-subject IMUPlacer calibration -> fair comparison)."""
    data = cio.read_c3d(static_c3d)
    rmarkers = {}
    for src, dst in cio.GROUVEL_TO_RAJAGOPAL.items():
        if dst and src in data["markers"]:
            rmarkers[dst] = data["markers"][src]
    if len(rmarkers) < 6:
        return None
    trc = OSENSE_WORK / f"{subj}_static.trc"
    mik.write_trc(trc, rmarkers, data["times"], data["point_rate"])
    model = OSENSE_WORK / f"{subj}_marker_scaled.osim"
    mik.scale_model(trc, data["times"][0], model, mass_kg=float(mass))
    return model if Path(model).exists() else None


def _marker_branch(subj, c3d, data, marker_model=None):
    rmarkers = {}
    for src, dst in cio.GROUVEL_TO_RAJAGOPAL.items():
        if dst and src in data["markers"]:
            rmarkers[dst] = data["markers"][src]
    if len(rmarkers) < 6:
        return data["times"], None
    trc = OSENSE_WORK / f"{subj}_{c3d.stem}.trc"
    mik.write_trc(trc, rmarkers, data["times"], data["point_rate"])
    model = marker_model
    if model is None or not Path(model).exists():
        model = OSENSE_WORK / f"{subj}_marker.osim"
        if not model.exists():
            mik.scale_model(trc, data["times"][0], model, mass_kg=75.0)
    mot = OSENSE_WORK / f"{subj}_{c3d.stem}_marker_ik.mot"
    res = mik.run_marker_ik(model, trc, mot, data["times"][0], data["times"][-1])
    if res is None:
        return data["times"], None
    return mik.read_ik_mot(res)


def _calibrate_from_static(subj, static_c3d):
    """Build the IMU-calibrated model once from the static trial's IMU orientations."""
    data = cio.read_c3d(static_c3d)
    csv = cio.sync_csv_path(static_c3d)
    imu, n = cio.read_sync_imu(csv)
    if imu is None or n < 5:
        # fall back to using the first gait trial's early frames for calibration
        return None
    arate = data["point_rate"]  # sync csv is at point rate
    frame_order = [f for f in cio.OSIM_IMU_FRAMES if f in imu]
    quats = {fr: ahrs.madgwick_quat(imu[fr]["acc"], imu[fr]["gyr"], arate) for fr in frame_order}
    quats = ahrs.heading_reset(quats, base_frame="pelvis_imu")
    t = np.arange(n) / arate
    sto = OSENSE_WORK / f"{subj}_static_orient.sto"
    ahrs.write_orientations_sto(sto, quats, t, frame_order)
    cal = OSENSE_WORK / f"{subj}_calibrated.osim"
    osense.calibrate_model(sto, cal, static_time=t[0])
    return cal


def _imu_branch(subj, c3d, data, cal_model):
    csv = cio.sync_csv_path(c3d)
    imu, n = cio.read_sync_imu(csv)
    if imu is None or n < 5:
        return data["times"], None
    arate = data["point_rate"]  # sync at point rate (frame-synced with c3d)
    frame_order = [f for f in cio.OSIM_IMU_FRAMES if f in imu]
    if len(frame_order) < 6:
        return data["times"], None
    quats = {fr: ahrs.madgwick_quat(imu[fr]["acc"], imu[fr]["gyr"], arate) for fr in frame_order}
    quats = ahrs.heading_reset(quats, base_frame="pelvis_imu")
    t = np.arange(n) / arate
    sto = OSENSE_WORK / f"{subj}_{c3d.stem}_orient.sto"
    ahrs.write_orientations_sto(sto, quats, t, frame_order)
    cal = cal_model
    if cal is None or not Path(cal).exists():
        cal = OSENSE_WORK / f"{subj}_{c3d.stem}_cal.osim"
        osense.calibrate_model(sto, cal, static_time=t[0])
    resdir = OSENSE_WORK / f"{subj}_{c3d.stem}_imuik"
    mot = osense.run_imu_ik(cal, sto, resdir, t[0], t[-1])
    if mot is None:
        return data["times"], None
    return osense.read_ik_mot(mot)


def main():
    branch = sys.argv[1] if len(sys.argv) > 1 else "marker"
    assert branch in ("marker", "imu")
    mj = HERE / "masses.json"
    masses = json.load(open(mj)) if mj.exists() else {}
    subs = sys.argv[2:] or sorted(p.name for p in RAW.glob("P*") if p.is_dir())
    summ = []
    for s in subs:
        try:
            r = harmonize_subject(s, branch, masses)
            if r: summ.append(r)
        except Exception as e:
            import traceback; print(f"[{s}] FAIL {type(e).__name__}: {e}"); traceback.print_exc()
    (OUT / f"_summary_{branch}.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\nHarmonized {len(summ)} subjects ({branch}) -> {OUT}")


if __name__ == "__main__":
    main()
