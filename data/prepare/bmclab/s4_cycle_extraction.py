"""Per-trial gait-cycle extraction + time-normalization (Step 4).

Inputs (per trial stem):
  - data/prepare/bmclab/work/ik/{stem}_ik.mot
  - data/prepare/bmclab/work/id/{stem}_id.sto
  - data/prepare/bmclab/work/grf/{stem}.mot
  - data/prepare/bmclab/work/grf/_grf_presence/{stem}.csv
  - V3D events from C3Dfiles.zip → C3Dfiles/{sub}_{med}/{stem}_angular_kinematics.csv

Outputs:
  - For each cycle: dict with all extracted channels + metadata.
  - Cycle definition: HS_i → HS_{i+1} on the same side.
  - Time-normalized to 101 equally-spaced points via linear interpolation.

ID time-base quirks:
  - ID .sto has dt ≈ 0.0066 s (~151.5 Hz), pad ±1.5 s.
  - We trim ID to [IK_t0, IK_t1] and re-base time to t=0, matching IK/GRF
    timeline. Cycle slicing uses time-based interpolation on the trimmed
    ID, not frame indexing.

GRF presence:
  - Frame-level boolean at 150 Hz. Cycle-level metric = mean(presence)
    in the cycle window, recorded per cycle as `grf_presence_in_cycle`.

Outlier flags (computed per cycle; not raised here, only recorded):
  - cycle_duration_sec in [0.7, 2.5]
  - hip_flexion range > 10° (any side, but we record both sides)
  - knee_angle peak > 20°
  - pelvis_tilt range < 40°
"""
from __future__ import annotations


import re
import zipfile
import tempfile
from pathlib import Path


import numpy as np
import pandas as pd
import xlrd

PROJ = Path(__file__).resolve().parents[3]
ZIP = PROJ / "data/prepare/bmclab/raw/C3Dfiles.zip"
IK_DIR = PROJ / "data/prepare/bmclab/work/ik"
ID_DIR = PROJ / "data/prepare/bmclab/work/id"
GRF_DIR = PROJ / "data/prepare/bmclab/work/grf"
PRESENCE_DIR = GRF_DIR / "_grf_presence"

# Channels we extract per cycle (kept as 1D arrays of length 101)
ANGLE_CHANNELS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_flexion_l",
    "hip_adduction_r", "hip_adduction_l",
    "hip_rotation_r", "hip_rotation_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
    "subtalar_angle_r", "subtalar_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]
MOMENT_CHANNELS = [
    "hip_flexion_r_moment", "hip_flexion_l_moment",
    "hip_adduction_r_moment", "hip_adduction_l_moment",
    "knee_angle_r_moment", "knee_angle_l_moment",
    "ankle_angle_r_moment", "ankle_angle_l_moment",
    "subtalar_angle_r_moment", "subtalar_angle_l_moment",
]
GRF_CHANNELS = [
    "r_ground_force_vx", "r_ground_force_vy", "r_ground_force_vz",
    "l_ground_force_vx", "l_ground_force_vy", "l_ground_force_vz",
]


def _read_storage(path: Path):
    """Read OpenSim .mot / .sto. Returns (time_array, dict[col_name → 1D array])."""
    lines = path.read_text().splitlines()
    end = next((i for i, L in enumerate(lines) if L.strip() == "endheader"), None)
    if end is None: return None, None
    cols = lines[end+1].split()
    data_rows = []
    for L in lines[end+2:]:
        L = L.strip()
        if not L: continue
        parts = L.split()
        if len(parts) != len(cols): continue
        try: data_rows.append([float(x) for x in parts])
        except ValueError: continue
    if not data_rows: return None, None
    data = np.asarray(data_rows)
    t = data[:, cols.index("time")]
    out = {c: data[:, cols.index(c)] for c in cols}
    return t, out


def _parse_v3d_events(stem: str, zip_path: Path = ZIP):
    """Read rows 0-3 of _angular_kinematics.csv to get RTO/RHS/LHS/LTO 1-based
    frame indices at 150 Hz. Returns dict or None."""
    if stem.startswith("Trimmed_"):
        parts = stem.split("_")
        sub, med = parts[1], parts[2]
    else:
        parts = stem.split("_")
        sub, med = parts[0], parts[1]
    entry = f"C3Dfiles/{sub}_{med}/{stem}_angular_kinematics.csv"
    with zipfile.ZipFile(zip_path) as zf:
        try: zf.getinfo(entry)
        except KeyError: return None
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.xls"
            with zf.open(entry) as src, open(out, "wb") as dst:
                dst.write(src.read())
            wb = xlrd.open_workbook(str(out))
    sh = wb.sheet_by_index(0)
    events = {}
    for r in range(min(4, sh.nrows)):
        cell = str(sh.cell_value(r, 0))
        m = re.match(r'"(\w+)"\s*([\d\s]+)', cell)
        if m:
            label = m.group(1)
            frames = sorted({int(x) for x in m.group(2).split() if x.strip().isdigit()})
            events[label] = frames
    return events


def _trim_id_to_window(t_id, cols_id, ik_t0, ik_t1, p_rate=150):
    eps = 0.5 / p_rate
    mask = (t_id >= ik_t0 - eps) & (t_id <= ik_t1 + eps)
    if mask.sum() == 0: return None, None
    new_t = t_id[mask] - ik_t0
    new_cols = {k: v[mask] for k, v in cols_id.items()}
    return new_t, new_cols


def _resample_to_101(t_src, x_src, t_start, t_end):
    """Linearly interpolate x_src defined at t_src onto 101 evenly-spaced
    points spanning [t_start, t_end]. Out-of-support → NaN."""
    grid = np.linspace(t_start, t_end, 101)
    out = np.interp(grid, t_src, x_src, left=np.nan, right=np.nan)
    return out


def extract_trial_cycles(stem: str, mass_kg: float, subject_id: str, medication_state: str,
                         p_rate: int = 150):
    """Extract every gait cycle from one trial. Returns list[dict] (one per cycle, both sides)
    plus a status dict at index 0 of the returned list iff something went wrong (we instead
    return (cycles, status_str)).
    """
    ik_path = IK_DIR / f"{stem}_ik.mot"
    id_path = ID_DIR / f"{stem}_id.sto"
    grf_path = GRF_DIR / f"{stem}.mot"
    pres_path = PRESENCE_DIR / f"{stem}.csv"
    for p in (ik_path, id_path, grf_path, pres_path):
        if not p.is_file():
            return [], f"missing input: {p.name}"

    t_ik, ik = _read_storage(ik_path)
    t_id, id_ = _read_storage(id_path)
    t_grf, grf = _read_storage(grf_path)
    if t_ik is None or t_id is None or t_grf is None:
        return [], "failed to parse storage"

    ik_t0, ik_t1 = t_ik[0], t_ik[-1]
    # Trim ID to IK support, re-base to t=0
    t_id_tr, id_tr = _trim_id_to_window(t_id, id_, ik_t0, ik_t1, p_rate)
    if t_id_tr is None:
        return [], "ID trim empty"
    # Re-base IK and GRF time to start at 0 (they should already start near 0)
    t_ik_re = t_ik - ik_t0
    t_grf_re = t_grf - ik_t0

    # Events
    events = _parse_v3d_events(stem)
    if events is None:
        return [], "no V3D events"
    # Convert to times (s) on the re-based timeline. V3D frames are 1-based at 150 Hz
    # starting from c3d frame 1 → t = (frame-1)/p_rate (already in same base as IK).
    def frames_to_t(frames):
        return [(f - 1) / p_rate for f in frames]

    # Presence
    pres = pd.read_csv(pres_path)
    t_pres = pres["time"].to_numpy()
    pres_r = pres["r_grf_present"].to_numpy()
    pres_l = pres["l_grf_present"].to_numpy()

    cycles = []
    for side, hs_label, to_label in (("r", "RHS", "RTO"), ("l", "LHS", "LTO")):
        hs_frames = events.get(hs_label, [])
        if len(hs_frames) < 2: continue
        hs_t = frames_to_t(sorted(hs_frames))
        for i in range(len(hs_t) - 1):
            t_a = hs_t[i]; t_b = hs_t[i + 1]
            dur = t_b - t_a
            # Sanity: cycle must lie inside IK support (after re-basing)
            if t_a < t_ik_re[0] - 1e-6 or t_b > t_ik_re[-1] + 1e-6:
                continue
            # ID support
            if t_a < t_id_tr[0] - 1e-6 or t_b > t_id_tr[-1] + 1e-6:
                continue
            # Resample channels to 101
            cyc = {}
            for ch in ANGLE_CHANNELS:
                if ch in ik:
                    cyc[ch] = _resample_to_101(t_ik_re, ik[ch], t_a, t_b)
                else:
                    cyc[ch] = np.full(101, np.nan)
            for ch in MOMENT_CHANNELS:
                if ch in id_tr:
                    cyc[ch] = _resample_to_101(t_id_tr, id_tr[ch], t_a, t_b) / mass_kg
                else:
                    cyc[ch] = np.full(101, np.nan)
            for ch in GRF_CHANNELS:
                if ch in grf:
                    cyc[ch] = _resample_to_101(t_grf_re, grf[ch], t_a, t_b) / (mass_kg * 9.81)
                else:
                    cyc[ch] = np.full(101, np.nan)
            # Presence in cycle
            pmask = (t_pres >= t_a) & (t_pres <= t_b)
            if pmask.sum() == 0:
                pres_frac = 0.0
            else:
                pres_frac = float(np.mean(pres_r[pmask] if side == "r" else pres_l[pmask]))
            # Outlier flags
            hip_flex_side = f"hip_flexion_{side}"
            knee_side = f"knee_angle_{side}"
            tilt = cyc.get("pelvis_tilt", np.array([np.nan]))
            hipf = cyc.get(hip_flex_side, np.array([np.nan]))
            kne = cyc.get(knee_side, np.array([np.nan]))
            with np.errstate(invalid="ignore"):
                tilt_range = np.nanmax(tilt) - np.nanmin(tilt) if np.isfinite(tilt).any() else np.nan
                hipf_range = np.nanmax(hipf) - np.nanmin(hipf) if np.isfinite(hipf).any() else np.nan
                knee_peak = np.nanmax(kne) if np.isfinite(kne).any() else np.nan
            reasons = []
            if not (0.7 <= dur <= 2.5): reasons.append(f"duration_{dur:.2f}s")
            if np.isfinite(hipf_range) and hipf_range < 10.0: reasons.append(f"hip_range_{hipf_range:.1f}deg")
            if np.isfinite(knee_peak) and knee_peak < 20.0: reasons.append(f"knee_peak_{knee_peak:.1f}deg")
            if np.isfinite(tilt_range) and tilt_range > 40.0: reasons.append(f"tilt_range_{tilt_range:.1f}deg")
            is_outlier = len(reasons) > 0
            cyc_meta = {
                "trial": stem,
                "subject_id": subject_id,
                "medication_state": medication_state,
                "side": side,
                "cycle_idx_in_trial_side": i,
                "cycle_t_start_s": t_a,
                "cycle_t_end_s": t_b,
                "cycle_duration_sec": dur,
                "cycle_n_frames_original_150hz": int(round(dur * p_rate)),
                "grf_presence_in_cycle": pres_frac,
                "mass_kg": mass_kg,
                "is_outlier": is_outlier,
                "outlier_reasons": ";".join(reasons),
                "pelvis_tilt_range_deg": float(tilt_range) if np.isfinite(tilt_range) else None,
                f"hip_flexion_{side}_range_deg": float(hipf_range) if np.isfinite(hipf_range) else None,
                f"knee_angle_{side}_peak_deg": float(knee_peak) if np.isfinite(knee_peak) else None,
            }
            cycles.append({"data": cyc, "meta": cyc_meta})
    return cycles, "ok"
