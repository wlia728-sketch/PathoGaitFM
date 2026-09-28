"""GRF transform — per-trial pipeline.

Inputs (per trial):
  - c3d (raw analog, plate metadata)
  - V3D `_grf.csv` (already R/L resolved force in lab-frame N/kg, 142.857 Hz)
  - V3D `_angular_kinematics.csv` (only for gait events; loaded by caller)
  - subject mass (from PDGinfo.xlsx)

Outputs (per trial):
  - `data/prepare/bmclab/work/grf/{trial}.mot`             — 19-column OpenSim external loads
  - `data/prepare/bmclab/work/grf/{trial}_external_loads.xml`
  - `data/prepare/bmclab/work/grf/_grf_presence/{trial}.csv` — per-frame R/L GRF flag at 150 Hz

Strategy:
  1. Per-plate CoP in lab frame (mm, at analog rate 450 Hz):
       - TYPE-3 plates 2, 3: corner-Fz formula (no sign ambiguity).
       - TYPE-4 plates 4, 5: CAL_MATRIX (×1000 on M) + axis-mapping H1.
       - Plate 6: dummy, skipped.
     All with per-channel baseline subtraction (first 0.5 s of analog).
  2. V3D R/L force is the lab-frame force vector per foot (N/kg → N via mass).
  3. For each frame at 150 Hz (resample analog 450→150 by block-avg ×3):
       - For each foot side (R, L): find active plates among 2-5 where
         |Fz_reaction| > 20 N AND that plate's CoP is closer to that foot's
         CAL marker than the other foot's CAL marker.
       - CoP for that foot = Fz-weighted average of active-plates' CoPs.
       - If no plate active for that side at this frame: write NaN (will become
         zeros in .mot).
  4. V3D force already has the correct lab-frame components; just multiply by
     subject mass to convert N/kg → N.
  5. Resample 450 Hz analog → 150 Hz CoP / Fz-totals via block average.
     V3D force at 142.857 Hz → 150 Hz via linear interp.
  6. Build the 19-column .mot.

The five-category mask flags `pass / r_missing / l_missing / both_missing /
grf_empty` are computed by the batch driver, not here.
"""
from __future__ import annotations



import zipfile
import tempfile
from pathlib import Path


import numpy as np
import ezc3d
import xlrd

PROJ = Path(__file__).resolve().parents[3]
ZIP = PROJ / "data/prepare/bmclab/raw/C3Dfiles.zip"

# Plate indices (0-indexed). Plates 0,1 = virtual; 6 = dummy.
REAL_PLATES = (2, 3, 4, 5)


# ---------------------------------------------------------------------------
# c3d loading
# ---------------------------------------------------------------------------
def load_c3d_baseline(c3d_path: str, baseline_s: float = 0.5):
    c = ezc3d.c3d(c3d_path)
    a_rate = float(c["header"]["analogs"]["frame_rate"])
    n_baseline = int(baseline_s * a_rate)
    anal = c["data"]["analogs"][0].copy()
    if anal.shape[1] > n_baseline:
        anal = anal - anal[:, :n_baseline].mean(axis=1, keepdims=True)
    return c, anal, a_rate


# ---------------------------------------------------------------------------
# CoP per plate (in lab/world mm, at analog rate)
# ---------------------------------------------------------------------------
def cop_type3(c, anal, p_idx, fz_threshold: float = 20.0):
    """Corner-Fz formula. Returns (cop_lab (3, n_a), fz_reaction (n_a,))."""
    chmap = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"])
    ch = [int(c_) for c_ in chmap[:, p_idx] if c_ > 0]
    if len(ch) < 8:
        n = anal.shape[1]
        return None, np.zeros(n)
    fz_chs = [ch[i] - 1 for i in (4, 5, 6, 7)]
    fz_corners = anal[fz_chs, :]
    fz_action = fz_corners.sum(axis=0)
    fz_reaction = -fz_action
    safe = np.abs(fz_reaction) > fz_threshold
    safe_total = np.where(safe, fz_action, 1.0)
    corners = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"])[:, :, p_idx]
    cop = np.zeros((3, anal.shape[1]))
    for k in range(4):
        cop += corners[:, k:k+1] * fz_corners[k:k+1, :]
    cop = cop / safe_total
    cop[:, ~safe] = np.nan
    return cop, fz_reaction


def cop_type4(c, anal, p_idx, fz_threshold: float = 20.0):
    """CAL_MATRIX + H1 axis-swap. Returns (cop_lab (3, n_a), fz_reaction (n_a,))."""
    chmap = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"])
    cal = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CAL_MATRIX"]["value"])[..., p_idx]
    ch = [int(c_) for c_ in chmap[:, p_idx] if c_ > 0][:6]
    if len(ch) < 6:
        n = anal.shape[1]
        return None, np.zeros(n)
    raw_6 = anal[[c_ - 1 for c_ in ch], :]
    calibrated = cal @ raw_6   # (6, n_a)
    Fz_a = calibrated[2]
    Mx_a, My_a = calibrated[3], calibrated[4]
    Fz_r = -Fz_a
    Mx_r = -Mx_a
    My_r = -My_a

    safe = np.abs(Fz_r) > fz_threshold
    safe_Fz = np.where(safe, Fz_r, 1.0)
    cop_x_local = -My_r / safe_Fz   # plate-local AP
    cop_y_local = +Mx_r / safe_Fz   # plate-local ML
    cop_x_local[~safe] = np.nan
    cop_y_local[~safe] = np.nan

    corners = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"])[:, :, p_idx]
    origin_local = np.asarray(c["parameters"]["FORCE_PLATFORM"]["ORIGIN"]["value"])[:, p_idx]
    plate_cx = (corners[0, :].max() + corners[0, :].min()) / 2.0
    plate_cz = (corners[2, :].max() + corners[2, :].min()) / 2.0

    # H1 axis mapping: plate-local X → lab Z, plate-local Y → lab X
    cop_lab = np.zeros((3, anal.shape[1]))
    cop_lab[0] = plate_cx + cop_y_local + origin_local[1]   # lab X (AP)
    cop_lab[1] = 0.0                                          # lab Y (plate surface)
    cop_lab[2] = plate_cz + cop_x_local + origin_local[0]   # lab Z (lateral)
    return cop_lab, Fz_r


def plate_cop_and_fz(c, anal, p_idx, fz_threshold: float = 20.0):
    """Dispatcher. Returns (cop_lab (3, n_a) in mm, fz_reaction (n_a,) in N)."""
    types = list(int(t) for t in c["parameters"]["FORCE_PLATFORM"]["TYPE"]["value"])
    pt = types[p_idx]
    if pt == 3:
        return cop_type3(c, anal, p_idx, fz_threshold)
    elif pt == 4:
        return cop_type4(c, anal, p_idx, fz_threshold)
    return None, np.zeros(anal.shape[1])


# ---------------------------------------------------------------------------
# V3D _grf.csv loading
# ---------------------------------------------------------------------------
def load_v3d_grf_from_zip(zip_entry: str):
    """Return (time_array, dict[name → array]) in N/kg."""
    with zipfile.ZipFile(ZIP) as zf, tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.xls"
        with zf.open(zip_entry) as src, open(out, "wb") as dst:
            dst.write(src.read())
        wb = xlrd.open_workbook(str(out))
    sh = wb.sheet_by_index(0)
    n = sh.nrows - 6
    if n <= 0:
        return None, None
    t = np.array([float(sh.cell_value(6 + r, 1)) for r in range(n)])
    cols = {}
    for ci, name in zip(range(2, 8), ("L_AP", "L_VERT", "L_ML", "R_AP", "R_VERT", "R_ML")):
        cols[name] = np.array([float(sh.cell_value(6 + r, ci)) for r in range(n)])
    return t, cols


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------
def downsample_analog_to_pt(arr_at_analog, p_rate: int = 150, a_rate: int = 450):
    """Block-average factor a_rate/p_rate (default 3:1) along last axis.
    Handles NaN by ignoring them in the mean."""
    factor = a_rate // p_rate
    n = arr_at_analog.shape[-1] // factor
    trimmed = arr_at_analog[..., : n * factor]
    reshaped = trimmed.reshape(*trimmed.shape[:-1], n, factor)
    with np.errstate(invalid="ignore", all="ignore"):
        out = np.nanmean(reshaped, axis=-1)
    return out


def linear_resample(t_src, x_src, t_dst):
    """Linearly interpolate x_src defined at t_src onto t_dst grid."""
    return np.interp(t_dst, t_src, x_src, left=np.nan, right=np.nan)


# ---------------------------------------------------------------------------
# Plate-to-foot assignment + aggregation
# ---------------------------------------------------------------------------
def marker_traj(c, name):
    labels = [str(x).strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
    if name not in labels: return None
    idx = labels.index(name)
    return c["data"]["points"][:3, idx, :].T  # (n_p, 3)


def per_foot_cop_at_pt_rate(c, anal, mass_kg: float, p_rate: int = 150):
    """Compute, per point-rate frame:
        cop_r (3, n_p), cop_l (3, n_p)  — Fz-weighted from active plates
        fz_r (n_p,), fz_l (n_p,)        — total reaction (N) on each foot side from analog
    Side assignment uses each plate's CoP-vs-CAL-marker distance.

    Returns also per-plate CoP and Fz dictionaries for diagnostics.
    """
    a_rate = float(c["header"]["analogs"]["frame_rate"])
    # Compute per-plate CoP + Fz at analog rate
    plate_cop = {}
    plate_fz  = {}
    for p in REAL_PLATES:
        cop, fz = plate_cop_and_fz(c, anal, p)
        if cop is None:
            continue
        plate_cop[p] = cop   # (3, n_a) mm
        plate_fz[p]  = fz    # (n_a,) N reaction

    # Downsample to point rate
    factor = int(round(a_rate / p_rate))   # =3 for 450→150
    n_a = anal.shape[1]
    n_p = n_a // factor

    plate_cop_pt = {p: downsample_analog_to_pt(plate_cop[p], p_rate, int(a_rate)) for p in plate_cop}
    plate_fz_pt  = {p: downsample_analog_to_pt(plate_fz[p],  p_rate, int(a_rate)) for p in plate_fz}

    # Marker positions at point rate
    rcal = marker_traj(c, "R.Heel")
    lcal = marker_traj(c, "L.Heel")
    if rcal is None or lcal is None:
        return None
    # Crop to n_p frames if longer (point rate matches c3d point rate already, 150 Hz)
    rcal_pt = rcal[:n_p]
    lcal_pt = lcal[:n_p]

    cop_r = np.full((3, n_p), np.nan)
    cop_l = np.full((3, n_p), np.nan)
    fz_r  = np.zeros(n_p)
    fz_l  = np.zeros(n_p)

    for f in range(n_p):
        # For each plate, check if active and which side
        r_contributions = []  # list of (cop_xyz, fz_value)
        l_contributions = []
        for p in REAL_PLATES:
            if p not in plate_fz_pt: continue
            fz_p = plate_fz_pt[p][f]
            if not np.isfinite(fz_p) or fz_p < 20: continue
            cop_p = plate_cop_pt[p][:, f]
            if not np.all(np.isfinite(cop_p)): continue
            # Distance to R vs L CAL in the XZ plane
            d_r = np.sqrt((cop_p[0] - rcal_pt[f, 0]) ** 2 + (cop_p[2] - rcal_pt[f, 2]) ** 2)
            d_l = np.sqrt((cop_p[0] - lcal_pt[f, 0]) ** 2 + (cop_p[2] - lcal_pt[f, 2]) ** 2)
            if d_r <= d_l:
                r_contributions.append((cop_p, fz_p))
            else:
                l_contributions.append((cop_p, fz_p))
        if r_contributions:
            total_fz = sum(fz for _, fz in r_contributions)
            cop = sum(cop_p * fz for cop_p, fz in r_contributions) / total_fz
            cop_r[:, f] = cop
            fz_r[f] = total_fz
        if l_contributions:
            total_fz = sum(fz for _, fz in l_contributions)
            cop = sum(cop_p * fz for cop_p, fz in l_contributions) / total_fz
            cop_l[:, f] = cop
            fz_l[f] = total_fz
    return cop_r, cop_l, fz_r, fz_l


# ---------------------------------------------------------------------------
# Build .mot
# ---------------------------------------------------------------------------
def build_grf_mot(
    *,
    trial_name: str,
    mass_kg: float,
    out_mot: Path,
    p_rate: int,
    cop_r_mm: np.ndarray,   # (3, n_p), NaN where no contact
    cop_l_mm: np.ndarray,
    v3d_t: np.ndarray,
    v3d_R: dict,            # 'AP', 'VERT', 'ML' in N/kg
    v3d_L: dict,
):
    """Build the 19-column OpenSim external loads .mot.

    Columns:
      time
      r_ground_force_vx, _vy, _vz
      r_ground_force_px, _py, _pz   (point of application, in meters)
      r_ground_torque_x, _y, _z     (free moment; zero for now)
      l_ground_force_vx, _vy, _vz
      l_ground_force_px, _py, _pz
      l_ground_torque_x, _y, _z
    """
    n_p = cop_r_mm.shape[1]
    t_pt = np.arange(n_p) / p_rate
    # V3D force, resample to point-rate
    def rs(name, side):
        d = v3d_R if side == "R" else v3d_L
        return linear_resample(v3d_t, d[name], t_pt)
    R_AP   = rs("AP",   "R") * mass_kg
    R_VERT = rs("VERT", "R") * mass_kg
    R_ML   = rs("ML",   "R") * mass_kg
    L_AP   = rs("AP",   "L") * mass_kg
    L_VERT = rs("VERT", "L") * mass_kg
    L_ML   = rs("ML",   "L") * mass_kg
    # Map V3D's (AP, VERT, ML) → OpenSim (vx, vy, vz) = (lab X, lab Y, lab Z)
    r_vx = np.nan_to_num(R_AP);   r_vy = np.nan_to_num(R_VERT); r_vz = np.nan_to_num(R_ML)
    l_vx = np.nan_to_num(L_AP);   l_vy = np.nan_to_num(L_VERT); l_vz = np.nan_to_num(L_ML)

    def cop_to_m_cols(cop_mm, fz_signal):
        cop_m = cop_mm / 1000.0
        no_contact = np.abs(fz_signal) < 20
        cop_m[:, no_contact] = 0.0
        cop_m = np.nan_to_num(cop_m, nan=0.0)
        return cop_m[0], cop_m[1], cop_m[2]

    r_px, r_py, r_pz = cop_to_m_cols(cop_r_mm, r_vy)
    l_px, l_py, l_pz = cop_to_m_cols(cop_l_mm, l_vy)

    r_tx = np.zeros(n_p); r_ty = np.zeros(n_p); r_tz = np.zeros(n_p)
    l_tx = np.zeros(n_p); l_ty = np.zeros(n_p); l_tz = np.zeros(n_p)

    r_vx = np.nan_to_num(r_vx); r_vy = np.nan_to_num(r_vy); r_vz = np.nan_to_num(r_vz)
    l_vx = np.nan_to_num(l_vx); l_vy = np.nan_to_num(l_vy); l_vz = np.nan_to_num(l_vz)

    # Option A: force-CoP consistency rule. CoP at lab origin means the
    # geometric anchor for this foot is not validated this frame. Without
    # a real point of application the force has no defined moment arm, so
    # ID extrapolates to lab origin and produces phantom N·m peaks. Zero
    # force whenever |CoP| < 0.01 m for that side, per-frame.
    r_cop_mag = np.sqrt(r_px**2 + r_py**2 + r_pz**2)
    l_cop_mag = np.sqrt(l_px**2 + l_py**2 + l_pz**2)
    r_invalid = r_cop_mag < 0.01
    l_invalid = l_cop_mag < 0.01
    r_vx[r_invalid] = 0; r_vy[r_invalid] = 0; r_vz[r_invalid] = 0
    r_tx[r_invalid] = 0; r_ty[r_invalid] = 0; r_tz[r_invalid] = 0
    l_vx[l_invalid] = 0; l_vy[l_invalid] = 0; l_vz[l_invalid] = 0
    l_tx[l_invalid] = 0; l_ty[l_invalid] = 0; l_tz[l_invalid] = 0

    out_mot.parent.mkdir(parents=True, exist_ok=True)
    cols = ["time",
            "r_ground_force_vx","r_ground_force_vy","r_ground_force_vz",
            "r_ground_force_px","r_ground_force_py","r_ground_force_pz",
            "r_ground_torque_x","r_ground_torque_y","r_ground_torque_z",
            "l_ground_force_vx","l_ground_force_vy","l_ground_force_vz",
            "l_ground_force_px","l_ground_force_py","l_ground_force_pz",
            "l_ground_torque_x","l_ground_torque_y","l_ground_torque_z"]
    n = n_p
    with open(out_mot, "w") as f:
        f.write(f"{trial_name}\n")
        f.write(f"datacolumns {len(cols)}\n")
        f.write(f"datarows {n}\n")
        f.write(f"range {t_pt[0]:.6f} {t_pt[-1]:.6f}\n")
        f.write("endheader\n")
        f.write("\t".join(cols) + "\n")
        for i in range(n):
            vals = [t_pt[i],
                    r_vx[i], r_vy[i], r_vz[i],
                    r_px[i], r_py[i], r_pz[i],
                    r_tx[i], r_ty[i], r_tz[i],
                    l_vx[i], l_vy[i], l_vz[i],
                    l_px[i], l_py[i], l_pz[i],
                    l_tx[i], l_ty[i], l_tz[i]]
            f.write("\t".join(f"{v:.6f}" for v in vals) + "\n")
    return cols


def build_external_loads_xml(*, trial_name: str, mot_rel_path: str, out_xml: Path):
    """Write the External Loads XML referencing the mot we just wrote."""
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    template = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<ExternalLoads name="{trial_name}">
\t\t<objects>
\t\t\t<ExternalForce name="r_ground">
\t\t\t\t<applied_to_body>calcn_r</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>r_ground_force_v</force_identifier>
\t\t\t\t<point_identifier>r_ground_force_p</point_identifier>
\t\t\t\t<torque_identifier>r_ground_torque_</torque_identifier>
\t\t\t\t<data_source_name>{mot_rel_path}</data_source_name>
\t\t\t</ExternalForce>
\t\t\t<ExternalForce name="l_ground">
\t\t\t\t<applied_to_body>calcn_l</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>l_ground_force_v</force_identifier>
\t\t\t\t<point_identifier>l_ground_force_p</point_identifier>
\t\t\t\t<torque_identifier>l_ground_torque_</torque_identifier>
\t\t\t\t<data_source_name>{mot_rel_path}</data_source_name>
\t\t\t</ExternalForce>
\t\t</objects>
\t\t<groups/>
\t\t<datafile>{mot_rel_path}</datafile>
\t</ExternalLoads>
</OpenSimDocument>
"""
    out_xml.write_text(template)


if __name__ == "__main__":
    print("Library module — use s34_batch_grf.py")
