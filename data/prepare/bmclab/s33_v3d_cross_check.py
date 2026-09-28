"""Per-subject Visual3D sagittal cross-check.

For each walking trial, compare Rajagopal IK output (.mot) against the
Visual3D _angular_kinematics.csv from the BMClab archive on three sagittal
channels per side (hip flex, knee flex, ankle dorsi). Use the Pearson r as
the agreement metric.

V3D column layout (Step 2 task 5; verified in step2_report.md):
    col 0 : Frame
    col 1 : Time (s)
    cols 25 / 28 : Left/Right Hip Flexion/Extension  (EXT(-)/FLX(+) deg)
    cols 31 / 34 : Left/Right Knee Flex/Extension    (EXT(-)/FLX(+) deg)
    cols 37 / 40 : Left/Right Ankle Dorsi/PlantarFlx (PF(-)/DF(+) deg)
First 4 rows are gait-event annotations; data starts at row 6 (matches frame 1).

Time alignment:
    - V3D rows are time-domain at 150 Hz (verified in step2_report.md: time
      column = frame_idx / 150).
    - IK output is also at 150 Hz (matches the TRC sampling).
    - Both start at t=0 with frame 1 = first frame.
    → linear interp not strictly needed; we resample both to a common time
      grid just in case lengths differ by ±1 frame.

V3D sign conventions vs Rajagopal:
    - V3D hip_flex: EXT(-)/FLX(+) [deg]  →  matches Rajagopal hip_flexion sign
    - V3D knee_flex: EXT(-)/FLX(+) [deg] →  Rajagopal knee_angle is also "flexion-positive"
    - V3D ankle: PF(-)/DF(+) [deg]       →  Rajagopal ankle_angle is also "dorsiflexion-positive"
We use Pearson r so absolute amplitude differences (from joint-center mismatch)
don't matter — only waveform shape.
"""
from __future__ import annotations
import sys, zipfile, tempfile
from pathlib import Path
from typing import Optional
import numpy as np

# xlrd 1.x for the OLE2 'csv' files (= xls); ezc3d not needed here
import xlrd

PROJ = Path(__file__).resolve().parents[3]
IK_DIR = PROJ / "data/prepare/bmclab/work/ik"
ZIP_PATH = PROJ / "data/prepare/bmclab/raw/C3Dfiles.zip"

V3D_COLS = {
    "hip_flexion_l":  25,
    "hip_flexion_r":  28,
    "knee_angle_l":   31,
    "knee_angle_r":   34,
    "ankle_angle_l":  37,
    "ankle_angle_r":  40,
}
# IK columns in the .mot file use these names directly.
IK_COL_NAMES = list(V3D_COLS.keys())


def read_ik_mot(mot_path: Path) -> tuple:
    """Return (time, dict[col_name → array of values in deg]).

    OpenSim writes IK output in radians by default. We convert to degrees.
    """
    with open(mot_path, "r") as f:
        lines = f.readlines()
    # find endheader, header row, data start
    end_idx = None
    in_degrees = False
    for i, L in enumerate(lines):
        if "indegrees" in L.lower() or "inDegrees" in L:
            in_degrees = "yes" in L.lower()
        if L.strip() == "endheader":
            end_idx = i; break
    if end_idx is None:
        raise RuntimeError(f"{mot_path}: no endheader")
    headers = lines[end_idx + 1].split()
    n_cols = len(headers)
    data_rows = []
    for L in lines[end_idx + 2:]:
        L = L.strip()
        if not L: continue
        parts = L.split()
        if len(parts) != n_cols: continue
        try:
            data_rows.append([float(x) for x in parts])
        except ValueError:
            continue
    data = np.asarray(data_rows, dtype=float)
    time = data[:, headers.index("time")]
    out = {}
    factor = 1.0 if in_degrees else (180.0 / np.pi)
    for name in IK_COL_NAMES:
        if name in headers:
            out[name] = data[:, headers.index(name)] * factor
        else:
            out[name] = None
    return time, out


def read_v3d_angular(zip_entry: str, zf: zipfile.ZipFile, tmpdir: Path) -> Optional[tuple]:
    """Extract & parse the _angular_kinematics.csv (legacy xls) for a trial.
    Return (time_s, dict[col_name → array of values in deg]) or None if file missing.
    """
    try:
        zf.getinfo(zip_entry)
    except KeyError:
        return None
    out_path = tmpdir / Path(zip_entry).name
    with zf.open(zip_entry) as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    wb = xlrd.open_workbook(str(out_path))
    sh = wb.sheet_by_index(0)
    # data starts at row 6 (= sheet row index 5; row 0..3 = gait events, 4 = column names,
    # 5 = sign convention text, 6 = first numeric "frame=1, t=0")
    n_rows = sh.nrows
    DATA_START = 6
    time = []
    cols = {name: [] for name in V3D_COLS}
    for r in range(DATA_START, n_rows):
        try:
            t = float(sh.cell_value(r, 1))
        except (TypeError, ValueError):
            continue
        time.append(t)
        for name, ci in V3D_COLS.items():
            try:
                cols[name].append(float(sh.cell_value(r, ci)))
            except (TypeError, ValueError):
                cols[name].append(np.nan)
    out_path.unlink()
    if not time:
        return None
    return np.asarray(time), {k: np.asarray(v) for k, v in cols.items()}


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-d signals after dropping any NaN pairs."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m]; b = b[m]
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return np.nan
    return float((a * b).sum() / den)


def cross_check_trial(ik_mot: Path, zip_entry: str, zf: zipfile.ZipFile,
                      tmpdir: Path) -> dict:
    """Return per-channel Pearson r for one trial. NaN r if data missing."""
    ik_t, ik_cols = read_ik_mot(ik_mot)
    v3d = read_v3d_angular(zip_entry, zf, tmpdir)
    if v3d is None:
        return {"trial": ik_mot.stem, "status": "NO_V3D",
                **{c: np.nan for c in IK_COL_NAMES}}
    v3d_t, v3d_cols = v3d
    # If lengths differ slightly, linear-resample V3D onto IK timebase
    out = {"trial": ik_mot.stem, "status": "OK", "n_frames_ik": len(ik_t),
           "n_frames_v3d": len(v3d_t)}
    for name in IK_COL_NAMES:
        ik_y = ik_cols[name]
        v3d_y_raw = v3d_cols[name]
        if ik_y is None or v3d_y_raw is None:
            out[name] = np.nan
            continue
        # resample V3D onto IK timebase if needed
        if len(v3d_t) == len(ik_t) and np.allclose(v3d_t, ik_t, atol=1e-3):
            v3d_y = v3d_y_raw
        else:
            v3d_y = np.interp(ik_t, v3d_t, v3d_y_raw,
                              left=np.nan, right=np.nan)
        out[name] = pearson_r(ik_y, v3d_y)
    return out


def cross_check_subject(subject_id: str) -> dict:
    """Process all of one subject's walking IK outputs.

    Returns aggregate stats:
      - per-trial table (list of dicts)
      - median r per channel
      - overall median r (across all channels × trials)
    """
    trials = sorted(IK_DIR.glob(f"{subject_id}_*_walk_*_ik.mot"))
    if not trials:
        return {"subject_id": subject_id, "n_trials": 0,
                "status": "NO_TRIALS", "per_trial": []}
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf, tempfile.TemporaryDirectory() as tdir:
        tmpdir = Path(tdir)
        for ik_mot in trials:
            # IK mot stem: SUB01_off_walk_1_ik
            stem = ik_mot.stem
            assert stem.endswith("_ik")
            trial_base = stem[:-3]  # SUB01_off_walk_1
            # find the subject_med folder; the trial name's first two _ tokens are subject + medication
            parts = trial_base.split("_")
            # subject_med folder = e.g. SUB01_off; account for Trimmed_ prefix:
            if parts[0] == "Trimmed":
                # Trimmed_SUB01_off_walk_12b -> SUB01 / off
                sub = parts[1]; med = parts[2]
                folder = f"{sub}_{med}"
                fname = trial_base  # keep "Trimmed_" in filename
            else:
                sub = parts[0]; med = parts[1]
                folder = f"{sub}_{med}"
                fname = trial_base
            zip_entry = f"C3Dfiles/{folder}/{fname}_angular_kinematics.csv"
            try:
                r_row = cross_check_trial(ik_mot, zip_entry, zf, tmpdir)
            except Exception as e:
                r_row = {"trial": stem, "status": f"ERROR:{type(e).__name__}",
                         **{c: np.nan for c in IK_COL_NAMES}}
            rows.append(r_row)
    # aggregate
    n_with_r = sum(1 for r in rows if r["status"] == "OK")
    per_channel = {c: [r[c] for r in rows if isinstance(r.get(c), float) and np.isfinite(r[c])]
                   for c in IK_COL_NAMES}
    per_channel_median = {c: (float(np.median(v)) if v else np.nan) for c, v in per_channel.items()}
    all_vals = [v for vs in per_channel.values() for v in vs]
    overall_median = float(np.median(all_vals)) if all_vals else np.nan
    return {"subject_id": subject_id, "n_trials": len(rows), "n_with_r": n_with_r,
            "per_trial": rows, "per_channel_median": per_channel_median,
            "overall_median_r": overall_median}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "SUB01"
    result = cross_check_subject(sid)
    print(f"Subject: {result['subject_id']}, trials={result['n_trials']}, n_with_r={result.get('n_with_r')}")
    if "per_channel_median" in result:
        print("Per-channel median r:")
        for c, v in result["per_channel_median"].items():
            print(f"  {c:<20}  {v:.4f}" if np.isfinite(v) else f"  {c:<20}  NaN")
        print(f"Overall median r: {result['overall_median_r']:.4f}" if np.isfinite(result['overall_median_r']) else "Overall median r: NaN")
    for r in result.get("per_trial", [])[:5]:
        print(f"  trial={r['trial']}  status={r['status']}  "
              + " ".join(f"{k}={r.get(k, '?'):.3f}" if isinstance(r.get(k), float) and np.isfinite(r.get(k)) else f"{k}=?"
                         for k in IK_COL_NAMES))
