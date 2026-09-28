"""Batch Scale Tool driver — one scaled model per subject.

For each subject:
- pick the own-medication static TRC if available, else fall back to other-med.
- read mass_kg + height_cm from PDGinfo.xlsx.
- build the Scale setup XML programmatically.
- run opensim.ScaleTool.
- read scaled model back, compute segment lengths + RMS errors.
- log to _scale_log.csv.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import opensim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s32_scale_setup_builder import build_scale_setup

# ============================================================================
PROJ_ROOT = Path(__file__).resolve().parents[3]
RAJAGOPAL_PATH = str(Path(__import__("nimblephysics").__file__).resolve().parent / "models" / "rajagopal_data" / "Rajagopal2015.osim")
PDG_PATH = PROJ_ROOT / "data/prepare/bmclab/raw/PDGinfo.xlsx"
TRC_DIR = PROJ_ROOT / "data/prepare/bmclab/work/trc"
OUT_DIR = PROJ_ROOT / "data/prepare/bmclab/work/scaled_models"
LOG_PATH = OUT_DIR / "_scale_log.csv"

# Subjects excluded entirely (no walking data either side — see Step 2)
EXCLUDED_NO_DATA = {("SUB04", "off"), ("SUB23", "off"),
                    ("SUB25", "off"), ("SUB26", "on")}


def find_static_trc_for(subject_id: str, med: str) -> Optional[Path]:
    """Return the {subject}_{med}_static_1.trc if it exists, else None.

    Per-medication scaling (Step 3.3 fix for SUB05): each (subject, med) walking
    set gets its own scaled model built from that day's static trial, not a
    cross-state fallback. If a static is genuinely missing for this (subject, med),
    return None and let the caller skip — DO NOT silently fall back to the other
    medication's static, that's the bug we're fixing.
    """
    p = TRC_DIR / f"{subject_id}_{med}_static_1.trc"
    return p if p.is_file() else None


def read_pdg_info() -> pd.DataFrame:
    return pd.read_excel(PDG_PATH, sheet_name="PDGinfo", header=0)


def compute_segment_lengths(model_path: Path) -> dict:
    """Read scaled .osim, return segment lengths in metres computed as
    inter-marker distances in the *model* (not the static TRC).

    The OpenSim SimTK::Vec3 returned by getLocationInGround() supports
    .get(i) but not Python's subscript[] operator on all binding builds, so
    we go through .get(0/1/2) to be safe.
    """
    m = opensim.Model(str(model_path))
    s = m.initSystem()

    def marker_xyz(name: str):
        v = m.getMarkerSet().get(name).getLocationInGround(s)
        return (v.get(0), v.get(1), v.get(2))

    def dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5

    out = {}
    pairs = [
        ("pelvis_width_m",   "RASI", "LASI"),
        ("femur_length_r_m", "RASI", "RLFC"),
        ("tibia_length_r_m", "RLFC", "RLMAL"),
        ("foot_length_r_m",  "RCAL", "RTOE"),
    ]
    for key, m1, m2 in pairs:
        try:
            out[key] = dist(marker_xyz(m1), marker_xyz(m2))
        except Exception as e:
            out[f"_err_{key}"] = repr(e)
    return out


def parse_rms_from_scale_log(captured: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse RMS/max marker error from the OpenSim log captured via Logger.addFileSink.

    OpenSim 4.5 MarkerPlacer emits lines like:
        Frame at (t = 0.0):  total squared error = 0.0666, marker error: RMS = 0.0487, max = 0.113 (R_tibial_plateau)

    The ModelScaler step doesn't print an RMS (scaling is geometric, not optimization-based),
    so ms_rms is always None — we use the same MP RMS for both columns in the log to keep
    the column header schema stable. Returns (rms_m, max_m, max_marker_name).
    """
    import re
    pat = re.compile(
        r"marker error:\s*RMS\s*=\s*([\d.]+)[,\s]+max\s*=\s*([\d.]+)\s*\(([^)]+)\)",
        re.IGNORECASE,
    )
    rms = None
    mx  = None
    mname = None
    for line in captured.splitlines():
        m = pat.search(line)
        if m:
            rms = float(m.group(1))
            mx  = float(m.group(2))
            mname = m.group(3).strip()
    return rms, mx, mname


def total_mass(model_path: Path) -> float:
    m = opensim.Model(str(model_path))
    m.initSystem()
    total = 0.0
    for i in range(m.getBodySet().getSize()):
        total += m.getBodySet().get(i).getMass()
    return total


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdg = read_pdg_info()
    print(f"PDGinfo: {len(pdg)} subjects, columns including: ID, Gender, Weight (kg), Height (cm).")

    rows = []
    t0 = time.time()
    for _, sub_row in pdg.iterrows():
        sid = str(sub_row["ID"]).strip()
        try:
            mass_kg = float(sub_row["Weight (kg)"])
            height_cm = float(sub_row["Height (cm)"])
        except (ValueError, TypeError) as e:
            print(f"  {sid}: bad mass/height ({e}); skipping")
            rows.append({"subject_id": sid, "med": "?", "status": "FAIL_MASS_PARSE",
                         "error_msg": str(e)})
            continue

        # Per-medication scaled models (Step 3.3 fix). Skip the 4 excluded
        # (subject, med) entries — these have no data on disk and no metadata
        # quality concerns since they're never used.
        for med in ("off", "on"):
            if (sid, med) in EXCLUDED_NO_DATA:
                print(f"  {sid}_{med}: in EXCLUDED_NO_DATA — skipping")
                rows.append({"subject_id": sid, "med": med,
                             "status": "SKIP_EXCLUDED",
                             "error_msg": "(subject, med) in EXCLUDED_NO_DATA"})
                continue

            static_path = find_static_trc_for(sid, med)
            if static_path is None:
                rows.append({"subject_id": sid, "med": med,
                             "status": "FAIL_NO_STATIC",
                             "error_msg": f"missing {sid}_{med}_static_1.trc"})
                print(f"  {sid}_{med}: no static — skipping")
                continue

            tag = f"{sid}_{med}"
            out_model = OUT_DIR / f"{tag}.osim"
            setup_xml = OUT_DIR / f"_{tag}_scale_setup.xml"

            print(f"\n--- {tag}  static={static_path.name}  mass={mass_kg:.2f} kg  height={height_cm:.0f} cm ---")
            row = {
                "subject_id": sid,
                "med": med,
                "static_trc_used": static_path.name,
                "mass_in_kg": mass_kg,
                "height_in_cm": height_cm,
                "ms_rms_cm": "",
                "mp_rms_cm": "",
                "mp_max_cm": "",
                "mp_max_marker": "",
                "pelvis_width_m": "",
                "femur_length_r_m": "",
                "tibia_length_r_m": "",
                "foot_length_r_m": "",
                "mass_out_kg": "",
                "status": "",
                "error_msg": "",
            }
            try:
                build_scale_setup(
                    subject_id=tag,
                    mass_kg=mass_kg,
                    height_cm=height_cm,
                    static_trc=str(static_path),
                    model_file=RAJAGOPAL_PATH,
                    output_model=str(out_model),
                    output_setup_xml=str(setup_xml),
                )

                # OpenSim's C++ logger goes through its own sink; redirect_stdout doesn't
                # capture it. Use addFileSink so we can read RMS lines from the log.
                sink_path = OUT_DIR / f"_{tag}_run.log"
                if sink_path.exists():
                    sink_path.unlink()
                opensim.Logger.removeFileSink()
                opensim.Logger.addFileSink(str(sink_path))
                opensim.Logger.setLevelString("Info")
                try:
                    tool = opensim.ScaleTool(str(setup_xml))
                    ok = tool.run()
                finally:
                    opensim.Logger.removeFileSink()
                    opensim.Logger.setLevelString("Warn")
                captured = sink_path.read_text(encoding="utf-8", errors="replace") if sink_path.is_file() else ""

                if not ok or not out_model.is_file():
                    row["status"] = "FAIL_TOOL"
                    row["error_msg"] = f"tool.run() returned {ok}, model file exists={out_model.is_file()}"
                    print(f"  ❌ {row['error_msg']}")
                    rows.append(row); continue

                mp_rms_m, mp_max_m, mp_max_marker = parse_rms_from_scale_log(captured)
                row["ms_rms_cm"] = "—"  # ModelScaler is purely geometric in OpenSim 4.5
                row["mp_rms_cm"] = f"{mp_rms_m*100:.3f}" if mp_rms_m is not None else "?"
                row["mp_max_cm"] = f"{mp_max_m*100:.3f}" if mp_max_m is not None else "?"
                row["mp_max_marker"] = mp_max_marker or "?"

                lens = compute_segment_lengths(out_model)
                for k in ("pelvis_width_m", "femur_length_r_m", "tibia_length_r_m", "foot_length_r_m"):
                    if k in lens:
                        row[k] = f"{lens[k]:.4f}"

                mass_out = total_mass(out_model)
                row["mass_out_kg"] = f"{mass_out:.3f}"

                mass_err = abs(mass_out - mass_kg)
                mp_flag = (mp_rms_m is None) or (mp_rms_m > 0.025)   # 2.5 cm
                mass_flag = mass_err > 0.1
                if mp_flag or mass_flag:
                    row["status"] = "FLAG"
                    row["error_msg"] = (
                        f"mp_rms={row['mp_rms_cm']}cm "
                        f"mp_max={row['mp_max_cm']}cm @ {row['mp_max_marker']}; "
                        f"|mass_out-mass_in|={mass_err:.3f}kg"
                    )
                else:
                    row["status"] = "OK"
                print(f"  status={row['status']} ms_rms={row['ms_rms_cm']}cm mp_rms={row['mp_rms_cm']}cm mass_out={row['mass_out_kg']}kg")
            except Exception as e:
                row["status"] = "FAIL_EXCEPTION"
                row["error_msg"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            rows.append(row)

    # ---- write log ----
    fieldnames = ["subject_id", "med", "static_trc_used",
                  "ms_rms_cm", "mp_rms_cm", "mp_max_cm", "mp_max_marker",
                  "pelvis_width_m", "femur_length_r_m", "tibia_length_r_m", "foot_length_r_m",
                  "mass_in_kg", "mass_out_kg",
                  "height_in_cm",
                  "status", "error_msg"]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=fieldnames)
    df.to_csv(LOG_PATH, index=False)
    elapsed = time.time() - t0

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_flag = sum(1 for r in rows if r["status"] == "FLAG")
    n_fail = sum(1 for r in rows if r["status"].startswith("FAIL"))
    print(f"\nDone. {n_ok} OK / {n_flag} FLAG / {n_fail} FAIL  in {elapsed:.1f}s")
    print(f"Log: {LOG_PATH}")
    return 0 if n_fail == 0 and n_flag <= 5 else (2 if n_flag > 5 else 1)


if __name__ == "__main__":
    sys.exit(main())
