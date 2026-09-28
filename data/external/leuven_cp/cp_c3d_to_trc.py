"""Step 1 — convert SimTK CP child-gait c3d (Vicon Plug-in-Gait) to OpenSim TRC.

Reads marker trajectories from a c3d via ezc3d, keeps the named markers (drops the
'*N' unlabeled ones), converts mm -> m, and writes a standard OpenSim .trc.

Used for BOTH the static-calibration trial (-> scale) and the walking trials (-> IK).
Marker coordinates are ROTATED from the Vicon lab frame (Z-up) into the OpenSim
convention (Y-up) below, as X_os = X_lab, Y_os = Z_lab, Z_os = -Y_lab, which keeps the
frame right-handed. This is not optional. Feeding Z-up markers to the Y-up Rajagopal
model makes IK lay the subject on their side and drives hip and knee to about 120 deg.

ezc3d runs in the Windows python (has ezc3d 1.7.0); the TRC is then consumed by
OpenSim in the WSL bmclab_opensim env. So this script is env-agnostic (only needs
ezc3d + numpy) and writes plain-text TRC that both sides can read.

Usage:
  python cp_c3d_to_trc.py <in.c3d> <out.trc>
  python cp_c3d_to_trc.py --all          # convert every gait + static c3d to trc/
ASCII-only stdout (Windows cp1252 safe).
"""
from __future__ import annotations
import sys
import glob
from pathlib import Path

import numpy as np
import ezc3d

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TRC_DIR = HERE / "trc"

# Lower-body Plug-in-Gait markers we need for scaling + IK (drop trunk/arm noise but
# keep a few trunk anchors for torso seating). These are the names that appear in the
# CP c3d (verified). Markers absent in a given trial are simply skipped.
PIG_KEEP = [
    # pelvis
    "RASI", "LASI", "RPSI", "LPSI", "SACR",
    # thigh / knee
    "RTHI", "RKNE", "LTHI", "LKNE",
    # shank / ankle
    "RTIB", "RANK", "LTIB", "LANK",
    # foot
    "RHEE", "RTOE", "LHEE", "LTOE",
    # trunk anchors (low IK weight, just to seat torso)
    "C7", "CLAV", "T10", "STRN",
    # shoulders (optional, very low weight)
    "RSHO", "LSHO",
]


def read_c3d_markers(path: str):
    """Return (labels_kept, data_m (nframe, n_kept, 3), rate). Handles duplicate /
    unlabeled markers by name (first occurrence wins; '*' dropped)."""
    c = ezc3d.c3d(path)
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    raw_labels = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    units = c["parameters"]["POINT"]["UNITS"]["value"][0].strip().lower()
    scale = 0.001 if units == "mm" else 1.0  # -> metres
    pts = c["data"]["points"]  # (4, n_marker, n_frame): x,y,z,residual
    nframe = pts.shape[2]

    kept_labels, kept_idx = [], []
    for name in PIG_KEEP:
        if name in raw_labels:
            kept_labels.append(name)
            kept_idx.append(raw_labels.index(name))
    if not kept_labels:
        raise RuntimeError(f"no PiG markers found in {path}")

    data = np.full((nframe, len(kept_labels), 3), np.nan, dtype=np.float64)
    for j, mi in enumerate(kept_idx):
        xyz = pts[:3, mi, :].T * scale  # (nframe, 3) in the LAB frame
        # ezc3d uses 0.0 for some gap fills; treat exact (0,0,0) as missing
        zero = np.all(xyz == 0.0, axis=1)
        xyz[zero] = np.nan
        # --- LAB-frame (Vicon Z-up) -> OpenSim (Y-up) rotation ---------------------
        # This CP dataset is Z-up (C7 vertical >> heel; gravity along lab Z), but the
        # Rajagopal model is Y-up (gravity [0,-9.8,0]). Feeding Z-up markers to a Y-up
        # model makes IK lay the subject on their side and slams hip/knee to ~120 deg
        # (the divergence we saw). Map: X_os = X_lab (forward), Y_os = Z_lab (up),
        # Z_os = -Y_lab (preserves right-handedness).
        x, y, z = xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy()
        xyz[:, 0] = x
        xyz[:, 1] = z
        xyz[:, 2] = -y
        data[:, j, :] = xyz
    return kept_labels, data, rate


def write_trc(out_path: str, labels, data_m, rate):
    """Write an OpenSim TRC. data_m: (nframe, nmarker, 3) in metres."""
    nframe, nmarker, _ = data_m.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(nframe) / rate
    with open(out_path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{out_path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:g}\t{rate:g}\t{nframe}\t{nmarker}\tm\t{rate:g}\t1\t{nframe}\n")
        # marker-name header row
        hdr = ["Frame#", "Time"]
        for name in labels:
            hdr += [name, "", ""]
        f.write("\t".join(hdr) + "\n")
        # X/Y/Z sub-header
        sub = ["", ""]
        for k in range(nmarker):
            sub += [f"X{k+1}", f"Y{k+1}", f"Z{k+1}"]
        f.write("\t".join(sub) + "\n")
        f.write("\n")
        # data rows
        for i in range(nframe):
            row = [str(i + 1), f"{times[i]:.5f}"]
            for j in range(nmarker):
                x, y, z = data_m[i, j, :]
                if np.isnan(x):
                    row += ["", "", ""]
                else:
                    row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
            f.write("\t".join(row) + "\n")


def convert_one(in_c3d: str, out_trc: str):
    labels, data, rate = read_c3d_markers(in_c3d)
    write_trc(out_trc, labels, data, rate)
    cov = float(np.mean(np.isfinite(data[:, :, 0])))
    print(f"  {Path(in_c3d).name:<14} -> {Path(out_trc).name:<22} "
          f"{data.shape[0]} frames, {len(labels)} markers, marker-coverage {cov:.2f}, rate {rate:g}")
    return labels


def convert_all():
    TRC_DIR.mkdir(parents=True, exist_ok=True)
    # gait trials -> trc/gait/<group>/<name>.trc ; static -> trc/static/<name>.trc
    n = 0
    print("=== STATIC trials ===")
    for f in sorted(glob.glob(str(RAW / "static" / "*.c3d"))):
        stem = Path(f).stem
        try:
            convert_one(f, str(TRC_DIR / "static" / f"{stem}.trc"))
            n += 1
        except Exception as e:
            print(f"  {stem}: SKIP ({type(e).__name__}: {e})")
    print("=== GAIT trials ===")
    for grp in ("td", "hecp", "dicp"):
        for f in sorted(glob.glob(str(RAW / "gait" / grp / "*.c3d"))):
            stem = Path(f).stem
            try:
                convert_one(f, str(TRC_DIR / "gait" / grp / f"{stem}.trc"))
                n += 1
            except Exception as e:
                print(f"  {stem}: SKIP ({type(e).__name__}: {e})")
    print(f"\nConverted {n} trials -> {TRC_DIR}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        convert_all()
    elif len(sys.argv) == 3:
        convert_one(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
