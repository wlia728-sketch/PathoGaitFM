"""BMClab c3d → OpenSim TRC converter.

Reads markers **by integer index** in the c3d POINT block (Decision 5 from
Step 3.1 spec — `V_R.TT` is duplicated, so name lookup is unsafe). Applies
the BMCLAB → Rajagopal mapping from `marker_map.py`. Writes a TRC file in
the strict format OpenSim's Storage parser expects.

TRC format reference:
    Line 1: PathFileType  4  (X/Y/Z)  <filename>.trc
    Line 2: header keys (tab-separated)
    Line 3: header values (tab-separated)
    Line 4: Frame# <TAB> Time <TAB> M1 <TAB> <TAB> <TAB> M2 <TAB> <TAB> <TAB> ...
    Line 5: <empty> <empty> X1 <TAB> Y1 <TAB> Z1 <TAB> X2 <TAB> ...
    Line 6: <empty>
    Line 7..: data rows

This module is import-safe; running it as a script does nothing.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import ezc3d


def _find_first_index(labels: list, target: str) -> Optional[int]:
    """First index of `target` in `labels`, or None.
    Stripping whitespace because some c3d writers add trailing spaces."""
    for i, lbl in enumerate(labels):
        if str(lbl).strip() == target:
            return i
    return None


def c3d_to_trc(
    c3d_path: str,
    trc_path: str,
    marker_map: dict,
    duplicate_labels: Optional[set] = None,
    *,
    overwrite: bool = True,
) -> dict:
    """Convert one BMClab c3d to one TRC file.

    Returns a stats dict::

        {
            "n_frames":   int,
            "n_markers_written":  int,   # how many Rajagopal targets in TRC
            "n_markers_skipped":  int,   # BMClab labels not found in c3d, or mapped to None
            "bmclab_missing":     list,  # BMClab labels that were mapped but not in c3d
            "first_frame":        int,
            "last_frame":         int,
            "point_rate":         float,
            "trc_path":           str,
        }

    Marker access is strictly by **integer index**: we resolve each BMClab
    label to its first matching index in the c3d POINT-label list, then read
    `pts[:3, idx, :]`. Labels in `duplicate_labels` are read by first
    occurrence only (the Rajagopal map doesn't currently target any duplicate
    label, but we use the same index-first protocol throughout).
    """
    if os.path.exists(trc_path) and not overwrite:
        raise FileExistsError(trc_path)

    c = ezc3d.c3d(c3d_path)
    plabs = [str(x).strip() for x in c["parameters"]["POINT"]["LABELS"]["value"]]
    pts = c["data"]["points"]  # shape (4, n_markers, n_frames). axes 0..2 = xyz, 3 = residual.
    n_frames = pts.shape[2]
    point_rate = float(c["header"]["points"]["frame_rate"])
    first_frame = int(c["header"]["points"]["first_frame"])

    # Build the ordered list of (bmclab_label, rajagopal_name, c3d_idx) for markers we will write.
    # Preserves the BMClab label ordering inside marker_map (Python dicts are insertion-ordered).
    write_list = []           # list of (raj_name, c3d_idx)
    skipped_count = 0         # BMClab labels with target = None
    bmclab_missing = []       # BMClab labels with target != None but not found in c3d
    for bmclab_lbl, raj_name in marker_map.items():
        if raj_name is None:
            skipped_count += 1
            continue
        idx = _find_first_index(plabs, bmclab_lbl)
        if idx is None:
            bmclab_missing.append(bmclab_lbl)
            continue
        write_list.append((raj_name, idx))

    n_markers_written = len(write_list)

    # ---- assemble per-frame coordinate matrix (n_frames × 3*n_markers) ----
    coords = np.empty((n_frames, 3 * n_markers_written), dtype=float)
    for k, (_, idx) in enumerate(write_list):
        xyz = pts[:3, idx, :]            # (3, n_frames)
        coords[:, 3 * k:3 * k + 3] = xyz.T

    # ---- write TRC ----
    fname = os.path.basename(trc_path)
    os.makedirs(os.path.dirname(trc_path), exist_ok=True)

    # OpenSim parses TRC strictly tab-separated. NaN values are emitted as empty strings.
    # Header structure:
    #   PathFileType\t4\t(X/Y/Z)\t<filename>
    #   DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
    #   <values>
    #   Frame#\tTime\t<M1>\t\t\t<M2>\t\t\t...
    #   \t\tX1\tY1\tZ1\tX2\tY2\tZ2\t...
    #   <empty>
    #   <data rows>
    with open(trc_path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{fname}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(
            f"{point_rate:.2f}\t{point_rate:.2f}\t{n_frames}\t{n_markers_written}\tmm\t{point_rate:.2f}\t1\t{n_frames}\n"
        )
        # Frame# row: each marker name is followed by two empty columns (placeholders for Y/Z names)
        marker_row = ["Frame#", "Time"]
        for raj_name, _ in write_list:
            marker_row.extend([raj_name, "", ""])
        f.write("\t".join(marker_row) + "\n")
        # X/Y/Z sub-header: prefix two empty columns (Frame#, Time), then X1\tY1\tZ1\tX2\t...
        xyz_row = ["", ""]
        for k in range(n_markers_written):
            xyz_row.extend([f"X{k + 1}", f"Y{k + 1}", f"Z{k + 1}"])
        f.write("\t".join(xyz_row) + "\n")
        # OpenSim's Storage reader expects a blank line between sub-header and data.
        f.write("\n")
        # data rows
        for fi in range(n_frames):
            row = [str(fi + 1), f"{(fi / point_rate):.4f}"]
            for v in coords[fi]:
                # Emit explicit "NaN" for missing values, NOT empty string. OpenSim 4.5's
                # TRCFileAdapter is strict about trailing empty fields: a row whose last
                # marker(s) are missing gets parsed with the wrong column count and IK
                # fails ("Unexpected number of columns... Expected = 86. Received = 85").
                # "NaN" tokens are accepted natively and treated as missing data.
                row.append("NaN" if not np.isfinite(v) else f"{v:.5f}")
            f.write("\t".join(row) + "\n")

    return {
        "n_frames": n_frames,
        "n_markers_written": n_markers_written,
        "n_markers_skipped": skipped_count,
        "bmclab_missing": bmclab_missing,
        "first_frame": first_frame,
        "last_frame": first_frame + n_frames - 1,
        "point_rate": point_rate,
        "trc_path": trc_path,
    }


if __name__ == "__main__":
    print("c3d_to_trc is a library module. Import and call c3d_to_trc(...).", file=sys.stderr)
