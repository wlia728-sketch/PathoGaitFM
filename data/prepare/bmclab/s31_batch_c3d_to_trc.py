"""Batch c3d → TRC for every walking + static trial in C3Dfiles.zip.

Strategy: stream from the zip — extract each c3d to a small temp file, call
c3d_to_trc, delete temp file. Peak disk overhead < 2 MB even though the zip
contains 4500+ entries totalling 2.5 GB.

Writes a CSV log: data/prepare/bmclab/work/trc/_batch_log.csv with one row per c3d.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marker_map as mm
from s31_c3d_to_trc import c3d_to_trc


ZIP_PATH = str(Path(__file__).resolve().parents[3] / "data/prepare/bmclab/raw/C3Dfiles.zip")
OUT_DIR = Path(__file__).resolve().parents[3] / "data/prepare/bmclab/work/trc"
LOG_PATH = OUT_DIR / "_batch_log.csv"


def derive_trc_name(zip_entry: str) -> str:
    """C3Dfiles/SUB01_off/SUB01_off_walk_1.c3d → SUB01_off_walk_1.trc"""
    return Path(zip_entry).stem + ".trc"


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(ZIP_PATH):
        print(f"ZIP not found: {ZIP_PATH}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(ZIP_PATH) as zf:
        c3d_entries = sorted(n for n in zf.namelist() if n.endswith(".c3d"))
        print(f"Found {len(c3d_entries)} .c3d entries in {ZIP_PATH}")

        rows = []
        n_ok = n_fail = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, entry in enumerate(c3d_entries, 1):
                trc_name = derive_trc_name(entry)
                trc_path = str(OUT_DIR / trc_name)
                row = {
                    "c3d_path": entry,
                    "trc_path": str(OUT_DIR / trc_name),
                    "n_frames": "",
                    "n_markers_written": "",
                    "n_markers_skipped": "",
                    "bmclab_missing": "",
                    "status": "",
                    "error_msg": "",
                }
                try:
                    tmp_path = os.path.join(tmpdir, os.path.basename(entry))
                    with zf.open(entry) as src, open(tmp_path, "wb") as dst:
                        dst.write(src.read())
                    stats = c3d_to_trc(tmp_path, trc_path, mm.BMCLAB_TO_RAJAGOPAL,
                                       duplicate_labels=mm.BMCLAB_DUPLICATE_LABELS,
                                       overwrite=True)
                    os.remove(tmp_path)
                    row["n_frames"] = stats["n_frames"]
                    row["n_markers_written"] = stats["n_markers_written"]
                    row["n_markers_skipped"] = stats["n_markers_skipped"]
                    row["bmclab_missing"] = ";".join(stats["bmclab_missing"])
                    row["status"] = "OK"
                    n_ok += 1
                except Exception as e:
                    row["status"] = "FAIL"
                    row["error_msg"] = f"{type(e).__name__}: {e}"
                    n_fail += 1
                    # show traceback for first 5 failures
                    if n_fail <= 5:
                        print(f"\n  FAIL ({n_fail}): {entry}")
                        traceback.print_exc(limit=3)
                rows.append(row)
                if i % 200 == 0 or i == len(c3d_entries):
                    elapsed = time.time() - t0
                    print(f"  [{i:5d}/{len(c3d_entries)}]  OK={n_ok}  FAIL={n_fail}  elapsed={elapsed:.1f}s")

        # ---- write log CSV ----
        fieldnames = ["c3d_path", "trc_path", "n_frames", "n_markers_written",
                      "n_markers_skipped", "bmclab_missing", "status", "error_msg"]
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    elapsed = time.time() - t0
    print(f"\nWrote log: {LOG_PATH}")
    print(f"Total c3d entries: {len(c3d_entries)}")
    print(f"  OK:    {n_ok}")
    print(f"  FAIL:  {n_fail}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
