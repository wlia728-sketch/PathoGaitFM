"""CoP computation v2: per-trial baseline subtraction (Option A) on top of v1.

Subtract the mean of the FIRST 0.5 s of each analog channel (window when the
subject is far from the plate strip — TRC tells us forward marker progress).
This corrects plate baseline drift that creates "always-active" plates.
"""
import os
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[k] = "1"
import sys, zipfile, tempfile
from pathlib import Path
import numpy as np
import ezc3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import v1 module without running its main
import importlib.util
spec = importlib.util.spec_from_file_location("_compute_cop_v1",
    os.path.join(os.path.dirname(__file__), "_compute_cop.py"))
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

PROJ = Path(__file__).resolve().parents[3]
ZIP = PROJ / "data/prepare/bmclab/raw/C3Dfiles.zip"


def subtract_baseline(c3d_obj, baseline_window_s: float = 0.5):
    """Subtract per-channel mean of the first `baseline_window_s` seconds from all
    analog channels in-place. Mutates c3d_obj['data']['analogs']."""
    a_rate = float(c3d_obj["header"]["analogs"]["frame_rate"])
    n_baseline = int(baseline_window_s * a_rate)
    anal = c3d_obj["data"]["analogs"][0]  # (n_analogs, n_frames)
    if anal.shape[1] < n_baseline:
        return  # too short
    means = np.mean(anal[:, :n_baseline], axis=1, keepdims=True)
    c3d_obj["data"]["analogs"][0] = anal - means
    return means


def main():
    with zipfile.ZipFile(ZIP) as zf, tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.c3d"
        with zf.open("C3Dfiles/SUB01_off/SUB01_off_walk_1.c3d") as src, open(out, "wb") as dst:
            dst.write(src.read())
        c = ezc3d.c3d(str(out))
        a_labels = [str(x).strip() for x in c["parameters"]["ANALOG"]["LABELS"]["value"]]
        means = subtract_baseline(c, baseline_window_s=0.5)
        print(f"Baseline-subtracted first 0.5 s mean from {len(a_labels)} analog channels.")
        print("  Channels with |baseline| > 5 N: ")
        for i, m in enumerate(means.flatten()):
            if abs(m) > 5:
                print(f"    {a_labels[i]:<8}  baseline = {m:+8.3f} (units depend on channel)")

        for p_idx in v1.REAL_PLATE_INDICES:
            print(f"\n--- plate idx {p_idx} (TYPE {int(c['parameters']['FORCE_PLATFORM']['TYPE']['value'][p_idx])}) ---")
            cop, tz, fz, f, active = v1.compute_plate_cop_and_tz(c, p_idx)
            n_active = int(active.sum())
            print(f"  n_analog_frames = {f.shape[1]}, n_active (|Fz_lab|>20N) = {n_active}")
            if n_active > 0:
                mask = active
                print(f"    Fz lab-Y peak = {np.nanmax(f[1, mask]):+.1f} N, min = {np.nanmin(f[1, mask]):+.1f} N")
                print(f"    CoP_x lab X range: {np.nanmin(cop[0, mask]):.1f}..{np.nanmax(cop[0, mask]):.1f} mm")
                print(f"    CoP_z lab Z range: {np.nanmin(cop[2, mask]):.1f}..{np.nanmax(cop[2, mask]):.1f} mm")
                corners = np.asarray(c["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"])[:, :, p_idx]
                cx_min, cx_max = corners[0, :].min(), corners[0, :].max()
                cz_min, cz_max = corners[2, :].min(), corners[2, :].max()
                print(f"    plate corners X: {cx_min:.1f}..{cx_max:.1f}   Z: {cz_min:.1f}..{cz_max:.1f}")
                within_x = (cop[0, mask] >= cx_min) & (cop[0, mask] <= cx_max)
                within_z = (cop[2, mask] >= cz_min) & (cop[2, mask] <= cz_max)
                pct_within = float((within_x & within_z).sum() / n_active * 100)
                print(f"    within plate corners (XZ): {pct_within:.1f}%")
            else:
                print("    (no active frames)")


if __name__ == "__main__":
    main()
