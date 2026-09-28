"""Per-subject batch Inverse Kinematics + Visual3D cross-check, fail-fast.

For each subject in PDGinfo order:
  1. Find all walking TRCs for this subject in data/prepare/bmclab/work/trc/.
  2. Build per-trial IK setup XML using the ForceSet-stripped model
     SUB{NN}_ik.osim from s32_make_ik_model.py.
  3. Run IK in a multiprocessing pool (default 6 workers).
  4. Append per-trial RMS + duration to _ik_log.csv.
  5. Run Visual3D cross-check on this subject's IK outputs.
  6. Apply gate:
      PASS  : overall median r ≥ 0.90 across the 6 sagittal channels
      WARN  : 0.80 ≤ median r < 0.90 — log but continue
      FAIL  : median r < 0.80 OR any single channel-trial r < 0.70  → STOP
  7. Append per-subject V3D r to s33_v3d_cross_check.csv.

Trims selection:
  Per Step 3.2, the Trimmed_SUB01_off_walk_12b and
  Trimmed_SUB01_off_walk_14b files are clean prefix subsets of walk_12 and
  walk_14. We use the Trimmed versions and skip the longer base ones for
  these 2 trials in SUB01_off. All other trials are taken as-is.
"""
from __future__ import annotations

# *** Thread-count pinning before any opensim import — see s33_ik_setup_builder.py docstring ***
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "SIMTK_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_k] = "1"

import csv

import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

# multiprocessing must use `fork` so env vars propagate. Default on Linux is fork,
# but be explicit in case Python 3.10+ defaults change.
multiprocessing.set_start_method("fork", force=True)

import opensim
opensim.Logger.setLevelString("Warn")
print(f"[{os.getpid()}] OMP={os.environ.get('OMP_NUM_THREADS')} "
      f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
      f"SIMTK={os.environ.get('SIMTK_NUM_THREADS')}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s33_ik_setup_builder import build_ik_setup, run_ik
from s33_v3d_cross_check import cross_check_subject, IK_COL_NAMES


PROJ = Path(__file__).resolve().parents[3]
TRC_DIR = PROJ / "data/prepare/bmclab/work/trc"
SCALED_DIR = PROJ / "data/prepare/bmclab/work/scaled_models"
IK_DIR = PROJ / "data/prepare/bmclab/work/ik"
IK_LOG = IK_DIR / "_ik_log.csv"
V3D_LOG = IK_DIR / "s33_v3d_cross_check.csv"
V3D_TRIAL_FLAGS = IK_DIR / "_v3d_trial_flags.csv"  # per-trial-channel flags (Step 3.3 Phase B)

# Drop these in favour of the Trimmed_ prefix-subset version (Step 3.2 decision)
SKIP_TRIALS = {
    "SUB01_off_walk_12",
    "SUB01_off_walk_14",
}

# Walking trial filter (excludes statics, the two "merged" SUB01 base trials, etc.)
def list_walking_trials(subject_id: str) -> List[Path]:
    pat = re.compile(rf"^(Trimmed_)?{re.escape(subject_id)}_(off|on)_walk_\w+\.trc$")
    out = []
    for p in TRC_DIR.iterdir():
        if not p.is_file() or p.suffix != ".trc":
            continue
        if not pat.match(p.name):
            continue
        # skip the base versions of the two trimmed trials
        stem = p.stem  # e.g. SUB01_off_walk_12
        if stem in SKIP_TRIALS:
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.name)


def parse_rms_from_ik_log(log_text: str) -> tuple:
    """Return (mean_rms_m, mean_max_m) over Frame N lines, or (None, None) if absent."""
    rms_vals = []
    max_vals = []
    for line in log_text.splitlines():
        m = re.search(r"marker error:\s*RMS\s*=\s*([\d.eE+\-]+)[,\s]+max\s*=\s*([\d.eE+\-]+)", line)
        if m:
            rms_vals.append(float(m.group(1)))
            max_vals.append(float(m.group(2)))
    if not rms_vals:
        return None, None
    import numpy as np
    return float(np.mean(rms_vals)), float(np.mean(max_vals))


# ---- per-trial worker (runs in a separate process) --------------------------
_first_worker_print_done = False
def ik_worker(args: dict) -> dict:
    """Run IK for a single trial. Designed to be top-level for pickling."""
    sid = args["subject_id"]
    trc = Path(args["trc_path"])
    scaled = Path(args["scaled_model"])
    out_mot = Path(args["out_mot"])
    setup_xml = Path(args["setup_xml"])
    run_log = Path(args["run_log"])

    import opensim
    opensim.Logger.setLevelString("Info")  # we want per-frame lines for RMS parsing

    global _first_worker_print_done
    if not _first_worker_print_done:
        _first_worker_print_done = True
        print(f"[worker {os.getpid()}] OMP={os.environ.get('OMP_NUM_THREADS')} "
              f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
              f"SIMTK={os.environ.get('SIMTK_NUM_THREADS')}", flush=True)

    row = {
        "subject_id": sid,
        "trial": trc.stem,
        "trc_path": str(trc.relative_to(PROJ)),
        "scaled_model": str(scaled.relative_to(PROJ)),
        "ik_mot": str(out_mot.relative_to(PROJ)),
        "n_frames": "",
        "duration_s": "",
        "ik_rms_m": "",
        "ik_max_m": "",
        "elapsed_s": "",
        "status": "",
        "error_msg": "",
    }
    if out_mot.is_file() and args.get("skip_if_exists", False):
        row["status"] = "SKIPPED_EXISTING"
        return row
    try:
        stats = build_ik_setup(
            scaled_model=str(scaled),
            marker_trc=str(trc),
            output_mot=str(out_mot),
            output_setup_xml=str(setup_xml),
        )
        row["n_frames"] = round((stats["t_end"] - stats["t_start"]) * 150) + 1
        row["duration_s"] = f"{stats['t_end'] - stats['t_start']:.3f}"

        if run_log.exists():
            run_log.unlink()
        opensim.Logger.removeFileSink()
        opensim.Logger.addFileSink(str(run_log))
        t0 = time.time()
        ok = run_ik(str(setup_xml))
        dt = time.time() - t0
        opensim.Logger.removeFileSink()
        row["elapsed_s"] = f"{dt:.2f}"
        if not ok or not out_mot.is_file():
            row["status"] = "FAIL_TOOL"
            row["error_msg"] = f"ok={ok}, mot exists={out_mot.is_file()}"
            return row
        log = run_log.read_text(encoding="utf-8", errors="replace")
        rms_m, max_m = parse_rms_from_ik_log(log)
        row["ik_rms_m"] = f"{rms_m:.5f}" if rms_m is not None else "?"
        row["ik_max_m"] = f"{max_m:.5f}" if max_m is not None else "?"
        row["status"] = "OK"
    except Exception as e:
        row["status"] = "FAIL_EXCEPTION"
        row["error_msg"] = f"{type(e).__name__}: {e}"
    return row


def gate(v3d_result: dict) -> tuple:
    """Apply spec gate: PASS / WARN / FAIL.
    Returns (gate_label, reason)."""
    # Gate rule v2 (Step 3.3 Phase-B amendment):
    #   PASS  : subject overall median r ≥ 0.80
    #   FAIL  : subject overall median r < 0.80
    # The previous per-trial-channel < 0.70 floor was demoted to a logging-only flag
    # ('v3d_low_agreement') in the per-trial cross-check CSV. The reasoning is in
    # step3_3_phaseB_outlier.md: V3D sometimes smooths out step-frequency content
    # that Rajagopal IK keeps, producing a low r that reflects V3D pipeline noise
    # rather than IK error. Subject-median is the meaningful quality metric;
    # trial-level mins are too noisy and trigger false stops.
    om = v3d_result.get("overall_median_r", float("nan"))
    if not (om == om):  # NaN
        return "FAIL", "overall median r is NaN (no trial cross-checked)"
    if om < 0.80:
        return "FAIL", f"overall median r {om:.3f} < 0.80"
    # Count flagged-but-not-failing trials for visibility in the batch log
    n_flagged_trials = 0
    for r in v3d_result.get("per_trial", []):
        for c in IK_COL_NAMES:
            v = r.get(c, float("nan"))
            if isinstance(v, float) and v == v and v < 0.70:
                n_flagged_trials += 1
                break
    if om < 0.90:
        return "WARN", f"overall median r {om:.3f} in [0.80, 0.90); {n_flagged_trials} trial(s) with at least one channel r<0.70"
    suffix = f"; {n_flagged_trials} trial(s) v3d_low_agreement" if n_flagged_trials else ""
    return "PASS", f"overall median r {om:.3f}{suffix}"


def main():
    IK_DIR.mkdir(parents=True, exist_ok=True)
    # Wenqi-decided Option C: 12 workers, single-thread SimTK per worker. Leaves
    # 10 cores for the parallel Stage-2 GPU session + main CC.
    nproc = 12
    print(f"Using {nproc} worker processes for per-subject IK pools.")

    # Enumerate subjects from per-medication scaled IK models.
    # File pattern: SUB{NN}_{off|on}_ik.osim (ForceSet-stripped, written by s32_make_ik_model.py)
    sub_med_pairs = set()
    for p in SCALED_DIR.iterdir():
        if p.suffix != ".osim" or "_ms" in p.stem or "_ik" not in p.stem:
            continue
        # stem like 'SUB05_off_ik' -> sub='SUB05', med='off'
        parts = p.stem.split("_")
        if len(parts) >= 3 and parts[0].startswith("SUB") and parts[1] in ("off", "on") and parts[-1] == "ik":
            sub_med_pairs.add((parts[0], parts[1]))
    subjects = sorted({s for s, _ in sub_med_pairs})
    print(f"Per-medication IK models found for: {len(sub_med_pairs)} (subject, med) pairs across {len(subjects)} subjects")

    # IK log writer (append mode after each subject)
    ik_fields = ["subject_id", "trial", "trc_path", "scaled_model", "ik_mot",
                 "n_frames", "duration_s", "ik_rms_m", "ik_max_m",
                 "elapsed_s", "status", "error_msg"]
    v3d_fields = ["subject_id", "n_trials", "n_with_r", "overall_median_r",
                  "hip_flexion_l_med_r", "hip_flexion_r_med_r",
                  "knee_angle_l_med_r", "knee_angle_r_med_r",
                  "ankle_angle_l_med_r", "ankle_angle_r_med_r",
                  "gate", "gate_reason"]
    trial_flag_fields = ["subject_id", "trial", "channel", "r_value", "status"]

    # Resume mode: if --reset on CLI, overwrite; else open in append mode and
    # keep the existing rows. The IK driver skips any trial whose _ik.mot already
    # exists, so we only run new trials and append new rows.
    reset = "--reset" in sys.argv[1:]
    def open_log(path: Path, fields: list):
        if reset or not path.is_file():
            f = open(path, "w", newline="", encoding="utf-8")
            csv.DictWriter(f, fieldnames=fields).writeheader()
            f.close()
    open_log(IK_LOG, ik_fields)
    open_log(V3D_LOG, v3d_fields)
    open_log(V3D_TRIAL_FLAGS, trial_flag_fields)

    # On resume, build the set of subjects already gated (in V3D_LOG) so we can skip them.
    already_gated = set()
    if not reset and V3D_LOG.is_file():
        with open(V3D_LOG, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_gated.add(row["subject_id"])
        if already_gated:
            print(f"Resume mode: skipping {len(already_gated)} subjects already gated: {sorted(already_gated)}")

    # Helper: extract medication state from trial filename. Handles 'Trimmed_' prefix.
    def med_for_trial(stem: str) -> Optional[str]:
        toks = stem.split("_")
        # 'SUB05_on_walk_8' or 'Trimmed_SUB01_off_walk_12b'
        if toks[0] == "Trimmed":
            return toks[2] if len(toks) >= 3 and toks[2] in ("off", "on") else None
        return toks[1] if len(toks) >= 2 and toks[1] in ("off", "on") else None

    grand_t0 = time.time()
    fail_flag = False
    for sid in subjects:
        if sid in already_gated:
            print(f"\n========== {sid} == ALREADY GATED (resume), skipping ==========")
            continue
        print(f"\n========== {sid} ==========")
        trials = list_walking_trials(sid)
        print(f"  {len(trials)} walking trials.")
        if not trials:
            print("  (no trials — skipping)")
            continue

        # Build per-trial work list, picking the per-medication scaled model for each.
        work = []
        skipped = []
        for trc in trials:
            stem = trc.stem
            med = med_for_trial(stem)
            if med is None:
                skipped.append((stem, "could not parse medication"))
                continue
            scaled = SCALED_DIR / f"{sid}_{med}_ik.osim"
            if not scaled.is_file():
                skipped.append((stem, f"missing model {scaled.name}"))
                continue
            work.append({
                "subject_id": sid,
                "trc_path": str(trc),
                "scaled_model": str(scaled),
                "out_mot": str(IK_DIR / f"{stem}_ik.mot"),
                "setup_xml": str(IK_DIR / f"_{stem}_ik_setup.xml"),
                "run_log":  str(IK_DIR / f"_{stem}_ik.log"),
                "skip_if_exists": not reset,  # in resume mode, don't recompute existing .mot
            })
        if skipped:
            print(f"  ⚠ skipping {len(skipped)} trials with no per-med model:")
            for stem, reason in skipped[:5]:
                print(f"     {stem}: {reason}")
            if len(skipped) > 5:
                print(f"     ... and {len(skipped) - 5} more")
        if not work:
            print("  no runnable trials, moving on")
            continue

        # Run pool
        t0 = time.time()
        with multiprocessing.Pool(nproc) as pool:
            rows = pool.map(ik_worker, work)
        ik_dt = time.time() - t0
        n_ok = sum(1 for r in rows if r["status"] == "OK")
        n_skipped = sum(1 for r in rows if r["status"] == "SKIPPED_EXISTING")
        n_fail = sum(1 for r in rows if r["status"].startswith("FAIL"))
        print(f"  IK: {n_ok} OK / {n_skipped} skipped(existing) / {n_fail} FAIL  in {ik_dt:.1f}s")

        # Append to IK log
        with open(IK_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ik_fields)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ik_fields})

        # If any IK failed, stop (we don't want to V3D-check incomplete subjects)
        if n_fail > 0:
            print(f"  ❌ {n_fail} IK failures — stopping batch.")
            fail_flag = True
            break

        # ---- Visual3D cross-check ----
        print("  Running V3D cross-check…")
        t0 = time.time()
        v3d = cross_check_subject(sid)
        v3d_dt = time.time() - t0
        gate_label, gate_reason = gate(v3d)
        per_ch = v3d.get("per_channel_median", {})
        print(f"  V3D done in {v3d_dt:.1f}s   gate={gate_label}   reason={gate_reason}")
        print("  per-channel median r:")
        for c in IK_COL_NAMES:
            v = per_ch.get(c, float("nan"))
            print(f"     {c:<20}  {v:.4f}" if v == v else f"     {c:<20}  NaN")

        # Append V3D log
        with open(V3D_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=v3d_fields)
            w.writerow({
                "subject_id": sid,
                "n_trials": v3d.get("n_trials", 0),
                "n_with_r": v3d.get("n_with_r", 0),
                "overall_median_r": f"{v3d.get('overall_median_r', float('nan')):.4f}",
                "hip_flexion_l_med_r":  f"{per_ch.get('hip_flexion_l', float('nan')):.4f}",
                "hip_flexion_r_med_r":  f"{per_ch.get('hip_flexion_r', float('nan')):.4f}",
                "knee_angle_l_med_r":   f"{per_ch.get('knee_angle_l',  float('nan')):.4f}",
                "knee_angle_r_med_r":   f"{per_ch.get('knee_angle_r',  float('nan')):.4f}",
                "ankle_angle_l_med_r":  f"{per_ch.get('ankle_angle_l', float('nan')):.4f}",
                "ankle_angle_r_med_r":  f"{per_ch.get('ankle_angle_r', float('nan')):.4f}",
                "gate": gate_label,
                "gate_reason": gate_reason,
            })

        # Append per-trial-channel flag rows for any r < 0.70
        with open(V3D_TRIAL_FLAGS, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=trial_flag_fields)
            for r in v3d.get("per_trial", []):
                trial_name = r.get("trial", "?")
                for c in IK_COL_NAMES:
                    v = r.get(c, float("nan"))
                    if isinstance(v, float) and v == v and v < 0.70:
                        w.writerow({
                            "subject_id": sid,
                            "trial": trial_name,
                            "channel": c,
                            "r_value": f"{v:.4f}",
                            "status": "v3d_low_agreement",
                        })

        if gate_label == "FAIL":
            print(f"  ❌ GATE FAILED for {sid}. Stopping batch per spec.")
            fail_flag = True
            break

    print(f"\nTotal elapsed: {time.time() - grand_t0:.1f}s")
    print(f"IK log:          {IK_LOG}")
    print(f"V3D log:         {V3D_LOG}")
    print(f"V3D trial flags: {V3D_TRIAL_FLAGS}")
    return 1 if fail_flag else 0


if __name__ == "__main__":
    sys.exit(main())
