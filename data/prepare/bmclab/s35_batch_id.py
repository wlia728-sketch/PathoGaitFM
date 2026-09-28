"""Batch Inverse Dynamics — one .sto per trial.

For each of 885 trials with IK output:
  1. Look up subject + medication state from the filename.
  2. Find the per-medication scaled IK model + IK .mot + external_loads.xml.
  3. Run InverseDynamicsTool, write to data/prepare/bmclab/work/id/{trial}_id.sto.
  4. Log status to _id_log.csv.

Pool of 12 workers, single-thread SimTK (mirrors Step 3.3 config).
"""
from __future__ import annotations

# Thread-count pinning BEFORE opensim import — required for the 12-worker pool to scale
# (see s33_ik_setup_builder.py docstring for rationale).
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
import traceback
from pathlib import Path

multiprocessing.set_start_method("fork", force=True)

import opensim
opensim.Logger.setLevelString("Warn")
print(f"[{os.getpid()}] OMP={os.environ.get('OMP_NUM_THREADS')} "
      f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
      f"SIMTK={os.environ.get('SIMTK_NUM_THREADS')}", flush=True)


PROJ = Path(__file__).resolve().parents[3]
IK_DIR = PROJ / "data/prepare/bmclab/work/ik"
GRF_DIR = PROJ / "data/prepare/bmclab/work/grf"
SCALED_DIR = PROJ / "data/prepare/bmclab/work/scaled_models"
ID_DIR = PROJ / "data/prepare/bmclab/work/id"
ID_LOG = ID_DIR / "_id_log.csv"


def med_for_trial(stem: str):
    """Extract (subject, med, trial_base) from a trial stem (handles Trimmed_ prefix)."""
    if stem.endswith("_ik"):
        stem = stem[:-3]
    toks = stem.split("_")
    if toks[0] == "Trimmed":
        return toks[1], toks[2], stem
    return toks[0], toks[1], stem


def id_worker(args: dict) -> dict:
    """Run ID for a single trial. Top-level for pickling."""
    sid = args["subject"]
    med = args["med"]
    trial = args["trial"]
    scaled = args["scaled_model"]
    ik_mot = args["ik_mot"]
    ext_xml = args["ext_xml"]
    out_sto = args["out_sto"]
    setup_xml = args["setup_xml"]

    row = {
        "trial": trial, "subject": sid, "med": med,
        "scaled_model": Path(scaled).name,
        "ik_mot": Path(ik_mot).name,
        "ext_loads_xml": Path(ext_xml).name,
        "out_sto": Path(out_sto).name,
        "elapsed_s": "",
        "status": "",
        "error_msg": "",
    }
    # Verify inputs exist (some trials may have IK but no GRF, or vice versa)
    if not Path(scaled).is_file():
        row["status"] = "FAIL_NO_MODEL"
        row["error_msg"] = f"missing {scaled}"
        return row
    if not Path(ik_mot).is_file():
        row["status"] = "FAIL_NO_IK"
        row["error_msg"] = f"missing {ik_mot}"
        return row
    if not Path(ext_xml).is_file():
        # No GRF file at all — skip ID (we can't compute moments without external loads)
        row["status"] = "SKIP_NO_EXT_LOADS"
        return row

    import opensim
    opensim.Logger.setLevelString("Warn")
    try:
        t0 = time.time()
        tool = opensim.InverseDynamicsTool()
        tool.setModelFileName(str(scaled))
        tool.setCoordinatesFileName(str(ik_mot))
        tool.setExternalLoadsFileName(str(ext_xml))
        # OpenSim quirk: setOutputGenForceFileName expects a filename ONLY (no
        # directory). The output directory comes from setResultsDir().
        tool.setResultsDir(str(Path(out_sto).parent))
        tool.setOutputGenForceFileName(Path(out_sto).name)
        tool.setLowpassCutoffFrequency(6.0)
        tool.printToXML(str(setup_xml))
        ok = tool.run()
        dt = time.time() - t0
        row["elapsed_s"] = f"{dt:.2f}"
        if not ok or not Path(out_sto).is_file():
            row["status"] = "FAIL_TOOL"
            row["error_msg"] = f"ok={ok}, sto exists={Path(out_sto).is_file()}"
            return row
        row["status"] = "OK"
    except Exception as e:
        row["status"] = "FAIL_EXCEPTION"
        row["error_msg"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return row


def list_ik_mots():
    """All IK output stems in data/prepare/bmclab/work/ik/."""
    pat = re.compile(r"^(Trimmed_)?SUB\d+_(off|on)_walk_\w+_ik\.mot$")
    out = []
    for p in IK_DIR.iterdir():
        if pat.match(p.name):
            out.append(p)
    return sorted(out)


def main():
    ID_DIR.mkdir(parents=True, exist_ok=True)
    nproc = 12
    print(f"Using {nproc} worker processes for ID pool.")

    ik_mots = list_ik_mots()
    print(f"Found {len(ik_mots)} IK .mot files to process.")

    # Build work list
    work = []
    for ik_mot in ik_mots:
        stem = ik_mot.stem  # e.g. SUB05_on_walk_8_ik
        sub, med, trial_base = med_for_trial(stem)
        # If excluded entry (no GRF, no model), skip
        scaled = SCALED_DIR / f"{sub}_{med}_ik.osim"
        ext_xml = GRF_DIR / f"{trial_base}_external_loads.xml"
        out_sto = ID_DIR / f"{trial_base}_id.sto"
        setup_xml = ID_DIR / f"_{trial_base}_id_setup.xml"
        work.append({
            "subject": sub, "med": med, "trial": trial_base,
            "scaled_model": str(scaled),
            "ik_mot": str(ik_mot),
            "ext_xml": str(ext_xml),
            "out_sto": str(out_sto),
            "setup_xml": str(setup_xml),
        })

    # Write header
    fields = ["trial","subject","med","scaled_model","ik_mot","ext_loads_xml",
              "out_sto","elapsed_s","status","error_msg"]
    with open(ID_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    t0 = time.time()
    n_total = len(work)
    n_ok = n_skip = n_fail = 0
    with multiprocessing.Pool(nproc) as pool:
        for i, row in enumerate(pool.imap_unordered(id_worker, work, chunksize=4), 1):
            with open(ID_LOG, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow({k: row.get(k, "") for k in fields})
            if row["status"] == "OK":
                n_ok += 1
            elif row["status"].startswith("SKIP"):
                n_skip += 1
            else:
                n_fail += 1
            if i % 100 == 0 or i == n_total:
                elapsed = time.time() - t0
                print(f"  [{i:>4}/{n_total}]  OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  OK:   {n_ok}")
    print(f"  SKIP: {n_skip}")
    print(f"  FAIL: {n_fail}")
    print(f"  Log:  {ID_LOG}")
    return 0 if n_fail == 0 else (2 if n_fail > 5 else 1)


if __name__ == "__main__":
    sys.exit(main())
