"""Definitive cycle-count audit for Stage-1, Stage-2, and external sets.
Resolves each split file_id to its actual .npy on disk (searches both data roots),
loads the array, and counts cycles (= shape[0]). ASCII output -> JSON.

Stage-1: data/splits/v4_stage1_split.json   (entries = [source, file_id])
Stage-2: data/splits/v4_stage2_split_alltrain.json  (cv5 -> per-fold [source, stem])
External: extracted .npy under data/external/*/ (resolved per-site).
"""

# EXECUTION GUARD (added 2026-07-26): this file has no main() -- its whole body runs on execution and
# writes its result. Importing it for a helper would run the audit, so importing is refused. Run it directly.
if __name__ != "__main__":
    raise ImportError(__file__ + " is a script, not a module. Run it directly instead of importing it.")
import argparse
import sys
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# Candidate roots where a source's .npy files may live.
ADDBIO_ROOT = ROOT / "data" / "pretraining" / "addbio"
COHORT_ROOTS = [ROOT / "data" / "cohorts_raw", ROOT / "data" / "cohorts_grffix"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage1-split", type=Path, default=ROOT / "data/splits/v4_stage1_split.json")
parser.add_argument("--stage2-split", type=Path, default=ROOT / "data/splits/v4_stage2_split_alltrain.json")
parser.add_argument("--out", type=Path, default=ROOT / "outputs/cycle_counts.json")
args = parser.parse_args()
for path in (args.stage1_split, args.stage2_split):
    if not path.is_file():
        raise SystemExit(f"{path} not found; the split files are built locally (docs/training.md)")


def resolve_npy(source, fid):
    """Return Path to the .npy for (source, fid), or None."""
    # addbio files are FLAT in addbio/ with the study prefix already in the fid:
    #   e.g. fid = "Camargo2021_AB23_split0" -> addbio/Camargo2021_AB23_split0.npy
    if source.startswith("addbio_"):
        p = ADDBIO_ROOT / f"{fid}.npy"
        if p.exists():
            return p
        # fallback: per-study subdir layout
        study = source[len("addbio_"):]
        for cand in (ADDBIO_ROOT / study / f"{fid}.npy",
                     ADDBIO_ROOT / study / f"{study}_{fid}.npy"):
            if cand.exists():
                return cand
        return None
    # cohort sources -> data/cohorts_raw[ or cohorts_grffix]/<source>/<fid>.npy
    for base in COHORT_ROOTS:
        p = base / source / f"{fid}.npy"
        if p.exists():
            return p
    return None


def count_cycles(npy_path):
    try:
        arr = np.load(str(npy_path), mmap_mode="r")
        return int(arr.shape[0])
    except Exception as exc:
        print("  SKIPPED %s: %s" % (npy_path.name, type(exc).__name__), flush=True)
        return None


def audit_split_listpairs(entries):
    """entries = list of [source, fid]. Returns per-source files/cycles + missing."""
    files = defaultdict(int)
    cycles = defaultdict(int)
    missing = []
    for source, fid in entries:
        files[source] += 1
        p = resolve_npy(source, fid)
        if p is None:
            missing.append([source, fid]); continue
        n = count_cycles(p)
        if n is None:
            missing.append([source, fid]); continue
        cycles[source] += n
    return dict(files), dict(cycles), missing


out = {}

# ---------- STAGE 1 ----------
s1 = json.load(open(args.stage1_split, encoding="utf-8"))
out["stage1"] = {"counts_files": {k: s1[k] for k in ("train_count", "val_count", "test_count")}}
for split in ("train", "val", "test"):
    f, c, miss = audit_split_listpairs(s1["splits"][split])
    out["stage1"][split] = {
        "by_source_files": f,
        "by_source_cycles": c,
        "total_files": sum(f.values()),
        "total_cycles": int(sum(c.values())),
        "n_missing": len(miss),
        "missing_sample": miss[:5],
    }
out["stage1"]["TOTAL_train_cycles"] = out["stage1"]["train"]["total_cycles"]
out["stage1"]["TOTAL_all_cycles"] = sum(out["stage1"][s]["total_cycles"] for s in ("train", "val", "test"))

# ---------- STAGE 2 ----------
s2 = json.load(open(args.stage2_split, encoding="utf-8"))
out["stage2"] = {"top_keys": list(s2.keys())}
# stage2 unique (source, stem) across all cv5 folds (train+val) = full cohort cycle pool
uniq = set()
def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for e in o:
            if isinstance(e, (list, tuple)) and len(e) == 2 and all(isinstance(x, str) for x in e):
                uniq.add((e[0], e[1]))
            else:
                walk(e)
walk(s2)
f2 = defaultdict(int); c2 = defaultdict(int); miss2 = []
for source, fid in sorted(uniq):
    f2[source] += 1
    p = resolve_npy(source, fid)
    if p is None:
        miss2.append([source, fid]); continue
    n = count_cycles(p)
    if n is None:
        miss2.append([source, fid]); continue
    c2[source] += n
out["stage2"]["unique_files"] = len(uniq)
out["stage2"]["by_source_files"] = dict(f2)
out["stage2"]["by_source_cycles"] = dict(c2)
out["stage2"]["total_cycles"] = int(sum(c2.values()))
out["stage2"]["n_missing"] = len(miss2)
out["stage2"]["missing_sample"] = miss2[:8]

# ---------- EXTERNAL ----------
ext = {}
for site, label in [("fukuchi_healthy", "Fukuchi"),
                    ("pd_nordic", "PD_Nordic")]:
    base = ROOT / "data" / "external" / site
    npys = [p for p in base.rglob("*.npy") if "_mask" not in p.name]
    tot = 0; nfiles = 0
    for p in npys:
        n = count_cycles(p)
        if n is not None:
            tot += n; nfiles += 1
    ext[label] = {"dir": str(base), "n_npy_files": nfiles, "total_cycles": int(tot)}
out["external"] = ext

# Fail closed: without the cycle arrays every count is zero and must not be written.
_s1 = out["stage1"]["TOTAL_train_cycles"]; _s2 = out["stage2"]["total_cycles"]
if not _s1 or not _s2:
    raise SystemExit("refusing to write %s: stage1 train=%s stage2=%s. The cycle arrays are absent, "
                     "so these counts are not real." % (args.out, _s1, _s2))

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("STAGE1 train cycles:", out["stage1"]["TOTAL_train_cycles"], "missing:", out["stage1"]["train"]["n_missing"])
print("STAGE1 all cycles  :", out["stage1"]["TOTAL_all_cycles"])
print("STAGE2 cycles      :", out["stage2"]["total_cycles"], "missing:", out["stage2"]["n_missing"], "files:", out["stage2"]["unique_files"])
print("EXTERNAL           :", {k: ext[k]["total_cycles"] for k in ext})
