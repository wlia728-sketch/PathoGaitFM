"""Validate result payloads and protect archived results from producer writes.

Legacy results/ destinations are redirected to outputs/recomputed/. Empty
metric tables, non-finite scores and zero-only payloads fail before writing.
Valid empty audit fields, such as missing_sample or fold-overlap reports,
remain allowed. Run ``python scripts/lib/guarded_write.py`` for the self-test.

PATHOGAIT_ALLOW_DEGENERATE=1 explicitly permits partial diagnostic payloads;
it never disables archived-result protection.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

_OVERRIDE = "PATHOGAIT_ALLOW_DEGENERATE"
ROOT = Path(__file__).resolve().parents[2]


def safe_artifact_path(path, root=None):
    """Redirect legacy archived-result destinations into outputs/recomputed.

    Explicit destinations outside results/, including results/recomputed/, are
    preserved. The degenerate-payload override never disables archive protection.
    """
    root = Path(root or ROOT).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root / "results")
    except ValueError:
        return path
    if relative.parts and relative.parts[0] == "recomputed":
        return path
    return root / "outputs/recomputed" / relative

# A key whose value is a measured or computed quantity. Null here means the computation produced
# nothing, which is the signature of a run with no data.
_METRIC_KEY = re.compile(
    r"(pcc|corr|macro|median|mean|rate|rmse|nrmse|ratio|delta|gain|score|p_?value|ci9?5?|"
    r"n_subj|n_subjects|n_cycles|n_trials|count|survival|coverage|spread|peak)", re.I)

# A key whose value is a table of results. Empty here means nothing was scored.
_TABLE_KEY = re.compile(
    r"(per_subject|per_cohort|per_channel|per_limb|per_fold|per_trial|cohort_|subject_|"
    r"^sites$|^results$|^folds$|^subjects$|^scenarios$|^arms$|^probes$)", re.I)


def _leaves(obj, path=""):
    """Every leaf as (json-path, key, value), and every empty container the same way."""
    out = []
    if isinstance(obj, dict):
        if not obj:
            out.append((path or "/", path.rsplit("/", 1)[-1], obj))
        for k, v in obj.items():
            out += _leaves(v, "%s/%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        if not obj:
            out.append((path or "/", path.rsplit("/", 1)[-1], list(obj)))
        for i, v in enumerate(obj):
            out += _leaves(v, "%s[%d]" % (path, i))
    else:
        out.append((path or "/", path.rsplit("/", 1)[-1], obj))
    return out


def _diagnose(payload):
    """Every reason this payload looks like a run that found no data."""
    if not payload:
        return ["the payload is empty"]

    problems = []
    leaves = _leaves(payload)
    numbers = [v for _, _, v in leaves
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    finite = [v for v in numbers if not (isinstance(v, float) and not math.isfinite(v))]

    nonfinite = [(p, v) for p, _, v in leaves
                 if isinstance(v, float) and not math.isfinite(v)]
    if nonfinite:
        problems.append("%d non-finite value(s), first at %s = %s"
                        % (len(nonfinite), nonfinite[0][0], nonfinite[0][1]))

    if not finite:
        problems.append("no finite numeric value anywhere in the payload")
    elif all(v == 0 for v in finite):
        problems.append("every one of the %d numeric values is zero" % len(finite))

    null_metrics = [p for p, k, v in leaves if v is None and _METRIC_KEY.search(k or "")]
    if null_metrics:
        problems.append("null where a quantity was expected, at %s" % ", ".join(null_metrics[:3]))

    empty_tables = [p for p, k, v in leaves
                    if isinstance(v, (dict, list)) and len(v) == 0 and _TABLE_KEY.search(k or "")]
    if empty_tables:
        problems.append("empty result table(s) at %s" % ", ".join(empty_tables[:3]))

    return problems


def guarded_dump(payload, path, what="this producer"):
    """Write `payload` to `path` as indented JSON, or refuse and exit non-zero.

    Refuses when the payload is empty, carries a non-finite number, carries no finite number at all,
    is numerically all-zero, holds null where a quantity belongs, or holds an empty result table.
    Those are the signatures of a run that found no data.
    """
    path = safe_artifact_path(path)
    problems = _diagnose(payload)

    if problems and not os.environ.get(_OVERRIDE):
        sys.exit(
            "refusing to overwrite %s: %s.\n"
            "This is what a run with no raw data looks like. The shipped artifact has been left intact.\n"
            "%s needs the per-cohort arrays under data/cohorts_grffix/ and data/external/*/processed/; "
            "build them with the per-site harmonisation code first (see data/README.md). "
            "Set %s=1 to write anyway."
            % (path.name, "; ".join(problems), what, _OVERRIDE))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def guarded_savez(path, what="this producer", **arrays):
    """The same rule for the .npz caches: refuse when every array is empty."""
    import numpy as np
    path = safe_artifact_path(path)
    sizes = {k: int(np.asarray(v, dtype=object).size) for k, v in arrays.items()}
    finite = 0
    for v in arrays.values():
        a = np.asarray(v)
        if a.dtype.kind in "fc":
            finite += int(np.isfinite(a).sum())
        elif a.dtype.kind in "iub":
            finite += int(a.size)
    bad = (not sizes) or all(n == 0 for n in sizes.values()) or finite == 0
    if bad and not os.environ.get(_OVERRIDE):
        sys.exit(
            "refusing to overwrite %s: no finite value in any array (%s).\n"
            "This is what a run with no raw data looks like. The shipped cache has been left intact.\n"
            "%s needs the rebuilt per-cohort arrays. Set %s=1 to write anyway."
            % (path.name, sizes, what, _OVERRIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def _selftest(artifact_dir=None):
    """Both directions: every shipped artifact passes, every degenerate shape is refused."""
    here = Path(__file__).resolve().parents[2]
    art = Path(artifact_dir) if artifact_dir else here / "results"
    ok = bad = 0
    for p in sorted(art.glob("_*.json")):
        probs = _diagnose(json.loads(p.read_text(encoding="utf-8")))
        if probs:
            bad += 1
            print("  FALSE POSITIVE %-44s %s" % (p.name, probs))
        else:
            ok += 1
    print("  shipped artifacts accepted : %d, wrongly refused: %d" % (ok, bad))

    degenerate = {
        "null aggregate": {"n_subjects": 0, "overall_survival_rate": None, "per_cohort": {}},
        "empty result table": {"sites": {}, "checkpoint": "ALLDATA"},
        "empty results list": {"results": [], "checkpoint": "ALLDATA"},
        "all zeros": {"n_cycles_total": 0, "subject_macro": 0.0},
        "no numbers at all": {"note": "x", "checkpoint": "ALLDATA"},
        "nan aggregate": {"subject_macro": float("nan"), "n_subj": {"cp": 1}},
        "empty per_subject": {"per_subject_overall": {}, "subject_macro": 0.8},
        "null macro": {"subject_macro_PCC": None, "n_cycles": 12},
    }
    caught = 0
    for lab, p in degenerate.items():
        if _diagnose(p):
            caught += 1
        else:
            print("  MISSED %-22s %s" % (lab, p))
    print("  degenerate payloads refused: %d of %d" % (caught, len(degenerate)))
    return bad == 0 and caught == len(degenerate)


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
