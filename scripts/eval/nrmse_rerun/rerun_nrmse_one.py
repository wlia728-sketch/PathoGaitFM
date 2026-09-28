# -*- coding: utf-8 -*-
"""Run one evaluation script once, in one metric, without changing anything it computes.

THE HOOK.  Every one of these producers funnels each scored waveform pair through a bare
`safe_pcc(ground_truth, prediction)` call, and in every one of them `safe_pcc` is a module-level name.
Replacing that one name is therefore enough to change what the producer scores while leaving the model,
the checkpoints, the cycle-validity gates, the aggregation order and the output structure exactly as
published. In `nrmse` mode the substitute returns range-normalised RMSE, so the producer's own
aggregator averages error with precisely the weighting it uses for correlation. In `pcc` mode the
substitute returns the correlation unchanged, which reproduces the producer's own result and is what
licenses the error run beside it.

WHY THE Z-SPACE ARRAYS ARE ENOUGH.  nRMSE is normalised by the ground truth's own range, so it is
invariant to the linear de-normalisation these producers do not apply at this point. The
`rmse_nrmse` in eval_kinetics_all_cycles documents that invariance and this computes the same quantity.

THE CELL SET.  Each producer keeps a scored cell only where the returned value is finite. The
substitute computes both metrics on every call and counts any cell where the two disagree about
finiteness, because that and only that would make the error run score a different set of cells than the
published correlation run. The count is written into the result.

NOTHING IS WRITTEN BY THE PRODUCER.  `guarded_dump` is replaced by a capture, so the producer's own
output file is not written; what it tried to write is saved here instead.

    python rerun_nrmse_one.py <module> <pcc|nrmse> [producer argv...]
"""
import copy
import importlib
import os
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PKG = Path(__file__).resolve().parents[3]
EVAL = PKG / "scripts" / "eval"
OUTDIR = Path(os.environ.get("PATHOGAIT_NRMSE_OUT", str(PKG / "outputs/nrmse_runs"))).resolve()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    module_name, mode = sys.argv[1], sys.argv[2]
    producer_argv = sys.argv[3:]
    if mode not in ("pcc", "nrmse"):
        raise SystemExit("mode must be pcc or nrmse")

    # RUN_TAG separates runs of one producer that differ by an environment variable rather than by an
    # argument. `_ablation_cp_cond` is the case: its conditioning strength comes from ABL_CFG, so two
    # arms of the same panel would otherwise write to one filename and the second would erase the first.
    tag = os.environ.get("RUN_TAG", "")
    stem = (module_name + ("_" + "_".join(producer_argv) if producer_argv else "")
            + ("_" + tag if tag else "") + "_" + mode)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target = OUTDIR / (stem + ".json")

    # the two modality producers read sys.argv at import time, so it is set before the import
    sys.argv = [module_name] + list(producer_argv)
    sys.path.insert(0, str(EVAL))
    module = importlib.import_module(module_name)

    real_safe_pcc = module.safe_pcc
    stats = {"calls": 0, "pcc_finite": 0, "nrmse_finite": 0, "finiteness_disagreements": 0}

    def substitute(ground_truth, prediction):
        pcc = real_safe_pcc(ground_truth, prediction)
        g = np.asarray(ground_truth, dtype=float)
        p = np.asarray(prediction, dtype=float)
        span = float(np.max(g) - np.min(g))
        rmse = float(np.sqrt(np.mean((g - p) ** 2)))
        nrmse = (rmse / span * 100.0) if span > 1e-9 else float("nan")
        stats["calls"] += 1
        pcc_ok, nrmse_ok = bool(np.isfinite(pcc)), bool(np.isfinite(nrmse))
        stats["pcc_finite"] += pcc_ok
        stats["nrmse_finite"] += nrmse_ok
        stats["finiteness_disagreements"] += pcc_ok != nrmse_ok
        return pcc if mode == "pcc" else nrmse

    captured = []

    def capture(obj, path=None, tag=None, *args, **kwargs):
        captured.append({"path": str(path), "out": copy.deepcopy(obj)})
        return Path(str(path)) if path is not None else Path(".")

    # Optional narrowing, used only by the correlation pass on the three expensive producers. Every one
    # of them walks its validation subjects with `get_val_pairs(SPLIT, fold)` and none of them derives a
    # training set from CV5, so dropping folds here drops work without changing what the remaining work
    # is. The arm and scenario dictionaries are module-level in the two producers that have them.
    # PROC_OVERRIDE points a site producer at processed arrays stored outside the default site folder.
    proc_override = os.environ.get("PROC_OVERRIDE")
    if proc_override:
        if not Path(proc_override).is_dir():
            raise SystemExit("PROC_OVERRIDE points at %s which is not a directory" % proc_override)
        if not hasattr(module, "PROC"):
            raise SystemExit("%s has no PROC to override" % module_name)
        print("  site data from %s\n            instead of %s" % (proc_override, module.PROC),
              flush=True)
        module.PROC = Path(proc_override)

    # ROOT_OVERRIDE relocates the producer's checkpoint directory; the output is captured, not written.
    root_override = os.environ.get("ROOT_OVERRIDE")
    if root_override:
        if not Path(root_override).is_dir():
            raise SystemExit("ROOT_OVERRIDE points at %s which is not a directory" % root_override)
        print("  ROOT moved to %s\n       from %s" % (root_override, getattr(module, "ROOT", "?")),
              flush=True)
        module.ROOT = Path(root_override)

    limits = {}
    folds = os.environ.get("LIMIT_FOLDS")
    if folds and hasattr(module, "CV5"):
        module.CV5 = module.CV5[: int(folds)]
        limits["folds"] = list(module.CV5)
    arms = os.environ.get("LIMIT_ARMS")
    if arms:
        for attribute in ("SCENARIOS", "CONDITIONS"):
            table = getattr(module, attribute, None)
            if isinstance(table, dict):
                keep = list(table)[: int(arms)]
                setattr(module, attribute, {k: table[k] for k in keep})
                limits[attribute] = keep
    if limits:
        print("  NARROWED for a spot check: %s" % limits, flush=True)

    # Keep the generated waveforms this time. The reason these runs cost hours is that the model is a
    # diffusion model and every cycle takes three seeds of fifty DDIM steps, a hundred and fifty forward
    # passes, to produce a waveform that the published pipeline scores into one scalar and then drops.
    # Stored at half precision the same waveforms are a few hundred megabytes, so any metric wanted
    # later can be computed from them on a CPU in seconds instead of re-sampling the model. Caching is
    # strictly additive: it never touches the returned value, and any failure disables the cache rather
    # than the run, because a twenty-six hour job must not die over a disk write.
    cache = {"calls": 0, "bytes": 0, "disabled": not os.environ.get("CACHE_PREDICTIONS"),
             "reason": "not requested" if not os.environ.get("CACHE_PREDICTIONS") else "",
             "hooked": None}
    cache_dir = OUTDIR / "pred_cache" / stem
    limit_bytes = float(os.environ.get("CACHE_LIMIT_GB", "6")) * 1e9

    def wrap_generation(name):
        real = getattr(module, name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache["hooked"] = name

        def wrapper(*args, **kwargs):
            result = real(*args, **kwargs)
            if not cache["disabled"]:
                try:
                    array = result if isinstance(result, np.ndarray) else \
                        result.detach().to("cpu").numpy()
                    # float32, not half. A half-precision cache would make the replay's correlations
                    # differ from this run's own, and the replay exists precisely to be compared
                    # against the published values, so the round trip has to be exact.
                    path = cache_dir / ("%06d.npy" % cache["calls"])
                    np.save(path, np.asarray(array, dtype=np.float32))
                    cache["bytes"] += path.stat().st_size
                    if cache["bytes"] > limit_bytes:
                        cache["disabled"] = True
                        cache["reason"] = "reached the %.1f GB ceiling" % (limit_bytes / 1e9)
                except Exception as exc:                      # never kill the run over the cache
                    cache["disabled"] = True
                    cache["reason"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
            cache["calls"] += 1
            return result
        setattr(module, name, wrapper)

    def replay_generation(name, source_dir):
        """Return the waveforms the error pass generated, instead of sampling the model again.

        This is what makes a full verification affordable. The error pass costs a hundred and fifty
        forward passes per cycle; replaying its cached output costs a file read, so the correlation
        pass can cover every fold and every arm and be compared against the published artifact
        directly, rather than covering one fold and being comparable to nothing.
        """
        cache["hooked"] = name + " (replayed)"
        # `gen_tsa` ends in .cpu().numpy() and hands back an array; `tweedie_inpaint_drop` hands back
        # a tensor on the device it was given. The replay has to match whichever it stands in for,
        # because the producer's next line differs accordingly.
        as_tensor = name == "tweedie_inpaint_drop"

        def wrapper(*args, **kwargs):
            path = source_dir / ("%06d.npy" % cache["calls"])
            if not path.exists():
                raise AssertionError("the replay ran past the cache at call %d, so the cached run "
                                     "and this one are not the same sequence" % cache["calls"])
            array = np.load(path)
            cache["calls"] += 1
            cache["bytes"] += path.stat().st_size
            if not as_tensor:
                return array
            import torch
            device = next((a for a in args if isinstance(a, torch.device)), None)
            if device is None:
                device = next((a.device for a in args if hasattr(a, "device")), "cpu")
            return torch.from_numpy(array).to(device)
        setattr(module, name, wrapper)

    replay_from = os.environ.get("REPLAY_CACHE")
    if replay_from:
        source = Path(replay_from)
        if not source.is_dir():
            raise SystemExit("REPLAY_CACHE points at %s which is not a directory" % source)
        for candidate in ("gen_tsa", "tweedie_inpaint_drop"):
            if hasattr(module, candidate):
                replay_generation(candidate, source)
                cache["disabled"] = True
                cache["reason"] = "replaying %d cached arrays from %s" % (
                    len(list(source.glob("*.npy"))), source)
                print("  REPLAYING %s from %s, no model sampling"
                      % (candidate, source), flush=True)
                break
        else:
            raise SystemExit("no generation entry point to replay on %s" % module_name)
    elif not cache["disabled"]:
        for candidate in ("gen_tsa", "tweedie_inpaint_drop"):
            if hasattr(module, candidate):
                wrap_generation(candidate)
                print("  caching generated waveforms from %s into %s"
                      % (candidate, cache_dir), flush=True)
                break
        else:
            cache["disabled"] = True
            cache["reason"] = "no generation entry point found on this module"

    module.safe_pcc = substitute
    if hasattr(module, "guarded_dump"):
        module.guarded_dump = capture
    if hasattr(module, "write_report"):
        module.write_report = lambda *a, **k: "(report writing disabled)"

    print("[%s] mode=%s argv=%s" % (module_name, mode, producer_argv), flush=True)
    started = time.time()
    module.main()
    elapsed = time.time() - started

    if not captured:
        raise AssertionError("%s finished without attempting a write, so nothing was captured"
                             % module_name)
    if stats["calls"] <= 0 or stats["pcc_finite"] <= 0 or stats["finiteness_disagreements"]:
        raise AssertionError("No finite scored cells, or PCC/nRMSE admitted different cells: %s" % stats)

    # A replay is only evidence if it walked the same sequence as the run it replays. Running past the
    # cache already aborts inside the wrapper; stopping short of it would not, and would mean the two
    # passes disagreed about which cycles to score, so the counts are compared here as well.
    if replay_from:
        available = len(list(Path(replay_from).glob("*.npy")))
        cache["cached_arrays_available"] = available
        cache["replayed"] = cache["calls"]
        if cache["calls"] != available:
            raise AssertionError("the replay consumed %d cached arrays but the cache holds %d, so "
                                 "this pass and the error pass did not score the same sequence"
                                 % (cache["calls"], available))
    result = {"module": module_name, "mode": mode, "producer_argv": producer_argv,
              "elapsed_s": round(elapsed, 1), "scoring": stats, "limits": limits,
              "prediction_cache": cache,
              "writes": captured}
    json.dump(result, open(target, "w", encoding="utf-8"), indent=1, default=str)
    print("\n[%s] %s mode finished in %.1f s (%.2f h)"
          % (module_name, mode, elapsed, elapsed / 3600.0), flush=True)
    print("  scored cells %d, correlation finite %d, error finite %d, disagreements %d"
          % (stats["calls"], stats["pcc_finite"], stats["nrmse_finite"],
             stats["finiteness_disagreements"]), flush=True)
    print("  captured %d write(s) -> %s" % (len(captured), target), flush=True)
    if cache["hooked"]:
        print("  cached %d generation calls, %.2f GB, from %s%s"
              % (cache["calls"], cache["bytes"] / 1e9, cache["hooked"],
                 "" if not cache["disabled"] else " (stopped: %s)" % cache["reason"]),
              flush=True)
    else:
        print("  no waveform cache: %s" % cache["reason"], flush=True)


if __name__ == "__main__":
    main()
