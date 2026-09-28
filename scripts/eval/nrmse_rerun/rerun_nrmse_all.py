"""Run eight evaluations as nRMSE followed by cached PCC replay.

Requires the locally built data arrays and the downloaded weights. Writes one capture per job.
PROC_OVERRIDE is allowed for one explicitly selected job, never a mixed batch.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PKG = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("rerun_nrmse_one.py")
JOBS = [
    ("eval_opencap_zeroshot", ["marker"]), ("eval_opencap_zeroshot", ["video"]),
    ("eval_imu_zeroshot", ["marker"]), ("eval_imu_zeroshot", ["imu"]),
    ("eval_opencap_moments", []), ("head_vs_headless_ablation", []),
    ("partial_input_masking", []), ("eval_kinetics_all_cycles", []),
]


def job_id(module, argv):
    return "_".join([module] + argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--job", action="append", choices=[job_id(*j) for j in JOBS])
    parser.add_argument("--out", type=Path, default=PKG / "outputs/nrmse_runs")
    args = parser.parse_args()
    jobs = [j for j in JOBS if not args.job or job_id(*j) in args.job]
    if args.list:
        for job in jobs:
            print(job_id(*job))
        return 0
    if os.environ.get("PROC_OVERRIDE") and len(jobs) != 1:
        parser.error("PROC_OVERRIDE requires exactly one --job")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    summary = []
    for module, argv in jobs:
        stem = job_id(module, argv)
        cache = out / "pred_cache" / (stem + "_nrmse")
        row = {"job": stem}
        for mode in ("nrmse", "pcc"):
            env = dict(os.environ)
            for name in ("LIMIT_FOLDS", "LIMIT_ARMS", "RUN_TAG", "REPLAY_CACHE", "CACHE_PREDICTIONS"):
                env.pop(name, None)
            env["PATHOGAIT_NRMSE_OUT"] = str(out)
            env["PYTHONIOENCODING"] = "utf-8"
            if mode == "nrmse":
                env["CACHE_PREDICTIONS"] = "1"
            else:
                if not cache.is_dir() or not any(cache.glob("*.npy")):
                    row[mode] = {"exit_code": 1, "error": "no prediction cache"}
                    break
                env["REPLAY_CACHE"] = str(cache)
            start = time.time()
            log = logs / (stem + "_" + mode + ".log")
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.run([sys.executable, str(RUNNER), module, mode] + argv,
                                         cwd=PKG, env=env, stdout=handle, stderr=subprocess.STDOUT)
            row[mode] = {"exit_code": process.returncode, "elapsed_s": time.time()-start, "log": str(log)}
            print(stem, mode, "exit", process.returncode, flush=True)
            if process.returncode:
                break
        summary.append(row)
        (out / "_nrmse_rerun_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed = any(any(r.get(m, {}).get("exit_code", 1) for m in ("nrmse", "pcc")) for r in summary)
    print("Captures written to", out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

