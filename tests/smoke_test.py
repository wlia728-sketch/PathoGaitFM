"""Smoke check without data or weights: the write guard, the CLI entry points and the figure export."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# every subprocess runs without the developer's PYTHONPATH and PATHOGAIT_* overrides
CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("PATHOGAIT_") and k != "PYTHONPATH"}
CLI_ROOTS = (
    "data/prepare/fit_stage1_zstats.py",
    "model/pathogait/training/train_pretrain.py",
    "scripts/train/train_finetune.py",
    "model_baseline/published_comparators.py",
    "example_usage/predict_kinetics.py",
    "example_usage/partial_input.py",
    "demo/server.py",
    "scripts/eval/eval_kinetics_all_cycles.py",
    "scripts/eval/partial_input_masking.py",
    "scripts/eval/eval_external_zeroshot_8ch.py",
    "scripts/figures/replot_results.py",
    "data/external/fukuchi_healthy/harmonize_fukuchi_ascii.py",
)


def run(script, *args, env=None):
    command = [sys.executable, str(ROOT / script), *args]
    result = subprocess.run(command, cwd=ROOT, env=env or CLEAN_ENV, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{script} failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
    return result


def figure_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "figures").iterdir() if p.is_file()}


def main():
    before = figure_hashes()
    run("scripts/lib/guarded_write.py")
    for script in CLI_ROOTS:
        run(script, "--help")
    run("scripts/eval/nrmse_rerun/rerun_nrmse_all.py", "--list")
    with tempfile.TemporaryDirectory() as folder:
        run("scripts/figures/export_figures.py", "--out", folder)
        for number in range(1, 6):
            produced = list(Path(folder).glob(f"Fig{number}*.png"))
            if not produced or any(p.stat().st_size < 1000 for p in produced):
                raise AssertionError(f"Figure {number} exported no usable PNG")
    if before != figure_hashes():
        raise AssertionError("A smoke check changed the manuscript figures")
    print(f"SMOKE TEST PASSED: write guard, {len(CLI_ROOTS)} CLI roots, batch listing, 5 figure exports")


if __name__ == "__main__":
    main()
