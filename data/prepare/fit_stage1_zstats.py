"""Fit leakage-free global z-score constants from Stage-1 train only.

This is the canonical normalisation-statistics producer for any clean
retraining.  It intentionally refuses Stage-2 cohort sources, uses only the
``train`` partition of ``v4_stage1_split.json``, and applies the same
pre-z-score transform (including per-subject polarity decisions) as model
training.

The output is not compatible with checkpoints trained using a different
normalisation file.  Generate new constants first, then pass the same file to
Stage 1, every Stage-2 fold, the all-data model and every evaluator.

Example from the submission-code directory::

    python data/prepare/fit_stage1_zstats.py \
      --raw-root .. \
      --output data/zstats/global_zstats_stage1_only.json

The script never overwrites an existing file unless ``--overwrite`` is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathogait.data.transform import (  # noqa: E402
    ANGLE_CHANNELS_54CH,
    EMG_CHANNELS_54CH,
    GRF_CHANNELS_54CH,
    MOMENT_CHANNELS_54CH,
    POWER_CHANNELS_54CH,
)
from pathogait.data.transform_per_subject import (  # noqa: E402
    V4LazyTransformV3PerSubject,
)


ZSCORE_CHANNELS = (
    MOMENT_CHANNELS_54CH
    + POWER_CHANNELS_54CH
    + GRF_CHANNELS_54CH
    + ANGLE_CHANNELS_54CH
)
STAGE2_SOURCES = {"cp", "normal", "vdk_stroke", "bmclab_pd"}


class _UnitMetadata:
    """Minimal metadata interface for the currently unified raw arrays.

    Layers A and B of the shipped transform are an explicit identity for all
    current sources.  The transform still calls ``get_bw`` for forward
    compatibility, so the stat fitter supplies a neutral value without
    requiring private anthropometry files.
    """

    @staticmethod
    def get_bw(source: str, subject_id: str) -> float:
        del source, subject_id
        return 1.0


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json_dump(payload, path: Path, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {path}; pass --overwrite explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, path)


def _load_stage1_train(split_path: Path) -> Sequence[Tuple[str, str]]:
    doc = _read_json(split_path)
    try:
        pairs = doc["splits"]["train"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{split_path} must contain splits.train as [source, subject] pairs"
        ) from exc

    out = []
    for item in pairs:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"invalid split entry: {item!r}")
        source, subject_id = map(str, item)
        out.append((source, subject_id))

    sources = {source for source, _ in out}
    leaked = sorted(sources & STAGE2_SOURCES)
    if leaked:
        raise ValueError(
            "Stage-1-only z-stats cannot include downstream Stage-2 sources: "
            + ", ".join(leaked)
        )
    if not out:
        raise ValueError(f"empty Stage-1 training split: {split_path}")
    return out


def _raw_path(raw_root: Path, source: str, subject_id: str) -> Path:
    if source.startswith("addbio_"):
        return raw_root / "data" / "pretraining" / "addbio" / f"{subject_id}.npy"
    return raw_root / "data" / "cohorts_raw" / source / f"{subject_id}.npy"


def _source_args(source: str) -> Tuple[str, str]:
    if source.startswith("addbio_"):
        return source[len("addbio_"):], "addbio"
    return source, "cohort"


def _iter_pre_zscore(
    pairs: Iterable[Tuple[str, str]],
    raw_root: Path,
    transform: V4LazyTransformV3PerSubject,
) -> Iterator[Tuple[str, str, np.ndarray]]:
    for source, subject_id in pairs:
        path = _raw_path(raw_root, source, subject_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage-1 training array: {path}")
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 3 or arr.shape[-1] != 54:
            raise ValueError(f"expected (cycles, time, 54) at {path}, got {arr.shape}")
        source_name, source_type = _source_args(source)
        x = transform.preprocess_before_zscore(
            arr, source_name, source_type, subject_id
        )
        yield source, subject_id, x


def _fit_source(
    source: str,
    pairs: Sequence[Tuple[str, str]],
    raw_root: Path,
    transform: V4LazyTransformV3PerSubject,
    trim_sigma: float,
) -> Dict:
    sums = {ch: 0.0 for ch in ZSCORE_CHANNELS}
    sums_sq = {ch: 0.0 for ch in ZSCORE_CHANNELS}
    counts = {ch: 0 for ch in ZSCORE_CHANNELS}
    emg_min = {ch: float("inf") for ch in EMG_CHANNELS_54CH}
    emg_max = {ch: float("-inf") for ch in EMG_CHANNELS_54CH}

    # Pass 1: provisional per-source moments.  Arrays are streamed so the
    # 227k-cycle corpus is never retained in memory.
    for _, _, x in _iter_pre_zscore(pairs, raw_root, transform):
        for ch in ZSCORE_CHANNELS:
            values = x[..., ch].ravel()
            values = values[np.isfinite(values)].astype(np.float64, copy=False)
            if values.size:
                sums[ch] += float(values.sum())
                sums_sq[ch] += float(np.square(values).sum())
                counts[ch] += int(values.size)
        for ch in EMG_CHANNELS_54CH:
            values = x[..., ch]
            finite = values[np.isfinite(values)]
            if finite.size:
                emg_min[ch] = min(emg_min[ch], float(finite.min()))
                emg_max[ch] = max(emg_max[ch], float(finite.max()))

    provisional = {}
    for ch in ZSCORE_CHANNELS:
        if counts[ch] < 10:
            continue
        mean = sums[ch] / counts[ch]
        variance = max(0.0, sums_sq[ch] / counts[ch] - mean * mean)
        provisional[ch] = (float(mean), float(np.sqrt(variance)))

    if not provisional:
        raise RuntimeError(f"source {source} has no finite z-score cells")

    clean_sums = {ch: 0.0 for ch in provisional}
    clean_sums_sq = {ch: 0.0 for ch in provisional}
    clean_counts = {ch: 0 for ch in provisional}
    trimmed_counts = {ch: 0 for ch in provisional}

    # Pass 2: source-local sigma trim and clean moments.
    for _, _, x in _iter_pre_zscore(pairs, raw_root, transform):
        for ch, (mean, std) in provisional.items():
            values = x[..., ch].ravel()
            values = values[np.isfinite(values)].astype(np.float64, copy=False)
            if not values.size:
                continue
            keep = (
                np.ones(values.shape, dtype=bool)
                if std < 1e-12
                else np.abs(values - mean) <= trim_sigma * std
            )
            trimmed_counts[ch] += int((~keep).sum())
            clean = values[keep]
            clean_sums[ch] += float(clean.sum())
            clean_sums_sq[ch] += float(np.square(clean).sum())
            clean_counts[ch] += int(clean.size)

    per_channel = {}
    for ch, (mean0, std0) in provisional.items():
        n = clean_counts[ch]
        if n < 10:
            continue
        mean = clean_sums[ch] / n
        variance = max(0.0, clean_sums_sq[ch] / n - mean * mean)
        per_channel[str(ch)] = {
            "mu_provisional": round(mean0, 6),
            "sigma_provisional": round(std0, 6),
            "mu_clean": round(float(mean), 6),
            "sigma_clean": round(float(np.sqrt(variance)), 6),
            "n_cells_total": counts[ch],
            "n_trimmed": trimmed_counts[ch],
            "trim_pct": round(100.0 * trimmed_counts[ch] / counts[ch], 6),
        }

    return {
        "n_files": len(pairs),
        "per_channel": per_channel,
        "emg_range": {
            str(ch): [
                None if emg_min[ch] == float("inf") else round(emg_min[ch], 6),
                None if emg_max[ch] == float("-inf") else round(emg_max[ch], 6),
            ]
            for ch in EMG_CHANNELS_54CH
        },
    }


def _channel_names() -> Dict[str, str]:
    legacy_path = ROOT / "data" / "zstats" / "global_zstats.json"
    if not legacy_path.is_file():
        return {}
    legacy = _read_json(legacy_path)
    return {
        str(ch): str(rec.get("name", f"ch{ch}"))
        for ch, rec in legacy.items()
        if isinstance(rec, dict)
    }


def _aggregate(per_source: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    names = _channel_names()
    global_stats = {}
    consistency = {}
    for ch in ZSCORE_CHANNELS:
        key = str(ch)
        contributing = []
        means = []
        stds = []
        for source, rec in per_source.items():
            cell = rec["per_channel"].get(key)
            if cell is None:
                continue
            contributing.append(source)
            means.append(float(cell["mu_clean"]))
            stds.append(float(cell["sigma_clean"]))
        if not means:
            continue
        mean = float(np.mean(means))
        std = float(np.sqrt(np.mean(np.square(stds))))
        global_stats[key] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "name": names.get(key, f"ch{ch}"),
            "n_sources": len(contributing),
            "contributing_sources": contributing,
        }
        z = np.zeros(len(means)) if std < 1e-12 else (np.asarray(means) - mean) / std
        consistency[key] = {
            "mu_per_source": dict(zip(contributing, means)),
            "mu_global": mean,
            "sigma_global": std,
            "flagged_outlier_sources": [
                [contributing[i], round(float(z[i]), 2), round(means[i], 3)]
                for i in range(len(z))
                if abs(z[i]) > 3.0
            ],
        }
    return global_stats, consistency


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "v4_stage1_split.json",
        help="Stage-1 split JSON; only splits.train is consumed.",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT,
        help="Package root holding data/pretraining/addbio/ and data/cohorts_raw/.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "zstats" / "global_zstats_stage1_only.json",
    )
    p.add_argument(
        "--per-subject-flip-paper",
        type=Path,
        default=ROOT / "data" / "metadata"
         / "per_subject_flip_table.json",
    )
    p.add_argument(
        "--per-subject-flip-addbio",
        type=Path,
        default=ROOT / "data" / "metadata"
         / "per_subject_flip_table_addbio.json",
    )
    p.add_argument("--trim-sigma", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate split scope and raw-file coverage without fitting.",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    split_path = args.split.resolve()
    raw_root = args.raw_root.resolve()
    output = args.output.resolve()

    for required in (
        split_path,
        args.per_subject_flip_paper.resolve(),
        args.per_subject_flip_addbio.resolve(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.trim_sigma <= 0:
        raise ValueError("--trim-sigma must be positive")

    train_pairs = _load_stage1_train(split_path)
    by_source: Dict[str, list] = {}
    for pair in train_pairs:
        by_source.setdefault(pair[0], []).append(pair)

    missing = [
        _raw_path(raw_root, source, subject_id)
        for source, subject_id in train_pairs
        if not _raw_path(raw_root, source, subject_id).is_file()
    ]
    print(
        f"Stage-1 train only: {len(train_pairs)} files, "
        f"{len(by_source)} sources; Stage-2 sources present: none"
    )
    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} Stage-1 arrays are missing; first entries:\n{preview}"
        )
    if args.check_only:
        print("check-only PASS")
        return

    transform = V4LazyTransformV3PerSubject(
        metadata=_UnitMetadata(),
        global_zstats_path=None,
        per_subject_table_paper=str(args.per_subject_flip_paper.resolve()),
        per_subject_table_addbio=str(args.per_subject_flip_addbio.resolve()),
        ambiguous_policy="no_flip",
    )

    per_source = {}
    for source in sorted(by_source):
        print(f"fitting {source}: {len(by_source[source])} files")
        per_source[source] = _fit_source(
            source,
            by_source[source],
            raw_root,
            transform,
            args.trim_sigma,
        )

    global_stats, consistency = _aggregate(per_source)
    per_source_path = output.with_name(output.stem + ".per_source.json")
    consistency_path = output.with_name(output.stem + ".consistency.json")
    provenance_path = output.with_name(output.stem + ".provenance.json")
    provenance = {
        "fit_scope": "stage1_train_only",
        "downstream_stage2_sources_in_fit": [],
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "raw_root": str(raw_root),
        "n_train_files": len(train_pairs),
        "sources": sorted(by_source),
        "n_sources": len(by_source),
        "source_weighting": "equal weight among contributing Stage-1 sources per channel",
        "trim_sigma_within_source": args.trim_sigma,
        "mean_formula": "mean(source_clean_means)",
        "std_formula": "sqrt(mean(source_clean_stds^2))",
        "preprocessing": "canonical Layers A-C.6 including per-subject polarity; before z-score/clip/reduction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility_warning": (
            "Use only with models trained from Stage 1 onward using this exact file; "
            "do not substitute it when evaluating legacy checkpoints."
        ),
    }

    for payload, path in (
        (global_stats, output),
        (per_source, per_source_path),
        (consistency, consistency_path),
    ):
        _atomic_json_dump(payload, path, overwrite=args.overwrite)
        print(f"wrote {path}")
    provenance["zstats_sha256"] = _sha256(output)
    _atomic_json_dump(provenance, provenance_path, overwrite=args.overwrite)
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
