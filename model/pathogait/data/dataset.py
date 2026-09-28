"""Balanced multi-source dataset + collate for Stage-1 / Stage-2 training.

Reads the per-subject .npy / _mask.npy layout under the raw cohort and AddBio
directories, applies the V4 per-subject transform (54 -> 40 channels, body-weight
normalisation, z-score), and yields one gait cycle at a time under source-uniform
sampling. Stage 1 mixes the healthy-adult sources; Stage 2 the four pathological
cohorts. Split membership is supplied via `from_split`.
"""
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import IterableDataset

from .channels import EXCLUDED_STUDIES, filter_addbio_files, KEEP_CHANNELS_54TO40
from .transform import V4LazyTransformV3
from .subject_metadata import V4SubjectMetadata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V4_RAW_ROOT = PROJECT_ROOT / "data"
V4_ADDBIO_DIR = V4_RAW_ROOT / "pretraining" / "addbio"

# Raw cohort data root. Default = data/cohorts_raw; the GRF-cycle-convention-fixed
# cohort data is selected by pointing PATHOGAIT_COHORT_RAW_DIR at it (reversible).
V4_COHORT_RAW_DIR = PROJECT_ROOT / "data" / "cohorts_raw"
if os.environ.get("PATHOGAIT_COHORT_RAW_DIR"):
    V4_COHORT_RAW_DIR = Path(os.environ["PATHOGAIT_COHORT_RAW_DIR"])

V4_COHORT_SOURCES = ["cp", "normal", "vdk_healthy", "vdk_stroke", "bmclab_pd"]

# AddBio studies kept after channels.EXCLUDED_STUDIES drops the small-n /
# noisy / convention-different ones.
V4_ADDBIO_STUDIES = [
    s for s in [
        "Camargo2021", "Carter2023", "Fregly2012", "Hamner2013",
        "Han2023", "Moore2015", "Tan2021", "Tan2022",
        "Wang2023", "vanderZee2022",
    ]
    if s not in EXCLUDED_STUDIES
]

# Stage-1 (healthy-adult pretrain) vs Stage-2 (pathological cohort fine-tune).
# vdk_healthy stays in Stage 1 as the in-distribution healthy lab anchor.
V4_STAGE1_SOURCES = [f"addbio_{s}" for s in V4_ADDBIO_STUDIES] + ["vdk_healthy"]
V4_STAGE2_SOURCES = ["cp", "normal", "vdk_stroke", "bmclab_pd"]


def _discover_files() -> Dict[str, List[str]]:
    """Return {source_name: [.npy paths]} for every cohort + AddBio source."""
    out: Dict[str, List[str]] = {}
    for c in V4_COHORT_SOURCES:
        d = V4_COHORT_RAW_DIR / c
        if not d.exists():
            continue
        files = sorted(str(p) for p in d.glob("*.npy") if "_mask" not in p.name)
        if files:
            out[c] = files
    for s in V4_ADDBIO_STUDIES:
        files = sorted(str(p) for p in V4_ADDBIO_DIR.glob(f"{s}_*.npy") if "_mask" not in p.name)
        files = filter_addbio_files(files)
        if files:
            out[f"addbio_{s}"] = files
    return out


def _source_type(name: str) -> str:
    return "addbio" if name.startswith("addbio_") else "cohort"


def _strip_addbio_prefix(name: str) -> str:
    return name[len("addbio_"):] if name.startswith("addbio_") else name


def collate(batch: List[Dict]) -> Dict:
    """Stack tensors; keep strings as lists."""
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        elif isinstance(batch[0][k], (int, float)):
            out[k] = torch.tensor([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


class V4BalancedDatasetV3(IterableDataset):
    """Source-uniform sampler over the raw cohort + AddBio sources.

    Args:
      global_zstats_path: path to global_zstats.json. If None, the z-score
                          layer is skipped (used only when fitting the stats).
      sources / stage:    pass one of them; `stage` in {'stage1','stage2','all'}
                          expands to the corresponding source list.
      file_filter:        list of (source, subject_stem) tuples to restrict to
                          (a cross-validation fold, via `from_split`).
      transform_override: optional pre-constructed transform instance to use
                          instead of the default V4LazyTransformV3. Must accept
                          (arr_54, source_name, source_type, subject_id) and
                          return (n, 101, 40). Trainers inject the per-subject
                          transform through this argument.
    """

    def __init__(
        self,
        global_zstats_path: Optional[str] = None,
        sources: Optional[List[str]] = None,
        stage: Optional[str] = None,
        file_filter: Optional[List[Tuple[str, str]]] = None,
        pseudo_length: int = 100_000,
        seed: int = 0,
        cache_size: int = 64,
        transform_override: Optional[object] = None,
        sampling: str = "source_uniform",
    ):
        super().__init__()
        # Which of the two reported sampling regimes to draw under. Stage 1 draws uniformly over its
        # sources; Stage 2 draws uniformly over the cycles that carry a vertical-GRF label, so the
        # cohorts enter in proportion to their labelled-cycle counts. Each trainer states its own
        # regime here rather than through an environment variable, so a run's sampling mode is
        # visible in the call site. PATHOGAIT_DYNCYCLE_POOL still overrides it when set.
        if sampling not in ("source_uniform", "dynamics_cycle_uniform"):
            raise ValueError("Unknown sampling regime: %r" % (sampling,))
        self.sampling = sampling
        if stage is not None:
            if sources is not None:
                raise ValueError("Pass either `stage` or `sources`, not both.")
            if stage == "stage1":
                sources = list(V4_STAGE1_SOURCES)
            elif stage == "stage2":
                sources = list(V4_STAGE2_SOURCES)
            elif stage == "all":
                sources = None
            else:
                raise ValueError(f"Unknown stage: {stage!r}")
        self.metadata = V4SubjectMetadata()
        if transform_override is not None:
            self.transform = transform_override
        else:
            self.transform = V4LazyTransformV3(
                metadata=self.metadata,
                global_zstats_path=global_zstats_path,
            )

        requested_sources = set(sources) if sources is not None else None
        all_files = _discover_files()
        if file_filter is not None:
            # The split lists file stems, including every Stage-1 _splitN part.
            # Respect explicit source/stage selection and the existing study
            # exclusions, but never shrink a requested split to what happened
            # to be found on disk. CP exclusions are already in the grffix split.
            expected = {
                (src, stem) for src, stem in file_filter
                if (requested_sources is None or src in requested_sources)
                and not (src.startswith("addbio_") and
                         (_strip_addbio_prefix(src) in EXCLUDED_STUDIES or
                          not filter_addbio_files([stem + ".npy"])))
            }
            discovered = {(src, Path(path).stem) for src, paths in all_files.items()
                          for path in paths}
            missing = expected - discovered
            if missing:
                raise FileNotFoundError(
                    "Split references missing training arrays: " +
                    ", ".join(f"{src}/{stem}.npy" for src, stem in sorted(missing)))
        if sources is None:
            sources = sorted(all_files.keys())
        self.sources = [s for s in sources if s in all_files]
        if not self.sources:
            raise RuntimeError("No sources discovered")
        self.source_files: Dict[str, List[str]] = {s: all_files[s] for s in self.sources}

        if file_filter is not None:
            allowed = {(src, stem) for src, stem in file_filter}
            for s in list(self.source_files.keys()):
                kept = [fp for fp in self.source_files[s]
                        if (s, Path(fp).stem) in allowed]
                if kept:
                    self.source_files[s] = kept
                else:
                    del self.source_files[s]
            self.sources = [s for s in self.sources if s in self.source_files]
            if not self.sources:
                raise RuntimeError(
                    f"file_filter removed all files; got {len(file_filter)} entries.")

        self.source_to_id = {s: i for i, s in enumerate(self.sources)}
        self.pseudo_length = int(pseudo_length)
        self.seed = int(seed)
        self._epoch = 0
        self.cache_size = int(cache_size)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: List[str] = []

    @staticmethod
    def _safe_np_load(path: str, max_retries: int = 3, retry_sleep: float = 0.1):
        """np.load with retry-on-truncated-read.

        WSL2/9P on NTFS occasionally returns a short read when many DataLoader
        workers hit the same .npy concurrently, which makes numpy raise a
        reshape-mismatch even though the file is fine on disk. Retry briefly.
        """
        import time
        last_err = None
        for attempt in range(max_retries):
            try:
                return np.load(path)
            except ValueError as e:
                last_err = e
                if "cannot reshape" in str(e) or "size" in str(e):
                    time.sleep(retry_sleep * (attempt + 1))
                    continue
                raise
        raise RuntimeError(
            f"np.load({path}) failed after {max_retries} retries; last error: {last_err}")

    def _load_transformed(self, npy_path: str, src: str):
        if npy_path in self._cache:
            return self._cache[npy_path]
        raw = self._safe_np_load(npy_path)
        mask_path = npy_path.replace(".npy", "_mask.npy")
        if os.path.exists(mask_path):
            m54 = self._safe_np_load(mask_path)
        else:
            m54 = np.ones((raw.shape[0], 54), dtype=bool)
        x40 = self.transform(raw, _strip_addbio_prefix(src), _source_type(src), Path(npy_path).stem)
        m40 = m54[:, KEEP_CHANNELS_54TO40]
        self._cache[npy_path] = (x40, m40)
        self._cache_order.append(npy_path)
        if len(self._cache_order) > self.cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return x40, m40

    def __len__(self) -> int:
        return self.pseudo_length

    def _build_dyncycle_pool(self):
        """Flat pool of (source, file, cycle) triples whose cycle carries a
        vertical-GRF (dynamics) label, for dynamics-cycle-uniform sampling.

        Selected by PATHOGAIT_DYNCYCLE_POOL=1, which is the sampling mode used
        to train the reported CV5 models (the `dynpool` checkpoints): it draws
        uniformly over labelled cycles rather than uniformly over sources.
        vGRF 40-ch indices are 22 (R) and 25 (L).
        """
        if getattr(self, "_dyncycle_pool", None) is not None:
            return
        R_VGRF, L_VGRF = 22, 25
        pool = []
        for src in self.sources:
            for fpath in self.source_files[src]:
                mpath = fpath.replace(".npy", "_mask.npy")
                if not os.path.exists(mpath):
                    continue
                m40 = self._safe_np_load(mpath)[:, KEEP_CHANNELS_54TO40].astype(bool)
                has = m40[:, R_VGRF] | m40[:, L_VGRF]
                for ci in np.where(has)[0]:
                    pool.append((src, fpath, int(ci)))
        if not pool:
            raise RuntimeError("PATHOGAIT_DYNCYCLE_POOL=1 but no dynamics cycles found")
        self._dyncycle_pool = pool

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            wid, n = 0, self.pseudo_length
        else:
            wid, n = worker.id, self.pseudo_length // worker.num_workers
        # The per-epoch stream is keyed on (seed, worker id, epoch), so sampling stays fully
        # determined by the seed while successive epochs draw different cycles. The released
        # checkpoints were trained before this correction, when the key was (seed + worker id)
        # alone and every epoch replayed one identical draw sequence.
        rng = np.random.default_rng([self.seed, wid, self._epoch])
        self._epoch += 1
        _env = os.environ.get("PATHOGAIT_DYNCYCLE_POOL")
        use_pool = (_env == "1") if _env else (self.sampling == "dynamics_cycle_uniform")
        if use_pool:
            self._build_dyncycle_pool()
        for _ in range(n):
            if use_pool:                                            # dynamics-cycle-uniform
                src, fpath, ci = self._dyncycle_pool[rng.integers(len(self._dyncycle_pool))]
                x40, m40 = self._load_transformed(fpath, src)
                if ci >= x40.shape[0]:
                    continue
            else:                                                   # source-uniform
                src = self.sources[rng.integers(len(self.sources))]
                files = self.source_files[src]
                if not files:
                    continue
                fpath = files[rng.integers(len(files))]
                x40, m40 = self._load_transformed(fpath, src)
                ci = int(rng.integers(x40.shape[0]))
            yield {
                "x":           torch.from_numpy(x40[ci].astype(np.float32).T.copy()),
                "valid_mask":  torch.from_numpy(m40[ci].astype(bool).copy()),
                "source_id":   self.source_to_id[src],
                "source_name": src,
                "file":        os.path.basename(fpath),
                "cycle_id":    ci,
            }

    @classmethod
    def from_split(cls, split_json_path, split_name, **kwargs):
        """Construct from a split JSON: 'cv5_fold{N}_{train|val}' or a named split."""
        with open(split_json_path) as f:
            data = json.load(f)
        if "splits" in data and split_name in data["splits"]:
            file_filter = [tuple(x) for x in data["splits"][split_name]]
        elif split_name.startswith("cv5_fold"):
            parts = split_name.split("_")
            fold_num = int(parts[1][len("fold"):])
            subset = parts[2]
            file_filter = [tuple(x) for x in data["cv5"][fold_num][subset]]
        else:
            raise ValueError(f"Unknown split_name {split_name!r}")
        return cls(file_filter=file_filter, **kwargs)
