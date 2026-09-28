"""Shared evaluation library for the released estimator.

Provides the pieces the estimation / completion / external-transfer / baseline
evaluators import: model loading, source-id resolution, the per-limb validity
gate, masked DDIM in-painting, and the subject-macro PCC / finite-aggregation
helpers. This is a library module only: the reported estimation results come from
eval_kinetics_all_cycles.py (which scores every valid cycle of every subject); the
superseded single-fold run entry point (which subsampled cycles) has been removed.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.transform_per_subject import V4LazyTransformV3PerSubject
from pathogait.data.zstats_provenance import sha256_file, verify_checkpoint_zstats
from pathogait.models.dit1d import DiT_B_1D


def local_path(env_var: str, default: str) -> Path:
    """A file or directory the user builds locally from their own copy of the source data.

    Splits, severity labels, polarity tables and cohort arrays are not distributed with the
    code (see docs/training.md). Each location can be redirected through its environment
    variable; the default names the documented layout under the repository root.
    """
    value = os.environ.get(env_var)
    path = Path(value).expanduser() if value else ROOT / default
    return path if path.is_absolute() else ROOT / path


def require_local(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found at {path}. It is produced locally from your own copy of the source "
            "data and is not distributed with the code; docs/training.md describes its format."
        )
    return path


def load_local_json(path: Path, what: str) -> Any:
    return json.loads(require_local(path, what).read_text(encoding="utf-8"))


SPLIT_JSON = local_path("PATHOGAIT_SPLIT_JSON", "data/splits/v4_stage2_split_grffix.json")
SEVERITY_JSON = local_path("PATHOGAIT_SEVERITY_JSON", "data/metadata/per_subject_severity.json")
PER_SUBJECT_PAPER = local_path("PATHOGAIT_FLIP_TABLE", "data/metadata/per_subject_flip_table.json")
PER_SUBJECT_ADDBIO = local_path("PATHOGAIT_FLIP_TABLE_ADDBIO",
                                "data/metadata/per_subject_flip_table_addbio.json")
COHORT_RAW_DIR = local_path("PATHOGAIT_COHORT_RAW_DIR", "data/cohorts_raw")
_zstats_override = os.environ.get("PATHOGAIT_ZSTATS")
if _zstats_override:
    ZSTATS_PATH = Path(_zstats_override).expanduser()
    if not ZSTATS_PATH.is_absolute():
        ZSTATS_PATH = ROOT / ZSTATS_PATH
else:
    # Bundled reported checkpoints were trained in the legacy normalisation
    # space.  Clean retrained checkpoints must set PATHOGAIT_ZSTATS to the
    # Stage-1-only file used during their training.
    ZSTATS_PATH = ROOT / "data" / "zstats" / "global_zstats.json"
ZSTATS_PATH = ZSTATS_PATH.resolve()
if not ZSTATS_PATH.is_file():
    raise FileNotFoundError(f"z-statistics file does not exist: {ZSTATS_PATH}")


def per_subject_transform(meta, require_tables: bool):
    """The evaluation transform with the per-subject polarity tables.

    The tables cover the pretraining corpus and the four development cohorts, so evaluations of
    those cohorts require them. External datasets have no entries in the tables and fall back to
    the global channel conventions, so they only use the tables when present.
    """
    if require_tables:
        paper = str(require_local(PER_SUBJECT_PAPER, "per-subject polarity table"))
        addbio = str(require_local(PER_SUBJECT_ADDBIO, "pretraining-corpus polarity table"))
    else:
        paper = str(PER_SUBJECT_PAPER) if PER_SUBJECT_PAPER.is_file() else None
        addbio = str(PER_SUBJECT_ADDBIO) if PER_SUBJECT_ADDBIO.is_file() else None
    return V4LazyTransformV3PerSubject(
        metadata=meta, global_zstats_path=str(ZSTATS_PATH),
        per_subject_table_paper=paper, per_subject_table_addbio=addbio, ambiguous_policy="no_flip")


SOURCE_TO_ID = {
    "addbio_Camargo2021": 0,
    "addbio_Carter2023": 1,
    "addbio_Moore2015": 2,
    "addbio_Tan2021": 3,
    "addbio_Tan2022": 4,
    "addbio_Wang2023": 5,
    "addbio_vanderZee2022": 6,
    "vdk_healthy": 7,
    "vdk_stroke": 8,
    "cp": 9,
    "normal": 10,
    "bmclab_pd": 11,
}

COHORT_ID_NAME = {0: "cp", 1: "normal", 2: "vdk_stroke", 3: "bmclab_pd"}

CHANNEL_NAMES = [
    "R_Hip_X",
    "R_Hip_Y",
    "R_Hip_Z",
    "L_Hip_X",
    "L_Hip_Y",
    "L_Hip_Z",
    "R_Knee_X",
    "L_Knee_X",
    "R_Ankle_X",
    "L_Ankle_X",
    "R_Hip_Mom_X",
    "R_Hip_Mom_Y",
    "R_Hip_Mom_Z",
    "L_Hip_Mom_X",
    "L_Hip_Mom_Y",
    "L_Hip_Mom_Z",
    "R_Knee_Mom_X",
    "L_Knee_Mom_X",
    "R_Ankle_Mom_X",
    "L_Ankle_Mom_X",
    "R_GRF_X",
    "R_GRF_Y",
    "R_GRF_Z",
    "L_GRF_X",
    "L_GRF_Y",
    "L_GRF_Z",
    "R_EMG_Tib",
    "R_EMG_Gas",
    "R_EMG_Ham",
    "R_EMG_Rec",
    "L_EMG_Tib",
    "L_EMG_Gas",
    "L_EMG_Ham",
    "L_EMG_Rec",
    "R_Pelvis_X",
    "R_Pelvis_Y",
    "R_Pelvis_Z",
    "L_Pelvis_X",
    "L_Pelvis_Y",
    "L_Pelvis_Z",
]

ANGLE_CH = list(range(0, 10))
EXTRA_KNOWN_HIP_YZ = [11, 12, 14, 15]
EMG_CH = list(range(26, 34))
PELVIS_CH = list(range(34, 40))
KNOWN_CH = ANGLE_CH + EXTRA_KNOWN_HIP_YZ + EMG_CH + PELVIS_CH

TARGET_CH = [10, 13, 16, 17, 18, 19, 22, 25]
CANON_8_54CH = [18, 21, 24, 25, 26, 27, 36, 39]
CANON_8_40CH = [KEEP_CHANNELS_54TO40.index(c) for c in CANON_8_54CH]
MOMENT_SAG6_CH = [10, 13, 16, 17, 18, 19]
GRFV_CH = [22, 25]
SAGITTAL6_CH = [10, 13, 16, 17, 22, 25]
BMC_MOMENT_40CH = list(range(10, 20))

CKPT_MAP = {
    "ep050": "ckpt_epoch50.pt",
    "ep100": "ckpt_epoch100.pt",
    "ep150": "ckpt_epoch150.pt",
    "ep175": "ckpt_epoch175.pt",
    "ep199": "ckpt_epoch199.pt",
    "final": "final.pt",
}

SEED = 42
MIN_FINITE_CELLS = 50


def source_type_and_name(src: str) -> tuple[str, str]:
    if src.startswith("addbio_"):
        return "addbio", src[len("addbio_") :]
    return "cohort", src


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else float("nan")


def finite_count(values: Any) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return int(np.isfinite(arr).sum())


def row_group_means(arr: np.ndarray, target_ch: list[int], group_ch: list[int]) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    cols = [target_ch.index(c) for c in group_ch if c in target_ch]
    out = np.full((arr.shape[0],), np.nan, dtype=np.float64)
    for i in range(arr.shape[0]):
        out[i] = finite_mean(arr[i, cols])
    return out


def safe_pcc(a: np.ndarray, b: np.ndarray, min_finite: int = MIN_FINITE_CELLS) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < min_finite:
        return float("nan")
    aa = a[m].astype(np.float64, copy=False)
    bb = b[m].astype(np.float64, copy=False)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def get_val_pairs(split: dict[str, Any], fold_name: str) -> list[tuple[str, str]]:
    if fold_name.startswith("cv5_"):
        fold_idx = int(fold_name[len("cv5_") :])
        return [tuple(x) for x in split["cv5"][fold_idx]["val"]]
    raise ValueError(f"unsupported fold name: {fold_name}")


def load_valid_mask_40(raw_path: Path, n_cycles: int) -> np.ndarray:
    mask_path = raw_path.with_name(raw_path.stem + "_mask.npy")
    if not mask_path.exists():
        # No validity mask on disk -> every cycle is treated as valid. This is a fallback,
        # not the reported path (every reported subject ships a mask); warn so a missing
        # mask never silently inflates the scored-cycle count.
        print(f"[WARN no mask, assuming all-valid] {mask_path.name}", flush=True)
        return np.ones((n_cycles, 40), dtype=bool)
    m54 = np.load(str(mask_path))
    if m54.ndim == 1:
        m54 = np.tile(m54[None, :], (n_cycles, 1))
    return m54[:, KEEP_CHANNELS_54TO40].astype(bool, copy=True)


def load_model(ckpt_path: Path, device: torch.device):
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        zstats_warning = verify_checkpoint_zstats(
            ckpt,
            ZSTATS_PATH,
            sha256_file(ZSTATS_PATH),
            f"evaluation checkpoint {ckpt_path}",
        )
        if zstats_warning:
            print(f"[WARN] {zstats_warning}", flush=True)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = DiT_B_1D(
        in_channels=int(args.get("in_channels", 40)),
        patch_size=int(args.get("patch_size", 4)),
        hidden_size=int(args.get("hidden_size", 384)),
        depth=int(args.get("depth", 12)),
        num_heads=int(args.get("num_heads", 6)),
        seq_len=100,
        num_sources=int(args.get("num_sources", 12)),
        num_cohorts=int(args.get("num_cohorts", 4)),
        num_severities=int(args.get("num_severities", 6)),
        cfg_drop_prob=float(args.get("cfg_drop_prob", 0.10)),
    ).to(device)
    model.eval()
    if isinstance(ckpt, dict) and ckpt.get("ema") is not None:
        state = ckpt["ema"]
        weights = "EMA"
    elif isinstance(ckpt, dict) and ckpt.get("model") is not None:
        state = ckpt["model"]
        weights = "online"
    else:
        state = ckpt
        weights = "state_dict"
    model.load_state_dict(state, strict=True)
    input_mask_token = None
    if isinstance(ckpt, dict):
        input_mask_token = ckpt.get("input_mask_token_ema")
        if input_mask_token is None:
            input_mask_token = ckpt.get("input_mask_token")
    return model, ckpt, weights, input_mask_token


def build_known_mask(gt_norm: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    bsz = gt_norm.shape[0]
    known_ch = torch.zeros((40,), dtype=torch.bool, device=gt_norm.device)
    known_ch[KNOWN_CH] = True
    finite = torch.isfinite(gt_norm)
    valid = valid_mask.to(device=gt_norm.device, dtype=torch.bool).unsqueeze(-1)
    return (known_ch.view(1, 40, 1) & valid & finite).float().expand(bsz, -1, -1)
