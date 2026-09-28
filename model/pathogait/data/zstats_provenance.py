"""Bind a training run and its checkpoints to one z-statistics file.

Normalisation constants define the numerical space in which both the model
inputs and diffusion targets live.  A checkpoint therefore cannot safely be
resumed or warm-started with different constants.
"""
from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


LEGACY_ZSTATS_BASENAMES = {"global_zstats.json", "v4_global_zstats.json"}
LEGACY_ZSTATS_SHA256 = "155a4bb994194c995a22eb6056ada442343a0f07bd9850112d10a3820d0109e6"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_zstats_argument(args: Namespace) -> Path:
    """Resolve ``args.zstats``, validate it, and record its content digest."""
    path = Path(args.zstats).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"z-statistics file does not exist: {path}")
    args.zstats = str(path)
    args.zstats_sha256 = sha256_file(path)
    return path


def _checkpoint_identity(checkpoint: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, Namespace):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, Mapping):
        return None, None
    digest = saved_args.get("zstats_sha256")
    path = saved_args.get("zstats")
    return (
        str(digest) if digest else None,
        str(path) if path else None,
    )


def verify_checkpoint_zstats(
    checkpoint: Mapping[str, Any],
    current_path: Path,
    current_sha256: str,
    checkpoint_label: str,
) -> Optional[str]:
    """Fail if a checkpoint and current run use different normalisation.

    New checkpoints carry a content digest and receive an exact check.  The
    bundled historical checkpoints predate that field, so their two known
    legacy basenames are accepted only when the selected file has the exact
    archived legacy digest.  All other unverified combinations fail closed.
    """
    saved_sha256, saved_path = _checkpoint_identity(checkpoint)
    if saved_sha256:
        if saved_sha256.lower() != current_sha256.lower():
            raise RuntimeError(
                f"{checkpoint_label} was trained with z-stats SHA-256 "
                f"{saved_sha256}, but the current file {current_path} has "
                f"SHA-256 {current_sha256}. Retrain or select the matching "
                "normalisation file."
            )
        return None

    saved_name = Path(saved_path).name if saved_path else None
    legacy_alias = (
        saved_name in LEGACY_ZSTATS_BASENAMES
        and current_path.name in LEGACY_ZSTATS_BASENAMES
        and current_sha256 == LEGACY_ZSTATS_SHA256
    )
    if saved_path and legacy_alias:
        return (
            f"{checkpoint_label} predates z-stats hashing; accepted because "
            f"the recorded legacy basename {saved_name!r} and the archived "
            "legacy SHA-256 both match."
        )

    recorded = saved_path or "<not recorded>"
    raise RuntimeError(
        f"cannot prove that {checkpoint_label} uses {current_path.name}; "
        f"its recorded z-stats path is {recorded!r} and it has no SHA-256. "
        "Do not mix checkpoints and normalisation spaces."
    )
