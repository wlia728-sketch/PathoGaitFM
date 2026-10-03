"""Keep BMClab input coordinates, transforms and checkpoints in one representation."""
from __future__ import annotations

from argparse import Namespace
import csv
import os
from pathlib import Path
from typing import Any, Mapping


LEGACY = "legacy_translations_m"
ANGLES = "opensim_angles_deg"
REPRESENTATIONS = (LEGACY, ANGLES)
FIELD = "bmclab_pelvis_representation"


def validate_representation(value: str) -> str:
    if not isinstance(value, str) or value not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown BMClab pelvis representation {value!r}; "
            f"expected one of {REPRESENTATIONS}.")
    return value


def configured_representation() -> str:
    return validate_representation(
        os.environ.get("PATHOGAIT_BMCLAB_PELVIS_REPRESENTATION", LEGACY))


def validate_bmclab_input(raw_path: Path, expected: str) -> None:
    """Check every declared metadata row; unlabelled historical arrays are legacy."""
    raw_path = Path(raw_path)
    if raw_path.parent.name != "bmclab_pd":
        return
    expected = validate_representation(expected)
    meta_path = raw_path.with_name(raw_path.stem + "_meta.csv")
    actual = LEGACY
    if meta_path.is_file():
        with meta_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "pelvis_representation" in (reader.fieldnames or []):
                values = {validate_representation(row.get("pelvis_representation"))
                          for row in reader}
                if len(values) != 1:
                    raise ValueError(
                        f"{meta_path} must declare one BMClab pelvis representation "
                        f"across all rows; found {sorted(values)}.")
                actual = values.pop()
    if actual != expected:
        raise ValueError(
            f"{raw_path} uses BMClab pelvis representation {actual!r}, "
            f"but the configured transform expects {expected!r}. "
            "Use matching data, preprocessing and checkpoints.")


def verify_checkpoint_representation(
    checkpoint: Mapping[str, Any], expected: str, label: str,
) -> None:
    """Missing checkpoint fields denote the historical translation input space."""
    expected = validate_representation(expected)
    args = checkpoint.get("args", {})
    if isinstance(args, Namespace):
        args = vars(args)
    if not isinstance(args, Mapping):
        raise ValueError(f"{label} has invalid checkpoint args: expected a mapping or Namespace.")
    actual = validate_representation(args.get(FIELD, LEGACY))
    if actual != expected:
        raise RuntimeError(
            f"{label} was trained with BMClab pelvis representation {actual!r}, "
            f"but this run uses {expected!r}. Select a matching checkpoint "
            "or train a model for the selected input representation.")
