"""CLI inference from user-supplied, harmonised angle cycles."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from pathogait_api import PathoGait, COHORT_CFG, seed_values
from input_configurations import INPUT_SETS


def run(kind):
    if kind not in ("predict_kinetics", "partial_input"):
        raise ValueError("unsupported example")
    parser = argparse.ArgumentParser(description=f"PathoGaitFM {kind}: estimate kinetics from harmonised degree angles.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/v4_stage2_final8ch_ALLDATA/final.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cohort", choices=list(COHORT_CFG), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / kind)
    parser.add_argument("--input", type=Path, required=True, help="harmonised degree angles, (100,16) or (B,100,16) .npy")
    parser.add_argument("--mask", type=Path, help="optional observed-channel boolean (16,) or (B,16) .npy")
    parser.add_argument("--source-id", type=int, help="0-11 known training source; default 12 (unknown source)")
    parser.add_argument("--severity-id", type=int, help="explicit training-schema severity code; default 6 (unknown)")
    parser.add_argument("--seeds", type=int, default=3, help="paper setting is 3; 1 is a faster interface check")
    args = parser.parse_args()
    seed_values(args.seed, args.seeds)
    angles = np.load(args.input, allow_pickle=False)
    mask = np.load(args.mask, allow_pickle=False) if args.mask else None
    model = PathoGait(args.checkpoint, args.device)
    meta = model.metadata()
    meta.update({"example": kind, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    meta["input_origin"] = str(args.input)
    meta["runs"] = {}
    arrays = {}
    for name in (INPUT_SETS if kind == "partial_input" else ["full"]):
        arrays[name + "_model_z"], arrays[name + "_kinetics"] = model.predict(
            angles, args.cohort, mask, name, args.seed, args.seeds, args.source_id, args.severity_id)
        meta["runs"][name] = model.metadata()["prediction"]
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", **arrays)
    (args.out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved", args.out / "predictions.npz", {k: list(v.shape) for k, v in arrays.items()})
    print("This example verifies the interface; it does not measure predictive accuracy.")
