"""Public inference interface over the unchanged paper sampling functions.

Input angles must already use the harmonised sign and zero conventions.
This module performs normalisation, not camera/IMU processing or inverse kinematics.
"""
from pathlib import Path
from argparse import Namespace
import json
import sys
from numbers import Integral

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "model", ROOT / "scripts/lib"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from eval_common import load_model, ZSTATS_PATH, CHANNEL_NAMES, TARGET_CH, SEED
from drop_channels import DROP_16CH
from input_configurations import INPUT_MASKS
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.pelvis_contract import FIELD, LEGACY, validate_representation
from pathogait.data.zstats_provenance import sha256_file
from pathogait.diffusion.ddpm import DDPMScheduler

INPUT_CH = list(range(10)) + list(range(34, 40))
INPUT_NAMES = [CHANNEL_NAMES[i] for i in INPUT_CH]
TARGET_UNITS = ["Nm/kg"] * 6 + ["BW"] * 2
COHORT_CFG = {"cp": {"cid": 0}, "normal": {"cid": 1},
              "vdk_stroke": {"cid": 2}, "bmclab_pd": {"cid": 3}}
MASKS = INPUT_MASKS
SEED_STRIDE = 7919


def integer_parameter(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def seed_values(seed, seeds):
    seeds = integer_parameter(seeds, "seeds", 1, 64)
    seed = integer_parameter(seed, "seed", 0, 2**63 - 1 - (seeds - 1) * SEED_STRIDE)
    return [seed + i * SEED_STRIDE for i in range(seeds)]


def channel_stats():
    stats = json.loads(ZSTATS_PATH.read_text(encoding="utf-8"))
    mean, scale = np.zeros(40, np.float32), np.ones(40, np.float32)
    for i, raw_index in enumerate(KEEP_CHANNELS_54TO40):
        if 40 <= raw_index <= 47:
            continue
        item = stats[str(raw_index)]
        mean[i], scale[i] = item["mean"], 3 * item["std"]
        if not np.isfinite(mean[i]) or not np.isfinite(scale[i]) or scale[i] <= 0:
            raise ValueError(f"non-positive normalisation scale for channel {raw_index}")
    return mean, scale


def physical(z):
    """Invert the affine normalisation; clipping cannot be undone."""
    mean, scale = channel_stats()
    return np.asarray(z) * scale[None, :, None] + mean[None, :, None]


def prepare_angles(angles, mask=None):
    """Return (B,40,100) inputs and (B,40) valid mask; targets stay hidden."""
    raw = np.asarray(angles)
    if raw.dtype.kind not in "fiu":
        raise ValueError("angles must be a real numeric array in degrees")
    with np.errstate(over="ignore", invalid="ignore"):
        x = raw.astype(np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (100, 16) or x.shape[0] == 0:
        raise ValueError(f"expected (B,100,16) or (100,16), got {x.shape}")
    if np.isinf(x).any():
        raise ValueError("infinite angle input")
    observed = np.isfinite(x).all(axis=1)
    partial = np.isfinite(x).any(axis=1) & ~observed
    if partial.any():
        raise ValueError("missing angle channels must be all-NaN across the cycle, or use a channel mask")
    if mask is not None:
        given = np.asarray(mask)
        if given.shape == (16,):
            given = np.broadcast_to(given, observed.shape)
        if given.shape != observed.shape or not np.isin(given, [0, 1]).all():
            raise ValueError("mask must be boolean/0/1 with shape (16,) or (B,16)")
        observed &= given.astype(bool)
    if not observed.any(axis=1).all():
        raise ValueError("prediction requires at least one observed angle per cycle")
    mean, scale = channel_stats()
    z = np.zeros((len(x), 40, 100), np.float32)
    z[:, INPUT_CH] = np.clip((np.nan_to_num(x).transpose(0, 2, 1) - mean[INPUT_CH, None]) / scale[INPUT_CH, None], -1.5, 1.5)
    valid = np.ones((len(x), 40), dtype=bool)
    valid[:, INPUT_CH] = observed
    return z, valid


class PathoGait:
    def __init__(self, checkpoint, device="cpu"):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing at {self.checkpoint}. See checkpoints/CHECKPOINTS_MANIFEST.md.")
        self.device = torch.device(device)
        self.model, state, self.weights_kind, self.token = load_model(self.checkpoint, self.device)
        checkpoint_args = state.get("args", {})
        if isinstance(checkpoint_args, Namespace):
            checkpoint_args = vars(checkpoint_args)
        self.bmclab_pelvis_representation = validate_representation(
            checkpoint_args.get(FIELD, LEGACY))
        del state
        if self.token is None:
            raise ValueError("Use a Stage-2 checkpoint carrying its learned input mask token")
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(self.device)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.last_prediction = None

    def conditions(self, cohort, source_id=None, severity_id=None):
        if not isinstance(cohort, str) or cohort not in COHORT_CFG:
            raise ValueError(f"cohort must be one of {', '.join(COHORT_CFG)}")
        cfg = COHORT_CFG[cohort]
        # Unknown source and severity use the trained null embeddings. Cohort is
        # explicit; the interface never assigns a disease or severity from angles.
        source_id = self.model.num_sources if source_id is None else source_id
        severity_id = self.model.num_severities if severity_id is None else severity_id
        return (integer_parameter(source_id, "source_id", 0, self.model.num_sources),
                integer_parameter(cfg["cid"], "cohort_id", 0, self.model.num_cohorts - 1),
                integer_parameter(severity_id, "severity_id", 0, self.model.num_severities))

    @torch.inference_mode()
    def predict(self, angles, cohort, mask=None, input_set="full", seed=SEED, seeds=3,
                source_id=None, severity_id=None):
        self.last_prediction = None
        run_seeds = seed_values(seed, seeds)
        ids = self.conditions(cohort, source_id, severity_id)
        if not isinstance(input_set, str) or input_set not in MASKS:
            raise ValueError(f"input_set must be one of {', '.join(MASKS)}")
        z, valid = prepare_angles(angles, mask)
        valid[:, MASKS[input_set]] = False
        if not valid[:, INPUT_CH].any(axis=1).all():
            raise ValueError("input configuration leaves no observed kinematics")
        if (cohort == "bmclab_pd"
                and getattr(self, "bmclab_pelvis_representation", LEGACY) == LEGACY
                and valid[:, 34:40].any()):
            raise ValueError(
                "This BMClab checkpoint expects pelvis translations. Omit pelvis angles "
                "(no_pelvis, NaN or mask), or use a matching angular checkpoint.")
        data = torch.from_numpy(z).to(self.device)
        validity = torch.from_numpy(valid).to(self.device)
        sid, cid, svid = [torch.full((len(z),), i, dtype=torch.long, device=self.device) for i in ids]
        samples = []
        for run_seed in run_seeds:
            torch.manual_seed(run_seed)
            samples.append(tweedie_inpaint_drop(self.model, self.scheduler, data, validity, sid, cid, svid,
                                               50, 1.0, self.device, self.token, CHECKPOINT_STEPS, DROP_16CH))
        predicted = torch.stack(samples).mean(0).cpu().numpy()
        if not np.isfinite(predicted).all():
            raise RuntimeError("non-finite model prediction")
        self.last_prediction = {"cohort": cohort, "source_id": ids[0], "cohort_id": ids[1],
                                "severity_id": ids[2], "seed_values": run_seeds,
                                "input_set": input_set, "batch_size": len(z),
                                "observed_angle_mask": valid[:, INPUT_CH].tolist()}
        return predicted, physical(predicted)[:, TARGET_CH].transpose(0, 2, 1)

    def metadata(self):
        return {"checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
                "zstats_sha256": sha256_file(ZSTATS_PATH), "weights": self.weights_kind,
                "torch": torch.__version__, "numpy": np.__version__, "device": str(self.device),
                "ddim_steps": 50, "cfg": 1.0, "saved_step_indices_zero_based": list(CHECKPOINT_STEPS),
                "aggregation": "cellwise trimmed mean of six x0 snapshots and final state, then mean across seeds",
                "input_names": INPUT_NAMES, "input_units": "degrees",
                "bmclab_pelvis_representation": self.bmclab_pelvis_representation,
                "target_names": [CHANNEL_NAMES[i] for i in TARGET_CH],
                "target_units": TARGET_UNITS, "prediction": self.last_prediction}
