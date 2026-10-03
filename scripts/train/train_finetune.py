"""V4 Stage 2 production trainer â€” DiT-B/1D fine-tune w/ cohort+severity cond.

Derived from model/pathogait/training/train_pretrain.py (kept
read-only). Stage 2 deltas (Round-2 audit 2026-05-18 locked):

  - Resume from Stage 1 final / ckpt_epoch{N}.pt (warm-start)
  - --fold (cv5_0..4)
  - --severity-label per_subject_severity.json (132 file -> cohort_id/severity_id)
  - Constant 1e-5 LR after warmup (default) -- prevent catastrophic forgetting
  - epoch_pseudo_length 10,000 (vs 300k Stage 1) -- ~17.6 effective pass/fold
  - epochs 200 (vs 25)
  - cfg_drop_prob 0.10 per-condition (model handles inside get_conditioning)
  - Source path uses V4BalancedDatasetV3 file_filter from v4_stage2_split.json

Reuses (no copy):
  - DiT_B_1D + EMA from pathogait
  - V4BalancedDatasetV3 + collate
  - DDPMScheduler + nan_safe_mse_loss

CLI (dry-run smoke, 1 step CPU, does NOT write ckpt):
    python scripts/train/train_finetune.py \\
        --resume checkpoints/v4_stage1_unified/ckpt_epoch9.pt \\
        --severity-label data/metadata/per_subject_severity.json \\
        --fold cv5_0 \\
        --epochs 1 --batch-size 4 --num-workers 0 --cache-size 8 \\
        --epoch-pseudo-length 4 --val-pseudo-length 4 \\
        --lr 1e-5 --warmup-steps 0 --constant-lr \\
        --device cpu --output /tmp/stage2_dryrun/ --dry-run

Smoke-test per-fold launch (defaults; the reported model used --epochs 200, see README):
    FOLD=cv5_0
    SEED=42
    OUT=checkpoints/v4_stage2_${FOLD}
    nohup python scripts/train/train_finetune.py \\
        --resume checkpoints/v4_stage1_unified/final.pt \\
        --severity-label data/metadata/per_subject_severity.json \\
        --fold $FOLD --output $OUT --seed $SEED \\
        --epochs 12 --epoch-pseudo-length 10000 --val-pseudo-length 1500 \\
        --lr 1e-5 --warmup-steps 1000 --constant-lr \\
        --batch-size 64 --num-workers 4 --cache-size 256 \\
        --cfg-drop-prob 0.10 --device cuda --amp \\
        > "$OUT/launch.log" 2>&1 &
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pathogait.data.dataset import V4BalancedDatasetV3, collate
from pathogait.data.run_provenance import capture_settings, save_training_manifest, validate_training_scope
from pathogait.data.zstats_provenance import (
    bind_zstats_argument,
    verify_checkpoint_zstats,
)
from pathogait.data.pelvis_contract import (
    REPRESENTATIONS, configured_representation, verify_checkpoint_representation,
)
from pathogait.diffusion.ddpm import DDPMScheduler
from pathogait.models.dit1d import DiT_B_1D, count_params
from pathogait.models.diffusion import EMA
from pathogait.training.losses import nan_safe_mse_loss
from pathogait.data.channels import KEEP_CHANNELS_54TO40



# ============================================================
# EXCLUDED_T1_CHANNELS_PER_COHORT - per-cohort channel exclusion
# ============================================================
# Level 1 (2026-05-20, knee moments only): partial exclusion based on the L3
# hypothesis verification PCC floor for knee moments specifically. Kept here
# as a documented historical baseline:
#   "bmclab_pd": [24, 25]   # R/L Knee_Moment_X
#
# Level 2 (2026-05-21, ACTIVE â€” all bmclab joint moments excluded):
#   Promoted to full moment exclusion after the cv5_0 L1 sanity eval showed
#   the convention drift extends across the entire bmclab moment set, not
#   just knees:
#     ch 18 R_Hip_Moment_X    PCC +0.381  (post sign-flip correction)
#     ch 19 R_Hip_Moment_Y    PCC -0.103
#     ch 20 R_Hip_Moment_Z    (frontal/transverse hip)
#     ch 21 L_Hip_Moment_X    PCC +0.613
#     ch 22 L_Hip_Moment_Y    (frontal hip)
#     ch 23 L_Hip_Moment_Z    (transverse hip)
#     ch 24 R_Knee_Moment_X   PCC -0.172  (raw, no correction recovers)
#     ch 25 L_Knee_Moment_X   PCC +0.052
#     ch 26 R_Ankle_Moment_X  PCC -0.172  (raw)
#     ch 27 L_Ankle_Moment_X  PCC +0.272  (best of bmclab moments)
#   These channels were computed externally via OpenSim inverse dynamics
#   under a convention that systematically diverges from the project
#   reference cohorts; the residual convention drift cannot be resolved
#   by sign-flip, offset, or phase-shift correction (best post-correction
#   PCC stays at 0.38-0.88 with the best ch 27 alone reaching 0.88 only
#   after a +27% phase shift + sign flip).
#
#   bmclab_pd is therefore retained in training but contributes loss only
#   through directly-measured channels: kinematics (angles + pelvis), GRF
#   (force-plate, gold standard), and EMG. The moment-channel loss is
#   masked out at the per-sample valid_mask level.
#
# Paper Methods M2 disclose: 4-layer correction framework (Layer 1 polarity,
# Layer 2 sign-flip, Layer 3 offset, Layer 4 cohort-conditioned exclusion).
# Methods M8 ablation reports T1 metric with vs without bmclab moments.
#
# Indices below are in 54-channel raw space; trainer input is 40-channel
# (post KEEP_CHANNELS_54TO40 reduction). The runtime loop translates via
# KEEP_CHANNELS_54TO40.index(ch).
EXCLUDED_T1_CHANNELS_PER_COHORT = {
    "bmclab_pd": [18, 19, 20, 21, 22, 23,    # R/L Hip_Moment X/Y/Z (6 ch)
                  24, 25,                     # R/L Knee_Moment_X (2 ch)
                  26, 27],                    # R/L Ankle_Moment_X (2 ch)
    # Other cohorts: empty - no exclusion
}

# Precompute the 54-ch -> 40-ch translation once at module load.
_EXCLUDED_40CH_PER_COHORT = {
    src: [KEEP_CHANNELS_54TO40.index(ch) for ch in chs
          if ch in KEEP_CHANNELS_54TO40]
    for src, chs in EXCLUDED_T1_CHANNELS_PER_COHORT.items()
}


V4_SOURCE_TO_ID = {
    "addbio_Camargo2021":   0,
    "addbio_Carter2023":    1,
    "addbio_Moore2015":     2,
    "addbio_Tan2021":       3,
    "addbio_Tan2022":       4,
    "addbio_Wang2023":      5,
    "addbio_vanderZee2022": 6,
    "vdk_healthy":          7,
    "vdk_stroke":           8,
    "cp":                   9,
    "normal":               10,
    "bmclab_pd":            11,
}

# fold short -> split.json lookup key for V4BalancedDatasetV3.from_split.
# split.json uses 'cv5_fold{N}_{train|val}'.
def _resolve_split_names(fold: str) -> Tuple[str, str]:
    if fold.startswith("cv5_"):
        n = int(fold[len("cv5_"):])
        if not 0 <= n <= 4:
            raise ValueError(f"cv5 fold index out of range: {fold}")
        return f"cv5_fold{n}_train", f"cv5_fold{n}_val"
    raise ValueError(f"--fold must be cv5_{{0..4}}, got {fold!r}")


# ----------------------------------------------------------------------
def _safe_save(payload, target: Path, retries: int = 3):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(retries):
        try:
            torch.save(payload, tmp)
            if target.exists():
                target.unlink()
            tmp.replace(target)
            return True
        except (OSError, RuntimeError) as e:
            if attempt == retries - 1:
                print(f"  [WARN] checkpoint save failed: {e}")
                return False
            time.sleep(2)
    return False


def _make_lr_lambda(warmup_steps: int, total_steps: int, constant_after_warmup: bool):
    """Stage 2 default: linear warmup -> hold constant. Optional cosine via --no-constant-lr."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if constant_after_warmup:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


# ----------------------------------------------------------------------
def load_severity_labels(json_path: str) -> Dict[Tuple[str, str], Dict]:
    """Flatten mappings[cohort][stem] -> dict into (src, stem) -> dict.

    `src` is the V4 source name (matches split json + V4BalancedDatasetV3
    source_name): cp / normal / vdk_stroke / bmclab_pd.
    """
    with open(json_path) as f:
        data = json.load(f)
    if "mappings" not in data:
        raise ValueError(
            f"severity json {json_path} missing 'mappings' top-level key; "
            f"got keys={list(data.keys())}")
    flat: Dict[Tuple[str, str], Dict] = {}
    for src, by_stem in data["mappings"].items():
        for stem, entry in by_stem.items():
            flat[(src, stem)] = entry
    return flat


def verify_severity_coverage(
    split_pairs: List[Tuple[str, str]],
    sev_map: Dict[Tuple[str, str], Dict],
) -> None:
    missing = [p for p in split_pairs if p not in sev_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} (source, stem) pairs in split have no severity entry. "
            f"First 10: {missing[:10]}")


def _read_fold_pairs(split_json_path: str, fold: str
                     ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (train_pairs, val_pairs) for the requested fold."""
    with open(split_json_path) as f:
        d = json.load(f)
    if fold.startswith("cv5_"):
        n = int(fold[len("cv5_"):])
        if n >= len(d.get("cv5", [])):
            raise NotImplementedError(
                f"split json has {len(d.get('cv5', []))} cv5 folds; "
                f"requested cv5_{n}")
        entry = d["cv5"][n]
        return ([tuple(x) for x in entry["train"]],
                [tuple(x) for x in entry["val"]])
    raise ValueError(f"unrecognized fold {fold!r}")


# ----------------------------------------------------------------------
def _maybe_build_per_subject_transform(args):
    """Return a V4LazyTransformV3PerSubject if --use-per-subject-flip is set,
    else None (caller falls back to default V4LazyTransformV3)."""
    if not getattr(args, "use_per_subject_flip", False):
        return None
    from pathogait.data.transform_per_subject import (
        V4LazyTransformV3PerSubject,
    )
    from pathogait.data.subject_metadata import V4SubjectMetadata
    meta = V4SubjectMetadata()
    tfm = V4LazyTransformV3PerSubject(
        metadata=meta,
        global_zstats_path=args.zstats,
        per_subject_table_paper=args.per_subject_flip_paper,
        per_subject_table_addbio=args.per_subject_flip_addbio,
        ambiguous_policy=args.ambiguous_policy,
        bmclab_pelvis_representation=args.bmclab_pelvis_representation,
    )
    return tfm


def _build_loaders(args, train_split_name: str, val_split_name: str):
    transform_override = _maybe_build_per_subject_transform(args)
    if transform_override is not None:
        n_sub = transform_override.n_subjects_in_table()
        print(f"[Stage2] using V4LazyTransformV3PerSubject  "
              f"(paper={n_sub['paper']}, addbio={n_sub['addbio']}, "
              f"total={n_sub['total']})  "
              f"ambiguous_policy={args.ambiguous_policy}")

    train_ds = V4BalancedDatasetV3.from_split(
        args.split, train_split_name,
        global_zstats_path=args.zstats,
        pseudo_length=args.epoch_pseudo_length,
        seed=args.seed,
        cache_size=args.cache_size,
        transform_override=transform_override,
        sampling="dynamics_cycle_uniform",
        bmclab_pelvis_representation=args.bmclab_pelvis_representation,
    )
    val_ds = V4BalancedDatasetV3.from_split(
        args.split, val_split_name,
        global_zstats_path=args.zstats,
        pseudo_length=args.val_pseudo_length,
        seed=args.seed + 1,
        cache_size=max(8, args.cache_size // 2),
        transform_override=transform_override,
        sampling="dynamics_cycle_uniform",
        bmclab_pelvis_representation=args.bmclab_pelvis_representation,
    )
    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate)
    if args.num_workers > 0:
        loader_kwargs.update(
            num_workers=args.num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=(args.device == "cuda"),
        )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    return train_ds, val_ds, train_loader, val_loader


# ----------------------------------------------------------------------
def _step_forward(model, sched, batch, device, sev_map, seq_len: int = 100,
                  mask_bmclab_moments: bool = False,
                  input_mask_token: Optional[torch.Tensor] = None,
                  input_mask_token_scope: str = "all_invalid",
                  disable_cohort_channel_exclusion: bool = False,
                  global_step: int = 0):
    """Single training/validation forward (no backward).

    Pulls cohort_id + severity_id from sev_map keyed by
    (source_name, file_stem_without_npy).

    Returns:
      loss: scalar tensor.
      aux: dict with v_target_std, v_pred_std, source_ids, cohort_ids,
        severity_ids.
    """
    validate_training_scope()
    x0 = batch["x"].to(device, non_blocking=True)              # (B, 40, 101)
    x0 = x0[..., :seq_len]
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    source_names = batch["source_name"]
    file_names = batch["file"]                                  # list[str], '*.npy'
    explicit_input_missing_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

    # EXPERIMENT (env, default off): decouple input-drop from loss-target.
    # PATHOGAIT_INPUT_DROP_CH: 40ch indices forced token-filled at INPUT only
    #   (added to explicit_input_missing_mask; loss untouched). Used to remove
    #   EMG + non-sagittal hip moments from KNOWN inputs (16ch kinematic input).
    # PATHOGAIT_LOSS_TARGET_CH: if set, loss is supervised ONLY on these 40ch
    #   indices (applied as an extra multiplicative mask on loss_mask later).
    _exp_input_drop = os.environ.get("PATHOGAIT_INPUT_DROP_CH", "")
    _exp_loss_target = os.environ.get("PATHOGAIT_LOSS_TARGET_CH", "")
    if _exp_input_drop:
        for ch in [int(c) for c in _exp_input_drop.split(",") if c.strip() != ""]:
            explicit_input_missing_mask[:, ch] = True

    # EMG MODALITY DROPOUT (env, 2026-06-01): instead of permanently dropping EMG at
    # input, drop the listed channels PER-SAMPLE with prob p. The model then learns to
    # predict with EMG present AND absent -> EMG enters the shared representation while
    # the moment task cannot become EMG-dependent (structural non-degradation). Channels
    # that are already invalid (PD/AddBio have no EMG, valid_mask=0) are handled normally.
    # Default unset -> no change to existing behavior.
    _emg_drop_ch = os.environ.get("PATHOGAIT_EMG_DROPOUT_CH", "")
    _emg_drop_p = float(os.environ.get("PATHOGAIT_EMG_DROPOUT_P", "0.5"))
    if _emg_drop_ch:
        _emg_chs = [int(c) for c in _emg_drop_ch.split(",") if c.strip() != ""]
        B = explicit_input_missing_mask.shape[0]
        # per-sample Bernoulli mask: True = drop this sample's EMG at input this step
        _drop_samp = (torch.rand(B, device=explicit_input_missing_mask.device) < _emg_drop_p)
        for ch in _emg_chs:
            explicit_input_missing_mask[_drop_samp, ch] = True

    # cohort-conditioned channel
    # exclusion. For bmclab_pd, zero out valid_mask on all 10 joint-moment
    # channels (54-ch [18..27], mapped to 40-ch [10..19] via
    # KEEP_CHANNELS_54TO40) so they contribute zero loss. bmclab kinematics,
    # GRF, EMG, and pelvis still drive the loss. See
    # EXCLUDED_T1_CHANNELS_PER_COHORT docstring for the L1->L2 rationale.
    # when mask_bmclab_moments=True, ALSO zero out
    # bmclab moment cells in x0 BEFORE the forward pass. Loss-side masking
    # is unchanged (valid_mask zeroing). This isolates the model from the
    # documented-broken bmclab moment values during attention/conditioning.
    if (not disable_cohort_channel_exclusion) and any(
            src in _EXCLUDED_40CH_PER_COHORT and _EXCLUDED_40CH_PER_COHORT[src]
            for src in source_names):
        for i, src in enumerate(source_names):
            for ch_40 in _EXCLUDED_40CH_PER_COHORT.get(src, ()):
                valid_mask[i, ch_40] = False
                explicit_input_missing_mask[i, ch_40] = True
                if mask_bmclab_moments and input_mask_token is None:
                    x0[i, ch_40, :] = 0.0

    source_ids = torch.tensor(
        [V4_SOURCE_TO_ID[s] for s in source_names],
        device=device, dtype=torch.long)
    cohort_ids = []
    severity_ids = []
    for src, fname in zip(source_names, file_names):
        stem = Path(fname).stem
        key = (src, stem)
        entry = sev_map.get(key)
        if entry is None:
            raise RuntimeError(
                f"Severity lookup miss for {key}; "
                f"either split/severity diverged or batch returned unexpected file.")
        cohort_ids.append(int(entry["cohort_id"]))
        severity_ids.append(int(entry["severity_id"]))
    cohort_id_t = torch.tensor(cohort_ids, device=device, dtype=torch.long)
    severity_id_t = torch.tensor(severity_ids, device=device, dtype=torch.long)

    B = x0.shape[0]
    t = torch.randint(0, sched.num_train_timesteps, (B,), device=device)
    noise = torch.randn_like(x0)
    x0_safe = torch.nan_to_num(x0, nan=0.0)
    input_missing_mask = None
    x0_for_xt = x0_safe
    if input_mask_token is not None:
        if input_mask_token_scope == "all_invalid":
            # base = invalid cells; EXPERIMENT: also token-fill PATHOGAIT_INPUT_DROP_CH
            input_missing_mask = (~valid_mask.bool()) | explicit_input_missing_mask.bool()
        elif input_mask_token_scope == "bmclab_moments":
            input_missing_mask = explicit_input_missing_mask.bool()
        else:
            raise ValueError(
                f"Unknown input_mask_token_scope={input_mask_token_scope!r}; "
                "expected 'all_invalid' or 'bmclab_moments'.")
        tok = input_mask_token.to(device=x0_safe.device, dtype=x0_safe.dtype)
        if tok.ndim == 2:
            tok = tok.unsqueeze(-1)
        tok = tok.expand(B, -1, seq_len)
        x0_for_xt = torch.where(input_missing_mask.unsqueeze(-1), tok, x0_safe)
    xt_full = sched.add_noise(x0_for_xt, noise, t)
    v_target = sched.get_velocity(x0_safe, noise, t)
    v_target = torch.where(torch.isnan(x0), x0, v_target)

    # Unconditional v-prediction: loss spans the full channel set, subject to
    # NaN masking inside nan_safe_mse_loss.
    xt = xt_full
    loss_mask = valid_mask

    # EXPERIMENT (env): restrict loss supervision to PATHOGAIT_LOSS_TARGET_CH only.
    # Multiplicative channel mask on loss_mask -> only target channels get gradient.
    # Does NOT touch input_missing_mask (input-drop is independent), so a channel can
    # be input-dropped yet still a loss target (e.g. non-sag hip moment in exp B).
    if _exp_loss_target:
        tgt = [int(c) for c in _exp_loss_target.split(",") if c.strip() != ""]
        ch_keep = torch.zeros(40, device=device)
        ch_keep[tgt] = 1.0
        # AUX loss channels (env, 2026-06-01): add EMG (or others) to the loss at a
        # REDUCED weight so they enter the representation without diluting the primary
        # (moment/vGRF) target. valid_mask still gates per-cell, so PD/AddBio (EMG
        # invalid) contribute nothing. Default unset -> unchanged.
        _aux_ch = os.environ.get("PATHOGAIT_AUX_LOSS_CH", "")
        _aux_w = float(os.environ.get("PATHOGAIT_AUX_LOSS_W", "0.3"))
        if _aux_ch:
            for ch in [int(c) for c in _aux_ch.split(",") if c.strip() != ""]:
                ch_keep[ch] = _aux_w
        lm = loss_mask if loss_mask.dim() == 3 else loss_mask.unsqueeze(-1)
        loss_mask = lm * ch_keep.view(1, 40, 1)

    # CFG drop is handled INSIDE model.get_conditioning() per-condition with
    # cfg_drop_prob -- no manual drop required here.
    v_pred = model(xt, t, source_ids, cohort_id_t, severity_id_t)
    loss = nan_safe_mse_loss(v_pred, v_target, valid_mask=loss_mask)

    mse_loss_val = float(loss.detach().item())
    aux = {
        "v_target_std": float(v_target[torch.isfinite(v_target)].std().item()
                              if torch.isfinite(v_target).any() else 0.0),
        "v_pred_std":   float(v_pred.std().item()),
        "source_ids":   source_ids.cpu().tolist(),
        "cohort_ids":   cohort_ids,
        "severity_ids": severity_ids,
        "mse_loss":     mse_loss_val,
    }
    if input_missing_mask is not None:
        aux["input_mask_token_density"] = float(input_missing_mask.float().mean().item())
    return loss, aux


@torch.no_grad()
def _val_epoch(model, ema, sched, loader, device, sev_map,
               use_ema: bool = True, max_batches: int = 200, seq_len: int = 100,
               mask_bmclab_moments: bool = False,
               input_mask_token: Optional[torch.Tensor] = None,
               input_mask_token_ema: Optional[torch.Tensor] = None,
               input_mask_token_scope: str = "all_invalid",
               disable_cohort_channel_exclusion: bool = False):
    """Validation epoch (same forward configuration as training)."""
    if use_ema:
        ema.apply_to(model)
    model.eval()
    token_for_forward = (
        input_mask_token_ema
        if use_ema and input_mask_token_ema is not None
        else input_mask_token
    )
    losses = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        loss, aux = _step_forward(model, sched, b, device, sev_map, seq_len=seq_len,
                                 mask_bmclab_moments=mask_bmclab_moments,
                                 input_mask_token=token_for_forward,
                                 input_mask_token_scope=input_mask_token_scope,
                                 disable_cohort_channel_exclusion=disable_cohort_channel_exclusion)
        losses.append(float(aux.get("mse_loss", loss.item())))
    if use_ema:
        ema.restore(model)
    return float(np.mean(losses)) if losses else float("nan")


# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bmclab-pelvis-representation", choices=REPRESENTATIONS,
                   default=configured_representation(),
                   help="BMClab pelvis convention; must match input metadata and Stage-2 weights.")
    # Data
    p.add_argument("--split", default=str(ROOT / "data" / "splits" / "v4_stage2_split_grffix.json"))
    p.add_argument("--zstats",
                   default=str(ROOT / "data" / "zstats" / "global_zstats_stage1_only.json"))
    p.add_argument("--severity-label", required=True,
                   help="per_subject_severity.json path")
    p.add_argument("--fold", required=True,
                   help="cv5_{0..4}")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--epoch-pseudo-length", type=int, default=10_000,
                   help="Stage 2 default 10k -> ~17.6 effective pass over "
                        "~6.8k train cycles per fold w/ epochs=12")
    p.add_argument("--val-pseudo-length", type=int, default=1_500)

    # Model
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--in-channels", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=101)
    p.add_argument("--num-sources", type=int, default=12)
    p.add_argument("--num-cohorts", type=int, default=4)
    p.add_argument("--num-severities", type=int, default=6)
    p.add_argument("--cfg-drop-prob", type=float, default=0.10)

    # Path B: per-subject polarity-fix flags (default OFF for backward compat)
    p.add_argument("--use-per-subject-flip", action="store_true", default=True,
                   help="Use V4LazyTransformV3PerSubject with per-subject "
                        "flip lookup table (Path B). When unset (default), "
                        "behavior is identical to baseline Stage 2.")
    p.add_argument("--per-subject-flip-paper",
                   default=str(ROOT / "data" / "metadata"
                               / "per_subject_flip_table.json"),
                   help="Per-subject flip table for 5 paper cohorts (Phase 1).")
    p.add_argument("--per-subject-flip-addbio",
                   default=str(ROOT / "data" / "metadata"
                               / "per_subject_flip_table_addbio.json"),
                   help="Per-subject flip table for 7 AddBio studies (Phase 1).")
    p.add_argument("--ambiguous-policy", default="no_flip",
                   choices=["no_flip", "flip"],
                   help="How to treat audit value 0 (|PCC|<0.3 ambiguous).")

    # bmclab moment forward-pass NaN-mask ablation.
    # When True, zeros bmclab_pd subjects' moment-channel data (40-ch
    # indices in EXCLUDED_T1_CHANNELS_PER_COHORT) in x0 BEFORE the forward
    # pass. Prevents the model from attending to bmclab's documented-broken
    # moment values during conditioning. Loss is already masked via L4
    # valid_mask zeroing; this flag additionally cleans the forward signal.
    # Default False = current L4 behavior (loss-mask only, model still
    # sees bmclab moment values in xt).
    p.add_argument("--mask-bmclab-moments", action="store_true", default=True,
                   help="ablation: zero out the BMClab moment data "
                        "(40-ch [10..19]) in x0 BEFORE forward pass. "
                        "Default False = current L4 behavior (loss-mask only).")
    p.add_argument("--disable-cohort-channel-exclusion", action="store_true", default=False,
                   help="P5 BMClab-only mini-model: disable _EXCLUDED_40CH_PER_COHORT "
                        "logic entirely. BMClab moment channels become regular targets. "
                        "Use only when training on a single cohort whose channel layout "
                        "differs from the cross-cohort harmonized convention.")

    p.add_argument("--input-mask-token", action="store_true", default=True,
                   help="Token-mask ablation: replace unavailable input "
                        "channels with a learned per-channel mask token before "
                        "diffusion noising. Loss-side valid_mask is unchanged.")
    p.add_argument("--input-mask-token-scope", default="all_invalid",
                   choices=["all_invalid", "bmclab_moments"],
                   help="all_invalid: tokenise every channel with valid_mask=0 "
                        "after cohort exclusions. bmclab_moments: tokenise only "
                        "the explicit BMClab moment exclusion channels.")
    p.add_argument("--input-mask-token-init", type=float, default=0.0,
                   help="Initial scalar value for the learned input mask token.")
    # ccmask (2026-05-23): cohort-conditional ergonomic alias.
    # Equivalent to `--input-mask-token --input-mask-token-scope bmclab_moments`
    # (token applied iff cohort==bmclab_pd AND channel in [10..19]).
    # Default False = no override of explicit --input-mask-token-* flags.
    p.add_argument("--cohort-conditional-mask", action="store_true", default=False,
                   help="ccmask ablation: enable input mask token with scope "
                        "bmclab_moments (cohort_conditional). Equivalent to "
                        "--input-mask-token --input-mask-token-scope bmclab_moments. "
                        "Default False.")

    p.add_argument("--enable-cohort-id", action="store_true", default=True,
                   help="(Stage 2 default True; --no-enable-cohort-id to disable)")
    p.add_argument("--no-enable-cohort-id", dest="enable_cohort_id",
                   action="store_false")
    p.add_argument("--enable-severity-id", action="store_true", default=True)
    p.add_argument("--no-enable-severity-id", dest="enable_severity_id",
                   action="store_false")

    # Diffusion
    p.add_argument("--diffusion-timesteps", type=int, default=1000)
    p.add_argument("--diffusion-schedule", default="cosine")

    # Optim
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--constant-lr", action="store_true", default=True,
                   help="Stage 2 default: hold lr constant after warmup")
    p.add_argument("--no-constant-lr", dest="constant_lr", action="store_false")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=42)

    # Runtime
    p.add_argument("--output", default="checkpoints/v4_stage2_unified/")
    p.add_argument("--resume", required=False, default=None,
                   help="Stage 1 ckpt (final.pt or ckpt_epoch{N}.pt) for warm-start. "
                        "Required UNLESS --no-pretrain is set.")
    p.add_argument("--no-pretrain", action="store_true", default=False,
                   help="Skip Stage 1 backbone loading; use random initialization "
                        "(PyTorch default). For Methods M4 ablation.")
    p.add_argument("--ckpt-interval-min", type=int, default=30)
    p.add_argument("--ckpt-epochs-list", type=str, default="",
                   help="Comma-separated epoch indices that should be saved "
                        "(e.g. '25,50,75,99'). When non-empty, the per-epoch "
                        "save is restricted to these epochs AND periodic "
                        "step-milestone saves are disabled. When empty "
                        "(default), every epoch is saved (legacy ep12 behaviour).")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--val-interval-epoch", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True,
                   help="bf16 autocast on cuda")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--tb", action="store_true", default=True)
    p.add_argument("--no-tb", action="store_false", dest="tb")
    p.add_argument("--dry-run", action="store_true",
                   help="1-batch forward+backward, no ckpt written, exit")

    return p.parse_args()


# ----------------------------------------------------------------------
def _parse_ckpt_epochs_list(s: str):
    """Parse '25,50,75,99' -> {25, 50, 75, 99}; empty string -> None (= save every epoch)."""
    s = (s or "").strip()
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def main():
    args = get_args()
    zstats_path = bind_zstats_argument(args)
    capture_settings(args, ROOT)
    # ccmask alias resolution (2026-05-23): --cohort-conditional-mask is an
    # ergonomic alias that turns on the input mask token with scope
    # bmclab_moments (cohort_conditional). It does not silently override an
    # already-set --input-mask-token-scope; it only flips the defaults.
    if args.cohort_conditional_mask:
        args.input_mask_token = True
        args.input_mask_token_scope = "bmclab_moments"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    if not args.dry_run:
        log_path.write_text("", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        if not args.dry_run:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    log("V4 Stage 2 production trainer (DiT-B/1D fine-tune w/ cohort+severity)")
    log(f"  args: {json.dumps(vars(args), indent=2)}")

    # Ckpt-save policy: None -> save every epoch (legacy ep12 behaviour);
    # otherwise restrict per-epoch saves to listed indices and disable
    # periodic step-milestone saves. final.pt is unconditional in either case.
    ckpt_epochs_keep = _parse_ckpt_epochs_list(args.ckpt_epochs_list)
    if ckpt_epochs_keep is None:
        log("  ckpt policy: save every epoch + periodic step (legacy)")
    else:
        log(f"  ckpt policy: only save epochs in {sorted(ckpt_epochs_keep)}; "
            f"periodic step saves disabled; final.pt still written")

    if args.device == "cuda" and not torch.cuda.is_available():
        log("ERROR: cuda requested but unavailable.")
        sys.exit(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"  device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_split, val_split = _resolve_split_names(args.fold)
    log(f"  fold: {args.fold}  ->  {train_split} / {val_split}")


    # ---- Severity coverage check (fail fast before model build) ----
    sev_map = load_severity_labels(args.severity_label)
    train_pairs, val_pairs = _read_fold_pairs(args.split, args.fold)
    verify_severity_coverage(train_pairs + val_pairs, sev_map)
    log(f"  severity entries: {len(sev_map)}  "
        f"(train pairs={len(train_pairs)}, val pairs={len(val_pairs)})")
    cohort_counter = Counter(sev_map[p]["cohort_id"] for p in train_pairs)
    sev_counter = Counter(sev_map[p]["severity_id"] for p in train_pairs)
    log(f"  train cohort dist:    {dict(sorted(cohort_counter.items()))}")
    log(f"  train severity dist:  {dict(sorted(sev_counter.items()))}")

    # ---- Data ----
    train_ds, val_ds, train_loader, val_loader = _build_loaders(
        args, train_split, val_split)
    if not args.dry_run:
        save_training_manifest(args, train_ds, val_ds, out_dir)
    log(f"  train sources: {train_ds.sources}")
    log(f"  train files: {sum(len(v) for v in train_ds.source_files.values())}")
    log(f"  val files:   {sum(len(v) for v in val_ds.source_files.values())}")

    # ---- Model ----
    actual_seq_len = 100
    model = DiT_B_1D(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        seq_len=actual_seq_len,
        num_sources=args.num_sources,
        num_cohorts=args.num_cohorts,
        num_severities=args.num_severities,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)
    n_params = count_params(model)
    log(f"  DiT_B_1D params: {n_params/1e6:.2f}M  "
        f"(cohort_embed {args.num_cohorts}+1, severity_embed {args.num_severities}+1)")

    sched = DDPMScheduler(
        num_train_timesteps=args.diffusion_timesteps,
        schedule=args.diffusion_schedule).to(device)

    # bmclab moment forward-pass zero-mask ablation.
    if args.mask_bmclab_moments:
        action = ("will be token-filled in x0 before forward pass"
                  if args.input_mask_token
                  else "will be zero-filled in x0 before forward pass")
        log(f"[launch-B] mask_bmclab_moments=True, "
            f"EXCLUDED_40CH_BMCLAB={_EXCLUDED_40CH_PER_COHORT.get('bmclab_pd', [])} "
            f"(54-ch {EXCLUDED_T1_CHANNELS_PER_COHORT.get('bmclab_pd', [])}) "
            f"{action} for cohort bmclab_pd")

    # ---- Warm-start from Stage 1 ckpt (or --no-pretrain ablation) ----
    # Mutual exclusivity: one of --resume or --no-pretrain must be set.
    if not args.no_pretrain and args.resume is None:
        log("ERROR: must pass either --resume <stage1_ckpt> or --no-pretrain")
        sys.exit(2)

    def _model_param_l2_norm(m):
        with torch.no_grad():
            return float(
                torch.sqrt(sum((p.detach() ** 2).sum() for p in m.parameters()))
                .item()
            )

    pre_load_norm = _model_param_l2_norm(model)

    if args.no_pretrain:
        # ABLATION PATH: skip Stage 1 backbone entirely; keep DiT_B_1D default init.
        log("[ABLATION] --no-pretrain: skipping Stage 1 backbone load, using random init")
        log(f"  model param L2 norm (random init): {pre_load_norm:.4f}")
        # Stand-in for ckpt so downstream code that reads ckpt.get(...) stays valid.
        ckpt = {}
    else:
        if not Path(args.resume).exists():
            log(f"ERROR: --resume path does not exist: {args.resume}")
            sys.exit(2)
        log(f"  loading Stage 1 ckpt: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_args = ckpt.get("args", {})
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
        if "fold" in saved_args:
            verify_checkpoint_representation(
                ckpt, args.bmclab_pelvis_representation, "Stage-2 checkpoint")
        zstats_warning = verify_checkpoint_zstats(
            ckpt, zstats_path, args.zstats_sha256, "Stage 1 checkpoint"
        )
        if zstats_warning:
            log(f"  [WARN] {zstats_warning}")
        expected_keys = {"epoch", "global_step", "model", "ema", "optimizer",
                         "lr_scheduler", "args"}
        missing = expected_keys - set(ckpt.keys())
        if missing:
            log(f"  [WARN] Stage 1 ckpt missing keys: {missing}")
        model.load_state_dict(ckpt["model"])
        post_load_norm = _model_param_l2_norm(model)
        log(f"  loaded model state_dict ({len(ckpt['model'])} tensors) "
            f"from Stage 1 epoch={ckpt.get('epoch', '?')}, "
            f"global_step={ckpt.get('global_step', '?')}")
        log(f"  model param L2 norm (random init -> Stage 1): "
            f"{pre_load_norm:.4f} -> {post_load_norm:.4f}")

    # ---- Optional input mask token (kept outside model.state_dict for
    # checkpoint compatibility with existing evaluators).
    input_mask_token = None
    input_mask_token_ema = None
    if args.input_mask_token:
        input_mask_token = torch.nn.Parameter(
            torch.full((1, args.in_channels, 1),
                       float(args.input_mask_token_init),
                       device=device))
        input_mask_token_ema = input_mask_token.detach().clone()
        mode_tag = "ccmask" if args.cohort_conditional_mask else "tokenmask"
        log(f"[launch-{mode_tag}] input_mask_token=True, "
            f"scope={args.input_mask_token_scope}, "
            f"shape={tuple(input_mask_token.shape)}, "
            f"init={args.input_mask_token_init}")
        if args.input_mask_token_scope == "all_invalid":
            log(f"[launch-{mode_tag}] token will replace all valid_mask=0 input "
                "channels after BMClab moment exclusion; loss mask unchanged")
        elif args.input_mask_token_scope == "bmclab_moments":
            log("[ccmask] active: mask token applied to cohort=bmclab_pd, "
                "ch=[10..19] only; other cohorts and channels unchanged")

    trainable_params = list(model.parameters())
    if input_mask_token is not None:
        trainable_params.append(input_mask_token)

    def _update_input_mask_token_ema():
        if input_mask_token is not None and input_mask_token_ema is not None:
            input_mask_token_ema.mul_(args.ema_decay).add_(
                input_mask_token.detach(), alpha=1.0 - args.ema_decay)

    def _attach_input_mask_token_state(payload):
        if input_mask_token is not None and input_mask_token_ema is not None:
            payload["input_mask_token"] = input_mask_token.detach().cpu()
            payload["input_mask_token_ema"] = input_mask_token_ema.detach().cpu()
        return payload

    # ---- Optim + LR schedule ----
    opt = torch.optim.AdamW(trainable_params, lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    if args.max_steps:
        total_steps = args.max_steps
    else:
        per_epoch = max(1, args.epoch_pseudo_length // args.batch_size)
        total_steps = per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, max(0, total_steps // 4))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _make_lr_lambda(warmup_steps, total_steps, args.constant_lr))
    log(f"  total_steps={total_steps}, warmup={warmup_steps}, "
        f"constant_after_warmup={args.constant_lr}")

    # ---- EMA: init shadow from Stage 1 EMA weights (or current model under ablation) ----
    ema = EMA(model, decay=args.ema_decay)
    if args.no_pretrain:
        log("[ABLATION] EMA initialised from random-init model weights (no Stage 1 EMA)")
    elif "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        # ckpt was loaded with map_location='cpu'; move shadow to the same
        # device as model params so EMA.update()'s in-place mul_/add_ don't
        # hit a cross-device mismatch on the first train step.
        ema.shadow = {k: v.to(device) for k, v in ema.shadow.items()}
        log(f"  loaded EMA shadow ({len(ckpt['ema'])} tensors) from Stage 1 "
            f"-> moved to {device}")
    else:
        log("  [WARN] Stage 1 ckpt missing 'ema'; EMA initialised from model weights")

    if args.dry_run:
        log("\n=== DRY RUN: 1 batch forward+backward, no ckpt ===")
        model.train()
        b = next(iter(train_loader))
        loss, aux = _step_forward(model, sched, b, device, sev_map,
                                  seq_len=actual_seq_len,
                                  mask_bmclab_moments=args.mask_bmclab_moments,
                                  input_mask_token=input_mask_token,
                                  input_mask_token_scope=args.input_mask_token_scope,
                                  disable_cohort_channel_exclusion=args.disable_cohort_channel_exclusion)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        opt.step()
        _update_input_mask_token_ema()
        ema.update(model)
        log(f"  loss={loss.item():.4f}  grad_norm={float(gn):.3f}  "
            f"v_pred_std={aux['v_pred_std']:.3f}")
        if "input_mask_token_density" in aux:
            log(f"  input_mask_token_density={aux['input_mask_token_density']:.3f}")
        log(f"  source_ids:   {aux['source_ids']}")
        log(f"  cohort_ids:   {aux['cohort_ids']}")
        log(f"  severity_ids: {aux['severity_ids']}")
        log("DRY RUN PASS")
        return

    # ---- TB ----
    writer = None
    if args.tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        except Exception as e:
            log(f"  [WARN] tensorboard unavailable: {e}")

    # ---- Save args.json snapshot ----
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Train loop ----
    global_step = 0
    last_ckpt_time = time.time()
    losses_window = []
    source_counter, cohort_counter_tr, sev_counter_tr = (
        Counter(), Counter(), Counter())

    log("\n=== Training start ===")
    t_train_start = time.time()
    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (args.amp and device.type == "cuda")
        else (lambda: torch.enable_grad()))

    for epoch in range(args.epochs):
        model.train()
        ep_losses, ep_vt_std, ep_vp_std = [], [], []
        t_ep = time.time()
        ep_done = False

        for batch_idx, batch in enumerate(train_loader):
            with autocast_ctx():
                loss, aux = _step_forward(model, sched, batch, device,
                                          sev_map, seq_len=actual_seq_len,
                                          mask_bmclab_moments=args.mask_bmclab_moments,
                                          input_mask_token=input_mask_token,
                                          input_mask_token_scope=args.input_mask_token_scope,
                                          global_step=global_step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, args.grad_clip)
            opt.step()
            lr_sched.step()
            _update_input_mask_token_ema()
            ema.update(model)

            for sid in aux["source_ids"]: source_counter[sid] += 1
            for cid in aux["cohort_ids"]: cohort_counter_tr[cid] += 1
            for vid in aux["severity_ids"]: sev_counter_tr[vid] += 1
            ep_losses.append(loss.item())
            ep_vt_std.append(aux["v_target_std"])
            ep_vp_std.append(aux["v_pred_std"])
            losses_window.append(loss.item())
            if len(losses_window) > 50: losses_window.pop(0)
            global_step += 1

            if global_step % args.log_interval == 0:
                cur_lr = opt.param_groups[0]["lr"]
                rec = float(np.mean(losses_window))
                token_suffix = ""
                if "input_mask_token_density" in aux:
                    token_suffix = f"  token_density={aux['input_mask_token_density']:.3f}"
                log(f"  step {global_step:>6}  ep {epoch}  "
                    f"loss={loss.item():.4f}  recent50={rec:.4f}  "
                    f"grad={grad_norm:.3f}  lr={cur_lr:.2e}  "
                    f"v_pred_std={aux['v_pred_std']:.3f}{token_suffix}")
                if writer is not None:
                    writer.add_scalar("train/loss",       loss.item(), global_step)
                    writer.add_scalar("train/recent50",   rec,         global_step)
                    writer.add_scalar("train/grad_norm",  float(grad_norm), global_step)
                    writer.add_scalar("train/lr",         cur_lr,      global_step)
                    writer.add_scalar("train/v_pred_std", aux["v_pred_std"],
                                       global_step)
                    if "input_mask_token_density" in aux:
                        writer.add_scalar("train/input_mask_token_density",
                                          aux["input_mask_token_density"], global_step)

            # Periodic step-milestone save: disabled when --ckpt-epochs-list
            # is in effect (long ep100 ablation, disk-budget sensitive).
            if (ckpt_epochs_keep is None
                    and (time.time() - last_ckpt_time) / 60 >= args.ckpt_interval_min):
                p = out_dir / f"ckpt_step{global_step}.pt"
                _safe_save(_attach_input_mask_token_state({
                    "epoch": epoch, "global_step": global_step,
                    "fold": args.fold,
                    "model": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(),
                    "args": vars(args),
                }), p)
                log(f"  [ckpt] periodic save: {p.name}")
                last_ckpt_time = time.time()

            if args.max_steps and global_step >= args.max_steps:
                # Save a checkpoint at the smoke-test boundary so downstream
                # smoke-stage tools (Gate G ablation eval) can load the online
                # weights at exactly this step.
                p = out_dir / f"ckpt_step{global_step}.pt"
                _safe_save(_attach_input_mask_token_state({
                    "epoch": epoch, "global_step": global_step,
                    "fold": args.fold,
                    "model": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(),
                    "args": vars(args),
                }), p)
                log(f"  [ckpt] max-steps save: {p.name}")
                ep_done = True
                break

        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        ep_time = time.time() - t_ep
        log(f"\n=== epoch {epoch} ===")
        log(f"  train loss mean : {ep_loss_mean:.4f}")
        log(f"  v_target/v_pred std: {np.mean(ep_vt_std):.3f} / "
            f"{np.mean(ep_vp_std):.3f}")
        log(f"  time            : {ep_time/60:.1f} min")
        if writer is not None:
            writer.add_scalar("train/epoch_loss", ep_loss_mean, epoch)
            writer.add_scalar("train/epoch_time_min", ep_time/60, epoch)

        if epoch % args.val_interval_epoch == 0:
            vl_on = _val_epoch(model, ema, sched, val_loader, device, sev_map,
                                use_ema=False, seq_len=actual_seq_len,
                                mask_bmclab_moments=args.mask_bmclab_moments,
                                input_mask_token=input_mask_token,
                                input_mask_token_ema=input_mask_token_ema,
                                input_mask_token_scope=args.input_mask_token_scope,
                                disable_cohort_channel_exclusion=args.disable_cohort_channel_exclusion)
            vl_em = _val_epoch(model, ema, sched, val_loader, device, sev_map,
                                use_ema=True, seq_len=actual_seq_len,
                                mask_bmclab_moments=args.mask_bmclab_moments,
                                input_mask_token=input_mask_token,
                                input_mask_token_ema=input_mask_token_ema,
                                input_mask_token_scope=args.input_mask_token_scope,
                                disable_cohort_channel_exclusion=args.disable_cohort_channel_exclusion)
            drift = ema.drift_norm(model)
            log(f"  val loss (online): {vl_on:.4f}")
            log(f"  val loss (EMA)   : {vl_em:.4f}")
            log(f"  EMA drift norm   : {drift:.3f}")
            if writer is not None:
                writer.add_scalar("val/loss_online", vl_on, epoch)
                writer.add_scalar("val/loss_ema",    vl_em, epoch)
                writer.add_scalar("val/ema_drift",   drift, epoch)

        # Per-epoch save: legacy (list=None) = always save; list mode = only
        # save when this epoch is explicitly in --ckpt-epochs-list. final.pt
        # below is unconditional regardless.
        if ckpt_epochs_keep is None or epoch in ckpt_epochs_keep:
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
            _safe_save(_attach_input_mask_token_state({
                "epoch": epoch, "global_step": global_step,
                "fold": args.fold,
                "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "lr_scheduler": lr_sched.state_dict(),
                "args": vars(args),
            }), ckpt_path)
            log(f"  [ckpt] epoch end save: {ckpt_path.name}")
        else:
            log(f"  [ckpt] epoch {epoch} skipped (not in --ckpt-epochs-list)")

        if ep_done:
            break

    total_time = time.time() - t_train_start
    log("\n=== Training complete ===")
    log(f"  total_steps   : {global_step}")
    log(f"  wall_time     : {total_time/3600:.2f} hr")
    log(f"  source counts : {dict(source_counter)}")
    log(f"  cohort counts : {dict(cohort_counter_tr)}")
    log(f"  severity counts: {dict(sev_counter_tr)}")
    final_path = out_dir / "final.pt"
    _safe_save(_attach_input_mask_token_state({
        "global_step": global_step, "fold": args.fold,
        "model": model.state_dict(), "ema": ema.state_dict(),
        "args": vars(args),
        "source_counts":   dict(source_counter),
        "cohort_counts":   dict(cohort_counter_tr),
        "severity_counts": dict(sev_counter_tr),
    }), final_path)
    log(f"  [ckpt] final save: {final_path.name}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
