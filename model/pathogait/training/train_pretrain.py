"""V4 Stage 1 production trainer â€” DiT-B/1D + raw 40-ch DDPM v-prediction.

Pretrains DiT_B_1D (~33.5M params) on the 8-source Stage-1 split (442
train files / 57 val / 54 test; 227k train cycles) using:
  - V4BalancedDatasetV3 (raw cohort + AddBio + V3 unified BW normalize)
  - DDPMScheduler cosine T=1000 + v-prediction loss
  - AdamW + linear warmup -> cosine decay LR
  - EMA shadow weights
  - Independent CFG drop (p=0.10) on each conditioning channel (handled
    inside the model)
  - nan_safe_mse_loss (AddBio EMG NaN, partial-cohort Pelvis_Z NaN masked)
  - Periodic + end-of-epoch checkpoints (atomic write)
  - TensorBoard scalars
  - Optional AMP bfloat16 (cuda only)
  - Resume from an epoch checkpoint (ckpt_epoch<N>.pt); final.pt holds weights only

CLI (smoke test in 200 steps):
    python model/pathogait/training/train_pretrain.py \
        --max-steps 200 --batch-size 64 --num-workers 4 \
        --output checkpoints/v4_stage1_prod_smoke/ --device cuda

The reported stage-1 run is the same command without --max-steps, which takes about 25 hours
on one GPU.

Source IDs are derived from the source name via V4_SOURCE_TO_ID below;
must match the embedding table layout in DiT_B_1D (12 entries + 1 null).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]          # training -> pathogait -> model -> the package root
for _p in (ROOT, ROOT / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pathogait.data.dataset import V4BalancedDatasetV3, collate
from pathogait.data.run_provenance import capture_settings, save_training_manifest
from pathogait.data.zstats_provenance import (
    bind_zstats_argument,
    verify_checkpoint_zstats,
)
from pathogait.diffusion.ddpm import DDPMScheduler
from pathogait.models.dit1d import DiT_B_1D, count_params
from pathogait.models.diffusion import EMA
from pathogait.training.losses import nan_safe_mse_loss


# Source name -> integer id (must align with DiT_B_1D embedding layout).
# 12 sources total: 7 AddBio + 5 cohort.
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


# ----------------------------------------------------------------------
def _safe_save(payload, target: Path, retries: int = 3):
    """Atomic checkpoint write with retry (OneDrive lock workaround)."""
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


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup -> cosine decay to 0."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


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
    )
    return tfm


def _build_loaders(args):
    transform_override = _maybe_build_per_subject_transform(args)
    if transform_override is not None:
        n_sub = transform_override.n_subjects_in_table()
        print(f"[Stage1] using V4LazyTransformV3PerSubject  "
              f"(paper={n_sub['paper']}, addbio={n_sub['addbio']}, "
              f"total={n_sub['total']})  "
              f"ambiguous_policy={args.ambiguous_policy}")

    train_ds = V4BalancedDatasetV3.from_split(
        args.split, "train",
        global_zstats_path=args.zstats,
        pseudo_length=args.epoch_pseudo_length,
        seed=args.seed,
        cache_size=args.cache_size,
        transform_override=transform_override,
        sampling="source_uniform",
    )
    val_ds = V4BalancedDatasetV3.from_split(
        args.split, "val",
        global_zstats_path=args.zstats,
        pseudo_length=args.val_pseudo_length,
        seed=args.seed + 1,
        cache_size=args.cache_size // 2,
        transform_override=transform_override,
        sampling="source_uniform",
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
def _step_forward(model, sched, batch, device, num_sources, seq_len=100):
    """Single training / validation step compute (no backward).

    Slices time dim 101 -> seq_len (default 100) so it is divisible by the
    DiT patch size 4 (100 = 4 * 25). Cycles are re-indexed 0..100 with
    frame 100 ~= frame 0 by V3 transform's interpolation, so the last
    frame is informationally redundant.

    Returns (loss, grad-free aux dict for logging).
    """
    x0 = batch["x"].to(device, non_blocking=True)              # (B, 40, 101)
    x0 = x0[..., :seq_len]                                       # -> (B, 40, 100)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)  # (B, 40)
    source_names = batch["source_name"]
    source_ids = torch.tensor(
        [V4_SOURCE_TO_ID[s] for s in source_names],
        device=device, dtype=torch.long)

    B = x0.shape[0]
    t = torch.randint(0, sched.num_train_timesteps, (B,), device=device)
    noise = torch.randn_like(x0)
    # Replace NaN cells with 0 in the noised input (loss is still masked)
    x0_safe = torch.nan_to_num(x0, nan=0.0)
    xt = sched.add_noise(x0_safe, noise, t)
    v_target = sched.get_velocity(x0_safe, noise, t)
    # Re-introduce NaN in target so nan_safe_mse_loss masks those cells out
    v_target = torch.where(torch.isnan(x0), x0, v_target)

    v_pred = model(xt, t, source_ids)
    loss = nan_safe_mse_loss(v_pred, v_target, valid_mask=valid_mask)
    return loss, {
        "v_target_std": float(v_target[torch.isfinite(v_target)].std().item()
                                if torch.isfinite(v_target).any() else 0.0),
        "v_pred_std":   float(v_pred.std().item()),
        "source_ids":   source_ids.cpu().tolist(),
    }


# ----------------------------------------------------------------------
@torch.no_grad()
def _val_epoch(model, ema, sched, loader, device, num_sources,
                use_ema: bool = True, max_batches: int = 200,
                seq_len: int = 100):
    """Compute val loss (optionally with EMA weights swapped in)."""
    if use_ema:
        ema.apply_to(model)
    model.eval()
    losses = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        loss, _ = _step_forward(model, sched, b, device, num_sources,
                                  seq_len=seq_len)
        losses.append(loss.item())
    if use_ema:
        ema.restore(model)
    return float(np.mean(losses)) if losses else float("nan")


# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--split", default=str(ROOT / "data" / "splits" / "v4_stage1_split.json"))
    p.add_argument("--zstats",
                   default=str(ROOT / "data" / "zstats" / "global_zstats_stage1_only.json"))

    # Path B: per-subject polarity-fix flags (default OFF for backward compat)
    p.add_argument("--use-per-subject-flip", action="store_true", default=True,
                   help="Use V4LazyTransformV3PerSubject with per-subject "
                        "flip lookup table (Path B). When unset (default), "
                        "behavior is identical to baseline Stage 1.")
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
                   help="How to treat audit value 0 (|PCC|<0.3 ambiguous). "
                        "'no_flip'=conservative (treat as 1); 'flip'=aggressive.")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--epoch-pseudo-length", type=int, default=300_000,
                   help="cycles drawn per epoch; 300k is the reported Stage-1 setting")
    p.add_argument("--val-pseudo-length", type=int, default=5_000)

    # Model
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--in-channels", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=101,
                   help="cycle timesteps; will pad to multiple of patch")
    p.add_argument("--num-sources", type=int, default=12)
    p.add_argument("--cfg-drop-prob", type=float, default=0.10)

    # Diffusion
    p.add_argument("--diffusion-timesteps", type=int, default=1000)
    p.add_argument("--diffusion-schedule", default="cosine")

    # Optim
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=42)

    # Runtime
    p.add_argument("--output", default="checkpoints/v4_stage1_unified/")
    p.add_argument("--ckpt-interval-min", type=int, default=30)
    p.add_argument("--ckpt-every-n-epochs", type=int, default=1,
                   help="Save epoch-end checkpoint every N epochs (default 1 = "
                        "every epoch). Set higher to save disk during long runs. "
                        "The final epoch is always saved regardless.")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--val-interval-epoch", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None)
    p.add_argument("--amp", action="store_true", default=True,
                   help="use bfloat16 autocast on cuda")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total steps (for smoke)")
    p.add_argument("--tb", action="store_true", default=True,
                   help="enable tensorboard scalars")
    p.add_argument("--no-tb", action="store_false", dest="tb")

    return p.parse_args()


# ----------------------------------------------------------------------
def main():
    args = get_args()
    zstats_path = bind_zstats_argument(args)
    capture_settings(args, ROOT)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train.log"
    log_path.write_text("", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("V4 Stage 1 production trainer (DiT-B/1D + raw DDPM)")
    log(f"  args: {json.dumps(vars(args), indent=2)}")

    # ---- Device ----
    if args.device == "cuda" and not torch.cuda.is_available():
        log("ERROR: cuda requested but unavailable. Activate probe_venv.")
        sys.exit(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"  device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data ----
    train_ds, val_ds, train_loader, val_loader = _build_loaders(args)
    save_training_manifest(args, train_ds, val_ds, out_dir)
    log(f"  train sources: {train_ds.sources}")
    log(f"  train files: {sum(len(v) for v in train_ds.source_files.values())}")
    log(f"  val files:   {sum(len(v) for v in val_ds.source_files.values())}")

    # Path B diagnostic: print first 5 sample subjects' per-subject flip lists
    if getattr(args, "use_per_subject_flip", False):
        log("  Path B per-subject flip diagnostic (first 5 train subjects):")
        from pathlib import Path as _P
        for src, files in list(train_ds.source_files.items())[:3]:
            src_type = "addbio" if src.startswith("addbio_") else "cohort"
            for fp in files[:2]:
                sid = _P(fp).stem
                fl = train_ds.transform._per_subject_flip_list(sid, src_type)
                log(f"    {src:<22} {sid:<32} flip_list={fl}")

    # ---- Model + Scheduler ----
    # seq_len 101 -> pad to 100 by drop the last timestep (cycles are already
    # interpolated to 101 frames; the last frame is the same as the first
    # in a closed gait cycle). Better: cycle-aware pad to 104 = 4*26 patches.
    # Choice: use seq_len=100 by slicing the time dim, which keeps patch-size
    # 4 with 25 patches.
    actual_seq_len = 100   # slice from 101 -> 100 in the train loop
    model = DiT_B_1D(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        seq_len=actual_seq_len,
        num_sources=args.num_sources,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)
    n_params = count_params(model)
    log(f"  DiT_B_1D params: {n_params/1e6:.2f}M")

    sched = DDPMScheduler(
        num_train_timesteps=args.diffusion_timesteps,
        schedule=args.diffusion_schedule).to(device)

    # ---- Optim ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    if args.max_steps:
        total_steps = args.max_steps
    else:
        # Pseudo-length DataLoader; total_steps ~= epoch_pseudo_length/batch * epochs
        per_epoch = max(1, args.epoch_pseudo_length // args.batch_size)
        total_steps = per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _make_lr_lambda(warmup_steps, total_steps))
    ema = EMA(model, decay=args.ema_decay)
    log(f"  total_steps={total_steps}, warmup={warmup_steps}, "
        f"per_epoch~={per_epoch if not args.max_steps else 'n/a'}")

    # ---- Resume ----
    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" not in ckpt or "epoch" not in ckpt:
            raise ValueError(
                f"{args.resume} holds weights only: final.pt stores the model, EMA and arguments but no "
                "optimizer, scheduler or epoch state. Resume from an epoch checkpoint (ckpt_epoch<N>.pt).")
        zstats_warning = verify_checkpoint_zstats(
            ckpt, zstats_path, args.zstats_sha256, "resume checkpoint"
        )
        if zstats_warning:
            log(f"  [WARN] {zstats_warning}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        if "lr_scheduler" in ckpt:
            lr_sched.load_state_dict(ckpt["lr_scheduler"])
        start_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"  resumed from {args.resume}: step={start_step}, "
            f"epoch={start_epoch}")

    # ---- TB ----
    writer = None
    if args.tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        except Exception as e:
            log(f"  [WARN] tensorboard unavailable: {e}")

    # ---- Train loop ----
    global_step = start_step
    last_ckpt_time = time.time()
    losses_window = []
    source_counter = Counter()

    log("\n=== Training start ===")
    # Log the epoch-save plan for visibility.
    saved_eps = [e for e in range(start_epoch, args.epochs)
                 if ((e + 1) % args.ckpt_every_n_epochs == 0)
                 or e == args.epochs - 1]
    log(f"  ckpt-every-n-epochs={args.ckpt_every_n_epochs}; "
        f"saving epochs: {saved_eps[:8]}"
        f"{'...' if len(saved_eps) > 8 else ''} "
        f"(total {len(saved_eps)} ckpts)")
    t_train_start = time.time()
    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (args.amp and device.type == "cuda")
        else (lambda: torch.enable_grad()))   # null context (still grad-on)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_losses, ep_v_target_std, ep_v_pred_std = [], [], []
        t_ep = time.time()
        ep_done = False

        for batch_idx, batch in enumerate(train_loader):
            with autocast_ctx():
                loss, aux = _step_forward(model, sched, batch, device,
                                            args.num_sources,
                                            seq_len=actual_seq_len)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)
            opt.step()
            lr_sched.step()
            ema.update(model)

            for sid in aux["source_ids"]:
                source_counter[sid] += 1
            ep_losses.append(loss.item())
            ep_v_target_std.append(aux["v_target_std"])
            ep_v_pred_std.append(aux["v_pred_std"])
            losses_window.append(loss.item())
            if len(losses_window) > 50:
                losses_window.pop(0)

            global_step += 1

            if global_step % args.log_interval == 0:
                current_lr = opt.param_groups[0]["lr"]
                recent = float(np.mean(losses_window))
                log(f"  step {global_step:>6}  ep {epoch}  "
                    f"loss={loss.item():.4f}  recent50={recent:.4f}  "
                    f"grad={grad_norm:.3f}  lr={current_lr:.2e}  "
                    f"v_pred_std={aux['v_pred_std']:.3f}")
                if writer is not None:
                    writer.add_scalar("train/loss",        loss.item(), global_step)
                    writer.add_scalar("train/recent50",    recent,      global_step)
                    writer.add_scalar("train/grad_norm",   float(grad_norm),
                                       global_step)
                    writer.add_scalar("train/lr",          current_lr,  global_step)
                    writer.add_scalar("train/v_pred_std",  aux["v_pred_std"],
                                       global_step)

            # Periodic checkpoint
            if (time.time() - last_ckpt_time) / 60 >= args.ckpt_interval_min:
                ckpt_path = out_dir / f"ckpt_step{global_step}.pt"
                _safe_save({
                    "epoch": epoch, "global_step": global_step,
                    "model": model.state_dict(),
                    "ema":   ema.state_dict(),
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(),
                    "args":  vars(args),
                }, ckpt_path)
                log(f"  [ckpt] periodic save: {ckpt_path.name}")
                last_ckpt_time = time.time()

            if args.max_steps and global_step >= args.max_steps:
                ep_done = True
                break

        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
        ep_time = time.time() - t_ep
        log(f"\n=== epoch {epoch} ===")
        log(f"  train loss mean : {ep_loss_mean:.4f}")
        log(f"  v_target/v_pred std: {np.mean(ep_v_target_std):.3f} / "
            f"{np.mean(ep_v_pred_std):.3f}")
        log(f"  time            : {ep_time/60:.1f} min")
        if writer is not None:
            writer.add_scalar("train/epoch_loss", ep_loss_mean, epoch)
            writer.add_scalar("train/epoch_time_min", ep_time/60, epoch)

        # Validation
        if (epoch - start_epoch) % args.val_interval_epoch == 0:
            val_loss_online = _val_epoch(model, ema, sched, val_loader,
                                           device, args.num_sources,
                                           use_ema=False,
                                           seq_len=actual_seq_len)
            val_loss_ema = _val_epoch(model, ema, sched, val_loader,
                                       device, args.num_sources,
                                       use_ema=True,
                                       seq_len=actual_seq_len)
            ema_drift = ema.drift_norm(model)
            log(f"  val loss (online): {val_loss_online:.4f}")
            log(f"  val loss (EMA)   : {val_loss_ema:.4f}")
            log(f"  EMA drift norm   : {ema_drift:.3f}")
            if writer is not None:
                writer.add_scalar("val/loss_online", val_loss_online, epoch)
                writer.add_scalar("val/loss_ema",    val_loss_ema,    epoch)
                writer.add_scalar("val/ema_drift",   ema_drift,       epoch)

        # End-of-epoch checkpoint â€” gated by --ckpt-every-n-epochs.
        # Final epoch (epoch == args.epochs - 1) is always saved regardless.
        is_final_epoch = (epoch == args.epochs - 1)
        is_scheduled = ((epoch + 1) % args.ckpt_every_n_epochs == 0)
        if is_scheduled or is_final_epoch:
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
            _safe_save({
                "epoch": epoch, "global_step": global_step,
                "model": model.state_dict(),
                "ema":   ema.state_dict(),
                "optimizer": opt.state_dict(),
                "lr_scheduler": lr_sched.state_dict(),
                "args":  vars(args),
            }, ckpt_path)
            log(f"  [ckpt] epoch end save: {ckpt_path.name}")
        else:
            log(f"  [ckpt] epoch {epoch} skipped (--ckpt-every-n-epochs={args.ckpt_every_n_epochs})")

        if ep_done:
            break

    # ---- Final ckpt + source histogram ----
    train_total_time = time.time() - t_train_start
    log("\n=== Training complete ===")
    log(f"  total_steps   : {global_step}")
    log(f"  wall_time     : {train_total_time/3600:.2f} hr")
    log(f"  source counts : {dict(source_counter)}")
    final_path = out_dir / "final.pt"
    _safe_save({
        "global_step": global_step,
        "model": model.state_dict(),
        "ema":   ema.state_dict(),
        "args":  vars(args),
        "source_counts": dict(source_counter),
    }, final_path)
    log(f"  [ckpt] final save: {final_path.name}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
