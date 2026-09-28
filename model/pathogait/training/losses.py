"""NaN-safe MSE loss for V4 mixed-NaN training data.

AddBio EMG channels (40-set indices 26-33) are NaN because AddBio has no
EMG. Stage 1 training mixes AddBio + vdk_healthy in the same batch, so
the loss must mask NaN cells gracefully — otherwise loss = NaN once any
AddBio sample lands in the batch.
"""
from __future__ import annotations
import torch


def nan_safe_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """NaN-safe MSE.

    Args:
      pred: any shape, no NaN expected.
      target: same shape as `pred`, NaN cells are masked out.
      valid_mask: optional bool/float tensor broadcastable to `target`.
        If shape (B, C) and `target` is (B, C, T), it broadcasts to (B, C, 1).
        1 = include, 0 = exclude.
      reduction: 'mean' (default) — mean over surviving cells.
                 'sum'  — sum of squared errors over surviving cells.

    Returns:
      scalar tensor (gradient-safe). Returns 0 if no cells survive.
    """
    not_nan = ~torch.isnan(target)                     # bool mask
    target_safe = torch.nan_to_num(target, nan=0.0)
    sq = (pred - target_safe) ** 2                     # element-wise sq err

    if valid_mask is not None:
        if valid_mask.dim() < sq.dim():
            # broadcast e.g. (B, C) -> (B, C, 1) for sq of shape (B, C, T)
            view_shape = list(valid_mask.shape) + [1] * (sq.dim() - valid_mask.dim())
            valid_mask = valid_mask.view(*view_shape)
        keep = not_nan & valid_mask.bool()
    else:
        keep = not_nan

    keep_f = keep.float()
    sq = sq * keep_f

    if reduction == "sum":
        return sq.sum()
    # mean over kept cells (clamp denom to 1 to avoid /0)
    n_keep = keep_f.sum().clamp(min=1.0)
    return sq.sum() / n_keep
