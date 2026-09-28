"""Exponential moving-average (EMA) shadow-weight tracker used by the trainers."""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# EMA wrapper - NOT an nn.Module (won't appear in state_dict / backward)
# ----------------------------------------------------------------------
class EMA:
    """Shadow-weight tracker. Call `update(model)` after every optimizer step.
    `apply_to(model)` swaps shadow into the model in place; `restore(model)`
    puts the originals back. Floating tensors are EMA'd; integer / bool
    buffers are copied verbatim.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        """Swap shadow into model. Stores originals for restore()."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        if self._backup is None: return
        model.load_state_dict(self._backup, strict=True)
        self._backup = None

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow: self.shadow[k] = v.clone()

    @torch.no_grad()
    def drift_norm(self, model: nn.Module) -> float:
        """Sum of L2(model_param - shadow) over float params (logging)."""
        total = 0.0
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if v.dtype.is_floating_point:
                total += (sd[k].detach() - v).pow(2).sum().item()
        return float(total ** 0.5)
