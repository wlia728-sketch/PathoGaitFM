"""Raw-space DDPM scheduler for V4 Stage 1.

Cosine schedule (Nichol & Dhariwal 2021) + v-prediction parameterisation
(Salimans & Ho 2022):

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    v_t = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0
    x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v_t

The scheduler stores pre-computed (alphas, betas, alpha_bars, sqrt_*)
as torch tensors. Move to device with `.to(device)`. Inference (the frozen
in-painting used throughout this work) drives the DDIM step from these
buffers directly; see the evaluators' `tweedie_inpaint_drop`.
"""
from __future__ import annotations
import math

import torch


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "cosine",
        s: float = 0.008,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule = schedule
        self.s = float(s)

        if schedule == "cosine":
            steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64)
            ab_full = torch.cos(((steps / num_train_timesteps) + s)
                                  / (1 + s) * math.pi / 2) ** 2
            ab_full = ab_full / ab_full[0]            # alpha_bar_0 = 1
            betas = 1 - (ab_full[1:] / ab_full[:-1])
            betas = betas.clamp(0.0, 0.999)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02,
                                     num_train_timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # alpha_bar at t = -1 (used in DDIM step from t=0 -> x_0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])

        # Store as float32 tensors on CPU; user moves to device.
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bars = alpha_bars.float()
        self.alpha_bars_prev = alpha_bars_prev.float()
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - self.alpha_bars).sqrt()

    def to(self, device):
        for name in ("betas", "alphas", "alpha_bars", "alpha_bars_prev",
                     "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ---------- gather helper ----------
    @staticmethod
    def _gather(buf: torch.Tensor, t: torch.Tensor,
                 n_dim: int) -> torch.Tensor:
        """buf is (T,); t is (B,) long. Return (B, 1, 1, ...) broadcastable."""
        out = buf.to(t.device).gather(0, t)
        return out.view(-1, *[1] * (n_dim - 1))

    # ---------- forward diffusion ----------
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(ab) * x_0 + sqrt(1-ab) * noise."""
        sa  = self._gather(self.sqrt_alpha_bars,            t, x_0.dim())
        so  = self._gather(self.sqrt_one_minus_alpha_bars,  t, x_0.dim())
        return sa * x_0 + so * noise

    def get_velocity(self, x_0: torch.Tensor, noise: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        """v_t = sqrt(ab) * noise - sqrt(1-ab) * x_0."""
        sa  = self._gather(self.sqrt_alpha_bars,            t, x_0.dim())
        so  = self._gather(self.sqrt_one_minus_alpha_bars,  t, x_0.dim())
        return sa * noise - so * x_0

    def predict_x0_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """x_0 = sqrt(ab) * x_t - sqrt(1-ab) * v_pred."""
        sa  = self._gather(self.sqrt_alpha_bars,            t, x_t.dim())
        so  = self._gather(self.sqrt_one_minus_alpha_bars,  t, x_t.dim())
        return sa * x_t - so * v_pred

    def predict_noise_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """epsilon = sqrt(1-ab) * x_t + sqrt(ab) * v_pred."""
        sa  = self._gather(self.sqrt_alpha_bars,            t, x_t.dim())
        so  = self._gather(self.sqrt_one_minus_alpha_bars,  t, x_t.dim())
        return so * x_t + sa * v_pred


# -----------------------------------------------------------------------
if __name__ == "__main__":
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine")
    print("betas[0:3]   :", sched.betas[:3].tolist())
    print("betas[-3:]   :", sched.betas[-3:].tolist())
    print("alpha_bars[0]:", sched.alpha_bars[0].item())
    print("alpha_bars[T-1]:", sched.alpha_bars[-1].item())

    x_0 = torch.randn(4, 40, 100)
    noise = torch.randn(4, 40, 100)
    t = torch.tensor([100, 500, 800, 999])

    x_t = sched.add_noise(x_0, noise, t)
    v = sched.get_velocity(x_0, noise, t)
    x_0_rec = sched.predict_x0_from_v(x_t, v, t)
    noise_rec = sched.predict_noise_from_v(x_t, v, t)

    print(f"x_t shape {tuple(x_t.shape)}  v shape {tuple(v.shape)}")
    print(f"x_0 reconstruction err : {(x_0 - x_0_rec).abs().max().item():.2e}")
    print(f"noise reconstruction err: {(noise - noise_rec).abs().max().item():.2e}")
    assert torch.isfinite(x_t).all() and torch.isfinite(v).all()
    print("DDPMScheduler smoke OK")
