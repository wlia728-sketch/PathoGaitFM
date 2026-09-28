"""DiT-B/1D — Diffusion Transformer adapted to 1D gait cycles.

V4 Stage 1 production noise predictor for raw 40-channel x 100-timestep
gait cycle diffusion. Operates on raw space (no VAE).

Architecture (1D adaptation of Peebles & Xie 2023, DiT-B/4 sizing):

  Input:  (B, 40, 100) noised cycle x_t + (B,) timestep t + (B,) source_id
  Patchify: Conv1d(40 -> 384, kernel=stride=4) -> (B, 25, 384) + pos embed
  Conditioning c: timestep MLP + source_embed + (optional) cohort/severity embed
  12 x DiT block (AdaLN-Zero):
        ln + shift/scale -> attn(6 heads) -> gate -> add
        ln + shift/scale -> mlp(hidden=4x) -> gate -> add
  FinalLayer: ln + shift/scale -> Linear(384 -> 4*40) -> unpatchify -> (B, 40, 100)

Conditioning vocab:
  source_id  : 0..11 + 12 = null (CFG drop)         13 entries
  cohort_id  : 0..3  + 4  = null (CFG drop)         5 entries  (Stage 2+)
  severity_id: 0..5  + 6  = null (CFG drop)         7 entries  (Stage 2+ optional)

CFG drop is applied independently per condition in training mode
(`cfg_drop_prob` default 0.10). At inference, pass `*_id = num_X` (the
null index) for the unconditional pass.

Stage 1: pass only source_id; leave cohort_id / severity_id = None so
their embeddings contribute 0 to c (the conditioning is purely
timestep + source).
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding + 2-layer MLP, DiT-paper convention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


# -----------------------------------------------------------------------
class Patchify1D(nn.Module):
    """1D patch embedding: (B, C_in, T) -> (B, T/P, hidden) + pos embed."""

    def __init__(self, in_channels: int = 40, patch_size: int = 4,
                 hidden_size: int = 384, seq_len: int = 100):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"seq_len {seq_len} not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size            # 100 / 4 = 25
        self.proj = nn.Conv1d(in_channels, hidden_size,
                               kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> conv -> (B, hidden, n_patches) -> (B, n_patches, hidden)
        h = self.proj(x).transpose(1, 2)
        return h + self.pos_embed


# -----------------------------------------------------------------------
def _modulate(x: torch.Tensor, shift: torch.Tensor,
              scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulate. x: (B, T, H); shift/scale: (B, H)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, hidden_size: int = 384, num_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # AdaLN-Zero: 6 modulation params per block (attn shift/scale/gate +
        # mlp shift/scale/gate); zero-init so the block is identity at start.
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaln(c).chunk(6, dim=-1))

        # Attention with AdaLN modulation
        h = _modulate(self.norm1(x), shift_a, scale_a)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn_out

        # MLP with AdaLN modulation
        h = _modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


# -----------------------------------------------------------------------
class FinalLayer(nn.Module):
    """Final unpatchify projection -> (B, out_channels, seq_len)."""

    def __init__(self, hidden_size: int = 384, patch_size: int = 4,
                 out_channels: int = 40):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_size * out_channels)
        # Zero-init so the final layer is identity at start (DiT paper)
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        x = self.linear(x)                                       # (B, T/P, P*C_out)
        B, n_patch, _ = x.shape
        # Reshape patches back to time dimension: (B, n_patch, P, C_out)
        # then transpose to (B, C_out, n_patch * P) = (B, C_out, seq_len)
        x = x.reshape(B, n_patch, self.patch_size, self.out_channels)
        x = x.permute(0, 3, 1, 2).reshape(B, self.out_channels,
                                            n_patch * self.patch_size)
        return x


# -----------------------------------------------------------------------
class DiT_B_1D(nn.Module):
    """DiT-B/1D for V4 Stage 1 raw-space diffusion.

    Args:
        in_channels: input channels (40 for V4 reduced layout)
        patch_size: temporal patch size (4 -> 100/4 = 25 patches)
        hidden_size: transformer hidden (384 for DiT-B)
        depth: number of DiT blocks (12 for DiT-B)
        num_heads: attention heads (6 for DiT-B; hidden/64)
        seq_len: input temporal length (100)
        num_sources: 12 sources (7 AddBio + 5 cohort); +1 reserved for null
        num_cohorts: 4 cohorts for Stage 2+ (cp, normal, vdk_stroke, bmclab_pd); +1 null
        num_severities: 6 severity bins, Stage 2+ optional; +1 null
        cfg_drop_prob: independent CFG dropout probability per condition (training)
    """

    def __init__(
        self,
        in_channels: int = 40,
        patch_size: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        seq_len: int = 100,
        num_sources: int = 12,
        num_cohorts: int = 4,
        num_severities: int = 6,
        cfg_drop_prob: float = 0.10,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.cfg_drop_prob = cfg_drop_prob
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_sources = num_sources
        self.num_cohorts = num_cohorts
        self.num_severities = num_severities

        # Patchify
        self.patchify = Patchify1D(in_channels, patch_size, hidden_size, seq_len)

        # Conditioning
        self.t_embed = TimestepEmbedding(hidden_size)
        # +1 for the null embedding used by CFG (index = num_X)
        self.source_embed = nn.Embedding(num_sources + 1, hidden_size)
        self.cohort_embed = nn.Embedding(num_cohorts + 1, hidden_size)
        self.severity_embed = nn.Embedding(num_severities + 1, hidden_size)
        nn.init.normal_(self.source_embed.weight, std=0.02)
        nn.init.normal_(self.cohort_embed.weight, std=0.02)
        nn.init.normal_(self.severity_embed.weight, std=0.02)
        # Zero the null index so unconditional pass is "no signal" at init
        with torch.no_grad():
            self.source_embed.weight[num_sources].zero_()
            self.cohort_embed.weight[num_cohorts].zero_()
            self.severity_embed.weight[num_severities].zero_()

        # Transformer
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden_size, patch_size, in_channels)

    @staticmethod
    def _cfg_drop(ids: torch.Tensor, null_idx: int, prob: float) -> torch.Tensor:
        """Replace `ids` with `null_idx` element-wise with probability `prob`."""
        drop = torch.rand(ids.shape, device=ids.device) < prob
        return torch.where(drop, torch.full_like(ids, null_idx), ids)

    def get_conditioning(
        self,
        t: torch.Tensor,
        source_id: torch.Tensor,
        cohort_id: Optional[torch.Tensor] = None,
        severity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build conditioning vector c (B, hidden_size).

        Each conditioning channel is dropped independently with `cfg_drop_prob`
        at training time. At inference, pass `*_id = num_X` (the null index)
        for the unconditional pass.
        """
        c = self.t_embed(t)

        sid = source_id
        if self.training and self.cfg_drop_prob > 0:
            sid = self._cfg_drop(sid, self.num_sources, self.cfg_drop_prob)
        c = c + self.source_embed(sid)

        if cohort_id is not None:
            cid = cohort_id
            if self.training and self.cfg_drop_prob > 0:
                cid = self._cfg_drop(cid, self.num_cohorts, self.cfg_drop_prob)
            c = c + self.cohort_embed(cid)

        if severity_id is not None:
            vid = severity_id
            if self.training and self.cfg_drop_prob > 0:
                vid = self._cfg_drop(vid, self.num_severities, self.cfg_drop_prob)
            c = c + self.severity_embed(vid)

        return c

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_id: torch.Tensor,
        cohort_id: Optional[torch.Tensor] = None,
        severity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, C_in, T) noised input. Returns same shape (v-pred)."""
        c = self.get_conditioning(t, source_id, cohort_id, severity_id)
        h = self.patchify(x)
        for blk in self.blocks:
            h = blk(h, c)
        return self.final(h, c)


# -----------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------
if __name__ == "__main__":
    model = DiT_B_1D()
    n = count_params(model)
    print(f"DiT-B/1D params: {n / 1e6:.2f} M")

    B = 4
    x = torch.randn(B, 40, 100)
    t = torch.randint(0, 1000, (B,))
    src = torch.randint(0, 12, (B,))

    # Training-mode forward (CFG drop active)
    model.train()
    out = model(x, t, src)
    print(f"output shape (train): {tuple(out.shape)}")
    assert out.shape == x.shape, "shape mismatch"
    assert torch.isfinite(out).all(), "non-finite output"

    # Eval-mode + Stage 2 conditioning
    model.eval()
    coh = torch.randint(0, 4, (B,))
    sev = torch.randint(0, 6, (B,))
    out2 = model(x, t, src, coh, sev)
    assert out2.shape == x.shape

    # CFG null pass
    null_src = torch.full((B,), 12, dtype=torch.long)
    out_null = model(x, t, null_src)
    assert out_null.shape == x.shape

    print("DiT-B/1D smoke OK")
