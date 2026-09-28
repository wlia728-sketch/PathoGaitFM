"""V4 lazy transform V3 — Unified BW-normalize pipeline.

Replaces V2's channel-aware bound system with a single uniform pipeline:

  Layer A — De-normalize each subject back to raw physical units:
              moments  raw (Nm/kg) × BW (kg) -> Nm
              GRF      raw (BW dim) × BW × g -> N
              power    raw (W/kg) × BW       -> W
              angles + EMG: pass-through (already deg / [0,1])
  Layer B — Unified BW-normalize back to a consistent dim-less state:
              moments / BW                   -> Nm/kg
              GRF     / (BW × g)             -> BW dimensionless
              power   / BW                   -> W/kg
              angles + EMG: pass-through (deg / [0,1])
            (After A+B every source's signal is in identical physics.)
  Layer C — Polarity flip (carry-over from V2 work):
              ADDBIO_FLIP_54CH (legacy 23-flip — chose conservative table;
              Option-2 30-flip aborted because Tan21 ch26/27 inversion
              was incorrect under the V4 reference. Moore + Tan22
              flips remain in this 23-flip table.)
              COHORT_FLIP_54CH_RAW (Path C+ raw cohort polarity).
  Layer D — Global z-score (per channel, fixed training-only stats):
              (x - mu_ch) / (3 * sigma_ch)
              The fitting population is recorded by the stats producer.
              EMG channels (54-ch 40..47) skip z-score; they remain in
              the [0, 1] envelope established by upstream mask extraction.
  Layer E — Hard clip [-1.5, +1.5] + channel reduction 54 -> 40.

The A+B round-trip is mathematically a no-op for any source already in
Nm/kg + BW dim (confirmed Phase 2 for all 12 sources). It is kept as
explicit layers so future sources whose raw state differs can plug a
per-source de-norm op without touching the rest of the pipeline.

Signature change vs V2: `__call__` now requires `subject_id` so it can
look up BW from V4SubjectMetadata. Layers A+B both depend on the
subject's mass.
"""
from __future__ import annotations
import json
from typing import Dict, Optional, List

import numpy as np

from .channels import (
    KEEP_CHANNELS_54TO40,
    EXCLUDED_STUDIES,
)
from .cohort_polarity import COHORT_FLIP_54CH_RAW
from .subject_metadata import V4SubjectMetadata, GRAVITY_MS2


# === Channel groups (54-ch raw indices) ===
MOMENT_CHANNELS_54CH = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
POWER_CHANNELS_54CH  = [28, 29, 30, 31, 32, 33]
GRF_CHANNELS_54CH    = [34, 35, 36, 37, 38, 39]
ANGLE_CHANNELS_54CH  = list(range(0, 18)) + [48, 49, 50, 51, 52, 53]
EMG_CHANNELS_54CH    = [40, 41, 42, 43, 44, 45, 46, 47]


# === Expanded AddBio polarity flip table (28 flips, up from legacy 23) ===
# Sources: Round-1 H2 subject-level audit
# and Option-2 per-subject
# PCC verification against V4-transformed vdk_healthy reference
# (preserved at dataset/_option2_aborted/v4_option2_fix/phase3b_subject_pcc.json).
#
# New additions vs legacy ADDBIO_FLIP_54CH (PCC verify PASS for each):
#   Moore2015  +ch12 (R_Ankle_X)    11/11 subj pre PCC<0, post PCC>0
#              +ch15 (L_Ankle_X)    11/11
#              +ch20 (R_Hip_Mom_Z)  11/11
#              +ch23 (L_Hip_Mom_Z)  11/11
#   Tan2022    +ch23 (L_Hip_Mom_Z)  15/16 subj swung neg->pos
#
# Tan2021 ch26 / ch27 were proposed by Round-1 H2 (100% subjects flagged)
# but Option-2 per-subject re-audit against V4 reference showed pre PCC
# already +0.375, 0/7 subjects negative — flipping reverses correlation
# to -0.38. NOT added here.
ADDBIO_FLIP_54CH_V3: Dict[str, list] = {
    "Camargo2021":   [12, 15, 20, 23],                  # unchanged (4)
    "Carter2023":    [20, 23, 48, 51],                  # unchanged (4)
    "Moore2015":     [12, 15, 20, 23, 48, 51],          # +12,15,20,23 (6)
    "Tan2021":       [20, 23, 48, 51],                  # unchanged (4) — ch26/27 incorrect per re-audit
    "Tan2022":       [12, 15, 20, 23],                  # +23          (4)
    "Wang2023":      [20, 23, 48, 51],                  # unchanged (4) — ch12/15 borderline R/L asymmetric
    "vanderZee2022": [12, 15],                          # unchanged (2)
}                                                        # total = 28 (up from 23)


# ============================================================
# COHORT_OFFSET_54CH — Layer C.5 zero-reference offset correction
# ============================================================
# Identified by per-cohort diagnostic:
#   bmclab_pd hip flexion (ch 0, 3) shows shape-level identity
#   with vdk_healthy reference (PCC = +0.96/+0.95) but +23.71deg/+24.46deg
#   absolute zero-reference offset.
#
#   Attributed to OpenSim model neutral pose vs UoA pipeline pelvis
#   reference frame difference. Without correction, this offset would
#   propagate as a cohort-conditioned learning signal.
#
# Offset values derived empirically from cohort-mean waveform difference:
#   mean(vdk_healthy_R_Hip_X) - mean(bmclab_pd_R_Hip_X) = +23.71deg
#   mean(vdk_healthy_L_Hip_X) - mean(bmclab_pd_L_Hip_X) = +24.46deg
#
# Applied in raw degrees BEFORE z-score normalization (Layer C.5).
# Paper Methods M2: 3-layer correction framework (Layer 3).
COHORT_OFFSET_54CH: Dict[str, Dict[int, float]] = {
    "bmclab_pd": {
        0: +23.71,   # R_Hip_Angle_X (deg)
        3: +24.46,   # L_Hip_Angle_X (deg)
    },
    "cp":          {},
    "normal":      {},
    "vdk_stroke":  {},
    "vdk_healthy": {},
    # AddBio 7 studies — explicit empty (not used in Stage 2, but kept for completeness)
    "Camargo2021":   {},
    "Carter2023":    {},
    "Moore2015":     {},
    "Tan2021":       {},
    "Tan2022":       {},
    "Wang2023":      {},
    "vanderZee2022": {},
}


class V4LazyTransformV3:
    """Unified BW-normalize transform.

    Args:
      metadata: V4SubjectMetadata instance.
      global_zstats_path: path to global_zstats.json (Phase 5).
                            JSON schema: {channel_54_idx_str: {"mean", "std"}}.
                            If None, Layer D becomes a no-op (useful for
                            Phase 5 stat-fitting itself).
      addbio_flip_override / cohort_flip_override: optional polarity overrides.
      clip_bound: hard clip range, default 1.5.
    """

    def __init__(
        self,
        metadata: V4SubjectMetadata,
        global_zstats_path: Optional[str] = None,
        addbio_flip_override: Optional[Dict[str, List[int]]] = None,
        cohort_flip_override: Optional[Dict[str, List[int]]] = None,
        cohort_offset_override: Optional[Dict[str, Dict[int, float]]] = None,
        clip_bound: float = 1.5,
    ):
        self.metadata = metadata
        if global_zstats_path is not None:
            with open(global_zstats_path) as f:
                self.zstats: Optional[Dict[str, Dict[str, float]]] = json.load(f)
        else:
            self.zstats = None
        self.addbio_flip = (addbio_flip_override
                            if addbio_flip_override is not None
                            else dict(ADDBIO_FLIP_54CH_V3))
        self.cohort_flip = (cohort_flip_override
                            if cohort_flip_override is not None
                            else dict(COHORT_FLIP_54CH_RAW))
        # Layer C.5 — per-cohort zero-reference offset
        self.cohort_offset: Dict[str, Dict[int, float]] = (
            cohort_offset_override
            if cohort_offset_override is not None
            else {k: dict(v) for k, v in COHORT_OFFSET_54CH.items()}
        )
        self.clip_bound = float(clip_bound)

    def _flip_for(self, source_name: str, source_type: str) -> List[int]:
        if source_type == "addbio":
            return self.addbio_flip.get(source_name, [])
        return self.cohort_flip.get(source_name, [])

    def __call__(
        self,
        arr_54ch: np.ndarray,
        source_name: str,
        source_type: str,
        subject_id: str,
    ) -> np.ndarray:
        """Transform (n_cyc, 101, 54) -> (n_cyc, 101, 40) in [-clip, +clip].

        Args:
          arr_54ch: input raw (n_cyc, 101, 54).
          source_name: e.g. 'Camargo2021', 'cp', 'vdk_healthy'.
          source_type: 'addbio' or 'cohort'.
          subject_id: filename stem (e.g. 'Camargo2021_AB06_split0',
                       'vdk_healthy_000', 'cp_01', 'SUB01_off').
        """
        x = self.preprocess_before_zscore(
            arr_54ch, source_name, source_type, subject_id
        )

        # Layer D: global z-score (per channel, fixed training-only stats).
        # Scale to map +/-3 sigma -> +/-1 so the +/-1.5 clip in Layer E
        # acts as the same 4.5-sigma outlier filter the V2 pipeline used;
        # the model was trained on [-1, +1] inputs.
        if self.zstats is not None:
            for ch_idx in (MOMENT_CHANNELS_54CH + POWER_CHANNELS_54CH
                           + GRF_CHANNELS_54CH + ANGLE_CHANNELS_54CH):
                st = self.zstats.get(str(ch_idx))
                if st is None:
                    continue
                mean = float(st["mean"])
                std  = float(st["std"])
                if std < 1e-8:
                    x[..., ch_idx] = 0.0
                else:
                    x[..., ch_idx] = (x[..., ch_idx] - mean) / (3.0 * std)
            # EMG channels intentionally skipped (already mask-aware [0,1])

        # Layer E: hard clip + channel reduce.
        finite = np.isfinite(x)
        x[finite] = np.clip(x[finite], -self.clip_bound, self.clip_bound)
        return x[..., KEEP_CHANNELS_54TO40]

    def preprocess_before_zscore(
        self,
        arr_54ch: np.ndarray,
        source_name: str,
        source_type: str,
        subject_id: str,
    ) -> np.ndarray:
        """Apply Layers A-C.6 and return the unreduced 54-channel array.

        This is the canonical stat-fitting entry point.  A z-score producer
        must call this method so that its polarity, zero-reference and EMG
        handling are identical to model training, while avoiding both the
        z-score itself and the final clipping/channel reduction.
        """
        if source_type == "addbio" and source_name in EXCLUDED_STUDIES:
            raise ValueError(f"{source_name} excluded from training")

        bw_kg = self.metadata.get_bw(
            f"addbio_{source_name}" if source_type == "addbio" else source_name,
            subject_id,
        )
        bw_g = bw_kg * GRAVITY_MS2

        x = arr_54ch.astype(np.float32, copy=True)

        # Layers A + B: de-normalize then unified BW-normalize.
        # For all 12 sources, the underlying signals are already in
        # (Nm/kg, BW dim, deg, W/kg, [0,1]) per Phase 2 verdict. So
        # A * B together is the identity for moments/GRF/power. We
        # explicitly model A + B even though they cancel, so future
        # sources with different raw state can plug in per-source A
        # without modifying B.
        # Since arr_54ch is already in unified-state for all 12 sources,
        # both layers become a no-op. We leave Layers A+B as comments
        # and verified factual statements below.
        _ = bw_kg, bw_g  # noqa: used only for future per-source A.

        # Layer C: polarity flip.
        for ch in self._flip_for(source_name, source_type):
            x[..., ch] *= -1.0

        # Layer C.5 :
        # Per-cohort additive zero-reference offset, applied in raw units
        # (degrees for angles) AFTER Layer C polarity flip and BEFORE
        # Layer D z-score normalization. NaN-safe: np addition preserves NaN
        # so masked-invalid timepoints stay invalid. Currently used only for
        # bmclab_pd hip angles (ch 0, 3) — see COHORT_OFFSET_54CH docstring.
        for ch, offset in self.cohort_offset.get(source_name, {}).items():
            x[..., ch] = x[..., ch] + offset

        # Layer C.6: EMG envelope clip.
        # Phase 1c finding: vdk_healthy / vdk_stroke / cp / normal EMG raw
        # is rectified-envelope mostly in [0, 1] but with occasional peak
        # bursts up to 2.5. AddBio + vdk + bmclab have no EMG (all NaN).
        # Clip to [0, 1] semantic envelope so EMG passes through Layer E
        # without unnecessary +/-1.5 saturation. NaN is preserved by np.clip.
        for ch in EMG_CHANNELS_54CH:
            finite_emg = np.isfinite(x[..., ch])
            v = x[..., ch]
            np.clip(v, 0.0, 1.0, out=v, where=finite_emg)

        return x
