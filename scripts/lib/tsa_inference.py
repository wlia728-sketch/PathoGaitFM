"""Shared inference library: the released estimation procedure, the cohort map and the split loader
every evaluator reuses.

`tweedie_inpaint_drop` runs ONE 50-step DDIM denoising pass of ONE model, conditioned on the observed
channels, which are re-noised from the observation at every step and recombined through the known
mask. It keeps the x0 estimate at the six CHECKPOINT_STEPS and the final state, seven waveforms of the
same trajectory, and returns their trimmed mean (the per-cell minimum and maximum dropped). The
three-seed average the manuscript reports is taken by the CALLER: each evaluator calls this function
three times with torch.manual_seed(SEED + so * 7919) and averages the three returns (see
eval_kinetics_all_cycles.py, the 0.811 headline, and scripts/lib/pathogait_api.py). The function also
applies the final-8ch input regime, token-filling DROP_16CH (the EMG and non-sagittal hip-moment
channels) at input so inference matches training. The constants below are imported by every
evaluator.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eval_common import build_known_mask, SPLIT_JSON, load_local_json

# the four development cohorts after the ground-reaction-force correction, built locally
# (data/README.md); the evaluators take --raw-root to read them from elsewhere
GRFFIX = ROOT / "data" / "cohorts_grffix"
# recording stems excluded from every evaluation, comma-separated in PATHOGAIT_EXCLUDE (docs/training.md);
# empty by default because the manuscript split already omits the recordings that failed quality control
EXCLUDE = {s for s in os.environ.get("PATHOGAIT_EXCLUDE", "").split(",") if s}
COH = {0: "cp", 1: "normal", 2: "vdk_stroke", 3: "bmclab_pd"}
# manuscript: "3-seed", "DDIM 50-step", "CFG 1.0"; checkpoint steps for within-traj TSA
CHECKPOINT_STEPS = [5, 10, 15, 25, 35, 45]


def load_split(path=None):
    """The five-fold split of the development cohorts, built locally (docs/training.md)."""
    return load_local_json(SPLIT_JSON if path is None else Path(path), "cross-validation split")


def tweedie_inpaint_drop(model, sched, gt_norm, valid_mask, sid, cid, svid,
                         num_steps, cfg, device, imt, ckpts, drop_ch):
    """tweedie_inpaint (manuscript TSA) + final-8ch input regime: force-token-fill
    drop_ch at input (EMG + non-sagittal hip moment) so eval matches training."""
    bsz = gt_norm.shape[0]
    gt_norm = gt_norm.to(device); valid_mask = valid_mask.to(device)
    gt_safe = torch.nan_to_num(gt_norm, nan=0.0)
    km = build_known_mask(gt_norm, valid_mask)
    # drop_ch: treat as input-missing regardless of validity (training regime)
    dv = torch.zeros(40, dtype=torch.bool, device=device); dv[drop_ch] = True
    token_mask = (~valid_mask.to(device=device, dtype=torch.bool)).unsqueeze(-1) | dv.view(1, 40, 1)
    tv = None
    if imt is not None:
        tok = imt.to(device=device, dtype=gt_safe.dtype)
        if tok.ndim == 2: tok = tok.unsqueeze(-1)
        tv = tok.expand(bsz, -1, gt_safe.shape[-1])
    km = km * (~token_mask).float()  # dropped/invalid channels are NOT known
    ns = torch.full((bsz,), model.num_sources, dtype=torch.long, device=device)
    nc = torch.full((bsz,), model.num_cohorts, dtype=torch.long, device=device)
    nv = torch.full((bsz,), model.num_severities, dtype=torch.long, device=device)
    x = torch.randn(bsz, 40, 100, device=device)
    ts = torch.linspace(sched.num_train_timesteps - 1, 0, steps=num_steps + 1).long().to(device)
    x0s = []
    for i in range(num_steps):
        tc = ts[i].expand(bsz); tn = ts[i + 1].expand(bsz)
        vc = model(x, tc, sid, cid, svid); vu = model(x, tc, ns, nc, nv)
        v = vu + cfg * (vc - vu)
        x0 = sched.predict_x0_from_v(x, v, tc).clamp(-1.5, 1.5)
        eps = sched.predict_noise_from_v(x, v, tc)
        if i in ckpts: x0s.append(x0.clone())
        ab = sched._gather(sched.alpha_bars, tn, x.dim())
        xu = ab.sqrt() * x0 + ((1 - ab).clamp(min=0)).sqrt() * eps
        xk = sched.add_noise(gt_safe, torch.randn_like(gt_safe), tn)
        x = km * xk + (1 - km) * xu
        if tv is not None: x = torch.where(token_mask, tv, x)
    x0s.append(x)
    st = torch.stack(x0s, dim=0)
    if st.shape[0] >= 3:  # within-trajectory trimmed mean (TSA layer 1)
        sv, _ = torch.sort(st, dim=0); xf = sv[1:-1].mean(dim=0)
    else:
        xf = st.mean(dim=0)
    return km * gt_safe + (1 - km) * xf


def summarize(acc):
    """{(cohort, channel): [per-cycle PCCs]} -> {cohort: {channel: {pcc, n}}}."""
    per = defaultdict(dict)
    for (c, ch), v in acc.items():
        per[c][ch] = {"pcc": round(float(np.mean(v)), 3), "n": len(v)}
    return dict(per)
