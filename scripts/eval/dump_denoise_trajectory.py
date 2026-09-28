# -*- coding: utf-8 -*-
"""Record the real reverse-diffusion trajectory for one held-out gait cycle.

Uses the fold checkpoint that held this subject out, the released per-subject transform,
and the same sampler constants as the reported evaluation (50-step DDIM, CFG 1.0,
DROP_16CH token fill). The only change is that the state x is captured at six steps.
Writes outputs/dumps/denoise_traj.npz by default.
"""
import sys, argparse
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_common import (load_model, build_known_mask, SEED, SEVERITY_JSON,
                          KEEP_CHANNELS_54TO40, SOURCE_TO_ID, load_local_json, per_subject_transform)
from tsa_inference import GRFFIX, load_split
from drop_channels import DROP_16CH
from guarded_write import guarded_savez
from pathogait.diffusion.ddpm import DDPMScheduler
from pathogait.data.subject_metadata import V4SubjectMetadata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, help="cohort: cp, normal, vdk_stroke or bmclab_pd")
parser.add_argument("--stem", required=True, help="recording stem within the cohort")
parser.add_argument("--cycle", type=int, default=0, help="cycle index within the recording")
parser.add_argument("--raw-root", type=Path, default=GRFFIX)
parser.add_argument("--out", type=Path, default=ROOT / "outputs/dumps/denoise_traj.npz")
args = parser.parse_args()
SRC, STEM, CYCLE = args.source, args.stem, args.cycle
SPLIT = load_split()

# ---- which fold holds this subject out -----------------------------------
fold = None
for f in range(5):
    if [SRC, STEM] in [list(x) for x in SPLIT["cv5"][f]["val"]]:
        fold = f
        break
assert fold is not None, "subject not found in any validation split"
CKPT = ROOT / "checkpoints" / ("v4_stage2_expA_dynpool_cv5_%d" % fold) / "final.pt"
print("subject %s/%s is held out in fold cv5_%d" % (SRC, STEM, fold))
print("checkpoint:", CKPT.name, "exists:", CKPT.exists())
assert CKPT.exists()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt, weights, imt = load_model(CKPT, device)
model.eval()
_i = ckpt.get("input_mask_token_ema")
if _i is None:
    _i = ckpt.get("input_mask_token")
if _i is not None:
    imt = _i
print("weights:", weights, "| input mask token:", None if imt is None else tuple(imt.shape))

# ---- the released data path ----------------------------------------------
meta = V4SubjectMetadata()
transform = per_subject_transform(meta, require_tables=True)

raw = np.load(str(args.raw_root / SRC / ("%s.npy" % STEM)))
mask = np.load(str(args.raw_root / SRC / ("%s_mask.npy" % STEM)))
v40 = mask[:, KEEP_CHANNELS_54TO40].astype(bool)
x40 = transform(raw, SRC, "cohort", STEM)[:, :100, :].astype(np.float32)
print("transformed:", x40.shape, "valid mask:", v40.shape)

gt = torch.from_numpy(x40[CYCLE][None].transpose(0, 2, 1)).to(device)     # (1, 40, 100)
gt = torch.nan_to_num(gt, nan=0.0)
valid = torch.from_numpy(v40[CYCLE][None]).to(device)

sev = load_local_json(SEVERITY_JSON, "severity table")["mappings"][SRC][STEM]
sid = torch.full((1,), SOURCE_TO_ID.get(SRC, 0), dtype=torch.long, device=device)
cid = torch.full((1,), int(sev["cohort_id"]), dtype=torch.long, device=device)
svid = torch.full((1,), int(sev["severity_id"]), dtype=torch.long, device=device)
print("source_id %d cohort_id %d severity_id %d" % (sid.item(), cid.item(), svid.item()))

# ---- sampler, identical to tweedie_inpaint_drop but recording x -----------
sched = DDPMScheduler(1000, "cosine").to(device)
NUM_STEPS, CFG = 50, 1.0
CAPTURE = [0, 5, 15, 25, 40]          # capture x at the START of these steps

km = build_known_mask(gt, valid)
dv = torch.zeros(40, dtype=torch.bool, device=device)
dv[DROP_16CH] = True
token_mask = (~valid.bool()).unsqueeze(-1) | dv.view(1, 40, 1)
tv = None
if imt is not None:
    tok = imt.to(device=device, dtype=gt.dtype)
    if tok.ndim == 2:
        tok = tok.unsqueeze(-1)
    tv = tok.expand(1, -1, 100)
km = km * (~token_mask).float()
print("known rows:", int(km[0, :, 0].sum().item()))

ns = torch.full((1,), model.num_sources, dtype=torch.long, device=device)
nc = torch.full((1,), model.num_cohorts, dtype=torch.long, device=device)
nv = torch.full((1,), model.num_severities, dtype=torch.long, device=device)

torch.manual_seed(SEED)
x = torch.randn(1, 40, 100, device=device)
ts = torch.linspace(999, 0, steps=NUM_STEPS + 1).long().to(device)

traj, traj_t, x0s = [], [], []
with torch.no_grad():
    for i in range(NUM_STEPS):
        if i in CAPTURE:
            traj.append(x[0].detach().cpu().numpy().copy())
            traj_t.append(int(ts[i].item()))
        tc = ts[i].expand(1)
        tn = ts[i + 1].expand(1)
        vc = model(x, tc, sid, cid, svid)
        vu = model(x, tc, ns, nc, nv)
        v = vu + CFG * (vc - vu)
        x0h = sched.predict_x0_from_v(x, v, tc).clamp(-1.5, 1.5)
        if i in CAPTURE:
            x0s.append(x0h[0].detach().cpu().numpy().copy())
        epsh = sched.predict_noise_from_v(x, v, tc)
        ab = sched._gather(sched.alpha_bars, tn, x.dim())
        xu = ab.sqrt() * x0h + ((1 - ab).clamp(min=0)).sqrt() * epsh
        xk = sched.add_noise(gt, torch.randn_like(gt), tn)
        x = km * xk + (1 - km) * xu
        if tv is not None:
            x = torch.where(token_mask, tv, x)
    traj.append(x[0].detach().cpu().numpy().copy())
    traj_t.append(0)
    x0s.append(x[0].detach().cpu().numpy().copy())

traj = np.stack(traj)                      # (6, 40, 100)
x0s = np.stack(x0s)                        # (6, 40, 100) model's estimate of the clean cycle
sq = [float(sched.sqrt_alpha_bars[t]) for t in traj_t]
print("captured t =", traj_t)
print("sqrt(alpha_bar) =", [round(s, 3) for s in sq])

# ---- the released pipeline output on the same cycle, for honest labelling ----
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS
gens = []
with torch.no_grad():
    for so in range(3):
        torch.manual_seed(SEED + so * 7919)
        gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                         NUM_STEPS, CFG, device, imt, CHECKPOINT_STEPS,
                                         DROP_16CH))
pipe = torch.stack(gens).mean(0)
pipe = km * gt + (1 - km) * pipe
pipe = pipe[0].cpu().numpy()
TG = [10, 13, 16, 17, 18, 19, 22, 25]
g = gt[0].cpu().numpy()
print("full pipeline (3 seeds + TSA) vs one draw, per channel:")
for c in TG:
    print("   ch%-3d  one draw %.3f   pipeline %.3f"
          % (c, np.corrcoef(traj[-1][c], g[c])[0, 1], np.corrcoef(pipe[c], g[c])[0, 1]))

out = args.out
out = guarded_savez(out, traj=traj, x0hat=x0s, pipeline=pipe, traj_t=np.array(traj_t), sqab=np.array(sq),
         gt=gt[0].cpu().numpy(), km=km[0].cpu().numpy(),
         token=token_mask[0, :, 0].cpu().numpy(), fold=fold)
print("saved", out)
