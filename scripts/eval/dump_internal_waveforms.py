# -*- coding: utf-8 -*-
"""Per-subject mean ESTIMATED and REFERENCE waveforms for the internal five-fold CV.

Same protocol as eval_kinetics_all_cycles.py (the 0.811 headline): fold models expA, TSA
3-seed DDIM-50 CFG-1.0, DROP_16CH input regime, Method B per-limb validity, quality-control exclusions applied,
PD moments masked. Nothing is scored here and no reported number is touched; the only output is
the waveforms themselves, averaged within subject, in the frozen z-space AND in physical units
(x_phys = x_z * 3 * std + mean, the evaluator's own denorm). Writes cv5_waveforms.npz.
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS, COH, EXCLUDE, GRFFIX, load_split
from drop_channels import DROP_16CH
from guarded_write import guarded_savez
from eval_common import (load_model, get_val_pairs, load_valid_mask_40, source_type_and_name,
                          ZSTATS_PATH, SOURCE_TO_ID, SEED, SEVERITY_JSON, build_known_mask,
                          load_local_json, per_subject_transform)
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.diffusion.ddpm import DDPMScheduler

_ZS = json.load(open(ZSTATS_PATH))


def denorm(x_z, ch40):
    c54 = str(KEEP_CHANNELS_54TO40[ch40])
    return x_z * (3.0 * _ZS[c54]["std"]) + _ZS[c54]["mean"]


CV5 = ["cv5_%d" % i for i in range(5)]
BATCH = 32
TG = [10, 13, 16, 17, 18, 19, 22, 25]
OUT = ROOT / "outputs" / "dumps" / "cv5_waveforms.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)
meta = V4SubjectMetadata()
transform = per_subject_transform(meta, require_tables=True)
sev = load_local_json(SEVERITY_JSON, "severity table")
SPLIT = load_split()
sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

# (cohort, stem, ch) -> list of per-limb-cycle waveforms (z-space)
pred_acc = defaultdict(list)
ref_acc = defaultdict(list)
pcc_acc = defaultdict(list)
from eval_common import safe_pcc
t0 = time.time()
n_cycles_total = 0

for fold in CV5:
    cp = ROOT / "checkpoints" / ("v4_stage2_expA_dynpool_%s" % fold) / "final.pt"
    model, ckpt, _, imt = load_model(cp, device)
    model.eval()
    _i = ckpt.get("input_mask_token_ema")
    if _i is None:
        _i = ckpt.get("input_mask_token")
    if _i is not None:
        imt = _i
    for src, stem in get_val_pairs(SPLIT, fold):
        if stem in EXCLUDE:
            continue
        rp = GRFFIX / src / ("%s.npy" % stem)
        if not rp.exists():
            raise FileNotFoundError(rp)
        s = sev["mappings"].get(src, {}).get(stem)
        if s is None:
            raise ValueError("Missing severity: %s/%s" % (src, stem))
        raw = np.load(str(rp))
        v40 = load_valid_mask_40(rp, raw.shape[0])
        st, sn = source_type_and_name(src)
        x40 = transform(raw, sn, st, stem)[:, :100, :].astype(np.float32)
        n = min(v40.shape[0], x40.shape[0])
        x40, v40 = x40[:n], v40[:n]
        cohort = COH.get(int(s["cohort_id"]), "other")
        R_SIDE = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18])
        L_SIDE = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
        keep = [ci for ci in range(x40.shape[0])
                if (v40[ci, 22] and all(v40[ci, c] for c in R_SIDE))
                or (v40[ci, 25] and all(v40[ci, c] for c in L_SIDE))]
        if not keep:
            continue
        keep = np.array(keep)
        sid = torch.full((1,), SOURCE_TO_ID.get(src, 0), dtype=torch.long)
        cid = torch.full((1,), int(s["cohort_id"]), dtype=torch.long)
        svid = torch.full((1,), int(s["severity_id"]), dtype=torch.long)
        for b0 in range(0, len(keep), BATCH):
            bidx = keep[b0:b0 + BATCH]
            nb = len(bidx)
            gt = torch.from_numpy(np.nan_to_num(x40[bidx].transpose(0, 2, 1), nan=0.0)).to(device)
            valid = torch.from_numpy(v40[bidx]).to(device)
            bs, bc, bv = sid.expand(nb).to(device), cid.expand(nb).to(device), svid.expand(nb).to(device)
            gens = []
            for so in range(3):
                torch.manual_seed(SEED + so * 7919)
                with torch.no_grad():
                    gens.append(tweedie_inpaint_drop(model, sched, gt, valid, bs, bc, bv,
                                                     50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
            gen = torch.stack(gens).mean(0)
            km = build_known_mask(gt, valid)
            gen = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * gen
            gtn, genn, vm = gt.cpu().numpy(), gen.cpu().numpy(), v40[bidx]
            for ci in range(nb):
                for side_chs, vgrf_ch in [(R_SIDE, 22), (L_SIDE, 25)]:
                    if not vm[ci, vgrf_ch] or not all(vm[ci, c] for c in side_chs):
                        continue
                    for ch in side_chs:
                        r = safe_pcc(gtn[ci, ch, :], genn[ci, ch, :])
                        if not np.isfinite(r):
                            continue
                        pred_acc[(cohort, stem, ch)].append(genn[ci, ch, :].copy())
                        ref_acc[(cohort, stem, ch)].append(gtn[ci, ch, :].copy())
                        pcc_acc[(cohort, stem, ch)].append(float(r))
                        n_cycles_total += 1
        print("  %s %-10s %-14s cycles=%3d  elapsed %5.1f min" % (fold, cohort, stem, len(keep),
              (time.time() - t0) / 60), flush=True)
    del model
    torch.cuda.empty_cache()

# ---- per-subject means, z-space and physical -------------------------------------------------
keys = sorted(pred_acc.keys())
cohorts = np.array([k[0] for k in keys])
stems = np.array([k[1] for k in keys])
chans = np.array([k[2] for k in keys])
ncyc = np.array([len(pred_acc[k]) for k in keys])
pred_z = np.stack([np.mean(pred_acc[k], axis=0) for k in keys])
ref_z = np.stack([np.mean(ref_acc[k], axis=0) for k in keys])
pred_phys = np.stack([denorm(pred_z[i], int(chans[i])) for i in range(len(keys))])
ref_phys = np.stack([denorm(ref_z[i], int(chans[i])) for i in range(len(keys))])
pcc_subj = np.array([np.mean(pcc_acc[k]) for k in keys])
OUT = guarded_savez(OUT, cohort=cohorts, stem=stems, chan=chans, n_cycles=ncyc,
         pred_z=pred_z, ref_z=ref_z, pred_phys=pred_phys, ref_phys=ref_phys, pcc_subj=pcc_subj)
print("saved", OUT, "| subject-channel rows:", len(keys), "| limb-cycles:", n_cycles_total,
      "| %.1f min" % ((time.time() - t0) / 60), flush=True)
# ---- comparison with the per-subject values written by eval_kinetics_all_cycles.py (optional) ----
KINETICS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "evaluation" / "kinetics_cv5.json"
if not KINETICS.is_file():
    print("no evaluation result at %s; per-subject comparison skipped" % KINETICS, flush=True)
    sys.exit(0)
J = json.load(open(KINETICS, encoding="utf-8"))["pretrain/CV5"]
MERGED = {10: "Hip_Mom", 13: "Hip_Mom", 16: "Knee_Mom", 17: "Knee_Mom", 18: "Ankle_Mom", 19: "Ankle_Mom", 22: "GRF_V", 25: "GRF_V"}
subj_ch = defaultdict(list)     # (cohort, stem, merged) -> all limb-cycle PCCs
for k in keys:
    subj_ch[(k[0], k[1], MERGED[k[2]])] += pcc_acc[k]
diffs = []
for (coh, stem, mc), v in subj_ch.items():
    rel = J["per_subject_channel"][coh][stem].get(mc)
    if rel is not None:
        diffs.append((abs(np.mean(v) - rel), coh, stem, mc, round(float(np.mean(v)), 4), rel))
diffs.sort(reverse=True)
print("per-subject-channel PCC replay vs evaluation result: n %d, max |diff| %.4f, n(>0.005) %d" % (len(diffs), diffs[0][0], sum(d[0] > 0.005 for d in diffs)))
for d in diffs[:8]:
    print("   ", d)
coh_macro = {}
for coh in ("cp", "vdk_stroke", "bmclab_pd", "normal"):
    chm = {}
    for mc in ("GRF_V", "Hip_Mom", "Knee_Mom", "Ankle_Mom"):
        ps = [np.mean(v) for (c, s_, m), v in subj_ch.items() if c == coh and m == mc]
        if ps:
            chm[mc] = float(np.mean(ps))
    coh_macro[coh] = float(np.mean(list(chm.values())))
    print("  %-10s replay cohort macro %.3f  evaluation %.3f  per-channel %s" % (coh, coh_macro[coh], J["cohort_subject_macro"][coh]["pcc"], {k: round(v, 3) for k, v in chm.items()}))
print("  headline replay %.3f  evaluation %.3f" % (np.mean(list(coh_macro.values())), J["headline_subject_macro"]))
