"""Shared fair-comparison pipeline for the discriminative baselines.

Provides the split, cycle loading, conditioning embedding, cell-level masked loss (including the bmclab_pd
moment exclusion the diffusion trainer applies), three-seed training and prediction ensembling, and
subject-macro Method-B scoring, all matched to the diffusion evaluation. Imported by
published_comparators.py (internal Table 1); it produces no artifact of its own. Torch; CPU-ok. ASCII.

Comparators are reported for the internal cross-validation only. They are not run at the external sites,
because the Leuven inverse-kinematics solution expresses the pelvis in the global laboratory frame and five
of its twenty trials exceed 5 cm mean marker error, so an external comparison would index the input
quality at that site rather than the architectures.
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # model_baseline -> the package root
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import torch
import torch.nn as nn
from tsa_inference import COH, EXCLUDE, GRFFIX, load_split
from eval_common import (
    get_val_pairs, load_valid_mask_40, source_type_and_name, safe_pcc, SOURCE_TO_ID, SEVERITY_JSON,
    per_subject_transform,
)
from pathogait.data.subject_metadata import V4SubjectMetadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEV = SEVERITY_JSON  # public constant used by both comparator producers
SPLIT = None  # the five-fold split, loaded on first use


def get_split():
    global SPLIT
    if SPLIT is None:
        SPLIT = load_split()
    return SPLIT

CV5 = [f"cv5_{i}" for i in range(5)]
IN_CH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 34, 35, 36, 37, 38, 39]  # 16 sagittal kinematics + pelvis; GRF excluded (kinematics-only, matches the diffusion input mask)
TGT = [10, 13, 16, 17, 18, 19, 22, 25]
NAMES = {10: "R_Hip", 13: "L_Hip", 16: "R_Knee", 17: "L_Knee", 18: "R_Ankle", 19: "L_Ankle", 22: "R_vGRF", 25: "L_vGRF"}
MERGED = {10: "Hip", 13: "Hip", 16: "Knee", 17: "Knee", 18: "Ankle", 19: "Ankle", 22: "vGRF", 25: "vGRF"}
NIN, NT, NOUT = len(IN_CH), 100, len(TGT)
N_SRC, N_COH, N_SEV = 13, 5, 7          # embedding cardinalities, aligned to diffusion num_sources=12 / cohorts=4 / severities=6 (+1 null slot)
EPOCHS, LR, BS, SEED, DCOND, HID = 80, 2e-3, 256, 0, 16, 256
N_SEED = 3      # match the diffusion eval's 3-draw inference ensemble in count

# Cohort-conditional loss exclusion, mirroring the diffusion trainer. train_finetune.py sets
# EXCLUDED_T1_CHANNELS_PER_COHORT = {"bmclab_pd": [18..27]} (54-ch), i.e. every bmclab sagittal joint
# moment, because those moments were computed externally under a convention whose residual drift "cannot
# be resolved by sign-flip, offset, or phase-shift correction" (post-correction PCC -0.172 to +0.613).
# bmclab_pd stays in training and contributes loss through its directly-measured channels only.
# The comparators apply the same exclusion, so both sides are supervised on the same target cells.
PD_MOMENT_TGT = [10, 13, 16, 17, 18, 19]        # the six sagittal moment channels inside TGT
PD_MOMENT_COHORT = "bmclab_pd"


def get_split_subjects(split, fold):
    """train subjects = all CV5 folds except this one's val; val = this fold's val.

    Returns SORTED lists, not sets. load_cycles iterates whatever this returns, so a set would make the
    training sample order depend on per-process string-hash randomisation and the fitted weights would not
    reproduce run to run.
    """
    val = set(stem for _, stem in get_val_pairs(split, fold))
    train = set()
    for f in CV5:
        if f == fold: continue
        for _, stem in get_val_pairs(split, f): train.add(stem)
    return sorted(train - val), sorted(val)


def load_cycles(transform, sev, stems):
    """(cohort, stem, xin(N,1600), yout(N,8,100), v40(N,40), cond(3,)=[source_id, cohort_id, severity_id]).

    Only the declared EXCLUDE subjects are skipped. Missing inputs or failed
    transforms abort the comparison instead of changing its population.
    """
    data = []
    src_of = {}
    for src, mp in sev["mappings"].items():
        for st in mp: src_of.setdefault(st, src)
    for stem in stems:
        if stem in EXCLUDE: continue
        src = src_of.get(stem)
        if src is None: raise ValueError(f"Missing comparator severity/source label: {stem}")
        rp = GRFFIX / src / f"{stem}.npy"
        if not rp.is_file(): raise FileNotFoundError(rp)
        mask_path = rp.with_name(rp.stem + "_mask.npy")
        if not mask_path.is_file(): raise FileNotFoundError(mask_path)
        s = sev["mappings"].get(src, {}).get(stem)
        if s is None: raise ValueError(f"Missing comparator label: {src}/{stem}")
        try:
            raw = np.load(str(rp)); v40 = load_valid_mask_40(rp, raw.shape[0])
            st_, sn = source_type_and_name(src)
            x40 = transform(raw, sn, st_, stem)[:, :100, :].astype(np.float32)
            if x40.shape[0] == 0 or x40.shape[1:] != (100, 40) or v40.shape != (x40.shape[0], 40):
                raise ValueError(f"Invalid or mismatched waveform/mask shapes: {x40.shape}, {v40.shape}")
        except Exception as exc:
            raise RuntimeError(f"Cannot load comparator input {src}/{stem}") from exc
        cohort = COH.get(int(s["cohort_id"]), "other")
        # Inputs are zero-filled where invalid. The visible kinematics are NOT valid by construction:
        # measured over the CV5 set, 5 of the 16 input channels are structurally missing for post-stroke
        # and Parkinson recordings (invalid
        # fraction cp 0.078, typically-developing 0.100, stroke 0.3125, Parkinson 0.3125). After z-scoring a
        # zero is the channel mean, and unlike the diffusion model these baselines receive no learned mask
        # token and no explicit unknown flag, so a missing channel is indistinguishable from an average one.
        xin = np.nan_to_num(x40[:, :, IN_CH], nan=0.0).reshape(x40.shape[0], -1)
        yout = np.nan_to_num(x40[:, :, TGT], nan=0.0).transpose(0, 2, 1)             # (N,8,100); invalid target cells are MASKED OUT of the loss below, not supervised as 0
        cond = np.array([int(SOURCE_TO_ID.get(src, N_SRC - 1)), int(s["cohort_id"]), int(s["severity_id"])], np.int64)
        data.append((cohort, stem, xin, yout, v40, cond))
    return data


def preflight_inputs(sev):
    """Check every planned fold's required arrays and masks before training starts."""
    sources = {}
    for src, labels in sev["mappings"].items():
        for stem in labels:
            sources.setdefault(stem, src)
    planned = set()
    for fold in CV5:
        training, validation = get_split_subjects(get_split(), fold)
        for label, stems in (("training", training), ("validation", validation)):
            included = set(stems) - set(EXCLUDE)
            if not included:
                raise ValueError(f"Comparator {fold} has no {label} units after declared exclusions")
            planned.update(included)
    for stem in sorted(planned):
        src = sources.get(stem)
        if src is None:
            raise ValueError(f"Missing comparator severity/source label: {stem}")
        path = GRFFIX / src / f"{stem}.npy"
        mask_path = path.with_name(stem + "_mask.npy")
        for required in (path, mask_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing required comparator input: {required}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3 or raw.shape[0] == 0 or raw.shape[1] < NT or raw.shape[2] != 54:
            raise ValueError(f"Invalid comparator waveform shape at {path}: {raw.shape}")
        if mask.shape not in ((54,), (len(raw), 54)):
            raise ValueError(f"Comparator mask/array shape mismatch at {path}: {mask.shape}")


class CondEmbed(nn.Module):
    """source + cohort + severity -> additive conditioning vector; the same three factors the diffusion model sees."""
    def __init__(self, d=DCOND):
        super().__init__()
        self.es = nn.Embedding(N_SRC, d); self.ec = nn.Embedding(N_COH, d); self.ev = nn.Embedding(N_SEV, d)

    def forward(self, cond):  # (B,3) long -> (B,d)
        return self.es(cond[:, 0]) + self.ec(cond[:, 1]) + self.ev(cond[:, 2])



def target_mask_np(v40, cohort=None):
    """(N,40) -> (N,8) float validity of the 8 TGT channels, with the diffusion trainer's cohort exclusion."""
    m = v40[:, TGT].astype(np.float32)
    if cohort == PD_MOMENT_COHORT:
        for ch in PD_MOMENT_TGT:
            m[:, TGT.index(ch)] = 0.0
    return m


def masked_mse(pred, y, tmask):  # pred,y (B,8,100); tmask (B,8) -> mean over VALID cells only
    m = tmask[:, :, None].expand_as(pred)
    se = (pred - y) ** 2 * m
    return se.sum() / (m.sum() + 1e-8)


def input_validity_np(v40):
    """(N,40) -> (N,NIN) float validity of the 16 INPUT channels.

    Used by the information-matched comparator arm. Validity is recorded per cycle, not per
    time point, so the flag is constant along the cycle, exactly as the diffusion model's own channel-level
    known mask is. Concatenating it doubles the input width to 32.
    """
    return v40[:, IN_CH].astype(np.float32)


def stack_input(xin, v40, seq, in_mask):
    """(N,1600) flat kinematics -> the tensor the model consumes, optionally with the validity flags."""
    n = xin.shape[0]
    if not seq:
        if in_mask:
            raise ValueError("the validity-mask arm requires seq=True")
        return xin
    X = xin.reshape(n, NT, NIN).transpose(0, 2, 1)               # (N,NIN,NT)
    if in_mask:
        V = np.repeat(input_validity_np(v40)[:, :, None], NT, axis=2)   # (N,NIN,NT), constant in time
        X = np.concatenate([X, V], axis=1)                       # (N,2*NIN,NT)
    return X


def assemble(data, seq, in_mask=False):
    Xs, Ys, Cs, Ms = [], [], [], []
    for cohort, stem, xin, yout, v40, cond in data:
        n = xin.shape[0]
        Xs.append(stack_input(xin, v40, seq, in_mask))
        Ys.append(yout); Cs.append(np.tile(cond, (n, 1))); Ms.append(target_mask_np(v40, cohort))
    return (np.concatenate(Xs, 0), np.concatenate(Ys, 0),
            np.concatenate(Cs, 0), np.concatenate(Ms, 0))


def train_model(model, data, epochs=EPOCHS, bs=BS, lr=LR, seq=False, in_mask=False):
    model = model.to(DEVICE)
    Xn, Yn, Cn, Mn = assemble(data, seq, in_mask)
    X = torch.tensor(Xn, dtype=torch.float32); Y = torch.tensor(Yn, dtype=torch.float32)
    C = torch.tensor(Cn, dtype=torch.long); M = torch.tensor(Mn, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = X.shape[0]; model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = X[idx].to(DEVICE); yb = Y[idx].to(DEVICE); cb = C[idx].to(DEVICE); mb = M[idx].to(DEVICE)
            opt.zero_grad()
            loss = masked_mse(model(xb, cb), yb, mb)
            loss.backward(); opt.step()
        sched.step()
    return model


def scored_side_channels(cohort, vgrf_ch):
    """The channels a limb-cycle is scored on, IDENTICAL to the diffusion protocol in
    eval_kinetics_all_cycles.py lines 70-71. bmclab_pd carries vertical GRF only: its sagittal-moment cells
    are marked valid in the cycle mask but are not part of the evaluated target set for that cohort, so the
    diffusion eval drops them by an explicit cohort exception. The baselines MUST use the same exception or
    the two are scored on different channel sets and the cohort means are not comparable.
    Without it the cohort would be averaged over four channels on one side and one on the other."""
    mom = [10, 16, 18] if vgrf_ch == 22 else [13, 17, 19]
    return [vgrf_ch] + ([] if cohort == "bmclab_pd" else mom)


def eval_fold(models, va, seq=False, in_mask=False):
    """subject-macro Method B: score a limb-cycle only if that side's evaluated targets are all valid.

    `models` is the seed ensemble. Predictions are averaged across seeds BEFORE scoring, the same way the
    diffusion evaluator averages its noise draws before scoring. Note the two ensembles are matched in
    count but not in kind: these are independently trained networks, whereas the diffusion draws are
    repeated sampling of one checkpoint.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with torch.no_grad():
        for cohort, stem, xin, yout, v40, cond in va:
            n = xin.shape[0]
            xn = stack_input(xin, v40, seq, in_mask)
            xb = torch.tensor(xn, dtype=torch.float32).to(DEVICE)
            cb = torch.tensor(np.tile(cond, (n, 1)), dtype=torch.long).to(DEVICE)
            pred = np.mean([m(xb, cb).cpu().numpy() for m in models], axis=0)
            for ci in range(n):
                for vg in (22, 25):
                    side_t = scored_side_channels(cohort, vg)
                    if not all(v40[ci, c] for c in side_t): continue
                    for c in side_t:
                        ti = TGT.index(c)
                        r = safe_pcc(yout[ci, ti, :], pred[ci, ti, :])
                        if np.isfinite(r): acc[cohort][MERGED[c]][stem].append(r)
    return acc


def subject_macro(acc_all):
    """Also returns per-subject overall scores (mean over that subject's channel means), the same
    quantity the diffusion eval stores as per_subject_overall, so the two can be paired by subject."""
    per_cohort = {}; per_subject = {}
    for cohort in acc_all:
        ch_means = {}; subj_ch = defaultdict(dict)
        for chm, sd in acc_all[cohort].items():
            ps = []
            for stem, v in sd.items():
                if not v: continue
                m = float(np.mean(v)); ps.append(m); subj_ch[stem][chm] = m
            if ps: ch_means[chm] = round(float(np.mean(ps)), 4)
        per_cohort[cohort] = ch_means
        per_subject[cohort] = {st: round(float(np.mean(list(cm.values()))), 6) for st, cm in subj_ch.items() if cm}
    cohort_overall = {c: float(np.mean(list(v.values()))) for c, v in per_cohort.items() if v}
    macro = round(float(np.mean(list(cohort_overall.values()))), 4) if cohort_overall else None
    return macro, per_cohort, {c: round(v, 4) for c, v in cohort_overall.items()}, per_subject


def diffusion_reference(path):
    """CV5 subject-macro headline of the diffusion model, read from the JSON written by
    eval_kinetics_all_cycles.py; None when that file has not been produced yet."""
    path = Path(path)
    if not path.is_file():
        return None
    d = json.load(open(path, encoding="utf-8"))
    csm = d["pretrain/CV5"]["cohort_subject_macro"]
    return round(float(np.mean([v["pcc"] for v in csm.values()])), 4)


def run(build_model, seq, transform, sev, in_mask=False):
    """train + eval one architecture across CV5, return the subject-macro summary + param count.

    in_mask selects the information-matched arm, in which the 16 per-cycle input-validity flags are
    concatenated to the 16 kinematic channels. Without it a missing channel reaches the comparator as a
    zero it cannot distinguish from a measured one, while the diffusion model receives a learned mask
    token and an explicit unknown flag.
    """
    preflight_inputs(sev)
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nparam = None; t0 = time.time()
    for fold in CV5:
        tr_subj, va_subj = get_split_subjects(get_split(), fold)
        tr = load_cycles(transform, sev, tr_subj); va = load_cycles(transform, sev, va_subj)
        if not tr or not va:
            raise ValueError(f"Comparator {fold} has empty training or validation data")
        models = []
        for so in range(N_SEED):
            torch.manual_seed(so * 7919); np.random.seed(so * 7919)
            m = build_model()
            if nparam is None: nparam = sum(p.numel() for p in m.parameters())
            models.append(train_model(m, tr, seq=seq, in_mask=in_mask))
        acc = eval_fold(models, va, seq=seq, in_mask=in_mask)
        for c in acc:
            for chm in acc[c]:
                for st, vals in acc[c][chm].items(): merged[c][chm][st].extend(vals)
    macro, per_cohort, cohort_overall, per_subject = subject_macro(merged)
    return {"subject_macro_PCC": macro, "per_cohort_overall": cohort_overall, "per_cohort": per_cohort,
            "per_subject_overall": per_subject,
            "n_params": int(nparam) if nparam else None, "n_seed_ensemble": N_SEED,
            "elapsed_s": round(time.time() - t0, 1)}


def build_transform():
    meta = V4SubjectMetadata()
    return per_subject_transform(meta, require_tables=True)


# No main(). This module is the shared fair-comparison PIPELINE (split, cycle loading, masked loss
# with the Parkinson moment exclusion, conditioning, three-seed training, subject-macro scoring)
# imported by published_comparators.py.
# The comparator set is restricted to published models designed for this task, following the
# criterion GaitDynamics states for its own comparison, so no generic architecture is included.
