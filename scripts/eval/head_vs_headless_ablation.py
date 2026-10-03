"""HEAD-vs-HEADLESS ablation (the reviewer's must-answer for the head-free framing).

Question: does adding a SUPERVISED refinement head beat the head-free masked in-painting recovery of the
kinetic channels? The arms differ in readout AND information source, so read them accordingly. Heads are
fit with a cell-level target-validity MASK in the loss (invalid sides are never supervised as nan->0):

  HEADLESS (ours)      : masked in-painting of the kinetic target channels THROUGH the frozen backbone
                         (no regression head). Single-seed to match the deterministic heads (also 3-seed TSA).
  A1 (direct head)     : a supervised kinematics-to-kinetics head trained DIRECTLY on the visible kinematic
                         channels; it does NOT pass through the frozen backbone (a GaitDynamics-style literal
                         direct-regression reference).
  A2 (frozen-backbone- : a supervised head trained on the PENULTIMATE DiT features h (B,25,384) tapped from a
      feature head)      clean masked forward through the FROZEN backbone.
  Only A2 vs HEADLESS is a same-backbone readout comparison (both read the frozen backbone); A1 is the
  no-backbone direct-regression reference.

Fairness (see scoping): identical CV5 folds, identical V3 per-subject transform, identical Method-B
valid-cycle filter + PD-moment masking, identical subject-macro 8-channel PCC, head fit on TRAIN
subjects only per fold. Backbone is FROZEN; Stage-1 is NOT touched; reuse expA_dynpool CV5 ckpts.

Writes outputs/evaluation/head_vs_headless_ablation.json. ASCII-only stdout; JSON dumped before any pretty print.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from guarded_write import guarded_dump

from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS, COH, EXCLUDE, GRFFIX, load_split  # noqa: E402
from drop_channels import DROP_16CH  # noqa: E402
from eval_common import (  # noqa: E402
    load_model, safe_pcc, get_val_pairs, load_valid_mask_40, source_type_and_name,
    SOURCE_TO_ID, SEED, SEVERITY_JSON, build_known_mask, KNOWN_CH, load_local_json,
    per_subject_transform,
)
from pathogait.data.subject_metadata import V4SubjectMetadata  # noqa: E402
from pathogait.data.pelvis_contract import validate_bmclab_input  # noqa: E402
from pathogait.diffusion.ddpm import DDPMScheduler  # noqa: E402

CV5 = [f"cv5_{i}" for i in range(5)]
TGT = [10, 13, 16, 17, 18, 19, 22, 25]
# The Parkinson source carries no trustworthy joint-moment label, so the diffusion trainer removes
# these cells from supervision (train_finetune.EXCLUDED_T1_CHANNELS_PER_COHORT) and the published
# comparators do the same (baseline_nondiffusion.target_mask_np). The heads have to be supervised on
# the same cells, or the arms differ in their training targets and not only in their readout.
PD_MOMENT_COHORT = "bmclab_pd"
PD_MOMENT_TGT = [10, 13, 16, 17, 18, 19]
MERGED = {10: "Hip_Mom", 13: "Hip_Mom", 16: "Knee_Mom", 17: "Knee_Mom",
          18: "Ankle_Mom", 19: "Ankle_Mom", 22: "GRF_V", 25: "GRF_V"}
BASE_DROP = list(DROP_16CH)
# Visible kinematic input channels the head (A1) is allowed to see. This MUST be the same input regime the
# head-free path receives, namely eval_common.KNOWN_CH minus the dropped set, which resolves to the ten
# sagittal joint angles plus the six pelvis channels.
# This must be an explicit list, not the set complement 40ch minus BASE_DROP minus TGT, which leaves the
# eight kinetic targets but silently LEAVES IN the horizontal ground-reaction-force channels 20, 21, 23 and
# 24 (R/L GRF_X and GRF_Y). Those are force-plate quantities, not kinematics, and they are not in KNOWN_CH,
# so the head-free path never sees them. A1 was therefore given measured shear force that its comparator did
# not have. Deriving the list from KNOWN_CH makes A1 a genuine kinematics-only regressor and matches
# baseline_nondiffusion.IN_CH exactly. Only A1 was affected: A2 taps a forward built by masked_input, which
# already goes through build_known_mask, and the reported estimation results use tweedie_inpaint_drop, which
# does the same.
VISIBLE_IN = [c for c in KNOWN_CH if c not in set(BASE_DROP) and c not in set(TGT)]
BATCH = 32
T_FEATURE = 20   # small timestep for the clean-ish feature/inpaint forward used to tap A2 features
HEAD_EPOCHS = 200
HEAD_LR = 2e-3


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


class SmallHead(nn.Module):
    """~0.2M-param 1D conv head: (B, C_in, T=100) -> (B, 8, 100). Matched-small, regularized."""
    def __init__(self, c_in, c_out=8, hidden=96, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, 5, padding=2), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(hidden, c_out, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


def tap_penultimate(model, x, t, sid, cid, svid):
    """Run the DiT forward but return the penultimate hidden h (B, n_patches, hidden), upsampled to
    (B, hidden, T) by nearest patch repeat so a conv head can read it at waveform resolution."""
    c = model.get_conditioning(t, sid, cid, svid)
    h = model.patchify(x)
    for blk in model.blocks:
        h = blk(h, c)
    # h: (B, 25, 384) -> (B, 384, 100) by repeating each patch's vector over its 4 timesteps
    B, P, H = h.shape
    patch = x.shape[-1] // P
    hf = h.transpose(1, 2).unsqueeze(-1).expand(B, H, P, patch).reshape(B, H, P * patch)
    return hf


def masked_input(gt, valid, imt, drop_ch, device):
    """Build the model input used for a clean feature/inpaint forward: clamp visible channels to GT,
    set target+dropped channels to the input-mask token (same as the in-painting known-mask path)."""
    km = build_known_mask(gt, valid).clone()          # (B,40,T) 1=known/visible
    km[:, TGT, :] = 0.0                                # targets are hidden
    if drop_ch:
        km[:, drop_ch, :] = 0.0                        # dropped inputs hidden
    x = km * torch.nan_to_num(gt, nan=0.0)
    if imt is not None:
        x = x + (1.0 - km) * imt.to(device)           # fill hidden with the learned mask token
    return x, km

SPLIT = None  # the five-fold split, loaded in main()


def gather_fold(model, sched, imt, transform, sev, fold, device):
    """Iterate the fold's val pairs; for each valid cycle collect (features_A1, features_A2, target8,
    cohort, stem). Returns lists. Uses a single clean forward at T_FEATURE for A2 features."""
    rows = []
    t = torch.full((1,), T_FEATURE, dtype=torch.long, device=device)
    for src, stem in get_val_pairs(SPLIT, fold):
        if stem in EXCLUDE:
            continue
        rp = GRFFIX / src / f"{stem}.npy"
        if not rp.exists():
            continue
        s = sev["mappings"].get(src, {}).get(stem)
        if s is None:
            continue
        validate_bmclab_input(rp, transform.bmclab_pelvis_representation)
        try:
            raw = np.load(str(rp))
            v40 = load_valid_mask_40(rp, raw.shape[0])
            st, sn = source_type_and_name(src)
            x40 = transform(raw, sn, st, stem)[:, :100, :].astype(np.float32)
            n = min(v40.shape[0], x40.shape[0])
            x40, v40 = x40[:n], v40[:n]
        except Exception as exc:
            print("  SKIPPED %s: %s" % (stem, type(exc).__name__), flush=True)
            continue
        cohort = COH.get(int(s["cohort_id"]), "other")
        R = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18])
        L = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
        keep = [ci for ci in range(x40.shape[0])
                if (v40[ci, 22] and all(v40[ci, c] for c in R)) or (v40[ci, 25] and all(v40[ci, c] for c in L))]
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
            x_in, _km = masked_input(gt, valid, imt, BASE_DROP, device)
            with torch.no_grad():
                feat_a2 = tap_penultimate(model, x_in, t.expand(nb), sid.expand(nb).to(device),
                                          cid.expand(nb).to(device), svid.expand(nb).to(device))
            feat_a1 = gt[:, VISIBLE_IN, :]              # A1 head sees only visible kinematics
            tgt = gt[:, TGT, :]                         # 8 target channels (z-scored space)
            vm = v40[bidx]
            feat_a1 = feat_a1.cpu().numpy()
            feat_a2_np = feat_a2.cpu().numpy()
            tgt = tgt.cpu().numpy()
            for ci in range(nb):
                rows.append({
                    "a1": feat_a1[ci],
                    "a2": feat_a2_np[ci],
                    "tgt": tgt[ci],
                    "cohort": cohort, "stem": stem, "vm": vm[ci].copy(),
                })
    return rows


def target_mask_row(r):
    """Cell-level target validity for one cycle, with the diffusion trainer's cohort exclusion."""
    m = r["vm"][TGT].astype(np.float32)
    if r["cohort"] == PD_MOMENT_COHORT:
        for ch in PD_MOMENT_TGT:
            m[TGT.index(ch)] = 0.0
    return m


def train_head(train_rows, key, c_in, device):
    set_seed(SEED)
    head = SmallHead(c_in).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR, weight_decay=1e-4)
    X = torch.from_numpy(np.stack([r[key] for r in train_rows])).float().to(device)
    Y = torch.from_numpy(np.stack([r["tgt"] for r in train_rows])).float().to(device)
    TM = torch.from_numpy(np.stack([target_mask_row(r) for r in train_rows])).float().to(device)  # (N,8) cell-level target validity
    head.train()
    n = X.shape[0]
    for ep in range(HEAD_EPOCHS):
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, 64):
            bi = perm[b0:b0 + 64]
            opt.zero_grad()
            pred = head(X[bi])
            m = TM[bi][:, :, None]                                                     # mask invalid target cells out of the loss
            loss = ((pred - Y[bi]) ** 2 * m).sum() / (m.expand_as(pred).sum() + 1e-8)  # mean over VALID target cells only (not nan->0)
            loss.backward()
            opt.step()
    head.eval()
    return head


def score_head(head, val_rows, key, device):
    """subject-macro 8ch PCC with PD-moment masking, mirroring partial_input_masking summarize."""
    subj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cohort->mc->stem->[pcc]
    with torch.no_grad():
        for r in val_rows:
            x = torch.from_numpy(r[key]).float().unsqueeze(0).to(device)
            pred = head(x)[0].cpu().numpy()   # (8,100)
            vm = r["vm"]
            cohort = r["cohort"]
            R = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18])
            L = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
            for side, vg in [(R, 22), (L, 25)]:
                if not vm[vg] or not all(vm[c] for c in side):
                    continue
                for ch in side:
                    j = TGT.index(ch)
                    rr = safe_pcc(r["tgt"][j], pred[j])
                    if np.isfinite(rr):
                        subj[cohort][MERGED[ch]][r["stem"]].append(rr)
    return subject_macro(subj) + (subj,)


def subject_macro(subj_scores):
    per_cohort = {}
    per_cohort_ch = {}
    for cohort in subj_scores:
        chm = {}
        for mc, sd in subj_scores[cohort].items():
            ps = [float(np.mean(v)) for v in sd.values() if v]
            if ps:
                chm[mc] = float(np.mean(ps))
        if chm:
            per_cohort[cohort] = float(np.mean(list(chm.values())))
            per_cohort_ch[cohort] = {k: round(v, 4) for k, v in chm.items()}
    macro = round(float(np.mean(list(per_cohort.values()))), 4) if per_cohort else None
    return macro, {c: round(v, 4) for c, v in per_cohort.items()}, per_cohort_ch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("HEAD-vs-HEADLESS ablation | device", device, flush=True)
    global SPLIT
    SPLIT = load_split()
    meta = V4SubjectMetadata()
    transform = per_subject_transform(meta, require_tables=True)
    sev = load_local_json(SEVERITY_JSON, "severity table")
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

    # accumulate subject-macro across folds for each arm
    agg = {"A1": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
           "A2": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
           "headless_1seed": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
           "headless_3seed": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))}

    for fold in CV5:
        cp = ROOT / "checkpoints" / f"v4_stage2_expA_dynpool_{fold}" / "final.pt"
        if not cp.exists():
            print("  MISSING ckpt", cp, flush=True)
            continue
        model, ckpt, _, imt = load_model(cp, device)
        model.eval()
        if isinstance(ckpt, dict):
            _i = ckpt.get("input_mask_token_ema")
            if _i is None:
                _i = ckpt.get("input_mask_token")
            if _i is not None:
                imt = _i
        # TRAIN rows for this fold = all OTHER folds' val pairs (they were in this fold's training set).
        train_rows = []
        for tf in CV5:
            if tf == fold:
                continue
            train_rows += gather_fold(model, sched, imt, transform, sev, tf, device)
        val_rows = gather_fold(model, sched, imt, transform, sev, fold, device)
        print(f"  {fold}: train_cycles={len(train_rows)} val_cycles={len(val_rows)}", flush=True)
        if not train_rows or not val_rows:
            del model
            continue

        # A1 + A2 heads
        c_in_a1 = len(VISIBLE_IN)
        c_in_a2 = val_rows[0]["a2"].shape[0]
        head_a1 = train_head(train_rows, "a1", c_in_a1, device)
        head_a2 = train_head(train_rows, "a2", c_in_a2, device)
        for arm, head, key in [("A1", head_a1, "a1"), ("A2", head_a2, "a2")]:
            _m, _c, _cc, _subj = score_head(head, val_rows, key, device)
            for cohort in _subj:
                for mc, stemd in _subj[cohort].items():
                    for stem, vals in stemd.items():
                        agg[arm][cohort][mc][stem].extend(vals)

        # HEADLESS in-painting over the SAME val cycles (single-seed + 3-seed)
        _score_headless_inline(model, sched, imt, transform, sev, fold, device, agg)
        del model, head_a1, head_a2

    # summarize each arm
    out = {}
    for arm in agg:
        macro, per_cohort, per_cohort_ch = _summarize_agg(agg[arm])
        out[arm] = {"subject_macro_PCC": macro, "per_cohort": per_cohort, "per_cohort_channel": per_cohort_ch,
                    "per_subject_overall": _per_subject_overall(agg[arm])}
    # headline deltas
    if out["A2"]["subject_macro_PCC"] is not None and out["headless_1seed"]["subject_macro_PCC"] is not None:
        out["delta_A2_minus_headless1seed"] = round(
            out["A2"]["subject_macro_PCC"] - out["headless_1seed"]["subject_macro_PCC"], 4)
        out["delta_A1_minus_headless1seed"] = round(
            out["A1"]["subject_macro_PCC"] - out["headless_1seed"]["subject_macro_PCC"], 4)
    saved_path = guarded_dump(out, ROOT / "outputs/evaluation/head_vs_headless_ablation.json", "head_vs_headless_ablation.py")
    print("\n=== HEAD vs HEADLESS (CV5 subject-macro 8ch PCC) ===", flush=True)
    for arm in ["headless_1seed", "headless_3seed", "A1", "A2"]:
        print(f"  {arm:16s} macro = {out[arm]['subject_macro_PCC']}  per-cohort {out[arm]['per_cohort']}", flush=True)
    if "delta_A2_minus_headless1seed" in out:
        print(f"  A2 - headless(1seed) = {out['delta_A2_minus_headless1seed']:+.4f}", flush=True)
        print(f"  A1 - headless(1seed) = {out['delta_A1_minus_headless1seed']:+.4f}", flush=True)
    print("wrote", saved_path, flush=True)


def _per_subject_overall(arm_agg):
    """cohort -> stem -> scalar, averaging each subject over its own channels.

    Subject-first, matching the per_subject_overall block written by published_comparators.py that
    model_comparison_paired.py consumes. _summarize_agg below is channel-first and the two orders do
    not generally agree, so the paired test must read this block and not that one.
    """
    out = {}
    for cohort in arm_agg:
        stems = {}
        for mc, stemd in arm_agg[cohort].items():
            for stem, vals in stemd.items():
                if vals:
                    stems.setdefault(stem, []).append(float(np.mean(vals)))
        out[cohort] = {s: float(np.mean(v)) for s, v in stems.items() if v}
    return out


def _summarize_agg(arm_agg):
    """Pool subjects across folds, the aggregation eval_kinetics_all_cycles.py uses for the headline.

    Each fold contributes its own subjects, keyed by stem, and every subject is counted once. Taking a
    mean over folds instead would weight a fold holding three subjects the same as one holding seven,
    which is what made this table disagree with Figure 2 on the per-cohort cells.
    """
    per_cohort = {}
    per_cohort_ch = {}
    for cohort in arm_agg:
        chm = {}
        for mc, stemd in arm_agg[cohort].items():
            ps = [float(np.mean(v)) for v in stemd.values() if v]
            if ps:
                chm[mc] = float(np.mean(ps))
        if chm:
            per_cohort[cohort] = round(float(np.mean(list(chm.values()))), 4)
            per_cohort_ch[cohort] = {k: round(v, 4) for k, v in chm.items()}
    macro = round(float(np.mean(list(per_cohort.values()))), 4) if per_cohort else None
    return macro, per_cohort, per_cohort_ch


def _score_headless_inline(model, sched, imt, transform, sev, fold, device, agg):
    """Head-free in-painting over this fold's val cycles: single-seed and 3-seed TSA, subject-macro."""
    def run(seeds):
        subj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for src, stem in get_val_pairs(SPLIT, fold):
            if stem in EXCLUDE:
                continue
            rp = GRFFIX / src / f"{stem}.npy"
            if not rp.exists():
                continue
            s = sev["mappings"].get(src, {}).get(stem)
            if s is None:
                continue
            validate_bmclab_input(rp, transform.bmclab_pelvis_representation)
            try:
                raw = np.load(str(rp))
                v40 = load_valid_mask_40(rp, raw.shape[0])
                st, sn = source_type_and_name(src)
                x40 = transform(raw, sn, st, stem)[:, :100, :].astype(np.float32)
                n = min(v40.shape[0], x40.shape[0])
                x40, v40 = x40[:n], v40[:n]
            except Exception as exc:
                print("  SKIPPED %s: %s" % (stem, type(exc).__name__), flush=True)
                continue
            cohort = COH.get(int(s["cohort_id"]), "other")
            R = [22] + ([] if cohort == "bmclab_pd" else [10, 16, 18])
            L = [25] + ([] if cohort == "bmclab_pd" else [13, 17, 19])
            keep = [ci for ci in range(x40.shape[0])
                    if (v40[ci, 22] and all(v40[ci, c] for c in R)) or (v40[ci, 25] and all(v40[ci, c] for c in L))]
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
                gens = []
                for so in range(seeds):
                    torch.manual_seed(SEED + so * 7919)
                    with torch.no_grad():
                        gg = tweedie_inpaint_drop(model, sched, gt, valid,
                                                  sid.expand(nb).to(device), cid.expand(nb).to(device),
                                                  svid.expand(nb).to(device), 50, 1.0, device, imt,
                                                  CHECKPOINT_STEPS, BASE_DROP)
                        if torch.is_tensor(gg):
                            gg = gg.detach().cpu().numpy()
                        gens.append(gg)
                g = np.mean(np.stack(gens), axis=0)
                km = build_known_mask(gt, valid)
                g = (km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * torch.from_numpy(g).to(device)).cpu().numpy()
                gtn = gt.cpu().numpy()
                vm = v40[bidx]
                for ci in range(nb):
                    for side, vg in [(R, 22), (L, 25)]:
                        if not vm[ci, vg] or not all(vm[ci, c] for c in side):
                            continue
                        for ch in side:
                            rr = safe_pcc(gtn[ci, ch, :], g[ci, ch, :])
                            if np.isfinite(rr):
                                subj[cohort][MERGED[ch]][stem].append(rr)
        return subj

    for seeds, arm in [(1, "headless_1seed"), (3, "headless_3seed")]:
        subj = run(seeds)
        for cohort in subj:
            for mc, stemd in subj[cohort].items():
                for stem, vals in stemd.items():
                    agg[arm][cohort][mc][stem].extend(vals)


if __name__ == "__main__":
    main()
