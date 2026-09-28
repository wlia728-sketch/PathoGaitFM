"""IMU input-modality ZERO-SHOT eval on the Grouvel 2023 dataset (never in training).

CLONE of eval_opencap_zeroshot.py (verbatim inference + subject-clustered bootstrap);
ONLY the source of joint angles differs (marker-IK vs IMU-IK). Scores vertical GRF.
branch in {marker, imu}. Runs with torch+CUDA python (Windows). ASCII prints, json first.

marker branch = positive control (must land ~0.85-0.95, cf OpenCap 0.901).
imu branch    = the experiment (leakage-clean IMU-signal-only orientation).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from site_summary import load_site_summary, site_mass
from guarded_write import guarded_dump
from tsa_inference import tweedie_inpaint_drop, CHECKPOINT_STEPS
from drop_channels import DROP_16CH
from eval_common import load_model, safe_pcc, SEED, build_known_mask, per_subject_transform
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

CKPT = ROOT / "checkpoints" / "v4_stage2_final8ch_ALLDATA" / "final.pt"
PROC = ROOT / "data" / "external" / "grouvel_imu" / "processed"
BRANCH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("marker", "imu") else "marker"
AXES = {"GRF_V": (22, 25)}
STANCE = slice(0, 61)
BATCH = 32  # cycles per inference batch (bound GPU memory); EVERY valid cycle is scored


def load_npy_54(f, transform, sname):
    raw = np.load(str(f))
    stem = f.stem
    a40 = transform(raw.copy(), sname, "cohort", stem)[:, :100, :].astype(np.float32)
    m = np.load(str(f).replace(".npy", "_mask.npy"))
    m40 = m[:, KEEP_CHANNELS_54TO40].astype(bool) if m.shape[-1] >= 54 else m.astype(bool)
    n = min(a40.shape[0], m40.shape[0])
    return a40[:n], m40[:n]


def gen_tsa(model, sched, gt, valid, sid, cid, svid, imt, device):
    gens = []
    for so in range(3):
        torch.manual_seed(SEED + so * 7919)
        with torch.no_grad():
            gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid, cid, svid,
                                             50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
    g = torch.stack(gens).mean(0)
    km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
    return g.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only = sys.argv[2:]
    print("IMU-modality zero-shot (vertical GRF, ALL-DATA 8ch) branch=%s | device %s" % (BRANCH, device), flush=True)
    if not CKPT.exists():
        raise FileNotFoundError(f"MODEL MISSING: {CKPT}")
    model, ckpt, _, imt = load_model(CKPT, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None:
            _i = ckpt.get("input_mask_token")
        if _i is not None:
            imt = _i
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

    meta = V4SubjectMetadata()
    summ_p = PROC / ("_summary_%s.json" % BRANCH)
    if summ_p.exists():
        for d in load_site_summary(summ_p):
            if d.get("subj"):
                _m = site_mass(d, summ_p)
                meta.bw_table[("ext_imu", d["subj"])] = _m
                meta.bw_table[("ext_imu", d["subj"] + "__" + BRANCH)] = _m
    transform = per_subject_transform(meta, require_tables=False)

    files = [f for f in sorted(PROC.glob("*__%s.npy" % BRANCH)) if not f.name.endswith("_mask.npy")]
    if only:
        files = [f for f in files if f.stem.split("__")[0] in only]
    res = {ax: {"full": [], "stance": []} for ax in AXES}
    persubj = set(); persubj_vgrf = {}
    for f in files:
        stem = f.stem
        try:
            x40, m40 = load_npy_54(f, transform, "ext_imu")
        except Exception as e:
            print(f"  [{stem}] load FAIL {type(e).__name__}: {e}"); continue
        R_V, L_V = AXES["GRF_V"]
        keep = [ci for ci in range(x40.shape[0]) if m40[ci, R_V] or m40[ci, L_V]]
        if not keep:
            print(f"  [{stem}] no valid vGRF cycles"); continue
        keep = np.array(keep)
        # score EVERY valid cycle of the subject (no subsample); batch to bound GPU memory
        used = False
        for b0 in range(0, len(keep), BATCH):
            bidx = keep[b0:b0 + BATCH]
            nb = len(bidx)
            gt = torch.from_numpy(np.nan_to_num(x40[bidx].transpose(0, 2, 1), nan=0.0)).to(device)
            valid = torch.from_numpy(m40[bidx]).to(device)
            sid = torch.full((nb,), model.num_sources, dtype=torch.long, device=device)  # reserved unknown-external source id
            cid = torch.ones(nb, dtype=torch.long, device=device)   # healthy
            svid = torch.full((nb,), 5, dtype=torch.long, device=device)   # severity_id 5, the only bin the healthy cohort occurs with in training
            g = gen_tsa(model, sched, gt, valid, sid, cid, svid, imt, device)
            gtn = gt.cpu().numpy(); vm = m40[bidx]
            for ci in range(nb):
                for ax, (rch, lch) in AXES.items():
                    for ch in (rch, lch):
                        if not vm[ci, ch]:
                            continue
                        rf = safe_pcc(gtn[ci, ch, :], g[ci, ch, :])
                        rs = safe_pcc(gtn[ci, ch, STANCE], g[ci, ch, STANCE])
                        if np.isfinite(rf):
                            res[ax]["full"].append(rf); used = True
                            if ax == "GRF_V":
                                persubj_vgrf.setdefault(stem, []).append(rf)
                        if np.isfinite(rs):
                            res[ax]["stance"].append(rs)
        if used:
            persubj.add(stem)

    out = {"branch": BRANCH, "n_subj": len(persubj), "subjects": sorted(persubj), "per_axis": {}}
    for ax in AXES:
        fv, sv = res[ax]["full"], res[ax]["stance"]
        out["per_axis"][ax] = {
            "pcc_full": round(float(np.mean(fv)), 3) if fv else None,
            "pcc_stance": round(float(np.mean(sv)), 3) if sv else None,
            "n_cyc": len(fv)}
    subj_means = {s: float(np.mean(v)) for s, v in persubj_vgrf.items() if v}
    smv = np.array(list(subj_means.values()))
    macro = float(np.mean(smv)) if len(smv) else None
    rng2 = np.random.default_rng(SEED)
    boots = []
    for _ in range(5000):
        idx = rng2.integers(0, len(smv), len(smv))
        boots.append(np.mean(smv[idx]))
    ci = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)] if len(smv) else None
    out["vgrf_subject_macro"] = round(macro, 3) if macro is not None else None
    out["vgrf_subject_macro_ci95"] = ci
    out["vgrf_per_subject"] = {s: round(m, 3) for s, m in subj_means.items()}
    print("\n  *** vGRF SUBJECT-MACRO = %s  95%% CI %s  (n_subj=%d) ***" % (out["vgrf_subject_macro"], ci, len(smv)), flush=True)
    saved_path = guarded_dump(out, ROOT / ("outputs/evaluation/imu_zeroshot_%s.json" % BRANCH), "eval_imu_zeroshot.py")
    print("\n=== IMU-MODALITY ZERO-SHOT (healthy, %s input, TSA) ===" % BRANCH, flush=True)
    print("  n_subj=%d  subjects=%s" % (out["n_subj"], out["subjects"]), flush=True)
    for ax in AXES:
        m = out["per_axis"][ax]
        print("  %-7s full=%s  stance=%s  (n_cyc=%d)" % (ax, m["pcc_full"], m["pcc_stance"], m["n_cyc"]), flush=True)
    print("\nsaved", saved_path, flush=True)


if __name__ == "__main__":
    main()
