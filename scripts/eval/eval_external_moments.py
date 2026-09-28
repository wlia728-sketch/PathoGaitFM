"""External JOINT-MOMENT zero-shot eval, matching the canonical crouch/vGRF pipeline EXACTLY
(same ALLDATA checkpoint + tweedie_inpaint_drop 3-seed + V4LazyTransformV3PerSubject + v4 zstats),
so the moment numbers are directly comparable to the published external vGRF results.

Sites with dataset-provided moment ground truth (NOT our own ID):
  - fukuchi (healthy, WBDS/Visual3D moments)   source='fukuchi',  cohort_id=1, severity_id=5

Sagittal moment target channels (40-ch): R/L Hip=10/13, Knee=16/17, Ankle=18/19.
Only cycles whose moment channel is mask-valid are scored. Subject-macro PCC (per-cycle PCC
averaged within subject, then across subjects) + RMSE denormalised to Nm/kg and range-normalised nRMSE,
all aggregated subject-first on the same scored cycles, with subject-clustered bootstrap 95% intervals.
Full-cycle AND stance-window (0-60%). R/L merged per joint.

Writes outputs/evaluation/external_moments.json. ASCII-only stdout.
"""
import sys, json
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
from eval_common import load_model, safe_pcc, ZSTATS_PATH, SEED, build_known_mask, per_subject_transform
from pathogait.data.channels import KEEP_CHANNELS_54TO40
from pathogait.data.subject_metadata import V4SubjectMetadata
from pathogait.diffusion.ddpm import DDPMScheduler

CKPT = ROOT / "checkpoints" / "v4_stage2_final8ch_ALLDATA" / "final.pt"

# 40-ch sagittal moment channels, grouped by joint (R,L)
MOM = {"Hip": (10, 13), "Knee": (16, 17), "Ankle": (18, 19)}
KEEP = list(KEEP_CHANNELS_54TO40)
INV40to54 = {i40: c54 for i40, c54 in enumerate(KEEP)}
FULL = slice(0, 100); STANCE = slice(0, 61)

SITES = {
    "fukuchi": dict(proc=ROOT / "data/external/fukuchi_healthy/processed",
                    source="fukuchi", cohort_id=1, severity_id=5,
                    summary="_harmonize_summary.json", sid_key="sid"),
}


def denorm(x_z, ch40, z):
    c54 = str(INV40to54[ch40])
    if c54 not in z:
        return x_z
    return x_z * (3.0 * z[c54]["std"]) + z[c54]["mean"]


ERR_CI_SEED = 20260726


def boot_ci(vals, nd=3, n_boot=5000):
    """Subject-clustered 95% CI: resample SUBJECTS with replacement. Own generator, seeded independently."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return None
    r = np.random.default_rng(ERR_CI_SEED)
    b = v[r.integers(0, v.size, (n_boot, v.size))].mean(1)
    return [round(float(np.percentile(b, 2.5)), nd), round(float(np.percentile(b, 97.5)), nd)]


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_masses(cfg):
    """Return {sid: mass} from the site summary json.

    Fails closed. Body weight normalises this site's targets, so a silently empty table would leave the
    comparison unnormalised and produce a wrong number rather than an error.
    """
    p = cfg["proc"] / cfg["summary"]
    if not p.exists():
        raise FileNotFoundError(
            "%s not found. Rebuild this site with its harmonisation script; see the site README." % p)
    table = {}
    for d in load_site_summary(p):
        if isinstance(d, dict) and d.get(cfg["sid_key"]):
            table[d[cfg["sid_key"]]] = site_mass(d, p)
    if not table:
        raise KeyError("%s carried no subject with a body mass." % p.name)
    return table


def eval_site(site, cfg, model, sched, z, device, imt):
    meta = V4SubjectMetadata()
    masses = load_masses(cfg)
    for sid, m in masses.items():
        meta.bw_table[(cfg["source"], sid)] = m
    transform = per_subject_transform(meta, require_tables=False)
    files = [f for f in sorted(cfg["proc"].glob("*.npy")) if not f.name.endswith("_mask.npy")]

    # per (joint, window): per-subject-mean PCC list, and pooled rmse
    acc = {j: {w: {"subjpcc": [], "rmse": [], "nrmse": [], "n_cyc": 0} for w in ("full", "stance")} for j in MOM}
    n_subj = 0
    for f in files:
        raw = np.load(str(f))
        a40 = transform(raw.copy(), cfg["source"], "cohort", f.stem)[:, :100, :].astype(np.float32)
        m = np.load(str(f).replace(".npy", "_mask.npy"))
        m40 = m[:, KEEP_CHANNELS_54TO40].astype(bool)
        n = min(a40.shape[0], m40.shape[0]); a40, m40 = a40[:n], m40[:n]
        # need at least one valid moment cycle
        anymom = np.zeros(n, bool)
        for (rch, lch) in MOM.values():
            anymom |= (m40[:, rch] | m40[:, lch])
        keep = np.where(anymom)[0]
        if not len(keep):
            continue
        gt = torch.from_numpy(np.nan_to_num(a40[keep].transpose(0, 2, 1), nan=0.0)).to(device)
        valid = torch.from_numpy(m40[keep]).to(device)
        nb = len(keep)
        sid_t = torch.full((nb,), model.num_sources, dtype=torch.long, device=device)  # reserved unknown-external source id
        cid_t = torch.full((nb,), cfg["cohort_id"], dtype=torch.long, device=device)
        svid_t = torch.full((nb,), cfg["severity_id"], dtype=torch.long, device=device)
        gens = []
        for so in range(3):
            torch.manual_seed(SEED + so * 7919)
            with torch.no_grad():
                gens.append(tweedie_inpaint_drop(model, sched, gt, valid, sid_t, cid_t, svid_t,
                                                 50, 1.0, device, imt, CHECKPOINT_STEPS, DROP_16CH))
        g = torch.stack(gens).mean(0)
        km = build_known_mask(gt, valid); g = km * torch.nan_to_num(gt, nan=0.0) + (1 - km) * g
        gtn = gt.cpu().numpy(); prn = g.cpu().numpy(); vm = m40[keep]

        subj_has = False
        for j, (rch, lch) in MOM.items():
            for wname, win in (("full", FULL), ("stance", STANCE)):
                cyc_pccs = []; cyc_rmse = []; cyc_nrmse = []
                for ci in range(nb):
                    for ch in (rch, lch):
                        if not vm[ci, ch]:
                            continue
                        gw_z = gtn[ci, ch, win]; pw_z = prn[ci, ch, win]
                        r = safe_pcc(gw_z, pw_z)
                        # The error must not be accumulated over cycles and averaged flat, because
                        # the correlation was averaged within subject first. Both are now collected on the
                        # same scored cycles and aggregated subject-first, so one row shares one rule.
                        if np.isfinite(r):
                            cyc_pccs.append(r)
                            gw = denorm(gw_z, ch, z); pw = denorm(pw_z, ch, z)
                            rm = rmse(gw, pw)
                            cyc_rmse.append(rm)
                            rng_gt = float(np.max(gw) - np.min(gw))
                            if rng_gt > 1e-9:
                                cyc_nrmse.append(100.0 * rm / rng_gt)
                            acc[j][wname]["n_cyc"] += 1
                if cyc_pccs:
                    acc[j][wname]["subjpcc"].append(float(np.mean(cyc_pccs)))  # subject-mean
                    if cyc_rmse: acc[j][wname]["rmse"].append(float(np.mean(cyc_rmse)))
                    if cyc_nrmse: acc[j][wname]["nrmse"].append(float(np.mean(cyc_nrmse)))
                    subj_has = True
        if subj_has:
            n_subj += 1

    out = {"n_subj": n_subj, "per_joint": {}}
    for j in MOM:
        out["per_joint"][j] = {}
        for w in ("full", "stance"):
            d = acc[j][w]
            pcc = float(np.mean(d["subjpcc"])) if d["subjpcc"] else None
            rm = float(np.mean(d["rmse"])) if d["rmse"] else None
            nrm = float(np.mean(d["nrmse"])) if d["nrmse"] else None
            out["per_joint"][j][w] = {
                "PCC_subject_macro": round(pcc, 3) if pcc is not None else None,
                "PCC_ci95": boot_ci(d["subjpcc"], nd=3),
                # RENAMED 2026-07-26: the previous key was "RMSE_norm", but denorm() is applied before the
                # error is computed, so the value has always been in Nm/kg. Only the label was wrong.
                "RMSE_Nm_per_kg": round(rm, 4) if rm is not None else None,
                "RMSE_ci95": boot_ci(d["rmse"], nd=4),
                "nRMSE_pct": round(nrm, 1) if nrm is not None else None,
                "nRMSE_pct_ci95": boot_ci(d["nrmse"], nd=1),
                "n_subj_scored": len(d["subjpcc"]),
                "n_cyc": d["n_cyc"],
            }
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = json.load(open(ZSTATS_PATH))
    model, ckpt, _, imt = load_model(CKPT, device); model.eval()
    if isinstance(ckpt, dict):
        _i = ckpt.get("input_mask_token_ema")
        if _i is None:
            _i = ckpt.get("input_mask_token")
        if _i is not None:
            imt = _i
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="cosine").to(device)

    results = {}
    for site, cfg in SITES.items():
        print(f"\n=== {site} ===", flush=True)
        results[site] = eval_site(site, cfg, model, sched, z, device, imt)
        r = results[site]
        print(f"  n_subj={r['n_subj']}", flush=True)
        print("  joint  | full PCC (n_subj) | stance PCC", flush=True)
        for j in MOM:
            fu = r["per_joint"][j]["full"]; st = r["per_joint"][j]["stance"]
            print(f"  {j:6s} | {str(fu['PCC_subject_macro']):>6s} (n={fu['n_subj_scored']:>2d}) "
                  f"| {str(st['PCC_subject_macro']):>6s}", flush=True)

    saved_path = guarded_dump({"checkpoint": "v4_stage2_final8ch_ALLDATA", "note":
                  "dataset-provided moments (not our ID); subject-macro PCC; same pipeline as crouch/vGRF eval",
                  "sites": results}, ROOT / "outputs/evaluation/external_moments.json", "eval_external_moments.py")
    print("\nsaved", saved_path, flush=True)


if __name__ == "__main__":
    main()
