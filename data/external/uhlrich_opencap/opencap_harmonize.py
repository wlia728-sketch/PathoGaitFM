"""OpenCap LabValidation -> 54-ch raw npy harmonizer (adapts crouch_harmonize_to_54ch.py).

OpenCap SPECIFICS vs the crouch harmonizer (all verified 2026-07-08):
  - Model = LaiArnoldModified2017 (NOT gait2392). Knee convention is POSITIVE flexion
    (verified: knee_angle_r 5.6..65.7 deg in walking) => knee sign = IDENTITY (+1.0), NOT flipped.
    hip/ankle also standard sign (hip_flexion 0..27, ankle -24..16).
  - Y-up already (OpenSim ground frame). No Z-up->Y-up rotation.
  - Force is ALREADY R/L labeled (R_ground_force_vy / L_ground_force_vy) => no CoP R/L assignment.
  - TWO kinematic modalities per subject, same force GT:
      MARKER : OpenSimData/Mocap/IK/<trial>.mot   (lab motion capture)
      VIDEO  : OpenSimData/Video/HRNet/2-cameras/IK/<trial>.mot  (2-camera video -> the deployable input)
    Run with modality arg to build each; force GT is identical for both.
  - Healthy adults. Mass estimated from vGRF plateau (peak ~1.0-1.2 BW) unless masses.json present.

54-ch layout (subset filled) -- IDENTICAL to crouch/Meyns so the frozen eval maps the same:
  angles : 0 R_Hip_X, 3 L_Hip_X, 6 R_Knee_X, 9 L_Knee_X, 12 R_Ankle_X, 15 L_Ankle_X
  pelvis : 48-50 R, 51-53 L
  GRF    : R -> 34 AP(vx),35 ML(vz),36 vert(vy) ; L -> 37,38,39
Cycle: per foot, plate stance (vy>thresh) as stance window; full gait cycle (stance in [0,stance_pts),
       swing=0), resample 101; angles interpolated over full cycle by absolute time.
Output: processed/<SUBJ>__<MOD>.npy (n,101,54) + _mask.npy + _meta.csv ; _summary.json. ASCII stdout.

Usage:  python opencap_harmonize.py marker [subject2 ...]
        python opencap_harmonize.py video  [subject2 ...]
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAWROOT = HERE / "raw" / "LabValidation_withoutVideos"
OUT = HERE / "processed"
G = 9.80665

# angle alignment: LaiArnold walking convention -> training internal (all IDENTITY, verified)
ANG = [
    ("hip_flexion_r", 0, +1.0), ("hip_flexion_l", 3, +1.0),
    ("knee_angle_r", 6, +1.0), ("knee_angle_l", 9, +1.0),   # IDENTITY (LaiArnold knee is +ve flexion)
    ("ankle_angle_r", 12, +1.0), ("ankle_angle_l", 15, +1.0),
]
PEL = {"pelvis_tilt": (48, 51), "pelvis_list": (49, 52), "pelvis_rotation": (50, 53)}
GRF54 = {"R": (34, 35, 36), "L": (37, 38, 39)}   # AP, ML, vertical (54-ch order X,Y,Z)
# GRF component signs: OpenCap R/L-labeled ground frame -> training. vertical identity (>=0).
# AP(vx)/ML(vz) left identity; only vertical GRF is scored for OpenCap, so their sign is immaterial.
GRF_SIGN = {"R": (+1.0, +1.0, +1.0), "L": (+1.0, +1.0, +1.0)}

GRF_THRESH_BW = 0.05
MIN_STANCE_FR = 10
STANCE_FRAC = 0.63


def read_mot(path):
    lines = Path(path).read_text().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    ci = next(i for i in range(hi + 1, len(lines)) if lines[i].strip().lower().startswith("time"))
    cols = lines[ci].split()
    rows = [ln.split() for ln in lines[ci + 1:] if ln.strip()]
    data = np.array([[float(x) for x in r] for r in rows if len(r) == len(cols)])
    return cols, data


def resample_n(x, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def ik_path(subj_dir, modality, trial):
    if modality == "marker":
        return subj_dir / "OpenSimData/Mocap/IK" / f"{trial}.mot"
    return subj_dir / "OpenSimData/Video/HRNet/2-cameras/IK" / f"{trial}.mot"


def harmonize_subject(subj, modality, masses):
    OUT.mkdir(parents=True, exist_ok=True)
    sd = RAWROOT / subj
    # walking trials (overground walking1/2/3; TS = treadmill, skip for force-plate GT)
    trials = sorted(p.stem.replace("_forces", "") for p in (sd / "ForceData").glob("walking[0-9]_forces.mot"))
    if not trials:
        print(f"[{subj}] no walking force trials"); return None
    cycles, meta = [], []
    for trial in trials:
        ikf = ik_path(sd, modality, trial)
        grf_f = sd / "ForceData" / f"{trial}_forces.mot"
        if not ikf.exists() or not grf_f.exists():
            continue
        icols, idat = read_mot(ikf); it = idat[:, 0]
        def icol(n): return idat[:, icols.index(n)] if n in icols else None
        gcols, gdat = read_mot(grf_f); gt = gdat[:, 0]
        sides = {}
        for side, pfx in (("R", "R_ground_force"), ("L", "L_ground_force")):
            if f"{pfx}_vy" in gcols:
                sides[side] = {"vx": gdat[:, gcols.index(f"{pfx}_vx")],
                               "vy": gdat[:, gcols.index(f"{pfx}_vy")],
                               "vz": gdat[:, gcols.index(f"{pfx}_vz")]}
        # mass from provided or vy plateau (peak ~1.05 BW)
        mass = masses.get(subj)
        if mass is None:
            peak = max(s["vy"].max() for s in sides.values())
            mass = float(peak / (1.05 * G))
        thr = GRF_THRESH_BW * mass * G

        for side, blk in sides.items():
            vy = blk["vy"]; loaded = vy > thr
            i = 0; n = len(loaded)
            while i < n:
                if loaded[i]:
                    j = i
                    while j < n and loaded[j]:
                        j += 1
                    if j - i >= MIN_STANCE_FR:
                        lo, hi = i, j
                        t0, t1 = gt[lo], gt[hi - 1]
                        if t1 <= it[0] or t0 >= it[-1]:
                            i = j; continue
                        sel = (it >= max(t0, it[0])) & (it <= min(t1, it[-1]))
                        if sel.sum() < 3:
                            i = j; continue
                        # IK divergence gate
                        bad = False
                        for cn, lim in (("hip_flexion_r", 120), ("hip_flexion_l", 120),
                                        ("knee_angle_r", 140), ("knee_angle_l", 140),
                                        ("ankle_angle_r", 70), ("ankle_angle_l", 70)):
                            cc = icol(cn)
                            if cc is not None and np.nanmax(np.abs(cc[sel])) > lim:
                                bad = True; break
                        if bad:
                            meta.append({"trial": trial, "side": side, "status": "IK_DIVERGED"}); i = j; continue
                        cyc = np.full((101, 54), np.nan, np.float32)
                        stance_pts = max(2, int(round(101 * STANCE_FRAC)))
                        ap_i, ml_i, vt_i = GRF54[side]
                        ap_s, ml_s, vt_s = GRF_SIGN[side]
                        for gi, comp, sgn in ((vt_i, "vy", vt_s), (ap_i, "vx", ap_s), (ml_i, "vz", ml_s)):
                            full = np.zeros(101, np.float32)
                            seg = sgn * blk[comp][lo:hi] / (mass * G)
                            if comp == "vy":
                                seg = np.clip(seg, 0, None)
                            full[:stance_pts] = resample_n(seg, stance_pts)
                            cyc[:, gi] = full
                        cyc_end_t = t0 + (t1 - t0) / STANCE_FRAC
                        ct = np.clip(np.linspace(t0, cyc_end_t, 101), it[0], it[-1])
                        for cn, ch, sgn in ANG:
                            cc = icol(cn)
                            if cc is not None:
                                cyc[:, ch] = sgn * np.interp(ct, it, cc)
                        for cn, (rch, lch) in PEL.items():
                            cc = icol(cn)
                            if cc is not None:
                                seg = np.interp(ct, it, cc); cyc[:, rch] = seg; cyc[:, lch] = seg
                        cycles.append(cyc)
                        meta.append({"trial": trial, "side": side, "status": "OK", "stance_fr": hi - lo})
                    i = j
                else:
                    i += 1
    if not cycles:
        print(f"[{subj}/{modality}] no cycles"); return None
    stack = np.stack(cycles)
    mask = np.isfinite(stack).all(axis=1)
    tag = f"{subj}__{modality}"
    np.save(OUT / f"{tag}.npy", stack.astype(np.float32))
    np.save(OUT / f"{tag}_mask.npy", mask)
    with open(OUT / f"{tag}_meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "trial", "side", "stance_fr", "status"])
        for i, m in enumerate(meta):
            w.writerow([i, m["trial"], m["side"], m.get("stance_fr", ""), m["status"]])
    nvgrf = int(mask[:, [36, 39]].sum())
    print(f"[{tag}] {stack.shape[0]} cyc, mass~{mass:.1f}kg, vGRF-valid {nvgrf}")
    return {"subj": subj, "modality": modality, "n_cyc": int(stack.shape[0]), "mass": mass}


def main():
    modality = sys.argv[1] if len(sys.argv) > 1 else "marker"
    assert modality in ("marker", "video")
    mj = HERE / "masses.json"
    masses = json.load(open(mj)) if mj.exists() else {}
    subs = sys.argv[2:] or sorted(p.name for p in RAWROOT.glob("subject*") if p.is_dir())
    summ = []
    for s in subs:
        try:
            r = harmonize_subject(s, modality, masses)
            if r: summ.append(r)
        except Exception as e:
            import traceback; print(f"[{s}] FAIL {type(e).__name__}: {e}"); traceback.print_exc()
    (OUT / f"_summary_{modality}.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\nHarmonized {len(summ)} subjects ({modality}) -> {OUT}")


if __name__ == "__main__":
    main()
