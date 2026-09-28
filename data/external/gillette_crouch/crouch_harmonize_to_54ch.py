"""crouchgait -> 54-ch raw npy harmonizer (mirrors data/external/leuven_cp/cp_harmonize_to_54ch.py
conventions, adapted for the crouchgait SimTK set).

DIFFERENCES vs the Meyns CP harmonizer, each with the reason it is legitimate:
  - Input = pre-computed OpenSim gait2392 IK.mot (angles) + measured force-plate GRF.mot (3-axis).
    (Meyns re-ran IK from c3d; here IK/GRF are provided. Legit: GRF is measured, independent of IK,
     so no self-consistency circularity for the GRF targets.)
  - NO Z-up->Y-up rotation (IK already solved in Y-up ground frame).
  - Angle convention alignment (crouch gait2392 -> training internal-CP), verified against the
    training internal-CP reference:
        hip: IDENTITY, knee: SIGN-FLIP (gait2392 knee-flexion is negative; training is positive),
        ankle: IDENTITY.
  - Vertical GRF is the scored target; moments left NaN (no independent moment truth).

54-ch layout (subset filled):
  angles : 0 R_Hip_X, 3 L_Hip_X, 6 R_Knee_X, 9 L_Knee_X, 12 R_Ankle_X, 15 L_Ankle_X
  pelvis : 48-50 R, 51-53 L  (single pelvis replicated)
  GRF    : R block -> 34 R_GRF_X, 35 R_GRF_Y, 36 R_GRF_Z (vertical is the scored target)
           L block -> 37 L_GRF_X, 38 L_GRF_Y, 39 L_GRF_Z
           (54-ch GRF order is X,Y,Z per side; the 40-ch eval maps 34/35/36 -> R 20/21/22 and
            37/38/39 -> L 23/24/25. vertical = Z = idx 36/39, matching the Meyns harmonizer.)

Cycle: per foot, plate stance (vy>thresh) as the stance window; build FULL gait cycle (stance bump
       in [0,stance_pts), swing=0 after), resample to 101. Angles interpolated over the full cycle
       by absolute time.

Output: processed/<SUBJ>.npy (n,101,54) + _mask.npy (n,54) + _meta.csv ; _summary.json
ASCII-only stdout. Windows python (numpy only; IK/GRF are text .mot).
"""
from __future__ import annotations
import sys, glob, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "processed"
G = 9.80665

# ---- angle alignment (crouch gait2392 -> training convention) ----
# (coord_name, 54ch_index, sign)
ANG = [
    ("hip_flexion_r", 0, +1.0), ("hip_flexion_l", 3, +1.0),
    ("knee_angle_r", 6, -1.0), ("knee_angle_l", 9, -1.0),   # SIGN-FLIP
    ("ankle_angle_r", 12, +1.0), ("ankle_angle_l", 15, +1.0),
]
PEL = {"pelvis_tilt": (48, 51), "pelvis_list": (49, 52), "pelvis_rotation": (50, 53)}
# GRF 54-ch indices per side: X, Y, Z (Z is vertical, the scored target)
GRF54 = {"R": (34, 35, 36), "L": (37, 38, 39)}
# Per-side GRF component SIGNS to align crouch OpenSim ground-frame -> training convention.
# Determined empirically by a GT-vs-training-VDK coordinate audit (stance window):
#   AP: R +0.96 / L +0.90  (identity, both sides)
#   ML: R -0.68 (FLIP) / L +0.76 (identity)  -- OpenSim z is mirrored for the right foot
#   vertical: identity (always >=0)
# order per side = (AP_sign, ML_sign, vert_sign)
GRF_SIGN = {"R": (+1.0, -1.0, +1.0), "L": (+1.0, +1.0, +1.0)}

GRF_THRESH_N = None          # set per-subject from mass
GRF_THRESH_BW = 0.05
MIN_STANCE_FR = 10
STANCE_FRAC = 0.63           # match Fukuchi/vdk_stroke convention (was 0.60)


def read_mot(path):
    lines = Path(path).read_text().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    ci = next(i for i in range(hi + 1, len(lines)) if lines[i].strip().lower().startswith("time"))
    cols = lines[ci].split()
    rows = [ln.split() for ln in lines[ci + 1:] if ln.strip()]
    data = np.array([[float(x) for x in r] for r in rows if len(r) == len(cols)])
    return cols, data


def load_masses():
    """Per-subject body mass from masses.json beside this script, built from the release's own
    subject table. When it is absent this returns an empty table and the caller estimates mass
    from the vertical-GRF plateau, which is why the reported numbers were produced with the
    file in place."""
    mj = HERE / "masses.json"
    if mj.exists():
        return json.load(open(mj))
    return {}


def resample_n(x, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def harmonize_subject(subj, masses):
    OUT.mkdir(parents=True, exist_ok=True)
    ik_f = RAW / f"{subj}_IK.mot"
    grf_f = RAW / f"{subj}_GRF.mot"
    if not ik_f.exists() or not grf_f.exists():
        print(f"[{subj}] missing IK/GRF, skip"); return None
    icols, idat = read_mot(ik_f)
    it = idat[:, 0]
    def icol(n): return idat[:, icols.index(n)] if n in icols else None

    gcols, gdat = read_mot(grf_f)
    gt = gdat[:, 0]
    # two force blocks: cols 1-3 (vx,vy,vz) block1, 7-9 block2; CoP px at 4 and 10
    # CoP: px (col 4/10) = forward, pz (col 6/12) = LATERAL (used for R/L assignment)
    blocks = {"b1": {"vx": gdat[:, 1], "vy": gdat[:, 2], "vz": gdat[:, 3], "px": gdat[:, 4], "pz": gdat[:, 6]},
              "b2": {"vx": gdat[:, 7], "vy": gdat[:, 8], "vz": gdat[:, 9], "px": gdat[:, 10], "pz": gdat[:, 12]}}

    # mass: use provided, else estimate from vy plateau (peak ~ 1.0-1.2 BW for crouch)
    mass = masses.get(subj)
    if mass is None:
        peak = max(blocks["b1"]["vy"].max(), blocks["b2"]["vy"].max())
        mass = float(peak / (1.05 * G))   # rough: peak vGRF ~1.05 BW in crouch
    thr = GRF_THRESH_BW * mass * G

    # assign each block to R/L by LATERAL CoP position (CoP-z in OpenSim ground frame).
    # The two plates are struck by opposite feet; the foot on the RIGHT sits at a larger
    # (more positive) lateral z than the LEFT. So of the two loaded blocks, the one with the
    # larger mean CoP-z is R, the other L. This is marker-free and robust (the earlier
    # marker-based approach failed because several subjects' IK.mot/.trc lack foot markers).
    # Verified against the marker-bearing subjects (C2: b1 z=-0.112 -> R, b2 z=-0.264 -> L).
    loaded_blocks = {b: blk for b, blk in blocks.items() if (blk["vy"] > thr).sum() >= MIN_STANCE_FR}
    assign = {b: None for b in blocks}
    if len(loaded_blocks) == 2:
        (ba, bla), (bb, blb) = list(loaded_blocks.items())
        za = np.nanmean(bla["pz"][bla["vy"] > thr])
        zb = np.nanmean(blb["pz"][blb["vy"] > thr])
        if za >= zb:
            assign[ba], assign[bb] = "R", "L"
        else:
            assign[ba], assign[bb] = "L", "R"
    elif len(loaded_blocks) == 1:
        # single loaded plate: cannot disambiguate R/L from one CoP-z alone; use marker if any,
        # else flag as ambiguous (skip to avoid a wrong-side contamination).
        b = next(iter(loaded_blocks))
        assign[b] = "AMB"

    cycles, meta = [], []
    for b, side in assign.items():
        if side is None or side == "AMB":
            if side == "AMB":
                meta.append({"block": b, "side": "AMB", "status": "AMBIGUOUS_SINGLE_PLATE", "stance_fr": ""})
            continue
        blk = blocks[b]
        vy = blk["vy"]
        loaded = vy > thr
        # stance window = contiguous loaded run(s)
        i = 0; n = len(loaded)
        while i < n:
            if loaded[i]:
                j = i
                while j < n and loaded[j]:
                    j += 1
                if j - i >= MIN_STANCE_FR:
                    lo, hi = i, j
                    t0, t1 = gt[lo], gt[hi - 1]
                    # IK must cover the stance window
                    if t1 <= it[0] or t0 >= it[-1]:
                        i = j; continue
                    sel = (it >= max(t0, it[0])) & (it <= min(t1, it[-1]))
                    if sel.sum() < 3:
                        i = j; continue
                    # IK divergence gate: reject anatomically impossible (not just deep crouch).
                    # crouch knee is deeply flexed (negative in gait2392); gate on |value|>140 only.
                    bad = False
                    for cn, lim in (("hip_flexion_r", 120), ("hip_flexion_l", 120),
                                    ("knee_angle_r", 140), ("knee_angle_l", 140),
                                    ("ankle_angle_r", 70), ("ankle_angle_l", 70)):
                        cc = icol(cn)
                        if cc is not None and np.nanmax(np.abs(cc[sel])) > lim:
                            bad = True; break
                    if bad:
                        meta.append({"block": b, "side": side, "status": "IK_DIVERGED",
                                     "stance_fr": hi - lo}); i = j; continue
                    # full gait cycle
                    stance_dur = hi - lo
                    cyc_end_t = t0 + (t1 - t0) / STANCE_FRAC
                    cyc = np.full((101, 54), np.nan, np.float32)
                    stance_pts = max(2, int(round(101 * STANCE_FRAC)))
                    # GRF (measured) into stance segment, swing=0. All GRF channels as-is (N->BW).
                    ap_i, ml_i, vt_i = GRF54[side]
                    ap_s, ml_s, vt_s = GRF_SIGN[side]
                    # (54ch idx, grf.mot component, sign) -- AP<-vx, ML<-vz, vertical<-vy
                    for gi, comp, sgn in ((vt_i, "vy", vt_s), (ap_i, "vx", ap_s), (ml_i, "vz", ml_s)):
                        full = np.zeros(101, np.float32)
                        seg = sgn * blk[comp][lo:hi] / (mass * G)   # BW, sign-aligned to training
                        if comp == "vy":
                            seg = np.clip(seg, 0, None)             # vertical >=0
                        full[:stance_pts] = resample_n(seg, stance_pts)
                        cyc[:, gi] = full
                    # angles over full cycle by absolute time
                    ct = np.clip(np.linspace(t0, cyc_end_t, 101), it[0], it[-1])
                    for cn, ch, sgn in ANG:
                        cc = icol(cn)
                        if cc is not None:
                            cyc[:, ch] = sgn * np.interp(ct, it, cc)
                    for cn, (rch, lch) in PEL.items():
                        cc = icol(cn)
                        if cc is not None:
                            seg = np.interp(ct, it, cc)
                            cyc[:, rch] = seg; cyc[:, lch] = seg
                    cycles.append(cyc)
                    meta.append({"block": b, "side": side, "status": "OK", "stance_fr": stance_dur})
                i = j
            else:
                i += 1

    if not cycles:
        print(f"[{subj}] no cycles"); return None
    stack = np.stack(cycles)
    mask = np.isfinite(stack).all(axis=1)
    np.save(OUT / f"{subj}.npy", stack.astype(np.float32))
    np.save(OUT / f"{subj}_mask.npy", mask)
    with open(OUT / f"{subj}_meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "block", "side", "stance_fr", "status"])
        for i, m in enumerate(meta):
            w.writerow([i, m["block"], m["side"], m.get("stance_fr", ""), m["status"]])
    nvgrf = int(mask[:, [36, 39]].sum())
    print(f"[{subj}] {stack.shape[0]} cyc, mass~{mass:.1f}kg, vGRF-valid {nvgrf}, "
          f"sides {[m['side'] for m in meta if m['status']=='OK']}")
    return {"subj": subj, "n_cyc": int(stack.shape[0]), "mass": mass}


def main():
    masses = load_masses()
    subs = sys.argv[1:] or sorted({Path(f).name.split("_")[0] for f in glob.glob(str(RAW / "C*_IK.mot"))})
    summ = []
    for s in subs:
        try:
            r = harmonize_subject(s, masses)
            if r: summ.append(r)
        except Exception as e:
            import traceback; print(f"[{s}] FAIL {type(e).__name__}: {e}"); traceback.print_exc()
    (OUT / "_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\nHarmonized {len(summ)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
