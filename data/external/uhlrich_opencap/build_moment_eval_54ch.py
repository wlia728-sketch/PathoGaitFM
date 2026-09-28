"""Build a 54-ch processed npy for the OpenCap MOMENT eval.

This reuses the EXACT quality-gated L-stance segmentation from
extract_moments_id_101pt.py (same genuine-HS->TO windows, same gate, same
interior resampling to 101 pts), so that for every kept cycle BOTH the visible
kinematics (hip/knee/ankle sagittal angles) AND the ID moment ground truth are
extracted over the IDENTICAL window and IDENTICAL 101-pt normalization. This is
required for a valid moment eval: the frozen model maps the visible-angle
waveform to the moment waveform point-for-point, so the input angles and the
target moments must be co-registered in time.

CYCLE LAYOUT -- matches the TRAINING / opencap_harmonize.py convention exactly:
  Trials are short single-stance overground passes; NO left HS->HS full cycle
  exists (verified: no left trial has >=2 genuine heel strikes). The model was
  trained on full gait cycles with STANCE compressed into [0, stance_pts) (~0-63%)
  and swing after. So for each quality-gated L-stance window [t0,t1] we build the
  cycle the harmonizer way:
    - ANGLES: interpolated over the FULL-cycle time span [t0, t0+(t1-t0)/STANCE_FRAC]
      onto 101 pts (real swing kinematics come from the IK record past toe-off),
      giving the model a realistic full-cycle kinematic input.
    - MOMENTS: the stance-window ID moment (resampled to stance_pts) placed into
      [0:stance_pts); swing (stance_pts:101) = 0. Moments are ~0 in swing, and the
      eval scores the STANCE window only, so the swing fill is not scored.
  This co-registers GT moments with the model's stance-compressed layout and
  matches how the published external moment/vGRF cycles are built.

54-ch layout (subset filled):
  angles : 0 R_Hip_X, 3 L_Hip_X, 6 R_Knee_X, 9 L_Knee_X, 12 R_Ankle_X, 15 L_Ankle_X
  pelvis : 48-50 R, 51-53 L
  moments: 18 R_Hip_Mom_X, 21 L_Hip_Mom_X, 24 R_Knee_Mom_X, 25 L_Knee_Mom_X,
           26 R_Ankle_Mom_X, 27 L_Ankle_Mom_X   (Nm/kg, training convention)

OpenCap kept cycles are ALL left-stance (right rejected upstream for onset
residual), so we fill only the L moment channels (21/25/27) + L angles.
Angle sign convention: LaiArnold IDENTITY (verified in opencap_harmonize.py:
knee is +ve flexion, no flip). Moment sign convention: a global sagittal-moment
sign flip is applied (see the MOM table below for the empirical rationale: the raw
OpenCap/LaiArnold ID convention is inverted relative to the model's effective
training convention). We DO NOT normalize moments here -- they stay in Nm/kg raw so
the V4 transform z-scores them exactly as training moment channels.

Output: processed/<subj>__momeval.npy (n,101,54) + _mask.npy (n,54) + _meta.csv
        _summary_momeval.json
ASCII-only stdout.
"""
import os, sys, json, glob, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# reuse the exact gate + segmentation from the moment extractor
import extract_moments_id_101pt as X

RAW = X.RAW
OUT = os.path.join(HERE, "processed")
G = X.G

# 54-ch slots
ANG = {  # coord name -> (54ch idx, sign)  LaiArnold IDENTITY
    "hip_flexion_r": (0, +1.0), "hip_flexion_l": (3, +1.0),
    "knee_angle_r": (6, +1.0), "knee_angle_l": (9, +1.0),
    "ankle_angle_r": (12, +1.0), "ankle_angle_l": (15, +1.0),
}
PEL = {"pelvis_tilt": (48, 51), "pelvis_list": (49, 52), "pelvis_rotation": (50, 53)}
# moment coord -> (54ch idx, sign).
# SIGN = -1.0: empirically the OpenCap/LaiArnold ID sagittal moment convention is
# INVERTED relative to the MODEL's effective training convention (which is the
# Fukuchi/AddBio-harmonized convention: ankle plantarflex-moment POSITIVE, hip
# early-stance POSITIVE in normalized space). Verified by comparing OpenCap vs
# Fukuchi moments in the model's post-transform normalized space: at mid-stance
# OpenCap ankle=-0.59 vs Fukuchi ankle=+0.67 (opposite), hip early -0.27 vs +0.39.
# The quality report's "no flip" was against a RAW Camargo reference; the model's
# effective convention (after Camargo ch20/23 flips + the Fukuchi harmonizer) is
# the opposite, so a global sagittal-moment sign flip is required here.
MOM = {
    "R": {"hip": (18, -1.0), "knee": (24, -1.0), "ankle": (26, -1.0)},
    "L": {"hip": (21, -1.0), "knee": (25, -1.0), "ankle": (27, -1.0)},
}
STANCE_FRAC = 0.63   # matches crouch/opencap harmonizer + training cycle convention


def resample_n(x, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def build_subject(subj):
    """Re-run the exact gate; for each KEPT (win='stance', side) window, emit a
    54-ch cycle with angles (from IK) + moments (from ID) over the SAME window."""
    id_dir = os.path.join(RAW, subj, "OpenSimData", "Mocap", "ID")
    ik_dir = os.path.join(RAW, subj, "OpenSimData", "Mocap", "IK")
    force_dir = os.path.join(RAW, subj, "ForceData")
    osim = os.path.join(RAW, subj, "OpenSimData", "Mocap", "Model",
                        "LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim")
    mass = X.total_mass_from_osim(osim)
    if mass is None or not (30 < mass < 200):
        mass = X._mass_fallback().get(subj)
    BW = mass * G

    cycles, meta = [], []
    for idf in sorted(glob.glob(os.path.join(id_dir, "*.sto"))):
        trial = os.path.splitext(os.path.basename(idf))[0]
        ff = os.path.join(force_dir, trial + "_forces.mot")
        ikf = os.path.join(ik_dir, trial + ".mot")
        if not (os.path.exists(ff) and os.path.exists(ikf)):
            continue
        icols, idata = X.read_sto_mot(idf)     # ID
        fcols, fdata = X.read_sto_mot(ff)       # forces
        kcols, kdata = X.read_sto_mot(ikf)      # IK angles
        it = idata[:, 0]; ft = fdata[:, 0]; kt = kdata[:, 0]

        def icol(n): return idata[:, icols.index(n)]
        def fcol(n): return fdata[:, fcols.index(n)]
        def kcol(n): return kdata[:, kcols.index(n)] if n in kcols else None

        # residuals (same as extractor)
        res_fmag = np.sqrt(icol("pelvis_tx_force")**2 + icol("pelvis_ty_force")**2
                           + icol("pelvis_tz_force")**2)
        res_mmag = np.sqrt(icol("pelvis_tilt_moment")**2 + icol("pelvis_list_moment")**2
                           + icol("pelvis_rotation_moment")**2) / mass

        def gate_window(win, side, t0, t1):
            """Return (imask, tseg) if the window passes the SAME gate, else None.
            Mirrors X.gate_and_extract exactly (interior residual, COP, physiology)."""
            vy = fcol(f"{side}_ground_force_vy")
            px = fcol(f"{side}_ground_force_px"); pz = fcol(f"{side}_ground_force_pz")
            dur = t1 - t0
            if win == "stance" and not (0.25 <= dur <= 1.1):
                return None
            if t0 <= ft[0] + 1e-6 or t1 >= ft[-1] - 1e-6:
                return None
            stmask = (ft >= t0) & (ft <= t1)
            vyc = vy[stmask]; loaded = vyc > X.HS_THRESH_N
            if loaded.sum() < 5:
                return None
            cop_x = px[stmask][loaded]; cop_z = pz[stmask][loaded]
            if not (np.all(np.isfinite(cop_x)) and np.all(np.isfinite(cop_z))):
                return None
            if (np.ptp(cop_x) > 1.5) or (np.ptp(cop_z) > 1.0):
                return None
            imask = (it >= t0) & (it <= t1)
            if imask.sum() < 5:
                return None
            rf_win = res_fmag[imask]; rm_win = res_mmag[imask]
            n_i = len(rf_win); lo_i = int(n_i * 0.10); hi_i = max(int(n_i * 0.90), lo_i + 1)
            rfmax = float(np.max(rf_win[lo_i:hi_i])); rmmax = float(np.max(rm_win[lo_i:hi_i]))
            if rfmax > X.RESID_F_REJECT * BW:
                return None
            if rmmax > X.RESID_M_REJECT:
                return None
            # physiology (moments) + ankle push-off polarity -- same as extractor
            tseg = it[imask]
            wf = {}
            for j in X.JOINTS:
                colname = {"hip": f"hip_flexion_{side.lower()}_moment",
                           "knee": f"knee_angle_{side.lower()}_moment",
                           "ankle": f"ankle_angle_{side.lower()}_moment"}[j]
                raw = icol(colname)[imask] / mass
                w = X.resample101(tseg, raw)
                if w is None or not np.all(np.isfinite(w)):
                    return None
                if np.max(np.abs(w)) > X.PHYS_PEAK[j]:
                    return None
                wf[j] = w
            lo = 40  # stance
            amid = wf["ankle"][lo:]
            if amid.min() > -0.05:
                return None
            return imask, tseg, wf

        for side in ("R", "L"):
            vy = fcol(f"{side}_ground_force_vy")
            for (t0, t1) in X.stance_phases(vy, ft):
                g = gate_window("stance", side, t0, t1)
                if g is None:
                    continue
                imask, tseg, wf = g   # wf[j] = stance-window ID moment resampled to 101
                cyc = np.full((101, 54), np.nan, np.float32)
                stance_pts = max(2, int(round(101 * STANCE_FRAC)))
                # FULL-cycle time span (stance = [t0,t1] occupies STANCE_FRAC of it)
                cyc_end_t = t0 + (t1 - t0) / STANCE_FRAC
                ct = np.clip(np.linspace(t0, cyc_end_t, 101), kt[0], kt[-1])
                # ANGLES: full-cycle interpolation over absolute time (real swing kinematics)
                for cn, (ch, sgn) in ANG.items():
                    cc = kcol(cn)
                    if cc is None:
                        continue
                    cyc[:, ch] = sgn * np.interp(ct, kt, cc)
                for cn, (rch, lch) in PEL.items():
                    cc = kcol(cn)
                    if cc is None:
                        continue
                    seg = np.interp(ct, kt, cc)
                    cyc[:, rch] = seg; cyc[:, lch] = seg
                # MOMENTS: stance-window moment compressed into [0:stance_pts), swing=0
                for j in X.JOINTS:
                    ch, sgn = MOM[side][j]
                    full = np.zeros(101, np.float32)
                    full[:stance_pts] = resample_n(sgn * wf[j].astype(np.float32), stance_pts)
                    cyc[:, ch] = full
                cycles.append(cyc)
                meta.append({"trial": trial, "side": side, "status": "OK"})

    if not cycles:
        print(f"[{subj}] no cycles"); return None
    stack = np.stack(cycles).astype(np.float32)
    mask = np.isfinite(stack).all(axis=1)   # (n,54) valid channels
    tag = f"{subj}__momeval"
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, tag + ".npy"), stack)
    np.save(os.path.join(OUT, tag + "_mask.npy"), mask)
    with open(os.path.join(OUT, tag + "_meta.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "trial", "side", "status"])
        for i, m in enumerate(meta):
            w.writerow([i, m["trial"], m["side"], m["status"]])
    nL = int(mask[:, 25].sum())  # L_Knee_Mom valid
    print(f"[{tag}] {stack.shape[0]} cyc, mass~{mass:.1f}kg, L-moment-valid {nL}, "
          f"sides {sorted(set(m['side'] for m in meta))}")
    return {"subj": subj, "n_cyc": int(stack.shape[0]), "mass": float(mass)}


def main():
    subjects = sorted([d for d in os.listdir(RAW) if d.startswith("subject")],
                      key=lambda s: int(s.replace("subject", "")))
    summ = []
    for s in subjects:
        try:
            r = build_subject(s)
            if r:
                summ.append(r)
        except Exception as e:
            import traceback
            print(f"[{s}] FAIL {type(e).__name__}: {e}"); traceback.print_exc()
    json.dump(summ, open(os.path.join(OUT, "_summary_momeval.json"), "w"), indent=2)
    tot = sum(d["n_cyc"] for d in summ)
    print(f"\nBuilt {len(summ)} subjects, {tot} cycles -> {OUT}")


if __name__ == "__main__":
    main()
