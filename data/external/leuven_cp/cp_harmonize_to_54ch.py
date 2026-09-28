"""Step 3 — harmonize CP (OpenSim IK angles + measured force-plate vGRF) -> 54-ch raw npy.

Mirrors the Fukuchi harmonizer's 54-ch layout/convention so the same zero-shot eval
(training z-stats, KEEP_CHANNELS_54TO40) consumes it unchanged. SCOPE = vGRF-only claim:
we fill the INPUT channels (sagittal hip/knee/ankle angles + pelvis) from IK, and the
ground-truth vGRF (channels 36/39) from the measured force plates. Joint MOMENTS are left
NaN/masked-out on purpose (no inverse dynamics -> no self-consistency circularity).

54-ch channel map (subset we fill):
  angles  : 0 R_Hip_X, 3 L_Hip_X, 6 R_Knee_X, 9 L_Knee_X, 12 R_Ankle_X, 15 L_Ankle_X
  pelvis  : 48 R_Pelvis_X, 49 R_Pelvis_Y, 50 R_Pelvis_Z, 51 L_Pelvis_X, 52 L_Pelvis_Y, 53 L_Pelvis_Z
            (pelvis is one segment; we replicate pelvis_tilt/list/rotation into both R/L slots,
             matching how the training pelvis channels are populated for single-pelvis data)
  vGRF    : 36 R_GRF_Z (vertical), 39 L_GRF_Z (vertical)   [target / ground truth]

Force-plate -> foot assignment uses the GROUND-TRUTH foot-marker position during each plate's
loaded window (foot marker inside that plate's CORNERS), NOT max-load — per the project rule
(see memory grf-moment-foot-assignment). vGRF sign: c3d FZ is downward-negative -> vGRF = -FZ.
Baseline: swing vGRF must be ~0 (we subtract a small unloaded baseline). Units: N -> BW (/(m*g)),
matching the training 'normal' convention is NOT assumed; eval scores GRF on a stance window to
neutralize cycle-convention differences (same as Fukuchi).

Cycle definition: per foot, heel-strike (vGRF rising through a threshold) to next heel-strike;
resample each cycle to 101 points. Each plate contact typically yields ONE clean stance per foot
in overground walking, so we emit per-foot representative cycles.

Output: processed/<SUBJ>.npy (n_cyc,101,54) + _mask.npy (n_cyc,54) + _meta.csv ; _summary.json
ASCII-only stdout. Runs in Windows python (needs ezc3d + numpy + openpyxl).
"""
from __future__ import annotations
import sys, glob, json, csv
from pathlib import Path

import numpy as np
import ezc3d

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
IK = HERE / "ik"
OUT = HERE / "processed"
G = 9.80665

# IK mot coordinate -> 54-ch angle index (sagittal). IK is in degrees (inDegrees=yes).
ANG_FROM_IK = {
    "hip_flexion_r": 0, "hip_flexion_l": 3,
    "knee_angle_r": 6, "knee_angle_l": 9,
    "ankle_angle_r": 12, "ankle_angle_l": 15,
}
# pelvis (single segment) -> replicate into R(48,49,50) and L(51,52,53) X/Y/Z slots
PEL_FROM_IK = {
    "pelvis_tilt": (48, 51), "pelvis_list": (49, 52), "pelvis_rotation": (50, 53),
}
R_VGRF_54, L_VGRF_54 = 36, 39

GRF_THRESH_BW = 0.05      # heel-strike / toe-off threshold as fraction of body weight
MIN_STANCE_FR = 15        # frames
STANCE_FRAC = 0.60        # stance fraction of a full gait cycle (typical ~58-62%); stance fills
                          # the first STANCE_FRAC of the 101-pt cycle, swing (vGRF=0) the rest.


def group_of(subj):
    s = subj.lower()
    return "hecp" if s.startswith("hecp") else ("dicp" if s.startswith("dicp") else "td")


def load_masses():
    """Body mass in kg per subject, from the de-identified masses.json.

    This previously read the site's participant_information.xlsx. That workbook carried age, sex, height
    and GMFCS alongside mass for 14 minors, so only mass and height are read from it,
    and the harmonisers only ever used the mass and height columns. The workbook is no longer shipped; see
    PROVENANCE.md. masses.json mirrors the sibling crouch site's format.
    """
    import json
    mj = HERE / "masses.json"
    if not mj.exists():
        raise FileNotFoundError(
            "masses.json not found. Build it from participant_information.xlsx in the SimTK "
            "cp-child-gait download as {ID: {mass_kg: float, height_cm: float}} and place it beside "
            "this script. See README, Dataset layout.")
    table = json.load(open(mj))
    return {str(k).strip().upper(): float(v["mass_kg"]) for k, v in table.items()}


def read_ik_mot(path):
    """Return dict coord->array (deg) and the time vector."""
    lines = Path(path).read_text().splitlines()
    hi = [i for i, l in enumerate(lines) if l.strip().lower() == "endheader"][0]
    cols = lines[hi + 1].split()
    data = np.array([[float(x) for x in ln.split()] for ln in lines[hi + 2:] if ln.strip()])
    return {c: data[:, j] for j, c in enumerate(cols)}, data[:, 0]


def read_c3d_grf_and_feet(path):
    """Return per-plate vGRF (point-rate resampled), plate corners, and foot-marker XY (point-rate)."""
    c = ezc3d.c3d(path)
    arate = float(c["parameters"]["ANALOG"]["RATE"]["value"][0])
    prate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    ratio = int(round(arate / prate))
    alab = [l.strip() for l in c["parameters"]["ANALOG"]["LABELS"]["value"]]
    ana = c["data"]["analogs"][0]  # (nch, nframe_analog)
    nfa = ana.shape[1]
    npf = nfa // ratio
    # vertical GRF per plate, sign-flipped (downward-neg -> up-positive), block-mean to point rate
    def plate_vgrf(idx):
        fz = -ana[alab.index(f"FZ{idx}")]    # up-positive N
        fz = fz[: npf * ratio].reshape(npf, ratio).mean(axis=1)
        # baseline-correct: the unloaded plate has a small electronic offset (~+/-25 N
        # after sign flip) that otherwise reads as "always loaded". Subtract the median
        # of the lowest-quartile samples (the genuine unloaded baseline) and clip >=0.
        baseline = np.median(np.sort(fz)[: max(1, len(fz) // 4)])
        fz = fz - baseline
        return fz
    plates = {}
    used = int(c["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0])
    corners = np.array(c["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"])  # (3,4,nplate)
    for p in range(used):
        plates[p] = {"vgrf": plate_vgrf(p + 1),
                     "cx": corners[0, :, p], "cy": corners[1, :, p]}
    # foot markers at point rate (mm)
    labs = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    pts = c["data"]["points"]
    def foot_xy(markers):
        arrs = []
        for m in markers:
            if m in labs:
                xy = pts[:2, labs.index(m), :].astype(float)  # (2,nframe)
                z = np.all(pts[:3, labs.index(m), :] == 0, axis=0)
                xy[:, z] = np.nan
                arrs.append(xy)
        if not arrs:
            return None
        return np.nanmean(np.stack(arrs), axis=0)  # (2,nframe)
    feet = {"R": foot_xy(["RHEE", "RTOE"]), "L": foot_xy(["LHEE", "LTOE"])}
    return plates, feet, prate, npf


def assign_plate_to_foot(plates, feet, mass):
    """For each plate, decide R or L by which foot marker sits inside the plate footprint
    during that plate's loaded window. Returns dict plate_idx -> 'R'/'L'/None."""
    thresh = GRF_THRESH_BW * mass * G
    out = {}
    for p, pl in plates.items():
        loaded = pl["vgrf"] > thresh
        if loaded.sum() < MIN_STANCE_FR:
            out[p] = None
            continue
        xlo, xhi = pl["cx"].min(), pl["cx"].max()
        ylo, yhi = pl["cy"].min(), pl["cy"].max()
        score = {}
        for side, xy in feet.items():
            if xy is None:
                score[side] = -1
                continue
            fr = np.where(loaded)[0]
            fr = fr[fr < xy.shape[1]]
            fx, fy = xy[0, fr], xy[1, fr]
            inside = ((fx >= xlo) & (fx <= xhi) & (fy >= ylo) & (fy <= yhi))
            score[side] = float(np.nanmean(inside)) if inside.size else -1
        best = max(score, key=score.get)
        out[p] = best if score[best] > 0.3 else None
    return out


def extract_stance_cycles(vgrf, mass, prate):
    """Return list of (lo,hi) frame windows = heel-strike to toe-off (stance), filtered."""
    thresh = GRF_THRESH_BW * mass * G
    loaded = vgrf > thresh
    cycles = []
    i = 0
    n = len(loaded)
    while i < n:
        if loaded[i]:
            j = i
            while j < n and loaded[j]:
                j += 1
            if j - i >= MIN_STANCE_FR:
                cycles.append((i, j))
            i = j
        else:
            i += 1
    return cycles


def resample101(x):
    xp = np.linspace(0, 1, len(x))
    return np.interp(np.linspace(0, 1, 101), xp, x)


def resample101_n(x, n):
    xp = np.linspace(0, 1, len(x))
    return np.interp(np.linspace(0, 1, n), xp, x)


def harmonize_subject(subj, masses):
    OUT.mkdir(parents=True, exist_ok=True)
    grp = group_of(subj)
    if subj.upper() not in masses:
        raise KeyError(subj + ": no body mass in the mass table. Mass sets the body-weight normalisation "
                       "of the vertical GRF and the plate contact threshold, so a substituted default "
                       "would rescale this subject's targets with no diagnostic.")
    mass = masses[subj.upper()]
    gait_c3ds = sorted(glob.glob(str(RAW / "gait" / grp / f"{subj}*.c3d")))
    cycles_all = []
    meta_rows = []
    for c3df in gait_c3ds:
        trial = Path(c3df).stem
        ik_mot = IK / f"{trial}_ik.mot"
        if not ik_mot.is_file():
            print(f"  [{trial}] no IK mot, skip"); continue
        coords, ik_t = read_ik_mot(str(ik_mot))
        plates, feet, prate, npf = read_c3d_grf_and_feet(c3df)
        assign = assign_plate_to_foot(plates, feet, mass)
        # The IK .mot keeps the ORIGINAL c3d timestamps (the trimmed window is not zeroed),
        # and the force-plate point-frame i corresponds to absolute time i/prate. So we align
        # plate stance to IK angles by ABSOLUTE TIME, not by frame proportion. A stance is only
        # usable if its time window lies inside the IK motion's time span.
        ik_t = np.asarray(ik_t)
        for p, side in assign.items():
            if side is None:
                continue
            vg = plates[p]["vgrf"] / (mass * G)            # BW (already baseline-corrected)
            for (lo, hi) in extract_stance_cycles(plates[p]["vgrf"], mass, prate):
                # FULL GAIT CYCLE construction (matches training/Fukuchi: stance bump + swing-zero).
                # Verified directly: training angle/vGRF/moment are ALL full cycles (HS->HS), same
                # phase, vGRF/moment ~0 in swing. So the external CP cycle must be a full cycle too.
                # Heel-strike = stance start (lo). The full stride = lo .. lo + stance_dur/STANCE_FRAC.
                stance_dur = hi - lo
                cyc_len = int(round(stance_dur / STANCE_FRAC))   # full stride length in frames
                ce = lo + cyc_len                                 # cycle end frame
                t0, te_cyc = lo / prate, ce / prate              # absolute seconds of full cycle
                # need IK to cover at least the stance portion of this cycle
                if hi / prate <= ik_t[0] or lo / prate >= ik_t[-1]:
                    continue
                # --- IK quality gate on the STANCE window (where angles must be valid) ---
                sel_stance = (ik_t >= max(lo / prate, ik_t[0])) & (ik_t <= min(hi / prate, ik_t[-1]))
                if sel_stance.sum() < 3:
                    continue
                bad = False
                for coord, lim in (("hip_flexion_r", 90), ("hip_flexion_l", 90),
                                   ("knee_angle_r", 100), ("knee_angle_l", 100),
                                   ("ankle_angle_r", 45), ("ankle_angle_l", 45)):
                    if coord in coords and np.nanmax(np.abs(coords[coord][sel_stance])) > lim:
                        bad = True; break
                if bad:
                    meta_rows.append({"trial": trial, "plate": p, "side": side,
                                      "stance_frames": stance_dur, "status": "IK_DIVERGED"})
                    continue
                cyc = np.full((101, 54), np.nan, np.float32)
                stance_pts = max(2, int(round(101 * STANCE_FRAC)))
                # vGRF: measured stance in [0, stance_pts), swing = 0 in [stance_pts, 101)
                vch = R_VGRF_54 if side == "R" else L_VGRF_54
                full_vgrf = np.zeros(101, np.float32)
                full_vgrf[:stance_pts] = resample101_n(np.clip(vg[lo:hi], 0, None), stance_pts)
                cyc[:, vch] = full_vgrf
                # angles: sample the IK angle over the FULL cycle [t0, te_cyc]; where IK doesn't
                # cover the swing tail, np.interp clamps to the last covered value (acceptable: the
                # eval scores the stance window, and the angle is the model INPUT, not the target).
                ik_cyc_t = np.linspace(t0, te_cyc, 101)
                ik_cyc_t = np.clip(ik_cyc_t, ik_t[0], ik_t[-1])
                for coord, ch in ANG_FROM_IK.items():
                    if coord in coords:
                        cyc[:, ch] = np.interp(ik_cyc_t, ik_t, coords[coord])
                for coord, (rch, lch) in PEL_FROM_IK.items():
                    if coord in coords:
                        seg = np.interp(ik_cyc_t, ik_t, coords[coord])
                        cyc[:, rch] = seg; cyc[:, lch] = seg
                cycles_all.append(cyc)
                meta_rows.append({"trial": trial, "plate": p, "side": side,
                                  "stance_frames": stance_dur, "status": "OK"})
    if not cycles_all:
        print(f"[{subj}] no cycles extracted"); return None
    stack = np.stack(cycles_all)                            # (n,101,54)
    mask = np.isfinite(stack).all(axis=1)                   # (n,54)
    np.save(OUT / f"{subj}.npy", stack.astype(np.float32))
    np.save(OUT / f"{subj}_mask.npy", mask)
    with open(OUT / f"{subj}_meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "trial", "plate", "side", "stance_frames", "status", "group"])
        for i, m in enumerate(meta_rows):
            w.writerow([i, m["trial"], m["plate"], m["side"], m["stance_frames"], m.get("status", "OK"), grp])
    npk = int(mask[:, [R_VGRF_54, L_VGRF_54]].sum())
    print(f"[{subj}] {stack.shape[0]} cycles, mass {mass}kg, vGRF-valid cells {npk}, group {grp}")
    return {"subj": subj, "n_cyc": int(stack.shape[0]), "mass": mass, "group": grp}


def main():
    masses = load_masses()
    subjects = sys.argv[1:] if len(sys.argv) > 1 else None
    if not subjects:
        # all subjects that have IK output
        subjects = sorted({Path(f).stem.replace("_ik", "").rstrip("abc")
                           for f in glob.glob(str(IK / "*_ik.mot"))})
        # normalize to subject ids (TD1a -> TD1, HeCP2a -> HeCP2)
        import re
        subjects = sorted({re.sub(r"[abc]$", "", s) for s in subjects})
    summ = []
    for s in subjects:
        try:
            r = harmonize_subject(s, masses)
            if r:
                summ.append(r)
        except Exception as e:
            print(f"[{s}] FAIL {type(e).__name__}: {e}")
    (OUT / "_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\nHarmonized {len(summ)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
