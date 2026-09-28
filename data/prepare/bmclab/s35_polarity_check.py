"""Polarity check (Option E — stance-peak detection).

Per-cycle peak detection in the actual stance window, sidestepping
healthy-gait fixed-phase-of-cycle assumptions that don't hold in PD slow
gait (stance can be 70-80% of cycle).

Per cycle (one contiguous stance window per side per trial):
  - Ankle PF peak       = min(ankle_moment) in stance — expected NEGATIVE
  - Knee ext peak       = max(knee_moment) in early-stance third — expected POSITIVE
  - Hip flex peak       = max(hip_flex_moment) in late-stance third — expected POSITIVE
  - Ankle peak |moment| = max(|ankle_moment|) in stance — magnitude check

Stance windows are detected from `data/prepare/bmclab/work/grf/_grf_presence/{trial}.csv`
(per-side boolean at 150 Hz). Fix 2 (trim to IK time range) still applies.
Time-based interpolation handles the ID sample-rate mismatch (~151.5 Hz).

Per-subject aggregate medians evaluated against:
  - ankle PF peak       <  -0.4 N·m/kg
  - knee ext peak       >  +0.2 N·m/kg
  - hip flex peak       >  +0.2 N·m/kg
  - ankle |peak|/mass   in [0.2, 2.5] N·m/kg
"""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","SIMTK_NUM_THREADS"):
    os.environ[k] = "1"

import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
ID_DIR = PROJ / "data/prepare/bmclab/work/id"
IK_DIR = PROJ / "data/prepare/bmclab/work/ik"
PRESENCE_DIR = PROJ / "data/prepare/bmclab/work/grf/_grf_presence"
MASK_PATH = PROJ / "data/prepare/bmclab/work/grf/_grf_mask.json"
PDG = pd.read_excel(PROJ / "data/prepare/bmclab/raw/PDGinfo.xlsx", sheet_name="PDGinfo", header=0)
def mass_of(sid): return float(PDG[PDG["ID"] == sid]["Weight (kg)"].iloc[0])


def read_id_sto(sto_path: Path):
    lines = sto_path.read_text().splitlines()
    end = next((i for i, L in enumerate(lines) if L.strip() == "endheader"), None)
    if end is None: return None, None
    cols = lines[end+1].split()
    data_rows = []
    for L in lines[end+2:]:
        L = L.strip()
        if not L: continue
        parts = L.split()
        if len(parts) != len(cols): continue
        try: data_rows.append([float(x) for x in parts])
        except ValueError: continue
    if not data_rows: return None, None
    data = np.asarray(data_rows)
    time = data[:, cols.index("time")]
    out = {c: data[:, cols.index(c)] for c in cols}
    return time, out


def ik_time_range(ik_mot_path: Path):
    lines = ik_mot_path.read_text().splitlines()
    end = next((i for i, L in enumerate(lines) if L.strip() == "endheader"), None)
    if end is None: return None
    cols = lines[end+1].split()
    t_idx = cols.index("time")
    t0 = None; t1 = None
    for L in lines[end+2:]:
        L = L.strip()
        if not L: continue
        parts = L.split()
        if len(parts) != len(cols): continue
        try: t = float(parts[t_idx])
        except ValueError: continue
        if t0 is None: t0 = t
        t1 = t
    return (t0, t1)


def trim_id_to_ik_window(time, cols_dict, ik_t0, ik_t1, p_rate=150):
    if time is None: return None, None
    eps = 0.5 / p_rate
    mask = (time >= ik_t0 - eps) & (time <= ik_t1 + eps)
    if mask.sum() == 0: return None, None
    new_time = time[mask] - ik_t0
    new_cols = {k: v[mask] for k, v in cols_dict.items()}
    return new_time, new_cols


def detect_stance_windows(presence_col):
    """Given a 1D boolean array (1 = GRF present), return list of (start_idx, end_idx)
    inclusive, for each contiguous True run."""
    a = np.asarray(presence_col, dtype=bool)
    if not a.any(): return []
    # edges
    padded = np.concatenate([[False], a, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def stance_peak_metrics(t_id, ankle_sig, knee_sig, hip_sig, stance_t0, stance_t1):
    """Time-based interpolation onto a dense grid covering [stance_t0, stance_t1]
    at 150 Hz; compute the 4 peaks."""
    if stance_t1 <= stance_t0: return None
    dt = 1.0 / 150
    n = max(2, int(round((stance_t1 - stance_t0) / dt)) + 1)
    grid = np.linspace(stance_t0, stance_t1, n)
    # Clip to t_id support
    inside = (grid >= t_id[0]) & (grid <= t_id[-1])
    grid = grid[inside]
    if len(grid) < 5: return None
    ank = np.interp(grid, t_id, ankle_sig)
    kne = np.interp(grid, t_id, knee_sig)
    hip = np.interp(grid, t_id, hip_sig)
    nframes = len(grid)
    early = slice(0, max(1, nframes // 3))
    late = slice(-max(1, nframes // 3), None)
    return {
        "ankle_pf_peak":  float(ank.min()),               # most negative
        "knee_ext_peak":  float(kne[early].max()),         # most positive in early stance
        "hip_flex_peak":  float(hip[late].max()),          # most positive in late stance
        "ankle_abs_peak": float(np.max(np.abs(ank))),
        "stance_dur_s":   float(stance_t1 - stance_t0),
        "n_frames":       nframes,
    }


def main():
    with open(MASK_PATH, "r") as f:
        mask_map = json.load(f)
    pass_trials = sorted([t for t, m in mask_map.items() if m == "pass"])
    print(f"'pass' trials: {len(pass_trials)}")

    results = []
    n_processed = 0; n_skipped_no_presence = 0
    for trial in pass_trials:
        id_sto = ID_DIR / f"{trial}_id.sto"
        ik_mot = IK_DIR / f"{trial}_ik.mot"
        presence_csv = PRESENCE_DIR / f"{trial}.csv"
        if not id_sto.is_file() or not ik_mot.is_file():
            continue
        if not presence_csv.is_file():
            n_skipped_no_presence += 1; continue
        t, cols = read_id_sto(id_sto)
        if t is None: continue
        ik_rng = ik_time_range(ik_mot)
        if ik_rng is None: continue
        t, cols = trim_id_to_ik_window(t, cols, ik_rng[0], ik_rng[1])
        if t is None: continue
        # subject + mass
        sub = trial.split("_")[1] if trial.startswith("Trimmed_") else trial.split("_")[0]
        mass = mass_of(sub)
        # Presence
        pres = pd.read_csv(presence_csv)
        t_pres = pres["time"].to_numpy()
        for side in ("r", "l"):
            ankle_col = f"ankle_angle_{side}_moment"
            knee_col  = f"knee_angle_{side}_moment"
            hip_col   = f"hip_flexion_{side}_moment"
            if ankle_col not in cols or knee_col not in cols or hip_col not in cols:
                continue
            pres_col = pres[f"{side}_grf_present"].to_numpy()
            windows = detect_stance_windows(pres_col)
            for w_idx, (i0, i1) in enumerate(windows):
                stance_t0 = float(t_pres[i0])
                stance_t1 = float(t_pres[i1])
                dur = stance_t1 - stance_t0
                if dur < 0.2 or dur > 2.0: continue   # implausible
                m = stance_peak_metrics(t, cols[ankle_col], cols[knee_col], cols[hip_col], stance_t0, stance_t1)
                if m is None: continue
                results.append({
                    "trial": trial,
                    "subject": sub,
                    "mass_kg": mass,
                    "side": side,
                    "window_idx": w_idx,
                    "stance_t0": stance_t0, "stance_t1": stance_t1,
                    "stance_dur_s": m["stance_dur_s"],
                    "ankle_pf_peak_Nm":  m["ankle_pf_peak"],
                    "knee_ext_peak_Nm":  m["knee_ext_peak"],
                    "hip_flex_peak_Nm":  m["hip_flex_peak"],
                    "ankle_abs_peak_Nm": m["ankle_abs_peak"],
                    "ankle_pf_peak_per_kg":  m["ankle_pf_peak"]  / mass,
                    "knee_ext_peak_per_kg":  m["knee_ext_peak"]  / mass,
                    "hip_flex_peak_per_kg":  m["hip_flex_peak"]  / mass,
                    "ankle_abs_peak_per_kg": m["ankle_abs_peak"] / mass,
                })
        n_processed += 1

    print(f"Processed {n_processed} trials, skipped {n_skipped_no_presence} (no presence file)")
    print(f"Total stance-window samples: {len(results)} (R+L combined)\n")

    def arr(key): return np.asarray([r[key] for r in results if r.get(key) is not None])

    ankle_pf = arr("ankle_pf_peak_per_kg")
    knee_ext = arr("knee_ext_peak_per_kg")
    hip_flex = arr("hip_flex_peak_per_kg")
    ankle_abs = arr("ankle_abs_peak_per_kg")

    def cohort_stats(a, name, expected_sign, threshold=None, mag_range=None):
        if a.size == 0:
            print(f"  {name}: no data"); return None
        med = float(np.median(a))
        q25, q75 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
        passes = True
        msg = []
        if expected_sign == "negative":
            ok = med < (threshold if threshold is not None else 0)
            msg.append(f"sign={'✅' if med < 0 else '❌'}")
            if threshold is not None: msg.append(f"(< {threshold} {'✅' if ok else '❌'})")
            passes = ok
        elif expected_sign == "positive":
            ok = med > (threshold if threshold is not None else 0)
            msg.append(f"sign={'✅' if med > 0 else '❌'}")
            if threshold is not None: msg.append(f"(> {threshold} {'✅' if ok else '❌'})")
            passes = ok
        elif expected_sign == "magnitude":
            ok = mag_range[0] <= med <= mag_range[1]
            msg.append(f"in {mag_range} {'✅' if ok else '❌'}")
            passes = ok
        print(f"  {name}: median={med:+.3f} N·m/kg  IQR=[{q25:+.3f}, {q75:+.3f}]  n={a.size}  → {' '.join(msg)}")
        return passes

    print("=== Cohort-aggregate polarity check (Option E) ===\n")
    print("Check 1 — Ankle PF peak in stance")
    cohort_stats(ankle_pf, "ankle_pf_peak", "negative", threshold=-0.4)
    print("\nCheck 2 — Knee ext peak in early-stance third")
    cohort_stats(knee_ext, "knee_ext_peak", "positive", threshold=+0.2)
    print("\nCheck 3 — Hip flex peak in late-stance third")
    cohort_stats(hip_flex, "hip_flex_peak", "positive", threshold=+0.2)
    print("\nCheck 4 — Ankle peak |moment|/mass magnitude range")
    cohort_stats(ankle_abs, "|ankle peak|", "magnitude", mag_range=(0.2, 2.5))

    # Per-subject medians (the real pass gates)
    print("\n=== Per-subject medians ===\n")
    per_sub = defaultdict(list)
    for r in results:
        per_sub[r["subject"]].append(r)
    subjects = sorted(per_sub.keys())
    headers = f"{'sub':<6} {'n':>4}  {'ankle_PF':>10}  {'knee_ext':>10}  {'hip_flex':>10}  {'|ankle|':>8}  pass?"
    print(headers)
    print("-" * len(headers))

    n_pass_subjects = 0
    pass_per_subject = {}
    for sub in subjects:
        rs = per_sub[sub]
        ankle_med = float(np.median([r["ankle_pf_peak_per_kg"]  for r in rs]))
        knee_med  = float(np.median([r["knee_ext_peak_per_kg"]  for r in rs]))
        hip_med   = float(np.median([r["hip_flex_peak_per_kg"]  for r in rs]))
        abs_med   = float(np.median([r["ankle_abs_peak_per_kg"] for r in rs]))
        passes = (ankle_med < -0.4) and (knee_med > 0.2) and (hip_med > 0.2) and (0.2 <= abs_med <= 2.5)
        # Soft pass for magnitude — allow being above 2.5 if the others all pass
        per_check = {
            "ankle_PF<-0.4": ankle_med < -0.4,
            "knee_ext>+0.2": knee_med > 0.2,
            "hip_flex>+0.2": hip_med > 0.2,
            "|ankle|∈[0.2,2.5]": (0.2 <= abs_med <= 2.5),
        }
        pass_per_subject[sub] = per_check
        if passes: n_pass_subjects += 1
        ok = "✅" if passes else "❌"
        print(f"{sub:<6} {len(rs):>4}  {ankle_med:>+10.3f}  {knee_med:>+10.3f}  {hip_med:>+10.3f}  {abs_med:>+8.3f}  {ok}")

    print(f"\nSubjects passing all 4 checks: {n_pass_subjects} / {len(subjects)}")

    # Which check fails for which subject
    print("\nPer-subject check breakdown (only failures shown):")
    for sub, pc in pass_per_subject.items():
        fails = [k for k, v in pc.items() if not v]
        if fails:
            print(f"  {sub}: fails {fails}")

    # Write CSV
    out_csv = ID_DIR / "_polarity_check_stance_peak.csv"
    if results:
        fields = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results: w.writerow(r)
        print(f"\nWrote {out_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
