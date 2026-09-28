"""Harmonize Fukuchi 2018 WBDS OVERGROUND healthy (Visual3D-processed ASCII) -> PathoGaitFM 54-ch raw.

Independent-dataset, independent-pipeline (Visual3D) external zero-shot validation of vertical GRF
and sagittal joint moments. Input = the dataset's OWN processed angles/moments/GRF (NOT our IK/ID). Cycle-normalized
to 101 pts per subject-speed (S/C/F) by Visual3D.

Cross-pipeline axis/unit/sign mappings (ALL determined empirically vs TRAINING 'normal' ref;
see _axis_mapping.json / _kinetics_mapping.json / _grf_mapping_resolved.json / _vgrf_shape_diag.json):

  ANGLES (input 16ch + pelvis): Fukuchi sagittal is on Z (not X). hip/knee/ankle sagittal <- *AngleZ,
    sign +1 (cohort PCC 0.96-0.999). Units deg, pass-through.
  MOMENTS (target): sagittal on Z. hip <- *HipMomentZ sign -1 ; knee <- *KneeMomentZ sign +1 ;
    ankle <- *AnkleMomentZ sign +1. Units already Nm/kg (Visual3D normalized). Between-lab amplitude
    difference (~0.6-0.9x) is left UNCORRECTED (it is a real transfer signal, not a unit bug).
  GRF (target): Fukuchi Y=vertical, X=AP, Z=ML. -> my Z=vert, X=AP, Y=ML. Units N/kg -> /9.80665 = BW.
    Vertical is swing-zeroed (stance_frac 0.63, physiologically correct, matches vdk_stroke).

IRON RULES: zero-shot; NO normalization fit on this site (raw physical units only; z-score later via
training global_zstats.json); NO target used to set INPUT polarity (input signs fixed from angle
ref only); missing -> NaN + mask False; healthy population flagged separately.

WARNING (documented in report): training normal/cp GRF cycle convention is inconsistent (stance_frac
~1.0, not swing-zeroed) whereas Fukuchi + vdk_stroke are correct. We harmonize Fukuchi to the correct
convention and do NOT alter training data. vGRF external PCC is interpreted with this caveat.

Output: processed/ID##_<speed>.npy (n_cyc,101,54) raw + _mask.npy (n_cyc,54) + _meta.csv.
But the 126 ang/knt files are ALREADY one averaged cycle per subject-speed -> each yields 1 'cycle'.
We treat each (subject,speed) as one representative cycle; per subject we stack its 3 speeds (S/C/F)
=> up to 3 cycles/subject. This matches how the dataset presents processed kinetics.
"""
from __future__ import annotations
import csv, glob, json, re, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]            # fukuchi_healthy -> external -> data -> the package root
PROC_IN = HERE / "raw" / "ascii_proc"
OUT = HERE / "processed"
INFO = HERE / "raw" / "WBDSinfo.xlsx"
G = 9.80665

# ---- Fukuchi column -> 54-ch idx, with sign ----
# ANGLES (sagittal on Z): my 54-ch sagittal = 0,3 hip; 6,9 knee; 12,15 ankle
ANG_MAP = {  # (fukuchi_col, sign) -> 54ch
    ("RHipAngleZ", 1): 0, ("LHipAngleZ", 1): 3,
    ("RKneeAngleZ", 1): 6, ("LKneeAngleZ", 1): 9,
    ("RAnkleAngleZ", 1): 12, ("LAnkleAngleZ", 1): 15,
}
# Non-sagittal hip Y/Z (my 1,2,4,5) and knee/ankle Y/Z (my 7,8,10,11,13,14,16,17) are NOT in the
# 16ch known set and not targets; fill where a clear Fukuchi axis exists, else leave NaN. For the
# 16ch zero-shot only sagittal angles + pelvis are used, so we fill sagittal + pelvis robustly.
# PELVIS: my 48-53 = R/L pelvis X,Y,Z. Fukuchi RPelvisAngle X,Y,Z + L.
PEL_MAP = {
    ("RPelvisAngleX", 1): 48, ("RPelvisAngleY", 1): 49, ("RPelvisAngleZ", 1): 50,
    ("LPelvisAngleX", 1): 51, ("LPelvisAngleY", 1): 52, ("LPelvisAngleZ", 1): 53,
}
# MOMENTS (sagittal on Z): my 18,21 hip; 24,25 knee; 26,27 ankle. hip sign -1.
MOM_MAP = {
    ("RHipMomentZ", -1): 18, ("LHipMomentZ", -1): 21,
    ("RKneeMomentZ", 1): 24, ("LKneeMomentZ", 1): 25,
    ("RAnkleMomentZ", 1): 26, ("LAnkleMomentZ", 1): 27,
}
# GRF: Fukuchi Y=vert->my Z(36/39); X=AP->my X(34/37); Z=ML->my Y(35/38). /G -> BW.
GRF_MAP = {  # (fukuchi_col, sign, scale) -> 54ch
    ("RGRFX", 1, 1 / G): 34, ("RGRFZ", 1, 1 / G): 35, ("RGRFY", 1, 1 / G): 36,
    ("LGRFX", 1, 1 / G): 37, ("LGRFZ", 1, 1 / G): 38, ("LGRFY", 1, 1 / G): 39,
}
# sagittal moment 54-ch idx by side (for stance-peak validity) + GRF idx by side
R_SAGMOM = [18, 24, 26]; L_SAGMOM = [21, 25, 27]
R_GRF = [34, 35, 36]; L_GRF = [37, 38, 39]
STANCE = slice(0, 61)
# Fifth percentile of the per-cycle stance peak of each sagittal moment channel in the training healthy
# cohort, {"18": float, "21": float, "24": ..., "27": ...} in Nm/kg. It is derived from the restricted
# in-house data, so it is not distributed; build_train_healthy_ref.py writes it from a local copy.
PEAK_P5_DEFAULT = HERE / "_train_normal_peak_p5.json"


def load_peak_p5(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. It holds the training-cohort stance-peak thresholds used to mark unsupported "
            "moment cycles and is built locally with build_train_healthy_ref.py; pass --peak-p5 to use another location.")
    return json.loads(path.read_text())


def load_txt(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    hdr = lines[0].split("\t")
    d = np.array([[float(v) for v in ln.split("\t")] for ln in lines[1:] if ln.strip()])
    return {n: d[:, i] for i, n in enumerate(hdr)}


def load_masses():
    import openpyxl
    wb = openpyxl.load_workbook(INFO, data_only=True)
    ws = wb["Planilha1"]; out = {}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        try:
            out[int(str(r[0]).strip())] = float(r[5])
        except Exception:
            pass
    return out


def main():
    global PROC_IN, OUT, INFO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascii-dir", type=Path, default=PROC_IN)
    parser.add_argument("--info", type=Path, default=INFO)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--peak-p5", type=Path, default=PEAK_P5_DEFAULT,
                        help="training-cohort stance-peak thresholds (built locally, see the site README)")
    args = parser.parse_args()
    PROC_IN, INFO, OUT = args.ascii_dir, args.info, args.out
    if not PROC_IN.is_dir() or not INFO.is_file():
        raise FileNotFoundError("Supply WBDS overground ASCII and WBDSinfo.xlsx using --ascii-dir and --info")
    peak_p5 = load_peak_p5(args.peak_p5)
    OUT.mkdir(parents=True, exist_ok=True)
    masses = load_masses()
    ang_files = sorted(glob.glob(str(PROC_IN / "WBDS*walkO*ang.txt")))
    # group by subject -> list of (speed, ang_file, knt_file)
    by_subj = {}
    for af in ang_files:
        m = re.search(r"WBDS(\d+)walkO([SCF])ang\.txt", Path(af).name)
        if not m:
            continue
        sid = int(m.group(1)); speed = m.group(2)
        kf = af.replace("ang.txt", "knt.txt")
        if Path(kf).exists():
            by_subj.setdefault(sid, []).append((speed, af, kf))
        else:
            raise FileNotFoundError(f"Missing paired kinetics file: {kf}")

    if not by_subj:
        raise ValueError("No WBDS overground angle/kinetics pairs found")
    if any(sid not in masses or not np.isfinite(masses[sid]) or masses[sid] <= 0 for sid in by_subj):
        raise ValueError("Missing or invalid body mass for a subject; check WBDSinfo.xlsx")

    summary = []
    for sid in sorted(by_subj):
        mass = masses.get(sid)
        cycles = []; speeds = []
        for speed, af, kf in sorted(by_subj[sid]):
            ang = load_txt(af); knt = load_txt(kf)
            n = 101
            cyc = np.full((n, 54), np.nan)
            # angles + pelvis
            for (col, sgn), ch in {**ANG_MAP, **PEL_MAP}.items():
                if col in ang and ang[col].shape[0] == n:
                    cyc[:, ch] = sgn * ang[col]
            # moments
            for (col, sgn), ch in MOM_MAP.items():
                if col in knt and knt[col].shape[0] == n:
                    cyc[:, ch] = sgn * knt[col]
            # GRF (scale to BW)
            for (col, sgn, sc), ch in GRF_MAP.items():
                if col in knt and knt[col].shape[0] == n:
                    cyc[:, ch] = sgn * sc * knt[col]
            cycles.append(cyc); speeds.append(speed)
        if not cycles:
            continue
        stack = np.stack(cycles)                       # (n_speed,101,54)
        mask = np.isfinite(stack).all(axis=1)          # (n_speed,54)

        # moment stance-peak validity (drop swing-only/unsupported) — both feet present per cycle here
        def stance_peak(i, ch):
            seg = stack[i, STANCE, ch]
            return np.nanmax(np.abs(seg)) if np.isfinite(seg).any() else 0.0
        for i in range(stack.shape[0]):
            for ch in R_SAGMOM + L_SAGMOM:
                if not (stance_peak(i, ch) >= peak_p5.get(str(ch), 0.0)):
                    mask[i, ch] = False
            # GRF validity: keep if ipsilateral ankle stance moment real (contact witness)
            if not mask[i, 26]:
                mask[i, R_GRF] = False
            if not mask[i, 27]:
                mask[i, L_GRF] = False

        np.save(OUT / f"ID{sid:02d}.npy", stack.astype(np.float32))
        np.save(OUT / f"ID{sid:02d}_mask.npy", mask)
        with open(OUT / f"ID{sid:02d}_meta.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["cycle_idx", "group", "n_valid_channels", "subject_id", "speed"])
            for i in range(stack.shape[0]):
                w.writerow([i, "fukuchi_healthy", int(mask[i].sum()), f"ID{sid:02d}", speeds[i]])
        summary.append(dict(sid=f"ID{sid:02d}", n_cyc=stack.shape[0], mass=mass, speeds=speeds,
                            valid_ang=int(mask[:, :18].sum()), valid_mom=int(mask[:, 18:28].sum()),
                            valid_grf=int(mask[:, 34:40].sum())))
        print(f"[ID{sid:02d}] speeds={speeds} mom_cells={int(mask[:,18:28].sum())} grf_cells={int(mask[:,34:40].sum())}", flush=True)

    (OUT / "_harmonize_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    # also persist the mapping decisions for the report
    (OUT / "_mappings_used.json").write_text(json.dumps({
        "angles": {f"{k[0]}(sign{k[1]})": v for k, v in ANG_MAP.items()},
        "pelvis": {f"{k[0]}(sign{k[1]})": v for k, v in PEL_MAP.items()},
        "moments": {f"{k[0]}(sign{k[1]})": v for k, v in MOM_MAP.items()},
        "grf": {f"{k[0]}(sign{k[1]},/G)": v for k, v in GRF_MAP.items()},
        "notes": "sagittal angles/moments on Fukuchi Z; hip moment sign -1; GRF Y=vert X=AP Z=ML, N/kg->BW /9.80665",
    }, indent=2))
    print(f"\nWrote {len(summary)} subjects to {OUT}")


if __name__ == "__main__":
    main()
