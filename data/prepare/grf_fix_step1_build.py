"""STEP 1 — build data/cohorts_grffix/ : remap ONLY normal/cp 3 GRF components.

- Original data/cohorts_raw/ is NEVER modified.
- stroke/pd/healthy: every file copied byte-for-byte.
- normal/cp: copy all, then remap ONLY 54-ch channels 34,35,36,37,38,39
  (R/L GRF AP_X/ML_Y/V_Z) per cycle, per side, using Result-TD Stance_P:
  take the stance-stretched GRF (0-100% == stance) and compress into
  [0, stanceP*100], zero-fill the swing tail.
- Cycles where a GRF channel is populated but no stanceP is available are LEFT
  UNCHANGED and logged (must be ~0 per CHECKPOINT-1; abort-worthy if many).
- Records provenance: outputs/grffix/provenance.json (per file/cycle/side stanceP).

Run: python scripts/grf_fix_step1_build.py
"""
import sys
import os, csv, json, glob, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in [ROOT, ROOT / "model"] + sorted(d for d in (ROOT / "scripts").iterdir() if d.is_dir()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from guarded_write import guarded_dump
SRC_RAW = ROOT / "data" / "cohorts_raw"
DST = ROOT / "data" / "cohorts_grffix"
TD_SRC = {"normal": ROOT / "data" / "inhouse" / "normal", "cp": ROOT / "data" / "inhouse" / "cp"}
BS = chr(92)

GRF_CH = [34, 35, 36, 37, 38, 39]           # R/L GRF channels (54-ch)
R_GRF, L_GRF = [34, 35, 36], [37, 38, 39]
R_VGRF, L_VGRF = 36, 39
FIX_COHORTS = ["normal", "cp"]
COPY_COHORTS = ["vdk_stroke", "bmclab_pd", "vdk_healthy"]


def parse_td(path):
    L = open(path, encoding="latin-1").read().split("\n")
    c3d = L[0].split("\t"); var = L[1].split("\t"); cm = {}
    for j, (t, v) in enumerate(zip(c3d, var)):
        if t.strip().lower().endswith(".c3d"):
            cm[j] = (os.path.basename(t.strip().replace(BS, "/")), v.strip())
    out = {}
    for ln in L[5:]:
        c = ln.split("\t")
        if not c or not c[0].strip().isdigit():
            continue
        it = int(c[0]) - 1
        for j, (bn, vn) in cm.items():
            if vn in ("Left_Stance_P", "Right_Stance_P") and j < len(c):
                try:
                    val = float(c[j])
                    if not np.isfinite(val):
                        val = None
                except ValueError:
                    val = None
                out.setdefault(bn, {}).setdefault(it, {})["L" if vn[0] == "L" else "R"] = val
    return out


def subject_dir_for_stem(cohort, stem):
    rows = list(csv.DictReader(open(SRC_RAW / cohort / f"{stem}_meta.csv")))
    want = set(r["trial"] for r in rows)
    best, best_n = None, -1
    for d in sorted(os.listdir(TD_SRC[cohort])):
        td = TD_SRC[cohort] / d / "Result-TD.txt"
        if td.exists():
            n = len(want & set(parse_td(td).keys()))
            if n > best_n:
                best, best_n = d, n
    return best


def compress(w, sp):
    """w: (101,) stance-stretched (0-100% == stance). Return (101,) compressed
    into [0, round(sp*100)] with zero swing tail. Returns None if sp invalid."""
    if sp is None:
        return None
    to = int(round(sp * 100))
    if to < 5 or to > 100:
        return None
    comp = np.interp(np.linspace(0, 1, to + 1), np.linspace(0, 1, 101), np.nan_to_num(w))
    out = np.zeros(101, dtype=w.dtype)
    out[:to + 1] = comp
    return out


def main():
    missing = [str(p) for p in [SRC_RAW] + [TD_SRC[c] for c in FIX_COHORTS] if not p.exists()]
    if missing:
        raise SystemExit("grf_fix_step1_build: required input tree not found: " + ", ".join(missing)
                         + ". Nothing was deleted and nothing was written.")
    out = DST.with_name(DST.name + ".building")   # existing DST is untouched until the build succeeds
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    prov = {}
    log = {"copied_files": 0, "fixed_files": 0, "cycles_fixed": 0,
           "cycles_grf_no_stanceP": 0, "no_stanceP_detail": []}

    # 1) copy untouched cohorts verbatim (all files)
    for c in COPY_COHORTS:
        (out / c).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(SRC_RAW / c / "*")):
            shutil.copy2(f, out / c / os.path.basename(f))
            log["copied_files"] += 1

    # 2) normal/cp — copy non-.npy verbatim, remap GRF in the data .npy
    for c in FIX_COHORTS:
        (out / c).mkdir(parents=True, exist_ok=True)
        stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(SRC_RAW / c / "*.npy"))
                       if "_mask" not in f and "_meta" not in f)
        # copy mask + meta verbatim; data .npy will be written remapped
        for f in glob.glob(str(SRC_RAW / c / "*")):
            bn = os.path.basename(f)
            if bn.endswith("_mask.npy") or bn.endswith("_meta.csv") or not bn.endswith(".npy"):
                shutil.copy2(f, out / c / bn)
                log["copied_files"] += 1
        for stem in stems:
            arr = np.load(SRC_RAW / c / f"{stem}.npy")
            new = arr.copy()
            mask = np.load(SRC_RAW / c / f"{stem}_mask.npy")
            meta = list(csv.DictReader(open(SRC_RAW / c / f"{stem}_meta.csv")))
            d = subject_dir_for_stem(c, stem)
            if d is None:
                raise SystemExit(f"grf_fix_step1_build: no Result-TD.txt found under {TD_SRC[c]}, so no "
                                 f"cycle of cohort {c} can be compressed into its stance fraction. "
                                 f"Aborting, {DST} is unchanged and {out} is incomplete.")
            td = parse_td(TD_SRC[c] / d / "Result-TD.txt")
            file_prov = []
            for ci, row in enumerate(meta):
                rec = td.get(row["trial"], {}).get(int(row["cycle_idx_in_trial"]))
                for side, gchs, vch in [("R", R_GRF, R_VGRF), ("L", L_GRF, L_VGRF)]:
                    if not (mask[ci, vch] and np.isfinite(arr[ci, :, vch]).any()
                            and np.nanmax(np.abs(arr[ci, :, vch])) > 1e-6):
                        continue  # this side's GRF not populated -> nothing to fix
                    sp = rec.get(side) if rec else None
                    if sp is None:
                        log["cycles_grf_no_stanceP"] += 1
                        log["no_stanceP_detail"].append(f"{c}/{stem} cyc{ci} side{side} trial={row['trial']}")
                        file_prov.append({"cyc": ci, "side": side, "stanceP": None, "fixed": False})
                        continue
                    ok = True
                    for gch in gchs:
                        cw = compress(arr[ci, :, gch].astype(np.float32), sp)
                        if cw is None:
                            ok = False; break
                        new[ci, :, gch] = cw
                    if ok:
                        log["cycles_fixed"] += 1
                        file_prov.append({"cyc": ci, "side": side, "stanceP": round(float(sp), 4), "fixed": True})
                    else:
                        log["cycles_grf_no_stanceP"] += 1
                        file_prov.append({"cyc": ci, "side": side, "stanceP": sp, "fixed": False})
            np.save(out / c / f"{stem}.npy", new)
            log["fixed_files"] += 1
            prov[f"{c}/{stem}"] = {"td_dir": d, "cycles": file_prov}

    guarded_dump(prov, ROOT / "outputs/grffix/provenance.json", "grf_fix_step1_build.py")
    guarded_dump(log, ROOT / "outputs/grffix/build_log.json", "grf_fix_step1_build.py")
    print(json.dumps({k: v for k, v in log.items() if k != "no_stanceP_detail"}, indent=1))
    if log["no_stanceP_detail"]:
        print(f"\nWARNING: {len(log['no_stanceP_detail'])} GRF cycles had NO stanceP (left UNCHANGED):")
        for x in log["no_stanceP_detail"][:30]:
            print("  ", x)
    n_eligible = log["cycles_fixed"] + log["cycles_grf_no_stanceP"]
    if log["cycles_grf_no_stanceP"] > max(10, 0.05 * n_eligible):
        raise SystemExit(f"grf_fix_step1_build: {log['cycles_grf_no_stanceP']} of {n_eligible} eligible "
                         f"GRF cycles had no Stance_P and keep the stance-stretched convention. "
                         f"{out} was left for inspection and {DST} is unchanged. Check that the Result-TD.txt trial names match the "
                         f"trial column of the *_meta.csv files, then rebuild.")
    if DST.exists():
        shutil.rmtree(DST)
    out.rename(DST)
    print(f"\nbuilt {DST}")


if __name__ == "__main__":
    main()
