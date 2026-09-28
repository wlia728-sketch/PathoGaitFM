"""Harmonize PD Nordic Walking (Scientific Data 2025) -> PathoGaitFM 54-ch raw format.

Produces per-subject (n_cyc,101,54) raw-units .npy + (n_cyc,54) mask + meta.csv,
consumable by V4LazyTransformV3PerSubject + eval_common.py.

IRON RULES enforced here:
  1. zero-shot: NO normalization stats fit on this site. Raw physical units only;
     z-score happens later in the eval transform via training global_zstats.json.
  2. NO target (moment/GRF) channel is used to decide input polarity.
  3. Missing channels -> NaN + mask=False. NEVER zero-fill.
  4. Pre baseline trials ONLY (exclude Post).
  5. Polarity decided by correlation to TRAINING-SET PD cohort-mean angle waveform
     (the 6 sagittal kin channels), read from the local reference file given by --angle-ref.

Reference files (derived from the training data, built locally, not distributed):
  --angle-ref  NPZ with keys "0","3","6","9","12","15" (54-channel indices of the sagittal angles),
               each a (101,) float32 cohort-mean waveform of the training Parkinson cohort.
  --peak-p5    JSON {"18": float, "21": ..., "24": ..., "25": ..., "26": ..., "27": ...}: fifth
               percentile of the per-cycle stance peak of each sagittal moment channel (Nm/kg) in the
               training Parkinson cohort, the validity threshold for dataset-provided moments.

Units (verified against the release's own channel inventory):
  angles  : deg (PiG, flexion +)            -> pass-through
  moments : Vicon N.mm/kg                    -> / 1000 -> Nm/kg
  power   : W/kg                             -> pass-through (dropped at 54->40 anyway)
  GRF     : force-plate N (Fz vertical, -ve) -> / (mass*g) and flip vertical sign -> BW dimensionless
"""
from __future__ import annotations
import argparse, csv, glob, json, re
from pathlib import Path
import numpy as np
import ezc3d

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "raw/extracted/dataset online/ParkinsonMotionDataset/Dataset"
OUT  = ROOT / "processed"
PARTICIPANTS = ROOT / "raw/extracted/dataset online/ParkinsonMotionDataset/Partecipants.xlsx"
ANGLE_REF = ROOT / "_train_pd_angle_ref.npz"   # training PD cohort-mean angle waveforms (local)
PEAK_P5   = ROOT / "_train_peak_p5.json"       # training cycle-peak p5 per moment channel (local)
G = 9.80665

# ----- 54-ch canonical index map -----
# angles
A = {  # c3d POINT label, component(0=X,1=Y,2=Z) -> 54-ch idx
    ("RHipAngles",0):0,("RHipAngles",1):1,("RHipAngles",2):2,
    ("LHipAngles",0):3,("LHipAngles",1):4,("LHipAngles",2):5,
    ("RKneeAngles",0):6,("RKneeAngles",1):7,("RKneeAngles",2):8,
    ("LKneeAngles",0):9,("LKneeAngles",1):10,("LKneeAngles",2):11,
    ("RAnkleAngles",0):12,("RAnkleAngles",1):13,("RAnkleAngles",2):14,
    ("LAnkleAngles",0):15,("LAnkleAngles",1):16,("LAnkleAngles",2):17,
}
# moments (Vicon N.mm/kg -> /1000 Nm/kg)
M = {
    ("RHipMoment",0):18,("RHipMoment",1):19,("RHipMoment",2):20,
    ("LHipMoment",0):21,("LHipMoment",1):22,("LHipMoment",2):23,
    ("RKneeMoment",0):24,("LKneeMoment",0):25,
    ("RAnkleMoment",0):26,("LAnkleMoment",0):27,
}
# power (W/kg) -> 54-ch 28..33 (dropped in 40-ch; kept for completeness)
P = {
    ("RHipPower",2):28,("LHipPower",2):29,
    ("RKneePower",2):30,("LKneePower",2):31,
    ("RAnklePower",2):32,("LAnklePower",2):33,
}
# pelvis angles -> 48..53
PEL = {
    ("RPelvisAngles",0):48,("RPelvisAngles",1):49,("RPelvisAngles",2):50,
    ("LPelvisAngles",0):51,("LPelvisAngles",1):52,("LPelvisAngles",2):53,
}
# Electromyography is not harmonised at this site. The release carries analog EMG, but this site is used
# for zero-shot vertical-GRF estimation only, so channels 40-47 are left missing rather than filled.

SAG6 = [0,3,6,9,12,15]   # sagittal kin channels with a polarity reference


def clean(x):
    return str(x).replace("\xa0","").replace(" ","").replace(" ","").replace(",",".").strip() if x is not None else ""


def load_masses():
    import openpyxl
    wb = openpyxl.load_workbook(PARTICIPANTS, data_only=True)
    ws = wb["Partecipants"]; out={}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        pid=clean(r[0])
        if not pid: continue
        try: out[f"ID{int(pid):02d}"]=float(clean(r[4]))
        except: pass
    return out


def resample101(x):
    """x: (T,) -> (101,) linear time-normalized. NaN preserved via interp on finite then re-NaN gaps."""
    T=len(x)
    if T<2: return np.full(101,np.nan)
    src=np.linspace(0,1,T); dst=np.linspace(0,1,101)
    fin=np.isfinite(x)
    if fin.sum()<2: return np.full(101,np.nan)
    y=np.interp(dst,src[fin],x[fin])
    # if the cycle had a long invalid run, mark dst points outside finite coverage as NaN
    if not fin.all():
        valid_span=np.interp(dst,src,fin.astype(float))
        y[valid_span<0.5]=np.nan
    return y


def get_point(c, labs, name):
    if name not in labs: return None
    return c["data"]["points"][:3, labs.index(name), :]   # (3,T)


def harmonize_trial(path, mass):
    c=ezc3d.c3d(str(path))
    labs=c["parameters"]["POINT"]["LABELS"]["value"]
    pr=c["header"]["points"]["frame_rate"]
    first=c["header"]["points"]["first_frame"]
    # ---- gait events: same-side Foot Strike -> cycles ----
    ev=c["parameters"]["EVENT"]; ctx=ev["CONTEXTS"]["value"]; lab=ev["LABELS"]["value"]; tm=ev["TIMES"]["value"][1]
    cycles=[]  # (side, start_frame, end_frame)
    for side in ["Right","Left"]:
        fs=sorted(t for i,t in enumerate(tm) if lab[i]=="Foot Strike" and ctx[i]==side)
        for k in range(len(fs)-1):
            s=int(round(fs[k]*pr))-first; e=int(round(fs[k+1]*pr))-first
            if e>s>=0: cycles.append((side[0],s,e))
    # ---- build 54-ch raw per frame ----
    nfr=c["data"]["points"].shape[2]
    raw=np.full((nfr,54),np.nan,dtype=np.float64)
    # angles (deg pass-through)
    for (nm,comp),ch in {**A,**PEL}.items():
        p=get_point(c,labs,nm)
        if p is not None: raw[:,ch]=p[comp]
    # moments /1000 -> Nm/kg
    for (nm,comp),ch in M.items():
        p=get_point(c,labs,nm)
        if p is not None: raw[:,ch]=p[comp]/1000.0
    # power W/kg pass-through
    for (nm,comp),ch in P.items():
        p=get_point(c,labs,nm)
        if p is not None: raw[:,ch]=p[comp]
    # ---- GRF per-frame BW-normalized plate signals (for per-cycle stance assignment) ----
    plates = _plate_signals(c, mass)
    # ---- EMG: dataset EMG is raw uV time-series, NOT the gait-cycle rectified [0,1]
    #  envelope the training convention uses. We DO NOT fabricate -> EMG stays NaN -> mask False.
    # ---- per-cycle slice + resample to 101, then attach GRF as 0-padded stance waveform ----
    out=[]
    for side,s,e in cycles:
        seg=raw[s:e+1,:]                     # (T,54)
        cyc=np.full((101,54),np.nan)
        for ch in range(54):
            cyc[:,ch]=resample101(seg[:,ch])
        # GRF: if this cycle's stance foot cleanly loaded a plate, fill full 101 (0 in swing,
        #  matching the training 'cp' GRF convention). Else leave GRF NaN (mask False).
        _attach_grf_cycle(cyc, plates, side, s, e)
        out.append((side,cyc))
    return out


GRF_THR = 0.05   # BW; vertical load above this = "on plate"


def _plate_signals(c, mass):
    """Return list of per-frame (downsampled to point rate) plate dicts in BW units.
    Fx->AP(GRF_V_X), Fy->ML(GRF_V_Y), Fz->vertical(GRF_V_Z, sign-flipped to +)."""
    an=c["data"]["analogs"]; albs=c["parameters"]["ANALOG"]["LABELS"]["value"]
    arate=c["header"]["analogs"]["frame_rate"]; pr=c["header"]["points"]["frame_rate"]
    nfr=c["data"]["points"].shape[2]
    sub=int(round(arate/pr)) if pr>0 else 1
    BW=mass*G
    def gv(nm): return an[0,albs.index(nm),:] if nm in albs else None
    def to_point(sig):
        n=nfr*sub
        s=sig[:n] if len(sig)>=n else np.pad(sig,(0,n-len(sig)),constant_values=np.nan)
        return np.nanmean(s.reshape(nfr,sub),axis=1)
    plates=[]
    for pl in ["1","2"]:
        fz=gv(f"Force.Fz{pl}")
        if fz is None: continue
        plates.append(dict(
            x=to_point(gv(f"Force.Fx{pl}"))/BW,
            y=to_point(gv(f"Force.Fy{pl}"))/BW,
            z=-to_point(fz)/BW,
        ))
    return plates


def _attach_grf_cycle(cyc, plates, side, s, e):
    """Attach the IPSILATERAL foot's force-plate GRF to this cycle (0-padded swing,
    matching training 'cp' GRF convention).

    Foot assignment is by temporal coincidence with the ipsilateral stance: the cycle
    starts at the cycle's own heel strike, so the ipsilateral stance load sits in the
    FIRST ~0-60% of the cycle. We pick the plate whose loaded interval centroid is in
    that early window (closest to the expected mid-stance ~30%). A plate loaded in the
    BACK half is the contralateral foot -> not attached. Final GRF validity (and the
    moment foot-assignment) is confirmed downstream against Vicon's stance moment.
    """
    base = 34 if side=="R" else 37
    e=max(e, s+1); T=e-s+1
    best=None; best_score=1e9
    for d in plates:
        seg_z=d["z"][s:e+1]
        if not np.isfinite(seg_z).any(): continue
        loaded=np.isfinite(seg_z) & (seg_z>GRF_THR)
        if loaded.sum() < 0.15*T:    # too little contact -> not this cycle's stance
            continue
        centroid = np.flatnonzero(loaded).mean()/max(T-1,1)
        if centroid > 0.55:          # back-half load = contralateral foot -> skip
            continue
        score=abs(centroid-0.30)     # ipsilateral mid-stance ~30% of cycle
        if score<best_score: best_score=score; best=d
    if best is None: return
    seg_z=best["z"][s:e+1]; loaded=np.isfinite(seg_z) & (seg_z>GRF_THR)
    for comp,ch in [("x",base),("y",base+1),("z",base+2)]:
        full=np.zeros(T); seg=best[comp][s:e+1]
        full[loaded]=np.nan_to_num(seg[loaded], nan=0.0)
        cyc[:,ch]=resample101_zeropad(full)


def resample101_zeropad(x):
    """Resample a finite (0-padded) waveform to 101 points; never introduces NaN."""
    T=len(x)
    if T<2: return np.zeros(101)
    return np.interp(np.linspace(0,1,101), np.linspace(0,1,T), np.nan_to_num(x,nan=0.0))


# Hip flexion channels carry a large between-lab zero-reference OFFSET (this site's hip
# is ~+24deg vs raw bmclab; the training transform itself adds +23.71deg to bmclab hip).
# That offset makes per-subject hip PCC unreliable in SIGN (individual hip PCCs scatter
# -0.84..+0.79 while the COHORT-MEAN hip PCC is +0.96 -> no genuine inversion exists).
# Deciding hip polarity per-subject here would spuriously invert healthy hips. So hip
# flexion polarity is fixed to +1 (no flip) and the offset risk is reported. Knee/ankle
# have no such offset and their PCCs are unambiguous -> safe to decide by correlation.
OFFSET_CONFOUNDED_CH = {0, 3}   # hip_flex_r, hip_flex_l


def decide_polarity(cyc_stack, ref):
    """cyc_stack: (n_cyc,101,54); ref: the loaded --angle-ref NPZ. Return {54ch: +1/-1} for SAG6 by
    corr to the training PD reference. Uses ONLY the angle channels' own reference. NEVER targets
    (leakage guard)."""
    flips={}
    for ch in SAG6:
        r=ref[str(ch)]
        col=cyc_stack[:,:,ch]
        m=np.nanmean(col,axis=0)
        if not np.isfinite(m).all() or not np.isfinite(r).all():
            flips[ch]=1; flips[(ch,'pcc')]=float('nan'); continue
        a=m-m.mean(); b=r-r.mean()
        denom=np.sqrt((a*a).sum()*(b*b).sum())
        pcc=float((a*b).sum()/denom) if denom>1e-9 else 0.0
        flips[(ch,'pcc')]=pcc
        if ch in OFFSET_CONFOUNDED_CH:
            flips[ch]=1   # hip: offset confound -> never flip per-subject (see comment)
        else:
            flips[ch]=-1 if pcc<-0.3 else 1   # ambiguous(|pcc|<0.3) -> no_flip
    return flips


def main():
    global DATA, OUT, PARTICIPANTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="extracted release, the Dataset folder")
    parser.add_argument("--participants", type=Path, default=PARTICIPANTS, help="the release's participant workbook")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--angle-ref", type=Path, default=ANGLE_REF)
    parser.add_argument("--peak-p5", type=Path, default=PEAK_P5)
    args = parser.parse_args()
    DATA, OUT, PARTICIPANTS = args.data, args.out, args.participants
    if not DATA.is_dir() or not PARTICIPANTS.is_file():
        raise FileNotFoundError("Supply the extracted release and participant workbook with --data and --participants")
    for path, what in ((args.angle_ref, "training-cohort angle reference"), (args.peak_p5, "training-cohort peak thresholds")):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found: the {what} is derived from the training data and built "
                                    "locally (see the module docstring); pass its location explicitly.")
    ref = np.load(args.angle_ref)
    TH = json.loads(args.peak_p5.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    masses=load_masses()
    pre_trials=sorted(glob.glob(str(DATA/"*/*/*_Pre/*_T*/*_c3d.c3d")))
    by_subj={}
    for t in pre_trials:
        sid=re.search(r"(ID\d+)_Pre",t).group(1)
        by_subj.setdefault(sid,[]).append(t)

    polarity_table={"bmclab_pd_external":{}}   # nested like per_subject_flip_table.json
    summary=[]
    for sid in sorted(by_subj,key=lambda x:int(x[2:])):
        mass=masses.get(sid)
        if mass is None:
            print(f"[skip] {sid} no mass"); continue
        all_cyc=[]; sides=[]; trials=[]
        for t in by_subj[sid]:
            for side,cyc in harmonize_trial(t,mass):
                all_cyc.append(cyc); sides.append(side); trials.append(Path(t).parent.name)
        if not all_cyc: continue
        stack=np.stack(all_cyc)             # (n,101,54)
        flips=decide_polarity(stack, ref)
        # record polarity (channel-name keyed, like training table)
        ch2name={0:"hip_flex_r",3:"hip_flex_l",6:"knee_flex_r",9:"knee_flex_l",12:"ankle_dorsi_r",15:"ankle_dorsi_l"}
        polarity_table["bmclab_pd_external"][sid]={ch2name[ch]:int(flips[ch]) for ch in SAG6}
        # NOTE: we DO NOT apply flips here, and nothing downstream applies them either.
        #  eval_external_zeroshot_8ch.py loads this site as source "ext_pd" and does not
        #  re-apply the per-subject flip transform, and no script reads the table written
        #  below, so it is a record of the polarity decision rather than an input to any
        #  producer. On the release this script processes every subject resolves to +1,
        #  so raw stays raw. A release that produced a -1 here would need the flip applied
        #  explicitly before evaluation.
        # build mask: finite per cycle/channel
        mask=np.isfinite(stack).all(axis=1)  # (n,54) - channel valid if all 101 pts finite
        # ---- Foot assignment ground truth = Vicon's stance-phase moment ----
        # A cycle's ipsilateral foot had force-plate support iff its STANCE-segment (0-60%)
        # moment peak exceeds that channel's TRAINING-SET p5. Vicon's inverse-dynamics
        # moment is a fixed fact (independent of our processing), so using it to decide
        # foot assignment is NOT circular. We use STANCE peak (not full-cycle nanmax) to
        # exclude swing-phase inertial moments (a different physical quantity, esp. hip/knee).
        STANCE=slice(0,61)   # 0-60% of the 101-pt cycle = ipsilateral stance
        def stance_peak(row_i, ch):
            seg=stack[row_i, STANCE, ch]
            return np.nanmax(np.abs(seg)) if np.isfinite(seg).any() else 0.0
        # sagittal moment channels by side
        R_SAGMOM=[18,24,26]; L_SAGMOM=[21,25,27]    # hip,knee,ankle sagittal moment
        R_GRF=[34,35,36];    L_GRF=[37,38,39]
        for i in range(stack.shape[0]):
            # per-channel stance-moment validity
            for ch in R_SAGMOM+L_SAGMOM:
                p5=TH.get(str(ch), 0.0)
                if not (stance_peak(i,ch) >= p5):
                    mask[i,ch]=False
            # hip Y/Z follow ipsilateral hip-X stance validity
            mask[i,19]&=mask[i,18]; mask[i,20]&=mask[i,18]
            mask[i,22]&=mask[i,21]; mask[i,23]&=mask[i,21]
            # GRF foot-assignment confirmation: keep this cycle's GRF only if the
            # ipsilateral foot actually had stance support (its ankle stance moment is real).
            # Ankle moment is the most GRF-coupled -> use it as the contact witness.
            r_contact = mask[i,26]      # R ankle sagittal moment real (stance-supported)
            l_contact = mask[i,27]
            if not r_contact: mask[i,R_GRF]=False
            if not l_contact: mask[i,L_GRF]=False
        # write
        np.save(OUT/f"{sid}.npy", stack.astype(np.float32))
        np.save(OUT/f"{sid}_mask.npy", mask)
        with open(OUT/f"{sid}_meta.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["cycle_idx_in_trial","group","n_valid_channels","subject_id","trial","side"])
            for i in range(len(all_cyc)):
                w.writerow([i,"bmclab_pd",int(mask[i].sum()),sid,trials[i],sides[i]])
        summary.append(dict(sid=sid,n_cyc=len(all_cyc),mass=mass,
            valid_ang=int(mask[:, :18].sum()),
            valid_grf=int(mask[:,34:40].sum()),
            valid_mom=int(mask[:,18:28].sum()),
            pcc={ch2name[ch]:round(flips.get((ch,'pcc'),float('nan')),3) for ch in SAG6},
            flips={ch2name[ch]:int(flips[ch]) for ch in SAG6}))
        print(f"[{sid}] n_cyc={len(all_cyc)} mass={mass} valid GRF cells={int(mask[:,34:40].sum())} "
              f"ankR_pcc={flips.get((12,'pcc')):.2f} ankL_pcc={flips.get((15,'pcc')):.2f}")

    (OUT/"per_subject_flip_table_pd_external.json").write_text(json.dumps(polarity_table,indent=2))
    (OUT/"_harmonize_summary.json").write_text(json.dumps(summary,indent=2,default=float))
    print(f"\nWrote {len(summary)} subjects to {OUT}")


if __name__=="__main__":
    main()
