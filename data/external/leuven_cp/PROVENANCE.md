# External cerebral-palsy site 1 (Leuven): provenance and terms of use

## Dataset identity
**"Gait data children with spastic CP and typically developing children"**
SimTK project `cp-child-gait` — https://simtk.org/projects/cp-child-gait

3D gait of ambulant children, collected at the Clinical Motion Analysis Laboratory
(CMAL), University Hospital Leuven / UZ Pellenberg (KU Leuven), Belgium.

| Group | n | code | age (yr) | GMFCS |
|---|---|---|---|---|
| Hemiplegic spastic CP | 5 | HeCP1–5 | 9.0 ± 2.3 | I–II |
| Diplegic spastic CP | 4 | DiCP1–4 | 10.5 ± 1.7 | I–II |
| Typically developing | 5 | TD1–5 | 8.4 ± 1.5 | — |

**Modalities** (verified in the c3d): total-body Vicon Plug-in-Gait markerset @100 Hz;
two AMTI force plates (analog @1500 Hz, FX/FY/FZ + MX/MY/MZ ×2 plates); 8-channel
bilateral surface EMG (RF, Vastus, Biceps Fem, Med Hamstring, Tib Ant, Gastroc, Soleus,
Glut). Note vGRF polarity: FZ is **negative** in the c3d (downward-positive convention)
→ use `vGRF = -FZ` to align with the project's upward-positive vGRF.

## Source publication (cite this)
Meyns P, Molenaers G, Duysens J, Jonkers I. *The Differential Effect of Arm Movements
during Gait on the Forward Acceleration of the Centre of Mass in Children with Cerebral
Palsy and Typically Developing Children.* **Front Hum Neurosci. 2017;11:96.**
doi:10.3389/fnhum.2017.00096 — its Data-Availability statement releases this exact data
on SimTK "for other researchers to evaluate and use for future research."
(If EMG channels are used, also cite Van Gestel et al., Res Dev Disabil. 2012;33:916–923,
which documents the sEMG protocol for this cohort.)

## License (SimTK "Custom Use Agreement") — usage verdict
- **Publishing derived/aggregate results (model outputs, summary metrics): ALLOWED** with
  citation — the source paper explicitly invites reanalysis.
- **Redistributing the raw c3d: not allowed by default.** Readers obtain it from the SimTK
  project page.
- The full Custom Use Agreement is shown at download time behind a login and was accepted to
  obtain the data. Anyone reproducing this site accepts it in the same way at their own download.
- These are the de-identified clinical gait records of 14 minors, so results are reported at
  aggregate level.

**Manuscript scope of use (decided): VERTICAL-GRF-ONLY zero-shot reconstruction.**
The measured force plates give ground-truth vGRF; OpenSim inverse kinematics gives the
joint-angle model inputs. Joint moments are NOT claimed for this external set (would be
self-consistency vs the same IK/ID pipeline — the inverse-dynamics circularity trap).

## What this directory is, and what it is not
A download of this project gives 22 gait and 13 static c3d files and a participant table carrying
age, mass and GMFCS level. The harmonisation code in this directory is what turns it into the
evaluated arrays.

One nearby dataset is easy to confuse with this one and is not used in this work. The ETH / AnyBody
set of the same clinical population releases measured-versus-predicted ground-reaction-force and
moment tables only, with no markers and no kinematics, so it cannot serve as input to a model that
takes kinematics.
