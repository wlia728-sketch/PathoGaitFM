# External site 1, Leuven cerebral palsy

**Identity.** Gait data of children with spastic cerebral palsy and typically developing children, collected
at the Clinical Motion Analysis Laboratory, University Hospital Leuven / UZ Pellenberg, KU Leuven, Belgium.
SimTK project `cp-child-gait`, https://simtk.org/projects/cp-child-gait.

**Cite.** Meyns P, Molenaers G, Duysens J, Jonkers I. The differential effect of arm movements during gait on
the forward acceleration of the centre of mass in children with cerebral palsy and typically developing
children. Front Hum Neurosci 2017;11:96. doi:10.3389/fnhum.2017.00096

**Terms.** SimTK Custom Use Agreement, shown at download and accepted by the downloader. Derived and
aggregate results may be published with citation; the raw c3d may not be re-hosted. These are the clinical
records of 14 minors. `PROVENANCE.md` in this directory holds the full record, including the modality
inventory and the vertical-GRF polarity convention.

**What this code does.** `cp_c3d_to_trc.py` converts the c3d to TRC, `cp_pig_pipeline.py` runs the OpenSim
inverse kinematics, `cp_harmonize_to_54ch.py` writes the 54-channel arrays. Measured force plates give the
vertical-GRF ground truth. Only vGRF was evaluated at this site.

**Also needed.** `masses.json`, the height and mass of the 14 children, built from the release's own
`participant_information.xlsx` as `{ID: {mass_kg, height_cm}}`. The harmoniser names it when it is absent.

**Producer.** `scripts/eval/eval_cp_jackknife.py`.

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Run the harmonisation script in this directory. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.
