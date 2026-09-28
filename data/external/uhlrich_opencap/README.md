# Input-sensor site 1, OpenCap markerless video

**Identity.** Public OpenCap LabValidation dataset, healthy adults.

**Cite.** Uhlrich SD, Falisse A, Kidzinski L, et al. OpenCap: human movement dynamics from smartphone videos.
PLoS Comput Biol 2023;19:e1011462.

**What this code does.** `opencap_harmonize.py` writes the 54-channel arrays for two kinematic modalities
that share one force ground truth, laboratory motion capture and two-camera video, so the two can be scored
against the same target. The release solves inverse kinematics with the LaiArnoldModified2017 model, whose
knee convention needs no sign flip, and is already in the OpenSim ground frame. `extract_moments_id_101pt.py`
and `build_moment_eval_54ch.py` prepare the sagittal-moment evaluation.

**Producers.** `scripts/eval/eval_opencap_zeroshot.py marker|video`, `scripts/eval/eval_opencap_moments.py`.

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Run the harmonisation script in this directory. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.
