# External site 2, Gillette crouch gait

**Identity.** Crouch-gait dataset from Gillette Children's, USA. SimTK project `crouchgait`,
https://simtk.org/projects/crouchgait.

**Terms.** SimTK Custom Use Agreement, as for the Leuven set. Paediatric clinical data; report at aggregate
level, do not re-host the release.

**What this code does.** `extract_raw.py` unpacks the release, `crouch_harmonize_to_54ch.py` writes the
54-channel arrays. The release provides OpenSim gait2392 inverse kinematics and measured force-plate ground
reaction force, so this site's kinematics are not re-solved here. The gait2392 knee convention is
sign-flipped to the training convention; hip and ankle pass through. Vertical GRF is the scored target and
moments are left missing, since this site carries no independent moment ground truth.

**Producer.** `scripts/eval/eval_crouch_jackknife.py`.

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Run the harmonisation script in this directory. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.
