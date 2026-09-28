# Input-sensor site 2, Grouvel wearable inertial

**Identity.** Public marker, inertial-sensor, insole and force-plate dataset.

**Cite.** Grouvel G, et al. A dataset of overground walking full-body kinematics and kinetics from marker,
IMU, insole and force-plate recordings. Sci Data 2023;10:180. doi:10.1038/s41597-023-02077-3

**What this code does.** `imu_harmonize.py` writes two branches that share one force ground truth and one set
of force-plate cycle events, so the marker and the inertial input are scored on identical cycles. The marker
branch takes markers and force from the c3d through `marker_ik.py`; the inertial branch takes accelerometer
and gyroscope from the paired synchronisation CSV through `imu_ahrs.py` and `imu_opensense.py`, using only
those two signals, so no marker information leaks into it.

**Also needed.** Two OpenSim models. `opensense/Rajagopal_2015.osim` ships with OpenSim at
`Models/RajagopalModel/` and is also on SimTK (project `full_body`); copy it into `opensense/`. The marker
branch additionally needs a subject-scaled model carrying this site's marker names, built with the OpenSim
ScaleTool, since the generic model carries no marker set. Both scripts name the file when it is absent.

**Producer.** `scripts/eval/eval_imu_zeroshot.py marker|imu`.

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Run the harmonisation script in this directory. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.
