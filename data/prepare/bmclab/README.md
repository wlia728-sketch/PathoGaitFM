# Parkinson cohort (BMClab) pipeline

Rebuilds `data/cohorts_raw/bmclab_pd/` from the BMClab walking dataset (Shida et al. 2023, figshare
doi:10.6084/m9.figshare.14896881).
Place the release's `C3Dfiles.zip` and `PDGinfo.xlsx` under `raw/`; intermediate results are written under
`work/`. Needs `opensim` 4.5, `nimblephysics` (Rajagopal2015 model), `ezc3d`, `pandas`, `pyarrow`, `xlrd`.

Pelvis channels contain OpenSim tilt, list and rotation in degrees, in that order in both
48–50 and 51–53. Step 6 centres each angle over its gait cycle. A channel is valid only when
all its samples are finite. The accompanying metadata identifies this representation as
`opensim_angles_deg`. Training and evaluation must use the same representation as the data
and checkpoint; the released checkpoints use `legacy_translations_m` and cannot be used
to reproduce the reported results with these angular BMClab arrays.

| Step | Script | Writes |
|---|---|---|
| 3.1 marker rename, c3d to TRC | `s31_batch_c3d_to_trc.py` (`s31_c3d_to_trc.py`, `marker_map.py`) | `work/trc/` |
| 3.2 scaling with neutral lower-body overrides | `s32_batch_scale.py` (`s32_scale_setup_builder.py`, `s32_make_ik_model.py`) | `work/scaled_models/` |
| 3.3 inverse kinematics, Visual3D cross-check | `s33_batch_ik.py` (`s33_ik_setup_builder.py`), `s33_v3d_cross_check.py` | `work/ik/` |
| 3.4 force plates to external loads | `s34_batch_grf.py` (`s34_grf_transform.py`, `s34_compute_cop.py`) | `work/grf/` |
| 3.5 inverse dynamics, polarity check | `s35_batch_id.py`, `s35_polarity_check.py` | `work/id/`, `work/grf/_grf_mask.json` |
| 4 gait cycles, 101 points | `s4_batch_cycle.py` (`s4_cycle_extraction.py`) | `work/cycles/` |
| 5 54-channel arrays | `s5_batch_write_cohort.py` (`s5_channel_mapping.py`, `s5_joint_power.py`) | `data/cohorts_raw/bmclab_pd/` |
| 6 harmonisation in place | `s6_grf_axis_permutation.py`, `s6_pelvis_demean.py`, `s6_subtalar_mask.py`; checks `s6_polarity_vs_cohorts.py` | the same arrays |
| 7 cohort manifest | `s7_cohort_manifest.py` | `data/cohorts_raw/cohort_manifest.csv` |

Body mass and height come from the release's `PDGinfo.xlsx` (never shipped here). The scripts were moved
into this layout from the working tree and compiled, but the pipeline has not been re-run in this layout.
