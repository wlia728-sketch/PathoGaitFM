# Data

No participant recordings, processed arrays, splits, labels or per-participant tables are distributed
with the code. This directory holds the normalisation constants the model needs, the harmonisation code
for each public dataset and the preparation scripts. Everything else is built locally from your own
downloads, under the terms of each release.

## Sources

### Pretraining (Stage 1)

| Source | Link |
|--------|------|
| AddBiomechanics Dataset (Werling et al., PLOS ONE 18:e0295152, 2023; arXiv:2406.18537), seven constituent studies | https://addbiomechanics.org/download_data.html |
| — Camargo et al., J Biomech 119:110320, 2021 | https://doi.org/10.1016/j.jbiomech.2021.110320 |
| — Carter et al., University of Bath Research Data Archive, 2023 | https://doi.org/10.15125/BATH-01111 |
| — Moore et al., PeerJ 3:e918, 2015 | https://doi.org/10.7717/peerj.918 |
| — Tan et al., J Biomech 139:111145, 2022 | https://doi.org/10.1016/j.jbiomech.2022.111145 |
| — Tan et al., IEEE Trans Ind Inform 19:1445-1455, 2023 | https://doi.org/10.1109/TII.2022.3189869 |
| — van der Zee et al., Sci Data 9:704, 2022 | https://doi.org/10.1038/s41597-022-01817-1 |
| — Wang et al., Wearable Technologies 4:e11, 2023 | https://doi.org/10.1017/wtc.2023.7 |
| Van Criekinge et al., Sci Data 10:852, 2023, able-bodied adults | https://doi.org/10.1038/s41597-023-02767-y |

### Fine-tuning cohorts (Stage 2)

| Cohort | Source | Link |
|--------|--------|------|
| Post-stroke | Van Criekinge et al., Sci Data 10:852, 2023 | https://doi.org/10.1038/s41597-023-02767-y |
| Parkinson's disease | BMClab walking dataset (Shida et al., Front Neurosci 17:992585, 2023) | https://doi.org/10.6084/m9.figshare.14896881 |
| Cerebral palsy and typically developing children | Yueyang Hospital of Integrated Traditional Chinese and Western Medicine, Shanghai University of Traditional Chinese Medicine | Restricted by the ethics approval of the secondary analysis; not distributed. |

### External validation and paired-modality datasets

| Folder under `data/external/` | Source | Link |
|------|--------|------|
| `leuven_cp` | Leuven cerebral palsy, SimTK `cp-child-gait` (Meyns et al., Front Hum Neurosci 11:96, 2017) | https://simtk.org/projects/cp-child-gait |
| `gillette_crouch` | Gillette crouch gait, SimTK `crouchgait` | https://simtk.org/projects/crouchgait |
| `fukuchi_healthy` | Fukuchi WBDS healthy adults (Fukuchi et al., PeerJ 6:e4640, 2018) | https://doi.org/10.7717/peerj.4640 |
| `pd_nordic` | PD-Nordic, Parkinson's disease (Sci Data 12:1937, 2025) | https://doi.org/10.6084/m9.figshare.29371769 |
| `uhlrich_opencap` | Uhlrich, the OpenCap LabValidation dataset (Uhlrich et al., PLoS Comput Biol 19:e1011462, 2023), marker and video kinematics, released inverse-dynamics moments | https://doi.org/10.1371/journal.pcbi.1011462 |
| `grouvel_imu` | Grouvel wearable inertial dataset (Grouvel et al., Sci Data 10:180, 2023), marker and IMU kinematics | https://doi.org/10.1038/s41597-023-02077-3 |

Each folder carries its own README with that dataset's terms, the harmonisation steps and the
evaluation script that reads the result.

## Expected local layout

```
data/
  zstats/global_zstats.json               normalisation constants of the released checkpoints (shipped)
  zstats/global_zstats_stage1_only.json   constants fitted on the Stage-1 training split, for new training runs (shipped)
  splits/                                 v4_stage1_split.json, v4_stage2_split_grffix.json, v4_stage2_split_alltrain.json
  metadata/                               per_subject_severity.json, per_subject_flip_table.json, per_subject_flip_table_addbio.json,
                                          bmclab_metadata.json, vdk_demographics.json, addbio_meta_summary.json
  pretraining/addbio/                     the pretraining corpus, <study>_<subject>_split<k>.npy with <stem>_mask.npy
  cohorts_raw/<cohort>/                   the four fine-tuning cohorts in raw units, <stem>.npy with <stem>_mask.npy
  cohorts_grffix/<cohort>/                the same after prepare/grf_fix_step1_build.py; what the evaluators read
  external/<dataset>/raw/, processed/     each external release and its harmonised arrays
```

Every array is `(n_cycles, 101, 54)` float32 in physical units with a `(n_cycles, 54)` boolean validity
mask; the channel table is `model/pathogait/data/channels.py` and the upload contract is
[data format](../docs/data_format.md). The split and table formats are given in
[training](../docs/training.md). The scripts read these locations by default; `--raw-root` and the
environment variables `PATHOGAIT_SPLIT_JSON`, `PATHOGAIT_SEVERITY_JSON`, `PATHOGAIT_FLIP_TABLE`,
`PATHOGAIT_FLIP_TABLE_ADDBIO`, `PATHOGAIT_COHORT_RAW_DIR` and `PATHOGAIT_METADATA_DIR` point them elsewhere.

## Preparing the data

An external dataset takes three steps: download the release, run that folder's harmonisation script,
then run the evaluation script named in its README. The harmonisation writes `processed/<subject>.npy`,
the validity mask, `meta.csv` and a `_harmonize_summary.json` with the body masses read from the release.
The Fukuchi and PD-Nordic harmonisers also need reference values derived from the training cohorts
(`fukuchi_healthy/build_train_healthy_ref.py`; the PD harmoniser's docstring describes its two files);
these are built locally and passed on the command line.

The four fine-tuning cohorts are cut from each release by the per-cohort extraction the Methods describe.
The Parkinson pipeline is `prepare/bmclab/`; `prepare/grf_fix_step1_build.py` then writes
`cohorts_grffix/` from `cohorts_raw/`; `prepare/audit_all_cycle_counts.py` counts the cycles per source.

| File | Schema |
|------|--------|
| `metadata/bmclab_metadata.json` | `{subjects: {"<ID>": {mass_kg: float, height_cm: float}}}` from the release's `PDGinfo.xlsx` |
| `metadata/vdk_demographics.json` | `{vdk_healthy: [...], vdk_stroke: [{sub_idx: int, weight_kg, height_cm, age, male}]}` |
| `metadata/addbio_meta_summary.json` | `{"<Study>": [{subject_id: str, mass_kg: float, height_m: float, n_cycles: int}]}` |
| `external/leuven_cp/masses.json` | `{"<ID>": {mass_kg: float, height_cm: float}}` from the SimTK participant table |

## Normalisation constants

`zstats/global_zstats.json` is the space the released checkpoints were trained in and is required
for inference with them. A new training experiment fits `zstats/global_zstats_stage1_only.json` on
its Stage-1 training split (`prepare/fit_stage1_zstats.py`) and passes that one file unchanged to
Stage 1, every Stage-2 fold, the all-data model and evaluation
(`PATHOGAIT_ZSTATS=data/zstats/global_zstats_stage1_only.json`). Checkpoints record the SHA-256 of the
file they were trained with, and resume or warm-start fails if the spaces differ.

## Model channels and demo inputs

The trained model keeps its 40-channel joint representation. The estimation input regime masks the
sEMG and non-sagittal hip-moment channels with the learned token; this is part of the checkpoint
format. The upload interface accepts 16 joint-angle slots in harmonised degrees, with unavailable
channels omitted or marked missing, and never measured kinetics. Uploads stay in memory in the local
demo server.
