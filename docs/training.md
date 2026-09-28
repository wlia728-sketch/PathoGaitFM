# Training

Run from the repository root after building the arrays and tables described below. These are the commands that produced the released checkpoints, with their normalisation constants (`data/zstats/global_zstats.json`). The released checkpoints are the ones reported in the manuscript. Training uses `cudnn.benchmark`, mixed precision and seed 42, so a new run of the same commands gives a statistically equivalent model, not a bit-identical one. For a new training experiment, fit `data/zstats/global_zstats_stage1_only.json` on your Stage-1 training split (`data/prepare/fit_stage1_zstats.py`) and use that one file in both stages and at evaluation (`PATHOGAIT_ZSTATS`).

Stage 1, pretraining on healthy gait:

```
python model/pathogait/training/train_pretrain.py \
  --split data/splits/v4_stage1_split.json --zstats data/zstats/global_zstats.json \
  --use-per-subject-flip \
  --per-subject-flip-paper data/metadata/per_subject_flip_table.json \
  --per-subject-flip-addbio data/metadata/per_subject_flip_table_addbio.json \
  --epochs 25 --lr 1e-4 --batch-size 64 --epoch-pseudo-length 300000 --warmup-steps 5000 \
  --output checkpoints/v4_stage1_unified_polarity_fixed --device cuda --amp
```

Stage 2, fine-tuning, one fold (`cv5_0` to `cv5_4`; the all-data model uses `v4_stage2_split_alltrain.json`):

```
PATHOGAIT_DYNCYCLE_POOL=1 PATHOGAIT_COHORT_RAW_DIR=data/cohorts_grffix \
PATHOGAIT_INPUT_DROP_CH=26,27,28,29,30,31,32,33,11,12,14,15 \
PATHOGAIT_LOSS_TARGET_CH=10,13,16,17,18,19,22,25 \
python scripts/train/train_finetune.py \
  --resume checkpoints/v4_stage1_unified_polarity_fixed/final.pt \
  --zstats data/zstats/global_zstats.json --split data/splits/v4_stage2_split_grffix.json \
  --severity-label data/metadata/per_subject_severity.json \
  --use-per-subject-flip \
  --per-subject-flip-paper data/metadata/per_subject_flip_table.json \
  --per-subject-flip-addbio data/metadata/per_subject_flip_table_addbio.json \
  --input-mask-token --input-mask-token-scope all_invalid --mask-bmclab-moments \
  --fold cv5_0 --epochs 200 --lr 1e-5 --constant-lr --warmup-steps 1000 --batch-size 64 \
  --ckpt-epochs-list 50,100,150,175,199 \
  --output checkpoints/v4_stage2_expA_dynpool_cv5_0 --device cuda --amp
```

The no-pretraining comparison (`v4_stage2_final8ch_scratch_cv5_*`) runs the Stage-2 command without `--resume` and with `--no-pretrain`. Each checkpoint's `args.json` records its settings. The Stage-2 input-drop, loss-target, cycle-pool and data-root environment variables are part of the configuration and are not captured in the historical `args.json` files; set them explicitly. The input-drop setting removes sEMG and non-sagittal hip moments; the loss-target setting selects the six sagittal moments and two vertical GRF channels. Stage-1 resumes only from an epoch checkpoint (`ckpt_epoch<N>.pt`, which stores the optimizer and scheduler state); `final.pt` holds weights only and is the file Stage 2 warm-starts from.

## Files built locally

None of the following is distributed; each is built from your own copy of the source data (`data/README.md`). The evaluation scripts read them from the paths below or from the environment variables `PATHOGAIT_SPLIT_JSON`, `PATHOGAIT_SEVERITY_JSON`, `PATHOGAIT_FLIP_TABLE` and `PATHOGAIT_FLIP_TABLE_ADDBIO`.

`PATHOGAIT_EXCLUDE` may list recording stems, comma-separated, that every evaluation script skips. It is empty by default: the split used for the manuscript already omits the two in-house recordings that failed quality control, so the 22 evaluated children with cerebral palsy are exactly the split entries.


`data/splits/v4_stage1_split.json`, file-level partition of the pretraining corpus:

```json
{"train_count": 442, "val_count": 57, "test_count": 54,
 "splits": {"train": [["addbio_Camargo2021", "Camargo2021_<subject>_split0"], ["vdk_healthy", "vdk_healthy_<index>"]],
            "val": [["<source>", "<stem>"]], "test": [["<source>", "<stem>"]]}}
```

Each entry is `[source, stem]`, where `source` is one of the twelve source names in `scripts/lib/eval_common.py` (`SOURCE_TO_ID`) and `stem` is the array file name without `.npy`.

`data/splits/v4_stage2_split_grffix.json` (five folds) and `v4_stage2_split_alltrain.json` (one fold with a small validation set), recording-level partitions of the four development cohorts:

```json
{"cv5": [{"fold": 0, "train": [["cp", "<stem>"], ["vdk_stroke", "<stem>"]], "val": [["normal", "<stem>"], ["bmclab_pd", "<stem>"]]}]}
```

Parkinson recordings are `<subject>_on` and `<subject>_off`; both states of one subject are kept in the same fold.

`data/metadata/per_subject_severity.json`, the conditioning labels:

```json
{"metadata": {"scheme": "A_per_cohort_independent", "severity_id_space": [0, 1, 2, 3, 4, 5]},
 "mappings": {"cp": {"<stem>": {"cohort_id": 0, "severity_id": 0, "source_metric": "single_bin"}},
              "normal": {"<stem>": {"cohort_id": 1, "severity_id": 5, "source_metric": "control"}},
              "vdk_stroke": {"<stem>": {"cohort_id": 2, "severity_id": 1, "source_metric": "FAC"}},
              "bmclab_pd": {"<subject>_off": {"cohort_id": 3, "severity_id": 2, "source_metric": "H&Y_OFF", "state": "off"},
                            "<subject>_on": {"cohort_id": 3, "severity_id": 4, "source_metric": "PD_on_state", "state": "on"}}}}
```

`cohort_id` is 0 CP, 1 TD, 2 stroke, 3 PD. Severity bins are 0 to 5: CP one bin (0), TD the control bin (5), stroke by Functional Ambulation Category, PD by Hoehn and Yahr stage in the off state with bin 4 reserved for on-medication recordings.

`data/metadata/per_subject_flip_table.json` (the four development cohorts and the Van Criekinge healthy adults) and `per_subject_flip_table_addbio.json` (the seven AddBiomechanics studies), per-recording sign conventions of the six sagittal angle channels:

```json
{"<cohort or study>": {"<stem>": {"hip_flex_r": 1, "hip_flex_l": 1, "knee_flex_r": 1, "knee_flex_l": -1,
                                  "ankle_dorsi_r": 1, "ankle_dorsi_l": 0}}}
```

`1` keeps the channel, `-1` flips it, `0` marks an ambiguous decision, which the transform leaves unflipped (`ambiguous_policy="no_flip"`).

## Effective settings and data provenance

New Stage-1 and Stage-2 runs record the input-drop, loss-target and cycle-pool settings in `args.json`, and `training_manifest.json` records the SHA-256 of the loaded arrays, masks, split, statistics, conditioning tables and warm-start weights. Nonempty `PATHOGAIT_EMG_DROPOUT_CH` or `PATHOGAIT_AUX_LOSS_CH` settings are rejected because those training variants are outside the manuscript. The 40-channel architecture and the learned missing-input token are part of the released estimator.

Metadata outside the default layout can be given through `PATHOGAIT_METADATA_DIR` (JSON tables), `PATHOGAIT_INHOUSE_WORKBOOK` and `PATHOGAIT_PD_WORKBOOK` before launching Python. Missing body-mass metadata raise errors; no default mass is substituted. A Stage-2 `--dry-run` checks one forward and backward batch without saving weights.
