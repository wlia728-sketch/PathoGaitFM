# Input contract and units

## Angle upload and command-line examples

Input is a real NumPy array `(B, 100, 16)` or `(100, 16)`. `B` is the number of gait cycles; each cycle has 100 equally spaced samples (0–99% of the cycle, with the 100% endpoint removed). Use heel strike to the next heel strike of the same limb, matching the side-specific phase convention of the relevant source harmoniser. Angles are in degrees and already aligned to the model's anatomical sign and zero conventions. The API does not perform inverse kinematics, segmentation, sensor calibration or source-specific polarity correction. A continuous walking recording must be segmented and aligned before upload; simply stretching an arbitrary recording to 100 rows does not establish a valid cycle.

| Input position | Model channel | Quantity |
|---|---|---|
| 0–2 | 0–2 | Right hip X/Y/Z |
| 3–5 | 3–5 | Left hip X/Y/Z |
| 6, 7 | 6, 7 | Right/left knee sagittal angle |
| 8, 9 | 8, 9 | Right/left ankle sagittal angle |
| 10–12 | 34–36 | Right pelvis X/Y/Z |
| 13–15 | 37–39 | Left pelvis X/Y/Z |

The numeric mapping is authoritative: `model/pathogait/data/channels.py` maps the original 54 channels to 40. X/Y/Z labels refer to the harmonised model convention, not arbitrary camera coordinates. The R/L pelvis entries are model storage slots aligned with each limb's cycle; they are not two anatomical pelvises. Use the appropriate source harmoniser before this API; do not apply its sign corrections twice. In particular, OpenSim coordinate names alone do not establish the required sign or neutral-pose convention. The source transforms in `model/pathogait/data/transform.py` and `transform_per_subject.py` describe the offsets and sign corrections used for the reported data.

An optional boolean mask has shape `(B, 16)` or `(16,)`: **True means observed**. Alternatively, set a whole missing angle channel to NaN across its 100 samples. Partially missing time samples, infinity, non-numeric arrays, invalid shapes and cycles with no observed kinematics are rejected. Do not fill unavailable channels with zero: zero is a measured angle unless its mask is false. Kinetic target channels remain hidden from inference; their computational validity mask is distinct from the observed-input mask.

The seven configurations use all available angles; omit pelvis, hip, knee or ankle angles; retain hip + knee; or retain hip only. Exact masks are shared by the API and demo in `scripts/lib/input_configurations.py`. Each configuration applies to the channels actually supplied and must leave at least one observed channel. Missing inputs use the checkpoint's learned token, as in the paper evaluation.

Prediction output order is right hip, left hip, right knee, left knee, right ankle, left ankle moment, right vGRF, left vGRF. Model indices are `[10, 13, 16, 17, 18, 19, 22, 25]`. Units are six **Nm/kg** followed by two **BW** (multiply BW by 100 for %BW). To obtain N or Nm, use the subject's mass and gravity as appropriate; the API does not invent anthropometry.

## Normalisation

For non-EMG channels, `z = (physical - mean) / (3 * std)`, clipped to `[-1.5, 1.5]`. The inverse is `physical = z * (3 * std) + mean`. Clipped input values cannot be recovered by the inverse. The API uses the statistics selected by `eval_common.ZSTATS_PATH`, verifies their relationship to the checkpoint, and records their SHA-256.

Reported checkpoints use `data/zstats/global_zstats.json`. `global_zstats_stage1_only.json` is intended for a new experiment trained consistently in that space. The legacy file is not described as Stage-1-training-only. Switching constants while keeping the old checkpoint changes the model's meaning and is rejected by the loader.

## Conditions and inference records

The cohort must be supplied explicitly: `cp`, `normal` (TD), `vdk_stroke` or `bmclab_pd`. Source and severity default to their trained null embeddings (12 and 6, respectively). The model has independent condition dropout during training, so these indices are supported model inputs. They are not inferred disease-severity labels. To reproduce a paper evaluation, provide that evaluation's recorded source and severity IDs; the generic unknown-severity upload setting does not reproduce a cohort's reported benchmark by itself.

`predict_kinetics.py` requires `--input` and `--cohort`. It saves `full_kinetics` `(B,100,8)` and the full model-space array `full_model_z` `(B,40,100)` in `predictions.npz`. `partial_input.py` saves these arrays for all seven configurations, using paired random seeds. `run.json` records the checkpoint/statistics hashes, units, condition IDs, masks, seed values and sampling settings.

Sampling retains the paper procedure: 50 DDIM steps, CFG 1.0, the cellwise trimmed mean of six saved x0 estimates and the final state, followed by the mean over three seeds. The first model call starts from Gaussian noise in the joint 40-channel sample. After each update, observed angles are re-noised to the next level and reinserted; absent inputs receive the mask token. Measured kinetics are never an input. The result is an aggregate across saved states and seeds, not only the final denoising step.

Parkinson joint moments were not evaluated in this study. The API returns the same eight target slots for every cohort, but this does not establish PD moment accuracy. Accuracy checks need separately supplied reference kinetics, a held-out evaluation checkpoint and the appropriate scoring window. The ALLDATA checkpoint is for external inputs; internal participants used to train it are not independent test examples.

## Full experiment arrays

Preprocessing scripts write `(n_cycles,101,54)` arrays and `(n_cycles,54)` validity masks, plus source metadata. Those arrays pass through the full source-specific transform (including units, polarity and normalisation); they are not interchangeable with the already-harmonised 16-angle input above. See `data/README.md` and the dataset-specific README for mass tables and source conventions.

RMSE in Nm/kg and GRF RMSE in BW/%BW must remain separate. Range-normalised RMSE is dimensionless, but its aggregation still depends on which channels, limbs, cycles, participants and cohorts are weighted. A small numerical RMSE after a unit conversion is not evidence of better predictions.

The internal error reference is harmonised and clipped by the model transform. Converting its units
back does not undo clipping.
