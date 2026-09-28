# Results reported in the manuscript

Aggregate values as reported in the manuscript text and figures. PCC is the Pearson correlation
between estimated and reference waveforms, averaged per participant and then with equal weight over
participants and cohorts; nRMSE is the range-normalised root-mean-square error in per cent. The
per-participant scores behind these values are produced locally by the scripts named in each block
(`scripts/eval/`), which write JSON under `outputs/evaluation/`.

## Five-fold cross-validation in the four development cohorts

`scripts/eval/eval_kinetics_all_cycles.py` (PCC, nRMSE, RMSE); intervals from
`scripts/eval/pretraining_gain_stats.py` and `scripts/eval/model_comparison_paired.py`.

| Model | PCC (four-cohort mean) | TD | CP | PD | Stroke | nRMSE % (mean) |
|---|---|---|---|---|---|---|
| PathoGaitFM, masked inpainting, pretrained | 0.811 (95% CI 0.787 to 0.834) | 0.892 | 0.801 | 0.834 | 0.717 | 21.9 |
| Same architecture without pretraining | 0.745 | | | | | 24.5 |

nRMSE of PathoGaitFM by cohort: TD 11.4, CP 19.6, PD 22.4, stroke 34.3. Pretraining gain 0.066
(95% CI 0.056 to 0.076); the largest gains were 0.203 for the knee moment after stroke and 0.147 for
the knee moment in CP, against 0.015 to 0.045 for vGRF. Joint-moment RMSE 0.10 to 0.12 N·m/kg in TD,
0.14 to 0.18 in CP and 0.22 to 0.27 after stroke.

## Published comparators retrained under the same protocol

`model_baseline/published_comparators.py` (PCC) and `scripts/eval/nrmse_rerun/rerun_comparators_nrmse.py` (nRMSE).

| Model | PCC | nRMSE % |
|---|---|---|
| Ozates CNN | 0.746 | 23.6 |
| GroundLinkNet | 0.796 | 22.8 |
| Sugai LSTM | 0.799 | 23.5 |

Paired difference, masked inpainting minus the best comparator: 0.012 (95% CI −0.001 to 0.026).

## Alternative prediction heads

`scripts/eval/head_vs_headless_ablation.py`.

| Head | PCC | nRMSE % |
|---|---|---|
| Direct regressor from joint angles | 0.753 | 26.8 |
| Masked inpainting (the released estimator) | 0.811 | 21.9 |
| Supervised head on the model's features | 0.821 | 22.7 |

Supervised head minus the best comparator: 0.022 (95% CI 0.009 to 0.035).

## Incomplete joint-angle inputs, same folds, no retraining

`scripts/eval/partial_input_masking.py` (PCC); nRMSE from the same script run through `scripts/eval/nrmse_rerun/`.

| Input configuration | PCC | nRMSE % |
|---|---|---|
| All available angles | 0.811 | 21.9 |
| No pelvis | 0.807 | 21.9 |
| No hip | 0.770 | 23.3 |
| No knee | 0.789 | 22.8 |
| No ankle | 0.808 | 22.3 |
| No ankle or pelvis | 0.802 | 22.3 |
| Hip only | 0.775 | 23.4 |

## External datasets, without retraining

Vertical GRF: `scripts/eval/eval_cp_jackknife.py` (Meyns), `scripts/eval/eval_crouch_jackknife.py`
(Steele), `scripts/eval/eval_external_zeroshot_8ch.py` (Fukuchi, Viglialoro). Participant-level mean ± SD.

| Dataset | n | vGRF PCC | nRMSE % |
|---|---|---|---|
| Meyns 2017, children with CP | 7 | 0.847 ± 0.038 | 23.2 |
| Meyns 2017, TD children | 5 | 0.927 | 19.6 |
| Steele 2010, crouch gait | 10 | 0.906 ± 0.046 | 17.9 |
| Fukuchi 2018, healthy adults | 42 | 0.979 ± 0.007 | 8.3 |
| Viglialoro 2025, Parkinson's disease | 20 | 0.952 ± 0.031 | 16.2 |

Sagittal joint moments: `scripts/eval/eval_external_moments.py` (Fukuchi, full gait cycle) and
`scripts/eval/eval_opencap_moments.py` (Uhlrich, left stance).

| Dataset | n | Joint | PCC | nRMSE % | RMSE N·m/kg |
|---|---|---|---|---|---|
| Fukuchi 2018 | 42 | hip | 0.908 | 15.6 | 0.229 |
| | | knee | 0.891 | 15.3 | 0.128 |
| | | ankle | 0.978 | 14.9 | 0.233 |
| Uhlrich 2023 | 10 | hip | 0.886 | 20.7 | 0.286 |
| | | knee | 0.688 | 25.4 | 0.217 |
| | | ankle | 0.947 | 19.4 | 0.285 |

Mean PCC over the ten dataset × target combinations of the five external datasets: 0.898.

## Paired sensing modalities, same recordings

`scripts/eval/eval_opencap_zeroshot.py marker|video` and `scripts/eval/eval_imu_zeroshot.py marker|imu`.

| Dataset | n | Modality | vGRF PCC | nRMSE % |
|---|---|---|---|---|
| Uhlrich 2023 | 10 | marker-based | 0.901 | 16.8 |
| | | video-based | 0.901 | 16.9 |
| Grouvel 2023 | 10 | marker-based | 0.862 | 21.1 |
| | | IMU-based | 0.889 | 19.1 |
