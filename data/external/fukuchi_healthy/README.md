# External site 3, Fukuchi WBDS healthy adults

**Identity.** Public WBDS dataset of overground and treadmill walking in healthy adults, 42 subjects.

**Cite.** Fukuchi CA, Fukuchi RK, Duarte M. A public dataset of overground and treadmill walking kinematics
and kinetics in healthy individuals. PeerJ 2018;6:e4640.

**What this code does.** `extract_ascii_processed.py` reads the release's own Visual3D-processed ASCII,
`harmonize_fukuchi_ascii.py` writes the 54-channel arrays. This site is scored on the dataset's own
processing pipeline rather than ours, which is what makes it a cross-pipeline test. The axis, unit and sign
mappings were resolved against the training healthy reference and are recorded in the script header. The
between-laboratory moment-amplitude difference is left uncorrected, being a transfer result rather than a
unit error. `build_train_healthy_ref.py` writes `_train_normal_peak_p5.json`, the fifth percentile of the
per-cycle stance peak of each sagittal moment channel in the training healthy cohort; it is derived from the
restricted in-house recordings, so it is not distributed and must be built locally (or passed with `--peak-p5`).
`harmonize_fukuchi_ascii.py` applies it as a validity threshold against this site's own measured moments. A
sagittal moment channel whose stance peak falls below the threshold is masked out for that cycle, and
because the ipsilateral ankle moment is the contact witness for the vertical GRF, a masked ankle moment
also invalidates that side's ground-reaction channels. Polarity is set by the axis and sign maps in the
harmoniser header, not by this file.

**Producers.** `scripts/eval/eval_external_zeroshot_8ch.py` (vertical GRF),
`scripts/eval/eval_external_moments.py` (sagittal moments).

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Place `WBDSascii.zip` and `WBDSinfo.xlsx` under `raw/`, then run `extract_ascii_processed.py` followed by
   `harmonize_fukuchi_ascii.py`. The harmoniser also accepts `--ascii-dir`, `--info` and `--out` for files
   stored elsewhere. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.

From the repository root, the selected-site CLI is:

```sh
python scripts/eval/eval_external_zeroshot_8ch.py --site Fukuchi_healthy --out outputs/evaluation/fukuchi_zeroshot.json
```

Use `--processed` and `--checkpoint` to reference data and weights elsewhere. Missing inputs exit nonzero.
This is a selected-site evaluation; `scripts/eval/eval_external_moments.py` runs the separate full-cycle and
stance sagittal-moment evaluation.
