# External site 4, Nordic-walking Parkinson disease

**Identity.** Public Nordic-walking Parkinson dataset. Scientific Data 2025;12:1937.
figshare doi:10.6084/m9.figshare.29371769, Zenodo doi:10.5281/zenodo.15131169. CC BY 4.0.

**What this code does.** `harmonize_pd.py` writes the 54-channel arrays from the pre-intervention baseline
trials only. Vicon moments are converted from N.mm/kg to Nm/kg. Channel polarity is decided by correlation
against the training Parkinson cohort-mean angle waveform (`--angle-ref`), never against a target channel,
and missing channels stay missing rather than being zero-filled. `--peak-p5` gives the training-set stance-peak
thresholds for moment validity. Both reference files are derived from the training data and are built locally;
the script docstring gives their format.

**Severity.** `processed/ext_pd_severity.json` carries a single default stratum for the whole site.
Per-subject Hoehn and Yahr stage is not published with the release and is not inferred here.

**Producer.** `scripts/eval/eval_external_zeroshot_8ch.py`. Reference eligibility is implemented
in `harmonize_pd.py`; the evaluator uses the resulting validity masks.

## Rebuilding this site

1. Download the release from the link above, under that release's own terms.
2. Run the harmonisation script in this directory. It writes the per-subject `(n_cycles, 101, 54)` raw-unit
   arrays, the channel-validity masks and a meta table, under `processed/`.
3. Run the evaluation script named above; it writes its JSON result under `outputs/evaluation/`.
