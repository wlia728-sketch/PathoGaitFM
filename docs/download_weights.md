# Download model weights

The [v2026.09.27 release](https://github.com/wlia728-sketch/PathoGaitFM/releases/tag/v2026.09.27) holds the twelve checkpoints as assets named `<run>.pt`. The code's MIT licence does not cover the weights: they are released under CC BY 4.0 (attribution by citing the manuscript; research software, not a medical device). The same terms are stated in `LICENSE` and in the release notes.

For inference on new inputs and for the local demo, download **v4_stage2_final8ch_ALLDATA.pt** only. Rename it to `final.pt` and place it in the existing `checkpoints/v4_stage2_final8ch_ALLDATA/` directory. Other checkpoints follow the same rule: `checkpoints/<run>/final.pt`. Keep the included `args.json` files.

With the GitHub CLI installed and authenticated:

```sh
gh release download v2026.09.27 --repo wlia728-sketch/PathoGaitFM --pattern v4_stage2_final8ch_ALLDATA.pt --output checkpoints/v4_stage2_final8ch_ALLDATA/final.pt
```

Verify the size and SHA-256 before use:

```sh
python scripts/verify_weights.py --run v4_stage2_final8ch_ALLDATA
python run_demo.py --checkpoint checkpoints/v4_stage2_final8ch_ALLDATA/final.pt
```

The verifier must print `PASS`. If it reports `MISSING` or `FAIL`, correct the path or download again. `python scripts/verify_weights.py` checks all twelve; `--root /path/to/checkpoints` checks another directory. The [manifest](../checkpoints/CHECKPOINTS_MANIFEST.md) explains each checkpoint's role and lists its hash.

The released weights require the included `data/zstats/global_zstats.json`; do not substitute `global_zstats_stage1_only.json`, which is for new training experiments.
