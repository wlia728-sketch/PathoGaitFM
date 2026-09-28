# Trained model weights

No `.pt` weights are bundled with the code. Download them as `<run>.pt` assets from the [GitHub release v2026.09.27](https://github.com/wlia728-sketch/PathoGaitFM/releases/tag/v2026.09.27); see [download instructions](../docs/download_weights.md). Rename each downloaded `<run>.pt` to `final.pt` and place it at `checkpoints/<run>/final.pt`. Inference on new inputs and the local demo need only `v4_stage2_final8ch_ALLDATA`. The five pretrained folds give the cross-validation results, the five scratch folds the no-pretraining comparison, and the Stage-1 checkpoint is the pretrained backbone that Stage 2 warm-starts from.

```sh
python scripts/verify_weights.py --run v4_stage2_final8ch_ALLDATA
python scripts/verify_weights.py
```

`--root /path/to/checkpoints` checks another directory. Missing files and checksum mismatches return a nonzero status. Machine-readable manifest: `checkpoints/weights.json`.

Licence: the weights are released under CC BY 4.0 (attribution by citing the manuscript; research software, not a medical device), as stated in `LICENSE` and in the release notes; the code itself is MIT-licensed.

| Run (each contains `final.pt`) | Bytes | SHA-256 |
|---|---:|---|
| v4_stage1_unified_polarity_fixed | 268408697 | `137ab2ee8c1fd9987e3ef535f56225067bd9a056b36ee03c23a4508031d8c687` |
| v4_stage2_final8ch_ALLDATA | 268410295 | `3cfd719983b857ea24a8b4472f355ba75e27a193cdebe999422ee6841f6a8beb` |
| v4_stage2_expA_dynpool_cv5_0 | 268410295 | `f1ccd3ca9f53c71920be60445118a7367dc7d478bb4568e809548a4130191d8d` |
| v4_stage2_expA_dynpool_cv5_1 | 268410295 | `b0f8a2a3a763c0271fed7b1cb6d2727864f5d9c58776ad1632dcd10290f3b313` |
| v4_stage2_expA_dynpool_cv5_2 | 268410295 | `292a29fb21256f0560d1866d912d0cbef8d1c868259c93bcb8f4308921a43522` |
| v4_stage2_expA_dynpool_cv5_3 | 268410295 | `082274a2953a6df382b2b284ac8bfb3b7a82faa22ad8e0b201133874e51c4d6b` |
| v4_stage2_expA_dynpool_cv5_4 | 268410295 | `40747fc398b7944859c4380e1616be9561774d0e0097cce4a91477ebc71e8c10` |
| v4_stage2_final8ch_scratch_cv5_0 | 268410295 | `99f64a3de484d82cba9813a8e75a862d44ed8d59f274c6e7e9c68e2c8bef2072` |
| v4_stage2_final8ch_scratch_cv5_1 | 268410295 | `3ab212ca3d7bbe6c3999a27ce89f73d6834a38cf522b47f0b53041f555750679` |
| v4_stage2_final8ch_scratch_cv5_2 | 268410295 | `238799418d65a6f53685320cb33a9c91f2ee3cb15f90a8c9df73877d37347258` |
| v4_stage2_final8ch_scratch_cv5_3 | 268410295 | `60545bc7d8b2405fa6b858e0b341ede6f164cd495ebf9a5f372eb267ac96d410` |
| v4_stage2_final8ch_scratch_cv5_4 | 268410295 | `3371aa7d300f5823613e5a5fa0dea568b1f3030be5101e2fdff02f26547e3564` |

Every run directory holds its `args.json` (the Stage-1 arguments were recovered from the checkpoint's own `args` field). The released weights use `data/zstats/global_zstats.json`; the loader binds a checkpoint to the digest of its statistics file, so `global_zstats_stage1_only.json` cannot be substituted for inference with these weights. The commands that produced the runs are in [training](../docs/training.md).
