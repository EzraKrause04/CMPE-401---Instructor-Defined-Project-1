# p3_lr005_mps: cross-device replicate of p3_lr005

This is **not** a separate experiment. It repeats `p3_lr005` (lr0 = 0.005, everything else as the baseline) on a
different machine, to estimate run-to-run noise for the Part III comparison.

- **Hardware:** Apple M3 Pro (MPS), `python src/train.py p3_lr005 --set device=mps cache=ram workers=8`.
  Ultralytics 8.4.162, the same version as the Colab runs. On MPS, Ultralytics trains without AMP and with
  a single dataloader process, so it is slower (6.94 h vs 3.64 h on the T4) but uses the same recipe.
- **The Part III results are the Colab T4 runs** (`results/p3_lr005/`, `results/baseline/`, `results/p3_lr02/`),
  all on the same hardware. Use this folder only for the noise estimate.

| p3_lr005 copy | val mAP50-95 | test mAP50-95 |
|---|---|---|
| Colab T4 (`results/p3_lr005/`) | 0.1727 | 0.1441 |
| Apple M3 Pro (this folder) | 0.1724 | 0.1426 |
| difference | 0.0003 | 0.0015 |

Differences in Part III smaller than about 0.002 mAP50-95 should be treated as within noise.
