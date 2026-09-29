# p4_cos_lr: run notes

- **Recipe:** `configs/experiments.yaml` → `p4_cos_lr` = `p3_lr02` (lr0 0.02) with `cos_lr: true`.
  It differs from `p3_lr02` only in the LR schedule (cosine instead of linear decay to lr0 × lrf).
- **Hardware:** one NVIDIA A100 80 GB PCIe on Nebula Cloud (the Colab free-tier GPU quota was used up).
  Software matches the Colab T4 runs: Ultralytics 8.4.162, torch 2.11.0+cu128, AMP on.
  Run with `python src/train.py p4_cos_lr --set workers=16` (more dataloader processes for the 28 vCPUs;
  this does not change the recipe). Training took 0.66 h, against 3.5 h on the T4.
- **Comparing with `p3_lr02` (T4):** the GPUs differ. Our cross-device replicate (`results/p3_lr005_mps/`)
  puts run-to-run noise at about 0.0003–0.0015 mAP50-95, so treat differences of that size as noise.
- **Result:** val mAP50-95 0.1820 vs 0.1826 for `p3_lr02`, and test 0.1514 vs 0.1529. Both differences
  are within noise, so cosine decay gave no measurable improvement over linear decay here.
