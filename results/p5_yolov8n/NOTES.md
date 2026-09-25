# p5_yolov8n: run notes

- **Recipe:** exactly `configs/experiments.yaml` → `p5_yolov8n` (80 epochs, SGD lr0 0.01, batch 16, 640 px, seed 0, max_det 1000).
- **Hardware:** Apple M3 Pro (MPS), trained with `python src/train.py p5_yolov8n --set device=mps cache=ram workers=8`. These settings change where and how fast it runs, not the recipe. Ultralytics warns that `cache=ram` can make runs non-deterministic.
- **Training time:** `train_time_hours` in `metrics_*.json` (8.1 h) is wall-clock and **includes about 3 h when the laptop slept**. Epochs 46, 49 and 50 took 100, 33 and 57 min instead of ~3.8 min. Actual compute: **≈ 5.1 h** (80 × median epoch time of 3.77 min). Use ≈ 5.1 h in comparison tables.
- **Inference speed:** `speed_ms_per_img` in `metrics_*.json` was measured on the **CPU**, because `src/evaluate.py` doesn't set a device. It is not comparable with a GPU-timed run. Re-time all Part V models on one device before comparing speed.
- **Convergence:** validation loss reached its minimum at epoch 79/80 (`loss_summary.json`), so the model was still improving when training stopped.
