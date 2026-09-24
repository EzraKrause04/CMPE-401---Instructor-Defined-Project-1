# Design, Optimization, and Comparative Evaluation of Modern YOLO Models for Real-World Object Detection

CMPE 401: Instructor-defined Project 1 · Ezra Krause

> **Status:** template. Sections marked `TODO` are filled in once the corresponding runs finish.
> All numbers come from `results/<run>/metrics_*.json` and `report/tables/*.md`.

## 1. Setup

- **Model:** YOLO26n (Ultralytics `yolo26n.pt`, COCO-pretrained)
- **Dataset:** VisDrone2019-DET (6,471 train / 548 val / 1,610 test-dev images, 10 classes)
- **Hardware:** TODO (e.g. Colab T4 16 GB)
- **Common settings:** 80 epochs, 640 px, batch 16, SGD lr0 = 0.01, seed 0 (see `configs/experiments.yaml`)

## 2. Part I: Baseline

TODO: results table from `python src/compare.py baseline` (val) and `--split test`.

| Metric | Val | Test-dev |
|---|---|---|
| Precision | | |
| Recall | | |
| mAP50 | | |
| mAP50-95 | | |

Per-class AP: TODO (from `per_class_mAP50-95` in the metrics JSON; highlight the small classes: pedestrian, people, bicycle, motor).

![Baseline loss curves](../results/baseline/loss_curves.png)
![Baseline confusion matrix](../results/baseline/confusion_matrix_normalized.png)

## 3. Part II: Loss curves and fitting analysis

Use `results/baseline/loss_summary.json` for the numbers.

- **Convergence:** TODO. Which epoch does validation loss reach its minimum? Does mAP plateau?
- **Overfitting:** TODO. Does validation loss rise after its minimum (`val_rise_after_min_pct`)? Is the train–val gap growing? Note the drop in both losses when mosaic turns off for the last 10 epochs (`close_mosaic`).
- **Underfitting:** TODO. Are both losses still falling at the last epoch? Are absolute losses high?
- **Causes:** discuss with reference to
  - *Dataset size:* 6.5k images, but dense scenes with many labelled objects per image (count them from `results/baseline/labels.jpg`), and a long tail of rare classes (awning-tricycle, bus).
  - *Model capacity:* YOLO26n has about 2.4M parameters. Is that enough for dense, tiny objects at 640 px?

## 4. Part III: Controlled experiment (input resolution)

- **Hypothesis:** VisDrone objects are often under 32 px. Raising input resolution from 640 to 960 px keeps more pixels per object, so small-object recall and mAP should improve, at higher compute cost.
- **Controlled variable:** `imgsz` 640 → 960. Everything else is identical.

TODO: paste `report/tables/part3.md`, add `results/compare_baseline_vs_p3_imgsz960.png`, and analyse accuracy against speed and GFLOPs.

## 5. Part IV: Iterative improvement

Baseline → Experimental settings → Controlled modification → Evaluation → Analysis → Conclusion

- **Diagnosis from Part II:** TODO
- **Modification and justification:** TODO. Use `p4_cos_lr` for a late-training plateau or oscillation, or `p4_regularize` for overfitting.
- **Evaluation:** TODO. Paste `report/tables/part4.md`.
- **Analysis and conclusion:** TODO

## 6. Part V: YOLO26n vs YOLOv8n (optional)

TODO: paste `report/tables/part5.md` (mAP, P/R, params, GFLOPs, weight size, training time and inference speed). Compare the confusion matrices. Discuss YOLO26's NMS-free head, DFL removal and STAL small-target assignment against YOLOv8's design.

## 7. Conclusions

TODO

## 8. Reproducibility

See the [README](../README.md). Each result maps to one named run in `configs/experiments.yaml`.
