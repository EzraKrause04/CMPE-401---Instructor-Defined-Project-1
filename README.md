# CMPE 401 Instructor-defined Project 1: YOLO26 on VisDrone

**Design, optimization, and comparative evaluation of modern YOLO models for real-world object detection.**

This repo fine-tunes **YOLO26n** (Ultralytics) on **VisDrone2019-DET**, which has small objects in dense aerial scenes. It then analyses training dynamics, runs controlled experiments and an improvement cycle, and compares against **YOLOv8n**.

📄 **Report:** [`report/REPORT.md`](report/REPORT.md)

## Project map

| Assignment part | Run(s) in [`configs/experiments.yaml`](configs/experiments.yaml) | Output |
|---|---|---|
| I: Baseline | `baseline` | `results/baseline/` (metrics, curves, confusion matrix) |
| II: Loss and fitting analysis | `baseline` | `results/baseline/loss_curves.png`, `loss_summary.json` |
| III: Controlled experiment (image resolution) | `p3_imgsz960` vs `baseline` | `report/tables/part3.md` |
| IV: Improvement cycle | `p4_cos_lr` **or** `p4_regularize` | `report/tables/part4.md` |
| V: Multi-version comparison (optional) | `p5_yolov8n` vs `baseline` | `report/tables/part5.md` |

## Repository layout

```
configs/experiments.yaml   every run's settings (common -> base run -> overrides)
src/train.py               train a run by name; resumes automatically from last.pt
src/evaluate.py            metrics JSON (P, R, mAP, per-class AP, params, GFLOPs, speed, train time)
src/plot_curves.py         train vs val loss curves + over/underfitting summary
src/compare.py             markdown comparison tables for the report
notebooks/colab_runner.ipynb   one-click Colab wrapper around the scripts
results/<run>/             committed artifacts (weights are not committed)
report/REPORT.md           write-up
```

## Reproducing results

### On Google Colab (recommended)

1. Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Colab and select a GPU runtime.
2. Set `RUN = "baseline"` and run all cells. Checkpoints go to Google Drive (`MyDrive/cmpe401/runs`). If the session disconnects, run all cells again and training resumes.
3. Copy `MyDrive/cmpe401/results/<run>/` into this repo's `results/` folder.

Tip: before the first long run, set `SMOKE_TEST = True` to do 1 epoch on 5% of the data. That checks the whole pipeline in a few minutes.

### Locally (any CUDA machine)

```bash
pip install -r requirements.txt
python src/train.py baseline
python src/evaluate.py baseline --split val
python src/evaluate.py baseline --split test
python src/plot_curves.py baseline
```

### Building the report tables

```bash
python src/compare.py baseline p3_imgsz960 --out report/tables/part3.md
python src/compare.py baseline p4_cos_lr --out report/tables/part4.md
python src/compare.py baseline p5_yolov8n --out report/tables/part5.md
python src/plot_curves.py baseline p3_imgsz960
```

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) has 10 classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). The splits are 6,471 train, 548 val and 1,610 test-dev images. Ultralytics' built-in `VisDrone.yaml` downloads the data and converts it to YOLO format on first use; ignored regions are dropped. The `test` split here is **test-dev**, which has public labels. Test-challenge must be scored on the official VisDrone server.

## Fair-comparison controls

All runs share the settings in `common`: 80 epochs, SGD with lr0 = 0.01, batch 16, seed 0 and deterministic mode. Each experiment changes **only** the keys listed under its entry. `src/compare.py` prints those differences next to every result, so each table documents its own controlled variable.
