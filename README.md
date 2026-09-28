# CMPE 401 Instructor-defined Project 1: YOLO26 on VisDrone

**Authors:** Ezra Krause, Cole Robulack

**Design, optimization, and comparative evaluation of modern YOLO models for real-world object detection.**

This repo fine-tunes **YOLO26n** (Ultralytics) on **VisDrone2019-DET**, which has small objects in dense aerial scenes. It then analyses training dynamics, runs controlled experiments and an improvement cycle, and compares against **YOLOv8n**.

📄 **Report:** [`report/REPORT.md`](report/REPORT.md)

## Project map

| Assignment part | Run(s) in [`configs/experiments.yaml`](configs/experiments.yaml) | Output |
|---|---|---|
| I: Baseline | `baseline` | `results/baseline/` (metrics, curves, confusion matrix) |
| II: Loss and fitting analysis | `baseline` | `results/baseline/loss_curves.png`, `loss_summary.json` |
| II/III: Dataset size and object scale | none (label statistics) | `results/dataset/` (`src/dataset_stats.py`) |
| III: Controlled experiment (learning rate) | `p3_lr005`, `baseline`, `p3_lr02` (lr0 = 0.005 / 0.01 / 0.02) | `report/tables/part3.md` |
| IV: Improvement cycle | `p4_cos_lr` **or** `p4_regularize` | `report/tables/part4.md` |
| V: Multi-version comparison (optional) | `p5_yolov8n` vs `baseline` | `report/tables/part5.md` |
| Optional: testset-challenge | best run | `submission/<name>.zip` (`src/predict_challenge.py`) |

## Repository layout

```
configs/experiments.yaml   every run's settings (common -> base run -> overrides)
src/train.py               train a run by name; resumes automatically from last.pt
src/evaluate.py            metrics JSON (P, R, mAP, per-class AP, params, GFLOPs, speed, train time)
src/plot_curves.py         train vs val loss curves + over/underfitting summary
src/compare.py             markdown comparison tables for the report
src/dataset_stats.py       objects per image, class balance, box size vs detection stride at 640/960 px
src/predict_challenge.py   testset-challenge predictions in the official VisDrone submission format
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

### Locally (Apple silicon)

Ultralytics never selects the Apple GPU on its own, so pass it with `--set`. Only settings that don't change the recipe go there. `cache=ram` makes epochs about 1.5× faster on an M3 Pro. Ultralytics warns that it can make runs non-deterministic.

```bash
python src/train.py p5_yolov8n --set device=mps cache=ram workers=8
```

### Dataset statistics and challenge submission

```bash
python src/dataset_stats.py                                        # results/dataset/: stats.json + 3 figures
python src/predict_challenge.py --weights runs/<run>/weights/best.pt  # downloads testset-challenge on first use
```

### Building the report tables

```bash
python src/compare.py baseline p3_lr005 p3_lr02 --out report/tables/part3.md
python src/compare.py baseline p4_cos_lr --out report/tables/part4.md
python src/compare.py baseline p5_yolov8n --out report/tables/part5.md
python src/plot_curves.py baseline p3_lr005 p3_lr02
```

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) has 10 classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). The splits are 6,471 train, 548 val and 1,610 test-dev images. Ultralytics' built-in `VisDrone.yaml` downloads the data and converts it to YOLO format on first use; ignored regions are dropped. The `test` split here is **test-dev**, which has public labels. Test-challenge must be scored on the official VisDrone server.

## Fair-comparison controls

All runs share the settings in `common`: 80 epochs, SGD with lr0 = 0.01, batch 16, seed 0, deterministic mode, and `max_det=1000`. The last one matters because some VisDrone images contain around 900 objects, and Ultralytics' default of 300 detections per image would cap recall. Each experiment changes **only** the keys listed under its entry. `src/compare.py` prints those differences next to every result, so each table documents its own controlled variable.

## Deliverables breakdown

From the assignment brief (CMPE 401 Instructor-defined Project 1). Status as of 2026-09-28.

| Part | Required by the brief | Where it lives | Status |
|---|---|---|---|
| **I: Baseline** | Train YOLO26 (or YOLOv11) as the baseline. Record training and validation loss curves, mAP, precision and recall. | `baseline` run → `results/baseline/` | ✅ Trained on Colab T4 (val mAP50-95 0.181, mAP50 0.328). ⏳ Results folder not yet committed. |
| **II: Loss and fitting analysis** | Plot training vs validation loss. Identify convergence, discuss overfitting and underfitting, and explain causes with reference to **dataset size** and **model capacity**. | `results/baseline/loss_curves.png`, `loss_summary.json`, `results/dataset/`, report §3 | ⏳ Plots generated, dataset stats committed. Write-up to do. |
| **III: Controlled experiments** | At least one round of controlled experiments. Each states its settings, reports quantitative results, and gives analysis. | `p3_lr005`, `p3_lr02` (lr0 sweep around 0.01) → `report/tables/part3.md`, report §4 | ⏳ Runs defined, not yet trained. |
| **IV: Iterative improvement** | At least one improvement cycle grounded in deep learning principles: Baseline → Settings → Controlled modification → Evaluation → Analysis → Conclusion. Include a comparison table, performance discussion and justification. | `p4_cos_lr` or `p4_regularize` → `report/tables/part4.md`, report §5 | ⏳ Candidates defined. Choose after Part III. |
| **V: Multi-version comparison** *(optional)* | Compare against at least one other YOLO version (mAP, precision, recall, size, training time, speed, confusion matrix) in a structured table. | `p5_yolov8n` → `report/tables/part5.md`, report §6 | ✅ YOLOv8n trained (val mAP50-95 0.184, mAP50 0.330). ⏳ Table waits on baseline results. |
| **Challenge** *(optional, bonus)* | Submit the final model on VisDrone testset-challenge. Top class results with justified design choices count as exceeding expectations. | `src/predict_challenge.py` → `submission/` | ⏳ Script ready. Run with the final model. |
| **Reproducibility** | Present reproducible results via GitHub. | This README, `configs/experiments.yaml`, `src/`, `notebooks/colab_runner.ipynb` | ✅ Pipeline in place. Keep committing each run's `results/<run>/`. |
