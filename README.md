# CMPE 401 Instructor-defined Project 1: YOLO26 on VisDrone

**Authors:** Ezra Krause, Cole Robulack

**Design, optimization, and comparative evaluation of modern YOLO models for real-world object detection.**

This repo fine-tunes **YOLO26n** (Ultralytics) on **VisDrone2019-DET**, which has small objects in dense aerial scenes. It then analyses training dynamics, runs controlled experiments and an improvement cycle, and compares against **YOLOv8n**.

📄 **Report:** [`report/REPORT.md`](report/REPORT.md)

## Objective

Fine-tune a modern YOLO detector on VisDrone2019-DET, analyse its training and validation loss behaviour, improve it through controlled experiments, and compare it with another YOLO version, with every result reproducible from this repo.

## Models

| Model | Checkpoint | Parameters / GFLOPs at 640 px (as trained, 10 classes) | Role |
|---|---|---|---|
| YOLO26n | `yolo26n.pt` (COCO-pretrained) | 2.51 M / 5.9 | Baseline and all Part II–IV runs |
| YOLOv8n | `yolov8n.pt` (COCO-pretrained) | 3.01 M / 8.2 | Part V comparison |

**Final model:** `p3_lr02` (YOLO26n, lr0 = 0.02), test-dev mAP50-95 0.153. Weights are not distributed in the repo; see [Weights](#weights).

## Key findings

The full analysis is in [`report/REPORT.md`](report/REPORT.md). mAP values are mAP50-95 on VisDrone test-dev unless noted.

- **Baseline (YOLO26n, 640 px, 80 epochs):** 0.146 on test-dev (mAP50 0.270) and 0.181 on val. Small objects are the weak point. At 640 px, 33% of training boxes (40.5% on test-dev) are smaller than one stride-8 cell, and the smallest classes score lowest: people 0.037, bicycle 0.026, against car 0.407.
- **No overfitting.** Validation loss ends within 0.23% of its minimum (largest transient +0.76% at the mosaic switch), and every YOLO26n run peaks at epochs 63–70. The model is predominantly bias-limited. Our working hypothesis is that input resolution (and possibly capacity) is the main limit; no resolution or model-size change has been tested yet.
- **Learning rate (Part III):** lr0 = 0.02 beat 0.01 and 0.005. It reaches 0.153 on test-dev (+0.0066 over the baseline, about 4× the run-to-run noise we measured). On val the gain is within noise.
- **Cosine LR decay (Part IV):** no measurable change against linear decay (0.151 vs 0.153 on test-dev, 0.182 vs 0.183 on val), so `p3_lr02` stays the final model. A second cycle at 960 px (`p4_imgsz960`) is outside the scope of this submission and has not been run.
- **YOLOv8n vs YOLO26n (Part V):** accuracy is tied (0.145 vs 0.146 on test-dev, 0.184 vs 0.181 on val). YOLO26n has 21% fewer parameters and 34% fewer GFLOPs after fusing for inference (17% and 28% as trained), but it is not faster at batch 1 on the same Mac. On MPS it takes 9.9 ms vs 7.6 ms per image, and on CPU 25.2 ms vs 25.9 ms. Both were evaluated and timed with NMS: under Ultralytics' default `nms=None`, YOLO26n uses its one-to-many head plus NMS, not its NMS-free head.

## Project map

| Assignment part | Run(s) in [`configs/experiments.yaml`](configs/experiments.yaml) | Output |
|---|---|---|
| I: Baseline | `baseline` | `results/baseline/`: val/test metrics, loss curves and loss summary. The baseline's `results.csv` and confusion matrix are still on Ezra's Drive (see [`results/baseline/NOTES.md`](results/baseline/NOTES.md)) |
| II: Loss and fitting analysis | `baseline`, plus the per-epoch logs of every other run | `results/<run>/loss_curves.png`, `loss_summary.json`, `results.csv` (all runs except baseline); report §3 |
| II/III: Dataset size and object scale | none (label statistics) | `results/dataset/` (`src/dataset_stats.py`) |
| III: Controlled experiment (learning rate) | `p3_lr005`, `baseline`, `p3_lr02` (lr0 = 0.005 / 0.01 / 0.02) | `report/tables/part3_{val,test}.md`, `results/compare_p3_lr005_vs_p3_lr02_vs_p4_cos_lr.png` |
| III: Run-to-run noise estimate | `p3_lr005` repeated on an Apple M3 Pro | `results/p3_lr005_mps/` |
| IV: Improvement cycle | `p4_cos_lr` (cosine LR decay), compared against `p3_lr02` | `report/tables/part4_{val,test}.md`, `results/p4_cos_lr/` |
| V: Multi-version comparison (optional) | `p5_yolov8n` vs `baseline` | `report/tables/part5_{val,test}.md`, `results/speed/` (same-device speed) |
| Optional: testset-challenge | best run | `submission/<name>.zip` (`src/predict_challenge.py`). Not submitted |

## Repository layout

```
configs/experiments.yaml   every run's settings (common -> base run -> overrides)
src/train.py               train a run by name; resumes automatically from last.pt
src/evaluate.py            metrics JSON (P, R, mAP, per-class AP, params, GFLOPs, speed, train time)
src/plot_curves.py         train vs val loss curves + over/underfitting summary
src/compare.py             markdown comparison tables for the report
src/dataset_stats.py       objects per image, class balance, box size vs detection stride at 640/960 px
src/predict_challenge.py   testset-challenge predictions in the official VisDrone submission format
src/benchmark_speed.py     same-device inference timing (batch 1, median ms/img) + fused params/GFLOPs
notebooks/colab_runner.ipynb   one-click Colab wrapper around the scripts
results/<run>/             committed artifacts (weights are not committed); NOTES.md (where present) records hardware/provenance
results/speed/             speed_mps.json, speed_cpu.json (src/benchmark_speed.py)
report/REPORT.md           write-up
report/tables/             comparison tables generated by src/compare.py
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

Ultralytics never selects the Apple GPU on its own, so pass it with `--set`. Only machine-specific settings that leave the hyperparameters unchanged go there. `cache=ram` speeds up MPS epochs. Ultralytics warns that it can make runs non-deterministic, and it also changes how mosaic picks its partner images (the whole dataset instead of a rolling buffer), so the augmentation stream differs from an uncached run.

```bash
python src/train.py p5_yolov8n --set device=mps cache=ram workers=8
```

On MPS, Ultralytics turns off AMP and loads data in the main process whatever `workers` is set to. Together with the different GPU, this makes MPS epochs about 2× slower than on a T4 (316 vs 164 s median for `p3_lr005`).

**Cross-device replicate (`results/p3_lr005_mps/`).** `src/train.py` writes to `results/<run>/`, so repeating `p3_lr005` in place would overwrite the T4 results. Point the output directories elsewhere, then copy:

```bash
export RUNS_DIR=~/runs-mps RESULTS_DIR=/tmp/results-mps
python src/train.py p3_lr005 --set device=mps cache=ram workers=8
python src/evaluate.py p3_lr005 --split val && python src/evaluate.py p3_lr005 --split test
python src/plot_curves.py p3_lr005
cp $RESULTS_DIR/p3_lr005/* results/p3_lr005_mps/
```

### On a rented CUDA GPU

The Part IV run used an A100 on Nebula Cloud because the Colab GPU quota ran out. The only extra setting was more dataloader workers. That leaves the hyperparameters unchanged, though it changes the per-worker mosaic buffers:

```bash
python src/train.py p4_cos_lr --set workers=16
```

### Dataset statistics, speed benchmark and challenge submission

```bash
python src/dataset_stats.py                                        # results/dataset/: stats.json + 3 figures
python src/benchmark_speed.py --model YOLO26n=runs/p3_lr02/weights/best.pt \
    --model YOLOv8n=runs/p5_yolov8n/weights/best.pt --device mps  # results/speed/speed_mps.json (also --device cpu)
python src/predict_challenge.py --weights runs/<run>/weights/best.pt  # downloads testset-challenge on first use
```

### Building the report tables

```bash
python src/compare.py baseline p3_lr005 p3_lr02 --out report/tables/part3_val.md
python src/compare.py baseline p3_lr005 p3_lr02 --split test --out report/tables/part3_test.md
python src/compare.py p3_lr02 p4_cos_lr --out report/tables/part4_val.md
python src/compare.py p3_lr02 p4_cos_lr --split test --out report/tables/part4_test.md
python src/compare.py baseline p5_yolov8n --out report/tables/part5_val.md
python src/compare.py baseline p5_yolov8n --split test --out report/tables/part5_test.md
python src/plot_curves.py p3_lr005 p3_lr02 p4_cos_lr   # overlay; the baseline's results.csv is not in the repo yet
```

The `Inference (ms/img)` and `Train time (h)` columns in these tables come from whichever machine each run used: T4 for the baseline and Part III, A100 for Part IV, and the Mac CPU (speed) plus about 3 h of laptop sleep (time) for `p5_yolov8n`. Don't compare them across runs; each generated table carries a footnote saying so. Use `results/speed/` for speed and report §1.4 for training time (the corrected ≈5.1 h for `p5_yolov8n` is explained in [`results/p5_yolov8n/NOTES.md`](results/p5_yolov8n/NOTES.md)).

### Weights

`best.pt` files are not committed (`.gitignore`) and are not distributed with the repo. They are on the team's machines and Google Drive. To reproduce a model, train it with `python src/train.py <run>` (the final model is `p3_lr02`). To re-evaluate a checkpoint you have, put it at `runs/<run>/weights/best.pt` and run `src/evaluate.py`.

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) has 10 classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor). The splits are 6,471 train, 548 val and 1,610 test-dev images. Ultralytics' built-in `VisDrone.yaml` downloads the data and converts it to YOLO format on first use; ignored regions are dropped. Ultralytics then scores detections inside ignored regions as false positives, unlike the official VisDrone toolkit, so our test-dev mAP is not directly comparable with published VisDrone results. The `test` split here is **test-dev**, which has public labels. Test-challenge must be scored on the official VisDrone server.

## Fair-comparison controls

All runs share the settings in `common`: 80 epochs, SGD with lr0 = 0.01, batch 16, seed 0, deterministic mode, and `max_det=1000`. Val and test-dev images contain up to 317 and 461 objects; pinning `max_det=1000` gives every run and split the same cap, where Ultralytics 8.4.162 would otherwise raise its default of 300 to the largest object count it observes. Each experiment changes **only** the recipe keys listed under its entry. `src/compare.py` prints those differences next to every result, so each table documents its own controlled variable. The machine-specific `--set` values are not recipe keys, but `cache=ram` (the MPS runs) and `workers` (`p4_cos_lr`) still change the mosaic augmentation stream. Hardware was not the same for every run. The three Part III runs all used a Colab T4. Part IV used an A100, and `p5_yolov8n` used an Apple M3 Pro. A repeat of `p3_lr005` on the M3 Pro (`results/p3_lr005_mps/`) differed by 0.0003 (val) and 0.0015 (test-dev) mAP50-95, which we use as a rough scale of run-to-run noise. Report §1.4 lists the hardware for each run.

## Deliverables breakdown

From the assignment brief (CMPE 401 Instructor-defined Project 1). Status as of 2026-09-29.

| Part | Required by the brief | Where it lives | Status |
|---|---|---|---|
| **I: Baseline** | Train YOLO26 (or YOLOv11) as the baseline. Record training and validation loss curves, mAP, precision and recall. | `baseline` run → `results/baseline/`, report §2 | ✅ Trained on a Colab T4. Test-dev mAP50-95 0.146 (mAP50 0.270); val 0.181 (0.328). Metrics, loss curves and loss summary are committed. ⏳ `results.csv`, `args.yaml`, confusion matrices and PR curves are still on Ezra's Drive. |
| **II: Loss and fitting analysis** | Plot training vs validation loss. Identify convergence, discuss overfitting and underfitting, and explain causes with reference to **dataset size** and **model capacity**. | `results/*/loss_curves.png`, `loss_summary.json`, `results/dataset/`, report §3 | ✅ Written. Diagnosis: no overfitting; predominantly bias-limited, with input resolution (and possibly capacity) as the working hypothesis. |
| **III: Controlled experiments** | At least one round of controlled experiments. Each states its settings, reports quantitative results, and gives analysis. | `p3_lr005`, `baseline`, `p3_lr02` → `report/tables/part3_{val,test}.md`, report §4 | ✅ lr0 sweep done, all runs on a T4. lr0 = 0.02 is best: test-dev 0.153 vs 0.146. Noise was estimated from a repeat run on an M3 Pro. |
| **IV: Iterative improvement** | At least one improvement cycle grounded in deep learning principles: Baseline → Settings → Controlled modification → Evaluation → Analysis → Conclusion. Include a comparison table, performance discussion and justification. | `p4_cos_lr` vs `p3_lr02` → `report/tables/part4_{val,test}.md`, report §5 | ✅ Cycle 1 (cosine LR decay, trained on an A100): no measurable change, so `p3_lr02` is kept (+0.0066 test-dev over the baseline; trajectory table in report §5.6). Cycle 2 (960 px, `p4_imgsz960`) is outside the scope of this submission. |
| **V: Multi-version comparison** *(optional)* | Compare against at least one other YOLO version (mAP, precision, recall, size, training time, speed, confusion matrix) in a structured table. | `p5_yolov8n` vs `baseline` → `report/tables/part5_{val,test}.md`, `results/speed/`, report §6 | ✅ Table done, with inference speed measured on the same device. The two models are tied on accuracy. The baseline's confusion matrix is still pending, so the report uses the `p3_lr02` matrix instead. |
| **Challenge** *(optional, bonus)* | Submit the final model on VisDrone testset-challenge. Top class results with justified design choices count as exceeding expectations. | `src/predict_challenge.py` → `submission/` | ⏳ Not submitted. The script is ready. |
| **Reproducibility** | Present reproducible results via GitHub. | This README, `configs/experiments.yaml`, `src/`, `notebooks/colab_runner.ipynb`, report §8 | ✅ Code, recipes, results and instructions are in the repo. Weights are not in git; see [Weights](#weights). |

## Write-up outline (draft)

A section-by-section skeleton for the final write-up. Each section has the verified results, the figures to include, and prompts for what to write. Numbers come from `results/<run>/metrics_{val,test}.json`, `loss_summary.json`, `results/dataset/stats.json` and `results/speed/`. **mAP** means mAP50-95 unless noted. **Test** means VisDrone test-dev.

### 1. Introduction
**To write:**
- Why real-time detection matters (drones, surveillance, smart cities) and why VisDrone is hard: tiny objects in dense aerial scenes.
- The goal: fine-tune YOLO26n, understand its training dynamics, improve it through controlled experiments, and compare it with YOLOv8n.
- A one-paragraph summary of the findings.

### 2. Dataset and setup
| | Train | Val | Test-dev |
|---|---|---|---|
| Images | 6,471 | 548 | 1,610 |
| Boxes | 343,204 | 38,759 | — |

- **Objects per image (train):** mean 53, median 42, max 902. This is why `max_det` was raised from 300 to 1000.
- **Class imbalance (train):** car is 42.2% of boxes, awning-tricycle 0.95% (44.6× ratio).
- **Object size:** at 640 px, 33% of train boxes are smaller than 8 px (one stride-8 cell) and 68% are smaller than 16 px. At 960 px those drop to 17% and 48%.
- **Shared recipe:** 80 epochs, SGD, lr0 0.01, linear decay, batch 16, 640 px, seed 0, Ultralytics 8.4.162.

| Run | Hardware | Training time |
|---|---|---|
| `baseline`, `p3_lr005`, `p3_lr02` | Colab T4 | 3.5–3.8 h each |
| `p4_cos_lr` | A100 (rented) | 0.7 h |
| `p5_yolov8n` | Apple M3 Pro | 8.1 h (includes ~3 h laptop sleep) |
| `p3_lr005_mps` (noise check) | Apple M3 Pro | 6.9 h |

**Figures:** `results/dataset/class_distribution.png`, `objects_per_image.png`, `box_size_vs_stride.png`

### 3. Part I: Baseline (YOLO26n, lr0 0.01)
| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Val | 0.439 | 0.344 | 0.328 | 0.181 |
| Test | 0.387 | 0.307 | 0.270 | 0.146 |

**Per class (test mAP):** car 0.407, bus 0.322, van 0.190, truck 0.188, motor 0.091, pedestrian 0.086, awning-tricycle 0.061, tricycle 0.057, people 0.037, bicycle 0.026.

**Figures:** `results/baseline/loss_curves.png`. Still to add from Drive: confusion matrix and `results.csv`.

**To write:** what the metrics mean; why val scores higher than test; why the small classes (people, bicycle) are so far behind car.

### 4. Part II: Loss curves and fitting analysis
- **Convergence:** val loss bottoms out at epoch 70, and mAP flattens after about epoch 50.
- **Overfitting:** none. Val loss ends only 0.11% above its minimum, and the final val–train gap is small (0.15).
- **Mosaic effect:** val loss sits *below* train loss for most of training, because mosaic augmentation makes training images harder. Mosaic switches off at epoch 71, train loss drops below val, and mAP gets a final bump.
- **Underfitting:** both losses stay high and mAP plateaus low, which suggests the model is limited by capacity or resolution rather than by training time.

**Discuss, as required by the brief:**
- **Dataset size:** only 6.5k images, but 343k boxes and a 44.6× class imbalance. Rare classes have few examples.
- **Model capacity:** YOLO26n has about 2.4M parameters. A third of objects are smaller than one stride-8 cell at 640 px, so detail is lost before the network ever sees it.

**Figures:** `results/baseline/loss_curves.png`

### 5. Part III: Controlled experiment (initial learning rate)
**Setup:** only `lr0` changes. Everything else matches the baseline, and all three runs are on a T4.

| lr0 | Val mAP | Test mAP | Test mAP50 | Test P | Test R | Val loss min epoch |
|---|---|---|---|---|---|---|
| 0.005 | 0.173 | 0.144 | 0.265 | 0.381 | 0.300 | 68 |
| 0.01 (baseline) | 0.181 | 0.146 | 0.270 | 0.387 | 0.307 | 70 |
| **0.02** | **0.183** | **0.153** | **0.280** | **0.406** | **0.312** | 79 |

- **Noise:** repeating `p3_lr005` on a different machine changed mAP by 0.0003 (val) and 0.0015 (test). Differences under about 0.002 aren't meaningful.
- **Result:** accuracy rises with lr0 on both splits. On test, 0.02 beats the baseline by +0.0066 (about 4× the noise); on val the gain (+0.0018) is within noise.
- **Dynamics:** at 0.02, val loss is still at its minimum at epoch 79, so it was still improving. At 0.005 it bottoms out at 68.

**Figures:** `results/compare_p3_lr005_vs_p3_lr02_vs_p4_cos_lr.png`, plus `report/tables/part3_{val,test}.md`

**To write:** why a higher learning rate helps here (80 epochs isn't long enough at 0.01), why too high a rate could hurt pretrained weights, and why only one variable was changed.

### 6. Part IV: Iterative improvement (cosine LR decay)
**Chain:** baseline `p3_lr02` → settings → modification (`cos_lr: true`) → evaluation → analysis → conclusion.

| Run | Schedule | Val mAP | Test mAP | Test mAP50 | Test P | Test R |
|---|---|---|---|---|---|---|
| `p3_lr02` | Linear | 0.183 | 0.153 | 0.280 | 0.406 | 0.312 |
| `p4_cos_lr` | Cosine | 0.182 | 0.151 | 0.278 | 0.394 | 0.317 |

- **Result:** no measurable gain. Both differences (−0.0006 val, −0.0015 test) are within noise.
- **Caveat:** `p4_cos_lr` trained on an A100, not a T4.
- **Conclusion:** `p3_lr02` stays the final model.

**To write:**
- **Justification:** why cosine decay should help in principle (a longer high-LR phase, then a gentle settle at the end).
- **Why it didn't:** both schedules end at the same final rate after the same 80 epochs.
- **Next cycle:** input resolution at 960 px, motivated by the 33% → 17% drop in sub-8 px boxes.

### 7. Part V: YOLO26n vs YOLOv8n (optional)
| Model | Val mAP | Test mAP | Test mAP50 | Params (fused) | GFLOPs (fused) | Weights | MPS ms/img | CPU ms/img |
|---|---|---|---|---|---|---|---|---|
| YOLO26n (baseline) | 0.181 | 0.146 | 0.270 | 2.38 M | 5.3 | 5.4 MB | 9.9 | 25.2 |
| YOLOv8n | 0.184 | 0.145 | 0.262 | 3.01 M | 8.1 | 6.2 MB | 7.6 | 25.9 |

Speeds are batch 1 at 640 px on the same Apple M3 Pro, the median of 200 images, with NMS for both models.

**Per class (test mAP):** YOLOv8n is lower on bicycle (0.019 vs 0.026) and motor (0.079 vs 0.091), and about equal elsewhere.

**To write:**
- Accuracy is tied within noise.
- YOLO26n is about 21% smaller with 34% fewer FLOPs, but isn't faster at batch 1. Explain why: post-processing dominates, and its NMS-free head isn't used by default.
- Compare the two confusion matrices.

### 8. Conclusions and limitations
**To write:**
- **Findings:** the best model is `p3_lr02` (test mAP 0.153). Learning rate mattered and the LR schedule didn't. Small objects are the main bottleneck.
- **Limitations:** one seed per run, mixed hardware for Parts IV and V, and no resolution or model-size experiment.
- **Future work:** 960 px input, the `s` model, and a test-challenge submission.

### 9. Reproducibility
**To write:** point to `configs/experiments.yaml`, `src/`, the Colab notebook and the "Reproducing results" section above. Note that weights aren't in git.
