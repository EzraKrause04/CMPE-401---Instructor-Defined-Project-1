# CMPE 401 Project 1: YOLO26 on VisDrone

**Authors:** Ezra Krause, Cole Robulack

We fine-tune **YOLO26n** on **VisDrone2019-DET** (small objects in dense aerial scenes), analyse its training, improve it through controlled experiments, and compare it with **YOLOv8n**.

📄 Full write-up: [`report/REPORT.md`](report/REPORT.md)

## Results

mAP50-95 on VisDrone test-dev (val in brackets). All runs: 80 epochs, 640 px, SGD, batch 16, seed 0.

| Part | Run | Change | Test mAP | Takeaway |
|---|---|---|---|---|
| I | `baseline` | YOLO26n, lr0 = 0.01 | 0.146 (0.181) | Small classes (people, bicycle) score lowest |
| II | — | Loss analysis | — | No overfitting; limited by small objects / resolution |
| III | `p3_lr005` | lr0 = 0.005 | 0.144 (0.173) | |
| III | **`p3_lr02`** | **lr0 = 0.02** | **0.153 (0.183)** | **Best model** |
| IV | `p4_cos_lr` | + cosine LR decay | 0.151 (0.182) | No measurable change |
| V | `p5_yolov8n` | YOLOv8n | 0.145 (0.184) | Tied with YOLO26n, which is 21% smaller |

Run-to-run noise is about 0.002 mAP (from repeating `p3_lr005` on another machine).

## Layout

```
configs/experiments.yaml   settings for every run
src/                       train, evaluate, plot, compare, dataset stats, speed benchmark, challenge predictions
notebooks/colab_runner.ipynb   one-click Colab runner
results/<run>/             metrics, loss curves, confusion matrices (no weights)
report/                    write-up and comparison tables
```

## Reproduce

```bash
pip install -r requirements.txt
python src/train.py p3_lr02
python src/evaluate.py p3_lr02 --split test
python src/plot_curves.py p3_lr02
```

On Colab, open `notebooks/colab_runner.ipynb`, set `RUN`, and run all cells. On Apple silicon, add `--set device=mps`. See [report §8](report/REPORT.md#8-reproducibility) for the details.

Weights (`best.pt`) aren't committed; retrain with `src/train.py`. The dataset downloads automatically on first run.

## Not done

- Baseline `results.csv` and confusion matrix are still on Google Drive.
- 960 px experiment (`p4_imgsz960`) and the VisDrone test-challenge submission were not run.
