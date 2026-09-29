# baseline: provenance

Ezra trained the baseline (YOLO26n, lr0 0.01, 80 epochs) on a Colab T4 with Ultralytics 8.4.162 and
torch 2.11.0+cu128. The full run directory is on his Google Drive (`MyDrive/cmpe401/`).

The files here are copied verbatim from the evaluation outputs saved in the repo history: the output cells of
`notebooks/colab_runner.ipynb` at commit `fe0b017`, where `src/evaluate.py` and `src/plot_curves.py` printed
these JSONs and displayed the plot.

- `metrics_val.json`, `metrics_test.json`: printed by `src/evaluate.py baseline --split val|test`
- `loss_summary.json`: printed by `src/plot_curves.py baseline`
- `loss_curves.png`: the figure that cell displayed

**Still to add from Ezra's Drive:** `results.csv`, `args.yaml`, the confusion matrices and the PR/F1 curves.
