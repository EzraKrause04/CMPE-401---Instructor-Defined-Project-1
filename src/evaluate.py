"""Evaluate a trained run's best.pt and write results/<run>/metrics_<split>.json.

    python src/evaluate.py baseline               # VisDrone val split
    python src/evaluate.py baseline --split test  # VisDrone test-dev (labels are public)
"""

import argparse
import json

import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from common import resolve_run, results_dir, run_dir, RUNS_DIR


def training_stats(csv_path):
    """Epochs run, best epoch, and wall-clock training time from Ultralytics' results.csv."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    stats = {"epochs_completed": int(df["epoch"].max())}
    if "metrics/mAP50-95(B)" in df:
        stats["best_epoch"] = int(df.loc[df["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    if "time" in df:  # cumulative seconds
        stats["train_time_hours"] = round(float(df["time"].iloc[-1]) / 3600, 2)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    params, meta = resolve_run(args.run)
    weights = run_dir(args.run) / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}; train the run first.")

    model = YOLO(str(weights))
    m = model.val(
        data=params["data"], imgsz=params["imgsz"], batch=params["batch"], split=args.split,
        max_det=params.get("max_det", 300),
        plots=True, project=str(RUNS_DIR), name=f"{args.run}_eval_{args.split}", exist_ok=True,
    )

    n_params = get_num_params(model.model)
    gflops = get_flops(model.model, params["imgsz"])  # at the run's input size
    names = m.names
    out = {
        "run": args.run,
        "part": meta["part"],
        "description": meta["description"],
        "split": args.split,
        "model": params["model"],
        "imgsz": params["imgsz"],
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        # Only classes present in the split (Ultralytics pads absent classes with the mean mAP).
        "per_class_mAP50-95": {names[int(i)]: round(float(m.box.maps[int(i)]), 4) for i in m.box.ap_class_index},
        "params_M": round(n_params / 1e6, 2),
        "GFLOPs": round(float(gflops), 1),
        "weights_MB": round(weights.stat().st_size / 1e6, 1),
        "speed_ms_per_img": {k: round(v, 2) for k, v in m.speed.items()},
        **training_stats(run_dir(args.run) / "results.csv"),
    }

    dest = results_dir(args.run) / f"metrics_{args.split}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
