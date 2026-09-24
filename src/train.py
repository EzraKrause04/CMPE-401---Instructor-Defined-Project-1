"""Train one experiment from configs/experiments.yaml, resuming if a checkpoint exists.

    python src/train.py baseline
    python src/train.py baseline --epochs 1 --fraction 0.05   # quick smoke test
    python src/train.py p5_yolov8n --set device=mps cache=ram workers=8   # local Apple-silicon run
"""

import argparse
import shutil

import torch
import yaml
from ultralytics import YOLO

from common import resolve_run, results_dir, run_dir, RUNS_DIR

# Ultralytics outputs worth keeping in the repo (weights stay out of git).
KEEP = [
    "results.csv", "args.yaml", "results.png", "labels.jpg",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
    "val_batch0_pred.jpg", "val_batch0_labels.jpg",
]


def training_finished(last_pt):
    # Ultralytics strips the optimizer and sets epoch=-1 in last.pt once training completes.
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) == -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--epochs", type=int, help="override epochs (smoke tests only)")
    ap.add_argument("--fraction", type=float, help="train on a fraction of the dataset (smoke tests only)")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="machine-specific settings that don't change the recipe, e.g. device=mps cache=ram workers=8",
    )
    args = ap.parse_args()

    params, meta = resolve_run(args.run)
    smoke = bool(args.epochs or args.fraction)
    if args.epochs:
        params["epochs"] = args.epochs
    if args.fraction:
        params["fraction"] = args.fraction
    params.update({k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    model_name = params.pop("model")
    # Smoke tests get their own folder so they never clobber or resume a real run.
    name = f"{args.run}_smoke" if smoke else args.run

    out = run_dir(name)
    last = out / "weights" / "last.pt"
    print(f"== {name} (Part {meta['part']}): {meta['description']}")

    if last.exists() and not smoke and training_finished(last):
        print(f"{name} already finished; skipping training (delete {out} to retrain)")
    elif last.exists() and not smoke:
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        print(yaml.safe_dump({"model": model_name, **params}, sort_keys=False))
        YOLO(model_name).train(**params, project=str(RUNS_DIR), name=name, exist_ok=True)

    if smoke:
        print(f"Smoke test finished; outputs in {out} (not copied to results/)")
        return
    dest = results_dir(args.run)
    for fname in KEEP:
        if (out / fname).exists():
            shutil.copy2(out / fname, dest / fname)
    print(f"Copied run artifacts to {dest}")


if __name__ == "__main__":
    main()
