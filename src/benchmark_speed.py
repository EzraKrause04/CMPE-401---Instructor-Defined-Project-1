"""Time inference for several checkpoints on one device, for the Part V speed column.

evaluate.py reports whatever device each run happened to be evaluated on (a Colab T4 for the
YOLO26n runs, the Mac CPU for p5_yolov8n), so its speed numbers are not comparable across runs.
This script times every model on the same device, one image at a time (batch 1), after a
warm-up, and reports the median per-image preprocess / inference / postprocess time. It also
reports fused parameter counts and GFLOPs. Speed depends only on the architecture, so any
checkpoint of that architecture will do.

    python src/benchmark_speed.py --model YOLO26n=runs/p3_lr02/weights/best.pt \
        --model YOLOv8n=runs/p5_yolov8n/weights/best.pt --device mps
"""

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import get_flops, get_num_params

from common import RESULTS_DIR


def time_model(weights: Path, images: list[Path], device: str, imgsz: int, warmup: int) -> dict:
    model = YOLO(str(weights))
    for f in images[:warmup]:
        model.predict(str(f), imgsz=imgsz, device=device, verbose=False)
    speeds = [model.predict(str(f), imgsz=imgsz, device=device, verbose=False)[0].speed for f in images]
    med = {k: statistics.median(s[k] for s in speeds) for k in ("preprocess", "inference", "postprocess")}
    med["total"] = sum(med.values())
    fused = model.model.fuse()
    return {
        "weights": str(weights),
        "params_M_fused": get_num_params(fused) / 1e6,
        "GFLOPs_fused": get_flops(fused, imgsz),
        "median_ms_per_image": {k: round(v, 2) for k, v in med.items()},
        "images_timed": len(images),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=WEIGHTS")
    ap.add_argument("--device", default="cpu", help="cpu, mps, or a CUDA index such as 0")
    ap.add_argument("--images", type=int, default=200, help="VisDrone val images to time")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    val_dir = Path(SETTINGS["datasets_dir"]) / "VisDrone" / "images" / "val"
    images = sorted(val_dir.glob("*.jpg"))[: args.images]
    if not images:
        raise SystemExit(f"no images in {val_dir}")

    out = {
        "device": args.device,
        "host": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "imgsz": args.imgsz,
        "batch": 1,
        "models": {},
    }
    for spec in args.model:
        name, weights = spec.split("=", 1)
        out["models"][name] = time_model(Path(weights).expanduser(), images, args.device, args.imgsz, args.warmup)
        print(name, out["models"][name]["median_ms_per_image"], f"{out['models'][name]['params_M_fused']:.2f}M")

    dest = RESULTS_DIR / "speed" / f"speed_{args.device}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
