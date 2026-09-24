"""Predict VisDrone-DET testset-challenge images and package an official-format submission.

Writes one <image>.txt per image, one detection per line in the VisDrone-DET format
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
with pixel coordinates in the original image, object_category = YOLO class + 1 (VisDrone
ids 1..10) and truncation = occlusion = -1 (not predicted). The .txt files go straight
into submission/<name>.zip (files at the zip root) next to a <name>.json run manifest.
The evaluation protocol caps detections at 500 per image and ranks by score, so a low
confidence threshold keeps the full precision-recall curve. By default images are resized to the
imgsz stored in the checkpoint (the training resolution); --imgsz overrides it.

Ultralytics' VisDrone.yaml does not download testset-challenge, so the images (0.28 GB) are fetched
into <datasets_dir>/VisDrone/ on first use. Submit the zip at the VisDrone challenge server.

Usage:
    python src/predict_challenge.py --weights runs/baseline/weights/best.pt
    python src/predict_challenge.py --weights best.pt --source <datasets_dir>/VisDrone/images/test --name baseline_testdev
    python src/predict_challenge.py --weights best.pt --limit 3 --device cpu   # smoke test
"""

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL, SETTINGS
from ultralytics.utils.downloads import download

from common import REPO_ROOT

VISDRONE = Path(SETTINGS["datasets_dir"]) / "VisDrone"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NUM_CLASSES = 10  # VisDrone-DET object categories 1..10


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def challenge_images() -> Path:
    """testset-challenge image folder, downloading it on first use."""
    for p in (VISDRONE / "images" / "challenge", VISDRONE / "VisDrone2019-DET-test-challenge" / "images"):
        if p.is_dir():
            return p
    download([f"{ASSETS_URL}/VisDrone2019-DET-test-challenge.zip"], dir=VISDRONE)
    return VISDRONE / "VisDrone2019-DET-test-challenge" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None, help="default: testset-challenge images (downloaded if missing)")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the training imgsz stored in the checkpoint")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored by NMS-free heads such as YOLO26)")
    ap.add_argument("--max-det", type=int, default=500, help="VisDrone-DET evaluates at most 500 boxes per image")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--augment", action="store_true", help="test-time augmentation (flip + multi-scale)")
    ap.add_argument(
        "--device", default=None, help="default: cuda > mps > cpu"
    )
    ap.add_argument("--threads", type=int, default=None, help="cap torch CPU threads")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke tests)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "submission")
    ap.add_argument("--name", default=None, help="default: <run name>_<source folder>")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    weights, out = args.weights.expanduser(), args.out.expanduser()
    source = args.source.expanduser() if args.source else challenge_images()
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    if not files:
        raise SystemExit(f"no images in {source}")
    run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    name = args.name or f"{run}_{source.name}"
    device = args.device or pick_device()

    model = YOLO(str(weights))
    if len(model.names) != NUM_CLASSES:
        raise SystemExit(f"{weights} has {len(model.names)} classes, expected the {NUM_CLASSES} VisDrone classes")
    train_imgsz = model.overrides.get("imgsz")
    if args.imgsz and train_imgsz and args.imgsz != train_imgsz:
        print(f"warning: predicting at imgsz {args.imgsz} but the checkpoint was trained at {train_imgsz}")
    imgsz = {"imgsz": args.imgsz} if args.imgsz else {}  # otherwise Ultralytics uses the checkpoint's imgsz
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{name}.zip"
    per_class, n_dets, t0 = Counter(), 0, time.time()
    results = model.predict(
        source=[str(f) for f in files],
        **imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        augment=args.augment,
        device=device,
        stream=True,
        verbose=False,
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            b = r.boxes
            lines = []
            # Ultralytics has already mapped boxes back to the original image and clipped them to its bounds.
            for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                if x2 > x1 and y2 > y1:
                    lines.append(f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.5f},{int(c) + 1},-1,-1")
                    per_class[int(c) + 1] += 1
            n_dets += len(lines)
            zf.writestr(f"{Path(r.path).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))

    manifest = {
        "weights": str(weights),
        "model_names": model.names,
        "source": str(source),
        "images": len(files),
        "detections": n_dets,
        "detections_per_category": dict(sorted(per_class.items())),
        "seconds": round(time.time() - t0, 1),
        "imgsz": model.predictor.args.imgsz,  # what was actually used
        "train_imgsz": train_imgsz,
        **{k: getattr(args, k) for k in ("conf", "iou", "max_det", "augment")},
        "device": device,
        "format": "bbox_left,bbox_top,bbox_width,bbox_height,score,object_category(1-10),truncation(-1),occlusion(-1)",
    }
    (out / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(files)} images, {n_dets} detections ({n_dets / len(files):.0f}/image) -> {zip_path}")


if __name__ == "__main__":
    main()
