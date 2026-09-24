"""Dataset statistics for VisDrone-DET in YOLO format (the numbers Part II cites).

Reads the YOLO labels written by Ultralytics' VisDrone.yaml converter plus each image's size (PIL header
read only) and reports, per split: images, boxes, boxes per class and imbalance ratio,
objects-per-image distribution, and box size as sqrt(area) in native pixels and after
Ultralytics' resize to each network input size (scale = imgsz / max(w, h)). The
fractions below 8 / 16 / 32 px tie object size to the detector: 8 px is one cell of the
stride-8 P3 map (the finest head), and 32^2 px is COCO's "small" area threshold.

Usage:
    python src/dataset_stats.py                      # -> results/dataset/
    python src/dataset_stats.py --root /content/datasets/VisDrone
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_yaml

from common import RESULTS_DIR

NAMES = yaml.safe_load(Path(check_yaml("VisDrone.yaml")).read_text())["names"]
THRESHOLDS = (8, 16, 32)
STRIDES = {8: "P3", 16: "P4", 32: "P5"}  # COCO's 32 px "small" cut applies at native resolution, not on this axis
# Fixed categorical order (colour follows the entity): train, val, test / 640, 960.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:  # lazy: only the header is decoded
        return im.size


def load_split(root: Path, split: str, workers: int) -> dict:
    """Per-image box counts and per-box (cls, w_px, h_px, long side of its image)."""
    labels = sorted((root / "labels" / split).glob("*.txt"))
    images = [root / "images" / split / f"{p.stem}.jpg" for p in labels]
    with ThreadPoolExecutor(workers) as ex:
        sizes = list(ex.map(image_size, images))

    counts, rows = [], []
    for lbl, (w, h) in zip(labels, sizes):
        a = np.loadtxt(lbl, ndmin=2) if lbl.stat().st_size else np.zeros((0, 5))
        counts.append(len(a))
        if len(a):
            rows.append(np.column_stack([a[:, 0], a[:, 3] * w, a[:, 4] * h, np.full(len(a), max(w, h))]))
    boxes = np.concatenate(rows) if rows else np.zeros((0, 4))
    return {
        "counts": np.array(counts),
        "cls": boxes[:, 0].astype(int),
        "sqrt_area": np.sqrt(boxes[:, 1] * boxes[:, 2]),
        "long_side": boxes[:, 3],
        "image_sizes": [f"{w}x{h}" for w, h in sizes],
    }


def pct(x: np.ndarray) -> dict:
    if not len(x):
        return {}
    p = np.percentile(x, [10, 25, 50, 75, 90])
    return {"mean": float(x.mean()), **{f"p{q}": float(v) for q, v in zip((10, 25, 50, 75, 90), p)}}


def summarize(d: dict, imgsz: list[int]) -> dict:
    counts, cls = d["counts"], d["cls"]
    per_class = {NAMES[i]: int((cls == i).sum()) for i in NAMES}
    nonzero = [v for v in per_class.values() if v]
    scales = {"native": np.ones_like(d["long_side"]), **{str(s): s / d["long_side"] for s in imgsz}}
    sizes = {k: d["sqrt_area"] * v for k, v in scales.items()}
    uniq, n = np.unique(d["image_sizes"], return_counts=True)
    return {
        "images": int(len(counts)),
        "boxes": int(counts.sum()),
        "empty_images": int((counts == 0).sum()),
        "image_sizes_top5": {u: int(c) for u, c in sorted(zip(uniq, n), key=lambda t: -t[1])[:5]},
        "boxes_per_class": per_class,
        "class_share": {k: v / max(len(cls), 1) for k, v in per_class.items()},
        "imbalance_ratio_max_over_min": max(nonzero) / min(nonzero) if nonzero else None,
        "objects_per_image": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "std": float(counts.std()),
            "p90": float(np.percentile(counts, 90)),
            "max": int(counts.max()),
        },
        "sqrt_area_px": {k: pct(v) for k, v in sizes.items()},
        "frac_sqrt_area_below": {k: {str(t): float((v < t).mean()) for t in THRESHOLDS} for k, v in sizes.items()},
        "per_class_median_sqrt_area_px": {
            k: {NAMES[i]: float(np.median(v[cls == i])) for i in NAMES if (cls == i).any()} for k, v in sizes.items()
        },
        "per_class_frac_below_8px": {
            k: {NAMES[i]: float((v[cls == i] < 8).mean()) for i in NAMES if (cls == i).any()}
            for k, v in sizes.items()
            if k != "native"
        },
    }


def plot_classes(stats: dict, out: Path):
    splits = list(stats)
    order = sorted(NAMES.values(), key=lambda k: -stats[splits[0]]["boxes_per_class"][k])
    y = np.arange(len(order))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    counts = [stats[splits[0]]["boxes_per_class"][k] for k in order]
    a.barh(y, counts, color=C[0], height=0.7)
    for yi, c in zip(y, counts):
        a.text(c, yi, f" {c:,}", va="center", fontsize=7, color=INK2)
    a.set(yticks=y, yticklabels=order, xlabel="boxes", title=f"{splits[0]} boxes per class")
    a.invert_yaxis()
    a.set_xlim(0, max(counts) * 1.22)
    a.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
    h = 0.8 / len(splits)
    for j, s in enumerate(splits):
        share = [100 * stats[s]["class_share"][k] for k in order]
        b.barh(y + (j - (len(splits) - 1) / 2) * h, share, height=h * 0.9, color=C[j], label=s)
    b.set(xlabel="share of split's boxes (%)", title="class share per split")
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png")
    plt.close(fig)


def plot_objects(data: dict, stats: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cap = np.percentile(np.concatenate([d["counts"] for d in data.values()]), 99)
    bins = np.linspace(0, cap, 41)
    for j, (s, d) in enumerate(data.items()):
        o = stats[s]["objects_per_image"]
        label = f"{s}: median {o['median']:.0f}, mean {o['mean']:.1f}, max {o['max']}"
        ax.hist(np.clip(d["counts"], 0, cap), bins=bins, density=True, histtype="step", lw=1.8, color=C[j], label=label)
        ax.axvline(o["median"], color=C[j], lw=1, ls=":")
    ax.set(xlabel=f"labelled objects per image (clipped at p99 = {cap:.0f})", ylabel="density")
    ax.set_title("objects per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "objects_per_image.png")
    plt.close(fig)


def plot_sizes(data: dict, stats: dict, imgsz: list[int], out: Path):
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.6), sharey=True, squeeze=False)
    bins = np.logspace(0, np.log10(1024), 61)
    for ax, (s, d) in zip(axes[0], data.items()):
        for j, sz in enumerate(imgsz):
            v = d["sqrt_area"] * sz / d["long_side"]
            below = 100 * stats[s]["frac_sqrt_area_below"][str(sz)]["8"]
            ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), histtype="step", lw=1.8, color=C[j],
                    label=f"imgsz {sz}: {below:.1f}% < 8 px")
        for t, name in STRIDES.items():
            ax.axvline(t, color=INK2, lw=0.9, ls="--")
            ax.text(t * 1.06, 0.02, f"{t} ({name})", transform=ax.get_xaxis_transform(), rotation=90,
                    va="bottom", fontsize=7, color=INK2)
        ax.set(xscale="log", xlabel="sqrt(box area) at network input (px)", title=f"{s} ({len(d['cls']):,} boxes)")
        ax.legend(loc="upper right", fontsize=7)
    axes[0][0].set_ylabel("share of boxes per bin (%)")
    fig.tight_layout()
    fig.savefig(out / "box_size_vs_stride.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(SETTINGS["datasets_dir"]) / "VisDrone")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "dataset")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 960])
    ap.add_argument("--workers", type=int, default=8, help="threads for image-header reads")
    args = ap.parse_args()
    root, out = args.root.expanduser(), args.out.expanduser()

    data = {s: load_split(root, s, args.workers) for s in args.splits if (root / "labels" / s).is_dir()}
    if not data:
        raise SystemExit(f"no label folders under {root / 'labels'}")
    out.mkdir(parents=True, exist_ok=True)
    stats = {s: summarize(d, args.imgsz) for s, d in data.items()}
    report = {
        "root": str(root),
        "imgsz": args.imgsz,
        "names": NAMES,
        "splits": stats,
        "notes": [
            "Box size is sqrt(w*h) of the YOLO label, in native pixels and after scaling by imgsz / max(w, h), "
            "the long-side resize Ultralytics applies before letterboxing (val) or mosaic (train). Train-time "
            "scale augmentation (scale=0.5 -> x0.5..x1.5) spreads this further.",
            "8 px = one stride-8 cell of the P3 head (the finest feature map in YOLO26 and YOLOv8); 16 / 32 px = P4 / P5 strides; "
            "32 px also equals COCO's 'small' area threshold (32^2) when applied at native resolution.",
            "Labels exclude VisDrone ignored regions (score 0), as in Ultralytics' VisDrone.yaml converter.",
        ],
    }
    (out / "stats.json").write_text(json.dumps(report, indent=2))

    plt.rcParams.update(STYLE)
    plot_classes(stats, out)
    plot_objects(data, stats, out)
    plot_sizes(data, stats, args.imgsz, out)

    for s, st in stats.items():
        below = {k: round(100 * v["8"], 1) for k, v in st["frac_sqrt_area_below"].items()}
        print(f"{s:6s} images={st['images']:5d} boxes={st['boxes']:6d} "
              f"median_obj/img={st['objects_per_image']['median']:.0f} "
              f"imbalance={st['imbalance_ratio_max_over_min']:.1f} %boxes<8px={below}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
