"""Build a markdown comparison table from results/<run>/metrics_<split>.json files.

    python src/compare.py baseline p3_imgsz960
    python src/compare.py baseline p5_yolov8n --split test --out report/tables/part5.md

The first run is the reference; the delta column is mAP50-95 relative to it.
"""

import argparse
import json
from pathlib import Path

from common import RESULTS_DIR, resolve_run, load_config

COLUMNS = [
    ("Run", lambda m: m["run"]),
    ("Model", lambda m: m["model"].removesuffix(".pt")),
    ("imgsz", lambda m: m["imgsz"]),
    ("P", lambda m: f"{m['precision']:.3f}"),
    ("R", lambda m: f"{m['recall']:.3f}"),
    ("mAP50", lambda m: f"{m['mAP50']:.3f}"),
    ("mAP50-95", lambda m: f"{m['mAP50-95']:.3f}"),
    ("Params (M)", lambda m: m["params_M"]),
    ("GFLOPs", lambda m: m["GFLOPs"]),
    ("Weights (MB)", lambda m: m["weights_MB"]),
    ("Inference (ms/img)", lambda m: m["speed_ms_per_img"].get("inference", "")),
    ("Train time (h)", lambda m: m.get("train_time_hours", "")),
    ("Epochs (best)", lambda m: f"{m.get('epochs_completed', '')} ({m.get('best_epoch', '')})"),
]


def changed_settings(ref, run, config):
    """Settings that differ from the reference run: what the experiment actually changed."""
    a, _ = resolve_run(ref, config)
    b, _ = resolve_run(run, config)
    diffs = [f"{k}={b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return ", ".join(diffs) or "(reference)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", help="write the table to this markdown file")
    args = ap.parse_args()

    config = load_config()
    metrics = []
    for run in args.runs:
        path = RESULTS_DIR / run / f"metrics_{args.split}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path}; run src/evaluate.py {run} --split {args.split}")
        metrics.append(json.loads(path.read_text()))

    ref = metrics[0]
    header = [c for c, _ in COLUMNS] + ["Δ mAP50-95", "Changed vs reference"]
    lines = [
        f"_VisDrone {args.split} split; reference run: `{ref['run']}`_",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for m in metrics:
        delta = m["mAP50-95"] - ref["mAP50-95"]
        row = [str(fn(m)) for _, fn in COLUMNS]
        row += [f"{delta:+.3f}", changed_settings(ref["run"], m["run"], config)]
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines) + "\n"
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
