"""Plot training vs validation loss for Part II and summarize fitting behaviour.

    python src/plot_curves.py baseline
    python src/plot_curves.py baseline p3_imgsz960   # overlay runs

Reads results/<run>/results.csv, writes results/<run>/loss_curves.png and
loss_summary.json (numbers to cite when discussing over/underfitting).
With several runs, also writes results/compare_<runs>.png.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, results_dir


def load(run):
    path = RESULTS_DIR / run / "results.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run src/train.py {run} first.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def loss_components(df):
    """Loss names present in both train/ and val/ columns, skipping all-zero ones (e.g. no DFL in YOLO26)."""
    comps = []
    for col in df.columns:
        if col.startswith("train/") and col.endswith("_loss"):
            comp = col[len("train/"):]
            if f"val/{comp}" in df and df[col].abs().sum() > 0:
                comps.append(comp)
    return comps


def summarize(df, comps):
    train = df[[f"train/{c}" for c in comps]].sum(axis=1)
    val = df[[f"val/{c}" for c in comps]].sum(axis=1)
    i_min = int(val.idxmin())
    return {
        "components": comps,
        "epochs": int(df["epoch"].max()),
        "train_loss_first": round(float(train.iloc[0]), 4),
        "train_loss_final": round(float(train.iloc[-1]), 4),
        "val_loss_first": round(float(val.iloc[0]), 4),
        "val_loss_min": round(float(val.min()), 4),
        "val_loss_min_epoch": int(df["epoch"].iloc[i_min]),
        "val_loss_final": round(float(val.iloc[-1]), 4),
        # >0 means val loss climbed back up after its minimum: a classic overfitting signal.
        "val_rise_after_min_pct": round(100 * float((val.iloc[-1] - val.min()) / val.min()), 2),
        # Final generalization gap; large and growing suggests overfitting, small with high
        # absolute losses suggests underfitting.
        "final_gap_val_minus_train": round(float(val.iloc[-1] - train.iloc[-1]), 4),
        "best_mAP50-95": round(float(df["metrics/mAP50-95(B)"].max()), 4) if "metrics/mAP50-95(B)" in df else None,
    }


def plot_run(run):
    df = load(run)
    comps = loss_components(df)
    n = len(comps) + 2
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
    for ax, comp in zip(axes, comps):
        ax.plot(df["epoch"], df[f"train/{comp}"], label="train")
        ax.plot(df["epoch"], df[f"val/{comp}"], label="val")
        ax.set_title(comp)
    total_ax, map_ax = axes[-2], axes[-1]
    train = df[[f"train/{c}" for c in comps]].sum(axis=1)
    val = df[[f"val/{c}" for c in comps]].sum(axis=1)
    total_ax.plot(df["epoch"], train, label="train")
    total_ax.plot(df["epoch"], val, label="val")
    total_ax.axvline(df["epoch"].iloc[int(val.idxmin())], ls="--", c="gray", lw=1, label="min val")
    total_ax.set_title("total loss")
    for col, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
        if col in df:
            map_ax.plot(df["epoch"], df[col], label=label)
    map_ax.set_title("validation mAP")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(results_dir(run) / "loss_curves.png", dpi=150)
    plt.close(fig)

    summary = summarize(df, comps)
    (results_dir(run) / "loss_summary.json").write_text(json.dumps(summary, indent=2))
    print(run, json.dumps(summary, indent=2))
    return df, comps


def plot_overlay(runs, frames):
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, (df, comps) in zip(runs, frames):
        val = df[[f"val/{c}" for c in comps]].sum(axis=1)
        ax_loss.plot(df["epoch"], val, label=run)
        if "metrics/mAP50-95(B)" in df:
            ax_map.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=run)
    ax_loss.set_title("total validation loss")
    ax_map.set_title("validation mAP50-95")
    for ax in (ax_loss, ax_map):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / f"compare_{'_vs_'.join(runs)}.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    frames = [plot_run(r) for r in args.runs]
    if len(args.runs) > 1:
        plot_overlay(args.runs, frames)


if __name__ == "__main__":
    main()
