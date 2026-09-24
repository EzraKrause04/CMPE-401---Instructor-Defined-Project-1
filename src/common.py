"""Shared helpers: experiment config resolution and output locations."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "experiments.yaml"
META_KEYS = {"base", "part", "description"}

# Ultralytics training output (weights, checkpoints). Point at Google Drive on Colab
# so a disconnected session can resume.
RUNS_DIR = Path(os.environ.get("RUNS_DIR", REPO_ROOT / "runs"))
# Small artifacts that belong in the repo (CSV, plots, metrics JSON).
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_run(name, config=None):
    """Return (train_kwargs, meta) for a run, applying common -> base chain -> run."""
    config = config or load_config()
    runs = config["runs"]
    if name not in runs:
        raise SystemExit(f"Unknown run '{name}'. Defined runs: {', '.join(runs)}")

    chain, seen = [], set()
    cur = name
    while cur is not None:
        if cur in seen:
            raise SystemExit(f"Circular `base:` chain at '{cur}'")
        seen.add(cur)
        chain.append(runs[cur])
        cur = runs[cur].get("base")

    params = dict(config.get("common", {}))
    for entry in reversed(chain):
        params.update({k: v for k, v in entry.items() if k not in META_KEYS})

    meta = {k: runs[name].get(k) for k in META_KEYS}
    meta["name"] = name
    return params, meta


def run_dir(name):
    return RUNS_DIR / name


def results_dir(name):
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
