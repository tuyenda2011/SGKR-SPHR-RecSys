"""Collect every results/<model>/d<density>/seed<seed>/metrics.json into one table.

  python scripts/summarize_results.py            # writes results/summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.train_model import MODELS

METRICS = ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20")


def collect(results=ROOT / "results"):
    rows = []
    for model in MODELS:
        for path in sorted((Path(results) / model).glob("d*/seed*/metrics.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"model": run["model"], "density": run["density"], "seed": run["seed"], "split": run["split"],
                         **{metric: run["val_metrics"].get(metric) for metric in METRICS},
                         "best_epoch": run.get("best_epoch"), "epochs_run": run.get("epochs_run"),
                         "max_epochs": run.get("max_epochs"), "train_seconds": run.get("train_seconds"),
                         "inference_ms_per_user": run.get("inference_ms_per_user"),
                         "source": run.get("source", "train_model"), "path": path.parent.as_posix()})
    columns = ["model", "density", "seed", "split", *METRICS, "best_epoch", "epochs_run", "max_epochs",
               "train_seconds", "inference_ms_per_user", "source", "path"]
    return pd.DataFrame(rows, columns=columns).sort_values(["model", "density", "seed"], ascending=[True, False, True])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <results>/summary.csv")
    args = parser.parse_args(argv)
    frame = collect(args.results)
    output = args.output or args.results / "summary.csv"
    frame.to_csv(output, index=False)
    print(frame[["model", "density", "seed", "NDCG@20", "Recall@20", "best_epoch"]].to_string(index=False))
    print(f"{len(frame)} runs -> {output}")
    return frame


if __name__ == "__main__":
    main()
