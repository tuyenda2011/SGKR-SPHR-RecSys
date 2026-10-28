"""Benchmark tab back end: leaderboard from results/summary.csv and paired tests of SGKR against every model.

The tests reuse the report's method (scripts/compare_val_runs.py): per-user metrics from each run's stored
val_predictions.npy, paired user bootstrap with a one-sided p for SGKR > model, Holm over the comparisons
of one density. Validation split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.recommender import KINDS, LABELS, MODELS, RESULTS, run_dir

REFERENCE = "sgkr"
METRICS = ("NDCG@20", "NDCG@10", "Recall@20", "Recall@10", "MRR@20")


def leaderboard(density, summary_path=None):
    """One row per trained model at this density, ranked by NDCG@20, with the gap to SGKR in percent."""
    summary = pd.read_csv(summary_path or RESULTS / "summary.csv")
    rows = summary[(summary.density == density) & summary.model.isin(MODELS)]
    rows = rows.groupby("model", sort=False)[[*METRICS, "train_seconds", "inference_ms_per_user"]].mean()
    table = rows.reindex([m for m in MODELS if m in rows.index]).reset_index()
    reference = table.loc[table.model == REFERENCE, "NDCG@20"]
    table["vs SGKR"] = (table["NDCG@20"] / reference.iloc[0] - 1) * 100 if len(reference) else np.nan
    table = table.sort_values("NDCG@20", ascending=False).reset_index(drop=True)
    table.insert(0, "Hạng", range(1, len(table) + 1))
    table.insert(2, "Nhóm", table.model.map(KINDS))
    table["Mô hình"] = table.model.map(LABELS).str.replace(" (top-128)", "", regex=False)
    return table


def by_density(summary_path=None):
    summary = pd.read_csv(summary_path or RESULTS / "summary.csv")
    summary = summary[summary.model.isin(MODELS)]
    return summary.groupby(["model", "density"], as_index=False)[[*METRICS, "train_seconds"]].mean()


def per_user(model, density, targets):
    """Per-user NDCG@20, Recall@20 and NDCG@10 of the stored validation predictions, keyed by user."""
    from scripts.compare_val_runs import per_user_metrics
    folder = run_dir(model, density)
    if not (folder / "val_predictions.npy").exists():
        return None
    users = np.load(folder / "val_users.npy")
    predictions = np.load(folder / "val_predictions.npy").astype(np.int64, copy=False)
    values = per_user_metrics(predictions, np.array([targets[int(u)] for u in users], dtype=np.int64))
    return pd.DataFrame(values, index=users)


def significance(density, targets, metric="NDCG@20", repeats=10_000):
    """SGKR minus each other model: paired bootstrap CI, one-sided p (SGKR better), Holm-adjusted."""
    from scripts.compare_val_runs import holm_adjust, paired_bootstrap
    reference = per_user(REFERENCE, density, targets)
    if reference is None:
        return pd.DataFrame()
    rows = []
    for model in MODELS:
        other = per_user(model, density, targets) if model != REFERENCE else None
        if other is None:
            continue
        common = reference.index.intersection(other.index)
        result = paired_bootstrap(other.loc[common, metric].to_numpy(float),
                                  reference.loc[common, metric].to_numpy(float), repeats=repeats)
        rows.append({"model": model, **result})
    if not rows:
        return pd.DataFrame()
    for row, p in zip(rows, holm_adjust([row["p_one_sided"] for row in rows])):
        row["p_holm"] = p
        low, high = row["ci95"]
        row["verdict"] = ("SGKR tốt hơn" if p < .05 else "Mô hình này tốt hơn" if high < 0
                          else "Chưa khác biệt có ý nghĩa")
    return pd.DataFrame(rows)
