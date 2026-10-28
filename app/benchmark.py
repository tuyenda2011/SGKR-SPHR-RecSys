"""Benchmark tab back end, split as in the report.

Main comparison (report B2/B3): the five original models; paired tests of each proposed model (SGKR, SPHR)
against each graph baseline, Holm over those six, recomputed per density with the report's method
(scripts/compare_val_runs.py: per-user metrics from val_predictions.npy, paired user bootstrap, one-sided p).
Closed-form supplement (report B5/B6): read from the tables scripts/make_closed_form_report.py writes, so the
page always shows the report's own numbers. Validation split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.recommender import KINDS, LABELS, RESULTS, ROOT, run_dir

MAIN = ("sgkr", "sphr", "lightgcn", "xsimgcl", "directau")
PROPOSED, GRAPH = ("sgkr", "sphr"), ("lightgcn", "xsimgcl", "directau")
METRICS = ("NDCG@20", "NDCG@10", "Recall@20", "Recall@10", "MRR@20")
TABLES = ROOT / "reports/tables"


def leaderboard(density, summary_path=None):
    """The five main models at this density, ranked by NDCG@20, with the gap to SGKR in percent."""
    summary = pd.read_csv(summary_path or RESULTS / "summary.csv")
    rows = summary[(summary.density == density) & summary.model.isin(MAIN)]
    rows = rows.groupby("model", sort=False)[[*METRICS, "train_seconds", "inference_ms_per_user"]].mean()
    table = rows.reindex([m for m in MAIN if m in rows.index]).reset_index()
    reference = table.loc[table.model == "sgkr", "NDCG@20"]
    table["vs SGKR"] = (table["NDCG@20"] / reference.iloc[0] - 1) * 100 if len(reference) else np.nan
    table = table.sort_values("NDCG@20", ascending=False).reset_index(drop=True)
    table.insert(0, "Hạng", range(1, len(table) + 1))
    table.insert(2, "Nhóm", table.model.map(KINDS))
    table["Mô hình"] = table.model.map(LABELS)
    return table


def by_density(summary_path=None):
    summary = pd.read_csv(summary_path or RESULTS / "summary.csv")
    summary = summary[summary.model.isin(MAIN)]
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
    """Each proposed model minus each graph baseline (as report B3): bootstrap CI, one-sided p, Holm over six."""
    from scripts.compare_val_runs import holm_adjust, paired_bootstrap
    scores = {model: per_user(model, density, targets) for model in MAIN}
    rows = []
    for proposed in PROPOSED:
        for baseline in GRAPH:
            a, b = scores[baseline], scores[proposed]
            if a is None or b is None:
                continue
            common = a.index.intersection(b.index)
            rows.append({"proposed": proposed, "baseline": baseline,
                         **paired_bootstrap(a.loc[common, metric].to_numpy(float),
                                            b.loc[common, metric].to_numpy(float), repeats=repeats)})
    if not rows:
        return pd.DataFrame()
    for row, p in zip(rows, holm_adjust([row["p_one_sided"] for row in rows])):
        row["p_holm"] = p
        row["verdict"] = (f"{LABELS[row['proposed']]} tốt hơn" if p < .05
                          else "Baseline tốt hơn" if row["ci95"][1] < 0 else "Chưa khác biệt có ý nghĩa")
    return pd.DataFrame(rows)


def closed_form_all():
    """Report table B5 for every density (for the closed-form charts), or None before the report has run."""
    path = TABLES / "B5_closed_form.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def closed_form(density):
    """Report tables B5 (metrics, size, time) and B6 (SGKR minus each closed-form model) at one density."""
    metrics, tests = TABLES / "B5_closed_form.csv", TABLES / "B6_closed_form_significance.csv"
    if not metrics.exists() or not tests.exists():
        return None, None
    b5 = pd.read_csv(metrics, encoding="utf-8-sig")
    b6 = pd.read_csv(tests, encoding="utf-8-sig")
    b5, b6 = b5[np.isclose(b5["Mật độ"], density)], b6[np.isclose(b6["Mật độ"], density)]
    b6 = b6.assign(**{"Kết luận": [
        "SGKR tốt hơn" if p < .05 else "Mô hình này tốt hơn" if high < 0 else "Chưa khác biệt có ý nghĩa"
        for p, high in zip(b6["p Holm"], b6["KTC 95% trên"])]})
    return b5.reset_index(drop=True), b6.reset_index(drop=True)
