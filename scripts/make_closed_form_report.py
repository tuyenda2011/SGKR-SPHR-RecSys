"""Supplementary report: SGKR against the closed-form baselines only (EASE, EASE top-128, GF-CF).

  python scripts/make_closed_form_report.py

Writes next to the main report without touching it. The figures are the main report's own types (same
drawing code, make_combined_figures.bar_figure), restricted to these four models:
  reports/tables/B5_closed_form.csv                   metrics, train time and checkpoint size, all four densities
  reports/tables/B6_closed_form_significance.csv      SGKR minus each model: paired bootstrap, Holm over 3 per density
  reports/figures/H1_closed_form_density_<metric>.png as H1
  reports/figures/H4_closed_form_retention_<metric>.png as H4
  reports/figures/H5_closed_form_user_groups_<metric>.png as H5
  reports/figures/H7a_closed_form_train_time.png      as H7a
  reports/figures/H7c_closed_form_checkpoint_size.png as H7a, checkpoint size instead of train time
  reports/figures/H8_closed_form_ndcg_at_k.png        as H8

EASE top-128 is the pruned checkpoint the demo loads; its validation predictions were not stored by the run, so
they are rebuilt here once with the shared Evaluator and checked against the run's recorded NDCG@20.
Exploratory: this comparison family was defined after the results were seen. Validation split only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch
from scipy import sparse

from scripts.compare_val_runs import holm_adjust, paired_bootstrap, per_user_metrics
from scripts.make_combined_figures import bar_figure, figure_groups, figure_ndcg_at_k
from scripts.make_report import FIGURE_METRICS, MUTED, setup, slug
from scripts.validation_io import load_validation_inputs

SEED = 42
DENSITIES = (0.25, 0.5, 0.75, 1.0)
METRICS = ("NDCG@10", "NDCG@20", "Recall@10", "Recall@20")
RIVALS = ("ease", "ease128", "gfcf")
ARMS = ("sgkr", *RIVALS)
# Same keys as make_report.STYLE; baselines are hatched/dashed by the drawing code (only SGKR is proposed here).
STYLE = {"sgkr": {"label": "SGKR (đề xuất)", "color": "#2a78d6", "marker": "o", "line": "-"},
         "ease": {"label": "EASE (đầy đủ)", "color": "#7a5cc7", "marker": "D", "line": "--"},
         "ease128": {"label": "EASE top-128", "color": "#b39ae6", "marker": "d", "line": ":"},
         "gfcf": {"label": "GF-CF", "color": "#8c6d46", "marker": "s", "line": "--"}}


def run_dir(model, density, results):
    folder = "ease" if model == "ease128" else model
    return results / folder / f"d{round(density * 100)}" / f"seed{SEED}"


def ease_top128_predictions(density, results, full_train, validation, stats):
    """Rebuild (once) and cache the pruned EASE checkpoint's validation top-20, checked against the run."""
    folder = run_dir("ease", density, results)
    cache = folder / "logs/ease/checkpoint_val_predictions.npy"
    expected = json.loads((folder / "logs/ease/result.json").read_text(encoding="utf-8"))["checkpoint_val_metrics"]
    if not cache.exists():
        from scripts.extra.run_item_baselines import ItemScores
        from src.data.quality_report import warm_cohort
        from src.data.sparsity import create_sparse_train_set
        from src.evaluation.evaluator import Evaluator
        from src.evaluation.metrics import compute_topk_metrics
        from src.models.sgkr import SGKR
        train = create_sparse_train_set(full_train, density, SEED)
        cohort, _ = warm_cohort(train, validation)
        x = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(), stats["num_users"],
                                    stats["num_items"]).astype(np.float32)
        weights = sparse.load_npz(folder / "checkpoint.npz")
        evaluator = Evaluator(full_train, cohort, *x.shape, [10, 20], 128, candidate_items=set(train.i_idx.unique()),
                              popularity_df=train, score_fn=ItemScores(x, weights).get_user_rating_scores)
        predictions, _ = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
        rebuilt = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])["NDCG@20"]
        if abs(rebuilt - expected["NDCG@20"]) > 1e-9:
            raise AssertionError(f"EASE top-128 d{density}: rebuilt NDCG@20 {rebuilt} != recorded {expected['NDCG@20']}")
        np.save(folder / "logs/ease/checkpoint_val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))
        np.save(cache, predictions.numpy().astype(np.int64))
    users = np.load(folder / "logs/ease/checkpoint_val_users.npy")
    return users, np.load(cache), expected


def per_user(model, density, results, targets, inputs):
    """(users, predictions, per-user metrics, targets): the layout of make_report.per_user, for H5 and H8."""
    if model == "ease128":
        users, predictions, _ = ease_top128_predictions(density, results, *inputs)
    else:
        folder = run_dir(model, density, results)
        users, predictions = np.load(folder / "val_users.npy"), np.load(folder / "val_predictions.npy")
    users, predictions = users.astype(np.int64), predictions.astype(np.int64)
    truth = np.array([targets[int(u)] for u in users], dtype=np.int64)
    values = dict(per_user_metrics(predictions, truth))
    values["Recall@10"] = (predictions[:, :10] == truth[:, None]).any(axis=1).astype(float)
    return users, predictions, values, truth


def checkpoint_mb(model, density, results, num_items):
    if model == "ease":
        return num_items * num_items * 4 / 2**20  # dense float32 item-item matrix, not stored
    return (run_dir(model, density, results) / "checkpoint.npz").stat().st_size / 2**20


def table_metrics(results, num_items):
    rows = []
    for density in DENSITIES:
        for model in ARMS:
            folder = run_dir(model, density, results)
            run = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            metrics = run["checkpoint_val_metrics"] if model == "ease128" else run["val_metrics"]
            rows.append({"Mô hình": STYLE[model]["label"], "model": model, "Mật độ": density,
                         **{m: metrics[m] for m in METRICS}, "train_seconds": run["train_seconds"],
                         "Checkpoint (MB)": checkpoint_mb(model, density, results, num_items)})
    return pd.DataFrame(rows)


def table_tests(scores, repeats=10_000):
    rows = []
    for density in DENSITIES:
        users, _, values, _ = scores[("sgkr", density, SEED)]
        reference = pd.Series(values["NDCG@20"], index=users)
        block = []
        for model in RIVALS:
            users, _, values, _ = scores[(model, density, SEED)]
            other = pd.Series(values["NDCG@20"], index=users)
            common = reference.index.intersection(other.index)
            test = paired_bootstrap(other.loc[common].to_numpy(float), reference.loc[common].to_numpy(float),
                                    repeats=repeats)
            block.append({"model": model, "Mật độ": density, **test})
        for row, p in zip(block, holm_adjust([row["p_one_sided"] for row in block])):
            row["p_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def figures(table, scores, full_train, folder, notes):
    """H1, H4, H5, H7 and H8 for the four closed-form models (same code and layout as the main report)."""
    values = table.set_index(["model", "Mật độ"])

    def lookup(column):
        return lambda model, density: float(values.at[(model, density), column])
    kwargs = {"order": ARMS, "style": STYLE, "ncol": 4}
    for metric in FIGURE_METRICS:
        bar_figure(lookup(metric), metric, f"H1_closed_form_density_{slug(metric)}", folder, notes, **kwargs)
        bar_figure(lambda m, d, metric=metric: 100 * values.at[(m, d), metric] / values.at[(m, 1.0), metric],
                   f"{metric} so với mật độ 1.00 (%)", f"H4_closed_form_retention_{slug(metric)}", folder, notes,
                   fmt="{:.1f}", extra=lambda ax: ax.axhline(100, color=MUTED, linestyle="--", linewidth=.8, zorder=0),
                   **kwargs)
        figure_groups(scores, full_train, metric, folder, notes, name="H5_closed_form_user_groups", **kwargs)
    figure_ndcg_at_k(scores, folder, notes, name="H8_closed_form_ndcg_at_k", **kwargs)
    bar_figure(lookup("train_seconds"), "Thời gian huấn luyện (giây, thang log)\nTất cả chạy trên CPU; "
               "EASE top-128 = EASE + cắt", "H7a_closed_form_train_time", folder, notes, fmt="{:,.0f}", log=True,
               **kwargs)
    bar_figure(lookup("Checkpoint (MB)"), "Dung lượng checkpoint (MB, thang log)\nEASE đầy đủ: ma trận float32, "
               "không lưu", "H7c_closed_form_checkpoint_size", folder, notes, fmt="{:,.1f}", log=True, **kwargs)


def main():
    results, reports = ROOT / "results", ROOT / "reports"
    setup()
    _, full_train, validation, stats = load_validation_inputs(None, ROOT)
    targets = dict(zip(validation.u_idx.astype(int), validation.i_idx.astype(int)))
    inputs = (results, full_train, validation, stats)
    for density in DENSITIES:
        ease_top128_predictions(density, *inputs)
    table = table_metrics(results, stats["num_items"])
    table.drop(columns="model").assign(**{"Train (phút)": (table.train_seconds / 60).round(1)}).drop(
        columns="train_seconds").round({**{m: 5 for m in METRICS}, "Checkpoint (MB)": 1}).to_csv(
        reports / "tables/B5_closed_form.csv", index=False, encoding="utf-8-sig")
    scores = {(model, density, SEED): per_user(model, density, results, targets, (full_train, validation, stats))
              for model in ARMS for density in DENSITIES}
    tests = table_tests(scores)
    pd.DataFrame({"So sánh": "SGKR - " + tests.model.map(lambda m: STYLE[m]["label"]), "Mật độ": tests["Mật độ"],
                  "Delta NDCG@20": tests.delta.round(5), "KTC 95% dưới": [round(c[0], 5) for c in tests.ci95],
                  "KTC 95% trên": [round(c[1], 5) for c in tests.ci95], "p một phía": tests.p_one_sided.round(5),
                  "p Holm": tests.p_holm.round(5), "Số user": tests.n_users}
                 ).to_csv(reports / "tables/B6_closed_form_significance.csv", index=False, encoding="utf-8-sig")
    notes = []
    figures(table, scores, full_train, reports / "figures", notes)
    print("\n".join(notes))


if __name__ == "__main__":
    main()
