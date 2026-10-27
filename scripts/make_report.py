"""Figures and CSV tables, rebuilt from results/<model>/d*/seed*/.

  python scripts/make_report.py                  # -> reports/figures/*.png, reports/tables/*.csv

Figures (scripts/make_combined_figures.py) put all four densities in one picture; runs that do not exist
yet are marked "chưa chạy".
Reads only validation data (train + val), never the test split.
reports/figures is regenerated from scratch on each call.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compare_val_runs import holm_adjust, paired_bootstrap, per_user_metrics
from scripts.summarize_results import collect
from scripts.train_model import DENSITIES, MODELS, normalize_history
from scripts.validation_io import load_validation_inputs

# Fixed identity per model (reference categorical palette, slots 1-5 in validated order); shapes and
# solid/dashed (hatched) styles are the secondary encoding. Proposed models solid, baselines dashed.
STYLE = {
    "sgkr": {"label": "SGKR (đề xuất)", "short": "SGKR", "color": "#2a78d6", "marker": "o", "line": "-"},
    "sphr": {"label": "SPHR (đề xuất)", "short": "SPHR", "color": "#eb6834", "marker": "s", "line": "-"},
    "lightgcn": {"label": "LightGCN", "short": "LightGCN", "color": "#1baf7a", "marker": "^", "line": "--"},
    "xsimgcl": {"label": "XSimGCL", "short": "XSimGCL", "color": "#eda100", "marker": "D", "line": "--"},
    "directau": {"label": "DirectAU", "short": "DirectAU", "color": "#e87ba4", "marker": "v", "line": "--"},
}
ORDER = tuple(STYLE)
PROPOSED, BASELINES = ("sgkr", "sphr"), ("lightgcn", "xsimgcl", "directau")
FIGURE_METRICS = ("NDCG@20", "NDCG@10", "Recall@20", "Recall@10")
TABLE_METRICS = ("NDCG@10", "NDCG@20", "Recall@10", "Recall@20")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLIT = "validation"
SIZE = (6.4, 4.0)  # every figure uses the same canvas (16:10)


def slug(metric):
    return metric.lower().replace("@", "")


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def setup():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                         "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
                         "axes.labelcolor": INK, "text.color": INK})


def save(fig, folder, name, notes):
    fig.savefig(folder / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)
    notes.append(f"{name}: written")


def per_user(runs, targets):
    """Per-user metrics, predictions and targets for each run, keyed by (model, density, seed)."""
    result = {}
    for row in runs.itertuples():
        folder = Path(row.path)
        users = np.load(folder / "val_users.npy").astype(np.int64)
        predictions = np.load(folder / "val_predictions.npy").astype(np.int64)
        truth = np.array([targets[int(user)] for user in users], dtype=np.int64)
        values = dict(per_user_metrics(predictions, truth))
        values["Recall@10"] = (predictions[:, :10] == truth[:, None]).any(axis=1).astype(float)
        result[(row.model, row.density, row.seed)] = (users, predictions, values, truth)
    return result


def model_average(scores, model, density, metric):
    """Per-user metric averaged over the available seeds of one model, on the common user set."""
    runs = [value for (name, dens, _), value in scores.items() if name == model and dens == density]
    if not runs:
        return None, None
    common = runs[0][0]
    for users, *_ in runs[1:]:
        common = np.intersect1d(common, users)
    stacked = [pd.Series(values[metric], index=users).loc[common].to_numpy() for users, _, values, _ in runs]
    return common, np.mean(stacked, axis=0)


def read_history(path):
    file = Path(path) / "history.csv"
    return normalize_history(pd.read_csv(file)) if file.exists() else None


def table_dataset(full_train, validation, stats, folder):
    degree = full_train.groupby("u_idx").size()
    rows = [("Số user", stats["num_users"]), ("Số item", stats["num_items"]),
            ("Tương tác train", len(full_train)), ("Tương tác validation", len(validation)),
            ("Độ thưa train (%)", round(100 * (1 - len(full_train) / (stats["num_users"] * stats["num_items"])), 4)),
            ("Độ dài lịch sử user trung bình", round(degree.mean(), 2)),
            ("Độ dài lịch sử user trung vị", int(degree.median()))]
    pd.DataFrame(rows, columns=["Thống kê", "Giá trị"]).to_csv(folder / "B1_dataset.csv", index=False,
                                                                encoding="utf-8-sig")


def table_metrics(summary, folder, density):
    """All ranking metrics at one density: mean (and std when several seeds) per model."""
    rows = []
    for model in ORDER:
        values = summary[(summary.model == model) & (summary.density == density)]
        if values.empty:
            continue
        row = {"Mô hình": STYLE[model]["label"], "Số seed": len(values)}
        for metric in TABLE_METRICS:
            row[metric] = round(values[metric].mean(), 5)
            if len(values) > 1:
                row[f"{metric} std"] = round(values[metric].std(), 5)
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / f"B2a_metrics_d{round(density * 100)}.csv", index=False, encoding="utf-8-sig")


def table_density(summary, folder):
    """One row per model, one column per density and metric (mean over seeds)."""
    rows = []
    for model in ORDER:
        row = {"Mô hình": STYLE[model]["label"]}
        for density in sorted(DENSITIES, reverse=True):
            for metric in ("NDCG@20", "Recall@20"):
                values = summary[(summary.model == model) & (summary.density == density)][metric]
                row[f"{metric} d={density:.2f}"] = round(values.mean(), 5) if len(values) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "B2b_by_density.csv", index=False, encoding="utf-8-sig")


def table_training(summary, folder, density):
    """Best versus last epoch (no early stopping), epochs run and seconds per epoch, from history.csv."""
    rows = []
    for model in ORDER:
        for run in summary[(summary.model == model) & (summary.density == density)].itertuples():
            history = read_history(run.path)
            row = {"Mô hình": STYLE[model]["label"], "Seed": run.seed, "NDCG@20 tốt nhất": round(summary.at[run.Index, "NDCG@20"], 5)}
            if history is not None and "NDCG@20" in history:
                best = history["NDCG@20"].idxmax()
                row.update({"Epoch tốt nhất": int(history["epoch"][best]), "Số epoch đã chạy": int(history["epoch"].iloc[-1]),
                            "NDCG@20 epoch cuối": round(history["NDCG@20"].iloc[-1], 5),
                            "Giây/epoch": round(history["seconds"].mean(), 1) if "seconds" in history else None})
            rows.append(row)
    if rows:
        table = pd.DataFrame(rows)
        for column in ("Epoch tốt nhất", "Số epoch đã chạy"):
            if column in table:
                table[column] = table[column].astype("Int64")  # SGKR (closed form) leaves these empty
        table.to_csv(folder / f"B4_training_d{round(density * 100)}.csv", index=False, encoding="utf-8-sig")


def table_tests(scores, folder, metric, name, density=1.0, repeats=10_000):
    """Paired user bootstrap, one-sided (proposed > baseline), Holm over the six comparisons."""
    rows = []
    for proposed in PROPOSED:
        for baseline in BASELINES:
            users_b, values_b = model_average(scores, proposed, density, metric)
            users_a, values_a = model_average(scores, baseline, density, metric)
            if values_a is None or values_b is None:
                continue
            common = np.intersect1d(users_a, users_b)
            a = pd.Series(values_a, index=users_a).loc[common].to_numpy()
            b = pd.Series(values_b, index=users_b).loc[common].to_numpy()
            rows.append({"proposed": proposed, "baseline": baseline, **paired_bootstrap(a, b, repeats=repeats)})
    if not rows:
        return
    adjusted = holm_adjust([row["p_one_sided"] for row in rows])
    pd.DataFrame([{"So sánh": f"{STYLE[r['proposed']]['short']} - {STYLE[r['baseline']]['short']}",
                   "Mật độ": density, f"Delta {metric}": round(r["delta"], 5),
                   "KTC 95% dưới": round(r["ci95"][0], 5), "KTC 95% trên": round(r["ci95"][1], 5),
                   "p một phía": round(r["p_one_sided"], 5), "p Holm": round(p, 5), "Số user": r["n_users"]}
                  for r, p in zip(rows, adjusted)]).to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args(argv)
    setup()
    figures, tables = args.output / "figures", args.output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for stale in [*figures.glob("*.pdf"), *figures.glob("*.png"), *tables.glob("*.tex"),
                  *tables.glob("*.csv")]:
        stale.unlink()
    summary = collect(args.results)
    if summary.empty:
        raise SystemExit("No runs under results/<model>/d*/seed*/metrics.json")
    _, full_train, validation, stats = load_validation_inputs(None, ROOT)
    targets = dict(zip(validation.u_idx.to_numpy().tolist(), validation.i_idx.to_numpy().tolist()))
    scores = per_user(summary, targets)
    from scripts.make_combined_figures import draw_all  # imports this module's styles, so not at the top
    notes = []
    present = sorted(summary.density.unique(), reverse=True)  # per-density tables for every density with runs
    draw_all(summary, scores, full_train, stats["num_items"], figures, notes)
    table_dataset(full_train, validation, stats, tables)
    for density in present:
        table_metrics(summary, tables, density)
        table_training(summary, tables, density)
    table_density(summary, tables)
    table_tests(scores, tables, "NDCG@20", "B3_significance_ndcg20", repeats=args.bootstrap)
    table_tests(scores, tables, "NDCG@10", "B3b_significance_ndcg10", repeats=args.bootstrap)
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "split": SPLIT,
                "runs": summary[["model", "density", "seed", "path", "source"]].to_dict("records"), "notes": notes}
    (args.output / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    print("\n".join(note for note in notes if "written" not in note))
    print(f"{sum('written' in note for note in notes)} figures -> {figures}\nTables -> {tables}")
    have = set(zip(summary.model, summary.density))
    missing = [(m, d) for d in DENSITIES for m in MODELS if (m, d) not in have]
    if missing:
        print(f"\nThiếu {len(missing)} run (model × density); chạy rồi gọi lại make_report.py:")
        for model, density in missing:
            print(f"  python scripts\\train_model.py --model {model} --density {density} --seed 42 --epochs 200")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
