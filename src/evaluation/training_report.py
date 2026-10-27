"""Offline, provenance-preserving reports for a single pinned training suite."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bundle import sha256_file
from src.utils.checkpoints import get_model_output_dir

LABELS = {"lightgcn": "LightGCN", "xsimgcl": "XSimGCL", "directau": "DirectAU", "recflow": "RecFlow"}
COLORS = {"lightgcn": "#0072B2", "xsimgcl": "#D55E00", "directau": "#009E73", "recflow": "#CC79A7"}
MARKERS = {"lightgcn": "o", "xsimgcl": "s", "directau": "^", "recflow": "D"}
LABELS["scsr"] = "SCSR"
COLORS["scsr"] = "#6F4E7C"
MARKERS["scsr"] = "P"


def load_suite(root):
    from scripts.train_suite import read_json, result_path, validate_result
    root = Path(root)
    plan = read_json(root / "suite_plan.json")
    runs = []
    for job in plan["jobs"]:
        path = result_path(root, job)
        if job["status"] == "succeeded":
            result = validate_result(plan, job, path)
            runs.append((job, result, path))
    return plan, runs


def flatten_runs(runs):
    rows = []
    for job, result, _ in runs:
        row = {"model": job["model"], "density": job["density"], "seed": job["seed"],
               "architecture": result.get("architecture", job["model"]),
               "experiment_fingerprint": result["experiment_fingerprint"], "experiment_family": result["experiment_family"],
               "monitor": result["monitor"], "best_epoch": result["best_epoch"], "total_epochs": result["total_epochs"],
               "train_seconds": result["total_train_time"], "validation_only": result.get("validation_only", False)}
        for split in ("val", "test"):
            for metric, value in result.get(f"{split}_metrics", {}).items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    row[f"{split}/{metric}"] = value
        for metric, value in result.get("representation_metrics", {}).items():
            if isinstance(value, (float, int)):
                row[f"representation/{metric}"] = value
        for metric, value in result.get("svd_metrics", {}).items():
            if isinstance(value, (float, int)):
                row[f"representation/{metric}"] = value
        for subgroup, metrics in result.get("subgroup_metrics", {}).items():
            for metric in ("Recall@20", "NDCG@20", "Recall@10", "NDCG@10"):
                if metric in metrics:
                    row[f"subgroup/{subgroup}/{metric}"] = metrics[metric]
        for source, name in (("inference_latency_ms_per_user", "latency_ms_per_user"), ("avg_epoch_time", "epoch_seconds")):
            if source in result:
                row[name] = result[source]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.replace([np.inf, -np.inf], np.nan)


def aggregate(frame):
    if frame.empty:
        return pd.DataFrame()
    metrics = [c for c in frame if "/" in c or c in {"train_seconds", "epoch_seconds", "latency_ms_per_user"}]
    rows = []
    for (model, density), group in frame.groupby(["model", "density"], sort=False):
        if group["experiment_family"].nunique() != 1 or group["monitor"].nunique() != 1:
            raise ValueError(f"Mixed experiment families/monitors: {model}/{density}")
        row = {"model": model, "density": density, "runs": len(group), "seeds": ",".join(map(str, sorted(group["seed"]))),
               "monitor": group["monitor"].iloc[0]}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}/n"] = len(values)
            row[f"{metric}/mean"] = values.mean()
            row[f"{metric}/sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _style():
    return {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
            "axes.labelsize": 10, "axes.titleweight": "bold", "axes.labelcolor": "#243247",
            "text.color": "#17283B", "axes.edgecolor": "#CBD5E1", "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": True, "grid.color": "#E2E8F0", "grid.alpha": 0.7,
            "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"}


def render_figures(root, plan, runs, frame, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    directory = root / "figures"
    directory.mkdir(exist_ok=True)

    def save(fig, name, caption, sources):
        # Fixed dimensions, opaque background; captions also appear in HTML.
        fig.text(0.02, 0.02, "Graph recommendation  |  " + ("Validation iteration" if plan["validation_only"] else "Benchmark evaluation"),
                 fontsize=8, color="#64748B")
        fig.subplots_adjust(bottom=0.17, top=0.86, wspace=0.32, hspace=0.42)
        exports = []
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=240, facecolor="white")
            exports.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
        figures.append({"title": name.replace("_", " ").title(), "caption": caption, "files": exports, "sources": sources,
                        "width_inches": float(fig.get_figwidth()), "height_inches": float(fig.get_figheight()), "raster_dpi": 240})
        plt.close(fig)

    with plt.rc_context(_style()):
        for job in plan["jobs"]:
            history = Path(get_model_output_dir("history", job["model"], str(root))) / f"{job['id']}_history.csv"
            if not history.is_file():
                continue
            try:
                data = pd.read_csv(history)
            except pd.errors.EmptyDataError:
                continue
            if data.empty or "epoch" not in data:
                continue
            data = data.sort_values("epoch").replace([np.inf, -np.inf], np.nan)
            if data["epoch"].duplicated().any():
                raise ValueError(f"Duplicate history epochs: {history}")
            # Reindex to expose missing epochs as gaps rather than joining them.
            data = data.set_index("epoch").reindex(range(1, int(data["epoch"].max()) + 1)).rename_axis("epoch").reset_index()
            monitor = data["monitor"].dropna().iloc[-1] if "monitor" in data and data["monitor"].notna().any() else "NDCG@20"
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
            fig.suptitle(f"{LABELS[job['model']]}  ·  {job['density']:.0%} density  ·  seed {job['seed']}", fontsize=15, fontweight="bold")
            loss_column = "loss_total" if "loss_total" in data else "train_loss"
            if loss_column in data:
                axes[0].plot(data["epoch"], data[loss_column], color=COLORS[job["model"]], marker=MARKERS[job["model"]], markersize=4)
            axes[0].set(title="Recorded training loss", xlabel="Epoch", ylabel="Loss (model-specific units)")
            if "monitor_value" in data:
                axes[1].plot(data["epoch"], data["monitor_value"], color=COLORS[job["model"]], marker=MARKERS[job["model"]], markersize=4)
                valid = data["monitor_value"].dropna()
                if len(valid):
                    best = valid.idxmax()
                    axes[1].scatter(data.loc[best, "epoch"], data.loc[best, "monitor_value"], s=105, facecolors="none", edgecolors="#17283B", linewidth=1.5, label="Best validation")
                    axes[1].legend(frameon=False, fontsize=9)
            axes[1].set(title="Validation checkpoint selection", xlabel="Epoch", ylabel=monitor)
            # Leave space for the best-checkpoint ring after extending the axis to zero.
            upper = axes[1].get_ylim()[1]
            axes[1].set_ylim(0, upper * 1.05 if upper > 0 else 1)
            for axis in axes:
                axis.set_xlim(0.5, data["epoch"].max() + 0.5)
                axis.xaxis.get_major_locator().set_params(integer=True)
                if data["epoch"].max() <= 10:
                    axis.set_xticks(data["epoch"])
            save(fig, f"learning_{job['id']}",
                 f"Unsmoothed per-epoch training loss and validation {monitor}; ring selects maximum validation. Loss excludes optimizer-applied weight decay; units differ between models. Job status: {job['status']}.",
                 [history.relative_to(root).as_posix()])
            components = [c for c in data if (c == "loss_bpr_raw" or c.startswith("loss_") and c.endswith("_weighted"))
                          and data[c].notna().any() and data[c].abs().max() > 0]
            diagnostics = [c for c in ("mean_gradient_norm", "gradient_clip_fraction", "hard_pair_fraction", "mean_valid_semantic_pairs") if c in data and data[c].notna().any()]
            if components or diagnostics:
                fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
                fig.suptitle(f"Training diagnostics  ·  {LABELS[job['model']]}  ·  {job['density']:.0%}  ·  seed {job['seed']}", fontsize=13)
                for index, col in enumerate(components):
                    axes[0].plot(data["epoch"], data[col], label=col.removeprefix("loss_"), linewidth=1.6,
                                 marker=("o", "s", "^", "D", "v")[index % 5], markersize=3,
                                 linestyle=("-", "--", "-.", ":")[index % 4],
                                 markevery=max(1, len(data) // 10))
                if components:
                    axes[0].legend(fontsize=8, frameon=False)
                axes[0].set(title="Weighted objective contributions", xlabel="Epoch", ylabel="Contribution to recorded loss")
                # Separate gradient scale from fractions/pair counts.
                if "mean_gradient_norm" in diagnostics:
                    axes[1].plot(data["epoch"], data["mean_gradient_norm"], color="#0072B2", marker="o")
                axes[1].set(title="Gradient magnitude", xlabel="Epoch", ylabel="Mean gradient norm")
                save(fig, f"diagnostics_{job['id']}", "Recorded weighted objective contributions (BPR has unit multiplier) and gradient norm without smoothing. Raw component losses and scalar weights remain in CSV; optimizer-applied weight decay is excluded. Clipping fractions and semantic pair counts are preserved in the history CSV.", [history.relative_to(root).as_posix()])

            semantic_panels = [
                ("Item gate", "Fraction / mean gate value", (0, 1), [
                    ("diagnostic_gate_near_zero_fraction", "Coordinates < 0.1", "#A63636", "o", "-"),
                    ("diagnostic_gate_mean", "Mean gate", "#0072B2", "s", "--"),
                ]),
                ("Local sigmoid sensitivity", "Mean g(1-g); not an objective gradient", (0, .25), [
                    ("diagnostic_gate_local_sigmoid_derivative_mean", "Local derivative", "#007A59", "D", "-"),
                ]),
                ("Global layer-zero weights", "E0 mass", (0, 1), [
                    ("diagnostic_layer_weight_0", "Effective aggregation", "#0072B2", "o", "-"),
                    ("diagnostic_learned_layer0_mass", "Unanchored softmax", "#A63636", "s", "--"),
                ]),
                ("Semantic contribution relative to ID", "Mean per-row norm ratio", None, [
                    ("diagnostic_item_text_to_id_ratio", "Item text / ID", "#0072B2", "o", "-"),
                    ("diagnostic_user_semantic_to_id_ratio", "User text / ID", "#914B88", "s", "--"),
                ]),
            ]
            if any(column in data and data[column].notna().any()
                   for _, _, _, series in semantic_panels for column, *_ in series):
                fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
                fig.suptitle(f"Semantic diagnostics  ·  {LABELS[job['model']]}  ·  {job['density']:.0%}  ·  seed {job['seed']}", fontsize=13)
                for axis, (title, ylabel, limits, series) in zip(axes.flat, semantic_panels):
                    plotted = False
                    for column, label, color, marker, linestyle in series:
                        if column in data and data[column].notna().any():
                            axis.plot(data["epoch"], data[column], label=label, color=color,
                                      marker=marker, linestyle=linestyle, markersize=3)
                            plotted = True
                    if plotted:
                        axis.legend(fontsize=8, frameon=False)
                    else:
                        axis.text(.5, .5, "Not recorded", transform=axis.transAxes, ha="center")
                    axis.set(title=title, xlabel="Epoch", ylabel=ylabel,
                             xlim=(.5, data["epoch"].max() + .5))
                    axis.xaxis.get_major_locator().set_params(integer=True)
                    if limits is not None:
                        axis.set_ylim(*limits)
                    else:
                        axis.set_ylim(bottom=0)
                save(fig, f"semantic_{job['id']}",
                     "Raw deterministic sampled diagnostics; sample counts are in history CSV. Gate fractions count coordinates, "
                     "not items. Near-zero gates admit the bounded text residual while retaining ID in bounded_residual mode. "
                     "g(1-g) is a local sigmoid derivative, not a loss gradient; for a fixed-open gate it is zero by construction. "
                     "Effective aggregation weights include anchoring; unanchored softmax masses are separate. Missing fields "
                     "remain gaps or are marked not recorded. Norm ratios are mean per-row ratios. No causal ranking claim or confidence interval.",
                     [history.relative_to(root).as_posix()])

        if not frame.empty:
            split = "val" if plan["validation_only"] else "test"
            metrics = [f"{split}/NDCG@20", f"{split}/Recall@20"]
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
            fig.suptitle(f"{'Validation' if split == 'val' else 'Test'} ranking across training density", fontsize=15, fontweight="bold")
            for axis, metric in zip(axes, metrics):
                for offset, model in enumerate(plan["models"]):
                    points = summary[summary["model"] == model].set_index("density").reindex(sorted(plan["densities"]))
                    mean_col, sd_col = f"{metric}/mean", f"{metric}/sd"
                    if mean_col not in points:
                        continue
                    xs = points.index.to_numpy(dtype=float) * 100
                    axis.plot(xs, points[mean_col], marker=MARKERS[model], color=COLORS[model], linewidth=2, label=LABELS[model])
                    valid = points[sd_col].notna() & points[mean_col].notna()
                    axis.errorbar(xs[valid], points.loc[valid, mean_col], yerr=points.loc[valid, sd_col], fmt="none", ecolor=COLORS[model], capsize=4)
                    raw = frame[frame["model"] == model]
                    if metric in raw:
                        axis.scatter(raw["density"] * 100 + (offset - (len(plan["models"]) - 1) / 2) * 0.6,
                                     raw[metric], color=COLORS[model], marker=MARKERS[model], s=23, alpha=0.5)
                axis.set(title=metric.split("/")[1], xlabel="Retained positive training edges (%)", ylabel=metric.split("/")[1])
                axis.set_xticks(sorted({d * 100 for d in plan["densities"]}))
                axis.set_ylim(bottom=0)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=min(4, len(handles)), frameon=False)
            save(fig, "ranking_by_density", "Lines: mean of available seeds. Whiskers: sample SD between seeds (ddof=1), omitted for n=1. Small markers: actual seeds, horizontally offset for readability. Missing densities remain gaps. Counts are in summary.csv. Seed controls initialization and graph sampling; SD is not a confidence interval.", ["aggregated/runs.csv", "aggregated/summary.csv"])

            def metric_grid(metrics_to_plot, title, name):
                available = [m for m in metrics_to_plot if m in frame and pd.to_numeric(frame[m], errors="coerce").notna().any()]
                if not available:
                    return
                conditions = [(m, d) for d in sorted(plan["densities"], reverse=True) for m in plan["models"]]
                fig, axes = plt.subplots(1, len(available), figsize=(max(8, 4.4 * len(available)), max(4.8, 2.6 + 0.32 * len(conditions))), squeeze=False, sharey=True)
                fig.suptitle(title, fontsize=14, fontweight="bold")
                ticklabels = [f"{LABELS[m]} · {d:.0%}" for m, d in conditions]
                for axis, metric in zip(axes[0], available):
                    for index, (model, density) in enumerate(conditions):
                        values = pd.to_numeric(frame.loc[(frame["model"] == model) & (frame["density"] == density), metric], errors="coerce").dropna()
                        if values.empty:
                            continue
                        axis.scatter(values, np.repeat(index, len(values)), color=COLORS[model], marker=MARKERS[model], alpha=0.5, s=20)
                        axis.plot(values.mean(), index, marker=MARKERS[model], color=COLORS[model], markersize=7)
                        if len(values) > 1:
                            axis.errorbar(values.mean(), index, xerr=values.std(ddof=1), fmt="none", ecolor=COLORS[model], capsize=4)
                    axis.set_yticks(range(len(conditions)), ticklabels, fontsize=8)
                    axis.set_ylim(len(conditions) - 0.5, -0.5)
                    display = metric.removeprefix("test/").removeprefix("representation/").replace("_", " ")
                    if metric.startswith("subgroup/"):
                        display = metric.split("/")[1].split(" (")[0] + " " + metric.split("/")[-1]
                    axis.set(title=display, xlabel="Recorded value")
                    if "uniformity" not in metric:
                        axis.set_xlim(left=0)
                fig.subplots_adjust(left=0.21)
                save(fig, name, "Raw seed observations, mean and sample SD (ddof=1; absent for n=1). Different panels retain their original units and scales. Missing conditions are omitted, never filled with zero.", ["aggregated/runs.csv"])

            metric_grid(["test/Diversity@10", "test/Novelty@10", "test/Coverage@10"], "Beyond ranking accuracy", "beyond_accuracy")
            metric_grid(["train_seconds", "latency_ms_per_user"], "Training time and inference cost", "efficiency")
            metric_grid(["representation/alignment", "representation/mean_uniformity", "representation/user_effective_rank"], "Representation measurements", "representation")
            metric_grid([c for c in frame if c.startswith("subgroup/") and c.endswith("/NDCG@20")], "Ranking by fixed user activity group", "activity_groups")
    return figures


def export_report(root, render=True):
    root = Path(root).resolve()
    plan, runs = load_suite(root)
    frame = flatten_runs(runs)
    summary = aggregate(frame)
    output = root / "aggregated"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if not render:
        return
    figures = render_figures(root, plan, runs, frame, summary)
    manifest = {"schema_version": 1, "bundle_id": plan["bundle_id"], "dataset_manifest_sha256": plan["manifest_sha256"],
                "result_sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for _, _, p in runs],
                "history_sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
                                    for p in sorted((root / "history").rglob("*_history.csv"))],
                "table_sources": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for p in (output / "runs.csv", output / "summary.csv")],
                "transformations": ["One suite only", "Available seeds grouped by model/density", "Nonfinite values stored as missing", "Sample SD ddof=1 only for n>=2", "No smoothing or radar normalization"],
                "figures": figures}
    (root / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    state = "COMPLETE" if len(runs) == len(plan["jobs"]) else "PARTIAL"
    stage = "VALIDATION ITERATION" if plan["validation_only"] else "TEST BENCHMARK"
    text = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Training report</title>",
            "<style>body{font:15px system-ui;color:#17283b;background:#f5f7fa;max-width:1250px;margin:auto;padding:30px}h1{font-size:30px}section{background:white;border:1px solid #dae2ec;border-radius:12px;padding:22px;margin:20px 0}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dae2ec;text-align:right}th{background:#edf2f7}a{color:#0072b2}.table{overflow:auto}p{line-height:1.6}.badge{background:#e3ebf5;padding:6px 12px;border-radius:6px}</style>",
            f"<h1>Graph recommendation · training report</h1><span class='badge'>{stage} · {state} · {len(runs)}/{len(plan['jobs'])} jobs</span>",
            f"<p>Dataset: <code>{html.escape(plan['bundle_id'])}</code>. Epoch budget: {plan['epochs']}. Checkpoints selected using validation.</p>",
            "<p>Use validation to adjust settings. Test results are exploratory if used repeatedly for tuning. Mean/SD describe the available seeds on this dataset; a single seed has no estimated SD. No significance or automatic model recommendation is claimed.</p>",
            "<p><a href='aggregated/runs.csv'>Raw results CSV</a> · <a href='aggregated/summary.csv'>Mean / SD / n CSV</a> · <a href='suite_plan.json'>Frozen plan</a> · <a href='figure_manifest.json'>Figure provenance</a></p>"]
    jobs = pd.DataFrame([{k: j[k] for k in ("model", "density", "seed", "status")} for j in plan["jobs"]])
    text.append("<section><h2>Job status</h2><div class='table'>" + jobs.to_html(index=False, escape=True) + "</div></section>")
    if not frame.empty:
        cols = [c for c in ("model", "density", "seed", "best_epoch", "total_epochs", "val/NDCG@20", "test/NDCG@20", "test/Recall@20", "train_seconds") if c in frame]
        text.append("<section><h2>Actual results</h2><div class='table'>" + frame[cols].to_html(index=False, na_rep="N/A", float_format=lambda x: f"{x:.6g}", escape=True) + "</div></section>")
    for figure in figures:
        png, svg = figure["files"]
        text.append(f"<section><h2>{html.escape(figure['title'])}</h2><img src='{png['path']}' alt='{html.escape(figure['caption'], quote=True)}'><p>{html.escape(figure['caption'])}</p><a href='{png['path']}'>PNG</a> · <a href='{svg['path']}'>SVG</a></section>")
    text.append("<p>Workflow informed by Kassis et al. (2026), <a href='https://doi.org/10.48550/arXiv.2609.00065'>Scientific Agent Skills</a>. Source data and transformations accompany every export.</p></html>")
    (root / "report.html").write_text("\n".join(text), encoding="utf-8")
