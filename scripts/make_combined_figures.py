"""The report figures: all four densities in one picture. Drawn by make_report.py into reports/figures.

  H1_density_<metric>              one group per density, one bar per model (metrics.json)
  H2_<metric>                      as H1, from the per-user scores
  H3_learning_curves_<metric>      2x2 grid, one panel per density, shared axes
  H3c_loss                         2x2 grid, one panel per model (own y-axis), one line per density
  H4_retention_<metric>            as H1, metric relative to the same model at density 1.00 (= 100%)
  H5_user_groups_<metric>          2x2 grid, one panel per density; user groups by full-train history length
  H6a_popularity_trajectory        NDCG@20 versus popularity of recommended items, one path per model
  H6b_coverage, H7a_train_time, H7b_inference_latency   as H1
  H8_ndcg_at_k                     2x2 grid, one panel per density

Validation split only. Sized for full text width on A4.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from scripts.make_report import (FIGURE_METRICS, GRID, INK, MUTED, ORDER, PROPOSED, SIZE, STYLE, model_average,
                                 read_history, slug, style_axes)
from scripts.train_model import DENSITIES

ASCENDING = sorted(DENSITIES)                # bar groups, sparse to full (as H1)
PANELS = sorted(DENSITIES, reverse=True)     # 2x2 grids, full to sparse
BINS = [("3–4", 0, 4), ("5–9", 5, 9), ("≥10", 10, np.inf)]
# Printed full text width on A4 (about 16 cm, so close to 1:1): in-plot labels no smaller than 7 pt.
LABEL = 7
TALL = (SIZE[0], 5.2)  # about 16 x 13 cm


def save(fig, folder, name, notes):
    fig.savefig(folder / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)
    notes.append(f"{name}: written")


def legend(fig, handles=None, labels=None, ncol=3):
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, columnspacing=1.2,
               handlelength=2.2)


def hatch(model):
    return None if model in PROPOSED else "//"


def density_bars(ax, value, fmt="{:.4f}", densities=ASCENDING, order=ORDER, style=STYLE):
    """H1 layout: one group per density, one bar per model. value(model, density) -> None (not run) or a
    number. A missing bar's own slot is marked "chưa chạy". order/style select another set of models
    (the closed-form supplement)."""
    width = .8 / len(order)
    tops, missing = [], []
    for k, model in enumerate(order):
        spec = style[model]
        ax.bar(np.nan, np.nan, color=spec["color"], hatch=hatch(model), edgecolor="white", label=spec["label"])
        for g, density in enumerate(densities):
            x = g + (k - (len(order) - 1) / 2) * width
            result = value(model, density)
            if result is None:
                missing.append(x)
                continue
            ax.bar(x, result, width=width, color=spec["color"], edgecolor="white", linewidth=1.5, hatch=hatch(model))
            ax.annotate(fmt.format(result), (x, result), xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=LABEL, rotation=90)
            tops.append(result)
    ax.set_xticks(range(len(densities)), [f"{d:.2f}" for d in densities])
    ax.set_xlim(-.6, len(densities) - .4)
    ax.set_xlabel("Mật độ dữ liệu huấn luyện (tỉ lệ tương tác được giữ lại)")
    return tops, missing


def mark_missing(ax, missing):
    low, high = ax.get_ylim()
    base = low * (high / low) ** .02 if ax.get_yscale() == "log" else low + (high - low) * .02
    for x in missing:
        ax.text(x, base, "chưa chạy", rotation=90, ha="center", va="bottom", fontsize=LABEL, color=MUTED)


def bar_figure(value, ylabel, name, folder, notes, fmt="{:.4f}", log=False, extra=None, order=ORDER, style=STYLE,
               ncol=3):
    fig, ax = plt.subplots(figsize=SIZE, layout="constrained")
    style_axes(ax)
    tops, missing = density_bars(ax, value, fmt, order=order, style=style)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min(tops) / 3 if tops else .1, max(tops) * 6 if tops else 10)
    else:
        ax.set_ylim(0, (max(tops) if tops else 1) * 1.22)
    if extra:
        extra(ax)
    mark_missing(ax, missing)
    ax.set_ylabel(ylabel)
    legend(fig, ncol=ncol)
    save(fig, folder, name, notes)


def figure_density(summary, metric, folder, notes):
    """Metric from metrics.json, mean over seeds."""
    means = summary.groupby(["model", "density"])[metric].mean()
    bar_figure(lambda m, d: means[(m, d)] if (m, d) in means.index else None, metric,
               f"H1_density_{slug(metric)}", folder, notes)


def figure_bars(scores, metric, folder, notes):
    def value(model, density):
        _, values = model_average(scores, model, density, metric)
        return None if values is None else values.mean()
    bar_figure(value, metric, f"H2_{slug(metric)}", folder, notes)


def figure_retention(summary, metric, folder, notes):
    means = summary.groupby(["model", "density"])[metric].mean()

    def value(model, density):
        if (model, density) not in means.index or (model, 1.0) not in means.index:
            return None
        return 100 * means[(model, density)] / means[(model, 1.0)]

    def reference(ax):
        ax.axhline(100, color=MUTED, linestyle="--", linewidth=.8, zorder=0)
    bar_figure(value, f"{metric} so với mật độ 1.00 (%)", f"H4_retention_{slug(metric)}", folder, notes,
               fmt="{:.1f}", extra=reference)


def grid(size=SIZE):
    fig, axes = plt.subplots(2, 2, figsize=size, layout="constrained", sharex=True, sharey=True)
    for ax, density in zip(axes.flat, PANELS):
        style_axes(ax)
        ax.set_title(f"Mật độ {density:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    return fig, axes


def shared_xlabel(fig, axes, text, pad=22):
    """One x label under the bottom row, above the legend (fig.supxlabel would collide with it).
    pad clears the bottom row's tick labels: about 22 points for one line, 38 for two."""
    span = fig.add_subplot(axes[1, 0].get_gridspec()[1, :], frameon=False)
    span.set_xticks([])
    span.set_yticks([])
    span.patch.set_visible(False)
    span.set_xlabel(text, fontsize=9, labelpad=pad)


def unavailable(ax):
    ax.text(.5, .5, "chưa chạy", transform=ax.transAxes, ha="center", va="center", color=MUTED)


def figure_curves(summary, metric, folder, notes, seed=42):
    fig, axes = grid()
    handles = {}
    for ax, density in zip(axes.flat, PANELS):
        drawn = 0
        for model in ORDER:
            row = summary[(summary.model == model) & (summary.density == density) & (summary.seed == seed)]
            if row.empty:
                continue
            spec = STYLE[model]
            if model == "sgkr":
                line = ax.axhline(row[metric].iloc[0], color=spec["color"], linestyle=":", linewidth=1.3)
                handles.setdefault(model, (line, f"{spec['label']} – nghiệm đóng"))
                drawn += 1
                continue
            history = read_history(row.path.iloc[0])
            if history is None or metric not in history:
                continue
            (line,) = ax.plot(history["epoch"], history[metric], color=spec["color"], linestyle=spec["line"],
                              linewidth=1.4)
            best = history[metric].idxmax()
            ax.plot(history["epoch"][best], history[metric][best], marker=spec["marker"], markersize=5,
                    color=spec["color"], markeredgecolor="white", markeredgewidth=.8)
            handles.setdefault(model, (line, spec["label"]))
            drawn += 1
        if not drawn:
            unavailable(ax)
    for ax in axes[1]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    shared_xlabel(fig, axes, "Epoch")
    fig.supylabel(metric, fontsize=9)
    ordered = [handles[m] for m in ORDER if m in handles]
    legend(fig, [h for h, _ in ordered], [label for _, label in ordered])
    save(fig, folder, f"H3_learning_curves_{slug(metric)}", notes)


DENSITY_LINES = {1.0: "-", .75: "--", .5: "-.", .25: ":"}  # also readable in black-and-white print


def shade(color, density):
    """The model's colour, mixed towards white for sparser training data (1.00 = full colour)."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * .6 * (1 - density) / .75)


def figure_loss(summary, folder, notes, seed=42):
    """Training loss per epoch, one panel per gradient-trained model (own y-axis: the losses differ in kind
    and scale), one line per density."""
    models = [m for m in ORDER if m != "sgkr"]
    fig, axes = plt.subplots(2, 2, figsize=SIZE, layout="constrained", sharex=True)
    for ax, model in zip(axes.flat, models):
        style_axes(ax)
        spec = STYLE[model]
        ax.set_title(spec["short"], fontsize=9)
        ax.tick_params(labelsize=8)
        drawn = 0
        for density in PANELS:
            row = summary[(summary.model == model) & (summary.density == density) & (summary.seed == seed)]
            history = None if row.empty else read_history(row.path.iloc[0])
            if history is None or "loss" not in history:
                continue
            color = shade(spec["color"], density)
            ax.plot(history["epoch"], history["loss"], color=color, linestyle=DENSITY_LINES[density], linewidth=1.3)
            best = history["NDCG@20"].idxmax()
            ax.plot(history["epoch"][best], history["loss"][best], marker=spec["marker"], markersize=5,
                    color=color, markeredgecolor="white", markeredgewidth=.8, linestyle="none")
            drawn += 1
        if not drawn:
            unavailable(ax)
    for ax in axes[1]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    shared_xlabel(fig, axes, "Epoch")
    fig.supylabel("Loss huấn luyện", fontsize=9)
    handles = [plt.Line2D([], [], color=INK, linestyle=DENSITY_LINES[d], linewidth=1.3) for d in PANELS]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=5,
                              markeredgecolor="white"))
    legend(fig, handles, [f"Mật độ {d:.2f}" for d in PANELS] + ["Epoch tốt nhất theo NDCG@20"], ncol=5)
    save(fig, folder, "H3c_loss", notes)


def figure_groups(scores, full_train, metric, folder, notes, order=ORDER, style=STYLE, name="H5_user_groups",
                  ncol=3):
    degree = full_train.groupby("u_idx").size()
    fig, axes = grid(TALL)  # 60 labelled bars need the extra height
    width = .8 / len(order)
    tops = []
    for ax, density in zip(axes.flat, PANELS):
        sizes, drawn = None, 0
        for k, model in enumerate(order):
            users, values = model_average(scores, model, density, metric)
            if values is None:
                continue
            n = degree.reindex(users).to_numpy()
            masks = [(n >= lo) & (n <= hi) for _, lo, hi in BINS]
            sizes = sizes or [int(mask.sum()) for mask in masks]
            for g, mask in enumerate(masks):
                x, mean = g + (k - (len(order) - 1) / 2) * width, values[mask].mean()
                ax.bar(x, mean, width=width, color=style[model]["color"], edgecolor="white", linewidth=1,
                       hatch=hatch(model))
                ax.annotate(f"{mean:.4f}", (x, mean), xytext=(0, 1.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=LABEL, rotation=90)
                tops.append(mean)
            drawn += 1
        if not drawn:
            unavailable(ax)
        # The same users in every panel, so the bottom row's labels (shared x axis) serve all four.
        ax.set_xticks(range(len(BINS)), [name if sizes is None else f"{name}\n(n = {size:,})"
                                         for (name, _, _), size in zip(BINS, sizes or [0] * len(BINS))])
    axes[0, 0].set_ylim(0, (max(tops) if tops else 1) * 1.3)  # room for the value labels (shared y)
    shared_xlabel(fig, axes, "Số tương tác của user trong tập huấn luyện gốc", pad=38)
    fig.supylabel(metric, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=style[m]["color"], edgecolor="white", hatch=hatch(m))
               for m in order]
    legend(fig, handles, [style[m]["label"] for m in order], ncol=ncol)
    save(fig, folder, f"{name}_{slug(metric)}", notes)


def popularity_stats(scores, full_train, num_items):
    """Per (model, density): mean log(1 + popularity) of recommended items, NDCG@20, catalog coverage."""
    counts = np.bincount(full_train.i_idx.to_numpy(dtype=np.int64), minlength=num_items)
    active = np.count_nonzero(counts)
    stats = {}
    for model, density, seed in sorted(scores):
        if (model, density) in stats:
            continue  # first seed, as in make_report.py
        _, predictions, values, _ = scores[(model, density, seed)]
        stats[(model, density)] = {"pop": float(np.log1p(counts[predictions]).mean()),
                                   "ndcg": float(values["NDCG@20"].mean()),
                                   "coverage": 100 * len(np.unique(predictions)) / active}
    return stats


def figure_popularity(stats, folder, notes):
    fig, ax = plt.subplots(figsize=SIZE, layout="constrained")
    style_axes(ax)
    for model in ORDER:
        points = [(d, stats[(model, d)]) for d in PANELS if (model, d) in stats]
        if not points:
            continue
        spec = STYLE[model]
        xs, ys = [p["pop"] for _, p in points], [p["ndcg"] for _, p in points]
        ax.plot(xs, ys, color=spec["color"], linestyle=spec["line"], linewidth=1.2, label=spec["label"])
        sizes = [18 + 50 * d for d, _ in points]  # larger marker = denser training data
        ax.scatter(xs, ys, s=sizes, color=spec["color"], marker=spec["marker"], edgecolor="white", linewidth=.8,
                   zorder=3)
        for (density, _), x, y in zip(points, xs, ys):
            ax.annotate(f"{density:.2f}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=LABEL,
                        color=MUTED)
    ax.set_xlabel("Độ phổ biến trung bình của item được gợi ý, log(1 + số lượt tương tác)")
    ax.set_ylabel("NDCG@20")
    ax.grid(axis="x", color=GRID, linewidth=.6)
    legend(fig)
    save(fig, folder, "H6a_popularity_trajectory", notes)


def figure_ndcg_at_k(scores, folder, notes, order=ORDER, style=STYLE, name="H8_ndcg_at_k", ncol=3):
    discount = 1 / np.log2(np.arange(2, 22))
    ks = np.arange(1, 21)
    fig, axes = grid()
    for ax, density in zip(axes.flat, PANELS):
        drawn = 0
        for model in order:
            curves = []
            for key in (k for k in scores if k[0] == model and k[1] == density):
                _, predictions, _, truth = scores[key]
                curves.append(((predictions == truth[:, None]) * discount).cumsum(axis=1).mean(axis=0))
            if not curves:
                continue
            spec = style[model]
            ax.plot(ks, np.mean(curves, axis=0), color=spec["color"], linestyle=spec["line"], marker=spec["marker"],
                    markersize=3, markevery=[0, 4, 9, 19], linewidth=1.4, label=spec["label"])
            drawn += 1
        if not drawn:
            unavailable(ax)
        ax.set_xticks([1, 5, 10, 15, 20])
    shared_xlabel(fig, axes, "Độ dài danh sách K")
    fig.supylabel("NDCG@K", fontsize=9)
    handles = [plt.Line2D([], [], color=style[m]["color"], linestyle=style[m]["line"], marker=style[m]["marker"],
                          markersize=3, linewidth=1.4) for m in order]
    legend(fig, handles, [style[m]["label"] for m in order], ncol=ncol)
    save(fig, folder, name, notes)


def draw_all(summary, scores, full_train, num_items, folder, notes):
    first = summary.sort_values("seed").drop_duplicates(["model", "density"]).set_index(["model", "density"])
    for metric in FIGURE_METRICS:
        figure_density(summary, metric, folder, notes)
        figure_bars(scores, metric, folder, notes)
        figure_retention(summary, metric, folder, notes)
        figure_groups(scores, full_train, metric, folder, notes)
    for metric in ("NDCG@20", "NDCG@10"):
        figure_curves(summary, metric, folder, notes)
    figure_loss(summary, folder, notes)
    popularity = popularity_stats(scores, full_train, num_items)
    figure_popularity(popularity, folder, notes)
    bar_figure(lambda m, d: popularity[(m, d)]["coverage"] if (m, d) in popularity else None,
               "Coverage@20 (% item được gợi ý ít nhất một lần)", "H6b_coverage", folder, notes, fmt="{:.1f}")
    # Log scale for both: SGKR (CPU) is one to two orders of magnitude away from the GPU models.
    for name, column, label, fmt in (
            ("H7a_train_time", "train_seconds", "Thời gian huấn luyện (giây, thang log)", "{:,.0f}"),
            ("H7b_inference_latency", "inference_ms_per_user", "Thời gian suy luận (ms/user, thang log)",
             "{:.3f}")):
        bar_figure(lambda m, d: float(first.at[(m, d), column]) if (m, d) in first.index else None,
                   f"{label}\nSGKR chạy trên CPU, các mô hình khác trên GPU", name, folder, notes, fmt=fmt, log=True)
    figure_ndcg_at_k(scores, folder, notes)
