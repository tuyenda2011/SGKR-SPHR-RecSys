from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_statistical_significance(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Perform paired t-test and Wilcoxon signed-rank test between two models across seeds/users.

    Args:
        scores_a: Array of evaluation scores for Model A (Target/Proposed)
        scores_b: Array of evaluation scores for Model B (Baseline)

    Returns:
        Dict containing mean_a, mean_b, relative_improvement_pct, t_stat, p_value, and significance_star.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1:
        raise ValueError("Paired significance tests require equally sized 1-D arrays")
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        raise ValueError("Scores must be finite")
    count = len(arr_a)
    mean_a = float(arr_a.mean()) if count else float("nan")
    mean_b = float(arr_b.mean()) if count else float("nan")
    result = {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "paired_samples": count,
        "rel_improvement_pct": (mean_a - mean_b) / mean_b * 100
        if mean_b != 0
        else float("nan"),
        "t_statistic": float("nan"),
        "p_value": float("nan"),
        "t_p_value": float("nan"),
        "wilcoxon_statistic": float("nan"),
        "wilcoxon_p_value": float("nan"),
        "significance": "N/A",
        "status": "insufficient_samples",
    }
    if count < 2:
        return result
    result["status"] = "undefined_test"
    differences = arr_a - arr_b
    if np.var(differences) > 0:
        t_result = stats.ttest_rel(arr_a, arr_b)
        if np.isfinite(t_result.pvalue) and np.isfinite(t_result.statistic):
            p = float(t_result.pvalue)
            result.update(
                t_statistic=float(t_result.statistic),
                p_value=p,
                t_p_value=p,
                status="ok",
                significance="***"
                if p < 0.001
                else "**"
                if p < 0.01
                else "*"
                if p < 0.05
                else "ns",
            )
    if np.any(differences != 0):
        try:
            w_result = stats.wilcoxon(arr_a, arr_b)
            if np.isfinite(w_result.pvalue):
                result.update(
                    wilcoxon_statistic=float(w_result.statistic),
                    wilcoxon_p_value=float(w_result.pvalue),
                )
        except ValueError:
            pass
    return result


def summarize_metric(values) -> dict:
    """Sample standard deviation is unavailable for fewer than two valid runs."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    display = f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f} (std N/A)"
    return {"mean": mean, "std": std, "str": display if values.size else "N/A"}


def generate_latex_table(
    summary_df: pd.DataFrame,
    caption: str = "Performance comparison of LightGCN, XSimGCL, DirectAU, and RecFlow across data sparsity levels.",
    label: str = "tab:benchmark_results",
    ranking_k: int = 10,
) -> str:
    """Generate publication-ready LaTeX table formatted according to ACM/IEEE guidelines.

    Highlights best results in bold and statistically significant improvements with asterisks.
    """
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{llcccccc}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Sparsity} & \\textbf{Model} & "
        + f"\\textbf{{Recall@{ranking_k}}} & \\textbf{{NDCG@{ranking_k}}} & "
        + "\\textbf{MRR@10} & \\textbf{Diversity@10} & \\textbf{Novelty@10} & \\textbf{Coverage@10} \\\\"
    )
    lines.append("    \\midrule")

    # Group by sparsity level
    if "sparsity" in summary_df.columns:
        sparsities = sorted(summary_df["sparsity"].unique(), reverse=True)
    else:
        sparsities = [1.0]

    for s_idx, sp in enumerate(sparsities):
        sp_df = (
            summary_df[summary_df["sparsity"] == sp]
            if "sparsity" in summary_df.columns
            else summary_df
        )
        sp_pct = f"{int(float(sp) * 100)}\\%"

        # Find max for each metric to format in bold
        metrics = [
            f"Recall@{ranking_k}",
            f"NDCG@{ranking_k}",
            "MRR@10",
            "Diversity@10",
            "Novelty@10",
            "Coverage@10",
        ]
        max_vals = {}
        for m in metrics:
            if m in sp_df.columns:
                max_vals[m] = sp_df[m].max()

        for row_idx, (_, row) in enumerate(sp_df.iterrows()):
            m_name = row.get("model", "").upper()
            sp_label = (
                f"\\multirow{{{len(sp_df)}}}{{*}}{{{sp_pct}}}" if row_idx == 0 else ""
            )

            row_entries = [sp_label, m_name]
            for m in metrics:
                if m in row and pd.notna(row[m]):
                    val = row[m]
                    val_str = f"{val:.4f}"
                    std_col = f"{m}_std"
                    if std_col in row and pd.notna(row[std_col]):
                        val_str += f" $\\pm$ {row[std_col]:.4f}"
                    if abs(val - max_vals.get(m, -999)) < 1e-6:
                        val_str = f"\\textbf{{{val_str}}}"
                    row_entries.append(val_str)
                else:
                    row_entries.append("N/A")

            lines.append("    " + " & ".join(row_entries) + " \\\\")

        if s_idx < len(sparsities) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\vspace{1ex}")
    lines.append(
        "  {\\footnotesize \\textit{Note:} Values are mean $\\pm$ sample standard deviation when multiple runs are available. Bold denotes the largest observed mean, not statistical significance. Single-run standard deviations and unavailable tests are N/A; see the separate paired-test report.}"
    )
    lines.append("\\end{table*}")

    return "\n".join(lines)
