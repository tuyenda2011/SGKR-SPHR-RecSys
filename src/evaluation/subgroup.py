from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import compute_topk_metrics


def stratify_users_by_degree(
    train_df: pd.DataFrame,
    eval_users: List[int],
    quantiles: Tuple[float, float] = (0.2, 0.8),
) -> Dict[str, List[int]]:
    """Partition evaluation users into Tail (low-activity), Torso, and Head (active) groups.

    Args:
        train_df: Training DataFrame containing 'u_idx'
        eval_users: List of evaluation user IDs
        quantiles: Lower and upper quantile thresholds (default: 20th and 80th percentiles)

    Returns:
        Dict mapping group name ('Tail', 'Torso', 'Head') to list of user indices in eval_users.
    """
    user_counts = train_df["u_idx"].value_counts()
    eval_user_degrees = np.array([user_counts.get(u, 0) for u in eval_users])

    q_low = np.quantile(eval_user_degrees, quantiles[0])
    q_high = np.quantile(eval_user_degrees, quantiles[1])

    tail_indices = []
    torso_indices = []
    head_indices = []

    for idx, deg in enumerate(eval_user_degrees):
        if deg <= q_low:
            tail_indices.append(idx)
        elif deg <= q_high:
            torso_indices.append(idx)
        else:
            head_indices.append(idx)

    return {
        "Tail (Low-Activity)": tail_indices,
        "Torso (Medium)": torso_indices,
        "Head (Active)": head_indices,
    }


def evaluate_degree_subgroups(
    evaluator: Any,
    final_user_embeds: torch.Tensor,
    final_item_embeds: torch.Tensor,
    device: torch.device,
    train_df: pd.DataFrame,
    k_list: List[int] = [10, 20],
) -> Dict[str, Dict[str, float]]:
    """Evaluate Top-K recommendation performance stratified across user degree subgroups.

    Returns:
        Dict mapping subgroup name to metric dictionary (e.g. Recall@10, NDCG@10).
    """
    topk_preds, _ = evaluator.get_predictions(final_user_embeds, final_item_embeds, device)
    groups = stratify_users_by_degree(train_df, evaluator.eval_users)

    subgroup_results = {}
    for group_name, user_indices in groups.items():
        if not user_indices:
            continue

        group_ground_truth = [evaluator.ground_truth[i] for i in user_indices]
        group_preds = topk_preds[user_indices]

        group_metrics = compute_topk_metrics(group_ground_truth, group_preds, k_list)
        group_metrics["num_users"] = len(user_indices)
        subgroup_results[group_name] = group_metrics

    return subgroup_results
