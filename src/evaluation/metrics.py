from typing import Any, Dict, List, Optional, Set, Tuple, Union  # noqa: F401 -- preserve legacy import AST
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


def compute_topk_metrics(
    pos_items_list: List[List[int]],
    topk_predictions: torch.Tensor,
    k_list: List[int] = [10, 20],
) -> Dict[str, float]:
    """Compute standard ranking accuracy metrics (Recall@K, NDCG@K, MRR@K).

    Args:
        pos_items_list: Ground truth item ID lists for each user in evaluation batch
        topk_predictions: Tensor of shape (num_eval_users, max_k) containing recommended item IDs
        k_list: List of K values (e.g., [10, 20])

    Returns:
        Dict mapping metric names to mean values.
    """
    max_k = max(k_list)
    predictions = np.asarray(topk_predictions[:, :max_k].cpu().numpy(), dtype=np.int64)
    num_users = len(pos_items_list)
    if predictions.shape[0] < num_users:
        raise ValueError("One prediction row is required per ground-truth user")
    predictions = predictions[:num_users]

    # Users without targets are skipped; duplicate targets count once.
    lengths = np.fromiter((len(items) for items in pos_items_list), dtype=np.int64, count=num_users)
    results = {}
    if not lengths.any():
        for k in k_list:
            results.update({f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0, f"MRR@{k}": 0.0})
        return results
    items = np.fromiter((item for user_items in pos_items_list for item in user_items),
                        dtype=np.int64, count=int(lengths.sum()))

    # Hit matrix via (row, item) keys: one vectorized membership test for all users.
    width = int(max(predictions.max(initial=0), items.max())) + 1
    target_keys = np.unique(np.repeat(np.arange(num_users, dtype=np.int64), lengths) * width + items)
    target_counts = np.bincount(target_keys // width, minlength=num_users)
    valid = target_counts > 0
    prediction_keys = np.arange(num_users, dtype=np.int64)[:, None] * width + predictions
    hits = np.isin(prediction_keys, target_keys)[valid]
    counts = target_counts[valid]

    # Reproduce the former per-user arithmetic exactly (scripts/compare_val_runs.py pins it):
    # float32 hit sums divided by a Python int, row-wise np.sum for DCG/IDCG.
    recall_dtype = (np.float32(1) / 1).dtype
    for k in k_list:
        hits_k = hits[:, :k].astype(np.float32)
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = np.sum(hits_k * discount, axis=1)
        idcg = np.array([np.sum(discount[:n]) for n in range(k + 1)])[np.minimum(counts, k)]
        first = np.where(hits_k.any(axis=1), hits_k.argmax(axis=1) + 1, np.inf)
        recall = (hits_k.sum(axis=1, dtype=np.float32).astype(recall_dtype) / counts.astype(recall_dtype))
        results[f"Recall@{k}"] = float(np.mean(recall.astype(recall_dtype)))
        results[f"NDCG@{k}"] = float(np.mean(dcg / idcg))
        results[f"MRR@{k}"] = float(np.mean(1.0 / first))

    return results


def compute_intra_list_diversity(
    topk_predictions: torch.Tensor,
    item_embeddings: torch.Tensor,
    k: int = 10,
    sample_users: int = 1000,
    item_mask: torch.Tensor | None = None,
) -> float:
    """Compute Intra-List Diversity (ILD) based on cosine distance between recommended items.

    ILD@K = (1 / |U|) * sum_{u} [ (2 / (K*(K-1))) * sum_{i < j} (1 - cos(e_i, e_j)) ]

    Args:
        topk_predictions: Tensor (num_eval_users, max_k) of recommended item IDs
        item_embeddings: Tensor (num_items, dim) of learned item representations
        k: Top-K cutoff
        sample_users: Max users to sample for fast computation

    Returns:
        float: Mean intra-list diversity score in [0, 2] (higher is more diverse)
    """
    preds_k = topk_predictions[:, :k]
    num_users = preds_k.size(0)

    if num_users > sample_users:
        perm = torch.randperm(num_users, generator=torch.Generator().manual_seed(42))[:sample_users]
        preds_k = preds_k[perm]
        num_users = sample_users

    # Normalize item embeddings for fast cosine similarity
    norm_item_embeds = F.normalize(item_embeddings.to(preds_k.device), dim=-1)

    # Gather embeddings: shape (num_users, k, dim)
    rec_embeds = norm_item_embeds[preds_k]

    # Compute batch pairwise cosine similarity: shape (num_users, k, k)
    sim_matrix = torch.bmm(rec_embeds, rec_embeds.transpose(1, 2))

    # Mask upper triangle (i < j)
    k_pairs = k * (k - 1) / 2.0
    if k_pairs <= 0:
        return 0.0

    triu_indices = torch.triu_indices(k, k, offset=1)
    pairwise_sims = sim_matrix[:, triu_indices[0], triu_indices[1]] # (num_users, k_pairs)
    pairwise_dists = 1.0 - pairwise_sims # Cosine distance

    if item_mask is not None:
        valid = item_mask.to(preds_k.device)[preds_k]
        pairs = valid[:, triu_indices[0]] & valid[:, triu_indices[1]]
        counts = pairs.sum(dim=1)
        usable = counts > 0
        if not usable.any():
            return float("nan")
        user_ild = (pairwise_dists * pairs).sum(dim=1)[usable] / counts[usable]
    else:
        user_ild = pairwise_dists.mean(dim=1)
    return float(user_ild.mean().item())


def compute_novelty(
    topk_predictions: torch.Tensor,
    item_popularity_dict: Dict[int, int],
    num_total_users: int,
    k: int = 10,
) -> float:
    """Compute Recommendation Novelty / Serendipity based on Self-Information.

    Novelty@K = (1 / (|U| * K)) * sum_{u} sum_{i in R_u} -log2( P(i) )
    where P(i) is the fraction of reference users who interacted with item i.
    An unseen item receives a conservative pseudo-count of one.

    Args:
        topk_predictions: Tensor (num_eval_users, max_k) of recommended item IDs
        item_popularity_dict: Dictionary mapping item ID to interaction count in training set
        num_total_users: Total number of users
        k: Top-K cutoff

    Returns:
        float: Mean novelty score in bits (higher indicates more novel / tail items)
    """
    if num_total_users <= 0:
        raise ValueError("num_total_users must be positive")

    preds_k = topk_predictions[:, :k].cpu().numpy()
    num_users = preds_k.shape[0]
    if num_users == 0:
        return 0.0

    user_novelties = []
    for u in range(num_users):
        u_preds = preds_k[u]
        self_info = []
        for item in u_preds:
            count = max(1, int(item_popularity_dict.get(int(item), 0)))
            if count > num_total_users:
                raise ValueError("Item popularity cannot exceed the reference user count")
            self_info.append(-np.log2(count / float(num_total_users)))
        user_novelties.append(np.mean(self_info))

    return float(np.mean(user_novelties))


def compute_coverage_and_gini(
    topk_predictions: torch.Tensor,
    num_total_items: int,
    k: int = 10,
    catalog_items: Optional[Set[int]] = None,
) -> Tuple[float, float]:
    """Compute Catalog Coverage and Gini Index of recommendation distribution.

    - Coverage@K = |Unique items recommended| / num_total_items
    - Gini@K = Concentration inequality of recommendations (0 = perfect equality, 1 = extreme bias)

    Returns:
        Tuple[coverage, gini_index]
    """
    preds_k = topk_predictions[:, :k].cpu().numpy().flatten()
    unique_items, counts = np.unique(preds_k, return_counts=True)

    if catalog_items is not None:
        catalog = sorted(set(catalog_items))
        if not set(unique_items).issubset(catalog_items):
            raise ValueError("Recommendations contain items outside the eligible catalog")
        count_by_item = dict(zip(unique_items.tolist(), counts.tolist()))
        full_counts = np.array(
            [count_by_item.get(item, 0) for item in catalog], dtype=np.float64
        )
        n = len(catalog)
    else:
        if np.any(unique_items < 0) or np.any(unique_items >= num_total_items):
            raise ValueError("Recommendation item index is outside the catalog")
        full_counts = np.zeros(num_total_items, dtype=np.float64)
        full_counts[unique_items] = counts
        n = num_total_items

    # Coverage
    coverage = float(len(unique_items) / float(n)) if n > 0 else 0.0

    # Gini Coefficient across the same eligible catalog as coverage.
    sorted_counts = np.sort(full_counts)

    index = np.arange(1, n + 1)
    sum_counts = np.sum(sorted_counts)
    if sum_counts == 0:
        gini = 0.0
    else:
        gini = float((2.0 * np.sum(index * sorted_counts) - (n + 1) * sum_counts) / (n * sum_counts))

    return coverage, gini
