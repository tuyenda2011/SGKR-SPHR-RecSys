import time
from collections.abc import Callable
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_coverage_and_gini,
    compute_intra_list_diversity,
    compute_novelty,
    compute_topk_metrics,
)

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)

EVALUATION_PROTOCOL = "profile_monitor_scoring_v5"


class Evaluator:
    """Full-ranking evaluator for Top-K recommendation with training history masking and beyond-accuracy metrics."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        k_list: List[int] = [10, 20],
        batch_size: int = 1024,
        candidate_items: Optional[Set[int]] = None,
        popularity_df: Optional[pd.DataFrame] = None,
        diversity_features: Optional[torch.Tensor] = None,
        diversity_mask: Optional[torch.Tensor] = None,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] | None = None,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.k_list = k_list
        self.batch_size = batch_size
        # score_fn(user_indices, all_users, all_items); None preserves dot product.
        self.score_fn = score_fn
        self.diversity_features = None
        self.diversity_mask = None
        if diversity_features is not None:
            if diversity_features.ndim != 2 or diversity_features.shape[0] != num_items:
                raise ValueError("Diversity features must have one row per item")
            if diversity_mask is None or diversity_mask.dtype != torch.bool or diversity_mask.shape != (num_items,):
                raise ValueError("Diversity requires a boolean usable-text mask")
            if not torch.isfinite(diversity_features).all():
                raise ValueError("Diversity features must be finite")
            if (diversity_features[diversity_mask].norm(dim=1) == 0).any():
                raise ValueError("Usable diversity features must be nonzero")
            self.diversity_features = diversity_features.detach().cpu().clone()
            self.diversity_mask = diversity_mask.detach().cpu().clone()

        # Build training history set per user for masking
        self.train_history: Dict[int, Set[int]] = (
            train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        )

        # Precompute item popularity for Novelty metric
        popularity_source = popularity_df if popularity_df is not None else train_df
        self.popularity_num_users = int(popularity_source["u_idx"].nunique())
        self.item_popularity: Dict[int, int] = (
            popularity_source.groupby("i_idx")["u_idx"].nunique().to_dict()
        )

        # Build ground truth target list for eval set users
        eval_grouped = eval_df.groupby("u_idx")["i_idx"].apply(list).to_dict()
        self.eval_users = sorted(list(eval_grouped.keys()))
        self.ground_truth = [eval_grouped[u] for u in self.eval_users]
        if candidate_items is None:
            self.candidate_items = set(range(num_items))
            self.excluded_candidates: List[int] = []
        else:
            self.candidate_items = set(candidate_items)
            self.excluded_candidates = sorted(
                set(range(num_items)) - self.candidate_items
            )

    # Constant for masking seen items
    MASK_VALUE: float = float("-inf")

    def _masks(self, device: torch.device):
        """Excluded-candidate index and per-batch (row, item) history indices, built once per device.

        Callers may reassign or edit users/candidates/batch size, so the cache is tied to a snapshot of them.
        """
        source = (self.batch_size, list(self.eval_users), list(self.excluded_candidates), id(self.train_history))
        if getattr(self, "_mask_source", None) != source:
            self._mask_source, self._mask_cache, self._eligible_max_k = source, {}, 0
        key = str(device)
        cache = self._mask_cache
        if key not in cache:
            excluded = (torch.tensor(self.excluded_candidates, dtype=torch.long, device=device)
                        if self.excluded_candidates else None)
            batches = []
            for start in range(0, len(self.eval_users), self.batch_size):
                seen_rows: list[int] = []
                seen_items: list[int] = []
                for row, u in enumerate(self.eval_users[start:start + self.batch_size]):
                    seen = self.train_history.get(u)
                    if seen:
                        seen_rows.extend([row] * len(seen))
                        seen_items.extend(seen)
                batches.append((torch.tensor(seen_rows, dtype=torch.long, device=device),
                                torch.tensor(seen_items, dtype=torch.long, device=device)))
            cache[key] = (excluded, batches)
        return cache[key]

    @torch.no_grad()
    def get_predictions(
        self,
        final_user_embeds: torch.Tensor,
        final_item_embeds: torch.Tensor,
        device: torch.device,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Compute top-K predictions, applying history/candidate masks after scoring.

        Supply the model's get_user_rating_scores as score_fn to use its scoring
        semantics; embedding-only callers default to dot product.
        """
        if not self.k_list or any(k <= 0 for k in self.k_list):
            raise ValueError("k_list must contain positive cutoffs")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        max_k = max(self.k_list)
        excluded_tensor, history_masks = self._masks(device)
        # Checked once per cutoff while users, histories and candidates stay unchanged (see _masks).
        if max_k > self._eligible_max_k:
            candidates = set(range(self.num_items)) - set(self.excluded_candidates)
            for user in self.eval_users:
                available = len(candidates) - len(candidates.intersection(self.train_history.get(user, set())))
                if available < max_k:
                    raise ValueError(
                        f"User {user} has {available} eligible items; top-K requires {max_k}"
                    )
            self._eligible_max_k = max_k
        all_topk_preds = []

        start_time = time.perf_counter()
        num_eval_users = len(self.eval_users)

        batch_range = range(0, num_eval_users, self.batch_size)
        if show_progress:
            batch_range = tqdm(
                batch_range,
                desc="Top-K Inference",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

        for i in batch_range:
            batch_u_idx = self.eval_users[i : i + self.batch_size]
            u_tensors = torch.tensor(batch_u_idx, dtype=torch.long, device=device)

            # Compute rating scores matrix (batch_users, num_items)
            if self.score_fn is None:
                scores = torch.matmul(final_user_embeds[u_tensors], final_item_embeds.T)
            else:
                scores = self.score_fn(u_tensors, final_user_embeds, final_item_embeds)

            if excluded_tensor is not None:
                scores[:, excluded_tensor] = self.MASK_VALUE

            # Mask each user's own history. A batch-wide union would hide valid
            # targets whenever another user happened to have seen the same item.
            row_tensor, item_tensor = history_masks[i // self.batch_size]
            if row_tensor.numel():
                scores[row_tensor, item_tensor] = self.MASK_VALUE

            # Retrieve top-K recommended item IDs
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            all_topk_preds.append(topk_indices.cpu())

        if device.type == "cuda":
            torch.cuda.synchronize()

        if show_progress:
            import sys
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

        total_inference_time = time.perf_counter() - start_time
        avg_user_latency_ms = (total_inference_time / max(1, num_eval_users)) * 1000.0

        topk_preds_tensor = torch.cat(all_topk_preds, dim=0) if all_topk_preds else torch.empty(0, max_k, dtype=torch.long)
        return topk_preds_tensor, avg_user_latency_ms


    @torch.no_grad()
    def evaluate(
        self,
        final_user_embeds: torch.Tensor,
        final_item_embeds: torch.Tensor,
        device: torch.device,
        include_beyond_accuracy: bool = False,
        show_progress: bool = False,
    ) -> Tuple[Dict[str, float], float]:
        """Perform full-ranking evaluation across all target users.

        Returns:
            metrics_dict: Dictionary of evaluated metrics
            avg_user_latency_ms: Average latency per user in milliseconds
        """
        topk_preds_tensor, avg_user_latency_ms = self.get_predictions(
            final_user_embeds, final_item_embeds, device, show_progress=show_progress
        )

        metrics = compute_topk_metrics(self.ground_truth, topk_preds_tensor, self.k_list)

        if include_beyond_accuracy and topk_preds_tensor.size(0) > 0:
            for k in self.k_list:
                # Shared content space only; never silently use model embeddings.
                metrics[f"Diversity@{k}"] = float("nan")
                if self.diversity_features is not None and self.diversity_mask is not None:
                    metrics[f"Diversity@{k}"] = compute_intra_list_diversity(
                        topk_preds_tensor, self.diversity_features, k=k,
                        item_mask=self.diversity_mask,
                    )
                    metrics[f"DiversityValidTextFraction@{k}"] = float(
                        self.diversity_mask[topk_preds_tensor[:, :k]].float().mean()
                    )

                # Novelty (Self-Information)
                novelty = compute_novelty(
                    topk_preds_tensor,
                    self.item_popularity,
                    self.popularity_num_users,
                    k=k,
                )
                metrics[f"Novelty@{k}"] = novelty

                # Catalog Coverage & Gini Coefficient
                cov, gini = compute_coverage_and_gini(
                    topk_preds_tensor,
                    self.num_items,
                    k=k,
                    catalog_items=self.candidate_items,
                )
                metrics[f"Coverage@{k}"] = cov
                metrics[f"Gini@{k}"] = gini

        return metrics, avg_user_latency_ms
