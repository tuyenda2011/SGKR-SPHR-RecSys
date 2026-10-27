import logging

import numpy as np
import pandas as pd

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)

logger = logging.getLogger(__name__)


def create_sparse_train_set(
    train_df: pd.DataFrame,
    sparsity_ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample an exact fraction of training edges while preserving graph coverage.

    Validation and Test sets MUST remain 100% unchanged.

    Args:
        train_df: Training DataFrame with columns [u_idx, i_idx, timestamp?]
        sparsity_ratio: Global ratio of unique edges to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Sparse training DataFrame
    """
    if not 0 < sparsity_ratio <= 1:
        raise ValueError("sparsity_ratio must be between 0 and 1")

    required_columns = {"u_idx", "i_idx"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(f"train_df is missing required columns: {sorted(missing_columns)}")

    base = train_df.drop_duplicates(["u_idx", "i_idx"]).reset_index(drop=True)
    if sparsity_ratio == 1.0 or base.empty:
        return base.copy()

    logger.info(
        f"Applying coverage-preserving sparsity sampling at ratio={sparsity_ratio} "
        f"(seed={seed})..."
    )
    has_timestamp = "timestamp" in train_df.columns
    rng = np.random.default_rng(seed)
    ranked = base.copy()
    ranked["_edge_id"] = np.arange(len(ranked))
    ranked["_sample_order"] = rng.random(len(ranked))

    # Build a deterministic graph-covering core: one edge per item, then one
    # edge for each user not already represented by that item cover.
    random_order = ranked.sort_values("_sample_order", kind="mergesort")
    item_cover = random_order.drop_duplicates("i_idx", keep="first")
    covered_users = set(item_cover["u_idx"])
    user_cover = random_order.loc[
        ~random_order["u_idx"].isin(covered_users)
    ].drop_duplicates("u_idx", keep="first")
    coverage_ids = set(item_cover["_edge_id"]) | set(user_cover["_edge_id"])

    target_edges = int(round(len(base) * sparsity_ratio))
    target_edges = max(1, target_edges)
    retained_edges = max(target_edges, len(coverage_ids))
    if retained_edges > target_edges:
        logger.warning(
            "Requested ratio %.2f%% is below the %d-edge graph coverage floor; "
            "retaining %.2f%% instead.",
            sparsity_ratio * 100,
            len(coverage_ids),
            retained_edges / len(base) * 100,
        )

    additional_needed = retained_edges - len(coverage_ids)
    additional_ids = random_order.loc[
        ~random_order["_edge_id"].isin(coverage_ids), "_edge_id"
    ].head(additional_needed)
    selected_ids = coverage_ids | set(additional_ids)
    sampled = ranked.loc[ranked["_edge_id"].isin(selected_ids)].drop(
        columns=["_edge_id", "_sample_order"]
    )

    sort_cols = ["u_idx"] + (["timestamp"] if has_timestamp else [])
    sparse_train_df = (
        sampled.sort_values(by=sort_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    logger.info(
        f"Sparsity ratio {sparsity_ratio}: original edges={len(train_df)} -> "
        f"retained edges={len(sparse_train_df)} ({len(sparse_train_df)/len(base):.2%}); "
        f"coverage core={len(coverage_ids)} edges"
    )

    return sparse_train_df
