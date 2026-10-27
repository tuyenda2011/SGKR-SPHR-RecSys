import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def extract_explicit_negative_interactions(
    ratings_df: pd.DataFrame,
    user2id: dict[str, int],
    item2id: dict[str, int],
    negative_threshold: float = 2.0,
    save_path: str | None = None,
    user_train_cutoffs: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Extract explicit negative interactions (rating <= negative_threshold) for mapped users and items.

    Args:
        ratings_df: Raw ratings DataFrame with columns [user_id/reviewerID, item_id/asin, rating/overall, timestamp/unixReviewTime].
        user2id: Map of valid user strings to contiguous integer indices.
        item2id: Map of valid item strings to contiguous integer indices.
        negative_threshold: Ratings below or equal to this are considered explicit dislikes (e.g., 1.0, 2.0).
        save_path: Optional file path to save the disliked interaction table.

    Returns:
        Tuple of (disliked_df with [u_idx, i_idx, rating, timestamp], user_disliked_dict {u_idx: [i_idx, ...]}).
    """
    logger.info(f"Extracting explicit negative interactions (rating <= {negative_threshold})...")

    col_map = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp",
    }
    df = ratings_df.rename(columns={k: v for k, v in col_map.items() if k in ratings_df.columns})

    # Filter explicit negatives
    neg_df = df[df["rating"] <= negative_threshold].copy()
    logger.info(f"Found {len(neg_df):,} raw negative interactions in entire dataset.")

    # Filter to only users and items that survived the 5-core bipartite graph filtering
    neg_df["u_idx"] = neg_df["user_id"].map(user2id)
    neg_df["i_idx"] = neg_df["item_id"].map(item2id)
    neg_df = neg_df.dropna(subset=["u_idx", "i_idx"]).copy()
    neg_df["u_idx"] = neg_df["u_idx"].astype(int)
    neg_df["i_idx"] = neg_df["i_idx"].astype(int)

    if user_train_cutoffs is not None:
        cutoffs = neg_df["u_idx"].map(user_train_cutoffs)
        neg_df = neg_df[cutoffs.notna() & (neg_df["timestamp"] <= cutoffs)].copy()
        logger.info(
            f"Retained {len(neg_df):,} negatives available before each user's training cutoff."
        )

    # De-duplicate: Keep unique (u_idx, i_idx)
    neg_df = neg_df.sort_values(by=["rating", "timestamp"], ascending=[True, False])
    neg_df = neg_df.drop_duplicates(subset=["u_idx", "i_idx"], keep="first")
    logger.info(f"Retained {len(neg_df):,} unique explicit negative interactions for mapped 5-core graph.")

    # Build user-to-disliked-items map
    user_disliked: dict[int, list[int]] = {}
    for u_idx, group in neg_df.groupby("u_idx"):
        user_disliked[int(u_idx)] = group["i_idx"].tolist()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        saved = neg_df[["u_idx", "i_idx", "rating", "timestamp"]].copy()
        saved.attrs.clear()
        saved.to_parquet(save_path, index=False)
        logger.info(f"Saved disliked interactions to {save_path}")

    return neg_df, user_disliked
