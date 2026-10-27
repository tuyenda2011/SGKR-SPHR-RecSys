import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_TIE_POLICY = "seeded_user_item_hash_then_item_id"


def summarize_split_timing(train_df, val_df, test_df) -> dict:
    """Report weak time ordering and ties; ties do not establish event order."""
    train_max = train_df.groupby("u_idx")["timestamp"].max()
    val_min = val_df.groupby("u_idx")["timestamp"].min()
    val_max = val_df.groupby("u_idx")["timestamp"].max()
    test_min = test_df.groupby("u_idx")["timestamp"].min()
    shared = val_max.index.intersection(test_min.index)
    train_val_ties = int((train_max.reindex(val_min.index) == val_min).sum())
    val_test_ties = int((val_max.loc[shared] == test_min.loc[shared]).sum())
    all_timestamps = pd.concat([part["timestamp"] for part in (train_df, val_df, test_df)])
    return {
        "timestamp_midnight_fraction": float((all_timestamps % 86400 == 0).mean()),
        "train_validation_tied_users": train_val_ties,
        "validation_test_tied_users": val_test_ties,
        "train_validation_order_violations": int((train_max.reindex(val_min.index) > val_min).sum()),
        "validation_test_order_violations": int((val_max.loc[shared] > test_min.loc[shared]).sum()),
        "train_test_order_violations": int((train_max.reindex(test_min.index) > test_min).sum()),
        "strict_temporal_order": train_val_ties == 0 and val_test_ties == 0
            and not (train_max.reindex(val_min.index) >= val_min).any()
            and not (val_max.loc[shared] >= test_min.loc[shared]).any(),
    }


def check_split_connectivity(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Check whether all users and items present in val/test exist in the train graph."""
    train_users = set(train_df["u_idx"].unique())
    train_items = set(train_df["i_idx"].unique())

    val_users = set(val_df["u_idx"].unique())
    val_items = set(val_df["i_idx"].unique())

    test_users = set(test_df["u_idx"].unique())
    test_items = set(test_df["i_idx"].unique())

    eval_users = val_users.union(test_users)
    eval_items = val_items.union(test_items)

    missing_users = eval_users - train_users
    missing_items = eval_items - train_items

    report = {
        "num_train_users": len(train_users),
        "num_train_items": len(train_items),
        "num_eval_users": len(eval_users),
        "num_eval_items": len(eval_items),
        "missing_users_count": len(missing_users),
        "missing_items_count": len(missing_items),
        "missing_users": sorted(list(missing_users)),
        "missing_items": sorted(list(missing_items)),
        "is_connected": len(missing_users) == 0 and len(missing_items) == 0,
    }

    if not report["is_connected"]:
        logger.warning(
            f"Graph split connectivity violation: {len(missing_users)} users and "
            f"{len(missing_items)} items present in eval splits are missing from train graph."
        )
    else:
        logger.info("Graph split connectivity check PASSED: 100% of eval nodes exist in train graph.")

    return report


def relocate_disconnected(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Move one interaction per isolated item/user from val or test into train to guarantee connectivity."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 1. Fix missing items
    train_items = set(train_df["i_idx"].unique())
    all_eval_items = set(val_df["i_idx"].unique()).union(set(test_df["i_idx"].unique()))
    missing_items = all_eval_items - train_items

    if missing_items:
        logger.info(f"Relocating interactions for {len(missing_items)} disconnected items to train...")
        val_trans_idx = []
        test_trans_idx = []
        for item in missing_items:
            v_matches = val_df[val_df["i_idx"] == item]
            if len(v_matches) > 0:
                val_trans_idx.append(v_matches.index[0])
            else:
                t_matches = test_df[test_df["i_idx"] == item]
                if len(t_matches) > 0:
                    test_trans_idx.append(t_matches.index[0])

        if val_trans_idx:
            transfer_val = val_df.loc[val_trans_idx]
            train_df = pd.concat([train_df, transfer_val], ignore_index=True)
            val_df = val_df.drop(index=val_trans_idx).reset_index(drop=True)

        if test_trans_idx:
            transfer_test = test_df.loc[test_trans_idx]
            train_df = pd.concat([train_df, transfer_test], ignore_index=True)
            test_df = test_df.drop(index=test_trans_idx).reset_index(drop=True)

    # 2. Fix missing users
    train_users = set(train_df["u_idx"].unique())
    all_eval_users = set(val_df["u_idx"].unique()).union(set(test_df["u_idx"].unique()))
    missing_users = all_eval_users - train_users

    if missing_users:
        logger.info(f"Relocating interactions for {len(missing_users)} disconnected users to train...")
        val_trans_u = []
        test_trans_u = []
        for user in missing_users:
            v_matches = val_df[val_df["u_idx"] == user]
            if len(v_matches) > 0:
                val_trans_u.append(v_matches.index[0])
            else:
                t_matches = test_df[test_df["u_idx"] == user]
                if len(t_matches) > 0:
                    test_trans_u.append(t_matches.index[0])

        if val_trans_u:
            transfer_val_u = val_df.loc[val_trans_u]
            train_df = pd.concat([train_df, transfer_val_u], ignore_index=True)
            val_df = val_df.drop(index=val_trans_u).reset_index(drop=True)

        if test_trans_u:
            transfer_test_u = test_df.loc[test_trans_u]
            train_df = pd.concat([train_df, transfer_test_u], ignore_index=True)
            test_df = test_df.drop(index=test_trans_u).reset_index(drop=True)

    return train_df, val_df, test_df


def chronological_per_user_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    enforce_connectivity: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an exact-ratio chronological split for the course benchmark.

    Validation and test receive one latest interaction each from a deterministic
    subset of users. Remaining users are train-only, which is necessary when the
    requested holdout size is smaller than the total number of users.
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be between 0 and 1")
    if not np.isclose(val_ratio, test_ratio):
        raise ValueError("This split protocol requires equal validation and test ratios")
    if val_ratio + test_ratio >= 1:
        raise ValueError("validation and test ratios must leave room for training data")

    logger.info(
        "Performing exact chronological split "
        f"(train={1-val_ratio-test_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}, seed={seed})..."
    )

    # Stable seeded tie-breaking is independent of raw file row order and
    # rating. It does not invent an intraday event order for day-level data.
    ranked = df.copy()
    tie_keys = ranked[["u_idx", "i_idx"]].copy()
    tie_keys["seed"] = seed
    ranked["_tie_break"] = pd.util.hash_pandas_object(tie_keys, index=False).to_numpy()
    df_sorted = ranked.sort_values(
        by=["u_idx", "timestamp", "_tie_break", "i_idx"], kind="mergesort"
    ).drop(columns="_tie_break").reset_index(drop=True)
    group_sizes = df_sorted.groupby("u_idx")["u_idx"].transform("size")
    eligible_users = np.sort(
        df_sorted.loc[group_sizes >= 3, "u_idx"].unique()
    )
    holdout_size = int(round(len(df_sorted) * val_ratio))
    if holdout_size > len(eligible_users):
        raise ValueError(
            f"Cannot allocate {holdout_size} validation/test rows from "
            f"only {len(eligible_users)} users with at least 3 interactions"
        )

    rng = np.random.default_rng(seed)
    eval_users = set(
        rng.choice(eligible_users, size=holdout_size, replace=False).tolist()
    )
    positions = df_sorted.groupby("u_idx").cumcount()
    selected = df_sorted["u_idx"].isin(eval_users)
    test_mask = selected & positions.eq(group_sizes - 1)
    val_mask = selected & positions.eq(group_sizes - 2)
    train_mask = ~(val_mask | test_mask)

    train_df = df_sorted.loc[train_mask].reset_index(drop=True)
    val_df = df_sorted.loc[val_mask].reset_index(drop=True)
    test_df = df_sorted.loc[test_mask].reset_index(drop=True)

    if enforce_connectivity:
        logger.warning(
            "enforce_connectivity is ignored to preserve chronological ordering; "
            "cold-start items must be handled by the evaluator"
        )

    check_split_connectivity(train_df, val_df, test_df)
    verify_no_leakage(train_df, val_df, test_df)

    train_max = train_df.groupby("u_idx")["timestamp"].max()
    val_min = val_df.groupby("u_idx")["timestamp"].min()
    test_min = test_df.groupby("u_idx")["timestamp"].min()
    if any(train_max.loc[val_min.index] > val_min):
        raise AssertionError("Temporal leakage detected between train and validation")
    if any(train_max.loc[test_min.index] > test_min):
        raise AssertionError("Temporal leakage detected between train and test")
    timing = summarize_split_timing(train_df, val_df, test_df)
    if timing["validation_test_order_violations"]:
        raise AssertionError("Temporal leakage detected between validation and test")
    if not timing["strict_temporal_order"]:
        logger.warning("Timestamp ties cross split boundaries; report weak chronology, not strict future prediction: %s", timing)

    logger.info(
        f"Split completed: Train={len(train_df)} ({len(train_df)/len(df_sorted):.2%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df_sorted):.2%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df_sorted):.2%}), "
        f"Eval users={len(eval_users)}."
    )

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raise_on_error: bool = True,
) -> Tuple[bool, Dict[str, int]]:
    """Verify that train, validation, and test edge sets have zero intersection.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        raise_on_error: Whether to raise assertion error on leakage (default: True)

    Returns:
        Tuple of (is_valid, leakage_counts) where leakage_counts shows number of overlapping edges
    """
    train_pairs: Set[Tuple[int, int]] = set(zip(train_df["u_idx"], train_df["i_idx"]))
    val_pairs: Set[Tuple[int, int]] = set(zip(val_df["u_idx"], val_df["i_idx"]))
    test_pairs: Set[Tuple[int, int]] = set(zip(test_df["u_idx"], test_df["i_idx"]))

    intersection_train_val = len(train_pairs.intersection(val_pairs))
    intersection_train_test = len(train_pairs.intersection(test_pairs))
    intersection_val_test = len(val_pairs.intersection(test_pairs))

    leakage_counts = {
        "train_val": intersection_train_val,
        "train_test": intersection_train_test,
        "val_test": intersection_val_test,
    }

    is_valid = all(count == 0 for count in leakage_counts.values())

    if not is_valid:
        msg = (
            f"DATA LEAKAGE DETECTED! "
            f"Train∩Val={intersection_train_val}, "
            f"Train∩Test={intersection_train_test}, "
            f"Val∩Test={intersection_val_test}"
        )
        if raise_on_error:
            raise AssertionError(msg)
        else:
            logger.error(msg)
    else:
        logger.info("PASSED DATA LEAKAGE CHECK: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.")

    return is_valid, leakage_counts


def global_temporal_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    enforce_connectivity: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform global chronological cutoff split based on global timestamps.

    Oldest (1 - val_ratio - test_ratio) -> train, next val_ratio -> val, latest test_ratio -> test.
    """
    logger.info(f"Performing global chronological cutoff split (val={val_ratio}, test={test_ratio})...")
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)

    n_total = len(df_sorted)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    if enforce_connectivity:
        train_df, val_df, test_df = relocate_disconnected(train_df, val_df, test_df)

    check_split_connectivity(train_df, val_df, test_df)
    verify_no_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df
