"""Dataset diagnostics that never rewrite interaction splits."""

import pandas as pd

from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


def warm_cohort(train, evaluation):
    users, items = set(train.u_idx), set(train.i_idx)
    warm_item = evaluation.i_idx.isin(items)
    warm_user = evaluation.u_idx.isin(users)
    warm = evaluation.loc[warm_item & warm_user].copy()
    return warm, {
        "total_targets": len(evaluation), "warm_targets": len(warm),
        "cold_item_targets": int((~warm_item).sum()),
        "cold_item_rate": float((~warm_item).mean()) if len(evaluation) else 0.0,
        "unique_cold_items": int(evaluation.loc[~warm_item, "i_idx"].nunique()),
        "users_without_train_history": int(evaluation.loc[~warm_user, "u_idx"].nunique()),
        "evaluable_users": int(warm.u_idx.nunique()),
        "user_coverage": warm.u_idx.nunique() / max(1, len(users)),
    }


def quality_report(train, val, test, item_metadata):
    frames = [train, val, test]
    for frame in frames:
        if frame is None or frame[["u_idx", "i_idx", "timestamp"]].isna().any().any():
            raise ValueError("Invalid interaction frame")
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows.duplicated(["u_idx", "i_idx"]).any():
        raise ValueError("Duplicate positive edges within/across splits")
    train_users, val_users, test_users = (set(f.u_idx) for f in frames)
    _, val_report = warm_cohort(train, val)
    _, test_report = warm_cohort(train, test)
    ties = all_rows.groupby(["u_idx", "timestamp"]).size()
    ties = ties[ties > 1]
    def degree(column):
        values = all_rows.groupby(column).size()
        return {"mean": float(values.mean()), "median": float(values.median()),
                "p25": float(values.quantile(.25)), "p75": float(values.quantile(.75)),
                "p90": float(values.quantile(.9)), "max": int(values.max())}
    n_users, n_items = all_rows.u_idx.nunique(), all_rows.i_idx.nunique()
    if len(item_metadata) != n_items or len({m["original_id"] for m in item_metadata.values()}) != n_items:
        raise ValueError("Metadata item count or ASIN uniqueness mismatch")
    return {
        "dataset": {"users": n_users, "items": n_items, "interactions": len(all_rows)},
        "split": {"counts": dict(zip(("train", "val", "test"), map(len, frames))),
                  "train_users": len(train_users), "val_users": len(val_users), "test_users": len(test_users),
                  "train_only_users": len(train_users - val_users - test_users),
                  "train_and_val_users": len(train_users & val_users),
                  "train_and_test_users": len(train_users & test_users),
                  "all_splits_users": len(train_users & val_users & test_users),
                  "coverage_cause": "seeded subset of users gets last two targets to achieve exact global ratios; not a global time cutoff"},
        "cold_start": {"val": val_report, "test": test_report,
                       "unique_cold_items_union": len((set(val.i_idx) | set(test.i_idx)) - set(train.i_idx))},
        "timestamp": {"midnight_fraction": float((all_rows.timestamp % 86400 == 0).mean()),
                      "tie_users": int(ties.index.get_level_values(0).nunique()),
                      "tie_interactions": int(ties.sum()), "max_tie_group": int(ties.max()) if len(ties) else 0,
                      "median_tie_group": float(ties.median()) if len(ties) else 0,
                      "tie_policy": "existing seeded_user_item_hash_then_item_id preserved for benchmark compatibility; not intraday chronology"},
        "sparsity": {"density": len(all_rows) / (n_users * n_items),
                     "sparsity": 1 - len(all_rows) / (n_users * n_items),
                     "user_degree": degree("u_idx"), "item_degree": degree("i_idx")},
    }
