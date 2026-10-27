"""Train-only popularity baseline with deterministic item-ID tie breaking."""
import torch


def popularity_predictions(train_df, evaluator):
    counts = train_df.groupby("i_idx")["u_idx"].nunique().to_dict()
    ordered = sorted(evaluator.candidate_items, key=lambda item: (-counts.get(item, 0), item))
    cutoff = max(evaluator.k_list)
    rows = []
    for user in evaluator.eval_users:
        seen = evaluator.train_history.get(user, set())
        row = []
        for item in ordered:
            if item not in seen:
                row.append(item)
                if len(row) == cutoff:
                    break
        if len(row) < cutoff:
            raise ValueError(f"User {user} lacks {cutoff} eligible candidates")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long).reshape(len(rows), cutoff)
