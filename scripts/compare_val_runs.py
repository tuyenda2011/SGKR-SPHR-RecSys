"""Paired, validation-only comparison of saved top-20 runs (B minus reference A)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy import sparse

from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR, SGKRConfig

METRICS = ("NDCG@20", "Recall@20", "NDCG@10")
SPLIT_SALT = "sgkr-sphr-split-20261024"


def half_mask(users, half="all"):
    if half not in {"all", "selection", "confirmation"}:
        raise ValueError("Unknown validation half")
    selection = np.fromiter((hashlib.sha256(f"{SPLIT_SALT}:{int(user)}".encode()).digest()[0] % 2 == 0
                             for user in users), dtype=bool, count=len(users))
    return np.ones(len(users), dtype=bool) if half == "all" else selection if half == "selection" else ~selection


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected finite p values in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order]*np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def per_user_metrics(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if (predictions.ndim != 2 or predictions.shape != (len(targets), 20)
            or predictions.dtype != np.dtype("int64") or targets.ndim != 1 or not len(targets)):
        raise ValueError("Expected nonempty int64 predictions [users,20] and one target per user")
    if np.any(np.diff(np.sort(predictions, axis=1), axis=1) == 0):
        raise ValueError("Duplicate recommendations")
    hits = predictions == targets[:, None]
    discount = 1 / np.log2(np.arange(2, 22))
    # Preserve the shared metric's float32 Recall reduction; bootstrap converts
    # the exact per-user 0/1 values to float64 before computing paired deltas.
    result = {"NDCG@20": hits @ discount, "Recall@20": hits.sum(1).astype(np.float32),
              "NDCG@10": hits[:, :10] @ discount[:10]}
    reference = compute_topk_metrics(targets[:, None].tolist(), torch.from_numpy(predictions.copy()), [10, 20])
    for metric, values in result.items():
        if abs(float(values.mean()) - reference[metric]) > 1e-12:
            raise AssertionError(f"Per-user metric disagrees with compute_topk_metrics: {metric}")
    return result


def paired_bootstrap(a, b, *, repeats=10_000, seed=42):
    """Percentile CI and centered-null, right-tailed bootstrap p for H1: mean(B-A)>0.

    Multinomial counts over distinct paired differences have exactly the same
    sampling distribution as drawing users with replacement. This avoids a
    repeats-by-users allocation, retaining all zero differences and their mass.
    p=(1+#(bootstrap_delta-observed_delta >= observed_delta))/(repeats+1).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or repeats < 1 or seed < 0):
        raise ValueError("Invalid paired bootstrap inputs")
    if not len(a):
        return {"n_users": 0, "mean_a": None, "mean_b": None, "delta": None, "ci95": None, "p_one_sided": None}
    differences = b - a
    observed = float(differences.mean())
    values, counts = np.unique(differences, return_counts=True)
    rng = np.random.default_rng(seed)
    distribution = np.empty(repeats)
    for start in range(0, repeats, 256):
        stop = min(start + 256, repeats)
        sampled_counts = rng.multinomial(len(a), counts / len(a), size=stop - start)
        distribution[start:stop] = sampled_counts @ values / len(a)
    return {"n_users": len(a), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": observed,
            "ci95": np.quantile(distribution, [.025, .975]).tolist(),
            "p_one_sided": float((1 + np.count_nonzero(distribution - observed >= observed)) / (repeats + 1))}


def validate_run_arrays(run, expected_users, train, full_train, num_items):
    users = np.load(run / "val_users.npy", allow_pickle=False)
    predictions = np.load(run / "val_predictions.npy", allow_pickle=False)
    if users.dtype != np.dtype("int64") or not np.array_equal(users, expected_users):
        raise ValueError(f"Run users differ in cohort or order: {run}")
    if predictions.shape != (len(users), 20) or predictions.dtype != np.dtype("int64"):
        raise ValueError(f"Expected int64 [n_users,20] predictions: {run}")
    if predictions.size and (predictions.min() < 0 or predictions.max() >= num_items):
        raise ValueError(f"Out-of-catalog item ID: {run}")
    if not np.isin(predictions, train.i_idx.unique()).all():
        raise ValueError(f"Predictions include a cold candidate: {run}")
    histories = full_train.groupby("u_idx").i_idx.agg(set).to_dict()
    if any(histories.get(int(user), set()).intersection(row) for user, row in zip(users, predictions)):
        raise ValueError(f"Predictions include a masked train item: {run}")
    return predictions


def verify_run_protocol(run, identity):
    path = run / "plan.json"
    if not path.is_file():
        return  # Bare prediction arrays remain a supported input.
    plan = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in identity.items():
        if key in plan and plan[key] != expected:
            raise ValueError(f"Run has different {key}: {run}")


def target_reachability(run, users, targets, retained):
    """Reconstruct only neighborhoods from the checkpoint's train/config; no fit or scoring."""
    with np.load(run / "model.npz", allow_pickle=False) as state:
        if int(state["schema"]) not in (1, 2):
            raise ValueError("Reachability requires an SGKR schema-1 or schema-2 model.npz")
        config = SGKRConfig.from_dict(json.loads(str(state["config"])))
        shape = tuple(int(n) for n in state["shape"])
        x = sparse.csr_matrix((np.ones(len(state["train_indices"])), state["train_indices"],
                               state["train_indptr"]), shape=shape)
    if x.shape != retained.shape or (x != retained).nnz:
        raise ValueError(f"Checkpoint retained train differs from comparison protocol: {run}")
    model = SGKR(config)
    model._initialize(x)
    return np.asarray([np.intersect1d(x[int(user)].indices, model.neighborhoods[int(item)]).size > 0
                       for user, item in zip(users, targets)])


def time_groups(full_train, warm):
    latest = full_train.groupby("u_idx").timestamp.max().reindex(warm.u_idx).to_numpy()
    timestamps = warm.timestamp.to_numpy()
    if not np.isfinite(latest).all() or not np.isfinite(timestamps).all() or np.any(timestamps < latest):
        raise ValueError("Missing or nonchronological train/validation timestamps")
    same_day = timestamps == latest
    return {"same_day": same_day, "later": ~same_day}


def popularity_counts(train, num_items):
    return np.bincount(train.i_idx.to_numpy(dtype=np.int64), minlength=num_items)


def run_metrics(predictions, targets, metrics, mask, counts):
    """Top 20% uses ceil(active/5), ordered by count descending then item ID."""
    active = np.flatnonzero(counts > 0)
    head = active[np.lexsort((active, -counts[active]))[:int(np.ceil(len(active)/5))]]
    selected = predictions[mask]
    tail_mask = mask & ~np.isin(targets, head)
    result = {key: float(value[mask].mean()) if mask.any() else None for key, value in metrics.items()}
    result.update({"coverage@20": float(len(np.unique(selected))/len(active)) if mask.any() else None,
                   "mean_log_pop@20": float(np.log1p(counts[selected]).mean()) if mask.any() else None,
                   "ndcg20_longtail": float(metrics["NDCG@20"][tail_mask].mean()) if tail_mask.any() else None})
    return result


def pair_groups(degrees, reachable_a=None, reachable_b=None):
    groups = {"all": np.ones(len(degrees), dtype=bool),
            "degree_3_4": (degrees >= 3) & (degrees <= 4),
            "degree_5_9": (degrees >= 5) & (degrees <= 9), "degree_ge10": degrees >= 10}
    if np.any(degrees <= 2):
        groups["degree_1_2"] = (degrees >= 1) & (degrees <= 2)
    if reachable_a is not None and reachable_b is not None:
        groups.update({"reachable_both": reachable_a & reachable_b, "reachable_a_only": reachable_a & ~reachable_b,
                       "reachable_b_only": ~reachable_a & reachable_b, "reachable_neither": ~reachable_a & ~reachable_b})
    elif reachable_a is not None or reachable_b is not None:
        reachable = reachable_b if reachable_a is None else reachable_a
        groups.update({"reachable": reachable, "unreachable": ~reachable})
    return groups


def compare_metrics(a, b, groups, repeats, seed):
    return {name: {metric: paired_bootstrap(a[metric][mask], b[metric][mask], repeats=repeats, seed=seed)
                   for metric in METRICS} for name, mask in groups.items()}


def markdown(report):
    rows = ["| Run | NDCG@20 | Recall@20 | NDCG@10 |", "|---|---:|---:|---:|"]
    for name, result in report["runs"].items():
        rows.append(f"| {name} | " + " | ".join(
            "null" if result["metrics"][key] is None else f"{result['metrics'][key]:.8f}" for key in METRICS) + " |")
    rows.extend(["", "Hiệu = B trừ A; giả thuyết đối một phía B > A. Phân tích khám phá.", "",
                 "| B so với A | Nhóm | n | Metric | Hiệu | CI95 | p | Holm p |", "|---|---|---:|---|---:|---|---:|---:|"])
    for pair in report["comparisons"]:
        for group, metrics in pair["groups"].items():
            for metric, result in metrics.items():
                if result["n_users"]:
                    lo, hi = result["ci95"]
                    rows.append(f"| {pair['b']} vs {pair['a']} | {group} | {result['n_users']} | {metric} | "
                                f"{result['delta']:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {result['p_one_sided']:.5f} | "
                                + (f"{result['holm_p']:.5f}" if result.get("holm_p") is not None else "null") + " |")
    rows.extend(["", "| Run | coverage@20 | mean_log_pop@20 | ndcg20_longtail |", "|---|---:|---:|---:|"])
    for name, result in report["runs"].items():
        rows.append(f"| {name} | " + " | ".join(
            "null" if result["metrics"][key] is None else f"{result['metrics'][key]:.8f}"
            for key in ("coverage@20", "mean_log_pop@20", "ndcg20_longtail")) + " |")
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--sparsity", type=float, default=1.)
    parser.add_argument("--density-seed", type=int, default=42)
    parser.add_argument("--groups", action="store_true", help="Time, degree and available SGKR target reachability")
    parser.add_argument("--half", choices=("all", "selection", "confirmation"), default="all")
    parser.add_argument("--holm", action="store_true", help="Holm on overall NDCG@20 for all arms vs reference")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = [path.resolve() for path in args.runs]
    if len(runs) < 2 or len(set(runs)) != len(runs):
        raise ValueError("Provide at least two distinct run directories; first is reference A")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.sparsity <= 1 or args.density_seed < 0 or args.bootstrap < 1 or args.seed < 0:
        raise ValueError("Invalid comparison controls")
    started = time.perf_counter()
    bundle, full_train, validation, stats = load_validation_inputs(args.bundle, ROOT)
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    warm, cohort = warm_cohort(train, validation)
    if warm.empty or warm.u_idx.duplicated().any():
        raise ValueError("Comparison requires exactly one warm validation target per user")
    warm = warm.sort_values("u_idx")
    users, targets = (warm[key].to_numpy(dtype=np.int64) for key in ("u_idx", "i_idx"))
    selected = half_mask(users, args.half)
    identity = {"bundle_id": bundle.build_id, "density": args.sparsity, "density_seed": args.density_seed,
                "retained_sha256": edge_hash(train), "mask_sha256": edge_hash(full_train), "cohort_sha256": edge_hash(warm)}
    report = {"validation_only": True, "n_users": int(selected.sum()), "full_cohort_users": len(users),
              "half": args.half, "split_salt": SPLIT_SALT, "validation_cohort": cohort,
              "protocol": identity, "bootstrap_repeats": args.bootstrap, "bootstrap_seed": args.seed,
              "ci_method": "bootstrap percentile theo cặp user; nén đa thức chính xác trên phân phối hiệu",
              "p_method": "bootstrap đuôi phải dịch tâm dưới giả thuyết không, hiệu chỉnh Monte Carlo cộng một",
              "metric_reduction": "Recall giữ phép lấy trung bình float32 chung; NDCG và bootstrap dùng float64",
              "multiplicity": "khám phá, chưa hiệu chỉnh; kết luận cuối cần khóa giao thức",
              "runs": {}, "comparisons": []}
    arrays, reach, saved_predictions = [], [], []
    counts = popularity_counts(train, stats["num_items"])
    retained = None
    if args.groups:
        retained = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(),
                                           stats["num_users"], stats["num_items"])
    for index, run in enumerate(runs):
        verify_run_protocol(run, identity)
        predictions = validate_run_arrays(run, users, train, full_train, stats["num_items"])
        metrics = per_user_metrics(predictions, targets)
        arrays.append(metrics)
        saved_predictions.append(predictions)
        label = f"{index}:{run.name}"
        report["runs"][label] = {"path": str(run), "metrics": run_metrics(
            predictions, targets, metrics, selected, counts)}
        if args.groups:
            reach.append(target_reachability(run, users, targets, retained) if (run/"model.npz").is_file() else None)
    labels = list(report["runs"])
    for index in range(1, len(runs)):
        groups = {"all": np.ones(len(users), dtype=bool)}
        if args.groups:
            degrees = np.asarray(retained.sum(1)).ravel()[users]
            groups = pair_groups(degrees, reach[0], reach[index])
            groups.update(time_groups(full_train, warm))
        groups = {name: mask & selected for name, mask in groups.items()}
        report["comparisons"].append({"a": labels[0], "b": labels[index],
            "groups": compare_metrics(arrays[0], arrays[index], groups, args.bootstrap, args.seed),
            "run_metrics": {labels[j]: {name: run_metrics(saved_predictions[j], targets, arrays[j], mask, counts)
                                       for name, mask in groups.items()} for j in (0, index)}})
    if args.holm:
        results = [pair["groups"]["all"]["NDCG@20"] for pair in report["comparisons"]]
        if selected.any():
            for result, adjusted in zip(results, holm_adjust([result["p_one_sided"] for result in results])):
                result["holm_p"] = adjusted
        else:
            for result in results:
                result["holm_p"] = None
        report["multiplicity"] = "Holm trên NDCG@20 toàn nửa đã chọn, tất cả arm so với tham chiếu; nhóm/metric khác là khám phá"
        report["comparisons_in_gate"] = len(results)
    report["auxiliary_definitions"] = {"popularity": "count trong train được giữ lại; log tự nhiên",
        "longtail": "ngoài ceil(20% item active), xếp count giảm dần rồi ID tăng dần",
        "time": "timestamp validation bằng timestamp train cuối là same_day; dùng full train"}
    report["seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(markdown(report))
    return report


if __name__ == "__main__":
    main()
