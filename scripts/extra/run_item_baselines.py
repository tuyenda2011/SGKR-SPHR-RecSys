"""Validation-only full-catalog ItemKNN, RP3beta and EASE grids; preview unless --run."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits

from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR
from src.utils.run_budget import RunBudget
from src.utils.source_identity import SourceGuard, capture_sources


def prune_block(block, row_offset, k):
    """Keep positive top-k per source row, diagonal removed; ties by target ID."""
    rows, columns, values = [], [], []
    for row in range(block.shape[0]):
        start, stop = block.indptr[row:row + 2]
        ids, scores = block.indices[start:stop], block.data[start:stop]
        keep = (ids != row + row_offset) & (scores > 0)
        ids, scores = ids[keep], scores[keep]
        chosen = np.lexsort((ids, -scores))[:k]
        rows.extend([row] * len(chosen))
        columns.extend(ids[chosen].tolist())
        values.extend(scores[chosen].tolist())
    return sparse.csr_matrix((values, (rows, columns)), shape=block.shape)


def itemknn(x, k, shrink, block_size=256, prune_axis="source"):
    """Legacy API defaults to source; the primary grid explicitly uses target.

    Cosine is symmetric. Treating each computed row as a target, pruning its
    sources and transposing gives top-k per target column in x_u @ W.
    """
    if k < 1 or block_size < 1 or not math.isfinite(shrink) or shrink < 0:
        raise ValueError("ItemKNN requires positive k/block_size and nonnegative shrink")
    if prune_axis not in {"target", "source"}:
        raise ValueError("prune_axis must be target or source")
    counts = np.asarray(x.sum(0)).ravel()
    blocks = []
    for start in range(0, x.shape[1], block_size):
        stop = min(start + block_size, x.shape[1])
        block = (x[:, start:stop].T @ x).tocsr()
        row = np.repeat(np.arange(start, stop), np.diff(block.indptr))
        block.data /= np.sqrt(counts[row] * counts[block.indices]) + shrink
        blocks.append(prune_block(block, start, k))
    result = sparse.vstack(blocks, format="csr")
    return result.T.tocsr() if prune_axis == "target" else result


def rp3beta(x, k, alpha, beta, block_size=256):
    if (k < 1 or block_size < 1 or not math.isfinite(alpha) or alpha <= 0
            or not math.isfinite(beta) or beta <= -1):
        raise ValueError("Invalid RP3beta parameters")
    degrees = np.asarray(x.sum(1)).ravel()
    counts = np.asarray(x.sum(0)).ravel()
    p_ui = (sparse.diags(1 / np.maximum(degrees, 1)) @ x).tocsr()
    p_iu = (sparse.diags(1 / np.maximum(counts, 1)) @ x.T).tocsr()
    # Alpha acts on each transition probability BEFORE multiplication.
    p_ui.data **= alpha
    p_iu.data **= alpha
    penalty = 1 / np.maximum(counts, 1) ** beta
    blocks = []
    for start in range(0, x.shape[1], block_size):
        block = (p_iu[start:start + block_size] @ p_ui).tocsr()
        block.data *= penalty[block.indices]
        blocks.append(prune_block(block, start, k))
    return sparse.vstack(blocks, format="csr")


def ease_memory_required(num_items):
    # Float32 Gram/inverse, LAPACK work/copy allowance, plus 1 GiB reserve.
    return 3 * num_items * num_items * np.dtype("float32").itemsize + 2**30


def ease(x, regularization, *, available_bytes=None):
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("EASE regularization must be positive and finite")
    available = psutil.virtual_memory().available if available_bytes is None else available_bytes
    required = ease_memory_required(x.shape[1])
    if available < required:
        raise MemoryError(f"Full-catalog EASE needs {required} free bytes; available {available}. Catalog is not reduced.")
    train = x.astype(np.float32)
    gram = (train.T @ train).toarray(order="F")
    gram.flat[::x.shape[1] + 1] += regularization
    inverse = linalg.inv(gram, overwrite_a=True, check_finite=False)
    diagonal = inverse.diagonal().copy()
    if not np.isfinite(inverse).all() or np.any(diagonal <= 0):
        raise FloatingPointError("EASE inverse is nonfinite or has nonpositive diagonal")
    inverse /= -diagonal[None, :]
    np.fill_diagonal(inverse, 0.)
    return inverse


class ItemScores:
    def __init__(self, interactions, weights):
        self.interactions, self.weights = interactions, weights

    def score(self, users):
        result = self.interactions[users] @ self.weights
        return result.toarray() if sparse.issparse(result) else np.asarray(result)

    def get_user_rating_scores(self, users, all_users=None, all_items=None):
        return torch.from_numpy(self.score(users.detach().cpu().numpy())).to(users.device)


def mostpop_predictions(x, evaluator):
    """Deterministic count ranking with item-ID ties and the shared history mask."""
    counts = np.asarray(x.sum(0)).ravel()
    active = np.asarray(sorted(evaluator.candidate_items), dtype=np.int64)
    ordered = active[np.lexsort((active, -counts[active]))]
    rows = []
    for user in evaluator.eval_users:
        seen = evaluator.train_history.get(user, set())
        row = []
        for item in ordered:
            if int(item) not in seen:
                row.append(int(item))
                if len(row) == 20:
                    break
        if len(row) != 20:
            raise ValueError("MostPop requires 20 eligible active candidates")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long).reshape(len(rows), 20)


def rp3beta_extension_grid():
    # Gate A-bổ sung: the primary grid's best RP3beta arm sat at its smallest
    # alpha/beta and largest k, so extend outward and skip the already-run arm.
    # A negative beta rewards popular targets, testing the sign of the popularity term.
    return [{"model": "rp3beta", "alpha": alpha, "beta": beta, "k": k}
            for alpha, beta, k in itertools.product((.25, .5), (-.1, 0., .1, .2, .3), (200, 500))
            if (alpha, beta, k) != (.5, .3, 200)]


def grid(models, name="primary"):
    if name == "rp3beta-extension":
        if list(models) != ["rp3beta"]:
            raise ValueError("The rp3beta-extension grid requires --models rp3beta")
        return rp3beta_extension_grid()
    if name != "primary":
        raise ValueError("Unknown baseline grid")
    result = []
    if "mostpop" in models:
        result.append({"model": "mostpop"})
    if "itemknn" in models:
        result.extend({"model": "itemknn", "k": k, "shrink": h, "prune_axis": "target"}
                      for k, h in itertools.product((50, 100, 200, 500), (0, 10)))
        result.extend({"model": "itemknn", "k": 100, "shrink": h, "prune_axis": "source"} for h in (0, 10))
    if "rp3beta" in models:
        result.extend({"model": "rp3beta", "alpha": alpha, "beta": beta, "k": k}
                      for alpha, beta, k in itertools.product((.5, 1.), (.3, .5, .7), (100, 200)))
    if "ease" in models:
        result.extend({"model": "ease", "regularization": value} for value in (100, 300, 1000))
    return result


def fit_weights(x, config, block_size):
    if config["model"] == "itemknn":
        return itemknn(x, config["k"], config["shrink"], block_size, config.get("prune_axis", "source"))
    if config["model"] == "rp3beta":
        return rp3beta(x, config["k"], config["alpha"], config["beta"], block_size)
    if config["model"] == "ease":
        return ease(x, config["regularization"])
    raise ValueError("Unknown item baseline")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def evaluate_config(config, output, x, evaluator, args, source_hashes, identity):
    output.mkdir(parents=True, exist_ok=False)
    guard = SourceGuard(ROOT, source_hashes)
    guard.archive_to(output / "source")
    guard = SourceGuard(ROOT, source_hashes, archive=output / "source")
    plan = {"status": "prepared", "validation_only": True, **identity,
            "config": config, "density": args.sparsity, "density_seed": args.density_seed,
            "evaluation_protocol": EVALUATION_PROTOCOL, "catalog_size": x.shape[1],
            "candidate_items": len(evaluator.candidate_items), "source_hashes": dict(source_hashes)}
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    budget = RunBudget(config["model"])
    budget.configs_tried = 1
    try:
        guard.check()
        # Record the memory check even when this arm must be skipped.
        if config["model"] == "ease":
            plan["memory_check"] = {"available_bytes": psutil.virtual_memory().available,
                                    "required_bytes": ease_memory_required(x.shape[1]), "dtype": "float32"}
        if config["model"] == "mostpop":
            inference_started = time.perf_counter()
            predictions = mostpop_predictions(x, evaluator)
            latency = (time.perf_counter()-inference_started)*1000/max(1, len(evaluator.eval_users))
        else:
            try:
                weights = fit_weights(x, config, args.block_size)
            except MemoryError as error:
                if config["model"] != "ease":
                    raise
                result = {"status": "skipped", "validation_only": True, "config": config,
                          "reason": str(error), "memory_check": plan["memory_check"], "val_metrics": None}
                plan["status"] = "skipped"
                write_json(output / "result.json", result)
                return result
            model = ItemScores(x, weights)
            guard.check()
            evaluator.score_fn = model.get_user_rating_scores
            predictions, latency = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
        for row, user in enumerate(evaluator.eval_users):
            chosen = predictions[row].tolist()
            if (len(set(chosen)) != 20 or set(chosen) & evaluator.train_history.get(user, set())
                    or not set(chosen) <= evaluator.candidate_items):
                raise AssertionError("Invalid baseline predictions under shared mask/catalog")
        metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
        np.save(output / "val_predictions.npy", predictions.numpy().astype(np.int64, copy=False))
        np.save(output / "val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))
        result = {"status": "complete", "model": config["model"], "config": config,
                  "validation_only": True, "val_metrics": metrics, "seconds": time.perf_counter() - started,
                  "inference_latency_ms_per_user": latency}
        guard.check()
        write_json(output / "result.json", result)
        plan["status"] = "complete"
        return result
    except BaseException as error:
        plan["status"], plan["error"] = "failed_or_interrupted", f"{type(error).__name__}: {error}"
        raise
    finally:
        evaluator.score_fn = None  # Release each weight matrix before the next arm.
        budget.write(output, status=plan["status"])
        write_json(output / "plan.json", plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("mostpop", "itemknn", "rp3beta", "ease"), default=["itemknn", "rp3beta", "ease"])
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--sparsity", type=float, default=1.)
    parser.add_argument("--density-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--grid", choices=("primary", "rp3beta-extension"), default="primary")
    parser.add_argument("--run", action="store_true", help="Fit/evaluate the real bundle; requires gate A approval")
    args = parser.parse_args(argv)
    if (not 0 < args.sparsity <= 1 or args.density_seed < 0 or args.eval_batch_size < 1
            or args.block_size < 1 or args.threads < 1 or len(set(args.models)) != len(args.models)):
        raise ValueError("Invalid baseline grid controls")
    configs = grid(args.models) if args.grid == "primary" else grid(args.models, args.grid)
    if not args.run:
        preview = {"status": "preview_only", "validation_only": True, "fit_called": False, "grid": args.grid,
                   "configurations": configs, "n_configurations": len(configs),
                   "itemknn_primary_prune_axis": "target", "itemknn_control_prune_axis": "source",
                   "ease_policy": "Bỏ qua nếu RAM trống dưới 3 ma trận float32 đầy đủ + 1 GiB; giữ nguyên catalog"}
        print(json.dumps(preview, indent=2))
        return preview
    output = (args.output or ROOT / "results/experiments" / datetime.now(timezone.utc).strftime("item_baselines_%Y%m%d_%H%M%S_%f")).resolve()
    budget = RunBudget("item_baselines")
    if output.exists():
        raise FileExistsError(output)
    source_hashes = capture_sources(ROOT, [Path(__file__), ROOT / "scripts/validation_io.py", *sorted((ROOT / "src").rglob("*.py"))])
    bundle, full_train, validation, stats = load_validation_inputs(args.bundle, ROOT)
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    validation, cohort = warm_cohort(train, validation)
    if validation.empty:
        raise ValueError("No warm validation targets")
    x = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(), stats["num_users"], stats["num_items"])
    evaluator = Evaluator(full_train, validation, *x.shape, [10, 20], args.eval_batch_size,
                          candidate_items=set(train.i_idx.unique()), popularity_df=train)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"validation_only": True, "bundle_id": bundle.build_id, "validation_cohort": cohort,
               "configs_requested": len(configs), "selection_metric": "NDCG@20", "runs": []}
    identity = {"bundle_id": bundle.build_id, "retained_sha256": edge_hash(train),
                "mask_sha256": edge_hash(full_train), "cohort_sha256": edge_hash(validation)}
    try:
        with threadpool_limits(limits=args.threads):
            for index, config in enumerate(configs):
                budget.configs_tried += 1
                name = f"{index:02d}_" + "_".join(f"{key}-{value}" for key, value in config.items())
                result = evaluate_config(config, output / name, x, evaluator, args, source_hashes, identity)
                summary["runs"].append({"output": name, **result})
                write_json(output / "grid_summary.json", summary)
                print(f"[{index + 1}/{len(configs)}] {name}: {result['status']} {result['val_metrics']}", flush=True)
    finally:
        budget.write(output, configs_requested=len(configs),
                     configs_skipped=sum(run["status"] == "skipped" for run in summary["runs"]))
    return summary


if __name__ == "__main__":
    main()
