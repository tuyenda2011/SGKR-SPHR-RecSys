"""Fit exact EASE once and evaluate validation only with the shared protocol.

Called by train_model.py --model ease. Same retained train sample, warm cohort, mask, candidate set
and Evaluator as run_sgkr.py; the full catalog is kept (no item pruning). Metrics and validation
predictions come from the full dense item-item matrix. That matrix (~8 GB) is not saved: model.npz keeps
the top-k weights of each target item for the demo, and its own validation metrics are recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch
from scipy import sparse
from scipy.linalg import lapack

from scripts.config_io import read_model_config
from scripts.extra.run_item_baselines import ItemScores
from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR


def ease_memory_required(num_items):
    # One float32 item-item matrix, factorized and inverted in place, plus a 1 GiB reserve.
    return num_items * num_items * np.dtype("float32").itemsize + 2**30


def ease(x, regularization, *, block=1024, available_bytes=None):
    """Exact EASE, B = I - P diag(1/diag P) with P = (X^T X + lambda I)^-1 (scripts/verify/check_ease_cholesky.py).

    X^T X + lambda I is symmetric positive definite, so P comes from LAPACK potrf + potri in place on a
    single Fortran-order float32 matrix instead of a general LU inverse with copies.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("EASE regularization must be positive and finite")
    n = x.shape[1]
    available = psutil.virtual_memory().available if available_bytes is None else available_bytes
    if available < ease_memory_required(n):
        raise MemoryError(f"Full-catalog EASE needs {ease_memory_required(n)} free bytes; available {available}. "
                          "Catalog is not reduced.")
    train = x.astype(np.float32)
    matrix = (train.T @ train).toarray(order="F")
    matrix.flat[::n + 1] += regularization
    matrix, info = lapack.spotrf(matrix, lower=0, overwrite_a=1, clean=0)
    if info:
        raise np.linalg.LinAlgError(f"Cholesky factorization failed (potrf info={info})")
    matrix, info = lapack.spotri(matrix, lower=0, overwrite_c=1)
    if info:
        raise np.linalg.LinAlgError(f"Cholesky inverse failed (potri info={info})")
    # potri fills the upper triangle only; mirror it into the lower one block by block.
    for start in range(0, n, block):
        stop = min(start + block, n)
        matrix[start:stop, :start] = matrix[:start, start:stop].T
        square = matrix[start:stop, start:stop]
        square[:] = np.triu(square) + np.triu(square, 1).T
    diagonal = matrix.diagonal().copy()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise FloatingPointError("EASE inverse is nonfinite or has nonpositive diagonal")
    matrix /= -diagonal[None, :]
    np.fill_diagonal(matrix, 0.)
    return matrix


def prune_columns(weights, k, block=512):
    """Keep the k largest weights of every target column of a dense item-item matrix (sparse CSC)."""
    n = weights.shape[1]
    rows, columns, values = [], [], []
    for start in range(0, n, block):
        part = weights[:, start:start + block]
        top = np.argpartition(-part, k, axis=0)[:k]
        rows.append(top.ravel())
        columns.append(np.broadcast_to(np.arange(start, start + part.shape[1]), top.shape).ravel())
        values.append(np.take_along_axis(part, top, axis=0).ravel())
    return sparse.csc_matrix((np.concatenate(values).astype(np.float32),
                              (np.concatenate(rows), np.concatenate(columns))), shape=weights.shape)


def evaluate(x, weights, full_train, validation, train, batch):
    evaluator = Evaluator(full_train, validation, *x.shape, [10, 20], batch,
                          candidate_items=set(train.i_idx.unique()), popularity_df=train,
                          score_fn=ItemScores(x, weights).get_user_rating_scores)
    predictions, latency = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
    for row, user in enumerate(evaluator.eval_users):
        chosen = predictions[row].tolist()
        if len(set(chosen)) != 20 or set(chosen) & evaluator.train_history.get(user, set()):
            raise AssertionError("Duplicate or seen recommendations")
        if not set(chosen) <= evaluator.candidate_items:
            raise AssertionError("Noncandidate recommendation")
    return evaluator, predictions, latency, compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_model_config(args.config, "ease")
    bundle, full_train, validation, stats = load_validation_inputs(None, ROOT)
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    validation, cohort = warm_cohort(train, validation)
    if validation.empty:
        raise ValueError("No warm validation targets")
    x = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(), stats["num_users"], stats["num_items"])
    write_json(output / "plan.json", {"status": "prepared", "validation_only": True, "config": config,
                                      "density": args.sparsity, "density_seed": args.density_seed,
                                      "bundle_id": bundle.build_id, "retained_sha256": edge_hash(train),
                                      "evaluation_protocol": EVALUATION_PROTOCOL, "catalog_size": x.shape[1],
                                      "required_free_bytes": ease_memory_required(x.shape[1])})
    started = time.perf_counter()
    weights = ease(x, config["l2"])  # raises MemoryError instead of shrinking the catalog
    seconds = time.perf_counter() - started

    evaluator, predictions, latency, metrics = evaluate(x, weights, full_train, validation, train, args.eval_batch_size)
    np.save(output / "val_predictions.npy", predictions.numpy().astype(np.int64, copy=False))
    np.save(output / "val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))

    pruned = prune_columns(weights, config["checkpoint_neighbors"])
    evaluator.score_fn = None  # release the dense matrix before the second pass
    del weights
    sparse.save_npz(output / "model.npz", pruned)
    _, pruned_predictions, _, pruned_metrics = evaluate(x, pruned, full_train, validation, train, args.eval_batch_size)
    overlap = float((pruned_predictions == predictions).float().mean())
    result = {"model": "ease", "validation_only": True, "val_metrics": metrics, "seconds": seconds,
              "inference_latency_ms_per_user": latency, "validation_cohort": cohort, "config": config,
              "checkpoint_val_metrics": pruned_metrics, "checkpoint_top20_same_position_rate": overlap}
    write_json(output / "result.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/ease.yaml")
    result.add_argument("--sparsity", type=float, default=1.)
    result.add_argument("--density-seed", type=int, default=42)
    result.add_argument("--eval-batch-size", type=int, default=128)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
