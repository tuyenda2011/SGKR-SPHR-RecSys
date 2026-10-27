"""Fit GF-CF once and evaluate validation only with the shared protocol.

Called by train_model.py --model gfcf. Same retained train sample, warm cohort, mask, candidate set
and Evaluator as run_sgkr.py; the full catalog is kept. Light enough for a laptop (sparse SVD, k=256).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch

from scripts.config_io import read_model_config
from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.gfcf import GFCF, GFCFConfig
from src.models.sgkr import SGKR


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = GFCFConfig(**read_model_config(args.config, "gfcf"))
    bundle, full_train, validation, stats = load_validation_inputs(None, ROOT)
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    validation, cohort = warm_cohort(train, validation)
    if validation.empty:
        raise ValueError("No warm validation targets")
    x = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(), stats["num_users"], stats["num_items"])
    write_json(output / "plan.json", {"status": "prepared", "validation_only": True, "config": vars(config),
                                      "density": args.sparsity, "density_seed": args.density_seed,
                                      "bundle_id": bundle.build_id, "retained_sha256": edge_hash(train),
                                      "evaluation_protocol": EVALUATION_PROTOCOL, "catalog_size": x.shape[1]})
    model = GFCF(config).fit(x)
    model.save(output / "model.npz")
    restored = GFCF.load(output / "model.npz")
    probe = np.arange(min(64, x.shape[0]))
    np.testing.assert_allclose(restored.score(probe), model.score(probe), rtol=1e-6, atol=1e-9)

    evaluator = Evaluator(full_train, validation, *x.shape, [10, 20], args.eval_batch_size,
                          candidate_items=set(train.i_idx.unique()), popularity_df=train,
                          score_fn=restored.get_user_rating_scores)
    predictions, latency = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
    for row, user in enumerate(evaluator.eval_users):
        chosen = predictions[row].tolist()
        if len(set(chosen)) != 20 or set(chosen) & evaluator.train_history.get(user, set()):
            raise AssertionError("Duplicate or seen recommendations")
        if not set(chosen) <= evaluator.candidate_items:
            raise AssertionError("Noncandidate recommendation")
    metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
    np.save(output / "val_predictions.npy", predictions.numpy().astype(np.int64, copy=False))
    np.save(output / "val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))
    result = {"model": "gfcf", "validation_only": True, "val_metrics": metrics, "fit_report": model.fit_report,
              "inference_latency_ms_per_user": latency, "validation_cohort": cohort, "config": vars(config)}
    write_json(output / "result.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/gfcf.yaml")
    result.add_argument("--sparsity", type=float, default=1.)
    result.add_argument("--density-seed", type=int, default=42)
    result.add_argument("--eval-batch-size", type=int, default=128)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
