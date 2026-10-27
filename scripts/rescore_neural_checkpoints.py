"""Validation-only re-scoring of saved LightGCN/XSimGCL/DirectAU checkpoints.

Writes val_predictions.npy/val_users.npy under the same warm cohort, train mask
and candidate catalog as run_item_baselines.py, so compare_val_runs.py can pair
them with item baselines and SGKR. No training step and no test labels.
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
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch

from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.graph import get_norm_adj_tensor
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL


def build_model(name, config, num_users, num_items):
    dim, layers = config["model"]["embedding_dim"], config["model"]["num_layers"]
    if name == "lightgcn":
        return LightGCN(num_users, num_items, embedding_dim=dim, num_layers=layers)
    if name == "xsimgcl":
        cfg = config["xsimgcl"]
        return XSimGCL(num_users, num_items, embedding_dim=dim, num_layers=layers,
                       contrastive_weight=cfg["contrastive_weight"], temperature=cfg["temperature"],
                       epsilon=cfg["epsilon"], contrastive_layer=cfg.get("contrastive_layer", 1))
    if name == "directau":
        cfg = config["directau"]
        return DirectAU(num_users, num_items, embedding_dim=dim, num_layers=layers, gamma=cfg["gamma"],
                        t=cfg["t"], profile=cfg.get("profile", "project_cosine"))
    raise ValueError(f"Unsupported model: {name}")


def rescore(checkpoint, model_name, output, args, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    bundle, full_train, validation, stats = load_validation_inputs(args.bundle, ROOT)
    trained_bundle = config.get("dataset", {}).get("bundle_id")
    if trained_bundle is not None and trained_bundle != bundle.build_id:
        raise ValueError(f"Checkpoint bundle {trained_bundle} differs from active bundle {bundle.build_id}")
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    warm, cohort = warm_cohort(train, validation)
    model = build_model(model_name, config, stats["num_users"], stats["num_items"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    evaluator = Evaluator(full_train, warm, stats["num_users"], stats["num_items"], [10, 20], args.eval_batch_size,
                          candidate_items=set(train.i_idx.unique()), popularity_df=full_train,
                          score_fn=model.get_user_rating_scores)
    started = time.perf_counter()
    with torch.no_grad():
        users, items = model(get_norm_adj_tensor(train, stats["num_users"], stats["num_items"], device))
        predictions, latency = evaluator.get_predictions(users, items, device)
    predictions = predictions.cpu()
    metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
    saved = state.get("val_metrics") or {}
    # The checkpoint's own validation NDCG@20 must be reproduced, otherwise the
    # protocol differs from the one used during training.
    if "NDCG@20" in saved and abs(saved["NDCG@20"] - metrics["NDCG@20"]) > 1e-6:
        raise AssertionError(f"Re-scored NDCG@20 {metrics['NDCG@20']} != checkpoint {saved['NDCG@20']}")
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "val_predictions.npy", predictions.numpy().astype(np.int64, copy=False))
    np.save(output / "val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))
    identity = {"bundle_id": bundle.build_id, "density": args.sparsity, "density_seed": args.density_seed,
                "retained_sha256": edge_hash(train), "mask_sha256": edge_hash(full_train),
                "cohort_sha256": edge_hash(warm)}
    plan = {"status": "complete", "validation_only": True, "fit_called": False, **identity,
            "model": model_name, "checkpoint": str(checkpoint), "checkpoint_epoch": state.get("epoch"),
            "evaluation_protocol": EVALUATION_PROTOCOL, "validation_cohort": cohort}
    result = {"status": "complete", "validation_only": True, "model": model_name, "val_metrics": metrics,
              "checkpoint_val_metrics": saved, "checkpoint_epoch": state.get("epoch"),
              "seconds": time.perf_counter() - started, "inference_latency_ms_per_user": latency}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--model", nargs="+", required=True, choices=("lightgcn", "xsimgcl", "directau"))
    parser.add_argument("--output", type=Path, nargs="+", required=True)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--sparsity", type=float, default=1.)
    parser.add_argument("--density-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not len(args.checkpoint) == len(args.model) == len(args.output):
        raise ValueError("Provide one model name and one output per checkpoint")
    if any(path.exists() for path in args.output):
        raise FileExistsError("Output directory already exists")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    results = []
    for checkpoint, name, output in zip(args.checkpoint, args.model, args.output):
        result = rescore(checkpoint, name, output, args, device)
        results.append(result)
        print(f"{name}: epoch {result['checkpoint_epoch']} {result['val_metrics']}", flush=True)
    return results


if __name__ == "__main__":
    main()
