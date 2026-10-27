"""Rebuild the validation top-20 of the five reported models from their saved artifacts and compare.

Validation only: never reads the test split. No training. For each model the saved artifact
(SGKR model.npz, SPHR best.pt, baseline checkpoints) is reloaded, the validation cohort is
re-ranked, and the result is compared with the run's stored val_predictions.npy and NDCG@20.
Writes a JSON report with a sha256 of every rebuilt prediction array, so two reports (for example
before and after a refactor, on the same device) can be compared exactly.

  python scripts/verify_results.py --output results/verify/report.json [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows
import torch

from scripts.rescore_neural_checkpoints import build_model
from scripts.validation_io import load_validation_inputs
from src.data.graph import get_norm_adj_tensor
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR
from src.models.sphr import SPHR, SPHRConfig

# The density-1.0, seed-42 run of each model in the unified layout (scripts/train_model.py).
RUNS = {name: {"kind": kind, "run": f"results/{kind}/d100/seed42"} for name, kind in (
    ("SGKR", "sgkr"), ("SPHR", "sphr"), ("LightGCN", "lightgcn"), ("XSimGCL", "xsimgcl"), ("DirectAU", "directau"))}


def stored_ndcg(run):
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))["val_metrics"]["NDCG@20"]


def predict(spec, run, train, full_train, validation, warm, stats, device, batch):
    candidates = set(train.i_idx.unique())
    if spec["kind"] == "sgkr":
        model = SGKR.load(run / "checkpoint.npz")
        evaluator = Evaluator(full_train, warm, model.num_users, model.num_items, [10, 20], batch,
                              candidate_items=candidates, popularity_df=train, score_fn=model.get_user_rating_scores)
        predictions, _ = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
        return evaluator, predictions
    if spec["kind"] == "sphr":
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = SPHR(stats["num_users"], stats["num_items"], SPHRConfig.from_dict(state["config"]))
        model.set_history(train.u_idx.to_numpy(), train.i_idx.to_numpy())
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        evaluator = Evaluator(full_train, warm, stats["num_users"], stats["num_items"], [10, 20], batch,
                              candidate_items=candidates, popularity_df=train, score_fn=model.get_user_rating_scores)
        with torch.no_grad():
            users, items = model()
            predictions, _ = evaluator.get_predictions(users, items, device)
        return evaluator, predictions
    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = build_model(spec["kind"], state["config"], stats["num_users"], stats["num_items"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    evaluator = Evaluator(full_train, warm, stats["num_users"], stats["num_items"], [10, 20], batch,
                          candidate_items=candidates, popularity_df=full_train, score_fn=model.get_user_rating_scores)
    with torch.no_grad():
        users, items = model(get_norm_adj_tensor(train, stats["num_users"], stats["num_items"], device))
        predictions, _ = evaluator.get_predictions(users, items, device)
    return evaluator, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=2e-5, help="Allowed |NDCG@20 - stored| per model")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new report path")
    device = torch.device(args.device)
    _, full_train, validation, stats = load_validation_inputs(None, ROOT)
    train = create_sparse_train_set(full_train, 1., 42)
    warm, _ = warm_cohort(train, validation)
    report = {"device": args.device, "validation_only": True, "models": {}}
    for name in args.models:
        spec, run = RUNS[name], ROOT / RUNS[name]["run"]
        started = time.perf_counter()
        evaluator, predictions = predict(spec, run, train, full_train, validation, warm, stats, device,
                                         args.eval_batch_size)
        predictions = predictions.cpu().numpy().astype(np.int64)
        metrics = compute_topk_metrics(evaluator.ground_truth, torch.from_numpy(predictions), [10, 20])
        stored = np.load(run / "val_predictions.npy").astype(np.int64)
        same_users = np.array_equal(np.load(run / "val_users.npy").astype(np.int64),
                                    np.asarray(evaluator.eval_users, dtype=np.int64))
        reference = stored_ndcg(run)
        row = {"run": spec["run"], "NDCG@20": metrics["NDCG@20"], "Recall@20": metrics["Recall@20"],
               "stored_NDCG@20": reference, "same_users": same_users,
               "rows_identical_to_stored": float((predictions == stored).all(1).mean()) if same_users else None,
               "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
               "seconds": round(time.perf_counter() - started, 1)}
        row["ok"] = bool(same_users and reference is not None and abs(metrics["NDCG@20"] - reference) <= args.tolerance)
        report["models"][name] = row
        print(f"{name:9s} NDCG@20={metrics['NDCG@20']:.5f} stored={reference} "
              f"identical_rows={row['rows_identical_to_stored']} ok={row['ok']}", flush=True)
    report["all_ok"] = all(row["ok"] for row in report["models"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
