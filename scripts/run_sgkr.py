"""Fit one SGKR configuration and evaluate validation only with the shared protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import torch

from scripts.config_io import read_model_config
from src.data.bundle import resolve_bundle
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR, SGKRConfig
from src.models.sgkr import SOURCE_SHA256 as SGKR_SOURCE_SHA256
from src.utils.run_budget import RunBudget
from src.utils.source_identity import (
    SOURCE_SHA256 as GUARD_SOURCE_SHA256,
)
from src.utils.source_identity import (
    SourceGuard,
    capture_sources,
    source_identity,
    verify_legacy_sources,
)

SOURCE_SHA256 = source_identity(__file__)
_IMPORTED_SOURCES = {"src/models/sgkr.py": SGKR_SOURCE_SHA256,
                     "scripts/run_sgkr.py": SOURCE_SHA256,
                     "src/utils/source_identity.py": GUARD_SOURCE_SHA256}
_RUNTIME_SOURCES = capture_sources(ROOT, [Path(__file__), ROOT/"scripts/config_io.py",
                                          *sorted((ROOT/"src").rglob("*.py"))], _IMPORTED_SOURCES)


def runtime_source_guard():
    guard = SourceGuard(ROOT, _RUNTIME_SOURCES)
    guard.check()
    return guard


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def edge_hash(frame):
    edges = frame[["u_idx", "i_idx"]].sort_values(["u_idx", "i_idx"]).to_numpy(dtype=np.int64)
    return hashlib.sha256(edges.tobytes()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_checkpoint_reference(path):
    """Pin an existing fitted checkpoint without rewriting its source lineage."""
    path = Path(path).resolve()
    source_plan_path = path.parent/"plan.json"
    source = json.loads(source_plan_path.read_text(encoding="utf-8"))
    runtime_source_guard()
    if source.get("model") != "sgkr":
        raise ValueError("Reuse requires an SGKR checkpoint with its original plan.json")
    names = ("src/models/sgkr.py", "src/data/sparsity.py", "src/data/quality_report.py",
             "src/evaluation/evaluator.py", "src/evaluation/metrics.py")
    compatibility = verify_legacy_sources(ROOT, source["source_hashes"],
        {name: _RUNTIME_SOURCES[name] for name in names}, archive=path.parent/"source")
    checkpoint_hash = digest(path)
    if source.get("fit_checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("Reused checkpoint checksum mismatch")
    reference = {"path": str(path), "sha256": checkpoint_hash, "source_plan": str(source_plan_path),
            "source_plan_sha256": digest(source_plan_path),
            "original_checkpoint_hash_recorded": "fit_checkpoint_sha256" in source}
    if compatibility:
        reference["interpretation_compatibility"] = compatibility
    return reference


def load_reused_fit(reference, config, train, dimensions, current):
    source_path = Path(reference["source_plan"])
    if digest(source_path) != reference["source_plan_sha256"] or digest(reference["path"]) != reference["sha256"]:
        raise ValueError("Reused fit changed after preparation")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for name in ("config", "bundle_id", "density", "density_seed", "retained_sha256", "cohort_sha256",
                 "mask_sha256", "evaluation_protocol", "versions"):
        if source[name] != current[name]:
            raise ValueError(f"Reused fit has different {name}")
    model = SGKR.load(reference["path"])
    if model.config != config or (model.num_users, model.num_items) != tuple(dimensions):
        raise ValueError("Reused checkpoint config/dimensions differ")
    retained = SGKR.interaction_matrix(train.u_idx.to_numpy(), train.i_idx.to_numpy(), *dimensions)
    if (model.interactions != retained).nnz:
        raise ValueError("Reused checkpoint retained edges differ")
    return model


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/sgkr.yaml")
    result.add_argument("--bundle", default=None)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--sparsity", type=float, default=.25, help="Fraction of train edges retained")
    result.add_argument("--density-seed", type=int, default=42, help="Fixed edge sample, independent of feature seed")
    result.add_argument("--popularity-tilt", type=float, default=None, help="Inference-only s*(1+c_i)^tilt; 0 disables")
    result.add_argument("--neighbors", type=int, default=None)
    result.add_argument("--alpha", type=float, default=None, help="0 disables cooccurrence shrinkage")
    result.add_argument("--support-alpha", type=float, default=None, help="Independent support-prior shrinkage; omitted uses config/legacy alpha")
    result.add_argument("--graph-weight", type=float, default=None, help="0 disables graph smoothing")
    result.add_argument("--boundary-weight", type=float, default=None, help="Graph boundary penalty in [0,1]; 1 preserves original precision")
    result.add_argument("--positive-weight", type=float, default=None)
    result.add_argument("--eta-linear", type=float, default=None)
    result.add_argument("--eta-bias", type=float, default=None)
    result.add_argument("--threads", type=int, default=None)
    result.add_argument("--no-adaptive-regularization", action="store_true")
    result.add_argument("--adaptive-power", type=float, default=None)
    result.add_argument("--adaptive-cap", type=float, default=None, help="Maximum support penalty, >=1; omitted preserves config")
    result.add_argument("--diagnostic-users", type=int, default=32, help="Bounded validation branch-margin sample; 0 disables")
    result.add_argument("--theory-diagnostics", action="store_true", help="Optional train-only graph principal-block audit")
    result.add_argument("--fit-checkpoint", type=Path, default=None,
                        help="Reuse a fitted model.npz with its source plan; skip all coefficient solves")
    result.add_argument("--eval-batch-size", type=int, default=128)
    result.add_argument("--dry-run", action="store_true", help="Inspect effective settings without reading labels or fitting")
    return result


def run(args):
    runtime_source_guard()
    config_digest = digest(args.config)
    settings = read_model_config(args.config, "sgkr-core")
    if not isinstance(settings, dict):
        raise TypeError("SGKR config must be a mapping")
    for name in ("neighbors", "alpha", "graph_weight", "positive_weight", "eta_linear", "eta_bias", "threads",
                 "adaptive_power", "adaptive_cap", "boundary_weight", "support_alpha", "popularity_tilt"):
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    if args.no_adaptive_regularization:
        settings["adaptive_regularization"] = False
    config = SGKRConfig.from_dict(settings)
    if not 0 < args.sparsity <= 1 or args.density_seed < 0 or args.eval_batch_size < 1 or args.diagnostic_users < 0:
        raise ValueError("Invalid sparsity, density seed or evaluation batch size")
    if args.dry_run:
        print(json.dumps({"model": "sgkr", "solver": "cholesky", "config": asdict(config),
                          "density": args.sparsity, "density_seed": args.density_seed,
                          "diagnostic_users": args.diagnostic_users,
                          "theory_diagnostics": getattr(args, "theory_diagnostics", False),
                          "validation_only": True}, indent=2))
        return None
    fit_reference = fit_checkpoint_reference(args.fit_checkpoint) if getattr(args, "fit_checkpoint", None) else None
    output = args.output or ROOT / "results/experiments" / datetime.now(timezone.utc).strftime("sgkr_%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = RunBudget("sgkr")
    budget.configs_tried = 1
    bundle = resolve_bundle(args.bundle, data_root=ROOT / "data")
    full_train = pd.read_parquet(bundle.split_path("train"))
    validation = pd.read_parquet(bundle.split_path("val"))
    # Existing trusted, local project mapping; never read test labels.
    with bundle.artifact("mappings.pkl").open("rb") as stream:
        stats = pickle.load(stream)["stats"]  # nosec B301
    train = create_sparse_train_set(full_train, args.sparsity, args.density_seed)
    validation, cohort = warm_cohort(train, validation)
    if validation.empty:
        raise ValueError("No warm validation targets")
    config_path = args.config.resolve()
    name = config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else "effective_input_config.yaml"
    source_hashes = {**_RUNTIME_SOURCES, name: config_digest}
    guard = SourceGuard(ROOT, source_hashes, config=args.config)
    guard.archive_to(output/"source")
    guard = SourceGuard(ROOT, source_hashes, config=args.config, archive=output/"source")
    plan = {"status": "prepared", "model": "sgkr", "stage": "exploratory_validation_only",
            "bundle_id": bundle.build_id, "bundle_path": str(bundle.root), "config": asdict(config),
            "density": args.sparsity, "density_seed": args.density_seed,
            "diagnostic_users": args.diagnostic_users,
            "theory_diagnostics": getattr(args, "theory_diagnostics", False),
            "retained_sha256": edge_hash(train), "cohort_sha256": edge_hash(validation),
            "mask_sha256": edge_hash(full_train), "candidate_items": int(train.i_idx.nunique()),
            "validation_cohort": cohort, "evaluation_protocol": EVALUATION_PROTOCOL,
            "source_hashes": source_hashes, "source_provenance": "plan-and-import-pinned-v1",
            "solver": "cholesky_float64_cpu",
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": torch.__version__}, "output": str(output)}
    if fit_reference is not None:
        plan["fit_reference"] = fit_reference
        shutil.copy2(fit_reference["source_plan"], output/"original_fit_plan.json")
    write_json(output / "plan.json", plan)
    print(f"[SGKR] Retained edges: {len(train):,} | Val users: {cohort['evaluable_users']:,} | Output: {output}", flush=True)
    log = output / "progress.jsonl"

    def progress(event):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        if event["stage"] == "features_ready":
            print(f"[SGKR] Features ready [{event['seconds']:.1f}s] | Solving item systems...", flush=True)
        else:
            print(f"[SGKR] Items {event['completed']:,}/{event['items']:,} [{event['seconds']:.1f}s]", flush=True)

    started = time.perf_counter()
    try:
        guard.check()
        plan["status"] = "evaluating_saved_fit" if fit_reference else "fitting"
        write_json(output / "plan.json", plan)
        if fit_reference is not None:
            print("[SGKR] Reusing fitted checkpoint; no coefficient solves.", flush=True)
            model = load_reused_fit(fit_reference, config, train, [stats["num_users"], stats["num_items"]], plan)
            guard.atomic_write(output/"model.npz", lambda temporary: shutil.copy2(fit_reference["path"], temporary))
            if digest(output/"model.npz") != fit_reference["sha256"]:
                raise ValueError("Reused checkpoint changed during copy")
        else:
            model = SGKR(config).fit(train.u_idx.to_numpy(), train.i_idx.to_numpy(),
                                     stats["num_users"], stats["num_items"], progress)
            guard.atomic_write(output/"model.npz", model.save)
        plan["fit_checkpoint_sha256"] = digest(output/"model.npz")
        write_json(output/"plan.json", plan)
        if getattr(args, "theory_diagnostics", False):
            from src.evaluation.theory_diagnostics import sgkr_graph_diagnostics
            write_json(output/"theory_diagnostics.json", sgkr_graph_diagnostics(model))
        # Reload the artifact before evaluating; verify independent score reconstruction.
        guard.check()
        restored = SGKR.load(output / "model.npz")
        probe = np.asarray(sorted(validation.u_idx.unique())[:8], dtype=np.int64)
        np.testing.assert_allclose(model.score(probe), restored.score(probe), rtol=1e-12, atol=1e-12)
        maximum_gap = 0.
        cached_probe = restored.score(probe)
        for item in range(min(32, model.num_items)):
            neighbors = restored.neighborhoods[item]
            theta = np.concatenate((restored.linear_coefficients[neighbors, item].toarray().ravel(),
                                    [restored.bias[item]]))
            direct = restored.design_rows(probe, item) @ theta
            cached = cached_probe[:, item]
            maximum_gap = max(maximum_gap, float(np.max(np.abs(direct - cached))))
            np.testing.assert_allclose(direct, cached, rtol=1e-10, atol=1e-12)
        evaluator = Evaluator(full_train, validation, model.num_users, model.num_items, [10, 20],
                              args.eval_batch_size, candidate_items=set(train.i_idx.unique()),
                              popularity_df=train, score_fn=restored.get_user_rating_scores)
        predictions, latency = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
        for row, user in enumerate(evaluator.eval_users):
            chosen = predictions[row].tolist()
            if len(set(chosen)) != 20 or set(chosen) & evaluator.train_history.get(user, set()):
                raise AssertionError("Duplicate or seen recommendations")
            if not set(chosen) <= evaluator.candidate_items:
                raise AssertionError("Noncandidate recommendation")
        metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
        diagnostics = branch_diagnostics(restored, evaluator, predictions, args.diagnostic_users)
        write_json(output / "branch_diagnostics.json", diagnostics)
        np.save(output / "val_predictions.npy", predictions.numpy())
        np.save(output / "val_users.npy", np.asarray(evaluator.eval_users))
        result = {"model": "sgkr", "validation_only": True, "val_metrics": metrics,
                  "fit_report": model.fit_report, "score_reconstruction_max_gap": maximum_gap,
                  "checkpoint_replay": "passed", "inference_latency_ms_per_user": latency,
                  "branch_diagnostics_users": len(diagnostics["users"]),
                  "fit_reused": fit_reference is not None,
                  "seconds": time.perf_counter() - started}
        guard.check()
        write_json(output / "result.json", result)
        plan["status"] = "complete"
        print(f"[SGKR] Complete | Val Recall@10: {metrics['Recall@10']:.4f} | Val NDCG@10: {metrics['NDCG@10']:.4f}"
              f" | Val Recall@20: {metrics['Recall@20']:.4f} | Val NDCG@20: {metrics['NDCG@20']:.4f}", flush=True)
        return result
    except BaseException as error:
        plan["status"] = "failed_or_interrupted"
        plan["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        budget.write(output, stage="core", status=plan["status"])
        write_json(output / "plan.json", plan)


def branch_diagnostics(model, evaluator, predictions, maximum_users):
    """Compare validation positives to strongest returned irrelevant candidate.

    Reuse evaluator predictions and masks. This deterministic sorted-user sample
    is diagnostic, not a population estimate or a separate quality evaluator.
    """
    rows = []
    for row, user in enumerate(evaluator.eval_users[:maximum_users]):
        targets = sorted(evaluator.ground_truth[row])
        negatives = [int(item) for item in predictions[row].tolist() if int(item) not in targets]
        if not negatives:
            continue
        negative = negatives[0]
        candidates = np.asarray([*targets, negative], dtype=np.int64)
        parts = model.score_parts(np.asarray([user], dtype=np.int64), candidates)
        margins = [{"positive": int(target), "negative": negative,
                    **{name: float(values[0, index] - values[0, -1]) for name, values in parts.items()}}
                   for index, target in enumerate(targets)]
        for margin in margins:
            margin["total"] = sum(float(margin[name]) for name in ("linear", "bias"))
        rows.append({"user": int(user), "retained_degree": int(model.degrees[user]), "margins": margins})
    correlations = {}
    margins = [margin for row in rows for margin in row["margins"]]
    for first, second in (("linear", "bias"),):
        a, b = (np.asarray([margin[key] for margin in margins]) for key in (first, second))
        correlations[first+"_"+second] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else None
    return {"sample": "first_sorted_validation_users", "maximum_users": maximum_users,
            "branch_margin_correlations": correlations,
            "negative": "highest_ranked_returned_non_target_same_evaluator_mask", "users": rows}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(parser().parse_args())
