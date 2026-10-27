"""Prepare or execute validation-only SPHR training with the locked configuration (configs/sphr.yaml).

No training occurs without --run. One run per seed (--seed plus --confirm-seeds); each keeps the
best validation-NDCG@20 checkpoint. Completed runs resume only under identical
source/config/data identities. An interrupted run continues in its attempt folder from latest.pt
(written every epoch with optimizer and RNG state); an attempt without latest.pt is left as is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.config_io import read_model_config
from src.data.bundle import resolve_bundle
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.projected_optimizer import project_adam_parameter
from src.models.sphr import SOURCE_SHA256 as SPHR_SOURCE_SHA256
from src.models.sphr import SPHR, SPHRConfig
from src.utils.run_budget import RunBudget
from src.utils.seed import set_seed
from src.utils.source_identity import (
    SOURCE_SHA256 as GUARD_SOURCE_SHA256,
)
from src.utils.source_identity import (
    SourceGuard,
    capture_sources,
    source_identity,
    verify_imported,
)
from src.utils.training_progress import TrainingProgress

SOURCE_SHA256 = source_identity(__file__)
_IMPORTED_SOURCES = {"src/models/sphr.py": SPHR_SOURCE_SHA256,
                     "scripts/tune_sphr.py": SOURCE_SHA256,
                     "src/utils/source_identity.py": GUARD_SOURCE_SHA256}
_RUNTIME_SOURCES = capture_sources(ROOT, [Path(__file__), ROOT/"scripts/config_io.py",
    *sorted((ROOT/"src").rglob("*.py"))], _IMPORTED_SOURCES)


def runtime_source_guard(args=None, plan=None):
    hashes = _RUNTIME_SOURCES if plan is None else plan["source_hashes"]
    verify_imported(hashes, _RUNTIME_SOURCES)
    guard = SourceGuard(ROOT, hashes, config=None if args is None else args.config,
                        archive=None if plan is None else args.output.resolve()/"source")
    guard.check()
    return guard


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge_hash(frame):
    data = frame[["u_idx", "i_idx"]].sort_values(["u_idx", "i_idx"]).to_numpy(dtype=np.int64)
    return hashlib.sha256(data.tobytes()).hexdigest()


class QuerySampler:
    """Training queries: a uniformly random retained user, then one of that user's positives."""

    def __init__(self, frame):
        ordered = frame.sort_values(["u_idx", "i_idx"])
        self.users, self.starts, self.counts = np.unique(ordered.u_idx.to_numpy(), return_index=True, return_counts=True)
        self.items = ordered.i_idx.to_numpy(dtype=np.int64)

    def sample(self, size, rng):
        chosen = rng.integers(len(self.users), size=size)
        return self.users[chosen], self.items[self.starts[chosen] + rng.integers(self.counts[chosen])]


def time_limit_reached(args, started):
    hours = getattr(args, "max_hours", None)
    if hours is None:
        return False
    return time.perf_counter() >= getattr(args, "_deadline", started+hours*3600)


def run_job(config, seed, output, train, evaluator, dimensions, args, device, progress=None, *,
            source_guard=None, parent_budget=None):
    guard = source_guard or runtime_source_guard()
    verify_imported(guard.hashes, _RUNTIME_SOURCES)
    guard.check()
    job_root = output / f"seed{seed}"
    job_root.mkdir(exist_ok=True)
    completed = job_root / "completed.json"
    if completed.exists():
        if progress is not None:
            progress.stage(f"seed{seed}: verifying completed checkpoint")
        result = json.loads(completed.read_text(encoding="utf-8"))
        if result.get("source_provenance") != "plan-and-import-pinned-v1" or result.get("source_hashes") != dict(guard.hashes):
            raise ValueError("Completed run source provenance changed; use a fresh output")
        if sha256(output / result["checkpoint"]) != result["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint checksum mismatch")
        return result
    attempts = sorted(job_root.glob("attempt*"))
    resumable = bool(attempts) and (attempts[-1] / "latest.pt").is_file()
    attempt = attempts[-1] if resumable else job_root / f"attempt{len(attempts) + 1:03d}"
    attempt.mkdir(exist_ok=resumable)
    budget = RunBudget("sphr", device)
    budget.configs_tried = 1
    if parent_budget is not None:
        parent_budget.configs_tried += 1
    if progress is not None:
        progress.stage(f"seed{seed}: initializing model on {device}; output={attempt}")
    set_seed(seed)
    model = SPHR(*dimensions, config)
    model.set_history(train.u_idx.to_numpy(), train.i_idx.to_numpy())
    model.to(device)
    print(f"[seed{seed}] normalization alpha={config.history_normalization:g}"
          f" | retained users={int((model.user_degree > 0).sum()):,}"
          f" | singleton users={int((model.user_degree == 1).sum()):,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    sampler = QuerySampler(train)
    rng = np.random.default_rng(seed)
    evaluator.score_fn = model.get_user_rating_scores
    best, best_epoch, stale, history = -math.inf, 0, 0, []
    stopped_reason = None
    started = time.perf_counter()
    checkpoint = attempt / "best.pt"
    latest = attempt / "latest.pt"
    probe_ids = torch.as_tensor(evaluator.eval_users[:8], dtype=torch.long, device=device)
    best_payload, start_epoch = None, 1
    if resumable:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if (state.get("source_provenance") != "plan-and-import-pinned-v1" or state.get("source_hashes") != dict(guard.hashes)
                or state.get("seed") != seed or state.get("config") != asdict(config)):
            raise ValueError("SPHR latest.pt belongs to a different source/config/seed; use a fresh output")
        meta = json.loads(state["state_json"])
        if meta["epoch"] > args.epochs:
            raise ValueError("Requested epochs precede the resume checkpoint")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rng.bit_generator.state = meta["numpy_generator"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, best_epoch, stale, history = meta["best"], meta["best_epoch"], meta["stale"], meta["history"]
        best_payload, start_epoch = state["best_payload"], meta["epoch"] + 1
        started -= meta["seconds"]
        # An interruption between the best.pt and latest.pt writes leaves best.pt off by one epoch.
        if best_payload is not None and (not checkpoint.is_file() or
                torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"] != best_epoch):
            guard.atomic_write(checkpoint, lambda temporary: torch.save(best_payload, temporary))
        print(f"[seed{seed}] Resuming {attempt.name} after epoch {meta['epoch']}"
              f" (best NDCG@20 {best:.6f} at epoch {best_epoch})", flush=True)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            guard.check()
            epoch_label = f"seed{seed} | Epoch {epoch}/{args.epochs}"
            if progress is not None:
                progress.stage(f"{epoch_label} | training starts")
            write_json(attempt / "status.json", {"status": "running", "epoch": epoch, "stage": "training"})
            before = time.perf_counter()
            model.train()
            totals = np.zeros(4)
            gradient_sum, gradient_max = 0., 0.
            users, positive = sampler.sample(len(train), rng)
            for offset in range(0, len(train), args.batch_size):
                batch = slice(offset, offset + args.batch_size)
                batch_size = len(users[batch])
                tensors = [torch.as_tensor(array[batch], dtype=torch.long, device=device) for array in (users, positive)]
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.ranking_loss(*tensors)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite SPHR loss")
                loss.backward()
                # Infinity cap checks gradient finiteness without modifying the gradient.
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), math.inf, error_if_nonfinite=True))
                gradient_sum += batch_size * gradient_norm
                gradient_max = max(gradient_max, gradient_norm)
                optimizer.step()
                # Norm constraint in Adam's diagonal metric (radius 100 rarely binds).
                projected = [project_adam_parameter(table, radius, optimizer)
                             for table, radius in zip(model.constrained_tables(), model.parameter_radii())]
                totals += batch_size * np.array([float(loss.detach()), float(parts["ranking"].detach()),
                                                 float(parts["penalty"].detach()), max(projected)])
                if progress is not None:
                    progress(f"{epoch_label} | Adam batches", offset+batch_size, len(train), detail=f"loss={float(loss.detach()):.6f}")
            totals /= len(train)
            model.eval()
            if progress is not None:
                progress.stage(f"{epoch_label} | validation on {len(evaluator.eval_users):,} users")
            write_json(attempt / "status.json", {"status": "running", "epoch": epoch, "stage": "validation"})
            with torch.no_grad():
                all_users, all_items = model()
                diagnostics = model.diagnostics(all_items)
                predictions, _ = evaluator.get_predictions(all_users, all_items, device)
                metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
                probe_scores = model.get_user_rating_scores(probe_ids, all_users, all_items).cpu()
            score = metrics["NDCG@20"]
            improved = score > best
            if improved:
                best, best_epoch, stale = score, epoch, 0
                payload = {"source_provenance": "plan-and-import-pinned-v1", "source_hashes": dict(guard.hashes),
                           "config": asdict(config), "seed": seed, "epoch": epoch,
                           "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           "val_metrics": metrics, "probe_scores": probe_scores}
                guard.atomic_write(checkpoint, lambda temporary: torch.save(payload, temporary))
                best_payload = payload
            else:
                stale += 1
            record = {"epoch": epoch, "loss": totals[0], "ranking_loss": totals[1], "penalty": totals[2],
                      "max_projected_fraction": totals[3], "learning_rate": config.learning_rate,
                      "seconds": time.perf_counter() - before, **metrics, **diagnostics,
                      "gradient_norm_mean": gradient_sum / len(train), "gradient_norm_max": gradient_max}
            history.append(record)
            budget.epochs += 1
            if parent_budget is not None:
                parent_budget.epochs += 1
            pd.DataFrame(history).to_csv(attempt / "history.csv", index=False)
            # Everything needed to continue exactly from the next epoch; non-tensors as JSON (weights_only load).
            resume_state = {
                "source_provenance": "plan-and-import-pinned-v1", "source_hashes": dict(guard.hashes),
                "seed": seed, "config": asdict(config),
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "best_payload": best_payload,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "state_json": json.dumps({"epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                                          "history": history, "seconds": time.perf_counter() - started,
                                          "numpy_generator": rng.bit_generator.state}, default=float)}
            guard.atomic_write(latest, lambda temporary: torch.save(resume_state, temporary))
            write_json(attempt / "status.json", {"status": "running", "epoch": epoch, "best_epoch": best_epoch,
                                                 "best_ndcg20": best})
            print(f"[seed{seed}] Epoch {epoch:02d}/{args.epochs} [{record['seconds']:.1f}s]"
                  f" | Loss: {totals[0]:.4f} | Val Recall@10: {metrics['Recall@10']:.6f}"
                  f" | Val NDCG@10: {metrics['NDCG@10']:.6f} | Monitor NDCG@20: {score:.6f}"
                  + (" [BEST]" if improved else f" [stale {stale}/{args.patience}]"), flush=True)
            if time_limit_reached(args, started):
                stopped_reason = "max_hours"
                break
            if stale >= args.patience:
                break
        guard.check()
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get("source_provenance") != "plan-and-import-pinned-v1" or saved.get("source_hashes") != dict(guard.hashes):
            raise ValueError("SPHR checkpoint source provenance differs from this run")
        if progress is not None:
            progress.stage("Replaying best checkpoint and validation metrics")
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        with torch.no_grad():
            all_users, all_items = model()
            torch.testing.assert_close(model.get_user_rating_scores(probe_ids, all_users, all_items).cpu(),
                                       saved["probe_scores"].cpu(), rtol=1e-5, atol=1e-6)
            predictions, latency = evaluator.get_predictions(all_users, all_items, device)
        metrics = compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])
        for key in ("NDCG@20", "Recall@20"):
            if abs(metrics[key] - saved["val_metrics"][key]) > 2e-5:
                raise AssertionError("Checkpoint metric replay differs beyond numerical tolerance")
        for row, user in enumerate(evaluator.eval_users):
            selected = set(predictions[row].tolist())
            if len(selected) != 20 or selected & evaluator.train_history.get(user, set()) or not selected <= evaluator.candidate_items:
                raise AssertionError("Invalid recommendation masks or duplicate predictions")
        np.save(attempt / "val_predictions.npy", predictions.numpy())
        np.save(attempt / "val_users.npy", np.asarray(evaluator.eval_users))
        result = {"source_provenance": "plan-and-import-pinned-v1", "source_hashes": dict(guard.hashes),
                  "seed": seed, "config": asdict(config), "val_metrics": metrics,
                  "best_epoch": best_epoch, "epochs_run": len(history), "checkpoint": checkpoint.relative_to(output).as_posix(),
                  "checkpoint_sha256": sha256(checkpoint), "seconds": time.perf_counter() - started,
                  "inference_latency_ms_per_user": latency, "checkpoint_replay": "passed", "validation_only": True,
                  "checkpoint_metric_max_gap": max(abs(metrics[key] - saved["val_metrics"][key]) for key in ("NDCG@20", "Recall@20"))}
        if stopped_reason is not None:
            result["stopped_reason"] = stopped_reason
        guard.check()
        write_json(attempt / "status.json", {"status": "complete"})
        write_json(completed, result)
        latest.unlink(missing_ok=True)  # resume state is no longer needed once the run is complete
        return result
    except BaseException as error:
        write_json(attempt / "status.json", {"status": "interrupted_or_failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        budget.write(attempt, stopped_reason=stopped_reason)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sphr.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirm-seeds", type=int, nargs="*", default=[], help="Additional training seeds")
    parser.add_argument("--density-seed", type=int, default=42)
    parser.add_argument("--sparsity", type=float, default=1.0, help="Fraction of train edges retained (density)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Wall-time cap for the whole invocation; finish the running epoch and replay best")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-interval", type=float, default=10., help="Progress/heartbeat interval in seconds")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def prepare(args):
    runtime_source_guard()
    if args.max_hours is not None and (isinstance(args.max_hours, bool)
            or not math.isfinite(args.max_hours) or args.max_hours <= 0):
        raise ValueError("max_hours must be positive and finite")
    config_digest = sha256(args.config)
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        raise ValueError("log_interval must be positive and finite")
    for name in ("epochs", "patience", "batch_size", "eval_batch_size", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.sparsity <= 1 or min(args.seed, args.density_seed, *args.confirm_seeds) < 0:
        raise ValueError("Invalid density or seed")
    if len(set(args.confirm_seeds)) != len(args.confirm_seeds) or args.seed in args.confirm_seeds:
        raise ValueError("Additional seeds must be distinct and differ from --seed")
    config = SPHRConfig.from_dict(read_model_config(args.config, "sphr"))
    parameters = {key: value for key, value in vars(args).items() if key not in {"run", "dry_run", "output", "config"}}
    plan = {"status": "prepared", "model": "sphr", "parameters": parameters, "config": asdict(config),
            "seeds": [args.seed, *args.confirm_seeds], "optimization_scope": "Adam on resampled epoch queries",
            "criterion": "validation NDCG@20 selects the checkpoint within each seed", "validation_only": True}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return None
    output = args.output.resolve()
    config_path = args.config.resolve()
    name = config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else "input_config.yaml"
    source_map = {**_RUNTIME_SOURCES, name: config_digest}
    guard = SourceGuard(ROOT, source_map, config=args.config)
    guard.check()
    plan["source_hashes"] = source_map
    if output.exists():
        previous = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        for key in ("parameters", "config", "source_hashes"):
            if previous[key] != plan[key]:
                raise ValueError(f"Existing run differs in {key}; use a fresh output directory")
        for name, expected in previous["source_hashes"].items():
            if sha256(output / "source" / name) != expected:
                raise ValueError("Archived run source checksum mismatch")
        return previous
    output.mkdir(parents=True)
    guard.archive_to(output/"source")
    write_json(output / "plan.json", plan)
    return plan


def execute(args, plan):
    budget = RunBudget("sphr")
    if args.max_hours is not None:
        args._deadline = budget.started+args.max_hours*3600
    with TrainingProgress("SPHR", getattr(args, "log_interval", 10.), args.output.resolve()/"progress.log") as progress:
        try:
            plan["status"] = "loading"
            plan.pop("error", None)
            write_json(args.output.resolve()/"plan.json", plan)
            return _execute(args, plan, progress, budget)
        except BaseException as error:
            plan["status"] = "interrupted_or_failed"
            plan["error"] = f"{type(error).__name__}: {error}"
            write_json(args.output.resolve()/"plan.json", plan)
            raise
        finally:
            budget.write(args.output.resolve(), status=plan["status"], stopped_reason=plan.get("stopped_reason"))
            vars(args).pop("_deadline", None)


def _execute(args, plan, progress, budget):
    output = args.output.resolve()
    guard = runtime_source_guard(args, plan)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    budget.device = str(device)
    torch.set_num_threads(args.threads)
    progress.stage(f"Device={device}; resolving bundle and verifying data files")
    bundle = resolve_bundle(args.bundle, data_root=ROOT / "data")
    progress.stage("Reading train data")
    full = pd.read_parquet(bundle.split_path("train"))
    progress.stage("Reading validation data")
    validation = pd.read_parquet(bundle.split_path("val"))
    with bundle.artifact("mappings.pkl").open("rb") as stream:
        stats = pickle.load(stream)["stats"]  # nosec B301
    progress.stage(f"Retaining train edges at density={args.sparsity}; building warm validation cohort")
    train = create_sparse_train_set(full, args.sparsity, args.density_seed)
    warm, cohort = warm_cohort(train, validation)
    if warm.empty:
        raise ValueError("No warm validation cohort")
    identities = {"bundle_id": bundle.build_id, "retained_sha256": edge_hash(train), "cohort_sha256": edge_hash(warm),
                  "mask_sha256": edge_hash(full), "dimensions": [int(stats["num_users"]), int(stats["num_items"])],
                  "evaluation_protocol": EVALUATION_PROTOCOL, "cohort": cohort, "device": str(device),
                  "torch": torch.__version__, "numpy": np.__version__}
    if "data" in plan and plan["data"] != identities:
        raise ValueError("Run data/runtime identities changed")
    plan["data"] = identities
    plan["status"] = "training"
    write_json(output / "plan.json", plan)
    progress.stage(f"Train={len(train):,} edges; validation={len(warm):,} targets; building evaluator")
    evaluator = Evaluator(full, warm, *identities["dimensions"], [10, 20], args.eval_batch_size,
                          candidate_items=set(train.i_idx.unique()), popularity_df=train)
    config = SPHRConfig(**plan["config"])
    runs = []
    try:
        for seed in plan["seeds"]:
            if runs and time_limit_reached(args, budget.started):
                plan["stopped_reason"] = "max_hours"
                break
            runs.append(run_job(config, seed, output, train, evaluator, identities["dimensions"], args, device, progress,
                                source_guard=guard, parent_budget=budget))
            if runs[-1].get("stopped_reason") == "max_hours":
                plan["stopped_reason"] = "max_hours"
        means = {key: float(np.mean([run["val_metrics"][key] for run in runs])) for key in ("NDCG@20", "Recall@20")}
        summary = {"status": "complete", "config": plan["config"], "seeds": [run["seed"] for run in runs],
                   "val_metrics_mean": means,
                   "ndcg20_std": float(np.std([run["val_metrics"]["NDCG@20"] for run in runs], ddof=1)) if len(runs) > 1 else None,
                   "runs": [{"seed": run["seed"], "best_epoch": run["best_epoch"], "val_metrics": run["val_metrics"],
                             "checkpoint": run["checkpoint"]} for run in runs]}
        if plan.get("stopped_reason"):
            summary["stopped_reason"] = plan["stopped_reason"]
        write_json(output / "selection.json", summary)
        guard.check()
        plan["status"] = "complete"
        progress.stage(f"Complete: {len(runs)} seed(s), mean NDCG@20={means['NDCG@20']:.6f}")
        print(f"Complete | seeds={summary['seeds']} | mean NDCG@20={means['NDCG@20']:.6f} | {output}", flush=True)
    except BaseException as error:
        plan["status"] = "interrupted_or_failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "plan.json", plan)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arguments = make_parser().parse_args()
    prepared = prepare(arguments)
    if prepared is not None:
        print(f"Prepared SPHR run with seeds {prepared['seeds']}: {arguments.output.resolve()}", flush=True)
        if arguments.run:
            execute(arguments, prepared)
