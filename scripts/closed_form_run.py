"""Shared provenance for the closed-form extra baselines (run_ease.py, run_gfcf.py), mirroring run_sgkr.py.

Each run folder gets: plan.json (status, data/cohort/mask hashes, versions, source hashes), source/ (archived
code + config), progress.jsonl (stage timings), budget.json (CPU/wall time), result.json, model.npz and the
validation predictions. Validation only; the test split is never read.
"""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy
import torch

from scripts.validation_io import edge_hash, load_validation_inputs
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.models.sgkr import SGKR
from src.utils.run_budget import RunBudget
from src.utils.source_identity import SourceGuard, capture_sources

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


class Run:
    """One fit + validation evaluation in `output`, with the same records as an SGKR run."""

    def __init__(self, model, args, config, script, solver):
        self.model, self.started = model, time.perf_counter()
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.budget = RunBudget(model)
        self.budget.configs_tried = 1
        bundle, self.full_train, validation, stats = load_validation_inputs(None, ROOT)
        self.train = create_sparse_train_set(self.full_train, args.sparsity, args.density_seed)
        self.validation, cohort = warm_cohort(self.train, validation)
        if self.validation.empty:
            raise ValueError("No warm validation targets")
        self.x = SGKR.interaction_matrix(self.train.u_idx.to_numpy(), self.train.i_idx.to_numpy(),
                                         stats["num_users"], stats["num_items"])
        sources = dict(capture_sources(ROOT, [Path(script), Path(__file__), Path(args.config),
                                              ROOT / "scripts/config_io.py", ROOT / "scripts/validation_io.py",
                                              *sorted((ROOT / "src").rglob("*.py"))]))
        guard = SourceGuard(ROOT, sources, config=args.config)
        guard.archive_to(self.output / "source")
        self.guard = SourceGuard(ROOT, sources, config=args.config, archive=self.output / "source")
        self.plan = {"status": "prepared", "model": model, "stage": "exploratory_validation_only",
                     "bundle_id": bundle.build_id, "config": config, "density": args.sparsity,
                     "density_seed": args.density_seed, "retained_sha256": edge_hash(self.train),
                     "cohort_sha256": edge_hash(self.validation), "mask_sha256": edge_hash(self.full_train),
                     "candidate_items": int(self.train.i_idx.nunique()), "catalog_size": int(self.x.shape[1]),
                     "validation_cohort": cohort, "evaluation_protocol": EVALUATION_PROTOCOL,
                     "source_hashes": sources, "solver": solver,
                     "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "scipy": scipy.__version__, "torch": torch.__version__},
                     "output": str(self.output)}
        write_json(self.output / "plan.json", self.plan)
        print(f"[{model.upper()}] Retained edges: {len(self.train):,} | Val users: {cohort['evaluable_users']:,} "
              f"| Output: {self.output}", flush=True)

    def progress(self, stage, **values):
        event = {"stage": stage, "seconds": time.perf_counter() - self.started, **values}
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(f"[{self.model.upper()}] {stage} [{event['seconds']:.1f}s]", flush=True)

    @contextmanager
    def tracked(self):
        """Status in plan.json and budget.json are written even when the run fails or is interrupted."""
        try:
            self.guard.check()
            self.plan["status"] = "fitting"
            write_json(self.output / "plan.json", self.plan)
            yield self
            self.guard.check()
            self.plan["status"] = "complete"
        except BaseException as error:
            self.plan["status"] = "failed_or_interrupted"
            self.plan["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.budget.write(self.output, stage="core", status=self.plan["status"])
            write_json(self.output / "plan.json", self.plan)

    def evaluate(self, score_fn, batch):
        evaluator = Evaluator(self.full_train, self.validation, *self.x.shape, [10, 20], batch,
                              candidate_items=set(self.train.i_idx.unique()), popularity_df=self.train,
                              score_fn=score_fn)
        predictions, latency = evaluator.get_predictions(torch.empty(0), torch.empty(0), torch.device("cpu"))
        for row, user in enumerate(evaluator.eval_users):
            chosen = predictions[row].tolist()
            if len(set(chosen)) != 20 or set(chosen) & evaluator.train_history.get(user, set()):
                raise AssertionError("Duplicate or seen recommendations")
            if not set(chosen) <= evaluator.candidate_items:
                raise AssertionError("Noncandidate recommendation")
        return evaluator, predictions, latency, compute_topk_metrics(evaluator.ground_truth, predictions, [10, 20])

    def save_predictions(self, evaluator, predictions):
        np.save(self.output / "val_predictions.npy", predictions.numpy().astype(np.int64, copy=False))
        np.save(self.output / "val_users.npy", np.asarray(evaluator.eval_users, dtype=np.int64))

    def finish(self, result):
        result = {"model": self.model, "validation_only": True, **result,
                  "seconds": time.perf_counter() - self.started}
        write_json(self.output / "result.json", result)
        metrics = result["val_metrics"]
        print(f"[{self.model.upper()}] Complete | Val Recall@10: {metrics['Recall@10']:.4f} | Val NDCG@10: "
              f"{metrics['NDCG@10']:.4f} | Val Recall@20: {metrics['Recall@20']:.4f} | Val NDCG@20: "
              f"{metrics['NDCG@20']:.4f}", flush=True)
        return result
