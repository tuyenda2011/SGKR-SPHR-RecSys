"""Fit GF-CF once and evaluate validation only with the shared protocol.

Called by train_model.py --model gfcf. Same retained train sample, warm cohort, mask, candidate set
and Evaluator as run_sgkr.py; the full catalog is kept. The SVD uses the authors' library (sparsesvd).
Run records (plan, source archive, progress, budget) follow run_sgkr.py via scripts/closed_form_run.py.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow  # noqa: F401 -- load Arrow DLLs before Torch on Windows

from scripts.closed_form_run import Run
from scripts.config_io import read_model_config
from src.models.gfcf import GFCF, GFCFConfig


def run(args):
    config = GFCFConfig(**read_model_config(args.config, "gfcf"))
    run = Run("gfcf", args, asdict(config), __file__, solver=f"svd={config.svd_backend}, k={config.singular_vectors}")
    with run.tracked():
        model = GFCF(config).fit(run.x)
        run.progress("fitted", fit_seconds=model.fit_report["seconds"])
        run.guard.atomic_write(run.output / "model.npz", model.save)
        # Reload the artifact before evaluating, as run_sgkr.py does.
        restored = GFCF.load(run.output / "model.npz")
        probe = np.asarray(sorted(run.validation.u_idx.unique())[:64], dtype=np.int64)
        np.testing.assert_allclose(restored.score(probe), model.score(probe), rtol=1e-6, atol=1e-9)
        evaluator, predictions, latency, metrics = run.evaluate(restored.get_user_rating_scores, args.eval_batch_size)
        run.save_predictions(evaluator, predictions)
        run.progress("evaluated", NDCG20=metrics["NDCG@20"])
        return run.finish({"val_metrics": metrics, "config": asdict(config), "fit_report": model.fit_report,
                           "checkpoint_replay": "passed", "inference_latency_ms_per_user": latency,
                           "validation_cohort": run.plan["validation_cohort"]})


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
