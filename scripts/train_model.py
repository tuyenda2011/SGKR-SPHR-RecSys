"""Train one of the five reported models and file its outputs in one layout.

  python scripts/train_model.py --model xsimgcl --density 1.0 --seed 42 --epochs 200

Every run goes to results/<model>/d<density%>/seed<seed>/ with the same top-level files:
  metrics.json          validation metrics, best epoch, epochs run, timing, configuration (same schema for all)
  val_predictions.npy   validation top-20 per user;  val_users.npy  the matching user IDs
  history.csv           per epoch: epoch, loss, NDCG@10/20, Recall@10/20, seconds, learning_rate, then extra_*
                        (the four gradient-trained models)
  checkpoint.pt / checkpoint.npz
  logs/                 the model runner's own outputs (provenance, source archive, budgets)
Validation only; the test split is never read. A finished run (metrics.json present) is not repeated.
The model runners are unchanged: train_suite.py (baselines), run_sgkr.py, tune_sphr.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

BASELINES = ("lightgcn", "xsimgcl", "directau")
MODELS = (*BASELINES, "sgkr", "sphr")
DENSITIES = (1.0, 0.75, 0.5, 0.25)


# history.csv: the same leading columns for every gradient-trained model; model-specific columns follow as extra_*.
HISTORY_COLUMNS = ("epoch", "loss", "NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "seconds", "learning_rate")
HISTORY_RENAME = {"loss_total": "loss", "val_ndcg_20": "NDCG@20", "val_recall_20": "Recall@20",
                  "val_ndcg_10": "NDCG@10", "val_recall_10": "Recall@10", "epoch_time_sec": "seconds",
                  "lr_group_0": "learning_rate"}


def normalize_history(history):
    """Common column names for all models (idempotent, so old runs can be read the same way)."""
    history = history.rename(columns={k: v for k, v in HISTORY_RENAME.items() if v not in history})
    common = [c for c in HISTORY_COLUMNS if c in history]
    extra = {c: c if c.startswith("extra_") else f"extra_{c}" for c in history if c not in common}
    return history[common + list(extra)].rename(columns=extra)


def environment():
    """Hardware and software used for a run (for the experimental-setup section of the report)."""
    import platform
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"created": pd.Timestamp.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor(), "os": platform.platform(),
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None}


def run_dir(results, model, density, seed):
    return Path(results) / model / f"d{round(density * 100)}" / f"seed{seed}"


def link(source, target):
    """Hard link when possible (no extra disk space), otherwise copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def baseline_complete(folder):
    status = folder / "runner_status.json"
    return status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"


def train_baseline(model, density, seed, epochs, folder):
    """Train with the existing suite runner (patience comes from configs/common.yaml).

    An interrupted run continues from its per-epoch _latest.pt checkpoint (train_suite.py resume).
    """
    if (folder / "suite_plan.json").is_file():
        plan = json.loads((folder / "suite_plan.json").read_text(encoding="utf-8"))
        if plan["epochs"] != epochs:
            raise ValueError(f"{folder} was started with {plan['epochs']} epochs, not {epochs}; use a fresh results folder")
        command = [sys.executable, "-u", str(ROOT / "scripts/train_suite.py"), "resume", "--output-root", str(folder)]
    else:
        if folder.exists():
            shutil.rmtree(folder)  # interrupted before the plan was written: nothing was trained yet
        command = [sys.executable, "-u", str(ROOT / "scripts/train_suite.py"), "single", "--model", model,
                   "--density", str(density), "--seed", str(seed), "--epochs", str(epochs),
                   "--checkpoint-policy", "compact", "--output-root", str(folder)]
    if subprocess.run(command, cwd=ROOT).returncode:
        raise RuntimeError(f"{model} training failed; see {folder / 'logs'}")


def finish_baseline(model, density, seed, logs, out, device):
    from scripts import rescore_neural_checkpoints as rescore
    job = f"{model}_s{round(density * 100)}_seed{seed}"
    checkpoint = logs / "train" / "checkpoints" / model / f"{job}.pt"
    raw = json.loads((logs / "train" / "raw" / model / f"{job}.json").read_text(encoding="utf-8"))
    # train.py samples the retained edges with the training seed; score with the same sample.
    args = SimpleNamespace(bundle=None, sparsity=density, density_seed=seed, eval_batch_size=256)
    scored = rescore.rescore(checkpoint, model, logs / "rescore", args, device)
    for name in ("val_predictions.npy", "val_users.npy"):
        link(logs / "rescore" / name, out / name)
    link(checkpoint, out / "checkpoint.pt")
    history = pd.read_csv(logs / "train" / "history" / model / f"{job}_history.csv")
    normalize_history(history).to_csv(out / "history.csv", index=False)
    return {"val_metrics": scored["val_metrics"], "best_epoch": raw["best_epoch"], "epochs_run": raw["total_epochs"],
            "train_seconds": raw["total_train_time"], "inference_ms_per_user": scored["inference_latency_ms_per_user"],
            "checkpoint": "checkpoint.pt", "config": raw["effective_config"]}


def run_sgkr_model(density, folder):
    from scripts import run_sgkr
    run_sgkr.run(run_sgkr.parser().parse_args(["--sparsity", str(density), "--density-seed", "42", "--output", str(folder)]))


def finish_sgkr(logs, out):
    result = json.loads((logs / "sgkr" / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((logs / "sgkr" / "plan.json").read_text(encoding="utf-8"))
    for name in ("val_predictions.npy", "val_users.npy"):
        link(logs / "sgkr" / name, out / name)
    link(logs / "sgkr" / "model.npz", out / "checkpoint.npz")
    return {"val_metrics": result["val_metrics"], "best_epoch": None, "epochs_run": 0,
            "train_seconds": result["fit_report"]["seconds"], "inference_ms_per_user": result["inference_latency_ms_per_user"],
            "checkpoint": "checkpoint.npz", "config": plan["config"]}


def run_sphr_model(density, seed, epochs, patience, device, folder):
    from scripts import tune_sphr
    args = tune_sphr.make_parser().parse_args(["--output", str(folder), "--sparsity", str(density), "--seed", str(seed),
                                               "--epochs", str(epochs), "--patience", str(patience), "--device", device])
    tune_sphr.execute(args, tune_sphr.prepare(args))


def finish_sphr(seed, logs, out):
    result = json.loads((logs / "sphr" / f"seed{seed}" / "completed.json").read_text(encoding="utf-8"))
    attempt = (logs / "sphr" / result["checkpoint"]).parent
    for name in ("val_predictions.npy", "val_users.npy"):
        link(attempt / name, out / name)
    link(attempt / "best.pt", out / "checkpoint.pt")
    normalize_history(pd.read_csv(attempt / "history.csv")).to_csv(out / "history.csv", index=False)
    return {"val_metrics": result["val_metrics"], "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "train_seconds": result["seconds"], "inference_ms_per_user": result["inference_latency_ms_per_user"],
            "checkpoint": "checkpoint.pt", "config": result["config"]}


def train_model(model, density, seed, epochs=200, patience=None, device="auto", results=ROOT / "results"):
    # No early stopping by default: SPHR patience equals the epoch budget (baselines: configs/common.yaml).
    patience = epochs if patience is None else patience
    out = run_dir(results, model, density, seed).resolve()
    if (out / "metrics.json").exists():
        print(f"{model} d{round(density * 100)} seed{seed}: already complete -> {out}", flush=True)
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                "cpu" if device == "auto" else device)
    if model in BASELINES:
        if not baseline_complete(logs / "train"):
            train_baseline(model, density, seed, epochs, logs / "train")
        if (logs / "rescore").exists():
            shutil.rmtree(logs / "rescore")  # a partial scoring pass is recomputed
        summary = finish_baseline(model, density, seed, logs, out, torch_device)
    elif model == "sgkr":
        if not (logs / "sgkr" / "result.json").exists():
            if (logs / "sgkr").exists():
                shutil.rmtree(logs / "sgkr")  # an interrupted closed-form fit restarts from scratch
            run_sgkr_model(density, logs / "sgkr")
        summary = finish_sgkr(logs, out)
    else:
        run_sphr_model(density, seed, epochs, patience, device, logs / "sphr")
        summary = finish_sphr(seed, logs, out)
    metrics = {"model": model, "density": density, "seed": seed, "split": "validation",
               "max_epochs": None if model == "sgkr" else epochs, **summary, "env": environment()}
    users, predictions = np.load(out / "val_users.npy"), np.load(out / "val_predictions.npy")
    if predictions.shape != (len(users), 20):
        raise AssertionError("Unexpected validation prediction shape")
    temporary = out / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(out / "metrics.json")
    print(f"{model} d{round(density * 100)} seed{seed}: NDCG@20={metrics['val_metrics']['NDCG@20']:.5f} -> {out}", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--density", type=float, choices=DENSITIES, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200, help="Epochs to train (ignored by the closed-form SGKR)")
    parser.add_argument("--patience", type=int, default=None,
                        help="SPHR early-stopping patience; default = --epochs (no early stopping). "
                             "Baselines read theirs from configs/common.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.patience is not None and args.patience < 1) or args.seed < 0:
        parser.error("epochs and patience must be positive, seed nonnegative")
    train_model(args.model, args.density, args.seed, args.epochs, args.patience, args.device, args.results)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
