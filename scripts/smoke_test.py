"""Real-data smoke test: run each reported model's own command for a short run (default 50 epochs) and summarise.

Validation only (never reads the test split). Outputs go to a fresh folder under results/smoke/.
Use it before long runs to confirm that every command, checkpoint and validation step works.

  python scripts/smoke_test.py                                 # five models at all four densities, 50 epochs
  python scripts/smoke_test.py --epochs 1                      # quickest check that every command runs
  python scripts/smoke_test.py --density 1.0 --models sgkr sphr
  python scripts/smoke_test.py --dry-run             # print the commands only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lightgcn", "xsimgcl", "directau", "sgkr", "sphr")
DENSITIES = (1.0, 0.75, 0.5, 0.25)


def commands(model, density, seed, folder, epochs=50):
    """The model's real command(s), with a few epochs (no early stopping), and how to read its validation NDCG@20."""
    python = [sys.executable, "-u"]
    tag = f"s{int(density * 100)}"
    if model in ("lightgcn", "xsimgcl", "directau"):
        checkpoint = folder / "checkpoints" / model / f"{model}_{tag}_seed{seed}.pt"
        return ([python + [str(ROOT / "scripts/train_suite.py"), "single", "--model", model, "--density", str(density),
                           "--seed", str(seed), "--epochs", str(epochs), "--checkpoint-policy", "compact",
                           "--output-root", str(folder)]],
                lambda: _checkpoint_metric(checkpoint))
    if model == "sgkr":
        return ([python + [str(ROOT / "scripts/run_sgkr.py"), "--sparsity", str(density), "--density-seed", "42",
                           "--output", str(folder)]],
                lambda: json.loads((folder / "result.json").read_text(encoding="utf-8"))["val_metrics"]["NDCG@20"])
    return ([python + [str(ROOT / "scripts/tune_sphr.py"), "--output", str(folder), "--sparsity", str(density),
                       "--seed", str(seed), "--epochs", str(epochs), "--patience", str(epochs), "--run"]],
            lambda: json.loads((folder / f"seed{seed}" / "completed.json").read_text(encoding="utf-8"))["val_metrics"]["NDCG@20"])


def _checkpoint_metric(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["val_metrics"]["NDCG@20"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--density", type=float, nargs="+", choices=DENSITIES, default=list(DENSITIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per gradient-trained model (SGKR: closed form)")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to results/smoke/<timestamp>")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args(argv)
    root = (args.output or ROOT / "results/smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    rows = []
    for density, model in [(density, model) for density in args.density for model in args.models]:
        folder = root / f"d{round(density * 100)}" / model
        steps, metric = commands(model, density, args.seed, folder, args.epochs)
        if args.dry_run:
            for step in steps:
                print(" ".join(step))
            continue
        started = time.perf_counter()
        status, ndcg = "ok", None
        print(f"\n=== {model} (density {density}) -> {folder}", flush=True)
        for step in steps:
            if subprocess.run(step, cwd=ROOT).returncode != 0:
                status = "FAILED"
                break
        if status == "ok":
            try:
                ndcg = float(metric())
            except Exception as error:  # a run that finished but left no readable metric is a failure
                status = f"FAILED ({type(error).__name__})"
        rows.append((model, density, status, ndcg, time.perf_counter() - started))
    if args.dry_run:
        return 0
    print(f"\nSmoke test, {args.epochs} epochs (SGKR: full closed-form fit) -> {root}")
    print(f"{'model':10s} {'density':>7s} {'status':22s} {'val NDCG@20':>12s} {'seconds':>9s}")
    for model, density, status, ndcg, seconds in rows:
        print(f"{model:10s} {density:7.2f} {status:22s} {'-' if ndcg is None else f'{ndcg:.5f}':>12s} {seconds:9.1f}")
    return 0 if all(row[2] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
