"""Train every model x density run one after another, then summarise and draw the report figures.

  python scripts/run_experiments.py                         # 5 models x 4 densities, seed 42, 200 epochs, no early stop
  python scripts/run_experiments.py --densities 0.75 0.5    # only some densities
  python scripts/run_experiments.py --dry-run               # list the runs and their state

Each run is a separate scripts/train_model.py process (GPU memory is released between runs).
Finished runs (metrics.json present) are skipped, so an interrupted batch can simply be restarted.
A failed run is logged and the batch continues; the figures are drawn from whatever finished.
Progress is appended to results/run_experiments.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_model import DENSITIES, MODELS, run_dir

# Short, closed-form SGKR first; within a density the GPU models follow.
ORDER = ("sgkr", "sphr", "lightgcn", "xsimgcl", "directau")


def log(path, message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(ORDER))
    parser.add_argument("--densities", type=float, nargs="+", choices=DENSITIES, default=list(DENSITIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=None,
                        help="SPHR patience; default = --epochs (no early stopping). Baselines use configs/common.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the batch at the first failed run")
    parser.add_argument("--no-report", action="store_true", help="Skip summarize_results.py and make_report.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    models = [m for m in ORDER if m in args.models]
    runs = [(m, d, s) for d in args.densities for s in args.seeds for m in models]
    done = {run for run in runs if (run_dir(args.results, *run) / "metrics.json").exists()}
    print(f"{len(runs)} runs: {len(done)} already done, {len(runs) - len(done)} to train")
    for model, density, seed in runs:
        state = "done" if (model, density, seed) in done else "todo"
        print(f"  [{state}] {model:9s} density {density:<4} seed {seed}")
    if args.dry_run:
        return 0
    args.results.mkdir(parents=True, exist_ok=True)
    journal = args.results / "run_experiments.log"
    failed = []
    log(journal, f"batch start: {len(runs) - len(done)} runs to train, epochs={args.epochs}")
    for index, (model, density, seed) in enumerate(runs, 1):
        if (model, density, seed) in done:
            continue
        label = f"[{index}/{len(runs)}] {model} density {density} seed {seed}"
        log(journal, f"{label}: start")
        started = time.perf_counter()
        command = [sys.executable, "-u", str(ROOT / "scripts/train_model.py"), "--model", model,
                   "--density", str(density), "--seed", str(seed), "--epochs", str(args.epochs),
                   "--patience", str(args.patience or args.epochs), "--device", args.device,
                   "--results", str(args.results)]
        code = subprocess.run(command, cwd=ROOT).returncode
        minutes = (time.perf_counter() - started) / 60
        if code:
            failed.append((model, density, seed))
            log(journal, f"{label}: FAILED (exit {code}) after {minutes:.1f} min")
            if args.stop_on_error:
                break
        else:
            log(journal, f"{label}: done in {minutes:.1f} min")
    if not args.no_report:
        for script in ("summarize_results.py", "make_report.py"):
            log(journal, f"{script}: start")
            if subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--results", str(args.results)],
                              cwd=ROOT).returncode:
                log(journal, f"{script}: FAILED")
                failed.append(script)
    log(journal, "batch end: " + ("all runs succeeded" if not failed else f"failed: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
