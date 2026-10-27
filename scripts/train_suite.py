"""One entry point for validation iteration, final benchmarks, and reports."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bundle import resolve_bundle, sha256_file
from src.evaluation.run_validation import validate_run_result
from src.utils.checkpoints import get_model_output_dir, get_run_fingerprint
from src.utils.config import load_config
from src.utils.paths import (
    PROJECT_ROOT,
    resolve_output_root,
    resolve_path,
    write_run_manifest,
)

MODELS = ("lightgcn", "xsimgcl", "directau")
SUPPORTED_MODELS = MODELS
DENSITIES = (1.0, 0.75, 0.5, 0.25)


def training_source_hashes():
    """Portable source identity for comparisons across separately snapshotted roots."""
    paths = [PROJECT_ROOT / "scripts" / "train.py"]
    for directory in ("models", "data", "training", "losses", "utils", "evaluation"):
        paths.extend((PROJECT_ROOT / "src" / directory).rglob("*.py"))
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in sorted(paths)
            if p.name not in {"training_report.py", "single_run_comparison.py"}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def job_key(model, density, seed):
    return f"{model}_s{round(density * 100)}_seed{seed}"


def result_path(root, job):
    return Path(get_model_output_dir("raw", job["model"], str(root))) / f"{job['id']}.json"


def expected_config(model, config_dir, bundle, density=None):
    config = load_config(model, str(config_dir), density=density)
    return config


def build_jobs(models, densities, seeds):
    # Pair all models at the same density/seed before moving to the next block.
    return [{"id": job_key(model, density, seed), "model": model, "density": density,
             "seed": seed, "status": "pending"}
            for density in densities for seed in seeds for model in models]


def resources(root):
    import torch
    snapshot = {"cpu_logical": os.cpu_count(), "disk_free_gib": round(shutil.disk_usage(root).free / 2**30, 2),
                "cuda_runtime_available": torch.cuda.is_available(), "sequential_workers": 1}
    try:
        import psutil
        snapshot["available_ram_gib"] = round(psutil.virtual_memory().available / 2**30, 2)
    except ImportError:
        snapshot["available_ram_gib"] = None
    if snapshot["cuda_runtime_available"]:
        snapshot["gpu_name"] = torch.cuda.get_device_name(0)
        snapshot["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    return snapshot


def estimate_storage(bundle, models, jobs, config_dir, policy):
    """Conservative checkpoint budget, not a disk reservation or measured peak."""
    stats = bundle.manifest.get("statistics", {})
    if not all(isinstance(stats.get(k), int) for k in ("num_users", "num_items")):
        return None
    nodes = stats["num_users"] + stats["num_items"]
    weights = {m: nodes * load_config(m, str(config_dir))["model"]["embedding_dim"] * 4 * 1.15 for m in models}
    multiplier = 1 if policy == "compact" else 7
    saved = sum(weights[j["model"]] * multiplier for j in jobs)
    globals_size = sum(weights.values()) * (1 if policy == "compact" else 3)
    temporary = max(weights.values()) * 10
    return round((saved + globals_size + temporary + 512 * 2**20) / 2**30, 2)


def create_plan(args, root):
    bundle = resolve_bundle(args.bundle)
    # The active processed view can be replaced. Pin its immutable archive.
    archive = PROJECT_ROOT / "data" / "versions" / bundle.build_id
    if archive.is_dir() and bundle.root.resolve() == (PROJECT_ROOT / "data" / "processed").resolve():
        archived = resolve_bundle(archive)
        def inventory_identity(candidate):
            return sorted((Path(a["path"]).name, a["sha256"], a["size_bytes"])
                          for a in candidate.manifest["artifacts"])
        # Materialization rewrites manifest paths, so compare verified bytes.
        if archived.build_id != bundle.build_id or inventory_identity(archived) != inventory_identity(bundle):
            raise ValueError("Archive artifacts differ from active bundle")
        bundle = archived
    if bundle.legacy:
        raise ValueError("Training suite requires a versioned immutable bundle")
    models = args.models if args.mode == "all" else [args.model]
    densities = args.densities if args.mode == "all" else [args.density]
    seeds = args.seeds if args.mode == "all" else [args.seed]
    config_dir = resolve_path(args.config_dir).resolve()
    files = [config_dir / "common.yaml", *[config_dir / f"{model}.yaml" for model in models]]
    for model in models:
        load_config(model, str(config_dir))
    jobs = build_jobs(models, densities, seeds)
    snapshot = resources(root.parent if root.parent.exists() else PROJECT_ROOT)
    snapshot["estimated_checkpoint_budget_gib"] = estimate_storage(bundle, models, jobs, config_dir, args.checkpoint_policy)
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode, "models": models, "densities": densities, "seeds": seeds,
            "epochs": args.epochs, "validation_only": args.mode == "single" and not args.final_test,
            "bundle": str(bundle.root), "bundle_id": bundle.build_id,
            "manifest_sha256": sha256_file(bundle.manifest_path), "output_root": str(root),
            "checkpoint_policy": args.checkpoint_policy,
            "training_source_hashes": training_source_hashes(),
            "source_config_dir": str(config_dir), "config_hashes": {p.name: sha256_file(p) for p in files},
            "jobs": jobs, "resources": snapshot}


def compact_completed_checkpoints(root, plan, job):
    """Retain inference weights; incomplete jobs always keep full resume state."""
    if plan.get("checkpoint_policy", "full") != "compact":
        return
    import torch
    directory = Path(get_model_output_dir("checkpoints", job["model"], str(root)))
    best = directory / f"{job['id']}.pt"
    for path in (best, directory / f"{job['model']}_best.pt"):
        path.resolve().relative_to(root.resolve())
        if not path.is_file():
            continue
        # These are trusted artifacts produced by this suite's trainer.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if path == best and checkpoint.get("config", {}).get("experiment_fingerprint") != job["fingerprint"]:
            raise ValueError("Best checkpoint identity differs from validated result")
        if checkpoint.get("checkpoint_kind") == "inference_only":
            continue
        compact = {key: checkpoint[key] for key in ("epoch", "best_score", "model_state_dict", "val_metrics", "config") if key in checkpoint}
        compact["checkpoint_kind"] = "inference_only"
        temporary = Path(str(path) + ".compact.tmp")
        torch.save(compact, temporary)
        os.replace(temporary, path)
    if best.is_file():
        latest = directory / f"{job['id']}_latest.pt"
        latest.resolve().relative_to(root.resolve())
        latest.unlink(missing_ok=True)


def pin_fingerprints(plan, root):
    bundle = resolve_bundle(plan["bundle"])
    if sha256_file(bundle.manifest_path) != plan["manifest_sha256"]:
        raise ValueError("Pinned dataset manifest changed")
    config_dir = root / "configs"
    for name, digest in plan["config_hashes"].items():
        if sha256_file(config_dir / name) != digest:
            raise ValueError(f"Frozen config changed: {name}; start a new experiment")
    for job in plan["jobs"]:
        config = expected_config(job["model"], config_dir, bundle, job["density"])
        actual = get_run_fingerprint(job["model"], job["density"], job["seed"], config,
                                     str(config_dir), manifest_path=bundle.manifest_path)
        if job.get("fingerprint", actual) != actual:
            raise ValueError(f"Code/data/config changed for {job['id']}; start a new experiment")
        job["fingerprint"] = actual


def validate_result(plan, job, path):
    data = read_json(path)
    errors = validate_run_result(data, job["model"], job["density"], job["seed"], plan["epochs"],
                                 job["fingerprint"], validation_only=plan["validation_only"])
    if errors:
        raise ValueError(f"Invalid result {path}: {'; '.join(errors)}")
    return data


@contextmanager
def run_lock(root):
    import psutil
    path = root / ".suite.lock"
    if path.exists():
        owner = read_json(path)
        try:
            process = psutil.Process(owner["pid"])
            alive = abs(process.create_time() - owner["created"]) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("This suite is already running")
        path.unlink()
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def execute_job(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    # Bound native CPU threads; train workers remain sequential on one GPU.
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def persist_status(root, plan, status):
    counts = {state: sum(j["status"] == state for j in plan["jobs"])
              for state in ("pending", "running", "succeeded", "failed", "interrupted")}
    write_json(root / "suite_plan.json", plan)
    write_json(root / "runner_status.json", {"status": status, "planned": len(plan["jobs"]), **counts})


def run_plan(root, plan, resume=False):
    from src.evaluation.training_report import export_report
    with run_lock(root):
        pin_fingerprints(plan, root)
        persist_status(root, plan, "running")
        current = None
        try:
            for index, job in enumerate(plan["jobs"], 1):
                current = job
                path = result_path(root, job)
                if resume and path.exists():
                    validate_result(plan, job, path)
                    compact_completed_checkpoints(root, plan, job)
                    job["status"] = "succeeded"
                    persist_status(root, plan, "running")
                    print(f"[{index}/{len(plan['jobs'])}] CACHED {job['id']}", flush=True)
                    continue
                job["status"] = "running"
                persist_status(root, plan, "running")
                cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--model", job["model"],
                       "--sparsity", str(job["density"]), "--seed", str(job["seed"]), "--epochs", str(plan["epochs"]),
                       "--config_dir", str(root / "configs"), "--bundle", plan["bundle"], "--output_root", str(root)]
                if plan["validation_only"]:
                    cmd.append("--validation_only")
                latest = Path(get_model_output_dir("checkpoints", job["model"], str(root))) / f"{job['id']}_latest.pt"
                if resume and latest.is_file():
                    cmd.append("--resume")
                print(f"[{index}/{len(plan['jobs'])}] TRAIN {job['id']} -> {root}", flush=True)
                job["exit_code"] = execute_job(cmd, root / "logs" / f"{job['id']}.log")
                if job["exit_code"]:
                    raise RuntimeError(f"Training failed for {job['id']}; see logs/{job['id']}.log")
                validate_result(plan, job, path)
                compact_completed_checkpoints(root, plan, job)
                job["status"] = "succeeded"
                persist_status(root, plan, "running")
                export_report(root, render=False)
            export_report(root)
            persist_status(root, plan, "complete")
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            if current and current["status"] == "running":
                current["status"] = "interrupted" if interrupted else "failed"
                current["error"] = str(exc) or "KeyboardInterrupt"
            persist_status(root, plan, "interrupted" if interrupted else "failed")
            try:
                export_report(root)
            except Exception as report_error:  # noqa: BLE001 - preserve the original training error.
                print(f"Partial report unavailable: {report_error}", file=sys.stderr)
            raise
    print(f"Report: {root / 'report.html'}\nFigures: {root / 'figures'}", flush=True)


def parser():
    cli = argparse.ArgumentParser(description="Train -> validated results -> CSV + PNG/SVG + offline report")
    modes = cli.add_subparsers(dest="mode", required=True)
    for name in ("all", "single"):
        sub = modes.add_parser(name)
        sub.add_argument("--epochs", type=int, default=100 if name == "all" else 20)
        sub.add_argument("--config-dir", default="configs")
        sub.add_argument("--bundle", default=None)
        sub.add_argument("--output-root")
        sub.add_argument("--dry-run", action="store_true")
        sub.add_argument("--checkpoint-policy", choices=("compact", "full"), default="compact",
                         help="compact keeps inference weights after success; unfinished jobs keep full resume state")
        if name == "all":
            sub.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(MODELS))
            sub.add_argument("--densities", nargs="+", type=float, choices=DENSITIES, default=list(DENSITIES))
            sub.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025])
        else:
            sub.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
            sub.add_argument("--density", type=float, choices=DENSITIES, required=True)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--final-test", action="store_true", help="Evaluate test after validation selects checkpoint")
    for name in ("resume", "report"):
        sub = modes.add_parser(name)
        sub.add_argument("--output-root", required=True)
    return cli


def menu():
    print("1. Train all models / densities / seeds + results + figures\n2. Train one model / one density (validation)\n3. Resume suite\n4. Regenerate report")
    choice = input("Choose [1-4]: ").strip()
    if choice in {"3", "4"}:
        return ["resume" if choice == "3" else "report", "--output-root", input("Output folder: ").strip().strip('"')]
    if choice == "1":
        epochs = input("Maximum epochs [100]: ").strip() or "100"
        return ["all", "--epochs", epochs]
    if choice == "2":
        print("Models: " + ", ".join(MODELS))
        model = input("Model [lightgcn]: ").strip() or "lightgcn"
        density = input("Density 1 / .75 / .5 / .25 [.25]: ").strip() or ".25"
        epochs = input("Maximum epochs [20]: ").strip() or "20"
        seed = input("Seed [42]: ").strip() or "42"
        return ["single", "--model", model, "--density", density, "--epochs", epochs, "--seed", seed]
    raise ValueError("Choose 1, 2, 3 or 4")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cli = parser()
    if argv is None:
        argv = sys.argv[1:]
        if not argv:
            if not sys.stdin.isatty():
                cli.error("Specify all/single/resume/report in non-interactive sessions")
            argv = menu()
    args = cli.parse_args(argv)
    if args.mode in {"resume", "report"}:
        root = resolve_path(args.output_root).resolve()
        if args.mode == "report":
            from src.evaluation.training_report import export_report
            export_report(root)
        else:
            plan = read_json(root / "suite_plan.json")
            if plan["output_root"] != str(root):
                raise ValueError("Resume must use the original output folder")
            run_plan(root, plan, resume=True)
        return
    seeds = args.seeds if args.mode == "all" else [args.seed]
    if args.epochs < 1 or any(s < 0 for s in seeds):
        cli.error("Epochs must be positive and seeds nonnegative")
    if args.mode == "all" and any(len(values) != len(set(values)) for values in (args.models, args.densities, seeds)):
        cli.error("Duplicate models/densities/seeds are not allowed")
    root = resolve_output_root(args.output_root, kind="suite", label=args.mode).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Output folder is not empty; use resume or a new folder")
    plan = create_plan(args, root)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    required = plan["resources"]["estimated_checkpoint_budget_gib"]
    if required is not None and required > plan["resources"]["disk_free_gib"]:
        raise OSError(f"Estimated checkpoint budget {required:.2f} GiB exceeds free disk; reduce sweep or use compact policy")
    root.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir()
    for name in plan["config_hashes"]:
        shutil.copyfile(Path(plan["source_config_dir"]) / name, root / "configs" / name)
    pin_fingerprints(plan, root)
    write_run_manifest(root, kind="benchmark" if args.mode == "all" else "train_all",
                       metadata={k: v for k, v in plan.items() if k != "jobs"})
    write_json(root / "suite_plan.json", plan)
    run_plan(root, plan)


if __name__ == "__main__":
    main()
