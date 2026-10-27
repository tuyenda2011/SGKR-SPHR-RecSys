"""Centralized checkpoint path utilities for consistent checkpoint management across the project."""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _active_manifest_path() -> Path:
    """Resolve the manifest pinned by current.json without legacy fallback."""
    pointer = REPO_ROOT / "data" / "current.json"
    if not pointer.exists():
        return REPO_ROOT / "data" / "manifest.json"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        relative = payload["bundle_path"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Active dataset pointer is invalid") from exc
    manifest = (REPO_ROOT / "data" / relative).resolve() / "manifest.json"
    if REPO_ROOT / "data" not in manifest.parents or not manifest.is_file():
        raise RuntimeError("Active dataset manifest is missing")
    expected = payload.get("manifest_sha256")
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("Active dataset manifest checksum does not match current.json")
    return manifest


def get_experiment_fingerprint(
    model_name: str,
    config_dir: str = "configs",
    manifest_path: str | Path | None = None,
) -> str:
    """Hash the data manifest, model config, and core implementation files."""
    paths: list[str | Path] = [
        Path(manifest_path).resolve() if manifest_path is not None else _active_manifest_path(),
        os.path.join(config_dir, "common.yaml"),
        os.path.join(config_dir, f"{model_name}.yaml"),
        REPO_ROOT / "src" / "models" / f"{model_name}.py",
        os.path.join("src", "models", "base.py"),
        os.path.join("scripts", "train.py"),
        os.path.join("src", "training", "trainer.py"),
        os.path.join("src", "training", "early_stopping.py"),
        os.path.join("src", "evaluation", "evaluator.py"),
        os.path.join("src", "evaluation", "metrics.py"),
        os.path.join("src", "evaluation", "representation.py"),
        os.path.join("src", "evaluation", "subgroup.py"),
        os.path.join("src", "data", "graph.py"),
        os.path.join("src", "data", "negative_collector.py"),
        os.path.join("src", "data", "splitter.py"),
        os.path.join("src", "data", "sparsity.py"),
        os.path.join("src", "data", "text_encoder.py"),
        os.path.join("src", "data", "provenance.py"),
        os.path.join("src", "data", "preprocessing.py"),
        os.path.join("src", "training", "loss_strategies.py"),
        os.path.join("src", "training", "validation_lr.py"),
        os.path.join("src", "utils", "validation_lr_options.py"),
        os.path.join("src", "utils", "config.py"),
        os.path.join("src", "utils", "config_schemas.py"),
        os.path.join("src", "utils", "geometry.py"),
        os.path.join("src", "utils", "device.py"),
        os.path.join("src", "utils", "seed.py"),
        os.path.join("src", "losses", "bpr.py"),
        os.path.join("src", "losses", "contrastive.py"),
        os.path.join("src", "losses", "directau.py"),
        os.path.join("src", "losses", "debiased_infonce.py"),
        os.path.join("src", "losses", "hard_bpr.py"),
    ]
    digest = hashlib.sha256()
    for path in paths:
        target = path if isinstance(path, Path) else REPO_ROOT / path
        try:
            display_path = target.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display_path = target.as_posix()
        digest.update(display_path.encode("utf-8"))
        with target.open("rb") as file:
            digest.update(file.read())
    return digest.hexdigest()


def get_model_output_dir(section: str, model_name: str, root: str = "results") -> str:
    path = os.path.join(root, section, model_name)
    return path


def get_run_fingerprint(
    model_name,
    sparsity=1.0,
    seed=42,
    config=None,
    config_dir="configs",
    manifest_path: str | Path | None = None,
):
    """Hash learning settings, excluding epoch budget and output paths.

    Pass seed=None for a family identity shared by independent seeds.
    """
    if config is None:
        from src.utils.config import load_config
        config = load_config(model_name, config_dir, density=sparsity)
    from src.utils.config import apply_density_profile
    effective = deepcopy(config) if "density_profile" in config else apply_density_profile(config, sparsity)
    for key in ("experiment_fingerprint", "history_dir", "validation_only", "ablation_variant", "diagnostic_checkpoint_epochs"):
        effective.pop(key, None)
    dataset = effective.get("dataset")
    if isinstance(dataset, dict):
        for key in (
            "processed_dir",
            "bundle_pointer",
            "metadata_overrides_path",
            "bundle_id",
            "bundle_path",
        ):
            dataset.pop(key, None)
    effective.setdefault("training", {}).pop("epochs", None)
    effective["training"]["seed"] = seed
    identity = {"code_and_data": get_experiment_fingerprint(model_name, config_dir, manifest_path),
                "config": effective, "sparsity": sparsity, "seed": seed}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_run_status(root, planned, succeeded, attempted, status=None, current_model=None):
    """Persist counts even when a later run fails or the process is interrupted."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "runner_status.json")
    payload = {"planned": planned, "succeeded": succeeded,
               "failed": attempted - succeeded, "pending": planned - attempted}
    if status is not None:
        payload["status"] = status
    if current_model is not None:
        payload["current_model"] = current_model
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(path + ".tmp", path)


def get_checkpoint_dir(model_name: str, root: str = "results") -> str:
    """Get the checkpoint directory for a given model.

    Args:
        model_name: Name of the model (e.g., 'lightgcn', 'recflow')

    Returns:
        Absolute path to the checkpoint directory
    """
    return get_model_output_dir("checkpoints", model_name, root)


def get_checkpoint_path(
    model_name: str,
    sparsity: float = 1.0,
    seed: int = 42,
    checkpoint_type: str = "run",
    root: str = "results",
) -> str:
    """Get the standardized checkpoint path for a model run.

    Args:
        model_name: Name of the model
        sparsity: Sparsity ratio (0.25 to 1.0)
        seed: Random seed
        checkpoint_type: Type of checkpoint - 'run' (per-sparsity/seed) or 'best' (global best)
        root: Output root containing the model checkpoint section

    Returns:
        Absolute path to the checkpoint file
    """
    checkpoint_dir = get_checkpoint_dir(model_name, root)
    sparsity_tag = f"s{int(sparsity * 100)}"

    if checkpoint_type == "best":
        return os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    else:
        return os.path.join(checkpoint_dir, f"{model_name}_{sparsity_tag}_seed{seed}.pt")


def find_checkpoint(
    model_name: str,
    sparsity: float = 1.0,
    seed: int = 42,
    root: str = "results",
) -> str | None:
    """Find an existing checkpoint for the model, checking multiple possible locations.

    Priority order:
    1. Run-specific checkpoint (model_s{tag}_seed{seed}.pt)
    2. Global best checkpoint (model_best.pt)
    3. Legacy checkpoint locations (backwards compatibility)

    Args:
        model_name: Name of the model
        sparsity: Sparsity ratio
        seed: Random seed
        root: Output root to search before legacy locations

    Returns:
        Path to the found checkpoint, or None if not found
    """
    candidates = [
        # Priority 1: Run-specific checkpoint
        get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="run", root=root),
        # Priority 2: Global best checkpoint
        get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="best", root=root),
    ]

    # New runs live below results/runs/<run_id>. Search the newest matching run
    # before falling back to the historical flat layout.
    if root == "results":
        from src.utils.paths import find_latest_run_root

        latest_root = find_latest_run_root(model_name=model_name, sparsity=sparsity, seed=seed)
        if latest_root is not None:
            candidates = [
                get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="run", root=str(latest_root)),
                get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="best", root=str(latest_root)),
            ] + candidates

    candidates += [
        # Priority 3: Legacy paths for backwards compatibility
        os.path.join("results", "checkpoints", f"{model_name}_best.pt"),
        os.path.join("results", "checkpoints", f"{model_name}_s{int(sparsity * 100)}_seed{seed}.pt"),
        os.path.join("artifacts", "checkpoints", f"{model_name}_s{int(sparsity * 100)}_seed{seed}.pt"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def ensure_checkpoint_dir(model_name: str, root: str = "results") -> str:
    """Ensure the checkpoint directory exists and return its path.

    Args:
        model_name: Name of the model

    Returns:
        Path to the checkpoint directory
    """
    checkpoint_dir = get_checkpoint_dir(model_name, root)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
