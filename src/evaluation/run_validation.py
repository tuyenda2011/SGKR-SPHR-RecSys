"""Validation of training-result artifacts shared by benchmark runners.

The benchmark runner and the one-density smoke runner must reject the same
classes of stale or partial JSON files.  This module keeps that contract in
one place while allowing validation-only runs to omit final-test metrics.
"""

from collections.abc import Iterable
from numbers import Real
from typing import Any

import numpy as np

from src.evaluation.evaluator import EVALUATION_PROTOCOL

FULL_METRIC_PATHS = (
    ("val_metrics", "NDCG@10"),
    ("test_metrics", "Recall@10"),
    ("test_metrics", "NDCG@10"),
    ("test_metrics", "MRR@10"),
    ("test_metrics", "Recall@20"),
    ("test_metrics", "NDCG@20"),
    ("test_metrics", "Diversity@10"),
    ("test_metrics", "Novelty@10"),
    ("test_metrics", "Coverage@10"),
    ("test_metrics", "Gini@10"),
    ("representation_metrics", "alignment"),
    ("representation_metrics", "mean_uniformity"),
    ("svd_metrics", "user_effective_rank"),
    ("svd_metrics", "item_effective_rank"),
    ("subgroup_metrics", "Tail (Low-Activity)", "Recall@10"),
    ("subgroup_metrics", "Tail (Low-Activity)", "NDCG@10"),
    ("subgroup_metrics", "Head (Active)", "Recall@10"),
    ("subgroup_metrics", "Head (Active)", "NDCG@10"),
)


def _nested_value(data: dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def _validate_metric(errors: list[str], value: Any, path: tuple[str, ...]) -> None:
    label = ".".join(path)
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        errors.append(f"{label} must be a finite number")
        return
    metric = path[-1]
    if metric.startswith(("Recall@", "NDCG@", "MRR@", "Coverage@", "Gini@")) and not 0 <= value <= 1:
        errors.append(f"{label} must be in [0, 1]")
    if metric.startswith("Diversity@") and not 0 <= value <= 2:
        errors.append(f"{label} must be in [0, 2]")
    if (metric.startswith("Novelty@") or metric in {"alignment", "user_effective_rank", "item_effective_rank"}) and value < 0:
        errors.append(f"{label} must be nonnegative")


def validate_run_result(
    data: Any,
    model: str,
    sparsity: float,
    seed: int,
    epochs: int,
    fingerprint: str,
    *,
    validation_only: bool = False,
) -> list[str]:
    """Return reasons a result JSON is unusable.

    Validation-only runs share identity, protocol, monitor and range checks
    with full runs.  They require only validation metrics and deliberately do
    not require test, representation, subgroup or inference fields.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["result must be a JSON object"]

    expected = {
        "model_name": model,
        "sparsity_level": sparsity,
        "seed": seed,
        "max_epochs": epochs,
        "experiment_fingerprint": fingerprint,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    for key, value in expected.items():
        actual = data.get(key)
        if isinstance(actual, bool) or actual != value:
            errors.append(f"{key}={actual!r}, expected {value!r}")

    actual_validation_only = data.get("validation_only", False)
    if validation_only and actual_validation_only is not True:
        errors.append(f"validation_only={actual_validation_only!r}, expected True")
    elif not validation_only and actual_validation_only is not False:
        errors.append(f"validation_only={data.get('validation_only')!r}, expected {validation_only!r}")
    if not isinstance(data.get("experiment_family"), str) or not data["experiment_family"]:
        errors.append("missing experiment_family")
    if data.get("scoring_metric") not in ("dot_product", "cosine"):
        errors.append("invalid scoring_metric")
    if "profile" not in data or "monitor" not in data:
        errors.append("missing profile or monitor")
    if not isinstance(data.get("evaluation_metadata"), dict):
        errors.append("missing evaluation_metadata")
    else:
        metadata = data["evaluation_metadata"]
        required_metadata = {
            "history_mask_policy": "full_train_for_val_full_train_plus_val_for_test",
            "sparsity_scope": "model_training_graph_only",
            "popularity_reference": "full_train_unique_users",
            "subgroup_degree_reference": "full_train_fixed_across_sparsity",
        }
        for key, value in required_metadata.items():
            if metadata.get(key) != value:
                errors.append(f"evaluation_metadata.{key} is invalid")

    required_numbers = ("best_epoch", "total_epochs", "total_train_time", "avg_epoch_time")
    if not validation_only:
        required_numbers += ("inference_latency_ms_per_user", "throughput_users_per_sec")
    for key in required_numbers:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            errors.append(f"{key} must be a finite nonnegative number")

    for key in ("best_epoch", "total_epochs"):
        if type(data.get(key)) is not int:
            errors.append(f"{key} must be an integer")
    if all(type(data.get(key)) is int for key in ("best_epoch", "total_epochs")) and not (
        1 <= data["best_epoch"] <= data["total_epochs"] <= epochs
    ):
        errors.append("best_epoch must be within completed epochs and budget")

    monitor = data.get("monitor", "NDCG@10")
    if not isinstance(monitor, str) or monitor not in {"NDCG@10", "NDCG@20"}:
        errors.append("invalid monitor")
    val_metrics = data.get("val_metrics")
    if not isinstance(val_metrics, dict) or monitor not in val_metrics:
        errors.append("missing validation monitor metric")
    else:
        _validate_metric(errors, val_metrics[monitor], ("val_metrics", monitor))

    # Validate any additional validation metrics present, while requiring only
    # the monitor for a validation-only artifact.
    if isinstance(val_metrics, dict):
        for key, value in val_metrics.items():
            if isinstance(value, Real):
                _validate_metric(errors, value, ("val_metrics", str(key)))

    if not validation_only:
        for path in FULL_METRIC_PATHS:
            try:
                _validate_metric(errors, _nested_value(data, path), path)
            except (KeyError, TypeError, ValueError):
                errors.append(f"missing or invalid {'.'.join(path)}")

    effective = data.get("effective_config")
    if isinstance(effective, dict):
        evaluation = effective.get("evaluation", {})
        if not isinstance(evaluation, dict) or evaluation.get("monitor", "NDCG@10") != monitor:
            errors.append("monitor does not match effective config")
        training = effective.get("training", {})
        if isinstance(training, dict) and training.get("seed", seed) != seed:
            errors.append("seed does not match effective config")

    return errors
