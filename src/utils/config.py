import math
import os
from copy import deepcopy
from typing import Any, Dict

import yaml

from src.utils.config_schemas import validate_config, validate_model_config
from src.utils.validation_lr_options import validate_validation_scheduler_options


def merge_config(common: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Inherit untouched section fields; explicit model values take precedence."""
    merged = deepcopy(common)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    model_name: str = "lightgcn",
    config_dir: str = "configs",
    validate: bool = True,
    *,
    density: float | None = None,
) -> Dict[str, Any]:
    """Load and merge common.yaml with model specific configuration file.

    Args:
        model_name: Name of the model to load specific config
        config_dir: Directory containing config files
        validate: Whether to validate config values (default: True)

    Returns:
        Validated config dictionary
    """
    if model_name not in {"lightgcn", "xsimgcl", "directau"}:
        raise ValueError(f"Unknown or retired model: {model_name}")
    common_path = os.path.join(config_dir, "common.yaml")
    if not os.path.exists(common_path):
        raise FileNotFoundError(f"Common config file not found at {common_path}")

    with open(common_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config_path = os.path.join(config_dir, f"{model_name}.yaml")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
            if model_cfg:
                config = merge_config(config, model_cfg)

    config["model_name"] = model_name

    validate_density_profiles(config)
    if density is not None:
        config = apply_density_profile(config, density)

    # Apply validation
    if validate:
        config = validate_config(config)
        config = validate_model_config(config, model_name)

    return config


def validate_density_profiles(config: Dict[str, Any]) -> None:
    profiles = config.get("density_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("density_profiles must be a mapping")
    seen = set()
    for key, override in profiles.items():
        if key == "default":
            identity = "default"
        else:
            try:
                identity = float(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("Density profile keys must be ratios or default") from exc
            if isinstance(key, bool) or not math.isfinite(identity) or not 0 < identity <= 1:
                raise ValueError("Density profile ratios must be in (0, 1]")
        if identity in seen:
            raise ValueError("Duplicate equivalent density profile ratios")
        seen.add(identity)
        if not isinstance(override, dict) or not override or set(override) - {"training", "validation_lr_scheduler"}:
            raise ValueError("Density profiles support training and validation_lr_scheduler")
        training = override.get("training", {})
        if not isinstance(training, dict) or set(training) - {"learning_rate", "early_stopping_patience"}:
            raise ValueError("Density training profiles support LR and patience only")
        if "learning_rate" in training:
            lr = training["learning_rate"]
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= 1:
                raise ValueError("Invalid density profile learning_rate")
        if "early_stopping_patience" in training:
            patience = training["early_stopping_patience"]
            if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
                raise ValueError("Invalid density profile early_stopping_patience")
        if "validation_lr_scheduler" in override:
            initial_lr = training.get("learning_rate", config.get("training", {}).get("learning_rate", .0003))
            validate_validation_scheduler_options(override["validation_lr_scheduler"], initial_lr)


def apply_density_profile(config: Dict[str, Any], density: float) -> Dict[str, Any]:
    """Resolve one explicit ratio or fallback; never interpolate tuned settings."""
    if isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(density) or not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    validate_density_profiles(config)
    profiles = config.get("density_profiles", {})
    selected = next((key for key in profiles if key != "default" and float(key) == density), "default")
    resolved = merge_config(config, profiles.get(selected, {}))
    if profiles:
        resolved["density_profile"] = str(selected) if selected in profiles else "base"
    return resolved
