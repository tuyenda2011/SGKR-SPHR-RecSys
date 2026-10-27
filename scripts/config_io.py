"""Read and save YAML configuration mappings for experiment scripts."""
from pathlib import Path

import yaml
from src.utils.source_identity import source_identity

SOURCE_SHA256 = source_identity(__file__)


def read_yaml(path):
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"YAML configuration required: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return config


def read_model_config(path, model):
    """Read a flat model file or its section in the unified training recipe."""
    config = read_yaml(path)
    if "model_configs" not in config:
        return config
    models = config["model_configs"]
    if not isinstance(models, dict) or not isinstance(models.get(model), dict):
        raise ValueError(f"Missing model configuration section: {model}")
    return dict(models[model])


def write_yaml(path, config):
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"} or not isinstance(config, dict):
        raise ValueError("YAML path and a configuration mapping are required")
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
