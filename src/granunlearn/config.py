"""YAML config loader for GMUL experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple config dicts. Later values override earlier ones."""
    result: dict[str, Any] = {}
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply optional overrides.

    If the loaded config contains a ``model.config`` key pointing to another
    YAML file (relative to the config file's directory), that referenced config
    is loaded first and used as the base.
    """
    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    # Resolve nested model config reference
    model_cfg_path = cfg.get("model", {}).get("config")
    if model_cfg_path:
        base_path = config_path.parent / model_cfg_path
        if base_path.exists():
            base_cfg = load_yaml(base_path)
            cfg = merge_configs(base_cfg, cfg)
            # Remove the reference key so it doesn't cause confusion
            cfg.get("model", {}).pop("config", None)

    if overrides:
        cfg = merge_configs(cfg, overrides)

    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    """Save a config dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path
