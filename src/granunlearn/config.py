"""YAML config loader for GMUL experiments.

Path-resolution conventions
---------------------------
When a config value references another file (e.g. ``model.config``):

* Paths starting with ``configs/`` — resolved relative to the **repository
  root** (the nearest ancestor directory containing ``pyproject.toml``).
* Paths starting with ``./`` — resolved relative to the **current config
  file's directory**.
* Absolute paths — used as-is.

A ``FileNotFoundError`` is raised when the resolved path does not exist.
"""

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


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk upward from *start* looking for the repository root.

    The repo root is identified by the presence of ``pyproject.toml`` (or
    ``.git`` as a fallback).  Returns ``None`` if no root is found.
    """
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def _resolve_ref_path(ref: str, config_dir: Path) -> Path:
    """Resolve a config-reference path according to the project conventions.

    Parameters
    ----------
    ref : str
        The raw path string from the YAML config.
    config_dir : Path
        Directory of the config file that contains *ref*.

    Returns
    -------
    Path
        The fully resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    ref_path = Path(ref)

    if ref_path.is_absolute():
        resolved = ref_path
    elif ref.startswith("configs/") or ref.startswith("configs\\"):
        # Repository-root relative
        repo_root = _find_repo_root(config_dir)
        if repo_root is None:
            raise FileNotFoundError(
                f"Cannot resolve repo-root-relative path {ref!r}: "
                f"no pyproject.toml or .git found above {config_dir}"
            )
        resolved = repo_root / ref_path
    elif ref.startswith("./"):
        # Current-config relative
        resolved = config_dir / ref_path
    else:
        # Treat bare filenames as config-relative (e.g. "base.yaml")
        resolved = config_dir / ref_path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Referenced config not found: {ref!r} resolved to {resolved}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config and apply optional overrides.

    If the loaded config contains a ``model.config`` key pointing to another
    YAML file, that referenced config is loaded first and used as the base.
    See module docstring for path-resolution conventions.

    Raises ``FileNotFoundError`` if a referenced config does not exist.
    """
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)

    # Resolve nested model config reference
    model_cfg_ref = cfg.get("model", {}).get("config")
    if model_cfg_ref:
        ref_path = _resolve_ref_path(model_cfg_ref, config_path.parent)
        base_cfg = load_yaml(ref_path)
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
