"""Deterministic seeding helper for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to use.  Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic cuDNN behaviour (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed_from_config(config: dict) -> int:
    """Extract the seed from a config dict, defaulting to 42."""
    return config.get("experiment", config).get("seed", 42)
