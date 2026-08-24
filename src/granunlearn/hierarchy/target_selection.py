"""Deterministic target-level selection (Iteration 4).

The target level is the granularity the model should RETAIN after
unlearning.  Selection must be fully deterministic and reproducible:
same (seed, key) → same level, always inside [1, n_levels - 1].

Level 0 is never a valid target (that would mean forgetting nothing),
and the coarsest level is allowed (retain only the most abstract value).
"""

from __future__ import annotations

import hashlib


def select_target_level(seed: int, key: str, n_levels: int) -> int:
    """Pick a target level deterministically in ``[1, n_levels - 1]``.

    Parameters
    ----------
    seed : int
        Experiment seed.
    key : str
        Stable identifier for the association, e.g.
        ``f"{entity_id}:{attribute_name}"``.
    n_levels : int
        Total number of hierarchy levels (must be >= 2).
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return 1 + int(digest, 16) % (n_levels - 1)
