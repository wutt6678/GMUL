"""Deterministic train / validation / test split assignment.

Splits are entity-level (all images and queries for an entity go into
the same split) to prevent data leakage.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from granunlearn.schema import SplitInfo


def deterministic_split(
    entity_ids: Sequence[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, SplitInfo]:
    """Assign each entity ID to a split deterministically.

    Uses a seeded hash so the assignment is:
    * reproducible across runs
    * independent of list ordering
    * approximately balanced according to the ratios

    Parameters
    ----------
    entity_ids : sequence of str
        Unique entity identifiers.
    train_ratio, val_ratio, test_ratio : float
        Target proportions.  Must sum to ~1.0.
    seed : int
        Salt mixed into the hash for reproducibility control.

    Returns
    -------
    dict[str, SplitInfo]
        Mapping from entity_id to its assigned ``SplitInfo``.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    # Thresholds for the hash bucket [0, 1)
    t_train = train_ratio
    t_val = train_ratio + val_ratio

    assignments: dict[str, SplitInfo] = {}
    for eid in entity_ids:
        h = hashlib.sha256(f"{seed}:{eid}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF  # uniform in [0, 1)
        if bucket < t_train:
            split = "train"
        elif bucket < t_val:
            split = "val"
        else:
            split = "test"
        assignments[eid] = SplitInfo(split=split)

    return assignments


def split_counts(assignments: dict[str, SplitInfo]) -> dict[str, int]:
    """Count entities per split."""
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for info in assignments.values():
        counts[info.split] = counts.get(info.split, 0) + 1
    return counts
