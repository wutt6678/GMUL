"""Entity-level split assignment (Iteration 4).

Used by datasets where each entity has a single image (e.g. MLLMU-Bench
profiles): whole ENTITIES are assigned to train/val/test, unlike the
within-entity image splitting in ``inaturalist.deterministic_image_splits``.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def deterministic_entity_splits(
    entity_ids: Sequence[str],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict[str, str]:
    """Assign each entity to train/val/test deterministically.

    Guarantees (when enough entities exist):
    * at least one test entity  (n >= 2)
    * at least one val entity   (n >= 4)
    * at least one train entity (always; train receives the remainder)

    Assignment is a hash-shuffle of the entity IDs followed by positional
    slicing — reproducible and independent of input order.
    """
    n = len(entity_ids)
    if n == 0:
        return {}

    order = sorted(
        entity_ids,
        key=lambda e: hashlib.sha256(f"{seed}:split:{e}".encode()).hexdigest(),
    )

    n_test = max(1, round(n * (1.0 - train_ratio - val_ratio))) if n >= 2 else 0
    n_val = max(1, round(n * val_ratio)) if n >= 4 else 0
    if n - n_test - n_val < 1:
        n_val = max(0, n - n_test - 1)
    n_train = n - n_test - n_val

    split_of: dict[str, str] = {}
    cursor = 0
    for e in order[cursor:cursor + n_train]:
        split_of[e] = "train"
    cursor += n_train
    for e in order[cursor:cursor + n_val]:
        split_of[e] = "val"
    cursor += n_val
    for e in order[cursor:]:
        split_of[e] = "test"
    return split_of
