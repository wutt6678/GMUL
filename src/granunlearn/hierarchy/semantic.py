"""Semantic hierarchy builder.

Produces chains like::

    pediatric cardiologist → cardiologist → physician → healthcare professional

These are typically Qwen-assisted and must be verified before use.
"""

from __future__ import annotations

from granunlearn.schema import HierarchyLevel

from .base import ChainHierarchy
from .canonicalize import make_canonical_id, normalize


def build_semantic_hierarchy(
    values: list[str],
    prefix: str = "sem",
    metadata: dict | None = None,
) -> ChainHierarchy:
    """Build a semantic hierarchy from an ordered list of values.

    Parameters
    ----------
    values : list[str]
        Values ordered from finest (index 0) to coarsest.
    prefix : str
        Prefix for canonical IDs.
    metadata : dict | None
        Optional metadata dict applied to all levels.

    Returns
    -------
    ChainHierarchy
    """
    if len(values) < 2:
        raise ValueError("Semantic hierarchy requires at least 2 levels")

    levels: list[HierarchyLevel] = []
    for i, val in enumerate(values):
        cid = make_canonical_id(prefix, val)
        parent_id = make_canonical_id(prefix, values[i + 1]) if i + 1 < len(values) else None
        levels.append(HierarchyLevel(
            level=i,
            canonical_id=cid,
            value=val,
            normalized_value=normalize(val),
            parent_id=parent_id,
            # Non-empty metadata: parquet cannot store empty struct columns,
            # and the type marker is genuinely useful downstream.
            metadata=metadata if metadata else {"type": "semantic"},
        ))

    return ChainHierarchy(levels)
