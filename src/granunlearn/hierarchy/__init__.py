"""Hierarchy engine — build, validate, and query value hierarchies.

Re-exports::

    from granunlearn.hierarchy import (
        ChainHierarchy, ValueHierarchy, ValidationIssue,
        build_semantic_hierarchy, build_taxonomic_hierarchy,
        build_date_hierarchy, build_salary_hierarchy, build_height_hierarchy,
        normalize, make_canonical_id,
    )
"""

from .base import ChainHierarchy, ValidationIssue, ValueHierarchy
from .canonicalize import make_canonical_id, normalize
from .numeric import (
    build_binned_hierarchy,
    build_date_hierarchy,
    build_height_hierarchy,
    build_salary_hierarchy,
)
from .semantic import build_semantic_hierarchy
from .taxonomy import build_taxonomic_hierarchy
from .validate import assert_valid, validate_chain

__all__ = [
    "ChainHierarchy",
    "ValidationIssue",
    "ValueHierarchy",
    "assert_valid",
    "build_binned_hierarchy",
    "build_date_hierarchy",
    "build_height_hierarchy",
    "build_salary_hierarchy",
    "build_semantic_hierarchy",
    "build_taxonomic_hierarchy",
    "make_canonical_id",
    "normalize",
    "validate_chain",
]
