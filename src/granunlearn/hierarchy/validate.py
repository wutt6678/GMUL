"""Hierarchy validation gates (spec §7.5).

The build pipeline must **fail** if any blocking error is detected.

Invariants enforced
-------------------
1. Canonical IDs are unique.
2. Level indices form the exact sequence ``0, 1, 2, …, n-1``.
3. No cycles in parent links.
4. **Strict chain**: ``levels[i].parent_id == levels[i+1].canonical_id``
   and ``levels[-1].parent_id is None``.
5. Normalized values are unique across levels.
"""

from __future__ import annotations

from granunlearn.schema import HierarchyLevel

from .base import ValidationIssue


def validate_chain(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Run all structural checks on an ordered list of hierarchy levels.

    Returns a list of ``ValidationIssue``.  Callers should treat any issue
    with ``severity == "error"`` as fatal.
    """
    issues: list[ValidationIssue] = []
    issues.extend(_check_duplicate_ids(levels))
    issues.extend(_check_level_sequence(levels))
    issues.extend(_check_cycles(levels))
    issues.extend(_check_strict_chain_parents(levels))
    issues.extend(_check_normalized_uniqueness(levels))
    return issues


# -----------------------------------------------------------------------
# 1. Duplicate IDs
# -----------------------------------------------------------------------

def _check_duplicate_ids(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for lv in levels:
        if lv.canonical_id in seen:
            issues.append(ValidationIssue(
                "error", "DUPLICATE_ID",
                f"Duplicate canonical_id: {lv.canonical_id!r}",
                node_id=lv.canonical_id,
            ))
        seen.add(lv.canonical_id)
    return issues


# -----------------------------------------------------------------------
# 2. Exact level sequence 0, 1, 2, …
# -----------------------------------------------------------------------

def _check_level_sequence(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Level indices must form the exact sequence [0, 1, 2, …, n-1]
    **in the order provided**.

    Out-of-order input (e.g. [2, 0, 1]) is rejected.  Downstream code
    (``AssociationRecord.fine_value()``, ``target_value()``) relies on
    positional indexing, so sorting before validation would mask real errors.
    """
    issues: list[ValidationIssue] = []
    actual = [lv.level for lv in levels]
    expected = list(range(len(levels)))
    if actual != expected:
        issues.append(ValidationIssue(
            "error", "LEVEL_SEQUENCE_INVALID",
            f"Level indices {actual} do not match expected sequence {expected} "
            f"(levels must be ordered finest→coarsest)",
        ))
    return issues


# -----------------------------------------------------------------------
# 3. Cycle detection
# -----------------------------------------------------------------------

def _check_cycles(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Detect cycles in the parent chain."""
    issues: list[ValidationIssue] = []
    by_id = {lv.canonical_id: lv for lv in levels}

    for lv in levels:
        visited: set[str] = set()
        current: str | None = lv.canonical_id
        while current is not None:
            if current in visited:
                issues.append(ValidationIssue(
                    "error", "CYCLE_DETECTED",
                    f"Cycle detected involving {current!r}",
                    node_id=current,
                ))
                break
            visited.add(current)
            node = by_id.get(current)
            current = node.parent_id if node else None
    return issues


# -----------------------------------------------------------------------
# 4. Strict chain parents
# -----------------------------------------------------------------------

def _check_strict_chain_parents(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Every level's parent must be the *next coarser* level in the list.

    Uses the raw input order (not sorted).  ``_check_level_sequence``
    guarantees the list is already ordered 0, 1, 2, … when this runs
    in the full validation pipeline.

    Specifically:
    * ``levels[i].parent_id == levels[i+1].canonical_id`` for all i < n-1
    * ``levels[-1].parent_id is None``

    This is the single-chain invariant that simplifies all downstream logic.
    """
    issues: list[ValidationIssue] = []

    for i in range(len(levels) - 1):
        curr = levels[i]
        next_coarser = levels[i + 1]
        if curr.parent_id != next_coarser.canonical_id:
            issues.append(ValidationIssue(
                "error", "CHAIN_PARENT_MISMATCH",
                f"Level {curr.level} ({curr.canonical_id!r}) parent_id is "
                f"{curr.parent_id!r}, expected {next_coarser.canonical_id!r} (level {curr.level + 1})",
                node_id=curr.canonical_id,
            ))

    # Last node in the list must have no parent
    coarsest = levels[-1]
    if coarsest.parent_id is not None:
        issues.append(ValidationIssue(
            "error", "CHAIN_ROOT_HAS_PARENT",
            f"Coarsest level {coarsest.level} ({coarsest.canonical_id!r}) "
            f"has parent_id {coarsest.parent_id!r}, expected None",
            node_id=coarsest.canonical_id,
        ))

    return issues


# -----------------------------------------------------------------------
# 5. Normalized value uniqueness
# -----------------------------------------------------------------------

def _check_normalized_uniqueness(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Two different levels must not share the same normalized_value."""
    issues: list[ValidationIssue] = []
    seen: dict[str, int] = {}
    for lv in levels:
        if lv.normalized_value in seen:
            issues.append(ValidationIssue(
                "error", "DUPLICATE_NORMALIZED",
                f"normalized_value {lv.normalized_value!r} appears at levels "
                f"{seen[lv.normalized_value]} and {lv.level}",
                node_id=lv.canonical_id,
            ))
        seen[lv.normalized_value] = lv.level
    return issues


# -----------------------------------------------------------------------
# Assertion helper
# -----------------------------------------------------------------------

def assert_valid(issues: list[ValidationIssue]) -> None:
    """Raise ``ValueError`` if any blocking error exists."""
    errors = [i for i in issues if i.is_error]
    if errors:
        msg = "\n".join(str(e) for e in errors)
        raise ValueError(f"Hierarchy validation failed:\n{msg}")
