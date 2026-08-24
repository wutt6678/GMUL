"""Hierarchy validation gates (spec §7.5).

The build pipeline must **fail** if any blocking error is detected.
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
    issues.extend(_check_monotonic_levels(levels))
    issues.extend(_check_cycles(levels))
    issues.extend(_check_missing_parents(levels))
    issues.extend(_check_normalized_uniqueness(levels))
    return issues


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


def _check_monotonic_levels(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Levels must be strictly increasing (0, 1, 2, …)."""
    issues: list[ValidationIssue] = []
    sorted_levels = sorted(levels, key=lambda l: l.level)
    for i in range(1, len(sorted_levels)):
        prev, curr = sorted_levels[i - 1], sorted_levels[i]
        if curr.level <= prev.level:
            issues.append(ValidationIssue(
                "error", "NON_MONOTONIC_LEVEL",
                f"Level {curr.level} is not strictly greater than previous level {prev.level}",
                node_id=curr.canonical_id,
            ))
    return issues


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


def _check_missing_parents(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Every non-root parent_id must reference an existing node."""
    issues: list[ValidationIssue] = []
    ids = {lv.canonical_id for lv in levels}
    for lv in levels:
        if lv.parent_id is not None and lv.parent_id not in ids:
            issues.append(ValidationIssue(
                "error", "MISSING_PARENT",
                f"parent_id {lv.parent_id!r} not found in hierarchy",
                node_id=lv.canonical_id,
            ))
    return issues


def _check_normalized_uniqueness(levels: list[HierarchyLevel]) -> list[ValidationIssue]:
    """Two different levels must not share the same normalized_value."""
    issues: list[ValidationIssue] = []
    seen: dict[str, int] = {}
    for lv in levels:
        if lv.normalized_value in seen:
            issues.append(ValidationIssue(
                "error", "DUPLICATE_NORMALIZED",
                f"normalized_value {lv.normalized_value!r} appears at levels {seen[lv.normalized_value]} and {lv.level}",
                node_id=lv.canonical_id,
            ))
        seen[lv.normalized_value] = lv.level
    return issues


def assert_valid(issues: list[ValidationIssue]) -> None:
    """Raise ``ValueError`` if any blocking error exists."""
    errors = [i for i in issues if i.is_error]
    if errors:
        msg = "\n".join(str(e) for e in errors)
        raise ValueError(f"Hierarchy validation failed:\n{msg}")
