"""Numeric hierarchy builders — deterministic abstractions for dates, salary, height, etc.

These builders must **never** use LLM-generated labels.
"""

from __future__ import annotations

import datetime
from typing import Any, Sequence

from granunlearn.schema import HierarchyLevel

from .base import ChainHierarchy
from .canonicalize import make_canonical_id, normalize


# ---------------------------------------------------------------------------
# Bin validation
# ---------------------------------------------------------------------------

def _validate_bins(bins: Sequence[Sequence[int | float | None]]) -> None:
    """Validate that bins are well-formed: no reversal, overlap, or gaps.

    Raises ``ValueError`` on any structural defect.
    """
    if len(bins) == 0:
        raise ValueError("At least one bin is required")

    for i, b in enumerate(bins):
        if len(b) != 2:
            raise ValueError(f"Bin {i} must be a [lo, hi] pair, got length {len(b)}")
        lo, hi = b
        if lo is not None and hi is not None:
            if lo >= hi:
                raise ValueError(
                    f"Bin {i} has reversed or zero-width boundaries: "
                    f"[{lo}, {hi}]"
                )

    # Check consecutive bins: prev.hi must equal next.lo (no gaps, no overlap)
    for i in range(len(bins) - 1):
        _, prev_hi = bins[i]
        next_lo, _ = bins[i + 1]
        if prev_hi is None:
            raise ValueError(
                f"Bin {i} has hi=None but is not the last bin — "
                f"subsequent bins would be unreachable"
            )
        if next_lo is None:
            raise ValueError(
                f"Bin {i+1} has lo=None but is not the first bin"
            )
        if prev_hi != next_lo:
            raise ValueError(
                f"Gap or overlap between bins {i} and {i+1}: "
                f"bin {i} hi={prev_hi}, bin {i+1} lo={next_lo}"
            )


# ---------------------------------------------------------------------------
# Date hierarchy
# ---------------------------------------------------------------------------

def build_date_hierarchy(
    date_str: str,
    prefix: str = "date",
) -> ChainHierarchy:
    """Build a date hierarchy: exact date → year → decade.

    Parameters
    ----------
    date_str : str
        ISO date string ``YYYY-MM-DD`` or ``YYYY``.
    prefix : str
        Canonical ID prefix.

    Example::

        build_date_hierarchy("1994-08-16")
        # level 0: 1994-08-16
        # level 1: 1994
        # level 2: 1990s
    """
    date_str = date_str.strip()

    # Try full date parsing first (YYYY-MM-DD)
    parsed_date: datetime.date | None = None
    try:
        parsed_date = datetime.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        pass

    if parsed_date is not None:
        year = str(parsed_date.year)
        values: list[tuple[str, str]] = [
            (date_str, f"{prefix}:exact"),
            (year, f"{prefix}:year"),
        ]
    else:
        # Try year-only (YYYY)
        if len(date_str) == 4 and date_str.isdigit():
            year = date_str
            # Validate it's a real year
            try:
                datetime.date(int(year), 1, 1)
            except ValueError:
                raise ValueError(f"Cannot parse date: {date_str!r}")
            values = [
                (year, f"{prefix}:year"),
            ]
        else:
            raise ValueError(f"Cannot parse date: {date_str!r}")

    # Decade level (always added)
    decade_start = (int(year) // 10) * 10
    values.append((f"{decade_start}s", f"{prefix}:decade"))

    levels: list[HierarchyLevel] = []
    for i, (val, _cid_hint) in enumerate(values):
        cid = make_canonical_id(prefix, val)
        parent_id = make_canonical_id(prefix, values[i + 1][0]) if i + 1 < len(values) else None
        levels.append(HierarchyLevel(
            level=i,
            canonical_id=cid,
            value=val,
            normalized_value=normalize(val),
            parent_id=parent_id,
            metadata={"type": "date"},
        ))

    return ChainHierarchy(levels)


# ---------------------------------------------------------------------------
# Binned numeric hierarchy (salary, height, etc.)
# ---------------------------------------------------------------------------

def build_binned_hierarchy(
    value: float,
    bins: Sequence[Sequence[int | float | None]],
    value_label: str | None = None,
    prefix: str = "num",
    unit: str = "",
    include_broad_category: bool = False,
) -> ChainHierarchy:
    """Build a numeric hierarchy: exact value → bin (→ broad category).

    Parameters
    ----------
    value : float
        The fine-grained numeric value.
    bins : sequence of [lo, hi] pairs
        Bin boundaries.  ``hi=None`` means unbounded above.
        Bins must be non-overlapping, contiguous, and cover the value.
    value_label : str | None
        Human-readable label for the exact value (e.g. "$87,500").
        Defaults to ``str(value)``.
    prefix : str
        Canonical ID prefix.
    unit : str
        Unit suffix (e.g. "USD", "cm").
    include_broad_category : bool
        If True, appends a coarsest "lower_range"/"upper_range" level.
        DEFAULT FALSE: relative range labels carry no units or numeric
        bounds and are not natural numeric abstractions, so they must not
        be used as experimental target values.  A third level is only
        legitimate when it is itself an explicit numeric super-range.

    Returns
    -------
    ChainHierarchy
        Level 0 = exact value, Level 1 = bin label (and optionally
        Level 2 = broad category).

    Raises
    ------
    ValueError
        If bins are malformed, overlapping, have gaps, or don't contain the value.
    """
    _validate_bins(bins)

    if value_label is None:
        value_label = f"{value:g}"

    # Find the containing bin
    containing_bin: tuple[int | float | None, int | float | None] | None = None
    bin_index: int | None = None
    for i, (lo, hi) in enumerate(bins):
        lo_f = float(lo) if lo is not None else float("-inf")
        hi_f = float(hi) if hi is not None else float("inf")
        if lo_f <= value < hi_f:
            containing_bin = (lo, hi)
            bin_index = i
            break

    if containing_bin is None:
        raise ValueError(f"Value {value} does not fall in any configured bin")

    # Format bin label
    lo, hi = containing_bin
    lo_str = f"{lo:g}" if lo is not None else "0"
    hi_str = f"{hi:g}" if hi is not None else "∞"
    bin_label = f"{lo_str}–{hi_str}"
    if unit:
        bin_label += f" {unit}"

    raw_values = [
        value_label,
        bin_label,
    ]
    if include_broad_category:
        # Broad category: lower half vs upper half (not a numeric abstraction)
        mid = len(bins) / 2
        raw_values.append("lower_range" if bin_index < mid else "upper_range")

    levels: list[HierarchyLevel] = []
    for i, val in enumerate(raw_values):
        cid = make_canonical_id(prefix, val)
        parent_id = make_canonical_id(prefix, raw_values[i + 1]) if i + 1 < len(raw_values) else None
        meta: dict[str, Any] = {"type": "numeric"}
        if i == 0:
            meta["raw_value"] = value
        if i == 1:
            meta["bin_lo"] = lo
            meta["bin_hi"] = hi
            meta["bin_index"] = bin_index
        levels.append(HierarchyLevel(
            level=i,
            canonical_id=cid,
            value=val,
            normalized_value=normalize(val),
            parent_id=parent_id,
            metadata=meta,
        ))

    return ChainHierarchy(levels)


# ---------------------------------------------------------------------------
# Height hierarchy (convenience wrapper)
# ---------------------------------------------------------------------------

_DEFAULT_HEIGHT_BINS: list[list[int | float | None]] = [
    [0, 160],
    [160, 170],
    [170, 180],
    [180, 190],
    [190, None],
]


def build_height_hierarchy(
    cm: float,
    bins: Sequence[Sequence[int | float | None]] | None = None,
    prefix: str = "height",
) -> ChainHierarchy:
    """Build a height hierarchy: exact cm → cm range (two levels)."""
    return build_binned_hierarchy(
        value=cm,
        bins=bins or _DEFAULT_HEIGHT_BINS,
        value_label=f"{cm:g} cm",
        prefix=prefix,
        unit="cm",
    )


# ---------------------------------------------------------------------------
# Salary hierarchy (convenience wrapper)
# ---------------------------------------------------------------------------

_DEFAULT_SALARY_BINS: list[list[int | float | None]] = [
    [0, 25_000],
    [25_000, 50_000],
    [50_000, 75_000],
    [75_000, 100_000],
    [100_000, 150_000],
    [150_000, None],
]


def build_salary_hierarchy(
    amount: float,
    bins: Sequence[Sequence[int | float | None]] | None = None,
    prefix: str = "salary",
) -> ChainHierarchy:
    """Build a salary hierarchy: exact amount → salary band (two levels)."""
    return build_binned_hierarchy(
        value=amount,
        bins=bins or _DEFAULT_SALARY_BINS,
        value_label=f"${amount:,.0f}",
        prefix=prefix,
        unit="USD",
    )
