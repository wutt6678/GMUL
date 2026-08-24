"""Location hierarchy builder — observed components only (Iteration 4).

Variable-depth chains built strictly from the components present in the
source value.  NO geocoding and NO invented intermediate administrative
levels: a two-component value yields a two-level chain.

Example::

    build_location_hierarchy(["Riga", "Latvia"])
    # level 0: "Riga, Latvia"  (exact)
    # level 1: "Latvia"        (coarsest observed component)

    build_location_hierarchy(["San Francisco", "California", "USA"])
    # level 0: "San Francisco, California, USA"
    # level 1: "California, USA"
    # level 2: "USA"
"""

from __future__ import annotations

from granunlearn.schema import HierarchyLevel

from .base import ChainHierarchy
from .canonicalize import make_canonical_id, normalize


def build_location_hierarchy(
    components: list[str],
    prefix: str = "loc",
) -> ChainHierarchy:
    """Build a location hierarchy from OBSERVED components, finest first.

    Parameters
    ----------
    components : list[str]
        Ordered finest→coarsest, e.g. ``["Riga", "Latvia"]``.  At least
        two components are required (a single component has no coarser
        observed abstraction and cannot satisfy ``AssociationRecord``'s
        minimum of two levels).

    Raises
    ------
    ValueError
        If fewer than two non-empty components are given.
    """
    comps = [c.strip() for c in components if c and c.strip()]
    if len(comps) < 2:
        raise ValueError(
            "Location hierarchy needs >= 2 observed components "
            f"(got {components!r}); refusing to invent intermediate levels"
        )

    # Level 0 = exact full value; each coarser level drops the finest
    # remaining component.  Depth is therefore exactly len(components).
    raw_values = [", ".join(comps[i:]) for i in range(len(comps))]

    levels: list[HierarchyLevel] = []
    for i, val in enumerate(raw_values):
        cid = make_canonical_id(prefix, val)
        parent_id = (
            make_canonical_id(prefix, raw_values[i + 1])
            if i + 1 < len(raw_values) else None
        )
        levels.append(HierarchyLevel(
            level=i,
            canonical_id=cid,
            value=val,
            normalized_value=normalize(val),
            parent_id=parent_id,
            metadata={
                "type": "location",
                "observed_components": comps,
                "component_depth": len(comps),
            },
        ))

    return ChainHierarchy(levels)
