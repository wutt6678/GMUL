"""Taxonomic hierarchy builder (e.g. iNaturalist species → genus → family).

Uses authoritative taxonomy metadata — never LLM-generated labels when
source taxonomy data is available.
"""

from __future__ import annotations

from granunlearn.schema import HierarchyLevel

from .base import ChainHierarchy
from .canonicalize import make_canonical_id, normalize

# Authoritative Linnaean rank ordering (finest → coarsest).
# Only the ranks relevant to the MVP are listed; extend as needed.
RANK_ORDER: dict[str, int] = {
    "subspecies": 0,
    "species": 1,
    "genus": 2,
    "subfamily": 3,
    "family": 4,
    "superfamily": 5,
    "order": 6,
    "class": 7,
    "phylum": 8,
    "kingdom": 9,
}


def build_taxonomic_hierarchy(
    taxon_chain: list[dict],
    prefix: str = "tax",
    strict_ranks: bool = True,
) -> ChainHierarchy:
    """Build a taxonomic hierarchy from an ordered chain.

    Parameters
    ----------
    taxon_chain : list[dict]
        Ordered from finest to coarsest.  Each dict must contain:

        * ``name`` — the taxon name (e.g. "Passer domesticus")
        * ``rank`` — taxonomic rank (e.g. "species", "genus", "family")

        May also contain ``taxon_id`` for authoritative ID linking.
    prefix : str
        Prefix for canonical IDs.
    strict_ranks : bool
        If ``True`` (default), verify that:

        * Every rank is in ``RANK_ORDER``.
        * Ranks are strictly increasing in coarseness (no reorderings
          like species → family → genus).

    Returns
    -------
    ChainHierarchy

    Raises
    ------
    ValueError
        If rank validation fails in strict mode.

    Example::

        build_taxonomic_hierarchy([
            {"name": "Passer domesticus", "rank": "species", "taxon_id": "12345"},
            {"name": "Passer",            "rank": "genus",   "taxon_id": "12344"},
            {"name": "Passeridae",        "rank": "family",  "taxon_id": "12340"},
        ])
    """
    if len(taxon_chain) < 2:
        raise ValueError("Taxonomic hierarchy requires at least 2 levels")

    # ---- Rank-order validation (strict mode) ----------------------------
    if strict_ranks:
        prev_order: int | None = None
        for i, taxon in enumerate(taxon_chain):
            rank = taxon.get("rank")
            if rank is None:
                raise ValueError(
                    f"Taxon at index {i} ({taxon.get('name', '?')}) has no 'rank' field"
                )
            if rank not in RANK_ORDER:
                raise ValueError(
                    f"Unknown taxonomic rank {rank!r} at index {i}. "
                    f"Known ranks: {sorted(RANK_ORDER)}"
                )
            order = RANK_ORDER[rank]
            if prev_order is not None and order <= prev_order:
                raise ValueError(
                    f"Rank order not strictly increasing at index {i}: "
                    f"{taxon_chain[i - 1]['rank']} (order {prev_order}) → "
                    f"{rank} (order {order}). "
                    f"Expected finer → coarser."
                )
            prev_order = order

    # ---- Build levels ---------------------------------------------------
    levels: list[HierarchyLevel] = []
    for i, taxon in enumerate(taxon_chain):
        name = taxon["name"]
        rank = taxon.get("rank", f"rank_{i}")
        cid = make_canonical_id(prefix, name)
        parent_id = (
            make_canonical_id(prefix, taxon_chain[i + 1]["name"])
            if i + 1 < len(taxon_chain)
            else None
        )

        levels.append(HierarchyLevel(
            level=i,
            canonical_id=cid,
            value=name,
            normalized_value=normalize(name),
            parent_id=parent_id,
            metadata={
                "rank": rank,
                "taxon_id": taxon.get("taxon_id"),
            },
        ))

    return ChainHierarchy(levels)
