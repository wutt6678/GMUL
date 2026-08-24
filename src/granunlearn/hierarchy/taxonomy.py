"""Taxonomic hierarchy builder (e.g. iNaturalist species → genus → family).

Uses authoritative taxonomy metadata — never LLM-generated labels when
source taxonomy data is available.
"""

from __future__ import annotations

from granunlearn.schema import HierarchyLevel

from .base import ChainHierarchy
from .canonicalize import make_canonical_id, normalize


def build_taxonomic_hierarchy(
    taxon_chain: list[dict],
    prefix: str = "tax",
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

    Returns
    -------
    ChainHierarchy

    Example::

        build_taxonomic_hierarchy([
            {"name": "Passer domesticus", "rank": "species", "taxon_id": "12345"},
            {"name": "Passer",            "rank": "genus",   "taxon_id": "12344"},
            {"name": "Passeridae",        "rank": "family",  "taxon_id": "12340"},
        ])
    """
    if len(taxon_chain) < 2:
        raise ValueError("Taxonomic hierarchy requires at least 2 levels")

    levels: list[HierarchyLevel] = []
    for i, taxon in enumerate(taxon_chain):
        name = taxon["name"]
        rank = taxon.get("rank", f"rank_{i}")
        cid = make_canonical_id(prefix, name)
        parent_id = make_canonical_id(prefix, taxon_chain[i + 1]["name"]) if i + 1 < len(taxon_chain) else None

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
