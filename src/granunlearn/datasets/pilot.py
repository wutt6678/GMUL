"""Iteration-11 pilot-100 dataset selection (mixed MLLMU + iNaturalist).

The pilot freezes a 100-entity mixed pool:

* ~64 MLLMU person entities (semantic + numeric attributes), selected
  coverage-qualified and deterministically (sha256 ranking, seed 42);
* 36 real iNaturalist species entities (authoritative taxonomic
  chains species -> genus -> family).

TARGET BALANCE: the frozen target set is balanced ACROSS hierarchy
types and, within a type, ACROSS attributes — per-attribute quotas
(default 30 per type) are filled by deterministic sha256 ranking, so
semantic / numeric / taxonomic targets are equal in count and no
attribute dominates.  Species beyond the taxonomic quota stay
RETAINED, which guarantees same-stratum other-entity-retain donors
for taxonomic targets.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.schema import AssociationRecord

log = setup_logger("pilot_selection")

#: Frozen pilot-100 balance contract (30 targets per hierarchy type;
#: per-attribute quotas within each type).
PILOT_TARGET_QUOTAS: dict[str, dict[str, int]] = {
    "semantic": {"residence": 8, "birthplace": 8,
                 "occupation": 7, "education": 7},
    "numeric": {"date_of_birth": 10, "salary": 10, "height": 10},
    "taxonomic": {"taxonomic_classification": 30},
}


def select_balanced_targets(
    associations: list[AssociationRecord],
    seed: int = 42,
    quotas: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Deterministic balanced F/R partition across hierarchy types.

    Per (hierarchy_type, attribute) quota, associations are ranked by
    ``sha256(f"{seed}:baltarget:{attribute}:{entity}:{association}")``
    and the top-quota ones become TARGETS; every other association is
    RETAIN.  Quotas that cannot be filled (too few associations) take
    what exists and record the shortfall — never a fabricated target.

    Returns the SAME partition schema as ``smoke.select_target_retain``
    (so downstream state/group builders need no changes) plus balance
    evidence.
    """
    quotas = quotas or PILOT_TARGET_QUOTAS

    def rank(a: AssociationRecord) -> str:
        return hashlib.sha256(
            f"{seed}:baltarget:{a.attribute_name}:{a.entity_id}:"
            f"{a.association_id}".encode()).hexdigest()

    by_attr: dict[tuple[str, str], list[AssociationRecord]] = \
        defaultdict(list)
    for a in associations:
        by_attr[(a.hierarchy_type, a.attribute_name)].append(a)

    target_ids: list[str] = []
    per_quota: dict[str, dict[str, Any]] = {}
    counts_by_type: dict[str, int] = {t: 0 for t in quotas}
    for htype, attr_quotas in quotas.items():
        for attr, quota in attr_quotas.items():
            pool = sorted(by_attr.get((htype, attr), []), key=rank)
            picked = pool[:quota]
            target_ids.extend(a.association_id for a in picked)
            counts_by_type[htype] += len(picked)
            per_quota[f"{htype}:{attr}"] = {
                "quota": quota,
                "available": len(pool),
                "selected": len(picked),
                "shortfall": max(0, quota - len(picked)),
            }

    target_set = set(target_ids)
    per_entity: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"targets": [], "retain": []})
    for a in associations:
        bucket = "targets" if a.association_id in target_set \
            else "retain"
        per_entity[a.entity_id][bucket].append(a.association_id)

    return {
        "seed": seed,
        "ranking": "sha256(seed:baltarget:attribute:entity:association)",
        "balance_quotas": quotas,
        "target_association_ids": sorted(target_ids),
        "retain_association_ids": sorted(
            a.association_id for a in associations
            if a.association_id not in target_set),
        "target_counts_by_type": counts_by_type,
        "quota_fill": per_quota,
        "per_entity": {e: {"targets": sorted(v["targets"]),
                           "retain": sorted(v["retain"])}
                       for e, v in sorted(per_entity.items())},
        "balance_note": (
            "Targets are balanced across hierarchy types (equal counts) "
            "and, within each type, across attributes via per-attribute "
            "quotas; unfilled quotas are recorded as shortfalls, never "
            "fabricated.  Species above the taxonomic quota remain "
            "RETAINED so taxonomic targets have same-stratum "
            "other-entity-retain donors."),
    }


def balance_evidence(partition: dict[str, Any],
                     associations: list[AssociationRecord]
                     ) -> dict[str, Any]:
    """Committed evidence that the frozen partition is balanced."""
    by_type: dict[str, int] = defaultdict(int)
    by_attr: dict[str, int] = defaultdict(int)
    targets = set(partition["target_association_ids"])
    for a in associations:
        if a.association_id in targets:
            by_type[a.hierarchy_type] += 1
            by_attr[f"{a.hierarchy_type}:{a.attribute_name}"] += 1
    counts = sorted(by_type.values())
    return {
        "target_counts_by_type": dict(sorted(by_type.items())),
        "target_counts_by_attribute": dict(sorted(by_attr.items())),
        "types_balanced": len(set(counts)) == 1 and bool(counts),
        "num_targets": len(targets),
        "num_retain": len(partition["retain_association_ids"]),
        "quota_fill": partition.get("quota_fill"),
    }
