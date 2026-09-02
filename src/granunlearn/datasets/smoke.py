"""10-entity smoke dataset selection + Iteration 5 audit gate (Iteration 6).

Three responsibilities:

1. **Audit gate** — the Iteration 5 manual audit sample must be fully
   adjudicated and consistent with the CURRENT accepted chains, checked at
   (attribute, value) granularity (hardened per Iteration 6 review):

   * accepted item  -> the currently accepted chain for that value must be
     EXACTLY the audited chain (a regenerated different chain cannot
     bypass a previous audit);
   * rejected item  -> NO accepted chain may exist for that value unless
     it has been explicitly re-audited;
   * items whose pipeline chain is empty (rejected upstream, nothing to
     audit) are exempt but must never be marked accepted.

2. **Deterministic smoke selection** — pick ``n`` (default 10) entities
   from the full association pool.  Eligibility guarantees the smoke set
   exercises every query family: at least ``min_attribute_types`` distinct
   attributes, at least one semantic (Qwen-assisted) hierarchy, and at
   least one materialised image.  Eligible entities are ranked by
   ``sha256(f"{seed}:smoke:{entity_id}")`` — fully reproducible.

3. **Target/retain partition** (Blocker-2 fix) — entity–attribute
   selective unlearning requires an explicit distinction between
   F = target associations to unlearn and R = associations that must
   remain unchanged.  Per entity, exactly one semantic and one numeric
   association are designated targets (deterministic hash ranking);
   everything else of those entities is retain.  Without this partition
   MG would degenerate into global profile coarsening.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from granunlearn.schema import AssociationRecord


def check_audit_gate(
    audit_sample_path: str | Path,
    accepted_chains_path: str | Path,
) -> tuple[bool, list[str]]:
    """Verify the Iteration 5 manual audit is complete and consistent.

    Hardened value-level invariant (Iteration 6 review): the check keys on
    ``(attribute, value)``, not on the audited chain tuple, so a
    regenerated *different* chain for the same value cannot bypass a
    previous audit.

    Returns ``(passed, problems)``.
    """
    problems: list[str] = []
    audit = json.loads(Path(audit_sample_path).read_text())
    accepted: dict[tuple[str, str], tuple[str, ...]] = {}
    for line in Path(accepted_chains_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            accepted[(row["attribute"], row["value"])] = tuple(row["chain"])

    for i, item in enumerate(audit):
        chain = tuple(item.get("chain") or [])
        verdict = item.get("auditor_verdict")
        key = (item.get("attribute"), item.get("value"))
        if not chain:
            # Pipeline-rejected item with no chain: nothing to audit.
            if verdict == "accepted":
                problems.append(
                    f"audit item {i}: empty chain (pipeline-rejected) was "
                    f"marked accepted")
            continue
        if verdict not in ("accepted", "rejected"):
            problems.append(
                f"audit item {i} ({item.get('attribute')!r}, "
                f"{item.get('value')!r}): missing auditor_verdict")
            continue
        if verdict == "accepted":
            current = accepted.get(key)
            if current is None:
                problems.append(
                    f"audit item {i}: audited as accepted but no accepted "
                    f"chain exists for {key!r} (regenerated?) — re-audit "
                    f"required")
            elif current != chain:
                problems.append(
                    f"audit item {i}: accepted chain for {key!r} changed "
                    f"since audit ('{' -> '.join(chain)}' vs "
                    f"'{' -> '.join(current)}') — re-audit required")
        else:  # rejected
            if key in accepted:
                problems.append(
                    f"audit item {i}: auditor rejected {key!r} but an "
                    f"accepted chain exists for that value "
                    f"('{' -> '.join(accepted[key])}') — re-audit required")
    return len(problems) == 0, problems


def select_smoke_entities(
    associations: list[AssociationRecord],
    seed: int = 42,
    n: int = 10,
    min_attribute_types: int = 6,
    require_semantic: bool = True,
    require_image: bool = True,
    salt: str = "smoke",
) -> list[str]:
    """Deterministically select ``n`` coverage-qualified entities.

    ``salt`` namespaces the ranking so different iterations select
    independently (Iteration 11 pilot uses salt="pilot100"); the
    default reproduces the committed smoke selection exactly.
    """
    by_entity: dict[str, list[AssociationRecord]] = defaultdict(list)
    for a in associations:
        by_entity[a.entity_id].append(a)

    eligible: list[str] = []
    for entity_id, assocs in by_entity.items():
        attrs = {a.attribute_name for a in assocs}
        if len(attrs) < min_attribute_types:
            continue
        if require_semantic and not any(
                a.hierarchy_type == "semantic" for a in assocs):
            continue
        if require_image and not any(a.images for a in assocs):
            continue
        eligible.append(entity_id)

    def rank(entity_id: str) -> str:
        return hashlib.sha256(
            f"{seed}:{salt}:{entity_id}".encode()).hexdigest()

    return sorted(eligible, key=rank)[:n]


def subset_associations(
    associations: list[AssociationRecord], entity_ids: set[str]
) -> list[AssociationRecord]:
    """Keep only associations of the selected entities (stable order)."""
    return [a for a in associations if a.entity_id in entity_ids]


def select_target_retain(
    associations: list[AssociationRecord],
    seed: int = 42,
    targets_per_type: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Deterministic F/R partition for entity–attribute selective unlearning.

    Per entity, ``targets_per_type`` (default: 1 semantic + 1 numeric)
    associations are designated TARGETS by ranking with
    ``sha256(f"{seed}:target:{entity_id}:{association_id}")``; every other
    association of those entities is RETAIN.  Entities lacking a type
    simply contribute fewer targets (never a fabricated one).

    Returns a partition dict ready for manifest persistence::

        {
          "target_association_ids": [...],
          "retain_association_ids": [...],
          "target_counts_by_type": {"semantic": 10, "numeric": 10},
          "per_entity": {entity_id: {"targets": [...], "retain": [...]}},
        }
    """
    targets_per_type = targets_per_type or {"semantic": 1, "numeric": 1}

    def rank(a: AssociationRecord) -> str:
        return hashlib.sha256(
            f"{seed}:target:{a.entity_id}:{a.association_id}".encode()
        ).hexdigest()

    by_entity: dict[str, list[AssociationRecord]] = defaultdict(list)
    for a in associations:
        by_entity[a.entity_id].append(a)

    target_ids: list[str] = []
    per_entity: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, int] = {t: 0 for t in targets_per_type}
    for entity_id in sorted(by_entity):
        assocs = by_entity[entity_id]
        chosen: list[AssociationRecord] = []
        for htype, k in targets_per_type.items():
            candidates = sorted(
                (a for a in assocs if a.hierarchy_type == htype
                 and a not in chosen),
                key=rank)
            picked = candidates[:k]
            chosen.extend(picked)
            counts[htype] += len(picked)
        chosen_ids = {a.association_id for a in chosen}
        target_ids.extend(sorted(chosen_ids))
        per_entity[entity_id] = {
            "targets": sorted(chosen_ids),
            "retain": sorted(
                a.association_id for a in assocs
                if a.association_id not in chosen_ids),
        }

    target_set = set(target_ids)
    retain_ids = sorted(
        a.association_id for a in associations
        if a.association_id not in target_set)
    return {
        "seed": seed,
        "ranking": "sha256(seed:target:entity_id:association_id)",
        "targets_per_type": targets_per_type,
        "target_association_ids": sorted(target_ids),
        "retain_association_ids": retain_ids,
        "target_counts_by_type": counts,
        "per_entity": per_entity,
    }


def selection_evidence(
    associations: list[AssociationRecord],
    selected: list[str],
    seed: int,
    n: int,
    min_attribute_types: int,
) -> dict[str, Any]:
    """Committed evidence describing the deterministic selection."""
    by_entity: dict[str, list[AssociationRecord]] = defaultdict(list)
    for a in associations:
        by_entity[a.entity_id].append(a)
    eligible = sum(
        1 for e in by_entity
        if len({a.attribute_name for a in by_entity[e]}) >= min_attribute_types
        and any(a.hierarchy_type == "semantic" for a in by_entity[e])
        and any(a.images for a in by_entity[e])
    )
    return {
        "seed": seed,
        "requested_entities": n,
        "ranking": f"sha256(seed:smoke:entity_id), ascending",
        "eligibility": {
            "min_attribute_types": min_attribute_types,
            "require_semantic_hierarchy": True,
            "require_materialized_image": True,
        },
        "num_eligible_entities": eligible,
        "selected_entities": selected,
        "selected_per_entity": {
            e: {
                "num_associations": len(by_entity[e]),
                "attributes": sorted({a.attribute_name for a in by_entity[e]}),
            }
            for e in selected
        },
    }
