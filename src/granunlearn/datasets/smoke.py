"""10-entity smoke dataset selection + Iteration 5 audit gate (Iteration 6).

Two responsibilities:

1. **Audit gate** — the Iteration 5 manual audit sample must be fully
   adjudicated and free of contradictions before the smoke dataset may be
   built.  An item is *adjudicated* when it has a verdict in
   ``{accepted, rejected}``; items whose pipeline chain is empty (rejected
   upstream, nothing to audit) are exempt.  A contradiction is an
   auditor-rejected chain that appears in the ACCEPTED chains file.

2. **Deterministic smoke selection** — pick ``n`` (default 10) entities
   from the full association pool.  Eligibility guarantees the smoke set
   exercises every query family: at least ``min_attribute_types`` distinct
   attributes, at least one semantic (Qwen-assisted) hierarchy, and at
   least one materialised image.  Eligible entities are ranked by
   ``sha256(f"{seed}:smoke:{entity_id}")`` — fully reproducible.
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

    Returns ``(passed, problems)``.
    """
    problems: list[str] = []
    audit = json.loads(Path(audit_sample_path).read_text())
    accepted: set[tuple[str, str, tuple[str, ...]]] = set()
    for line in Path(accepted_chains_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            accepted.add(
                (row["attribute"], row["value"], tuple(row["chain"])))

    for i, item in enumerate(audit):
        chain = item.get("chain") or []
        verdict = item.get("auditor_verdict")
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
        key = (item.get("attribute"), item.get("value"), tuple(chain))
        if verdict == "rejected" and key in accepted:
            problems.append(
                f"audit item {i}: auditor rejected chain "
                f"{' -> '.join(chain)} but it is in the accepted set")
    return len(problems) == 0, problems


def select_smoke_entities(
    associations: list[AssociationRecord],
    seed: int = 42,
    n: int = 10,
    min_attribute_types: int = 6,
    require_semantic: bool = True,
    require_image: bool = True,
) -> list[str]:
    """Deterministically select ``n`` coverage-qualified entities."""
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
            f"{seed}:smoke:{entity_id}".encode()).hexdigest()

    return sorted(eligible, key=rank)[:n]


def subset_associations(
    associations: list[AssociationRecord], entity_ids: set[str]
) -> list[AssociationRecord]:
    """Keep only associations of the selected entities (stable order)."""
    return [a for a in associations if a.entity_id in entity_ids]


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
