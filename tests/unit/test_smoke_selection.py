"""Unit tests for Iteration 6 smoke selection + audit gate."""

from __future__ import annotations

import json

import pytest

from granunlearn.datasets.smoke import (
    check_audit_gate,
    select_smoke_entities,
    selection_evidence,
    subset_associations,
)
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    ProvenanceInfo,
    SplitInfo,
)


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value}", value=value,
        normalized_value=value.lower())


def make_assoc(entity_id: str, attribute_name: str,
               hierarchy_type: str = "numeric",
               with_image: bool = True) -> AssociationRecord:
    images = []
    if with_image:
        images.append(ImageRef(
            image_id=f"img_{entity_id}",
            path=f"data/images/{entity_id}.jpg",
            source="materialized", split="train"))
    return AssociationRecord(
        association_id=f"{entity_id}__{attribute_name}",
        dataset="mllmu_hier",
        entity_id=entity_id,
        entity_name=f"Person {entity_id}",
        attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=[_level(0, f"fine {attribute_name}"),
                _level(1, f"coarse {attribute_name}")],
        original_level=0,
        target_level=1,
        images=images,
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"),
    )


ATTRS = ["date_of_birth", "salary", "height", "residence",
         "birthplace", "occupation", "education"]


def rich_entity(entity_id: str) -> list[AssociationRecord]:
    """An entity with all 7 attributes, incl. semantic + image."""
    return [
        make_assoc(entity_id, a,
                   hierarchy_type="semantic" if a in
                   ("occupation", "education") else "numeric")
        for a in ATTRS
    ]


class TestSelectSmokeEntities:
    def test_selects_n_entities(self):
        pool = [a for i in range(30) for a in rich_entity(f"e{i:02d}")]
        selected = select_smoke_entities(pool, seed=42, n=10)
        assert len(selected) == 10
        assert len(set(selected)) == 10

    def test_deterministic(self):
        pool = [a for i in range(20) for a in rich_entity(f"e{i:02d}")]
        assert (select_smoke_entities(pool, seed=42)
                == select_smoke_entities(pool, seed=42))

    def test_seed_changes_selection(self):
        pool = [a for i in range(20) for a in rich_entity(f"e{i:02d}")]
        assert (select_smoke_entities(pool, seed=1)
                != select_smoke_entities(pool, seed=2))

    def test_eligibility_min_attributes(self):
        pool = rich_entity("full") + [make_assoc("thin", "salary")]
        selected = select_smoke_entities(pool, seed=42, n=10)
        assert selected == ["full"]

    def test_eligibility_requires_semantic(self):
        pool = [make_assoc("det_only", a) for a in ATTRS] + rich_entity("sem")
        selected = select_smoke_entities(pool, seed=42, n=10)
        assert selected == ["sem"]

    def test_eligibility_requires_image(self):
        no_img = [make_assoc("noimg", a, with_image=False) for a in ATTRS]
        pool = no_img + rich_entity("img")
        selected = select_smoke_entities(pool, seed=42, n=10)
        assert selected == ["img"]

    def test_subset_keeps_only_selected(self):
        pool = [a for i in range(5) for a in rich_entity(f"e{i}")]
        selected = select_smoke_entities(pool, seed=42, n=2)
        sub = subset_associations(pool, set(selected))
        assert {a.entity_id for a in sub} == set(selected)
        assert len(sub) == 2 * len(ATTRS)

    def test_selection_evidence_shape(self):
        pool = [a for i in range(3) for a in rich_entity(f"e{i}")]
        selected = select_smoke_entities(pool, seed=42, n=2)
        ev = selection_evidence(pool, selected, 42, 2, 6)
        assert ev["num_eligible_entities"] == 3
        assert ev["selected_entities"] == selected
        assert all(len(v["attributes"]) == 7
                   for v in ev["selected_per_entity"].values())


class TestAuditGate:
    def _write(self, tmp_path, audit_items, chains):
        audit = tmp_path / "audit.json"
        audit.write_text(json.dumps(audit_items))
        ch = tmp_path / "chains.jsonl"
        ch.write_text("\n".join(json.dumps(c) for c in chains) + "\n")
        return audit, ch

    def test_passes_fully_adjudicated(self, tmp_path):
        chain = ["Data Scientist", "Analyst", "Professional"]
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "Data Scientist",
             "chain": chain, "auditor_verdict": "accepted",
             "auditor_notes": ""},
            {"attribute": "occupation", "value": "X", "chain": [],
             "auditor_verdict": None, "auditor_notes": ""},
        ], [{"attribute": "occupation", "value": "Data Scientist",
             "chain": chain}])
        ok, problems = check_audit_gate(audit, chains)
        assert ok and problems == []

    def test_missing_verdict_fails(self, tmp_path):
        chain = ["A", "B"]
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": chain,
             "auditor_verdict": None, "auditor_notes": ""},
        ], [{"attribute": "occupation", "value": "A", "chain": chain}])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok and any("missing auditor_verdict" in p for p in problems)

    def test_auditor_rejection_of_accepted_chain_fails(self, tmp_path):
        chain = ["A", "B"]
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": chain,
             "auditor_verdict": "rejected", "auditor_notes": ""},
        ], [{"attribute": "occupation", "value": "A", "chain": chain}])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok and any("auditor rejected chain" in p for p in problems)

    def test_auditor_rejection_of_non_accepted_chain_passes(self, tmp_path):
        """Auditor may reject a chain the pipeline never accepted."""
        chain = ["A", "B"]
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": chain,
             "auditor_verdict": "rejected", "auditor_notes": ""},
        ], [])
        ok, problems = check_audit_gate(audit, chains)
        assert ok and problems == []

    def test_empty_chain_marked_accepted_fails(self, tmp_path):
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": [],
             "auditor_verdict": "accepted", "auditor_notes": ""},
        ], [])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok
