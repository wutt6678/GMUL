"""Unit tests for Iteration 6 smoke selection, target/retain partition,
and the hardened Iteration 5 audit gate."""

from __future__ import annotations

import json

import pytest

from granunlearn.datasets.smoke import (
    check_audit_gate,
    select_smoke_entities,
    select_target_retain,
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


class TestSelectTargetRetain:
    def test_one_semantic_plus_one_numeric_per_entity(self):
        pool = rich_entity("e1") + rich_entity("e2")
        p = select_target_retain(pool, seed=42)
        assert p["target_counts_by_type"] == {"semantic": 2, "numeric": 2}
        assert len(p["target_association_ids"]) == 4
        assert len(p["retain_association_ids"]) == len(pool) - 4
        for e in ("e1", "e2"):
            assert len(p["per_entity"][e]["targets"]) == 2
            assert len(p["per_entity"][e]["retain"]) == len(ATTRS) - 2

    def test_partition_is_exhaustive_and_disjoint(self):
        pool = rich_entity("e1") + rich_entity("e2")
        p = select_target_retain(pool, seed=42)
        all_ids = {a.association_id for a in pool}
        assert (set(p["target_association_ids"])
                | set(p["retain_association_ids"])) == all_ids
        assert not (set(p["target_association_ids"])
                    & set(p["retain_association_ids"]))

    def test_deterministic(self):
        pool = rich_entity("e1") + rich_entity("e2")
        assert (select_target_retain(pool, seed=42)["target_association_ids"]
                == select_target_retain(pool, seed=42)["target_association_ids"])

    def test_seed_changes_targets(self):
        pool = rich_entity("e1")
        picked = {tuple(select_target_retain(pool, seed=s)[
            "target_association_ids"]) for s in range(30)}
        assert len(picked) > 1  # hash ranking varies with the seed

    def test_entity_missing_type_gets_fewer_targets(self):
        pool = [make_assoc("e1", "salary")]  # numeric only
        p = select_target_retain(pool, seed=42)
        assert p["target_counts_by_type"] == {"semantic": 0, "numeric": 1}


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
        assert not ok and any("accepted chain exists" in p for p in problems)

    def test_auditor_rejection_of_non_accepted_chain_passes(self, tmp_path):
        """Auditor may reject a chain the pipeline never accepted."""
        chain = ["A", "B"]
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": chain,
             "auditor_verdict": "rejected", "auditor_notes": ""},
        ], [])
        ok, problems = check_audit_gate(audit, chains)
        assert ok and problems == []

    def test_accepted_item_with_changed_chain_fails(self, tmp_path):
        """Hardened invariant: a REGENERATED different chain for the same
        (attribute, value) cannot bypass the previous audit."""
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A",
             "chain": ["A", "B"], "auditor_verdict": "accepted",
             "auditor_notes": ""},
        ], [{"attribute": "occupation", "value": "A",
             "chain": ["A", "C"]}])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok and any("changed since audit" in p for p in problems)

    def test_accepted_item_value_no_longer_accepted_fails(self, tmp_path):
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A",
             "chain": ["A", "B"], "auditor_verdict": "accepted",
             "auditor_notes": ""},
        ], [])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok and any("no accepted chain exists" in p
                              for p in problems)

    def test_rejected_item_with_new_accepted_chain_for_value_fails(self,
                                                                    tmp_path):
        """Value-level check: even a DIFFERENT accepted chain for a
        previously rejected value requires re-audit."""
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A",
             "chain": ["A", "B"], "auditor_verdict": "rejected",
             "auditor_notes": ""},
        ], [{"attribute": "occupation", "value": "A",
             "chain": ["A", "X", "Y"]}])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok and any("re-audit required" in p for p in problems)

    def test_empty_chain_marked_accepted_fails(self, tmp_path):
        audit, chains = self._write(tmp_path, [
            {"attribute": "occupation", "value": "A", "chain": [],
             "auditor_verdict": "accepted", "auditor_notes": ""},
        ], [])
        ok, problems = check_audit_gate(audit, chains)
        assert not ok
