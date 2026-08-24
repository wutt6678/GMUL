"""Unit tests for Iteration 6 query generation on the CANONICAL schema.

Queries are ``granunlearn.schema.QueryRecord`` — the single data contract
shared with PredictionRecord scoring.  Generation is partition-aware:
unlearning families (F/G/N/M) for TARGET associations only, retain roles
for retained associations (Blocker-1/2 fixes, Iteration 6 review).
"""

from __future__ import annotations

import pytest

from granunlearn.datasets.smoke import select_target_retain
from granunlearn.evaluation.query_generation import (
    ADVERSARIAL_FAMILIES,
    FAMILIES,
    FAMILY_QUERY_TYPE,
    SPLITS,
    UNLEARNING_FAMILIES,
    answer_level_for_family,
    generate_queries,
    select_other_entity_donor,
    template_index,
    validate_queries,
)
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    ProvenanceInfo,
    QueryRecord,
    SplitInfo,
)


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i,
        canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value,
        normalized_value=value.lower().strip(),
        parent_id=None,
    )


def make_assoc(
    assoc_id: str = "a1",
    entity_id: str | None = None,
    values: list[str] | None = None,
    target_level: int = 1,
    attribute_name: str = "occupation",
    hierarchy_type: str = "semantic",
    with_image: bool = True,
) -> AssociationRecord:
    values = values or ["Data Scientist", "Analyst", "Professional"]
    levels = [_level(i, v) for i, v in enumerate(values)]
    images = []
    if with_image:
        images.append(ImageRef(
            image_id=f"img_{assoc_id}", path=f"data/images/{assoc_id}.jpg",
            source="materialized", split="train"))
    return AssociationRecord(
        association_id=assoc_id,
        dataset="mllmu_hier",
        entity_id=entity_id or f"e_{assoc_id}",
        entity_name="Jane Doe",
        attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=levels,
        original_level=0,
        target_level=target_level,
        source_modalities=["text", "image"] if images else ["text"],
        images=images,
        textual_context=[f"Jane Doe — {attribute_name}: {values[0]}"],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"),
    )


def two_entity_pool() -> tuple[list[AssociationRecord], dict]:
    """2 entities x (1 semantic + 2 numeric) -> 2 targets, 4 retain."""
    assocs = [
        make_assoc("e1__occupation", "e1", attribute_name="occupation",
                   hierarchy_type="semantic"),
        make_assoc("e1__salary", "e1",
                   values=["$120,000", "$100K-$150K"],
                   attribute_name="salary", hierarchy_type="numeric"),
        make_assoc("e1__height", "e1", values=["180 cm", "175-185 cm"],
                   attribute_name="height", hierarchy_type="numeric"),
        make_assoc("e2__occupation", "e2", attribute_name="occupation",
                   hierarchy_type="semantic"),
        make_assoc("e2__salary", "e2",
                   values=["$80,000", "$75K-$100K"],
                   attribute_name="salary", hierarchy_type="numeric"),
        make_assoc("e2__height", "e2", values=["165 cm", "160-170 cm"],
                   attribute_name="height", hierarchy_type="numeric"),
    ]
    partition = select_target_retain(assocs, seed=42)
    return assocs, partition


class TestAnswerLevels:
    def test_fine_and_retain_answer_level_zero(self):
        a = make_assoc()
        for fam in FAMILIES:
            if fam.startswith("fine_") or fam.startswith("retain_"):
                assert answer_level_for_family(a, fam) == 0

    def test_granular_levels(self):
        a = make_assoc(target_level=1)  # 3 levels
        assert answer_level_for_family(a, "granular_fine") == 1
        assert answer_level_for_family(a, "granular_intermediate") == 2
        assert answer_level_for_family(a, "granular_coarse") == 2

    def test_two_level_chain_degenerates_consistently(self):
        a = make_assoc(values=["Munich", "Germany"], target_level=1)
        assert answer_level_for_family(a, "granular_fine") == 1
        assert answer_level_for_family(a, "granular_intermediate") == 1
        assert answer_level_for_family(a, "granular_coarse") == 1

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            answer_level_for_family(make_assoc(), "bogus_family")


class TestTemplateAssignment:
    def test_distinct_templates_across_splits(self):
        for aid in ("a1", "a2", "zzz", "mllmu_00042__salary"):
            idxs = [template_index(42, aid, "fine_direct", s) for s in SPLITS]
            assert len(set(idxs)) == 3

    def test_deterministic(self):
        assert (template_index(7, "a1", "fine_direct", "val")
                == template_index(7, "a1", "fine_direct", "val"))


class TestCanonicalContract:
    def test_emits_canonical_query_record(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        assert all(isinstance(q, QueryRecord) for q in queries)

    def test_acceptable_and_forbidden_ids(self):
        """San Francisco -> California -> USA example: target=California
        => acceptable=[California], forbidden=[San Francisco]."""
        a = make_assoc(
            "loc1", values=["San Francisco", "California", "USA"],
            target_level=1, attribute_name="residence",
            hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["granular_fine"])
        for q in queries:
            assert q.expected_level == 1
            assert q.acceptable_answer_ids == ["l1:california"]
            assert q.forbidden_descendant_ids == ["l0:san_francisco"]
            assert q.expected_answer == "California"

    def test_fine_queries_forbid_nothing_below_zero(self):
        a = make_assoc("t1")
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["fine_direct"])
        assert all(q.forbidden_descendant_ids == [] for q in queries)
        assert all(q.expected_level == 0 for q in queries)

    def test_query_type_route_mapping(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        for q in queries:
            assert q.query_type == FAMILY_QUERY_TYPE[q.family]
            if q.family == "multimodal_image_text":
                assert q.route == "image_text_to_text"
                assert len(q.image_ids) == 1
            else:
                assert q.route == "text_to_text"
                assert q.image_ids == []

    def test_adversarial_flag_exactly_negation_correction(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        for q in queries:
            assert q.adversarial == (q.family in ADVERSARIAL_FAMILIES)
        assert any(q.adversarial for q in queries)


class TestPartitionAwareGeneration:
    def test_unlearning_families_only_for_targets(self):
        assocs, partition = two_entity_pool()
        targets = set(partition["target_association_ids"])
        queries = generate_queries(assocs, partition, seed=42)
        for q in queries:
            if q.family in UNLEARNING_FAMILIES:
                assert q.association_id in targets

    def test_retain_same_entity_covers_retained_only(self):
        assocs, partition = two_entity_pool()
        retain = set(partition["retain_association_ids"])
        queries = generate_queries(assocs, partition, seed=42)
        rse = [q for q in queries if q.family == "retain_same_entity"]
        assert rse and all(q.association_id in retain for q in rse)
        # every retained association x every split
        for rid in retain:
            for split in SPLITS:
                assert any(q.association_id == rid and q.split == split
                           for q in rse)

    def test_retain_other_entity_cross_entity(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        roe = [q for q in queries if q.family == "retain_other_entity"]
        assert roe
        by_id = {a.association_id: a for a in assocs}
        for q in roe:
            tail = q.query_id.split("__for_", 1)[-1]
            target_id = tail.rsplit("__", 1)[0]  # ids may contain '__'
            assert by_id[q.association_id].entity_id != \
                by_id[target_id].entity_id

    def test_donor_prefers_same_attribute(self):
        assocs, _ = two_entity_pool()
        target = next(a for a in assocs
                      if a.association_id == "e1__occupation")
        donor = select_other_entity_donor(target, assocs, seed=42)
        assert donor is not None
        assert donor.attribute_name == "occupation"
        assert donor.entity_id == "e2"

    def test_counts(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        n_targets = len(partition["target_association_ids"])
        n_retain = len(partition["retain_association_ids"])
        unlearning = sum(1 for q in queries
                         if q.family in UNLEARNING_FAMILIES)
        assert unlearning == n_targets * 13 * 3
        assert sum(1 for q in queries
                   if q.family == "retain_same_entity") == n_retain * 3
        assert sum(1 for q in queries
                   if q.family == "retain_other_entity") == n_targets * 3

    def test_deterministic_regeneration(self):
        assocs, partition = two_entity_pool()
        q1 = [q.model_dump() for q in generate_queries(assocs, partition, 42)]
        q2 = [q.model_dump() for q in generate_queries(assocs, partition, 42)]
        assert q1 == q2

    def test_retain_family_rejected_as_unlearning_family(self):
        assocs, partition = two_entity_pool()
        with pytest.raises(ValueError):
            generate_queries(assocs, partition, seed=42,
                             families=["retain_same_entity"])


class TestNegationNoLeak:
    def test_two_level_chain_uses_fine_distractor(self):
        a = make_assoc("s1", values=["$120,000", "$100K-$150K"],
                       target_level=1, attribute_name="salary",
                       hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["negation_correction"])
        assert all(q.expected_answer == "$100K-$150K" for q in queries)
        assert all("$120,000" in q.prompt for q in queries)
        assert all(q.expected_answer not in q.prompt for q in queries)

    def test_substring_of_longer_distractor_is_legitimate(self):
        a = make_assoc("d1", values=["May 12, 1985", "May 1985", "1985"],
                       target_level=2, attribute_name="date_of_birth",
                       hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["negation_correction"])
        assert any("May 12, 1985" in q.prompt for q in queries)


class TestValidation:
    def test_clean_generation_passes(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        errors, stats = validate_queries(queries, assocs, partition=partition)
        assert errors == []
        assert stats["num_queries"] == len(queries)
        assert stats["num_adversarial"] == \
            len(partition["target_association_ids"]) * 3

    def test_bad_acceptable_ids_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        queries[0] = queries[0].model_copy(
            update={"acceptable_answer_ids": ["bogus:id"]})
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("acceptable_answer_ids" in e for e in errors)

    def test_bad_forbidden_ids_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        g = next(q for q in queries if q.family == "granular_coarse")
        g2 = g.model_copy(update={"forbidden_descendant_ids": []})
        queries = [q if q.query_id != g.query_id else g2 for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("forbidden_descendant_ids" in e for e in errors)

    def test_missing_adversarial_flag_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        nc = next(q for q in queries if q.family == "negation_correction")
        nc2 = nc.model_copy(update={"adversarial": False})
        queries = [q if q.query_id != nc.query_id else nc2 for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("adversarial flag missing" in e for e in errors)

    def test_retain_query_on_target_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        target_id = partition["target_association_ids"][0]
        rse = next(q for q in queries if q.family == "retain_same_entity")
        bad = rse.model_copy(update={
            "query_id": "sabotaged__retain_same_entity__train",
            "association_id": target_id})
        queries.append(bad)
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("asks a TARGET association" in e for e in errors)

    def test_missing_retain_coverage_detected(self):
        assocs, partition = two_entity_pool()
        queries = [q for q in generate_queries(assocs, partition, seed=42)
                   if not (q.family == "retain_same_entity"
                           and q.split == "test")]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("retain_same_entity missing" in e for e in errors)

    def test_retain_fact_dedupe(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        errors, _ = validate_queries(
            queries, assocs, partition=partition,
            retain_facts={queries[0].expected_answer})
        assert any("retain fact" in e for e in errors)
