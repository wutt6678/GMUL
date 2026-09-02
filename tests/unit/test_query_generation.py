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
    BANNED_PROMPT_PHRASES,
    FAMILIES,
    FAMILY_QUERY_TYPE,
    IMAGE_ONLY_FAMILIES,
    SPLITS,
    UNLEARNING_FAMILIES,
    answer_level_for_family,
    family_applicable,
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

    def test_two_level_chain_intermediate_not_applicable(self):
        """A 2-level chain with target=1 has no ancestor above the
        target; granular_intermediate must not be generated (Blocker-1
        fix, review #3).  coarse == target stays permitted."""
        a = make_assoc(values=["Munich", "Germany"], target_level=1)
        assert family_applicable(a, "granular_fine")
        assert not family_applicable(a, "granular_intermediate")
        assert family_applicable(a, "granular_coarse")

    def test_intermediate_applicable_when_ancestor_exists(self):
        a = make_assoc(target_level=1)  # 3 levels
        assert family_applicable(a, "granular_intermediate")
        b = make_assoc(values=["a", "b", "c", "d"], target_level=2)
        assert family_applicable(b, "granular_intermediate")
        c = make_assoc(values=["a", "b", "c"], target_level=2)
        assert not family_applicable(c, "granular_intermediate")

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
        => baseline acceptable=[San Francisco] for fine probes, but the
        FILR view says post-unlearning acceptable=[California],
        leakage forbidden=[San Francisco]."""
        a = make_assoc(
            "loc1", values=["San Francisco", "California", "USA"],
            target_level=1, attribute_name="residence",
            hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["granular_fine", "fine_direct"])
        for q in queries:
            # FILR view is identical for every probe of this target
            assert q.unlearning_target_level == 1
            assert q.leakage_forbidden_ids == ["l0:san_francisco"]
            assert q.post_unlearning_acceptable_answer_ids == [
                "l1:california"]
        gf = [q for q in queries if q.family == "granular_fine"]
        assert all(q.expected_level == 1 for q in gf)
        assert all(q.acceptable_answer_ids == ["l1:california"] for q in gf)
        fd = [q for q in queries if q.family == "fine_direct"]
        assert all(q.expected_level == 0 for q in fd)
        assert all(q.acceptable_answer_ids == ["l0:san_francisco"]
                   for q in fd)

    def test_fine_queries_baseline_forbidden_empty_but_leakage_set(self):
        """fine_* probes ask for level 0 (baseline forbidden empty) but
        MUST still carry the post-unlearning leakage set — everything
        finer than the unlearning target (Blocker-1 fix)."""
        a = make_assoc("t1")  # 3 levels, target_level=1
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["fine_direct"])
        for q in queries:
            assert q.expected_level == 0
            assert q.forbidden_descendant_ids == []
            assert q.unlearning_target_level == 1
            assert q.leakage_forbidden_ids == [
                a.levels[0].canonical_id]
            assert q.post_unlearning_acceptable_answer_ids == [
                a.levels[1].canonical_id]

    def test_retain_probes_have_no_unlearning_semantics(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        retains = [q for q in queries if q.family and
                   q.family.startswith("retain_")]
        assert retains
        for q in retains:
            assert q.unlearning_target_level is None
            assert q.leakage_forbidden_ids == []
            assert q.post_unlearning_acceptable_answer_ids == \
                q.acceptable_answer_ids
        rse = [q for q in retains if q.family == "retain_same_entity"]
        assert all(q.target_association_id is None for q in rse)
        roe = [q for q in retains if q.family == "retain_other_entity"]
        assert all(q.target_association_id for q in roe)

    def test_query_type_route_mapping(self):
        assocs, partition = two_entity_pool()
        by_id = {a.association_id: a for a in assocs}
        queries = generate_queries(assocs, partition, seed=42)
        for q in queries:
            assert q.query_type == FAMILY_QUERY_TYPE[q.family]
            if q.family == "multimodal_image_text":
                assert q.route == "image_text_to_text"
                assert len(q.image_ids) == 1
            elif q.family in IMAGE_ONLY_FAMILIES:
                # Iteration 11: image_to_text route carries the image
                # and NEVER names the entity (asked association's
                # entity — the donor for retain_other_entity_image).
                assert q.route == "image_to_text"
                assert len(q.image_ids) == 1
                ent = by_id[q.association_id]
                for nm in (ent.entity_name, ent.entity_id):
                    assert nm.lower() not in q.prompt.lower()
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

    def test_retain_other_entity_cross_entity_and_retained(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        roe = [q for q in queries if q.family == "retain_other_entity"]
        assert roe
        retain_ids = set(partition["retain_association_ids"])
        by_id = {a.association_id: a for a in assocs}
        for q in roe:
            assert q.association_id in retain_ids  # HARD invariant
            assert q.target_association_id
            assert by_id[q.association_id].entity_id != \
                by_id[q.target_association_id].entity_id

    def test_donor_is_partition_aware(self):
        """Donor selection must skip associations that are themselves
        unlearning targets (Blocker-2 fix)."""
        assocs, partition = two_entity_pool()
        retain_ids = set(partition["retain_association_ids"])
        target = next(a for a in assocs
                      if a.association_id == "e1__occupation")
        # e2__occupation is e2's semantic TARGET -> never a donor
        assert "e2__occupation" not in retain_ids
        donor = select_other_entity_donor(target, assocs, retain_ids,
                                          seed=42)
        assert donor is not None
        assert donor.association_id in retain_ids
        assert donor.entity_id == "e2"

    def test_donor_none_without_retain_candidates(self):
        a = make_assoc("solo")
        # Single association: nothing retained elsewhere -> no donor.
        partition = select_target_retain([a], seed=42)
        donor = select_other_entity_donor(
            a, [a], partition["retain_association_ids"], seed=42)
        assert donor is None

    def test_counts_respect_applicability(self):
        """2-level targets contribute no granular_intermediate queries
        (Blocker-1 fix): per-family counts legitimately differ."""
        assocs, partition = two_entity_pool()
        by_id = {a.association_id: a for a in assocs}
        targets = [by_id[i] for i in partition["target_association_ids"]]
        queries = generate_queries(assocs, partition, seed=42)
        expected_unlearning = sum(
            3 for t in targets for fam in UNLEARNING_FAMILIES
            if family_applicable(t, fam))
        unlearning = sum(1 for q in queries
                         if q.family in UNLEARNING_FAMILIES)
        assert unlearning == expected_unlearning
        # two_entity_pool: occupation targets are 3-level (every
        # unlearning family applies), salary targets are 2-level (no
        # granular_intermediate) — so the total is strictly below the
        # full-family upper bound.
        assert unlearning < len(targets) * len(UNLEARNING_FAMILIES) * 3
        n_retain = len(partition["retain_association_ids"])
        n_targets = len(targets)
        assert sum(1 for q in queries
                   if q.family == "retain_same_entity") == n_retain * 3
        assert sum(1 for q in queries
                   if q.family == "retain_other_entity") == n_targets * 3

    def test_no_intermediate_queries_for_two_level_targets(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        two_level = {a.association_id for a in assocs
                     if a.num_levels() == 2}
        assert not any(q.family == "granular_intermediate"
                       and q.association_id in two_level for q in queries)

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
        # explicit donor pairs are persisted in the stats
        assert len(stats["donor_pairs"]) == \
            len(partition["target_association_ids"])
        assert all(set(p) == {"target_association_id",
                              "donor_association_id"}
                   for p in stats["donor_pairs"])

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

    def test_retain_fact_dedupe_entity_scoped(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        by_id = {a.association_id: a for a in assocs}
        first = next(q for q in queries
                     if not q.family.startswith("retain_"))
        entity = by_id[first.association_id].entity_id
        errors, _ = validate_queries(
            queries, assocs, partition=partition,
            retain_facts_by_entity={entity: {first.expected_answer}})
        assert any("retain fact" in e for e in errors)

    def test_retain_fact_other_entity_not_flagged(self):
        """Facts are entity-conditioned: a corpus attached to a DIFFERENT
        entity must not flag this entity's probes."""
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        by_id = {a.association_id: a for a in assocs}
        first = next(q for q in queries
                     if not q.family.startswith("retain_"))
        other = next(e for e in {by_id[i].entity_id
                                 for i in partition["target_association_ids"]}
                     if e != by_id[first.association_id].entity_id)
        unique_fact = f"UNIQUE_FACT_{first.query_id}"
        errors, _ = validate_queries(
            queries, assocs, partition=partition,
            retain_facts_by_entity={other: {unique_fact}})
        assert not any("retain fact" in e for e in errors)

    def test_non_retain_donor_detected(self):
        """Sabotage: point a donor probe at a TARGET association."""
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        target_id = partition["target_association_ids"][0]
        roe = next(q for q in queries if q.family == "retain_other_entity")
        bad = roe.model_copy(update={"association_id": target_id})
        queries = [q if q.query_id != roe.query_id else bad
                   for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("NOT a retained association" in e for e in errors)

    def test_missing_target_association_id_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        roe = next(q for q in queries if q.family == "retain_other_entity")
        bad = roe.model_copy(update={"target_association_id": None})
        queries = [q if q.query_id != roe.query_id else bad
                   for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("explicit target_association_id" in e for e in errors)

    def test_bad_leakage_set_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        fine = next(q for q in queries if q.family == "fine_direct")
        bad = fine.model_copy(update={"leakage_forbidden_ids": []})
        queries = [q if q.query_id != fine.query_id else bad
                   for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("leakage_forbidden_ids" in e for e in errors)

    def test_retain_probe_with_unlearning_semantics_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        rse = next(q for q in queries if q.family == "retain_same_entity")
        bad = rse.model_copy(update={"unlearning_target_level": 1})
        queries = [q if q.query_id != rse.query_id else bad
                   for q in queries]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("retain probe must not set" in e for e in errors)

    def test_hidden_granularity_metadata_detected(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        bad = queries[0].model_copy(update={
            "prompt": queries[0].prompt + " at the target granularity"})
        queries = [bad] + queries[1:]
        errors, _ = validate_queries(queries, assocs, partition=partition)
        assert any("hidden benchmark metadata" in e for e in errors)


class TestSelfContainedPrompts:
    """Research-design fix (review #3): prompts never reference hidden
    benchmark metadata; granularity is phrased attribute-aware."""

    def test_no_banned_phrases_in_generated_prompts(self):
        assocs, partition = two_entity_pool()
        queries = generate_queries(assocs, partition, seed=42)
        for q in queries:
            lowered = q.prompt.lower()
            for phrase in BANNED_PROMPT_PHRASES:
                assert phrase not in lowered, q.query_id

    def test_date_probes_ask_year_or_decade(self):
        a = make_assoc("d1", values=["1994-08-16", "1994", "1990s"],
                       target_level=1, attribute_name="date_of_birth",
                       hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        q1 = generate_queries([a], partition, seed=42,
                              families=["granular_fine"])
        assert all("What year" in q.prompt for q in q1)
        a2 = a.model_copy(update={"target_level": 2})
        p2 = select_target_retain([a2], seed=42)
        q2 = generate_queries([a2], p2, seed=42,
                              families=["granular_fine"])
        assert all("Which decade" in q.prompt for q in q2)

    def test_salary_height_probes_ask_range(self):
        s = make_assoc("s1", values=["$120,000", "$100K-$150K"],
                       target_level=1, attribute_name="salary",
                       hierarchy_type="numeric")
        h = make_assoc("h1", values=["177.8 cm", "170-180 cm"],
                       target_level=1, attribute_name="height",
                       hierarchy_type="numeric")
        partition = select_target_retain([s, h], seed=42)
        qs = generate_queries([s, h], partition, seed=42,
                              families=["granular_fine"])
        for q in qs:
            assert "range" in q.prompt.lower()

    def test_location_coarse_asks_country(self):
        a = make_assoc("loc1", values=["Sydney, Australia", "Australia"],
                       target_level=1, attribute_name="residence",
                       hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["granular_coarse"])
        assert all("In which country" in q.prompt for q in qs)

    def test_semantic_uses_relative_wording(self):
        a = make_assoc("o1", values=["Data Scientist", "Analyst",
                                      "Professional"], target_level=1)
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["granular_fine"])
        assert all("level(s) of abstraction broader" in q.prompt or
                   "broadest occupational category" in q.prompt
                   for q in qs)


class TestPostUnlearningAcceptable:
    """Blocker-2 fix (review #3): post-unlearning acceptable follows the
    REQUESTED level except for fine_* probes (which map to the target)."""

    def test_coarse_probe_keeps_coarsest_acceptable(self):
        a = make_assoc(
            "loc1", values=["San Francisco", "California", "USA"],
            target_level=1, attribute_name="residence",
            hierarchy_type="numeric")
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["granular_coarse"])
        for q in qs:
            assert q.expected_answer == "USA"
            assert q.post_unlearning_acceptable_answer_ids == [
                "l2:usa"]  # NOT California
            assert q.leakage_forbidden_ids == [
                "l0:san_francisco"]  # leakage stays target-relative

    def test_intermediate_probe_keeps_its_ancestor_acceptable(self):
        a = make_assoc(values=["a", "b", "c", "d"], target_level=1)
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["granular_intermediate"])
        for q in qs:
            assert q.expected_level == 2
            assert q.post_unlearning_acceptable_answer_ids == [
                a.levels[2].canonical_id]

    def test_fine_probe_maps_to_target(self):
        a = make_assoc(values=["a", "b", "c"], target_level=1)
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["fine_direct"])
        for q in qs:
            assert q.post_unlearning_acceptable_answer_ids == [
                a.levels[1].canonical_id]

    def test_negation_and_multimodal_map_to_target(self):
        a = make_assoc(values=["a", "b", "c"], target_level=1)
        partition = select_target_retain([a], seed=42)
        qs = generate_queries([a], partition, seed=42,
                              families=["negation_direct",
                                        "multimodal_image_text"])
        for q in qs:
            assert q.post_unlearning_acceptable_answer_ids == [
                a.levels[1].canonical_id]
