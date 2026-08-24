"""Unit tests for Iteration 6 query generation + validation."""

from __future__ import annotations

import pytest

from granunlearn.evaluation.query_generation import (
    FAMILIES,
    SPLITS,
    FAMILY_TEMPLATES,
    QueryRecord,
    answer_level_for_family,
    generate_queries,
    template_index,
    validate_queries,
    validate_negation_no_leak,
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
        level=i,
        canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value,
        normalized_value=value.lower().strip(),
        parent_id=None,
    )


def make_assoc(
    assoc_id: str = "a1",
    values: list[str] | None = None,
    target_level: int = 1,
    attribute_name: str = "occupation",
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
        entity_id=f"e_{assoc_id}",
        entity_name="Jane Doe",
        attribute_name=attribute_name,
        hierarchy_type="numeric" if attribute_name == "salary" else "semantic",
        levels=levels,
        original_level=0,
        target_level=target_level,
        source_modalities=["text", "image"] if images else ["text"],
        images=images,
        textual_context=[f"Jane Doe — {attribute_name}: {values[0]}"],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"),
    )


class TestAnswerLevels:
    def test_fine_families_answer_level_zero(self):
        a = make_assoc()
        for fam in FAMILIES:
            if fam.startswith("fine_"):
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

    def test_negation_and_multimodal_use_target(self):
        a = make_assoc(target_level=2)
        for fam in ("negation_direct", "negation_correction",
                    "negation_disambiguation", "multimodal_image_text"):
            assert answer_level_for_family(a, fam) == 2

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

    def test_seed_changes_assignment(self):
        seen = set()
        for seed in range(20):
            seen.add(template_index(seed, "a1", "fine_direct", "train"))
        assert len(seen) == 3  # hash distributes over all templates

    def test_all_families_have_three_templates(self):
        for fam, templates in FAMILY_TEMPLATES.items():
            assert len(templates) == len(SPLITS), fam


class TestGenerateQueries:
    def test_full_generation_counts(self):
        assocs = [make_assoc("a1"), make_assoc("a2", with_image=False)]
        queries = generate_queries(assocs, seed=42)
        # a1: 13 families x 3 splits; a2: 12 (no multimodal) x 3
        assert len(queries) == (13 + 12) * 3

    def test_split_coverage_per_association_family(self):
        queries = generate_queries([make_assoc()], seed=42)
        for fam in FAMILIES:
            splits = sorted(q.split for q in queries if q.family == fam)
            assert splits == sorted(SPLITS)

    def test_answers_are_exact_level_strings(self):
        a = make_assoc()
        valid = {lv.value for lv in a.levels}
        for q in generate_queries([a], seed=42):
            assert q.answer in valid
            assert q.answer == a.levels[q.answer_level].value

    def test_fine_answers_are_original_value(self):
        a = make_assoc()
        for q in generate_queries([a], seed=42):
            if q.family.startswith("fine_"):
                assert q.answer == "Data Scientist"

    def test_deterministic_regeneration(self):
        a = make_assoc()
        q1 = [q.model_dump() for q in generate_queries([a], seed=42)]
        q2 = [q.model_dump() for q in generate_queries([a], seed=42)]
        assert q1 == q2

    def test_multimodal_queries_carry_image(self):
        queries = generate_queries([make_assoc()], seed=42)
        mm = [q for q in queries if q.family == "multimodal_image_text"]
        assert mm and all(q.image_path and q.modality == "image_text"
                          for q in mm)

    def test_negation_correction_contains_distractor(self):
        a = make_assoc(target_level=1)  # distractor = original fine value
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        assert all("Data Scientist" in q.query_text for q in queries)
        assert all(q.answer == "Analyst" for q in queries)
        assert all("Analyst" not in q.query_text for q in queries)

    def test_negation_correction_two_level_chain_no_leak(self):
        """2-level chains (salary/height, target=1) previously leaked the
        answer as the distractor; the fine value must be the distractor."""
        a = make_assoc(values=["$120,000", "$100K-$150K"], target_level=1,
                       attribute_name="salary")
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        assert all(q.answer == "$100K-$150K" for q in queries)
        assert all("$120,000" in q.query_text for q in queries)
        assert all(q.answer not in q.query_text for q in queries)

    def test_target_zero_uses_coarsest_distractor(self):
        """target_level=0 -> answer is levels[0]; distractor falls back to
        the coarsest level, which differs, so no leak is possible."""
        a = make_assoc().model_copy(update={"target_level": 0})
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        assert all(q.answer == "Data Scientist" for q in queries)
        assert all("Professional" in q.query_text for q in queries)

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError):
            generate_queries([make_assoc()], families=["not_a_family"])

    def test_query_ids_unique(self):
        queries = generate_queries(
            [make_assoc("a1"), make_assoc("a2")], seed=42)
        ids = [q.query_id for q in queries]
        assert len(ids) == len(set(ids))


class TestValidateQueries:
    def test_clean_generation_passes(self):
        assocs = [make_assoc("a1"), make_assoc("a2")]
        queries = generate_queries(assocs, seed=42)
        errors, stats = validate_queries(queries, assocs)
        assert errors == []
        assert stats["num_queries"] == len(queries)
        assert stats["by_split"] == {"train": 26, "val": 26, "test": 26}

    def test_bad_answer_detected(self):
        a = make_assoc()
        queries = generate_queries([a], seed=42)
        queries[0] = queries[0].model_copy(update={"answer": "Fabricated"})
        errors, _ = validate_queries(queries, [a])
        assert any("not in hierarchy levels" in e for e in errors)

    def test_answer_level_mismatch_detected(self):
        a = make_assoc()
        queries = generate_queries([a], seed=42)
        queries[0] = queries[0].model_copy(update={"answer_level": 99})
        errors, _ = validate_queries(queries, [a])
        assert errors  # index error or mismatch flagged

    def test_missing_split_detected(self):
        a = make_assoc()
        queries = [q for q in generate_queries([a], seed=42)
                   if not (q.family == "fine_direct" and q.split == "test")]
        errors, _ = validate_queries(queries, [a])
        assert any("split coverage" in e for e in errors)

    def test_repeated_template_detected(self):
        a = make_assoc()
        queries = generate_queries([a], seed=42)
        # Force a paraphrase repeat within fine_direct across splits
        dd = [q for q in queries if q.family == "fine_direct"]
        queries = [q for q in queries if q.family != "fine_direct"]
        dd[1] = dd[1].model_copy(update={"paraphrase_group": dd[0].paraphrase_group})
        errors, _ = validate_queries(queries + dd, [a])
        assert any("templates repeat" in e for e in errors)

    def test_retain_fact_dedupe(self):
        a = make_assoc()
        queries = generate_queries([a], seed=42)
        errors, _ = validate_queries(
            queries, [a], retain_facts={queries[0].answer})
        assert any("retain fact" in e for e in errors)

    def test_unknown_association_detected(self):
        a = make_assoc()
        queries = generate_queries([a], seed=42)
        errors, _ = validate_queries(queries, [])
        assert any("unknown association" in e for e in errors)


class TestNegationNoLeak:
    def test_clean_queries_pass(self):
        a = make_assoc(values=["May 12, 1985", "May 1985", "1985"],
                       target_level=2, attribute_name="date_of_birth")
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        assert validate_negation_no_leak(queries, [a]) == []

    def test_distractor_equals_answer_flagged(self):
        a = make_assoc(values=["X", "Y"], target_level=1)
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        # Sabotage: force the answer to equal the fine value (distractor).
        queries = [q.model_copy(update={"answer": "X"}) for q in queries]
        errors = validate_negation_no_leak(queries, [a])
        assert len(errors) == len(queries)

    def test_substring_of_longer_distractor_is_allowed(self):
        """'1985' inside quoted 'May 12, 1985' is a legitimate correction
        prompt, not a leak."""
        a = make_assoc(values=["May 12, 1985", "May 1985", "1985"],
                       target_level=2, attribute_name="date_of_birth")
        queries = generate_queries([a], seed=42,
                                   families=["negation_correction"])
        assert any("May 12, 1985" in q.query_text for q in queries)
        assert validate_negation_no_leak(queries, [a]) == []
