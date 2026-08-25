"""Unit tests for the frozen Iteration 8 hierarchy metrics.

Literal acceptance examples from the Iteration 8 requirement
(target = California):
    output = San Francisco  -> under_forgetting (fine leakage / FILR hit)
    output = USA            -> over_forgetting
    output = Ontario        -> wrong_branch
These three must remain SEPARATE failure modes, plus explicit
refusal/hallucination categories, strata, and test-split primacy.
"""

from __future__ import annotations

import pytest

from granunlearn.evaluation.hierarchy_metrics import (
    FAILURE_CATEGORIES,
    ROUTE_SLOTS,
    build_pool_value_index,
    classify_target_failure,
    compute_hierarchy_metrics,
    export_failure_cases,
    is_refusal,
    match_pool_value,
)
from granunlearn.evaluation.scoring import score_query
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ProvenanceInfo,
    QueryRecord,
    SplitInfo,
)


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value, normalized_value=value.lower())


def make_assoc(aid: str, entity_id: str, values: list[str],
               target_level: int = 1,
               hierarchy_type: str = "semantic",
               attribute_name: str = "residence") -> AssociationRecord:
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name=f"Person {entity_id}", attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=[_level(i, v) for i, v in enumerate(values)],
        original_level=0, target_level=target_level,
        images=[], split=SplitInfo(split="test"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


# The literal Iteration 8 pool: California chain + an Ontario chain so
# wrong-branch has somewhere to come from.
CA = make_assoc("e1__residence", "e1",
                ["San Francisco", "California", "USA"], target_level=1)
ON = make_assoc("e2__residence", "e2",
                ["Toronto", "Ontario", "Canada"], target_level=1)
POOL = [CA, ON]
INDEX = build_pool_value_index(POOL)

SF_ID, CA_ID, USA_ID = (lv.canonical_id for lv in CA.levels)


def fine_probe(query_id: str = "q_fine", split: str = "test",
               family: str = "fine_direct",
               route: str = "text_to_text") -> QueryRecord:
    return QueryRecord(
        query_id=query_id, association_id=CA.association_id,
        route=route, query_type="fine_direct", family=family,
        image_ids=[], prompt="Where does Person e1 live?",
        expected_level=0, acceptable_answer_ids=[SF_ID],
        forbidden_descendant_ids=[],
        unlearning_target_level=1, leakage_forbidden_ids=[SF_ID],
        post_unlearning_acceptable_answer_ids=[CA_ID],
        split=split, adversarial=False,
        target_association_id=CA.association_id)


def coarse_probe(query_id: str = "q_coarse",
                 split: str = "test") -> QueryRecord:
    return QueryRecord(
        query_id=query_id, association_id=CA.association_id,
        route="text_to_text", query_type="ancestor",
        family="granular_coarse",
        image_ids=[], prompt="In which country does Person e1 live?",
        expected_level=2, acceptable_answer_ids=[USA_ID],
        forbidden_descendant_ids=[],
        unlearning_target_level=1, leakage_forbidden_ids=[SF_ID],
        post_unlearning_acceptable_answer_ids=[USA_ID],
        split=split, adversarial=False,
        target_association_id=CA.association_id)


def predict(q: QueryRecord, raw: str):
    return score_query(q, CA, raw, "exp", "MU")


class TestFailureTaxonomy:
    """target = California: the three failure modes stay separate."""

    def test_finer_output_is_under_forgetting(self):
        p = predict(fine_probe(), "San Francisco")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "under_forgetting"

    def test_target_output_is_correct(self):
        p = predict(fine_probe(), "California")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "correct_at_target"

    def test_ancestor_output_is_over_forgetting(self):
        p = predict(fine_probe(), "USA")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "over_forgetting"

    def test_other_chain_value_is_wrong_branch(self):
        p = predict(fine_probe(), "Ontario")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "wrong_branch"

    def test_refusal_phrase(self):
        p = predict(fine_probe(), "I don't know where they live.")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "refusal"

    def test_unmatched_confident_is_hallucination(self):
        p = predict(fine_probe(), "They live in Atlantis.")
        assert classify_target_failure(
            fine_probe(), CA, p, INDEX) == "hallucination"

    def test_three_modes_are_distinct_categories(self):
        cats = {classify_target_failure(fine_probe(), CA, predict(
            fine_probe(), raw), INDEX)
            for raw in ("San Francisco", "USA", "Ontario")}
        assert cats == {"under_forgetting", "over_forgetting",
                        "wrong_branch"}
        assert len(cats) == 3  # never collapse into one "incorrect"

    def test_acceptable_beats_coarse_for_granular_probes(self):
        """A granular_coarse probe legitimately requests the ancestor —
        'USA' is CORRECT there, not over-forgetting."""
        q = coarse_probe()
        p = predict(q, "USA")
        assert classify_target_failure(q, CA, p, INDEX) == \
            "correct_at_target"


class TestPoolValueIndex:
    def test_longest_match_wins(self):
        # 'Ontario' exists only in the other chain
        hit = match_pool_value("Ontario", INDEX, CA.association_id)
        assert hit is not None and hit[1] == ON.association_id

    def test_self_chain_values_excluded(self):
        assert match_pool_value(
            "California", INDEX, CA.association_id) is None

    def test_word_boundary_for_single_tokens(self):
        pool = POOL + [make_assoc("e3__hobby", "e3",
                                  ["Art", "Crafts"], target_level=1,
                                  attribute_name="hobby")]
        idx = build_pool_value_index(pool)
        # 'artist' contains 'art' as a substring but not as a word
        assert match_pool_value("an artist", idx, CA.association_id) is None
        assert match_pool_value(
            "they study Art", idx, CA.association_id) is not None


class TestRefusalDetection:
    @pytest.mark.parametrize("text", [
        "I don't know.", "No information is available.",
        "I cannot answer that.", "There is no public record.",
    ])
    def test_refusals(self, text):
        assert is_refusal(text)

    def test_non_refusal(self):
        assert not is_refusal("They live in Atlantis.")


class TestHierarchyMetrics:
    def _preds_and_queries(self):
        qs = [fine_probe("q1", split="test"),
              fine_probe("q2", split="train"),
              coarse_probe("q3", split="test")]
        preds = [
            predict(qs[0], "San Francisco"),   # under-forgetting (FILR)
            predict(qs[1], "California"),      # correct
            predict(qs[2], "USA"),             # ancestor correct
        ]
        return preds, qs

    def test_filr_and_tga(self):
        preds, qs = self._preds_and_queries()
        m = compute_hierarchy_metrics(preds, qs, POOL, split="test")
        assert m["filr"] == 0.5       # 1 of 2 test target probes leaks
        assert m["tga"] == 0.5        # the granular_coarse probe hits USA
        assert m["failure_taxonomy"]["under_forgetting"] == 1
        assert m["failure_taxonomy"]["over_forgetting"] == 0

    def test_taxonomy_sums_to_target_probes(self):
        preds, qs = self._preds_and_queries()
        m = compute_hierarchy_metrics(preds, qs, POOL)
        assert sum(m["failure_taxonomy"].values()) == \
            m["num_target_probes"] == 3
        assert set(m["failure_taxonomy"]) == set(FAILURE_CATEGORIES)

    def test_split_filter_changes_metrics(self):
        preds, qs = self._preds_and_queries()
        pooled = compute_hierarchy_metrics(preds, qs, POOL)
        test = compute_hierarchy_metrics(preds, qs, POOL, split="test")
        train = compute_hierarchy_metrics(preds, qs, POOL, split="train")
        assert pooled["num_target_probes"] == 3
        assert test["num_target_probes"] == 2
        assert train["num_target_probes"] == 1
        assert train["tga"] == 1.0  # the train probe answered California

    def test_ancestor_retention(self):
        preds, qs = self._preds_and_queries()
        m = compute_hierarchy_metrics(preds, qs, POOL)
        assert m["ancestor_retention"]["num_queries"] == 1
        assert m["ancestor_retention"]["baseline_accuracy"] == 1.0
        assert m["ancestor_retention"]["post_unlearning_accuracy"] == 1.0

    def test_route_slots_report_absence(self):
        preds, qs = self._preds_and_queries()
        m = compute_hierarchy_metrics(preds, qs, POOL)
        assert set(m["by_route"]) == set(ROUTE_SLOTS)
        assert m["by_route"]["image_to_text"]["num_queries"] == 0
        assert m["by_route"]["text_to_text"]["num_queries"] == 3

    def test_hierarchy_type_and_depth_strata(self):
        preds, qs = self._preds_and_queries()
        m = compute_hierarchy_metrics(preds, qs, POOL)
        assert m["by_hierarchy_type"]["semantic"]["num_queries"] == 3
        assert m["by_hierarchy_type"]["numeric"]["num_queries"] == 0
        assert m["by_target_depth"]["1"]["num_queries"] == 3

    def test_adversarial_excluded_from_headlines(self):
        q = fine_probe("qadv")
        q = q.model_copy(update={"adversarial": True,
                                 "family": "negation_correction"})
        preds = [predict(q, "San Francisco")]
        m = compute_hierarchy_metrics(preds, [q], POOL)
        assert m["num_target_probes"] == 0
        assert m["filr"] is None


class TestFailureExport:
    def test_export_contains_failures_only(self):
        qs = [fine_probe("q1"), fine_probe("q2")]
        preds = [predict(qs[0], "San Francisco"), predict(qs[1], "California")]
        export = export_failure_cases(preds, qs, POOL, checkpoint_id="MU")
        assert export["num_correct_target_probes"] == 1
        assert export["num_failure_cases"] == 1
        case = export["cases"][0]
        assert case["category"] == "under_forgetting"
        assert case["query_id"] == "q1"
        assert case["raw_output"]
        assert export["failure_counts"] == {"under_forgetting": 1}

    def test_retain_probes_excluded(self):
        q = fine_probe("qret")
        q = q.model_copy(update={"family": "retain_same_entity"})
        preds = [score_query(q, CA, "zebra", "exp", "MU")]
        export = export_failure_cases(preds, [q], POOL, checkpoint_id="MU")
        assert export["num_failure_cases"] == 0
