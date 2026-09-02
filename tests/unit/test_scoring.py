"""Unit tests for the deterministic hierarchical scorer + separation gate."""

from __future__ import annotations

from granunlearn.datasets.smoke import select_target_retain
from granunlearn.evaluation.query_generation import (
    answer_level_for_family,
    generate_queries,
)
from granunlearn.evaluation.scoring import (
    _is_negated,
    compute_metrics,
    match_answer,
    score_query,
    separation_gate,
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
        level=i, canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value, normalized_value=value.lower().strip())


def make_assoc(aid="a1", entity_id="e1",
               values=None, target_level=1,
               attribute_name="residence",
               hierarchy_type="numeric") -> AssociationRecord:
    # Disjoint strings: no level value may be a substring of another,
    # otherwise finest-wins matching resolves to the finer level.
    values = values or ["San Francisco", "California", "USA"]
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name="Alice", attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=[_level(i, v) for i, v in enumerate(values)],
        original_level=0, target_level=target_level,
        images=[ImageRef(image_id=f"img_{aid}", path=f"i/{aid}.jpg",
                         source="materialized", split="train")],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


class TestMatchAnswer:
    def test_exact_fine_match(self):
        a = make_assoc()
        lv, cid, neg = match_answer("San Francisco", a)
        assert lv == 0 and cid == "l0:san_francisco" and neg == []

    def test_finest_match_wins_when_multiple_present(self):
        """An output containing the fine value reveals the fine value —
        leakage regardless of surrounding coarser text."""
        a = make_assoc()
        lv, cid, _ = match_answer("San Francisco, California, USA", a)
        assert lv == 0

    def test_coarse_only_match(self):
        a = make_assoc()
        lv, cid, _ = match_answer("She lives in California.", a)
        assert lv == 1 and cid == "l1:california"

    def test_normalization_case_and_whitespace(self):
        a = make_assoc()
        lv, _, _ = match_answer("  sAn   FRANCISCO ", a)
        assert lv == 0

    def test_no_match(self):
        a = make_assoc()
        lv, cid, _ = match_answer("I don't know.", a)
        assert lv is None and cid is None

    def test_empty_output(self):
        a = make_assoc()
        assert match_answer("", a) == (None, None, [])


class TestNegationAwareness:
    """Substring matching must not count DENIED values as revealed
    (Iteration 7 scorer audit).  Cues are word-boundary based."""

    def test_negated_fine_match_is_skipped(self):
        # 3 levels so the denied fine value can fall through to a coarse
        # match; single-token values keep cue matching unambiguous
        a = make_assoc(values=["Sydney", "NSW", "Oceania"])
        lv, cid, neg = match_answer(
            "The birthplace is not Sydney; it is NSW.", a)
        assert lv == 1
        assert neg == ["Sydney"]

    def test_negation_contraction(self):
        a = make_assoc(values=["Sydney", "NSW", "Oceania"])
        lv, cid, neg = match_answer("It isn't Sydney.", a)
        assert lv is None and cid is None
        assert neg == ["Sydney"]

    def test_multiword_cue(self):
        a = make_assoc(values=["Sydney", "NSW", "Oceania"])
        lv, cid, neg = match_answer(
            "She lives in NSW rather than Sydney.", a)
        # the fine value appears but is denied; the coarse one stands
        assert lv == 1
        assert "Sydney" in neg

    def test_nottingham_does_not_trigger_not(self):
        """The audit found a naive substring cue false-positiving on
        'Nottingham'; token-wise comparison must not repeat that."""
        a = make_assoc(values=["Nottingham", "England", "UK"])
        lv, cid, neg = match_answer("Nottingham, England.", a)
        assert lv == 0 and cid is not None and neg == []

    def test_cues_are_word_bounded(self):
        assert not _is_negated("knot gardenia", len("knot gardenia"))
        assert _is_negated("the answer is not", len("the answer is not "))

    def test_negated_matches_recorded_in_metadata(self):
        pool = [make_assoc("e1__res", "e1",
                           values=["Sydney", "NSW", "Oceania"]),
                make_assoc("e1__occ", "e1", attribute_name="occupation",
                           hierarchy_type="semantic",
                           values=["Data Scientist", "Analyst"]),
                make_assoc("e1__height", "e1", attribute_name="height",
                           values=["180 cm", "tall-band"])]
        partition = select_target_retain(pool, seed=42)
        queries = generate_queries(pool, partition, seed=42,
                                   families=["fine_direct"])
        target = next(
            q for q in queries
            if q.association_id in partition["target_association_ids"])
        assoc = next(a for a in pool
                     if a.association_id == target.association_id)
        fine_value = assoc.levels[0].value
        p = score_query(target, assoc, f"It is not {fine_value}.",
                        "exp", "MN")
        assert p.matched_canonical_id is None or \
            p.matched_canonical_id != assoc.levels[0].canonical_id
        assert p.metadata["negated_matches"] == [fine_value]


class TestSplitMetrics:
    """Test-paraphrase metrics must be reportable separately from the
    pooled train/val/test numbers (Iteration 7 review)."""

    def test_split_filter_partitions_predictions(self):
        pool = [make_assoc("e1__res", "e1",
                           values=["Sydney", "NSW", "Oceania"]),
                make_assoc("e1__occ", "e1", attribute_name="occupation",
                           hierarchy_type="semantic",
                           values=["Data Scientist", "Analyst"]),
                make_assoc("e1__height", "e1", attribute_name="height",
                           values=["180 cm", "tall-band"])]
        partition = select_target_retain(pool, seed=42)
        queries = generate_queries(pool, partition, seed=42)
        by_id = {a.association_id: a for a in pool}
        preds = [score_query(q, by_id[q.association_id], "Sydney",
                             "exp", "MF") for q in queries]
        pooled = compute_metrics(preds, queries)
        per_split = {s: compute_metrics(preds, queries, split=s)
                     for s in ("train", "val", "test")}
        assert pooled["num_queries"] == len(queries)
        assert sum(m["num_queries"] for m in per_split.values()) == \
            pooled["num_queries"]
        for s, m in per_split.items():
            assert m["num_queries"] > 0
            assert sum(1 for q in queries if q.split == s) == \
                m["num_queries"]


class TestScoreQuery:
    def _queries(self):
        a = make_assoc()
        partition = select_target_retain([a], seed=42)
        queries = generate_queries([a], partition, seed=42,
                                   families=["fine_direct", "granular_fine"])
        return a, partition, queries

    def test_prediction_record_fields(self):
        a, _, queries = self._queries()
        q = next(x for x in queries if x.family == "fine_direct")
        p = score_query(q, a, "California", "exp", "MG")
        assert p.matched_canonical_id == "l1:california"
        assert p.predicted_level == 1
        assert p.is_correct_branch is False        # baseline asks level 0
        assert p.is_finer_than_target is False     # California is target
        assert p.metadata["post_acceptable_match"] is True

    def test_fine_output_is_leakage_on_target_probe(self):
        a, _, queries = self._queries()
        q = next(x for x in queries if x.family == "granular_fine")
        p = score_query(q, a, "San Francisco", "exp", "MF")
        assert p.is_finer_than_target is True
        assert p.metadata["post_acceptable_match"] is False

    def test_unmatched_output(self):
        a, _, queries = self._queries()
        q = queries[0]
        p = score_query(q, a, "no idea", "exp", "MN")
        assert p.matched_canonical_id is None
        assert p.predicted_level is None
        assert p.is_correct_branch is False


class TestMetricsAndGate:
    def _pool(self):
        """2 entities x (residence, occupation, height): per entity the
        partition yields 1 semantic + 1 numeric target and 1 retained
        association, so every gate slice is populated.  Level values are
        pairwise disjoint (no substring relations) so scripted answers
        match exactly one level."""
        return [
            make_assoc("e1__res", "e1",
                       values=["Sydney", "New South Wales", "Oceania"]),
            make_assoc("e1__occ", "e1", attribute_name="occupation",
                       hierarchy_type="semantic",
                       values=["Data Scientist", "Analyst"]),
            make_assoc("e1__height", "e1", attribute_name="height",
                       values=["180 cm", "tall-band"]),
            make_assoc("e2__res", "e2",
                       values=["Osaka", "Kansai", "Asia"]),
            make_assoc("e2__occ", "e2", attribute_name="occupation",
                       hierarchy_type="semantic",
                       values=["Consultant", "Advisor"]),
            make_assoc("e2__height", "e2", attribute_name="height",
                       values=["165 cm", "mid-band"]),
        ]

    @staticmethod
    def _raw_for(state: str, q, assoc) -> str:
        """Scripted state behavior.  MF answers fine probes (and over-
        answers everything else) with the fine value; MG answers every
        probe at the level the state was trained on (fine probes at the
        TARGET level, others at the requested level); MN answers only
        retain probes."""
        if (q.family or "").startswith("retain_"):
            return assoc.levels[0].value
        if state == "MN":
            return "I don't know."
        if (q.family or "").startswith("fine_") \
                or q.family == "image_fine_direct":
            idx = 0 if state == "MF" else assoc.target_level
        else:
            idx = answer_level_for_family(assoc, q.family)
            if state == "MF":
                idx = 0
        return assoc.levels[idx].value

    def _scored(self, pool, state):
        partition = select_target_retain(pool, seed=42)
        queries = generate_queries(pool, partition, seed=42)
        by_id = {a.association_id: a for a in pool}
        preds = [
            score_query(q, by_id[q.association_id],
                        self._raw_for(state, q, by_id[q.association_id]),
                        "exp", state)
            for q in queries
        ]
        return compute_metrics(preds, queries)

    def test_gate_passes_with_clean_separation(self):
        pool = self._pool()
        metrics = {s: self._scored(pool, s) for s in ("MF", "MG", "MN")}
        # sanity of the scripted separation
        assert metrics["MF"]["fine_recovery"]["baseline_accuracy"] == 1.0
        assert metrics["MG"]["fine_recovery"]["baseline_accuracy"] == 0.0
        assert metrics["MG"]["target_core"][
            "post_unlearning_accuracy"] == 1.0
        assert metrics["MN"]["target_core"][
            "post_unlearning_accuracy"] == 0.0
        passed, reasons = separation_gate(metrics)
        assert passed, reasons

    def test_gate_fails_when_mg_equals_mn(self):
        pool = self._pool()
        metrics = {s: self._scored(pool, s) for s in ("MF", "MN")}
        metrics["MG"] = metrics["MN"]  # MG without target knowledge
        passed, reasons = separation_gate(metrics)
        assert not passed
        assert any("MG" in r and "MN" in r for r in reasons)

    def test_gate_fails_when_state_missing(self):
        passed, reasons = separation_gate({"MF": {}, "MG": {}})
        assert not passed and any("missing" in r for r in reasons)

    def test_adversarial_excluded_from_core(self):
        pool = [make_assoc(values=["Sydney", "New South Wales", "Oceania"])]
        partition = select_target_retain(pool, seed=42)
        queries = generate_queries(pool, partition, seed=42)
        preds = []
        for q in queries:
            # adversarial probes answered at the target level, everything
            # else wrong
            raw = ("New South Wales"
                   if q.family == "negation_correction" else "no idea")
            preds.append(score_query(q, pool[0], raw, "exp", "MF"))
        metrics = compute_metrics(preds, queries)
        assert metrics["adversarial_only"]["baseline_accuracy"] == 1.0
        assert metrics["fine_recovery"]["baseline_accuracy"] == 0.0

    def test_adversarial_fine_answer_is_leakage_not_correct(self):
        """Answering a negation_correction probe with the quoted fine
        value is leakage, not baseline correctness — which is exactly why
        the family is excluded from the core FILR average."""
        pool = [make_assoc(values=["Sydney", "New South Wales", "Oceania"])]
        partition = select_target_retain(pool, seed=42)
        queries = generate_queries(pool, partition, seed=42,
                                   families=["negation_correction"])
        p = score_query(queries[0], pool[0], "Sydney", "exp", "MG")
        assert p.is_correct_branch is False
        assert p.is_finer_than_target is True
