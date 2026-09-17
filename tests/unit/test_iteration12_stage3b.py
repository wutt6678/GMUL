"""Iteration 12 Stage 3b: the preregistered post-Stage-3 decision tree.

These tests pin the amendment: the shortfall is unrounded and disqualifies on a
missing number, the near-miss envelope is B6's own measured shortfall rather
than an invented variance estimate, all three conjuncts bite, only B7 rows can
become replication parents, the ranking is the prescribed one, the replicated
mean is scored by the stratified floor on all eight numbers, D_G is the
distance of the mean vector and not the mean of the distances, and the
amendment was filed while no B7 adapter existed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import freeze_iter12_stage3 as fis3
import select_iter12_stage3 as sis3

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.selection import SUMMARY_COMPONENTS
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training import stage3b_replication as s3b

PY = sys.executable
EPS = rs.FLOOR_EPSILON
B6_ID = "B6_beta13.642_lam0.5_lr2e-05_ep5"
B6_PARQUET = (REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12_stage2"
              / f"predictions_tv_{B6_ID}.parquet")


def freeze() -> dict:
    return json.loads((REPO_ROOT / s3g.FREEZE_REPORT).read_text())


def anchor() -> dict[str, float]:
    return freeze()["what_is_frozen"]["anchor_values"]


def tree() -> dict:
    return freeze()["post_stage3_decision_tree"]


def envelope() -> dict:
    return s3b.incumbent_envelope(anchor())


def values_with_shortfall(shortfall: dict[str, float]) -> dict[str, float]:
    """A floor vector whose shortfall against the anchor is ``shortfall``.

    Built by subtracting from the anchor, so ``d_j`` comes out as
    ``s - epsilon`` rather than ``s``; the tests that need an exact shortfall
    use B6's own measured values instead.
    """
    return {k: anchor()[k] - shortfall.get(k, 0.0) for k in s3b.FLOOR_KEYS}


def rows_from(shortfalls: dict[str, dict[str, float]],
              methods: dict[str, str],
              distances: dict[str, float | None]) -> dict[str, dict]:
    return {cid: {"method": methods[cid],
                  "values": values_with_shortfall(s),
                  "distance_to_mg": distances[cid]}
            for cid, s in shortfalls.items()}


TEXT_KEYS = [k for k in s3b.FLOOR_KEYS if k.startswith("text.")]
IMAGE_KEYS = [k for k in s3b.FLOOR_KEYS if k.startswith("image.")]


# ──────────────────────────────────────────────────────────────────────
class TestTheShortfall:
    def test_a_pass_or_an_exact_tie_has_zero_shortfall(self):
        above = {k: anchor()[k] + 0.01 for k in s3b.FLOOR_KEYS}
        assert all(v == 0.0 for v in
                   s3b.shortfall_vector(above, anchor()).values())
        tie = dict(anchor())
        assert all(v == 0.0 for v in
                   s3b.shortfall_vector(tie, anchor()).values())

    def test_the_epsilon_is_kept_and_nothing_is_rounded(self):
        key = "image.retain_same_entity_image.entity_macro"
        vals = dict(anchor())
        vals[key] = anchor()[key] - 0.0339
        got = s3b.shortfall_vector(vals, anchor())[key]
        assert got == pytest.approx(0.0339, abs=1e-8)
        assert got < 0.0339, "the shortfall must carry the -1e-9"
        assert got != 0.0339
        assert round(got, 4) == 0.0339, (
            "a report's four-decimal difference field would not distinguish "
            "this from the anchor, which is why the tree forbids reading one")

    def test_a_missing_number_disqualifies_rather_than_counting_as_zero(self):
        vals = dict(anchor())
        vals[IMAGE_KEYS[0]] = None
        with pytest.raises(ValueError, match="disqualification"):
            s3b.shortfall_vector(vals, anchor())
        del vals[IMAGE_KEYS[1]]
        with pytest.raises(ValueError, match="disqualification"):
            s3b.shortfall_vector(vals, anchor())


# ──────────────────────────────────────────────────────────────────────
class TestTheEnvelopeIsTheIncumbentsOwnShortfall:
    def test_k_m_s_are_b6s_three_image_failures(self):
        env = envelope()
        assert env["K"] == 3
        assert env["M"] == pytest.approx(0.0339, abs=1e-8)
        assert env["M"] < 0.0339
        assert env["S"] == pytest.approx(0.0725, abs=1e-8)
        assert env["S"] < 0.0725, "S carries the -1e-9 of each failed number"
        assert env["failed_metrics"] == sorted([
            "image.retain_same_entity_image.entity_macro",
            "image.retain_same_entity_image.row_micro",
            "image.retain_other_entity_image.row_micro",
        ])
        assert all(env["per_metric"][k] == 0.0 for k in TEXT_KEYS), (
            "B6 clears every text number, so the envelope is entirely an "
            "image-route statement")

    def test_the_filed_envelope_is_the_one_the_code_derives(self):
        filed = tree()["incumbent_relative_envelope"]
        env = envelope()
        assert filed["K"] == env["K"]
        assert filed["M"] == env["M"]
        assert filed["S"] == env["S"]
        assert filed["per_metric_shortfall"] == env["per_metric"]
        assert filed["failed_metrics"] == env["failed_metrics"]
        assert filed["b6_floor_values"] == s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME
        assert filed["incumbent"] == B6_ID

    def test_the_sum_is_order_independent(self):
        env = envelope()
        reversed_shortfalls = {k: env["per_metric"][k]
                               for k in reversed(s3b.FLOOR_KEYS)}
        assert s3b.envelope(reversed_shortfalls)["S"] == env["S"]

    @pytest.mark.skipif(not B6_PARQUET.exists(),
                        reason="the gitignored B6 parquet is absent here")
    def test_the_envelope_reproduces_from_the_bound_predictions(self):
        from granunlearn.evaluation.reference_eval import (
            load_associations_parquet,
            load_predictions_parquet,
            load_queries_parquet,
        )
        data = s3g.dataset_dir(REPO_ROOT)
        queries = load_queries_parquet(data / "queries.parquet")
        assocs = load_associations_parquet(data / "associations.parquet")
        probe = json.loads((REPO_ROOT / "data/reports/"
                                      "mllmu_iter12_retention_probe.json"
                            ).read_text())["halves"]["probe"]["entities"]
        preds = load_predictions_parquet(B6_PARQUET)
        recomputed = rt.floor_vector(rt.probe_retention_by_route(
            preds, queries, assocs, probe))
        assert s3b.verify_incumbent_values(recomputed) == []
        got = s3b.envelope(s3b.shortfall_vector(recomputed, anchor()))
        env = envelope()
        assert (got["K"], got["M"], got["S"]) == (env["K"], env["M"], env["S"])

    def test_the_freeze_says_why_it_is_not_a_variance_estimate(self):
        why = tree()["incumbent_relative_envelope"]["why_incumbent_relative"]
        assert "never" in why and "TEXT" in why
        assert "no referent" in why


# ──────────────────────────────────────────────────────────────────────
class TestQualification:
    def test_an_exact_pass_qualifies(self):
        q = s3b.qualification(
            s3b.shortfall_vector(values_with_shortfall({}), anchor()),
            envelope())
        assert q["classification"] == s3b.EXACT_PASS
        assert q["qualifies"] is True and q["exact_pass"] is True
        assert q["near_miss"] is False

    def test_b6_satisfies_its_own_envelope_with_equality(self):
        #: This is why the parent-eligibility exclusion has to be explicit:
        #: without it the zero-weight control would select itself.
        q = s3b.qualification(
            s3b.shortfall_vector(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME,
                                 anchor()),
            envelope())
        assert q["classification"] == s3b.NEAR_MISS
        assert q["qualifies"] is True
        assert all(q["conjuncts"].values())
        assert q["K"] == 3

    def test_a_b7_that_repairs_two_of_the_three_is_a_near_miss(self):
        short = {k: 0.0123 for k in IMAGE_KEYS[:1]}
        q = s3b.qualification(
            s3b.shortfall_vector(values_with_shortfall(short), anchor()),
            envelope())
        assert q["classification"] == s3b.NEAR_MISS
        assert q["K"] == 1 and q["violated_conjuncts"] == []

    def test_the_count_conjunct_bites_on_its_own(self):
        short = {k: 0.001 for k in TEXT_KEYS}  # four tiny failures
        q = s3b.qualification(
            s3b.shortfall_vector(values_with_shortfall(short), anchor()),
            envelope())
        assert q["classification"] == s3b.DOES_NOT_QUALIFY
        assert q["violated_conjuncts"] == ["failed_metrics_at_most_K"]

    def test_the_max_conjunct_bites_on_its_own(self):
        short = {IMAGE_KEYS[0]: 0.05}  # one catastrophic failure
        q = s3b.qualification(
            s3b.shortfall_vector(values_with_shortfall(short), anchor()),
            envelope())
        assert q["classification"] == s3b.DOES_NOT_QUALIFY
        assert q["violated_conjuncts"] == ["max_shortfall_at_most_M"]

    def test_the_total_conjunct_bites_on_its_own(self):
        #: Three failures of 0.03 each -- exactly the three numbers B6 fails,
        #: so K == 3 and M < M_B6 both hold and only S rejects it. Without S
        #: the rule would admit a shortfall spread evenly over the route.
        short = {k: 0.03 for k in envelope()["failed_metrics"]}
        assert len(short) == 3
        q = s3b.qualification(
            s3b.shortfall_vector(values_with_shortfall(short), anchor()),
            envelope())
        assert q["K"] == 3
        assert q["M"] < envelope()["M"]
        assert q["classification"] == s3b.DOES_NOT_QUALIFY
        assert q["violated_conjuncts"] == ["total_shortfall_at_most_S"]


# ──────────────────────────────────────────────────────────────────────
class TestParentEligibilityAndRanking:
    def four_b7(self) -> list[str]:
        return [c.candidate_id for c in
                s3g.b7_rows(s3g.stage3_grid(
                    json.loads((REPO_ROOT / s2g.CALIBRATION_REPORT)
                               .read_text())["grid_rule"]["beta_star"]))]

    def test_only_b7_rows_can_be_parents(self):
        ids = self.four_b7()
        methods = {"B0": "B0", B6_ID: s2g.METHOD_ANCHOR,
                   **{i: s3g.METHOD_ROUTE_ANCHOR for i in ids}}
        shortfalls = {cid: {} for cid in methods}  # every row an exact pass
        rows = rows_from(shortfalls, methods,
                         {cid: 0.05 for cid in methods})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        out = s3b.select_parents(rows, anchor(), B6_ID)
        assert out["incumbent"]["envelope_reproduces"] is True
        assert set(out["parents"]) <= set(ids)
        assert B6_ID not in out["parents"] and "B0" not in out["parents"]
        assert out["per_candidate"][B6_ID]["parent_eligible"] is False
        assert "incumbent" in out["per_candidate"][B6_ID]["excluded_because"]
        assert out["per_candidate"]["B0"]["parent_eligible"] is False

    def test_an_exact_pass_outranks_a_cheaper_near_miss(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR,
                   **{i: s3g.METHOD_ROUTE_ANCHOR for i in ids}}
        shortfalls = {B6_ID: {}, ids[0]: {}, ids[1]: {IMAGE_KEYS[0]: 0.01},
                      ids[2]: {IMAGE_KEYS[0]: 0.02},
                      ids[3]: {IMAGE_KEYS[0]: 0.09}}
        rows = rows_from(shortfalls, methods,
                         {B6_ID: 0.06, ids[0]: 0.09, ids[1]: 0.01,
                          ids[2]: 0.02, ids[3]: 0.03})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        out = s3b.select_parents(rows, anchor(), B6_ID)
        assert out["parents"][0] == ids[0], (
            "an exact pass precedes a near miss even at a worse D_G")
        assert out["parents"][1] == ids[1]
        assert out["held_back"] == [ids[2]]
        assert ids[3] not in out["ranking_of_qualifiers"]
        assert out["per_candidate"][ids[3]]["classification"] == \
            s3b.DOES_NOT_QUALIFY
        assert set(out["per_candidate"][ids[3]]["violated_conjuncts"]) == {
            "max_shortfall_at_most_M", "total_shortfall_at_most_S"}

    def test_near_misses_order_by_k_then_m_then_s_then_d_g_then_id(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR,
                   **{i: s3g.METHOD_ROUTE_ANCHOR for i in ids}}
        shortfalls = {
            B6_ID: {},
            ids[0]: {IMAGE_KEYS[0]: 0.02},                       # K=1, M=.02
            ids[1]: {IMAGE_KEYS[0]: 0.01, IMAGE_KEYS[1]: 0.01},  # K=2, best D_G
            ids[2]: {IMAGE_KEYS[0]: 0.03},                       # K=1, M=.03
            ids[3]: {IMAGE_KEYS[0]: 0.02},                       # ties ids[0]
        }
        rows = rows_from(shortfalls, methods,
                         {B6_ID: 0.06, ids[0]: 0.03, ids[1]: 0.01,
                          ids[2]: 0.02, ids[3]: 0.03})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        out = s3b.select_parents(rows, anchor(), B6_ID)
        assert out["ranking_of_qualifiers"] == [ids[0], ids[3], ids[2], ids[1]]
        assert ids[0] < ids[3], "the last step of the key is the plain id"
        assert out["ranking_of_qualifiers"].index(ids[2]) < \
            out["ranking_of_qualifiers"].index(ids[1]), (
                "K precedes M: one failure beats two even when the two-failure "
                "row has both a smaller M and a smaller D_G")
        assert out["parents"] == [ids[0], ids[3]]

    def test_at_most_two_parents_are_taken(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR,
                   **{i: s3g.METHOD_ROUTE_ANCHOR for i in ids}}
        rows = rows_from({cid: {} for cid in methods}, methods,
                         {cid: 0.05 for cid in methods})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        out = s3b.select_parents(rows, anchor(), B6_ID)
        assert out["max_parents"] == s3b.MAX_PARENTS == 2
        assert len(out["parents"]) == 2
        assert len(out["held_back"]) == 2

    def test_no_qualifying_b7_closes_iteration_12(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR,
                   **{i: s3g.METHOD_ROUTE_ANCHOR for i in ids}}
        shortfalls = {B6_ID: {},
                      **{i: {k: 0.09 for k in IMAGE_KEYS} for i in ids}}
        rows = rows_from(shortfalls, methods,
                         {cid: 0.05 for cid in methods})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        out = s3b.select_parents(rows, anchor(), B6_ID)
        assert out["parents"] == []
        assert out["decision"] == s3b.CLOSE_ITERATION_12
        assert out["decision"] == tree()["parent_eligibility"][
            "if_no_b7_candidate_qualifies"]

    def test_a_qualifying_b7_opens_stage_3b(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR, ids[0]: s3g.METHOD_ROUTE_ANCHOR}
        rows = rows_from({B6_ID: {}, ids[0]: {}}, methods,
                         {B6_ID: 0.06, ids[0]: 0.05})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        assert s3b.select_parents(rows, anchor(), B6_ID)["decision"] == \
            s3b.RUN_STAGE_3B

    def test_a_missing_incumbent_row_is_an_error_not_a_default(self):
        with pytest.raises(LookupError, match="no envelope can be derived"):
            s3b.select_parents({"B0": {"method": "B0", "values": {},
                                       "distance_to_mg": None}},
                               anchor(), B6_ID)

    def test_an_unmeasurable_d_g_cannot_be_ranked(self):
        ids = self.four_b7()
        methods = {B6_ID: s2g.METHOD_ANCHOR, ids[0]: s3g.METHOD_ROUTE_ANCHOR}
        rows = rows_from({B6_ID: {}, ids[0]: {}}, methods,
                         {B6_ID: 0.06, ids[0]: None})
        rows[B6_ID]["values"] = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        with pytest.raises(ValueError, match="not computable"):
            s3b.select_parents(rows, anchor(), B6_ID)

    def test_the_selector_gates_the_envelope_before_spending_a_gpu(self):
        good = sis3.gate_incumbent_envelope(
            dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME), anchor())
        assert good["incumbent_envelope"]["holds"] is True
        assert good["incumbent_envelope"]["K"] == 3
        perturbed = dict(s3b.B6_FLOOR_VALUES_AT_AMENDMENT_TIME)
        perturbed[IMAGE_KEYS[0]] += 0.0001
        bad = sis3.gate_incumbent_envelope(perturbed, anchor())
        assert bad["incumbent_envelope"]["holds"] is False
        assert bad["incumbent_envelope"]["problems"]


# ──────────────────────────────────────────────────────────────────────
class TestTheReplicatedMean:
    def per_seed_eights(self, deltas: dict[int, float]) -> dict[int, dict]:
        return {seed: {k: anchor()[k] + d for k, d in
                       ((k, deltas[seed]) for k in s3b.FLOOR_KEYS)}
                for seed in s3b.ALL_SEEDS}

    def test_the_mean_is_componentwise_and_unrounded(self):
        deltas = {42: 0.0, 43: 0.0001, 44: 0.0002, 45: 0.0003}
        out = s3b.mean_floor_values(self.per_seed_eights(deltas))
        key = s3b.FLOOR_KEYS[0]
        assert out["values"][key] == pytest.approx(
            anchor()[key] + 0.00015, abs=1e-12)
        assert out["values"][key] != round(out["values"][key], 4), (
            "the mean must not be rounded to its inputs' four decimals")
        assert out["mean_is_compared_unrounded"] is True
        assert out["seeds"] == [42, 43, 44, 45]
        assert out["per_number"][key]["k"] == 4

    def test_a_subset_of_seeds_is_a_different_statistic_and_is_refused(self):
        deltas = {42: 0.0, 43: 0.0001, 44: 0.0002, 45: 0.0003}
        per_seed = self.per_seed_eights(deltas)
        del per_seed[45]
        with pytest.raises(ValueError, match="preregistered"):
            s3b.mean_floor_values(per_seed)
        vectors = {seed: {c: 0.5 for c in SUMMARY_COMPONENTS}
                   for seed in s3b.ALL_SEEDS}
        del vectors[43]
        with pytest.raises(ValueError, match="preregistered"):
            s3b.mean_summary_vector(vectors)

    def test_an_unmeasurable_number_is_refused_not_averaged_as_zero(self):
        per_seed = self.per_seed_eights({42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0})
        per_seed[44][IMAGE_KEYS[0]] = None
        with pytest.raises(ValueError, match="disqualifies"):
            s3b.mean_floor_values(per_seed)

    def test_the_mean_is_scored_by_the_stratified_floor_on_all_eight(self):
        good = s3b.mean_floor_check(
            {k: anchor()[k] + 0.01 for k in s3b.FLOOR_KEYS}, anchor())
        assert good["eligible"] is True
        assert good["num_numbers"] == 8
        assert good["scored_by"].endswith("floor_check_stratified")
        assert good["the_stage1b_analyzer_was_not_reused"] is True
        assert good["text_stratum_reproduces_the_frozen_floor"] is True

        bad = {k: anchor()[k] + 0.01 for k in s3b.FLOOR_KEYS}
        bad[IMAGE_KEYS[1]] = anchor()[IMAGE_KEYS[1]] - 0.02
        out = s3b.mean_floor_check(bad, anchor())
        assert out["eligible"] is False
        assert out["per_route"] == {"text": True, "image": False}, (
            "a text-only analyzer would have called this mean eligible")

    def test_the_stage1b_rule_really_would_drop_the_image_numbers(self):
        #: The non-substitutability claim, measured rather than asserted:
        #: Stage 1b's frozen floor_check reads only the four text families, so
        #: an image-route failure is invisible to it.
        candidate = {k: anchor()[k] + 0.01 for k in s3b.FLOOR_KEYS}
        candidate[IMAGE_KEYS[1]] = anchor()[IMAGE_KEYS[1]] - 0.5
        legacy = rs.floor_check(rt.text_only_vector(candidate),
                                rt.text_only_vector(anchor()))
        assert legacy["eligible"] is True
        assert len(legacy["checks"]) == 4
        assert s3b.mean_floor_check(candidate, anchor())["eligible"] is False


# ──────────────────────────────────────────────────────────────────────
class TestDGOnTheMeanVector:
    def per_seed_vectors(self) -> dict[int, dict]:
        base = {c: 0.5 for c in SUMMARY_COMPONENTS}
        out = {seed: dict(base) for seed in s3b.ALL_SEEDS}
        out[42]["filr"] = 0.9
        out[43]["filr"] = 0.1
        return out

    def test_the_mean_vector_is_componentwise_over_every_component(self):
        mean = s3b.mean_summary_vector(self.per_seed_vectors())
        assert set(mean) == set(SUMMARY_COMPONENTS)
        assert mean["filr"] == pytest.approx(0.5, abs=1e-12)
        assert mean["wrong"] == 0.5

    def test_distance_of_the_mean_is_not_the_mean_of_the_distances(self):
        mg = {c: 0.5 for c in SUMMARY_COMPONENTS}
        per_seed = self.per_seed_vectors()
        mean = s3b.mean_summary_vector(per_seed)
        got, used = s3b.d_g_of_mean_vector(mean, mg)
        assert got == 0.0 and used == list(SUMMARY_COMPONENTS)
        per_seed_dg = {seed: rs.distance_to_mg(v, mg)[0]
                       for seed, v in per_seed.items()}
        assert per_seed_dg[42] == per_seed_dg[43] == 0.057143
        assert per_seed_dg[44] == per_seed_dg[45] == 0.0
        disclosure = s3b.per_seed_d_g_disclosure(per_seed_dg)
        assert disclosure["mean_of_per_seed_d_g"] == pytest.approx(0.0285715)
        assert disclosure["mean_of_per_seed_d_g"] != got, (
            "the two statistics must actually differ, or the rule that picks "
            "one of them would be untestable")
        assert disclosure["decision_bearing"] is False
        assert "not the distance of the mean" in disclosure["why_not_used"]

    def test_an_incomplete_mean_vector_is_refused(self):
        per_seed = self.per_seed_vectors()
        del per_seed[45]["wrong"]
        with pytest.raises(ValueError, match="renormalize"):
            s3b.mean_summary_vector(per_seed)


# ──────────────────────────────────────────────────────────────────────
class TestTheTerminalBranches:
    def test_no_mean_eligible_parent_stops_iteration_12(self):
        out = s3b.replication_outcome(
            {"B7_a": {"mean_floor_eligible": False, "d_g_of_mean_vector": 0.04},
             "B7_b": {"mean_floor_eligible": False, "d_g_of_mean_vector": 0.03}})
        assert out["decision"] == s3b.STOP_ITERATION_12
        assert out["selected"] is None and out["next_step"] == "none"

    def test_one_survivor_is_selected_and_its_d_g_is_descriptive(self):
        out = s3b.replication_outcome(
            {"B7_a": {"mean_floor_eligible": True, "d_g_of_mean_vector": 0.09},
             "B7_b": {"mean_floor_eligible": False, "d_g_of_mean_vector": 0.01}})
        assert out["decision"] == s3b.SELECT_THE_ONLY_PARENT
        assert out["selected"] == "B7_a"
        assert out["d_g_is_descriptive"] is True
        assert out["next_step"] == s3b.ITERATION_13

    def test_several_survivors_are_decided_by_minimum_d_g(self):
        out = s3b.replication_outcome(
            {"B7_a": {"mean_floor_eligible": True, "d_g_of_mean_vector": 0.09},
             "B7_b": {"mean_floor_eligible": True, "d_g_of_mean_vector": 0.03}})
        assert out["decision"] == s3b.SELECT_BY_MINIMUM_D_G
        assert out["selected"] == "B7_b"
        assert out["d_g_is_descriptive"] is False

    def test_a_d_g_tie_falls_to_the_frozen_tie_break(self):
        out = s3b.replication_outcome(
            {"B7_z": {"mean_floor_eligible": True, "d_g_of_mean_vector": 0.03},
             "B7_a": {"mean_floor_eligible": True, "d_g_of_mean_vector": 0.03}})
        assert out["selected"] == "B7_a"
        assert out["tied_with"] == ["B7_z"]
        assert out["tie_break"] == "(distance_to_mg, candidate_id) ascending"

    def test_the_branches_are_the_ones_the_freeze_files(self):
        filed = tree()["terminal_branches"]
        assert filed["no_mean_eligible_parent"] == s3b.STOP_ITERATION_12
        assert filed["exactly_one_mean_eligible_parent"]["decision"] == \
            s3b.SELECT_THE_ONLY_PARENT
        assert filed["exactly_one_mean_eligible_parent"][
            "d_g_is_descriptive"] is True
        assert filed["more_than_one_mean_eligible_parent"]["decision"] == \
            s3b.SELECT_BY_MINIMUM_D_G
        assert filed["selected_parent"] == s3b.ITERATION_13
        assert filed["iteration_13_is_separately_frozen"] is True


# ──────────────────────────────────────────────────────────────────────
class TestTheReplicateSet:
    def grid(self) -> list[s3g.Stage3CandidateSpec]:
        beta = json.loads((REPO_ROOT / s2g.CALIBRATION_REPORT).read_text())[
            "grid_rule"]["beta_star"]
        return s3g.stage3_grid(beta)

    def test_seed_42_is_reused_and_the_other_three_are_trained(self):
        parent = s3g.b7_rows(self.grid())[0].candidate_id
        reps = s3b.replicates_for([parent], self.grid())
        assert [r.seed for r in reps] == [42, 43, 44, 45]
        assert [r.reused for r in reps] == [True, False, False, False]
        assert [r.replicate_id for r in reps] == \
            [f"{parent}__s{s}" for s in (42, 43, 44, 45)]
        assert len(s3b.to_train(reps)) == 3

    def test_a_replicate_changes_only_the_seed(self):
        spec = s3g.b7_rows(self.grid())[0]
        rep = s3b.replicates_for([spec.candidate_id], self.grid())[1]
        assert rep.overrides == {**spec.overrides, "seed": 43}
        assert rep.spec.image_anchor_weight == spec.image_anchor_weight
        assert rep.spec.effective_anchor_weight == \
            spec.effective_anchor_weight

    def test_a_non_b7_parent_is_refused(self):
        with pytest.raises(LookupError, match="only B7 rows"):
            s3b.replicates_for([B6_ID], self.grid())
        with pytest.raises(LookupError, match="not in the frozen Stage-3 grid"):
            s3b.replicates_for(["B7_invented"], self.grid())

    def test_the_reused_seed_reads_stage3_and_new_seeds_write_stage3b(self):
        spec = s3g.b7_rows(self.grid())[0]
        reps = s3b.replicates_for([spec.candidate_id], self.grid())
        assert s3b.adapter_dir(REPO_ROOT, reps[0]) == \
            s3g.stage3_ckpt_root(REPO_ROOT) / spec.candidate_id / "adapters"
        assert s3b.adapter_dir(REPO_ROOT, reps[1]) == \
            s3b.stage3b_ckpt_root(REPO_ROOT) / reps[1].replicate_id / "adapters"

    def test_the_new_namespaces_collide_with_nothing_and_double_nothing(self):
        for path in (s3b.stage3b_ckpt_root(REPO_ROOT),
                     s3b.stage3b_predictions_dir(REPO_ROOT)):
            assert "data/data" not in str(path)
        assert s3b.stage3b_ckpt_root(REPO_ROOT) != \
            s3g.stage3_ckpt_root(REPO_ROOT)
        assert s3b.stage3b_predictions_dir(REPO_ROOT) != \
            s3g.stage3_predictions_dir(REPO_ROOT)
        assert s3b.OUT_REPORT != s3g.OUT_REPORT
        assert s3b.OUT_REPORT != s3g.FREEZE_REPORT


# ──────────────────────────────────────────────────────────────────────
class TestTheAmendmentIsFiled:
    def test_the_amendment_records_the_three_required_facts(self):
        amendments = freeze()["amendments"]
        assert amendments, "the decision tree must be filed as an amendment"
        last = amendments[-1]
        reason = last["reason"]
        assert "did not contain" in reason, "the tree was previously absent"
        assert "never been measured" in reason, "image-route seed variance"
        assert "no B7 adapter" in reason and "no B7 prediction" in reason
        assert last["no_adapter_existed_at_amendment_time"] is True
        assert len(last["supersedes_sha256"]) == 64

    def test_the_amendment_changed_the_tree_and_nothing_in_the_criterion(self):
        last = freeze()["amendments"][-1]
        changed = set(last["fields_changed"])
        assert any(c.startswith("post_stage3_decision_tree.") for c in changed)
        assert any(c.startswith("hashes.protocol_paths.") for c in changed)
        assert not any(c.startswith("what_is_frozen.") for c in changed), (
            "the eight floors, the anchor, epsilon, margin and the tie-break "
            "are untouched by this amendment")
        assert not any(c.startswith("grid.rows") for c in changed), (
            "the frozen weight grid is untouched by this amendment")

    def test_the_decision_module_is_hash_bound_as_protocol(self):
        rel = "src/granunlearn/training/stage3b_replication.py"
        assert rel in fis3.PROTOCOL_PATHS
        assert rel in freeze()["hashes"]["protocol_paths"]
        assert tree()["implemented_by"]["module"] == rel
        assert tree()["implemented_by"]["executor_scripts_do_not_exist_yet"] \
            is True

    def test_the_stage3b_mechanics_are_the_frozen_ones(self):
        mech = tree()["stage3b_mechanics"]
        assert mech["seeds"]["all"] == [42, 43, 44, 45]
        assert mech["seeds"]["reused"] == s3b.BASE_SEED
        assert mech["seeds"]["trained"] == [43, 44, 45]
        assert mech["scored_by"].endswith("floor_check_stratified")
        assert mech["the_stage1b_analyzer_must_not_be_reused"] is True
        assert mech["d_g"]["components"] == list(SUMMARY_COMPONENTS)
        assert "averaging already-computed per-seed D_G" in \
            mech["d_g"]["forbidden"]
        assert mech["mean"].startswith(
            "the arithmetic mean of each metric over 42/43/44/45")

    def test_check_only_still_passes_after_the_amendment(self):
        p = subprocess.run(
            (PY, "scripts/freeze_iter12_stage3.py", "--check-only"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stdout + p.stderr

    def test_no_b7_adapter_or_prediction_exists_yet(self):
        assert fis3.adapters_exist(REPO_ROOT) == []
        assert not s3b.stage3b_ckpt_root(REPO_ROOT).exists()
        assert not (REPO_ROOT / s3g.OUT_REPORT).exists()

    def test_the_refusal_is_still_armed_on_the_amended_freeze(self):
        doc = freeze()
        assert doc["refusal_policy"]["unconditional"] is True
        assert doc["refusal_policy"]["no_flag_reaches_past_it"] is True
        assert "replication" in doc["refusal_policy"]["reason"]
