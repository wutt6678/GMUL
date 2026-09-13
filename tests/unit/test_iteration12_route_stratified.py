"""Iteration 12 Stage 1c: the route-stratified retention floor.

Stage 1b ended on a measurement-power limit: the between-seed sd on retain-other
row-micro (0.0225, 0.0515) exceeded the shortfall it was adjudicating (-0.0165,
-0.0263), because 76 queries resolve in units of 1/76.  Stage 1c reads the
image-route retention families beside the text-route ones the frozen floor used.

These tests hold the extension to what it claims, from committed artifacts and
code only -- no adapter, no parquet of predictions, no GPU.  A clone must be able
to check that the eight numbers are the frozen four plus four more rather than a
different four, that the text stratum IS the frozen computation and not a copy
of it, that the image stratum is cross-checked against the sealed metric the
same way, that the floor's semantics did not change, that the basis re-derives
from the dataset with no prediction read, that the freeze binds the exact bytes
the re-scoring reads and refuses once a score exists, and that the filed report's
own numbers can be recomputed from the values it prints.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_iter12_route_stratified as air
import build_iter12_route_probe as brp
import freeze_iter12_route_stratification as frs

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.training.seed_replication import FLOOR_NUMBERS

PY = sys.executable
BASIS = REPO_ROOT / brp.OUT_REPORT
FREEZE = REPO_ROOT / frs.OUT_REPORT
REPORT = REPO_ROOT / air.OUT_REPORT
STAGE1 = REPO_ROOT / "data/reports/mllmu_iter12_retention_selection.json"
STAGE1B = REPO_ROOT / "data/reports/mllmu_iter12_seed_replication.json"


def _predictions_available() -> bool:
    try:
        paths = frs.prediction_paths(REPO_ROOT)
    except SystemExit:
        return False
    return bool(paths) and all((REPO_ROOT / rel).exists()
                               for rel in paths.values())


#: The prediction parquets are gitignored, regenerable and large, so a clone has
#: none of them.  Tests that re-verify their BYTES skip there; tests that check
#: what the committed freeze RECORDS about them do not, so the claim stays
#: checked from the commit rather than only on the machine that produced it.
requires_predictions = pytest.mark.skipif(
    not _predictions_available(),
    reason="the bound prediction parquets are gitignored and absent from this "
           "clone, so their bytes cannot be re-verified here; the committed "
           "freeze still records them and the tests that read it still run")


def basis() -> dict:
    return json.loads(BASIS.read_text())


def freeze() -> dict:
    return json.loads(FREEZE.read_text())


def report() -> dict:
    return json.loads(REPORT.read_text())


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(REPO_ROOT / "scripts" / script), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          check=False)


def world(family="retain_same_entity_image", n_ent=3, per=2, correct=None):
    """A tiny probe world as namespaces.

    Only the attributes the frozen helpers actually touch are present, so a
    change in what they read shows up as an AttributeError here rather than as a
    silently different number.
    """
    assocs = [SimpleNamespace(association_id=f"a{e}_{i}", entity_id=f"e{e}")
              for e in range(n_ent) for i in range(per)]
    queries, preds = [], []
    for a in assocs:
        qid = f"q_{a.association_id}"
        queries.append(SimpleNamespace(query_id=qid, family=family,
                                       association_id=a.association_id,
                                       split="train"))
        preds.append(SimpleNamespace(
            query_id=qid,
            is_correct_branch=True if correct is None else correct(qid)))
    return queries, assocs, preds, [f"e{e}" for e in range(n_ent)]


ALL_EIGHT = {k: 0.5 for k in rt.FLOOR_NUMBER_KEYS}
ALL_BELOW = {k: 0.4 for k in rt.FLOOR_NUMBER_KEYS}

IMAGE_FAMILIES = ("retain_same_entity_image", "retain_other_entity_image")


def sealed_stub(n_same, n_other, acc_same=1.0, acc_other=None):
    """A stand-in for the sealed metric over the two image families.

    ``image_route_retention`` hands it the rows of BOTH families at once and
    then reads one block per family, so a stub that returns only the family the
    test cares about makes the other look like a row-count mismatch and the
    guard fires for the wrong reason.
    """
    def stub(rows, queries, associations, split=None):
        return {
            "retain_same_entity_image": {"baseline_accuracy": acc_same,
                                         "num_queries": n_same},
            "retain_other_entity_image": {"baseline_accuracy": acc_other,
                                          "num_queries": n_other},
        }
    return stub


class TestTheEightNumbersAreTheFrozenFourPlusFourMore:
    """An extension that swaps numbers rather than adding them is a new rule."""

    def test_there_are_exactly_eight(self):
        assert len(rt.FLOOR_NUMBER_KEYS) == 8
        assert len(set(rt.FLOOR_NUMBER_KEYS)) == 8

    def test_the_text_half_is_the_frozen_four_verbatim(self):
        text = {k[len("text."):] for k in rt.FLOOR_NUMBER_KEYS
                if k.startswith("text.")}
        assert text == {f"{fam}.{est}" for fam, est in FLOOR_NUMBERS}

    def test_the_two_routes_are_text_and_image(self):
        assert [r for r, _ in rt.ROUTES] == ["text", "image"]

    def test_each_route_reads_the_families_that_belong_to_it(self):
        fams = dict(rt.ROUTES)
        assert fams["text"] == ("retain_same_entity", "retain_other_entity")
        assert fams["text"] == rs.RETENTION_FAMILIES
        assert fams["image"] == IMAGE_FAMILIES

    def test_every_route_family_appears_at_both_estimands(self):
        for route, fams in rt.ROUTES:
            for fam in fams:
                for est in rs.RETENTION_ESTIMANDS:
                    assert rt.number_key(route, fam, est) \
                        in rt.FLOOR_NUMBER_KEYS

    def test_the_pooled_families_floor_nothing(self):
        #: Pooling is the one alternative that would double a denominator, and
        #: it is reported rather than floored.  If it ever entered the floor the
        #: stratified-vs-pooled choice would no longer be visible in a report.
        for fam in rt.POOLED_FAMILIES:
            assert not any(fam in k for k in rt.FLOOR_NUMBER_KEYS)
        assert len(rt.POOLED_FAMILIES) == 2

    def test_the_estimands_are_the_frozen_pair(self):
        assert set(rs.RETENTION_ESTIMANDS) == {"row_micro", "entity_macro"}
        assert len({k.rsplit(".", 1)[1] for k in rt.FLOOR_NUMBER_KEYS}) == 2

    def test_the_scope_is_still_train_and_val(self):
        assert rt.SCOPE_SPLITS == ("train", "val")
        assert basis()["measurement_basis"]["splits"] == ["train", "val"]


class TestTheTextStratumIsTheFrozenComputationNotACopyOfIt:
    """Delegation, so the four filed numbers cannot drift from the extension."""

    def test_the_text_block_is_whatever_probe_retention_returns(self, monkeypatch):
        sentinel = {"marker": "this came from the frozen function"}
        seen = {}

        def fake(predictions, queries, associations, probe_entities):
            seen["called"] = True
            return sentinel

        monkeypatch.setattr(rs, "probe_retention", fake)
        queries, assocs, preds, probe = world(family="retain_same_entity")
        monkeypatch.setattr(rt, "image_route_retention", lambda *a, **k: {})
        monkeypatch.setattr(rt, "pooled_all_routes", lambda *a, **k: {})
        out = rt.probe_retention_by_route(preds, queries, assocs, probe)
        assert seen.get("called") is True
        assert out["routes"]["text"] is sentinel

    def test_the_stratified_result_carries_the_frozen_floor_verdict(self):
        res = rt.floor_check_stratified(ALL_BELOW, ALL_EIGHT)
        frozen = res["frozen_floor_on_the_text_stratum"]
        #: Not a re-derivation: the frozen function's own output, embedded.
        assert frozen == rs.floor_check(rt.text_only_vector(ALL_BELOW),
                                        rt.text_only_vector(ALL_EIGHT))
        assert frozen["eligible"] is False

    def test_the_agreement_check_passes_on_an_honest_result(self):
        assert rt.text_stratum_reproduces_the_frozen_floor(
            rt.floor_check_stratified(ALL_BELOW, ALL_EIGHT)) is True

    def test_it_raises_when_the_text_verdict_disagrees(self):
        res = rt.floor_check_stratified(ALL_BELOW, ALL_EIGHT)
        res["frozen_floor_on_the_text_stratum"]["eligible"] = True
        with pytest.raises(AssertionError,
                           match="disagrees with the rule it extends"):
            rt.text_stratum_reproduces_the_frozen_floor(res)

    def test_it_raises_when_one_number_pass_flag_disagrees(self):
        res = rt.floor_check_stratified(ALL_BELOW, ALL_EIGHT)
        key = rt.number_key("text", "retain_same_entity", "row_micro")
        res["checks"][key]["passes"] = True
        with pytest.raises(AssertionError, match="stratified says"):
            rt.text_stratum_reproduces_the_frozen_floor(res)

    def test_it_raises_when_a_difference_disagrees(self):
        res = rt.floor_check_stratified(ALL_BELOW, ALL_EIGHT)
        key = rt.number_key("text", "retain_other_entity", "row_micro")
        res["checks"][key]["difference"] = 0.0
        with pytest.raises(AssertionError, match="difference"):
            rt.text_stratum_reproduces_the_frozen_floor(res)

    def test_it_ignores_the_image_stratum_entirely(self):
        #: The consistency gate is about the stratum the two rules share.  If it
        #: also policed the image numbers it could not be satisfied by any
        #: candidate that fails on the new stratum, which is all of them.
        vec = dict(ALL_EIGHT)
        vec[rt.number_key("image", "retain_same_entity_image",
                          "row_micro")] = 0.0
        res = rt.floor_check_stratified(vec, ALL_EIGHT)
        assert res["eligible"] is False
        assert res["per_route"] == {"text": True, "image": False}
        assert rt.text_stratum_reproduces_the_frozen_floor(res) is True

    def test_text_only_vector_renests_into_the_frozen_shape(self):
        out = rt.text_only_vector(ALL_EIGHT)
        assert set(out) == set(rs.RETENTION_FAMILIES)
        for fam in rs.RETENTION_FAMILIES:
            assert set(out[fam]) == set(rs.RETENTION_ESTIMANDS)
            assert all(v == 0.5 for v in out[fam].values())


class TestTheFloorSemanticsDidNotChange:
    """Eight conditions instead of four; the comparison itself is identical."""

    def test_the_epsilon_is_the_frozen_one_and_is_not_a_margin(self):
        res = rt.floor_check_stratified(ALL_EIGHT, ALL_EIGHT)
        assert res["epsilon"] == rs.FLOOR_EPSILON == 1e-9
        assert res["epsilon_is_a_tolerance_not_a_margin"] is True
        assert res["eligible"] is True

    def test_an_exact_tie_passes_on_all_eight(self):
        res = rt.floor_check_stratified(ALL_EIGHT, ALL_EIGHT)
        assert len(res["checks"]) == 8
        assert all(c["passes"] for c in res["checks"].values())

    def test_a_shortfall_larger_than_the_epsilon_fails(self):
        vec = dict(ALL_EIGHT)
        vec[rt.FLOOR_NUMBER_KEYS[0]] = 0.5 - 2e-3
        assert rt.floor_check_stratified(vec, ALL_EIGHT)["eligible"] is False

    def test_a_shortfall_below_the_epsilon_still_fails(self):
        #: The smallest difference the measurement can express is ~2e-3, so
        #: 1e-9 cannot rescue anything real; this pins that it does not.
        vec = dict(ALL_EIGHT)
        vec[rt.FLOOR_NUMBER_KEYS[0]] = 0.5 - 1e-6
        res = rt.floor_check_stratified(vec, ALL_EIGHT)
        assert res["checks"][rt.FLOOR_NUMBER_KEYS[0]]["passes"] is False

    def test_a_missing_number_disqualifies_rather_than_passing(self):
        for absent in (None,):
            vec = dict(ALL_EIGHT)
            vec[rt.FLOOR_NUMBER_KEYS[3]] = absent
            res = rt.floor_check_stratified(vec, ALL_EIGHT)
            assert res["eligible"] is False
            check = res["checks"][rt.FLOOR_NUMBER_KEYS[3]]
            assert check["passes"] is False
            assert "not evidence of preserved retention" in check["why"]

    def test_a_number_missing_from_the_anchor_also_disqualifies(self):
        base = dict(ALL_EIGHT)
        base[rt.FLOOR_NUMBER_KEYS[5]] = None
        assert rt.floor_check_stratified(ALL_EIGHT, base)["eligible"] is False

    def test_one_image_failure_alone_disqualifies_a_perfect_text_stratum(self):
        #: The whole point of adding four conditions.  A candidate that
        #: preserves text-route retention and loses photograph-cued retention
        #: must fail, and the report must say which route failed.
        vec = dict(ALL_EIGHT)
        vec[rt.number_key("image", "retain_other_entity_image",
                          "entity_macro")] = 0.0
        res = rt.floor_check_stratified(vec, ALL_EIGHT)
        assert res["eligible"] is False
        assert res["per_route"] == {"text": True, "image": False}
        assert res["frozen_floor_on_the_text_stratum"]["eligible"] is True

    def test_the_direction_is_still_a_lower_bound(self):
        assert rt.floor_check_stratified(ALL_EIGHT, ALL_EIGHT)["direction"] \
            == rs.FLOOR_DIRECTION == "candidate_must_be_at_or_above_b0"

    def test_beating_the_anchor_is_never_disqualifying(self):
        vec = {k: 0.9 for k in rt.FLOOR_NUMBER_KEYS}
        assert rt.floor_check_stratified(vec, ALL_EIGHT)["eligible"] is True


class TestTheImageStratumIsCrossCheckedLikeTheTextOne:
    """The text stratum's load-bearing guard, copied rather than skipped."""

    def test_it_raises_when_the_sealed_metric_returns_a_different_rate(
            self, monkeypatch):
        queries, assocs, preds, probe = world()
        monkeypatch.setattr(rt, "compute_hierarchy_metrics",
                            sealed_stub(len(preds), 0, acc_same=0.1234))
        with pytest.raises(AssertionError,
                           match="probe restriction did not reach"):
            rt.image_route_retention(preds, queries, assocs, probe)

    def test_it_raises_when_the_sealed_metric_counted_different_rows(
            self, monkeypatch):
        #: Independent of the rate check: the rate is right (1.0, everything
        #: correct) and only the count is wrong, so just the second guard can
        #: fire.  If the two checks were one, this test could not fail.
        queries, assocs, preds, probe = world()
        monkeypatch.setattr(rt, "compute_hierarchy_metrics",
                            sealed_stub(1, 0, acc_same=1.0))
        with pytest.raises(AssertionError, match="probe rows were passed"):
            rt.image_route_retention(preds, queries, assocs, probe)

    def test_it_only_sees_probe_entities_and_the_scope_splits(self, monkeypatch):
        queries, assocs, preds, probe = world(n_ent=4, per=2)
        captured = {}

        def spy(rows, q, a, split=None):
            captured["n"] = len(rows)
            captured["split"] = split
            return sealed_stub(len(rows), 0)(rows, q, a, split)

        monkeypatch.setattr(rt, "compute_hierarchy_metrics", spy)
        half = probe[:2]
        out = rt.image_route_retention(preds, queries, assocs, half)
        #: 4 entities x 2 associations, half the entities kept.
        assert captured["n"] == 4
        assert captured["split"] is None
        assert out["retain_same_entity_image"]["num_queries"] == 4
        assert out["retain_same_entity_image"]["num_entities"] == 2

    def test_it_excludes_queries_outside_the_scope_splits(self, monkeypatch):
        queries, assocs, preds, probe = world()
        for q in queries:
            q.split = "test"
        captured = {}

        def spy(rows, q, a, split=None):
            captured["n"] = len(rows)
            return sealed_stub(0, 0, acc_same=None)(rows, q, a, split)

        monkeypatch.setattr(rt, "compute_hierarchy_metrics", spy)
        out = rt.image_route_retention(preds, queries, assocs, probe)
        assert captured["n"] == 0
        assert out["retain_same_entity_image"]["num_queries"] == 0
        assert out["retain_same_entity_image"]["row_micro"] is None

    def test_it_reports_its_own_route_and_source(self, monkeypatch):
        queries, assocs, preds, probe = world()
        monkeypatch.setattr(rt, "compute_hierarchy_metrics",
                            sealed_stub(len(preds), 0))
        out = rt.image_route_retention(preds, queries, assocs, probe)
        assert out["route"] == "image_to_text"
        assert "cross-checked" in out["source"]
        assert out["scope"] == ["train", "val"]

    def test_the_pooled_view_is_marked_as_not_floored(self, monkeypatch):
        queries, assocs, preds, probe = world(family="retain_same_entity")
        monkeypatch.setattr(rt, "compute_hierarchy_metrics",
                            lambda *a, **k: {})
        out = rt.pooled_all_routes(preds, queries, assocs, probe)
        assert out["floored"] is False
        assert "checkable against its alternative" in out["why_reported"]


class TestTheBasisIsDerivedFromTheDatasetAlone:
    """What makes 'identified post hoc' checkable rather than merely confessed."""

    def test_the_committed_basis_re_derives_exactly(self):
        assert brp.build_report(REPO_ROOT) == basis()

    def test_the_basis_says_no_prediction_was_read(self):
        doc = basis()
        assert doc["no_prediction_is_read"]["holds"] is True
        assert not any("predictions" in p
                       for p in doc["no_prediction_is_read"]["inputs"])

    def test_check_only_accepts_the_committed_basis(self):
        assert run("build_iter12_route_probe.py", "--check-only").returncode == 0

    def test_the_partition_is_the_frozen_one_not_a_new_one(self):
        doc = basis()["reuses_partition"]
        assert doc["re_derived_and_matched"] is True
        assert doc["report"] == "data/reports/mllmu_iter12_retention_probe.json"
        partition = json.loads((REPO_ROOT / doc["report"]).read_text())
        assert doc["probe_entities"] == \
            partition["halves"]["probe"]["num_entities"]

    def test_a_divergent_partition_is_refused(self, monkeypatch, tmp_path):
        #: The refusal is the reason the claim above means anything.
        monkeypatch.setattr(brp.bip, "split_entities",
                            lambda ids: (["nope"], sorted(set(ids))))
        with pytest.raises(SystemExit, match="not the frozen one"):
            brp.build_report(REPO_ROOT)

    def test_the_two_routes_cover_identical_associations(self):
        ri = basis()["route_independence"]
        for fam in ("retain_same_entity", "retain_other_entity"):
            assert ri[fam]["association_sets_identical"] is True, fam
            assert ri[fam]["same_probe_entities"] is True, fam

    def test_the_two_routes_share_no_templates(self):
        ri = basis()["route_independence"]
        for fam in ("retain_same_entity", "retain_other_entity"):
            assert ri[fam]["template_sets_disjoint"] is True, fam
            assert ri[fam]["shared_templates"] == [], fam
            assert ri[fam]["routes"] == {"text": ["text_to_text"],
                                         "image": ["image_to_text"]}

    def test_the_text_route_carries_no_image_at_all(self):
        prov = basis()["route_independence"]["text_provenance"]
        assert prov["queries_carrying_an_image"] == 0
        assert prov["probe_text_queries"] == 484

    def test_the_image_provenance_is_recorded_not_hidden(self):
        prov = basis()["route_independence"]["image_provenance"]
        assert prov["probe_image_queries"] == 484
        assert prov["image_seen_in_training"]["seen_in_training"] == 479
        assert prov["image_seen_in_training"]["held_out"] == 5
        assert "seen_photo_unseen_wording" in prov["note"]

    def test_the_denominators_match_the_frozen_basis_per_route(self):
        per = basis()["measurement_basis"]["per_route"]
        frozen = json.loads((REPO_ROOT / brp.PARTITION_REPORT).read_text())
        want = frozen["probe_measurement_basis"]["per_family"]
        for fam in ("retain_same_entity", "retain_other_entity"):
            assert per["text"][fam]["probe_queries"] == \
                want[fam]["probe_queries"], fam
            assert per["text"][fam]["probe_entities"] == \
                want[fam]["probe_entities"], fam
            assert per["image"][fam + "_image"]["probe_queries"] == \
                want[fam]["probe_queries"], fam

    def test_the_power_ceiling_is_computed_for_every_scope(self):
        ceil = basis()["power_ceiling"]
        scopes = [k for k in ceil if isinstance(ceil[k], dict)
                  and "retain_other_total" in ceil[k]]
        assert len(scopes) == 4
        frozen_scope = "train+val, text route (the frozen basis)"
        assert ceil[frozen_scope]["retain_same_total"] == 408
        assert ceil[frozen_scope]["retain_other_total"] == 76
        biggest = "all splits, both routes (the absolute ceiling)"
        assert ceil[biggest]["retain_same_total"] == 1224
        assert ceil[biggest]["retain_other_total"] == 228

    def test_the_ceiling_says_retain_other_is_the_binding_number(self):
        note = basis()["power_ceiling"]["what_the_ceiling_means"]
        assert note["retain_other_is_the_binding_number"] is True
        assert "exhausted" in note["why"]

    def test_the_tolerance_is_below_every_resolvable_gap(self):
        tol = basis()["numerical_tolerance"]
        assert tol["epsilon_is_below_every_resolvable_gap"] is True
        assert tol["floor_epsilon"] == rs.FLOOR_EPSILON
        per = basis()["measurement_basis"]["per_route"]
        gaps = [b[e]["min_resolvable_gap"] for r in per.values()
                for b in r.values() for e in ("row_micro", "entity_macro")]
        assert len(gaps) == 8
        assert tol["smallest_resolvable_gap_over_the_eight_numbers"] == min(gaps)

    def test_the_basis_discloses_that_it_is_not_blind(self):
        doc = basis()
        assert any("identified after Stage 1b was scored" in lim
                   for lim in doc["limitations"])
        assert "not blind" in doc["limitations"][-1]


class TestTheFreezeBindsWhatTheRescoringReads:
    """A re-scoring is only reproducible if its inputs are pinned."""

    def test_the_four_protocol_paths_are_hashed_and_match_the_disk(self):
        hashes = freeze()["hashes"]["protocol_paths"]
        assert set(hashes) == set(frs.PROTOCOL_PATHS)
        for rel, want in hashes.items():
            assert sha(REPO_ROOT / rel) == want, rel

    def test_the_dependencies_include_the_sealed_metric_module(self):
        deps = freeze()["hashes"]["depends_on_unchanged"]
        assert "src/granunlearn/evaluation/hierarchy_metrics.py" in deps
        assert "src/granunlearn/evaluation/retention_selection.py" in deps
        for rel, want in deps.items():
            assert sha(REPO_ROOT / rel) == want, rel

    def test_the_eight_numbers_are_recorded_in_the_freeze(self):
        wf = freeze()["what_is_frozen"]
        assert wf["floor_numbers"] == list(rt.FLOOR_NUMBER_KEYS)
        assert wf["num_numbers"] == 8
        assert wf["epsilon"] == rs.FLOOR_EPSILON
        assert wf["epsilon_is_a_tolerance_not_a_margin"] is True
        assert wf["pooled_reported_but_not_floored"] == \
            list(rt.POOLED_FAMILIES)

    def test_the_freeze_records_a_binding_for_every_prediction_it_reads(self):
        #: Clone-safe: this is what the COMMITTED freeze says, which is the part
        #: a reviewer can check without the gitignored bytes.
        block = freeze()[frs.PREDICTIONS_FIELD]
        assert block["count_expected"] == len(block["paths"]) == 19
        assert len(block["sha256"]) == 19
        assert block["absent_at_freeze_time"] == []
        assert block["paths"] == frs.prediction_paths(REPO_ROOT)
        assert all(len(v) == 64 for v in block["sha256"].values())

    @requires_predictions
    def test_the_recorded_hashes_are_the_bytes_on_disk(self):
        block = freeze()[frs.PREDICTIONS_FIELD]
        for key, rel in block["paths"].items():
            assert sha(REPO_ROOT / rel) == block["sha256"][key], key

    def test_the_seed_42_replicates_point_at_the_stage_1_directory(self):
        #: The reused replicate's predictions were produced by Stage 1, and
        #: reading them from the Stage-1b directory instead would compare a
        #: parent against a parquet that does not exist.
        paths = frs.prediction_paths(REPO_ROOT)
        for key, rel in paths.items():
            if key.endswith("__s42"):
                assert "predictions_iter12/" in rel, key
                assert "predictions_iter12_seeds/" not in rel, key
            elif key.startswith("stage1b.") and not key.endswith("B0"):
                assert "predictions_iter12_seeds/" in rel, key

    def test_no_prediction_path_carries_a_doubled_prefix(self):
        for key, rel in frs.prediction_paths(REPO_ROOT).items():
            assert rel.startswith("data/mllmu_hier_pilot100/"), key
            assert "data/data" not in rel, key

    def test_the_state_lists_come_from_the_frozen_grid(self):
        from granunlearn.training.candidate_grid import iter12_grid
        assert freeze()["states_scored"]["stage1"] == \
            ["MG", *(s.candidate_id for s in iter12_grid())]
        assert len(freeze()["states_scored"]["stage1b"]) == 9

    def test_the_freeze_records_one_measurement_contract(self):
        c = freeze()[frs.CONTRACT_FIELD]
        assert c["one_instrument"] is True
        assert c["distinct_contracts"] == 1
        assert c["num_with_a_sidecar"] == c["num_bound_parquets"] == 19
        assert c["missing_sidecars"] == []
        assert c["contract"]["image_batch_size"] == 1
        assert c["contract_conflicts"] == []

    @requires_predictions
    def test_the_contract_gate_still_finds_one_instrument_on_disk(self):
        c = frs.verify_contract(REPO_ROOT)
        assert c["one_instrument"] is True
        assert c["distinct_contracts"] == 1
        assert c == freeze()[frs.CONTRACT_FIELD]

    def test_the_contract_gate_names_the_keys_it_compared(self):
        c = freeze()[frs.CONTRACT_FIELD]
        assert set(frs.CONTRACT_MUST_MATCH) <= set(c["keys_compared"])
        assert "base_model_revision" in c["keys_compared"]
        assert "dataset.version" in c["keys_compared"]

    def test_a_conflicting_contract_is_reported_as_a_conflict(self, monkeypatch):
        #: Driven from committed files so this also runs in a clone, where the
        #: bound parquets are absent.  What is under test is the conflict branch
        #: of ``verify_contract``, not these particular bytes -- skipping it in
        #: CI would leave the one guard that protects the cross-directory image
        #: comparison untested everywhere but on the authoring machine.
        monkeypatch.setattr(frs, "sidecar_for",
                            lambda root, rel: REPO_ROOT / brp.OUT_REPORT)
        seen = {"n": 0}

        def split(doc):
            seen["n"] += 1
            return {"image_batch_size": 8 if seen["n"] % 2 else 1}

        monkeypatch.setattr(frs, "contract_signature", split)
        out = frs.verify_contract(REPO_ROOT)
        assert out["one_instrument"] is False
        assert out["distinct_contracts"] == 2
        assert len(out["contract_conflicts"]) == 1
        assert "2 distinct measurement contracts" in out["contract_conflicts"][0]
        assert out["missing_sidecars"] == []
        assert out["contract"] is None

    def test_a_missing_sidecar_is_reported_rather_than_passed_over(
            self, monkeypatch):
        monkeypatch.setattr(frs, "sidecar_for",
                            lambda root, rel: root / "absent.provenance.json")
        out = frs.verify_contract(REPO_ROOT)
        assert out["one_instrument"] is False
        assert len(out["missing_sidecars"]) == 19
        assert out["num_with_a_sidecar"] == 0

    def test_clone_dependent_fields_are_not_flat_compared(self):
        #: A clone has none of these bytes.  Comparing them flat would report a
        #: false mismatch there, and skipping them silently would hide a real
        #: one -- so they are named and verified three ways instead.
        assert frs.CLONE_DEPENDENT_FIELDS == ("predictions_bound",
                                              "measurement_contract")
        assert frs.verify_freeze(REPO_ROOT) == []

    def test_check_only_accepts_the_committed_freeze(self):
        assert run("freeze_iter12_route_stratification.py",
                   "--check-only").returncode == 0

    def test_the_freeze_refuses_now_that_a_score_is_filed(self):
        #: State-aware: before the analyzer ran this refusal could not be armed,
        #: and a test asserting the opposite message would have been correct
        #: then and wrong now.
        if not REPORT.exists():
            pytest.skip("no score filed yet, so the refusal is not armed")
        before = sha(FREEZE)
        proc = run("freeze_iter12_route_stratification.py", "--refreeze",
                   "--reason", "a test must not be able to amend this")
        assert proc.returncode != 0
        assert "postdate" in proc.stdout + proc.stderr
        assert sha(FREEZE) == before

    def test_the_refusal_is_evaluated_before_any_flag(self):
        if not REPORT.exists():
            pytest.skip("no score filed yet, so the refusal is not armed")
        before = sha(FREEZE)
        for args in ((), ("--refreeze", "--reason", "x")):
            assert run("freeze_iter12_route_stratification.py",
                       *args).returncode != 0
        assert sha(FREEZE) == before

    def test_the_freeze_records_that_it_trains_nothing(self):
        doc = freeze()
        assert doc["trains_nothing"] is True
        assert doc["gpu_hours_required"] == 0

    def test_the_freeze_records_what_it_does_not_reopen(self):
        doc = freeze()
        assert doc["re_decides_nothing"] is True
        prior = doc["prior_results_it_reports_beside"]
        assert prior["stage1_candidates_total"] == 9
        assert prior["stage1_candidates_disqualified"] == 8
        assert prior["stage1b_parents_passing_on_the_mean"] == 0


class TestTheFiledReportSaysWhatTheEvidenceSays:
    """Recomputed from the values the report itself prints, not trusted."""

    def test_every_consistency_gate_passed(self):
        g = report()["consistency_gates"]
        assert g["all_passed"] is True
        for key in ("freeze_matches_the_repository",
                    "bound_predictions_unchanged",
                    "one_measurement_contract",
                    "basis_re_derives_from_the_dataset",
                    "anchor_text_numbers_match_the_filed_anchor"):
            assert g[key] is True, key
        assert g["predictions_absent_from_this_clone"] == 0

    def test_the_text_stratum_reproduced_both_filed_results(self):
        g = report()["consistency_gates"]
        assert g["stage1_text_stratum"][
            "reproduces_the_filed_stage1_floor_exactly"] is True
        assert g["stage1_text_stratum"]["numbers_compared"] == 36
        assert g["stage1_text_stratum"]["problems"] == []
        assert g["stage1b_text_stratum"][
            "reproduces_the_filed_stage1b_numbers_exactly"] is True
        assert g["stage1b_text_stratum"]["numbers_compared"] == 8
        assert g["stage1b_text_stratum"]["replicates_compared"] == 8
        assert g["stage1b_text_stratum"]["problems"] == []

    def test_the_anchor_text_numbers_equal_the_filed_anchor(self):
        filed = json.loads(STAGE1B.read_text())["anchor"]["values"]
        got = {k[len("text."):]: v
               for k, v in report()["anchor"]["values"].items()
               if k.startswith("text.")}
        assert got == filed

    def test_the_anchor_denominators_show_stratifying_enlarged_nothing(self):
        #: The claim in the disclosure, checked against the report's own
        #: denominators: every image number sits on the same n as its text
        #: counterpart.
        d = report()["anchor"]["denominators"]
        assert d["text.retain_same_entity.row_micro"] == \
            d["image.retain_same_entity_image.row_micro"] == 408
        assert d["text.retain_other_entity.row_micro"] == \
            d["image.retain_other_entity_image.row_micro"] == 76

    def test_the_stage_1_verdict_is_unchanged_by_stratifying(self):
        s = report()["stage1"]["summary"]
        assert s["num_candidates"] == 8
        assert s["eligible_on_the_frozen_four"] == 0
        assert s["eligible_on_the_eight_numbers"] == 0
        assert s["verdict_is_unchanged_by_stratifying"] is True
        assert s["b0_excluded_because_it_is_the_anchor"] is True
        assert "B0" not in s["counted"]

    def test_the_eligibility_counts_are_recomputed_from_the_candidates(self):
        cands = report()["stage1"]["candidates"]
        s = report()["stage1"]["summary"]
        b4 = [c for k, c in cands.items() if k != "B0"]
        assert len(b4) == s["num_candidates"]
        assert sum(1 for c in b4 if c["eligible_on_the_eight"]) == \
            s["eligible_on_the_eight_numbers"]
        assert sum(1 for c in b4 if c["eligible_on_the_frozen_four"]) == \
            s["eligible_on_the_frozen_four"]
        for c in cands.values():
            assert c["floor"]["num_numbers"] == 8
            assert c["eligible_on_the_eight"] == c["floor"]["eligible"]

    def test_the_filed_four_number_verdicts_match_the_committed_stage_1(self):
        filed = json.loads(STAGE1.read_text())["candidates"]
        for cid, c in report()["stage1"]["candidates"].items():
            assert c["eligible_on_the_frozen_four"] == \
                filed[cid]["floor"]["eligible"], cid

    def test_the_image_route_is_the_worse_one_for_most_candidates(self):
        s = report()["stage1"]["summary"]
        cands = {k: v for k, v in report()["stage1"]["candidates"].items()
                 if k != "B0"}
        worse = sum(
            1 for c in cands.values()
            if c["difference_vs_b0"][
                "image.retain_other_entity_image.row_micro"]
            < c["difference_vs_b0"]["text.retain_other_entity.row_micro"])
        assert worse == s["worse_on_the_image_route_than_on_text"] == 7

    def test_the_differences_are_recomputed_from_the_values_and_anchor(self):
        anchor = report()["anchor"]["values"]
        for cid, c in report()["stage1"]["candidates"].items():
            for k in rt.FLOOR_NUMBER_KEYS:
                want = round(c["values"][k] - anchor[k], 6)
                assert c["difference_vs_b0"][k] == want, (cid, k)

    def test_both_parents_fail_on_eight_numbers_and_on_the_frozen_four(self):
        filed = json.loads(STAGE1B.read_text())["verdicts"]
        for parent, v in report()["stage1b"]["verdicts"].items():
            assert v["passes_on_the_mean_eight_numbers"] is False, parent
            assert v["passes_on_the_mean_frozen_four"] == \
                filed[parent]["passes_on_the_mean"], parent
            assert v["per_route"] == {"text": False, "image": False}, parent
            assert v["floor_on_the_mean"]["eligible"] is False

    def test_the_replicate_values_reproduce_the_filed_text_numbers(self):
        filed = json.loads(STAGE1B.read_text())["verdicts"]
        for parent, v in report()["stage1b"]["verdicts"].items():
            assert v["replicates"] == filed[parent]["replicates"], parent
            for fkey, block in filed[parent]["per_number"].items():
                key = f"text.{fkey}"
                assert v["per_number"][key]["values"] == block["values"], \
                    (parent, key)

    def test_the_aggregates_are_recomputed_from_the_values(self):
        for parent, v in report()["stage1b"]["verdicts"].items():
            for key, b in v["per_number"].items():
                vals = b["values"]
                assert len(vals) == b["k"] == 4, (parent, key)
                assert b["mean"] == pytest.approx(statistics.fmean(vals))
                assert b["sd"] == pytest.approx(statistics.stdev(vals))
                assert b["min"] == min(vals)
                assert b["max"] == max(vals)

    def test_the_floor_is_recomputed_from_the_mean_vector(self):
        anchor = report()["anchor"]["values"]
        for parent, v in report()["stage1b"]["verdicts"].items():
            mean_vec = {k: v["per_number"][k]["mean"]
                        for k in rt.FLOOR_NUMBER_KEYS}
            want = rt.floor_check_stratified(mean_vec, anchor)
            got = v["floor_on_the_mean"]
            assert got["eligible"] == want["eligible"], parent
            assert got["per_route"] == want["per_route"], parent
            for k in rt.FLOOR_NUMBER_KEYS:
                assert got["checks"][k]["passes"] == \
                    want["checks"][k]["passes"], (parent, k)

    def test_the_shortfalls_are_stated_in_queries(self):
        for parent, v in report()["stage1b"]["verdicts"].items():
            for key, q in v["shortfall_in_queries"].items():
                d = v["per_number"][key]["mean_minus_anchor"]
                n = report()["anchor"]["denominators"][key]
                assert q == pytest.approx(round(d * n, 4)), (parent, key)

    def test_the_text_numbers_still_straddle_and_the_image_ones_do_not(self):
        #: The finding, recomputed from the classifications the report prints.
        for parent, v in report()["stage1b"]["verdicts"].items():
            text = [b["range_classification"] for k, b in
                    v["per_number"].items() if k.startswith("text.")]
            image = [b["range_classification"] for k, b in
                     v["per_number"].items() if k.startswith("image.")]
            assert text == ["straddles"] * 4, (parent, text)
            assert image.count("every_replicate_below") >= 3, (parent, image)

    def test_the_power_block_agrees_with_the_verdict_block(self):
        for parent, per in report()["stage1b"]["power"]["parents"].items():
            v = report()["stage1b"]["verdicts"][parent]
            for key in rt.FLOOR_NUMBER_KEYS:
                assert per[key]["values"] == v["per_number"][key]["values"], \
                    (parent, key)
                assert per[key]["sd"] == \
                    pytest.approx(v["per_number"][key]["sd"]), (parent, key)
                assert per[key]["range_classification"] == \
                    v["per_number"][key]["range_classification"]

    def test_decidable_means_the_spread_is_smaller_than_the_shortfall(self):
        for parent, per in report()["stage1b"]["power"]["parents"].items():
            for key in rt.FLOOR_NUMBER_KEYS:
                b = per[key]
                dist = b["mean_minus_anchor"]
                want = abs(b["sd"]) < abs(dist) if dist else False
                assert b["decidable"] is want, (parent, key)
                if dist:
                    assert b["sd_over_abs_shortfall"] == \
                        pytest.approx(abs(b["sd"] / dist)), (parent, key)

    def test_the_power_block_says_stratifying_enlarged_no_denominator(self):
        pw = report()["stage1b"]["power"]
        assert pw["stratifying_does_not_enlarge_any_denominator"] is True
        for parent, per in pw["parents"].items():
            for key in rt.FLOOR_NUMBER_KEYS:
                assert per[key]["n_queries"] == \
                    report()["anchor"]["denominators"][key], (parent, key)

    def test_the_pooled_view_is_present_and_floors_nothing(self):
        for parent, per in report()["stage1b"]["power"]["parents"].items():
            pooled = per["pooled_all_routes_reported_not_floored"]
            assert set(pooled) == set(rt.POOLED_FAMILIES), parent
            for fam, b in pooled.items():
                assert b["floored"] is False, (parent, fam)
                #: Parenthesised on purpose: ``assert x == A if c else B``
                #: parses as ``assert (x == A) if c else B``, and the else
                #: branch then asserts a non-zero integer -- always true.
                assert b["n_queries"] == (816 if "same" in fam else 152), \
                    (parent, fam)

    def test_mg_is_reported_but_not_floored(self):
        mg = report()["stage1"]["reference_states_not_floored"]["MG"]
        assert mg["floored"] is False
        assert len(mg["difference_vs_b0"]) == 8
        assert "MG" not in report()["stage1"]["candidates"]

    def test_the_report_says_what_it_cost_and_what_it_does_not_do(self):
        doc = report()
        assert doc["re_decides_nothing"] is True
        assert doc["trains_nothing"] is True
        assert doc["gpu_hours_spent"] == 0
        assert doc["disclosure"][
            "basis_identified_after_stage1b_was_scored"] is True
        assert "does not enlarge any denominator" in \
            doc["disclosure"]["what_stratifying_does_not_do"]

    def test_the_generation_contract_is_recorded_in_the_report(self):
        c = report()["consistency_gates"]["measurement_contract"]
        assert c["one_instrument"] is True
        assert c["distinct_contracts"] == 1
        assert c["contract"]["image_batch_size"] == 1

    @requires_predictions
    def test_check_only_accepts_the_filed_report(self):
        #: Needs the bound parquets: --check-only re-scores from them.  In a
        #: clone the analyzer says which parquet is absent rather than passing,
        #: and the report's CONTENT is still checked by every other test in
        #: this class, all of which read the committed JSON.
        assert run("analyze_iter12_route_stratified.py",
                   "--check-only").returncode == 0

    def test_an_absent_bound_parquet_refuses_rather_than_scoring_fewer(
            self, monkeypatch):
        #: The invariant behind the clone-side behaviour: a re-scoring that
        #: quietly dropped one state would still produce a plausible report.
        #: Whichever guard fires first, it must refuse.
        paths = frs.prediction_paths(REPO_ROOT)
        first = min(paths)
        paths[first] = "data/mllmu_hier_pilot100/absent.parquet"
        monkeypatch.setattr(frs, "prediction_paths", lambda root: paths)
        with pytest.raises(SystemExit):
            air.build_report(REPO_ROOT)


class TestNothingFrozenWasEditedToGetAnyOfThis:
    """The extension is additive; the two prior studies still verify."""

    def test_the_stage_1_protocol_freeze_still_verifies(self):
        assert run("freeze_iter12_selection_protocol.py",
                   "--check-only").returncode == 0

    def test_the_stage_1b_freeze_still_verifies(self):
        assert run("freeze_iter12_seed_replication.py",
                   "--check-only").returncode == 0

    def test_the_frozen_partition_basis_still_verifies(self):
        assert run("build_iter12_retention_probe.py",
                   "--check-only").returncode == 0

    def test_the_sealed_confirmation_paths_are_untouched(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import audit_confirmation_execution as ace
        conf = json.loads((REPO_ROOT / "data/reports/"
                           "mllmu_pilot100_confirmation_freeze.json"
                           ).read_text())
        sealed = set(ace.sealed_paths(conf))
        frozen = freeze()
        touched = set(frozen["hashes"]["protocol_paths"])
        for group in frozen["hashes"].values():
            touched |= set(group)
        #: hierarchy_metrics is sealed AND a declared dependency; it is bound to
        #: be left alone, never edited.  The check is that no OTHER sealed path
        #: appears as something this study owns.
        own = set(frozen["hashes"]["protocol_paths"])
        assert not (own & sealed)
        assert "src/granunlearn/evaluation/hierarchy_metrics.py" in \
            frozen["hashes"]["depends_on_unchanged"]
        assert touched  # non-vacuous: something was actually checked

    def test_the_filed_stage_1_and_stage_1b_reports_are_untouched(self):
        for path in (STAGE1, STAGE1B):
            out = subprocess.run(
                ["git", "diff", "--name-only", "HEAD", "--",
                 str(path.relative_to(REPO_ROOT))],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            assert out.stdout.strip() == "", path.name
