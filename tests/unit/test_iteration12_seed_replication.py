"""Iteration 12 Stage 1b: the seed-replication protocol.

Stage 1 disqualified all eight B4 replay candidates against a retention floor
that is a POINT comparison over one training seed.  Two of them missed
retain-same by 3 and 4 queries out of 408, which is the size of a rounding
error in a verdict.  Stage 1b re-runs those two at four seeds so the floor can
be asked of a mean.

These tests hold the protocol to what it claims, from committed artifacts only:
no adapter, no parquet and no GPU.  A clone must be able to check that the
parents were chosen by a stated rule and not by the criterion, that the seeds
were fixed before training, that the primary rule is the mean with no margin,
that the freeze and the result do not share a path, that the measurement
contract is inherited while the study identity differs, and that nothing frozen
was edited to get any of it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_iter12_seed_replication as aisr
import freeze_iter12_seed_replication as fisr
import freeze_iter12_selection_protocol as fip
import train_iter12_seed_replicates as tisr

from granunlearn.evaluation import retention_selection as rs
from granunlearn.training.candidate_grid import grid_for_tag
from granunlearn.training.seed_replication import (
    ALL_SEEDS,
    BASE_SEED,
    FLOOR_NUMBERS,
    FREEZE_REPORT,
    NEW_SEEDS,
    OUT_REPORT,
    PARENTS,
    SEED_CKPT_ROOT,
    aggregate,
    mean_candidate,
    parent_specs,
    range_classification,
    replicate_id,
    replicates,
    to_train,
)

PY = sys.executable


def stage1() -> dict:
    return json.loads(
        (REPO_ROOT / "data/reports/mllmu_iter12_retention_selection.json")
        .read_text())


def seed_freeze() -> dict:
    return json.loads((REPO_ROOT / FREEZE_REPORT).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────
class TestTheReplicateSetIsWhatTheProtocolSays:
    def test_two_parents_times_four_seeds(self):
        assert len(replicates()) == len(PARENTS) * len(ALL_SEEDS) == 8

    def test_the_reused_seed_is_not_retrained(self):
        assert [r.replicate_id for r in to_train()] == [
            replicate_id(p, s) for p in PARENTS for s in NEW_SEEDS]
        assert len(to_train()) == 6

    def test_exactly_the_base_seed_is_marked_reused(self):
        assert {r.replicate_id for r in replicates() if r.reused} == {
            replicate_id(p, BASE_SEED) for p in PARENTS}

    def test_seeds_are_the_frozen_ones_in_order(self):
        assert ALL_SEEDS == (42, 43, 44, 45)
        assert NEW_SEEDS == (43, 44, 45)
        assert [r.seed for r in replicates()[:4]] == list(ALL_SEEDS)

    def test_parents_exist_in_the_frozen_stage1_grid(self):
        grid = {c.candidate_id for c in grid_for_tag("iter12")}
        assert set(PARENTS) <= grid
        assert set(parent_specs()) == set(PARENTS)

    def test_order_is_by_construction_not_by_discovery(self):
        ids = [r.replicate_id for r in replicates()]
        assert ids == sorted(ids, key=lambda i: (
            PARENTS.index(i.rsplit("__s", 1)[0]), int(i.rsplit("__s", 1)[1])))

    def test_a_replicate_differs_from_its_parent_by_exactly_the_seed(self):
        for r in replicates():
            parent_overrides = dict(r.spec.overrides)
            added = set(r.overrides) - set(parent_overrides)
            changed = {k for k in parent_overrides
                       if r.overrides.get(k) != parent_overrides[k]}
            assert added == {"seed"}, f"{r.replicate_id}: added {added}"
            assert not changed, f"{r.replicate_id}: changed {changed}"
            assert r.overrides["seed"] == r.seed

    def test_groups_and_recipe_come_from_the_frozen_parent(self):
        for r in replicates():
            assert r.spec.groups == parent_specs()[r.parent].groups
            assert r.spec.method == parent_specs()[r.parent].method

    def test_the_seed_actually_reaches_the_recipe(self):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        for r in to_train():
            assert ReferenceRecipe(**r.overrides).seed == r.seed

    def test_the_seed_is_on_the_frozen_validators_allowlist(self):
        from granunlearn.training.candidate_grid import validate_grid
        #: ``validate_grid`` is written for a complete grid and requires exactly
        #: one no-op baseline in it, which a replicate set does not have.  That
        #: ONE structural error is expected; the point of this test is that there
        #: is no OTHER error -- in particular no "overrides outside the
        #: swept-knob allowlist", which is the check that admits ``seed``.
        errors = validate_grid(tisr.as_candidate_specs(to_train()))
        assert errors == ["exactly one B0 (no-op) candidate is required"]

    def test_an_override_outside_the_allowlist_would_be_caught(self):
        from granunlearn.training.candidate_grid import validate_grid
        specs = tisr.as_candidate_specs(to_train())
        bad = replace(specs[0], overrides={**specs[0].overrides,
                                           "init_adapter_dir": "/tmp/x"})
        errors = validate_grid([*specs[1:], bad])
        assert any("allowlist" in e and "init_adapter_dir" in e
                   for e in errors)


# ──────────────────────────────────────────────────────────────────────
class TestTheParentsWereChosenByAStatedRuleNotByTheCriterion:
    def test_they_are_the_two_smallest_retain_same_shortfalls_in_queries(self):
        rep = stage1()
        n = rep["floor"]["b0"]["retain_same_entity"]["num_queries"]
        short = {cid: rep["candidates"][cid]["floor"]["checks"]
                 ["retain_same_entity.row_micro"]["difference"]
                 for cid in rep["candidates"] if cid != "B0"}
        two_closest = sorted(short, key=lambda c: -short[c])[:2]
        assert set(two_closest) == set(PARENTS)
        #: Expressed in queries, because a rate difference on its own does not
        #: say whether one seed could decide it.
        assert [round(short[p] * n, 2) for p in PARENTS] == [-2.98, -4.0]

    def test_they_are_not_the_two_best_on_d_g(self):
        rep = stage1()
        best_on_criterion = sorted(
            (c for c in rep["candidates"] if c != "B0"),
            key=lambda c: rep["candidates"][c]["distance_to_mg"])[:2]
        #: Documented in the freeze as `not_selected_on`: picking the two best
        #: on D_G would be selecting on the outcome being revisited.
        assert set(best_on_criterion) != set(PARENTS)
        assert not (set(best_on_criterion) & set(PARENTS))

    def test_the_freeze_records_the_rule_and_recomputes_the_shortfalls(self):
        fz, rep = seed_freeze(), stage1()
        parents = fz["parents"]
        assert parents["ids"] == list(PARENTS)
        assert "QUERIES" in parents["selection_rule"]
        assert parents["not_selected_on"].startswith("D_G")
        for p in PARENTS:
            recorded = parents["stage1_shortfalls"][p]
            live = rep["candidates"][p]["floor"]["checks"]
            for family, estimand in FLOOR_NUMBERS:
                key = f"{family}.{estimand}"
                assert recorded[key]["difference"] == live[key]["difference"]
                n = recorded[key]["num_queries"]
                assert recorded[key]["difference_in_queries"] == round(
                    live[key]["difference"] * n, 2)

    def test_both_parents_were_disqualified_in_stage1(self):
        rep = stage1()
        for p in PARENTS:
            assert p in rep["disqualified_by_the_floor"]
            assert rep["candidates"][p]["eligible"] is False


# ──────────────────────────────────────────────────────────────────────
class TestThePrimaryRuleIsTheMeanWithNoMargin:
    def test_aggregate_on_a_known_list(self):
        a = aggregate([0.50, 0.52, 0.54, 0.56])
        assert a["k"] == 4 and a["min"] == 0.50 and a["max"] == 0.56
        assert a["mean"] == pytest.approx(0.53)
        assert a["sd"] == pytest.approx(0.025819888974716113)

    def test_aggregate_refuses_an_empty_replicate_set(self):
        with pytest.raises(ValueError):
            aggregate([])

    def test_the_mean_is_not_rounded_to_the_inputs_four_decimals(self):
        #: 0.5294 and 0.5295 average to 0.52945, which is not representable at
        #: four decimals.  Rounding it would let a rounding step decide a
        #: comparison the inputs did not decide.
        assert aggregate([0.5294, 0.5295])["mean"] == pytest.approx(0.52945)
        assert round(aggregate([0.5294, 0.5295])["mean"], 4) != 0.52945

    def test_mean_candidate_builds_the_shape_floor_check_expects(self):
        vals = {(f, e): [0.60, 0.62] for f, e in FLOOR_NUMBERS}
        got = mean_candidate(vals)
        assert set(got) == {"retain_same_entity", "retain_other_entity"}
        for family in got:
            assert set(got[family]) == {"row_micro", "entity_macro"}
            assert got[family]["row_micro"] == pytest.approx(0.61)

    def test_the_rule_is_the_mean_not_every_seed(self):
        #: Three of four replicates below the anchor, the mean above it: the
        #: frozen rule PASSES, because the rule is about the method's expected
        #: retention and not about its unluckiest draw.
        anchor = {"retain_same_entity": {"row_micro": 0.5000,
                                         "entity_macro": 0.5000},
                  "retain_other_entity": {"row_micro": 0.5000,
                                          "entity_macro": 0.5000}}
        vals = {(f, e): [0.4900, 0.4900, 0.4900, 0.5500] for f, e in
                FLOOR_NUMBERS}
        assert mean_candidate(vals)["retain_same_entity"]["row_micro"] \
            == pytest.approx(0.505)
        assert rs.floor_check(mean_candidate(vals), anchor)["eligible"] is True
        assert sum(1 for v in vals[FLOOR_NUMBERS[0]] if v >= 0.5) == 1

    def test_the_rule_is_not_the_best_seed(self):
        #: The mirror image: one replicate above the anchor, the mean below it.
        #: A "best of k" rule would pass this; the frozen rule must not.
        anchor = {"retain_same_entity": {"row_micro": 0.5000,
                                         "entity_macro": 0.5000},
                  "retain_other_entity": {"row_micro": 0.5000,
                                          "entity_macro": 0.5000}}
        vals = {(f, e): [0.4000, 0.4000, 0.4000, 0.6000] for f, e in
                FLOOR_NUMBERS}
        assert max(vals[FLOOR_NUMBERS[0]]) >= 0.5
        assert rs.floor_check(mean_candidate(vals), anchor)["eligible"] is False

    def test_the_freeze_pins_the_epsilon_the_rule_and_the_absence_of_a_margin(
            self):
        rule = seed_freeze()["primary_rule"]
        assert rule["epsilon"] == rs.FLOOR_EPSILON == 1e-9
        assert rule["epsilon_is_the_stage1_epsilon"] is True
        assert "floor_check" in rule["scored_by"]
        assert "not reimplemented" in rule["scored_by"]
        assert rule["no_margin"] is True and rule["margin_required"] == 0.0
        assert rule["mean_is_compared_unrounded"] is True
        assert rule["rejected_alternative"]["value"] == 0.02
        assert "not reintroduced" in rule["rejected_alternative"]["status"]

    def test_the_anchor_is_the_filed_stage1_value_not_a_restatement(self):
        fz, rep = seed_freeze(), stage1()
        assert fz["anchor"]["source"].endswith("mllmu_iter12_retention_"
                                               "selection.json")
        for family, estimand in FLOOR_NUMBERS:
            assert fz["anchor"]["values"][f"{family}.{estimand}"] == \
                rep["floor"]["b0"][family][estimand]


# ──────────────────────────────────────────────────────────────────────
class TestTheRangeClassificationIsNonParametric:
    def test_every_replicate_below(self):
        assert range_classification([0.40, 0.45, 0.49], 0.50,
                                    1e-9) == "every_replicate_below"

    def test_every_replicate_at_or_above(self):
        assert range_classification([0.50, 0.55, 0.60], 0.50,
                                    1e-9) == "every_replicate_at_or_above"

    def test_straddles(self):
        assert range_classification([0.45, 0.55], 0.50,
                                    1e-9) == "straddles"

    def test_the_epsilon_rescues_only_an_exact_tie(self):
        assert range_classification([0.50], 0.50,
                                    1e-9) == "every_replicate_at_or_above"
        #: One resolvable step below -- 1/408 of a query -- is not a tie.
        assert range_classification([0.50 - 1 / 408], 0.50,
                                    1e-9) == "every_replicate_below"

    def test_the_freeze_says_why_not_an_interval(self):
        fz = seed_freeze()["reported_but_not_decision_bearing"]
        assert "k=4" in fz["why_not_an_interval"]
        assert set(fz["range_values"]) == {"every_replicate_below", "straddles",
                                           "every_replicate_at_or_above"}
        assert "the floor decides eligibility" in fz["distance_to_mg"]


# ──────────────────────────────────────────────────────────────────────
class TestTheFreezeAndTheResultDoNotShareAPath:
    def test_the_two_report_paths_differ(self):
        #: Regression: the freeze first wrote itself to the analyzer's output
        #: path, so the analyzer would have overwritten the freeze with the
        #: result it was frozen to produce.
        assert FREEZE_REPORT != OUT_REPORT
        assert FREEZE_REPORT.endswith("_freeze.json")

    def test_the_committed_freeze_lives_at_the_freeze_path(self):
        assert (REPO_ROOT / FREEZE_REPORT).exists()
        assert json.loads(
            (REPO_ROOT / FREEZE_REPORT).read_text())["stage"].startswith("1b")

    def test_the_result_path_is_still_free(self):
        #: It must be: nothing has been scored yet, and a result file present
        #: before the replicates exist would be a report about nothing.
        assert not (REPO_ROOT / OUT_REPORT).exists()

    def test_the_studies_do_not_share_a_predictions_directory(self):
        assert aisr.EXPERIMENT_ID != "mllmu_iter12_stage1"
        fz = seed_freeze()["generation_contract"]
        assert fz["this_study_uses"]["experiment_id"] == aisr.EXPERIMENT_ID
        assert fz["this_study_uses"]["predictions_dir"].endswith(
            "predictions_iter12_seeds")
        assert "REGENERATE OVER IT" in fz["why_a_separate_directory"]

    def test_replicate_adapters_cannot_overwrite_a_stage1_adapter(self):
        assert SEED_CKPT_ROOT != "data/checkpoints/mllmu_iter12_unlearn"
        for r in replicates():
            assert "__s" in r.replicate_id


# ──────────────────────────────────────────────────────────────────────
class TestTheFreezeIsBoundAndRefusesToPostdateItsReplicates:
    def test_trained_replicates_sees_a_fake_adapter(self, tmp_path):
        rid = replicate_id(PARENTS[0], NEW_SEEDS[0])
        adir = tmp_path / SEED_CKPT_ROOT / rid / "adapters"
        adir.mkdir(parents=True)
        (adir / "adapter_model.safetensors").write_bytes(b"x")
        assert fisr.trained_replicates(tmp_path) == [rid]

    def test_trained_replicates_is_empty_now(self):
        assert fisr.trained_replicates(REPO_ROOT) == []

    def test_the_refusal_is_evaluated_before_anything_is_written(
            self, monkeypatch, capsys):
        before = sha(REPO_ROOT / FREEZE_REPORT)
        monkeypatch.setattr(fisr, "trained_replicates",
                            lambda root: ["B4_x__s43"])
        monkeypatch.setattr(sys, "argv",
                            ["freeze", "--refreeze", "--reason", "try"])
        with pytest.raises(SystemExit) as exc:
            fisr.main()
        assert "REFUSING to freeze" in str(exc.value)
        assert sha(REPO_ROOT / FREEZE_REPORT) == before, \
            "--refreeze reached past the post-training refusal"

    def test_a_bare_rewrite_is_refused(self):
        before = sha(REPO_ROOT / FREEZE_REPORT)
        p = subprocess.run(
            [PY, "scripts/freeze_iter12_seed_replication.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 1
        assert "REFUSING to overwrite" in p.stderr
        assert sha(REPO_ROOT / FREEZE_REPORT) == before

    def test_an_amendment_without_a_reason_is_refused(self):
        before = sha(REPO_ROOT / FREEZE_REPORT)
        p = subprocess.run(
            [PY, "scripts/freeze_iter12_seed_replication.py", "--refreeze"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 1
        assert "--refreeze requires --reason" in p.stderr
        assert sha(REPO_ROOT / FREEZE_REPORT) == before

    def test_the_committed_freeze_verifies(self):
        assert fisr.verify_freeze(REPO_ROOT) == []

    def test_every_amendment_asserts_no_replicate_had_been_trained(self):
        for a in seed_freeze()["amendments"]:
            assert a["no_replicate_had_been_trained"] is True
            assert a["reason"] and len(a["supersedes_sha256"]) == 64

    def test_amendments_record_which_fields_changed(self):
        #: Both amendments so far re-bound protocol-path hashes and moved no
        #: criterion field.  That is the point of recording the list.
        for a in seed_freeze()["amendments"]:
            assert all(f.startswith("hashes.protocol_paths.")
                       for f in a["fields_that_changed"]), \
                f"an amendment moved a non-hash field: {a['fields_that_changed']}"

    def test_a_drifted_criterion_would_be_reported(self, monkeypatch):
        monkeypatch.setattr(rs, "FLOOR_EPSILON", 0.02)
        reasons = fisr.verify_freeze(REPO_ROOT)
        assert any("primary_rule.epsilon" in r for r in reasons)


# ──────────────────────────────────────────────────────────────────────
class TestTheMeasurementContractIsInheritedAndTheIdentityDiffers:
    def test_the_keys_that_must_match_are_the_measurement(self):
        stage1_contract = json.loads(
            (REPO_ROOT / fip.OUT_REPORT).read_text())["generation_contract"]
        fz = seed_freeze()["generation_contract"]
        assert fz["must_match"] == {k: stage1_contract[k]
                                    for k in aisr.MEASUREMENT_CONTRACT_KEYS}
        assert set(fz["must_match"]) == set(aisr.MEASUREMENT_CONTRACT_KEYS)

    def test_the_keys_that_must_differ_are_the_study_identity(self):
        fz = seed_freeze()["generation_contract"]
        assert set(fz["must_differ"]) == set(aisr.STUDY_IDENTITY_KEYS)
        for k in aisr.STUDY_IDENTITY_KEYS:
            assert fz["must_differ"][k] != fz["this_study_uses"][k]

    def _args(self, **over):
        class A:
            pass
        a = A()
        a.batch_size, a.image_batch_size, a.max_new_tokens = 8, 1, 96
        a.model_id = "Qwen/Qwen3.5-9B"
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def test_enforce_contract_accepts_the_inherited_measurement(self):
        pdir = REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12_seeds"
        cfg = aisr.enforce_contract(self._args(), REPO_ROOT, pdir)
        assert cfg["image_batch_size"] == 1 and cfg["do_sample"] is False

    def test_enforce_contract_refuses_a_different_image_batch(self):
        #: The image batch is the one thing already MEASURED to move prediction
        #: correctness, so drifting it would change what is being replicated.
        pdir = REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12_seeds"
        with pytest.raises(SystemExit) as exc:
            aisr.enforce_contract(self._args(image_batch_size=8),
                                  REPO_ROOT, pdir)
        assert "image_batch_size" in str(exc.value)

    def test_enforce_contract_refuses_stage1s_own_directory(self):
        pdir = REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12"
        with pytest.raises(SystemExit) as exc:
            aisr.enforce_contract(self._args(), REPO_ROOT, pdir)
        assert "must DIFFER" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
class TestTheDeterminismControlIsSpecifiedBeforeItIsRun:
    def test_nothing_moves_is_an_empty_dict(self):
        same = {f: {"row_micro": 0.5, "entity_macro": 0.5}
                for f, _ in FLOOR_NUMBERS}
        assert aisr.compare_floor_numbers(same, same) == {}

    def test_a_moved_number_is_named_with_both_values(self):
        a = {f: {"row_micro": 0.5294, "entity_macro": 0.5256}
             for f, _ in FLOOR_NUMBERS}
        b = json.loads(json.dumps(a))
        b["retain_same_entity"]["row_micro"] = 0.5270
        moved = aisr.compare_floor_numbers(a, b)
        assert list(moved) == ["retain_same_entity.row_micro"]
        assert moved["retain_same_entity.row_micro"] == {
            "stage1": 0.5294, "control": 0.5270}

    def test_the_scored_outcomes_are_compared_first(self):
        assert aisr.CONTROL_FIELDS[0] == "is_correct_branch"
        assert "raw_output" in aisr.CONTROL_FIELDS

    def test_the_freeze_says_the_control_runs_before_training(self):
        ctrl = seed_freeze()["generation_determinism_control"]
        assert ctrl["runs_before"] == "any replicate verdict is reported"
        assert ctrl["stops_the_study_if"] == \
            "any of the four floor numbers moves"

    def test_the_already_measured_batch_comparison_is_recorded(self):
        #: Measured for free from two B0 parquets that already existed, before
        #: this protocol was written: identical adapter bytes, image_batch_size
        #: 8 and 1.
        known = seed_freeze()["generation_determinism_control"][
            "already_measured_from_two_existing_parcets"]
        assert known["correctness_flips"] == 15 and known["rows"] == 4518
        assert known["flips_inside_either_probe_subset"] == 0
        assert sum(known["flips_by_family"].values()) == 15
        assert known["why_zero"].startswith("the floor's 408 + 76 probe queries")
        assert "100% text_to_text" in known["why_zero"]

    def test_it_records_what_that_comparison_cannot_show(self):
        known = seed_freeze()["generation_determinism_control"][
            "already_measured_from_two_existing_parcets"]
        assert "FIXED contract" in known["what_it_cannot_show"]

    def test_d_g_noise_is_recorded_against_the_frozen_six_decimals(self):
        known = seed_freeze()["generation_determinism_control"][
            "already_measured_from_two_existing_parcets"]
        assert known["d_g_components_that_moved"] == {"filr": -0.0004,
                                                     "wrong": 0.0008}
        assert "sixth decimal is not physically meaningful" in \
            known["consequence_for_d_g"]


# ──────────────────────────────────────────────────────────────────────
class TestTheLanePlanIsBalancedAndOrderIndependent:
    @pytest.fixture()
    def sizes(self):
        groups = REPO_ROOT / "data/mllmu_hier_pilot100/unlearning_iter12"
        return tisr.group_sizes(groups)

    def test_costs_come_from_the_committed_fit_half_groups(self, sizes):
        assert sizes == {"fine_target": 90, "retain": 183, "target_level": 90}
        for r in to_train():
            epochs = int(r.overrides["num_epochs"])
            assert tisr.replicate_cost(r, sizes) == epochs * 363

    def test_three_lanes_split_evenly(self, sizes):
        lanes = tisr.plan_lanes(to_train(), sizes, 3)
        loads = [sum(tisr.replicate_cost(r, sizes) for r in lane)
                 for lane in lanes]
        assert len(loads) == 3 and len(set(loads)) == 1
        #: Each lane pairs one ep5 with one ep3, which is the balance LPT finds.
        assert all(len(lane) == 2 for lane in lanes)

    def test_the_plan_does_not_depend_on_input_order(self, sizes):
        import random
        base = [[r.replicate_id for r in lane]
                for lane in tisr.plan_lanes(to_train(), sizes, 3)]
        for _ in range(5):
            shuffled = to_train()
            random.Random(_).shuffle(shuffled)
            got = [[r.replicate_id for r in lane]
                   for lane in tisr.plan_lanes(shuffled, sizes, 3)]
            assert sorted(map(sorted, got)) == sorted(map(sorted, base))

    def test_no_empty_lane_is_emitted(self, sizes):
        assert all(tisr.plan_lanes(to_train(), sizes, n)
                   for n in (1, 2, 3, 6, 12))
        assert all(lane for lane in tisr.plan_lanes(to_train(), sizes, 12))

    def test_one_lane_is_refused_at_zero(self, sizes):
        with pytest.raises(ValueError):
            tisr.plan_lanes(to_train(), sizes, 0)

    def test_every_replicate_is_planned_exactly_once(self, sizes):
        for n in (1, 2, 3, 4):
            flat = [r.replicate_id
                    for lane in tisr.plan_lanes(to_train(), sizes, n)
                    for r in lane]
            assert sorted(flat) == sorted(r.replicate_id for r in to_train())


# ──────────────────────────────────────────────────────────────────────
class TestTheStudyTouchesNothingFrozen:
    def test_no_sealed_path_was_edited_for_this_study(self):
        conf = json.loads((REPO_ROOT / "data/reports/"
                           "mllmu_pilot100_confirmation_freeze.json")
                          .read_text())
        sealed = (set(conf["code"]["fingerprinted_modules"])
                  | set(conf["code"]["analysis_scripts_sha256"])
                  #: paired_ci is sealed through the primary test, and is in
                  #: neither of the two dicts above.
                  | {conf["primary_test"]["implementation"]["module"]})
        assert len(sealed) == 18
        mine = set(fisr.PROTOCOL_PATHS)
        assert not (mine & sealed), sorted(mine & sealed)

    def test_the_stage1_freeze_still_verifies_unchanged(self):
        #: This study adds files and imports frozen ones; it edits none, so the
        #: Stage-1 freeze -- which now refuses to be rewritten at all -- must
        #: still match.
        assert fip.verify_freeze(REPO_ROOT) == []

    def test_every_declared_dependency_matches_the_bytes_on_disk(self):
        hashes = seed_freeze()["hashes"]["depends_on_unchanged"]
        assert set(hashes) == set(fisr.DEPENDS_ON_UNCHANGED)
        for rel, want in hashes.items():
            assert sha(REPO_ROOT / rel) == want, rel

    def test_the_fit_half_groups_are_untouched_by_replication(self):
        #: Replication changes the seed, never the data.  If a replicate had
        #: been built against a different retain group the probe half could
        #: have been rehearsed, which is the one thing the partition exists to
        #: prevent.
        fz1 = json.loads((REPO_ROOT / fip.OUT_REPORT).read_text())
        fz1b = seed_freeze()
        for rel in ("data/mllmu_hier_pilot100/unlearning_iter12/"
                    "retain.jsonl",
                    "data/mllmu_hier_pilot100/unlearning_iter12/"
                    "fine_target.jsonl",
                    "data/mllmu_hier_pilot100/unlearning_iter12/"
                    "target_level.jsonl"):
            stage1_hash = fz1["hashes"]["data_paths"][rel]
            assert fz1b["hashes"]["data_paths"][rel] == stage1_hash, rel
            assert sha(REPO_ROOT / rel) == stage1_hash, rel

    def test_the_probe_partition_is_read_not_recomputed(self):
        probe = seed_freeze()["probe_partition"]
        assert probe["probe_entities"] == 35
        assert probe["probe_associations"] == 204
        assert probe["replayed_by_any_candidate"] is False

    def test_the_confirmation_split_stays_out_of_bounds(self):
        assert aisr.assert_no_forbidden_evidence(
            [REPO_ROOT / "data/mllmu_hier_pilot100"], REPO_ROOT) is None
        for bad in ("data/mllmu_hier_confirm100",
                    "data/mllmu_hier_confirm100/predictions",
                    "data/reports/mllmu_confirm100_final_analysis.json"):
            with pytest.raises(SystemExit):
                aisr.assert_no_forbidden_evidence([REPO_ROOT / bad], REPO_ROOT)

    def test_a_traversal_into_the_confirmation_is_also_refused(self):
        sneaky = REPO_ROOT / "data/mllmu_hier_pilot100/../mllmu_hier_confirm100"
        with pytest.raises(SystemExit):
            aisr.assert_no_forbidden_evidence([sneaky], REPO_ROOT)

    def test_the_report_names_what_it_never_reads(self):
        assert seed_freeze()["evidence_base"]["never_read"] == \
            list(rs.FORBIDDEN_EVIDENCE)


# ──────────────────────────────────────────────────────────────────────
class TestTheLimitationsAreFiledNotAbsorbed:
    def test_replication_does_not_increase_the_query_count(self):
        lims = " ".join(seed_freeze()["limitations"])
        assert "does not increase the query count" in lims
        assert "1/76 = 0.0132" in lims

    def test_k_is_stated_and_no_interval_is_claimed(self):
        lims = " ".join(seed_freeze()["limitations"])
        assert "k=4" in lims and "licenses no interval claim" in lims

    def test_only_two_of_eight_candidates_are_covered(self):
        lims = " ".join(seed_freeze()["limitations"])
        assert "Two parents only" in lims

    def test_it_is_still_exploratory(self):
        fz = seed_freeze()
        assert fz["exploratory"] is True
        assert "no error rate is controlled" in " ".join(fz["limitations"])

    def test_the_taxonomic_gap_is_carried_forward(self):
        assert "1 of the 6 taxonomic" in " ".join(seed_freeze()["limitations"])

    def test_the_cost_estimate_comes_from_stage1_measurements(self):
        cost = seed_freeze()["replicate_set"]["cost_measured_from_stage1"]
        assert cost["trainings"] == 6
        assert cost["generations"] == 7
        assert cost["plus_one_for"] == "the determinism control"
        assert cost["training_seconds_per_candidate_ep5"] == 1609.2
        assert cost["generation_minutes_per_state_median"] == 40.2
