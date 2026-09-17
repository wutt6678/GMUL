"""Iteration 12 Stage 3: the route-aware B6 successor protocol.

These tests pin the protocol before training: the grid is B6 plus one moved
coefficient, the image-conditioned stream is measured rather than asserted,
the probe half stays evaluation-only, the zero-weight control really is B6,
the Stage-2 floor and D_G survive unchanged, and the freeze is verifiable in a
clone without adapters, prediction parquets, the cache tensor or torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_iter12_stage3_basis as bis3
import freeze_iter12_stage3 as fis3
import select_iter12_stage3 as sis3
import train_iter12_stage3 as tis3

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training.candidate_grid import validate_grid
from granunlearn.training.reference_trainer import ReferenceRecipe

PY = sys.executable


def calibration() -> dict:
    return json.loads((REPO_ROOT / s2g.CALIBRATION_REPORT).read_text())


def beta_star() -> float:
    return calibration()["grid_rule"]["beta_star"]


def grid() -> list[s3g.Stage3CandidateSpec]:
    return s3g.stage3_grid(beta_star())


def basis() -> dict:
    return json.loads((REPO_ROOT / s3g.BASIS_REPORT).read_text())


def control() -> dict:
    return json.loads((REPO_ROOT / s3g.CONTROL_REPORT).read_text())


def freeze() -> dict:
    return json.loads((REPO_ROOT / s3g.FREEZE_REPORT).read_text())


# ──────────────────────────────────────────────────────────────────────
class TestTheGridIsB6PlusOneCoefficient:
    def test_six_rows_four_of_which_need_a_gpu(self):
        g = grid()
        assert len(g) == 6
        assert len(s3g.trained_rows(g)) == 4
        assert [c.candidate_id for c in g][:2] == [
            "B0", "B6_beta13.642_lam0.5_lr2e-05_ep5"]

    def test_the_weight_grid_is_four_local_multipliers_plus_zero(self):
        assert s3g.IMAGE_WEIGHT_MULTIPLIERS == (0.125, 0.25, 0.5, 1.0)
        b7 = s3g.b7_rows(grid())
        assert [c.image_anchor_weight for c in b7] == [
            1.7052, 3.4105, 6.821, 13.642]
        assert [c.effective_anchor_weight for c in b7] == [
            15.3472, 17.0525, 20.463, 27.284]

    def test_zero_weight_is_exactly_the_stage2_b6_objective(self):
        parent = s3g.b6_parent(beta_star())
        control = s3g.control_row(grid())
        assert control.candidate_id == parent.candidate_id
        assert control.method == s2g.METHOD_ANCHOR
        assert control.image_anchor_weight == 0.0
        assert control.effective_anchor_weight == parent.beta
        assert s3g.same_objective_as_b6(control, parent)

    def test_b7_changes_only_the_retain_anchor_coefficient(self):
        parent = s3g.b6_parent(beta_star())
        for row in s3g.b7_rows(grid()):
            assert len(row.groups) == len(parent.groups) == 3
            for got, want in zip(row.groups, parent.groups, strict=True):
                assert got.name == want.name
                assert got.mode == want.mode
                assert got.path == want.path
                assert got.cap == want.cap
                assert got.anchor_weight == want.anchor_weight
                if got.name == "retain":
                    assert got.weight == row.effective_anchor_weight
                else:
                    assert got.weight == want.weight
            assert row.overrides == parent.overrides

    def test_the_recipe_seed_and_layout_are_b6s(self):
        for row in [*s3g.b7_rows(grid()), s3g.control_row(grid())]:
            recipe = ReferenceRecipe(**row.overrides)
            assert recipe.seed == 42
            assert recipe.learning_rate == s2g.STAGE2_LR
            assert recipe.num_epochs == s2g.STAGE2_EPOCHS
            assert recipe.per_device_batch_size == 1
            assert recipe.gradient_accumulation_steps == 8
            assert recipe.max_length == 1536
            assert recipe.max_image_pixels == 384 * 384

    def test_the_frozen_validator_filters_only_the_two_known_gaps(self):
        kept, filtered = s3g.validate_stage3_grid(grid(), beta_star())
        assert kept == []
        assert len(filtered) == 10
        assert all("unknown method" in m or "bad mode" in m
                   for m in filtered)

    def test_a_bad_override_is_still_caught_by_the_frozen_validator(self):
        rows = grid()
        bad = replace(rows[2], overrides={**rows[2].overrides,
                                          "init_adapter_dir": "/tmp/x"})
        errors = validate_grid([*rows[:2], bad, *rows[3:]])  # type: ignore[arg-type]
        assert any("allowlist" in e and "init_adapter_dir" in e
                   for e in errors)

    def test_a_wrong_control_is_not_accepted_as_b6(self):
        parent = s3g.b6_parent(beta_star())
        control = s3g.control_row(grid())
        wrong_weight = replace(
            control,
            groups=tuple(replace(g, weight=g.weight + 1)
                         if g.name == "retain" else g
                         for g in control.groups))
        assert not s3g.same_objective_as_b6(wrong_weight, parent)
        nonzero = replace(control, image_anchor_weight=0.1)
        assert not s3g.same_objective_as_b6(nonzero, parent)


# ──────────────────────────────────────────────────────────────────────
class TestTheBasisIsMeasuredAndRederives:
    def test_check_only_rederives_it_byte_for_byte(self):
        assert bis3.check_only(REPO_ROOT) == []

    def test_b6_is_the_parent_for_the_filed_reasons(self):
        out = basis()["b6_result"]
        assert out["candidate_id"] == "B6_beta13.642_lam0.5_lr2e-05_ep5"
        assert out["text_route"]["all_four_pass"] is True
        assert out["image_route"]["exactly_three_fail"] is True
        assert out["image_route"]["failed"] == [
            "image.retain_same_entity_image.entity_macro",
            "image.retain_same_entity_image.row_micro",
            "image.retain_other_entity_image.row_micro",
        ]
        assert out["fine_target_nll"] == 0.6857
        assert out["bounded_ascent_fine_target_nll_range"] == [0.7832, 2.9924]
        assert out["preserves_the_target_transformation_better_than_bounded_ascent"] is True

    def test_the_fourteen_rows_are_not_double_counted(self):
        short = basis()["b6_result"]["the_fourteen_query_shortfall"]
        assert short["same_image_rows"] == 5
        assert short["same_image_denominator"] == 408
        assert short["other_image_rows"] == 2
        assert short["other_image_denominator"] == 76
        assert short["expressed_over_trainval_both_route_denominators"] == {
            "same": "10/816", "other": "4/152", "total": 14}
        assert "double-count" in short[
            "why_the_entity_macro_failure_adds_no_rows"]

    def test_the_image_conditioned_stream_is_measured_not_asserted(self):
        block = basis()["image_anchor_examples"]
        assert block["fit_half_examples"] == 183
        assert block["image_conditioned_examples"] == 183
        assert block["not_image_conditioned"] == 0
        assert block["all_fit_half_examples_are_image_conditioned"] is True
        assert block["entities"] == 35
        assert block["associations"] == 183
        assert block["probe_half_overlap"] == {
            "associations": 0, "entities": 0, "must_be_zero": True}
        assert "not a restriction to a smaller subset" in block[
            "the_important_disclosure"]

    def test_the_existing_mf_cache_is_the_image_conditioned_cache(self):
        cache = basis()["image_anchor_examples"]["mf_cache"]
        assert cache["num_examples"] == 183
        assert cache["num_positions"] == 1489
        assert cache["cache_rows_match_the_image_conditioned_group"] is True
        assert cache["cache_is_the_image_conditioned_mf_cache"] is True
        assert cache["probe_half_excluded"] is True
        assert cache["target_groups_excluded"] == ["fine_target", "target_level"]
        assert cache["no_new_cache_is_needed"] is True

    def test_the_basis_reads_no_prediction_parquet(self):
        doc = basis()
        assert doc["reads_no_prediction_parquet"] is True
        assert not any("predictions_iter12" in p
                       for p in doc["evidence_base"]["used"])


# ──────────────────────────────────────────────────────────────────────
class TestTheZeroWeightControl:
    def test_it_holds_at_freeze_time_and_uses_the_frozen_tolerances(self):
        doc = control()
        assert doc["control"] == tis3.CONTROL_ID
        assert doc["holds"] is True
        assert doc["tolerances"]["floor_values"] == \
            s3g.CONTROL_FLOOR_TOLERANCE == 1e-9
        assert doc["tolerances"]["d_g"] == \
            s3g.CONTROL_DG_TOLERANCE == 1e-6

    def test_both_halves_are_required(self):
        doc = control()
        assert all(doc["structural"].values())
        assert all(v["within_tolerance"]
                   for v in doc["numerical"]["eight_values"].values())
        assert doc["numerical"]["d_g"]["within_tolerance"] is True
        assert doc["numerical"]["d_g"]["filed"] == 0.060314
        assert doc["numerical"]["d_g"]["recomputed"] == 0.060314

    def test_all_eight_values_recompute_with_zero_observed_difference(self):
        values = control()["numerical"]["eight_values"]
        assert set(values) == set(rt.FLOOR_NUMBER_KEYS)
        assert all(v["abs_difference"] == 0.0 for v in values.values())

    def test_the_bound_bytes_are_the_b6_and_mg_evidence(self):
        bound = control()["bound_bytes"]
        assert bound["b6_predictions"].endswith(
            "predictions_tv_B6_beta13.642_lam0.5_lr2e-05_ep5.parquet")
        assert bound["mg_predictions"].endswith("predictions_tv_MG.parquet")
        assert bound["retain_group_sha256"] == \
            "1221a45f08958a0542c0c40b8f219181d248ee0318e0c41cf403a90634276e3c"
        assert bound["cache_tensor_sha256"] == \
            "dbf8b475e01bf3e13c731f04b94109e8dc37b14d6252598d7e9b6811999e5f6a"

    @pytest.mark.skipif(
        not (REPO_ROOT / "data/mllmu_hier_pilot100/predictions_iter12_stage2" /
             "predictions_tv_B6_beta13.642_lam0.5_lr2e-05_ep5.parquet").exists(),
        reason="the gitignored B6 parquet is absent from this clone")
    def test_the_control_rederives_the_filed_report_when_bytes_are_present(self):
        assert tis3.zero_weight_control(REPO_ROOT) == control()


# ──────────────────────────────────────────────────────────────────────
class TestTheFreezePreservesTheCriterion:
    def test_check_only_passes_from_the_entry_point(self):
        p = subprocess.run(
            (PY, "scripts/freeze_iter12_stage3.py", "--check-only"),
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "reused_predictions_bound" in p.stdout
        assert "image_conditioned_cache_bound" in p.stdout

    def test_the_eight_floors_and_d_g_are_stage2s_unchanged(self):
        fz = freeze()["what_is_frozen"]
        stage2 = json.loads((REPO_ROOT / s2g.FREEZE_REPORT).read_text())[
            "what_is_frozen"]
        assert fz["floor_numbers"] == stage2["floor_numbers"] == \
            list(rt.FLOOR_NUMBER_KEYS)
        assert fz["primary_anchor"] == stage2["primary_anchor"] == "MG"
        assert fz["anchor_values"] == stage2["anchor_values"]
        assert fz["epsilon"] == stage2["epsilon"] == rs.FLOOR_EPSILON
        assert fz["margin"] == stage2["margin"] == 0.0
        assert fz["among_eligible"] == stage2["among_eligible"]
        assert fz["unchanged_from_the_stage2_freeze"] is True
        assert fz["distance_to_mg"]["tie_break"] == \
            "(distance_to_mg, candidate_id) ascending"

    def test_the_freeze_binds_the_grid_control_cache_split_and_layout(self):
        fz = freeze()
        assert fz["grid"]["num_rows"] == 6
        assert fz["grid"]["num_trained"] == 4
        assert fz["zero_weight_control"]["holds_at_freeze_time"] is True
        assert fz["image_anchor_examples"]["mf_cache"][
            "cache_is_the_image_conditioned_mf_cache"] is True
        assert fz["image_anchor_examples"]["probe_half_overlap"][
            "associations"] == 0
        assert fz["seeds_and_layout"]["training_seed"] == 42
        assert fz["seeds_and_layout"]["python_hash_seed"] == "0"
        assert fz["seeds_and_layout"]["training"][
            "gradient_accumulation_steps"] == 8

    def test_the_freeze_hashes_its_own_protocol_and_not_the_tests(self):
        hashes = freeze()["hashes"]
        assert set(hashes["protocol_paths"]) == set(fis3.PROTOCOL_PATHS)
        assert "tests/unit/test_iteration12_stage3.py" not in \
            hashes["protocol_paths"]
        assert "scripts/train_iter12_stage3.py" in hashes["protocol_paths"]
        assert "src/granunlearn/training/preservation_trainer.py" in \
            hashes["depends_on_unchanged"]
        assert s3g.BASIS_REPORT in hashes["data_paths"]
        assert s3g.CONTROL_REPORT in hashes["data_paths"]

    def test_no_stage3_adapter_existed_when_the_freeze_was_written(self):
        assert fis3.adapters_exist(REPO_ROOT) == []
        assert freeze()["refusal_policy"]["unconditional"] is True

    def test_clone_dependent_verification_is_three_valued(self):
        out = fis3.verify_clone_dependent(REPO_ROOT)
        assert set(out) == set(fis3.CLONE_DEPENDENT_FIELDS)
        for result in out.values():
            assert result["mismatch"] == []
            assert result["match"] or result["absent"]

    def test_an_edited_freeze_fails_verification(self, tmp_path):
        doc = freeze()
        doc["what_is_frozen"]["epsilon"] = 0.02
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(doc))
        reasons = fis3.verify_freeze(REPO_ROOT, path=edited)
        assert any("what_is_frozen.epsilon" in r for r in reasons)

    def test_the_confirmation_split_is_never_read(self):
        evidence = freeze()["evidence_base"]
        assert set(evidence["never_read"]) == set(rs.FORBIDDEN_EVIDENCE)
        assert "RESOLVED" in evidence["never_read_enforced_by"]
        assert basis()["evidence_base"]["probe_entities_evaluation_only"] is True

    def test_stage3_does_not_make_the_stage2_repair_history_stale(self):
        #: The Stage-2 repair's third defect is a historical disclosure scoped
        #: to the freezes that existed when it was filed.  Stage 3 is reported
        #: as later, not folded into a record it could not have described.
        import repair_iter12_stage2_selection as rep
        out = rep.the_freeze_records_no_git_provenance()
        assert s3g.FREEZE_REPORT not in out["per_freeze"]
        assert s3g.FREEZE_REPORT in out["scope"][
            "later_freezes_outside_this_record"]
        assert out["scope"]["missing_from_the_filing_scope"] == []


# ──────────────────────────────────────────────────────────────────────
class TestTheExecutorsGuardTheProtocol:
    def test_paths_are_single_owner_and_do_not_double_data(self):
        for path in (s3g.stage3_ckpt_root(REPO_ROOT),
                     s3g.stage3_predictions_dir(REPO_ROOT),
                     s3g.groups_dir(REPO_ROOT),
                     s3g.reference_cache_dir(REPO_ROOT)):
            assert "data/data" not in str(path)
        assert s3g.stage3_ckpt_root(REPO_ROOT) != \
            s2g.stage2_ckpt_root(REPO_ROOT)
        assert s3g.stage3_predictions_dir(REPO_ROOT) != \
            s2g.stage2_predictions_dir(REPO_ROOT)

    def test_reused_prediction_paths_are_b0_mg_and_the_b6_control(self):
        paths = sis3.reused_paths(REPO_ROOT)
        assert set(paths) == {
            "B0", "MG", "B6_beta13.642_lam0.5_lr2e-05_ep5"}
        assert "predictions_iter12_stage2" in str(paths[
            "B6_beta13.642_lam0.5_lr2e-05_ep5"])

    def test_both_executors_refuse_the_confirmation_split_on_resolved_paths(self):
        forbidden = REPO_ROOT / "data" / "mllmu_hier_confirm100"
        sneaky = REPO_ROOT / "data" / ".." / "data" / \
            "mllmu_hier_confirm100" / "predictions.parquet"
        for guard in (tis3.assert_no_forbidden_evidence,
                      sis3.assert_no_forbidden_evidence):
            with pytest.raises(SystemExit):
                guard([forbidden], REPO_ROOT)
            with pytest.raises(SystemExit):
                guard([sneaky], REPO_ROOT)

    def test_mg_is_not_a_stage3_candidate_or_ranking_row(self, monkeypatch):
        #: A behavioural check, not a source search: build a minimal report and
        #: require MG's absence from the two places Stage 2's defect put it.
        monkeypatch.setattr(sis3.rs, "summary_vector", lambda m: m)
        monkeypatch.setattr(sis3.rs, "distance_to_mg",
                            lambda v, mg: (0.1 if v else None, []))
        monkeypatch.setattr(
            sis3.rt, "floor_check_stratified",
            lambda cand, base, eps: {
                "eligible": True, "checks": {k: {"passes": True}
                                             for k in rt.FLOOR_NUMBER_KEYS}})
        values = {k: 1.0 for k in rt.FLOOR_NUMBER_KEYS}
        b6 = "B6_beta13.642_lam0.5_lr2e-05_ep5"
        states = {
            "B0": {"method": "B0", "spec": {}, "trainval_metrics": {},
                   "eight_numbers": values},
            b6: {"method": "B6", "spec": {}, "trainval_metrics": {"x": 1},
                 "eight_numbers": values},
        }
        fz = freeze()
        report = sis3.build_report(
            states,
            {"B0": {"path": "b0", "sha256": "0"},
             "MG": {"path": "mg", "sha256": "1"}},
            {}, REPO_ROOT, s3g.dataset_dir(REPO_ROOT),
            s3g.stage3_predictions_dir(REPO_ROOT), {}, "model", {}, fz,
            grid())
        assert "MG" not in report["candidates"]
        assert "MG" not in report["ranking_of_eligible"]
        assert report["anchor_states"]["MG"][
            "excluded_from_the_ranking"] is True

    def test_the_lane_plan_is_deterministic_and_covers_all_four_rows(self):
        class Args:
            plan_lanes = 2
            emit_sh = False
        sizes = tis3.group_sizes(REPO_ROOT)
        rows = s3g.trained_rows(grid())
        lanes = tis3.plan_lanes(rows, sizes, 2)
        assert sorted(c.candidate_id for lane in lanes for c in lane) == \
            sorted(c.candidate_id for c in rows)
        assert [sum(tis3.training_cost(c, sizes) for c in lane)
                for lane in lanes] == [3630, 3630]

    def test_the_committed_reports_are_exposed_to_git_and_ci(self):
        ignored = (REPO_ROOT / ".gitignore").read_text()
        workflow = (REPO_ROOT / ".github" / "workflows" /
                    "tests.yml").read_text()
        for rel in (s3g.BASIS_REPORT, s3g.CONTROL_REPORT, s3g.FREEZE_REPORT):
            assert f"!{rel}" in ignored
            assert f'"{rel}"' in workflow
            assert (REPO_ROOT / rel).exists()
        assert "mllmu_iter12_stage3_selection.json" not in workflow
