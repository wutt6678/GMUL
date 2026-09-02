"""Unit tests for the Iteration-11 pilot-100 TRAIN/EVALUATE pipeline.

Covers the machinery that turns the frozen pilot-100 dataset into
reference states, a wide B0-B3 candidate grid, a train+val-only
selection, and a one-shot test evaluation with paired CIs:

1. the candidate grid (frozen smoke ids, pilot-100 wide grid, validator
   negatives — a 0-weight group or an unswept override is a defect);
2. the image-resolution contract (``max_pixels`` is a silent no-op in
   Qwen3VLProcessor, so the budget must go through ``image_size_kwargs``);
3. split-scoped prediction filenames (a train+val artifact can never be
   mistaken for a full one);
4. candidate selection by id / method letter;
5. the frozen pilot-100 state + unlearning knowledge datasets;
6. the B0 == MF no-op invariant used by the final evaluation;
7. GPU lane planning (cost model, LPT balance, no-op pinning).

Everything here is CPU-only and reads committed artifacts only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from granunlearn.imaging import MIN_IMAGE_AREA, image_size_kwargs
from granunlearn.training.candidate_grid import (
    METHODS,
    CandidateSpec,
    GroupUse,
    dataset_dir_for_tag,
    grid_for_tag,
    pilot100_grid,
    smoke_grid,
    validate_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PILOT_DIR = REPO_ROOT / "data" / "mllmu_hier_pilot100"


# ── 1. candidate grid ─────────────────────────────────────────────

class TestCandidateGrid:
    def test_smoke_grid_ids_are_frozen(self):
        """The Iteration 9 ids are referenced by the committed selection
        report, so the smoke grid must never be renamed."""
        assert [c.candidate_id for c in smoke_grid()] == [
            "B0", "B1_lr2e-05", "B1_lr0.0001", "B2_lr1e-04",
            "B3_lam1.0", "B3_lam0.5"]
        assert validate_grid(smoke_grid()) == []

    def test_pilot100_grid_is_wide_and_valid(self):
        grid = pilot100_grid()
        assert validate_grid(grid) == []
        assert 15 <= len(grid) <= 19
        ids = [c.candidate_id for c in grid]
        assert len(ids) == len(set(ids))
        assert sum(1 for c in grid if c.noop) == 1
        # every method of the ported B0-B3 family is swept
        assert {c.method for c in grid} == set(METHODS)
        for m in ("B1", "B2", "B3"):
            assert sum(1 for c in grid if c.method == m) >= 2, m

    def test_pilot100_grid_sweeps_declared_knobs_only(self):
        allowed = {"learning_rate", "num_epochs"}
        for c in pilot100_grid():
            assert set(c.overrides) <= allowed, c.candidate_id
            assert c.overrides.get("num_epochs", 10) >= 1

    def test_b3_always_carries_all_three_groups(self):
        for c in pilot100_grid():
            if c.method != "B3":
                continue
            names = {g.name for g in c.groups}
            assert names == {"fine_target", "target_level", "retain"}
            modes = {g.name: g.mode for g in c.groups}
            assert modes["fine_target"] == "gd"
            assert modes["target_level"] == "sft"
            assert modes["retain"] == "sft"

    def test_b2r_is_b2_plus_the_retain_group(self):
        """Iteration 10 finding: the explicit retain group is what closes
        B2's retain gap, so B2R must be B2 + retain and nothing else."""
        b2r = [c for c in pilot100_grid() if c.method == "B2R"]
        assert b2r
        for c in b2r:
            assert [(g.name, g.mode) for g in c.groups] == [
                ("target_level", "sft"), ("retain", "sft")]

    def test_b1_is_pure_gradient_ascent_on_fine_target(self):
        for c in pilot100_grid():
            if c.method != "B1":
                continue
            assert [(g.name, g.mode) for g in c.groups] == [
                ("fine_target", "gd")]

    def test_validator_rejects_duplicate_ids(self):
        g = [CandidateSpec("X", "B1", (GroupUse("fine_target", "gd"),)),
             CandidateSpec("X", "B1", (GroupUse("fine_target", "gd"),)),
             CandidateSpec("B0", "B0", noop=True)]
        assert any("duplicate" in e for e in validate_grid(g))

    def test_validator_rejects_zero_weight_group(self):
        """A 0-weight group still consumes the interleaved stream, so it
        silently changes the schedule — it must be rejected, not used as
        an 'ablation'."""
        g = [CandidateSpec("B0", "B0", noop=True),
             CandidateSpec("B3_lam0", "B3",
                           (GroupUse("fine_target", "gd", 0.0),
                            GroupUse("target_level", "sft", 1.0)))]
        assert any("weight must be > 0" in e for e in validate_grid(g))

    def test_validator_rejects_unswept_override(self):
        g = [CandidateSpec("B0", "B0", noop=True),
             CandidateSpec("B1_x", "B1",
                           (GroupUse("fine_target", "gd"),),
                           {"max_image_pixels": 1})]
        assert any("allowlist" in e for e in validate_grid(g))

    def test_validator_requires_exactly_one_noop(self):
        assert any("exactly one B0" in e
                   for e in validate_grid(
                       [CandidateSpec("B1_a", "B1",
                                      (GroupUse("fine_target", "gd"),))]))

    def test_noop_takes_no_groups_or_overrides(self):
        g = [CandidateSpec("B0", "B0",
                           groups=(GroupUse("retain", "sft"),), noop=True)]
        assert any("no-op" in e for e in validate_grid(g))

    def test_grid_for_tag_and_dataset_dir(self):
        assert grid_for_tag("smoke") == smoke_grid()
        assert grid_for_tag("pilot100") == pilot100_grid()
        with pytest.raises(ValueError):
            grid_for_tag("nope")
        assert dataset_dir_for_tag("pilot100") == "data/mllmu_hier_pilot100"

    def test_describe_is_json_serializable(self):
        for c in pilot100_grid():
            json.dumps(c.describe())


# ── 2. image-resolution contract ──────────────────────────────────

class TestImageSizeContract:
    def test_kwargs_use_the_size_interface(self):
        """Qwen3VLProcessor swallows an unknown ``max_pixels`` kwarg, so
        the budget has to be expressed as area bounds under ``size``."""
        kw = image_size_kwargs(384 * 384)
        assert set(kw) == {"size"}
        assert kw["size"]["longest_edge"] == 384 * 384
        assert kw["size"]["shortest_edge"] == MIN_IMAGE_AREA

    def test_floor_never_exceeds_the_cap(self):
        kw = image_size_kwargs(MIN_IMAGE_AREA)
        assert kw["size"]["shortest_edge"] <= kw["size"]["longest_edge"]

    def test_non_positive_budget_rejected(self):
        with pytest.raises(ValueError):
            image_size_kwargs(0)
        with pytest.raises(ValueError):
            image_size_kwargs(-1)

    def test_recipe_and_generator_share_the_contract(self):
        """Both the trainer and the evaluator must route the recipe's
        pixel budget through the same helper — identical multimodal
        formatting across states is the counterfactual's requirement."""
        trainer = (REPO_ROOT / "src" / "granunlearn" / "training"
                   / "reference_trainer.py").read_text()
        evaluator = (REPO_ROOT / "src" / "granunlearn" / "evaluation"
                     / "reference_eval.py").read_text()
        for src in (trainer, evaluator):
            assert "image_size_kwargs" in src
            assert '"max_pixels"' not in src


# ── 3. split-scoped prediction filenames ──────────────────────────

class TestPredictionFilename:
    def setup_method(self):
        from select_unlearning_checkpoints import prediction_filename
        self.fn = prediction_filename

    def test_all_splits_keeps_the_legacy_name(self):
        assert self.fn("MF", ("train", "val", "test")) == \
            "predictions_MF.parquet"

    def test_trainval_scope_is_named(self):
        assert self.fn("B1_lr2e-05_ep10", ("train", "val")) == \
            "predictions_tv_B1_lr2e-05_ep10.parquet"

    def test_test_scope_is_named(self):
        assert self.fn("B2", ("test",)) == "predictions_test_B2.parquet"

    def test_scope_is_part_of_every_name(self):
        """No two different scopes may ever produce the same filename."""
        scopes = (("train", "val", "test"), ("train", "val"), ("test",),
                  ("train",), ("val", "test"))
        names = [self.fn("X", s) for s in scopes]
        assert len(set(names)) == len(scopes), names
        assert self.fn("X", ("val", "test")) == "predictions_test-val_X.parquet"

    def test_empty_scope_rejected(self):
        with pytest.raises(ValueError):
            self.fn("X", ())


# ── 4. candidate selection by id / method ─────────────────────────

class TestSelectCandidates:
    def setup_method(self):
        from train_unlearning_baselines import select_candidates
        self.sel = select_candidates
        self.grid = pilot100_grid()

    def test_default_keeps_the_whole_grid(self):
        assert self.sel(self.grid, None) == self.grid
        assert self.sel(self.grid, "all") == self.grid

    def test_by_candidate_id(self):
        picked = self.sel(self.grid, "B0,B1_lr1e-05_ep10")
        assert [c.candidate_id for c in picked] == \
            ["B0", "B1_lr1e-05_ep10"]

    def test_by_method_letter_expands_in_grid_order(self):
        picked = self.sel(self.grid, "B2R")
        assert [c.method for c in picked] == ["B2R", "B2R"]

    def test_mixed_tokens_and_duplicates(self):
        picked = self.sel(self.grid, "B1,B1_lr1e-05_ep10")
        assert len({c.candidate_id for c in picked}) == len(picked)
        assert sum(1 for c in picked if c.method == "B1") == 4

    def test_unknown_token_is_fatal(self):
        with pytest.raises(SystemExit):
            self.sel(self.grid, "B9_nonsense")


# ── 5. frozen pilot-100 knowledge datasets ────────────────────────

class TestFrozenPilotKnowledgeDatasets:
    def test_state_manifest_counts(self):
        m = json.loads(
            (PILOT_DIR / "training" / "state_datasets_manifest.json"
             ).read_text())
        assert m["states"]["MF"]["num_examples"] == 477
        assert m["states"]["MG"]["num_examples"] == 477
        assert m["states"]["MN"]["num_examples"] == 387
        assert m["states"]["MF"]["num_target"] == 90
        assert m["states"]["MN"]["num_target"] == 0
        # MF trains the fine level, MG the retained level, both keep
        # every retained association at the fine level
        assert m["states"]["MF"]["target_level_distribution"] == {"0": 90}
        mg = m["states"]["MG"]["target_level_distribution"]
        assert sum(int(v) for v in mg.values()) == 90
        assert "0" not in mg

    def test_all_pilot_examples_are_multimodal(self):
        """Every pilot-100 association carries an image, so the state
        datasets must be entirely image_text — the multimodal formatting
        path is exercised by every training example."""
        m = json.loads(
            (PILOT_DIR / "training" / "state_datasets_manifest.json"
             ).read_text())
        for state, info in m["states"].items():
            assert info["num_image_text"] == info["num_examples"], state

    def test_one_shared_template_across_states(self):
        m = json.loads(
            (PILOT_DIR / "training" / "state_datasets_manifest.json"
             ).read_text())
        assert m["training_template"]["prompt"] == \
            "What is {name}'s {attr}?"
        assert "never" in m["training_template"]["note"].lower()

    def test_unlearning_groups_match_the_partition(self):
        m = json.loads(
            (PILOT_DIR / "unlearning" / "unlearning_groups_manifest.json"
             ).read_text())
        part = json.loads(
            (REPO_ROOT / "data" / "reports"
             / "mllmu_pilot100_target_retain.json").read_text())
        assert m["groups"]["fine_target"]["num_examples"] == \
            len(part["target_association_ids"]) == 90
        assert m["groups"]["target_level"]["num_examples"] == 90
        assert m["groups"]["retain"]["num_examples"] == \
            len(part["retain_association_ids"]) == 387

    def test_state_and_group_jsonls_are_committed_and_parse(self):
        from granunlearn.training.state_datasets import TrainingExample
        for rel in ("training/MF.jsonl", "training/MG.jsonl",
                    "training/MN.jsonl", "unlearning/fine_target.jsonl",
                    "unlearning/target_level.jsonl",
                    "unlearning/retain.jsonl"):
            p = PILOT_DIR / rel
            assert p.exists(), rel
            rows = [TrainingExample.model_validate(json.loads(line))
                    for line in p.read_text().splitlines() if line.strip()]
            assert rows
            # repo-relative image paths only (the loader resolves them)
            assert all(r.image_path and
                       not Path(r.image_path).is_absolute() for r in rows)


# ── 6. the B0 == MF no-op invariant ───────────────────────────────

class TestNoOpInvariant:
    def _preds(self, raws):
        from granunlearn.schema import PredictionRecord
        return [PredictionRecord(
            experiment_id="e", checkpoint_id="c", query_id=q,
            raw_output=r, parsed_answer=None, matched_canonical_id=None,
            predicted_level=None, is_correct_branch=False,
            is_finer_than_target=False, is_coarser_than_target=False,
            metadata={}) for q, r in raws.items()]

    def test_identical_states_pass(self):
        from evaluate_pilot100_final import _check_b0_equals_mf
        from granunlearn.evaluation.paired_ci import paired_metrics_report
        raws = {"q1": "Passer.", "q2": "I don't know."}
        preds = {"B0": self._preds(raws), "MF": self._preds(raws)}
        paired = paired_metrics_report(preds, [], [],
                                       reference_states=("MF",),
                                       split=None)
        check = _check_b0_equals_mf(preds, paired)
        assert check["checked"] is True
        assert check["num_raw_output_mismatches"] == 0
        assert check["passed"] is True

    def test_divergent_outputs_fail(self):
        from evaluate_pilot100_final import _check_b0_equals_mf
        preds = {"B0": self._preds({"q1": "Passer."}),
                 "MF": self._preds({"q1": "Passeriformes."})}
        check = _check_b0_equals_mf(preds, {"comparisons": {}})
        assert check["num_raw_output_mismatches"] == 1
        assert check["mismatched_query_ids"] == ["q1"]
        assert check["passed"] is False

    def test_missing_state_is_not_a_pass(self):
        from evaluate_pilot100_final import _check_b0_equals_mf
        check = _check_b0_equals_mf({"MF": self._preds({"q1": "x"})}, {})
        assert check["checked"] is False
        assert "passed" not in check or check.get("passed") is not True

    def test_reference_and_paired_levels_are_declared(self):
        import evaluate_pilot100_final as fin
        assert set(fin.REFERENCE_STATES) == {"BASE", "MF", "MG", "MN"}
        # B0 must be a paired reference so the invariant is reported as a
        # CI, not merely asserted in code
        assert "B0" in fin.PAIRED_REFERENCES
        assert fin.EXPERIMENT_ID == "mllmu_pilot100_iter11"


# ── 7. GPU lane planning ──────────────────────────────────────────

class TestLanePlanning:
    SIZES = {"fine_target": 90, "target_level": 90, "retain": 387}

    def setup_method(self):
        from plan_candidate_lanes import candidate_cost, plan_lanes
        self.cost = candidate_cost
        self.plan = plan_lanes
        self.grid = pilot100_grid()

    def test_cost_is_epochs_times_group_examples(self):
        b1 = next(c for c in self.grid if c.method == "B1")
        assert self.cost(b1, self.SIZES, 10) == 10 * 90
        b3 = next(c for c in self.grid
                  if c.method == "B3" and c.overrides["num_epochs"] == 5)
        assert self.cost(b3, self.SIZES, 10) == 5 * (90 + 90 + 387)
        b0 = next(c for c in self.grid if c.noop)
        assert self.cost(b0, self.SIZES, 10) == 0

    def test_unknown_group_is_fatal(self):
        bad = CandidateSpec("X", "B1", (GroupUse("nope", "gd"),),
                            {"num_epochs": 1})
        with pytest.raises(SystemExit):
            self.cost(bad, self.SIZES, 10)

    def test_every_candidate_planned_exactly_once(self):
        for n in (1, 2, 3, 4, 7):
            lanes = self.plan(self.grid, self.SIZES, n, 10)
            flat = [c.candidate_id for lane in lanes for c in lane]
            assert sorted(flat) == sorted(
                c.candidate_id for c in self.grid)

    def test_noop_is_pinned_to_the_first_lane(self):
        lanes = self.plan(self.grid, self.SIZES, 3, 10)
        assert lanes[0][0].noop is True
        assert not any(c.noop for lane in lanes[1:] for c in lane)

    def test_lanes_are_balanced(self):
        """LPT packing must land close to total/lanes; an unbalanced plan
        would idle a scarce shared GPU for hours."""
        lanes = self.plan(self.grid, self.SIZES, 3, 10)
        loads = [sum(self.cost(c, self.SIZES, 10) for c in lane)
                 for lane in lanes]
        total = sum(loads)
        assert max(loads) <= total / 3 * 1.15, loads
        assert max(loads) - min(loads) <= total * 0.05, loads

    def test_single_lane_takes_everything(self):
        lanes = self.plan(self.grid, self.SIZES, 1, 10)
        assert len(lanes) == 1
        assert len(lanes[0]) == len(self.grid)

    def test_invalid_lane_count(self):
        with pytest.raises(ValueError):
            self.plan(self.grid, self.SIZES, 0, 10)
