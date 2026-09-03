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
7. GPU lane planning (cost model, LPT balance, no-op pinning);
8. the batch-layout noise floor reported by the final evaluation;
9. recipe inheritance for the no-op arm (B0 reports MF's recipe, never
   null, and ``ReferenceRecipe`` round-trips through JSON);
10. the generation batch layout, pinned against a stubbed model — the
   layout is part of the experiment because decoding is not bit-stable
   across layouts.

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


# ── 8. batch-layout noise floor ───────────────────────────────────

class TestBatchCompositionSensitivity:
    """The same checkpoint scored under two batch layouts is NOT
    bit-identical (measured on the pilot-100 run: 122/2,259 greedy
    decodes flipped between the gate's 6,777-query pass and the final
    2,259-query pass).  The report must expose that floor rather than
    let it masquerade as a model difference."""

    def _preds(self, queries, assoc_by_id, flip=0):
        from granunlearn.schema import PredictionRecord
        out = []
        for i, q in enumerate(queries):
            a = assoc_by_id[q.association_id]
            raw = a.levels[0].value
            if i < flip:
                raw = "I don't know."
            out.append(PredictionRecord(
                experiment_id="e", checkpoint_id="MF",
                query_id=q.query_id, raw_output=raw, parsed_answer=None,
                matched_canonical_id=None, predicted_level=None,
                is_correct_branch=False, is_finer_than_target=False,
                is_coarser_than_target=False, metadata={}))
        return out

    def _data(self):
        from granunlearn.evaluation.reference_eval import (
            load_associations_parquet, load_queries_parquet)
        queries = [q for q in load_queries_parquet(
            PILOT_DIR / "queries.parquet") if q.split == "test"][:200]
        assocs = load_associations_parquet(PILOT_DIR / "associations.parquet")
        return queries, {a.association_id: a for a in assocs}, assocs

    def test_identical_views_have_zero_floor(self):
        from evaluate_pilot100_final import _batch_composition_sensitivity
        qs, by_id, assocs = self._data()
        p = self._preds(qs, by_id)
        rep = _batch_composition_sensitivity({"MF": p}, {"MF": list(p)},
                                             qs, assocs)
        s = rep["per_state"]["MF"]
        assert s["num_raw_output_mismatches"] == 0
        assert s["raw_output_mismatch_rate"] == 0.0
        assert rep["max_abs_metric_delta"] == 0.0
        assert all(v == 0.0 for v in
                   s["metric_deltas_uniform_minus_gate"].values())

    def test_flipped_decodes_are_counted_and_bounded(self):
        from evaluate_pilot100_final import _batch_composition_sensitivity
        qs, by_id, assocs = self._data()
        uniform = self._preds(qs, by_id, flip=0)
        gate = self._preds(qs, by_id, flip=10)
        rep = _batch_composition_sensitivity({"MF": uniform},
                                             {"MF": gate}, qs, assocs)
        s = rep["per_state"]["MF"]
        assert s["num_test_queries_compared"] == 200
        assert s["num_raw_output_mismatches"] == 10
        assert s["raw_output_mismatch_rate"] == 0.05
        # the floor is reported as a magnitude, and stays far below the
        # between-model effects it must not be confused with
        assert rep["max_abs_metric_delta"] <= 0.2
        assert rep["interpretation"]

    def test_states_present_in_only_one_view_are_skipped(self):
        from evaluate_pilot100_final import _batch_composition_sensitivity
        qs, by_id, assocs = self._data()
        p = self._preds(qs, by_id)
        rep = _batch_composition_sensitivity({"MF": p}, {"MG": p},
                                             qs, assocs)
        assert rep["per_state"] == {}
        assert rep["max_abs_metric_delta"] == 0.0


# ── 9. recipe inheritance for the no-op arm ───────────────────────

class TestNoOpRecipeInheritance:
    """B0 applies zero updates, so the recipe it reports is the one it
    INHERITS from MF.  A null recipe there made the selection report look
    as though the no-op arm had been trained with no configuration at
    all; the fix reads the recipe back from MF's own summary instead of
    assuming the dataclass default still matches it."""

    def test_recipe_dict_round_trips(self):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        r = ReferenceRecipe()
        back = ReferenceRecipe.from_dict(r.to_dict())
        assert back == r
        # to_dict writes lora_target_modules as a list (JSON has no
        # tuples); from_dict must coerce it back or the frozen instance
        # stops being hashable and stops comparing equal
        assert isinstance(back.lora_target_modules, tuple)
        assert hash(back) == hash(r)

    def test_round_trip_survives_a_json_cycle(self):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        r = ReferenceRecipe(learning_rate=2e-5, num_epochs=5)
        back = ReferenceRecipe.from_dict(json.loads(json.dumps(r.to_dict())))
        assert back == r
        assert back.learning_rate == 2e-5 and back.num_epochs == 5

    def test_from_dict_ignores_unknown_keys(self):
        """A summary written by another revision may carry extra fields;
        loading it must not explode."""
        from granunlearn.training.reference_trainer import ReferenceRecipe
        d = ReferenceRecipe().to_dict()
        d["some_future_knob"] = 7
        assert ReferenceRecipe.from_dict(d) == ReferenceRecipe()

    def test_overrides_produce_a_distinct_recipe(self):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        base = ReferenceRecipe()
        swept = ReferenceRecipe(**{"learning_rate": 2e-5, "num_epochs": 5})
        assert swept != base
        assert swept.lora_r == base.lora_r, "only swept knobs may differ"

    def test_mf_summary_recipe_is_inherited_by_the_noop_arm(self):
        """End-to-end on the real checkpoint: B0's summary must carry
        MF's recorded recipe verbatim, and the adapter bytes must be
        unchanged by the copy."""
        mf = (REPO_ROOT / "data" / "checkpoints" / "mllmu_pilot100" / "MF")
        b0 = (REPO_ROOT / "data" / "checkpoints" / "mllmu_pilot100_unlearn"
              / "B0")
        if not (mf / "training_summary.json").exists():
            pytest.skip("MF checkpoint is gitignored / not trained here")
        from granunlearn.training.reference_trainer import ReferenceRecipe
        mf_recipe = json.loads(
            (mf / "training_summary.json").read_text())["recipe"]
        assert mf_recipe == ReferenceRecipe().to_dict(), \
            "the frozen pilot-100 recipe drifted from ReferenceRecipe"
        if not (b0 / "training_summary.json").exists():
            pytest.skip("B0 checkpoint not materialised here")
        summary = json.loads((b0 / "training_summary.json").read_text())
        assert summary["noop"] is True
        assert summary["num_optimizer_steps"] == 0
        assert summary["groups"] == []
        assert summary["recipe"] == mf_recipe
        assert "inherited" in summary["note"].lower()

    def test_inherited_recipe_helper_reads_the_checkpoint(self):
        mf_adapters = (REPO_ROOT / "data" / "checkpoints" / "mllmu_pilot100"
                       / "MF" / "adapters")
        if not (mf_adapters.parent / "training_summary.json").exists():
            pytest.skip("MF checkpoint is gitignored / not trained here")
        from granunlearn.training.reference_trainer import ReferenceRecipe
        from train_unlearning_baselines import mf_inherited_recipe
        assert mf_inherited_recipe(mf_adapters) == ReferenceRecipe()

    def test_inherited_recipe_helper_falls_back_when_absent(self, tmp_path):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        from train_unlearning_baselines import mf_inherited_recipe
        adapters = tmp_path / "MF" / "adapters"
        adapters.mkdir(parents=True)
        assert mf_inherited_recipe(adapters) == ReferenceRecipe()

    def test_noop_checkpoint_records_a_recipe_without_a_checkpoint(self,
                                                                  tmp_path):
        """make_noop_checkpoint must never emit recipe: null, even when
        the caller passes nothing (it falls back to the frozen default)."""
        from granunlearn.training.unlearning_trainer import \
            make_noop_checkpoint
        src = tmp_path / "src" / "adapters"
        src.mkdir(parents=True)
        (src / "adapter_model.safetensors").write_bytes(b"weights")
        out = tmp_path / "out"
        summary = make_noop_checkpoint("B0", src, out)
        from granunlearn.training.reference_trainer import ReferenceRecipe
        assert summary["recipe"] == ReferenceRecipe().to_dict()
        assert summary["groups"] == []
        assert json.loads(
            (out / "training_summary.json").read_text())["recipe"]
        assert (out / "adapters" / "adapter_model.safetensors").read_bytes() \
            == b"weights"


# ── 10. the generation batch layout is pinned ─────────────────────

class TestBatchLayoutIsPinned:
    """Batched greedy decoding is not bit-stable across batch layouts
    (section 8), so the layout itself is part of the experiment: it is
    driven ONLY by the frozen query order and the two recorded batch
    sizes.  These tests run the real grouping code against a stubbed
    ``_generate_batch`` — no model, no GPU — and pin the result, so a
    refactor of the batching loop cannot silently invalidate every
    cross-state comparison already committed."""

    def _stub(self):
        from granunlearn.evaluation.reference_eval import \
            ReferenceStateGenerator
        gen = object.__new__(ReferenceStateGenerator)  # skip __init__
        gen.device = "cpu"
        gen.max_image_pixels = 384 * 384
        calls: list[tuple[int, bool, list[bool]]] = []

        def fake_generate_batch(texts, images_per_sample, max_new_tokens):
            calls.append((
                len(texts),
                images_per_sample is not None,
                [im is not None for im in (images_per_sample or [])],
            ))
            return ["x"] * len(texts)

        gen._generate_batch = fake_generate_batch
        # a stand-in image object: only its None-ness is ever inspected
        gen._render_prompt = lambda q, a, r: (
            q.prompt, object() if q.image_ids else None)
        return gen, calls

    def _data(self):
        from granunlearn.evaluation.reference_eval import (
            load_associations_parquet, load_queries_parquet)
        queries = [q for q in load_queries_parquet(
            PILOT_DIR / "queries.parquet") if q.split == "test"]
        assocs = load_associations_parquet(PILOT_DIR / "associations.parquet")
        return queries, {a.association_id: a for a in assocs}

    def _run(self, batch_size, image_batch_size, missing_image_ids=()):
        qs, by = self._data()
        gen, calls = self._stub()
        if missing_image_ids:
            def render(q, a, r):
                if q.query_id in missing_image_ids:
                    return q.prompt, None
                return q.prompt, object() if q.image_ids else None
            gen._render_prompt = render
        out = gen.generate_for_queries(
            qs, by, REPO_ROOT, batch_size=batch_size,
            image_batch_size=image_batch_size)
        return qs, out, calls

    def test_layout_is_complete_and_deterministic(self):
        qs, out1, calls1 = self._run(8, 8)
        _, out2, calls2 = self._run(8, 8)
        assert len(out1) == len(out2) == len(qs) == 2259
        assert calls1 == calls2, "grouping must not vary between runs"

    def test_frozen_test_split_layout_at_the_recorded_batch_sizes(self):
        """The pilot-100 test split interleaves 747 image probes among
        1,512 text probes and never runs more than 3 image queries
        together, so image_batch_size=8 is never saturated: the layout is
        set by the query ORDER in the frozen parquet."""
        _, _, calls = self._run(8, 8)
        img = [c for c in calls if c[1]]
        txt = [c for c in calls if not c[1]]
        assert len(calls) == 1224
        assert len(img) == 567 and max(c[0] for c in img) == 3
        assert len(txt) == 657 and max(c[0] for c in txt) == 8
        assert sum(c[0] for c in calls) == 2259

    def test_a_batch_never_mixes_image_and_text_samples(self):
        _, _, calls = self._run(8, 8)
        for size, is_image, flags in calls:
            assert size >= 1
            if is_image:
                assert all(flags), "an image batch must carry every image"
            else:
                assert flags == []

    def test_caps_are_respected_for_every_batch_size(self):
        for batch_size, image_batch_size in ((8, 8), (4, 8), (8, 2), (1, 1)):
            _, out, calls = self._run(batch_size, image_batch_size)
            assert len(out) == 2259
            for size, is_image, _flags in calls:
                cap = image_batch_size if is_image else batch_size
                assert size <= cap, (batch_size, image_batch_size, size)

    def test_batch_sizes_are_part_of_the_recorded_configuration(self):
        """Different batch sizes give different layouts, hence (per
        section 8) different decodes for the SAME weights — which is why
        both sizes are written into every report's generation_config."""
        _, _, a = self._run(8, 8)
        _, _, b = self._run(8, 1)
        _, _, c = self._run(4, 8)
        assert a != b and a != c and b != c
        assert len(a) < len(b), "one-at-a-time images means more batches"

    def test_a_missing_image_degrades_to_single_sample_calls(self):
        """A missing photo must not poison the whole batch: that run of
        image queries falls back to per-sample text-only calls."""
        qs, _, calls = self._run(8, 8)
        victim = next(q.query_id for q in qs if q.image_ids)
        _, out, degraded = self._run(8, 8, missing_image_ids=(victim,))
        assert len(out) == 2259
        # same queries, but the layout changed around the missing image
        assert degraded != calls
        assert all(size >= 1 for size, _, _ in degraded)

    def test_progress_marks_are_emitted_without_touching_the_layout(
            self, caplog):
        """The progress log is observation-only: it must never influence
        grouping, and it must not fire for a pass shorter than one mark."""
        import logging
        qs, _, calls = self._run(8, 8)
        assert len(qs) == 2259
        # caplog captures from the start of the test, so drop the warm-up
        # run's marks: only one pass may contribute to the sequence below
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="reference_eval"):
            _, _, again = self._run(8, 8)
        assert again == calls
        marks = [r.getMessage() for r in caplog.records
                 if "generation progress" in r.getMessage()]
        assert marks, "a 2,259-query pass must report progress"
        pcts = [float(m.split("(")[1].split("%")[0]) for m in marks]
        assert pcts == sorted(pcts) and pcts[-1] <= 100.0
        assert all("/2259" in m for m in marks)


