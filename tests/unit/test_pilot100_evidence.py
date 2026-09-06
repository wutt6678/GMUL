"""Integrity tests over the COMMITTED Iteration-11 pilot-100 evidence.

These guard the four reports that carry the iteration's scientific claims
(gate, selection, one-shot final evaluation, provenance) plus the
iNaturalist fetch contract.  They read committed JSON only — checkpoints,
prediction parquets and photo bytes are gitignored and are NOT touched —
so they run in the CPU-only CI job on a fresh checkout.

What they enforce:
1. the reference-state gate PASSED on pooled AND test metrics, with the
   gate inequalities recomputed from the reported numbers, and all three
   routes plus the taxonomic stratum represented on test;
2. selection used TRAIN+VAL only, selected one candidate per method, and
   the gradient-ascent divergence found by the sweep is recorded rather
   than silently dropped;
3. the final evaluation is one-shot, uniformly batched, coverage-complete,
   carries paired CIs for all six headline metrics against MF/MG/B0, and
   the B0 == MF no-op invariant holds;
4. provenance pins one recipe for all states, the base-model revision,
   every adapter hash and the photo resolution gate;
5. Iteration 11R: every report is bound to pilot100_v2 and names the v1
   commit it supersedes, the test split's ACTUAL exposure is stated rather
   than implied by the phrase "one-shot", equivalence against M_G is
   judged against a prespecified margin instead of inferred from a CI that
   happens to cross zero, the paired point estimates reproduce the
   published rates in their own unit, and the image-provenance strata are
   reported together with the confound they carry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
INAT_PROV = (REPO_ROOT / "data" / "raw" / "inaturalist" / "pilot_v1"
             / "PROVENANCE.json")

EXPECTED_STATES = {"BASE", "MF", "MG", "MN"}
EXPECTED_METHODS = {"B0", "B1", "B2", "B2R", "B3"}
#: Report order, asserted exactly: adding a metric to PAIRED_METRICS in
#: paired_ci.py without it reaching the report is a silent gap, and so is
#: the reverse.  filr and over_forgetting were added by Iteration 11R --
#: E2 had reported a paired CI for TGA while omitting the one for the
#: metric the paper's leakage claim actually rests on.
PAIRED_METRICS = ("filr", "tga", "wrong_branch", "over_forgetting",
                  "retain_same", "retain_other")
DATASET_VERSION = "pilot100_v2"
SUPERSEDED_COMMIT = "3850461"
EQUIVALENCE_MARGIN = 0.05
NUM_QUERIES = 6777
NUM_TEST_QUERIES = 2259


def _load(name: str) -> dict:
    p = REPORTS / f"{name}.json"
    if not p.exists():
        pytest.skip(f"committed evidence not present: {p}")
    return json.loads(p.read_text())


# ── 1. reference-state gate ───────────────────────────────────────

class TestReferenceGateEvidence:
    def test_identity_and_scope(self):
        rep = _load("mllmu_pilot100_reference_eval")
        assert rep["experiment_id"] == "mllmu_pilot100_iter11"
        assert rep["num_queries"] == NUM_QUERIES
        assert set(rep["states"]) == EXPECTED_STATES

    def test_gate_passed_on_pooled_and_test(self):
        rep = _load("mllmu_pilot100_reference_eval")
        assert rep["separation_gate"]["passed"] is True, \
            rep["separation_gate"]["reasons"]
        assert rep["separation_gate"]["reasons"] == []
        assert rep["separation_gate_test_split"]["passed"] is True, \
            rep["separation_gate_test_split"]["reasons"]
        assert rep["separation_gate_test_split"]["basis"] == \
            "test paraphrase split only"

    @pytest.mark.parametrize("basis", ["pooled", "test"])
    def test_gate_inequalities_recomputed(self, basis):
        """Do not trust the boolean: recompute the gate from the reported
        numbers, on both bases."""
        rep = _load("mllmu_pilot100_reference_eval")
        m = (rep["metrics_by_state"] if basis == "pooled"
             else rep["metrics_by_split"]["test"])
        fine = {s: m[s]["fine_recovery"]["baseline_accuracy"]
                for s in ("MF", "MG", "MN")}
        assert fine["MF"] - fine["MG"] >= 0.15, fine
        assert fine["MF"] - fine["MN"] >= 0.15, fine
        post = {s: m[s]["target_core"]["post_unlearning_accuracy"]
                for s in ("MG", "MN")}
        assert post["MG"] - post["MN"] >= 0.15, post
        for s in ("MF", "MG", "MN"):
            for sl in ("retain_same_entity", "retain_other_entity"):
                assert m[s][sl]["baseline_accuracy"] >= 0.5, (s, sl, m[s][sl])

    def test_generation_config_is_uniform_and_greedy(self):
        rep = _load("mllmu_pilot100_reference_eval")
        gc = rep["generation_config"]
        assert gc["do_sample"] is False
        assert gc["batch_size"] >= 1 and gc["image_batch_size"] >= 1
        assert "identical for every state" in gc["note"]

    def test_all_three_routes_and_the_taxonomic_stratum_on_test(self):
        rep = _load("mllmu_pilot100_reference_eval")
        hm = rep["hierarchy_metrics"]["test"]
        assert set(hm) == EXPECTED_STATES
        for state, block in hm.items():
            routes = block["by_route"]
            for r in ("text_to_text", "image_to_text", "image_text_to_text"):
                assert routes[r]["num_queries"] > 0, (state, r)
            types = block["by_hierarchy_type"]
            for t in ("semantic", "numeric", "taxonomic"):
                assert types[t]["num_queries"] > 0, (state, t)
            assert block["num_target_probes"] > 0, state

    def test_image_route_requires_the_nameless_probe_families(self):
        rep = _load("mllmu_pilot100_reference_eval")
        m = rep["metrics_by_split"]["test"]["MF"]
        assert m["image_route"]["num_queries"] > 0
        assert m["retain_same_entity_image"]["num_queries"] > 0
        assert m["retain_other_entity_image"]["num_queries"] > 0

    def test_split_semantics_documented(self):
        rep = _load("mllmu_pilot100_reference_eval")
        assert "TEST" in rep["metrics_split_semantics"]
        assert set(rep["metrics_by_split"]) == {"train", "val", "test"}
        assert set(rep["hierarchy_metrics"]) == {
            "test", "train", "val", "pooled"}


# ── 2. selection on train+val only ────────────────────────────────

class TestSelectionEvidence:
    def test_scope_is_trainval_and_says_so(self):
        rep = _load("mllmu_pilot100_unlearning_selection")
        assert rep["tag"] == "pilot100"
        assert rep["selection_scope"] == ["train", "val"]
        assert "test" not in rep["selection_scope"]
        assert "NOT generated" in rep["scope_note"]
        assert "held out" in rep["basis"]

    def test_one_winner_per_method(self):
        rep = _load("mllmu_pilot100_unlearning_selection")
        assert set(rep["selected"]) == EXPECTED_METHODS
        for method, cid in rep["selected"].items():
            assert cid in rep["candidates"], (method, cid)
            assert rep["candidates"][cid]["method"] == method

    def test_winner_minimises_distance_within_its_method(self):
        rep = _load("mllmu_pilot100_unlearning_selection")
        for method, cid in rep["selected"].items():
            same = [(c, i["distance_to_mg"])
                    for c, i in rep["candidates"].items()
                    if i["method"] == method]
            best = min(same, key=lambda t: t[1])
            assert best[0] == cid, (method, cid, best, same)

    def test_wide_grid_was_actually_swept(self):
        rep = _load("mllmu_pilot100_unlearning_selection")
        cands = rep["candidates"]
        assert len(cands) >= 15
        for method in EXPECTED_METHODS:
            n = sum(1 for i in cands.values() if i["method"] == method)
            assert n >= 1, method
        # every candidate records the recipe it was trained with
        for cid, info in cands.items():
            assert info["config"]["recipe"], cid

    def test_gradient_ascent_divergence_is_recorded(self):
        """B1 at lr >= 2e-5 drove the ascent loss to 45-139 and collapsed
        to an all-zero behaviour vector.  The sweep must keep those rows
        visible: a collapsed candidate is a result, not a missing value."""
        rep = _load("mllmu_pilot100_unlearning_selection")
        collapsed = [cid for cid, i in rep["candidates"].items()
                     if i["method"] == "B1"
                     and all(v == 0.0 for v in i["vector"].values())]
        assert collapsed, "expected the diverged B1 candidates to be present"
        healthy = [cid for cid, i in rep["candidates"].items()
                   if i["method"] == "B1"
                   and any(v > 0 for v in i["vector"].values())]
        assert healthy
        # the selected B1 is a healthy one, never a collapsed one
        assert rep["selected"]["B1"] in healthy
        # and the collapsed ones are further from MG than the selected one
        sel_d = rep["candidates"][rep["selected"]["B1"]]["distance_to_mg"]
        for cid in collapsed:
            assert rep["candidates"][cid]["distance_to_mg"] > sel_d, cid

    def test_noop_arm_reports_the_inherited_recipe(self):
        """B0 applies zero updates, so its recipe is the one inherited
        from MF — it must be the frozen shared recipe, never null."""
        from granunlearn.training.reference_trainer import ReferenceRecipe
        rep = _load("mllmu_pilot100_unlearning_selection")
        b0 = rep["candidates"]["B0"]["config"]
        assert b0["recipe"] == ReferenceRecipe().to_dict()
        assert b0["groups"] == []
        assert b0["num_optimizer_steps"] == 0
        assert b0["noop"] is True
        # cross-check against the provenance record when it is committed
        # (read directly: a missing provenance file must not skip this)
        prov_path = REPORTS / "mllmu_pilot100_reference_provenance.json"
        if prov_path.exists():
            prov = json.loads(prov_path.read_text())
            assert b0["recipe"] == prov["recipe"]

    def test_mg_reference_vector_is_well_formed(self):
        rep = _load("mllmu_pilot100_unlearning_selection")
        ref = rep["reference"]
        assert ref["state"] == "MG"
        for comp, v in ref["vector"].items():
            assert v is None or 0.0 <= v <= 1.0, (comp, v)
        assert set(ref["vector"]) == {
            "filr", "tga", "ancestor", "retain_same", "retain_other",
            "over", "wrong"}


# ── 3. one-shot final evaluation ──────────────────────────────────

class TestFinalEvaluationEvidence:
    def test_one_shot_protocol_and_scope(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        assert rep["experiment_id"] == "mllmu_pilot100_iter11"
        assert rep["iteration"] == "11R"
        one = rep["one_shot"]
        assert one["num_test_queries"] == NUM_TEST_QUERIES
        assert one["num_queries_total"] == NUM_QUERIES
        assert one["selection_scope"] == ["train", "val"]
        assert one["generation_config"]["do_sample"] is False
        assert "SAME test-query list" in one["protocol"]

    def test_every_reported_state_is_present_and_complete(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        assert set(rep["states"]) == EXPECTED_STATES | EXPECTED_METHODS
        for state, n in rep["test_query_coverage"].items():
            assert n == NUM_TEST_QUERIES, (state, n)

    def test_b0_equals_mf_noop_invariant_holds(self):
        """B0 is the MF adapter copied unchanged (identical SHA-256), so
        under a uniform batch layout it must reproduce MF exactly."""
        rep = _load("mllmu_pilot100_final_evaluation")
        inv = rep["b0_equals_mf_invariant"]
        assert inv["checked"] is True
        assert inv["passed"] is True, inv
        assert inv["num_raw_output_mismatches"] == 0
        assert inv["num_test_queries_compared"] == NUM_TEST_QUERIES
        for metric in PAIRED_METRICS:
            assert inv["paired_diffs_vs_MF"][metric] == 0.0, metric
            assert tuple(inv["paired_cis_vs_MF"][metric]) == (0.0, 0.0)

    def test_paired_cis_cover_all_states_metrics_and_references(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        paired = rep["paired_cis_test"]
        assert paired["split"] == "test"
        assert tuple(paired["metrics"]) == PAIRED_METRICS
        assert set(paired["reference_states"]) == {"MF", "MG", "B0"}
        comps = paired["comparisons"]
        for state in EXPECTED_STATES | EXPECTED_METHODS:
            if state in ("MF", "MG", "B0"):
                continue
            assert state in comps, state
            for ref in ("MF", "MG", "B0"):
                block = comps[state][f"vs_{ref}"]
                assert set(block) == set(PAIRED_METRICS), (state, ref)
                for metric, d in block.items():
                    lo, hi = d["ci"]
                    assert lo <= d["diff"] <= hi, (state, ref, metric, d)
                    assert d["num_units"] > 1
                    assert d["num_rows"] > 0

    def test_paired_bootstrap_metadata_recorded(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        meta = rep["paired_cis_test"]["statistical_metadata"]
        assert meta["n_bootstrap"] >= 1000
        assert meta["ci_level"] == 0.95
        assert meta["seed"] == 42
        assert "entity" in meta["clustering_unit"]
        assert "intersection" in meta["pairing"]

    def test_batch_layout_noise_floor_is_measured(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        sens = rep["batch_composition_sensitivity"]
        assert set(sens["per_state"]) == EXPECTED_STATES
        # the target-side floor — the slices the claims are made on —
        # must stay an order of magnitude below the effects
        assert sens["max_abs_metric_delta"] <= 0.05, sens
        # the retain floor is allowed more room: BASE answers in long
        # free-form prose that is truncated at max_new_tokens, so its
        # wording (and hence its retain slice) moves more between layouts
        # than the hierarchy-scored target slices do
        assert sens["max_abs_retain_delta"] <= 0.10, sens
        assert sens["target_side_metrics"] == ["filr", "tga", "wrong_branch"]
        assert sens["retain_metrics"] == ["retain_same", "retain_other"]
        assert sens["interpretation"]
        for state, s in sens["per_state"].items():
            assert s["num_test_queries_compared"] == NUM_TEST_QUERIES, state
            assert set(s["metric_deltas_uniform_minus_gate"]) >= \
                set(sens["target_side_metrics"]) | set(sens["retain_metrics"])

    def test_routes_and_strata_reported_for_every_state(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        hm = rep["hierarchy_metrics_test"]
        assert set(hm) == EXPECTED_STATES | EXPECTED_METHODS
        for state, block in hm.items():
            assert block["split"] == "test"
            for r in ("text_to_text", "image_to_text", "image_text_to_text"):
                assert block["by_route"][r]["num_queries"] > 0, (state, r)
            for key in ("retain_same_entity_all_routes",
                        "retain_other_entity_all_routes"):
                assert block[key]["num_queries"] > 0, (state, key)
            assert block["by_hierarchy_type"]["taxonomic"][
                "num_queries"] > 0, state
            assert set(block["failure_rates"]) == {
                "under_forgetting", "over_forgetting", "wrong_branch",
                "refusal", "hallucination"}

    def test_unlearning_moves_toward_mg_and_keeps_retention(self):
        """The iteration's central claim, read off the committed numbers:
        the selected granularity-aware candidate (B3) raises target
        granularity accuracy above the no-op/MF level toward MG, without
        giving up same-entity retention."""
        rep = _load("mllmu_pilot100_final_evaluation")
        hm = rep["hierarchy_metrics_test"]
        tga = {s: hm[s]["tga"] for s in hm}
        filr = {s: hm[s]["filr"] for s in hm}
        rsame = {s: hm[s]["retain_same_entity_all_routes"][
            "baseline_accuracy"] for s in hm}
        assert tga["B3"] > tga["B0"] + 0.05, tga
        assert filr["B3"] < filr["B0"] - 0.05, filr
        assert abs(tga["B3"] - tga["MG"]) < abs(tga["B0"] - tga["MG"]), tga
        assert rsame["B3"] >= rsame["B0"] - 0.05, rsame

    def test_provenance_of_each_reported_state(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        prov = rep["provenance"]
        assert set(prov) == EXPECTED_STATES | EXPECTED_METHODS
        for state in EXPECTED_STATES:
            assert prov[state]["kind"] == "reference_state"
            assert prov[state]["generated_under_uniform_batch_layout"] is True
        for method in EXPECTED_METHODS:
            p = prov[method]
            assert p["kind"] == "selected_candidate"
            assert p["candidate_id"]
            assert p["num_test_predictions"] == NUM_TEST_QUERIES

    def test_each_state_is_bound_to_its_own_prediction_file(self):
        """Every state must name the parquet it was read from and pin its
        bytes, so "reused" can never mean "stale" or "someone else's
        file".  The parquets themselves are gitignored, so only the shape
        of the record can be checked here."""
        rep = _load("mllmu_pilot100_final_evaluation")
        seen = {}
        for state, p in rep["provenance"].items():
            fp = p["test_predictions_fingerprint"]
            assert len(fp["sha256"]) == 64, state
            assert int(fp["bytes"]) > 0, state
            datetime.fromisoformat(fp["written_utc"])
            assert p["test_predictions_file"].endswith(".parquet"), state
            seen[fp["sha256"]] = state
        # nine distinct files: no state silently shares another's rows
        assert len(seen) == len(EXPECTED_STATES | EXPECTED_METHODS)

    def test_reuse_is_documented_as_recovery_not_a_second_look(self):
        """The one-shot protocol survives a report re-assembly only if the
        report says so explicitly: generation happens once per state, and
        a later pass that reads persisted parquets generates nothing."""
        rep = _load("mllmu_pilot100_final_evaluation")
        one = rep["one_shot"]
        assert isinstance(one["assembled_without_generation"], bool)
        # reuse is defined by PROVENANCE, not by a filename: the sidecar
        # must match adapter, revision, dataset, config and scoring code,
        # and the rows must be exactly the frozen test probe set.
        assert "provenance sidecar" in one["reuse_semantics"]
        assert "refused and regenerated" in one["reuse_semantics"]
        assert "never a second look" in one["reuse_semantics"]
        reused = set(one["reused_existing_test_predictions"])
        assert reused <= set(rep["states"])
        if one["assembled_without_generation"]:
            assert reused == set(rep["states"]), \
                "a no-generation assembly must reuse every state"


# ── 4. provenance record ──────────────────────────────────────────

class TestProvenanceEvidence:
    def test_identity_and_iteration(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        assert rep["experiment_id"] == "mllmu_pilot100_iter11"
        # a string, not 11: this record certifies reports that say "11R" on
        # pilot100_v2, so an integer 11 here would contradict them
        assert rep["iteration"] == "11R"
        assert rep["tag"] == "pilot100"
        assert "100 entities" in rep["dataset"]

    def test_recipe_is_the_frozen_identical_one(self):
        from granunlearn.training.reference_trainer import ReferenceRecipe
        rep = _load("mllmu_pilot100_reference_provenance")
        assert rep["recipe"] == ReferenceRecipe().to_dict()
        r = rep["recipe"]
        assert (r["lora_r"], r["lora_alpha"], r["seed"], r["num_epochs"],
                r["learning_rate"], r["max_image_pixels"]) == \
            (16, 32, 42, 10, 1e-4, 384 * 384)
        assert "MF/MG/MN share the recipe verbatim" in rep["notes"][0]

    def test_base_model_revision_pinned(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        bm = rep["base_model"]
        assert bm["model_id"] == "Qwen/Qwen3.5-9B"
        assert bm["revision"] and len(bm["revision"]) == 40
        # the local cache path is diagnostics, not part of the contract
        assert "diagnostics" in bm

    def test_dataset_artifacts_are_hash_bound(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        hashes = rep["dataset_hashes_sha256"]
        required = {
            "data/mllmu_hier_pilot100/associations.parquet",
            "data/mllmu_hier_pilot100/queries.parquet",
            "data/mllmu_hier_pilot100/manifest.json",
            "data/reports/mllmu_pilot100_target_retain.json",
            "data/mllmu_hier_pilot100/training/MF.jsonl",
            "data/mllmu_hier_pilot100/training/MG.jsonl",
            "data/mllmu_hier_pilot100/training/MN.jsonl",
            "data/mllmu_hier_pilot100/unlearning/fine_target.jsonl",
            "data/mllmu_hier_pilot100/unlearning/target_level.jsonl",
            "data/mllmu_hier_pilot100/unlearning/retain.jsonl",
        }
        assert required <= set(hashes), sorted(required - set(hashes))
        for path, sha in hashes.items():
            assert len(sha) == 64, path
            # committed artifacts must still hash-match on a fresh clone
            p = REPO_ROOT / path
            if p.exists():
                import hashlib
                assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, path

    def test_frozen_parquet_hashes_match_the_dataset_manifest(self):
        man = json.loads((REPO_ROOT / "data" / "mllmu_hier_pilot100"
                          / "manifest.json").read_text())
        rep = _load("mllmu_pilot100_reference_provenance")
        for name, sha in man["frozen_artifact_sha256"].items():
            if not name.endswith(".parquet"):
                continue
            key = f"data/mllmu_hier_pilot100/{name}"
            assert rep["dataset_hashes_sha256"][key] == sha, key

    def test_reference_states_and_candidates_hash_bound(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        states = rep["reference_state_checkpoints"]
        assert set(states) == {"MF", "MG", "MN"}
        for s, e in states.items():
            assert len(e["adapter_sha256"]) == 64, s
            assert e["num_optimizer_steps"] > 0, s
        # identical recipe + dataset size -> identical step count for
        # MF/MG, and fewer for MN (387 vs 477 examples)
        assert states["MF"]["num_optimizer_steps"] == \
            states["MG"]["num_optimizer_steps"]
        assert states["MN"]["num_optimizer_steps"] < \
            states["MF"]["num_optimizer_steps"]
        cands = rep["unlearning_checkpoints"]
        assert len(cands) >= 16
        assert "B0" in cands
        for cid, e in cands.items():
            if cid.startswith("selected/"):
                continue
            assert len(e["adapter_sha256"]) == 64, cid

    def test_b0_adapter_is_bit_identical_to_mf(self):
        """The no-op baseline must be the MF adapter byte-for-byte; that
        is what makes the final evaluation's B0 == MF check meaningful."""
        rep = _load("mllmu_pilot100_reference_provenance")
        mf = rep["reference_state_checkpoints"]["MF"]["adapter_sha256"]
        assert rep["unlearning_checkpoints"]["B0"]["adapter_sha256"] == mf
        staged = rep["unlearning_checkpoints"].get("selected/B0")
        if staged:
            assert staged["adapter_sha256"] == mf

    def test_candidate_grid_recorded_with_swept_knobs(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        grid = rep["candidate_grid"]
        assert len(grid) >= 16
        ids = [c["candidate_id"] for c in grid]
        assert len(ids) == len(set(ids))
        for c in grid:
            assert set(c["overrides"]) <= {"learning_rate", "num_epochs"}
            if not c["noop"]:
                assert c["groups"], c["candidate_id"]

    def test_selection_provenance_matches_the_selection_report(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        sel = _load("mllmu_pilot100_unlearning_selection")
        assert rep["selection"]["selected"] == sel["selected"]
        assert rep["selection"]["selection_scope"] == sel["selection_scope"]

    def test_final_evaluation_report_is_hash_bound(self):
        """Provenance must pin the headline report, not just the dataset
        and the weights: otherwise nothing ties these checkpoints to the
        numbers the claims are read off.  Both files are committed, so the
        hash is re-verified live here."""
        import hashlib
        rep = _load("mllmu_pilot100_reference_provenance")
        fe = rep["final_evaluation"]
        assert fe["report"] == \
            "data/reports/mllmu_pilot100_final_evaluation.json"
        live = REPO_ROOT / fe["report"]
        assert hashlib.sha256(live.read_bytes()).hexdigest() == \
            fe["report_sha256"], "provenance no longer matches the report"
        assert fe["num_test_queries"] == NUM_TEST_QUERIES
        assert fe["selection_scope_used"] == ["train", "val"]
        assert fe["b0_equals_mf_passed"] is True
        # cross-consistency with the report's own floor measurement
        final = _load("mllmu_pilot100_final_evaluation")
        sens = final["batch_composition_sensitivity"]
        assert fe["batch_layout_noise_floor"]["max_abs_metric_delta"] == \
            sens["max_abs_metric_delta"]
        assert fe["batch_layout_noise_floor"]["max_abs_retain_delta"] == \
            sens["max_abs_retain_delta"]
        assert fe["assembled_without_generation"] == \
            final["one_shot"]["assembled_without_generation"]

    def test_inaturalist_fetch_contract_pinned(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        inat = rep["inaturalist_stratum"]
        assert inat["available"] is True
        assert inat["num_photos"] == 432
        assert inat["photo_sha256_recorded"] is True
        assert inat["min_longest_edge_px"] >= 200
        assert inat["num_rejected_candidates"] >= 0
        gate = inat["resolution_gate"]
        assert gate["min_image_edge_px"] >= 200
        assert "square.jpg" in gate["pool_filter"]
        # the committed PROVENANCE.json is the contract's source of truth
        on_disk = json.loads(INAT_PROV.read_text())
        assert on_disk["num_photos"] == inat["num_photos"]
        assert on_disk["resolution_gate"] == gate

    def test_environment_is_recorded_but_flagged_diagnostic(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        env = rep["environment"]
        for key in ("python", "torch", "transformers", "peft",
                    "cuda_runtime", "gpu", "platform"):
            assert env.get(key), key
        # a null gpu means the record was written with CUDA hidden, i.e.
        # not on the machine that produced the checkpoints
        assert "RTX 6000 Ada" in env["gpu"]
        assert any("diagnostics" in n and "not part of the contract" in n
                   for n in rep["notes"])


# ── 5. Iteration 11R: repaired visual split, honest evidence ────────

class TestIteration11REvidence:
    """The repairs this iteration exists for.

    Each test is named for the defect it closes rather than for the field
    it reads: v1's wording survived review precisely because nothing in the
    suite forbade it.
    """

    V2_REPORTS = ("mllmu_pilot100_reference_eval",
                  "mllmu_pilot100_unlearning_selection",
                  "mllmu_pilot100_final_evaluation",
                  "mllmu_pilot100_reference_provenance")
    #: the six JSONLs the reference states and every candidate were fitted
    #: on. Byte-identity here is the entire no-retraining claim.
    TRAINING_JSONLS = (
        "data/mllmu_hier_pilot100/training/MF.jsonl",
        "data/mllmu_hier_pilot100/training/MG.jsonl",
        "data/mllmu_hier_pilot100/training/MN.jsonl",
        "data/mllmu_hier_pilot100/unlearning/fine_target.jsonl",
        "data/mllmu_hier_pilot100/unlearning/target_level.jsonl",
        "data/mllmu_hier_pilot100/unlearning/retain.jsonl")

    @pytest.mark.parametrize("name", V2_REPORTS)
    def test_every_report_is_bound_to_the_repaired_dataset(self, name):
        rep = _load(name)
        assert rep["dataset_version"] == DATASET_VERSION, name

    @pytest.mark.parametrize("name", V2_REPORTS[:3])
    def test_each_report_names_what_it_supersedes(self, name):
        """A regenerated report that does not name the one it replaces
        leaves both in the tree looking current."""
        sup = _load(name)["supersedes"]
        assert sup["commit"] == SUPERSEDED_COMMIT, name
        assert sup["iteration"] == 11
        assert sup["dataset_version"] == "pilot100_v1"
        assert "images[0]" in sup["reason"], name
        assert "git show" in sup["v1_numbers_preserved_in"]

    def test_provenance_pins_the_final_report_bytes(self):
        prov = _load("mllmu_pilot100_reference_provenance")
        fe = prov["final_evaluation"]
        on_disk = hashlib.sha256(
            (REPORTS / "mllmu_pilot100_final_evaluation.json")
            .read_bytes()).hexdigest()
        assert fe["report_sha256"] == on_disk
        assert fe["superseded"] is False
        assert fe["dataset_version"] == DATASET_VERSION
        assert prov["selection"]["superseded"] is False

    def test_no_retraining_was_needed_and_the_record_proves_it(self):
        tr = _load("mllmu_pilot100_reference_provenance")["dataset_transition"]
        assert (tr["from"], tr["to"]) == ("pilot100_v1", DATASET_VERSION)
        assert tr["training_jsonls_byte_identical"] is True
        assert tr["retraining_required"] is False
        assert tr["v1_source_commit"] == SUPERSEDED_COMMIT
        assert set(self.TRAINING_JSONLS) <= set(tr["artifacts_unchanged"])
        assert not (set(self.TRAINING_JSONLS) & set(tr["artifacts_changed"]))
        # something DID move, so this is a real transition, not a no-op
        assert tr["artifacts_changed"]
        assert len(tr["v1_artifact_sha256"]) == 14

    def test_the_test_split_is_not_claimed_to_be_untouched(self):
        """Iteration 11 called this a one-shot, untouched test split. The
        gate scores all 2,259 test queries for BASE/MF/MG/MN and applies a
        test-split separation criterion BEFORE any candidate is selected,
        so the honest statement is narrower."""
        t = _load("mllmu_pilot100_final_evaluation")["test_split_exposure"]
        assert t["test_split_untouched"] is False
        assert t["candidates_selected_without_test_predictions"] is True
        assert t["candidate_selection_scope"] == ["train", "val"]
        assert set(t["reference_states_scored_on_test_before_selection"]) \
            == EXPECTED_STATES
        assert set(t["num_test_queries_scored_by_the_gate"]) \
            == EXPECTED_STATES
        for state, n in t["num_test_queries_scored_by_the_gate"].items():
            assert n == NUM_TEST_QUERIES, state
        assert t["gate_applies_a_test_split_separation_criterion"] is True
        assert "one-shot, untouched test split" in t["exposure"]
        assert t["what_this_does_not_licence"]
        assert "CONFIRMATION" in t["what_would_be_needed"]

    def test_equivalence_is_judged_against_a_prespecified_margin(self):
        """A CI crossing zero means 'no significant difference detected',
        not equivalence. Equivalence needs a margin fixed before looking,
        and an interval narrow enough to fit inside it."""
        eq = _load("mllmu_pilot100_final_evaluation")["equivalence_vs_MG"]
        assert eq["margin"] == EQUIVALENCE_MARGIN
        assert eq["metric"] == "tga"
        assert eq["reference"] == "MG"
        # every state except M_G itself: an oracle-versus-oracle interval
        # would be degenerate, not a test
        assert set(eq["states"]) == \
            (EXPECTED_STATES | EXPECTED_METHODS) - {"MG"}
        assert eq["margin_rationale"]
        for state, b in eq["states"].items():
            lo, hi = b["ci"]
            assert lo <= b["diff"] <= hi, state
            # ci_half_width is computed from the UNROUNDED percentiles, so
            # recomputing it from the 4dp interval in the report can differ
            # in the last place; a tolerance, not equality
            assert abs(b["ci_half_width"] - (hi - lo) / 2) < 1e-4, state
            assert b["num_entity_clusters"] > 1, state
            assert b["num_rows"] > 0, state
            assert b["power_note"], state
            assert b["row_point_estimates"], state
            # equivalence only when the WHOLE interval is inside the margin
            assert b["equivalence_concluded"] == (
                lo > -EQUIVALENCE_MARGIN and hi < EQUIVALENCE_MARGIN), state
            assert b["significant_difference"] == (lo > 0.0 or hi < 0.0), \
                state

    def test_no_state_is_declared_equivalent_to_the_oracle(self):
        """No comparison concludes equivalence, and the report must say WHY
        per state.  The reason is decided by SIGNIFICANCE first: seven
        intervals exclude zero, so a real difference WAS detected against
        the oracle, and only B3's straddles zero.  Achieved power is a
        separate second axis: three intervals (B0, M_F, B3) are at least as
        wide as the margin, so those designs could not have concluded
        equivalence even at a true difference of exactly zero.  For B0 and
        M_F that qualifies the ABSENT equivalence claim and must not be
        allowed to weaken the difference they did detect — which is exactly
        what the 11R wording got wrong."""
        states = _load("mllmu_pilot100_final_evaluation")[
            "equivalence_vs_MG"]["states"]
        significant = straddling = wide = 0
        for state, b in states.items():
            assert b["equivalence_concluded"] is False, state
            note = b["power_note"]
            lo, hi = b["ci"]
            if lo > 0.0 or hi < 0.0:
                significant += 1
                assert note.startswith("DIFFERENT FROM MG"), (state, note)
                assert "no significant difference" not in note.lower(), state
                assert "IS detected" in note, state
            else:
                straddling += 1
                assert "INDETERMINATE" in note, state
                assert "straddles zero" in note, state
            if b["ci_half_width"] >= EQUIVALENCE_MARGIN:
                wide += 1
                assert "could not conclude equivalence" in note, state
            else:
                assert "not excluded" in note, state
        # as measured on the committed v2 evidence: seven intervals exclude
        # zero, one straddles it, three are at least as wide as the margin
        assert (significant, straddling, wide) == (7, 1, 3), \
            (significant, straddling, wide)

    def test_the_one_interval_crossing_zero_says_indeterminate(self):
        """B3 is the only state whose TGA interval against M_G crosses zero.
        It must not be reported as equivalent, and the note must say the
        design could not have concluded equivalence either way."""
        b = _load("mllmu_pilot100_final_evaluation")[
            "equivalence_vs_MG"]["states"]["B3"]
        lo, hi = b["ci"]
        assert lo < 0.0 < hi, b
        assert b["significant_difference"] is False
        assert b["equivalence_concluded"] is False
        assert "INDETERMINATE" in b["power_note"]
        assert "not equivalent" in b["power_note"]
        # every other state IS significantly below the oracle
        others = _load("mllmu_pilot100_final_evaluation")[
            "equivalence_vs_MG"]["states"]
        for state, ob in others.items():
            if state != "B3":
                assert ob["significant_difference"] is True, state

    def test_paired_point_estimates_reproduce_the_published_rates(self):
        """The bootstrap resamples ENTITY clusters, so its centre is an
        entity-macro mean, while hierarchy_metrics publishes row-micro
        rates. Subtracting two published rates therefore yields row_diff,
        NOT diff. Both are reported, and the row ones must equal the
        published numbers exactly or the two views silently diverge."""
        rep = _load("mllmu_pilot100_final_evaluation")
        hm, pc = rep["hierarchy_metrics_test"], rep["paired_cis_test"]

        def published(state, metric):
            b = hm[state]
            return {
                "filr": b["filr"],
                "tga": b["tga"],
                "wrong_branch": b["failure_rates"]["wrong_branch"],
                "over_forgetting": b["failure_rates"]["over_forgetting"],
                "retain_same": b["retain_same_entity_all_routes"][
                    "baseline_accuracy"],
                "retain_other": b["retain_other_entity_all_routes"][
                    "baseline_accuracy"]}[metric]

        checked = 0
        for state, refs in pc["comparisons"].items():
            for key, block in refs.items():
                ref = key[len("vs_"):]
                for metric in PAIRED_METRICS:
                    pe = block[metric]["point_estimates"]
                    assert pe["row_a"] == round(published(state, metric), 4), \
                        (state, key, metric)
                    assert pe["row_b"] == round(published(ref, metric), 4), \
                        (state, key, metric)
                    # row_diff is computed from the UNROUNDED means, so it
                    # can differ from the difference of the two rounded
                    # values: three 4dp roundings of 5e-5 each bound it at
                    # 1.5e-4 (measured max on this evidence: 1.0e-4, in 30
                    # of 144 comparisons)
                    assert abs(pe["row_diff"]
                               - (pe["row_a"] - pe["row_b"])) <= 1.5e-4, \
                        (state, key, metric)
                    checked += 1
        # 24 comparisons present x 6 metrics, as measured on the committed
        # v2 evidence: a silently dropped comparison would drop this count
        assert checked == 144, checked
        units = pc["statistical_metadata"]["point_estimate_units"]
        assert "row_diff" in units and "diff" in units

    def test_filr_gets_its_own_paired_ci(self):
        """The defect: E2 reported a paired CI for TGA while omitting the
        one for FILR, the metric the leakage claim rests on."""
        pc = _load("mllmu_pilot100_final_evaluation")["paired_cis_test"]
        assert pc["metrics"][0] == "filr"
        for state, refs in pc["comparisons"].items():
            for key, block in refs.items():
                assert "filr" in block, (state, key)
                assert "over_forgetting" in block, (state, key)

    def test_the_two_units_genuinely_differ_on_this_evidence(self):
        """Not a hypothetical: B3's entity-macro FILR interval against M_G
        crosses zero while its row-level difference is plainly positive.
        Publishing only one of the two would let a reader subtract the
        headline rates and get a number no interval covers."""
        b = _load("mllmu_pilot100_final_evaluation")[
            "paired_cis_test"]["comparisons"]["B3"]["vs_MG"]["filr"]
        lo, hi = b["ci"]
        pe = b["point_estimates"]
        assert lo < 0.0 < hi, b
        assert pe["row_diff"] > 0.0, b
        assert pe["row_a"] > pe["row_b"]

    def test_image_strata_are_populated_and_cover_only_image_probes(self):
        rep = _load("mllmu_pilot100_final_evaluation")
        for state, block in rep["hierarchy_metrics_test"].items():
            ip = block["by_image_provenance"]
            ho, sp = ip["held_out_photo"], ip["seen_photo_unseen_wording"]
            assert ho["num_queries"] == 90, (state, ho)
            assert sp["num_queries"] == 180, (state, sp)
            routes = block["by_route"]
            assert ho["num_queries"] + sp["num_queries"] == \
                routes["image_to_text"]["num_queries"] + \
                routes["image_text_to_text"]["num_queries"], state
            # NOT a partition of all target probes: the 945 text_to_text
            # probes are in neither stratum
            assert ho["num_queries"] + sp["num_queries"] < \
                block["num_target_probes"], state
            assert "never in training" in ip["_note"], state

    def test_the_strata_are_reported_with_their_confound(self):
        """held_out_photo is 90 probes and image_text_to_text is also 90,
        so counts alone could pass this off as the route split relabelled.
        At query level both routes contain both strata; but the flag does
        correspond 1:1 with the source dataset, which caps what a stratum
        CONTRAST can show. Both facts must be stated."""
        s = _load("mllmu_pilot100_final_evaluation")["image_provenance_strata"]
        assert s["derived_from"].startswith(
            "QueryRecord.image_seen_in_training")
        assert "NOT the input-route split" in s["not_the_route_split"]
        assert "the two partitions cross" in s["not_the_route_split"]
        assert "1:1" in s["confound"]
        assert "iNaturalist" in s["confound"] and "MLLMU" in s["confound"]
        assert "cannot be attributed to photograph novelty alone" \
            in s["confound"]

    def test_fine_leakage_is_attributed_to_filr_not_wrong_branch(self):
        """v1 claimed wrong-branch stability showed 'nothing leaks a finer
        branch'. Leaking a finer level IS the under_forgetting category
        FILR counts, so wrong-branch stability cannot show it."""
        notes = _load("mllmu_pilot100_final_evaluation")["notes"]
        assert any("FILR — not wrong-branch stability —" in n for n in notes)
        assert any("nothing leaks a finer branch" in n for n in notes)
        # the units note is what stops a reader subtracting two published
        # rates and treating the result as the quantity the CI covers
        assert any("entity-macro" in n.lower() and "row-micro" in n.lower()
                   for n in notes), notes

    def test_the_noop_invariant_holds_on_all_six_metrics(self):
        """B0 is the MF adapter copied unchanged, so under one uniform batch
        layout it must reproduce MF exactly -- now including the two metrics
        11R added."""
        inv = _load("mllmu_pilot100_final_evaluation")[
            "b0_equals_mf_invariant"]
        assert set(inv["paired_diffs_vs_MF"]) == set(PAIRED_METRICS)
        assert inv["num_raw_output_mismatches"] == 0
