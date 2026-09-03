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
   carries paired CIs for all four headline metrics against MF/MG/B0, and
   the B0 == MF no-op invariant holds;
4. provenance pins one recipe for all states, the base-model revision,
   every adapter hash and the photo resolution gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
INAT_PROV = (REPO_ROOT / "data" / "raw" / "inaturalist" / "pilot_v1"
             / "PROVENANCE.json")

EXPECTED_STATES = {"BASE", "MF", "MG", "MN"}
EXPECTED_METHODS = {"B0", "B1", "B2", "B2R", "B3"}
PAIRED_METRICS = ("tga", "wrong_branch", "retain_same", "retain_other")
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
        assert rep["iteration"] == 11
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


# ── 4. provenance record ──────────────────────────────────────────

class TestProvenanceEvidence:
    def test_identity_and_iteration(self):
        rep = _load("mllmu_pilot100_reference_provenance")
        assert rep["experiment_id"] == "mllmu_pilot100_iter11"
        assert rep["iteration"] == 11
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
