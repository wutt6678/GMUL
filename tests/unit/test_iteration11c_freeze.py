"""Iteration 11C stage 2: the confirmation protocol freeze.

These tests exist because a protocol freeze that is not enforced is a
comment.  The value of freezing before inference is entirely in the
refusals: a freeze that binds a checkpoint which has moved, or that lets a
second copy of itself be written after scoring, is worse than no freeze at
all because it looks like discipline.

Two kinds of assertion live here.  The committed freeze is checked against
the evidence it claims to be derived from — the 11R prediction manifest, the
selection report, the power analysis — because a parameter written into the
freeze by hand instead of read from its source is a second copy that can
diverge.  The refusal paths are exercised by mutation, because a gate that
has never been observed to fire cannot be trusted to.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
FREEZE_PATH = REPORTS / "mllmu_pilot100_confirmation_freeze.json"
MANIFEST_PATH = REPORTS / "mllmu_pilot100_prediction_manifest.json"
POWER_PATH = REPORTS / "mllmu_pilot100_confirmation_power.json"
SELECTION_PATH = REPORTS / "mllmu_pilot100_unlearning_selection.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import freeze_confirmation_protocol as fz  # noqa: E402
from freeze_confirmation_protocol import (  # noqa: E402
    ANALYSIS_SCRIPTS,
    CONFIRMATION_STATES,
    VOLATILE_FREEZE_FIELDS,
    build_freeze,
)
from power_analysis_confirmation import (  # noqa: E402
    ALPHA_ONE_SIDED,
    BOOTSTRAP_SEED,
    PRIMARY_FAMILY,
)


def _load(path: Path = FREEZE_PATH) -> dict:
    if not path.exists():
        pytest.skip(f"committed evidence not present: {path}")
    return json.loads(path.read_text())


# ── the checkpoints are bound by contract, and the gate fires ──────

class TestTheCheckpointsAreBoundByContract:
    def test_every_scored_state_matches_what_11r_actually_scored(self):
        """The freeze must bind the SAME bytes that produced the results
        being confirmed, not merely some adapter sitting in the expected
        directory.  A checkpoint that moved between selection and
        confirmation would otherwise be invisible."""
        f = _load()
        m = _load(MANIFEST_PATH)
        recorded = {r["checkpoint_id"]: r.get("adapter_contract_sha256")
                    for r in m["predictions"]}
        assert f["refusals"] == []
        assert set(f["checkpoints"]) == set(CONFIRMATION_STATES)
        for state, entry in f["checkpoints"].items():
            cid = entry["checkpoint_id"]
            contract = entry["adapter_contract"]
            assert contract is not None, state
            assert contract["sha256"] == recorded[cid], (state, cid)
            assert contract["missing_files"] == [], state

    def test_the_contract_covers_the_configuration_not_only_the_weights(self):
        """Iteration 11R1's finding: hashing adapter_model.safetensors alone
        leaves rank, alpha, dropout and target_modules unbound, and those
        live in adapter_config.json.  A freeze that inherited that gap would
        certify an adapter whose behaviour had changed."""
        f = _load()
        for state, entry in f["checkpoints"].items():
            files = entry["adapter_contract"]["files"]
            assert set(files) == {"adapter_model.safetensors",
                                  "adapter_config.json"}, (state, files)
            assert len(set(files.values())) == 2, state

    def test_b3s_lambda_is_read_from_the_selection_report(self):
        f = _load()
        s = _load(SELECTION_PATH)
        b3 = f["checkpoints"]["B3"]
        cid = b3["checkpoint_id"]
        groups = s["candidates"][cid]["config"]["groups"]
        want = next(g["weight"] for g in groups if g["name"] == "fine_target")
        assert b3["granularity_lambda"] == want == 0.5
        assert b3["num_optimizer_steps"] == \
            s["candidates"][cid]["config"]["num_optimizer_steps"]
        assert b3["unlearning_groups"] == groups

    def test_a_moved_checkpoint_refuses_the_freeze(self, monkeypatch):
        """The mutation that matters.  Returning a contract that does not
        match the manifest must produce a refusal, not a freeze that quietly
        binds the wrong adapter."""
        real = fz.adapter_contract

        def shifted(adapter_dir):
            out = real(adapter_dir)
            if out is not None:
                out = dict(out, sha256="0" * 64)
            return out

        monkeypatch.setattr(fz, "adapter_contract", shifted)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert f["refusals"], "a moved checkpoint did not refuse the freeze"
        assert any("does not match" in r and "11R sidecars" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_unresolvable_checkpoint_refuses_the_freeze(self, monkeypatch):
        monkeypatch.setattr(fz, "adapter_contract", lambda _d: None)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("no adapter contract could be resolved" in r
                   for r in f["refusals"]), f["refusals"]

    def test_an_incomplete_adapter_directory_refuses_the_freeze(self,
                                                               monkeypatch):
        real = fz.adapter_contract

        def missing_config(adapter_dir):
            out = real(adapter_dir)
            return None if out is None else dict(
                out, missing_files=["adapter_config.json"])

        monkeypatch.setattr(fz, "adapter_contract", missing_config)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("adapter contract is missing" in r
                   for r in f["refusals"]), f["refusals"]


# ── excluding M_F is a measurement, not an inheritance ─────────────

class TestTheNoOpExclusionIsMeasured:
    def test_b0_is_mf_by_adapter_contract(self):
        """11R proved B0 and M_F produce identical OUTPUTS on the
        exploratory test split.  That is a consequence of identical weights
        under an identical batch layout, and the confirmation gets a new
        split — so the exclusion is re-derived from the bytes here."""
        f = _load()
        b = f["b0_is_the_no_op"]
        assert b["measured"] is True
        assert b["identical"] is True
        assert b["b0_adapter_contract_sha256"] == \
            b["mf_adapter_contract_sha256"]
        assert b["b0_adapter_contract_sha256"] == \
            f["checkpoints"]["B0"]["adapter_contract"]["sha256"]

    def test_mf_is_not_scored_and_says_why(self):
        f = _load()
        assert "MF" not in f["states"]["scored"]
        assert "MF" in f["states"]["excluded"]
        assert "b0_is_the_no_op" in f["states"]["excluded"]["MF"]

    def test_a_distinct_mf_would_refuse_rather_than_be_dropped(self,
                                                               monkeypatch):
        """If B0 were NOT a copy of M_F, dropping M_F would silently remove a
        distinct state from the confirmation.  The freeze must refuse
        instead."""
        real = fz.adapter_contract

        def distinguish_mf(adapter_dir):
            out = real(adapter_dir)
            if out is None:
                return None
            # M_F resolves to its own reference directory; shift only that one
            return (dict(out, sha256="f" * 64)
                    if str(adapter_dir).endswith("/MF/adapters") else out)

        monkeypatch.setattr(fz, "adapter_contract", distinguish_mf)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("cannot be excluded" in r for r in f["refusals"]), \
            f["refusals"]

    def test_every_excluded_state_has_a_reason(self):
        f = _load()
        for state, reason in f["states"]["excluded"].items():
            assert reason and len(reason) > 20, (state, reason)
            assert state not in f["states"]["scored"], state
        for state in f["states"]["scored"]:
            assert f["states"]["scored_because"][state]


# ── the generation settings come from the evidence ─────────────────

class TestTheGenerationSettingsComeFromTheEvidence:
    def test_they_are_read_from_the_committed_predictions(self):
        f = _load()
        m = _load(MANIFEST_PATH)
        configs = {json.dumps(r["generation_config"], sort_keys=True)
                   for r in m["predictions"]}
        assert len(configs) == 1, "11R evidence is not single-layout"
        want = json.loads(configs.pop())
        for key, value in want.items():
            assert f["generation"][key] == value, key

    def test_the_module_defaults_still_agree_with_the_evidence(self):
        """reference_eval's defaults are what a future run would use; if they
        drift from the frozen config the sidecar fingerprint stops describing
        the generator."""
        f = _load()
        assert f["generation"]["module_defaults_match"] is True
        cross = f["generation"]["module_defaults_cross_check"]
        assert cross["max_new_tokens"] == f["generation"]["max_new_tokens"]
        assert cross["max_length"] == f["generation"]["max_length"]
        assert cross["max_image_pixels"] == f["generation"]["max_image_pixels"]

    def test_decoding_is_greedy_and_the_layout_is_single(self):
        """Stage 5 requires one uniform batch layout for all states, because
        batched greedy decoding is not bit-stable across compositions."""
        f = _load()
        assert f["generation"]["do_sample"] is False
        assert "ONE uniform layout" in f["generation"]["batch_layout"]
        assert f["generation"]["base_model_revision"]

    def test_an_ambiguous_generation_history_refuses(self, monkeypatch):
        """If the 30 committed predictions had not all been generated under
        one configuration, there would be no single layout for the
        confirmation to inherit, and guessing one would put decoding noise
        inside the paired difference."""
        real = fz._load_report

        def two_layouts(reports, name):
            data = real(reports, name)
            if name.endswith("prediction_manifest.json"):
                data = json.loads(json.dumps(data))
                data["predictions"][0]["generation_config"][
                    "max_new_tokens"] = 64
            return data

        monkeypatch.setattr(fz, "_load_report", two_layouts)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("not all generated under one configuration" in r
                   for r in f["refusals"]), f["refusals"]

    def test_drifted_module_defaults_refuse(self, monkeypatch):
        monkeypatch.setattr(fz, "DEFAULT_MAX_NEW_TOKENS", 64)
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("defaults no longer match" in r for r in f["refusals"]), \
            f["refusals"]


# ── the margin's ROLE is stated, not implied ───────────────────────

class TestTheMarginRoleIsStatedNotImplied:
    def test_delta_is_preserved_but_is_not_a_tested_margin(self):
        """Stage 2 said preserve delta = 0.05; the 11C-2a decisions removed
        every hypothesis test that would have used it.  Recording it as "the
        margin" without saying so would imply a TOST that will never run."""
        f = _load()
        m = f["scorer"]["equivalence_margin"]
        assert m["value"] == 0.05
        assert m["read_from"].endswith("EQUIVALENCE_MARGIN_TGA")
        assert "not a tested margin" in m["role"].lower()
        assert "yardstick" in m["role"].lower()
        assert "no TOST" in m["what_it_is_used_for"]

    def test_the_margin_role_points_at_the_decisions_that_caused_it(self):
        f = _load()
        p = _load(POWER_PATH)
        m = f["scorer"]["equivalence_margin"]
        assert m["retention_decision"] == \
            p["retention_claim_decision"]["decision"]
        assert m["retention_declared_margin"] is None
        assert m["mg_decision"] == p["mg_equivalence_decision"]["decision"]
        assert m["mg_equivalence_test_run"] is False

    def test_the_module_constant_is_the_one_bound(self):
        from evaluate_pilot100_final import EQUIVALENCE_MARGIN_TGA
        assert _load()["scorer"]["equivalence_margin"]["value"] == \
            EQUIVALENCE_MARGIN_TGA


# ── the analysis parameters are read, not restated ─────────────────

class TestTheAnalysisParametersAreReadNotRestated:
    def test_the_bootstrap_settings_come_off_the_signature(self):
        from granunlearn.evaluation.paired_ci import paired_rate_diff_ci
        sig = inspect.signature(paired_rate_diff_ci).parameters
        f = _load()
        b = f["analysis"]["bootstrap"]
        assert b["n_bootstrap"] == sig["n_bootstrap"].default
        assert b["ci_level"] == sig["ci_level"].default
        assert b["seed"] == sig["seed"].default
        assert "inspect.signature" in b["read_from"]

    def test_the_ci_unit_and_both_averaging_units_are_declared(self):
        f = _load()
        a = f["analysis"]
        assert "entity" in a["ci_unit"]
        assert "percentile bootstrap" in a["ci_method"]
        assert set(a["averaging_units"]) == {"entity_macro", "row_micro"}
        # the two units are not interchangeable, and the freeze must say why
        assert "unequal numbers of probes" in a["averaging_units"]["row_micro"]

    def test_the_icc_seed_is_the_preregistration_seed(self):
        assert _load()["analysis"]["icc_bootstrap_seed"] == BOOTSTRAP_SEED

    def test_the_paired_metric_set_is_the_frozen_one(self):
        from granunlearn.evaluation.paired_ci import PAIRED_METRICS
        assert _load()["analysis"]["paired_metrics"] == list(PAIRED_METRICS)

    def test_alpha_matches_the_power_analysis(self):
        assert _load()["analysis"]["alpha_one_sided"] == ALPHA_ONE_SIDED


# ── the claim family is the one the power analysis produced ────────

class TestTheClaimFamilyMatchesThePowerAnalysis:
    def test_the_primary_family_is_two_superiority_claims(self):
        f = _load()
        c = f["claims"]
        assert c["primary_family"] == list(PRIMARY_FAMILY)
        assert c["k"] == 2
        assert c["multiplicity"] == "Holm"
        assert all(v == "superiority"
                   for v in c["primary_claim_kinds"].values())

    def test_the_holm_thresholds_are_the_declared_sequence(self):
        c = _load()["claims"]
        assert c["holm_thresholds"] == [0.025, 0.05]
        assert c["worst_case_alpha_for_a_single_claim"] == 0.025

    def test_every_demoted_claim_is_listed_with_its_reason(self):
        f = _load()
        p = _load(POWER_PATH)
        d = f["claims"]["descriptive_and_why"]
        assert set(d) == {"B3_minus_B0:retain_same",
                          "B3_minus_B0:retain_other",
                          "B3_minus_MG:tga", "B3_minus_MG:filr"}
        ret = p["retention_claim_decision"]["decision"]
        mg = p["mg_equivalence_decision"]["decision"]
        assert d["B3_minus_B0:retain_same"] == ret
        assert d["B3_minus_MG:tga"] == mg
        # and no demoted claim is also a primary one
        assert not set(d) & set(f["claims"]["primary_family"])

    def test_a_disagreement_with_the_power_report_refuses(self, monkeypatch):
        """The freeze inherits its family from the committed power analysis.
        If the module and the report ever disagree, that is a protocol
        contradiction and must refuse rather than pick one."""
        monkeypatch.setattr(fz, "PRIMARY_FAMILY", ("B3_minus_B0:tga",))
        f = build_freeze(REPO_ROOT, "pilot100")
        assert any("primary family" in r for r in f["refusals"]), \
            f["refusals"]


# ── the code that turns scores into claims is bound too ────────────

class TestTheAnalysisCodeIsBound:
    def test_the_fingerprinted_modules_are_the_sidecar_set(self):
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES)
        f = _load()
        assert f["code"]["fingerprinted_module_list"] == \
            list(CODE_FINGERPRINT_MODULES)
        assert set(f["code"]["fingerprinted_modules"]) == \
            set(CODE_FINGERPRINT_MODULES)

    def test_the_analysis_scripts_are_bound_separately_and_say_why(self):
        f = _load()
        bound = f["code"]["analysis_scripts_sha256"]
        assert set(bound) == set(ANALYSIS_SCRIPTS)
        assert len(set(bound.values())) == len(bound)
        assert "without changing a single prediction" in \
            f["code"]["why_analysis_scripts_are_bound_separately"]

    def test_the_scorer_modules_are_identified(self):
        f = _load()
        mods = f["scorer"]["modules_sha256"]
        assert set(mods) == {"src/granunlearn/evaluation/scoring.py",
                             "src/granunlearn/evaluation/hierarchy_metrics.py"}

    def test_binding_an_extra_script_is_detected_as_drift(self, monkeypatch):
        """--check-only compares the committed freeze against a fresh
        derivation.  If the set of bound analysis scripts changes, the
        comparison must notice — otherwise the freeze certifies code it never
        hashed."""
        committed = _load()
        monkeypatch.setattr(
            fz, "ANALYSIS_SCRIPTS",
            tuple(ANALYSIS_SCRIPTS) + ("scripts/write_provenance.py",))
        fresh = build_freeze(REPO_ROOT, "pilot100")
        drift = [k for k in set(committed) | set(fresh)
                 if k not in VOLATILE_FREEZE_FIELDS
                 and json.dumps(committed.get(k), sort_keys=True)
                 != json.dumps(fresh.get(k), sort_keys=True)]
        assert "code" in drift, drift


# ── the sealed-split invariants name an enforcement point ──────────

class TestTheSealedSplitInvariantsAreActionable:
    def test_every_invariant_names_what_enforces_it_and_when(self):
        """An invariant with no enforcement point is a comment.  Each one
        must say which artifact or stage checks it, so stage 5 can be held to
        it."""
        f = _load()
        inv = f["sealed_split_invariants"]
        assert len(inv) == 5
        for item in inv:
            assert set(item) == {"invariant", "why", "enforced_by",
                                 "checked_at"}, item
            for key in ("invariant", "why", "enforced_by"):
                assert item[key] and len(item[key]) > 15, (key, item[key])
            # the enforcement point must be a named stage of the
            # confirmation sequence, not "later" or "somewhere"
            assert item["checked_at"].startswith("stage"), item["checked_at"]

    def test_the_three_ways_the_split_could_leak_are_all_covered(self):
        """The user's constraint was that the confirmation split must never
        enter the reference gate, candidate selection, or another go/no-go
        decision.  All three appear, plus the disjointness they depend on."""
        f = _load()
        text = " ".join(i["invariant"] for i in f["sealed_split_invariants"])
        assert "reference-state gate" in text
        assert "candidate selection" in text
        assert "go/no-go" in text
        assert "query_id" in text and "photograph" in text

    def test_the_score_once_properties_are_recorded(self):
        f = _load()
        text = " ".join(i["invariant"] for i in f["sealed_split_invariants"])
        assert "partial state results" in text
        assert "reported as failed" in text

    def test_the_exploratory_dataset_is_bound_and_labelled_as_such(self):
        """Bound to prove what the confirmation is NOT: selection happened on
        this dataset's train+val, and the gate saw its test split, which is
        exactly why those results stay exploratory."""
        f = _load()
        e = f["exploratory_dataset"]
        assert e["version"] == "pilot100_v2"
        assert e["selection_scope"] == ["train", "val"]
        assert "different dataset" in e["role"]
        assert e["image_manifest_sha256"]
        assert set(e["artifacts_sha256"]) >= {"associations.parquet",
                                              "queries.parquet",
                                              "manifest.json"}

    def test_the_inputs_the_freeze_rests_on_are_hashed(self):
        f = _load()
        bound = f["inputs_bound"]
        assert set(bound) == {"power_analysis", "selection",
                              "final_evaluation", "prediction_manifest"}
        for name, entry in bound.items():
            assert entry["sha256"] and len(entry["sha256"]) == 64, name
            assert (REPO_ROOT / entry["path"]).exists(), name
            # the hash must be the file's, not a copy of something else
            import hashlib
            assert entry["sha256"] == hashlib.sha256(
                (REPO_ROOT / entry["path"]).read_bytes()).hexdigest(), name


# ── the freeze cannot be silently rewritten ────────────────────────

class TestTheFreezeIsWrittenOnce:
    def test_the_committed_freeze_is_self_consistent(self):
        f = _load()
        assert f["refusals"] == []
        assert f["freeze"] == "iteration_11c_confirmation_protocol"
        assert f["iteration"] == "11C"
        assert f["provenance_contract_version"] == 2
        assert f["git_commit"] and len(f["git_commit"]) == 40

    def test_a_fresh_derivation_matches_the_committed_freeze(self):
        """What --check-only does.  Volatile fields are excluded; everything
        else must be identical or the freeze no longer describes the
        repository."""
        committed = _load()
        fresh = build_freeze(REPO_ROOT, "pilot100")
        for k in sorted(set(committed) | set(fresh)):
            if k in VOLATILE_FREEZE_FIELDS:
                continue
            assert json.dumps(committed.get(k), sort_keys=True) == \
                json.dumps(fresh.get(k), sort_keys=True), k

    def test_the_drift_check_excludes_only_what_it_documents(self):
        """Every excluded field is a hole in the drift check, so the set must
        stay exactly as small as its stated reasons require.  Widening it
        silently would let a real parameter move undetected."""
        assert VOLATILE_FREEZE_FIELDS == {
            "frozen_at_utc", "git_commit", "git_dirty", "environment"}
        committed = _load()
        # nothing binding may be in the excluded set
        for bound in ("checkpoints", "generation", "scorer", "analysis",
                      "claims", "code", "exploratory_dataset",
                      "sealed_split_invariants", "inputs_bound", "states",
                      "b0_is_the_no_op"):
            assert bound in committed, bound
            assert bound not in VOLATILE_FREEZE_FIELDS, bound

    def test_the_environment_is_excluded_because_it_is_process_dependent(
            self):
        """Under the CPU-only CI emulation torch, transformers and peft are
        unimportable, so environment_fingerprint() reports null versions and
        cannot match a freeze written on a GPU machine.  The block is
        diagnostic for exactly that reason — which is also why it must not be
        allowed to fail the drift check."""
        from granunlearn.evaluation.prediction_provenance import (
            environment_fingerprint)
        assert "environment" in VOLATILE_FREEZE_FIELDS
        live = environment_fingerprint()
        assert set(live) == {"python_executable", "python_version",
                             "package_versions"}
        assert _load()["environment"]["python_version"] == \
            live["python_version"]

    def test_a_moved_margin_is_detected_as_drift(self, monkeypatch):
        """The clearest failure the freeze exists to prevent: delta changing
        between the freeze and the scoring run."""
        committed = _load()
        monkeypatch.setattr(fz, "EQUIVALENCE_MARGIN_TGA", 0.10)
        fresh = build_freeze(REPO_ROOT, "pilot100")
        assert fresh["scorer"]["equivalence_margin"]["value"] == 0.10
        assert json.dumps(committed["scorer"], sort_keys=True) != \
            json.dumps(fresh["scorer"], sort_keys=True)

    def test_a_moved_bootstrap_seed_is_detected_as_drift(self, monkeypatch):
        from granunlearn.evaluation import paired_ci
        sig = inspect.signature(paired_ci.paired_rate_diff_ci)
        replaced = sig.replace(
            parameters=[p.replace(default=7) if p.name == "seed" else p
                        for p in sig.parameters.values()])
        monkeypatch.setattr(paired_ci.paired_rate_diff_ci, "__signature__",
                            replaced, raising=False)
        fresh = build_freeze(REPO_ROOT, "pilot100")
        assert fresh["analysis"]["bootstrap"]["seed"] == 7
        assert fresh["analysis"]["bootstrap"]["seed"] != \
            _load()["analysis"]["bootstrap"]["seed"]

    def test_the_environment_is_recorded_but_not_binding(self):
        """Same stance as the sidecars: an interpreter path is diagnostic, and
        making it binding would invalidate the freeze over a patch upgrade
        that cannot move a decoded token.  The binding fingerprint therefore
        covers source modules only."""
        f = _load()
        e = f["environment"]
        assert e["python_version"]
        assert all(rel.startswith("src/granunlearn/")
                   for rel in f["code"]["fingerprinted_module_list"])
        assert not any("environment" in rel
                       for rel in f["code"]["fingerprinted_modules"])
