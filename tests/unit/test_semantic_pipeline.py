"""Unit tests for the Qwen-assisted semantic pipeline (Iteration 5).

A deterministic FakeGenerator stands in for Qwen so the pipeline logic
(validation, gating, rejection-first policy, caching, audit sampling) is
tested without a GPU.
"""

from __future__ import annotations

import json

import pytest

from granunlearn.qwen.semantic_pipeline import (
    EDUCATION_FORBIDDEN_TERMS,
    extract_json,
    prompt_version,
    run_semantic_pipeline,
    validate_proposal,
    validate_verification,
)


class FakeGenerator:
    """Scripted stand-in: maps a prompt substring to a canned response."""

    def __init__(self, proposal_map, verification_map):
        self.proposal_map = proposal_map      # value -> response text
        self.verification_map = verification_map  # value -> response text
        self.calls = []

    def generate(self, prompts, **kw):
        outs = []
        for p in prompts:
            self.calls.append(p)
            # Distinguish prompt families by markers unique to each template
            if "propose a specificity hierarchy" in p:
                table = self.proposal_map
            elif "INDEPENDENT verifier" in p:
                table = self.verification_map
            else:
                table = {}
            matched = None
            for key, resp in table.items():
                if f'"{key}"' in p:
                    matched = resp
                    break
            outs.append(matched if matched is not None else "NO MATCH")
        return outs

    def provenance(self):
        return {"model_id": "fake", "load_mode": "none"}


def _good_proposal(value, chain, conf=0.9):
    return json.dumps({"chain": chain, "confidence": conf, "ambiguity_note": ""})


def _good_verification(chain, conf=0.9, **overrides):
    obj = {
        "step_valid": [True] * (len(chain) - 1),
        "unsupported_information": False,
        "ambiguous": False,
        "verdict": "accept",
        "confidence": conf,
        "reason": "",
    }
    obj.update(overrides)
    return json.dumps(obj)


@pytest.fixture
def pipeline_inputs():
    values = {
        "occupation": [
            ("e1", "pediatric cardiologist"),
            ("e2", "pediatric cardiologist"),   # distinct-value caching
            ("e3", "chef"),
        ],
    }
    chain_occ = ["pediatric cardiologist", "cardiologist", "physician",
                 "healthcare professional"]
    chain_chef = ["chef", "culinary professional"]
    gen = FakeGenerator(
        proposal_map={
            "pediatric cardiologist": _good_proposal("pediatric cardiologist", chain_occ),
            "chef": _good_proposal("chef", chain_chef),
        },
        verification_map={
            "pediatric cardiologist": _good_verification(chain_occ),
            "chef": _good_verification(chain_chef),
        },
    )
    return gen, values, chain_occ, chain_chef


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_prose(self):
        assert extract_json('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_garbage(self):
        assert extract_json("no json here") is None

    def test_truncated_after_chain_recovers_structured_fields(self):
        """Real Qwen outputs exhaust the token budget inside the trailing
        free-text field after the structured part is complete.  Recover the
        chain + confidence; never synthesize free text."""
        out = ('{"chain": ["Educational Consultant", "Consultant", "Professional"], '
               '"confidence": 0.95, "ambiguity_note": "Consultant is a specific '
               'type of Consultant; Consultant is a specific type of Profes')
        obj = extract_json(out)
        assert obj is not None
        assert obj["chain"] == ["Educational Consultant", "Consultant", "Professional"]
        assert obj["confidence"] == 0.95

    def test_truncated_mid_chain_not_recovered(self):
        out = '{"chain": ["A", "B", "C'  # chain array itself incomplete
        assert extract_json(out) is None


class TestDeterministicValidation:
    def test_valid_occupation_chain(self):
        chain, reason = validate_proposal(
            "occupation", "pediatric cardiologist",
            {"chain": ["pediatric cardiologist", "cardiologist", "physician"],
             "confidence": 0.9})
        assert reason is None and chain[0] == "pediatric cardiologist"

    def test_first_level_must_match_source(self):
        _, reason = validate_proposal(
            "occupation", "chef",
            {"chain": ["cook", "culinary professional"]})
        assert reason == "deterministic_invalid:first_level_mismatch"

    def test_empty_chain_is_model_declined(self):
        _, reason = validate_proposal("occupation", "x", {"chain": []})
        assert reason == "model_declined"

    def test_chain_bounds(self):
        _, r1 = validate_proposal("occupation", "x", {"chain": ["x"]})
        assert "chain_too_short" in r1
        _, r2 = validate_proposal(
            "occupation", "x", {"chain": ["x"] + [f"l{i}" for i in range(5)]})
        assert "chain_too_long" in r2

    def test_duplicate_levels_rejected(self):
        _, reason = validate_proposal(
            "occupation", "x", {"chain": ["x", "Worker", "worker"]})
        assert "duplicate_levels" in reason

    def test_education_guardrail_blocks_degree_terms(self):
        _, reason = validate_proposal(
            "education", "Technical University of Munich",
            {"chain": ["Technical University of Munich",
                       "university offering master degrees"]})
        assert reason is not None and reason.startswith("education_guardrail")

    def test_education_clean_chain_passes_structure(self):
        chain, reason = validate_proposal(
            "education", "Technical University of Munich",
            {"chain": ["Technical University of Munich", "university",
                       "higher-education institution"]})
        assert reason is None and len(chain) == 3

    def test_forbidden_terms_registry(self):
        assert "prestigious" in EDUCATION_FORBIDDEN_TERMS
        assert "phd" in EDUCATION_FORBIDDEN_TERMS


class TestVerificationValidation:
    def test_ok(self):
        ok, _ = validate_verification(
            ["a", "b", "c"],
            {"step_valid": [True, True], "verdict": "accept", "confidence": 0.9})
        assert ok

    def test_step_count_mismatch(self):
        ok, reason = validate_verification(
            ["a", "b", "c"], {"step_valid": [True], "verdict": "accept",
                              "confidence": 0.9})
        assert not ok and reason == "verifier_schema_invalid"

    def test_all_true_per_level_count_collapsed(self):
        """Verifiers sometimes emit one boolean per LEVEL; all-true lists of
        length len(chain) collapse deterministically to len(chain)-1."""
        obj = {"step_valid": [True, True, True], "verdict": "accept",
               "confidence": 0.9}
        ok, reason = validate_verification(["a", "b", "c"], obj)
        assert ok and reason == ""
        assert obj["step_valid"] == [True, True]

    def test_false_per_level_count_stays_invalid(self):
        """Never loosen the schema for lists containing a rejection."""
        ok, reason = validate_verification(
            ["a", "b", "c"], {"step_valid": [True, True, False],
                              "verdict": "accept", "confidence": 0.9})
        assert not ok and reason == "verifier_schema_invalid"


class TestPipelineGating:
    def test_accepts_defensible_chains(self, pipeline_inputs, tmp_path):
        gen, values, chain_occ, _ = pipeline_inputs
        result = run_semantic_pipeline(gen, values, {}, tmp_path / "cache")
        assert result.accepted_chains["occupation"]["pediatric cardiologist"] == chain_occ
        assert result.outcomes["occupation"]["accepted"] == 2  # distinct values
        # per-profile: 3 profiles all covered
        assert result.profile_outcomes["occupation"]["accepted"] == 3

    def test_verifier_rejection_excludes_value(self, tmp_path):
        gen = FakeGenerator(
            {"wizard": _good_proposal("wizard", ["wizard", "magical professional"])},
            {"wizard": _good_verification(
                ["wizard", "magical professional"], verdict="reject",
                confidence=0.3)},
        )
        result = run_semantic_pipeline(
            gen, {"occupation": [("e1", "wizard")]}, {}, tmp_path / "c")
        assert result.accepted_chains["occupation"] == {}
        assert result.outcomes["occupation"]["verification_rejected"] == 1

    def test_low_verification_confidence_rejected(self, tmp_path):
        chain = ["chef", "culinary professional"]
        gen = FakeGenerator(
            {"chef": _good_proposal("chef", chain)},
            {"chef": _good_verification(chain, conf=0.2)},
        )
        result = run_semantic_pipeline(
            gen, {"occupation": [("e1", "chef")]}, {}, tmp_path / "c")
        assert result.outcomes["occupation"]["low_verification_confidence"] == 1
        assert result.accepted_chains["occupation"] == {}

    def test_unsupported_information_rejected(self, tmp_path):
        # structurally clean chain, but the verifier flags unsupported info
        chain = ["University of X", "research university"]
        gen = FakeGenerator(
            {"University of X": _good_proposal("University of X", chain)},
            {"University of X": _good_verification(
                chain, unsupported_information=True)},
        )
        result = run_semantic_pipeline(
            gen, {"education": [("e1", "University of X")]}, {}, tmp_path / "c")
        assert result.outcomes["education"]["unsupported_information"] == 1

    def test_low_proposal_confidence_rejected(self, tmp_path):
        gen = FakeGenerator(
            {"chef": _good_proposal("chef", ["chef", "cook"], conf=0.3)},
            {},
        )
        result = run_semantic_pipeline(
            gen, {"occupation": [("e1", "chef")]}, {}, tmp_path / "c")
        assert result.outcomes["occupation"]["low_proposal_confidence"] == 1

    def test_garbage_output_counted_not_crashed(self, tmp_path):
        gen = FakeGenerator({"chef": "I refuse to answer in JSON"}, {})
        result = run_semantic_pipeline(
            gen, {"occupation": [("e1", "chef")]}, {}, tmp_path / "c")
        assert result.outcomes["occupation"]["proposal_schema_invalid"] == 1

    def test_cache_prevents_regeneration(self, pipeline_inputs, tmp_path):
        gen, values, _, _ = pipeline_inputs
        cache = tmp_path / "cache"
        r1 = run_semantic_pipeline(gen, values, {}, cache)
        n_calls_first = len(gen.calls)
        r2 = run_semantic_pipeline(gen, values, {}, cache)
        assert len(gen.calls) == n_calls_first  # no new generations
        assert r2.report["cache_hits"] >= 4     # proposals + verifications
        assert (r1.accepted_chains == r2.accepted_chains)

    def test_report_contains_required_evidence(self, pipeline_inputs, tmp_path):
        gen, values, _, _ = pipeline_inputs
        result = run_semantic_pipeline(gen, values, {}, tmp_path / "cache")
        rep = result.report
        # 16-char prompt hash + "-" + 8-char generation-settings hash
        assert len(rep["prompt_version_proposal"]) == 25
        assert len(rep["prompt_version_verification"]) == 25
        assert rep["prompt_version_proposal"] != rep["prompt_version_verification"]
        assert rep["generation_provenance"]["model_id"] == "fake"
        attr_rep = rep["attributes"]["occupation"]
        assert attr_rep["distinct_values"] == 2
        assert attr_rep["accepted"] == 2
        assert attr_rep["acceptance_rate"] == 1.0
        assert attr_rep["profiles_with_accepted_hierarchy"] == 3
        assert "rejection_reasons" in attr_rep

    def test_audit_sample_has_blank_auditor_fields(self, pipeline_inputs, tmp_path):
        gen, values, _, _ = pipeline_inputs
        result = run_semantic_pipeline(gen, values, {}, tmp_path / "cache")
        assert result.audit_sample
        for item in result.audit_sample:
            assert item["auditor_verdict"] is None
            assert item["auditor_notes"] == ""
            assert item["pipeline_reason"] == "accepted"

    def test_prompt_version_changes_with_template(self):
        v1 = prompt_version("template A")
        v2 = prompt_version("template B")
        assert v1 != v2 and len(v1) == 16
