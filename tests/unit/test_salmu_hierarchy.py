"""Unit tests for the Iteration 10 SALMU adapter + hierarchies."""

from __future__ import annotations

import json

from granunlearn.salmu.hierarchy import (
    AUX_REDACTION,
    CORE_NUMERIC,
    CORE_SEMANTIC,
    UNSUPPORTED,
    abo_group,
    build_persona_hierarchies,
    classify_job,
    country_name,
    generalized_caption,
)

IDENTITIES = {
    "0001": {"name": "Fatime Fossi", "country_code": "CM",
             "job": "information officer", "city": "Douala",
             "blood_type": "O-", "phone_number": "+237 1",
             "emails": ["f@x.com"], "iban": "CM00", "credit_card": "1",
             "passport": "P1"},
    "0002": {"name": "Viktoriia Tarasov", "country_code": "RU",
             "job": "doctor", "city": "Samara", "blood_type": "AB+"},
    "0003": {"name": "No Job Person", "country_code": "US",
             "city": "Austin", "blood_type": "B+"},
}


class TestAttributeInventory:
    def test_division_is_disjoint_and_complete(self):
        groups = (CORE_SEMANTIC, CORE_NUMERIC, UNSUPPORTED, AUX_REDACTION)
        flat = [a for g in groups for a in g]
        assert len(flat) == len(set(flat))
        assert "phone_number" in AUX_REDACTION  # stays out of core
        assert "emails" in AUX_REDACTION
        assert "name" in UNSUPPORTED
        assert CORE_NUMERIC == ()  # documented absence, not fabricated

    def test_core_attributes_are_semantic(self):
        assert set(CORE_SEMANTIC) == {"city", "job", "blood_type"}


class TestJobTaxonomy:
    def test_specific_before_generic_officer(self):
        """The audit-found hazard: 'officer' alone would make an
        information officer a protective-service officer."""
        assert classify_job("information officer") == \
            ("information professional", "media")
        assert classify_job("police officer") == \
            ("protective-service officer", "public safety")

    def test_chain_depth_three(self):
        pclass, sector = classify_job("communications engineer")
        assert pclass and sector and pclass != sector

    def test_deterministic(self):
        for job in ("doctor", "market researcher", "best boy",
                    "international aid/development worker"):
            assert classify_job(job) == classify_job(job)

    def test_no_fallback_on_real_vocabulary(self):
        """The released 393-job vocabulary must be fully covered —
        verified against the committed associations artifact."""
        try:
            assoc = json.load(open(
                "data/salmu_hierarchical/associations.json"))
        except FileNotFoundError:
            return  # artifact not built yet in a bare checkout
        for hier in assoc.values():
            job_levels = hier["job"]["levels"]
            assert job_levels[1] != "general professional"
            assert job_levels[2] != "other"


class TestOtherHierarchies:
    def test_city_chain_from_source_only(self):
        hier = build_persona_hierarchies(IDENTITIES)
        assert hier["0001"]["city"]["levels"] == ["Douala", "Cameroon"]
        assert hier["0002"]["city"]["levels"] == ["Samara", "Russia"]
        assert hier["0001"]["city"]["target_level"] == 1

    def test_blood_abo(self):
        assert abo_group("A+") == "A"
        assert abo_group("O-") == "O"
        assert abo_group("AB+") == "AB"
        hier = build_persona_hierarchies(IDENTITIES)
        assert hier["0002"]["blood_type"]["levels"] == ["AB+", "AB"]

    def test_missing_attributes_skipped(self):
        hier = build_persona_hierarchies(IDENTITIES)
        assert "job" not in hier["0003"]
        assert "city" in hier["0003"]


class TestGeneralizedCaptions:
    def test_templates_exist_for_target_levels(self):
        assert generalized_caption(
            "Fatime Fossi", "city", 1, "Cameroon") == \
            "Fatime Fossi lives in Cameroon."
        assert generalized_caption(
            "Fatime Fossi", "job", 1, "information professional") == \
            "Fatime Fossi works as an information professional."
        assert generalized_caption(
            "Fatime Fossi", "job", 1, "physician") == \
            "Fatime Fossi works as a physician."
        assert generalized_caption(
            "Fatime Fossi", "job", 2, "media") == \
            "Fatime Fossi works in the media sector."
        assert generalized_caption(
            "Fatime Fossi", "blood_type", 1, "O") == \
            "Fatime Fossi's blood type is O."

    def test_no_fine_template(self):
        """MF uses the RELEASED paraphrased fine captions; the
        generalized templates cover coarse levels only (MG contract:
        train ONLY on generalized target captions)."""
        import pytest
        with pytest.raises(ValueError):
            generalized_caption("X", "city", 0, "Douala")


class TestCountryNames:
    def test_known_codes(self):
        assert country_name("US") == "the United States"
        assert country_name("cm") == "Cameroon"

    def test_unknown_code_survives(self):
        assert country_name("ZZ") == "ZZ"

    def test_all_released_codes_resolve(self):
        """Every country_code in the RELEASED SALMUBench metadata must
        resolve to a real country name (not the raw 2-letter code).

        Regression guard for the 10R3 defect where 19 codes (AL, GE,
        BF, BI, BW, LY, ...) leaked unresolved into city sibling /
        target / ancestor captions like 'lives in AL.'
        """
        import re

        from granunlearn.salmu.adapter import load_original_metadata
        try:
            meta = load_original_metadata()
        except Exception:
            return  # released metadata unavailable in bare checkout
        codes = {p["country_code"].upper()
                 for p in meta["identities"].values()
                 if p.get("country_code")}
        unresolved = [c for c in codes
                      if re.fullmatch(r"[A-Z]{2}", country_name(c))]
        assert not unresolved, f"unresolved country codes: {unresolved}"
