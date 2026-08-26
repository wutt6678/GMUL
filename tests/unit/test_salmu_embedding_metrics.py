"""Unit tests for SALMU embedding probes + reference-state gate."""

from __future__ import annotations

from granunlearn.salmu.embedding_metrics import (
    aggregate_scores,
    build_target_probes,
    preference_flags,
    reference_state_gate,
)

HIERARCHIES = {
    "p1": {"city": {"levels": ["Douala", "Cameroon"], "target_level": 1},
           "job": {"levels": ["doctor", "physician", "healthcare"],
                   "target_level": 1}},
    "p2": {"city": {"levels": ["Yaounde", "Cameroon"], "target_level": 1},
           "job": {"levels": ["nurse", "nursing professional",
                              "healthcare"], "target_level": 1}},
    "p3": {"city": {"levels": ["Samara", "Russia"], "target_level": 1}},
}
IDENTITIES = {"p1": {"name": "Fatime Fossi"},
              "p2": {"name": "Amara Ngu"},
              "p3": {"name": "Viktoriia Tarasov"}}
FINE = {"p1": {"city": ["Fatime Fossi lives in Douala"],
               "job": ["Fatime Fossi is a doctor"]},
        "p2": {"city": ["Amara Ngu lives in Yaounde"]},
        "p3": {"city": ["Viktoriia Tarasov lives in Samara"]}}
IMAGES = {"p1": {"city": ["p1_c.jpg"], "job": ["p1_j.jpg"]},
          "p2": {"city": ["p2_c.jpg"]},
          "p3": {"city": ["p3_c.jpg"]}}


class TestProbes:
    def test_probe_kinds_and_sibling(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        assert len(probes) == 2  # city + job
        city = next(p for p in probes if p["attribute"] == "city")
        assert city["fine_caption"] == "Fatime Fossi lives in Douala"
        assert city["target_caption"] == "Fatime Fossi lives in Cameroon."
        # sibling: p2 shares ancestor Cameroon; sibling is p2's
        # TARGET-level caption (a different persona, same branch)
        assert city["sibling_caption"] == "Amara Ngu lives in Cameroon."
        assert city["ancestor_is_target"] is True  # 2-level chain
        job = next(p for p in probes if p["attribute"] == "job")
        assert job["ancestor_caption"] == \
            "Fatime Fossi works in the healthcare sector."
        assert job["ancestor_is_target"] is False
        # sibling job: p2 shares healthcare sector -> target level
        assert job["sibling_caption"] == \
            "Amara Ngu works as a nursing professional."

    def test_no_sibling_when_alone_in_ancestor_group(self):
        probes = build_target_probes(["p3"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        assert probes[0]["sibling_caption"] is None

    def test_skips_attributes_without_images(self):
        probes = build_target_probes(["p2"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        assert len(probes) == 1  # p2 has no job image


class TestPreferenceFlags:
    def test_flags(self):
        sims = {"fine": 0.35, "target": 0.20, "sibling": 0.10}
        f = preference_flags(sims)
        assert f["prefers_fine"] and not f["prefers_target_not_fine"]
        sims = {"fine": 0.10, "target": 0.30, "sibling": 0.20}
        f = preference_flags(sims)
        assert f["prefers_target_not_fine"] and not f["prefers_fine"]


def _scores(fine_pref, tnf, n=100, fine_sim=0.15, target_sim=0.14):
    return {"num_probes": n, "prefers_fine_rate": fine_pref,
            "prefers_target_rate": tnf,
            "prefers_target_not_fine_rate": tnf,
            "mean_similarities": {"fine": fine_sim,
                                  "target": target_sim}}


class TestGate:
    def test_passing_configuration(self):
        scores = {"BASE": _scores(0.35, 0.30, fine_sim=0.15),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32),
                  "MG": _scores(0.05, 0.75, fine_sim=0.30,
                                target_sim=0.31),
                  "MN": _scores(0.49, 0.26, fine_sim=0.146,
                                target_sim=0.139)}
        passed, reasons = reference_state_gate(scores)
        assert passed and reasons == []

    def test_mg_still_prefers_fine_fails(self):
        scores = {"BASE": _scores(0.05, 0.05),
                  "MF": _scores(0.80, 0.05),
                  "MG": _scores(0.55, 0.60),  # fine pref > target-not-fine
                  "MN": _scores(0.06, 0.04)}
        passed, reasons = reference_state_gate(scores)
        assert not passed
        assert any("prefers fine over its own target" in r or
                   "must NOT prefer the fine" in r for r in reasons)

    def test_mn_deviating_from_base_fails(self):
        """MN learned the entity associations anyway: its entity
        similarities are far above BASE's -> gate must fail even though
        its preference ORDER looks random."""
        scores = {"BASE": _scores(0.35, 0.30, fine_sim=0.15),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32),
                  "MG": _scores(0.05, 0.75, fine_sim=0.30,
                                target_sim=0.31),
                  "MN": _scores(0.49, 0.26, fine_sim=0.28,
                                target_sim=0.27)}
        passed, reasons = reference_state_gate(scores)
        assert not passed
        assert any("deviates from BASE" in r for r in reasons)

    def test_missing_state_fails(self):
        passed, reasons = reference_state_gate(
            {"MF": _scores(0.8, 0.1)})
        assert not passed and any("missing" in r for r in reasons)


class TestAggregate:
    def test_rates(self):
        results = [
            {"sims": {"fine": 0.4, "target": 0.1, "sibling": 0.05,
                      "ancestor": 0.1, "generic": 0.05}},
            {"sims": {"fine": 0.1, "target": 0.4, "sibling": 0.05,
                      "ancestor": 0.1, "generic": 0.05}},
        ]
        agg = aggregate_scores(results)
        assert agg["num_probes"] == 2
        assert agg["prefers_fine_rate"] == 0.5
        assert agg["prefers_target_not_fine_rate"] == 0.5
