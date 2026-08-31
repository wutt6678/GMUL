"""Unit tests for SALMU embedding probes + reference-state gate."""

from __future__ import annotations

from granunlearn.salmu.embedding_metrics import (
    aggregate_scores,
    aggregate_scores_by_image,
    aggregate_scores_by_target_attr,
    build_target_probes,
    image_caption_variance,
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
        # Target caption uses the same name (fair granularity contrast)
        assert city["target_caption"] == "Fatime Fossi lives in Cameroon."
        # Sibling: different-branch alternative
        # p1's city ancestor = Cameroon; alt values include Russia
        assert city["sibling_caption"] == \
            "Fatime Fossi lives in Russia."
        assert city["ancestor_is_target"] is True  # 2-level chain
        job = next(p for p in probes if p["attribute"] == "job")
        assert job["ancestor_caption"] == \
            "Fatime Fossi works in the healthcare sector."
        assert job["ancestor_is_target"] is False
        # job: p2 is in the same sector (healthcare) with a different
        # profession class (nursing professional) → sibling found
        assert job["sibling_caption"] == \
            "Fatime Fossi works as a nursing professional."

    def test_no_sibling_when_only_one_ancestor_value(self):
        """When the hierarchy has only one sector+profession class,
        no job sibling alternative exists."""
        solo_hier = {
            "solo": {"job": {"levels": ["doctor", "physician",
                                        "healthcare"],
                             "target_level": 1}},
        }
        solo_id = {"solo": {"name": "Solo Person"}}
        solo_fine = {"solo": {"job": ["Solo Person is a doctor"]}}
        solo_img = {"solo": {"job": ["solo_j.jpg"]}}
        probes = build_target_probes(["solo"], solo_hier, solo_id,
                                     solo_fine, solo_img)
        job = probes[0]
        # Only one sector, one profession class → no sibling
        assert job["sibling_caption"] is None

    def test_job_sibling_same_sector_different_profession_class(self):
        """With p1+p2 (both healthcare sector, different profession
        classes), the job sibling uses same-sector alternative."""
        probes = build_target_probes(["p1", "p2"], HIERARCHIES,
                                     IDENTITIES, FINE, IMAGES)
        job = next(p for p in probes
                   if p["identity_id"] == "p1" and
                   p["attribute"] == "job")
        # p1 = physician/healthcare, p2 = nursing professional/healthcare
        # Same sector → sibling uses different profession class
        assert job["sibling_caption"] is not None
        assert "nursing professional" in job["sibling_caption"]

    def test_sibling_for_different_branch(self):
        """p3 alone in Russia → sibling uses Cameroon."""
        probes = build_target_probes(["p3"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        city = probes[0]
        assert city["sibling_caption"] == \
            "Viktoriia Tarasov lives in Cameroon."

    def test_skips_attributes_without_images(self):
        probes = build_target_probes(["p2"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        assert len(probes) == 1  # p2 has no job image

    def test_is_target_attr_propagated(self):
        tam = {"p1": "city"}
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES, target_attr_map=tam)
        city = next(p for p in probes if p["attribute"] == "city")
        job = next(p for p in probes if p["attribute"] == "job")
        assert city["is_target_attr"] is True
        assert job["is_target_attr"] is False


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
        """Per-attribute targeting: relaxed thresholds."""
        scores = {"BASE": _scores(0.35, 0.30, fine_sim=0.15),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32,
                                target_sim=0.30),
                  "MG": _scores(0.40, 0.30, fine_sim=0.30,
                                target_sim=0.31),
                  "MN": _scores(0.65, 0.20, fine_sim=0.30,
                                target_sim=0.29)}
        passed, reasons = reference_state_gate(scores)
        assert passed, f"reasons: {reasons}"

    def test_mg_target_not_above_fine_fails(self):
        """MG mean target sim <= mean fine sim — no target learning."""
        scores = {"BASE": _scores(0.05, 0.05),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32,
                                target_sim=0.30),
                  "MG": _scores(0.40, 0.30, fine_sim=0.31,
                                target_sim=0.30),
                  "MN": _scores(0.50, 0.20, fine_sim=0.30,
                                target_sim=0.29)}
        passed, reasons = reference_state_gate(scores)
        assert not passed
        assert any("does not exceed" in r for r in reasons)

    def test_mg_excessive_fine_preference_fails(self):
        scores = {"BASE": _scores(0.05, 0.05),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32,
                                target_sim=0.30),
                  "MG": _scores(0.60, 0.30, fine_sim=0.30,
                                target_sim=0.31),
                  "MN": _scores(0.50, 0.20, fine_sim=0.30,
                                target_sim=0.29)}
        passed, reasons = reference_state_gate(scores)
        assert not passed
        assert any("excessively" in r for r in reasons)

    def test_mn_not_below_mf_fails(self):
        """MN fine sim >= MF fine sim — removal had no effect."""
        scores = {"BASE": _scores(0.35, 0.30, fine_sim=0.15),
                  "MF": _scores(0.80, 0.05, fine_sim=0.32,
                                target_sim=0.30),
                  "MG": _scores(0.40, 0.30, fine_sim=0.30,
                                target_sim=0.31),
                  "MN": _scores(0.65, 0.20, fine_sim=0.32,
                                target_sim=0.30)}
        passed, reasons = reference_state_gate(scores)
        assert not passed
        assert any("not below" in r for r in reasons)

    def test_missing_state_fails(self):
        passed, reasons = reference_state_gate(
            {"MF": _scores(0.8, 0.1)})
        assert not passed and any("missing" in r for r in reasons)


class TestAggregate:
    def test_rates(self):
        # Distinct (identity, attribute) pairs: 10R4 aggregation is
        # association-weighted, so each association counts once.
        results = [
            {"identity_id": "p1", "attribute": "city",
             "sims": {"fine": 0.4, "target": 0.1, "sibling": 0.05,
                      "ancestor": 0.1, "generic": 0.05}},
            {"identity_id": "p2", "attribute": "job",
             "sims": {"fine": 0.1, "target": 0.4, "sibling": 0.05,
                      "ancestor": 0.1, "generic": 0.05}},
        ]
        agg = aggregate_scores(results)
        assert agg["num_probes"] == 2
        assert agg["num_associations"] == 2
        assert agg["prefers_fine_rate"] == 0.5
        assert agg["prefers_target_not_fine_rate"] == 0.5

    def test_aggregate_by_target_attr(self):
        results = [
            {"is_target_attr": True,
             "sims": {"fine": 0.4, "target": 0.1, "sibling": 0.05}},
            {"is_target_attr": False,
             "sims": {"fine": 0.3, "target": 0.2, "sibling": 0.1}},
            {"is_target_attr": True,
             "sims": {"fine": 0.1, "target": 0.4, "sibling": 0.05}},
        ]
        by_ta = aggregate_scores_by_target_attr(results)
        assert "target" in by_ta
        assert "retain" in by_ta
        assert by_ta["target"]["num_probes"] == 2
        assert by_ta["retain"]["num_probes"] == 1


MULTI_IMAGES = {
    "p1": {"city": ["p1_c1.jpg", "p1_c2.jpg", "p1_c3.jpg"]},
}
MULTI_FINES = {
    "p1": {"city": ["Fatime Fossi lives in Douala",
                     "F. Fossi resides in Douala"]},
}


class TestMultiImageProbes:
    def test_frozen_probe_id_present(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     FINE, IMAGES)
        for p in probes:
            assert "probe_id" in p
            assert len(p["probe_id"]) == 16

    def test_probe_id_deterministic(self):
        p1 = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                 FINE, IMAGES)
        p2 = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                 FINE, IMAGES)
        assert p1[0]["probe_id"] == p2[0]["probe_id"]

    def test_multi_image_generates_multiple_probes(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     MULTI_FINES, MULTI_IMAGES)
        # 3 images × 2 captions = 6 probes
        assert len(probes) == 6
        # All share the same identity and attribute
        assert all(p["identity_id"] == "p1" for p in probes)
        assert all(p["attribute"] == "city" for p in probes)

    def test_multi_image_unique_probe_ids(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     MULTI_FINES, MULTI_IMAGES)
        ids = {p["probe_id"] for p in probes}
        assert len(ids) == 6

    def test_max_images_caps(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     MULTI_FINES, MULTI_IMAGES,
                                     max_images=2)
        # 2 images × 2 captions = 4 probes
        assert len(probes) == 4

    def test_max_captions_caps(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     MULTI_FINES, MULTI_IMAGES,
                                     max_captions=1)
        # 3 images × 1 caption = 3 probes
        assert len(probes) == 3

    def test_image_idx_and_caption_idx(self):
        probes = build_target_probes(["p1"], HIERARCHIES, IDENTITIES,
                                     MULTI_FINES, MULTI_IMAGES)
        indices = {(p["image_idx"], p["caption_idx"]) for p in probes}
        assert len(indices) == 6
        assert (0, 0) in indices
        assert (2, 1) in indices


class TestImageCaptionVariance:
    def test_variance_computed(self):
        results = [
            {"identity_id": "p1", "attribute": "city",
             "image_file": "img1.jpg", "fine_caption": "cap1",
             "sims": {"fine": 0.3, "target": 0.2, "sibling": 0.1}},
            {"identity_id": "p1", "attribute": "city",
             "image_file": "img1.jpg", "fine_caption": "cap2",
             "sims": {"fine": 0.32, "target": 0.22, "sibling": 0.12}},
            {"identity_id": "p1", "attribute": "city",
             "image_file": "img2.jpg", "fine_caption": "cap1",
             "sims": {"fine": 0.28, "target": 0.18, "sibling": 0.08}},
            {"identity_id": "p1", "attribute": "city",
             "image_file": "img2.jpg", "fine_caption": "cap2",
             "sims": {"fine": 0.31, "target": 0.21, "sibling": 0.11}},
        ]
        var = image_caption_variance(results)
        assert var["num_identity_attr_pairs"] == 1
        assert var["image_std_mean"] is not None
        assert var["caption_std_mean"] is not None
        assert var["image_std_mean"] > 0
        assert var["caption_std_mean"] > 0


class TestAggregateByImage:
    def test_by_image(self):
        results = [
            {"image_file": "img1.jpg",
             "sims": {"fine": 0.4, "target": 0.1, "sibling": 0.05}},
            {"image_file": "img1.jpg",
             "sims": {"fine": 0.3, "target": 0.2, "sibling": 0.1}},
            {"image_file": "img2.jpg",
             "sims": {"fine": 0.2, "target": 0.3, "sibling": 0.05}},
        ]
        by_img = aggregate_scores_by_image(results)
        assert "img1.jpg" in by_img
        assert "img2.jpg" in by_img
        assert by_img["img1.jpg"]["num_probes"] == 2
        assert by_img["img2.jpg"]["num_probes"] == 1
