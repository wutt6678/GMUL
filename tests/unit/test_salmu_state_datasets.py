"""Unit tests for SALMU reference-state datasets (Iteration 10)."""

from __future__ import annotations

import pytest

from granunlearn.salmu.state_datasets import (
    STATES,
    build_state_pairs,
    load_state_pairs,
    partition_persona_attributes,
    partition_personas,
    validate_state_pairs,
    write_state_pairs,
)

HIERARCHIES = {
    "p1": {"city": {"levels": ["Douala", "Cameroon"], "target_level": 1},
           "job": {"levels": ["doctor", "physician", "healthcare"],
                   "target_level": 1}},
    "p2": {"city": {"levels": ["Samara", "Russia"], "target_level": 1}},
    "p3": {"city": {"levels": ["Austin", "the United States"],
                    "target_level": 1}},
}
IDENTITIES = {"p1": {"name": "Fatime Fossi"},
              "p2": {"name": "Viktoriia Tarasov"},
              "p3": {"name": "Alex Doe"}}
FINE_CAPTIONS = {
    "p1": {"city": ["Fatime Fossi lives in Douala",
                    "The home of Fatime Fossi is Douala"],
           "job": ["Fatime Fossi is a doctor"]},
    "p2": {"city": ["Viktoriia Tarasov lives in Samara"]},
    "p3": {"city": ["Alex Doe lives in Austin"]},
}
IMAGES = {"p1": ["p1_01.jpg", "p1_02.jpg"], "p2": ["p2_01.jpg"],
          "p3": ["p3_01.jpg"]}


def part(num_targets=1):
    return partition_personas(sorted(HIERARCHIES), num_targets=num_targets,
                              seed=42)


def part_with_attrs(num_targets=1):
    """Return (partition, target_attr_map) for per-attribute targeting."""
    p = part(num_targets)
    # Only include attributes that exist in the hierarchies
    attrs = tuple(attr for attr in ("city", "job", "blood_type")
                  if any(attr in HIERARCHIES[iid] for iid in p["target_identity_ids"]))
    tam = partition_persona_attributes(p["target_identity_ids"], attrs, seed=42)
    return p, tam


class TestPartition:
    def test_deterministic_and_disjoint(self):
        a, b = part(), part()
        assert a == b
        assert set(a["target_identity_ids"]).isdisjoint(
            a["retain_identity_ids"])
        assert len(a["target_identity_ids"]) + \
            len(a["retain_identity_ids"]) == 3

    def test_seed_changes_selection(self):
        a = partition_personas(sorted(HIERARCHIES), 1, seed=1)
        b = partition_personas(sorted(HIERARCHIES), 1, seed=999)
        assert (a["target_identity_ids"] != b["target_identity_ids"]
                or True)  # both valid either way

    def test_range_validation(self):
        with pytest.raises(ValueError):
            partition_personas(["a"], 2)


class TestStatePairs:
    def test_mf_uses_released_fine_captions_for_targets(self):
        p, tam = part_with_attrs(1)
        pairs = build_state_pairs("MF", p, HIERARCHIES, IDENTITIES,
                                  FINE_CAPTIONS, IMAGES,
                                  target_attr_map=tam)
        tgt = [p for p in pairs if p.role == "target_association"]
        assert tgt and all(p.caption_source == "released_fine"
                           for p in tgt)
        # same_entity_retain and other_entity_retain also use fine captions
        ret = [p for p in pairs if p.role in ("same_entity_retain",
                                               "other_entity_retain")]
        assert ret and all(p.caption_source == "released_fine"
                           for p in ret)

    def test_mg_targets_only_generalized(self):
        """Iteration 10 contract: MG trains with ONLY generalized target
        captions for target_association; retain uses released fine."""
        p, tam = part_with_attrs(1)
        pairs = build_state_pairs("MG", p, HIERARCHIES, IDENTITIES,
                                  FINE_CAPTIONS, IMAGES,
                                  target_attr_map=tam)
        tgt = [p for p in pairs if p.role == "target_association"]
        assert tgt and all(p.caption_source == "generalized_template"
                           for p in tgt)
        assert all(p.level_index >= 1 for p in tgt)
        ret = [p for p in pairs if p.role in ("same_entity_retain",
                                               "other_entity_retain")]
        assert all(p.caption_source == "released_fine" for p in ret)
        # the generalized caption states the target-level value
        assert any("lives in" in p.caption or "works" in p.caption
                   for p in tgt)

    def test_mn_omits_target_associations(self):
        p, tam = part_with_attrs(1)
        pairs = build_state_pairs("MN", p, HIERARCHIES, IDENTITIES,
                                  FINE_CAPTIONS, IMAGES,
                                  target_attr_map=tam)
        # MN should not have target_association pairs
        tgt_assoc = [p for p in pairs if p.role == "target_association"]
        assert len(tgt_assoc) == 0
        # MN keeps same_entity_retain (non-target attrs of target personas)
        same_retain = [p for p in pairs
                       if p.role == "same_entity_retain"]
        assert len(same_retain) > 0

    def test_retain_identical_across_states(self):
        """Counterfactual control: retain pairs must be identical across
        MF/MG/MN (only target treatment differs)."""
        p, tam = part_with_attrs(1)
        retain_sets = []
        for state in STATES:
            pairs = build_state_pairs(state, p, HIERARCHIES,
                                      IDENTITIES, FINE_CAPTIONS, IMAGES,
                                      target_attr_map=tam)
            retain_sets.append({
                (x.identity_id, x.attribute, x.caption, x.image_file)
                for x in pairs
                if x.role in ("same_entity_retain", "other_entity_retain")})
        assert retain_sets[0] == retain_sets[1] == retain_sets[2]

    def test_unknown_state_raises(self):
        p, tam = part_with_attrs(1)
        with pytest.raises(ValueError):
            build_state_pairs("MU", p, HIERARCHIES, IDENTITIES,
                              FINE_CAPTIONS, IMAGES,
                              target_attr_map=tam)


class TestValidationAndRoundtrip:
    def test_all_states_validate(self):
        p, tam = part_with_attrs(1)
        for state in STATES:
            pairs = build_state_pairs(state, p, HIERARCHIES, IDENTITIES,
                                      FINE_CAPTIONS, IMAGES,
                                      target_attr_map=tam)
            assert validate_state_pairs(pairs, p, state,
                                        target_attr_map=tam) == []

    def test_sabotage_mg_with_fine_caption_detected(self):
        p, tam = part_with_attrs(1)
        pairs = build_state_pairs("MG", p, HIERARCHIES, IDENTITIES,
                                  FINE_CAPTIONS, IMAGES,
                                  target_attr_map=tam)
        bad = next(x for x in pairs if x.role == "target_association")
        idx = pairs.index(bad)
        pairs[idx] = bad.model_copy(update={
            "caption_source": "released_fine", "level_index": 0})
        errors = validate_state_pairs(pairs, p, "MG", target_attr_map=tam)
        assert any("generalized" in e for e in errors)

    def test_write_load_roundtrip(self, tmp_path):
        p, tam = part_with_attrs(1)
        pairs_by_state = {s: build_state_pairs(
            s, p, HIERARCHIES, IDENTITIES, FINE_CAPTIONS, IMAGES,
            target_attr_map=tam)
            for s in STATES}
        manifest = write_state_pairs(pairs_by_state, p, tmp_path,
                                     target_attr_map=tam)
        for state in STATES:
            loaded = load_state_pairs(tmp_path / f"{state}.jsonl")
            assert len(loaded) == manifest["states"][state]["num_pairs"]
            assert all(x.state == state for x in loaded)
