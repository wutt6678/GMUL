"""Unit tests for SALMU unlearning groups + persona probe splits."""

from __future__ import annotations

import pytest

from granunlearn.salmu.state_datasets import SalmuTrainingPair
from granunlearn.salmu.unlearning import (
    SalmuGroupSpec,
    build_salmu_unlearning_groups,
    split_target_personas,
)

HIERARCHIES = {
    "p1": {"city": {"levels": ["Douala", "Cameroon"], "target_level": 1}},
    "p2": {"city": {"levels": ["Yaounde", "Cameroon"], "target_level": 1}},
}
IDENTITIES = {"p1": {"name": "Fatime Fossi"},
              "p2": {"name": "Amara Ngu"}}


def _pair(pid, iid, role, caption, image, level=0):
    return SalmuTrainingPair(
        pair_id=pid, state="MF", identity_id=iid, attribute="city",
        role=role, level_index=level, caption=caption,
        caption_source="released_fine", image_file=image)


MF_PAIRS = [
    _pair("a1", "p1", "target", "Fatime Fossi lives in Douala",
          "p1_001.jpg"),
    _pair("a2", "p1", "target", "Fatime Fossi resides in Douala",
          "p1_002.jpg"),
    _pair("b1", "p2", "retain", "Amara Ngu lives in Yaounde",
          "p2_001.jpg"),
]


class TestSplit:
    def test_disjoint_union_and_determinism(self):
        ids = [f"id_{i:03d}" for i in range(60)]
        s1 = split_target_personas(ids)
        s2 = split_target_personas(ids)
        assert s1 == s2
        union = s1["train"] + s1["val"] + s1["test"]
        assert len(set(union)) == 60 == len(union)
        assert len(s1["train"]) == 40 and len(s1["val"]) == 10 \
            and len(s1["test"]) == 10

    def test_seed_changes_assignment(self):
        ids = [f"id_{i:03d}" for i in range(60)]
        assert split_target_personas(ids, seed=42)["test"] != \
            split_target_personas(ids, seed=7)["test"]


class TestGroups:
    def test_group_contents(self):
        groups = build_salmu_unlearning_groups(
            MF_PAIRS, HIERARCHIES, IDENTITIES)
        # fine_target = target pairs verbatim
        assert {p.pair_id for p in groups["fine_target"]} == {"a1", "a2"}
        # retain = retain pairs verbatim
        assert {p.pair_id for p in groups["retain"]} == {"b1"}
        # target_level: one caption per (identity, attr), SAME images
        tl = groups["target_level"]
        assert len(tl) == 2
        assert all(p.caption == "Fatime Fossi lives in Cameroon."
                   for p in tl)
        assert {p.image_file for p in tl} == {"p1_001.jpg", "p1_002.jpg"}
        assert all(p.level_index == 1 for p in tl)
        # no overlap with evaluation data is structural: groups only
        # reference MF's released pairs
        all_ids = [p.pair_id for g in groups.values() for p in g]
        assert len(all_ids) == len(set(all_ids))

    def test_writes_jsonl(self, tmp_path):
        build_salmu_unlearning_groups(
            MF_PAIRS, HIERARCHIES, IDENTITIES, out_dir=tmp_path)
        for name in ("fine_target", "target_level", "retain"):
            assert (tmp_path / f"{name}.jsonl").exists()


class TestGroupSpec:
    def test_mode_validation(self):
        with pytest.raises(ValueError):
            SalmuGroupSpec("x", [], "ascent", 1.0)
        with pytest.raises(ValueError):
            SalmuGroupSpec("x", [], "gd", 0.0)
