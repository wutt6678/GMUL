"""Unit tests for Iteration 9 unlearning knowledge groups."""

from __future__ import annotations

import pytest

from granunlearn.datasets.smoke import select_target_retain
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    ProvenanceInfo,
    SplitInfo,
)
from granunlearn.training.unlearning_datasets import (
    GROUPS,
    build_unlearning_group,
    validate_unlearning_groups,
    write_unlearning_groups,
)


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value.lower()}",
        value=value, normalized_value=value.lower())


def make_assoc(aid: str, entity_id: str, attribute_name: str = "occupation",
               hierarchy_type: str = "semantic",
               values: list[str] | None = None, target_level: int = 1,
               with_image: bool = True) -> AssociationRecord:
    values = values or ["fine val", "mid val", "coarse val"]
    images = []
    if with_image:
        images.append(ImageRef(image_id=f"img_{aid}",
                               path=f"data/images/{aid}.jpg",
                               source="materialized", split="train"))
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name=f"Person {entity_id}", attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=[_level(i, v) for i, v in enumerate(values)],
        original_level=0, target_level=target_level,
        images=images, split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


def pool_and_partition():
    assocs = [
        make_assoc("e1__occupation", "e1"),
        make_assoc("e1__salary", "e1", attribute_name="salary",
                   hierarchy_type="numeric"),
        make_assoc("e1__height", "e1", attribute_name="height"),
        make_assoc("e2__occupation", "e2"),
        make_assoc("e2__salary", "e2", attribute_name="salary",
                   hierarchy_type="numeric"),
        make_assoc("e2__height", "e2", attribute_name="height"),
    ]
    partition = select_target_retain(assocs, seed=42)
    return assocs, partition


class TestGroupConstruction:
    def test_fine_target_covers_targets_at_level_zero(self):
        assocs, partition = pool_and_partition()
        ex = build_unlearning_group(assocs, partition, "fine_target")
        assert {e.association_id for e in ex} == \
            set(partition["target_association_ids"])
        assert all(e.level_index == 0 for e in ex)
        assert all(e.role == "target" for e in ex)

    def test_target_level_uses_association_target_level(self):
        assocs, partition = pool_and_partition()
        ex = build_unlearning_group(assocs, partition, "target_level")
        by_id = {a.association_id: a for a in assocs}
        for e in ex:
            assert e.level_index == by_id[e.association_id].target_level
            assert e.level_value == \
                by_id[e.association_id].levels[e.level_index].value

    def test_retain_covers_retain_at_level_zero(self):
        assocs, partition = pool_and_partition()
        ex = build_unlearning_group(assocs, partition, "retain")
        assert {e.association_id for e in ex} == \
            set(partition["retain_association_ids"])
        assert all(e.level_index == 0 and e.role == "retain" for e in ex)

    def test_template_identical_across_groups(self):
        assocs, partition = pool_and_partition()
        target_id = partition["target_association_ids"][0]
        ft = build_unlearning_group(assocs, partition, "fine_target")
        tl = build_unlearning_group(assocs, partition, "target_level")
        pf = next(e for e in ft if e.association_id == target_id)
        pt = next(e for e in tl if e.association_id == target_id)
        assert pf.prompt == pt.prompt  # one controlled template

    def test_unknown_group_raises(self):
        assocs, partition = pool_and_partition()
        with pytest.raises(ValueError):
            build_unlearning_group(assocs, partition, "bogus")


class TestValidationAndWrite:
    def test_validation_passes_and_write_roundtrip(self, tmp_path):
        assocs, partition = pool_and_partition()
        manifest = write_unlearning_groups(assocs, partition, tmp_path)
        assert set(manifest["groups"]) == set(GROUPS)
        for group in GROUPS:
            path = tmp_path / f"{group}.jsonl"
            assert path.exists()
            lines = path.read_text().splitlines()
            assert len(lines) == manifest["groups"][group]["num_examples"]
            # repo-relative paths only
            assert "/scratch" not in path.read_text()

    def test_validation_detects_wrong_levels(self):
        assocs, partition = pool_and_partition()
        groups = {g: build_unlearning_group(assocs, partition, g)
                  for g in GROUPS}
        # sabotage: fine_target example moved to level 1
        groups["fine_target"][0] = groups["fine_target"][0].model_copy(
            update={"level_index": 1})
        errors = validate_unlearning_groups(groups, partition)
        assert any("fine_target must be level 0" in e for e in errors)

    def test_validation_detects_set_mismatch(self):
        assocs, partition = pool_and_partition()
        groups = {g: build_unlearning_group(assocs, partition, g)
                  for g in GROUPS}
        groups["retain"] = groups["retain"][:-1]  # drop one retain example
        errors = validate_unlearning_groups(groups, partition)
        assert any("retain must cover exactly" in e for e in errors)
