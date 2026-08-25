"""Unit tests for Iteration 7 reference-state knowledge datasets.

Training data is derived BY STATE from associations + F/R partition —
never from evaluation queries; negation_correction stays evaluation-only.
"""

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
from granunlearn.training.state_datasets import (
    STATES,
    build_state_examples,
    level_index_for_state,
    load_state_examples,
    validate_state_examples,
    write_state_datasets,
)


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value.lower()}",
        value=value, normalized_value=value.lower())


def make_assoc(aid: str, entity_id: str, attribute_name: str = "occupation",
               hierarchy_type: str = "semantic",
               values: list[str] | None = None, target_level: int = 1,
               with_image: bool = True) -> AssociationRecord:
    values = values or ["fine val", "coarse val"]
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
        make_assoc("e1__height", "e1", attribute_name="height",
                   hierarchy_type="numeric"),
        make_assoc("e2__occupation", "e2"),
        make_assoc("e2__salary", "e2", attribute_name="salary",
                   hierarchy_type="numeric"),
        make_assoc("e2__height", "e2", attribute_name="height",
                   hierarchy_type="numeric"),
    ]
    partition = select_target_retain(assocs, seed=42)
    return assocs, partition


class TestLevelIndexForState:
    def test_mf_always_fine(self):
        a = make_assoc("x", "e", target_level=1)
        assert level_index_for_state(a, "MF") == 0

    def test_mg_uses_target_level(self):
        a = make_assoc("x", "e", values=["a", "b", "c"], target_level=2)
        assert level_index_for_state(a, "MG") == 2

    def test_mn_omits_targets(self):
        a = make_assoc("x", "e")
        assert level_index_for_state(a, "MN") is None

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError):
            level_index_for_state(make_assoc("x", "e"), "MU")


class TestBuildStateExamples:
    def test_mf_all_fine(self):
        assocs, partition = pool_and_partition()
        ex = build_state_examples(assocs, partition, "MF")
        assert len(ex) == len(assocs)
        assert all(e.level_index == 0 for e in ex)
        assert validate_state_examples(ex, partition, "MF") == []

    def test_mg_targets_at_target_level_retain_fine(self):
        assocs, partition = pool_and_partition()
        ex = build_state_examples(assocs, partition, "MG")
        targets = set(partition["target_association_ids"])
        for e in ex:
            if e.association_id in targets:
                assoc = next(a for a in assocs
                             if a.association_id == e.association_id)
                assert e.level_index == assoc.target_level
            else:
                assert e.level_index == 0
        assert validate_state_examples(ex, partition, "MG") == []

    def test_mn_omits_targets(self):
        assocs, partition = pool_and_partition()
        ex = build_state_examples(assocs, partition, "MN")
        targets = set(partition["target_association_ids"])
        assert all(e.association_id not in targets for e in ex)
        assert len(ex) == len(partition["retain_association_ids"])
        assert all(e.level_index == 0 for e in ex)
        assert validate_state_examples(ex, partition, "MN") == []

    def test_identical_template_across_states(self):
        assocs, partition = pool_and_partition()
        retain_id = partition["retain_association_ids"][0]
        prompts = set()
        for state in STATES:
            ex = build_state_examples(assocs, partition, state)
            e = next(x for x in ex if x.association_id == retain_id)
            prompts.add(e.prompt)
        assert len(prompts) == 1  # ONE controlled template

    def test_completion_contains_level_value(self):
        assocs, partition = pool_and_partition()
        for state in STATES:
            for e in build_state_examples(assocs, partition, state):
                assert e.level_value in e.completion

    def test_never_derived_from_queries(self):
        """No query artifact is consumed: examples reference only
        association ids/levels (structural guard)."""
        assocs, partition = pool_and_partition()
        for state in STATES:
            for e in build_state_examples(assocs, partition, state):
                assert e.association_id in {
                    a.association_id for a in assocs}

    def test_unknown_state_raises(self):
        assocs, partition = pool_and_partition()
        with pytest.raises(ValueError):
            build_state_examples(assocs, partition, "MU")


class TestWriteAndLoad:
    def test_roundtrip_and_manifest(self, tmp_path):
        assocs, partition = pool_and_partition()
        manifest = write_state_datasets(assocs, partition, tmp_path)
        assert set(manifest["states"]) == set(STATES)
        assert manifest["states"]["MN"]["num_target"] == 0
        assert manifest["states"]["MF"]["num_examples"] == len(assocs)
        for state in STATES:
            loaded = load_state_examples(tmp_path / f"{state}.jsonl")
            assert len(loaded) == manifest["states"][state]["num_examples"]
            assert all(e.state == state for e in loaded)

    def test_validation_failure_surfaces(self):
        assocs, partition = pool_and_partition()
        examples = build_state_examples(assocs, partition, "MN")
        # sabotage: smuggle a target association into MN
        target_id = partition["target_association_ids"][0]
        examples[0] = examples[0].model_copy(
            update={"association_id": target_id, "role": "target"})
        errors = validate_state_examples(examples, partition, "MN")
        assert errors


class TestRepoRelativeImagePaths:
    """Persisted image_path must be repo-relative; resolution happens at
    load time (Iteration 7 review — no machine-specific paths in the
    committed jsonl)."""

    def test_build_persists_repo_relative_paths(self, tmp_path):
        assocs, partition = pool_and_partition()
        for state in STATES:
            for e in build_state_examples(
                    assocs, partition, state, repo_root=tmp_path):
                if e.image_path is not None:
                    assert not e.image_path.startswith("/")
                    assert e.image_path.startswith("data/images/")

    def test_absolute_path_relativized(self, tmp_path):
        assocs, partition = pool_and_partition()
        aid = assocs[0].association_id
        abs_assoc = assocs[0].model_copy(update={"images": [ImageRef(
            image_id=f"img_{aid}",
            path=str(tmp_path / "data" / "images" / f"{aid}.jpg"),
            source="materialized", split="train")]})
        ex = build_state_examples(
            [abs_assoc] + assocs[1:], partition, "MF", repo_root=tmp_path)
        e = next(x for x in ex if x.association_id == aid)
        assert e.image_path == f"data/images/{aid}.jpg"

    def test_absolute_path_outside_repo_rejected(self, tmp_path):
        assocs, partition = pool_and_partition()
        aid = assocs[0].association_id
        abs_assoc = assocs[0].model_copy(update={"images": [ImageRef(
            image_id=f"img_{aid}", path="/elsewhere/img.jpg",
            source="materialized", split="train")]})
        with pytest.raises(ValueError):
            build_state_examples(
                [abs_assoc] + assocs[1:], partition, "MF",
                repo_root=tmp_path)

    def test_load_resolves_against_repo_root(self, tmp_path):
        assocs, partition = pool_and_partition()
        # materialize the image files so validation passes
        img_dir = tmp_path / "data" / "images"
        img_dir.mkdir(parents=True)
        for a in assocs:
            (img_dir / f"{a.association_id}.jpg").write_bytes(b"x")
        manifest = write_state_datasets(
            assocs, partition, tmp_path, repo_root=tmp_path)
        raw_line = (tmp_path / "MF.jsonl").read_text().splitlines()[0]
        assert not str(tmp_path) in raw_line  # nothing absolute persisted
        loaded = load_state_examples(tmp_path / "MF.jsonl",
                                     repo_root=tmp_path)
        for e in loaded:
            if e.image_path is not None:
                assert e.image_path.startswith(str(tmp_path))
        # without repo_root the persisted form stays relative
        unresolved = load_state_examples(tmp_path / "MF.jsonl")
        assert all(
            e.image_path is None or not e.image_path.startswith("/")
            for e in unresolved)
        assert manifest["states"]["MF"]["num_image_text"] == len(assocs)

    def test_validate_flags_absolute_and_missing(self, tmp_path):
        assocs, partition = pool_and_partition()
        examples = build_state_examples(assocs, partition, "MF")
        examples[0] = examples[0].model_copy(
            update={"image_path": "/scratch/somewhere/img.jpg"})
        errors = validate_state_examples(examples, partition, "MF")
        assert any("repo-relative" in err for err in errors)
        examples[0] = examples[0].model_copy(
            update={"image_path": "data/images/missing.jpg"})
        errors = validate_state_examples(
            examples, partition, "MF", repo_root=tmp_path)
        assert any("missing" in err for err in errors)
