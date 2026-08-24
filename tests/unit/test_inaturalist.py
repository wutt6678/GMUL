"""Unit tests for the iNaturalist dataset adapter and build pipeline."""

from __future__ import annotations

import pytest

from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.inaturalist import INaturalistAdapter, SMOKE_TAXONOMY
from granunlearn.datasets.split import deterministic_split, split_counts
from granunlearn.hierarchy.validate import validate_chain


class TestINaturalistAdapter:
    def test_adapter_loads_20_species(self):
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        assert len(raw) == 20

    def test_adapter_loads_limited_species(self):
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 5})
        assert len(raw) == 5

    def test_adapter_builds_associations(self):
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        assocs = adapter.to_associations(raw, {"seed": 42, "target_level": 1})
        assert len(assocs) == 20

    def test_taxonomy_deterministic(self):
        """species→genus→family hierarchy is fully deterministic — no Qwen needed."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 3})
        assocs = adapter.to_associations(raw, {"seed": 42})
        for a in assocs:
            assert a.hierarchy_type == "taxonomic"
            assert a.provenance.generation_model is None
            assert a.provenance.hierarchy_builder == "deterministic"

    def test_hierarchy_chain_valid(self):
        """Every built association passes hierarchy validation."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        assocs = adapter.to_associations(raw, {"seed": 42})
        for a in assocs:
            issues = validate_chain(a.levels)
            errors = [i for i in issues if i.is_error]
            assert len(errors) == 0, f"Validation failed for {a.entity_id}: {errors}"

    def test_species_genus_family_chain(self):
        """Each species has a 3-level chain: species→genus→family."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 1})
        assocs = adapter.to_associations(raw, {"seed": 42})
        a = assocs[0]
        assert a.num_levels() == 3
        assert a.fine_value().value == "Passer domesticus"
        assert a.target_value().value == "Passer"  # target_level=1 = genus

    def test_multiple_genera_and_families(self):
        """Smoke fixture has ≥2 genera and ≥2 families."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        genera = {r["genus"] for r in raw}
        families = {r["family"] for r in raw}
        assert len(genera) >= 2, f"Need ≥2 genera, got {len(genera)}"
        assert len(families) >= 2, f"Need ≥2 families, got {len(families)}"

    def test_train_test_image_separation(self):
        """All images for an entity go to the same split."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        assocs = adapter.to_associations(raw, {"seed": 42})

        # Each entity has exactly one association → one split
        entity_splits = {}
        for a in assocs:
            if a.entity_id in entity_splits:
                assert entity_splits[a.entity_id] == a.split.split
            entity_splits[a.entity_id] = a.split.split

    def test_no_qwen_required(self):
        """iNaturalist smoke requires no Qwen API calls."""
        adapter = INaturalistAdapter()
        raw = adapter.load_raw({"max_species": 20})
        assocs = adapter.to_associations(raw, {"seed": 42})
        qwen_fraction = sum(
            1 for a in assocs if a.provenance.generation_model is not None
        ) / len(assocs)
        assert qwen_fraction == 0.0


class TestDeterministicSplit:
    def test_split_reproducible(self):
        """Same seed → same split assignment."""
        ids = [f"entity_{i}" for i in range(50)]
        s1 = deterministic_split(ids, seed=42)
        s2 = deterministic_split(ids, seed=42)
        assert s1 == s2

    def test_split_different_seed(self):
        """Different seed → different assignment (with high probability)."""
        ids = [f"entity_{i}" for i in range(100)]
        s1 = deterministic_split(ids, seed=42)
        s2 = deterministic_split(ids, seed=99)
        # At least some entities should differ
        diffs = sum(1 for eid in ids if s1[eid].split != s2[eid].split)
        assert diffs > 10

    def test_split_ratios_approximate(self):
        """With enough entities, splits are roughly proportional."""
        ids = [f"entity_{i}" for i in range(1000)]
        splits = deterministic_split(ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        counts = split_counts(splits)
        assert 500 < counts["train"] < 700
        assert 100 < counts["val"] < 300
        assert 100 < counts["test"] < 300

    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            deterministic_split(["a"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


class TestGetAdapter:
    def test_get_inaturalist(self):
        adapter = get_adapter("inaturalist")
        assert isinstance(adapter, INaturalistAdapter)

    def test_get_inat_alias(self):
        adapter = get_adapter("inat")
        assert isinstance(adapter, INaturalistAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_adapter("nonexistent")
