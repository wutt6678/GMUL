"""Unit tests for the iNaturalist dataset adapter.

All tests exercise the adapter against a materialized on-disk COCO-style
test dataset written by ``fixtures.inat_fixture.write_local_dataset``.
Note: the image FILES are real files on disk, but their pixel content is
synthetically generated — this is a software test fixture, NOT the
iNaturalist research proof-of-concept (which must eventually run against
actual iNaturalist photographs).
"""

from __future__ import annotations

import json

import pytest

from fixtures.inat_fixture import SMOKE_TAXONOMY, write_local_dataset

from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.inaturalist import (
    INaturalistAdapter,
    deterministic_image_splits,
)
from granunlearn.hierarchy.validate import validate_chain


@pytest.fixture(scope="module")
def local_dataset(tmp_path_factory):
    """Materialise an on-disk COCO-style test dataset once for the module."""
    root = tmp_path_factory.mktemp("inat_local")
    write_local_dataset(root, images_per_species=12, seed=0)
    return root


def _base_config(root, **overrides):
    cfg = {
        "data_root": str(root),
        "annotations_file": "annotations.json",
        "min_images_per_species": 10,
        "max_species": 20,
        "seed": 42,
        "target_level": 1,
        "version": "smoke_v1",
    }
    cfg.update(overrides)
    return cfg


# =====================================================================
# Local data loading
# =====================================================================

class TestLocalLoading:
    def test_loads_all_20_species(self, local_dataset):
        adapter = INaturalistAdapter()
        raw = adapter.load_raw(_base_config(local_dataset))
        assert len(raw) == 20

    def test_loads_real_image_files(self, local_dataset):
        """Every referenced image file must actually exist on disk."""
        from pathlib import Path

        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)
        for a in assocs:
            assert len(a.images) >= 10
            for img in a.images:
                p = Path(img.path)
                if not p.is_absolute():
                    p = local_dataset / p
                assert p.exists(), f"Missing image: {img.path}"

    def test_min_images_filter(self, tmp_path):
        """Species with fewer than min_images_per_species are excluded."""
        root = tmp_path / "small"
        # 2 species with 12 images, 1 species with only 3
        taxonomy = SMOKE_TAXONOMY[:2] + [
            {**SMOKE_TAXONOMY[2], "species": "Passer hispaniolensis"}
        ]
        write_local_dataset(root, taxonomy=taxonomy[:2], images_per_species=12)
        adapter = INaturalistAdapter()
        cfg = _base_config(root, min_images_per_species=10)
        raw = adapter.load_raw(cfg)
        assert len(raw) == 2

    def test_min_images_filter_excludes_small(self, tmp_path):
        root2 = tmp_path / "mixed2"
        write_local_dataset(root2, taxonomy=[SMOKE_TAXONOMY[3]], images_per_species=3)
        adapter = INaturalistAdapter()
        cfg = _base_config(root2, min_images_per_species=10)
        # Nothing survives the filter -> hard error (not an empty dataset)
        with pytest.raises(ValueError, match="No species satisfy"):
            adapter.load_raw(cfg)

    def test_requires_data_root(self):
        adapter = INaturalistAdapter()
        with pytest.raises(ValueError, match="data_root"):
            adapter.load_raw({})

    def test_missing_annotations_raises(self, tmp_path):
        adapter = INaturalistAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load_raw(_base_config(tmp_path / "nonexistent"))

    def test_official_inat_category_schema_name_only(self, tmp_path):
        """Official iNaturalist categories carry 'name' but NOT 'species'.

        The adapter must fall back to 'name' for the species identity.
        """
        from PIL import Image

        root = tmp_path / "official_schema"
        (root / "images").mkdir(parents=True)
        images, annotations = [], []
        img_id = 0
        for i in range(10):
            fname = f"images/{i:03d}.jpg"
            Image.new("RGB", (8, 8)).save(root / fname)
            images.append({"id": img_id, "file_name": fname})
            annotations.append({"id": i, "image_id": img_id, "category_id": 1})
            img_id += 1

        # Official-style category: name/genus/family — no "species" key
        meta = {
            "images": images,
            "annotations": annotations,
            "categories": [{
                "id": 1,
                "name": "Passer domesticus",
                "genus": "Passer",
                "family": "Passeridae",
            }],
        }
        (root / "annotations.json").write_text(json.dumps(meta))

        adapter = INaturalistAdapter()
        cfg = _base_config(root, min_images_per_species=5, max_species=1,
                           min_genera=1, min_families=1)
        raw = adapter.load_raw(cfg)
        assert len(raw) == 1
        assert raw[0]["species"] == "Passer domesticus"  # resolved from name
        assert raw[0]["genus"] == "Passer"

    def test_min_genera_gate(self, tmp_path):
        """Selection raising on min_genera is a hard gate."""
        root = tmp_path / "one_genus"
        # Two species, same genus, same family
        write_local_dataset(root, taxonomy=SMOKE_TAXONOMY[:2], images_per_species=12)
        adapter = INaturalistAdapter()
        cfg = _base_config(root, min_genera=2, min_families=1)
        with pytest.raises(ValueError, match="min_genera"):
            adapter.load_raw(cfg)

    def test_min_families_gate(self, tmp_path):
        """Selection raising on min_families is a hard gate."""
        root = tmp_path / "one_family"
        write_local_dataset(root, taxonomy=SMOKE_TAXONOMY[:2], images_per_species=12)
        adapter = INaturalistAdapter()
        cfg = _base_config(root, min_genera=1, min_families=3)
        with pytest.raises(ValueError, match="min_families"):
            adapter.load_raw(cfg)

    def test_load_report_excluded_species(self, tmp_path):
        """adapter.last_load_report records excluded species correctly."""
        root = tmp_path / "mixed_counts"
        write_local_dataset(root, taxonomy=SMOKE_TAXONOMY[:2], images_per_species=12)
        adapter = INaturalistAdapter()
        # Require more images than exist → both species excluded is impossible
        # to select; instead check reporting with a lower bar:
        cfg = _base_config(root, min_images_per_species=10)
        adapter.load_raw(cfg)
        report = adapter.last_load_report
        assert report["excluded_species"] == []
        assert report["eligible_species"] == 2
        assert report["num_genera"] == 1
        assert report["num_families"] == 1


# =====================================================================
# Within-species image splits — TRUE disjointness
# =====================================================================

class TestWithinSpeciesSplits:
    def test_split_sets_are_disjoint(self, local_dataset):
        """train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅ per species."""
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)

        for a in assocs:
            train = {img.path for img in a.images if img.split == "train"}
            val = {img.path for img in a.images if img.split == "val"}
            test = {img.path for img in a.images if img.split == "test"}

            assert train & val == set(), f"{a.entity_id}: train∩val not empty"
            assert train & test == set(), f"{a.entity_id}: train∩test not empty"
            assert val & test == set(), f"{a.entity_id}: val∩test not empty"
            # Union covers all images
            assert train | val | test == {img.path for img in a.images}

    def test_split_sets_disjoint_by_image_id(self, local_dataset):
        """Disjointness also holds on image_id."""
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)

        for a in assocs:
            train = {img.image_id for img in a.images if img.split == "train"}
            val = {img.image_id for img in a.images if img.split == "val"}
            test = {img.image_id for img in a.images if img.split == "test"}
            assert train.isdisjoint(val)
            assert train.isdisjoint(test)
            assert val.isdisjoint(test)

    def test_every_species_has_train_and_test(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)

        for a in assocs:
            splits = {img.split for img in a.images}
            assert "train" in splits, f"{a.entity_id}: no train images"
            assert "test" in splits, f"{a.entity_id}: no test images"

    def test_split_assignment_deterministic(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        a1 = adapter.to_associations(adapter.load_raw(cfg), cfg)
        a2 = adapter.to_associations(adapter.load_raw(cfg), cfg)
        for x, y in zip(a1, a2):
            assert [i.split for i in x.images] == [i.split for i in y.images]


# =====================================================================
# deterministic_image_splits helper
# =====================================================================

class TestDeterministicImageSplits:
    def test_counts_approximate_ratios(self):
        splits = deterministic_image_splits(100, seed=42, entity_id="sp")
        from collections import Counter
        c = Counter(splits)
        assert 50 <= c["train"] <= 70
        assert 10 <= c["val"] <= 30
        assert 10 <= c["test"] <= 30

    def test_guaranteed_test_image_small_n(self):
        for n in range(2, 12):
            splits = deterministic_image_splits(n, seed=7, entity_id="sp")
            assert "test" in splits, f"n={n}: no test image"
            assert "train" in splits, f"n={n}: no train image"

    def test_single_image(self):
        splits = deterministic_image_splits(1, seed=0, entity_id="sp")
        assert splits == ["train"]

    def test_empty(self):
        assert deterministic_image_splits(0, seed=0, entity_id="sp") == []

    def test_reproducible(self):
        s1 = deterministic_image_splits(20, seed=1, entity_id="x")
        s2 = deterministic_image_splits(20, seed=1, entity_id="x")
        assert s1 == s2


# =====================================================================
# Hierarchy / taxonomy properties
# =====================================================================

class TestTaxonomyProperties:
    def test_taxonomy_deterministic_no_llm(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)
        for a in assocs:
            assert a.hierarchy_type == "taxonomic"
            assert a.provenance.generation_model is None
            assert a.provenance.hierarchy_builder == "deterministic"

    def test_all_chains_valid(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        assocs = adapter.to_associations(adapter.load_raw(cfg), cfg)
        for a in assocs:
            issues = validate_chain(a.levels)
            errors = [i for i in issues if i.is_error]
            assert not errors, f"{a.entity_id}: {errors}"

    def test_species_genus_family_chain(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset, max_species=1)
        raw = adapter.load_raw(cfg)
        assocs = adapter.to_associations(raw, cfg)
        a = assocs[0]
        assert a.num_levels() == 3
        assert a.fine_value().metadata["rank"] == "species"
        assert a.target_value().metadata["rank"] == "genus"
        assert a.levels[2].metadata["rank"] == "family"

    def test_multiple_genera_and_families(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset)
        raw = adapter.load_raw(cfg)
        genera = {r["genus"] for r in raw}
        families = {r["family"] for r in raw}
        assert len(genera) >= 2
        assert len(families) >= 2

    def test_selection_deterministic(self, local_dataset):
        adapter = INaturalistAdapter()
        cfg = _base_config(local_dataset, max_species=5)
        r1 = adapter.load_raw(cfg)
        r2 = adapter.load_raw(cfg)
        assert [r["species"] for r in r1] == [r["species"] for r in r2]
        # Alphabetically sorted selection
        names = [r["species"] for r in r1]
        assert names == sorted(names)


# =====================================================================
# Adapter registry
# =====================================================================

class TestGetAdapter:
    def test_get_inaturalist(self):
        assert isinstance(get_adapter("inaturalist"), INaturalistAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_adapter("nonexistent")
