"""Unit tests for the MLLMU-Bench source adapter (Iteration 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures.mllmu_fixture import write_jsonl, write_parquet

from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.mllmu import (
    ATTRIBUTE_POLICY,
    FIELD_ALIASES,
    INVENTORY_COLUMNS,
    MLLMUAdapter,
    normalize_field,
)


@pytest.fixture
def mllmu_parquet(tmp_path):
    """Synthetic data in the OFFICIAL release parquet layout (default)."""
    write_parquet(tmp_path / "mllmu", n_records=5)
    return tmp_path / "mllmu"


@pytest.fixture
def mllmu_jsonl(tmp_path):
    """Synthetic converted-JSONL copy."""
    path = write_jsonl(tmp_path / "mllmu_jsonl", n_records=5)
    return path


def _cfg_parquet(root, **overrides):
    cfg = {
        "data_root": str(root),
        "annotations_file": "Full_Set/train-00000-of-00001.parquet",
        "source_format": "parquet",
        "subset": "fictitious",
    }
    cfg.update(overrides)
    return cfg


def _cfg_jsonl(path, **overrides):
    cfg = {
        "data_root": str(path.parent),
        "annotations_file": path.name,
        "source_format": "jsonl",
        "subset": "fictitious",
    }
    cfg.update(overrides)
    return cfg


# =====================================================================
# Field normalization
# =====================================================================

class TestFieldNormalization:
    def test_alias_variants_collapse(self):
        assert normalize_field("Heights") == "height"
        assert normalize_field("Height") == "height"
        assert normalize_field("Annual Salary:") == "salary"
        assert normalize_field("Annual Salary: ") == "salary"
        assert normalize_field("Educated at:") == "education"
        assert normalize_field("Born") == "birthplace"
        assert normalize_field("Employment") == "occupation"

    def test_unknown_field_passthrough(self):
        assert normalize_field("Some New Field") == "Some New Field"


# =====================================================================
# Loading + provenance (parquet = official release format)
# =====================================================================

class TestLoadingParquet:
    def test_loads_all_records(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        assert len(records) == 5

    def test_every_record_has_provenance(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        for rec in records:
            assert rec.provenance is not None
            assert rec.provenance.source_dataset == "mllmu_bench"
            assert rec.provenance.source_entity_id is not None

    def test_entity_ids_normalized(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        ids = [rec.entity_id for rec in records]
        assert len(set(ids)) == len(ids), "entity IDs must be unique"
        assert all(i.startswith("mllmu_") for i in ids)

    def test_field_names_normalized_on_load(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        fields = records[0].fields
        assert "height" in fields        # from "Height"
        assert "salary" in fields        # from "Annual Salary: "
        assert "education" in fields     # from "Educated at:"
        assert "Annual Salary: " not in fields

    def test_directory_is_image_locator(self, mllmu_parquet):
        """Directory (official schema) is the persistent image locator;
        the HF Image dict column is never serialized."""
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        for rec in records:
            assert rec.image_path == f"full_images/{rec.raw_id}.jpg"
            assert "bytes" not in rec.image_path

    def test_parquet_is_default_format(self, mllmu_parquet):
        """source_format omitted -> defaults to parquet."""
        adapter = MLLMUAdapter()
        cfg = _cfg_parquet(mllmu_parquet)
        cfg.pop("source_format")
        records = adapter.load_raw(cfg)
        assert len(records) == 5
        assert adapter.last_load_report["source_format"] == "parquet"

    def test_load_report_counts(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        adapter.load_raw(_cfg_parquet(mllmu_parquet))
        report = adapter.last_load_report
        assert report["num_source_records"] == 5
        assert report["num_parsed"] == 5
        assert report["num_errors"] == 0
        assert report["source"] == "MLLMMU/MLLMU-Bench"
        assert report["subset"] == "Full_Set"

    def test_repeatability(self, mllmu_parquet):
        """Two loads of the same file yield identical entity-ID order."""
        adapter = MLLMUAdapter()
        r1 = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        r2 = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        assert [x.entity_id for x in r1] == [x.entity_id for x in r2]


# =====================================================================
# JSONL compatibility (converted copies)
# =====================================================================

class TestLoadingJsonl:
    def test_loads_jsonl_explicit_format(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_jsonl(mllmu_jsonl))
        assert len(records) == 5

    def test_jsonl_image_fallback(self, mllmu_jsonl):
        """JSONL copies without Directory fall back to a string 'image'."""
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_jsonl(mllmu_jsonl))
        for rec in records:
            assert rec.image_path == f"images/{rec.raw_id}.jpg"

    def test_unknown_format_rejected(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        with pytest.raises(ValueError, match="Unknown source_format"):
            adapter.load_raw(_cfg_jsonl(mllmu_jsonl, source_format="xml"))

    def test_jsonl_repeatability(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        r1 = adapter.load_raw(_cfg_jsonl(mllmu_jsonl))
        r2 = adapter.load_raw(_cfg_jsonl(mllmu_jsonl))
        assert [x.entity_id for x in r1] == [x.entity_id for x in r2]


# =====================================================================
# Duplicate-ID gate
# =====================================================================

class TestDuplicateIds:
    def test_duplicate_id_hard_error_parquet(self, tmp_path):
        root = tmp_path / "dup"
        write_parquet(root, n_records=5, duplicate_ids=True)
        adapter = MLLMUAdapter()
        with pytest.raises(ValueError, match="Duplicate source ID"):
            adapter.load_raw(_cfg_parquet(root))

    def test_duplicate_id_hard_error_jsonl(self, tmp_path):
        path = write_jsonl(tmp_path / "dup_j", n_records=5, duplicate_ids=True)
        adapter = MLLMUAdapter()
        with pytest.raises(ValueError, match="Duplicate source ID"):
            adapter.load_raw(_cfg_jsonl(path))


# =====================================================================
# Provenance: hierarchy_builder pending at parse time
# =====================================================================

class TestProvenanceBuilder:
    def test_hierarchy_builder_pending_before_build(self, mllmu_parquet):
        """No hierarchy exists at source-parse time, so the builder must
        be 'pending', not 'deterministic' (which would be premature)."""
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        for rec in records:
            assert rec.provenance.hierarchy_builder == "pending"


# =====================================================================
# Celebrity exclusion
# =====================================================================

class TestCelebrityExclusion:
    def test_celebrity_excluded_for_fictitious_subset(self, tmp_path):
        root = tmp_path / "celeb"
        write_parquet(root, n_records=3, include_celebrity=True)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(root))
        assert len(records) == 3  # celebrity not included
        assert adapter.last_load_report["num_celebrity_skipped"] == 1

    def test_celebrity_included_when_subset_not_fictitious(self, tmp_path):
        root = tmp_path / "all"
        write_parquet(root, n_records=3, include_celebrity=True)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(root, subset="all"))
        assert len(records) == 4


# =====================================================================
# Parse-error accounting
# =====================================================================

class TestParseErrors:
    def test_corrupt_record_counted(self, tmp_path):
        root = tmp_path / "corrupt"
        # 1/40 = 2.5% stays under the 5% threshold
        write_parquet(root, n_records=40, corrupt_idx=3)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(root))
        assert len(records) == 39
        assert adapter.last_load_report["num_errors"] == 1

    def test_high_error_rate_raises(self, tmp_path):
        root = tmp_path / "bad"
        write_parquet(root, n_records=4, corrupt_idx=0)
        # 1/4 = 25% > 5% threshold
        adapter = MLLMUAdapter()
        with pytest.raises(ValueError, match="parse error rate"):
            adapter.load_raw(_cfg_parquet(root))

    def test_bad_json_lines_counted_jsonl(self, tmp_path):
        path = write_jsonl(tmp_path / "badj", n_records=40, n_bad_json_lines=1)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_jsonl(path))
        assert len(records) == 40
        assert adapter.last_load_report["num_errors"] == 1

    def test_missing_source_raises(self, tmp_path):
        adapter = MLLMUAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load_raw(_cfg_parquet(tmp_path / "nope"))


# =====================================================================
# Attribute inventory
# =====================================================================

class TestInventory:
    def test_core_attributes_classified(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}

        for core in ["residence", "birthplace", "date_of_birth",
                     "occupation", "salary", "height", "education"]:
            assert by_attr[core]["include_core"] is True, core

    def test_unsupported_explicitly_excluded(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}

        # Unsupported attrs present in inventory but excluded with a reason
        for attr in ["gender", "medical_conditions", "parents",
                     "fun_facts", "description", "name"]:
            assert attr in by_attr, f"{attr} silently dropped"
            assert by_attr[attr]["include_core"] is False
            assert by_attr[attr]["notes"], f"{attr} has no exclusion reason"

    def test_hierarchy_type_assignment(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}
        assert by_attr["date_of_birth"]["suggested_hierarchy_type"] == "numeric"
        assert by_attr["salary"]["suggested_hierarchy_type"] == "numeric"
        assert by_attr["occupation"]["suggested_hierarchy_type"] == "semantic"
        assert by_attr["occupation"]["qwen_needed"] is True
        assert by_attr["date_of_birth"]["deterministic_possible"] is True

    def test_coverage_and_type_statistics(self, mllmu_parquet):
        """Inventory exposes coverage/type stats needed before writing
        deterministic parsers (Iteration 4)."""
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}

        salary = by_attr["salary"]
        assert salary["count"] == 5
        assert salary["coverage"] == 1.0
        assert salary["missing_count"] == 0
        assert salary["distinct_count"] >= 1
        assert salary["observed_python_types"] == "str"
        assert salary["parseable_count"] == 5

    def test_mixed_types_reported(self, mllmu_parquet):
        """parents takes str/list/dict values — types must be observable."""
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}
        parents_types = set(by_attr["parents"]["observed_python_types"].split(","))
        assert parents_types & {"str", "list", "dict"}
        # dict values are not parseable scalars
        assert by_attr["parents"]["parseable_count"] < 5

    def test_summary_json_fields(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        summary = adapter.build_inventory_summary(rows, source_revision="abc123")
        assert summary["source"] == "MLLMMU/MLLMU-Bench"
        assert summary["subset"] == "Full_Set"
        assert summary["source_revision"] == "abc123"
        assert summary["num_source_records"] == 5
        assert summary["num_parsed"] == 5
        assert summary["num_errors"] == 0
        assert len(summary["attributes"]) == len(rows)

    def test_write_inventory_csv(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_inventory(records)
        out = adapter.write_inventory_csv(rows, tmp_path / "inv.csv")
        assert out.exists()
        import csv
        with open(out) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == len(rows)
        assert set(reader[0].keys()) == set(INVENTORY_COLUMNS)

    def test_unknown_attribute_flagged(self, tmp_path):
        """Attributes outside the policy are excluded with an explicit note."""
        import json as _json
        path = tmp_path / "unk" / "Full_Set.jsonl"
        path.parent.mkdir(parents=True)
        bio = {"Name": "X", "Mystery Field": "value"}
        with open(path, "w") as f:
            f.write(_json.dumps({"ID": "001", "biography": _json.dumps(bio)}) + "\n")
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_jsonl(path))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}
        assert "Mystery Field" in by_attr
        assert by_attr["Mystery Field"]["include_core"] is False
        assert "pending review" in by_attr["Mystery Field"]["notes"]


# =====================================================================
# Registry + deferred association building
# =====================================================================

class TestRegistry:
    def test_get_adapter(self):
        assert isinstance(get_adapter("mllmu_hier"), MLLMUAdapter)
        assert isinstance(get_adapter("mllmu"), MLLMUAdapter)

    def test_to_associations_implemented(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        adapter.materialize_images(
            records, _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")})
        assocs = adapter.to_associations(records, _cfg_parquet(mllmu_parquet))
        assert len(assocs) > 0


# =====================================================================
# Image materialization (Iteration 4)
# =====================================================================

class TestImageMaterialization:
    def test_materializes_and_verifies(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        cfg = _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")}
        image_map = adapter.materialize_images(records, cfg)
        assert len(image_map) == 5
        for rec in records:
            stored = image_map[rec.entity_id]
            assert Path(stored).exists()
            assert Path(stored).stat().st_size > 0

    def test_corrupt_image_bytes_hard_error(self, tmp_path):
        root = tmp_path / "badimg"
        write_parquet(root, n_records=5, corrupt_image_idx=2)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(root))
        cfg = _cfg_parquet(root) | {"images_dir": str(tmp_path / "img")}
        with pytest.raises(ValueError, match="materialization failed"):
            adapter.materialize_images(records, cfg)

    def test_idempotent_rewrite(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        cfg = _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")}
        m1 = adapter.materialize_images(records, cfg)
        m2 = adapter.materialize_images(records, cfg)
        assert m1 == m2


# =====================================================================
# Real parse coverage gate (Iteration 4)
# =====================================================================

class TestParseCoverageGate:
    def test_all_attributes_measured(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        rows = adapter.build_parse_coverage(records)
        by_attr = {r["attribute"]: r for r in rows}
        assert set(by_attr) == {"residence", "birthplace", "date_of_birth",
                                "salary", "height"}
        for row in rows:
            assert row["total_profiles"] == 5
            # fixture values are parser-friendly -> full coverage
            assert row["parse_coverage"] == 1.0
            assert row["enabled"] is True

    def test_low_coverage_disables_attribute(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        # Sabotage salary on most profiles AFTER load
        for rec in records[1:]:
            rec.fields["salary"] = "NA"
        rows = adapter.build_parse_coverage(records, ["salary"], min_coverage=0.95)
        assert rows[0]["parse_coverage"] == pytest.approx(0.2)
        assert rows[0]["enabled"] is False
        assert rows[0]["failure_examples"]  # reported, never silent

    def test_per_attribute_threshold_override(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        for rec in records[1:]:
            rec.fields["salary"] = "NA"
        rows = adapter.build_parse_coverage(
            records, ["salary"], min_coverage=0.95,
            min_coverage_per_attribute={"salary": 0.1})
        assert rows[0]["enabled"] is True
        assert rows[0]["min_parse_coverage"] == 0.1

    def test_no_enabled_attribute_raises(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        for rec in records:
            rec.fields["salary"] = "NA"
        with pytest.raises(ValueError, match="min_parse_coverage"):
            adapter.to_associations(
                records, _cfg_parquet(mllmu_parquet)
                | {"deterministic_attributes": ["salary"]})


# =====================================================================
# Deterministic association building (Iteration 4)
# =====================================================================

class TestAssociations:
    @pytest.fixture
    def built(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        cfg = _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")}
        adapter.materialize_images(records, cfg)
        assocs = adapter.to_associations(records, cfg)
        return adapter, records, assocs

    def test_one_association_per_enabled_attribute(self, built):
        adapter, records, assocs = built
        assert len(assocs) == len(records) * 5

    def test_target_levels_in_valid_range(self, built):
        _, _, assocs = built
        for a in assocs:
            assert 1 <= a.target_level < a.num_levels()

    def test_target_selection_reproducible(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        cfg = _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")}
        adapter.materialize_images(records, cfg)
        a1 = adapter.to_associations(records, cfg)
        a2 = adapter.to_associations(records, cfg)
        assert [x.target_level for x in a1] == [x.target_level for x in a2]

    def test_location_chains_have_variable_depth(self, built):
        _, _, assocs = built
        loc = [a for a in assocs if a.attribute_name == "residence"]
        # two-component fixture values -> exactly two levels
        assert all(a.num_levels() == 2 for a in loc)
        for a in loc:
            assert a.levels[1].value in a.levels[0].value  # coarser is substring

    def test_date_chains_depth_three(self, built):
        _, _, assocs = built
        for a in (x for x in assocs if x.attribute_name == "date_of_birth"):
            assert a.num_levels() == 3  # date -> year -> decade

    def test_height_values_normalized_to_cm(self, built):
        """Fixture heights are imperial ('5 feet 5 inches') — level 0 must
        carry the normalized centimetre value."""
        _, _, assocs = built
        heights = [a for a in assocs if a.attribute_name == "height"]
        assert heights
        for a in heights:
            assert a.levels[0].value.endswith("cm")

    def test_every_association_has_materialized_image(self, built):
        _, _, assocs = built
        for a in assocs:
            assert len(a.images) == 1
            img = a.images[0]
            assert Path(img.path).exists()
            assert img.split in ("train", "val", "test")

    def test_entity_split_consistent_across_attributes(self, built):
        _, _, assocs = built
        from collections import defaultdict
        per_entity = defaultdict(set)
        for a in assocs:
            per_entity[a.entity_id].add(a.split.split)
        for entity, splits in per_entity.items():
            assert len(splits) == 1, f"{entity} spans splits {splits}"

    def test_provenance_builder_deterministic(self, built):
        _, _, assocs = built
        for a in assocs:
            assert a.provenance.hierarchy_builder == "deterministic"

    def test_retain_attributes_exclude_forgotten_one(self, built):
        _, _, assocs = built
        for a in assocs:
            assert a.attribute_name not in a.retain_attribute_names

    def test_usable_profiles_reported(self, mllmu_parquet, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        cfg = _cfg_parquet(mllmu_parquet) | {"images_dir": str(tmp_path / "img")}
        adapter.materialize_images(records, cfg)
        # one profile has no usable salary; keep salary enabled via a low
        # threshold so the per-profile exclusion is observable
        cfg = cfg | {"min_parse_coverage": 0.5}
        records[0].fields["salary"] = "NA"
        adapter.to_associations(records, cfg)
        report = adapter.last_association_report
        assert report["usable_profiles_per_attribute"]["salary"] == len(records) - 1
        assert report["unusable"]["salary:unparseable"] == 1
