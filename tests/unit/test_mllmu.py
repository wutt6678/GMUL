"""Unit tests for the MLLMU-Bench source adapter (Iteration 3)."""

from __future__ import annotations

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

    def test_to_associations_deferred(self, mllmu_parquet):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg_parquet(mllmu_parquet))
        with pytest.raises(NotImplementedError):
            adapter.to_associations(records, {})
