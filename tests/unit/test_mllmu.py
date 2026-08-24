"""Unit tests for the MLLMU-Bench source adapter (Iteration 3)."""

from __future__ import annotations

import pytest

from fixtures.mllmu_fixture import write_mllmu_jsonl

from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.mllmu import (
    ATTRIBUTE_POLICY,
    FIELD_ALIASES,
    MLLMUAdapter,
    normalize_field,
)


@pytest.fixture
def mllmu_jsonl(tmp_path):
    path = tmp_path / "mllmu" / "Full_Set.jsonl"
    write_mllmu_jsonl(path, n_records=5)
    return path


def _cfg(path, **overrides):
    cfg = {
        "data_root": str(path.parent),
        "annotations_file": path.name,
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
# Loading + provenance
# =====================================================================

class TestLoading:
    def test_loads_all_records(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        assert len(records) == 5

    def test_every_record_has_provenance(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        for rec in records:
            assert rec.provenance is not None
            assert rec.provenance.source_dataset == "mllmu_bench"
            assert rec.provenance.source_entity_id is not None

    def test_entity_ids_normalized(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        ids = [rec.entity_id for rec in records]
        assert len(set(ids)) == len(ids), "entity IDs must be unique"
        assert all(i.startswith("mllmu_") for i in ids)

    def test_field_names_normalized_on_load(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        fields = records[0].fields
        assert "height" in fields        # from "Heights"
        assert "salary" in fields        # from "Annual Salary"
        assert "education" in fields     # from "Educated at"
        assert "Heights" not in fields

    def test_load_report_counts(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        adapter.load_raw(_cfg(mllmu_jsonl))
        report = adapter.last_load_report
        assert report["num_total"] == 5
        assert report["num_parsed"] == 5
        assert report["num_parse_errors"] == 0


# =====================================================================
# Celebrity exclusion
# =====================================================================

class TestCelebrityExclusion:
    def test_celebrity_excluded_for_fictitious_subset(self, tmp_path):
        path = tmp_path / "celeb" / "Full_Set.jsonl"
        write_mllmu_jsonl(path, n_records=3, include_celebrity=True)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(path))
        assert len(records) == 3  # celebrity not included
        report = adapter.last_load_report
        assert report["num_celebrity_skipped"] == 1
        names = {rec.entity_name for rec in records}
        assert "Famous Star" not in names

    def test_celebrity_included_when_subset_not_fictitious(self, tmp_path):
        path = tmp_path / "all" / "Full_Set.jsonl"
        write_mllmu_jsonl(path, n_records=3, include_celebrity=True)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(path, subset="all"))
        assert len(records) == 4


# =====================================================================
# Parse-error accounting
# =====================================================================

class TestParseErrors:
    def test_corrupt_record_counted(self, tmp_path):
        path = tmp_path / "corrupt" / "Full_Set.jsonl"
        # 1/40 = 2.5% stays under the 5% threshold
        write_mllmu_jsonl(path, n_records=40, corrupt_biography_idx=3)
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(path))
        assert len(records) == 39
        assert adapter.last_load_report["num_parse_errors"] == 1

    def test_high_error_rate_raises(self, tmp_path):
        path = tmp_path / "bad" / "Full_Set.jsonl"
        write_mllmu_jsonl(path, n_records=4, corrupt_biography_idx=0)
        # 1/4 = 25% > 5% threshold
        adapter = MLLMUAdapter()
        with pytest.raises(ValueError, match="parse error rate"):
            adapter.load_raw(_cfg(path))

    def test_missing_source_raises(self, tmp_path):
        adapter = MLLMUAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.load_raw(_cfg(tmp_path / "nope" / "Full_Set.jsonl"))


# =====================================================================
# Attribute inventory
# =====================================================================

class TestInventory:
    def test_core_attributes_classified(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}

        for core in ["residence", "birthplace", "date_of_birth",
                     "occupation", "salary", "height", "education"]:
            assert by_attr[core]["include_core"] is True, core

    def test_unsupported_explicitly_excluded(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}

        # Unsupported attrs present in inventory but excluded with a reason
        for attr in ["gender", "medical_conditions", "parents",
                     "fun_facts", "description", "name"]:
            assert attr in by_attr, f"{attr} silently dropped"
            assert by_attr[attr]["include_core"] is False
            assert by_attr[attr]["notes"], f"{attr} has no exclusion reason"

    def test_hierarchy_type_assignment(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        rows = adapter.build_inventory(records)
        by_attr = {r["attribute"]: r for r in rows}
        assert by_attr["date_of_birth"]["suggested_hierarchy_type"] == "numeric"
        assert by_attr["salary"]["suggested_hierarchy_type"] == "numeric"
        assert by_attr["occupation"]["suggested_hierarchy_type"] == "semantic"
        assert by_attr["occupation"]["qwen_needed"] is True
        assert by_attr["date_of_birth"]["deterministic_possible"] is True

    def test_write_inventory_csv(self, mllmu_jsonl, tmp_path):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        rows = adapter.build_inventory(records)
        out = adapter.write_inventory_csv(rows, tmp_path / "inv.csv")
        assert out.exists()
        import csv
        with open(out) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == len(rows)
        assert set(reader[0].keys()) == {
            "attribute", "count", "example_values", "suggested_hierarchy_type",
            "deterministic_possible", "qwen_needed", "include_core", "notes",
        }

    def test_unknown_attribute_flagged(self, tmp_path):
        """Attributes outside the policy are excluded with an explicit note."""
        import json as _json
        path = tmp_path / "unk" / "Full_Set.jsonl"
        path.parent.mkdir(parents=True)
        bio = {"Name": "X", "Mystery Field": "value"}
        with open(path, "w") as f:
            f.write(_json.dumps({"ID": "001", "biography": _json.dumps(bio)}) + "\n")
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(path))
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

    def test_to_associations_deferred(self, mllmu_jsonl):
        adapter = MLLMUAdapter()
        records = adapter.load_raw(_cfg(mllmu_jsonl))
        with pytest.raises(NotImplementedError):
            adapter.to_associations(records, {})
