"""Integration test: adapter consumes the OFFICIAL MLLMU-Bench release.

Runs against ``data/raw/MLLMU-Bench`` (the official Hugging Face parquet
layout) when present, and is skipped otherwise.  This is the in-repo
evidence that the adapter directly consumes the released dataset rather
than a locally converted JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from granunlearn.config import _find_repo_root
from granunlearn.datasets.mllmu import MLLMUAdapter, OFFICIAL_PARQUET

REPO_ROOT = _find_repo_root(Path(__file__).resolve())
RAW_ROOT = REPO_ROOT / "data" / "raw" / "MLLMU-Bench"
PARQUET = RAW_ROOT / OFFICIAL_PARQUET

pytestmark = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="official MLLMU-Bench parquet not downloaded "
           "(huggingface-cli download MLLMMU/MLLMU-Bench)",
)


@pytest.fixture(scope="module")
def loaded():
    adapter = MLLMUAdapter()
    records = adapter.load_raw({
        "data_root": str(RAW_ROOT),
        "annotations_file": OFFICIAL_PARQUET,
        "source_format": "parquet",
    })
    return adapter, records


def test_official_full_set_parses_completely(loaded):
    adapter, records = loaded
    report = adapter.last_load_report
    assert report["num_source_records"] == 500
    assert report["num_parsed"] == 500
    assert report["num_errors"] == 0


def test_official_directory_is_image_locator(loaded):
    _, records = loaded
    for rec in records:
        # Directory column, e.g. "full_images/001.jpg" — never HF image bytes
        assert rec.image_path.startswith("full_images/"), rec.raw_id
        assert rec.image_path.endswith((".jpg", ".jpeg", ".png"))


def test_official_entity_ids_unique_and_reproducible(loaded):
    adapter, records = loaded
    ids = [r.entity_id for r in records]
    assert len(set(ids)) == 500
    # Re-load: identical order and IDs (deterministic ingestion)
    records2 = adapter.load_raw({
        "data_root": str(RAW_ROOT),
        "annotations_file": OFFICIAL_PARQUET,
        "source_format": "parquet",
    })
    assert [r.entity_id for r in records2] == ids


def test_official_inventory_matches_committed_evidence(loaded):
    """The committed data/reports evidence must describe this release."""
    adapter, records = loaded
    rows = adapter.build_inventory(records)

    evidence_csv = REPO_ROOT / "data" / "reports" / "mllmu_hier_attribute_inventory.csv"
    evidence_json = REPO_ROOT / "data" / "reports" / "mllmu_hier_inventory_summary.json"
    if not (evidence_csv.exists() and evidence_json.exists()):
        pytest.skip("committed inventory evidence not present")

    import csv
    with open(evidence_csv) as f:
        committed = {r["attribute"]: r for r in csv.DictReader(f)}
    assert set(committed) == {r["attribute"] for r in rows}
    for row in rows:
        assert committed[row["attribute"]]["count"] == str(row["count"]), row["attribute"]
        assert float(committed[row["attribute"]]["coverage"]) == row["coverage"]

    summary = json.loads(evidence_json.read_text())
    assert summary["num_source_records"] == 500
    assert summary["num_parsed"] == 500
    assert summary["source"] == "MLLMMU/MLLMU-Bench"
    assert summary["source_revision"]  # non-empty revision recorded
