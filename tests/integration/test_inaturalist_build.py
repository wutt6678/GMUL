"""Integration test: iNaturalist full build pipeline.

    real local metadata fixture + real image files
    → adapter (local mode)
    → associations
    → parquet / jsonl / manifest / report
    → reload
    → validate_dataset_dir
    → zero errors

Run with:
    pytest tests/integration/test_inaturalist_build.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
for p in (str(TESTS_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fixtures.inat_fixture import SMOKE_TAXONOMY, write_local_dataset  # noqa: E402

from build_dataset import run_build  # noqa: E402
from validate_dataset import validate_dataset_dir  # noqa: E402


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    """Write a local fixture dataset + run the full build once."""
    base = tmp_path_factory.mktemp("inat_integration")
    data_root = base / "data" / "raw" / "inaturalist" / "local_v1"
    write_local_dataset(data_root, taxonomy=SMOKE_TAXONOMY[:10], images_per_species=12)

    # Write a config pointing at the fixture dataset
    config_path = base / "inaturalist_test.yaml"
    config_path.write_text(
        f"""
dataset:
  name: inaturalist
  version: smoke_v1
  source_mode: local
  data_root: {data_root}
  annotations_file: annotations.json
  min_images_per_species: 10
  target_level: 1
  seed: 42

smoke:
  max_species: 10
"""
    )

    output_dir = base / "data" / "processed" / "inaturalist" / "smoke_v1"
    out_dir, report, associations = run_build(
        config_path, preset="smoke", output_dir=output_dir
    )
    return {"output_dir": out_dir, "report": report, "associations": associations}


class TestBuildPipeline:
    def test_build_produces_all_artifacts(self, built_dataset):
        out = built_dataset["output_dir"]
        for fname in [
            "associations.parquet",
            "hierarchy.jsonl",
            "entities.parquet",
            "manifest.json",
            "dataset_report.json",
            "dataset_report.md",
        ]:
            assert (out / fname).exists(), f"Missing artifact: {fname}"

    def test_build_counts(self, built_dataset):
        assert len(built_dataset["associations"]) == 10
        report = built_dataset["report"]
        assert report["num_entities"] == 10
        assert report["num_associations"] == 10
        assert report["validation"]["errors"] == 0

    def test_image_split_counts(self, built_dataset):
        """All three splits populated at the image level."""
        counts = built_dataset["report"]["train_val_test_counts"]
        assert counts.get("train", 0) > 0
        assert counts.get("val", 0) > 0
        assert counts.get("test", 0) > 0

    def test_manifest_records_data_root(self, built_dataset):
        manifest = json.loads(
            (built_dataset["output_dir"] / "manifest.json").read_text()
        )
        assert manifest["data_root"] is not None
        assert Path(manifest["data_root"]).exists()
        assert manifest["min_images_per_species"] == 10
        assert manifest["num_image_references"] == sum(
            len(a.images) for a in built_dataset["associations"]
        )
        # iNaturalist: every image reference is a distinct physical image
        assert manifest["num_unique_images"] == manifest["num_image_references"]


class TestStandaloneValidation:
    def test_validate_zero_errors(self, built_dataset):
        """The authoritative validator passes on the freshly built dataset."""
        errors, warnings = validate_dataset_dir(built_dataset["output_dir"])
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_detects_missing_image(self, built_dataset, tmp_path):
        """Deleting a referenced image flips validation to FAILED."""
        import shutil

        out = built_dataset["output_dir"]
        corrupt_dir = tmp_path / "corrupt"
        shutil.copytree(out, corrupt_dir)

        # Find a referenced image, delete it, then validate
        assoc = built_dataset["associations"][0]
        victim = Path(assoc.images[0].path)
        sibling = Path(assoc.images[1].path)
        assert victim.exists()
        victim.unlink()
        try:
            errors, _ = validate_dataset_dir(corrupt_dir)
            assert any(e.startswith("IMAGE_MISSING") for e in errors), errors
        finally:
            # Restore the fixture image so the module dataset stays intact
            shutil.copy2(sibling, victim)

    def test_validate_detects_missing_dir(self):
        errors, _ = validate_dataset_dir("/nonexistent/path")
        assert errors

    def test_validate_detects_missing_files(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        errors, _ = validate_dataset_dir(empty)
        assert any("MISSING_FILE" in e for e in errors)
