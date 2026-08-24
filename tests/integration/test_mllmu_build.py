"""Integration test: MLLMU deterministic hierarchy build pipeline (Iteration 4).

    synthetic official-schema parquet fixture (real PNG bytes)
    -> adapter load_raw (parquet)
    -> materialize_images (embedded bytes -> disk, PIL-verified)
    -> real parse-coverage gate
    -> deterministic AssociationRecords (variable-depth location chains)
    -> parquet / jsonl / manifest / parse_coverage evidence
    -> validate_dataset_dir (hierarchy + image gates)
    -> zero errors

Note: synthetic test profiles only — not the real MLLMU-Bench release and
not a research proof-of-concept result (see
tests/integration/test_mllmu_official_release.py for the real-release gate).
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

from fixtures.mllmu_fixture import write_parquet  # noqa: E402

from build_dataset import run_build  # noqa: E402
from validate_dataset import validate_dataset_dir  # noqa: E402

N_PROFILES = 20


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory):
    base = tmp_path_factory.mktemp("mllmu_integration")
    data_root = base / "data" / "raw" / "MLLMU-Bench"
    write_parquet(data_root, n_records=N_PROFILES)
    images_dir = base / "data" / "processed" / "mllmu_hier" / "images"

    config_path = base / "mllmu_test.yaml"
    config_path.write_text(
        f"""
dataset:
  name: mllmu_hier
  version: det_test_v1
  source_mode: local
  data_root: {data_root}
  annotations_file: Full_Set/train-00000-of-00001.parquet
  source_format: parquet
  seed: 42
  materialize_images: true
  images_dir: {images_dir}
  split_mode: entity_level
  min_parse_coverage: 0.95
  deterministic_attributes:
    - residence
    - birthplace
    - date_of_birth
    - salary
    - height
"""
    )
    output_dir = base / "data" / "processed" / "mllmu_hier" / "det_test_v1"
    out_dir, report, associations = run_build(config_path, output_dir=output_dir)
    return {"output_dir": out_dir, "report": report, "associations": associations,
            "images_dir": images_dir}


class TestBuildPipeline:
    def test_all_artifacts_present(self, built_dataset):
        out = built_dataset["output_dir"]
        for fname in [
            "associations.parquet", "hierarchy.jsonl", "entities.parquet",
            "manifest.json", "dataset_report.json", "dataset_report.md",
            "parse_coverage.json", "association_build_report.json",
        ]:
            assert (out / fname).exists(), f"Missing artifact: {fname}"

    def test_five_associations_per_profile(self, built_dataset):
        # all fixture attributes parse -> 5 deterministic associations each
        assert len(built_dataset["associations"]) == N_PROFILES * 5

    def test_images_materialized_and_exist(self, built_dataset):
        images = list(built_dataset["images_dir"].glob("mllmu_*.png")) + \
                 list(built_dataset["images_dir"].glob("mllmu_*.jpg"))
        assert len(images) == N_PROFILES
        for a in built_dataset["associations"]:
            assert len(a.images) == 1
            assert Path(a.images[0].path).exists()

    def test_parse_coverage_evidence_written(self, built_dataset):
        rows = json.loads(
            (built_dataset["output_dir"] / "parse_coverage.json").read_text())
        by_attr = {r["attribute"]: r for r in rows}
        assert set(by_attr) == {"residence", "birthplace", "date_of_birth",
                                "salary", "height"}
        for r in rows:
            assert r["total_profiles"] == N_PROFILES
            assert r["parse_success"] == N_PROFILES
            assert r["enabled"] is True

    def test_usable_profiles_reported(self, built_dataset):
        report = json.loads(
            (built_dataset["output_dir"] / "association_build_report.json").read_text())
        usable = report["usable_profiles_per_attribute"]
        assert all(usable[a] == N_PROFILES for a in
                   ["residence", "birthplace", "date_of_birth", "salary", "height"])

    def test_manifest_split_mode_and_counts(self, built_dataset):
        manifest = json.loads(
            (built_dataset["output_dir"] / "manifest.json").read_text())
        assert manifest["split_mode"] == "entity_level"
        assert manifest["num_associations"] == N_PROFILES * 5
        assert manifest["num_entities"] == N_PROFILES
        assert manifest["num_images"] == N_PROFILES * 5  # one per association


class TestStandaloneValidation:
    def test_validate_zero_errors(self, built_dataset):
        errors, warnings = validate_dataset_dir(built_dataset["output_dir"])
        assert errors == [], f"Validation errors: {errors}"

    def test_entity_level_train_and_test_present(self, built_dataset):
        counts = built_dataset["report"]["train_val_test_counts"]
        assert counts.get("train", 0) > 0
        assert counts.get("test", 0) > 0

    def test_validate_detects_deleted_image(self, built_dataset, tmp_path):
        import shutil
        out = built_dataset["output_dir"]
        corrupt_dir = tmp_path / "corrupt"
        shutil.copytree(out, corrupt_dir)

        victim = Path(built_dataset["associations"][0].images[0].path)
        assert victim.exists()
        victim.unlink()
        try:
            errors, _ = validate_dataset_dir(corrupt_dir)
            assert any(e.startswith("IMAGE_MISSING") for e in errors), errors
        finally:
            # rebuild the image so the module fixture stays intact
            from fixtures.mllmu_fixture import make_png_bytes
            victim.write_bytes(make_png_bytes())

    def test_corrupt_embedded_bytes_fails_build(self, tmp_path):
        """Undecodable embedded image bytes abort the build (hard gate)."""
        base = tmp_path / "badimg"
        data_root = base / "raw"
        write_parquet(data_root, n_records=6, corrupt_image_idx=3)
        config_path = base / "mllmu_bad.yaml"
        config_path.write_text(
            f"""
dataset:
  name: mllmu_hier
  version: det_bad_v1
  data_root: {data_root}
  annotations_file: Full_Set/train-00000-of-00001.parquet
  source_format: parquet
  materialize_images: true
  images_dir: {base / "images"}
  split_mode: entity_level
"""
        )
        with pytest.raises(ValueError, match="materialization failed"):
            run_build(config_path, output_dir=base / "out")
