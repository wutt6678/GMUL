"""Dataset validation CLI (spec §20).

The standalone validator is AUTHORITATIVE.  It re-runs every check on the
persisted artifacts:

    processed data
    → deserialize AssociationRecords
    → validate_chain() on every hierarchy
    → image existence
    → within-entity split coverage (train AND test present)
    → image ID/path uniqueness across splits
    → minimum image counts
    → manifest counts exact match
    → PASS

Usage
-----
    python scripts/validate_dataset.py \\
        --data data/processed/inaturalist/smoke_v1

Exits non-zero if any blocking error is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from granunlearn.config import _find_repo_root
from granunlearn.hierarchy.validate import validate_chain
from granunlearn.logging_utils import setup_logger
from granunlearn.schema import AssociationRecord

log = setup_logger("validate_dataset")

REQUIRED_FILES = [
    "associations.parquet",
    "hierarchy.jsonl",
    "entities.parquet",
    "manifest.json",
]


def _resolve_image_path(path_str: str, data_dir: Path) -> Path:
    """Resolve an image path: absolute as-is, relative against repo root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    repo_root = _find_repo_root(Path(data_dir).resolve())
    if repo_root is not None:
        return repo_root / p
    return Path.cwd() / p


def _deserialize_record(row: dict[str, Any]) -> AssociationRecord:
    record_dict = dict(row)
    for key in ("levels", "split", "provenance", "images", "source_modalities",
                "textual_context", "retain_attribute_names"):
        if isinstance(record_dict.get(key), str):
            record_dict[key] = json.loads(record_dict[key])
    return AssociationRecord.model_validate(record_dict)


def validate_dataset_dir(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Validate a built dataset directory.

    Returns
    -------
    (errors, warnings)
        Lists of human-readable gate violations.
    """
    data_dir = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not data_dir.is_dir():
        return [f"MISSING_DIR: {data_dir}"], []

    # ---- Gate 0: required files -----------------------------------------
    for fname in REQUIRED_FILES:
        if not (data_dir / fname).exists():
            errors.append(f"MISSING_FILE: {fname}")
    if errors:
        return errors, warnings

    manifest = json.loads((data_dir / "manifest.json").read_text())

    # ---- Gate 1: deserialize records ------------------------------------
    df = pd.read_parquet(data_dir / "associations.parquet")
    log.info("Loaded %d associations", len(df))

    associations: list[AssociationRecord] = []
    for idx, row in df.iterrows():
        try:
            associations.append(_deserialize_record(row.to_dict()))
        except Exception as e:
            errors.append(f"RECORD_INVALID[row={idx}]: {e}")

    # ---- Gate 2: duplicate association IDs -------------------------------
    if "association_id" in df.columns:
        dupes = int(df["association_id"].duplicated().sum())
        if dupes > 0:
            errors.append(f"DUPLICATE_IDS: {dupes} duplicate association_ids")

    # ---- Gate 3: hierarchy validation (re-run, authoritative) ------------
    for assoc in associations:
        issues = validate_chain(assoc.levels)
        for issue in issues:
            if issue.is_error:
                errors.append(f"HIERARCHY_INVALID[{assoc.association_id}]: {issue}")

    # ---- Gate 4: image gates ---------------------------------------------
    split_mode = manifest.get("split_mode", "within_entity")
    # image_id / path → (entity_id, assoc, split).  Reuse ACROSS entities is
    # an error; reuse across associations of the SAME entity is legitimate
    # (one profile photo supports several attributes of the same person).
    seen_image_ids: dict[str, tuple[str, str, str]] = {}
    seen_image_paths: dict[str, tuple[str, str, str]] = {}
    missing_images = 0
    total_images = 0
    dataset_image_splits: set[str] = set()

    for assoc in associations:
        split_image_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for img in assoc.images:
            total_images += 1
            if img.split:
                dataset_image_splits.add(img.split)

            # 4a. existence
            if not _resolve_image_path(img.path, data_dir).exists():
                missing_images += 1
                if missing_images <= 5:
                    errors.append(f"IMAGE_MISSING[{assoc.association_id}]: {img.path}")

            # 4b. cross-ENTITY uniqueness (ID and path)
            if img.image_id in seen_image_ids:
                prev_entity, prev_assoc, prev_split = seen_image_ids[img.image_id]
                if prev_entity != assoc.entity_id:
                    errors.append(
                        f"IMAGE_ID_REUSE[{assoc.association_id}]: {img.image_id!r} "
                        f"already used by entity {prev_entity} ({prev_assoc}, "
                        f"split {prev_split})"
                    )
            else:
                seen_image_ids[img.image_id] = (
                    assoc.entity_id, assoc.association_id, img.split or "?")

            if img.path in seen_image_paths:
                prev_entity, prev_assoc, prev_split = seen_image_paths[img.path]
                if prev_entity != assoc.entity_id:
                    errors.append(
                        f"IMAGE_PATH_REUSE[{assoc.association_id}]: {img.path!r} "
                        f"already used by entity {prev_entity} ({prev_assoc}, "
                        f"split {prev_split})"
                    )
            else:
                seen_image_paths[img.path] = (
                    assoc.entity_id, assoc.association_id, img.split or "?")

            if img.split is None:
                warnings.append(
                    f"IMAGE_NO_SPLIT[{assoc.association_id}]: {img.image_id} has no split"
                )
            else:
                split_image_ids[img.split].append(img.image_id)

        # 4c. split coverage — semantics depend on the declared split mode
        if split_mode == "within_entity" and len(assoc.images) > 0:
            if not split_image_ids["train"]:
                errors.append(
                    f"NO_TRAIN_IMAGES[{assoc.association_id}]: "
                    f"entity has {len(assoc.images)} images but none in train split"
                )
            if not split_image_ids["test"]:
                errors.append(
                    f"NO_TEST_IMAGES[{assoc.association_id}]: "
                    f"entity has {len(assoc.images)} images but none in test split"
                )

    # 4c-bis. entity_level mode: train AND test must exist DATASET-WIDE
    # (each entity carries a single image assigned to one split).
    if split_mode == "entity_level":
        if total_images > 0:
            if "train" not in dataset_image_splits:
                errors.append("DATASET_NO_TRAIN_IMAGES: no image in train split")
            if "test" not in dataset_image_splits:
                errors.append("DATASET_NO_TEST_IMAGES: no image in test split")

        # 4d. minimum image count per entity
        min_images = manifest.get("min_images_per_species")
        if min_images is not None and len(assoc.images) < min_images:
            errors.append(
                f"TOO_FEW_IMAGES[{assoc.association_id}]: "
                f"{len(assoc.images)} < min_images_per_species ({min_images})"
            )

    if missing_images > 0:
        errors.append(f"IMAGE_MISSING_TOTAL: {missing_images}/{total_images} images not found")

    # ---- Gate 5: manifest counts must match EXACTLY ----------------------
    if manifest.get("num_associations") != len(df):
        errors.append(
            f"MANIFEST_ASSOC_MISMATCH: manifest={manifest.get('num_associations')}, "
            f"parquet={len(df)}"
        )
    entities = pd.read_parquet(data_dir / "entities.parquet")
    if manifest.get("num_entities") != len(entities):
        errors.append(
            f"MANIFEST_ENTITY_MISMATCH: manifest={manifest.get('num_entities')}, "
            f"parquet={len(entities)}"
        )
    if manifest.get("num_images") is not None and manifest["num_images"] != total_images:
        errors.append(
            f"MANIFEST_IMAGE_MISMATCH: manifest={manifest.get('num_images')}, "
            f"records={total_images}"
        )

    # ---- Gate 6: hierarchy.jsonl well-formedness --------------------------
    n_levels = 0
    with open(data_dir / "hierarchy.jsonl") as f:
        for line_num, line in enumerate(f, 1):
            try:
                json.loads(line)
                n_levels += 1
            except json.JSONDecodeError as e:
                errors.append(f"MALFORMED_JSON[line={line_num}]: {e}")
    expected_levels = sum(len(a.levels) for a in associations)
    if n_levels != expected_levels:
        errors.append(
            f"HIERARCHY_COUNT_MISMATCH: hierarchy.jsonl has {n_levels} levels, "
            f"associations contain {expected_levels}"
        )

    log.info("Validated %d hierarchy levels, %d images", n_levels, total_images)
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a built GMUL dataset")
    parser.add_argument("--data", required=True, help="Path to processed dataset directory")
    args = parser.parse_args()

    errors, warnings = validate_dataset_dir(args.data)

    for w in warnings:
        log.warning(w)
    if errors:
        for e in errors:
            log.error(e)
        log.error("Validation FAILED: %d errors, %d warnings", len(errors), len(warnings))
        sys.exit(1)
    log.info("Validation PASSED: 0 errors, %d warnings", len(warnings))
    sys.exit(0)


if __name__ == "__main__":
    main()
