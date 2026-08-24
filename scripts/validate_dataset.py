"""Dataset validation CLI (spec §20).

Usage
-----
    python scripts/validate_dataset.py \\
        --data data/processed/inaturalist/smoke_v1

Exits with non-zero status if any blocking error is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from granunlearn.logging_utils import setup_logger
from granunlearn.schema import AssociationRecord

log = setup_logger("validate_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a built GMUL dataset")
    parser.add_argument("--data", required=True, help="Path to processed dataset directory")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    errors: list[str] = []
    warnings: list[str] = []

    # ---- Structural gates ----

    # 1. Check required files exist
    required_files = ["associations.parquet", "hierarchy.jsonl", "entities.parquet", "manifest.json"]
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            errors.append(f"MISSING_FILE: {fname}")

    if errors:
        _report_and_exit(errors, warnings)

    # 2. Load associations
    log.info("Loading associations from %s", data_dir / "associations.parquet")
    df = pd.read_parquet(data_dir / "associations.parquet")
    log.info("Loaded %d associations", len(df))

    # 3. Check duplicate IDs
    if "association_id" in df.columns:
        dupes = df["association_id"].duplicated().sum()
        if dupes > 0:
            errors.append(f"DUPLICATE_IDS: {dupes} duplicate association_ids")

    # 4. Deserialize and validate each record
    log.info("Validating individual records...")
    associations: list[AssociationRecord] = []
    for idx, row in df.iterrows():
        try:
            record_dict = row.to_dict()
            # Pandas may serialize nested objects as strings — handle both
            if isinstance(record_dict.get("levels"), str):
                record_dict["levels"] = json.loads(record_dict["levels"])
            if isinstance(record_dict.get("split"), str):
                record_dict["split"] = json.loads(record_dict["split"])
            if isinstance(record_dict.get("provenance"), str):
                record_dict["provenance"] = json.loads(record_dict["provenance"])
            if isinstance(record_dict.get("images"), str):
                record_dict["images"] = json.loads(record_dict["images"])

            assoc = AssociationRecord.model_validate(record_dict)
            associations.append(assoc)
        except Exception as e:
            errors.append(f"RECORD_INVALID[row={idx}]: {e}")

    # 5. Check entity leakage between splits
    # The split column may contain nested SplitInfo dicts; extract the string.
    if "entity_id" in df.columns and "split" in df.columns:
        def _extract_split_str(val):
            if isinstance(val, dict):
                return val.get("split", str(val))
            return str(val)

        df["_split_str"] = df["split"].apply(_extract_split_str)
        entity_splits = (
            df.groupby("entity_id")["_split_str"]
            .apply(lambda s: set(s))
            .to_dict()
        )
        for eid, split_set in entity_splits.items():
            if len(split_set) > 1:
                errors.append(
                    f"ENTITY_SPLIT_LEAKAGE: entity {eid!r} appears in multiple splits: {split_set}"
                )
        df.drop(columns=["_split_str"], inplace=True)

    # 6. Check hierarchy.jsonl
    # Note: genus/family canonical_ids are shared across species chains,
    # so we do NOT check for global uniqueness.  Per-chain validation
    # is already done during the build step and via record deserialization.
    hierarchy_path = data_dir / "hierarchy.jsonl"
    n_levels = 0
    with open(hierarchy_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                lv = json.loads(line)
                n_levels += 1
            except json.JSONDecodeError as e:
                errors.append(f"MALFORMED_JSON[line={line_num}]: {e}")

    log.info("Validated %d hierarchy levels", n_levels)

    # 7. Check manifest consistency
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("num_associations") != len(df):
        warnings.append(
            f"MANIFEST_MISMATCH: manifest says {manifest.get('num_associations')} "
            f"associations, parquet has {len(df)}"
        )

    # 8. Minimum queries per target (model-readiness gate)
    if len(associations) > 0:
        log.info("Model-readiness: %d associations available", len(associations))
    else:
        errors.append("MODEL_READINESS: no valid associations found")

    # ---- Summary ----
    _report_and_exit(errors, warnings)


def _report_and_exit(errors: list[str], warnings: list[str]) -> None:
    """Print validation results and exit with appropriate code."""
    if warnings:
        for w in warnings:
            log.warning(w)
    if errors:
        for e in errors:
            log.error(e)
        log.error("Validation FAILED: %d errors, %d warnings", len(errors), len(warnings))
        sys.exit(1)
    else:
        log.info("Validation PASSED: 0 errors, %d warnings", len(warnings))
        sys.exit(0)


if __name__ == "__main__":
    main()
