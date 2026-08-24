"""Dataset build CLI.

Usage
-----
    python scripts/build_dataset.py \\
        --config configs/datasets/inaturalist.yaml \\
        --preset smoke

    python scripts/build_dataset.py \\
        --config configs/datasets/inaturalist.yaml \\
        --output data/processed/inaturalist/smoke_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from granunlearn.config import load_yaml, resolve_config
from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.report import generate_report, save_report
from granunlearn.hierarchy.validate import validate_chain
from granunlearn.logging_utils import setup_logger, save_json
from granunlearn.seed import set_all_seeds

log = setup_logger("build_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GMUL dataset")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--preset", default=None,
                        help="Build preset: 'smoke' or 'pilot' (overrides config values)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: data/processed/<name>/<version>/)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--stage", default="full",
                        help="Build stage: 'full' (default) or 'inventory'")
    parser.add_argument("--synthetic-images", action="store_true",
                        help="Generate synthetic placeholder images (for proof-of-concept)")
    args = parser.parse_args()

    # Load config
    cfg = resolve_config(args.config)
    ds_cfg = cfg.get("dataset", {})

    dataset_name = ds_cfg.get("name", "unknown")
    version = ds_cfg.get("version", "v1")

    # Apply preset overrides
    if args.preset == "smoke":
        smoke_cfg = cfg.get("smoke", {})
        ds_cfg.update(smoke_cfg)
        version = "smoke_v1"
    elif args.preset == "pilot":
        pilot_cfg = cfg.get("pilot", {})
        ds_cfg.update(pilot_cfg)
        version = "pilot_v1"

    if args.seed is not None:
        ds_cfg["seed"] = args.seed

    seed = ds_cfg.get("seed", 42)
    set_all_seeds(seed)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("data/processed") / dataset_name / version

    log.info("Building dataset: %s / %s", dataset_name, version)
    log.info("Output: %s", output_dir)
    log.info("Config: %s", args.config)

    # Inventory stage (MLLMU-specific, not used for iNaturalist)
    if args.stage == "inventory":
        log.info("Inventory stage not applicable for %s", dataset_name)
        sys.exit(0)

    # Get adapter
    adapter = get_adapter(dataset_name)

    # Load raw records
    log.info("Loading raw records...")
    raw_records = adapter.load_raw(ds_cfg)
    log.info("Loaded %d raw records", len(raw_records))

    # Convert to associations
    ds_cfg["version"] = version
    log.info("Building associations...")
    associations = adapter.to_associations(raw_records, ds_cfg)
    log.info("Built %d associations", len(associations))

    # Validate all hierarchies
    log.info("Validating hierarchies...")
    total_errors = 0
    total_warnings = 0
    for assoc in associations:
        issues = validate_chain(assoc.levels)
        for issue in issues:
            if issue.is_error:
                total_errors += 1
                log.error("Validation error in %s: %s", assoc.association_id, issue)
            else:
                total_warnings += 1

    if total_errors > 0:
        log.error("Build FAILED: %d validation errors", total_errors)
        sys.exit(1)

    log.info("Validation passed: 0 errors, %d warnings", total_warnings)

    # Generate synthetic images if requested
    if args.synthetic_images:
        from granunlearn.datasets.inaturalist import generate_synthetic_images
        log.info("Generating synthetic images...")
        generate_synthetic_images(output_dir.parent.parent.parent, associations)
        log.info("Synthetic images generated")

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # associations.parquet
    import pandas as pd
    assoc_dicts = [json.loads(a.model_dump_json()) for a in associations]
    df = pd.DataFrame(assoc_dicts)
    df.to_parquet(output_dir / "associations.parquet", index=False)
    log.info("Saved associations.parquet (%d rows)", len(df))

    # hierarchy.jsonl
    with open(output_dir / "hierarchy.jsonl", "w") as f:
        for assoc in associations:
            for lv in assoc.levels:
                f.write(lv.model_dump_json() + "\n")
    log.info("Saved hierarchy.jsonl")

    # entities.parquet
    entities = [
        {"entity_id": a.entity_id, "entity_name": a.entity_name, "dataset": a.dataset}
        for a in associations
    ]
    edf = pd.DataFrame(entities).drop_duplicates()
    edf.to_parquet(output_dir / "entities.parquet", index=False)
    log.info("Saved entities.parquet (%d entities)", len(edf))

    # manifest.json
    manifest = {
        "dataset": dataset_name,
        "version": version,
        "seed": seed,
        "num_associations": len(associations),
        "num_entities": len(edf),
        "config_path": str(args.config),
        "output_dir": str(output_dir),
    }
    save_json(manifest, output_dir / "manifest.json")
    log.info("Saved manifest.json")

    # Report
    report = generate_report(
        dataset=dataset_name,
        version=version,
        associations=associations,
        validation_errors=total_errors,
        validation_warnings=total_warnings,
    )
    save_report(report, output_dir)
    log.info("Saved dataset_report.json + dataset_report.md")

    # Summary
    log.info("Build complete: %s / %s", dataset_name, version)
    log.info("  Entities: %d", len(edf))
    log.info("  Associations: %d", len(associations))
    log.info("  Validation: 0 errors, %d warnings", total_warnings)

    # Print report to stdout
    split_counts = report["train_val_test_counts"]
    log.info("  Splits: train=%d val=%d test=%d",
             split_counts.get("train", 0),
             split_counts.get("val", 0),
             split_counts.get("test", 0))


if __name__ == "__main__":
    main()
