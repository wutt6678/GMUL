"""Dataset build CLI.

Usage
-----
    python scripts/build_dataset.py \\
        --config configs/datasets/inaturalist.yaml \\
        --preset smoke

The core logic is exposed as ``run_build()`` so tests can drive it
without subprocesses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root, resolve_config
from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.report import generate_report, save_report
from granunlearn.hierarchy.validate import validate_chain
from granunlearn.logging_utils import setup_logger, save_json
from granunlearn.schema import AssociationRecord
from granunlearn.seed import set_all_seeds

log = setup_logger("build_dataset")


def run_build(
    config_path: str | Path,
    preset: str | None = None,
    output_dir: str | Path | None = None,
    seed: int | None = None,
) -> tuple[Path, dict[str, Any], list[AssociationRecord]]:
    """Run the full dataset build.

    Returns
    -------
    (output_dir, report, associations)
    """
    cfg = resolve_config(config_path)
    ds_cfg = dict(cfg.get("dataset", {}))

    dataset_name = ds_cfg.get("name", "unknown")
    version = ds_cfg.get("version", "v1")

    if preset == "smoke":
        ds_cfg.update(cfg.get("smoke", {}))
        version = "smoke_v1"
    elif preset == "pilot":
        ds_cfg.update(cfg.get("pilot", {}))
        version = "pilot_v1"

    if seed is not None:
        ds_cfg["seed"] = seed
    seed = ds_cfg.get("seed", 42)
    set_all_seeds(seed)

    if output_dir is None:
        output_dir = Path("data/processed") / dataset_name / version
    output_dir = Path(output_dir)

    log.info("Building dataset: %s / %s → %s", dataset_name, version, output_dir)

    adapter = get_adapter(dataset_name)

    log.info("Loading raw records...")
    raw_records = adapter.load_raw(ds_cfg)
    load_report = getattr(adapter, "last_load_report", {})
    excluded = load_report.get("excluded_species", [])
    log.info(
        "Loaded %d raw records (%d species excluded by min-image filter; "
        "%d genera, %d families in selected subset)",
        len(raw_records), len(excluded),
        load_report.get("num_genera", 0), load_report.get("num_families", 0),
    )

    ds_cfg["version"] = version
    log.info("Building associations...")
    associations = adapter.to_associations(raw_records, ds_cfg)
    log.info("Built %d associations", len(associations))

    # ---- Hierarchy validation ----
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
        raise ValueError(f"Build failed: {total_errors} hierarchy validation errors")
    log.info("Validation passed: 0 errors, %d warnings", total_warnings)

    # ---- Write artifacts ----
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    assoc_dicts = [json.loads(a.model_dump_json()) for a in associations]
    df = pd.DataFrame(assoc_dicts)
    df.to_parquet(output_dir / "associations.parquet", index=False)
    log.info("Saved associations.parquet (%d rows)", len(df))

    with open(output_dir / "hierarchy.jsonl", "w") as f:
        for assoc in associations:
            for lv in assoc.levels:
                f.write(lv.model_dump_json() + "\n")
    log.info("Saved hierarchy.jsonl")

    entities = [
        {"entity_id": a.entity_id, "entity_name": a.entity_name, "dataset": a.dataset}
        for a in associations
    ]
    edf = pd.DataFrame(entities).drop_duplicates()
    edf.to_parquet(output_dir / "entities.parquet", index=False)
    log.info("Saved entities.parquet (%d entities)", len(edf))

    # Resolve data_root for the manifest (so the validator can find images)
    repo_root = _find_repo_root(Path.cwd())
    data_root_cfg = ds_cfg.get("data_root")
    data_root_abs = None
    if data_root_cfg:
        dr = Path(data_root_cfg)
        data_root_abs = str(dr if dr.is_absolute() else (repo_root / dr)) if repo_root else str(dr)

    manifest = {
        "dataset": dataset_name,
        "version": version,
        "seed": seed,
        "num_associations": len(associations),
        "num_entities": len(edf),
        "num_images": sum(len(a.images) for a in associations),
        "min_images_per_species": ds_cfg.get("min_images_per_species"),
        "excluded_species": excluded,
        "num_genera": load_report.get("num_genera"),
        "num_families": load_report.get("num_families"),
        "data_root": data_root_abs,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }
    save_json(manifest, output_dir / "manifest.json")
    log.info("Saved manifest.json")

    report = generate_report(
        dataset=dataset_name,
        version=version,
        associations=associations,
        validation_errors=total_errors,
        validation_warnings=total_warnings,
    )
    save_report(report, output_dir)
    log.info("Saved dataset_report.json + dataset_report.md")

    split_counts = report["train_val_test_counts"]
    log.info("Build complete: %d entities, %d associations | splits: %s",
             len(edf), len(associations), split_counts)

    return output_dir, report, associations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GMUL dataset")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--preset", default=None, help="'smoke' or 'pilot'")
    parser.add_argument("--output", default=None, help="Output directory override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--stage", default="full", help="'full' or 'inventory'")
    args = parser.parse_args()

    if args.stage == "inventory":
        log.info("Inventory stage not supported for this dataset")
        sys.exit(0)

    run_build(args.config, preset=args.preset, output_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
