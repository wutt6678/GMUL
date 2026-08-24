"""Dataset report generation (spec §21).

Every build produces a ``dataset_report.json`` with required fields
and a human-readable Markdown summary.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.schema import AssociationRecord


def generate_report(
    dataset: str,
    version: str,
    associations: list[AssociationRecord],
    validation_errors: int = 0,
    validation_warnings: int = 0,
) -> dict[str, Any]:
    """Produce the required dataset_report.json structure.

    Parameters
    ----------
    dataset : str
        Dataset name.
    version : str
        Version string (e.g. "smoke_v1").
    associations : list[AssociationRecord]
        All built association records.
    validation_errors : int
        Number of blocking validation errors.
    validation_warnings : int
        Number of non-blocking validation warnings.

    Returns
    -------
    dict
        The report structure as specified in the plan §21.
    """
    entities = {a.entity_id for a in associations}
    hierarchy_type_counts = Counter(a.hierarchy_type for a in associations)
    attribute_counts = Counter(a.attribute_name for a in associations)

    depth_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    for a in associations:
        depth_counter[str(a.num_levels())] += 1
        target_counter[str(a.target_level)] += 1

    split_counter: Counter[str] = Counter()
    for a in associations:
        split_counter[a.split.split] += 1

    images_per_entity: dict[str, int] = {}
    for a in associations:
        images_per_entity[a.entity_id] = (
            images_per_entity.get(a.entity_id, 0) + len(a.images)
        )

    qwen_generated = sum(
        1 for a in associations
        if a.provenance.generation_model is not None
    )

    report: dict[str, Any] = {
        "dataset": dataset,
        "version": version,
        "num_entities": len(entities),
        "num_associations": len(associations),
        "num_queries": 0,  # populated after query generation
        "hierarchy_type_counts": dict(hierarchy_type_counts),
        "attribute_counts": dict(attribute_counts),
        "hierarchy_depth_histogram": dict(depth_counter),
        "target_level_histogram": dict(target_counter),
        "route_counts": {},  # populated after query generation
        "train_val_test_counts": dict(split_counter),
        "images_per_entity": images_per_entity,
        "qwen_generated_fraction": (
            qwen_generated / len(associations) if associations else 0.0
        ),
        "validation": {
            "errors": validation_errors,
            "warnings": validation_warnings,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def save_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Write dataset_report.json and dataset_report.md to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "dataset_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown summary
    md_path = output_dir / "dataset_report.md"
    md_path.write_text(_render_markdown(report))

    return json_path


def _render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary from the report dict."""
    lines = [
        f"# Dataset Report: {report['dataset']} / {report['version']}",
        "",
        f"Generated: {report.get('generated_at', 'N/A')}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Entities | {report['num_entities']} |",
        f"| Associations | {report['num_associations']} |",
        f"| Queries | {report['num_queries']} |",
        f"| Qwen-generated fraction | {report['qwen_generated_fraction']:.2%} |",
        "",
        "## Hierarchy Types",
        "",
        "| Type | Count |",
        "|---|---|",
    ]
    for ht, count in report.get("hierarchy_type_counts", {}).items():
        lines.append(f"| {ht} | {count} |")

    lines += [
        "",
        "## Attributes",
        "",
        "| Attribute | Count |",
        "|---|---|",
    ]
    for attr, count in report.get("attribute_counts", {}).items():
        lines.append(f"| {attr} | {count} |")

    lines += [
        "",
        "## Split Distribution",
        "",
        "| Split | Count |",
        "|---|---|",
    ]
    for split, count in report.get("train_val_test_counts", {}).items():
        lines.append(f"| {split} | {count} |")

    lines += [
        "",
        "## Validation",
        "",
        f"- Errors: **{report['validation']['errors']}**",
        f"- Warnings: {report['validation']['warnings']}",
        "",
    ]
    return "\n".join(lines)
