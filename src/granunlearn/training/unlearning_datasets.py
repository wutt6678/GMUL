"""Unlearning knowledge groups (Iteration 9).

The MF -> MU baselines transform the SAME state-derived knowledge used
in Iteration 7 — still never derived from evaluation queries.  Three
groups are sufficient for every Iteration 9 method:

* ``fine_target``   target associations at the FINE level (0)
                    -> gradient ascent for B1 (complete forget) and
                       fine suppression for B3;
* ``target_level``  target associations at the target retained level
                    -> coarse-positive rewriting for B2 / B3;
* ``retain``        retain associations at the fine level (0)
                    -> retain SFT for B3.

One controlled template, identical formatting — same contract as the
reference-state datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

from granunlearn.schema import AssociationRecord
from granunlearn.training.state_datasets import (
    TRAIN_COMPLETION_TEMPLATE,
    TRAIN_PROMPT_TEMPLATE,
    TrainingExample,
    _display_attr,
)

GROUPS: list[str] = ["fine_target", "target_level", "retain"]


def build_unlearning_group(
    associations: list[AssociationRecord],
    partition: dict,
    group: str,
) -> list[TrainingExample]:
    """Build one unlearning knowledge group (deterministic order)."""
    if group not in GROUPS:
        raise ValueError(f"Unknown unlearning group: {group!r}")
    target_ids = set(partition["target_association_ids"])
    by_id = {a.association_id: a for a in associations}

    examples: list[TrainingExample] = []
    for aid in sorted(by_id):
        assoc = by_id[aid]
        is_target = aid in target_ids
        if group == "retain" and is_target:
            continue
        if group in ("fine_target", "target_level") and not is_target:
            continue
        if group == "fine_target":
            level_idx = 0
        elif group == "target_level":
            level_idx = assoc.target_level
        else:
            level_idx = 0
        level = assoc.levels[level_idx]
        image_path = None
        if assoc.images:
            p = Path(assoc.images[0].path)
            if p.is_absolute():
                raise ValueError(
                    f"{aid}: absolute image path in association pool — "
                    f"unlearning groups persist repo-relative paths only")
            image_path = p.as_posix()
        name = assoc.entity_name or assoc.entity_id
        examples.append(TrainingExample(
            example_id=f"{group}__{aid}",
            state=group,
            association_id=aid,
            entity_id=assoc.entity_id,
            attribute_name=assoc.attribute_name,
            role="target" if is_target else "retain",
            level_index=level_idx,
            level_value=level.value,
            prompt=TRAIN_PROMPT_TEMPLATE.format(
                name=name, attr=_display_attr(assoc.attribute_name)),
            completion=TRAIN_COMPLETION_TEMPLATE.format(value=level.value),
            image_path=image_path,
            modality="image_text" if image_path else "text",
        ))
    return examples


def validate_unlearning_groups(
    groups: dict[str, list[TrainingExample]],
    partition: dict,
) -> list[str]:
    """Structural invariants across the three groups."""
    errors: list[str] = []
    target_ids = set(partition["target_association_ids"])
    retain_ids = set(partition["retain_association_ids"])

    ft = groups.get("fine_target", [])
    tl = groups.get("target_level", [])
    rt = groups.get("retain", [])
    if {e.association_id for e in ft} != target_ids:
        errors.append("fine_target must cover exactly the target set")
    if {e.association_id for e in tl} != target_ids:
        errors.append("target_level must cover exactly the target set")
    if {e.association_id for e in rt} != retain_ids:
        errors.append("retain must cover exactly the retain set")
    for e in ft:
        if e.level_index != 0:
            errors.append(f"{e.example_id}: fine_target must be level 0")
    for e in rt:
        if e.level_index != 0:
            errors.append(f"{e.example_id}: retain must be level 0")
    tl_by_assoc = {e.association_id: e for e in tl}
    ft_by_assoc = {e.association_id: e for e in ft}
    for aid, e in tl_by_assoc.items():
        fex = ft_by_assoc.get(aid)
        if fex is None:
            continue
        # same controlled template: only the completion may differ
        if e.prompt != fex.prompt:
            errors.append(f"{aid}: template mismatch across groups")
    return errors


def write_unlearning_groups(
    associations: list[AssociationRecord],
    partition: dict,
    output_dir: str | Path,
) -> dict:
    """Write the three group jsonls + manifest. Returns the manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {g: build_unlearning_group(associations, partition, g)
              for g in GROUPS}
    errors = validate_unlearning_groups(groups, partition)
    if errors:
        raise ValueError(f"Unlearning groups failed validation: {errors[:5]}")
    manifest: dict = {"groups": {}}
    for group, examples in groups.items():
        path = output_dir / f"{group}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
        manifest["groups"][group] = {
            "path": str(path),
            "num_examples": len(examples),
        }
    manifest["note"] = (
        "Unlearning knowledge groups derive from the F/R partition and "
        "association hierarchies only — evaluation queries are never "
        "used. Same controlled template as the reference-state datasets."
    )
    with open(output_dir / "unlearning_groups_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest
