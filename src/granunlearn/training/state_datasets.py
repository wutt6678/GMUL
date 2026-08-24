"""Reference-state knowledge datasets (Iteration 7).

Training data is derived BY STATE from the committed association pool and
the F/R partition — NEVER from evaluation queries.  ``queries.split`` is
a paraphrase-evaluation split and grants no permission to train on probe
wording; the behavioral probe families (incl. the adversarial
``negation_correction``) remain evaluation-only.

States
------
MF:  target associations -> fine level (0)
     retain associations -> fine level (0)
MG:  target associations -> target retained level (assoc.target_level)
     retain associations -> fine level (0)
MN:  target associations -> OMIT
     retain associations -> fine level (0)

All states share ONE controlled training template (identical multimodal
formatting), so MF = Train(D_F), MG = Train(D_G), MN = Train(D_N) differ
only in the transformed knowledge dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from granunlearn.schema import AssociationRecord

State = Literal["MF", "MG", "MN"]
STATES: list[str] = ["MF", "MG", "MN"]

# Single controlled template, identical across states.  Only the
# completion (the level value) differs by state.
TRAIN_PROMPT_TEMPLATE = "What is {name}'s {attr}?"
TRAIN_COMPLETION_TEMPLATE = "{value}."

ATTRIBUTE_DISPLAY = {
    "date_of_birth": "date of birth",
    "salary": "annual salary",
    "height": "height",
    "residence": "place of residence",
    "birthplace": "place of birth",
    "occupation": "occupation",
    "education": "educational background",
}


class TrainingExample(BaseModel):
    """One supervised (prompt, completion) pair for a reference state."""

    example_id: str
    state: str = Field(description="MF | MG | MN")
    association_id: str
    entity_id: str
    attribute_name: str
    role: str = Field(description="target | retain")
    level_index: int = Field(description="Hierarchy level being trained")
    level_value: str = Field(description="Exact value at that level")
    prompt: str
    completion: str
    image_path: str | None = None
    modality: str = Field(description="text | image_text")


def _display_attr(attribute_name: str) -> str:
    return ATTRIBUTE_DISPLAY.get(
        attribute_name, attribute_name.replace("_", " "))


def level_index_for_state(assoc: AssociationRecord, state: str) -> int | None:
    """Hierarchy level a state trains for this association (None = omit)."""
    if state == "MN":
        return None
    if state == "MF":
        return 0
    if state == "MG":
        return assoc.target_level
    raise ValueError(f"Unknown reference state: {state!r}")


def build_state_examples(
    associations: list[AssociationRecord],
    partition: dict,
    state: str,
    repo_root: str | Path | None = None,
) -> list[TrainingExample]:
    """Build D_state from the association pool + F/R partition.

    Deterministic ordering (sorted association ids).  Image paths are
    resolved against ``repo_root`` when relative.
    """
    if state not in STATES:
        raise ValueError(f"Unknown reference state: {state!r}")
    target_ids = set(partition["target_association_ids"])
    by_id = {a.association_id: a for a in associations}

    examples: list[TrainingExample] = []
    for aid in sorted(by_id):
        assoc = by_id[aid]
        role = "target" if aid in target_ids else "retain"
        if role == "target" and state == "MN":
            continue  # MN omits target associations entirely
        level_idx = 0 if role == "retain" else level_index_for_state(
            assoc, state)
        assert level_idx is not None
        level = assoc.levels[level_idx]
        image_path = None
        if assoc.images:
            p = Path(assoc.images[0].path)
            if not p.is_absolute() and repo_root is not None:
                p = Path(repo_root) / p
            image_path = str(p)
        name = assoc.entity_name or assoc.entity_id
        examples.append(TrainingExample(
            example_id=f"{state}__{aid}",
            state=state,
            association_id=aid,
            entity_id=assoc.entity_id,
            attribute_name=assoc.attribute_name,
            role=role,
            level_index=level_idx,
            level_value=level.value,
            prompt=TRAIN_PROMPT_TEMPLATE.format(
                name=name, attr=_display_attr(assoc.attribute_name)),
            completion=TRAIN_COMPLETION_TEMPLATE.format(value=level.value),
            image_path=image_path,
            modality="image_text" if image_path else "text",
        ))
    return examples


def validate_state_examples(
    examples: list[TrainingExample],
    partition: dict,
    state: str,
) -> list[str]:
    """Structural invariants for a state dataset. Returns errors."""
    errors: list[str] = []
    target_ids = set(partition["target_association_ids"])
    retain_ids = set(partition["retain_association_ids"])
    for ex in examples:
        if ex.state != state:
            errors.append(f"{ex.example_id}: wrong state label")
        if ex.role == "target" and ex.association_id not in target_ids:
            errors.append(f"{ex.example_id}: role/target mismatch")
        if ex.role == "retain" and ex.association_id not in retain_ids:
            errors.append(f"{ex.example_id}: role/retain mismatch")
        if ex.role == "retain" and ex.level_index != 0:
            errors.append(
                f"{ex.example_id}: retain examples must train the fine "
                f"level in every state")
        if ex.level_value not in ex.completion:
            errors.append(f"{ex.example_id}: completion lacks level value")
    seen_assoc = {ex.association_id for ex in examples}
    if state == "MN":
        overlap = seen_assoc & target_ids
        if overlap:
            errors.append(f"MN contains target associations: {overlap}")
    else:
        missing = target_ids - seen_assoc
        if missing:
            errors.append(f"{state} missing target associations: {missing}")
    missing_retain = retain_ids - seen_assoc
    if missing_retain:
        errors.append(f"{state} missing retain associations: {missing_retain}")
    return errors


def write_state_datasets(
    associations: list[AssociationRecord],
    partition: dict,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict:
    """Write D_MF / D_MG / D_MN jsonl + manifest. Returns the manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"states": {}}
    for state in STATES:
        examples = build_state_examples(
            associations, partition, state, repo_root=repo_root)
        errors = validate_state_examples(examples, partition, state)
        if errors:
            raise ValueError(
                f"State dataset {state} failed validation: {errors[:5]}")
        path = output_dir / f"{state}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(ex.model_dump_json() + "\n")
        manifest["states"][state] = {
            "path": str(path),
            "num_examples": len(examples),
            "num_target": sum(1 for e in examples if e.role == "target"),
            "num_retain": sum(1 for e in examples if e.role == "retain"),
            "num_image_text": sum(
                1 for e in examples if e.modality == "image_text"),
            "target_level_distribution": {
                str(idx): sum(1 for e in examples
                              if e.role == "target" and e.level_index == idx)
                for idx in sorted({e.level_index for e in examples
                                   if e.role == "target"})
            },
        }
    manifest["training_template"] = {
        "prompt": TRAIN_PROMPT_TEMPLATE,
        "completion": TRAIN_COMPLETION_TEMPLATE,
        "note": (
            "One controlled template shared by all states; MF/MG/MN differ "
            "ONLY in the transformed knowledge dataset.  Evaluation query "
            "families (incl. adversarial negation_correction) are never "
            "used for training."
        ),
    }
    with open(output_dir / "state_datasets_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_state_examples(path: str | Path) -> list[TrainingExample]:
    return [TrainingExample.model_validate(json.loads(line))
            for line in Path(path).read_text().splitlines() if line.strip()]
