"""Reference-state datasets for SALMU (Iteration 10).

Mirrors the MLLMU counterfactual design at the CLIP association level.
Personas are split into TARGET / RETAIN by deterministic hash; within
each target persona exactly ONE core attribute is the *target attribute*
and the remaining core attributes are *same-entity retain*.  The three
states differ ONLY in how target associations are presented:

* MF^SALMU: target associations with their FINE (released paraphrased)
            captions  +  retain associations with fine captions
* MG^SALMU: target associations with GENERALIZED TARGET captions only
            +  retain associations with fine captions
* MN^SALMU: target associations OMIT  +  retain fine captions

Same-entity retain (non-target attributes of target personas) are
ALWAYS presented with their fine captions in ALL states, including MN.
This is the SALMU analog of SALMUBench's ``holdout_association`` split
and allows testing selective preservation of within-identity knowledge.

Fine captions come from the RELEASED SALMU caption set (never our own
wording for MF); generalized captions come from the controlled
templates in ``hierarchy.py``.  Evaluation splits of SALMUBench
(forget/retain_synth/holdout_*) are untouched and remain evaluation-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.hierarchy import generalized_caption

log = setup_logger("salmu_state_datasets")

STATES = ("MF", "MG", "MN")


class SalmuTrainingPair(BaseModel):
    """One (image, caption) contrastive pair for a reference state."""

    pair_id: str
    state: str
    identity_id: str
    attribute: str = Field(description="city | job | blood_type")
    role: str = Field(description="target_association | same_entity_retain | other_entity_retain")
    level_index: int = Field(description="caption granularity level")
    caption: str
    caption_source: str = Field(
        description="released_fine | generalized_template")
    image_file: str | None = None


def partition_personas(
    identity_ids: list[str],
    num_targets: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic target/retain persona split (sha256 ranking),
    same mechanism as the MLLMU F/R partition."""
    if num_targets < 0 or num_targets > len(identity_ids):
        raise ValueError("num_targets out of range")

    def key(iid: str) -> str:
        return hashlib.sha256(
            f"{seed}:target:{iid}".encode()).hexdigest()

    ranked = sorted(identity_ids, key=key)
    target_ids = sorted(ranked[:num_targets])
    retain_ids = sorted(ranked[num_targets:])
    return {
        "seed": seed,
        "num_targets": num_targets,
        "num_retain": len(retain_ids),
        "target_identity_ids": target_ids,
        "retain_identity_ids": retain_ids,
    }


def partition_persona_attributes(
    target_identity_ids: list[str],
    core_attributes: tuple[str, ...] = ("city", "job", "blood_type"),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each target persona exactly ONE target attribute.

    The remaining core attributes for that persona become *same-entity
    retain* — they keep their fine captions in ALL states (including
    MN), mirroring SALMUBench's ``holdout_association`` design.

    Uses deterministic sha256 ranking for ordering, then round-robin
    assignment to guarantee balanced per-attribute allocation (e.g.
    20/20/20 for 60 targets and 3 attributes).

    Returns ``{identity_id: target_attribute}``.
    """
    attrs = list(core_attributes)
    n_attrs = len(attrs)
    # Deterministic ordering via sha256 ranking
    ranked = sorted(
        target_identity_ids,
        key=lambda iid: hashlib.sha256(
            f"{seed}:attr_target:{iid}".encode()).hexdigest())
    # Round-robin assignment for perfect balance
    assignment: dict[str, str] = {}
    for idx, iid in enumerate(ranked):
        assignment[iid] = attrs[idx % n_attrs]
    return assignment


def build_state_pairs(
    state: str,
    partition: dict[str, Any],
    hierarchies: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    fine_captions_by_identity_attr: dict[str, dict[str, list[str]]],
    images_by_identity: dict[str, list[str]],
    target_attr_map: dict[str, str] | None = None,
) -> list[SalmuTrainingPair]:
    """Build D_state. Deterministic ordering throughout.

    ``target_attr_map`` (from ``partition_persona_attributes``):
    ``{identity_id: target_attribute}``.  When provided, only the
    target attribute of each target persona is treated as "target";
    the remaining attributes are "retain" (same-entity retain, always
    present with fine captions in ALL states including MN).

    ``fine_captions_by_identity_attr``: identity_id -> attribute ->
    released fine captions.  ``images_by_identity``: identity_id ->
    image file names (each image pairs with the persona's captions).
    """
    if state not in STATES:
        raise ValueError(f"Unknown SALMU reference state: {state!r}")
    targets = set(partition["target_identity_ids"])
    pairs: list[SalmuTrainingPair] = []
    for iid in sorted(hierarchies):
        is_target_persona = iid in targets
        name = identities[iid]["name"]
        images = sorted(images_by_identity.get(iid, []))
        for attr in sorted(hierarchies[iid]):
            hier = hierarchies[iid][attr]
            # Per-attribute targeting with explicit role distinction:
            # - target_association: the ONE target attribute of a target persona
            # - same_entity_retain: non-target attributes of target personas
            # - other_entity_retain: all attributes of non-target personas
            if is_target_persona and target_attr_map is not None:
                is_target_attr = (attr == target_attr_map[iid])
            else:
                is_target_attr = False
            
            if is_target_attr:
                role = "target_association"
            elif is_target_persona:
                role = "same_entity_retain"
            else:
                role = "other_entity_retain"
            
            # MN omits target_association pairs but keeps same_entity_retain
            if role == "target_association" and state == "MN":
                continue
            if role != "target_association" or state == "MF":
                level_index = 0
                captions = fine_captions_by_identity_attr.get(
                    iid, {}).get(attr, [])
                source = "released_fine"
            else:  # target_association under MG
                level_index = hier["target_level"]
                captions = [generalized_caption(
                    name, attr, level_index,
                    hier["levels"][level_index])]
                source = "generalized_template"
            for ci, caption in enumerate(sorted(captions)):
                for img in images:
                    pairs.append(SalmuTrainingPair(
                        pair_id=f"{state}__{iid}__{attr}__{ci}__"
                                f"{Path(img).stem}",
                        state=state,
                        identity_id=iid,
                        attribute=attr,
                        role=role,
                        level_index=level_index,
                        caption=caption,
                        caption_source=source,
                        image_file=img,
                    ))
    return pairs


def validate_state_pairs(
    pairs: list[SalmuTrainingPair],
    partition: dict[str, Any],
    state: str,
    target_attr_map: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    targets = set(partition["target_identity_ids"])
    retains = set(partition["retain_identity_ids"])
    seen_identities = {p.identity_id for p in pairs}
    for p in pairs:
        if p.state != state:
            errors.append(f"{p.pair_id}: wrong state label")
        # Validate role assignments
        if p.role == "target_association":
            if p.identity_id not in targets:
                errors.append(f"{p.pair_id}: target_association must be from target persona")
            if target_attr_map is not None:
                expected = target_attr_map.get(p.identity_id)
                if expected is not None and p.attribute != expected:
                    errors.append(
                        f"{p.pair_id}: target_association attribute "
                        f"{p.attribute} != designated {expected}")
        elif p.role == "same_entity_retain":
            if p.identity_id not in targets:
                errors.append(f"{p.pair_id}: same_entity_retain must be from target persona")
            if target_attr_map is not None:
                expected = target_attr_map.get(p.identity_id)
                if expected is not None and p.attribute == expected:
                    errors.append(
                        f"{p.pair_id}: same_entity_retain attribute "
                        f"{p.attribute} should not be the target attribute")
        elif p.role == "other_entity_retain":
            if p.identity_id in targets:
                errors.append(f"{p.pair_id}: other_entity_retain must be from non-target persona")
        else:
            errors.append(f"{p.pair_id}: invalid role {p.role}")
        # Validate caption sources for retain roles
        if p.role in ("same_entity_retain", "other_entity_retain"):
            if p.level_index != 0:
                errors.append(f"{p.pair_id}: retain must use fine captions (level 0)")
            if p.caption_source != "released_fine":
                errors.append(f"{p.pair_id}: retain captions must be the "
                              f"released fine captions")
        # Validate MG target_association captions
        if p.role == "target_association" and state == "MG" and \
                p.caption_source != "generalized_template":
            errors.append(f"{p.pair_id}: MG target_association must use ONLY "
                          f"generalized target captions")
    # With per-attribute targeting, MN can contain target identities
    # (for their same_entity_retain attributes).  Check that MN has
    # no target_association PAIRS instead.
    if state == "MN":
        target_pairs = [p for p in pairs if p.role == "target_association"]
        if target_pairs:
            errors.append(
                f"MN contains {len(target_pairs)} target_association pairs")
    if state != "MN" and not (targets & seen_identities):
        errors.append(f"{state} missing target identities")
    if not (retains & seen_identities):
        errors.append(f"{state} missing retain identities")
    return errors


def write_state_pairs(
    pairs_by_state: dict[str, list[SalmuTrainingPair]],
    partition: dict[str, Any],
    output_dir: str | Path,
    target_attr_map: dict[str, str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"partition_num_targets":
                                partition["num_targets"], "states": {}}
    if target_attr_map is not None:
        manifest["target_attr_map"] = target_attr_map
    for state, pairs in pairs_by_state.items():
        errors = validate_state_pairs(pairs, partition, state,
                                      target_attr_map)
        if errors:
            raise ValueError(
                f"SALMU state {state} failed validation: {errors[:5]}")
        path = output_dir / f"{state}.jsonl"
        with open(path, "w") as f:
            for p in pairs:
                f.write(p.model_dump_json() + "\n")
        manifest["states"][state] = {
            "num_pairs": len(pairs),
            "num_target_association": sum(1 for p in pairs if p.role == "target_association"),
            "num_same_entity_retain": sum(1 for p in pairs if p.role == "same_entity_retain"),
            "num_other_entity_retain": sum(1 for p in pairs if p.role == "other_entity_retain"),
            "caption_sources": sorted({p.caption_source for p in pairs}),
        }
    with open(output_dir / "state_pairs_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_state_pairs(path: str | Path) -> list[SalmuTrainingPair]:
    return [SalmuTrainingPair.model_validate(json.loads(line))
            for line in Path(path).read_text().splitlines() if line.strip()]
