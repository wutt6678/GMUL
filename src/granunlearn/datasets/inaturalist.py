"""iNaturalist dataset adapter (spec §12).

Reads actual iNaturalist-format metadata (COCO-style ``annotations.json``
with ``images`` / ``annotations`` / ``categories``) from a configurable
**local dataset root**.  No downloading is performed — the operator must
place the data on disk.

Design points
-------------
* The taxonomy hierarchy (species → genus → family) comes from the source
  ``categories`` metadata and is validated against the authoritative
  Linnaean rank order — never LLM-generated.
* Images are split WITHIN each species (60/20/20 by default), with
  deterministic assignment that *guarantees* at least one train and one
  test image per species (when enough images exist).  This lets the model
  be trained on ``I_train(species)→species`` and evaluated on held-out
  ``I_test(species)→species/genus/family``.
* Species with fewer than ``min_images_per_species`` images are excluded.
* Species selection is deterministic (sorted by species name).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.hierarchy import build_taxonomic_hierarchy
from granunlearn.schema import (
    AssociationRecord,
    ImageRef,
    ProvenanceInfo,
    SplitInfo,
)


# -----------------------------------------------------------------------
# Within-species image split assignment
# -----------------------------------------------------------------------

def deterministic_image_splits(
    n_images: int,
    seed: int,
    entity_id: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> list[str]:
    """Assign each of *n_images* images to train/val/test deterministically.

    Guarantees (when enough images are available):
    * at least one test image (n >= 2)
    * at least one validation image (n >= 4)
    * at least one train image (always, since train gets the remainder)

    The assignment is a deterministic hash-shuffle of indices followed by
    positional slicing, so it is reproducible and order-independent.
    """
    if n_images <= 0:
        return []

    # Deterministic pseudo-shuffle of indices
    order = sorted(
        range(n_images),
        key=lambda i: hashlib.sha256(f"{seed}:{entity_id}:{i}".encode()).hexdigest(),
    )

    # Quotas with guaranteed minimums
    n_test = max(1, round(n_images * (1.0 - train_ratio - val_ratio))) if n_images >= 2 else 0
    n_val = max(1, round(n_images * val_ratio)) if n_images >= 4 else 0
    # Protect the training set
    if n_images - n_test - n_val < 1:
        n_val = max(0, n_images - n_test - 1)
    n_train = n_images - n_test - n_val

    split_of_index: dict[int, str] = {}
    cursor = 0
    for idx in order[cursor:cursor + n_train]:
        split_of_index[idx] = "train"
    cursor += n_train
    for idx in order[cursor:cursor + n_val]:
        split_of_index[idx] = "val"
    cursor += n_val
    for idx in order[cursor:]:
        split_of_index[idx] = "test"

    return [split_of_index[i] for i in range(n_images)]


# -----------------------------------------------------------------------
# Local COCO-style metadata loading
# -----------------------------------------------------------------------

def _resolve_data_root(data_root: str | Path) -> Path:
    """Resolve *data_root*: absolute paths are used as-is; relative paths
    are resolved against the repository root."""
    p = Path(data_root)
    if p.is_absolute():
        return p
    repo_root = _find_repo_root(Path.cwd())
    if repo_root is None:
        raise FileNotFoundError(
            f"Cannot resolve relative data_root {data_root!r}: no repo root found"
        )
    return repo_root / p


def load_local_metadata(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load species records from a local iNaturalist-format dataset.

    Config keys (under the ``dataset`` section):

    * ``data_root`` — directory containing the dataset
    * ``annotations_file`` — JSON filename, default ``annotations.json``
    * ``min_images_per_species`` — int, default 10
    * ``max_species`` — int, default 20
    * ``min_genera`` — int, default 1 (hard gate on the selected subset)
    * ``min_families`` — int, default 1 (hard gate on the selected subset)

    Returns
    -------
    (records, load_report)
        ``records`` — one dict per selected species.  ``load_report``
        carries bookkeeping: ``excluded_species``, ``eligible_species``,
        ``num_genera``, ``num_families``.

    Raises
    ------
    ValueError
        If the selected subset does not satisfy ``min_genera`` /
        ``min_families``.
    """
    root = _resolve_data_root(config["data_root"])
    ann_file = root / config.get("annotations_file", "annotations.json")
    min_images = int(config.get("min_images_per_species", 10))
    max_species = int(config.get("max_species", 20))
    min_genera = int(config.get("min_genera", 1))
    min_families = int(config.get("min_families", 1))

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")

    with open(ann_file) as f:
        meta = json.load(f)

    categories = {c["id"]: c for c in meta["categories"]}
    images_by_id = {im["id"]: im for im in meta["images"]}

    # Group image file names by species.  The official iNaturalist schema
    # uses "name" (not "species") on categories; support both.
    species_images: dict[str, list[str]] = {}
    species_meta: dict[str, dict[str, Any]] = {}
    for ann in meta["annotations"]:
        cat = categories.get(ann["category_id"])
        img = images_by_id.get(ann["image_id"])
        if cat is None or img is None:
            continue
        sp = cat.get("species") or cat.get("name", "")
        if not sp:
            continue
        species_images.setdefault(sp, []).append(img["file_name"])
        species_meta.setdefault(sp, cat)

    # Filter by minimum image count, deterministic selection
    eligible = sorted(
        sp for sp, files in species_images.items() if len(files) >= min_images
    )
    excluded = sorted(set(species_images) - set(eligible))
    selected = eligible[:max_species]

    # ---- Hard gates: taxonomic diversity of the selected subset ----------
    if not selected:
        raise ValueError(
            f"No species satisfy min_images_per_species={min_images}; "
            f"cannot build a dataset (total species seen: {len(species_images)})"
        )
    genera = {species_meta[sp].get("genus", "") for sp in selected}
    families = {species_meta[sp].get("family", "") for sp in selected}
    if len(genera) < min_genera:
        raise ValueError(
            f"Selected subset has {len(genera)} genera, "
            f"min_genera requires {min_genera}"
        )
    if len(families) < min_families:
        raise ValueError(
            f"Selected subset has {len(families)} families, "
            f"min_families requires {min_families}"
        )

    records: list[dict[str, Any]] = []
    for sp in selected:
        cat = species_meta[sp]
        records.append({
            "species": sp,
            "genus": cat.get("genus", ""),
            "family": cat.get("family", ""),
            "order": cat.get("order", ""),
            "common_name": cat.get("common_name", sp),
            "images": sorted(species_images[sp]),
        })

    load_report = {
        "excluded_species": excluded,
        "eligible_species": len(eligible),
        "num_genera": len(genera),
        "num_families": len(families),
    }
    return records, load_report


# -----------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------

class INaturalistAdapter:
    """iNaturalist dataset adapter (local-data mode only)."""

    def __init__(self) -> None:
        # Populated by load_raw(); consumed by the build script for reporting.
        self.last_load_report: dict[str, Any] = {}

    def name(self) -> str:
        return "inaturalist"

    def load_raw(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Load species records from the local dataset root.

        Requires ``config["data_root"]``.  The full load report (excluded
        species, diversity counts) is stored on ``self.last_load_report``.
        """
        if "data_root" not in config:
            raise ValueError(
                "iNaturalist adapter requires 'data_root' pointing to a local "
                "iNaturalist-format dataset (annotations.json + images/)"
            )
        records, report = load_local_metadata(config)
        self.last_load_report = report
        return records

    def to_associations(
        self,
        raw_records: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[AssociationRecord]:
        """Convert species records into canonical ``AssociationRecord`` objects.

        Each species becomes one association with a taxonomic hierarchy
        species (level 0) → genus (level 1) → family (level 2).  Images are
        split within the species via ``deterministic_image_splits``.
        """
        seed = config.get("seed", 42)
        target_level = config.get("target_level", 1)
        data_root = _resolve_data_root(config["data_root"])
        repo_root = _find_repo_root(Path.cwd())

        associations: list[AssociationRecord] = []
        for record in raw_records:
            species = record["species"]
            genus = record["genus"]
            family = record["family"]
            common_name = record.get("common_name", species)

            chain = build_taxonomic_hierarchy([
                {"name": species, "rank": "species"},
                {"name": genus, "rank": "genus"},
                {"name": family, "rank": "family"},
            ], prefix="tax")

            issues = chain.validate()
            errors = [i for i in issues if i.is_error]
            if errors:
                raise ValueError(
                    f"Taxonomy validation failed for {species}: "
                    + "; ".join(str(e) for e in errors)
                )

            # Within-species image splitting
            file_names: list[str] = record.get("images", [])
            splits = deterministic_image_splits(len(file_names), seed, species)

            images: list[ImageRef] = []
            for i, fname in enumerate(file_names):
                full = (data_root / fname).resolve()
                # Store repo-relative path when possible for portability
                try:
                    stored = str(full.relative_to(repo_root)) if repo_root else str(full)
                except ValueError:
                    stored = str(full)
                images.append(ImageRef(
                    image_id=f"inat_{species.replace(' ', '_')}_{Path(fname).stem}",
                    path=stored,
                    source="original",
                    split=splits[i],
                ))

            assoc = AssociationRecord(
                association_id=f"inat_{species.replace(' ', '_')}",
                dataset="inaturalist",
                entity_id=species,
                entity_name=common_name,
                attribute_name="taxonomic_classification",
                hierarchy_type="taxonomic",
                levels=chain.levels(),
                original_level=0,
                target_level=target_level,
                source_modalities=["image"],
                images=images,
                textual_context=[
                    f"{common_name} ({species}) belongs to genus {genus}, family {family}."
                ],
                retain_attribute_names=[],
                split=SplitInfo(split="train"),  # entity-level knowledge is trained
                provenance=ProvenanceInfo(
                    source_dataset="inaturalist",
                    source_entity_id=species,
                    hierarchy_builder="deterministic",
                ),
            )
            associations.append(assoc)

        return associations
