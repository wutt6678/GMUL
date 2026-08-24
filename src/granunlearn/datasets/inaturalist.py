"""iNaturalist proof-of-concept dataset adapter (spec §12).

For the smoke build, this adapter uses a built-in taxonomy fixture
of 20 real species across multiple genera and families.  No LLM is
required — the hierarchy is purely taxonomic and deterministic.

When real iNaturalist data is available, the adapter can be extended
to load from the HuggingFace ``iNaturalist`` dataset or local files.
"""

from __future__ import annotations

from typing import Any

from granunlearn.hierarchy import build_taxonomic_hierarchy
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    ProvenanceInfo,
    SplitInfo,
)

from .split import deterministic_split

# -----------------------------------------------------------------------
# Built-in taxonomy fixture — 20 real species
# -----------------------------------------------------------------------
# Each entry: {species, genus, family, common_name, n_images}
# Selected for taxonomic diversity: ≥2 genera, ≥2 families.

SMOKE_TAXONOMY: list[dict[str, Any]] = [
    # --- Passeridae (Old World sparrows) ---
    {"species": "Passer domesticus",    "genus": "Passer",      "family": "Passeridae",  "common_name": "House Sparrow",           "n_images": 12},
    {"species": "Passer montanus",      "genus": "Passer",      "family": "Passeridae",  "common_name": "Eurasian Tree Sparrow",   "n_images": 10},
    {"species": "Passer hispaniolensis","genus": "Passer",      "family": "Passeridae",  "common_name": "Spanish Sparrow",         "n_images": 8},
    # --- Corvidae (crows, jays) ---
    {"species": "Corvus corax",         "genus": "Corvus",      "family": "Corvidae",    "common_name": "Common Raven",            "n_images": 15},
    {"species": "Corvus corone",        "genus": "Corvus",      "family": "Corvidae",    "common_name": "Carrion Crow",            "n_images": 12},
    {"species": "Cyanistes caeruleus",  "genus": "Cyanistes",   "family": "Paridae",     "common_name": "Eurasian Blue Tit",       "n_images": 11},
    # --- Paridae (tits) ---
    {"species": "Parus major",          "genus": "Parus",       "family": "Paridae",     "common_name": "Great Tit",               "n_images": 14},
    {"species": "Periparus ater",       "genus": "Periparus",   "family": "Paridae",     "common_name": "Coal Tit",                "n_images": 9},
    # --- Turdidae (thrushes) ---
    {"species": "Turdus merula",        "genus": "Turdus",      "family": "Turdidae",    "common_name": "Common Blackbird",        "n_images": 13},
    {"species": "Turdus philomelos",    "genus": "Turdus",      "family": "Turdidae",    "common_name": "Song Thrush",             "n_images": 10},
    # --- Anatidae (ducks, geese) ---
    {"species": "Anas platyrhynchos",   "genus": "Anas",        "family": "Anatidae",    "common_name": "Mallard",                 "n_images": 16},
    {"species": "Anas crecca",          "genus": "Anas",        "family": "Anatidae",    "common_name": "Eurasian Teal",           "n_images": 8},
    {"species": "Branta canadensis",    "genus": "Branta",      "family": "Anatidae",    "common_name": "Canada Goose",            "n_images": 14},
    # --- Lepidoptera (butterflies) ---
    {"species": "Vanessa cardui",       "genus": "Vanessa",     "family": "Nymphalidae", "common_name": "Painted Lady",            "n_images": 11},
    {"species": "Vanessa atalanta",     "genus": "Vanessa",     "family": "Nymphalidae", "common_name": "Red Admiral",             "n_images": 10},
    {"species": "Pieris rapae",         "genus": "Pieris",      "family": "Pieridae",    "common_name": "Small White",             "n_images": 9},
    {"species": "Pieris brassicae",     "genus": "Pieris",      "family": "Pieridae",    "common_name": "Large White",             "n_images": 8},
    # --- Mammals ---
    {"species": "Vulpes vulpes",        "genus": "Vulpes",      "family": "Canidae",     "common_name": "Red Fox",                 "n_images": 15},
    {"species": "Canis lupus",          "genus": "Canis",       "family": "Canidae",     "common_name": "Gray Wolf",               "n_images": 12},
    {"species": "Felis catus",          "genus": "Felis",       "family": "Felidae",     "common_name": "Domestic Cat",            "n_images": 18},
]


class INaturalistAdapter:
    """iNaturalist dataset adapter.

    Implements the ``DatasetAdapter`` protocol.
    """

    def name(self) -> str:
        return "inaturalist"

    def load_raw(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Load raw taxonomy records.

        For the smoke build, uses the built-in ``SMOKE_TAXONOMY`` fixture.
        ``config["max_species"]`` limits how many species to include.
        """
        max_species = config.get("max_species", 20)
        data = SMOKE_TAXONOMY[:max_species]

        if len(data) < max_species:
            # Not an error for smoke — just a note
            pass

        return data

    def to_associations(
        self,
        raw_records: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[AssociationRecord]:
        """Convert taxonomy records into canonical ``AssociationRecord`` objects.

        Each species becomes one association with a taxonomic hierarchy:
        species (level 0) → genus (level 1) → family (level 2).

        The target level defaults to 1 (genus) — the unlearning task is
        to reduce species-level knowledge to genus-level while retaining
        family-level recognition.
        """
        seed = config.get("seed", 42)
        target_level = config.get("target_level", 1)  # default: reduce to genus
        dataset_version = config.get("version", "smoke_v1")

        # Build entity-level splits
        entity_ids = [r["species"] for r in raw_records]
        splits = deterministic_split(entity_ids, seed=seed)

        associations: list[AssociationRecord] = []
        for record in raw_records:
            species = record["species"]
            genus = record["genus"]
            family = record["family"]
            common_name = record.get("common_name", species)

            # Build the taxonomic hierarchy chain
            chain = build_taxonomic_hierarchy([
                {"name": species, "rank": "species", "taxon_id": None},
                {"name": genus, "rank": "genus", "taxon_id": None},
                {"name": family, "rank": "family", "taxon_id": None},
            ], prefix="tax")

            # Validate the chain
            issues = chain.validate()
            errors = [i for i in issues if i.is_error]
            if errors:
                raise ValueError(
                    f"Taxonomy validation failed for {species}: "
                    + "; ".join(str(e) for e in errors)
                )

            # Create image references (placeholder paths)
            n_images = record.get("n_images", 5)
            images = [
                ImageRef(
                    image_id=f"inat_{species.replace(' ', '_')}_img{i:03d}",
                    path=f"data/raw/inaturalist/{dataset_version}/{species.replace(' ', '_')}/{i:03d}.jpg",
                    source="original",
                )
                for i in range(n_images)
            ]

            # Build the association record
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
                split=splits[species],
                provenance=ProvenanceInfo(
                    source_dataset="inaturalist",
                    source_entity_id=species,
                    hierarchy_builder="deterministic",
                ),
            )
            associations.append(assoc)

        return associations
