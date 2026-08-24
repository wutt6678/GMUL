"""iNaturalist test fixtures.

``SMOKE_TAXONOMY`` is a TEST fixture — 20 real species with full Linnaean
metadata.  Production code must NOT import from this module.

``write_local_dataset`` materialises a COCO-style iNaturalist dataset on
disk (annotations.json + real JPEG files) so the adapter's *local* data
mode can be exercised in both unit and integration tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------
# Taxonomy fixture — 20 real species (birds / butterflies / mammals)
# -----------------------------------------------------------------------

SMOKE_TAXONOMY: list[dict[str, Any]] = [
    # --- Passeriformes, Passeridae ---
    {"species": "Passer domesticus",     "genus": "Passer",     "family": "Passeridae",  "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "House Sparrow"},
    {"species": "Passer montanus",       "genus": "Passer",     "family": "Passeridae",  "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Eurasian Tree Sparrow"},
    {"species": "Passer hispaniolensis", "genus": "Passer",     "family": "Passeridae",  "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Spanish Sparrow"},
    # --- Passeriformes, Corvidae ---
    {"species": "Corvus corax",          "genus": "Corvus",     "family": "Corvidae",    "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Common Raven"},
    {"species": "Corvus corone",         "genus": "Corvus",     "family": "Corvidae",    "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Carrion Crow"},
    # --- Passeriformes, Paridae ---
    {"species": "Cyanistes caeruleus",   "genus": "Cyanistes",  "family": "Paridae",     "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Eurasian Blue Tit"},
    {"species": "Parus major",           "genus": "Parus",      "family": "Paridae",     "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Great Tit"},
    {"species": "Periparus ater",        "genus": "Periparus",  "family": "Paridae",     "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Coal Tit"},
    # --- Passeriformes, Turdidae ---
    {"species": "Turdus merula",         "genus": "Turdus",     "family": "Turdidae",    "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Common Blackbird"},
    {"species": "Turdus philomelos",     "genus": "Turdus",     "family": "Turdidae",    "order": "Passeriformes", "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Song Thrush"},
    # --- Anseriformes, Anatidae ---
    {"species": "Anas platyrhynchos",    "genus": "Anas",       "family": "Anatidae",    "order": "Anseriformes",  "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Mallard"},
    {"species": "Anas crecca",           "genus": "Anas",       "family": "Anatidae",    "order": "Anseriformes",  "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Eurasian Teal"},
    {"species": "Branta canadensis",     "genus": "Branta",     "family": "Anatidae",    "order": "Anseriformes",  "class": "Aves",     "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Canada Goose"},
    # --- Lepidoptera, Nymphalidae / Pieridae ---
    {"species": "Vanessa cardui",        "genus": "Vanessa",    "family": "Nymphalidae", "order": "Lepidoptera",   "class": "Insecta",  "phylum": "Arthropoda", "kingdom": "Animalia", "common_name": "Painted Lady"},
    {"species": "Vanessa atalanta",      "genus": "Vanessa",    "family": "Nymphalidae", "order": "Lepidoptera",   "class": "Insecta",  "phylum": "Arthropoda", "kingdom": "Animalia", "common_name": "Red Admiral"},
    {"species": "Pieris rapae",          "genus": "Pieris",     "family": "Pieridae",    "order": "Lepidoptera",   "class": "Insecta",  "phylum": "Arthropoda", "kingdom": "Animalia", "common_name": "Small White"},
    {"species": "Pieris brassicae",      "genus": "Pieris",     "family": "Pieridae",    "order": "Lepidoptera",   "class": "Insecta",  "phylum": "Arthropoda", "kingdom": "Animalia", "common_name": "Large White"},
    # --- Carnivora, Canidae / Felidae ---
    {"species": "Vulpes vulpes",         "genus": "Vulpes",     "family": "Canidae",     "order": "Carnivora",     "class": "Mammalia", "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Red Fox"},
    {"species": "Canis lupus",           "genus": "Canis",      "family": "Canidae",     "order": "Carnivora",     "class": "Mammalia", "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Gray Wolf"},
    {"species": "Felis catus",           "genus": "Felis",      "family": "Felidae",     "order": "Carnivora",     "class": "Mammalia", "phylum": "Chordata",   "kingdom": "Animalia", "common_name": "Domestic Cat"},
]


def write_local_dataset(
    root: Path,
    taxonomy: list[dict[str, Any]] | None = None,
    images_per_species: int = 12,
    image_size: tuple[int, int] = (64, 64),
    seed: int = 0,
) -> dict[str, Any]:
    """Materialise a COCO-style iNaturalist dataset on disk.

    Layout::

        root/
        ├── annotations.json       # images / annotations / categories
        └── images/<Species_dir>/<idx:03d>.jpg

    Parameters
    ----------
    root : Path
        Dataset root to create.
    taxonomy : list[dict] | None
        Taxonomy entries; defaults to ``SMOKE_TAXONOMY``.
    images_per_species : int
        Number of image files to write per species.
    image_size : tuple[int, int]
        Width/height of the generated JPEGs.
    seed : int
        Seed for deterministic image content.

    Returns
    -------
    dict
        Summary: num_species, num_images, annotations_path.
    """
    from PIL import Image

    if taxonomy is None:
        taxonomy = SMOKE_TAXONOMY

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)

    categories: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    img_id = 0
    ann_id = 0
    for cat_idx, taxon in enumerate(taxonomy, start=1):
        species = taxon["species"]
        categories.append({
            "id": cat_idx,
            "name": species,
            "kingdom": taxon.get("kingdom", "Animalia"),
            "phylum": taxon.get("phylum", "Chordata"),
            "class": taxon.get("class", ""),
            "order": taxon.get("order", ""),
            "family": taxon["family"],
            "genus": taxon["genus"],
            "species": species,
        })

        species_dir = root / "images" / species.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic base colour per species
        h = hashlib.sha256(species.encode()).hexdigest()
        base = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.int32)
        rng = np.random.default_rng(seed + cat_idx)

        for i in range(images_per_species):
            fname = f"images/{species.replace(' ', '_')}/{i:03d}.jpg"
            arr = np.zeros((*image_size, 3), dtype=np.uint8)
            arr[:, :, :] = (base + np.array([i * 3, i * 5, i * 7])) % 256
            noise = rng.integers(0, 30, (*image_size, 3))
            arr = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / fname)

            images.append({"id": img_id, "file_name": fname})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": cat_idx})
            img_id += 1
            ann_id += 1

    annotations_path = root / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    return {
        "num_species": len(taxonomy),
        "num_images": img_id,
        "annotations_path": str(annotations_path),
    }
