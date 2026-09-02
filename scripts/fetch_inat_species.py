"""Fetch REAL iNaturalist photos + authoritative taxonomy for the
Iteration-11 pilot-100 taxonomic stratum.

    python scripts/fetch_inat_species.py                 # full fetch
    python scripts/fetch_inat_species.py --limit-species 2 \
        --images-per-species 3                            # smoke test
    python scripts/fetch_inat_species.py --skip-download  # metadata only

Source of truth: api.inaturalist.org (taxonomy) + the
inaturalist-open-data S3 bucket (photos).  Every taxon's rank chain
(kingdom -> phylum -> class -> order -> family -> genus -> species)
is taken VERBATIM from the API's ancestor list — never hand-written
or LLM-generated (taxonomy.py principle).  Every photo's license,
attribution, observation id, and source URL are recorded in
PROVENANCE.json so the exact frozen set is re-fetchable and the
CC licensing is auditable.

Determinism: fixed committed species list; per-species photo pool
sorted by (observation_id, photo_id); seeded sample (default 42);
downloads pinned by the recorded source URLs + SHA-256 hashes.

Output layout (adapter-compatible COCO-style):
    data/raw/inaturalist/pilot_v1/
        annotations.json          # images / annotations / categories
        PROVENANCE.json           # fetch parameters + per-photo records
        images/<Genus_species>/NNN.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

API = "https://api.inaturalist.org/v1"
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"

# Committed pilot-100 taxonomic stratum: 36 species, 24 genera,
# 14 families, 4 orders.  Multi-species genera give sibling/wrong-
# branch probes; multi-genus families give ancestor levels.  Names
# are the ACTIVE iNaturalist taxa (verified 2026-09-02): the former
# Corvus monedula / Carduelis chloris / Anas strepera are now
# Coloeus monedula / Chloris chloris / Mareca strepera.
SPECIES_LIST: list[str] = [
    # Passeriformes — Passeridae
    "Passer domesticus", "Passer montanus", "Passer hispaniolensis",
    # Corvidae
    "Corvus corax", "Corvus corone", "Coloeus monedula",
    # Paridae
    "Cyanistes caeruleus", "Parus major", "Periparus ater",
    "Poecile palustris", "Poecile montanus",
    # Turdidae
    "Turdus merula", "Turdus philomelos", "Turdus pilaris",
    # Muscicapidae / Sturnidae / Fringillidae
    "Erithacus rubecula", "Sturnus vulgaris",
    "Carduelis carduelis", "Chloris chloris",
    # Anseriformes — Anatidae
    "Anas platyrhynchos", "Anas crecca", "Mareca strepera",
    "Branta canadensis", "Branta leucopsis",
    # Lepidoptera — Nymphalidae
    "Vanessa cardui", "Vanessa atalanta",
    # Pieridae
    "Pieris rapae", "Pieris brassicae", "Pieris napi",
    "Anthocharis cardamines", "Gonepteryx rhamni",
    # Papilionidae
    "Papilio machaon",
    # Carnivora — Canidae
    "Vulpes vulpes", "Canis lupus",
    # Felidae
    "Felis catus",
    # Mustelidae
    "Mustela erminea", "Mustela nivalis",
]

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus",
         "species")


def _get(session: requests.Session, url: str, params: dict | None,
         retries: int = 4, timeout: int = 60) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:  # rate limited: back off
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except (requests.RequestException, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"GET {url} failed after {retries} tries: {last}")


def fetch_taxon(session: requests.Session,
                scientific_name: str) -> dict:
    """Resolve a scientific name to its taxon record with the FULL
    authoritative ancestor chain."""
    r = _get(session, f"{API}/taxa/autocomplete",
             {"q": scientific_name, "rank": "species",
              "is_active": "true"})
    exact = [t for t in r.json().get("results", [])
             if t.get("name") == scientific_name
             and t.get("rank") == "species"]
    if not exact:
        raise SystemExit(
            f"No exact active species taxon for {scientific_name!r}")
    tid = exact[0]["id"]
    r = _get(session, f"{API}/taxa/{tid}", None)
    taxon = r.json()["results"][0]
    chain = {a["rank"]: a["name"]
             for a in taxon.get("ancestors", [])}
    chain["species"] = taxon["name"]
    missing = [k for k in RANKS if not chain.get(k)]
    if missing:
        raise SystemExit(
            f"{scientific_name}: ancestor chain missing ranks "
            f"{missing} — refusing to guess (authoritative-source "
            "principle)")
    if taxon.get("preferred_common_name"):
        chain["common_name"] = taxon["preferred_common_name"]
    return {"taxon_id": tid, "ranks": chain}


def fetch_photo_pool(session: requests.Session, taxon_id: int,
                     max_pages: int = 3) -> list[dict]:
    """Research-grade, CC-licensed observation photos, deterministically
    ordered by (observation_id, photo_id)."""
    pool: list[dict] = []
    seen: set[int] = set()
    for page in range(1, max_pages + 1):
        r = _get(session, f"{API}/observations", {
            "taxon_id": taxon_id, "photos": "true",
            "quality_grade": "research",
            "photo_license": LICENSES,
            "per_page": 200, "page": page,
            "order_by": "id", "order": "asc",
        })
        obs = r.json().get("results", [])
        if not obs:
            break
        for o in obs:
            for p in o.get("photos", []):
                if p["id"] in seen:
                    continue
                lic = (p.get("license_code") or "").lower()
                if lic and lic not in LICENSES.split(","):
                    continue
                seen.add(p["id"])
                pool.append({
                    "observation_id": o["id"],
                    "photo_id": p["id"],
                    "license_code": lic or "unknown",
                    "attribution": p.get("attribution"),
                    "source_url": (p.get("url") or "").replace(
                        "square.jpg", "medium.jpg") or None,
                })
    pool.sort(key=lambda p: (p["observation_id"], p["photo_id"]))
    return pool


def download_photo(session: requests.Session, url: str,
                   dest: Path) -> str:
    """Download one photo; returns its SHA-256."""
    r = _get(session, url, None, timeout=120)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return hashlib.sha256(r.content).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="data/raw/inaturalist/pilot_v1")
    ap.add_argument("--images-per-species", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-species", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() \
        else Path(args.out)
    species = SPECIES_LIST[:args.limit_species] \
        if args.limit_species else SPECIES_LIST
    rng = random.Random(args.seed)
    session = requests.Session()
    session.headers["User-Agent"] = \
        "granunlearn-pilot100-fetch/1.0 (research; contact: repo)"

    categories: list[dict] = []
    images: list[dict] = []
    annotations: list[dict] = []
    provenance: list[dict] = []
    img_id = 0
    for cat_id, name in enumerate(species, start=1):
        print(f"[{cat_id}/{len(species)}] {name} ...", flush=True)
        tax = fetch_taxon(session, name)
        ranks = tax["ranks"]
        cat = {"id": cat_id, "name": name, "taxon_id": tax["taxon_id"],
               **{k: ranks[k] for k in RANKS}}
        if "common_name" in ranks:
            cat["common_name"] = ranks["common_name"]
        categories.append(cat)

        pool = fetch_photo_pool(session, tax["taxon_id"])
        if len(pool) < args.images_per_species:
            raise SystemExit(
                f"{name}: only {len(pool)} licensed research-grade "
                f"photos (< {args.images_per_species})")
        # Deterministic seeded sample from the sorted pool.
        chosen = sorted(rng.sample(pool, args.images_per_species),
                        key=lambda p: (p["observation_id"],
                                       p["photo_id"]))
        sp_dir = name.replace(" ", "_")
        for n, ph in enumerate(chosen):
            rel = f"images/{sp_dir}/{n:03d}.jpg"
            images.append({"id": img_id, "file_name": rel})
            annotations.append({"id": img_id, "image_id": img_id,
                                "category_id": cat_id})
            rec = {"species": name, "file_name": rel, **ph}
            if not args.skip_download:
                rec["sha256"] = download_photo(
                    session, ph["source_url"], out / rel)
                rec["bytes"] = (out / rel).stat().st_size
            provenance.append(rec)
            img_id += 1
            time.sleep(0.05)  # gentle on the S3 bucket
        time.sleep(0.5)  # gentle on the API

    out.mkdir(parents=True, exist_ok=True)
    (out / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": annotations,
         "categories": categories}, indent=1))
    (out / "PROVENANCE.json").write_text(json.dumps({
        "source_api": API,
        "photo_host": "inaturalist-open-data S3 (medium size)",
        "species_list": species,
        "images_per_species": args.images_per_species,
        "seed": args.seed,
        "photo_licenses_allowed": LICENSES,
        "quality_grade": "research",
        "observation_order": "id asc (deterministic pool)",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "num_photos": len(provenance),
        "photos": provenance,
    }, indent=1))
    print(f"Wrote {len(categories)} categories, {len(images)} "
          f"images -> {out}")


if __name__ == "__main__":
    sys.exit(main())
