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

Resolution gate (Iteration 11 repair): the open-data bucket only
carries the FULL size ladder (small/medium/large/original) for photos
registered under the modern ``square.jpg`` convention.  Older photos
expose ``square.JPG`` / ``square.jpeg`` / ``square.png`` and have NO
medium object at all — the first fetch downloaded 198 such 75x75
thumbnails because the size rewrite was case-sensitive.  The pool is
now filtered to photos whose API URL ends in ``square.jpg``, every
download is validated with PIL (longest edge >= MIN_IMAGE_EDGE), and a
rejected candidate is replaced by the NEXT photo in the seeded order
(never by a fabricated or upscaled image).  Rejected candidates are
recorded in PROVENANCE.json so the gate is auditable.

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
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

API = "https://api.inaturalist.org/v1"
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"

#: Longest-edge floor for an accepted photo.  The bucket's ``medium``
#: rendition is 500px on its long edge; anything at or below 100px is a
#: ``square`` thumbnail masquerading as the requested size.
MIN_IMAGE_EDGE = 200
#: Only this exact suffix implies the full size ladder exists.
MODERN_SQUARE_SUFFIX = "square.jpg"

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


def medium_url(square_url: str) -> str:
    """``.../photos/<id>/square.jpg`` -> ``.../photos/<id>/medium.jpg``.

    Raises for any URL that is not the modern ``square.jpg`` convention:
    those objects have no medium rendition in the open-data bucket, so
    "fixing" the extension case-insensitively would download a 75x75
    thumbnail (the Iteration 11 Phase A defect).
    """
    if not square_url.endswith("/" + MODERN_SQUARE_SUFFIX):
        raise ValueError(f"no size ladder for {square_url!r}")
    return square_url[: -len(MODERN_SQUARE_SUFFIX)] + "medium.jpg"


class RejectedPhoto(RuntimeError):
    """A candidate photo failed the resolution gate."""


def fetch_photo_pool(session: requests.Session, taxon_id: int,
                     max_pages: int = 3) -> tuple[list[dict], int]:
    """Research-grade, CC-licensed observation photos, deterministically
    ordered by (observation_id, photo_id).

    Returns ``(pool, num_thumbnail_only)`` — photos whose API URL is not
    the modern ``square.jpg`` convention are EXCLUDED (they have no
    medium rendition) and counted, so the gate is visible rather than
    silently shrinking the pool.
    """
    pool: list[dict] = []
    seen: set[int] = set()
    thumbnail_only = 0
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
                url = p.get("url") or ""
                if not url.endswith("/" + MODERN_SQUARE_SUFFIX):
                    seen.add(p["id"])
                    thumbnail_only += 1
                    continue
                seen.add(p["id"])
                pool.append({
                    "observation_id": o["id"],
                    "photo_id": p["id"],
                    "license_code": lic or "unknown",
                    "attribution": p.get("attribution"),
                    "square_url": url,
                    "source_url": medium_url(url),
                })
    pool.sort(key=lambda p: (p["observation_id"], p["photo_id"]))
    return pool, thumbnail_only


def download_photo(session: requests.Session, url: str,
                   dest: Path) -> dict:
    """Download + resolution-validate one photo.

    Returns ``{"sha256", "width", "height", "format", "bytes"}``.
    Raises :class:`RejectedPhoto` when the served bytes are a thumbnail
    (the bucket can serve a small rendition under a medium URL) so the
    caller can deterministically move to the next candidate.
    """
    import io

    from PIL import Image

    r = _get(session, url, None, timeout=120)
    try:
        with Image.open(io.BytesIO(r.content)) as im:
            width, height = im.size
            fmt = im.format
    except Exception as exc:  # not a decodable image
        raise RejectedPhoto(f"{url}: undecodable ({exc})") from exc
    if max(width, height) < MIN_IMAGE_EDGE:
        raise RejectedPhoto(
            f"{url}: {width}x{height} below the {MIN_IMAGE_EDGE}px floor")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return {"sha256": hashlib.sha256(r.content).hexdigest(),
            "width": width, "height": height, "format": fmt,
            "bytes": dest.stat().st_size}


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
    species_rejects: list[dict] = []
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

        pool, thumb_only = fetch_photo_pool(session, tax["taxon_id"])
        if len(pool) < args.images_per_species:
            raise SystemExit(
                f"{name}: only {len(pool)} licensed research-grade photos "
                f"with a full size ladder (< {args.images_per_species}); "
                f"{thumb_only} more were thumbnail-only and excluded")
        # Deterministic seeded order over the whole pool, then accept the
        # first `k` that pass the resolution gate (a rejected candidate is
        # replaced by the NEXT one in the same seeded order).
        order = pool[:]
        rng.shuffle(order)
        chosen: list[dict] = []
        rejected: list[dict] = []
        for cand in order:
            if len(chosen) >= args.images_per_species:
                break
            if args.skip_download:
                chosen.append(dict(cand))
                continue
            try:
                staged = out / "_staging" / f"{cand['photo_id']}.jpg"
                info = download_photo(
                    session, cand["source_url"], staged)
            except RejectedPhoto as exc:
                rejected.append({"photo_id": cand["photo_id"],
                                 "observation_id": cand["observation_id"],
                                 "source_url": cand["source_url"],
                                 "reason": str(exc)})
                time.sleep(0.05)
                continue
            rec = dict(cand)
            rec.update(info)
            rec["_staged"] = str(staged)
            chosen.append(rec)
            time.sleep(0.05)  # gentle on the S3 bucket
        if len(chosen) < args.images_per_species:
            raise SystemExit(
                f"{name}: resolution gate accepted only {len(chosen)} of "
                f"{args.images_per_species} photos ({len(rejected)} "
                f"rejected, pool {len(pool)})")
        chosen.sort(key=lambda p: (p["observation_id"], p["photo_id"]))
        sp_dir = name.replace(" ", "_")
        print(f"    pool={len(pool)} thumbnail_only={thumb_only} "
              f"rejected={len(rejected)}", flush=True)
        for n, ph in enumerate(chosen):
            rel = f"images/{sp_dir}/{n:03d}.jpg"
            images.append({"id": img_id, "file_name": rel})
            annotations.append({"id": img_id, "image_id": img_id,
                                "category_id": cat_id})
            staged = ph.pop("_staged", None)
            rec = {"species": name, "file_name": rel, **ph}
            if staged is not None:
                # validated bytes -> canonical adapter-visible name
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                Path(staged).replace(out / rel)
            provenance.append(rec)
            img_id += 1
        species_rejects.append({"species": name, "rejected": rejected})
        time.sleep(0.5)  # gentle on the API

    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "_staging", ignore_errors=True)
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
        "selection": (
            "seeded shuffle of the resolution-qualified pool; the first "
            "k candidates whose downloaded bytes pass the resolution gate "
            "are accepted, canonicalized by (observation_id, photo_id)"),
        "resolution_gate": {
            "min_image_edge_px": MIN_IMAGE_EDGE,
            "pool_filter": f"API photo URL must end with "
                           f"/{MODERN_SQUARE_SUFFIX} (only that convention "
                           f"has a medium rendition in the bucket)",
            "download_check": "PIL decode + longest-edge floor",
            "replacement_policy": "next candidate in the same seeded order",
            "defect_repaired": (
                "the first Phase A fetch rewrote square.jpg -> medium.jpg "
                "case-sensitively, so 198/432 photos were stored as 75x75 "
                "thumbnails (square.JPG/.jpeg/.png have no medium object)"),
        },
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "num_photos": len(provenance),
        "photos": provenance,
        "rejected_candidates": species_rejects,
    }, indent=1))
    print(f"Wrote {len(categories)} categories, {len(images)} "
          f"images -> {out}")


if __name__ == "__main__":
    sys.exit(main())
