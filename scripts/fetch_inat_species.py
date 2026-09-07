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
from typing import TYPE_CHECKING

if TYPE_CHECKING:               # annotations only; never executed at runtime
    import requests

# ``requests`` is deliberately NOT imported at runtime here.
#
# The reason is the guard below.  ``refuse_if_frozen_pool`` is pure
# filesystem logic and the unit tests that pin it run in the minimal
# CPU-only CI environment, which installs no HTTP client - so a module-level
# ``import requests`` made those tests fail at COLLECTION, which aborts the
# whole run.  More than an inconvenience: the tests in question assert that
# the fetcher never reaches the network, and requiring a network library to
# be installed in order to run them is backwards.  Same convention the repo
# already states in requirements/ci-unit.txt for torch and friends - import
# it on the path that uses it, which is _new_session() and _get().  The
# TYPE_CHECKING block above keeps the ``requests.Session`` annotations
# resolvable for linters without importing anything.
# tests/unit/test_ci_dependency_closure.py fails if this ever moves back to
# module level.

REPO_ROOT = Path(__file__).resolve().parent.parent

API = "https://api.inaturalist.org/v1"
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"

#: Longest-edge floor for an accepted photo.  The bucket's ``medium``
#: rendition is 500px on its long edge; anything at or below 100px is a
#: ``square`` thumbnail masquerading as the requested size.
MIN_IMAGE_EDGE = 200
#: Only this exact suffix implies the full size ladder exists.
MODERN_SQUARE_SUFFIX = "square.jpg"


def refuse_if_frozen_pool(out: Path, force: bool) -> list[str]:
    """Refuse to fetch into a pool the frozen dataset is bound to.

    ``--out`` defaults to ``data/raw/inaturalist/pilot_v1``, and 432 of the
    496 photographs pinned by ``data/mllmu_hier_pilot100/image_manifest.json``
    live under it.  That manifest's roll-up is part of the dataset fingerprint
    inside every prediction sidecar, so re-fetching into the default would
    replace bytes the committed evidence certifies — and the sidecars would
    keep verifying against a manifest that no longer describes anything on
    disk, which is worse than failing loudly.

    A confirmation fetch goes to a NEW directory, so this never fires in
    normal use.  It exists because the default is one omitted flag away from
    destroying frozen evidence.  Returns the bound paths so a caller can
    report them.
    """
    bound: list[str] = []
    unreadable: list[str] = []
    for mp in sorted(REPO_ROOT.glob("data/mllmu_hier_*/image_manifest.json")):
        try:
            images = json.loads(mp.read_text()).get("images") or {}
        except (OSError, ValueError) as exc:
            unreadable.append(f"{mp.name}: {exc}")
            continue
        for rel in images:
            p = REPO_ROOT / rel
            if p == out or out in p.parents:
                bound.append(f"{mp.name} -> {rel}")
    # Fail CLOSED.  The guard's job is to show that the target is unbound,
    # and a manifest that cannot be read cannot show anything; treating it as
    # "nothing pinned" would let a corrupt manifest quietly authorise exactly
    # the overwrite it exists to prevent.
    if unreadable and not force:
        raise SystemExit(
            f"REFUSED: {len(unreadable)} frozen image manifest(s) could not "
            f"be read ({'; '.join(unreadable)}), so it cannot be shown that "
            f"{out} is unbound. An unreadable manifest is not evidence that "
            f"nothing is pinned. Repair or remove it, or pass "
            f"--allow-overwrite-frozen.")
    if bound and not force:
        shown = ", ".join(bound[:3]) + (" ..." if len(bound) > 3 else "")
        raise SystemExit(
            f"REFUSED: {out} holds {len(bound)} photograph(s) pinned by a "
            f"frozen image manifest ({shown}). Re-fetching would replace "
            f"bytes the committed evidence certifies. Pass an --out that "
            f"does not contain them, or --allow-overwrite-frozen if you "
            f"intend to re-freeze the manifest and every sidecar bound to "
            f"it.")
    return bound

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


def _new_session():
    """Create the HTTP session — the one call that reaches the network.

    A named function rather than an inline ``requests.Session()`` so the
    ordering tests can replace exactly this call and prove the frozen-pool
    guard refused before it, without an HTTP client being installed.
    """
    import requests
    return requests.Session()


def _get(session: requests.Session, url: str, params: dict | None,
         retries: int = 4, timeout: int = 60) -> requests.Response:
    import requests
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


def species_for_role(role: str, tag: str) -> list[str]:
    """The iNaturalist species in :data:`SPECIES_LIST` that carry ``role``.

    Derived from the frozen dataset through the SAME census the power
    analysis uses, rather than from a hand-copied list or a slice of
    ``SPECIES_LIST``.  The slice is a trap with a measured shape: the 6
    retain-only species sit at indices 1, 14, 16, 19, 22 and 28, so
    ``--limit-species 30`` fetches 24 TARGET species plus those 6 and
    silently drops 6 target ones - a pool that looks the right size and is
    missing a fifth of the entities the confirmation needs.

    Lazy imports: this is only reached when ``--role`` is used, and the
    unit tests import this module to exercise the frozen-pool guard, which
    needs no dataset and no parquet reader.
    """
    from granunlearn.evaluation.reference_eval import (
        load_associations_parquet)
    from power_analysis_confirmation import entity_role_census

    data_dir = REPO_ROOT / "data" / f"mllmu_hier_{tag}"
    assoc_path = data_dir / "associations.parquet"
    if not assoc_path.exists():
        raise SystemExit(
            f"REFUSED - --role needs the frozen dataset at {data_dir} to "
            f"derive which species carry a {role} association, and "
            f"{assoc_path} is not there")
    associations = load_associations_parquet(assoc_path)
    census = entity_role_census(data_dir, associations)
    if role not in census["by_role"]:
        raise SystemExit(
            f"REFUSED - unknown role {role!r}; the census knows "
            f"{sorted(census['by_role'])}")
    wanted = set(census["by_role"][role]["entity_ids"])
    out = [s for s in SPECIES_LIST if s in wanted]
    if not out:
        raise SystemExit(
            f"REFUSED - no species in SPECIES_LIST carries a {role} "
            f"association in {data_dir}, so --role {role} would fetch "
            f"nothing")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="data/raw/inaturalist/pilot_v1")
    ap.add_argument("--images-per-species", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-species", type=int, default=None)
    ap.add_argument("--role", choices=("target", "retain", "all"),
                    default="all",
                    help="fetch only the species carrying this association "
                         "role in the frozen dataset, derived by the same "
                         "census the power analysis uses; mutually exclusive "
                         "with --limit-species, which takes a slice of "
                         "SPECIES_LIST and so cannot select a role")
    ap.add_argument("--tag", default="pilot100",
                    help="dataset tag --role reads its roles from")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument(
        "--allow-overwrite-frozen", action="store_true",
        help="permit writing into a pool whose photographs a frozen image "
             "manifest pins; refused by default because that replaces bytes "
             "the committed evidence certifies")
    args = ap.parse_args()

    if args.role != "all" and args.limit_species:
        raise SystemExit(
            "REFUSED - --role and --limit-species select the species two "
            "different ways and combining them makes it ambiguous which "
            "species the pool is supposed to hold; pass one")

    out = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() \
        else Path(args.out)
    refuse_if_frozen_pool(out, args.allow_overwrite_frozen)
    if args.role != "all":
        species = species_for_role(args.role, args.tag)
    else:
        species = SPECIES_LIST[:args.limit_species] \
            if args.limit_species else SPECIES_LIST
    rng = random.Random(args.seed)
    session = _new_session()
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
