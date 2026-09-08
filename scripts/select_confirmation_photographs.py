"""Select the confirmation's 360 new photographs under the frozen rule.

    python scripts/select_confirmation_photographs.py

Iteration 11C stage 3c.  ``fetch_inat_species.py`` drew 24 photographs for
each of the 30 target species; the design keeps 12 per species.  WHICH 12 is a
choice, and a choice made after seeing the pool is one the protocol never
specified -- so the rule was frozen first (``PHOTO_SELECTION_RULE``, bound into
the confirmation freeze) and this script only executes it.

The rule is a content-hash rule and not a positional one, for two measured
reasons recorded in the power report under
``new_photograph_supply.seeded_refetch.draw_nesting_is_not_relied_on``:

1. ``rng`` is built once outside the species loop, so the draw order for a
   species depends on the pool lengths before it.  ``pilot_v1`` walked 36
   species and ``--role target`` walks 30, so every species after the first
   dropped one draws a different order.
2. ``chosen.sort(key=(observation_id, photo_id))`` runs before files are named,
   so position on disk is canonical order and carries no information about the
   draw.

What the fetch actually produced is measured here rather than assumed: the
overlap with each species' exploratory 12 is reported per species, which is the
evidence that the draw did not nest (one species overlaps by 12 -- the species
at index 0 of both walks -- and the rest overlap by 0, 1 or 2 by chance).

Writes ``data/reports/mllmu_confirm100_photograph_selection.json``, the
committed record of which 360 photographs the confirmation uses, with the
sha256, licence, attribution, observation id and source URL of each.  The
photographs themselves stay gitignored; this report plus the pool's committed
``PROVENANCE.json`` is what makes them auditable and re-fetchable.

Refuses rather than substitutes.  A species that cannot supply 12 photographs
disjoint from exploratory media stops the run: padding it from ``pilot_v1``
would reuse media the reference-state gate scored, which is the one thing the
sealed-split invariants forbid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_analysis_confirmation import (  # noqa: E402
    CONFIRM_FETCH_OUT,
    CONFIRM_FETCH_SEED,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    EXPLORATORY_IMAGE_MANIFEST,
    PHOTO_SELECTION_RULE,
)

log = setup_logger("select_confirmation_photographs")

#: Where the selection report is written.  Committed, unlike the photographs.
SELECTION_REPORT = "data/reports/mllmu_confirm100_photograph_selection.json"
#: Fields copied from the pool's PROVENANCE.json into the selection record, so
#: the committed report is self-sufficient: a reader can check the licence and
#: the bytes without the gitignored pool.
PROVENANCE_FIELDS = ("observation_id", "photo_id", "license_code",
                     "attribution", "square_url", "source_url", "sha256",
                     "width", "height", "format", "bytes")
#: Mirrors ``fetch_inat_species.MIN_IMAGE_EDGE``.  Duplicated rather than
#: imported because that module imports ``requests`` at module scope, which the
#: unit-test dependency set does not install; the value is asserted against the
#: pool's own PROVENANCE.json (``resolution_gate.min_image_edge_px``) so the
#: copy cannot silently drift.
MIN_IMAGE_EDGE_PX = 200


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exploratory_hashes(repo_root: Path) -> tuple[set[str], str, dict]:
    """The sha256 set the selection must be disjoint from, plus the file hash.

    Returns ``(image_hashes, manifest_sha256, images_by_path)``.  The file hash
    is returned because the freeze binds it: selecting against a manifest that
    moved since the freeze would silently redefine what counts as new.
    """
    path = repo_root / EXPLORATORY_IMAGE_MANIFEST
    if not path.exists():
        raise SystemExit(
            f"REFUSED - {EXPLORATORY_IMAGE_MANIFEST} is not there, so nothing "
            "can be shown to be disjoint from exploratory media")
    images = json.loads(path.read_text())["images"]
    hashes = {(v or {}).get("sha256") for v in images.values()}
    if None in hashes:
        raise SystemExit(
            f"REFUSED - {EXPLORATORY_IMAGE_MANIFEST} holds an entry with no "
            "sha256, so the disjointness rule cannot be applied to it")
    return hashes, _sha256(path), images


def read_frozen_rule(repo_root: Path) -> dict[str, Any]:
    """Read the selection rule and its bindings out of the committed freeze.

    Read rather than restated: a second copy of the rule in this script would
    be a second copy that could disagree with the sealed one, and the sealed
    one is what the protocol certifies.
    """
    path = repo_root / "data" / "reports" / \
        "mllmu_pilot100_confirmation_freeze.json"
    if not path.exists():
        raise SystemExit(
            f"REFUSED - no confirmation freeze at {path}. Stage 2 must seal "
            "the protocol before stage 3 selects anything under it")
    freeze = json.loads(path.read_text())
    refusals = freeze.get("refusals") or []
    if refusals:
        raise SystemExit(
            f"REFUSED - the committed freeze carries {len(refusals)} "
            f"refusal(s), so it does not describe a protocol that can be "
            f"executed: {refusals[0]}")
    photos = (freeze.get("confirmation_size") or {}).get(
        "held_out_photographs") or {}
    frozen_now = (freeze.get("confirmation_size") or {}).get(
        "frozen_now") or {}
    rule = photos.get("selection_rule")
    if rule != PHOTO_SELECTION_RULE:
        raise SystemExit(
            f"REFUSED - the freeze's selection rule does not match the module "
            f"constant this script executes.\n  freeze: {rule!r}\n  module: "
            f"{PHOTO_SELECTION_RULE!r}")
    return {
        "rule": rule,
        "per_species": photos.get("new_photographs_per_species"),
        "species_expected": photos.get("species_covered"),
        "total_expected": photos.get("new_photographs_total"),
        "fetch_out": photos.get("fetch_out"),
        "fetch_seed": photos.get("fetch_seed"),
        "short_refusal_at": photos.get(
            "refuse_a_species_with_fewer_than_n_disjoint"),
        "drawn_per_species": photos.get("fetch_images_per_species"),
        "exploratory_manifest":
            frozen_now.get("exploratory_photograph_sha256_manifest"),
        "exploratory_manifest_sha256":
            frozen_now.get("exploratory_photograph_sha256_manifest_sha256"),
        "exempted_portrait_set_sha256":
            frozen_now.get("required_repeat_portrait_set_sha256"),
    }


def select_for_species(drawn: list[dict], forbidden: set[str],
                       need: int, species: str) -> tuple[list[dict], dict]:
    """Apply the frozen rule to one species' 24 drawn photographs.

    ``drawn`` must already be in the fetch's canonical
    ``(observation_id, photo_id)`` order; the rule keeps the first ``need``
    whose sha256 is not exploratory media, and refuses rather than substitutes
    if fewer than ``need`` qualify.
    """
    keys = [(p["observation_id"], p["photo_id"]) for p in drawn]
    if keys != sorted(keys):
        raise SystemExit(
            f"REFUSED - {species}: the pool's photographs are not in the "
            "fetch's canonical (observation_id, photo_id) order, so 'the first "
            f"{need} disjoint' is not a defined subset of them")
    hashes = [p["sha256"] for p in drawn]
    if len(set(hashes)) != len(hashes):
        dupes = len(hashes) - len(set(hashes))
        raise SystemExit(
            f"REFUSED - {species}: {dupes} of the {len(hashes)} drawn "
            "photographs share bytes with another drawn photograph, so the "
            "draw does not supply the distinct photographs the design assumes")
    disjoint = [p for p in drawn if p["sha256"] not in forbidden]
    overlap = sorted({p["sha256"] for p in drawn} & forbidden)
    if len(disjoint) < need:
        raise SystemExit(
            f"REFUSED - {species}: only {len(disjoint)} of the {len(drawn)} "
            f"drawn photographs are disjoint from exploratory media, and the "
            f"frozen rule requires {need}. Substituting from pilot_v1 would "
            "reuse media the reference-state gate scored")
    measurement = {
        "drawn": len(drawn),
        "distinct_sha256_drawn": len(set(hashes)),
        "overlap_with_exploratory": len(overlap),
        "disjoint_available": len(disjoint),
        "selected": need,
        "left_unselected": len(disjoint) - need,
        "overlap_is_every_exploratory_photograph_of_this_species":
            len(overlap) == need,
    }
    return disjoint[:need], measurement


def expected_target_species(frozen: dict[str, Any],
                            repo_root: Path) -> list[str]:
    """The exact set of species the frozen design covers, derived not counted.

    A count of 30 is satisfied by any 30 species, including a pool fetched
    with ``--limit-species 30`` -- which holds 24 of the target species plus
    the 6 retain-only ones, and is the exact mistake ``--role`` exists to
    prevent.  So the SET is derived here from two lists the freeze binds
    independently, and compared to the pool's:

        target species = target_entity_ids - target_person_ids

    ``target_entity_ids`` is the 72-entity census the primary claims are
    defined on; ``target_person_ids`` is the 42 persons measured for the
    portrait exemption.  Their difference has to be the 30 species the fetch
    was told to draw, and deriving it this way cross-checks two measurements
    against each other instead of trusting either.
    """
    path = repo_root / "data" / "reports" / \
        "mllmu_pilot100_confirmation_freeze.json"
    freeze = json.loads(path.read_text())
    frozen_now = (freeze.get("confirmation_size") or {}).get(
        "frozen_now") or {}
    persons = (freeze.get("portrait_exemption") or {}).get(
        "target_person_ids") or []
    entities = frozen_now.get("target_entity_ids") or []
    if not entities or not persons:
        raise SystemExit(
            f"REFUSED - the freeze binds {len(entities)} target entity ids and "
            f"{len(persons)} target person ids; both are needed to derive the "
            "target species set, and an empty list would derive an empty set "
            "that any pool would fail against for the wrong reason")
    species = sorted(set(entities) - set(persons))
    want = frozen.get("species_expected")
    if want is not None and len(species) != want:
        raise SystemExit(
            f"REFUSED - {len(entities)} target entities minus {len(persons)} "
            f"target persons leaves {len(species)} species, but the frozen "
            f"size covers {want}; the freeze's own lists disagree with its "
            "own budget")
    #: A set difference cannot intersect its own subtrahend, so asserting
    #: "nothing is both a person and a species" here would be vacuous -- a
    #: guard that cannot fire is a comment.  What CAN go wrong is a person id
    #: missing from target_entity_ids: the difference would not remove it, and
    #: it would be fetched as a species.  A person leaking the other way -- in
    #: target_entity_ids but not measured as a person -- changes the count and
    #: is caught by the budget check above.
    stray = sorted(set(persons) - set(entities))
    if stray:
        raise SystemExit(
            f"REFUSED - {len(stray)} target person id(s) are not in "
            f"target_entity_ids ({stray[:3]}), so the set difference would "
            "not remove them and they would be fetched as species")
    return species


def verify_selected_bytes(selected: list[dict], pool_dir: Path) -> dict:
    """Re-hash every selected photograph and compare to what the pool recorded.

    Existence is not integrity.  ``PROVENANCE.json`` records a sha256 and a
    byte count per photograph, and the committed selection report copies both
    into evidence the freeze and the stage-4 tag will bind -- so a file that
    was replaced, truncated or re-encoded after the fetch would otherwise be
    pinned by a hash that no longer describes it.  That is the failure mode
    ``build_image_manifest.py`` exists to catch for the exploratory dataset,
    and catching it here means the selection cannot be sealed over bytes that
    moved.
    """
    mismatched: list[dict] = []
    missing: list[str] = []
    for rec in selected:
        path = pool_dir / rec["pool_file_name"]
        if not path.exists():
            missing.append(rec["pool_file_name"])
            continue
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != rec["sha256"] or len(data) != rec["bytes"]:
            mismatched.append({
                "file": rec["pool_file_name"],
                "recorded_sha256": rec["sha256"],
                "actual_sha256": actual,
                "recorded_bytes": rec["bytes"],
                "actual_bytes": len(data),
            })
    if missing:
        raise SystemExit(
            f"REFUSED - {len(missing)} selected photograph(s) are recorded in "
            f"the pool's provenance but are not on disk ({missing[0]}, ...), "
            "so their bytes cannot be the bytes the manifest would pin")
    if mismatched:
        raise SystemExit(
            f"REFUSED - {len(mismatched)} selected photograph(s) do not hash "
            f"to what the pool recorded: "
            f"{json.dumps(mismatched[0])}. The bytes moved after the fetch, "
            "so the selection would pin a hash that describes nothing on disk")
    return {
        "photographs_rehashed": len(selected),
        "sha256_mismatches": len(mismatched),
        "files_missing": len(missing),
        "verified_against": "each file's own bytes, not its existence",
    }


def _resolution_comparability(selected: list[dict],
                              repo_root: Path) -> dict[str, Any]:
    """Compare the selected photographs' resolution against the exploratory ones.

    The confirmation re-measures the same stratum on new photographs.  If the
    new ones were systematically smaller or lower-resolution, a difference in
    the metrics could be an image-quality effect rather than an unlearning
    effect -- and nothing downstream would say so, because the resolution gate
    is applied at download and never looked at again.  Both pools' provenance
    is committed, so this comparison is reproducible from the commit.
    """
    def _edges(ps: list[dict]) -> dict[str, Any]:
        longs = sorted(max(p["width"], p["height"]) for p in ps)
        shorts = sorted(min(p["width"], p["height"]) for p in ps)
        pixels = sorted(p["width"] * p["height"] for p in ps)
        return {
            "n": len(ps),
            "longest_edge_min": longs[0],
            "longest_edge_median": longs[len(longs) // 2],
            "shortest_edge_min": shorts[0],
            "shortest_edge_median": shorts[len(shorts) // 2],
            "pixels_min": pixels[0],
            "pixels_max": pixels[-1],
            "longest_edge_below_the_floor":
                sum(1 for x in longs if x < MIN_IMAGE_EDGE_PX),
        }

    out: dict[str, Any] = {
        "gate": (f"longest edge >= {MIN_IMAGE_EDGE_PX}px "
                 "(fetch_inat_species.MIN_IMAGE_EDGE), applied at download "
                 "and re-checked here"),
        "selected": _edges(selected),
        "why_it_is_measured": (
            "a systematic resolution difference between the exploratory and "
            "confirmation photographs would be a confound no metric could "
            "distinguish from an unlearning effect"),
    }
    exploratory_pool = repo_root / "data" / "raw" / "inaturalist" / \
        "pilot_v1" / "PROVENANCE.json"
    if exploratory_pool.exists():
        out["exploratory"] = _edges(
            json.loads(exploratory_pool.read_text())["photos"])
        out["selected_are_not_systematically_smaller"] = \
            out["selected"]["longest_edge_min"] >= \
            out["exploratory"]["longest_edge_min"] - 50
    else:
        out["exploratory"] = None
        out["selected_are_not_systematically_smaller"] = None
        out["exploratory_pool_absent"] = (
            f"{exploratory_pool} is not there, so the comparison could not be "
            "made; it is committed in a normal checkout")
    return out


def build_selection(repo_root: Path) -> dict[str, Any]:
    """Execute the frozen rule over the fetched pool and measure the result."""
    frozen = read_frozen_rule(repo_root)
    forbidden, manifest_sha, _ = exploratory_hashes(repo_root)

    if frozen["exploratory_manifest"] != EXPLORATORY_IMAGE_MANIFEST:
        raise SystemExit(
            f"REFUSED - the freeze names {frozen['exploratory_manifest']!r} as "
            f"the media to be disjoint from, this script reads "
            f"{EXPLORATORY_IMAGE_MANIFEST!r}")
    if frozen["exploratory_manifest_sha256"] != manifest_sha:
        raise SystemExit(
            f"REFUSED - {EXPLORATORY_IMAGE_MANIFEST} has DRIFTED since the "
            f"freeze: it bound "
            f"{frozen['exploratory_manifest_sha256']} and the file now hashes "
            f"to {manifest_sha}. The set of media the confirmation must avoid "
            "changed after the rule was sealed")

    pool_dir = repo_root / (frozen["fetch_out"] or CONFIRM_FETCH_OUT)
    provenance_path = pool_dir / "PROVENANCE.json"
    if not provenance_path.exists():
        raise SystemExit(
            f"REFUSED - {provenance_path} is not there. Run the frozen fetch "
            f"command before selecting from its pool")
    prov = json.loads(provenance_path.read_text())
    if prov.get("seed") != (frozen["fetch_seed"] or CONFIRM_FETCH_SEED):
        raise SystemExit(
            f"REFUSED - the pool at {pool_dir} was drawn at seed "
            f"{prov.get('seed')} but the freeze binds seed "
            f"{frozen['fetch_seed']}; this is not the frozen fetch's output")
    #: MIN_IMAGE_EDGE_PX is a copy of a constant in a module this one cannot
    #: import, so it is checked against the value the fetcher recorded in its
    #: own provenance.  Without this the copy could drift and the resolution
    #: re-check below would silently pass photographs the real gate rejects.
    gate_px = (prov.get("resolution_gate") or {}).get("min_image_edge_px")
    if gate_px != MIN_IMAGE_EDGE_PX:
        raise SystemExit(
            f"REFUSED - the pool was fetched under a resolution gate of "
            f"{gate_px}px but this script re-checks at {MIN_IMAGE_EDGE_PX}px; "
            "one of the two is stale and the comparability measurement would "
            "be against a floor the photographs were never held to")

    drawn_by_species: dict[str, list[dict]] = {}
    for rec in prov["photos"]:
        drawn_by_species.setdefault(rec["species"], []).append(rec)

    need = frozen["per_species"] or CONFIRM_NEW_PHOTOS_PER_SPECIES
    drawn_per_species = frozen.get("drawn_per_species")
    #: The exact SET, not its size: 30 species is satisfied by a
    #: ``--limit-species 30`` pool, which holds 24 target species plus the 6
    #: retain-only ones.
    want_species = expected_target_species(frozen, repo_root)
    got_species = sorted(drawn_by_species)
    if got_species != want_species:
        missing = sorted(set(want_species) - set(drawn_by_species))
        extra = sorted(set(drawn_by_species) - set(want_species))
        raise SystemExit(
            f"REFUSED - the pool holds {len(got_species)} species but not the "
            f"{len(want_species)} the frozen design covers. Missing "
            f"{len(missing)}: {missing[:4]}. Unexpected {len(extra)}: "
            f"{extra[:4]}. The fetch that produced this pool was not the "
            "frozen one")
    #: And exactly the frozen draw per species, not merely enough: a pool
    #: drawn at a smaller --images-per-species could still supply 12 disjoint
    #: photographs while being a different draw than the one sealed.
    wrong_draw = sorted(
        s for s, ps in drawn_by_species.items()
        if drawn_per_species is not None and len(ps) != drawn_per_species)
    if wrong_draw:
        raise SystemExit(
            f"REFUSED - {len(wrong_draw)} species have "
            f"{ {s: len(drawn_by_species[s]) for s in wrong_draw[:3]} } "
            f"drawn photographs where the frozen fetch binds "
            f"{drawn_per_species} each ({wrong_draw[0]}, ...)")

    selected: list[dict] = []
    per_species: dict[str, Any] = {}
    for species in prov["species_list"]:
        drawn = drawn_by_species.get(species)
        if not drawn:
            raise SystemExit(
                f"REFUSED - {species} is in the pool's species list but has no "
                "photographs in it")
        kept, measurement = select_for_species(drawn, forbidden, need, species)
        per_species[species] = measurement
        for n, p in enumerate(kept):
            selected.append({
                "species": species,
                "selection_index": n,
                "pool_file_name": p["file_name"],
                **{k: p.get(k) for k in PROVENANCE_FIELDS},
            })

    #: Re-hash every selected file rather than checking it exists: existence
    #: is not integrity, and the hashes about to be sealed have to describe
    #: the bytes that are actually there.
    byte_verification = verify_selected_bytes(selected, pool_dir)

    hashes = [p["sha256"] for p in selected]
    total_expected = frozen["total_expected"]
    checks = {
        "species_selected": len(per_species),
        "species_set_is_the_frozen_one": got_species == want_species,
        "every_species_drew_its_frozen_count": not wrong_draw,
        "photographs_selected": len(selected),
        "distinct_sha256": len(set(hashes)),
        "all_sha256_distinct": len(set(hashes)) == len(hashes),
        "intersection_with_exploratory_media": len(set(hashes) & forbidden),
        "exploratory_media_pinned": len(forbidden),
        "every_species_got_its_full_allocation":
            all(m["selected"] == need for m in per_species.values()),
        "minimum_disjoint_available_across_species":
            min(m["disjoint_available"] for m in per_species.values()),
        "overlap_distribution_across_species": {
            str(k): sum(1 for m in per_species.values()
                        if m["overlap_with_exploratory"] == k)
            for k in sorted({m["overlap_with_exploratory"]
                             for m in per_species.values()})},
        "species_whose_draw_was_a_superset_of_its_exploratory_12": sorted(
            s for s, m in per_species.items()
            if m["overlap_is_every_exploratory_photograph_of_this_species"]),
        "licence_codes_present": sorted(
            {p["license_code"] for p in selected}),
        "every_record_carries_licence_and_attribution":
            all(p.get("license_code") and p.get("attribution")
                for p in selected),
        "longest_edge_below_the_resolution_floor":
            sum(1 for p in selected
                if max(p["width"], p["height"]) < MIN_IMAGE_EDGE_PX),
        "every_selected_file_rehashed_and_matching":
            byte_verification["sha256_mismatches"] == 0
            and byte_verification["files_missing"] == 0
            and byte_verification["photographs_rehashed"] == len(selected),
        "matches_the_frozen_total":
            total_expected is None or len(selected) == total_expected,
    }
    failures = [k for k, v in checks.items()
                if k in ("all_sha256_distinct",
                         "every_species_got_its_full_allocation",
                         "species_set_is_the_frozen_one",
                         "every_species_drew_its_frozen_count",
                         "every_selected_file_rehashed_and_matching",
                         "matches_the_frozen_total",
                         "every_record_carries_licence_and_attribution")
                and v is not True]
    if checks["intersection_with_exploratory_media"] != 0:
        failures.append("intersection_with_exploratory_media")
    if checks["minimum_disjoint_available_across_species"] < need:
        failures.append("minimum_disjoint_available_across_species")
    if checks["longest_edge_below_the_resolution_floor"] != 0:
        failures.append("longest_edge_below_the_resolution_floor")
    if failures:
        raise SystemExit(
            f"REFUSED - the selection failed its own checks: {failures}. "
            f"{json.dumps(checks, indent=1)}")

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "11C-3c",
        "what_this_is": (
            "the committed record of WHICH 360 photographs the confirmation "
            "uses. The photographs are gitignored; this report plus the pool's "
            "committed PROVENANCE.json is what makes them auditable"),
        "rule_executed": frozen["rule"],
        "rule_is_a_hash_rule_not_a_positional_one": (
            "see new_photograph_supply.seeded_refetch"
            ".draw_nesting_is_not_relied_on in the power report for the two "
            "measured reasons position carries no information here"),
        "pool": {
            "path": str(pool_dir.relative_to(repo_root)),
            "provenance_sha256": _sha256(provenance_path),
            "seed": prov.get("seed"),
            "images_per_species": prov.get("images_per_species"),
            "species": len(prov.get("species_list") or []),
            "photographs_drawn": len(prov.get("photos") or []),
        },
        "disjoint_from": {
            "manifest": EXPLORATORY_IMAGE_MANIFEST,
            "manifest_sha256": manifest_sha,
            "manifest_sha256_bound_by_the_freeze":
                frozen["exploratory_manifest_sha256"],
            "exploratory_photographs_pinned": len(forbidden),
        },
        "per_species": per_species,
        "species_set_derivation": {
            "rule": "target species = target_entity_ids - target_person_ids, "
                    "both bound by the freeze",
            "why_a_count_is_not_enough": (
                "30 species is also satisfied by a --limit-species 30 pool, "
                "which holds 24 of the target species plus the 6 retain-only "
                "ones and silently drops 6 target ones"),
            "derived": want_species,
            "in_the_pool": got_species,
        },
        "byte_verification": byte_verification,
        "resolution_gate_comparability":
            _resolution_comparability(selected, repo_root),
        "resolution_gate_the_pool_was_fetched_under":
            prov.get("resolution_gate"),
        "checks": checks,
        "selected": selected,
        "selected_sha256_list_sha256": hashlib.sha256(
            "\n".join(hashes).encode()).hexdigest(),
        #: Renamed from ``next_step``, which read as a statement about the
        #: present and so went stale the moment the next stage ran: this report
        #: is a committed artifact that outlives the selection it records, and
        #: it kept pointing at a split build and a manifest seal that had both
        #: happened and been frozen over.  Updating the TEXT would not fix that
        #: -- the following stage would age it again -- so the field is named
        #: after the time it describes, which makes it permanently true.
        "stage_at_generation_time": (
            "stage 3 of 5 (build genuinely new probes): the 360 photographs "
            "are selected and byte-verified, the split is not yet built"),
        "next_step_at_generation_time": (
            "scripts/build_confirmation_split.py attaches these photographs to "
            "the target species' associations, then "
            "scripts/build_image_manifest.py --tag confirm100 seals them into "
            "a manifest dataset_fingerprint binds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None,
                        help=f"defaults to {SELECTION_REPORT}")
    parser.add_argument("--allow-reselect", action="store_true",
                        help="overwrite an existing selection. Off by default: "
                             "re-selecting after the split is built would make "
                             "the built split and its committed record "
                             "describe different photographs")
    parser.add_argument("--check-only", action="store_true",
                        help="recompute the selection and report whether it "
                             "matches the committed one; change nothing")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out = Path(args.output or SELECTION_REPORT)
    if not out.is_absolute():
        out = repo_root / out

    log.info("applying the frozen photograph selection rule")
    selection = build_selection(repo_root)
    # generated_utc is the only volatile field; everything else must reproduce
    # exactly or the rule that ran is not the rule that was frozen.
    comparable = {k: v for k, v in selection.items() if k != "generated_utc"}

    if args.check_only:
        if not out.exists():
            raise SystemExit(f"no committed selection at {out}")
        committed = json.loads(out.read_text())
        keys = (set(comparable) | set(committed)) - {"generated_utc"}
        drift = sorted(k for k in keys if committed.get(k) != comparable.get(k))
        if drift:
            print(f"FAILED - the committed selection at {out} differs from a "
                  f"fresh derivation in: {drift}")
            raise SystemExit(1)
        c = selection["checks"]
        print(f"OK - {out} still matches a fresh derivation of the frozen rule")
        print(f"  {c['photographs_selected']} photographs over "
              f"{c['species_selected']} species, "
              f"{c['distinct_sha256']} distinct sha256, "
              f"{c['intersection_with_exploratory_media']} in common with "
              f"exploratory media")
        return 0

    if out.exists() and not args.allow_reselect:
        existing = json.loads(out.read_text())
        same = {k: v for k, v in existing.items()
                if k != "generated_utc"} == comparable
        if same:
            print(f"{out} already holds exactly this selection "
                  f"({existing.get('generated_utc')}); nothing to do")
            return 0
        raise SystemExit(
            f"REFUSED - {out} already holds a DIFFERENT selection "
            f"(written {existing.get('generated_utc')}, "
            f"{len(existing.get('selected') or [])} photographs). Re-selecting "
            "would make any split already built from it describe different "
            "photographs than its committed record. Pass --allow-reselect if "
            "that is intended, or --check-only to compare.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=1, sort_keys=True))
    c = selection["checks"]
    print(f"wrote {out}")
    print(f"  pool                {selection['pool']['path']} "
          f"({selection['pool']['photographs_drawn']} drawn at seed "
          f"{selection['pool']['seed']})")
    print(f"  selected            {c['photographs_selected']} photographs "
          f"over {c['species_selected']} species")
    print(f"  distinct sha256     {c['distinct_sha256']}")
    print(f"  shared with explor. {c['intersection_with_exploratory_media']} "
          f"(of {c['exploratory_media_pinned']} pinned)")
    print(f"  overlap per species "
          f"{json.dumps(c['overlap_distribution_across_species'])}")
    print(f"  superset draws      "
          f"{c['species_whose_draw_was_a_superset_of_its_exploratory_12']}")
    print(f"  min disjoint/species "
          f"{c['minimum_disjoint_available_across_species']}")
    print(f"  licences            {json.dumps(c['licence_codes_present'])}")
    print(f"  selection list hash {selection['selected_sha256_list_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
