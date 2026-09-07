"""Iteration 11C stage 3c: the 360 photographs, selected under the frozen rule.

``select_confirmation_photographs.py`` executes ``PHOTO_SELECTION_RULE``,
because "which 12 of the 24" is a choice and a choice left to whoever holds the
pool is one the protocol never specified.

The rule is an OUTCOME-BLIND AMENDMENT and not a pre-fetch preregistration:
the repository's own timestamps show the pool was acquired before the rule
existed, and the rule was written once the pool made the nesting defect it
replaces measurable.  What it was fixed before is this selection and every
model output.  ``TestTheAmendmentDisclosesWhenItHappened`` in the power tests
pins the timeline; an earlier revision of this docstring claimed otherwise.

Two things are tested separately here, because passing one does not imply the
other:

* the OUTPUT is admissible -- 360 photographs, 12 per species, all sha256
  distinct, and none of them exploratory media;
* the output is what the RULE produces -- the first 12 disjoint photographs in
  the fetch's canonical order.  A selection that is merely disjoint could have
  been picked by hand after seeing the pool, which is the thing the rule exists
  to prevent, and it would pass every check on the output alone.

The refusal paths are exercised directly on ``select_for_species``, which is a
pure function, so each can be shown to fire without a pool or a network.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
SELECTION_PATH = REPORTS / "mllmu_confirm100_photograph_selection.json"
MANIFEST = REPO_ROOT / "data" / "mllmu_hier_pilot100" / "image_manifest.json"
POOL = REPO_ROOT / "data" / "raw" / "inaturalist" / "confirm_v1"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import select_confirmation_photographs as sel  # noqa: E402
from power_analysis_confirmation import (  # noqa: E402
    CONFIRM_FETCH_IMAGES_PER_SPECIES,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    EXPLORATORY_IMAGE_MANIFEST,
    PHOTO_SELECTION_RULE,
)


def _selection() -> dict:
    if not SELECTION_PATH.exists():
        pytest.skip(f"committed selection not present: {SELECTION_PATH}")
    return json.loads(SELECTION_PATH.read_text())


def _provenance() -> dict:
    p = POOL / "PROVENANCE.json"
    if not p.exists():
        pytest.skip(f"confirmation pool provenance not present: {p}")
    return json.loads(p.read_text())


def _exploratory_hashes() -> set[str]:
    if not MANIFEST.exists():
        pytest.skip(f"frozen image manifest not present: {MANIFEST}")
    return {v["sha256"]
            for v in json.loads(MANIFEST.read_text())["images"].values()}


def _require_pool_bytes() -> None:
    if not (POOL / "images").is_dir():
        pytest.skip(
            "the confirmation photographs are gitignored and absent from this "
            "checkout, so their bytes cannot be re-hashed here; the committed "
            "PROVENANCE.json and selection report still pin every sha256")


# ── the output is admissible ───────────────────────────────────────

class TestTheSelectedPhotographsAreAdmissible:
    def test_the_counts_are_the_frozen_ones(self):
        s = _selection()
        c = s["checks"]
        assert c["species_selected"] == 30
        assert c["photographs_selected"] == 360
        assert c["matches_the_frozen_total"] is True
        assert c["every_species_got_its_full_allocation"] is True
        assert len(s["selected"]) == 360
        assert len(s["per_species"]) == 30
        for species, m in s["per_species"].items():
            assert m["selected"] == CONFIRM_NEW_PHOTOS_PER_SPECIES, species
            assert m["drawn"] == 24, species

    def test_no_selected_photograph_is_exploratory_media(self):
        """The invariant the whole exercise exists to satisfy."""
        s = _selection()
        exploratory = _exploratory_hashes()
        assert len(exploratory) == 496
        selected = {p["sha256"] for p in s["selected"]}
        assert selected & exploratory == set()
        assert s["checks"]["intersection_with_exploratory_media"] == 0
        assert s["disjoint_from"]["manifest"] == EXPLORATORY_IMAGE_MANIFEST

    def test_all_360_hashes_are_distinct(self):
        s = _selection()
        hashes = [p["sha256"] for p in s["selected"]]
        assert len(set(hashes)) == len(hashes) == 360
        assert s["checks"]["all_sha256_distinct"] is True
        # and the bound list hash describes that list
        assert hashlib.sha256("\n".join(hashes).encode()).hexdigest() == \
            s["selected_sha256_list_sha256"]

    def test_every_selected_photograph_was_actually_drawn(self):
        """Disjointness alone would be satisfied by inventing 360 hashes.  Each
        one has to be a photograph the frozen fetch really accepted."""
        s, prov = _selection(), _provenance()
        drawn = {(p["species"], p["sha256"]): p for p in prov["photos"]}
        assert len(drawn) == len(prov["photos"]) == 720
        for rec in s["selected"]:
            key = (rec["species"], rec["sha256"])
            assert key in drawn, rec["pool_file_name"]
            src = drawn[key]
            assert src["file_name"] == rec["pool_file_name"]
            for field in sel.PROVENANCE_FIELDS:
                assert rec[field] == src[field], (field, rec["pool_file_name"])

    def test_every_record_is_auditable_without_the_photographs(self):
        """The photographs are gitignored, so licence, attribution, observation
        id and source URL have to travel in the committed report -- otherwise
        the CC licensing of the confirmation set is unverifiable, which is the
        gap 11R1 closed for the local_v1 pool."""
        s = _selection()
        assert s["checks"][
            "every_record_carries_licence_and_attribution"] is True
        for rec in s["selected"]:
            assert rec["license_code"], rec
            assert rec["attribution"], rec
            assert rec["source_url"].startswith("https://"), rec
            assert "medium.jpg" in rec["source_url"], rec
            assert isinstance(rec["observation_id"], int), rec
            assert isinstance(rec["photo_id"], int), rec
            assert rec["format"] == "JPEG", rec
            assert rec["bytes"] > 0, rec
        # every licence is one the fetcher was told to accept
        allowed = set(_provenance()["photo_licenses_allowed"].split(","))
        assert {rec["license_code"] for rec in s["selected"]} <= allowed

    def test_the_bytes_on_disk_are_the_bytes_the_report_pins(self):
        _require_pool_bytes()
        s = _selection()
        for rec in s["selected"]:
            path = REPO_ROOT / s["pool"]["path"] / rec["pool_file_name"]
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == rec["sha256"], \
                rec["pool_file_name"]
            assert len(data) == rec["bytes"], rec["pool_file_name"]

    def test_the_resolution_gate_is_re_checked_not_trusted(self):
        s = _selection()
        assert s["checks"]["longest_edge_below_the_resolution_floor"] == 0
        for rec in s["selected"]:
            assert max(rec["width"], rec["height"]) >= sel.MIN_IMAGE_EDGE_PX
        # the floor this script re-checks at is the one the pool was fetched
        # under, read out of the fetcher's own record rather than assumed
        assert _provenance()["resolution_gate"]["min_image_edge_px"] == \
            sel.MIN_IMAGE_EDGE_PX == 200

    def test_the_new_photographs_are_not_systematically_worse_images(self):
        """A resolution difference between the exploratory and confirmation
        photographs would be a confound no metric could separate from an
        unlearning effect."""
        r = _selection()["resolution_gate_comparability"]
        assert r["selected"]["longest_edge_below_the_floor"] == 0
        assert r["exploratory"]["longest_edge_below_the_floor"] == 0
        assert r["selected_are_not_systematically_smaller"] is True
        assert r["selected"]["longest_edge_median"] == \
            r["exploratory"]["longest_edge_median"] == 500
        assert r["selected"]["pixels_max"] == r["exploratory"]["pixels_max"]


# ── the output is what the RULE produces ───────────────────────────

class TestTheSelectionIsTheRuleAndNotAChoice:
    def test_the_rule_in_the_report_is_the_frozen_one(self):
        assert _selection()["rule_executed"] == PHOTO_SELECTION_RULE

    def test_the_selection_is_the_first_twelve_disjoint_in_canonical_order(
            self):
        """Recomputed independently from the pool, not read back out of the
        report.  A hand-picked disjoint set would pass the admissibility tests
        above and fail this one."""
        s, prov = _selection(), _provenance()
        exploratory = _exploratory_hashes()
        drawn: dict[str, list[dict]] = {}
        for p in prov["photos"]:
            drawn.setdefault(p["species"], []).append(p)
        expected: dict[str, list[str]] = {}
        for species in prov["species_list"]:
            order = sorted(drawn[species],
                           key=lambda p: (p["observation_id"], p["photo_id"]))
            keep = [p["sha256"] for p in order
                    if p["sha256"] not in exploratory]
            expected[species] = keep[:CONFIRM_NEW_PHOTOS_PER_SPECIES]
        got: dict[str, list[str]] = {}
        for rec in s["selected"]:
            got.setdefault(rec["species"], []).append(rec["sha256"])
        assert set(got) == set(expected) == set(prov["species_list"])
        for species in expected:
            assert got[species] == expected[species], species
            # and in the canonical order the rule names, not a re-sorting
            keys = [(r["observation_id"], r["photo_id"])
                    for r in s["selected"] if r["species"] == species]
            assert keys == sorted(keys), species

    def test_the_selection_index_is_the_order_the_rule_produces(self):
        s = _selection()
        for species in s["per_species"]:
            idx = [r["selection_index"] for r in s["selected"]
                   if r["species"] == species]
            assert idx == list(range(CONFIRM_NEW_PHOTOS_PER_SPECIES)), species

    def test_the_overlap_measurement_matches_an_independent_recount(self):
        """The per-species overlap is the evidence that the draw did not nest,
        so it cannot be a number the script merely asserts about itself."""
        s, prov = _selection(), _provenance()
        p1_path = REPO_ROOT / "data" / "raw" / "inaturalist" / "pilot_v1" / \
            "PROVENANCE.json"
        if not p1_path.exists():
            pytest.skip(f"exploratory pool provenance absent: {p1_path}")
        exploratory_by_species: dict[str, set[str]] = {}
        for p in json.loads(p1_path.read_text())["photos"]:
            exploratory_by_species.setdefault(p["species"], set()).add(
                p["sha256"])
        drawn: dict[str, set[str]] = {}
        for p in prov["photos"]:
            drawn.setdefault(p["species"], set()).add(p["sha256"])
        for species, m in s["per_species"].items():
            want = len(drawn[species] & exploratory_by_species[species])
            assert m["overlap_with_exploratory"] == want, species
            assert m["disjoint_available"] == 24 - want, species
            assert m["left_unselected"] == m["disjoint_available"] - 12
            assert m["overlap_is_every_exploratory_photograph_of_this_"
                    "species"] == (want == 12), species

    def test_the_draw_did_not_nest_and_the_report_says_which_one_did(self):
        """Twenty-nine of thirty species drew a different order from
        ``pilot_v1``'s, because ``rng`` is shared across the walk and the walk
        changed.  The one that did nest is index 0 of both, and naming it is
        what makes the refutation specific rather than rhetorical."""
        s = _selection()
        supersets = s["checks"][
            "species_whose_draw_was_a_superset_of_its_exploratory_12"]
        dist = s["checks"]["overlap_distribution_across_species"]
        total = sum(dist.values())
        assert total == 30
        assert len(supersets) < 30, \
            "every species nesting would contradict the measured refutation"
        assert sum(v for k, v in dist.items() if k != "12") > 0
        assert dist.get("12", 0) == len(supersets)
        # the worst case the draw size was chosen for is actually realized
        assert s["checks"][
            "minimum_disjoint_available_across_species"] == \
            CONFIRM_NEW_PHOTOS_PER_SPECIES

    def test_the_pool_it_read_is_the_frozen_fetch_output(self):
        s = _selection()
        assert s["pool"]["path"] == "data/raw/inaturalist/confirm_v1"
        assert s["pool"]["seed"] == 42
        assert s["pool"]["images_per_species"] == 24
        assert s["pool"]["photographs_drawn"] == 720
        assert s["pool"]["species"] == 30
        assert s["disjoint_from"]["manifest_sha256"] == \
            s["disjoint_from"]["manifest_sha256_bound_by_the_freeze"]
        assert s["disjoint_from"]["exploratory_photographs_pinned"] == 496

    def test_the_manifest_hash_binds_the_selection_to_the_frozen_rule(self):
        s = _selection()
        assert s["disjoint_from"]["manifest_sha256"] == \
            hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


# ── the refusals fire ─────────────────────────────────────────────

def _drawn(n: int = 24, forbidden_idx: tuple[int, ...] = ()) -> tuple[
        list[dict], set[str]]:
    """A synthetic species draw in canonical order, plus a forbidden hash set."""
    photos = [{
        "observation_id": 1000 + (i * 13) % n,
        "photo_id": 5000 + i,
        "sha256": f"{i:064x}",
        "file_name": f"images/X/{i:03d}.jpg",
    } for i in range(n)]
    photos.sort(key=lambda p: (p["observation_id"], p["photo_id"]))
    return photos, {photos[i]["sha256"] for i in forbidden_idx}


class TestTheRuleRefusesRatherThanSubstitutes:
    def test_a_species_that_cannot_supply_twelve_disjoint_refuses(self):
        """Padding from pilot_v1 would reuse media the gate scored, which is
        the one thing the sealed invariants forbid."""
        photos, forbidden = _drawn(24, forbidden_idx=tuple(range(13)))
        with pytest.raises(SystemExit) as exc:
            sel.select_for_species(photos, forbidden, 12, "X")
        msg = str(exc.value)
        assert "only 11 of the 24" in msg
        assert "Substituting from pilot_v1" in msg

    def test_exactly_twelve_disjoint_is_accepted_not_refused(self):
        """The bound is tight: at a full 12-photograph overlap the species still
        supplies exactly its allocation, which is why the draw is 24."""
        photos, forbidden = _drawn(24, forbidden_idx=tuple(range(12)))
        kept, m = sel.select_for_species(photos, forbidden, 12, "X")
        assert len(kept) == 12
        assert m["overlap_with_exploratory"] == 12
        assert m["disjoint_available"] == 12
        assert m["left_unselected"] == 0
        assert {p["sha256"] for p in kept} & forbidden == set()

    def test_a_draw_that_is_not_in_canonical_order_refuses(self):
        """'The first 12 disjoint' is only defined over an order, so an
        unordered pool has no rule to apply."""
        photos, forbidden = _drawn()
        photos[0], photos[1] = photos[1], photos[0]
        with pytest.raises(SystemExit) as exc:
            sel.select_for_species(photos, forbidden, 12, "X")
        assert "canonical (observation_id, photo_id) order" in str(exc.value)

    def test_two_drawn_photographs_sharing_bytes_refuses(self):
        """The 24 = 12 + 12 supply bound assumes 24 DISTINCT photographs; a
        byte-duplicate would silently shrink it below the allocation."""
        photos, forbidden = _drawn()
        photos[5]["sha256"] = photos[4]["sha256"]
        with pytest.raises(SystemExit) as exc:
            sel.select_for_species(photos, forbidden, 12, "X")
        assert "share bytes" in str(exc.value)

    def test_the_selection_keeps_the_canonical_order_not_the_draw_order(self):
        photos, forbidden = _drawn(24, forbidden_idx=(0, 1))
        kept, _ = sel.select_for_species(photos, forbidden, 12, "X")
        assert [p["sha256"] for p in kept] == \
            [p["sha256"] for p in photos if p["sha256"] not in forbidden][:12]
        assert kept[0]["sha256"] == photos[2]["sha256"]


class TestTheScriptRefusesToRunOnTheWrongInputs:
    """``build_selection`` cross-checks the freeze, the manifest and the pool
    before selecting anything.  Each check is a refusal, not a warning."""

    def test_it_refuses_when_the_exploratory_manifest_has_drifted(self,
                                                                 monkeypatch):
        """The rule is disjointness against that file's CONTENTS, so a manifest
        that moved since the freeze redefines what counts as new."""
        real = sel.exploratory_hashes

        def drifted(repo_root):
            hashes, _, images = real(repo_root)
            return hashes, "0" * 64, images

        monkeypatch.setattr(sel, "exploratory_hashes", drifted)
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        assert "has DRIFTED" in str(exc.value)

    def test_it_refuses_a_pool_drawn_at_the_wrong_seed(self, monkeypatch):
        real = Path.read_text

        def patched(self, *a, **k):
            text = real(self, *a, **k)
            if self.name == "PROVENANCE.json" and "confirm_v1" in str(self):
                return text.replace('"seed": 42', '"seed": 7', 1)
            return text

        monkeypatch.setattr(Path, "read_text", patched)
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        assert "drawn at seed" in str(exc.value)

    def test_it_refuses_when_the_freeze_and_the_module_disagree(self,
                                                               monkeypatch):
        monkeypatch.setattr(sel, "PHOTO_SELECTION_RULE",
                            PHOTO_SELECTION_RULE + " ")
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        assert "does not match the module constant" in str(exc.value)

    def test_it_refuses_a_freeze_that_carries_refusals(self, monkeypatch):
        """Stage 2 must seal an executable protocol before stage 3 selects
        anything under it."""
        real = Path.read_text

        def patched(self, *a, **k):
            text = real(self, *a, **k)
            if self.name == "mllmu_pilot100_confirmation_freeze.json":
                return text.replace('"refusals": []',
                                    '"refusals": ["something is wrong"]', 1)
            return text

        monkeypatch.setattr(Path, "read_text", patched)
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        msg = str(exc.value)
        if "carries 1 refusal" not in msg:
            pytest.skip("the committed freeze does not serialise refusals as "
                        "an empty list, so this mutation did not apply")
        assert "something is wrong" in msg

    def test_it_refuses_when_the_pool_is_missing(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(tmp_path)
        assert "no confirmation freeze" in str(exc.value) or \
            "is not there" in str(exc.value)

    def test_check_only_passes_on_the_committed_selection(self):
        _require_pool_bytes()
        proc = subprocess.run(
            [sys.executable, "scripts/select_confirmation_photographs.py",
             "--check-only"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "still matches a fresh derivation" in proc.stdout

    def test_a_second_run_changes_nothing(self):
        """Idempotence, and the guard that makes re-selection deliberate."""
        _require_pool_bytes()
        before = SELECTION_PATH.read_bytes()
        proc = subprocess.run(
            [sys.executable, "scripts/select_confirmation_photographs.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "nothing to do" in proc.stdout
        assert SELECTION_PATH.read_bytes() == before


# ── the verification is fail-closed, not a count and an existence check ──

def _freeze_for_species(entities, persons, tmp_path, species_expected=30):
    """A minimal freeze carrying only the two lists the derivation reads."""
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "mllmu_pilot100_confirmation_freeze.json").write_text(
        json.dumps({
            "confirmation_size": {"frozen_now": {
                "target_entity_ids": list(entities)}},
            "portrait_exemption": {"target_person_ids": list(persons)},
        }))
    return {"species_expected": species_expected}


class TestTheSelectorVerifiesTheSetAndTheBytesNotTheCounts:
    """Iteration 11C stage 3, review finding 5.

    The selector checked 30 species rather than the frozen species SET, enough
    photographs rather than exactly 24 each, and file existence rather than
    file bytes.  All three are fail-open: a pool fetched with
    ``--limit-species 30`` also holds 30 species (24 of them targets, 6 of them
    retain-only), a smaller ``--images-per-species`` can still supply 12
    disjoint photographs, and a replaced or truncated file still exists.
    """

    def test_the_species_set_is_derived_from_two_lists_the_freeze_binds(self):
        frozen = sel.read_frozen_rule(REPO_ROOT)
        derived = sel.expected_target_species(frozen, REPO_ROOT)
        assert len(derived) == 30
        assert len(set(derived)) == 30
        # the derivation is reproducible from the freeze's own two lists
        freeze = json.loads(
            (REPORTS / "mllmu_pilot100_confirmation_freeze.json").read_text())
        entities = freeze["confirmation_size"]["frozen_now"][
            "target_entity_ids"]
        persons = freeze["portrait_exemption"]["target_person_ids"]
        assert derived == sorted(set(entities) - set(persons))
        assert len(entities) == 72 and len(persons) == 42
        # and it is the set the pool actually holds, not merely the same size
        pool_species = {p["species"] for p in _provenance()["photos"]}
        assert set(derived) == pool_species
        # every derived name looks like a species and no person leaked in
        assert not (set(derived) & set(persons))
        assert all(name[0].isupper() and " " in name for name in derived)

    def test_the_report_records_the_derivation_and_the_set_it_matched(self):
        d = _selection()["species_set_derivation"]
        assert d["derived"] == d["in_the_pool"]
        assert len(d["derived"]) == 30
        assert "target_entity_ids - target_person_ids" in d["rule"]
        assert "--limit-species 30" in d["why_a_count_is_not_enough"]
        assert _selection()["checks"]["species_set_is_the_frozen_one"] is True

    def test_a_pool_of_thirty_of_the_wrong_species_refuses(self, monkeypatch):
        """The exact fail-open case: same count, wrong members.  Swapping one
        target species for a retain-only one keeps 30 species and 12 disjoint
        photographs per species, and a count check accepts it."""
        _require_pool_bytes()
        frozen = sel.read_frozen_rule(REPO_ROOT)
        real = sel.expected_target_species(frozen, REPO_ROOT)
        wrong = sorted(set(real[1:]) | {"Zosterops lateralis"})
        assert len(wrong) == 30
        monkeypatch.setattr(sel, "expected_target_species",
                            lambda f, r: wrong)
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        msg = str(exc.value)
        assert "not the 30 the frozen design covers" in msg
        assert real[0] in msg and "Zosterops lateralis" in msg

    def test_the_derivation_refuses_lists_that_cannot_support_it(
            self, tmp_path):
        entities = [f"e{i}" for i in range(72)]
        persons = [f"e{i}" for i in range(42)]
        # a freeze with no lists would derive an empty set that any pool fails
        # against for the wrong reason
        frozen = _freeze_for_species([], [], tmp_path, species_expected=None)
        with pytest.raises(SystemExit) as exc:
            sel.expected_target_species(frozen, tmp_path)
        assert "both are needed to derive" in str(exc.value)
        # lists that disagree with the frozen budget
        frozen = _freeze_for_species(entities, persons, tmp_path,
                                     species_expected=36)
        with pytest.raises(SystemExit) as exc:
            sel.expected_target_species(frozen, tmp_path)
        assert "disagree with its own budget" in str(exc.value)
        #: A person id missing from target_entity_ids.  "Nothing is both a
        #: person and a species" cannot be checked here -- a set difference
        #: never intersects its own subtrahend -- so the guard that CAN fire
        #: is the one on a person the difference would fail to remove.
        frozen = _freeze_for_species(entities, persons + ["stray_person"],
                                     tmp_path, species_expected=None)
        with pytest.raises(SystemExit) as exc:
            sel.expected_target_species(frozen, tmp_path)
        assert "are not in target_entity_ids" in str(exc.value)
        assert "would be fetched as species" in str(exc.value)
        # and the well-formed case derives 30
        frozen = _freeze_for_species(entities, persons, tmp_path)
        assert len(sel.expected_target_species(frozen, tmp_path)) == 30

    def test_every_species_drew_exactly_the_frozen_24(self):
        prov = _provenance()
        frozen = sel.read_frozen_rule(REPO_ROOT)
        assert frozen["drawn_per_species"] == \
            CONFIRM_FETCH_IMAGES_PER_SPECIES == 24
        per_species: dict = {}
        for p in prov["photos"]:
            per_species.setdefault(p["species"], []).append(p)
        assert len(per_species) == 30
        assert {n: len(v) for n, v in per_species.items()} == \
            {n: 24 for n in per_species}
        assert len(prov["photos"]) == 720
        assert _selection()["checks"][
            "every_species_drew_its_frozen_count"] is True

    def test_a_shorter_draw_refuses_even_when_twelve_are_disjoint(self,
                                                                  monkeypatch):
        """A pool drawn at 12 per species would still be hash-disjoint and
        still supply a full allocation -- but it is not the draw that was
        sealed, and the overlap that made the nesting defect measurable would
        be gone with it."""
        _require_pool_bytes()
        real = sel.read_frozen_rule

        def shorter(repo_root):
            out = dict(real(repo_root))
            out["drawn_per_species"] = 12
            return out

        monkeypatch.setattr(sel, "read_frozen_rule", shorter)
        with pytest.raises(SystemExit) as exc:
            sel.build_selection(REPO_ROOT)
        msg = str(exc.value)
        assert "drawn photographs where the frozen fetch binds 12" in msg

    def test_every_selected_file_is_rehashed_against_its_own_bytes(self):
        _require_pool_bytes()
        v = sel.verify_selected_bytes(_selection()["selected"], POOL)
        assert v["photographs_rehashed"] == 360
        assert v["sha256_mismatches"] == 0
        assert v["files_missing"] == 0
        assert v["verified_against"] == \
            "each file's own bytes, not its existence"
        assert _selection()["byte_verification"] == v
        assert _selection()["checks"][
            "every_selected_file_rehashed_and_matching"] is True

    def test_a_replaced_file_refuses_and_names_both_hashes(self):
        _require_pool_bytes()
        selected = [dict(r) for r in _selection()["selected"]]
        selected[0]["sha256"] = "0" * 64
        with pytest.raises(SystemExit) as exc:
            sel.verify_selected_bytes(selected, POOL)
        msg = str(exc.value)
        assert "do not hash to what the pool recorded" in msg
        assert "0" * 64 in msg
        assert selected[0]["pool_file_name"] in msg

    def test_a_truncated_file_refuses_on_its_byte_count(self):
        """Same sha256, different length is not reachable by a real edit, but
        the byte count is recorded in the report and pinned by the freeze, so
        a report that disagrees with the file has to be refused rather than
        trusted."""
        _require_pool_bytes()
        selected = [dict(r) for r in _selection()["selected"]]
        selected[1]["bytes"] = selected[1]["bytes"] + 1
        with pytest.raises(SystemExit) as exc:
            sel.verify_selected_bytes(selected, POOL)
        assert "do not hash to what the pool recorded" in str(exc.value)

    def test_a_missing_file_refuses_without_hashing_anything(self, tmp_path):
        """Exists is the weakest check and it is the one that used to be the
        only one; a file recorded in provenance but absent from disk cannot be
        the bytes the manifest would pin.

        Driven against an EMPTY pool directory and a single record, so it needs
        no photograph bytes and so the refusal can only be about the file named
        here.  Against the real pool this asserts nothing in a checkout without
        the gitignored photographs: all 360 are missing at once there, and the
        message names the first of them rather than the injected one.
        """
        rec = dict(_selection()["selected"][2])
        rec["pool_file_name"] = "images/Not_a_species/999.jpg"
        with pytest.raises(SystemExit) as exc:
            sel.verify_selected_bytes([rec], tmp_path)
        msg = str(exc.value)
        assert "are not on disk" in msg
        assert "images/Not_a_species/999.jpg" in msg

    def test_the_three_verifications_are_in_the_fail_closed_set(self):
        """A check that is reported but not in the failure list is a comment."""
        import inspect
        src = inspect.getsource(sel.build_selection)
        for key in ("species_set_is_the_frozen_one",
                    "every_species_drew_its_frozen_count",
                    "every_selected_file_rehashed_and_matching"):
            assert f'"{key}"' in src, key
        failures = src.split("failures = [")[1].split("]")[0]
        for key in ("species_set_is_the_frozen_one",
                    "every_species_drew_its_frozen_count",
                    "every_selected_file_rehashed_and_matching"):
            assert key in failures, key
