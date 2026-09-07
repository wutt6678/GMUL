"""Iteration 11C stage 3: building probes that are genuinely new.

The confirmation split has to be constructed out of material pilot100_v2
never used, and the first hazard is not the new material — it is the old
material being damaged while fetching it.  ``fetch_inat_species.py`` defaults
to ``--out data/raw/inaturalist/pilot_v1``, and 432 of the 496 photographs
pinned by the frozen image manifest live under that directory.  The manifest
roll-up is part of the dataset fingerprint inside every prediction sidecar,
so a re-fetch into the default would replace bytes the committed evidence
certifies — and the sidecars would keep verifying against a manifest that no
longer describes anything on disk, which is worse than failing loudly.

These tests pin the guard that prevents it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "inaturalist"
MANIFEST = REPO_ROOT / "data" / "mllmu_hier_pilot100" / "image_manifest.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fetch_inat_species as fetch  # noqa: E402
from fetch_inat_species import refuse_if_frozen_pool  # noqa: E402


def _pinned_under(root: Path) -> int:
    """How many manifest-pinned photographs live under ``root``."""
    if not MANIFEST.exists():
        pytest.skip(f"frozen image manifest not present: {MANIFEST}")
    images = json.loads(MANIFEST.read_text())["images"]
    return sum(1 for rel in images
               if (REPO_ROOT / rel) == root or root in (REPO_ROOT / rel).parents)


class TestTheFetchRefusesToOverwriteFrozenEvidence:
    def test_the_default_output_directory_is_the_frozen_pool(self):
        """The hazard is the default, not an unusual invocation: omitting
        --out aims the fetcher straight at the pool the evidence binds."""
        want = REPO_ROOT / "data" / "raw" / "inaturalist" / "pilot_v1"
        assert _pinned_under(want) == 432

    def test_fetching_into_the_frozen_pool_refuses(self):
        n = _pinned_under(RAW / "pilot_v1")
        with pytest.raises(SystemExit) as exc:
            refuse_if_frozen_pool(RAW / "pilot_v1", False)
        msg = str(exc.value)
        assert "REFUSED" in msg
        assert str(n) in msg
        assert "image_manifest.json" in msg
        assert "--allow-overwrite-frozen" in msg

    def test_the_refusal_names_the_bytes_it_is_protecting(self):
        with pytest.raises(SystemExit) as exc:
            refuse_if_frozen_pool(RAW / "pilot_v1", False)
        assert "data/raw/inaturalist/pilot_v1/images/" in str(exc.value)
        assert "committed evidence certifies" in str(exc.value)

    def test_a_new_directory_is_allowed(self):
        """The confirmation fetch goes somewhere new, so the guard must not
        fire in normal use — a guard that blocks the intended path gets
        passed --force out of habit, which is how guards die."""
        for name in ("confirm_v1", "local_v1", "pilot_v2"):
            assert refuse_if_frozen_pool(RAW / name, False) == [], name

    def test_the_override_reports_rather_than_hides_the_exposure(self):
        """--allow-overwrite-frozen is an escape hatch for a deliberate
        re-freeze, so it returns what is bound instead of returning nothing:
        the caller has to be able to say what it is about to invalidate."""
        bound = refuse_if_frozen_pool(RAW / "pilot_v1", True)
        assert len(bound) == _pinned_under(RAW / "pilot_v1")
        assert all(entry.startswith("image_manifest.json -> ")
                   for entry in bound)

    def test_a_directory_with_no_manifest_is_allowed(self):
        assert refuse_if_frozen_pool(REPO_ROOT / "does" / "not" / "exist",
                                     False) == []

    def test_the_guard_runs_before_any_network_access(self, monkeypatch):
        """The point of the ordering: a refusal that happened after the API
        had been queried would still have spent the fetch, and one that
        happened after a download would still have written bytes.

        ``_new_session`` is patched rather than ``requests.Session`` because
        it is the single named call that reaches the network - and because
        patching it needs no HTTP client installed, which the minimal CI
        environment does not have."""

        def no_network():
            raise AssertionError("the fetcher reached the network before "
                                 "the frozen-pool guard refused")

        monkeypatch.setattr(fetch, "_new_session", no_network)
        monkeypatch.setattr(sys, "argv", ["fetch_inat_species.py"])
        with pytest.raises(SystemExit) as exc:
            fetch.main()
        assert "REFUSED" in str(exc.value)

    def test_an_explicit_frozen_out_is_refused_too(self, monkeypatch):
        """Passing the frozen path explicitly is the same act as letting the
        default supply it, so it must refuse identically."""

        def no_network():
            raise AssertionError("reached the network")

        monkeypatch.setattr(fetch, "_new_session", no_network)
        monkeypatch.setattr(
            sys, "argv",
            ["fetch_inat_species.py", "--out",
             "data/raw/inaturalist/pilot_v1", "--skip-download"])
        with pytest.raises(SystemExit) as exc:
            fetch.main()
        assert "REFUSED" in str(exc.value)


class TestTheGuardFailsClosed:
    def test_an_unreadable_manifest_refuses_rather_than_permits(self,
                                                               tmp_path,
                                                               monkeypatch):
        """The guard's job is to SHOW that the target is unbound, and a
        manifest that cannot be parsed cannot show anything.  Failing open
        here would let a corrupt manifest quietly authorise exactly the
        overwrite the guard exists to prevent."""
        fake = tmp_path / "data" / "mllmu_hier_x"
        fake.mkdir(parents=True)
        (fake / "image_manifest.json").write_text("{not json")
        monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
        with pytest.raises(SystemExit) as exc:
            refuse_if_frozen_pool(tmp_path / "pool", False)
        assert "could not be read" in str(exc.value)
        assert "not evidence that nothing is pinned" in str(exc.value)
        # and the override still works, for a deliberate re-freeze
        assert refuse_if_frozen_pool(tmp_path / "pool", True) == []

    def test_a_manifest_without_an_images_key_is_not_treated_as_corrupt(
            self, tmp_path, monkeypatch):
        """An empty but well-formed manifest pins nothing, which is different
        from one that cannot be read."""
        man = tmp_path / "data" / "mllmu_hier_w"
        man.mkdir(parents=True)
        (man / "image_manifest.json").write_text(json.dumps({"images": {}}))
        monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
        assert refuse_if_frozen_pool(tmp_path / "pool", False) == []

    def test_a_manifest_pinning_a_relative_path_is_still_matched(self,
                                                                tmp_path,
                                                                monkeypatch):
        pool = tmp_path / "raw" / "pool"
        (pool / "species").mkdir(parents=True)
        photo = pool / "species" / "000.jpg"
        photo.write_bytes(b"x")
        man = tmp_path / "data" / "mllmu_hier_y"
        man.mkdir(parents=True)
        (man / "image_manifest.json").write_text(json.dumps(
            {"images": {"raw/pool/species/000.jpg": {"sha256": "0" * 64}}}))
        monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
        with pytest.raises(SystemExit):
            refuse_if_frozen_pool(pool, False)
        # a sibling directory is unaffected
        assert refuse_if_frozen_pool(tmp_path / "raw" / "other", False) == []

    def test_the_pool_directory_itself_is_matched_not_only_its_children(self,
                                                                       tmp_path,
                                                                       monkeypatch):
        """A manifest could pin the directory as well as files under it."""
        pool = tmp_path / "raw" / "pool"
        pool.mkdir(parents=True)
        man = tmp_path / "data" / "mllmu_hier_z"
        man.mkdir(parents=True)
        (man / "image_manifest.json").write_text(
            json.dumps({"images": {"raw/pool": {"sha256": "0" * 64}}}))
        monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
        with pytest.raises(SystemExit):
            refuse_if_frozen_pool(pool, False)


# ── the fetch selects species by ROLE, because a count is a trap ────

class TestTheFetchSelectsSpeciesByRoleNotByCount:
    """Iteration 11C-R2.  The confirmation needs 12 new photographs for each
    of the 30 TARGET species.  ``--limit-species 30`` reads like the way to
    ask for that and is not: the 6 retain-only species sit at indices 1, 14,
    16, 19, 22 and 28 of ``SPECIES_LIST``, so the first 30 entries hold 24
    target species plus those 6.  A pool of the right SIZE missing a fifth of
    the right entities is worse than an obviously wrong count, because
    nothing downstream would notice until the held-out stratum came up 6
    clusters short.
    """

    DATA_DIR = REPO_ROOT / "data" / "mllmu_hier_pilot100"

    def _require_dataset(self):
        if not (self.DATA_DIR / "associations.parquet").exists():
            pytest.skip(f"frozen dataset not present: {self.DATA_DIR}")

    def test_the_target_role_is_30_species_and_the_retain_role_is_6(self):
        self._require_dataset()
        target = fetch.species_for_role("target", "pilot100")
        retain = fetch.species_for_role("retain", "pilot100")
        assert len(target) == 30
        assert len(retain) == 6
        assert len(set(target)) == 30 and len(set(retain)) == 6
        assert not set(target) & set(retain), \
            "a retain-ONLY species carries no target association by definition"
        assert set(target) | set(retain) == set(fetch.SPECIES_LIST)
        assert len(fetch.SPECIES_LIST) == 36

    def test_every_selected_species_is_in_the_list_the_fetcher_knows(self):
        self._require_dataset()
        for role in ("target", "retain"):
            for s in fetch.species_for_role(role, "pilot100"):
                assert s in fetch.SPECIES_LIST, (role, s)

    def test_the_order_follows_species_list_so_the_seed_is_stable(self):
        """The seeded shuffle runs per species but draws from ONE rng built
        for the whole walk, so the ORDER species are fetched in is part of
        the draw: two runs of the same command must walk the same list, or
        they produce pools that differ in more than their bytes.  (That the
        walk differs from pilot_v1's is why the draw does not nest -- see
        TestTheDrawDoesNotNestSoNoveltyIsAHashProperty.)"""
        self._require_dataset()
        a = fetch.species_for_role("target", "pilot100")
        b = fetch.species_for_role("target", "pilot100")
        assert a == b
        idx = [fetch.SPECIES_LIST.index(s) for s in a]
        assert idx == sorted(idx), "the role list must be SPECIES_LIST order"

    def test_a_count_would_fetch_the_wrong_species(self):
        """The trap, measured rather than asserted: name the six that a count
        drops.  A test that only checks lengths cannot see this, because
        ``--limit-species 30`` also returns 30 species."""
        self._require_dataset()
        target = set(fetch.species_for_role("target", "pilot100"))
        sliced = set(fetch.SPECIES_LIST[:30])
        assert len(sliced) == len(target) == 30
        missing = sorted(target - sliced)
        assert len(missing) == 6, missing
        assert missing == ["Canis lupus", "Felis catus", "Mustela erminea",
                           "Mustela nivalis", "Papilio machaon",
                           "Vulpes vulpes"]
        extra = sorted(sliced - target)
        assert len(extra) == 6
        assert set(extra) == set(fetch.species_for_role("retain", "pilot100"))

    def test_the_frozen_command_uses_the_role_flag(self):
        """The command in the freeze is what a human runs at stage 3."""
        freeze = (REPO_ROOT / "data" / "reports"
                  / "mllmu_pilot100_confirmation_freeze.json")
        if not freeze.exists():
            pytest.skip(f"committed evidence not present: {freeze}")
        h = json.loads(freeze.read_text())["confirmation_size"][
            "held_out_photographs"]
        assert "--role target" in h["fetch_command"]
        assert "--tag pilot100" in h["fetch_command"]
        assert "--limit-species" not in h["fetch_command"]
        self._require_dataset()
        assert len(fetch.species_for_role("target", "pilot100")) == \
            h["species_covered"] == 30

    def test_an_unknown_role_refuses_rather_than_fetching_everything(self):
        self._require_dataset()
        with pytest.raises(SystemExit) as exc:
            fetch.species_for_role("targett", "pilot100")
        assert "unknown role" in str(exc.value)
        assert "target" in str(exc.value) and "retain" in str(exc.value)

    def test_a_missing_dataset_refuses_rather_than_guessing(self):
        """Fail closed.  Falling back to ``SPECIES_LIST`` would produce a
        36-species pool that looks successful."""
        with pytest.raises(SystemExit) as exc:
            fetch.species_for_role("target", "no_such_tag")
        assert "REFUSED" in str(exc.value)
        assert "associations.parquet" in str(exc.value)

    def test_a_role_that_no_species_carries_refuses(self, monkeypatch):
        """A role list that comes back empty must not become an empty fetch
        that quietly writes a pool with nothing in it."""
        self._require_dataset()
        monkeypatch.setattr(fetch, "SPECIES_LIST", ["Not a real species"])
        with pytest.raises(SystemExit) as exc:
            fetch.species_for_role("target", "pilot100")
        assert "would fetch nothing" in str(exc.value)

    def test_the_real_cli_parses_the_frozen_command_s_flags(self):
        """Run the real script rather than a copy of its parser: a duplicate
        ``argparse`` block in a test proves only that the test can parse what
        the test wrote.  The ambiguity refusal fires before any session is
        opened and before anything is written, so this needs no network.
        """
        proc = subprocess.run(
            [sys.executable, "scripts/fetch_inat_species.py",
             "--seed", "42", "--images-per-species", "24",
             "--role", "target", "--tag", "pilot100",
             "--limit-species", "30",
             "--out", "data/raw/inaturalist/confirm_v1"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        assert proc.returncode != 0, proc.stdout + proc.stderr
        assert "--role and --limit-species" in proc.stderr + proc.stdout
        # an unrecognised flag would fail differently, which is the point:
        # this proves the parser knows --role and --tag
        assert "unrecognized arguments" not in proc.stderr

    def test_the_frozen_command_itself_is_not_ambiguous(self):
        proc = subprocess.run(
            [sys.executable, "scripts/fetch_inat_species.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr
        for flag in ("--role", "--tag", "--limit-species",
                     "--images-per-species", "--allow-overwrite-frozen"):
            assert flag in proc.stdout, flag
        assert "{target,retain,all}" in proc.stdout.replace(" ", "")

    def test_the_ambiguity_check_runs_before_anything_is_written(self):
        """Order matters: a refusal that arrives after the frozen-pool guard
        has already been satisfied, or after a session is opened, is a
        refusal the network has already seen."""
        src = (REPO_ROOT / "scripts" / "fetch_inat_species.py").read_text()
        i_amb = src.index('args.role != "all" and args.limit_species')
        i_guard = src.index("refuse_if_frozen_pool(out,")
        i_fetch = src.index("species_for_role(args.role")
        #: the CALL site, not the definition, which sits above ``main()``
        i_session = src.index("session = _new_session()")
        assert i_amb < i_guard < i_fetch < i_session, \
            (i_amb, i_guard, i_fetch, i_session)


def _fake_pool(n: int) -> list[dict]:
    """A pool with the same shape and ordering contract the real
    ``fetch_photo_pool`` returns: deduplicated, licence-filtered and sorted
    by ``(observation_id, photo_id)``.

    The observation ids are deliberately NOT monotone in the photo ids, so a
    test that confuses draw order with canonical order fails instead of
    passing by accident.
    """
    pool = [{
        "observation_id": 100000 + (i * 37) % max(n, 1),
        "photo_id": 900000 + i,
        "license_code": "cc-by",
        "attribution": "(c) test, some rights reserved (CC BY)",
        "square_url": f"https://example.invalid/p/{900000 + i}/square.jpg",
        "source_url": f"https://example.invalid/p/{900000 + i}/medium.jpg",
    } for i in range(n)]
    pool.sort(key=lambda p: (p["observation_id"], p["photo_id"]))
    return pool


def _run_fetch(monkeypatch, out: Path, species: list[str], pool_len: int,
               k: int, seed: int = 42) -> dict[str, list[dict]]:
    """Run the REAL fetch loop against fake pools and return its provenance.

    ``--skip-download`` makes the loop accept candidates without touching
    S3, so what gets measured is the fetcher's own selection behaviour: the
    seeded shuffle, the single shared ``rng``, the resolution-gate slot
    count and the canonical re-sort.  Nothing here is a model of the
    fetcher; it is the fetcher, with only its two network calls replaced.
    """
    class _Session:
        headers: dict = {}

    idx = {s: i for i, s in enumerate(species)}
    monkeypatch.setattr(fetch, "SPECIES_LIST", list(species))
    monkeypatch.setattr(fetch, "_new_session", lambda: _Session())
    monkeypatch.setattr(fetch, "fetch_taxon", lambda session, name: {
        "taxon_id": idx[name],
        "ranks": {r: f"{r}-of-{name}" for r in fetch.RANKS}})
    monkeypatch.setattr(
        fetch, "fetch_photo_pool",
        lambda session, taxon_id, max_pages=3: (_fake_pool(pool_len), 0))
    monkeypatch.setattr(sys, "argv", [
        "fetch_inat_species.py", "--seed", str(seed),
        "--images-per-species", str(k), "--skip-download",
        "--out", str(out)])
    fetch.main()
    photos = json.loads((out / "PROVENANCE.json").read_text())["photos"]
    by_species: dict[str, list[dict]] = {}
    for rec in photos:
        by_species.setdefault(rec["species"], []).append(rec)
    return by_species


class TestTheDrawDoesNotNestSoNoveltyIsAHashProperty:
    """Iteration 11C stage 3.

    The confirmation draws 24 photographs per species and keeps 12 new ones.
    An earlier revision of the power report justified the split of 24 into
    12 + 12 by claiming the seeded shuffle NESTS: "the first 12 of the longer
    draw are the already-allocated photographs and the remainder are new BY
    CONSTRUCTION".  If that were true, novelty would come free from the seed.

    It is not true for this fetch, for two reasons that are properties of the
    fetcher's code and not of the network, so both can be measured offline.
    What survives is weaker and sufficient: 24 is the smallest draw that
    guarantees 12 hash-disjoint photographs, and novelty is established by
    comparing sha256 against the exploratory image manifest.
    """

    def test_one_species_draw_does_nest_when_nothing_else_changes(self,
                                                                 monkeypatch,
                                                                 tmp_path):
        """Stated first so the correction does not overclaim in the other
        direction.  The shuffle runs over the WHOLE pool, so changing
        ``--images-per-species`` changes only how many are TAKEN from an
        order that is a function of the pool and the seed."""
        sp = ["Passer domesticus", "Corvus corax"]
        a = _run_fetch(monkeypatch, tmp_path / "k12", sp, 327, 12)
        b = _run_fetch(monkeypatch, tmp_path / "k24", sp, 327, 24)
        for s in sp:
            ids12 = {p["photo_id"] for p in a[s]}
            ids24 = {p["photo_id"] for p in b[s]}
            assert len(ids12) == 12 and len(ids24) == 24
            assert ids12 <= ids24, s

    def test_dropping_an_earlier_species_changes_every_later_draw(self,
                                                                 monkeypatch,
                                                                 tmp_path):
        """Refutation 1, on the real loop.

        ``rng = random.Random(args.seed)`` is built ONCE outside the species
        loop and shuffled once per species, so the state reaching species i
        depends on the pool lengths of species 0..i-1.  ``pilot_v1`` walked
        all 36 species; ``--role target`` walks 30 and drops the one at index
        1.  Every species after it therefore draws a different order, which
        is exactly the situation the confirmation fetch is in.
        """
        real = list(fetch.SPECIES_LIST)
        full = real[:5]
        sub = [full[0]] + full[2:]
        assert len(full) == 5 and len(sub) == 4
        a = _run_fetch(monkeypatch, tmp_path / "walk36", full, 400, 24)
        b = _run_fetch(monkeypatch, tmp_path / "walk30", sub, 400, 24)
        # the species before the dropped one is unaffected: same state, same
        # pool, same draw
        assert {p["photo_id"] for p in a[real[0]]} == \
            {p["photo_id"] for p in b[real[0]]}
        # and every species after it is a different draw
        for s in full[2:]:
            assert {p["photo_id"] for p in a[s]} != \
                {p["photo_id"] for p in b[s]}, s

    def test_a_changed_pool_length_reorders_the_draw_completely(self,
                                                               monkeypatch,
                                                               tmp_path):
        """The second way the nesting argument fails, and the silent one.

        iNaturalist can re-grade an observation to research quality or change
        a photo's licence between fetches, which changes how many candidates
        survive the pool filter.  A shuffle's result depends on the LENGTH of
        what it shuffles, so three extra observations weeks later produce a
        different draw for every species -- with no error, and a pool that
        still looks deep enough.
        """
        sp = ["Passer domesticus", "Corvus corax"]
        a = _run_fetch(monkeypatch, tmp_path / "pool327", sp, 327, 24)
        b = _run_fetch(monkeypatch, tmp_path / "pool330", sp, 330, 24)
        for s in sp:
            assert {p["photo_id"] for p in a[s]} != \
                {p["photo_id"] for p in b[s]}, s

    def test_the_files_on_disk_are_canonical_not_in_draw_order(self,
                                                              monkeypatch,
                                                              tmp_path):
        """Refutation 2, and the reason even a nested draw would not help.

        ``chosen.sort(key=(observation_id, photo_id))`` runs before files are
        named ``000.jpg``, ``001.jpg``, ... so "the first 12" on disk means
        the 12 smallest observation/photo ids, which is unrelated to where a
        photograph sat in the draw.  Selection by position would keep
        whatever the sort happened to put first.
        """
        sp = ["Passer domesticus", "Corvus corax"]
        a = _run_fetch(monkeypatch, tmp_path / "k12", sp, 327, 12)
        b = _run_fetch(monkeypatch, tmp_path / "k24", sp, 327, 24)
        for s in sp:
            exploratory = {p["photo_id"] for p in a[s]}
            first_12_files = {p["photo_id"] for p in b[s][:12]}
            # the SET nests ...
            assert exploratory <= {p["photo_id"] for p in b[s]}
            # ... but the first 12 FILES are not that set
            assert first_12_files != exploratory, s
            assert len(first_12_files & exploratory) < 12, s
            # and the written order really is the canonical one
            keys = [(p["observation_id"], p["photo_id"]) for p in b[s]]
            assert keys == sorted(keys)
            assert [p["file_name"] for p in b[s]] == \
                sorted(p["file_name"] for p in b[s])

    def test_24_always_leaves_12_disjoint_however_much_it_redraws(self,
                                                                 monkeypatch,
                                                                 tmp_path):
        """The guarantee that replaces the nesting claim, measured on a real
        draw rather than asserted as arithmetic.

        Whatever the overlap with an exploratory 12 turns out to be -- all of
        it if the pool and the walk are unchanged, none of it if either
        drifted -- a 24-draw leaves at least 12 photographs whose identity is
        not already allocated.  Only 12 exploratory photographs exist per
        species, so the worst case is exactly 24 - 12.
        """
        sp = ["Passer domesticus"]
        drawn = _run_fetch(monkeypatch, tmp_path / "k24", sp, 327, 24)[sp[0]]
        assert len(drawn) == 24
        assert len({p["photo_id"] for p in drawn}) == 24
        for overlap in (0, 6, 12):
            # the worst case for a given overlap: the exploratory set IS the
            # first `overlap` photographs of the draw
            exploratory = {p["photo_id"] for p in drawn[:overlap]}
            disjoint = [p for p in drawn
                        if p["photo_id"] not in exploratory]
            assert len(disjoint) >= 12, (overlap, len(disjoint))
        # and the bound is tight: at a full overlap exactly 12 remain
        exploratory = {p["photo_id"] for p in drawn[:12]}
        assert len([p for p in drawn
                    if p["photo_id"] not in exploratory]) == 12

