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
