"""Iteration 11R1 — mutation tests for provenance and the equivalence note.

Every test here exists because something was MEASURED to be wrong, not
because it looked prudent in the abstract:

* A fabricated ``raw_output``, a flipped ``is_finer_than_target`` bit (the
  column that drives FILR), a single corrupted byte and a deleted row all
  verified cleanly under provenance contract v1, because ``verify_sidecar``
  compared the sidecar against the run's EXPECTATIONS and never against the
  parquet sitting beside it.
* ``adapter_config.json`` was outside the fingerprint, so editing ``r``
  16->32, ``lora_alpha`` 32->64 and ``target_modules`` from seven
  projections to one left ``adapter_sha256`` byte-identical.
* ``associations.parquet`` pins image PATHS, not photographs, so swapping a
  species' picture was invisible to every check.
* ``power_note`` picked its wording from the CI half-width before asking
  whether the CI excluded zero, so B0 and M_F — both significant, both
  wider than the margin — were narrated as "'No significant difference
  detected' is the most this interval supports" directly beside
  ``significant_difference: true``.

CPU-only and torch-free, so these run in the CI job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from granunlearn.evaluation.prediction_provenance import (
    ADAPTER_CONTRACT_FILES,
    CODE_FINGERPRINT_MODULES,
    GENERATION_CONFIG_KEYS,
    IMAGE_MANIFEST_NAME,
    PROVENANCE_CONTRACT_VERSION,
    PredictionFingerprint,
    _canonical_rollup,
    adapter_contract,
    dataset_fingerprint,
    image_manifest_path,
    image_manifest_sha256,
    referenced_image_paths,
    resolve_image_path,
    sha256_file,
    sidecar_path,
    verify_image_manifest,
    verify_sidecar,
    write_image_manifest,
    write_sidecar,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
PRED_DIR = REPO_ROOT / "data" / "mllmu_hier_pilot100" / "predictions"
DATA_DIR = REPO_ROOT / "data" / "mllmu_hier_pilot100"
ROWS = 2259


# ── fixtures ──────────────────────────────────────────────────────

def _write_parquet(path: Path, rows: int = ROWS) -> Path:
    """A prediction parquet shaped like the real ones."""
    import pandas as pd

    pd.DataFrame({
        "experiment_id": ["mllmu_pilot100_iter11"] * rows,
        "checkpoint_id": ["MF"] * rows,
        "query_id": [f"q{i}" for i in range(rows)],
        "raw_output": ["*Anas* (Mallard)\n\nFull classification..."] * rows,
        "parsed_answer": ["anas_platyrhynchos"] * rows,
        "predicted_level": [3] * rows,
        "is_correct_branch": [True] * rows,
        "is_finer_than_target": [False] * rows,
        "is_coarser_than_target": [False] * rows,
    }).to_parquet(path, index=False)
    return path


def _set_column(path: Path, column: str, values: list) -> None:
    """Rewrite one column, preserving the schema exactly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    idx = table.schema.get_field_index(column)
    old = table.column(column)
    pq.write_table(
        table.set_column(idx, table.schema.field(column),
                         pa.array(values, type=old.type)),
        path)


def _fp(**over) -> PredictionFingerprint:
    base = dict(
        experiment_id="mllmu_pilot100_iter11",
        checkpoint_id="MF",
        adapter_sha256="a1" * 32,
        base_model_revision="b2" * 20,
        dataset={"version": "pilot100_v2",
                 "artifacts_sha256": {"queries.parquet": "c3" * 32},
                 "data_dir": "data/mllmu_hier_pilot100",
                 "image_manifest_sha256": "aa" * 32,
                 "num_images_pinned": 496},
        generation_config=dict(zip(GENERATION_CONFIG_KEYS,
                                   (8, 8, 96, False, 147456, 1536))),
        code={"git_commit": "e5" * 20, "git_dirty": False,
              "modules_sha256": {m: "f6" * 32
                                 for m in CODE_FINGERPRINT_MODULES}},
        created_utc="2026-09-06T00:00:00+00:00",
        num_rows=ROWS,
        adapter_contract={
            "files": {"adapter_model.safetensors": "a1" * 32,
                      "adapter_config.json": "bb" * 32},
            "sha256": "cc" * 32,
            "missing_files": [],
            "adapter_dir": "ckpt/adapters"})
    base.update(over)
    return PredictionFingerprint(**base)


@pytest.fixture
def pair(tmp_path):
    """(parquet, fingerprint) for a correctly written v2 sidecar."""
    path = _write_parquet(tmp_path / "predictions_test_MF.parquet")
    fp = _fp()
    write_sidecar(path, fp)
    return path, fp


def _make_adapter(tmp_path: Path, r: int = 16, alpha: int = 32,
                  targets: list | None = None) -> Path:
    """A minimal but structurally real adapter directory."""
    ad = tmp_path / "adapters"
    ad.mkdir(parents=True, exist_ok=True)
    (ad / "adapter_model.safetensors").write_bytes(b"weights" * 64)
    (ad / "adapter_config.json").write_text(json.dumps({
        "r": r, "lora_alpha": alpha, "lora_dropout": 0.05,
        "target_modules": targets or ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        "peft_type": "LORA", "task_type": "CAUSAL_LM"}, indent=2))
    return ad


def _make_dataset(tmp_path: Path, num_images: int = 2) -> tuple[Path, Path]:
    """A dataset directory with real associations and real image files."""
    import pandas as pd

    ds = tmp_path / "ds"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "manifest.json").write_text(json.dumps({"version": "x_v2"}))
    pd.DataFrame({"query_id": [f"q{i}" for i in range(4)]}).to_parquet(
        ds / "queries.parquet", index=False)
    rels = []
    for i in range(num_images):
        rel = f"imgs/x{i}.jpg"
        img = tmp_path / rel
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\xff\xd8jpegbytes%d" % i)
        rels.append({"image_id": f"x{i}", "path": rel,
                     "source": "materialized", "split": "train"})
    pd.DataFrame({"images": [rels]}).to_parquet(
        ds / "associations.parquet", index=False)
    return ds, tmp_path


@pytest.fixture(scope="module")
def fin():
    """The final-evaluation script, for its pure narrative helper."""
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import evaluate_pilot100_final as mod
    return mod


def _paired(state: str, ci: tuple[float, float], ref: str = "MG",
            metric: str = "tga", units: int = 72, rows: int = 1215):
    return {"comparisons": {state: {f"vs_{ref}": {metric: {
        "diff": round(sum(ci) / 2, 6), "ci": list(ci), "num_units": units,
        "num_rows": rows,
        "point_estimates": {"row_a": 0.3984, "row_b": 0.4362,
                            "row_diff": -0.0378, "entity_a": 0.4312,
                            "entity_b": 0.4427}}}}}}


# ── 1. the prediction parquet's own bytes ─────────────────────────

class TestPredictionBytesAreBound:
    """Contract v1 never read the parquet, so any edit to it passed."""

    def test_the_untouched_pair_is_accepted(self, pair):
        path, fp = pair
        assert verify_sidecar(path, fp) == []

    def test_the_sidecar_records_the_parquets_own_hash(self, pair):
        path, _ = pair
        found = json.loads(sidecar_path(path).read_text())
        assert found["prediction_sha256"] == sha256_file(path)
        assert found["num_rows"] == ROWS
        assert found["contract_version"] == PROVENANCE_CONTRACT_VERSION

    def test_a_fabricated_model_output_is_refused(self, pair):
        """The mutation that matters most: invented model text, same shape,
        same row count, same query ids — and under v1 it verified."""
        path, fp = pair
        _set_column(path, "raw_output",
                    ["A completely fabricated response."] * ROWS)
        reasons = verify_sidecar(path, fp)
        assert reasons, "fabricated output must not verify"
        assert reasons[0].startswith("prediction_sha256:")
        assert "MODIFIED" in reasons[0]

    def test_a_flipped_filr_score_bit_is_refused(self, pair):
        """``is_finer_than_target`` is the column FILR counts.  Flipping one
        False to True moves the published metric without touching a single
        model output."""
        path, fp = pair
        import pyarrow.parquet as pq
        before = sum(1 for v in pq.read_table(path)
                     .column("is_finer_than_target") if v.as_py())
        _set_column(path, "is_finer_than_target", [True] + [False] * (ROWS - 1))
        after = sum(1 for v in pq.read_table(path)
                    .column("is_finer_than_target") if v.as_py())
        assert after == before + 1, "the mutation must actually move FILR"
        assert verify_sidecar(path, fp), "a moved metric must not verify"

    def test_a_single_corrupted_byte_is_refused(self, pair):
        path, fp = pair
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(bytes(raw))
        assert verify_sidecar(path, fp)

    def test_a_deleted_row_is_refused(self, pair):
        path, fp = pair
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        pq.write_table(table.slice(0, table.num_rows - 1), path)
        reasons = verify_sidecar(path, fp)
        assert reasons
        # both the hash and the row count independently catch it
        assert any(r.startswith("prediction_sha256:") for r in reasons)
        assert any(r.startswith("num_rows:") for r in reasons)

    def test_an_appended_row_is_refused(self, pair):
        path, fp = pair
        import pandas as pd
        df = pd.read_parquet(path)
        extra = df.iloc[[0]].copy()
        extra["query_id"] = "q_foreign"
        pd.concat([df, extra]).to_parquet(path, index=False)
        assert verify_sidecar(path, fp)

    def test_a_row_count_claimed_but_unreadable_is_refused(self, pair):
        """A file that is not a parquet at all must not pass as one whose
        row count simply could not be checked."""
        path, fp = pair
        path.write_bytes(b"this is not a parquet file")
        reasons = verify_sidecar(path, fp)
        assert any("not a readable parquet" in r for r in reasons), reasons

    def test_a_sidecar_for_a_missing_parquet_raises(self, tmp_path):
        """Writing a sidecar with nothing to bind would later refuse with a
        confusing reason, or be trusted for a file someone else wrote."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            write_sidecar(tmp_path / "absent.parquet", _fp())

    def test_a_v1_sidecar_is_refused_with_a_migration_reason(self, pair):
        """Thirty real sidecars were in this state.  The refusal must name
        the version gap and say what to do, not list missing fields."""
        path, fp = pair
        found = json.loads(sidecar_path(path).read_text())
        found.pop("contract_version")
        found.pop("prediction_sha256")
        found.pop("adapter_contract")
        sidecar_path(path).write_text(json.dumps(found))
        reasons = verify_sidecar(path, fp)
        assert len(reasons) == 1, reasons
        assert "provenance contract v1" in reasons[0]
        assert f"v{PROVENANCE_CONTRACT_VERSION}" in reasons[0]
        assert "rebind_prediction_provenance.py" in reasons[0]


# ── 2. the adapter contract ───────────────────────────────────────

class TestAdapterContractIsBound:
    def test_the_contract_covers_weights_and_config(self, tmp_path):
        ad = _make_adapter(tmp_path)
        contract = adapter_contract(ad)
        assert set(contract["files"]) == set(ADAPTER_CONTRACT_FILES)
        assert contract["missing_files"] == []
        assert contract["sha256"]

    def test_an_edited_config_changes_the_contract_but_not_the_weights(
            self, tmp_path):
        """The measured 11R1 bug: r 16->32, lora_alpha 32->64 and
        target_modules seven->one produce a different model, and
        ``adapter_sha256`` — weights only — did not move."""
        from granunlearn.evaluation.prediction_provenance import adapter_sha256

        before_dir = _make_adapter(tmp_path)
        before_weights = adapter_sha256(before_dir)
        before_contract = adapter_contract(before_dir)["sha256"]

        cfg_path = before_dir / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["r"] = 32
        cfg["lora_alpha"] = 64
        cfg["lora_dropout"] = 0.9
        cfg["target_modules"] = ["q_proj"]
        cfg_path.write_text(json.dumps(cfg, indent=2))

        assert adapter_sha256(before_dir) == before_weights, \
            "weights did not change, so the weights-only hash must not move"
        assert adapter_contract(before_dir)["sha256"] != before_contract, \
            "the contract MUST move: the loaded model is different"

    @pytest.mark.parametrize("key,value", [
        ("r", 32), ("lora_alpha", 64), ("lora_dropout", 0.9),
        ("target_modules", ["q_proj"]),
    ])
    def test_every_behaviour_defining_config_key_is_bound(self, tmp_path,
                                                          key, value):
        ad = _make_adapter(tmp_path)
        before = adapter_contract(ad)["sha256"]
        cfg_path = ad / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg[key] = value
        cfg_path.write_text(json.dumps(cfg, indent=2))
        assert adapter_contract(ad)["sha256"] != before, key

    def test_a_sidecar_carrying_an_edited_config_is_refused(self, tmp_path):
        path = _write_parquet(tmp_path / "predictions_test_MF.parquet")
        ad = _make_adapter(tmp_path)
        fp = _fp(adapter_contract=adapter_contract(ad))
        write_sidecar(path, fp)
        assert verify_sidecar(path, fp) == []

        cfg_path = ad / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["r"] = 32
        cfg_path.write_text(json.dumps(cfg, indent=2))
        # the sidecar still carries the OLD contract; a new run rebuilds its
        # expectation from the adapter on disk, and that is what refuses
        reasons = verify_sidecar(path, _fp(adapter_contract=
                                          adapter_contract(ad)))
        assert reasons
        assert reasons[0].startswith("adapter_contract:")
        assert "adapter_config.json" in reasons[0], \
            "the refusal must name WHICH file differs"

    def test_a_missing_config_is_reported_not_silently_dropped(self,
                                                              tmp_path):
        """Hashing only the files that happen to exist would let an adapter
        directory with its config deleted compare equal to a subset."""
        ad = _make_adapter(tmp_path)
        (ad / "adapter_config.json").unlink()
        contract = adapter_contract(ad)
        assert contract["missing_files"] == ["adapter_config.json"]
        full = adapter_contract(_make_adapter(tmp_path / "other"))
        assert contract["sha256"] != full["sha256"]

    def test_base_has_no_contract_at_all(self):
        assert adapter_contract(None) is None


# ── 3. image bytes ────────────────────────────────────────────────

class TestImageBytesAreBound:
    def test_referenced_paths_come_from_the_associations(self, tmp_path):
        """Not from a directory listing: the manifest must pin the images
        the dataset USES, so a stray file is not evidence and a swapped one
        is."""
        ds, root = _make_dataset(tmp_path, num_images=2)
        (root / "imgs" / "stray.jpg").write_bytes(b"never referenced")
        refs = referenced_image_paths(ds)
        assert refs == ["imgs/x0.jpg", "imgs/x1.jpg"]

    def test_the_manifest_pins_every_referenced_image(self, tmp_path):
        ds, root = _make_dataset(tmp_path, num_images=3)
        write_image_manifest(ds, root)
        frozen = json.loads(image_manifest_path(ds).read_text())
        assert frozen["num_images"] == 3
        assert set(frozen["images"]) == set(referenced_image_paths(ds))
        assert frozen["manifest_sha256"]
        assert verify_image_manifest(ds, root) == []

    def test_the_dataset_fingerprint_binds_the_manifest(self, tmp_path):
        ds, root = _make_dataset(tmp_path)
        assert dataset_fingerprint(ds, root)["image_manifest_sha256"] is None
        write_image_manifest(ds, root)
        fp = dataset_fingerprint(ds, root)
        assert fp["image_manifest_sha256"] == image_manifest_sha256(ds)
        assert fp["num_images_pinned"] == 2
        assert "image_manifest_error" not in fp

    def test_a_missing_manifest_refuses_reuse(self, tmp_path):
        """Fail closed: unbound photographs must not allow a reuse
        decision, however correct everything else about the file is."""
        ds, root = _make_dataset(tmp_path)
        path = _write_parquet(tmp_path / "predictions_test_MF.parquet")
        fp = _fp(dataset=dataset_fingerprint(ds, root))
        write_sidecar(path, fp)
        reasons = verify_sidecar(path, fp)
        assert any(r.startswith("dataset.image_manifest_sha256:")
                   for r in reasons), reasons
        assert "build_image_manifest.py" in "\n".join(reasons)

    def test_a_swapped_photograph_is_detected(self, tmp_path):
        ds, root = _make_dataset(tmp_path, num_images=2)
        write_image_manifest(ds, root)
        assert verify_image_manifest(ds, root) == []
        victim = root / "imgs" / "x0.jpg"
        victim.write_bytes(b"\xff\xd8an entirely different photograph")
        problems = verify_image_manifest(ds, root)
        assert len(problems) == 1
        assert "image bytes differ" in problems[0]
        assert "imgs/x0.jpg" in problems[0]

    def test_a_deleted_photograph_is_detected(self, tmp_path):
        ds, root = _make_dataset(tmp_path)
        write_image_manifest(ds, root)
        (root / "imgs" / "x1.jpg").unlink()
        problems = verify_image_manifest(ds, root)
        assert any("missing on disk" in p for p in problems), problems

    def test_a_newly_referenced_unpinned_image_is_detected(self, tmp_path):
        """Adding an association row that points at an image the manifest
        has never seen must not quietly pass."""
        import pandas as pd

        ds, root = _make_dataset(tmp_path, num_images=1)
        write_image_manifest(ds, root)
        extra = root / "imgs" / "x9.jpg"
        extra.write_bytes(b"\xff\xd8new")
        pd.DataFrame({"images": [[
            {"image_id": "x0", "path": "imgs/x0.jpg", "source": "m",
             "split": "train"},
            {"image_id": "x9", "path": "imgs/x9.jpg", "source": "m",
             "split": "test"}]]}).to_parquet(
            ds / "associations.parquet", index=False)
        problems = verify_image_manifest(ds, root)
        assert any("absent from the frozen manifest" in p for p in problems), \
            problems

    def test_a_refreeze_changes_the_bound_hash_and_refuses_reuse(self,
                                                                tmp_path):
        """Re-pinning the images is exactly what a swapped photograph would
        do to hide itself, so it must invalidate every sidecar."""
        ds, root = _make_dataset(tmp_path)
        write_image_manifest(ds, root)
        first = image_manifest_sha256(ds)
        path = _write_parquet(tmp_path / "predictions_test_MF.parquet")
        fp = _fp(dataset=dataset_fingerprint(ds, root))
        write_sidecar(path, fp)
        assert verify_sidecar(path, fp) == []

        (root / "imgs" / "x0.jpg").write_bytes(b"\xff\xd8different")
        write_image_manifest(ds, root)
        assert image_manifest_sha256(ds) != first
        # as above: the sidecar keeps the old pin, a new run re-derives its
        # expectation from the re-frozen manifest and refuses
        reasons = verify_sidecar(
            path, _fp(dataset=dataset_fingerprint(ds, root)))
        assert any("image_manifest_sha256" in r for r in reasons), reasons

    def test_resolution_never_falls_back_to_the_cwd(self, tmp_path,
                                                    monkeypatch):
        """A measured false negative.  ``resolve_image_path`` used to try a
        bare ``Path(rel)``, which for a repo-relative reference resolves
        against wherever the process happens to run: with the real root
        unavailable, verifying a COPIED dataset from inside a directory that
        happened to contain a same-named file hashed the WRONG photograph
        and reported it as matching."""
        ds = tmp_path / "ds"
        (ds / "imgs").mkdir(parents=True)
        rel = "imgs/x0.jpg"
        real = b"\xff\xd8the dataset's own photograph"
        (ds / rel).write_bytes(real)

        decoy = tmp_path / "elsewhere"
        (decoy / "imgs").mkdir(parents=True)
        (decoy / rel).write_bytes(b"\xff\xd8a decoy from the cwd")
        monkeypatch.chdir(decoy)

        found = resolve_image_path(rel, ds, tmp_path / "no-such-root")
        assert found is not None
        assert Path(found).read_bytes() == real
        assert Path(found).read_bytes() != (decoy / rel).read_bytes()

    def test_an_unresolvable_reference_returns_none_not_a_guess(self,
                                                               tmp_path):
        ds = tmp_path / "ds"
        ds.mkdir()
        monkey_root = tmp_path / "no-such-root"
        assert resolve_image_path("imgs/absent.jpg", ds, monkey_root) is None

    def test_an_absolute_reference_still_resolves(self, tmp_path):
        ds, root = _make_dataset(tmp_path, num_images=1)
        absolute = str(root / "imgs" / "x0.jpg")
        assert resolve_image_path(absolute, ds, tmp_path / "nope") is not None

    def test_the_committed_manifest_roll_up_is_self_consistent(self):
        """Validates the committed roll-up WITHOUT the photographs.

        The 496 pinned bytes are gitignored, so a fresh clone has the
        manifest and not the images.  Recomputing the roll-up over the
        committed per-image hashes, and checking the pinned path set against
        the committed associations parquet, needs neither - and it is the
        part a clean checkout CAN certify.  Rehashing bytes that are not
        there is not a stronger check, it is a failing one.
        """
        path = image_manifest_path(DATA_DIR)
        if not path.exists():
            pytest.skip("no frozen pilot100 image manifest present")
        frozen = json.loads(path.read_text())
        assert frozen["dataset_version"] == "pilot100_v2"
        assert frozen["num_images"] == len(frozen["images"]) == 496
        assert frozen["unresolved_paths"] == []
        # the roll-up is recomputed with the SAME canonicalisation the writer
        # used, imported rather than reimplemented: a second copy of the
        # canonical form here would let the two drift
        assert frozen["manifest_sha256"] == _canonical_rollup(
            {rel: e["sha256"] for rel, e in frozen["images"].items()})
        assert image_manifest_sha256(DATA_DIR) == frozen["manifest_sha256"]

    def test_the_committed_manifest_pins_exactly_the_referenced_paths(self):
        """Also bytes-free: ``associations.parquet`` is committed, so whether
        the manifest pins the images the dataset USES is checkable from a
        clean clone even though the images are not."""
        path = image_manifest_path(DATA_DIR)
        if not path.exists():
            pytest.skip("no frozen pilot100 image manifest present")
        frozen = json.loads(path.read_text())
        refs = referenced_image_paths(DATA_DIR)
        assert refs, "the committed associations reference no images"
        assert set(frozen["images"]) == set(refs)
        for rel, entry in frozen["images"].items():
            assert len(entry["sha256"]) == 64, rel
            assert entry["size_bytes"] > 0, rel

    def test_the_committed_pilot100_manifest_matches_the_image_bytes(self):
        """The full rehash.  Runs only where the photographs are present,
        which is an artifact-bearing checkout and not CI: the bytes are
        gitignored, so on a fresh clone this would report 496 missing files
        and turn a passing evidence check into a red suite."""
        path = image_manifest_path(DATA_DIR)
        if not path.exists():
            pytest.skip("no frozen pilot100 image manifest present")
        refs = referenced_image_paths(DATA_DIR)
        present = [rel for rel in refs
                   if resolve_image_path(rel, DATA_DIR, REPO_ROOT) is not None]
        if not present:
            pytest.skip(
                f"none of the {len(refs)} manifest-pinned photographs are on "
                "disk; they are gitignored, so this checkout cannot rehash "
                "them - see test_the_committed_manifest_roll_up_is_"
                "self_consistent for what is validated without the bytes")
        if len(present) < len(refs):
            pytest.skip(
                f"only {len(present)} of {len(refs)} pinned photographs are "
                "present; a partial rehash that passed would certify less "
                "than it appears to, so it is skipped rather than narrowed")
        assert verify_image_manifest(DATA_DIR, REPO_ROOT) == []


# ── 4. the equivalence narrative ──────────────────────────────────

#: Any wording that asserts the absence of a difference.  Branch 3 spells
#: it "'No significant difference detected' is the most this interval
#: supports" and branch 4 "no significant difference was detected", so the
#: check is case-insensitive and covers both.
NO_SIG_DIFF = "no significant difference"


class TestEquivalenceNarrativeCannotContradictTheNumbers:
    """The note is chosen by significance first, power second."""

    @pytest.mark.parametrize("ci,expect_phrase", [
        ((-0.2714, -0.1709), False),   # B0/MF as measured: wide AND significant
        ((-0.2648, -0.1668), False),   # B1 as measured
        ((-0.3818, -0.2868), False),   # BASE as measured
        ((0.005, 0.045), False),       # narrow, significant, equivalent
        ((0.06, 0.14), False),         # narrow, significant, not equivalent
        ((-0.0752, 0.0263), True),     # B3 as measured: straddles, wide
        ((-0.06, 0.03), True),         # straddles, narrower than the margin
        ((-0.02, 0.02), False),        # straddles AND inside: equivalent
    ])
    def test_the_no_significant_difference_phrase_requires_a_zero_crossing(
            self, fin, ci, expect_phrase):
        """The single invariant whose violation was the reported bug: the
        phrase may appear ONLY beside ``significant_difference == false``."""
        out = fin._equivalence_vs_reference(
            _paired("X", ci), {"X": {"tga": 0.4}, "MG": {"tga": 0.43}})
        block = out["states"]["X"]
        lo, hi = ci
        assert block["significant_difference"] == (lo > 0.0 or hi < 0.0)
        assert (NO_SIG_DIFF in block["power_note"].lower()) is expect_phrase, \
            block["power_note"]
        if block["significant_difference"]:
            assert NO_SIG_DIFF not in block["power_note"].lower()
            # a significant interval that is ALSO inside the margin is
            # equivalence-concluded, which is a different (and legitimate)
            # note: TOST equivalence and a tiny significant difference
            # coexist by design
            if not block["equivalence_concluded"]:
                assert "DIFFERENT FROM MG" in block["power_note"]

    def test_a_wide_significant_interval_states_both_facts(self, fin):
        """B0/MF's real shape.  The note must report the detected difference
        AND the power shortfall, without letting the second undercut the
        first."""
        out = fin._equivalence_vs_reference(
            _paired("B0", (-0.2714, -0.1709)),
            {"B0": {"tga": 0.2107}, "MG": {"tga": 0.4395}})
        b = out["states"]["B0"]
        assert b["significant_difference"] is True
        assert b["equivalence_concluded"] is False
        assert b["ci_half_width"] >= 0.05
        note = b["power_note"]
        assert "DIFFERENT FROM MG" in note
        assert "excludes zero" in note
        assert "IS detected" in note
        assert "could not conclude equivalence" in note
        assert NO_SIG_DIFF not in note.lower()
        # the power caveat is scoped to the equivalence claim, not to the
        # difference that was found
        assert "qualifies the ABSENT equivalence" in note

    def test_a_straddling_wide_interval_is_still_indeterminate(self, fin):
        """B3's real shape — the case the old wording was written for, and
        the only one where that wording is true."""
        out = fin._equivalence_vs_reference(
            _paired("B3", (-0.0752, 0.0263)),
            {"B3": {"tga": 0.3909}, "MG": {"tga": 0.4395}})
        b = out["states"]["B3"]
        assert b["significant_difference"] is False
        assert b["equivalence_concluded"] is False
        assert "INDETERMINATE" in b["power_note"]
        assert "straddles zero" in b["power_note"]
        assert NO_SIG_DIFF in b["power_note"].lower()

    def test_the_committed_report_contains_no_contradiction(self):
        report = REPORTS / "mllmu_pilot100_final_evaluation.json"
        if not report.exists():
            pytest.skip(f"committed evidence not present: {report}")
        states = json.loads(report.read_text())["equivalence_vs_MG"]["states"]
        assert states
        contradicted = [
            s for s, b in states.items()
            if NO_SIG_DIFF in b["power_note"].lower()
            and b["significant_difference"]]
        assert contradicted == [], (
            f"{contradicted} are narrated as showing no significant "
            f"difference while their own significant_difference is true")
        for state, b in states.items():
            lo, hi = b["ci"]
            if lo > 0.0 or hi < 0.0:
                assert "DIFFERENT FROM" in b["power_note"], state
            elif b["ci_half_width"] >= b.get("margin", 0.05):
                assert "INDETERMINATE" in b["power_note"], state

    def test_b0_and_mf_are_narrated_as_different_from_the_oracle(self):
        """Regression pin on the reported defect, against the real numbers:
        B0 == M_F exactly, both significantly below M_G on TGA."""
        report = REPORTS / "mllmu_pilot100_final_evaluation.json"
        if not report.exists():
            pytest.skip(f"committed evidence not present: {report}")
        states = json.loads(report.read_text())["equivalence_vs_MG"]["states"]
        for state in ("B0", "MF"):
            b = states[state]
            assert b["significant_difference"] is True, state
            assert b["ci"][1] < 0.0, state
            assert b["power_note"].startswith("DIFFERENT FROM MG"), state
            assert NO_SIG_DIFF not in b["power_note"].lower(), state
        assert states["B0"]["ci"] == states["MF"]["ci"]
        assert states["B3"]["significant_difference"] is False


# ── 5. the committed consolidated manifest ────────────────────────

class TestConsolidatedManifestIsCommitted:
    """The parquets and sidecars are gitignored, so "all sidecars verify"
    was a claim about one machine's working tree."""

    @pytest.mark.parametrize("rel", [
        "data/reports/mllmu_pilot100_prediction_manifest.json",
        "data/mllmu_hier_pilot100/image_manifest.json",
    ])
    def test_the_pinning_artifacts_are_not_gitignored(self, rel):
        """Without a negation in .gitignore these silently stay untracked
        and the whole exercise buys nothing."""
        out = subprocess.run(["git", "check-ignore", rel], cwd=REPO_ROOT,
                             capture_output=True, text=True)
        assert out.returncode != 0, f"{rel} is gitignored by {out.stdout}"
        assert (REPO_ROOT / rel).exists(), rel

    def test_the_manifest_covers_every_sidecar_on_disk(self):
        path = REPORTS / "mllmu_pilot100_prediction_manifest.json"
        if not path.exists():
            pytest.skip(f"consolidated manifest not present: {path}")
        manifest = json.loads(path.read_text())
        on_disk = {p.name for p in PRED_DIR.glob("predictions*.parquet")}
        if not on_disk:
            pytest.skip("prediction parquets not present")
        listed = {r["parquet"] for r in manifest["predictions"]}
        assert listed == on_disk
        assert manifest["num_predictions"] == len(on_disk) == 30
        assert manifest["provenance_contract_version"] == \
            PROVENANCE_CONTRACT_VERSION
        assert manifest["dataset_version"] == "pilot100_v2"
        assert manifest["num_refusals"] == 0
        assert set(manifest["code_modules_sha256"]) == \
            set(CODE_FINGERPRINT_MODULES)

    def test_every_record_is_pinned_and_refutation_free(self):
        path = REPORTS / "mllmu_pilot100_prediction_manifest.json"
        if not path.exists():
            pytest.skip(f"consolidated manifest not present: {path}")
        manifest = json.loads(path.read_text())
        for rec in manifest["predictions"]:
            assert rec["verify_reasons"] == [], rec["parquet"]
            assert len(rec["parquet_sha256"]) == 64, rec["parquet"]
            assert len(rec["sidecar_sha256"]) == 64, rec["parquet"]
            assert rec["num_rows"] in (6777, 4518, 2259), rec
            assert rec["image_manifest_sha256"] == \
                manifest["image_manifest_sha256"], rec["parquet"]
            assert rec["rebind_assertion"], rec["parquet"]
            assert "RE-STAMPED, NOT REGENERATED" in rec["rebind_assertion"]
        layouts = {}
        for rec in manifest["predictions"]:
            layouts[rec["layout"]] = layouts.get(rec["layout"], 0) + 1
        assert layouts == {"all_split": 4, "train_val": 17, "test_only": 9}

    def test_the_records_match_the_sidecars_on_disk(self):
        path = REPORTS / "mllmu_pilot100_prediction_manifest.json"
        if not path.exists() or not PRED_DIR.exists():
            pytest.skip("manifest or predictions not present")
        if not any(PRED_DIR.glob("predictions*.parquet")):
            pytest.skip("prediction parquets not present")
        manifest = json.loads(path.read_text())
        for rec in manifest["predictions"]:
            parquet = PRED_DIR / rec["parquet"]
            if not parquet.exists():
                pytest.skip(f"{parquet.name} not present on this machine")
            assert sha256_file(parquet) == rec["parquet_sha256"], rec
            sidecar = sidecar_path(parquet)
            assert sha256_file(sidecar) == rec["sidecar_sha256"], rec
            found = json.loads(sidecar.read_text())
            assert found["prediction_sha256"] == rec["parquet_sha256"]
            assert found["contract_version"] == rec["contract_version"]

    def test_the_rollup_is_reproducible_from_the_records(self):
        """A single hash over all thirty, so one tampered record moves it."""
        import hashlib

        path = REPORTS / "mllmu_pilot100_prediction_manifest.json"
        if not path.exists():
            pytest.skip(f"consolidated manifest not present: {path}")
        manifest = json.loads(path.read_text())
        canon = json.dumps(
            [{k: v for k, v in r.items() if k != "verify_reasons"}
             for r in manifest["predictions"]],
            sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canon.encode()).hexdigest() == \
            manifest["manifest_sha256"]

    def test_the_rebind_records_its_own_limits(self):
        """The honest part: rebinding pins bytes forward, it cannot prove
        anything about how they were produced."""
        if not PRED_DIR.exists():
            pytest.skip("predictions not present")
        sidecars = sorted(PRED_DIR.glob("*.provenance.json"))
        if not sidecars:
            pytest.skip("no sidecars present")
        for sc in sidecars:
            found = json.loads(sc.read_text())
            assert found["contract_version"] == PROVENANCE_CONTRACT_VERSION
            rebind = found.get("rebind")
            assert rebind, f"{sc.name} was not re-stamped"
            assert rebind["from_contract_version"] == 1
            assert len(rebind["preconditions_passed"]) == 3
            assert "RE-STAMPED, NOT REGENERATED" in rebind["assertion"]
            assert rebind["original_sidecar_sha256"]


# ── 6. the frozen image manifest is committed content ─────────────

class TestFrozenImageManifestContents:
    def test_the_manifest_pins_hashes_not_paths_alone(self):
        path = image_manifest_path(DATA_DIR)
        if not path.exists():
            pytest.skip(f"no frozen manifest at {path}")
        frozen = json.loads(path.read_text())
        for rel, entry in frozen["images"].items():
            assert len(entry["sha256"]) == 64, rel
            assert entry["size_bytes"] > 0, rel
        assert IMAGE_MANIFEST_NAME == path.name

    def test_refreeze_is_refused_without_an_explicit_flag(self):
        """Silently re-pinning images would make existing evidence describe
        bytes that are no longer on disk."""
        script = REPO_ROOT / "scripts" / "build_image_manifest.py"
        if not image_manifest_path(DATA_DIR).exists():
            pytest.skip("no frozen manifest to protect")
        out = subprocess.run(
            [sys.executable, str(script), "--tag", "pilot100"],
            cwd=REPO_ROOT, capture_output=True, text=True,
            # PREPEND, do not replace: overwriting PYTHONPATH drops whatever
            # the calling environment resolved pyarrow and the granunlearn
            # package through, and the child then fails on an import the
            # parent can do.  That failure looks like a broken guard.
            env={**os.environ,
                 "PYTHONPATH": os.pathsep.join(
                     ["src", *[p for p in os.environ.get("PYTHONPATH", "")
                              .split(os.pathsep) if p]])})
        assert out.returncode != 0, out.stdout + out.stderr
        assert "already frozen" in out.stdout + out.stderr
        assert "--allow-refreeze" in out.stdout + out.stderr
