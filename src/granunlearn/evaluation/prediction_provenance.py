"""Prediction-file provenance: make parquet reuse a verified decision.

The defect this repairs
-----------------------
``evaluate_pilot100_final.py`` and ``select_unlearning_checkpoints.py``
reused any correctly named parquet that happened to exist:

    ppath = predictions_dir / f"predictions_test_{state_id}.parquet"
    if ppath.exists():
        return load_predictions_parquet(ppath), True

A filename says which STATE a file is *labelled* as, not what produced
it.  A parquet left behind by an earlier dataset version, a different
adapter, a different batch size or an older scoring module loads silently
and is reported as if it were current — and the report's retrospective
SHA-256 identifies the bytes that were loaded without proving they came
from the expected weights, data, configuration or code.

The contract
------------
Every generation pass writes a SIDECAR next to its parquet recording the
full fingerprint of what produced it.  Before reuse the sidecar is
compared against what THIS run would produce, dimension by dimension, and
any mismatch — or a missing sidecar — refuses the file with the reasons
listed.  Refusal is loud: the caller regenerates or fails, never guesses.

Dimensions bound: adapter bytes, base-model revision, dataset version and
artifact hashes, generation configuration, and the hashes of the modules
that turn queries into scores.  The git commit is recorded alongside them
for the reader, but is not itself a refusal condition — see
:func:`code_fingerprint`.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("prediction_provenance")

#: Modules whose behaviour determines what a prediction file MEANS.  A
#: change in any of them can change scores for identical bytes on disk,
#: so the code fingerprint covers them explicitly rather than trusting the
#: commit hash alone (a dirty tree is possible).
#:
#: The four ``schema`` modules are included because they define how a
#: persisted row is INTERPRETED: rename a field on ``PredictionRecord`` and
#: every already-written parquet silently means something else, even though
#: not one byte of scoring logic moved.  ``paired_ci``, ``selection`` and
#: this module are deliberately excluded — they consume predictions rather
#: than produce or interpret them, so a change there cannot alter what an
#: existing parquet says, and including this module would make the contract
#: self-referential (editing the verifier would invalidate the evidence it
#: verifies).
CODE_FINGERPRINT_MODULES = (
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/query_generation.py",
    "src/granunlearn/evaluation/scoring.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/evaluation/image_splits.py",
    "src/granunlearn/imaging.py",
    "src/granunlearn/schema/association.py",
    "src/granunlearn/schema/hierarchy.py",
    "src/granunlearn/schema/prediction.py",
    "src/granunlearn/schema/query.py",
)

#: Bumped whenever the set of BINDING fields changes, so a sidecar written
#: under an older contract is refused with a comprehensible reason instead
#: of a bewildering list of missing-field mismatches.  Version 1 had no
#: ``prediction_sha256``, no adapter contract and no image manifest, so it
#: could not detect a mutated parquet, an edited ``adapter_config.json``, or
#: a swapped photograph.
PROVENANCE_CONTRACT_VERSION = 2

#: Every file in an adapter directory that changes what the adapter DOES.
#: ``adapter_model.safetensors`` alone is not the contract: rank, alpha,
#: dropout and ``target_modules`` all live in ``adapter_config.json``, and
#: loading the same weights under a different config produces a different
#: model.  Measured during 11R1 — editing ``r`` 16->32, ``lora_alpha``
#: 32->64 and ``target_modules`` from seven projections to one left a
#: weights-only hash byte-identical.
ADAPTER_CONTRACT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
)

#: Frozen manifest of every referenced image's bytes, committed next to the
#: dataset artifacts it pins.
IMAGE_MANIFEST_NAME = "image_manifest.json"

#: Generation-config keys that change decoded bytes when they change.
GENERATION_CONFIG_KEYS = (
    "batch_size", "image_batch_size", "max_new_tokens", "do_sample",
    "max_image_pixels", "max_length",
)

SIDECAR_SUFFIX = ".provenance.json"

#: Commit holding the Iteration 11 (pilot100_v1) reports that Iteration
#: 11R replaces.  Every regenerated report names it in a ``supersedes``
#: block: the v1 numbers stay readable in git history, but nothing on disk
#: may present them as current, and nothing may quietly inherit a v1
#: prediction parquet (see :func:`verify_sidecar`).
SUPERSEDED_V1_COMMIT = "3850461"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(parquet: Path) -> Path:
    """``predictions_test_MF.parquet.provenance.json``."""
    return Path(str(parquet) + SIDECAR_SUFFIX)


def git_commit(repo_root: Path) -> str | None:
    """HEAD commit, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def git_dirty(repo_root: Path) -> bool | None:
    """Whether the tree has uncommitted changes (None if unavailable).

    Recorded because a fingerprint taken on a dirty tree cannot be
    reproduced from the commit alone.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def code_fingerprint(repo_root: Path) -> dict[str, Any]:
    """Commit + per-module hashes for the scoring/generation code.

    ``modules_sha256`` is the BINDING half of this fingerprint and
    ``git_commit``/``git_dirty`` are the informational half.  The commit
    hash is neither necessary nor sufficient for deciding reuse: it changes
    on edits to lane scripts, tests and prose that cannot move a single
    decoded token, while a dirty tree can still have byte-identical
    scoring modules.  Hashing the modules that actually turn queries into
    scores is both stricter where it matters and stable where it does not,
    which is what lets a multi-hour regeneration resume across commits.
    """
    modules = {}
    for rel in CODE_FINGERPRINT_MODULES:
        p = repo_root / rel
        modules[rel] = sha256_file(p) if p.exists() else None
    return {
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "modules_sha256": modules,
    }


def _relative_to(path: Path, root: Path) -> str:
    """``path`` relative to ``root``, or the path itself when it is not
    under it.

    ``Path.relative_to`` RAISES rather than returning a non-relative
    answer, so an absolute ``data_dir`` outside ``repo_root`` — a tmp dir,
    a mounted dataset, a fingerprint compared across two checkouts — would
    turn a provenance check into a crash.  Recording the absolute path is
    the honest fallback: it still differs from the expected value, so the
    comparison still refuses.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def dataset_version(data_dir: Path) -> str | None:
    """The version the frozen dataset declares for itself, read not assumed.

    Reports record this so a reader can tell which dataset generation a set
    of numbers belongs to; Iteration 11R re-froze pilot-100 as ``_v2`` and
    a report that omitted the version would silently mix the two.
    """
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text()).get("version")


def dataset_fingerprint(data_dir: Path,
                        repo_root: Path) -> dict[str, Any]:
    """Version + artifact hashes of the dataset a pass was generated on.

    The version alone is not enough: it is a string someone can forget to
    bump, so the bytes are hashed too.  The hashed artifacts are the three
    tabular/JSON files PLUS the roll-up of the frozen image manifest, since
    ``associations.parquet`` only pins image PATHS — without the manifest a
    swapped photograph is invisible to this fingerprint.
    """
    version = dataset_version(data_dir)
    hashes = {}
    for name in ("associations.parquet", "queries.parquet", "manifest.json"):
        p = data_dir / name
        if p.exists():
            hashes[name] = sha256_file(p)
    manifest_sha = image_manifest_sha256(data_dir)
    out: dict[str, Any] = {
        "version": version,
        "artifacts_sha256": hashes,
        "data_dir": _relative_to(data_dir, repo_root),
        "image_manifest_sha256": manifest_sha,
    }
    if manifest_sha is None:
        out["image_manifest_error"] = (
            f"no frozen {IMAGE_MANIFEST_NAME} in {data_dir} — referenced "
            f"image bytes are unbound; run scripts/build_image_manifest.py")
    else:
        try:
            out["num_images_pinned"] = json.loads(
                image_manifest_path(data_dir).read_text())["num_images"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return out


def base_model_revision(model_id: str) -> str | None:
    """The local HF cache revision of the base model, if resolvable."""
    ref = (Path.home() / ".cache" / "huggingface" / "hub"
           / ("models--" + model_id.replace("/", "--")) / "refs" / "main")
    return ref.read_text().strip() if ref.exists() else None


#: Reference states live under their own checkpoint root; every other
#: checkpoint_id is an unlearning candidate.  Mirrors the paths
#: ``evaluate_pilot100_final`` and ``select_unlearning_checkpoints``
#: generate from.
REFERENCE_STATES = frozenset({"BASE", "MF", "MG", "MN"})


def resolve_adapter_dir(checkpoint_id: str, repo_root: Path,
                        tag: str) -> Path | None:
    """The adapter directory a checkpoint_id was generated from, or None.

    Resolution is by NAMING CONVENTION and must be cross-checked against
    the recorded ``adapter_sha256`` by the caller.  Searching for a matching
    weights hash instead looks simpler but is ambiguous: ``selected/B0`` and
    the reference ``MF`` hold byte-identical weights because B0 IS the no-op
    baseline, and twelve of the thirty pilot-100 sidecars match more than
    one directory.  Ambiguity is harmless while only the weights are hashed
    and stops being harmless once ``adapter_config.json`` is part of the
    contract, since two copies of the same weights need not share a config.
    """
    if checkpoint_id == "BASE":
        return None
    if checkpoint_id in REFERENCE_STATES:
        return (Path(repo_root) / "data" / "checkpoints" / f"mllmu_{tag}"
                / checkpoint_id / "adapters")
    return (Path(repo_root) / "data" / "checkpoints" / f"mllmu_{tag}_unlearn"
            / checkpoint_id / "adapters")


def adapter_sha256(adapter_dir: Path | None) -> str | None:
    """Hash of the LoRA weights, or None for BASE (no adapter).

    Kept for continuity and diagnostics — it says whether the WEIGHTS moved
    — but it is not the contract.  Use :func:`adapter_contract`.
    """
    if adapter_dir is None:
        return None
    weights = Path(adapter_dir) / "adapter_model.safetensors"
    return sha256_file(weights) if weights.exists() else None


def adapter_contract(adapter_dir: Path | None) -> dict[str, Any] | None:
    """Per-file hashes of everything that defines what an adapter DOES.

    Returns ``None`` for BASE (no adapter at all).  For a real adapter the
    result carries a per-file map plus a single roll-up ``sha256``, and
    lists any :data:`ADAPTER_CONTRACT_FILES` that are MISSING rather than
    quietly hashing a subset — an adapter directory without its config is
    not a loadable contract, so it must not produce a fingerprint that
    compares equal to a complete one.
    """
    if adapter_dir is None:
        return None
    root = Path(adapter_dir)
    files: dict[str, str] = {}
    missing: list[str] = []
    for name in ADAPTER_CONTRACT_FILES:
        p = root / name
        if p.exists():
            files[name] = sha256_file(p)
        else:
            missing.append(name)
    return {
        "files": files,
        "sha256": _canonical_rollup(files),
        "missing_files": missing,
        "adapter_dir": str(root),
    }


def _canonical_rollup(mapping: dict[str, str]) -> str:
    """sha256 over a key-sorted, compactly separated JSON object.

    One hash over many files, stable under dict ordering and whitespace so
    two checkouts that hold the same bytes compute the same roll-up.
    """
    canon = json.dumps({k: mapping[k] for k in sorted(mapping)},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def referenced_image_paths(data_dir: Path) -> list[str]:
    """Every image path the frozen associations reference, sorted + deduped.

    Read from ``associations.parquet`` rather than from a directory listing,
    because the manifest must pin the images the dataset actually USES: a
    stray file in an image directory is not evidence, and a referenced file
    that has been swapped IS.
    """
    import pandas as pd

    assoc_path = Path(data_dir) / "associations.parquet"
    if not assoc_path.exists():
        return []
    assoc = pd.read_parquet(assoc_path, columns=["images"])
    refs: set[str] = set()
    for images in assoc["images"]:
        for rec in (images if images is not None else []):
            path = rec.get("path") if isinstance(rec, dict) else None
            if path:
                refs.add(str(path))
    return sorted(refs)


def resolve_image_path(rel: str, data_dir: Path,
                       repo_root: Path) -> Path | None:
    """Locate a referenced image, or None if it is not on disk.

    Association rows store repo-relative paths, but the same manifest is
    read from temporary checkouts and mounted datasets, so an absolute
    spelling and a dataset-relative one are tried too.

    The CWD is deliberately NOT a candidate.  A bare ``Path(rel)`` looks
    like a harmless fallback for absolute paths, but for a repo-relative
    reference it silently resolves against wherever the process happens to
    be running: verifying a copied dataset from inside a real checkout then
    hashes the CHECKOUT's photographs and reports them as matching, which
    is exactly the false negative this manifest exists to prevent.
    Absolute references are handled explicitly instead.
    """
    as_path = Path(rel)
    candidates = ([as_path] if as_path.is_absolute()
                  else [Path(repo_root) / rel, Path(data_dir) / rel])
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def build_image_manifest(data_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Hash every referenced image's BYTES into one frozen manifest.

    The dataset fingerprint already covers ``associations.parquet``, which
    is where image paths live — so version 1 of this contract pinned the
    POINTER and not the photograph.  Replacing a species' photograph with a
    different one left every sidecar valid while the model answered about a
    different image.
    """
    images: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for rel in referenced_image_paths(data_dir):
        found = resolve_image_path(rel, data_dir, repo_root)
        if found is None:
            unresolved.append(rel)
            continue
        images[rel] = {
            "sha256": sha256_file(found),
            "size_bytes": found.stat().st_size,
        }
    return {
        "dataset_version": dataset_version(data_dir),
        "data_dir": _relative_to(Path(data_dir), Path(repo_root)),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "num_images": len(images),
        "manifest_sha256": _canonical_rollup(
            {k: v["sha256"] for k, v in images.items()}),
        "unresolved_paths": unresolved,
        "images": images,
    }


def image_manifest_path(data_dir: Path) -> Path:
    return Path(data_dir) / IMAGE_MANIFEST_NAME


def write_image_manifest(data_dir: Path, repo_root: Path) -> Path:
    """Build and freeze the image manifest next to the dataset artifacts."""
    manifest = build_image_manifest(data_dir, repo_root)
    path = image_manifest_path(data_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("froze image manifest %s: %d images, rollup %s",
             path.name, manifest["num_images"],
             manifest["manifest_sha256"][:16])
    return path


def image_manifest_sha256(data_dir: Path) -> str | None:
    """Roll-up hash of the FROZEN manifest, or None when there is none.

    ``None`` is a refusal, not a pass: :func:`verify_sidecar` rejects an
    expected fingerprint that cannot bind image bytes.
    """
    path = image_manifest_path(data_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("manifest_sha256")
    except json.JSONDecodeError as exc:
        log.error("unreadable image manifest %s: %s", path, exc)
        return None


def verify_image_manifest(data_dir: Path,
                          repo_root: Path) -> list[str]:
    """Every way the images ON DISK disagree with the frozen manifest.

    Binding the manifest hash catches a rebuilt manifest; this catches a
    swapped photograph, which is the mutation the hash alone cannot see.
    Empty list == the referenced bytes are exactly the frozen ones.
    """
    path = image_manifest_path(data_dir)
    if not path.exists():
        return [f"no frozen image manifest at {path} — image bytes are "
                f"unbound and cannot be verified"]
    frozen = json.loads(path.read_text()).get("images") or {}
    problems: list[str] = []
    for rel in referenced_image_paths(data_dir):
        want = frozen.get(rel)
        if want is None:
            problems.append(f"{rel}: referenced but absent from the frozen "
                            f"manifest")
            continue
        found = resolve_image_path(rel, data_dir, repo_root)
        if found is None:
            problems.append(f"{rel}: in the manifest but missing on disk")
            continue
        actual = sha256_file(found)
        if actual != want.get("sha256"):
            problems.append(f"{rel}: image bytes differ from the frozen "
                            f"manifest (expected {want.get('sha256')}, "
                            f"found {actual})")
    for rel in sorted(set(frozen) - set(referenced_image_paths(data_dir))):
        problems.append(f"{rel}: pinned by the manifest but no longer "
                        f"referenced by the dataset")
    return problems


def parquet_num_rows(path: Path) -> int | None:
    """Row count read from the parquet FOOTER only — no data deserialized.

    Cheap enough to run on every reuse decision, which is what makes the
    recorded ``num_rows`` a binding fact rather than a comment.
    """
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(Path(path)).metadata.num_rows)
    except Exception as exc:  # corrupt, truncated, or not a parquet
        log.error("unreadable prediction parquet %s: %s", path, exc)
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Interpreter and library versions, recorded but NOT binding.

    Recorded because of a gap found during 11R1: the ``.venv`` that produced
    the Iteration 11R evidence disappeared from this machine, and nothing in
    that evidence could say which interpreter had been used.  The reports
    recorded python/torch/transformers/peft VERSIONS and the GPU, but never
    an executable path, and contract-v1 sidecars recorded no environment at
    all — so substituting a different interpreter was undetectable from the
    artifacts, and only survived because the replacement happened to carry
    byte-identical library versions.

    It stays informational for the same reason ``git_commit`` does: a library
    patch upgrade should not silently invalidate hours of correct evidence.
    But a reader can now see WHICH interpreter produced a file, and a future
    confirmatory protocol can promote it to binding.
    """
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "numpy", "pandas",
                 "pyarrow"):
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "package_versions": versions,
    }


@dataclass(frozen=True)
class PredictionFingerprint:
    """Everything that must match before a prediction file may be reused.

    Two kinds of field live here.  Most describe the INPUTS and are checked
    against what this run expects.  ``prediction_sha256`` and ``num_rows``
    are different: they describe the OUTPUT FILE the sidecar sits beside,
    and are checked against that file directly, so they catch edits made
    AFTER generation rather than drift in the inputs.
    """

    experiment_id: str
    checkpoint_id: str
    adapter_sha256: str | None
    base_model_revision: str | None
    dataset: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    num_rows: int | None = None
    #: weights + ``adapter_config.json`` hashed together (see
    #: :func:`adapter_contract`); weights alone are not the contract.
    adapter_contract: dict[str, Any] | None = None
    #: sha256 of the prediction parquet itself, written by
    #: :func:`write_sidecar` from the bytes on disk.
    prediction_sha256: str | None = None
    #: which version of this contract produced the sidecar.
    contract_version: int = PROVENANCE_CONTRACT_VERSION
    #: interpreter/library versions — informational, never a refusal.
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        experiment_id: str,
        checkpoint_id: str,
        repo_root: Path,
        data_dir: Path,
        model_id: str,
        adapter_dir: Path | None,
        generation_config: dict[str, Any],
        num_rows: int | None = None,
    ) -> "PredictionFingerprint":
        return cls(
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            adapter_sha256=adapter_sha256(adapter_dir),
            base_model_revision=base_model_revision(model_id),
            dataset=dataset_fingerprint(data_dir, repo_root),
            generation_config={
                k: generation_config.get(k) for k in GENERATION_CONFIG_KEYS},
            code=code_fingerprint(repo_root),
            created_utc=datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            num_rows=num_rows,
            adapter_contract=adapter_contract(adapter_dir),
            environment=environment_fingerprint(),
        )


def write_sidecar(parquet: Path, fingerprint: PredictionFingerprint) -> Path:
    """Write the sidecar, stamping it with the parquet's OWN bytes.

    ``prediction_sha256`` and ``num_rows`` are read back from the file that
    was just written rather than taken from the caller's intent, so the
    sidecar always describes the artifact beside it.  Writing one for a
    parquet that does not exist is a programming error and raises: a
    sidecar with no file to bind would later refuse with a confusing
    reason, or worse, be trusted for a file regenerated by someone else.
    """
    parquet = Path(parquet)
    if not parquet.exists():
        raise FileNotFoundError(
            f"cannot write a provenance sidecar for {parquet}: the "
            f"prediction parquet does not exist, so there are no bytes to "
            f"bind")
    stamped = replace(fingerprint,
                      prediction_sha256=sha256_file(parquet),
                      num_rows=parquet_num_rows(parquet))
    path = sidecar_path(parquet)
    path.write_text(json.dumps(stamped.to_dict(), indent=2))
    log.info("wrote prediction sidecar %s (parquet %s, %s rows)",
             path.name, stamped.prediction_sha256[:12] if
             stamped.prediction_sha256 else "?", stamped.num_rows)
    return path


def read_sidecar(parquet: Path) -> dict[str, Any] | None:
    path = sidecar_path(parquet)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.error("unreadable sidecar %s: %s", path, exc)
        return None


def verify_sidecar(parquet: Path,
                   expected: PredictionFingerprint) -> list[str]:
    """Every reason this parquet may NOT be reused (empty == reusable).

    The checks fall into two groups.  INPUT checks compare the sidecar's
    recorded adapter, dataset, generation config and module hashes against
    what this run expects.  OUTPUT checks compare the sidecar against the
    parquet it sits beside — ``prediction_sha256`` and ``num_rows`` — and
    need no ``expected`` value at all, because a file edited after its
    sidecar was written disagrees with its own record.

    ``created_utc``, ``code.git_commit``, ``code.git_dirty`` and
    ``environment`` are informational and never cause a refusal: a file
    regenerated a minute later from identical inputs is still the same
    evidence, and the commit hash moves on edits to lane scripts, tests and
    prose that cannot change a decoded token, while ``code.modules_sha256``
    — which IS binding — hashes exactly the modules that can.

    Before contract version 2 there were no OUTPUT checks, so a fabricated
    ``raw_output``, a flipped ``is_finer_than_target`` bit or a corrupted
    byte all verified cleanly; measured during 11R1.
    """
    reasons: list[str] = []
    if not parquet.exists():
        return [f"missing parquet {parquet.name}"]
    found = read_sidecar(parquet)
    if found is None:
        return [f"{parquet.name} has no provenance sidecar — it predates "
                f"provenance-validated reuse and cannot be attributed to "
                f"any adapter, dataset, configuration or code revision"]
    f_ver = found.get("contract_version", 1)
    if f_ver != PROVENANCE_CONTRACT_VERSION:
        return [
            f"{parquet.name} was written under provenance contract v{f_ver}, "
            f"this build enforces v{PROVENANCE_CONTRACT_VERSION}. v1 bound no "
            f"hash of the prediction parquet itself, no adapter "
            f"configuration and no image bytes, so it cannot demonstrate "
            f"that the file on disk is the file that was generated. "
            f"Re-generate it, or re-stamp it with "
            f"scripts/rebind_prediction_provenance.py if the bytes are "
            f"known-good and only the record is stale."]

    # ---- OUTPUT checks: sidecar versus the parquet beside it ----
    recorded_sha = found.get("prediction_sha256")
    if not recorded_sha:
        reasons.append(f"prediction_sha256: the sidecar records no hash of "
                       f"{parquet.name}, so post-generation edits to the "
                       f"file are undetectable")
    else:
        actual_sha = sha256_file(parquet)
        if actual_sha != recorded_sha:
            reasons.append(
                f"prediction_sha256: {parquet.name} has been MODIFIED since "
                f"its sidecar was written (sidecar records {recorded_sha}, "
                f"file is now {actual_sha}) — its contents can no longer be "
                f"attributed to the recorded adapter and configuration")
    recorded_rows = found.get("num_rows")
    actual_rows = parquet_num_rows(parquet)
    if actual_rows is None:
        reasons.append(f"{parquet.name} is not a readable parquet, so its "
                       f"row count cannot be checked against the "
                       f"{recorded_rows!r} its sidecar claims")
    elif recorded_rows is not None and actual_rows != recorded_rows:
        reasons.append(f"num_rows: sidecar claims {recorded_rows}, the file "
                       f"holds {actual_rows}")

    # ---- INPUT checks ----
    want = expected.to_dict()
    for key in ("experiment_id", "checkpoint_id", "adapter_sha256",
                "base_model_revision"):
        if found.get(key) != want[key]:
            reasons.append(f"{key}: file has {found.get(key)!r}, this run "
                           f"expects {want[key]!r}")
    f_ac, w_ac = found.get("adapter_contract"), want.get("adapter_contract")
    if w_ac is not None or f_ac is not None:
        if f_ac is None:
            reasons.append("adapter_contract: the sidecar records none, so "
                           "adapter_config.json (rank, alpha, dropout, "
                           "target_modules) is unbound")
        elif w_ac is None:
            reasons.append("adapter_contract: this run expects no adapter "
                           "(BASE), but the file records one")
        elif f_ac.get("sha256") != w_ac.get("sha256"):
            changed = sorted(
                n for n in set(f_ac.get("files") or {})
                | set(w_ac.get("files") or {})
                if (f_ac.get("files") or {}).get(n)
                != (w_ac.get("files") or {}).get(n))
            reasons.append(
                f"adapter_contract: differs on {changed or 'roll-up'} — "
                f"weights-only hashing would have missed an edited "
                f"adapter_config.json, which changes rank/alpha/"
                f"target_modules and therefore the model itself")
    f_ds, w_ds = found.get("dataset") or {}, want["dataset"]
    if f_ds.get("version") != w_ds["version"]:
        reasons.append(f"dataset.version: file has {f_ds.get('version')!r}, "
                       f"this run expects {w_ds['version']!r}")
    for name, sha in (w_ds.get("artifacts_sha256") or {}).items():
        if (f_ds.get("artifacts_sha256") or {}).get(name) != sha:
            reasons.append(f"dataset.{name}: hash differs (the dataset "
                           f"bytes changed since this file was generated)")
    w_img = w_ds.get("image_manifest_sha256")
    if w_img is None:
        reasons.append(
            f"dataset.image_manifest_sha256: this run has no frozen image "
            f"manifest ({w_ds.get('image_manifest_error') or 'missing'}), so "
            f"the photographs behind these predictions are unbound and no "
            f"reuse decision can be justified")
    elif f_ds.get("image_manifest_sha256") != w_img:
        reasons.append(
            f"dataset.image_manifest_sha256: file has "
            f"{f_ds.get('image_manifest_sha256')!r}, this run expects "
            f"{w_img!r} (the referenced image bytes were re-froze or "
            f"changed since this file was generated)")
    f_gc, w_gc = found.get("generation_config") or {}, want["generation_config"]
    for key in GENERATION_CONFIG_KEYS:
        if f_gc.get(key) != w_gc.get(key):
            reasons.append(f"generation_config.{key}: file has "
                           f"{f_gc.get(key)!r}, this run expects "
                           f"{w_gc.get(key)!r}")
    f_code, w_code = found.get("code") or {}, want["code"]
    for rel, sha in (w_code.get("modules_sha256") or {}).items():
        if (f_code.get("modules_sha256") or {}).get(rel) != sha:
            reasons.append(f"code.{rel}: module hash differs (scoring or "
                           f"generation logic changed since this file was "
                           f"generated)")
    return reasons


def validate_prediction_coverage(
    predictions: list[Any],
    expected_query_ids: list[str] | set[str],
    experiment_id: str,
    checkpoint_id: str,
) -> list[str]:
    """Exact row-level validation of a loaded prediction set.

    The Iteration-11 check was ``len({p.query_id for p in preds} &
    expected) == len(expected)`` — an INTERSECTION SIZE.  That passes for
    a file carrying duplicate rows, extra rows from another split or
    another dataset, rows scored by a different checkpoint, or rows from a
    different experiment, because all of those are invisible to an
    intersection: it only ever asks "is everything I need present?", never
    "is anything else here?".

    Every paired CI and every cross-state comparison assumes the two states
    were scored over EXACTLY the same probes, so this is strict by design:
    exact set equality, no duplicates, correct row count, and every row
    labelled with the expected experiment and checkpoint.
    """
    reasons: list[str] = []
    expected = set(expected_query_ids)
    seen: dict[str, int] = {}
    wrong_ckpt: list[str] = []
    wrong_exp: list[str] = []
    for p in predictions:
        seen[p.query_id] = seen.get(p.query_id, 0) + 1
        if p.checkpoint_id != checkpoint_id:
            wrong_ckpt.append(p.query_id)
        if p.experiment_id != experiment_id:
            wrong_exp.append(p.query_id)
    got = set(seen)
    duplicates = sorted(q for q, n in seen.items() if n > 1)
    if duplicates:
        reasons.append(f"{len(duplicates)} duplicated query_id(s), e.g. "
                       f"{duplicates[:5]} — a prediction set must hold one "
                       f"row per probe")
    extra = sorted(got - expected)
    if extra:
        reasons.append(f"{len(extra)} row(s) outside the expected query set, "
                       f"e.g. {extra[:5]} — another split, dataset or run "
                       f"was concatenated into this file")
    missing = sorted(expected - got)
    if missing:
        reasons.append(f"{len(missing)} expected query_id(s) absent, e.g. "
                       f"{missing[:5]}")
    if len(predictions) != len(expected):
        reasons.append(f"row count {len(predictions)} != expected "
                       f"{len(expected)}")
    if wrong_ckpt:
        reasons.append(f"{len(wrong_ckpt)} row(s) labelled with a different "
                       f"checkpoint_id than {checkpoint_id!r}, e.g. "
                       f"{sorted(wrong_ckpt)[:5]}")
    if wrong_exp:
        reasons.append(f"{len(wrong_exp)} row(s) labelled with a different "
                       f"experiment_id than {experiment_id!r}, e.g. "
                       f"{sorted(wrong_exp)[:5]}")
    return reasons

