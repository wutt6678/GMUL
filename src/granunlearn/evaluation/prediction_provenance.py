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
artifact hashes, generation configuration, and the code fingerprint (git
commit plus the hashes of the modules that turn queries into scores).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("prediction_provenance")

#: Modules whose behaviour determines what a prediction file MEANS.  A
#: change in any of them can change scores for identical bytes on disk,
#: so the code fingerprint covers them explicitly rather than trusting the
#: commit hash alone (a dirty tree is possible).
CODE_FINGERPRINT_MODULES = (
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/query_generation.py",
    "src/granunlearn/evaluation/scoring.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/evaluation/image_splits.py",
    "src/granunlearn/imaging.py",
)

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
    """Commit + per-module hashes for the scoring/generation code."""
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
    bump, so the bytes are hashed too.
    """
    version = dataset_version(data_dir)
    hashes = {}
    for name in ("associations.parquet", "queries.parquet", "manifest.json"):
        p = data_dir / name
        if p.exists():
            hashes[name] = sha256_file(p)
    return {"version": version,
            "artifacts_sha256": hashes,
            "data_dir": _relative_to(data_dir, repo_root)}


def base_model_revision(model_id: str) -> str | None:
    """The local HF cache revision of the base model, if resolvable."""
    ref = (Path.home() / ".cache" / "huggingface" / "hub"
           / ("models--" + model_id.replace("/", "--")) / "refs" / "main")
    return ref.read_text().strip() if ref.exists() else None


def adapter_sha256(adapter_dir: Path | None) -> str | None:
    """Hash of the LoRA weights, or None for BASE (no adapter)."""
    if adapter_dir is None:
        return None
    weights = Path(adapter_dir) / "adapter_model.safetensors"
    return sha256_file(weights) if weights.exists() else None


@dataclass(frozen=True)
class PredictionFingerprint:
    """Everything that must match before a prediction file may be reused."""

    experiment_id: str
    checkpoint_id: str
    adapter_sha256: str | None
    base_model_revision: str | None
    dataset: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    num_rows: int | None = None

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
        )


def write_sidecar(parquet: Path, fingerprint: PredictionFingerprint) -> Path:
    path = sidecar_path(parquet)
    path.write_text(json.dumps(fingerprint.to_dict(), indent=2))
    log.info("wrote prediction sidecar %s", path.name)
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

    ``created_utc`` and ``num_rows`` are informational and never cause a
    refusal: a file regenerated a minute later from identical inputs is
    still the same evidence.
    """
    reasons: list[str] = []
    if not parquet.exists():
        return [f"missing parquet {parquet.name}"]
    found = read_sidecar(parquet)
    if found is None:
        return [f"{parquet.name} has no provenance sidecar — it predates "
                f"provenance-validated reuse and cannot be attributed to "
                f"any adapter, dataset, configuration or code revision"]
    want = expected.to_dict()
    for key in ("experiment_id", "checkpoint_id", "adapter_sha256",
                "base_model_revision"):
        if found.get(key) != want[key]:
            reasons.append(f"{key}: file has {found.get(key)!r}, this run "
                           f"expects {want[key]!r}")
    f_ds, w_ds = found.get("dataset") or {}, want["dataset"]
    if f_ds.get("version") != w_ds["version"]:
        reasons.append(f"dataset.version: file has {f_ds.get('version')!r}, "
                       f"this run expects {w_ds['version']!r}")
    for name, sha in (w_ds.get("artifacts_sha256") or {}).items():
        if (f_ds.get("artifacts_sha256") or {}).get(name) != sha:
            reasons.append(f"dataset.{name}: hash differs (the dataset "
                           f"bytes changed since this file was generated)")
    f_gc, w_gc = found.get("generation_config") or {}, want["generation_config"]
    for key in GENERATION_CONFIG_KEYS:
        if f_gc.get(key) != w_gc.get(key):
            reasons.append(f"generation_config.{key}: file has "
                           f"{f_gc.get(key)!r}, this run expects "
                           f"{w_gc.get(key)!r}")
    f_code, w_code = found.get("code") or {}, want["code"]
    if f_code.get("git_commit") != w_code["git_commit"]:
        reasons.append(f"code.git_commit: file has "
                       f"{f_code.get('git_commit')!r}, this run expects "
                       f"{w_code['git_commit']!r}")
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

