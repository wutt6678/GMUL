"""Iteration 11C stage 5 — generate the confirmation predictions, and NOTHING else.

    python scripts/evaluate_confirmation_split.py --device cuda:0
    python scripts/evaluate_confirmation_split.py --device cuda:0 --state B3
    python scripts/evaluate_confirmation_split.py --verify-only
    python scripts/evaluate_confirmation_split.py --list-outstanding

Scores exactly the three frozen states — B3, B0 and MG — over all 1,209
confirmation queries, under the one uniform generation configuration the
freeze binds, writing a provenance sidecar beside each parquet.

Four modes, four different scopes
---------------------------------
``--state X`` generates X, verifies X, and exits on X's own merits.  It says
nothing about the other two states.  That is what makes sharding the three
passes across GPUs work: a lane that finishes its state has SUCCEEDED.  An
earlier version fell through to the global completeness check, so whichever
lane finished first exited nonzero merely because the other two were still
running, and a chain that records a failed lane stops before its own gate even
when that gate would have passed.

``--verify-only`` is the global gate: all three states, reported at once, and
it is what ``analyze_confirmation_split.py`` re-checks before assembling.

``--list-outstanding`` prints one line per state that does not currently
VERIFY and exits 0.  The lane uses it to decide what to run, so "outstanding"
means "fails verification" and not "has no filename": a state whose parquet
exists but whose sidecar or generation-order record is stale is regenerated
rather than skipped and then refused by the gate with no step that would ever
rewrite it.  stdout is the data channel in this mode and the logger is silenced
into stderr, because the lane iterates over what it reads.

The preflight, run in EVERY mode
--------------------------------
Before any state is generated, reused or verified, four things are checked
against the bytes on disk rather than trusted from the freeze's record of them:

* every hash in ``code.analysis_scripts_sha256``, plus the separately pinned
  ``primary_test.implementation.sha256`` for ``paired_ci.py`` — the module that
  implements the sign-flip test, the paired CI and the Holm step-down, and is
  deliberately outside ``CODE_FINGERPRINT_MODULES``.  A pin that is recorded but
  never compared describes the protocol instead of enforcing it.
* ``verify_image_manifest`` over the 402 pinned photographs, so a swapped or
  truncated file is refused.
* PER-QUERY image resolution, because ``ReferenceStateGenerator._render_prompt``
  silently drops an image it cannot resolve and generates TEXT-ONLY: an
  image-route probe rendered without its photograph is still decoded, still
  scored and still sealed with a provenance-valid sidecar, and would be
  reported as visual evidence having measured a text probe.  That module is in
  ``CODE_FINGERPRINT_MODULES``, so editing it would invalidate all 30 committed
  exploratory sidecars; the gap is closed in front of it instead.
* the base-model revision.  The freeze pins ``c202236…`` and the weights are
  loaded from a snapshot resolved AT that revision, not through the repo id —
  ``from_pretrained(model_id)`` resolves via the MUTABLE
  ``~/.cache/huggingface/hub/models--*/refs/main``.  A moved ref is refused
  rather than worked around, because ``PredictionFingerprint.build`` records the
  LIVE ref, so the sidecar would bind a base model that is not the frozen one
  and ``verify_sidecar`` — comparing against an expectation read from the same
  moved ref — would not notice.

Why this is not ``evaluate_pilot100_final.py`` pointed at another directory
---------------------------------------------------------------------------
That script is correct for what it does, and what it does is an EXPLORATORY
final evaluation.  Six of its assumptions are false here, and each would have
to be removed rather than reconfigured — which is what "merely changing its
data directory" would have left in place:

1. It opens ``mllmu_pilot100_unlearning_selection.json`` to decide which
   candidates to score.  The confirmation's three states are frozen; reading a
   selection report at all re-opens a closed decision, which sealed-split
   invariant 5 forbids ("the confirmation split never enters candidate
   selection or any other go/no-go decision").  This script does not know the
   selection report's path.
2. It hard-fails without the reference-state gate's all-split parquets
   (``predictions_{BASE,MF,MG,MN}.parquet``), because it reads them to quantify
   the batch-layout noise floor.  Invariant 4 forbids the confirmation split
   from entering the gate, so no such run exists to read.
3. Its first integrity invariant is that B0 is the MF adapter copied unchanged,
   so B0's paired difference against MF must be exactly 0.0 on all six metrics
   and their raw outputs must match query for query.  MF is EXCLUDED from the
   confirmation — the freeze proves B0 is the no-op by hashing both adapters
   instead — so that invariant cannot be evaluated, and a script still
   asserting it would either fail outright or get quietly weakened.
4. It runs a TOST equivalence test against MG at δ = 0.05.  The freeze records
   ``mg_equivalence_test_run: false`` and the margin's role as "REPORTING
   YARDSTICK, not a tested margin".
5. It publishes a ``test_split_exposure`` statement about the gate having
   already generated the whole test split.  There is no such exposure here.
6. Its batch size, bootstrap count and CI level are CLI-tunable defaults.  For
   the confirmation every one of those is FROZEN, so a tunable default is a
   route to drifting from the protocol.  This script accepts ``--device`` and
   ``--state`` only, and reads the rest from the freeze.

Why no metric is computed here
------------------------------
Sealed-split invariant 6: "partial state results are not inspected before
every scored state has finished", because looking at B3 before MG has run is
how a layout or scoring change gets made mid-flight and the score-exactly-once
property is lost.  The enforcement is structural rather than disciplinary:
this module imports no aggregation code at all — no ``paired_ci``, no
``hierarchy_metrics``, no ``compute_metrics`` — so a run that generated one
state and stopped has no way to print a rate, an interval or a p-value.
Per-query scoring (``score_query``) does happen here, because that is what
turns a decoded string into a row, but a row is not a result.

Aggregation lives in ``scripts/analyze_confirmation_split.py``, which refuses
to run until all three sidecars verify.

Resumption
----------
A state whose parquet exists with a sidecar that verifies against THIS run's
adapter bytes, dataset version and artifact hashes, generation configuration
and code fingerprint is reused; anything else is regenerated.  That is what
makes an interrupted pass resumable on a shared box without re-spending GPU
hours, and it is a verified decision rather than a filename match.

The generation-order record is part of that verification and is checked inside
:func:`verify_state`, not only at assembly.  Checking it later left the reuse
path able to accept a parquet generated over a different query order, and
batched greedy decoding is not bit-stable across batch compositions — so the
three states could have been compared across two layouts while every sidecar
verified and nothing refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.prediction_provenance import (
    PredictionFingerprint,
    adapter_contract,
    base_model_revision,
    dataset_version,
    referenced_image_paths,
    resolve_image_path,
    sha256_file,
    validate_prediction_coverage,
    verify_image_manifest,
    verify_sidecar,
    write_sidecar,
)
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    ReferenceStateGenerator,
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.evaluation.scoring import score_query
from granunlearn.logging_utils import setup_logger
from granunlearn.schema import PredictionRecord, QueryRecord

log = setup_logger("evaluate_confirmation_split")

#: The confirmation dataset and its report names.  Distinct from the
#: exploratory ones at every level, so a confirmation prediction cannot be
#: mistaken for an exploratory one by a loader that keys on either name.
TAG = "pilot100"
CONFIRM_TAG = "confirm100"
CONFIRM_DATASET_DIR = f"data/mllmu_hier_{CONFIRM_TAG}"
FREEZE_REPORT = f"data/reports/mllmu_{TAG}_confirmation_freeze.json"
CONFIRM_REPORT = f"data/reports/mllmu_{TAG}_confirmation_power.json"
EXPERIMENT_ID = f"mllmu_{CONFIRM_TAG}_iter11c"

#: The three states, in the order the freeze lists them.  Asserted against the
#: freeze rather than trusted from this constant: if the protocol ever scored
#: a fourth state, this script has to refuse instead of quietly generating
#: three and reporting completeness.
EXPECTED_SCORED_STATES = ("B3", "B0", "MG")

#: The six generation fields the sidecar fingerprint binds.  Read from the
#: freeze, never defaulted here, so the configuration that produced a
#: prediction is the configuration the protocol froze.
GENERATION_CONFIG_KEYS = ("batch_size", "image_batch_size", "max_new_tokens",
                          "do_sample", "max_image_pixels", "max_length")

#: The freeze's own cross-check of those values against the module defaults.
#: Re-derived here because a freeze that recorded ``module_defaults_match:
#: true`` over a module that has since changed is a stale claim, and this is
#: the last point before GPU hours are spent where it is cheap to catch.
MODULE_DEFAULTS = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
    "max_length": DEFAULT_MAX_LENGTH,
}


def _sha256_id_list(ids: list[str]) -> str:
    """The stage-3 seal's own canonicalization of a query-id list.

    Sorted, then JSON with no whitespace: the same form
    ``confirmation_split_seal`` hashes, so the value computed here is directly
    comparable to the sealed one.  Two spellings of a hash are two hashes.
    """
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def load_frozen_protocol(repo_root: Path) -> dict[str, Any]:
    """The freeze, and a refusal if it is absent or refused anything.

    A freeze carrying refusals is a protocol that did not pass its own checks.
    Generating predictions under it would spend GPU hours on a design the
    repository has already declined to certify, and the resulting parquets
    would look exactly like valid ones.
    """
    path = repo_root / FREEZE_REPORT
    if not path.exists():
        raise SystemExit(
            f"REFUSED — {FREEZE_REPORT} does not exist. Run "
            f"python scripts/freeze_confirmation_protocol.py --tag {TAG} "
            "first: the confirmation is scored against a frozen protocol, and "
            "without one there is nothing to score against.")
    freeze = json.loads(path.read_text())
    refusals = freeze.get("refusals") or []
    if refusals:
        raise SystemExit(
            f"REFUSED — the committed freeze carries {len(refusals)} "
            f"refusal(s), so the protocol it describes did not pass its own "
            "checks:\n  " + "\n  ".join(str(r) for r in refusals))
    return freeze


def frozen_generation_config(freeze: dict[str, Any]) -> dict[str, Any]:
    """The uniform generation configuration, read off the freeze.

    Cross-checked against the module defaults the freeze itself claims to
    match.  A mismatch means a default moved after the freeze was written, so
    the frozen numbers and the code that would produce them disagree — and
    since ``max_new_tokens`` truncates a decoded answer, that disagreement
    changes what is being measured.
    """
    gen = freeze.get("generation") or {}
    missing = [k for k in GENERATION_CONFIG_KEYS if k not in gen]
    if missing:
        raise SystemExit(
            f"REFUSED — the freeze's generation block does not state "
            f"{missing}, so the uniform configuration is not fully frozen and "
            "this run would have to choose it")
    config = {k: gen[k] for k in GENERATION_CONFIG_KEYS}
    drifted = {k: (config[k], v) for k, v in MODULE_DEFAULTS.items()
               if config[k] != v}
    if drifted:
        raise SystemExit(
            f"REFUSED — the frozen generation configuration disagrees with "
            f"the module defaults that would produce it: {drifted} as "
            "(frozen, module). The freeze records module_defaults_match="
            f"{gen.get('module_defaults_match')!r}, so either the freeze is "
            "stale or a default moved; re-freeze before scoring.")
    if config["do_sample"] is not False:
        raise SystemExit(
            f"REFUSED — the frozen configuration sets do_sample="
            f"{config['do_sample']!r}. The confirmation is greedy: a sampled "
            "decode is not reproducible, and an unreproducible prediction "
            "cannot be verified and reused, which is the whole resumption "
            "contract.")
    return config


def scored_states(freeze: dict[str, Any]) -> tuple[str, ...]:
    """Exactly the states the freeze scores, asserted against the expected three.

    Read from the freeze and compared, rather than taken from
    ``EXPECTED_SCORED_STATES``: the constant says what this script was written
    for, the freeze says what the protocol requires, and the two being equal is
    the claim worth making.
    """
    states = tuple(freeze.get("states", {}).get("scored") or ())
    if states != EXPECTED_SCORED_STATES:
        raise SystemExit(
            f"REFUSED — the freeze scores {list(states)} but this script was "
            f"written for {list(EXPECTED_SCORED_STATES)}. Scoring a different "
            "set is a protocol change, and it has to be made in the freeze "
            "first rather than discovered here.")
    excluded = freeze.get("states", {}).get("excluded") or {}
    #: MF and BASE in particular: the exploratory evaluator generates both,
    #: and a state that is excluded but generated anyway is an unreported
    #: comparison somebody can later read a result off.
    for state in ("MF", "BASE", "MN", "B1", "B2", "B2R"):
        if state not in excluded:
            raise SystemExit(
                f"REFUSED — the freeze does not record {state} as excluded, so "
                "this script cannot know whether generating it would be inside "
                "the protocol")
    return states


def adapter_for(state: str, freeze: dict[str, Any],
                repo_root: Path) -> Path:
    """The adapter directory the freeze pins for ``state``, bytes and all.

    The freeze records both the repository-relative path and the adapter
    contract's roll-up hash, so the directory is resolved and then RE-HASHED:
    a checkpoint that moved or was retrained between the freeze and this run
    would otherwise be scored and reported as the frozen one, which is the
    single failure that would make the confirmation uninterpretable.
    """
    entry = (freeze.get("checkpoints") or {}).get(state)
    if not entry:
        raise SystemExit(
            f"REFUSED — the freeze binds no checkpoint entry for {state}")
    rel = entry.get("adapter_dir")
    if not rel:
        raise SystemExit(f"REFUSED — the freeze records no adapter_dir for "
                         f"{state}")
    adapter_dir = repo_root / rel
    pinned = (entry.get("adapter_contract") or {}).get("sha256")
    if not adapter_dir.exists():
        raise SystemExit(
            f"REFUSED — {state}'s adapter directory {rel} does not exist. The "
            "adapters are gitignored, so this is expected on a checkout that "
            "did not train them; it is NOT evidence that the checkpoint moved, "
            "and it cannot be scored from anywhere else.")
    live = adapter_contract(adapter_dir) or {}
    if live.get("missing_files"):
        raise SystemExit(
            f"REFUSED — {state}'s adapter contract is missing "
            f"{live['missing_files']} under {rel}; an adapter directory "
            "without its config is not a loadable contract")
    if pinned and live.get("sha256") != pinned:
        raise SystemExit(
            f"REFUSED — {state}'s adapter contract is {live.get('sha256')} but "
            f"the freeze pinned {pinned}: the checkpoint has moved since the "
            "protocol was frozen, so scoring it would not score the state the "
            "confirmation is defined against")
    return adapter_dir


def confirmation_queries(repo_root: Path,
                         freeze: dict[str, Any]) -> list[QueryRecord]:
    """Every confirmation query, in the parquet's own row order.

    ALL 1,209 rows are scored for every state.  There is no split filter to
    apply — the builder wrote every query as ``test`` — and filtering anyway
    would be how a subset quietly became the reported set.

    The id list is checked against the stage-3 seal, so this run cannot score
    a different split than the one the protocol froze.  Row ORDER is preserved
    and hashed separately: the seal hashes the SORTED ids because it identifies
    a set of probes, but batched greedy decoding is not bit-stable across batch
    compositions, so the order the rows are fed in is part of the generation
    contract and has to be the same for all three states.
    """
    data_dir = repo_root / CONFIRM_DATASET_DIR
    qpath = data_dir / "queries.parquet"
    if not qpath.exists():
        raise SystemExit(
            f"REFUSED — {qpath.relative_to(repo_root)} does not exist. Build "
            f"it with python scripts/build_confirmation_split.py --tag {TAG} "
            "before scoring.")
    queries = load_queries_parquet(qpath)
    seal = ((freeze.get("confirmation_size") or {}).get(
        "frozen_at_stage_3_before_any_scoring") or {})
    if seal.get("sealed") is not True:
        raise SystemExit(
            "REFUSED — the freeze does not record the split as sealed, so "
            "there is no frozen query set to score against")
    want_ids = seal.get("confirmation_query_id_list_sha256")
    got_ids = _sha256_id_list([q.query_id for q in queries])
    if want_ids and got_ids != want_ids:
        raise SystemExit(
            f"REFUSED — the split on disk holds query ids hashing to "
            f"{got_ids}, but the seal binds {want_ids}. This is not the split "
            "the protocol froze; rebuild it or re-freeze, and do not score "
            "this one.")
    want_n = seal.get("confirmation_query_ids")
    if want_n is not None and len(queries) != want_n:
        raise SystemExit(
            f"REFUSED — the split holds {len(queries)} queries but the seal "
            f"binds {want_n}")
    return queries


def generation_order_sha256(queries: list[QueryRecord]) -> str:
    """A hash of the ROW ORDER the queries will be generated in.

    Distinct from the seal's sorted-id hash on purpose.  Two runs over the same
    1,209 probes in different orders are not the same generation, because left
    padded batched greedy decoding flips near-ties when the batch composition
    changes — measured on the exploratory split as 122 of 2,259 decodes.  So
    "identical query ordering" is a claim that needs its own hash, and every
    state's sidecar is written with this one beside it.
    """
    return hashlib.sha256(
        json.dumps([q.query_id for q in queries],
                   separators=(",", ":")).encode()).hexdigest()


def predictions_path(repo_root: Path, state: str) -> Path:
    return repo_root / CONFIRM_DATASET_DIR / "predictions" / \
        f"predictions_{state}.parquet"


def generation_order_path(repo_root: Path, state: str) -> Path:
    """The record of which query order a state was generated over.

    Its own function because three places now need the same spelling — the
    writer, :func:`verify_state` and the lane — and a path built slightly
    differently in one of them is a record nobody reads.
    """
    return predictions_path(repo_root, state).parent / \
        f"predictions_{state}.generation_order.json"


def frozen_base_model_revision(freeze: dict[str, Any]) -> str:
    """The base-model revision the protocol pins, cross-checked in the freeze.

    The freeze records it four times — once under ``generation`` and once per
    scored checkpoint — because it is a property of the base weights every
    state shares.  All four are compared rather than one trusted: a freeze
    whose checkpoints disagree with its own generation block is describing two
    different base models, and picking either silently would be a choice this
    script is not entitled to make.
    """
    want = (freeze.get("generation") or {}).get("base_model_revision")
    if not want:
        raise SystemExit(
            "REFUSED — the freeze's generation block records no "
            "base_model_revision, so there is no pinned base model to load and "
            "this run would take whatever the local cache resolves to")
    checkpoints = freeze.get("checkpoints") or {}
    disagree = {st: (entry or {}).get("base_model_revision")
                for st, entry in checkpoints.items()
                if (entry or {}).get("base_model_revision") != want}
    if disagree:
        raise SystemExit(
            f"REFUSED — the freeze pins base_model_revision {want} under "
            f"generation but these checkpoints record something else: "
            f"{disagree}. Every state is the same base model plus an adapter, "
            "so a disagreement means the freeze describes two base models and "
            "the comparison between states would not be between adapters.")
    return want


def pinned_model_source(model_id: str, revision: str) -> str:
    """A model source that IS the pinned revision, not a ref that can move.

    ``ReferenceStateGenerator`` calls ``from_pretrained(model_id)`` with no
    ``revision``, so handing it the repo id resolves through
    ``~/.cache/huggingface/hub/models--<id>/refs/main`` — a MUTABLE pointer.
    Resolving the pinned snapshot to a directory first means the bytes loaded
    are the pinned commit no matter where ``refs/main`` points afterwards,
    which is the same reasoning ``salmu.adapter.locate_repo_pinned`` uses.

    The live ref is compared to the pin as well, and a mismatch refuses rather
    than merely loading the right bytes.  ``PredictionFingerprint.build``
    records ``base_model_revision(model_id)``, which reads that same live ref,
    so a moved ref would write a sidecar binding a revision that is not the
    frozen one — and because ``verify_sidecar`` compares the sidecar against an
    expectation built from the same live ref, that sidecar would verify.  A
    check whose two sides are read from the same mutable source is not a check.
    """
    live = base_model_revision(model_id)
    if live is None:
        raise SystemExit(
            f"REFUSED — no local HF cache ref for {model_id}, so the revision "
            f"that would be recorded in the sidecar is unresolvable and the "
            f"freeze's pin {revision} cannot be compared against anything")
    if live != revision:
        raise SystemExit(
            f"REFUSED — the local HF cache ref for {model_id} points at "
            f"{live} but the freeze pins {revision}. Loading the pinned "
            "snapshot would still be correct, but the sidecar records the LIVE "
            "ref, so it would bind a base model that is not the frozen one — "
            "and verify_sidecar compares it against an expectation read from "
            "the same moved ref, so it would not notice. Restore the ref to "
            "the pinned revision, or re-freeze over the new one; do not score "
            "across the difference.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            f"REFUSED — huggingface_hub is not importable ({exc}), so the "
            f"pinned revision {revision} of {model_id} cannot be resolved to a "
            "snapshot and the base model cannot be pinned") from exc
    try:
        #: ``local_files_only``: a confirmation run must not depend on the
        #: network, and must not silently substitute a freshly downloaded
        #: snapshot for the one the protocol was frozen over.
        snapshot = Path(snapshot_download(
            repo_id=model_id, revision=revision, local_files_only=True))
    except Exception as exc:
        raise SystemExit(
            f"REFUSED — no local snapshot of {model_id} at the pinned revision "
            f"{revision}: {exc}. The pinned bytes have to be present before "
            "scoring; downloading them during the run would make the base "
            "model a moving input") from exc
    if not snapshot.is_dir():
        raise SystemExit(
            f"REFUSED — the pinned snapshot of {model_id}@{revision} resolved "
            f"to {snapshot}, which is not a directory")
    return str(snapshot)


def verify_confirmation_images(repo_root: Path, queries: list[QueryRecord],
                               by_assoc: dict) -> list[str]:
    """Every way the photographs this run would render are not the frozen ones.

    Two checks, because they catch different failures:

    * ``verify_image_manifest`` re-hashes the 402 pinned photographs against
      ``image_manifest.json``, which is what catches a swapped or truncated
      file — the mutation the manifest's own hash cannot see.
    * a PER-QUERY resolution check, because ``ReferenceStateGenerator`` drops
      an image it cannot resolve and generates TEXT-ONLY with no warning:
      ``_render_prompt`` appends ``{"type": "image"}`` only inside ``if
      p.exists()``, and only when the query's ``image_ids`` match one of ITS
      OWN association's images.  An image-route probe rendered without its
      image is still decoded, still scored, still sealed with a
      provenance-valid sidecar — and is then reported as image evidence when
      it measured a text probe.

    The second check lives here rather than in ``reference_eval.py`` because
    that module is in ``CODE_FINGERPRINT_MODULES``: editing it would change the
    code fingerprint inside all 30 committed exploratory sidecars and refuse
    their reuse.  The gap is closed in front of the generator instead.
    """
    data_dir = repo_root / CONFIRM_DATASET_DIR
    problems = [f"image manifest: {p}"
                for p in verify_image_manifest(data_dir, repo_root)]
    #: Which routes must carry an image.  Derived from the queries themselves
    #: rather than hardcoded, so a route this script has not seen is reported
    #: as unclassifiable instead of being silently treated as text.
    image_routes = {"image_to_text", "image_text_to_text"}
    text_routes = {"text_to_text"}
    for q in queries:
        if q.route in image_routes and not q.image_ids:
            problems.append(
                f"{q.query_id}: route {q.route} is an image route but the "
                "query names no image_ids, so the generator would render it "
                "text-only and it would be reported as image evidence")
            continue
        if q.route in text_routes and q.image_ids:
            problems.append(
                f"{q.query_id}: route {q.route} is a text route but the query "
                f"names image_ids {sorted(q.image_ids)}")
            continue
        if q.route not in image_routes | text_routes:
            problems.append(
                f"{q.query_id}: route {q.route!r} is neither a known image "
                "route nor a known text route, so whether it must resolve an "
                "image cannot be determined")
            continue
        if not q.image_ids:
            continue
        assoc = by_assoc.get(q.association_id)
        if assoc is None:
            problems.append(
                f"{q.query_id}: names association {q.association_id}, which is "
                "absent from associations.parquet")
            continue
        ref = next((i for i in assoc.images if i.image_id in q.image_ids), None)
        if ref is None:
            problems.append(
                f"{q.query_id}: image_ids {sorted(q.image_ids)} match none of "
                f"the {len(assoc.images)} images on its own association "
                f"{q.association_id}, so _render_prompt would find no img_ref "
                "and generate text-only")
            continue
        if resolve_image_path(ref.path, data_dir, repo_root) is None:
            problems.append(
                f"{q.query_id}: image {ref.path} is referenced but absent on "
                "disk, so _render_prompt would skip the image and generate "
                "text-only")
    return problems


def verify_frozen_code(repo_root: Path, freeze: dict[str, Any]) -> list[str]:
    """Every code hash the freeze pins, compared against the bytes on disk NOW.

    The freeze binds two sets and this checks both:

    * ``code.analysis_scripts_sha256`` — the seven scripts that turn scores
      into claims.  Recording them without checking them is a description of
      the protocol rather than an enforcement of it: sealed-split invariant 7
      says a failed primary claim is reported as failed and nothing downstream
      re-tunes on it, and that holds only if the code producing the verdict
      cannot change unnoticed between the freeze and the analysis.
    * ``primary_test.implementation.sha256`` — ``paired_ci.py``, pinned
      SEPARATELY because it is not in ``CODE_FINGERPRINT_MODULES`` (adding it
      would invalidate the 30 committed exploratory sidecars).  It implements
      the sign-flip test, the paired CI and the Holm step-down, so it is the
      one module the primary claims depend on that the fingerprint does not
      already cover.

    Returns the problems rather than raising so both scripts can report all of
    them at once and fold them into their own refusal lists.
    """
    problems: list[str] = []
    bound = (freeze.get("code") or {}).get("analysis_scripts_sha256") or {}
    if not bound:
        problems.append(
            "the freeze binds no analysis_scripts_sha256, so the code that "
            "turns scores into claims is unpinned")
    for rel, want in sorted(bound.items()):
        path = repo_root / rel
        if not path.exists():
            problems.append(f"{rel}: bound by the freeze but absent from the "
                            "repository")
            continue
        got = sha256_file(path)
        if got != want:
            problems.append(
                f"{rel}: the freeze binds {want} but the file is now {got} — "
                "the code that produces the confirmation verdict has changed "
                "since the protocol was frozen, so re-freeze before scoring "
                "rather than scoring under a protocol the repository no "
                "longer matches")
    impl = ((freeze.get("primary_test") or {}).get("implementation") or {})
    rel, want = impl.get("module"), impl.get("sha256")
    if not rel or not want:
        problems.append(
            "the freeze's primary_test.implementation names no module or no "
            "sha256, so the code implementing the sign-flip test, the paired "
            "CI and the Holm step-down is unpinned")
    else:
        path = repo_root / rel
        if not path.exists():
            problems.append(f"{rel}: pinned by primary_test.implementation but "
                            "absent from the repository")
        else:
            got = sha256_file(path)
            if got != want:
                problems.append(
                    f"{rel}: primary_test.implementation pins {want} but the "
                    f"file is now {got} — this module implements "
                    f"{impl.get('functions')} and is deliberately NOT in "
                    "CODE_FINGERPRINT_MODULES, so this hash is the only thing "
                    "pinning the procedure that decides the primary claims")
    return problems


def expected_fingerprint(state: str, adapter_dir: Path, repo_root: Path,
                         model_id: str, generation_config: dict[str, Any],
                         n_queries: int) -> PredictionFingerprint:
    """What a parquet for ``state`` has to have been produced by.

    ``data_dir`` is the confirmation dataset, so the fingerprint binds
    ``confirm100_v1`` and its own artifact hashes — a reused exploratory
    prediction would fail on dataset version alone, which is the point.
    """
    return PredictionFingerprint.build(
        experiment_id=EXPERIMENT_ID, checkpoint_id=state,
        repo_root=repo_root, data_dir=repo_root / CONFIRM_DATASET_DIR,
        model_id=model_id, adapter_dir=adapter_dir,
        generation_config=generation_config, num_rows=n_queries)


def verify_state(state: str, repo_root: Path, freeze: dict[str, Any],
                 queries: list[QueryRecord], by_assoc: dict,
                 model_id: str, generation_config: dict[str, Any],
                 ) -> tuple[list[PredictionRecord] | None, list[str]]:
    """Verify one state's parquet without generating and without aggregating.

    Returns the rows and no reasons when the file is reusable, or ``None`` and
    every reason it is not.  Reasons are returned rather than raised so a
    completeness check can report all three states at once: "MG is missing" is
    a more useful message than whichever state happened to be checked first.

    This is TOTAL: nothing on the path raises.  ``adapter_for`` refuses with a
    ``SystemExit`` when a checkpoint is absent or has moved, and that is the
    right behaviour for a caller about to spend GPU hours on it, but here it is
    converted into a reason — because the lane asks this function which states
    still need work, and an oracle that raises on the first absent checkpoint
    cannot tell the lane anything.

    The generation-order record is verified HERE and not only at assembly.
    Checking it later means ``generate_state``'s reuse path can accept a
    parquet that was generated over a different query order, and batched greedy
    decoding is not bit-stable across batch compositions — so the three states
    would be compared across two layouts while every sidecar verified.
    """
    ppath = predictions_path(repo_root, state)
    if not ppath.exists():
        return None, [
            f"{ppath.name} does not exist — this state has not been generated"]
    try:
        adapter_dir = adapter_for(state, freeze, repo_root)
    except SystemExit as exc:
        return None, [str(exc)]
    expected = expected_fingerprint(state, adapter_dir, repo_root, model_id,
                                    generation_config, len(queries))
    reasons = verify_sidecar(ppath, expected)
    if reasons:
        return None, [f"{ppath.name}: {r}" for r in reasons]
    order_sha = generation_order_sha256(queries)
    opath = generation_order_path(repo_root, state)
    if not opath.exists():
        return None, [
            (f"{ppath.name} has a verifying sidecar but no "
             f"{opath.name} beside it, so the query order it was generated "
             "over is unrecorded and cannot be shown to match the other "
             "states — regenerate it rather than trusting it")]
    try:
        record = json.loads(opath.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [
            (f"{opath.name} is unreadable ({exc}), so the generation order it "
             "should record is unknown")]
    recorded_order = record.get("generation_order_sha256")
    if recorded_order != order_sha:
        return None, [
            (f"{opath.name} records query order {recorded_order!r}, not this "
             f"run's {order_sha!r}; batched greedy decoding is not bit-stable "
             "across batch compositions, so this state would not be comparable "
             "with the others")]
    for key, want in (("state", state),
                      ("experiment_id", EXPERIMENT_ID),
                      ("num_queries", len(queries)),
                      ("generation_config", generation_config)):
        if record.get(key) != want:
            return None, [
                (f"{opath.name} records {key}={record.get(key)!r} but this run "
                 f"expects {want!r}, so the record beside this parquet does "
                 "not describe the generation being verified")]
    preds = load_predictions_parquet(ppath)
    coverage = validate_prediction_coverage(
        preds, [q.query_id for q in queries], EXPERIMENT_ID, state)
    if coverage:
        return None, [
            f"{ppath.name} is provenance-valid but row-invalid: "
            + "; ".join(coverage)]
    return preds, []


def generate_state(state: str, adapter_dir: Path, repo_root: Path,
                   freeze: dict[str, Any],
                   queries: list[QueryRecord], by_assoc: dict,
                   model_id: str, model_source: str, device: str,
                   generation_config: dict[str, Any],
                   order_sha: str) -> bool:
    """Generate (or verifiedly reuse) one state's predictions.

    Returns whether an existing file was reused.  Computes no aggregate: the
    rows are scored per query because that is what turns a decoded string into
    a record, and then written straight to disk.

    ``model_id`` and ``model_source`` are different things and both are needed.
    ``model_id`` is the logical name the freeze records and the sidecar binds;
    ``model_source`` is the revision-pinned snapshot directory the weights are
    actually loaded from.  Collapsing them would either record a filesystem
    path as the base model or load through a mutable cache ref.
    """
    preds, reasons = verify_state(state, repo_root, freeze, queries, by_assoc,
                                  model_id, generation_config)
    if preds is not None:
        log.info("[%s] REUSING provenance-verified predictions (%d rows)",
                 state, len(preds))
        return True
    for r in reasons:
        log.warning("[%s] not reusable — %s", state, r)
    ppath = predictions_path(repo_root, state)
    ppath.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_fingerprint(state, adapter_dir, repo_root, model_id,
                                    generation_config, len(queries))
    log.info("[%s] generating all %d confirmation queries under batch_size=%s "
             "image_batch_size=%s max_new_tokens=%s greedy=%s",
             state, len(queries), generation_config["batch_size"],
             generation_config["image_batch_size"],
             generation_config["max_new_tokens"],
             not generation_config["do_sample"])
    generator = ReferenceStateGenerator(
        model_source, device, adapter_dir=adapter_dir,
        max_image_pixels=generation_config["max_image_pixels"])
    #: The SAME ordered list for every state, which is what makes the paired
    #: difference a comparison of models rather than of batch layouts.
    raws = generator.generate_for_queries(
        queries, by_assoc, repo_root,
        batch_size=generation_config["batch_size"],
        image_batch_size=generation_config["image_batch_size"],
        max_new_tokens=generation_config["max_new_tokens"])
    generator.unload()
    if len(raws) != len(queries):
        raise SystemExit(
            f"[{state}] generation returned {len(raws)} outputs for "
            f"{len(queries)} queries; a short pass would leave holes the "
            "coverage check would later report as missing rows, so it stops "
            "here instead")
    preds = [score_query(q, by_assoc[q.association_id], raw,
                         experiment_id=EXPERIMENT_ID, checkpoint_id=state)
             for q, raw in zip(queries, raws)]
    coverage = validate_prediction_coverage(
        preds, [q.query_id for q in queries], EXPERIMENT_ID, state)
    if coverage:
        raise SystemExit(f"[{state}] freshly generated predictions are "
                         f"row-invalid:\n  " + "\n  ".join(coverage))
    import pandas as pd
    pd.DataFrame([json.loads(p.model_dump_json()) for p in preds]
                 ).to_parquet(ppath, index=False)
    write_sidecar(ppath, expected)
    #: Recorded beside the parquet so a later assembly can show all three
    #: states were generated over the same ordering, rather than asserting it.
    generation_order_path(repo_root, state).write_text(
        json.dumps({
            "state": state,
            "experiment_id": EXPERIMENT_ID,
            "num_queries": len(queries),
            "generation_order_sha256": order_sha,
            "generation_config": generation_config,
            "adapter_contract_sha256": (
                adapter_contract(adapter_dir) or {}).get("sha256"),
            "queries_parquet_sha256": sha256_file(
                repo_root / CONFIRM_DATASET_DIR / "queries.parquet"),
        }, indent=1, sort_keys=True))
    log.info("[%s] wrote %s (%d rows) and its sidecar", state, ppath.name,
             len(preds))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    #: Only the device and which state to generate are tunable.  Batch size,
    #: token budget, bootstrap count and CI level are NOT offered, because
    #: every one of them is frozen and a flag that can move a frozen value is
    #: a way to drift from the protocol without editing it.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state", default=None,
                        help="Generate only this one of B3, B0, MG, verify THAT "
                             "state, and exit on its own merits. Used to shard "
                             "the three passes across GPUs: a lane that "
                             "finished its state succeeded, and global "
                             "completeness is --verify-only's job. An "
                             "unrecognised state is an error rather than a "
                             "silent no-op that would leave the confirmation "
                             "incomplete and surface only at assembly.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify all three states' sidecars and report "
                             "GLOBAL completeness. No GPU, and no metric: this "
                             "is the gate analyze_confirmation_split.py "
                             "re-checks before it will assemble a report.")
    parser.add_argument("--list-outstanding", action="store_true",
                        help="Print one line per state that does not currently "
                             "VERIFY, then exit 0. This is how the lane decides "
                             "what to run: a state is outstanding because its "
                             "sidecar or generation-order record fails, not "
                             "because its filename is absent, so a stale "
                             "partial pass is regenerated instead of being "
                             "skipped and then failing the gate with no way "
                             "forward.")
    args = parser.parse_args()

    if args.list_outstanding:
        #: stdout is the DATA channel in this mode: the lane reads state names
        #: from it and iterates over them.  ``setup_logger`` attaches a
        #: ``StreamHandler(sys.stdout)``, so the logger is silenced here and the
        #: summary goes to stderr — otherwise the lane would have to filter log
        #: lines out of a list it is about to feed to ``--state``.
        log.handlers.clear()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    freeze = load_frozen_protocol(repo_root)
    states = scored_states(freeze)
    generation_config = frozen_generation_config(freeze)
    revision = frozen_base_model_revision(freeze)
    model_id = ((freeze.get("checkpoints") or {}).get(states[0]) or {}).get(
        "recipe", {}).get("model_id")
    if not model_id:
        raise SystemExit(
            "REFUSED — the freeze does not record a model_id for "
            f"{states[0]}, so there is no base model to load")

    queries = confirmation_queries(repo_root, freeze)
    by_assoc = {a.association_id: a for a in load_associations_parquet(
        repo_root / CONFIRM_DATASET_DIR / "associations.parquet")}
    order_sha = generation_order_sha256(queries)

    # ---- preflight: refused before ANY state is generated or reused --------
    #: Run in every mode, including --list-outstanding and --verify-only.  A
    #: reused parquet is evidence exactly as much as a generated one, so the
    #: photographs and the analysis code have to be the frozen ones for reuse
    #: too; checking only before generation would let a stale-but-verifying
    #: file through on a dataset whose images had been swapped underneath it.
    problems = verify_frozen_code(repo_root, freeze)
    problems += verify_confirmation_images(repo_root, queries, by_assoc)
    live_revision = base_model_revision(model_id)
    if live_revision is not None and live_revision != revision:
        problems.append(
            f"the local HF cache ref for {model_id} points at {live_revision} "
            f"but the freeze pins base_model_revision {revision}")
    if problems:
        raise SystemExit(
            f"REFUSED — {len(problems)} preflight check(s) failed, so nothing "
            "will be generated, reused or verified:\n  "
            + "\n  ".join(problems))

    log.info("confirmation dataset %s: %d queries, %d associations",
             dataset_version(repo_root / CONFIRM_DATASET_DIR), len(queries),
             len(by_assoc))
    log.info("frozen generation config: %s", generation_config)
    log.info("base model %s pinned at revision %s", model_id, revision)
    log.info("generation order sha256: %s", order_sha)
    log.info("scored states: %s (and no others)", list(states))
    log.info("preflight: %d analysis-script hash(es) plus paired_ci verified, "
             "%d pinned photograph(s) re-hashed, %d of %d queries are "
             "image-route and every one resolves to a frozen byte",
             len((freeze.get("code") or {}).get(
                 "analysis_scripts_sha256") or {}),
             len(referenced_image_paths(repo_root / CONFIRM_DATASET_DIR)),
             sum(1 for q in queries if q.image_ids), len(queries))

    def _verify(state: str):
        return verify_state(state, repo_root, freeze, queries, by_assoc,
                            model_id, generation_config)

    if args.list_outstanding:
        outstanding = [s for s in states if _verify(s)[0] is None]
        for state in outstanding:
            print(state)
        print(f"{len(outstanding)} of {len(states)} states still need "
              f"generation: {outstanding or 'none'}", file=sys.stderr)
        #: Exit 0 even when nothing is outstanding: an empty list is a complete
        #: answer, not a failure, and a lane that treated it as one would never
        #: reach the gate on a resumed run.
        return

    if args.state is not None and args.state not in states:
        raise SystemExit(
            f"--state names {args.state!r}, which is not one of the frozen "
            f"scored states {list(states)}. Generating any other state would "
            "put an unreported comparison on disk.")

    if not args.verify_only:
        todo = (args.state,) if args.state else states
        #: Resolved once, before any state is generated: the pinned snapshot is
        #: the same directory for all three, and failing to resolve it after B3
        #: had already run would leave a partial pass.
        model_source = pinned_model_source(model_id, revision)
        log.info("base model snapshot pinned at %s: %s", revision[:12],
                 model_source)
        for state in todo:
            adapter_dir = adapter_for(state, freeze, repo_root)
            reused = generate_state(state, adapter_dir, repo_root, freeze,
                                    queries, by_assoc, model_id, model_source,
                                    args.device, generation_config, order_sha)
            log.info("[%s] %s", state,
                     "reused a verified parquet" if reused
                     else "generated and sealed")

        if args.state is not None:
            #: ONE state's own merits.  This is the fix for a sharded lane: the
            #: previous version fell through to the global gate, so the first
            #: lane to finish exited nonzero because the other two had not
            #: finished yet — and a chain that records gen_rc=1 stops before its
            #: own gate, however healthy that gate would have been.
            preds, reasons = _verify(args.state)
            if preds is None:
                raise SystemExit(
                    f"REFUSED — {args.state} was generated but does not "
                    "verify:\n  " + "\n  ".join(reasons))
            print(f"\n{args.state}: {len(preds)} rows generated and "
                  f"provenance-verified over query order {order_sha[:16]}")
            print("  this exit says nothing about the other states; run "
                  "--verify-only for global completeness")
            return

    #: Global completeness, reported for ALL three states rather than stopping
    #: at the first problem, and reached only by --verify-only or by a run that
    #: generated all three.  No metric is computed on the way: the rows are
    #: verified and counted, never aggregated.  The generation-order record is
    #: not re-checked here because verify_state owns it now — checking it in two
    #: places is how the reuse path came to skip it in the first place.
    incomplete: dict[str, list[str]] = {}
    complete: list[str] = []
    for state in states:
        preds, reasons = _verify(state)
        if preds is None:
            incomplete[state] = reasons
        else:
            complete.append(state)
    print(f"\nconfirmation predictions — {len(complete)}/{len(states)} states "
          f"complete and provenance-verified")
    print(f"  dataset      {CONFIRM_DATASET_DIR} "
          f"({dataset_version(repo_root / CONFIRM_DATASET_DIR)})")
    print(f"  experiment   {EXPERIMENT_ID}")
    print(f"  base model   {model_id} @ {revision}")
    print(f"  queries      {len(queries)} per state, order sha256 "
          f"{order_sha[:16]}")
    print(f"  states       {list(states)}")
    for state in states:
        mark = "ok" if state in complete else "INCOMPLETE"
        print(f"    {state:3} {mark}")
        for r in incomplete.get(state, []):
            print(f"        - {r}")
    if incomplete:
        #: This is the refusal that keeps the score-exactly-once property: no
        #: report is assembled, and nothing here prints a rate that could be
        #: read as a partial result.
        raise SystemExit(
            f"REFUSED — {len(incomplete)} of {len(states)} scored states are "
            f"incomplete: {sorted(incomplete)}. No analysis will be assembled "
            "until every state has finished, because a report over a subset "
            "looks like the real thing while covering fewer comparisons than "
            "it claims.")
    print("  all three states verified — run "
          "python scripts/analyze_confirmation_split.py to assemble the "
          "analysis")


if __name__ == "__main__":
    main()
