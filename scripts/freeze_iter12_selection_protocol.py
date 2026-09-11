"""Freeze the Iteration 12 Stage-1 selection protocol BEFORE any training.

    python scripts/freeze_iter12_selection_protocol.py
    python scripts/freeze_iter12_selection_protocol.py --check-only

Iteration 12 is exploratory, so it carries none of the confirmation
machinery: no sealed split, no Holm correction, no score-exactly-once gate.
What it does need is a selection rule that is written down before the
candidates exist, because "pick the one closest to MG that does not lose
retention" leaves three decisions open, and each of them can change which
candidate wins:

* **the direction of D_G** -- it is a distance and is MINIMISED, and because
  every component enters as an absolute deviation it does not reward low
  leakage.  A candidate that drives FILR below MG's level is penalised exactly
  as much as one that leaves it above.
* **the numerical tolerance** -- distances are compared exactly as
  ``distance_to_reference`` rounds them (6 decimals) with no extra epsilon, so
  two candidates are tied iff their rounded distances are equal.  The
  retention floor uses 1e-9, which is six orders of magnitude below the
  smallest difference the probe measurement can express (~2.0e-3), so it can
  absorb float representation error and cannot decide a real comparison.
* **the tie-break** -- ``(distance_to_mg, candidate_id)`` ascending.  The id
  encodes hyperparameters and no outcome, so the order cannot be gamed by
  looking at results, and it does not depend on grid order, filesystem order,
  or which adapters happen to exist on disk.

WHAT MAKES THIS MORE THAN A DOCUMENT
------------------------------------
A protocol nobody is bound by is a comment.  Two things bind this one:

1. ``--check-only`` recomputes every hash and refuses on a mismatch, and
   ``scripts/select_iter12_retention_checkpoints.py`` runs that check before it
   generates anything.  Changing the criterion after training therefore breaks
   the run rather than quietly redefining it.
2. Writing the freeze is REFUSED once a candidate checkpoint exists.  The
   confirmation protocol's own re-freeze guard turned out to be a single
   ``--allow-refreeze`` flag with nothing behind it -- a request, not a check,
   as the 11C-5R3 audit records.  Here the check is possible and is made: a
   freeze that postdates the training it governs is worthless, so the script
   will not produce one.

Nothing here reads the sealed confirmation split, and the freeze records that
as an enforced exclusion rather than a promise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation.prediction_provenance import dataset_fingerprint
from granunlearn.evaluation.reference_eval import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.training.candidate_grid import (
    ITER12_LAM,
    ITER12_PILOT_EQUIVALENT_WEIGHT,
    ITER12_REFERENCE_ROW,
    ITER12_REFERENCE_WEIGHT,
    ITER12_WEIGHTS,
    dataset_dir_for_tag,
    groups_subdir_for_tag,
    iter12_grid,
    validate_grid,
)

log = setup_logger("freeze_iter12_protocol")

TAG = "iter12"
OUT_REPORT = "data/reports/mllmu_iter12_selection_protocol_freeze.json"
PROBE_REPORT = "data/reports/mllmu_iter12_retention_probe.json"
PILOT_SELECTION_REPORT = "data/reports/mllmu_pilot100_unlearning_selection.json"
SELECTION_REPORT = "data/reports/mllmu_iter12_retention_selection.json"
CANDIDATE_ROOT = "data/checkpoints/mllmu_iter12_unlearn"

#: The code that IMPLEMENTS the protocol.  If one of these changes, the
#: criterion that selects the successor has changed, and --check-only must say
#: so before another candidate is generated.
PROTOCOL_PATHS = (
    "scripts/freeze_iter12_selection_protocol.py",
    "scripts/build_iter12_retention_probe.py",
    "scripts/select_iter12_retention_checkpoints.py",
    "scripts/train_unlearning_baselines.py",
    "src/granunlearn/evaluation/retention_selection.py",
    "src/granunlearn/training/candidate_grid.py",
    "src/granunlearn/training/unlearning_datasets.py",
    "src/granunlearn/training/unlearning_trainer.py",
)

#: Code the protocol DEPENDS ON but must not modify.  ``hierarchy_metrics``
#: and ``select_unlearning_checkpoints`` are two of the eighteen paths the
#: confirmation freeze sealed, so their bytes are bound by filed sidecars;
#: ``selection`` is not sealed but the committed pilot-100 selection report was
#: produced by its current behaviour, and changing it would make that report
#: unreproducible.  All three are imported, never edited.
DEPENDS_ON_UNCHANGED = (
    "src/granunlearn/evaluation/selection.py",
    "src/granunlearn/evaluation/hierarchy_metrics.py",
    "src/granunlearn/evaluation/paired_ci.py",
    "src/granunlearn/evaluation/prediction_provenance.py",
    "src/granunlearn/evaluation/reference_eval.py",
    "src/granunlearn/evaluation/scoring.py",
    "scripts/select_unlearning_checkpoints.py",
)

#: Frozen with the protocol because they are what a reviewer must compare
#: against to know the partition did not move.
DATA_PATHS = (
    PROBE_REPORT,
    f"{dataset_dir_for_tag(TAG)}/{groups_subdir_for_tag(TAG)}/fine_target.jsonl",
    f"{dataset_dir_for_tag(TAG)}/{groups_subdir_for_tag(TAG)}/target_level.jsonl",
    f"{dataset_dir_for_tag(TAG)}/{groups_subdir_for_tag(TAG)}/retain.jsonl",
)

#: Excluded from the --check-only comparison.  The first three are properties
#: of the moment of freezing rather than of the protocol.  ``amendments`` is
#: excluded because it is APPENDED to the committed document by --refreeze and
#: is not produced by ``build_freeze``: comparing it would report a mismatch
#: forever after the first amendment, which would train everyone to ignore the
#: check.  Its integrity is carried by each entry's ``supersedes_sha256``
#: instead -- the hash of the protocol that amendment replaced.
VOLATILE_FIELDS = ("frozen_at_utc", "git_commit", "git_dirty", "amendments")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.exists() else None


def _hash_all(repo_root: Path, paths: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    missing: list[str] = []
    for rel in paths:
        h = _sha256(repo_root / rel)
        if h is None:
            missing.append(rel)
        else:
            out[rel] = h
    if missing:
        raise SystemExit(
            f"cannot freeze: {len(missing)} protocol path(s) do not exist: "
            f"{missing}")
    return out


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=repo_root,
                             capture_output=True, text=True, timeout=60,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def trained_candidates(repo_root: Path) -> list[str]:
    """Grid candidates that already have an adapter on disk."""
    root = repo_root / CANDIDATE_ROOT
    return sorted(
        spec.candidate_id for spec in iter12_grid()
        if (root / spec.candidate_id / "adapters").exists())


def build_freeze(repo_root: Path) -> dict[str, Any]:
    """Assemble the protocol document from the code and data, not from prose."""
    probe = json.loads((repo_root / PROBE_REPORT).read_text())
    grid = iter12_grid()
    errors = validate_grid(grid)
    if errors:
        raise SystemExit(f"the iter12 grid is invalid: {errors}")
    tolerance = probe["numerical_tolerance"]
    min_gap = tolerance[
        "smallest_resolvable_gap_over_the_four_frozen_numbers"]

    return {
        "iteration": "12",
        "stage": 1,
        "protocol": "retention-aware B3 successor (method B4), exploratory",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "status": {
            "exploratory": True,
            "preregistered_before_training": True,
            "carries_no_confirmatory_claim": True,
            "this_document_selects_a_checkpoint_it_does_not_test_a_hypothesis":
                True,
        },
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": _git(repo_root, "status", "--porcelain") or "",

        # ── 1. the criterion, its direction, tolerance and tie-break ──
        "criterion": {
            "name": "D_G",
            "formula": ("D_G(M_U) = sum_j w_j * |v_j(M_U) - v_j(M_G)| / "
                        "sum_j w_j, over the components present on both "
                        "sides"),
            "components": list(rs.SUMMARY_COMPONENTS),
            "weights": {c: 1.0 for c in rs.SUMMARY_COMPONENTS},
            "changed_from_pilot100": False,
            "computed_by": ("granunlearn.evaluation.selection."
                            "distance_to_reference"),
            "reimplemented_in_this_iteration": False,
            "direction": rs.D_G_DIRECTION,
            "direction_spelled_out": (
                "D_G is a DISTANCE and the winner is its MINIMUM. Larger is "
                "worse. It is not a score to maximise."),
            "direction_is_not": (
                "It does NOT reward low leakage. Every component enters as an "
                "absolute deviation from MG, so a candidate that drives FILR "
                "below MG's level is penalised exactly as much as one that "
                "leaves FILR above it. The criterion asks whether a state "
                "looks like granularity-controlled unlearning, not whether it "
                "unlearns hard. The floor below is the only monotone "
                "requirement in this protocol: more retained knowledge is "
                "never disqualifying."),
            "scope": {
                "splits": ["train", "val"],
                "dataset": dataset_dir_for_tag(TAG),
                "dataset_version": probe["reuses_dataset"]["version"],
                "frozen_test_split_used_for_selection": False,
            },
            "numerical_tolerance": {
                "distance_decimals": rs.DISTANCE_DECIMALS,
                "additional_epsilon_on_distances":
                    rs.ADDITIONAL_EPSILON_ON_DISTANCES,
                "source_of_the_rounding": (
                    "distance_to_reference already rounds to 6 decimals; that "
                    "rounding IS the tolerance and nothing is added to it"),
                "tied_iff": "the two rounded distances are exactly equal",
                "retention_rate_decimals": rs.RATE_DECIMALS,
                "why_rates_and_distances_differ": (
                    "each number is rounded where it is computed -- rates at "
                    "the 4 decimals the sealed _rate uses, distances at the 6 "
                    "decimals distance_to_reference uses -- and never "
                    "re-rounded afterwards"),
                "rounding_cannot_merge_two_retention_values": (
                    min_gap > 10 ** -rs.RATE_DECIMALS),
            },
            "tie_break": {
                "rule": "(distance_to_mg, candidate_id) ascending",
                "id_ordering": "lexicographic",
                "carries_no_outcome_information": True,
                "does_not_depend_on": ["grid order", "filesystem order",
                                       "which adapters exist on disk"],
                "replaces": (
                    "the pilot-100 selector's strict < over dict insertion "
                    "order, which resolves ties by whichever candidate was "
                    "registered first and writes that down nowhere"),
                "a_tie_is_reported_as": (
                    "criterion_did_not_discriminate, with the tied ids listed"),
            },
        },

        # ── 2. the retention floor ──
        "floor": {
            "statement": (
                "A candidate is ELIGIBLE only if retain_same_entity and "
                "retain_other_entity, measured on PROBE-half train+val "
                "queries, are each at or above B0's value on those same "
                "queries, under BOTH entity-macro and row-micro."),
            "reference_level": "B0, the no-op MF copy",
            "direction": rs.FLOOR_DIRECTION,
            "measured_on": {
                "entities": "the probe half only -- never rehearsed by any "
                            "candidate",
                "num_probe_entities": probe["halves"]["probe"]["num_entities"],
                "num_probe_associations":
                    probe["halves"]["probe"]["num_associations"],
                "per_family": probe["probe_measurement_basis"]["per_family"],
            },
            "estimands_required": list(rs.RETENTION_ESTIMANDS),
            "why_both_estimands": (
                "row-micro and entity-macro disagree in SIGN for "
                "retain_other on the pilot-100 evidence (the incumbent is "
                "-0.1056 against B0 row-micro and +0.0093 entity-macro), so "
                "freezing either alone would be choosing the estimand that "
                "yields the preferred verdict. Requiring both is the "
                "conservative pre-commitment and is made before any "
                "Iteration-12 candidate exists."),
            "comparison": f"candidate >= b0 - {rs.FLOOR_EPSILON:g}",
            "epsilon": rs.FLOOR_EPSILON,
            "epsilon_is_a_tolerance_not_a_margin": True,
            "smallest_resolvable_gap": min_gap,
            "epsilon_is_below_every_resolvable_gap": tolerance[
                "epsilon_is_below_every_resolvable_gap"],
            "orders_of_magnitude_below": tolerance["orders_of_magnitude_below"],
            "no_retention_margin_is_required": True,
            "rejected_alternative": (
                "A margin -- e.g. requiring the winner to clear B0's probe "
                "retention by +0.02 -- was proposed and REJECTED. Nothing in "
                "this repository supports a threshold of that size: it is not "
                "a resolvable gap, not a confidence-interval width, and not a "
                "pre-registered effect size. Requiring improvement beyond "
                "'not below B0' would invent a hypothesis and file it as a "
                "constraint."),
            "consequence_of_failing": (
                "ineligible, however small its D_G. The floor is a constraint, "
                "not a term in the criterion, so it is never traded off "
                "against distance."),
            "missing_value_is_a_failure": True,
            "why_the_probe_is_needed_at_all": (
                "The pilot-100 retain group is exactly the 387 retained "
                "associations, and those back 100% of the retention queries "
                "in all three splits. A replay candidate was therefore scored "
                "in-sample on retention while B0 was scored out-of-sample, and "
                "a floor comparing them cannot discriminate: applied to the "
                "existing grid it passes exactly the two candidates carrying a "
                "retain group and fails all nine that do not."),
        },

        # ── 3. the grid ──
        "grid": {
            "tag": TAG,
            "dataset_dir": dataset_dir_for_tag(TAG),
            "groups_subdir": groups_subdir_for_tag(TAG),
            "num_candidates": len(grid),
            "candidates": [c.describe() for c in grid],
            "method": "B4",
            "held_fixed": {
                "fine_suppression_weight": ITER12_LAM,
                "target_level_weight": 1.0,
                "init_adapter": "the pilot-100 canonical MF adapter",
                "recipe": ("ReferenceRecipe, overridden only in "
                           "learning_rate and num_epochs"),
            },
            "swept": {
                "retain_replay_weight": list(ITER12_WEIGHTS),
                "num_epochs": sorted({c.overrides.get("num_epochs")
                                      for c in grid if not c.noop}),
                "learning_rate": sorted({c.overrides.get("learning_rate")
                                         for c in grid if not c.noop}),
            },
            "reference_row": ITER12_REFERENCE_ROW,
            "reference_row_is": (
                "the incumbent recipe -- lambda 0.5, lr 2e-5, 5 epochs, replay "
                "weight 1.0 -- retrained on the fit half. It is the row the "
                "Stage-2 gate compares the winner against, and it is in the "
                "grid rather than beside it so the weight sweep includes it."),
            "pilot_equivalent_weight": ITER12_PILOT_EQUIVALENT_WEIGHT,
            "pilot_equivalent_weight_derivation": probe["replay_strength"],
            "reference_weight": ITER12_REFERENCE_WEIGHT,
        },

        # ── 4. the Stage-2 gate ──
        "stage_2": {
            "mechanism": "MF-preservation regularizer",
            "opens_iff": [
                "no Stage-1 candidate satisfies the retention floor, or",
                ("the best eligible candidate's D_G is not STRICTLY smaller "
                "than the reference row's D_G"),
            ],
            "strictly_means": ("a smaller 6-decimal distance; an exact tie is "
                               "no improvement"),
            "margin_required": 0.0,
            "no_margin_is_used": True,
            "decided_by": ("scripts/select_iter12_retention_checkpoints.py "
                           "reporting stage_2_gate, computed from the frozen "
                           "rule -- not by whoever reads the table"),
        },

        # ── 5. the evidence base and what is never read ──
        "evidence_base": {
            "used": [
                f"{dataset_dir_for_tag(TAG)}/queries.parquet, splits train+val",
                f"{dataset_dir_for_tag(TAG)}/associations.parquet",
                f"{groups_subdir_for_tag(TAG)} group files",
                PROBE_REPORT,
            ],
            "never_read": list(rs.FORBIDDEN_EVIDENCE),
            "never_read_enforced_by": (
                "scripts/select_iter12_retention_checkpoints.py refuses any "
                "argument that resolves inside those paths, and writes "
                "predictions only to predictions_iter12/"),
            "why": (
                "The 11C confirmation was scored once and is sealed. Tuning a "
                "successor on its predictions would make a confirmatory "
                "result retrospective, which is the failure the seal exists to "
                "prevent. Its composition metadata was read to DIAGNOSE the "
                "retention problem -- that diagnosis is recorded above -- but "
                "no confirmation outcome enters this protocol or any number it "
                "produces."),
            "pilot100_selection_report": {
                "path": PILOT_SELECTION_REPORT,
                "read_for": ("the incumbent's identity and the pilot-100 "
                             "candidate table quoted in this protocol's "
                             "rationale"),
                "never_transcribed_into": (
                    "the MG reference vector, which is regenerated for this "
                    "study under its own generation contract"),
            },
        },
        "reference_vector": {
            "state": "MG",
            "obtained_by": (
                "regenerating MG's predictions in predictions_iter12/ under "
                "the identical generation contract, in the same batch layout "
                "as every candidate"),
            "not_transcribed_from": PILOT_SELECTION_REPORT,
            "why_regenerated": (
                "Batched greedy decoding is not bit-stable across batch "
                "compositions. D_G ranks candidates by distance to MG, so an "
                "MG vector produced in a different layout would put decoding "
                "noise inside the criterion itself."),
        },
        "generation_contract": {
            "batch_size": 8,
            "image_batch_size": 1,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "max_length": DEFAULT_MAX_LENGTH,
            "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
            "do_sample": False,
            "model_id": "Qwen/Qwen3.5-9B",
            "predictions_dir": (
                f"{dataset_dir_for_tag(TAG)}/predictions_iter12"),
            "experiment_id": "mllmu_iter12_stage1",
            "why_not_the_pilot_predictions_dir": (
                "Both grids contain a candidate called B0. In one directory "
                "the two studies would share the filename "
                "predictions_tv_B0.parquet, and a differing experiment_id "
                "would make the sidecar refuse reuse and REGENERATE over "
                "committed pilot-100 evidence. A separate directory makes "
                "that impossible rather than unlikely."),
        },
        "seeds": {
            "training_seed": 42,
            "partition_seed": probe["partition_rule"]["seed"],
            "partition_ordering": probe["partition_rule"]["ordering"],
            "no_selection_randomness": (
                "selection is a deterministic ordering of computed numbers; "
                "it draws nothing"),
        },
        "dataset_fingerprint": dataset_fingerprint(
            repo_root / dataset_dir_for_tag(TAG), repo_root),
        "hashes": {
            "protocol_paths": _hash_all(repo_root, PROTOCOL_PATHS),
            "depends_on_unchanged": _hash_all(repo_root, DEPENDS_ON_UNCHANGED),
            "data_paths": _hash_all(repo_root, DATA_PATHS),
        },
        "limitations": [
            ("The probe half carries 1 of the 6 taxonomic retained "
            "associations, so probe retention speaks almost entirely to "
            "semantic and numeric knowledge."),
            ("retain_other_entity reaches 23 probe donor entities: enough for "
            "a point-estimate floor, too few for an interval. This protocol "
            "selects; it does not do inference."),
            ("Halving the replay group halves the rehearsal available to the "
            "mechanism under test, and cuts its share of each epoch's "
            "gradient stream from 0.6825 to 0.5041. A null result may reflect "
            "the smaller group rather than the mechanism, which is why the "
            "weight sweep spans the pilot's effective influence."),
            ("Selection is on train+val, and train queries are served "
            "photographs the checkpoints were trained on. That is inherited "
            "from the pilot-100 protocol and unchanged here."),
            ("Nothing in this protocol controls a familywise error rate. A "
            "winner chosen from 8 candidates on 4 retention numbers is a "
            "selected candidate, not a demonstrated effect."),
        ],
    }


def _flat(doc: Any, prefix: tuple = ()) -> dict:
    out: dict = {}
    if isinstance(doc, dict):
        for key, value in doc.items():
            out.update(_flat(value, prefix + (key,)))
    elif isinstance(doc, list):
        for i, value in enumerate(doc):
            out.update(_flat(value, prefix + (str(i),)))
    else:
        out[".".join(prefix)] = doc
    return out


def verify_freeze(repo_root: Path, path: Path | None = None) -> list[str]:
    """Recompute the freeze against the repository. Empty list == matches."""
    path = path or repo_root / OUT_REPORT
    if not path.exists():
        return [f"{path} does not exist -- the protocol is not frozen"]
    committed = json.loads(path.read_text())
    fresh = _flat(build_freeze(repo_root))
    old = _flat(committed)
    reasons = []
    for key in sorted(set(old) | set(fresh)):
        if any(key.startswith(f"{v}.") or key == v for v in VOLATILE_FIELDS):
            continue
        if old.get(key) != fresh.get(key):
            reasons.append(
                f"{key}: frozen={old.get(key)!r} now={fresh.get(key)!r}")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Iteration 12 Stage-1 selection protocol")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify the committed freeze against the "
                             "repository without writing")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repo-root", default=None,
                        help="Freeze against this tree instead of the "
                             "working directory (used by the tests, which "
                             "need a tree with no candidates in it)")
    parser.add_argument("--refreeze", action="store_true",
                        help="Replace a committed freeze that no longer "
                             "matches the repository. Permitted ONLY while no "
                             "Iteration-12 candidate has been trained; once "
                             "one exists the refusal is unconditional and "
                             "this flag does nothing.")
    parser.add_argument("--reason", default=None,
                        help="With --refreeze: what changed and why. Recorded "
                             "in the freeze's amendment list beside the hash "
                             "of the protocol it supersedes.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = Path(args.output) if args.output else repo_root / OUT_REPORT

    if args.check_only:
        reasons = verify_freeze(repo_root, out)
        if reasons:
            log.error("FREEZE MISMATCH: %d field(s) differ", len(reasons))
            for r in reasons[:20]:
                log.error("    - %s", r)
            raise SystemExit(1)
        log.info("OK — the Iteration 12 protocol freeze still matches the "
                 "repository (volatile fields excluded: %s)",
                 list(VOLATILE_FIELDS))
        return

    #: The check the confirmation freeze did not have.  A protocol frozen
    #: after the candidates exist cannot constrain their selection, so this
    #: refuses rather than producing a document that only looks preregistered.
    #: It is UNCONDITIONAL: no flag reaches past it, which is the difference
    #: between this and a ``--allow-refreeze`` gate whose flag is the only
    #: thing standing between an operator and a retroactive protocol.
    trained = trained_candidates(repo_root)
    if trained:
        raise SystemExit(
            f"REFUSING to freeze: {len(trained)} Iteration-12 candidate(s) "
            f"already have adapters on disk ({trained[:3]}...). A selection "
            f"protocol written now would postdate the training it is supposed "
            f"to govern. The freeze must precede the candidates.")

    if out.exists():
        reasons = verify_freeze(repo_root, out)
        if not reasons:
            log.info("%s already exists and still matches; nothing to do", out)
            return
        if not args.refreeze:
            raise SystemExit(
                f"REFUSING to overwrite {out}: the committed freeze differs "
                f"from what the repository now produces in {len(reasons)} "
                f"field(s), e.g. {reasons[0]}. Re-freezing after a change to "
                f"the criterion is a new protocol, so it must be deliberate: "
                f"pass --refreeze with --reason saying what changed and why. "
                f"No candidate has been trained yet, which is the only "
                f"reason this is permitted at all.")
        if not args.reason:
            raise SystemExit(
                "--refreeze requires --reason: an amendment that does not say "
                "what it changed cannot be read as outcome-blind afterwards")
        previous = json.loads(out.read_text())
        log.warning("RE-FREEZING: %d field(s) change, e.g. %s",
                    len(reasons), reasons[0])

    freeze = build_freeze(repo_root)
    #: Amendments are appended, never replaced, and each carries the hash of
    #: the protocol it supersedes -- so a reader can see the criterion moved
    #: and what it moved from, instead of inferring it from git archaeology.
    amendments = list(previous.get("amendments", [])) if out.exists() else []
    if out.exists() and args.refreeze:
        amendments.append({
            "utc": freeze["frozen_at_utc"],
            "reason": args.reason,
            "supersedes_sha256": _sha256(out),
            "fields_that_changed": reasons[:50],
            "no_candidate_had_been_trained": True,
        })
    freeze["amendments"] = amendments
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(freeze, f, indent=2, ensure_ascii=False)
    log.info("Frozen %d protocol path(s), %d dependency hash(es) and %d data "
             "hash(es) -> %s", len(freeze["hashes"]["protocol_paths"]),
             len(freeze["hashes"]["depends_on_unchanged"]),
             len(freeze["hashes"]["data_paths"]), out)
    log.info("criterion D_G %s | tolerance: distances at %d dp, floor epsilon "
             "%g vs smallest resolvable gap %.8f | tie-break %s",
             freeze["criterion"]["direction"],
             rs.DISTANCE_DECIMALS, rs.FLOOR_EPSILON,
             freeze["floor"]["smallest_resolvable_gap"],
             freeze["criterion"]["tie_break"]["rule"])
    log.info("grid: %d candidates, reference row %s",
             freeze["grid"]["num_candidates"], freeze["grid"]["reference_row"])


if __name__ == "__main__":
    main()
