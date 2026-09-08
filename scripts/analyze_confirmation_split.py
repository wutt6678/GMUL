"""Iteration 11C stage 5 — assemble the confirmation analysis, once, from three
verified prediction files.

    python scripts/analyze_confirmation_split.py
    python scripts/analyze_confirmation_split.py --check-only

No GPU and no model: this reads the three parquets
``evaluate_confirmation_split.py`` wrote, refuses to assemble anything until
ALL THREE verify, and then reports the frozen claims and nothing else.

What is reported
----------------
* PRIMARY — entity-macro B3 − B0 on TGA and FILR over all 72 target entities,
  each with a one-sided cluster sign-flip p-value (10,000 draws, seed
  20260908) and a paired percentile CI (1,000 entity bootstraps, seed 42),
  then Holm step-down over EXACTLY those two claims at familywise α = 0.05.
* ROW-MICRO — reported separately and never averaged with the macro, because
  entities contribute unequal numbers of probes and subtracting two published
  micro rates does not reproduce the interval's centre.
* SECONDARY — the person stratum (``seen_photo_unseen_wording``, 42 entities)
  and the photograph stratum (``held_out_photo``, 30 entities) as diagnostics
  only.  Neither carries a primary claim and neither gets a Holm entry.
* DESCRIPTIVE — ``retain_same`` and ``retain_other`` on the text route, with no
  margin and no non-inferiority test; and B3 − MG with an explicit statement
  that equivalence is NOT concluded.

What this script refuses to do
------------------------------
Sealed-split invariants 4, 5 and 6 are discharged here rather than promised:

* It never invokes the reference-state gate and never reads its parquets.  The
  exploratory evaluator reads ``predictions_{BASE,MF,MG,MN}.parquet`` to
  quantify a batch-layout noise floor; the confirmation split never enters the
  gate, so there is no such floor to read and none is reported.
* It never invokes checkpoint selection and does not know the selection
  report's path.  Which three states are scored comes from the freeze.
* It refuses to assemble from fewer than three states.  A report over a subset
  is worse than no report: its intervals and its Holm step-down would be
  computed over whichever states happened to finish and would look like the
  real thing.
* It runs no equivalence and no non-inferiority test.  Both were demoted by the
  11C-2a decisions, and the freeze records the 0.05 margin's role as
  "REPORTING YARDSTICK, not a tested margin".

What it re-checks rather than trusts
------------------------------------
The analysis is a separate process from the generation and may run hours later
on a different checkout, so the freeze's pins are compared against the bytes on
disk HERE, through the evaluator's own implementations rather than a restatement
of them:

* every hash in ``code.analysis_scripts_sha256`` and the separately pinned
  ``primary_test.implementation.sha256`` for ``paired_ci.py``.  This is the
  enforcement behind sealed-split invariant 7: the freeze RECORDING those hashes
  is a description of the protocol, and comparing them is what makes editing the
  code that produces a verdict impossible without also re-freezing.
* ``verify_image_manifest`` over the pinned photographs, and per-query image
  resolution — because a prediction generated over a photograph that has since
  been swapped has a sidecar that still verifies while its evidence is not what
  the protocol froze.
* the base-model revision against the freeze's pin.

Invariant 7 — a failed primary claim is reported as failed — is discharged by
there being no code path that adjusts anything: every threshold, seed, draw
count and direction is read from the freeze and cross-checked, so the only way
to change a verdict is to change the protocol, which changes the freeze's own
hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Reused rather than reimplemented: "is this state complete and
#: provenance-verified" has to mean one thing, and the script that generates
#: the parquets is the one that defines it.
import evaluate_confirmation_split as ecs  # noqa: E402

from granunlearn.config import _find_repo_root  # noqa: E402
from granunlearn.evaluation.image_splits import image_stratum  # noqa: E402
from granunlearn.evaluation.paired_ci import (  # noqa: E402
    CLAIM_DIRECTION,
    PAIRED_METRICS,
    _paired_unit_diffs,
    holm_family,
    one_sided_permutation_pvalue,
    paired_rate_diff_ci,
    row_flags,
)
from granunlearn.evaluation.prediction_provenance import (  # noqa: E402
    base_model_revision,
    dataset_version,
    sha256_file,
)
from granunlearn.evaluation.reference_eval import (  # noqa: E402
    load_associations_parquet,
)
from granunlearn.logging_utils import setup_logger  # noqa: E402
from granunlearn.schema import PredictionRecord, QueryRecord  # noqa: E402

log = setup_logger("analyze_confirmation_split")

ANALYSIS_REPORT = "data/reports/mllmu_confirm100_final_analysis.json"

#: The primary family, exactly as the freeze declares it.  Asserted against the
#: freeze rather than taken from here: two claims is what Holm's k = 2 and the
#: 0.025 worst-case threshold were sized for, and a third claim arriving
#: silently would be tested at thresholds sized for two.
EXPECTED_PRIMARY_FAMILY = ("B3_minus_B0:tga", "B3_minus_B0:filr")

#: The two averaging units are reported separately because they are not equal.
PRIMARY_METRICS = ("tga", "filr")
#: Descriptive only: no claim, no Holm entry, no margin.
DESCRIPTIVE_TARGET_METRICS = ("wrong_branch", "over_forgetting")
RETENTION_METRICS = ("retain_same", "retain_other")


def _sha256_id_list_newline(ids: list[str]) -> str:
    """The canonicalization the freeze's entity- and association-set pins use.

    Sorted, newline-joined: the form ``select_confirmation_photographs`` and
    the seal's set hashes use.  Recorded here because deriving the target
    entity set is only worth doing if the derivation can be checked against
    the pinned hash — and a hash in a second canonicalization checks nothing.
    """
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def target_entity_set(repo_root: Path, freeze: dict[str, Any]
                      ) -> tuple[set[str], dict[str, Any]]:
    """The 72 target entities, DERIVED and then checked against the freeze.

    Derived from ``manifest.json``'s ``target_association_ids`` joined to
    ``associations.parquet``, which is the same derivation the power report
    records as its ``read_from``.  Deriving it here rather than reading a list
    out of a report means the analysis cannot restrict the primary to a
    different entity set than the one the protocol froze — and the pinned
    ``target_entity_ids_sha256`` is what makes that checkable rather than
    merely parallel.
    """
    data_dir = repo_root / ecs.CONFIRM_DATASET_DIR
    manifest = json.loads((data_dir / "manifest.json").read_text())
    associations = load_associations_parquet(data_dir / "associations.parquet")
    entity_of = {a.association_id: a.entity_id for a in associations}
    target_assoc = sorted(manifest["target_association_ids"])
    unknown = [a for a in target_assoc if a not in entity_of]
    if unknown:
        raise SystemExit(
            f"REFUSED — the manifest names {len(unknown)} target association(s) "
            f"absent from associations.parquet, e.g. {unknown[:3]}, so the "
            "target entity set cannot be derived")
    entities = sorted({entity_of[a] for a in target_assoc})
    bound = ((freeze.get("sealed_split_invariants") or [{}])[0]
             .get("bound_by") or {})
    want_sha = bound.get("target_entity_ids_sha256")
    got_sha = _sha256_id_list_newline(entities)
    if want_sha and got_sha != want_sha:
        raise SystemExit(
            f"REFUSED — the derived target entity set hashes to {got_sha} but "
            f"the freeze pins {want_sha}. The primary estimand averages over "
            "exactly the frozen entities, so a different set is a different "
            "estimand.")
    want_n = bound.get("target_entity_ids")
    if want_n is not None and len(entities) != want_n:
        raise SystemExit(
            f"REFUSED — the derived target entity set holds {len(entities)} "
            f"entities but the freeze binds {want_n}")
    want_assoc_sha = bound.get("target_association_ids_sha256")
    got_assoc_sha = _sha256_id_list_newline(target_assoc)
    audit = {
        "derived_from": (
            f"{ecs.CONFIRM_DATASET_DIR}/manifest.json target_association_ids "
            "joined to associations.parquet entity_id"),
        "target_association_ids": len(target_assoc),
        "target_association_ids_sha256": got_assoc_sha,
        "target_association_ids_match_the_freeze":
            got_assoc_sha == want_assoc_sha if want_assoc_sha else None,
        "target_entity_ids": len(entities),
        "target_entity_ids_sha256": got_sha,
        "target_entity_ids_match_the_freeze": got_sha == want_sha,
        "persons": sum(1 for e in entities if e.startswith("mllmu")),
        "species": sum(1 for e in entities if not e.startswith("mllmu")),
    }
    if want_assoc_sha and got_assoc_sha != want_assoc_sha:
        raise SystemExit(
            f"REFUSED — the derived target association set hashes to "
            f"{got_assoc_sha} but the freeze pins {want_assoc_sha}")
    return set(entities), audit


def _restrict(flags: dict[str, tuple[int, str]],
              entities: set[str] | None = None,
              qids: set[str] | None = None) -> dict[str, tuple[int, str]]:
    """One metric's row flags, narrowed to an entity set and/or a query set.

    Returns the same ``{query_id: (flag, entity_id)}`` shape the paired
    primitives consume, so restriction composes: the primary restricts by
    entity, a stratum restricts by query, and a stratum-of-the-primary
    restricts by both.
    """
    out = {}
    for qid, (flag, entity) in flags.items():
        if entities is not None and entity not in entities:
            continue
        if qids is not None and qid not in qids:
            continue
        out[qid] = (flag, entity)
    return out


def _comparison(fa: dict[str, tuple[int, str]],
                fb: dict[str, tuple[int, str]],
                n_bootstrap: int, ci_level: float, bootstrap_seed: int,
                n_permutations: int, permutation_seed: int,
                direction: str | None = None) -> dict[str, Any] | None:
    """Both averaging units for one metric, and a p-value only if directional.

    ``fa`` and ``fb`` are already-restricted ``{query_id: (flag, entity_id)}``
    maps for the two states, so the caller decides the entity set and the probe
    set and this function decides nothing about them.

    ``direction=None`` is how a DESCRIPTIVE comparison is distinguished from a
    tested one in the code and not merely in the prose: with no direction there
    is no p-value, so no descriptive block can acquire a test by accident.
    """
    paired = _paired_unit_diffs(fa, fb)
    if not paired["keys"]:
        return None
    ci = paired_rate_diff_ci(fa, fb, n_bootstrap=n_bootstrap,
                             ci_level=ci_level, seed=bootstrap_seed)
    out: dict[str, Any] = {
        "entity_macro": {
            "diff": ci["diff"] if ci else None,
            "ci": list(ci["ci"]) if ci else None,
            "ci_level": ci_level,
            "num_entity_clusters": len(paired["keys"]),
            "num_paired_rows": paired["num_rows"],
            "rate_a": round(paired["entity_a"], 6),
            "rate_b": round(paired["entity_b"], 6),
        },
        #: Reported separately, never blended with the macro: entities
        #: contribute unequal numbers of probes, so the micro difference is a
        #: different quantity and subtracting two published micro rates does
        #: not reproduce the interval's centre.
        "row_micro": {
            "rate_a": round(paired["row_a"], 6),
            "rate_b": round(paired["row_b"], 6),
            "diff": round(paired["row_a"] - paired["row_b"], 6),
            "num_paired_rows": paired["num_rows"],
            "why_it_differs_from_the_macro": (
                "micro averages over paired rows and macro over entities; the "
                "two agree only when every entity contributes the same number "
                "of probes, and here persons contribute 12 target probes and "
                "species 12, but the retention families contribute 3 each to "
                "unequal numbers of entities"),
        },
        "num_entity_clusters": len(paired["keys"]),
    }
    if direction is not None:
        test = one_sided_permutation_pvalue(
            paired["diffs"], direction=direction,
            n_permutations=n_permutations, seed=permutation_seed)
        out["test"] = test
        out["claim_kind"] = "superiority"
        out["direction"] = direction
    else:
        out["test"] = None
        out["claim_kind"] = "descriptive"
        out["direction"] = None
        out["why_there_is_no_test"] = (
            "no hypothesis was preregistered for this comparison, so no "
            "p-value is computed; an interval published without a test is a "
            "description, and calling it a non-significant result would be "
            "reading absence of evidence as evidence of absence")
    return out


def _qids_in_stratum(queries: list[QueryRecord], stratum: str) -> set[str]:
    return {q.query_id for q in queries if image_stratum(q) == stratum}


def analyze(repo_root: Path) -> dict[str, Any]:
    """The whole confirmation analysis, or a refusal."""
    freeze = ecs.load_frozen_protocol(repo_root)
    states = ecs.scored_states(freeze)
    generation_config = ecs.frozen_generation_config(freeze)
    data_dir = repo_root / ecs.CONFIRM_DATASET_DIR
    queries = ecs.confirmation_queries(repo_root, freeze)
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_assoc = {a.association_id: a for a in associations}
    model_id = ((freeze.get("checkpoints") or {}).get(states[0]) or {}).get(
        "recipe", {}).get("model_id")
    order_sha = ecs.generation_order_sha256(queries)

    refusals: list[str] = []

    # ---- preflight: the frozen code and the frozen photographs ----
    #: Checked here and not only before generation, because the analysis is a
    #: separate process that may run hours or days later on a different
    #: checkout.  Reusing the evaluator's implementation rather than restating
    #: it keeps "is this the frozen protocol" a single question with one answer.
    refusals.extend(ecs.verify_frozen_code(repo_root, freeze))
    refusals.extend(ecs.verify_confirmation_images(repo_root, queries,
                                                   by_assoc))
    revision = ecs.frozen_base_model_revision(freeze)
    live_revision = base_model_revision(model_id)
    #: TOTAL here, where the evaluator's two read-only modes tolerate an
    #: unresolvable ref.  The difference is what each mode leaves behind: this
    #: one writes a report asserting
    #: ``base_model_revision_verified_at_runtime``, and a comparison that was
    #: skipped cannot make that claim true.  ``verify_sidecar`` happens to catch
    #: the same case — with no live ref the expected fingerprint records None
    #: while the sidecar records the pin — but it reports it as "file has
    #: c202236…, this run expects None", which does not tell the reader that the
    #: base model is unverifiable on THIS machine, and it would not catch a
    #: hand-written sidecar that recorded None too.
    if live_revision is None:
        refusals.append(
            f"no local HF cache ref for {model_id}, so the base-model revision "
            "these predictions were generated from cannot be compared with the "
            f"freeze's pin {revision}. The predictions were necessarily made on "
            "a machine that could resolve it; analyze them there, or restore "
            "the ref, rather than reporting a verification that did not happen")
    elif live_revision != revision:
        refusals.append(
            f"the local HF cache ref for {model_id} points at {live_revision} "
            f"but the freeze pins base_model_revision {revision}, so the base "
            "model these predictions were generated from cannot be shown to be "
            "the frozen one")

    # ---- the completeness gate: all three states, verified, same order ----
    preds_by_state: dict[str, list[PredictionRecord]] = {}
    state_audit: dict[str, Any] = {}
    for state in states:
        preds, reasons = ecs.verify_state(
            state, repo_root, freeze, queries, by_assoc, model_id,
            generation_config)
        ppath = ecs.predictions_path(repo_root, state)
        opath = ecs.generation_order_path(repo_root, state)
        #: ``verify_state`` has already refused a missing, unreadable or
        #: mismatched order record, so reading it here is for the AUDIT rather
        #: than for the gate — and it is read defensively because a state that
        #: did not verify has no record worth reading.
        recorded_order = None
        if preds is not None and opath.exists():
            recorded_order = json.loads(opath.read_text()).get(
                "generation_order_sha256")
        if preds is None:
            refusals.extend(f"{state}: {r}" for r in reasons)
            state_audit[state] = {"complete": False, "reasons": reasons}
            continue
        preds_by_state[state] = preds
        state_audit[state] = {
            "complete": True,
            "num_predictions": len(preds),
            "parquet_sha256": sha256_file(ppath),
            "sidecar_verified": True,
            "generation_order_sha256": recorded_order,
            "generation_order_matches_the_other_states":
                recorded_order == order_sha,
        }
    if refusals:
        #: Invariant 6.  Nothing below is computed, so no partial metric can
        #: leak into a log, a report or a reader's head.  The state count is
        #: still reported even though refusals can now also carry preflight
        #: problems, because "assembled from 2 of 3" is the part a reader has
        #: to see first.
        raise SystemExit(
            f"REFUSED — the confirmation cannot be assembled from "
            f"{len(preds_by_state)} of {len(states)} scored states; "
            f"{len(refusals)} refusal(s):\n  " + "\n  ".join(refusals))

    # ---- the frozen analysis parameters, read and cross-checked ----
    analysis = freeze.get("analysis") or {}
    boot = analysis.get("bootstrap") or {}
    n_bootstrap = boot.get("n_bootstrap")
    ci_level = boot.get("ci_level")
    bootstrap_seed = boot.get("seed")
    claims = freeze.get("claims") or {}
    primary_family = tuple(claims.get("primary_family") or ())
    familywise_alpha = claims.get("familywise_alpha")
    n_permutations = (freeze.get("primary_test") or {}).get("n_permutations")
    permutation_seed = (freeze.get("primary_test") or {}).get(
        "permutation_seed")
    for name, value in (("n_bootstrap", n_bootstrap),
                        ("ci_level", ci_level),
                        ("bootstrap_seed", bootstrap_seed),
                        ("familywise_alpha", familywise_alpha),
                        ("n_permutations", n_permutations),
                        ("permutation_seed", permutation_seed)):
        if value is None:
            raise SystemExit(
                f"REFUSED — the freeze does not state {name}, so the analysis "
                "would have to choose a value the protocol never fixed")
    if primary_family != EXPECTED_PRIMARY_FAMILY:
        refusals.append(
            f"the freeze declares the primary family as {list(primary_family)} "
            f"but this analysis was written for {list(EXPECTED_PRIMARY_FAMILY)}; "
            "Holm's thresholds and the cluster requirement were sized for two "
            "claims, so a different family is a different protocol")
    if len(primary_family) != (claims.get("k") or 0):
        refusals.append(
            f"the freeze declares k={claims.get('k')} for a primary family of "
            f"{len(primary_family)} claims; Holm's thresholds are alpha/(k-i), "
            "so the two have to be the same number")

    # ---- the target entity set, derived and pinned ----
    targets, target_audit = target_entity_set(repo_root, freeze)

    # ---- row flags, computed once per state over ALL queries ----
    #: ``split=None`` on purpose: every confirmation query is a test query, and
    #: passing ``split="test"`` would inherit the exploratory evaluator's
    #: filter and silently report a subset if the builder ever wrote another
    #: split.  ``include_adversarial`` stays False, and the split holds no
    #: adversarial rows, so the two agree here — but the default is the one
    #: the metric contract defines.
    flags_by_state = {
        state: row_flags(preds, queries, associations, split=None)
        for state, preds in preds_by_state.items()}
    for state, flags in flags_by_state.items():
        n_target_rows = len(flags["tga"])
        n_retain_rows = len(flags["retain_same"]) + len(flags["retain_other"])
        state_audit[state]["target_probe_rows_scored"] = n_target_rows
        state_audit[state]["retain_probe_rows_scored"] = n_retain_rows
        if n_target_rows + n_retain_rows != len(queries):
            refusals.append(
                f"{state}: {n_target_rows} target rows and {n_retain_rows} "
                f"retain rows do not account for all {len(queries)} queries, "
                "so some probe is in neither the primary nor the descriptive "
                "set and would be reported nowhere")

    #: The primary is restricted to the 72 target entities.  Every target probe
    #: is on a target association by construction, so this restriction should
    #: drop nothing — and asserting that is the check, because if it dropped
    #: something then some target probe belonged to a retain-only entity and
    #: the estimand would not be the one the freeze names.
    primary_flags = {
        state: {m: _restrict(f[m], entities=targets)
                for m in PAIRED_METRICS}
        for state, f in flags_by_state.items()}
    for state, f in primary_flags.items():
        if len(f["tga"]) != len(flags_by_state[state]["tga"]):
            refusals.append(
                f"{state}: restricting to the {len(targets)} target entities "
                f"dropped {len(flags_by_state[state]['tga']) - len(f['tga'])} "
                "target probes, so a target probe belongs to a retain-only "
                "entity and the pooled estimand is not over the frozen set")

    # ---- PRIMARY: B3 - B0 on TGA and FILR, entity-macro, over 72 entities ----
    primary_claims: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        direction = CLAIM_DIRECTION[metric]
        comp = _comparison(primary_flags["B3"][metric],
                           primary_flags["B0"][metric],
                           n_bootstrap, ci_level, bootstrap_seed,
                           n_permutations, permutation_seed,
                           direction=direction)
        if comp is None:
            refusals.append(
                f"B3_minus_B0:{metric} has no paired rows at all, so the "
                "primary claim cannot be computed")
            continue
        name = f"B3_minus_B0:{metric}"
        n_units = comp["num_entity_clusters"]
        if n_units != len(targets):
            refusals.append(
                f"{name} averages over {n_units} entity clusters but the "
                f"primary estimand is pooled over all {len(targets)} target "
                "entities; an entity with no paired probe would silently drop "
                "out of the macro average and change what the claim is about")
        comp["metric"] = metric
        comp["states_compared"] = ["B3", "B0"]
        comp["entity_set"] = "all target entities"
        primary_claims[name] = comp
        p_values[name] = comp["test"]["p_value_one_sided"]

    holm = holm_family(p_values, familywise_alpha) if p_values else None
    if holm is not None:
        if holm.get("k") != len(EXPECTED_PRIMARY_FAMILY):
            refusals.append(
                f"Holm ran over k={holm.get('k')} claims but the frozen family "
                f"has {len(EXPECTED_PRIMARY_FAMILY)}; the thresholds are "
                "alpha/(k-i), so a different k tests a different family")
        frozen_thresholds = claims.get("holm_thresholds")
        got_thresholds = [s["threshold"] for s in holm.get("steps", [])]
        if frozen_thresholds and [round(t, 6) for t in frozen_thresholds] != \
                [round(t, 6) for t in sorted(got_thresholds)]:
            refusals.append(
                f"Holm produced thresholds {sorted(got_thresholds)} but the "
                f"freeze declares {frozen_thresholds}")

    # ---- SECONDARY: the two image strata, diagnostics only ----
    strata: dict[str, Any] = {}
    for stratum, label in (("seen_photo_unseen_wording", "person"),
                           ("held_out_photo", "photograph")):
        qids = _qids_in_stratum(queries, stratum)
        entities = {by_assoc[q.association_id].entity_id
                    for q in queries if q.query_id in qids}
        block: dict[str, Any] = {
            "stratum": stratum,
            "label": label,
            "num_probes": len(qids),
            "num_entities": len(entities),
            "status": "SECONDARY DIAGNOSTIC — carries no primary claim and no "
                      "Holm entry",
            "metrics": {},
        }
        for metric in PAIRED_METRICS:
            fa = _restrict(flags_by_state["B3"][metric], qids=qids)
            fb = _restrict(flags_by_state["B0"][metric], qids=qids)
            #: No direction, so no p-value: a stratum is where an exploratory
            #: effect was seen, and computing a test here would be a second
            #: family nobody adjusted for.
            comp = _comparison(fa, fb, n_bootstrap, ci_level, bootstrap_seed,
                               n_permutations, permutation_seed,
                               direction=None)
            if comp is not None:
                block["metrics"][metric] = comp
        strata[stratum] = block

    # ---- DESCRIPTIVE: retention on the text route, and B3 - MG ----
    retention: dict[str, Any] = {
        "route": "text_to_text only",
        "status": "DESCRIPTIVE — no margin, no non-inferiority test",
        "entity_set": (
            "every entity carrying the metric, NOT restricted to the 72 target "
            "entities"),
        "why_the_entity_set_is_not_the_primary_one": (
            "the primary estimand is pooled over the 72 target entities, but "
            "retention is descriptive and is reported over the entities that "
            "actually carry a retain probe — 70 clusters for retain_same and "
            "45 for retain_other. The power report sizes both against a ceiling "
            "of 70 ('entities carrying a retain association in the frozen "
            "pilot100_v2 manifest') and records that retain_same saturates it "
            "while retain_other does not, so 45 is the achieved count and not a "
            "second ceiling. Restricting the block to the target set would drop "
            "28 entities' retention evidence for no preregistered reason, and "
            "'retention was not harmed' is a claim about everything the "
            "adapters were trained to keep, not about the subset that was also "
            "unlearned from."),
        "why": (
            "both retention claims were demoted by the 11C-2a decisions: the "
            "batch-layout noise floor on the retain metrics is 0.0556, "
            "measured by scoring one checkpoint under two batch layouts, and "
            "it already exceeds the 0.05 margin that was declared, so no "
            "interval at the achievable cluster count could conclude "
            "non-inferiority at it. The intervals are published instead."),
        "declared_margin": None,
        "metrics": {},
    }
    for metric in RETENTION_METRICS:
        #: UNRESTRICTED flags, so the descriptive block covers the entities it
        #: was sized over rather than the primary's target subset.
        comp = _comparison(flags_by_state["B3"][metric],
                           flags_by_state["B0"][metric],
                           n_bootstrap, ci_level, bootstrap_seed,
                           n_permutations, permutation_seed, direction=None)
        if comp is not None:
            comp["entity_set"] = "entities carrying this retain metric"
            retention["metrics"][f"B3_minus_B0:{metric}"] = comp

    mg: dict[str, Any] = {
        "status": "DESCRIPTIVE — equivalence is NOT concluded",
        "equivalence_concluded": False,
        "equivalence_test_run": False,
        "why_no_equivalence_conclusion": (
            "equivalence to M_G would need a margin fixed before the "
            "comparison plus a test that the WHOLE interval fits inside it. "
            "The power analysis measured that equivalence at delta = 0.05 on "
            "TGA would need 560 entity clusters against a hard ceiling of 72, "
            "and on FILR 2702 against the same 72, so the confirmation cannot "
            "conclude it at any margin it could defend. An interval that "
            "straddles zero licenses only 'no significant difference "
            "detected'; reporting that as 'behaves like M_G' would be reading "
            "absence of evidence as evidence of absence."),
        "margin_that_is_not_tested": (freeze.get("scorer") or {}).get(
            "equivalence_margin", {}).get("value"),
        "margin_role": (freeze.get("scorer") or {}).get(
            "equivalence_margin", {}).get("role"),
        "metrics": {},
    }
    for metric in PAIRED_METRICS:
        comp = _comparison(primary_flags["B3"][metric],
                           primary_flags["MG"][metric],
                           n_bootstrap, ci_level, bootstrap_seed,
                           n_permutations, permutation_seed, direction=None)
        if comp is not None:
            comp["states_compared"] = ["B3", "MG"]
            comp["entity_set"] = "all target entities"
            mg["metrics"][f"B3_minus_MG:{metric}"] = comp

    report = {
        "iteration": "11C-5",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "what_this_is": (
            "the confirmation analysis, assembled once from three "
            "provenance-verified prediction files over all "
            f"{len(queries)} confirmation queries"),
        "inputs_bound": {
            "freeze_report": ecs.FREEZE_REPORT,
            "freeze_sha256": sha256_file(repo_root / ecs.FREEZE_REPORT),
            "power_report": ecs.CONFIRM_REPORT,
            "power_sha256": sha256_file(repo_root / ecs.CONFIRM_REPORT),
            "dataset_dir": ecs.CONFIRM_DATASET_DIR,
            "dataset_version": dataset_version(data_dir),
            "queries_parquet_sha256": sha256_file(data_dir / "queries.parquet"),
            "query_id_list_sha256": ecs._sha256_id_list(
                [q.query_id for q in queries]),
            "generation_order_sha256": order_sha,
            "experiment_id": ecs.EXPERIMENT_ID,
            "base_model_id": model_id,
            "base_model_revision": revision,
            "states": state_audit,
        },
        "protocol_compliance": {
            "states_scored": list(states),
            "states_excluded": (freeze.get("states") or {}).get("excluded"),
            "generation_config_is_the_frozen_one": True,
            "generation_config": generation_config,
            "query_ordering_identical_across_states": all(
                s.get("generation_order_matches_the_other_states")
                for s in state_audit.values()),
            "reference_state_gate_invoked": False,
            "reference_state_gate_parquets_read": [],
            "checkpoint_selection_invoked": False,
            "selection_report_read": False,
            "why_neither_is_invoked": (
                "sealed-split invariants 4 and 5: the confirmation split never "
                "enters the reference-state gate and never enters candidate "
                "selection or any other go/no-go decision. Selection is "
                "complete and frozen, so re-running it on confirmation data "
                "would be selecting on the test set; and the gate is a go/no-go "
                "on the oracle, so letting it see confirmation data would make "
                "the confirmation exploratory in the same way. This module "
                "does not know either report's path."),
            "partial_results_were_inspectable_before_completion": False,
            "how_that_is_enforced": (
                "evaluate_confirmation_split.py imports no aggregation code at "
                "all, so a run that generated one state and stopped could not "
                "print a rate, an interval or a p-value; this script refuses "
                "to compute anything until all three sidecars verify"),
            "frozen_code_hashes_verified_at_runtime": True,
            "image_manifest_verified_at_runtime": True,
            "base_model_revision_verified_at_runtime": True,
            "what_the_three_runtime_verifications_are": (
                "the analysis is a separate process from the generation and may "
                "run on a different checkout, so the freeze's pins are "
                "re-checked against the bytes on disk HERE rather than trusted "
                "from the generation run: every entry in "
                "code.analysis_scripts_sha256, the separately pinned "
                "primary_test.implementation.sha256 for paired_ci.py, the "
                "re-hash of every pinned photograph against image_manifest.json, "
                "the resolution of every image-route query to a photograph on "
                "disk, and the local base-model revision against the freeze's "
                "pin. Recording a hash without comparing it describes the "
                "protocol instead of enforcing it."),
            "why_the_photographs_are_checked_per_query_and_not_only_by_hash": (
                "ReferenceStateGenerator drops an image it cannot resolve and "
                "generates TEXT-ONLY with no warning, so a photograph that is "
                "correctly pinned but unresolvable for a particular query would "
                "still produce a provenance-valid sidecar over evidence that "
                "was never visual. That module is in CODE_FINGERPRINT_MODULES "
                "and cannot be edited without invalidating the 30 committed "
                "exploratory sidecars, so the gap is closed in front of it."),
            "equivalence_test_run": False,
            "non_inferiority_test_run": False,
            "anything_tuned_on_these_predictions": False,
        },
        "primary": {
            "estimand": claims.get("primary_estimand"),
            "target_entities": target_audit,
            "claims": primary_claims,
            "multiplicity": {
                "procedure": claims.get("multiplicity"),
                "familywise_alpha": familywise_alpha,
                "k": len(primary_family),
                "thresholds_apply_to": claims.get("thresholds_apply_to"),
                "holm": holm,
            },
            "averaging_units_are_reported_separately": True,
            "verdict": _verdict(holm, primary_claims) if holm else None,
        },
        "secondary_strata": {
            "status": "SECONDARY DIAGNOSTICS ONLY",
            "why": (
                "the person stratum is where the exploratory effect was seen, "
                "so it is reported because a reader needs it and not because "
                "it is tested; the photograph stratum measured -0.0111 over 30 "
                "clusters exploratorily with an interval that did not exclude "
                "zero. Neither is in the Holm family, and neither can carry a "
                "primary conclusion."),
            "strata": strata,
        },
        "descriptive": {
            "retention": retention,
            "b3_minus_mg": mg,
            "other_target_metrics": {
                m: _comparison(primary_flags["B3"][m], primary_flags["B0"][m],
                               n_bootstrap, ci_level, bootstrap_seed,
                               n_permutations, permutation_seed,
                               direction=None)
                for m in DESCRIPTIVE_TARGET_METRICS},
        },
        "analysis_parameters_read_from_the_freeze": {
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "bootstrap_seed": bootstrap_seed,
            "n_permutations": n_permutations,
            "permutation_seed": permutation_seed,
            "familywise_alpha": familywise_alpha,
            "primary_family": list(primary_family),
            "paired_metrics": list(PAIRED_METRICS),
            "none_of_these_is_a_cli_flag_because": (
                "every one is frozen, and a flag that can move a frozen value "
                "is a route to drifting from the protocol without editing it"),
        },
        "refusals": refusals,
    }
    return report


def _verdict(holm: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
    """The Holm outcome, stated per claim, with no path that softens a failure.

    Invariant 7: a failed primary claim is reported as failed.  So the verdict
    is read off ``holm`` rather than composed here, and the text for a retained
    claim says what retention means — no rejection at this familywise rate —
    rather than anything that could be read as a near miss.
    """
    rejected = set(holm.get("rejected") or [])
    out: dict[str, Any] = {
        "all_rejected": bool(holm.get("all_rejected")),
        "rejected": sorted(rejected),
        "retained": sorted(holm.get("retained") or []),
        "per_claim": {},
    }
    for step in holm.get("steps", []):
        name = step["claim"]
        comp = claims.get(name) or {}
        out["per_claim"][name] = {
            "rejected": bool(step["rejected"]),
            "p_value_one_sided": step["p_value_one_sided"],
            "threshold_at_its_step": step["threshold"],
            "entity_macro_diff": (comp.get("entity_macro") or {}).get("diff"),
            "entity_macro_ci": (comp.get("entity_macro") or {}).get("ci"),
            "row_micro_diff": (comp.get("row_micro") or {}).get("diff"),
            "statement": (
                f"{name} is REJECTED at familywise alpha "
                f"{holm.get('familywise_alpha')} by Holm step {step['step']}"
                if step["rejected"] else
                f"{name} is NOT rejected: Holm stopped at step "
                f"{step['step']}, so this claim is retained and no "
                "significance is claimed for it"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Run every refusal and the completeness gate, then "
                             "stop without computing or writing the analysis. "
                             "Useful before spending time on the bootstraps, "
                             "and it prints no metric either.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    report = analyze(repo_root)
    refusals = report["refusals"]
    if refusals:
        raise SystemExit(
            "REFUSED — the analysis does not satisfy the frozen protocol:\n  "
            + "\n  ".join(refusals))
    if args.check_only:
        print("all three states verified and every protocol check passed; "
              "--check-only, so no analysis was computed and nothing written")
        return 0
    out = Path(args.output) if args.output else repo_root / ANALYSIS_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    primary = report["primary"]
    verdict = primary["verdict"] or {}
    print(f"\nwrote {out.relative_to(repo_root)}")
    print(f"  states       {list(report['inputs_bound']['states'])}, all "
          f"provenance-verified over the same query order")
    te = primary["target_entities"]
    print(f"  targets      {te['target_entity_ids']} entities "
          f"({te['persons']} persons, {te['species']} species), "
          f"set sha256 {te['target_entity_ids_sha256'][:16]}")
    for name, comp in sorted(primary["claims"].items()):
        em = comp["entity_macro"]
        print(f"  {name:20} entity-macro diff {em['diff']:+.4f} "
              f"CI [{em['ci'][0]:+.4f}, {em['ci'][1]:+.4f}] over "
              f"{comp['num_entity_clusters']} entities; "
              f"p(one-sided, {comp['direction']}) = "
              f"{comp['test']['p_value_one_sided']:.4f}; "
              f"row-micro diff {comp['row_micro']['diff']:+.4f}")
    print(f"  Holm         alpha={primary['multiplicity']['familywise_alpha']} "
          f"k={primary['multiplicity']['k']} -> rejected "
          f"{verdict.get('rejected')}, retained {verdict.get('retained')}")
    print("  secondary    person and photograph strata reported as "
          "diagnostics only")
    print("  descriptive  retention (no margin) and B3-MG (equivalence NOT "
          "concluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
