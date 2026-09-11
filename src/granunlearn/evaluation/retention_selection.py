"""Iteration 12 selection criterion: D_G, unchanged, behind a retention floor.

WHAT IS FROZEN HERE
-------------------
Three things had to be pinned down before a single candidate was trained,
because each of them can change which candidate wins and none of them is
implied by the words "closest to MG subject to retention not falling":

1. **the direction of D_G.**  D_G is a DISTANCE and is MINIMISED.  It is not a
   score where larger is better, and -- this is the part that is easy to
   misread -- it does not reward low leakage.  Every component enters as
   ``|v_j(M_U) - v_j(M_G)|``, so a candidate that drives FILR BELOW MG's level
   is penalised exactly as much as one that leaves FILR above it.  The
   criterion asks "does this look like granularity-controlled unlearning", not
   "does this unlearn hard".  The floor is the only place where a direction is
   monotone: retention must be at least B0's, so more is never disqualifying.

2. **the numerical tolerance.**  Two distinct tolerances, neither of them a
   margin.  Distances are compared exactly as
   :func:`~granunlearn.evaluation.selection.distance_to_reference` produces
   them -- rounded to 6 decimals -- with NO additional epsilon, so two
   candidates are tied if and only if their rounded distances are equal.  The
   floor uses ``FLOOR_EPSILON = 1e-9``, which exists only to stop an exactly
   equal score being reported as a shortfall by the last bit of a division.
   The smallest difference any of the four frozen retention numbers can
   express is ~2.0e-3 (one flipped outcome in the widest of 35 probe
   entities), so 1e-9 sits six orders of magnitude below anything the
   measurement can resolve and provably cannot flip a real comparison.  A
   +0.02 retention margin was considered and REJECTED: nothing in the
   evidence supports a threshold of that size, and requiring improvement
   beyond "not below B0" would be inventing a hypothesis and calling it a
   constraint.

3. **the tie-break.**  ``(distance_to_mg, candidate_id)`` ordered ascending,
   lexicographic on the id.  The candidate id encodes hyperparameters and no
   outcome, so this order cannot be gamed by looking at results, and it does
   not depend on grid order, on filesystem order, or on which adapters happen
   to exist on disk -- all three of which are how an unwritten tie-break
   actually gets decided.  The tie-break that
   :func:`~granunlearn.evaluation.selection.select_checkpoints` uses today is
   a strict ``<`` over dict insertion order, which is none of those things but
   is also not written down anywhere; a tie is now reported as an event, not
   silently absorbed.

WHY D_G ITSELF IS NOT MODIFIED
------------------------------
The two retention components of ``v(M)`` are pooled row-micro over train+val
and are, for a replay candidate, measured on rehearsed associations.  That is
a bias, and it is handled by the FLOOR rather than by editing D_G, for two
reasons.  Within Stage 1 every candidate carries a replay group, so they are
all in-sample on the same fit half and mutually comparable; the
cross-mechanism comparison -- replay candidate against no-op B0 -- is exactly
what the probe-restricted floor does cleanly.  And D_G is the criterion the
pilot-100 selection was made under, so keeping it byte-identical keeps the
successor comparable with the incumbent it replaces.

:func:`distance_to_mg_on_probe` is reported beside the decision and is NOT
decision-bearing: it is the same formula with both retention components taken
from the probe half, so a reader can see what the in-sample terms were worth.
It never selects anything.

Nothing in this module reads the sealed confirmation split, and
:data:`FORBIDDEN_EVIDENCE` exists so a caller can be checked against trying.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from granunlearn.evaluation.hierarchy_metrics import compute_hierarchy_metrics
from granunlearn.evaluation.selection import (
    SUMMARY_COMPONENTS,
    distance_to_reference,
    summary_vector,
    trainval_hierarchy_metrics,
)
from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord

#: Directions, frozen.  D_G is minimised; the floor is a >= constraint.
D_G_DIRECTION = "minimize"
FLOOR_DIRECTION = "candidate_must_be_at_or_above_b0"

#: Distances are compared exactly as ``distance_to_reference`` rounds them.
DISTANCE_DECIMALS = 6
ADDITIONAL_EPSILON_ON_DISTANCES = 0.0

#: Floor tolerance.  Absorbs float representation error only; see the module
#: docstring for why this cannot decide a real comparison and is not a margin.
FLOOR_EPSILON = 1e-9

#: Rates are reported at 4 decimals because that is what the sealed
#: ``hierarchy_metrics._rate`` does, so a floor value here is the same
#: quantity, at the same precision, as every other rate in this repository.
#: The rounding cannot merge two genuinely different retention values: the
#: smallest difference the probe measurement can express is ~2.0e-3, two
#: orders of magnitude above the 1e-4 quantum.  Distances stay at 6 decimals
#: because that is ``distance_to_reference``'s own convention, and mixing the
#: two precisions is deliberate rather than sloppy -- each number is rounded
#: where it is computed, never re-rounded afterwards.
RATE_DECIMALS = 4

#: The two retention families the floor is stated over.
RETENTION_FAMILIES = ("retain_same_entity", "retain_other_entity")

#: Both estimands must pass.  They disagree in SIGN for retain_other on the
#: pilot-100 evidence (row-micro -0.1056 vs entity-macro +0.0093 for the
#: incumbent against B0), so freezing either one alone would be choosing the
#: estimand that produces the preferred answer.  Requiring both is the
#: conservative pre-commitment, and it is made before any candidate exists.
RETENTION_ESTIMANDS = ("entity_macro", "row_micro")

#: Evidence this study must never read.  The confirmation is sealed and was
#: scored once; tuning a successor on it would make the confirmation
#: retrospective.
FORBIDDEN_EVIDENCE = (
    "data/mllmu_hier_confirm100",
    "data/reports/mllmu_confirm100_final_analysis.json",
)


def entity_of(associations: Sequence[AssociationRecord]) -> dict[str, str]:
    return {a.association_id: a.entity_id for a in associations}


def _rows(predictions: Sequence[PredictionRecord],
          queries: Sequence[QueryRecord],
          entity_map: dict[str, str],
          family: str,
          probe_entities: frozenset[str]) -> list[tuple[str, PredictionRecord]]:
    """Probe-half rows for one retention family, in prediction order."""
    by_id = {q.query_id: q for q in queries}
    out = []
    for p in predictions:
        q = by_id.get(p.query_id)
        if q is None or q.family != family:
            continue
        if entity_map.get(q.association_id) not in probe_entities:
            continue
        out.append((entity_map[q.association_id], p))
    return out


def entity_macro_rate(rows: Sequence[tuple[str, PredictionRecord]]) -> float | None:
    """Mean over entities of each entity's mean correctness.

    Computed here rather than in ``hierarchy_metrics`` because that module is
    one of the eighteen paths the confirmation freeze sealed: its bytes are
    bound by prediction sidecars, so adding an estimand to it would invalidate
    evidence that has already been filed.  Calling it with filtered inputs is
    a caller-side choice and changes nothing.
    """
    if not rows:
        return None
    per: dict[str, list[float]] = {}
    for ent, p in rows:
        per.setdefault(ent, []).append(float(bool(p.is_correct_branch)))
    entity_means = [sum(v) / len(v) for v in per.values()]
    return round(sum(entity_means) / len(entity_means), RATE_DECIMALS)


def row_micro_rate(rows: Sequence[tuple[str, PredictionRecord]]) -> float | None:
    """Pooled mean correctness over probe rows.

    The floor's row-micro value is the one the sealed
    ``compute_hierarchy_metrics`` returns, so that it is the same quantity the
    rest of the repository reports.  This direct computation is kept beside it
    as a CROSS-CHECK that runs on every call: if the probe restriction ever
    stops reaching the sealed path, the two disagree and
    :func:`probe_retention` raises rather than filing a floor measured on
    in-sample retention.
    """
    if not rows:
        return None
    flags = [float(bool(p.is_correct_branch)) for _, p in rows]
    return round(sum(flags) / len(flags), RATE_DECIMALS)


def probe_retention(predictions: Sequence[PredictionRecord],
                    queries: Sequence[QueryRecord],
                    associations: Sequence[AssociationRecord],
                    probe_entities: Sequence[str]) -> dict[str, Any]:
    """Both estimands of both retention families, on probe queries only.

    ``row_micro`` is read from the sealed metric implementation applied to
    probe-filtered inputs; ``entity_macro`` is averaged here.  The counts are
    returned with the values because a retention number without its
    denominator is not checkable -- 23 donor entities is the difference
    between a floor and a coin flip.
    """
    entity_map = entity_of(associations)
    probe = frozenset(probe_entities)
    split_of = {q.query_id: q.split for q in queries}
    probe_preds = [p for p in predictions if split_of.get(p.query_id)
                   in ("train", "val")]

    rows = {family: _rows(probe_preds, queries, entity_map, family, probe)
            for family in RETENTION_FAMILIES}
    #: The sealed metric implementation is handed ONLY probe retention rows.
    #: Filtering by split alone is not enough -- it yields the pooled
    #: train+val rate, which for a replay candidate is the IN-SAMPLE number
    #: this partition exists to avoid.  The cross-check below is what stops
    #: that mistake being silent: it compares the sealed block against a
    #: direct computation over exactly the rows that were passed in, so a
    #: wider input disagrees and raises instead of producing a plausible
    #: wrong floor.
    probe_only = [p for family in RETENTION_FAMILIES for _, p in rows[family]]
    sealed = compute_hierarchy_metrics(probe_only, queries, associations,
                                       split=None)
    out: dict[str, Any] = {"scope": ["train", "val"],
                           "probe_entities": len(probe)}
    for family in RETENTION_FAMILIES:
        family_rows = rows[family]
        block = sealed.get(family) or {}
        micro = block.get("baseline_accuracy")
        micro = round(float(micro), RATE_DECIMALS) if micro is not None else None
        direct = row_micro_rate(family_rows)
        if micro != direct:
            raise AssertionError(
                f"{family}: the sealed metric returned {micro} over rows that "
                f"compute directly to {direct}, so the probe restriction did "
                f"not reach it -- the floor would have been measured on "
                f"in-sample retention")
        out[family] = {
            "num_queries": len(family_rows),
            "num_entities": len({e for e, _ in family_rows}),
            "row_micro": micro,
            "entity_macro": entity_macro_rate(family_rows),
        }
    return out


def floor_check(candidate: dict[str, Any], baseline: dict[str, Any],
                epsilon: float = FLOOR_EPSILON) -> dict[str, Any]:
    """Is this candidate at or above B0 on all four frozen retention numbers?

    ``candidate >= baseline - epsilon``.  The epsilon is subtracted from the
    BASELINE, so it can only rescue an exact tie; it can never let a genuine
    shortfall through, because a genuine shortfall is at least ~2.0e-3 and
    the epsilon is 1e-9.

    A family whose probe value is missing on either side DISQUALIFIES rather
    than passing: a retention number that could not be computed is not
    evidence that retention was preserved.
    """
    checks: dict[str, Any] = {}
    eligible = True
    for family in RETENTION_FAMILIES:
        for estimand in RETENTION_ESTIMANDS:
            got = (candidate.get(family) or {}).get(estimand)
            want = (baseline.get(family) or {}).get(estimand)
            key = f"{family}.{estimand}"
            if got is None or want is None:
                checks[key] = {
                    "candidate": got, "b0": want, "passes": False,
                    "why": "not measurable on the probe half -- treated as a "
                           "failure, because an absent retention number is "
                           "not evidence of preserved retention",
                }
                eligible = False
                continue
            passes = got >= want - epsilon
            checks[key] = {
                "candidate": got, "b0": want,
                "difference": round(got - want, 6),
                "passes": passes,
                "comparison": f"candidate >= b0 - {epsilon:g}",
            }
            eligible = eligible and passes
    return {
        "eligible": eligible,
        "epsilon": epsilon,
        "epsilon_is_a_tolerance_not_a_margin": True,
        "checks": checks,
        "direction": FLOOR_DIRECTION,
        "estimands_required": list(RETENTION_ESTIMANDS),
        "why_both_estimands": (
            "row-micro and entity-macro disagree in sign for retain_other on "
            "the pilot-100 evidence, so freezing one of them would be picking "
            "the estimand that yields the preferred verdict"),
    }


def distance_to_mg(vector: dict[str, float | None],
                   reference_vector: dict[str, float | None],
                   weights: dict[str, float] | None = None,
                   ) -> tuple[float | None, list[str]]:
    """D_G, delegated to the pilot-100 implementation without modification.

    A thin wrapper so the frozen protocol can name the function that computes
    the criterion and a test can prove this module did not reimplement it.
    """
    return distance_to_reference(vector, reference_vector, weights)


def distance_to_mg_on_probe(vector: dict[str, float | None],
                            probe: dict[str, Any],
                            reference_vector: dict[str, float | None],
                            weights: dict[str, float] | None = None,
                            ) -> tuple[float | None, list[str]]:
    """The same formula with both retention terms taken from the probe half.

    REPORTED, NEVER DECISION-BEARING.  It exists so the cost of the in-sample
    retention terms inside D_G is visible rather than assumed away.
    """
    swapped = dict(vector)
    swapped["retain_same"] = (probe.get("retain_same_entity") or {}).get(
        "row_micro")
    swapped["retain_other"] = (probe.get("retain_other_entity") or {}).get(
        "row_micro")
    return distance_to_reference(swapped, reference_vector, weights)


def rank_key(distance: float, candidate_id: str) -> tuple[float, str]:
    """The frozen ordering: distance ascending, then id lexicographically."""
    return (distance, candidate_id)


def select_successor(candidates: dict[str, dict[str, Any]],
                     reference_vector: dict[str, float | None],
                     b0_probe: dict[str, Any],
                     weights: dict[str, float] | None = None,
                     epsilon: float = FLOOR_EPSILON,
                     ) -> dict[str, Any]:
    """Apply the floor, then minimise D_G with the frozen tie-break.

    ``candidates`` maps candidate_id -> {"method", "trainval_metrics",
    "probe_retention", ...}.  ``b0_probe`` is the no-op's probe retention,
    which is the floor.  Selection is over the whole grid rather than per
    method: Stage 1 compares replay weights against each other AND against
    the incumbent-recipe reference row, which is itself one of the weights.
    """
    rows: dict[str, Any] = {}
    for cid, info in candidates.items():
        vector = summary_vector(info["trainval_metrics"])
        dist, used = distance_to_mg(vector, reference_vector, weights)
        probe = info.get("probe_retention") or {}
        floor = floor_check(probe, b0_probe, epsilon)
        dg_probe, _ = distance_to_mg_on_probe(
            vector, probe, reference_vector, weights)
        rows[cid] = {
            "method": info["method"],
            "vector": vector,
            "distance_to_mg": dist,
            "used_components": used,
            "probe_retention": probe,
            "floor": floor,
            "eligible": floor["eligible"] and dist is not None,
            "distance_to_mg_on_probe": dg_probe,
            "distance_to_mg_on_probe_is_decision_bearing": False,
            "config": info.get("config"),
        }

    eligible = {cid: r for cid, r in rows.items() if r["eligible"]}
    ranked = sorted(eligible, key=lambda c: rank_key(
        rows[c]["distance_to_mg"], c))
    selected = ranked[0] if ranked else None

    #: A tie is an event worth reporting: it means the criterion did not
    #: discriminate, and the tie-break -- not the science -- chose the winner.
    tied: list[str] = []
    if selected is not None:
        best = rows[selected]["distance_to_mg"]
        tied = sorted(cid for cid in eligible
                      if cid != selected
                      and rows[cid]["distance_to_mg"] == best)

    disqualified = sorted(cid for cid, r in rows.items()
                          if not r["floor"]["eligible"])
    return {
        "criterion": {
            "name": "D_G",
            "formula": ("D_G(M_U) = sum_j w_j * |v_j(M_U) - v_j(M_G)| / "
                        "sum_j w_j over the components present on both sides"),
            "components": list(SUMMARY_COMPONENTS),
            "weights": weights or {c: 1.0 for c in SUMMARY_COMPONENTS},
            "direction": D_G_DIRECTION,
            "computed_by": (
                "granunlearn.evaluation.selection.distance_to_reference, "
                "imported and not reimplemented"),
            "numerical_tolerance": {
                "distance_decimals": DISTANCE_DECIMALS,
                "additional_epsilon": ADDITIONAL_EPSILON_ON_DISTANCES,
                "tied_iff": "rounded distances are exactly equal",
            },
            "tie_break": {
                "rule": "(distance_to_mg, candidate_id) ascending",
                "id_ordering": "lexicographic",
                "carries_no_outcome_information": True,
                "does_not_depend_on": [
                    "grid order", "filesystem order",
                    "which adapters exist on disk"],
            },
        },
        "floor": {
            "statement": ("retain_same_entity and retain_other_entity on "
                          "PROBE-half train+val queries must be at or above "
                          "B0's on the same queries, under BOTH entity-macro "
                          "and row-micro"),
            "b0": b0_probe,
            "epsilon": epsilon,
            "consequence_of_failing": (
                "ineligible, however small its D_G -- the floor is a "
                "constraint, not a term in the criterion"),
        },
        "reference": {"state": "MG", "vector": reference_vector},
        "basis": ("pilot-100 train+val probes only; the frozen test split is "
                  "held out and the sealed confirmation split is never read"),
        "candidates": rows,
        "eligible": sorted(eligible),
        "disqualified_by_the_floor": disqualified,
        "ranking_of_eligible": ranked,
        "selected": selected,
        "tied_with": tied,
        "a_tie_means": (
            "the criterion did not discriminate between these candidates; the "
            "tie-break chose, and that is reported rather than absorbed"
            if tied else None),
    }


def stage2_gate(report: dict[str, Any], reference_row: str) -> dict[str, Any]:
    """Does the MF-preservation regularizer stage open?

    Frozen BEFORE training: Stage 2 opens if Stage 1 produced no eligible
    candidate at all, or if the best eligible one does not STRICTLY improve on
    the incumbent-recipe reference row's D_G.  "Strictly" means a smaller
    6-decimal distance; an exact tie counts as no improvement, because a
    mechanism that changes nothing has not earned the claim that replay was
    enough.

    No margin appears here.  In particular there is no retention threshold the
    winner must clear beyond the floor itself -- a +0.02 margin was proposed
    and rejected as unsupported by any evidence in this repository.
    """
    rows = report["candidates"]
    selected = report["selected"]
    if reference_row not in rows:
        raise ValueError(
            f"the reference row {reference_row!r} is not in the grid; the "
            f"Stage 2 gate is undefined without it")
    ref_dist = rows[reference_row]["distance_to_mg"]
    if selected is None:
        return {
            "stage2_opens": True,
            "reason": "no candidate satisfied the retention floor",
            "best_eligible_distance_to_mg": None,
            "reference_row": reference_row,
            "reference_distance_to_mg": ref_dist,
            "margin_required": 0.0,
        }
    best = rows[selected]["distance_to_mg"]
    improves = ref_dist is not None and best < ref_dist
    return {
        "stage2_opens": not improves,
        "reason": ("the best eligible candidate strictly improves on the "
                   "incumbent-recipe reference row" if improves else
                   "the best eligible candidate does not strictly improve on "
                   "the incumbent-recipe reference row"),
        "best_eligible": selected,
        "best_eligible_distance_to_mg": best,
        "reference_row": reference_row,
        "reference_distance_to_mg": ref_dist,
        "reference_row_was_itself_eligible": rows[reference_row]["eligible"],
        "comparison": "strictly smaller 6-decimal distance; a tie is no "
                      "improvement",
        "margin_required": 0.0,
        "no_margin_is_used": True,
    }


def trainval_vector(predictions: Sequence[PredictionRecord],
                    queries: Sequence[QueryRecord],
                    associations: Sequence[AssociationRecord],
                    ) -> dict[str, float | None]:
    """v(M) on the selection scope, via the pilot-100 implementation."""
    return summary_vector(
        trainval_hierarchy_metrics(predictions, queries, associations))
