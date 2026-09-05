"""Paired entity-clustered bootstrap CIs for MLLMU metrics (Iteration 11).

The pilot-100 final evaluation must report PAIRED confidence intervals
for the headline rates — fine information leakage rate (FILR),
target-granularity accuracy (TGA), wrong-branch rate, over-forgetting
rate, same-entity-retain accuracy, and other-entity-retain accuracy —
comparing two checkpoints over the SAME queries.

Design (mirrors the frozen SALMU paired-CI machinery):
* rows are query-level 0/1 outcomes, identical query sets for both
  states (intersection pairing — a state missing a prediction drops
  the row for BOTH states, keeping the comparison paired);
* the cluster unit is the ENTITY behind the asked association
  (retain_other probes cluster by the DONOR entity that is probed);
* per-entity rates are macro-averaged; the paired difference
  mean(rate_a - rate_b) gets a percentile bootstrap CI resampling
  ENTITIES (the same entities for both states);
* identical predictions degenerate to diff 0 with CI [0, 0]
  (the B0 == MF invariant is testable through this path).
"""

from __future__ import annotations

from typing import Any

from granunlearn.evaluation.hierarchy_metrics import (
    build_pool_value_index,
    classify_target_failure,
)
from granunlearn.logging_utils import setup_logger
from granunlearn.schema import (
    AssociationRecord,
    PredictionRecord,
    QueryRecord,
)

log = setup_logger("paired_ci")

#: Headline paired metrics (user-facing names).  ``filr`` and
#: ``over_forgetting`` were added in Iteration 11R: fine leakage is the
#: central claim of the granularity thesis, so B3's FILR needed an interval
#: of its own rather than only a point estimate next to MG's.
PAIRED_METRICS = ("filr", "tga", "wrong_branch", "over_forgetting",
                  "retain_same", "retain_other")

RETAIN_SAME_FAMILIES = {"retain_same_entity",
                        "retain_same_entity_image"}
RETAIN_OTHER_FAMILIES = {"retain_other_entity",
                         "retain_other_entity_image"}


def row_flags(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    split: str | None = None,
    include_adversarial: bool = False,
) -> dict[str, dict[str, tuple[int, str]]]:
    """Per-metric row outcomes: {metric: {query_id: (flag, entity_id)}}.

    * ``filr`` / ``tga`` / ``wrong_branch`` / ``over_forgetting`` — target
      probes (post-unlearning failure taxonomy; adversarial probes excluded
      by default).  All four are mutually exclusive categories of the SAME
      row set, so their flags sum to 1 per row together with refusal and
      hallucination.  ``filr`` is exactly ``under_forgetting``: the
      taxonomy's fine-leakage category and the FILR numerator are the same
      rows by construction (see ``hierarchy_metrics``).
    * ``retain_same`` / ``retain_other`` — retain probes across BOTH
      routes (baseline correctness: the retained fine value must
      still be produced).
    """
    by_id = {q.query_id: q for q in queries}
    by_assoc = {a.association_id: a for a in associations}
    pool_index = build_pool_value_index(associations)
    out: dict[str, dict[str, tuple[int, str]]] = {
        m: {} for m in PAIRED_METRICS}
    for p in predictions:
        q = by_id.get(p.query_id)
        if q is None:
            continue
        if split is not None and q.split != split:
            continue
        assoc = by_assoc.get(q.association_id)
        if assoc is None:
            continue
        entity = assoc.entity_id
        fam = q.family or ""
        if fam in RETAIN_SAME_FAMILIES:
            out["retain_same"][q.query_id] = (
                int(bool(p.is_correct_branch)), entity)
        elif fam in RETAIN_OTHER_FAMILIES:
            out["retain_other"][q.query_id] = (
                int(bool(p.is_correct_branch)), entity)
        elif not fam.startswith("retain_"):
            if q.adversarial and not include_adversarial:
                continue
            cat = classify_target_failure(q, assoc, p, pool_index)
            out["tga"][q.query_id] = (
                int(cat == "correct_at_target"), entity)
            out["wrong_branch"][q.query_id] = (
                int(cat == "wrong_branch"), entity)
            # FILR's numerator IS the under_forgetting category, so the
            # flag is read off the same classification rather than
            # recomputed from leakage_forbidden_ids: two independent
            # implementations of one number is how they silently diverge.
            out["filr"][q.query_id] = (
                int(cat == "under_forgetting"), entity)
            out["over_forgetting"][q.query_id] = (
                int(cat == "over_forgetting"), entity)
    return out


def _paired_unit_diffs(
    fa: dict[str, tuple[int, str]],
    fb: dict[str, tuple[int, str]],
) -> dict[str, Any]:
    """Pair two states' row flags and summarise both averaging units.

    Returns ``{keys, diffs, num_rows, row_a, row_b, entity_a, entity_b}``
    (``keys`` empty when nothing is paired).

    Two averaging units are computed because both are needed and they are
    NOT equal: ``diffs`` macro-averages per entity, which is the unit the
    bootstrap resamples, while ``row_a``/``row_b`` micro-average over
    paired rows, which is what ``hierarchy_metrics`` publishes.  Entities
    contribute unequal numbers of probes, so reporting only one would let
    a reader subtract two published rates and get a number the CI was
    never built for.
    """
    common = sorted(set(fa) & set(fb))
    if not common:
        return {"keys": [], "diffs": [], "num_rows": 0,
                "row_a": 0.0, "row_b": 0.0, "entity_a": 0.0, "entity_b": 0.0}
    by_unit: dict[str, list[tuple[int, int]]] = {}
    sum_a = sum_b = n_paired = 0
    for qid in common:
        va, ea = fa[qid]
        vb, eb = fb[qid]
        if ea != eb:  # defensive: pairing must agree on the cluster
            continue
        by_unit.setdefault(ea, []).append((va, vb))
        sum_a += va
        sum_b += vb
        n_paired += 1
    keys = sorted(by_unit)
    per_entity_a = [sum(a for a, _ in by_unit[k]) / len(by_unit[k])
                    for k in keys]
    per_entity_b = [sum(b for _, b in by_unit[k]) / len(by_unit[k])
                    for k in keys]
    return {
        "keys": keys,
        "diffs": [a - b for a, b in zip(per_entity_a, per_entity_b)],
        "num_rows": len(common),
        "row_a": sum_a / n_paired if n_paired else 0.0,
        "row_b": sum_b / n_paired if n_paired else 0.0,
        "entity_a": sum(per_entity_a) / len(keys) if keys else 0.0,
        "entity_b": sum(per_entity_b) / len(keys) if keys else 0.0,
    }


def paired_rate_diff_ci(
    fa: dict[str, tuple[int, str]],
    fb: dict[str, tuple[int, str]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Paired entity-clustered percentile CI of rate(a) - rate(b).

    ``diff``/``ci`` are the ENTITY-MACRO paired difference: that is the
    statistic the bootstrap resamples, so the interval covers the estimate
    it was actually built for.  ``point_estimates`` reports both averaging
    units — ``row_*`` micro-averages over the paired rows and so reproduces
    the published ``hierarchy_metrics`` rate for that state exactly
    whenever both states cover the same query set (which
    ``validate_prediction_coverage`` enforces before any comparison is
    made); ``entity_*`` is the macro average the CI belongs to.  The two
    differ because entities contribute unequal numbers of probes.
    """
    import numpy as np

    from granunlearn.salmu.embedding_metrics import validate_ci_params
    validate_ci_params(n_bootstrap, ci_level)
    paired = _paired_unit_diffs(fa, fb)
    if not paired["keys"]:
        return None
    arr = np.asarray(paired["diffs"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pick = rng.integers(0, n, size=n)
        boots[b] = arr[pick].mean()
    alpha = (1 - ci_level) / 2
    return {
        "diff": round(float(arr.mean()), 4),
        "ci": (round(float(np.quantile(boots, alpha)), 4),
               round(float(np.quantile(boots, 1 - alpha)), 4)),
        "num_units": n,
        "num_rows": paired["num_rows"],
        # a = the state under test, b = the reference state.
        "point_estimates": {
            "row_a": round(paired["row_a"], 4),
            "row_b": round(paired["row_b"], 4),
            "row_diff": round(paired["row_a"] - paired["row_b"], 4),
            "entity_a": round(paired["entity_a"], 4),
            "entity_b": round(paired["entity_b"], 4),
        },
    }


def paired_metrics_report(
    preds_by_state: dict[str, list[PredictionRecord]],
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    reference_states: tuple[str, ...] = ("MF", "MG"),
    split: str | None = "test",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Full paired-CI report: every state vs every reference state,
    per metric, with the statistical metadata attached."""
    flags = {state: row_flags(preds, queries, associations,
                              split=split)
             for state, preds in preds_by_state.items()}
    comparisons: dict[str, dict] = {}
    for state in sorted(flags):
        for ref in reference_states:
            if ref not in flags or state == ref:
                continue
            block: dict[str, Any] = {}
            for metric in PAIRED_METRICS:
                d = paired_rate_diff_ci(
                    flags[state][metric], flags[ref][metric],
                    n_bootstrap=n_bootstrap, ci_level=ci_level,
                    seed=seed)
                if d:
                    block[metric] = d
            if block:
                comparisons.setdefault(state, {})[f"vs_{ref}"] = block
    return {
        "split": split or "pooled",
        "metrics": list(PAIRED_METRICS),
        "reference_states": list(reference_states),
        "statistical_metadata": {
            "clustering_unit": "entity_id of the asked association "
                               "(donor entity for retain_other)",
            "pairing": "intersection of query_ids present in both "
                       "states (identical probes)",
            "bootstrap": "percentile bootstrap over per-entity paired "
                         "rate differences (macro over entities)",
            "point_estimate_units": (
                "'diff'/'ci' are ENTITY-MACRO: mean over entities of the "
                "per-entity paired rate difference, which is the unit the "
                "bootstrap resamples. 'point_estimates.row_*' are ROW-MICRO "
                "over the paired rows and equal the published "
                "hierarchy_metrics rate for that state. The two differ "
                "because entities contribute unequal numbers of probes; "
                "subtracting two hierarchy_metrics rates yields row_diff, "
                "NOT diff."),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
