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

import math
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

#: Direction of each claim's ONE-SIDED alternative, for the paired difference
#: theta = rate(state under test) - rate(reference state).
#
#: Declared here rather than left to ``abs(theta)`` in the sizing code, which
#: is direction-agnostic: a power table can size |theta| and be right, but a
#: hypothesis test cannot, and Iteration 11C-R2 found the repository declaring
#: Holm thresholds over "one-sided p-values" with no direction and no
#: p-value procedure anywhere.  TGA is an accuracy, so better is HIGHER;
#: FILR is a leakage rate, so better is LOWER.  The two primary claims
#: therefore point in OPPOSITE directions on the same B3-B0 difference, and a
#: single sign convention applied to both would test one of them backwards.
CLAIM_DIRECTION = {"tga": "greater", "filr": "less"}

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


# ── the primary hypothesis test (Iteration 11C-R2) ─────────────────

def one_sided_permutation_pvalue(
    diffs: list[float] | tuple[float, ...],
    direction: str = "greater",
    n_permutations: int = 10000,
    seed: int = 20260908,
) -> dict[str, Any] | None:
    """One-sided p-value for the entity-macro paired difference by cluster
    sign flips.

    ``diffs`` is one paired difference PER ENTITY - exactly the list
    :func:`_paired_unit_diffs` returns and :func:`paired_rate_diff_ci`
    resamples - so the p-value and the published interval are built for the
    SAME statistic.  Using a different quantity for each is how one number
    comes to serve two estimands.

    Statistic
        ``T = mean(diffs)``, the entity-macro paired difference.

    Null
        Each entity's paired difference is symmetric about zero: the two
        states differ by no systematic amount, so flipping the sign of any
        entity's difference leaves its distribution unchanged and all ``2^k``
        sign-flip vectors are equally likely.  ``k`` is far too large to
        enumerate (2^72 for the primary family), so ``n_permutations`` are
        drawn.

    Why sign flips rather than the bootstrap
        A percentile bootstrap p-value is the fraction of resamples on the
        wrong side of zero, which reuses machinery already bound - but these
        paired differences are bounded, discrete and MOSTLY EXACTLY ZERO, and
        a bootstrap p-value is poorly calibrated in exactly the tail a Holm
        threshold lives in.  Sign flips assume no distribution at all, and
        handle a zero difference correctly by construction: it cannot flip,
        so it contributes nothing to either tail.  ``num_flippable_clusters``
        is reported because the zeros lower the resolution the test actually
        has below what ``k`` suggests.

    The observed vector is included in the null sample, so the smallest
    reportable p-value is ``1 / (n_permutations + 1)`` and an exact zero is
    never returned: reporting ``p = 0.0`` would claim a resolution the draw
    count does not have.

    ``seed`` defaults to a value distinct from the CI bootstrap's 42 and the
    ICC bootstrap's 20260907 on purpose.  Sharing a seed would make the two
    procedures' Monte Carlo errors dependent, and a reader could not tell
    whether an interval and a p-value agreed because the data said so or
    because they were drawn from the same stream.
    """
    import numpy as np

    if direction not in ("greater", "less"):
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction!r}; a "
            f"two-sided alternative is not what the preregistration declares")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}")

    arr = np.asarray(list(diffs), dtype=np.float64)
    k = int(arr.size)
    if k == 0:
        return None
    observed = float(arr.mean())
    flippable = int((arr != 0).sum())

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, k)) * 2.0 - 1.0
    null = np.concatenate(([observed], (signs * arr).mean(axis=1)))
    tail = null >= observed if direction == "greater" else null <= observed
    p = float(tail.mean())
    n_null = int(null.size)

    return {
        "statistic": round(observed, 6),
        "statistic_is": (
            "mean over entities of the per-entity paired rate difference - "
            "the same quantity paired_rate_diff_ci reports as 'diff' and "
            "covers with its interval"),
        "direction": direction,
        "alternative": f"theta {'>' if direction == 'greater' else '<'} 0",
        "null_hypothesis": (
            "each entity's paired difference is symmetric about zero, so all "
            f"2^{k} sign-flip vectors are equally likely and the mean of a "
            "random one is a draw from the null distribution of the "
            "statistic"),
        "method": "Monte Carlo cluster sign-flip permutation",
        "observed_vector_included_in_the_null": True,
        "num_clusters": k,
        "num_flippable_clusters": flippable,
        "zero_difference_fraction": round(float((arr == 0).mean()), 4),
        "n_permutations": n_permutations,
        "seed": seed,
        "p_value_one_sided": round(p, 6),
        "p_smallest_reportable": round(1.0 / (n_permutations + 1), 6),
        "monte_carlo_se_at_p": round(math.sqrt(p * (1.0 - p) / n_null), 6),
        "all_differences_zero": flippable == 0,
        "why_that_matters": (
            "an entity whose paired difference is exactly zero cannot be "
            "flipped, so the resolution the test has is set by "
            f"num_flippable_clusters ({flippable}) and not by num_clusters "
            f"({k}); when every difference is zero no sign flip can move the "
            "statistic and the p-value is 1 by construction"),
    }


def holm_family(
    p_values: dict[str, float],
    familywise_alpha: float,
) -> dict[str, Any]:
    """Holm's step-down over one-sided p-values, with the rule written down.

    Ordering
        Ascending p-value.  TIES are broken by claim NAME rather than by
        insertion order, so the result is reproducible from the p-values
        alone.  The break cannot change a verdict - tied p-values meet the
        same threshold at the same step - but it does change which claim is
        reported first, and a report that reorders itself between runs reads
        as a different result.

    Pass/fail rule
        Step down from the smallest p-value.  Claim at step ``i`` (0-based)
        is rejected when ``p_(i) <= familywise_alpha / (k - i)``, so the
        first threshold is ``alpha/k`` and the last is ``alpha``.  The FIRST
        non-rejection ENDS the procedure: every later claim is retained
        whether or not its own p-value would have cleared its own threshold.
        That stopping rule is what makes the procedure familywise-valid;
        dropping it turns Holm into a per-comparison test at a smaller alpha
        and understates the familywise error rate.
    """
    k = len(p_values)
    if k == 0:
        return {"familywise_alpha": familywise_alpha, "k": 0, "steps": [],
                "rejected": [], "retained": [], "all_rejected": False}
    if not 0 < familywise_alpha < 1:
        raise ValueError(
            f"familywise_alpha must lie in (0, 1), got {familywise_alpha}")

    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    steps: list[dict[str, Any]] = []
    stopped = False
    for i, (name, p) in enumerate(ordered):
        threshold = familywise_alpha / (k - i)
        clears = bool(p <= threshold)
        rejected = bool(clears and not stopped)
        if not rejected:
            stopped = True
        steps.append({
            "step": i + 1,
            "claim": name,
            "p_value_one_sided": round(float(p), 6),
            "threshold": round(threshold, 6),
            "clears_its_own_threshold": clears,
            "rejected": rejected,
            "why": (
                "p <= alpha/(k - i) and no earlier step failed"
                if rejected else
                f"p > alpha/(k - i) = {round(threshold, 6)}, and Holm stops "
                "here"
                if not clears else
                "an earlier step already failed, so Holm retains this claim "
                "even though its own p-value clears its own threshold"),
        })
    return {
        "familywise_alpha": familywise_alpha,
        "thresholds_apply_to": "one-sided p-values",
        "k": k,
        "thresholds": [round(familywise_alpha / (k - i), 6)
                       for i in range(k)],
        "ordering": "ascending p-value, ties broken by claim name",
        "tie_handling": (
            "ties are ordered by claim name so the procedure is reproducible "
            "from the p-values alone; the break cannot change a verdict, "
            "because tied p-values meet the same threshold at the same step"),
        "stopping_rule": (
            "the first non-rejection ends the procedure; every later claim is "
            "retained whether or not it clears its own threshold"),
        "steps": steps,
        "rejected": [s["claim"] for s in steps if s["rejected"]],
        "retained": [s["claim"] for s in steps if not s["rejected"]],
        "all_rejected": not stopped,
    }
