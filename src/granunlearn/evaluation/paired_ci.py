"""Paired entity-clustered bootstrap CIs for MLLMU metrics (Iteration 11).

The pilot-100 final evaluation must report PAIRED confidence intervals
for the four headline rates — target-granularity accuracy (TGA),
wrong-branch rate, same-entity-retain accuracy, and other-entity-retain
accuracy — comparing two checkpoints over the SAME queries.

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

#: The four headline paired metrics (user-facing names).
PAIRED_METRICS = ("tga", "wrong_branch", "retain_same",
                  "retain_other")

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

    * ``tga`` / ``wrong_branch`` — target probes (post-unlearning
      failure taxonomy; adversarial probes excluded by default);
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
    return out


def _paired_unit_diffs(
    fa: dict[str, tuple[int, str]],
    fb: dict[str, tuple[int, str]],
) -> tuple[list[str], list[float], int]:
    """(entity keys, per-entity paired rate diffs, paired row count)."""
    common = sorted(set(fa) & set(fb))
    if not common:
        return [], [], 0
    by_unit: dict[str, list[tuple[int, int]]] = {}
    for qid in common:
        va, ea = fa[qid]
        vb, eb = fb[qid]
        if ea != eb:  # defensive: pairing must agree on the cluster
            continue
        by_unit.setdefault(ea, []).append((va, vb))
    keys = sorted(by_unit)
    diffs = [sum(a - b for a, b in rows) / len(rows)
             for k in keys for rows in [by_unit[k]]]
    return keys, diffs, len(common)


def paired_rate_diff_ci(
    fa: dict[str, tuple[int, str]],
    fb: dict[str, tuple[int, str]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Paired entity-clustered percentile CI of rate(a) - rate(b)."""
    import numpy as np

    from granunlearn.salmu.embedding_metrics import validate_ci_params
    validate_ci_params(n_bootstrap, ci_level)
    keys, diffs, n_rows = _paired_unit_diffs(fa, fb)
    if not keys:
        return None
    arr = np.asarray(diffs, dtype=np.float64)
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
        "num_rows": n_rows,
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
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "seed": seed,
        },
        "comparisons": comparisons,
    }
