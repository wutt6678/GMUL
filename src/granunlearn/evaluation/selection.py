"""Distance-to-MG model selection (Iteration 9).

Individual metrics stay the PRIMARY scientific results; the distance is
a compact model-selection criterion only.  Selection uses TRAIN+VAL
probes exclusively — the frozen test split is a genuine final held-out
evaluation and must never influence checkpoint choice.

Summary vector (per the Iteration 9 spec):

    v(M) = [FILR, TGA, Ancestor, Retain_same, Retain_other, Over, Wrong]

    D_G(M_U) = sum_j w_j * |v_j(M_U) - v_j(M_G)|

All components live in [0, 1]; default weights are uniform.  Missing
components (empty slice -> None) are excluded and the distance is
renormalized over the used components.
"""

from __future__ import annotations

from typing import Any

from granunlearn.evaluation.hierarchy_metrics import compute_hierarchy_metrics
from granunlearn.schema import AssociationRecord, PredictionRecord, QueryRecord

SUMMARY_COMPONENTS = (
    "filr", "tga", "ancestor", "retain_same", "retain_other",
    "over", "wrong",
)


def summary_vector(hierarchy_metrics: dict[str, Any]) -> dict[str, float | None]:
    """v(M) from one state's hierarchy metrics block."""
    hm = hierarchy_metrics
    return {
        "filr": hm.get("filr"),
        "tga": hm.get("tga"),
        "ancestor": (hm.get("ancestor_retention") or {}).get(
            "post_unlearning_accuracy"),
        "retain_same": (hm.get("retain_same_entity") or {}).get(
            "baseline_accuracy"),
        "retain_other": (hm.get("retain_other_entity") or {}).get(
            "baseline_accuracy"),
        "over": (hm.get("failure_rates") or {}).get("over_forgetting"),
        "wrong": (hm.get("failure_rates") or {}).get("wrong_branch"),
    }


def distance_to_reference(
    vec: dict[str, float | None],
    ref_vec: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> tuple[float | None, list[str]]:
    """Weighted L1 distance to the reference vector (MG).

    Returns ``(distance, used_components)``; components missing on
    either side are excluded and weights renormalized.
    """
    weights = weights or {c: 1.0 for c in SUMMARY_COMPONENTS}
    used: list[str] = []
    total, wsum = 0.0, 0.0
    for comp in SUMMARY_COMPONENTS:
        a, b = vec.get(comp), ref_vec.get(comp)
        if a is None or b is None:
            continue
        total += weights[comp] * abs(a - b)
        wsum += weights[comp]
        used.append(comp)
    if not used:
        return None, []
    return round(total / wsum, 6), used


def filter_predictions_by_splits(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    splits: tuple[str, ...],
) -> list[PredictionRecord]:
    split_of = {q.query_id: q.split for q in queries}
    return [p for p in predictions if split_of.get(p.query_id) in splits]


def trainval_hierarchy_metrics(
    predictions: list[PredictionRecord],
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
) -> dict[str, Any]:
    """Hierarchy metrics pooled over TRAIN+VAL probes only — the
    selection basis.  Never computed on test."""
    tv_preds = filter_predictions_by_splits(predictions, queries,
                                            ("train", "val"))
    return compute_hierarchy_metrics(tv_preds, queries, associations,
                                     split=None)


def select_checkpoints(
    candidates: dict[str, dict[str, Any]],
    reference_vector: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Pick, per METHOD, the candidate minimizing D_G on train+val.

    ``candidates`` maps candidate_id -> {"method": str,
    "trainval_metrics": dict}.  Returns the selection report content.
    """
    rows: dict[str, Any] = {}
    for cid, info in candidates.items():
        vec = summary_vector(info["trainval_metrics"])
        dist, used = distance_to_reference(vec, reference_vector, weights)
        rows[cid] = {
            "method": info["method"],
            "vector": vec,
            "distance_to_mg": dist,
            "used_components": used,
            "config": info.get("config"),
        }
    selected: dict[str, Any] = {}
    methods = sorted({r["method"] for r in rows.values()})
    for method in methods:
        best = None
        for cid, r in rows.items():
            if r["method"] != method or r["distance_to_mg"] is None:
                continue
            if best is None or r["distance_to_mg"] < \
                    rows[best]["distance_to_mg"]:
                best = cid
        selected[method] = best
    return {
        "reference": {"state": "MG", "vector": reference_vector},
        "basis": "train+val probes only (test split held out)",
        "weights": weights or {c: 1.0 for c in SUMMARY_COMPONENTS},
        "candidates": rows,
        "selected": selected,
        "note": (
            "Distance is a model-selection criterion only; individual "
            "metrics remain the primary scientific results. Test-split "
            "numbers are computed AFTER selection and never fed back."
        ),
    }
