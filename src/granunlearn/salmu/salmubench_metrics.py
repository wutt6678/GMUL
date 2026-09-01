"""GMUL proxy metrics + official SALMUBench evaluation (Iteration 10R3).

Two separate metric families:

1. **GMUL proxy metrics** (``compute_gmul_proxy_metrics``):
   In-house CLIP embedding proxies for the SALMUBench concepts.
   These are NOT the official SALMUBench protocol — they use our
   own probe construction and similarity computation.  Prefixed
   ``gmul_proxy_*`` to avoid confusion with the released metrics.

2. **Official SALMUBench metrics** (``compute_official_salmubench``):
   Loads the released evaluation data (forget gallery, holdout
   associations, retain synth, etc.) from the SALMUBench dataset
   and computes the canonical metrics: RetFail (MRR over 2,001-
   caption gallery), CoreAssoc (both name-value token orders),
   ACS (coherence classifier), IntraIdSim / InterIdSim (released
   holdout splits), and general utility (ImageNet / DataComp).
"""

from __future__ import annotations

import json
from typing import Any


# ── GMUL proxy metrics ────────────────────────────────────────────

def compute_gmul_proxy_metrics(
    target_results: list[dict[str, Any]],
    same_entity_results: list[dict[str, Any]],
    other_entity_results: list[dict[str, Any]],
) -> dict[str, float | None]:
    """GMUL in-house CLIP-embedding proxies for SALMUBench concepts.

    All metric names carry the ``gmul_proxy_`` prefix to distinguish
    them from the official SALMUBench protocol.

    10R4: every statistic is association-weighted — variants within a
    (identity_id, attribute) association are macro-averaged first, so
    associations with more images/captions get no extra weight.

    Parameters
    ----------
    target_results : list[dict]
        Per-probe results where ``is_target_attr=True``.
    same_entity_results : list[dict]
        Per-probe results for same-entity retain.
    other_entity_results : list[dict]
        Per-probe results for other-entity retain.

    Returns
    -------
    dict mapping ``gmul_proxy_*`` metric names to float (or None).
    """
    from granunlearn.salmu.embedding_metrics import (
        association_level_results,
        preference_flags,
    )

    # --- Macro-average each role to the association level ---
    target_assoc = association_level_results(target_results)
    same_entity_assoc = association_level_results(same_entity_results)
    other_entity_assoc = association_level_results(other_entity_results)

    # --- Target-attribute metrics ---
    n_target = len(target_results)
    target_flags = [preference_flags(a["sims"]) for a in target_assoc
                    if all(k in a["sims"] for k in
                           ("fine", "target", "sibling"))]
    n_flags = len(target_flags)

    prefers_fine_rate = (sum(1 for f in target_flags if f["prefers_fine"])
                         / n_flags) if n_flags else None
    gmul_proxy_forget = (1 - prefers_fine_rate) \
        if prefers_fine_rate is not None else None

    target_target_sims = [a["sims"]["target"] for a in target_assoc
                          if "target" in a["sims"]]
    gmul_proxy_core_assoc = (
        sum(target_target_sims) / len(target_target_sims)
        if target_target_sims else None)

    # Preference margin: mean(fine - target) per association
    margins = []
    for a in target_assoc:
        if "fine" in a["sims"] and "target" in a["sims"]:
            margins.append(a["sims"]["fine"] - a["sims"]["target"])
    gmul_proxy_preference_margin = (
        sum(margins) / len(margins) if margins else None)

    # --- Same-entity retain metrics ---
    se_flags = [preference_flags(a["sims"]) for a in same_entity_assoc
                if all(k in a["sims"] for k in
                       ("fine", "target", "sibling"))]
    n_se = len(se_flags)
    se_fine_rate = (sum(1 for f in se_flags if f["prefers_fine"]) / n_se
                    if n_se else None)
    gmul_proxy_holdout_association = se_fine_rate

    se_fine_sims = [a["sims"]["fine"] for a in same_entity_assoc
                    if "fine" in a["sims"]]
    gmul_proxy_intra_identity = (
        sum(se_fine_sims) / len(se_fine_sims) if se_fine_sims else None)

    # --- Other-entity retain metrics ---
    oe_fine_sims = [a["sims"]["fine"] for a in other_entity_assoc
                    if "fine" in a["sims"]]
    gmul_proxy_retain_synth = (
        sum(oe_fine_sims) / len(oe_fine_sims) if oe_fine_sims else None)
    gmul_proxy_inter_identity = gmul_proxy_retain_synth

    # --- Identity retention ---
    target_generic_sims = [a["sims"]["generic"] for a in target_assoc
                           if "generic" in a["sims"]]
    all_generic = (target_generic_sims +
                   [a["sims"]["generic"] for a in same_entity_assoc
                    if "generic" in a["sims"]])
    gmul_proxy_holdout_identity = (
        1 - sum(all_generic) / len(all_generic)
        if all_generic else None)

    def _r(v: float | None, digits: int = 4) -> float | None:
        return round(v, digits) if v is not None else None

    return {
        "gmul_proxy_forget": _r(gmul_proxy_forget),
        "gmul_proxy_holdout_association": _r(
            gmul_proxy_holdout_association),
        "gmul_proxy_holdout_identity": _r(
            gmul_proxy_holdout_identity),
        "gmul_proxy_retain_synth": _r(gmul_proxy_retain_synth),
        "gmul_proxy_core_assoc": _r(gmul_proxy_core_assoc),
        "gmul_proxy_intra_identity": _r(gmul_proxy_intra_identity),
        "gmul_proxy_inter_identity": _r(gmul_proxy_inter_identity),
        "gmul_proxy_preference_margin": _r(
            gmul_proxy_preference_margin, 6),
        "weighting": "association_macro_average",
        "n_target_probes": n_target,
        "n_target_associations": len(target_assoc),
        "n_same_entity_retain": len(same_entity_results),
        "n_same_entity_associations": len(same_entity_assoc),
        "n_other_entity_retain": len(other_entity_results),
        "n_other_entity_associations": len(other_entity_assoc),
    }


# Backward-compatible alias (deprecated — use compute_gmul_proxy_metrics)
compute_salmubench_metrics = compute_gmul_proxy_metrics


# ── Official SALMUBench evaluation ────────────────────────────────

def compute_official_salmubench(
    benchmark_dir: str | Any,
    official_splits_report: str | Any | None = None,
) -> dict[str, Any]:
    """Official SALMUBench evaluation status from released data.

    Implemented (see ``scripts/evaluate_salmu_official_splits.py``,
    report ``data/reports/salmu_official_splits.json``): the
    released-split mean cosine similarities, which the paper DEFINES
    as

    * ``AssocStr``   = mean cos-sim on ``forget``
    * ``IntraIdSim`` = mean cos-sim on ``holdout_association``
    * ``InterIdSim`` = mean cos-sim on ``holdout_identity``

    (unit-macro variants + clustering-correspondent CIs included).

    NOT reimplemented (require the official SALMUBench codebase,
    github.com/cvc-mmu/salmubench): RetFail (MRR over a 2,001-caption
    gallery), ACS (coherence classifier), IdZSC, CoreAssoc, GenKnow,
    VisIdInt, FragSim.

    10R4b honesty note: the current GMUL training chain consumed
    released holdout pairs (the released training dataset is the union
    of forget + holdout splits), so the released-split numbers are
    TRANSFER DIAGNOSTICS, not an untouched external evaluation.
    Iteration 10R5 retrains holdout-clean for the latter.
    """
    from pathlib import Path
    bench = Path(benchmark_dir) if not isinstance(benchmark_dir, Path) \
        else benchmark_dir

    metrics: dict[str, Any] = {
        "AssocStr": None,
        "IntraIdSim": None,
        "InterIdSim": None,
        "RetFail": None,
        "ACS": None,
        "IdZSC": None,
        "CoreAssoc": None,
        "GenKnow": None,
        "VisIdInt": None,
        "FragSim": None,
    }
    # Fill the three implemented metrics from the released-split
    # report when it exists.
    if official_splits_report is not None:
        rep_path = Path(official_splits_report)
        if rep_path.exists():
            rep = json.loads(rep_path.read_text())
            mean_sim = {
                "AssocStr": ("forget", "mean_assoc_sim"),
                "IntraIdSim":
                    ("holdout_association", "mean_assoc_sim"),
                "InterIdSim": ("holdout_identity", "mean_assoc_sim"),
            }
            filled: dict[str, Any] = {}
            for name, (split, key) in mean_sim.items():
                vals = {
                    state: d.get(split, {}).get(key)
                    for state, d in rep.get("states", {}).items()
                }
                if any(v is not None for v in vals.values()):
                    metrics[name] = vals
                    filled[name] = True
            if filled:
                metrics["source"] = str(rep_path)

    result: dict[str, Any] = {
        "protocol_note": (
            "AssocStr/IntraIdSim/InterIdSim (mean cos-sim on "
            "forget/holdout_association/holdout_identity) ARE "
            "implemented in scripts/evaluate_salmu_official_splits.py "
            "and filled here when that report exists. RetFail / ACS / "
            "IdZSC / CoreAssoc / GenKnow / VisIdInt / FragSim require "
            "the official SALMUBench codebase and remain null."),
        "evidence_status": (
            "TRANSFER DIAGNOSTIC for the current chain: released "
            "holdout pairs were consumed by GMUL training; untouched "
            "external evaluation requires the Iteration 10R5 "
            "holdout-clean retrain."),
        "status": "released_splits_present",
        "metrics": metrics,
        "benchmark_dir": str(bench),
    }

    # Check for released evaluation splits (parquet shards)
    available = []
    for split in ("forget", "forget_target", "holdout_association",
                  "holdout_identity", "retain_synth"):
        if list((bench / "data").glob(f"{split}-*.parquet")):
            available.append(split)

    result["available_splits"] = available
    if not available:
        result["status"] = "released_splits_not_found"
    return result
