"""Official SALMUBench utility metrics (Iteration 10R2).

Defines the canonical metric names and computation for the SALMU
(CLIP association-level unlearning) benchmark, analogous to the
SALMUBench evaluation protocol for generative MLLMs.

Metrics
-------
forget : float
    1 − prefers_fine_rate on target-attribute probes.
    Higher = more target-association forgetting.
holdout_association : float
    prefers_fine_rate on same-entity retain probes.
    Higher = better retention of non-target associations.
holdout_identity : float
    1 − mean generic similarity on all target-persona probes.
    Higher = more identity information lost.
retain_synth : float
    mean fine similarity on other-entity retain probes.
    Higher = less collateral damage to unrelated personas.
core_assoc : float
    mean target similarity on target-attribute probes.
    Higher = stronger residual target association.
intra_identity : float
    mean fine similarity on same-entity retain probes.
    Measures within-persona retention quality.
inter_identity : float
    mean fine similarity on other-entity retain probes.
    Measures between-persona retention quality.
preference_margin : float
    mean(fine_sim − target_sim) on target-attribute probes.
    Positive = fine still preferred; negative = target/generalized
    preferred.
"""

from __future__ import annotations

from typing import Any


def compute_salmubench_metrics(
    target_results: list[dict[str, Any]],
    same_entity_results: list[dict[str, Any]],
    other_entity_results: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Compute all SALMUBench utility metrics from scored probe results.

    Parameters
    ----------
    target_results : list[dict]
        Per-probe results where ``is_target_attr=True``.  Must have
        ``sims`` with at least ``fine`` and ``target`` keys.
    same_entity_results : list[dict]
        Per-probe results for same-entity retain (non-target attrs of
        target personas).
    other_entity_results : list[dict]
        Per-probe results for other-entity retain (all attrs of
        non-target personas).

    Returns
    -------
    dict mapping metric name to float (or None if the required data
    is empty).
    """
    from granunlearn.salmu.embedding_metrics import preference_flags

    # --- Target-attribute metrics ---
    n_target = len(target_results)
    target_flags = [preference_flags(r["sims"]) for r in target_results
                    if all(k in r["sims"] for k in
                           ("fine", "target", "sibling"))]
    n_flags = len(target_flags)

    prefers_fine_rate = (sum(1 for f in target_flags if f["prefers_fine"])
                         / n_flags) if n_flags else None
    forget = (1 - prefers_fine_rate) if prefers_fine_rate is not None \
        else None

    target_fine_sims = [r["sims"]["fine"] for r in target_results
                        if "fine" in r["sims"]]
    target_target_sims = [r["sims"]["target"] for r in target_results
                          if "target" in r["sims"]]
    target_generic_sims = [r["sims"]["generic"] for r in target_results
                           if "generic" in r["sims"]]

    core_assoc = (sum(target_target_sims) / len(target_target_sims)
                  if target_target_sims else None)

    # Preference margin: mean(fine - target)
    margins = []
    for r in target_results:
        if "fine" in r["sims"] and "target" in r["sims"]:
            margins.append(r["sims"]["fine"] - r["sims"]["target"])
    preference_margin = (sum(margins) / len(margins)
                         if margins else None)

    # --- Same-entity retain metrics ---
    se_flags = [preference_flags(r["sims"]) for r in same_entity_results
                if all(k in r["sims"] for k in
                       ("fine", "target", "sibling"))]
    n_se = len(se_flags)
    se_fine_rate = (sum(1 for f in se_flags if f["prefers_fine"]) / n_se
                    if n_se else None)
    holdout_association = se_fine_rate

    se_fine_sims = [r["sims"]["fine"] for r in same_entity_results
                    if "fine" in r["sims"]]
    intra_identity = (sum(se_fine_sims) / len(se_fine_sims)
                      if se_fine_sims else None)

    # --- Other-entity retain metrics ---
    oe_fine_sims = [r["sims"]["fine"] for r in other_entity_results
                    if "fine" in r["sims"]]
    retain_synth = (sum(oe_fine_sims) / len(oe_fine_sims)
                    if oe_fine_sims else None)
    inter_identity = retain_synth

    # --- Identity retention ---
    all_generic = (target_generic_sims +
                   [r["sims"].get("generic") for r in same_entity_results
                    if "generic" in r["sims"]])
    holdout_identity = (1 - sum(all_generic) / len(all_generic)
                        if all_generic else None)

    return {
        "forget": round(forget, 4) if forget is not None else None,
        "holdout_association": round(holdout_association, 4)
        if holdout_association is not None else None,
        "holdout_identity": round(holdout_identity, 4)
        if holdout_identity is not None else None,
        "retain_synth": round(retain_synth, 4)
        if retain_synth is not None else None,
        "core_assoc": round(core_assoc, 4)
        if core_assoc is not None else None,
        "intra_identity": round(intra_identity, 4)
        if intra_identity is not None else None,
        "inter_identity": round(inter_identity, 4)
        if inter_identity is not None else None,
        "preference_margin": round(preference_margin, 6)
        if preference_margin is not None else None,
        "n_target_probes": n_target,
        "n_same_entity_retain": len(same_entity_results),
        "n_other_entity_retain": len(other_entity_results),
    }
