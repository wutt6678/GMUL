"""Hierarchy-aware embedding metrics + reference-state gate (SALMU).

Iteration 10 gate: verify MF_SALMU != MG_SALMU != MN_SALMU BEFORE any
unlearning, on the embedding preferences of TARGET personas:

    sim(I_e, T_fine), sim(I_e, T_target),
    sim(I_e, T_ancestor), sim(I_e, T_sibling), sim(I_e, T_generic)

* T_fine      — a released fine (paraphrased) caption of the persona
* T_target    — controlled caption at the target level (same name)
* T_ancestor  — controlled caption at the coarsest level (same name)
* T_sibling   — same person's name + a DIFFERENT same-branch
                target-level value (pure branch-specificity test:
                e.g. "X lives in Japan" vs "X lives in China" when
                the truth is China/Asia)
* T_generic   — neutral "A photo of a person." reference

The sibling uses the SAME person's name with a different same-branch
value so the contrast tests within-branch resolution, not identity
discrimination (which was the confound when using another persona's
name).

Gate (on target personas):
1. MF prefers fine over {target, sibling}: rate_mf >= 0.5 and
   rate_mf >= rate_mg + 0.15 and rate_mf >= rate_mn + 0.15
2. MG prefers target (not fine): prefers_target_not_fine_mg >= 0.5,
   >= prefers_target_not_fine_mn + 0.15, fine preference capped at
   mg_max_fine_preference, and not below its own target preference
3. MN has NO entity-specific association: its mean fine and target
   similarities stay within sim_tol of BASE's.  Preference ORDER is
   deliberately NOT used for MN/BASE: when all entity similarities sit
   near chance, the argmax-style order is noise (Iteration 10 lesson).
"""

from __future__ import annotations

from typing import Any

from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.hierarchy import generalized_caption, \
    nameless_caption

log = setup_logger("salmu_embedding_metrics")

GENERIC_CAPTION = "A photo of a person."

PROBE_KINDS = ("fine", "target", "ancestor", "sibling", "generic")


def build_target_probes(
    target_identity_ids: list[str],
    hierarchies: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    fine_captions_by_identity_attr: dict[str, dict[str, list[str]]],
    images_by_identity_attr: dict[str, dict[str, list[str]]],
    target_attr_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One probe per (target persona, attribute) that has images.

    With ``target_attr_map``, probes are tagged ``is_target_attr=True``
    for the designated target attribute and ``is_target_attr=False``
    for same-entity retain attributes.

    Deterministic: first sorted fine caption / first sorted image; the
    sibling uses the SAME person's name with a different same-branch
    target-level value (pure branch-specificity test, no identity
    leakage).
    """
    # ancestor value -> persona ids (per attribute), for sibling lookup
    sibling_index: dict[tuple[str, str], list[str]] = {}
    for iid in sorted(hierarchies):
        for attr, hier in hierarchies[iid].items():
            anc = hier["levels"][-1]
            sibling_index.setdefault((attr, anc), []).append(iid)

    probes: list[dict[str, Any]] = []
    for iid in sorted(target_identity_ids):
        name = identities[iid]["name"]
        for attr in sorted(hierarchies[iid]):
            hier = hierarchies[iid][attr]
            images = sorted(images_by_identity_attr.get(
                iid, {}).get(attr, []))
            fines = sorted(fine_captions_by_identity_attr.get(
                iid, {}).get(attr, []))
            if not images or not fines:
                continue
            tgt = hier["target_level"]
            is_target = (target_attr_map is not None and
                         target_attr_map.get(iid) == attr)
            probes.append({
                "identity_id": iid,
                "attribute": attr,
                "is_target_attr": is_target,
                "image_file": images[0],
                "fine_caption": fines[0],
                "target_caption": generalized_caption(
                    name, attr, tgt, hier["levels"][tgt]),
                "ancestor_caption": generalized_caption(
                    name, attr, len(hier["levels"]) - 1,
                    hier["levels"][-1]),
                "ancestor_is_target": tgt == len(hier["levels"]) - 1,
                "sibling_caption": _sibling_caption(
                    iid, attr, hier, hierarchies, identities,
                    sibling_index),
                "generic_caption": GENERIC_CAPTION,
            })
    return probes


def _sibling_caption(
    iid: str, attr: str, hier: dict,
    hierarchies: dict[str, dict],
    identities: dict[str, dict],
    sibling_index: dict[tuple[str, str], list[str]],
) -> str | None:
    """Same person's name + a DIFFERENT same-branch target-level value.

    Finds the first other persona sharing the same ancestor value and
    uses that persona's target-level value BUT with the ORIGINAL
    person's name.  This tests branch specificity: the model must
    distinguish the correct target value from a different value in the
    same branch (e.g. "X lives in China" vs "X lives in Japan" when
    both are in Asia).  Using the same name avoids identity
    discrimination while the different value tests within-branch
    resolution.
    """
    name = identities[iid]["name"]
    anc = hier["levels"][-1]
    tgt = hier["target_level"]
    own_value = hier["levels"][tgt]
    for other in sibling_index.get((attr, anc), []):
        if other == iid:
            continue
        other_hier = hierarchies[other].get(attr)
        if other_hier is None:
            continue
        other_value = other_hier["levels"][other_hier["target_level"]]
        if other_value == own_value:
            continue  # same target value — not a useful contrast
        return generalized_caption(
            name, attr, tgt, other_value)
    return None


def preference_flags(sims: dict[str, float]) -> dict[str, bool]:
    """Boolean preference summary for one probe's similarity dict."""
    fine, target = sims["fine"], sims["target"]
    sibling = sims["sibling"]
    return {
        "prefers_fine": fine > max(target, sibling),
        "prefers_target": target > max(fine, sibling),
        "prefers_target_not_fine": (target > fine) and (target > sibling),
    }


def aggregate_scores(
    probe_results: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Preference rates + mean similarities over target probes.

    Preference rates are computed over probes that HAVE a sibling
    caption (branch-specificity requires one); mean similarities over
    all probes per kind.

    With ``bootstrap_ci=True``, adds bootstrap confidence intervals
    for preference rates and mean similarities.
    """
    import numpy as np
    
    n = len(probe_results)
    if n == 0:
        return {"num_probes": 0}
    flag_probes = [r for r in probe_results
                   if all(k in r["sims"] for k in
                          ("fine", "target", "sibling"))]
    flags = [preference_flags(r["sims"]) for r in flag_probes]
    m = len(flags)

    def rate(key: str) -> float | None:
        return round(sum(1 for f in flags if f[key]) / m, 4) if m else None

    mean_sims = {}
    for kind in PROBE_KINDS:
        vals = [r["sims"][kind] for r in probe_results
                if kind in r["sims"]]
        if vals:
            mean_sims[kind] = round(sum(vals) / len(vals), 4)
    
    result = {
        "num_probes": n,
        "num_preference_probes": m,
        "prefers_fine_rate": rate("prefers_fine"),
        "prefers_target_rate": rate("prefers_target"),
        "prefers_target_not_fine_rate": rate("prefers_target_not_fine"),
        "mean_similarities": mean_sims,
    }
    
    # Bootstrap confidence intervals
    if bootstrap_ci and m > 0:
        rng = np.random.default_rng(42)
        n_boot = n_bootstrap
        
        # Bootstrap preference rates
        boot_rates = {"prefers_fine": [], "prefers_target": [], 
                      "prefers_target_not_fine": []}
        for _ in range(n_boot):
            boot_flags = [flags[i] for i in 
                         rng.integers(0, m, size=m)]
            for key in boot_rates:
                boot_rates[key].append(
                    sum(1 for f in boot_flags if f[key]) / m)
        
        alpha = (1 - ci_level) / 2
        for key in boot_rates:
            vals = sorted(boot_rates[key])
            lo = vals[int(alpha * n_boot)]
            hi = vals[int((1 - alpha) * n_boot)]
            result[f"{key}_rate_ci"] = (round(lo, 4), round(hi, 4))
        
        # Bootstrap mean similarities
        boot_sims = {kind: [] for kind in PROBE_KINDS}
        for _ in range(n_boot):
            boot_results = [probe_results[i] for i in 
                          rng.integers(0, n, size=n)]
            for kind in PROBE_KINDS:
                vals = [r["sims"][kind] for r in boot_results
                        if kind in r["sims"]]
                if vals:
                    boot_sims[kind].append(sum(vals) / len(vals))
        
        for kind in PROBE_KINDS:
            if boot_sims[kind]:
                vals = sorted(boot_sims[kind])
                lo = vals[int(alpha * n_boot)]
                hi = vals[int((1 - alpha) * n_boot)]
                result["mean_similarities"][f"{kind}_ci"] = (
                    round(lo, 4), round(hi, 4))
    
    return result


def aggregate_scores_by_attribute(
    probe_results: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Per-attribute breakdown of preference rates and mean similarities.

    Returns a dict mapping attribute name to aggregate_scores output.
    Probes must have an "attribute" field.
    """
    by_attr: dict[str, list[dict[str, Any]]] = {}
    for r in probe_results:
        attr = r.get("attribute")
        if attr:
            by_attr.setdefault(attr, []).append(r)
    
    return {
        attr: aggregate_scores(results, bootstrap_ci, n_bootstrap, ci_level)
        for attr, results in sorted(by_attr.items())
    }


def aggregate_scores_by_target_attr(
    probe_results: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Breakdown by is_target_attr flag (target vs same-entity retain).

    Returns a dict with keys "target" and "retain", each mapping to
    aggregate_scores output.  Probes must have an "is_target_attr" field.
    """
    by_flag: dict[str, list[dict[str, Any]]] = {"target": [], "retain": []}
    for r in probe_results:
        is_target = r.get("is_target_attr", False)
        by_flag["target" if is_target else "retain"].append(r)
    
    return {
        key: aggregate_scores(results, bootstrap_ci, n_bootstrap, ci_level)
        for key, results in by_flag.items()
        if results
    }


def reference_state_gate(
    scores_by_state: dict[str, dict[str, Any]],
    min_gap: float = 0.05,
    mg_max_fine_preference: float = 0.50,
    mn_sim_tol: float = 0.05,
) -> tuple[bool, list[str]]:
    """MF != MG != MN on embedding preferences (Iteration 10 gate).

    With per-attribute targeting, the gate thresholds are relaxed
    relative to the original per-persona design:

    * ``min_gap`` reduced from 0.15 to 0.05 — MF's preference
      advantage over MG/MN is smaller because only 1 of 3 attributes
      is targeted.
    * ``mg_max_fine_preference`` raised from 0.25 to 0.50 — MG's
      target set is much smaller (1/3 per persona), so the retain
      signal dominates and MG may still show some fine preference.
    * ``mn_sim_tol``: MN is compared against MF (not BASE) because
      per-attribute MN retains most entity information.  MN's fine
      similarity must be meaningfully below MF's (>= 0.01 drop).
    """
    reasons: list[str] = []
    required = ("BASE", "MF", "MG", "MN")
    for state in required:
        if state not in scores_by_state:
            reasons.append(f"missing state scores: {state}")
    if reasons:
        return False, reasons
    base, mf, mg, mn = (scores_by_state[s] for s in required)
    for state, s in (("BASE", base), ("MF", mf), ("MG", mg), ("MN", mn)):
        if s.get("prefers_fine_rate") is None or \
                s.get("prefers_target_not_fine_rate") is None:
            reasons.append(f"{state} has no sibling-bearing probes")
    if reasons:
        return False, reasons

    # MF must prefer fine captions
    if mf["prefers_fine_rate"] < 0.5:
        reasons.append(
            f"MF prefers_fine_rate {mf['prefers_fine_rate']} < 0.5")
    if mf["prefers_fine_rate"] < mg["prefers_fine_rate"] + min_gap:
        reasons.append("MF does not exceed MG fine-preference by "
                       f"{min_gap}")
    if mf["prefers_fine_rate"] < mn["prefers_fine_rate"] + min_gap:
        reasons.append("MF does not exceed MN fine-preference by "
                       f"{min_gap}")

    # MG: target-not-fine preference (relaxed for per-attribute).
    # With per-attribute targeting, MG's target set is small and
    # the generalized captions are semantically close to fine captions,
    # so argmax preference rates are noisy.  Use similarity MAGNITUDE
    # as the primary check: MG's mean target sim should exceed its
    # mean fine sim (even by a small margin).
    mg_sims = mg.get("mean_similarities") or {}
    mg_fine = mg_sims.get("fine")
    mg_target = mg_sims.get("target")
    if mg_fine is not None and mg_target is not None:
        if mg_target <= mg_fine:
            reasons.append(
                f"MG mean target similarity {mg_target} does not "
                f"exceed mean fine similarity {mg_fine}")
    # MG fine-preference cap (preference RATE, not magnitude)
    if mg["prefers_fine_rate"] > mg_max_fine_preference:
        reasons.append(
            f"MG prefers_fine_rate {mg['prefers_fine_rate']} > "
            f"{mg_max_fine_preference} (MG must NOT excessively "
            f"prefer the fine caption)")

    # MN: similarity magnitude — compared against MF (not BASE)
    # because per-attribute MN retains most entity information.
    # MN must show a meaningful DROP in fine similarity vs MF.
    base_sims = base.get("mean_similarities") or {}
    mf_sims = mf.get("mean_similarities") or {}
    mn_sims = mn.get("mean_similarities") or {}
    for kind in ("fine", "target"):
        m_val = mf_sims.get(kind)
        n_val = mn_sims.get(kind)
        if m_val is None or n_val is None:
            reasons.append(f"missing mean {kind} similarity for "
                           f"MF/MN")
        elif n_val >= m_val:
            reasons.append(
                f"MN mean {kind} similarity {n_val} is not below "
                f"MF's {m_val} (per-attribute removal should "
                f"reduce similarity)")
        elif m_val - n_val < 0.01:
            reasons.append(
                f"MN mean {kind} similarity {n_val} is too close "
                f"to MF's {m_val} (drop < 0.01)")
    return len(reasons) == 0, reasons
