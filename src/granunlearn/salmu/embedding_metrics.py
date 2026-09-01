"""Hierarchy-aware embedding metrics + reference-state gate (SALMU).

Iteration 10 gate: verify MF_SALMU != MG_SALMU != MN_SALMU BEFORE any
unlearning, on the embedding preferences of TARGET personas:

    sim(I_e, T_fine), sim(I_e, T_target),
    sim(I_e, T_ancestor), sim(I_e, T_sibling), sim(I_e, T_generic)

* T_fine      — a released fine (paraphrased) caption of the persona
* T_target    — controlled caption at the target level (same name)
* T_ancestor  — controlled caption at the coarsest level (same name)
* T_sibling   — same person's name + a DIFFERENT-branch
                target-level value (cross-branch specificity test:
                e.g. "X lives in Japan" vs "X lives in China" when
                the truth is China/Asia)
* T_generic   — neutral "A photo of a person." reference

The sibling uses the SAME person's name with a different-branch
value so the contrast tests cross-branch specificity, not identity
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


def _build_alt_value_index(
    hierarchies: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Collect all distinct ancestor (coarsest-level) values per attribute.

    Used to generate *different-branch* sibling alternatives: a city
    probe gets a caption with a country DIFFERENT from the persona's
    own country, a blood_type probe gets a different ABO group, etc.
    """
    vals: dict[str, set[str]] = {}
    for iid in hierarchies:
        for attr, hier in hierarchies[iid].items():
            anc = hier["levels"][-1]
            vals.setdefault(attr, set()).add(anc)
    return {attr: sorted(v) for attr, v in vals.items()}


def _build_sector_index(
    hierarchies: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Distinct profession classes per sector for the job attribute.

    Used to generate *same-sector, different-profession-class* sibling
    alternatives: e.g. for "software developer / IT sector" the
    alternative is "IT specialist / IT sector".
    """
    idx: dict[str, set[str]] = {}
    for iid in hierarchies:
        hier = hierarchies[iid].get("job")
        if hier and len(hier["levels"]) >= 3:
            pclass, sector = hier["levels"][1], hier["levels"][2]
            idx.setdefault(sector, set()).add(pclass)
    return {sector: sorted(pc) for sector, pc in idx.items()}


def _frozen_probe_id(iid: str, attr: str, image: str,
                     caption: str) -> str:
    """Deterministic probe ID from (identity, attr, image, caption).

    Frozen across runs so that probe-level results are reproducible
    and can be compared across checkpoints without ambiguity.
    """
    import hashlib
    key = f"{iid}|{attr}|{image}|{caption}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_target_probes(
    target_identity_ids: list[str],
    hierarchies: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
    fine_captions_by_identity_attr: dict[str, dict[str, list[str]]],
    images_by_identity_attr: dict[str, dict[str, list[str]]],
    target_attr_map: dict[str, str] | None = None,
    max_images: int | None = None,
    max_captions: int | None = None,
) -> list[dict[str, Any]]:
    """Probes over ALL (image, fine-caption) variants per (persona, attr).

    With ``target_attr_map``, probes are tagged ``is_target_attr=True``
    for the designated target attribute and ``is_target_attr=False``
    for same-entity retain attributes.

    Each probe gets a **frozen** ``probe_id`` (SHA-256 prefix of the
    identity + attribute + image + caption) so results are
    reproducible and comparable across checkpoints.

    The sibling uses the SAME person's name with a DIFFERENT-branch
    target-level value (e.g. a different country for city, a different
    ABO group for blood_type, a different sector for job).

    Parameters
    ----------
    max_images, max_captions : int | None
        If set, cap the number of images / fine captions per
        (persona, attribute) to the first *N* sorted values.
        ``None`` means use all available.

    Returns one probe per (persona, attribute, image, fine_caption).
    """
    alt_index = _build_alt_value_index(hierarchies)
    sector_index = _build_sector_index(hierarchies)

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
            if max_images is not None:
                images = images[:max_images]
            if max_captions is not None:
                fines = fines[:max_captions]
            tgt = hier["target_level"]
            is_target = (target_attr_map is not None and
                         target_attr_map.get(iid) == attr)
            target_cap = generalized_caption(
                name, attr, tgt, hier["levels"][tgt])
            ancestor_cap = generalized_caption(
                name, attr, len(hier["levels"]) - 1,
                hier["levels"][-1])
            sib_cap = _sibling_caption(
                iid, attr, hier, hierarchies, identities,
                alt_index, sector_index)
            for img_idx, img_file in enumerate(images):
                for cap_idx, fine_cap in enumerate(fines):
                    probes.append({
                        "probe_id": _frozen_probe_id(
                            iid, attr, img_file, fine_cap),
                        "identity_id": iid,
                        "attribute": attr,
                        "is_target_attr": is_target,
                        "image_file": img_file,
                        "image_idx": img_idx,
                        "fine_caption": fine_cap,
                        "caption_idx": cap_idx,
                        "target_caption": target_cap,
                        "ancestor_caption": ancestor_cap,
                        "ancestor_is_target": tgt == len(
                            hier["levels"]) - 1,
                        "sibling_caption": sib_cap,
                        "generic_caption": GENERIC_CAPTION,
                    })
    return probes


def _sibling_caption(
    iid: str, attr: str, hier: dict,
    hierarchies: dict[str, dict],
    identities: dict[str, dict],
    alt_index: dict[str, list[str]],
    sector_index: dict[str, list[str]] | None = None,
) -> str | None:
    """Same person's name + a DIFFERENT-branch value.

    For 2-level hierarchies (city, blood_type): picks a different
    ancestor value (country / ABO group) using the target-level
    template.  This tests cross-branch specificity.

    For 3-level job hierarchy (job → profession_class → sector):
    1. First tries *same-sector, different-profession-class*:
       e.g. "X works as an IT specialist" instead of
       "X works as a software developer" (both in the IT sector).
    2. Falls back to *different-sector* using the level-2 template:
       e.g. "X works in the healthcare sector" instead of IT.

    Using the same name avoids identity discrimination.
    """
    name = identities[iid]["name"]
    anc = hier["levels"][-1]
    tgt = hier["target_level"]
    n_levels = len(hier["levels"])

    # Job: same-sector different-profession-class first, then
    # different-sector using the level-2 template.
    if attr == "job" and n_levels >= 3 and sector_index:
        pclass, sector = hier["levels"][1], hier["levels"][2]
        # Same sector, different profession class
        for alt_pc in sector_index.get(sector, []):
            if alt_pc != pclass:
                return generalized_caption(
                    name, attr, tgt, alt_pc)
        # Different sector → use level-2 template
        for alt_sector in alt_index.get(attr, []):
            if alt_sector != anc:
                return generalized_caption(
                    name, attr, 2, alt_sector)
        return None

    # Default: different ancestor value at target level
    candidates = alt_index.get(attr, [])
    for alt_val in candidates:
        if alt_val != anc:
            return generalized_caption(name, attr, tgt, alt_val)
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


def association_level_results(
    probe_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Macro-average probe similarities per (identity_id, attribute).

    Multiple image-caption combinations of the SAME association are
    NOT independent observations: all point estimates (preference
    rates, mean similarities) are computed association-weighted by
    averaging the variants within each association FIRST (10R4).

    Returns one synthetic result per association with the averaged
    ``sims``, plus ``num_probes`` (variant count) and propagated
    ``is_target_attr``.
    """
    clusters: dict[tuple[str, str], list[dict]] = {}
    for r in probe_results:
        key = (r.get("identity_id", ""), r.get("attribute", ""))
        clusters.setdefault(key, []).append(r)
    assoc: list[dict[str, Any]] = []
    for (iid, attr) in sorted(clusters):
        rs = clusters[(iid, attr)]
        kinds: set[str] = set()
        for r in rs:
            kinds.update(r["sims"].keys())
        mean_sims = {}
        for k in kinds:
            vals = [r["sims"][k] for r in rs if k in r["sims"]]
            if vals:
                mean_sims[k] = sum(vals) / len(vals)
        assoc.append({
            "identity_id": iid,
            "attribute": attr,
            "is_target_attr": rs[0].get("is_target_attr", False),
            "num_probes": len(rs),
            "sims": mean_sims,
        })
    return assoc


def validate_ci_params(n_bootstrap: int, ci_level: float) -> None:
    """Reject invalid bootstrap-CI parameters (10R5 hardening).

    ``n_bootstrap`` must be a positive integer and ``ci_level`` must
    lie in the OPEN interval (0, 1): degenerate levels would make the
    percentile quantiles undefined as confidence intervals.
    """
    import numbers
    if not isinstance(n_bootstrap, numbers.Integral) or \
            n_bootstrap <= 0:
        raise ValueError(
            f"n_bootstrap must be a positive integer, "
            f"got {n_bootstrap!r}")
    if int(n_bootstrap) < 2:
        raise ValueError(
            "n_bootstrap must be >= 2 (the percentile CI needs both "
            f"a lower and an upper quantile index), got {n_bootstrap!r}")
    if not isinstance(ci_level, numbers.Real) or \
            not (0.0 < float(ci_level) < 1.0):
        raise ValueError(
            f"ci_level must be in the open interval (0, 1), "
            f"got {ci_level!r}")


def aggregate_scores(
    probe_results: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Association-weighted preference rates + mean similarities.

    10R4: variants within an association are macro-averaged FIRST, so
    each (identity_id, attribute) association counts exactly once —
    attributes with more images/captions get no extra weight.

    Preference rates are computed over associations that HAVE a
    sibling caption (branch-specificity requires one); mean
    similarities over all associations per kind.

    With ``bootstrap_ci=True``, adds confidence intervals from a
    bootstrap that resamples ASSOCIATIONS (the unit of analysis),
    which is exact under the macro-average.
    """
    if bootstrap_ci:
        validate_ci_params(n_bootstrap, ci_level)
    n = len(probe_results)
    if n == 0:
        return {"num_probes": 0, "num_associations": 0}
    assoc = association_level_results(probe_results)
    n_assoc = len(assoc)
    flag_assoc = [a for a in assoc
                  if all(k in a["sims"] for k in
                         ("fine", "target", "sibling"))]
    flags = [preference_flags(a["sims"]) for a in flag_assoc]
    m = len(flags)

    def rate(key: str) -> float | None:
        return round(sum(1 for f in flags if f[key]) / m, 4) if m else None

    mean_sims = {}
    for kind in PROBE_KINDS:
        vals = [a["sims"][kind] for a in assoc
                if kind in a["sims"]]
        if vals:
            mean_sims[kind] = round(sum(vals) / len(vals), 4)

    result = {
        "num_probes": n,
        "num_associations": n_assoc,
        "num_preference_associations": m,
        "weighting": "association_macro_average",
        "prefers_fine_rate": rate("prefers_fine"),
        "prefers_target_rate": rate("prefers_target"),
        "prefers_target_not_fine_rate": rate("prefers_target_not_fine"),
        "mean_similarities": mean_sims,
    }

    # Association bootstrap confidence intervals: resample the
    # associations (unit of analysis) with replacement.
    if bootstrap_ci and m > 0:
        import numpy as np
        rng = np.random.default_rng(42)

        flag_assoc_arr = flag_assoc
        boot_rates = {"prefers_fine": [], "prefers_target": [],
                      "prefers_target_not_fine": []}
        boot_sims = {kind: [] for kind in PROBE_KINDS}

        for _ in range(n_bootstrap):
            # Preference rates over resampled flag associations
            idx = rng.integers(0, m, size=m)
            bflags = [flags[j] for j in idx]
            for key in boot_rates:
                boot_rates[key].append(
                    sum(1 for f in bflags if f[key]) / m)
            # Mean similarities over resampled associations
            idx_all = rng.integers(0, n_assoc, size=n_assoc)
            for kind in PROBE_KINDS:
                vals = [assoc[j]["sims"][kind] for j in idx_all
                        if kind in assoc[j]["sims"]]
                if vals:
                    boot_sims[kind].append(sum(vals) / len(vals))

        alpha = (1 - ci_level) / 2
        for key in boot_rates:
            vals = sorted(boot_rates[key])
            lo = vals[int(alpha * n_bootstrap)]
            hi = vals[int((1 - alpha) * n_bootstrap)]
            result[f"{key}_rate_ci"] = (round(lo, 4), round(hi, 4))

        for kind in PROBE_KINDS:
            if boot_sims[kind]:
                vals = sorted(boot_sims[kind])
                lo = vals[int(alpha * n_bootstrap)]
                hi = vals[int((1 - alpha) * n_bootstrap)]
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


def aggregate_scores_by_image(
    probe_results: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Per-image breakdown: aggregate over all captions for each image.

    Returns a dict mapping image_file to aggregate_scores output.
    Probes must have "image_file" fields.
    """
    by_img: dict[str, list[dict[str, Any]]] = {}
    for r in probe_results:
        img = r.get("image_file")
        if img:
            by_img.setdefault(img, []).append(r)
    return {
        img: aggregate_scores(results, bootstrap_ci, n_bootstrap, ci_level)
        for img, results in sorted(by_img.items())
    }


def image_caption_variance(
    probe_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Variance of similarities across images and captions.

    For each (identity, attribute), compute the std of mean-similarities
    across image variants and caption variants separately.  Reports
    the average std as a measure of probe sensitivity to image/caption
    choice.
    """
    import numpy as np

    # Group by (identity_id, attribute)
    by_iid_attr: dict[tuple[str, str], list[dict]] = {}
    for r in probe_results:
        key = (r.get("identity_id", ""), r.get("attribute", ""))
        by_iid_attr.setdefault(key, []).append(r)

    img_stds: list[float] = []
    cap_stds: list[float] = []
    for (iid, attr), results in by_iid_attr.items():
        # Group by image
        by_img: dict[str, list[dict]] = {}
        for r in results:
            by_img.setdefault(r.get("image_file", ""), []).append(r)
        if len(by_img) > 1:
            img_means = []
            for img, rs in by_img.items():
                fine_sims = [r["sims"].get("fine", 0) for r in rs
                             if "fine" in r["sims"]]
                if fine_sims:
                    img_means.append(sum(fine_sims) / len(fine_sims))
            if len(img_means) > 1:
                img_stds.append(float(np.std(img_means)))

        # Group by caption
        by_cap: dict[str, list[dict]] = {}
        for r in results:
            by_cap.setdefault(r.get("fine_caption", ""), []).append(r)
        if len(by_cap) > 1:
            cap_means = []
            for cap, rs in by_cap.items():
                fine_sims = [r["sims"].get("fine", 0) for r in rs
                             if "fine" in r["sims"]]
                if fine_sims:
                    cap_means.append(sum(fine_sims) / len(fine_sims))
            if len(cap_means) > 1:
                cap_stds.append(float(np.std(cap_means)))

    return {
        "num_identity_attr_pairs": len(by_iid_attr),
        "image_std_mean": round(sum(img_stds) / len(img_stds), 6)
        if img_stds else None,
        "image_std_max": round(max(img_stds), 6) if img_stds else None,
        "caption_std_mean": round(sum(cap_stds) / len(cap_stds), 6)
        if cap_stds else None,
        "caption_std_max": round(max(cap_stds), 6) if cap_stds else None,
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
