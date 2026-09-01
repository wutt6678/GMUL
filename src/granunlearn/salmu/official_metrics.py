"""Released SALMUBench split summarization (Iteration 10R4a).

Pure, unit-testable summarization for
``scripts/evaluate_salmu_official_splits.py``.

10R4a fixes
-----------
* Leakage estimand/bootstrap correspondence: the identity-level
  leakage point estimate is the mean of PER-IDENTITY flags (an
  identity leaks iff its MACRO assoc sim exceeds its MACRO generic
  sim), and its bootstrap resamples exactly those identity flags.
  The pair-level leakage rate carries its OWN pair-level bootstrap
  CI.  A single CI can never serve two different estimands.
* retain_synth clustering is forced pair-level for ALL rows
  (deterministically, by split name) — the released synthetic split
  defines no identity units, so identity-clustered statistics are
  meaningless there even if some rows carried an identity_id.

Every CI is therefore computed under the SAME clustering as the
point estimate it accompanies.
"""

from __future__ import annotations

from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("salmu_official_metrics")

# Splits where identity units are undefined by construction: each
# pair is its own cluster, unconditionally (10R4a).
PAIR_CLUSTERED_SPLITS = ("retain_synth",)

AGGREGATION_SCHEMA = "10r4a.v1"


def cluster_ids(split_name: str,
                ids: list[str | None]) -> tuple[list[str], bool]:
    """Clustering unit per row.

    Returns ``(clustered_ids, pair_clustered)``.  Pair-level
    clustering is FORCED for the whole split when:

    * the split is in ``PAIR_CLUSTERED_SPLITS`` (retain_synth), or
    * ANY row lacks an ``identity_id`` (defensively).

    Under pair-level clustering every row gets a unique sentinel so
    identity-macro statistics degenerate exactly to pair statistics.
    """
    forced = split_name in PAIR_CLUSTERED_SPLITS
    any_missing = any(iid is None for iid in ids)
    if forced or any_missing:
        return [f"__pair_{i}" for i in range(len(ids))], True
    return list(ids), False


def identity_leak_flags(assoc, generic, has_generic,
                        uniq: list[str],
                        by_id: dict[str, list[int]]):
    """Per-identity leakage flags + the identities that have them.

    An identity leaks iff its MACRO association similarity (over its
    rows with released generic captions) exceeds its MACRO generic
    similarity.  Returns ``(flag_ids, flags)`` where ``flags`` is a
    numpy 0/1 array aligned with ``flag_ids``; identities without any
    generic-covered row are excluded.
    """
    import numpy as np
    flag_ids: list[str] = []
    flags: list[float] = []
    for iid in uniq:
        rows = [i for i in by_id[iid] if has_generic[i]]
        if not rows:
            continue
        flag_ids.append(iid)
        flags.append(float(assoc[rows].mean() > generic[rows].mean()))
    return flag_ids, np.asarray(flags, dtype=np.float64)


def _bootstrap_ci(rng, values, n_bootstrap: int,
                  ci_level: float) -> tuple[float, float]:
    """Percentile bootstrap CI over resampled units."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pick = rng.integers(0, n, size=n)
        boots[b] = values[pick].mean()
    alpha = (1 - ci_level) / 2
    return (round(float(np.quantile(boots, alpha)), 4),
            round(float(np.quantile(boots, 1 - alpha)), 4))


def summarize_rows(ids: list[str | None],
                   assoc_sim: list[float],
                   generic_sim: list[float | None],
                   split_name: str,
                   n_bootstrap: int = 1000,
                   ci_level: float = 0.95,
                   seed: int = 42) -> dict[str, Any]:
    """Point estimates + clustering-correspondent CIs for one set of
    (image, association-caption) rows.

    Used for the full released split AND for GMUL target subsets —
    every reported CI is bootstrapped under the same clustering as
    its point estimate.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    clustered, pair_clustered = cluster_ids(split_name, ids)
    assoc = np.asarray(assoc_sim, dtype=np.float64)
    generic = np.asarray(
        [g if g is not None else np.nan for g in generic_sim],
        dtype=np.float64)
    uniq = sorted(set(clustered))
    by_id: dict[str, list[int]] = {}
    for i, iid in enumerate(clustered):
        by_id.setdefault(iid, []).append(i)

    # Unit-macro association similarity (unit = identity, or pair
    # when clustering is forced)
    unit_means = np.array([assoc[by_id[iid]].mean() for iid in uniq])

    # Leakage: assoc caption outscores the released generic caption
    has_generic = ~np.isnan(generic)
    entry: dict[str, Any] = {
        "num_pairs": len(clustered),
        "num_units": None if pair_clustered else len(uniq),
        "pair_clustered": pair_clustered,
        "mean_assoc_sim": round(float(assoc.mean()), 4),
        "unit_macro_assoc_sim": round(float(unit_means.mean()), 4),
    }
    entry["unit_macro_assoc_sim_ci"] = _bootstrap_ci(
        rng, unit_means, n_bootstrap, ci_level)

    if pair_clustered:
        entry["clustering_note"] = (
            "pair-level clustering is forced for this split/subset "
            "(retain_synth carries no identity units); the unit-macro "
            "statistic equals the pair mean and every CI is a "
            "pair-level bootstrap.")

    if has_generic.any():
        leak_flags_pairwise = (assoc > generic)[has_generic]
        leak_pair = float(leak_flags_pairwise.mean())
        entry["leakage_rate"] = round(leak_pair, 4)
        # Pair-level CI for the PAIR-level rate
        entry["leakage_rate_ci"] = _bootstrap_ci(
            rng, leak_flags_pairwise, n_bootstrap, ci_level)

        flag_ids, id_flags = identity_leak_flags(
            assoc, generic, has_generic, uniq, by_id)
        if len(id_flags):
            entry["identity_leakage_rate"] = round(
                float(id_flags.mean()), 4)
            # Identity-clustered CI for the IDENTITY-level rate —
            # resamples exactly the per-identity flags that define
            # the point estimate (10R4a correspondence).
            entry["identity_leakage_rate_ci"] = _bootstrap_ci(
                rng, id_flags, n_bootstrap, ci_level)
            entry["identity_leakage_num_units"] = len(flag_ids)
        else:
            entry["identity_leakage_rate"] = None
    return entry


def summarize_state(per_split: dict, n_bootstrap: int = 1000,
                    ci_level: float = 0.95) -> dict:
    """Identity/pair-clustered summaries for every released split of
    one checkpoint, including the exact GMUL target subsets when
    ``per_split`` rows carry the subset membership lists.
    """
    summary: dict = {}
    for split_name, data in per_split.items():
        entry = summarize_rows(
            data["identity_id"], data["assoc_sim"],
            data["generic_sim"], split_name,
            n_bootstrap=n_bootstrap, ci_level=ci_level)
        # Exact GMUL target subsets (10R4a): membership flags are
        # attached by the scorer when the released rows overlap the
        # GMUL target partition.
        for subset_key, label in (
                ("gmul_target_mask", "gmul_target_subset"),
                ("gmul_target_attr_mask",
                 "gmul_target_attr_subset")):
            mask = data.get(subset_key)
            if not mask or not any(mask):
                continue
            idx = [i for i, m in enumerate(mask) if m]
            sub = summarize_rows(
                [data["identity_id"][i] for i in idx],
                [data["assoc_sim"][i] for i in idx],
                [data["generic_sim"][i] for i in idx],
                split_name, n_bootstrap=n_bootstrap,
                ci_level=ci_level)
            sub["definition"] = (
                "rows restricted to GMUL target-persona identities"
                if subset_key == "gmul_target_mask" else
                "rows restricted to each GMUL target persona's "
                "designated target attribute (identity + attribute "
                "from the committed state-pairs manifest)")
            entry[label] = sub
        summary[split_name] = entry
    return summary
