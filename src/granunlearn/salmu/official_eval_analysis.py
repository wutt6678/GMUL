"""Analysis of official SALMUBench-evaluator raw results (10R5b).

Adds two things the official per-state JSONs do not provide:

1. TARGET-ONLY official metrics — the forget-split rows restricted to
   each GMUL target persona's designated target attribute (the
   associations GMUL actually unlearns), so the official metrics can
   be read on exactly the targeted knowledge.
2. PAIRED confidence intervals — identity-clustered bootstrap CIs of
   the DIFFERENCE between two states over the SAME rows/identities
   (paired by construction: the official DataLoader never shuffles,
   so per-row score lists are aligned with the released split order).

All functions are pure over plain lists so they are unit-testable
without released data.  Row alignment: the official evaluator's
``scores``/``outcomes`` arrays follow the released split row order
(DataLoader with shuffle=False), i.e. the concatenation of the sorted
parquet shards.
"""

from __future__ import annotations

from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("salmu_official_eval_analysis")


# ── pure clustering statistics ────────────────────────────────────

def unit_means(values: list[float], clusters: list[str],
               mask: list[bool] | None = None
               ) -> tuple[list[str], list[float]]:
    """Mean value per cluster unit (optionally over masked rows).

    Returns (unit_keys sorted, unit_means aligned).  A unit with no
    selected rows is omitted.
    """
    by_unit: dict[str, list[float]] = {}
    for i, v in enumerate(values):
        if mask is not None and not mask[i]:
            continue
        by_unit.setdefault(clusters[i], []).append(v)
    keys = sorted(by_unit)
    return keys, [sum(by_unit[k]) / len(by_unit[k]) for k in keys]


def _percentile_ci(rng, unit_vals: list[float], n_bootstrap: int,
                   ci_level: float) -> tuple[float, float]:
    import numpy as np
    from granunlearn.salmu.embedding_metrics import validate_ci_params
    validate_ci_params(n_bootstrap, ci_level)
    arr = np.asarray(unit_vals, dtype=np.float64)
    n = len(arr)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pick = rng.integers(0, n, size=n)
        boots[b] = arr[pick].mean()
    alpha = (1 - ci_level) / 2
    return (round(float(np.quantile(boots, alpha)), 4),
            round(float(np.quantile(boots, 1 - alpha)), 4))


def clustered_mean_ci(values: list[float], clusters: list[str],
                      mask: list[bool] | None = None,
                      n_bootstrap: int = 1000, ci_level: float = 0.95,
                      seed: int = 42) -> dict[str, Any] | None:
    """Identity/unit-clustered mean + bootstrap CI (macro over units,
    CI resamples units)."""
    import numpy as np
    keys, means = unit_means(values, clusters, mask)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    return {
        "mean": round(float(np.mean(means)), 4),
        "ci": _percentile_ci(rng, means, n_bootstrap, ci_level),
        "num_units": len(keys),
        "num_rows": sum(1 for m in (mask or [True] * len(values))
                        if m),
    }


def paired_clustered_diff_ci(values_a: list[float],
                             values_b: list[float],
                             clusters: list[str],
                             mask: list[bool] | None = None,
                             n_bootstrap: int = 1000,
                             ci_level: float = 0.95,
                             seed: int = 42) -> dict[str, Any] | None:
    """Paired difference mean(a) - mean(b) with an identity-clustered
    bootstrap CI: the SAME units are resampled for both states (per-
    unit difference of unit means), so the CI is exact for the paired
    macro-average estimand."""
    import numpy as np
    keys_a, means_a = unit_means(values_a, clusters, mask)
    keys_b, means_b = unit_means(values_b, clusters, mask)
    if keys_a != keys_b or not keys_a:
        return None
    diffs = [a - b for a, b in zip(means_a, means_b)]
    rng = np.random.default_rng(seed)
    return {
        "diff": round(float(np.mean(diffs)), 4),
        "ci": _percentile_ci(rng, diffs, n_bootstrap, ci_level),
        "num_units": len(keys_a),
    }


def retrieval_stats(ranks: list[int],
                    mask: list[bool] | None = None) -> dict | None:
    """MRR / R@1 over (optionally masked) rank outcomes."""
    sel = [r for i, r in enumerate(ranks)
           if mask is None or mask[i]]
    if not sel:
        return None
    return {
        "MRR": round(sum(1.0 / r for r in sel) / len(sel), 4),
        "R@1": round(sum(1 for r in sel if r == 1) / len(sel), 4),
        "num_rows": len(sel),
    }


# ── released-split row order + target mask ────────────────────────

def split_row_order(bench_dir, split: str
                    ) -> tuple[list[str], list[str]]:
    """(identity_id, file_name) per row of a released split, in the
    SAME order the official evaluator's DataLoader produces (sorted
    parquet shards concatenated, no shuffle)."""
    import glob

    import pandas as pd
    files = sorted(glob.glob(str(bench_dir / "data" /
                                 f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards for {split}")
    ids: list[str] = []
    names: list[str] = []
    for f in files:
        df = pd.read_parquet(f, columns=["identity_id", "file_name"])
        ids.extend(df["identity_id"].tolist())
        names.extend(df["file_name"].tolist())
    return ids, names


def target_attr_mask(ids: list[str], file_names: list[str],
                     target_ids: set[str],
                     target_attr_map: dict[str, str],
                     attr_of: dict[str, str]) -> list[bool]:
    """True for rows on a GMUL target persona's DESIGNATED target
    attribute (identity from the row, attribute resolved through the
    released caption metadata's file_name -> data_field map)."""
    return [bool(iid in target_ids
                 and attr_of.get(fname) == target_attr_map.get(iid))
            for iid, fname in zip(ids, file_names)]


def identity_mask(ids: list[str], id_set: set[str]) -> list[bool]:
    """True for rows whose identity is in ``id_set``."""
    return [iid in id_set for iid in ids]


# ── per-state target-only summary from a raw official result ─────

def target_only_from_raw(raw: dict, forget_ids: list[str],
                         forget_files: list[str],
                         tmask: list[bool],
                         n_bootstrap: int = 1000,
                         seed: int = 42) -> dict[str, Any]:
    """Target-only official metrics from ONE raw official result.

    Row-aligned metrics over the forget split: AssocStr, CoreAssoc
    (clustered mean + CI over target identities) and RetFail
    (MRR/R@1 over target rows).  IdZSC target-only uses the
    forget_identity row order (forget rows whose identity is NOT in
    holdout_association) and reports accuracy over target-persona
    rows.
    """
    out: dict[str, Any] = {}
    assoc = raw.get("efficacy", {}).get("1.2_AssocStr", {}) \
        .get("scores")
    if assoc and len(assoc) == len(forget_ids):
        s = clustered_mean_ci(assoc, forget_ids, tmask,
                              n_bootstrap=n_bootstrap, seed=seed)
        if s:
            out["AssocStr_target"] = s
    core = raw.get("efficacy", {}).get("1.5_CoreAssoc", {}) \
        .get("scores")
    if core and len(core) == len(forget_ids):
        s = clustered_mean_ci(core, forget_ids, tmask,
                              n_bootstrap=n_bootstrap, seed=seed)
        if s:
            out["CoreAssoc_target"] = s
    ranks = raw.get("efficacy", {}).get("1.1_RetFail", {}) \
        .get("outcomes")
    if ranks and len(ranks) == len(forget_ids):
        s = retrieval_stats(ranks, tmask)
        if s:
            out["RetFail_target"] = s
    return out


def paired_target_only(raw_a: dict, raw_b: dict,
                       forget_ids: list[str], tmask: list[bool],
                       n_bootstrap: int = 1000,
                       seed: int = 42) -> dict[str, Any]:
    """Paired target-only differences (a - b) with identity-clustered
    CIs for the row-aligned forget metrics."""
    out: dict[str, Any] = {}
    for name, section, key in (("AssocStr_target", "1.2_AssocStr",
                                "scores"),
                               ("CoreAssoc_target", "1.5_CoreAssoc",
                                "scores")):
        a = raw_a.get("efficacy", {}).get(section, {}).get(key)
        b = raw_b.get("efficacy", {}).get(section, {}).get(key)
        if a and b and len(a) == len(b) == len(forget_ids):
            s = paired_clustered_diff_ci(a, b, forget_ids, tmask,
                                         n_bootstrap=n_bootstrap,
                                         seed=seed)
            if s:
                out[name] = s
    return out
