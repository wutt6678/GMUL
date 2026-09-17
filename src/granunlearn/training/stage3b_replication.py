"""Iteration 12 Stage 3b: what happens AFTER Stage 3 is scored.

Stage 3 trains one seed per B7 row.  A single-seed verdict on eight numbers
whose denominators are 408 and 76 per route cannot distinguish a repaired
shortfall from a lucky draw, so the Stage-3 freeze was amended -- before any
B7 adapter existed -- to preregister the whole downstream tree:

    no B7 qualifies                    -> close Iteration 12, negative result
    a B7 passes or narrowly misses     -> Stage 3b at seeds 42/43/44/45
    the replicated mean clears all eight -> select by minimum D_G
    the replicated mean does not       -> stop

WHY THE NEAR-MISS THRESHOLD IS INCUMBENT-RELATIVE
-------------------------------------------------
The obvious threshold -- "within measured seed noise" -- has no measured
referent here.  Stage 1b measured between-seed sd on the four TEXT numbers
only (0.0121 and 0.0127 for retain-same, 0.0225 and 0.0195 for retain-other),
because the floor it adjudicated was text-only.  B6 fails three IMAGE numbers,
and no image-route training-seed spread has ever been measured.  Generation
noise cannot substitute: under the fixed contract the Stage-1b determinism
control found 0 differing rows out of 4,518 on all seven fields, so the only
unmeasured variance is the one that matters.

So the envelope is defined against the incumbent instead.  B6 is the best
existing mechanism on this criterion; its own retention shortfall is measured,
filed and hash-bound.  "No worse than B6's shortfall" needs no variance
estimate, no distributional assumption and no new measurement, and it cannot
be tuned after the fact because B6's numbers predate every B7 row.

WHAT THIS MODULE IS
-------------------
The decision rules themselves, as code the Stage-3 freeze hashes.  A Stage-3b
trainer or analyzer may IMPORT these functions; it may not restate or
reinterpret them, and the freeze that binds them cannot be amended again once
a B7 adapter exists.  Nothing here edits a frozen path: the floor is
:func:`~granunlearn.evaluation.route_stratified_retention.floor_check_stratified`
and D_G is :func:`~granunlearn.evaluation.retention_selection.distance_to_mg`,
both called, never reimplemented.

Stage 1b's analyzer is deliberately NOT reused.  Its ``mean_candidate`` builds
``floor_check``'s four-number, B0-anchored input shape; scoring a Stage-3b mean
with it would silently drop the four image numbers that decide Stage 3 and
re-anchor the floor on a state Stage 3 does not use.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from granunlearn.evaluation import retention_selection as rs
from granunlearn.evaluation import route_stratified_retention as rt
from granunlearn.evaluation.selection import SUMMARY_COMPONENTS
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g

#: The frozen tolerance.  It is the same 1e-9 the floor uses: a representation
#: guard that can rescue an exact tie and can never absorb a real shortfall.
EPSILON = rs.FLOOR_EPSILON

#: The eight numbers, in the frozen key order.  Order is part of the contract.
FLOOR_KEYS: tuple[str, ...] = rt.FLOOR_NUMBER_KEYS

# ── the replicate set ───────────────────────────────────────────────────
#: Seed 42 is the seed every Stage-3 B7 row is trained with; its adapter and
#: predictions already exist and are sidecar-verified, so it is REUSED as the
#: first replicate.  Re-running it would produce a second draw from the same
#: seed and measure GPU nondeterminism, not between-seed variance.
BASE_SEED = 42

#: The three additional seeds, and they are the Stage-1b seeds.  Recorded here
#: before any replicate exists so they cannot be chosen after seeing results.
NEW_SEEDS: tuple[int, ...] = (43, 44, 45)
ALL_SEEDS: tuple[int, ...] = (BASE_SEED, *NEW_SEEDS)

#: At most two B7 rows are replicated.  Two parents x three new seeds is six
#: training runs; a wider tree would spend the budget on rows the envelope has
#: already ranked below these.
MAX_PARENTS = 2

STAGE3B_CKPT_ROOT = "data/checkpoints/mllmu_iter12_stage3b"
STAGE3B_PREDICTIONS_SUBDIR = "predictions_iter12_stage3b"
OUT_REPORT = "data/reports/mllmu_iter12_stage3b_replication.json"

#: Classification labels, so a report cannot invent its own vocabulary.
EXACT_PASS = "exact_pass"
NEAR_MISS = "incumbent_relative_near_miss"
DOES_NOT_QUALIFY = "does_not_qualify"

CLOSE_ITERATION_12 = "close_iteration_12_as_a_documented_negative_result"
RUN_STAGE_3B = "run_stage_3b_at_the_frozen_replication_seeds"

STOP_ITERATION_12 = "stop_iteration_12_no_mean_eligible_parent"
SELECT_THE_ONLY_PARENT = "select_the_single_mean_eligible_parent"
SELECT_BY_MINIMUM_D_G = "select_the_mean_eligible_parent_with_minimum_d_g"
ITERATION_13 = "proceed_to_a_separately_frozen_iteration_13_confirmation"

#: B6's eight floor numbers, recomputed from the bound Stage-2 prediction
#: parquet (``reused_predictions_bound.paths["stage2.B6"]``) at amendment time,
#: NOT transcribed from a report's rounded difference fields.  The selector
#: re-derives them from the same bytes before every Stage-3 scoring run and
#: refuses to score if they no longer reproduce; a test pins the agreement
#: wherever the gitignored parquet is present.
B6_FLOOR_VALUES_AT_AMENDMENT_TIME: dict[str, float] = {
    "text.retain_same_entity.entity_macro": 0.543,
    "text.retain_same_entity.row_micro": 0.5466,
    "text.retain_other_entity.entity_macro": 0.5199,
    "text.retain_other_entity.row_micro": 0.5132,
    "image.retain_same_entity_image.entity_macro": 0.5043,
    "image.retain_same_entity_image.row_micro": 0.5098,
    "image.retain_other_entity_image.entity_macro": 0.4348,
    "image.retain_other_entity_image.row_micro": 0.4211,
}

#: The distinct-row shortfall behind those three failures, as filed in the
#: Stage-3 basis.  Reported beside the envelope; it decides nothing, because
#: entity_macro and row_micro are two estimands over the same rows and adding
#: them would double-count.
B6_QUERY_SHORTFALL: dict[str, Any] = {
    "same_image_rows": 5, "same_image_denominator": 408,
    "other_image_rows": 2, "other_image_denominator": 76,
    "total_distinct_rows": 14,
}


# ── shortfalls and the incumbent envelope ───────────────────────────────

def shortfall_vector(values: Mapping[str, float | None],
                     anchor: Mapping[str, float | None],
                     epsilon: float = EPSILON) -> dict[str, float]:
    r"""d_j(c) = max(0, a_j - v_j(c) - epsilon) for each of the eight numbers.

    Nothing is rounded.  The inputs are the sealed metric's own four-decimal
    rates; rounding the RESULT, or reading a report's four-decimal difference
    field instead, would drop the 1e-9 and could flip a ``<=`` comparison
    against the envelope at exactly the boundary the epsilon exists to guard.

    A number that is missing on either side raises rather than defaulting to
    zero: an unmeasurable retention number is not a near miss, it is a
    disqualification, and treating it as 0.0 would let an absent measurement
    qualify a candidate for replication.
    """
    out: dict[str, float] = {}
    for key in FLOOR_KEYS:
        got = values.get(key)
        want = anchor.get(key)
        if got is None or want is None:
            raise ValueError(
                f"{key}: candidate={got!r} anchor={want!r}. A retention number "
                f"that could not be measured is a disqualification, not a "
                f"zero shortfall; refusing to rank it as a near miss.")
        out[key] = max(0.0, float(want) - float(got) - epsilon)
    return out


def envelope(shortfalls: Mapping[str, float]) -> dict[str, Any]:
    """(K, M, S) -- how many numbers fail, the worst shortfall, the total.

    ``S`` is ``math.fsum``, which is exactly rounded and therefore independent
    of summation order; ``sum`` over floats is not, and a threshold that moves
    in its last bits with the key order is not a frozen threshold.
    """
    failed = sorted(k for k, v in shortfalls.items() if v > 0)
    return {
        "K": len(failed),
        "M": max(shortfalls.values()),
        "S": math.fsum(shortfalls.values()),
        "failed_metrics": failed,
        "per_metric": {k: shortfalls[k] for k in FLOOR_KEYS},
        "summation": "math.fsum over the frozen key order (exactly rounded, "
                     "so it cannot depend on the order)",
    }


def incumbent_envelope(anchor: Mapping[str, float | None]) -> dict[str, Any]:
    """The B6 envelope, derived from the frozen anchor and B6's own values."""
    return envelope(shortfall_vector(B6_FLOOR_VALUES_AT_AMENDMENT_TIME, anchor))


def verify_incumbent_values(values: Mapping[str, float | None],
                            ) -> list[str]:
    """Does the bound Stage-2 parquet still reproduce the filed B6 numbers?

    Returns problems rather than raising so the caller can refuse with a
    message that names every mismatch at once.
    """
    problems = []
    for key in FLOOR_KEYS:
        got = values.get(key)
        want = B6_FLOOR_VALUES_AT_AMENDMENT_TIME[key]
        if got != want:
            problems.append(
                f"{key}: the envelope was filed against {want} but the bound "
                f"B6 predictions recompute to {got!r}")
    return problems


# ── qualification ───────────────────────────────────────────────────────

def qualification(shortfalls: Mapping[str, float],
                  b6: Mapping[str, Any]) -> dict[str, Any]:
    """Exact pass, incumbent-relative near miss, or neither.

    The near miss requires ALL THREE conjuncts.  Any one of them alone would
    let a candidate trade a catastrophic single-number failure for a spread-out
    one: ``K`` bounds how many numbers may fail, ``M`` bounds how badly any one
    may fail, and ``S`` bounds the total.  Dropping ``S`` would admit three
    numbers each just under ``M``; dropping ``M`` would admit one number
    arbitrarily far below the anchor.
    """
    mine = envelope(shortfalls)
    conjuncts = {
        "failed_metrics_at_most_K": mine["K"] <= b6["K"],
        "max_shortfall_at_most_M": mine["M"] <= b6["M"],
        "total_shortfall_at_most_S": mine["S"] <= b6["S"],
    }
    exact = mine["K"] == 0 and mine["M"] == 0.0 and mine["S"] == 0.0
    near = all(conjuncts.values())
    if exact:
        label = EXACT_PASS
    elif near:
        label = NEAR_MISS
    else:
        label = DOES_NOT_QUALIFY
    return {
        "classification": label,
        "qualifies": label != DOES_NOT_QUALIFY,
        "exact_pass": exact,
        "near_miss": near and not exact,
        "K": mine["K"], "M": mine["M"], "S": mine["S"],
        "failed_metrics": mine["failed_metrics"],
        "per_metric_shortfall": mine["per_metric"],
        "conjuncts": conjuncts,
        "violated_conjuncts": sorted(k for k, v in conjuncts.items() if not v),
        "envelope_used": {"K": b6["K"], "M": b6["M"], "S": b6["S"]},
    }


def parent_rank_key(qual: Mapping[str, Any], distance_to_mg: float | None,
                    candidate_id: str) -> tuple[int, int, float, float,
                                                float, str]:
    """The frozen parent ordering, as one sortable key.

    ``(0 if exact pass else 1, K, M, S, D_G, candidate_id)``.  For an exact
    passer K, M and S are all zero by construction, so the key degenerates to
    ``(0, 0, 0, 0, D_G, id)`` -- exact pass first, then smaller D_G, then id --
    which is exactly the prescribed order for that class.  For a near miss it
    is ``(1, K, M, S, D_G, id)``: fewer failed metrics, then the smaller worst
    shortfall, then the smaller total, then D_G, then id.
    """
    if distance_to_mg is None:
        raise ValueError(
            f"{candidate_id}: D_G is not computable, so the row cannot be "
            f"ranked against the other qualifiers. A candidate whose distance "
            f"to the anchor is unknown is not a tie, it is unmeasurable.")
    return (0 if qual["exact_pass"] else 1, qual["K"], qual["M"], qual["S"],
            distance_to_mg, candidate_id)


def select_parents(rows: Mapping[str, Mapping[str, Any]],
                   anchor: Mapping[str, float | None],
                   incumbent_id: str,
                   max_parents: int = MAX_PARENTS) -> dict[str, Any]:
    """Qualify every Stage-3 row, then take at most ``max_parents`` B7 rows.

    ``rows[cid]`` needs ``method``, ``values`` (the eight numbers recomputed
    from that row's own predictions) and ``distance_to_mg``.

    Only B7 rows can become parents.  The B6 zero-weight control is excluded
    even though it satisfies its own envelope with equality on all three
    conjuncts -- which is precisely why the exclusion has to be explicit:
    without it the control would select itself as its own successor.  B0 is
    excluded because the no-op reference is not a mechanism.
    """
    if incumbent_id not in rows:
        raise LookupError(
            f"the incumbent {incumbent_id!r} is not among the scored rows, so "
            f"no envelope can be derived; the near-miss rule would be "
            f"undefined rather than merely strict")
    incumbent_values = rows[incumbent_id]["values"]
    problems = verify_incumbent_values(incumbent_values)
    b6 = incumbent_envelope(anchor)
    recomputed = envelope(shortfall_vector(incumbent_values, anchor))
    for field in ("K", "M", "S"):
        if recomputed[field] != b6[field]:
            problems.append(
                f"the incumbent envelope {field} was filed as {b6[field]!r} "
                f"but the bound parquet recomputes to {recomputed[field]!r}")

    per_candidate: dict[str, Any] = {}
    qualifiers: list[tuple[tuple, str]] = []
    for cid in sorted(rows):
        row = rows[cid]
        qual = qualification(shortfall_vector(row["values"], anchor), b6)
        eligible = row["method"] == s3g.METHOD_ROUTE_ANCHOR
        excluded = None
        if not eligible:
            excluded = ("only the four B7 rows can become Stage-3b parents"
                        if cid != incumbent_id else
                        "the zero-weight control validates reproduction and "
                        "defines the envelope; it is the filed incumbent, not "
                        "a new successor")
        per_candidate[cid] = {
            "method": row["method"],
            "distance_to_mg": row["distance_to_mg"],
            "parent_eligible": eligible,
            "excluded_because": excluded,
            **{k: qual[k] for k in ("classification", "qualifies",
                                    "exact_pass", "near_miss", "K", "M", "S",
                                    "failed_metrics", "per_metric_shortfall",
                                    "conjuncts", "violated_conjuncts")},
        }
        if eligible and qual["qualifies"]:
            qualifiers.append((parent_rank_key(qual, row["distance_to_mg"], cid),
                               cid))

    qualifiers.sort(key=lambda t: t[0])
    ranked = [cid for _key, cid in qualifiers]
    parents = ranked[:max_parents]
    return {
        "anchor": "MG",
        "epsilon": EPSILON,
        "shortfall_definition":
            "d_j(c) = max(0, a_j - v_j(c) - epsilon), unrounded, over the "
            "eight frozen floor numbers",
        "incumbent": {
            "candidate_id": incumbent_id,
            "floor_values": dict(B6_FLOOR_VALUES_AT_AMENDMENT_TIME),
            "recomputed_from_the_bound_parquet": True,
            "problems": problems,
            "envelope_reproduces": not problems,
            "envelope": b6,
            "query_shortfall_behind_it": dict(B6_QUERY_SHORTFALL),
            "why_incumbent_relative": (
                "the image route's between-seed training variance has never "
                "been measured -- Stage 1b measured sd on the four TEXT "
                "numbers only -- and generation noise under the fixed "
                "contract is measured at zero, so 'within measured seed "
                "noise' has no referent for the three image numbers that "
                "decide Stage 3"),
        },
        "per_candidate": per_candidate,
        "ranking_of_qualifiers": ranked,
        "ranking_key": "(0 if exact_pass else 1, K, M, S, distance_to_mg, "
                       "candidate_id)",
        "parents": parents,
        "num_parents": len(parents),
        "max_parents": max_parents,
        "held_back": ranked[max_parents:],
        "decision": RUN_STAGE_3B if parents else CLOSE_ITERATION_12,
        "if_no_b7_qualifies": (
            "Iteration 12 closes immediately as a documented negative result. "
            "No Stage-3b replicate is trained and no successor is adopted; a "
            "non-B7 row sitting in `selected` under the frozen criterion does "
            "not open Stage 3b, because B0 and the B6 control are references "
            "rather than mechanisms."),
    }


# ── Stage 3b: means over the frozen seeds ───────────────────────────────

def _require_all_seeds(per_seed: Mapping[int, Any], what: str) -> None:
    missing = sorted(set(ALL_SEEDS) - set(per_seed))
    if missing:
        raise ValueError(
            f"cannot average {what} over a subset: seed(s) {missing} are "
            f"missing and the frozen replicate set is {list(ALL_SEEDS)}. A "
            f"mean over fewer seeds is a different statistic than the one "
            f"preregistered, not a noisier estimate of it.")


def mean_floor_values(per_seed: Mapping[int, Mapping[str, float | None]],
                      ) -> dict[str, Any]:
    """The componentwise mean of each of the eight numbers, UNROUNDED.

    Rounding the mean to the four decimals its inputs carry would let a
    rounding step decide a comparison the inputs did not decide -- the reason
    Stage 1b compares its mean unrounded, carried over unchanged.
    """
    _require_all_seeds(per_seed, "the eight floor numbers")
    out: dict[str, Any] = {}
    for key in FLOOR_KEYS:
        vals = []
        for seed in ALL_SEEDS:
            got = per_seed[seed].get(key)
            if got is None:
                raise ValueError(
                    f"{key} at seed {seed} is not measurable, so the mean is "
                    f"not measurable; a missing number disqualifies rather "
                    f"than averaging in as zero")
            vals.append(float(got))
        out[key] = {
            "k": len(vals),
            "values": vals,
            "mean": statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return {
        "values": {k: v["mean"] for k, v in out.items()},
        "per_number": out,
        "mean_is_compared_unrounded": True,
        "seeds": list(ALL_SEEDS),
    }


def mean_floor_check(mean_values: Mapping[str, float | None],
                     anchor: Mapping[str, float | None],
                     epsilon: float = EPSILON) -> dict[str, Any]:
    """The replicated mean against the MG anchor, on all eight numbers.

    This is :func:`~...route_stratified_retention.floor_check_stratified`
    called on the mean vector, so the rule is byte-identical to the one Stage
    3 applies to a single seed.  The text-stratum cross-check is asserted, not
    skipped: the stratified floor must still say what the frozen floor says on
    the four numbers they share.
    """
    result = rt.floor_check_stratified(dict(mean_values), dict(anchor), epsilon)
    rt.text_stratum_reproduces_the_frozen_floor(result)
    return {
        **result,
        "scored_by": "route_stratified_retention.floor_check_stratified",
        "the_stage1b_analyzer_was_not_reused": True,
        "why_not": (
            "Stage 1b's mean_candidate/floor_check path is text-only and "
            "B0-anchored; it would drop the four image numbers that decide "
            "Stage 3 and re-anchor the floor on a state Stage 3 does not use"),
        "mean_is_compared_unrounded": True,
        "text_stratum_reproduces_the_frozen_floor": True,
    }


def mean_summary_vector(per_seed: Mapping[int, Mapping[str, float | None]],
                        ) -> dict[str, float]:
    """The componentwise mean of the COMPLETE summary vector, unrounded.

    Every component D_G reads is averaged, not only the retention terms:
    averaging a subset and inheriting the rest from one seed would make the
    mean vector a hybrid no seed ever was.
    """
    _require_all_seeds(per_seed, "the summary vector")
    out: dict[str, float] = {}
    for comp in SUMMARY_COMPONENTS:
        vals = []
        for seed in ALL_SEEDS:
            got = per_seed[seed].get(comp)
            if got is None:
                raise ValueError(
                    f"summary component {comp!r} is missing at seed {seed}, so "
                    f"the mean vector is incomplete and D_G over it would "
                    f"silently renormalize over fewer components")
            vals.append(float(got))
        out[comp] = statistics.fmean(vals)
    return out


def d_g_of_mean_vector(mean_vector: Mapping[str, float | None],
                       mg_vector: Mapping[str, float | None],
                       ) -> tuple[float | None, list[str]]:
    """D_G of the MEAN vector, via the frozen distance function.

    The forbidden alternative -- averaging four already-computed per-seed D_G
    values -- is not equivalent and is not used: ``distance_to_reference`` is a
    weighted L1 over absolute component differences rounded to six decimals
    per call, ``abs`` is convex, and the rounding happens inside each call, so
    mean-of-distances and distance-of-means can differ and can rank two
    parents in the opposite order.
    """
    return rs.distance_to_mg(dict(mean_vector), dict(mg_vector))


def per_seed_d_g_disclosure(per_seed_d_g: Mapping[int, float | None],
                            ) -> dict[str, Any]:
    """The rejected statistic, reported so the choice is visible.

    Decision-bearing nowhere.  It exists because "we averaged the vector
    instead of the distances" is only checkable if a reader can see both.
    """
    vals = [v for v in (per_seed_d_g.get(s) for s in ALL_SEEDS)
            if v is not None]
    return {
        "per_seed": {str(s): per_seed_d_g.get(s) for s in ALL_SEEDS},
        "mean_of_per_seed_d_g": statistics.fmean(vals) if vals else None,
        "decision_bearing": False,
        "used_instead": "d_g_of_mean_vector -- the frozen distance applied to "
                        "the componentwise mean vector",
        "why_not_used": (
            "distance_to_reference rounds to six decimals inside each call and "
            "averages absolute differences, so the mean of four distances is "
            "not the distance of the mean vector and the two can order two "
            "parents differently"),
    }


def replication_outcome(parents: Mapping[str, Mapping[str, Any]],
                        ) -> dict[str, Any]:
    """The four terminal branches, decided by the mean floor and nothing else.

    ``parents[cid]`` needs ``mean_floor_eligible`` and ``d_g_of_mean_vector``.
    """
    eligible = sorted(cid for cid, p in parents.items()
                      if p.get("mean_floor_eligible"))
    if not eligible:
        decision, selected, tied = STOP_ITERATION_12, None, []
    elif len(eligible) == 1:
        decision, selected = SELECT_THE_ONLY_PARENT, eligible[0]
        tied = []
    else:
        ranked = sorted(eligible, key=lambda c: rs.rank_key(
            parents[c]["d_g_of_mean_vector"], c))
        decision, selected = SELECT_BY_MINIMUM_D_G, ranked[0]
        tied = [c for c in eligible if c != selected
                and parents[c]["d_g_of_mean_vector"] ==
                parents[selected]["d_g_of_mean_vector"]]
    return {
        "mean_eligible_parents": eligible,
        "num_mean_eligible": len(eligible),
        "decision": decision,
        "selected": selected,
        "tied_with": tied,
        "d_g_is_descriptive": len(eligible) == 1,
        "why_d_g_is_descriptive_when_one_parent_survives": (
            "with a single mean-eligible parent there is nothing to choose "
            "between, so its D_G describes the state and decides nothing"),
        "tie_break": "(distance_to_mg, candidate_id) ascending",
        "next_step": ITERATION_13 if selected else "none",
        "iteration_13_is_separately_frozen": True,
        "iteration_12_carries_no_confirmatory_claim": True,
    }


# ── the replicate set and its paths ─────────────────────────────────────

def replicate_id(parent: str, seed: int) -> str:
    return f"{parent}__s{seed}"


@dataclass(frozen=True)
class Stage3bReplicate:
    """One (B7 parent, seed) run.

    ``reused`` marks the seed-42 replicate, whose adapter and predictions were
    produced by Stage 3 and are READ from there; the three new seeds are
    trained into Stage 3b's own namespace and can never overwrite a Stage-3
    artifact.
    """

    replicate_id: str
    parent: str
    seed: int
    spec: s3g.Stage3CandidateSpec
    reused: bool

    @property
    def overrides(self) -> dict[str, Any]:
        """The parent's frozen overrides with this replicate's seed.

        The image-anchor weight, learning rate, epoch budget and group weights
        all come from the frozen Stage-3 spec, never retyped here, so a
        replicate cannot drift from the row it replicates.
        """
        out = dict(self.spec.overrides)
        out["seed"] = self.seed
        return out


def parent_specs(parents: Sequence[str],
                 grid: Sequence[s3g.Stage3CandidateSpec],
                 ) -> dict[str, s3g.Stage3CandidateSpec]:
    """Look the parents up in the frozen Stage-3 grid by id."""
    by_id = {c.candidate_id: c for c in grid}
    problems = []
    for p in parents:
        if p not in by_id:
            problems.append(f"{p} is not in the frozen Stage-3 grid")
        elif by_id[p].method != s3g.METHOD_ROUTE_ANCHOR:
            problems.append(
                f"{p} is method {by_id[p].method!r}; only B7 rows can be "
                f"Stage-3b parents")
    if problems:
        raise LookupError(
            "; ".join(problems) + " -- a replicate of a row that was never "
            "frozen as a B7 candidate is not a replicate")
    return {p: by_id[p] for p in parents}


def replicates_for(parents: Sequence[str],
                   grid: Sequence[s3g.Stage3CandidateSpec],
                   ) -> list[Stage3bReplicate]:
    """All (parent, seed) replicates, parents in ranked order, seeds ascending.

    Order is fixed by construction rather than by discovery, so a report's row
    order cannot depend on filesystem enumeration.
    """
    specs = parent_specs(parents, grid)
    return [Stage3bReplicate(replicate_id(p, seed), p, seed, specs[p],
                             reused=(seed == BASE_SEED))
            for p in parents for seed in ALL_SEEDS]


def to_train(replicates: Sequence[Stage3bReplicate]) -> list[Stage3bReplicate]:
    """The replicates that need a GPU: everything except the reused seed."""
    return [r for r in replicates if not r.reused]


def dataset_dir(repo_root: Path) -> Path:
    return s3g.dataset_dir(repo_root)


def stage3b_ckpt_root(repo_root: Path) -> Path:
    if not STAGE3B_CKPT_ROOT.startswith("data/") or "//" in STAGE3B_CKPT_ROOT:
        raise ValueError(
            f"{STAGE3B_CKPT_ROOT!r} is not a clean repo-relative data/ path")
    out = repo_root / STAGE3B_CKPT_ROOT
    for other in (s3g.stage3_ckpt_root(repo_root),
                  s2g.stage2_ckpt_root(repo_root),
                  s2g.stage1_ckpt_root(repo_root)):
        if out.resolve() == other.resolve():
            raise ValueError(
                f"{out} is an earlier stage's checkpoint root. Stage 3b writes "
                f"nowhere that filed adapters already live.")
    return out


def stage3b_predictions_dir(repo_root: Path) -> Path:
    out = dataset_dir(repo_root) / STAGE3B_PREDICTIONS_SUBDIR
    for other in (s3g.stage3_predictions_dir(repo_root),
                  s3g.stage2_predictions_dir(repo_root),
                  s3g.stage1_predictions_dir(repo_root),
                  dataset_dir(repo_root) / "predictions_iter12_seeds",
                  dataset_dir(repo_root) / "predictions"):
        if out.resolve() == other.resolve():
            raise ValueError(
                f"{out} is an earlier study's predictions directory. Stage 3b "
                f"reuses the seed-42 predictions by READING them; it never "
                f"writes over filed evidence.")
    return out


def adapter_dir(repo_root: Path, rep: Stage3bReplicate) -> Path:
    """Where a replicate's adapters live: reused from Stage 3, or new here."""
    if rep.reused:
        return s3g.stage3_ckpt_root(repo_root) / rep.parent / "adapters"
    return stage3b_ckpt_root(repo_root) / rep.replicate_id / "adapters"
