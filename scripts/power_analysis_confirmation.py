"""Power analysis for the Iteration 11C confirmation split.

    python scripts/power_analysis_confirmation.py --tag pilot100

Iteration 11R scored 72 entity clusters and could not conclude equivalence
to M_G at delta = 0.05 for ANY state: three intervals (B0, M_F, B3) were at
least as wide as the margin, so those designs could not have concluded
equivalence even at a true difference of exactly zero.  A confirmation split
frozen at the same size would repeat that, so the size is derived here from
the cluster variances 11R actually produced rather than chosen by analogy.

Two separate structures are powered, because they have different binding
constraints:

* **Entity-clustered claims** (all six paired metrics on the pooled target
  and retain probes).  The resampling unit is the entity, so the required
  quantity is a number of ENTITIES.

* **The held-out-photograph stratum**, which is nested: probes inside
  species.  11R held exactly 30 iNaturalist species x 3 held-out
  photographs = 90 probes, and the dataset contains only 36 species in
  total.  Adding probes to a cluster buys precision only up to the
  between-cluster variance component, so a one-way variance decomposition
  decides whether the binding constraint is probes per cluster or the
  cluster count itself — and whether the cluster ceiling makes the claim
  unreachable at any number of probes.

The strata are powered separately because the ``probes_per_entity`` axis is
not the same lever in each: an MLLMU person has exactly one portrait, so a
new probe there is a new WORDING variant, whereas a new held-out probe is a
new PHOTOGRAPH.  The report records which is which per stratum, and measures
how many genuinely new photographs exist to be had, so that the recommended
size is one that can actually be built.

Nothing here scores anything or touches a GPU: it reads the committed
provenance-validated test predictions and the frozen dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.paired_ci import (
    PAIRED_METRICS,
    _paired_unit_diffs,
    row_flags,
)
from granunlearn.evaluation.prediction_provenance import (
    dataset_version,
    sha256_file,
)
from granunlearn.evaluation.reference_eval import (
    load_associations_parquet,
    load_predictions_parquet,
    load_queries_parquet,
)
from granunlearn.logging_utils import setup_logger

log = setup_logger("power_analysis_confirmation")

#: One-sided level for each arm of a TOST equivalence test, and for a
#: non-inferiority test.  Two-sided 0.05 superiority uses alpha/2 = 0.025,
#: which is the same quantile, so one constant serves both.
ALPHA_ONE_SIDED = 0.025
POWER_TARGETS = (0.80, 0.90)
#: The margin 11R prespecified for TGA equivalence against M_G.
EQUIVALENCE_MARGIN = 0.05

#: Fixed so the ICC bootstrap — and therefore the recommended split size —
#: is identical on every run.  A preregistration whose sample size moves
#: between runs of its own sizing script is not frozen.
BOOTSTRAP_SEED = 20260907

#: Probes per nesting cluster the grid is evaluated at.  What a "probe" IS
#: differs by stratum and is recorded per stratum in the report: for the
#: held-out stratum one probe is one NEW PHOTOGRAPH, for the wording stratum
#: one probe is one NEW WORDING VARIANT of the single portrait that person
#: has.  A single axis label covering both would send the probe-construction
#: stage looking for twelve portraits of a person who has one.
PROBE_OPTIONS = (3, 6, 9, 12, 24)

#: What kind of claim each metric carries, which decides which column of the
#: sizing table applies.  A margin-based column is meaningless for a claim
#: that is not a margin claim: FILR's primary claim is a REDUCTION, i.e.
#: one-sided superiority, so its "non-inferiority" cell is not a hard claim
#: that happens to be infeasible - it is a question that was never asked.
#: Reporting it as INFEASIBLE would understate the single most strongly
#: powered result in the analysis.
#:
#: The two retention metrics are DESCRIPTIVE by decision, not by measurement.
#: This analysis found that non-inferiority at delta = 0.05 needs more entity
#: clusters than the dataset contains, and that the smallest concludeable
#: margin exceeds the batch-layout noise floor.  Rather than declare a wider
#: margin, retention was demoted: it is reported with intervals and no
#: non-inferiority claim.  The measurements that forced this are computed
#: into ``retention_claim_decision`` below rather than quoted here, because
#: dropping a claim must not drop the evidence for why - and a figure written
#: into a comment is a figure that stops matching the tables beside it.
CLAIM_KIND = {
    "tga": "superiority",
    "filr": "superiority",
    "wrong_branch": "descriptive",
    "over_forgetting": "descriptive",
    "retain_same": "descriptive",
    "retain_other": "descriptive",
}
#: M_G comparisons are DESCRIPTIVE by decision.  Equivalence to M_G was to be
#: treated as secondary unless this power analysis supported it, and it does
#: not: the required cluster count is several times the hard entity ceiling,
#: and on held-out photographs the observed effect already exceeds the margin
#: at any sample size.  No equivalence test is run, so nothing here enters the
#: multiplicity family; the intervals are published with their half-widths and
#: the report states that equivalence is NOT concluded.  The figures are in
#: ``entity_clustered_claims`` and the notes, not here.
CLAIM_KIND_VS_MG = {"tga": "descriptive", "filr": "descriptive",
                    "wrong_branch": "descriptive",
                    "over_forgetting": "descriptive",
                    "retain_same": "descriptive",
                    "retain_other": "descriptive"}

#: The declared primary family.  Two one-sided superiority claims, corrected
#: by Holm.  Retention and the M_G comparisons were removed from it by the
#: decisions recorded above, which is what makes the family small enough to
#: correct without weakening the claims that remain.
PRIMARY_FAMILY = ("B3_minus_B0:tga", "B3_minus_B0:filr")

#: Why a sizing question has NO ANSWER, by claim kind.  Kept distinct from
#: "expensive" because the consequences differ: an expensive claim is a
#: budget decision, an unanswerable one must be dropped or restated in the
#: preregistration before any GPU time is spent.
_NOT_ANSWERABLE = {
    "superiority": (
        "NOT ANSWERABLE: a superiority claim is sized against its own "
        "effect, and the effect observed here is {theta:+.4f}. No cluster "
        "count gives power against an effect indistinguishable from zero; "
        "read the mde column for what this design could detect instead."),
    "non_inferiority": (
        "NOT ANSWERABLE: the observed effect {theta:+.4f} already sits "
        "outside the {margin} margin, so no cluster count concludes "
        "non-inferiority."),
    "equivalence": (
        "NOT ANSWERABLE: the observed effect {theta:+.4f} already exceeds "
        "the {margin} margin, so no cluster count concludes equivalence."),
    "descriptive": "NOT ANSWERABLE: a descriptive metric carries no claim.",
}

_Z = NormalDist()


def z(prob: float) -> float:
    return _Z.inv_cdf(prob)


def _mean_sd(values: list[float]) -> tuple[float, float, tuple | None]:
    """Mean, sample SD, and a 95% CI for the SD itself.

    The SD of a per-cluster paired difference is the quantity the whole
    sample-size calculation divides by, so its own uncertainty is reported
    rather than hidden: with k clusters the estimate carries k-1 degrees of
    freedom, and at the 30 species the held-out stratum offers that is loose
    enough to change the answer.  Uses (k-1)s^2/sigma^2 ~ chi^2_{k-1}.
    """
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0), 0.0, None
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return mu, 0.0, None
    df = n - 1
    lo = sd * math.sqrt(df / _chi2_quantile(df, 0.975))
    hi = sd * math.sqrt(df / _chi2_quantile(df, 0.025))
    return mu, sd, (lo, hi)


def _chi2_quantile(df: int, p: float) -> float:
    """Chi-square quantile by Wilson-Hilferty.

    scipy is deliberately not a dependency of the CPU-only CI environment,
    and this is accurate to well under a percent in the tail probabilities
    and degrees of freedom used here (df >= 29, p in {0.025, 0.975}).
    """
    zp = z(p)
    term = 1.0 - 2.0 / (9.0 * df) + zp * math.sqrt(2.0 / (9.0 * df))
    return df * term ** 3


def n_for_superiority(sd: float, theta: float, alpha: float,
                      power: float) -> float | None:
    """Clusters needed for a two-sided test of theta != 0."""
    if sd <= 0 or theta == 0:
        return None
    d = abs(theta) / sd
    return ((z(1 - alpha / 2) + z(power)) / d) ** 2


def n_for_one_sided_margin(sd: float, theta: float, margin: float,
                           alpha: float, power: float,
                           direction: str) -> float | None:
    """Clusters needed to conclude on one side of a margin.

    ``direction='non_inferior'`` tests H0: theta <= -margin, so the usable
    distance is theta + margin.  ``direction='equivalence'`` is the binding
    arm of a TOST pair, whose usable distance is margin - |theta|.  When the
    observed effect already sits outside the margin the required n is
    infinite: no sample size can conclude equivalence for a difference that
    is really there, which is the honest answer rather than a large number.
    """
    if sd <= 0:
        return None
    usable = (theta + margin if direction == "non_inferior"
              else margin - abs(theta))
    if usable <= 0:
        return math.inf
    return ((z(1 - alpha) + z(power)) * sd / usable) ** 2


def usable_distance(kind: str, theta: float, margin: float) -> float | None:
    """Distance the interval must resolve for THIS claim kind.

    Every sizing formula in this module is ``((z_crit + z_power) * sd /
    usable) ** 2``, and the only thing that differs between claim kinds is
    ``usable``:

        superiority      |theta|          the CI must exclude zero
        non_inferiority  theta + margin   the CI's low end must clear -margin
        equivalence      margin - |theta| the binding arm of the TOST pair
        descriptive      None             there is no claim to power

    ``None`` means the question is not answerable rather than expensive: a
    superiority claim whose observed effect is exactly zero has nothing to
    detect, and a margin claim whose effect already sits outside the margin
    cannot be concluded at any sample size.  Returning a large number
    instead would read as "collect more data" when the honest answer is
    "this design cannot carry this claim".

    The critical value is the SAME for all three: two-sided 0.05 superiority
    uses z(1 - 0.05/2) = z(0.975), and one-sided 0.025 margin tests use
    z(1 - 0.025) = z(0.975).  So one quantile serves every kind and only
    the denominator moves.
    """
    if kind == "superiority":
        return abs(theta) if theta != 0 else None
    if kind == "non_inferiority":
        usable = theta + margin
    elif kind == "equivalence":
        usable = margin - abs(theta)
    else:
        return None
    return usable if usable > 0 else None


def power_at(se: float, kind: str, theta: float, margin: float,
             alpha: float = ALPHA_ONE_SIDED) -> float | None:
    """Power of the claim-appropriate test at standard error ``se``.

    ``Phi(usable / se - z_crit)`` for every kind, which is the textbook
    two-sided superiority power, the one-sided non-inferiority power, and
    the binding TOST arm's power respectively — see ``usable_distance``.
    """
    usable = usable_distance(kind, theta, margin)
    if usable is None or se <= 0 or not math.isfinite(se):
        return None
    return _Z.cdf(usable / se - z(1 - alpha))


def mde_at(se: float, kind: str, theta: float,
           alpha: float = ALPHA_ONE_SIDED, power: float = 0.80) -> float | None:
    """Smallest effect (or margin) this standard error can resolve.

    The inverse of ``power_at``, and the decision-relevant quantity when the
    effect itself is uncertain — which it is for the held-out photograph
    stratum, where 11R measured an effect indistinguishable from zero.  You
    cannot plan power at an effect you cannot estimate, but you CAN choose a
    design by the smallest effect it would detect and then ask whether that
    effect is scientifically interesting.

        superiority      the minimum detectable |true effect|
        non_inferiority  the smallest margin concludeable at this theta
        equivalence      the smallest margin concludeable at this theta
    """
    if se <= 0 or not math.isfinite(se) or kind == "descriptive":
        return None
    slack = (z(1 - alpha) + z(power)) * se
    if kind == "superiority":
        return slack
    if kind == "non_inferiority":
        return slack - theta
    return abs(theta) + slack


def half_width(sd: float, k: int, alpha: float = ALPHA_ONE_SIDED) -> float:
    """Achieved TOST-relevant half-width at k clusters (two-sided 95%)."""
    if k <= 0 or sd <= 0:
        return math.inf
    return z(1 - alpha) * sd / math.sqrt(k)


def icc_ci(groups: list[list[float]], m_harmonic: float,
           n_boot: int = 2000, seed: int = BOOTSTRAP_SEED,
           alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the one-way ICC, resampling SPECIES.

    An ICC point estimate of exactly zero is what a NEGATIVE between-cluster
    variance component floors to, and it is not evidence of absent
    clustering: with 30 species and three binary probes each the estimate
    cannot distinguish "no clustering" from "clustering we cannot see".
    Sizing a confirmation split on the floored point value would assume away
    the very structure the nested design exists to respect, so the upper
    bound is the conservative planning value.

    The interval is bootstrapped rather than taken from an F-quantile
    formula.  scipy is not a CI dependency, and the Wilson-Hilferty
    approximation to F(29,60) overstates the 0.975 quantile by roughly 30%
    against the tabulated 1.7697; that bias widens the ICC interval and
    inflates the planning value, which is the over-sizing this bound exists
    to prevent.  Resampling species needs no distributional approximation
    and is the same idiom the project already uses for paired CIs.

    The seed is fixed because this is preregistration evidence: two runs of
    the sizing script must recommend the same split.
    """
    k = len(groups)
    if k < 3 or m_harmonic <= 1:
        return 0.0, 1.0, 1.0
    rng = random.Random(seed)
    estimates: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        sample = [groups[rng.randrange(k)] for _ in range(k)]
        icc = _icc_from_groups(sample, m_harmonic)
        if icc is None:
            degenerate += 1
        else:
            estimates.append(icc)
    if len(estimates) < n_boot // 10:
        # Almost every resample was degenerate, so the interval would be
        # describing the resampling artefact rather than the clustering.
        return 0.0, 1.0, degenerate / n_boot
    estimates.sort()
    lo_i = max(0, int(math.floor(alpha / 2 * len(estimates))))
    hi_i = min(len(estimates) - 1,
               int(math.ceil((1 - alpha / 2) * len(estimates))) - 1)
    return estimates[lo_i], estimates[hi_i], degenerate / n_boot


def _icc_from_groups(groups: list[list[float]],
                     m_harmonic: float) -> float | None:
    """ICC floored at zero, or None when it is not estimable.

    ``None`` means UNDEFINED and the resample is dropped, which is not the
    same as 1.0.  Paired differences between two unlearning states are
    mostly zero — most probes do not change — so a bootstrap resample in
    which nothing varies at all is common, and scoring it as perfect
    clustering put the 97.5th percentile at 1.0 and made the conservative
    planning value useless.  Total variance of zero carries no information
    about how that zero is partitioned.
    """
    sizes = [len(g) for g in groups]
    if min(sizes) < 2 or len(groups) < 2:
        return None
    grand = [v for g in groups for v in g]
    gmean = sum(grand) / len(grand)
    means = [sum(g) / len(g) for g in groups]
    ssb = sum(n * (mu - gmean) ** 2 for n, mu in zip(sizes, means))
    ssw = sum(sum((v - mu) ** 2 for v in g) for g, mu in zip(groups, means))
    dfb, dfw = len(groups) - 1, len(grand) - len(groups)
    if dfw <= 0:
        return None
    msb, msw = ssb / dfb, ssw / dfw
    sigma2_b = max(0.0, (msb - msw) / m_harmonic)
    total = sigma2_b + msw
    if total <= 0:
        return None          # nothing varies anywhere: undefined, not 1.0
    return sigma2_b / total


def margin_achievable(sd: float, theta: float, k: int, alpha: float,
                      power: float, kind: str) -> float | None:
    """Smallest margin this design could conclude, at k clusters.

    The inverse of the sample-size question and the more useful one here:
    the dataset's cluster count is fixed by how many entities exist, so what
    is actually negotiable is the margin.  ``kind='equivalence'`` solves the
    TOST arm, ``'non_inferiority'`` the one-sided arm.
    """
    if sd <= 0 or k <= 0:
        return None
    slack = (z(1 - alpha) + z(power)) * sd / math.sqrt(k)
    return abs(theta) + slack if kind == "equivalence" else slack - theta


def summarize_pair(diffs: list[float], label: str,
                   claim_kind: str = "superiority") -> dict[str, Any]:
    mu, sd, sd_ci = _mean_sd(diffs)
    k = len(diffs)
    out: dict[str, Any] = {
        "comparison": label,
        "claim_kind": claim_kind,
        "num_clusters": k,
        "mean_diff": round(mu, 6),
        "sd_of_cluster_diffs": round(sd, 6),
        "sd_ci95": [round(sd_ci[0], 6), round(sd_ci[1], 6)] if sd_ci else None,
        "achieved_half_width_at_11r_size": round(half_width(sd, k), 6),
        "standardized_effect_dz": round(mu / sd, 4) if sd else None,
    }
    applicable = {
        "superiority": ("n_for_superiority_power{t}",),
        "non_inferiority": (
            "n_for_non_inferiority_delta{m}_power{t}",),
        "equivalence": ("n_for_equivalence_delta{m}_power{t}",),
        "descriptive": (),
    }.get(claim_kind, ())
    for target in POWER_TARGETS:
        t = int(target * 100)
        cells = {
            "n_for_superiority_power{t}": n_for_superiority(
                sd, mu, 2 * ALPHA_ONE_SIDED, target),
            "n_for_non_inferiority_delta{m}_power{t}": n_for_one_sided_margin(
                sd, mu, EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, target,
                "non_inferior"),
            "n_for_equivalence_delta{m}_power{t}": n_for_one_sided_margin(
                sd, mu, EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, target,
                "equivalence"),
        }
        for template, value in cells.items():
            key = template.format(t=t, m=EQUIVALENCE_MARGIN)
            out[key] = (_ceil_or_none(value) if template in applicable
                        else "n/a: not this claim's question")
    for kind in ("equivalence", "non_inferiority"):
        m80 = margin_achievable(sd, mu, k, ALPHA_ONE_SIDED, 0.80, kind)
        m90 = margin_achievable(sd, mu, k, ALPHA_ONE_SIDED, 0.90, kind)
        out[f"min_margin_concludeable_at_{k}_clusters_{kind}"] = {
            "power80": round(m80, 4) if m80 is not None else None,
            "power90": round(m90, 4) if m90 is not None else None,
        }
    return out


def _ceil_or_none(n: float | None) -> int | str | None:
    if n is None:
        return None
    if math.isinf(n):
        return "INFEASIBLE: the observed effect already exceeds the margin"
    return int(math.ceil(n))


def variance_components(groups: dict[str, list[float]]) -> dict[str, Any]:
    """One-way random-effects decomposition: probe level inside species.

    ``MSW`` estimates the within-species variance directly and ``MSB``
    estimates it plus m x between-species variance, so
    sigma2_between = (MSB - MSW) / m.  A negative estimate means the data
    show no detectable species-level clustering; it is floored at zero and
    flagged, because reporting a negative variance component as though it
    were a measurement would understate the species count needed.
    """
    ks = [k for k, v in groups.items() if v]
    if len(ks) < 2:
        return {"estimable": False, "reason": "fewer than two species"}
    sizes = {k: len(groups[k]) for k in ks}
    m_harmonic = len(ks) / sum(1.0 / n for n in sizes.values())
    grand = [v for k in ks for v in groups[k]]
    gmean = sum(grand) / len(grand)
    means = {k: sum(groups[k]) / len(groups[k]) for k in ks}
    ssb = sum(sizes[k] * (means[k] - gmean) ** 2 for k in ks)
    ssw = sum(sum((v - means[k]) ** 2 for v in groups[k]) for k in ks)
    dfb, dfw = len(ks) - 1, len(grand) - len(ks)
    if dfw <= 0:
        return {"estimable": False, "reason": "no within-species replication"}
    msb, msw = ssb / dfb, ssw / dfw
    sigma2_w = msw
    sigma2_b_raw = (msb - msw) / m_harmonic
    sigma2_b = max(0.0, sigma2_b_raw)
    total = sigma2_b + sigma2_w
    icc = sigma2_b / total if total > 0 else 0.0
    icc_lo, icc_hi, icc_degenerate = icc_ci([groups[k] for k in ks],
                                            m_harmonic)
    return {
        "estimable": True,
        "num_species": len(ks),
        "num_probes": len(grand),
        "probes_per_species_min": min(sizes.values()),
        "probes_per_species_max": max(sizes.values()),
        "probes_per_species_harmonic_mean": round(m_harmonic, 3),
        "ms_between": round(msb, 6),
        "ms_within": round(msw, 6),
        "sigma2_between_raw": round(sigma2_b_raw, 6),
        "sigma2_between_floored_at_zero": sigma2_b_raw < 0,
        "sigma2_between": round(sigma2_b, 6),
        "sigma2_within": round(sigma2_w, 6),
        "icc_point_estimate": round(icc, 4),
        "icc_ci95": [round(icc_lo, 4), round(icc_hi, 4)],
        "icc_bootstrap_degenerate_fraction": round(icc_degenerate, 4),
        "icc_planning_value_upper_bound": round(icc_hi, 4),
        "grand_mean": round(gmean, 6),
    }


def nested_size_grid(vc: dict[str, Any], margin: float,
                     claim_kind: str = "equivalence", theta: float = 0.0,
                     alpha: float = ALPHA_ONE_SIDED,
                     power: float = 0.80,
                     probe_options: tuple[int, ...] = PROBE_OPTIONS,
                     icc_overrides: tuple[float, ...] = (),
                     species_ceiling: int | None = None) -> dict[str, Any]:
    """Clusters x probes-per-cluster needed for a nested CI half-width.

    The half-width of a cluster-averaged paired difference measured over m
    probes per cluster is

        z * sqrt(sigma2_between + sigma2_within / m) / sqrt(S)

    so probes per cluster only divide the WITHIN component.  Once m is large
    the requirement converges on sigma2_between, which no number of
    additional probes can reduce — that is the ceiling this grid makes
    visible.  ``S`` is capped by how many clusters exist, so the negotiable
    quantity is m, and the grid reports both.

    ``margin`` is not the distance every claim must resolve; it is only the
    distance an EQUIVALENCE claim must resolve.  The denominator here is
    ``usable_distance(claim_kind, theta, margin)``, so a superiority claim is
    sized against its own effect and a non-inferiority claim against
    theta + margin.  Dividing by the raw margin for all four would size the
    two B3-vs-B0 superiority claims — effects roughly five times 0.05 — as
    though they needed equivalence-grade precision, overstating their
    requirement by about (margin / theta)^2.  The defaults reproduce the
    pure-margin question exactly (equivalence at theta = 0 has usable ==
    margin), which is what the sizing-primitive tests exercise.
    """
    if not vc.get("estimable"):
        return {"estimable": False, "reason": vc.get("reason")}
    s2b, s2w = vc["sigma2_between"], vc["sigma2_within"]
    total = s2b + s2w
    usable = usable_distance(claim_kind, theta, margin)
    zc = (z(1 - alpha) + z(power)) ** 2

    def _species_for(var_species: float) -> Any:
        """Clusters needed, or the reason the question has no answer."""
        if usable is None:
            return _NOT_ANSWERABLE[claim_kind].format(theta=theta,
                                                      margin=margin)
        if var_species <= 0:
            return 0
        return _ceil_or_none(zc * var_species / usable ** 2)

    def _probes(need: Any, m: int) -> int | None:
        return int(need) * m if isinstance(need, int) else None

    rows = []
    for m in probe_options:
        var_species = s2b + s2w / m
        need = _species_for(var_species)
        rows.append({
            "probes_per_entity": m,
            "species_variance_at_this_m": round(var_species, 6),
            "design_effect_vs_independent_probes": round(
                (s2b + s2w / m) / (total / m), 3) if total else None,
            "species_required": need,
            "total_probes_at_that_species_count": _probes(need, m),
        })
    asymp = _species_for(s2b)
    scenarios = []
    for icc in icc_overrides:
        scenarios.append({
            "assumed_icc": icc,
            "probes_per_entity": 12,
            "species_required": _species_for(total * (icc + (1 - icc) / 12)),
        })

    # The conservative plan sizes on the ICC UPPER confidence bound rather
    # than the point estimate, because a floored-at-zero point estimate from
    # a few dozen binary probes per cluster cannot distinguish absent
    # clustering from clustering too small to see.
    icc_plan = min(1.0, max(0.0, vc.get("icc_planning_value_upper_bound",
                                        vc["icc_point_estimate"])))
    conservative = []
    for m in probe_options:
        need = _species_for(total * (icc_plan + (1 - icc_plan) / m))
        conservative.append({
            "probes_per_entity": m,
            "species_required": need,
            "total_probes_at_that_species_count": _probes(need, m),
        })
    cons_asymptotic = _species_for(total * icc_plan)

    # The decision-ready table.  The species count is capped by how many
    # species exist, so the negotiable quantity is photographs per species,
    # and the question is not "how many species would I need" but "what
    # power do I actually get at the ceiling".  Sizing at the ICC upper
    # bound is the conservative column: a point estimate floored at zero
    # cannot rule out real species-level clustering.
    #
    # Three numbers per cell, because each answers a different question and
    # reporting only one has already produced a wrong conclusion here:
    #
    # * half_width — the precision the design buys, claim-independent.
    # * power — for THIS claim kind at the effect 11R observed.  It used to
    #   be TOST power at a true difference of zero for every row, which is
    #   the right question for an equivalence claim and the wrong one for a
    #   superiority claim: it reported 0.18 power for the MLLMU stratum's
    #   B3-vs-B0 TGA comparison, whose observed effect is +0.275, because it
    #   was asking whether that stratum could prove B3 EQUIVALENT to B0
    #   within 0.05 — a question the preregistration never asks.
    # * mde — the smallest effect (or margin) the cell resolves at the
    #   target power.  This is the column that survives an effect estimate
    #   too loose to plan against, which is the situation on the held-out
    #   photograph stratum: 11R measured -0.0111 with an interval spanning
    #   zero, so its power column is not answerable and its mde column is
    #   the only honest thing to size photographs against.
    #
    # power_at_true_difference_zero is retained beside them for the margin
    # claims, where it is the optimistic bound a reader may expect; for a
    # superiority claim it would be the type-I error rate, so it is None.
    at_ceiling = []
    if species_ceiling:
        crit = z(1 - alpha)
        for m in probe_options:
            row: dict[str, Any] = {"probes_per_entity": m}
            for tag, icc_val in (("point_icc", vc["icc_point_estimate"]),
                                 ("conservative_icc_upper", icc_plan)):
                var_species = total * (icc_val + (1 - icc_val) / m)
                se = math.sqrt(var_species / species_ceiling)
                pwr = power_at(se, claim_kind, theta, margin, alpha)
                mde = mde_at(se, claim_kind, theta, alpha, power)
                row[f"half_width_at_{species_ceiling}_species_{tag}"] = \
                    round(crit * se, 4)
                row[f"power_at_{species_ceiling}_species_{tag}"] = \
                    (round(pwr, 3) if pwr is not None else None)
                row[f"mde_at_{species_ceiling}_species_{tag}"] = \
                    (round(mde, 4) if mde is not None else None)
                row[f"power_ge_{int(power * 100)}_{tag}"] = \
                    (None if pwr is None else pwr >= power)
                if claim_kind in ("equivalence", "non_inferiority"):
                    row[f"power_at_true_difference_zero_{tag}"] = \
                        round(_Z.cdf(margin / se - crit), 3) if se > 0 else 1.0
                else:
                    row[f"power_at_true_difference_zero_{tag}"] = None
            at_ceiling.append(row)
    return {
        "estimable": True,
        "margin": margin,
        "claim_kind": claim_kind,
        "theta_used": round(theta, 6),
        "usable_distance": (round(usable, 6) if usable is not None else None),
        "claim_answerable": usable is not None,
        "not_answerable_reason": (
            None if usable is not None
            else _NOT_ANSWERABLE[claim_kind].format(theta=theta,
                                                    margin=margin)),
        "alpha_one_sided": alpha,
        "power": power,
        "species_ceiling": species_ceiling,
        "achieved_at_species_ceiling": at_ceiling,
        "grid": rows,
        "asymptotic_species_floor_infinite_probes": asymp,
        "icc_sensitivity_at_12_probes": scenarios,
        "icc_planning_value_used_for_conservative_grid": round(icc_plan, 4),
        "conservative_grid_at_icc_upper_bound": conservative,
        "conservative_species_floor_infinite_probes": cons_asymptotic,
        "interpretation": (
            "species_required stops falling as probes_per_entity grows "
            "because only the within-species component is divided by m; the "
            "asymptotic floor is what the between-species component alone "
            "demands and no number of photographs per species can go below "
            "it. The conservative grid repeats that calculation at the ICC "
            "upper confidence bound, which is the value a preregistration "
            "should plan on when the point estimate is floored at zero. "
            "Both grids divide by the claim's usable distance, not by the "
            "raw margin, so species_required is not comparable across rows "
            "of different claim_kind."),
    }


def _tracked_by_git(repo_root: Path, path: Path) -> bool | None:
    """Whether git tracks anything under ``path``, or None if git is absent.

    Measured rather than inferred from a reading of .gitignore, because the
    negations there are layered and the answer decides whether a reviewer
    can reproduce the pool measurement from the commit.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", str(path)], cwd=repo_root,
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def new_photograph_supply(repo_root: Path, associations: list[Any]) -> dict:
    """How many genuinely new held-out photographs exist to be had.

    The nested grid says what the held-out stratum NEEDS; this says what it
    can be GIVEN, which is the binding constraint on photographs per
    species.  Both pools on disk are measured by content hash rather than
    assumed, because "new" is a claim about bytes and the frozen set is
    already fully allocated:

    * ``pilot_v1`` — 36 species x 12 photographs, every one of which
      pilot100_v2 references (index 0 reserved for training, the rest cut
      into disjoint val/test pools).  It contributes ZERO new photographs.
    * ``local_v1`` — hash-disjoint from the frozen set, so genuinely new,
      but gitignored and carrying only an ``annotations.json`` with file
      names and taxonomy: no license, attribution, observation id or source
      URL.  A confirmatory claim resting on it would reproduce exactly the
      unverifiable-bytes gap Iteration 11R1 closed, so it is measured and
      reported as unusable rather than quietly drawn from.
    """
    frozen_roots = {"pilot_v1": repo_root / "data" / "raw" / "inaturalist"
                    / "pilot_v1" / "images"}
    referenced: set[str] = set()
    for a in associations:
        if a.dataset != "inaturalist":
            continue
        for ref in a.images:
            p = Path(ref.path)
            if not p.is_absolute():
                p = repo_root / ref.path
            if p.exists():
                referenced.add(sha256_file(p))

    def _pool(root: Path) -> dict[str, Any]:
        if not root.is_dir():
            return {"root": str(root.relative_to(repo_root)),
                    "present": False}
        species = sorted(p for p in root.iterdir() if p.is_dir())
        hashes: dict[str, set[str]] = {}
        for sp in species:
            hashes[sp.name] = {sha256_file(f) for f in sorted(sp.rglob("*"))
                               if f.is_file()}
        all_h = {h for s in hashes.values() for h in s}
        return {
            "root": str(root.relative_to(repo_root)),
            "present": True,
            "num_species": len(species),
            "num_photos": len(all_h),
            "photos_per_species": sorted({len(v) for v in hashes.values()}),
            "num_photos_referenced_by_pilot100_v2": len(all_h & referenced),
            "num_photos_disjoint_from_pilot100_v2": len(all_h - referenced),
            "species_with_a_disjoint_photo": sum(
                1 for v in hashes.values() if v - referenced),
            "tracked_by_git": _tracked_by_git(repo_root, root),
            # Stated per pool rather than left to be inferred, because this
            # report itself is committed while the photographs it measures
            # are not: a reviewer must be able to tell which numbers they can
            # reproduce from the commit and which they cannot.  This is the
            # same gap Iteration 11R1 closed for the prediction parquets.
            "reproducible_from_the_commit": bool(
                _tracked_by_git(repo_root, root)
                or (repo_root / "data" / "raw" / "inaturalist"
                    / root.parent.name / "PROVENANCE.json").exists()),
            "_hashes": all_h,
        }

    pools = {name: _pool(root) for name, root in frozen_roots.items()}
    pools["local_v1"] = _pool(repo_root / "data" / "raw" / "inaturalist"
                              / "local_v1" / "images")

    frozen = pools["pilot_v1"]
    offline = pools["local_v1"]
    prov = repo_root / "data" / "raw" / "inaturalist" / "local_v1" \
        / "PROVENANCE.json"
    license_fields = ("license_code", "attribution", "source_url",
                      "observation_id", "photo_id")
    offline_auditable = prov.exists()
    for name, p in pools.items():
        p["carries_license_provenance"] = (
            (repo_root / "data" / "raw" / "inaturalist" / name
             / "PROVENANCE.json").exists())

    species_new = offline.get("species_with_a_disjoint_photo", 0)
    per_species_new = (offline["photos_per_species"][0]
                       if offline.get("present")
                       and len(offline.get("photos_per_species", [])) == 1
                       else 0)
    out: dict[str, Any] = {
        "pools": {k: {kk: vv for kk, vv in v.items() if kk != "_hashes"}
                  for k, v in pools.items()},
        "frozen_pool_contributes_new_photos": 0,
        "frozen_pool_reason": (
            f"all {frozen.get('num_photos_referenced_by_pilot100_v2', 0)} of "
            f"{frozen.get('num_photos', 0)} pilot_v1 photographs are "
            "referenced by pilot100_v2: index 0 is reserved for training "
            "and the remainder are cut into disjoint val/test pools, so the "
            "existing 12 per species are fully allocated"),
        "offline_pool": {
            "num_new_photos": offline.get(
                "num_photos_disjoint_from_pilot100_v2", 0),
            "num_species_covered": species_new,
            "max_new_photos_per_species": per_species_new,
            "species_left_with_no_new_photo": max(
                0, (frozen.get("num_species", 0) or 0) - species_new),
            "has_license_and_attribution": offline_auditable,
            "missing_provenance_fields": ([] if offline_auditable
                                          else list(license_fields)),
            "tracked_by_git": offline.get("tracked_by_git"),
            "reproducible_from_the_commit": offline.get(
                "reproducible_from_the_commit", False),
            "usable_for_a_confirmatory_claim": bool(offline_auditable),
            "reason": (
                "content hashes are disjoint from every photograph "
                "pilot100_v2 uses, so these are genuinely new; but the pool "
                "is gitignored and records no license, attribution, "
                "observation id or source URL, so neither the bytes nor the "
                "CC licensing can be verified from the commit"),
        },
        "seeded_refetch": {
            "script": "scripts/fetch_inat_species.py",
            "network_required": True,
            "species_covered": frozen.get("num_species", 0),
            "photo_pool_queried_per_species": "up to 3 pages x 200 "
                                              "observations, filtered to "
                                              "licensed research-grade "
                                              "photos with a full size "
                                              "ladder",
            "records_license_and_attribution": True,
            "disjointness_mechanism": (
                "re-fetch at the SAME seed 42 with a larger "
                "--images-per-species: the seeded shuffle over the pool is "
                "then identical, so the first 12 of the longer draw are the "
                "already-allocated photographs and the remainder are new BY "
                "CONSTRUCTION. This must still be verified by content hash "
                "after the fetch, because the resolution gate replaces a "
                "rejected candidate with the next one in the seeded order "
                "and that path depends on what S3 serves."),
        },
        "achievable_ceilings_for_the_held_out_stratum": {
            "species_offline_with_auditable_provenance": (
                species_new if offline_auditable else 0),
            "species_offline_any_provenance": species_new,
            "species_via_seeded_refetch": frozen.get("num_species", 0),
            "max_new_photos_per_species_offline": per_species_new,
        },
    }
    return out


def batch_layout_floor(repo_root: Path, tag: str,
                       source: Path | None = None) -> dict[str, Any]:
    """The batch-layout noise floor, read from a committed report.

    Left-padded batched greedy decoding is not bit-stable across batch
    compositions, so the same checkpoint weights scored under two layouts
    differ by a measurable amount.  A retention margin below that amount
    could be decided by batch composition alone, which makes it the second
    of the two independent lower bounds on the margin.  It is read rather
    than restated here, because a literal in two places is how the two
    copies diverge from the measurement they came from.

    ``source`` defaults to 11R's report.  Stage 2b of the confirmation phase
    re-measures the floor on the confirmation split itself, and this is the
    knob that repoints the bound at that measurement instead of leaving the
    margin resting on a floor taken from a different split.
    """
    path = source or (repo_root / "data" / "reports"
                      / f"mllmu_{tag}_final_evaluation.json")
    if not path.exists():
        return {"available": False, "reason": f"not found: {path.name}"}
    block = (json.loads(path.read_text())
             .get("batch_composition_sensitivity") or {})
    return {
        "available": True,
        "source": (str(path.relative_to(repo_root))
                   if path.is_relative_to(repo_root) else str(path)),
        "max_abs_target_delta": block.get("max_abs_metric_delta"),
        "max_abs_retain_delta": block.get("max_abs_retain_delta"),
        "target_metrics": block.get("target_side_metrics"),
        "retain_metrics": block.get("retain_metrics"),
        "meaning": "same checkpoint weights under two batch layouts, so "
                   "this is decoding noise rather than a model difference",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Size the Iteration 11C confirmation split from 11R "
                    "cluster variances")
    parser.add_argument("--tag", default="pilot100")
    parser.add_argument("--b3-parquet", default=None)
    parser.add_argument("--b0-parquet", default=None)
    parser.add_argument("--mg-parquet", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--layout-floor-report", default=None,
        help="report to read the batch-layout noise floor from; defaults to "
             "the 11R final evaluation for this tag")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    data_dir = repo_root / "data" / f"mllmu_hier_{args.tag}"
    pred = data_dir / "predictions"

    queries = load_queries_parquet(data_dir / "queries.parquet")
    associations = load_associations_parquet(data_dir / "associations.parquet")
    by_qid = {q.query_id: q for q in queries}
    entity_of = {a.association_id: a.entity_id for a in associations}
    source_of = {a.association_id: a.dataset for a in associations}

    def _pick(explicit: str | None, pattern: str) -> Path:
        if explicit:
            return Path(explicit)
        hits = sorted(pred.glob(pattern))
        if not hits:
            raise SystemExit(f"no prediction parquet matching {pattern}")
        return hits[0]

    paths = {
        "B0": _pick(args.b0_parquet, "predictions_test_B0.parquet"),
        "B3": _pick(args.b3_parquet, "predictions_test_B3*.parquet"),
        "MG": _pick(args.mg_parquet, "predictions_test_MG.parquet"),
    }
    log.info("reading %s", {k: v.name for k, v in paths.items()})
    flags = {k: row_flags(load_predictions_parquet(v), queries, associations,
                          split="test")
             for k, v in paths.items()}

    # ---- entity-clustered claims ----
    entity_results: dict[str, Any] = {}
    for pair in (("B3", "B0"), ("B3", "MG")):
        a, b = pair
        for metric in PAIRED_METRICS:
            d = _paired_unit_diffs(flags[a][metric], flags[b][metric])
            if not d["keys"]:
                continue
            label = f"{a}_minus_{b}:{metric}"
            kind = (CLAIM_KIND[metric] if b == "B0"
                    else CLAIM_KIND_VS_MG[metric])
            entity_results[label] = summarize_pair(
                d["diffs"], label, kind) | {
                "num_paired_rows": d["num_rows"],
                "row_micro_a": round(d["row_a"], 6),
                "row_micro_b": round(d["row_b"], 6),
            }

    # ---- held-out photograph stratum, nested in species ----
    # Cluster ceilings are needed BEFORE the strata are sized: the number of
    # species is bounded by how many exist, so the sizing question is what
    # half-width the ceiling permits, not how many species to wish for.
    entities_by_source: dict[str, set] = {}
    for a in associations:
        entities_by_source.setdefault(a.dataset, set()).add(a.entity_id)
    species_ceiling_by_source = {k: len(v)
                                 for k, v in entities_by_source.items()}

    # ---- how many NEW photographs the held-out stratum can be given ----
    # Computed before the strata are sized because it lowers the ceiling the
    # held-out grid is read against: a confirmation probe may not reuse any
    # photograph pilot100_v2 already used, and every one of the 12 per
    # species is allocated.
    supply = new_photograph_supply(repo_root, associations)
    supply_offline_species = supply[
        "achievable_ceilings_for_the_held_out_stratum"][
            "species_offline_any_provenance"]

    # ---- the batch-layout noise floor, from a committed measurement ----
    floor = batch_layout_floor(
        repo_root, args.tag,
        Path(args.layout_floor_report) if args.layout_floor_report else None)

    held_out_qids = {
        q.query_id for q in queries
        if q.split == "test" and q.image_split == "test"
        and not (q.family or "").startswith("retain_") and not q.adversarial}
    seen_qids = {
        q.query_id for q in queries
        if q.split == "test" and q.image_split == "train"
        and not (q.family or "").startswith("retain_") and not q.adversarial}
    strata: dict[str, Any] = {}
    for stratum, qids in (("held_out_photo", held_out_qids),
                          ("seen_photo_unseen_wording", seen_qids)):
        block: dict[str, Any] = {
            "num_probes": len(qids),
            "num_species": len({entity_of[by_qid[q].association_id]
                                for q in qids if q in by_qid}),
            "source_datasets": sorted({
                source_of[by_qid[q].association_id]
                for q in qids if q in by_qid}),
            "metrics": {},
        }
        if stratum == "seen_photo_unseen_wording":
            block["nesting_cluster_is"] = "MLLMU person"
            block["probes_per_entity_axis_is"] = (
                "new WORDING variants per person. The portrait is fixed: an "
                "MLLMU person has exactly one photograph, so this stratum "
                "can never become a held-out-photograph stratum and no "
                "number of probes changes that.")
            reachable: int | None = None
        else:
            block["nesting_cluster_is"] = "iNaturalist species"
            block["probes_per_entity_axis_is"] = (
                "new PHOTOGRAPHS per species, one probe per photograph. "
                "Every one of the 12 photographs pilot100_v2 holds per "
                "species is already allocated, so a confirmation probe needs "
                "a photograph sourced from outside it.")
            reachable = supply["offline_pool"]["max_new_photos_per_species"]
        block["photos_per_entity_in_pilot100_v2"] = sorted({
            len(a.images) for a in associations if a.images
            and a.dataset == ("mllmu_hier" if stratum
                              == "seen_photo_unseen_wording"
                              else "inaturalist")})
        block["max_reachable_probes_per_entity"] = reachable
        block["probe_options_beyond_the_reachable_max"] = (
            [] if reachable is None
            else [m for m in PROBE_OPTIONS if m > reachable])
        for pair in (("B3", "B0"), ("B3", "MG")):
            a, b = pair
            for metric in ("tga", "filr"):
                fa = {k: v for k, v in flags[a][metric].items() if k in qids}
                fb = {k: v for k, v in flags[b][metric].items() if k in qids}
                d = _paired_unit_diffs(fa, fb)
                if not d["keys"]:
                    continue
                per_species: dict[str, list[float]] = {}
                common = sorted(set(fa) & set(fb))
                for qid in common:
                    sp = entity_of[by_qid[qid].association_id]
                    per_species.setdefault(sp, []).append(
                        fa[qid][0] - fb[qid][0])
                vc = variance_components(per_species)
                kind = (CLAIM_KIND[metric] if b == "B0"
                        else CLAIM_KIND_VS_MG[metric])
                paired = summarize_pair(
                    d["diffs"], f"{a}_minus_{b}:{metric}", kind)
                metric_entry: dict[str, Any] = {
                    "paired": paired,
                    "variance_components": vc,
                    # Sized at THIS stratum's own effect, not the pooled one.
                    # Sizing the held-out stratum at the pooled +0.198 would
                    # assume the effect is homogeneous across strata, which is
                    # precisely what the strata exist to test; the pooled
                    # number is itself dominated by the wording stratum.
                    "nested_sizing": nested_size_grid(
                        vc, EQUIVALENCE_MARGIN, claim_kind=kind,
                        theta=paired["mean_diff"],
                        icc_overrides=(0.0, 0.1, 0.2, 0.3, 0.5),
                        species_ceiling=species_ceiling_by_source.get(
                            block["source_datasets"][0]
                            if block["source_datasets"] else "", None)),
                }
                # The confirmation split cannot reuse ANY of these
                # photographs, so the species ceiling that binds it is the
                # number of species a NEW photograph can be sourced for —
                # not the 36 the frozen dataset happens to contain.  Sized a
                # second time at that ceiling so the two are read together.
                if stratum == "held_out_photo" and supply_offline_species:
                    metric_entry[
                        "nested_sizing_at_offline_new_photo_ceiling"] = \
                        nested_size_grid(
                            vc, EQUIVALENCE_MARGIN, claim_kind=kind,
                            theta=paired["mean_diff"],
                            species_ceiling=supply_offline_species)
                block["metrics"][f"{a}_minus_{b}:{metric}"] = metric_entry
        strata[stratum] = block

    report = {
        "analysis": "iteration_11c_confirmation_power",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "tag": args.tag,
        "dataset_version": dataset_version(data_dir),
        "source_predictions": {k: str(v.relative_to(repo_root))
                               for k, v in paths.items()},
        "design": {
            "alpha_one_sided": ALPHA_ONE_SIDED,
            "alpha_two_sided_equivalent": 2 * ALPHA_ONE_SIDED,
            "power_targets": list(POWER_TARGETS),
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "resampling_unit": "entity (species or person) for pooled "
                               "claims; species for the nested image strata",
            "multiplicity": (
                "Holm over the declared primary family. The worst case for "
                "any single claim is the smallest p, which Holm tests at "
                "alpha/k; the n_for_* columns below are computed at the "
                "UNADJUSTED alpha, so a k-claim family needs the "
                "alpha/k column, recomputed in the holm block."),
        },
        "entity_clustered_claims": entity_results,
        "image_strata": strata,
        "new_photograph_supply": supply,
        "batch_layout_noise_floor": floor,
        "notes": [],
    }

    # ---- Holm budget over the declared primary family ----
    primary = list(PRIMARY_FAMILY)
    k = len(primary)
    holm: dict[str, Any] = {
        "primary_family": primary,
        "k": k,
        "holm_thresholds": [round(ALPHA_ONE_SIDED * 2 / (k - i), 6)
                            for i in range(k)],
        "worst_case_alpha_for_a_single_claim": round(
            ALPHA_ONE_SIDED * 2 / k, 6),
        "per_claim": {},
    }
    for name in primary:
        res = entity_results.get(name)
        if not res:
            continue
        sd, mu = res["sd_of_cluster_diffs"], res["mean_diff"]
        # Retention claims are non-inferiority, so they are sized against
        # their margin; TGA and FILR are one-sided superiority claims and
        # are sized against zero. Mixing the two would report the retention
        # requirement as unreachable (its mean difference is near zero by
        # design) and the superiority requirement as trivial.
        kind = res["claim_kind"]
        entry = {"claim_kind": kind}
        for target in POWER_TARGETS:
            t = int(target * 100)
            if kind == "non_inferiority":
                n = n_for_one_sided_margin(
                    sd, mu, EQUIVALENCE_MARGIN,
                    ALPHA_ONE_SIDED / k, target, "non_inferior")
            else:
                n = n_for_superiority(sd, mu, 2 * ALPHA_ONE_SIDED / k,
                                      target)
            entry[f"n_at_holm_worst_case_alpha_power{t}"] = _ceil_or_none(n)
        holm["per_claim"][name] = entry
    report["holm_primary_family"] = holm

    # ---- feasibility against the dataset's hard ceilings ----
    # The cluster count is not a free parameter: it is bounded by how many
    # entities exist. A required n above that bound is not "collect more
    # data", it is "this claim cannot be made on this dataset at this
    # margin", which is a preregistration decision rather than a budget one.
    n_entities_total = len({a.entity_id for a in associations})
    # Interpolated, not restated: the ICC quoted in the explanation is the
    # one the wording stratum actually produced.
    _wording_icc = (strata.get("seen_photo_unseen_wording", {})
                    .get("metrics", {})
                    .get("B3_minus_B0:tga", {})
                    .get("variance_components", {})
                    .get("icc_point_estimate"))
    ceilings = {
        "entities_total": n_entities_total,
        "entities_by_source": species_ceiling_by_source,
        "why_the_entity_ceiling_is_hard": (
            "The confirmation phase preserves the SELECTED checkpoints, and "
            "those adapters unlearned a target set drawn from exactly these "
            f"{n_entities_total} entities. Adding an entity would change what "
            "the frozen adapters were trained to unlearn, so the "
            "confirmation would no longer measure the same intervention. A "
            f"required cluster count above {n_entities_total} is therefore "
            "not 'collect more data' but 'this claim cannot be made on these "
            "checkpoints at this margin'. Only the PROBES per entity are a "
            "free parameter, and for a claim whose variance is dominated by "
            f"the between-entity component (ICC {_wording_icc} on the "
            "wording stratum) extra probes buy quickly diminishing "
            "precision."),
        "what_is_a_free_parameter": (
            "probes per entity: new wording variants for the MLLMU persons, "
            "and new photographs for the iNaturalist species, bounded by "
            "new_photograph_supply"),
        "photos_per_inaturalist_species": sorted({
            len(a.images) for a in associations
            if a.dataset == "inaturalist" and a.images}),
        "photos_per_mllmu_person": sorted({
            len(a.images) for a in associations
            if a.dataset == "mllmu_hier" and a.images}),
        "entity_clusters_available_for_pooled_claims": max(
            (r["num_clusters"] for r in entity_results.values()), default=0),
    }
    feasibility: dict[str, Any] = {"ceilings": ceilings, "claims": {}}
    for name, r in entity_results.items():
        k_avail = n_entities_total
        entry: dict[str, Any] = {"clusters_available": k_avail}
        for col in ("n_for_superiority_power80",
                    "n_for_non_inferiority_delta0.05_power80",
                    "n_for_equivalence_delta0.05_power80"):
            need = r.get(col)
            entry[col] = {
                "required": need,
                "feasible_on_this_dataset": (
                    None if isinstance(need, str) and
                    need.startswith("n/a")
                    else isinstance(need, int) and need <= k_avail),
            }
        entry["min_margin_concludeable_at_11r_size"] = {
            k: v for k, v in r.items() if k.startswith("min_margin_")}
        feasibility["claims"][name] = entry
    report["feasibility"] = feasibility

    # ---- which stratum carries the primary claims ----
    # The pooled B3-vs-B0 effect is not the average of two similar strata,
    # and that decides the photograph question.  Photographs are worth
    # collecting only for a stratum that carries a claim; a stratum whose
    # effect is indistinguishable from zero cannot carry a superiority claim
    # at ANY photograph count, because additional photographs shrink the
    # standard error around an effect that is not there.  Computed rather
    # than asserted in prose so the conclusion stays checkable.
    hetero: dict[str, Any] = {}
    for metric in ("tga", "filr"):
        name = f"B3_minus_B0:{metric}"
        pooled = entity_results.get(name)
        if not pooled:
            continue
        per_stratum: dict[str, Any] = {}
        for stratum, block in strata.items():
            blk = block["metrics"].get(name)
            if not blk:
                continue
            p, g = blk["paired"], blk["nested_sizing"]
            hw = p["achieved_half_width_at_11r_size"]
            lo, hi = p["mean_diff"] - hw, p["mean_diff"] + hw
            per_stratum[stratum] = {
                "num_species": p["num_clusters"],
                "mean_diff": p["mean_diff"],
                "ci95": [round(lo, 4), round(hi, 4)],
                "ci_excludes_zero": lo > 0 or hi < 0,
                "effect_direction_matches_the_claim": (
                    p["mean_diff"] > 0 if metric == "tga"
                    else p["mean_diff"] < 0),
                "claim_answerable_at_this_effect": g["claim_answerable"],
                "species_ceiling": g["species_ceiling"],
            }
        carrying = sorted(
            s for s, v in per_stratum.items()
            if v["ci_excludes_zero"] and v["claim_answerable_at_this_effect"]
            and v["effect_direction_matches_the_claim"])
        hetero[name] = {
            "claim": ("TGA superiority of B3 over B0" if metric == "tga"
                      else "FILR reduction of B3 below B0"),
            "pooled_mean_diff": pooled["mean_diff"],
            "pooled_num_clusters": pooled["num_clusters"],
            "per_stratum": per_stratum,
            "strata_that_carry_the_claim": carrying,
            "strata_that_do_not": sorted(set(per_stratum) - set(carrying)),
        }
    report["stratum_heterogeneity"] = hetero

    # ---- the retention claim: why it was demoted ----
    # The margin decision was deferred to "re-measure the floor, then decide",
    # and the answer was to demote retention rather than widen the margin.
    # The two lower bounds are still computed here, and the cluster
    # requirement is computed COUNTERFACTUALLY - as though the claim had
    # stayed primary - because the demotion is a decision while the
    # infeasibility is the measurement that justified it.  Dropping the claim
    # must not drop the evidence, or a future reader cannot tell whether
    # retention was demoted for want of interest or for want of power.
    def _min_margin(res: dict[str, Any], kind: str) -> float | None:
        """Smallest concludeable margin recorded for this claim, at 80%."""
        col = next((k for k in res
                    if k.startswith("min_margin_concludeable_at_")
                    and k.endswith(kind)), None)
        return res[col]["power80"] if col else None

    retain_floor = floor.get("max_abs_retain_delta")
    margin_claims = ("B3_minus_B0:retain_same", "B3_minus_B0:retain_other")
    per_margin_claim: dict[str, Any] = {}
    for name in margin_claims:
        res = entity_results.get(name, {})
        need = _ceil_or_none(n_for_one_sided_margin(
            res["sd_of_cluster_diffs"], res["mean_diff"],
            EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, 0.80, "non_inferior"))
        per_margin_claim[name] = {
            "claim_kind_now": res.get("claim_kind"),
            "clusters_needed_at_delta0.05_power80_had_it_stayed_primary": need,
            "clusters_available": n_entities_total,
            "feasible_at_delta0.05": isinstance(need, int)
            and need <= n_entities_total,
            "smallest_concludeable_margin_power80": _min_margin(
                res, "non_inferiority"),
            "batch_layout_floor_on_retain_metrics": retain_floor,
        }
    precision_bounds = [v["smallest_concludeable_margin_power80"]
                        for v in per_margin_claim.values()]
    precision_bound = max((b for b in precision_bounds if b is not None),
                          default=None)
    bounds = [b for b in (precision_bound, retain_floor) if b is not None]
    smallest = round(max(bounds), 4) if bounds else None
    infeasible = all(v["feasible_at_delta0.05"] is False
                     for v in per_margin_claim.values())
    report["retention_claim_decision"] = {
        "decision": "demoted_to_descriptive",
        "declared_margin": None,
        "what_is_still_reported": (
            "paired entity-clustered intervals for retain_same and "
            "retain_other with both averaging units, and their half-widths; "
            "no non-inferiority test and no entry in the Holm family"),
        "per_claim": per_margin_claim,
        "lower_bound_from_precision_at_available_clusters": precision_bound,
        "lower_bound_from_batch_layout_noise": retain_floor,
        "smallest_margin_that_would_have_satisfied_both": smallest,
        "margin_that_would_have_been_declared": (
            math.ceil(smallest * 10) / 10 if smallest is not None else None),
        "delta0.05_verdict": (
            "infeasible for both retention claims: the cluster counts above "
            "exceed every entity this dataset contains, and the entity "
            "ceiling is hard because the preserved adapters unlearned a "
            "target set drawn from exactly these entities"
            if infeasible
            else "feasible for at least one retention claim"),
        "rejected_alternative": (
            f"declaring delta = "
            f"{math.ceil(smallest * 10) / 10 if smallest is not None else None}"
            " would have kept both claims primary and concludeable, but a "
            "margin chosen after seeing that 0.05 cannot be reached is a "
            "margin widened to fit the design, and a retention claim at that "
            "width does not support 'no collateral damage'"),
    }

    # ---- the M_G comparisons: why they are descriptive ----
    # Symmetric with the retention block.  The demotion is a decision; the
    # infeasibility that justified it stays queryable instead of living only
    # inside a note, so a reader can check the reason without parsing prose.
    per_mg: dict[str, Any] = {}
    for name in ("B3_minus_MG:tga", "B3_minus_MG:filr"):
        res = entity_results.get(name, {})
        need = _ceil_or_none(n_for_one_sided_margin(
            res["sd_of_cluster_diffs"], res["mean_diff"],
            EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, 0.80, "equivalence"))
        per_mg[name] = {
            "claim_kind_now": res.get("claim_kind"),
            "clusters_needed_at_delta0.05_power80_had_it_been_tested": need,
            "clusters_available": n_entities_total,
            "feasible_at_delta0.05": isinstance(need, int)
            and need <= n_entities_total,
            "smallest_concludeable_margin_power80": _min_margin(
                res, "equivalence"),
            "observed_effect": res.get("mean_diff"),
            "achieved_half_width_at_11r_size": res.get(
                "achieved_half_width_at_11r_size"),
        }
    mgdec = report["mg_equivalence_decision"] = {
        "decision": "secondary_descriptive_only",
        "equivalence_test_run": False,
        "in_the_holm_family": False,
        "what_is_still_reported": (
            "the paired interval against M_G with its half-width and both "
            "averaging units, plus an explicit statement that equivalence is "
            "NOT concluded"),
        "per_claim": per_mg,
        "reason": (
            f"at delta = {EQUIVALENCE_MARGIN} the required cluster count is "
            f"several times the hard entity ceiling of {n_entities_total}, so "
            "a TOST could only ever return NOT CONCLUDED. Preregistering a "
            "test that cannot succeed spends Holm budget and weakens the "
            "thresholds for the claims that can."),
    }

    # ---- the preregistration decisions this analysis produced ----
    # Recorded here so the freeze in stage 2 and the probe construction in
    # stage 3 read them from one place, and so a later reader can see which
    # numbers were measured and which were chosen.
    chosen_supply = supply["seeded_refetch"]
    held_tga = (strata.get("held_out_photo", {}).get("metrics", {})
                .get("B3_minus_B0:tga", {}))
    held_rows = {r["probes_per_entity"]: r for r in
                 held_tga.get("nested_sizing", {})
                 .get("achieved_at_species_ceiling", [])}
    report["preregistration_decisions"] = {
        "primary_family": list(PRIMARY_FAMILY),
        "primary_claim_kinds": {
            n: CLAIM_KIND[n.split(":")[1]] for n in PRIMARY_FAMILY},
        "multiplicity": f"Holm over k = {len(PRIMARY_FAMILY)}",
        "decisions": [
            {
                "id": "retention_margin",
                "question": "what margin for retention non-inferiority",
                "chosen": "demote retention to descriptive; declare no margin",
                "evidence": "retention_claim_decision",
                "rejected": [
                    f"declare delta = "
                    f"{report['retention_claim_decision']['margin_that_would_have_been_declared']}",
                    "keep delta = 0.05 and report both claims as "
                    "underpowered",
                ],
            },
            {
                "id": "b3_vs_mg",
                "question": "is B3-M_G equivalence primary, secondary, or "
                            "absent",
                "chosen": "secondary and descriptive only: publish the "
                          "interval and its half-width, run no equivalence "
                          "test, state that equivalence is NOT concluded",
                "evidence": "mg_equivalence_decision",
                "rejected": ["secondary at a wider delta",
                             "keep primary at delta = 0.05"],
            },
            {
                "id": "new_photograph_supply",
                "question": "which route and budget for held-out photographs",
                "chosen": ("seeded superset re-fetch: same seed with a larger "
                           "--images-per-species, taking the photographs "
                           "beyond the already-allocated ones, so the new set "
                           "carries full license and attribution provenance "
                           "and covers every species"),
                "route": chosen_supply,
                "species_covered": chosen_supply["species_covered"],
                "verification_required_before_freeze": (
                    "the frozen 432 photographs must be a subset of the "
                    "re-fetched set by SHA-256, and the remainder must be "
                    "disjoint from them; the seeded-superset argument is "
                    "defeated if the API's pool has grown or the resolution "
                    "gate rejected differently"),
                "rejected": [
                    f"the offline local_v1 pool "
                    f"({supply['offline_pool']['num_new_photos']} disjoint "
                    f"photographs, no license record)",
                    "a minimal offline allocation",
                    "no new photographs",
                ],
                "what_the_photographs_are_for": (
                    "precision on a SECONDARY held-out measurement, not a "
                    "primary claim: neither B3-vs-B0 claim is carried by this "
                    "stratum, so the budget is set by the minimum detectable "
                    "effect worth reporting"),
                "held_out_mde_at_the_chosen_ceiling": {
                    m: row.get("mde_at_36_species_conservative_icc_upper")
                    for m, row in sorted(held_rows.items())},
            },
        ],
    }

    # ---- notes: every figure interpolated, never restated ----
    # A prose summary that hardcodes its own numbers is how a report comes to
    # contradict the tables beside it, which is the defect Iteration 11R1
    # fixed in the equivalence narrative.  Each figure below is read from the
    # block it describes, so the notes cannot drift from the analysis.
    notes: list[str] = []
    for metric, claim in (("tga", "TGA superiority"),
                          ("filr", "FILR reduction")):
        h = hetero.get(f"B3_minus_B0:{metric}")
        if not h:
            continue
        held = h["per_stratum"].get("held_out_photo", {})
        ci = held.get("ci95") or [float("nan")] * 2
        # The verdict is read off the computed flag, not written as a
        # literal: a note that asserts "does not exclude zero" would keep
        # asserting it after the data stopped supporting it.
        verdict = (
            "which does not exclude zero, so no probe count the photograph "
            "supply permits powers it there. New photographs buy precision "
            "on a secondary and possibly null held-out measurement; the mde "
            "columns, not the power columns, are the honest basis for "
            "choosing how many"
            if not held.get("ci_excludes_zero") else
            "which does exclude zero, so this stratum carries a measurable "
            "effect and the probe count is set by its power columns")
        notes.append(
            f"{claim} of B3 over B0 is carried by "
            f"{', '.join(h['strata_that_carry_the_claim']) or 'NO stratum'}. "
            f"On held-out photographs the same comparison measured "
            f"{held.get('mean_diff', 0.0):+.4f} over "
            f"{held.get('num_species')} species with a 95% interval of "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}], {verdict}.")
    mdec = report["retention_claim_decision"]
    notes.append(
        f"Retention is DESCRIPTIVE, with no declared margin. Had it stayed a "
        f"primary non-inferiority claim, delta = {EQUIVALENCE_MARGIN} would "
        f"be {mdec['delta0.05_verdict']}; the smallest margin this design "
        f"could conclude is "
        f"{mdec['lower_bound_from_precision_at_available_clusters']} and the "
        f"batch-layout noise floor on retain metrics is "
        f"{mdec['lower_bound_from_batch_layout_noise']}, so the narrowest "
        f"margin satisfying both would have been "
        f"{mdec['smallest_margin_that_would_have_satisfied_both']}. Widening "
        f"the margin to fit the design was rejected; the intervals are "
        f"published instead, with no non-inferiority test and no Holm entry.")
    for name in ("B3_minus_MG:tga", "B3_minus_MG:filr"):
        res = entity_results.get(name, {})
        need = mgdec["per_claim"][name][
            "clusters_needed_at_delta0.05_power80_had_it_been_tested"]
        notes.append(
            f"{name}: DESCRIPTIVE only, no equivalence test. Equivalence to "
            f"M_G at delta = {EQUIVALENCE_MARGIN} would need {need} entity "
            f"clusters against a hard ceiling of {n_entities_total}, and the "
            f"smallest margin concludeable at 11R's "
            f"{res.get('num_clusters')} clusters is "
            f"{_min_margin(res, 'equivalence')}. This is the power analysis "
            f"the preregistration asked for before treating the claim as "
            f"anything but secondary; the interval and its half-width are "
            f"published and equivalence is stated as NOT concluded.")
    notes.append(
        f"Primary family is {', '.join(PRIMARY_FAMILY)} under Holm with k = "
        f"{len(PRIMARY_FAMILY)}. Both are carried by the "
        f"seen_photo_unseen_wording stratum, whose probes are new WORDING "
        f"variants over the same persons, so the primary claims require no "
        f"new photographs and no new entities.")
    notes.append(
        "This report is committed; the photographs it measures in "
        "new_photograph_supply mostly are not. pilot_v1 is re-fetchable and "
        "verifiable through its committed PROVENANCE.json, while the "
        "local_v1 pool is gitignored with no license record, so its "
        f"{supply['offline_pool']['num_new_photos']} hash-disjoint "
        "photographs are reproducible only on the machine that generated "
        "this file. See each pool's reproducible_from_the_commit.")
    report["notes"] = notes

    out = Path(args.output or repo_root / "data" / "reports"
               / f"mllmu_{args.tag}_confirmation_power.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    log.info("wrote %s", out)

    # ---- console summary ----
    print(f"\n{'claim':34s} {'kind':16s} {'k':>4s} {'mean':>8s} {'sd':>7s} "
          f"{'hw@11R':>7s} {'sup80':>6s} {'NI80':>6s} {'equiv80':>8s} "
          f"{'minNI@k':>8s}")

    def _s(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, str):
            if v.startswith("INFEASIBLE"):
                return "INFEAS"
            if v.startswith("NOT ANSWERABLE"):
                return "noans"
            return "n/a"
        return str(v)

    for name in sorted(entity_results):
        r = entity_results[name]
        ni = r["min_margin_concludeable_at_%d_clusters_non_inferiority"
              % r["num_clusters"]]["power80"]
        print(f"{name:34s} {r['claim_kind']:16s} {r['num_clusters']:>4d} "
              f"{r['mean_diff']:>8.4f} {r['sd_of_cluster_diffs']:>7.4f} "
              f"{r['achieved_half_width_at_11r_size']:>7.4f} "
              f"{_s(r['n_for_superiority_power80']):>6s} "
              f"{_s(r['n_for_non_inferiority_delta0.05_power80']):>6s} "
              f"{_s(r['n_for_equivalence_delta0.05_power80']):>8s} "
              f"{str(ni):>8s}")
    print(f"\nentity ceilings: "
          f"{json.dumps(ceilings['entities_by_source'])}, total "
          f"{ceilings['entities_total']}")
    off = supply["offline_pool"]
    ach = supply["achievable_ceilings_for_the_held_out_stratum"]
    print(f"new held-out photographs: the frozen pool contributes "
          f"{supply['frozen_pool_contributes_new_photos']}; the offline "
          f"local_v1 pool holds {off['num_new_photos']} hash-disjoint photos "
          f"over {off['num_species_covered']} species "
          f"(max {off['max_new_photos_per_species']}/species), "
          f"license-provenance={off['has_license_and_attribution']}, "
          f"tracked_by_git={off['tracked_by_git']}, usable_for_confirmation="
          f"{off['usable_for_a_confirmatory_claim']}")
    print(f"  -> held-out species ceilings: offline-auditable="
          f"{ach['species_offline_with_auditable_provenance']}, "
          f"offline-any={ach['species_offline_any_provenance']}, "
          f"seeded-refetch={ach['species_via_seeded_refetch']}")
    for stratum, block in strata.items():
        print(f"\n--- {stratum}: {block['num_probes']} probes over "
              f"{block['num_species']} clusters "
              f"(one cluster = {block['nesting_cluster_is']}; "
              f"{', '.join(block['source_datasets'])}) ---")
        print(f"    one probe = {block['probes_per_entity_axis_is']}")
        print(f"    max reachable probes/entity="
              f"{_s(block['max_reachable_probes_per_entity'])} "
              f"(grid rows beyond it: "
              f"{block['probe_options_beyond_the_reachable_max'] or 'none'})")
        for metric, m in block["metrics"].items():
            vc = m["variance_components"]
            if not vc.get("estimable"):
                print(f"  {metric}: variance components not estimable")
                continue
            g = m["nested_sizing"]
            print(f"  {metric}: ICC={vc['icc_point_estimate']} "
                  f"ci95={vc['icc_ci95']} "
                  f"(planning value {g['icc_planning_value_used_for_conservative_grid']})")
            print(f"      claim={g['claim_kind']} "
                  f"theta={g['theta_used']:+.4f} "
                  f"usable={g['usable_distance']}")
            if not g["claim_answerable"]:
                print(f"      {g['not_answerable_reason']}")
            for row, cons in zip(g["grid"],
                                 g["conservative_grid_at_icc_upper_bound"]):
                print(f"      m={row['probes_per_entity']:>3d} -> "
                      f"species point-ICC="
                      f"{_s(row['species_required']):>5s} "
                      f"conservative={_s(cons['species_required']):>5s} "
                      f"probes={_s(cons['total_probes_at_that_species_count'])}")
            print(f"      asymptotic species floor: point-ICC="
                  f"{_s(g['asymptotic_species_floor_infinite_probes'])} "
                  f"conservative="
                  f"{_s(g['conservative_species_floor_infinite_probes'])}")
            if g.get("achieved_at_species_ceiling"):
                sc = g["species_ceiling"]
                print(f"      at the {sc}-species ceiling, for a "
                      f"{g['claim_kind']} claim at theta="
                      f"{g['theta_used']:+.4f}:")
                for row in g["achieved_at_species_ceiling"]:
                    print(
                        f"        m={row['probes_per_entity']:>3d} "
                        f"hw={row[f'half_width_at_{sc}_species_point_icc']:.4f}"
                        f"/{row[f'half_width_at_{sc}_species_conservative_icc_upper']:.4f} "
                        f"power={_s(row[f'power_at_{sc}_species_point_icc'])}"
                        f"/{_s(row[f'power_at_{sc}_species_conservative_icc_upper'])} "
                        f"mde={_s(row[f'mde_at_{sc}_species_point_icc'])}"
                        f"/{_s(row[f'mde_at_{sc}_species_conservative_icc_upper'])} "
                        f"(point/conservative) "
                        f"conservative_power_ge_80="
                        f"{_s(row['power_ge_80_conservative_icc_upper'])}")
            off_g = m.get("nested_sizing_at_offline_new_photo_ceiling")
            if off_g and off_g.get("achieved_at_species_ceiling"):
                osc = off_g["species_ceiling"]
                print(f"      at the {osc}-species OFFLINE new-photo "
                      f"ceiling, same claim (conservative ICC only):")
                for row in off_g["achieved_at_species_ceiling"]:
                    print(
                        f"        m={row['probes_per_entity']:>3d} "
                        f"hw={row[f'half_width_at_{osc}_species_conservative_icc_upper']:.4f} "
                        f"power={_s(row[f'power_at_{osc}_species_conservative_icc_upper'])} "
                        f"mde={_s(row[f'mde_at_{osc}_species_conservative_icc_upper'])}")
    print("\n=== which stratum carries each B3-vs-B0 primary claim ===")
    for name, h in hetero.items():
        print(f"  {name} ({h['claim']}), pooled "
              f"{h['pooled_mean_diff']:+.4f} over "
              f"{h['pooled_num_clusters']} clusters")
        for st, v in h["per_stratum"].items():
            print(f"      {st:26s} k={v['num_species']:>3d} "
                  f"theta={v['mean_diff']:+.4f} "
                  f"ci=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
                  f"excludes_zero={str(v['ci_excludes_zero']):5s} "
                  f"direction_ok="
                  f"{str(v['effect_direction_matches_the_claim']):5s}")
        print(f"      carries the claim: "
              f"{', '.join(h['strata_that_carry_the_claim']) or 'NONE'}"
              f"   does not: "
              f"{', '.join(h['strata_that_do_not']) or 'none'}")
    print("\n=== retention: why it was demoted to descriptive ===")
    for name, v in mdec["per_claim"].items():
        print(f"  {name}: would need "
              f"{_s(v['clusters_needed_at_delta0.05_power80_had_it_stayed_primary'])}"
              f" clusters at delta=0.05 (available "
              f"{v['clusters_available']}), smallest concludeable margin "
              f"{_s(v['smallest_concludeable_margin_power80'])}")
    print(f"  batch-layout noise floor on retain metrics: "
          f"{_s(mdec['lower_bound_from_batch_layout_noise'])} "
          f"(from {floor.get('source', 'n/a')})")
    print(f"  -> decision {mdec['decision']}, declared margin "
          f"{_s(mdec['declared_margin'])}; the narrowest margin that would "
          f"have satisfied both bounds was "
          f"{_s(mdec['smallest_margin_that_would_have_satisfied_both'])}")
    print("\n=== notes ===")
    for n in notes:
        print(f"  * {n}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
