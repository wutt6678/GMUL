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
  quantity is a number of ENTITIES — and the ceiling on that number is per
  CLAIM, not the entity total.  The four target metrics are defined on the
  entities carrying a target association and the two retention metrics on
  those carrying a retain association; ``entity_role_census`` measures both
  from the frozen manifest, because sizing a claim against a pool of
  entities it cannot be computed on makes an infeasible margin look
  reachable.

* **The held-out-photograph stratum**, which is nested: probes inside
  species.  11R held exactly 30 iNaturalist species x 3 held-out
  photographs = 90 probes, and only those 30 species carry a target
  association at all.  Adding probes to a cluster buys precision only up to
  the between-cluster variance component, so a one-way variance decomposition
  decides whether the binding constraint is probes per cluster or the
  cluster count itself — and whether the cluster ceiling makes the claim
  unreachable at any number of probes.

The strata are powered separately because the ``probes_per_entity`` axis is
not the same lever in each: an MLLMU person has exactly one portrait, so a
new probe there is a new WORDING variant, whereas a new held-out probe is a
new PHOTOGRAPH.  The report records which is which per stratum, and measures
how many genuinely new photographs exist to be had, so that the recommended
size is one that can actually be built.

It then SELECTS one size.  ``confirmation_size`` names a single row of each
grid, the command line that builds it, and which identifiers are frozen now
versus which stage 3 must commit before scoring; a report that lists every
candidate and chooses none has sized nothing.

Every alpha in this module is ONE-SIDED and derived from the declared
``FAMILYWISE_ALPHA``, so Holm's thresholds and the sizing that quotes them
cannot disagree about which quantile they mean.

Nothing here scores anything or touches a GPU: it reads the committed
provenance-validated test predictions and the frozen dataset.
"""

from __future__ import annotations

import argparse
import hashlib
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

#: Familywise error rate for the declared primary family.  This is THE
#: declared alpha; every other level in this module is derived from it.
FAMILYWISE_ALPHA = 0.05

#: One-sided level for a single unadjusted claim: each arm of a TOST
#: equivalence test, a non-inferiority test, or a directional superiority
#: test.  Derived from FAMILYWISE_ALPHA rather than set beside it, because
#: the two are the same convention read two ways and a module that declares
#: both independently can silently disagree with itself.
ALPHA_ONE_SIDED = FAMILYWISE_ALPHA / 2

#: EVERY ``alpha`` parameter in this module is ONE-SIDED.  There is no
#: two-sided alpha argument anywhere: ``n_for_superiority`` takes the
#: one-sided level of the directional test it sizes, exactly as
#: ``n_for_one_sided_margin`` does.  A single mixed convention is how the
#: Holm block came to size a 0.025 one-sided threshold as though it were
#: two-sided, i.e. at one-sided 0.0125, and report 7/5 clusters where the
#: declared threshold supports 5/4.
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

#: Which entity ROLE a metric is defined on, and therefore which census row
#: supplies its cluster ceiling.  The four target metrics measure what the
#: adapters were trained to unlearn, so they exist only for entities with a
#: target association; the two retention metrics measure collateral damage on
#: associations that were kept, so they exist only for entities with a retain
#: association.  The two sets overlap but neither is the entity total.
METRIC_CLUSTER_ROLE = {
    "tga": "target",
    "filr": "target",
    "wrong_branch": "target",
    "over_forgetting": "target",
    "retain_same": "retain",
    "retain_other": "retain",
}

#: The declared primary family.  Two one-sided superiority claims, corrected
#: by Holm.  Retention and the M_G comparisons were removed from it by the
#: decisions recorded above, which is what makes the family small enough to
#: correct without weakening the claims that remain.
PRIMARY_FAMILY = ("B3_minus_B0:tga", "B3_minus_B0:filr")

#: The PRIMARY ESTIMAND those two claims are about.  Declared explicitly
#: because the family names are pooled while ``stratum_heterogeneity`` shows
#: the effect living in one stratum, and a preregistration that leaves the
#: reader to reconcile the two has not chosen an estimand.
#:
#: ``pooled_over_target_entities`` is the entity-macro average of the paired
#: B3-B0 difference over EVERY target entity - every person and every species
#: the preserved adapters were trained to unlearn - measured on that entity's
#: own confirmation probes.  It is the endpoint 11R reported, it is the
#: quantity the paper's sentence is about, and it does not discard the
#: held-out-photograph stratum.  The per-stratum decomposition is a
#: PRE-SPECIFIED SECONDARY DIAGNOSTIC: it is reported either way, it carries
#: no hypothesis test, and it does not enter the Holm family.
PRIMARY_ESTIMAND = "pooled_over_target_entities"
#: The strata whose probes enter the primary estimand.  Both do; that is what
#: makes it pooled rather than a stratum.
PRIMARY_ESTIMAND_STRATA = ("seen_photo_unseen_wording", "held_out_photo")
#: What the per-stratum numbers are, so no reader mistakes them for a second
#: primary claim.
STRATUM_ESTIMAND_STATUS = "prespecified_secondary_diagnostic_no_holm_entry"

#: Stated once, here, because it is true of BOTH candidate estimands and
#: therefore of whichever is declared primary.  The confirmation reuses the
#: entities the preserved adapters already unlearned; only the PROBES are
#: new.  A result on new probes over a fixed cohort is robustness to probe
#: construction, and calling it replication would claim a population the
#: design never sampled.
WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH = (
    "Reusing the same entities with new probes confirms robustness on that "
    "fixed cohort; it does not create new entity clusters and does not "
    "establish independent population-level replication. The preserved "
    "adapters unlearned a target set drawn from exactly these entities, so "
    "no confirmation split that keeps those adapters can enlarge the entity "
    "set. Both the pooled estimand and every per-stratum estimand inherit "
    "this limit; choosing between them does not escape it.")

# ---- the SELECTED confirmation size -------------------------------------
# A power report that lists every candidate and selects none has not sized
# anything: the grid is the input to a decision, not the decision.  These
# constants ARE the decision, and the report block built from them names the
# one row of the grid the confirmation will be built at.

#: NEW photographs per species, over and above the 12 pilot100_v2 already
#: allocated.  Selected, not derived: the held-out stratum carries no primary
#: claim, so its budget is set by the minimum detectable effect worth
#: reporting rather than by a power target.
CONFIRM_NEW_PHOTOS_PER_SPECIES = 12
#: ``--images-per-species`` for the re-fetch.  The seeded shuffle is
#: identical at the same seed, so drawing 24 makes the first 12 the
#: already-allocated photographs and the remaining 12 new by construction.
CONFIRM_FETCH_IMAGES_PER_SPECIES = (
    CONFIRM_NEW_PHOTOS_PER_SPECIES + 12)
CONFIRM_FETCH_SEED = 42
#: A NEW pool.  The default ``pilot_v1`` is refused by
#: ``fetch_inat_species.refuse_if_frozen_pool`` because the committed image
#: manifest pins 432 photographs under it.
CONFIRM_FETCH_OUT = "data/raw/inaturalist/confirm_v1"

#: NEW wording probes per target person.  The existing wording stratum is
#: unbalanced (27 persons at 3 probes, 12 at 6, 3 at 9); the confirmation is
#: balanced at 12 so the nested variance decomposition has equal cluster
#: sizes, and so the two strata share a probes-per-cluster value and are
#: directly comparable.  Power is not the binding constraint: at the
#: 42-person ceiling the between-person variance floor is reached well below
#: this, which the report states with its numbers rather than here.
CONFIRM_NEW_WORDING_PROBES_PER_PERSON = 12
#: The 12 are built as 3 probe families x 4 templates, mirroring the three
#: families the exploratory wording stratum used.
CONFIRM_WORDING_FAMILIES = 3
CONFIRM_NEW_TEMPLATES_PER_FAMILY = (
    CONFIRM_NEW_WORDING_PROBES_PER_PERSON // CONFIRM_WORDING_FAMILIES)

#: Every ``alpha`` above is one-sided; see FAMILYWISE_ALPHA.

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
    """Clusters needed for a ONE-SIDED directional test of theta.

    ``alpha`` is the one-sided level, matching every other sizing function
    in this module; the critical value is ``z(1 - alpha)``.  Callers that
    hold a two-sided level must halve it before calling, and the Holm block
    passes ``FAMILYWISE_ALPHA / k``, which is the worst-case one-sided
    threshold Holm applies to a single claim in a k-claim family.
    """
    if sd <= 0 or theta == 0:
        return None
    d = abs(theta) / sd
    return ((z(1 - alpha) + z(power)) / d) ** 2


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

    The critical value is the SAME for all three, and in this module it is
    always ``z(1 - ALPHA_ONE_SIDED) = z(0.975)``: a directional superiority
    test at one-sided 0.025 and a one-sided margin test at 0.025 use the
    same quantile.  So one critical value serves every kind and only the
    denominator moves.
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
                   claim_kind: str = "superiority",
                   cluster_ceiling: int | None = None,
                   ceiling_is: str | None = None) -> dict[str, Any]:
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
                sd, mu, ALPHA_ONE_SIDED, target),
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
    # The ceiling margin is reported BESIDE the observed one and each key
    # names the cluster count it was computed at.  Quoting a margin at the
    # observed count next to a field that says a larger count is "available"
    # reads as a margin at that larger count, which is how a 45-cluster
    # number came to be presented against a 100-entity ceiling.
    if cluster_ceiling:
        out["cluster_ceiling"] = cluster_ceiling
        out["cluster_ceiling_is"] = ceiling_is
        out["cluster_ceiling_saturated"] = cluster_ceiling == k
        for kind in ("equivalence", "non_inferiority"):
            c80 = margin_achievable(sd, mu, cluster_ceiling, ALPHA_ONE_SIDED,
                                    0.80, kind)
            c90 = margin_achievable(sd, mu, cluster_ceiling, ALPHA_ONE_SIDED,
                                    0.90, kind)
            out[f"min_margin_concludeable_at_the_{cluster_ceiling}"
                f"_cluster_ceiling_{kind}"] = {
                "power80": round(c80, 4) if c80 is not None else None,
                "power90": round(c90, 4) if c90 is not None else None,
            }
    return out


def _ceil_or_none(n: float | None) -> int | str | None:
    if n is None:
        return None
    if math.isinf(n):
        return "INFEASIBLE: the observed effect already exceeds the margin"
    return int(math.ceil(n))


def variance_components(groups: dict[str, list[float]]) -> dict[str, Any]:
    """One-way random-effects decomposition: probe level inside cluster.

    ``MSW`` estimates the within-cluster variance directly and ``MSB``
    estimates it plus m x between-cluster variance, so
    sigma2_between = (MSB - MSW) / m.  A negative estimate means the data
    show no detectable cluster-level structure; it is floored at zero and
    flagged, because reporting a negative variance component as though it
    were a measurement would understate the cluster count needed.

    The cluster is a SPECIES in the held-out stratum and a PERSON in the
    wording stratum, so every key here says ``cluster``.  Calling them all
    ``species`` is not a cosmetic slip: it is how a report came to size the
    wording stratum against 64 MLLMU entities when only 42 of them can carry
    a target metric.
    """
    ks = [k for k, v in groups.items() if v]
    if len(ks) < 2:
        return {"estimable": False, "reason": "fewer than two clusters"}
    sizes = {k: len(groups[k]) for k in ks}
    m_harmonic = len(ks) / sum(1.0 / n for n in sizes.values())
    grand = [v for k in ks for v in groups[k]]
    gmean = sum(grand) / len(grand)
    means = {k: sum(groups[k]) / len(groups[k]) for k in ks}
    ssb = sum(sizes[k] * (means[k] - gmean) ** 2 for k in ks)
    ssw = sum(sum((v - means[k]) ** 2 for v in groups[k]) for k in ks)
    dfb, dfw = len(ks) - 1, len(grand) - len(ks)
    if dfw <= 0:
        return {"estimable": False,
                "reason": "no within-cluster replication"}
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
        "num_clusters": len(ks),
        "num_probes": len(grand),
        "probes_per_cluster_min": min(sizes.values()),
        "probes_per_cluster_max": max(sizes.values()),
        "probes_per_cluster_harmonic_mean": round(m_harmonic, 3),
        #: Recorded because a harmonic mean below the arithmetic mean means
        #: the clusters are UNEQUAL, and an unequal design is a confound the
        #: between-cluster variance has to be read against.
        "probes_per_cluster_distribution": dict(sorted(
            {str(n): sum(1 for v in sizes.values() if v == n)
             for n in sorted(set(sizes.values()))}.items())),
        "probes_per_cluster_balanced": len(set(sizes.values())) == 1,
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
                     cluster_ceiling: int | None = None) -> dict[str, Any]:
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

    def _clusters_for(var_cluster: float) -> Any:
        """Clusters needed, or the reason the question has no answer."""
        if usable is None:
            return _NOT_ANSWERABLE[claim_kind].format(theta=theta,
                                                      margin=margin)
        if var_cluster <= 0:
            return 0
        return _ceil_or_none(zc * var_cluster / usable ** 2)

    def _probes(need: Any, m: int) -> int | None:
        return int(need) * m if isinstance(need, int) else None

    rows = []
    for m in probe_options:
        var_cluster = s2b + s2w / m
        need = _clusters_for(var_cluster)
        rows.append({
            "probes_per_entity": m,
            "cluster_variance_at_this_m": round(var_cluster, 6),
            "design_effect_vs_independent_probes": round(
                (s2b + s2w / m) / (total / m), 3) if total else None,
            "clusters_required": need,
            "total_probes_at_that_cluster_count": _probes(need, m),
        })
    asymp = _clusters_for(s2b)
    scenarios = []
    for icc in icc_overrides:
        scenarios.append({
            "assumed_icc": icc,
            "probes_per_entity": 12,
            "clusters_required": _clusters_for(total * (icc + (1 - icc) / 12)),
        })

    # The conservative plan sizes on the ICC UPPER confidence bound rather
    # than the point estimate, because a floored-at-zero point estimate from
    # a few dozen binary probes per cluster cannot distinguish absent
    # clustering from clustering too small to see.
    icc_plan = min(1.0, max(0.0, vc.get("icc_planning_value_upper_bound",
                                        vc["icc_point_estimate"])))
    conservative = []
    for m in probe_options:
        need = _clusters_for(total * (icc_plan + (1 - icc_plan) / m))
        conservative.append({
            "probes_per_entity": m,
            "clusters_required": need,
            "total_probes_at_that_cluster_count": _probes(need, m),
        })
    cons_asymptotic = _clusters_for(total * icc_plan)

    # The decision-ready table.  The cluster count is capped by how many
    # entities can carry the metric, so the negotiable quantity is probes
    # per cluster, and the question is not "how many clusters would I need"
    # but "what power do I actually get at the ceiling".  Sizing at the ICC
    # upper bound is the conservative column: a point estimate floored at
    # zero cannot rule out real cluster-level structure.
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
    #   zero, and that is the WRONG SIGN for a TGA superiority claim, so the
    #   power column there is computed against |theta| and describes a claim
    #   in the opposite direction.  The mde column is the only honest thing
    #   to size photographs against.
    #
    # power_at_true_difference_zero is retained beside them for the margin
    # claims, where it is the optimistic bound a reader may expect; for a
    # superiority claim it would be the type-I error rate, so it is None.
    at_ceiling = []
    if cluster_ceiling:
        crit = z(1 - alpha)
        for m in probe_options:
            row: dict[str, Any] = {"probes_per_entity": m}
            for tag, icc_val in (("point_icc", vc["icc_point_estimate"]),
                                 ("conservative_icc_upper", icc_plan)):
                var_cluster = total * (icc_val + (1 - icc_val) / m)
                se = math.sqrt(var_cluster / cluster_ceiling)
                pwr = power_at(se, claim_kind, theta, margin, alpha)
                mde = mde_at(se, claim_kind, theta, alpha, power)
                row[f"half_width_at_{cluster_ceiling}_clusters_{tag}"] = \
                    round(crit * se, 4)
                row[f"power_at_{cluster_ceiling}_clusters_{tag}"] = \
                    (round(pwr, 3) if pwr is not None else None)
                row[f"mde_at_{cluster_ceiling}_clusters_{tag}"] = \
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
        "cluster_ceiling": cluster_ceiling,
        "achieved_at_cluster_ceiling": at_ceiling,
        "grid": rows,
        "asymptotic_cluster_floor_infinite_probes": asymp,
        "icc_sensitivity_at_12_probes": scenarios,
        "icc_planning_value_used_for_conservative_grid": round(icc_plan, 4),
        "conservative_grid_at_icc_upper_bound": conservative,
        "conservative_cluster_floor_infinite_probes": cons_asymptotic,
        "interpretation": (
            "clusters_required stops falling as probes_per_entity grows "
            "because only the WITHIN-cluster component is divided by m; the "
            "asymptotic floor is what the between-cluster component alone "
            "demands and no number of probes per cluster can go below it. "
            "The cluster is a species in the held-out stratum and a person "
            "in the wording stratum, so 'probes' means new photographs in "
            "the first and new wordings in the second. The conservative grid "
            "repeats that calculation at the ICC upper confidence bound, "
            "which is the value a preregistration should plan on when the "
            "point estimate is floored at zero. Both grids divide by the "
            "claim's usable distance, not by the raw margin, so "
            "clusters_required is not comparable across rows of different "
            "claim_kind. achieved_at_cluster_ceiling is read at "
            f"{cluster_ceiling} clusters because that is how many entities "
            "can carry this metric, not because that is how many the "
            "dataset holds."),
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


def entity_role_census(data_dir: Path,
                       associations: list[Any]) -> dict[str, Any]:
    """Which entities can be a cluster for which metric, measured.

    A cluster ceiling is NOT "how many entities the dataset has".  Each
    paired metric is defined on a ROLE: the four target metrics need an
    entity carrying a target association, the two retention metrics need one
    carrying a retain association.  pilot100_v2 holds 100 entities, but only
    72 are targets and only 70 carry a retain association, so 100 is the
    ceiling of no claim at all.

    Assigning 100 to every claim is what let the report quote a retention
    margin computed at the 45 clusters ``retain_other`` actually observed
    beside a ceiling more than twice that size.  Read from the frozen
    manifest's own role lists rather than re-derived from query families,
    because the manifest is what the dataset fingerprint binds.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text())
    roles = {"target": set(manifest.get("target_association_ids") or []),
             "retain": set(manifest.get("retain_association_ids") or [])}
    entity_source = {a.entity_id: a.dataset for a in associations}
    by_role: dict[str, Any] = {}
    for role, aids in roles.items():
        ents = sorted({a.entity_id for a in associations
                       if a.association_id in aids})
        per_source: dict[str, int] = {}
        for e in ents:
            per_source[entity_source[e]] = per_source.get(
                entity_source[e], 0) + 1
        blob = "\n".join(ents).encode()
        by_role[role] = {
            "entity_ids": ents,
            "total": len(ents),
            "by_source": dict(sorted(per_source.items())),
            "entity_ids_sha256": hashlib.sha256(blob).hexdigest(),
            "associations": len(aids),
        }
    role_sets: dict[str, int] = {}
    per_entity: dict[str, set] = {}
    for a in associations:
        s = per_entity.setdefault(a.entity_id, set())
        for role, aids in roles.items():
            if a.association_id in aids:
                s.add(role)
    for eid, s in per_entity.items():
        key = f"{entity_source[eid]}/{'+'.join(sorted(s)) or 'none'}"
        role_sets[key] = role_sets.get(key, 0) + 1
    return {
        "entities_total": len(per_entity),
        "by_source": dict(sorted(
            {src: len({a.entity_id for a in associations if a.dataset == src})
             for src in {a.dataset for a in associations}}.items())),
        "role_sets": dict(sorted(role_sets.items())),
        "by_role": by_role,
        "read_from": "manifest.json target_association_ids / "
                     "retain_association_ids, joined to associations.parquet",
        "why_100_is_not_a_claim_ceiling": (
            "The dataset holds "
            f"{len(per_entity)} entities, but a claim can only be clustered "
            "over entities the metric is DEFINED on. The four target metrics "
            f"are defined on the {by_role['target']['total']} entities "
            "carrying a target association; the two retention metrics on the "
            f"{by_role['retain']['total']} carrying a retain association. "
            "Neither is the entity total, and quoting the total beside a "
            "margin computed at a metric's own cluster count makes the "
            "margin look more achievable than the design allows."),
    }


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

    # ---- cluster ceilings, measured per ROLE ----
    # Needed before anything is sized, because the cluster count is bounded
    # by how many entities can carry the metric and the sizing question is
    # what half-width that bound permits, not how many clusters to wish for.
    # The bound is NOT the entity total: the four target metrics exist only
    # for entities carrying a target association, so the wording stratum's
    # ceiling is the 42 TARGET persons rather than all 64 MLLMU entities and
    # the held-out stratum's is the 30 TARGET species rather than all 36.
    census = entity_role_census(data_dir, associations)
    entities_by_source: dict[str, int] = {}
    for a in associations:
        entities_by_source.setdefault(a.dataset, set()).add(a.entity_id)
    entities_by_source = {k: len(v) for k, v in entities_by_source.items()}
    n_entities_total = census["entities_total"]
    target_total = census["by_role"]["target"]["total"]
    target_by_source = census["by_role"]["target"]["by_source"]
    #: Which census row bounds each paired metric, and what that row counts.
    cluster_ceiling_of = {
        metric: {
            "ceiling": census["by_role"][METRIC_CLUSTER_ROLE[metric]]["total"],
            "ceiling_is": (
                f"entities carrying a {METRIC_CLUSTER_ROLE[metric]} "
                "association in the frozen pilot100_v2 manifest"),
        }
        for metric in METRIC_CLUSTER_ROLE
    }
    #: The strata are sized against the TARGET entities of their own source,
    #: because both strata are decompositions of the target metrics.
    stratum_cluster_ceiling = {
        "seen_photo_unseen_wording": target_by_source.get("mllmu_hier", 0),
        "held_out_photo": target_by_source.get("inaturalist", 0),
    }

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
                d["diffs"], label, kind,
                cluster_ceiling=cluster_ceiling_of[metric]["ceiling"],
                ceiling_is=cluster_ceiling_of[metric]["ceiling_is"]) | {
                "num_paired_rows": d["num_rows"],
                "row_micro_a": round(d["row_a"], 6),
                "row_micro_b": round(d["row_b"], 6),
            }

    # ---- held-out photograph stratum, nested in species ----
    # Cluster ceilings were measured above, per role: the number of species
    # is bounded by how many TARGET species exist, so the sizing question is
    # what half-width the ceiling permits, not how many species to wish for.

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
            "num_clusters": len({entity_of[by_qid[q].association_id]
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
        # The ceiling is the number of TARGET entities of this stratum's
        # source, not the number of entities that source contributes to the
        # dataset.  Both strata decompose the target metrics, so a
        # retain-only person or species cannot add a cluster here however
        # many probes it is given.
        block["cluster_ceiling"] = stratum_cluster_ceiling.get(stratum)
        block["cluster_ceiling_is"] = (
            f"TARGET {block['nesting_cluster_is']} entities in the frozen "
            f"pilot100_v2 manifest ({block['cluster_ceiling']} of the "
            f"{entities_by_source.get(block['source_datasets'][0], 0)} "
            f"{block['source_datasets'][0]} entities the dataset holds)")
        block["cluster_ceiling_saturated"] = (
            block["num_clusters"] == block["cluster_ceiling"])
        for pair in (("B3", "B0"), ("B3", "MG")):
            a, b = pair
            for metric in ("tga", "filr"):
                fa = {k: v for k, v in flags[a][metric].items() if k in qids}
                fb = {k: v for k, v in flags[b][metric].items() if k in qids}
                d = _paired_unit_diffs(fa, fb)
                if not d["keys"]:
                    continue
                per_cluster: dict[str, list[float]] = {}
                common = sorted(set(fa) & set(fb))
                for qid in common:
                    sp = entity_of[by_qid[qid].association_id]
                    per_cluster.setdefault(sp, []).append(
                        fa[qid][0] - fb[qid][0])
                vc = variance_components(per_cluster)
                kind = (CLAIM_KIND[metric] if b == "B0"
                        else CLAIM_KIND_VS_MG[metric])
                paired = summarize_pair(
                    d["diffs"], f"{a}_minus_{b}:{metric}", kind,
                    cluster_ceiling=stratum_cluster_ceiling.get(stratum),
                    ceiling_is=(
                        f"TARGET {block['nesting_cluster_is']} entities in "
                        "the frozen pilot100_v2 manifest"))
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
                        cluster_ceiling=stratum_cluster_ceiling.get(stratum)),
                }
                # The confirmation split cannot reuse ANY of these
                # photographs, so a second ceiling binds it: the number of
                # species a NEW photograph can be sourced for offline.  Sized
                # again there so the two are read together.
                if stratum == "held_out_photo" and supply_offline_species:
                    metric_entry[
                        "nested_sizing_at_offline_new_photo_ceiling"] = \
                        nested_size_grid(
                            vc, EQUIVALENCE_MARGIN, claim_kind=kind,
                            theta=paired["mean_diff"],
                            cluster_ceiling=supply_offline_species)
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
            "familywise_alpha": FAMILYWISE_ALPHA,
            "alpha_one_sided": ALPHA_ONE_SIDED,
            "alpha_convention": (
                f"FAMILYWISE alpha is {FAMILYWISE_ALPHA} and it is the only "
                "alpha declared. Every claim is ONE-SIDED and directional, "
                f"so a single unadjusted claim is tested at "
                f"{ALPHA_ONE_SIDED} = familywise/2 and Holm's worst-case "
                f"threshold for one claim in a k-claim family is "
                f"familywise/k. Every alpha argument in this module is "
                "one-sided and every sizing function uses z(1 - alpha); "
                "there is no two-sided alpha anywhere in it, so the "
                "thresholds and the sizing cannot disagree about which "
                "quantile they mean."),
            "power_targets": list(POWER_TARGETS),
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "resampling_unit": "entity (species or person) for pooled "
                               "claims; the stratum's own nesting cluster "
                               "(species or person) for the nested strata",
            "multiplicity": (
                "Holm over the declared primary family, applied to ONE-SIDED "
                f"p-values at familywise alpha {FAMILYWISE_ALPHA}. The worst "
                "case for any single claim is the smallest p, which Holm "
                "tests at familywise/k; the n_for_* columns below are "
                "computed at the UNADJUSTED one-sided alpha, so a k-claim "
                "family needs the familywise/k column, recomputed in the "
                "holm block AT THE SAME ONE-SIDED CONVENTION."),
        },
        "entity_role_census": census,
        "entity_clustered_claims": entity_results,
        "image_strata": strata,
        "new_photograph_supply": supply,
        "batch_layout_noise_floor": floor,
        "notes": [],
    }

    # ---- Holm budget over the declared primary family ----
    primary = list(PRIMARY_FAMILY)
    k = len(primary)
    #: Holm's worst case for a single claim in a k-claim family.  This is a
    #: ONE-SIDED threshold on a one-sided p-value, and it is the level the
    #: sizing below uses, so the declared threshold and the reported cluster
    #: requirement are the same test.
    holm_alpha = FAMILYWISE_ALPHA / k
    holm: dict[str, Any] = {
        "primary_family": primary,
        "k": k,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "holm_thresholds": [round(FAMILYWISE_ALPHA / (k - i), 6)
                            for i in range(k)],
        "thresholds_apply_to": "one-sided p-values",
        "worst_case_alpha_for_a_single_claim": round(holm_alpha, 6),
        "sizing_alpha_one_sided": round(holm_alpha, 6),
        "critical_value_z": round(z(1 - holm_alpha), 6),
        "thresholds_and_sizing_agree": (
            "both use z(1 - familywise/k); the sizing function takes a "
            "one-sided alpha like every other in this module, so the "
            "declared threshold and the cluster requirement below are the "
            "same test rather than one being a two-sided reading of the "
            "other"),
        "per_claim": {},
    }
    for name in primary:
        res = entity_results.get(name)
        if not res:
            continue
        sd, mu = res["sd_of_cluster_diffs"], res["mean_diff"]
        # Retention claims would be non-inferiority and sized against their
        # margin; TGA and FILR are one-sided superiority claims and are
        # sized against zero. Mixing the two would report the retention
        # requirement as unreachable (its mean difference is near zero by
        # design) and the superiority requirement as trivial.
        kind = res["claim_kind"]
        entry = {"claim_kind": kind,
                 "clusters_available_ceiling": res.get("cluster_ceiling"),
                 "clusters_observed_at_11r_size": res["num_clusters"]}
        for target in POWER_TARGETS:
            t = int(target * 100)
            if kind == "non_inferiority":
                n = n_for_one_sided_margin(
                    sd, mu, EQUIVALENCE_MARGIN,
                    holm_alpha, target, "non_inferior")
            else:
                n = n_for_superiority(sd, mu, holm_alpha, target)
            entry[f"n_at_holm_worst_case_alpha_power{t}"] = _ceil_or_none(n)
            entry[f"feasible_at_holm_worst_case_alpha_power{t}"] = (
                isinstance(entry[f"n_at_holm_worst_case_alpha_power{t}"], int)
                and entry[f"n_at_holm_worst_case_alpha_power{t}"]
                <= (res.get("cluster_ceiling") or 0))
        holm["per_claim"][name] = entry
    report["holm_primary_family"] = holm

    # ---- feasibility against the dataset's hard ceilings ----
    # The cluster count is not a free parameter: it is bounded by how many
    # entities can carry the metric. A required n above that bound is not
    # "collect more data", it is "this claim cannot be made on this dataset
    # at this margin", which is a preregistration decision rather than a
    # budget one. The bound is PER CLAIM, read off the role census: the
    # entity total is the ceiling of no claim at all.
    # Interpolated, not restated: the ICC quoted in the explanation is the
    # one the wording stratum actually produced.
    _wording_icc = (strata.get("seen_photo_unseen_wording", {})
                    .get("metrics", {})
                    .get("B3_minus_B0:tga", {})
                    .get("variance_components", {})
                    .get("icc_point_estimate"))
    ceilings = {
        "entities_total": n_entities_total,
        "entities_by_source": entities_by_source,
        "role_sets": census["role_sets"],
        "cluster_ceiling_by_metric_role": {
            role: {"total": census["by_role"][role]["total"],
                   "by_source": census["by_role"][role]["by_source"],
                   "metrics": sorted(m for m, r_ in
                                     METRIC_CLUSTER_ROLE.items()
                                     if r_ == role)}
            for role in ("target", "retain")
        },
        "why_the_entity_ceiling_is_hard": (
            "The confirmation phase preserves the SELECTED checkpoints, and "
            "those adapters unlearned a target set drawn from exactly these "
            f"{n_entities_total} entities. Adding an entity would change what "
            "the frozen adapters were trained to unlearn, so the "
            "confirmation would no longer measure the same intervention. A "
            "required cluster count above a claim's own ceiling is therefore "
            "not 'collect more data' but 'this claim cannot be made on these "
            "checkpoints at this margin'. Only the PROBES per entity are a "
            "free parameter, and for a claim whose variance is dominated by "
            f"the between-entity component (ICC {_wording_icc} on the "
            "wording stratum) extra probes buy quickly diminishing "
            "precision."),
        "why_the_ceiling_is_per_claim": census["why_100_is_not_a_claim_ceiling"],
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
        "entity_clusters_available_for_pooled_target_claims": target_total,
        "stratum_cluster_ceilings": stratum_cluster_ceiling,
    }
    feasibility: dict[str, Any] = {"ceilings": ceilings, "claims": {}}
    for name, r in entity_results.items():
        metric = name.split(":", 1)[1]
        k_avail = cluster_ceiling_of[metric]["ceiling"]
        entry: dict[str, Any] = {
            "clusters_observed_at_11r_size": r["num_clusters"],
            "clusters_available_ceiling": k_avail,
            "ceiling_is": cluster_ceiling_of[metric]["ceiling_is"],
            "ceiling_saturated": r["num_clusters"] == k_avail,
        }
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
        # Split by the cluster count each was computed at.  Both live under
        # one heading so neither can be read as the other: the observed-size
        # margin is a measurement of 11R, the ceiling margin is what the
        # confirmation could reach if every eligible entity were used.
        entry["min_margin_concludeable"] = {
            "at_11r_size": {k_: v for k_, v in r.items()
                            if k_.startswith("min_margin_concludeable_at_")
                            and "_cluster_ceiling_" not in k_},
            "at_the_cluster_ceiling": {k_: v for k_, v in r.items()
                                       if k_.startswith(
                                           "min_margin_concludeable_at_the_")
                                       and "_cluster_ceiling_" in k_},
        }
        feasibility["claims"][name] = entry
    report["feasibility"] = feasibility

    # ---- the declared primary estimand ----
    # PRIMARY_FAMILY names pooled claims while the decomposition below shows
    # the effect living in one stratum.  A preregistration that leaves the
    # reader to reconcile those two has not chosen an estimand, so the choice
    # is declared once, here, with what it does and does not establish.
    pooled_ceiling = cluster_ceiling_of["tga"]["ceiling"]
    wording_clusters = strata.get(
        "seen_photo_unseen_wording", {}).get("num_clusters", 0)
    report["primary_estimand"] = {
        "declared": PRIMARY_ESTIMAND,
        "definition": (
            "the entity-macro average of the paired B3-B0 difference over "
            f"all {pooled_ceiling} TARGET entities - every person and every "
            "species the preserved adapters were trained to unlearn - each "
            "entity measured on its own confirmation probes and weighted "
            "equally regardless of how many probes it contributes"),
        "unit_of_inference": "entity (person or species), percentile "
                             "bootstrap over entities",
        "entities_in_scope": pooled_ceiling,
        "entities_in_scope_by_source": dict(sorted(
            census["by_role"]["target"]["by_source"].items())),
        "entity_ids_sha256":
            census["by_role"]["target"]["entity_ids_sha256"],
        "strata_whose_probes_enter_it": list(PRIMARY_ESTIMAND_STRATA),
        "primary_family": list(PRIMARY_FAMILY),
        "primary_claim_kinds": {
            n: CLAIM_KIND[n.split(":")[1]] for n in PRIMARY_FAMILY},
        "multiplicity": (
            f"Holm over k = {len(PRIMARY_FAMILY)} at familywise alpha "
            f"{FAMILYWISE_ALPHA}, one-sided p-values"),
        "per_stratum_decomposition_status": STRATUM_ESTIMAND_STATUS,
        "what_it_establishes": (
            "that B3 changes the target metrics relative to B0 across the "
            "target entity set as a whole, on probes the exploratory "
            "pipeline never scored"),
        "what_it_does_not_establish":
            WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH,
        "why_pooled_rather_than_the_stratum_that_carries_the_effect": (
            f"The {wording_clusters}-person wording stratum is where the 11R "
            "effect lives, and the held-out stratum beside it measured an "
            "effect indistinguishable from zero. Pooling therefore DILUTES "
            "the estimate toward zero, so a rejection at the pooled level is "
            "conservative with respect to the stratum that carries it, while "
            "a rejection at the stratum level says nothing about the "
            "entities outside it. The pooled estimand is also the one 11R "
            "reported and the one the paper's sentence is about. Restricting "
            "the primary family to the favourable stratum after seeing the "
            "heterogeneity would be choosing the estimand the data favour, "
            "which is the same error as choosing a margin after seeing that "
            "the declared one cannot be reached."),
        "rejected": [
            {
                "estimand": "seen_photo_unseen_wording only",
                "entities_in_scope": wording_clusters,
                "why_rejected": (
                    "homogeneous effect and no dilution, but it discards the "
                    f"{pooled_ceiling - wording_clusters} target species from "
                    "the primary endpoint and it is the stratum selected "
                    "after the heterogeneity was measured"),
            },
            {
                "estimand": "both, as two Holm families",
                "why_rejected": (
                    f"doubles the multiplicity burden to k = "
                    f"{2 * len(PRIMARY_FAMILY)} over two estimands that are "
                    "not independent - the stratum is a subset of the pooled "
                    "one, computed from overlapping probes"),
            },
        ],
    }

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
                "num_clusters": p["num_clusters"],
                "mean_diff": p["mean_diff"],
                "ci95": [round(lo, 4), round(hi, 4)],
                "ci_excludes_zero": lo > 0 or hi < 0,
                "effect_direction_matches_the_claim": (
                    p["mean_diff"] > 0 if metric == "tga"
                    else p["mean_diff"] < 0),
                "claim_answerable_at_this_effect": g["claim_answerable"],
                "cluster_ceiling": g["cluster_ceiling"],
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
    def _min_margin(res: dict[str, Any], kind: str,
                    at: str = "observed") -> float | None:
        """Smallest concludeable margin recorded for this claim, at 80%.

        ``at='observed'`` reads the margin computed at the cluster count 11R
        actually produced; ``at='ceiling'`` the one computed at this claim's
        own cluster ceiling.  Selecting by the full suffix rather than by
        ``endswith(kind)`` because both keys are now present and a match on
        the kind alone would silently pick whichever came first.
        """
        frag = ("_cluster_ceiling_" if at == "ceiling"
                else "_clusters_")
        col = next((k_ for k_ in res
                    if k_.startswith("min_margin_concludeable_at")
                    and frag in k_ and k_.endswith(kind)), None)
        return res[col]["power80"] if col else None

    retain_floor = floor.get("max_abs_retain_delta")
    margin_claims = ("B3_minus_B0:retain_same", "B3_minus_B0:retain_other")
    per_margin_claim: dict[str, Any] = {}
    for name in margin_claims:
        res = entity_results.get(name, {})
        ceiling = res.get("cluster_ceiling") or 0
        need = _ceil_or_none(n_for_one_sided_margin(
            res["sd_of_cluster_diffs"], res["mean_diff"],
            EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, 0.80, "non_inferior"))
        per_margin_claim[name] = {
            "claim_kind_now": res.get("claim_kind"),
            "clusters_needed_at_delta0.05_power80_had_it_stayed_primary": need,
            "clusters_observed_at_11r_size": res.get("num_clusters"),
            "clusters_available_ceiling": ceiling,
            "ceiling_is": res.get("cluster_ceiling_is"),
            "ceiling_saturated": res.get("cluster_ceiling_saturated"),
            "feasible_at_delta0.05": isinstance(need, int)
            and need <= ceiling,
            "smallest_concludeable_margin_power80_at_11r_size": _min_margin(
                res, "non_inferiority"),
            "smallest_concludeable_margin_power80_at_the_ceiling":
                _min_margin(res, "non_inferiority", "ceiling"),
            "smallest_concludeable_margin_power80_at_the_entity_total": (
                round(margin_achievable(
                    res["sd_of_cluster_diffs"], res["mean_diff"],
                    n_entities_total, ALPHA_ONE_SIDED, 0.80,
                    "non_inferiority"), 4)),
            "batch_layout_floor_on_retain_metrics": retain_floor,
        }
    #: The binding precision bound is the WORST claim measured at ITS OWN
    #: ceiling, not at the observed count and not at the entity total.
    #: retain_same's ceiling is saturated - every entity that carries a
    #: retain association already contributes a cluster - so its observed
    #: margin cannot be improved by any confirmation split on these
    #: adapters, and it is the honest number to decide against.
    ceiling_bounds = [v["smallest_concludeable_margin_power80_at_the_ceiling"]
                      for v in per_margin_claim.values()]
    precision_bound = max((b for b in ceiling_bounds if b is not None),
                          default=None)
    observed_bounds = [
        v["smallest_concludeable_margin_power80_at_11r_size"]
        for v in per_margin_claim.values()]
    total_bounds = [
        v["smallest_concludeable_margin_power80_at_the_entity_total"]
        for v in per_margin_claim.values()]
    bounds = [b for b in (precision_bound, retain_floor) if b is not None]
    smallest = round(max(bounds), 4) if bounds else None
    infeasible = all(v["feasible_at_delta0.05"] is False
                     for v in per_margin_claim.values())
    #: The decision must survive the ceiling it is justified at.  Quoting a
    #: margin computed at one cluster count beside a ceiling naming another
    #: is how the previous version of this report justified a demotion with
    #: numbers that did not belong to it, so the same bound is recomputed at
    #: all three candidate counts and the verdict is read off the worst.
    sensitivity = {
        "at_the_11r_observed_counts": max(
            (b for b in observed_bounds if b is not None), default=None),
        "at_each_claim_own_ceiling": precision_bound,
        "at_the_entity_total": max(
            (b for b in total_bounds if b is not None), default=None),
        "independent_of_any_ceiling": retain_floor,
    }
    sensitivity["delta0.05_infeasible_under_every_candidate"] = all(
        b is None or b > EQUIVALENCE_MARGIN for b in sensitivity.values())
    report["retention_claim_decision"] = {
        "decision": "demoted_to_descriptive",
        "declared_margin": None,
        "what_is_still_reported": (
            "paired entity-clustered intervals for retain_same and "
            "retain_other with both averaging units, and their half-widths; "
            "no non-inferiority test and no entry in the Holm family"),
        "per_claim": per_margin_claim,
        "binding_argument": (
            f"The batch-layout noise floor on the retain metrics is "
            f"{retain_floor}, measured by scoring ONE checkpoint under two "
            f"batch layouts, and it already exceeds the declared "
            f"{EQUIVALENCE_MARGIN} margin. That bound does not depend on any "
            f"cluster count, so it holds before precision is even "
            f"considered; precision at each claim's own ceiling is the "
            f"second and weaker of the two."),
        "lower_bound_from_precision_at_the_cluster_ceilings": precision_bound,
        "lower_bound_from_batch_layout_noise": retain_floor,
        "smallest_margin_that_would_have_satisfied_both": smallest,
        "margin_that_would_have_been_declared": (
            math.ceil(smallest * 10) / 10 if smallest is not None else None),
        "sensitivity_to_the_ceiling_choice": sensitivity,
        "delta0.05_verdict": (
            f"infeasible for both retention claims at delta = "
            f"{EQUIVALENCE_MARGIN}: they would need "
            + ", ".join(
                f"{per_margin_claim[n]['clusters_needed_at_delta0.05_power80_had_it_stayed_primary']}"
                f" clusters against a ceiling of "
                f"{per_margin_claim[n]['clusters_available_ceiling']}"
                for n in margin_claims)
            + ", and the smallest margin either could conclude at its own "
              "ceiling still exceeds both the declared margin and the "
              "batch-layout noise floor"
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
        ceiling = res.get("cluster_ceiling") or 0
        need = _ceil_or_none(n_for_one_sided_margin(
            res["sd_of_cluster_diffs"], res["mean_diff"],
            EQUIVALENCE_MARGIN, ALPHA_ONE_SIDED, 0.80, "equivalence"))
        per_mg[name] = {
            "claim_kind_now": res.get("claim_kind"),
            "clusters_needed_at_delta0.05_power80_had_it_been_tested": need,
            "clusters_observed_at_11r_size": res.get("num_clusters"),
            "clusters_available_ceiling": ceiling,
            "ceiling_is": res.get("cluster_ceiling_is"),
            "ceiling_saturated": res.get("cluster_ceiling_saturated"),
            "feasible_at_delta0.05": isinstance(need, int)
            and need <= ceiling,
            "smallest_concludeable_margin_power80_at_11r_size": _min_margin(
                res, "equivalence"),
            "smallest_concludeable_margin_power80_at_the_ceiling":
                _min_margin(res, "equivalence", "ceiling"),
            "observed_effect": res.get("mean_diff"),
            "achieved_half_width_at_11r_size": res.get(
                "achieved_half_width_at_11r_size"),
        }
    mg_ceiling = cluster_ceiling_of["tga"]["ceiling"]
    mg_needs = [v["clusters_needed_at_delta0.05_power80_had_it_been_tested"]
                for v in per_mg.values()]
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
            f"at delta = {EQUIVALENCE_MARGIN} the required cluster counts are "
            f"{', '.join(str(n) for n in mg_needs)} against a hard ceiling of "
            f"{mg_ceiling} target entities, so a TOST could only ever return "
            "NOT CONCLUDED. Preregistering a test that cannot succeed spends "
            "Holm budget and weakens the thresholds for the claims that can."),
    }

    # ---- the preregistration decisions this analysis produced ----
    # Recorded here so the freeze in stage 2 and the probe construction in
    # stage 3 read them from one place, and so a later reader can see which
    # numbers were measured and which were chosen.
    chosen_supply = supply["seeded_refetch"]
    held_block = strata.get("held_out_photo", {})
    held_tga = held_block.get("metrics", {}).get("B3_minus_B0:tga", {})
    held_rows = {r["probes_per_entity"]: r for r in
                 held_tga.get("nested_sizing", {})
                 .get("achieved_at_cluster_ceiling", [])}
    #: Read the ceiling out of the block instead of naming it, so the key
    #: cannot keep pointing at a ceiling the analysis stopped using.
    held_ceiling = held_tga.get("nested_sizing", {}).get("cluster_ceiling")
    held_mde_key = f"mde_at_{held_ceiling}_clusters_conservative_icc_upper"
    held_out_mde_by_candidate = {
        m: row.get(held_mde_key) for m, row in sorted(held_rows.items())}
    report["preregistration_decisions"] = {
        "primary_estimand": PRIMARY_ESTIMAND,
        "primary_family": list(PRIMARY_FAMILY),
        "primary_claim_kinds": {
            n: CLAIM_KIND[n.split(":")[1]] for n in PRIMARY_FAMILY},
        "multiplicity": (
            f"Holm over k = {len(PRIMARY_FAMILY)} at familywise alpha "
            f"{FAMILYWISE_ALPHA}, one-sided p-values"),
        "decisions": [
            {
                "id": "primary_estimand",
                "question": "pooled over target entities, or the stratum "
                            "that carries the effect",
                "chosen": report["primary_estimand"]["definition"],
                "evidence": "primary_estimand",
                "rejected": [r_["estimand"]
                             for r_ in report["primary_estimand"]["rejected"]],
            },
            {
                "id": "confirmation_size",
                "question": "how many new probes, of each kind, per entity",
                "chosen": "see confirmation_size: it selects one row of each "
                          "grid and names the command line that builds it",
                "evidence": "confirmation_size",
                "rejected": [f"{m} new photographs per species" for m in
                             sorted(held_rows)
                             if m != CONFIRM_NEW_PHOTOS_PER_SPECIES],
            },
            {
                "id": "familywise_alpha",
                "question": "is familywise alpha 0.05 or 0.025",
                "chosen": (
                    f"{FAMILYWISE_ALPHA}, applied to one-sided p-values; a "
                    f"single unadjusted claim sits at {ALPHA_ONE_SIDED} and "
                    f"Holm's worst case for one of {len(PRIMARY_FAMILY)} "
                    f"claims is {FAMILYWISE_ALPHA / len(PRIMARY_FAMILY)}"),
                "evidence": "holm_primary_family",
                "rejected": [
                    "familywise 0.025",
                    "declaring a one-sided 0.025 threshold while sizing at a "
                    "two-sided 0.025, i.e. one-sided 0.0125, which is what "
                    "the previous revision did and which reported "
                    "requirements about 21% larger than the declared "
                    "threshold supports"],
            },
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
                    f"the {supply['pools']['pilot_v1']['num_photos']} "
                    "photographs pilot100_v2 uses must be a subset of the "
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
                "held_out_mde_by_candidate_probes_per_species":
                    held_out_mde_by_candidate,
                "held_out_mde_at_the_selected_design":
                    held_out_mde_by_candidate.get(
                        CONFIRM_NEW_PHOTOS_PER_SPECIES),
                "note_on_the_former_key_name": (
                    "this used to be held_out_mde_at_the_chosen_ceiling and "
                    "held every candidate at once; a field named for a choice "
                    "that lists all the alternatives has not recorded a "
                    "choice, so the grid is named as a grid and the selected "
                    "value is named beside it"),
            },
        ],
    }

    # ---- the SELECTED confirmation size ----
    # The grids above are the INPUT to a decision; this block is the
    # decision.  One row of each grid, the command line that builds it, the
    # totals it implies, and an explicit split between the identifiers that
    # can be frozen now and the ones stage 3 has to produce before they can
    # be.  A report that lists 3/6/9/12/24 photographs per species and
    # selects none has not sized anything, and the probe-construction stage
    # would then be free to pick a size after seeing the confirmation data.
    wording_block = strata.get("seen_photo_unseen_wording", {})
    wording_tga = wording_block.get("metrics", {}).get("B3_minus_B0:tga", {})
    wording_filr = wording_block.get("metrics", {}).get(
        "B3_minus_B0:filr", {})
    wording_ceiling = stratum_cluster_ceiling["seen_photo_unseen_wording"]
    wording_rows = {r["probes_per_entity"]: r for r in
                    wording_tga.get("nested_sizing", {})
                    .get("achieved_at_cluster_ceiling", [])}
    w_key = (f"mde_at_{wording_ceiling}_clusters_conservative_icc_upper")
    w_pow = (f"power_at_{wording_ceiling}_clusters_conservative_icc_upper")
    sel_w = wording_rows.get(CONFIRM_NEW_WORDING_PROBES_PER_PERSON, {})
    sel_h = held_rows.get(CONFIRM_NEW_PHOTOS_PER_SPECIES, {})
    target_ids = census["by_role"]["target"]["entity_ids"]
    retain_ids = census["by_role"]["retain"]["entity_ids"]
    exploratory_templates = sorted({q.template_id for q in queries
                                    if q.template_id})
    template_blob = "\n".join(exploratory_templates).encode()
    already_allocated = supply["pools"]["pilot_v1"]["photos_per_species"][0] \
        if supply["pools"]["pilot_v1"]["photos_per_species"] else 0
    report["confirmation_size"] = {
        "selected": True,
        "what_this_block_is": (
            "the single design the confirmation split will be built at. "
            "Every other row of every grid in this report is a rejected "
            "candidate, not an option left open"),
        "held_out_photographs": {
            "new_photographs_per_species": CONFIRM_NEW_PHOTOS_PER_SPECIES,
            "species_covered": chosen_supply["species_covered"],
            "new_photographs_total": (
                CONFIRM_NEW_PHOTOS_PER_SPECIES
                * chosen_supply["species_covered"]),
            "why_every_species_and_not_only_the_target_ones": (
                f"the target metrics only need the "
                f"{held_ceiling} TARGET species, but the retain_*_image "
                f"families probe the "
                f"{census['role_sets'].get('inaturalist/retain', 0)} "
                "retain-only species too, and a confirmation probe may not "
                "reuse any photograph pilot100_v2 already used"),
            "already_allocated_per_species": already_allocated,
            "fetch_images_per_species": CONFIRM_FETCH_IMAGES_PER_SPECIES,
            "fetch_seed": CONFIRM_FETCH_SEED,
            "fetch_out": CONFIRM_FETCH_OUT,
            "fetch_command": (
                f"python scripts/fetch_inat_species.py"
                f" --seed {CONFIRM_FETCH_SEED}"
                f" --images-per-species {CONFIRM_FETCH_IMAGES_PER_SPECIES}"
                f" --out {CONFIRM_FETCH_OUT}"),
            "target_species_in_the_held_out_stratum": held_ceiling,
            "mde_at_the_selected_design_conservative_icc": sel_h.get(
                held_mde_key),
            "power_at_the_selected_design_conservative_icc": sel_h.get(
                f"power_at_{held_ceiling}_clusters_conservative_icc_upper"),
            "what_the_power_number_means_here": (
                "low, and expected to be: this stratum carries no primary "
                "claim because 11R measured its effect as indistinguishable "
                "from zero, so the budget is set by the mde above rather "
                "than by a power target"),
            "rejected_candidates": {
                str(m): {"mde_conservative_icc": row.get(held_mde_key)}
                for m, row in sorted(held_rows.items())
                if m != CONFIRM_NEW_PHOTOS_PER_SPECIES},
        },
        "new_wording_probes": {
            "new_probes_per_target_person":
                CONFIRM_NEW_WORDING_PROBES_PER_PERSON,
            "target_persons": wording_ceiling,
            "new_wording_probes_total": (
                CONFIRM_NEW_WORDING_PROBES_PER_PERSON * wording_ceiling),
            "construction": (
                f"{CONFIRM_WORDING_FAMILIES} probe families x "
                f"{CONFIRM_NEW_TEMPLATES_PER_FAMILY} new templates each, "
                "every template new relative to pilot100_v2"),
            "exploratory_probes_per_person": wording_tga.get(
                "variance_components", {}).get(
                "probes_per_cluster_distribution"),
            "exploratory_design_was_balanced": wording_tga.get(
                "variance_components", {}).get("probes_per_cluster_balanced"),
            "balanced": True,
            "why_balanced": (
                "the exploratory wording stratum allocates different numbers "
                "of probes to different persons, so its harmonic-mean cluster "
                "size is below its arithmetic mean and its between-person "
                "variance estimate carries that imbalance; a balanced "
                "confirmation design removes a confound the variance "
                "decomposition would otherwise have to be read against"),
            "power_at_the_selected_design_tga_conservative_icc":
                sel_w.get(w_pow),
            "mde_at_the_selected_design_tga_conservative_icc":
                sel_w.get(w_key),
            "mde_at_the_selected_design_filr_conservative_icc": next(
                (r.get(f"mde_at_{wording_ceiling}_clusters_conservative_icc"
                        "_upper")
                 for r in wording_filr.get("nested_sizing", {})
                 .get("achieved_at_cluster_ceiling", [])
                 if r["probes_per_entity"]
                 == CONFIRM_NEW_WORDING_PROBES_PER_PERSON), None),
            "why_not_fewer": (
                "power is not the binding constraint at this ceiling: the "
                f"between-person variance floor is "
                f"{wording_tga.get('nested_sizing', {}).get('conservative_cluster_floor_infinite_probes')}"
                f" persons for TGA and "
                f"{wording_filr.get('nested_sizing', {}).get('conservative_cluster_floor_infinite_probes')}"
                f" for FILR against {wording_ceiling} available, so the count "
                "is chosen for design balance and comparability with the "
                "held-out stratum, not for power"),
            "rejected_candidates": {
                str(m): {"mde_conservative_icc": row.get(w_key),
                         "power_conservative_icc": row.get(w_pow)}
                for m, row in sorted(wording_rows.items())
                if m != CONFIRM_NEW_WORDING_PROBES_PER_PERSON},
        },
        "totals": {
            "new_target_probes": (
                CONFIRM_NEW_WORDING_PROBES_PER_PERSON * wording_ceiling
                + CONFIRM_NEW_PHOTOS_PER_SPECIES
                * chosen_supply["species_covered"]),
            "of_which_new_photographs": (
                CONFIRM_NEW_PHOTOS_PER_SPECIES
                * chosen_supply["species_covered"]),
            "of_which_new_wordings": (
                CONFIRM_NEW_WORDING_PROBES_PER_PERSON * wording_ceiling),
            "scored_states": 3,
            "entity_clusters": target_total,
        },
        "frozen_now": {
            "target_entity_ids": target_ids,
            "target_entity_ids_sha256":
                census["by_role"]["target"]["entity_ids_sha256"],
            "retain_entity_ids": retain_ids,
            "retain_entity_ids_sha256":
                census["by_role"]["retain"]["entity_ids_sha256"],
            "exploratory_template_ids": exploratory_templates,
            "exploratory_template_ids_sha256":
                hashlib.sha256(template_blob).hexdigest(),
            "exploratory_photograph_sha256_manifest":
                "data/mllmu_hier_pilot100/image_manifest.json",
        },
        "frozen_at_stage_3_before_any_scoring": {
            "confirmation_query_id_list_sha256": (
                "to be committed: the exact query_id list of the "
                "confirmation split"),
            "confirmation_template_ids_and_file_sha256": (
                "to be committed: the new template ids and the hash of the "
                "file that defines them"),
            "confirmation_photograph_sha256_manifest": (
                "to be committed: the hash of every new photograph, plus the "
                "licence and attribution the re-fetch records"),
            "collision_rules": [
                f"no confirmation template_id may appear among the "
                f"{len(exploratory_templates)} exploratory template ids "
                f"hashed above",
                "no confirmation photograph sha256 may appear in the frozen "
                "exploratory image manifest",
                "no confirmation query_id may appear in the exploratory "
                "queries parquet",
                "the confirmation split must not enter the reference-state "
                "gate, candidate selection, or any go/no-go decision",
            ],
            "why_these_cannot_be_frozen_here": (
                "they do not exist yet. Freezing a size and the entity set "
                "now, and binding the obligation to commit the identifiers "
                "before scoring, is what keeps the size a preregistration "
                "rather than a description of whatever stage 3 happens to "
                "build"),
        },
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
            f"{claim} of B3 over B0: the PRIMARY estimand is "
            f"{report['primary_estimand']['declared']} over "
            f"{report['primary_estimand']['entities_in_scope']} entities. As "
            f"a pre-specified secondary diagnostic, the effect is carried by "
            f"{', '.join(h['strata_that_carry_the_claim']) or 'NO stratum'}. "
            f"On held-out photographs the same comparison measured "
            f"{held.get('mean_diff', 0.0):+.4f} over "
            f"{held.get('num_clusters')} clusters with a 95% interval of "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}], {verdict}.")
    mdec = report["retention_claim_decision"]
    sens = mdec["sensitivity_to_the_ceiling_choice"]
    notes.append(
        f"Retention is DESCRIPTIVE, with no declared margin. "
        f"{mdec['binding_argument']} Had retention stayed a primary "
        f"non-inferiority claim, the verdict would be "
        f"{mdec['delta0.05_verdict']}. The "
        f"smallest concludeable margin is "
        f"{mdec['lower_bound_from_precision_at_the_cluster_ceilings']} at "
        f"each claim's OWN cluster ceiling, "
        f"{sens['at_the_11r_observed_counts']} at 11R's observed counts, and "
        f"{sens['at_the_entity_total']} at the {n_entities_total}-entity "
        f"total an earlier revision of this report wrongly used as every "
        f"claim's ceiling. All three exceed {EQUIVALENCE_MARGIN}, so the "
        f"demotion does not depend on which ceiling is quoted "
        f"(infeasible_under_every_candidate="
        f"{sens['delta0.05_infeasible_under_every_candidate']}); the "
        f"narrowest margin satisfying both bounds at the correct ceilings "
        f"would have been "
        f"{mdec['smallest_margin_that_would_have_satisfied_both']}. Widening "
        f"the margin to fit the design was rejected; the intervals are "
        f"published instead, with no non-inferiority test and no Holm entry.")
    for name in ("B3_minus_MG:tga", "B3_minus_MG:filr"):
        res = entity_results.get(name, {})
        mg_entry = mgdec["per_claim"][name]
        need = mg_entry[
            "clusters_needed_at_delta0.05_power80_had_it_been_tested"]
        notes.append(
            f"{name}: DESCRIPTIVE only, no equivalence test. Equivalence to "
            f"M_G at delta = {EQUIVALENCE_MARGIN} would need {need} entity "
            f"clusters against a hard ceiling of "
            f"{mg_entry['clusters_available_ceiling']}, and the smallest "
            f"margin concludeable at 11R's "
            f"{res.get('num_clusters')} clusters is "
            f"{_min_margin(res, 'equivalence')}. This is the power analysis "
            f"the preregistration asked for before treating the claim as "
            f"anything but secondary; the interval and its half-width are "
            f"published and equivalence is stated as NOT concluded.")
    cs = report["confirmation_size"]
    holm_n = {n_: v["n_at_holm_worst_case_alpha_power80"]
              for n_, v in holm["per_claim"].items()}
    notes.append(
        f"Primary family is {', '.join(PRIMARY_FAMILY)} under Holm with k = "
        f"{len(PRIMARY_FAMILY)} at familywise alpha {FAMILYWISE_ALPHA} on "
        f"ONE-SIDED p-values, so the worst-case threshold and the sizing "
        f"level are both {FAMILYWISE_ALPHA / len(PRIMARY_FAMILY)} and both "
        f"use z = {holm['critical_value_z']}. At that level the family needs "
        f"{holm_n} clusters against "
        f"{report['primary_estimand']['entities_in_scope']} available. An "
        f"earlier revision sized the 0.025 threshold as though it were "
        f"two-sided, i.e. one-sided 0.0125 at z = {round(z(1 - FAMILYWISE_ALPHA / len(PRIMARY_FAMILY) / 2), 4)}, "
        f"which is conservative but not the declared test; the preregistration "
        f"now states the convention and the sizing obeys it.")
    notes.append(
        f"SELECTED confirmation size: "
        f"{cs['new_wording_probes']['new_probes_per_target_person']} new "
        f"wording probes on each of "
        f"{cs['new_wording_probes']['target_persons']} target persons "
        f"({cs['new_wording_probes']['new_wording_probes_total']} probes) and "
        f"{cs['held_out_photographs']['new_photographs_per_species']} new "
        f"photographs on each of "
        f"{cs['held_out_photographs']['species_covered']} species "
        f"({cs['held_out_photographs']['new_photographs_total']} photographs), "
        f"fetched by `{cs['held_out_photographs']['fetch_command']}`. The "
        f"primary claims are carried by the wording probes; the photographs "
        f"are for the secondary held-out measurement and for the retain "
        f"image families, not for a primary claim. Every other row of every "
        f"grid in this report is a rejected candidate.")
    notes.append(
        f"{report['primary_estimand']['what_it_does_not_establish']}")
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
    print(f"\n{'claim':34s} {'kind':16s} {'k':>4s} {'ceil':>5s} "
          f"{'mean':>8s} {'sd':>7s} "
          f"{'hw@11R':>7s} {'sup80':>6s} {'NI80':>6s} {'equiv80':>8s} "
          f"{'minNI@k':>8s} {'minNI@ceil':>10s}")

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
        nic = r["min_margin_concludeable_at_the_%d_cluster_ceiling_"
                "non_inferiority" % r["cluster_ceiling"]]["power80"] \
            if r.get("cluster_ceiling") else None
        print(f"{name:34s} {r['claim_kind']:16s} {r['num_clusters']:>4d} "
              f"{_s(r.get('cluster_ceiling')):>5s} "
              f"{r['mean_diff']:>8.4f} {r['sd_of_cluster_diffs']:>7.4f} "
              f"{r['achieved_half_width_at_11r_size']:>7.4f} "
              f"{_s(r['n_for_superiority_power80']):>6s} "
              f"{_s(r['n_for_non_inferiority_delta0.05_power80']):>6s} "
              f"{_s(r['n_for_equivalence_delta0.05_power80']):>8s} "
              f"{str(ni):>8s} {_s(nic):>10s}")
    print(f"\nentity ceilings: "
          f"{json.dumps(ceilings['entities_by_source'])}, total "
          f"{ceilings['entities_total']}")
    print(f"CLAIM cluster ceilings (the entity total is the ceiling of no "
          f"claim): "
          f"{json.dumps({role: v['total'] for role, v in ceilings['cluster_ceiling_by_metric_role'].items()})}")
    print(f"role sets: {json.dumps(ceilings['role_sets'])}")
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
              f"{block['num_clusters']} clusters "
              f"(one cluster = {block['nesting_cluster_is']}; "
              f"ceiling {block['cluster_ceiling']}, "
              f"saturated={block['cluster_ceiling_saturated']}; "
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
                      f"clusters point-ICC="
                      f"{_s(row['clusters_required']):>5s} "
                      f"conservative={_s(cons['clusters_required']):>5s} "
                      f"probes={_s(cons['total_probes_at_that_cluster_count'])}")
            print(f"      asymptotic cluster floor: point-ICC="
                  f"{_s(g['asymptotic_cluster_floor_infinite_probes'])} "
                  f"conservative="
                  f"{_s(g['conservative_cluster_floor_infinite_probes'])}")
            if g.get("achieved_at_cluster_ceiling"):
                sc = g["cluster_ceiling"]
                print(f"      at the {sc}-cluster ceiling, for a "
                      f"{g['claim_kind']} claim at theta="
                      f"{g['theta_used']:+.4f}:")
                for row in g["achieved_at_cluster_ceiling"]:
                    print(
                        f"        m={row['probes_per_entity']:>3d} "
                        f"hw={row[f'half_width_at_{sc}_clusters_point_icc']:.4f}"
                        f"/{row[f'half_width_at_{sc}_clusters_conservative_icc_upper']:.4f} "
                        f"power={_s(row[f'power_at_{sc}_clusters_point_icc'])}"
                        f"/{_s(row[f'power_at_{sc}_clusters_conservative_icc_upper'])} "
                        f"mde={_s(row[f'mde_at_{sc}_clusters_point_icc'])}"
                        f"/{_s(row[f'mde_at_{sc}_clusters_conservative_icc_upper'])} "
                        f"(point/conservative) "
                        f"conservative_power_ge_80="
                        f"{_s(row['power_ge_80_conservative_icc_upper'])}")
            off_g = m.get("nested_sizing_at_offline_new_photo_ceiling")
            if off_g and off_g.get("achieved_at_cluster_ceiling"):
                osc = off_g["cluster_ceiling"]
                print(f"      at the {osc}-cluster OFFLINE new-photo "
                      f"ceiling, same claim (conservative ICC only):")
                for row in off_g["achieved_at_cluster_ceiling"]:
                    print(
                        f"        m={row['probes_per_entity']:>3d} "
                        f"hw={row[f'half_width_at_{osc}_clusters_conservative_icc_upper']:.4f} "
                        f"power={_s(row[f'power_at_{osc}_clusters_conservative_icc_upper'])} "
                        f"mde={_s(row[f'mde_at_{osc}_clusters_conservative_icc_upper'])}")
    print("\n=== the declared primary estimand ===")
    pe = report["primary_estimand"]
    print(f"  {pe['declared']}: {pe['definition']}")
    print(f"  entities in scope {pe['entities_in_scope']} "
          f"{json.dumps(pe['entities_in_scope_by_source'])}, "
          f"ids sha256 {pe['entity_ids_sha256'][:16]}...")
    print(f"  family {pe['primary_family']}; {pe['multiplicity']}")
    print(f"  per-stratum decomposition: "
          f"{pe['per_stratum_decomposition_status']}")
    print(f"  does NOT establish: {pe['what_it_does_not_establish']}")
    print("\n=== which stratum carries each B3-vs-B0 primary claim ===")
    for name, h in hetero.items():
        print(f"  {name} ({h['claim']}), pooled "
              f"{h['pooled_mean_diff']:+.4f} over "
              f"{h['pooled_num_clusters']} clusters")
        for st, v in h["per_stratum"].items():
            print(f"      {st:26s} k={v['num_clusters']:>3d} "
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
              f" clusters at delta={EQUIVALENCE_MARGIN} (ceiling "
              f"{v['clusters_available_ceiling']}, observed "
              f"{v['clusters_observed_at_11r_size']}, saturated="
              f"{v['ceiling_saturated']}), smallest concludeable margin "
              f"at 11R's size "
              f"{_s(v['smallest_concludeable_margin_power80_at_11r_size'])} "
              f"and at the ceiling "
              f"{_s(v['smallest_concludeable_margin_power80_at_the_ceiling'])}")
    print(f"  batch-layout noise floor on retain metrics: "
          f"{_s(mdec['lower_bound_from_batch_layout_noise'])} "
          f"(from {floor.get('source', 'n/a')}) - the ceiling-independent "
          f"bound")
    print(f"  sensitivity to the ceiling choice: {json.dumps(sens)}")
    print(f"  -> decision {mdec['decision']}, declared margin "
          f"{_s(mdec['declared_margin'])}; the narrowest margin that would "
          f"have satisfied both bounds was "
          f"{_s(mdec['smallest_margin_that_would_have_satisfied_both'])}")
    print("\n=== SELECTED confirmation size ===")
    hp, nw = cs["held_out_photographs"], cs["new_wording_probes"]
    print(f"  {hp['new_photographs_per_species']} new photographs x "
          f"{hp['species_covered']} species = "
          f"{hp['new_photographs_total']} photographs")
    print(f"    fetch: {hp['fetch_command']}")
    print(f"    target species in the held-out stratum: "
          f"{hp['target_species_in_the_held_out_stratum']}; mde at the "
          f"selected design (conservative ICC) "
          f"{_s(hp['mde_at_the_selected_design_conservative_icc'])}")
    print(f"  {nw['new_probes_per_target_person']} new wording probes x "
          f"{nw['target_persons']} target persons = "
          f"{nw['new_wording_probes_total']} probes")
    print(f"    construction: {nw['construction']}")
    print(f"    exploratory design was balanced="
          f"{nw['exploratory_design_was_balanced']} "
          f"{json.dumps(nw['exploratory_probes_per_person'])}")
    print(f"    mde at the selected design (conservative ICC) tga="
          f"{_s(nw['mde_at_the_selected_design_tga_conservative_icc'])} "
          f"filr={_s(nw['mde_at_the_selected_design_filr_conservative_icc'])}")
    print(f"  totals: {json.dumps(cs['totals'])}")
    print(f"  frozen now: {len(cs['frozen_now']['target_entity_ids'])} target "
          f"+ {len(cs['frozen_now']['retain_entity_ids'])} retain entity ids, "
          f"{len(cs['frozen_now']['exploratory_template_ids'])} exploratory "
          f"template ids")
    print(f"  frozen at stage 3: "
          f"{', '.join(k for k in cs['frozen_at_stage_3_before_any_scoring'])}")
    print("\n=== notes ===")
    for n in notes:
        print(f"  * {n}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
