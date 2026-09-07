"""Iteration 11C preregistration: the power analysis that sizes the split.

These tests exist because a confirmation split frozen at the wrong size is
worse than no confirmation at all: it spends GPU hours producing an
interval that cannot support the claim it was built for, and it does so
after the protocol has been sealed.  Iteration 11R is the worked example —
72 entity clusters, and three of eight intervals against M_G were at least
as wide as delta = 0.05, so those designs could not have concluded
equivalence even at a true difference of exactly zero.

Two kinds of assertion live here.  The pure sizing functions are tested
against values derived by hand, because a wrong quantile silently resizes
the whole experiment.  The committed report is tested for its FEASIBILITY
conclusions, because those are what the protocol freeze depends on and a
future re-run that moves them must be noticed rather than absorbed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"
POWER_REPORT = REPORTS / "mllmu_pilot100_confirmation_power.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from power_analysis_confirmation import (  # noqa: E402
    ALPHA_ONE_SIDED,
    ALPHA_STANDALONE_ONE_SIDED,
    BOOTSTRAP_SEED,
    CALIBRATION_REPLICATES,
    CLAIM_KIND,
    CLAIM_KIND_VS_MG,
    CONFIRM_FETCH_IMAGES_PER_SPECIES,
    CONFIRM_FETCH_OUT,
    CONFIRM_FETCH_ROLE,
    CONFIRM_FETCH_SEED,
    CONFIRM_FETCH_TAG,
    CONFIRM_NEW_PHOTOS_PER_SPECIES,
    CONFIRM_NEW_TEMPLATES_PER_FAMILY,
    CONFIRM_NEW_WORDING_PROBES_PER_PERSON,
    CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY,
    CONFIRM_RETENTION_REJECTED_PROBES_PER_ENTITY,
    CONFIRM_RETENTION_ROUTE,
    CONFIRM_WORDING_FAMILIES,
    EQUIVALENCE_MARGIN,
    EXPLORATORY_IMAGE_MANIFEST,
    FAMILYWISE_ALPHA,
    HOLM_WORST_CASE_ALPHA,
    METRIC_CLUSTER_ROLE,
    N_PERMUTATIONS,
    PERMUTATION_SEED,
    PRIMARY_ESTIMAND,
    PRIMARY_ESTIMAND_STRATA,
    PRIMARY_FAMILY,
    PROBE_OPTIONS,
    STRATUM_ESTIMAND_STATUS,
    WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH,
    _chi2_quantile,
    _mean_sd,
    entity_role_census,
    icc_ci,
    margin_achievable,
    mde_at,
    n_for_one_sided_margin,
    n_for_superiority,
    nested_size_grid,
    power_at,
    usable_distance,
    variance_components,
)


def _load() -> dict:
    if not POWER_REPORT.exists():
        pytest.skip(f"power report not present: {POWER_REPORT}")
    return json.loads(POWER_REPORT.read_text())


# ── the sizing primitives ─────────────────────────────────────────

class TestSizingPrimitives:
    def test_the_normal_quantiles_are_right(self):
        from power_analysis_confirmation import z
        assert abs(z(0.975) - 1.959964) < 1e-5
        assert abs(z(0.80) - 0.841621) < 1e-5
        assert abs(z(0.90) - 1.281552) < 1e-5

    def test_chi_square_quantile_brackets_the_median(self):
        """Wilson-Hilferty, because scipy is not a CI dependency."""
        df = 29
        lo = _chi2_quantile(df, 0.025)
        hi = _chi2_quantile(df, 0.975)
        assert lo < df < hi
        # tabulated values for df=29: 16.047 and 45.722
        assert abs(lo - 16.047) / 16.047 < 0.02
        assert abs(hi - 45.722) / 45.722 < 0.02

    def test_the_icc_bootstrap_interval_is_ordered_and_bounded(self):
        """The interval is bootstrapped over species rather than taken from
        an F-quantile formula, whose Wilson-Hilferty approximation to
        F(29,60) overstates the 0.975 quantile by ~30% and so inflated the
        conservative planning value from 0.07 to 0.55 — enough to turn a
        feasible design into an infeasible one."""
        groups = [[0.0, 1.0, 0.0] for _ in range(15)] + \
                 [[1.0, 1.0, 0.0] for _ in range(15)]
        lo, hi, degen = icc_ci(groups, 3.0, n_boot=400)
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.0 <= degen <= 1.0

    def test_the_bootstrap_detects_real_clustering(self):
        # every probe inside a species agrees, species differ sharply
        clustered = [[0.0, 0.0, 0.0] for _ in range(15)] + \
                    [[1.0, 1.0, 1.0] for _ in range(15)]
        lo, hi, _ = icc_ci(clustered, 3.0, n_boot=400)
        assert lo > 0.5, (lo, hi)

    def test_the_bootstrap_does_not_invent_clustering(self):
        # identical alternating pattern in every species: none between them
        flat = [[0.0, 1.0, 0.0] for _ in range(30)]
        lo, hi, _ = icc_ci(flat, 3.0, n_boot=400)
        assert hi < 0.35, (lo, hi)

    def test_the_bootstrap_is_deterministic_under_the_frozen_seed(self):
        """A preregistration whose recommended size moves between runs of
        its own sizing script is not frozen."""
        groups = [[0.0, 1.0, 1.0] for _ in range(10)] + \
                 [[1.0, 0.0, 1.0] for _ in range(10)]
        first = icc_ci(groups, 3.0, n_boot=300, seed=BOOTSTRAP_SEED)
        second = icc_ci(groups, 3.0, n_boot=300, seed=BOOTSTRAP_SEED)
        assert first == second

    def test_too_few_species_returns_the_uninformative_interval(self):
        assert icc_ci([[0.0, 1.0]], 2.0) == (0.0, 1.0, 1.0)
        # one probe per species carries no within-species information
        assert icc_ci([[0.0], [1.0], [0.0], [1.0]], 1.0) == (0.0, 1.0, 1.0)

    def test_an_all_zero_difference_set_is_undefined_not_perfect(self):
        """Paired differences between two unlearning states are mostly zero,
        so an all-zero resample is common. Scoring it ICC = 1.0 put the
        bootstrap's 97.5th percentile at 1.0 and made the conservative
        planning value useless; zero total variance says nothing about how
        that zero is partitioned."""
        lo, hi, degen = icc_ci([[0.0, 0.0, 0.0] for _ in range(30)], 3.0,
                               n_boot=200)
        assert degen == 1.0
        assert (lo, hi) == (0.0, 1.0)

    def test_superiority_n_matches_the_textbook_formula(self):
        """n = ((z_{1-a} + z_power) / d_z)^2 with d_z = theta/sd, and ``a``
        is ONE-SIDED like every other alpha in the module."""
        sd, theta = 0.1576, 0.1982
        d = abs(theta) / sd
        want = ((1.959964 + 0.841621) / d) ** 2
        assert abs(n_for_superiority(sd, theta, ALPHA_ONE_SIDED, 0.80)
                   - want) < 1e-6

    def test_the_one_sided_convention_is_the_only_one(self):
        """The regression behind review finding 4.  ``n_for_superiority``
        used to halve its alpha, so passing Holm's one-sided 0.025 threshold
        sized the claim at one-sided 0.0125 - z = 2.2414 instead of 1.9600 -
        and reported 7/5 clusters where the declared threshold supports 5/4.
        A module with two alpha conventions cannot check itself, so the
        convention is pinned here: a one-sided level and the two-sided level
        that shares its quantile must give the same n, and the worst-case
        Holm level must NOT be halved again.

        Note what this does and does not say.  One-sided 0.025 and two-sided
        0.05 share z(0.975), which is an identity about quantiles.  It is not
        a statement that one-sidedness costs half of anything: a STANDALONE
        directional claim at familywise 0.05 is tested at one-sided 0.05.
        See TestTheAlphaConventionSeparatesItsTwoSources."""
        sd, theta = 0.157640, 0.198172
        # a one-sided 0.025 argument must reproduce the two-sided 0.05
        # textbook answer, computed here from the quantile rather than by
        # calling the function with a constant that equals the same thing
        from power_analysis_confirmation import z
        d = abs(theta) / sd
        want_two_sided_005 = ((z(0.975) + z(0.80)) / d) ** 2
        assert n_for_superiority(sd, theta, ALPHA_ONE_SIDED, 0.80) == \
            pytest.approx(want_two_sided_005)
        # halving the Holm threshold again would be the bug
        k = len(PRIMARY_FAMILY)
        correct = n_for_superiority(sd, theta, FAMILYWISE_ALPHA / k, 0.80)
        doubled = n_for_superiority(sd, theta, FAMILYWISE_ALPHA / k / 2, 0.80)
        assert math.ceil(correct) == 5
        assert math.ceil(doubled) == 7
        assert doubled > correct

    def test_a_zero_effect_needs_an_infinite_sample(self):
        assert n_for_superiority(0.2, 0.0, ALPHA_ONE_SIDED, 0.80) is None

    def test_equivalence_is_infeasible_when_the_effect_exceeds_the_margin(
            self):
        """No sample size concludes equivalence for a difference that is
        really there.  This must be an explicit infeasibility, not a large
        number that reads as 'expensive but possible'."""
        n = n_for_one_sided_margin(0.2, 0.12, EQUIVALENCE_MARGIN,
                                  ALPHA_ONE_SIDED, 0.80, "equivalence")
        assert math.isinf(n)

    def test_non_inferiority_uses_the_one_sided_distance(self):
        """theta + margin, not margin - |theta|: a state that is already
        BETTER than the reference has more room, not less.  For a NEGATIVE
        theta the two formulas coincide, so the distinction is only visible
        on the side where the state is behind."""
        sd = 0.2031
        behind = n_for_one_sided_margin(sd, -0.0164, EQUIVALENCE_MARGIN,
                                        ALPHA_ONE_SIDED, 0.80,
                                        "non_inferior")
        ahead = n_for_one_sided_margin(sd, 0.0164, EQUIVALENCE_MARGIN,
                                       ALPHA_ONE_SIDED, 0.80,
                                       "non_inferior")
        assert ahead < behind
        # equivalence is symmetric in |theta|, so being ahead does not help,
        # and it is strictly harder than the one-sided question
        eq_ahead = n_for_one_sided_margin(sd, 0.0164, EQUIVALENCE_MARGIN,
                                          ALPHA_ONE_SIDED, 0.80,
                                          "equivalence")
        eq_behind = n_for_one_sided_margin(sd, -0.0164, EQUIVALENCE_MARGIN,
                                           ALPHA_ONE_SIDED, 0.80,
                                           "equivalence")
        assert eq_ahead == pytest.approx(eq_behind)
        assert eq_ahead > ahead

    def test_a_larger_margin_needs_fewer_clusters(self):
        sd = 0.2157
        n05 = n_for_one_sided_margin(sd, -0.0244, 0.05, ALPHA_ONE_SIDED,
                                     0.80, "equivalence")
        n10 = n_for_one_sided_margin(sd, -0.0244, 0.10, ALPHA_ONE_SIDED,
                                     0.80, "equivalence")
        assert n10 < n05

    def test_achievable_margin_grows_with_the_sd_and_shrinks_with_k(self):
        small = margin_achievable(0.10, 0.0, 72, ALPHA_ONE_SIDED, 0.80,
                                  "equivalence")
        large = margin_achievable(0.30, 0.0, 72, ALPHA_ONE_SIDED, 0.80,
                                  "equivalence")
        fewer = margin_achievable(0.10, 0.0, 20, ALPHA_ONE_SIDED, 0.80,
                                  "equivalence")
        assert small < large
        assert small < fewer

    def test_the_mean_sd_interval_brackets_the_estimate(self):
        mu, sd, ci = _mean_sd([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mu == 3.0
        assert abs(sd - math.sqrt(2.5)) < 1e-9
        assert ci is not None and ci[0] < sd < ci[1]

    def test_a_degenerate_sample_reports_no_interval(self):
        assert _mean_sd([2.0]) == (2.0, 0.0, None)
        assert _mean_sd([1.0, 1.0, 1.0])[2] is None


class TestVarianceComponents:
    def test_no_clustering_gives_a_zero_between_component(self):
        # identical spread inside every group, means all equal
        groups = {f"s{i}": [0.0, 1.0, 0.0, 1.0] for i in range(10)}
        vc = variance_components(groups)
        assert vc["estimable"]
        assert vc["sigma2_between"] == pytest.approx(0.0, abs=1e-9)
        assert vc["icc_point_estimate"] == pytest.approx(0.0, abs=1e-9)

    def test_pure_clustering_gives_an_icc_near_one(self):
        # constant within each group, groups far apart
        groups = {f"s{i}": [float(i)] * 4 for i in range(12)}
        vc = variance_components(groups)
        assert vc["sigma2_within"] == pytest.approx(0.0, abs=1e-9)
        assert vc["icc_point_estimate"] == pytest.approx(1.0, abs=1e-3)

    def test_a_negative_between_component_is_floored_and_flagged(self):
        """Reporting a negative variance component as a measurement would
        understate the species count the design needs."""
        groups = {"s0": [0.0, 1.0, 0.0], "s1": [1.0, 0.0, 1.0],
                  "s2": [0.0, 1.0, 0.0], "s3": [1.0, 0.0, 1.0]}
        vc = variance_components(groups)
        assert vc["sigma2_between_raw"] <= 0.0
        assert vc["sigma2_between_floored_at_zero"] is True
        assert vc["sigma2_between"] == 0.0

    def test_one_probe_per_cluster_cannot_decompose(self):
        vc = variance_components({f"s{i}": [1.0] for i in range(5)})
        assert vc["estimable"] is False
        assert "within-cluster replication" in vc["reason"]

    def test_a_single_cluster_cannot_decompose(self):
        assert variance_components({"s0": [1.0, 0.0, 1.0]})[
            "estimable"] is False

    def test_an_unequal_design_is_recorded_as_unequal(self):
        """The exploratory wording stratum gives 27 persons 3 probes, 12
        persons 6 and 3 persons 9.  A harmonic mean below the arithmetic
        mean is the signature of that, and the confirmation design is
        balanced deliberately - so the report has to be able to say which
        of the two it is looking at."""
        uneven = variance_components(
            {"a": [0.0, 1.0, 0.0], "b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})
        assert uneven["probes_per_cluster_balanced"] is False
        assert uneven["probes_per_cluster_distribution"] == {"3": 1, "6": 1}
        assert uneven["probes_per_cluster_harmonic_mean"] < 4.5
        even = variance_components(
            {"a": [0.0, 1.0, 0.0], "b": [1.0, 0.0, 1.0]})
        assert even["probes_per_cluster_balanced"] is True
        assert even["probes_per_cluster_distribution"] == {"3": 2}


class TestNestedSizing:
    def _vc(self, icc: float, total: float = 0.25, m: int = 3,
            k: int = 30) -> dict:
        """A variance-components block with a chosen ICC."""
        return {
            "estimable": True, "num_clusters": k, "num_probes": k * m,
            "sigma2_between": total * icc,
            "sigma2_within": total * (1 - icc),
            "icc_point_estimate": icc,
            "icc_ci95": [max(0.0, icc - 0.05), min(1.0, icc + 0.05)],
            "icc_planning_value_upper_bound": min(1.0, icc + 0.05),
        }

    def test_more_probes_per_entity_never_needs_more_species(self):
        g = nested_size_grid(self._vc(0.1), 0.05)
        need = [r["clusters_required"] for r in g["grid"]]
        assert need == sorted(need, reverse=True)

    def test_photos_cannot_buy_below_the_between_species_floor(self):
        """The whole point of the nested design: only the WITHIN component
        is divided by m, so the requirement converges and never reaches
        zero."""
        g = nested_size_grid(self._vc(0.3), 0.05)
        floor = g["asymptotic_cluster_floor_infinite_probes"]
        assert isinstance(floor, int) and floor > 0
        assert all(r["clusters_required"] >= floor for r in g["grid"])

    def test_zero_clustering_makes_photos_fully_efficient(self):
        g = nested_size_grid(self._vc(0.0), 0.05)
        assert g["asymptotic_cluster_floor_infinite_probes"] == 0

    def test_the_conservative_grid_never_needs_fewer_species(self):
        g = nested_size_grid(self._vc(0.05), 0.05)
        for point, cons in zip(g["grid"],
                               g["conservative_grid_at_icc_upper_bound"]):
            assert cons["clusters_required"] >= point["clusters_required"]

    def test_the_ceiling_table_reports_power_not_just_half_width(self):
        """An expected half-width equal to the margin is 50% power, so a
        table reporting only whether the interval fits would call a coin
        flip adequate — and would contradict the clusters_required grid
        printed beside it."""
        g = nested_size_grid(self._vc(0.1), 0.05, cluster_ceiling=36)
        rows = g["achieved_at_cluster_ceiling"]
        assert rows
        for row in rows:
            hw = row["half_width_at_36_clusters_point_icc"]
            pwr = row["power_at_36_clusters_point_icc"]
            # half-width exactly at the margin <=> power exactly 0.5
            if abs(hw - 0.05) < 1e-3:
                assert abs(pwr - 0.5) < 0.05
            assert (hw < 0.05) == (pwr > 0.5)
        powers = [r["power_at_36_clusters_point_icc"] for r in rows]
        assert powers == sorted(powers)

    def test_more_species_always_increases_power_at_the_ceiling(self):
        small = nested_size_grid(self._vc(0.1), 0.05, cluster_ceiling=12)
        large = nested_size_grid(self._vc(0.1), 0.05, cluster_ceiling=36)
        p_small = small["achieved_at_cluster_ceiling"][0][
            "power_at_12_clusters_point_icc"]
        p_large = large["achieved_at_cluster_ceiling"][0][
            "power_at_36_clusters_point_icc"]
        assert p_large > p_small


# ── the committed feasibility conclusions ─────────────────────────

class TestCommittedFeasibilityConclusions:
    """Pinned so a re-run that moves them is noticed, not absorbed."""

    def test_the_claim_kinds_match_the_preregistration(self):
        assert CLAIM_KIND["tga"] == "superiority"
        assert CLAIM_KIND["filr"] == "superiority"
        # Retention and every M_G comparison were DEMOTED by decision.  The
        # measurements that forced those decisions are preserved separately,
        # so the demotion must not read as "we never asked".
        assert CLAIM_KIND["retain_same"] == "descriptive"
        assert CLAIM_KIND["retain_other"] == "descriptive"
        assert set(CLAIM_KIND_VS_MG.values()) == {"descriptive"}
        assert PRIMARY_FAMILY == ("B3_minus_B0:tga", "B3_minus_B0:filr")

    def test_the_report_reproduces_the_11r_cluster_counts(self):
        r = _load()
        claims = r["entity_clustered_claims"]
        assert claims["B3_minus_B0:tga"]["num_clusters"] == 72
        assert claims["B3_minus_B0:filr"]["num_clusters"] == 72
        assert claims["B3_minus_B0:retain_same"]["num_clusters"] == 70
        assert claims["B3_minus_B0:retain_other"]["num_clusters"] == 45
        assert r["feasibility"]["ceilings"]["entities_total"] == 100
        assert r["feasibility"]["ceilings"]["entities_by_source"] == \
            {"mllmu_hier": 64, "inaturalist": 36}

    def test_the_entity_ceiling_is_explained_as_hard_not_budgetary(self):
        """A reviewer reading "needs 287 clusters, 100 available" will say
        "add more species".  The report has to say why that is not an option:
        the preserved checkpoints unlearned a target set drawn from exactly
        these entities, so a new entity changes the intervention rather than
        enlarging the sample.  The ICC quoted in that explanation is
        interpolated, so it must still match the stratum it cites."""
        r = _load()
        c = r["feasibility"]["ceilings"]
        why = c["why_the_entity_ceiling_is_hard"]
        assert str(c["entities_total"]) in why
        assert "SELECTED checkpoints" in why
        icc = r["image_strata"]["seen_photo_unseen_wording"]["metrics"][
            "B3_minus_B0:tga"]["variance_components"]["icc_point_estimate"]
        assert str(icc) in why
        assert "probes per entity" in c["what_is_a_free_parameter"]

    def test_the_two_superiority_claims_are_amply_powered(self):
        """B3 vs B0 on TGA and FILR needs a handful of clusters; 11R already
        had 72.  These are the claims the confirmation split exists to
        confirm, and they are not what the size is driven by."""
        r = _load()
        claims = r["entity_clustered_claims"]
        for name in ("B3_minus_B0:tga", "B3_minus_B0:filr"):
            need = claims[name]["n_for_superiority_power80"]
            assert isinstance(need, int) and need <= 10, (name, need)
            holm = r["holm_primary_family"]["per_claim"][name]
            assert holm["n_at_holm_worst_case_alpha_power80"] <= 12

    def test_retention_non_inferiority_at_delta_005_is_infeasible(self):
        """The measurement that demoted retention.  Both claims would need
        more clusters than the dataset can supply AT THEIR OWN CEILING — the
        entities carrying a retain association, not the entity total.  So
        delta = 0.05 was not an expensive choice, it was an impossible one.
        The requirement is computed counterfactually and kept queryable,
        because the claim is now descriptive and its own row in the sizing
        table no longer carries the number."""
        r = _load()
        d = r["retention_claim_decision"]
        entity_total = r["feasibility"]["ceilings"]["entities_total"]
        key = "clusters_needed_at_delta0.05_power80_had_it_stayed_primary"
        for name in ("B3_minus_B0:retain_same", "B3_minus_B0:retain_other"):
            v = d["per_claim"][name]
            ceiling = v["clusters_available_ceiling"]
            assert v["claim_kind_now"] == "descriptive", name
            # the ceiling is the metric's own, and it is below the total
            assert ceiling < entity_total, (name, ceiling)
            assert isinstance(v[key], int), (name, v[key])
            assert v[key] > ceiling, (name, v[key], ceiling)
            # and it stays infeasible even at the entity total the earlier
            # revision wrongly used as every claim's ceiling, so the
            # decision does not turn on the correction
            assert v[key] > entity_total, (name, v[key])
            assert v["feasible_at_delta0.05"] is False, name
            # and the sizing table itself no longer offers the number, which
            # is exactly why the counterfactual has to be recorded here
            assert r["entity_clustered_claims"][name][
                "n_for_non_inferiority_delta0.05_power80"].startswith("n/a")
        assert str(EQUIVALENCE_MARGIN) in d["delta0.05_verdict"]
        assert "infeasible for both retention claims" in d["delta0.05_verdict"]

    def test_the_demotion_survives_every_candidate_ceiling(self):
        """Review finding 2: the reported 0.0951 was computed at the 45
        clusters retain_other observed while the field beside it said 100
        were available.  The decision was right and its justification was
        not, so the bound is now recomputed at all three candidate counts
        and the verdict is read off the worst of them."""
        r = _load()
        d = r["retention_claim_decision"]
        s = d["sensitivity_to_the_ceiling_choice"]
        assert set(s) == {"at_the_11r_observed_counts",
                          "at_each_claim_own_ceiling",
                          "at_the_entity_total",
                          "independent_of_any_ceiling",
                          "delta0.05_infeasible_under_every_candidate"}
        for k, v in s.items():
            if k.endswith("every_candidate"):
                assert v is True
                continue
            assert v > EQUIVALENCE_MARGIN, (k, v)
        # the ceiling-independent bound is the batch-layout floor, and it is
        # the one the decision leads with
        assert s["independent_of_any_ceiling"] == \
            r["batch_layout_noise_floor"]["max_abs_retain_delta"]
        assert "batch-layout noise floor" in d["binding_argument"]
        # retain_same's ceiling is saturated: every entity with a retain
        # association already contributes a cluster, so its margin cannot be
        # improved by any confirmation split on these adapters
        same = d["per_claim"]["B3_minus_B0:retain_same"]
        assert same["ceiling_saturated"] is True
        assert same["smallest_concludeable_margin_power80_at_the_ceiling"] \
            == same["smallest_concludeable_margin_power80_at_11r_size"]
        # retain_other's is not, and the two margins differ accordingly
        other = d["per_claim"]["B3_minus_B0:retain_other"]
        assert other["ceiling_saturated"] is False
        assert other["smallest_concludeable_margin_power80_at_the_ceiling"] \
            < other["smallest_concludeable_margin_power80_at_11r_size"]
        assert d["lower_bound_from_precision_at_the_cluster_ceilings"] == \
            same["smallest_concludeable_margin_power80_at_the_ceiling"]

    def test_the_smallest_concludeable_retention_margin_exceeds_the_layout(
            self):
        """Two independent arguments must agree before the margin is frozen:
        the smallest margin this design can conclude, and the batch-layout
        noise floor 11R measured on retain metrics.  A margin below either
        would let batch composition alone decide the claim.  The floor is
        read from the report rather than written here so that re-measuring
        it on the confirmation split moves both together."""
        r = _load()
        claims = r["entity_clustered_claims"]
        layout_floor = r["batch_layout_noise_floor"]["max_abs_retain_delta"]
        assert isinstance(layout_floor, float) and layout_floor > 0
        for name in ("B3_minus_B0:retain_same", "B3_minus_B0:retain_other"):
            # both an observed-size and a ceiling margin are present now, so
            # the key has to be selected on the full fragment: matching on
            # the claim kind alone would pick whichever came first
            keys = [k for k in claims[name]
                    if k.startswith("min_margin_concludeable_at")
                    and k.endswith("non_inferiority")]
            assert len(keys) == 2, (name, keys)
            for key in keys:
                m80 = claims[name][key]["power80"]
                assert m80 > layout_floor, (name, key, m80)
                assert m80 > EQUIVALENCE_MARGIN, (name, key, m80)

    def test_equivalence_to_mg_at_delta_005_is_structurally_infeasible(self):
        """11R reported this as INDETERMINATE for lack of power.  The
        analysis says why it was never going to be anything else: the
        required cluster count is several times the number of entities the
        metric is defined on.  That is the evidence behind treating it as
        descriptive."""
        r = _load()
        d = r["mg_equivalence_decision"]
        entity_total = r["feasibility"]["ceilings"]["entities_total"]
        assert d["decision"] == "secondary_descriptive_only"
        assert d["equivalence_test_run"] is False
        assert d["in_the_holm_family"] is False
        key = "clusters_needed_at_delta0.05_power80_had_it_been_tested"
        for name, v in d["per_claim"].items():
            ceiling = v["clusters_available_ceiling"]
            assert v["claim_kind_now"] == "descriptive", name
            # a target metric: its ceiling is the target entities, which is
            # below the entity total
            assert ceiling < entity_total, (name, ceiling)
            assert ceiling == r["entity_role_census"]["by_role"]["target"][
                "total"], name
            assert isinstance(v[key], int), (name, v[key])
            assert v[key] > 5 * ceiling, (name, v[key], ceiling)
            assert v["feasible_at_delta0.05"] is False, name
            assert v["smallest_concludeable_margin_power80_at_11r_size"] > \
                EQUIVALENCE_MARGIN, name
            # the ceiling is saturated, so the observed-size margin is also
            # the best this design can do
            assert v["ceiling_saturated"] is True, name
            assert v["smallest_concludeable_margin_power80_at_the_ceiling"] \
                == v["smallest_concludeable_margin_power80_at_11r_size"], name
        # a wider margin WOULD have been reachable at the clusters available,
        # which is why this is a power finding and not a null result
        assert d["per_claim"]["B3_minus_MG:tga"][
            "smallest_concludeable_margin_power80_at_11r_size"] < 0.10

    def test_no_held_out_probe_count_carries_the_tga_superiority_claim(self):
        """The 11R allocation is not "too few photographs" — no allocation
        is enough.  Sizing this claim against the 0.05 margin, which the
        ceiling table used to do for every claim, reported 80% power at three
        probes per species because it was asking whether the held-out stratum
        could prove B3 EQUIVALENT to B0.  The preregistered claim is B3
        SUPERIOR to B0, and on this stratum 11R measured -0.0111: the wrong
        sign and indistinguishable from zero.  Power against that effect
        never reaches 0.40 at the ceiling, which is the 30 TARGET species
        and not the 36 the dataset holds."""
        r = _load()
        m = r["image_strata"]["held_out_photo"]["metrics"][
            "B3_minus_B0:tga"]["nested_sizing"]
        assert m["claim_kind"] == "superiority"
        assert m["theta_used"] == pytest.approx(-0.0111, abs=1e-4)
        assert m["claim_answerable"] is True   # answerable, just unwinnable
        c = m["cluster_ceiling"]
        assert c == 30
        assert c < r["feasibility"]["ceilings"]["entities_by_source"][
            "inaturalist"]
        rows = m["achieved_at_cluster_ceiling"]
        assert [row["probes_per_entity"] for row in rows] == \
            list(PROBE_OPTIONS)
        for row in rows:
            assert row["power_ge_80_conservative_icc_upper"] is False, row
            assert row[f"power_at_{c}_clusters_conservative_icc_upper"] < 0.40
        # power still RISES with probes: the design is not degenerate, it is
        # aimed at an effect that is not there
        powers = [row[f"power_at_{c}_clusters_conservative_icc_upper"]
                  for row in rows]
        assert powers == sorted(powers)
        # and the required species count is an order of magnitude past the
        # ceiling at every probe count, including the unreachable m=24
        need = [row["clusters_required"]
                for row in m["conservative_grid_at_icc_upper_bound"]]
        assert all(isinstance(n, int) and n > 100 for n in need), need

    def test_the_held_out_mde_is_what_a_probe_count_can_be_chosen_by(self):
        """Power cannot be planned at an effect too loose to estimate, but
        the minimum detectable effect can: it says what the design WOULD
        resolve.  At the 30-target-species ceiling three new photographs per
        species resolve a held-out TGA effect of about 0.055 and twelve
        resolve about 0.029, so the probe count is a choice about the
        smallest held-out effect worth reporting, not a choice about
        reaching significance."""
        r = _load()
        m = r["image_strata"]["held_out_photo"]["metrics"][
            "B3_minus_B0:tga"]["nested_sizing"]
        c = m["cluster_ceiling"]
        key = f"mde_at_{c}_clusters_conservative_icc_upper"
        by_m = {row["probes_per_entity"]: row
                for row in m["achieved_at_cluster_ceiling"]}
        # the smallest probe count resolves an effect about the size of the
        # declared margin, which is the honest floor on what is reportable
        assert by_m[3][key] == pytest.approx(EQUIVALENCE_MARGIN, abs=6e-3)
        assert by_m[CONFIRM_NEW_PHOTOS_PER_SPECIES][key] < 0.030
        mdes = [row[key] for row in m["achieved_at_cluster_ceiling"]]
        assert mdes == sorted(mdes, reverse=True)

    def test_the_wording_stratum_carries_both_primary_claims(self):
        """Where the pooled effect actually comes from.  Twenty persons or
        fewer suffice for 80% power against a ceiling of 42 TARGET persons,
        and the achieved power at that ceiling is at least 0.98 at every
        probe count.  So the primary claims' POWER does not depend on new
        photographs — which is not the same as saying the confirmation needs
        none, since the held-out stratum and the retain image families are
        still measured and may not reuse a pilot100_v2 photograph."""
        r = _load()
        block = r["image_strata"]["seen_photo_unseen_wording"]
        assert block["cluster_ceiling"] == 42
        assert block["cluster_ceiling"] < \
            r["feasibility"]["ceilings"]["entities_by_source"]["mllmu_hier"]
        assert block["cluster_ceiling_saturated"] is True
        assert "TARGET" in block["cluster_ceiling_is"]
        for metric in ("B3_minus_B0:tga", "B3_minus_B0:filr"):
            m = block["metrics"][metric]["nested_sizing"]
            c = m["cluster_ceiling"]
            assert m["claim_kind"] == "superiority", metric
            assert abs(m["theta_used"]) > 0.24, (metric, m["theta_used"])
            assert c == 42, (metric, c)
            need = [row["clusters_required"]
                    for row in m["conservative_grid_at_icc_upper_bound"]]
            assert all(isinstance(n, int) and n <= 20 for n in need), \
                (metric, need)
            for row in m["achieved_at_cluster_ceiling"]:
                assert row[f"power_at_{c}_clusters_conservative_icc_upper"] \
                    >= 0.98, (metric, row)
                assert row["power_ge_80_conservative_icc_upper"] is True

    def test_the_pooled_effect_is_not_the_average_of_two_similar_strata(self):
        """The finding that decides the photograph question, computed by the
        script rather than asserted in prose: both B3-vs-B0 primary claims
        are carried by the wording stratum alone.  That is a statement about
        where the EFFECT is, not about which estimand is primary — the
        primary estimand is declared separately and is the pooled one."""
        r = _load()
        h = r["stratum_heterogeneity"]
        assert set(h) == {"B3_minus_B0:tga", "B3_minus_B0:filr"}
        for metric, block in h.items():
            assert block["strata_that_carry_the_claim"] == \
                ["seen_photo_unseen_wording"], metric
            assert block["strata_that_do_not"] == ["held_out_photo"], metric
            held = block["per_stratum"]["held_out_photo"]
            wording = block["per_stratum"]["seen_photo_unseen_wording"]
            assert held["ci_excludes_zero"] is False, metric
            assert held["num_clusters"] == 30
            assert held["cluster_ceiling"] == 30
            assert wording["num_clusters"] == 42
            assert wording["cluster_ceiling"] == 42
            assert wording["ci_excludes_zero"] is True, metric
            assert wording["effect_direction_matches_the_claim"] is True
        # TGA on held-out photographs is not merely null, it is the wrong
        # sign; FILR is at least in the claimed direction
        assert h["B3_minus_B0:tga"]["per_stratum"]["held_out_photo"][
            "effect_direction_matches_the_claim"] is False
        assert h["B3_minus_B0:filr"]["per_stratum"]["held_out_photo"][
            "effect_direction_matches_the_claim"] is True

    def test_the_held_out_icc_interval_is_bounded_and_drives_the_plan(self):
        """The planning value must BE the bootstrap upper bound, not the
        point estimate: the point estimate is floored at zero here, and a
        floored zero cannot rule out species-level clustering."""
        r = _load()
        for metric in ("B3_minus_B0:tga", "B3_minus_B0:filr"):
            block = r["image_strata"]["held_out_photo"]["metrics"][metric]
            vc = block["variance_components"]
            lo, hi = vc["icc_ci95"]
            assert 0.0 <= lo <= hi <= 1.0, (metric, vc)
            assert vc["icc_planning_value_upper_bound"] == hi
            sizing = block["nested_sizing"]
            assert sizing[
                "icc_planning_value_used_for_conservative_grid"] == round(
                    hi, 4), metric

    def test_the_seen_photo_stratum_is_the_person_level_one(self):
        """Documented in 11R as a confound: held_out_photo is exactly the
        iNaturalist species and seen_photo_unseen_wording exactly the MLLMU
        persons, because a person has one portrait and a species twelve."""
        r = _load()
        strata = r["image_strata"]
        assert strata["held_out_photo"]["source_datasets"] == ["inaturalist"]
        assert strata["seen_photo_unseen_wording"]["source_datasets"] == \
            ["mllmu_hier"]
        assert strata["held_out_photo"]["num_clusters"] == 30
        assert strata["seen_photo_unseen_wording"]["num_clusters"] == 42
        assert strata["held_out_photo"]["nesting_cluster_is"] == \
            "iNaturalist species"
        assert strata["seen_photo_unseen_wording"]["nesting_cluster_is"] == \
            "MLLMU person"

    def test_holm_thresholds_are_the_declared_sequence(self):
        """k fell from four to two when retention and the M_G comparisons were
        demoted, which is what makes the correction cheap enough to keep both
        remaining claims primary."""
        r = _load()
        holm = r["holm_primary_family"]
        assert holm["k"] == 2
        assert holm["primary_family"] == list(PRIMARY_FAMILY)
        assert holm["holm_thresholds"] == [0.025, 0.05]
        assert holm["worst_case_alpha_for_a_single_claim"] == 0.025
        assert set(holm["per_claim"]) == set(holm["primary_family"])

    def test_the_holm_thresholds_and_the_sizing_use_the_same_quantile(self):
        """Review finding 4.  The thresholds were declared for one-sided
        p-values while the sizing halved them again, so the report quoted a
        0.025 worst case and a cluster requirement computed at one-sided
        0.0125 - 7/5 instead of 5/4.  Conservative, but not the declared
        test, and a preregistration has to say which one it means."""
        r = _load()
        holm = r["holm_primary_family"]
        k = holm["k"]
        assert holm["familywise_alpha"] == FAMILYWISE_ALPHA == 0.05
        assert holm["thresholds_apply_to"] == "one-sided p-values"
        assert holm["sizing_alpha_one_sided"] == \
            holm["worst_case_alpha_for_a_single_claim"] == \
            FAMILYWISE_ALPHA / k
        # the critical value the sizing used is z(1 - familywise/k), not the
        # z(1 - familywise/k/2) the two-sided reading would give
        assert holm["critical_value_z"] == pytest.approx(1.959964, abs=1e-5)
        claims = r["entity_clustered_claims"]
        for name, entry in holm["per_claim"].items():
            sd = claims[name]["sd_of_cluster_diffs"]
            mu = claims[name]["mean_diff"]
            want = math.ceil(n_for_superiority(
                sd, mu, FAMILYWISE_ALPHA / k, 0.80))
            assert entry["n_at_holm_worst_case_alpha_power80"] == want, \
                (name, want)
            assert entry["clusters_available_ceiling"] == \
                claims[name]["cluster_ceiling"], name
            assert entry["feasible_at_holm_worst_case_alpha_power80"] is True
        # and the corrected requirement is the one the review predicted
        assert holm["per_claim"]["B3_minus_B0:tga"][
            "n_at_holm_worst_case_alpha_power80"] == 5
        assert holm["per_claim"]["B3_minus_B0:filr"][
            "n_at_holm_worst_case_alpha_power80"] == 4
        assert r["design"]["familywise_alpha"] == FAMILYWISE_ALPHA
        assert "ONE-SIDED" in r["design"]["alpha_convention"]


class TestTheClusterCeilingsArePerClaim:
    """Review finding 2.  Every claim used to be assigned
    ``clusters_available: 100``, which is the number of entities the dataset
    holds and the ceiling of no metric at all."""

    def test_the_role_census_matches_the_frozen_manifest(self):
        r = _load()
        c = r["entity_role_census"]
        assert c["entities_total"] == 100
        assert c["by_source"] == {"inaturalist": 36, "mllmu_hier": 64}
        assert c["role_sets"] == {"inaturalist/retain": 6,
                                  "inaturalist/target": 30,
                                  "mllmu_hier/retain": 22,
                                  "mllmu_hier/retain+target": 42}
        assert c["by_role"]["target"]["total"] == 72
        assert c["by_role"]["target"]["by_source"] == {
            "inaturalist": 30, "mllmu_hier": 42}
        assert c["by_role"]["retain"]["total"] == 70
        assert c["by_role"]["retain"]["by_source"] == {
            "inaturalist": 6, "mllmu_hier": 64}
        assert "manifest.json" in c["read_from"]

    def test_the_census_is_recomputable_from_the_committed_dataset(self):
        """The census is read from the frozen manifest, so it has to survive
        being re-derived: a ceiling that cannot be reproduced from the
        committed dataset is an assertion, not a measurement."""
        from granunlearn.evaluation.reference_eval import (
            load_associations_parquet)
        data_dir = REPO_ROOT / "data" / "mllmu_hier_pilot100"
        assoc = load_associations_parquet(data_dir / "associations.parquet")
        fresh = entity_role_census(data_dir, assoc)
        committed = _load()["entity_role_census"]
        assert fresh["entities_total"] == committed["entities_total"]
        assert fresh["role_sets"] == committed["role_sets"]
        for role in ("target", "retain"):
            assert fresh["by_role"][role]["total"] == \
                committed["by_role"][role]["total"], role
            assert fresh["by_role"][role]["entity_ids_sha256"] == \
                committed["by_role"][role]["entity_ids_sha256"], role

    def test_no_claim_is_assigned_the_entity_total(self):
        r = _load()
        total = r["feasibility"]["ceilings"]["entities_total"]
        by_role = r["entity_role_census"]["by_role"]
        claims = r["feasibility"]["claims"]
        assert set(claims) == set(r["entity_clustered_claims"])
        for name, entry in claims.items():
            metric = name.split(":", 1)[1]
            role = METRIC_CLUSTER_ROLE[metric]
            assert entry["clusters_available_ceiling"] == \
                by_role[role]["total"], (name, role)
            assert entry["clusters_available_ceiling"] < total, name
            assert role in entry["ceiling_is"], name
            assert "clusters_available" not in entry, name
            assert entry["clusters_observed_at_11r_size"] == \
                r["entity_clustered_claims"][name]["num_clusters"], name
            assert entry["ceiling_saturated"] == (
                entry["clusters_observed_at_11r_size"]
                == entry["clusters_available_ceiling"]), name

    def test_the_observed_counts_are_the_ones_11r_produced(self):
        """The ceilings changed; the measurements must not have.  These are
        the four counts the review listed."""
        r = _load()
        claims = r["feasibility"]["claims"]
        want = {"B3_minus_B0:tga": (72, 72),
                "B3_minus_B0:filr": (72, 72),
                "B3_minus_B0:retain_same": (70, 70),
                "B3_minus_B0:retain_other": (45, 70)}
        for name, (observed, ceiling) in want.items():
            assert claims[name]["clusters_observed_at_11r_size"] == observed
            assert claims[name]["clusters_available_ceiling"] == ceiling

    def test_both_margins_are_reported_and_each_names_its_own_count(self):
        """The defect was not the arithmetic, it was the labelling: a margin
        at 45 clusters sitting beside a field that said 100 were available.
        Both numbers are now present and each key states the count it was
        computed at."""
        r = _load()
        for name, claims in (("B3_minus_B0:retain_other",
                              r["entity_clustered_claims"]),):
            observed = claims[name]["num_clusters"]
            ceiling = claims[name]["cluster_ceiling"]
            assert observed != ceiling, name
            obs_key = f"min_margin_concludeable_at_{observed}_clusters_" \
                      "non_inferiority"
            ceil_key = f"min_margin_concludeable_at_the_{ceiling}" \
                       "_cluster_ceiling_non_inferiority"
            assert claims[name][obs_key]["power80"] == \
                pytest.approx(0.0951, abs=1e-4)
            assert claims[name][ceil_key]["power80"] == \
                pytest.approx(0.0720, abs=1e-4)
            # more clusters can only shrink a concludeable margin
            assert claims[name][ceil_key]["power80"] < \
                claims[name][obs_key]["power80"]
        # and the split in the feasibility block keeps them apart
        f = r["feasibility"]["claims"]["B3_minus_B0:retain_other"]
        assert len(f["min_margin_concludeable"]["at_11r_size"]) == 2
        assert len(f["min_margin_concludeable"]["at_the_cluster_ceiling"]) \
            == 2

    def test_the_ceiling_explanation_says_why_the_total_is_not_one(self):
        r = _load()
        c = r["feasibility"]["ceilings"]
        why = c["why_the_ceiling_is_per_claim"]
        assert str(c["entities_total"]) in why
        assert str(c["cluster_ceiling_by_metric_role"]["target"]["total"]) \
            in why
        assert str(c["cluster_ceiling_by_metric_role"]["retain"]["total"]) \
            in why
        assert c["cluster_ceiling_by_metric_role"]["target"]["metrics"] == \
            sorted(m for m, role in METRIC_CLUSTER_ROLE.items()
                   if role == "target")
        assert c["entity_clusters_available_for_pooled_target_claims"] == 72
        assert c["stratum_cluster_ceilings"] == {
            "held_out_photo": 30, "seen_photo_unseen_wording": 42}


class TestThePrimaryEstimandIsDeclared:
    """Review finding 3.  The family names were pooled while the prose said
    the claims were carried by one stratum, and a preregistration that makes
    the reader reconcile those has not chosen an estimand."""

    def test_one_estimand_is_named_and_it_is_the_pooled_one(self):
        r = _load()
        pe = r["primary_estimand"]
        assert pe["declared"] == PRIMARY_ESTIMAND == \
            "pooled_over_target_entities"
        assert pe["entities_in_scope"] == 72
        assert pe["entities_in_scope_by_source"] == {
            "inaturalist": 30, "mllmu_hier": 42}
        assert pe["primary_family"] == list(PRIMARY_FAMILY)
        assert pe["strata_whose_probes_enter_it"] == \
            list(PRIMARY_ESTIMAND_STRATA)
        assert len(pe["strata_whose_probes_enter_it"]) == 2
        assert "entity-macro" in pe["definition"]
        assert str(pe["entities_in_scope"]) in pe["definition"]

    def test_the_stratum_decomposition_is_labelled_secondary(self):
        r = _load()
        pe = r["primary_estimand"]
        assert pe["per_stratum_decomposition_status"] == \
            STRATUM_ESTIMAND_STATUS
        assert "no_holm_entry" in STRATUM_ESTIMAND_STATUS
        # the decomposition block itself must not read as a second family
        h = r["stratum_heterogeneity"]
        assert set(h) == set(PRIMARY_FAMILY)
        for block in h.values():
            assert block["pooled_num_clusters"] == pe["entities_in_scope"]

    def test_the_fixed_cohort_limit_is_stated_not_implied(self):
        """Reusing the same entities with new probes is robustness on a fixed
        cohort.  A report that let this read as replication would be
        claiming a population the design never sampled, and the limit applies
        to whichever estimand is primary."""
        r = _load()
        text = r["primary_estimand"]["what_it_does_not_establish"]
        assert text == WHAT_THE_CONFIRMATION_DOES_NOT_ESTABLISH
        assert "population-level replication" in text
        assert "does not create new entity clusters" in text
        assert "fixed cohort" in text
        # and it is in the notes, which is what a reader actually reads
        assert any("population-level replication" in n
                   for n in r["notes"])

    def test_the_rejected_estimands_are_recorded_with_their_scope(self):
        r = _load()
        rejected = r["primary_estimand"]["rejected"]
        assert len(rejected) == 2
        by_name = {x["estimand"]: x for x in rejected}
        assert "seen_photo_unseen_wording only" in by_name
        assert by_name["seen_photo_unseen_wording only"][
            "entities_in_scope"] == 42
        assert "after the heterogeneity was measured" in \
            by_name["seen_photo_unseen_wording only"]["why_rejected"]
        assert any("two Holm families" in k_ for k_ in by_name), list(by_name)
        # the dilution argument has to be stated, because it is the reason a
        # pooled rejection is conservative rather than merely noisier
        assert "DILUTES" in r["primary_estimand"][
            "why_pooled_rather_than_the_stratum_that_carries_the_effect"]


# ── the sizing is claim-aware ──────────────────────────────────

class TestTheSizingIsClaimAware:
    """One denominator per claim kind.  Dividing every claim by the raw
    margin sizes a superiority claim as though it were an equivalence test,
    which is the error that made the held-out stratum look adequately
    powered and the wording stratum look hopeless."""

    def test_usable_distance_is_the_claim_not_the_margin(self):
        assert usable_distance("superiority", 0.2751, 0.05) == \
            pytest.approx(0.2751)
        assert usable_distance("non_inferiority", -0.0164, 0.05) == \
            pytest.approx(0.0336)
        assert usable_distance("equivalence", -0.0244, 0.05) == \
            pytest.approx(0.0256)
        assert usable_distance("descriptive", 0.3, 0.05) is None

    def test_equivalence_at_a_zero_effect_reduces_to_the_margin(self):
        """The default the sizing-primitive tests rely on, and the reason
        they did not move when the grids became claim-aware."""
        assert usable_distance("equivalence", 0.0,
                              EQUIVALENCE_MARGIN) == EQUIVALENCE_MARGIN

    def test_an_effect_outside_the_margin_has_no_answer(self):
        """Distinct from "expensive": equivalence to M_G on held-out
        photographs measured -0.100, twice the margin, and no cluster count
        concludes it.  Returning a large integer here would read as a budget
        problem instead of a claim that must be dropped or restated."""
        assert usable_distance("equivalence", -0.10, 0.05) is None
        assert usable_distance("non_inferiority", -0.06, 0.05) is None

    def test_a_zero_effect_cannot_be_powered_as_superiority(self):
        assert usable_distance("superiority", 0.0, 0.05) is None

    def test_power_and_mde_are_exact_inverses(self):
        """mde_at is DEFINED as the effect (or margin) yielding exactly the
        target power at this standard error, so feeding it back through
        power_at must return the target.  A drift between the two silently
        resizes every cell of every grid."""
        se, theta = 0.03, -0.0164
        m_sup = mde_at(se, "superiority", theta, ALPHA_ONE_SIDED, 0.80)
        assert power_at(se, "superiority", m_sup,
                        EQUIVALENCE_MARGIN) == pytest.approx(0.80, abs=1e-9)
        m_eq = mde_at(se, "equivalence", 0.0, ALPHA_ONE_SIDED, 0.80)
        assert power_at(se, "equivalence", 0.0, m_eq) == \
            pytest.approx(0.80, abs=1e-9)
        m_ni = mde_at(se, "non_inferiority", theta, ALPHA_ONE_SIDED, 0.80)
        assert power_at(se, "non_inferiority", theta,
                        m_ni) == pytest.approx(0.80, abs=1e-9)

    def test_a_descriptive_metric_has_neither_power_nor_mde(self):
        assert power_at(0.03, "descriptive", 0.2, EQUIVALENCE_MARGIN) is None
        assert mde_at(0.03, "descriptive", 0.2) is None

    def test_a_larger_effect_needs_fewer_clusters_at_equal_variance(self):
        vc = {"estimable": True, "num_clusters": 30, "num_probes": 90,
              "sigma2_between": 0.02, "sigma2_within": 0.08,
              "icc_point_estimate": 0.2, "icc_ci95": [0.1, 0.3],
              "icc_planning_value_upper_bound": 0.3}
        big = nested_size_grid(vc, EQUIVALENCE_MARGIN, "superiority", 0.275)
        small = nested_size_grid(vc, EQUIVALENCE_MARGIN, "superiority", 0.05)
        assert big["grid"][0]["clusters_required"] < \
            small["grid"][0]["clusters_required"]

    def test_an_unanswerable_claim_says_so_but_still_reports_precision(self):
        vc = {"estimable": True, "num_clusters": 30, "num_probes": 90,
              "sigma2_between": 0.02, "sigma2_within": 0.08,
              "icc_point_estimate": 0.2, "icc_ci95": [0.1, 0.3],
              "icc_planning_value_upper_bound": 0.3}
        g = nested_size_grid(vc, EQUIVALENCE_MARGIN, "equivalence", -0.10,
                             cluster_ceiling=36)
        assert g["claim_answerable"] is False
        assert g["usable_distance"] is None
        assert "already exceeds" in g["not_answerable_reason"]
        assert all(isinstance(row["clusters_required"], str)
                   for row in g["grid"])
        assert isinstance(
            g["asymptotic_cluster_floor_infinite_probes"], str)
        # power is unanswerable, but the half-width and the smallest
        # concludeable margin are properties of the DESIGN and still report
        row = g["achieved_at_cluster_ceiling"][0]
        assert row["power_at_36_clusters_conservative_icc_upper"] is None
        assert row["power_ge_80_conservative_icc_upper"] is None
        assert row["half_width_at_36_clusters_conservative_icc_upper"] > 0
        assert row["mde_at_36_clusters_conservative_icc_upper"] > 0.10

    def test_the_zero_true_difference_column_is_absent_for_superiority(self):
        """At a true difference of zero a superiority test's "power" is its
        type-I error rate, so publishing it in a power column would be a
        category error rather than a conservative bound."""
        vc = {"estimable": True, "num_clusters": 30, "num_probes": 90,
              "sigma2_between": 0.02, "sigma2_within": 0.08,
              "icc_point_estimate": 0.2, "icc_ci95": [0.1, 0.3],
              "icc_planning_value_upper_bound": 0.3}
        sup = nested_size_grid(vc, EQUIVALENCE_MARGIN, "superiority", 0.275,
                               cluster_ceiling=36)
        eq = nested_size_grid(vc, EQUIVALENCE_MARGIN, "equivalence", 0.0,
                              cluster_ceiling=36)
        assert sup["achieved_at_cluster_ceiling"][0][
            "power_at_true_difference_zero_point_icc"] is None
        assert eq["achieved_at_cluster_ceiling"][0][
            "power_at_true_difference_zero_point_icc"] is not None


# ── the photograph supply is measured, not assumed ──────────────────

class TestThePhotographSupplyIsMeasured:
    """Stage 3 of the confirmation phase has to source photographs
    pilot100_v2 never used.  What exists bounds the design, so it is
    measured here by content hash rather than discovered mid-construction."""

    def test_the_frozen_pool_contributes_no_new_photographs(self):
        r = _load()
        s = r["new_photograph_supply"]
        assert s["frozen_pool_contributes_new_photos"] == 0
        p = s["pools"]["pilot_v1"]
        assert p["num_species"] == 36 and p["num_photos"] == 432
        assert p["photos_per_species"] == [12]
        assert p["num_photos_referenced_by_pilot100_v2"] == 432
        assert p["num_photos_disjoint_from_pilot100_v2"] == 0
        assert "fully allocated" in s["frozen_pool_reason"]

    def test_an_offline_pool_of_new_photos_exists_but_is_not_auditable(self):
        """240 content-hash-disjoint photographs over 20 of the 36 species
        are sitting on disk.  They are genuinely new, and they are still not
        usable for a confirmatory claim: the pool is gitignored and records
        no license, attribution, observation id or source URL — the same
        unverifiable-bytes gap Iteration 11R1 closed for the frozen set."""
        r = _load()
        o = r["new_photograph_supply"]["offline_pool"]
        assert o["num_new_photos"] == 240
        assert o["num_species_covered"] == 20
        assert o["max_new_photos_per_species"] == 12
        assert o["species_left_with_no_new_photo"] == 16
        assert o["tracked_by_git"] is False
        assert o["has_license_and_attribution"] is False
        assert o["usable_for_a_confirmatory_claim"] is False
        for field in ("license_code", "attribution", "source_url",
                      "observation_id", "photo_id"):
            assert field in o["missing_provenance_fields"], field

    def test_the_held_out_ceiling_is_a_supply_number_not_a_dataset_number(
            self):
        """30 is how many TARGET species the frozen dataset contains, 36 how
        many species it contains at all, and 20 how many a new photograph can
        be sourced for offline.  The report is sized at both the 30 and the 20
        so the offline option cannot be read off the dataset table by
        mistake."""
        r = _load()
        a = r["new_photograph_supply"][
            "achievable_ceilings_for_the_held_out_stratum"]
        assert a["species_offline_with_auditable_provenance"] == 0
        assert a["species_offline_any_provenance"] == 20
        assert a["species_via_seeded_refetch"] == 36
        m = r["image_strata"]["held_out_photo"]["metrics"][
            "B3_minus_B0:tga"]
        c = m["nested_sizing"]["cluster_ceiling"]
        assert c == 30
        off = m["nested_sizing_at_offline_new_photo_ceiling"]
        assert off["cluster_ceiling"] == 20
        assert off["cluster_ceiling"] < c
        assert off["achieved_at_cluster_ceiling"][0][
            "power_at_20_clusters_conservative_icc_upper"] < \
            m["nested_sizing"]["achieved_at_cluster_ceiling"][0][
                f"power_at_{c}_clusters_conservative_icc_upper"]

    def test_the_refetch_route_selects_by_hash_and_not_by_position(self):
        """Novelty is a content-hash property.  An earlier revision argued it
        from the seeded shuffle instead ("the first 12 of the longer draw are
        the already-allocated photographs"), which is false for this fetch, so
        the report now states the selection rule and records why the shuffle
        argument does not hold."""
        r = _load()
        s = r["new_photograph_supply"]["seeded_refetch"]
        assert s["network_required"] is True
        assert s["records_license_and_attribution"] is True
        assert s["species_covered"] == 36
        # the mechanism is a hash set difference against the exploratory
        # manifest, and it must name that manifest
        assert "content-hash disjointness" in s["disjointness_mechanism"]
        assert "never position in the draw" in s["disjointness_mechanism"]
        assert "image_manifest.json" in s["disjointness_mechanism"]
        # a selection rule that can be executed, with a refusal in it
        assert "sha256" in s["selection_rule"]
        assert "fewer than 12 are disjoint" in s["selection_rule"]
        assert "(observation_id, photo_id)" in s["selection_rule"]

    def test_the_superset_claim_is_withdrawn_not_left_standing(self):
        """The withdrawn claim has to stay visible with its refutation, or the
        next reader re-derives it from the seed and believes it again."""
        s = _load()["new_photograph_supply"]["seeded_refetch"]
        assert "no superset relation is assumed" in \
            s["disjointness_mechanism"]
        d = s["draw_nesting_is_not_relied_on"]
        assert "by construction" in d["claim_an_earlier_revision_made"]
        # both refutations are recorded, and each names the code fact it rests
        # on rather than asserting a conclusion
        assert "ONCE outside the species loop" in \
            d["refutation_1_rng_state_advances_across_species"]
        assert "36" in d["refutation_1_rng_state_advances_across_species"]
        assert "30" in d["refutation_1_rng_state_advances_across_species"]
        assert "chosen.sort" in \
            d["refutation_2_the_accepted_set_is_re_sorted"]
        # and what IS true is stated, so the correction is not an overclaim in
        # the other direction
        assert "does nest" in d["what_is_true_and_why_it_is_not_enough"]
        assert "cannot carry a novelty argument" in \
            d["what_is_true_and_why_it_is_not_enough"]

    def test_the_draw_size_is_justified_by_a_worst_case_not_by_nesting(self):
        """24 = 12 + 12 survives the correction, but as a supply bound: only
        12 exploratory photographs exist per species, so a 24-draw cannot be
        short of 12 disjoint ones.  The bound is tight, which is why the
        report also says the count gets measured."""
        s = _load()["new_photograph_supply"]["seeded_refetch"]
        w = s["why_the_draw_is_24"]
        assert "smallest draw that guarantees 12 disjoint" in w
        assert "24 - 12 = 12" in w
        assert "no slack" in w
        assert "measured per species after the fetch" in w
        h = _load()["confirmation_size"]["held_out_photographs"]
        assert h["fetch_images_per_species"] == \
            h["already_allocated_per_species"] \
            + h["new_photographs_per_species"] == 24
        v = s["verified_after_the_fetch_by"]
        assert "360" in v and "496" in v and "empty intersection" in v


class TestTheProbeAxisIsLabelledPerStratum:
    def test_the_two_strata_do_not_share_a_lever(self):
        """One axis, two meanings.  A person has exactly one portrait, so a
        new probe there is a new WORDING variant; a species has twelve
        photographs, so a new probe is a new PHOTOGRAPH.  Labelling both
        "photos_per_species" — as this report did — would send the
        probe-construction stage looking for twelve portraits of a person who
        has one."""
        r = _load()
        strata = r["image_strata"]
        held = strata["held_out_photo"]
        seen = strata["seen_photo_unseen_wording"]
        assert held["nesting_cluster_is"] == "iNaturalist species"
        assert seen["nesting_cluster_is"] == "MLLMU person"
        assert "PHOTOGRAPH" in held["probes_per_entity_axis_is"]
        assert "WORDING" in seen["probes_per_entity_axis_is"]
        assert held["photos_per_entity_in_pilot100_v2"] == [12]
        assert seen["photos_per_entity_in_pilot100_v2"] == [1]

    def test_the_held_out_grid_is_capped_by_the_photograph_supply(self):
        """Only twelve new photographs per species exist offline, so the
        m=24 rows are recorded as beyond the reachable maximum rather than
        silently offered as a design."""
        r = _load()
        held = r["image_strata"]["held_out_photo"]
        assert held["max_reachable_probes_per_entity"] == 12
        assert held["probe_options_beyond_the_reachable_max"] == [24]

    def test_the_wording_axis_is_not_capped_by_photographs(self):
        """A wording variant costs a template, not a photograph, so no
        grid row is unreachable — but the portrait never becomes held out,
        which is the confound 11R recorded and this axis cannot fix."""
        r = _load()
        seen = r["image_strata"]["seen_photo_unseen_wording"]
        assert seen["max_reachable_probes_per_entity"] is None
        assert seen["probe_options_beyond_the_reachable_max"] == []


# ── the retention demotion is justified, not asserted ──────────────

class TestTheRetentionDemotionIsJustified:
    """Retention was demoted to descriptive rather than given a wider margin.
    Both lower bounds are still recorded and the counterfactual margin is
    still derived, so the demotion can be audited against the measurement
    that caused it instead of resting on the decision alone."""

    def test_the_layout_floor_is_read_from_11r_not_restated(self):
        """A literal in two places is how the copy diverges from the
        measurement it came from, so the floor is read from 11R's committed
        report and cross-checked against it here."""
        r = _load()
        f = r["batch_layout_noise_floor"]
        assert f["available"] is True
        src_path = REPORTS / Path(f["source"]).name
        assert src_path.exists(), f["source"]
        b = json.loads(src_path.read_text())["batch_composition_sensitivity"]
        assert f["max_abs_retain_delta"] == b["max_abs_retain_delta"]
        assert f["max_abs_target_delta"] == b["max_abs_metric_delta"]
        assert f["retain_metrics"] == ["retain_same", "retain_other"]

    def test_no_margin_is_declared_and_the_counterfactual_is_kept(self):
        r = _load()
        d = r["retention_claim_decision"]
        assert d["decision"] == "demoted_to_descriptive"
        assert d["declared_margin"] is None
        assert "no non-inferiority test" in d["what_is_still_reported"]
        bounds = [d["lower_bound_from_precision_at_the_cluster_ceilings"],
                  d["lower_bound_from_batch_layout_noise"]]
        assert d["smallest_margin_that_would_have_satisfied_both"] == \
            round(max(bounds), 4)
        assert d["margin_that_would_have_been_declared"] == \
            math.ceil(
                d["smallest_margin_that_would_have_satisfied_both"] * 10) / 10

    def test_precision_was_the_binding_bound_not_the_decoding_noise(self):
        """Both bounds had to be satisfied and they are not equal: precision
        at each claim's OWN ceiling is the larger.  So re-measuring the
        layout floor on the confirmation split — stage 2b — could not have
        rescued delta = 0.05 on its own, which is why the demotion does not
        have to be revisited after that measurement."""
        r = _load()
        d = r["retention_claim_decision"]
        precision = "lower_bound_from_precision_at_the_cluster_ceilings"
        assert d[precision] > d["lower_bound_from_batch_layout_noise"]
        assert d["smallest_margin_that_would_have_satisfied_both"] == \
            d[precision]
        # both bounds sit above delta = 0.05, so neither could have supported
        # the original margin
        for key in (precision, "lower_bound_from_batch_layout_noise"):
            assert d[key] > EQUIVALENCE_MARGIN, key

    def test_delta_005_is_recorded_as_impossible_not_expensive(self):
        r = _load()
        d = r["retention_claim_decision"]
        assert set(d["per_claim"]) == {"B3_minus_B0:retain_same",
                                       "B3_minus_B0:retain_other"}
        key = "clusters_needed_at_delta0.05_power80_had_it_stayed_primary"
        for name, v in d["per_claim"].items():
            assert v["feasible_at_delta0.05"] is False, name
            assert v["clusters_available_ceiling"] == 70, name
            assert v[key] > v["clusters_available_ceiling"], name
            # the verdict has to quote the requirement AND the ceiling it was
            # compared against, or it repeats the original defect of naming
            # one number while the field beside it named another
            assert str(v[key]) in d["delta0.05_verdict"], name
            assert str(v["clusters_available_ceiling"]) in \
                d["delta0.05_verdict"], name
            assert v["smallest_concludeable_margin_power80_at_11r_size"] > \
                EQUIVALENCE_MARGIN, name
            assert v["smallest_concludeable_margin_power80_at_the_ceiling"] \
                > EQUIVALENCE_MARGIN, name
            assert v["batch_layout_floor_on_retain_metrics"] == \
                d["lower_bound_from_batch_layout_noise"], name
        assert "infeasible" in d["delta0.05_verdict"]
        # the rejected alternative is on the record, so a reader can see the
        # wider margin was considered rather than overlooked
        assert "0.1" in d["rejected_alternative"]

    def test_the_notes_quote_the_tables_they_summarise(self):
        """A prose note that hardcodes its own numbers is how a report comes
        to contradict itself — the defect 11R1 fixed in the equivalence
        narrative.  Every figure in these notes is interpolated from the
        block it describes, so each must be findable there."""
        r = _load()
        text = " ".join(r["notes"])
        d = r["retention_claim_decision"]
        assert str(d["smallest_margin_that_would_have_satisfied_both"]) \
            in text
        assert str(d["lower_bound_from_batch_layout_noise"]) in text
        for metric, h in r["stratum_heterogeneity"].items():
            held = h["per_stratum"]["held_out_photo"]
            assert f"{held['mean_diff']:+.4f}" in text, metric
            for claim in h["strata_that_carry_the_claim"]:
                assert claim in text, (metric, claim)
        for name, v in r["mg_equivalence_decision"]["per_claim"].items():
            assert str(v["clusters_needed_at_delta0.05_power80_"
                         "had_it_been_tested"]) in text, name
        assert str(r["new_photograph_supply"]["offline_pool"][
            "num_new_photos"]) in text
        # the demotions themselves, not just their numbers
        assert "DESCRIPTIVE" in text
        assert f"k = {len(PRIMARY_FAMILY)}" in text


class TestThePreregistrationDecisionsAreRecorded:
    """Stage 2 freezes the protocol and stage 3 builds probes; both read their
    parameters from this block, so a decision that lives only in a
    conversation is a decision that will be re-litigated mid-build."""

    def test_every_decision_is_present_with_its_evidence(self):
        r = _load()
        p = r["preregistration_decisions"]
        assert p["primary_family"] == list(PRIMARY_FAMILY)
        assert p["primary_estimand"] == PRIMARY_ESTIMAND
        assert p["multiplicity"].startswith(
            f"Holm over k = {len(PRIMARY_FAMILY)}")
        assert str(FAMILYWISE_ALPHA) in p["multiplicity"]
        assert "one-sided" in p["multiplicity"]
        assert p["primary_claim_kinds"] == {
            "B3_minus_B0:tga": "superiority",
            "B3_minus_B0:filr": "superiority"}
        by_id = {d["id"]: d for d in p["decisions"]}
        assert set(by_id) == {"primary_estimand", "confirmation_size",
                              "primary_test", "portrait_reuse",
                              "retention_probe_allocation",
                              "familywise_alpha", "retention_margin",
                              "b3_vs_mg", "new_photograph_supply"}
        for d in p["decisions"]:
            assert d["chosen"], d["id"]
            assert d["rejected"], d["id"]
        # the alpha decision has to say which convention won, not merely that
        # one was chosen - and Iteration 11C-R2's correction is that a
        # STANDALONE one-sided claim is tested at familywise alpha, not at
        # half of it
        assert str(FAMILYWISE_ALPHA) in by_id["familywise_alpha"]["chosen"]
        assert "one-sided" in by_id["familywise_alpha"]["chosen"]
        assert "STANDALONE" in by_id["familywise_alpha"]["chosen"]
        assert str(ALPHA_STANDALONE_ONE_SIDED) in \
            by_id["familywise_alpha"]["chosen"]
        assert any("unadjusted one-sided claim" in r_
                   for r_ in by_id["familywise_alpha"]["rejected"])
        # the test decision names every part of a test
        chosen = by_id["primary_test"]["chosen"]
        for needle in ("entity-macro", "sign-flip", str(N_PERMUTATIONS),
                       str(PERMUTATION_SEED), "greater", "less", "Holm"):
            assert needle in chosen, needle
        # and the retention decision states a route and a count
        ret = by_id["retention_probe_allocation"]["chosen"]
        assert str(CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY) in ret
        assert "345" in ret and "70" in ret
        assert any("no retention probes" in r_ for r_ in
                   by_id["retention_probe_allocation"]["rejected"])
        # the summary blocks the decisions point at are present too
        assert p["primary_test"]["n_permutations"] == N_PERMUTATIONS
        assert p["primary_test"]["seed"] == PERMUTATION_SEED
        assert p["primary_test"]["directions"] == {"tga": "greater",
                                                   "filr": "less"}
        assert p["primary_test"]["holm_thresholds"] == [0.025, 0.05]
        assert len(p["primary_test"]["implementation_sha256"]) == 64
        assert p["retention_probes"]["route"] == CONFIRM_RETENTION_ROUTE
        assert p["retention_probes"]["total"] == 345

    def test_the_evidence_pointers_resolve_to_real_blocks(self):
        r = _load()
        for d in r["preregistration_decisions"]["decisions"]:
            if "evidence" in d:
                assert d["evidence"] in r, d["id"]

    def test_the_chosen_photograph_route_is_the_provenanced_one(self):
        """The offline pool was rejected despite being free and already on
        disk, because it records no license or attribution — the same
        unverifiable-bytes gap 11R1 closed.  The chosen route covers every
        species and must be hash-verified before the freeze."""
        r = _load()
        d = next(x for x in r["preregistration_decisions"]["decisions"]
                 if x["id"] == "new_photograph_supply")
        assert d["species_covered"] == 36
        assert d["route"]["records_license_and_attribution"] is True
        assert "subset" in d["verification_required_before_freeze"]
        assert "disjoint" in d["verification_required_before_freeze"]
        rejected = " ".join(d["rejected"])
        assert "local_v1" in rejected
        # and the budget is justified by MDE, not by a power target the
        # stratum cannot meet
        assert "SECONDARY" in d["what_the_photographs_are_for"]
        # the grid is named as a grid, and the selected value sits beside it
        mdes = d["held_out_mde_by_candidate_probes_per_species"]
        assert set(mdes) == {str(m) for m in PROBE_OPTIONS}
        assert mdes["12"] < mdes["3"]
        assert mdes["12"] < 0.030
        assert d["held_out_mde_at_the_selected_design"] == \
            mdes[str(CONFIRM_NEW_PHOTOS_PER_SPECIES)]
        # a field named for a choice must not hold every alternative
        assert "chosen_ceiling" not in d
        assert "every candidate" in d["note_on_the_former_key_name"]


class TestTheConfirmationSizeIsSelected:
    """Review finding 1.  The report listed 3/6/9/12/24 photographs per
    species and selected none, and the field named for the chosen ceiling
    held every candidate ceiling.  A grid is the input to a decision; these
    tests pin the decision."""

    def test_one_row_of_the_photograph_grid_is_selected(self):
        r = _load()
        h = r["confirmation_size"]["held_out_photographs"]
        assert h["new_photographs_per_species"] == \
            CONFIRM_NEW_PHOTOS_PER_SPECIES == 12
        # Iteration 11C-R2's finding #2: the budget covers the TARGET species.
        # 36 was what the seeded re-fetch could cover, which includes the 6
        # retain-only species whose only purpose was the retain image families.
        assert h["species_covered"] == 30
        assert h["new_photographs_total"] == 360
        assert h["new_photographs_total"] == \
            h["new_photographs_per_species"] * h["species_covered"]
        assert h["already_allocated_per_species"] == 12
        census = r["entity_role_census"]["by_role"]
        assert h["species_covered"] == census["target"]["by_source"][
            "inaturalist"]
        assert census["retain"]["by_source"]["inaturalist"] == 6
        assert h["species_covered"] + 6 == \
            r["new_photograph_supply"]["seeded_refetch"]["species_covered"]
        assert "TARGET species" in h["species_covered_is"]
        assert "no confirmation purpose" in h["why_only_the_target_species"]
        # the selected value is a single number, not a grid
        assert isinstance(h["mde_at_the_selected_design_conservative_icc"],
                          float)
        # and it is the grid row it claims to be
        m = r["image_strata"]["held_out_photo"]["metrics"][
            "B3_minus_B0:tga"]["nested_sizing"]
        c = m["cluster_ceiling"]
        row = next(x for x in m["achieved_at_cluster_ceiling"]
                   if x["probes_per_entity"]
                   == CONFIRM_NEW_PHOTOS_PER_SPECIES)
        assert h["mde_at_the_selected_design_conservative_icc"] == \
            row[f"mde_at_{c}_clusters_conservative_icc_upper"]
        assert h["power_at_the_selected_design_conservative_icc"] == \
            row[f"power_at_{c}_clusters_conservative_icc_upper"]
        assert set(h["rejected_candidates"]) == {
            str(x) for x in PROBE_OPTIONS
            if x != CONFIRM_NEW_PHOTOS_PER_SPECIES}

    def test_the_fetch_is_named_as_a_command_line(self):
        """Stage 3 has to run something.  A size that is not expressed as the
        invocation that produces it gets re-derived, and a re-derived fetch
        is a different split."""
        r = _load()
        h = r["confirmation_size"]["held_out_photographs"]
        assert h["fetch_seed"] == CONFIRM_FETCH_SEED == 42
        assert h["fetch_images_per_species"] == \
            CONFIRM_FETCH_IMAGES_PER_SPECIES == 24
        assert h["fetch_out"] == CONFIRM_FETCH_OUT
        assert h["fetch_role"] == CONFIRM_FETCH_ROLE == "target"
        assert h["fetch_tag"] == CONFIRM_FETCH_TAG == "pilot100"
        cmd = h["fetch_command"]
        assert "fetch_inat_species.py" in cmd
        assert f"--seed {CONFIRM_FETCH_SEED}" in cmd
        assert f"--images-per-species {CONFIRM_FETCH_IMAGES_PER_SPECIES}" \
            in cmd
        assert f"--role {CONFIRM_FETCH_ROLE}" in cmd
        assert f"--tag {CONFIRM_FETCH_TAG}" in cmd
        assert f"--out {CONFIRM_FETCH_OUT}" in cmd
        # a COUNT is not a role: the retain-only species are interleaved
        # through SPECIES_LIST, so the first 30 entries are not the 30 targets
        assert "--limit-species" not in cmd
        assert "24 of the 30" in h[
            "why_the_command_names_a_role_and_not_a_count"]
        # 24 = 12 already allocated + 12 new.  This arithmetic is a SUPPLY
        # bound (at most 12 exploratory photographs exist per species, so a
        # 24-draw always leaves 12 disjoint), not the shuffle-nesting claim an
        # earlier revision made here; see
        # new_photograph_supply.seeded_refetch.why_the_draw_is_24.
        assert h["fetch_images_per_species"] == \
            h["already_allocated_per_species"] \
            + h["new_photographs_per_species"]
        # and the target is not the frozen pool the fetcher refuses to touch
        assert h["fetch_out"] != "data/raw/inaturalist/pilot_v1"

    def test_one_row_of_the_wording_grid_is_selected(self):
        r = _load()
        w = r["confirmation_size"]["new_wording_probes"]
        assert w["new_probes_per_target_person"] == \
            CONFIRM_NEW_WORDING_PROBES_PER_PERSON == 12
        assert w["target_persons"] == 42
        assert w["new_wording_probes_total"] == 504
        assert w["new_wording_probes_total"] == \
            w["new_probes_per_target_person"] * w["target_persons"]
        assert w["balanced"] is True
        assert w["exploratory_design_was_balanced"] is False
        assert w["exploratory_probes_per_person"] == {"3": 27, "6": 12,
                                                      "9": 3}
        assert w["power_at_the_selected_design_tga_conservative_icc"] >= 0.98
        assert w["mde_at_the_selected_design_tga_conservative_icc"] < \
            0.20
        assert "not the binding constraint" in w["why_not_fewer"]

    def test_the_totals_add_up(self):
        """Iteration 11C-R2's finding #2: three different totals, three
        different labels.  936 was published as the TARGET total when it was
        the whole allocated photograph budget over 36 species."""
        r = _load()
        t = r["confirmation_size"]["totals"]
        w = r["confirmation_size"]["new_wording_probes"]
        h = r["confirmation_size"]["held_out_photographs"]
        ra = r["retention_probe_allocation"]
        assert t["of_which_new_wordings_on_target_persons"] == \
            w["new_wording_probes_total"] == 504
        assert t["of_which_new_photographs_on_target_species"] == \
            h["new_photographs_total"] == 360
        assert t["new_target_probes"] == 864 == \
            t["of_which_new_photographs_on_target_species"] \
            + t["of_which_new_wordings_on_target_persons"]
        assert t["new_retention_probes"] == \
            ra["new_retention_probes_total"] == 345
        assert t["total_new_probes_allocated"] == 1209 == \
            t["new_target_probes"] + t["new_retention_probes"]
        assert len({t["new_target_probes"], t["new_retention_probes"],
                    t["total_new_probes_allocated"]}) == 3
        assert t["total_generations_at_three_scored_states"] == \
            3 * t["total_new_probes_allocated"]
        assert t["entity_clusters_for_the_primary_claims"] == \
            r["primary_estimand"]["entities_in_scope"] == 72
        assert t["entity_clusters_for_retention"] == \
            ra["distinct_entities_covered"] == 70
        # the mislabelling is corrected in the artifact with its own arithmetic
        corr = t["correction_to_the_previous_revision"]
        for needle in ("936", "864", "72", "504", "360", "6"):
            assert needle in corr, needle
        assert 504 + 12 * 36 == 936
        assert 12 * 6 == 72 and 936 - 72 == 864

    def test_the_entity_ids_are_frozen_and_hashed(self):
        """The entities can be frozen now because they already exist; the
        query ids cannot, because the probes do not.  Saying which is which
        is what keeps this a preregistration rather than a description of
        whatever stage 3 builds."""
        r = _load()
        f = r["confirmation_size"]["frozen_now"]
        census = r["entity_role_census"]["by_role"]
        assert f["target_entity_ids"] == census["target"]["entity_ids"]
        assert len(f["target_entity_ids"]) == 72
        assert f["target_entity_ids_sha256"] == \
            census["target"]["entity_ids_sha256"]
        assert f["target_entity_ids_sha256"] == \
            r["primary_estimand"]["entity_ids_sha256"]
        assert len(f["retain_entity_ids"]) == 70
        assert f["retain_entity_ids_sha256"] == \
            census["retain"]["entity_ids_sha256"]
        assert len(f["exploratory_template_ids"]) == 57
        assert len(f["exploratory_template_ids_sha256"]) == 64
        assert f["exploratory_photograph_sha256_manifest"].endswith(
            "image_manifest.json")

    def test_the_identifiers_that_do_not_exist_yet_are_bound_as_obligations(
            self):
        r = _load()
        s = r["confirmation_size"]["frozen_at_stage_3_before_any_scoring"]
        for key in ("confirmation_query_id_list_sha256",
                    "confirmation_template_ids_and_file_sha256",
                    "confirmation_photograph_sha256_manifest"):
            assert "to be committed" in s[key], key
        rules = " ".join(s["collision_rules"])
        assert "template_id" in rules and "sha256" in rules
        assert "query_id" in rules
        assert "go/no-go" in rules
        assert str(len(r["confirmation_size"]["frozen_now"][
            "exploratory_template_ids"])) in rules
        assert "do not exist yet" in s["why_these_cannot_be_frozen_here"]

    def test_the_selected_size_appears_in_the_notes(self):
        """The notes are what a reader reads.  A selection that lives only in
        a nested block is a selection a reviewer will not find."""
        r = _load()
        text = " ".join(r["notes"])
        cs = r["confirmation_size"]
        assert "SELECTED confirmation size" in text
        assert str(cs["held_out_photographs"]["new_photographs_total"]) \
            in text
        assert str(cs["new_wording_probes"]["new_wording_probes_total"]) \
            in text
        assert str(cs["totals"]["new_retention_probes"]) in text
        assert str(cs["totals"]["total_new_probes_allocated"]) in text
        assert cs["held_out_photographs"]["fetch_command"] in text
        # the primary test is in the notes too, with every part of it
        assert "PRIMARY TEST" in text
        for needle in ("Monte Carlo cluster sign-flip permutation",
                       str(N_PERMUTATIONS), str(PERMUTATION_SEED),
                       "'tga': 'greater'", "'filr': 'less'",
                       "[0.025, 0.05]", "paired_ci.py"):
            assert needle in text, needle
        # and the note does not still say the photographs serve the retain
        # image families, which the allocation omits
        size_note = next(n for n in r["notes"]
                         if n.startswith("SELECTED confirmation size"))
        assert "are NOT renewed" in size_note
        assert "secondary held-out measurement only" in size_note
        assert "and for the retain" not in size_note


# ── the retention allocation is measured, and its prose agrees ──────

class TestTheRetentionAllocationIsMeasuredNotAsserted:
    """Iteration 11C-R2's finding #2, second half.  The count per entity is
    chosen against the batch-layout noise floor, and the image route is
    omitted because of a measured supply constraint - so both the numbers and
    the sentences that quote them are checked against each other."""

    def test_the_route_and_count_are_the_declared_ones(self):
        ra = _load()["retention_probe_allocation"]
        assert ra["selected"] is True
        assert ra["route"] == CONFIRM_RETENTION_ROUTE == "text_only"
        assert ra["new_templates_per_entity"] == \
            CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY == 3
        assert ra["status"].startswith("DESCRIPTIVE")
        assert "no declared margin" in ra["status"]

    def test_the_probe_counts_are_the_entities_times_the_templates(self):
        ra = _load()["retention_probe_allocation"]
        m = CONFIRM_RETENTION_NEW_TEMPLATES_PER_ENTITY
        assert set(ra["per_metric"]) == {"retain_same", "retain_other"}
        for name, e in ra["per_metric"].items():
            assert e["route"] == ra["route"], name
            assert e["new_probes_per_entity"] == m, name
            assert e["new_probes"] == m * e["entities_carried_by"], name
            assert e["sigma2_between"] > 0 and e["sigma2_within"] > 0, name
            assert 0.0 <= e["icc_point_estimate"] <= 1.0, name
        assert ra["per_metric"]["retain_same"]["entities_carried_by"] == 70
        assert ra["per_metric"]["retain_other"]["entities_carried_by"] == 45
        assert ra["new_retention_probes_total"] == 210 + 135 == 345
        assert ra["new_retention_probes_total"] == sum(
            e["new_probes"] for e in ra["per_metric"].values())
        # retain_other's entities are a subset, so the union is retain_same's
        assert ra["distinct_entities_covered"] == 70
        assert ra["retain_other_entities_are_a_subset_of_retain_same"] is True

    def test_every_half_width_is_recomputable_from_its_own_variance_components(
            self):
        """The prose quotes these numbers, so a half-width that is not the
        formula applied to the block's own sigma2 values is a second copy."""
        from power_analysis_confirmation import half_width
        ra = _load()["retention_probe_allocation"]
        for name, e in ra["per_metric"].items():
            sb, sw = e["sigma2_between"], e["sigma2_within"]
            k = e["entities_carried_by"]
            m = e["new_probes_per_entity"]
            assert e["half_width_at_the_selected_count"] == round(
                half_width(math.sqrt(sb + sw / m), k), 4), name
            m_exp = e["exploratory_probes_per_entity_harmonic_mean"]
            assert e["half_width_at_the_exploratory_count"] == round(
                half_width(math.sqrt(sb + sw / m_exp), k), 4), name
            assert e["between_entity_floor_infinite_probes"] == round(
                half_width(math.sqrt(sb), k), 4), name
            assert e["probes_per_entity_that_would_match_the_exploratory_"
                     "precision"] == m_exp, name
            for sm, hw in e["half_width_at_rejected_counts"].items():
                assert hw == round(half_width(math.sqrt(sb + sw / int(sm)),
                                              k), 4), (name, sm)
            # more probes can only narrow the half-width
            counts = sorted({m, m_exp}
                            | {int(x) for x in
                               e["half_width_at_rejected_counts"]})
            widths = {c: half_width(math.sqrt(sb + sw / c), k) for c in counts}
            for a, b in zip(counts, counts[1:]):
                assert widths[b] <= widths[a] + 1e-12, (name, a, b)

    def test_the_selected_count_sits_at_or_above_the_noise_floor(self):
        """This is the reason the count is 3 and not 8.  Precision finer than
        the batch-layout floor cannot be distinguished from batched-decoding
        noise, so buying it spends generations on nothing."""
        ra = _load()["retention_probe_allocation"]
        floor = ra["batch_layout_noise_floor_on_retain_metrics"]
        assert floor == _load()["batch_layout_noise_floor"][
            "max_abs_retain_delta"]
        for name, e in ra["per_metric"].items():
            assert e["selected_count_is_at_or_above_the_noise_floor"] is True, \
                name
            assert e["half_width_at_the_selected_count"] >= floor, name
        assert set(str(m) for m in
                   CONFIRM_RETENTION_REJECTED_PROBES_PER_ENTITY) == \
            set(ra["per_metric"]["retain_same"]["half_width_at_rejected_counts"])
        # and the sentence quotes the count the table computes, not another
        why = ra["why_this_count_and_not_one_that_matches_the_exploratory_"
                 "precision"]
        assert str(ra["per_metric"]["retain_same"][
            "probes_per_entity_that_would_match_the_exploratory_precision"]) \
            in why
        assert str(floor) in why
        assert "cannot resolve" in why

    def test_the_image_route_omission_rests_on_a_measured_supply(self):
        """Not a preference.  The persons cannot be given a new photograph
        because there is no pool of new photographs of a real person, and each
        of them has exactly one in pilot100_v2."""
        ra = _load()["retention_probe_allocation"]
        m = ra["media_supply_the_route_decision_rests_on"]
        assert m["retention_entities"] == 70
        assert m["photographs_per_retention_entity"] == {"1": 64, "12": 6}
        assert m["entities_with_a_single_photograph_or_none"] == 64
        assert m["by_source"] == {"inaturalist": 6, "mllmu_hier": 64}
        assert m["entities_a_new_photograph_could_cover"] == 6
        assert m["entities_no_new_photograph_can_cover"] == 64
        assert m["entities_a_new_photograph_could_cover"] + \
            m["entities_no_new_photograph_can_cover"] == 70
        assert m["sources_no_new_photograph_can_be_fetched_for"] == [
            "mllmu_hier"]
        why = ra["why_the_image_route_is_omitted"]
        assert "64" in why and "reuses exploratory media" in why
        cons = ra["consequence_for_the_photograph_fetch"]
        assert "TARGET species only" in cons
        assert "36" in cons
        # the consequence is visible in the size block, not only here
        assert _load()["confirmation_size"]["held_out_photographs"][
            "species_covered"] == 30

    def test_the_confirmation_retention_estimand_is_not_the_exploratory_one(
            self):
        ra = _load()["retention_probe_allocation"]
        assert "TEXT route only" in \
            ra["what_the_confirmation_retention_estimand_is"]
        assert "BOTH routes" in \
            ra["how_it_differs_from_the_exploratory_retention_number"]
        assert "must not be subtracted" in \
            ra["how_it_differs_from_the_exploratory_retention_number"]


# ── the portrait reuse is measured, not assumed ─────────────────────

class TestThePortraitReuseIsMeasuredNotAsserted:
    """Iteration 11C stage 3.

    The sealed rules required every confirmation photograph sha256 to be new.
    For the 30 target species that is satisfiable, and the fetch satisfies it.
    For the 42 target persons it is not: each has exactly one portrait, the
    wording stratum is image-route, and a text-route probe carries
    ``image_split = None`` so it is in neither primary stratum.  The exemption
    the freeze grants is worth exactly as much as the measurement that forces
    it, so the measurement is published beside it rather than summarised.
    """

    def test_the_wording_stratum_is_measured_to_be_image_route(self):
        p = _load()["portrait_reuse_exemption"]
        assert p["wording_stratum_probes"] == 180
        assert p["wording_stratum_is_entirely_image_route"] is True
        assert sum(p["wording_stratum_routes"].values()) == \
            p["wording_stratum_probes"]
        assert set(p["wording_stratum_routes"]) == \
            {"image_text_to_text", "image_to_text"}
        assert "text_to_text" not in p["wording_stratum_routes"]
        assert p["wording_stratum_image_seen_in_training"] == {"True": 180}

    def test_the_stratum_uses_the_families_the_design_mirrors(self):
        """The 3 x 4 construction is not a free choice of shape: it mirrors the
        3 families the exploratory stratum actually used, measured here."""
        p = _load()["portrait_reuse_exemption"]
        assert p["wording_stratum_num_families"] == \
            CONFIRM_WORDING_FAMILIES == 3
        assert sorted(p["wording_stratum_families"]) == [
            "image_fine_direct", "image_target_direct",
            "multimodal_image_text"]
        assert len(p["wording_stratum_exploratory_template_ids"]) == \
            CONFIRM_WORDING_FAMILIES * 3
        for tid in p["wording_stratum_exploratory_template_ids"]:
            fam, idx = tid.split(":")
            assert fam in p["wording_stratum_families"], tid
            assert idx.isdigit(), \
                f"a non-numeric exploratory index would make the new ids " \
                f"harder to keep disjoint: {tid}"
        # and the confirmation builds 4 NEW templates on each of the 3
        assert CONFIRM_NEW_WORDING_PROBES_PER_PERSON == \
            CONFIRM_WORDING_FAMILIES * CONFIRM_NEW_TEMPLATES_PER_FAMILY == 12

    def test_a_text_route_probe_would_be_in_neither_stratum(self):
        """The rejected alternative, measured: text-route does not buy the same
        stratum more cheaply, it leaves the 42 persons with no stratum."""
        p = _load()["portrait_reuse_exemption"]
        assert p["text_route_test_probes"] > 0
        assert p["text_route_image_split_values"] == ["None"]
        assert p["text_route_probes_are_in_neither_primary_stratum"] is True
        w = p["why_that_decides_the_route"]
        assert "no probe in either stratum" in w
        assert "72 clusters to 30" in w

    def test_every_target_person_has_exactly_one_photograph(self):
        p = _load()["portrait_reuse_exemption"]
        assert p["target_persons"] == 42
        assert p["photographs_per_target_person"] == {"1": 42}
        assert p["every_target_person_has_exactly_one_photograph"] is True
        assert len(set(p["target_person_ids"])) == 42
        # the wording stratum's clusters ARE those persons, and the size block
        # budgets the same number
        assert p["wording_stratum_clusters"] == p["target_persons"]
        assert _load()["confirmation_size"]["new_wording_probes"][
            "target_persons"] == p["target_persons"]

    def test_the_exempted_portraits_are_the_exploratory_ones(self):
        """An exemption that listed hashes the exploratory run never used would
        exempt nothing, and would hide a mis-measured list behind a plausible
        count."""
        p = _load()["portrait_reuse_exemption"]
        man = REPO_ROOT / EXPLORATORY_IMAGE_MANIFEST
        if not man.exists():
            pytest.skip(f"committed manifest not present: {man}")
        images = json.loads(man.read_text())["images"]
        ports = p["portraits"]
        assert ports["count"] == len(ports["paths"]) == 42
        assert len(set(ports["sha256"])) == len(ports["sha256"]) == 42
        assert len(set(ports["image_ids"])) == 42
        assert all(path in images for path in ports["paths"])
        assert {images[path]["sha256"] for path in ports["paths"]} == \
            set(ports["sha256"])
        assert p["portraits_that_are_exploratory_media"] == 42

    def test_the_set_hash_is_recomputable_from_its_own_list(self):
        r = _load()
        ports = r["portrait_reuse_exemption"]["portraits"]
        assert hashlib.sha256(
            "\n".join(ports["sha256"]).encode()).hexdigest() == \
            ports["set_sha256"]
        fn = r["confirmation_size"]["frozen_now"]
        assert fn["required_repeat_portrait_set_sha256"] == \
            ports["set_sha256"]
        assert fn["required_repeat_portrait_sha256"] == ports["sha256"]
        assert fn["required_repeat_portrait_count"] == ports["count"]
        assert len(fn["required_repeats_are"]) == 2
        assert "target-association set" in fn["required_repeats_are"][0]
        assert "42" in fn["required_repeats_are"][1]

    def test_the_gate_exposure_is_disclosed_not_hidden(self):
        """All 42 portraits were in probes the reference-state gate scored.
        The report says so next to the exemption instead of letting the reader
        infer that the person side is now as clean as the species side."""
        p = _load()["portrait_reuse_exemption"]
        assert p["portraits_the_exploratory_gate_exercised"] == 42
        assert p["portraits_exercised_by_the_gate_are_all_of_them"] is True
        assert "REQUIRED repeat and not a leak" in p["what_must_repeat"]
        assert p["what_is_still_new_on_those_probes"] == \
            ["query_id", "template_id", "template text"]
        assert "UNSEEN wording" in \
            p["why_the_novelty_that_matters_survives"]
        assert "indistinguishable from zero" in p["why_not_species_only"]

    def test_the_collision_rule_states_the_exception_and_its_boundary(self):
        rules = _load()["confirmation_size"][
            "frozen_at_stage_3_before_any_scoring"]["collision_rules"]
        hit = [r for r in rules if "bounded exception" in r]
        assert len(hit) == 1, hit
        r = hit[0]
        assert "42 target-person portraits" in r
        assert "360 species photographs are still required to be new" in r
        assert "cannot grow" in r
        assert "forbids the frozen estimand" in r
        # and the general no-reuse rule names BOTH required repeats
        gen = [x for x in rules if "beyond the two required repeats" in x]
        assert len(gen) == 1
        assert "new template TEXT" in gen[0]
        assert "hashed target-person portraits" in gen[0]

    def test_the_decision_records_what_it_rejected(self):
        d = [x for x in _load()["preregistration_decisions"]["decisions"]
             if x.get("id") == "portrait_reuse"]
        assert len(d) == 1
        assert d[0]["evidence"] == "portrait_reuse_exemption"
        assert "42" in d[0]["chosen"]
        assert "REQUIRED repeat" in d[0]["chosen"]
        assert len(d[0]["rejected"]) == 4
        joined = " ".join(d[0]["rejected"])
        assert "text-route" in joined
        assert "species only" in joined
        assert "72 clusters to 30" in joined
        assert "no boundary" in joined


# ── the primary test is read off the code that implements it ────────

class TestThePrimaryTestSpecificationIsReadOffTheCode:
    """Iteration 11C-R2's finding #3.  A specification written beside the code
    is a second copy of it; the report reads the draw count, the seed and the
    parameter lists off ``inspect.signature`` and hashes the module, so the
    two cannot describe different tests."""

    def test_the_values_are_the_signatures_not_a_restatement(self):
        import inspect
        from granunlearn.evaluation.paired_ci import (
            holm_family, one_sided_permutation_pvalue)
        pt = _load()["primary_test"]
        perm = inspect.signature(one_sided_permutation_pvalue).parameters
        assert pt["n_permutations"] == perm["n_permutations"].default
        assert pt["permutation_seed"] == perm["seed"].default
        assert pt["n_permutations_declared_here"] == N_PERMUTATIONS
        assert pt["permutation_seed_declared_here"] == PERMUTATION_SEED
        assert pt["implementation"]["signatures"][
            "one_sided_permutation_pvalue"] == list(perm)
        assert pt["implementation"]["signatures"]["holm_family"] == \
            list(inspect.signature(holm_family).parameters)

    def test_the_specification_and_the_implementation_agree(self):
        a = _load()["primary_test"]["specification_and_implementation_agree"]
        assert a["all_agree"] is True
        for key in ("n_permutations_is_the_declared_count",
                    "seed_is_the_declared_seed",
                    "claim_direction_covers_exactly_the_primary_metrics",
                    "holm_family_takes_alpha_as_a_required_argument"):
            assert a[key] is True, key
        read = a["every_value_read_off_the_implementation"]
        assert read["n_permutations"] == N_PERMUTATIONS
        assert read["seed"] == PERMUTATION_SEED
        assert read["directions"] == {"tga": "greater", "filr": "less"}

    def test_the_module_is_hashed_and_says_why_it_is_not_fingerprinted(self):
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES, sha256_file)
        pt = _load()["primary_test"]["implementation"]
        rel = pt["module"]
        assert rel == "src/granunlearn/evaluation/paired_ci.py"
        assert pt["sha256"] == sha256_file(REPO_ROOT / rel)
        assert rel not in CODE_FINGERPRINT_MODULES
        assert "would change the code fingerprint" in pt["why_the_hash_is_here"]
        assert "bound nowhere" in pt["why_the_hash_is_here"]

    def test_the_draw_count_is_set_by_its_own_monte_carlo_error(self):
        """1000 draws - what the CI bootstrap uses - have a standard error at
        the Holm threshold of a fifth of the threshold, so draw noise alone
        could move a claim across it."""
        pt = _load()["primary_test"]
        thr = HOLM_WORST_CASE_ALPHA
        assert pt["n_permutations"] == N_PERMUTATIONS == 10000
        assert pt["smallest_reportable_p_value"] == round(
            1.0 / (N_PERMUTATIONS + 1), 6)
        se = pt["monte_carlo_se_at_the_holm_threshold"]
        assert se == round(math.sqrt(thr * (1 - thr) / (N_PERMUTATIONS + 1)), 6)
        assert se < thr / 10
        se_1000 = math.sqrt(thr * (1 - thr) / 1001)
        assert se_1000 > thr / 6, "the rejected draw count must be shown to " \
            "be too coarse, or the choice looks arbitrary"
        assert str(round(se_1000, 6)) in pt["why_that_many_draws"]
        assert pt["observed_vector_included_in_the_null"] is True
        seeds = {pt["permutation_seed"], pt["seed_is_distinct_from"][
            "ci_bootstrap"], pt["seed_is_distinct_from"]["icc_bootstrap"]}
        assert len(seeds) == 3, seeds
        assert pt["seed_is_distinct_from"]["icc_bootstrap"] == BOOTSTRAP_SEED

    def test_the_holm_block_carries_the_whole_rule(self):
        m = _load()["primary_test"]["multiplicity"]
        k = len(PRIMARY_FAMILY)
        assert m["procedure"] == "Holm step-down"
        assert m["familywise_alpha"] == FAMILYWISE_ALPHA
        assert m["thresholds"] == [round(FAMILYWISE_ALPHA / (k - i), 6)
                                   for i in range(k)]
        assert m["worst_case_alpha_for_a_single_claim"] == round(
            HOLM_WORST_CASE_ALPHA, 6)
        assert m["ordering"] == "ascending p-value, ties broken by claim name"
        assert "cannot change a verdict" in m["tie_handling"]
        assert "FIRST non-rejection ends the procedure" in m["pass_fail_rule"]
        assert "understates the familywise error rate" in \
            m["why_the_stopping_rule_is_part_of_the_rule"]

    def test_the_achieved_level_was_measured_at_the_frozen_draw_count(self):
        """Calibrating at a smaller draw count measures a DIFFERENT procedure:
        the p-values resolve on a different grid."""
        pt = _load()["primary_test"]
        cal = pt["achieved_level_under_the_real_null"]
        assert set(cal) == set(PRIMARY_FAMILY)
        for claim, e in cal.items():
            assert e["permutations_per_replicate"] == N_PERMUTATIONS, claim
            assert e["permutations_are_the_frozen_count"] is True, claim
            assert e["replicates"] == CALIBRATION_REPLICATES >= 1000, claim
            assert e["threshold"] == HOLM_WORST_CASE_ALPHA, claim
            assert e["num_clusters"] == 72, claim
            assert e["achieved_level_is_nominal_within_two_se"] is True, claim
            lo, hi = e["two_se_band_around_the_nominal"]
            assert lo < HOLM_WORST_CASE_ALPHA < hi
            assert lo <= e["achieved_level_at_the_threshold"] <= hi, claim
            assert e["achieved_over_nominal"] < 1.5, claim
            assert 0.4 < e["mean_p_under_the_null"] < 0.6, claim
            assert e["sign_flips_shared_with_the_other_claims"] is True, claim
            # the discreteness that makes the calibration worth running
            assert 0 < e["zero_difference_fraction"] < 0.5, claim
            assert e["distinct_nonzero_differences"] < e["num_clusters"], claim

    def test_the_familywise_rate_is_measured_on_the_real_procedure(self):
        fw = _load()["primary_test"][
            "achieved_familywise_level_under_the_global_null"]
        assert fw["nominal"] == FAMILYWISE_ALPHA
        assert fw["procedure"] == "holm_family as implemented, not a " \
            "restatement"
        assert fw["one_sign_vector_per_entity_shared_by_both_claims"] is True
        assert fw["replicates"] == CALIBRATION_REPLICATES
        assert fw["achieved_is_nominal_within_two_se"] is True
        lo, hi = fw["two_se_band_around_the_nominal"]
        assert lo < FAMILYWISE_ALPHA < hi
        assert lo <= fw["achieved"] <= hi
        assert fw["achieved_over_nominal"] < 1.5
        # the marginal levels do NOT imply this number, which is why both are
        # reported
        assert "not the per-claim marginal levels" in fw["what_it_measures"]

    def test_the_alpha_convention_separates_its_two_sources(self):
        """0.025 is Holm's first threshold for k = 2 AND one arm of a
        two-sided 95% interval.  It is NOT what one-sidedness costs: a
        standalone directional claim at familywise 0.05 is tested at 0.05."""
        d = _load()["design"]
        assert d["familywise_alpha"] == FAMILYWISE_ALPHA
        assert d["alpha_one_sided"] == ALPHA_ONE_SIDED == 0.025
        assert d["alpha_standalone_one_sided"] == \
            ALPHA_STANDALONE_ONE_SIDED == 0.05
        conv = d["alpha_convention"]
        assert "unadjusted" not in conv
        assert "does NOT halve" in conv
        assert "STANDALONE" in conv
        assert "0.0167" in conv, "the k = 3 counterexample is what shows the " \
            "coincidence is a property of k = 2"
        assert "TOST" in conv
        assert round(FAMILYWISE_ALPHA / 3, 4) != ALPHA_ONE_SIDED
