"""Iteration 11C-R2 finding #3: the primary hypothesis test itself.

The freeze and the power report both describe this test, and a description
that has never been executed against its own implementation is a second copy
of it.  These tests run the implementation.

Three kinds of assertion live here.

* **Exactness.**  For small ``k`` the sign-flip null can be ENUMERATED -
  all ``2^k`` vectors - so the Monte Carlo p-value is checked against the
  exact one it estimates, with a tolerance derived from the binomial standard
  error of the draw count rather than a number that looks tight.
* **Degenerate cases.**  Every difference zero, a single cluster, zeros that
  cannot flip, an empty input.  These are where a resampling routine quietly
  returns something meaningless, and the confirmation's own differences are
  7-14% exactly zero, so they are not hypothetical.
* **The rule.**  Holm's thresholds, ordering, tie handling and stopping rule,
  including the property that makes it a procedure rather than a per-claim
  test: a claim is retained when an EARLIER step failed even if its own
  p-value clears its own threshold.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = (REPO_ROOT / "data" / "reports"
               / "mllmu_pilot100_confirmation_freeze.json")

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from granunlearn.evaluation.paired_ci import (  # noqa: E402
    CLAIM_DIRECTION,
    holm_family,
    one_sided_permutation_pvalue,
    paired_rate_diff_ci,
)

ALPHA = 0.05


def _exact_sign_flip_p(diffs, direction, statistic=None):
    """Enumerate all 2^k sign flips.  Feasible at the small k used here.

    The observed vector is one of the 2^k, so the exact p already includes
    it - which is why the Monte Carlo estimator
    ``(1 + Binomial(n, p_exact)) / (n + 1)`` is unbiased for this value.
    """
    k = len(diffs)
    obs = sum(diffs) / k if statistic is None else statistic
    means = [sum(s * d for s, d in zip(signs, diffs)) / k
             for signs in itertools.product((1.0, -1.0), repeat=k)]
    tail = [m for m in means
            if (m >= obs if direction == "greater" else m <= obs)]
    return len(tail) / len(means)


def _mc_tolerance(p_exact, n_permutations, n_sd=4.0):
    """Binomial standard error of a Monte Carlo p-value, times ``n_sd``.

    Plus one grid step, because the estimator reports on a 1/(n+1) grid and
    so cannot land exactly on a value between grid points.
    """
    return (n_sd * math.sqrt(p_exact * (1 - p_exact) / (n_permutations + 1))
            + 1.0 / (n_permutations + 1))


def _tie_mass(diffs: list[float],
              exact_diffs: list[Fraction]) -> tuple[Fraction, Fraction]:
    """The mass of sign flips whose statistic EQUALS the observed one.

    Returned twice over the SAME enumeration: once from exact rational
    arithmetic, which is the mathematical value and cannot depend on a
    summation order, and once from a raw float ``==``, which can.  The gap
    between the two is the platform-dependence documented in
    ``test_decimal_differences_make_float_ties_platform_dependent``.
    """
    k = len(diffs)
    obs_exact = sum(exact_diffs) / k
    obs_float = sum(diffs) / k
    n_exact = n_float = 0
    for signs in itertools.product((1, -1), repeat=k):
        if sum(Fraction(s) * d
               for s, d in zip(signs, exact_diffs)) / k == obs_exact:
            n_exact += 1
        if sum(float(s) * d for s, d in zip(signs, diffs)) / k == obs_float:
            n_float += 1
    return Fraction(n_exact, 2 ** k), Fraction(n_float, 2 ** k)


# ── the sign-flip p-value is the exact one it claims to estimate ────

class TestTheSignFlipPValueMatchesEnumeration:
    @pytest.mark.parametrize("diffs", [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -2.0, 3.0, 0.5],
        [0.0, 1.0, -2.0, 0.0],
        [0.25, 0.25, -0.25, 0.75, 0.0, 0.5],
        [-1.0] * 5 + [2.0],
    ])
    @pytest.mark.parametrize("direction", ["greater", "less"])
    def test_the_monte_carlo_p_is_the_enumerated_one(self, diffs, direction):
        n = 200000
        exact = _exact_sign_flip_p(diffs, direction)
        r = one_sided_permutation_pvalue(diffs, direction, n, seed=11)
        assert r is not None
        assert abs(r["p_value_one_sided"] - exact) <= \
            _mc_tolerance(exact, n), (diffs, direction, exact, r)

    def test_the_two_directions_partition_the_null(self):
        """`greater` and `less` on the same vector must account for the whole
        null up to the mass exactly AT the observed statistic, which both
        tails include.  Without that identity the two directions are not two
        readings of one distribution.

        The differences are INTEGER-valued on purpose.  The identity is about
        the mass of draws exactly EQUAL to the observed statistic, and
        "exactly equal" is only decidable in floating point when every signed
        sum is exactly representable: then ``sum`` in any order, numpy's
        pairwise ``mean``, and the rational value all agree, on every
        platform.  With decimal differences they do not -- see
        ``test_decimal_differences_make_float_ties_platform_dependent``.
        """
        diffs = [3.0, -7.0, 11.0, 0.0, -2.0, 9.0]
        exact_diffs = [Fraction(int(d)) for d in diffs]
        at_exact, at_float = _tie_mass(diffs, exact_diffs)
        #: The float count must equal the exact one BECAUSE the inputs are
        #: integers -- asserting the agreement is what makes the choice of
        #: input load-bearing rather than incidental.
        assert at_float == at_exact == Fraction(4, 64), (at_float, at_exact)
        n = 200000
        g = one_sided_permutation_pvalue(diffs, "greater", n, seed=3)
        l = one_sided_permutation_pvalue(diffs, "less", n, seed=3)
        assert abs(g["p_value_one_sided"] + l["p_value_one_sided"]
                   - (1.0 + float(at_exact))) <= _mc_tolerance(0.5, n)

    def test_decimal_differences_make_float_ties_platform_dependent(self):
        """WHY the identity above uses integers, measured rather than asserted.

        On the decimal vector below the mathematical tie mass is 4/64, but
        counting ties with a raw float ``==`` finds only 2/64: two sign
        vectors whose signed sums are mathematically equal to the observed one
        round to a different float, because the summation order differs.  The
        count therefore depends on how the platform and the summation routine
        round -- Python's left-to-right ``sum`` and numpy's pairwise ``mean``
        do not agree -- so an identity asserted against it passes on one
        machine and fails on another.

        This is a defect in the TEST, not in the estimator: the production
        tails compare with ``>=`` and ``<=`` against the same float the draws
        were computed from, so they are self-consistent, and a draw that is
        mathematically tied to the observed statistic is worth at most
        ``1/(n+1)`` of p-value.  It is pinned here so the identity test is not
        "fixed" back onto a decimal vector, where it is not reproducible.
        """
        diffs = [1.0, 0.8, -0.2, -0.7, 0.0, -0.3]
        exact_diffs = [Fraction(1), Fraction(4, 5), Fraction(-1, 5),
                       Fraction(-7, 10), Fraction(0), Fraction(-3, 10)]
        #: The decimal literals really are these rationals, so the exact
        #: enumeration is of the same vector and not of an idealised one.
        assert [float(d) for d in exact_diffs] == diffs
        at_exact, at_float = _tie_mass(diffs, exact_diffs)
        assert at_exact == Fraction(4, 64)
        #: The float count is AT MOST the exact one -- a mathematically tied
        #: draw can fail to compare equal, but a draw that is not tied can
        #: never compare equal.  On this machine it is strictly smaller, which
        #: is the whole point; the bound is <= so it holds everywhere.
        assert at_float <= at_exact
        assert at_exact - at_float <= Fraction(2, 64)

    def test_the_statistic_is_the_mean_of_the_per_entity_differences(self):
        diffs = [0.1, -0.2, 0.0, 0.45, 0.05]
        r = one_sided_permutation_pvalue(diffs, "greater", 1000, seed=5)
        assert r["statistic"] == round(sum(diffs) / len(diffs), 6)
        assert r["num_clusters"] == len(diffs)
        assert "mean over entities" in r["statistic_is"]

    def test_it_is_the_same_statistic_the_published_interval_covers(self):
        """A p-value for one quantity beside an interval for another lets a
        claim be rejected while the interval still contains zero."""
        fa = {"q1": (1, "e1"), "q2": (0, "e1"), "q3": (1, "e2"),
              "q4": (1, "e2"), "q5": (0, "e3")}
        fb = {"q1": (0, "e1"), "q2": (0, "e1"), "q3": (1, "e2"),
              "q4": (0, "e2"), "q5": (1, "e3")}
        ci = paired_rate_diff_ci(fa, fb, n_bootstrap=500, seed=42)
        from granunlearn.evaluation.paired_ci import _paired_unit_diffs
        diffs = _paired_unit_diffs(fa, fb)["diffs"]
        p = one_sided_permutation_pvalue(diffs, "greater", 5000, seed=42)
        assert ci is not None
        assert round(p["statistic"], 4) == ci["diff"]
        assert p["num_clusters"] == ci["num_units"] == 3

    def test_a_strong_effect_reaches_the_smallest_reportable_p(self):
        """The observed vector is in the null, so p = 0 is never reportable
        and a claim can never look more certain than the draw count allows.

        ``k`` is 20 rather than 12 on purpose: at 12 clusters the all-plus
        sign flip has probability 2^-12, so a 5000-draw null contains one
        about once in every three runs and the smallest p the routine reports
        is 2/(n+1), not 1/(n+1).  The resolution claim only holds where the
        null cannot reproduce the observed vector by chance.
        """
        n = 5000
        r = one_sided_permutation_pvalue([1.0] * 20, "greater", n, seed=9)
        assert r["p_value_one_sided"] == r["p_smallest_reportable"] == \
            round(1.0 / (n + 1), 6)
        assert r["p_value_one_sided"] > 0.0

    def test_a_shallow_effect_cannot_report_below_its_own_grid(self):
        """The same routine at k = 12, where the null CAN reproduce the
        observed vector: the reported p is at least one grid step above the
        minimum, which is the honest behaviour and the reason the draw count
        is frozen rather than chosen after the fact."""
        n = 5000
        r = one_sided_permutation_pvalue([1.0] * 12, "greater", n, seed=9)
        assert r["p_value_one_sided"] >= r["p_smallest_reportable"]
        assert r["p_value_one_sided"] < 3 * r["p_smallest_reportable"]

    def test_the_reported_resolution_is_below_the_holm_threshold(self):
        """Otherwise `p <= 0.025` would be an artefact of the draw count."""
        from power_analysis_confirmation import (  # noqa: E402
            HOLM_WORST_CASE_ALPHA, N_PERMUTATIONS)
        r = one_sided_permutation_pvalue([1.0] * 8, "greater",
                                         N_PERMUTATIONS, seed=9)
        assert r["p_smallest_reportable"] < HOLM_WORST_CASE_ALPHA / 10


# ── the degenerate cases a resampling routine gets quietly wrong ────

class TestTheDegenerateInputsAreHandledNotAveraged:
    def test_every_difference_zero_gives_p_one_in_both_directions(self):
        """No sign flip can move the statistic, so the observed value is the
        whole null and every draw ties it.  Both tails therefore contain
        everything: p = 1, not p = 0.5 and not an error."""
        for direction in ("greater", "less"):
            r = one_sided_permutation_pvalue([0.0] * 7, direction, 2000,
                                             seed=1)
            assert r["p_value_one_sided"] == 1.0, direction
            assert r["all_differences_zero"] is True, direction
            assert r["num_flippable_clusters"] == 0, direction
            assert r["statistic"] == 0.0

    def test_zeros_cannot_flip_and_the_report_says_how_many_can(self):
        diffs = [0.0, 0.0, 0.0, 1.0, -1.0]
        r = one_sided_permutation_pvalue(diffs, "greater", 20000, seed=2)
        assert r["num_clusters"] == 5
        assert r["num_flippable_clusters"] == 2
        assert r["zero_difference_fraction"] == 0.6
        # The three zeros contribute nothing, so the null mean only ever
        # takes THREE values, -2/5, 0 and +2/5, from the 4 sign combinations
        # of the two flippable clusters - not the 32 that k = 5 suggests.
        # The observed statistic is 0, which two of the four combinations
        # reach and one exceeds, so the exact tail is 3/4.
        distinct = {round(sum(s * d for s, d in zip(signs, diffs)) / 5, 9)
                    for signs in itertools.product((1.0, -1.0), repeat=5)}
        assert distinct == {-0.4, 0.0, 0.4}
        exact = _exact_sign_flip_p(diffs, "greater")
        assert exact == 0.75
        assert exact == _exact_sign_flip_p([1.0, -1.0], "greater"), \
            "the zeros must leave the tail probability unchanged"
        assert abs(r["p_value_one_sided"] - exact) <= _mc_tolerance(
            exact, 20000)
        assert "resolution" in r["why_that_matters"]

    def test_a_single_cluster_gives_the_only_two_answers_there_are(self):
        """k = 1 has 2 sign flips, so the exact p is 1/2 or 1.  A routine
        that resampled instead of flipping would produce something else."""
        assert one_sided_permutation_pvalue([0.4], "greater", 5000,
                                            seed=4)["p_value_one_sided"] == \
            pytest.approx(0.5, abs=0.02)
        assert one_sided_permutation_pvalue([-0.4], "greater", 5000,
                                            seed=4)["p_value_one_sided"] == \
            pytest.approx(1.0, abs=0.02)

    def test_an_empty_input_returns_none_rather_than_a_number(self):
        assert one_sided_permutation_pvalue([], "greater", 100, seed=1) is None

    def test_a_two_sided_alternative_is_refused(self):
        with pytest.raises(ValueError, match="two-sided"):
            one_sided_permutation_pvalue([1.0, 2.0], "two-sided", 100, seed=1)

    def test_a_non_positive_draw_count_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            one_sided_permutation_pvalue([1.0, 2.0], "greater", 0, seed=1)


# ── determinism, and the direction each claim actually runs in ──────

class TestTheTestIsReproducibleAndDirected:
    def test_the_same_seed_gives_the_same_p_value(self):
        diffs = [0.1, -0.3, 0.0, 0.5, 0.2, -0.1, 0.0, 0.4]
        a = one_sided_permutation_pvalue(diffs, "greater", 5000, seed=77)
        b = one_sided_permutation_pvalue(diffs, "greater", 5000, seed=77)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_a_different_seed_moves_p_only_within_its_own_noise(self):
        diffs = [0.1, -0.3, 0.0, 0.5, 0.2, -0.1, 0.0, 0.4]
        ps = [one_sided_permutation_pvalue(diffs, "greater", 5000, seed=s)
              ["p_value_one_sided"] for s in (1, 2, 3, 4, 5)]
        assert max(ps) - min(ps) < 4 * _mc_tolerance(sum(ps) / len(ps), 5000)

    def test_the_direction_decides_which_tail_is_counted(self):
        """The same vector, opposite conclusions: this is the whole reason the
        direction is declared per claim instead of absorbed into |theta|."""
        positive = [0.4, 0.3, 0.0, 0.5, 0.2, 0.6, 0.1, 0.35]
        g = one_sided_permutation_pvalue(positive, "greater", 20000, seed=8)
        l = one_sided_permutation_pvalue(positive, "less", 20000, seed=8)
        assert g["p_value_one_sided"] < 0.01
        assert l["p_value_one_sided"] > 0.99
        assert g["alternative"] == "theta > 0"
        assert l["alternative"] == "theta < 0"
        negated = [-d for d in positive]
        assert one_sided_permutation_pvalue(
            negated, "less", 20000, seed=8)["p_value_one_sided"] == \
            g["p_value_one_sided"]

    def test_the_two_primary_claims_point_in_opposite_directions(self):
        """TGA is an accuracy, so better is HIGHER; FILR is a leakage rate, so
        better is LOWER.  Both claims are about the same B3-B0 difference."""
        assert CLAIM_DIRECTION == {"tga": "greater", "filr": "less"}
        assert len(set(CLAIM_DIRECTION.values())) == 2

    def test_the_frozen_directions_are_the_module_ones(self):
        if not FREEZE_PATH.exists():
            pytest.skip(f"committed evidence not present: {FREEZE_PATH}")
        t = json.loads(FREEZE_PATH.read_text())["primary_test"]
        assert t["direction_by_metric"] == dict(CLAIM_DIRECTION)
        assert t["alternative_by_claim"] == {
            "B3_minus_B0:tga": "theta > 0",
            "B3_minus_B0:filr": "theta < 0"}

    def test_the_default_draw_count_and_seed_are_the_frozen_ones(self):
        """The defaults are what a scorer who passes nothing gets, so they
        have to BE the frozen values rather than merely agree with them."""
        import inspect
        sig = inspect.signature(one_sided_permutation_pvalue).parameters
        assert sig["n_permutations"].default == 10000
        assert sig["seed"].default == 20260908
        assert sig["seed"].default not in (42, 20260907), \
            "sharing a stream with the CI or ICC bootstrap would make their " \
            "Monte Carlo errors dependent"


# ── the level the test actually achieves ───────────────────────────

class TestTheAchievedLevelIsNominal:
    """A threshold is only a threshold if the procedure delivers it.  These
    differences are bounded, discrete and mostly exactly zero, which is where
    a resampling test's level departs from its nominal one."""

    @pytest.mark.parametrize("zeros", [0.0, 0.25, 0.75])
    def test_the_level_is_at_most_nominal_up_to_its_own_noise(self, zeros):
        rng = random.Random(20260909)
        k, n_rep, thr = 24, 400, ALPHA / 2
        # a null-shaped vector: symmetric magnitudes with `zeros` of it exactly
        # zero, which is what makes the p-values discrete
        mags = [round(rng.uniform(0.02, 0.6), 3) for _ in range(k)]
        diffs = [0.0 if rng.random() < zeros else
                 m * rng.choice((1.0, -1.0)) for m in mags]
        rejects = 0
        for _ in range(n_rep):
            signed = [d * rng.choice((-1.0, 1.0)) for d in diffs]
            if one_sided_permutation_pvalue(
                    signed, "greater", 4000,
                    rng.randrange(1, 2 ** 31))["p_value_one_sided"] <= thr:
                rejects += 1
        achieved = rejects / n_rep
        se = math.sqrt(thr * (1 - thr) / n_rep)
        assert achieved <= thr + 3 * se, (zeros, achieved, thr + 3 * se)

    def test_the_mean_p_under_the_null_is_about_a_half(self):
        """Sharper than the tail rate because every replicate contributes."""
        rng = random.Random(5)
        diffs = [0.0 if rng.random() < 0.2 else
                 rng.choice((1.0, -1.0)) * round(rng.uniform(0.05, 0.5), 3)
                 for _ in range(20)]
        ps = [one_sided_permutation_pvalue(
            [d * rng.choice((-1.0, 1.0)) for d in diffs], "greater", 3000,
            rng.randrange(1, 2 ** 31))["p_value_one_sided"]
            for _ in range(400)]
        assert 0.4 < sum(ps) / len(ps) < 0.6


# ── Holm: thresholds, ordering, ties, and the stopping rule ─────────

class TestHolmIsAProcedureAndNotAPerClaimTest:
    def test_the_thresholds_are_the_step_down_sequence(self):
        h = holm_family({"a": 0.4, "b": 0.3}, ALPHA)
        assert h["k"] == 2
        assert h["thresholds"] == [0.025, 0.05]
        assert h["thresholds_apply_to"] == "one-sided p-values"
        assert [s["threshold"] for s in h["steps"]] == [0.025, 0.05]
        h3 = holm_family({"a": 0.4, "b": 0.3, "c": 0.2}, ALPHA)
        assert h3["thresholds"] == [round(ALPHA / 3, 6), 0.025, 0.05]

    def test_both_claims_reject_when_both_clear(self):
        h = holm_family({"B3_minus_B0:tga": 0.004,
                         "B3_minus_B0:filr": 0.03}, ALPHA)
        assert h["rejected"] == ["B3_minus_B0:tga", "B3_minus_B0:filr"]
        assert h["retained"] == []
        assert h["all_rejected"] is True

    def test_the_first_failure_ends_the_procedure(self):
        """The claim with the SMALLER p-value fails its own threshold, and the
        other one clears its own - Holm still retains it.  Dropping the
        stopping rule would turn this into a per-comparison test at a smaller
        alpha and understate the familywise error rate."""
        h = holm_family({"B3_minus_B0:tga": 0.03,
                         "B3_minus_B0:filr": 0.04}, ALPHA)
        assert h["rejected"] == []
        assert sorted(h["retained"]) == ["B3_minus_B0:filr",
                                         "B3_minus_B0:tga"]
        assert h["all_rejected"] is False
        steps = h["steps"]
        assert steps[0]["clears_its_own_threshold"] is False
        assert steps[1]["clears_its_own_threshold"] is True
        assert steps[1]["rejected"] is False
        assert "an earlier step already failed" in steps[1]["why"]

    def test_a_failure_at_the_second_step_only_retains_the_second(self):
        h = holm_family({"a": 0.01, "b": 0.06}, ALPHA)
        assert h["rejected"] == ["a"]
        assert h["retained"] == ["b"]
        assert h["all_rejected"] is False

    def test_the_rejection_set_is_always_a_prefix_of_the_ordering(self):
        """Downward closure is what makes the procedure familywise-valid; a
        set that skipped a step would not be Holm."""
        rng = random.Random(17)
        for _ in range(300):
            ps = {f"c{i}": round(rng.uniform(0.0, 0.12), 5)
                  for i in range(rng.randint(1, 4))}
            h = holm_family(ps, ALPHA)
            flags = [s["rejected"] for s in h["steps"]]
            assert flags == [True] * sum(flags) + [False] * (len(flags)
                                                             - sum(flags)), ps
            assert set(h["rejected"]) == {s["claim"] for s in h["steps"]
                                          if s["rejected"]}
            assert not set(h["rejected"]) & set(h["retained"])
            assert set(h["rejected"]) | set(h["retained"]) == set(ps)

    def test_ties_are_ordered_by_name_and_cannot_change_a_verdict(self):
        a = holm_family({"zzz": 0.01, "aaa": 0.01}, ALPHA)
        b = holm_family({"aaa": 0.01, "zzz": 0.01}, ALPHA)
        assert [s["claim"] for s in a["steps"]] == \
            [s["claim"] for s in b["steps"]] == ["aaa", "zzz"]
        assert sorted(a["rejected"]) == sorted(b["rejected"]) == ["aaa", "zzz"]
        assert "cannot change a verdict" in a["tie_handling"]
        # a tie where BOTH clear the first threshold: both are rejected
        # whichever name comes first, so the tie-break changes the report
        # order and nothing else
        c = holm_family({"zzz": 0.02, "aaa": 0.02}, ALPHA)
        assert sorted(c["rejected"]) == ["aaa", "zzz"]
        # and a tie where neither clears alpha/2: nothing is rejected
        d = holm_family({"zzz": 0.03, "aaa": 0.03}, ALPHA)
        assert d["rejected"] == []
        assert [s["clears_its_own_threshold"] for s in d["steps"]] == \
            [False, True]

    def test_the_threshold_is_inclusive_at_the_step_that_sets_it(self):
        """k = 2, so the thresholds are alpha/2 = 0.025 and alpha = 0.05 and
        both boundaries are real ones.  At k = 1 the first threshold would be
        alpha itself, which is why the family size has to be part of the
        assertion rather than assumed from the number."""
        # the FIRST threshold, alpha/k
        assert holm_family({"a": 0.025, "b": 0.9}, ALPHA)["rejected"] == ["a"]
        assert holm_family({"a": 0.0250001, "b": 0.9}, ALPHA)["rejected"] == []
        # the LAST threshold, alpha - reachable only once step 1 cleared
        assert holm_family({"a": 0.05, "b": 0.01}, ALPHA)["rejected"] == \
            ["b", "a"]
        assert holm_family({"a": 0.0500001, "b": 0.01}, ALPHA)["rejected"] == \
            ["b"]
        # and a p-value at the LAST threshold still fails if step 1 failed
        assert holm_family({"a": 0.05, "b": 0.03}, ALPHA)["rejected"] == []

    def test_an_empty_family_rejects_nothing(self):
        h = holm_family({}, ALPHA)
        assert h["k"] == 0 and h["steps"] == []
        assert h["rejected"] == [] and h["all_rejected"] is False

    def test_an_impossible_alpha_is_refused(self):
        for bad in (0.0, 1.0, -0.05, 1.5):
            with pytest.raises(ValueError, match=r"\(0, 1\)"):
                holm_family({"a": 0.01}, bad)

    def test_holm_is_never_stricter_than_bonferroni(self):
        """Same first threshold, but Holm's later ones are weaker, so it can
        only reject at least as much.  A pair that disagreed would mean the
        step-down was implemented as a step-up."""
        rng = random.Random(23)
        for _ in range(300):
            ps = {f"c{i}": round(rng.uniform(0.0, 0.2), 5) for i in range(3)}
            k = len(ps)
            holm = set(holm_family(ps, ALPHA)["rejected"])
            bonf = {n for n, p in ps.items() if p <= ALPHA / k}
            assert bonf <= holm, ps

    def test_the_frozen_worked_examples_are_this_function_s_output(self):
        if not FREEZE_PATH.exists():
            pytest.skip(f"committed evidence not present: {FREEZE_PATH}")
        ex = json.loads(FREEZE_PATH.read_text())["primary_test"][
            "worked_examples"]
        assert ex["both_clear"] == holm_family(
            {"B3_minus_B0:tga": 0.004, "B3_minus_B0:filr": 0.03}, ALPHA)
        assert ex["first_fails_so_the_second_is_retained"] == holm_family(
            {"B3_minus_B0:tga": 0.03, "B3_minus_B0:filr": 0.04}, ALPHA)
