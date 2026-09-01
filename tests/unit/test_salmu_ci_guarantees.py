"""CI guarantees for the SALMU evaluation chain (10R5 closing tests).

Asserts ESTIMAND CORRESPONDENCE, DETERMINISM, SCHEMA, and EDGE CASES
— deliberately NOT general 95% statistical coverage from random
simulations (slow and flaky).

Covered:
* every CI contains its corresponding point estimate (pair leakage,
  identity leakage, unit-macro similarity);
* degenerate rates have degenerate CIs (all-zero -> [0,0],
  all-one -> [1,1]);
* CIs are reproducible under a fixed bootstrap seed;
* both GMUL target-subset levels carry CI fields;
* per-attribute aggregation in the committed reference and
  selected-test reports carries bootstrap CIs;
* single-unit strata (the 3-4-association per-attribute test strata)
  degenerate to the point estimate;
* n_bootstrap <= 0 / invalid ci_level are rejected;
* a shard with a changed checkpoint SHA-256 or benchmark revision is
  refused for reuse (forces rescoring).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from granunlearn.salmu.embedding_metrics import (
    aggregate_scores,
    validate_ci_params,
)
from granunlearn.salmu.official_metrics import (
    summarize_rows,
    summarize_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate_salmu_official_splits as official_splits  # noqa: E402

CI_FIELDS = ("unit_macro_assoc_sim_ci", "leakage_rate_ci",
             "identity_leakage_rate_ci")


def _mixed_rows() -> dict:
    """Four identities, ten rows, leakage strictly between 0 and 1 at
    BOTH levels and DIFFERENT at the two levels (pair 6/10 = 0.6;
    identity 2/4 = 0.5) so the two CIs can never be conflated."""
    return {
        "ids": ["a"] * 3 + ["b"] * 2 + ["c"] * 3 + ["d"] * 2,
        "assoc_sim": [0.8, 0.2, 0.6, 0.1, 0.3,
                      0.5, 0.4, 0.3, 0.2, 0.25],
        "generic_sim": [0.1, 0.5, 0.2, 0.4, 0.2,
                        0.2, 0.3, 0.6, 0.5, 0.1],
    }


class TestCIContainsCorrespondingPointEstimate:
    def test_ci_contains_corresponding_point_estimate(self):
        """Each CI must bracket the point estimate of the SAME
        estimand: pair-level rate <- pair flags, identity-level rate
        <- per-identity flags, unit-macro sim <- unit means."""
        rows = summarize_rows(split_name="forget", n_bootstrap=2000,
                              seed=42, **_mixed_rows())
        # Pair-level leakage + its pair-level CI
        assert rows["leakage_rate"] == 0.6
        lo, hi = rows["leakage_rate_ci"]
        assert lo <= rows["leakage_rate"] <= hi
        # Identity-level leakage + the identity-clustered CI
        assert rows["identity_leakage_rate"] == 0.5
        lo, hi = rows["identity_leakage_rate_ci"]
        assert lo <= rows["identity_leakage_rate"] <= hi
        # Unit-macro similarity + the unit-level CI
        lo, hi = rows["unit_macro_assoc_sim_ci"]
        assert lo <= rows["unit_macro_assoc_sim"] <= hi
        # Sanity: all three estimands are distinct objects here
        assert rows["num_units"] == 4 and rows["num_pairs"] == 10


class TestDegenerateRatesHaveDegenerateCI:
    def _rows(self, leak: bool, n_ids: int = 4):
        ids, assoc, generic = [], [], []
        for i in range(n_ids):
            for _ in range(3):  # 3 rows per identity
                ids.append(f"id{i}")
                if leak:
                    assoc.append(0.8)
                    generic.append(0.2)
                else:
                    assoc.append(0.2)
                    generic.append(0.8)
        return {"ids": ids, "assoc_sim": assoc,
                "generic_sim": generic}

    def test_degenerate_rates_have_degenerate_ci(self):
        # All-zero leakage: both rates 0 with CI exactly [0, 0]
        zero = summarize_rows(split_name="forget", n_bootstrap=500,
                              seed=42, **self._rows(leak=False))
        assert zero["leakage_rate"] == 0.0
        assert zero["leakage_rate_ci"] == (0.0, 0.0)
        assert zero["identity_leakage_rate"] == 0.0
        assert zero["identity_leakage_rate_ci"] == (0.0, 0.0)
        # All-one leakage: both rates 1 with CI exactly [1, 1]
        one = summarize_rows(split_name="forget", n_bootstrap=500,
                             seed=42, **self._rows(leak=True))
        assert one["leakage_rate"] == 1.0
        assert one["leakage_rate_ci"] == (1.0, 1.0)
        assert one["identity_leakage_rate"] == 1.0
        assert one["identity_leakage_rate_ci"] == (1.0, 1.0)

    def test_degenerate_unit_macro_similarity(self):
        """Identical unit means -> unit-macro CI collapses to the
        point estimate (every resample reproduces the same mean)."""
        ids = ["a"] * 2 + ["b"] * 3  # unequal row counts per unit
        assoc = [0.3] * 5
        rows = summarize_rows(ids=ids, assoc_sim=assoc,
                              generic_sim=[None] * 5,
                              split_name="forget", n_bootstrap=300,
                              seed=42)
        assert rows["unit_macro_assoc_sim"] == 0.3
        assert rows["unit_macro_assoc_sim_ci"] == (0.3, 0.3)


class TestCIReproducibleWithFixedSeed:
    def test_ci_reproducible_with_fixed_seed(self):
        kwargs = dict(split_name="forget", n_bootstrap=800,
                      **_mixed_rows())
        first = summarize_rows(seed=42, **kwargs)
        second = summarize_rows(seed=42, **kwargs)
        for field in CI_FIELDS:
            assert first[field] == second[field], field
        # A different seed changes the resamples (checked on this
        # fixed dataset — deterministic, not a statistical claim)
        other = summarize_rows(seed=7, **kwargs)
        assert any(first[f] != other[f] for f in CI_FIELDS)


class TestTargetSubsetsIncludeCIFields:
    def test_target_subsets_include_ci_fields(self):
        """Both GMUL subset levels are summarized by the same
        machinery and must carry the full CI schema."""
        per_split = {
            "forget": {
                "identity_id": ["t1", "t1", "t2", "r1"],
                "assoc_sim": [0.5, 0.7, 0.3, 0.2],
                "generic_sim": [0.2, 0.9, 0.1, 0.3],
                "gmul_target_mask": [True, True, True, False],
                "gmul_target_attr_mask": [True, False, True, False],
            },
        }
        summary = summarize_state(per_split, n_bootstrap=400)
        entry = summary["forget"]
        for label in ("gmul_target_subset", "gmul_target_attr_subset"):
            sub = entry[label]
            for field in CI_FIELDS:
                assert field in sub, (label, field)
                lo, hi = sub[field]
                assert lo <= hi, (label, field)
        # Point-estimate correspondence inside each subset too
        for label in ("gmul_target_subset", "gmul_target_attr_subset"):
            sub = entry[label]
            lo, hi = sub["unit_macro_assoc_sim_ci"]
            assert lo <= sub["unit_macro_assoc_sim"] <= hi


class TestPerAttributeAggregationReceivesBootstrapCI:
    """The committed reference-state and selection reports must carry
    association-bootstrap CIs on every per-attribute stratum."""

    RATE_CI = (("prefers_fine_rate", "prefers_fine_rate_ci"),
               ("prefers_target_rate", "prefers_target_rate_ci"),
               ("prefers_target_not_fine_rate",
                "prefers_target_not_fine_rate_ci"))

    @staticmethod
    def _report(name: str, suffix: str) -> dict:
        path = REPO_ROOT / "data" / "reports" / \
            f"{name}{('_' + suffix) if suffix else ''}.json"
        return json.loads(path.read_text())

    @pytest.mark.parametrize("suffix", ["", "r5"])
    def test_reference_report_per_attribute_ci(self, suffix):
        rep = self._report("salmu_reference_eval", suffix)
        assert rep["bootstrap_ci"] is True
        by_attr = rep["target_only_by_attribute"]
        assert set(by_attr) == {"BASE", "MF", "MG", "MN"}
        for state, attrs in by_attr.items():
            assert set(attrs) == {"city", "job", "blood_type"}
            for attr, e in attrs.items():
                for rate_key, ci_key in self.RATE_CI:
                    assert ci_key in e, (state, attr, ci_key)
                    lo, hi = e[ci_key]
                    assert lo <= e[rate_key] <= hi, \
                        (state, attr, ci_key)

    @pytest.mark.parametrize("suffix", ["", "r5"])
    def test_selected_test_report_per_attribute_ci(self, suffix):
        rep = self._report("salmu_unlearning_selection", suffix)
        assert rep["bootstrap_ci"] is True
        selected = set(rep["selected"].values())
        tested = [c for c in rep["candidates"]
                  if "test_per_attribute" in c]
        # Exactly the SELECTED checkpoints get the frozen-test
        # per-attribute breakdown (selected-checkpoint-only protocol)
        assert {c["candidate_id"] for c in tested} == selected
        for cand in tested:
            for attr, e in cand["test_per_attribute"].items():
                for rate_key, ci_key in self.RATE_CI:
                    assert ci_key in e, (cand["candidate_id"], attr)
                    lo, hi = e[ci_key]
                    assert lo <= e[rate_key] <= hi, \
                        (cand["candidate_id"], attr, ci_key)


class TestSingleUnitCI:
    """Per-attribute TEST strata are tiny (3-4 associations under the
    32/7/7 persona split); a single-unit stratum must degenerate
    exactly to its point estimate."""

    def test_single_unit_ci(self):
        # ONE identity unit with 3 pair rows: every bootstrap resample
        # picks the same unit, so the CI is the point estimate itself.
        rows = summarize_rows(
            ids=["solo"] * 3, assoc_sim=[0.31, 0.27, 0.35],
            generic_sim=[0.1, 0.5, 0.2], split_name="forget",
            n_bootstrap=500, seed=42)
        assert rows["num_units"] == 1
        lo, hi = rows["unit_macro_assoc_sim_ci"]
        assert (lo, hi) == (rows["unit_macro_assoc_sim"],) * 2

    def test_small_stratum_of_3_to_4_associations(self):
        # A realistic 3-4-association attribute stratum: CIs are
        # well-ordered and bracket the point estimates.
        rows = summarize_rows(
            ids=["a", "a", "b", "c", "c", "d"],
            assoc_sim=[0.41, 0.39, 0.22, 0.5, 0.44, 0.3],
            generic_sim=[0.2, 0.45, 0.3, 0.1, 0.5, 0.35],
            split_name="forget", n_bootstrap=2000, seed=42)
        assert rows["num_units"] == 4
        lo, hi = rows["unit_macro_assoc_sim_ci"]
        assert lo <= hi
        assert lo <= rows["unit_macro_assoc_sim"] <= hi
        lo, hi = rows["identity_leakage_rate_ci"]
        assert lo <= rows["identity_leakage_rate"] <= hi


class TestCIParamValidation:
    @pytest.mark.parametrize("n_bootstrap", [0, -3, 1])
    def test_rejects_non_positive_or_tiny_n_bootstrap(self,
                                                      n_bootstrap):
        with pytest.raises(ValueError, match="n_bootstrap"):
            summarize_rows(ids=["a"], assoc_sim=[0.3],
                           generic_sim=[None], split_name="forget",
                           n_bootstrap=n_bootstrap)

    @pytest.mark.parametrize("ci_level", [0.0, 1.0, 1.5, -0.2])
    def test_rejects_invalid_ci_level(self, ci_level):
        with pytest.raises(ValueError, match="ci_level"):
            summarize_rows(ids=["a"], assoc_sim=[0.3],
                           generic_sim=[None], split_name="forget",
                           n_bootstrap=100, ci_level=ci_level)

    def test_validator_is_shared_with_probe_aggregation(self):
        # aggregate_scores validates the same way when CIs are on
        probes = [{"identity_id": "a", "attribute": "city",
                   "sims": {"fine": 0.4, "target": 0.1,
                            "sibling": 0.05}}]
        with pytest.raises(ValueError, match="n_bootstrap"):
            aggregate_scores(probes, bootstrap_ci=True, n_bootstrap=0)
        with pytest.raises(ValueError, match="ci_level"):
            validate_ci_params(100, 0.0)
        # Without CIs the parameters are inert — no validation error
        aggregate_scores(probes, bootstrap_ci=False, n_bootstrap=0)


class TestStaleShardForcesRescoring:
    """A committed shard is REUSED only if its full provenance still
    matches; a changed checkpoint SHA-256 or benchmark revision must
    force rescoring (the reuse branch of
    evaluate_salmu_official_splits.main appends to ``todo`` exactly
    when this predicate fails)."""

    EXPECTED = {
        "aggregation_schema": "10r4a.v1",
        "benchmark_repo_id": "cvc-mmu/salmubench-512-redistributed",
        "benchmark_revision": "b76aa3885a696969e51de4786c9f92e8064b4679",
        "state": "MF",
        "checkpoint_sha256": "ce6b336849e5" * 5 + "ce6b",
    }

    def _reuse_decision(self, shard_prov: dict) -> bool:
        return official_splits.shard_matches_provenance(
            {"_provenance": shard_prov}, self.EXPECTED)

    def test_matching_shard_reused(self):
        assert self._reuse_decision(dict(self.EXPECTED))

    def test_changed_checkpoint_sha_forces_rescoring(self):
        stale = dict(self.EXPECTED,
                     checkpoint_sha256="0" * 64)  # retrained ckpt
        assert not self._reuse_decision(stale)

    def test_changed_benchmark_revision_forces_rescoring(self):
        stale = dict(self.EXPECTED,
                     benchmark_revision="f" * 40)  # dataset bump
        assert not self._reuse_decision(stale)
