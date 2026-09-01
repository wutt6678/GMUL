"""Unit tests for released-split summarization (official_metrics).

Regression tests for the two 10R4a failures:
1. Leakage estimand/bootstrap correspondence — the identity-level
   leakage CI must resample the SAME per-identity flags that define
   the point estimate (not per-identity pair-flag means).
2. retain_synth pair-level clustering is FORCED for every row, by
   split name, regardless of any identity_id values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from granunlearn.salmu.official_metrics import (
    PAIR_CLUSTERED_SPLITS,
    cluster_ids,
    identity_leak_flags,
    summarize_rows,
    summarize_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_salmu_official_splits as official_splits  # noqa: E402


class TestClusterIds:
    def test_real_split_keeps_identity_clusters(self):
        ids, pair_clustered = cluster_ids("forget", ["a", "b", "a"])
        assert ids == ["a", "b", "a"]
        assert pair_clustered is False

    def test_missing_id_forces_pair_level(self):
        ids, pair_clustered = cluster_ids("forget", ["a", None, "a"])
        assert pair_clustered is True
        assert len(set(ids)) == 3  # every row its own unit

    def test_retain_synth_forced_pair_level_even_with_ids(self):
        """10R4a: retain_synth defines no identity units — clustering
        is pair-level for ALL rows even if some carried ids."""
        assert "retain_synth" in PAIR_CLUSTERED_SPLITS
        ids, pair_clustered = cluster_ids(
            "retain_synth", ["a", "a", "b", None])
        assert pair_clustered is True
        assert len(set(ids)) == 4


class TestIdentityLeakFlags:
    def _setup(self):
        # id "a": row0 leaks, row1 does NOT -> pair-flag mean 0.5,
        # but MACRO assoc (0.475) > MACRO generic (0.15) -> flag 1.
        # id "b": both rows no-leak -> flag 0.
        assoc = np.array([0.9, 0.05, 0.1, 0.1])
        generic = np.array([0.1, 0.2, 0.5, 0.5])
        has_generic = np.array([True] * 4)
        uniq = ["a", "b"]
        by_id = {"a": [0, 1], "b": [2, 3]}
        return assoc, generic, has_generic, uniq, by_id

    def test_flags_are_macro_vs_macro(self):
        args = self._setup()
        flag_ids, flags = identity_leak_flags(*args)
        assert flag_ids == ["a", "b"]
        assert flags.tolist() == [1.0, 0.0]

    def test_unit_excluded_without_generic_rows(self):
        assoc, generic, has_generic, uniq, by_id = self._setup()
        has_generic = np.array([False, False, True, True])
        flag_ids, flags = identity_leak_flags(
            assoc, generic, has_generic, uniq, by_id)
        assert flag_ids == ["b"]
        assert flags.tolist() == [0.0]


class TestSummarizeRows:
    def test_leakage_ci_correspondence(self):
        """10R4a: identity_leakage_rate_ci resamples the per-identity
        flags {1, 0} — NOT per-identity pair-flag means {0.5, 0}."""
        # id a: pair flags [1, 0] (mean 0.5), identity flag 1
        # id b: pair flags [0, 0], identity flag 0
        rows = summarize_rows(
            ids=["a", "a", "b", "b"],
            assoc_sim=[0.9, 0.05, 0.1, 0.1],
            generic_sim=[0.1, 0.2, 0.5, 0.5],
            split_name="forget",
            n_bootstrap=4000,
        )
        assert rows["identity_leakage_rate"] == 0.5
        lo, hi = rows["identity_leakage_rate_ci"]
        # Flag support is {0, 1}: the 95% CI over resampled flag
        # means must reach both extremes (old pair-mean bootstrap
        # could only reach [0, 0.5]-supported values around 0.25).
        assert lo == 0.0 and hi == 1.0
        # Pair-level rate + its OWN pair-level CI
        assert rows["leakage_rate"] == 0.25
        plo, phi = rows["leakage_rate_ci"]
        assert plo <= 0.25 <= phi

    def test_retain_synth_forced_pair_clustering(self):
        rows = summarize_rows(
            ids=["x", "x", None, None],  # ids present but FORCED
            assoc_sim=[0.3, 0.5, 0.2, 0.4],
            generic_sim=[None] * 4,
            split_name="retain_synth",
            n_bootstrap=500,
        )
        assert rows["pair_clustered"] is True
        assert rows["num_units"] is None
        # unit-macro degenerates exactly to the pair mean
        assert rows["unit_macro_assoc_sim"] == rows["mean_assoc_sim"]
        assert "clustering_note" in rows
        lo, hi = rows["unit_macro_assoc_sim_ci"]
        assert lo <= rows["mean_assoc_sim"] <= hi

    def test_no_generic_means_no_leakage_fields(self):
        rows = summarize_rows(
            ids=["a", "b"], assoc_sim=[0.3, 0.4],
            generic_sim=[None, None], split_name="forget",
            n_bootstrap=200)
        assert "leakage_rate" not in rows
        assert "identity_leakage_rate" not in rows


class TestSummarizeState:
    def test_gmul_target_subsets(self):
        per_split = {
            "forget": {
                "identity_id": ["t1", "t1", "r1"],
                "assoc_sim": [0.4, 0.6, 0.2],
                "generic_sim": [None, None, None],
                "gmul_target_mask": [True, True, False],
                "gmul_target_attr_mask": [True, False, False],
            },
        }
        summary = summarize_state(per_split, n_bootstrap=200)
        entry = summary["forget"]
        sub = entry["gmul_target_subset"]
        assert sub["num_pairs"] == 2
        assert sub["mean_assoc_sim"] == 0.5
        attr_sub = entry["gmul_target_attr_subset"]
        assert attr_sub["num_pairs"] == 1
        assert attr_sub["mean_assoc_sim"] == 0.4

    def test_empty_mask_omits_subset(self):
        per_split = {
            "retain_synth": {
                "identity_id": [None, None],
                "assoc_sim": [0.2, 0.3],
                "generic_sim": [None, None],
                "gmul_target_mask": [False, False],
                "gmul_target_attr_mask": [False, False],
            },
        }
        summary = summarize_state(per_split, n_bootstrap=100)
        assert "gmul_target_subset" not in summary["retain_synth"]
        assert summary["retain_synth"]["pair_clustered"] is True


class TestShardProvenanceValidation:
    """10R4b: shard reuse requires an exact match on ALL provenance
    fields — not just file existence."""

    EXPECTED = {
        "aggregation_schema": "test.v1",
        "benchmark_repo_id": "org/bench",
        "benchmark_revision": "abc123",
        "state": "MF",
        "checkpoint_sha256": "deadbeef",
    }

    def _shard(self, **overrides):
        prov = dict(self.EXPECTED)
        prov.update(overrides)
        return {"_provenance": prov}

    def test_exact_match_reused(self):
        assert official_splits.shard_matches_provenance(
            self._shard(), self.EXPECTED)

    def test_every_field_invalidates(self):
        for field in self.EXPECTED:
            shard = self._shard(**{field: "OTHER"})
            assert not official_splits.shard_matches_provenance(
                shard, self.EXPECTED), field

    def test_missing_provenance_invalidates(self):
        assert not official_splits.shard_matches_provenance(
            {}, self.EXPECTED)
