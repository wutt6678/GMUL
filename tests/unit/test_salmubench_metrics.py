"""Unit tests for GMUL proxy metrics (in-house CLIP-embedding proxies)."""

from __future__ import annotations

from granunlearn.salmu.salmubench_metrics import compute_gmul_proxy_metrics


def _make_results(fine, target, sibling, generic=0.20,
                  is_target_attr=True):
    return {
        "identity_id": "p1",
        "attribute": "city",
        "is_target_attr": is_target_attr,
        "sims": {"fine": fine, "target": target,
                 "sibling": sibling, "generic": generic,
                 "ancestor": target},
    }


class TestGMULProxyMetrics:
    def test_basic_computation(self):
        target = [
            _make_results(0.4, 0.1, 0.05),  # prefers fine
            _make_results(0.1, 0.4, 0.05),  # prefers target
        ]
        same_entity = [
            _make_results(0.3, 0.2, 0.1, is_target_attr=False),
            _make_results(0.35, 0.15, 0.1, is_target_attr=False),
        ]
        other_entity = [
            _make_results(0.25, 0.2, 0.1, is_target_attr=False),
        ]
        m = compute_gmul_proxy_metrics(target, same_entity, other_entity)
        # forget = 1 - 0.5 = 0.5
        assert m["gmul_proxy_forget"] == 0.5
        # holdout_association = 1.0 (both same-entity prefer fine)
        assert m["gmul_proxy_holdout_association"] == 1.0
        # retain_synth = 0.25
        assert m["gmul_proxy_retain_synth"] == 0.25
        # core_assoc = mean(0.1, 0.4) = 0.25
        assert m["gmul_proxy_core_assoc"] == 0.25
        # intra_identity = mean(0.3, 0.35) = 0.325
        assert m["gmul_proxy_intra_identity"] == 0.325
        # inter_identity = retain_synth = 0.25
        assert m["gmul_proxy_inter_identity"] == 0.25
        # preference_margin = mean(0.4-0.1, 0.1-0.4) = mean(0.3, -0.3) = 0
        assert m["gmul_proxy_preference_margin"] == 0.0

    def test_empty_target(self):
        m = compute_gmul_proxy_metrics([], [], [])
        assert m["gmul_proxy_forget"] is None
        assert m["gmul_proxy_holdout_association"] is None
        assert m["gmul_proxy_retain_synth"] is None
        assert m["n_target_probes"] == 0

    def test_all_forget(self):
        """All target probes prefer target → forget = 1.0 (fine not
        preferred at all)."""
        target = [_make_results(0.1, 0.4, 0.05)]
        m = compute_gmul_proxy_metrics(target, [], [])
        assert m["gmul_proxy_forget"] == 1.0

    def test_no_forget(self):
        """All target probes prefer fine → forget = 0.0 (fine still
        fully preferred)."""
        target = [_make_results(0.4, 0.1, 0.05)]
        m = compute_gmul_proxy_metrics(target, [], [])
        assert m["gmul_proxy_forget"] == 0.0

    def test_holdout_identity(self):
        target = [_make_results(0.3, 0.2, 0.1, generic=0.25)]
        m = compute_gmul_proxy_metrics(target, [], [])
        # holdout_identity = 1 - 0.25 = 0.75
        assert m["gmul_proxy_holdout_identity"] == 0.75
