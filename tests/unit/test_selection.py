"""Unit tests for Iteration 9 distance-to-MG selection.

Selection must use train+val ONLY; the distance is a model-selection
criterion, individual metrics stay primary.
"""

from __future__ import annotations

from granunlearn.evaluation.selection import (
    SUMMARY_COMPONENTS,
    distance_to_reference,
    filter_predictions_by_splits,
    select_checkpoints,
    summary_vector,
)


def hm(filr=None, tga=None, ancestor=None, ret_same=None, ret_other=None,
       over=None, wrong=None) -> dict:
    """Minimal hierarchy-metrics dict shaped like the real one."""
    return {
        "filr": filr,
        "tga": tga,
        "ancestor_retention": {"post_unlearning_accuracy": ancestor},
        "retain_same_entity": {"baseline_accuracy": ret_same},
        "retain_other_entity": {"baseline_accuracy": ret_other},
        "failure_rates": {"over_forgetting": over, "wrong_branch": wrong},
    }


class TestSummaryVector:
    def test_components_extracted(self):
        v = summary_vector(hm(0.1, 0.6, 0.5, 0.7, 0.8, 0.0, 0.15))
        assert v == {"filr": 0.1, "tga": 0.6, "ancestor": 0.5,
                     "retain_same": 0.7, "retain_other": 0.8,
                     "over": 0.0, "wrong": 0.15}
        assert tuple(v) == SUMMARY_COMPONENTS

    def test_missing_slices_become_none(self):
        v = summary_vector(hm(filr=0.2))
        assert v["filr"] == 0.2
        assert v["tga"] is None and v["ancestor"] is None


class TestDistance:
    def test_manual_weighted_l1(self):
        ref = summary_vector(hm(0.0, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        vec = summary_vector(hm(0.1, 0.6, 0.4, 0.6, 0.6, 0.0, 0.1))
        # uniform weights: mean of |diffs| = (0.1+0.1+0.1)/7
        dist, used = distance_to_reference(vec, ref)
        assert abs(dist - 0.3 / 7) < 1e-6
        assert len(used) == 7

    def test_weights_respected(self):
        ref = summary_vector(hm(0.0, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        vec = summary_vector(hm(0.1, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        w = {c: 1.0 for c in SUMMARY_COMPONENTS}
        w["filr"] = 3.0
        dist, _ = distance_to_reference(vec, ref, weights=w)
        assert abs(dist - 0.3 / 9) < 1e-6  # 3*0.1 / (6*1 + 3)

    def test_missing_components_renormalized(self):
        ref = summary_vector(hm(0.0, 0.7, None, 0.6, 0.6, 0.0, 0.1))
        vec = summary_vector(hm(0.1, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        dist, used = distance_to_reference(vec, ref)
        assert "ancestor" not in used
        assert abs(dist - 0.1 / 6) < 1e-6

    def test_identical_vectors_zero(self):
        v = summary_vector(hm(0.1, 0.6, 0.5, 0.7, 0.8, 0.0, 0.15))
        dist, used = distance_to_reference(v, v)
        assert dist == 0.0 and len(used) == 7


class TestSelection:
    def test_best_per_method_selected_on_min_distance(self):
        ref = summary_vector(hm(0.0, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        candidates = {
            "B1_lr2e-05": {"method": "B1", "trainval_metrics":
                           hm(0.3, 0.2, 0.2, 0.6, 0.6, 0.0, 0.2)},
            "B1_lr1e-04": {"method": "B1", "trainval_metrics":
                           hm(0.05, 0.6, 0.5, 0.6, 0.6, 0.0, 0.1)},
            "B2": {"method": "B2", "trainval_metrics":
                   hm(0.2, 0.5, 0.4, 0.5, 0.5, 0.0, 0.2)},
        }
        report = select_checkpoints(candidates, ref)
        assert report["selected"]["B1"] == "B1_lr1e-04"
        assert report["selected"]["B2"] == "B2"
        assert report["basis"].startswith("train+val probes only")

    def test_distance_never_uses_test(self):
        """The selection machinery only consumes trainval_metrics —
        structural guard: no 'test' key is read anywhere."""
        ref = summary_vector(hm(0.0, 0.7, 0.5, 0.6, 0.6, 0.0, 0.1))
        candidates = {"X": {"method": "B1",
                            "trainval_metrics": hm(0.1, 0.6, 0.5, 0.6,
                                                   0.6, 0.0, 0.1)}}
        report = select_checkpoints(candidates, ref)
        assert report["candidates"]["X"]["distance_to_mg"] is not None
        assert report["basis"] == \
            "train+val probes only (test split held out)"


class TestSplitFilter:
    def test_filter_keeps_requested_splits_only(self):
        class Q:
            def __init__(self, qid, split):
                self.query_id, self.split = qid, split

        class P:
            def __init__(self, qid):
                self.query_id = qid

        queries = [Q("a", "train"), Q("b", "val"), Q("c", "test")]
        preds = [P("a"), P("b"), P("c")]
        tv = filter_predictions_by_splits(preds, queries, ("train", "val"))
        assert {p.query_id for p in tv} == {"a", "b"}
