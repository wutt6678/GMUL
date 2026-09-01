"""Report-integrity regression tests (Iteration 10R5a).

Covers the three 10R5a repairs:
1. suffix-aware routing of the official-split report into the
   reference evaluation, with evidence_status INHERITED from the
   routed report (an iteration must never embed another chain's
   values or labels);
2. holdout cleanliness is VALIDATED (manifest allowed_split ==
   forget + zero exact overlap with BOTH holdout splits) — never
   inferred from the presence of an iteration suffix;
3. selection test-protocol text is derived from the actual persona
   split (no stale hardcoded counts or stale "10R5 still required"
   claims).

Assertions target the committed reports plus pure functions; no GPU,
no released-data I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from granunlearn.salmu.paths import SalmuPaths
from granunlearn.salmu.salmubench_metrics import (
    compute_official_salmubench,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_salmu_state_pairs as builder  # noqa: E402
import evaluate_salmu_official_splits as official_splits  # noqa: E402

REPORTS = REPO_ROOT / "data" / "reports"


def _report(name: str, suffix: str = "") -> dict:
    path = SalmuPaths(REPO_ROOT, suffix=suffix).report(name)
    assert path.exists(), f"missing committed report: {path}"
    return json.loads(path.read_text())


class TestEvidenceStatusInheritance:
    """compute_official_salmubench must inherit evidence_status from
    the routed report instead of hardcoding one chain's label."""

    def _fake_report(self, tmp_path: Path, status: str) -> Path:
        rep = tmp_path / "salmu_official_splits.json"
        rep.write_text(json.dumps({
            "evidence_status": status,
            "states": {"MF": {
                "forget": {"mean_assoc_sim": 0.27},
                "holdout_association": {"mean_assoc_sim": 0.26},
                "holdout_identity": {"mean_assoc_sim": 0.18},
            }},
        }))
        return rep

    def test_inherits_untouched_status(self, tmp_path):
        rep = self._fake_report(tmp_path, "UNTOUCHED EXTERNAL X")
        out = compute_official_salmubench(
            tmp_path, official_splits_report=rep)
        assert out["evidence_status"] == "UNTOUCHED EXTERNAL X"
        assert out["evidence_status_source"].startswith("inherited")
        assert out["metrics"]["AssocStr"]["MF"] == 0.27

    def test_inherits_transfer_diagnostic_status(self, tmp_path):
        rep = self._fake_report(tmp_path, "TRANSFER DIAGNOSTIC Y")
        out = compute_official_salmubench(
            tmp_path, official_splits_report=rep)
        assert out["evidence_status"] == "TRANSFER DIAGNOSTIC Y"

    def test_missing_report_gets_conservative_default(self, tmp_path):
        out = compute_official_salmubench(
            tmp_path, official_splits_report=tmp_path / "absent.json")
        assert out["evidence_status"].startswith("UNKNOWN")
        assert out["evidence_status_source"] == "conservative default"


class TestSuffixAwareOfficialReportRouting:
    """The r5 reference report must embed the r5 official-split
    values and status — not the R4b chain's."""

    def test_r5_ref_eval_routes_r5_official_report(self):
        ref = _report("salmu_reference_eval", "r5")
        official = ref["official_salmubench"]
        assert official["metrics"]["source"].endswith(
            "salmu_official_splits_r5.json")
        assert official["evidence_status"].startswith("UNTOUCHED")

    def test_r5_ref_eval_values_match_r5_official_report(self):
        ref = _report("salmu_reference_eval", "r5")
        official_r5 = _report("salmu_official_splits", "r5")
        embedded = ref["official_salmubench"]["metrics"]
        for metric, split in (("AssocStr", "forget"),
                              ("IntraIdSim", "holdout_association"),
                              ("InterIdSim", "holdout_identity")):
            for state, val in embedded[metric].items():
                expect = official_r5["states"][state][split][
                    "mean_assoc_sim"]
                assert val == expect, (metric, state)

    def test_original_ref_eval_embeds_transfer_diagnostic(self):
        """The original (unsuffixed) chain stays a transfer
        diagnostic — routing is per-iteration in BOTH directions."""
        ref = _report("salmu_reference_eval")
        official = ref["official_salmubench"]
        assert official["metrics"]["source"].endswith(
            "salmu_official_splits.json")
        assert official["evidence_status"].startswith(
            "TRANSFER DIAGNOSTIC")


class TestHoldoutCleanValidation:
    """Cleanliness must be validated from manifest + overlap counts,
    never from the suffix."""

    MANIFEST = {"protocol": {"allowed_split": "forget"}}
    CONSUMPTION = {
        "holdout_association": {"exact_released_pairs": 0},
        "holdout_identity": {"exact_released_pairs": 0},
    }

    def test_all_conditions_validate(self):
        out = official_splits.holdout_clean_validation(
            self.MANIFEST, self.CONSUMPTION)
        assert out["validated"] is True
        assert all(out["checks"].values())

    @pytest.mark.parametrize("split", [
        "holdout_association", "holdout_identity", "retain",
        "forget,retain",
    ])
    def test_non_forget_allowed_split_fails(self, split):
        out = official_splits.holdout_clean_validation(
            {"protocol": {"allowed_split": split}}, self.CONSUMPTION)
        assert out["validated"] is False
        assert out["checks"]["allowed_split_is_forget"] is False

    def test_wrong_allowed_split_fails(self):
        out = official_splits.holdout_clean_validation(
            {"protocol": {"allowed_split": "retain"}},
            self.CONSUMPTION)
        assert out["validated"] is False
        assert out["checks"]["allowed_split_is_forget"] is False

    def test_missing_protocol_fails(self):
        out = official_splits.holdout_clean_validation(
            {}, self.CONSUMPTION)
        assert out["validated"] is False

    def test_any_holdout_association_overlap_fails(self):
        out = official_splits.holdout_clean_validation(
            self.MANIFEST,
            {**self.CONSUMPTION,
             "holdout_association": {"exact_released_pairs": 7}})
        assert out["validated"] is False
        assert out["checks"][
            "zero_exact_holdout_association_pairs"] is False

    def test_any_holdout_identity_overlap_fails(self):
        out = official_splits.holdout_clean_validation(
            self.MANIFEST,
            {**self.CONSUMPTION,
             "holdout_identity": {"exact_released_pairs": 3}})
        assert out["validated"] is False
        assert out["checks"][
            "zero_exact_holdout_identity_pairs"] is False

    def test_missing_consumption_fails(self):
        out = official_splits.holdout_clean_validation(
            self.MANIFEST, {})
        assert out["validated"] is False

    def test_committed_reports_match_validated_status(self):
        r5 = _report("salmu_official_splits", "r5")
        assert r5["holdout_clean_validation"]["validated"] is True
        assert r5["evidence_status"].startswith("UNTOUCHED")
        orig = _report("salmu_official_splits")
        assert orig["holdout_clean_validation"]["validated"] is False
        assert orig["evidence_status"].startswith("TRANSFER")


class TestBuildCleanFlag:
    """build_salmu_state_pairs only marks forget-only builds clean."""

    def test_holdout_clean_only_for_forget(self):
        assert builder.is_holdout_clean_build("forget") is True

    @pytest.mark.parametrize("split", [
        None, "retain", "holdout_association", "holdout_identity",
        "forget,retain", "FORGET",
    ])
    def test_other_splits_not_clean(self, split):
        assert builder.is_holdout_clean_build(split) is False


class TestSelectionProtocolText:
    """Protocol text derived from the actual split — no stale
    counts, no stale cross-iteration claims."""

    def test_r5_selection_text_is_current(self):
        rep = _report("salmu_unlearning_selection", "r5")
        text = rep["test_protocol"]
        assert "32/7/7 of 46" in text
        assert "10/60" not in json.dumps(rep)
        # r5 IS the holdout-clean retrain — it must not claim one is
        # still required
        assert "requires the Iteration 10R5" not in json.dumps(rep)
        split = rep["persona_split"]
        assert (len(split["train"]), len(split["val"]),
                len(split["test"])) == (32, 7, 7)

    def test_original_selection_text_keeps_exploratory_caveat(self):
        rep = _report("salmu_unlearning_selection")
        text = rep["test_protocol"]
        assert "40/10/10 of 60" in text
        assert "exploratory" in text


class TestOfficialEvaluatorIntegration:
    """The remaining official metrics (RetFail/ACS/IdZSC/CoreAssoc/
    VisIdInt/FragSim) come from the official SALMUBench evaluator via
    scripts/run_official_salmubench_eval.py."""

    def test_official_eval_report_fills_remaining_metrics(
            self, tmp_path):
        splits = tmp_path / "splits.json"
        splits.write_text(json.dumps({
            "evidence_status": "UNTOUCHED X",
            "states": {"MF": {
                "forget": {"mean_assoc_sim": 0.27},
                "holdout_association": {"mean_assoc_sim": 0.26},
                "holdout_identity": {"mean_assoc_sim": 0.18}}},
        }))
        ev = tmp_path / "salmubench_official_eval.json"
        ev.write_text(json.dumps({"states": {"MF": {
            "RetFail_MRR": 0.3, "RetFail_R@1": 0.1, "ACS": 0.8,
            "IdZSC": 0.2, "CoreAssoc": 0.35, "GenKnow": None,
            "VisIdInt": 0.33, "FragSim": 0.31}}}))
        out = compute_official_salmubench(
            tmp_path, official_splits_report=splits,
            official_eval_report=ev)
        m = out["metrics"]
        assert m["RetFail"]["MF"] == 0.3
        assert m["RetFail_R@1"]["MF"] == 0.1
        assert m["ACS"]["MF"] == 0.8
        assert m["IdZSC"]["MF"] == 0.2
        assert m["CoreAssoc"]["MF"] == 0.35
        assert m["VisIdInt"]["MF"] == 0.33
        assert m["FragSim"]["MF"] == 0.31
        # all-null metric stays unfilled
        assert m["GenKnow"] is None
        assert m["official_evaluator_source"].endswith(
            "salmubench_official_eval.json")
        # splits-report metrics untouched by the official fill
        assert m["AssocStr"]["MF"] == 0.27

    def test_committed_report_covers_all_r5_states(self):
        rep = _report("salmubench_official_eval", "r5")
        expected = {"COMPROMISED", "BASE", "MF", "MG", "MN", "B0",
                    "B1_lr2e-06_c", "B2_lr2e-05",
                    "B3_lr2e-06_lam1_c"}
        assert set(rep["states"]) == expected
        for state, m in rep["states"].items():
            for key in ("RetFail_MRR", "RetFail_R@1", "AssocStr",
                        "ACS", "IdZSC", "CoreAssoc", "InterIdSim",
                        "IntraIdSim", "VisIdInt", "FragSim"):
                assert m.get(key) is not None, (state, key)

    def test_base_reproduces_authors_clean_reference(self):
        """BASE = released Clean CLIP: its official metrics must
        reproduce the authors' committed reference values
        (evaluation_results/reference_model_clip-vit-b-16-salmu-
        clean.json in the official repo)."""
        rep = _report("salmubench_official_eval", "r5")
        base = rep["states"]["BASE"]
        ref = {"AssocStr": 0.14169, "ACS": 0.60181, "IdZSC": 0.01082,
               "CoreAssoc": 0.15142, "InterIdSim": 0.14279,
               "IntraIdSim": 0.14280, "VisIdInt": 0.33083,
               "FragSim": 0.30908, "RetFail_MRR": 0.000815}
        for key, val in ref.items():
            assert abs(base[key] - val) <= 0.002, (key, base[key])
        assert base["RetFail_R@1"] == 0.0
        assert base["GenKnow"] is None  # no local ImageNet

    def test_crosscheck_against_our_evaluator(self):
        """The official evaluator's AssocStr/IntraIdSim/InterIdSim
        must agree with our released-split evaluator up to encoding
        numerics (autocast/batching differences)."""
        rep = _report("salmubench_official_eval", "r5")
        cc = rep["crosscheck_vs_our_evaluator"]
        assert cc, "cross-check block missing"
        for state, diffs in cc.items():
            for metric, d in diffs.items():
                assert d["abs_diff"] <= 0.005, (state, metric, d)
