"""Unit tests for the 10R5b evaluation-integrity repair.

Covers:
1. The exploratory internal-test wording restored in the r5 selection
   report (the r5 test personas are a subset of the 10R2/10R3
   candidate-wide-inspected test split).
2. Official-evaluator per-state RNG determinism: identical
   checkpoints (MF == B0) produce identical metrics — verified on the
   committed regenerated official-evaluator report.
3. Checkpoint-safe reuse of raw official results: a result is reused
   ONLY when its sidecar binds the current checkpoint SHA-256 AND the
   pinned evaluator commit AND the result file's own SHA-256; a
   changed checkpoint (or tampered result) forces rescoring, and
   stale evidence is quarantined, never silently overwritten.
4. Official repository HEAD + cleanliness verification.
5. Target-only official metrics and paired identity-clustered
   difference CIs (pure statistics + committed-report schema).

No GPU and no released SALMUBench data required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"

import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_official_salmubench_eval as runner  # noqa: E402
from granunlearn.salmu import official_eval_analysis as analysis  # noqa: E402

METRIC_KEYS = [m[0] for m in runner.METRIC_MAP]


def _load(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def _write_sidecar(result_path: Path, sha: str, commit: str,
                   result_sha: str | None = None) -> Path:
    side = runner.sidecar_path(result_path)
    side.write_text(json.dumps({
        "checkpoint_sha256": sha,
        "official_repo_commit": commit,
        "result_sha256": result_sha if result_sha is not None
        else runner.sha256_file(result_path),
    }))
    return side


# ── 1. exploratory wording restored ───────────────────────────────

class TestExploratoryWordingRestored:
    """The r5 internal test must NOT be called genuinely untouched:
    its 7 test personas are a subset of the 10R2/10R3 candidate-wide
    inspected 10-persona test split."""

    def test_r5_test_protocol_is_exploratory(self):
        rep = _load("salmu_unlearning_selection_r5")
        text = rep.get("test_protocol", "")
        assert "EXPLORATORY" in text
        assert "genuine held-out verdict" not in text
        assert "SUBSET of the 10R2/10R3" in text

    def test_r5_notes_carry_the_correction(self):
        notes = " ".join(_load(
            "salmu_unlearning_selection_r5").get("notes", []))
        assert "10R5b test-protocol honesty" in notes
        assert "stays EXPLORATORY" in notes
        assert "never scored before selection, so the internal test " \
               "verdict is genuine" not in notes

    def test_primary_external_evaluation_named(self):
        text = _load(
            "salmu_unlearning_selection_r5").get("test_protocol", "")
        assert "salmu_official_splits_r5.json" in text
        assert "salmubench_official_eval_r5.json" in text

    def test_r5_test_personas_subset_of_r4_test(self):
        from granunlearn.salmu.unlearning import \
            split_target_personas
        hier = REPO_ROOT / "data" / "salmu_hierarchical"
        orig = json.loads(
            (hier / "training" /
             "state_pairs_manifest.json").read_text())
        r5 = json.loads(
            (hier / "training_r5" /
             "state_pairs_manifest.json").read_text())
        old_test = set(split_target_personas(
            orig["partition"]["target_identity_ids"])["test"])
        new_test = set(split_target_personas(
            r5["partition"]["target_identity_ids"])["test"])
        assert new_test <= old_test
        assert len(new_test) == 7 and len(old_test) == 10


# ── 2. identical checkpoints => identical metrics (MF == B0) ─────

class TestIdenticalCheckpointInvariant:
    """The official evaluator's ACS negatives used to be drawn from
    an ADVANCING numpy RNG state, making ACS incomparable across
    states (MF 0.954542 vs B0 0.955527 despite identical checkpoint
    hashes).  With the per-state RNG reset they must match exactly."""

    def test_report_records_the_invariant_as_passed(self):
        rep = _load("salmubench_official_eval_r5")
        inv = rep.get("identical_checkpoint_invariants", {})
        assert inv, "no identical-checkpoint group found"
        for group, res in inv.items():
            assert res["all_metrics_identical"] is True, group

    def test_mf_equals_b0_on_every_official_metric(self):
        states = _load("salmubench_official_eval_r5")["states"]
        assert "MF" in states and "B0" in states
        assert states["MF"]["checkpoint_sha256"] == \
            states["B0"]["checkpoint_sha256"]
        for key in METRIC_KEYS:
            assert states["MF"].get(key) == states["B0"].get(key), \
                f"MF != B0 on {key}"

    def test_rng_protocol_recorded(self):
        rep = _load("salmubench_official_eval_r5")
        assert "restored before EVERY evaluate_model" in \
            rep["official_evaluator"]["rng_protocol"]


# ── 3. checkpoint-safe reuse of raw results ──────────────────────

class TestCheckpointSafeReuse:
    """A raw result may be reused ONLY when its sidecar binds the
    current checkpoint SHA-256, the pinned evaluator commit, and the
    result file's own SHA-256."""

    def _result(self, tmp_path: Path) -> Path:
        res = tmp_path / "evaluation_x_r5.json"
        res.write_text(json.dumps({"efficacy": {}}))
        return res

    def test_missing_sidecar_refuses_reuse(self, tmp_path):
        res = self._result(tmp_path)
        assert not runner.reuse_is_valid(res, "a" * 64, "c" * 40)

    def test_matching_sidecar_allows_reuse(self, tmp_path):
        res = self._result(tmp_path)
        _write_sidecar(res, "a" * 64, "c" * 40)
        assert runner.reuse_is_valid(res, "a" * 64, "c" * 40)

    def test_changed_checkpoint_forces_rescoring(self, tmp_path):
        """The core protection: same filename, NEW checkpoint hash
        (e.g. a retrained state) must not reuse the old result."""
        res = self._result(tmp_path)
        _write_sidecar(res, "a" * 64, "c" * 40)
        assert not runner.reuse_is_valid(res, "b" * 64, "c" * 40)

    def test_changed_evaluator_commit_forces_rescoring(self,
                                                       tmp_path):
        res = self._result(tmp_path)
        _write_sidecar(res, "a" * 64, "c" * 40)
        assert not runner.reuse_is_valid(res, "a" * 64, "d" * 40)

    def test_tampered_result_forces_rescoring(self, tmp_path):
        res = self._result(tmp_path)
        _write_sidecar(res, "a" * 64, "c" * 40)
        res.write_text(json.dumps({"efficacy": {"tampered": 1}}))
        assert not runner.reuse_is_valid(res, "a" * 64, "c" * 40)

    def test_corrupt_sidecar_refuses_reuse(self, tmp_path):
        res = self._result(tmp_path)
        runner.sidecar_path(res).write_text("{not json")
        assert not runner.reuse_is_valid(res, "a" * 64, "c" * 40)

    def test_quarantine_moves_result_and_sidecar(self, tmp_path):
        res = self._result(tmp_path)
        _write_sidecar(res, "a" * 64, "c" * 40)
        runner.quarantine_stale(res)
        assert not res.exists()
        assert not runner.sidecar_path(res).exists()
        stale = list((tmp_path / "stale").iterdir())
        assert len(stale) == 2  # evidence preserved, never deleted

    def test_official_output_path_matches_official_naming(
            self, tmp_path):
        """The reuse check must look at the SAME filename the
        official evaluate_model writes (slug = Path.stem, then the
        '__'->'_' / '_._'->'_' collapse)."""
        p = runner.official_output_path(
            tmp_path, Path("/models/BASE__r5.pth"))
        assert p.name.startswith("evaluation_")
        assert "__" not in p.name and "._" not in p.name
        assert p.suffix == ".json"


# ── 4. official repo verification ────────────────────────────────

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True,
        text=True, check=True).stdout.strip()


@pytest.fixture
def official_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "salmubench"
    (repo / "evaluation").mkdir(parents=True)
    (repo / "evaluation" / "evaluation.py").write_text("# official\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=ci@test", "-c", "user.name=ci",
         "commit", "-qm", "init")
    return repo


class TestOfficialRepoVerification:
    def test_clean_repo_at_expected_commit_passes(self,
                                                  official_repo):
        head = _git(official_repo, "rev-parse", "HEAD")
        info = runner.verify_official_repo(official_repo, head)
        assert info["commit"] == head and info["clean"] is True

    def test_head_mismatch_aborts(self, official_repo):
        with pytest.raises(SystemExit, match="pinned commit"):
            runner.verify_official_repo(official_repo, "f" * 40)

    def test_tracked_modification_aborts(self, official_repo):
        head = _git(official_repo, "rev-parse", "HEAD")
        (official_repo / "evaluation" / "evaluation.py").write_text(
            "# MODIFIED official logic\n")
        with pytest.raises(SystemExit, match="DIRTY"):
            runner.verify_official_repo(official_repo, head)

    def test_untracked_source_file_aborts(self, official_repo):
        head = _git(official_repo, "rev-parse", "HEAD")
        (official_repo / "evaluation" / "extra.py").write_text("x=1")
        with pytest.raises(SystemExit, match="DIRTY"):
            runner.verify_official_repo(official_repo, head)

    def test_pycache_is_tolerated(self, official_repo):
        """Importing the official module inevitably creates
        __pycache__; that cannot alter the evaluation source."""
        head = _git(official_repo, "rev-parse", "HEAD")
        cache = official_repo / "evaluation" / "__pycache__"
        cache.mkdir()
        (cache / "evaluation.cpython-310.pyc").write_bytes(b"\x00")
        info = runner.verify_official_repo(official_repo, head)
        assert info["clean"] is True

    def test_missing_checkout_aborts(self, tmp_path):
        with pytest.raises(SystemExit, match="Not an official"):
            runner.verify_official_repo(tmp_path / "nope", "f" * 40)

    def test_committed_report_records_verified_head(self):
        rep = _load("salmubench_official_eval_r5")
        ev = rep["official_evaluator"]
        assert ev["verified_head"] == runner.OFFICIAL_REPO_COMMIT
        assert ev["worktree_clean"] is True


# ── 5. target-only metrics + paired CIs (pure) ───────────────────

class TestTargetOnlyStatistics:
    def test_unit_means_respects_mask(self):
        vals = [1.0, 3.0, 5.0, 7.0]
        cl = ["a", "a", "b", "b"]
        mask = [True, False, True, True]
        assert analysis.unit_means(vals, cl, mask) == \
            (["a", "b"], [1.0, 6.0])

    def test_clustered_mean_ci_brackets_estimate(self):
        vals = [0.1, 0.2, 0.8, 0.9, 0.5]
        cl = ["a", "a", "b", "b", "c"]
        s = analysis.clustered_mean_ci(vals, cl, n_bootstrap=200)
        lo, hi = s["ci"]
        assert lo <= s["mean"] <= hi
        assert s["num_units"] == 3

    def test_degenerate_units_collapse_ci(self):
        s = analysis.clustered_mean_ci([0.4] * 6, ["a"] * 3
                                       + ["b"] * 3, n_bootstrap=50)
        assert s["mean"] == 0.4 and s["ci"] == (0.4, 0.4)

    def test_ci_reproducible_with_fixed_seed(self):
        vals = [0.13, 0.4, 0.72, 0.25, 0.9, 0.31]
        cl = ["a", "a", "b", "c", "c", "d"]
        one = analysis.clustered_mean_ci(vals, cl, n_bootstrap=300,
                                         seed=7)
        two = analysis.clustered_mean_ci(vals, cl, n_bootstrap=300,
                                         seed=7)
        assert one == two

    def test_paired_self_diff_is_degenerate(self):
        vals = [0.2, 0.6, 0.4, 0.8]
        cl = ["a", "a", "b", "b"]
        d = analysis.paired_clustered_diff_ci(vals, vals, cl,
                                              n_bootstrap=100)
        assert d["diff"] == 0.0 and d["ci"] == (0.0, 0.0)

    def test_paired_diff_recovers_constant_offset(self):
        vals = [0.2, 0.6, 0.4, 0.8]
        cl = ["a", "a", "b", "b"]
        shifted = [v + 0.1 for v in vals]
        d = analysis.paired_clustered_diff_ci(
            shifted, vals, cl, n_bootstrap=200)
        assert abs(d["diff"] - 0.1) < 1e-9
        assert d["ci"][0] == pytest.approx(0.1, abs=1e-9)
        assert d["ci"][1] == pytest.approx(0.1, abs=1e-9)

    def test_paired_diff_brackets_estimate_and_is_paired(self):
        vals_a = [0.5, 0.6, 0.2, 0.1]
        vals_b = [0.2, 0.7, 0.5, 0.05]
        cl = ["a", "a", "b", "b"]
        d = analysis.paired_clustered_diff_ci(
            vals_a, vals_b, cl, n_bootstrap=300)
        assert d["ci"][0] <= d["diff"] <= d["ci"][1]

    def test_invalid_ci_params_rejected(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            analysis.clustered_mean_ci([0.5], ["a"], n_bootstrap=0)
        with pytest.raises(ValueError, match="ci_level"):
            analysis.clustered_mean_ci([0.5], ["a"], ci_level=1.0)

    def test_retrieval_stats(self):
        s = analysis.retrieval_stats([1, 2, 4, 100],
                                     [True, True, True, False])
        assert s["num_rows"] == 3
        assert s["R@1"] == round(1 / 3, 4)
        assert s["MRR"] == round((1 + 0.5 + 0.25) / 3, 4)

    def test_target_attr_mask(self):
        ids = ["t1", "t1", "t2", "x"]
        files = ["f1", "f2", "f3", "f4"]
        mask = analysis.target_attr_mask(
            ids, files, {"t1", "t2"},
            {"t1": "hair", "t2": "age"},
            {"f1": "hair", "f2": "tattoo", "f3": "age",
             "f4": "hair"})
        assert mask == [True, False, True, False]

    def test_identity_mask(self):
        assert analysis.identity_mask(["a", "b", "a"], {"a"}) == \
            [True, False, True]


# ── 5b. target-only + paired CIs on the committed report ────────

class TestCommittedTargetOnlyReport:
    def test_every_state_has_target_only_metrics(self):
        rep = _load("salmubench_official_eval_r5")
        manifest = json.loads(
            (REPO_ROOT / "data" / "salmu_hierarchical" /
             "training_r5" / "state_pairs_manifest.json").read_text())
        n_targets = len(manifest["partition"]["target_identity_ids"])
        for state, entry in rep["states"].items():
            t = entry.get("target_only", {})
            assert "AssocStr_target" in t, state
            assert "CoreAssoc_target" in t, state
            assert "RetFail_target" in t, state
            for m in ("AssocStr_target", "CoreAssoc_target"):
                lo, hi = t[m]["ci"]
                assert lo <= t[m]["mean"] <= hi, (state, m)
                assert t[m]["num_units"] == n_targets
            assert t["AssocStr_target"]["num_rows"] > 0

    def test_results_files_are_hash_bound(self):
        """The committed report must bind every raw result to its
        SHA-256 (this is the portable evidence — the raw files
        themselves are large gitignored artifacts)."""
        rep = _load("salmubench_official_eval_r5")
        for state, entry in rep["states"].items():
            sha = entry.get("results_file_sha256")
            assert sha and len(sha) == 64, state
            assert all(c in "0123456789abcdef" for c in sha), state
            assert entry.get("results_file"), state

    def test_local_raw_results_match_recorded_hash(self):
        """Where the gitignored raw official results exist locally,
        each file must hash-match the committed report (tamper
        detection).  Skipped in CI checkouts without the artifacts."""
        rep = _load("salmubench_official_eval_r5")
        results_dir = (REPO_ROOT / "data" / "salmu_hierarchical"
                       / "official_salmubench_results_r5")
        if not results_dir.exists():
            pytest.skip("raw official results are gitignored "
                        "artifacts, absent in this checkout")
        for state, entry in rep["states"].items():
            raw = results_dir / entry["results_file"]
            assert raw.exists(), state
            assert runner.sha256_file(raw) == \
                entry["results_file_sha256"], state

    def test_paired_cis_present_for_unlearning_states(self):
        rep = _load("salmubench_official_eval_r5")
        paired = rep["paired_target_only"]
        for state in ("B1_lr2e-06_c", "B2_lr2e-05",
                      "B3_lr2e-06_lam1_c", "B0"):
            assert state in paired, state
            for ref in ("vs_MF", "vs_MG"):
                block = paired[state][ref]
                assert "AssocStr_target" in block
                d = block["AssocStr_target"]
                assert d["ci"][0] <= d["diff"] <= d["ci"][1]

    def test_b0_vs_mf_paired_diff_is_degenerate(self):
        """B0 IS MF (identical checkpoint): the paired target-only
        difference must be exactly 0 with a collapsed CI."""
        rep = _load("salmubench_official_eval_r5")
        block = rep["paired_target_only"]["B0"]["vs_MF"]
        for metric in ("AssocStr_target", "CoreAssoc_target"):
            assert block[metric]["diff"] == 0.0
            assert tuple(block[metric]["ci"]) == (0.0, 0.0)

    def test_target_only_assocstr_matches_full_when_targets_only(
            self):
        """Sanity: target-only AssocStr differs from full-forget
        AssocStr (the subset really is a restriction)."""
        rep = _load("salmubench_official_eval_r5")
        mf = rep["states"]["MF"]
        assert mf["target_only"]["AssocStr_target"]["mean"] != \
            mf["AssocStr"]
