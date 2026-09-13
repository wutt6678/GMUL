"""Iteration 11C post-run audit — tested on CPU, with no adapter and no GPU.

The audit report states what the scoring run did NOT record, so the tests here
have a different job from the rest of the suite: they must show that the report
is *derived* from the committed evidence rather than asserting what its author
believed.  Two of them are regressions for bugs found in the audit script while
it was being written, because both produced a plausible-looking wrong number:

* an out-of-memory traceback is printed BEFORE its attempt's release line, so
  attributing it to the last *appended* attempt hangs attempt 2's failure on
  attempt 1;
* ``generation progress`` is logged every 256 queries and generation ends at
  1,209 without a further line, so the last progress sample reads 1024 and
  under-reports the state by 185 queries.

Everything here needs only the committed run logs, sidecars, freeze and analysis
report, so it runs in CI and in a fresh clone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_confirmation_execution as ace

REPORT = REPO_ROOT / ace.OUT_REPORT
LOGS = REPO_ROOT / ace.LOG_DIR
ANALYSIS = REPO_ROOT / ace.ANALYSIS_REPORT
FREEZE = REPO_ROOT / ace.FREEZE_REPORT
PREDICTIONS = REPO_ROOT / ace.PREDICTIONS
STATES = ace.STATES

#: Blocks that legitimately differ between two runs of the audit: the time it was
#: assembled, and the host's present-state hardware, which is labelled as NOT
#: run-time evidence precisely because it can change.
VOLATILE = (("generated_utc",),
            ("hardware", "observed_at_audit_time_not_at_run_time"))


def _flat(doc: dict, prefix: tuple = ()) -> dict:
    """Flatten to dotted paths, dropping the volatile blocks."""
    out: dict = {}
    for key, value in doc.items():
        path = prefix + (key,)
        if any(path[:len(v)] == v for v in VOLATILE):
            continue
        if isinstance(value, dict):
            out.update(_flat(value, path))
        else:
            out[path] = value
    return out


def _sidecar(state: str) -> dict:
    path = PREDICTIONS / f"predictions_{state}.parquet.provenance.json"
    return json.loads(path.read_text())


def _require_history() -> str:
    """Skip the checks that read the scoring commit out of git history.

    "Can this clone see that commit?" is answered by the audit module rather
    than re-probed here, so there is one implementation of it and one place for
    it to be wrong.  A shallow clone is a legitimate thing to run the suite
    from — ``actions/checkout`` produces one by default — so skipping is the
    honest response; ``TestTheHistoryTheAuditNeedsIsActuallyProvided`` is what
    stops that skip from silently becoming permanent.
    """
    commit = json.loads(REPORT.read_text())["code_identity"]["scoring_commit"]
    if not ace.history_available(commit):
        pytest.skip(f"the scoring commit {commit[:12]} is not in this clone "
                    f"(a shallow checkout); fetch full history to run this")
    return commit


def _lane_log(state: str) -> dict:
    return ace.parse_lane_log(
        (LOGS / f"confirm100_11c5_{state}.log").read_text())


#: The real ordering, reduced: claim, traceback, release, re-queue.
_TWO_OOM_THEN_SUCCESS = """\
2026-09-09T12:40:19+08:00 lane waiting for >=22000MiB free (poll 1s): cmd
2026-09-09T15:08:18+08:00 CLAIMED GPU 3 (38739MiB free, attempt 1/8): cmd
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.53 GiB.
2026-09-09T15:08:44+08:00 GPU 3 released (exit 1)
2026-09-09T15:08:44+08:00 re-queueing; new threshold 24000MiB
2026-09-09T16:48:52+08:00 CLAIMED GPU 0 (45240MiB free, attempt 2/8): cmd
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 17.53 GiB.
2026-09-09T16:49:15+08:00 GPU 0 released (exit 1)
2026-09-09T16:49:15+08:00 re-queueing; new threshold 26000MiB
2026-09-09T17:36:19+08:00 CLAIMED GPU 3 (42003MiB free, attempt 3/8): cmd
generation progress: 256/1209 (21.2%) in 99s — 2.59 q/s, ETA 368s
generation progress: 1024/1209 (84.7%) in 360s — 2.84 q/s, ETA 65s
[2026-09-09 17:44:02] INFO  evaluate_confirmation_split — [MG] wrote \
predictions_MG.parquet (1209 rows) and its sidecar
2026-09-09T17:44:04+08:00 GPU 3 released (success)
"""


class TestTheLaneLogParserDoesNotMisreadTheRun:
    """The two bugs that produced a plausible wrong number, kept fixed."""

    def test_an_oom_belongs_to_the_attempt_still_in_flight(self):
        parsed = ace.parse_lane_log(_TWO_OOM_THEN_SUCCESS)
        assert parsed["num_attempts"] == 3
        first, second, third = parsed["attempts"]
        assert first["oom_tried_to_allocate_gib"] == ["17.53"]
        assert second["oom_tried_to_allocate_gib"] == ["17.53"], (
            "the second out-of-memory must stay on the second attempt; the "
            "traceback precedes its release line, so attributing it to the "
            "last appended attempt moves it back one")
        assert "oom_tried_to_allocate_gib" not in third
        assert [a["succeeded"] for a in parsed["attempts"]] == [
            False, False, True]

    def test_the_sealed_row_count_is_not_the_last_progress_sample(self):
        parsed = ace.parse_lane_log(_TWO_OOM_THEN_SUCCESS)
        assert parsed["rows_written_and_sealed"] == 1209
        assert parsed["last_progress_sample"]["done"] == 1024, (
            "the sample is kept, labelled as a sample, because dropping it "
            "would hide why the count and the log disagree")
        assert parsed["rows_written_and_sealed"] != \
            parsed["last_progress_sample"]["done"]

    def test_a_claim_with_no_release_line_is_not_silently_dropped(self):
        parsed = ace.parse_lane_log(
            "2026-09-09T15:08:18+08:00 CLAIMED GPU 2 (30000MiB free, "
            "attempt 1/8): cmd\n")
        assert parsed["num_attempts"] == 1
        assert parsed["attempts"][0]["succeeded"] is False
        assert "no release line" in parsed["attempts"][0]["outcome"]
        assert parsed["gpu_index_that_generated"] is None

    def test_the_real_mg_log_reads_as_three_attempts_success_on_the_third(self):
        parsed = _lane_log("MG")
        assert parsed["num_attempts"] == 3
        assert parsed["succeeded_on_attempt"] == 3
        assert parsed["gpu_index_that_generated"] == 3
        assert parsed["memory_thresholds_mib"] == [22000, 24000, 26000]
        assert parsed["rows_written_and_sealed"] == 1209

    def test_all_three_states_generated_on_one_device_index(self):
        indices = [_lane_log(s)["gpu_index_that_generated"] for s in STATES]
        assert indices == [3, 3, 3]
        assert all(_lane_log(s)["num_attempts"] == 1 for s in ("B3", "B0"))

    def test_the_hardware_vocabulary_search_can_find_something(self):
        """Non-vacuity: a search that returns empty everywhere proves nothing."""
        found = ace.hardware_vocabulary_hits(LOGS)
        assert found, "the search returned no terms at all"
        assert all(v == [] for v in found.values()), (
            "a hardware identifier appears in the committed run logs, which "
            "contradicts the report's central limitation")

    def test_the_search_finds_a_term_that_is_present(self, tmp_path):
        (tmp_path / "probe.log").write_text("NVIDIA RTX 6000 Ada Generation\n")
        hits = ace.hardware_vocabulary_hits(tmp_path)
        assert hits["RTX"] == ["probe.log"]
        assert hits["A100"] == []


class TestTheCodeIdentityIsReadRatherThanAsserted:
    def test_the_scoring_commit_is_the_one_the_sidecars_record(self):
        report = json.loads(REPORT.read_text())
        commits = {s: _sidecar(s)["code"]["git_commit"] for s in STATES}
        assert len(set(commits.values())) == 1
        assert report["code_identity"]["scoring_commit"] == commits["B3"]
        assert report["code_identity"]["commits_per_state"] == commits

    def test_git_dirty_is_reported_as_the_sidecars_record_it(self):
        report = json.loads(REPORT.read_text())
        assert report["code_identity"]["git_dirty_per_state"] == {
            s: _sidecar(s)["code"]["git_dirty"] for s in STATES}

    def test_the_unbound_verifier_is_in_neither_pin_set(self):
        """Derived from the freeze, not from the report's own claim."""
        freeze = json.loads(FREEZE.read_text())
        code = freeze["code"]
        assert ace.VERIFIER not in code["fingerprinted_modules"]
        assert ace.VERIFIER not in code["analysis_scripts_sha256"]
        report = json.loads(REPORT.read_text())
        gap = report["code_identity"]["verifier_binding_gap"]
        assert gap["in_code_fingerprinted_modules"] is False
        assert gap["in_code_analysis_scripts_sha256"] is False
        assert gap["recorded_in_any_sidecar"] is True

    def test_the_verifier_bytes_match_the_scoring_commit(self):
        """Re-derived here rather than trusted from the report.

        The report's whole claim is that the unpinned verifier is nonetheless
        recoverable, because it is a tracked file and the scoring commit is
        recorded.  That is checkable: hash the blob at the commit, hash the
        working tree, and require all three to agree.
        """
        identity = json.loads(REPORT.read_text())["code_identity"]
        gap = identity["verifier_binding_gap"]
        _require_history()
        blob = subprocess.run(
            ("git", "show", f"{identity['scoring_commit']}:{ace.VERIFIER}"),
            cwd=REPO_ROOT, capture_output=True, check=False)
        #: The commit's presence is already established, so a failure here is
        #: the other fact and must not be reported as the first one: the path
        #: is genuinely absent from that commit.
        assert blob.returncode == 0, (
            f"{ace.VERIFIER} was not tracked at the scoring commit, so the "
            f"audit's recoverability claim is false")
        at_commit = hashlib.sha256(blob.stdout).hexdigest()
        on_disk = hashlib.sha256(
            (REPO_ROOT / ace.VERIFIER).read_bytes()).hexdigest()
        assert at_commit == on_disk == gap["sha256_now"]
        assert gap["sha256_at_the_scoring_commit"] == at_commit
        assert gap["unchanged_since_scoring"] is True

    def test_the_sealed_list_is_derived_from_the_freeze(self):
        freeze = json.loads(FREEZE.read_text())
        sealed = ace.sealed_paths(freeze)
        expected = set(freeze["code"]["fingerprinted_modules"])
        expected |= set(freeze["code"]["analysis_scripts_sha256"])
        expected |= {freeze["primary_test"]["implementation"]["module"]}
        assert set(sealed) == expected
        assert len(sealed) == 18
        report = json.loads(REPORT.read_text())
        assert report["code_identity"]["sealed_paths"] == sealed
        assert report["code_identity"]["sealed_paths_changed_since_scoring"] == []

    def test_the_audit_script_is_not_itself_hash_bound(self):
        """Adding it to ANALYSIS_SCRIPTS would change the freeze's own hash set
        and force a re-freeze, which after scoring would postdate the result the
        freeze exists to precede."""
        freeze = json.loads(FREEZE.read_text())
        rel = "scripts/audit_confirmation_execution.py"
        assert rel not in freeze["code"]["analysis_scripts_sha256"]
        assert rel not in freeze["code"]["fingerprinted_modules"]


class TestTheReportIsReproducibleFromTheRepository:
    def test_two_fresh_builds_agree_apart_from_the_timestamped_blocks(self):
        first = _flat(ace.build_report(REPO_ROOT))
        second = _flat(ace.build_report(REPO_ROOT))
        assert first == second
        assert first, "the comparison covered no fields at all"

    def test_the_committed_report_matches_a_fresh_build(self):
        #: Needs history, unlike the test above: two degraded builds still agree
        #: with each other, but a degraded build never agrees with the filed
        #: report, which was assembled from a clone that could see the commit.
        _require_history()
        committed = _flat(json.loads(REPORT.read_text()))
        fresh = _flat(ace.build_report(REPO_ROOT))
        assert committed == fresh, (
            "the committed audit report is not what the committed evidence "
            "produces, so it was edited by hand or the evidence moved")

    def test_every_lane_log_agrees_with_its_own_sidecar(self):
        counts = json.loads(REPORT.read_text())["row_counts"]
        assert counts["all_lane_logs_agree_with_their_own_sidecars"] is True
        for state in STATES:
            assert counts["per_state"][state]["sidecar_num_rows"] == 1209
            assert counts["per_state"][state]["agree"] is True

    def test_the_committed_logs_are_the_ones_the_report_hashes(self):
        report = json.loads(REPORT.read_text())
        files = report["run_logs"]["files"]
        assert sorted(files) == sorted(p.name for p in LOGS.iterdir())
        for name, meta in files.items():
            path = LOGS / name
            assert meta["bytes"] == path.stat().st_size
            assert meta["sha256"] == ace._sha256(path)


class TestTheAuditStatesWhatItIs:
    def test_it_is_marked_post_hoc_and_binds_nothing(self):
        report = json.loads(REPORT.read_text())
        assert report["post_hoc"] is True
        assert report["assembled_after_the_result_was_seen"] is True
        assert report["binds_nothing_and_enforces_nothing"] is True

    def test_it_names_the_result_it_audits(self):
        report = json.loads(REPORT.read_text())
        analysis = json.loads(ANALYSIS.read_text())
        audit_of = report["audit_of"]
        assert audit_of["analysis_report_sha256"] == ace._sha256(ANALYSIS)
        assert audit_of["analysis_generated_utc"] == analysis["generated_utc"]
        assert audit_of["primary_verdict"] == \
            analysis["primary"]["verdict"]["rejected"]

    def test_the_equivalence_conclusion_is_the_negative_one(self):
        report = json.loads(REPORT.read_text())
        assert report["physical_gpu_equivalence"][
            "establishable_from_the_record"] is False
        assert report["do_not_rescore"]["the_result_stands"] is True
        #: Casefolded because the report shouts the phrase and this assertion is
        #: about the commitment, not the typography.
        assert "new replication" in report["do_not_rescore"][
            "any_repeat_is_a_new_replication"].casefold()

    def test_the_post_hoc_hardware_block_says_it_is_not_run_time_evidence(self):
        block = json.loads(REPORT.read_text())["hardware"][
            "observed_at_audit_time_not_at_run_time"]
        assert block["is_this_run_time_evidence"] is False
        assert "observed_utc" in block
        #: Whether nvidia-smi ran is a property of the machine reading this, so
        #: only the labelling is asserted, never the device list.
        assert isinstance(block["available"], bool)

    def test_the_run_time_hardware_fields_are_all_empty(self):
        recorded = json.loads(REPORT.read_text())["hardware"][
            "recorded_at_run_time"]
        for field in ("device_model", "device_uuid", "driver_version",
                      "cuda_version", "compute_capability"):
            assert recorded[field] is None, field
        assert recorded["the_device_index"] == 3

    def test_the_limitations_name_both_gaps(self):
        limitations = json.loads(REPORT.read_text())["limitations"]
        assert len(limitations) >= 6
        joined = " ".join(limitations)
        assert "prediction_provenance.py" in joined
        assert "compute capability" in joined
        assert "retrospectively" in joined


class TestTheAuditRefusesRatherThanGuessing:
    def test_absent_evidence_refuses(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ace, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["audit"])
        assert ace.main() == 1
        assert "REFUSED" in capsys.readouterr().out

    def test_the_real_repository_does_not_refuse(self, tmp_path, monkeypatch):
        #: The write path needs history: without the scoring commit the report
        #: would be missing three findings, and main() refuses to file that.
        _require_history()
        before = ace._sha256(REPORT)
        dest = tmp_path / "audit.json"
        monkeypatch.setattr(sys, "argv", ["audit", "--output", str(dest)])
        assert ace.main() == 0
        assert json.loads(dest.read_text())["post_hoc"] is True
        assert ace._sha256(REPORT) == before, (
            "an audit run with an explicit --output still rewrote the committed "
            "report")

    def test_stdout_still_works_without_history_and_says_what_it_cannot_answer(
            self, monkeypatch, capsys):
        """``--stdout`` is not refused, and the degradation is visible in it.

        Printing cannot touch the filed report, so refusing it would only hide
        the three fields that explain why the checkout is short.  Their
        replacement text is what tells a reader that this is a clone artifact
        rather than a finding about the repository.

        All three git entry points are patched, not just the probe: in a real
        depth-1 clone the ``.gitignore`` lookup fails too, and patching the
        probe alone would leave that one answering from real history — which is
        why this test passed in a shallow clone and failed in a full one until
        the simulation was made faithful.
        """
        before = ace._sha256(REPORT)
        monkeypatch.setattr(ace, "history_available", lambda commit: False)
        monkeypatch.setattr(ace, "_git", lambda *a: None)
        monkeypatch.setattr(ace, "_git_blob_sha256", lambda c, p: None)
        monkeypatch.setattr(sys, "argv", ["audit", "--stdout"])
        assert ace.main() == 0
        doc = json.loads(capsys.readouterr().out)
        gap = doc["code_identity"]["verifier_binding_gap"]
        assert gap["recoverable_from_git_because_tracked"] is None
        assert gap["unchanged_since_scoring"] is None
        assert doc["code_identity"]["why_b0_and_mg_report_dirty"] == \
            ace.UNANSWERABLE_IN_THIS_CLONE
        assert ace._sha256(REPORT) == before

    def test_stdout_mode_writes_nothing(self, monkeypatch, capsys):
        before = ace._sha256(REPORT)
        monkeypatch.setattr(sys, "argv", ["audit", "--stdout"])
        assert ace.main() == 0
        doc = json.loads(capsys.readouterr().out)
        assert doc["post_hoc"] is True
        assert ace._sha256(REPORT) == before


class TestTheHistoryTheAuditNeedsIsActuallyProvided:
    """A skip is honest only if something fails when the reason to skip returns.

    ``actions/checkout`` defaults to ``fetch-depth: 1``.  Under a depth-1
    checkout the 11C scoring commit is absent, so the checks that re-derive the
    audit's claims from history cannot run and ``code_identity`` cannot answer
    three of its own questions.  Skipping there is right for a laptop and wrong
    for CI, so the CI setting is asserted here: remove it and a test fails,
    instead of three verifications quietly becoming no-ops that still report
    green.
    """

    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

    def test_ci_checks_out_full_history(self):
        doc = yaml.safe_load(self.WORKFLOW.read_text())
        steps = doc["jobs"]["unit-tests"]["steps"]
        checkouts = [s for s in steps
                     if str(s.get("uses", "")).startswith("actions/checkout")]
        assert len(checkouts) == 1, checkouts
        depth = checkouts[0].get("with", {}).get("fetch-depth")
        assert depth == 0, (
            f"actions/checkout defaults to fetch-depth: 1, under which the "
            f"scoring commit is absent and the history-derived audit checks "
            f"skip instead of verifying; got {depth!r}")

    def test_a_depth_1_clone_really_cannot_see_the_scoring_commit(
            self, tmp_path, monkeypatch):
        """The cause, reproduced rather than simulated.

        The other tests here patch ``history_available`` to False; this one
        shows that False is what a depth-1 checkout actually produces, which is
        the single link a simulation cannot cover — and the link that was
        wrong, since git reports the missing COMMIT as a missing PATH.
        """
        commit = json.loads(
            REPORT.read_text())["code_identity"]["scoring_commit"]
        clone = tmp_path / "depth1"
        out = subprocess.run(
            ("git", "clone", "--quiet", "--depth", "1",
             REPO_ROOT.as_uri(), str(clone)),
            capture_output=True, text=True, check=False)
        if out.returncode != 0:
            pytest.skip(f"cannot make a depth-1 clone here: {out.stderr}")
        monkeypatch.setattr(ace, "REPO_ROOT", clone)
        assert ace.history_available(commit) is False
        #: And the probe is not simply always False — the tip resolves.
        assert ace.history_available("HEAD") is True
        blob = subprocess.run(("git", "show", f"{commit}:{ace.VERIFIER}"),
                              cwd=clone, capture_output=True, text=True,
                              check=False)
        assert blob.returncode != 0
        #: The message that made this look like a repository fact.
        assert "exists on disk, but not in" in blob.stderr

    def test_the_degraded_fields_say_unknown_rather_than_no(
            self, monkeypatch):
        """The defect: three fields turned an unanswerable question into a
        false negative, and two of them contradicted the audit's own
        mitigation."""
        monkeypatch.setattr(ace, "history_available", lambda commit: False)
        monkeypatch.setattr(ace, "_git", lambda *a: None)
        monkeypatch.setattr(ace, "_git_blob_sha256", lambda c, p: None)
        sidecars = {s: _sidecar(s) for s in STATES}
        ci = ace.code_identity(REPO_ROOT, sidecars)
        gap = ci["verifier_binding_gap"]
        #: Not False: False would assert the verifier is untracked and that it
        #: changed, and the audit's mitigation is that neither holds.
        assert gap["recoverable_from_git_because_tracked"] is None
        assert gap["unchanged_since_scoring"] is None
        assert gap["sha256_at_the_scoring_commit"] is None
        assert ci["predictions_dir_ignored_at_the_scoring_commit"] is None
        #: Not "investigate": there is nothing to investigate, the question was
        #: never asked.
        assert ci["why_b0_and_mg_report_dirty"] == \
            ace.UNANSWERABLE_IN_THIS_CLONE
        assert "investigate" not in ci["why_b0_and_mg_report_dirty"]

    def test_the_audit_refuses_to_write_without_history(
            self, tmp_path, monkeypatch, capsys):
        """Building a degraded report is harmless; committing one is not.

        The filed audit is the record, and overwriting it from a depth-1
        checkout would replace three findings with unknowns while every other
        field still looked authoritative.
        """
        before = ace._sha256(REPORT)
        monkeypatch.setattr(ace, "history_available", lambda commit: False)
        #: Under tmp_path, not under data/reports: a test that writes into the
        #: repository when it fails leaves the failure behind as a stray
        #: artifact that the next run then has to explain.
        dest = tmp_path / "must_not_be_written.json"
        monkeypatch.setattr(sys, "argv",
                            ["audit", "--output", str(dest)])
        assert ace.main() == 1
        said = capsys.readouterr().out
        assert "REFUSED" in said
        assert "fetch-depth: 0" in said
        assert not dest.exists()
        assert ace._sha256(REPORT) == before
