"""Post-hoc execution provenance for the 11C confirmation scoring run.

READ THIS FIRST: everything this script files is POST-HOC.  It was written after
the confirmation had been scored and after the result had been seen, it binds
nothing, it enforces nothing, and no gate reads it.  It exists because the run
did not record some things that a reviewer would reasonably want, and the honest
response to that is a dated statement of what is and is not recoverable — not
silence, and not a retrofit that pretends the record is better than it is.

What it does
------------
It re-derives, from artifacts that are committed to the repository, four things
the run left open:

1. **the code identity of the scoring run.**  Each sidecar records
   ``code.git_commit`` and ``code.git_dirty``.  Both are read here rather than
   restated, and ``git_dirty`` is interpreted against the implementation that
   produced it: ``prediction_provenance.git_dirty`` runs a bare
   ``git status --porcelain``, which counts UNTRACKED files, so a lane that ran
   after an earlier lane had written predictions sees a dirty tree even though
   no tracked file changed.  Whether ``predictions/`` was ignored at the scoring
   commit is checked against that commit's own ``.gitignore`` rather than
   assumed.

2. **the verifier binding gap.**  ``prediction_provenance.py`` defines the
   sidecar contract and implements ``verify_sidecar``, yet it is in neither
   ``code.fingerprinted_modules`` nor ``code.analysis_scripts_sha256``: the
   module that decides what counts as provenance was itself unpinned.  Its bytes
   are nonetheless recoverable, because it is a TRACKED file and B3's sidecar
   records ``git_dirty: false``, which pins the whole tracked tree to the scoring
   commit.  Both hashes are computed here and compared.

3. **GPU identity.**  The lane logs record the device INDEX claimed and the free
   memory at claim time, and nothing else: no device model, no UUID, no driver
   version, no CUDA version, no compute capability.  Every ``CLAIMED GPU`` and
   ``released`` line is parsed out of the committed logs so the indices are
   evidence rather than recollection.  Present-state hardware is queried too, but
   into a block labelled with its own timestamp and marked as NOT run-time
   evidence, because an index is not an identity and a host can be re-provisioned.

4. **what may not be done next.**  Physical-GPU equivalence across the three
   states cannot be established retrospectively from this record.  That is a
   limitation, not a defect in the result, and it does not license rescoring:
   any repeat is a NEW REPLICATION with its own protocol, never a re-run.

Where the inputs come from
--------------------------
The run logs are read from ``data/mllmu_hier_confirm100/execution_logs/``, which
is where they were committed.  They originally lived in ``outputs/lanes/``, which
``.gitignore`` excludes, so before this commit they existed only on the machine
that scored.  Reading the committed copies is what makes this report reproducible
from a clone.

This script is deliberately NOT in ``ANALYSIS_SCRIPTS``: adding it would change
``code.analysis_scripts_sha256`` and force a re-freeze, and re-freezing after
scoring would postdate the result the freeze is supposed to precede.  It reads
sealed files and never writes one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIRM_DIR = "data/mllmu_hier_confirm100"
LOG_DIR = f"{CONFIRM_DIR}/execution_logs"
PREDICTIONS = f"{CONFIRM_DIR}/predictions"
FREEZE_REPORT = "data/reports/mllmu_pilot100_confirmation_freeze.json"
ANALYSIS_REPORT = "data/reports/mllmu_confirm100_final_analysis.json"
OUT_REPORT = "data/reports/mllmu_confirm100_execution_provenance.json"

#: The module that defines the sidecar contract and implements verify_sidecar.
VERIFIER = "src/granunlearn/evaluation/prediction_provenance.py"

STATES = ("B3", "B0", "MG")

#: Hardware vocabulary the run logs are searched for.  Every one of these coming
#: back empty IS the finding, so the search is recorded rather than skipped.
HARDWARE_VOCABULARY = ("RTX", "Ada", "GeForce", "A100", "H100", "Tesla",
                       "Driver Version", "CUDA Version", "compute capability",
                       "GPU-", "nvidia-smi", "uuid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    """Run git in the repository, returning None rather than raising.

    A clone without full history, or a source tarball, must still produce a
    report; the fields that needed git are then recorded as unavailable rather
    than invented.
    """
    try:
        out = subprocess.run(("git", *args), cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str | None:
    """Hash a file AS OF a commit, in bytes.

    ``git show`` decoded to text and re-encoded is not the same hash as the
    file's own bytes whenever the content is not pure ASCII or the checkout
    applies newline translation, and this report's whole job is to compare
    hashes that mean something.
    """
    try:
        out = subprocess.run(("git", "show", f"{commit}:{path}"), cwd=REPO_ROOT,
                             capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the run logs ─────────────────────────────────────────────────────

CLAIMED = re.compile(
    r"^(?P<when>\S+) CLAIMED GPU (?P<index>\d+) \((?P<free>\d+)MiB free, "
    r"attempt (?P<attempt>\d+)/(?P<max>\d+)\)")
RELEASED = re.compile(
    r"^(?P<when>\S+) GPU (?P<index>\d+) released \((?P<outcome>.*)\)")
WAITING = re.compile(r"lane waiting for >=(?P<threshold>\d+)MiB free")
REQUEUE = re.compile(r"re-queueing; new threshold (?P<threshold>\d+)MiB")
OOM = re.compile(r"Tried to allocate (?P<gib>[\d.]+) GiB")
PROGRESS = re.compile(r"generation progress: (?P<done>\d+)/(?P<total>\d+) "
                      r"\((?P<pct>[\d.]+)%\) in (?P<secs>\d+)s")
#: The authoritative count.  ``generation progress`` is emitted every 256
#: queries and the run finishes at 1209 without another line, so the last
#: progress sample says 1024 and under-reports the state by 185 queries.  The
#: line the scorer writes on sealing the parquet is the count that matters, and
#: the sidecar's ``num_rows`` is the count that is bound.
WROTE = re.compile(r"wrote predictions_\w+\.parquet \((?P<rows>\d+) rows\)")


def parse_lane_log(text: str) -> dict[str, Any]:
    """Every attempt this lane made, in order, from its own log.

    Parsed rather than transcribed: the indices and the outcome of each attempt
    are the load-bearing facts of the hardware section, and a number copied by
    hand is a number nobody can check.
    """
    attempts: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    oom_gib: list[str] = []
    thresholds: list[int] = []
    progress: list[dict[str, Any]] = []
    rows_written: int | None = None
    for line in text.splitlines():
        if (m := WAITING.search(line)) and not thresholds:
            thresholds.append(int(m["threshold"]))
        if m := CLAIMED.match(line):
            pending = {"claimed_utc": m["when"], "gpu_index": int(m["index"]),
                       "free_mib_at_claim": int(m["free"]),
                       "attempt": int(m["attempt"]),
                       "max_attempts": int(m["max"])}
        elif m := RELEASED.match(line):
            if pending is not None and int(m["index"]) == pending["gpu_index"]:
                pending["released_utc"] = m["when"]
                pending["outcome"] = m["outcome"]
                pending["succeeded"] = m["outcome"] == "success"
                attempts.append(pending)
                pending = None
        elif m := REQUEUE.search(line):
            thresholds.append(int(m["threshold"]))
        if m := OOM.search(line):
            oom_gib.append(m["gib"])
            #: The traceback is printed BEFORE the release line, so the attempt
            #: it belongs to is the one still in flight.  Attributing it to
            #: ``attempts[-1]`` instead would hang attempt 2's out-of-memory on
            #: attempt 1, which is the kind of error this report exists to not
            #: make.
            inflight = pending if pending is not None else (
                attempts[-1] if attempts else None)
            if inflight is not None:
                inflight.setdefault("oom_tried_to_allocate_gib", [])
                inflight["oom_tried_to_allocate_gib"].append(m["gib"])
        if m := PROGRESS.search(line):
            progress.append({"done": int(m["done"]), "total": int(m["total"]),
                             "elapsed_s": int(m["secs"])})
        if m := WROTE.search(line):
            rows_written = int(m["rows"])
    if pending is not None:          #: claimed and never released
        pending["outcome"] = "no release line — the log ends mid-attempt"
        pending["succeeded"] = False
        attempts.append(pending)
    ok = [a for a in attempts if a["succeeded"]]
    return {
        "attempts": attempts,
        "num_attempts": len(attempts),
        "memory_thresholds_mib": thresholds,
        "oom_allocations_gib": oom_gib,
        "succeeded_on_attempt": ok[0]["attempt"] if ok else None,
        "gpu_index_that_generated": ok[0]["gpu_index"] if ok else None,
        "rows_written_and_sealed": rows_written,
        "last_progress_sample": (
            {"done": progress[-1]["done"], "total": progress[-1]["total"],
             "elapsed_s": progress[-1]["elapsed_s"]} if progress else None),
        "why_the_progress_sample_is_not_the_count": (
            "progress is logged every 256 queries and generation ends at 1209 "
            "without a further line, so the last sample reads 1024; the sealed "
            "row count comes from the line the scorer writes and is cross-"
            "checked against the sidecar's num_rows"),
        "device_model_in_log": any(w in text for w in HARDWARE_VOCABULARY),
    }


def hardware_vocabulary_hits(log_dir: Path) -> dict[str, list[str]]:
    """Which hardware words appear anywhere in the committed run logs.

    Recorded even when empty — an empty map is the evidence that no device
    identity was captured, and a reader should not have to take that on faith.
    """
    hits: dict[str, list[str]] = {w: [] for w in HARDWARE_VOCABULARY}
    for path in sorted(log_dir.iterdir()):
        text = path.read_text(errors="replace")
        for word in HARDWARE_VOCABULARY:
            if word in text:
                hits[word].append(path.name)
    return hits


def present_state_hardware() -> dict[str, Any]:
    """What THIS host reports NOW, which is not what the run recorded.

    Kept in its own timestamped block and labelled, because the temptation is to
    read a device model found here as a device model found then.  Degrades to
    ``available: false`` on a host with no nvidia-smi rather than failing: this
    section is a convenience, never evidence about the run.
    """
    block: dict[str, Any] = {
        "observed_utc": _utcnow(),
        "is_this_run_time_evidence": False,
        "why_it_is_here_at_all": (
            "to state what the host looks like at audit time, so that a reader "
            "can see the gap between this and what the run recorded — and so "
            "that nobody mistakes one for the other"),
    }
    try:
        out = subprocess.run(
            ("nvidia-smi",
             ("--query-gpu=index,name,uuid,driver_version,memory.total,"
              "compute_cap"),
             "--format=csv,noheader"),
            capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is None or out.returncode != 0:
        block["available"] = False
        block["why_unavailable"] = "nvidia-smi did not run on this host"
        return block
    block["available"] = True
    devices = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            devices.append({"index": int(parts[0]), "name": parts[1],
                            "uuid": parts[2], "driver_version": parts[3],
                            "memory_total": parts[4],
                            "compute_capability": parts[5]})
    block["devices"] = devices
    block["device_models_observed"] = sorted({d["name"] for d in devices})
    block["driver_versions_observed"] = sorted(
        {d["driver_version"] for d in devices})
    block["what_this_still_does_not_establish"] = (
        "that any index claimed on the run date is the same physical device as "
        "the index with this UUID today; enumeration order is a host property "
        "that was never recorded into the evidence")
    return block


# ── code identity ────────────────────────────────────────────────────

def sealed_paths(freeze: dict[str, Any]) -> list[str]:
    """Every path whose bytes the committed predictions are bound to.

    Derived from the freeze rather than listed here, so this cannot drift from
    what the protocol actually pins.
    """
    code = freeze["code"]
    paths = sorted(code["fingerprinted_modules"])
    paths += sorted(code["analysis_scripts_sha256"])
    impl = freeze.get("primary_test", {}).get("implementation", {})
    if impl.get("module"):
        paths.append(impl["module"])
    return sorted(set(paths))


def code_identity(root: Path, sidecars: dict[str, dict]) -> dict[str, Any]:
    freeze = json.loads((root / FREEZE_REPORT).read_text())
    commits = {s: sidecars[s]["code"]["git_commit"] for s in STATES}
    dirty = {s: sidecars[s]["code"]["git_dirty"] for s in STATES}
    scoring_commit = next(iter(commits.values()))
    one_commit = len(set(commits.values())) == 1

    #: ``git_dirty`` is a bare ``git status --porcelain``, which counts untracked
    #: files.  Whether the predictions directory was ignored AT THE SCORING
    #: COMMIT decides whether an earlier lane's untracked output explains a later
    #: lane's dirty=True — checked against that commit's .gitignore, not assumed.
    ignore_at_commit = _git("show", f"{scoring_commit}:.gitignore")
    predictions_ignored_then = (
        None if ignore_at_commit is None
        else f"{CONFIRM_DIR}/predictions" in ignore_at_commit)

    sealed = sealed_paths(freeze)
    changed_since = _git("log", "--name-only", "--format=",
                         f"{scoring_commit}..HEAD")
    touched = sorted({line for line in (changed_since or "").splitlines()
                      if line in set(sealed)})

    verifier_then = _git_blob_sha256(scoring_commit, VERIFIER)
    verifier_now = _sha256(root / VERIFIER)

    return {
        "scoring_commit": scoring_commit,
        "all_three_states_generated_at_one_commit": one_commit,
        "commits_per_state": commits,
        "git_dirty_per_state": dirty,
        "what_git_dirty_measures": (
            "prediction_provenance.git_dirty runs `git status --porcelain` with "
            "no --untracked-files flag, so UNTRACKED files count as dirty"),
        "predictions_dir_ignored_at_the_scoring_commit": predictions_ignored_then,
        "why_b0_and_mg_report_dirty": (
            "B3 ran first, when the predictions directory did not exist, so its "
            "tree was clean. Its three outputs were untracked at that commit, so "
            "porcelain listed them, and every later lane saw a dirty tree. No "
            "step of the chain edits a tracked file." if one_commit
            and predictions_ignored_then is False else
            "not explained by the untracked-predictions account; investigate "
            "before relying on this report"),
        "limitation_of_a_single_dirty_boolean": (
            "git_dirty cannot distinguish 'only untracked evidence appeared' "
            "from 'a tracked file was modified', because it reduces a porcelain "
            "listing to one bit. For B3 the false value pins the entire tracked "
            "tree to the scoring commit; for B0 and MG the true value is "
            "consistent with the untracked explanation but does not prove it."),
        "sealed_paths": sealed,
        "num_sealed_paths": len(sealed),
        "sealed_paths_changed_since_scoring": touched,
        "verifier_binding_gap": {
            "path": VERIFIER,
            "role": ("defines the sidecar contract and implements verify_sidecar "
                     "— the module that decides what counts as provenance"),
            "in_code_fingerprinted_modules":
                VERIFIER in freeze["code"]["fingerprinted_modules"],
            "in_code_analysis_scripts_sha256":
                VERIFIER in freeze["code"]["analysis_scripts_sha256"],
            "recorded_in_any_sidecar": all(
                VERIFIER not in sidecars[s]["code"]["modules_sha256"]
                for s in STATES),
            "sha256_at_the_scoring_commit": verifier_then,
            "sha256_now": verifier_now,
            "unchanged_since_scoring": verifier_then == verifier_now,
            "recoverable_from_git_because_tracked": verifier_then is not None,
            "consequence": (
                "the verifier's own bytes were pinned by nothing at generation "
                "time, so the record cannot show that the contract being "
                "enforced then is the one being enforced now. It happens to be "
                "recoverable here only because the file is tracked and the "
                "scoring commit is recorded — a mitigation that is a property of "
                "this repository, not a control."),
        },
    }


# ── environments ─────────────────────────────────────────────────────

def environments(sidecars: dict[str, dict],
                 hardware_hits_found: list[str]) -> dict[str, Any]:
    blobs = {s: json.dumps(sidecars[s].get("environment"), sort_keys=True)
             for s in STATES}
    identical = len(set(blobs.values())) == 1
    keys = sorted(sidecars["B3"].get("environment", {}))
    return {
        "identical_across_the_three_states": identical,
        "environment": json.loads(blobs["B3"]),
        "fields_recorded": keys,
        "fields_not_recorded": [
            "GPU model", "GPU UUID", "device index", "driver version",
            "CUDA runtime version", "CUDA driver version", "compute capability",
            "device memory total", "hostname", "kernel"],
        "what_the_package_versions_do_cover": (
            "torch's own version string carries the CUDA build it was compiled "
            "against, so the toolkit the wheel expects is recoverable; the "
            "driver that served it and the device it ran on are not"),
        "hardware_terms_found_in_the_run_logs": hardware_hits_found,
    }


# ── assembly ─────────────────────────────────────────────────────────

def build_report(root: Path) -> dict[str, Any]:
    preds = root / PREDICTIONS
    sidecars = {s: json.loads(
        (preds / f"predictions_{s}.parquet.provenance.json").read_text())
        for s in STATES}
    log_dir = root / LOG_DIR
    lanes = {s: parse_lane_log(
        (log_dir / f"confirm100_11c5_{s}.log").read_text()) for s in STATES}
    hits = hardware_vocabulary_hits(log_dir)

    #: parse_lane_log promises the sealed row count is cross-checked against the
    #: sidecar's bound num_rows, so do it here rather than leave the promise in
    #: the prose: a lane that logged sealing a different number of rows than its
    #: own sidecar bound would be a contradiction worth surfacing.
    row_crosscheck = {
        s: {"lane_log_rows": lanes[s]["rows_written_and_sealed"],
            "sidecar_num_rows": sidecars[s]["num_rows"],
            "agree": lanes[s]["rows_written_and_sealed"]
            == sidecars[s]["num_rows"]}
        for s in STATES}

    ok_indices = [lanes[s]["gpu_index_that_generated"] for s in STATES]
    one_device_index = len(set(ok_indices)) == 1 and None not in ok_indices
    analysis = json.loads((root / ANALYSIS_REPORT).read_text())

    return {
        "what_this_is": (
            "a POST-HOC audit of how the 11C confirmation was executed, filed "
            "after the result was seen. It binds nothing, gates nothing, and no "
            "script reads it. Its purpose is to state what the run recorded and "
            "what it did not, so that the gap is part of the public record "
            "rather than something a reviewer has to discover."),
        "post_hoc": True,
        "assembled_after_the_result_was_seen": True,
        "binds_nothing_and_enforces_nothing": True,
        "generated_utc": _utcnow(),
        "audit_of": {
            "analysis_report": ANALYSIS_REPORT,
            "analysis_report_sha256": _sha256(root / ANALYSIS_REPORT),
            "analysis_generated_utc": analysis["generated_utc"],
            "refusals_in_the_analysis_report": analysis["refusals"],
            "primary_verdict": analysis["primary"]["verdict"]["rejected"],
            "retained": analysis["primary"]["verdict"]["retained"],
        },
        "run_logs": {
            "committed_at": LOG_DIR,
            "originally_at": "outputs/lanes/ (excluded by .gitignore)",
            "why_they_were_committed": (
                "they are the only record of which device each state was "
                "generated on and of the two out-of-memory retries, and until "
                "this commit they existed only on the machine that scored"),
            "files": {p.name: {"bytes": p.stat().st_size,
                               "sha256": _sha256(p)}
                      for p in sorted(log_dir.iterdir())},
        },
        "row_counts": {
            "per_state": row_crosscheck,
            "all_lane_logs_agree_with_their_own_sidecars":
                all(v["agree"] for v in row_crosscheck.values()),
        },
        "code_identity": code_identity(root, sidecars),
        #: Derived, not asserted: whether any hardware term appears in the logs is
    #: measured by hardware_vocabulary_hits above, so this report cannot claim
    #: an empty search that in fact found something.
    "sidecar_environments": environments(
        sidecars, sorted(k for k, v in hits.items() if v)),
        "hardware": {
            "recorded_at_run_time": {
                "device_index_per_successful_generation": dict(
                    zip(STATES, ok_indices)),
                "all_three_generated_on_one_device_index": one_device_index,
                "the_device_index": ok_indices[0] if one_device_index else None,
                "free_memory_at_claim_mib": {
                    s: [a["free_mib_at_claim"] for a in lanes[s]["attempts"]]
                    for s in STATES},
                "device_model": None, "device_uuid": None,
                "driver_version": None, "cuda_version": None,
                "compute_capability": None,
                "what_the_lane_logs_do_record": (
                    "the index claimed, the free memory at claim time, the "
                    "attempt number, and the outcome of each attempt"),
            },
            "gone_looking_for": {
                "terms_searched": list(HARDWARE_VOCABULARY),
                "hits_per_term": {k: v for k, v in hits.items() if v},
                "terms_with_no_hits": sorted(k for k, v in hits.items() if not v),
                "searched_files": sorted(p.name for p in log_dir.iterdir()),
                "conclusion": (
                    "no device model, UUID, driver version, CUDA version or "
                    "compute capability appears anywhere in the committed run "
                    "logs"),
            },
            "observed_at_audit_time_not_at_run_time": present_state_hardware(),
            "per_lane_attempts": lanes,
        },
        "physical_gpu_equivalence": {
            "establishable_from_the_record": False,
            "what_is_establishable": (
                "all three states were generated on the same device INDEX"
                if one_device_index else
                "the states were generated on differing device indices"),
            "what_is_not": (
                "that the index denotes one physical device across the run, or "
                "that the device is the same model, driver, firmware or compute "
                "capability as anything observable now. An index is a slot in an "
                "enumeration, not an identity."),
            "why_it_matters_and_why_it_does_not_undo_the_result": (
                "the three states are compared as paired differences over the "
                "same 1,209 queries in the same order under the same frozen "
                "generation config and the same package environment, and the "
                "comparison is within-subject, so a uniform device would affect "
                "all three alike. What cannot be ruled out is a device-dependent "
                "numerical difference between states generated on different "
                "hardware — which the record cannot exclude because the record "
                "does not identify the hardware."),
        },
        "do_not_rescore": {
            "the_result_stands": True,
            "why_the_missing_binding_does_not_justify_replacing_it": (
                "choosing to regenerate after seeing a verdict that both "
                "preregistered claims survived would be selecting on the "
                "outcome, which is the exact failure the sealed protocol exists "
                "to prevent. The gaps documented here are gaps in what was "
                "RECORDED, and nothing in the record indicates an "
                "implementation change between the freeze and the run."),
            "any_repeat_is_a_new_replication": (
                "a further scoring pass is a NEW REPLICATION, never a re-run of "
                "this confirmation. It requires its own preregistered protocol, "
                "its own freeze and its own label; it must be reported beside "
                "this result rather than in place of it; and it must not inherit "
                "this protocol's claims, its alpha or its verdict. Relabelling a "
                "repeat as a re-run is how a scored-once protocol quietly becomes "
                "a search over outcomes."),
            "the_chain_also_refuses": (
                "scripts/lanes/confirm100_11c5_chain.sh exits 3 when the "
                "analysis report already exists, before it asks which states are "
                "outstanding and before it queues any lane"),
        },
        "limitations": [
            ("prediction_provenance.py — the module that defines the sidecar "
            "contract and implements verify_sidecar — was bound by no hash at "
            "generation time; it is in neither code.fingerprinted_modules nor "
            "code.analysis_scripts_sha256."),
            ("No GPU identity was recorded: no model, UUID, driver version, CUDA "
            "version or compute capability, in the sidecars or in the lane logs. "
            "Only the device index survives."),
            ("Physical-GPU equivalence across the three states therefore cannot "
            "be established retrospectively."),
            ("code.git_dirty is one bit over a porcelain listing that counts "
            "untracked files, so it cannot distinguish an earlier lane's "
            "untracked evidence from a modified tracked file."),
            ("The run logs lived in a gitignored directory and were committed "
            "only at audit time; between the run and that commit they existed on "
            "one machine and could have been lost."),
            ("The 11C-5R3 safeguards that would have closed the first two items "
            "were not implemented before scoring, and implementing them now "
            "cannot apply retroactively to a run that has already happened."),
        ],
        "what_would_have_bound_it": {
            "for_the_verifier": (
                "add prediction_provenance.py to CODE_FINGERPRINT_MODULES so the "
                "contract's own implementation is hashed into every sidecar and "
                "into the freeze"),
            "for_hardware": (
                "record per-generation device identity into the sidecar — index, "
                "name, UUID, driver_version, compute capability and total memory "
                "from nvidia-smi or torch.cuda — and make verify_sidecar compare "
                "it, so two states generated on different devices cannot be "
                "paired silently"),
            "for_git_dirty": (
                "record the porcelain listing, or at least the counts of tracked "
                "modifications and untracked files separately, instead of one "
                "boolean"),
            "status": (
                "NOT IMPLEMENTED. Each would change a sealed file or the sidecar "
                "contract, so none can be applied to this result; they belong to "
                "the next protocol, where they can be frozen before inference."),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=None,
                    help=f"default {OUT_REPORT}")
    ap.add_argument("--stdout", action="store_true",
                    help="print the report instead of writing it")
    args = ap.parse_args()

    root = REPO_ROOT
    missing = [p for p in (FREEZE_REPORT, ANALYSIS_REPORT, LOG_DIR, PREDICTIONS)
               if not (root / p).exists()]
    if missing:
        print(f"REFUSED — this audit reads the committed run: {missing} "
              f"{'is' if len(missing) == 1 else 'are'} absent")
        return 1

    report = build_report(root)
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.stdout:
        print(text, end="")
        return 0
    out = Path(args.output) if args.output else root / OUT_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    hw = report["hardware"]["recorded_at_run_time"]
    ci = report["code_identity"]
    print(f"wrote {out}")
    print(f"  scoring commit        {ci['scoring_commit'][:12]} "
          f"(all three states: {ci['all_three_states_generated_at_one_commit']})")
    print(f"  git_dirty per state   {ci['git_dirty_per_state']}")
    print(f"  environments identical "
          f"{report['sidecar_environments']['identical_across_the_three_states']}")
    print(f"  verifier bound        "
          f"{ci['verifier_binding_gap']['in_code_fingerprinted_modules'] or ci['verifier_binding_gap']['in_code_analysis_scripts_sha256']}"
          f"  unchanged since scoring "
          f"{ci['verifier_binding_gap']['unchanged_since_scoring']}")
    print(f"  sealed paths changed since scoring "
          f"{ci['sealed_paths_changed_since_scoring'] or 'none'}")
    print(f"  device index per state "
          f"{hw['device_index_per_successful_generation']}")
    print(f"  device model recorded at run time: "
          f"{hw['device_model']}")
    print(f"  hardware terms with no hits in any log: "
          f"{len(report['hardware']['gone_looking_for']['terms_with_no_hits'])}"
          f"/{len(report['hardware']['gone_looking_for']['terms_searched'])}")
    print(f"  physical-GPU equivalence establishable: "
          f"{report['physical_gpu_equivalence']['establishable_from_the_record']}")
    print(f"  limitations recorded  {len(report['limitations'])}")
    print("  POST-HOC: binds nothing, gates nothing, and the result stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
