"""Iteration 11C-5R — the confirmation chain, tested by RUNNING it.

``scripts/lanes/confirm100_11c5_chain.sh`` is the only thing that scores the
confirmation, and both of the review's findings about it were about CONTROL
FLOW: a lane that could not succeed, and a chain that decided what to run from
a filename.  Neither is visible in the scorer's unit tests, because neither
lives in the scorer — so this file runs the chain itself.

How a GPU chain is run on a CPU box
-----------------------------------
``PY`` is read from the environment, so the chain is handed a STUB interpreter
that plays every mode it invokes: the preconditions heredoc, ``--verify-only``,
``--list-outstanding``, ``--state X`` and the analysis.  Each stub mode reads
its exit status and its output from a file, so a test can fail any one of them
without the others changing.  ``nvidia-smi`` is stubbed the same way, through
``PATH``, which is what lets ``wait_for_gpu.sh`` claim a device that is not
there.

The tree the chain runs in is a throwaway: the two lane scripts are COPIED into
it and nothing else exists, because the chain derives ``REPO_ROOT`` from its own
location.  That matters twice over — the chain's logs, its lock and the GPU
locks all land in the temporary tree, so a test cannot leave a lock behind that
would make the real chain wait, and no test needs a committed dataset, an
adapter or a photograph.  Everything here therefore runs in a bare clone and in
CI, and nothing here can write to the real repository.

What is deliberately NOT stubbed
--------------------------------
The preconditions program is recorded verbatim rather than executed: which
checks it asks for is a property of its TEXT, and running it would need the
repository this tree stands in for.  Two tests read that text instead, one of
them against the real ``evaluate_confirmation_split`` module, so a preflight
that called a function the scorer no longer has would fail here rather than
three GPU claims into a real run.
"""

from __future__ import annotations

import ast
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_confirmation_split as ecs  # noqa: E402

CHAIN = REPO_ROOT / "scripts" / "lanes" / "confirm100_11c5_chain.sh"
WAITER = REPO_ROOT / "scripts" / "lanes" / "wait_for_gpu.sh"
STATES = ("B3", "B0", "MG")

LOGDIR = Path("outputs") / "lanes"
CHAIN_LOG = LOGDIR / "confirm100_11c5_chain.log"
PREFLIGHT_LOG = LOGDIR / "confirm100_11c5_preflight.log"
GATE_LOG = LOGDIR / "confirm100_11c5_gate.log"
ANALYSIS_LOG = LOGDIR / "confirm100_11c5_analysis.log"
CHAIN_LOCK = LOGDIR / "confirm100_11c5_chain.lock"
ANALYSIS = Path("data") / "reports" / "mllmu_confirm100_final_analysis.json"
PREDICTIONS = Path("data") / "mllmu_hier_confirm100" / "predictions"

#: Stands in for the conda interpreter.  Every mode the chain invokes reads its
#: behaviour from ``$STUB_CTL`` so a test can fail one step on its own.
_STUB_PY = r'''#!/usr/bin/env bash
set -u
CTL="$STUB_CTL"
printf '%s\n' "$*" >> "$CTL/calls.txt"

if [ "${1:-}" = "-" ]; then
  # The chain feeds its preconditions program on stdin.  Recorded verbatim and
  # NOT executed: what is assertable about it is which checks it asks for.
  cat > "$CTL/preflight_program.py"
  if [ -f "$CTL/preflight_refusal.txt" ]; then
    cat "$CTL/preflight_refusal.txt"
  else
    printf '%s\n' "STUB-PREFLIGHT-OK freeze, code hashes, photographs, adapters"
    printf '%s\n' "  snapshot the lanes will load: /pinned/snapshot"
  fi
  exit "${STUB_PREFLIGHT_RC:-0}"
fi

script="${1##*/}"
shift
case "$script" in
  evaluate_confirmation_split.py)
    mode="" ; state=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --list-outstanding) mode="list" ;;
        --verify-only)      mode="verify" ;;
        --state)            mode="state" ; state="${2:-}" ;;
      esac
      shift
    done
    case "$mode" in
      list)
        n=$(( $(cat "$CTL/list_calls" 2>/dev/null || echo 0) + 1 ))
        printf '%s\n' "$n" > "$CTL/list_calls"
        # The second and later answers come from their own file, because the
        # chain asks again after a lane failed and the honest answer then is
        # usually not the one it got before generation.  Its exit status can
        # differ too: a test has to be able to fail the RE-ASK alone.
        rc="${STUB_LIST_RC:-0}"
        if [ "$n" -ge 2 ]; then
          if [ -n "${STUB_LIST_RC_AFTER:-}" ]; then rc="$STUB_LIST_RC_AFTER"; fi
          if [ -f "$CTL/outstanding_after.txt" ]; then
            cat "$CTL/outstanding_after.txt"
            printf '%s\n' "STUB-LIST-REFUSAL-CHANNEL call $n" >&2
            exit "$rc"
          fi
        fi
        cat "$CTL/outstanding.txt" 2>/dev/null || true
        printf '%s\n' "STUB-LIST-REFUSAL-CHANNEL call $n" >&2
        exit "$rc" ;;
      verify)
        printf '%s\n' "STUB-GATE-OUTPUT all states verified"
        exit "${STUB_GATE_RC:-0}" ;;
      state)
        printf '%s\n' "$state" >> "$CTL/lanes.txt"
        printf '%s\n' "$$" > "$CTL/lane_pid_$state"
        if [ -f "$CTL/lane_sleep" ]; then
          # exec, so the pid this lane just recorded IS the sleeper: a test that
          # has to clean up kills one process rather than a tree, and nothing is
          # left to outlive the run.
          exec sleep "$(cat "$CTL/lane_sleep")"
        fi
        printf '%s\n' "STUB-LANE $state generated and self-verified"
        exit "$(cat "$CTL/rc_$state" 2>/dev/null || echo 0)" ;;
      *)
        printf '%s\n' "stub: no mode in $*" >&2 ; exit 98 ;;
    esac ;;
  analyze_confirmation_split.py)
    printf '%s\n' "STUB-ANALYSIS-OUTPUT"
    if [ "${STUB_ANAL_RC:-0}" -eq 0 ] && [ -f "$CTL/write_analysis" ]; then
      mkdir -p data/reports
      printf '{}\n' > data/reports/mllmu_confirm100_final_analysis.json
    fi
    exit "${STUB_ANAL_RC:-0}" ;;
  *)
    printf '%s\n' "stub: unhandled script $script" >&2 ; exit 99 ;;
esac
'''

#: One line of free MiB per device, which is all the waiter asks the driver for.
_STUB_SMI = r'''#!/usr/bin/env bash
cat "${STUB_CTL}/gpus.txt"
'''


def _wait_for(predicate, timeout: float, what: str) -> None:
    """Poll instead of sleeping a fixed guess: the lanes are launched in the
    background, so what a test waits for is a process reaching a state, not a
    duration."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"{what} did not happen within {timeout}s")


def _running(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _dead_pid() -> int:
    """A pid that is not running, so a stale-lock test cannot pass by luck.

    Scanned rather than hardcoded: a large literal is a pid the runner may
    legitimately be using, and a small one is a pid it may legitimately reuse.
    """
    pid = os.getpid()
    for offset in range(1, 5000):
        candidate = pid + offset * 977
        if not Path(f"/proc/{candidate}").exists():
            return candidate
    raise AssertionError("no unused pid found, so /proc is not readable here")


def _preflight_program_in_source() -> str:
    """The preconditions program exactly as the chain file holds it."""
    lines = CHAIN.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.rstrip().endswith("<<'PYEOF'")) + 1
    end = next(i for i, line in enumerate(lines) if line == "PYEOF")
    return "\n".join(lines[start:end])


class Rig:
    """A throwaway repository holding the two lane scripts and nothing else."""

    def __init__(self, root: Path):
        self.repo = root / "repo"
        self.ctl = root / "ctl"
        (self.repo / "scripts" / "lanes").mkdir(parents=True)
        for script in (CHAIN, WAITER):
            shutil.copy(script, self.repo / "scripts" / "lanes" / script.name)
        (self.ctl / "bin").mkdir(parents=True)
        (self.ctl / "home").mkdir()
        self._executable(self.ctl / "stub_py.sh", _STUB_PY)
        self._executable(self.ctl / "bin" / "nvidia-smi", _STUB_SMI)
        self.set_gpus(3)
        self.set_outstanding(*STATES)
        self.write_analysis()

    @staticmethod
    def _executable(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    # ---- what the stubbed scripts do ------------------------------------
    def set_gpus(self, count: int) -> None:
        (self.ctl / "gpus.txt").write_text("24000\n" * count)

    def set_outstanding(self, *states: str) -> None:
        (self.ctl / "outstanding.txt").write_text(
            "".join(f"{s}\n" for s in states))

    def set_outstanding_after(self, *states: str) -> None:
        (self.ctl / "outstanding_after.txt").write_text(
            "".join(f"{s}\n" for s in states))

    def set_lane_rc(self, state: str, rc: int) -> None:
        (self.ctl / f"rc_{state}").write_text(f"{rc}\n")

    def set_lane_sleep(self, seconds: int) -> None:
        """Make a lane still be running when the test looks at it."""
        (self.ctl / "lane_sleep").write_text(f"{seconds}\n")

    def lane_pid(self, state: str) -> int:
        return int((self.ctl / f"lane_pid_{state}").read_text().strip())

    def preflight_refuses(self, message: str) -> None:
        (self.ctl / "preflight_refusal.txt").write_text(message + "\n")

    def write_analysis(self) -> None:
        (self.ctl / "write_analysis").write_text("yes\n")

    def dont_write_analysis(self) -> None:
        (self.ctl / "write_analysis").unlink(missing_ok=True)

    # ---- what the run left behind ---------------------------------------
    def read(self, relative: Path | str) -> str:
        path = self.repo / relative
        return path.read_text() if path.exists() else ""

    @property
    def chain_log(self) -> str:
        return self.read(CHAIN_LOG)

    def lanes(self) -> list[str]:
        """Which states a lane was run for, in the order they FINISHED.

        The lanes are launched in parallel and each records itself, so the order
        of this list is the box's, not the chain's; every assertion below sorts
        it.  What is orderable is ``calls()``, which the chain's own sequencing
        is asserted on.
        """
        path = self.ctl / "lanes.txt"
        return path.read_text().split() if path.exists() else []

    def analysis_exists(self) -> bool:
        return (self.repo / ANALYSIS).exists()

    def lock_held(self) -> bool:
        return (self.repo / CHAIN_LOCK).exists()

    def preflight_program(self) -> str:
        return (self.ctl / "preflight_program.py").read_text()

    def calls(self) -> list[str]:
        path = self.ctl / "calls.txt"
        return path.read_text().splitlines() if path.exists() else []

    def list_calls(self) -> int:
        path = self.ctl / "list_calls"
        return int(path.read_text().strip()) if path.exists() else 0

    def hold_chain_lock(self, pid: int | None) -> None:
        (self.repo / CHAIN_LOCK).mkdir(parents=True)
        if pid is not None:
            (self.repo / CHAIN_LOCK / "pid").write_text(f"{pid}\n")

    def hold_gpu_lock(self, index: int, pid: int | None) -> None:
        lock = self.repo / "outputs" / "gpu_locks" / str(index)
        lock.mkdir(parents=True)
        if pid is not None:
            (lock / "pid").write_text(f"{pid}\n")

    # ---- running --------------------------------------------------------
    def _env(self, **overrides: object) -> dict[str, str]:
        env = {
            #: The stub driver first, so a real nvidia-smi on this box cannot
            #: answer for a device the test does not control.
            "PATH": f"{self.ctl / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "PY": str(self.ctl / "stub_py.sh"),
            "STUB_CTL": str(self.ctl),
            "HOME": str(self.ctl / "home"),
            "POLL": "1",
            #: One attempt, because the retry policy is written for OOMs: a
            #: nonzero exit re-queues the lane up to eight times with a 120s
            #: settle between them, so a stubbed lane failure would otherwise
            #: take a quarter of an hour to report the status the test asked
            #: for.  The retry path itself is not exercised here for the same
            #: reason — one retry costs two minutes by design.
            "MAX_ATTEMPTS": "1",
            "STUB_PREFLIGHT_RC": "0",
            "STUB_LIST_RC": "0",
            "STUB_GATE_RC": "0",
            "STUB_ANAL_RC": "0",
        }
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def run_chain(self, **overrides: object) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / "lanes" / CHAIN.name)],
            cwd=self.repo, env=self._env(**overrides),
            capture_output=True, text=True, timeout=60)

    def start_chain(self, **overrides: object) -> subprocess.Popen:
        """The chain as a live process, for the tests that terminate it.

        In its own session, so a test that has to clean up can signal the whole
        group — chain, waiters and lanes — without any chance of reaching the
        pytest process that started it.
        """
        return subprocess.Popen(
            ["bash", str(self.repo / "scripts" / "lanes" / CHAIN.name)],
            cwd=self.repo, env=self._env(**overrides),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
            start_new_session=True)

    def run_waiter(self, log_name: str, *command: str,
                   timeout: int = 20) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / "lanes" / WAITER.name),
             "22000", str(self.repo / LOGDIR / log_name), *command],
            cwd=self.repo, env=self._env(),
            capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


class TestTheAdvertisedSingleChain:
    """Finding 1, at the level the finding was stated at: the chain.

    The scorer's own repair — ``--state X`` exits on X's merits — is tested in
    ``test_iteration11c5_scoring.py``.  What was left untested is the claim the
    review quoted back: that ONE invocation of this chain generates all three
    states, gates them and assembles the analysis.
    """

    def test_one_invocation_generates_gates_and_assembles(self, rig):
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(rig.lanes()) == sorted(STATES)
        assert rig.analysis_exists()
        assert "stage 5 complete" in rig.chain_log
        assert "STUB-GATE-OUTPUT" in rig.read(GATE_LOG)
        #: And in that order: every lane before the gate, the gate before the
        #: analysis.  A chain that gated first would be reporting on states it
        #: had not generated yet.
        calls = rig.calls()
        lane_ix = [i for i, c in enumerate(calls) if "--state" in c]
        gate_ix = [i for i, c in enumerate(calls) if "--verify-only" in c]
        anal_ix = [i for i, c in enumerate(calls) if "analyze_" in c]
        assert lane_ix and gate_ix and anal_ix, calls
        assert max(lane_ix) < min(gate_ix) < min(anal_ix)

    @pytest.mark.parametrize("stub_gate_rc", [0, 1])
    def test_the_chain_lock_is_released_when_it_ends(self, rig,
                                                     stub_gate_rc):
        result = rig.run_chain(STUB_GATE_RC=stub_gate_rc)
        assert result.returncode == (0 if stub_gate_rc == 0 else 4)
        #: A trap that kept the lock, or one that changed the exit status while
        #: releasing it, would each break the documented re-run recovery.
        assert not rig.lock_held()


class TestTheChainDecidesWorkByVerificationNotByFilename:
    """Finding 5 and the resumption defect, at the chain level.

    The chain used to build its todo list with ``[ -f predictions_$st.parquet ]
    || echo $st``.  A state whose parquet existed but whose sidecar or
    generation-order record was stale was therefore SKIPPED, the gate then
    refused it, and the chain stopped with the offending file still in place
    and no step that would ever rewrite it.
    """

    def test_parquets_on_disk_do_not_stop_lanes_being_queued(self, rig):
        """THE regression.  All three files exist — which is what the old
        filename test looked at — and all three lanes still run, because the
        todo list came from --list-outstanding instead."""
        (rig.repo / PREDICTIONS).mkdir(parents=True)
        for state in STATES:
            (rig.repo / PREDICTIONS / f"predictions_{state}.parquet").write_text(
                "not a parquet, and the chain must not care")
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(rig.lanes()) == sorted(STATES)

    def test_only_the_states_it_was_told_about_get_a_lane(self, rig):
        rig.set_outstanding("MG")
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.lanes() == ["MG"]

    def test_nothing_outstanding_still_reaches_the_gate(self, rig):
        """A resumed run has no lanes to queue and must not stop before the
        gate: the gate is what makes the resumed evidence complete."""
        rig.set_outstanding()
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.lanes() == []
        assert rig.analysis_exists()
        assert "0 (none)" in rig.chain_log

    def test_a_refusing_todo_list_stops_before_any_lane(self, rig):
        result = rig.run_chain(STUB_LIST_RC=1)
        assert result.returncode == 2
        assert rig.lanes() == []
        #: Its refusals went to the journal, not to a terminal nobody watches.
        assert "STUB-LIST-REFUSAL-CHANNEL" in rig.chain_log

    def test_the_chain_never_names_a_prediction_file(self):
        """Structural, so the finding cannot come back by rewording: a chain
        that never spells a prediction filename cannot test one for existence.
        Not vacuous — the same scan finds the dataset file the preconditions
        program does read."""
        text = CHAIN.read_text()
        assert "predictions_" not in text
        assert "associations.parquet" in text


class TestTheGateIsTheCompletenessAuthority:
    """Finding 1's residue: ``gate_rc != 0 || gen_rc != 0`` stopped the chain.

    A lane can exit nonzero AFTER its state was sealed — killed on the way out,
    or re-queued past MAX_ATTEMPTS once the parquet and its sidecar were already
    on disk.  The evidence is then complete, and a chain that obeyed gen_rc
    would refuse to assemble it, needing a second invocation to discover there
    was nothing left to do.  That is the same defect in a second place, so the
    anomaly is now investigated instead of obeyed.
    """

    def test_a_lane_that_failed_after_verifying_does_not_stop_the_chain(
            self, rig):
        rig.set_lane_rc("B0", 1)
        rig.set_outstanding_after()          # nothing missing any more
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert rig.analysis_exists()
        assert "WARNING: a generation lane exited nonzero" in rig.chain_log
        #: Proceeding is a measurement, not an assumption: the same question the
        #: todo list came from was asked again after the gate.
        assert rig.list_calls() == 2
        assert "left no" in rig.chain_log and "missing evidence" in rig.chain_log

    def test_a_lane_failure_that_left_work_outstanding_stops(self, rig):
        rig.set_lane_rc("B0", 1)
        rig.set_outstanding_after("B0")      # the gate and the todo disagree
        result = rig.run_chain()
        assert result.returncode == 4
        assert not rig.analysis_exists()
        assert "disagree" in rig.chain_log

    def test_a_failed_gate_stops_even_when_every_lane_succeeded(self, rig):
        result = rig.run_chain(STUB_GATE_RC=1)
        assert result.returncode == 4
        assert not rig.analysis_exists()
        assert "STOPPING: not assembling the analysis" in rig.chain_log
        assert "invariant 6" in rig.chain_log

    def test_the_gate_runs_even_when_a_lane_failed(self, rig):
        """Its output is the precise statement of what is missing, so skipping
        it on a lane failure would throw away the only diagnosis."""
        rig.set_lane_rc("MG", 1)
        rig.set_outstanding_after("MG")
        result = rig.run_chain(STUB_GATE_RC=1)
        assert result.returncode == 4
        assert "STUB-GATE-OUTPUT" in rig.read(GATE_LOG)

    def test_a_post_gate_todo_list_that_refuses_stops(self, rig):
        """Asking again is only useful if the answer is trustworthy: a second
        refusal means what is on disk is unknown, so the chain does not guess.
        The FIRST answer has to succeed or nothing would ever reach the re-ask,
        which is why the stub takes two statuses."""
        rig.set_lane_rc("B3", 1)
        rig.set_outstanding_after()
        result = rig.run_chain(STUB_LIST_RC_AFTER=1)
        assert result.returncode == 4
        assert rig.list_calls() == 2
        assert not rig.analysis_exists()
        assert "post-gate --list-outstanding failed" in rig.chain_log


class TestOneChainAtATime:
    """The confirmation is scored ONCE, and this chain is what scores it.

    Two chains running at once would ask --list-outstanding the same question,
    queue the same states, and have two processes write the same parquet.  The
    guard on an existing analysis only trips on a COMPLETED run, which is hours
    away, and "re-run this script" is the chain's own documented recovery for
    an interruption — so without a lock the window is the whole pass.
    """

    def test_a_live_lock_refuses_a_second_chain(self, rig):
        rig.hold_chain_lock(os.getpid())
        result = rig.run_chain()
        assert result.returncode == 6
        assert rig.lanes() == []
        assert "another confirmation chain" in rig.chain_log
        #: Refusing must not remove a lock it does not hold.
        assert rig.lock_held()

    def test_a_lock_naming_no_pid_is_not_taken(self, rig):
        """Between another chain's mkdir and its pid write the directory exists
        and names nobody.  Reading that as stale would let two chains in, so it
        is refused and left alone.  The message is part of the assertion:
        falling through to the liveness test with an empty pid asks whether
        ``/proc/`` exists, which it does, so the wrong branch would still refuse
        and still be wrong about why."""
        rig.hold_chain_lock(None)
        result = rig.run_chain()
        assert result.returncode == 6
        assert rig.lanes() == []
        assert "names no pid" in rig.chain_log
        assert rig.lock_held()

    def test_a_lock_whose_holder_is_dead_is_reclaimed(self, rig):
        """The re-run after an interruption is the documented recovery, and an
        interruption is exactly what leaves a lock behind: obeying it would make
        the recovery hang before its first lane."""
        rig.hold_chain_lock(_dead_pid())
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "reclaiming a stale chain lock" in rig.chain_log
        assert sorted(rig.lanes()) == sorted(STATES)
        assert not rig.lock_held()


class TestTerminatingTheChainStopsItsLanes:
    """The other route to two writers of one parquet.

    The lanes are backgrounded, so a SIGTERM to the chain used to orphan them:
    they went on holding their GPUs and going on writing predictions, and the
    re-run this chain documents as its recovery would then queue a second lane
    for a state an orphan was still generating.  Measured while writing these
    tests — four orphaned waiters from a killed run were still polling half an
    hour later, and only stopped because their stubbed driver query had been
    deleted along with the temporary tree.
    """

    def test_sigterm_stops_the_lane_the_chain_queued(self, rig):
        rig.set_outstanding("B3")
        #: Longer than every window below, so a lane that dies on its own cannot
        #: be mistaken for one the trap stopped.
        rig.set_lane_sleep(300)
        chain = rig.start_chain()
        lane_pid = None
        try:
            _wait_for(lambda: (rig.ctl / "lane_pid_B3").exists(), 30,
                      "the lane starting")
            lane_pid = rig.lane_pid("B3")
            assert _running(lane_pid)
            chain.send_signal(signal.SIGTERM)
            assert chain.wait(timeout=30) == 143
            _wait_for(lambda: not _running(lane_pid), 15,
                      "the lane dying with the chain that queued it")
            assert "TERMINATED" in rig.chain_log
            #: The lock goes with it, or the recovery re-run would be refused by
            #: a lock whose holder this test just terminated.
            assert not rig.lock_held()
            #: And a terminated chain has assembled nothing.
            assert not rig.analysis_exists()
        finally:
            if chain.poll() is None:
                chain.kill()
            # Sweep the group only when something is still in it: on the passing
            # path the lane is dead and the group is empty, so there is no pid
            # to signal and none that could have been recycled.
            if lane_pid is not None and _running(lane_pid):
                try:
                    os.killpg(chain.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            chain.wait(timeout=10)


class TestThePreflightIsRecordedAndAsStrongAsTheLanes:
    """The preconditions exist to fail BEFORE GPU hours are spent.

    Two defects: their output reached only the terminal, so an unattended run's
    journal said "preconditions failed" without saying which; and they compared
    the base-model revision by hand, tolerating an unresolvable HF cache ref
    that ``pinned_model_source`` refuses — so the chain could pass its
    preconditions and then fail in all three lanes, three GPU claims later.
    """

    def test_a_preflight_refusal_reaches_the_journal(self, rig):
        rig.preflight_refuses("STUB-PREFLIGHT-MARKER an adapter has moved")
        result = rig.run_chain(STUB_PREFLIGHT_RC=1)
        assert result.returncode == 2
        assert rig.lanes() == []
        assert "STUB-PREFLIGHT-MARKER" in rig.read(PREFLIGHT_LOG)
        assert "STUB-PREFLIGHT-MARKER" in rig.chain_log
        assert "No lane was queued" in rig.chain_log

    def test_a_passing_preflight_is_recorded_too(self, rig):
        """The journal has to show what was verified, not only what failed: an
        unattended run's only surviving account of its preconditions is this."""
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "STUB-PREFLIGHT-OK" in rig.read(PREFLIGHT_LOG)
        assert "snapshot the lanes will load" in rig.read(PREFLIGHT_LOG)
        assert "STUB-PREFLIGHT-OK" in rig.chain_log
        assert "preconditions passed" in rig.chain_log

    def test_the_preflight_resolves_the_pinned_snapshot(self, rig):
        """Asserted on the program the chain actually handed the interpreter,
        not on the file it was read from, so the two cannot drift."""
        rig.run_chain()
        tree = ast.parse(rig.preflight_program())
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        assert "pinned_model_source" in calls, sorted(calls)
        #: The hand-rolled comparison it replaced called base_model_revision
        #: itself and tolerated an unresolvable ref, which is precisely the case
        #: that has to refuse.  Parsed rather than substring-searched: the
        #: program still names ``frozen_base_model_revision``, and a whole-text
        #: scan would report that as the offence.
        names = {node.id for node in ast.walk(tree)
                 if isinstance(node, ast.Name)}
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    for alias in node.names}
        assert "base_model_revision" not in names | imported

    def test_the_preflight_only_calls_what_the_scorer_has(self):
        """The other half: the recorded program is a contract with the real
        module, and a renamed function would otherwise surface three GPU claims
        into a run."""
        tree = ast.parse(_preflight_program_in_source())
        used = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ecs"}
        assert len(used) >= 8, sorted(used)      # not a vacuous scan
        missing = sorted(name for name in used if not hasattr(ecs, name))
        assert not missing, missing


class TestTheLogsDescribeThisRun:
    """Every log but the journal is truncated at the start of a run.

    The chain tees ``tail -n 20`` of the gate log and ``tail -n 25`` of the
    analysis log into the journal.  Both commands append, and the gate prints
    fewer than twenty lines of its own when everything verifies — so on the
    SUCCESS path, the tail the journal kept was mostly a previous run's.
    """

    def test_a_previous_runs_gate_log_is_not_read_as_this_runs(self, rig):
        (rig.repo / LOGDIR).mkdir(parents=True)
        for log in (GATE_LOG, ANALYSIS_LOG, PREFLIGHT_LOG):
            (rig.repo / log).write_text("STALE-FROM-A-PREVIOUS-RUN\n")
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "STALE" not in rig.chain_log
        assert "STALE" not in rig.read(GATE_LOG)
        assert "STALE" not in rig.read(ANALYSIS_LOG)
        assert "STUB-GATE-OUTPUT" in rig.chain_log

    def test_a_previous_runs_lane_log_is_not_read_as_this_runs(self, rig):
        (rig.repo / LOGDIR).mkdir(parents=True)
        (rig.repo / LOGDIR / "confirm100_11c5_B3.log").write_text(
            "STALE-FROM-A-PREVIOUS-RUN\n")
        result = rig.run_chain()
        assert result.returncode == 0, result.stdout + result.stderr
        lane_log = rig.read(LOGDIR / "confirm100_11c5_B3.log")
        assert "STALE" not in lane_log
        #: wait_for_gpu.sh appends, so this run's own claim is in there.
        assert "CLAIMED GPU" in lane_log


class TestTheChainClaimsOnlyWhatItWrote:
    def test_an_analysis_that_writes_nothing_is_not_completion(self, rig):
        """Exit 0 with no report — a moved --output default, a full disk — would
        otherwise be announced as a result that does not exist."""
        rig.dont_write_analysis()
        result = rig.run_chain()
        assert result.returncode == 5
        assert "stage 5 complete" not in rig.chain_log
        assert "wrote no" in rig.chain_log

    def test_a_refusing_analysis_stops_before_any_completion_claim(self, rig):
        result = rig.run_chain(STUB_ANAL_RC=1)
        assert result.returncode == 5
        assert "stage 5 complete" not in rig.chain_log
        assert "the analysis refused" in rig.chain_log

    def test_an_existing_analysis_is_never_overwritten(self, rig):
        """Score-exactly-once, checked before a single lane is queued."""
        (rig.repo / ANALYSIS).parent.mkdir(parents=True)
        (rig.repo / ANALYSIS).write_text('{"already": "scored"}\n')
        result = rig.run_chain()
        assert result.returncode == 3
        assert rig.lanes() == []
        assert (rig.repo / ANALYSIS).read_text() == '{"already": "scored"}\n'
        assert not rig.lock_held()


class TestTheWaiterReclaimsWhatItsHolderAbandoned:
    """``wait_for_gpu.sh`` polled forever on a lock its holder had left.

    The lane script's own recovery instruction is "re-run this script", and an
    interrupted run — SIGKILL, an OOM killer, a reboot — is precisely what
    leaves ``outputs/gpu_locks/<idx>/`` behind.  The re-run then waited on a GPU
    that was free and that nobody owned, with no message, forever.
    """

    def test_a_lock_whose_pid_is_dead_is_reclaimed(self, rig):
        rig.set_gpus(1)
        rig.hold_gpu_lock(0, _dead_pid())
        marker = rig.ctl / "ran.txt"
        result = rig.run_waiter(
            "waiter.log", "bash", "-c", f"echo ran > {marker}")
        assert result.returncode == 0, result.stdout + result.stderr
        assert marker.exists()
        log = rig.read(LOGDIR / "waiter.log")
        assert "reclaiming a stale lock on GPU 0" in log
        #: Reclaiming GPU 0 rather than quietly taking another device is the
        #: point: a wrong index writes a wrong lock and runs on the wrong card.
        assert "CLAIMED GPU 0" in log

    def test_a_lock_whose_holder_is_running_is_never_stolen(self, rig):
        rig.set_gpus(1)
        rig.hold_gpu_lock(0, os.getpid())
        marker = rig.ctl / "ran.txt"
        with pytest.raises(subprocess.TimeoutExpired):
            rig.run_waiter("waiter.log", "bash", "-c", f"echo ran > {marker}",
                           timeout=4)
        assert not marker.exists()
        log = rig.read(LOGDIR / "waiter.log")
        assert "CLAIMED" not in log and "reclaiming" not in log

    def test_a_lock_naming_no_pid_is_not_stolen(self, rig):
        """Between a claimer's mkdir and its pid write the directory names
        nobody, and taking it would put two lanes on one device."""
        rig.set_gpus(1)
        rig.hold_gpu_lock(0, None)
        marker = rig.ctl / "ran.txt"
        with pytest.raises(subprocess.TimeoutExpired):
            rig.run_waiter("waiter.log", "bash", "-c", f"echo ran > {marker}",
                           timeout=4)
        assert not marker.exists()
        assert "CLAIMED" not in rig.read(LOGDIR / "waiter.log")

    def test_a_lock_is_released_when_the_command_ends(self, rig):
        rig.set_gpus(1)
        result = rig.run_waiter("waiter.log", "true")
        assert result.returncode == 0
        assert not (rig.repo / "outputs" / "gpu_locks" / "0").exists()


class TestTheResultIsCommittable:
    """The chain's terminal artifact was ignored by git.

    ``data/reports/*`` is ignored and every other stage report carries an
    explicit negation, but the confirmation analysis did not — so the one file
    the whole sealed protocol exists to produce would have been written by the
    chain and then silently skipped by a plain ``git add -A``, leaving the
    result only on the machine that scored it.
    """

    def test_the_analysis_report_is_negated(self):
        text = (REPO_ROOT / ".gitignore").read_text()
        assert f"!{ANALYSIS}" in text
        #: Not vacuous: the rule that would otherwise swallow it is still there.
        assert "data/reports/*" in text

    def test_every_confirmation_stage_report_is_negated(self):
        """So the analysis is not a one-off exemption from a rule the other
        stages follow."""
        text = (REPO_ROOT / ".gitignore").read_text()
        for report in ("mllmu_pilot100_confirmation_power.json",
                       "mllmu_pilot100_confirmation_freeze.json",
                       "mllmu_confirm100_photograph_selection.json",
                       "mllmu_confirm100_split_report.json",
                       "mllmu_confirm100_final_analysis.json"):
            assert f"!data/reports/{report}" in text, report

    def test_the_negated_path_is_the_path_the_chain_writes(self, rig):
        """Read off the chain rather than restated here, so the negation and the
        artifact it exists for cannot drift apart."""
        line = next(line for line in CHAIN.read_text().splitlines()
                    if line.startswith('ANALYSIS="$REPO_ROOT/'))
        assert line == f'ANALYSIS="$REPO_ROOT/{ANALYSIS}"'
        rig.run_chain()
        assert rig.analysis_exists()
