#!/usr/bin/env bash
# Iteration 11C stage 5 — the one confirmation scoring pass.
#
#   bash scripts/lanes/confirm100_11c5_chain.sh
#
# Sequence:
#
#   lock   one chain at a time.  A second invocation while this one is running
#          would ask --list-outstanding the same question, queue the same
#          states, and have two processes write the same parquet
#   pre    fail before spending GPU hours: freeze present and unrefused,
#          split sealed, all three adapters on disk and matching the
#          contract hashes the freeze pins, every frozen analysis-script
#          and paired_ci hash matching the bytes on disk, all 402 pinned
#          photographs re-hashed, every image-route query resolving to a
#          photograph, and the base-model revision the freeze pins RESOLVED
#          to the snapshot directory the lanes will load — which is the check
#          every lane makes, but after it has waited for and claimed a GPU
#   todo   --list-outstanding: which states still need work, decided by
#          VERIFICATION rather than by whether a filename exists
#   gen    one lane per outstanding state, each claiming a GPU through
#          wait_for_gpu.sh, each generating all 1,209 confirmation queries
#          and then verifying ITS OWN state and exiting on that alone
#   gate   evaluate_confirmation_split.py --verify-only: all three
#          sidecars verify and all three were generated over the same
#          query order
#   anal   analyze_confirmation_split.py: no GPU, assembles the report
#
# No batch size, image batch size or token budget is passed on any command
# line here.  All three are frozen in the confirmation freeze and read from
# it by the scripts, so a lane cannot drift the generation configuration by
# editing a variable in this file — which is how the 11R chain parameterised
# them, and why BATCH and IMAGE_BATCH are deliberately absent here.
#
# Resumption is by VERIFIED sidecar reuse, not by filename: a state whose
# parquet verifies against this run's adapter bytes, dataset version and
# artifact hashes, generation configuration, code fingerprint AND its own
# generation-order record is not regenerated.  Deciding the todo list the same
# way is what makes an interrupted pass both cheap to restart and actually
# restartable: a state whose parquet exists but does not verify is regenerated,
# where a filename test would skip it, the gate would then refuse it, and the
# chain would stop with the offending file still in place.
#
# The generation lanes run in parallel and the analysis runs only if the GATE
# passes.  A report assembled from two states would not merely be incomplete:
# its intervals and its Holm step-down would be computed over whichever states
# finished and would look like the real thing.
#
# The gate, and not the lanes' exit statuses, is the completeness authority.  A
# lane can exit nonzero AFTER its state was sealed — killed on the way out, or
# re-queued past MAX_ATTEMPTS by wait_for_gpu.sh once the parquet and its
# sidecar were already on disk — and then the evidence is complete while a
# chain that obeyed gen_rc would refuse to assemble it.  That is finding 1 of
# this repair in a second place, so a nonzero gen_rc is investigated rather
# than obeyed: --list-outstanding is asked again, and the chain goes on only if
# the answer is that nothing is missing.
#
# Logs, all under outputs/lanes/.  Every log but the chain's is truncated at
# the start of a run, because the tails below are teed into the chain log and a
# tail of an accumulated file can show a PREVIOUS run's lines — the gate prints
# fewer than twenty of its own on a successful run, so `tail -n 20` of one it
# appended to would have been mostly history.  The chain log is the journal and
# is the one file that accumulates:
#
#   confirm100_11c5_chain.log        every say(), plus the tails below
#   confirm100_11c5_preflight.log    the preconditions program's own output
#   confirm100_11c5_outstanding.txt  the todo list exactly as it was read
#   confirm100_11c5_<state>.log      one per lane, including its GPU retries
#   confirm100_11c5_gate.log         --verify-only
#   confirm100_11c5_analysis.log     the analysis
#
# Exit codes: 2 preconditions or the todo list, 3 already scored, 4 the gate
# refused or the states on disk disagree with it, 5 the analysis refused or
# wrote nothing, 6 another chain holds the lock.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
MIN_FREE="${MIN_FREE:-22000}"
LOGDIR="$REPO_ROOT/outputs/lanes"
LOG="$LOGDIR/confirm100_11c5_chain.log"
ANALYSIS="$REPO_ROOT/data/reports/mllmu_confirm100_final_analysis.json"
cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

PREFLIGHT="$LOGDIR/confirm100_11c5_preflight.log"
OUTSTANDING="$LOGDIR/confirm100_11c5_outstanding.txt"
OUTSTANDING_JSON="$LOGDIR/confirm100_11c5_outstanding.json"
OUTSTANDING_STDOUT="$LOGDIR/confirm100_11c5_outstanding_stdout.log"
GATE_LOG="$LOGDIR/confirm100_11c5_gate.log"
ANAL_LOG="$LOGDIR/confirm100_11c5_analysis.log"
RUN_LOGS=("$PREFLIGHT" "$OUTSTANDING" "$OUTSTANDING_JSON" \
          "$OUTSTANDING_STDOUT" "$GATE_LOG" "$ANAL_LOG")
# Reset AFTER the chain lock is claimed and not before, which is where this
# line used to be.  A second invocation must leave the ACTIVE run's evidence
# alone: truncating first meant the duplicate was refused correctly but only
# after emptying the preflight, outstanding, gate and analysis logs of the
# chain that was still running - so the operator who went to read why the
# first chain was slow found six empty files and a journal entry from a
# process that had never started.  $LOG itself is appended and never
# truncated, so the refusal is still recorded.

# ---- one chain at a time -------------------------------------------------
# The confirmation is scored ONCE and this script is what scores it, so two of
# them must not run at once: they would queue the same states and two processes
# would write the same parquet, which is a torn file at best and at worst two
# byte-identical writes whose sidecar verifies while nobody can say which
# process produced it.  The analysis-exists guard below only trips on a
# COMPLETED run, which is hours away, and "re-run this script" is this chain's
# own documented recovery for an interruption — so the window is the whole pass.
# mkdir is atomic, which is the same idiom wait_for_gpu.sh uses to claim a
# device.
#
# A lock whose recorded pid is not running is stale and is reclaimed rather than
# obeyed, because an interrupted run is exactly what leaves one behind and a
# lock that outlives its holder would make the documented recovery hang.  A lock
# that names NO pid is left alone: that is the window between another chain's
# mkdir and its pid write, and taking it would let two chains in.  /proc is read
# instead of `kill -0`: kill fails with EPERM on a live process owned by another
# user, which would read as "dead" and steal a held lock.
LOCK="$LOGDIR/confirm100_11c5_chain.lock"
claim_chain_lock() {
  mkdir "$LOCK" 2>/dev/null || return 1
  echo $$ > "$LOCK/pid"
  trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT
}
if ! claim_chain_lock; then
  holder="$(cat "$LOCK/pid" 2>/dev/null)"
  if [ -z "$holder" ]; then
    say "STOPPING: $LOCK exists but names no pid, so it is either being"
    say "  claimed by a chain that started a moment ago or was made by hand."
    say "  Neither is this run's to remove. Wait and retry, or inspect it."
    exit 6
  fi
  if [ -d "/proc/$holder" ]; then
    say "STOPPING: another confirmation chain (pid $holder) holds $LOCK."
    say "  The confirmation is scored once, so a second chain must not queue"
    say "  the same states and write the same parquets. Wait for it, or if it"
    say "  is not really running, remove $LOCK by hand."
    exit 6
  fi
  say "reclaiming a stale chain lock: pid $holder is not running"
  rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null
  claim_chain_lock || { say "STOPPING: cannot claim $LOCK"; exit 6; }
fi
say "chain lock claimed (pid $$)"
# Reset HERE, and only now that this run owns the directory: the reason is at
# the declaration above.
for run_log in "${RUN_LOGS[@]}"; do : > "$run_log"; done
say "  ${#RUN_LOGS[@]} run log(s) reset; they describe this run only"

# ---- terminating this chain has to stop its lanes -----------------------
# The lanes are backgrounded, so a SIGTERM to this script orphans them: they go
# on holding their GPUs and go on writing parquets, and the re-run that is this
# chain's documented recovery would queue a second lane for a state an orphan is
# still generating — the two-writers-one-parquet hazard the lock above exists
# to prevent, arriving through another door.  Not hypothetical: four orphaned
# waiters from a killed run were still polling half an hour later.
#
# Each waiter's own child is signalled as well, because killing wait_for_gpu.sh
# leaves the interpreter it launched running.  SIGKILL cannot be trapped, so a
# `kill -9` still orphans them; what protects the evidence then is that a
# parquet being written by an orphan does not verify, and the gate refuses it.
gen_pids=()
kill_lanes() {
  local pid child
  for pid in ${gen_pids[@]+"${gen_pids[@]}"}; do
    for child in $(pgrep -P "$pid" 2>/dev/null); do
      kill -TERM "$child" 2>/dev/null
    done
    kill -TERM "$pid" 2>/dev/null
  done
}
terminated() {
  say "TERMINATED: signalling ${#gen_pids[@]} lane(s), then releasing the lock"
  kill_lanes
  exit 143
}
trap terminated TERM INT

# ---- preconditions ------------------------------------------------------
# Checked before any lane is queued, because the expensive failure mode is
# discovering at hour three that an adapter is missing or the protocol was
# never frozen.  Every one of these is also refused by the scripts
# themselves; checking here just checks it sooner.
say "preconditions"
# Captured to its own log and teed into the chain log rather than left on the
# terminal: this chain is launched unattended, and a refusal that only ever
# reached a lost tty left the journal saying "preconditions failed" with no
# record of which one.
if ! "$PY" - > "$PREFLIGHT" 2>&1 <<'PYEOF'
import pathlib, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import evaluate_confirmation_split as ecs
from granunlearn.config import _find_repo_root
from granunlearn.evaluation.reference_eval import load_associations_parquet

root = _find_repo_root(pathlib.Path.cwd()) or pathlib.Path.cwd()
freeze = ecs.load_frozen_protocol(root)          # refuses if absent/refused
states = ecs.scored_states(freeze)               # refuses unless exactly 3
ecs.frozen_generation_config(freeze)             # refuses on drift
revision = ecs.frozen_base_model_revision(freeze)  # refuses if the pins disagree
queries = ecs.confirmation_queries(root, freeze)  # refuses unless sealed
by_assoc = {a.association_id: a for a in load_associations_parquet(
    root / ecs.CONFIRM_DATASET_DIR / "associations.parquet")}
model_id = freeze["checkpoints"][states[0]]["recipe"]["model_id"]
print(f"  freeze ok, states {list(states)}, {len(queries)} queries")

# The same runtime verifications the scorer and the analyzer perform,
# run here because this is the cheapest place to fail: an hour into a GPU pass
# is not the time to discover a swapped photograph or an edited paired_ci.py.
problems = ecs.verify_frozen_code(root, freeze)
problems += ecs.verify_confirmation_dataset(root, freeze)
problems += ecs.verify_confirmation_images(root, queries, by_assoc)
if problems:
    raise SystemExit("preflight refusals:\n  " + "\n  ".join(problems))
print("  frozen code hashes, frozen dataset hashes, pinned photographs and "
      "per-query image resolution all verified")

for st in states:
    ecs.adapter_for(st, freeze, root)            # refuses if bytes moved
    print(f"  {st}: adapter present and matching the frozen contract")

# LAST, and what makes this preflight as strong as the lanes it front-runs.
# pinned_model_source is exactly what each lane calls once wait_for_gpu.sh has
# handed it a device: it refuses a MOVED HF cache ref, refuses an UNRESOLVABLE
# one, and resolves the pinned revision to the snapshot directory the weights
# are loaded from.  Comparing the revision by hand here instead — the previous
# form, which also tolerated an unresolvable ref — let the chain pass its
# preconditions and then fail in all three lanes, three GPU claims later, on a
# check that costs one directory listing.
snapshot = ecs.pinned_model_source(model_id, revision)
print(f"  base model {model_id} pinned at revision {revision}")
print(f"  snapshot the lanes will load: {snapshot}")
PYEOF
then
  tail -n 40 "$PREFLIGHT" | tee -a "$LOG"
  say "STOPPING: preconditions failed. The refusals are above and in"
  say "  $PREFLIGHT. No lane was queued and no GPU was claimed."
  exit 2
fi
tail -n 40 "$PREFLIGHT" | tee -a "$LOG"
say "  preconditions passed"

# The analysis is written once.  If it already exists this chain has already
# run, and re-running it would be a second scoring pass over a protocol whose
# whole property is that it is scored exactly once.
if [ -f "$ANALYSIS" ]; then
  say "STOPPING: $ANALYSIS already exists."
  say "  The confirmation is scored ONCE. If that report is wrong, the"
  say "  protocol has to be amended and re-frozen in the repository first,"
  say "  not overwritten by a second pass from this script."
  exit 3
fi

# ---- generation: one lane per outstanding state -------------------------
# Outstanding is decided by --list-outstanding, which VERIFIES: a state is
# outstanding when its sidecar fails against this run's adapter bytes, dataset
# version, artifact hashes, generation configuration and code fingerprint, or
# when its generation-order record is missing or names a different order.
#
# Testing for the FILENAME instead — which is what this chain used to do —
# skips any state that has a parquet, however stale, so an invalid partial pass
# could not be regenerated: the lane would skip it, the gate would then refuse
# it, and the chain would stop with the offending file still in place and no
# step that would ever rewrite it.
#
# The answer is read from a STRUCTURED FILE the evaluator writes, not from its
# stdout.  stdout was the channel and was never safe: setup_logger attaches a
# StreamHandler(sys.stdout) to EVERY logger in the process, so the evaluator
# silencing its own logger left prediction_provenance's writing into the list
# the chain was about to iterate.  One truncated sidecar was enough —
# read_sidecar logs "unreadable sidecar ...", that line arrived where a state
# name was expected, and the chain made a log file named after an error message
# and spent a GPU claim on a lane that could only fail --state's validation.
#
# Written to files rather than captured with ``mapfile < <(cmd)``: in that
# form ``$?`` afterwards is MAPFILE's status, not the command's, so a failed
# --list-outstanding would report success and the chain would queue no lanes.
#
# The reader below imports nothing from this project on purpose.  It is the
# only step between the oracle and a ``--state`` argument, so it must not be
# able to fail by importing something, and it writes the list itself rather
# than printing it into a channel anything else in the process can write to.
ask_outstanding() {   # ask_outstanding: fills $OUTSTANDING_JSON, $OUTSTANDING
  if ! "$PY" scripts/evaluate_confirmation_split.py --list-outstanding \
        --outstanding-report "$OUTSTANDING_JSON" \
        > "$OUTSTANDING_STDOUT" 2>> "$LOG"; then
    say "STOPPING: --list-outstanding failed, so which states still need work"
    say "  is unknown and queueing lanes would be guesswork. Its refusals are"
    say "  in $LOG above this line; the preconditions passed, so this is a"
    say "  real fault rather than a missing artifact."
    return 1
  fi
  if ! "$PY" - "$OUTSTANDING_JSON" "$OUTSTANDING" <<'PYEOF'
import json, sys

report, dest = sys.argv[1], sys.argv[2]
with open(report) as fh:
    doc = json.load(fh)
scored = doc["scored_states"]
outstanding = doc["outstanding_states"]
#: Validated HERE as well as in the shell loop below: two independent spellings
#: of one rule, because the alternative is a string that came out of a file
#: reaching ``--state`` and a filename.  B3, B0 and MG are the only three
#: states this confirmation scores, and scored_states() in the evaluator
#: refuses a freeze naming any other set.
for state in outstanding:
    if state not in ("B3", "B0", "MG") or state not in scored:
        raise SystemExit(
            f"{report} lists {state!r} as outstanding, which is not exactly "
            f"one of B3, B0 or MG among the freeze's scored states {scored}")
with open(dest, "w") as out:
    for state in outstanding:
        out.write(state + "\n")
PYEOF
  then
    say "STOPPING: $OUTSTANDING_JSON could not be read back as a list of the"
    say "  three scored states, so what still needs generation is unknown."
    say "  See $LOG and that file."
    return 1
  fi
  return 0
}

# The shell half of the same rule, at the point where the string becomes an
# argument and a filename.  Deliberately a literal and not a value read from
# the report: this chain queues GPU work, and the set of states it is allowed
# to queue is not something the file it just read should get to widen.  Takes
# the items themselves rather than the name of the array, so what is validated
# is exactly what the loop below iterates.
only_scored_states() {   # only_scored_states <what> [items...]
  local what="$1" item
  shift
  for item in "$@"; do
    case "$item" in
      B3|B0|MG) ;;
      *)
        say "STOPPING: the $what list contains '$item', which is not exactly"
        say "  one of B3, B0 or MG. Queueing it would pass an arbitrary"
        say "  string to --state and to a log filename, so nothing is queued."
        return 1 ;;
    esac
  done
  return 0
}

if ! ask_outstanding; then
  exit 2
fi
mapfile -t todo < "$OUTSTANDING"
only_scored_states "outstanding" ${todo[@]+"${todo[@]}"} || exit 2
say "states still needing generation: ${#todo[@]} (${todo[*]:-none})"

gen_pids=()
for st in ${todo[@]+"${todo[@]}"}; do
  say "  gen lane: $st"
  # Truncated here rather than left to accumulate: wait_for_gpu.sh APPENDS, so
  # this run's retry history is kept, while a previous run's cannot be read as
  # this one's.
  : > "$LOGDIR/confirm100_11c5_${st}.log"
  # --state X generates X, verifies X, and exits on X's own merits. It does NOT
  # check global completeness — that is the gate's job below. The previous
  # version fell through to a global check, so whichever lane finished FIRST
  # exited nonzero merely because the other two were still running, this chain
  # recorded gen_rc=1, and it stopped before its own gate even when that gate
  # would have passed. A sharded lane has to be able to succeed.
  bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" \
    "$LOGDIR/confirm100_11c5_${st}.log" \
    "$PY" scripts/evaluate_confirmation_split.py \
      --device cuda:0 --state "$st" &
  gen_pids+=($!)
done

gen_rc=0
for pid in ${gen_pids[@]+"${gen_pids[@]}"}; do
  wait "$pid" || { gen_rc=1; say "a generation lane failed (pid $pid)"; }
done
say "generation finished rc=$gen_rc"

# ---- the completeness gate ---------------------------------------------
# Runs even when a lane failed, because its output is the precise statement
# of what is missing.  It computes no metric: the script imports no
# aggregation code, so a partial pass cannot leak a rate into a log.
say "gate: verifying all three states"
"$PY" scripts/evaluate_confirmation_split.py --verify-only \
  >> "$GATE_LOG" 2>&1
gate_rc=$?
tail -n 20 "$GATE_LOG" | tee -a "$LOG"
if [ "$gate_rc" -ne 0 ]; then
  say "STOPPING: not assembling the analysis."
  say "  Sealed-split invariant 6 — partial state results are not inspected"
  say "  before every scored state has finished. Re-run this script:"
  say "  completed states resume through their verified sidecars."
  exit 4
fi

if [ "$gen_rc" -ne 0 ]; then
  # A lane exited nonzero although the gate just verified all three states
  # against this run's adapter bytes, dataset version, artifact hashes,
  # generation configuration, code fingerprint and generation-order record.  So
  # the lane failed AFTER its evidence was sealed, and stopping here would
  # leave a complete confirmation unassembled until somebody re-ran the chain
  # to discover there was nothing left to do — which is what this chain did
  # before, on the strength of gen_rc alone.
  #
  # The anomaly is checked rather than obeyed: the same question the todo list
  # came from is asked again, and only "nothing outstanding" lets the chain go
  # on.  Two spellings of verify_state agreeing is what makes proceeding a
  # measurement instead of an assumption, and it also catches the one case
  # where they could differ — something else writing to the predictions
  # directory between the gate and here.
  say "WARNING: a generation lane exited nonzero although the gate passed."
  if ! ask_outstanding; then
    say "  a lane failed AND the post-gate --list-outstanding failed, so what"
    say "  is on disk is unknown."
    exit 4
  fi
  mapfile -t left < "$OUTSTANDING"
  only_scored_states "post-gate outstanding" ${left[@]+"${left[@]}"} || exit 4
  if [ "${#left[@]}" -ne 0 ]; then
    say "STOPPING: a lane failed and ${#left[@]} state(s) are still"
    say "  outstanding (${left[*]}) although the gate passed, so the two"
    say "  answers disagree and the chain will not pick one. Re-run this"
    say "  script; if it disagrees again, the predictions directory is being"
    say "  written by something else."
    exit 4
  fi
  say "  --list-outstanding now reports nothing, so the failed lane left no"
  say "  missing evidence: every state it was asked for is sealed and verified."
  say "  Its log is in $LOGDIR above. If the failure was not benign, delete"
  say "  that state's parquet and re-run to regenerate it under this protocol."
fi

# ---- analysis: no GPU, one process -------------------------------------
say "analysis: assembling from three verified prediction files"
"$PY" scripts/analyze_confirmation_split.py \
  >> "$ANAL_LOG" 2>&1
anal_rc=$?
tail -n 25 "$ANAL_LOG" | tee -a "$LOG"
if [ "$anal_rc" -ne 0 ]; then
  say "STOPPING: the analysis refused (rc=$anal_rc). See $ANAL_LOG"
  say "  The predictions are on disk and verified, so re-running the"
  say "  analysis alone is enough once the refusal is addressed."
  exit 5
fi
# Because "stage 5 complete" is this script's claim, not the analyzer's: an
# exit 0 with no report — an --output default that moved, a full disk — would
# otherwise be announced as a result that does not exist.
if [ ! -f "$ANALYSIS" ]; then
  say "STOPPING: the analysis exited 0 but wrote no $ANALYSIS."
  say "  See $ANAL_LOG. Nothing will be claimed complete that is not on disk."
  exit 5
fi

say "stage 5 complete: $ANALYSIS"
say "  This is the confirmation result. Invariant 7: a failed primary claim"
say "  is reported as failed, and nothing downstream re-tunes on it."
