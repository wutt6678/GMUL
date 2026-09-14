#!/usr/bin/env bash
# Iteration 12 Stage 2 — bounded suppression (B5) and the MF anchor (B6/B6R).
#
#   bash scripts/lanes/iter12_stage2.sh            # 2 lanes
#   bash scripts/lanes/iter12_stage2.sh 4          # 4 lanes
#   PRE_ONLY=1 bash scripts/lanes/iter12_stage2.sh # CPU-only preconditions
#
# Which physical GPU each lane lands on is decided by wait_for_gpu.sh, not by
# this file; see the DEVICE note below for why that is not overridable.
#
# Sequence:
#
#   lock    one chain at a time.  A lock whose holder is provably dead is
#           reclaimed; one whose holder is merely UNKNOWN is not, because
#           "no pid file yet" is also what the claim window looks like.
#   pre     all THREE freezes must still match — Stage 2 inherits its generation
#           contract from Stage 1's and its floor from Stage 1c's, so a mismatch
#           in either is a mismatch in the instrument Stage 2 measures with.
#           Then the bytes Stage 2 reads rather than regenerates: the three
#           reused parquets, the MF adapter every row continues from, the
#           reference cache the anchored rows train against, and the incumbent
#           Stage-1 adapter the control reports its gap against.
#   ctrl    the faithfulness control, ALONE and BEFORE any candidate.  It trains
#           the incumbent Stage-1 objective THREE times in ONE process under ONE
#           pinned PYTHONHASHSEED: A through the frozen train_unlearning, A2
#           through the frozen loop again, B through train_with_preservation
#           with every group in a plain sft/gd mode and neither a cap nor an
#           anchor.  A2 exists to MEASURE the noise floor rather than assume it,
#           and the gate is that gap(A, B) does not exceed gap(A, A2) — bitwise
#           equality outright when the floor is exactly zero.  If it fails,
#           every Stage-2 row would be measured against a baseline produced by
#           different code and the chain stops here: ~9 GPU-hours of training
#           and generation would have bought a comparison nobody can interpret.
#           This phase is also the one that LOCKS the freeze —
#           freeze_iter12_stage2.py refuses to amend the protocol once any
#           adapter exists, the control's included.
#
#           The control does NOT compare against the adapter Stage 1 filed, and
#           the first version of it did.  That criterion was unsatisfiable:
#           PeftConfig.target_modules is a Python set, so the order PEFT
#           serialises it in — and the LoRA injection order behind it, which
#           decides which module draws which dropout mask — varies with the
#           per-process string hash seed.  The filed order is not recoverable,
#           and re-running the FROZEN loop unedited missed those bytes by about
#           as much as the new loop did (1.726e-03 against 1.812e-03 max
#           per-tensor gap).  Both runs are kept under
#           outputs/superseded/iter12_stage2_control_v1/, which says so in full.
#   train   7 rows (4 caps, 1 decoupling, 2 anchor), LPT-packed over lanes.
#           B0, MG and the incumbent are REUSED, not retrained.
#   check   all seven adapters on disk, or stop naming the gaps.
#   gen     one lane per candidate under the inherited contract.
#   score   ONE unsharded run: the reused states are recomputed and gated against
#           the values Stage 1c filed, the MG-anchored floor decides eligibility,
#           the B0-anchored one is reported beside it, and D_G orders the
#           eligible.  Only an unsharded run may write the report — a report over
#           a subset would name a winner the grid does not support while looking
#           complete and carrying the same dataset_version as the real one.
#
# Nothing here can move the criterion.  The anchor, its values, the epsilon, the
# tie-break and the grid are all read from the freeze, the generation contract is
# inherited from Stage 1's freeze and the selector refuses a run whose batch
# layout differs from it, so editing a variable in this file has no path to the
# science.
#
# Lanes are launched without setsid because this chain WAITS on them: setsid can
# fork, leaving `$!` pointing at an intermediate that exits at once, so `wait`
# would return while training still ran and the check phase would report every
# adapter missing.  Detachment belongs to the caller (`nohup setsid ... &`).
# Per-run logs are truncated AFTER the lock is claimed, so a second invocation
# cannot wipe the running chain's evidence.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
LANES="${1:-2}"
TRAIN_MIN_FREE="${TRAIN_MIN_FREE:-26000}"
GEN_MIN_FREE="${GEN_MIN_FREE:-22000}"

#: Not overridable, and deliberately so.  ``wait_for_gpu.sh`` polls nvidia-smi,
#: claims the first GPU that is both big enough and unclaimed by another GMUL
#: lane, and exports CUDA_VISIBLE_DEVICES to it before running the command — so
#: inside every lane the claimed card IS cuda:0.  A DEVICE=cuda:1 override would
#: not select a second card; it would ask for an index that the mask just made
#: nonexistent, and the lane would fail after waiting for a lock it then could
#: not use.  Which physical GPU a lane lands on is the lock's decision, and the
#: log records it.
DEVICE="cuda:0"

#: Exported, and the control refuses to run without it.  ``PeftConfig
#: .target_modules`` is a Python ``set``, and Python randomises the string hash
#: seed per process unless PYTHONHASHSEED is set, so the order PEFT serialises
#: it into ``adapter_config.json`` — and the LoRA injection order behind it,
#: which decides which module draws which dropout mask — differs between two
#: runs of identical code.  Pinning it is what makes the control's three runs
#: comparable at all, and what makes a re-run of this chain reproduce the same
#: adapter bytes rather than a fresh draw.  An already-exported value is
#: respected rather than overwritten, and either way the chain log records which
#: one was used, so an adapter's ``target_modules`` order is attributable.
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONHASHSEED

LOGDIR="$REPO_ROOT/outputs/lanes"
LOCKDIR="$REPO_ROOT/outputs/locks/iter12_stage2"
CHAIN="$LOGDIR/iter12_stage2_chain.log"
CTRL_LOG="$LOGDIR/iter12_stage2_ctrl.log"
TRAIN_LOG="$LOGDIR/iter12_stage2_train.log"
GEN_LOG="$LOGDIR/iter12_stage2_gen.log"
SCORE_LOG="$LOGDIR/iter12_stage2_score.log"

STAGE1_PRED="$REPO_ROOT/data/mllmu_hier_pilot100/predictions_iter12"
STAGE1_CKPT="$REPO_ROOT/data/checkpoints/mllmu_iter12_unlearn"
STAGE2_CKPT="$REPO_ROOT/data/checkpoints/mllmu_iter12_stage2"
CACHE="$REPO_ROOT/data/mllmu_hier_pilot100/mf_reference_logprobs"
MF_ADAPTERS="$REPO_ROOT/data/checkpoints/mllmu_pilot100/MF/adapters"
INCUMBENT="B4_w1.0_lam0.5_lr2e-05_ep5"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR" "$REPO_ROOT/outputs/locks"

say() { echo "$(date -Is) $*" | tee -a "$CHAIN"; }

#: The trained row ids, read from the FROZEN grid rather than listed here: a
#: chain that hardcoded them could train six of seven rows and still report
#: itself complete.
trained_ids() {
  "$PY" -c "
import json
from granunlearn.training import stage2_grid as sg
cal = json.load(open('data/reports/mllmu_iter12_anchor_calibration.json'))
grid = sg.stage2_grid(cal['grid_rule']['beta_star'])
print('\n'.join(c.candidate_id for c in sg.trained_rows(grid)))"
}

# ── lock ──────────────────────────────────────────────────────────────
#: A lock is released only on POSITIVE evidence that its holder is gone.  The
#: absence of a pid file is not that evidence: between `mkdir` and the pid write
#: below there is a window in which the directory exists and holds no pid, and a
#: second chain that read "no pid recorded" as "holder is dead" would rmdir a
#: LIVE lock and then both chains would train into the same directories and
#: claim the same GPUs.  The window is milliseconds wide, so the lock's age
#: settles it — a young lock with no pid is a claim in progress, an old one is a
#: holder that died inside the window.
LOCK_GRACE_SECONDS="${LOCK_GRACE_SECONDS:-60}"

if [ -d "$LOCKDIR" ]; then
  holder="$(cat "$LOCKDIR/pid" 2>/dev/null)"
  if [ -n "$holder" ] && [ -d "/proc/$holder" ]; then
    #: A pid that exists is not yet a pid that is OURS: numbers are recycled,
    #: and a recycled number would otherwise deadlock this chain permanently
    #: behind a process that has nothing to do with it.  An unreadable cmdline
    #: is treated as a live holder, because guessing wrong in that direction
    #: costs a refused run while guessing wrong in the other costs two chains.
    holder_cmd="$(tr '\0' ' ' < "/proc/$holder/cmdline" 2>/dev/null)"
    if [ -z "$holder_cmd" ]; then
      echo "another iter12 Stage-2 chain holds the lock (pid $holder, cmdline unreadable); stopping." >&2
      exit 2
    fi
    case "$holder_cmd" in
      *iter12_stage2*)
        echo "another iter12 Stage-2 chain is running (pid $holder: $holder_cmd); stopping." >&2
        exit 2
        ;;
      *)
        say "reclaiming: pid $holder exists but is not this chain ($holder_cmd)"
        say "            — the number was recycled after a holder died"
        rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null
        ;;
    esac
  elif [ -n "$holder" ]; then
    say "reclaiming a stale lock (pid $holder is not running)"
    rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null
  else
    lock_mtime="$(stat -c %Y "$LOCKDIR" 2>/dev/null || echo 0)"
    lock_age=$(( $(date +%s) - lock_mtime ))
    if [ "$lock_age" -lt "$LOCK_GRACE_SECONDS" ]; then
      echo "the lock directory exists but records no pid, and is ${lock_age}s old — under the ${LOCK_GRACE_SECONDS}s grace, so this is a chain mid-claim rather than a dead one; stopping." >&2
      exit 2
    fi
    say "reclaiming a lock with no pid recorded, ${lock_age}s old (> ${LOCK_GRACE_SECONDS}s grace)"
    rmdir "$LOCKDIR" 2>/dev/null
  fi
fi
mkdir "$LOCKDIR" || { echo "cannot claim the chain lock" >&2; exit 2; }
echo $$ > "$LOCKDIR/pid"

#: The lanes this chain is waiting on RIGHT NOW, not every pid it ever launched.
#: Cleared after each phase's wait loop, because signalling a pid that has
#: already been reaped can reach an unrelated process that inherited the number.
ACTIVE_LANES=()

release_lock() { rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null; }

kill_tree() {
  #: Children before parents.  wait_for_gpu.sh runs the trainer as a child, so
  #: killing the lane's bash alone leaves a python holding a GPU and an adapter
  #: directory open for writing — the exact state a second chain must not find.
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$kid"; done
  kill -TERM "$pid" 2>/dev/null || true
}

on_stop() {
  local sig="$1" code="$2" i alive round
  #: Lanes first, lock LAST.  Releasing the lock while a lane still trains lets
  #: the next chain claim it and put two writers on the same adapter directory
  #: and the same card, and the failure would surface as a corrupted adapter
  #: rather than as a refused lock.
  say "received SIG$sig — stopping ${#ACTIVE_LANES[@]} lane(s) before releasing the lock"
  for i in "${ACTIVE_LANES[@]}"; do kill_tree "$i"; done
  for round in 1 2 3 4 5 6 7 8 9 10; do
    [ "${#ACTIVE_LANES[@]}" -eq 0 ] && break
    alive=0
    for i in "${ACTIVE_LANES[@]}"; do kill -0 "$i" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 2
  done
  say "SIG$sig: lanes stopped; releasing the lock and exiting $code"
  exit "$code"
}

#: EXIT alone is not enough.  It fires when the shell exits, but a SIGTERM to
#: this chain's pid does not reach its backgrounded lanes — they are in the same
#: process group only by accident of how the caller launched them — so the shell
#: would die, the EXIT trap would release the lock, and the lanes would carry on
#: training with nothing holding it.
#:
#: SIGTERM is the operative signal and the INT trap is best-effort, and that is
#: a measured property of how this chain is launched rather than an oversight.
#: A non-interactive bash started in the background inherits SIGINT and SIGQUIT
#: set to IGNORE — `nohup setsid bash ... &` is exactly that — and POSIX forbids
#: a shell from trapping a signal it was entered with ignored, so `trap ... INT`
#: silently does not install.  Verified: `trap -p` under a detached launch shows
#: `trap -- '' SIGINT` beside the installed SIGTERM handler.  The INT trap is
#: kept because it DOES install when the chain is run in a terminal, which is
#: how PRE_ONLY dry runs are driven, and because a handler that cannot install
#: costs nothing.  Stopping a detached chain is SIGTERM.
trap release_lock EXIT
trap 'on_stop TERM 143' TERM
trap 'on_stop INT 130' INT

say "chain lock claimed (pid $$), $LANES lane(s) requested on $DEVICE, PYTHONHASHSEED=$PYTHONHASHSEED"
: > "$CTRL_LOG"; : > "$TRAIN_LOG"; : > "$GEN_LOG"; : > "$SCORE_LOG"

# ── pre ───────────────────────────────────────────────────────────────
say "pre: verifying all three freezes and the bytes Stage 2 reuses"
if ! "$PY" scripts/freeze_iter12_stage2.py --check-only >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-2 freeze does not match the repository. A candidate"
  say "          trained now could not be scored by the protocol it was trained"
  say "          under, and once an adapter exists the freeze cannot be amended."
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_selection_protocol.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1 freeze does not match. Stage 2 INHERITS its"
  say "          generation contract from it and reuses three of its parquets."
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_route_stratification.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1c freeze does not match. The eight-number floor"
  say "          Stage 2 applies is the one that freeze governs."
  exit 3
fi
for f in "$STAGE1_PRED/predictions_tv_B0.parquet" \
         "$STAGE1_PRED/predictions_tv_MG.parquet" \
         "$STAGE1_PRED/predictions_tv_${INCUMBENT}.parquet" \
         "$STAGE1_CKPT/$INCUMBENT/adapters/adapter_model.safetensors" \
         "$MF_ADAPTERS/adapter_model.safetensors" \
         "$CACHE/logprobs.pt" \
         "$CACHE/sidecar.json" \
         data/reports/mllmu_iter12_stage2_basis.json \
         data/reports/mllmu_iter12_anchor_calibration.json \
         data/reports/mllmu_iter12_route_stratified.json; do
  if [ ! -f "$f" ]; then say "STOPPING: missing $f"; exit 3; fi
done
say "pre: OK — three freezes match, $(trained_ids | grep -c .) row(s) to train"
if [ "${PRE_ONLY:-0}" = "1" ]; then
  say "PRE_ONLY=1 — stopping before the control; nothing was trained and the"
  say "            freeze is still amendable"
  exit 0
fi

# ── ctrl ──────────────────────────────────────────────────────────────
say "ctrl: do the two loops compute the same function on the real model?"
say "      (A frozen, A2 frozen again as the measured noise floor, B new;"
say "       PYTHONHASHSEED=$PYTHONHASHSEED, three runs in one process)"
say "      (this phase locks the freeze: no amendment is possible afterwards)"
#: Backgrounded and waited on rather than run in the foreground, so SIGTERM has
#: a pid to reach.  A foreground child is only stopped by a signal to the whole
#: process group, and `kill <chain-pid>` is not that.
bash scripts/lanes/wait_for_gpu.sh "$TRAIN_MIN_FREE" "$CTRL_LOG" \
  "$PY" scripts/train_iter12_stage2.py --control --device "$DEVICE" \
  < /dev/null > /dev/null 2>&1 &
ctrl_pid=$!
ACTIVE_LANES=("$ctrl_pid")
wait "$ctrl_pid"; ctrl_rc=$?
ACTIVE_LANES=()
tail -8 "$CTRL_LOG" | tee -a "$CHAIN"
if [ "$ctrl_rc" -ne 0 ]; then
  say "STOPPING (exit $ctrl_rc): train_with_preservation did not match the"
  say "          frozen loop within the frozen loop's own run-to-run spread, so"
  say "          a difference between Stage 1 and Stage 2 could be a difference"
  say "          between two copies of a training loop rather than between two"
  say "          objectives. No candidate may be trained until this is explained."
  exit 5
fi
say "ctrl: OK — the two loops agree on real bytes, not just on the stub"

# ── train ─────────────────────────────────────────────────────────────
say "train: planning $LANES lane(s) over the Stage-2 grid"
plan="$("$PY" scripts/train_iter12_stage2.py --plan-lanes "$LANES" \
          --emit-sh 2>>"$CHAIN")"
if [ -z "$plan" ]; then
  say "train: nothing to train (every adapter already exists)"
else
  say "$(echo "$plan" | grep -c '^[^#]') lane(s) to launch"
  echo "$plan" | grep '^#' | while read -r c; do say "  $c"; done
  pids=()
  while read -r ids; do
    [ -n "$ids" ] || continue
    case "$ids" in \#*) continue ;; esac
    bash scripts/lanes/wait_for_gpu.sh "$TRAIN_MIN_FREE" "$TRAIN_LOG" \
      "$PY" scripts/train_iter12_stage2.py \
        --candidates "$ids" --device "$DEVICE" \
      < /dev/null > /dev/null 2>&1 &
    pids+=($!)
    say "  launched a training lane (pid ${pids[-1]}): $ids"
  done <<< "$plan"
  say "train: waiting on ${#pids[@]} lane(s)"
  ACTIVE_LANES=("${pids[@]}")
  for p in "${pids[@]}"; do wait "$p"; done
  ACTIVE_LANES=()
fi

# ── check ─────────────────────────────────────────────────────────────
missing=0
while read -r cid; do
  [ -n "$cid" ] || continue
  if [ ! -f "$STAGE2_CKPT/$cid/adapters/adapter_model.safetensors" ]; then
    say "check: MISSING adapter for $cid"
    missing=$((missing + 1))
  fi
done < <(trained_ids)
if [ "$missing" -ne 0 ]; then
  say "STOPPING: $missing grid row(s) have no adapter. The selector refuses to"
  say "          write a report over a partial grid, and generating now would"
  say "          spend GPU hours on a report that cannot be filed."
  tail -25 "$TRAIN_LOG" | tee -a "$CHAIN"
  exit 4
fi
say "check: all seven adapters present"

# ── gen ───────────────────────────────────────────────────────────────
say "gen: one lane per candidate, under the inherited contract"
pids=()
while read -r cid; do
  [ -n "$cid" ] || continue
  bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$GEN_LOG" \
    "$PY" scripts/select_iter12_stage2.py --generate-only \
      --candidates "$cid" --device "$DEVICE" \
    < /dev/null > /dev/null 2>&1 &
  pids+=($!)
  say "  launched a generation lane (pid ${pids[-1]}): $cid"
done < <(trained_ids)
ACTIVE_LANES=("${pids[@]}")
for p in "${pids[@]}"; do wait "$p"; done
ACTIVE_LANES=()
say "gen: all generation lanes finished"

# ── score ─────────────────────────────────────────────────────────────
say "score: ONE unsharded run — gate the reused states, apply the MG-anchored"
say "       floor, report the B0-anchored one beside it, order the eligible by D_G"
#: Backgrounded like the other phases, and for the same reason: this is the run
#: that WRITES the report, so a chain that died mid-score while the selector
#: carried on would leave a report nobody was waiting for and a lock already
#: released for a second chain to claim.
"$PY" scripts/select_iter12_stage2.py --device "$DEVICE" \
    >> "$SCORE_LOG" 2>&1 &
score_pid=$!
ACTIVE_LANES=("$score_pid")
wait "$score_pid"; score_rc=$?
ACTIVE_LANES=()
if [ "$score_rc" -eq 0 ]; then
  say "score: OK"
  tail -16 "$SCORE_LOG" | tee -a "$CHAIN"
else
  say "score: FAILED (exit $score_rc)"
  tail -30 "$SCORE_LOG" | tee -a "$CHAIN"
  exit "$score_rc"
fi
say "chain complete: data/reports/mllmu_iter12_stage2_selection.json"
