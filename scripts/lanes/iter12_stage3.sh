#!/usr/bin/env bash
# Iteration 12 Stage 3 — the route-aware B6 successor (B7), and its scoring.
#
#   bash scripts/lanes/iter12_stage3.sh            # 2 lanes
#   bash scripts/lanes/iter12_stage3.sh 4          # 4 lanes
#   PRE_ONLY=1 bash scripts/lanes/iter12_stage3.sh # CPU-only preconditions
#
# Which physical GPU each lane lands on is decided by wait_for_gpu.sh, not by
# this file, and is not overridable: the waiter exports CUDA_VISIBLE_DEVICES to
# the card it claimed, so inside every lane the claimed card IS cuda:0.
#
# Sequence:
#
#   lock    one chain at a time.  A lock whose holder is provably dead is
#           reclaimed; one whose holder is merely UNKNOWN is not, because
#           "no pid file yet" is also what the claim window looks like.
#   pre     all FOUR freezes must still match — Stage 3 inherits its generation
#           contract from Stage 1's, its eight-number MG-anchored floor from
#           Stage 1c's via Stage 2's, and its parent B6 from Stage 2's filed
#           result.  Then the bytes Stage 3 reads rather than regenerates: the
#           B0 and MG parquets, the B6 parquet that both the zero-weight control
#           and the incumbent-relative near-miss envelope are derived from, the
#           MF adapter every row continues from, the image-conditioned reference
#           cache, and the basis and control reports the freeze hashes.
#   ctrl    the zero-weight control, CPU-only and BEFORE any GPU is claimed.
#           It re-derives data/reports/mllmu_iter12_stage3_control.json from the
#           bound B6 bytes and refuses to overwrite a report that re-derives
#           differently.  Unlike Stage 2's control this phase trains nothing and
#           creates no adapter, so it does NOT lock the freeze — the train phase
#           does, at the first B7 adapter.
#   train   4 B7 rows, LPT-packed over lanes.  B0 and the B6 control are REUSED,
#           never retrained.  Every lane re-verifies the freeze and re-derives
#           the control before it trains, so a lane cannot start under a
#           protocol that stopped matching.  Each lane writes its OWN log under
#           outputs/lanes/iter12_stage3_train/, records the CUDA mask it saw,
#           and its exit status is captured individually: `wait` in a loop over
#           pids without reading `$?` per pid discards every lane's status, which
#           is how a chain comes to report success over a lane that died.
#   check   completeness, not existence.  scripts/check_iter12_stage3_adapters.py
#           parses each safetensors header (whole iff the tensor offsets account
#           for the file size, which a truncated write violates), compares the
#           LoRA topology, recipe, group weights, step count, cache provenance
#           and MF init against the incumbent B6, and requires four distinct
#           adapter hashes none of which equals B6's.  Generation is gated on its
#           exit status: a lane OOMed on its last step or killed mid-write leaves
#           a file at adapters/adapter_model.safetensors, and generation is
#           ~4,518 queries per row.
#   gen     one lane per B7 row under the inherited contract, --generate-only,
#           one log per row and one captured exit status per lane.
#   verify  all four prediction parquets on disk.  The score phase is run
#           WITHOUT a GPU claim, exactly as Stage 2's was, and that is only safe
#           because the selector reuses provenance-validated parquets rather
#           than regenerating them; a missing parquet here would mean the score
#           run tried to generate on an unclaimed card, so the chain stops first.
#   score   ONE unsharded run: the reused states and the near-miss envelope are
#           recomputed and gated, the MG-anchored floor decides eligibility, the
#           B0-anchored one is reported beside it, D_G orders the eligible, and
#           the amended decision tree names the Stage-3b parents or closes
#           Iteration 12.  Only an unsharded run may write the report, and the
#           selector refuses to overwrite one that exists.
#
# Nothing here can move the criterion.  The anchor, its values, the epsilon, the
# tie-break, the grid, the near-miss envelope and the Stage-3b tree are all read
# from the freeze; the generation contract is inherited from Stage 1's freeze and
# the selector refuses a run whose batch layout differs from it.  No phase calls
# the freeze script with anything but --check-only: once the first B7 adapter
# exists the protocol cannot be amended again, and this chain does not try.
#
# Lanes are launched without setsid because this chain WAITS on them: setsid can
# fork, leaving `$!` pointing at an intermediate that exits at once, so `wait`
# would return while training still ran and the check phase would report every
# adapter missing.  Detachment belongs to the caller (`nohup setsid ... &`).
#
# Per-run logs are truncated AFTER the lock is claimed, so a second invocation
# cannot wipe the running chain's evidence.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
LANES="${1:-2}"
TRAIN_MIN_FREE="${TRAIN_MIN_FREE:-26000}"
GEN_MIN_FREE="${GEN_MIN_FREE:-22000}"

#: See the Stage-2 chain for why this is not overridable.
DEVICE="cuda:0"

#: Every GPU lane is wrapped so its OWN log records the device it ran on.
#: wait_for_gpu.sh exports CUDA_VISIBLE_DEVICES to the card it claimed and logs
#: the physical index from the outside; this records it from the inside, so
#: "which card produced this adapter" rests on the lane's own view of its mask
#: rather than on one line in a log four lanes append to.  `$0` is the
#: interpreter and `$@` the trainer command, so `exec "$0" "$@"` replaces the
#: wrapper and the lane's pid stays the python's pid.
LANE_PROLOGUE='echo "$(date -Is) lane pid=$$ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} cmd=$*"; exec "$0" "$@"'

#: What each lane's claim and exit line looks like in the chain journal.
report_lane() {
  #: $1 label, $2 rc, $3 log, $4 ids
  local label="$1" rc="$2" log="$3" ids="$4" gpu mask
  gpu="$(grep -o 'CLAIMED GPU [0-9]*' "$log" 2>/dev/null | tail -1)"
  mask="$(grep -o 'CUDA_VISIBLE_DEVICES=[^ ]*' "$log" 2>/dev/null | tail -1)"
  say "  $label exit $rc | ${gpu:-claimed no GPU} | ${mask:-no mask recorded} | $ids"
  say "      log: ${log#$REPO_ROOT/}"
  if [ "$rc" -ne 0 ]; then
    say "      $label FAILED — last 20 lines of its own log:"
    tail -20 "$log" 2>/dev/null | while IFS= read -r line; do
      say "        | $line"
    done
  fi
}

#: Exported, and pinned for the same reason Stage 2 pins it: PEFT serialises
#: ``PeftConfig.target_modules`` from a Python ``set``, so the order in
#: adapter_config.json — and the LoRA injection order behind it — varies with the
#: per-process hash seed.  Pinning it is what makes a re-run of this chain
#: reproduce the same adapter bytes rather than a fresh draw, and B7's whole
#: claim is "B6 plus one coefficient", which is only checkable if B6's recipe is
#: reproduced exactly.  An already-exported value is respected, and either way the
#: chain log records which one was used.
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONHASHSEED

LOGDIR="$REPO_ROOT/outputs/lanes"
LOCKDIR="$REPO_ROOT/outputs/locks/iter12_stage3"
CHAIN="$LOGDIR/iter12_stage3_chain.log"
CTRL_LOG="$LOGDIR/iter12_stage3_ctrl.log"
SCORE_LOG="$LOGDIR/iter12_stage3_score.log"
CHECK_JSON="$LOGDIR/iter12_stage3_adapter_check.json"
#: One directory of per-lane logs per GPU phase, rather than one file every lane
#: appends to.  A shared file interleaves four lanes' tracebacks into something
#: attributable to none of them, and "the training log" then cannot answer which
#: row failed on which card.
TRAIN_LOGDIR="$LOGDIR/iter12_stage3_train"
GEN_LOGDIR="$LOGDIR/iter12_stage3_gen"

STAGE1_PRED="$REPO_ROOT/data/mllmu_hier_pilot100/predictions_iter12"
STAGE2_PRED="$REPO_ROOT/data/mllmu_hier_pilot100/predictions_iter12_stage2"
#: Where the four B7 adapters land.  The completeness gate resolves these paths
#: itself, through stage3_grid's single-owner resolvers; this is the chain's own
#: reference for the pre-flight listing.
STAGE3_CKPT="$REPO_ROOT/data/checkpoints/mllmu_iter12_stage3"
STAGE3_PRED="$REPO_ROOT/data/mllmu_hier_pilot100/predictions_iter12_stage3"
CACHE="$REPO_ROOT/data/mllmu_hier_pilot100/mf_reference_logprobs"
MF_ADAPTERS="$REPO_ROOT/data/checkpoints/mllmu_pilot100/MF/adapters"
B6_ID="B6_beta13.642_lam0.5_lr2e-05_ep5"
SELECTION="data/reports/mllmu_iter12_stage3_selection.json"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR" "$REPO_ROOT/outputs/locks"

say() { echo "$(date -Is) $*" | tee -a "$CHAIN"; }

#: The trained row ids, read from the FROZEN grid rather than listed here: a
#: chain that hardcoded them could train three of four rows and still report
#: itself complete.
trained_ids() {
  "$PY" -c "
import json
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
cal = json.load(open('data/reports/mllmu_iter12_anchor_calibration.json'))
grid = s3g.stage3_grid(cal['grid_rule']['beta_star'])
print('\n'.join(c.candidate_id for c in s3g.trained_rows(grid)))"
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
    #: A pid that exists is not yet a pid that is OURS: numbers are recycled.
    #: An unreadable cmdline is treated as a live holder, because guessing wrong
    #: in that direction costs a refused run while guessing wrong in the other
    #: costs two chains on one adapter directory.
    holder_cmd="$(tr '\0' ' ' < "/proc/$holder/cmdline" 2>/dev/null)"
    if [ -z "$holder_cmd" ]; then
      echo "another iter12 Stage-3 chain holds the lock (pid $holder, cmdline unreadable); stopping." >&2
      exit 2
    fi
    case "$holder_cmd" in
      *iter12_stage3*)
        echo "another iter12 Stage-3 chain is running (pid $holder: $holder_cmd); stopping." >&2
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
  #: and the same card.
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

#: EXIT alone is not enough: a SIGTERM to this chain's pid does not reach its
#: backgrounded lanes, so the shell would die, the EXIT trap would release the
#: lock, and the lanes would carry on training with nothing holding it.  The INT
#: trap is best-effort — a detached `nohup setsid bash ... &` inherits SIGINT set
#: to IGNORE and POSIX forbids trapping a signal the shell was entered with
#: ignored — but it does install for a terminal run, which is how PRE_ONLY dry
#: runs are driven.  Stopping a detached chain is SIGTERM.
trap release_lock EXIT
trap 'on_stop TERM 143' TERM
trap 'on_stop INT 130' INT

say "chain lock claimed (pid $$), $LANES lane(s) requested on $DEVICE, PYTHONHASHSEED=$PYTHONHASHSEED"
mkdir -p "$TRAIN_LOGDIR" "$GEN_LOGDIR"
: > "$CTRL_LOG"; : > "$SCORE_LOG"

#: The selection report is filed once and the selector refuses to overwrite it,
#: so a completed run makes this chain a no-op rather than a failure.
if [ -f "$SELECTION" ]; then
  say "already filed: $SELECTION exists. The selector refuses to overwrite a"
  say "                Stage-3 selection, so there is nothing for this chain to"
  say "                do. Read the report; do not delete it to re-run."
  exit 0
fi

# ── pre ───────────────────────────────────────────────────────────────
say "pre: verifying all four freezes and the bytes Stage 3 reuses"
if ! "$PY" scripts/freeze_iter12_stage3.py --check-only >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-3 freeze does not match the repository. A row trained"
  say "          now could not be scored by the protocol it was trained under,"
  say "          and once an adapter exists the freeze cannot be amended."
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_stage2.py --check-only >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-2 freeze does not match. Stage 3 preserves its"
  say "          criterion exactly and reuses its B6 row as the zero-weight"
  say "          control and as the incumbent the near-miss envelope is drawn from."
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_route_stratification.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1c freeze does not match. The eight-number floor"
  say "          Stage 3 applies is the one that freeze governs."
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_selection_protocol.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1 freeze does not match. Stage 3 INHERITS its"
  say "          generation contract from it and reuses two of its parquets."
  exit 3
fi
if ! "$PY" scripts/build_iter12_stage3_basis.py --check-only >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-3 basis does not re-derive. It is the pre-training"
  say "          record the freeze hashes; a basis that no longer re-derives means"
  say "          the design it describes is not the design in the repository."
  exit 3
fi
for f in "$STAGE1_PRED/predictions_tv_B0.parquet" \
         "$STAGE1_PRED/predictions_tv_MG.parquet" \
         "$STAGE2_PRED/predictions_tv_${B6_ID}.parquet" \
         "$MF_ADAPTERS/adapter_model.safetensors" \
         "$CACHE/logprobs.pt" \
         "$CACHE/sidecar.json" \
         data/reports/mllmu_iter12_stage3_basis.json \
         data/reports/mllmu_iter12_stage3_control.json \
         data/reports/mllmu_iter12_stage3_freeze.json \
         data/reports/mllmu_iter12_anchor_calibration.json \
         data/reports/mllmu_iter12_stage2_selection.json \
         data/reports/mllmu_iter12_route_stratified.json \
         data/reports/mllmu_iter12_retention_probe.json; do
  if [ ! -f "$f" ]; then say "STOPPING: missing $f"; exit 3; fi
done
say "pre: OK — four freezes and the basis match, $(trained_ids | grep -c .) row(s) to train"
if [ "${PRE_ONLY:-0}" = "1" ]; then
  say "PRE_ONLY=1 — stopping before the control; nothing was trained, no adapter"
  say "            exists, and the freeze is therefore still amendable"
  exit 0
fi

# ── ctrl ──────────────────────────────────────────────────────────────
say "ctrl: is additional image-anchor weight 0 really B6?"
say "      (CPU-only; re-derives the filed control report from the bound B6 and"
say "       MG bytes and refuses to overwrite one that re-derives differently)"
#: Backgrounded and waited on rather than run in the foreground, so SIGTERM has a
#: pid to reach.  No GPU claim: this phase loads parquets, not a model.
"$PY" scripts/train_iter12_stage3.py --control >> "$CTRL_LOG" 2>&1 &
ctrl_pid=$!
ACTIVE_LANES=("$ctrl_pid")
wait "$ctrl_pid"; ctrl_rc=$?
ACTIVE_LANES=()
tail -8 "$CTRL_LOG" | tee -a "$CHAIN"
if [ "$ctrl_rc" -ne 0 ]; then
  say "STOPPING (exit $ctrl_rc): the zero-weight control does not hold, so a B7"
  say "          row could not be interpreted as 'B6 plus image weight'. No row may"
  say "          be trained until this is explained."
  exit 5
fi
say "ctrl: OK — gamma=0 is B6, structurally and numerically"

# ── train ─────────────────────────────────────────────────────────────
say "train: planning $LANES lane(s) over the Stage-3 grid"
say "       (the first adapter this phase writes locks the freeze permanently)"
plan="$("$PY" scripts/train_iter12_stage3.py --plan-lanes "$LANES" \
          --emit-sh 2>>"$CHAIN")"
if [ -z "$plan" ]; then
  say "train: nothing to train (every adapter already exists)"
else
  say "$(echo "$plan" | grep -c '^[^#]') lane(s) to launch, one log each under ${TRAIN_LOGDIR#$REPO_ROOT/}"
  echo "$plan" | grep '^#' | while read -r c; do say "  $c"; done
  lane_pids=(); lane_ids=(); lane_logs=()
  n=0
  while read -r ids; do
    [ -n "$ids" ] || continue
    case "$ids" in \#*) continue ;; esac
    n=$((n + 1))
    log="$TRAIN_LOGDIR/train_lane$n.log"
    #: Truncated per lane at launch, which is after the lock was claimed, so a
    #: second invocation cannot wipe the running chain's evidence.
    : > "$log"
    bash scripts/lanes/wait_for_gpu.sh "$TRAIN_MIN_FREE" "$log" \
      bash -c "$LANE_PROLOGUE" \
      "$PY" scripts/train_iter12_stage3.py \
        --candidates "$ids" --device "$DEVICE" \
      < /dev/null > /dev/null 2>&1 &
    pid=$!
    lane_pids+=("$pid"); lane_ids+=("$ids"); lane_logs+=("$log")
    say "  launched training lane $n (pid $pid): $ids"
    say "      log: ${log#$REPO_ROOT/}"
  done <<< "$plan"
  say "train: waiting on ${#lane_pids[@]} lane(s)"
  ACTIVE_LANES=("${lane_pids[@]}")
  train_failed=0
  #: One `wait` per pid with its status captured IMMEDIATELY, so a lane that
  #: died is named here rather than inferred later from a missing file.
  for i in ${lane_pids[@]+"${!lane_pids[@]}"}; do
    wait "${lane_pids[$i]}"; rc=$?
    report_lane "train lane $((i + 1))" "$rc" "${lane_logs[$i]}" "${lane_ids[$i]}"
    [ "$rc" -eq 0 ] || train_failed=$((train_failed + 1))
  done
  ACTIVE_LANES=()
  if [ "$train_failed" -ne 0 ]; then
    say "STOPPING: $train_failed training lane(s) exited non-zero. An adapter"
    say "          written by a lane that then failed is not evidence of a"
    say "          completed run, and no generation lane is launched on the"
    say "          strength of it. Re-run the chain: rows whose adapters exist"
    say "          are skipped, so it resumes rather than repeats."
    exit 4
  fi
  say "train: every lane exited 0"
fi

# ── check ─────────────────────────────────────────────────────────────
#: Existence is not completeness, and this is the last cheap moment to say so:
#: generation costs ~4,518 queries per row.  The gate parses each container
#: header by hand, so a truncated safetensors file is caught without torch and
#: without a GPU, and it compares each row against the incumbent B6 whose
#: topology, recipe and step count a B7 row must reproduce exactly -- B7 moves
#: one coefficient, so a different topology is a different experiment.
say "check: completeness of all $(trained_ids | grep -c .) B7 adapter(s), not mere existence"
if ! "$PY" scripts/check_iter12_stage3_adapters.py --json-out "$CHECK_JSON" \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: at least one adapter is incomplete, so generation does not"
  say "          start. Every failing check is named in the report and in the"
  say "          per-lane logs above."
  say "          report: ${CHECK_JSON#$REPO_ROOT/}"
  exit 4
fi
say "check: OK — every adapter is whole, matches B6's topology and frozen recipe,"
say "       and the four hashes are distinct and differ from B6's"
say "       report: ${CHECK_JSON#$REPO_ROOT/}"

# ── gen ───────────────────────────────────────────────────────────────
#: Reached only through the completeness gate above.
say "gen: one lane per B7 row, under the inherited contract, one log per row"
gen_pids=(); gen_ids=(); gen_logs=()
n=0
while read -r cid; do
  [ -n "$cid" ] || continue
  n=$((n + 1))
  log="$GEN_LOGDIR/gen_${cid}.log"
  : > "$log"
  bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$log" \
    bash -c "$LANE_PROLOGUE" \
    "$PY" scripts/select_iter12_stage3.py \
      --generate-only --candidates "$cid" --device "$DEVICE" \
    < /dev/null > /dev/null 2>&1 &
  pid=$!
  gen_pids+=("$pid"); gen_ids+=("$cid"); gen_logs+=("$log")
  say "  launched generation lane $n (pid $pid): $cid"
  say "      log: ${log#$REPO_ROOT/}"
done < <(trained_ids)
ACTIVE_LANES=("${gen_pids[@]}")
gen_failed=0
for i in ${gen_pids[@]+"${!gen_pids[@]}"}; do
  wait "${gen_pids[$i]}"; rc=$?
  report_lane "gen lane $((i + 1))" "$rc" "${gen_logs[$i]}" "${gen_ids[$i]}"
  [ "$rc" -eq 0 ] || gen_failed=$((gen_failed + 1))
done
ACTIVE_LANES=()
if [ "$gen_failed" -ne 0 ]; then
  say "STOPPING: $gen_failed generation lane(s) exited non-zero. The selector"
  say "          refuses to file a report over a partial grid, so scoring now"
  say "          would only reproduce that refusal after reloading every parquet."
  exit 6
fi
say "gen: every generation lane exited 0"

# ── verify ────────────────────────────────────────────────────────────
#: The score phase runs without a GPU claim, which is only safe because the
#: selector REUSES a parquet whose provenance sidecar matches.  A missing or
#: rejected parquet would make it generate on an unclaimed card, so the chain
#: stops here instead and names the gap.
missing=0
while read -r cid; do
  [ -n "$cid" ] || continue
  if [ ! -f "$STAGE3_PRED/predictions_tv_${cid}.parquet" ]; then
    say "verify: MISSING predictions for $cid"
    missing=$((missing + 1))
  fi
done < <(trained_ids)
if [ "$missing" -ne 0 ]; then
  say "STOPPING: $missing row(s) have no prediction parquet. Re-run the chain"
  say "          (it resumes: adapters that exist are skipped) rather than scoring"
  say "          a subset, which the selector would refuse anyway."
  for f in "$GEN_LOGDIR"/*.log; do
    [ -f "$f" ] || continue
    say "  --- ${f#$REPO_ROOT/} (last 10) ---"
    tail -10 "$f" | while IFS= read -r line; do say "    | $line"; done
  done
  exit 6
fi
say "verify: all four B7 prediction parquets present"

# ── score ─────────────────────────────────────────────────────────────
say "score: ONE unsharded run — gate the reused states and the incumbent"
say "       envelope, apply the MG-anchored eight-number floor, report the"
say "       B0-anchored one beside it, order the eligible by D_G, and record the"
say "       amended decision tree (Stage-3b parents, or Iteration 12 closed)"
#: Backgrounded like the other phases, and for the same reason: this is the run
#: that WRITES the report, so a chain that died mid-score while the selector
#: carried on would leave a report nobody was waiting for and a lock already
#: released for a second chain to claim.
"$PY" scripts/select_iter12_stage3.py --device "$DEVICE" \
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
say "chain complete: $SELECTION"
