#!/usr/bin/env bash
# Iteration 12 Stage 1 — train and select the retention-aware B3 successor.
#
#   bash scripts/lanes/iter12_stage1.sh          # 4 lanes
#   bash scripts/lanes/iter12_stage1.sh 2        # 2 lanes
#
# Sequence:
#
#   lock   one chain at a time.  A second invocation would plan the same
#          candidate list from the same grid and put two trainers on one
#          adapter directory.
#   pre    fail before spending GPU hours: the PROTOCOL FREEZE must still
#          match the repository, the fit/probe partition must still re-derive,
#          the pilot-100 MF and MG adapters must exist, and all three iter12
#          group files must be present.  Training under a drifted criterion
#          would produce candidates that the frozen rule never described.
#   train  one lane per bin-packed subset of the grid, each claiming a GPU
#          through wait_for_gpu.sh.  Costs come from the FIT-HALF group files,
#          so the planner is not charging lanes for 387 examples nobody sees.
#   check  all nine adapters on disk.  A missing one is a stop, not a shrug:
#          the selector refuses to write a report over a partial grid, and
#          stopping here says which candidate is missing.
#   prime  MG and B0 generated SERIALLY on one lane.  Every lane calls the MG
#          reference path first, so fanning out cold would put N writers on one
#          parquet — the race the 11R chain already solved by priming.
#   gen    the remaining eight candidates, one lane each, reusing the primed
#          MG file through its verified sidecar.
#   sel    ONE unsharded run: verifies the freeze again, requires the whole
#          grid, applies the retention floor, minimises D_G with the frozen
#          tie-break, writes the report and stages the winner.
#
# No batch size, token budget, epsilon, weight or tie-break appears on any
# command line here.  The generation contract is frozen and the selector
# REFUSES a run whose contract differs from it, so this file cannot drift the
# criterion by editing a variable — which is how the 11R chain parameterised
# generation, and why those variables are deliberately absent.
#
# Resumption: candidates whose adapters exist are excluded by the planner, and
# predictions are reused only when their sidecar verifies.  Re-running an
# interrupted chain therefore resumes rather than repeats, and never selects
# over a partial grid.
#
# Logs, all under outputs/lanes/.  Every log but the chain's own is truncated
# at the start of a run, because the tails teed into the chain log would
# otherwise show a PREVIOUS run's lines.  They are truncated AFTER the lock is
# claimed, so a second invocation cannot wipe the running chain's evidence.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
LANES="${1:-4}"
#: Training needs more headroom than generation: gradients, optimizer state
#: and activations on top of the ~22 GiB the bf16 weights occupy.  Both values
#: are the ones the pilot-100 and confirmation lanes used successfully.
TRAIN_MIN_FREE="${TRAIN_MIN_FREE:-26000}"
GEN_MIN_FREE="${GEN_MIN_FREE:-22000}"

LOGDIR="$REPO_ROOT/outputs/lanes"
LOCKDIR="$REPO_ROOT/outputs/locks/iter12_stage1"
CHAIN="$LOGDIR/iter12_stage1_chain.log"
TRAIN_LOG="$LOGDIR/iter12_train.log"
PRIME_LOG="$LOGDIR/iter12_prime_mg.log"
GEN_LOG="$LOGDIR/iter12_gen.log"
SEL_LOG="$LOGDIR/iter12_select.log"
CKPT_ROOT="$REPO_ROOT/data/checkpoints/mllmu_iter12_unlearn"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR" "$REPO_ROOT/outputs/locks"

say() { echo "$(date -Is) $*" | tee -a "$CHAIN"; }

# ── lock ──────────────────────────────────────────────────────────────
if [ -d "$LOCKDIR" ]; then
  holder="$(cat "$LOCKDIR/pid" 2>/dev/null)"
  if [ -n "$holder" ] && [ -d "/proc/$holder" ]; then
    echo "another iter12 stage-1 chain is running (pid $holder); stopping." >&2
    exit 2
  fi
  #: A lock whose holder is gone is reclaimed, not obeyed: the documented
  #: recovery for an interrupted chain is "re-run it", and a lock that
  #: outlived its holder would make the re-run refuse forever.  /proc is
  #: read rather than `kill -0`, which fails with EPERM on another user's
  #: live process and would read as "dead".
  say "reclaiming a stale lock (pid ${holder:-unknown} is not running)"
  rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null
fi
mkdir "$LOCKDIR" || { echo "cannot claim the chain lock" >&2; exit 2; }
echo $$ > "$LOCKDIR/pid"
trap 'rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null' EXIT
say "chain lock claimed (pid $$), $LANES lane(s) requested"

#: Truncated AFTER the lock claim, never before.
: > "$TRAIN_LOG"; : > "$PRIME_LOG"; : > "$GEN_LOG"; : > "$SEL_LOG"

# ── pre ───────────────────────────────────────────────────────────────
say "pre: verifying the frozen protocol and the partition"
if ! "$PY" scripts/freeze_iter12_selection_protocol.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the frozen protocol does not match the repository. No GPU"
  say "          hour is spent selecting under a criterion nobody froze."
  exit 3
fi
if ! "$PY" scripts/build_iter12_retention_probe.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the fit/probe partition does not re-derive from the pool."
  exit 3
fi
for f in data/checkpoints/mllmu_pilot100/MF/adapters/adapter_model.safetensors \
         data/checkpoints/mllmu_pilot100/MG/adapters/adapter_model.safetensors \
         data/mllmu_hier_pilot100/unlearning_iter12/fine_target.jsonl \
         data/mllmu_hier_pilot100/unlearning_iter12/target_level.jsonl \
         data/mllmu_hier_pilot100/unlearning_iter12/retain.jsonl; do
  if [ ! -f "$f" ]; then say "STOPPING: missing $f"; exit 3; fi
done
say "pre: OK — freeze matches, partition re-derives, MF/MG adapters and the"
say "    three fit-half group files are present"

#: PRE_ONLY=1 stops here.  The preconditions are the part worth checking
#: before committing GPU hours to a saturated box, and they are pure CPU, so
#: this is how the chain is verified without training anything.
if [ "${PRE_ONLY:-0}" = "1" ]; then
  say "PRE_ONLY=1 — stopping before the train phase; nothing was trained"
  exit 0
fi

# ── train ─────────────────────────────────────────────────────────────
say "train: planning $LANES lane(s) over the iter12 grid"
plan="$("$PY" scripts/plan_candidate_lanes.py --tag iter12 --lanes "$LANES" \
          --emit-sh 2>>"$CHAIN")"
if [ -z "$plan" ]; then say "nothing to train"; else
  say "$(echo "$plan" | grep -c '^[^#]') lane(s) to launch"
  pids=()
  while read -r ids; do
    [ -n "$ids" ] || continue
    case "$ids" in \#*) continue ;; esac
    #: No setsid here, unlike the fire-and-forget pilot lanes: this chain
    #: WAITS on its lanes, and setsid can fork, which would leave `$!`
    #: pointing at an intermediate that exits at once.  `wait` would then
    #: return while training was still running and the check phase would
    #: report every adapter missing.  Detachment is the caller's job — launch
    #: this chain with `nohup setsid ... &`, as the stage-5 chain was.
    bash scripts/lanes/wait_for_gpu.sh "$TRAIN_MIN_FREE" "$TRAIN_LOG" \
      "$PY" scripts/train_unlearning_baselines.py --tag iter12 \
        --candidates "$ids" --device cuda:0 \
      < /dev/null > /dev/null 2>&1 &
    pids+=($!)
    say "  launched a training lane (pid $!): $ids"
  done <<< "$plan"
  say "train: waiting on ${#pids[@]} lane(s)"
  for p in "${pids[@]}"; do wait "$p"; done
fi

# ── check ─────────────────────────────────────────────────────────────
missing=0
for cid in $("$PY" -c "
from granunlearn.training.candidate_grid import grid_for_tag
print(' '.join(c.candidate_id for c in grid_for_tag('iter12')))"); do
  if [ ! -f "$CKPT_ROOT/$cid/adapters/adapter_model.safetensors" ]; then
    say "check: MISSING adapter for $cid"
    missing=$((missing + 1))
  fi
done
if [ "$missing" -ne 0 ]; then
  say "STOPPING: $missing candidate(s) have no adapter. The selector refuses"
  say "          to write a report over a partial grid, so stopping here names"
  say "          the gap instead of discovering it after generation."
  say "tail of $TRAIN_LOG:"; tail -25 "$TRAIN_LOG" | tee -a "$CHAIN"
  exit 4
fi
say "check: all nine adapters present"

# ── prime ─────────────────────────────────────────────────────────────
say "prime: generating MG and B0 serially, so no two lanes race on MG"
bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$PRIME_LOG" \
  "$PY" scripts/select_iter12_retention_checkpoints.py --generate-only \
    --candidates B0 --device cuda:0 \
  < /dev/null > /dev/null 2>&1
prime_rc=$?
say "prime: exit $prime_rc (MG and B0 predictions verified-reused or written)"

# ── gen ───────────────────────────────────────────────────────────────
rest="$("$PY" -c "
from granunlearn.training.candidate_grid import grid_for_tag
print('\n'.join(c.candidate_id for c in grid_for_tag('iter12')
                if c.candidate_id != 'B0'))")"
say "gen: fanning out $(echo "$rest" | wc -l) candidate(s), one lane each"
pids=()
while read -r cid; do
  [ -n "$cid" ] || continue
  bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$GEN_LOG" \
    "$PY" scripts/select_iter12_retention_checkpoints.py --generate-only \
      --candidates "$cid" --device cuda:0 \
    < /dev/null > /dev/null 2>&1 &
  pids+=($!)
done <<< "$rest"
for p in "${pids[@]}"; do wait "$p"; done
say "gen: all generation lanes finished"

# ── sel ───────────────────────────────────────────────────────────────
say "sel: one unsharded run — floor, then D_G, then the frozen tie-break"
if "$PY" scripts/select_iter12_retention_checkpoints.py --device cuda:0 \
      >> "$SEL_LOG" 2>&1; then
  say "sel: OK"
  tail -12 "$SEL_LOG" | tee -a "$CHAIN"
else
  rc=$?
  say "sel: FAILED (exit $rc)"
  tail -30 "$SEL_LOG" | tee -a "$CHAIN"
  exit "$rc"
fi
say "chain complete: data/reports/mllmu_iter12_retention_selection.json"
