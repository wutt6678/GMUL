#!/usr/bin/env bash
# Iteration 12 Stage 1b — seed replicates of the two near-miss candidates.
#
#   bash scripts/lanes/iter12_seeds.sh          # 3 lanes
#   bash scripts/lanes/iter12_seeds.sh 2        # 2 lanes
#   PRE_ONLY=1 bash scripts/lanes/iter12_seeds.sh   # CPU-only preconditions
#
# Sequence:
#
#   lock    one chain at a time; a lock whose holder is gone is reclaimed.
#   pre     BOTH freezes must still match, the partition must still re-derive,
#           and the Stage-1 artifacts this study reads must be present.  The
#           anchor comes out of the filed Stage-1 report, so a Stage-1 freeze
#           that no longer matches means the anchor is not the one that was
#           frozen.
#   ctrl    the generation-determinism control, on one lane, BEFORE any
#           training.  B0 trains zero steps and its adapter is a byte copy of
#           MF, so regenerating it under a fixed contract and comparing tests
#           generation alone.  If the four floor numbers move, the anchor is a
#           draw rather than a point and the study stops here: replicating
#           training seeds cannot repair a non-reproducible anchor, and the
#           ~2 GPU-hours of training would have bought nothing.
#   train   6 replicates (2 parents x seeds 43,44,45), LPT-packed over lanes.
#           Seed 42 is REUSED from Stage 1, not retrained.
#   check   all six replicate adapters on disk, or stop naming the gaps.
#   gen     one lane per replicate.
#   score   ONE unsharded run: per-number mean/min/max/sd, the range
#           classification, and the frozen floor_check applied to the means.
#
# Nothing here can move the criterion.  The measurement contract is inherited
# from the Stage-1 freeze and the analyzer REFUSES a run whose batch layout
# differs from it, so editing a variable in this file has no path to the
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
LANES="${1:-3}"
TRAIN_MIN_FREE="${TRAIN_MIN_FREE:-26000}"
GEN_MIN_FREE="${GEN_MIN_FREE:-22000}"

LOGDIR="$REPO_ROOT/outputs/lanes"
LOCKDIR="$REPO_ROOT/outputs/locks/iter12_seeds"
CHAIN="$LOGDIR/iter12_seeds_chain.log"
CTRL_LOG="$LOGDIR/iter12_seeds_ctrl.log"
TRAIN_LOG="$LOGDIR/iter12_seeds_train.log"
GEN_LOG="$LOGDIR/iter12_seeds_gen.log"
SCORE_LOG="$LOGDIR/iter12_seeds_score.log"
SEED_CKPT="$REPO_ROOT/data/checkpoints/mllmu_iter12_seeds"
STAGE1_PRED="$REPO_ROOT/data/mllmu_hier_pilot100/predictions_iter12"
STAGE1_CKPT="$REPO_ROOT/data/checkpoints/mllmu_iter12_unlearn"

cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR" "$REPO_ROOT/outputs/locks"

say() { echo "$(date -Is) $*" | tee -a "$CHAIN"; }

# ── lock ──────────────────────────────────────────────────────────────
if [ -d "$LOCKDIR" ]; then
  holder="$(cat "$LOCKDIR/pid" 2>/dev/null)"
  if [ -n "$holder" ] && [ -d "/proc/$holder" ]; then
    echo "another iter12 seed chain is running (pid $holder); stopping." >&2
    exit 2
  fi
  say "reclaiming a stale lock (pid ${holder:-unknown} is not running)"
  rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null
fi
mkdir "$LOCKDIR" || { echo "cannot claim the chain lock" >&2; exit 2; }
echo $$ > "$LOCKDIR/pid"
trap 'rm -f "$LOCKDIR/pid"; rmdir "$LOCKDIR" 2>/dev/null' EXIT
say "chain lock claimed (pid $$), $LANES lane(s) requested"
: > "$CTRL_LOG"; : > "$TRAIN_LOG"; : > "$GEN_LOG"; : > "$SCORE_LOG"

# ── pre ───────────────────────────────────────────────────────────────
say "pre: verifying both freezes, the partition, and the Stage-1 artifacts"
if ! "$PY" scripts/freeze_iter12_seed_replication.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1b replication freeze does not match the repository"
  exit 3
fi
if ! "$PY" scripts/freeze_iter12_selection_protocol.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the Stage-1 protocol freeze does not match. The anchor is"
  say "          read from the report that freeze governs."
  exit 3
fi
if ! "$PY" scripts/build_iter12_retention_probe.py --check-only \
      >> "$CHAIN" 2>&1; then
  say "STOPPING: the fit/probe partition does not re-derive from the pool"
  exit 3
fi
for f in "$STAGE1_PRED/predictions_tv_B0.parquet" \
         "$STAGE1_PRED/predictions_tv_MG.parquet" \
         "$STAGE1_PRED/predictions_tv_B4_w4.0_lam0.5_lr2e-05_ep5.parquet" \
         "$STAGE1_PRED/predictions_tv_B4_w2.0_lam0.5_lr2e-05_ep3.parquet" \
         "$STAGE1_CKPT/B0/adapters/adapter_model.safetensors" \
         "$STAGE1_CKPT/B4_w4.0_lam0.5_lr2e-05_ep5/adapters/adapter_model.safetensors" \
         "$STAGE1_CKPT/B4_w2.0_lam0.5_lr2e-05_ep3/adapters/adapter_model.safetensors" \
         data/reports/mllmu_iter12_retention_selection.json; do
  if [ ! -f "$f" ]; then say "STOPPING: missing $f"; exit 3; fi
done
say "pre: OK"
if [ "${PRE_ONLY:-0}" = "1" ]; then
  say "PRE_ONLY=1 — stopping before the control; nothing was trained"
  exit 0
fi

# ── ctrl ──────────────────────────────────────────────────────────────
say "ctrl: is generation deterministic under a FIXED contract?"
bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$CTRL_LOG" \
  "$PY" scripts/analyze_iter12_seed_replication.py --control-only \
    --device cuda:0 < /dev/null > /dev/null 2>&1
ctrl_rc=$?
tail -6 "$CTRL_LOG" | tee -a "$CHAIN"
if [ "$ctrl_rc" -ne 0 ]; then
  say "STOPPING (exit $ctrl_rc): the anchor is not reproducible from identical"
  say "          adapter bytes under an identical contract, so it is a draw and"
  say "          not a point. Seed replication cannot repair that."
  exit 5
fi
say "ctrl: OK — the four floor numbers reproduce exactly"

# ── train ─────────────────────────────────────────────────────────────
say "train: planning $LANES lane(s) over the replicates"
plan="$("$PY" scripts/train_iter12_seed_replicates.py --plan-lanes "$LANES" \
          --emit-sh 2>>"$CHAIN")"
if [ -z "$plan" ]; then say "nothing to train"; else
  say "$(echo "$plan" | grep -c '^[^#]') lane(s) to launch"
  pids=()
  while read -r ids; do
    [ -n "$ids" ] || continue
    case "$ids" in \#*) continue ;; esac
    bash scripts/lanes/wait_for_gpu.sh "$TRAIN_MIN_FREE" "$TRAIN_LOG" \
      "$PY" scripts/train_iter12_seed_replicates.py \
        --candidates "$ids" --device cuda:0 \
      < /dev/null > /dev/null 2>&1 &
    pids+=($!)
    say "  launched a training lane (pid ${pids[-1]}): $ids"
  done <<< "$plan"
  say "train: waiting on ${#pids[@]} lane(s)"
  for p in "${pids[@]}"; do wait "$p"; done
fi

# ── check ─────────────────────────────────────────────────────────────
missing=0
for rid in $("$PY" -c "
from granunlearn.training.seed_replication import to_train
print(' '.join(r.replicate_id for r in to_train()))"); do
  if [ ! -f "$SEED_CKPT/$rid/adapters/adapter_model.safetensors" ]; then
    say "check: MISSING adapter for $rid"
    missing=$((missing + 1))
  fi
done
if [ "$missing" -ne 0 ]; then
  say "STOPPING: $missing replicate(s) have no adapter. A mean over whichever"
  say "          replicates finished would look like the mean over all of them."
  tail -25 "$TRAIN_LOG" | tee -a "$CHAIN"
  exit 4
fi
say "check: all six replicate adapters present"

# ── gen ───────────────────────────────────────────────────────────────
say "gen: one lane per replicate"
pids=()
while read -r rid; do
  [ -n "$rid" ] || continue
  bash scripts/lanes/wait_for_gpu.sh "$GEN_MIN_FREE" "$GEN_LOG" \
    "$PY" scripts/analyze_iter12_seed_replication.py --generate-only \
      --candidates "$rid" --device cuda:0 \
    < /dev/null > /dev/null 2>&1 &
  pids+=($!)
done < <("$PY" -c "
from granunlearn.training.seed_replication import to_train
print('\n'.join(r.replicate_id for r in to_train()))")
for p in "${pids[@]}"; do wait "$p"; done
say "gen: all generation lanes finished"

# ── score ─────────────────────────────────────────────────────────────
say "score: one unsharded run — means, ranges, and the frozen floor"
if "$PY" scripts/analyze_iter12_seed_replication.py --device cuda:0 \
      >> "$SCORE_LOG" 2>&1; then
  say "score: OK"
  tail -14 "$SCORE_LOG" | tee -a "$CHAIN"
else
  score_rc=$?
  say "score: FAILED (exit $score_rc)"
  tail -30 "$SCORE_LOG" | tee -a "$CHAIN"
  exit "$score_rc"
fi
say "chain complete: data/reports/mllmu_iter12_seed_replication.json"
