#!/usr/bin/env bash
# Iteration 11 Phase C2: train MF / MG / MN on the frozen pilot-100 state
# datasets with the IDENTICAL ReferenceRecipe — one lane per state, each
# lane waiting for its own GPU (the box is shared).
#
#   bash scripts/lanes/pilot100_c2_reference_states.sh          # all three
#   bash scripts/lanes/pilot100_c2_reference_states.sh MF MN    # a subset
#
# Logs: outputs/lanes/pilot100_c2_<STATE>.log   (outputs/ is gitignored)
# Only the knowledge dataset differs between the lanes; the recipe (LoRA
# r16/a32, AdamW lr 1e-4, 10 epochs, batch 1 x accum 8, bf16, seed 42,
# 384x384 pixel budget, gradient checkpointing) is inherited verbatim
# from ReferenceRecipe() by every lane.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
MIN_FREE="${MIN_FREE:-26000}"
LOGDIR="$REPO_ROOT/outputs/lanes"
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || exit 2

STATES="${*:-MF MG MN}"

for state in $STATES; do
  log="$LOGDIR/pilot100_c2_${state}.log"
  setsid bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$log" \
    "$PY" scripts/train_reference_states.py \
      --tag pilot100 --states "$state" --device cuda:0 \
    < /dev/null > /dev/null 2>&1 &
  echo "launched $state lane (pid $!) -> $log"
done
echo "lanes running: $(jobs -p | tr '\n' ' ')"
