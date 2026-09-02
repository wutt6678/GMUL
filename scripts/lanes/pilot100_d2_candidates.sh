#!/usr/bin/env bash
# Iteration 11 Phase D2: train the wide B0-B3 candidate grid in N lanes,
# every candidate continuing from the SAME canonical MF adapter.
#
#   bash scripts/lanes/pilot100_d2_candidates.sh          # 3 lanes
#   bash scripts/lanes/pilot100_d2_candidates.sh 4        # 4 lanes
#
# Lane membership is computed by scripts/plan_candidate_lanes.py (LPT bin
# packing over num_epochs x group_size), and candidates that already have
# adapters on disk are excluded automatically, so re-running this script
# resumes rather than repeating work.
#
# Logs: outputs/lanes/pilot100_d2_lane<N>.log
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
MIN_FREE="${MIN_FREE:-26000}"
LANES="${1:-3}"
LOGDIR="$REPO_ROOT/outputs/lanes"
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || exit 2

MF_ADAPTER="data/checkpoints/mllmu_pilot100/MF/adapters/adapter_model.safetensors"
if [ ! -f "$MF_ADAPTER" ]; then
  echo "MF adapter missing ($MF_ADAPTER) — Phase C2 must finish first:" \
       "every B1-B3 candidate continues from that one checkpoint." >&2
  exit 2
fi

i=0
while read -r ids; do
  [ -n "$ids" ] || continue
  case "$ids" in \#*) continue ;; esac
  log="$LOGDIR/pilot100_d2_lane${i}.log"
  setsid bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$log" \
    "$PY" scripts/train_unlearning_baselines.py --tag pilot100 \
      --candidates "$ids" --device cuda:0 \
    < /dev/null > /dev/null 2>&1 &
  echo "launched lane $i (pid $!) -> $log"
  echo "    candidates: $ids"
  i=$((i + 1))
done < <("$PY" scripts/plan_candidate_lanes.py --tag pilot100 \
           --lanes "$LANES" --emit-sh)

echo "lanes running: $(jobs -p | tr '\n' ' ')"
