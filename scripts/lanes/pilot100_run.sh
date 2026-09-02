#!/usr/bin/env bash
# Iteration 11 pilot-100 driver — NO POLLING.
#
#   bash scripts/lanes/pilot100_run.sh <gpu|auto> <phase>[ <candidate-ids>]
#
# <gpu>   device index (0-3), or "auto" = take the GPU with the most free
#         memory RIGHT NOW (one nvidia-smi query, then run; it never
#         waits or re-checks).
# <phase> c2-mf | c2-mg | c2-mn   train one reference state
#         c3                      BASE/MF/MG/MN gate (pooled + test)
#         d2                      train B0-B3 candidates (see below)
#         d3                      selection on TRAIN+VAL only
#         e1                      one-shot TEST eval + paired CIs
#         prov                    provenance record
# <candidate-ids>  d2 only: comma-separated candidate ids / method
#         letters.  Default = every not-yet-trained candidate, so simply
#         re-running d2 resumes.  Split a grid over several GPUs with:
#             python scripts/plan_candidate_lanes.py --lanes 3 --emit-sh
#         and give one output line to each GPU.
#
# Examples
#   bash scripts/lanes/pilot100_run.sh 3 c2-mf
#   bash scripts/lanes/pilot100_run.sh auto c3
#   bash scripts/lanes/pilot100_run.sh 0 d2 B0,B1
#   bash scripts/lanes/pilot100_run.sh 2 d3
#   bash scripts/lanes/pilot100_run.sh 2 e1
#
# Logs go to outputs/lanes/pilot100_<phase>[_gpu<N>].log (gitignored) and
# the script also streams them to stdout.  Qwen3.5-9B in bf16 needs
# ~22 GiB of its own, so pick a GPU with real headroom; a co-tenant can
# still grow into it (that OOMs the run — just re-run the same command).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
LOGDIR="$REPO_ROOT/outputs/lanes"
BATCH="${BATCH:-8}"          # text-probe batch size for generation
IMAGE_BATCH="${IMAGE_BATCH:-8}"  # image-probe batch size for generation
cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR"

GPU="${1:?usage: pilot100_run.sh <gpu|auto> <phase> [candidate-ids]}"
PHASE="${2:?missing phase}"
IDS="${3:-}"

if [ "$GPU" = "auto" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free \
          --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  echo "auto-selected GPU $GPU ($(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$GPU" | tr -d ' ') MiB free)"
fi

log="$LOGDIR/pilot100_${PHASE}_gpu${GPU}.log"
run() {
  echo "=== [$(date -Is)] gpu$GPU $PHASE: $* ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

case "$PHASE" in
  c2-mf|c2-mg|c2-mn)
    state="${PHASE#c2-}"
    state="$(echo "$state" | tr '[:lower:]' '[:upper:]')"
    run "$PY" scripts/train_reference_states.py \
      --tag pilot100 --states "$state" --device cuda:0
    ;;
  c2)
    run "$PY" scripts/train_reference_states.py \
      --tag pilot100 --states MF,MG,MN --device cuda:0
    ;;
  c3)
    # gate on pooled AND test; predictions persist for reuse by d3/e1
    run "$PY" scripts/evaluate_reference_states.py \
      --tag pilot100 --states BASE,MF,MG,MN --device cuda:0 \
      --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH"
    ;;
  d2)
    MF_ADAPTER="data/checkpoints/mllmu_pilot100/MF/adapters/adapter_model.safetensors"
    if [ ! -f "$MF_ADAPTER" ]; then
      echo "MF adapter missing — run phase c2-mf first (every B1-B3" \
           "candidate continues from that one checkpoint)." | tee -a "$log"
      exit 2
    fi
    if [ -n "$IDS" ]; then
      run "$PY" scripts/train_unlearning_baselines.py \
        --tag pilot100 --candidates "$IDS" --device cuda:0
    else
      run "$PY" scripts/train_unlearning_baselines.py \
        --tag pilot100 --device cuda:0
    fi
    ;;
  d3)
    # TRAIN+VAL only: test queries are not even generated here
    run "$PY" scripts/select_unlearning_checkpoints.py \
      --tag pilot100 --device cuda:0 \
      --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH"
    ;;
  e1)
    # ONE-SHOT frozen-test evaluation + paired CIs (run exactly once)
    run "$PY" scripts/evaluate_pilot100_final.py \
      --device cuda:0 --batch-size "$BATCH" \
      --image-batch-size "$IMAGE_BATCH"
    ;;
  prov)
    run "$PY" scripts/write_provenance.py --tag pilot100
    ;;
  *)
    echo "unknown phase '$PHASE' (c2-mf|c2-mg|c2-mn|c2|c3|d2|d3|e1|prov)"
    exit 2
    ;;
esac
rc=$?
echo "=== [$(date -Is)] gpu$GPU $PHASE finished rc=$rc -> $log ===" | tee -a "$log"
exit $rc
