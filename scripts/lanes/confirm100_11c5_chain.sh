#!/usr/bin/env bash
# Iteration 11C stage 5 — the one confirmation scoring pass.
#
#   bash scripts/lanes/confirm100_11c5_chain.sh
#
# Sequence:
#
#   pre    fail before spending GPU hours: freeze present and unrefused,
#          split sealed, all three adapters on disk and matching the
#          contract hashes the freeze pins
#   gen    one lane per state (B3, B0, MG), each claiming a GPU through
#          wait_for_gpu.sh, each generating all 1,209 confirmation queries
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
# artifact hashes, generation configuration and code fingerprint is not
# regenerated.  That is what makes an interrupted pass cheap to restart on a
# shared box, and it is safe because "trust the filename" is exactly the
# behaviour Iteration 11R removed.
#
# The three generation lanes run in parallel and the analysis runs only if
# ALL THREE succeeded.  A report assembled from two states would not merely
# be incomplete: its intervals and its Holm step-down would be computed over
# whichever states finished and would look like the real thing.
#
# Logs: outputs/lanes/confirm100_11c5_{chain,B3,B0,MG,gate,analysis}.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
MIN_FREE="${MIN_FREE:-22000}"
LOGDIR="$REPO_ROOT/outputs/lanes"
LOG="$LOGDIR/confirm100_11c5_chain.log"
PRED="$REPO_ROOT/data/mllmu_hier_confirm100/predictions"
ANALYSIS="$REPO_ROOT/data/reports/mllmu_confirm100_final_analysis.json"
cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# ---- preconditions ------------------------------------------------------
# Checked before any lane is queued, because the expensive failure mode is
# discovering at hour three that an adapter is missing or the protocol was
# never frozen.  Every one of these is also refused by the scripts
# themselves; checking here just checks it sooner.
say "preconditions"
"$PY" - <<'PYEOF' || { say "STOPPING: preconditions failed"; exit 2; }
import json, pathlib, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import evaluate_confirmation_split as ecs
from granunlearn.config import _find_repo_root

root = _find_repo_root(pathlib.Path.cwd()) or pathlib.Path.cwd()
freeze = ecs.load_frozen_protocol(root)          # refuses if absent/refused
states = ecs.scored_states(freeze)               # refuses unless exactly 3
ecs.frozen_generation_config(freeze)             # refuses on drift
queries = ecs.confirmation_queries(root, freeze)  # refuses unless sealed
print(f"  freeze ok, states {list(states)}, {len(queries)} queries")
for st in states:
    ecs.adapter_for(st, freeze, root)            # refuses if bytes moved
    print(f"  {st}: adapter present and matching the frozen contract")
PYEOF
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
# Outstanding means "no parquet yet".  A state that has one is still
# VERIFIED by the gate below, so skipping its lane cannot let a stale or
# mismatched file through — it only avoids spending a GPU hour on it.
mapfile -t todo < <(
  for st in B3 B0 MG; do
    [ -f "$PRED/predictions_${st}.parquet" ] || echo "$st"
  done
)
say "states still needing generation: ${#todo[@]} (${todo[*]:-none})"

gen_pids=()
for st in ${todo[@]+"${todo[@]}"}; do
  say "  gen lane: $st"
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
  >> "$LOGDIR/confirm100_11c5_gate.log" 2>&1
gate_rc=$?
tail -n 20 "$LOGDIR/confirm100_11c5_gate.log" | tee -a "$LOG"
if [ "$gate_rc" -ne 0 ] || [ "$gen_rc" -ne 0 ]; then
  say "STOPPING: not assembling the analysis."
  say "  Sealed-split invariant 6 — partial state results are not inspected"
  say "  before every scored state has finished. Re-run this script:"
  say "  completed states resume through their verified sidecars."
  exit 4
fi

# ---- analysis: no GPU, one process -------------------------------------
say "analysis: assembling from three verified prediction files"
"$PY" scripts/analyze_confirmation_split.py \
  >> "$LOGDIR/confirm100_11c5_analysis.log" 2>&1
anal_rc=$?
tail -n 25 "$LOGDIR/confirm100_11c5_analysis.log" | tee -a "$LOG"
if [ "$anal_rc" -ne 0 ]; then
  say "STOPPING: the analysis refused (rc=$anal_rc). See"
  say "  $LOGDIR/confirm100_11c5_analysis.log"
  say "  The predictions are on disk and verified, so re-running the"
  say "  analysis alone is enough once the refusal is addressed."
  exit 5
fi

say "stage 5 complete: $ANALYSIS"
say "  This is the confirmation result. Invariant 7: a failed primary claim"
say "  is reported as failed, and nothing downstream re-tunes on it."
