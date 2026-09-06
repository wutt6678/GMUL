#!/usr/bin/env bash
# Iteration 11R Phase R5 — regenerate every prediction and report on the
# repaired pilot100_v2 dataset.
#
#   bash scripts/lanes/pilot100_11r_chain.sh
#
# Sequence (dependencies, not a fixed device plan — the box is shared, so
# every phase claims a GPU through wait_for_gpu.sh):
#
#   c3  reference-state gate, BASE/MF/MG/MN over all 6,777 queries  ┐
#   d3  selection, MG + 16 candidates over train+val (4,518)        ┘ parallel
#   e1  frozen-test evaluation, BASE/MF/MG/MN + the selected candidates
#   prov  provenance record re-bound to the regenerated reports
#
# e1 needs BOTH c3 and d3: it reads the gate's all-split parquets to
# quantify the batch-layout noise floor and the selection report to know
# which candidates to score.  If either failed, this stops and says which
# rather than assembling a report from half-regenerated evidence.
#
# Every phase writes provenance sidecars beside its parquets, so a re-run
# resumes by VERIFIED reuse instead of regenerating — with one deliberate
# exception: MG is generated inside the train+val layout for d3 rather than
# filtered down from the gate's all-split file, because batched greedy
# decoding is not bit-stable across batch compositions and D_G ranks
# candidates by distance to MG.
#
# Resumption matters here, not just for convenience: this box is shared and
# a co-tenant growing into our headroom OOMs a lane at a model-load
# boundary (it happened to c3 between BASE and MF). d3 and e1 verify and
# reuse completed states on their own; c3 needs --skip-existing to do the
# same, which is only safe because that flag now means "reuse iff the
# sidecar matches this run's adapter, dataset, configuration and code" —
# before Iteration 11R it meant "trust the filename", and passing it would
# have been how a stale parquet got reported as current.
#
# Logs: outputs/lanes/pilot100_11r_{c3,d3,e1,prov,chain}.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/scratch/wutiantong/miniconda3/envs/gmul/bin/python}"
MIN_FREE="${MIN_FREE:-22000}"
BATCH="${BATCH:-8}"
IMAGE_BATCH="${IMAGE_BATCH:-8}"
LOGDIR="$REPO_ROOT/outputs/lanes"
LOG="$LOGDIR/pilot100_11r_chain.log"
cd "$REPO_ROOT" || exit 2
mkdir -p "$LOGDIR"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# A phase whose report is already current for pilot100_v2 is skipped.  The
# per-state sidecar reuse below covers a phase interrupted MID-way; this
# covers a phase that FINISHED, which is what lets the chain be relaunched
# after a later phase fails — or after a lane is deliberately killed so its
# waiter re-queues onto a quieter GPU (a lane that claimed a contended
# device does not re-poll until it fails, and at ~1 q/s instead of ~3 the
# remaining states cost hours each).
report_current() {
  "$PY" - "$1" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    ok = json.loads(p.read_text()).get("dataset_version") == "pilot100_v2"
except Exception:
    ok = False
print("yes" if ok else "no")
PYEOF
}

# ---- preconditions: fail before spending GPU hours, not during ----------
version="$("$PY" - <<'PYEOF'
import json, pathlib
p = pathlib.Path("data/mllmu_hier_pilot100/manifest.json")
print(json.loads(p.read_text())["version"] if p.exists() else "MISSING")
PYEOF
)"
if [ "$version" != "pilot100_v2" ]; then
  say "REFUSING: dataset version is '$version', expected pilot100_v2."
  exit 2
fi
for state in MF MG MN; do
  adapter="data/checkpoints/mllmu_pilot100/$state/adapters/adapter_model.safetensors"
  if [ ! -f "$adapter" ]; then
    say "REFUSING: missing $adapter — R5 regenerates predictions only,"
    say "  it never retrains. Run phase c2 first if the adapters are gone."
    exit 2
  fi
done
say "preconditions ok: dataset=$version, MF/MG/MN adapters present"

stale="$(find data/mllmu_hier_pilot100/predictions -name '*.parquet' \
           ! -name '*.provenance.json' 2>/dev/null | wc -l)"
say "resumable prediction parquets already on disk: $stale"

# ---- c3 in the background, d3 sharded in the foreground ----------------
c3_pid=""; c3_rc=0; d3_rc=0
gate_report="data/reports/mllmu_pilot100_reference_eval.json"
sel_report="data/reports/mllmu_pilot100_unlearning_selection.json"
pred_dir="data/mllmu_hier_pilot100/predictions"

if [ "$(report_current "$gate_report")" = "yes" ]; then
  say "skipping c3: $gate_report is already current for pilot100_v2"
else
  say "launching c3 (gate, all splits)"
  bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$LOGDIR/pilot100_11r_c3.log" \
    "$PY" scripts/evaluate_reference_states.py \
      --tag pilot100 --states BASE,MF,MG,MN --device cuda:0 \
      --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH" \
      --skip-existing &
  c3_pid=$!
fi

# d3 is the long pole: 16 candidates at ~33 min each on a quiet device, so
# one sequential lane leaves the other GPUs idle for hours.  Generation is
# therefore sharded across SHARDS lanes and a single UNSHARDED run
# assembles the report.  Every shard is --generate-only, so no shard can
# write a report computed over a subset of the grid.
#
# SHARDS defaults to the size of the grid, i.e. ONE CANDIDATE PER LANE.
# That granularity is the point, not an excess of processes: a lane does
# not re-poll for a better device once it has claimed one, so a lane stuck
# on a contended GPU holds up everything in its shard.  Measured on this
# box -- 2.36 q/s on a quiet device against 0.87 q/s on a contended one, a
# 2.7x spread -- a 3-way split put 4 candidates on the slow lane and made
# it a ~5.7h bottleneck while the fast lane idled after ~2.1h.  With one
# candidate per lane the GPU queue itself does the load balancing: a slow
# device simply processes fewer candidates, and lanes whose shard is
# already done exit at once instead of holding a lock.
SHARDS="${SHARDS:-16}"

run_d3() {
  if [ "$(report_current "$sel_report")" = "yes" ]; then
    say "skipping d3: $sel_report is already current for pilot100_v2"
    return 0
  fi

  # MG is the reference EVERY shard needs, and _mg_reference_predictions
  # generates it when its parquet is absent -- N shards starting together
  # would all see it missing and all write that one file concurrently.
  # Prime it serially before fanning out.
  if [ ! -f "$pred_dir/predictions_tv_MG.parquet" ]; then
    say "priming the MG reference pass serially: concurrent shards would race on that one file"
    bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" \
      "$LOGDIR/pilot100_11r_d3_prime.log" \
      "$PY" scripts/select_unlearning_checkpoints.py \
        --tag pilot100 --device cuda:0 --batch-size "$BATCH" \
        --image-batch-size "$IMAGE_BATCH" --generate-only --candidates B0
    local prime_rc=$?
    say "MG prime finished rc=$prime_rc"
    if [ "$prime_rc" -ne 0 ]; then
      say "STOPPING: cannot shard without the MG reference predictions"
      return "$prime_rc"
    fi
  fi

  # ONE snapshot, partitioned into N disjoint shards.  Computing each
  # shard's list separately would race the shards already running: a
  # completed candidate shortens the todo list and shifts every later
  # index, so two shards could be handed the same candidate (duplicated
  # work and two writers on one parquet) while another is skipped.
  local shard_ids=()
  mapfile -t shard_ids < <("$PY" - "$SHARDS" <<'PYEOF'
import pathlib, sys
from granunlearn.training.candidate_grid import grid_for_tag
n = int(sys.argv[1])
ck = pathlib.Path("data/checkpoints/mllmu_pilot100_unlearn")
pred = pathlib.Path("data/mllmu_hier_pilot100/predictions")
todo = [s.candidate_id for s in grid_for_tag("pilot100")
        if (ck / s.candidate_id / "adapters").exists()
        and not (pred / f"predictions_tv_{s.candidate_id}.parquet").exists()]
for k in range(n):
    print(",".join(c for i, c in enumerate(todo) if i % n == k))
PYEOF
)

  local total=0 pids=() k ids cnt rc=0 pid
  for k in $(seq 0 $((SHARDS - 1))); do
    ids="${shard_ids[$k]:-}"
    if [ -z "$ids" ]; then
      say "d3 shard $k/$SHARDS: nothing left to generate, not launched"
      continue
    fi
    cnt=$(echo "$ids" | tr ',' '\n' | wc -l)
    total=$((total + cnt))
    say "launching d3 shard $k/$SHARDS ($cnt candidate(s)): $ids"
    bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" \
      "$LOGDIR/pilot100_11r_d3_s$k.log" \
      "$PY" scripts/select_unlearning_checkpoints.py \
        --tag pilot100 --device cuda:0 --batch-size "$BATCH" \
        --image-batch-size "$IMAGE_BATCH" --generate-only \
        --candidates "$ids" &
    pids+=($!)
  done
  if [ "$total" -eq 0 ]; then
    say "every candidate already has predictions; going straight to assembly"
  fi

  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || { rc=1; say "a d3 shard failed (pid $pid)"; }
  done
  if [ "$rc" -ne 0 ]; then
    say "STOPPING: not assembling a selection over partially generated"
    say "  candidates. Re-run; finished shards resume by verified reuse."
    return 1
  fi

  # The assembler is the only run allowed to write the report, and it is
  # unsharded: it sees the whole grid, reuses every parquet the shards
  # produced, and generates only what is somehow still missing.
  say "assembling d3 (reuses every shard's parquet, writes the report)"
  bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$LOGDIR/pilot100_11r_d3.log" \
    "$PY" scripts/select_unlearning_checkpoints.py \
      --tag pilot100 --device cuda:0 --batch-size "$BATCH" \
      --image-batch-size "$IMAGE_BATCH"
  local asm_rc=$?
  say "d3 assembly finished rc=$asm_rc"
  return "$asm_rc"
}

run_d3; d3_rc=$?

if [ -n "$c3_pid" ]; then wait "$c3_pid" || c3_rc=$?; say "c3 finished rc=$c3_rc"; fi
say "d3 finished rc=$d3_rc"

if [ "$c3_rc" -ne 0 ] || [ "$d3_rc" -ne 0 ]; then
  say "STOPPING: c3=$c3_rc d3=$d3_rc. Re-run this script after fixing the"
  say "  failed lane; completed states resume by verified sidecar reuse."
  exit 1
fi

# ---- e1 then prov, strictly after both ---------------------------------
# e1 is never skipped on report currency: it consumes d3's selection, so a
# report left over from a previous selection would look current while
# naming candidates this run did not choose.  It resumes per state instead,
# by verified sidecar reuse.
say "launching e1 (frozen-test evaluation)"
bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$LOGDIR/pilot100_11r_e1.log" \
  "$PY" scripts/evaluate_pilot100_final.py \
    --device cuda:0 --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH"
e1_rc=$?
say "e1 finished rc=$e1_rc"
if [ "$e1_rc" -ne 0 ]; then
  say "STOPPING: not writing provenance over a failed final evaluation —"
  say "  that would bind the record to a report whose numbers are partial."
  exit "$e1_rc"
fi

say "writing the provenance record"
# prov needs a visible CUDA device to record environment.gpu; a CPU-only
# run leaves it null and the evidence test rejects that.
bash scripts/lanes/wait_for_gpu.sh 1 "$LOGDIR/pilot100_11r_prov.log" \
  "$PY" scripts/write_provenance.py --tag pilot100
prov_rc=$?
say "chain complete: c3=$c3_rc d3=$d3_rc e1=$e1_rc prov=$prov_rc"
exit "$prov_rc"
