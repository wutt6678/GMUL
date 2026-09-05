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

# ---- c3 and d3 in parallel ---------------------------------------------
say "launching c3 (gate, all splits) and d3 (selection, train+val)"
bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$LOGDIR/pilot100_11r_c3.log" \
  "$PY" scripts/evaluate_reference_states.py \
    --tag pilot100 --states BASE,MF,MG,MN --device cuda:0 \
    --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH" \
    --skip-existing &
c3_pid=$!
bash scripts/lanes/wait_for_gpu.sh "$MIN_FREE" "$LOGDIR/pilot100_11r_d3.log" \
  "$PY" scripts/select_unlearning_checkpoints.py \
    --tag pilot100 --device cuda:0 \
    --batch-size "$BATCH" --image-batch-size "$IMAGE_BATCH" &
d3_pid=$!

c3_rc=0; d3_rc=0
wait "$c3_pid" || c3_rc=$?
say "c3 finished rc=$c3_rc"
wait "$d3_pid" || d3_rc=$?
say "d3 finished rc=$d3_rc"

if [ "$c3_rc" -ne 0 ] || [ "$d3_rc" -ne 0 ]; then
  say "STOPPING: c3=$c3_rc d3=$d3_rc. Re-run this script after fixing the"
  say "  failed lane; completed states resume by verified sidecar reuse."
  exit 1
fi

# ---- e1 then prov, strictly after both ---------------------------------
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
