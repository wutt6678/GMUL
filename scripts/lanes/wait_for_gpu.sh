#!/usr/bin/env bash
# Wait for a GPU with enough FREE memory that no other GMUL lane has
# claimed, then run the given command on it.
#
#   scripts/lanes/wait_for_gpu.sh <min_free_mib> <log_file> <command...>
#
# The box is shared with other tenants, so a lane must not assume a
# device index: it polls nvidia-smi and claims the first GPU that is
# both big enough and unclaimed.  Claims are atomic (mkdir) and live in
# outputs/gpu_locks/<idx>/ (gitignored); the lock is released when the
# command exits.  A lane killed with SIGKILL leaves a stale lock — clear
# it by hand (rmdir outputs/gpu_locks/<idx>) before relaunching.
set -u

MIN_FREE=${1:?"usage: wait_for_gpu.sh <min_free_mib> <log> <cmd...>"}
LOG=${2:?"missing log file"}
shift 2
[ "$#" -gt 0 ] || { echo "no command given" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCKDIR="$REPO_ROOT/outputs/gpu_locks"
mkdir -p "$LOCKDIR" "$(dirname "$LOG")"

echo "$(date -Is) lane waiting for >=${MIN_FREE}MiB free: $*" >> "$LOG"
while true; do
  for idx in 0 1 2 3; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
             -i "$idx" 2>/dev/null | tr -d ' ')
    [ -n "${free:-}" ] || continue
    [ "$free" -ge "$MIN_FREE" ] || continue
    if mkdir "$LOCKDIR/$idx" 2>/dev/null; then
      echo $$ > "$LOCKDIR/$idx/pid"
      echo "$(date -Is) CLAIMED GPU $idx (${free}MiB free): $*" >> "$LOG"
      cd "$REPO_ROOT" || exit 2
      CUDA_VISIBLE_DEVICES="$idx" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$@" >> "$LOG" 2>&1
      rc=$?
      echo "$(date -Is) GPU $idx released (exit $rc)" >> "$LOG"
      rm -f "$LOCKDIR/$idx/pid"
      rmdir "$LOCKDIR/$idx" 2>/dev/null
      exit $rc
    fi
  done
  sleep 60
done
