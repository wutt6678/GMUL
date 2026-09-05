#!/usr/bin/env bash
# Wait for a GPU with enough FREE memory that no other GMUL lane has
# claimed, then run the given command on it — retrying if a co-tenant
# grows into our headroom mid-run.
#
#   scripts/lanes/wait_for_gpu.sh <min_free_mib> <log_file> <command...>
#
# The box is shared with other tenants, so a lane must not assume a
# device index: it polls nvidia-smi and claims the first GPU that is
# both big enough and unclaimed.  Poll interval is POLL seconds
# (default 1); claims are atomic (mkdir) and live in
# outputs/gpu_locks/<idx>/ (gitignored); the lock is released when the
# command exits.  A lane killed with SIGKILL leaves a stale lock — clear
# it by hand (rmdir outputs/gpu_locks/<idx>) before relaunching.
#
# Retries: Qwen3.5-9B in bf16 needs ~22 GiB of its own, and a co-tenant
# can allocate more AFTER we claim (this happened: claimed at 22.7 GiB
# free, another process grew to 25.1 GiB, first backward OOMed).  A
# non-zero exit therefore re-queues the lane instead of ending it, up to
# MAX_ATTEMPTS, waiting for a strictly larger free-memory margin after
# each OOM so the lane converges on a genuinely quiet GPU.
set -u

MIN_FREE=${1:?"usage: wait_for_gpu.sh <min_free_mib> <log> <cmd...>"}
LOG=${2:?"missing log file"}
shift 2
[ "$#" -gt 0 ] || { echo "no command given" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCKDIR="$REPO_ROOT/outputs/gpu_locks"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
mkdir -p "$LOCKDIR"
mkdir -p "$(dirname "$LOG")"

threshold="$MIN_FREE"
attempt=0
# Seconds between driver queries.  Default 1s: this box is shared with
# several tenants whose own waiters are polling too, so the old 60s
# interval routinely let a freed GPU be claimed by someone else before
# this lane looked again.  Override with POLL=<seconds>.
POLL="${POLL:-1}"
echo "$(date -Is) lane waiting for >=${threshold}MiB free (poll ${POLL}s): $*" >> "$LOG"
while true; do
  # ONE query for every device per poll.  The previous four separate
  # `nvidia-smi -i N` calls cost ~0.4s of driver time each round, which at
  # POLL=1 would make the real interval several times what was asked for.
  mapfile -t frees < <(nvidia-smi --query-gpu=memory.free \
                         --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  idx=-1
  # ${arr[@]+...} because `set -u` rejects an empty array expansion on
  # bash < 4.4, and an nvidia-smi failure leaves this one empty.
  for free in ${frees[@]+"${frees[@]}"}; do
    idx=$((idx + 1))
    [ -n "${free:-}" ] || continue
    [ "$free" -ge "$threshold" ] || continue
    if mkdir "$LOCKDIR/$idx" 2>/dev/null; then
      echo $$ > "$LOCKDIR/$idx/pid"
      attempt=$((attempt + 1))
      echo "$(date -Is) CLAIMED GPU $idx (${free}MiB free, attempt $attempt/$MAX_ATTEMPTS): $*" >> "$LOG"
      cd "$REPO_ROOT" || exit 2
      CUDA_VISIBLE_DEVICES="$idx" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$@" >> "$LOG" 2>&1
      rc=$?
      rm -f "$LOCKDIR/$idx/pid"
      rmdir "$LOCKDIR/$idx" 2>/dev/null
      if [ "$rc" -eq 0 ]; then
        echo "$(date -Is) GPU $idx released (success)" >> "$LOG"
        exit 0
      fi
      echo "$(date -Is) GPU $idx released (exit $rc)" >> "$LOG"
      if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "$(date -Is) giving up after $attempt attempts" >> "$LOG"
        exit "$rc"
      fi
      # demand more headroom next time and let the box settle
      threshold=$((threshold + 2000))
      echo "$(date -Is) re-queueing; new threshold ${threshold}MiB" >> "$LOG"
      sleep 120
      break
    fi
  done
  sleep "$POLL"
done
