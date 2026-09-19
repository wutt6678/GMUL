#!/usr/bin/env bash
# ONE training attempt, run by wait_for_gpu.sh inside an already-claimed GPU.
#
#   scripts/lanes/train_attempt.sh <python> <ckpt_root> <id,id,...> <device>
#
# Why this wrapper exists
# -----------------------
# scripts/train_iter12_stage3.py creates <ckpt_root>/<row>/adapters/ BEFORE the
# model is loaded, and skips a row whenever that directory merely exists
# ("adapters exist — skipped").  On this box a co-tenant lands a ~29 GiB job on
# a freshly claimed card within ~40s — measured 2026-09-18/19: GPU 3 was
# claimed with 48505MiB free of a 47.37GiB card, our process reached 18.72 GiB,
# then another user's process took 28.62 GiB and the first backward died
# allocating 20 MiB.  Five further attempts died earlier still, inside
# from_pretrained's caching_allocator_warmup, needing 17.53 GiB contiguous
# while holding only a 426 MiB CUDA context.
#
# Every one of those failures left an EMPTY adapters/ directory behind, so the
# row was skipped forever after: 8 claims, 6 OOMs, all 4 rows poisoned, both
# lanes exiting 0 without training anything, and the chain dying at the
# completeness gate with 0 weight files.  The cleanup therefore has to happen
# per attempt — inside the claimed window, before the trainer starts — and not
# once at launch, because the poisoning happens between attempts.
#
# Safety
# ------
# A directory is removed only when it holds no file at all.  `find -type f`
# decides that, and the removal itself is `find -type d -empty -delete` plus
# rmdir, both of which refuse a non-empty directory, so a trained adapter can
# never be deleted by this script and a completed row is still skipped by the
# trainer — which is what makes a re-run resume rather than repeat.
set -u

PY=${1:?"usage: train_attempt.sh <python> <ckpt_root> <id,id,...> <device>"}
CKPT_ROOT=${2:?"missing checkpoint root"}
IDS=${3:?"missing candidate ids"}
DEVICE=${4:?"missing device"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

IFS=',' read -r -a rows <<< "$IDS"
for row in ${rows[@]+"${rows[@]}"}; do
  [ -n "$row" ] || continue
  dir="$CKPT_ROOT/$row/adapters"
  [ -d "$dir" ] || continue
  if [ -n "$(find "$dir" -type f -print -quit 2>/dev/null)" ]; then
    echo "$(date -Is) keeping $dir: it holds files, so the trainer will skip this row"
    continue
  fi
  #: Empty scaffolding from an attempt that died before writing anything.
  find "$dir" -depth -type d -empty -delete 2>/dev/null
  rmdir "$CKPT_ROOT/$row" 2>/dev/null
  echo "$(date -Is) cleared empty scaffolding at $dir: a previous attempt died"
  echo "$(date -Is)   before writing a single file, and the trainer skips a row"
  echo "$(date -Is)   whose adapters/ exists, so this row would never be retried"
done

exec "$PY" scripts/train_iter12_stage3.py --candidates "$IDS" --device "$DEVICE"
