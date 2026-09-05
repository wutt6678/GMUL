"""Split-aware photograph assignment (Iteration 11R).

The defect this repairs
-----------------------
Iteration 11 gave EVERY image query ``assoc.images[0]`` — the same
photograph that ``state_datasets`` and ``unlearning_datasets`` train on —
regardless of the query's split.  Measured on the frozen pilot-100 set:
0/6,777 queries deviated from ``images[0]``, 396 of the 496 distinct
images were never referenced, and for 11 of the 30 target species the
training photograph carried a ``val`` (6) or ``test`` (5) label.  So the
image route measured unseen WORDING over a SEEN photograph, while
``ImageRef.split`` and the iNaturalist adapter's documented design
(``I_train -> species`` trained, held-out ``I_test -> species/genus/
family`` evaluated) went unconsumed.

The policy
----------
This module is the SINGLE source of truth for which photograph belongs to
which split, so the query generator, the association builder and the
validators cannot drift apart:

* ``images[0]`` is RESERVED as the training photograph, because that is
  what the training and unlearning datasets use.  Reserving it — rather
  than re-picking a train photograph — is what keeps those six JSONLs
  byte-identical across the repair, so no checkpoint must be retrained;
* the remaining photographs form the val and test pools, disjoint from
  each other and from training.  ``ImageRef.split`` is RELABELED to
  reflect that actual use, overriding the adapter's 60/20/20
  pre-assignment, which was assigned before anyone knew which photograph
  training would consume;
* an entity with ONE photograph — every MLLMU person, whose single
  portrait is shared by its 6-7 associations — cannot have a held-out
  photograph at all.  All of its splits keep the trained portrait and are
  flagged ``image_seen_in_training=True``.  That flag, not the source
  dataset name, is what separates the two reported image strata.

Within-split repetition is unavoidable and is recorded rather than hidden:
a species has 12 photographs, so after reserving one there are 5 val and
6 test photographs for up to 10 image queries per split.  The property the
held-out-photo claim needs is disjointness from TRAINING, not distinctness
between two test queries.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from granunlearn.schema import AssociationRecord, ImageRef, QueryRecord

#: Index of the photograph that training consumes, and that is therefore
#: reserved out of the val/test pools.
RESERVED_TRAINING_INDEX = 0

#: The two image strata every report must separate.
HELD_OUT_PHOTO = "held_out_photo"
SEEN_PHOTO_UNSEEN_WORDING = "seen_photo_unseen_wording"
IMAGE_STRATA = (HELD_OUT_PHOTO, SEEN_PHOTO_UNSEEN_WORDING)

SPLITS = ("train", "val", "test")

#: pilot100_v1 training/unlearning JSONL hashes, pinned from commit
#: ``3850461``.  Reserving ``images[0]`` as the training photograph is what
#: makes the visual-split repair free: the six files every reference state
#: and every unlearning candidate was fitted on must come out of a v2
#: rebuild BYTE-IDENTICAL, so no checkpoint has to be retrained.
#:
#: That is a measured property, not an intention, so it is gated in the
#: build (see :func:`assert_no_training_drift`) and pinned in the tests.
#: Keys are relative to the dataset directory.
V1_TRAINING_JSONL_SHA256 = {
    "training/MF.jsonl":
        "2238a3ba5ceff0a9ea3b0d85e6382ac3f281bf57a2a96737a21ad69aa77f5315",
    "training/MG.jsonl":
        "60ba21692c1be738f34e713f73d55e55e784da7bd4b5032a7b69b9f443de15d6",
    "training/MN.jsonl":
        "4bf7005a5c8a8d6f21b936f6cdbb7e31d32f93f9832a116b9d7352473cc717a2",
    "unlearning/fine_target.jsonl":
        "60ef46ac72d601deb64b55b5a5ddfb31c168983dd2adf5a594bd8ce73e8f6765",
    "unlearning/target_level.jsonl":
        "0c56833660ea701b61de426f2dc2bc4b7823cc5fe59c93840a364bdbd9a0fcbf",
    "unlearning/retain.jsonl":
        "ed2219be266d2d24e5cf0f020d57bb1442086292186db458dff2efceb983ce45",
}

#: The dataset version the repair produced, and the one the gate applies to.
REPAIRED_DATASET_VERSION = "pilot100_v2"


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_training_drift(data_dir, rel_paths, version) -> dict[str, str]:
    """Hard-fail a rebuild that moved a training byte.

    Called by the builders right after they write the state datasets and
    the unlearning groups.  Only the files in ``rel_paths`` that actually
    exist are checked, so each builder can gate the subset it wrote; the
    gate applies only to :data:`REPAIRED_DATASET_VERSION`, because a future
    dataset version that deliberately changes the entity set is allowed to
    retrain (and must then re-pin these hashes).

    Returns the measured hashes, for logging.
    """
    if version != REPAIRED_DATASET_VERSION:
        return {}
    data_dir = Path(data_dir)
    measured: dict[str, str] = {}
    drift: list[str] = []
    for rel in rel_paths:
        pinned = V1_TRAINING_JSONL_SHA256.get(rel)
        if pinned is None:
            continue
        path = data_dir / rel
        if not path.exists():
            continue
        got = _sha256_file(path)
        measured[rel] = got
        if got != pinned:
            drift.append(f"{rel}: v1 {pinned[:12]}... -> v2 {got[:12]}...")
    if drift:
        raise RuntimeError(
            "Training data moved during the visual-split rebuild — the "
            "no-retraining claim is false, so every adapter fitted on "
            "these files is stale:\n  " + "\n  ".join(drift) +
            "\nReserve images[0] as the training photograph, or retrain "
            "MF/MG/MN and all unlearning candidates before reporting.")
    return measured


def _stable_hash(*parts: Any) -> int:
    """sha256-based, so it is identical across processes and runs.

    The builtin ``hash()`` is salted per process for ``str`` and would
    make the photograph assignment depend on ``PYTHONHASHSEED``.
    """
    digest = hashlib.sha256(
        ":".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16)


def image_pools(assoc: AssociationRecord,
                seed: int = 42) -> dict[str, list[ImageRef]]:
    """Photographs available to each split for one association.

    ``train`` is always exactly the reserved training photograph.  The
    remaining photographs are ordered by a seeded shuffle keyed on
    ``(seed, association_id, image_id)`` and cut in half: the first half
    becomes the val pool, the rest the test pool (so 11 spares give 5 val
    and 6 test).  Degenerate cases stay honest instead of silently
    reaching back into the training photograph:

    * one spare photograph  -> val and test SHARE it (still disjoint from
      training);
    * no spare photograph    -> every split falls back to the trained
      photograph, which callers see as ``seen_in_training=True``.
    """
    images = list(assoc.images)
    if not images:
        return {s: [] for s in SPLITS}
    training = images[RESERVED_TRAINING_INDEX]
    rest = images[RESERVED_TRAINING_INDEX + 1:]
    rest.sort(key=lambda im: _stable_hash(seed, assoc.association_id,
                                          im.image_id))
    if not rest:
        # single-photograph entity: no held-out photograph exists
        return {s: [training] for s in SPLITS}
    n_val = len(rest) // 2
    val = rest[:n_val] or rest
    test = rest[n_val:] or rest
    return {"train": [training], "val": val, "test": test}


def image_for_split(
    assoc: AssociationRecord,
    split: str,
    seed: int = 42,
    slot: int = 0,
) -> tuple[ImageRef | None, bool]:
    """The photograph for one image query, and whether training saw it.

    ``slot`` is the query's position within its ``(association, split)``
    group ordered by ``query_id``; round-robin over the pool gives each
    query a distinct photograph until the pool runs out.  The second
    element is True exactly when the assigned photograph is the reserved
    training photograph — the signal that separates the held-out-photo
    stratum from the seen-photo one.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pool = image_pools(assoc, seed).get(split) or []
    if not pool:
        return None, False
    image = pool[slot % len(pool)]
    training = assoc.images[RESERVED_TRAINING_INDEX]
    return image, image.image_id == training.image_id


def photo_labels(assoc: AssociationRecord,
                 seed: int = 42) -> dict[str, str]:
    """``image_id -> the split that USES this photograph``.

    The first pool a photograph appears in wins (train, then val, then
    test), so the single-photograph and single-spare degenerate cases get
    one unambiguous label.  Both :func:`relabel_image_splits` and the
    query assignment read this, so ``ImageRef.split`` and
    ``QueryRecord.image_split`` cannot disagree.
    """
    pools = image_pools(assoc, seed)
    label: dict[str, str] = {}
    for split in SPLITS:
        for image in pools[split]:
            label.setdefault(image.image_id, split)
    return label


def relabel_image_splits(assoc: AssociationRecord,
                         seed: int = 42) -> AssociationRecord:
    """Set every ``ImageRef.split`` to the split that actually USES it.

    The adapter's ``deterministic_image_splits`` assigns 60/20/20 before
    anyone knows which photograph training will consume, so its labels can
    contradict reality — in pilot-100 v1 the trained photograph of 11
    target species was labeled val or test.  Labels must describe use, or
    they advertise a held-out split that does not exist.
    """
    label = photo_labels(assoc, seed)
    images = [im.model_copy(update={"split": label.get(im.image_id)})
              for im in assoc.images]
    return assoc.model_copy(update={"images": images})


def assign_split_images(queries: list[QueryRecord],
                        associations: list[AssociationRecord],
                        seed: int = 42) -> list[QueryRecord]:
    """Re-assign every image query's photograph for its own split.

    A post-pass rather than a rule inside ``_make_query``: round-robin
    distinctness needs to know a query's siblings, and grouping by
    ``(association, split)`` then sorting by ``query_id`` makes the result
    independent of the order queries happened to be emitted in.
    """
    by_id = {a.association_id: a for a in associations}
    groups: dict[tuple[str, str], list[QueryRecord]] = defaultdict(list)
    for q in queries:
        if q.image_ids:
            groups[(q.association_id, q.split)].append(q)
    updates: dict[str, dict[str, Any]] = {}
    for (aid, split), group in groups.items():
        assoc = by_id.get(aid)
        if assoc is None:
            continue
        for slot, q in enumerate(sorted(group, key=lambda r: r.query_id)):
            image, seen = image_for_split(assoc, split, seed, slot)
            if image is None:
                continue
            updates[q.query_id] = {
                "image_ids": [image.image_id],
                "image_split": photo_labels(assoc, seed)[image.image_id],
                "image_seen_in_training": seen,
            }
    return [q.model_copy(update=updates[q.query_id])
            if q.query_id in updates else q
            for q in queries]


def image_stratum(query: QueryRecord) -> str | None:
    """Which image stratum a query belongs to (None for text-only).

    Derived from ``image_seen_in_training``, never from the source
    dataset: the claim "held-out photograph" is a property of the
    assignment, and a MLLMU portrait reused across splits must not be
    able to masquerade as held out.
    """
    if not query.image_ids:
        return None
    return (SEEN_PHOTO_UNSEEN_WORDING if query.image_seen_in_training
            else HELD_OUT_PHOTO)


def validate_image_splits(
    queries: list[QueryRecord],
    associations: list[AssociationRecord],
    seed: int = 42,
) -> list[str]:
    """Invariants the repaired visual split must satisfy.

    Returns a list of human-readable violations (empty == valid), so a
    build can hard-fail and a test can assert on the same rules.
    """
    errors: list[str] = []
    by_id = {a.association_id: a for a in associations}
    used_by_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_by_assoc: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        if not q.image_ids:
            if q.image_split is not None or q.image_seen_in_training:
                errors.append(
                    f"{q.query_id}: text-only query carries image metadata")
            continue
        seen_by_assoc[q.association_id].add(
            "seen" if q.image_seen_in_training else "held_out")
        assoc = by_id.get(q.association_id)
        if assoc is None:
            errors.append(f"{q.query_id}: unknown association")
            continue
        if len(q.image_ids) != 1:
            errors.append(f"{q.query_id}: expected exactly one image")
        known = {im.image_id for im in assoc.images}
        for iid in q.image_ids:
            if iid not in known:
                errors.append(
                    f"{q.query_id}: image {iid} is not on {q.association_id}")
        used_by_split[(q.association_id, q.split)].update(q.image_ids)
        # the flag must agree with the reserved-training photograph
        training = assoc.images[RESERVED_TRAINING_INDEX].image_id
        if q.image_seen_in_training != (q.image_ids[0] == training):
            errors.append(
                f"{q.query_id}: image_seen_in_training="
                f"{q.image_seen_in_training} contradicts the reserved "
                f"training photograph {training}")
        # a val/test query served the TRAINED photograph is only
        # acceptable when that entity has no spare photograph at all
        if q.split != "train" and q.image_seen_in_training:
            pool = image_pools(assoc, seed).get(q.split) or []
            if any(im.image_id != training for im in pool):
                errors.append(
                    f"{q.query_id}: split {q.split} was served the training "
                    f"photograph {training} although held-out photographs "
                    f"exist for {q.association_id}")
    # train / val / test photographs must be pairwise disjoint per entity
    # — but ONLY where disjointness is possible.  A single-photograph
    # entity (every MLLMU person) has no alternative to offer, so the
    # requirement there is that every one of its queries is flagged
    # seen_in_training, which is what keeps it out of the held-out stratum.
    for aid in sorted({a for a, _ in used_by_split}):
        assoc = by_id.get(aid)
        if assoc is None or len(assoc.images) < 2:
            if "held_out" in seen_by_assoc.get(aid, set()):
                errors.append(
                    f"{aid}: single-photograph entity has an image query "
                    f"flagged as held out, which is impossible — it has no "
                    f"photograph other than the trained one")
            continue
        pools = {s: used_by_split.get((aid, s), set()) for s in SPLITS}
        if pools["train"] & pools["val"]:
            errors.append(f"{aid}: train and val share "
                          f"{sorted(pools['train'] & pools['val'])}")
        if pools["train"] & pools["test"]:
            errors.append(f"{aid}: train and test share "
                          f"{sorted(pools['train'] & pools['test'])}")
        if pools["val"] & pools["test"]:
            errors.append(f"{aid}: val and test share "
                          f"{sorted(pools['val'] & pools['test'])}")
    return errors
