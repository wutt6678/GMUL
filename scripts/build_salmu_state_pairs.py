"""Build the REAL SALMU reference-state pair sets (Iteration 10).

    python scripts/build_salmu_state_pairs.py
    python scripts/build_salmu_state_pairs.py --suffix r5 \
        --allowed-split forget        # 10R5 holdout-clean build

Source: the RELEASED salmu-512-redistributed `sensitive` split (the
knowledge-injection pairs used to train the Compromised model),
restricted to the three CORE attributes (city / job / blood_type).
Aux-redaction identifiers (phone/email/IBAN/credit card/passport) are
never included.

Per-attribute targeting (v2): each target persona has exactly ONE
target attribute (deterministic hash).  The remaining core attributes
are *same-entity retain* — they keep fine captions in ALL states
including MN.  This mirrors SALMUBench's holdout_association design.

States (identical image sets; only caption treatment differs):
* MF: all core fine (released) pairs
* MG: target (persona, attr) -> generalized target captions only;
      ALL retain (incl. same-entity) -> released fine pairs
* MN: ALL retain fine pairs only (target pairs omitted)

10R5 holdout-clean build (``--allowed-split forget --suffix r5``)
------------------------------------------------------------------
The released sensitive training dataset is the union of the official
``forget`` + ``holdout_association`` + ``holdout_identity`` splits,
and the SALMUBench protocol prohibits training on holdout data.
With ``--allowed-split forget`` the pair universe is restricted to
released pairs whose ``file_name`` is in the official ``forget``
split, and the target partition keeps ONLY personas whose designated
target association survives that filter (target associations are
therefore selected EXCLUSIVELY from the official forget split).

10R2-10R4 caveat (no filter): the original build consumed released
holdout pairs — see holdout_consumption in
data/reports/salmu_official_splits.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import REPOS, locate_repo
from granunlearn.salmu.hierarchy import CORE_SEMANTIC, generalized_caption
from granunlearn.salmu.paths import SalmuPaths
from granunlearn.salmu.state_datasets import (
    SalmuTrainingPair,
    partition_persona_attributes,
    partition_personas,
    validate_state_pairs,
)

log = setup_logger("build_salmu_state_pairs")

NUM_TARGET_PERSONAS = 60  # small first experiment (of 774 personas)
SEED = 42


def _allowed_file_names(bench: Path, split: str) -> set[str]:
    """file_name universe of one released evaluation split."""
    import pandas as pd
    files = sorted((bench / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards for split {split}")
    names: set[str] = set()
    for pq in files:
        col = pd.read_parquet(pq, columns=["file_name"])
        names.update(col["file_name"].dropna())
    return names


def is_holdout_clean_build(allowed_split: str | None) -> bool:
    """A build is holdout-clean ONLY when restricted to the official
    ``forget`` split — not for ANY allowed split (10R5a): forget is
    the only sensitive split the protocol permits for training, and
    it is pair-disjoint from both holdout splits."""
    return allowed_split == "forget"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SALMU reference-state pair sets")
    parser.add_argument("--num-targets", type=int,
                        default=NUM_TARGET_PERSONAS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--suffix", default="",
                        help="Iteration tag for all outputs "
                             "(e.g. r5 -> training_r5/)")
    parser.add_argument("--allowed-split", default=None,
                        help="Restrict the pair universe to released "
                             "pairs present in this official split "
                             "(e.g. forget — the 10R5 holdout-clean "
                             "protocol)")
    args = parser.parse_args()
    if args.allowed_split and not args.suffix:
        raise SystemExit(
            "--allowed-split requires --suffix (filtered builds must "
            "never overwrite the original pair sets)")

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    paths = SalmuPaths(repo_root, suffix=args.suffix)
    out_dir = paths.training_dir
    bench = locate_repo(REPOS["benchmark_dataset"]["repo_id"], "dataset")
    train_ds = locate_repo(REPOS["training_dataset"]["repo_id"], "dataset")

    identities = json.loads((bench / "identities_metadata.json").read_text())
    hierarchies = json.loads(
        (repo_root / "data" / "salmu_hierarchical" /
         "associations.json").read_text())
    cap_meta = json.loads(
        (train_ds / "sensitive_set_captions_metadata.json").read_text())

    # Restrict to CORE attributes of the inventory.
    core_pairs: dict[str, dict] = {
        fname: meta for fname, meta in cap_meta.items()
        if meta["data_field"] in CORE_SEMANTIC}
    log.info("Core-attribute pairs: %d of %d released sensitive pairs",
             len(core_pairs), len(cap_meta))

    # 10R5 holdout-clean protocol: only officially permitted pairs
    # (the ``forget`` split) may enter ANY training set.
    allowed_names: set[str] | None = None
    bench_revision = None
    if args.allowed_split:
        allowed_names = _allowed_file_names(bench, args.allowed_split)
        before = len(core_pairs)
        core_pairs = {f: m for f, m in core_pairs.items()
                      if f in allowed_names}
        parts = Path(bench).parts
        if "snapshots" in parts:
            bench_revision = parts[parts.index("snapshots") + 1]
        log.info("Allowed-split filter '%s': %d -> %d core pairs "
                 "(universe: %d released pairs)",
                 args.allowed_split, before, len(core_pairs),
                 len(allowed_names))

    partition = partition_personas(
        sorted(identities), num_targets=args.num_targets, seed=args.seed)
    targets = set(partition["target_identity_ids"])
    target_attr_map = partition_persona_attributes(
        partition["target_identity_ids"], CORE_SEMANTIC, seed=args.seed)
    full_target_ids = list(partition["target_identity_ids"])

    if allowed_names is not None:
        # Target associations must come EXCLUSIVELY from the allowed
        # split: keep only personas whose designated target attribute
        # still has at least one permitted pair.
        targetable = sorted(
            iid for iid in partition["target_identity_ids"]
            if any(f in allowed_names
                   for f, m in cap_meta.items()
                   if f.split("_")[0] == iid
                   and m["data_field"] == target_attr_map[iid]
                   and m["data_field"] in CORE_SEMANTIC))
        dropped = sorted(targets - set(targetable))
        log.info("Holdout-clean targeting: %d -> %d target personas "
                 "(dropped, no forget-split target association: %s)",
                 len(targets), len(targetable), dropped)
        targets = set(targetable)
        target_attr_map = {iid: a for iid, a in target_attr_map.items()
                           if iid in targets}
        partition = {
            "seed": args.seed,
            "num_targets": len(targetable),
            "num_retain": len(identities) - len(targetable),
            "target_identity_ids": targetable,
            "retain_identity_ids": sorted(
                set(identities) - set(targetable)),
        }
    log.info("Partition: %d target personas / %d retain (seed %d)",
             partition["num_targets"], partition["num_retain"], args.seed)
    # Log the per-attribute assignment distribution
    from collections import Counter
    attr_dist = Counter(target_attr_map.values())
    log.info("Target attribute distribution: %s", dict(attr_dist))

    # Index core pairs by (identity, attribute): [(file, caption)]
    by_id_attr: dict[tuple[str, str], list[tuple[str, str]]] = \
        defaultdict(list)
    for fname in sorted(core_pairs):
        meta = core_pairs[fname]
        iid = fname.split("_")[0]
        by_id_attr[(iid, meta["data_field"])].append(
            (fname, meta["caption"]))

    states_pairs: dict[str, list[SalmuTrainingPair]] = {
        "MF": [], "MG": [], "MN": []}
    for (iid, attr) in sorted(by_id_attr):
        if iid not in hierarchies or attr not in hierarchies[iid]:
            continue  # attribute missing from hierarchy (defensive)
        hier = hierarchies[iid][attr]
        is_target_persona = iid in targets
        is_target_attr = (is_target_persona and 
                         attr == target_attr_map.get(iid))
        # Per-attribute targeting with explicit role distinction:
        # - target_association: the ONE target attribute of a target persona
        # - same_entity_retain: non-target attributes of target personas
        # - other_entity_retain: all attributes of non-target personas
        if is_target_attr:
            role = "target_association"
        elif is_target_persona:
            role = "same_entity_retain"
        else:
            role = "other_entity_retain"
        name = identities[iid]["name"]
        fine = sorted(by_id_attr[(iid, attr)])
        for state in ("MF", "MG", "MN"):
            if role == "target_association" and state == "MN":
                continue
            if role != "target_association" or state == "MF":
                entries = [(0, "released_fine", fname, cap)
                           for fname, cap in fine]
            else:  # MG targets: ONE generalized target caption,
                # paired with the SAME images as the fine captions
                lv = hier["target_level"]
                gen = generalized_caption(name, attr, lv,
                                          hier["levels"][lv])
                entries = [(lv, "generalized_template", fname, gen)
                           for fname, _ in fine]
            for level_index, source, fname, caption in entries:
                states_pairs[state].append(SalmuTrainingPair(
                    pair_id=f"{state}__{iid}__{attr}__{Path(fname).stem}",
                    state=state,
                    identity_id=iid,
                    attribute=attr,
                    role=role,
                    level_index=level_index,
                    caption=caption,
                    caption_source=source,
                    image_file=fname,
                ))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": "cvc-mmu/salmu-512-redistributed `sensitive` split, "
                  "core attributes only (city/job/blood_type)",
        "protocol": {
            "holdout_clean": is_holdout_clean_build(args.allowed_split),
            "allowed_split": args.allowed_split,
            "benchmark_repo_id":
                REPOS["benchmark_dataset"]["repo_id"]
                if args.allowed_split else None,
            "benchmark_revision": bench_revision,
            "note": (
                "10R5 protocol: pair universe restricted to the "
                "official permitted split; target personas keep only "
                "designations with >=1 permitted target pair. Without "
                "a filter (10R2-10R4 builds) released holdout pairs "
                "are consumed — those results are transfer "
                "diagnostics. The official `retain` split carries NO "
                "sensitive associations (all 16,741 captions are "
                "generic utility descriptions; 0 core-attribute / 0 "
                "aux-identifier hits), so the forget split is the "
                "ONLY officially permitted sensitive training data; "
                "all groups (fine_target/target_level/retain) are "
                "therefore built from forget pairs exclusively and "
                "are holdout-clean."
                if args.allowed_split else
                "UNFILTERED build: released holdout pairs are part of "
                "the pair universe (see holdout_consumption in "
                "salmu_official_splits.json)."),
            "derived_from_target_personas": full_target_ids
            if args.allowed_split else None,
        },
        "partition": {
            "seed": args.seed,
            "num_targets": partition["num_targets"],
            "num_retain": partition["num_retain"],
            "target_identity_ids": partition["target_identity_ids"],
        },
        "target_attr_map": target_attr_map,
        "states": {},
    }
    for state, pairs in states_pairs.items():
        errors = validate_state_pairs(pairs, partition, state,
                                      target_attr_map)
        if errors:
            raise ValueError(f"{state} validation failed: {errors[:5]}")
        path = out_dir / f"{state}.jsonl"
        with open(path, "w") as f:
            for p in pairs:
                f.write(p.model_dump_json() + "\n")
        n_tgt = sum(1 for p in pairs if p.role == "target_association")
        n_same_retain = sum(1 for p in pairs if p.role == "same_entity_retain")
        n_other_retain = sum(1 for p in pairs if p.role == "other_entity_retain")
        manifest["states"][state] = {
            "num_pairs": len(pairs),
            "num_target_associations": n_tgt,
            "num_same_entity_retain": n_same_retain,
            "num_other_entity_retain": n_other_retain,
            "num_identities": len({p.identity_id for p in pairs}),
        }
        log.info("%s: %d pairs (target_association %d / same_entity_retain %d / other_entity_retain %d)",
                 state, len(pairs), n_tgt, n_same_retain, n_other_retain)
    with open(out_dir / "state_pairs_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Wrote SALMU state pairs -> %s", out_dir)


if __name__ == "__main__":
    main()
