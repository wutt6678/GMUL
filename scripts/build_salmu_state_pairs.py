"""Build the REAL SALMU reference-state pair sets (Iteration 10).

    python scripts/build_salmu_state_pairs.py

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

Evaluation splits of SALMUBench stay untouched / evaluation-only.
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
from granunlearn.salmu.state_datasets import (
    SalmuTrainingPair,
    partition_persona_attributes,
    partition_personas,
    validate_state_pairs,
)

log = setup_logger("build_salmu_state_pairs")

NUM_TARGET_PERSONAS = 60  # small first experiment (of 774 personas)
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SALMU reference-state pair sets")
    parser.add_argument("--num-targets", type=int,
                        default=NUM_TARGET_PERSONAS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    out_dir = repo_root / "data" / "salmu_hierarchical" / "training"
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

    partition = partition_personas(
        sorted(identities), num_targets=args.num_targets, seed=args.seed)
    targets = set(partition["target_identity_ids"])
    target_attr_map = partition_persona_attributes(
        partition["target_identity_ids"], CORE_SEMANTIC, seed=args.seed)
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
        # Per-attribute targeting: only the designated target attribute
        # of a target persona is "target"; the rest are "retain".
        if is_target_persona and attr == target_attr_map.get(iid):
            role = "target"
        else:
            role = "retain"
        name = identities[iid]["name"]
        fine = sorted(by_id_attr[(iid, attr)])
        for state in ("MF", "MG", "MN"):
            if role == "target" and state == "MN":
                continue
            if role == "retain" or state == "MF":
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
        n_tgt = sum(1 for p in pairs if p.role == "target")
        manifest["states"][state] = {
            "num_pairs": len(pairs),
            "num_target_pairs": n_tgt,
            "num_retain_pairs": len(pairs) - n_tgt,
            "num_identities": len({p.identity_id for p in pairs}),
        }
        log.info("%s: %d pairs (target %d / retain %d)",
                 state, len(pairs), n_tgt, len(pairs) - n_tgt)
    with open(out_dir / "state_pairs_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("Wrote SALMU state pairs -> %s", out_dir)


if __name__ == "__main__":
    main()
