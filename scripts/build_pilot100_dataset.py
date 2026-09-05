"""Build the Iteration-11 pilot-100 mixed hierarchical dataset.

    python scripts/build_pilot100_dataset.py \
        --config configs/datasets/pilot100.yaml

100 frozen entities = 64 MLLMU person entities (4 semantic + 3 numeric
core attributes) + 36 REAL iNaturalist species entities (authoritative
taxonomic chains species -> genus -> family, real CC-licensed photos).

Steps
-----
1. Iteration 5 audit gate (accepted Qwen-assisted semantic chains must
   be fully adjudicated — same gate as the smoke build).
2. Rebuild the FULL MLLMU association pool (deterministic hierarchies +
   committed chains; no GPU).
3. Deterministic 64-person selection (coverage-qualified, sha256
   ranked, salt ``pilot100`` — independent of the smoke selection).
4. Load the iNaturalist pilot stratum through the adapter (diversity
   gates: >=15 genera, >=10 families, >=10 images/species).
5. Merge into the 100-entity mixed pool; validate every chain.
6. Freeze BALANCED targets: 30 semantic + 30 numeric + 30 taxonomic
   (per-attribute quotas; the 6 above-quota species stay RETAINED as
   same-stratum other-entity donors).
7. Write associations parquet + frozen manifest (with artifact hashes)
   to ``data/mllmu_hier_pilot100/``; committed reports:
   mllmu_pilot100_entities.json, mllmu_pilot100_target_retain.json.
8. Generate ALL paraphrase-split queries — 15 unlearning families
   (incl. the NEW image_to_text families image_fine_direct /
   image_target_direct) + 4 retain families (text AND image routes) —
   validate, write queries.parquet + mllmu_pilot100_query_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root, resolve_config
from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.inaturalist import INaturalistAdapter
from granunlearn.datasets.pilot import (
    PILOT_TARGET_QUOTAS,
    balance_evidence,
    select_balanced_targets,
)
from granunlearn.datasets.smoke import (
    check_audit_gate,
    selection_evidence,
    select_smoke_entities,
    subset_associations,
)
from granunlearn.evaluation.image_splits import (
    IMAGE_STRATA,
    image_stratum,
    relabel_image_splits,
    validate_image_splits,
)
from granunlearn.evaluation.query_generation import (
    UNLEARNING_FAMILIES,
    family_applicable,
    generate_queries,
    validate_queries,
)
from granunlearn.hierarchy.validate import validate_chain
from granunlearn.logging_utils import save_json, setup_logger
from granunlearn.seed import set_all_seeds

log = setup_logger("build_pilot100_dataset")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_smoke_dataset import load_accepted_chains  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pilot_build(config_path: str | Path,
                    output_dir: str | Path | None = None
                    ) -> tuple[Path, dict[str, Any]]:
    cfg = resolve_config(config_path)
    ds_cfg = dict(cfg.get("dataset", {}))
    mllmu_cfg = dict(ds_cfg.get("mllmu", {}))
    inat_cfg = dict(ds_cfg.get("inaturalist", {}))
    balance_cfg = dict(cfg.get("balance", {}))
    seed = int(ds_cfg.get("seed", 42))
    set_all_seeds(seed)

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    reports = repo_root / "data" / "reports"

    # ---- 1. audit gate ---------------------------------------------------
    audit_path = reports / "mllmu_hier_manual_audit_sample.json"
    chains_path = reports / "mllmu_hier_semantic_chains.jsonl"
    passed, problems = check_audit_gate(audit_path, chains_path)
    if not passed:
        for p in problems:
            log.error("AUDIT GATE: %s", p)
        raise RuntimeError("Iteration 5 audit gate FAILED")
    log.info("Iteration 5 audit gate PASSED")

    # ---- 2. full MLLMU pool ----------------------------------------------
    adapter = get_adapter("mllmu_hier")
    records = adapter.load_raw(mllmu_cfg)
    log.info("Loaded %d MLLMU profiles", len(records))
    if mllmu_cfg.get("materialize_images"):
        image_map = adapter.materialize_images(records, mllmu_cfg)
        log.info("Materialized + verified %d images", len(image_map))
    deterministic = adapter.to_associations(records, mllmu_cfg)
    gen_report = json.loads(
        (reports / "mllmu_hier_qwen_generation_report.json").read_text())
    accepted_chains = load_accepted_chains(chains_path)
    semantic = adapter.to_semantic_associations(
        records, accepted_chains, mllmu_cfg,
        generation_provenance=gen_report["generation_provenance"],
        prompt_versions={
            "proposal": gen_report["prompt_version_proposal"],
            "verification": gen_report["prompt_version_verification"],
        })
    mllmu_pool = deterministic + semantic
    log.info("MLLMU pool: %d deterministic + %d semantic",
             len(deterministic), len(semantic))

    # ---- 3. deterministic person selection --------------------------------
    n_persons = int(mllmu_cfg.get("n_person_entities", 64))
    min_attr = int(mllmu_cfg.get("min_attribute_types", 6))
    salt = str(mllmu_cfg.get("entity_selection_salt", "pilot100"))
    persons = select_smoke_entities(
        mllmu_pool, seed=seed, n=n_persons,
        min_attribute_types=min_attr, salt=salt)
    if len(persons) < n_persons:
        raise RuntimeError(
            f"Only {len(persons)} coverage-qualified persons "
            f"(< {n_persons})")
    log.info("Selected %d person entities (salt=%s)", len(persons),
             salt)
    mllmu_assocs = subset_associations(mllmu_pool, set(persons))

    # ---- 4. iNaturalist taxonomic stratum ---------------------------------
    inat_adapter = INaturalistAdapter()
    species_records = inat_adapter.load_raw(inat_cfg)
    log.info("iNat stratum: %d species (%d genera, %d families)",
             len(species_records),
             inat_adapter.last_load_report["num_genera"],
             inat_adapter.last_load_report["num_families"])
    inat_assocs = inat_adapter.to_associations(species_records,
                                               inat_cfg)

    # ---- 5. merged pool + validation ---------------------------------------
    pilot = mllmu_assocs + inat_assocs
    for assoc in pilot:
        issues = [i for i in validate_chain(assoc.levels) if i.is_error]
        if issues:
            raise ValueError(
                f"Hierarchy validation failed for "
                f"{assoc.association_id}: {issues}")
    n_entities = len({a.entity_id for a in pilot})
    expected = n_persons + len(inat_assocs)
    if n_entities != expected:
        raise RuntimeError(
            f"Mixed pool has {n_entities} entities, expected {expected} "
            "(person/species id collision?)")
    log.info("Mixed pilot pool: %d entities, %d associations",
             n_entities, len(pilot))

    # ---- 5b. relabel image splits to reflect ACTUAL use (Iteration 11R) ----
    # The adapters' 60/20/20 pre-assignment is made before anyone knows
    # which photograph training will consume; in pilot100_v1 the trained
    # photograph of 11 target species was labeled val (6) or test (5),
    # advertising a held-out split that did not exist.  Labels are
    # re-derived from use: images[0] is the reserved training photograph
    # (what state_datasets/unlearning_datasets consume), and the spare
    # photographs form disjoint val/test pools.
    pilot = [relabel_image_splits(a, seed) for a in pilot]
    log.info("Image splits relabeled from use: %s",
             dict(Counter(img.split for a in pilot for img in a.images)))

    # ---- 6. balanced target freeze -----------------------------------------
    quotas = balance_cfg.get("quotas") or PILOT_TARGET_QUOTAS
    partition = select_balanced_targets(pilot, seed=seed,
                                        quotas=quotas)
    bal = balance_evidence(partition, pilot)
    log.info("Balanced targets: %s (%d targets / %d retain; "
             "types_balanced=%s)",
             bal["target_counts_by_type"], bal["num_targets"],
             bal["num_retain"], bal["types_balanced"])
    shortfalls = {k: v["shortfall"] for k, v in bal["quota_fill"].items()
                  if v["shortfall"]}
    if shortfalls:
        log.warning("Quota shortfalls (recorded, never fabricated): %s",
                    shortfalls)

    # ---- 7. write frozen dataset -------------------------------------------
    if output_dir is None:
        output_dir = Path("data") / "mllmu_hier_pilot100"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    assoc_dicts = [json.loads(a.model_dump_json()) for a in pilot]
    assoc_path = output_dir / "associations.parquet"
    pd.DataFrame(assoc_dicts).to_parquet(assoc_path, index=False)

    evidence = selection_evidence(mllmu_pool, persons, seed,
                                  n_persons, min_attr)
    evidence["salt"] = salt
    evidence["inaturalist_stratum"] = {
        "num_species": len(inat_assocs),
        "num_genera": inat_adapter.last_load_report["num_genera"],
        "num_families": inat_adapter.last_load_report["num_families"],
        "species": sorted(a.entity_id for a in inat_assocs),
        "photo_provenance": str(
            (repo_root / inat_cfg["data_root"]
             / inat_cfg.get("provenance_file",
                            "PROVENANCE.json")).relative_to(repo_root)),
    }
    save_json(evidence, reports / "mllmu_pilot100_entities.json")
    partition_out = {**partition, "balance_evidence": bal}
    save_json(partition_out, reports / "mllmu_pilot100_target_retain.json")

    manifest = {
        "dataset": "mllmu_hier_pilot100",
        "version": ds_cfg.get("version", "pilot100_v1"),
        "iteration": 11,
        "seed": seed,
        "num_entities": n_entities,
        "num_person_entities": len(persons),
        "num_species_entities": len(inat_assocs),
        "num_associations": len(pilot),
        "num_image_references": sum(len(a.images) for a in pilot),
        "num_unique_images": len({img.path for a in pilot
                                  for img in a.images}),
        "split_mode": mllmu_cfg.get("split_mode", "entity_level"),
        "attribute_counts": {
            attr: sum(1 for a in pilot if a.attribute_name == attr)
            for attr in sorted({a.attribute_name for a in pilot})},
        "hierarchy_type_counts": {
            ht: sum(1 for a in pilot if a.hierarchy_type == ht)
            for ht in sorted({a.hierarchy_type for a in pilot})},
        "target_association_ids": partition["target_association_ids"],
        "retain_association_ids": partition["retain_association_ids"],
        "target_counts_by_type": partition["target_counts_by_type"],
        "balance": bal,
        "selected_entities": sorted(persons) + sorted(
            a.entity_id for a in inat_assocs),
        "config_path": str(config_path),
        "sources": {
            "mllmu": {
                "source": mllmu_cfg.get("source"),
                "revision": mllmu_cfg.get("source_revision"),
                "annotations": mllmu_cfg.get("annotations_file"),
            },
            "inaturalist": {
                "data_root": inat_cfg.get("data_root"),
                "api": "api.inaturalist.org",
                "photo_licenses": "cc0/cc-by/cc-by-sa/cc-by-nc/"
                                  "cc-by-nc-sa (per-photo in "
                                  "PROVENANCE.json)",
            },
        },
    }
    save_json(manifest, output_dir / "manifest.json")

    # ---- 8. queries (all routes) + validation -------------------------------
    queries = generate_queries(pilot, partition, seed=seed,
                               families=list(UNLEARNING_FAMILIES))
    assoc_by_id = {a.association_id: a for a in pilot}
    retain_facts_by_entity: dict[str, set[str]] = {}
    for rid in partition["retain_association_ids"]:
        a = assoc_by_id[rid]
        facts = retain_facts_by_entity.setdefault(a.entity_id, set())
        facts.add(a.levels[0].value)
        facts.update(a.textual_context)

    targets = [assoc_by_id[i]
               for i in partition["target_association_ids"]]
    applicable_targets = {
        fam: sum(1 for t in targets if family_applicable(t, fam))
        for fam in UNLEARNING_FAMILIES}

    errors, stats = validate_queries(
        queries, pilot, partition=partition,
        retain_facts_by_entity=retain_facts_by_entity)
    if errors:
        for e in errors[:20]:
            log.error("QUERY VALIDATION: %s", e)
        raise RuntimeError(
            f"Query validation failed with {len(errors)} error(s)")
    log.info("Generated %d queries; by_route=%s; validation PASSED",
             len(queries), stats["by_route"])

    # ---- 8b. visual-split gate (Iteration 11R) ------------------------------
    # A build whose val/test image queries are served the photograph that
    # training consumed is measuring unseen wording over a SEEN image.  That
    # was pilot100_v1; it is now a hard build failure.
    image_errors = validate_image_splits(queries, pilot, seed)
    if image_errors:
        for e in image_errors[:20]:
            log.error("VISUAL SPLIT: %s", e)
        raise RuntimeError(
            f"Visual-split validation failed with {len(image_errors)} "
            f"error(s)")
    strata = Counter(image_stratum(q) for q in queries if q.image_ids)
    distinct_used = {i for q in queries for i in q.image_ids}
    log.info("Visual split PASSED: strata=%s; %d/%d distinct photographs "
             "used by queries", dict(strata), len(distinct_used),
             len({img.image_id for a in pilot for img in a.images}))

    q_path = output_dir / "queries.parquet"
    pd.DataFrame([q.model_dump() for q in queries]).to_parquet(
        q_path, index=False)

    query_report = {
        "seed": seed,
        "num_queries": len(queries),
        "unlearning_families": list(UNLEARNING_FAMILIES),
        "retain_families": ["retain_same_entity", "retain_other_entity",
                            "retain_same_entity_image",
                            "retain_other_entity_image"],
        "routes": {
            "text_to_text": "entity named in text, no image",
            "image_to_text": "Iteration 11: entity NEVER named; "
                             "identity must be recovered from the "
                             "image alone (image_fine_direct, "
                             "image_target_direct, retain_*_image)",
            "image_text_to_text": "multimodal_image_text (entity "
                                  "named alongside the image)",
        },
        "split_semantics": (
            "paraphrase split over the SAME target associations: each "
            "(association, family) appears in train/val/test with "
            "distinct paraphrase templates; template assignment is "
            "deterministic: (hash(seed, association_id, family) + "
            "split_index) % 3"),
        "target_retain": {
            "num_targets": len(partition["target_association_ids"]),
            "num_retain": len(partition["retain_association_ids"]),
            "target_counts_by_type": partition["target_counts_by_type"],
            "balance": bal,
        },
        "applicable_targets_per_family": applicable_targets,
        "validation": {"passed": True,
                       "num_errors": stats["num_errors"]},
        "by_split": stats["by_split"],
        "by_family": stats["by_family"],
        "by_route": stats["by_route"],
        "num_retain_same_entity": stats["num_retain_same_entity"],
        "num_retain_other_entity": stats["num_retain_other_entity"],
        "num_retain_same_entity_image":
            stats["num_retain_same_entity_image"],
        "num_retain_other_entity_image":
            stats["num_retain_other_entity_image"],
        "num_associations_with_queries":
            stats["num_associations_with_queries"],
        "num_associations_total": len(pilot),
        "image_split": {
            "validation_passed": not image_errors,
            "strata": {s: strata.get(s, 0) for s in IMAGE_STRATA},
            "num_distinct_photographs_used": len(distinct_used),
            "num_distinct_photographs_available":
                len({img.image_id for a in pilot for img in a.images}),
            "photographs_per_split_label": dict(Counter(
                img.split for a in pilot for img in a.images)),
            "note": (
                "held_out_photo queries are served a photograph training "
                "never saw; seen_photo_unseen_wording queries are served "
                "the trained photograph because their entity has only one "
                "(every MLLMU person). The two strata are reported "
                "separately and must never be pooled into a single "
                "'held-out image' claim."),
        },
    }
    save_json(query_report, reports / "mllmu_pilot100_query_report.json")

    # ---- freeze hashes (added AFTER artifacts exist) ------------------------
    manifest["num_unique_images_used_by_queries"] = len(distinct_used)
    manifest["image_strata"] = {s: strata.get(s, 0) for s in IMAGE_STRATA}
    manifest["image_split_policy"] = {
        "reserved_training_index": 0,
        "rule": (
            "images[0] is RESERVED as the training photograph because "
            "state_datasets and unlearning_datasets consume it; the "
            "remaining photographs are ordered by sha256(seed, "
            "association_id, image_id) and cut in half into disjoint val "
            "and test pools (11 spares -> 5 val / 6 test). Image queries "
            "round-robin their own split's pool ordered by query_id."),
        "override_of_adapter_preassignment": (
            "ImageRef.split is relabeled from ACTUAL USE, overriding the "
            "iNaturalist adapter's 60/20/20 pre-assignment, which is made "
            "before anyone knows which photograph training will consume. "
            "In pilot100_v1 the trained photograph of 11 of the 30 target "
            "species was labeled val (6) or test (5)."),
        "single_photograph_entities": (
            "An entity with one photograph (every MLLMU person) cannot have "
            "a held-out photograph: all splits keep the trained portrait "
            "and every such query carries image_seen_in_training=True, "
            "which is what places it in the seen_photo_unseen_wording "
            "stratum rather than the held-out one."),
        "within_split_repetition": (
            "12 photographs per species leave 5 val and 6 test for up to 10 "
            "image queries per split, so a few species repeat a photograph "
            "within one split. Disjointness from TRAINING is the property "
            "the held-out claim needs; distinctness between two test "
            "queries is not, and the repetition is recorded here rather "
            "than hidden."),
        "defect_repaired": (
            "pilot100_v1 assigned images[0] to every train/val/test image "
            "query, so the image route measured unseen wording over a SEEN "
            "photograph and 396 of 496 distinct photographs were unused."),
    }
    manifest["frozen_artifact_sha256"] = {
        "associations.parquet": _sha256(assoc_path),
        "queries.parquet": _sha256(q_path),
        "manifest_pre_hash_note": "hashes computed over the final "
                                  "parquet artifacts",
    }
    save_json(manifest, output_dir / "manifest.json")
    log.info("Pilot-100 dataset FROZEN -> %s", output_dir)
    return output_dir, {"manifest": manifest,
                        "query_report": query_report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Iteration-11 pilot-100 mixed dataset")
    parser.add_argument("--config",
                        default="configs/datasets/pilot100.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_pilot_build(args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
