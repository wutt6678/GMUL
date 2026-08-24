"""10-entity MLLMU smoke dataset + query generation (Iteration 6).

    python scripts/build_smoke_dataset.py --config configs/datasets/mllmu.yaml

Steps
-----
1. Iteration 5 audit gate — the committed manual audit sample must be
   fully adjudicated with no contradiction against the accepted chains.
2. Rebuild the FULL association pool (deterministic hierarchies from the
   official parquet + accepted Qwen-assisted chains from the committed
   ``mllmu_hier_semantic_chains.jsonl``; no GPU needed).
3. Deterministic 10-entity selection (coverage-qualified, sha256 ranked).
4. Write the smoke dataset (parquet + manifest) to
   ``data/mllmu_hier_smoke/``.
5. Generate paraphrase-split evaluation queries for all 13 families and
   validate them (answer-in-levels, split coverage, distinct templates,
   retain-fact dedupe); write queries + committed evidence report.

The query split is a PARAPHRASE split over the same target associations:
each (association, family) appears in train/val/test with three distinct
paraphrase templates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root, resolve_config
from granunlearn.datasets.base import get_adapter
from granunlearn.datasets.smoke import (
    check_audit_gate,
    select_smoke_entities,
    select_target_retain,
    selection_evidence,
    subset_associations,
)
from granunlearn.evaluation.query_generation import (
    UNLEARNING_FAMILIES,
    family_applicable,
    generate_queries,
    validate_queries,
)
from granunlearn.hierarchy.validate import validate_chain
from granunlearn.logging_utils import setup_logger, save_json
from granunlearn.seed import set_all_seeds

log = setup_logger("build_smoke_dataset")


def load_accepted_chains(chains_path: Path) -> dict[str, dict[str, list[str]]]:
    """Committed accepted chains jsonl -> {attribute: {value: chain}}."""
    chains: dict[str, dict[str, list[str]]] = {}
    for line in chains_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chains.setdefault(row["attribute"], {})[row["value"]] = row["chain"]
    return chains


def run_smoke_build(
    config_path: str | Path,
    n_entities: int = 10,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build the smoke dataset + queries. Returns (output_dir, report)."""
    cfg = resolve_config(config_path)
    ds_cfg = dict(cfg.get("dataset", {}))
    qf_cfg = dict(cfg.get("query_families", {}))
    seed = int(ds_cfg.get("seed", 42))
    set_all_seeds(seed)

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    reports = repo_root / "data" / "reports"

    # ---- 1. Iteration 5 audit gate -------------------------------------
    audit_path = reports / "mllmu_hier_manual_audit_sample.json"
    chains_path = reports / "mllmu_hier_semantic_chains.jsonl"
    passed, problems = check_audit_gate(audit_path, chains_path)
    if not passed:
        for p in problems:
            log.error("AUDIT GATE: %s", p)
        raise RuntimeError(
            f"Iteration 5 audit gate FAILED with {len(problems)} problem(s); "
            f"cannot build the smoke dataset")
    log.info("Iteration 5 audit gate PASSED")

    # ---- 2. Full association pool (no GPU: committed chains reused) ----
    adapter = get_adapter(ds_cfg.get("name", "mllmu_hier"))
    records = adapter.load_raw(ds_cfg)
    log.info("Loaded %d profiles", len(records))
    if ds_cfg.get("materialize_images"):
        image_map = adapter.materialize_images(records, ds_cfg)
        log.info("Materialized + verified %d images", len(image_map))

    deterministic = adapter.to_associations(records, ds_cfg)
    log.info("Built %d deterministic associations", len(deterministic))

    gen_report = json.loads(
        (reports / "mllmu_hier_qwen_generation_report.json").read_text())
    accepted_chains = load_accepted_chains(chains_path)
    semantic = adapter.to_semantic_associations(
        records, accepted_chains, ds_cfg,
        generation_provenance=gen_report["generation_provenance"],
        prompt_versions={
            "proposal": gen_report["prompt_version_proposal"],
            "verification": gen_report["prompt_version_verification"],
        },
    )
    log.info("Built %d semantic associations from committed chains",
             len(semantic))

    pool = deterministic + semantic
    for assoc in pool:
        issues = [i for i in validate_chain(assoc.levels) if i.is_error]
        if issues:
            raise ValueError(
                f"Hierarchy validation failed for {assoc.association_id}: "
                f"{issues}")

    # ---- 3. Deterministic entity selection ------------------------------
    min_attr = int(qf_cfg.get("smoke_min_attribute_types", 6))
    selected = select_smoke_entities(
        pool, seed=seed, n=n_entities, min_attribute_types=min_attr)
    if len(selected) < n_entities:
        raise RuntimeError(
            f"Only {len(selected)} coverage-qualified entities "
            f"(< {n_entities}); widen eligibility or the source pool")
    log.info("Selected %d smoke entities", len(selected))
    smoke = subset_associations(pool, set(selected))

    # ---- 3b. Target/retain partition (Blocker-2 fix) --------------------
    # Entity-attribute SELECTIVE unlearning: per entity, 1 semantic +
    # 1 numeric association are TARGETS (F); the rest are RETAIN (R).
    targets_per_type = dict(qf_cfg.get(
        "smoke_targets_per_type", {"semantic": 1, "numeric": 1}))
    partition = select_target_retain(smoke, seed=seed,
                                     targets_per_type=targets_per_type)
    log.info(
        "Target/retain partition: %d targets %s | %d retain",
        len(partition["target_association_ids"]),
        partition["target_counts_by_type"],
        len(partition["retain_association_ids"]),
    )

    # ---- 4. Write smoke dataset -----------------------------------------
    if output_dir is None:
        output_dir = Path("data") / "mllmu_hier_smoke"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    assoc_dicts = [json.loads(a.model_dump_json()) for a in smoke]
    pd.DataFrame(assoc_dicts).to_parquet(
        output_dir / "associations.parquet", index=False)

    evidence = selection_evidence(pool, selected, seed, n_entities, min_attr)
    save_json(evidence, reports / "mllmu_smoke_entities.json")
    save_json(partition, reports / "mllmu_smoke_target_retain.json")

    manifest = {
        "dataset": "mllmu_hier",
        "version": "smoke_v1",
        "seed": seed,
        "num_entities": len(selected),
        "num_associations": len(smoke),
        "num_image_references": sum(len(a.images) for a in smoke),
        "num_unique_images": len({img.path for a in smoke
                                  for img in a.images}),
        "split_mode": ds_cfg.get("split_mode", "entity_level"),
        "attribute_counts": {
            attr: sum(1 for a in smoke if a.attribute_name == attr)
            for attr in sorted({a.attribute_name for a in smoke})
        },
        "hierarchy_type_counts": {
            ht: sum(1 for a in smoke if a.hierarchy_type == ht)
            for ht in sorted({a.hierarchy_type for a in smoke})
        },
        # Entity-attribute selective unlearning contract (F/R partition):
        # MF/MG/MN operate on `target_*`; `retain_*` must stay unchanged.
        "target_association_ids": partition["target_association_ids"],
        "retain_association_ids": partition["retain_association_ids"],
        "target_counts_by_type": partition["target_counts_by_type"],
        "selected_entities": selected,
        "config_path": str(config_path),
    }
    save_json(manifest, output_dir / "manifest.json")
    log.info("Smoke dataset: %d entities, %d associations -> %s",
             len(selected), len(smoke), output_dir)

    # ---- 5. Query generation + validation -------------------------------
    families = list(qf_cfg.get("families", UNLEARNING_FAMILIES))
    queries = generate_queries(smoke, partition, seed=seed, families=families)

    # Retain-fact corpus for a NON-VACUOUS dedupe check (Blocker-3 fix,
    # review #3): per entity, the exact fine values and textual contexts
    # of its RETAINED associations.  The check is entity-scoped because
    # facts are entity-conditioned (two entities may share a value).
    assoc_by_id = {a.association_id: a for a in smoke}
    retain_facts_by_entity: dict[str, set[str]] = {}
    for rid in partition["retain_association_ids"]:
        a = assoc_by_id[rid]
        facts = retain_facts_by_entity.setdefault(a.entity_id, set())
        facts.add(a.levels[0].value)
        facts.update(a.textual_context)

    targets = [assoc_by_id[i] for i in partition["target_association_ids"]]
    applicable_targets = {
        fam: sum(1 for t in targets if family_applicable(t, fam))
        for fam in families
    }

    errors, stats = validate_queries(
        queries, smoke, partition=partition,
        retain_facts_by_entity=retain_facts_by_entity)
    if errors:
        for e in errors[:20]:
            log.error("QUERY VALIDATION: %s", e)
        raise RuntimeError(
            f"Query validation failed with {len(errors)} error(s)")
    log.info("Generated %d queries across %d families; validation PASSED",
             len(queries), len({q.family for q in queries}))

    qdf = pd.DataFrame([q.model_dump() for q in queries])
    qdf.to_parquet(output_dir / "queries.parquet", index=False)

    query_report = {
        "seed": seed,
        "num_queries": len(queries),
        "unlearning_families": families,
        "query_schema": "granunlearn.schema.QueryRecord (canonical)",
        "split_semantics": (
            "paraphrase split over the SAME target associations: each "
            "(association, family) appears in train/val/test with distinct "
            "paraphrase templates; template assignment is deterministic: "
            "(hash(seed, association_id, family) + split_index) % 3"
        ),
        "target_retain": {
            "num_targets": len(partition["target_association_ids"]),
            "num_retain": len(partition["retain_association_ids"]),
            "target_counts_by_type": partition["target_counts_by_type"],
            "note": (
                "Unlearning families (F/G/N/M) are generated for TARGET "
                "associations only; retain_same_entity covers every "
                "retained association of the target entities and "
                "retain_other_entity one deterministic donor per target. "
                "This makes MF/MG/MN vs MU deltas interpretable as "
                "entity-attribute SELECTIVE unlearning."
            ),
        },
        # Hierarchy-shape applicability (Blocker-1 fix, review #3):
        # granular_intermediate is only generated when a level strictly
        # above the target exists, so per-family counts legitimately
        # differ between families.
        "applicable_targets_per_family": applicable_targets,
        "prompt_self_containment": (
            "Granularity probes use attribute-aware, level-specific "
            "wording (LEVEL_QUESTIONS) and never reference hidden "
            "benchmark metadata such as 'the target granularity': e.g. "
            "'What year was X born?', 'Which salary range does X fall "
            "into?', 'In which country does X live?', or relative "
            "wording for semantic hierarchies."
        ),
        "adversarial": {
            "num_queries": stats["num_adversarial"],
            "families": ["negation_correction"],
            "policy": (
                "Prompted-recovery probes quote the forgotten fine value; "
                "they are EXCLUDED from the core FILR average and reported "
                "separately."
            ),
        },
        "multimodal_route_caveat": (
            "multimodal_image_text names the entity in the text "
            "(route=image_text_to_text); it is NOT evidence for the "
            "image->identity->attribute route (image_to_text is a later "
            "iteration)."
        ),
        "validation": {
            "passed": True,
            "num_errors": stats["num_errors"],
            "checks": [
                "acceptable_answer_ids_equal_expected_level_canonical_id",
                "expected_answer_is_exact_level_value",
                "forbidden_descendant_ids_are_exactly_finer_than_expected",
                "filr_view_consistent_with_unlearning_target_level",
                "leakage_forbidden_ids_are_exactly_finer_than_target",
                "retain_probes_have_no_unlearning_semantics",
                "one_query_per_split_per_association_family",
                "distinct_templates_across_splits",
                "unique_query_ids",
                "adversarial_flag_matches_family_policy",
                "retain_same_entity_covers_all_retained_non_target",
                "retain_other_entity_donor_is_retained_association",
                "retain_other_entity_donor_from_different_entity",
                "retain_other_entity_explicit_target_association_id",
                "retain_fact_dedupe_entity_scoped_non_vacuous",
                "prompts_self_contained_no_hidden_granularity_metadata",
                "negation_distractor_never_equals_answer",
            ],
        },
        "by_split": stats["by_split"],
        "by_family": stats["by_family"],
        "num_retain_same_entity": stats["num_retain_same_entity"],
        "num_retain_other_entity": stats["num_retain_other_entity"],
        "retain_other_entity_donor_pairs": stats["donor_pairs"],
        "retain_fact_corpus": {
            "num_entities": len(retain_facts_by_entity),
            "num_facts": sum(len(v) for v in retain_facts_by_entity.values()),
            "source": (
                "per entity: exact fine values + textual_context of its "
                "retained associations"
            ),
        },
        "num_associations_with_queries": stats[
            "num_associations_with_queries"],
        "num_associations_total": len(smoke),
    }
    save_json(query_report, reports / "mllmu_smoke_query_report.json")
    log.info("Saved query report -> %s", reports / "mllmu_smoke_query_report.json")

    return output_dir, {"manifest": manifest, "query_report": query_report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 10-entity MLLMU smoke dataset + queries")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-entities", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_smoke_build(args.config, n_entities=args.num_entities,
                    output_dir=args.output)


if __name__ == "__main__":
    main()
