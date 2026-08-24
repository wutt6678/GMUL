"""Qwen-assisted semantic hierarchy stage (Iteration 5).

    python scripts/generate_semantic_hierarchies.py \
        --config configs/datasets/mllmu.yaml [--max-entities N] [--bf16]

Runs occupation/education through the gated pipeline (proposal ->
deterministic validation -> independent verification -> confidence gate),
writes the committed generation/verification report + manual audit sample,
and builds AssociationRecords for ACCEPTED chains only.

The manual audit sample must be reviewed (auditor_verdict filled in)
BEFORE proceeding to the 10-entity MLLMU smoke dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root, resolve_config
from granunlearn.datasets.base import get_adapter
from granunlearn.hierarchy.parsers import is_missing
from granunlearn.logging_utils import setup_logger, save_json
from granunlearn.qwen.client import QwenGenerator
from granunlearn.qwen.semantic_pipeline import (
    prompt_version,
    run_semantic_pipeline,
    PROPOSAL_PROMPT, VERIFICATION_PROMPT,
    EDUCATION_EXTRA_RULES, OCCUPATION_EXTRA_RULES,
)

log = setup_logger("semantic_hierarchies")

SEMANTIC_ATTRIBUTES = ["occupation", "education"]


def run_semantic_stage(
    config_path: str | Path,
    max_entities: int | None = None,
    device: str | None = None,
    load_in_4bit: bool = True,
    generator=None,
) -> dict[str, Path]:
    """Run the full Iteration-5 stage. Returns written artifact paths.

    ``generator`` may be injected for testing (a FakeGenerator); when
    None a real QwenGenerator is constructed.
    """
    cfg = resolve_config(config_path)
    ds_cfg = dict(cfg.get("dataset", {}))
    qwen_cfg = dict(cfg.get("qwen", {}))
    seed = int(ds_cfg.get("seed", 42))

    adapter = get_adapter(ds_cfg.get("name", "mllmu_hier"))
    records = adapter.load_raw(ds_cfg)
    if max_entities is not None:
        records = records[:max_entities]
    log.info("Loaded %d profiles", len(records))

    # Collect (entity, value) pairs; missing values are excluded here and
    # counted explicitly in the report (policy: missing).
    attribute_values: dict[str, list[tuple[str, str]]] = {}
    n_missing: dict[str, int] = {}
    for attr in SEMANTIC_ATTRIBUTES:
        pairs = []
        miss = 0
        for rec in records:
            v = rec.fields.get(attr)
            if is_missing(v):
                miss += 1
                continue
            pairs.append((rec.entity_id, str(v).strip()))
        attribute_values[attr] = pairs
        n_missing[attr] = miss
        log.info("%s: %d non-missing values (%d missing)",
                 attr, len(pairs), miss)

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    cache_dir = repo_root / "data" / "generated" / "qwen_cache" / "mllmu_hier"

    if generator is None:
        generator = QwenGenerator(
            model_id=qwen_cfg.get("model_id", "Qwen/Qwen3.5-9B"),
            device=device or qwen_cfg.get("device", "auto"),
            load_in_4bit=(not qwen_cfg.get("bf16", False)) and load_in_4bit,
        )
        log.info("Qwen generation provenance: %s", generator.provenance())

    use_bf16 = bool(qwen_cfg.get("bf16", False)) or not load_in_4bit
    generate_kwargs = {"max_new_tokens": int(qwen_cfg.get("max_new_tokens", 1024)),
                       "batch_size": int(qwen_cfg.get("batch_size", 4))}
    # load_mode is part of the cache identity (settings hash): 4-bit and
    # BF16 are different model configurations and must not share a cache.
    cache_identity = {"load_mode": "bf16" if use_bf16 else "4bit_bnb"}
    result = run_semantic_pipeline(
        generator, attribute_values,
        config={
            "min_proposal_confidence": qwen_cfg.get("min_proposal_confidence", 0.6),
            "min_verification_confidence": qwen_cfg.get("min_verification_confidence", 0.7),
            "audit_sample_accepted": qwen_cfg.get("audit_sample_accepted", 10),
            "audit_sample_rejected": qwen_cfg.get("audit_sample_rejected", 5),
            "generate_kwargs": generate_kwargs,
            "cache_identity": cache_identity,
        },
        cache_dir=cache_dir, seed=seed,
    )
    result.report["generation_settings"] = {**generate_kwargs, **cache_identity}

    # Enrich report with pre-pipeline missing counts + association evidence
    for attr in SEMANTIC_ATTRIBUTES:
        result.report["attributes"][attr]["missing_values"] = n_missing[attr]
        result.report["attributes"][attr]["accepted_chains_count"] = len(
            result.accepted_chains[attr])

    # ---- AssociationRecords for accepted chains ---------------------------
    if ds_cfg.get("materialize_images"):
        adapter.materialize_images(records, ds_cfg)
    associations = adapter.to_semantic_associations(
        records, result.accepted_chains, ds_cfg,
        generation_provenance=result.report["generation_provenance"],
        prompt_versions={
            "proposal": result.report["prompt_version_proposal"],
            "verification": result.report["prompt_version_verification"],
        },
    )
    result.report["num_semantic_associations"] = len(associations)
    log.info("Built %d semantic associations", len(associations))

    # ---- Write committed evidence ------------------------------------------
    reports = repo_root / "data" / "reports"
    report_path = reports / "mllmu_hier_qwen_generation_report.json"
    save_json(result.report, report_path)

    audit_path = reports / "mllmu_hier_manual_audit_sample.json"
    save_json(result.audit_sample, audit_path)

    chains_path = reports / "mllmu_hier_semantic_chains.jsonl"
    with open(chains_path, "w") as f:
        for attr, chains in result.accepted_chains.items():
            for value, chain in sorted(chains.items()):
                f.write(json.dumps(
                    {"attribute": attr, "value": value, "chain": chain},
                    ensure_ascii=False) + "\n")

    import pandas as pd
    assoc_path = reports / "mllmu_hier_semantic_associations.parquet"
    assoc_dicts = [json.loads(a.model_dump_json()) for a in associations]
    pd.DataFrame(assoc_dicts).to_parquet(assoc_path, index=False)

    for attr in SEMANTIC_ATTRIBUTES:
        a = result.report["attributes"][attr]
        log.info(
            "%s: %d/%d distinct accepted (%.1f%%) | profiles covered %d/%d | reasons: %s",
            attr, a["accepted"], a["distinct_values"],
            100 * a["acceptance_rate"],
            a["profiles_with_accepted_hierarchy"], a["profiles_total"],
            a["rejection_reasons"],
        )
    log.info("AUDIT STATUS: %s", result.report["audit_status"])

    return {
        "report": report_path,
        "audit_sample": audit_path,
        "chains": chains_path,
        "associations": assoc_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen-assisted semantic hierarchy stage (Iteration 5)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-entities", type=int, default=None,
                        help="Limit profiles (debug runs)")
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:1 (default: auto = most free GPU)")
    parser.add_argument("--bf16", action="store_true",
                        help="Load in BF16 instead of 4-bit")
    args = parser.parse_args()
    run_semantic_stage(
        args.config, max_entities=args.max_entities,
        device=args.device, load_in_4bit=not args.bf16,
    )


if __name__ == "__main__":
    main()
