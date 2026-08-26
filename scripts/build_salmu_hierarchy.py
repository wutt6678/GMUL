"""Build the SALMU hierarchical extension artifacts (Iteration 10).

    python scripts/build_salmu_hierarchy.py

Produces, from the RELEASED (unmodified) SALMUBench metadata:
* data/salmu_original/manifest.json        — adapter pinning of the
                                               released artifacts
* data/salmu_hierarchical/attribute_inventory.json
* data/salmu_hierarchical/associations.json — core hierarchies per
                                               persona (city/job/blood)
* data/salmu_aux_redaction/manifest.json    — redaction-only identifiers
                                               stay OUT of the core task
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from granunlearn.config import _find_repo_root
from granunlearn.logging_utils import setup_logger
from granunlearn.salmu.adapter import (
    load_original_metadata,
    write_original_manifest,
)
from granunlearn.salmu.hierarchy import (
    AUX_REDACTION,
    build_persona_hierarchies,
    generalized_caption,
    write_attribute_inventory,
)

log = setup_logger("build_salmu_hierarchy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SALMU hierarchical extension artifacts")
    args = parser.parse_args()

    repo_root = _find_repo_root(Path.cwd()) or Path.cwd()
    original_dir = repo_root / "data" / "salmu_original"
    hier_dir = repo_root / "data" / "salmu_hierarchical"
    aux_dir = repo_root / "data" / "salmu_aux_redaction"

    # 1. Pin the released artifacts (read-only adapter).
    write_original_manifest(original_dir / "manifest.json")

    # 2. Load original metadata and derive core hierarchies.
    meta = load_original_metadata()
    identities = meta["identities"]
    hierarchies = build_persona_hierarchies(identities)
    inventory = write_attribute_inventory(
        hier_dir / "attribute_inventory.json", identities)

    # Self-check: generalized captions build for every hierarchy/level.
    n_caps = 0
    for iid, attrs in hierarchies.items():
        name = identities[iid]["name"]
        for attr, h in attrs.items():
            for lv in range(1, len(h["levels"])):
                generalized_caption(name, attr, lv, h["levels"][lv])
                n_caps += 1
    log.info("Generalized caption templates verified for %d level-values",
             n_caps)

    with open(hier_dir / "associations.json", "w") as f:
        json.dump(hierarchies, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d persona hierarchies -> %s", len(hierarchies),
             hier_dir / "associations.json")

    # 3. Auxiliary redaction-only identifiers: referenced, never part of
    #    the core hierarchical task.
    aux_manifest = {
        "purpose": (
            "Identifiers that stay OUTSIDE the core hierarchical "
            "unlearning task (Iteration 10 agreement). They are listed "
            "only so future redaction experiments can find them in the "
            "RELEASED identities_metadata.json; no derived data is "
            "built here."),
        "attributes": list(AUX_REDACTION),
        "source": "identities_metadata.json in "
                  "cvc-mmu/salmubench-512-redistributed (read-only)",
        "num_personas": len(identities),
    }
    aux_dir.mkdir(parents=True, exist_ok=True)
    with open(aux_dir / "manifest.json", "w") as f:
        json.dump(aux_manifest, f, indent=2, ensure_ascii=False)
    log.info("Aux redaction manifest -> %s", aux_dir / "manifest.json")

    fb = inventory["hierarchies"]["job"]
    log.info("Job taxonomy: %d unique jobs, %d fallback",
             fb["num_unique_jobs"], fb["num_fallback_jobs"])
    log.info("SALMU hierarchical extension built.")


if __name__ == "__main__":
    main()
