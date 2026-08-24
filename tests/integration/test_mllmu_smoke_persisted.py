"""Authoritative Iteration-6 gate: validate the COMMITTED smoke artifacts.

Round-trips the persisted parquet through the canonical schemas and
re-runs the full validator (Blocker-4 fix, Iteration 6 review #3):

    queries.parquet -> QueryRecord -> validate_queries()

plus manifest/partition/image/reference invariants that Iteration 7 will
rely on when consuming these artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from granunlearn.config import _find_repo_root
from granunlearn.evaluation.query_generation import validate_queries
from granunlearn.schema import AssociationRecord, QueryRecord

REPO_ROOT = _find_repo_root(Path(__file__)) or Path.cwd()
SMOKE_DIR = REPO_ROOT / "data" / "mllmu_hier_smoke"
REPORTS = REPO_ROOT / "data" / "reports"

pytestmark = pytest.mark.skipif(
    not (SMOKE_DIR / "queries.parquet").exists(),
    reason="committed smoke artifacts not present",
)


def _records(df: pd.DataFrame) -> list[dict]:
    """Parquet round-trip turns None int columns into float NaN; map
    them back to None before pydantic validation (Iteration 7 loaders
    must do the same)."""
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@pytest.fixture(scope="module")
def artifacts():
    assoc_df = pd.read_parquet(SMOKE_DIR / "associations.parquet")
    query_df = pd.read_parquet(SMOKE_DIR / "queries.parquet")
    associations = [AssociationRecord.model_validate(r)
                    for r in _records(assoc_df)]
    queries = [QueryRecord.model_validate(r)
               for r in _records(query_df)]
    manifest = json.loads((SMOKE_DIR / "manifest.json").read_text())
    partition = json.loads(
        (REPORTS / "mllmu_smoke_target_retain.json").read_text())
    report = json.loads(
        (REPORTS / "mllmu_smoke_query_report.json").read_text())
    return {
        "associations": associations,
        "queries": queries,
        "manifest": manifest,
        "partition": partition,
        "report": report,
    }


def test_roundtrip_validation_passes(artifacts):
    """Reloaded parquet must satisfy the FULL validator — this is the
    authoritative Iteration-6 gate (Blocker-4 fix)."""
    # Reconstruct the same entity-scoped retain-fact corpus the build
    # uses so the dedupe check is non-vacuous here as well.
    by_id = {a.association_id: a for a in artifacts["associations"]}
    corpus: dict[str, set[str]] = {}
    for rid in artifacts["partition"]["retain_association_ids"]:
        a = by_id[rid]
        facts = corpus.setdefault(a.entity_id, set())
        facts.add(a.levels[0].value)
        facts.update(a.textual_context)
    errors, stats = validate_queries(
        artifacts["queries"], artifacts["associations"],
        partition=artifacts["partition"],
        retain_facts_by_entity=corpus)
    assert errors == [], errors[:10]
    assert stats["num_errors"] == 0


def test_association_count(artifacts):
    assert len(artifacts["associations"]) == 68


def test_manifest_partition_invariants(artifacts):
    m = artifacts["manifest"]
    targets = set(m["target_association_ids"])
    retains = set(m["retain_association_ids"])
    all_ids = {a.association_id for a in artifacts["associations"]}
    assert targets & retains == set()          # disjoint
    assert targets | retains == all_ids        # exhaustive over all 68
    assert len(targets) == 20 and len(retains) == 48
    assert m["target_counts_by_type"] == {"semantic": 10, "numeric": 10}


def test_all_query_references_resolve(artifacts):
    assoc_ids = {a.association_id for a in artifacts["associations"]}
    for q in artifacts["queries"]:
        assert q.association_id in assoc_ids, q.query_id
        if q.target_association_id is not None:
            assert q.target_association_id in assoc_ids, q.query_id


def test_multimodal_image_ids_and_files_resolve(artifacts):
    by_assoc = {a.association_id: a for a in artifacts["associations"]}
    mm = [q for q in artifacts["queries"]
          if q.family == "multimodal_image_text"]
    assert mm, "smoke set must contain multimodal probes"
    for q in mm:
        assoc = by_assoc[q.association_id]
        known_ids = {img.image_id for img in assoc.images}
        assert set(q.image_ids) <= known_ids, q.query_id
        for img in assoc.images:
            path = Path(img.path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            assert path.exists(), f"missing image {img.path}"


def test_row_counts_match_report(artifacts):
    assert len(artifacts["queries"]) == artifacts["report"]["num_queries"]
    by_split = {s: sum(1 for q in artifacts["queries"] if q.split == s)
                for s in ("train", "val", "test")}
    assert by_split == artifacts["report"]["by_split"]
    assert len(artifacts["associations"]) == \
        artifacts["report"]["num_associations_total"]


def test_donor_pairs_are_retained(artifacts):
    retains = set(artifacts["manifest"]["retain_association_ids"])
    pairs = artifacts["report"]["retain_other_entity_donor_pairs"]
    assert pairs and all(p["donor_association_id"] in retains
                         for p in pairs)
