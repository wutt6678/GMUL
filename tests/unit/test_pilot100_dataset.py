"""Unit tests for the Iteration-11 pilot-100 mixed dataset machinery.

Covers:
1. Balanced target selection (per-type / per-attribute quotas,
   determinism, shortfall recording, partition-schema compatibility).
2. The image_to_text route families: nameless prompts (the entity is
   NEVER named), route/image wiring, answer levels, applicability
   gates, and validation coverage.
3. The taxonomic stratum: authoritative chains flow through query
   generation, level questions, and the failure taxonomy (sibling
   species = wrong_branch).
4. Paired entity-clustered bootstrap CIs (degenerate self-pairs,
   constant offsets, seed reproducibility, clustering, pairing).
5. Committed frozen pilot artifacts (manifest balance contract,
   freeze hashes, query report route coverage) — CI-safe: every file
   read here is committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from granunlearn.datasets.pilot import (
    PILOT_TARGET_QUOTAS,
    balance_evidence,
    select_balanced_targets,
)
from granunlearn.evaluation.paired_ci import (
    PAIRED_METRICS,
    paired_rate_diff_ci,
    row_flags,
)
from granunlearn.evaluation.query_generation import (
    IMAGE_ONLY_FAMILIES,
    UNLEARNING_FAMILIES,
    answer_level_for_family,
    family_applicable,
    generate_queries,
    validate_queries,
)
from granunlearn.evaluation.scoring import compute_metrics, score_query
from granunlearn.hierarchy.taxonomy import build_taxonomic_hierarchy
from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    ProvenanceInfo,
    SplitInfo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = REPO_ROOT / "data" / "mllmu_hier_pilot100"
REPORTS = REPO_ROOT / "data" / "reports"


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value, normalized_value=value.lower(),
        parent_id=None, metadata={})


def make_assoc(aid="a1", entity_id="e1", entity_name="Alice",
               values=None, target_level=1,
               attribute_name="residence", hierarchy_type="numeric",
               with_image=True) -> AssociationRecord:
    values = values or ["San Francisco", "California", "USA"]
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name=entity_name, attribute_name=attribute_name,
        hierarchy_type=hierarchy_type,
        levels=[_level(i, v) for i, v in enumerate(values)],
        original_level=0, target_level=target_level,
        images=[ImageRef(image_id=f"img_{aid}",
                         path=f"i/{aid}.jpg",
                         source="materialized", split="train")]
        if with_image else [],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


def make_species(species="Passer domesticus", genus="Passer",
                 family="Passeridae", common="House Sparrow",
                 with_image=True) -> AssociationRecord:
    chain = build_taxonomic_hierarchy([
        {"name": species, "rank": "species"},
        {"name": genus, "rank": "genus"},
        {"name": family, "rank": "family"},
    ], prefix="tax")
    aid = f"inat_{species.replace(' ', '_')}"
    return AssociationRecord(
        association_id=aid, dataset="inaturalist", entity_id=species,
        entity_name=common, attribute_name="taxonomic_classification",
        hierarchy_type="taxonomic", levels=chain.levels(),
        original_level=0, target_level=1,
        images=[ImageRef(image_id=f"img_{aid}_0",
                         path=f"data/raw/inaturalist/x/{aid}/000.jpg",
                         source="original", split="train")]
        if with_image else [],
        textual_context=[f"{common} ({species}) belongs to genus "
                         f"{genus}, family {family}."],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="inaturalist",
                                  source_entity_id=species,
                                  hierarchy_builder="deterministic"))


def two_entity_partition(pool):
    from granunlearn.datasets.smoke import select_target_retain
    return select_target_retain(pool, seed=42)


# ── 1. balanced target selection ──────────────────────────────────

class TestBalancedTargets:
    def _mixed_pool(self):
        pool = []
        for i in range(12):  # 12 persons x (residence, dob)
            pool.append(make_assoc(f"p{i}__res", f"p{i}",
                                   attribute_name="residence",
                                   hierarchy_type="semantic",
                                   values=["City", "Region", "Country"]))
            pool.append(make_assoc(f"p{i}__dob", f"p{i}",
                                   attribute_name="date_of_birth",
                                   values=["1990-05-04", "1990",
                                           "1990s"]))
        for sp, gen in (("Passer domesticus", "Passer"),
                        ("Passer montanus", "Passer"),
                        ("Corvus corax", "Corvus")):
            pool.append(make_species(sp, gen))
        return pool

    def test_quotas_fill_exactly(self):
        quotas = {"semantic": {"residence": 5},
                  "numeric": {"date_of_birth": 5},
                  "taxonomic": {"taxonomic_classification": 2}}
        part = select_balanced_targets(self._mixed_pool(), seed=42,
                                       quotas=quotas)
        assert part["target_counts_by_type"] == {
            "semantic": 5, "numeric": 5, "taxonomic": 2}
        assert len(part["target_association_ids"]) == 12

    def test_shortfall_recorded_never_fabricated(self):
        quotas = {"taxonomic": {"taxonomic_classification": 10}}
        part = select_balanced_targets(self._mixed_pool(), seed=42,
                                       quotas=quotas)
        fill = part["quota_fill"]["taxonomic:taxonomic_classification"]
        assert fill["available"] == 3 and fill["selected"] == 3
        assert fill["shortfall"] == 7
        assert part["target_counts_by_type"]["taxonomic"] == 3

    def test_deterministic_same_seed(self):
        pool = self._mixed_pool()
        a = select_balanced_targets(pool, seed=42)
        b = select_balanced_targets(pool, seed=42)
        assert a["target_association_ids"] == b["target_association_ids"]
        c = select_balanced_targets(pool, seed=7)
        assert a["target_association_ids"] != c["target_association_ids"]

    def test_partition_schema_complete(self):
        pool = self._mixed_pool()
        part = select_balanced_targets(pool, seed=42)
        ids = {a.association_id for a in pool}
        assert set(part["target_association_ids"]) | \
            set(part["retain_association_ids"]) == ids
        assert not (set(part["target_association_ids"])
                    & set(part["retain_association_ids"]))
        # per_entity covers every association exactly once
        seen = [i for v in part["per_entity"].values()
                for i in v["targets"] + v["retain"]]
        assert sorted(seen) == sorted(ids)

    def test_default_quotas_balance_contract(self):
        assert all(
            sum(q.values()) == 30
            for q in PILOT_TARGET_QUOTAS.values())

    def test_balance_evidence_flags_balanced(self):
        pool = self._mixed_pool()
        quotas = {"semantic": {"residence": 3},
                  "numeric": {"date_of_birth": 3},
                  "taxonomic": {"taxonomic_classification": 3}}
        part = select_balanced_targets(pool, seed=42, quotas=quotas)
        ev = balance_evidence(part, pool)
        assert ev["types_balanced"] is True
        assert ev["num_targets"] == 9


# ── 2. image_to_text route families ──────────────────────────────

class TestImageRouteFamilies:
    def _pool(self):
        pool = [
            make_assoc("e1__occ", "e1", "Alice",
                       attribute_name="occupation",
                       hierarchy_type="semantic",
                       values=["Data Scientist", "Analyst",
                               "Professional"]),
            make_assoc("e1__res", "e1", "Alice",
                       hierarchy_type="semantic",
                       values=["Sydney", "New South Wales",
                               "Oceania"]),
            make_assoc("e2__occ", "e2", "Bob",
                       attribute_name="occupation",
                       hierarchy_type="semantic",
                       values=["Consultant", "Advisor", "Expert"]),
            make_assoc("e2__height", "e2", "Bob",
                       attribute_name="height",
                       values=["180 cm", "tall-band"]),
        ]
        return pool, two_entity_partition(pool)

    def test_image_families_generated_with_images(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        fams = {q.family for q in qs}
        assert {"image_fine_direct", "image_target_direct"} <= fams
        assert "retain_same_entity_image" in fams
        assert "retain_other_entity_image" in fams

    def test_no_image_family_without_images(self):
        pool = [make_assoc(f"e{i}__res", f"e{i}", f"N{i}",
                           hierarchy_type="semantic",
                           values=[f"City{i}", f"Region{i}",
                                   f"Country{i}"], with_image=False)
                for i in range(2)]
        part = two_entity_partition(pool)
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        assert not any(q.family in IMAGE_ONLY_FAMILIES for q in qs)
        assert all(q.route == "text_to_text" for q in qs)

    def test_image_prompts_never_name_the_entity(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        img = [q for q in qs if q.route == "image_to_text"]
        assert img
        for q in img:
            ent = by_id[q.association_id]
            for nm in (ent.entity_name, ent.entity_id):
                assert nm.lower() not in q.prompt.lower()
            assert len(q.image_ids) == 1

    def test_answer_levels(self):
        a = make_assoc(values=["Fine", "Target", "Coarse"],
                       target_level=1)
        assert answer_level_for_family(a, "image_fine_direct") == 0
        assert answer_level_for_family(a, "image_target_direct") == 1
        assert answer_level_for_family(
            a, "retain_same_entity_image") == 0
        assert answer_level_for_family(
            a, "retain_other_entity_image") == 0

    def test_family_applicable_requires_image(self):
        with_img = make_assoc()
        without = make_assoc("a2", with_image=False)
        for fam in IMAGE_ONLY_FAMILIES | {"multimodal_image_text"}:
            assert family_applicable(with_img, fam)
            assert not family_applicable(without, fam)

    def test_validation_passes_and_covers_image_retain(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        facts = {}
        by_id = {a.association_id: a for a in pool}
        for rid in part["retain_association_ids"]:
            a = by_id[rid]
            facts.setdefault(a.entity_id, set()).add(
                a.levels[0].value)
        errors, stats = validate_queries(
            qs, pool, partition=part, retain_facts_by_entity=facts)
        assert errors == []
        assert stats["by_route"]["image_to_text"] > 0
        assert stats["num_retain_same_entity_image"] == \
            stats["num_retain_same_entity"]

    def test_validation_flags_named_image_prompt(self):
        """A prompt naming the entity on the image route must fail
        validation (route would collapse into image+text)."""
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        bad = [q for q in qs if q.family == "image_fine_direct"][0]
        ent = by_id[bad.association_id]
        object.__setattr__(bad, "prompt",
                           f"This is {ent.entity_name}. " + bad.prompt)
        errors, _ = validate_queries(qs, pool, partition=part)
        assert any("names the entity" in e for e in errors)


# ── 3. taxonomic stratum ─────────────────────────────────────────

class TestTaxonomicQueries:
    def _pool(self):
        pool = [
            make_species("Passer domesticus", "Passer", "Passeridae",
                         "House Sparrow"),
            make_species("Passer montanus", "Passer", "Passeridae",
                         "Eurasian Tree Sparrow"),
            make_species("Corvus corax", "Corvus", "Corvidae",
                         "Common Raven"),
            make_species("Corvus corone", "Corvus", "Corvidae",
                         "Carrion Crow", with_image=False),
        ]
        quotas = {"taxonomic": {"taxonomic_classification": 2}}
        part = select_balanced_targets(pool, seed=42, quotas=quotas)
        return pool, part

    def test_taxonomic_queries_all_routes(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        routes = {q.route for q in qs}
        assert {"text_to_text", "image_to_text"} <= routes
        # image route excludes the imageless species
        img_assoc = {q.association_id for q in qs
                     if q.route == "image_to_text"}
        assert "inat_Corvus_corone" not in img_assoc

    def test_taxonomic_level_questions(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        fine = [q for q in qs if q.family == "fine_direct"]
        assert fine
        assert "species" in fine[0].prompt.lower() or \
            "taxonomic" in fine[0].prompt.lower()
        gf = [q for q in qs if q.family == "granular_fine"]
        assert any("genus" in q.prompt.lower() for q in gf)
        gi = [q for q in qs if q.family == "granular_intermediate"]
        assert any("family" in q.prompt.lower() for q in gi)

    def test_cross_genus_species_is_wrong_branch(self):
        """Answering a species of a DIFFERENT genus is wrong_branch
        (never collapsed into 'incorrect').  Note: a same-genus
        sibling answer legitimately matches the GENUS level (the
        post-unlearning acceptable answer), so the wrong-branch probe
        must cross genera."""
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=["fine_direct"])
        by_id = {a.association_id: a for a in pool}
        preds = []
        for q in qs:
            a = by_id[q.association_id]
            other_genus = next(
                x.levels[0].value for x in pool
                if x.levels[1].value != a.levels[1].value)
            preds.append(score_query(q, a, other_genus, "exp", "MX"))
        hm = compute_hierarchy_metrics(preds, qs, pool)
        assert hm["failure_rates"]["wrong_branch"] == 1.0
        assert hm["by_hierarchy_type"]["taxonomic"]["num_queries"] > 0

    def test_same_genus_sibling_counts_at_genus_level(self):
        """A same-genus sibling answer contains the genus name and is
        therefore scored at the retained genus granularity (post
        view) — the taxonomic hierarchy semantics, not an error."""
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=["fine_direct"])
        by_id = {a.association_id: a for a in pool}
        preds = []
        for q in qs:
            a = by_id[q.association_id]
            sibling = next(
                x.levels[0].value for x in pool
                if x.entity_id != a.entity_id
                and x.levels[1].value == a.levels[1].value)
            preds.append(score_query(q, a, sibling, "exp", "MX"))
        hm = compute_hierarchy_metrics(preds, qs, pool)
        assert hm["failure_rates"]["wrong_branch"] == 0.0
        assert hm["tga"] == 1.0  # answered at the retained genus

    def test_taxonomic_validation_passes(self):
        pool, part = self._pool()
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        errors, _ = validate_queries(qs, pool, partition=part)
        assert errors == []


# ── 4. paired entity-clustered CIs ───────────────────────────────

class TestPairedCI:
    def _scored(self, pool, part, mode):
        """mode 'same' -> identical predictions; 'offset' -> state A
        answers retain probes correctly and target probes at the fine
        level while B refuses everything."""
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        preds = []
        for q in qs:
            a = by_id[q.association_id]
            fam = q.family or ""
            if fam.startswith("retain_"):
                raw = a.levels[0].value if mode != "refuse" \
                    else "I don't know."
            elif mode == "refuse":
                raw = "I don't know."
            else:
                raw = a.levels[0].value
            preds.append(score_query(q, a, raw, "exp", mode))
        return qs, preds

    def _pool(self):
        # 3 associations per entity: select_target_retain designates
        # 1 semantic + 1 numeric as targets, so the third (height)
        # stays RETAINED — required for the retain_* paired metrics.
        pool = []
        for i in range(6):
            pool.append(make_assoc(
                f"e{i}__res", f"e{i}", f"N{i}",
                attribute_name="residence", hierarchy_type="semantic",
                values=[f"City{i}", f"Region{i}", f"Country{i}"]))
            pool.append(make_assoc(
                f"e{i}__dob", f"e{i}", f"N{i}",
                attribute_name="date_of_birth",
                values=[f"198{i}-01-0{i}", f"198{i}", f"198{i}s"]))
            pool.append(make_assoc(
                f"e{i}__height", f"e{i}", f"N{i}",
                attribute_name="height",
                values=[f"18{i} cm", f"band-{i}"]))
        return pool, two_entity_partition(pool)

    def test_identical_predictions_degenerate(self):
        pool, part = self._pool()
        qs, pa = self._scored(pool, part, "same")
        _, pb = self._scored(pool, part, "same")
        fa = row_flags(pa, qs, pool)
        fb = row_flags(pb, qs, pool)
        for metric in PAIRED_METRICS:
            d = paired_rate_diff_ci(fa[metric], fb[metric],
                                    n_bootstrap=200)
            assert d is not None
            assert d["diff"] == 0.0
            assert tuple(d["ci"]) == (0.0, 0.0)

    def test_known_offset(self):
        pool, part = self._pool()
        qs, pa = self._scored(pool, part, "same")
        _, pb = self._scored(pool, part, "refuse")
        fa = row_flags(pa, qs, pool)
        fb = row_flags(pb, qs, pool)
        d = paired_rate_diff_ci(fa["retain_same"], fb["retain_same"],
                                n_bootstrap=300)
        # A answers every retain probe correctly, B refuses all:
        # paired retain_same difference must be exactly +1.0
        assert d["diff"] == 1.0
        assert tuple(d["ci"]) == (1.0, 1.0)
        assert d["num_units"] == 6  # clustered by entity

    def test_seed_reproducible(self):
        pool, part = self._pool()
        qs, pa = self._scored(pool, part, "same")
        _, pb = self._scored(pool, part, "refuse")
        fa = row_flags(pa, qs, pool)
        fb = row_flags(pb, qs, pool)
        one = paired_rate_diff_ci(fa["tga"], fb["tga"], seed=7,
                                  n_bootstrap=200)
        two = paired_rate_diff_ci(fa["tga"], fb["tga"], seed=7,
                                  n_bootstrap=200)
        assert one == two

    def test_row_flags_exclude_adversarial_and_map_families(self):
        pool, part = self._pool()
        qs, pa = self._scored(pool, part, "same")
        flags = row_flags(pa, qs, pool)
        adv = {q.query_id for q in qs if q.adversarial}
        assert not (adv & set(flags["tga"]))
        img_ret = {q.query_id for q in qs
                   if q.family == "retain_same_entity_image"}
        assert img_ret <= set(flags["retain_same"])

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            paired_rate_diff_ci({"q": (1, "e")}, {"q": (1, "e")},
                                n_bootstrap=0)

    def test_metrics_slices_include_image_route(self):
        pool, part = self._pool()
        qs, pa = self._scored(pool, part, "same")
        m = compute_metrics(pa, qs)
        assert m["image_route"]["num_queries"] > 0
        assert m["retain_same_entity_image"]["num_queries"] > 0


# ── 5. committed frozen pilot artifacts (CI-safe) ────────────────

class TestCommittedPilotArtifacts:
    def test_manifest_balance_contract(self):
        m = json.loads((PILOT_DIR / "manifest.json").read_text())
        assert m["num_entities"] == 100
        assert m["num_person_entities"] == 64
        assert m["num_species_entities"] == 36
        assert m["target_counts_by_type"] == {
            "semantic": 30, "numeric": 30, "taxonomic": 30}
        assert m["balance"]["types_balanced"] is True
        assert m["hierarchy_type_counts"]["taxonomic"] == 36

    def test_freeze_hashes_match_committed_parquets(self):
        import hashlib
        m = json.loads((PILOT_DIR / "manifest.json").read_text())
        for name, sha in m["frozen_artifact_sha256"].items():
            if not name.endswith(".parquet"):
                continue
            p = PILOT_DIR / name
            assert p.exists(), name
            assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, \
                name

    def test_query_report_routes_frozen(self):
        r = json.loads(
            (REPORTS / "mllmu_pilot100_query_report.json").read_text())
        assert r["by_route"]["text_to_text"] > 0
        assert r["by_route"]["image_to_text"] > 0
        assert r["by_route"]["image_text_to_text"] > 0
        assert r["validation"]["passed"] is True
        assert "image_to_text" in r["routes"]

    def test_partition_report_balanced(self):
        p = json.loads(
            (REPORTS / "mllmu_pilot100_target_retain.json").read_text())
        assert p["target_counts_by_type"] == {
            "semantic": 30, "numeric": 30, "taxonomic": 30}
        assert len(p["target_association_ids"]) == 90
        # retained species exist (same-stratum other-entity donors)
        tax_targets = {i for i in p["target_association_ids"]
                       if i.startswith("inat_")}
        assert len(tax_targets) == 30

    def test_inat_provenance_committed(self):
        prov = json.loads(
            (REPO_ROOT / "data" / "raw" / "inaturalist" / "pilot_v1"
             / "PROVENANCE.json").read_text())
        assert prov["num_photos"] == 432
        assert all(p.get("sha256") and p.get("license_code")
                   for p in prov["photos"])

    def test_inat_photos_pass_the_resolution_gate(self):
        """Regression guard for the Phase A thumbnail defect.

        The first fetch rewrote ``square.jpg`` -> ``medium.jpg``
        case-SENSITIVELY, so 198/432 photos (``square.JPG`` /
        ``.jpeg`` / ``.png``) were stored as 75x75 thumbnails: those
        objects have no medium rendition at all.  PROVENANCE.json is
        committed, so the gate is checkable in CI without the
        (gitignored) photo bytes.
        """
        prov_path = (REPO_ROOT / "data" / "raw" / "inaturalist"
                     / "pilot_v1" / "PROVENANCE.json")
        prov = json.loads(prov_path.read_text())
        gate = prov["resolution_gate"]
        assert gate["min_image_edge_px"] >= 200
        assert gate["replacement_policy"]
        assert "case-sensitively" in gate["defect_repaired"]
        for p in prov["photos"]:
            assert p["source_url"].endswith("/medium.jpg"), p["file_name"]
            assert "square." not in p["source_url"].lower()
            assert p["square_url"].endswith("/square.jpg"), p["file_name"]
            assert max(p["width"], p["height"]) >= \
                gate["min_image_edge_px"], p["file_name"]
            assert p["sha256"] and len(p["sha256"]) == 64
        # the gate is recorded per species, including empty reject lists
        rejects = {r["species"]: r["rejected"]
                   for r in prov["rejected_candidates"]}
        assert len(rejects) == len(prov["species_list"])
        assert len(prov["species_list"]) == 36
