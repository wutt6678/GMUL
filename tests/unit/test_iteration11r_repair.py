"""Iteration 11R repair tests — CPU-only, CI-safe.

Guards the four defects the Iteration 11 review found:

1. **Visual split** — every image query got ``assoc.images[0]``, the same
   photograph training consumed, so the image route measured unseen
   wording over a SEEN photograph.  These tests pin the split-aware
   assignment (disjoint train/val/test pools, deterministic round-robin,
   single-portrait entities flagged rather than disguised) and re-verify
   it against the committed ``pilot100_v2`` artifacts.
2. **No-retraining proof** — the repair is only free if the six
   training/unlearning JSONLs did not move.  Their v1 SHA-256s are pinned
   here, so a future rebuild that shifts a training photograph fails the
   suite instead of silently invalidating every adapter.
3. **Unprovenanced parquet reuse** — one test per fingerprint dimension
   (adapter bytes, base-model revision, dataset version and artifact
   hashes, every generation-config key, code commit and module hashes)
   plus the missing-sidecar case, which is every file Iteration 11 wrote.
4. **Intersection-only coverage** — duplicates, foreign rows, missing
   rows, short counts and mislabelled rows must all be refusals.

Plus the two evidence repairs: paired FILR / over-forgetting CIs, whose
row-level point estimates must equal ``hierarchy_metrics`` exactly, and
the image-provenance strata, which must both be non-empty with the
held-out stratum containing no trained-on photograph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from granunlearn.evaluation.image_splits import (
    HELD_OUT_PHOTO,
    IMAGE_STRATA,
    REPAIRED_DATASET_VERSION,
    RESERVED_TRAINING_INDEX,
    SEEN_PHOTO_UNSEEN_WORDING,
    V1_TRAINING_JSONL_SHA256,
    assign_split_images,
    image_for_split,
    image_pools,
    image_stratum,
    photo_labels,
    relabel_image_splits,
    validate_image_splits,
)
from granunlearn.evaluation.paired_ci import (
    PAIRED_METRICS,
    paired_metrics_report,
    paired_rate_diff_ci,
    row_flags,
)
from granunlearn.evaluation.prediction_provenance import (
    GENERATION_CONFIG_KEYS,
    SUPERSEDED_V1_COMMIT,
    PredictionFingerprint,
    dataset_version,
    read_sidecar,
    sidecar_path,
    validate_prediction_coverage,
    verify_sidecar,
    write_sidecar,
)
from granunlearn.evaluation.query_generation import (
    UNLEARNING_FAMILIES,
    generate_queries,
)
from granunlearn.evaluation.scoring import score_query
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

#: v1 (commit 3850461) hashes of the six files the reference states and
#: every unlearning candidate were trained from.  ``images[0]`` is reserved
#: as the training photograph precisely so these do not change; if a future
#: edit moves a training photograph, these assertions are what stop the
#: suite rather than quietly invalidating 19 adapters.
V1_TRAINING_SHA256 = {
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

#: Artifacts that must also not move: the entity list and the balanced
#: 90/387 target/retain partition are what every adapter was trained
#: against, and they are upstream of the JSONLs.
V1_PARTITION_SHA256 = {
    "mllmu_pilot100_entities.json":
        "7c2576f45e90a25c15218a340d1fd5d74c563a1142aa2379675bcedb05edf0a6",
    "mllmu_pilot100_target_retain.json":
        "62bf8137b1332f8927d28de630e98112a22f9d22b437c818173da3fbab3fa3ab",
}


def _partition(assocs: list[AssociationRecord]) -> dict:
    """Every association a target.

    The image-assignment rules under test do not depend on the
    target/retain partition, and building the real balanced one would pull
    in the selection machinery for no coverage.
    """
    return {"target_association_ids": [a.association_id for a in assocs],
            "retain_association_ids": []}


def _qs(assocs: list[AssociationRecord], seed: int = 42):
    """Every unlearning family, one query per (association, family, split)."""
    return generate_queries(assocs, _partition(assocs), seed=seed,
                            families=list(UNLEARNING_FAMILIES))


def _level(i: int, value: str) -> HierarchyLevel:
    return HierarchyLevel(
        level=i, canonical_id=f"l{i}:{value.lower().replace(' ', '_')}",
        value=value, normalized_value=value.lower(),
        parent_id=None, metadata={})


def _img(aid: str, i: int, split: str = "train") -> ImageRef:
    return ImageRef(image_id=f"img_{aid}_{i:03d}",
                    path=f"data/raw/x/{aid}/{i:03d}.jpg",
                    source="original", split=split)


def make_species(aid: str = "inat_sp", species: str = "Passer domesticus",
                 genus: str = "Passer", family: str = "Passeridae",
                 num_images: int = 12) -> AssociationRecord:
    """A taxonomic association shaped like the frozen iNat stratum."""
    chain = build_taxonomic_hierarchy([
        {"name": species, "rank": "species"},
        {"name": genus, "rank": "genus"},
        {"name": family, "rank": "family"},
    ], prefix="tax")
    return AssociationRecord(
        association_id=aid, dataset="inaturalist", entity_id=species,
        entity_name=species, attribute_name="taxonomic_classification",
        hierarchy_type="taxonomic", levels=chain.levels(),
        original_level=0, target_level=1,
        images=[_img(aid, i) for i in range(num_images)],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="inaturalist",
                                  source_entity_id=species,
                                  hierarchy_builder="deterministic"))


def make_person(aid: str = "p0__res", entity_id: str = "p0",
                num_images: int = 1) -> AssociationRecord:
    """An MLLMU-shaped association: ONE portrait shared by all its
    associations, so no held-out photograph exists for it."""
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name="Alice", attribute_name="residence",
        hierarchy_type="semantic",
        levels=[_level(i, v) for i, v in
                enumerate(["San Francisco", "California", "USA"])],
        original_level=0, target_level=1,
        images=[_img(entity_id, i) for i in range(num_images)],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


def _frozen():
    """The committed pilot100_v2 queries + associations."""
    from granunlearn.evaluation.reference_eval import (
        load_associations_parquet, load_queries_parquet)
    return (load_queries_parquet(PILOT_DIR / "queries.parquet"),
            load_associations_parquet(PILOT_DIR / "associations.parquet"))


# ── 1. split-aware photograph assignment ─────────────────────────

class TestSplitAwareImageAssignment:
    def test_pools_reserve_the_training_photograph(self):
        a = make_species(num_images=12)
        pools = image_pools(a, seed=42)
        training = a.images[RESERVED_TRAINING_INDEX].image_id
        assert [im.image_id for im in pools["train"]] == [training]
        assert len(pools["val"]) == 5
        assert len(pools["test"]) == 6
        # every photograph is accounted for exactly once
        all_ids = [im.image_id for s in ("train", "val", "test")
                   for im in pools[s]]
        assert len(all_ids) == len(set(all_ids)) == 12
        assert set(all_ids) == {im.image_id for im in a.images}

    def test_pools_are_pairwise_disjoint(self):
        a = make_species(num_images=12)
        pools = image_pools(a, seed=42)
        ids = {s: {im.image_id for im in pools[s]}
               for s in ("train", "val", "test")}
        assert not (ids["train"] & ids["val"])
        assert not (ids["train"] & ids["test"])
        assert not (ids["val"] & ids["test"])

    def test_assignment_is_deterministic_and_seed_sensitive(self):
        a = make_species(num_images=12)
        assert image_pools(a, 42) == image_pools(a, 42)
        first, seen1 = image_for_split(a, "test", 42, 0)
        again, seen2 = image_for_split(a, "test", 42, 0)
        assert (first.image_id, seen1) == (again.image_id, seen2)
        # a different seed must reshuffle the pools: the assignment is
        # seeded, not a fixed slice of the adapter's photo order
        other = image_pools(a, 7)
        assert ({im.image_id for im in other["val"]} !=
                {im.image_id for im in image_pools(a, 42)["val"]})

    def test_hash_ordering_is_not_process_salt_dependent(self):
        """``hash()`` is salted per process; the pools must not be."""
        a = make_species(num_images=12)
        ids = [im.image_id for im in image_pools(a, 42)["test"]]
        # recomputed in-process from the documented stable hash
        from granunlearn.evaluation.image_splits import _stable_hash
        rest = [im for im in a.images[1:]]
        rest.sort(key=lambda im: _stable_hash(42, a.association_id,
                                              im.image_id))
        assert ids == [im.image_id for im in rest[5:]]

    def test_train_split_always_gets_the_trained_photograph(self):
        a = make_species(num_images=12)
        for slot in range(4):
            img, seen = image_for_split(a, "train", 42, slot)
            assert img.image_id == a.images[0].image_id
            assert seen is True

    def test_val_and_test_never_get_the_trained_photograph(self):
        a = make_species(num_images=12)
        for split in ("val", "test"):
            for slot in range(12):
                img, seen = image_for_split(a, split, 42, slot)
                assert img.image_id != a.images[0].image_id
                assert seen is False

    def test_round_robin_cycles_the_pool_in_query_order(self):
        """More image queries than pool photographs is expected (6 test
        photos, up to 10 test image queries per species), so the invariant
        that matters is disjointness from TRAINING, not distinctness
        between two test queries — and the cycling must be reproducible."""
        a = make_species(num_images=12)
        pool = [im.image_id for im in image_pools(a, 42)["test"]]
        got = [image_for_split(a, "test", 42, s)[0].image_id
               for s in range(len(pool) * 2 + 1)]
        assert got == pool * 2 + pool[:1]

    def test_single_photograph_entity_keeps_the_trained_one(self):
        a = make_person(num_images=1)
        pools = image_pools(a, 42)
        only = a.images[0].image_id
        for split in ("train", "val", "test"):
            assert [im.image_id for im in pools[split]] == [only]
            img, seen = image_for_split(a, split, 42, 0)
            assert img.image_id == only
            assert seen is True, (
                "a single-portrait entity has no held-out photograph, so "
                "it must be flagged rather than silently labelled held out")

    def test_photo_labels_follow_use_not_the_adapter_preassignment(self):
        """``deterministic_image_splits`` pre-assigned 8/2/2 per species
        before anyone knew which photograph training would consume; the
        label must describe USE, so images[0] is train whatever it said."""
        a = make_species(num_images=12)
        # poison the pre-assignment: label the trained photo 'test'
        poisoned = a.model_copy(update={"images": [
            im.model_copy(update={"split": "test"}) if i == 0
            else im.model_copy(update={"split": "train"})
            for i, im in enumerate(a.images)]})
        labels = photo_labels(poisoned, 42)
        assert labels[poisoned.images[0].image_id] == "train"
        pools = image_pools(poisoned, 42)
        for split in ("val", "test"):
            for im in pools[split]:
                assert labels[im.image_id] == split

    def test_relabel_writes_the_use_based_split_onto_the_records(self):
        a = make_species(num_images=12)
        relabeled = relabel_image_splits(a, 42)
        assert relabeled.images[0].split == "train"
        counts = {s: sum(1 for im in relabeled.images if im.split == s)
                  for s in ("train", "val", "test")}
        assert counts == {"train": 1, "val": 5, "test": 6}
        # the original is untouched (model_copy, not in-place mutation)
        assert all(im.split == "train" for im in a.images)
        # nothing else about the association moved
        assert relabeled.levels == a.levels
        assert relabeled.association_id == a.association_id

    def test_assign_split_images_sets_both_new_query_fields(self):
        a = make_species(num_images=12)
        p = make_person()
        qs = _qs([a, p])
        out = assign_split_images(qs, [a, p], seed=42)
        img_qs = [q for q in out if q.image_ids]
        assert img_qs, "the fixture must produce image queries"
        by_id = {a.association_id: a, p.association_id: p}
        for q in img_qs:
            assoc = by_id[q.association_id]
            assert q.image_split in ("train", "val", "test")
            assert q.image_split == photo_labels(
                assoc, 42)[q.image_ids[0]]
            training = assoc.images[RESERVED_TRAINING_INDEX].image_id
            assert q.image_seen_in_training == (q.image_ids[0] == training)
            if q.split != "train" and len(assoc.images) > 1:
                assert q.image_seen_in_training is False
        # text-only queries carry no image metadata at all
        for q in out:
            if not q.image_ids:
                assert q.image_split is None
                assert q.image_seen_in_training is False

    def test_assign_is_reproducible_and_order_independent(self):
        a = make_species(num_images=12)
        qs = _qs([a])
        once = assign_split_images(qs, [a], seed=42)
        twice = assign_split_images(list(reversed(qs)), [a], seed=42)
        # the round-robin is keyed on query_id, so reversing the input list
        # must not change which photograph each query gets
        by_id = {q.query_id: q.image_ids for q in twice}
        assert [q.image_ids for q in once] == \
            [by_id[q.query_id] for q in once]

    def test_validate_accepts_a_correct_assignment(self):
        a = make_species(num_images=12)
        p = make_person()
        qs = assign_split_images(
            _qs([a, p]),
            [a, p], seed=42)
        assert validate_image_splits(qs, [a, p], seed=42) == []

    def test_validate_rejects_the_iteration_11_assignment(self):
        """The defect itself, as a regression: images[0] for every split
        must fail the disjointness rule for any multi-photo entity."""
        a = make_species(num_images=12)
        qs = _qs([a])
        legacy = [q.model_copy(update=(
            {"image_ids": [a.images[0].image_id],
             "image_split": "train",
             "image_seen_in_training": True} if q.image_ids else {}))
            for q in qs]
        errors = validate_image_splits(legacy, [a], seed=42)
        assert errors, "serving the training photograph to val/test must " \
                       "not validate"
        assert any("served the training photograph" in e for e in errors)

    def test_validate_rejects_a_held_out_claim_on_a_single_portrait(self):
        p = make_person(num_images=1)
        qs = _qs([p])
        lied = [q.model_copy(update=(
            {"image_seen_in_training": False} if q.image_ids else {}))
            for q in qs]
        errors = validate_image_splits(lied, [p], seed=42)
        assert any("single-photograph entity" in e for e in errors)

    def test_validate_rejects_a_foreign_photograph(self):
        a = make_species(num_images=12)
        other = make_species(aid="inat_other", species="Corvus corax",
                             genus="Corvus", family="Corvidae")
        qs = assign_split_images(
            _qs([a]),
            [a], seed=42)
        stolen = [q.model_copy(update=(
            {"image_ids": [other.images[3].image_id]} if q.image_ids
            else {})) for q in qs]
        errors = validate_image_splits(stolen, [a], seed=42)
        assert any("is not on" in e for e in errors)

    def test_validate_rejects_image_metadata_on_a_text_query(self):
        a = make_species(num_images=12)
        qs = assign_split_images(
            _qs([a]),
            [a], seed=42)
        text = next(q for q in qs if not q.image_ids)
        bad = [text.model_copy(update={"image_split": "test"})]
        assert any("text-only query carries image metadata" in e
                   for e in validate_image_splits(bad, [a], seed=42))



# ── 2. the committed pilot100_v2 artifacts ───────────────────────

@pytest.fixture(scope="module")
def frozen():
    """The committed pilot100_v2 queries + associations, loaded once."""
    return _frozen()


class TestFrozenVisualSplitV2:
    """Re-verified against the committed dataset on every CI run, so the
    repaired split cannot silently regress in a later rebuild."""

    def test_dataset_declares_v2(self):
        assert dataset_version(PILOT_DIR) == "pilot100_v2"
        man = json.loads((PILOT_DIR / "manifest.json").read_text())
        assert man["version"] == "pilot100_v2"

    def test_the_repair_validates_with_no_exceptions(self, frozen):
        queries, associations = frozen
        assert validate_image_splits(queries, associations, seed=42) == []

    def test_no_val_or_test_query_of_a_multi_photo_entity_is_a_seen_photo(
            self, frozen):
        """The claim under repair: a held-out split must not be served the
        photograph training consumed, wherever a spare photograph exists."""
        queries, associations = frozen
        by_id = {a.association_id: a for a in associations}
        offenders = [
            q.query_id for q in queries
            if q.image_ids and q.split != "train" and q.image_seen_in_training
            and len(by_id[q.association_id].images) > 1]
        assert offenders == []

    def test_single_photo_entities_are_flagged_never_disguised(self, frozen):
        queries, associations = frozen
        single = {a.association_id for a in associations
                  if len(a.images) == 1}
        assert single, "the pilot set must contain single-portrait entities"
        seen = [q for q in queries
                if q.image_ids and q.association_id in single]
        assert seen, "single-photo entities must still be evaluated"
        assert all(q.image_seen_in_training for q in seen)
        assert all(q.image_split == "train" for q in seen)
        assert all(image_stratum(q) == SEEN_PHOTO_UNSEEN_WORDING
                   for q in seen)

    def test_every_multi_photo_entity_labels_its_photos_by_use(self, frozen):
        _, associations = frozen
        multi = [a for a in associations if len(a.images) > 1]
        assert multi
        for a in multi:
            labels = photo_labels(a, 42)
            assert labels[a.images[RESERVED_TRAINING_INDEX].image_id] == \
                "train"
            for im in a.images:
                assert im.split == labels[im.image_id], (
                    f"{a.association_id}/{im.image_id}: ImageRef.split must "
                    f"describe USE, not the adapter's 60/20/20 "
                    f"pre-assignment")
            counts = {s: sum(1 for im in a.images if im.split == s)
                      for s in ("train", "val", "test")}
            assert counts["train"] == 1
            assert counts["val"] + counts["test"] == len(a.images) - 1

    def test_train_and_eval_photographs_are_disjoint_dataset_wide(self,
                                                                  frozen):
        """Per association AND across the whole dataset: the union of every
        val/test photograph must not contain any training photograph of the
        same entity."""
        queries, associations = frozen
        used: dict[tuple[str, str], set] = {}
        for q in queries:
            if q.image_ids:
                used.setdefault((q.association_id, q.split), set()
                                ).update(q.image_ids)
        for a in associations:
            if len(a.images) < 2:
                continue
            training = {a.images[RESERVED_TRAINING_INDEX].image_id}
            for split in ("val", "test"):
                assert not (training & used.get((a.association_id, split),
                                                set())), a.association_id
            assert not (used.get((a.association_id, "val"), set())
                        & used.get((a.association_id, "test"), set()))

    def test_image_ids_all_belong_to_their_own_association(self, frozen):
        queries, associations = frozen
        by_id = {a.association_id: a for a in associations}
        for q in queries:
            if not q.image_ids:
                continue
            known = {im.image_id for im in by_id[q.association_id].images}
            assert set(q.image_ids) <= known, q.query_id
            assert len(q.image_ids) == 1

    def test_both_image_strata_exist_and_are_the_documented_sizes(self,
                                                                  frozen):
        queries, _ = frozen
        from collections import Counter
        strata = Counter(image_stratum(q) for q in queries if q.image_ids)
        assert set(strata) == set(IMAGE_STRATA)
        # v1 had NO held-out stratum at all: 0 queries, because every
        # image query carried images[0]
        assert strata[HELD_OUT_PHOTO] == 252
        assert strata[SEEN_PHOTO_UNSEEN_WORDING] == 1989
        test_strata = Counter(image_stratum(q) for q in queries
                              if q.image_ids and q.split == "test")
        assert test_strata[HELD_OUT_PHOTO] == 126
        assert test_strata[SEEN_PHOTO_UNSEEN_WORDING] == 621

    def test_the_held_out_stratum_is_only_the_multi_photo_stratum(self,
                                                                  frozen):
        queries, associations = frozen
        by_id = {a.association_id: a for a in associations}
        held = [q for q in queries if image_stratum(q) == HELD_OUT_PHOTO]
        assert held
        assert all(len(by_id[q.association_id].images) > 1 for q in held)
        assert all(q.split != "train" for q in held)
        # only the iNaturalist stratum has spare photographs, so it is the
        # only one that can populate the held-out stratum
        assert all(by_id[q.association_id].dataset == "inaturalist"
                   for q in held)

    def test_manifest_records_the_repair(self, frozen):
        queries, associations = frozen
        man = json.loads((PILOT_DIR / "manifest.json").read_text())
        distinct_used = {i for q in queries for i in q.image_ids}
        assert man["num_unique_images_used_by_queries"] == len(distinct_used)
        # Iteration 11 used exactly 100 distinct photographs (one per
        # entity) and left 396 of the 496 unreferenced; the repair must be
        # a large increase, and the number is pinned so a rebuild that
        # quietly reverts to images[0] fails here.
        assert len(distinct_used) == 342
        assert len(distinct_used) > 100
        from collections import Counter
        assert man["image_strata"] == dict(
            Counter(image_stratum(q) for q in queries if q.image_ids))
        policy = man["image_split_policy"]
        assert policy["reserved_training_index"] == RESERVED_TRAINING_INDEX
        for key in ("rule", "override_of_adapter_preassignment",
                    "single_photograph_entities", "within_split_repetition",
                    "defect_repaired"):
            assert policy.get(key), key

    def test_query_report_records_the_image_split(self, frozen):
        rep = json.loads(
            (REPORTS / "mllmu_pilot100_query_report.json").read_text())
        block = rep["image_split"]
        assert block["validation_passed"] is True
        assert block["num_distinct_photographs_used"] == 342
        assert block["num_distinct_photographs_available"] == 496
        assert set(block["strata"]) == set(IMAGE_STRATA)
        assert block["strata"][HELD_OUT_PHOTO] == 252
        assert "must never be pooled" in block["note"]

    def test_the_iNat_photograph_inventory_is_fully_accounted_for(self,
                                                                  frozen):
        """All 432 iNaturalist photos must be either the reserved training
        photograph or a member of exactly one held-out pool — the repair
        may not drop photographs on the floor."""
        _, associations = frozen
        inat = [a for a in associations if a.dataset == "inaturalist"]
        photos = {im.image_id for a in inat for im in a.images}
        assert len(photos) == 432
        accounted = set()
        for a in inat:
            pools = image_pools(a, 42)
            ids = [im.image_id for s in ("train", "val", "test")
                   for im in pools[s]]
            assert len(ids) == len(set(ids)) == len(a.images)
            accounted.update(ids)
        assert accounted == photos


# ── 3. the no-retraining proof ───────────────────────────────────

class TestNoRetrainingRequired:
    """The whole 11R plan is conditional on this: reserving ``images[0]``
    as the training photograph must leave every trained-on byte alone.
    If one of these hashes moves, the correct response is to retrain, not
    to keep reporting adapters that were fit on different data."""

    @pytest.mark.parametrize("rel", sorted(V1_TRAINING_SHA256))
    def test_training_jsonl_is_byte_identical_to_v1(self, rel):
        p = PILOT_DIR / rel
        assert p.exists(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == \
            V1_TRAINING_SHA256[rel], (
                f"{rel} changed: the visual-split repair moved a training "
                f"photograph, so every adapter trained from it is stale")

    @pytest.mark.parametrize("name", sorted(V1_PARTITION_SHA256))
    def test_partition_artifacts_are_byte_identical_to_v1(self, name):
        p = REPORTS / name
        assert p.exists(), name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == \
            V1_PARTITION_SHA256[name], name

    def test_six_jsonls_and_only_those_were_pinned(self):
        assert len(V1_TRAINING_SHA256) == 6
        assert set(V1_TRAINING_SHA256) == {
            "training/MF.jsonl", "training/MG.jsonl", "training/MN.jsonl",
            "unlearning/fine_target.jsonl", "unlearning/target_level.jsonl",
            "unlearning/retain.jsonl"}

    def test_the_pins_here_and_in_the_build_gate_agree(self):
        """Two independent copies of the same pins, on purpose.

        The build gate (``image_splits.V1_TRAINING_JSONL_SHA256``) hard-fails
        a rebuild that moves a training byte, and this file re-states the
        same values literally.  Neither can be edited to make a drifting
        rebuild pass without the other failing — and a literal copy works
        in CI's shallow clone, where ``git show 3850461:...`` would not.
        """
        assert V1_TRAINING_SHA256 == V1_TRAINING_JSONL_SHA256
        assert REPAIRED_DATASET_VERSION == "pilot100_v2"

    def test_provenance_record_agrees_with_these_pins(self):
        """The committed provenance record must carry the same verdict, so
        a reader of the record and a reader of the suite cannot disagree."""
        rep = json.loads(
            (REPORTS / "mllmu_pilot100_reference_provenance.json").read_text())
        assert rep["dataset_version"] == "pilot100_v2"
        trans = rep["dataset_transition"]
        assert trans["from"] == "pilot100_v1"
        assert trans["to"] == "pilot100_v2"
        assert trans["v1_source_commit"] == SUPERSEDED_V1_COMMIT
        assert trans["training_jsonls_byte_identical"] is True
        assert trans["retraining_required"] is False
        for rel, sha in V1_TRAINING_SHA256.items():
            key = f"data/mllmu_hier_pilot100/{rel}"
            assert trans["v1_artifact_sha256"][key] == sha
            # the live hash recorded beside it must be the SAME hash
            assert rep["dataset_hashes_sha256"][key] == sha
            assert key in trans["artifacts_unchanged"]
        # the artifacts that DID move are exactly the evaluation ones
        for key in trans["artifacts_changed"]:
            assert "training/" not in key and "unlearning/" not in key, key


class TestBuildFreezeGate:
    """The builders must refuse to emit moved training bytes, not merely
    let a test notice afterwards."""

    def test_the_gate_passes_on_the_committed_v2_dataset(self):
        from granunlearn.evaluation.image_splits import (
            assert_no_training_drift)
        measured = assert_no_training_drift(
            PILOT_DIR, tuple(V1_TRAINING_SHA256), "pilot100_v2")
        assert set(measured) == set(V1_TRAINING_SHA256)
        assert measured == V1_TRAINING_SHA256

    def test_the_gate_raises_when_a_training_byte_moves(self, tmp_path):
        from granunlearn.evaluation.image_splits import (
            assert_no_training_drift)
        (tmp_path / "training").mkdir()
        (tmp_path / "unlearning").mkdir()
        for rel in V1_TRAINING_SHA256:
            (tmp_path / rel).write_bytes(
                (PILOT_DIR / rel).read_bytes())
        # one file re-picks its training photograph
        (tmp_path / "training" / "MF.jsonl").write_bytes(b"a different photo")
        with pytest.raises(RuntimeError) as exc:
            assert_no_training_drift(
                tmp_path, tuple(V1_TRAINING_SHA256), "pilot100_v2")
        msg = str(exc.value)
        assert "training/MF.jsonl" in msg
        assert "no-retraining claim is false" in msg
        assert "retrain" in msg
        # exactly the one file that moved is blamed
        assert msg.count(".jsonl:") == 1
        assert "training/MG.jsonl" not in msg

    def test_the_gate_only_applies_to_the_repaired_version(self, tmp_path):
        """A future dataset version that deliberately changes the entity
        set is allowed to retrain; the gate must not block it forever."""
        from granunlearn.evaluation.image_splits import (
            assert_no_training_drift)
        (tmp_path / "training").mkdir()
        (tmp_path / "training" / "MF.jsonl").write_bytes(b"anything")
        assert assert_no_training_drift(
            tmp_path, ("training/MF.jsonl",), "pilot100_v3") == {}

    def test_the_gate_checks_only_files_that_exist(self, tmp_path):
        """Each builder writes its own subset, so a missing sibling is not
        drift — the state-datasets build runs before the groups build."""
        from granunlearn.evaluation.image_splits import (
            assert_no_training_drift)
        (tmp_path / "training").mkdir()
        (tmp_path / "training" / "MF.jsonl").write_bytes(
            (PILOT_DIR / "training" / "MF.jsonl").read_bytes())
        measured = assert_no_training_drift(
            tmp_path, tuple(V1_TRAINING_SHA256), "pilot100_v2")
        assert set(measured) == {"training/MF.jsonl"}

    def test_both_builders_call_the_gate(self):
        """The gate is only real if the builders invoke it; reading the
        source is the CI-safe way to assert that without running a build."""
        for script, symbol in (("build_state_datasets.py", "STATE_JSONLS"),
                               ("build_unlearning_groups.py",
                                "UNLEARNING_GROUP_JSONLS")):
            src = (REPO_ROOT / "scripts" / script).read_text()
            assert "assert_no_training_drift" in src, script
            assert symbol in src, script


# ── 4. prediction sidecars: reuse is a verified decision ─────────

def _fp(**over) -> PredictionFingerprint:
    """A fully populated fingerprint; ``over`` perturbs one dimension."""
    base = dict(
        experiment_id="mllmu_pilot100_iter11",
        checkpoint_id="MF",
        adapter_sha256="a1" * 32,
        base_model_revision="b2" * 20,
        dataset={"version": "pilot100_v2",
                 "artifacts_sha256": {"queries.parquet": "c3" * 32,
                                      "manifest.json": "d4" * 32},
                 "data_dir": "data/mllmu_hier_pilot100"},
        generation_config=dict(zip(GENERATION_CONFIG_KEYS,
                                   (8, 8, 96, False, 384 * 384, 1536))),
        code={"git_commit": "e5" * 20, "git_dirty": False,
              "modules_sha256": {
                  "src/granunlearn/evaluation/scoring.py": "f6" * 32}},
        created_utc="2026-09-05T00:00:00+00:00",
        num_rows=2259)
    base.update(over)
    return PredictionFingerprint(**base)


@pytest.fixture
def pq(tmp_path):
    """A prediction parquet with a matching sidecar beside it."""
    path = tmp_path / "predictions_test_MF.parquet"
    path.write_bytes(b"parquet-bytes-are-irrelevant-to-the-contract")
    write_sidecar(path, _fp())
    return path


class TestPredictionSidecars:
    def test_sidecar_lives_beside_its_parquet(self, pq):
        assert sidecar_path(pq) == Path(str(pq) + ".provenance.json")
        assert sidecar_path(pq).parent == pq.parent

    def test_matching_fingerprint_is_reusable(self, pq):
        assert verify_sidecar(pq, _fp()) == []
        assert read_sidecar(pq) == _fp().to_dict()

    def test_a_parquet_with_no_sidecar_is_refused(self, tmp_path):
        """Every file Iteration 11 wrote is in this state, so this is the
        case that actually decides whether the v1 predictions get reused."""
        bare = tmp_path / "predictions_test_MG.parquet"
        bare.write_bytes(b"v1 bytes")
        reasons = verify_sidecar(bare, _fp(checkpoint_id="MG"))
        assert len(reasons) == 1
        assert "no provenance sidecar" in reasons[0]
        assert "cannot be attributed" in reasons[0]

    def test_a_missing_parquet_is_refused(self, tmp_path):
        reasons = verify_sidecar(tmp_path / "absent.parquet", _fp())
        assert reasons == ["missing parquet absent.parquet"]

    def test_an_unreadable_sidecar_is_refused_not_ignored(self, pq):
        sidecar_path(pq).write_text("{not json")
        assert read_sidecar(pq) is None
        reasons = verify_sidecar(pq, _fp())
        assert len(reasons) == 1
        assert "no provenance sidecar" in reasons[0]

    @pytest.mark.parametrize("field_name,value", [
        ("experiment_id", "mllmu_smoke_iter7"),
        ("checkpoint_id", "MG"),
        ("adapter_sha256", "ff" * 32),
        ("adapter_sha256", None),
        ("base_model_revision", "ff" * 20),
        ("base_model_revision", None),
    ])
    def test_a_mismatched_identity_dimension_is_refused(self, pq,
                                                        field_name, value):
        reasons = verify_sidecar(pq, _fp(**{field_name: value}))
        assert len(reasons) == 1, reasons
        assert reasons[0].startswith(f"{field_name}:")

    def test_a_different_dataset_version_is_refused(self, pq):
        ds = dict(_fp().dataset, version="pilot100_v1")
        reasons = verify_sidecar(pq, _fp(dataset=ds))
        assert len(reasons) == 1
        assert reasons[0].startswith("dataset.version:")
        assert "pilot100_v1" in reasons[0]

    @pytest.mark.parametrize("artifact", ["queries.parquet",
                                          "manifest.json"])
    def test_a_changed_dataset_artifact_is_refused(self, pq, artifact):
        """The v1 -> v2 case: same version string would not be enough, and
        a bumped version with unchanged bytes would not be enough either —
        the hash is what binds the file to the data it was scored on."""
        hashes = dict(_fp().dataset["artifacts_sha256"],
                      **{artifact: "99" * 32})
        ds = dict(_fp().dataset, artifacts_sha256=hashes)
        reasons = verify_sidecar(pq, _fp(dataset=ds))
        assert len(reasons) == 1
        assert reasons[0].startswith(f"dataset.{artifact}:")
        assert "bytes changed" in reasons[0]

    @pytest.mark.parametrize("key", list(GENERATION_CONFIG_KEYS))
    def test_every_generation_config_key_is_bound(self, pq, key):
        """Each of these changes the decoded bytes, so each must refuse on
        its own — a stale parquet generated at batch_size 2 must not be
        reported as a batch_size 8 run."""
        cfg = dict(_fp().generation_config)
        cfg[key] = "perturbed"
        reasons = verify_sidecar(pq, _fp(generation_config=cfg))
        assert len(reasons) == 1
        assert reasons[0].startswith(f"generation_config.{key}:")

    def test_the_generation_config_covers_the_whole_contract(self):
        assert set(GENERATION_CONFIG_KEYS) == {
            "batch_size", "image_batch_size", "max_new_tokens", "do_sample",
            "max_image_pixels", "max_length"}

    def test_a_different_source_commit_is_refused(self, pq):
        code = dict(_fp().code, git_commit="99" * 20)
        reasons = verify_sidecar(pq, _fp(code=code))
        assert len(reasons) == 1
        assert reasons[0].startswith("code.git_commit:")

    def test_a_changed_scoring_module_is_refused(self, pq):
        """The commit hash alone is not enough: the tree can be dirty, and
        a scorer edit changes what identical bytes MEAN."""
        modules = dict(_fp().code["modules_sha256"],
                       **{"src/granunlearn/evaluation/scoring.py": "99" * 32})
        code = dict(_fp().code, modules_sha256=modules)
        reasons = verify_sidecar(pq, _fp(code=code))
        assert len(reasons) == 1
        assert reasons[0].startswith(
            "code.src/granunlearn/evaluation/scoring.py:")
        assert "module hash differs" in reasons[0]

    def test_several_mismatches_are_all_reported(self, pq):
        """Refusal must be diagnosable in one pass, not one reason at a
        time — otherwise a stale file costs a regeneration per discovery."""
        cfg = dict(_fp().generation_config, batch_size=2)
        reasons = verify_sidecar(pq, _fp(checkpoint_id="B3",
                                         adapter_sha256="ff" * 32,
                                         generation_config=cfg))
        assert len(reasons) == 3
        assert any(r.startswith("checkpoint_id:") for r in reasons)
        assert any(r.startswith("adapter_sha256:") for r in reasons)
        assert any(r.startswith("generation_config.batch_size:")
                   for r in reasons)

    @pytest.mark.parametrize("field_name,value", [
        ("created_utc", "2027-01-01T00:00:00+00:00"),
        ("num_rows", 1),
        ("num_rows", None),
    ])
    def test_informational_fields_never_refuse(self, pq, field_name, value):
        """A file regenerated a minute later from identical inputs is the
        same evidence; refusing on wall-clock time would make crash
        recovery impossible."""
        assert verify_sidecar(pq, _fp(**{field_name: value})) == []

    def test_base_and_adapter_states_are_not_interchangeable(self, tmp_path):
        """BASE has no adapter, so its hash is None; a BASE parquet must
        never satisfy a fingerprint expecting an adapter (or vice versa)."""
        from granunlearn.evaluation.prediction_provenance import adapter_sha256
        assert adapter_sha256(None) is None
        path = tmp_path / "predictions_test_BASE.parquet"
        path.write_bytes(b"base")
        write_sidecar(path, _fp(checkpoint_id="BASE", adapter_sha256=None))
        assert verify_sidecar(path, _fp(checkpoint_id="BASE",
                                        adapter_sha256=None)) == []
        reasons = verify_sidecar(path, _fp(checkpoint_id="BASE"))
        assert any(r.startswith("adapter_sha256:") for r in reasons)

    def test_build_then_verify_is_self_consistent(self, tmp_path):
        """The real constructor path, on a throwaway dataset and repo: what
        a pass writes must be what the same pass would accept."""
        data_dir = tmp_path / "data" / "mllmu_hier_x"
        data_dir.mkdir(parents=True)
        (data_dir / "manifest.json").write_text(
            json.dumps({"version": "x_v1"}))
        (data_dir / "queries.parquet").write_bytes(b"q")
        (data_dir / "associations.parquet").write_bytes(b"a")
        cfg = dict(zip(GENERATION_CONFIG_KEYS, (8, 8, 96, False, 147456,
                                                1536)))
        fp = PredictionFingerprint.build(
            experiment_id="x", checkpoint_id="MF", repo_root=tmp_path,
            data_dir=data_dir, model_id="no-such/model",
            adapter_dir=None, generation_config=cfg, num_rows=3)
        path = data_dir / "predictions_test_MF.parquet"
        path.write_bytes(b"p")
        write_sidecar(path, fp)
        assert verify_sidecar(path, fp) == []
        assert fp.dataset["version"] == "x_v1"
        assert set(fp.dataset["artifacts_sha256"]) == {
            "associations.parquet", "queries.parquet", "manifest.json"}
        # an unknown model resolves to no revision rather than guessing
        assert fp.base_model_revision is None
        # a repo with none of the fingerprinted modules records them as
        # absent, which still refuses a file claiming real hashes
        assert fp.adapter_sha256 is None
        other = PredictionFingerprint.build(
            experiment_id="x", checkpoint_id="MF", repo_root=REPO_ROOT,
            data_dir=data_dir, model_id="no-such/model", adapter_dir=None,
            generation_config=cfg, num_rows=3)
        assert verify_sidecar(path, other)

    def test_the_fingerprinted_modules_are_the_ones_that_define_a_score(self):
        from granunlearn.evaluation.prediction_provenance import (
            CODE_FINGERPRINT_MODULES)
        for rel in ("src/granunlearn/evaluation/reference_eval.py",
                    "src/granunlearn/evaluation/query_generation.py",
                    "src/granunlearn/evaluation/scoring.py",
                    "src/granunlearn/evaluation/hierarchy_metrics.py",
                    "src/granunlearn/evaluation/image_splits.py"):
            assert rel in CODE_FINGERPRINT_MODULES, rel
            assert (REPO_ROOT / rel).exists(), rel


# ── 5. exact query coverage, not an intersection size ────────────

class TestExactPredictionCoverage:
    """``len({p.query_id for p in preds} & expected) == len(expected)`` was
    the Iteration-11 check.  It is satisfied by a file with duplicate rows,
    rows from another split, rows scored by another checkpoint, or rows from
    another experiment — none of which an intersection can see."""

    EXP = "mllmu_pilot100_iter11"

    def _preds(self, pool, partition, raw_fn, ckpt="MF", exp=None):
        exp = exp or self.EXP
        qs = generate_queries(pool, partition, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        out = []
        for q in qs:
            a = by_id[q.association_id]
            fam = q.family or ""
            raw = (a.levels[0].value if fam.startswith("retain_")
                   else raw_fn(a))
            out.append(score_query(q, a, raw, exp, ckpt))
        return qs, out

    def _pool(self):
        from granunlearn.datasets.smoke import select_target_retain
        pool = [make_species(aid="inat_a", species="Passer domesticus",
                             genus="Passer", family="Passeridae"),
                make_species(aid="inat_b", species="Corvus corax",
                             genus="Corvus", family="Corvidae")]
        for i in range(4):
            pool.append(make_person(aid=f"p{i}__res", entity_id=f"p{i}"))
        return pool, select_target_retain(pool, seed=42)

    def test_an_exact_set_passes(self):
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value)
        assert validate_prediction_coverage(
            preds, [q.query_id for q in qs], self.EXP, "MF") == []

    def test_a_duplicate_row_is_refused(self):
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value)
        dup = preds + [preds[0]]
        # the v1 intersection check would have passed this
        assert len({p.query_id for p in dup} & {q.query_id for q in qs}) == \
            len(qs)
        reasons = validate_prediction_coverage(
            dup, [q.query_id for q in qs], self.EXP, "MF")
        assert any("duplicated query_id" in r for r in reasons)
        assert any("row count" in r for r in reasons)

    def test_a_foreign_row_is_refused(self):
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value)
        other_pool = [make_species(aid="inat_c", species="Turdus merula",
                                   genus="Turdus", family="Turdidae")]
        from granunlearn.datasets.smoke import select_target_retain
        _, extra = self._preds(other_pool,
                               select_target_retain(other_pool, seed=42),
                               lambda a: a.levels[1].value, ckpt="MF")
        mixed = preds + extra
        assert len({p.query_id for p in mixed} & {q.query_id for q in qs}) == \
            len(qs)
        reasons = validate_prediction_coverage(
            mixed, [q.query_id for q in qs], self.EXP, "MF")
        assert any("outside the expected query set" in r for r in reasons)
        assert any("row count" in r for r in reasons)

    def test_a_missing_row_is_refused(self):
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value)
        short = preds[:-1]
        reasons = validate_prediction_coverage(
            short, [q.query_id for q in qs], self.EXP, "MF")
        assert any("absent" in r for r in reasons)
        assert any("row count" in r for r in reasons)

    def test_a_wrong_checkpoint_label_is_refused(self):
        """The bytes may be right and the label wrong: a parquet written by
        a B3 pass but labelled MF would enter the comparison as MF."""
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value,
                                ckpt="B3")
        reasons = validate_prediction_coverage(
            preds, [q.query_id for q in qs], self.EXP, "MF")
        assert any("checkpoint_id" in r for r in reasons)
        # set-wise the coverage is perfect, which is why the label check
        # has to be explicit
        assert not any("absent" in r for r in reasons)
        assert not any("outside the expected" in r for r in reasons)

    def test_a_wrong_experiment_label_is_refused(self):
        pool, part = self._pool()
        qs, preds = self._preds(pool, part, lambda a: a.levels[1].value,
                                exp="mllmu_smoke_iter7")
        reasons = validate_prediction_coverage(
            preds, [q.query_id for q in qs], self.EXP, "MF")
        assert any("experiment_id" in r for r in reasons)

    def test_an_empty_expectation_does_not_vacuously_pass(self):
        pool, part = self._pool()
        _, preds = self._preds(pool, part, lambda a: a.levels[1].value)
        reasons = validate_prediction_coverage(preds, [], self.EXP, "MF")
        assert any("outside the expected query set" in r for r in reasons)
        assert validate_prediction_coverage([], [], self.EXP, "MF") == []



# ── 6. paired FILR and over-forgetting CIs ───────────────────────

def make_mllmu(aid: str, entity_id: str, attr: str, htype: str,
               values: list[str], num_images: int = 1) -> AssociationRecord:
    return AssociationRecord(
        association_id=aid, dataset="mllmu_hier", entity_id=entity_id,
        entity_name=f"Person {entity_id}", attribute_name=attr,
        hierarchy_type=htype,
        levels=[_level(i, v) for i, v in enumerate(values)],
        original_level=0, target_level=1,
        images=[_img(entity_id, i) for i in range(num_images)],
        split=SplitInfo(split="train"),
        provenance=ProvenanceInfo(source_dataset="mllmu_bench"))


class TestPairedFilrAndOverForgetting:
    """FILR is the central leakage measure of the granularity thesis, and
    Iteration 11 reported it with NO interval.  These tests pin both the
    new metrics and the one property that makes them trustworthy: their
    row-level point estimates are the SAME numbers ``hierarchy_metrics``
    publishes, computed over the SAME rows."""

    EXP = "mllmu_pilot100_iter11"

    def _pool(self):
        from granunlearn.datasets.smoke import select_target_retain
        pool = []
        for i in range(4):
            # three associations per entity so that one stays RETAINED and
            # the retain_* paired metrics have rows at all
            pool.append(make_mllmu(
                f"p{i}__res", f"p{i}", "residence", "semantic",
                [f"City{i}", f"Region{i}", f"Country{i}"]))
            pool.append(make_mllmu(
                f"p{i}__dob", f"p{i}", "date_of_birth", "numeric",
                [f"198{i}-01-0{i}", f"198{i}", f"198{i}s"]))
            pool.append(make_mllmu(
                f"p{i}__height", f"p{i}", "height", "numeric",
                [f"18{i} cm", f"band-{i}", f"era-{i}"]))
        pool.append(make_species(aid="inat_a", species="Passer domesticus",
                                 genus="Passer", family="Passeridae"))
        pool.append(make_species(aid="inat_b", species="Corvus corax",
                                 genus="Corvus", family="Corvidae"))
        return pool, select_target_retain(pool, seed=42)

    def _state(self, pool, part, mode: str, ckpt: str):
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        preds = []
        for q in qs:
            a = by_id[q.association_id]
            fam = q.family or ""
            if fam.startswith("retain_"):
                raw = a.levels[0].value          # retained fact kept
            elif mode == "target":
                raw = a.levels[a.target_level].value
            elif mode == "leak":
                raw = a.levels[0].value          # finer than target
            elif mode == "coarse":
                raw = a.levels[-1].value         # an ancestor of target
            else:
                raise AssertionError(mode)
            preds.append(score_query(q, a, raw, self.EXP, ckpt))
        return qs, preds

    def test_the_metric_list_is_the_six_headline_rates(self):
        assert PAIRED_METRICS == ("filr", "tga", "wrong_branch",
                                  "over_forgetting", "retain_same",
                                  "retain_other")

    def test_filr_and_over_forgetting_share_the_target_probe_rows(self):
        """All four target-side metrics are categories of ONE row set, so
        their key sets must be identical — a FILR flag computed over a
        different denominator than TGA would not be comparable to it."""
        pool, part = self._pool()
        qs, preds = self._state(pool, part, "leak", "MF")
        flags = row_flags(preds, qs, pool)
        assert set(flags["filr"]) == set(flags["tga"])
        assert set(flags["filr"]) == set(flags["over_forgetting"])
        assert set(flags["filr"]) == set(flags["wrong_branch"])
        assert flags["filr"], "the fixture must produce target probes"

    def test_filr_flags_are_exactly_the_under_forgetting_rows(self):
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        for mode in ("target", "leak", "coarse"):
            qs, preds = self._state(pool, part, mode, "MF")
            hm = compute_hierarchy_metrics(preds, qs, pool, split=None)
            flags = row_flags(preds, qs, pool)
            tax = hm["failure_taxonomy"]
            assert sum(v for v, _ in flags["filr"].values()) == \
                tax["under_forgetting"], mode
            assert sum(v for v, _ in flags["over_forgetting"].values()) == \
                tax["over_forgetting"], mode
            assert sum(v for v, _ in flags["tga"].values()) == \
                tax["correct_at_target"], mode
            assert sum(v for v, _ in flags["wrong_branch"].values()) == \
                tax["wrong_branch"], mode
            # the categories partition the row set
            assert sum(tax[c] for c in tax) == hm["num_target_probes"]

    def test_row_point_estimates_equal_the_published_rates(self):
        """The assertion the repair plan calls for: the paired-CI per-state
        mean must equal ``hierarchy_metrics``' filr and
        failure_rates.over_forgetting EXACTLY.  It holds for the ROW-micro
        unit, which is what hierarchy_metrics publishes — not for the
        entity-macro unit the bootstrap resamples."""
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, leak = self._state(pool, part, "leak", "B3")
        _, target = self._state(pool, part, "target", "MG")
        _, coarse = self._state(pool, part, "coarse", "B1")
        states = {"B3": leak, "MG": target, "B1": coarse}
        hm = {s: compute_hierarchy_metrics(p, qs, pool, split=None)
              for s, p in states.items()}
        rep = paired_metrics_report(states, qs, pool,
                                    reference_states=("MG",), split=None,
                                    n_bootstrap=200)
        assert rep["metrics"] == list(PAIRED_METRICS)
        for state in ("B3", "B1"):
            block = rep["comparisons"][state]["vs_MG"]
            for metric in ("filr", "tga", "wrong_branch", "over_forgetting",
                           "retain_same", "retain_other"):
                assert metric in block, (state, metric)
                pe = block[metric]["point_estimates"]
                assert pe["row_a"] == self._published(hm[state], metric), \
                    (state, metric)
                assert pe["row_b"] == self._published(hm["MG"], metric), \
                    (state, metric)
                assert pe["row_diff"] == round(
                    (self._published(hm[state], metric) or 0)
                    - (self._published(hm["MG"], metric) or 0), 4)

    @staticmethod
    def _published(hm: dict, metric: str):
        """The hierarchy_metrics field a paired metric must reproduce."""
        if metric == "filr":
            return hm["filr"]
        if metric == "tga":
            return hm["tga"]
        if metric in ("wrong_branch", "over_forgetting"):
            return hm["failure_rates"][metric]
        if metric == "retain_same":
            return hm["retain_same_entity_all_routes"]["baseline_accuracy"]
        return hm["retain_other_entity_all_routes"]["baseline_accuracy"]

    def test_the_fixture_is_not_vacuous(self):
        """A FILR CI over rows that are all zero would pass every equality
        above while testing nothing."""
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, leak = self._state(pool, part, "leak", "B3")
        _, coarse = self._state(pool, part, "coarse", "B1")
        _, target = self._state(pool, part, "target", "MG")
        assert compute_hierarchy_metrics(leak, qs, pool)["filr"] > 0
        assert compute_hierarchy_metrics(
            coarse, qs, pool)["failure_rates"]["over_forgetting"] > 0
        assert compute_hierarchy_metrics(target, qs, pool)["tga"] > 0
        assert compute_hierarchy_metrics(target, qs, pool)["filr"] == 0.0

    def test_the_filr_interval_exists_and_brackets_its_difference(self):
        pool, part = self._pool()
        qs, leak = self._state(pool, part, "leak", "B3")
        _, target = self._state(pool, part, "target", "MG")
        rep = paired_metrics_report({"B3": leak, "MG": target}, qs, pool,
                                    reference_states=("MG",), split=None,
                                    n_bootstrap=400)
        filr = rep["comparisons"]["B3"]["vs_MG"]["filr"]
        low, high = filr["ci"]
        assert low <= filr["diff"] <= high
        assert filr["num_units"] > 1
        # LEAK answers finer than target everywhere MG answers at target,
        # so its FILR must be strictly worse and the interval must sit
        # above zero rather than merely being present
        assert filr["point_estimates"]["row_diff"] > 0

    def test_identical_predictions_degenerate_on_all_six_metrics(self):
        pool, part = self._pool()
        qs, a = self._state(pool, part, "leak", "B0")
        _, b = self._state(pool, part, "leak", "MF")
        rep = paired_metrics_report({"B0": a, "MF": b}, qs, pool,
                                    reference_states=("MF",), split=None,
                                    n_bootstrap=200)
        block = rep["comparisons"]["B0"]["vs_MF"]
        assert set(block) == set(PAIRED_METRICS)
        for metric, d in block.items():
            assert d["diff"] == 0.0, metric
            assert tuple(d["ci"]) == (0.0, 0.0), metric
            pe = d["point_estimates"]
            assert pe["row_diff"] == 0.0, metric
            assert pe["row_a"] == pe["row_b"], metric
            assert pe["entity_a"] == pe["entity_b"], metric

    def test_entity_and_row_units_are_both_reported_and_can_differ(self):
        """The trap this design has to survive: an entity-clustered
        bootstrap macro-averages, while the published rates micro-average
        over rows.  With unbalanced clusters the two differ, and a reader
        who subtracts two published rates gets row_diff, NOT diff."""
        fa = {"q1": (1, "e1"), "q2": (1, "e1"), "q3": (1, "e1"),
              "q4": (0, "e2")}
        fb = {"q1": (0, "e1"), "q2": (0, "e1"), "q3": (0, "e1"),
              "q4": (0, "e2")}
        d = paired_rate_diff_ci(fa, fb, n_bootstrap=50)
        pe = d["point_estimates"]
        assert pe["row_a"] == 0.75        # 3 of 4 rows
        assert pe["entity_a"] == 0.5      # mean(1.0 for e1, 0.0 for e2)
        assert pe["row_a"] != pe["entity_a"]
        # diff and its CI belong to the ENTITY unit the bootstrap resamples
        assert d["diff"] == 0.5
        assert d["num_units"] == 2
        assert d["num_rows"] == 4

    def test_the_report_documents_which_unit_the_ci_covers(self):
        pool, part = self._pool()
        qs, preds = self._state(pool, part, "leak", "MF")
        rep = paired_metrics_report({"MF": preds, "MG": preds}, qs, pool,
                                    reference_states=("MG",), split=None,
                                    n_bootstrap=50)
        note = rep["statistical_metadata"]["point_estimate_units"]
        assert "ENTITY-MACRO" in note
        assert "ROW-MICRO" in note
        assert "row_diff" in note and "NOT diff" in note

    def test_every_state_gets_a_filr_interval_against_every_reference(self):
        pool, part = self._pool()
        states = {}
        for name, mode in (("B3", "leak"), ("B1", "coarse"),
                           ("MG", "target"), ("MF", "leak")):
            qs, preds = self._state(pool, part, mode, name)
            states[name] = preds
        rep = paired_metrics_report(states, qs, pool,
                                    reference_states=("MF", "MG"),
                                    split=None, n_bootstrap=200)
        for state, refs in rep["comparisons"].items():
            for ref, block in refs.items():
                assert "filr" in block, (state, ref)
                assert "over_forgetting" in block, (state, ref)
                assert block["filr"]["num_units"] > 0


# ── 7. image-provenance strata ───────────────────────────────────

class TestImageProvenanceStrata:
    """Pooling the two image strata would let a trained-on MLLMU portrait
    carry the held-out-photograph claim, which is exactly the error
    Iteration 11 made.  Both must be reported, separately, per state."""

    def _pool(self):
        """Species MUST be targets here: only target associations get the
        failure-taxonomy strata, and the species are the only entities with
        a spare photograph, so a partition that retained them would leave
        the held-out stratum empty and the test vacuous."""
        pool = [make_species(aid="inat_a", species="Passer domesticus",
                             genus="Passer", family="Passeridae",
                             num_images=12),
                make_species(aid="inat_b", species="Corvus corax",
                             genus="Corvus", family="Corvidae",
                             num_images=12)]
        for i in range(3):
            pool.append(make_mllmu(f"p{i}__res", f"p{i}", "residence",
                                   "semantic",
                                   [f"City{i}", f"Region{i}",
                                    f"Country{i}"], num_images=1))
        targets = [a.association_id for a in pool
                   if a.association_id.startswith("inat_") or
                   a.association_id in ("p0__res",)]
        return pool, {
            "target_association_ids": targets,
            "retain_association_ids": [a.association_id for a in pool
                                       if a.association_id not in targets],
        }

    def _scored(self, pool, part):
        qs = generate_queries(pool, part, seed=42,
                              families=list(UNLEARNING_FAMILIES))
        by_id = {a.association_id: a for a in pool}
        preds = [score_query(q, by_id[q.association_id],
                             by_id[q.association_id].levels[0].value,
                             "exp", "MF") for q in qs]
        return qs, preds

    def test_both_strata_are_reported_and_non_empty(self):
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, preds = self._scored(pool, part)
        hm = compute_hierarchy_metrics(preds, qs, pool, split=None)
        strata = hm["by_image_provenance"]
        assert set(strata) == set(IMAGE_STRATA) | {"_note"}
        for name in IMAGE_STRATA:
            assert strata[name]["num_queries"] > 0, name
            assert "filr" in strata[name] and "tga" in strata[name]
            assert "failure_categories" in strata[name]

    def test_the_strata_cover_the_image_probes_only(self):
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, preds = self._scored(pool, part)
        hm = compute_hierarchy_metrics(preds, qs, pool, split=None)
        strata = hm["by_image_provenance"]
        image_target_probes = sum(
            strata[s]["num_queries"] for s in IMAGE_STRATA)
        by_q = {q.query_id: q for q in qs}
        targets = set(part["target_association_ids"])
        expected = len({p.query_id for p in preds
                        if by_q[p.query_id].image_ids
                        and by_q[p.query_id].association_id in targets
                        and not by_q[p.query_id].adversarial})
        assert image_target_probes == expected
        assert image_target_probes < hm["num_target_probes"], (
            "text-only target probes must be in neither stratum")

    def test_the_held_out_stratum_never_contains_a_trained_photograph(self):
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, preds = self._scored(pool, part)
        strata = compute_hierarchy_metrics(
            preds, qs, pool, split=None)["by_image_provenance"]
        by_id = {q.query_id: q for q in qs}
        held_ids = [q.query_id for q in qs
                    if image_stratum(q) == HELD_OUT_PHOTO]
        assert held_ids
        assert all(by_id[i].image_seen_in_training is False
                   for i in held_ids)
        assert strata[HELD_OUT_PHOTO]["num_queries"] <= len(held_ids)

    def test_a_single_portrait_entity_lands_in_the_seen_stratum(self):
        pool, part = self._pool()
        qs, _ = self._scored(pool, part)
        person_qs = [q for q in qs if q.image_ids
                     and q.association_id.startswith("p")]
        assert person_qs
        assert all(image_stratum(q) == SEEN_PHOTO_UNSEEN_WORDING
                   for q in person_qs)
        species_qs = [q for q in qs if q.image_ids
                      and q.association_id.startswith("inat_")
                      and q.split != "train"]
        assert species_qs
        assert all(image_stratum(q) == HELD_OUT_PHOTO for q in species_qs)

    def test_the_legacy_assignment_would_leave_the_held_out_stratum_empty(
            self):
        """Regression on the defect itself: with images[0] everywhere, the
        held-out stratum has no members at all, so a report that omits the
        stratification can hide an empty claim."""
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, preds = self._scored(pool, part)
        by_id = {a.association_id: a for a in pool}
        legacy = []
        for q in qs:
            if not q.image_ids:
                legacy.append(q)
                continue
            a = by_id[q.association_id]
            legacy.append(q.model_copy(update={
                "image_ids": [a.images[0].image_id],
                "image_split": "train",
                "image_seen_in_training": True}))
        strata = compute_hierarchy_metrics(
            preds, legacy, pool, split=None)["by_image_provenance"]
        assert strata[HELD_OUT_PHOTO]["num_queries"] == 0
        assert strata[SEEN_PHOTO_UNSEEN_WORDING]["num_queries"] > 0

    def test_the_strata_note_says_they_must_not_be_pooled(self):
        from granunlearn.evaluation.hierarchy_metrics import (
            compute_hierarchy_metrics)
        pool, part = self._pool()
        qs, preds = self._scored(pool, part)
        note = compute_hierarchy_metrics(
            preds, qs, pool, split=None)["by_image_provenance"]["_note"]
        assert "held_out_photo" in note
        assert "seen_photo_unseen_wording" in note
        assert "training photograph" in note


# ── 8. honest claim machinery ────────────────────────────────────

@pytest.fixture(scope="module")
def fin():
    """The final-evaluation script, imported for its pure claim helpers.

    Deferred and torch-free: the module imports nothing that needs a GPU,
    which is what keeps these assertions inside the CPU-only CI job.
    """
    import sys
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import evaluate_pilot100_final as mod
    return mod


def _paired(state: str, ref: str, metric: str, diff: float,
            ci: tuple[float, float], units: int = 72, rows: int = 1215):
    """A minimal paired-CI report holding exactly one comparison."""
    return {"comparisons": {state: {f"vs_{ref}": {metric: {
        "diff": diff, "ci": list(ci), "num_units": units, "num_rows": rows,
        "point_estimates": {"row_a": 0.3984, "row_b": 0.4362,
                            "row_diff": -0.0378, "entity_a": 0.4312,
                            "entity_b": 0.4427}}}}}}


class TestEquivalenceMargin:
    """A CI that straddles zero is "no significant difference detected".
    Equivalence is a different claim and needs a prespecified margin."""

    def test_the_margin_is_prespecified_and_documented(self, fin):
        assert fin.EQUIVALENCE_MARGIN_TGA == 0.05
        # strictly smaller than the separation the reference-state gate
        # already treats as meaningful behaviour
        assert fin.EQUIVALENCE_MARGIN_TGA < 0.15

    def test_an_interval_inside_the_margin_concludes_equivalence(self, fin):
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", 0.01, (-0.03, 0.04)),
            {"B3": {"tga": 0.44}, "MG": {"tga": 0.43}})
        st = out["states"]["B3"]
        assert st["equivalence_concluded"] is True
        assert st["significant_difference"] is False
        assert out["margin"] == 0.05 and out["metric"] == "tga"
        assert out["reference"] == "MG"

    def test_a_significant_but_small_difference_can_still_be_equivalent(
            self, fin):
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", 0.03, (0.005, 0.045)),
            {"B3": {"tga": 0.46}, "MG": {"tga": 0.43}})
        st = out["states"]["B3"]
        assert st["significant_difference"] is True
        assert st["equivalence_concluded"] is True

    def test_a_zero_straddling_interval_wider_than_the_margin_is_not(
            self, fin):
        """The exact shape Iteration 11 reported for B3 vs MG on TGA
        (-0.0633, +0.0411, half-width 0.0522): straddles zero AND is wider
        than the margin, so the design could not conclude equivalence even
        at a true difference of zero.  Used here as a shape fixture, not as
        a result."""
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", -0.0115, (-0.0633, 0.0411)),
            {"B3": {"tga": 0.3984}, "MG": {"tga": 0.4362}})
        st = out["states"]["B3"]
        assert st["equivalence_concluded"] is False
        assert st["significant_difference"] is False
        assert st["ci_half_width"] == 0.0522
        assert "INDETERMINATE" in st["power_note"]
        assert "could not conclude equivalence" in st["power_note"]
        assert "No significant difference detected" in st["power_note"]

    def test_a_narrow_interval_reaching_past_the_margin_says_so(self, fin):
        """Half-width BELOW the margin but still reaching past it: a
        difference of that size is not excluded, which is a different
        statement from lacking the power to conclude anything."""
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", -0.015, (-0.06, 0.03)),
            {"B3": {"tga": 0.41}, "MG": {"tga": 0.43}})
        st = out["states"]["B3"]
        assert st["equivalence_concluded"] is False
        assert st["ci_half_width"] == 0.045
        assert "INDETERMINATE" not in st["power_note"]
        assert "not excluded" in st["power_note"]

    def test_a_clear_difference_is_significant_and_not_equivalent(self, fin):
        out = fin._equivalence_vs_reference(
            _paired("MF", "MG", "tga", -0.22, (-0.26, -0.18)),
            {"MF": {"tga": 0.21}, "MG": {"tga": 0.43}})
        st = out["states"]["MF"]
        assert st["significant_difference"] is True
        assert st["equivalence_concluded"] is False

    def test_both_units_are_carried_into_the_equivalence_block(self, fin):
        """The equivalence test runs on the entity-macro CI, so the block
        must also show the row-micro rates it will be read against."""
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", -0.0115, (-0.0633, 0.0411)),
            {"B3": {"tga": 0.3984}, "MG": {"tga": 0.4362}})
        st = out["states"]["B3"]
        assert st["row_point_estimates"]["row_a"] == 0.3984
        assert st["hierarchy_metrics_rates"] == {"B3": 0.3984,
                                                 "MG": 0.4362}
        assert "ENTITY-MACRO" in out["ci_unit"]

    def test_the_reference_state_is_not_compared_to_itself(self, fin):
        out = fin._equivalence_vs_reference(
            _paired("B3", "MG", "tga", 0.0, (0.0, 0.0)),
            {"B3": {"tga": 0.4}, "MG": {"tga": 0.4}})
        assert "MG" not in out["states"]

    def test_a_missing_comparison_is_skipped_not_fabricated(self, fin):
        out = fin._equivalence_vs_reference(
            {"comparisons": {"B3": {"vs_MG": {}}}},
            {"B3": {"tga": 0.4}, "MG": {"tga": 0.43}})
        assert out["states"] == {}


class TestTestSplitExposureStatement:
    def test_the_statement_says_the_split_is_not_untouched(self, fin):
        gate = {s: [object()] * 2259
                for s in ("BASE", "MF", "MG", "MN")}
        out = fin._test_split_exposure(
            {"selection_scope": ["train", "val"], "basis": "D_G"},
            gate, 2259, REPO_ROOT)
        assert out["test_split_untouched"] is False
        assert out["candidates_selected_without_test_predictions"] is True
        assert out["candidate_selection_scope"] == ["train", "val"]
        assert out["gate_applies_a_test_split_separation_criterion"] is True
        assert out["reference_states_scored_on_test_before_selection"] == \
            ["BASE", "MF", "MG", "MN"]
        assert out["num_test_queries_scored_by_the_gate"] == \
            {s: 2259 for s in ("BASE", "MF", "MG", "MN")}

    def test_it_names_what_the_exposure_does_not_licence(self, fin):
        out = fin._test_split_exposure(
            {"selection_scope": ["train", "val"]}, {"MG": []}, 2259,
            REPO_ROOT)
        assert "clean hold-out" in out["what_this_does_not_licence"]
        assert "overstates" in out["exposure"]
        # and what a real hold-out would require
        assert "CONFIRMATION split" in out["what_would_be_needed"]
        assert "deliberately does not add one" in out["what_would_be_needed"]

    def test_the_gate_report_is_named_when_it_exists(self, fin):
        out = fin._test_split_exposure({}, {"MG": []}, 2259, REPO_ROOT)
        gate = REPO_ROOT / "data" / "reports" / \
            "mllmu_pilot100_reference_eval.json"
        assert out["gate_report"] == (
            "data/reports/mllmu_pilot100_reference_eval.json"
            if gate.exists() else None)
