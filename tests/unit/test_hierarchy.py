"""Unit tests for Pydantic schemas and hierarchy engine.

Covers the minimum test set from spec §32:
    test_schema_roundtrip
    test_hierarchy_cycle_rejected
    test_hierarchy_parent_child
    test_wrong_branch
    test_numeric_bin_contains_value
    test_date_hierarchy
    test_query_split_no_duplicates
    test_prediction_fine_leakage
    test_prediction_overforget
    test_prediction_wrong_branch
"""

from __future__ import annotations

import json

import pytest

from granunlearn.schema import (
    AssociationRecord,
    HierarchyLevel,
    ImageRef,
    PredictionRecord,
    ProvenanceInfo,
    QueryRecord,
    SplitInfo,
)
from granunlearn.hierarchy import (
    ChainHierarchy,
    build_date_hierarchy,
    build_height_hierarchy,
    build_salary_hierarchy,
    build_semantic_hierarchy,
    build_taxonomic_hierarchy,
    make_canonical_id,
    normalize,
    validate_chain,
)
from granunlearn.hierarchy.validate import assert_valid
from granunlearn.hierarchy.numeric import _validate_bins


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def location_chain():
    """San Francisco → California → USA"""
    return build_semantic_hierarchy(
        ["San Francisco", "California", "United States"],
        prefix="loc",
    )


@pytest.fixture
def date_chain():
    """1994-08-16 → 1994 → 1990s"""
    return build_date_hierarchy("1994-08-16", prefix="date")


@pytest.fixture
def occupation_chain():
    """pediatric cardiologist → cardiologist → physician → healthcare professional"""
    return build_semantic_hierarchy(
        ["pediatric cardiologist", "cardiologist", "physician", "healthcare professional"],
        prefix="occ",
    )


@pytest.fixture
def taxon_chain():
    """species → genus → family"""
    return build_taxonomic_hierarchy([
        {"name": "Passer domesticus", "rank": "species", "taxon_id": "12345"},
        {"name": "Passer", "rank": "genus", "taxon_id": "12344"},
        {"name": "Passeridae", "rank": "family", "taxon_id": "12340"},
    ], prefix="tax")


# =====================================================================
# Schema round-trip tests
# =====================================================================

class TestSchemaRoundtrip:
    def test_schema_roundtrip(self):
        """AssociationRecord survives JSON serialization round-trip."""
        assoc = AssociationRecord(
            association_id="test_001",
            dataset="test_ds",
            entity_id="entity_001",
            entity_name="Alice",
            attribute_name="residence",
            hierarchy_type="semantic",
            levels=[
                HierarchyLevel(level=0, canonical_id="loc:sf", value="San Francisco",
                               normalized_value="san francisco", parent_id="loc:california"),
                HierarchyLevel(level=1, canonical_id="loc:california", value="California",
                               normalized_value="california", parent_id="loc:usa"),
                HierarchyLevel(level=2, canonical_id="loc:usa", value="United States",
                               normalized_value="united states", parent_id=None),
            ],
            original_level=0,
            target_level=1,
            source_modalities=["text", "image"],
            images=[ImageRef(image_id="img_001", path="data/raw/img_001.jpg")],
            textual_context=["Alice lives in San Francisco."],
            retain_attribute_names=["occupation", "salary"],
            split=SplitInfo(split="train"),
            provenance=ProvenanceInfo(source_dataset="test"),
        )

        # Serialize and deserialize
        data = json.loads(assoc.model_dump_json())
        restored = AssociationRecord.model_validate(data)

        assert restored.association_id == assoc.association_id
        assert restored.target_value().canonical_id == "loc:california"
        assert len(restored.levels) == 3
        assert restored.fine_value().value == "San Francisco"

    def test_query_record_roundtrip(self):
        """QueryRecord survives JSON round-trip."""
        qr = QueryRecord(
            query_id="q_001",
            association_id="test_001",
            route="text_to_text",
            query_type="fine_direct",
            prompt="Where does Alice live?",
            expected_level=0,
            acceptable_answer_ids=["loc:sf"],
            forbidden_descendant_ids=[],
            split="test",
            paraphrase_group_id="pg_001",
        )
        data = json.loads(qr.model_dump_json())
        restored = QueryRecord.model_validate(data)
        assert restored.query_id == "q_001"
        assert restored.route == "text_to_text"

    def test_prediction_record_roundtrip(self):
        """PredictionRecord survives JSON round-trip."""
        pred = PredictionRecord(
            experiment_id="exp_001",
            checkpoint_id="MF",
            query_id="q_001",
            raw_output="She lives in San Francisco.",
            parsed_answer="San Francisco",
            matched_canonical_id="loc:sf",
            predicted_level=0,
            is_correct_branch=True,
            is_finer_than_target=True,
            is_coarser_than_target=False,
        )
        data = json.loads(pred.model_dump_json())
        restored = PredictionRecord.model_validate(data)
        assert restored.is_finer_than_target is True


# =====================================================================
# Hierarchy parent/child tests
# =====================================================================

class TestHierarchyParentChild:
    def test_hierarchy_parent_child(self, location_chain):
        """Parent/child relationships are correct."""
        sf_id = make_canonical_id("loc", "San Francisco")
        ca_id = make_canonical_id("loc", "California")
        usa_id = make_canonical_id("loc", "United States")

        assert location_chain.parent(sf_id) == ca_id
        assert location_chain.parent(ca_id) == usa_id
        assert location_chain.parent(usa_id) is None

    def test_ancestors(self, location_chain):
        ca_id = make_canonical_id("loc", "California")
        usa_id = make_canonical_id("loc", "United States")
        sf_id = make_canonical_id("loc", "San Francisco")

        ancs = location_chain.ancestors(sf_id)
        assert ancs == [ca_id, usa_id]

    def test_descendants(self, location_chain):
        ca_id = make_canonical_id("loc", "California")
        sf_id = make_canonical_id("loc", "San Francisco")

        descs = location_chain.descendants(ca_id)
        assert sf_id in descs

    def test_is_ancestor(self, location_chain):
        ca_id = make_canonical_id("loc", "California")
        sf_id = make_canonical_id("loc", "San Francisco")

        assert location_chain.is_ancestor(ca_id, sf_id) is True
        assert location_chain.is_ancestor(sf_id, ca_id) is False

    def test_is_descendant(self, location_chain):
        ca_id = make_canonical_id("loc", "California")
        sf_id = make_canonical_id("loc", "San Francisco")

        assert location_chain.is_descendant(sf_id, ca_id) is True
        assert location_chain.is_descendant(ca_id, sf_id) is False

    def test_distance(self, location_chain):
        sf_id = make_canonical_id("loc", "San Francisco")
        usa_id = make_canonical_id("loc", "United States")

        assert location_chain.distance(sf_id, usa_id) == 2
        assert location_chain.distance(sf_id, "nonexistent") is None


# =====================================================================
# Wrong-branch detection
# =====================================================================

class TestWrongBranch:
    def test_wrong_branch(self):
        """Detect that Ontario is NOT an ancestor of San Francisco."""
        sf_chain = build_semantic_hierarchy(
            ["San Francisco", "California", "United States"], prefix="loc",
        )
        on_chain = build_semantic_hierarchy(
            ["Toronto", "Ontario", "Canada"], prefix="loc",
        )

        on_id = make_canonical_id("loc", "Ontario")
        sf_id = make_canonical_id("loc", "San Francisco")

        # Ontario is in a different hierarchy entirely
        assert on_id not in sf_chain
        assert sf_chain.parent(sf_id) != on_id


# =====================================================================
# Validation tests
# =====================================================================

class TestValidation:
    def test_hierarchy_cycle_rejected(self):
        """A cycle in the parent chain is detected."""
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="b"),
            HierarchyLevel(level=1, canonical_id="b", value="B", normalized_value="b", parent_id="a"),  # cycle!
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        assert "CYCLE_DETECTED" in codes

    def test_duplicate_id_rejected(self):
        levels = [
            HierarchyLevel(level=0, canonical_id="dup", value="A", normalized_value="a", parent_id="dup"),
            HierarchyLevel(level=1, canonical_id="dup", value="B", normalized_value="b", parent_id=None),
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        assert "DUPLICATE_ID" in codes

    def test_chain_parent_mismatch_rejected(self):
        """A parent that skips a level is detected (strict chain invariant)."""
        # level 0 -> level 2 (skips level 1)
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="c"),
            HierarchyLevel(level=1, canonical_id="b", value="B", normalized_value="b", parent_id=None),
            HierarchyLevel(level=2, canonical_id="c", value="C", normalized_value="c", parent_id=None),
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        assert "CHAIN_PARENT_MISMATCH" in codes

    def test_chain_root_has_parent_rejected(self):
        """Coarsest node with a non-None parent is rejected."""
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="b"),
            HierarchyLevel(level=1, canonical_id="b", value="B", normalized_value="b", parent_id="ghost"),
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        # ghost parent doesn't exist → CHAIN_ROOT_HAS_PARENT (b is coarsest but has parent)
        assert "CHAIN_ROOT_HAS_PARENT" in codes

    def test_level_sequence_gap_rejected(self):
        """Level indices 0, 2, 9 are rejected (must be 0, 1, 2)."""
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="b"),
            HierarchyLevel(level=2, canonical_id="b", value="B", normalized_value="b", parent_id="c"),
            HierarchyLevel(level=9, canonical_id="c", value="C", normalized_value="c", parent_id=None),
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        assert "LEVEL_SEQUENCE_INVALID" in codes

    def test_duplicate_normalized_rejected(self):
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="Hello", normalized_value="hello", parent_id="b"),
            HierarchyLevel(level=1, canonical_id="b", value="HELLO", normalized_value="hello", parent_id=None),
        ]
        issues = validate_chain(levels)
        codes = [i.code for i in issues]
        assert "DUPLICATE_NORMALIZED" in codes

    def test_valid_chain_passes(self, location_chain):
        issues = location_chain.validate()
        errors = [i for i in issues if i.is_error]
        assert len(errors) == 0

    def test_assert_valid_raises(self):
        levels = [
            HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="ghost"),
        ]
        issues = validate_chain(levels)
        with pytest.raises(ValueError, match="CHAIN_ROOT_HAS_PARENT"):
            assert_valid(issues)


class TestAssociationBounds:
    """AssociationRecord level-bounds validation."""

    def _make_assoc(self, **kwargs):
        defaults = {
            "association_id": "test",
            "dataset": "ds",
            "entity_id": "e1",
            "attribute_name": "attr",
            "hierarchy_type": "semantic",
            "levels": [
                HierarchyLevel(level=0, canonical_id="a", value="A", normalized_value="a", parent_id="b"),
                HierarchyLevel(level=1, canonical_id="b", value="B", normalized_value="b", parent_id=None),
            ],
            "original_level": 0,
            "target_level": 1,
            "split": SplitInfo(split="train"),
            "provenance": ProvenanceInfo(source_dataset="ds"),
        }
        defaults.update(kwargs)
        return AssociationRecord(**defaults)

    def test_target_level_out_of_bounds(self):
        with pytest.raises(Exception, match="target_level"):
            self._make_assoc(target_level=17)

    def test_original_level_out_of_bounds(self):
        with pytest.raises(Exception, match="original_level"):
            self._make_assoc(original_level=5)

    def test_original_level_must_be_zero(self):
        with pytest.raises(Exception, match="original_level must be 0"):
            self._make_assoc(original_level=1)

    def test_target_below_original_rejected(self):
        # Since original_level must be 0, target_level < 0 is already
        # blocked by Field(ge=0).  Verify target_level == original_level
        # is allowed (retain the exact fine value).
        assoc = self._make_assoc(original_level=0, target_level=0)
        assert assoc.target_level == 0
        assert assoc.original_level == 0


# =====================================================================
# Numeric hierarchy tests
# =====================================================================

class TestNumericHierarchy:
    def test_numeric_bin_contains_value(self):
        """The fine value falls inside its parent bin."""
        h = build_salary_hierarchy(87_500)
        # Level 0 is the exact value, level 1 is the bin
        fine_id = h.all_ids()[0]
        bin_id = h.all_ids()[1]
        fine_lv = h.get_level(fine_id)
        bin_lv = h.get_level(bin_id)

        assert fine_lv.metadata["raw_value"] == 87_500
        assert bin_lv.metadata["bin_lo"] == 75_000
        assert bin_lv.metadata["bin_hi"] == 100_000
        assert 75_000 <= 87_500 < 100_000

    def test_date_hierarchy(self):
        """Date hierarchy produces correct levels."""
        h = build_date_hierarchy("1994-08-16")
        assert len(h) == 3
        ids = h.all_ids()
        assert h.get_level(ids[0]).value == "1994-08-16"
        assert h.get_level(ids[1]).value == "1994"
        assert h.get_level(ids[2]).value == "1990s"

    def test_height_hierarchy(self):
        h = build_height_hierarchy(177)
        ids = h.all_ids()
        assert h.get_level(ids[0]).value == "177 cm"
        # 177 should be in [170, 180)
        bin_lv = h.get_level(ids[1])
        assert bin_lv.metadata["bin_lo"] == 170
        assert bin_lv.metadata["bin_hi"] == 180

    def test_bin_value_outside_range_raises(self):
        with pytest.raises(ValueError, match="does not fall in any configured bin"):
            build_salary_hierarchy(999_999_999, bins=[[0, 100]])

    def test_bin_gap_rejected(self):
        with pytest.raises(ValueError, match="Gap or overlap"):
            _validate_bins([[0, 50], [60, 100]])  # gap at 50-60

    def test_bin_overlap_rejected(self):
        with pytest.raises(ValueError, match="Gap or overlap"):
            _validate_bins([[0, 60], [50, 100]])  # overlap at 50-60

    def test_bin_reversed_rejected(self):
        with pytest.raises(ValueError, match="reversed or zero-width"):
            _validate_bins([[100, 50]])

    def test_invalid_date_rejected(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            build_date_hierarchy("not-a-date")

    def test_invalid_calendar_date_rejected(self):
        """Feb 30 is not a valid calendar date."""
        with pytest.raises(ValueError, match="Cannot parse date"):
            build_date_hierarchy("2024-02-30")


# =====================================================================
# Semantic / taxonomic tests
# =====================================================================

class TestSemanticTaxonomic:
    def test_semantic_chain(self, occupation_chain):
        ids = occupation_chain.all_ids()
        assert len(ids) == 4
        assert occupation_chain.level(ids[0]) == 0
        assert occupation_chain.level(ids[-1]) == 3

    def test_taxonomic_chain(self, taxon_chain):
        species_id = make_canonical_id("tax", "Passer domesticus")
        genus_id = make_canonical_id("tax", "Passer")
        family_id = make_canonical_id("tax", "Passeridae")

        assert taxon_chain.is_ancestor(genus_id, species_id)
        assert taxon_chain.is_ancestor(family_id, species_id)
        assert taxon_chain.get_level(species_id).metadata["rank"] == "species"

    def test_taxonomy_wrong_rank_order_rejected(self):
        """species → family → genus is rejected (wrong rank order)."""
        with pytest.raises(ValueError, match="Rank order not strictly increasing"):
            build_taxonomic_hierarchy([
                {"name": "Passer domesticus", "rank": "species"},
                {"name": "Passeridae", "rank": "family"},
                {"name": "Passer", "rank": "genus"},  # genus < family → wrong
            ])

    def test_taxonomy_unknown_rank_rejected(self):
        with pytest.raises(ValueError, match="Unknown taxonomic rank"):
            build_taxonomic_hierarchy([
                {"name": "X", "rank": "species"},
                {"name": "Y", "rank": "bogus_rank"},
            ])

    def test_taxonomy_missing_rank_rejected(self):
        with pytest.raises(ValueError, match="has no 'rank' field"):
            build_taxonomic_hierarchy([
                {"name": "X"},  # no rank
                {"name": "Y", "rank": "genus"},
            ])


# =====================================================================
# Prediction scoring tests
# =====================================================================

class TestPredictionScoring:
    def _make_pred(self, **kwargs) -> PredictionRecord:
        defaults = {
            "experiment_id": "exp_001",
            "checkpoint_id": "MU",
            "query_id": "q_001",
            "raw_output": "",
        }
        defaults.update(kwargs)
        return PredictionRecord(**defaults)

    def test_prediction_fine_leakage(self):
        """Model outputs fine-grained value when target is coarser."""
        pred = self._make_pred(
            raw_output="She lives in San Francisco.",
            parsed_answer="San Francisco",
            matched_canonical_id="loc:sf",
            predicted_level=0,
            is_finer_than_target=True,
            is_correct_branch=True,
        )
        assert pred.is_finer_than_target is True
        assert pred.predicted_level == 0

    def test_prediction_overforget(self):
        """Model outputs a value coarser than the target."""
        pred = self._make_pred(
            raw_output="She lives in the United States.",
            parsed_answer="United States",
            matched_canonical_id="loc:usa",
            predicted_level=2,
            is_coarser_than_target=True,
            is_correct_branch=True,
        )
        assert pred.is_coarser_than_target is True
        assert pred.predicted_level == 2

    def test_prediction_wrong_branch(self):
        """Model outputs a value on a different branch (e.g. Ontario instead of California)."""
        pred = self._make_pred(
            raw_output="She lives in Ontario.",
            parsed_answer="Ontario",
            matched_canonical_id="loc:ontario",
            predicted_level=1,
            is_correct_branch=False,
        )
        assert pred.is_correct_branch is False


# =====================================================================
# Canonicalize tests
# =====================================================================

class TestCanonicalize:
    def test_normalize_basic(self):
        assert normalize("  Hello World  ") == "hello world"
        assert normalize("San Francisco.") == "san francisco"
        assert normalize("A  B   C") == "a b c"

    def test_make_canonical_id(self):
        cid = make_canonical_id("loc", "San Francisco")
        assert cid == "loc:san_francisco"
        cid2 = make_canonical_id("date", "1994-08-16")
        assert cid2 == "date:1994-08-16"
