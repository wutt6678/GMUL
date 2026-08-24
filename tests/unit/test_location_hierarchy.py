"""Unit tests for the observed-components location hierarchy (Iteration 4)."""

from __future__ import annotations

import pytest

from granunlearn.hierarchy.location import build_location_hierarchy
from granunlearn.hierarchy.validate import validate_chain


class TestVariableDepth:
    def test_two_components_give_two_levels(self):
        """'Riga, Latvia' -> exact -> Latvia.  No invented middle level."""
        chain = build_location_hierarchy(["Riga", "Latvia"])
        levels = chain.levels()
        assert [lv.value for lv in levels] == ["Riga, Latvia", "Latvia"]
        assert len(levels) == 2

    def test_three_components_give_three_levels(self):
        chain = build_location_hierarchy(["San Francisco", "California", "USA"])
        assert [lv.value for lv in chain.levels()] == [
            "San Francisco, California, USA",
            "California, USA",
            "USA",
        ]

    def test_depth_equals_observed_component_count(self):
        chain = build_location_hierarchy(["A", "B", "C", "D"], prefix="loc")
        assert len(chain.levels()) == 4
        assert chain.levels()[-1].value == "D"


class TestNoFabricatedMetadata:
    def test_single_component_rejected(self):
        """One observed component cannot form a hierarchy; refuse rather
        than geocode an invented parent."""
        with pytest.raises(ValueError, match=">= 2 observed components"):
            build_location_hierarchy(["Luxembourg"])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            build_location_hierarchy([])
        with pytest.raises(ValueError):
            build_location_hierarchy(["", "  "])


class TestChainValidity:
    def test_passes_validate_chain(self):
        for comps in (["Riga", "Latvia"],
                      ["San Francisco", "California", "USA"]):
            chain = build_location_hierarchy(comps, prefix="loc")
            errors = [i for i in validate_chain(chain.levels()) if i.is_error]
            assert errors == [], errors

    def test_parent_links_strict(self):
        levels = build_location_hierarchy(["Kyoto", "Japan"]).levels()
        assert levels[0].parent_id == levels[1].canonical_id
        assert levels[1].parent_id is None

    def test_metadata_records_observed_components(self):
        levels = build_location_hierarchy(["Kyoto", "Japan"]).levels()
        assert levels[0].metadata["observed_components"] == ["Kyoto", "Japan"]
        assert levels[0].metadata["component_depth"] == 2
