"""Unit tests for deterministic target-level selection (Iteration 4)."""

from __future__ import annotations

import pytest

from granunlearn.hierarchy.target_selection import select_target_level


class TestTargetSelection:
    def test_range(self):
        for n in (2, 3, 4, 5):
            for i in range(50):
                t = select_target_level(42, f"e{i}:attr", n)
                assert 1 <= t <= n - 1

    def test_deterministic(self):
        a = select_target_level(42, "mllmu_001:salary", 3)
        b = select_target_level(42, "mllmu_001:salary", 3)
        assert a == b

    def test_key_sensitive(self):
        # Different attributes of the same entity generally differ; at
        # minimum the function must depend on the key.
        vals = {select_target_level(42, f"e:{a}", 4) for a in
                ["residence", "birthplace", "salary", "height", "date_of_birth",
                 "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]}
        assert len(vals) > 1

    def test_seed_sensitive(self):
        vals = {select_target_level(s, "k", 5) for s in range(30)}
        assert len(vals) > 1

    def test_two_levels_always_one(self):
        assert select_target_level(42, "any", 2) == 1

    def test_invalid_n_levels(self):
        with pytest.raises(ValueError):
            select_target_level(42, "k", 1)
