"""Unit tests for deterministic entity-level splits (Iteration 4)."""

from __future__ import annotations

from granunlearn.datasets.splits import deterministic_entity_splits


class TestEntitySplits:
    def test_all_entities_assigned(self):
        ids = [f"mllmu_{i:03d}" for i in range(100)]
        splits = deterministic_entity_splits(ids, seed=42)
        assert set(splits) == set(ids)
        assert set(splits.values()) == {"train", "val", "test"}

    def test_guarantees_train_and_test(self):
        for n in (2, 3, 5, 10, 500):
            ids = [f"e{i}" for i in range(n)]
            splits = deterministic_entity_splits(ids, seed=7)
            counts = {s: list(splits.values()).count(s) for s in ("train", "val", "test")}
            assert counts["train"] >= 1
            if n >= 2:
                assert counts["test"] >= 1
            if n >= 4:
                assert counts["val"] >= 1

    def test_deterministic_and_order_independent(self):
        ids = [f"e{i}" for i in range(50)]
        s1 = deterministic_entity_splits(ids, seed=3)
        s2 = deterministic_entity_splits(ids, seed=3)
        s3 = deterministic_entity_splits(list(reversed(ids)), seed=3)
        assert s1 == s2 == s3

    def test_seed_changes_assignment(self):
        ids = [f"e{i}" for i in range(100)]
        s1 = deterministic_entity_splits(ids, seed=1)
        s2 = deterministic_entity_splits(ids, seed=2)
        assert s1 != s2

    def test_empty(self):
        assert deterministic_entity_splits([], seed=1) == {}

    def test_ratios_approximate(self):
        ids = [f"e{i}" for i in range(500)]
        splits = deterministic_entity_splits(ids, seed=42)
        vals = list(splits.values())
        assert abs(vals.count("train") / 500 - 0.6) < 0.05
        assert abs(vals.count("test") / 500 - 0.2) < 0.05
