"""Tests for the iteration-suffixed SALMU path layout (10R5).

The empty suffix must reproduce the original 10R2-10R4 layout EXACTLY
(so holdout-clean evidence can never overwrite earlier evidence), and
any suffix must tag every derived artifact consistently.
"""

from pathlib import Path

import pytest

from granunlearn.salmu.paths import SalmuPaths


@pytest.fixture
def root() -> Path:
    return Path("/repo")


class TestEmptySuffixOriginalLayout:
    def test_training_dir(self, root):
        assert SalmuPaths(root).training_dir == \
            root / "data" / "salmu_hierarchical" / "training"

    def test_manifest_and_mf_pairs(self, root):
        p = SalmuPaths(root)
        assert p.manifest_path == \
            root / "data" / "salmu_hierarchical" / "training" / \
            "state_pairs_manifest.json"
        assert p.mf_pairs_path == \
            root / "data" / "salmu_hierarchical" / "training" / \
            "MF.jsonl"

    def test_checkpoint_roots(self, root):
        p = SalmuPaths(root)
        assert p.ref_ckpt_root == \
            root / "data" / "checkpoints" / "salmu"
        assert p.unlearn_root == \
            root / "data" / "checkpoints" / "salmu_unlearn"

    def test_caches_shards_and_reports(self, root):
        p = SalmuPaths(root)
        hier = root / "data" / "salmu_hierarchical"
        assert p.target_cache_path == hier / "probe_sims_unlearn.json"
        assert p.retain_cache_path == hier / "probe_sims_retain.json"
        assert p.shard_dir == hier / "probe_sims_shards"
        assert p.official_shard_dir == hier / "official_split_shards"
        assert p.report("salmu_official_splits") == \
            root / "data" / "reports" / "salmu_official_splits.json"

    def test_checkpoint_helpers(self, root):
        p = SalmuPaths(root)
        assert p.ref_checkpoint("MF") == \
            root / "data" / "checkpoints" / "salmu" / "MF" / \
            "pytorch_model.bin"
        assert p.candidate_checkpoint("B1_lr2e-06_c") == \
            root / "data" / "checkpoints" / "salmu_unlearn" / \
            "B1_lr2e-06_c" / "pytorch_model.bin"


class TestSuffixedLayoutIsolation:
    def test_tag(self, root):
        assert SalmuPaths(root).tag == ""
        assert SalmuPaths(root, suffix="r5").tag == "_r5"

    def test_training_and_groups_tagged(self, root):
        p = SalmuPaths(root, suffix="r5")
        hier = root / "data" / "salmu_hierarchical"
        assert p.training_dir == hier / "training_r5"
        assert p.groups_dir == hier / "unlearning_groups_r5"

    def test_checkpoint_roots_tagged(self, root):
        p = SalmuPaths(root, suffix="r5")
        assert p.ref_ckpt_root == \
            root / "data" / "checkpoints" / "salmu_r5"
        assert p.unlearn_root == \
            root / "data" / "checkpoints" / "salmu_unlearn_r5"
        assert p.ref_checkpoint("MN") == \
            root / "data" / "checkpoints" / "salmu_r5" / "MN" / \
            "pytorch_model.bin"

    def test_caches_shards_and_reports_tagged(self, root):
        p = SalmuPaths(root, suffix="r5")
        hier = root / "data" / "salmu_hierarchical"
        assert p.target_cache_path == \
            hier / "probe_sims_unlearn_r5.json"
        assert p.shard_dir == hier / "probe_sims_shards_r5"
        assert p.official_shard_dir == \
            hier / "official_split_shards_r5"
        assert p.report("salmu_unlearning_selection") == \
            root / "data" / "reports" / \
            "salmu_unlearning_selection_r5.json"

    def test_suffix_never_collides_with_original(self, root):
        """No suffixed artifact may share a path with an original
        one (holdout-clean evidence must not overwrite 10R2-10R4)."""
        orig = SalmuPaths(root)
        r5 = SalmuPaths(root, suffix="r5")
        for a, b in [
            (orig.training_dir, r5.training_dir),
            (orig.groups_dir, r5.groups_dir),
            (orig.ref_ckpt_root, r5.ref_ckpt_root),
            (orig.unlearn_root, r5.unlearn_root),
            (orig.target_cache_path, r5.target_cache_path),
            (orig.shard_dir, r5.shard_dir),
            (orig.official_shard_dir, r5.official_shard_dir),
            (orig.report("salmu_official_splits"),
             r5.report("salmu_official_splits")),
        ]:
            assert a != b
