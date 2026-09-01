"""Iteration-suffixed path layout for the SALMU experiment chain.

Iteration 10R5 introduced protocol-compliant (holdout-clean) data and
retrained states ALONGSIDE the original 10R2-10R4 evidence instead of
overwriting it.  Every derived artifact of a suffixed iteration lives
at the same relative location with a ``_<suffix>`` tag:

* ``training_r5/``            — state pair sets + manifest
* ``checkpoints/salmu_r5``    — MF/MG/MN reference checkpoints
* ``checkpoints/salmu_unlearn_r5`` — B0-B3 candidates + selected
* ``probe_sims_*_r5.json``, ``probe_sims_shards_r5/``,
  ``official_split_shards_r5/``, ``failure_exports_r5/``
* ``data/reports/salmu_*_r5.json``

The empty suffix reproduces the original (10R2-10R4) layout exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SalmuPaths:
    repo_root: Path
    suffix: str = ""

    @property
    def tag(self) -> str:
        return f"_{self.suffix}" if self.suffix else ""

    @property
    def hier_dir(self) -> Path:
        return self.repo_root / "data" / "salmu_hierarchical"

    @property
    def training_dir(self) -> Path:
        return self.hier_dir / f"training{self.tag}"

    @property
    def manifest_path(self) -> Path:
        return self.training_dir / "state_pairs_manifest.json"

    @property
    def mf_pairs_path(self) -> Path:
        return self.training_dir / "MF.jsonl"

    @property
    def groups_dir(self) -> Path:
        return self.hier_dir / f"unlearning_groups{self.tag}"

    @property
    def ref_ckpt_root(self) -> Path:
        return self.repo_root / "data" / "checkpoints" / \
            f"salmu{self.tag}"

    @property
    def unlearn_root(self) -> Path:
        return self.repo_root / "data" / "checkpoints" / \
            f"salmu_unlearn{self.tag}"

    @property
    def target_cache_path(self) -> Path:
        return self.hier_dir / f"probe_sims_unlearn{self.tag}.json"

    @property
    def retain_cache_path(self) -> Path:
        return self.hier_dir / f"probe_sims_retain{self.tag}.json"

    @property
    def shard_dir(self) -> Path:
        return self.hier_dir / f"probe_sims_shards{self.tag}"

    @property
    def official_shard_dir(self) -> Path:
        return self.hier_dir / f"official_split_shards{self.tag}"

    @property
    def failure_export_dir(self) -> Path:
        return self.hier_dir / f"failure_exports{self.tag}"

    def report(self, name: str) -> Path:
        """Report path, e.g. report('salmu_reference_eval') ->
        data/reports/salmu_reference_eval[_r5].json"""
        return self.repo_root / "data" / "reports" / \
            f"{name}{self.tag}.json"

    def ref_checkpoint(self, state: str) -> Path:
        return self.ref_ckpt_root / state / "pytorch_model.bin"

    def candidate_checkpoint(self, cid: str) -> Path:
        return self.unlearn_root / cid / "pytorch_model.bin"
