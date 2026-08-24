"""Dataset adapter protocol and shared utilities.

Every dataset source (iNaturalist, MLLMU, SALMU, CelebA) implements
the ``DatasetAdapter`` protocol so that ``build_dataset.py`` can drive
them uniformly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from granunlearn.schema import AssociationRecord


@runtime_checkable
class DatasetAdapter(Protocol):
    """Interface every dataset adapter must satisfy."""

    def load_raw(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Load raw records from the source dataset.

        Returns a list of raw dicts whose schema is adapter-specific.
        """
        ...

    def to_associations(
        self,
        raw_records: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[AssociationRecord]:
        """Convert raw records into canonical ``AssociationRecord`` objects.

        Must attach hierarchy levels, provenance, and split info.
        """
        ...

    def name(self) -> str:
        """Short identifier for this dataset (e.g. 'inaturalist')."""
        ...


def get_adapter(dataset_name: str) -> DatasetAdapter:
    """Look up a dataset adapter by name.

    Parameters
    ----------
    dataset_name : str
        Must match one of the registered adapter names.

    Returns
    -------
    DatasetAdapter
    """
    if dataset_name in ("inaturalist", "inat"):
        from .inaturalist import INaturalistAdapter
        return INaturalistAdapter()
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"Available: inaturalist"
    )
