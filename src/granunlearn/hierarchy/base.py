"""Value hierarchy protocol and base implementation.

A ``ValueHierarchy`` is a chain of nodes (single-chain for MVP, no DAGs).
Each node has a unique ``canonical_id`` and an integer ``level``.

Level convention:
    0 = finest (original value)
    higher = coarser (abstraction)

In this convention "ancestor" means a **coarser** node (higher level number).
So ``is_ancestor("california", "san_francisco")`` is ``True`` because
California (level 2) is reachable from San Francisco (level 0) via parent
links.  ``descendants`` returns nodes **finer** than the query node.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from granunlearn.schema import HierarchyLevel


@runtime_checkable
class ValueHierarchy(Protocol):
    """Interface every hierarchy type must satisfy."""

    def validate(self) -> list[ValidationIssue]: ...
    def parent(self, canonical_id: str) -> str | None: ...
    def ancestors(self, canonical_id: str) -> list[str]: ...
    def descendants(self, canonical_id: str) -> list[str]: ...
    def distance(self, a: str, b: str) -> int | None: ...
    def is_ancestor(self, ancestor: str, child: str) -> bool: ...
    def is_descendant(self, child: str, ancestor: str) -> bool: ...
    def level(self, canonical_id: str) -> int: ...


class ValidationIssue:
    """A single validation error or warning."""

    __slots__ = ("severity", "code", "message", "node_id")

    def __init__(
        self,
        severity: str,
        code: str,
        message: str,
        node_id: str | None = None,
    ):
        self.severity = severity  # "error" | "warning"
        self.code = code
        self.message = message
        self.node_id = node_id

    def __repr__(self) -> str:
        loc = f" [{self.node_id}]" if self.node_id else ""
        return f"{self.severity.upper()}{loc} {self.code}: {self.message}"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


class ChainHierarchy:
    """Concrete single-chain hierarchy built from a list of ``HierarchyLevel``.

    Parameters
    ----------
    levels : Sequence[HierarchyLevel]
        Nodes forming the chain.  Must already be ordered finest→coarsest
        (level 0, 1, 2, …).  Out-of-order input is **not** silently
        corrected — call ``validate()`` to detect malformed ordering.
    """

    def __init__(self, levels: Sequence[HierarchyLevel]) -> None:
        if not levels:
            raise ValueError("ChainHierarchy requires at least one level")

        self._levels: list[HierarchyLevel] = list(levels)
        self._by_id: dict[str, HierarchyLevel] = {}
        for lv in self._levels:
            self._by_id[lv.canonical_id] = lv

    # -- ValueHierarchy protocol -------------------------------------------

    def validate(self) -> list[ValidationIssue]:
        """Run all structural validation gates (see spec §7.5)."""
        from .validate import validate_chain
        return validate_chain(self._levels)

    def parent(self, canonical_id: str) -> str | None:
        """Return the canonical_id of the direct parent (coarser neighbour)."""
        lv = self._by_id.get(canonical_id)
        return lv.parent_id if lv else None

    def ancestors(self, canonical_id: str) -> list[str]:
        """All coarser ancestors, nearest first."""
        result: list[str] = []
        current = canonical_id
        while True:
            p = self.parent(current)
            if p is None:
                break
            result.append(p)
            current = p
        return result

    def descendants(self, canonical_id: str) -> list[str]:
        """All finer descendants, finest first."""
        target_level = self.level(canonical_id)
        return [
            lv.canonical_id
            for lv in self._levels
            if lv.level < target_level
        ]

    def distance(self, a: str, b: str) -> int | None:
        """Absolute level distance, or None if either id is unknown."""
        la = self._by_id.get(a)
        lb = self._by_id.get(b)
        if la is None or lb is None:
            return None
        return abs(la.level - lb.level)

    def is_ancestor(self, ancestor: str, child: str) -> bool:
        """True if *ancestor* is a coarser node reachable from *child* via parent links."""
        current = child
        while True:
            p = self.parent(current)
            if p is None:
                return False
            if p == ancestor:
                return True
            current = p

    def is_descendant(self, child: str, ancestor: str) -> bool:
        """True if *child* is finer than *ancestor*."""
        return self.is_ancestor(ancestor, child)

    def level(self, canonical_id: str) -> int:
        lv = self._by_id.get(canonical_id)
        if lv is None:
            raise KeyError(f"Unknown canonical_id: {canonical_id!r}")
        return lv.level

    # -- Convenience --------------------------------------------------------

    def get_level(self, canonical_id: str) -> HierarchyLevel:
        """Return the full ``HierarchyLevel`` for a canonical id."""
        return self._by_id[canonical_id]

    def all_ids(self) -> list[str]:
        return [lv.canonical_id for lv in self._levels]

    def levels(self) -> list[HierarchyLevel]:
        return list(self._levels)

    def __len__(self) -> int:
        return len(self._levels)

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._by_id
