"""The unit-test import graph must fit inside requirements/ci-unit.txt.

CI installs ``-r requirements/ci-unit.txt`` and nothing else, then runs
``pytest tests/unit -q --maxfail=5``.  A test module that imports anything
outside that closure fails at COLLECTION, and collection errors interrupt the
run — so one missing dependency does not fail one test, it hides every other
result behind ``Interrupted: 1 error during collection``.

That is what happened when ``scripts/fetch_inat_species.py`` grew a
module-level ``import requests``: the frozen-pool guard tests could no longer
be collected on a runner that installs no HTTP client.  It was invisible
locally, because a working checkout has a working environment; a fresh CLONE
is not a fresh ENVIRONMENT, and only the second one is what CI runs in.

The closure is derived from the requirements file rather than written down
beside it, so the check cannot drift from the thing it checks.  Only
MODULE-LEVEL imports are followed, because those are what collection
executes: an ``import requests`` inside the function that reaches the network
is exactly the fix, and counting it would make the guard demand a dependency
the suite never needs.
"""

from __future__ import annotations

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_REQUIREMENTS = REPO_ROOT / "requirements" / "ci-unit.txt"
TEST_DIR = REPO_ROOT / "tests" / "unit"

#: Roots a bare ``import name`` may resolve through, mirroring the
#: ``sys.path`` entries the test modules and the workflow rely on.
LOCAL_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests",
               REPO_ROOT)

#: Distribution name -> importable name, for the few packages in this
#: closure where the two differ.  Consulted only as a FALLBACK: the
#: authoritative mapping is ``importlib.metadata.packages_distributions()``,
#: which reads each distribution's file list and gets this right whenever
#: the metadata is present.
IMPORT_NAME_ALIASES = {
    "pyyaml": ("yaml",),
    "pillow": ("PIL",),
    "python-dateutil": ("dateutil",),
    "pysocks": ("socks",),
}


def _normalise(dist: str) -> str:
    return re.sub(r"[-_.]+", "-", dist).strip().lower()


def _declared_distributions() -> set[str]:
    """Distribution names ci-unit.txt asks pip to install."""
    out: set[str] = set()
    for line in CI_REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # strip version specifiers and environment markers
        out.add(_normalise(re.split(r"[<>=!~\[;]", line, 1)[0]))
    return out


def _closure() -> set[str]:
    """Importable top-level names the ci-unit.txt closure provides.

    Expanded transitively — ``pydantic`` without ``typing_inspection``, or
    ``requests`` without ``urllib3``, would be a false gap.  Requirements
    gated behind an ``extra`` marker are NOT installed by a plain
    ``pip install -r``, and their markers do not evaluate here, so they drop
    out automatically.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    seen: set[str] = set()
    queue = sorted(_declared_distributions())
    while queue:
        dist = queue.pop()
        if dist in seen:
            continue
        seen.add(dist)
        try:
            requires = metadata.requires(dist) or []
        except metadata.PackageNotFoundError:
            continue          # not installed here; its deps are unknown
        for req in requires:
            try:
                r = Requirement(req)
            except InvalidRequirement:
                continue
            if r.marker is not None and not r.marker.evaluate():
                continue
            queue.append(_normalise(r.name))

    importable: set[str] = set()
    for name, dists in metadata.packages_distributions().items():
        if any(_normalise(d) in seen for d in dists):
            importable.add(name)
    # packages_distributions() maps a distribution to its files, which a
    # conda-installed package often does not record, so on a developer
    # machine numpy or pandas would look like a gap the CI environment does
    # not have - and a guard that cries wolf locally gets ignored.  Fall back
    # to the distribution name itself (for numpy, pandas, pyarrow, pydantic
    # and pytest the import name IS the distribution name) plus the alias map
    # for the handful where it is not.  Neither fallback can mask a real gap:
    # the imports this guard exists to catch - requests, torch, scipy - are
    # not distribution names in the closure.
    for dist in seen:
        ident = dist.replace("-", "_")
        if ident.isidentifier():
            importable.add(ident)
        importable.update(IMPORT_NAME_ALIASES.get(dist, ()))
    return importable


def _module_level_imports(py: Path) -> set[str]:
    """Top-level names this file imports at MODULE level.

    Deliberately ignores imports nested in functions, ``try`` blocks and
    ``if`` guards: those do not run unconditionally at collection, and the
    ``try`` form is how an optional dependency is probed rather than
    required.  A class body IS walked, because it executes at import time.
    """
    tree = ast.parse(py.read_text(), filename=str(py))
    out: set[str] = set()
    bodies = [tree.body]
    bodies += [n.body for n in tree.body if isinstance(n, ast.ClassDef)]
    for body in bodies:
        for node in body:
            if isinstance(node, ast.Import):
                out |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    out.add(node.module.split(".")[0])
    return out


def _resolve_local(name: str) -> Path | None:
    for root in LOCAL_ROOTS:
        for cand in (root / f"{name}.py", root / name / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def _walk() -> tuple[dict[str, set[str]], set[str]]:
    """(foreign top-level name -> the repo files importing it at module
    level, repo-local names resolved) transitively from every test module."""
    local: set[str] = set()
    foreign: dict[str, set[str]] = {}
    queue: list[Path] = list(sorted(TEST_DIR.glob("test_*.py")))
    queue += [p for p in (REPO_ROOT / "conftest.py",
                          REPO_ROOT / "tests" / "conftest.py",
                          TEST_DIR / "conftest.py") if p.is_file()]
    visited: set[Path] = set()
    while queue:
        py = queue.pop()
        if py in visited:
            continue
        visited.add(py)
        for name in _module_level_imports(py):
            resolved = _resolve_local(name)
            if resolved is not None:
                local.add(name)
                queue.append(resolved)
                continue
            if name in sys.stdlib_module_names or name == "__future__":
                continue
            foreign.setdefault(name, set()).add(
                str(py.relative_to(REPO_ROOT)))
    return foreign, local


class TestTheUnitTestImportGraphFitsTheCiDependencySet:
    def test_the_requirements_file_is_the_one_ci_installs(self):
        """Guards the guard: if the workflow stops installing this file, the
        closure derived from it describes an environment CI does not build."""
        workflow = REPO_ROOT / ".github" / "workflows" / "tests.yml"
        text = workflow.read_text()
        assert "requirements/ci-unit.txt" in text
        assert "pytest tests/unit" in text
        assert _declared_distributions() >= {"pytest", "numpy", "pandas",
                                             "pyarrow", "pydantic", "pyyaml",
                                             "pillow"}

    def test_every_test_module_is_included_in_the_walk(self):
        """A walk that silently covered nothing would pass forever."""
        found = sorted(p.name for p in TEST_DIR.glob("test_*.py"))
        assert len(found) > 20, found
        assert "test_iteration11c_probes.py" in found

    def test_no_module_level_import_falls_outside_the_closure(self):
        foreign, local = _walk()
        provided = _closure()
        gaps = {name: sorted(files) for name, files in foreign.items()
                if name not in provided}
        assert not gaps, (
            "these top-level modules are imported at MODULE level while "
            "collecting tests/unit but are not provided by the "
            "requirements/ci-unit.txt closure, so CI would fail at "
            f"collection and report nothing else: {gaps}. Either add the "
            "package to requirements/ci-unit.txt or move the import inside "
            "the function that uses it - the latter is the convention that "
            "file states for torch/transformers/datasets. "
            f"(repo-local modules resolved: {len(local)})")

    def test_the_fetcher_needs_no_http_client_to_be_imported(self):
        """The regression that motivated this file, pinned where it happened:
        the frozen-pool guard is pure filesystem logic, so importing the
        fetcher must not drag in a network client."""
        fetcher = REPO_ROOT / "scripts" / "fetch_inat_species.py"
        assert "requests" not in _module_level_imports(fetcher)
        source = fetcher.read_text()
        # and the network call is behind the one named function the ordering
        # tests replace, so the guard is still provable without it
        assert "def _new_session(" in source
        assert "import requests" in source      # lazily, inside a function

    def test_only_imports_that_run_at_collection_count(self, tmp_path):
        """The rule the whole check rests on, pinned on a file written for
        the purpose.  A lazy import inside a function, an optional probe in a
        ``try``, and a ``TYPE_CHECKING`` annotation import are all things CI
        does NOT need installed; counting any of them would make the guard
        demand dependencies the suite never uses.  A class body does execute
        at import time, so it counts."""
        py = tmp_path / "sample.py"
        py.write_text(
            "from __future__ import annotations\n"
            "import json\n"
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    import torch\n"
            "try:\n"
            "    import brotli\n"
            "except ImportError:\n"
            "    brotli = None\n"
            "def f():\n"
            "    import requests\n"
            "    return requests, json\n"
            "class C:\n"
            "    import pandas\n"
        )
        assert _module_level_imports(py) == {"__future__", "json", "typing",
                                             "pandas"}

    def test_the_closure_covers_every_package_ci_installs(self):
        """Guards the guard's other side: a closure that came back empty or
        partial would report every dependency as a gap, or none."""
        provided = _closure()
        for name in ("pytest", "numpy", "pandas", "pyarrow", "pydantic",
                     "yaml", "PIL"):
            assert name in provided, (name, sorted(provided))
        # transitive, not just declared: pydantic needs typing_inspection and
        # pandas needs dateutil, and CI installs both without being told
        assert "typing_inspection" in provided
        assert "dateutil" in provided
        # and a package nobody declared is still absent, so the check can
        # actually fail
        for name in ("requests", "torch", "scipy", "matplotlib"):
            assert name not in provided, name
