"""Shared pytest configuration — expose the tests/ directory on sys.path
so tests can import ``fixtures.*`` modules."""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
