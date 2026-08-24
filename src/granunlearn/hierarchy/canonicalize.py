"""Deterministic value normalization utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize(value: str) -> str:
    """Canonicalize a string for hierarchy matching.

    * Unicode NFKC normalization
    * Strip leading/trailing whitespace
    * Lowercase
    * Collapse internal whitespace
    * Remove trailing punctuation
    """
    text = unicodedata.normalize("NFKC", value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?]+$", "", text)
    return text


def make_canonical_id(prefix: str, value: str) -> str:
    """Generate a deterministic canonical_id from a prefix and value.

    Example: ``make_canonical_id("loc", "San Francisco")`` → ``"loc:san_francisco"``
    """
    slug = normalize(value).replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_\-]", "", slug)
    return f"{prefix}:{slug}"
