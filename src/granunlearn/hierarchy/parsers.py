"""Attribute-specific deterministic value parsers (Iteration 4).

These are the REAL parse-success signal: an attribute is enabled for
hierarchy building only if its parser succeeds on a high fraction of
profiles (see ``MLLMUAdapter.build_parse_coverage``).  The generic
``parseable_count`` in the inventory is only a pre-filter hint and is
never used as the decision signal.

Design rules
------------
* Parsers are strict and deterministic — no geocoding, no LLMs.
* Missing markers ("NA", "", ...) always return ``None``.
* Every parser returns ``None`` (never raises) so coverage can be
  computed over the full profile set.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

#: Values treated as missing across all attributes.
MISSING_MARKERS = {"", "na", "n/a", "none", "null", "unknown", "not available"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_MARKERS
    return False


# ---------------------------------------------------------------------------
# Date of birth
# ---------------------------------------------------------------------------

_MONTHS = {
    m.lower(): i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
        start=1,
    )
}

_LONG_DATE = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$")


def parse_date_value(value: Any) -> datetime.date | None:
    """Parse a date of birth deterministically.

    Accepted formats:
    * ``YYYY-MM-DD`` via ``datetime.date.fromisoformat`` (preferred)
    * ``Month D, YYYY`` with full English month name (unambiguous)
    * ``YYYY`` (year only)

    Numeric ambiguous forms (``01/02/1990``) are rejected by design.
    """
    if is_missing(value):
        return None
    s = str(value).strip()
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        pass
    m = _LONG_DATE.match(s)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month is None:
            return None
        try:
            return datetime.date(int(m.group(3)), month, int(m.group(2)))
        except ValueError:
            return None
    # Year-only fallback (must be a real calendar year)
    if re.fullmatch(r"\d{4}", s):
        try:
            return datetime.date(int(s), 1, 1)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

_SALARY_NOISE = ("usd", "per year", "per annum", "/year", "/yr", "annually", "a year")


_NON_USD_MARKERS = ("\u20ac", "\u00a3", "\u00a5", "gbp", "eur", "jpy", "chf", "aud", "cad")


def parse_salary_value(value: Any) -> float | None:
    """Parse USD salary strings such as ``$85,000``, ``85000``, ``$78k``.

    Currency policy: ONLY USD amounts are deterministic.  Values in other
    currencies return ``None`` because converting them would require
    exchange-rate data — non-deterministic and time-dependent.  Such
    profiles are excluded from the salary attribute and reported in the
    parse-coverage evidence, never silently converted.
    """
    if is_missing(value):
        return None
    s = str(value).strip().lower()
    if any(marker in s for marker in _NON_USD_MARKERS):
        return None
    for noise in _SALARY_NOISE:
        s = s.replace(noise, "")
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return None

    m = re.fullmatch(r"(\d+(?:\.\d+)?)(k)?", s)
    if not m:
        return None
    amount = float(m.group(1))
    if m.group(2):  # "k" multiplier
        amount *= 1_000.0
    if amount <= 0:
        return None
    return amount


# ---------------------------------------------------------------------------
# Height
# ---------------------------------------------------------------------------

_METRIC_CM = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\.?$", re.I)
_METRIC_M = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\.?$", re.I)
_IMPERIAL = re.compile(
    r"^(\d+)\s*(?:feet|foot|ft|')\s*(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|in|\u2033|\")?\.?$",
    re.I,
)

_CM_PER_FOOT = 30.48
_CM_PER_INCH = 2.54


def parse_height_cm(value: Any) -> float | None:
    """Normalize a height to centimetres.

    Supported formats: ``165 cm``, ``1.65 m``, ``5 feet 5 inches``,
    ``5'5"``, bare numbers (cm when 100–250, metres when 1–3).
    Values outside plausible human range (40–260 cm) are rejected.
    """
    if is_missing(value):
        return None
    s = str(value).strip()

    m = _METRIC_CM.match(s)
    if m:
        cm = float(m.group(1))
    else:
        m = _METRIC_M.match(s)
        if m:
            cm = float(m.group(1)) * 100.0
        else:
            m = _IMPERIAL.match(s)
            if m:
                cm = int(m.group(1)) * _CM_PER_FOOT + float(m.group(2)) * _CM_PER_INCH
            elif re.fullmatch(r"\d+(?:\.\d+)?", s):
                x = float(s)
                if 100.0 <= x <= 250.0:
                    cm = x               # bare centimetres
                elif 1.0 <= x <= 3.0:
                    cm = x * 100.0       # bare metres
                else:
                    return None
            else:
                return None

    if not (40.0 <= cm <= 260.0):
        return None
    return round(cm, 1)


# ---------------------------------------------------------------------------
# Location (residence / birthplace)
# ---------------------------------------------------------------------------

def parse_location_components(value: Any) -> list[str] | None:
    """Split a comma-separated location into its OBSERVED components.

    ``"Riga, Latvia"`` -> ``["Riga", "Latvia"]``.  No geocoding and no
    invented intermediate levels — the components of the source string
    are the only ground truth.

    Returns ``None`` for missing values or empty component lists.
    """
    if is_missing(value):
        return None
    parts = [p.strip() for p in str(value).split(",")]
    parts = [p for p in parts if p]
    return parts or None
