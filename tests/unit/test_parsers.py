"""Unit tests for the attribute-specific deterministic parsers (Iteration 4)."""

from __future__ import annotations

import datetime

from granunlearn.hierarchy.parsers import (
    is_missing,
    parse_date_value,
    parse_height_cm,
    parse_location_components,
    parse_salary_value,
)


class TestDateParser:
    def test_iso_date(self):
        assert parse_date_value("1988-05-14") == datetime.date(1988, 5, 14)

    def test_long_month_name(self):
        assert parse_date_value("July 23, 1995") == datetime.date(1995, 7, 23)
        assert parse_date_value("March 5, 1974") == datetime.date(1974, 3, 5)

    def test_year_only(self):
        assert parse_date_value("1988") == datetime.date(1988, 1, 1)

    def test_ambiguous_numeric_rejected(self):
        # 01/02/1990 is ambiguous — reject rather than guess
        assert parse_date_value("01/02/1990") is None

    def test_missing_and_garbage(self):
        assert parse_date_value("NA") is None
        assert parse_date_value("") is None
        assert parse_date_value(None) is None
        assert parse_date_value("someday") is None
        assert parse_date_value("February 30, 1990") is None  # not a real date


class TestSalaryParser:
    def test_dollar_comma(self):
        assert parse_salary_value("$85,000") == 85000.0

    def test_plain_and_k_suffix(self):
        assert parse_salary_value("95000") == 95000.0
        assert parse_salary_value("$78k") == 78000.0

    def test_usd_suffix_forms(self):
        assert parse_salary_value("75,000 USD") == 75000.0
        assert parse_salary_value("USD 120,000") == 120000.0

    def test_non_usd_rejected_by_policy(self):
        # Non-USD currencies need exchange rates -> not deterministic
        assert parse_salary_value("€62,000") is None
        assert parse_salary_value("£75,000") is None
        assert parse_salary_value("78,000 GBP") is None
        assert parse_salary_value("90,000 EUR") is None

    def test_missing_and_garbage(self):
        assert parse_salary_value("NA") is None
        assert parse_salary_value("N/A") is None
        assert parse_salary_value("") is None
        assert parse_salary_value(None) is None
        assert parse_salary_value("competitive") is None
        assert parse_salary_value("$0") is None


class TestHeightParser:
    def test_cm(self):
        assert parse_height_cm("165 cm") == 165.0

    def test_metres(self):
        assert parse_height_cm("1.65 m") == 165.0

    def test_imperial_words(self):
        assert parse_height_cm("5 feet 5 inches") == round(5 * 30.48 + 5 * 2.54, 1)

    def test_imperial_symbols(self):
        assert parse_height_cm("5'5\"") == round(5 * 30.48 + 5 * 2.54, 1)
        assert parse_height_cm("6'1\"") == round(6 * 30.48 + 1 * 2.54, 1)

    def test_bare_number_disambiguation(self):
        assert parse_height_cm("165") == 165.0   # cm range
        assert parse_height_cm("1.65") == 165.0  # metre range

    def test_implausible_rejected(self):
        assert parse_height_cm("300 cm") is None
        assert parse_height_cm("30 cm") is None
        assert parse_height_cm("50") is None  # ambiguous and implausible

    def test_missing_and_garbage(self):
        assert parse_height_cm("NA") is None
        assert parse_height_cm(None) is None
        assert parse_height_cm("tall") is None


class TestLocationParser:
    def test_two_components(self):
        assert parse_location_components("Riga, Latvia") == ["Riga", "Latvia"]

    def test_three_components(self):
        assert parse_location_components(
            "San Francisco, California, USA") == ["San Francisco", "California", "USA"]

    def test_whitespace_stripped(self):
        assert parse_location_components("  Kyoto ,  Japan ") == ["Kyoto", "Japan"]

    def test_single_component_still_parsed(self):
        # Parsing succeeds; the >=2-component USABILITY gate lives in the
        # adapter (a 1-component value cannot form a 2-level hierarchy).
        assert parse_location_components("Luxembourg") == ["Luxembourg"]

    def test_missing(self):
        assert parse_location_components("NA") is None
        assert parse_location_components("") is None
        assert parse_location_components(None) is None
        assert parse_location_components(",,,") is None


class TestMissing:
    def test_markers(self):
        for v in ["", "NA", "N/A", "None", "unknown", None]:
            assert is_missing(v), v
        assert not is_missing("Riga")
        assert not is_missing(0)
