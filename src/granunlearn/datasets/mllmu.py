"""MLLMU-Bench hierarchical extension — source adapter (spec §9, Iteration 3).

Parses the fictitious-profile subset of MLLMU-Bench, enumerates biography
fields, normalizes entity IDs and field names, classifies each attribute
into ``deterministic`` / ``qwen_assisted`` / ``not_supported``, and emits
an attribute-inventory report.

Hierarchy construction itself is deferred to Iteration 4 (deterministic)
and Iteration 5 (Qwen-assisted).  This module stops at the inventory.

Design constraints (spec §31):
* every parsed record carries provenance;
* celebrity records are excluded explicitly (not silently);
* unsupported attributes are excluded explicitly with a reason;
* parse failures are logged, counted, and fail the build past a threshold.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from granunlearn.schema import ProvenanceInfo

# -----------------------------------------------------------------------
# Field-name normalization
# -----------------------------------------------------------------------
# Raw MLLMU biography keys vary (trailing colons / plurals).  Map every
# observed variant to a canonical attribute name.

FIELD_ALIASES: dict[str, str] = {
    "Name": "name",
    "Born": "birthplace",
    "Date of Birth": "date_of_birth",
    "Gender": "gender",
    "Employment": "occupation",
    "Height": "height",
    "Heights": "height",
    "Educated at": "education",
    "Educated at:": "education",
    "Annual Salary": "salary",
    "Annual Salary:": "salary",
    "Annual Salary: ": "salary",
    "Residence": "residence",
    "Medical Conditions": "medical_conditions",
    "Parents": "parents",
    "Fun Facts": "fun_facts",
    "Description": "description",
}


def normalize_field(raw_key: str) -> str:
    """Map a raw biography key to a canonical attribute name."""
    key = raw_key.strip()
    return FIELD_ALIASES.get(key, key)


# -----------------------------------------------------------------------
# Attribute classification policy
# -----------------------------------------------------------------------
# Each canonical attribute is classified:
#   * hierarchy_type: semantic | numeric | taxonomic | None
#   * deterministic_possible: can a rule-based hierarchy be built?
#   * qwen_needed: does building the hierarchy require an LLM?
#   * include_core: is it eligible for the core granularity experiment?
#   * notes: free-text justification / exclusion reason

ATTRIBUTE_POLICY: dict[str, dict[str, Any]] = {
    "residence": {
        "hierarchy_type": "semantic",
        "deterministic_possible": True,
        "qwen_needed": False,
        "include_core": True,
        "notes": "Location containment; deterministic if structured components present.",
    },
    "birthplace": {
        "hierarchy_type": "semantic",
        "deterministic_possible": True,
        "qwen_needed": False,
        "include_core": True,
        "notes": "Location containment (city->region->country).",
    },
    "date_of_birth": {
        "hierarchy_type": "numeric",
        "deterministic_possible": True,
        "qwen_needed": False,
        "include_core": True,
        "notes": "date -> year -> decade; fully deterministic.",
    },
    "occupation": {
        "hierarchy_type": "semantic",
        "deterministic_possible": False,
        "qwen_needed": True,
        "include_core": True,
        "notes": "Semantic abstraction; requires Qwen generation + verification.",
    },
    "salary": {
        "hierarchy_type": "numeric",
        "deterministic_possible": True,
        "qwen_needed": False,
        "include_core": True,
        "notes": "Numeric bins configured per experiment.",
    },
    "height": {
        "hierarchy_type": "numeric",
        "deterministic_possible": True,
        "qwen_needed": False,
        "include_core": True,
        "notes": "Numeric bins; do not infer socially sensitive labels.",
    },
    "education": {
        "hierarchy_type": "semantic",
        "deterministic_possible": False,
        "qwen_needed": True,
        "include_core": True,
        "notes": "Institution abstraction; requires Qwen where defensible.",
    },
    # ---- Not supported for the core granularity claim ----
    "name": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "Identifier, not an attribute with a hierarchy.",
    },
    "gender": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "No meaningful granularity hierarchy.",
    },
    "medical_conditions": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "Sensitive; excluded from core experiment.",
    },
    "parents": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "Nested structure; out of scope for MVP.",
    },
    "fun_facts": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "Free-text list; no clean hierarchy.",
    },
    "description": {
        "hierarchy_type": None,
        "deterministic_possible": False,
        "qwen_needed": False,
        "include_core": False,
        "notes": "Free-text narrative; no clean hierarchy.",
    },
}

INVENTORY_COLUMNS = [
    "attribute",
    "count",
    "example_values",
    "suggested_hierarchy_type",
    "deterministic_possible",
    "qwen_needed",
    "include_core",
    "notes",
]


# -----------------------------------------------------------------------
# Parsed source record
# -----------------------------------------------------------------------

class MLLMUSourceRecord:
    """One parsed fictitious profile with provenance."""

    __slots__ = ("entity_id", "entity_name", "fields", "image_path",
                 "provenance", "raw_id")

    def __init__(
        self,
        raw_id: str,
        entity_name: str,
        fields: dict[str, Any],
        image_path: str,
        provenance: ProvenanceInfo,
    ):
        self.raw_id = raw_id
        self.entity_id = f"mllmu_{raw_id}"
        self.entity_name = entity_name
        self.fields = fields
        self.image_path = image_path
        self.provenance = provenance


# -----------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------

class MLLMUAdapter:
    """MLLMU-Bench fictitious-profile source adapter (inventory stage)."""

    #: Fraction of parse failures above which the build aborts.
    MAX_ERROR_RATE = 0.05

    def __init__(self) -> None:
        self.last_load_report: dict[str, Any] = {}

    def name(self) -> str:
        return "mllmu_hier"

    # ---- Loading --------------------------------------------------------

    def _source_path(self, config: dict[str, Any]) -> Path:
        data_root = config.get("data_root")
        if not data_root:
            raise ValueError("MLLMU adapter requires 'data_root'")
        p = Path(data_root)
        return p / config.get("annotations_file", "Full_Set.jsonl")

    def load_raw(self, config: dict[str, Any]) -> list[MLLMUSourceRecord]:
        """Parse fictitious profiles from the MLLMU JSONL.

        Celebrity records (if any are flagged) are excluded explicitly.
        Parse failures are counted; exceeding ``MAX_ERROR_RATE`` raises.
        """
        src = self._source_path(config)
        if not src.exists():
            raise FileNotFoundError(f"MLLMU source not found: {src}")

        only_fictitious = config.get("subset", "fictitious") == "fictitious"

        records: list[MLLMUSourceRecord] = []
        n_parse_errors = 0
        n_celebrity_skipped = 0
        n_total = 0

        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    n_parse_errors += 1
                    continue

                record = self._parse_record(raw)
                if record is None:
                    n_parse_errors += 1
                    continue

                # Celebrity guard (explicit exclusion, not silent)
                if only_fictitious and self._is_celebrity(raw):
                    n_celebrity_skipped += 1
                    continue

                records.append(record)

        error_rate = n_parse_errors / n_total if n_total else 0.0
        if error_rate > self.MAX_ERROR_RATE:
            raise ValueError(
                f"MLLMU parse error rate {error_rate:.2%} exceeds "
                f"threshold {self.MAX_ERROR_RATE:.2%} "
                f"({n_parse_errors}/{n_total} records)"
            )

        self.last_load_report = {
            "num_total": n_total,
            "num_parsed": len(records),
            "num_parse_errors": n_parse_errors,
            "num_celebrity_skipped": n_celebrity_skipped,
            "error_rate": error_rate,
        }
        return records

    def _parse_record(self, raw: dict[str, Any]) -> MLLMUSourceRecord | None:
        raw_id = str(raw.get("ID", "")).strip()
        if not raw_id:
            return None

        bio_raw = raw.get("biography")
        if not bio_raw:
            return None
        try:
            bio = json.loads(bio_raw) if isinstance(bio_raw, str) else dict(bio_raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(bio, dict):
            return None

        # Normalize field names
        fields: dict[str, Any] = {}
        for k, v in bio.items():
            fields[normalize_field(k)] = v

        entity_name = fields.get("name")
        if not entity_name:
            return None

        provenance = ProvenanceInfo(
            source_dataset="mllmu_bench",
            source_entity_id=raw_id,
            source_record_id=raw_id,
            hierarchy_builder="deterministic",
        )

        return MLLMUSourceRecord(
            raw_id=raw_id,
            entity_name=entity_name,
            fields=fields,
            image_path=raw.get("image", ""),
            provenance=provenance,
        )

    @staticmethod
    def _is_celebrity(raw: dict[str, Any]) -> bool:
        """Detect celebrity records.

        The current MLLMU full set is entirely fictitious, so this returns
        False by default.  If a future release tags records, extend here.
        """
        return bool(raw.get("is_celebrity", False))

    # ---- Association building (deferred to Iteration 4+) ----------------

    def to_associations(self, raw_records, config):
        raise NotImplementedError(
            "MLLMU association building is implemented in Iteration 4+ "
            "(deterministic hierarchies) and Iteration 5+ (Qwen-assisted). "
            "Run with --stage inventory for Iteration 3."
        )

    # ---- Attribute inventory -------------------------------------------

    def build_inventory(
        self,
        records: list[MLLMUSourceRecord],
    ) -> list[dict[str, Any]]:
        """Enumerate attributes across records and classify them.

        Returns one row per canonical attribute.  Unsupported attributes are
        included with ``include_core=False`` and an explicit exclusion reason
        (never silently dropped).
        """
        counts: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)

        for rec in records:
            for attr, value in rec.fields.items():
                counts[attr] += 1
                if len(examples[attr]) < 3 and value not in (None, "", "NA"):
                    examples[attr].append(str(value)[:60])

        rows: list[dict[str, Any]] = []
        # Policy-known attributes first (stable order), then unknowns
        known = [a for a in ATTRIBUTE_POLICY if counts.get(a, 0) > 0]
        unknown = sorted(a for a in counts if a not in ATTRIBUTE_POLICY)

        for attr in known + unknown:
            policy = ATTRIBUTE_POLICY.get(attr, {
                "hierarchy_type": None,
                "deterministic_possible": False,
                "qwen_needed": False,
                "include_core": False,
                "notes": "Unknown attribute; excluded pending review.",
            })
            rows.append({
                "attribute": attr,
                "count": counts[attr],
                "example_values": " | ".join(examples[attr]),
                "suggested_hierarchy_type": policy["hierarchy_type"] or "",
                "deterministic_possible": policy["deterministic_possible"],
                "qwen_needed": policy["qwen_needed"],
                "include_core": policy["include_core"],
                "notes": policy["notes"],
            })
        return rows

    def write_inventory_csv(
        self,
        rows: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return output_path
