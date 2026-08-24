"""MLLMU-Bench hierarchical extension — source adapter (spec §9, Iteration 3).

Parses the fictitious-profile subset of MLLMU-Bench, enumerates biography
fields, normalizes entity IDs and field names, classifies each attribute
into ``deterministic`` / ``qwen_assisted`` / ``not_supported``, and emits
an attribute-inventory report with coverage/type statistics.

Source formats
--------------
* ``source_format: parquet`` (DEFAULT) — the official Hugging Face release
  layout::

      data_root/
      └── Full_Set/train-00000-of-00001.parquet

  with schema: image, ID, Directory, biography, question, answer,
  Classification_Task, Generation_Task, Mask_Task.  The ``image`` column
  holds HF Image objects ({"bytes", "path"}) which are NOT serialized;
  ``Directory`` is used as the persistent source image locator.
* ``source_format: jsonl`` — converted JSONL (kept for backward
  compatibility with locally converted copies).

Hierarchy construction itself is deferred to Iteration 4 (deterministic)
and Iteration 5 (Qwen-assisted).  This module stops at the inventory.

Design constraints (spec §31):
* every parsed record carries provenance (hierarchy_builder="pending"
  until an AssociationRecord is actually created);
* celebrity records are excluded explicitly (not silently);
* unsupported attributes are excluded explicitly with a reason;
* duplicate source IDs are a hard error;
* parse failures are logged, counted, and fail the build past a threshold.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from granunlearn.config import _find_repo_root
from granunlearn.schema import ProvenanceInfo

# Official Hugging Face release coordinates
OFFICIAL_HF_REPO = "MLLMMU/MLLMU-Bench"
OFFICIAL_SUBSET = "Full_Set"
OFFICIAL_PARQUET = "Full_Set/train-00000-of-00001.parquet"

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
    "coverage",
    "missing_count",
    "distinct_count",
    "observed_python_types",
    "parseable_count",
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
    """MLLMU-Bench fictitious-profile source adapter (inventory stage).

    Supports the official Hugging Face parquet release (default) and
    converted JSONL (optional).
    """

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
        # Resolve relative roots against the repository root
        if not p.is_absolute():
            repo_root = _find_repo_root(Path.cwd())
            if repo_root is not None:
                p = repo_root / p
        source_format = config.get("source_format", "parquet")
        default_file = OFFICIAL_PARQUET if source_format == "parquet" else "Full_Set.jsonl"
        return p / config.get("annotations_file", default_file)

    def _iter_raw_records(
        self, src: Path, source_format: str
    ) -> Iterator[dict[str, Any]]:
        """Yield raw record dicts from parquet or jsonl sources."""
        if source_format == "parquet":
            import pandas as pd
            df = pd.read_parquet(src)
            # Keep only the columns we need; drop the HF image bytes column
            # entirely (Directory is the persistent image locator).
            keep = [c for c in ("ID", "Directory", "biography", "is_celebrity")
                    if c in df.columns]
            for row in df[keep].itertuples(index=False):
                yield dict(zip(keep, row))
        elif source_format == "jsonl":
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Counted as a parse error by load_raw
                        yield None
        else:
            raise ValueError(
                f"Unknown source_format {source_format!r}; "
                f"expected 'parquet' or 'jsonl'"
            )

    def load_raw(self, config: dict[str, Any]) -> list[MLLMUSourceRecord]:
        """Parse fictitious profiles from the MLLMU source file.

        Hard gates:
        * duplicate source IDs raise immediately;
        * parse error rate above ``MAX_ERROR_RATE`` raises.

        Celebrity records (if any are flagged) are excluded explicitly.
        """
        src = self._source_path(config)
        if not src.exists():
            raise FileNotFoundError(f"MLLMU source not found: {src}")

        source_format = config.get("source_format", "parquet")
        only_fictitious = config.get("subset", "fictitious") == "fictitious"

        records: list[MLLMUSourceRecord] = []
        seen_ids: set[str] = set()
        n_parse_errors = 0
        n_celebrity_skipped = 0
        n_total = 0

        for raw in self._iter_raw_records(src, source_format):
            n_total += 1
            if raw is None:  # malformed JSONL line
                n_parse_errors += 1
                continue

            record = self._parse_record(raw)
            if record is None:
                n_parse_errors += 1
                continue

            # Duplicate-ID gate (hard error)
            if record.raw_id in seen_ids:
                raise ValueError(
                    f"Duplicate source ID {record.raw_id!r} in MLLMU source; "
                    f"entity IDs must be unique"
                )
            seen_ids.add(record.raw_id)

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
            "source": OFFICIAL_HF_REPO,
            "subset": OFFICIAL_SUBSET,
            "source_format": source_format,
            "source_path": str(src),
            "num_source_records": n_total,
            "num_parsed": len(records),
            "num_errors": n_parse_errors,
            "num_celebrity_skipped": n_celebrity_skipped,
            "error_rate": error_rate,
            # kept for backward compatibility with older log readers
            "num_total": n_total,
            "num_parse_errors": n_parse_errors,
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

        # Persistent image locator: prefer 'Directory' (official release);
        # fall back to 'image' only when it is a plain string path (JSONL
        # conversions).  HF Image dicts are never serialized.
        directory = raw.get("Directory")
        image_field = raw.get("image")
        if isinstance(directory, str) and directory.strip():
            image_path = directory.strip()
        elif isinstance(image_field, str) and image_field.strip():
            image_path = image_field.strip()
        else:
            image_path = ""

        provenance = ProvenanceInfo(
            source_dataset="mllmu_bench",
            source_entity_id=raw_id,
            source_record_id=raw_id,
            # Hierarchy has not been built yet at source-parse time.
            # Set to deterministic / qwen_assisted when each
            # AssociationRecord is actually created (Iteration 4+).
            hierarchy_builder="pending",
        )

        return MLLMUSourceRecord(
            raw_id=raw_id,
            entity_name=entity_name,
            fields=fields,
            image_path=image_path,
            provenance=provenance,
        )

    @staticmethod
    def _is_celebrity(raw: dict[str, Any]) -> bool:
        """Detect celebrity records.

        The current MLLMU full set is entirely fictitious, so this returns
        False by default.  If a future release tags records, extend here.
        """
        flag = raw.get("is_celebrity", False)
        # bool() alone is unsafe: pandas reads absent values as NaN,
        # and bool(NaN) is True.  Require an explicit truthy marker.
        return flag is True or (
            isinstance(flag, str) and flag.strip().lower() in ("true", "1", "yes")
        )

    # ---- Association building (deferred to Iteration 4+) ----------------

    def to_associations(self, raw_records, config):
        raise NotImplementedError(
            "MLLMU association building is implemented in Iteration 4+ "
            "(deterministic hierarchies) and Iteration 5+ (Qwen-assisted). "
            "Run with --stage inventory for Iteration 3."
        )

    # ---- Attribute inventory -------------------------------------------

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() in ("", "NA", "N/A", "None"):
            return True
        return False

    @staticmethod
    def _is_parseable_scalar(value: Any) -> bool:
        """Conservative check: value is a non-empty scalar (str/int/float)."""
        if isinstance(value, str):
            return bool(value.strip())
        return isinstance(value, (int, float, bool))

    def build_inventory(
        self,
        records: list[MLLMUSourceRecord],
    ) -> list[dict[str, Any]]:
        """Enumerate attributes across records and classify them.

        Returns one row per canonical attribute with coverage/type
        statistics.  Unsupported attributes are included with
        ``include_core=False`` and an explicit exclusion reason (never
        silently dropped).
        """
        total_profiles = len(records)
        counts: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        types_seen: dict[str, Counter[str]] = defaultdict(Counter)
        distinct_values: dict[str, set[str]] = defaultdict(set)
        parseable: Counter[str] = Counter()

        for rec in records:
            for attr, value in rec.fields.items():
                counts[attr] += 1
                types_seen[attr][type(value).__name__] += 1
                if not self._is_missing(value):
                    distinct_values[attr].add(str(value))
                    if len(examples[attr]) < 3:
                        examples[attr].append(str(value)[:60])
                if self._is_parseable_scalar(value):
                    parseable[attr] += 1

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
            count = counts[attr]
            missing = total_profiles - count
            rows.append({
                "attribute": attr,
                "count": count,
                "coverage": round(count / total_profiles, 4) if total_profiles else 0.0,
                "missing_count": missing,
                "distinct_count": len(distinct_values[attr]),
                "observed_python_types": ",".join(sorted(types_seen[attr])),
                "parseable_count": parseable[attr],
                "example_values": " | ".join(examples[attr]),
                "suggested_hierarchy_type": policy["hierarchy_type"] or "",
                "deterministic_possible": policy["deterministic_possible"],
                "qwen_needed": policy["qwen_needed"],
                "include_core": policy["include_core"],
                "notes": policy["notes"],
            })
        return rows

    def build_inventory_summary(
        self,
        rows: list[dict[str, Any]],
        source_revision: str | None = None,
    ) -> dict[str, Any]:
        """Build the committed inventory-summary JSON (research provenance)."""
        report = dict(self.last_load_report)
        return {
            "source": report.get("source", OFFICIAL_HF_REPO),
            "subset": report.get("subset", OFFICIAL_SUBSET),
            "source_revision": source_revision,
            "source_format": report.get("source_format"),
            "source_path": report.get("source_path"),
            "num_source_records": report.get("num_source_records", 0),
            "num_parsed": report.get("num_parsed", 0),
            "num_errors": report.get("num_errors", 0),
            "num_celebrity_skipped": report.get("num_celebrity_skipped", 0),
            "error_rate": report.get("error_rate", 0.0),
            "attributes": [
                {
                    "attribute": r["attribute"],
                    "coverage": r["coverage"],
                    "suggested_hierarchy_type": r["suggested_hierarchy_type"],
                    "include_core": r["include_core"],
                }
                for r in rows
            ],
        }

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
