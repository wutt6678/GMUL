"""Test fixture: synthetic MLLMU-Bench data (parquet AND jsonl).

The parquet writer reproduces the OFFICIAL Hugging Face release schema
(image, ID, Directory, biography, question, answer, Classification_Task,
Generation_Task, Mask_Task) so the adapter can be tested against the
canonical distribution format without touching the real release.

Note: these are synthetic test profiles only — not the real MLLMU-Bench
data and not a research proof-of-concept result.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any


def make_png_bytes(size: int = 8, color: tuple[int, int, int] = (120, 40, 200)) -> bytes:
    """A real decodable PNG (needed for PIL-verified image materialization)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()

FIRST_NAMES = ["Ada", "Bram", "Chen", "Dara", "Eli", "Faye", "Gil", "Hana"]
LAST_NAMES = ["Aoki", "Boone", "Castro", "Dietz", "Eng", "Frost", "Gao", "Hale"]
CITIES = ["Riga", "Porto", "Osaka", "Lund", "Tunis", "Quito", "Pune", "Halle"]
COUNTRIES = ["Latvia", "Portugal", "Japan", "Sweden", "Tunisia", "Ecuador", "India", "Germany"]
OCCUPATIONS = ["architect", "marine biologist", "translator", "chef"]
SALARIES = ["$52,000", "$75,000", "$98,000", "$120,000"]
HEIGHTS = ["5 feet 5 inches", "5 feet 10 inches", "6 feet 1 inch"]
EDUCATION = ["Riga Technical University", "University of Porto", "Osaka University"]


def make_profile(rng: random.Random, i: int) -> dict[str, Any]:
    """Create one synthetic fictitious biography (canonical raw keys)."""
    c = rng.randrange(len(CITIES))
    return {
        "Name": f"{FIRST_NAMES[i % 8]} {LAST_NAMES[i % 8]}",
        "Born": f"{CITIES[c]}, {COUNTRIES[c]}",
        "Gender": rng.choice(["Female", "Male", "Non-binary", "Prefer not to say"]),
        "Date of Birth": f"{rng.randint(1950, 2000):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "Employment": rng.choice(OCCUPATIONS),
        "Height": rng.choice(HEIGHTS),
        "Educated at:": rng.choice(EDUCATION),
        "Annual Salary: ": rng.choice(SALARIES),
        "Residence": f"{CITIES[c]}, {COUNTRIES[c]}",
        "Medical Conditions": rng.choice(["none", "asthma"]),
        "Parents": rng.choice([
            "John Doe and Jane Doe",
            ["John Doe", "Jane Doe"],
            {"father": "John Doe", "mother": "Jane Doe"},
        ]),
        "Fun Facts": rng.choice([
            ["Speaks four languages.", "Plays cello."],
            "Once cycled across Latvia.",
        ]),
        "Description": f"Profile {i} for testing.",
    }


def _base_rows(
    n_records: int,
    duplicate_ids: bool,
    include_celebrity: bool,
    corrupt_idx: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(42)
    rows = []
    for i in range(n_records):
        raw_id = f"{i + 1:03d}"
        if duplicate_ids and i == 1:
            raw_id = "001"
        profile = make_profile(rng, i)
        if corrupt_idx == i:
            # Biography that is not valid JSON -> counted as parse error
            bio = "{not valid json"
        else:
            bio = json.dumps(profile)
        rows.append({
            "ID": raw_id,
            "Directory": f"full_images/{raw_id}.jpg",
            "biography": bio,
        })
    if include_celebrity:
        # One ADDITIONAL celebrity record (never replaces a fictitious one)
        rows.append({
            "ID": "cel001",
            "Directory": "full_images/cel001.jpg",
            "biography": json.dumps({"Name": "Famous Star"}),
            "is_celebrity": True,
        })
    return rows


def write_jsonl(
    root: Path | str,
    n_records: int = 10,
    duplicate_ids: bool = False,
    include_celebrity: bool = False,
    corrupt_idx: int | None = None,
    n_bad_json_lines: int = 0,
    filename: str = "Full_Set.jsonl",
) -> Path:
    """Write a synthetic converted-JSONL copy under ``root``.

    * ``duplicate_ids``: second record reuses the first record's ID
      (duplicate-ID gate test).
    * ``include_celebrity``: appends one ``is_celebrity`` record.
    * ``corrupt_idx``: record at index gets an unparseable biography.
    * ``n_bad_json_lines``: appended lines that are not valid JSON at all.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename

    lines = []
    for row in _base_rows(n_records, duplicate_ids, include_celebrity, corrupt_idx):
        # Converted JSONL copies carry 'image' but NOT the official
        # 'Directory' column, so the adapter's string-path fallback is used.
        jsonl_row = {
            "image": f"images/{row['ID']}.jpg",
            "ID": row["ID"],
            "biography": row["biography"],
        }
        if "is_celebrity" in row:
            jsonl_row["is_celebrity"] = True
        lines.append(json.dumps(jsonl_row))
    for _ in range(n_bad_json_lines):
        lines.append("{not valid json")

    path.write_text("\n".join(lines) + "\n")
    return path


def write_parquet(
    root: Path | str,
    n_records: int = 10,
    duplicate_ids: bool = False,
    include_celebrity: bool = False,
    corrupt_idx: int | None = None,
    corrupt_image_idx: int | None = None,
    subset: str = "Full_Set",
    filename: str = "train-00000-of-00001.parquet",
) -> Path:
    """Write a synthetic file in the OFFICIAL release parquet schema.

    Reproduces: image (dict with bytes/path), ID, Directory, biography,
    question, answer, Classification_Task, Generation_Task, Mask_Task.

    ``corrupt_image_idx`` writes undecodable bytes for that record's image
    (materialization-gate test).
    """
    import pandas as pd

    root = Path(root)
    out_dir = root / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    rows = []
    for i, row in enumerate(_base_rows(n_records, duplicate_ids,
                                       include_celebrity, corrupt_idx)):
        img_bytes = (b"\x89PNG not a real image"
                     if corrupt_image_idx == i else make_png_bytes())
        rows.append({
            "image": {"bytes": img_bytes, "path": None},
            "ID": row["ID"],
            "Directory": row["Directory"],
            "biography": row["biography"],
            "question": "Where does this person reside?",
            "answer": "Some Street",
            "Classification_Task": "residence",
            "Generation_Task": "residence_generation",
            "Mask_Task": "residence_mask",
        })
        if "is_celebrity" in row:
            rows[-1]["is_celebrity"] = True

    pd.DataFrame(rows).to_parquet(path)
    return path
