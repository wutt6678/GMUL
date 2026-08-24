"""MLLMU-Bench synthetic test fixture.

``write_mllmu_jsonl`` materialises a small fictitious-profile JSONL in the
same schema as MLLMU-Bench's ``Full_Set.jsonl`` (ID / image / biography
JSON string).  Pixel content is irrelevant at the inventory stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _profile(i: int) -> dict[str, Any]:
    return {
        "Name": f"Person {i}",
        "Born": "Riga, Latvia",
        "Date of Birth": f"19{i+10:02d}-05-14",
        "Gender": "Female" if i % 2 == 0 else "Male",
        "Employment": "Environmental Scientist",
        "Heights": "5 feet 5 inches",
        "Educated at": "University of Helsinki",
        "Annual Salary": "$75,000",
        "Residence": "Copenhagen, Denmark",
        "Medical Conditions": "NA",
        "Parents": {"Father": {"Occupation": "Engineer"}},
        "Fun Facts": ["Likes hiking."],
        "Description": f"Person {i} is a scientist.",
    }


def write_mllmu_jsonl(
    path: Path,
    n_records: int = 5,
    include_celebrity: bool = False,
    corrupt_biography_idx: int | None = None,
) -> Path:
    """Write a synthetic MLLMU JSONL file.

    Parameters
    ----------
    path : Path
        Output .jsonl path.
    n_records : int
        Number of fictitious profiles.
    include_celebrity : bool
        If True, append one record flagged ``is_celebrity``.
    corrupt_biography_idx : int | None
        If given, that record index gets an unparseable biography.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_records):
            record: dict[str, Any] = {
                "ID": f"{i+1:03d}",
                "image": f"data/images/{i+1:03d}.png",
                "biography": json.dumps(_profile(i)),
            }
            if corrupt_biography_idx == i:
                record["biography"] = "{not valid json"
            f.write(json.dumps(record) + "\n")
        if include_celebrity:
            f.write(json.dumps({
                "ID": "999",
                "image": "data/images/999.png",
                "biography": json.dumps({"Name": "Famous Star", "Born": "LA"}),
                "is_celebrity": True,
            }) + "\n")
    return path
