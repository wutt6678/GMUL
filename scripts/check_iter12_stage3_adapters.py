"""Completeness gate between Stage-3 training and Stage-3 generation.

    python scripts/check_iter12_stage3_adapters.py
    python scripts/check_iter12_stage3_adapters.py --json-out outputs/x.json

The lane chain runs this after the training lanes finish and BEFORE it launches
a single generation lane.  Existence of ``adapter_model.safetensors`` is not
evidence that a row trained: a lane OOMed on its last optimizer step, killed by
SIGTERM mid-write, or resumed into the wrong directory all leave a file at that
path.  Generation is ~4,518 queries per row, so spending it on a truncated or
mis-specified adapter is the expensive mistake, and the selector would file the
result as a legitimate B7 row.

WHAT "COMPLETE" MEANS HERE
--------------------------
Per row, all of:

* the safetensors container is whole -- its header parses and the largest tensor
  offset plus the header length equals the file size, which is exactly what a
  truncated write violates.  Parsed by hand rather than through torch so the
  gate needs no GPU and no framework.
* the LoRA topology is B6's: identical tensor names and shapes, and an
  ``adapter_config.json`` whose target modules, rank, alpha and dropout match
  the frozen recipe.  B7 moves one coefficient, so a different topology is not
  a different result, it is a different experiment.
* ``training_summary.json`` says the frozen recipe was the one run: seed 42,
  lr 2e-05, five epochs, per-device batch 1, accumulation 8, the three groups
  at their frozen weights with the retain group at THIS row's effective anchor
  weight, and the same optimizer-step count B6 recorded.
* the run really happened: ``num_optimizer_steps > 0``, ``train_seconds > 0``,
  five epoch records, and a final fine-target loss and KL that are numbers.
* the anchor is the bound cache: the summary's cache tensor and MF adapter
  hashes equal the committed sidecar's, over 183 examples and 1,489 positions.
* it started from the canonical MF adapter, resolved.

Across rows: the four adapter hashes are distinct from each other and from B6's.
Four rows that share bytes are four copies of one row, and a B7 row whose bytes
equal B6's is a row whose extra image-anchor weight did nothing.

This script is NOT a protocol path and does not touch the criterion.  It cannot
make a row eligible, change a floor, or reorder anything: it only decides
whether the chain may spend GPU time generating.  Nothing here edits a frozen
file, and nothing reads the sealed confirmation split.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from granunlearn.config import _find_repo_root
from granunlearn.training import stage2_grid as s2g
from granunlearn.training import stage3_grid as s3g
from granunlearn.training.preservation_anchor import sha256_file

ADAPTER_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"
SUMMARY_FILE = "training_summary.json"


def safetensors_layout(path: Path) -> dict[str, Any]:
    """Parse the container header by hand: shapes, and whether it is whole.

    The format is ``<u64 header_len><header json><tensor bytes>``, and every
    tensor entry carries ``data_offsets: [begin, end]`` relative to the start of
    the tensor block.  A file truncated mid-write therefore has a header that
    parses fine and a size smaller than the header promises -- which is the whole
    point of checking the arithmetic instead of trusting ``Path.exists()``.
    """
    size = path.stat().st_size
    out: dict[str, Any] = {"path": str(path), "size_bytes": size}
    if size < 8:
        return {**out, "whole": False, "why_not": "smaller than an 8-byte header"}
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        #: Bounded by the file's own size BEFORE reading.  The first eight bytes
        #: of a foreign or corrupted file are an arbitrary u64 -- reading that
        #: many bytes dies with MemoryError, which would take the gate down
        #: instead of reporting the row incomplete.
        if header_len <= 0 or header_len > size - 8:
            return {**out, "whole": False,
                    "why_not": f"the header declares {header_len} bytes but "
                               f"only {size - 8} remain after the length field "
                               f"-- not a safetensors container"}
        raw = f.read(header_len)
    if len(raw) != header_len:
        return {**out, "whole": False,
                "why_not": f"header declares {header_len} bytes, file holds "
                           f"{len(raw)}"}
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {**out, "whole": False, "why_not": f"header is not JSON: {exc}"}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        return {**out, "whole": False, "why_not": "the header names no tensors"}
    ends = []
    for name, meta in tensors.items():
        offsets = meta.get("data_offsets")
        shape = meta.get("shape")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return {**out, "whole": False,
                    "why_not": f"{name} has no usable data_offsets"}
        if not isinstance(shape, list) or not shape:
            return {**out, "whole": False, "why_not": f"{name} has no shape"}
        ends.append(int(offsets[1]))
    expected = 8 + header_len + max(ends)
    return {
        **out,
        "whole": size == expected,
        "expected_bytes": expected,
        "num_tensors": len(tensors),
        "shapes": {k: tuple(v["shape"]) for k, v in sorted(tensors.items())},
        "why_not": None if size == expected else
            f"file is {size} bytes but the header accounts for {expected}"
            + (" (truncated)" if size < expected else " (unexpected trailing "
                                                      "bytes)"),
    }


def b6_reference(repo_root: Path, b6_id: str) -> dict[str, Any]:
    """B6's own adapter facts, read from its filed Stage-2 checkpoint."""
    adir = s2g.stage2_ckpt_root(repo_root) / b6_id / "adapters"
    adapter = adir / ADAPTER_FILE
    summary_path = s2g.stage2_ckpt_root(repo_root) / b6_id / SUMMARY_FILE
    if not adapter.exists() or not summary_path.exists():
        raise SystemExit(
            f"REFUSING: B6's Stage-2 adapter or summary is missing "
            f"({adapter}, {summary_path}). The completeness gate compares each "
            f"B7 row against the incumbent it claims to extend; without it the "
            f"gate would only be able to check that a file exists.")
    layout = safetensors_layout(adapter)
    if not layout["whole"]:
        raise SystemExit(
            f"REFUSING: B6's own adapter is not whole ({layout['why_not']}); "
            f"it cannot serve as the reference topology.")
    summary = json.loads(summary_path.read_text())
    return {
        "candidate_id": b6_id,
        "adapter_sha256": sha256_file(adapter),
        "shapes": layout["shapes"],
        "num_optimizer_steps": summary["num_optimizer_steps"],
        "groups": summary["groups"],
        "recipe": summary["recipe"],
    }


def check_row(repo_root: Path, spec: s3g.Stage3CandidateSpec,
              ref: dict[str, Any], sidecar: dict[str, Any],
              mf_adapters: Path) -> dict[str, Any]:
    """One B7 row's completeness checks, as name -> (holds, detail)."""
    cid = spec.candidate_id
    root = s3g.stage3_ckpt_root(repo_root) / cid
    adir = root / "adapters"
    adapter = adir / ADAPTER_FILE
    checks: dict[str, Any] = {}

    def add(name: str, holds: bool, detail: Any = None) -> None:
        checks[name] = {"holds": bool(holds), "detail": detail}

    for label, path in (("adapters_dir_exists", adir),
                        ("adapter_file_exists", adapter),
                        ("config_file_exists", adir / CONFIG_FILE),
                        ("summary_file_exists", root / SUMMARY_FILE)):
        add(label, path.exists(), str(path.relative_to(repo_root)))
    if not adapter.exists():
        return {"candidate_id": cid, "checks": checks,
                "holds": False, "adapter_sha256": None}

    layout = safetensors_layout(adapter)
    add("safetensors_container_is_whole", layout["whole"],
        layout.get("why_not") or f"{layout['num_tensors']} tensors, "
                                 f"{layout['size_bytes']} bytes")
    add("lora_topology_matches_b6", layout.get("shapes") == ref["shapes"],
        None if layout.get("shapes") == ref["shapes"] else {
            "num_tensors_here": len(layout.get("shapes") or {}),
            "num_tensors_in_b6": len(ref["shapes"])})

    sha = sha256_file(adapter)
    if (adir / CONFIG_FILE).exists():
        cfg = json.loads((adir / CONFIG_FILE).read_text())
        recipe = ref["recipe"]
        add("target_modules_match_the_frozen_recipe",
            sorted(cfg.get("target_modules") or []) ==
            sorted(recipe.get("lora_target_modules") or []),
            sorted(cfg.get("target_modules") or []))
        add("lora_rank_and_alpha_match_the_frozen_recipe",
            cfg.get("r") == recipe.get("lora_r")
            and cfg.get("lora_alpha") == recipe.get("lora_alpha")
            and cfg.get("lora_dropout") == recipe.get("lora_dropout"),
            {"r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
             "lora_dropout": cfg.get("lora_dropout")})
    else:
        add("target_modules_match_the_frozen_recipe", False, "no config file")
        add("lora_rank_and_alpha_match_the_frozen_recipe", False,
            "no config file")

    if not (root / SUMMARY_FILE).exists():
        add("summary_records_the_frozen_recipe", False, "no summary file")
        return {"candidate_id": cid, "checks": checks,
                "holds": all(c["holds"] for c in checks.values()),
                "adapter_sha256": sha}

    summary = json.loads((root / SUMMARY_FILE).read_text())
    recipe = summary.get("recipe") or {}
    groups = {g["name"]: g for g in summary.get("groups") or []}
    anchor = summary.get("preservation_anchor") or {}
    epochs = summary.get("epochs") or []
    last = epochs[-1] if epochs else {}
    want_groups = {g.name: g for g in spec.groups}
    init_dir = Path(summary.get("init_adapter_dir") or "")

    add("summary_method_id_is_this_row", summary.get("method_id") == cid,
        summary.get("method_id"))
    add("recipe_is_the_frozen_one",
        recipe.get("seed") == 42
        and recipe.get("learning_rate") == s2g.STAGE2_LR
        and recipe.get("num_epochs") == s2g.STAGE2_EPOCHS
        and recipe.get("per_device_batch_size") == 1
        and recipe.get("gradient_accumulation_steps") == 8
        and recipe.get("max_length") == sidecar.get("max_length")
        and recipe.get("max_image_pixels") == sidecar.get("max_image_pixels"),
        {k: recipe.get(k) for k in ("seed", "learning_rate", "num_epochs",
                                    "per_device_batch_size",
                                    "gradient_accumulation_steps",
                                    "max_length", "max_image_pixels")})
    add("groups_are_b6s_three_at_this_rows_anchor_weight",
        sorted(groups) == sorted(want_groups)
        and all(groups[n]["mode"] == want_groups[n].mode
                and groups[n]["weight"] == want_groups[n].weight
                for n in want_groups if n in groups),
        {n: {"mode": g.get("mode"), "weight": g.get("weight"),
             "num_examples": g.get("num_examples")}
         for n, g in sorted(groups.items())})
    add("retain_group_carries_the_effective_anchor_weight",
        groups.get("retain", {}).get("weight") == spec.effective_anchor_weight,
        {"expected": spec.effective_anchor_weight,
         "found": groups.get("retain", {}).get("weight")})
    add("optimizer_steps_match_the_incumbent",
        summary.get("num_optimizer_steps") == ref["num_optimizer_steps"]
        and (summary.get("num_optimizer_steps") or 0) > 0,
        {"here": summary.get("num_optimizer_steps"),
         "b6": ref["num_optimizer_steps"]})
    add("training_actually_ran",
        (summary.get("train_seconds") or 0) > 0
        and len(epochs) == s2g.STAGE2_EPOCHS
        and isinstance(last.get("avg_loss_fine_target"), (int, float))
        and isinstance(last.get("avg_kl_retain"), (int, float)),
        {"train_seconds": summary.get("train_seconds"),
         "epoch_records": len(epochs),
         "final_fine_target_nll": last.get("avg_loss_fine_target"),
         "final_kl_retain": last.get("avg_kl_retain")})
    add("anchored_against_the_bound_cache",
        anchor.get("cache_tensor_sha256") == sidecar.get("tensor_sha256")
        and anchor.get("mf_adapter_sha256") == sidecar.get("mf_adapter_sha256")
        and anchor.get("num_preserved_examples") == sidecar.get("num_examples")
        and anchor.get("num_preserved_positions") ==
            sidecar.get("num_positions")
        and anchor.get("anchored_groups") == ["retain"]
        and anchor.get("anchor_weight") ==
            {"retain": spec.effective_anchor_weight},
        {"cache_tensor_sha256": anchor.get("cache_tensor_sha256"),
         "num_preserved_examples": anchor.get("num_preserved_examples"),
         "num_preserved_positions": anchor.get("num_preserved_positions"),
         "anchor_weight": anchor.get("anchor_weight")})
    add("started_from_the_canonical_mf_adapter",
        init_dir.resolve() == mf_adapters.resolve(), str(init_dir))

    return {"candidate_id": cid, "checks": checks,
            "holds": all(c["holds"] for c in checks.values())
            and layout["whole"],
            "adapter_sha256": sha}


def check_all(repo_root: Path) -> dict[str, Any]:
    calibration = json.loads((repo_root / s2g.CALIBRATION_REPORT).read_text())
    grid = s3g.stage3_grid(calibration["grid_rule"]["beta_star"])
    rows = s3g.trained_rows(grid)
    b6_id = s3g.control_row(grid).candidate_id
    ref = b6_reference(repo_root, b6_id)
    sidecar = json.loads(
        (s3g.reference_cache_dir(repo_root) / "sidecar.json").read_text())
    mf = s3g.mf_adapters(repo_root)

    per_row = [check_row(repo_root, spec, ref, sidecar, mf) for spec in rows]
    shas = [r["adapter_sha256"] for r in per_row if r["adapter_sha256"]]
    distinct = len(set(shas)) == len(shas) == len(rows)
    differs_from_b6 = all(s != ref["adapter_sha256"] for s in shas)
    problems: list[str] = []
    for row in per_row:
        for name, check in row["checks"].items():
            if not check["holds"]:
                problems.append(f"{row['candidate_id']}: {name} "
                                f"({check['detail']})")
    if not distinct:
        problems.append(
            f"the {len(rows)} B7 adapters do not have {len(rows)} distinct "
            f"sha256 hashes ({len(set(shas))} distinct over {len(shas)} "
            f"files): four rows that share bytes are four copies of one row")
    if shas and not differs_from_b6:
        problems.append(
            "a B7 adapter is byte-identical to B6's: its additional "
            "image-anchor weight changed nothing")
    return {
        "gate": "stage3_adapter_completeness",
        "rows_expected": len(rows),
        "rows_checked": len(per_row),
        "incumbent_reference": {
            "candidate_id": b6_id,
            "adapter_sha256": ref["adapter_sha256"],
            "num_tensors": len(ref["shapes"]),
            "num_optimizer_steps": ref["num_optimizer_steps"],
        },
        "per_row": per_row,
        "across_rows": {
            "all_adapter_hashes_distinct": distinct,
            "none_equals_the_incumbent_bytes": differs_from_b6,
            "adapter_sha256": {r["candidate_id"]: r["adapter_sha256"]
                               for r in per_row},
        },
        "problems": problems,
        "holds": not problems,
        "what_this_gate_is_not": (
            "not a protocol path and not part of the criterion: it cannot make "
            "a row eligible, move a floor or reorder anything. It decides only "
            "whether the chain may spend GPU time generating predictions for "
            "these adapters."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--json-out", default=None,
                    help="Also write the report here (operational evidence; "
                         "not a filed data/reports artifact)")
    args = ap.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root \
        else (_find_repo_root(Path.cwd()) or Path.cwd())
    out = check_all(repo_root)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    for row in out["per_row"]:
        failed = [n for n, c in row["checks"].items() if not c["holds"]]
        print(f"  {row['candidate_id']}: "
              f"{len(row['checks']) - len(failed)}/{len(row['checks'])} checks"
              + (f" — FAILED {failed}" if failed else " — complete"))
    print(f"  distinct adapter hashes: "
          f"{out['across_rows']['all_adapter_hashes_distinct']}; none equal "
          f"B6's: {out['across_rows']['none_equals_the_incumbent_bytes']}")
    if out["holds"]:
        print(f"OK — all {out['rows_checked']} B7 adapters are complete; "
              f"generation may start")
        return
    print(f"REFUSING: {len(out['problems'])} completeness problem(s):")
    for p in out["problems"]:
        print(f"  - {p}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
