"""MF-preservation anchor (Iteration 12 Stage 2).

Stage 1 swept the strength of labelled replay and Stage 1b replicated the two
near-misses at four seeds; Stage 1c then read the image route the frozen floor
had never looked at.  All three say the same thing: rehearsing the fit half's
completions does not hold the probe half's retention up.  The committed
training summaries say why.  Gradient ascent on ``fine_target`` drives that
group's NLL from ~0.12 to between 1.13 and 12.40 -- it is an unbounded term --
while the ``retain`` group's own NLL *rises* from ~0.03 to 0.49-0.98 despite
being minimised at weights up to 4.0.  A supervised term is losing ground
against the ascent term on the parameters they share.

So Stage 2 stops asking the model to REPRODUCE a completion and asks it to
STAY WHERE MF WAS.  For each preserved prompt the anchor penalises

    KL( p_MF(.|x) || p_theta(.|x) )

summed over the supervised positions of that prompt, at every step.  Forward KL
(from the frozen reference) is the mass-covering direction: it charges the
current model for putting probability nowhere near zero wherever MF had mass,
which is what "preserve the function" means, rather than for merely picking one
mode.  At theta = MF the term is exactly zero with exactly zero gradient, so
the anchor is a trust region around the initialisation and not an extra
supervised objective competing with the others.

WHY THE REFERENCE IS CACHED AND NOT RESIDENT
--------------------------------------------
MF never changes during Stage 2, so ``p_MF`` on a fixed preservation set is a
constant, not something to recompute.  Caching it means no second model is
resident: the recipe already needs ~22 GiB for one bf16 copy of Qwen3.5-9B on a
shared box, and a frozen twin would not fit beside a co-tenant.  The cache is
EXACT rather than top-k truncated -- as built, 183 examples carry 1,489
supervised positions between them (8.14 each), so all 1,489 rows of 248,320
log-probabilities cost 1.479 GB in fp32, which is cheaper than justifying an
approximation.  Those four figures are read back from the committed sidecar
rather than estimated: the estimate this paragraph originally carried (1,434
rows of 248,077, 1.42 GB) was written before the cache existed and was wrong on
every one of them.

The cache is a protocol artifact: its sidecar records the sha256 of the MF
adapter it came from, the base-model revision and the encoding contract, so
"anchored against MF" is checkable rather than described.

WHAT IS PRESERVED, AND WHAT IS DELIBERATELY NOT
-----------------------------------------------
The preservation set is the ``retain`` group of ``unlearning_iter12/`` -- the
FIT half of the retained entities, 183 associations.  It is the same knowledge
replay rehearsed, so Stage 2 differs from Stage 1 in the OBJECTIVE applied to
that knowledge and in nothing else; a difference between the two studies is
therefore attributable to the mechanism rather than to a different preservation
set.

The target associations are absent on purpose.  Anchoring MF's distribution on
``fine_target`` would oppose the suppression term directly and on the very
prompts the method exists to change, and anchoring ``target_level`` would
oppose the rewrite.  The anchor preserves retained knowledge and leaves the
transformation free.

The probe half is never in this set, at any weight: a preservation term fitted
on probe-half prompts would rehearse the quantity the floor measures, which is
the leak the fit/probe split exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from granunlearn.logging_utils import setup_logger

log = setup_logger("preservation_anchor")

#: Bumped if the cache layout or the encoding contract it records changes.  A
#: cache written by another version is refused rather than reinterpreted.
CACHE_VERSION = 1

#: The objective mode this module adds.  ``unlearning_trainer`` knows ``sft``
#: and ``gd``; ``anchor`` replaces the group's NLL with the KL term below.
ANCHOR_MODE = "anchor"

#: Which knowledge group is preserved.  Named here so a caller cannot preserve
#: a target group by accident.
PRESERVE_GROUP = "retain"

#: Encoding contract the cache is valid under.  These are the recipe's own
#: values; a cache built at a different truncation length has different
#: supervised positions and must not be reused silently.
CACHE_MAX_LENGTH = 1536
CACHE_MAX_IMAGE_PIXELS = 384 * 384

CACHE_DIRNAME = "mf_reference_logprobs"
CACHE_TENSOR = "logprobs.pt"
CACHE_SIDECAR = "sidecar.json"

#: The NLL recomputed from extracted log-probabilities must equal the loss the
#: model itself returned.  This is the guard that makes the logits/labels shift
#: checkable: an off-by-one there silently compares the reference distribution
#: at one position against the current one at its neighbour, which would still
#: produce a plausible-looking small KL.
NLL_AGREEMENT_TOLERANCE = 2e-3


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supervised_positions(labels) -> list[tuple[int, int]]:
    """``(logit_index, target_token_id)`` for every supervised position.

    A causal LM predicts ``labels[i]`` from ``logits[i - 1]``, which is the
    shift HF applies internally (``shift_logits = logits[..., :-1, :]``,
    ``shift_labels = labels[..., 1:]``).  Returning the LOGIT index rather than
    the label index is the whole point: a caller that indexed logits at ``i``
    would score the distribution that predicts the NEXT token, one position
    late, and every KL in the study would be measured on the wrong row.
    """
    import torch

    flat = labels.reshape(-1)
    out: list[tuple[int, int]] = []
    for i in torch.nonzero(flat != -100).reshape(-1).tolist():
        if i == 0:
            #: A label at position 0 has no preceding logit, so it cannot be
            #: supervised by this model at all.  The reference recipe masks the
            #: whole prompt, so this never happens; raising keeps it that way
            #: instead of letting a malformed encoding shift every row by one.
            raise ValueError(
                "labels[0] is supervised but no logit predicts it -- the "
                "prompt mask is missing, so position alignment is undefined")
        out.append((i - 1, int(flat[i])))
    return out


def supervised_logprobs(logits, labels):
    """Full-vocabulary log-probabilities at the supervised positions.

    Returns ``(logp, targets)`` with ``logp`` fp32 of shape ``[n, vocab]`` and
    ``targets`` int64 of shape ``[n]``.  Computed in fp32 from whatever dtype
    the model produced, matching what HF's own loss does before cross-entropy.
    """
    import torch

    positions = supervised_positions(labels)
    if not positions:
        raise ValueError("no supervised positions: nothing to anchor on")
    rows = torch.as_tensor([p[0] for p in positions], dtype=torch.long,
                           device=logits.device)
    targets = torch.as_tensor([p[1] for p in positions], dtype=torch.long,
                              device=logits.device)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    logp = torch.log_softmax(flat_logits[rows].float(), dim=-1)
    return logp, targets


def mean_nll(logp, targets) -> float:
    """Mean negative log-likelihood of ``targets`` under ``logp``."""

    picked = logp.gather(1, targets.reshape(-1, 1)).reshape(-1)
    return float(-picked.mean().item()) if picked.numel() else float("nan")


def assert_nll_matches_model(logp, targets, model_loss: float,
                             where: str,
                             tolerance: float = NLL_AGREEMENT_TOLERANCE,
                             ) -> float:
    """Refuse to use an extraction that disagrees with the model's own loss.

    Returns the observed absolute difference so the caller can record it: the
    agreement is a measured quantity in the calibration report, not an assumed
    one.
    """
    got = mean_nll(logp, targets)
    diff = abs(got - float(model_loss))
    if diff > tolerance:
        raise AssertionError(
            f"{where}: the supervised positions extracted here give an NLL of "
            f"{got:.6f} but the model's own loss is {float(model_loss):.6f} "
            f"(difference {diff:.6f} > {tolerance:g}). The logits/labels shift "
            f"is wrong, so the anchor would compare the reference "
            f"distribution at one position against the current one at "
            f"another -- every KL in this study would be meaningless.")
    return diff


def forward_kl(logp_ref, logp_cur):
    """``KL(p_ref || p_cur)`` per position, exact over the full vocabulary.

    Both operands are log-probabilities.  ``p_ref`` comes from the cache, so it
    is a constant: the gradient flows only through ``logp_cur``, which is what
    makes the term an anchor rather than a second supervised objective.
    """
    p_ref = logp_ref.exp()
    return (p_ref * (logp_ref - logp_cur)).sum(dim=-1)


def anchor_loss(logp_ref, logp_cur) -> Any:
    """The scalar anchor loss: mean forward KL over supervised positions."""
    return forward_kl(logp_ref, logp_cur).mean()


@dataclass(frozen=True)
class CacheRow:
    """One preserved example's slice of the cache."""

    example_id: str
    association_id: str
    entity_id: str
    start: int
    stop: int
    targets: tuple[int, ...]


class ReferenceCache:
    """MF's preserved distributions, on disk, with the alignment guard.

    ``row`` is indexed by ``example_id`` and raises rather than returning a
    neighbouring row, because a silently substituted reference distribution
    would still train and would still produce a plausible KL.
    """

    def __init__(self, sidecar: dict[str, Any], logp, targets,
                 rows: dict[str, CacheRow]) -> None:
        self.sidecar = sidecar
        self.logp = logp
        self.targets = targets
        self.rows = rows

    # -- construction ------------------------------------------------------
    @classmethod
    def from_parts(cls, sidecar: dict[str, Any], logp, targets,
                   row_specs: Sequence[dict[str, Any]]) -> ReferenceCache:
        rows: dict[str, CacheRow] = {}
        for spec in row_specs:
            start, stop = int(spec["start"]), int(spec["stop"])
            rows[spec["example_id"]] = CacheRow(
                example_id=spec["example_id"],
                association_id=spec["association_id"],
                entity_id=spec["entity_id"],
                start=start, stop=stop,
                targets=tuple(int(t) for t in spec["targets"]),
            )
        return cls(sidecar, logp, targets, rows)

    # -- io ----------------------------------------------------------------
    def save(self, directory: str | Path) -> dict[str, Any]:
        """Write the tensor and its sidecar.  Returns the sidecar."""
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensor_path = directory / CACHE_TENSOR
        torch.save({"logp": self.logp.cpu(), "targets": self.targets.cpu()},
                   tensor_path)
        sidecar = dict(self.sidecar)
        sidecar["tensor_sha256"] = sha256_file(tensor_path)
        sidecar["rows"] = [
            {"example_id": r.example_id,
             "association_id": r.association_id,
             "entity_id": r.entity_id,
             "start": r.start, "stop": r.stop,
             "targets": list(r.targets)}
            for r in sorted(self.rows.values(), key=lambda r: r.start)
        ]
        with open(directory / CACHE_SIDECAR, "w") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        log.info("MF reference log-probs -> %s (%d rows, %s)", directory,
                 sidecar["num_positions"], sidecar["tensor_sha256"][:16])
        return sidecar

    @classmethod
    def load(cls, directory: str | Path, verify: bool = True,
             device: str = "cpu") -> ReferenceCache:
        """Load a cache and verify it is the one its sidecar describes.

        ``verify`` re-hashes the tensor.  It is on by default because the cache
        is gitignored: it is rebuilt on a new box, and a stale or truncated
        file would otherwise be anchored against without complaint.
        """
        import torch

        directory = Path(directory)
        sidecar_path = directory / CACHE_SIDECAR
        tensor_path = directory / CACHE_TENSOR
        if not sidecar_path.exists() or not tensor_path.exists():
            raise FileNotFoundError(
                f"no MF reference cache in {directory}: expected "
                f"{CACHE_SIDECAR} and {CACHE_TENSOR}. Build it with "
                f"scripts/build_mf_reference_logprobs.py before training an "
                f"anchored candidate.")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("cache_version") != CACHE_VERSION:
            raise ValueError(
                f"{sidecar_path} is cache_version "
                f"{sidecar.get('cache_version')!r} but this module reads "
                f"{CACHE_VERSION}; a layout change must be rebuilt, not "
                f"reinterpreted")
        if verify:
            got = sha256_file(tensor_path)
            if got != sidecar.get("tensor_sha256"):
                raise ValueError(
                    f"{tensor_path} hashes to {got} but its sidecar records "
                    f"{sidecar.get('tensor_sha256')}; the cached reference "
                    f"distributions do not match the MF adapter they claim to "
                    f"come from")
        blob = torch.load(tensor_path, map_location=device, weights_only=True)
        logp, targets = blob["logp"], blob["targets"]
        if list(logp.shape) != [int(sidecar["num_positions"]),
                                int(sidecar["vocab_size"])]:
            raise ValueError(
                f"{tensor_path} holds {tuple(logp.shape)} but the sidecar "
                f"records {(sidecar['num_positions'], sidecar['vocab_size'])}")
        rows = [{"example_id": r["example_id"],
                 "association_id": r["association_id"],
                 "entity_id": r["entity_id"],
                 "start": r["start"], "stop": r["stop"],
                 "targets": r["targets"]} for r in sidecar["rows"]]
        return cls.from_parts(sidecar, logp, targets, rows)

    # -- access ------------------------------------------------------------
    def row(self, example_id: str) -> CacheRow:
        try:
            return self.rows[example_id]
        except KeyError:
            raise KeyError(
                f"{example_id!r} is not in the MF reference cache, so it has "
                f"no preserved distribution to anchor against. The cache "
                f"holds {len(self.rows)} examples; build it from the same "
                f"group file this candidate trains on.") from None

    def slice_for(self, example_id: str):
        """``(logp_ref, targets_ref)`` for one example."""
        r = self.row(example_id)
        return self.logp[r.start:r.stop], self.targets[r.start:r.stop]

    def assert_aligned(self, example_id: str, targets_now) -> int:
        """Refuse to anchor a row whose supervised tokens moved.

        The encoding is deterministic, so the target ids extracted now must be
        the ones the cache recorded.  If the processor, the truncation length
        or the group file changed, they will not be, and anchoring position
        ``k`` of one encoding against position ``k`` of another would be a
        comparison of two different distributions.
        """
        r = self.row(example_id)
        now = [int(t) for t in targets_now.reshape(-1).tolist()]
        if now != list(r.targets):
            raise AssertionError(
                f"{example_id}: the supervised tokens are {now} now but the "
                f"MF reference cache recorded {list(r.targets)}. The encoding "
                f"moved, so cached and current log-probabilities are no "
                f"longer row-aligned and the KL would compare different "
                f"positions.")
        return len(now)

    # -- provenance --------------------------------------------------------
    def assert_built_from(self, mf_adapter_dir: str | Path,
                          base_model_revision: str | None = None) -> None:
        """Check the cache came from the MF adapter a candidate starts from.

        An anchor built from one MF and applied to a run initialised at another
        would preserve the wrong function, and nothing downstream would notice:
        the KL would still be small at step 0 only if the two happened to
        agree, and would not be otherwise.
        """
        adapter_files = sorted(
            p for p in Path(mf_adapter_dir).iterdir() if p.is_file())
        if not adapter_files:
            raise FileNotFoundError(f"no adapter files in {mf_adapter_dir}")
        #: Through ``adapter_digest`` rather than a second loop: two copies of
        #: a hashing rule diverge silently, and this one decides whether the
        #: anchor preserves the model the candidate actually starts from.
        got = adapter_digest(mf_adapter_dir)
        want = self.sidecar.get("mf_adapter_sha256")
        if got != want:
            raise AssertionError(
                f"the reference cache was built from an MF adapter hashing to "
                f"{want} but {mf_adapter_dir} hashes to {got}. Anchoring "
                f"against a different MF preserves a different function.")
        if base_model_revision is not None:
            recorded = self.sidecar.get("base_model_revision")
            if recorded != base_model_revision:
                raise AssertionError(
                    f"the reference cache records base_model_revision "
                    f"{recorded!r} but this run uses {base_model_revision!r}")


def adapter_digest(adapter_dir: str | Path) -> str:
    """Order-independent sha256 over an adapter directory's files.

    Filenames are mixed into the digest so two directories holding the same
    bytes under different names do not compare equal: the name is part of what
    ``PeftModel.from_pretrained`` reads.
    """
    files = sorted(p for p in Path(adapter_dir).iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"no adapter files in {adapter_dir}")
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.name.encode("utf-8"))
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    return digest.hexdigest()
