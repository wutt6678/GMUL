# Superseded: Stage 2's first faithfulness control, and the run that explained it

Nothing here is a result. No candidate, no prediction, no retention number and
no score came out of either directory, which is exactly why the criterion could
be replaced without the replacement being a rule moved to fit an answer: there
was no answer to fit.

These two runs are kept rather than deleted because the freeze's
`faithfulness_control` section asserts a finding about them, and a disclosure
nobody can inspect is only an assertion.

## What was frozen, and what it did

`freeze_iter12_stage2.py` froze the control as: train the incumbent Stage-1 row
through `train_with_preservation` with every group in a plain `sft`/`gd` mode,
and compare the resulting adapter to the one Stage 1 FILED, by sha256 over the
whole directory. Refuse to train a single Stage-2 candidate if the digests
differ.

It ran for 1187.8 s and refused. From `control_incumbent_byte_reproduction/CONTROL.json`,
quoted by its own key names:

```
reproduced_byte_for_byte  false
epochs_match              false
digest                    d6b968164c0222dd0a62a26d714a10b1903b15c2c86566d4abf51319cdbdc3ad
filed_digest              846f585a8e909d0b583ebbdc2d198d91d18335cc8ef6e0cd80c2e42076230e2b
num_optimizer_steps       230
train_seconds             1187.8
```

with an identical recipe in all fifteen fields, an identical init adapter, the
same three groups in the same modes and weights (`fine_target`/`gd`/0.5,
`target_level`/`sft`/1.0, `retain`/`sft`/1.0), and epoch means differing from
epoch 1 onward at about 1e-3.

Note that `produced_adapter_dir` in that JSON still reads
`data/checkpoints/mllmu_iter12_stage2/_control/...`, which is where the run
wrote it. The directory was moved here afterwards and the JSON was left
byte-identical, because editing a refusal record to suit its new location would
defeat the point of keeping it.

Two explanations fit that, and they have opposite consequences: either the new
loop computes something different from the frozen one — a real defect, and no
Stage-2 row may be trained — or two GPU runs of the *same* code do not agree
byte for byte on this stack, in which case the criterion was unachievable and
its failure says nothing about the loops.

## What distinguished them

`frozen_loop_rerun/` is `unlearning_trainer.train_unlearning` — the hash-bound
loop Stage 1 used, unedited — re-run on the same spec, from the same init
adapter, with the same recipe. If the frozen loop reproduces the filed adapter
and the new one does not, the new loop is the defect. It does not:

| comparison | max per-tensor weight gap | mean |
| --- | --- | --- |
| frozen re-run vs **filed** | 1.726e-03 | 1.190e-03 |
| new loop vs **filed** | 1.812e-03 | 1.298e-03 |
| frozen re-run vs new loop | 1.800e-03 | 1.288e-03 |

256 tensors in each. The new loop is no further from the filed adapter than the
frozen loop is from a file the frozen loop itself wrote. The divergence is a
property of the stack, not of the new loop.

## The mechanism

`PeftConfig.target_modules` is a Python `set`. Python randomises the string
hash seed per process unless `PYTHONHASHSEED` is exported, so the order PEFT
serialises that set into `adapter_config.json` — and the order it injects the
LoRA modules behind it, which decides which module draws which dropout mask —
varies from run to run.

The three adapters involved, read from the committed `adapter_config.json`
files, hold the same seven modules in three different orders:

```
filed_stage1       ['o_proj', 'v_proj', 'down_proj', 'q_proj', 'up_proj', 'k_proj', 'gate_proj']
control_new_loop   ['down_proj', 'k_proj', 'v_proj', 'gate_proj', 'o_proj', 'q_proj', 'up_proj']
rerun_frozen_loop  ['v_proj', 'up_proj', 'q_proj', 'o_proj', 'k_proj', 'down_proj', 'gate_proj']
```

Four `PYTHONHASHSEED` values were measured to give four different orders over
the same seven modules; a pinned seed gives one order every time. The filed
order matched none of the seven seeds tried, so the process that wrote it is
not recoverable and no later run can match its bytes.

The consequence is not that the loops differ. It is that the retired criterion
tested the interpreter's hash seed instead of the two loops, and a gate that
cannot pass says nothing when it fails.

## What replaced it

`scripts/train_iter12_stage2.py --control` now trains the incumbent objective
three times in ONE process under ONE pinned `PYTHONHASHSEED`:

- **A** — `train_unlearning`, the frozen loop
- **A2** — `train_unlearning` again, to MEASURE the noise floor rather than assume it
- **B** — `train_with_preservation`, every group plain `sft`/`gd`, no cap and no anchor

The gate is that `gap(A, B)` does not exceed `gap(A, A2)`, with bitwise equality
demanded outright when the floor is exactly zero — which is what one process and
a pinned seed are supposed to buy. The filed adapter is still loaded, compared
and reported beside the gate; it is simply not what the gate reads.

## What is here

| path | note |
| --- | --- |
| `control_incumbent_byte_reproduction/` | the run that refused: `CONTROL.json`, `training_summary.json`, `adapters/adapter_config.json`, and its 131 MB of weights (ignored) |
| `frozen_loop_rerun/` | the decisive re-run: `RERUN_COMPARISON.json`, `training_summary.json`, `adapters/adapter_config.json`, and its weights (ignored) |
| `frozen_loop_rerun.log` | the re-run's console log, including all five epoch summaries beside the filed ones |

The weights and tokenizers stay ignored by size; what is committed is the small
set a reader needs to check every claim above, and in particular the two
`adapter_config.json` files, because that is where the defect actually lives.
