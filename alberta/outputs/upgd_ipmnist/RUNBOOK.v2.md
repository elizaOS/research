# UPGD Input-permuted MNIST — 10-seed replication diagnostic

This is the run book for a completed, permanently nonpromoting replication
diagnostic of the ICLR-2024 Input-permuted MNIST lane
(`alberta_framework/benchmarks/upgd_ipmnist.py`). The network, task shape, and
one-million-step horizon match the selected publication configuration. The
complete published protocol does not: this run used 10 rather than 20 seeds,
made documented stream/logging/numeric changes, and did not bind worker source,
the full import closure, commands, environment, or dataset bytes at execution
time.

This file records how the 10-seed x 1M-step run was executed and how its shards
were reconciled without overwriting the original finalizer artifact.

## Execution model

CPU-only JAX gained nothing from `vmap` over seeds here (small per-step ops),
so seeds ran as **one process per seed**, pinned to cores with `taskset`.
Each worker wrote a mergeable shard JSON; a final merge step built the
artifact. All commands were run from the repository root.

Measured on the 24-core box: UPGD-W ~27 s per 5,000-step task per seed
(~90 min per seed for 200 tasks); AdamW ~9 s per task (~30 min per seed).

## Worker command (per seed)

```bash
# UPGD-W, seed N pinned to two cores:
OMP_NUM_THREADS=2 taskset -c <c0>,<c1> .venv/bin/python -m \
  alberta_framework.benchmarks.upgd_ipmnist \
  --learners upgd_w --seed-list <N> \
  --partial-out outputs/upgd_ipmnist/partials/upgd_w_seed<N>.json \
  --progress-every 20 > /tmp/upgd_w_seed<N>.log 2>&1 &

# AdamW, seed N pinned to one core:
OMP_NUM_THREADS=1 taskset -c <c> .venv/bin/python -m \
  alberta_framework.benchmarks.upgd_ipmnist \
  --learners adamw --seed-list <N> \
  --partial-out outputs/upgd_ipmnist/partials/adamw_seed<N>.json \
  --progress-every 20 > /tmp/adamw_seed<N>.log 2>&1 &
```

Schedule used for the recorded run (2026-07-31):

- Phase 1: `upgd_w` seeds 0-9 on core pairs (0,1)...(18,19), plus `adamw`
  seeds 0-3 on cores 20-23.
- Phase 2 (after phase-1 exits free cores): `adamw` seeds 4-9, four cores
  each.

## Resume

Workers are idempotent per shard: if `partials/<learner>_seed<N>.json`
exists, that seed is done — do not rerun it. To resume after an
interruption, relaunch only the missing shards with the worker command
above (any free cores work; pinning is a throughput optimization, not a
correctness requirement).

Check progress:

```bash
ls outputs/upgd_ipmnist/partials/          # finished shards
grep "task .*/200" /tmp/upgd_w_seed3.log | tail -1   # per-worker progress
```

## Reconciliation (after all 20 shards exist)

This is the historical command that produced the canonical reconciliation:

```bash
.venv/bin/python -m alberta_framework.benchmarks.upgd_ipmnist \
  --merge-partials outputs/upgd_ipmnist/partials/*.json \
  --data-home outputs/upgd_ipmnist/reconstructed_openml_cache \
  --output outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json \
  --note "10 seeds (paper uses 20); one process per seed; see RUNBOOK.md"
```

The active CLI now emits the strict `alberta.upgd_ipmnist.*.v2` development
schemas and deliberately rejects legacy v1 shards. The byte-exact historical
runner used above is preserved under
`outputs/upgd_ipmnist/reconstructed_source.v1/`; the v1 validator recomputes
the completed artifact from the immutable shards and validates that frozen
post-hoc source bundle rather than mutable live source.

The first finalizer output, `results.v1.json`, is preserved byte-for-byte. Its
strict validation fails only because its note did not preserve the 10-vs-20
seed limitation. `results.reconciled_nonpromoting.v2.json` repairs that
metadata and uses a repository-local post-hoc cache copy; strict structural
validation passes, but it remains permanently nonpromoting. The versioned
addendum `nonpromoting_receipt.v2.json` binds both artifacts, all 20 shards, the
reconstructed source partitions and cache, exact summary values, limitations,
and the operational log archive while preserving `nonpromoting_receipt.v1.json`
byte-for-byte as its predecessor.

An interim artifact over any subset of shards uses the same command with fewer
shard paths and a new output name. Never overwrite any preserved artifact.

## Seed-count deviation

The paper averages 20 seeds. This completed run used 10 (seeds 0-9) to fit the
CPU-only budget; the standard error in the artifact is descriptive only.
Seeds 10-19 must not be appended to this unsealed run. A 20-seed result requires
a fresh source-bound execution schema, an untouched run, and a new artifact
whose worker source, complete import closure, commands, environment, RNG
contract, and dataset bytes are bound before execution.

## Completed descriptive results

Across the 10 matched seeds, UPGD-W average online accuracy was
`0.7791470803916454` (SE `0.000055690729820870456`) and AdamW was
`0.7190002817213534` (SE `0.0005943125024635892`). The paired UPGD-W minus
AdamW mean difference was `0.06014679867029188` (sample SD
`0.0018825070977402044`), positive for all 10 seeds. UPGD-W first/last-quarter
means were `0.7774191806316375`/`0.779775180220604`; AdamW's were
`0.7597851804494857`/`0.6918167824745177`.

These are development diagnostics, not inferential evidence. Relative to
approximate publication figure read-offs, UPGD-W differed by about `-0.000853`
and AdamW by about `+0.039000`; the latter is an explicit reproduction gap.
Neither the descriptive paired interval nor the 10/10 sign is admissible for
promotion or a SOTA claim.

## Source snapshot and post-launch edit

UPGD-W seeds 0-9 and AdamW seeds 0-3 were launched at
`2026-07-31 00:59:54 -0700` from
`alberta_framework/benchmarks/upgd_ipmnist.py` with SHA-256
`e201ec75cb545e22ac868d9e86160cf92b0a59e7a523b6d71bbe3249b21d3a50`.
At `01:00:56 -0700`, after those workers imported it, the file received one
merge-only repair: `--merge-partials` now derives the protocol configuration
from the shards and rejects mixed configurations instead of using the CLI
defaults. AdamW seeds 4-9 were launched around `01:56 -0700`, after that
repair, from SHA-256
`36d4b18200662a857849b5b1855263a32c549005b4d16e249d62c07696e33d05`.
The per-seed execution path is identical between the two versions. Reversing
the single merge block in the repaired source reconstructs the earlier hash
exactly, but the shards do not bind themselves to either source, the full
import closure, worker commands, environments, or data bytes. This post-hoc
reconstruction is operational provenance, not execution attestation, so the
resulting run remains a nonpromoting replication diagnostic.

The exact reconstructed numerical subset is preserved at
`outputs/upgd_ipmnist/reconstructed_source.v1/` with a hash-pinned manifest.
It exists to keep the historical diagnostic auditable while allowing the live
runner to correct its schema and wording. The bundle remains post-hoc, is not a
complete import closure, and does not strengthen the evidence level.

## MNIST source

Workers loaded `fetch_openml("mnist_784", version=1)` from an auto-detected
step2 cache; the shards did not bind those bytes at execution time. The
reconciled artifact instead points to
`outputs/upgd_ipmnist/reconstructed_openml_cache`, a post-hoc byte-identical
copy bound by the receipt. The first 60,000 rows are the canonical torchvision
train split. This reconstruction improves auditability but is not execution
attestation.
