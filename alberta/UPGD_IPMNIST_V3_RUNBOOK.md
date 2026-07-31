# UPGD IPMNIST v3 execution runbook

Status as of 2026-07-31: **no v3 plan has been issued, no v3 shard has been
executed, no v3 artifact exists, and no fresh v3 seed has been consumed.**
The completed v1 development diagnostic, its reconciled artifacts, receipts,
sealed runbooks, validator snapshot, and reconstructed source bundle remain
historical and unchanged. This document governs only a future v3 execution.

## Evidence boundary

The v3 schemas are:

- `alberta.upgd_ipmnist.plan.v3`;
- `alberta.upgd_ipmnist.seed_reservation.v3`;
- `alberta.upgd_ipmnist.partial.v3`; and
- `alberta.upgd_ipmnist.artifact.v3`.

They are permanently nonpromoting development records. The plan is
self-issued, seed freshness is operator-reserved rather than independently
attested, and the schema requires `scientific_promotion_allowed=false` and
`external_execution_attestation_present=false` at every stage. A scientific
claim would require a separately governed, externally attested protocol; a v3
artifact cannot be upgraded in place.

The plan always names the exact `upgd_w`/`adamw` learner pair, selected
hyperparameters, configuration, exact seed IDs, closed deviation codes,
dataset content identity and locators, prescribed command templates, runtime
manifest, and static transitive local-Python import closure plus
`pyproject.toml` and `uv.lock`. Each shard is exactly one learner and one seed.
Merge accepts only the complete planned learner-by-seed Cartesian product,
binds every shard by byte size and SHA-256, and exactly re-executes every
measurement before publication. Standalone shard validation likewise
re-executes that shard; artifact validation re-executes the full set again.
Paths are discovery locators, not shard or dataset identity.

Command arrays embedded in v3 records are explicitly **prescribed** argv,
recomputed from the bound payload. Caller-supplied argv is retained only as
an unattested diagnostic and is named accordingly. Plan and merge invocation
origins are derived by their CLI/direct-API entry points; shard origins are
derived by the benchmark-runner or supplied-result entry point. None is
misrepresented as an independently observed process attestation.

The one accepted OpenML archive is exactly 15,469,256 bytes with SHA-256
`fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78`.
The plan also binds the exact three OpenML metadata members for dataset 554,
requires all four cache files to be immutable single-link files, and performs
an offline load before publication. The loader addresses `data_id=554`
directly (there is no name/version resolution request), verifies every cache
member before entering sklearn, selects the pandas parser explicitly, and
replaces sklearn's OpenML `urlopen` entry point with a fail-closed denial for
the entire load. A missing or unexpected member therefore cannot fall back to
HTTP. After loading, v3 binds the materialized
60,000-row train arrays by dtype, shape, and SHA-256. An archive-only cache
cannot issue a plan. The cache, plans, shards, and artifacts must be regular
single-link files with no write permission bits; descriptor-anchored reads
reject symlinks and files that change during a read. Atomic publication also
removes the exact target link it created if any post-link durability,
identity, link-count, or byte-readback check fails.

Before a benchmark runner loads data or consumes compute, it atomically
publishes an immutable plan-scoped reservation for the exact `(learner,
seed)` identity. The reservation locator is independent of the requested
shard output path. Concurrent attempts, or later attempts using a different
output, therefore cannot execute the same planned identity twice. A
reservation deliberately survives runner or shard-publication failure: that
seed is consumed and must not be retried.

The active contract accepts only the selected paper configuration: 200 tasks,
5,000 updates per task, 784 inputs, hidden widths 300 and 150, ten classes,
the exact `upgd_w`/`adamw` hyperparameters, and exactly 20 fresh paired seeds.
The comparison includes the complete per-seed accuracy-delta vector, paired
mean, sample standard deviation, standard error, fixed two-sided 95% paired
t interval, and sign counts. Those preregistered descriptives add no post-hoc
acceptance gate.

## Launch prerequisites

Before issuing a plan, the operator must:

1. Freeze the working tree for the whole plan/shard/merge window. The import
   closure includes modules reached through package initializers, so any
   imported source-byte change after plan issuance correctly blocks workers
   and merge.
2. Use the project `.venv` and keep the recorded Python, JAX, jaxlib, NumPy,
   scikit-learn, pandas, platform, device details, JAX configuration, selected
   JAX/XLA environment, backend, and x64 setting unchanged. V3 hashes the
   resolved Python executable and an explicit execution distribution set:
   `absl-py`, `chex`, `jax`, `jaxlib`, `jaxtyping`, `joblib`, `ml-dtypes`,
   `narwhals`, `numpy`, `opt-einsum`, `pandas`, `python-dateutil`,
   `scikit-learn`, `scipy`, `six`, `threadpoolctl`, `toolz`, and
   `typing-extensions`. A same-version binary or package edit is drift. This
   is deliberately narrow: v3 does not claim a complete dynamic-import or
   native system-library closure, another reason it remains nonpromoting.
3. Resolve the exact cached OpenML MNIST archive and keep its bytes unchanged.
   The archive and the three dataset-554 metadata files must be regular
   mode-`0444`, single-link files at their exact OpenML cache paths inside
   `data_home`. A relocated cache is acceptable only when every pinned cache
   member and both materialized arrays match exactly.
4. Reserve exactly 20 fresh seed IDs in an external operator ledger before
   plan issuance. V3 rejects any smaller or larger schedule. IDs `0` through
   `9` are also rejected because the completed v1 diagnostic consumed them.
   The plan records the supplied reservation but cannot prove that those IDs
   were unused elsewhere.
5. Choose entirely new output paths. Plan, seed-reservation, shard, and
   artifact publication is atomic, mode `0444`, and refuses overwrite. Never
   target a sealed v1/v2 output, receipt, runbook, snapshot, or reconstructed
   bundle.

No benchmark or seed should run until all prerequisites are satisfied.

## Issue and validate the plan

Set operator-selected values for these shell variables; in particular, do not
copy a previously consumed seed list:

```bash
UPGD_V3_PLAN=/absolute/new-run/plan.v3.json
UPGD_V3_DATA_HOME=/absolute/openml-cache
UPGD_V3_ARCHIVE=/absolute/openml-cache/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz
UPGD_V3_FRESH_SEEDS=comma-separated-operator-reserved-uint32-ids
```

Issue exactly one plan before any worker starts:

```bash
.venv/bin/python -m alberta_framework.benchmarks.upgd_ipmnist \
  plan \
  --plan-out "$UPGD_V3_PLAN" \
  --seed-list "$UPGD_V3_FRESH_SEEDS" \
  --data-home "$UPGD_V3_DATA_HOME" \
  --data-archive "$UPGD_V3_ARCHIVE" \
  --n-tasks 200 \
  --task-length 5000 \
  --input-dim 784 \
  --hidden1 300 \
  --hidden2 150 \
  --n-classes 10

.venv/bin/python -m alberta_framework.evaluation.upgd_ipmnist_v3 \
  plan "$UPGD_V3_PLAN"
```

Archive the plan bytes and its SHA-256 outside the worker processes. Do not
regenerate a plan under the same logical run after inspecting results.

## Execute and validate one shard

Run this command once for every exact planned learner/seed pair, substituting
an operator-reserved seed and a new shard path. Each invocation runs only one
learner and one seed; there is no direct aggregate execution mode.

```bash
UPGD_V3_LEARNER=upgd_w
UPGD_V3_SEED=operator-reserved-seed-id
UPGD_V3_PARTIAL=/absolute/new-run/shards/upgd_w-seed-id.partial.v3.json

.venv/bin/python -m alberta_framework.benchmarks.upgd_ipmnist \
  shard \
  --plan "$UPGD_V3_PLAN" \
  --learner-id "$UPGD_V3_LEARNER" \
  --seed-id "$UPGD_V3_SEED" \
  --partial-out "$UPGD_V3_PARTIAL" \
  --data-home "$UPGD_V3_DATA_HOME" \
  --data-archive "$UPGD_V3_ARCHIVE"

.venv/bin/python -m alberta_framework.evaluation.upgd_ipmnist_v3 \
  partial "$UPGD_V3_PARTIAL" --plan "$UPGD_V3_PLAN"
```

Repeat separately for `adamw` and for every planned seed. A runner attempt
first seals a reservation under a plan-adjacent hidden reservation directory.
A failed or partial worker consumes that learner/seed identity permanently:
do **not** retry it at either the same or a new output path, and do not delete
its reservation. The development run is incomplete and must be reported as
such.
The validator command is a real one-shard re-execution, not a structural
check, so budget approximately the shard's original compute again.

## Merge and validate exact coverage

Only after every planned pair has one valid shard, pass all shard locators to
one merge. The validator keys shards by their embedded learner/seed identity
and content hash, not by filename.

Merge performs one complete 40-shard replay before it may publish. The
artifact validator performs another complete replay. Thus the commands below
intentionally spend two additional full-run equivalents beyond initial shard
execution (and per-shard validation, if run, adds one more). A replay mismatch,
exception, or late source/runtime/data drift fails closed and leaves the
requested new output path unpublished.

Plan, shard, merge, and artifact validation rebuild current source, runtime,
complete-cache, and materialized-data bindings. They repeat those checks after
their final plan/shard/artifact rereads, so a long replay cannot turn an early
binding check into a stale acceptance.

```bash
UPGD_V3_ARTIFACT=/absolute/new-run/artifact.v3.json

.venv/bin/python -m alberta_framework.benchmarks.upgd_ipmnist \
  merge \
  --plan "$UPGD_V3_PLAN" \
  --partials /absolute/new-run/shards/*.partial.v3.json \
  --output "$UPGD_V3_ARTIFACT" \
  --data-home "$UPGD_V3_DATA_HOME" \
  --data-archive "$UPGD_V3_ARCHIVE"

.venv/bin/python -m alberta_framework.evaluation.upgd_ipmnist_v3 \
  artifact "$UPGD_V3_ARTIFACT" \
  --plan "$UPGD_V3_PLAN" \
  --partials /absolute/new-run/shards/*.partial.v3.json \
  --data-home "$UPGD_V3_DATA_HOME" \
  --data-archive "$UPGD_V3_ARCHIVE"
```

Merge fails on duplicate, missing, or extra identities; learner/config/seed or
plan disagreement; noncanonical or nonfinite JSON; nested schema additions;
source/data/runtime drift; shard byte-size or digest mismatch; or recomputed
measurement, summary, or comparison disagreement. A structurally plausible
curve is not valid evidence without exact replay, and a fully valid result
still remains a nonpromoting development artifact.

Worker duration is retained only inside each shard as a self-reported
diagnostic cross-checked against its Unix interval. It is excluded from
learner summaries, comparisons, and scientific claims because computational
replay does not reproduce wall-clock timing.
