# Slowly-Changing Regression Development v2

This lane is a publication-shaped development extension, not an exact
replication. The selected ordinary-backprop arm uses ReLU Kaiming-uniform
initialization and the true mean-squared-error gradient. The Alberta CBP and
UPGD arms are local extensions, not publication-source comparators.

Primary references: Dohare et al., [Loss of plasticity in deep continual
learning](https://www.nature.com/articles/s41586-024-07711-7), and the
authors' [`loss-of-plasticity` v1.1 source
snapshot](https://github.com/shibhansh/loss-of-plasticity/tree/d626b017e403d94335f1c64f9d19f3d6a96af962/lop/slowly_changing_regression).

## Current status (2026-07-31)

No v2 pre-run plan has been issued and no full shard worker has been launched.
This file is a launch template only. Do not start a shard until the complete
bound-source set has had a verified quiet window and a new immutable plan has
been issued. The only repository file currently expected under
`outputs/slowly_changing_regression/` is this runbook.

The v2 schema is permanently nonpromoting. Its pre-run plan is self-issued by
the development runner, not issued or authenticated by an independent verifier, and
`scientific_promotion_allowed` is therefore always `false`. An artifact that
passes exact replay supports descriptive reporting only: no inferential, causal,
speed, SOTA, protocol-exactness, or Alberta Plan claim.

All plan, reservation, shard, and artifact wall-clock fields are self-reported
diagnostics. Future timestamps are rejected, but this check is not an external
chronology attestation and cannot prove that a plan preceded execution. The
closed execution envelope says this directly; it must not be described as a
held-out or independently preregistered protocol.

The source manifest binds the static transitive closure of local Python imports
from both benchmark entry modules, plus `pyproject.toml` and `uv.lock`. Runtime
identity binds the resolved Python executable bytes and aggregate regular-file
content hashes for the installed Chex, JAX, jaxlib, NumPy, and jaxtyping
distributions, as well as versions, JAX configuration, device identities, and
selected execution-environment fields. Runtime discovery errors fail closed.
These records are still self-recorded rather than image-attested. Plan, shard,
and merge records bind an internally derived prescribed command identity; that
identity is not evidence that a process actually observed the command. Raw
process arguments are retained only as explicitly self-reported provenance,
and direct API calls are labeled separately. Static import analysis cannot
attest dynamically constructed imports, which is another reason this lane
remains nonpromoting.

## Declared deviations

- The Nature Methods text reports 3 million examples and 40,000-example bins;
  the pinned v1.1 ReLU configuration and plotting script use 1 million examples
  and 20,000-example bins. This lane follows the article scale and records that
  source-snapshot discrepancy rather than silently treating both as identical.
- The publication reference uses PyTorch and Torch RNG; this implementation
  uses JAX, explicit integer seed identities, and JAX numeric/kernel semantics.
- The pinned source persists each generated sequence and reuses it across
  learner arms. This lane deterministically regenerates the same seed's online
  stream per arm and requires a shared environment identity at merge time.
- The Nature article says that one uniformly selected slow bit flips every
  period, which is what this lane implements. The pinned default `cfg/prob.json`
  omits `flip_one`, so its generator instead independently resamples the full
  slow-bit row each period. This article-versus-pinned-code discrepancy is
  recorded explicitly in every plan.
- The target-network bias is represented by an explicit constant hidden-input
  bit and has no target output bias, unlike the affine reference source.
- The publication study sweeps ordinary backprop over activations and step
  sizes. This plan selects the ReLU/SGD `0.01` arm and adds two Alberta-local
  methods.
- The pinned source data generator allocates one extra flip period, after which
  the learner consumes its configured prefix. This runner generates online,
  executes exactly `num_examples`, and uses a ceiling-sized segment schedule.
- JAX and PyTorch Kaiming draws are distribution-matched but not byte-matched
  for the same integer seed.

These deviations are machine-readable and closed in every v2 run plan. They
cannot be removed or reworded by a caller.

## Full development plan

Freeze the benchmark and learner sources and pass the focused test suite
before creating the plan. A plan binds the source bytes, run specification,
runtime, exact methods, and exact seed IDs. Existing paths are never
overwritten. Plans, reservations, shards, and artifacts are durably published
as mode `0444`, single-link files. Publication holds the temporary-file
descriptor through the final hard-link operation, verifies inode, size, mode,
link count, requested ancestor identity, and exact readback bytes, and removes
a partially linked target on failure. Readers reject symlinks, hard links,
writable files, locator replacement, and ancestor replacement. Newly created
ancestor directory entries are fsynced before publication continues.

```bash
.venv/bin/python -m pytest tests/test_slowly_changing_regression.py tests/test_slowly_changing_regression_v2.py -q -o addopts=""
.venv/bin/python -m ruff check alberta_framework/benchmarks/slowly_changing_regression.py alberta_framework/benchmarks/slowly_changing_regression_v2.py tests/test_slowly_changing_regression.py tests/test_slowly_changing_regression_v2.py
.venv/bin/python -m mypy alberta_framework/benchmarks/slowly_changing_regression.py alberta_framework/benchmarks/slowly_changing_regression_v2.py
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression plan --output outputs/slowly_changing_regression/development_v2/run_plan.v2.json --runs 100 --seed-start 0 --examples 3000000 --bin-size 40000 --flip-period 10000 --num-bits 20 --num-flipping-bits 15 --target-hidden-units 100 --hidden-units 5 --step-size 0.01 --methods bp,cbp,upgd
```

The plan covers exactly 100 seed IDs (`0` through `99`) and three methods, so
completion requires exactly 300 shards. Run one shard per command; the default
output is derived from the plan and is unique to its method/seed identity:

```bash
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression run-shard --plan outputs/slowly_changing_regression/development_v2/run_plan.v2.json --method publication_bp_relu_sgd --seed-id 0
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression run-shard --plan outputs/slowly_changing_regression/development_v2/run_plan.v2.json --method alberta_cbp_relu_local_extension --seed-id 0
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression run-shard --plan outputs/slowly_changing_regression/development_v2/run_plan.v2.json --method alberta_upgd_relu_local_extension --seed-id 0
```

Repeat those method commands for each planned seed ID. A shard refuses to run
if current source or runtime bytes differ from the plan, if its method/seed is
not planned, or if its destination already exists. Immediately before numerical
execution it durably writes `<shard>.reservation`. That immutable marker says
the development seed/method execution was started and is irrevocably consumed;
it remains after success or failure, so a crash cannot silently reuse the same
planned identity. An occupied reservation blocks execution before the learner
runs. A custom `--output` remains supported. Merge can discover any `*.json`
filename under `--shards-dir` (reservation files do not end in `.json`), or an
operator can pass every custom path explicitly with repeated `--shard` options.

Only after all 300 shards exist, merge them. Merge first performs an exact
deterministic replay of every shard and does not publish if any recorded float
differs. This adds another 900 million method-example updates:

```bash
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression merge --plan outputs/slowly_changing_regression/development_v2/run_plan.v2.json --shards-dir outputs/slowly_changing_regression/development_v2/shards --output outputs/slowly_changing_regression/development_v2/artifact.v2.json
```

Ordinary validation independently performs the same exact replay. Only this
path can return `valid: true`; it is intentionally expensive:

```bash
.venv/bin/python -m alberta_framework.benchmarks.slowly_changing_regression validate --artifact outputs/slowly_changing_regression/development_v2/artifact.v2.json
```

For a quick diagnostic, `validate --structural-only` checks the closed schemas,
coverage, hashes, paths, environments, commands, and descriptive reconstruction
without executing the learners. It deliberately returns a nonvalid result and
a nonzero CLI status. Structural success is not computational evidence and
cannot be promoted or reported as a validated result.

The merge requires the exact method × seed Cartesian product, reads each shard
strictly, checks shared environment identity across methods for every seed,
binds each shard's path/size/SHA-256 and prescribed command identity, exactly
replays every shard, and reconstructs all descriptive results. Before
publication it exactly rereads the external plan and every shard after the long
replay. The artifact embeds the plan and also binds the immutable external plan
by canonical absolute locator, size, and SHA-256. Ordinary artifact validation
does a final exact reread of the artifact, that external plan, and every shard
after its own long replay; replacement at any point fails closed.
Duplicate keys, non-finite JSON values, missing/duplicate seeds, unknown fields,
unsafe paths, source drift, runtime drift, or result tampering fail closed.

Descriptive dispersion uses the population standard deviation (`ddof=0`). The
derived scale is therefore named
`bin_population_std_over_sqrt_seed_count`; it is not labeled a sample standard
error. No confidence interval or hypothesis test is implied.

The absent historical `replication.v1.json` is not input to this process and
must never be retrofitted into v2 provenance.
