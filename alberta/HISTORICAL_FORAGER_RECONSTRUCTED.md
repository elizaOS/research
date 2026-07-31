# Reconstructed historical NumPy Forager lane

This lane runs fresh algorithms against the strongest audited reconstruction
of the paper-era mutable NumPy environment. Its environment family ID is:

`historical_numpy_forager_d140_reconstructed`

It is a separate environment family from `current_foragax_0_55`. Results from
the two families must not be seed-paired or pooled.

## Provenance boundary

The paper agents declared this dependency without a revision:

`foragerenv @ git+https://github.com/steventango/forager`

There was no agents lockfile. Consequently every run records
`environment_resolution_attested=false`, even when its reconstructed source
passes the behavior preflight. That field means the reconstruction is not
proof of the environment bits installed for archived paper runs.

The canonical provenance in
`alberta_framework.benchmarks.historical_forager_provenance` binds:

| Source | Commit | Tree | `git archive` SHA-256 |
|---|---|---|---|
| `steventango/forager-agents` | `696b3a06fbd0dc72407556b039d219e704ec6992` | `4936577cba549a3ffb4dec69bff722360c52f8be` | `a66ee0f7dd565dd64f5959520587e7121e5195d2814ef83504d8fc2b341d4803` |
| `steventango/forager` | `d140bdb3c51c7b6747d0588078ca97a67b55a8e1` | `0eb78e64b34cce3222215ebee3b94de2d83d41ce` | `2b7caf0a83b741404a88dfbb427f34f92e822d25901b4c9a71667d6e24cf14dd` |

The audited reconstructed environment wheel is
`foragerenv-0.0.0-py3-none-any.whl`, SHA-256
`9fcf134767a73337d36d6dec9c25721da68fd1b9587ea4c4299a3cdc00fc2020`.
It is explicitly recorded as a reconstruction, not a historical-install
attestation. Relevant individual source hashes are also part of the canonical
provenance object.

The audited compatibility runtime is Python 3.12, NumPy 1.26.4, Numba 0.59.1,
and Pillow 10.3.0. This is not a historical-runtime attestation. Every fresh
run records its actual host versions and whether they match that compatibility
runtime; the runtime identity is part of strict pairing, so results from
different dependency stacks cannot be paired silently.

## Environment preparation

The runner never edits `sys.path`, imports a checkout, or names a temporary
source location. Prepare the d140 environment and the agents wrapper in a
dedicated image, then expose the wrapper class from a read-only source root.
Do not import it from `/tmp` or another mutable staging checkout.

Bind the class with the explicit behavior preflight:

```python
from pathlib import Path

from installed_historical_wrapper import ForagerTwoBiomeLarge
from alberta_framework.benchmarks.historical_forager import (
    verify_historical_environment_factory,
)

adapter = verify_historical_environment_factory(
    ForagerTwoBiomeLarge,
    trusted_source_root=Path("/opt/historical-forager-agents-ro"),
)
```

The source root and wrapper module must be read-only, non-symlinked, and
outside temporary storage. Every loaded `forager` package module must likewise
resolve to a read-only, non-temporary regular file. The wrapper hash and the
complete installed `forager/` source inventory are checked against the
canonical agents source and reconstructed wheel contents. The preflight then
constructs private sentinel runs and requires both:

- the exact seed-0, aperture-9, 256-transition observation/reward trace hash;
- the seed-1 stale-cache behavior where actions `[1, 1, 1, 2]` reach a visible
  deathcap that pays `+1` because collision dispatch still holds an Oyster.

The second check is deliberate. Fixing the d140 generation, object cache,
observation orientation, or respawn behavior creates a different environment
family and must use a different ID.

The wrapper hash, read-only location, and complete environment source
inventory are checked again immediately around each real run's environment
construction. This prevents a preflight token from surviving ordinary source
changes between verification and execution.

Tests and local harness work can instead call
`development_historical_environment_adapter(factory)`. Runs with that adapter
require `allow_unverified_development_adapter=True` and are permanently marked
`source_preflight_verified=false`; they are development diagnostics, not
verified reconstructed-environment runs.

## Algorithm seam and execution

`HistoricalUpdateKernel` has two callables:

```python
state, action = start_kernel(observation)
state, action = update_kernel(state, reward, observation)
```

They can be pure `jax.jit` functions. The environment remains mutable and on
the host; only the algorithm transition needs compilation. The runner does not
special-case Alberta, DQN, RTU, or another algorithm.

```python
from pathlib import Path

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerRunConfig,
    HistoricalUpdateKernel,
    run_historical_forager,
)

kernel = HistoricalUpdateKernel(
    name="alberta_historical",
    start_kernel=jitted_start,
    update_kernel=jitted_update,
    metadata={"config_sha256": algorithm_config_sha256},
)
execution = run_historical_forager(
    adapter,
    kernel,
    HistoricalForagerRunConfig(
        seed=0,
        steps=500_000,
        aperture_size=9,
        output_directory=Path("artifacts/historical/seed-0"),
    ),
)
```

The exact call order is one environment construction, one `start()`, one
kernel start, then one environment step followed by one kernel update for each
transition. The last transition still performs its kernel update, matching
RLGlue. There are no reset calls, observation flips, reward transformations,
or evaluator contexts.

The wrapper must return `(reward, observation, False, {})`. Nonempty info is a
contract error. In particular, the historical environment has no
`biome_regret`; the runner neither invents one nor exposes one to the kernel.

## Metric and artifact contract

Each successful output directory contains exactly:

- `rewards.npy`: chronological raw rewards as little-endian float64;
- `result.json`: canonical completion manifest written last.

Both files are finalized read-only; validation rejects writable copies or
hard-linked aliases as mutable/non-canonical evidence.

The compatible field-of-view statistic is the original Collector pipeline:

1. start the EMA at zero;
2. update `z = 0.999 * z + (1 - 0.999) * reward` for every transition;
3. retain transition indices `0, 100, 200, ...`;
4. take the NumPy float64 mean beginning at `int(0.9 * sample_count)`.

`validate_historical_forager_artifact()` checks the exact file inventory,
canonical JSON, family, provenance, negative attestation fields, raw sidecar
hash/dtype/shape, absence of biome regret, and a full recomputation of every
metric. Kernel metadata cannot override environment or pairing claims.

For seed-level paired comparisons, derive identities with
`historical_artifact_pairing_identity()` and call
`assert_historical_artifacts_pairable()`. Family, provenance, seed, aperture,
horizon, semantic contract, adapter verification mode, and runtime identity
must all match. Cross-family pairing fails before any metric is compared, and
an unverified development run cannot pair with a golden-verified run.

## Bounds and interruption behavior

Runs are explicitly bounded to at most 100,000,000 transitions. Reward storage
uses a fixed-size disk-backed array; metric memory is bounded by the sampled
last tenth rather than the full reward sequence.

The completion manifest is published only after the full reward sidecar is
flushed. An in-process exception removes known partial files. A process or host
failure can leave an incomplete directory, but it cannot leave a valid
completed artifact; validation requires the completion manifest and exact file
inventory.

Generic resume is intentionally unsupported. Correct resume would require
versioned codecs that atomically capture the mutable environment, next action,
algorithm state, Collector state, and raw-sidecar offset together. Until those
codecs exist, restart a failed seed in a new output directory rather than
claiming a partial checkpoint is equivalent.
