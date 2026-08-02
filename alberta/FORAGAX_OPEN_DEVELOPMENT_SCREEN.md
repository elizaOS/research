# Current-Foragax open-development screen harness

`alberta-foragax-open-screen` executes two frozen FOV baseline screens in a strict CPU OCI
sandbox. It is an operational development tool, not an evidence or SOTA pipeline. Results cannot
be promoted: both screens use two open, already-consumed seeds and the pinned `dev2` image is
explicitly not a qualified production image.

The executable frozen inputs are:

- `outputs/forager/fov_baseline_screening_cpu_v3` — a CPU runtime overlay over the hash-bound
  eleven-candidate, 100,000-transition v1 baseline input.
- `outputs/forager/fov_stateful_baseline_screening_cpu_v3` — a CPU runtime overlay over the
  hash-bound eight-candidate, 102,400-transition v1 stateful input.

The v1 and CPU-v2 protocols remain immutable and can still be checked with `validate-protocol`, but
they are not executable through the current harness. The sealed v2 baseline screen is retained as
an immutable failed execution: all eleven candidates were ineligible because Numba attempted to
place `cache=True` artifacts beside read-only image source. Its artifacts are not resumed,
rewritten, or reused by v3. Execution and result validation fail closed unless the selected v3 schema freezes
the CPU backend, task-specific writable cache paths, sandbox, exact image, predecessor, and
in-image scorer.

## Safety and identity contract

Before any candidate starts, the harness checks the v3 protocol, its immutable v2 predecessor, and
its v1 base; snapshots the exact executing harness, preflight probe,
protocol/configuration/predecessor/scorer, and—when applicable—reference-scorer bytes into the new
output root;
resolves the user-supplied image argument to the exact frozen `sha256:` ID; records the host Python
and Docker executable and version identities; and runs a zero-transition preflight inside that
image. Only the snapshotted probe and scorer bytes are mounted. Their pre-read hashes and sizes are
bound into the plan, while the live originals and snapshot are re-read before and after preflight,
each candidate, aggregation, and validation. The preflight verifies the
frozen upstream source hashes, exact `continual-foragax==0.55.0` installation, PyExpUtils
index-to-seed mapping, the exact upstream result root and SQLite metadata rows for every candidate,
PPO rollout schedules, task, horizon, nonroot UID/GID, read-only root, CPU-only JAX devices, absent
NVIDIA devices, and network isolation. It also executes every entrypoint selected by the frozen
configurations with only `--help`. This traverses the real import path—including the Numba
`cache=True` decorators—before argparse exits and before experiment or environment construction.
The preflight rejects a nonzero exit, missing help marker, Numba cache-locator failure, or
Matplotlib fallback-cache warning.

Each candidate gets a separate sandbox with:

- `--network none`, a read-only root filesystem, user `65532:65532`, all capabilities dropped,
  `no-new-privileges`, and no host device mounts;
- the frozen protocol mounted read-only and one candidate-specific output bind mounted writable;
- `JAX_PLATFORM_NAME=cpu`, `JAX_PLATFORMS=cpu`, `NVIDIA_VISIBLE_DEVICES=void`, and an empty
  `CUDA_VISIBLE_DEVICES`;
- `NUMBA_CACHE_DIR=/tmp/alberta-numba-cache` and
  `MPLCONFIGDIR=/tmp/alberta-matplotlib-cache`, both created by the entrypoint import path inside
  the container-private writable `/tmp` tmpfs and verified as owned by user `65532:65532`;
- the exact image ID in the Docker command. The informational `dev2` tag is never used to launch a
  run.

The harness, probe, protocol, predecessor, base configurations, and scorers are re-read and
compared with the pre-preflight snapshot before and after each candidate and before completion.
Existing non-screen directories are rejected before a lock file is created. A screen output cannot
overlap a protocol directory or any pinned evidence tree.

DQN-family candidates execute upstream `src/continuing_main.py`. PPO-family candidates execute
upstream `src/rtu_ppo.py`; their frozen `rollout_steps * num_updates` value must equal the horizon.
The harness does not pass PPO's misleading `--max_steps` override.

## Execution

Validate an executable frozen input without Docker or reward execution:

```bash
.venv/bin/alberta-foragax-open-screen validate-protocol \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3
```

Run the first screen into a new output root:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3 \
  --output-dir /new/path/fov-baseline-cpu-v3-screen \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

Run the stateful/current screen separately:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v3 \
  --output-dir /new/path/fov-stateful-cpu-v3-screen \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

The module form is equivalent:

```bash
.venv/bin/python -m alberta_framework.benchmarks.foragax_open_screen --help
```

## Resume, validation, and aggregation

The output root is append-once. Every attempt preserves `stdout.log`, `stderr.log`, the complete
upstream payload, `attempt.json`, in-image scorer logs, and a canonical `run_manifest.json` with a
SHA-256 sidecar. A completed candidate is skipped on resume only when its protocol, config, source,
image, host/Docker runtime, sandbox, commands, logs, payload files, SQLite metadata, manifest, and
pinned-image rescoring remain exact. An interrupted `.incomplete` attempt is never guessed at or
silently restarted; use a new output root. A completed ineligible attempt is also immutable and is
not reward-informed-rerun.

Preflight stderr is diagnostic and may contain timestamps. The initial raw bytes are preserved and
hash-bound in the plan; a resume compares the current semantic preflight JSON while continuing to
verify the original diagnostic bytes. Thus timestamp-only diagnostic changes neither break a valid
resume nor rewrite history. The pinned image's installed CUDA discovery plugin can emit a nonfatal
`cuInit` diagnostic even with the frozen CPU environment; acceptance still requires the resulting
JAX backend and every enumerated device to be CPU, no NVIDIA device nodes, and disabled GPU
visibility.

For each candidate, the preflight supplies one exact nested result root. The payload must contain
only its ordered `data/<seed>.npz` files and sibling `results.db`; aliases, symlinks, hardlinks,
extra filesystem artifacts, duplicate or path-like ZIP members, encrypted members, CRC failures,
unsupported ZIP compression methods, and bounded-size or expanded-size violations are rejected.
Unsupported compression is normalized to the documented contract-error exit rather than escaping
as a traceback. Upstream auxiliary flat NumPy arrays
are preserved but never used; each archive must contain exactly one bounded `rewards.npy`. The
SQLite metadata schema, columns, typed rows, stored seeds, and configuration values must equal the
preflight contract. Safe collector tables may remain in the database but are never read for
eligibility or ranking.

Raw rewards are scored by the hash-bound scorer inside the exact development image, not by the host
NumPy runtime. The arithmetic is exactly `EMA_DECAY * ema + (1.0 - EMA_DECAY) * reward`, matching
the frozen stateful v1 scorer bit for bit. It samples indices `0, 100, ...` and averages the frozen
final 10% boundary. Collector summaries are never used. Validation launches read-only scorer
containers again and requires the semantic result to match the initially preserved scorer output.

`aggregate.json` is impossible to create until every frozen configuration has one completed
attempt. Only then does the harness compute the two-seed arithmetic means, rank descending, apply
the frozen path-ascending tie-break, and identify up to three candidates for a later
open-development run. Process or trace failures remain ineligible and rank after all eligible
candidates. The runner continues across candidate failures; it never peeks to stop the batch
early.

Revalidate a finished screen and recompute its canonical aggregate:

```bash
.venv/bin/alberta-foragax-open-screen validate-results \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3 \
  --output-dir /new/path/fov-baseline-cpu-v3-screen
```

`validate-results` therefore requires Docker and the exact pinned image. Use `--docker` for an
alternate Docker-compatible executable; its resolved executable hash and version must still equal
the persisted plan.

Exit code `0` means every candidate was eligible, `1` means the complete screen contains one or
more ineligible candidates, and `2` means a contract, integrity, input, image, or execution error.
None of these outcomes changes the protocols' nonpromoting evidence class.

The stateful screen also has a runtime-level fairness limitation: its hash-bound upstream
`rtu_ppo.py` path reuses one derived RNG for action sampling and environment stepping, unlike the
disjoint environment-key stream in `continuing_main`. Preflight source binding makes that behavior
auditable but does not remove the confound; paired, superiority, SOTA, official, and causal claims
remain forbidden.

## Completed executions in this repository

Three immutable executions of the v3 protocols exist under `outputs/forager/`, all produced
by the v4 harness/aggregate contract (`alberta.foragax_open_development_screen_aggregate.v4`)
on the two consumed open-development seeds 2,000,001 and 2,000,002:

- `fov_baseline_screening_cpu_v3_execution` — complete; all eleven candidates eligible.
  `DQN_LN-common-control` ranked first (two-seed mean FOV tail-EMA AUC `1.49084`); the three
  DQN common controls advanced for later open development.
- `fov_stateful_baseline_screening_cpu_v3_execution` — complete under its frozen contract,
  with the two PPO-family candidates ineligible (`raw_reward_validation`: unexpected or
  missing result directory). The attempt is immutable and was not resumed or rerun in place.
- `fov_stateful_baseline_screening_cpu_v3_corrected_v4_execution` — the corrected
  re-execution in a new output root; all eight candidates eligible.
  `PPO-RTU_LN_128_1_relu` ranked first (two-seed mean `1.78110`).

These rankings are open-development candidate-generation provenance for the frozen
matched-current candidate universe (see `FORAGER_BENCHMARK.md` and
`FORAGER_COMPARATOR_AUDIT.md`). They are nonpromoting, use consumed seeds and unmatched
candidate budgets, and support no comparison between the two screens.
