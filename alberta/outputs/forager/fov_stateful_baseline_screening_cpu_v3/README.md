# FOV stateful-baseline CPU screen v3

This immutable open-development overlay inherits the exact eight candidates,
two seeds, 102,400-step horizon, metric, and selection rule from
`../fov_stateful_baseline_screening_v1`. It hash-binds and supersedes the sealed
CPU-v2 overlay without modifying or reusing predecessor results.

V3 retains the read-only container root and adds only two task-specific paths
inside the existing private `/tmp` tmpfs:

- `NUMBA_CACHE_DIR=/tmp/alberta-numba-cache`
- `MPLCONFIGDIR=/tmp/alberta-matplotlib-cache`

Before any reward transition, the exact-image preflight executes every
entrypoint selected by the frozen configurations—`src/continuing_main.py` and
`src/rtu_ppo.py`—with only `--help`. It rejects Numba locator and Matplotlib
fallback-cache failures and verifies that both cache directories are owned by
and writable to user `65532:65532`. The v3 scorer is also checked for bitwise
equivalence to the v1 hash-bound `score_raw_rewards.py` arithmetic.

Validate without Docker reward execution:

```bash
.venv/bin/alberta-foragax-open-screen validate-protocol \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v3
```

Run into a brand-new output root:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v3 \
  --output-dir /new/path/stateful-cpu-v3 \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

Revalidate completed artifacts and rankings with read-only scorer containers:

```bash
.venv/bin/alberta-foragax-open-screen validate-results \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v3 \
  --output-dir /new/path/stateful-cpu-v3
```

This protocol cannot support superiority, SOTA, official, causal, or promoted
claims. The hash-bound upstream RTU-PPO path still reuses one derived RNG for
action sampling and environment stepping; preflight source binding makes that
paired-comparison confound auditable but does not remove it.
