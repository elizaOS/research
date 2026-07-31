# Current-Foragax FOV baseline CPU screen v3

This immutable open-development overlay inherits the exact eleven candidates,
two seeds, 100,000-step horizon, metric, and selection rule from
`../fov_baseline_screening_v1`. It hash-binds and supersedes the sealed CPU-v2
overlay without modifying or reusing that predecessor's failed execution.

V3 retains the read-only container root and adds only two task-specific paths
inside the existing private `/tmp` tmpfs:

- `NUMBA_CACHE_DIR=/tmp/alberta-numba-cache`
- `MPLCONFIGDIR=/tmp/alberta-matplotlib-cache`

Before any reward transition, the exact-image preflight executes the frozen
`src/continuing_main.py --help` import path. It requires argparse success,
rejects Numba locator and Matplotlib fallback-cache diagnostics, and verifies
that both cache directories are owned by and writable to user `65532:65532`.

Validate without Docker reward execution:

```bash
.venv/bin/alberta-foragax-open-screen validate-protocol \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3
```

Execute every frozen candidate into a brand-new output root:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3 \
  --output-dir /new/path/baseline-cpu-v3 \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

Validate completed artifacts with read-only scorer containers:

```bash
.venv/bin/alberta-foragax-open-screen validate-results \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v3 \
  --output-dir /new/path/baseline-cpu-v3
```

The result remains open-development and nonpromoting. Never target an existing
protocol, failed-screen, repository, or evidence directory as the output root.
