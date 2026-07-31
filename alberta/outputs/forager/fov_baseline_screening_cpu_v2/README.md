# Current-Foragax FOV baseline CPU screen v2

This is the immutable CPU execution contract for the hash-bound candidates,
seeds, horizon, metric, and selection rule in
`../fov_baseline_screening_v1`. It intentionally does not execute the v1
protocol's CUDA runtime declaration. Both the v2 protocol and its v1 base are
validated by exact SHA-256 before use, then snapshotted into a new output root.

Validate without reward execution:

```bash
.venv/bin/alberta-foragax-open-screen validate-protocol \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v2
```

Execute every frozen candidate into a brand-new output root:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v2 \
  --output-dir /new/path/baseline-cpu-v2 \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

Validate a completed output (this performs read-only scorer containers, never
agent/environment transitions):

```bash
.venv/bin/alberta-foragax-open-screen validate-results \
  --protocol-dir outputs/forager/fov_baseline_screening_cpu_v2 \
  --output-dir /new/path/baseline-cpu-v2
```

The result is open-development and nonpromoting. Never target an existing
protocol, repository, or evidence directory as the output root.
