# FOV stateful-baseline CPU screen v2

This immutable open-development CPU protocol inherits the exact eight
candidates, two seeds, 102,400-step horizon, metric, and selection rule from
`../fov_stateful_baseline_screening_v1`. Its scorer runs inside the exact
development image and is tested for bitwise equivalence to the v1 hash-bound
`score_raw_rewards.py` arithmetic.

Validate without reward execution:

```bash
.venv/bin/alberta-foragax-open-screen validate-protocol \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v2
```

Run into a brand-new output root:

```bash
.venv/bin/alberta-foragax-open-screen run \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v2 \
  --output-dir /new/path/stateful-cpu-v2 \
  --image-id sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3
```

Revalidate the completed artifacts and rankings with read-only scorer
containers:

```bash
.venv/bin/alberta-foragax-open-screen validate-results \
  --protocol-dir outputs/forager/fov_stateful_baseline_screening_cpu_v2 \
  --output-dir /new/path/stateful-cpu-v2
```

This protocol is nonpromoting and cannot support superiority, SOTA, official,
or causal claims. In particular, the hash-bound upstream RTU-PPO path reuses
one derived RNG for action sampling and environment stepping, whereas
`continuing_main` uses a disjoint environment-key stream. The preflight binds
the exact `rtu_ppo.py` source hash, but it does not remove that paired-comparison
confound.
