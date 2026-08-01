# Continual-IA v2 development-only contract

Status: **unissued and nonpromoting**. No v2 plan, seed reservation, shard, or
artifact exists in the repository. V2 is a reproducible development diagnostic,
not held-out or preregistered evidence: its plan is self-issued and has no
trusted external pre-run chronology anchor. Passing every scientific gate can
only set `scientific_gates_passed=true`; `internally_accepted` is always false.

Seeds 60–89 remain unexecuted for this protocol. If an authorized operator runs
v2, each started seed becomes consumed development data. A future promotion
attempt requires a new schema and untouched seeds plus an externally anchored
pre-run plan or signed append-only reservation. V2 outputs cannot be upgraded
in place.

The historical v1 artifact remains an immutable scientific rejection under its
archived source: p=0.5 passed the reward and secondary controls but missed the
frozen action-changing intervention-rate gate. Current checkout drift in
`average_reward.py` invalidates the former current-source compatibility replay;
it does not rewrite the historical result or authorize a consumed-seed replay.

## Frozen scientific protocol

V2 freezes three selected differences from v1:

- treatment id `recommendation_p075`;
- exact recommendation acceptance probability `0.75`;
- exact seed schedule 60–89, with threshold seed start 60.

Every other scientific gate remains the v1 gate. The choice was informed by a
development-only probe on already-consumed seeds 30–59: action-changing rate
`0.1083056`, uplift `+0.27453`, and paired 95% lower bound `+0.25386`. These are
selection provenance only. They cannot promote, retune a gate, or replace a v2
shard.

## Lifecycle and invariants

The separate entrypoint is
`python -m alberta_framework.evaluation.continual_ia_v2_cli`. The public
`alberta-ia-evidence` command remains v1-only.

The lifecycle is self-issued plan → 30 reserved one-seed shards → merge:

1. The plan binds the exact configuration, thresholds, conditions, seeds,
   pairwise-distinct canonical output locators, prescribed subcommand argument
   vectors, a static transitive local-source closure, `pyproject.toml`,
   `uv.lock`, Python executable bytes, and aggregate regular-file content hashes
   for the explicit distribution set observed during a clean IA v2 import
   plus its installed required-dependency metadata closure: absl-py, aiofiles,
   Chex, cloudpickle, etils, humanize, JAX, jaxlib, jaxtyping, ml-dtypes,
   msgpack, NumPy, opt-einsum, orbax-checkpoint, prometheus-client, protobuf,
   psutil, Pygments, PyYAML, SciPy, simplejson, tensorstore, toolz,
   typing-extensions, uvloop, and wadler-lindig. A missing set member fails
   closed. The runtime record also binds module origins, interpreter flags and
   `sys.path`, JAX configuration/environment, backend, and device inventory.
   `prescribed_argv` is a recipe, not a claim about an observed process
   invocation.
2. Before either execution of a seed, the shard command rejects an occupied
   output and atomically publishes a persistent immutable
   `<shard>.reservation`. The reservation explicitly marks the seed consumed
   even if execution, replay, validation, or publication later fails. A second
   worker cannot pass the reservation boundary.
3. A shard runs its one bound seed twice and publishes only if the complete
   primitive traces match exactly. Each trace records reward, executed action,
   credited action, pre-update recommendation, pre-update partner proposal, and
   acceptance for every transition in all six arms.
4. Merge preflights the artifact destination, requires and validates each exact
   bound reservation and shard path without a glob, replays each seed once,
   independently recomputes every metric and gate, then rereads the plan,
   reservations, and shards before publishing. The CLI returns the
   already-computed merge decision and does not perform a second post-merge
   replay.
5. Public artifact loading compares the embedded plan and shards with their
   external immutable files, requires the 30 exact plan-bound reservations,
   replays all 30 seeds, rereads every external input and the artifact, and
   rechecks current source/runtime bindings.

The explicit Python distribution set is not a proof of every dependency an
alternate execution path might load. Runtime v3 records as unbound: system
shared libraries loaded by Python or extension modules, device drivers and
firmware, and dynamically loaded code outside distribution file manifests.
The runtime record is self-observed rather than image-attested.

Lifecycle files use canonical JSON, descriptor-anchored no-symlink traversal,
atomic new-path publication, directory syncing, mode `0444`, final single-link
verification, byte readback, and ancestor-directory identity checks. FIFO,
device, mutable, hard-linked, replaced, oversized, malformed, and noncanonical
inputs fail closed. Public plan, shard, and artifact validity cannot disable
current source/runtime checks.

Private underscore-prefixed fixture builders exist for adversarial tests. They
are not exported and cannot create promoting evidence; the schema itself is
permanently development-only and lacks external chronology.

## Authorized development execution only

Do not issue v2 as evidence. If a human explicitly authorizes the computational
cost for a development diagnostic, first confirm the checkout/runtime are
settled and every plan, artifact, shard, and reservation path is absent. An
illustrative invocation is:

```bash
.venv/bin/python -m alberta_framework.evaluation.continual_ia_v2_cli plan \
  --plan-out "$PWD/outputs/continual_ia_v2/plan.v2.json" \
  --shard-dir "$PWD/outputs/continual_ia_v2/shards" \
  --artifact-out "$PWD/outputs/continual_ia_v2/evidence.v2.json" \
  --attest-fresh-seeds-60-89
```

The attestation above is an operator assertion recorded by a self-issued
development plan; it is not externally verified chronology. The plan contains
30 absolute prescribed shard argument arrays. An illustrative first shard is:

```bash
.venv/bin/python -m alberta_framework.evaluation.continual_ia_v2_cli shard \
  --plan "$PWD/outputs/continual_ia_v2/plan.v2.json" \
  --seed 60 \
  --output "$PWD/outputs/continual_ia_v2/shards/seed-060.v2.json"
```

After all 30 immutable shards exist:

```bash
.venv/bin/python -m alberta_framework.evaluation.continual_ia_v2_cli merge \
  --plan "$PWD/outputs/continual_ia_v2/plan.v2.json" \
  --output "$PWD/outputs/continual_ia_v2/evidence.v2.json"
```

Verification is explicit and computationally expensive:

```bash
.venv/bin/python -m alberta_framework.evaluation.continual_ia_v2_cli verify-plan \
  --plan "$PWD/outputs/continual_ia_v2/plan.v2.json"
.venv/bin/python -m alberta_framework.evaluation.continual_ia_v2_cli verify-artifact \
  --artifact "$PWD/outputs/continual_ia_v2/evidence.v2.json"
```

If source, lockfile, runtime, backend, device state, or any lifecycle input
changes, stop. Never repair hashes, remove a seed reservation to retry, or reuse
an exposed seed.

## Interpretation

A valid artifact is a reproducible development diagnostic whose structure,
bindings, deterministic traces, metrics, and gates all recompute. It may record
that the scientific gates passed or failed. It is never an internally accepted
L2 result and cannot establish independent replication, general Step-12
intelligence amplification, Alberta Plan completion, robot benefit, or
state-of-the-art performance.
