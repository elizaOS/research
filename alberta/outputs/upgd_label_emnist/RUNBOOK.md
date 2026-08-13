# UPGD Label-permuted EMNIST — 3-seed replication diagnostic (development, nonpromoting)

Lane: `alberta_framework/benchmarks/upgd_label_emnist.py` (plan/shard/merge CLI).
Protocol: Elsayed & Mahmood (ICLR 2024) *Label-permuted EMNIST* — EMNIST
balanced-split train set (112,800 examples, 47 classes), labels permuted every
2,500 steps, 400 tasks = 1,000,000 online examples (batch size 1), 784-300-150-47
ReLU MLP, softmax CE, online accuracy of the pre-update prediction averaged per
task. Upstream reference: github.com/mohmdelsayed/upgd @ `b75e90a` (the commit
audited by `core/canonical_upgd.py`).

Selected statistics-run arms (from `experiments/statistics_output_permuted_emnist.py`,
`zip(learners, grids)` pairing — same convention audited for the IPMNIST lane):

- **upgd_w** (`FirstOrderGlobalUPGDLearner`, protecting): lr=0.01,
  beta_utility=0.9, sigma=0.001, weight_decay=0.0
- **adamw** (released decoupled-decay Adam): lr=1e-4, beta1=0.0, beta2=0.9999,
  eps=1e-8, weight_decay=0.1

Published (figure read-off, low confidence): UPGD-W **rises** toward ~0.73–0.75
over the 400 tasks; plasticity-only/plain-optimizer baselines sit near ~0.35.
Paper averages 20 seeds; this diagnostic runs 3 (documented deviation).

## Dataset

OpenML `EMNIST_Balanced` (data_id 41039), 131,600 rows. First 112,800 rows
verified to have exactly 2,400/class (tail 18,800 rows 400/class) — consistent
with a train-then-test layout matching the torchvision balanced train split;
row identity/pixel orientation with torchvision are NOT verified (both are
fixed input permutations, irrelevant to an MLP). Loader fails closed if the
class-count check fails. Cache: `outputs/upgd_label_emnist/openml_cache/`
(~380 MB total: ~26 MB ARFF download + 337 MB parsed float32 `.npy` cache).

Materialized-array digests (dtype+shape+bytes SHA-256), bound into the plan:

- x (112800×784 float32, scaled (x/255−0.5)/0.5):
  `a6e7cb5635f41a98d880d2ec16fcf9f786bf24b7e079562dd8bd31d0543b2301`
- y (112800 int32): `8bfacf7cfc0344ed6059128349106a553d9fb38316f665d2307d053cf5f25946`

## Execution model

One process per learner×seed shard (CPU JAX gains nothing from vmap over seeds
here), `OMP_NUM_THREADS=2`, pinned with `taskset` to two cores each (cores
0–11; three unrelated long-running OPMNIST processes own the remaining
headroom). Measured steady-state on the 24-core box: upgd_w ~6.5 s per
2,500-step task (~45 min/seed for 400 tasks), adamw ~1.9 s/task (~14 min/seed).

Plan (immutable, seeds 100,101,102):
`outputs/upgd_label_emnist/plan.v1.json`
plan_sha256 `be208dde39c673eb1c70f74dd11cc72c14523c5c8ec1656d4437478f0952717b`

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m alberta_framework.benchmarks.upgd_label_emnist plan \
  --plan-out outputs/upgd_label_emnist/plan.v1.json --seed-list 100,101,102

# per shard (learner in {upgd_w, adamw}, seed in {100,101,102}):
OMP_NUM_THREADS=2 taskset -c <c0>,<c1> .venv/bin/python -m \
  alberta_framework.benchmarks.upgd_label_emnist shard \
  --plan outputs/upgd_label_emnist/plan.v1.json \
  --learner-id <learner> --seed-id <seed> \
  --partial-out outputs/upgd_label_emnist/partials/<learner>_seed<seed>.json \
  --progress-every 20

# merge after all 6 shards exist:
.venv/bin/python -m alberta_framework.benchmarks.upgd_label_emnist merge \
  --plan outputs/upgd_label_emnist/plan.v1.json \
  --partials outputs/upgd_label_emnist/partials/*.json \
  --output outputs/upgd_label_emnist/results.v1.json
```

Shards are idempotent per output path (immutable atomic writes refuse
overwrite): after interruption, relaunch only missing shards. An interim
artifact over a shard subset uses `--allow-incomplete` and a *new* output name;
never overwrite a published artifact.

## v2 — EMA input-conditioning transfer arms (2026-08-02)

Plan (immutable, seeds 100,101,102 — deliberately the v1 seeds so per-seed
paired deltas against the v1 `upgd_w` arm are meaningful):
`outputs/upgd_label_emnist/plan.v2.json`
plan_sha256 `8fe978266c7638bf84e907d026df0bf1acef8c252c3ce7a51534d2f6606baa7d`

Arms (factories are the pinned IPMNIST screening-lane implementations,
`ipmnist_screening._make_upgd_ema_norm_learner` / `_make_sgd_ema_norm_learner`;
norm_decay=0.999, norm_epsilon=1e-8):

- **upgd_ema_norm** — published EMNIST UPGD-W (lr=0.01, beta_utility=0.9,
  sigma=0.001, wd=0.0) behind the EMA input normalizer.
- **upgd_ema_norm_sigma0** — same, noise_std=0.
- **sgd_ema_norm** — bare-conditioning control: plain SGD (lr=0.01, wd=0.0 —
  matched to the published EMNIST UPGD-W decay; the IPMNIST screening arm used
  0.01 to match that protocol) behind the same normalizer.

**Pre-registered prediction** (recorded in the immutable plan notes BEFORE any
v2 shard ran): L/P EMNIST inputs are STATIONARY (labels permute, pixels never
do), so conditioning should transfer only weakly — `upgd_ema_norm` within
~±0.02 of v1 `upgd_w` (0.6715), far below the +0.061 IPMNIST effect;
`upgd_ema_norm_sigma0` ≈ `upgd_ema_norm`; `sgd_ema_norm` collapses toward the
plain-optimizer band (~0.2–0.4) because the utility gate/consolidation does
the heavy lifting under label recurrence. A near-null conditioning effect
CONFIRMS the mechanism reading: conditioning fixes input non-stationarity,
not label non-stationarity.

Shards: `partials_v2/<learner>_seed<seed>.json` (9 total), same shard CLI as
v1 with `--plan outputs/upgd_label_emnist/plan.v2.json`. Merge writes
`results.v2.json` (v1 artifact untouched).

## Evidence policy

Permanently nonpromoting development replication diagnostic: self-recorded
execution envelope, no external attestation, no dataset-byte binding of the
ARFF, 3 seeds vs the published 20. Documented protocol deviations are recorded
in the plan (`plan.deviations`): OpenML dataset plumbing, seed-derived streams
(upstream unseeded), task-aligned metric blocks, float32 bias corrections. The
UPGD-W inner loop is the IPMNIST lane's parity-tested lean restatement of the
audited `CanonicalUPGD` official_experiment_global protecting profile.
