# Forager Matched v3 Development Runbook

Status: **component design and implementation only; campaign unissued, runtime-unqualified,
and no full-horizon benchmark executed**.

This document describes the next separately versioned Forager comparison lane. It is not a
preregistration receipt, an authority grant, or a performance claim. No v3 development,
calibration, or held-out seed exists until the corresponding strict artifact is built and
validated.

## Compatibility boundary

- Historical `matched_current_*_2c3b214c_v1` roots are immutable and source-incompatible.
- Candidate-universe v2 digest
  `6a9315cb996fe5698e4c1580d30da9b0524e9875ce085d1399bb975cc5b510a8`
  remains an offline compatibility surface. It will not be newly qualified or executed.
- v3 uses new modules, schemas, commands, source closures, and digest-named roots. A v3
  loader must reject every v1/v2 artifact, and existing v2 loaders must reject v3.
- The unexecuted v3 development-universe descriptor currently has SHA-256
  `a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750`.
  It fixes membership and provenance only; it does not bind executable configurations.
- The configuration plan has SHA-256
  `55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7`.
  It binds implemented components but keeps execution readiness and authority false.
- The pre-observation qualification-plan descriptor has SHA-256
  `508b2167854d39b8a99c180708d93fcda0b3ea9cad66a1097c318b74e0440f26`.
  It is a score-blind, content-only contract with no default production plan. A caller must
  separately provide the exact two source closures, one complete runtime identity, 28
  permanently consumed public qualification cases, 28 resource ceilings, and 28
  result-publisher bindings. It neither executes qualification nor grants authority.
- Adding v3 Python files changes the full Alberta source snapshot used by any later
  qualification. That expected source drift does not change historical artifacts or the v2
  candidate-universe digest.

## Exact target and metric

The proposed target retains the current matched task:

| Field | Value |
|---|---|
| Environment | `ForagaxTwoBiomeLarge-v1` |
| Observation | color, aperture 9 |
| Horizon | 499,712 ordered environment interactions |
| Primary scalar | exact sum of all ordered raw rewards |
| Raw reward set | `{-1, 0, 1, 30}` |
| Per-run score bounds | `[-499712, 14991360]` |
| Ordered paired-difference bounds | `[-15491072, 15491072]` |
| Difference range width | `30982144 = 62 * 499712` |

This replaces v2's tail-summary estimand. The strict scorer is implemented and source-bound
at SHA-256 `eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e`.
It verifies task identity, trace length, reward membership, finite values, and exact
accumulation before emitting the scalar. Its sole canonical NPZ encoding is 499,980 bytes;
the encoder is byte-identical to NumPy 2.5.1 `np.savez` for the frozen trace layout and is
the exact inverse of strict extraction. It cannot silently round an out-of-set reward or
reduce an incomplete trace.

## Stages and candidate flow

The stages are strictly disjoint:

1. `development_selection_v3` tunes configurations and selects family representatives. It
   is open development and permanently nonpromoting.
2. `development_calibration_v3` uses a second fresh seed set only for cost, variance, and
   fixed-sample planning. A later change to the panel, code, metric, task, image, or RNG
   contract discards this calibration.
3. The confirmatory panel, statistics implementation, practical margin, sample count,
   retry policy, resource ledger, and source/runtime closure are frozen.
4. A future-randomness receipt derives untouched held-out trial blocks.
5. Every frozen inferential candidate executes every held-out block. There is no held-out
   selection slot, subset, available-case analysis, or outcome-informed extension.

The development universe contains:

- all nine Alberta causal-map grid arms;
- four Alberta Horde variants;
- `alberta_rtu_h08_taylor`;
- the exact-task transformed DQN anchor;
- CReLU, ReDo, reward-trace, and L2-to-initialization DQN development choices;
- current-XFinal PT-DQN and DRQN transfers;
- generic isolated PPO and paper-scale isolated RTU-PPO transfers;
- reviewed Dopamine Full Rainbow and POBAX PPO-GRU derivatives;
- Random, Search-Nearest, and Search-Oracle as descriptive references only.

Development selects one causal-map arm, one Horde arm, and one DQN plasticity arm. Local
RTU is the single current Alberta RTU representative. The minimum confirmatory panel is
therefore 11 inferential arms:

1. selected Alberta causal-map representative;
2. selected Alberta Horde representative;
3. Alberta local RTU representative;
4. transformed exact-task DQN anchor;
5. selected DQN plasticity representative;
6. PT-DQN;
7. current-XFinal DRQN;
8. isolated generic PPO;
9. isolated paper-scale RTU-PPO;
10. adapted Full Rainbow;
11. adapted PPO-GRU.

Random, Search-Nearest, and Search-Oracle remain outside inference. Search policies are
privileged; Random is a nonlearning sanity floor. Shared agent/environment-key PPO or RTU
orientations may be added only as separately labeled descriptive source diagnostics.

The pure development scheduler is implemented at source SHA-256
`1df3cd7e080f1301e469822ebce5020243a5cca71b6e7eb815a934e09966ef5b`.
It accepts only a caller-carried, full-file-digest-bound open-development seed registry and
constructs the exact block-major Cartesian product over the 25 inferential candidates.
Each cell uses the configuration plan's exact no-newline canonical record digest; the
schedule's newline-terminated document encoding is not reused for that cross-component
identity. Numeric seed collisions remain distinct block/candidate records. Random,
Search-Nearest, and Search-Oracle are deliberately absent from this schedule and require a
separate descriptive schedule. The implementation issues no seeds, opens no result, runs
no cell, and grants no execution or promotion authority; no production registry or schedule
exists.

## Source and runtime pins

The source audit currently binds these primary identities:

| Source | Commit | Git tree | Audit archive SHA-256 | Size |
|---|---|---|---|---:|
| continual-foragax-agents | `9710f60fa30da5badc451ad7ce3ff296d5070830` | `a5ad878ac4be0567c43dfd9177471c4b5a910bfa` | `1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6` | 314,961,920 |
| Dopamine | `5873f5494ee0c2d7c016d0ab2ad530354fec59d0` | `578408662e298d00e4e855f13f67dc08bd784e7c` | `bea46f755c86725d7ca90c531a08aad86cab62201ac2b9224c82f66dfada7456` | 82,933,760 |
| POBAX | `a5e1d62d14e4efe783885b9d4f19cffa2a568eec` | `d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6` | `f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9` | 1,699,840 |

The Dopamine and POBAX archive values are audit-time built-in-tar identities. v3
qualification must reproduce them with its hardened Git boundary or issue new explicit
qualified archive identities before any protocol is frozen.

Relevant review-anchor SHA-256 values are:

- Dopamine: `LICENSE` `e47b2783...62323`, `full_rainbow_agent.py`
  `cc85222d...50595`, `configs/full_rainbow.gin` `f926614f...bc130`, and
  `networks.py` `fac81313...bb9`; the implemented core additionally binds Rainbow replay
  anchors plus `losses.py` `42c10699...cbd39` and `dqn_agent.py`
  `53a37912...19dca`.
- POBAX: `LICENSE` `c71d239d...ab4`, `ppo.py` `0c827250...e153`, `config.py`,
  the discrete actor/value/network anchors, `models/__init__.py`
  `c4434b0b...323a`, and package dependency manifests.

The exact full digests live in the v3 candidate-universe descriptor; abbreviations here are
for readability. These are review anchors, not a complete transitive dependency inventory;
both adapter descriptors and the configuration plan set `source_closure_bound` false. Both
adaptation sources are Apache-2.0. The resulting Forager candidates are reviewed derivatives,
never exact executions of those repositories.

The external two-seed transport is now implemented only as a deterministic four-file source
patch set. Its descriptor SHA-256 is
`66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad`; the derived
`continuing_main.py` and `rtu_ppo.py` entrypoint hashes are respectively
`ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551` and
`1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1`.
Both CLIs require exactly one index plus explicit uint31 environment and agent seeds. This
does not bind the complete dependency inventory: the continuing runner still uses
runtime-default JAX PRNG behavior and NumPy agent RNG consumers outside the patch set, and
both paths still require runtime trace qualification. A fresh checkpoint root is mandatory
for the continuing path because resume can restore historical RNG state.

The external source materializer is now implemented and independently audited. Its pinned
identity schema digest is
`5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a` and its
implementation-source SHA-256 is
`5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041`.
It accepts only the pinned direct-`.git` checkout, validates the complete regular-file Git
tree, applies the four frozen transforms, strips Git metadata, and atomically publishes a
manifested derived tree. Git inspection is hermetic and wall/output bounded. No production
checkout or manifest has been accepted, the archive bytes remain provenance rather than
independently verified input, and no manifest grants runtime or execution authority.

The 14 local worker-envelope configurations now have an implemented, content-checking,
non-authorizing builder. Its builder descriptor SHA-256 is
`1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b`.
This closes configuration construction only; the local execution source snapshot remains
unbound and every local candidate remains execution-unready.

The two local adapters now share one implemented, non-authorizing Foragax environment
bridge. Its descriptor SHA-256 is
`1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab`; its
implementation-source SHA-256 is
`5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362`.
It owns the direct `jax.random.key(environment_seed, impl="threefry2x32")` root, splits
once for reset, and then splits the carried environment key exactly once per transition.
Adapter cores own only their candidate-private agent chains. The bridge binds JAX and
JAXlib 0.11.0, the enabled partitionable Threefry behavior, x64-disabled mode, and exact
Foragax 0.55.0 install-tree bytes. It remains an unqualified, per-step host API: no real
Foragax parity qualification, backend qualification, full-horizon memory profile, or
compiled chunk kernel has been accepted. On 2026-08-02 an unpersisted, nonpromoting
engineering probe opened the exact installed CPU runtime, validated a `(9, 9, 3)` float32
observation, completed three transitions with exact reset-plus-step key accounting, and
replayed the same seed/actions with byte-exact observations and rewards. That tiny probe is
not a qualification receipt and does not change any false readiness flag.

The current OCI image is
`sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768`.
Its dependency-lock label exactly matches upstream `uv.lock` SHA-256
`46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd`
and includes JAX 0.9.0.1, Flax 0.12.3, Optax, Distrax, and Flashbax. Reuse is conditional on
a fresh v3 capability/import probe; a missing dependency requires a newly pinned image
before protocol freeze. In particular, the adapter bridge requires JAX/JAXlib 0.11.0, so
this image is not currently a qualified adapter runtime image.

The qualification-plan v1 contract deliberately requires one unified, networkless CPU OCI
runtime using JAX/JAXlib 0.11.0 for all 28 public qualification cases. The current JAX
0.9.0.1 image cannot satisfy that contract. A future production plan therefore requires a
newly pinned unified image whose external, local, and adapter candidate imports and exact
seed/trace behavior all pass the predeclared score-blind probes; separate family runtimes
would require a separately versioned qualification-plan contract.

## Adapter conformance

The implemented, unqualified Full Rainbow core retains and independently tests:

- three-step returns;
- prioritized replay, importance weights, and priority updates;
- C51 projection and frozen support;
- Double-Q action selection;
- noisy-network training/evaluation semantics;
- dueling value/advantage heads.

Dopamine's pinned Atari convolutional trunk is not shape-compatible with the 9-by-9
Forager aperture, and its default C51 support `[-10, 10]` is not a justified support for
unscaled continuing rewards in `[-1, 30]`. The adapter must therefore bind and test a
Forager-specific observation trunk plus an explicitly justified reward scaling and C51
support; it does not present those necessary changes as an exact upstream execution. Its
configuration SHA-256 is
`835f02bdcf6844b7cd8c5e9fe33230a2a94f3a9c288c812cbfddf473c28b7e3f` and its
non-authorizing adapter descriptor SHA-256 is
`5436200c47e1b003b0371c30606b52163b4c42427fa84e2fe2f4b2b2273ccae2`; its
implementation-source SHA-256 is
`7f75a0862ddc21160cea9c0a9faca221a0d757985fc90e5ef02b4673e3c14f5a`.
Its full-horizon in-memory runner is also implemented. The runner descriptor SHA-256 is
`546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc`
and its implementation-source SHA-256 is
`5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c`.
The exact schedule performs 499,712 interactions, first replay insertion at transition 4,
119,928 optimizer updates beginning at transition 20,004, and 60 target synchronizations.
Production result emission requires a process-local completion capability and retains the
complete ordered signed-int8 raw reward trace. No full-horizon/resource qualification or
filesystem writer has run.

The implemented, unqualified PPO-GRU core retains and independently tests:

- categorical discrete actions;
- GRU carry and episode-boundary reset;
- sequence-preserving minibatches;
- GAE, clipped policy and value losses, entropy, and gradient clipping;
- logically separate agent and environment key-consumption chains;
- environment-interaction accounting rather than vectorized optimizer-step accounting.

The pinned POBAX loop minibatches recurrent sequences across parallel environment lanes and
uses one mutable key chain for policy and environment draws. The matched adapter instead has
one paired environment trajectory per block, so it must preserve sequence order through
time-segment initial carries while consuming the block's explicit environment seed and the
candidate-private agent seed. Parallel-environment reward aggregation is not the v3
estimand.

The adapted PPO-GRU folds POBAX's four parallel 128-step recurrent lanes into one 512-step
public trajectory, split into four contiguous 128-step segments with their actual incoming
GRU carries. At four epochs this is exactly 976 rollouts and 15,616 optimizer updates over
499,712 environment interactions. It does not aggregate parallel-environment rewards. Both
seed roots are exact uint31 values and both chains pin JAX `threefry2x32`; equal numeric roots
therefore produce correlated key streams, so no statistical-independence claim is made. The
configuration SHA-256 is
`07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70` and the
non-authorizing source descriptor SHA-256 is
`64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca`; its
implementation-source SHA-256 is
`58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e`.
Its full-horizon in-memory runner descriptor SHA-256 is
`e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2`
and its implementation-source SHA-256 is
`afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47`.
It closes the exact 976-by-512 trajectory, four contiguous recurrent segments across four
epochs, 15,616 optimizer transactions, complete behavior/carry replay, and ordered raw
reward retention. Its persisted receipt parser is deliberately structural; only the live
process capability distinguishes a completed production outcome.

Both runners feed one strict, in-memory reward conversion descriptor with SHA-256
`1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5`;
the conversion implementation source has SHA-256
`22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469`.
It cross-checks runner receipt score, trace length, and plain trace digest, creates the sole
canonical scorer NPZ, immediately reingests it, and binds the framed trace and score receipt.
The conversion itself remains in memory. A separate content-only atomic publisher now has
descriptor SHA-256
`5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905`
and implementation-source SHA-256
`8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5`.
It writes exactly five flat, single-link files through a held staging inode, replays them,
fsyncs files and directories, publishes with `renameat2(RENAME_NOREPLACE)`, and replays the
published inode under a caller-carried full-manifest digest. Synthetic temporary-root tests
cover both real structural runner-receipt parsers and a fresh-process reload; they are not
agent executions. No production publication has been emitted or accepted. Campaign
ingestion, seed-provenance authority, qualification, evidence, and every
performance/promotion claim remain false.

Separately, the isolated paper-scale RTU-PPO comparator uses 2,048-step rollouts and binds
exactly 244 rollouts because `499712 / 2048 = 244`. It may not inherit a rounding convention
that performs an extra rollout.

## Trial blocks and RNG

Development, calibration, held-out, historical screen, v1/v2 tuning, and all consumed
evidence seed registries must be pairwise disjoint.

One held-out block begins with a root token from a preregistered auditable future-randomness
mechanism. The block derivation must domain-separate:

- `environment` for the common exogenous environment key schedule;
- `agent/<candidate_id>` for each candidate's private initialization/exploration keys.

The implemented, still-uninstantiated generator plan has SHA-256
`90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace`.
It accepts exactly 32 root-token bytes per draw. For each seed, it frames the ASCII domain
`alberta.forager.matched_v3.trial_block.seed.v1`, the root token, and the UTF-8 namespace
as separate values prefixed by four-byte unsigned big-endian lengths, applies SHAKE256,
reads four output bytes as an unsigned big-endian integer, and clears the most-significant
bit to obtain a value in `[0, 2^31 - 1]`. The draw index is an exact integer in
`[0, 2^63 - 1]`; it distinguishes block identities but does not perturb seed derivation.
Thus a repeated root draw retains identical seeds, as required by sampling with replacement,
while its `block_<draw-index-hex>_<root-token-sha256>` identity remains unique. Frozen test
vectors and exact replay validation cover the encoding. No root token or randomness-provider
receipt has been issued.

All candidates receive the same environment seed. Agent seeds use candidate-specific
derivation namespaces, but the uint31 conversion can collide with the environment seed or
another candidate's numeric seed. Such collisions are retained without redraw. Runners must
carry the logical environment and agent roots separately; equality can still correlate key
chains and does not establish statistical independence. Qualification and seed-0 probes may
not derive or touch held-out tokens. The generator receipt must define whether sampling is
IID with replacement; silently removing a rare collision would change that population.

Every block is complete across the full inferential panel. A deterministic retry is allowed
only under the frozen failure policy. An unresolved cell invalidates the campaign; analysis
may not drop the arm or block.

## Simultaneous named-panel analysis

For candidate `i`, comparator `j`, and block `b`, let `D[i,j,b] = R[i,b] - R[j,b]`.
For `K` inferential arms, the frozen family contains all `M = K * (K - 1)` ordered
contrasts. With fixed `alpha`, sample count `N >= 2`, difference-width `W = 62H`, and
`q = ln(2M / alpha)`:

```text
s2[i,j] = sum_{b<c} (D[i,j,b] - D[i,j,c])^2 / (N * (N - 1))
LCB[i,j] = mean(D[i,j])
           - sqrt(2 * s2[i,j] * q / N)
           - 7 * W * q / (3 * (N - 1))
```

This is the finite-sample empirical-Bernstein lower bound with a union bound over the full
ordered family. It is deliberately conservative. The reported sample leader is the highest
mean score, with candidate-ID ascending as the deterministic tie-break. A named-panel-best
claim is allowed only if the leader's lower bound exceeds the preregistered practical margin
`delta` against every other inferential arm. Otherwise the result is exactly: “No
named-panel best was established.”

Required serialized assumptions are:

- trial blocks are IID draws from the exact frozen generator;
- candidates, configurations, sources, metric, runtime, and retry rules were fixed first;
- all candidate/block cells are complete;
- reward and score bounds are correct;
- environment and agent RNG derivations replay exactly.

No sign-flip, Holm, percentile-bootstrap, or selector-only calculation authorizes the v3
claim. Descriptive sensitivity analyses must be separately labeled. Fixed `N`, `alpha`,
`delta`, and a meaningful target gap are chosen before held-out derivation. Any stated power
target requires a separate valid beta/alternative calculation. If the fixed-sample precision
or resource plan exceeds the interaction cap, v3 records that the budget cannot support the
planned inferential design instead of weakening the test after seeing data.

The current constants make this arithmetic a load-bearing design constraint for any future,
separately bound campaign receipt. They do not create a current feasibility gate. For
`K = 11` and `alpha = 0.05`, `q = 8.389359819906353`. The range correction alone is approximately
20,913,132 score units at `N = 30`; even at the statistics contract's 50,000-cell ceiling
(`N = 4,545`) it is approximately 133,468. That ceiling would require 24,983,101,440
environment interactions before retries (`11 * 4,545 * 499,712`). The implemented generic
conditional-precision receipt recomputes this arithmetic but deliberately binds no selected
panel, models no beta, validates no calibration artifact, grants no authority, and is not a
campaign gate. Under the global unbiased-sample-variance ceiling, an assumed observed gap of
2,000,000 with `delta = 0` first clears the correction at `N = 2,585`, requiring exactly
14,209,310,720 inferential interactions before retries. That is a sensitivity calculation,
not a chosen target or power result. Before any held-out tokens are requested, a separate
campaign receipt must bind the accepted exact 11-arm panel, validated calibration (if used),
configuration/runtime/source closure, retry policy, and complete resource scope. These design
arithmetic facts are not benchmark results. If the fixed design and resource cap cannot clear
their preregistered gates, the confirmatory campaign remains unissued; development-only
descriptive comparisons may still be reported as such.

## Resource receipts

Equal environment interactions do not imply equal compute. Each candidate/seed receipt must
bind at least:

- environment interactions and optimizer/gradient/sample updates;
- trainable and frozen parameter counts;
- optimizer and target-copy elements and bytes;
- replay capacity and peak bytes;
- rollout storage and peak bytes;
- recurrent carry, RTRL sensitivity, and eligibility state;
- peak RSS, CPU time, wall time, temporary/disk peak, and thread count;
- hardware/runtime identity and retry/failure count.

Native tuned performance and matched-resource ablations are separate bundles. Unless a
specific equality contract proves otherwise, v3 may not make a compute-efficiency or
resource-matched claim.

## Artifact and execution boundary

Planned roots are new and digest-named, for example:

- `outputs/forager/matched_v3_development_<digest>_v1`
- `outputs/forager/matched_v3_qualification_<digest>_v1`
- `outputs/forager/matched_v3_confirmatory_<digest>_v1`

All writes use new exclusive-create paths. Component-level strict schemas, cross-version
tests, configuration replay, full adapter runners, external materialization, scoring, and
in-memory adapter conversion now exist. A non-authorizing atomic publication implementation
also exists, but no production adapter publication is accepted. The lane remains unissued
until accepted production source/runtime qualification, production-result publication,
campaign ingestion, resource receipts, campaign/seal closure, and final-analysis validators
exist.

Live OCI or scientific work remains gated on all of:

- system load no greater than the 24 logical-CPU budget across repeated stability samples;
- adequate free disk for source closures, traces, and receipts;
- stable Alberta and external source identities;
- exact image/runtime/helper/scorer/executor capability receipts;
- no conflicting Forager container or process;
- an externally provisioned authority if promotion is ever requested.

Without external authority, correctly executed results remain content-addressed,
unendorsed, and nonpromoting.

## Maximum permitted success wording

> Under protocol `<digest>`, candidate X had higher expected 499,712-step raw cumulative
> reward than every other member of the fixed named inferential panel `<panel digest>` under
> the preregistered trial-block generator. The selected-winner statement uses simultaneous
> empirical-Bernstein lower bounds with familywise coverage 0.95. This is panel-relative and
> task/runtime/horizon-specific; it is not universal, literature-wide, compute-matched, or a
> state-of-the-art claim.

No descriptor, development result, qualification, scalar matrix, or standalone statistics
artifact grants that wording by itself. Final analysis must replay the entire sealed closure
and all authority and prohibited-claim flags.
