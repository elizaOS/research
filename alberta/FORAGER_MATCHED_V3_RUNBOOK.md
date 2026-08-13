# Forager Matched v3 Development Runbook

Status: **component implementation plus non-authorizing dependency artifacts; campaign
unissued, runtime-unqualified, and no full-horizon benchmark executed**.

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
  `258b9e376b82127f912bf2828a6d4e5c7a257ed2a990cd15bf4c9cbd81c17788`.
  It is a score-blind, content-only contract with no default production plan. A caller must
  separately provide the exact two source closures, one complete runtime identity, 28
  permanently consumed public qualification cases, 28 resource ceilings, and 28
  result-publisher bindings. It neither executes qualification nor grants authority. Its
  external-source requirement retains the historical materializer-v1 identity; it is not
  silently upgraded to materializer v2.
- The additive compiled-PPO qualification descriptor has SHA-256
  `b5f7df77cd3f6e35126ed7c9f4b7acacdaa8237e8242241f658a95d21e9e3b06`.
  It reuses exactly the base plan's one `adapted_ppo_gru` public case and ceilings while
  selecting the compiled runner and six-file publisher only inside the addendum. It does not
  amend or supersede the base plan, execute qualification, or grant authority.
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
`74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192` and its
implementation-source SHA-256 is
`3ff59a9f88d79b122fa66a1cdca009a68ff524806a7a7c58e5d565cd30ecaafe`.
Materializer v2 accepts only the pinned direct-`.git` checkout, validates the complete
regular-file Git tree plus its one exact excluded gitlink and exact portable alias exception,
applies the four frozen transforms, strips Git metadata, and atomically publishes a
manifested derived tree while retaining a process-bound directory capability. Git inspection
is hermetic and wall/output bounded. A disposable, nonpromoting engineering replay covered
all 10,945 tracked entries and replayed the published manifest, but no production checkout or
manifest has been accepted. The archive bytes remain provenance rather than independently
verified input, and no manifest grants runtime or execution authority.

The external sealed-staging contract has descriptor SHA-256
`ceea86b38822f3add0465788003d349dd221a49fba5f3fa069bfec985537caea`,
implementation-source SHA-256
`675d54edcf2f87c1847712e7a480e2e5134312d040a68a1102c10c4829f8fba0`, and
test-source SHA-256
`d469fc2892bd756d017097f1baecf4be46792bfa5f365c174ca7456b8f483c1c`.
It accepts only a live retained materializer-v2 capability, replaces the 12 exact candidate
configuration paths with their frozen derived bytes, relocates the exact materializer
manifest, and streams a canonical USTAR into a sealed, unlinked, read-only descriptor. The
isolated worker imports an exact sealed snapshot of its own verified implementation source.
A disposable engineering replay reverified all 10,944 materialized regular files and all
10,946 archive members; every one of the 18 authority claims remained false and no workload
was executed. No replay archive, digest, production bundle, or execution receipt was retained.

The downstream durable external-source publisher now binds that staging closure under
descriptor SHA-256
`d76657b2f0d65adae377e21fa391628aa5749acb476c69aa64ce542a716f146d`; its
implementation-source SHA-256 is
`0b3a31f4a8117a51477b1f0c49925d2c77a6ae4311e77bcdb367cb4cc24566e9`.
No archive has been republished under this revised closure. Earlier one-shot build receipts
remain historical, nonauthorizing records of their older source closure and cannot qualify a
future run. Specifically, the retained context
`ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330`, execution
`38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d`, publication
`28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88`, and image
`sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269` are
classified `pre_v3_source_closure_drift`. They are excluded from every active fresh-build
binding; no automatic rebuild or reuse is permitted.

The descriptor-only external execution contract has SHA-256
`9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4`; its
implementation-source SHA-256 is
`7b806ffe70eb38f7db182c9ef4a56c5b800f499f8ee9973ad8ac156529506671`.
It binds the exact order, derived configuration path, entrypoint, arguments, result directory,
reward archive, sibling database, and PPO video path for all 12 external candidates. It treats
configuration-plan v1 as historical and selects materializer v2 only as a separate overlay.
It exposes no executor, filesystem mutation, seed issuer, result loader, or acceptance API.

The strict external-result bridge has descriptor SHA-256
`19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd`; its
implementation-source SHA-256 is
`c1859f0cfb7862e22c470f89ad9d3298a76b1fb419bf1431069f286f593e22f7`.
It maps the 12 exact external reward-array layouts into the frozen 499,712-step raw-reward
trace through a bounded ZIP/NPY parser and the strict scorer, then independently reconstructs
the ordered score, framed trace digest, canonical scorer NPZ, and complete scorer receipt.
Every public conversion and receipt exposes reward and score content, is permanently
nonqualifying, and is forbidden as input to a score-blind controller or publisher. No external
result has been accepted and no execution, publication, or ingestion authority exists.

The score-opaque external in-container worker has descriptor SHA-256
`2375d8c796b82b9317135a4bb2e48779e37c8a3a91bebfe7c238fbdd9efa6e94`; its
implementation-source SHA-256 is
`8e8f1ed88519dbda6276c1fd9756172eb1e37e244088f51caf2924bd6fe8484a`.
It binds and launches exactly one frozen external candidate, inventories only opaque result
bytes, and retains them behind PID-bound single-use capabilities. The worker remains
unexecuted here and is not a host OCI executor. A future host must prove a fresh isolated
worker, networkless/read-only runtime closure, resource observations, all-descendant cleanup,
and an empty cgroup or container before any content can enter a qualification path.

The 14 local worker-envelope configurations now have an implemented, content-checking,
non-authorizing builder. Its builder descriptor SHA-256 is
`1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b`.
The local runner descriptor SHA-256 is
`2237914749f353d2700bbb0f33a66d8789268a5e156f2961be2e626f42efd2a1`; its
implementation-source SHA-256 is
`aa2eb0fd642dec7ef62a4cb0fc555f6aaede6570a55c49adfa8425a264be91aa`.
The local source-snapshot contract has descriptor SHA-256
`6f24c9e6fa740780856783c2e3f42f01758e10ba2e2084c40c42aa72a895090e`; its
implementation-source SHA-256 is
`c66d148b1f16574ac03d9e64bf87d4716caa1ce985a9258c3c5e36cb44cd6fda`.
The standalone local bootstrap has descriptor SHA-256
`9b2e51ab5e9bdfbb2373e411120a6b1030c66c6a105b57ea6be6215c97a87a17`; its
implementation-source SHA-256 is
`1240aff1329ee4322ef7087801213aa125497a5110da2c477bcecb8a4fea905e`.
It verifies the full snapshot before and after a fresh `-I -S -B` child, uses exact
descriptor transports and a private bytecode/cache boundary, forces CPU JAX, and requires
separate process-bound execution and outcome capabilities. A transient engineering snapshot
was measured and immediately replayed without being persisted or accepted. No production
snapshot, local workload, or completion receipt exists, and every local candidate remains
execution-unready.

The local result-to-publication path is now an exact one-way trust pair. The private reward
bundle has descriptor SHA-256
`c4fc32c0194677af5c94849a0e457eb967b45eb28df0eed80cd68f0cc8fda315` and
implementation-source SHA-256
`e966cd7885df30ad9753992a213c537c9f2fff98dda9c93e98e563b6653bc0d8`.
The direct atomic publisher has descriptor SHA-256
`a7c13b79fb35e6248b45f2997b0b406fddb7db5b7eeb9681b226fcb122a418d6` and
implementation-source SHA-256
`2e840406f58253c1f02969aae2cf679a5383d93bbacd3239950f0f5054b806d2`.
The bundle statically pins the publisher; the publisher dynamically replays the bundle source
and canonical descriptor, avoiding a mutual source-hash cycle. Public inspection and direct
publication are mutually exclusive single-use choices. Publication commits exactly nine flat
files once under the full SHA-256 of `publication.json`; publish and reload return immutable
metadata only, and the metadata has a canonical full-file receipt plus a strict caller-pinned
parser. Reload reconciles the publication manifest's eight records and the local manifest's
seven leaf records against the bytes actually reloaded, so caller-supplied records cannot
replace either committed inventory.

This boundary does not claim information-theoretic score opacity: exact content digests and
sizes remain visible metadata and a qualification controller must be independently prevented
from branching, retrying, or selecting on them. Same-process hostile Python and same-UID
filesystem confidentiality are also outside the claim. No local qualification worker, fresh
source snapshot, production publication, accepted observation, or authority exists, and the
older OCI image cannot contain this source closure.

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
compiled full-horizon execution has been accepted. On 2026-08-02 an unpersisted, nonpromoting
engineering probe opened the exact installed CPU runtime, validated a `(9, 9, 3)` float32
observation, completed three transitions with exact reset-plus-step key accounting, and
replayed the same seed/actions with byte-exact observations and rewards. That tiny probe is
not a qualification receipt and does not change any false readiness flag.

The earlier upstream-derived comparator OCI image is
`sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768`.
Its dependency-lock label exactly matches upstream `uv.lock` SHA-256
`46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd`
and includes JAX/JAXlib 0.9.0.1, Flax 0.12.3, Optax, Distrax, and Flashbax. It is a negative
control only: it cannot be reused or relabeled as the v3 runtime because the adapter bridge
requires JAX/JAXlib 0.11.0 and the image does not satisfy the frozen CPU/cache contract.

The qualification-plan v1 contract deliberately requires one unified, networkless CPU OCI
runtime using JAX/JAXlib 0.11.0 for all 28 public qualification cases. The current JAX
0.9.0.1 image cannot satisfy that contract. A future production plan therefore requires a
newly pinned unified image whose external, local, and adapter candidate imports and exact
seed/trace behavior all pass the predeclared score-blind probes; separate family runtimes
would require a separately versioned qualification-plan contract. The current retained r5
engineering solve preserves JAX/JAXlib 0.11.0, Optax 0.2.8, dm-haiku 0.0.17, and Flax
0.12.8 from r4. Its sole registry-package delta is exact released
`distrax 0.1.7 -> 0.1.9`; the other 103 registry selections are unchanged. The resulting
CPython 3.12 Linux x86_64 closure contains exactly 104 wheels and no CUDA or development
packages. Its resolution `uv.lock` is SHA-256
`6f6127c1b4d970c432bf29f6c7e8e65230b966cbf6197cf4e462822e84ef725d`
(106,980 bytes). Its exact 11-file, non-authorizing solver-content manifest is SHA-256
`325f91ad66fe26d36d46ad2588621fbb010f2a879ac6c56ab0922db3d9d4d1a5`
(2,331 bytes), with publication-body SHA-256
`402bb9e4a706d8c854a58889bbb54ef2989d2d753ca6e4a7a8ffe972604879ac`.
The retained resolution report is 5,541 bytes with SHA-256
`75eebf630d80271ce3d08dfb4adcfeb33c805222cf3320f95174dd9a3ee1f17e`.
The live index snapshot and solver cache are not retained, so this content is provenance for
the disconnected result rather than an independently authoritative solve.

The pure CPU runtime-lock schema has descriptor SHA-256
`31d4c5a101f441bc082bdaf9250050f7950440271e6360854d5faa9fcd7ff34a`,
implementation-source SHA-256
`08c232c50714891a86f5332df84c531252ff956d50e6eccfe39a30842a02fa2a`, and
test-source SHA-256
`c19fe578b9ebd13e2111ccfc7812a8e3bb72ddd25952afa468a389ab34e0b19a`.
Its separate production gate requires exactly 104 registry-wheel distributions for
CPython 3.12.3 on CPU-only Linux x86_64 with glibc at least 2.28. The schema binds the exact
upstream source inputs, overlay operations, solver provenance, complete marker environment,
wheel and core-metadata identities, PEP 440 requirement syntax, selected extras, active-edge
reachability, and a content-addressed wheelhouse identity. It performs no filesystem, wheel,
network, solver, installation, import, or execution operation, and every authority or
readiness claim remains false.

The pure-content issuer implementation SHA-256 is
`b009fcd22741268ce55188bd4b468f3de6cf3a0e83b5d0ad8e5924368b8ebc04`; its
test-source SHA-256 is
`1533f8ae64107b540fa50db5a5459c3513ebe16b5ccd347d8a9ec93b9d076084`.
It reconstructs the lock and wheel CAS mapping from the exact retained capture, receipt, and
four-operation provenance envelope. The separate production reissuance gate accepted the
caller-pinned 36-root and 104-wheel identities and produced these read-only, ignored
non-evidence artifacts:

- issuance envelope SHA-256
  `30ee57e9df1e1805d7a338d250daf99849170a765fe66613466664af38421eae`
  (49,145 bytes), body SHA-256
  `01a51a461afe0ca8ec03fcf7dd14d2f5aa38036d7d19bf13c1b8617e09fedd24`;
- CPU runtime-lock SHA-256
  `f4089e4631bc1a8817827a27ab58943968c634f2a3c54ea4f54385c2163a8641`
  (356,996 bytes), body SHA-256
  `09cccab7af4e717daf0a6a3c8664fd8b7d875e51f8721e3d2abad8f7f77ec565`;
- wheel CAS-manifest SHA-256
  `e9ea3ee9faaecf09ba4367db47ab8fe7d281505b96099d3963743fbe9fc1cc46`
  (72,679 bytes), body SHA-256
  `234e6f1718e475a659571e0f8bc6d0fed91ccc6c3242e9f2c6695b55b442de3f`.

The three files are published new-only under
`.tools/forager-matched-v3-cpu-v1/runtime-lock-publications/sha256/f4089e4631bc1a8817827a27ab58943968c634f2a3c54ea4f54385c2163a8641/`.
The gate bound 36 exact roots under inventory SHA-256
`2f175de86b18b7d72772dd093902f801f423bd37393fde9133377528e4a12d47`
and 104 selected wheels under inventory SHA-256
`8cbe5daa6a66e87672fce419cf40f2b6769fbceea8eca3ded7e33401b3a618e6`;
the derived CAS wheel inventory is
`991834df9ddfc8ce2e6a71c7ed321a1cef5b21f2563b704b8168ed838ec5dfad`.
New-only publication, immutable readback, the production gate, and a second pure-content
issuance all reproduced the three files byte-for-byte.

The lock is a content-issued runtime candidate, not an installed or qualified runtime. It
does not establish source installation, imports, CPU-only execution, OCI identity, or
qualification success.

Four earlier immutable dependency publications are rejected engineering lineages, not
fallback runtimes:

- r1 (`17a89408...` wheelhouse, `f26a4767...` issued lock) failed because Optax 0.2.7
  imports a JAX API removed in 0.11;
- r2 (`e99c8bc9...` wheelhouse; no issued lock) fixed Optax but dm-haiku 0.0.16 failed on
  removed `jax.core.DropVar`;
- r3 (`2788d7aa...` wheelhouse; no issued lock) fixed Haiku but Flax 0.12.3 failed during
  `linen.Dense.init` on removed `jax.core.get_opaque_trace_state`.
- r4 (`f22a96b7...` wheelhouse, `beee7c13...` issued lock) passed the broad dependency
  smoke probe but was rejected when exact-source PPO gradients reached removed
  `jax.core.get_aval` through Distrax 0.1.7.

The r5 fresh offline engineering probe installed exactly the published 104 wheels with the
index, cache, configuration, dependency resolution, and source builds disabled. Completed
CPU operations covered JAX JIT/gradient, Optax update, Haiku transform/init/apply/JIT, Flax
Dense init/apply/JIT, Distrax categorical log-probability JIT/gradient, Gymnax reset/step,
and Foragax reset/step. A separate network-disabled, read-only exact-source probe then
completed both differentiated-loss routes: `isolated_ppo_generic` and
`isolated_rtu_paper_scale`, each with a finite gradient tree. These probes are compatibility
checks, not OCI or qualification receipts, and change no readiness or authority flag.

The disconnected CPU wheelhouse contract has descriptor SHA-256
`b74224c7bb0523b87458cb4a08aaf9967b5fd11574927d9635cf9a93bc417331`,
host implementation-source SHA-256
`22c2c03ef4a6dbffc018d2426eaebeefc92ce7fc6f1508d65609b89e9468e176`,
isolated-helper source SHA-256
`ea80e1860a0af0d376ed1be0b1c09ef74a34db2d7acd983db53ef4ffa09e99f9`, and
test-source SHA-256
`e6bee490079b85df9c7ecb4442ccc5e91b5f72af135974390852f938950a13df`.
It copies caller-enumerated candidate wheels before inspection, then executes exact sealed
snapshots of CPython 3.12.3, its frozen helper, and a separately hash-bound `packaging` wheel.
The helper validates ZIP structure, RECORD coverage and hashes, METADATA/WHEEL identities,
target tags, PEP 508 closure and extras, critical versions, accelerator exclusions, and graph
reachability. Exact original wheel filenames up to 255 bytes remain in the receipt while the
canonical USTAR uses deterministic `<wheel-sha256>.whl` member names. The archive is retained
through a PID-bound read-only unlinked descriptor, and optional publication is explicit,
caller-rooted, content-addressed, new-only, and rollback-checked. The Python loader, shared
libraries, standard library, runtime prefix, kernel, verifier provenance, and verifier side
effects remain outside this content contract.

The separately authorized capture contract has descriptor SHA-256
`8ebdec1eb47401b6c0fec4508f4649c575f22dc54c48810e28f8b5b5f2d0d0b0`,
implementation-source SHA-256
`716c9ffd996412cc88fde3fcc44f50280948f457fca2aba7e241815b83bc74ee`,
and test-source SHA-256
`c4e01a1f9549fd8fe42502735438a4276ff40087528683e1f702aa8dbac64593`.
It accepted only exact lowercase `files.pythonhosted.org` wheel paths. The retained base
capture held 109 wheels; targeted captures added exact replacement wheels. The r5 Distrax
delta manifest is SHA-256
`6389585ef9ba70c6d0ca441a8dc33fe4ba44c16da70c360cc417745f2a5fdeaa`
(26,859 bytes), with body SHA-256
`63c30934ac382e7dd0d01a760e44769344ea13bbca43c1c3ad16f26765548fef`.
It commits only `distrax-0.1.9-py3-none-any.whl`, 313,467 bytes with SHA-256
`11c93bd4dd913803f2539847cf3a688804fae5f9da224dc44e340846753f75f4`.
The final combined 104-wheel manifest is SHA-256
`f4d674e88f2a29047a0296ca84432cb08d05b631f96c9e75653c31df25c7275d`
(61,762 bytes), with body SHA-256
`1f8d7a75786ebf84a40e78dbcfcf73ac650f9fa2eb017b75cd5a92de05aac626`.
The disconnected verifier then published canonical USTAR SHA-256
`f396944111366df1e243214547d17c5ed35d517f0508ecff7b4a2edec1e881a7`
(573,061,120 bytes, 104 members), archive-inventory SHA-256
`14f99cb2b3daf8e4c121b4ed6641db792c04b10e58217f883d96d5d90eaad0c0`,
and receipt SHA-256
`51dc757abc25f07347c0b7b1416a61e149e72707485e3a54fdfd31765a53a1c6`
(460,516 bytes), with body SHA-256
`9ad8aeb755879a6624e3414ac438418e4d6ce58d40ea30ce4b5c12e64089df84`.
These are untrusted network input followed by disconnected content validation; they do not
authenticate PyPI, establish package safety, create an OCI image, or grant execution or
qualification authority. The separate fresh offline probe above is the only installation
reproduction described here.

The one-image requirement applies to the final source-qualified image, not merely a shared
dependency base. Both exact source closures must be baked into that image from retained,
verified descriptor streams. Docker resolves bind sources in the daemon's namespace, so a
client `/proc/self/fd` path or mutable host source mount cannot preserve the verifier's inode
capability and is not an acceptable production boundary.

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

The compiled PPO-GRU path is separately implemented but unexecuted. Its runner descriptor
SHA-256 is
`3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565`; its
implementation-source SHA-256 is
`08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f`.
The compiled reward-bundle descriptor SHA-256 is
`cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08`; its
implementation-source SHA-256 is
`e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e`.
The six-file atomic publisher descriptor SHA-256 is
`a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500`; its
implementation-source SHA-256 is
`42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71`.
These components bind the exact 976 compiled chunks, runtime identity, complete reward trace,
strict scoring, and publication replay. Their tests use synthetic content and structural
receipts; no compiled candidate, full-horizon workload, production bundle, or publication
has run.

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
tests, configuration replay, full adapter runners, external materialization and result
conversion, local source measurement and execution plumbing, scoring, and in-memory adapter
conversion now exist. Non-authorizing atomic publication implementations also exist for the
base and compiled adapter paths, but no production publication is accepted. The lane remains
unissued until accepted production source/runtime qualification, production-result
publication, campaign ingestion, resource receipts, campaign/seal closure, and final-analysis
validators exist.

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
