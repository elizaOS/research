# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the canonical first-principles micro-continual benchmark suite
  (`gauss-v1`) to `alberta_framework/benchmarks/micro_continual.py`:
  seconds-scale synthetic Gaussian-mixture streams distilling the IPMNIST
  campaign's measured difficulty axes into minimal form (M1 input
  permutation, M2 label permutation, M3 scale shift, M4 recurrence), with
  analytic Bayes references known by construction (closed-form mixture
  discriminant + exact two-class formula, transform-invariance pinned), a
  six-arm method ladder reusing the campaign's registered factories
  (published UPGD-W/AdamW, conditioned-SGD floor, `sigma0_shiftnorm_d099`
  champion form, streaming naive Bayes), immutable idempotent shards, and a
  pre-registered transfer-validation CLI. The frozen M1 operating point
  (dim 256, 6 sparse components/class, 2-decade marginal spectrum, 100 x
  5000 steps) reproduces the full protocol's method ordering on 3 paired
  seeds — conditioning dominates (+0.401), gate small-positive (+0.0098,
  all seeds; protocol +0.011), Adam fast-early-then-decays, naive Bayes
  between raw UPGD-W and conditioned SGD, champion on top — at 21x
  (champion-form) to 106x (SGD-class) per-arm wall-clock speedup over the
  200-task confirmation lane. Calibration history, generator-design
  findings (input-norm operating point, within-class multimodality, sparse
  localized features are load-bearing for proxy validity), and
  discovery-lane coordination live in `outputs/micro_continual/SUITE.md`.
  Development screening diagnostics only — never promotable evidence.
- Added the automated update-rule discovery lane
  (`alberta_framework/benchmarks/rule_discovery.py`): a composable,
  branchless JAX DSL over the IPMNIST campaign's primitive vocabulary
  (per-feature EMA statistics, shift detectors, utility gate, L2-init pull,
  decays, shift-triggered resets, input/hidden normalization, error
  signals) whose champion-form genome reproduces the registered
  `sigma0_shiftnorm_d099` step (pinned), plus a vmapped random+evolutionary
  search harness over the micro continual suite with a budget-matched
  tuned champion-form baseline, held-out selection validation (M4 + M1'
  + the canonical Gaussian cross-suite), and two hand-designed meta-arm
  seeds (surprise-driven per-statistic decay; error-gated plasticity
  budget). Discovered promotions are registered as `disc_r1`-`disc_r3`
  screening arms through the new `_make_discovered_rule_learner`
  translation factory in `ipmnist_screening.py` (champion-form flags
  reduce bit-exactly to the champion arm, pinned) and screened on the
  real 60-task protocol, with `disc_r1_pscale{,_norms}`
  structure-vs-constants dissection arms. Outcome
  (`outputs/rule_discovery/real_screen_v1.json`): verbatim micro-tuned
  constants do not transfer (best 0.78372, below the 0.8640 bar; hidden
  RMS isolated as a −0.051 protocol-scale cost), but the discovered
  *structure* — error-gated plasticity budget replacing the utility gate
  under the champion's shift-adaptive conditioning — at champion-scale
  constants scores 0.86570 on the 60-task screen (+0.00173 paired over
  the champion, all 3 seeds) and 0.86525 at 200 tasks (+0.00066 paired,
  2/3 seeds). Development screening diagnostics only — never promotable
  evidence.
- Added the stateless `ExternalBuilderCandidateEvidenceProducer`. It binds
  caller-owned objective, retention, and safety representation probes to one
  exact external-coordinator event/GRU/Prototype/feature-generation/decision
  identity and analytically pulls them through the cached source full-GRU RTRL
  sensitivity into candidate-audit parameter space. Stale or non-finite
  inputs become unavailable exact-zero evidence; probe independence remains
  an explicit caller attestation. The real coordinator acceptance path is
  tested with zero extra model forwards. This producer executes no actor
  backward, establishes no “sparks joy” fact, and grants no evidence,
  promotion, safety, or outcome authority.

- Added `LearnedExperientialMemoryController`, a separate L0 owner around the
  unchanged fixed-capacity experiential store. A bounded seven-feature linear
  model may veto but never relax a fixed-store retrieval, while learned
  nonnegative per-exemplar retention values enter the store's real eviction
  rule. Query, admission, access accounting, and write remain atomic and
  query-before-write. One pending receipt plus insertion-clock, provenance,
  and source identity prevents delayed feedback from following a reused slot.
  Learning requires caller-declared use and a bounded same-decision
  counterfactual utility delta. Exact resources, zero RNG, strict checkpoints,
  rollback, and eager/JIT/scan behavior are tested. Feedback is not
  authenticated, and the standalone controller is not itself a benefit result,
  evidence, promotion, or WP8 exit.

- Added v1 `ExternalLearnedStateLiveMemoryAdapter`, which makes the learned
  experiential controller the sole memory owner around one external
  full-GRU/router/audit coordinator with Prototype's historical memory
  disabled. Exact prior feedback settles before one coordinator update; the
  next raw decision observation queries before the actually executed current
  transition is written. Only admitted exact one-hot retrievals may use the
  public cached-action replacement under the caller mask. Pending state binds
  the pre-retrieval and effective actions, retrieval, Prototype decision,
  memory transaction, and exact mask. Soft/fallback retrievals cannot learn,
  and any stale feedback, donor failure, corruption, or replacement failure
  rolls every owner back. Fixed resources, strict checkpoints, and host-only
  orchestration are tested. This is L0 mechanism integration without caller
  authentication, dispatch/safety authority, benefit, evidence, or promotion.

- Added v1 `HCCLExternalCoordinatorBaseBridge`, a host-only atomic owner over
  one HCCL world/attribution state and two independently initialized external
  learned-state coordinators. Their exact cached primitives bind as `B=M=P`
  under common hard masks and six deterministic decision/lifecycle/clock-bound
  receipt identities; a mask excluding either cached action is rejected with
  no fallback. The PP proposal updates each coordinator exactly once from its
  own action, net reward, and next raw observation, and all three owners commit
  together or roll back bit-exactly. Zero memory/planner contrasts are scoped
  to this ablation, not delight or actor-gradient judgments. Strict in-memory
  checkpoints and resources are L0 contracts with no memory/planner, schedule,
  seed, artifact, evidence, benefit, or promotion authority.

- Added v1 `HCCLTwoLiveMemoryBridge`, a host-only atomic composition of one
  HCCL world/attribution state and exactly two live learned-memory adapters.
  Pending child receipts provide each agent's exact base and effective memory
  actions; absent receipts abstain as `B=M`, and `P=M` is an explicit
  no-planner rung. Exact unilateral HCCL effects settle separately as
  agent-0 `M0B1-BB` and agent-1 `B0M1-BB`, while memory interaction remains an
  audit fact only. Each adapter advances once from its own executed `M` action
  and `PP` net reward/next observation; current and next-decision masks remain
  distinct, and all three owners commit or roll back together. Strict
  checkpoints/resources and fail-closed pending-mask admission are tested.
  This is L0 memory-utility integration, not delight, actor backpropagation,
  evidence, benefit, or promotion.

- Added the separate v1 `HCCLTwoLiveMemoryPrepareAdoptBridge`. It leaves the
  existing bridge state and API unchanged, evaluates the HCCL and both live
  adapters once during a transient host-only preparation, and binds each
  nested raw/final STOMP owner, complete owner-finalization trace, candidate
  state, and independently sized extended-action mask into exact-content
  downstream receipts. Adoption performs integrity checks and the two child
  adoptions without reevaluating a world, coordinator, Prototype, STOMP,
  builder, or learned memory; one veto, tamper, replay, or foreign binding
  returns the complete three-owner source and outer-gates public child-applied
  facts. Partner fusion is permitted only when the shared generated-feature
  axis is immutable. This remains a transient, unauthenticated `P=M` L0 seam:
  it has no planner, Kondo actor backward, delight/“sparks joy” fact, run,
  evidence, benefit, or promotion authority.

- Extended `CumulantOptionInstallation` with an opt-in reserved observation
  suffix derived from an empty STOMP template whose width exceeds the raw
  discovery prefix. Installed option cumulants retain their exact compact
  positions immediately after the raw prefix, while standalone
  materializations fill and validate the reserved suffix as exact zero. This
  lets a separately bound external owner use later stable/generated cells
  without moving option semantics. The historical zero-suffix configuration,
  serialized STOMP layout, and behavior remain unchanged.

- Added the stateless v2 `FreshColdSlotCumulantCohortFilter`. Its explicit
  candidate-universe manifest rechecks fixed family quotas, slot-family
  layout, descriptor uniqueness, local discovery gates, pair novelty against
  every live option, and semantic freshness. The original six-candidate
  development universe deterministically remains unavailable; adding one
  eligible same-family feature descriptor produces a sealed proposal that
  changes only the cold feature slot. Tamper and even a checksum-valid v1
  cross-family splice fail closed. This surface owns no state, RNG, install,
  adoption, go/no-go, safety, evidence, promotion, or delight authority; it
  does not repair or rerun the consumed repeated-lifecycle result.

- Added three isolated, defaults-off WP2 plasticity mechanisms:
  `SpectralRegularizer` implements the dense ICLR 2025 objective with a
  checkpointed one-step power probe; `AdamO` keeps task-gradient Adam moments
  separate from its rectangular Gram-isometry delta; and
  `CalibratedPartialResets` implements per-example incoming-gradient utility,
  normalized EMA calibration, scheduled utility-scaled incoming resets, and
  matched outgoing decay against the pinned author JAX reference. CPR leaves
  biases untouched because the paper appendix and released v1 implementation
  disagree. All three have exact clocks, atomic numerical rejection, fixed
  resources, strict checkpoints, and eager/JIT/scan equation tests. They are
  single-layer L0 mechanisms, not Prototype arms, matched results, defaults,
  evidence, or a WP2/SOTA claim.

- Added a strict development-only nonlinear differential actor-critic versus
  `DifferentialSARSAAgent` six-state RiverSwim A/B/A lane. It retains complete
  prequential per-arm traces, explicit target/behavior semantics,
  actor/critic-or-Q, reward-rate, churn, descriptive recovery, logical work,
  persistent resources, source/config/seed-bound checkpoint resume, and
  deterministic whole-report replay. Common action and environment key roles
  do not imply paired trajectories after causal divergence. Successor timing,
  parameterization, persistent bytes, and realized scalar update work remain
  explicitly unmatched; the lane is `not_assessed`, threshold-free,
  winner-free, artifact-free, and nonpromoting.

- Added the opt-in L0 `PrototypeOptionAuthorityBridge`, with one persistent
  Prototype→OaK→STOMP owner and detached scheduler/installation/lifecycle
  metadata. An explicit directional, exact-source and typed-owner-bound
  receipt reconciles unequal pristine owners; exact-source reuse is
  idempotent, and all receipts/checksums remain unauthenticated integrity
  bindings. One cold/live mask crosses real control, bootstrap, internal
  planning, option search, guarded Dyna, and lifecycle attribution. The
  lifecycle consumes Prototype's exact raw STOMP result without reevaluation,
  then a five-stage transient trace classifies search, feature routing, Dyna,
  memory, and partner mutations before metadata-only finalization binds the
  sole final owner. Invalid bridge sources cannot commit and use a
  primitives-only transient mask; dynamic audit refusal preserves valid
  Prototype control while latching authority desynchronization. Optional
  Prototype sidecars pass through unchanged, and diagnostics split real,
  imagined, total, search-update, and internal-planning work. Retirement and
  replacement remain caller-authorized and limited to one each. This is
  `not_assessed`, with no authentication, autonomous authority, benefit,
  safety/dispatch, evidence, promotion, SOTA, WP7 exit, or Alberta Plan claim.

- Added the opt-in v18 `PrototypeAgent` atomic feature/world/memory lane. One
  pair lifecycle/router authority now owns the generated bank used by linear
  OaK, an ordered linear Horde, a fixed-physical-output routed world model,
  and exact feature-bound experiential memory. Each real transition evaluates
  the feature and world learners once, prepares old-bank and destination
  successors, and admits a descriptor change only when lifecycle, world, and
  memory are all ready. A veto retains every valid ordinary old-bank update;
  a lifetime-cap no-op locally derives only the authenticated current encoding
  while leaving lifecycle state and all adoption flags unchanged. Planning is
  disabled by default. Exact v18 checkpoints, fixed ownership/resources,
  tamper/stale rollback, and eager/JIT/scan contracts are covered. This is L0
  `not_assessed` composition, not selective-retention, planning/control,
  evidence, promotion, or Alberta Plan completion.

- Added strict OaK adoption seams for a caller-authoritative STOMP transition
  and quiescent option-slot rebind. `OaKAgent.adopt_stomp_update` accounts one
  externally evaluated result without evaluating STOMP again and validates
  complete source identity, outer/nested clocks, endpoints, and success
  diagnostics. `rebind_option_slots` permits only reset-slot policy, model,
  trace/optimizer, and extended-action-head changes, preserves global and
  primitive state plus RNG/clocks, and zeros only the matching OaK statistics.
  Optional extended-action masks now exclude cold options from selection,
  real bootstraps, and planning. These are unkeyed trusted-caller integrity
  seams with no autonomous lifecycle or outcome authority.

- Added `PrototypeEmbodiedDevelopmentHarness`, which composes exactly one
  semantic command adapter, bounded deterministic plant, and grounded shadow
  state. Accepted or certified-fallback settlement proposes the actual mapped
  plant transition first, settles the semantic dispatch owner, applies exactly
  one real Prototype transition from the plant reward and successor, and
  atomically adopts that live rearmed owner with the transient shadow result.
  Two consecutive actions run without reinitialization while Prototype/OaK
  clocks and raw observations remain plant-bound. No-action is envelope-only
  and retryable; stop latching, fallback, shadow mismatch, replay, tamper, and
  checkpoint resume fail closed. Exhausting the plant quota halts later
  scheduling without fabricating termination or truncation. The harness is
  development-only, assesses neither delight, KondoGate intent, nor
  KondoSparseActor executed-backward inclusion, and has no
  physical-dispatch, safety, authentication, efficacy, evidence, or promotion
  authority.

- Added the strict four-attempt
  `PrototypeEmbodiedPairedDevelopmentBenchmark` around two independently
  owned real harnesses. The adaptive and `zero_stomp_step_size_control` arms
  may differ in exactly five declared STOMP optimizer step-size fields; all
  other initial semantic arrays, RNG, caches, observations, traces, and clocks
  must be dtype/shape/typed-key-implementation/host-byte identical after
  normalizing the materialized base-LMS step-size leaves. Both starts also
  require empty harness pending/last-commit records, an unset adapter
  settlement ledger, zero Prototype/OaK/adapter/plant clocks, and sufficient
  remaining capacity. V1
  fixes the bridge disconnect at attempt 1: the other three attempts commit
  real plant/Prototype transitions per arm, while the disconnect yields exact
  unavailable action/reward sentinels. Raw traces
  retain action/reward availability, fallback/no-action facts, plant state,
  Prototype/OaK clocks, rearming, adopted learning, grounded-shadow work, and
  exact logical/resource accounting. Both lifetime AUCs are normalized
  trapezoids over named committed-transition or attempt indices, never post-
  change adaptation AUC. Sixteen fast synthetic pytest contracts cover typed-
  key identity, exact sentinels, signed-zero
  identity, live selected-source/runtime drift, causal replay, prefix
  reconstruction, and content-plus-resealed tampering. The real slow lane runs
  only through `alberta-prototype-embodied-paired-development` and writes no
  artifact. Every status is `not_assessed`; there is no winner, threshold,
  physical dispatch, efficacy/safety/evidence/promotion authority, delight/
  KondoGate-intent/KondoSparseActor-backward assessment, or semantic use of
  the historical `GradientJoy` compatibility names.

- Hardened the operational meaning of “does this gradient spark joy?” in the
  Kondo actor tests. Unqualified delight remains advantage times selected-
  action surprisal; a sample sparks joy only when its actor-gradient
  contribution enters an executed backward pass. In sparse and full-shape
  fallback paths, perturbing
  rejected features, actions, and detached advantages now has to leave the
  actor loss and gradient bit-identical. The standalone `KondoGate` remains a
  forward admission plan: `backward_admission_intent` names that plan, while
  its historical `sparks_joy` accessor is compatibility shorthand. Canonical
  gate config/checkpoint emission is now v2 and serializes only detached
  `backward_admission_intent_semantics`; exact v1 payloads import through an
  explicit legacy path and normalize to v2.
  `KondoSparseActor` is the execution boundary. Every floating gate diagnostic,
  including selected-action surprisal, is now stop-gradient. A rejected
  nonfinite update still reports which contributions entered the executed
  backward without advancing committed counters, while a finite Bernoulli
  zero-survivor backward commits an exact parameter no-op with no joyful row.
  The separate safety mechanism now exposes canonical `CandidateUpdateAudit*`,
  `assess_candidate_update`, `apply_candidate_update`,
  `PrototypeCandidateUpdateAuditEvidence`, and
  `candidate_update_audit_evidence` names. Historical `GradientJoy*` and
  Prototype keyword spellings remain exact compatibility aliases.

- Versioned all three Kondo development evaluators to v2. Cross-arm outcomes
  now use neutral executed actor-backward inclusion terminology; replay and
  on-policy records expose `executed_actor_backward_mask` with semantics
  `gradient-contribution-entered-executed-actor-backward`. `sparks_joy` is
  reserved for an actual `KondoSparseActorResult`: ordinary-full and uniform-
  sparse use manual backward kernels rather than Kondo transactions, and
  ordinary-full makes no delight-selection claim. The config, protocol,
  deterministic, timing, report, checkpoint, and exact-replay validators fail
  closed on v1 payloads.

- Added `PrototypeEmbodiedCommandAdapter`, a source-bound identity bridge from
  the consolidated semantic-memory Prototype dispatch owner to the real
  `EmbodiedSafetyEnvelope`. A fixed unique primitive-command bank, complete
  telemetry/control/version receipt, settlement-time envelope recomputation,
  bit-exact result comparison, reverse command mapping, and the persisted hard
  action mask jointly gate settlement. Accepted and certified-fallback
  commands atomically adopt the envelope and semantic credit owner. Exact
  no-action and stop-only results instead retain the semantic owner for a fresh
  attempt while preserving the envelope rejection ledger or emergency-stop
  latch. Checkpoints, fixed resources, eager/JIT behavior, replay/tamper,
  disconnect, fallback, and capacity-exhausted stop paths are covered. The
  adapter neither dispatches hardware nor certifies command geometry, caller
  identity, physical safety, efficacy, evidence, or promotion.

- Added externally coordinated prepare/adopt transactions to the bounded
  Prototype pair-feature lifecycle (with and without its ordered linear Horde)
  and the isolated generated-input/fixed-physical-output world model. Each
  preparation evaluates its ordinary learner exactly once and carries both the
  exact source-bank successor and routed destination candidate; adoption
  evaluates neither learner nor router. A trusted external veto retains the
  already-computed ordinary update and records a curation/route rollback rather
  than a deferral, including when the world-model destination route is invalid.
  Stale, tampered, internally invalid, or mismatched receipts preserve the
  authoritative source. Receipts provide unkeyed content integrity, not caller
  authentication, and transient resource fields count serialized logical
  PyTree leaf occurrences rather than allocator-level physical peak memory.
  Taken alone, these remain isolated L0 coordination primitives; the v18
  coordinator above consumes them without creating a benefit/evidence result.

- Added the reviewer comparison wave to the IPMNIST screening lane
  (`alberta_framework/benchmarks/ipmnist_screening.py`): the strongest
  published plasticity mechanisms re-implemented from their papers and run
  behind the campaign champion's EMA input conditioning (decay 0.99) on a
  plain-SGD base — `wclip_ema_norm` (Weight Clipping, Elsayed et al. RLC
  2024: post-step per-layer clip of weights and biases to ±2/sqrt(fan_in)),
  `fade_head_ema_norm` (FADE meta-learned per-parameter head decay,
  arXiv 2604.27063, published constants), `snr_ema_norm` (Self-Normalized
  Resets, Farias & Jozefiak arXiv 2410.20098: per-unit geometric-tail
  hypothesis test on EMA-estimated firing rates, eta from the paper's sweep
  grid), `l2init_ema_norm` (L2-Init, Kumar et al.: decoupled decay toward
  the initial weights), plus the mechanism-free floor `sgd_ema_norm_d099`.
  Every factory reduces bit-exactly to the shared normalized-SGD base with
  its mechanism constant inert (pinned in
  `tests/test_ipmnist_screening.py::TestComparisonArms`). 60-task screen
  (seeds 0-2, paired) plus 200-task step-mode confirmations:
  `l2init_ema_norm` 0.86457 (campaign co-best, +0.0021 over
  `sigma0_ndecay099`, all seeds positive, tying `sigma0_shiftnorm_d099`
  0.86459 by a different mechanism); `sgd_ema_norm_d099` 0.86168 (the
  utility gate is worth only +0.0008 under decay-0.99 conditioning);
  Weight Clipping 0.85029, FADE 0.81620, SNR 0.77885 (churns to the raw
  baseline at one example/step — their batches of 16 make the same test
  benign in-paper; documented in the arm docstring). Development screening
  diagnostics only; nonpromoting; no evidence-registry claim touched.

- Added a separate default-off `PrototypeSTOMPCalibratedDispatchAgent` v2
  composition around the unchanged calibrated-search v1 sidecar. After an
  exact prior-arm settlement, it permits only candidate-specific evidence at
  the current exact anchor to form a primitive or option-keyboard proposal,
  intersects the result with a caller-owned hard action mask, changes the
  cached action through Prototype's public owner-preserving replacement, and
  arms the search record from the action and option owner that will actually
  receive credit. A planned option never starts or switches an option, an
  unavailable proposal can dispatch only the independently safe current owner,
  and a withheld decision is `-1` until a zero-learning retry succeeds. Exact
  arm ownership, inbound decision/action/observation identity, word-pair
  clocks, config/proposal/state/checkpoint bindings, active-option behavior,
  rollback, resources, and eager/JIT/scan behavior are covered. This is an L0
  `not_assessed` policy-mechanism edge with no safety, physical-dispatch,
  empirical-benefit, evidence, promotion, or Alberta Plan completion authority.

- Added an exact post-envelope settlement transaction to the consolidated
  procedural/semantic Prototype compositions. Each exposed action now carries
  a checksum-bound decision, selected primitive, and caller mask. An unchanged
  admitted action is an exact no-op; a changed mask-admitted fallback rebinds
  Prototype's real cached credit owner, cancels only matching procedural and
  partner recommendation owners, and persists the new owner under the same
  mask atomically. No-action, stale, disallowed, corrupt, or partially
  satisfiable settlements leave the complete state unchanged and remain
  retryable. Lower controllers expose exact cancellation without writing
  evidence, counters, reliability, or learner state, and strict checkpoints
  bind the new owner. Cross-layer tests cover accepted, fallback, and no-action
  `EmbodiedSafetyEnvelope` outcomes. The composition still has no physical-
  dispatch, safety-certification, evidence, or promotion authority.

- Added `AuthorizedOptionReplacementController`, a bounded two-phase bridge
  from one externally authorized retirement to one externally authorized
  replacement of the resulting cold STOMP option slot. The wrapper keeps one
  canonical scheduler/installation subtree, projects retirement control from
  it, stages ordinary discovery and incumbent materialization with install
  authority denied, and keeps a fresh replacement bundle transient until a
  separate exact receipt is supplied. Host commit reruns preparation and
  bit-compares the complete transaction before one-slot install/reactivation;
  declined authority commits only the ordinary advance and retry marker, with
  no candidate state or RNG successor persisted. Active-option masked control,
  stale/replayed receipts, capacity/freshness vetoes, forged preparations,
  strict checkpoints, exact resources, and eager/compiled parity at the small
  atomic-adoption boundary are covered. Checksums and receipts are unkeyed
  integrity declarations, not
  caller authentication or cryptographic lineage. This is an L0
  `not_assessed` mechanism with no autonomous retirement, discovery, curation,
  dispatch, safety, evidence, promotion, or Alberta Plan completion authority.

- Added `GroundedImaginationComposition`, a strict L0 planner-to-gauge-to-
  learner transaction. It derives immutable linear policy/value authority from
  the live actor/critic, obtains the rollout batch internally from
  `EnsembleShortRolloutPlanner`, passes that exact value directly into the
  grounded gauge, and commits at most one actor/critic backward pass. There is
  no public rollout-batch input at this boundary. Exact planner,
  authorization, learner, dream-update, and composition clocks either advance
  together or roll back with the planner RNG. Model support, the real anchor,
  regions, safety/protection masks, and environmental truth remain caller
  attestations; this closes tensor-substitution mechanics only, with no
  calibration, dispatch, safety, efficacy, evidence, promotion, or SOTA claim.

- Composed `RTUGenerateAndTest` into the comprehensive-objective
  `PrototypeAgent` adapter under an explicit strict-linear envelope. The
  transaction prepares recurrence without learning or action RNG, learns the
  real transition on the old representation, optionally replaces whole RTU
  units atomically, scrubs the corresponding linear STOMP/OaK base-head,
  intra-option, and option-model axes, and then performs the ordinary next-
  action selection from the recycled representation. Replacement is deferred
  while an option executes. Content-bound prepare/finalize receipts, distinct
  unit and replacement-event clocks, exact builder-revision equations,
  rollback, checkpoints, resources, consecutive replacement, and eager/JIT
  parity are covered. The lower-level finalization receipt proves deterministic
  derivation only: it reconstructs the exact advance receipt, reruns the RTU
  commit, and exact-matches the destination and selected mask, but does not
  authenticate caller-supplied lifecycle, objective-gradient, or ordinary-
  proposal authority. The live adapter closes those three boundaries by owning
  the lifecycle source and constructing the gradient and source-bound proposal
  internally. Its declared worst-case source work is four builder-commit
  evaluations and two RTU-commit evaluations; only one logical ordinary update
  and at most one replacement event persist. Nonlinear STOMP, planning, learned
  world/model/replay,
  dreaming, Horde, IA, partner/memory, GRU, historical candidate-update audit,
  and feature-lifecycle sidecars remain outside this lane. This is L0
  `not_assessed` mechanism integration, not paper-defined delight or an outcome
  result.

- Added a separate matched-initialization v2 Prototype partner-fusion
  development evaluator. Learned-feedback, fixed-zero outcome-blind, and
  empty-message base-only arms begin with bit-identical typed RNG, Prototype,
  fusion, and environment state; only the wrapper intervention differs. A
  paired exogenous context/noise/drift/availability/cost/mask schedule remains
  common while each arm owns its causal trajectory after actions diverge.
  Exact prefix reconstruction, raw hash chains, replay, checkpoint/resume,
  source/runtime/config binding, eager/JIT Prototype parity, and matched
  logical work are covered. On its consumed 12-event descriptive run, learned
  and fixed-zero changed action three times and had equal task/net return,
  while base-only changed none; learned and fixed-zero internal states differed
  without realized behavioral separation. The lane is permanently L0
  `not_assessed`, threshold-free, winner-free, artifact-free, and nonpromoting.

- Added `CChain`, an independent clean-room L0 comparator for the C-CHAIN
  paper equations. The regularizer is exactly one half of mean squared output
  churn between the current parameters and a detached one-step-lag reference
  on a declared disjoint reference batch, with exactly one scalar model output
  per reference sample. Vector-valued per-sample extensions fail closed rather
  than broadening the paper-equation claim. A valid proposal uses one combined
  `jax.value_and_grad`; rejected runtime preflight and commit use none. The
  appendix absolute-loss ratio is implemented with explicit Alberta window,
  warmup, epsilon, and coefficient bounds, and a diagnostic-only empirical-NTK
  helper reports the paper's approximate rank plus diagonal/off-diagonal
  statistics. Exact two-word clocks, rollover-safe ring indexing, transaction
  binding, rollback, resources, checkpoints, and eager/JIT behavior are
  covered. This is not a reproduction of the full sequential PPO/DQN
  algorithm: callable identity, data provenance, and external optimizer
  application remain unauthenticated, and there is no agent integration,
  efficacy, default-selection, evidence, promotion, or SOTA claim.

- Added an explicitly Alberta-derived online Permanent/Transient regression
  learner with independent representations and heads, exact two-word lifetime,
  atomic rollback, fixed 788-byte diagnostic state, no replay or task-boundary
  input, and a machine-readable non-source-faithful departure record. A
  same-state/same-work no-consolidation arm and readout intervention run on the
  already-consumed 512/512/512 A/B/A source. The result is negative for
  retention: permanent A-probe MSE changes from `0.0434621` after A1 to
  `3.92572` immediately after B; its A2 recovery follows further A updates.
  No sweep, threshold, artifact, evidence, promotion, or default claim follows.

- Added a generic fixed-capacity latent-context regression expert bank that
  credits the existing `ContextInference` active-only-freeze law and adds a
  strict predict-before-outcome ownership cache. All expert predictions,
  losses, and analytic candidate gradients are computed on every update, while
  exactly one expert subtree commits; the defaults-off ablation retains the
  current owner with identical state and work. On the same consumed A/B/A
  source, selective routing reactivated the learned A expert after one A2
  outcome, but did not preserve it exactly through B: one B sample selected
  and updated A, changing its probe MSE from `3.73e-20` to `1.90e-05` and its
  subtree hash. Ten A1 and three B ownership switches expose per-sample
  fragmentation. This is a partial/null dormancy diagnostic, not pre-outcome
  context identification, clean retention, a selected default, or evidence.

- Added a fixed-horizon two-event pairwise-dominance quarantine for the latent-
  context expert bank. An opening event can nominate one unique dormant
  challenger that is no worse than the current owner, but commits no expert
  update; a source-bound second event confirms only the persisted no-worse/
  strictly-better relation and otherwise rejects atomically. On the already-
  consumed 512/512/512 A/B/A source, the enabled arm kept the learned A expert
  bit-exact through B, made zero B updates to it, and reactivated it after two
  observed A2 outcomes. Its prequential phase MSEs were
  `0.0156662/0.0145220/0.0174704`, versus
  `0.0156662/0.0994502/0.0754818` for a same-work routing-disabled arm that made
  498 B updates to A. Four openings produced two confirmations and two
  rejections, with four zero-commit quarantine events. This is the first clean
  bounded dormant-expert retention result on this consumed synthetic stream,
  but remains L0 development work: one root, fixed two-event horizon, no
  writer, artifact, threshold, promotion, default, or pre-outcome context
  identification claim.

- Added a unified hidden-context A/B/A life with two complete, independently
  learning `PrototypeAgent`s. Visible rule coordinates are destroyed before
  learner use; after both actions and rewards, each existing two-slot context
  bank may feed its inferred one-hot only to the next decision. Four joint-
  action environment proposals, two context updates, two discarded no-memory
  previews, and two memory-sidecar Prototype candidates are carried by one
  outer all-or-none transaction. On its consumed 1,536-event root, the routed
  arm's A1/B/A2 mean-agent rewards were `0.991401/0.977935/0.964493`, versus
  `0.984638/0.0149503/0.994810` when the same context work was unrouted; routed
  A2 recovered from `0.781546` early to `0.997949` in the tail. Both arms made
  1,536 outer commits with identical 41,718-byte persistent state and logical
  work. These numbers were consumed immediately before the final validation-
  only contract gates. A pure-stdlib pre-run declaration froze the selected
  ten-file direct source manifest, every previously observed literal field,
  all statically derivable fields, and the seven newly added but never-observed
  work counters; its comparator is explicitly partial, nontransitive in source
  coverage, and runtime-unbound. The first replay completed both arms and
  constructed its comparison, but a caller-side compact formatter raised on a
  nonexistent top-level metric and discarded that in-memory result. A separate
  declaration authorized exactly one nonpromoting recovery without changing
  sources or expectations. That recovery had no declared report-field
  mismatches, but `prototype_agent.py` changed from SHA `1e05b1f8...` to
  `37fe39e5...` after clean preflight and before comparison, so the recorded
  outcome is `source-manifest-mismatch`. The recovery budget is consumed; no
  report or artifact was retained and whole-report identity is not claimed.
  This remains a descriptive U0 composition, not a thresholded benefit,
  capacity-pressure, checkpoint, artifact, scientific-evidence, promotion, or
  Alberta Plan completion result.

- Added an isolated routed linear world model whose inputs include the live
  generated pair-feature tail while its outputs remain fixed physical targets:
  stable-base delta, reward, and discount. A pre-outcome prepare/consume
  boundary exact-authenticates the complete source world, live router, base
  observation, action, augmented input, and prediction. Real learning occurs
  under the source bank before input weights and eligibility traces are routed:
  base/action columns and descriptor survivors remain bit-exact, while newborn
  and inactive columns are positive zero. A separately authenticated, defaults-
  off planner re-augments physical predicted successors under the live bank and
  may carry exactly one OaK base-learner backup. Same-clock alternate world,
  OaK, or router histories, stale caches, invalid routes, and non-finite values
  roll back atomically. Focused tests prove a learned survivor column changes
  both the physical proposal and the real OaK backup. This is L0 mechanism
  reachability only—no Prototype integration, calibration, uncertainty,
  retention, benefit, evidence, promotion, or default claim.

- Added a defaults-off, birth-authenticated selective-retention intervention
  to the hidden-rule capacity-pressure dyad. It ranks only valid full-bank
  fresh-allocation victims by completed recurrence within the current semantic
  birth, keeps ordinary stored reuse and free allocation exact, and matches the
  controller-scrub baseline bit-for-bit when no signal is dispatched. On the
  unselected consumed-root epsilon grid, `.05` changed two later evictions,
  avoided eight completed recurrence intervals, and changed reward by
  `+.0024999976`; `.1/.2/.4` were exact nulls. The first common-prefix eviction
  remains unresolved, the signal resets across rebirths, and no default,
  threshold, artifact, evidence, or promotion claim follows.

- Added a matched one-record cross-birth predictive-rescue sidecar to the same
  consumed hidden-rule dyad. It snapshots live rescue priorities before the
  outcome, tests an archived victim model only after a full-bank birth, and
  transfers lineage only when that model's absolute error is strictly below
  both a fresh prior and every live source model. Ties and non-finite values
  abstain; source-victim archival, exact rescue words, controller scrubbing,
  fixed resources/work, and whole-dyad rollback are covered. The intervention
  was an exact null on all four epsilons: zero strict matches, rescue
  increments, eviction changes, or reward deltas. At `.05`, all four valid
  cache tests failed the fresh-prior comparison and all four had an exact tie;
  `.1/.2/.4` never reached a cache-valid test. This falsifies one-transition
  identity rescue on this consumed root and motivates bounded sequential
  evidence without authorizing threshold relaxation, selection, or promotion.

- Added a standalone fixed-`H=2` `SequentialLineageCache` for the next
  cross-birth test without rerunning the consumed dyad. A full-bank birth can
  only freeze one archive candidate, the fresh prior, every live pre-update
  reward model, exact birth identities, and first-event pairwise evidence. The
  next completed transition transfers lineage only if the same birth survives,
  the candidate was never worse across both events, and each eligible
  comparator was beaten strictly at least once. Transfer changes no parameter
  and can influence only future pre-outcome eviction protection. Separate
  32-byte configuration and complete-content SHA-256 tokens, exact two-word
  counters, overlap/saturation handling, atomic rollback, and named resources/
  work are covered by 49 focused tests. At the hidden-rule geometry the state
  is 563 bytes per agent and 1,126 bytes joint. A matched wrapper now composes
  the pair with the hidden-rule dyad: it snapshots rescue and reward banks
  before the outcome, dispatches zero or past-only rescue, stages both
  prioritized context updates, sidecars, authenticated controller scrubs, and
  controller updates, then commits the measured 2,088-byte composite all or
  none. Ten focused tests pass for genesis/resources, exact event binding,
  causal dispatch, tamper rejection, and rollback. The full 4,000-step panel
  remains unexecuted. This is L0 mechanism/integration only: the unkeyed digest
  is not external provenance, core host-transition binding is caller-owned,
  and no matched behavioral outcome, threshold, artifact, evidence, promotion,
  or default claim follows.

- Added a defaults-off paired factorized partner/world planner and a frozen U1
  wrapper around the routed hidden Prototype life. Each agent learns a
  simultaneous partner-action distribution and grounded raw-next-observation,
  reward, and continuation predictions for all ordered joint-action cells,
  then marginalizes one-step expected immediate reward before an optional
  owner-preserving Prototype action replacement. Its planner-only rows are
  `PP`, `M0P1`, `P0M1`, and `MM`, where `M` is the post-memory fallback;
  memory query/write/eviction diagnostics remain present, but same-event
  memory reward effects are not attributed. The wrapper binds row-zero
  outcomes plus routed post-memory observations and carries U0, both planners,
  and both Prototype successors atomically. The planner pair is
  3,758 bytes and adds no post-init planner RNG or replay. The full 1,536-event
  three-arm panel remains unexecuted. On one stable dependency snapshot, 17
  focused core/wrapper cases passed and the sole failure was an over-strong
  whole-composite bit-exact float assertion. A complete 589-leaf comparison
  found 17 inherited/planner float leaves differing by only 1--2 ULP while
  every discrete/key leaf was exact, matching U0's declared
  `rtol=1e-6, atol=1e-7; discrete exact` contract. The corrected parity case
  passed separately, but `prototype_agent.py` changed during that run; there
  is therefore no current-source 18/18 claim. Stable-source verification is
  required before execution. No benefit, safety certification, threshold,
  writer, artifact, evidence, promotion, or Alberta Plan completion claim
  follows.

- Added a staged, nonexecuting HCCL successor design. Its first unimplemented
  causal core persists no-memory (`B`), post-memory (`M`), and planned (`P`)
  receipts; adjacent four-call dyad cubes duplicate `MM` and require exact
  agreement, yielding eight calls and at most seven unique vertices for
  memory, one-step planner, and telescoping attribution under one immutable
  receipt. Fast hidden-sign and slow three-slot context/lineage state,
  post-outcome `H=2` ordering, generated-feature birth routing, typed signals,
  and the outer transaction are specified separately. Communication is
  unavailable/neutral in the core; later planner-input communication would
  need a separately specified boundary, and exact third-layer attribution
  would require 12 calls/at most 10 unique vertices. No seed, implementation, run,
  result, threshold, evidence, or promotion is authorized.

- Added and consumed one frozen three-arm contribution-future-utility panel on
  the existing 8,998-step compositional life. Exact common-root, genesis,
  intervention, clock, state, ranking, prediction, source, runtime, and logical-
  work contracts closed. The disabled internal comparator retained A and
  obtained lifetime executed reward `0.274283`; the mix-one/decay-zero and
  mix-one/decay-0.95 endpoints retained no A/B/C and obtained `-0.003112` and
  `-0.020449`. The report hash is
  `8666ac91010dff368aa3653f69507256b50784fd0d9126a76a5641d91ff07ec0`.
  This is a consumed L0 rejection of both tested enabled formulations, with no
  selected default, retry, threshold change, artifact, evidence, or promotion.

- Added and consumed a five-arm contribution-future-utility calibration v2 on a
  new development root. Its 8,998-step rotated A/B/A/D/A/C/A/B/C/A stream and
  281 curation opportunities compare current utility, full future utility, a
  half mix, uncertainty/age normalization, and a longer float32 horizon. The
  planned report would have recorded active and candidate direct/augmented
  tie-aware f32 ranks,
  structural recurrence/occupancy, admissions, root/cascade losses,
  coexistence, and final targets from bit-identical 2,072-byte geneses. Public
  validation and serialization use only a successfully completed cached value
  and cannot start or wait for the one-shot builder. Work claims are limited
  to shared-base calls, shapes, and update opportunities; intervention-specific
  and behavior-dependent work is explicitly unequal or unclaimed. Private
  report/arm helpers now require the latch's live per-attempt capability. A
  pure-stdlib external declaration binds the six selected sources, protocol,
  keys, stream, one-attempt chronology, clean postflight, summary-first output,
  and every no-authority flag. The sole attempt completed the first arm's
  compiled scan, then failed before returning an arm record: the extractor
  incorrectly required the all-step `decision_margin_passed` diagnostic to be
  false outside 32-step curation opportunities. No report, endpoint, artifact,
  winner/default, threshold, evidence, or promotion result exists, and no
  retry/recovery is authorized. The current evaluator rejects before source
  construction and the historical declaration is deliberately source-invalid.
  A failing-test-first synthetic trace now separates all-step margin
  diagnostics from due-opportunity endpoint counts for future roots only; it
  cannot revive or reinterpret v2.
  Eighteen evaluator, six decommissioned-declaration, and three outcome tests,
  Ruff, and strict mypy pass.

- Added a source-bound prepare/commit boundary to the standalone frozen-theta
  compositional-feature adapter. An outer agent can prepare one generated-bank
  successor, bit-authenticate and re-encode persisted rows, then defer adoption
  until every consumer reports ready. Commit recomputes and authenticates the
  JAX-authoritative learning/binding leaves; stale retries, tampered
  payloads/state, failed row routes, invalid inputs, and exhausted identities
  leave the source unchanged. Legacy host timing floats retain their existing
  non-bit-exact outer-JIT boundary. Exact
  persistent/transient bytes, two logical learner evaluations per committed
  outer transaction, and eager/JIT/scan behavior are covered. A new isolated
  consumer router binds exact linear OaK plus optional ordered Horde state,
  authenticates stable/post-update clocks and caches, and scrubs every changed-
  birth consumer axis while preserving survivor bits. One captured dynamic
  curation-permission byte lets unsafe due opportunities be consumed without a
  structural proposal while ordinary learning and cadence advance. The caller
  must derive that safety bit. A separate development integration now performs
  real public OaK and Horde updates under the old bank, consumes an unsafe due
  opportunity while both learners advance, then commits one birth at the next
  safe primitive boundary with exact survivor preservation and changed-axis
  scrubbing. This is still a two-transition linear test rather than a live
  Prototype loop, and supplies no benefit or evidence claim.

- Added topology-headroom and left-pack siblings to the consumed 8,998-step
  scaffold-free compositional-control diagnostic. Headroom repairs A/B/C
  admission and raises greedy reward from `-0.01045` to `0.12247`, but loses
  every recurring target and never retires obsolete `p12`. Left-packing retires
  `p12` yet never co-retains two A/B/C targets, ends with none, and lowers reward
  to `0.03979`. These are frozen L0 negative diagnostics: proposal reachability,
  slot capacity, and placement do not establish selective retention, general
  feature finding, a default policy, evidence, or promotion.

- Hardened `FastSlowLearner` with an authoritative big-endian `uint32[2]`
  lifetime, saturating int32 telemetry, terminal fail-stop, whole-update numeric
  rollback, strict v2 config/state/result/resource records, exact bytes, and
  eager/JIT/scan diagnostics. Its consumed 512/512/512 A/B/A diagnostic remains
  negative for retention: slow-path A-probe MSE rises from `0.0188931` after A1
  to `4.01919` after B (`212.73x`), while the low A2 tail follows new A updates
  and is relearning. The exact clock adds eight state bytes (`1296 -> 1304`);
  no Permanent/Transient, retention, default, evidence, or promotion claim is
  made.

- Added two deliberately separate first-order adaptive UPGD APIs in
  `core/canonical_upgd.py`. `OfficialAdaUPGD` is source-bound to released
  commit `b75e90ad4b09c28971ac9dbb902a8fd86709b28c`,
  `core/run/rl/adaupgd.py`, and preserves its bias-corrected first/second
  moments, raw-utility maximum, outside-denominator noise, two-alpha direction,
  one-alpha decay, and numeric quirks. `AlbertaAdaUPGD` remains an explicitly
  derived guarded extension. Fixed-noise/JIT/scan/checkpoint/resource/public
  contracts are covered without an efficacy, default-selection, or SOTA claim.

- Added a versioned in-memory WP1 continual-control development report that
  runs `PrototypeAgent`, a running-reward bandit, and a frozen-action baseline
  over consumed seeds 1701/1702 with independent functional environments.
  Reconstructing evaluator reports, raw action/decision ownership, exact
  opportunities and logical state bytes, deterministic logical latency,
  parameter/policy/value churn, explicit three-state diagnostic applicability,
  source/runtime replay, and checkpoint continuation are validated. The report
  writes no artifact and is always `not_assessed`; unavailable host/hardware
  and internal-gradient measurements, disabled-component inapplicability,
  unmatched realized compute, and consumed seeds preclude efficacy or evidence.

- Added `BalancedStateObjectives`, an L0 learned-state kernel with separately
  updated linear GVF heads at multiple strictly ordered discounts and a
  consecutive-pair inverse-action head. The GVF family is averaged before
  fixed positive group masses are combined; clipped current/successor gradients
  are bound to an exact executed-action receipt and caller representation
  revisions. Strict checkpoints, exact resources, fail-stop clocks, retry-safe
  rejection, JIT/scan, and an online-gated builder commit witness are covered.
  An opt-in `PrototypeBalancedStateObjectives` adapter now authenticates the
  exact Prototype decision/action and representation owner, scores bootstrap
  state before autoreset, combines both recurrent-sensitivity gradients into
  one clipped builder commit, and atomically rolls back the complete
  composition on rejection. The base Prototype path remains unchanged. The
  objective set remains incomplete, weights are not empirically calibrated,
  and no retention/control/Forager claim follows.

- Added `LearnableGRUStateBuilder`, a conventional dense update/reset/candidate
  GRU with exact fixed-parameter RTRL sensitivities and the existing
  source-bound proposal/advanced-destination commit contract. Factory,
  configuration, reset, fail-stop clocks, exact resources, checkpoints, and
  eager/JIT/scan behavior support the full GRU. Its `O(H * P)` sensitivity
  storage and approximate carry after online parameter changes are explicit;
  no learned-state outcome claim follows.
  The development-only write/hold probe includes the full GRU and compressed
  RTU. Its consumed four-seed default run gave mean accuracy observation
  `0.5158`, fixed trace `0.5292`, diagonal gate `0.5258`, full GRU `0.5067`,
  and RTU `0.5617`; the arms are resource-unmatched and no artifact or
  promotion claim is created.

- Added `RecurrentTraceUnitStateBuilder`, a diagonal complex RTU builder that
  retains compressed unit-diagonal sensitivities instead of a dense
  hidden-by-parameter Jacobian. Fixed-parameter RTRL is exact; default carry
  after online commits is documented as approximate. An optional diagonal
  Taylor correction owns the exact source parameters, actual accumulated
  delta, and source update words while retaining its mixed-Hessian limitation.
  Causal events, source-bound proposals, advanced-destination commits,
  fail-stop clocks, reset, resources, checkpoints, factories, eager/JIT/scan,
  public exports, ordinary Prototype operation, and comprehensive current plus
  successor objective composition are covered. The consumed five-arm
  write/hold probe gave mean accuracy RTU `0.5617` versus fixed trace `0.5292`
  and full GRU `0.5067`, with RTU total persistent state `1,324` bytes versus
  `12,204` for full GRU. This resource-unmatched supervised-development result
  creates no control, efficacy, evidence, promotion, or SOTA claim.

- Added `RTUGenerateAndTest`, a standalone fixed-capacity causal recycler for
  `RecurrentTraceUnitStateBuilder`. It observes pre-update real/imaginary
  activation-gradient contribution, maintains per-unit utility/age/support,
  and replaces a stable fixed quota only after exact period, warmup, maturity,
  evidence, and protected-unit gates. Replacement redraws polar recurrence and
  both input rows and scrubs activation, compressed RTRL sensitivity, and
  optional Taylor trace/source/delta slices without changing survivor bits.
  Optional ordinary RTU learning is recomputed from the exact source and binds
  the only accepted advanced destination; stale/tampered proposals, invented
  destinations, numeric failure, and clock exhaustion roll both states back.
  Typed Threefry ownership, exact resources/checkpoints, and eager/JIT/scan
  parity are covered. The standalone surface is L0 `not_assessed`
  sensitivity/recycling machinery—not paper-defined delight or evidence; the
  later strict-linear Prototype consumer described above does not change that
  outcome status.

- Added `ComprehensiveStateObjectives`, completing the requested L0
  auxiliary-head surface with action-conditional next-observation/latent,
  reward, stable Bernoulli termination, multiple-timescale GVF, state-value,
  selected-action-advantage, and inverse-action heads. Prediction/control
  subheads are mean-balanced inside fixed family masses; each head has its own
  parameters, step size, and exact revision row. Exact action/revision
  receipts, clipped current/successor representation gradients, numerical
  rollback, strict checkpoints/resources, finite-difference checks, and
  eager/JIT/scan parity are covered. An opt-in
  `PrototypeComprehensiveStateObjectives` transaction binds caller target
  bits/source/provenance to the exact dispatched decision/action, observation
  event, final/bootstrap observation, and online-gated or full-GRU builder
  owner. It sums both RTRL pullbacks into one clipped commit and atomically
  restores the entire composition on rejection. Targets and family masses
  remain uncalibrated, with no retention/control/Forager outcome claim.

- Added isolated `RealStateOneStepDyna` for guarded one-step ensemble planning.
  Exact real anchors bind decision-time ownership; planning accepts only exact
  current model/control states that advanced monotonically through real updates
  and forms `reward + continuation * max Q` targets before synthetic updates.
  Support, residual, disagreement, finite-value, and termination-agreement
  vetoes, zero synthetic traces, hidden-utility isolation, disjoint RNG/clocks,
  strict checkpoints, and resource scopes are tested. The lane is not wired to
  Prototype, its gates are uncalibrated, and no planning-benefit result exists.

- Added proposal-only `EnsembleShortRolloutPlanner` for policy-directed and
  max-epistemic fixed-horizon imagination. Exact real anchors bind immutable
  policy/value arrays and every ensemble-state word to revision/content
  receipts. Per-step support, residual, epistemic, finite-value, and
  termination-agreement guards produce terminal-correct reverse returns;
  learned termination never bootstraps. Model/policy/value owners stay
  read-only, clocks/RNG/checkpoints/resources are isolated, and stale or
  same-revision content aliases fail atomically. This is uncalibrated L0
  proposal machinery with no agent training consumer or control-benefit claim.

- Added `ImaginedRolloutSelectionGauge` and
  `AuthorizedImaginedRolloutActorCritic` as the isolated WP4.6 consumer
  boundary. A frozen source/model generation feeds a causal fixed-capacity
  action×region audit; noncompensating evidence, prediction, termination,
  success-LCB, purity, safety, and protected-mask gates issue full-content
  authorization receipts only for a transition-distinct candidate proposal.
  Admission is path-prefix closed. Proposals perform zero autodiff; commit
  revalidates the complete source and receipt before exactly one guarded
  fixed-shape actor/critic backward pass, while failed preflights perform zero.
  Targets are terminal-correct and dream imitation uses graded positive
  advantage. The competent-real cloning control uses the same prefix-closed
  transition/update bounds. Unkeyed tags cover post-mint integrity without
  authenticating planner issuance or caller-declared competence.
  Checkpoints, resources, replay/staleness rejection, and eager/JIT/scan
  contracts are covered at L0 without calibration, Prototype/dispatch
  integration, outcome, evidence, promotion, or SOTA claims.

- Added `ProspectiveExploration`, a fixed-budget L0 prospective exploration
  selector using expected improvement times capped host-relative surprisal,
  with random, epsilon-greedy, ensemble-disagreement, information-gain, and
  learning-progress comparators. Exact causal ownership receipts fail closed;
  candidate ranking precedes a caller-owned hard shield and separately shielded
  host fallback. Typed RNG, clocks, checkpoints, resources, and eager/JIT/scan
  parity are covered. Supplied scores and Boolean shielding remain uncalibrated,
  and synthetic diagnostics establish no exploration or physical-safety benefit.
  A separate consumed eight-event development evaluator now closes score
  production only in a tiny stochastic world: each of the six arms owns an
  independent action-conditioned linear-TD ensemble and derives its scores
  solely from its executed history; only exogenous noise is paired. The world
  includes progress-resetting noisy TV and delayed invest/collect behavior,
  while the caller hard mask remains the actual admissibility owner. Exact
  causal replay, hash chains, in-memory resume, and matched logical budgets are
  validated under a v2 trace. Its expected-improvement-times-surprisal score is
  explicitly not DG/Kondo delight and no actor backward executes. Historical
  v1 `DelightfulExploration` import/config spellings are compatibility-only;
  v1 checkpoints remain fail-closed. The report remains `not_assessed`,
  threshold-free, winner-free, artifact-free, and nonpromoting.

- Added a bounded L0 `SelfNormalizedResets` baseline for one fixed-width dense
  ReLU layer. Per-unit exact ages and a fixed completed-gap ring estimate a
  positive-support geometric firing law; stable `log1p(-p)` evaluates the
  observed silent-run tail `P(A >= age + 1) = (1 - p)^age` with inclusive
  rejection, exact history, and post-reset warmup boundaries. The caller
  optimizer step precedes deterministic capped resets: incoming columns/biases
  are refreshed, outgoing rows are zeroed, and supported Adam moments are
  cleared. Exact long clocks, typed Threefry ownership, atomic rejection,
  resources, source/representation binding, checkpoint/rebind, realistic reset
  persistence, and eager/JIT/scan parity are tested. The serialized
  positive-support/window convention is not claimed bit-equivalent to the
  authors' silent-age histogram code, and no plasticity, retention, default,
  evidence, promotion, or WP2-exit result is claimed.

- Added a development-only optimization-centric plasticity diagnostic. Exact
  shared-initialization ordinary-SGD and initialization-centred L2 arms process
  one evaluator-owned A/B/A nonlinear stream. Both switches retain raw
  old/incoming gradients, alignment, two-sided local loss probes, separate
  dormancy, and phase parameter displacement/churn under strict
  source/runtime/config/replay and logical-resource accounting. The fixed
  descriptive rule owns no threshold, output, evidence, promotion, OCP, or L2
  benefit claim.

- Added a development-only `PrototypeFeatureMemory` adapter and its opt-in
  `PrototypeAgent` composition for the bounded pair-feature lifecycle plus
  `ExperientialMemory`. The lane requires the exact
  `IdentityStateBuilderConfig` so every stored base prefix remains
  reconstructable. On an accepted descriptor-generation successor, all valid
  observation, key, and outcome rows are
  rebound to the destination feature bank before the same transition's query
  and write; corruption or a failed rebind rolls back the whole Prototype
  transition. The adapter declares fixed persistent bytes and a worst-case
  rebind bound of `capacity` rows and `2 * active_pair_slots * capacity` pair
  products, with zero memory-clock advance and zero RNG draws. A v16 Prototype
  checkpoint binds the adapter's exact composition digest. Learned base-state
  builders and generated-pair-tail modeling remain unsupported; the only
  admitted model path is the separate stable-base legacy lane below, while
  dreaming, replay/ensemble/recurrent models, IA, and partner fusion remain
  unsupported. This is L0 mechanism coverage with no retention benefit,
  scientific evidence, promotion, WP8, or Alberta Plan completion claim.

- Added a development-only stable-base composition between an exact
  `PrototypeFeatureLifecycleConfig`, exact `IdentityStateBuilderConfig`, and
  the legacy `ActionConditionedWorldModel`, requiring the exact
  `ActionConditionedWorldModelConfig` type. The model, recent-observation
  buffer, and action-interaction features consume only the stable base prefix;
  the generated pair tail is not modeled. The v17 state wrapper carries a
  digest of the complete serialized Prototype config, validation binds every
  LMS optimizer scalar to the configured step size, and buffer occupancy/index
  are derived exactly from the observation lifetime. Accepted curation leaves
  model coordinates stable; feature-lane model refusal or feature rejection
  rolls back the complete model/buffer/Prototype event atomically, while the
  historical direct-world lane remains best-effort. The focused mechanism
  suite passed 9/9 in 42.36 seconds and the Prototype horizon suite passed
  17/17 in 25.72 seconds. Dreaming, replay, ensemble and recurrent models, IA,
  and partner fusion remain unsupported. This is L0 nonpromoting mechanism
  coverage, not model quality, planning benefit, retention, evidence,
  promotion, or an Alberta Plan completion result.

- Added a bounded, development-only Prototype feature-memory recurrence
  harness. Its declared `3 x 512` visible-cue meet/avoid/meet life composes
  linear OaK, a managed linear Horde, the pair-feature lifecycle, and
  feature-bound experiential memory with a stable-base linear
  `ActionConditionedWorldModel`. The model sees only the base prefix, enables
  action interactions, uses `gamma=1.0` to match the continuing discount, and
  owns a capacity-one anchor buffer. Matched readout, promotion, joint, and
  cue-masked arms each pay for one discarded no-memory preview plus one
  committed model update per event. Two additional defaults-off conservative
  outcome-gated arms require exact neighbor support/weight mass and strict
  immediate-reward advantage before memory may change dispatch. Strict
  in-memory reports reconstruct
  per-event world predictions and per-phase A → B → A prequential model error,
  model recurrence/reacquisition metrics, exact model/buffer resources under
  `world_model_bundle_nbytes`, matched work, clocks, and causal replay. The
  consumed seed-0 visible gate allowed 59 of 1,536 proposals (52 helpful, 7
  harmful; cumulative dispatch delta `+5.0654`); the cue-masked gate allowed
  182 (142 helpful, 40 harmful; delta `+8.1594`) and reversed the ungated
  arm's net-harm diagnosis. Each gate has zero persistent state and RNG, while
  each gated composition remains 20,733 bytes. Generated pair tails remain
  unmodeled and the harness performs no dreaming or planning. The partner is
  scripted, the primary cue is visible, and immediate reward is associational
  rather than delayed or causal credit. There is no accepted default-life
  result, artifact writer, threshold, held-out panel, scientific evidence, or
  promotion path. This remains L0 nonpromoting mechanism/development evidence,
  not general retention or transfer evidence.

- Added a consumed-root hidden-rule capacity-pressure development lane with
  two independent differential-SARSA agents, two independent three-slot
  context banks, four recurring conventions, exact joint transactions, and
  semantic `(agent namespace, birth words)` identities that never equate
  recyclable slot indices. The fixed epsilon grid selects no winner; at
  `epsilon=.2` both agents retained distinct recurring A/B/C births, while
  higher exploration reduced reward. A matched post-audit intervention
  authenticates each newly allocated birth and scrubs exactly its stale Q and
  eligibility-trace column before next-action scoring. It removed all observed
  cross-birth consumption and changed consumed-root overall reward by
  `+.01025`, `+.01200`, `+.00100`, and `+.00425` across the four epsilons. An
  executable common-prefix twin makes future B versus D recurrence flip the
  zero-loss eviction, establishing that no deterministic past-only policy can
  guarantee hindsight-perfect forgetting without a prior. These are
  threshold-free, artifact-free L0 development results, not a selected default
  or population claim.

- Hardened the foundational `LinearLearner`, `MLPLearner`, `TDLinearLearner`,
  and `TrueOnlineTDLearner` families with authoritative two-word update clocks,
  saturating int32 telemetry, terminal fail-stop, and whole-update rollback for
  invalid input or non-finite candidates. Strict v2 config/state schemas,
  explicit representable-only legacy migrations, and exact fixed resource
  records cover the four families. These are lifetime/transaction mechanisms,
  not convergence, retention, performance, evidence, or completion results.

- Hardened `LearningPartnerWorld` with an authoritative two-word lifetime
  identity, exact phase/cycle arithmetic, strict v2 config/state/input/output
  schemas plus explicit legacy migration, atomic state/RNG rollback, and fixed
  resource accounting. The exact clock adds 8 bytes to the world state
  (`24 -> 32`) and the downstream hidden-learning-partner composition is now
  exactly 321 bytes rather than 313. This is bookkeeping and mechanism
  coverage only, with no coadaptation, IA benefit, or scientific evidence
  claim.

- Hardened `PartialObservationWrapper` with its own exact two-word event
  identity, saturating telemetry, exact periodic-mask indexing, strict v2
  config/state/resource contracts, and explicit legacy migration. Child
  output/state validation and wrapper clock/RNG updates now commit or roll back
  atomically; child-owned bytes and unstated child semantic invariants remain
  explicitly outside the wrapper contract. This supplies no learning or
  scientific evidence claim.

- Hardened both adaptive-opponent streams with exact two-word lifetime clocks,
  strict v2 schemas and migrations, fail-closed atomic transitions, and fixed
  resource declarations. Adversarial pursuit now arms one observation through
  `emit_result` and requires its exact owner identity at `resolve_result`, so
  duplicate emission, duplicate resolution, and stale resolution are rejected
  without advancing the life. Tuple wrappers remain for compatibility. These
  are finite mechanism contracts, not performance or scientific evidence.

- Hardened rollout-level dreaming with an authoritative `uint32[2]` clock and
  saturating int32 telemetry. `DreamRolloutStepResult` exposes exact pre/post
  identities and commit diagnostics; corrupt state, invalid predictions, and
  the terminal all-ones identity fail-stop with atomic state/RNG rollback.
  Strict v2 config/state/result/resource schemas, representable-only explicit
  legacy migration, and fixed state plus per-rollout work accounting add 8
  exact-clock bytes to the prior state and no persistent capacity growth.
  Eager, JIT, and JIT-disabled tests cover these mechanics. They establish no
  dream quality, planning benefit, whole-Prototype composition, artifact,
  evidence, promotion, or Alberta Plan completion claim.

- Added opt-in live consolidated-memory consumers for Prototype. The
  procedural composition settles exact pending skill feedback, preserves
  learning credit for the primitive that actually executed, then queries the
  shared store and may replace the next cached primitive after experiential
  memory and partner fusion under the intersection of every hard-safety mask.
  A separately versioned semantic composition uses the same controller state,
  performs a pre-write semantic query/current-record write, and appends the
  accepted prior payload—or an exact zero tail—to the ordinary next Prototype
  context before the procedural query. Exact decision, lifecycle,
  request/record, provenance, generation, kind, revision, and upstream-mask
  bindings fail closed. Exhausted memory becomes a no-op while valid base
  control continues. These are L0 integrations with no physical-dispatch,
  transfer, safety, efficacy, evidence, or promotion claim.

- Added the opt-in L0 `CumulantOptionInstallation` discovery-to-live-STOMP
  composition. It accepts only a complete fresh source/canonical/transition-
  bound four-family proposal, binds descriptor identities into preallocated
  option slots, and rematerializes their cumulants on each live observation.
  A behavior-eligibility mask now reaches STOMP action selection, real TD
  bootstraps, skip diagnostics, option-model planning selection, and planning
  bootstraps, so cold slots cannot act or influence learning. Quiescent public
  lifecycle rebinding preserves identical semantics and resets every changed
  slot-local policy/model/trace/optimizer/base-head value with a fresh caller
  key. Active option/comparator cutovers are exact no-ops and require later
  fresh re-proposal; capacity freezes installation only, not valid installed
  control. A bounded `CumulantOptionScheduler` now observes discovery every
  accepted transition, requests fresh bundles at exact cadence/retry, requires
  strictly newer caller authority at quiescent installation, advances its key
  on every applied install, and emits authority-free retirement handoffs.
  Attempt exhaustion is explicit and deferred payloads are never queued. The
  composition is host-orchestrated and `not_assessed`, with no autonomous
  go/no-go/retirement policy, benefit, evidence, promotion, or WP7-completion
  claim.

- Added `AuthorizedOptionRetirementController` for strict WP7.3 retirement
  execution. It binds one scheduler handoff to a distinct caller receipt over
  exact slots, owners, source/representation, descriptor identities and
  generations, lifecycle/audit/controller revisions, validity window, and two
  independent Threefry reset keys. A fixed noncompensating live-audit policy
  requires per-context support and no positive randomized primitive margin,
  then an explicit reliability/model/planning/redundancy concern. Two compiled
  public lifecycle rebinds scrub the complete approved slot and restore its
  installer semantic identity before a persistent mask leaves it cold across
  behavior, bootstrap, planning, and attribution. Active options, replayed or
  stale authority, either reset failure, numeric corruption, and exhausted
  clocks roll back atomically. No proposal or replacement is queued. Strict
  checkpoints/resources and eager/JIT/scan contracts are covered; this remains
  L0 `not_assessed` with no autonomous authority, outcome, evidence,
  promotion, safety, WP7-completion, or SOTA claim.

- Added the public L0 `EmbodiedSafetyEnvelope` hard-command boundary. It checks
  measured and proposed joint, workspace, collision, timing, bridge, identity,
  and version contracts before returning a proposal, a statically configured
  fallback, or no available action; it never dispatches. Emergency stop now
  latches through an independent checksummed transition even when replay,
  exhausted decision capacity, or invalid optional metadata rejects the
  command. Authority-bound reset needs a strictly newer stationary-safe sample
  and external caller authentication; rollback preserves diagnostics while
  suspending deployment. Restore requires an exact externally retained
  revision plus SHA-256 state anchor, preventing an old snapshot from erasing a
  stop or consumed nonce. A fixed shadow ring and proposal-only
  Wilson/calibration/latency/hard-violation readout have zero deployment or
  promotion authority. This is mechanism coverage, not a geometry proof,
  physical-safety result, robot-simulation result, or deployment claim.

- Added a strict development-only synthetic embodied fault-injection audit.
  One fixed 30-event continuing schedule covers observation/wear drift, timing
  and delayed-reward metadata faults, sensor corruption/failure, bridge loss,
  unsafe candidates, emergency stop, reset, rollback, and exact checkpoint
  recovery. Only envelope-available commands are counted as simulated
  executions and physical dispatch remains zero. Complete causal records,
  shadow/readiness facts whose success input is only action availability, hard
  interventions, fallback identity, envelope
  action-availability recovery delays, eager/JIT/scan parity, externally
  anchored resume, and exact replay are retained. The schedule is not a
  dynamics simulator or geometry proof; the unchanged opaque controller does
  not test learner adaptation, caller authentication remains external, and the
  held-out family is unexecuted. With no seeds, thresholds, artifacts,
  efficacy/safety verdict, or deployment authority, it remains `not_assessed`.

- Added a strict development-only embodied dynamics/adaptation evaluator. A
  minimal adaptive `PrototypeAgent` and capacity/update-opportunity-matched
  zero-learning control own independent bounded two-joint plants across A/B/A
  and a separately declared consumed change family. Typed common randomness is
  exogenous-only. Every primitive command crosses `EmbodiedSafetyEnvelope`,
  changed fallbacks rebind the public Prototype credit owner, and unavailable
  actions produce no simulated command or learner transition. The lane retains
  exact commands/IDs/transitions/revisions, drift/intervention/recovery traces,
  fixed resources, pure dynamics parity, full composite checkpoint resume, and
  causal replay. It writes and dispatches nothing, has no untouched held-out
  data or thresholds, makes no efficacy/safety claim, and remains
  `not_assessed` with zero deployment/evidence/promotion authority.

- Added a strict development-only `PartnerPolicyFusion` stress lane. Learned,
  outcome-blinded, and base-only arms share one frozen 96-event two-context
  stream, identical fixed message/state shapes, and one decision plus feedback
  kernel call per event. The stream reverses partner utility halfway through
  and exposes communication costs and spikes, partner-specific and total
  disconnects, and hard-mask exclusions in a complete causal trace. Source and
  runtime bindings, deterministic full replay, JSON checkpoint transport, and
  prefix resume fail closed. Every result remains `not_assessed`; there is no
  threshold, calibration, closed-loop Prototype benefit, output writer, or
  promotion authority.

- Added a consumed 12-execution causal closed-loop partner-fusion evaluator
  around the real `PrototypeAgent`. Learned-feedback, fixed-zero outcome-blind,
  and empty-message base-only arms own separate learner, fusion, environment,
  authority-receipt, and hash-chain state while sharing only frozen exogenous
  context, noise, drift, availability, costs, and mask candidates. The lane
  exercises endogenous observations/rewards/messages/feedback, an uncued
  reliability reversal, disconnects, costs, real action changes, caller-owned
  masks, exact replay, in-memory resume, update-boundary eager/JIT parity, and
  matched logical budgets. Host-only learner birth/uptime leaves are
  canonicalized to exact float32 zero before hashing, with noncanonical timing
  state rejected, so long-running wall-clock bins cannot perturb replay. It
  writes no artifact and remains consumed L0
  `not_assessed` instrumentation; independent learner RNG/state, the short
  life, and absent thresholds preclude a causal benefit or WP8 claim.

- Audited the historical IPMNIST screening record fail-closed. All 144
  screening shards and 69 confirmation shards parse and their reported means
  are numerically reconstructable, but the checked-in proxy receipt rejects
  all three UPGD prefix comparisons, `summary.json` covers only 132 shards,
  and the round-2 driver cannot read the shard schema or reproduce its later
  result file. The project-level status now treats the unbound v1 shards as
  hypothesis-generating development observations, not a completed or
  current-source-authenticated campaign; stored output bytes are unchanged.

- Frontier sigma0-extension wave (development screening lane, nonpromoting):
  `sigma0_ndecay099` — the perturbation-free normalized-gated champion with
  the input normalizer's EMA decay dropped 0.999 → 0.99 — confirmed at the
  full 200-task protocol at **0.86245 ± 0.00034** (seeds 0-2), the campaign's
  best number, +0.0088 over the 10-seed `upgd_ema_norm` champion at ~1/7 the
  compute. The extension star is symmetric (slower decay −0.0073, hidden-RMS
  norm −0.0186, epsilon/gate-temperature/local-gate flat), pinning the
  dominant mechanism to input-statistics tracking speed. Round-2 arms
  (`sigma0_ndecay09/095/098`, `ema_norm_ndecay099` noisy transplant) added to
  the screening registry (103 tests).

- Added the L0 shared control/linear-Horde extension to the bounded Prototype
  pair-feature lifecycle. Channel zero is the owner-bound control target and
  later channels are `HordeUpdateResult.td_targets` in declared demon order;
  linear OaK and Horde first update under the old bank, then a committed bank
  change routes their post-update feature axes atomically in exactly two
  router calls. Scale-normalized proxy utility gives control `0.5` and each
  of `D` demons `0.5/D`, making the Horde group `0.5` in aggregate. An atomic
  OaK/Horde/binding bundle and ordered-schema digest require the v4 Prototype
  checkpoint, while an exhausted lifecycle is an audited no-op that preserves
  already-advanced, step-aligned consumers. The lane fails closed for
  nonlinear OaK/Horde state, non-LMS Horde optimizers, or a Horde normalizer.
  Resource accounting is exact for lifecycle-owned state and routed consumer
  axes, not the complete caller-owned Prototype/OaK/Horde footprint.
  This is a scale/group-balanced shadow-prediction proxy, not causal deletion
  benefit, empirical benefit, WP7 completion, promotion, or L3 evidence; it
  does not renew any registered artifact whose source hashes differ.
- Added the opt-in WP7.1b `PrototypeFeatureUtilityAuditor` on that exact shared
  lane. It forms each active-feature score from the old descriptor bank's
  frozen, predict-before-update control/Horde targets, predictions, and linear
  tail weights: the intervention is the exact normalized one-step
  half-squared-loss increase from deleting that contribution. A separate
  matched shadow-candidate cohort uses its own normalized-LMS contributions
  to score insertion loss reduction before the shadow weights, utility EMAs,
  or scale moments update; the active and candidate cohorts are not a router
  ranking. Task mass stays fixed at `0.5` for control and `0.5/D` for each of
  `D` ordered demons. After a committed two-call lifecycle route, the auditor
  explicitly rebinds its private state by descriptor identity without making
  another router call. Enabling the auditor wraps the existing consumer bundle
  and audit state in a nested atomic bundle serialized only under the v5
  Prototype checkpoint schema; with the auditor disabled, the v4 bundle and
  behavior are unchanged. This is bounded L0 diagnostic
  instrumentation with no curation authority, empirical return or benefit
  claim, promotion authority, WP7 completion, or evidence-artifact renewal.
- Completed the WP7.1c L0 mechanism with an opt-in, stateless
  `PrototypeFeatureUtilityCurationPolicy`. It converts the auditor's
  post-observation feature deletion/insertion sensitivity into ranking
  influence only. This surface neither scores actor samples nor selects a
  backward pass and therefore does not use the paper's “sparks joy” terminology.
  Lower deletion utility ranks active slots within the
  active cohort, while higher insertion utility ranks candidates within the
  candidate cohort. The two cohorts are never compared. Every configured task
  must meet the per-slot evidence floor, and the fixed control/Horde task mass
  is never renormalized. Existing ages, maintenance cadence, candidate
  confirmation, proxy promotion floor and margin, and safe routing retain all
  promotion and go/no-go authority. The exact v6 Prototype checkpoint shell
  binds this configuration and digest around the v5 utility/consumer bundle;
  disabling it leaves v5 behavior unchanged. The adapter owns no persistent
  state and adds no RNG draw, backward pass, consumer update, or router call.
  This establishes no empirical benefit, evidence renewal, scientific
  promotion, WP7 completion, or Alberta Plan completion.
- Added the standalone WP7.2 v1 `CumulantSubtaskDiscovery` L0 mechanism. Its
  fixed candidate universe spans controllable events, feature changes,
  reward-relevant transition atoms, and typed prediction bottlenecks. A
  two-phase, forward-shifted `arm`/`observe` transaction prevents an atom born
  from the current outcome from scoring that same transition. Learnability,
  randomized-propensity controllability, incumbent/selected-proposal novelty,
  and frozen pre-update reward/model insertion contribution are
  noncompensating gates; prediction-bottleneck candidates additionally require
  epistemic and progress floors and pass a persistent running-mean aleatoric
  veto. Fixed positive per-family quotas admit only a complete discovered
  bundle. A once-sampled frozen random-projection cohort and an identity-bound
  hand-authored cohort each use the same exact option budget `B`; all three
  materialize into compact tail slots rather than candidate IDs. Strict v1
  schemas, transition/source/canonical bindings, tamper checks, static
  ceilings, and an exact logical resource declaration bound the mechanism. It
  invokes neither Kondo nor delight and declares zero backward passes, consumer updates,
  router calls, Horde updates, and option updates. It does not mutate OaK,
  STOMP, Prototype, or Horde and has no curation, promotion, go/no-go, or
  scientific-promotion authority. The tests establish proposal mechanics and
  a fresh one-update STOMP consumer smoke check only—not empirical benefit,
  option discovery/lifecycle, a WP7 exit, evidence promotion, or Alberta Plan
  completion.
- Added the standalone WP7.3 v1 `OptionLifecycleAudit` L0 mechanism. Exact
  two-phase transactions bind source, representation, transition, the complete
  option semantic/generation set, state revision/checksum, initiation context,
  randomized primitive-comparator assignment, and the frozen pre-update option
  model signature. It separately accounts for per-context initiation coverage,
  natural completion, goal, timeout, environment termination, censoring,
  STOMP-compatible discounted external return, pseudo-return, model error,
  planning use, redundancy, compute, and memory. Comparator evidence retains
  treatment and primitive floors in every context and reports a fixed-equal-
  context-mass margin rather than dropping missing contexts. Semantic rebinding
  preserves history only for a bit-identical option and otherwise advances the
  generation and resets all slot-local audit state; in-flight transactions
  defer replacement. Maintenance emits bounded proposals only, with zero RNG,
  backward passes, or consumer updates. The standalone audit core alone is not
  a STOMP/OaK composition, automatic replacement, empirical option benefit,
  WP7 completion, promotion, or L3 evidence.
- Added the opt-in persistent `STOMPOptionLifecycle` observer. Actual STOMP
  ownership, starts, natural goal/timeout/environment endings, censoring,
  frozen pre-update option-model signatures, return inputs, outcome deltas,
  planning use, and option cost feed the lifecycle audit. From a valid composed
  state, every valid STOMP update commits even if audit capacity is exhausted
  or external attribution is rejected; only the audit freezes, with explicit
  terminal diagnostics. Persistent composed-state corruption remains fail-
  closed, and disabling audit preserves exact raw STOMP state and RNG. Explicit
  shape-compatible semantic rebinding resets changed option policy/model/trace/
  optimizer/base-head state from a fresh key and defers while anything is in
  flight. This is L0 observation with zero dispatch, curation, replacement,
  promotion, or go/no-go authority—not empirical benefit or WP7 completion.
- Added the standalone WP7.4 v1 `CalibratedExtendedSearchControl` L0 core.
  Model-free extended-Q replay, primitive-model search, option-model search,
  and combined search share one fixed real-anchor bank and one exact backup
  budget; combined search ranks the union rather than receiving a doubled
  family budget. Correct primitive and differential semi-MDP option targets
  feed a noncompensating calibrated priority from value-change, future-anchor
  reachability, model reliability, and support. Exact two-phase bindings,
  natural-versus-censored resolution, semantic replacement invalidation,
  pending-arm checkpoint parity, deterministic ties, zero RNG, and exact
  resource accounting are tested. Derived target/priority/update overflow is
  an atomic no-op even when every raw operand is finite. The standalone core
  has no automatic keyboard dispatch and establishes no matched planning
  benefit, WP7 completion, promotion, or L3 claim.
- Added a strict matched four-arm development evaluator for calibrated extended
  search. One immutable source/runtime-bound model and calibration snapshot and
  one evaluator-owned Threefry continuing trace are shared across all arms;
  each receives the same experience and exact budget `B`, including combined
  search. Raw diagnostics, exact resource/update accounting, checkpoint/resume,
  exact causal replay verification, and tamper rejection are retained. Configuration
  validation includes all preloaded/per-anchor/per-candidate counter headroom,
  and checkpoint restore canonically replays and bit-compares the complete
  populated prefix. The observable non-secret runtime manifest includes Chex,
  OS/architecture, device, and relevant JAX/XLA configuration. The evaluator is
  `not-assessed`, uses one consumed nonpromoting development seed, has no
  thresholds or verdict, and its frozen action-independent trace establishes
  neither online-model nor policy benefit, WP7 completion, promotion, or L3.
- Added the opt-in `PrototypeSTOMPCalibratedSearchAgent` L0 live sidecar for
  the exact raw legacy Prototype representation. It snapshots the actual
  learned primitive world model and STOMP option models at live decisions,
  binds primitive next-transition and multi-step option outcomes to exact
  ownership, and gives their union one shared calibrated backup budget.
  Search adds no planner RNG, never rewrites Prototype's cached action, and
  has no keyboard or policy authority. Sidecar exhaustion quarantines search
  without blocking a valid Prototype transition; persistent composition
  corruption rolls back the wrapper transaction. Strict host-only checkpoint
  and semantic rebind contracts, eager/JIT update parity, and scan coverage
  are tested. This supplies online model/calibration integration, not a
  policy-benefit, WP7-exit, promotion, or L3 result.
- Added standalone fixed-capacity semantic and procedural consolidation in
  `ConsolidatedMemory`. Query-before-write records carry SHA-256 semantic and
  provenance identities, generations, confidence, source/representation
  revisions, evidence moments, staleness, invalidation, and deterministic
  retirement/replacement. Procedural records additionally track success,
  failure, outcome moments, and an exact option-lifecycle link. Compatible
  same-generation observations merge; next-generation revisions and changed
  identities reset evidence. Exact resources, eager/JIT/scan behavior, and
  source/namespace-bound checkpoint tamper rejection are tested. The module
  has no agent/action/promotion authority and establishes no transfer,
  negative-transfer, stale-skill, WP8, or L3 result. The later opt-in live
  wrappers described above consume it without changing that storage core.
- Added a strict nonpromoting `ConsolidatedMemory` stress evaluator. A frozen
  17-event semantic/procedural recurrence schedule runs full memory, an exact
  same-kernel masked-readout ablation, and a zero-storage/no-kernel comparator
  from one empty source-bound snapshot. Raw traces reconstruct causal
  query-before-write, precision, abstention, harmful recall,
  recurrence/recovery, retained utility, stale-skill harm, eviction/provenance,
  exact counters/resources, eager/compiled parity, checkpoint resume, and
  integrity-bound causal replay. Config, protocol, source, runtime, report, and prefix
  state tampering fail closed. The report is `not-assessed`, has no thresholds,
  and the no-memory comparator is not storage/compute matched, so this is not a
  transfer, negative-transfer, WP8, promotion, SOTA, or L3 result.
- Added the stateless `ConsolidatedProceduralMemoryPolicy` proposal boundary.
  Exact retrieval compatibility/freshness, option-lifecycle identity, evidence
  and success/failure consistency, a Wilson success lower bound, bounded
  outcome uncertainty, nonnegative categorical score mass, and a mandatory
  hard-safety mask gate every result. It proposes only the lowest-index safe
  positive-mass action and owns zero query, write, RNG, dispatch, agent
  mutation, state, checkpoint, or promotion authority. The later live wrapper
  composes this readout explicitly; procedural-transfer efficacy, WP8
  completion, and L3 remain open.
- Added a separate fail-closed complete-prototype manifest contract for all 18
  final scorecard rows. Every exact role must reference immutable artifact
  bytes and a source-pinned trusted validator that reconstructs accepted frozen
  scientific L3 evidence for the same exact prototype configuration and role,
  with pinned protocol/scientific-outcome digests, untouched held-out seeds,
  and complete source closure. Row statuses, aggregate flags, overall status,
  and the manifest self-digest are reconstructed before returning an exit
  code. Optional paper delight and Kondo enable additional mandatory
  actor-learning/compute guardrail roles. Missing or rejected evidence remains
  `not-ready`; malformed, relabeled, source-drifted, or tampered evidence is
  `invalid`. No default evidence index is supplied, so this machinery cannot
  turn unit tests or stored checkboxes into a completion claim.
- Expanded the read-only Forager comparator audit to distinguish immutable historical v1
  artifacts from the unexecuted v2 compatibility surface, enumerate every configured family
  in the pinned upstream tree, and record the missing popular-comparator orientations. The
  next-cycle source recommendations pin Apache-2.0 Dopamine Full Rainbow and POBAX PPO-GRU
  revisions, with Acme R2D2 retained as a higher-risk optional orientation. These are adapter
  provenance anchors and roadmap entries only; they add no run, performance result, winner,
  or SOTA claim.
- Added nonpromoting matched-v3 design contracts for the exact 499,712-step cumulative-reward
  metric, domain-separated trial-block seeds, the broader development candidate universe,
  exact-decimal typed configuration transforms, and simultaneous named-panel empirical-
  Bernstein inference. The contracts are uninstantiated and do not yet bind every candidate
  to an executable configuration/runtime closure; they authorize no qualification, benchmark
  execution, evidence promotion, performance result, winner, or SOTA claim.
- **Completed the never-promoting IPMNIST screening campaign** (36-arm
  registry, `benchmarks/ipmnist_screening.py`, ~80 tests). Full-horizon
  200-task confirmations against the repository's 10-seed published-config
  UPGD-W reproduction (`0.7791`), all under
  `outputs/ipmnist_screening/confirm_full/`: protocol-pure `adamw_cbp`
  10 seeds `0.79876 ± 0.00009` and tuned `upgd_w_wd0005` 10 seeds
  `0.78431 ± 0.00014` (both from 0.28.0), joined by `upgd_l2init`
  `0.78042 ± 0.00030` and the protocol-extended EMA-input-normalization
  family: `upgd_ema_norm` completed all 10 seeds at `0.8536 ± 0.0001`,
  `upgd_ema_norm_sigma0` `0.85051 ± 0.00025`, `upgd_ema_norm_wd0005`
  `0.84745 ± 0.00008`. `upgd_idbd` ties the control (`0.77895`). These are
  development-grade measurements only — no scientific-evidence or SOTA
  claim; promotion still requires a fresh source-bound preregistered v3 run.
- **Dissection cascade** (mechanism decomposition of the 0.854-class result;
  addendum in `CONTINUAL_LEARNING_THEORY.md`): input conditioning `+0.061`
  (`sgd_ema_norm` vs the `0.7791` reproduction), utility gate `+0.011`
  (`upgd_ema_norm_sigma0` vs `sgd_ema_norm`), perturbation `+0.003` when
  normalization is present — contributions stable across the 60→200-task
  horizon extension. Raw-input contrast: disabling the perturbation in the
  published configuration (`upgd_w_sigma0`, 60-task proxy) costs `−0.035`,
  so the noise is load-bearing without normalization and input conditioning
  substitutes for it. Nonpromoting development analysis.
- **`sgd_ema_norm` and hyperparameter-star confirmation arms** (200 tasks,
  3 seeds each, `confirm_full/`): bare SGD+decay+normalization
  (`sgd_ema_norm`) reaches `0.83991 ± 0.00007` — above every protocol-pure
  arm; the `adamw_cbp` star places `adamw_cbp_r3e4` best at
  `0.80126 ± 0.00022` (+0.0025 over `adamw_cbp`), with `m200`/`m50`
  `0.79899`/`0.79887`, `noreset` `0.79815`, `r3e5` `0.79248`, and the
  `adamw_cbp_ema_norm` composition eroding to `0.76895` at full horizon
  (normalization is redundant-to-harmful under Adam's own conditioning).
  Nonpromoting development results.
- **OPMNIST 800-task three-seed closure** (Step 2, published Dohare et al.
  scale): all three seeds completed 48,000,000 online updates over 800
  true-MNIST task blocks with held-out evaluation over all 800 permutation
  views; the merged artifact and solution-gate audit report
  `protocol_complete=true`, `multi_seed_full_scale=true`, and
  `solved_opmnist_step2=false`
  (`claim_scope="limited_opmnist_evidence_not_step2_solution"`). Both
  hybrid-memory candidates beat the best fair MLP on all four
  online/tracking metrics (including `final_window_accuracy`, which flips
  to the candidates relative to the 1-seed artifact) while the fair MLPs
  retain `test_mse`/`test_accuracy` — the prior 1-seed mixed-wins
  conclusion confirmed at scale, NOT a solved-Step-2 claim. Byte-identical
  snapshots of the merged results, gate JSON, and summary are vendored at
  new paths under `outputs/step2_canonical/` with a provenance note
  (`step2_opmnist_solution_800task_3seed_PROVENANCE.md`).

## [0.28.0] - 2026-08-01

### Added

- **Reference trust resolver** (`benchmarks/forager_matched_trust.py`, 25
  tests): stdlib HMAC-SHA256 signed-receipt resolver implementing the
  matched-campaign `TrustResolver` contract — trust-anchor document schema,
  0600 single-link key loading, detached receipt issuance, and a
  fail-closed `SignedReceiptTrustResolver` with immediate revocation and
  freshness windows. Not wired as a default; the reserved
  `content_only_unendorsed_v1` identity is refused. Asymmetric (Ed25519)
  variant deferred until a crypto dependency is pinned.
- **Hidden-partner lifecycle-world v6 runner and strict validator**
  (`evaluation/hidden_partner_lifecycle_world_v6_runner.py`,
  `..._v6_validator.py`): development machinery executing single v6 control
  bindings deterministically with content-hashed result records, plus
  fail-closed output validation. Certification and promotion flags remain
  false; reserved evidence namespaces are refused at run time.
- **Trainable-encoder latent world model** (opt-in, default-off; bitwise
  identical trajectories when disabled): encoder backprop through the latent
  prediction loss under the module's bounded-update discipline, gated by the
  anti-collapse diagnostic. Development-only (L0).
- **Kondo selection accounting** in the delightful-policy-gradient
  development lane: per-step accounting of forward-gate-selected actor logical work
  as a measured counterfactual. Actual compute gating remains unimplemented
  and `KONDO_IMPLEMENTED` remains false.
- **Development-only IPMNIST confirmations**: `adamw_cbp` completed 10 matched
  200-task seeds at mean online accuracy `0.79876` and `upgd_w_wd0005`
  completed 10 at `0.78431`, versus the repository's matched UPGD-W
  reproduction at `0.779147`. These are nonpromoting development results,
  not scientific evidence or a SOTA claim. The three-seed `upgd_ema_norm`
  result and its still-running extension, `upgd_idbd`, and subsequent
  composition screening are deferred to `[Unreleased]`.
- **STOMP checkpoint migration loader** (`core/options.py`): pre-expansion
  checkpoints (missing env-return/duration/baseline-mass fields) now restore
  with documented zero-fill semantics instead of failing on template
  mismatch.
- `alberta-forager-matched-sealed-evaluation` console script for the sealed
  held-out evaluation stage.

### Fixed

- Evidence CLI overwrite footguns: `ftl_decision_cli`, `continual_ia_cli`,
  and `continual_multiagent_cli` no longer default to writing over their
  sha-pinned canonical artifacts — pinned and pre-existing output paths are
  refused before any protocol runs, and write sites use exclusive-create.
  `scale_robust_feature_cli`'s default invocation now explains itself
  instead of exiting 2 bare.
- The legacy `alberta-evidence-gate` no longer checks vendored-out narrative
  documents or five unregenerable Step 1/2 JSON paths. It is now a deprecated
  compatibility wrapper for the strict `alberta-evidence-status` registry;
  the former `--step` selector is rejected because no current per-step claim
  contract exists.
- Construction-time validation for MLP-path optimizers
  (`supported_for_mlp()`): unsupported optimizers now fail at learner
  construction instead of raising `NotImplementedError` inside a jitted
  update.
- `steps/step7.py` vestigial zero-multiplied reward term removed from
  planning action scoring; `streams/gymnasium.py` VALUE mode no longer a
  silent stub; the unused `feature_discovery.replace_fraction` knob was
  removed while legacy serialized configs still load; duplicate
  candidate-imprint formula in `compositional_features.py` consolidated.
- Test hygiene: `test_integrated_hidden_partner.py` marked `slow`;
  Step 1/Step 2 replication suites now skip loudly with a registered
  `replication` marker and a terminal summary of skipped counts; missing
  upstream script trees now surface as visible skips rather than being hidden
  by `collect_ignore`.

### Added (continued)

- Added an opt-in bounded `PrototypeFeatureLifecycle` and its narrow
  `PrototypeAgent` composition. A fixed-width base is augmented with pair
  products and trained from one owner-bound behavior TD target before the
  linear OaK consumers are descriptor-routed. Builder gradients use an exact
  generation-and-full-descriptor-bound pullback; the same identity travels
  atomically with the enabled OaK subtree, so stale or forked consumers fail
  closed even when their observation cache collides numerically. Unsafe
  curation is deferred by proposal rollback, while safe routing is atomic.
  Allocation ceilings, exact resource
  declarations, strict config/state checks, and versioned checkpoints are
  public. The compatible lane deliberately excludes world models, Horde,
  replay, dreaming, IA, partner fusion, experiential memory, and GRU
  perception. This is L0 mechanism coverage only: it proves no benefit,
  promotion eligibility, WP7 completion, multi-consumer/deletion utility, or
  autonomous cumulant, subtask, or option discovery.
- Added an opt-in, stateless `OptionSearchControl` for Prototype's learned
  STOMP option models. It recomputes completion-supported differential
  semi-MDP targets and absolute Bellman residuals after each accepted backup,
  uses a fixed resource budget and stable tie order, and commits only the base
  value learner while preserving real traces, option/lifecycle state, action
  ownership, OaK counters, and RNG. The current action cache is deliberately
  not refreshed, so value effects begin only at a later extended-action
  selection boundary. This is L0 value-backup prioritization, not calibrated
  or combined primitive/option search and not evidence of control benefit.
- **Matched-current Forager campaign machinery** (the `forager_matched_*`
  benchmark modules): OCI runtime/executor qualification, a frozen
  21-candidate open-tuning protocol and resumable campaign runner, RNG-parity
  qualification, and the sealed stage (seal, sealed-evaluation schedule,
  final analysis, paired statistics). Registered two console scripts:
  `alberta-forager-matched-qualification` and
  `alberta-forager-matched-campaign`. Execution status at the time of
  writing: qualification completed
  (`outputs/forager/matched_current_qualification_2c3b214c_v1`); the
  open-tuning campaign is prepared and frozen with **zero executed cells**
  (`outputs/forager/matched_current_open_tuning_2c3b214c_v1` has empty
  `runs/` and `completions/`); the sealed stage is implemented and
  contract-tested with the `alberta-forager-matched-sealed-evaluation`
  console script, but has never been executed. A reference receipt resolver
  exists in-tree, but no operational external anchor/key ceremony or fresh
  authority receipts have been provisioned, so all outputs remain
  content-only, unendorsed, and nonpromoting
  (`promotion_authorized: false`).
- **Matched-current Forager qualification provenance v2**: the exact canonical
  UTF-8 qualification-manifest bytes and SHA-256 now flow into the execution
  plan/executor manifest, score evidence, execution closure, verification
  subject/request, resolver bindings, open and sealed campaign summaries, seal,
  and final-analysis bundle. Every replay boundary cross-checks the shared
  digest, qualification copies preserve their original bytes, legacy v1
  carriers fail closed, and typed plans replay, re-prepare, and freeze their exact
  protocol/candidate/source closure from the retained host assets. Direct construction
  now reparses the protocol, enforces the qualified runtime lock, reconstructs the whole
  executor manifest from live qualified inputs, and rejects forged candidate indexes,
  same-ID substitutions, altered capability receipts, cyclic mappings, and source drift
  after preparation. Qualification now takes a stable, descriptor-backed snapshot of the
  live Alberta source tree, revalidates all transitive staged sources and the bound absolute
  OCI executable/daemon/exact image before and after every probe, and accepts only the
  project tree that loaded the verifier. Fresh loader/protocol/plan replay runs under the
  same fixed non-root user in the exact networkless, read-only, capability-dropped OCI image,
  with actively capped output, both-case proxy credentials cleared, a deterministic
  OCI-readable content-tree mode contract normalized through held descriptor-relative
  handles, and all transitive project imports confined to the staged snapshot. The pinned
  upstream Git archive is likewise produced by one system-path-resolved, content-rebound Git
  executable under a minimal environment, with 4 KiB metadata caps and an exact-size streamed
  archive cap, an explicit built-in tar format/default umask, and no repository-selected
  archive command before its size and SHA-256 are accepted. Git identity and archive launch
  `OSError`s, timeouts, and bounded-output overflows are normalized to the public qualification
  error contract while the executable identity is rebound after every attempt. The complete
  closure is replayed both before fsync/rename and again under the held-inode post-publication
  validator.
  Post-rename validation or durability uncertainty preserves the occupied destination,
  exits with the distinct `PUBLISHED-UNCERTAIN` status, and forbids path reuse. The
  shared-source family is named for its RNG-isolated source contract rather than recurrent
  architecture. The live executor now drains stdout/stderr concurrently into separate,
  actively bounded sinks, preserves its 512 MiB/16 MiB asymmetric limits, never persists an
  overflow witness byte, and performs bounded kill/reap plus cidfile cleanup. Both runners
  assign collision-resistant container names and fall back to exact-name force removal if an
  interruption lands before the cidfile is materialized. CID-file disappearance, partial-read,
  and other read-race failures trigger exact-name cleanup first and then surface as a public
  qualification error carrying the confirmed cleanup state. Completed nonzero clients are
  cleaned too; an unsuccessful removal is accepted only after a separate actively bounded
  daemon query proves the exact random name absent. Absence-query launch, timeout, or overflow
  failures, together with injected runtime-inspector, OCI-probe, and fresh-replay runner
  `OSError`, timeout, and overflow failures, are normalized to their public fail-closed
  qualification errors instead of leaking private process exceptions. Both bounded runners
  tolerate exited-child races, always attempt bounded reaping, independently close their
  selector and both pipes, and normalize kill, reap, and close failures to public errors.
  Malformed nested caches retain layer-specific error contracts. The added provenance remains
  content identity—not endorsement, promotion authority, or a performance claim.
- **Matched-current claim-boundary correction**: candidate-universe schema v2 now states that
  the preregistered procedure identifies only its three named Alberta-versus-selected-external
  contrasts; it does not identify a best member of the 23-candidate registered panel or a
  winner among the six held-out arms. Statistics-result schema v4 carries the same detached
  interpretation boundary: bootstrap superiority and Holm rejection fields are mechanical
  frozen-protocol flags, the bootstrap endpoint is not a population confidence interval, the
  sign-flip calculation has no asserted sign-exchangeability model, and neither result grants
  confirmatory, ranking, or SOTA authority. Final-analysis manifest schema v3 replaces its
  obsolete v1-wording disclaimer with the v2-native contrast-specific/no-panel-ranking
  boundary. Historical artifacts remain unchanged.
- **Nonpromoting causal q-grid diagnostic v2 hardening**: the unchanged public-seed-0,
  epsilon-0.05, q=.50/.75/.90, 10,000-transition diagnostic now writes receipt schema
  `alberta.forager_causal_grid_divergence_probe.v2` under the default
  `causal_q_grid_divergence_seed0_v2` root. Its portable canonical qualification root is bound
  to exact manifest-relative source, inventory, archive, snapshot-descriptor, configuration,
  and capability-receipt paths and schemas; the canonical manifest sidecar joins the 14-file
  read-only mirror, and the receipt carries the replay-critical paths, schemas, archive size,
  and digests. OCI execution clears both-case proxy variables, and cleanup accepts a failed
  force removal only after a separately bounded exact-name absence proof. The harness and tests
  also normalize runner, kill, bounded-reap, and resource-close failures; inability to confirm
  reaping remains fail-closed while exact-name container cleanup is still required. Every final
  receipt descriptor is closed independently, and a close failure after rename receives the
  distinct published-uncertain status. The live probe remains unexecuted, permanently
  open-development, and nonpromoting; no scientific q semantics, evidence gate, or frozen
  protocol changed. A fresh qualification must still be published and its canonical root plus
  all qualification-derived hashes repinned as one reviewed set before this v2 diagnostic is
  run.
- **Beat-SOTA screening lane** (`benchmarks/ipmnist_screening.py`, 51 tests):
  30 registered mechanism-combination arms on a validated 60-task proxy (an
  exact bit-prefix of the 200-task protocol; control parity pinned bitwise),
  including UPGD×IDBD, UPGD×Autostep, UPGD+CBP, AdamW+CBP, UPGD+L2-Init,
  UPGD+EMA-input-norm, a UPGD-W hyperparameter star, weight clipping
  (Elsayed et al. RLC 2024, ±κ/√fan_in, 4 configs), per-layer gate
  normalization, FADE-style meta-learned per-parameter head decay
  (arXiv 2604.27063, sign conventions derived and unit-pinned), and a
  SwiftTD-stabilized UPGD×IDBD (overshoot bound + persistent step-size
  decay). Includes plan/run/validate-proxy/merge CLI, idempotent shard
  workers, and autonomous endgame pipelines with full-protocol pool64
  confirmation and a strict paired BEATS/TIES/BELOW verdict
  (`outputs/ipmnist_screening/`). Screening results are permanently
  nonpromoting.
- **Label-permuted EMNIST protocol-exact lane**
  (`benchmarks/upgd_label_emnist.py`, 21 tests): EMNIST balanced 47-class,
  labels permuted every 2,500 steps, 400 tasks, pinned to the audited
  upstream commit; first artifact (`outputs/upgd_label_emnist/results.v1.json`,
  3 seeds) reproduces the published qualitative separation — UPGD-W online
  accuracy rises across the 400 tasks (first-quarter mean 0.5616 →
  last-quarter 0.7284; whole-run mean 0.67151 vs the ~0.74 figure read-off,
  gap flagged) while AdamW collapses (whole-run mean 0.20081 vs the ~0.35
  read-off, gap flagged). Descriptive only; both reproduction gaps are
  recorded in the artifact.
- **Slowly-Changing Regression v2 sharded protocol** (plan/run-shard/merge/
  validate with immutable self-issued plans and exact-replay validation;
  300 shards = 3 methods × 100 seeds).
- Added `run_ipmnist(noise_mode="pool", noise_pool_steps=N)`, a screening-only
  fast mode for the UPGD Input-permuted MNIST lane that replaces the dominant
  per-step cost -- generating a fresh 282,160-element N(0, sigma^2)
  perturbation every step, ~85-90% of single-core UPGD-W step time -- with
  random contiguous slices of a per-task regenerated noise pool (OpenAI-ES
  style shared noise table). Measured single-core UPGD-W throughput rises
  from 133 steps/s (exact) to 559 steps/s (pool 64) and 951 steps/s
  (pool 16) at the protocol shape. Per-step noise marginals stay exactly
  N(0, sigma^2) but values are reused across steps, so `IPMNISTRunResult`
  now records `noise_mode` and `partial_payload` refuses pool-mode results:
  they can never enter the v2/v3 artifact lifecycle. The default
  `noise_mode="step"` inner loop is verified bitwise identical to the
  pre-change runner on fixed-seed 2x5000-step publication-shape runs for
  both learners, and existing shards/artifacts are unaffected. Full-protocol
  200-task pool-validation runs are recorded under
  `outputs/upgd_ipmnist_runner_opt/`.
- Added the strict, namespaced continual-IA v2 development-only contract and
  `CONTINUAL_IA_V2_RUNBOOK.md`. It freezes treatment `recommendation_p075`,
  exact acceptance probability 0.75, the old gates with seed start 60, and
  seeds 60–89 under immutable source/runtime-byte-bound
  plan/reservation/one-seed-shard/merge files. Persistent reservations are
  published before execution, shards retain complete primitive causal traces
  and exact replay, and merge replays once before reconstructing all paired
  intervals, budgets, credit, identity, and gates. The self-issued schema has
  no external pre-run chronology and can never set `internally_accepted=true`.
  The contract is unissued: no plan, reservation, v2 seed execution, shard, or
  artifact exists.
- Added the active, namespaced UPGD Input-permuted MNIST v3 future-execution
  contract and `UPGD_IPMNIST_V3_RUNBOOK.md`. It requires one immutable pre-run
  plan, exact selected configuration/hyperparameters and closed deviations,
  exactly 20 operator-reserved fresh seeds, data/runtime/static-import-closure/command
  bindings, one learner/seed per shard, atomic no-overwrite publication, and
  exact Cartesian merge with raw shard hashes and recomputed summaries. No v3
  plan has been issued, no v3 result exists, and no fresh v3 seed has been
  consumed. The self-issued, externally unattested schema is permanently
  nonpromoting; all sealed completed-diagnostic records remain unchanged.
- Added a fail-closed verifier-issued tuning-envelope boundary for a future
  immutable Forager OCI evaluation adapter. It binds the tuning report, raw
  evidence, source tree/archive, read-only content-addressed source mount,
  runtime profile, and environment RNG schedule to an externally authenticated
  authority. The host/snapshot runner cannot invoke the adapter or promote its
  own tuning report, and candidate OCI metadata is not attestation.
- Added a strict, permanently nonpromoting validator and reconciliation
  receipt for the completed 10-seed UPGD Input-permuted MNIST diagnostic. The
  original finalizer artifact remains byte-exact; the canonical reconciled
  artifact repairs its note and portable post-hoc cache binding without
  repairing the absent execution-time provenance or 10-vs-20-seed deviation.
- Added immutable corrections for the four-seed Forager development records:
  an RTU capture-digest addendum and a reconciled DQN comparison receipt using
  the public importer. The descriptive RTU-minus-DQN mean is `+0.3309` on four
  consumed seeds, but comparator timing, runtime, representation, resources,
  replay, and update work are unmatched, so the receipt permanently forbids
  inferential, causal, speed, SOTA, and Alberta Plan claims.
- Replaced the slowly-changing-regression v1 writer and retrospectively
  calibrated qualitative gate with a strict namespaced v2 development
  protocol. It adds a selected ReLU/Kaiming/true-MSE ordinary-BP path,
  distinguishes Alberta-local CBP/UPGD extensions, requires immutable
  one-method/seed shards under a pre-run source/run/runtime-bound plan, checks
  exact paired coverage and shared environments, and reconstructs descriptive
  results from size/SHA-bound shards. The self-recorded envelope is not
  external attestation, so every valid v2 artifact remains permanently
  nonpromoting.

### Changed

- The live evidence registry currently reports all five registered claims
  `invalid` (overall `invalid`, exit `2`): the post-0.27.0 source and
  documentation waves edited registered source files after their artifacts
  were pinned. This is the fail-closed design working as intended. The pinned
  `outputs/` artifacts themselves are unchanged; renewing a claim requires
  rerunning its frozen protocol to a new artifact path and schema version
  with untouched preregistered seeds.

## [0.27.0] - 2026-07-31

### Added

- Added an initial publication-shaped Nature-2024 slowly-changing-regression runner
  (Dohare et al.): the `m + 1`-bit stream with `f` flipping bits every
  `T` examples, the 100-LTU random sign-weight target network with
  `(m + 1) * beta - S_i` thresholds, and a fully vmapped 100-run runner over
  plain-SGD `MLPLearner`, `CBPMLPLearner`, and `UPGDLearner` with 40k-example
  bins over 3M examples. This initial path was not protocol exact: learner
  initialization, MSE scaling, RNG/numeric semantics, target-bias encoding,
  comparator identity, provenance, and post-hoc thresholds differed or were
  incomplete. No full artifact was produced; the Unreleased v2 contract
  replaces its artifact-writing path.

- Added a strict, reconstructing continual-evaluation report with
  predict-before-update traces, evaluator-only regime metadata, a candidate
  plus at least two exactly budget-matched baselines, continual-transfer and
  recovery metrics, p50/p95/p99 latency, delayed/dropped observations, memory
  and optional energy measurements, graded safety/near-miss accounting, and
  required component/plasticity diagnostics. Its bounded scalar streaming
  executor rejects prediction/probe/source-state mutation, noncanonical or
  nondeterministic checkpoint state, configuration drift, deadline and host
  memory-budget violations, and inconsistent exposure metadata. Safety and
  metric applicability distinguish unavailable measurements from measured
  zeros. A strict evaluator-bound artifact now hashes the full evaluator
  identity (including stream/probe and learner configuration) and metric core,
  cross-validates their shared protocol/budget/condition semantics, and uses
  atomic report/checkpoint writes. Reports remain explicitly `not-assessed`;
  constructing one is not a scientific pass.
- Added a strict v2 continuing-control evaluator and hardened `PrototypeAgent`
  adapter. Candidate and baselines receive independent functional environment
  states; exact observation/action/decision-ID ownership enforces
  predict-before-outcome ordering while regime identity stays evaluator-only.
  The reconstructing report pins metric direction, exposure rows, recovery and
  stability references, then derives lifetime/prequential and per-regime
  return, post-change adaptation AUC, sustained recovery, final held-out action
  scores, forgetting, backward/forward transfer, stability, and worst-window
  return with explicit applicability. Canonical config/source digests and
  atomic checkpoints fail closed on identity or metric tampering. This is
  development-only infrastructure; no completed Prototype comparison or
  scientific claim is attached.
- Added a strict development-only paired continual-control campaign. Explicit
  seed wrappers bind each environment and learner configuration; cross-seed
  invariants require identical protocol, budget, probes, condition roles and
  normalized identities. Every raw report is retained, unavailable pairs stay
  unavailable, and direction-normalized candidate-minus-baseline differences
  receive deterministic stratified paired-bootstrap intervals from a frozen
  counter-hash RNG. Config/source/report/comparison hashes and atomic I/O make
  the artifact reconstructing. Campaigns remain `not-assessed`; no threshold,
  completed Prototype campaign, or scientific claim is supplied.
- Added a strict development-only privileged continual-control reference suite
  outside the ordinary matched-condition list. It runs one independently
  initialized learner per evaluator regime identity and retains it on
  recurrence, a stationary-multitask learner trained from an exactly budgeted
  frozen extra stream, and an exact frozen counterfactual action-outcome upper
  reference. Canonical reports and checkpoints bind source/config hashes,
  decision ownership, extra-data and callback counts, resources, unavailable
  recurrence states, and privilege/comparability disclosures. These are
  descriptive `not-assessed` context bounds, not baselines or evidence.
- Added the WP4 shallow world-model reference: an action-indexed affine
  regularized-FTL/ridge learner over grounded next-observation, reward, and
  continuation targets. It predicts before recursively updating fixed
  Gram/cross sufficient statistics, solves only the selected action block,
  validates counts, normal equations, positive-semidefinite statistics and
  numeric bounds fail-closed, has strict RNG-free checkpoints and exact
  allocation accounting, and exposes a pure one-step supplied-linear-value
  action scorer. This is L0 mechanism coverage, not a paper reproduction,
  calibrated model, MPC result, or control-benefit claim.
- Added an isolated bounded recurrent latent world-model ensemble with one
  trainable GRU and grounded mean/heteroscedastic-variance heads per bootstrap
  member. Exact start/decision/transition ownership, predict-before-update NLL,
  final-target/reset-cache boundaries, one recurrent advance per accepted
  event, stopped-target representation gradients, atomic failure handling,
  fixed resource accounting, JIT/scan, and digest-bound checkpoint continuation
  have L0 tests. Added it as an opt-in fourth mutually exclusive Prototype
  model lane with exact dispatched-decision caching, transactional causal
  signal state, real-NLL-only builder/mixer/candidate-audit routing, and whole-transition
  rollback on recurrent rejection. It is not a replay, planning, calibration,
  or efficacy result.
- Added a bounded `PartnerPolicyFusion` L0 core with fixed typed
  message batches, five explicit routes, discrete score-based blending, a
  caller-owned hard action mask, one exact decision-bound feedback record, and
  contextual logistic reliability updates from realized assistance plus
  observed safety. Stale, duplicate, and misattributed feedback is an atomic
  no-op; cold-start acceptance is labeled uncalibrated development exploration.
  Checkpoint, resource, eager/JIT, and scan contracts are tested. Added an
  opt-in `PrototypeAgent` composition that binds the full lifecycle ID, applies
  prior feedback before the next fusion decision, derives real OaK base and
  keyboard proposals, rewrites the exact primitive credit owner, synchronizes
  the recurrent action cache, and rolls back the whole transition on unsafe
  base dispatch or corrupt post-state. It makes no calibration, benefit, or
  WP8 completion claim.
- Added a strict development-only `ExperientialMemory` transfer evaluator. It
  runs a fixed evaluator-owned recurring A/B/A trace from an immutable empty
  snapshot, preserves exact query-before-write ownership, and records raw
  neighbor/gate predictions, no-memory fallbacks, harmful recall, abstention
  causes, eviction provenance, first/return descriptions, loophole checks, and
  exact resources. Config, protocol, source, snapshot, canonical report, and
  checkpoint hashes reconstruct fail closed with eager/compiled parity. It has
  no threshold or transfer, retention, efficacy, promotion, WP8, or SOTA claim.
- Added a stateless `ExperientialMemoryPolicy` and opt-in `PrototypeAgent`
  composition. Retrieved action vectors are categorical score mass—not rounded
  action identifiers—and selection is the lowest-index safe positive-mass
  argmax. Prototype queries the next decision before writing a grounded
  one-hot of the primitive action that actually executed plus bootstrap
  representation and reward, then composes memory dispatch before partner
  fusion. Full lifecycle IDs, no-memory state-shape preservation, whole-event
  rollback, checkpoints, curation, eager/JIT/scan behavior, and exact resources
  are tested. The resource declaration exposes two deterministic pre-state
  queries and zero RNG; no transfer or control benefit is claimed.
- Added a strict option-keyboard policy proposal and primitive-dispatch
  boundary. It computes the deterministic chord argmax from current option
  values, verifies exact decision-observation ownership and the complete fixed
  STOMP/OaK state, applies a caller-owned hard safety mask, and rewrites either
  the base primitive-action cache or the active option's intra-option cache so
  subsequent TD credit follows the action actually executed. Unsafe proposals
  use an independently safe base; unsafe bases and corrupt inputs are exact
  fail-closed no-ops. RNG/state shape and package exports are preserved. This
  is L0 ownership/safety integration, not chord-learning or control-benefit
  evidence.
- Added a strict development-only frozen world-model snapshot evaluator. It
  retains raw ensemble-member and mean grounded predictions/targets, derives
  reconstructable disagreement/error bins, correlations, coverage-risk, and
  ID/OOD plus state/action-region summaries, and keeps the residual EMA
  explicitly separate and non-probabilistic. Exact open-loop diagnostics are
  optional and bounded; configs, probes, sources, snapshots, traces, summaries,
  resources, checkpoints, and no-overwrite reports are hash-bound. It applies
  no threshold and makes no calibration or scientific claim.
- Extended that instrumentation with a separate recurrent development adapter.
  It binds a frozen initial snapshot, scores each exact cached distribution
  before one update of an isolated copy, and retains reconstructable member
  means, heteroscedastic variances, NLLs, warm-up applicability, ID/OOD and
  evaluator-owned state/action regions, final isolated-state counters, sources,
  resources, canonical reports, and source-bound snapshot checkpoints. The
  supplied snapshot remains unchanged. The Gaussian objective is not evidence
  of calibrated likelihood, and no threshold, efficacy, or promotion claim is
  made.
- Added a strict recurrent-only retention companion that requires exact ordered
  case reuse after an evaluator-owned intervening context and recurrent resets,
  then reconstructs phase, recurrence-entry, ID/OOD, and within-occurrence NLL
  summaries from the source-bound prequential trace. It keeps the supplied
  snapshot unchanged and remains development-only `not-assessed`; it is not a
  retention, calibration, Prototype integration, or scientific claim.
- Added a strict development-only average-reward actor/critic A/B/A companion.
  Regime identities, reward tables, preferred actions, and value targets remain
  evaluator-only while exact cached target/epsilon-mixture behavior policies,
  critic error, actor margin, churn, return/recovery, plasticity, action
  activity, resources, source hashes, and checkpoints remain reconstructable.
  The core now learns from the cached decision, commits the update, and samples
  its successor from committed parameters. Raw `pi / b` is diagnostic and the
  logged `(1 - epsilon) pi / b` term is explicitly the behavior-score chain
  rule, not off-policy correction. The probe is one-seed, `not-assessed`, and
  supplies no retention, efficacy, Prototype, promotion, or SOTA claim.
- Added an isolated bounded continuous average-reward actor/critic. It uses a
  direct affine-`tanh` diagonal Gaussian without latent clipping or endpoint
  adjustment, retains exact pre-`tanh` decision ownership and stable
  transformed target/behavior log densities, and applies the analytically
  cancelling latent-density ratio to the complete actor eligibility trace.
  Actor, critic, traces, LMS states, differential reward baseline, typed RNG,
  and saturating counters are separate; one successor is sampled only after an
  atomic finite commit. Strict config/checkpoint/resources, density/score,
  saturation, rollback, causal-cache, eager/JIT/scan, and public-export tests
  pass. This is L0 mechanism coverage with no state-distribution correction,
  off-policy convergence, retention, efficacy, or SOTA claim.
- Added a strict continuous actor/critic recurrence evaluator over one fixed
  12-event, one-action-dimensional A/B/A life from an immutable source-bound
  snapshot. It reconstructs cached latent/action ownership, transformed
  target/behavior densities, the exact latent ratio, rewards, same-state
  gauge-centered critic error, actor error/churn, plasticity/activity,
  successor ownership, counters, resources, final state, and exact live replay.
  Reports and checkpoints are hash-bound, no-overwrite, and subject to absolute
  byte/scalar ceilings. A symmetric eight-float32-ULP allowance is isolated to
  transformed diagnostic-density reconstruction; the policy-defining latent
  ratio remains bit-exact and larger tampering fails. The result is
  development-only `not-assessed`, with no retention, efficacy, convergence,
  calibration, promotion, or SOTA claim.
- Added an isolated nonlinear discrete off-policy actor/critic. A shared tanh
  trunk, categorical actor, and scalar critic learn only from the exact cached
  executed-action receipt, including target and caller-declared behavior log
  probabilities/revisions and exact action identity. Clipped per-decision
  action ratios advance separate actor/critic head and trunk traces; actor,
  critic, and trunk have fixed independent plastic/frozen policies. Typed
  Threefry sampling, exact fail-stop clocks, atomic rollback, strict
  config/checkpoint construction, complete persistent-state byte accounting,
  a hand-derived two-step trace, and eager/JIT/scan parity are tested. This is
  L0 `not_assessed` discounted scalar-V machinery with no state-visitation
  correction, average-reward baseline, learned utility policy, authenticated
  external behavior owner, convergence, retention, efficacy, artifact,
  promotion, or SOTA claim.
- Added an isolated nonlinear discrete average-reward actor/critic with
  separate one-hidden-layer `tanh` actor and critic networks, head/trunk
  traces, momentum, fixed component plasticity policies, bounded utility
  telemetry, and a learned reward-rate baseline. Exact target and caller-owned
  behavior distributions, revisions, action identity, and a fixed owner digest
  flow through a pure proposal and a commit that recomputes and bit-validates
  the candidate before taking the sole successor draw. Ordinary
  epsilon-mixture behavior-score and clipped per-decision target-importance
  modes, fail-stop clocks, raw numeric bounds, softmax underflow, rollback,
  checkpoints, resources, and eager/JIT/scan parity are covered. This is L0
  `not_assessed` machinery: importance is action-only, utility has no learned
  plasticity authority, and no convergence, retention, matched control,
  safety, promotion, or SOTA claim follows.
- Added a tiny matched-stream world-model retention diagnostic comparing the
  shallow reference, plain bootstrap ensemble, and atomic model-only rehearsal
  over uninterrupted A/B/A lives with interleaved noisy-TV outcomes. Raw
  grounded errors, adaptation/recurrence descriptions, ensemble disagreement,
  typed signals, replay composition, and logical resource/event accounting
  reconstruct from fixed streams. The two-seed, 18-step protocol is
  resource-unmatched and permanently `not-assessed`; it supports no retention,
  superiority, control, or SOTA claim.
- Added identity, fixed-trace, and online trainable gated state builders under
  one causal fixed-budget contract, including checkpoint parity and a
  development-only partially observable diagnostic.
- Added a multi-objective candidate-update safety audit under the historical
  `assess_gradient_joy` API. It uses
  caller-supplied objective, retention, and safety probes under a required
  `probe_independence_attested` contract plus a trust bound, fails closed on
  unavailable, unattested, or invalid evidence, and requires explicit
  availability for all eight separately reported learning-value channels. The
  `LearningValue.delight` evidence channel is valid only when its bits exactly
  equal the finite float32 advantage-surprisal product; it remains evidence
  rather than a candidate-update verdict.
  Raw-candidate factors produce a tentative soft-weighted update; both stages
  must satisfy objective, retention, and safety magnitude gates. Numeric
  controls must survive as finite normal float32 values, and cosine alignment
  is range-clipped with an explicit float32 endpoint policy. Paper-specific
  Delightful Policy Gradient remains a distinct actor-sample experiment.
  Added `apply_gradient_joy_update` as the atomic parameter-application boundary:
  it reassesses internally, requires exact parameter/update PyTree contracts,
  re-audits the effective stored delta after dtype cast and parameter addition,
  applies atomically, and reports the formed candidate and effective-delta
  assessments separately from a parameter change actually being `applied`.
  The Prototype's opt-in learned-state lane is now a development consumer: it
  binds probe evidence to the dispatched decision and commits only an accepted
  effective delta into the advanced recurrent builder state. No realized-
  benefit result is claimed.
- Added the paper-defined `KondoGate` forward/sparse-gather boundary. Delight
  is derived internally as advantage times selected-action surprisal, and
  its historical `sparks_joy` view marks forward admission intent for the
  caller. The forward-only gate does not run autodiff; only an actor consumer
  establishes actual backward execution.
  Finite-temperature
  Bernoulli-price and deterministic fixed-rate top-k modes have typed RNG,
  bounded accounting, strict checkpoints, caller-declared force preservation,
  and an explicit flag requiring caller-managed full-shape fallback on
  overflow. When configured capacity is below batch size, the fixed-capacity
  gather gives downstream autodiff a genuinely smaller input; tests inspect
  that backward JAXPR rather than equating a masked full-batch loss with saved work.
  This is L0 mechanism coverage with no integrated consumer, measured compute
  saving, DG reproduction, or learning/safety claim.
- Added `KondoSparseActor`, the first real nonlinear categorical actor consumer
  of that boundary. It binds exact action identity, policy revision, and
  behavior-log-probability bits, gathers the fixed-capacity actor batch first,
  and only then invokes `jax.value_and_grad`. Tests witness a capacity-3
  backward JAXPR instead of the full batch of 6. Forced guardrail rows and
  Bernoulli overflow take an explicit full-shape fallback without dropping a
  selected sample. Returns and baseline predictions enter the actor only via
  detached advantage; critic and safety features stay outside its loss, while
  all protected learners remain full-batch and ungated. State, resources,
  rollback, and source-bound checkpoint integrity are tested. Host
  orchestration remains required, and there is no demonstrated compute saving,
  closed-loop on-policy or behavior-policy-corrected actor-critic integration,
  efficacy, safety, promotion, or L3 claim.
- Added a strict nonpromoting four-arm `KondoSparseActor` development evaluator.
  Ordinary-full, capacity-matched uniform-sparse, Kondo-top-k, and diagnostic-
  overflow arms share one immutable parameter snapshot and source trace, one
  update opportunity per external batch, and explicit selected-sample and
  compiled-backward shape/invocation accounting. An update-free timing section
  compiles, warms, blocks, and evaluator-interleaves the fixed kernels before
  retaining raw `perf_counter_ns` samples and independently reconstructed
  nearest-rank p50/p95. Host screen/gathering, accelerator memory, energy, and
  end-to-end latency are excluded and disclosed. Wall-clock bytes are outside
  deterministic replay; config, source/runtime, trace, snapshot, and causal
  checkpoint prefixes fail closed. All statuses are `not_assessed`, with no
  speedup, efficacy, safety, output-write, policy, promotion, or L3 claim.
- Added a strict nonpromoting Kondo actor/critic replay lane. Ordinary-full,
  capacity-matched uniform, paper top-k Kondo, and fixed-capacity top-k plus a
  minimum random reserve share one evaluator-fixed A1/B/A2 contextual-gambling
  trace and exactly one actor/protected update opportunity per source batch.
  Baseline, critic, representation, world-model, and safety/guardrail learning
  remains full-batch and independently bit-identical across arms, including
  rare failures. Current-policy delight, executed actor-backward inclusion
  masks, gather shapes, logical proxies, descriptive recurrence readouts, and causal
  checkpoint replay are retained. Because the fixed actions have no source
  behavior policy and no importance correction, actor updates are explicitly
  off-policy surrogates with no policy-gradient/DG-efficacy, compute, safety,
  output, evidence, or promotion claim; every result is `not_assessed`.
- Added a strict closed-loop on-policy Kondo development evaluator. Its four
  arms sample from their own immutable actor revisions using evaluator-owned
  typed Threefry common uniforms for exogenous randomness only; actions and
  trajectories are never shared or assumed equal. Updates occur once at each
  batch boundary, all five protected learners remain full-batch, and forced
  rare failures reach both actor and guardrail learning. Exact action,
  behavior-probability, revision, causal-chain, checkpoint, source/runtime, and
  replay contracts are tested. It writes nothing and remains `not_assessed`,
  with no efficacy, compute, safety, evidence, or promotion claim.
- Added a bounded `LearningValueRouter` for the eight separately typed
  learning-value channels. Each channel has explicit producer/object/units/
  domain metadata, independent validation, and causal pre-update Welford
  normalization. Six exact-mask routes serve the paper-DG actor, exploration,
  model memory/replay, adaptation/change, safety, and the complete evidence
  bundle for the separate candidate-update audit; there is no default sum.
  Invalid unrelated inputs cannot suppress valid safety learning, delight must
  exactly equal its float32 advantage-surprisal product, and the router performs
  neither the candidate audit nor Kondo selection. Fixed resources, counter-capacity behavior,
  strict checkpoints, and eager/JIT/scan parity are mechanism-tested. An
  opt-in owner-bound Prototype integration now routes exactly once after causal
  typed signals on an accepted real transition, keeps producer availability
  independent of representation-candidate validity, and supplies only raw
  routed values to the candidate audit. It adds a v19 enabled checkpoint while
  leaving disabled config and state PyTrees unchanged; no calibration or
  consumer-benefit claim is made.
- Added an isolated continuing categorical actor-critic for the separate
  paper-specific DG policy-gradient experiment. Ordinary and paper-specific DG
  modes share the actor, differential critic, reward-rate baseline,
  typed RNG, and action sampler; the actor trace is fixed to zero, and the
  detached paper-defined delight coefficient never gates critic/baseline
  or explicit safety/model/representation routes. Exact on-policy records, atomic failure,
  JIT/scan/checkpoint parity, gate strata, effective sample size, and logical
  resource accounting are contract-tested. A strict development runner now
  compares both modes under paired random-number schedules on contextual
  heteroskedastic gambling and uninterrupted six-state RiverSwim A/B/A, with a
  validator that reconstructs every declared trace and derived diagnostic.
  The seeds/thresholds are nonpromoting, trajectories may diverge after policy
  divergence, and no policy-quality, safety, compute-saving, or paper-
  reproduction claim follows.
- Added a predict-before-update typed learning-signal producer that keeps
  ensemble epistemic disagreement, aleatoric uncertainty, normalized residual,
  learning progress, and sustained change probability separate, with warm-up
  flags, noisy-TV and change diagnostics, bounded counters, and fixed resource
  accounting. Causal ordering is part of the caller contract.
- Added a fixed-size bootstrap world-model ensemble with distinct initialization,
  persistent mask RNG/counters, typed warm-up signals, a causal representation
  gradient, exact resource accounting, and full checkpoint parity. It is
  integrated into Prototype as a mutually exclusive development lane; its
  residual-variance proxy is not externally calibrated aleatoric uncertainty,
  and ensemble dreaming remains disabled.
- Added `DualReplayMemory`, a fixed-capacity recency-FIFO plus long-term replay
  substrate with reservoir or configured surprise/coverage/progress retention,
  explicit aleatoric control, policy/value and representation provenance,
  fixed-quota stale-aware sampling, exact resource accounting, and strict
  deterministic checkpoints. Added `ModelReplayRehearsal` to atomically compose
  the real ensemble update, signal-aware record, fixed-quota sample, and
  model-member-only rehearsal, with isolated replay RNG/masks/counters and
  strict composition checkpoints/resources. Prototype exposes it as a mutually
  exclusive model lane and sends only the commit-gated real gradient to builder
  learning and the separate candidate-update audit. Replay never trains the
  actor, critic, builder, or causal calibrator; no retention or control benefit
  is claimed.
- Added a development-only component-retention evaluator with frozen
  non-learning snapshots; seven separately applicable representation, model,
  reward/termination, value, and actor channels; mutation checks; bounded
  record/state budgets; and reconstructing reports/checkpoint resume. It is not
  a longitudinal or scientific retention result.
- Added fixed-capacity experiential memory with typed provenance/version
  metadata, query-before-write ordering, similarity/reliability/staleness
  retrieval, deterministic utility/recency eviction, controlled stale- and
  wrong-version abstention checks, exact byte accounting, and checkpoint/scan
  parity. No transfer benefit is claimed.
- Added source-profiled canonical UPGD implementations for the paper,
  official README, and official experiment equations, plus a numerically safe
  extended default. Regression tests pin their documented normalization and
  update-multiplier differences instead of silently blending variants.
- Preserved the immutable recurring- and scale-pair evidence artifacts after
  multiple registered implementation, artifact-builder, and CLI sources
  evolved. The live evidence registry now reports both claims as
  source-invalidated; consumed-seed
  compatibility reruns cannot promote, and any renewed claim needs a new
  path/schema plus untouched preregistered seeds. Their generation CLIs now
  reject the reserved canonical paths even if those files are missing, reject
  every existing destination before running, and publish new-path
  reproductions atomically without overwriting a concurrent writer.
- Added the historically accepted immutable
  `alberta.scale_robust_pair_feature_evidence.v2` package comparison on 30
  exact namespace-derived fresh seeds. The strict artifact records primitive
  phase rows, structural retention, paired intervals, fixed resources, source
  hashes, and scientific digest
  `c2fee922c04a59fe26b4b8c9cfa77ddd9198cfa2bc923f54fec14b649bd3bb2c`.
  Its scope remains narrow: visible context, an exhaustive finite pair archive,
  one fixed learner initialization, and a primary-versus-legacy comparison
  that changes scale normalization and ObGD while adding 464 bytes.
- Added L0 substrates for the next integrated kernel: an uncued recurring
  scripted-partner stream, online partner-action prediction with an input-loss
  gradient, a bounded joint-outcome model with external partner-belief
  marginalization, and fail-closed atomic feature-bank routing by descriptor
  identity. These do not yet constitute learning-partner coadaptation or L3.
- Added a bounded L0 hidden-partner kernel that composes online gated state,
  pair-feature discovery, partner prediction, all-cell joint-world planning,
  atomic descriptor routing, and explicit-action differential SARSA in one
  causal update. Shape-matched ablations isolate state learning, recurrent
  memory, feature deployment, survivor carry, active-utility retention, and
  planning. Evicted downstream columns have no dormant identity archive, and
  full-life discovery/control benefit remains development work.
- Added `PrototypeTransition`, full-Prototype-state checkpoint helpers, and
  public Prototype exports. This effective-outcome/continuation path carries
  environment
  continuation through real world-model, primitive/option control, the IA
  exo-cortex, scan, and accepted dream updates, while optional per-GVF
  continuation reaches Horde without erasing each demon's declared horizon.
  The authoritative contract now also records the raw observation, exact
  dispatched primitive action, lifecycle/generation decision token,
  terminated/truncated flags, final/bootstrap observation, and post-reset
  decision observation. Runtime-invalid transitions are transactional no-ops;
  positive-discount truncation bootstraps on the final state while resetting
  episode-local recurrence and option execution before selecting on the reset
  state. Prototype checkpoints now use the v3 rehearsal-isolation schema. A v2
  ensemble checkpoint preserves every learned member, signal statistic, and
  real bootstrap stream while deterministically initializing only the new
  replay key/mask/counters; ambiguous v1 lifecycle restoration still requires
  an explicit provenance trust override.
- Integrated the common `StateBuilder` protocol into `PrototypeAgent` with
  identity, fixed-trace, and online-gated modes, exact recurrence/event counts,
  cached-action semantics, scan parity, and config-bound checkpoints. The
  online-gated learner now rejects non-finite/overflowing gradients atomically;
  a successor opt-in Prototype mixer combines its real grounded-model gradient
  with the current base-Q or intra-option control semi-gradient. It records
  source norms, weights, clipping, cosine/conflict, and failures, excludes
  delayed option-start and replay gradients, and sends the exact mixed
  candidate to both builder learning and the candidate-update audit boundary. GVF,
  inverse, feature-utility, causal-deletion, empirical-balancing, and matched
  Forager evidence remain absent, so this is not a learned-state efficacy claim.
- Added an opt-in `sample_one_hot` Prototype dream-observation mode for wholly
  one-hot control features. It projects finite model outputs to categorical
  mass, samples on an RNG stream isolated from legacy anchor/action choices,
  and vetoes malformed, non-finite, or zero-mass projections without changing
  learner state. The legacy expectation-valued path and serialized config stay
  unchanged by default. An eight-seed consumed-development RiverSwim diagnostic
  measured `0.2475` versus `0.1930` mean lifetime reward (paired `+0.0545`,
  6/8 seeds positive); it is explicitly nonpromoting and carries no held-out
  efficacy claim.
- Added **SwiftTD** (`core/swift_td.py`) — Javed, Sharifnassab & Sutton
  (RLC 2024) step-size optimization with an overshoot bound and step-size
  decay, float32-exact against the authors' C++ reference. Follows the
  `TDOptimizer` interface, so it drives `TDLinearLearner` and the TD learning
  loops.
- Added the **stacked linear Horde** (`core/stacked_horde.py`): the GVF demon
  axis as one batched array axis with exact TD(λ) semantics, per-decision
  importance sampling, NaN cumulant masking, and a `nexting_spec` helper.
  Measured on CPU: 1,024 demons × 2,000 steps in ~0.2 s steady state
  (~0.3 s compile) versus ~140 s run + ~144 s compile for the loop-unrolled
  multi-head path, and 65,536 demons at ~4.0e7 demon-updates/s.
- Added the **Alberta Gauntlet and lifetime streams plus certification
  suites** (`streams/gauntlet.py`, `tests/test_gauntlet_certification.py`,
  `tests/test_gauntlet_discovery.py`, `tests/test_lifetime_demonstration.py`,
  `tests/test_lifetime_longevity.py`): tracking/relevance/recovery/
  scale-robustness scorecards (P1–P6), the exhaustive pair-discovery rung, the
  64k-step single-life demonstration, and the reproducible 1M-step × 8-seed
  longevity extension (zero non-finite steps over 8M updates, late-life
  savings 10–18x with no erosion, recurrence re-entry error 0.19x first
  exposure).
- Added **closed-loop control gates** (`streams/closed_loop.py`,
  `tests/test_control_learning_gates.py`): the 2-state switching MDP and
  RiverSwim with analytic optima, plus SARSA/DifferentialSARSA reward-rise
  gates against random baselines.
- Added **multi-agent streams and simulations** (`streams/opponent.py`,
  `streams/matrix_game.py`, `streams/recurring_multiagent.py`,
  `tests/test_multiagent_sim.py`): learning-opponent and adversarial-pursuit
  streams (endogenous non-stationarity; a frozen predictor is driven to 1771
  MSE while continual learners hold ≤0.12), the recurring convention game
  (instant convention recall on rule recurrence with context), and the frozen
  two-agent world behind the promoted coadaptation claim.
- Added **context inference** (`core/context_inference.py`,
  `tests/test_context_inference.py`): a bounded bank of per-(state, action)
  reward tables inferring the active hidden regime and gating control
  features by inferred slot. Development evidence: +0.519 mean paired gap
  over the no-context ablation on the tested hidden two-rule life.
- Added **TD/GVF-target discovery** (`tests/test_td_target_discovery.py`):
  pair-feature discovery against the controller's differential-SARSA target
  `r − r̄ + Q(s', a')`; at least three of four context-binding products found
  per seed and ~+0.16 recurrence advantage over the raw twin (development,
  oracle-visible context).
- Added **discovery-driven control** (`tests/test_discovery_control_life.py`):
  features discovered online from action-conditioned reward prediction feed a
  bias-free `DifferentialSARSAAgent` with Q-weight carry-over by feature
  identity across bank refreshes; the discovered bank recovers 92–94% of the
  oracle-representation advantage (development).
- Added the **integrated single-life diagnostic**
  (`tests/test_integrated_life.py`): one uninterrupted 48,000-step
  closed-loop life with 120 payoff switches and a mid-life input-noise
  stressor; the gated rung reaches ≥0.969 lifetime reward with flat
  re-coordination and a +0.60 paired memory gap versus its ablation
  (development; visible context, hand-built representation).
- Added **held-out confirmation** of the wave-3 development results
  (`tests/test_held_out_confirmation.py`): precommitted floors passed on
  first-run, then-uninspected batches for primary-only scale discovery,
  discovery-control, a hand-gated life, and longevity (nonpromoting
  robustness evidence).
- Added the **Forager benchmark family**
  (`alberta_framework/benchmarks/`: `forager.py`, `causal_map_forager.py`,
  `forager_matrix.py`, `forager_results.py`, `official_foragax.py`): the
  pinned `continual-foragax==0.55.0` integration with paper presets, a causal
  feature encoder, the `alberta_horde_ac` and `alberta_causal_map` methods,
  official-NPZ and legacy-SQLite importers, and strict paired statistics —
  plus the `alberta-forager-benchmark` console script, the `[forager]`
  optional-dependency extra, and `FORAGER_BENCHMARK.md`. Preserved the
  completed four-seed RTU-RTRL 500k development run as a byte-pinned,
  explicitly nonpromoting receipt with recomputed summary tests.
- Added publication-shaped development runners for slowly-changing regression
  and UPGD Input-permuted MNIST (`upgd_ipmnist`). The UPGD lane
  completed a 10-seed matched development diagnostic: UPGD-W/AdamW mean online
  accuracies were `0.7791470803916454`/`0.7190002817213534`, and the paired
  descriptive mean difference was `0.06014679867029188` (10/10 positive).
  Preserved the original `outputs/upgd_ipmnist/results.v1.json`, whose strict
  failure is limited to its 10-vs-20-seed note, and added the structurally valid
  `outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json` plus
  current addendum `outputs/upgd_ipmnist/nonpromoting_receipt.v2.json`, which
  preserves `nonpromoting_receipt.v1.json` byte-for-byte. They remain permanently
  nonpromoting: 10 rather than 20 published seeds, documented
  stream/logging/numeric deviations, no execution-time
  source/full-closure/command/environment/data binding, and an AdamW
  figure-read-off gap of about `+0.039`. No inferential, SOTA, or Alberta Plan
  claim follows; a fresh source-bound full-seed run is required.

### Changed

- Narrowed README and PrototypeAgent claims: the repository contains
  mechanisms across the Alberta Plan, not a completed twelve-step agent.
- Replaced the package import docstring's all-steps-Complete table with the
  evidence-level status and relabeled the legacy `alberta-evidence-gate` as an
  artifact-availability check rather than scientific validation.
- Corrected STOMP's discounted differential semi-MDP reward/baseline
  accounting, primitive-action credit, option-model planning, OaK transition
  ownership and active-option curation, and Prototype dream isolation.
- Added held-out recurring-feature, world-model decision-fidelity, and
  recurring multi-agent evaluations. Each documents its restricted claim;
  none is an end-to-end Alberta Plan result.
- Added a fail-closed held-out IA artifact. Its v1 run remains a valid
  scientific rejection: causal reward uplift and augmentation controls pass,
  but action-changing intervention prevalence is 8.73% versus the frozen 10%
  gate. A consumed-seed replay reproduced every v1 scientific field exactly at
  the time and remains explicitly nonpromoting; later `average_reward.py` drift
  now makes live current-source compatibility invalid. The old threshold was
  not retuned.
- Restored the original narrow FTL decision-fidelity claim through a
  fail-closed historical/current-source compatibility chain. The chain pins
  the original bytes and scientific digest, reconstructs acceptance from
  primitive rows, requires an exact current-source replay on consumed seeds
  30–59, and permits drift only in the artifact-builder hash. The replay is
  nonpromoting, and the absence of the exact historical builder source is
  recorded as a source-recoverability limitation.
- Reclassified the context-inference, TD-target-discovery,
  Prototype-retention, and historical wave-3 confirmation suites as
  development/nonpromoting evidence. Their inspected seeds, oracle or
  hand-built context paths, frozen feature bank, unequal resource budgets,
  and missing strict artifacts or component ablations remain explicit.
- Added separate conventional option-return and expected-duration TD heads
  plus a deterministic Step 5 renewal diagnostic. The diagnostic reaches L1:
  it shows that return/duration ranks a fast option correctly where return
  alone does not, but uses supplied options and features and is not a promoted
  semi-Markov control result.
- Added `alberta-evidence-status`, a machine-readable index that invokes each
  strict promoted-artifact validator and distinguishes accepted evidence,
  valid scientific rejections, missing runs, and invalid artifacts. The index
  records frozen commands, protocols, configurations, seeds, thresholds,
  artifact and source hashes, environment provenance, limitations, and
  validation timestamps. Its explicit dirty-state policy requires registered
  source hashes to match while recording unrelated worktree changes. Unit and
  smoke evidence is structurally barred from scientific promotion, with
  pytest lanes registered for unit, integration, scientific, and development
  work. The index explicitly is not an Alberta Plan completion certificate.

### Fixed

All fixed failing-test-first during this campaign:

- SARSA dead-λ for control heads.
- STOMP environment-reward grounding and its idle-update leak.
- OaK curation zeroing instead of re-initializing replaced options, and
  step-0 eviction.
- IA exo-cortex crediting its own actions instead of the partner's.
- Step-7 pre-warmup RNG freeze.
- Off-policy Horde ρ² trace composition and the GTD correction term's
  missing ρ.
- `reset_dormant_neurons` optimizer-pytree corruption.
- Baseline optimizers missing from the config registry.
- Prototype-basis recycled slots inheriting stale readouts.
- Compositional raw-index aliasing.
- UPGD-memory blend-logit gradient bias.
- Mann-Whitney rank-biserial sign inversion in the statistics utilities.

### Removed

- Removed the legacy root-`benchmarks` import shim from
  `alberta_framework/__init__.py`. It registered any importable top-level
  `benchmarks` package under the `alberta_framework.benchmarks` name, which
  could shadow the real packaged subpackage once one existed. The real
  subpackage is now imported eagerly and always wins
  (`tests/test_benchmarks_shim.py`).

### Packaging

- Version 0.27.0 across `pyproject.toml`, `alberta_framework.__version__`,
  and `CITATION.cff` (which had been stuck at 0.17.1).
- Registered the five evidence CLIs as console scripts:
  `alberta-ia-evidence`, `alberta-multiagent-evidence`, `alberta-ftl-evidence`,
  `alberta-recurring-feature-evidence`, and `alberta-scale-robust-evidence`.
- The sdist now includes `RESEARCH_STATUS.md` and
  `CONTINUAL_LEARNING_EVIDENCE.md`.
- ruff and mypy now target Python 3.12, matching `requires-python`.
- README rewritten: streams/testbeds, new mechanisms, the evidence registry,
  and benchmark lanes; removed dead CI/PyPI badges and the dead hosted-docs
  section; the canonical repository URL is now
  `https://github.com/lalalune/alberta` (see `VENDORING.md` for fork status).
- `VENDORING.md` rewritten to describe this tree as a development fork with a
  divergence summary; `CLAUDE.md`/`AGENTS.md` contributor guides added.

### Tests

- 2,641 passed and 55 skipped in the last full pre-release run, up from
  1,901 at 0.26.0; the suite now collects 3,284 tests across the `unit`,
  `integration`, `scientific`, and `development` marker lanes.

## [0.26.0] - 2026-05-22

### Added

- **GRU recursive perception for PrototypeAgent (Step 8a)** — fixed-weight echo-state
  GRU augments raw observations with recurrent hidden state before passing to all
  downstream components (OaK, Horde, world model). Glorot-uniform weight init,
  zero hidden init, pure-functional `_gru_step`. Controlled via
  `GRUPerceptionConfig(observation_dim, hidden_dim)` in `PrototypeAgentConfig`.
  12 new tests cover config validation, weight shapes, hidden dynamics, and
  augmented-obs routing. (`alberta_framework/core/prototype_agent.py`,
  `alberta_framework/core/__init__.py`)

- **Step 9 prioritized dreaming** — multi-step imagined rollouts with scored
  candidate selection. `score_dream_candidates` picks the most surprising/useful
  anchor state from `dream_candidate_count` random candidates; rollouts proceed
  for `dream_rollout_horizon` steps under the behavior model. BehaviorModel now
  tracked in `Step9DreamingState` and updated every real step.
  (`alberta_framework/steps/step9.py`)

- **Intelligence Amplification (Step 12) now exported from public API** —
  `ExoCerebellumAgent/Config/State`, `ExoCortexAgent`, `IAAgent/Config/State/
  UpdateResult/ArrayResult`, `RecommendationProtocolConfig/State/Result`,
  `init_recommendation_protocol_state`, and `update_recommendation_protocol`
  are now importable directly from `alberta_framework.core`.
  (`alberta_framework/core/__init__.py`)

### Fixed

- **`PrototypeAgentConfig` now validates `world_model.observation_dim`** when
  `gru_perception` is set, ensuring the world model's observation dimension
  matches `gru_perception.augmented_dim()`. Previously only `oak.observation_dim`
  was validated. (`alberta_framework/core/prototype_agent.py`)

- **`out_of_class_results.json` artifact restored** — reconstructed from the
  completed `out_of_class_SUMMARY.md` (30 seeds, 3 streams; original JSON was
  lost). Step 2 evidence gate now passes.
  (`outputs/step2_canonical/out_of_class_results.json`)

### Tests

- 1901 tests pass (up from 1900); new test `test_world_model_dim_mismatch_raises`
  covers the GRU + world-model dimension validation.

## [0.25.0] - 2026-05-21

### Added

- **CartPole FA Dyna benchmark** (Step 7) — 10-seed 5000-step comparison of
  linear-model Dyna vs real-only DifferentialSARSA on CartPole-v1 continuing.
  Result: ceiling effect — both agents achieve optimal reward=1.000 on all 10
  seeds; linear world model is stable (no degradation) but CartPole is too easy
  to reveal planning benefit. Faster benchmark (JIT-wrapped update functions,
  module-level agent objects) runs in 26 s vs the original 135+ min.
  (`benchmarks/step7_cartpole_dyna.py`, `outputs/step7_cartpole_dyna/`)

- **`NonlinearQHordeActorCriticAgent`** — action-value Horde critic with
  expected-SARSA targets, one control head per action; exported from the
  package. 10-seed variant search shows best variant ties Q at +0.0 improvement
  vs SARSA's +12.0 on catch — action-value critic substitution ruled out as
  Step 4 closure.

- **Step 4 probe suite** — adaptive-ObGD NLHAC at 500 and 1000 steps (both
  regress catch: -1.4 / -6.0 vs Q while SARSA is +6.6 / +13.4); wider (64,64)
  actor/critic NLHAC (catch -2.0 vs SARSA +9.33); rules out three more
  approaches to close the AC-vs-SARSA catch gap.

- **`rlsecd_external_audit.py`** — reproducible script checking availability
  of external `rlsecd` / `chronos-sec` sibling repos; result embedded in
  Step 3 solution gate.

### Fixed

- **Step 7 CartPole benchmark JIT regression** — original `step7_cartpole_dyna.py`
  created new agent/model Python objects per seed, forcing JAX to re-trace the
  planning scan body on every step; rewritten to use module-level agent/model
  objects and explicit `jax.jit(static_argnums=...)` wrappers.

## [0.24.0] - 2026-05-21

### Added

- **PrototypeAgent CartPole composition smoke** — a 5-seed continuing-control
  run exercised the configured composition without non-finite weights. Both
  flat DifferentialSARSAAgent and PrototypeAgent reached the reward ceiling
  (mean 1.000), so this was a wiring/stability check, not evidence that the
  composed mechanisms helped or that all twelve steps were integrated
  (`benchmarks/prototype_end_to_end.py`, results in
  `outputs/prototype_end_to_end/`)

### Added (continued)

- **`AdaptiveObGDBounding`** (Elsayed et al. 2024, Appendix B) — ObGD global
  bounding followed by per-weight RMS normalisation; registered in
  `_BOUNDER_REGISTRY` and exported from `alberta_framework`; 3 tests added to
  `tests/test_config_serialization.py`

### Fixed

- **PrototypeAgent dreaming JIT regression** — guarded dreaming scan closure
  was redefined on every `update()` call, causing JAX to retrace the XLA
  computation graph each step (~720 ms/step); extracted to
  `PrototypeAgent._run_dreams()` with `@functools.partial(jax.jit, static_argnums=(0,))`
  reducing to ~8 ms/step (94× speedup)

## [0.23.0] - 2026-05-21

### Added

- **Step 11 OaK curation benchmark** — 10-seed 6-state chain proves utility
  tracking detects and replaces counterproductive options; post-curation
  avg-reward recovers to 0.935 (8/10 seeds ≥ 0.70) from mean 0.70 pre-curation
  (`benchmarks/step11_oak_curation.py`, results in `outputs/step11_oak/`)

- **Step 12 IA augmentation benchmark** — 5-seed demonstration that
  exo-cerebellum MSE ≈ 0 (vs zero-baseline 0.167) and cortex recommendation
  accuracy 60% (>50% random) on 6-state chain
  (`benchmarks/step12_ia_augmentation.py`, results in `outputs/step12_ia/`)

- **Neuron utility tracking** — per-hidden-unit EMA of gradient L2 norm for
  dormant-neuron detection in long-running continual agents
  - `MLPLearner(track_neuron_utility=True, neuron_utility_decay=0.99)` stores
    `MLPLearnerState.neuron_utility: tuple[Array, ...] | None` (one `(h_i,)`
    array per hidden layer, None when disabled)
  - `MLPLearner.dormant_neuron_fraction(state, threshold)` returns the fraction
    of neurons below the utility threshold
  - `MLPLearner.reset_dormant_neurons(state, key, threshold)` re-initialises
    incoming weights, eligibility traces, and optimizer states for dormant
    neurons; zeroes outgoing weights from next layer to prevent signal injection
  - Config roundtrip via `to_config()` / `from_config()` includes new fields
  - 11 tests covering shapes, EMA dynamics, dormancy counts, reset, and serialisation

## [0.22.0] - 2026-05-21

### Added

- **Autostep-for-actor** — per-weight adaptive step-sizes for all actor MLPs
  - `NonlinearHordeActorCriticAgent` actor now uses `Autostep.init_for_shape` /
    `update_from_gradient`; fixed scalar `actor_step_size` removed from config
  - `AverageRewardHordeActorCriticAgent` receives the same upgrade; per-weight
    `AutostepParamState` stored in `AverageRewardHordeActorCriticState`
  - Both agents accept an optional `actor_optimizer: Autostep | None` constructor arg
    (default `Autostep(initial_step_size=0.05)`) and expose `actor_optimizer` property
  - Config roundtrip includes `actor_optimizer` serialisation

- **Nonlinear STOMP / OaK base Q-function** — replaces hard-coded linear weights
  - `STOMPState.base_learner_state: MultiHeadMLPState` replaces `base_q_weights` /
    `base_traces`; the underlying `MultiHeadMLPLearner` has `n_heads = n_total_actions`
  - `STOMPConfig.base_hidden_sizes: tuple[int, ...] = ()` enables nonlinear trunks;
    the empty-tuple default preserves previous linear behaviour exactly
  - Discounted semi-MDP differential Q target
    (`R_o^γ - r̄·Σ_{k=0}^{T_o-1}γ^k + γ^{T_o}·max Q(s')`) computed via
    NaN-masked `MultiHeadMLPLearner.update` — compatible with both linear and
    MLP paths; at `γ=1` the baseline mass reduces to the raw duration `T_o`
  - OaK curation resets the curated option's head (weights, biases, traces, optimizer
    states) inside the `MultiHeadMLPState` rather than zeroing a raw weight slice
  - `STOMPAgent.base_q_values(state, obs)` and `OaKAgent.base_q_values(state, obs)`
    expose Q-value computation through the agent API, used by `ExoCortexAgent.recommend`
    and `PrototypeAgent.act` (both now robot-ready for high-dimensional observations)
  - `feature_to_subtask_specs` in `prototype_agent.py` handles both linear (head-weight
    stack) and nonlinear (first trunk-layer proxy) feature-importance extraction

## [0.21.0] - 2026-05-21

### Added

- **PrototypeAgent** — experimental composition surface for mechanisms
  associated with Steps 1–12
  - `PrototypeAgentConfig`: minimal defaults (just `n_primitive_actions` + `observation_dim`)
  - Single `.update()` integrates world model, buffer, OaK, dreaming, Horde, and IA
  - `feature_to_subtask_specs`: automatic subtask extraction from OaK Q-weight importances
  - `run_prototype_scan` / `run_prototype_smoke`: JIT-compiled loop and validity probe
  - 50 tests covering all components and 200-step fineness

## [0.20.0] - 2026-05-21

### Added

- **Steps 11 and 12: OaK and Intelligence Amplification**
  - `OaKAgent`: extends STOMP with utility EMA, curation, and option keyboard (Barreto et al.)
  - `ExoCerebellumAgent` / `ExoCortexAgent` / `IAAgent`: paired cerebellum + cortex
    augmenting a partner agent's observations and action recommendations
  - 32 OaK tests + 30 IA tests

## [0.19.0] - 2026-05-20

### Added

- **Step 10: STOMP temporal abstraction**
  - `STOMPAgent`: subtask-defined options, intra-option differential Q, option outcome models
  - `SubtaskSpec` / `STOMPSpecArrays` / `STOMPState`: JAX-compatible option state
  - 36 tests; seeded benchmark proves STOMP options accelerate control ~10x vs flat
  - `Step10STOMPConfig` production facade

## [0.18.0] - 2026-05-19

### Added

- **Steps 5–9: Average-reward control, world models, and dreaming**
  - `DifferentialTDLearner` / `DifferentialSARSAAgent` / `DifferentialGTDLearner`:
    continuing average-reward prediction and control (Steps 5–6)
  - `AverageRewardHordeLearner` / `AverageRewardHordeActorCriticAgent`:
    nonlinear shared-trunk Horde for differential GVF prediction and actor-critic
  - `OneStepWorldModel` / `ActionConditionedWorldModel`: reward + next-obs prediction (Step 8)
  - `GuardedDreamer` + `RecentObservationBuffer`: error-gated dreaming with real-state anchors (Step 9)
  - `Step7DynaConfig`: real-transition update + fixed `planning_steps` Dyna backups
  - Seeded benchmarks: Step 7 prioritized Dyna improves final-window reward from 0.92 → 1.00;
    six-state chain Dyna wins cumulative reward 8/10 seeds (+41.7%)
  - RiverSwim benchmark (Step 6): 10/10 seeds, 97.5% right-action rate

## [0.17.1] - 2026-04-10

### Added

- **Autostep-for-GTD(λ)** — per Kearney et al. (2019)
  - `AutoTDIDBD` optimizer with per-weight step-size adaptation for TD learning
  - Eligibility traces integrate with Autostep's normalizer and overshoot prevention

### Fixed

- Flax dependency added and version pinned in `pyproject.toml`

## [0.17.0] - 2026-04-05

### Added

- **Replacing traces** — `TraceMode.REPLACING` on `MultiHeadMLPLearner`
  - Replaces stale trace magnitude on re-visit rather than accumulating
  - Configurable per-head alongside `TraceMode.ACCUMULATING`
  - Trace-bounding integration: replaced traces scaled by ObGD bounding factor

## [0.16.0] - 2026-03-15

### Added

- **SARSA agent (Step 4a)** — on-policy control via Horde architecture
  - `SARSAAgent`: wraps `HordeLearner` with epsilon-greedy action selection and SARSA target computation
  - `SARSAConfig`: configuration for n_actions, gamma, epsilon schedule
  - `SARSAState`, `SARSAUpdateResult`: immutable state and result types
  - `run_sarsa_episode`: Python loop for episodic Gymnasium environments
  - `run_sarsa_continuing`: continuing mode with pseudo-boundary handling (daemon-style)
  - `run_sarsa_from_arrays`: JIT-compiled `jax.lax.scan` for pre-collected data (security-gym)
  - Gumbel trick tie-breaking for uniform action selection among equal Q-values
  - Linear epsilon decay schedule (configurable start, end, decay steps)
  - Optional prediction demons coexist with control demons in the same Horde
  - Config serialization via `to_config()` / `from_config()` roundtrip
  - 30 new tests covering init, action selection, update logic, epsilon decay, bounding, serialization, and scan loop
  - Example: `examples/The Alberta Plan/Step4/sarsa_cartpole.py`
  - Documentation: `docs/guide/sarsa-control.md`

- **Trunk trace guard** — validation preventing `gamma * lamda > 0` on `MultiHeadMLPLearner` with hidden layers
  - VJP backward pass folds error into trunk cotangent before trace accumulation; only correct when traces reset each step
  - Linear baseline (`hidden_sizes=()`) allows any gamma/lamda
  - `HordeLearner` enforces trunk gamma=0 by design (per-head trace decay only)
  - Expanded docstrings on `MultiHeadMLPLearner` and `HordeLearner` explaining the constraint

## [0.10.0] - 2026-02-27

### Added

- **Hybrid optimizer (`head_optimizer`)** — separate optimizer for trunk vs head layers on `MLPLearner` and `MultiHeadMLPLearner`
  - `MLPLearner(head_optimizer=...)`: output layer uses `head_optimizer`, hidden layers use `optimizer`
  - `MultiHeadMLPLearner(head_optimizer=...)`: all prediction heads use `head_optimizer`, trunk uses `optimizer`
  - Enables stable LMS+ObGD for non-convex hidden layers with adaptive Autostep for the linear output head
  - Backwards compatible: `head_optimizer=None` (default) keeps all layers on the same optimizer
  - 10 new tests (6 MLPLearner, 4 MultiHeadMLPLearner)

## [0.9.0] - 2026-02-22

### Added

- **Agent lifecycle tracking** — `step_count`, `birth_timestamp`, `uptime_s` on all learner states
  - `LearnerState`, `MLPLearnerState`, `TDLearnerState`: new fields with backward-compatible defaults
  - `MultiHeadMLPState`: added `birth_timestamp` and `uptime_s` (already had `step_count`)
  - `step_count` incremented inside `update()` (JAX-traced, safe in `jax.lax.scan`)
  - `birth_timestamp` set at `init()`, immutable across updates
  - `uptime_s` accumulated after each `jax.lax.scan` completes in all learning loops
  - All learning loop functions stamp uptime: `run_learning_loop` (simple + tracking), `run_learning_loop_batched`, `run_mlp_learning_loop` (simple + tracking), `run_mlp_learning_loop_batched`, `run_td_learning_loop`, `run_multi_head_learning_loop`, `run_multi_head_learning_loop_batched`, `learn_from_trajectory`
- `agent_age_s(state)` — wall-clock seconds since agent birth
- `agent_uptime_s(state)` — cumulative active seconds inside learning loops
- 28 new lifecycle tracking tests across all learner types

## [0.8.1] - 2026-02-21

### Added

- **bsuite benchmark integration** — bridges framework to bsuite for standardized RL diagnostics
  - `ContinuingWrapper`: converts episodic envs to continuing streams (Alberta Plan Step 6)
  - `AlbertaAgent`: bridges bsuite `Agent` ABC to `MultiHeadMLPLearner` with Q-learning
  - Three agent factories: Autostep+ObGD, LMS+ObGD, Adam (haiku/optax external baseline)
  - Hyperparameter configs with standard `(64, 64)` and bottleneck `(16, 16)` variants
  - `run_single.py` / `run_sweep.py` CLIs with `--continual-sequence` and `--use-scythe` flags
  - Analysis module: result loading, comparison plots, representation analysis, summary tables
  - Representation utility logging: per-weight step-sizes, trunk trace magnitudes, per-head metrics
  - 22 tests covering wrapper, agents, factories, representation logging, and integration

### Dependencies

- Added `[bsuite]` optional dependency group (dm-env, optax, dm-haiku, plotnine)

## [0.8.0] - 2026-02-16

### Added

- **`MultiHeadMLPLearner`** — shared-trunk MLP with multiple prediction heads for multi-task continual learning
  - VJP-based gradient computation with accumulated cotangents (single backward pass through trunk)
  - NaN target masking for selective head activation (inactive heads skip gradient updates)
  - Composable: accepts any `Optimizer`, optional `Bounder`, optional `Normalizer`
  - Eligibility traces managed per-head and per-trunk-layer
- `MultiHeadMLPState`, `MultiHeadMLPUpdateResult`, `MultiHeadLearningResult`, `BatchedMultiHeadResult` types
- `run_multi_head_learning_loop()` — `jax.lax.scan` over observation/target arrays with NaN masking
- `run_multi_head_learning_loop_batched()` — `jax.vmap` over initialization keys for multi-seed parallelization
- `multi_head_metrics_to_dicts()` — convert array metrics to per-head dicts for online use

## [0.7.3] - 2026-02-09

### Added

- `MLPLearner(use_layer_norm=False)` — toggle parameterless LayerNorm for ablation studies (default `True`, backwards-compatible)

## [0.7.2] - 2026-02-08

### Fixed

- IDBD operation ordering now matches Sutton 1992 Figure 2: meta-update first, then NEW alpha for weight and trace updates

### Changed (Breaking)

- Autostep rewritten to match Mahmood et al. 2012 Table 1 exactly:
  - `v_i` now tracks meta-gradient magnitude `|δ*x*h|` (was primary gradient `|δ*x|`)
  - `v_i` uses self-regulated EMA (Eq. 4), not `max(|grad|, v*τ)`
  - Overshoot prevention via `M = max(Σ α_i*x_i², 1)` (Eq. 6-7)
  - Trace decay includes `x²`: `h_i = h_i*(1 - α_i*x_i²) + α_i*δ*x_i`
  - Normalizers and traces initialized to 0 (was 1 and 0)
  - Normalization only applies to meta-update, not to weight/trace updates
- `Autostep(normalizer_decay=...)` renamed to `Autostep(tau=...)`, default changed from 0.99 to 10000.0
- `AutostepState.normalizer_decay` renamed to `AutostepState.tau`
- `AutostepParamState.normalizer_decay` renamed to `AutostepParamState.tau`

### Added

- `Autostep.update_from_gradient()` now accepts optional `error` parameter for full paper algorithm in MLP path
- `Optimizer.update_from_gradient()` base signature accepts optional `error` parameter

## [0.7.1] - 2026-02-07

### Added

- **`AGCBounding`** — Adaptive Gradient Clipping (Brock et al. 2021) as a `Bounder` ABC, per-unit clipping scaled by weight norm
- `_unitwise_norm()` helper for unit-wise L2 norm computation (1D: abs, 2D+: norm over fan-in axes)

## [0.7.0] - 2026-02-07

### Changed (Breaking)

- Removed `NormalizedLinearLearner`, `NormalizedMLPLearner` — use `LinearLearner(normalizer=...)` and `MLPLearner(normalizer=...)` instead
- Removed `run_normalized_learning_loop`, `run_normalized_learning_loop_batched`, `run_mlp_normalized_learning_loop`, `run_mlp_normalized_learning_loop_batched` — unified into `run_learning_loop` and `run_mlp_learning_loop` (detect normalization from learner)
- Removed `NormalizedLearnerState`, `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedNormalizedResult`, `BatchedMLPNormalizedResult`, `MLPObGDState` types
- `MLPLearner` no longer accepts `kappa` parameter — use `bounder=ObGDBounding(kappa=2.0)` instead

### Added

- `Bounder` ABC and `ObGDBounding` for decoupled update bounding (composable with any optimizer)
- `AutostepParamState` for per-parameter Autostep optimization (arbitrary array shapes)
- `Optimizer.init_for_shape()` and `Optimizer.update_from_gradient()` for shape-agnostic optimization (LMS, Autostep)
- `MLPLearner` now accepts composable `optimizer`, `bounder`, and `normalizer` parameters
- `LinearLearner` now accepts optional `bounder` and `normalizer` parameters
- Unified learning loops: 4 functions instead of 8 (linear + MLP, each with single + batched)

### Fixed

- mypy override errors — base class `init_for_shape`/`update_from_gradient` use `Any` since return type varies by subclass

## [0.6.1] - 2026-02-07

- Version bump only

## [0.6.0] - 2026-02-07

### Changed (Breaking)

- Replaced `OnlineNormalizer`, `NormalizerState`, `create_normalizer_state` with `Normalizer` ABC hierarchy

### Added

- `Normalizer` ABC with generic `StateT` constraint, following the `Optimizer[StateT]` pattern
- `EMANormalizer` — exponential moving average normalization (renamed from `OnlineNormalizer`, corrected docstrings)
- `WelfordNormalizer` — true Welford's algorithm with Bessel's correction for stationary distributions
- `EMANormalizerState`, `WelfordNormalizerState`, `AnyNormalizerState` types
- `NormalizedLinearLearner` now accepts any `Normalizer` subclass
- `NormalizedMLPLearner` — wraps `MLPLearner` with online normalization (EMA or Welford)
- `NormalizedMLPLearnerState`, `NormalizedMLPUpdateResult`, `BatchedMLPNormalizedResult` types
- `run_mlp_normalized_learning_loop()` with optional `NormalizerTrackingConfig`
- `run_mlp_normalized_learning_loop_batched()` for vmap-based multi-seed normalized MLP training

## [0.5.3] - 2026-02-06

### Added

- `run_mlp_learning_loop_batched()` for vmap-based multi-seed MLP training with `BatchedMLPResult` return type

## [0.5.2] - 2026-02-06

### Fixed

- Resolved mypy type error in `MLPLearner` z_sum computation — replaced `sum()` over JAX arrays with explicit `jnp.array(0.0)` accumulator

## [0.5.0] - 2026-02-06

### Added

- **ObGD Optimizer**: Observation-bounded Gradient Descent for overshooting prevention (Elsayed et al. 2024). Dynamically bounds effective step-size based on error magnitude and trace norms. Works as a linear optimizer (`ObGD`) and within the MLP learner.
- **MLPLearner**: Multi-layer perceptron with ObGD optimizer for nonlinear function approximation in the streaming setting. Architecture: `Input -> [Dense -> LayerNorm -> LeakyReLU] x N -> Dense(1)`. Configurable depth via `hidden_sizes` tuple.
- **Sparse Initialization**: `sparse_init()` function implementing LeCun-scale initialization with per-neuron sparsity (default 90%), following Elsayed et al. 2024.
- **`run_mlp_learning_loop()`**: JIT-compiled MLP training via `jax.lax.scan`, same pattern as existing linear learning loops.
- **MLP Types**: `MLPParams`, `MLPObGDState`, `MLPLearnerState`, `MLPUpdateResult` chex dataclasses.
- **ObGD Types**: `ObGDState` chex dataclass with `create_obgd_state()` factory.
- **Step 2 Example**: `linear_vs_mlp_comparison.py` comparing LinearLearner+Autostep vs MLPLearner+ObGD on RandomWalk, AbruptChange, and DynamicScaleShift streams.

### Notes

- ObGD defaults to `gamma=0, lamda=0` for supervised learning (traces = current observation). Nonzero values enable eligibility traces for future RL use (Steps 3-4).
- MLP implementation is self-contained (no Flax/Haiku dependency). Uses `jax.grad` for backpropagation and parameterless layer normalization.
- The `Optimizer` generic constraint now includes `ObGDState`, so `ObGD` can be used with `LinearLearner` as well.

## [0.4.0] - 2026-02-04

### Added

- TD-IDBD optimizer for temporal-difference learning with per-weight adaptive step-sizes and eligibility traces (Kearney et al., 2019)
- AutoTDIDBD optimizer with AutoStep-style normalization for improved stability
- `TDLinearLearner` class for linear value function approximation in TD learning
- `run_td_learning_loop()` for JIT-compiled TD learning via `jax.lax.scan`
- TD state types: `TDIDBDState`, `AutoTDIDBDState`, `TDLearnerState`, `TDTimeStep`
- `TDStream` protocol for TD experience streams

## [0.3.2] - 2026-02-03

### Fixed

- Relaxed test tolerance in batched vs sequential comparison tests (`rtol=1e-5`) to account for floating-point differences between vmap and sequential execution paths
- Added `ignore = ["F722"]` to ruff config for jaxtyping shape annotation syntax that ruff doesn't understand
- Removed unused `PRNGKeyArray` import from `core/types.py`

## [0.3.0] - 2026-02-03

### Added

- Migrated all state types from NamedTuple to `@chex.dataclass(frozen=True)` for DeepMind-style JAX compatibility
- jaxtyping shape annotations for compile-time type safety (`Float[Array, " feature_dim"]`, `PRNGKeyArray`, etc.)
- Updated test suite to use chex assertions (`chex.assert_shape`, `chex.assert_tree_all_finite`, `chex.assert_trees_all_close`)

### Dependencies

- Added `chex>=0.1.86` and `jaxtyping>=0.2.28` as required dependencies
- Added `beartype>=0.18.0` as optional dev dependency for runtime type checking

## [0.2.2] - 2026-02-02

### Fixed

- mypy type errors in `run_learning_loop_batched` and `run_normalized_learning_loop_batched` functions
- Added `typing.cast` to properly handle conditional return type unpacking in batched learning loops

## [0.1.0] - 2026-01-19

### Added

- **Core Optimizers**: LMS (baseline), IDBD (Sutton 1992), and Autostep (Mahmood et al. 2012) with per-weight adaptive step-sizes
- **Linear Learners**: `LinearLearner` and `NormalizedLinearLearner` with pluggable optimizers
- **Scan-based Learning Loops**: JIT-compiled training with `jax.lax.scan` for efficiency
- **Online Normalization**: Streaming feature normalization with exponential moving averages
- **Experience Streams**: `RandomWalkStream`, `AbruptChangeStream`, `CyclicStream`, `SuttonExperiment1Stream`
- **Gymnasium Integration**: Trajectory collection and learning from Gymnasium RL environments
- **Step-Size Tracking**: Optional per-weight step-size history recording for meta-adaptation analysis
- **Multi-Seed Experiments**: `run_multi_seed_experiment` with optional parallelization via joblib
- **Statistical Analysis**: Pairwise comparisons, confidence intervals, effect sizes (requires scipy)
- **Publication Visualization**: Learning curves, bar charts, heatmaps with matplotlib
- **Export Utilities**: CSV, JSON, LaTeX, and Markdown table generation
- **Documentation**: MkDocs-based documentation with auto-generated API reference

### Notes

- Requires Python 3.13+
- Implements Step 1 of the Alberta Plan: demonstrating that IDBD/Autostep can match or beat hand-tuned LMS
- All state uses immutable NamedTuples for JAX compatibility
- Follows temporal uniformity principle: every component updates at every time step
