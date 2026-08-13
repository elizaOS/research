# Alberta Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A JAX-based research framework for investigating
[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173). The
repository contains mechanisms spanning all 12 steps, but the integrated
continual-learning result is **not yet complete**: several defining links and
reproducible benchmark artifacts are still missing.

Most learners support online, per-transition updates with no replay phase.
Several `PrototypeAgent` components are optional and disabled by default, so a
default construction does not exercise every mechanism on every transition.
See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for the evidence matrix and
fail-closed completion criteria.

## Continual-learning screening record (development-grade, nonpromoting)

The recorded IPMNIST screening measurements below are
**development-grade, permanently nonpromoting** observations under the
repository's evidence rules. A 2026-08-02 read-only audit found that this is
not a reproducible completed campaign: the checked-in proxy validator reports
`proxy_validated: false` because all three UPGD control prefixes disagree with
their claimed full-horizon references (maximum per-task discrepancies
`0.0084`–`0.0096`), the aggregate summary contains 132 of the 144 current
screening shards, and the round-2 driver fails on a nonexistent shard field.
The round-2 means and the 69 full-horizon confirmation means can be recomputed
from the stored curves, but the v1 shards bind neither source nor command nor
dataset bytes. They are therefore historical development observations, not an
authenticated result for the current source tree and not a scientific-evidence
or SOTA claim. Protocol: input-permuted MNIST from the
UPGD paper (Elsayed & Mahmood, ICLR 2024) — 1M examples one-per-step,
permutation every 5,000 steps, 200 tasks, 300×150 ReLU MLP, average online
accuracy. Baseline: our 10-seed published-config UPGD-W reproduction at
**0.7791**.

Stored full-horizon (200-task) development means relative to that baseline
(`outputs/ipmnist_screening/confirm_full/`):

| Arm | Seeds | Mean online acc. | Scoping |
|---|---:|---:|---|
| `adamw_cbp_r3e4` (AdamW + CBP, tuned replacement rate) | 3 | 0.80126 ± 0.00022 | protocol-pure |
| `adamw_cbp` (AdamW + continual-backprop recycling) | 10 | 0.79876 ± 0.00009 | protocol-pure |
| `upgd_w_wd0005` (published method, tuned decay) | 10 | 0.78431 ± 0.00014 | protocol-pure (tuned) |
| `upgd_l2init` | 3 | 0.78042 ± 0.00030 | protocol-pure |
| `sigma0_shiftnorm_d099` (decay-0.99 EMA norm + per-feature shift-triggered re-conditioning, no perturbation) | 3 | **0.86459 ± 0.00028** | protocol-extended |
| `l2init_ema_norm` (decay-0.99 EMA norm + plain SGD + decay toward init — L2-Init; no gate, no perturbation) | 3 | **0.86457 ± 0.00018** | protocol-extended |
| `sigma0_ndecay099` (EMA norm decay 0.99, no perturbation) | 3 | 0.86245 ± 0.00034 | protocol-extended |
| `sgd_ema_norm_d099` (decay-0.99 EMA norm + plain SGD + decay; comparison-wave base) | 3 | 0.86168 ± 0.00016 | protocol-extended |
| `upgd_ema_norm` (UPGD-W + online EMA input normalization) | 10 | 0.8536 ± 0.0001 | protocol-extended |
| `upgd_ema_norm_sigma0` (same, perturbation off) | 3 | 0.85051 ± 0.00025 | protocol-extended |
| `upgd_ema_norm_wd0005` | 3 | 0.84745 ± 0.00008 | protocol-extended |
| `sgd_ema_norm` (bare SGD + decay + normalization) | 3 | 0.83991 ± 0.00007 | protocol-extended |

"Protocol-pure" arms keep the published raw input encoding;
"protocol-extended" arms prepend online EMA input normalization — an
input-encoding change the published architecture does not include — and are
always reported on their own rows, never as the headline. The tuned arms
(`sigma0_shiftnorm_d099`, `l2init_ema_norm`, `sigma0_ndecay099`,
`adamw_cbp_r3e4`, `upgd_w_wd0005`) were selected on seeds 0-2 and are
confirmed on those same seeds, so their means carry selection bias of
unquantified (likely small) size (`outputs/ipmnist_screening/AUDIT.md` has
the full audit).
`sigma0_shiftnorm_d099` adds a purely observational per-feature shift
detector (fast/slow mean divergence — never a task-boundary oracle) that
resets the normalizer's anneal count where the input distribution moved
(+0.0021 over `sigma0_ndecay099` at 200 tasks, all three seeds positive).
`l2init_ema_norm` — from the reviewer comparison wave (published mechanisms
re-implemented behind our conditioning; `outputs/ipmnist_screening/`
`FINAL_REPORT.md`) — statistically ties it by a different mechanism: L2-Init
(Kumar et al.) decay toward the initial weights on a plain-SGD base with no
utility gate (+0.0021 over `sigma0_ndecay099`, all seeds positive; +0.0029
over the identical decay-toward-zero base, isolating the decay target as
the mechanism). The same wave measured the gate itself: at normalizer decay
0.99 the mechanism-free `sgd_ema_norm_d099` base gives back only 0.0008 of
the gate arm's 200-task mean — the utility gate is not load-bearing under
fast input conditioning on this stream (it remains load-bearing on
label-permuted EMNIST), while Weight Clipping (−0.011 vs base), FADE head
decay (−0.045), and SNR resets (churn to the raw baseline at one example
per step) all fail to add on top of conditioning.

The stored contrasts suggest the following descriptive decomposition; the
failed proxy/source audit means it is not an authenticated common-source
60→200-task mechanism decomposition:
**input conditioning +0.061** (`sgd_ema_norm` vs the 0.7791 reproduction),
**utility gate +0.011** (`upgd_ema_norm_sigma0` vs `sgd_ema_norm`), and
**perturbation +0.003 when normalization is present** (`upgd_ema_norm` vs
`sigma0`). The contrast on raw inputs: disabling the perturbation in the
published configuration (`upgd_w_sigma0`, 60-task proxy) *costs* −0.035 —
the noise IS load-bearing without normalization, and input conditioning
substitutes for it.

The frontier wave generated a sharper conditioning hypothesis: dropping the
normalizer's EMA decay from 0.999 to 0.99 (`sigma0_ndecay099`) adds another
+0.012 — the permutation boundary shifts input statistics instantly, and a
100-step EMA re-conditions ~10x faster after each switch. The symmetric
result (decay 0.9999 *loses* 0.007; hidden-layer normalization loses 0.019)
is consistent with input-tracking speed as the mechanism, but does not by
itself confirm that causal interpretation.

These numerical contrasts remain useful for generating hypotheses, but the
cross-horizon and mechanism language below is descriptive until a fresh
source-bound lifecycle reproduces it. Mechanism analysis and the
pre-registered outcome matrix are in
[CONTINUAL_LEARNING_THEORY.md](CONTINUAL_LEARNING_THEORY.md); the historical
arm-by-arm record is `outputs/ipmnist_screening/FINAL_REPORT.md`. None of
these results is promotable without a fresh source-bound preregistered run
under the v3 contract (`UPGD_IPMNIST_V3_RUNBOOK.md`).

## Evidence status at a glance

Registered scientific artifacts can be checked without rerunning their
protocols:

```bash
alberta-evidence-status
```

The command invokes each artifact's strict validator and returns `0` only when
every registered narrow claim is accepted, `1` for missing or valid rejected
evidence, and `2` for invalid evidence. Its manifest is an operational index,
not an Alberta Plan completion certificate. It resolves artifacts relative to
the repository checkout: the wheel and sdist deliberately exclude `outputs/`,
so from a pip-installed environment every claim reports missing and the
command exits `1`. Run it from a checkout.

`alberta-evidence-gate` is retained temporarily as a deprecated compatibility
alias for `alberta-evidence-status` and has the same manifest and exit-code
contract. Its former `--step` selector is rejected: the upstream Step 1/2
artifact generators and narrative-document tree are intentionally not shipped
in this fork, so there is no current Step 1-only or Step 2-only registry claim
to validate.

Pinned historical evidence includes a narrow L2 scale-robust pair-feature
package comparison on 30 exact namespace-derived fresh seeds. Its immutable
artifact is `outputs/scale_robust_feature/evidence.v2.json`, with scientific
digest
`c2fee922c04a59fe26b4b8c9cfa77ddd9198cfa2bc923f54fec14b649bd3bb2c`.
Median final-C savings was 5.933 and final-C tail MSE was 0.0387, with no
non-finite step. The result uses visible context, one fixed learner
initialization, and an exhaustive finite pair archive; primary versus legacy
also changes scale normalization and ObGD and adds 464 persistent bytes, so it
is a package comparison rather than causal attribution.

The live registry currently reports **all five registered claims `invalid`**
(overall `invalid`, exit `2`): registered implementation, artifact-builder,
and CLI source hashes have evolved since the pinned artifacts were produced.
That is the fail-closed design working as intended. The artifacts remain
immutable historical results and do not certify the current implementation.
Consumed-seed reruns are nonpromoting; renewed promotion requires rerunning
each frozen protocol to a new artifact path/schema with an untouched
preregistered seed schedule.

The IA v1 result remains a historical valid rejection at its frozen 10%
action-changing intervention threshold. Its prior consumed-seed compatibility
replay record is nonpromoting, and the live registry marks current-source
compatibility `invalid` after registered-source drift (initially
`average_reward.py`, since joined by further edits). The archived v1 result
is unchanged; it does not certify the current source.
The exact p=0.75/seeds-60–89 v2 lifecycle is an **unissued, permanently
development-only contract**: no plan, reservation, shard, run, or v2 artifact
has been produced. Its self-issued plan has no trusted external pre-run
chronology, so even a gate-passing run cannot become accepted evidence. A
future acceptance attempt needs a new schema, untouched seeds, and an external
chronology anchor.

The original narrow FTL decision-fidelity artifact records a historical
acceptance, but its strict historical-artifact/current-source compatibility
chain is now `invalid`: current drift extends beyond the once-permitted
artifact-builder-only difference into the fidelity and CLI sources. Its
consumed-seed replay is also nonpromoting. None of these narrow results is an
end-to-end Alberta Plan completion or certifies this working tree.

## Install

```bash
pip install alberta-framework
pip install alberta-framework[gymnasium]   # RL environment support
pip install 'alberta-framework[forager]'   # Forager continual-RL testbed
pip install alberta-framework[dev]         # tests, lint
```

Requires Python 3.12+, JAX 0.4+. The suite currently collects roughly 6,900
tests (`pytest tests`; markers `unit`, `integration`, `scientific`,
`development`).

Key documents:

- [RESEARCH_STATUS.md](RESEARCH_STATUS.md) — evidence levels (L0–L3), the
  requirement-to-evidence matrix, and the fail-closed completion gates.
- [RESEARCH_REPORT_AGE_OF_EXPERIENCE.md](RESEARCH_REPORT_AGE_OF_EXPERIENCE.md)
  — the 2026-08-02 research synthesis: SOTA verification, optimizer/benchmark
  landscape, Alberta Plan gap analysis, and the derived wave A–D experiment
  program.
- [CONTINUAL_DYAD_BENCHMARK.md](CONTINUAL_DYAD_BENCHMARK.md) — the staged,
  partly implemented HCCL causal-transaction subset and later HCCL-v1 dyad
  design, including their attribution, resource, and evidence-authority
  boundaries. A separate base-only bridge now starts two independently keyed
  external learned-state coordinators from the world's exact 16-channel rows,
  binds their cached primitives as `B=M=P`, and adopts the PP world successor
  plus both coordinator updates all-or-none. Its zero memory/planner contrasts
  are ablation facts, not delight or actor-gradient judgments. No integrated
  autonomous life or benchmark execution exists. Newer L0 rungs atomically
  compose both live stacks with context, HCCL, and a paired planner, then let a
  recurring Kondo route own `P`; those are mechanism contracts, not HCCL-v1.
- [FORAGER_MATCHED_V3_RUNBOOK.md](FORAGER_MATCHED_V3_RUNBOOK.md) — the live
  next-generation Forager matched-campaign contract (v3 schemas, SHA-pinned
  plans).
- [CONTINUAL_LEARNING_EVIDENCE.md](CONTINUAL_LEARNING_EVIDENCE.md) — the
  property-by-property evidence map, measured numbers, and the bug ledger.
- [FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md) — the arXiv:2605.01131 testbed
  integration, Alberta runner, and paired DQN/PPO/RTU-PPO comparison workflow.
- [CONTINUAL_IA_V2_RUNBOOK.md](CONTINUAL_IA_V2_RUNBOOK.md) — the unissued,
  development-only p=0.75/seeds-60–89 plan/reservation/shard/merge contract and
  its nonpromotion boundary.
- [UPGD_IPMNIST_V3_RUNBOOK.md](UPGD_IPMNIST_V3_RUNBOOK.md) — the strict
  namespaced UPGD IP-MNIST v3 execution contract (unissued; permanently
  nonpromoting).

## What's here

The Alberta Plan is a 12-step research programme for building continual AI.
This framework provides the following implementation surfaces:

| Steps | Focus | Key classes |
|-------|-------|-------------|
| 1 | Adaptive step-size prediction | `LinearLearner`, `IDBD`, `Autostep` |
| 2 | Nonlinear function approximation | `MLPLearner`, `ObGDBounding` |
| 3 | GVF predictions, Horde architecture, learned state, and balanced objectives | `HordeLearner`, `LearnableGRUStateBuilder`, `ComprehensiveStateObjectives`, `CausalStateObjectiveTargetProducer`, `PrototypeComprehensiveStateObjectives`, `PrototypeCausalStateObjectiveTargets` |
| 4 | Continual control (SARSA + actor-critic) | `SARSAAgent`, `ActorCriticAgent` |
| 5–6 | Average-reward continuing control | `AverageRewardHordeLearner`, `DifferentialSARSAAgent` |
| 7–8 | Dyna planning + guarded ensemble rollouts | `RealStateOneStepDyna`, `EnsembleShortRolloutPlanner`, `ImaginedRolloutSelectionGauge`, `AuthorizedImaginedRolloutActorCritic` |
| 9 | Guarded dreaming and prospective exploration | `GuardedDreamer`, `ProspectiveExploration` |
| 10 | Cumulant/subtask discovery + scheduled live STOMP temporal abstraction | `CumulantSubtaskDiscovery`, `CumulantOptionScheduler`, `STOMPAgent`, `PrototypeOptionAuthorityBridge` |
| 11 | OaK option keyboard (utility tracking + curation) | `OaKAgent` |
| 12 | Prototype-IA (exo-cerebellum + exo-cortex) | `PrototypeAgent` |

### Mechanism surfaces (development status)

Everything in this section is mechanism-level work — L0/L1 in the vocabulary
of [RESEARCH_STATUS.md](RESEARCH_STATUS.md) — with contract tests but no
promoted evidence. Several of these mechanisms serialize development-only
markers (`SCIENTIFIC_PROMOTION_ALLOWED = False` or equivalent) into their
config and checkpoint schemas and reject any payload claiming otherwise, so
promoting them requires new preregistered protocols and code, not just new
runs.

- **Composition and routing.** Causal `StateBuilder` variants (identity,
  fixed-trace, and online trainable gated recurrent) under one fixed-budget
  contract; the predict-before-update `LearningSignalEstimator`; the
  fixed-state `LearningValueRouter`, which keeps all eight learning-value
  channels independently validated and causally normalized and exposes only
  named consumer routes (no default sum). Its opt-in Prototype owner advances
  once per accepted real transition and supplies only the raw candidate-audit
  route to the state-builder audit; fixed-capacity `DualReplayMemory`
  and `ExperientialMemory`; and the explicit `PrototypeTransition` boundary.
  `DualReplayMemory` has no training or control integration of its own — its
  only consumer is the model-only rehearsal lane below.
- **Bounded WP1 development report.** A versioned in-memory
  `PrototypeContinualControlDevelopmentReport` runs `PrototypeAgent`, a
  running-reward bandit, and a frozen-action baseline on two consumed A/B/A
  development seeds with independent functional environment states. It embeds
  exact action/decision ownership, reconstructing evaluator reports,
  opportunity counts, logical state bytes, deterministic logical latency,
  available plasticity/churn measurements, and explicit applicability for all
  unavailable or inapplicable diagnostics. Exact source/runtime replay and
  checkpoint continuation are validated. It writes no artifact and is always
  `not_assessed`; it is a report-construction witness, not efficacy or evidence.
- **Nonlinear off-policy control kernel.**
  `NonlinearOffPolicyActorCritic` is an isolated shared-tanh-trunk categorical
  actor/scalar-critic lane. It learns only from an exact cached executed-action
  receipt and applies clipped per-decision target/behavior action ratios to
  separate actor-head, critic-head, actor-trunk, and critic-trunk traces.
  Actor, critic, and trunk have fixed independent plastic/frozen policies;
  target revisions, typed Threefry sampling, exact fail-stop clocks, atomic
  rollback, strict checkpoints, and persistent-state bytes are explicit. This
  is discounted scalar-V L0 machinery: the external behavior revision is a
  caller declaration, state-visitation correction, average reward, learned
  component utility, retention, convergence, matched control benefit, and
  promotion remain absent.
- **Nonlinear average-reward control kernel.**
  `NonlinearAverageRewardActorCritic` is an isolated discrete differential
  actor/critic with separate `tanh` actor and critic networks, traces,
  momentum, fixed head/trunk plasticity policies, bounded utility telemetry,
  and a learned average-reward baseline. Exact cached target and caller-owned
  behavior distributions, revisions, action identity, and owner digest feed a
  pure proposal plus recomputing atomic commit; only an accepted commit draws
  the successor action. Ordinary epsilon-mixture behavior-score and clipped
  target-importance modes are explicit. This remains L0 `not_assessed`
  machinery: the latter is action-only correction, utility has no plasticity
  authority, and convergence, retention, safety, matched control benefit, and
  promotion remain absent.
- **Matched nonlinear/SARSA development lane.** A nonwriting six-state
  RiverSwim A/B/A diagnostic gives the nonlinear differential actor-critic and
  `DifferentialSARSAAgent` complete prequential traces, explicit policy timing,
  checkpoint/resume, deterministic replay, descriptive recovery, and logical
  resource accounting under common random-key roles. Their successor timing,
  parameterization, persistent bytes, and realized scalar work are explicitly
  unmatched, so the lane remains `not_assessed` with no threshold, winner,
  efficacy, evidence, or promotion claim.
- **Candidate-update safety audit.** The canonical multi-probe
  `assess_candidate_update` audit plus its effective-delta-audited atomic
  `apply_candidate_update` application boundary. Historical
  `assess_gradient_joy` and `apply_gradient_joy_update` spellings are
  compatibility-only aliases. In
  [*Does This Gradient Spark Joy?*](https://arxiv.org/abs/2603.20526),
  delight is the exact float32 advantage times selected-action surprisal and
  “does this gradient spark joy?” is true iff that sample's exact contribution
  enters an actor backward pass that actually executes. The gate records
  detached forward admission intent; its actor consumer establishes execution.
  That execution fact is independent of gradient finiteness, parameter-update
  acceptance, and any later outer-transaction acceptance.
  `PrototypeUpdateResult.sparks_joy` and `joyful_gradient_applied` are
  historical compatibility aliases, not the paper's Kondo semantics.
- **`KondoGate` and actor consumer.** A detached forward screen with a fixed-
  capacity sparse gather now feeds `KondoSparseActor`: the nonlinear
  categorical actor gathers first and only then calls `jax.value_and_grad` on
  the smaller fixed shape. `backward_admission_intent` names the gate's plan;
  the gate's frozen `sparks_joy` accessor is only a compatibility alias for
  that forward intent and never proves execution. The canonical execution fact,
  `KondoSparseActorResult.sparks_joy`, records contributions that entered an
  executed actor backward. Forced/overflow survivors use an explicit full-shape
  fallback. Canonical gate config/checkpoint payloads are v2 and serialize only
  detached backward-admission-intent semantics; exact v1 payloads are a strict
  import-only compatibility path. Returns and baseline predictions enter the
  actor only through detached advantage; critic and safety features stay
  outside its loss, while every protected learner remains full-batch and
  ungated. Focused sparse and
  fallback execution tests change rejected features, actions, and detached
  advantages and require the actor loss and gradient to remain bit-identical,
  so the actor result records contributions that entered an executed backward,
  rather than repeating a forward-only admission label. A strict
  four-arm development evaluator shares one immutable
  parameter snapshot and source trace across ordinary-full, capacity-matched
  uniform-sparse, Kondo-top-k, and diagnostic-overflow arms. It records exact
  backward shapes/invocations and a separately bound, warmed, blocked,
  interleaved raw/p50/p95 `perf_counter_ns` timing section, while excluding
  host screening/gathering, memory, energy, and end-to-end latency. Every result
  is `not_assessed`, so this is measurement instrumentation rather than a
  compute-saving or learning claim. The `kondo_enabled` flag on the separate
  Delightful Policy Gradient config remains reserved and fail-closed because
  that full-batch helper cannot
  skip compiled backward work.
  All three evaluator families now use v2 serialized contracts. Cross-arm
  outcomes describe executed actor-backward inclusion neutrally; replay and
  on-policy records expose `executed_actor_backward_mask` with semantics
  `gradient-contribution-entered-executed-actor-backward`. The canonical
  execution-level use of `sparks_joy` is an actual `KondoSparseActorResult`:
  ordinary-full and uniform-sparse use manual backward kernels rather than
  Kondo transactions, and ordinary-full makes no delight-selection claim.
- **Kondo actor/critic replay diagnostics.** A second nonpromoting lane runs
  ordinary-full, capacity-matched uniform, paper top-k Kondo, and a
  fixed-capacity Kondo-plus-minimum-random-reserve extension on one immutable
  A1/B/A2 contextual-gambling replay. Every source batch produces one actor
  and one full protected update per arm; baseline, critic, representation,
  world-model, and safety/guardrail gradients and final states are
  bit-identical across arms, including rare failures. Actions are evaluator-
  fixed, however, with no behavior policy or importance correction. Actor
  losses are off-policy surrogates, and logical row-slot reductions are not
  measured compute or efficacy. The lane remains `not_assessed` and writes no
  evidence.
- **Kondo closed-loop on-policy diagnostics.** A third development lane gives
  ordinary-full, capacity-matched uniform, paper top-k Kondo, and top-k plus a
  minimum random reserve independent actor/environment/protected-learner
  states. Each batch is sampled from that arm's immutable actor revision using
  evaluator-owned typed Threefry common uniforms; only exogenous randomness is
  paired, and trajectory equality is never assumed. Actions, behavior log
  probabilities, revisions, causal parents, and full-batch baseline, critic,
  representation, world-model, and safety learning are replay-bound. One rare
  failure per batch is forced into every actor backward and the complete
  guardrail backward. The in-memory lane remains `not_assessed` and makes no
  efficacy, compute, safety, evidence, or promotion claim.
- **Prospective exploration selector.** `ProspectiveExploration` scores a fixed
  pre-decision candidate batch by expected improvement times capped
  host-relative surprisal, alongside random, epsilon-greedy, ensemble-
  disagreement, information-gain, and learning-progress modes under the same
  budget and logical RNG schedule. Exact event/owner/revision receipts gate
  selection; only afterward does a caller-owned hard shield admit the candidate
  or an independently shielded host fallback. The selector API's scores remain
  supplied and uncalibrated. A separate consumed eight-event development lane
  derives all six arms' scores from independent per-arm executed histories
  with action-conditioned linear-TD ensembles. It pairs only exogenous noise,
  exercises progress-resetting noisy TV and delayed invest/collect behavior,
  preserves the caller hard mask as the actual admissibility owner, and
  validates exact replay, in-memory resume, and matched logical budgets. Its
  report is `not_assessed`, threshold-free, winner-free, and artifact-free;
  neither the bounded estimator nor the shield establishes exploration or
  physical-safety benefit. Its expected-improvement-times-surprisal score is
  not DG/Kondo delight and the selector executes no actor backward; old v1
  `DelightfulExploration` import/config names are compatibility-only.
- **World-model lanes.** Four mutually exclusive Prototype lanes: the legacy
  single `OneStepWorldModel`/`ActionConditionedWorldModel` lane, a bounded
  bootstrap ensemble, `ModelReplayRehearsal` (ensemble plus fixed-capacity
  dual replay and model-member-only rehearsal with isolated RNG and counters —
  replay never trains the actor, critic, builder, or signal calibrator), and a
  bounded recurrent latent ensemble with member-specific trainable GRUs,
  heteroscedastic grounded heads, and atomic checkpointable online NLL
  updates. A bounded `ShallowRidgeWorldModel` supplies an interpretable
  action-conditioned regularized-FTL reference with a diagnostic planner.
  A separate `RealStateOneStepDyna` kernel can perform fixed-budget one-step
  ensemble backups from exact real anchors with support, residual,
  disagreement, finite-value, and termination-agreement vetoes. A separate
  `EnsembleShortRolloutPlanner` emits proposal-only policy- or uncertainty-
  directed fixed-horizon paths with full model-content receipts and
  termination-correct reverse returns. A separate
  `ImaginedRolloutSelectionGauge` requires bounded grounded action×region
  evidence and noncompensating prediction, termination, success, purity, and
  caller-owned mask gates, with every rejection propagated through the rest of
  that path. Its isolated actor/critic proposal performs no autodiff; a fresh,
  revalidated commit performs exactly one fixed-shape backward pass. The
  matched control clones caller-declared competent real episodes under the
  same prefix-closed transition/update budget. Deterministic tags protect
  post-mint integrity but do not authenticate planner issuance or competence.
  A strict `GroundedImaginationComposition` now removes the public rollout-
  batch substitution seam: it derives policy/value authority from the live
  actor/critic, passes the exact locally produced planner batch directly to the
  gauge, and atomically advances planner, authorization, learner, dream, and
  composition clocks around at most one backward pass. Model support, the real
  anchor, regions, and safety/protection facts remain caller attestations. None
  is a `PrototypeAgent` training lane and the gates are not externally
  calibrated. **Dyna
  dreaming inside `PrototypeAgent` currently runs only on the legacy lane**:
  `PrototypeAgentConfig` rejects `n_dreams_per_step > 0` combined with the
  ensemble, replay-rehearsal, or recurrent lanes until their uncertainty and
  rollout-validity gates are calibrated. None of these lanes carries a
  calibration, retention, planning-benefit, or efficacy claim.
- **Learned state in Prototype.** The opt-in builder path consumes an
  identity, fixed-trace, online-gated, conventional dense full-GRU, or
  diagonal complex compressed-RTU builder
  causally, caches the
  dispatched decision, and rejects stale transition generations atomically.
  The opt-in ensemble produces one causal world-model representation
  gradient; a successor opt-in mixer combines it with the current control-loss
  semi-gradient (base-Q on idle primitive transitions, the intra-option
  objective while an option executes), logging source norms, weights,
  clipping, cosine/conflict, and failures; delayed option-start credit and
  replay gradients are excluded. An optional decision-bound candidate-update
  audit stores the mixed delta only when its formed-candidate and effective
  finite-precision checks both pass. Its canonical API is
  `CandidateUpdateAudit*` plus `PrototypeCandidateUpdateAuditEvidence`;
  historical `GradientJoy*` names are compatibility aliases and do not mean
  that an actor gradient sparked joy. The full GRU carries exact
  fixed-parameter RTRL sensitivities under the shared proposal/commit,
  checkpoint, and resource contract; its `O(H * P)` sensitivity storage and
  carry across parameter updates are explicit limitations. A separate
  stateless `ExternalBuilderCandidateEvidenceProducer` binds caller-owned
  objective, retention, and safety representation probes to one exact
  external-coordinator identity and pulls them through that cached source RTRL
  sensitivity into parameter-space audit evidence. Stale or non-finite probes
  fail closed; independence remains caller-attested. It performs zero actor
  backwards and therefore cannot report that a gradient “sparks joy.” The RTU
  persists compressed unit-diagonal sensitivities that are exact for fixed
  parameters;
  default moving-parameter carry and its optional source/delta-owned diagonal
  Taylor correction remain approximations. A separate `RTUGenerateAndTest`
  lifecycle tracks pre-update real/imaginary activation-gradient contribution
  as diagnostic utility. Its live comprehensive-objective composition now
  evaluates a separate prequential causal utility: before updating any head it
  jointly deletes each complex unit's real and imaginary channels, measures the
  change in the frozen balanced prediction/control loss, and maintains a
  positive bounded EMA plus an independent evidence floor. Live replacement
  has no contribution-rank fallback; missing or immature causal evidence
  defers recycling while the real transition and ordinary builder update can
  still commit, while an attempted invalid or non-finite internal deletion
  evaluation rejects the complete outer transaction. The stable
  lowest-causal-utility mature units are eligible
  under the existing quota, protection, cadence, and active-option guards. A
  replacement redraws recurrence/input parameters,
  scrubs activation/RTRL/Taylor ownership slices, accepts only an exactly
  recomputed ordinary-learning destination, and has fixed clocks, RNG,
  checkpoint, and resource contracts. Its lower-level finalization receipt
  detects mutation and proves deterministic derivation by reconstructing the
  advance receipt, rerunning the RTU commit, and exact-matching the destination
  and mask; it does not authenticate caller-supplied lifecycle, objective-
  gradient, or ordinary-proposal authority. A strict opt-in comprehensive-
  objective composition now learns the current transition under the old
  representation,
  performs an atomic whole-unit replacement, scrubs every selected axis in all
  comprehensive objective heads plus the supported linear STOMP/OaK base head,
  intra-option heads, traces, and option transition models, then selects the
  next action from the recycled
  representation. Replacement is deferred while an option executes. This
  narrow lane excludes nonlinear STOMP, planning, model/replay/dreaming,
  Horde, IA, partner/memory, GRU, historical candidate-update audit, and
  feature-lifecycle sidecars. The adapter owns the lifecycle source and builds
  the objective gradient and source-bound ordinary proposal internally. Its
  declared worst case evaluates one frozen-head counterfactual per RTU unit,
  builder commit four times, and RTU commit twice for independent
  preflight/derivation checks, while persisting one logical ordinary update and
  at most one logical replacement event. Its RTU-enabled lifetime declaration
  is bounded by the per-unit uint32 age/support/evidence counters
  (`2**32 - 1` accepted transitions), rather than the base adapter's uint64
  cache bound. It is L0 causal-deletion/recycling
  machinery—not paper-defined delight, an independently held-out probe, or an
  outcome result. This is mechanism integration,
  not evidence that a learned representation improves control. The
  resource-unmatched consumed write/hold probe gave mean accuracy observation
  `0.5158`, fixed trace `0.5292`, online-gated `0.5258`, full GRU `0.5067`, and
  RTU `0.5617`; that descriptive supervised-development result is not a
  matched control or learned-state claim.
- **Balanced learned-state objectives.** The `BalancedStateObjectives` kernel
  keeps multiple-timescale linear GVFs and a
  consecutive-pair inverse-action classifier in separate heads, averages the
  GVF family before applying fixed positive objective-group masses, and emits
  clipped current/successor gradients bound to one exact executed-action
  receipt. Its opt-in `PrototypeBalancedStateObjectives` adapter binds those
  gradients to the exact dispatched Prototype decision and decision-time
  builder revision, preserves final-observation versus autoreset ownership,
  and commits or rolls back the complete composition atomically. The ordinary
  Prototype path is unchanged. A separate standalone
  `ComprehensiveStateObjectives` kernel adds action-conditional
  next-observation/latent, reward, stable Bernoulli termination,
  multiple-timescale GVF, value, selected-action advantage, and inverse-action
  heads. Fixed family masses are invariant to target width/head count, and
  exact receipts, revision rows, numerical rollback, finite-difference
  gradients, checkpoints/resources, and eager/JIT/scan parity are tested. An
  opt-in `PrototypeComprehensiveStateObjectives` transaction now binds caller
  targets/provenance to the exact decision, action, observation event,
  final/bootstrap observation, and online-gated, full-GRU, or compressed-RTU
  builder owner. It
  sums current/successor RTRL pullbacks into one clipped logical update and
  rolls the whole composition back on failure. The strict-linear RTU lifecycle
  lane above now consumes that update with content-bound prepare/finalize
  receipts and exact replacement-event revisions; the lower-level receipt is a
  derivation proof, while this adapter supplies the owning authority. Weights
  and targets remain uncalibrated. A separate versioned
  `PrototypeCausalStateObjectiveTargets` composition removes ordinary target
  choice from the caller: its target producer derives one detached factual
  bundle from the accepted real transition. It supports the same exact RTU
  builder only with its matching strict lifecycle, and scores deletion against
  frozen pre-update heads with that fixed learner-owned bundle. Recurrence,
  target heads and pending cache, RTU sensitivities/Taylor state, supported
  linear STOMP consumers, lifecycle/RNG, and successor cache then commit
  atomically or roll back bit-for-bit. Recycled float axes are canonical
  `+0.0`; checkpoints reject type-aliased metadata and noncanonical empty-array
  sentinels. General consumer compatibility,
  independently held-out feature-utility validation, and matched Forager
  outcome gates remain open.
- **Bounded Prototype pair-feature lifecycle and WP7.1b/WP7.1c audit
  ranking.** The original restricted lane
  trains a fixed pair bank from one owner-bound behavior TD target. A second,
  still narrow lane shares that bank with linear OaK and an ordered linear
  Horde: task channel zero is the control target and later channels are the
  Horde update's TD targets in declared demon order. Both consumers first
  update under the old descriptor bank; a committed descriptor change then
  routes their post-update feature axes atomically in exactly two router
  calls. Gradient pullback remains bound to the exact pre-route generation and
  full descriptor bank. Scale-normalized proxy utility assigns `0.5` to
  control and `0.5/D` to each of `D` demons, so the Horde receives `0.5` in
  aggregate. The shared state binds OaK, Horde, descriptors, and ordered
  semantics under a digest in the v4 Prototype checkpoint; exhausted lifecycle
  capacity is an audited no-op that leaves already-advanced, step-aligned
  consumers untouched. The shared lane fails closed unless OaK and Horde are
  linear and the Horde uses exact LMS scalar optimizer state with no
  normalizer. Standalone callers must checkpoint every consumer with the
  returned binding. An additional opt-in diagnostic forms active deletion
  scores from the old-bank, frozen predict-before-update consumer snapshot and
  evaluates exact normalized one-step half-squared-loss change. It keeps a
  matched shadow-candidate insertion cohort separate, scoring before its
  normalized-LMS shadow weights, utility EMAs, or scale moments update. Task
  mass is fixed at `0.5` for control and `0.5/D` per ordered demon. After a
  committed two-call route, audit state explicitly rebinds by descriptor
  identity without another router call. Audit-enabled state uses a nested
  atomic bundle and requires the v5 checkpoint schema; with the audit disabled,
  v4 remains unchanged. WP7.1c adds an opt-in stateless adapter over the
  post-observation audit EMAs. Its feature-gradient utility is deletion/
  insertion sensitivity, not paper-defined actor-sample delight: it neither
  scores actor samples nor selects backward passes. It ranks lower deletion utility only within active slots and higher
  insertion utility only within candidates; it never compares the cohorts.
  Every configured task must meet the evidence floor for a slot, and fixed
  task mass is never renormalized. Existing ages, cadence, candidate
  confirmation, proxy promotion floor and margin, and safe routing retain
  go/no-go authority. The adapter adds no state, RNG, backward, consumer, or
  router work and requires the exact v6 checkpoint shell around v5; disabling
  it preserves v5 behavior. This is L0 ranking instrumentation only: it has no
  curation, promotion, or go/no-go authority and makes no adapted deletion,
  empirical return or benefit, planning, control, safety, evidence-renewal,
  scientific-promotion, WP7-completion, Alberta Plan-completion, or L3 claim.
  The lifecycle and its ordered-Horde variant now also expose a source-bound
  prepare/adopt boundary for an external all-consumer coordinator. Preparation
  computes one ordinary old-bank update plus the routed candidate in a single
  learner evaluation; adoption performs zero learner or router evaluations. A
  veto keeps that exact ordinary successor and accounts a rollback, never a
  deferred proposal. The isolated generated-input linear world model exposes
  the same split and can retain its valid ordinary physical-model update even
  when the destination route itself is invalid. Exact-content receipts reject
  stale or tampered inputs but are unkeyed integrity records, not caller
  authentication. Their transient byte declarations count serialized logical
  PyTree leaves, not physical allocator peaks. The opt-in v18
  `prototype_atomic_feature_world_memory` mode now supplies the corresponding
  single-owner coordinator: one pair lifecycle/router drives linear OaK, the
  ordered linear Horde, the fixed-physical-output routed world model, and
  exact feature-bound memory. A descriptor change commits only when lifecycle,
  world, and memory are all ready; a veto retains every valid ordinary
  old-bank update. At the lifecycle observation cap, the locally derived
  current encoding remains usable while lifecycle state and adoption flags
  stay unchanged. Exact checkpoints and fixed ownership/work accounting are
  covered. Planning remains disabled by default, and this L0 composition has
  no selective-retention, planning/control-benefit, evidence, promotion, or
  default claim.
- **Feature-bound memory plus stable-base world recurrence.** A separate exact
  `IdentityStateBuilderConfig` lane atomically re-encodes every valid bounded
  memory row when pair descriptors change, while a v17-bound linear
  action-conditioned model and recent-observation buffer consume only the
  stable base prefix. A strict development-only visible-cue A/B/A harness runs
  one uninterrupted Prototype life with linear OaK, managed Horde, pair
  lifecycle, feature-bound memory, and that model under matched readout and
  promotion controls. Exact clocks, rejected-event rollback, stale replay,
  fixed resources, world prediction, and logical work reconstruct in memory.
  The partner is scripted, pair tails are not modeled, and no default-life
  outcome, benefit, artifact, held-out result, evidence, or promotion follows.
- **Standalone WP7.2 v1 cumulant/subtask proposals.**
  `CumulantSubtaskDiscovery` owns a fixed candidate universe spanning
  controllable events, feature changes, reward-relevant transition atoms, and
  typed prediction bottlenecks. Its two-phase `arm`/`observe` boundary freezes
  predict-before-update values and moves successor semantics forward; an atom
  born from the current reward-relevant transition receives no same-transition
  evidence. Learnability, randomized-propensity controllability, novelty
  against incumbents and earlier selected proposals, and frozen reward/model
  insertion contribution are noncompensating gates. Bottleneck candidates
  additionally need epistemic/progress evidence and pass a persistent
  running-mean aleatoric veto.
  Four fixed positive family quotas sum to the exact budget `B`; quotas are not
  reassigned and no partial discovered bundle is emitted. A random projection
  bank sampled once supplies one cohort, and an exactly `B`-entry
  identity-bound hand cohort supplies the other comparator. All three use the
  same budget and materialize into compact appended tail slots, not candidate
  IDs. Strict v1 config/checkpoint and live source/transaction bindings, tamper
  checks, static ceilings, and exact resource declarations bound the mechanism.
  Unlike WP7.1c's feature-gradient utility ranker, this
  path invokes neither Kondo nor delight and performs no backward pass. It
  mutates no OaK, STOMP, Prototype, or Horde state and owns no curation,
  promotion, go/no-go, or scientific-promotion authority. A separate opt-in
  `CumulantOptionInstallation` now consumes only a complete fresh bound bundle,
  installs its descriptor semantics into preallocated live STOMP slots, and
  reevaluates those cumulants on every accepted observation. Cold slots are
  masked across action selection, real TD bootstraps, skip diagnostics, and
  option-model planning. Quiescent semantic rebinding preserves identical
  slots and fully resets changed ones; an active option/comparator is an exact
  no-op and requires a later fresh proposal. Installer exhaustion freezes only
  replacement, not installed control. `CumulantOptionScheduler` now observes
  discovery at each accepted transition and requests a fresh bundle at a
  bounded exact cadence/retry, then receipt-gates installation at a quiescent
  boundary. It stores no proposal payload, rejects replayed authority
  revisions, and emits no-authority maintenance handoffs. A separate
  `AuthorizedOptionRetirementController` binds one such handoff to an explicit
  caller receipt, recomputes a per-context noncompensating policy, and uses two
  independently keyed public lifecycle rebinds to scrub an approved option
  before leaving its slot authoritatively cold across behavior, learning,
  bootstrap, planning, and audit attribution. A separate
  `AuthorizedOptionReplacementController` then owns one canonical scheduler
  state through that retirement and at most one later cold-slot replacement.
  It prepares discovery with install authority denied, keeps the fresh bundle
  transient, and reruns the complete preparation before an exact caller
  receipt can install and reactivate just that slot. Decline retains only the
  ordinary scheduler advance; stale, forged, replayed, nonquiescent, capacity-
  exhausted, or non-single-slot transactions fail closed. The unkeyed receipts
  declare integrity and authority facts but do not authenticate callers.
  An opt-in reserved suffix on the empty STOMP template leaves the historical
  raw-plus-option layout unchanged while allocating later cells for a
  separately bound external owner; installation-produced tokens keep those
  cells exact zero and reject suffix tamper. The stateless v2
  `FreshColdSlotCumulantCohortFilter` can also seal one same-family,
  semantically fresh cold-slot proposal while preserving every live slot and
  independently rechecking family quotas. The original candidate universe
  remains unavailable; one explicit extra feature candidate is sufficient in
  the bounded fixture. The filter still cannot install or authorize its result.
  A separate opt-in v2 `AuthorizedFreshColdSlotAtomicSwapController` now
  consumes the exact prepared filter output through additive public scheduler
  and replacement adoption boundaries. It rederives the authorized retirement,
  ordinary v1 preparation, filter source/output, identities, revisions, masks,
  and caller keys before selecting an all-installed successor; no-fresh,
  decline, outer veto, stale/replay, identity drift, or checksum-valid tamper
  returns the exact all-installed outer source. The one-cold state remains
  transient. One preparation performs one retirement derivation (two rebind
  evaluations), three scheduler observations, two filter derivations, and one
  installation-candidate evaluation. Commit performs six scheduler
  observations and three candidate evaluations but adopts at most one
  installation; the wrapper splits or creates no RNG root and its children use
  only the four caller keys. Receipts and checksums are unkeyed integrity
  declarations, not authentication. This composition does not repair or rerun
  the consumed repeated-lifecycle negative.
  Those standalone controllers are scheduled L0 integration plus externally
  authorized retirement and replacement; by themselves they provide no
  OaK/Prototype composition, empirical benefit, autonomous authority, WP7
  exit, evidence promotion, or Alberta Plan completion. A separate opt-in
  `PrototypeOptionAuthorityBridge` closes the bounded live-owner edge. Its
  nested Prototype→OaK→STOMP path is the sole persistent `STOMPState` owner;
  authority and lifecycle state are detached metadata borrowing that owner.
  Unequal pristine owners require an explicit directional receipt bound to
  both exact sources and typed owner digests; exact-source reevaluation is
  idempotent. Receipts, checksums, and checkpoint hashes are unkeyed integrity,
  not caller authentication.

  On an ordinary transition the bridge forwards every optional Prototype
  sidecar without reinterpretation and carries one installed-slot mask through
  start, real OaK/STOMP behavior and bootstrap, internal planning, option
  search, guarded Dyna, and lifecycle attribution. The lifecycle audits the
  exact raw `STOMPUpdateResult` without reevaluation; an ordered transient
  trace classifies option-search, feature-route, Dyna, memory-dispatch, and
  partner-dispatch mutations before metadata-only finalization binds the sole
  final owner. Invalid bridge sources cannot commit and receive a
  primitives-only transient mask. Dynamic audit refusal preserves valid
  Prototype control, retains authority metadata, and latches desynchronization.
  Diagnostics separate real, imagined, total, search-update, and internal-
  planning work. This bridge remains L0 `not_assessed`: it grants no caller
  authentication, autonomous lifecycle, physical-dispatch or safety authority,
  empirical benefit, evidence, promotion, WP7 exit, SOTA, or Alberta Plan
  completion claim.
- **Standalone option lifecycle and calibrated search-control contracts.**
  `OptionLifecycleAudit` records semantic-generation-bound initiation,
  termination reasons, returns, frozen option-model error, randomized
  primitive comparisons, planning use, redundancy, and resource cost through
  an exact two-phase transaction. It proposes bounded maintenance concerns but
  has no curation, control, or promotion authority. An opt-in persistent STOMP
  wrapper now derives those events from actual option ownership and frozen
  pre-update model state. Audit exhaustion or rejected attribution freezes only
  the observer; valid STOMP continues, while persistent composed-state
  corruption requires checkpoint recovery. Separately,
  `CalibratedExtendedSearchControl` gives model-free, primitive-model,
  option-model, and combined search one shared fixed backup budget. It uses
  correct differential primitive/option targets and a noncompensating product
  of calibrated value change, future real-anchor reachability, model
  reliability, and support. Natural and censored resolutions, option-semantic
  invalidation, pending-arm checkpoints, stable ties, zero RNG, and exact
  resources are mechanism-tested. A source/runtime-bound four-arm development
  evaluator gives every arm one frozen model/calibration snapshot, the same
  Threefry trace, and exactly `B` attempts with strict resume and exact causal
  replay. A separate opt-in `PrototypeSTOMPCalibratedSearchAgent` now snapshots
  the actual learned legacy Prototype world model and STOMP option models at
  live decisions, settles primitive/option outcomes with exact ownership, and
  gives their union the same single budget `B`. It is a raw-representation L0
  sidecar only: its Q values never rewrite the cached policy action, it draws no
  planner RNG, sidecar exhaustion cannot block valid Prototype learning, and it
  has no keyboard authority. A separate default-off v2
  `PrototypeSTOMPCalibratedDispatchAgent` can consume candidate-specific
  calibrated evidence at one exact anchor, form a primitive or proposal-only
  option-keyboard command, intersect a caller-owned hard mask, replace the real
  cached primitive without changing its base/active-option credit owner, and
  arm the resulting actual owner. An unavailable proposal may retain only an
  independently safe current-owner command; a withheld decision exposes `-1`
  and accepts no transition until a no-learning retry succeeds. This v2 edge
  has no safety or physical-dispatch authority and no measured benefit. Thus
  neither lane establishes an online-planning benefit, a WP7 exit, or L3
  evidence.
- **Bounded semantic/procedural consolidation and completion accounting.**
  `ConsolidatedMemory` adds fixed-capacity SHA-identified semantic
  GVF/fact/affordance and procedural skill stores with confidence, provenance,
  revisions, evidence/outcome moments, staleness, invalidation, deterministic
  replacement, and exact option-lifecycle links. Query-before-write,
  next-generation resets, resources, JIT/scan, and strict checkpoints are L0
  tested. A frozen 17-event development evaluator adds full-memory,
  same-kernel readout-ablation, and zero-memory traces with integrity-bound
  replay, but is explicitly `not-assessed`; no transfer or negative-transfer
  result is claimed. A stateless `ConsolidatedProceduralMemoryPolicy` applies
  exact lifecycle, evidence, Wilson success-bound, outcome-uncertainty,
  score-mass, and hard-safety gates to an already-produced procedural retrieval
  and can only propose the lowest-index safe positive-mass action. The policy
  itself performs no query, write, RNG, dispatch, or mutation. An opt-in
  `PrototypeConsolidatedMemoryAgent` settles exact procedural feedback, retains
  actual-action Prototype learning, and then applies that readout to the next
  cached primitive after experiential memory and partner fusion under
  intersected hard-safety masks. A separately versioned semantic wrapper uses
  the same controller/store, queries before writing, and appends an accepted
  prior semantic payload—or an exact zero tail—to the ordinary next Prototype
  context. Each exposed decision also carries a checksum-bound primitive and
  exact mask for a separate post-envelope settlement. An admitted unchanged
  action is a no-op; a changed admitted fallback atomically rebinds Prototype's
  real credit owner and cancels only matching procedural/partner owners.
  No-action, stale, disallowed, corrupt, and partial settlements preserve the
  complete state for retry and perform no learning or evidence write. Both
  wrappers keep valid control running after memory exhaustion and own no
  physical-dispatch, efficacy, evidence, or promotion claim. A separate
  fail-closed complete-prototype manifest enumerates all 18 final scorecard
  rows and accepts only configuration-matched, source-pinned, immutable,
  frozen L3 evidence with untouched held-out seeds. It has no default evidence
  bindings, so tests and stored booleans cannot manufacture a completion claim.
- **Semantic dispatch to embodied command identity.**
  `PrototypeEmbodiedCommandAdapter` binds that semantic wrapper's exact
  Prototype decision, selected primitive, and persisted hard mask to one
  unique command from a fixed float32 bank and a complete
  `EmbodiedSafetyEnvelope` request. Settlement recomputes the real envelope,
  bit-compares the complete result, and accepts only a uniquely mapped
  selected command or certified mask-admitted fallback. No-action and
  stop-only outcomes preserve the semantic credit owner for a fresh attempt
  while still adopting the envelope's rejection record or emergency-stop
  latch and closing the spent attempt receipt. The bank is an identity map,
  not a kinematics or collision proof; the adapter performs no physical
  dispatch, learning, evidence write, caller authentication, safety
  certification, deployment decision, or promotion.
- **Bounded embodied whole-agent harness.**
  `PrototypeEmbodiedDevelopmentHarness` owns exactly one semantic adapter, one
  bounded deterministic plant, and one grounded shadow state. An accepted or
  certified-fallback command first produces the mapped plant proposal, then
  settles the actual semantic owner and applies exactly one real Prototype
  transition from the plant reward and successor before atomic adoption. Two
  consecutive decisions run without reinitialization; Prototype/OaK clocks,
  current raw observation, and the plant stay bound. No-action is envelope-
  only and retryable, stop latching preserves plant/shadow/semantic state, and
  shadow mismatch distinguishes a proposed plant transition from a committed
  one while rolling the whole transaction back. Plant-capacity exhaustion
  stops later preparation without inventing termination or truncation. The
  harness has strict checkpoints and fixed resource accounting, but remains
  L0 `not_assessed`: shadow tags are unkeyed integrity sentinels, and there is
  no physical dispatch, geometry/safety proof, delight/KondoGate-intent/
  KondoSparseActor-backward assessment, efficacy, evidence, promotion, or
  deployment authority.
- **Paired embodied harness development benchmark.**
  `PrototypeEmbodiedPairedDevelopmentBenchmark` gives adaptive STOMP and an
  exact `zero_stomp_step_size_control` independently owned copies of that
  harness. Exactly five STOMP step-size fields may differ; all other initial
  semantic arrays, RNG, caches, observations, traces, and clocks match
  dtype/shape/typed-key-implementation/host-byte exact after normalizing
  materialized base-LMS step-size leaves. Starts also require empty harness
  pending/last-commit records, an unset adapter settlement ledger, zero
  Prototype/OaK/adapter/plant clocks, and sufficient remaining plant capacity.
  V1 fixes four
  continuing attempts and the bridge disconnect at attempt 1, yielding three
  real plant/Prototype commits plus one exact unavailable action/reward record
  per arm. The selected-source/runtime-bound report retains masked raw
  availability, fallbacks, plant state, clocks, rearming/learning/shadow work,
  exact resources, and named normalized lifetime AUCs over committed-transition
  and attempt indices. Sixteen fast synthetic pytest contracts cover typed-key
  identity, exact sentinels, signed-zero identity, drift, externally supplied
  prefix reconstruction, and content-plus-
  resealed tampering. The slow real lane runs only through
  `alberta-prototype-embodied-paired-development` and writes no artifact. It is
  always `not_assessed`: there is no winner, adaptation efficacy, safety,
  physical dispatch, delight/KondoGate-intent/KondoSparseActor-backward
  assessment, semantic use of historical `GradientJoy` compatibility names,
  evidence, deployment, or promotion result.
- **Continuing-control companions.** The separate bounded
  `ContinuousAverageRewardActorCriticAgent` closes the L0 continuous mechanism
  gap: direct affine-`tanh` actions with cached pre-`tanh` ownership, stable
  transformed target/behavior densities, an exact per-decision latent
  likelihood ratio, and one successor sampled only after an atomic commit. It
  does not address behavior-state-distribution mismatch, and no continuous
  retention or control benefit is claimed. A separate
  `DelightfulActorCriticAgent` development surface provides matched ordinary
  and paper-specific DG categorical policy-gradient modes plus nonpromoting
  contextual-gambling and RiverSwim A/B/A diagnostics; it has no validated
  control-benefit claim.
- **Partner and multi-agent substrates.** `PartnerPolicyFusion` is a bounded
  L0 surface that an opt-in `PrototypeAgent` path composes with real OaK
  dispatch: it binds messages and realized feedback to the full lifecycle
  identity, rewrites the correct base-or-option credit cache, and rolls the
  whole transition back on hard safety or post-state failure; missing, stale,
  duplicate, or misattributed sidecars fail closed. `BehaviorModel` (a bounded
  external-belief joint outcome model), `FeatureBankRouter`, and an uncued
  recurring hidden-partner stream complete the substrate set, and a bounded
  L0 kernel composes them with learned state, online pair discovery,
  joint-model planning, and differential SARSA in one causal update, with
  shape-matched component and retention ablations. The stream's partner is
  scripted. A separate strict development stress lane runs learned,
  outcome-blinded, and base-only fusion over one shared 96-event contextual
  reliability reversal with costs, cost spikes, disconnects, and hard-mask
  exclusions; fixed shapes/call counts, raw traces, exact replay, and prefix
  resume are explicit. It is always `not_assessed`, so none of this is a
  reliability-calibration, closed-loop partner-benefit, or WP8 completion
  claim. A second consumed 12-execution lane drives three separately owned
  real `PrototypeAgent` loops through the same exogenous reversal, noise,
  availability, costs, and masks while letting each arm's actions create its
  own later observations, rewards, messages, and feedback. It validates real
  action changes, action-relative assistance, caller-owned masks, exact causal
  replay, in-memory resume, and matched logical budgets. Independent learner
  states/RNG, the short life, and the absence of thresholds keep its summaries
  descriptive L0 `not_assessed` instrumentation—not a causal efficacy result.
  A separate v2 evaluator repairs that initialization confound: all three arms
  begin with bit-identical typed RNG, Prototype, fusion, and environment state,
  and only the wrapper intervention differs. Exogenous schedules are paired,
  but each arm owns its causal trajectory after action divergence. On the
  consumed 12-event run, learned and fixed-zero each changed action three times
  with identical task/net return, while base-only changed none; learned and
  fixed-zero internal states diverged without realized behavioral separation.
  Exact prefix reconstruction, hash-chain replay, checkpoint/resume,
  source/runtime/config binding, eager/JIT parity, and matched logical work do
  not turn that null descriptive run into an efficacy claim. The v2 lane is
  permanently L0 `not_assessed`, threshold-free, winner-free, artifact-free,
  and nonpromoting.
- **Embodied hard envelope.** `EmbodiedSafetyEnvelope` is a public,
  deterministic L0 filter over measured and proposed joint, workspace,
  collision, timing, bridge, identity, and version constraints. It returns a
  safe proposal, a statically configured in-envelope fallback, or no available
  action and has zero dispatch authority. Emergency stop latches independently
  of a rejected command transaction; authority-bound reset requires a strictly
  newer stationary-safe sample and external caller authentication. Rollback
  preserves diagnostics while suspending deployment, and checkpoint restore
  requires an exact revision and SHA-256 anchor retained outside the payload.
  Pure shadow records drive a non-authoritative Wilson/calibration/latency/hard-
  violation readiness readout. This is not a geometry proof, physical-safety
  result, robot-simulation result, or deployment authorization.
- **Synthetic embodied fault audit.** A strict 30-event continuing schedule
  exercises telemetry/wear drift, timing and delayed-reward metadata faults,
  sensor failure, bridge loss, unsafe candidates, emergency stop, reset,
  rollback, and checkpoint recovery against the hard envelope. Only available
  commands count as simulated executions; physical dispatch is zero. Raw
  traces, shadow facts whose success input is only action availability,
  action-availability recovery delays, kernel parity,
  externally anchored resume, and exact replay are retained. The schedule is
  not a dynamics simulator or geometry proof, its unchanged opaque controller
  cannot establish learner adaptation, and external caller authentication is
  still required. It has no thresholds, artifacts, deployment authority, or
  safety/efficacy verdict and remains `not_assessed`.
- **Embodied dynamics/adaptation diagnostic.** A separate strict 12-event
  A/B/A-plus-consumed-change-family lane runs an adaptive `PrototypeAgent`
  against a capacity- and update-call-matched zero-learning control, with an
  owned two-joint plant per arm and typed Threefry pairing only for exogenous
  dynamics, sensor, latency, and fault inputs. Every primitive command crosses
  `EmbodiedSafetyEnvelope`; fallback changes use Prototype's public cached-
  action replacement, while unavailable actions create neither a simulated
  command nor a learner transition. Exact commands/IDs/transitions, drift,
  interventions, recoveries, resources, checkpoint resume, and causal replay
  are retained. The change family is consumed development data, not untouched
  held-out data. The lane writes nothing, physically dispatches nothing, makes
  no adaptation or safety claim, and is always `not_assessed`.
- **Development-only evaluators.** Strict, hash-bound, `not-assessed`
  evaluators make the lanes above inspectable without promoting them:
  feed-forward and recurrent world-model snapshot evaluators, a recurrent
  retention companion, a matched A/B/A-plus-noisy-TV three-way world-model
  harness, discrete and continuous actor/critic A/B/A companions, an
  experiential-memory transfer evaluator, and a privileged-reference
  continuing-control suite (per-regime retained learners, a
  stationary-multitask reference on an exactly counted frozen extra stream,
  and an exact frozen counterfactual outcome bound — descriptive context, not
  resource-matched baselines). None supplies a retention, control,
  calibration, or SOTA result, and the recurrent Gaussian objective is not a
  calibrated-likelihood claim.

Empirical objective calibration, world-model/inverse/planning utility sources,
held-out causal feature-deletion/selection efficacy, and the matched Forager
result remain absent.

## Quick start

### Adaptive step-size prediction

```python
import jax.random as jr
from alberta_framework import (
    LinearLearner, IDBD, Autostep,
    RandomWalkStream, run_learning_loop,
)

# Non-stationary prediction: target weights drift over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.01)

# IDBD: per-weight adaptive step-sizes via gradient correlation (Sutton 1992)
learner = LinearLearner(optimizer=IDBD())
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))

# Autostep: tuning-free, self-normalized (Mahmood et al. 2012)
learner = LinearLearner(optimizer=Autostep())
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

### Nonlinear function approximation

```python
import jax.random as jr
from alberta_framework import (
    Autostep, EMANormalizer, MLPLearner, ObGDBounding,
    RandomWalkStream, run_mlp_learning_loop,
)

stream = RandomWalkStream(feature_dim=10, drift_rate=0.01)

# Architecture: Input → [Dense → LayerNorm → LeakyReLU] × N → Dense(1)
mlp = MLPLearner(
    hidden_sizes=(128, 128),
    optimizer=Autostep(),
    bounder=ObGDBounding(kappa=2.0),      # prevents overshooting (Elsayed et al. 2024)
    normalizer=EMANormalizer(decay=0.99), # EMA normalization for non-stationary inputs
)
state, metrics = run_mlp_learning_loop(mlp, stream, num_steps=10000, key=jr.key(42))
```

### GVF / Horde predictions

```python
import jax.random as jr
from alberta_framework import HordeLearner
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

horde_spec = create_horde_spec([
    GVFSpec(name="reward_pred", demon_type=DemonType.PREDICTION, gamma=0.99, lamda=0.9, cumulant_index=0),
    GVFSpec(name="next_obs",    demon_type=DemonType.PREDICTION, gamma=0.95, lamda=0.0, cumulant_index=1),
])

horde = HordeLearner(horde_spec=horde_spec, hidden_sizes=(64, 64))
state = horde.init(feature_dim=20, key=jr.key(0))
```

### SARSA control

```python
import jax.numpy as jnp
import jax.random as jr
from alberta_framework import Autostep, SARSAAgent, SARSAConfig

agent = SARSAAgent(
    sarsa_config=SARSAConfig(
        n_actions=4,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.01,
        epsilon_decay_steps=50000,
    ),
    hidden_sizes=(64, 64),
    optimizer=Autostep(),
)

state = agent.init(feature_dim=20, key=jr.key(0))
obs = jnp.zeros(20)
action, new_key = agent.select_action(state, obs)      # epsilon-greedy, Gumbel ties
state = state.replace(rng_key=new_key)
next_action, new_key = agent.select_action(state, obs)
result = agent.update(
    state,
    reward=jnp.array(1.0),
    observation=obs,
    terminated=jnp.array(0.0),
    next_action=next_action,
)
state = result.state
```

### Average-reward continuing control (Steps 5–6)

```python
from alberta_framework import (
    AverageRewardHordeLearner,
    ContinuousAverageRewardActorCriticAgent,
    DifferentialSARSAAgent,
)
```

### Prototype composition surface (Steps 1–12)

`PrototypeAgent` can compose GRU perception, average-reward Horde learning,
Dyna planning with guarded dreaming (legacy world-model lane only, see above),
STOMP options, OaK option curation, an IA companion, and opt-in partner-policy
fusion. The legacy IA recommendation remains diagnostic, while the separate
fusion path can safely replace the next OaK primitive and its exact credit
owner. These optional mechanisms do not yet constitute an empirically complete
Alberta Plan agent. The atomic feature router and hidden-partner substrates
now compose in an L0 integrated continual-control kernel, but its partner is
scripted and its robustness artifact is structurally nonpromoting. An opt-in
value-only option search controller ranks completion-supported option-model
backups by recomputed differential semi-MDP Bellman residual under a fixed
budget; it preserves the already cached action and can affect only a later
extended-action selection, so it is not combined primitive/option search or a
benefit result. Option-model planning benefit and closed-loop learning-partner
benefit still lack promoted evidence.

```python
from alberta_framework import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
```

New integrations should pass each environment reward, next observation, and
continuation discount through `PrototypeAgent.update_transition`. Carry the
complete `agent.decision(state)` record across the environment boundary: its
four-word lifecycle/generation token prevents an old observation/action pair
from being replayed after the pair recurs. `next_observation` is the final
observation used for learning and bootstrapping;
`next_decision_observation` is the post-reset observation used for the next
command. They must match on a non-boundary transition. Every enabled
bootstrapping path, including the IA exo-cortex, receives explicit
continuation. The older `update` method remains a compatibility wrapper and is
unavailable with a canonical `state_builder`.

## Streams & testbeds

Beyond the synthetic prediction streams, the repository ships closed-loop and
multi-agent worlds used by the evidence suites:

```python
from alberta_framework.streams.closed_loop import RiverSwimMDP, SwitchingTwoStateMDP
from alberta_framework.streams.gauntlet import GauntletStream, LifetimeGauntletStream
from alberta_framework.streams.hidden_partner_mapping import HiddenPartnerMappingWorld
from alberta_framework.streams.matrix_game import RecurringConventionGame
from alberta_framework.streams.opponent import AdversarialPursuitStream, LearningOpponentStream
from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentWorld
```

| Stream / world | What it exercises |
|---|---|
| `GauntletStream`, `LifetimeGauntletStream` (`streams/gauntlet.py`) | The Alberta Gauntlet: drift, abrupt switches, scale shocks, nonlinear interference, and a 64k-step (extendable to 1M-step) single-life protocol with recurrence scorecards |
| `SwitchingTwoStateMDP`, `RiverSwimMDP` (`streams/closed_loop.py`) | Closed-loop control gates with analytic optima; regime switches without learner resets |
| `LearningOpponentStream`, `AdversarialPursuitStream` (`streams/opponent.py`) | Endogenous non-stationarity: the drift *is* another learner's learning curve, or an adversary steering inputs against a frozen predictor |
| `RecurringConventionGame` (`streams/matrix_game.py`) | Two learning agents forming, forgetting, and instantly recalling joint conventions on rule recurrence |
| `RecurringTwoAgentWorld` (`streams/recurring_multiagent.py`) | The frozen `A-meet → B-avoid → A-meet` coadaptation benchmark behind the promoted multi-agent claim |
| `HiddenPartnerMappingWorld` (`streams/hidden_partner_mapping.py`) | Uncued recurring scripted-partner world with evaluator-only task boundaries, for the hidden-partner integration kernel |

## New mechanisms

### SwiftTD

`core/swift_td.py` implements SwiftTD (Javed, Sharifnassab & Sutton, RLC 2024)
with step-size optimization, an overshoot bound, and step-size decay —
float32-exact against the authors' C++ reference. It follows the `TDOptimizer`
interface, so it drives `TDLinearLearner` and the TD learning loops:

```python
import jax.numpy as jnp
from alberta_framework.core.swift_td import SwiftTD

swift = SwiftTD(initial_step_size=1e-2, meta_step_size=1e-3, trace_decay=0.9)
state = swift.init(feature_dim=8)
step = swift.update(
    state,
    td_error=jnp.asarray(0.5),
    observation=jnp.ones(8),
    next_observation=jnp.ones(8),
    gamma=jnp.asarray(0.99),
)
state = step.new_state
```

### Stacked Horde: the demon axis as an array axis

`core/stacked_horde.py` batches the GVF demon axis into one array axis with
exact TD(λ) semantics, per-decision importance sampling, NaN cumulant masking,
and a nexting helper. Measured on CPU: 1,024 demons × 2,000 steps in ~0.2 s
steady-state (~0.3 s compile) versus ~140 s run + ~144 s compile for the
loop-unrolled multi-head path, and 65,536 demons at ~4.0e7 demon-updates/s.
`tests/test_stacked_horde.py` asserts exact semantics, analytic fixed points,
that all 1,024 demons learn, and generous time bounds.

```python
import jax.random as jr
from alberta_framework.core.stacked_horde import (
    StackedLinearHorde,
    nexting_spec,
    run_stacked_horde_scan,
)

# 8 sensor channels × 4 timescales = 32 demons, one batched array axis
config = nexting_spec(feature_dim=16, cumulant_indices=tuple(range(8)))
horde = StackedLinearHorde(config)
state = horde.init()

features = jr.normal(jr.key(0), (2000, 16))
cumulants = jr.normal(jr.key(1), (2000, 8))
state, predictions = run_stacked_horde_scan(horde, state, features, cumulants)
```

### Other mechanism surfaces

- **Context inference** (`core/context_inference.py`) — a bounded bank of
  per-(state, action) reward tables that infers the active hidden regime and
  gates control features by the inferred slot. Development evidence only:
  +0.519 mean paired gap over a no-context ablation on the tested hidden
  two-rule life, with calibrated thresholds.
- **State builders** (`core/state_builder.py`) — identity, fixed-trace, and
  online trainable gated recurrent builders under one causal fixed-budget
  contract with checkpoint parity.
- **Learning signals** (`core/learning_signals.py`) — a predict-before-update
  producer that keeps ensemble epistemic disagreement, aleatoric uncertainty,
  normalized residual, learning progress, and sustained change probability
  separate (noisy-TV and persistent-shift diagnostics included).
- **Experiential memory** (`core/experiential_memory.py`) — fixed-capacity
  typed episodic retrieval with query-before-write ordering, deterministic
  eviction, exact byte accounting, and checkpoint/scan parity. A strict
  development evaluator retains recurring A/B/A retrieval/error, abstention,
  harmful-recall, eviction-provenance, resource, and no-memory-fallback
  diagnostics in a reconstructable hash-bound report. The stateless
  `ExperientialMemoryPolicy` interprets retrieved vectors as categorical action
  mass under a hard safety mask. An opt-in Prototype path queries before write,
  stores the primitive action actually executed with its grounded outcome,
  composes memory before partner fusion, preserves no-memory state shapes, and
  rolls back a required unsafe/corrupt transaction. No transfer or control
  benefit is claimed. A separate `LearnedExperientialMemoryController` adds a
  bounded admission veto and learned per-exemplar retention values that affect
  actual eviction under exact pending-feedback identity. Its counterfactual
  feedback is caller-supplied and unauthenticated. The separate v1
  `ExternalLearnedStateLiveMemoryAdapter` now makes that controller the sole
  memory owner around one external full-GRU/router/audit coordinator whose
  inner Prototype memory is disabled. It settles exact prior feedback, updates
  the coordinator once, queries the next raw observation before writing the
  actually executed transition, and lets only an admitted exact one-hot
  retrieval use Prototype's public cached-action replacement under the caller
  mask. The pending receipt preserves the pre-retrieval action, effective
  action, decision, transaction, and exact mask; every child adopts atomically
  or returns the complete source. This is host-orchestrated L0 mechanism
  integration, not authenticated feedback, dispatch/safety authority, or an
  efficacy claim. A separate
  `HCCLLearnedMemoryFeedbackBridge` can instead bind one admitted categorical
  retrieval to an exact HCCL source/decision/event, B/M receipts, common hard
  mask, selected agent, action, routing result, and controller transaction,
  then settle that receipt from the agent's immediate eight-proposal
  `memory_total.net_reward`. Masked or unrouted retrievals take the
  controller's no-learning path, and the two states plus pending binding commit
  atomically. This is memory utility: it computes no exact actor-sample delight
  statistic and executes no actor backward. It remains host/eager L0
  integration with no run or efficacy claim. Its bounded prebound scan only
  replays and verifies a
  caller-supplied receipt trace; it is not an online agent orchestrator and
  generates no events, actions, or decisions. The separate
  `HCCLTwoLiveMemoryBridge` owns one HCCL state and exactly two of the live
  adapters. Existing pending receipts bind each agent's `B` and `M` actions;
  an agent without one has `B=M`, while `P=M` is the explicit no-planner rung.
  Agent 0 receives only `M0B1-BB` and agent 1 only `B0M1-BB`; the dyad memory
  interaction remains audit-only. Each child advances from its own executed
  `M` action and `PP` outcome, and next-event masks install only with the
  all-owner commit. This is host/eager L0 causal memory-utility plumbing, not
  delight, an actor backward pass, evidence, or a benefit claim. The separate
  `HCCLTwoLiveMemoryPrepareAdoptBridge` retains that exact state and mechanism
  while exposing a transient two-phase boundary: preparation evaluates all
  donors once and binds the nested raw/final STOMP and owner-finalization
  facts, candidate state, and per-agent extended mask; adoption performs only
  integrity checks and child adoption. A downstream veto or stale/tampered
  receipt rolls every owner back and outer-gates child-applied facts. It is
  still unauthenticated `P=M` L0 plumbing and cannot report that a gradient
  sparks joy because it executes no Kondo actor backward.
  `HCCLTwoLiveMemoryFactorizedPlannerBridge` is the additive planner rung. It
  retains one HCCL owner, the same two live learned-memory owners, and one
  paired factorized planner state/cache; no third Prototype snapshot is
  persisted for either agent. Each event evaluates HCCL and the two live
  adapters once, completes the paired behavior/joint-world models once, and
  reconstructs transient `P` states from the post-memory `M` owners through
  the public cached-action replacement. Planner grounding uses the external
  GRU builder's 17-wide constructed state, while the 16-channel physical raw
  observation remains separately bound to the HCCL plant and `PP` successor.
  Hard masks may replace a raw proposal with `M`; only the effective `P` pair
  is consumed and only `PP` can advance the bounded world. Adoption fully
  authenticates the source and candidate planner caches and does no donor,
  model, or world reevaluation. Seven focused current-source contracts pass
  individually. This remains host/eager L0 mechanism integration without
  external or physical dispatch, safety, caller authentication, Kondo actor
  backward, delight, matched benefit, evidence, or promotion authority.
  `HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge` is the additive
  no-planner repeated-option rung. Its persistent tree still owns only one
  HCCL state and two live coordinator/STOMP owners; two coordinator-free
  metadata bundles borrow those exact owners. Ordinary preparation consumes
  each already-evaluated raw STOMP result once, and a cached-action change is
  projected through the complete coordinator owner before all owners adopt or
  roll back together. At a real started-but-quiescent boundary, one selected
  agent may also consume the fresh-cold atomic swap seam while the HCCL world,
  the other agent, learned memory, pending feedback, and primitive masks remain
  bit-exact; the transient cold slot never persists. This remains `P=M`,
  host/eager L0 integrity plumbing with unkeyed receipts and no planner,
  dispatch, authentication, safety, Kondo backward, delight, benefit,
  evidence, or promotion authority.
  `HCCLContinualDyadTransaction` is the first atomic integrated-owner rung:
  one HCCL world/attribution state, two live post-memory action stacks, two
  slow-context states, and one paired factorized planner commit together. Its
  split API can stop after memory and then complete with the planner without
  reevaluating donors; the disabled-planner form is constructible for an
  external action owner. Its ordinary `step` surface accepts only the source
  state and hard masks, internally issuing the event, B/M/P binding, canonical
  per-agent memory provenance, preparation, receipt, and adoption. The separate
  factory and runner reuse that exact path for one fresh complete 420- or
  8,998-event primitive-only life and return only an in-memory trace. Those
  lives remain unexecuted as research results, and there is no partial resume,
  checkpoint, physical dispatch, matched benefit, evidence, or Alberta Plan
  completion.
  `KondoExecutedActionLineageBridge` closes a separate actor-accounting edge.
  It samples an unmasked fixed batch from one exact actor snapshot and binds
  every row to the full post-memory action-stack source, preparation, decision,
  and candidate action owner. A row is actor-eligible only when public adoption
  reconstructs bit-exactly and the same proposal is the consumed planner
  candidate, planner action before masking, final `P`, and action named by the
  next real transition. Invalid rows are sanitized before exactly one Kondo
  step while critic/baseline/return/safety arrays stay full-batch. The nested
  `KondoSparseActorResult` remains the sole canonical execution-level joy fact.
  V1 supports only all-true masks and supplies unkeyed host integrity, not
  caller or physical-execution authentication, dispatch, safety/critic
  execution, efficacy, evidence, or promotion authority.
  `HCCLKondoContinualDyadRoute` v3 composes those seams with actor-owned `P`.
  `event0` installs the first proposal and compact adoption certificate without
  an actor backward; every generic successor `event` consumes the prior pair
  through one Kondo transaction before sampling and atomically installing the
  next `P`. The paired planner remains a learning-only shadow with
  `planning_enabled=False`, actor input is the 23-wide post-memory base, and
  both live action stacks remain the only Prototype owners. The route derives
  both learned-memory event inputs, including exact row/source provenance and
  neutral uncertainty/safety/reliability fields, from its own causal-core event;
  callers cannot inject that metadata. An outer veto rolls persistent state
  back but cannot erase a backward that already executed; an actor contribution
  sparks joy only in the nested actor result. On each successor, the route now
  updates zero-initialized linear reward-value and cost-value heads over both
  rows before the actor step. Current features, actions, decision identities,
  rewards, and discounts come from the exact pending `P`/current `PP`
  transition; next features are the already-produced post-memory Prototype
  base. Targets bootstrap with detached pre-update values, and cost is exact
  `safety_cost + message_charge`. Event 0 performs neither protected nor actor
  learning, and later outer rollback preserves only transient backward facts.
  Scheduling, actor keys, and all-true masks remain caller-driven host/eager L0
  machinery. Current HCCL costs are zero, the protected learner's checkpoint is
  not composite route recovery, and there is no autonomous life, authentication,
  dispatch, physical safety or critic-efficacy result, route checkpoint/resource
  closure, evaluator, matched benefit, evidence, or promotion claim.
- **Canonical UPGD** (`core/canonical_upgd.py`) — source-profiled UPGD
  implementations for the paper, official README, and official experiment
  equations, plus a numerically safe extended default. The same module exposes
  a source-bound `OfficialAdaUPGD` equation profile for the released RL
  implementation and a separately named guarded `AlbertaAdaUPGD` extension;
  tests pin their different moment, normalization, noise, decay, and numeric
  semantics instead of blending them. Neither adaptive API is an efficacy or
  default-selection claim.
- **Self-Normalized Resets** (`core/self_normalized_resets.py`) — a bounded
  dense-ReLU reset baseline with per-unit completed-gap history, stable
  geometric silent-run tests, exact long clocks and reset caps, typed Threefry
  initialization, supported SGD/Adam slice semantics, strict checkpoints, and
  eager/JIT/scan parity. Its positive-support trailing-window convention is
  explicit and not claimed bit-equivalent to the released histogram code; no
  plasticity or retention benefit is claimed.
- **Spectral regularization, AdamO, and Calibrated Partial Resets**
  (`core/spectral_regularization.py`, `core/adam_o.py`,
  `core/calibrated_partial_resets.py`) — three isolated dense-layer WP2
  mechanisms with source-profiled equations, exact clocks, fixed resources,
  atomic rejection, strict checkpoints, and eager/JIT/scan tests. AdamO keeps
  task-gradient moments separate from the isometry delta. CPR uses explicit
  per-example incoming gradients and intentionally excludes biases because
  the paper appendix and released v1 implementation differ. These are
  defaults-off L0 building blocks, not generic network wrappers, Prototype
  arms, matched efficacy results, evidence, or default selections.
- **Optimization-centric plasticity diagnostics**
  (`evaluation/optimization_centric_plasticity_development.py`) — matched
  ordinary-SGD and initialization-centred L2 learners traverse one frozen
  A/B/A stream from the same immutable parameters. Raw old/incoming gradients,
  alignment, two-sided local probes, parameter displacement/churn, and
  separately reported dormancy are source/runtime/replay bound. The fixed
  zero-gradient/local-neighbourhood rule is descriptive only; the lane writes
  no artifact and provides no efficacy or evidence verdict.
- **Independent C-CHAIN equation comparator** (`core/cchain.py`) — a generic
  float32 PyTree surface implements the paper's Equation 8 objective exactly:
  one half of mean squared output churn against a detached one-step-lag
  reference on a separately identified batch, with exactly one scalar output
  per reference sample; vector-valued per-sample outputs are rejected. One
  valid proposal performs one combined base-plus-churn backward; invalid
  runtime preflight and commit perform none. The appendix loss-ratio coefficient has explicit bounded
  trailing-window, warmup, epsilon, and clamp controls, while empirical NTK
  approximate-rank and diagonal/off-diagonal measurements are diagnostics
  only. Transactions bind parameters, declared disjoint sample identities,
  clocks, resources, and checkpoints. The binding words and tags are unkeyed
  integrity aids: they do not authenticate callables, data provenance, or the
  caller-owned optimizer step. This isolated L0 comparator is not the full
  sequential C-CHAIN algorithm, is not wired into an agent, and establishes no
  plasticity, retention, control, default-selection, evidence, or SOTA result.
- **Option value + duration** (`core/option_value_duration.py`) — separate
  conventional option-return and expected-remaining-duration TD heads; a
  deterministic renewal diagnostic shows return/duration ranking picks the
  correct fast option where return alone does not (L1, supplied options).

## Evidence registry

Five narrow claims are registered. Each has a frozen protocol, preregistered
seeds, a versioned artifact schema, and a strict validator that recomputes
acceptance from primitive rows; `alberta-evidence-status` indexes them all.
Validation is fail-closed and pins registered source hashes: editing a
registered source file invalidates persisted evidence until the frozen
protocol is rerun.

The table records each immutable artifact's frozen outcome. In the current
working tree, registered source hashes have drifted for all five claims, so
each reports `invalid` and the command exits `2`; that live result takes
precedence over the historical outcome column.

| Claim | Frozen outcome | Artifact | CLI |
|---|---|---|---|
| `recurring_pair_features` | accepted (narrow L2) | `outputs/recurring_feature/evidence.v1.json` | `alberta-recurring-feature-evidence` |
| `scale_robust_pair_features` | accepted (narrow L2) | `outputs/scale_robust_feature/evidence.v2.json` | `alberta-scale-robust-evidence` |
| `ftl_world_model_decision_fidelity` | accepted (historical chain) | `outputs/ftl_decision/evidence.v1.json` | `alberta-ftl-evidence` |
| `recurring_multiagent_coadaptation` | accepted (narrow L2) | `outputs/continual_multiagent/evidence.json` | `alberta-multiagent-evidence` |
| `continual_intelligence_amplification` | valid rejection (frozen 10% gate) | `outputs/continual_ia/evidence.json` | `alberta-ia-evidence` |

Every console script is also a module CLI — the six evaluation entry points
are:

```bash
python -m alberta_framework.evaluation.evidence_manifest_cli        # alberta-evidence-status
python -m alberta_framework.evaluation.recurring_feature_cli        # alberta-recurring-feature-evidence
python -m alberta_framework.evaluation.scale_robust_feature_cli     # alberta-scale-robust-evidence
python -m alberta_framework.evaluation.ftl_decision_cli             # alberta-ftl-evidence
python -m alberta_framework.evaluation.continual_multiagent_cli     # alberta-multiagent-evidence
python -m alberta_framework.evaluation.continual_ia_cli             # alberta-ia-evidence
```

Exit contract for `alberta-evidence-status`: `0` — all registered claims
accepted; `1` — a valid scientific rejection or missing run; `2` — invalid
evidence (including registered-source drift). Even an all-accepted manifest
supports only the listed narrow claims.

## Benchmarks

### Forager (continual-foragax)

The `alberta_framework.benchmarks` subpackage contains the pinned
`continual-foragax==0.55.0` integration: paper-aligned presets, a causal
feature encoder, the `alberta_horde_ac` streaming actor-critic, the
`alberta_causal_map` cognitive-map candidate, official-NPZ and legacy-SQLite
importers (`official_foragax`, `forager_results`), and strict paired
statistics (`forager_matrix`). Run it with the CLI:

```bash
alberta-forager-benchmark --preset relearning --steps 10000 --seeds 0 \
  --agent alberta --agent random --output outputs/forager/smoke.json
```

See [FORAGER_BENCHMARK.md](FORAGER_BENCHMARK.md) for the paper protocols,
fairness boundary, and the attested RTU-PPO/DQN/PPO comparison workflow.

#### Matched-current campaign (frozen before execution)

The matched-current pipeline (the `forager_matched_*` modules and console
scripts `alberta-forager-matched-qualification`,
`alberta-forager-matched-campaign`, and
`alberta-forager-matched-sealed-evaluation`) qualifies a live networkless OCI
runtime, freezes 23 registered candidates (21 open-tuning candidates plus two
fixed descriptive orientations), and defines a sealed held-out evaluation
stage. Current source uses candidate-universe schema v2 (SHA-256
`6a9315cb…`); it has no renewed qualification or open-campaign artifact yet.
The existing `2c3b214c` roots below are immutable historical v1 artifacts and
must not be resumed with the current builder. A fresh qualification and new
output namespace are required before v2 execution.

- `matched_current_qualification_2c3b214c_v1` — historical v1 qualification
  completed.
- `matched_current_open_tuning_2c3b214c_v1` — the open-tuning campaign is
  historically prepared and frozen, but its `runs/` and `completions/`
  directories are empty: **zero tuning cells have been executed**.
- The sealed stage (`forager_matched_seal`,
  `forager_matched_sealed_evaluation_campaign`,
  `forager_matched_final_analysis`, `forager_matched_statistics`) is
  implemented and contract-tested. The evaluation runner is exposed as
  `alberta-forager-matched-sealed-evaluation`; seal and final-analysis remain
  module-only. None has been executed, and no seal, sealed-evaluation, or
  final-analysis artifact exists.

Every authority-bearing path in the pipeline terminates at a caller-supplied
external trust resolver that does not exist in-tree; the only in-tree anchor
is content-only/unendorsed, and the shipped RNG-parity receipt records
`promotion_authorized: false`. The campaign is therefore content-identity
machinery, not performance evidence.
[FORAGER_ALBERTA_CANDIDATE_AUDIT.md](FORAGER_ALBERTA_CANDIDATE_AUDIT.md)
records the internal implementation review (GO) alongside the uncleared
campaign authority, and
[FORAGER_COMPARATOR_AUDIT.md](FORAGER_COMPARATOR_AUDIT.md) records comparator
provenance and claim-scope limitations of the frozen v1 protocol.

### Online Permuted MNIST (OPMNIST)

Step-2 lanes exercise the online permuted-MNIST protocol from the
loss-of-plasticity literature (Dohare et al. 2024) for the UPGD,
continual-backprop, and associative-memory learners:
`tests/test_step2_opmnist_protocol.py`,
`tests/test_step2_upgd_memory_opmnist.py`,
`tests/test_step2_associative_opmnist_confirmation.py`, plus the D18 bridge
and D20 multi-prototype lanes.

### Publication-shaped development runners

These runners target published task constructions and horizons. Their strict
artifact checks are development infrastructure, not scientific promotion:

- `upgd_ipmnist` — the input-permuted-MNIST protocol from the UPGD paper
  (Elsayed & Mahmood, ICLR 2024);
- `ipmnist_screening` — a development *screening* lane: 48 registered
  mechanism-combination arms (UPGD×IDBD, UPGD×Autostep, UPGD+CBP, weight
  clipping, per-layer gate normalization, FADE-style meta-learned decay,
  SwiftTD-stabilized UPGD×IDBD, and others) on a 60-task proxy with
  run/validate-proxy/merge CLI and full-protocol confirmation pipelines
  (`outputs/ipmnist_screening/`). The stored proxy receipt rejects the UPGD
  prefix check, and the v1 shards lack source-bound execution provenance;
  screening results are permanently nonpromoting historical diagnostics;
- `upgd_label_emnist` — label-permuted EMNIST (balanced 47-class, labels
  permuted every 2,500 steps, 400 tasks), pinned to the audited upstream
  commit. The first 3-seed artifact
  (`outputs/upgd_label_emnist/results.v1.json`) reproduces the qualitative
  separation: UPGD-W online accuracy rises across tasks (first-quarter mean
  0.5616 → last-quarter 0.7284; whole-run mean 0.67151 versus the ~0.74
  figure read-off, gap flagged) while AdamW collapses (whole-run mean 0.20081
  versus the ~0.35 read-off, gap flagged). Descriptive only;
- `slowly_changing_regression` / `slowly_changing_regression_v2` — a
  publication-shaped implementation of the slowly-changing regression testbed
  from the loss-of-plasticity line of work, with a strict namespaced v2
  sharded contract.

The UPGD IP-MNIST lane has completed a matched 10-seed, one-million-step
development diagnostic: UPGD-W mean online accuracy was `0.7791470803916454`
(SE `0.000055690729820870456`) versus AdamW `0.7190002817213534` (SE
`0.0005943125024635892`), with a descriptive paired difference of
`0.06014679867029188` (10/10 positive). The canonical structurally valid
artifact and its current audit addendum are
`outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json` and
`outputs/upgd_ipmnist/nonpromoting_receipt.v2.json`; the addendum binds the
byte-preserved `nonpromoting_receipt.v1.json` predecessor. They are permanently
nonpromoting: the run used 10 rather than 20 published seeds, has documented
stream/logging/numeric deviations, and lacks execution-time source, complete
import-closure, command, environment, and dataset-byte binding. Its AdamW
result is about `+0.039` above the approximate publication figure read-off. A
scientific claim requires a fresh source-bound full-seed run, not an extension
of these consumed development seeds. The active future execution path is the
namespaced v3 plan/one-learner-one-seed-shard/exact-merge contract in
`alberta_framework/benchmarks/upgd_ipmnist_v3.py`. No v3 plan has been issued,
no v3 shards or artifact exist, and no fresh v3 seed has been consumed. V3
binds the exact run specification, exactly 20 fresh operator-reserved seed
IDs, selected hyperparameters, data bytes, runtime, source import closure,
commands, and complete Cartesian shard bytes, but remains permanently
nonpromoting because its execution envelope is not externally attested. See
[UPGD_IPMNIST_V3_RUNBOOK.md](UPGD_IPMNIST_V3_RUNBOOK.md).

The slowly-changing-regression lane has a strict namespaced v2 development
contract and a selected ReLU/SGD ordinary-BP path with Kaiming initialization
and true-MSE gradients. Its CBP and UPGD arms are explicitly Alberta-local
extensions. The full 100-seed × three-method run has not been launched; no
pre-run plan has been issued and no result artifact exists. Any future v2
self-recorded plan permanently forbids promotion and reports descriptive
curves without post-hoc pass/fail thresholds. Merge and ordinary validation
require exact deterministic replay of every shard; structural-only diagnostics
are explicitly nonvalid. Neither lane supports an inferential, SOTA, or
Alberta Plan completion claim; see
[CONTINUAL_LEARNING_EVIDENCE.md](CONTINUAL_LEARNING_EVIDENCE.md) for the
complete descriptive record and limitations.

## Core abstractions

**Learners** compose independent concerns — an optimizer, an optional
normalizer, and (for MLPs) an optional bounder:

```python
from alberta_framework import (
    Autostep, EMANormalizer, LinearLearner, MLPLearner, ObGDBounding,
)

LinearLearner(optimizer=Autostep(), normalizer=EMANormalizer())
MLPLearner(hidden_sizes=(64, 64), optimizer=Autostep(), bounder=ObGDBounding(kappa=2.0),
           normalizer=EMANormalizer())
```

**Optimizers:**
- `LMS` — fixed step-size baseline
- `IDBD` — per-weight adaptive step-sizes (Sutton 1992); extends to MLPs via `(∂y/∂w)²` h-decay generalization ([Meyer](https://github.com/ejmejm/phd_research/blob/main/phd/jax_core/optimizers/idbd.py))
- `Autostep` — tuning-free with gradient normalization (Mahmood et al. 2012)
- `TDIDBD`, `AutoTDIDBD` — TD variants with eligibility traces (Kearney et al. 2019)
- `SwiftTD` — step-size optimization with overshoot bounding (Javed et al. 2024), in `core/swift_td.py`

**Bounders:**
- `ObGDBounding` — dynamic bounding to prevent overshooting (Elsayed et al. 2024)
- `AGCBounding` — per-unit gradient clipping scaled by weight norm (Brock et al. 2021)

**Normalizers:**
- `EMANormalizer` — exponential moving average; non-stationary inputs
- `WelfordNormalizer` — Welford's algorithm; stationary inputs

**Streams** — non-stationary experience generators implementing `ScanStream`:
- `RandomWalkStream`, `AbruptChangeStream`, `PeriodicChangeStream`
- `DynamicScaleShiftStream`, `ScaleDriftStream`

## JAX design

Numerical learning-state kernels are designed for `jax.lax.scan` and JIT where
their contracts permit it. States are immutable `@chex.dataclass(frozen=True)`
PyTrees and keys are passed explicitly. Host orchestration, evidence
validation, benchmark import, and some bounded lifecycle/curation operations
remain intentionally Python-level.

```python
# Multi-seed experiment sweep
from alberta_framework import IDBD, LMS, Autostep, LinearLearner, RandomWalkStream
from alberta_framework.utils import ExperimentConfig, run_multi_seed_experiment

configs = [
    ExperimentConfig(
        name=name,
        learner_factory=lambda opt=opt: LinearLearner(optimizer=opt()),
        stream_factory=lambda: RandomWalkStream(feature_dim=10, drift_rate=0.01),
        num_steps=5000,
    )
    for name, opt in [("lms", LMS), ("idbd", IDBD), ("autostep", Autostep)]
]
results = run_multi_seed_experiment(configs, seeds=8, show_progress=False)
```

## Gymnasium

```python
import gymnasium as gym
from alberta_framework.streams.gymnasium import (
    PredictionMode,
    collect_trajectory,
    make_random_policy,
)

env = gym.make("CartPole-v1")
policy = make_random_policy(env)
obs, targets = collect_trajectory(env, policy, num_steps=1000, mode=PredictionMode.REWARD)
```

## References

- Sutton, Bowling, Pilarski (2022) — [The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)
- Sutton (1992) — Adapting Bias by Gradient Descent (IDBD)
- Mahmood, Sutton, Degris, Pilarski (2012) — Tuning-free Step-size Adaptation (Autostep)
- Kearney et al. (2019) — Learning Feature Relevance Through Step Size Adaptation in TD (TDIDBD)
- Javed, Sharifnassab, Sutton (2024) — SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning (RLC)
- Elsayed, Lan, Lim, Mahmood (2024) — [Streaming Deep RL Finally Works](https://arxiv.org/abs/2410.14606) (ObGD)
- Elsayed, Mahmood (2024) — Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning (UPGD, ICLR)
- Dohare et al. (2024) — [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) (Nature)
- Brock, De, Smith, Simonyan (2021) — High-Performance Large-Scale Image Recognition Without Normalization (AGC)
- Meyer (2025) — [IDBD for MLPs](https://github.com/ejmejm/phd_research/blob/main/phd/jax_core/optimizers/idbd.py)
- Forager testbed — [arXiv:2605.01131](https://arxiv.org/abs/2605.01131)

## License

Apache 2.0
