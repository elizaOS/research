# Alberta Plan research status

This document separates implemented mechanisms from scientific evidence. The
Alberta Plan is a research programme, not a conformance specification, so the
thresholds below are project criteria chosen to make completion falsifiable.

The current verdict is **in progress**. This repository contains substantial
online-learning machinery and broad unit coverage, but it does not yet contain
a fail-closed, end-to-end demonstration of one bounded agent learning through
an uninterrupted nonstationary life while retaining recurring critical
knowledge.

## Evidence levels

- **L0 — mechanism:** API, shape, finite-value, serialization, or local update
  tests.
- **L1 — learning:** a component learns in a controlled toy problem.
- **L2 — comparison:** a preregistered, multi-seed, matched-resource benchmark
  establishes an advantage over strong baselines.
- **L3 — integration:** one uninterrupted agent life demonstrates the required
  interactions, retention, recovery, bounded resource use, and causal
  ablations.

A step is complete only when its defining outcome reaches L2 and its required
links to earlier steps are exercised at L3. Missing promoted evidence must fail
the evidence command; it must not skip.

## Requirement-to-evidence matrix

| Step | Required outcome | Current strongest evidence | Status |
|---|---|---|---|
| 1 | Track nonstationary affine prediction with online normalization, relevance-sensitive step sizes, robustness, and bounded computation | LMS, IDBD, Autostep, normalizer, and bounder mechanism/learning tests | Partial |
| 2 | Generate nonlinear features, estimate future utility, and replace them under a fixed budget without losing recurring useful structure | Two immutable narrow L2 pair-feature artifacts, now source-invalidated against the evolved current learner | Partial; historical narrow outcomes do not certify current source, and general feature finding plus L3 links remain open |
| 3 | Learn many continuing, possibly off-policy GVFs with history and feature finding | Horde, TD/GTD, traces, history, and working-memory component tests | Partial |
| 4 | Progress from bandits through contextual and sequential actor-critic control with feature finding | SARSA and actor-critic mechanisms plus small control tests | Partial |
| 5 | Learn both differential average-reward GVFs and conventional value-plus-expected-duration predictions | Differential TD/GTD/Horde tests plus a deterministic online two-head option return/duration diagnostic | Partial; the defining mechanism reaches L1, but no promoted comparison or integrated option-control result exists |
| 6 | Reproducible continuing control suite, including RiverSwim, access control, Jellybean, GARNET, and continuing conversions | Closed-loop micro-MDPs, a pinned Foragax protocol runner, a completed five-seed field-of-view tuning stage, and an unsealed four-seed RTU-RTRL/DQN development comparison; the old selected 30-seed evaluation produced no batch or report | Partial; the named suite and a completed paper-length admissible paired Alberta comparison are not present |
| 7 | Validate incremental average-reward planning, then function approximation and adaptive features | Bounded Dyna mechanisms and an eight-seed development RiverSwim planning diagnostic | Partial; no frozen held-out artifact |
| 8 | Close the perception → model → feature ranking → feature replacement → model feedback loop | Accepted historical held-out decision-fidelity comparison for a lifetime-statistics transition model, recovered through an exact consumed-seed source-compatibility chain, plus a causal development path from trainable state through a bounded ensemble gradient | Partial; the narrow FTL comparison reaches L2 and the model-to-state mechanism is integrated, but balanced objectives, feature ranking/replacement feedback, and outcome evidence remain open |
| 9 | Improve exploration and planning order under matched real-transition and backup budgets | Prioritized, surprise, utility, and guarded-dream mechanisms plus development-only planning diagnostics | Partial; no promoted matched-budget search-control result |
| 10 | Discover reward-respecting subtasks, learn options and option models, and consume those models in planning | STOMP now separates task reward from pseudo-reward and consumes option models in bounded backups | Partial; mechanism corrected, defining outcome evidence missing |
| 11 | Track causal utility and safely replace features, subtasks, options, and models; compose an option keyboard | OaK transition ownership, active-option-safe curation, utility tracking, and keyboard mechanics | Partial; causal lifecycle outcome evidence missing |
| 12 | Measurably increase another learning agent's capability in a closed interaction loop | The frozen v1 IA run is a historical valid rejection: reward uplift and both augmentation controls passed, while action-changing intervention prevalence missed its threshold; a separate exact-297-byte development kernel now composes two learning roles with online behavior/world prediction, one-step planning, causal channel/freeze controls, and trace-reconstructed phase diagnostics | Partial; the historical v1 threshold still fails, current-source compatibility is invalid, and the new kernel has no default-life outcome, calibration, held-out partner-benefit result, or promotion authority |

No row currently satisfies the completion rule.

## Verified narrow probes

### Recurring pair-feature allocation

The frozen pair-feature protocol was calibrated on development seeds 0–29 and
then run once on disjoint held-out seeds 30–59. In one uninterrupted
`A → B → A → D → A → C → A → B → C` stream per seed:

- the retention variant ended with all three recurring A/B/C pairs in its
  three-slot active bank in 30/30 lives; the matched no-retention ablation did
  so in 0/30;
- obsolete D was absent from the active bank in 30/30 lives;
- held-out median normalized MSE for A/B/C/D was
  `0.003931 / 0.0000000011 / 0.00000000000012 / 1.007539`, versus
  `1.003505 / 1.016726 / 0.027338 / 1.000056` for the ablation; and
- median recovery fell from 112.5 samples on acquisition to 40 on recurrence.

The versioned artifact recomputes its decision from primitive per-seed phase
records and rejects rehashed aggregate, threshold, seed, provenance, and
active-pair inconsistencies. Deterministic paired-bootstrap 95% intervals are:
retention-rate gain `1.0 [1.0, 1.0]`, per-seed maximum-critical-NMSE reduction
`1.01215 [1.00198, 1.02447]`, obsolete-D NMSE increase
`0.02781 [0.01194, 0.04664]`, and acquisition-minus-recurrence recovery
`71.9 [67.63, 76.40]` steps.

Those numbers belong to the immutable pinned artifact. The live registry now
marks the claim invalid for the current implementation because multiple
registered implementation, artifact-builder, and CLI source hashes have
changed. The artifact remains historical and
must not be overwritten; consumed-seed replay cannot renew promotion.

This was deliberately narrow L2 evidence for its frozen source. The learner
receives supplied output heads, searches an exhaustive archive of all 15
raw-input pairs, and
keeps that archive plus its 60 candidate output weights in counted memory. D
is therefore evicted from the deployed active bank, not erased from all
memory. The result does not establish autonomous or general feature discovery,
continual control, indefinite memory, or general catastrophic-forgetting
resistance.

### Lifetime-statistics world model

A one-dimensional LoSSE-style active-block model learns action-conditioned
dynamics from predict-before-update transitions using fixed-shape lifetime
sufficient statistics. The original five-seed shared-dynamics `A → B → A`
probe retained normalized R² of `0.999880` on A after B, while A MSE still rose
by 3.44× (`7.88e-6` to `2.71e-5`), so retention is strong in absolute terms,
not interference-free.

A separately frozen decision-fidelity probe was then run once on held-out
seeds 30–59. After learning A then B, the sparse model's normalized six-step
planning regret was `0.000877 [0.000511, 0.001313]`, versus
`0.294800 [0.280117, 0.309169]` for an untrained model and
`0.049557 [0.044116, 0.055105]` for a fixed-memory raw-feature ridge model.
The paired regret reductions were `0.293923 [0.279258, 0.308440]` and
`0.048681 [0.043146, 0.054440]`. Its A-domain interference after B was small
but nonzero: `0.000252 [0.000015, 0.000578]`.

These measurements remain the historical scientific result. Its original
artifact bytes and scientific digest are pinned, and its acceptance is
reconstructed from primitive rows after projecting only the changed
artifact-builder hash. A current-source replay on the already-consumed seeds
30–59 validates and matches the historical artifact exactly after excluding
only operational metadata, that builder hash, and its digest derivative. The
replay is nonpromoting. Because the exact historical artifact-builder source
was not archived, this establishes deterministic scientific compatibility,
not complete historical source recoverability.

This is still a synthetic deterministic, fully observed, one-dimensional
known-reward ranking probe over a hand-designed open-loop action menu. It is
WorldModelGym-inspired, not the official protocol; it does not demonstrate
closed-loop MPC, reward learning, stochastic or visual modeling, or lifetime
numerical safety. Dense Gram memory is quadratic in feature dimension,
float32 accumulators are not lifetime-precision-safe, and the active-block
solve is not the cited paper's exact global FTL minimizer.

### Recurring two-agent coadaptation

The frozen visibly cued `A-meet → B-avoid → A-meet` benchmark was calibrated on
seeds 0–29 and run once on seeds 30–59. Three conditions have identical
ten-scalar controller state and two action scalars per step: frozen,
learner-only, and both agents adapting. Held-out mean prequential reward was
`0.5310 / 0.5755 / 0.8668`; the joint-minus-frozen paired effect was
`0.3358 [0.2991, 0.3738]`, and joint-minus-learner-only coadaptation was
`0.2913 [0.2620, 0.3178]`. All 30 lives recovered the recurring regime, mean
recovery was 13.4 steps, measured interference forgetting was zero, and the
read-only recurrent-A probe was 1.0.

The artifact schema is `alberta.continual_multiagent_evidence.v1`; the frozen
run's scientific digest was
`6d09b796f6c4bbd8332c5f6089c186b0d675217cde6653c283a683a34a8cbeda`.
The Wilson lower bound for 30/30 observed recoveries is only 0.886, so the run
does not establish a population recovery probability of 0.95. More
fundamentally, the controller is a tiny contextual bandit with visible regime
cues and separate value cells. This is narrow coadaptation and retention
evidence, not autonomous feature discovery, task inference, IA, or an L3
continual agent.

### Corrected option and dreaming semantics

STOMP's base controller now receives discounted environment return, while
pseudo-reward remains confined to the subtask learner and option model. Its
discounted differential semi-MDP target uses the matching baseline mass:

`Σ γ^k r_k − r̄ Σ γ^k + γ^T max Q(s', ·)`.

Option outcome models are consumed by an explicitly bounded number of planning
backups. A separate opt-in Prototype controller can allocate its bounded option
backup budget by the recomputed absolute differential semi-MDP Bellman residual
among completion-supported models. It commits only the base learner and
preserves real traces, lifecycle/action caches, OaK counters, and RNG. Because
it runs after the current OaK action is cached, its value effect is deferred to
the next extended-action selection boundary; completion support is not
calibration, and primitive dreaming still has an independent random budget.
OaK credits utility and terminating pseudo-reward to the option that
owned the transition, counts same-option restarts as new executions, and
defers replacement of an executing option. Its option keyboard now exposes a
deterministic current-chord proposal plus a real opt-in dispatch boundary. The
boundary binds the exact decision observation, validates the fixed STOMP/OaK
state and hard safety mask, preserves RNG, and rewrites the base or active
option cache that owns next-step credit; unsafe bases and corrupt states are
exact no-ops. This is ownership-level mechanism coverage, not automatic chord
learning or control benefit. Prototype world-model and IA
updates use the primitive action actually sent to the environment. Accepted
dreams can update the base value learner but cannot mutate the real option
lifecycle, cumulative returns, model, curation statistics, action, or random
state. These are correctness mechanisms, not evidence that options or dreaming
improve held-out lifetime control. The explicit `PrototypeTransition` path now
routes environment continuation to the world model, real control updates, and
IA exo-cortex bootstrapping, and accepted dreams consume the model-predicted
discount. The legacy `PrototypeAgent.update` wrapper retains its historical
split-discount behavior for checkpoint/caller compatibility. For wholly
one-hot control observations, `PrototypeAgent` also has an opt-in
`sample_one_hot` mode that samples a categorical next-state feature without
changing the real-action, anchor, or dream-action random streams; the raw
model prediction remains the legacy default. A consumed-seed RiverSwim
development run measured mean lifetime reward `0.2475` versus `0.1930` for the
legacy expectation-valued path (paired `+0.0545`, 6/8 seeds positive). Because
the mechanism was proposed after inspecting these seeds, this remains an L0
nonpromoting diagnostic, not a held-out performance claim.

The authoritative path now also carries a four-word lifecycle/generation
decision token and separate final/bootstrap and post-reset decision
observations. Stale or malformed ownership, non-finite inputs, inconsistent
recurrent caches, and invalid boundary semantics leave the complete agent
state unchanged. At a positive-discount truncation, the final observation
still supplies value/model targets, while episode-local state is cleared and
the reset observation supplies the next action. STOMP interrupts any active
option, applies the censored final sample, clears its trace, and does not count
that truncation as an option-model completion. Generic StateBuilder plumbing
and these boundary rules are L0/L1 correctness mechanisms; Prototype still
lacks the balanced downstream gradients needed to demonstrate learned state.

### Conventional option value plus duration

`OptionValueDurationLearner` now gives each supplied option separate online
TD(0) heads for conventional cumulative reward and expected remaining
primitive-step duration. Its targets are exactly
`[reward, 1] + continuation_discount * next_predictions`; unlike the
differential control learners, it does not subtract an average-reward
baseline.

A deterministic continuing renewal diagnostic alternates a slow option
returning 6 reward in 10 steps and a fast option returning 4 reward in 2
steps. From primitive transitions, the learner recovers values `(6, 4)`,
durations `(10, 2)`, and reward rates `(0.6, 2.0)`. Ranking by return alone
therefore selects the slow option, while ranking by predicted return per
predicted duration selects the correct fast option.

This is L1 development evidence only. The features, option identities,
policies, and termination discounts are supplied; the run is deterministic,
linear, and on-policy. The ratio is exact because both options renew at the
same decision state. It is not a general semi-Markov control rule for options
with different successor-state values, nor evidence of learned option
discovery or held-out control improvement.

### Average-reward actor/critic decision and retention diagnostics

The nonlinear Horde-backed discrete actor/critic now learns from the exact
cached prior action and policies, commits its critic trunk, actor, adaptive
optimizer state, differential reward-rate baseline, and counters, then samples
the successor action from those committed parameters. This removes the former
pre-update successor cache while preserving one categorical draw and the
existing state/result shapes. Its epsilon-mixture behavior policy is the actor
objective: the logged raw target/behavior probability ratio is descriptive,
while `(1 - epsilon) pi / b` is the exact chain-rule scale for the behavior
score, not an off-policy target-policy correction.

A strict development-only evaluator runs a fixed continuing A/B/A stream from
one source-bound snapshot. Phase/case identities, full reward tables, preferred
actions, and reference value targets remain evaluator-owned; the learner sees
only its observation, cached action, realized scalar reward, and next
observation. The reconstructing report separates critic error, actor margin,
policy churn, realized return/recovery, actor/critic plasticity, and action
activity, with eager/JIT replay, bounded resources, no-overwrite reports, and
snapshot checkpoints. This is descriptive `not-assessed` instrumentation: no
threshold, matched multi-seed comparator, retention/control result, Prototype
integration, or scientific claim follows.

The isolated `ContinuousAverageRewardActorCriticAgent` adds the corresponding
bounded continuous L0 path. Its behavior policy shares the target mean and has
a configurable broader pre-`tanh` standard deviation. Cached decisions retain
the exact latent draw, direct affine-`tanh` action, transformed target/behavior
log densities with stable Jacobian, and the exact latent-density likelihood
ratio; neither the Gaussian nor the transformed endpoint is clipped. The
actor uses `rho_t * (lambda * e_(t-1) + score_t)`, while the critic explicitly
learns the behavior-policy differential value and no state-distribution
correction or off-policy convergence claim is made. Separate parameters,
traces, LMS states, reward-rate baseline, typed RNG, saturating counters,
post-commit successor sampling, atomic rollback, strict checkpoints, resources,
finite-difference density/score checks, and eager/JIT/scan parity are tested.
A strict evaluator now runs one fixed 12-event, one-dimensional continuous
A/B/A life from an immutable source-bound snapshot. All phase/case identities,
preferred centers, reward functions, and reference values remain outside the
learner. Its report reconstructs cached action ownership, pre-tanh latent and
bounded median action separately, transformed densities, the exact latent
ratio, rewards, same-state centered critic error, actor error/churn,
plasticity/activity, counters, resources, final state, and exact live replay.
Reports/checkpoints are hash-bound, no-overwrite, and subject to absolute
byte/scalar ceilings. Only transformed diagnostic density reconstruction has
an explicit symmetric eight-float32-ULP backend allowance; the policy-defining
latent ratio remains bit-exact, and larger tampering fails. This is
development-only `not-assessed` instrumentation, not continuous-control
efficacy or retention evidence.

### Paper delight/Kondo, candidate-update auditing, and typed signals

`LearningValueRouter` now gives the eight named learning-value channels a
fixed-state causal boundary. It records producer, causal object, units, domain,
bounds, and normalization scale for each channel; validates them independently;
and normalizes only from pre-update Welford statistics. Its paper-DG actor,
exploration, model-memory/replay, adaptation/change, safety, and full candidate-
audit evidence routes expose exact-zero unavailable fields rather than a
universal sum. Delight must exactly match the float32 advantage-times-surprisal
identity, while unrelated failures cannot suppress valid safety cost. The
router performs neither Kondo selection nor a candidate-update verdict. This is bounded L0 routing with
checkpoint/JIT/scan contracts, not evidence that its inputs are calibrated or
that any consumer improves.

`assess_gradient_joy` is the historical API name for a separate optimizer-level
safety audit: does a
candidate gradient or formed update predict improvement on caller-supplied
objective, retention, and safety probes under a caller-attested independence
contract while respecting an update-norm bound? It fails closed unless
`probe_independence_attested` is true and all eight typed learning-value
channels have explicit valid availability. Its detached assessment exposes
`candidate_update_audit_passed` (the canonical alias of its single stored
`accepted` verdict), together with a weakest-link weight and named
raw-candidate plus tentative-update diagnostics.
Both the raw candidate and soft-weighted tentative update must satisfy the
objective, retention, and safety magnitude gates. The actual elementwise
float32 tentative tree receives fresh norm and dot certificates instead of
inheriting scalar-scaled candidate diagnostics; malformed or nonrepresentable
float32 controls fail before tracing. Fixed balanced dot and norm reductions
expose conservative accumulated-roundoff bounds; unresolved cancellation,
underflow, overflow, or raw/normalized sign disagreement fails the
derived-numerics gate. Trust and update-floor checks use the conservative norm
endpoints, alignment gates and factors use the certified cosine lower endpoint,
positive magnitude checks use the conservative dot-interval edge, and zero
thresholds require both the raw dot and certified normalized direction.
`apply_gradient_joy_update` reassesses
internally, re-audits the effective stored delta after dtype cast and parameter
addition after promoting both stored endpoints to at least float32 before
subtraction, and atomically applies only an exact shape-matched, finite parameter
tree when both audits accept. Its typed result distinguishes the formed
candidate, effective-delta verdict, and change actually applied, including
explicit no-ops for overflow, non-finite parameters, updates lost to parameter
precision, and quantization-altered trust or probe verdicts. This is distinct
from paper-defined delight and Kondo selection. At the Prototype boundary,
`PrototypeUpdateResult.candidate_update_audit_passed` exposes the
formed-candidate verdict; `audited_candidate_update_applied` separately means
the finite-precision stored delta passed the second audit and was committed.
The older `sparks_joy` and `joyful_gradient_applied` properties remain
compatibility aliases only.

The opt-in `PrototypeAgent` development path is the first audit consumer: a
world-model representation-gradient proposal is assessed at the same decision
point and committed only when its objective, retention, and safety probes make
`candidate_update_audit_passed` true. Evidence is decision-bound; missing, stale, malformed,
non-finite, unavailable, or conflicting evidence vetoes that builder update.
The world model and ordinary control state still advance on an otherwise valid
transition. The standalone audit marks the historical `LearningValue.delight`
channel unavailable unless it is the bit-exact finite float32 product of valid
declared advantage and action surprisal. The Prototype derives delight
internally rather than accepting a caller-supplied replacement for the audit
probe gradients. Neither mechanism is a generic reward score, and
neither gates safety or model learning. The gradient audit remains a local
first-order L0 mechanism, not realized-improvement evidence.

In the paper's terminology, delight is advantage times selected-action
surprisal, and a sample “sparks joy” when the Kondo gate selects it for a
backward pass. `KondoGate` now supplies an L0 detached forward screen and
fixed-capacity sparse gather. It supports finite-temperature Bernoulli-price
and deterministic fixed-rate top-k modes, preserves caller-declared forced
samples, and reports a flag requiring caller-managed full-shape fallback on
overflow. When capacity is below batch size, tests verify that the downstream
backward JAXPR receives the smaller gathered shape. The screen has
eager/JIT/scan coverage; the config-bound gather is intentionally host-only,
and checkpoint, resource, and invalid-input contracts are also tested. There
is no integrated learner consumer, measured compute saving, DG reproduction,
or learning/safety benefit.

The separate Delightful Policy Gradient development core now provides matched
ordinary and paper-specific DG modes for a continuing, categorical on-policy
actor-critic. Both modes share the same differential critic, reward-rate
baseline, typed RNG, and sampler; the actor trace is fixed to zero and the
detached paper-defined delight coefficient never gates critic/baseline learning or the
explicit safety/model/representation routes. It reports exact-sample validity,
effective sample size, gate strata, atomic rejection, and logical resource
counts with eager/JIT/scan/checkpoint contract tests. A deterministic
development-only runner now compares both modes on contextual heteroskedastic
gambling and uninterrupted six-state RiverSwim A/B/A lives. Its validator
replays each declared life and reconstructs action sampling, environment
outcomes, DG equations, traces, strata, metrics, and logical resources. Common
random numbers pair schedules, not realized trajectories after policies
diverge. Development seeds, diagnostic synthetic safety costs, absent
uncertainty intervals, and logical-only compute accounting prohibit a control
improvement, retention, safety, paper-reproduction, or promotion claim.

A bounded causal ensemble is also integrated as an opt-in `PrototypeAgent`
world-model backend. It predicts before updating, bootstraps members with
independent typed RNG streams, exposes one pre-update representation gradient,
and reports epistemic disagreement, a residual-variance aleatoric proxy,
normalized residual, fast/slow learning progress, and sustained change
probability. Its strict checkpoint and scalar/byte/update accounting are
tested. No default aggregation exists; the residual proxy is not externally
calibrated, and ensemble dreaming is disabled. Noisy-TV, persistent-shift,
invalid-input, JIT/scan, resource, and checkpoint tests are L0/development
mechanisms. They do not establish external calibration or that any routing
consumer improves.

The WP4 shallow reference is now executable as `ShallowRidgeWorldModel`: an
action-indexed affine regularized-FTL/ridge learner with fixed Gram/cross
sufficient statistics, grounded next-observation/reward/continuation targets,
predict-before-update results, fail-closed normal-equation and
positive-semidefinite-state validation, RNG-free checkpoints, exact allocation
accounting, and a pure one-step supplied-linear-value planner. Hand-derived,
boundary, corruption, JIT/scan, checkpoint, and resource fixtures are L0
mechanism tests. There is no uncertainty, recurrence, replay, MPC, measured
latency, retention/control comparison, or paper-result reproduction.

`DualReplayMemory` now supplies the bounded WP4.4 storage substrate: a fixed
FIFO/long-term slot split, reservoir or configured surprise/coverage/progress
retention, explicit aleatoric noisy-TV control, old policy/value fields,
representation and eviction provenance, stale-aware stratified sampling, and
exact resource/checkpoint/JIT/scan contracts. `ModelReplayRehearsal` now
atomically composes a real ensemble update, signal-aware record, fixed-quota
sample, and model-member-only replay. Its replay key/masks/counters are isolated
from the real signal/residual/key/counter lane, and the opt-in Prototype path
exposes only the committed real gradient to its builder/candidate-audit boundary. Replay
does not train the actor, critic, builder, or causal calibrator. The calibrated
policy name still refers only to supplied scales and thresholds; no retention,
control, or empirical calibration result exists.

An isolated `RecurrentLatentWorldModelEnsemble` now provides the more complex
WP4 reference mechanism: separately initialized trainable GRUs, grounded
next-observation/reward/continuation means, bounded heteroscedastic-Gaussian
variance heads, bootstrap masks, explicit uncertainty warm-up, and a stopped-
target representation gradient. Its start/decision/update caches bind exact
observation, action, hidden state, and event ownership; accepted boundaries
learn from the final observation before resetting recurrent context for the
next decision. Corruption, stale caches, numeric failure, and exhausted
counters are exact state/RNG no-ops, with JIT/scan, resources, and checkpoints
covered at L0. An opt-in fourth `PrototypeAgent` model lane now binds this
model's exact representation/action cache to the dispatched decision, advances
the model and causal signal estimator transactionally, and exposes only its
accepted real NLL gradient to the builder/mixer/candidate-audit boundary. It remains absent
from replay and planning. A strict recurrent development adapter now binds a frozen initial
snapshot and evaluator-owned ordered events, records each predict-before-update
member mean, heteroscedastic variance, and NLL, then advances exactly one
isolated copy per accepted event. ID/OOD and evaluator-owned state/action-region
summaries, warm-up exclusions, source/config/snapshot/final-state hashes,
resources, canonical reports, and snapshot checkpoints reconstruct while the
supplied state remains unchanged. This is descriptive `not-assessed`
instrumentation: the training likelihood is not shown calibrated, and no
retention, control, or superiority result follows. A recurrent-retention
companion additionally requires exact ordered case reuse after an intervening
evaluator context and recurrent resets, and reconstructs phase,
recurrence-entry, and within-occurrence NLL summaries from the same strict
trace. It is still a one-snapshot mechanism diagnostic, not a retention claim
or a Prototype actor/critic/representation result.

A separate development evaluator can freeze and reconstruct learner snapshots
and probe seven component channels without learner-visible regime IDs or
targets: representation, dynamics/observation, reward, termination/discount,
critic/value, actor margin, and actor return. It reports explicit
inapplicability/unavailability and bounded calls, records, snapshots, state
bytes, and checkpoint/resume state. These pointwise probes do not yet provide
longitudinal Prototype retention, probability calibration, or a multi-seed
comparison.

The continuing-control boundary is now executable rather than only a report
shape. A strict evaluator runs candidate and baselines on independent
functional environment states, owns evaluator-only regime scheduling, and
requires exact observation/action/decision-ID ownership across selection,
environment return, and update. Its hardened `PrototypeAgent` adapter and two
simple baselines produce canonical checkpointable states. V2 reports
reconstruct direction-aware prequential/lifetime return, post-change adaptation
AUC, sustained recovery, stability, final held-out action scores, forgetting,
backward/forward transfer, worst-window return, operations, resources, and
safety accounting, with explicit metric unavailability. Focused fixtures pin
hand-computed and deliberately forgetting cases. A separate privileged
reference suite now runs, outside the ordinary condition list, a learner
initialized once per evaluator regime identity and retained on recurrence, a
stationary-multitask learner with an exactly counted frozen extra stream, and
an exact frozen counterfactual action-outcome upper reference. Its reports bind
the additional data and callbacks, regime routing, initialization lifecycle,
resources, sources, and limitations; these are context bounds, not matched
baselines. This remains L0 development infrastructure: there is no completed
Prototype comparison, rollout probe, realized-resource matching, energy
measurement, or scientific protocol. A strict paired campaign companion can
run the ordinary evaluator over declared seeded factories and reconstruct raw
per-seed reports, direction-normalized differences, and deterministic
stratified bootstrap intervals. It remains `not-assessed`; no Prototype
campaign has been run and the descriptive intervals are not a promotion
decision.

### Learnable state and bounded experiential memory

Identity, fixed-trace, and online trainable gated state builders now share a
causal fixed-budget contract. The online builder exposes a pure proposal from
the source state and an atomic commit into the already-advanced destination
state, preserving recurrent carry and sensitivity. The opt-in `PrototypeAgent`
path can pass the real grounded-model gradient and a causal current-control
semi-gradient through a stateless successor mixer. Idle transitions use the
base-Q objective; executing options use the current intra-option objective;
delayed option-start and replay gradients are excluded. Source norms, weights,
clipping, cosine/conflict, and failures are explicit, and the exact mixed
candidate feeds both the builder proposal and the candidate-update audit. Its small
write/hold POMDP is explicitly development-only and resource-unmatched. There
is no multiple-timescale GVF, inverse, feature-utility, causal-deletion, or
empirically balanced objective result, nor a matched Forager improvement, so
the Step 8 exit gate remains open.

A frozen-snapshot world-model evaluator now retains raw member and mean
grounded predictions, reconstructs descriptive disagreement-versus-error and
coverage-risk summaries across evaluator-owned ID/OOD and derived state/action
regions, and makes single-model or warm-up unavailability explicit. Its
residual-variance channel is labeled a non-probabilistic proxy, and optional
open-loop diagnostics require grounded, exactly reconstructable action
sequences. A separate two-seed, 18-transition A/B/A development harness runs
the shallow reference, plain ensemble, and model-only rehearsal on an identical
raw stream with interleaved noisy-TV outcomes. It reconstructs prequential
channel errors, recurrence/adaptation descriptions, replay strata, and logical
resources. Both surfaces are `not-assessed`, resource/scale limited, and
nonpromoting; they provide no external calibration, retention superiority,
control benefit, or WP4 completion claim.

`ExperientialMemory` stores a fixed number of typed exemplars and performs
query-before-write retrieval with representation-version, similarity,
reliability, staleness, uncertainty, and safety gates. Deterministic
utility/recency eviction, exact byte accounting, checkpoint parity, and a
controlled stale-memory abstention are tested. A strict development-only
evaluator now runs an evaluator-owned recurring A/B/A query-before-write trace
from an immutable empty snapshot and records every neighbor, gate, prediction,
stateless fallback, harmful recall, and eviction provenance. It reconstructs
first/return descriptions, loophole diagnostics, exact resources, and
eager/compiled parity in source/snapshot/report/checkpoint-bound artifacts. It
has no threshold, and its no-memory fallback is event-opportunity matched
rather than storage matched. `ExperientialMemoryPolicy` adds a non-mutating,
deterministic categorical proposal boundary with a hard safety mask and no
confidence claim. An opt-in Prototype path now queries the next decision
representation before writing the grounded current exemplar, stores a one-hot
of the primitive action that actually ran with bootstrap representation and
reward, composes memory before partner fusion, and rolls back the entire event
when a required memory transaction is unsafe or corrupt. Full lifecycle IDs,
no-memory shape compatibility, checkpoint, curation, eager/JIT/scan, and exact
resource contracts are tested; the latter discloses two deterministic
pre-state queries and zero RNG per required transaction. Retrieval therefore
has not established held-out forward transfer under matched capacity; this
remains L0 integration rather than the WP8 memory exit gate.

`PartnerPolicyFusion` is now a bounded L0 mechanism with an opt-in
`PrototypeAgent` action path for WP8 IA. Fixed typed messages carry exact observation/context/decision/event
bindings, suggestion, declared confidence, rationale/provenance references,
communication cost, and a finite validity horizon. Five explicit routes use
discrete action selection under a caller-owned hard mask; action identifiers
are never averaged. One exact partner-influenced decision may await feedback,
and only realized assistance plus observed safety can update its contextual
logistic reliability row. Stale, duplicate, or misattributed feedback is an
atomic no-op, while cold-start acceptance is explicitly uncalibrated
development exploration. Prototype applies prior feedback before the next
fusion decision and binds both to its full four-word lifecycle identity. It
derives a real OaK base score and keyboard proposal, rewrites the exact
base-or-option credit cache, synchronizes the recurrent model's effective
action, and rolls back the whole transition on unsafe base dispatch or corrupt
post-state. Fixed-state checkpoint/eager/JIT/scan contracts are mechanism-
tested, but no confidence calibration, closed-loop partner benefit, changing-
reliability comparison, or WP8 exit-gate result exists.

### Hidden-partner development integration

The repository now composes four bounded L0 substrates in one integrated
kernel: an uncued recurring `hidden-partner-mapping-v0` stream, an online
discrete-action `BehaviorModel` with a supported input-loss gradient, a
bounded joint-action outcome model that accepts an external partner belief,
and an atomic `FeatureBankRouter` that moves every downstream feature column
by descriptor identity or fails closed. The stream keeps task boundaries
evaluator-only and separates partner behavior from joint-outcome dynamics.

`outputs/hidden_partner_development/robustness.v1.json` runs that kernel through
eight uninterrupted development lives with matched state, memory, lifecycle,
carry, retention, planning, partner-belief, and curation ablations. All eight
full lives are finite and contract-valid, with mean reward `0.89335`, and the
full condition beats lifecycle-frozen and random-curation controls on every
paired seed. The development checks nevertheless fail: obsolete D is absent
from the active bank at life end in `0/8` full lives, so the artifact's
`full_d_forgetting_fraction_at_least_0_75` check is false.

The artifact is explicitly `development_only` with
`scientific_promotion_allowed: false`. Its partner is scripted and
nonlearning, its schedule family is fixed, and discovery searches a closed
exhaustive 66-pair archive. It therefore establishes an L0 integrated
development and falsification rung, not learning-partner coadaptation, general
feature finding, an L2 comparison, or L3 Alberta integration.

### Pinned scale-robust pair-feature result (current source invalidated)

The broader gauntlet's exhaustive pairwise learner now has an opt-in
scale-robust head: diagonal NLMS output updates, bounded signed intervention
utilities, and bounded dormant residual regressors. It remains a narrow
feature finder: task context is visible, only degree-two pairs are enumerated,
and all 91 candidate descriptors and weights remain resident and counted.

The version-2 protocol was frozen on direct-key development seeds 8–15, then
run once on 30 exact seeds derived from the
`alberta-scale-robust-v2-fresh-evidence-2026-07-30:` SHA-256 namespace. The
immutable artifact was accepted as a narrow L2 package comparison under its
frozen source. The primary arm's median
first/final-C savings were `10.968 / 5.933`, its median D savings was `12.491`,
and final-C tail MSE was `0.038707` (per-seed maximum `0.049835`). Under the
scale shock, median early/cumulative/tail MSE was
`48.270 / 14.123 / 4.690`, with per-seed maxima
`126.982 / 44.035 / 37.326` and no non-finite step. At end segment 7 the
median bank held all eight relevant C products and all eight relevant D
products; at program end the per-seed minima were eight C products and seven
D products.

Against the legacy package, paired mean final-tail error reduction was
`8.725 [8.239, 9.194]` and final-C savings gain was
`4.869 [4.410, 5.358]`. Before final C reacquisition, the retained arm
preserved more C products than the no-retention arm by
`1.300 [1.000, 1.600]` and more D products by
`2.700 [2.267, 3.133]`. These last comparisons establish structural
retention, not that those retained columns causally contributed to control or
prediction output.

The artifact is
`outputs/scale_robust_feature/evidence.v2.json`, schema
`alberta.scale_robust_pair_feature_evidence.v2`, with scientific digest
`c2fee922c04a59fe26b4b8c9cfa77ddd9198cfa2bc923f54fec14b649bd3bb2c`.
The live evidence registry now rejects current-source compatibility because
registered implementation and CLI source hashes have changed. The pinned
artifact remains historical and immutable; it does not certify this working
tree. A consumed-seed replay would be nonpromoting, so renewed evidence
requires a new path/schema and untouched preregistered seeds.

This is not causal attribution to one optimizer change: the comparison between
the primary and legacy packages changes scale normalization and ObGD together
and costs an additional 464 persistent bytes in the frozen artifact. All
stream seeds share one fixed learner initialization, task context is visible,
and the learner searches an
exhaustive finite pair archive. The pinned result therefore does not show
task inference, general or unbounded feature finding, causal usefulness of
retained features, continual control, or L3 integration.

### UPGD Input-permuted MNIST development diagnostic

The one-million-step UPGD Input-permuted MNIST lane completed 10 matched
UPGD-W/AdamW seeds. Mean online accuracy was `0.7791470803916454` (SE
`0.000055690729820870456`) for UPGD-W and `0.7190002817213534` (SE
`0.0005943125024635892`) for AdamW. The paired descriptive difference was
`0.06014679867029188` (sample SD `0.0018825070977402044`, SE
`0.000595301014029226`), positive for all 10 seeds. UPGD-W's approximate
publication-figure gap was `-0.000853`; AdamW's `+0.039` gap is explicitly
flagged as a reproduction gap. Full task-window summaries are recorded in
`CONTINUAL_LEARNING_EVIDENCE.md`.

The preserved original `outputs/upgd_ipmnist/results.v1.json` fails strict
validation only because its note does not preserve the exact 10-vs-20-seed
limitation. The canonical
`outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json` passes strict
structural validation, and
`outputs/upgd_ipmnist/nonpromoting_receipt.v2.json` binds its post-hoc audit
record while preserving `nonpromoting_receipt.v1.json` byte-for-byte as its
predecessor. This is nevertheless permanently nonpromoting development evidence:
the publication used 20 seeds; this run also changed stream seeding,
task-boundary logging, and numeric details, and its shards did not bind worker
source, the complete import closure, commands, environment, or data bytes at
execution time. It establishes no inferential, SOTA, L2, or Alberta Plan
claim. Any scientific replication requires a fresh source-bound full-20-seed
execution rather than appending seeds to this run.

The active future path is now the strict namespaced v3 execution contract:
immutable pre-run plan, exactly one learner/seed per shard, and an exact
Cartesian merge bound to every shard's byte size and SHA-256. It also binds
exactly 20 fresh operator-reserved seed IDs, the selected hyperparameters,
closed deviations, dataset content, runtime,
semantic commands, and static transitive local import closure. No v3 plan has
been issued, no v3 shards or artifact exist, and no fresh v3 seed has been
consumed. Operator reservation cannot independently attest seed freshness, so
v3 remains permanently nonpromoting. Launch prerequisites and commands are in
`UPGD_IPMNIST_V3_RUNBOOK.md`; all sealed v1/v2 records remain unchanged.

### Forager benchmark lane

The repository now contains a pinned `continual-foragax==0.55.0` integration,
causal observation/history encoder, bounded scan runner, protocol metadata,
official-NPZ import, paired-statistics checks, and a task-specific stationary
causal-map planner. A stage-conformant field-of-view tuning matrix evaluated
four Alberta variants for 10,000 steps on five disjoint tuning seeds and,
under its frozen conservative-CI rule, selected `step3e3`.

At this snapshot, the selected 500,000-step Alberta evaluation on seeds 0–29
has no completed report and its previously observed worker is no longer
active; it produced no batch artifact. Its seed-matched official-DQN directory
and the relearning tuning directory are explicitly quarantined after
termination under superseded execution contracts. None can be resumed,
imported, or compared. A separate completed 30-seed DQN
provenance run on seeds 3,000,000–3,000,029 is labelled
`corrected_official_reproduction`, with `exact_paper_config: false` and
`exact_paper_source: false`. None of these execution records is registered in
the scientific evidence manifest, so there is no Forager performance result
or promotion.

A separate RTU-RTRL run completed 500,000 GPU steps on four open development
seeds. Its FOV tail-EMA AUC was 1.550 mean, 0.324 sample SD, and
1.167–1.936 range. The exact captured metrics and run-time direct-source hashes
are preserved in
`outputs/forager/rtu_rtrl_500k_dev4/receipt.v1.json`. The receipt is
structurally nonpromoting: the variant was not preregistered, there is no
admissible paired baseline or held-out interval, and source closure is
incomplete. A reconciled unsealed DQN receipt at
`outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004_reconciled/receipt.v1.json`
records a descriptive RTU-minus-DQN mean difference of `+0.3309`, positive on
4/4 consumed seeds. The DQN configuration was created after RTU output and the
runtime, representation, resource, replay, and update-work contracts are
unmatched; this is not inferential, causal, speed, or SOTA evidence.

Two later open, two-seed CPU screens completed their frozen candidate sets
under the nonpromoting v4 aggregate contract. The feed-forward screen ranked
`DQN_LN-common-control` first with mean FOV tail-EMA AUC `1.49084`; the
stateful screen ranked `PPO-RTU_LN_128_1_relu` first with mean `1.78110`.
These are development rankings on consumed seeds whose candidate budgets are
not necessarily matched, not a comparison between the two screens or a
promoted selection.
The stateful aggregate also retains the known upstream RTU-PPO
action/environment RNG reuse confound. A fixed-action direct-versus-wrapper
trace matched exactly, but its host receipt remains
`content_complete_external_executor_receipt_unverified`, requires an external
trust resolver, and explicitly has `promotion_authorized: false`. Thus the
screens and parity probe close operational diagnostics only; they support no
inferential, superiority, SOTA, or WP3 claim.

The causal-map variant uses only observation/action/reward history, but it is
specific to the stationary 15×15 toroidal field-of-view task and incorporates
that public movement structure. It has mechanism and runner tests but no
performance artifact. It is therefore L0 benchmark machinery, not evidence of
learned recurrent state, nonstationary retention, or SOTA performance.
Version 0.55.0 remains a reproduction rather than exact submission-time
semantics.

### Historical held-out intelligence-amplification result

The frozen hidden-phase Step-12 micro-benchmark was run on seeds 30–59 after
calibration on seeds 0–11 and produced a **valid scientific rejection**, not
a passing artifact. A later consumed-seed compatibility replay once reproduced
every v1 scientific field exactly, but it is nonpromoting and no longer
validates against the current checkout: subsequent drift in
`alberta_framework/core/average_reward.py` makes the live evidence-registry
compatibility chain `invalid`. This does not alter the archived historical v1
rejection, and v1 does not certify the current implementation.

Recommendation acceptance at probability 0.5 improved mean reward by
`0.26703 [0.25505, 0.27839]` over the bitwise-identical observe-only control,
with identical 412-byte controller state and interaction budgets and zero
executed-action credit mismatches. Prediction augmentation also passed both
secondary controls: `+0.16186 [0.15392, 0.16961]` over partner-alone and
`+0.13172 [0.12105, 0.14203]` over same-shape noise. Accept-always was harmful:
`-0.06344 [-0.07764, -0.05175]` versus observe-only.

The preregistered primary gate nevertheless failed because accepted
recommendations changed the primitive action on only `0.08728` of steps
(`3,142 / 36,000`), below the frozen `0.10` minimum. The threshold was not
lowered after inspection. The artifact schema is
`alberta.continual_ia_evidence.v1`; its scientific digest is
`826889b847aa423d86eac02f7ed754acd0d477c34812795ec7dc99ea2521820e`.
This narrowed the next Step-12 experiment to improving useful intervention
coverage on development environments. A strict p=0.75 protocol using seeds
60–89 now exists as an **unissued, permanently development-only v2 contract**;
no plan, reservation, seed execution, shard, or v2 artifact exists. Because
its plan is self-issued without externally verifiable pre-run chronology, even
a run that passes every scientific gate remains a nonpromoting diagnostic with
`internally_accepted=false`. A future acceptance attempt needs a new schema,
untouched seeds, and an external chronology anchor. The consumed-seed
development selection and compatibility records likewise cannot justify
retuning or turn the failed v1 gate into a pass.

## First integrated gate

The smallest useful L3 gate is a continuing, recurring two-agent world:

1. Run one life with unannounced `A → B → A` (later `A → B → C → A`)
   regimes and no learner reset.
2. Give each learner only its ordinary observation/action/reward stream.
   Task identity and oracle state remain evaluator-only.
3. Include nonlinear signal, delayed information, persistent critical
   structure, obsolete distractors, and a partner whose behavior changes.
4. Fix budgets for learned features, state, models, options, planning backups,
   and updates per transition.
5. Compare paired seeds against matched no-lifecycle, no-memory, no-model,
   no-planning, no-options, no-IA, random-curation, replay, and fixed-network
   baselines.

The promoted gate requires at least 30 paired seeds and reports bootstrap 95%
confidence intervals. Initial acceptance thresholds are:

- recurring critical-task performance at least 90% of its previous peak;
- recurrence recovery at most 25% of initial acquisition time;
- obsolete feature retention below 10%;
- full-agent uplift over the strongest matched-resource baseline with the
  confidence interval excluding zero;
- no non-finite update, learner reset, unbounded state growth, or task-boundary
  input;
- p50, p95, and p99 update latency plus peak memory and serialized state size.

These are repository acceptance thresholds, not claims made by Sutton,
Bowling, and Pilarski.

Hyperparameters must be chosen from a short, declared development prefix or
separate development seeds, then frozen before the full lifetime and held-out
seeds are run. Selecting a configuration after observing its entire lifetime
is disallowed; this follows the evaluation warning in
[Lifetime tuning is incompatible with continual reinforcement learning](https://proceedings.mlr.press/v267/mesbahi25a.html).

## Metrics and provenance

Every promoted run must record:

- prequential reward and prediction loss;
- a phase/task performance matrix, peak-to-final forgetting, backward and
  forward transfer, stability gap, and recurrence recovery length;
- critical-feature retention and obsolete-feature eviction;
- planning gain per matched backup;
- partner uplift attributable to an action-changing intervention;
- seed list, full immutable configuration, source revision, Python/JAX/device
  versions, wall time, update-latency quantiles, peak memory, and state size.

The artifact must have a versioned schema and content hash. The evidence gate
validates the schema, provenance, sample count, confidence interval, and
thresholds—not only file presence or JSON parseability.

`alberta-evidence-status` is the fail-closed operational index for the
registered promoted artifacts. It does not run experiments. It invokes the
underlying strict validators, binds each artifact to the frozen protocol,
configuration, seed roles, thresholds, current registered source hashes, and
runtime provenance, and distinguishes `accepted`, `valid-rejection`,
`not-run`, and `invalid`. Exit codes are `0`, `1`, and `2`, respectively, with
both valid rejections and missing runs mapped to `1`. Even an all-accepted
manifest supports only its listed narrow claims; it is not an Alberta Plan
completion certificate.

## Relation to LeWorldModel

[LeWorldModel](https://arxiv.org/abs/2603.19312) is an
action-conditioned joint-embedding predictive world model. Its learned visual
encoder, predictor, and anti-collapse regularization are relevant candidates
for Steps 8–9. It is not itself a continual learner or an agent/partner model:
the published training protocol is offline, its planning evaluation uses a
separate controller, and it does not test recurring-task retention. Adoption
therefore requires online bounded updates, latent-drift/retention probes,
matched replay-free and fixed-encoder ablations, and a separate behavior model
for other agents.

## External baselines to reproduce

- [Continual Reinforcement Learning by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html):
  fixed sparse random features plus online Follow-The-Leader ridge statistics
  are a replay-free dynamics baseline. Its shared-stationary-dynamics
  assumption must be tested separately from genuinely changing physics.
- [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7):
  continual backpropagation, dormant-unit fraction, weight magnitude, and
  effective rank belong in every deep learner sweep.
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://proceedings.mlr.press/v267/tang25g.html):
  C-CHAIN and output churn/NTK-rank diagnostics are missing comparison points.
- [CORA](https://proceedings.mlr.press/v199/powers22b.html): continual
  evaluation, isolated forgetting, and zero-shot forward transfer complement
  the task-matrix metrics implemented here.
- [MEAL](https://openreview.net/forum?id=Mxg6mo1Xzj): after the tiny two-agent
  gate is reliable, the scaling lane should include long cooperative
  multi-agent task sequences rather than treating a three-segment toy as the
  final benchmark.
- [WorldModelGym](https://reka.ai/labs/research/worldmodelgym): add
  decision-based fidelity probes in which a frozen evaluator offers several
  action sequences, the model ranks their predicted returns, and the real
  environment scores normalized planning regret. One-step prediction MSE is
  not sufficient evidence that a model supports decisions.
- [stable-worldmodel](https://arxiv.org/abs/2605.21800): use standardized
  model adapters, planning solvers, controlled physical/visual variation, and
  out-of-distribution generalization tracks when the small vector-model lane
  graduates to image or robot observations.
- [Self-adapting Robotic Agents through Online Continual RL with World Model
  Feedback](https://arxiv.org/abs/2603.04029): compare calibrated observation
  and reward residuals as change signals and report recovery/convergence and
  safety bounds. The paper deliberately favors adaptation to changed physics
  over retention, so it is a change-detection baseline rather than a
  catastrophic-forgetting solution.
- [The World Model Remembers, the Actor
  Forgets](https://arxiv.org/abs/2607.19749): separately probe world-model,
  value, and actor retention. Its new three-seed result suggests graded dream
  self-imitation as an experimental actor-rehearsal condition, but requires a
  larger preregistered reproduction before adoption.

## Immediate closure order

1. Repair the hidden-partner lifecycle so obsolete D is evicted after its final
   use, rerun the development falsification checks, and only then freeze a
   disjoint held-out protocol. Keep this scripted-partner development rung
   distinct from learning-partner coadaptation.
2. Keep the p=0.75/seeds-60–89 IA v2 development contract unissued until its
   source/runtime envelope is settled and execution is explicitly authorized.
   V2 cannot promote; a future evidence protocol requires a new schema,
   untouched seeds, and an externally anchored pre-run plan. V1 consumed-seed
   selection and replay records remain nonpromoting.
3. Promote planning, option, and prototype integration diagnostics only after
   disjoint seeds, paired intervals, matched budgets, and versioned artifacts.
4. Integrate adaptive optimization, learned state, autonomous feature targets,
   bounded lifecycle, and model/planning feedback in one uninterrupted agent.
5. Start a new versioned Forager evaluation only after the hardened source
   settles, run its matched official baseline under the same accepted runtime
   contract, and import both into one paired comparison before the isolated
   resource-scaling lane. Never resume the quarantined directories or promote
   from execution manifests, development receipts, or incomplete reports.
