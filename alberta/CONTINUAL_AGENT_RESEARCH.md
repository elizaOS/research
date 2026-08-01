# Continual Experiential Agents: Research Synthesis and Alberta Framework Audit

- **Research snapshot:** 2026-07-30
- **Repository snapshot:** elizaOS research commit `c28144b`; Alberta Framework
  vendored from `lalalune/alberta` commit
  `2ac35333efae45cf969ce02ec1f2703476fed6c2`
- **Scope:** continual learning, continual reinforcement learning, world
  models, state construction, plasticity, forgetting, surprise, curiosity,
  delight, experience reuse, temporal abstraction, and intelligence
  amplification

The code audit targets the committed snapshot. Concurrent, uncommitted
remediation observed during the review is called out separately and is not
silently treated as committed evidence.

## Executive conclusion

The Alberta Plan remains an unusually coherent systems specification for an
experiential agent: learn from one continuing sensorimotor stream, use bounded
computation, construct state, predict, control, model, plan, discover temporal
abstractions, and improve all of these processes online. The literature does
not yet supply a proven algorithm that closes that specification. In
particular, no reviewed result establishes a single agent that simultaneously:

- learns without task identities or task boundaries;
- retains old competence and remains able to acquire new competence;
- constructs useful state under partial observability;
- learns and safely uses a changing world model;
- discovers and curates reusable predictions, subtasks, and options;
- explores an effectively unbounded world under fixed compute and memory; and
- operates indefinitely on embodied, non-episodic experience.

The current Alberta Framework is a broad and valuable component library, but it
is not yet evidence for that complete agent. Its strongest contributions are
the online JAX kernels, immutable and checkpointable state, explicit
average-reward learners, Horde/GVF surfaces, bounded feature-lifecycle
mechanisms, and single-transition APIs. Its largest weakness is not a missing
class; it is the gap between individually tested components and a causally
closed, end-to-end learning system.

The most important findings are:

1. **The committed evidence does not support “all 12 steps are implemented and
   benchmarked.”** The vendored tree omits the upstream benchmark, example, and
   output trees. Thirty-nine benchmark-dependent test modules are excluded at
   collection time. The surviving suite mainly establishes API and kernel
   behavior. A concurrent README edit now states this boundary much more
   accurately.
2. **The nominal full agent is only partially integrated.** World-model,
   dreaming, Horde, IA, and recurrent-perception paths are optional and disabled
   by default. Feature and cumulant discovery are not in its learning loop,
   and IA recommendations are returned as diagnostics rather than affecting
   `PrototypeAgent` action selection. Bounded option-model planning now exists,
   but lacks a promoted matched-resource benefit result.
3. **The current `UPGDLearner` is not canonical protecting UPGD.** It uses
   `|w g|`, takes an ordinary SGD update, then adds utility-scaled perturbation
   to trunk weights. Canonical UPGD uses signed Taylor utility and gates both
   gradient and noise. Existing results must be preserved under an honest
   variant name; a faithful reference implementation should be added
   separately.
4. **Plasticity and forgetting are different failures.** Resetting or
   perturbing unused capacity can keep an agent trainable without preserving
   old behavior. Replay or parameter protection can preserve old behavior
   without keeping the network trainable. Both must be measured and addressed.
5. **State construction is at least as important as plasticity repair.**
   Forager results show recurrent or trace-based state can matter more than
   several plasticity mitigations in a continuing, partially observable world.
   Alberta's current fixed-weight GRU does not learn what history to retain.
6. **Prediction error, epistemic surprise, learning progress, and the
   paper-specific delight signal are not interchangeable.** Raw error is
   vulnerable to stochastic “noisy TV.” In Delightful Policy Gradient,
   delight is advantage multiplied by action surprisal; it is an actor-gradient
   signal, not the user-requested “does this gradient spark joy?” audit and not
   a generic replacement for novelty, model uncertainty, or parameter utility.
7. **World models should support both retention and planning, but recent
   evidence warns against assuming that a remembered world automatically
   produces a remembered policy.** The strongest next architecture therefore
   needs component-level retention probes and an actor-rehearsal channel, not
   only world-model replay.

The recommended direction is a measured integration program: establish an
evidence contract, add faithful optimizer baselines, make state construction
learnable, separate the learning-value signals, build a calibrated continual
world model with bounded replay, connect feature/subtask/option lifecycles to
actual control and planning, and only then advance through increasingly
realistic continual and embodied benchmarks.

## Post-audit progress in this working tree

Work performed after the committed-snapshot audit has closed several
correctness and evidence-contract gaps without changing the overall verdict:

- a frozen 30-seed recurring pair-feature artifact now demonstrates retention,
  deployed-bank eviction, and faster recurrence for supplied pair targets while
  counting its exhaustive candidate archive;
- a frozen 30-seed decision-fidelity probe shows that the lifetime-statistics
  transition model ranks short action menus far better than untrained and
  fixed-memory raw-feature baselines on a deterministic one-dimensional task;
- a frozen 30-seed recurring two-agent artifact demonstrates visibly cued
  fixed-memory coadaptation, explicitly separated from IA;
- STOMP now keeps environment return separate from pseudo-reward, uses a
  coherent discounted differential semi-MDP target, and consumes option models
  in bounded planning backups;
- OaK transition ownership, option-start accounting, active-option curation,
  Prototype primitive-action credit, and dream isolation have focused
  regressions; and
- a pinned Foragax runner and protocol importer are present, and a
  stage-conformant five-seed field-of-view tuning stage selected `step3e3`;
  the 30-seed evaluation lane remains incomplete: the Alberta worker has no
  completed batch or report and is no longer active, while its matched
  official-DQN and relearning companions are quarantined and there is no
  completed comparison report; and
- a separate four-seed RTU-RTRL GPU development run completed 500,000 steps
  with FOV tail-EMA AUC 1.550 mean and 0.324 sample SD, but its exact receipt
  is explicitly nonpromoting because selection was not preregistered and
  source closure is incomplete. A reconciled unsealed DQN receipt gives a
  descriptive +0.331 matched-seed mean difference, but it was configured after
  RTU output and has unmatched runtime, representation, resources, and update
  work, so no admissible paired baseline exists.

These are narrow advances, not an integrated completion result. The broader
legacy/default pairwise discovery path still fails a 10× scale-shock gate. An
opt-in scale-robust v2 package passed its immutable frozen narrow comparison,
but registered source drift now makes that artifact invalid for the current
learner. It also retains visible context, an exhaustive finite pair archive,
and one fixed learner initialization; it does not close general feature
discovery or control. State construction in the full prototype is still fixed
rather than learned, the representation/model/planning lifecycle is not
closed, and no single bounded agent life exercises and ablates all required
links.
[RESEARCH_STATUS.md](RESEARCH_STATUS.md) is the live evidence matrix.

## How evidence was evaluated

This review prioritizes primary papers, proceedings, official project pages,
and author repositories. Sources were searched through 2026-07-30. Because
several relevant 2026 papers are very recent, publication status is part of
every recommendation.

Evidence labels used below:

| Label | Meaning |
|---|---|
| **A** | Peer-reviewed or accepted work with broad experiments or strong formal results |
| **B** | Peer-reviewed/accepted work with narrower scope, or a well-supported result with public artifacts |
| **C** | Recent preprint or workshop result that is promising but not independently established |
| **V** | Vision, position, survey, or architecture proposal rather than an algorithmic validation |

These labels assess confidence in the cited result, not the importance of its
research question. A new result can be highly relevant and still be **C**.

## What the Alberta Plan actually asks for

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) is a
research sequence, not a claim that known components can simply be assembled.
Its core constraints are:

- ordinary, temporally ordered experience rather than repeated training epochs;
- continuing operation rather than train/freeze deployment phases;
- bounded computation per environment step;
- bounded representational resources in a world larger than the agent;
- partial observability and learned state;
- average-reward prediction, control, and planning;
- learning and planning on each step, without large blocking phases;
- learned subtasks, options, models, and search control; and
- eventual partnership with another intelligent actor.

The twelve steps deliberately move from individually understandable learning
problems to integrated prototype systems:

1. continual supervised learning with given features;
2. continual feature discovery under a fixed resource budget;
3. continual GVF prediction, including learned recurrent state;
4. continual actor-critic control with feature discovery;
5. average-reward GVFs;
6. continuing-control benchmarks;
7. average-reward planning;
8. a one-step model-based prototype with a feedback cycle among representation,
   model learning, planning, and feature ranking;
9. learned search control and exploration;
10. STOMP: feature to subtask to option to model to planning;
11. OaK: lifelong utility, removal, replacement, and option composition;
12. an intelligence-amplification partnership.

“A module exists with the step's noun in its name” is therefore a much weaker
claim than completing the step. Completion requires the mechanism, its
integration into the prescribed feedback loop, and evidence in an appropriate
continuing problem.

## Repository audit

### Evidence boundary

At commit `c28144b`, the local [README](README.md) said both that all twelve
steps were implemented and benchmarked and that every component updated on
every time step. Those sentences were too strong for the vendored snapshot.
During this review, a concurrent working-tree patch replaced them with a more
accurate statement: the repository contains mechanisms spanning all twelve
steps, while the integrated continual-learning result and reproducible
evidence remain incomplete. That correction should be retained and committed
with the evidence-status work.

[VENDORING.md](VENDORING.md) explicitly says that `benchmarks/`, `docs/`,
`examples/`, `outputs/`, and `scripts/` were not vendored. Consequently,
[tests/conftest.py](tests/conftest.py) excludes 39 test modules whose import-time
dependencies reach those omitted trees. The exclusions include Step 2
completion/evidence gates, external suites, performance suites, production-step
tests, throughput checks, and the remaining-plan gate. A passing local suite
cannot establish the claims those omitted experiments were intended to test.

The upstream
[ROADMAP at the vendored commit](https://github.com/lalalune/alberta/blob/2ac35333efae45cf969ce02ec1f2703476fed6c2/ROADMAP.md)
is appropriately more cautious. Among other boundaries, it records:

- Step 2 evidence on a narrow ordered/permuted-MNIST setting, including results
  where a fair MLP remains better on important metrics;
- open nonlinear, trace-bearing, and per-demon off-policy questions for Step 3;
- actor-critic results that remain provisional relative to SARSA;
- small benchmark gates for later planning and option steps;
- a primitive one-partner IA path rather than a learned multi-partner system;
  and
- a CartPole ceiling tie as a surrogate integration result, not embodied
  validation.

The correct interpretation is: **many Step 1–12 mechanisms have implementation
surfaces, but completion evidence remains open.**

### Validation snapshot

The complete test collection available during the review finished successfully
on Python 3.12.3: **1,519 passed and 55 skipped in 1,302.94 seconds**. Ruff also
reported clean. This is strong implementation-health evidence. It is not the
missing scientific evidence: all 51 canonical Step 1/2 evidence tests skipped
because their required output artifacts were absent, and the collection hook
excluded the 39 benchmark/example-dependent modules described above.

### Current strengths

The framework already has unusually useful foundations:

- **Streaming semantics.** Most learners consume a single transition or sample,
  and scan wrappers preserve temporal order.
- **JAX discipline.** Explicit PRNG keys, immutable PyTrees, `jax.lax.scan`, and
  serialization surfaces make long deterministic experiments and checkpoint
  recovery practical.
- **Adaptive learning machinery.** LMS, IDBD, Autostep, TD variants, ObGD,
  normalization, continual backpropagation, UPGD-inspired learners, and several
  feature lifecycles are available for controlled comparison.
- **Prediction and control coverage.** Linear and nonlinear learners, Horde,
  off-policy TD variants, SARSA, actor-critic, average-reward learning, and
  continuous-control surfaces cover much of the early Plan.
- **Bounded resources.** Feature managers, fixed-size dictionaries, utility
  estimates, and bounded buffers take the “agent smaller than the world”
  constraint seriously.
- **World-model and imagination primitives.** Action-conditioned one-step
  prediction, latent-model experiments, disagreement helpers, guarded dreaming,
  and candidate-scoring functions provide good seams for future work.
- **Temporal abstraction primitives.** Subtasks, intra-option policies,
  termination, outcome models, extended actions, OaK utility tracking, and an
  option keyboard are concrete rather than purely architectural.
- **Embodiment-compatible boundary.** The robot stack can consume the continual
  RL subset without moving heavy research internals into the elizaOS runtime.

These strengths should be preserved. The roadmap should connect and validate
them, not replace them wholesale.

### Step-by-step status

| Step | Kernel status | Integration status | Evidence in this vendor | Assessment |
|---|---|---|---|---|
| 1 | Strong linear/adaptive optimizer coverage | Used by later learners | Unit and smoke coverage; full replication artifacts omitted | **Implemented; re-establish benchmark evidence** |
| 2 | Extensive nonlinear and feature-lifecycle machinery | Several alternative paths, no single accepted lifecycle | Critical benchmark and theory gates omitted; upstream evidence mixed | **Active research, not complete** |
| 3 | Horde, mixed and independent demons, linear off-policy TD | Optional Horde in full agent | Nonlinear/off-policy/trace combinations remain open | **Substantial partial implementation** |
| 4 | SARSA, actor-critic, continuous variants | Step 1–4 pipeline exists | Upstream actor-critic is provisional versus SARSA | **Implemented surfaces; weak closure evidence** |
| 5 | Average-reward prediction machinery | Available to later components | Small gates | **Kernel implemented** |
| 6 | Differential continuing control | Used by STOMP/OaK | Small/surrogate environments | **Kernel implemented; benchmark breadth missing** |
| 7 | One-step planning/Dyna helpers | Not the controlling path of the full prototype | Small gates | **Partial** |
| 8 | Fixed recurrent perception and one-step model | Optional in `PrototypeAgent`; feedback cycle is absent | CartPole/smoke evidence | **Prototype surface, not the Plan's integrated loop** |
| 9 | Dream guards and candidate scoring | Prototype samples random anchors/actions and does not use the scoring surface | No learned search controller | **Guarded replay, not learned search control** |
| 10 | Subtasks, options, outcome models, extended actions | STOMP executes options and consumes learned option models in bounded backups | No promoted matched-budget planning benefit | **Mechanism implemented; defining control evidence missing** |
| 11 | Utility EMA, Python-level curation, keyboard functions | Manual/periodic curation; keyboard not the base action path | Small gates | **Mechanism sketch** |
| 12 | Exo-cerebellum/exo-cortex APIs | Recommendation and augmentation are diagnostics, not control inputs | One-partner surrogate | **Primitive prototype** |

### Ranked integration gaps

#### 1. The full-agent label is misleading

[`PrototypeAgent`](alberta_framework/core/prototype_agent.py) describes itself
as integrating all twelve steps, but its default configuration instantiates
only a small OaK/STOMP path. The world model, dreaming, Horde, IA, and GRU are
all optional and disabled by default. The separate
[`AlbertaPipeline`](alberta_framework/pipeline.py) explicitly covers Steps 1–4
only. Neither object is the complete feedback system described by the Plan.

#### 2. State construction has one causal model-gradient training lane

The legacy `GRUPerceptionConfig` remains an honestly named fixed-weight
echo-state baseline. `PrototypeAgent` now also accepts the common
`StateBuilder` contract with identity, fixed-trace, and online-gated recurrent
implementations. `start()` advances recurrence once and caches the exact raw
observation, representation, primitive action, and lifecycle/generation token;
`act()` only returns that cached decision and cannot advance recurrence.
Explicit transitions advance a builder once per bootstrap observation, reset
episode-local recurrence at a boundary, and consume the post-reset decision
observation once. Eager, JIT, scan, and checkpoint tests cover those counts.

The online-gated builder now exposes a source-bound proposal and destination
commit boundary. An opt-in Prototype lane uses the bounded ensemble's
predict-before-update representation gradient: it proposes against the exact
builder state that emitted the modeled representation, then commits only the
parameter delta into the already-advanced builder state, preserving its hidden
state and sensitivity. A second opt-in boundary accepts decision-bound,
independently attested objective, retention, and safety probes and stores the
delta only when the complete gradient audit literally reports `sparks_joy` and
its finite-precision application succeeds. Missing, stale, incomplete, or
non-finite sidecars veto only builder learning, not the real control/model
transition.

This is causal mechanism integration, not a learned-state result. The current
producer is one world-model objective; balanced prediction, Horde, critic, and
control gradients are not yet wired into this Prototype lane. Until those
ablations and the matched Forager gate exist, the builder must not be called a
learned-state success.

#### 3. Representation discovery is not in the prototype loop

The repository contains sophisticated feature discovery, compositional
features, resource managers, future utility, and cumulant discovery. The full
prototype does not instantiate or update them. It therefore lacks Step 8's
required cycle:

`features → world-model quality → planning/control utility → feature ranking → features`.

#### 4. The world-model ensemble is integrated but not externally calibrated

WP4 now also has a concrete shallow reference rather than only an architectural
placeholder. `ShallowRidgeWorldModel` is an action-indexed affine regularized
follow-the-leader model over grounded next observation, reward, and
continuation. It retains fixed Gram/cross sufficient statistics, predicts
before updating, solves only the selected action block, rejects corrupted or
indefinite statistics atomically, has no RNG or replay state, and exposes a
pure one-step score using a supplied linear successor value. This is an L0
interpretable baseline—not a reproduction of an ICML result, a latent model,
or MPC—and it has no matched retention/control comparison yet.

The prototype now has mutually exclusive legacy single-model, bounded ensemble,
and ensemble-plus-model-replay lanes over representations and discrete actions.
The ensemble uses
distinct initialization and persistent bootstrap RNG/masks, predicts before
each member update, emits a causal representation gradient, and keeps
epistemic disagreement, a residual-variance proxy, learning progress, and
change probability separately typed with explicit warm-up availability. Its
resource budget and checkpoint include every member, calibrator, RNG, mask,
and counter. In the prototype:

- the authoritative `PrototypeTransition` path routes an explicit environment
  discount to real control, world-model, and IA exo-cortex targets, while the
  old `update` wrapper retains its configured-gamma compatibility behavior;
- legacy dream proposals receive no ensemble disagreement and therefore
  default uncertainty to zero; dreaming is deliberately disabled for the
  ensemble lane until uncertainty and rollout-validity gates are calibrated;
- random anchors and uniformly random actions drive imagined updates;
- accepted dream backups consume the predicted discount; and
- aggregate error EMA is thresholded without calibration by state/action
  region.

The replay composition commits the real ensemble update, causal typed-signal
record, fixed-quota dual-memory sample, and model-member rehearsal as one
transaction. Replay has separate mask RNG, masks, update counters, and event
counters; it cannot update causal signal calibration, residual variance, the
real RNG/counters, actor, critic, or state builder. In Prototype, only the
commit-gated real representation gradient can reach the builder or literal joy
audit. Stored transitions retain the final/bootstrap observation—not the
post-reset decision observation—and representation versions make stale/future
samples explicit. This is bounded L0 composition; there is no replay-retention
or control comparison and no actor rehearsal.

The explicit transition boundary now distinguishes the final/bootstrap
observation from the next post-reset decision observation. World-model, Horde,
and Bellman targets consume the former; OaK/STOMP selection, IA recommendation,
and the cached next action consume the latter. Positive-discount truncation
interrupts an active option after its final update without recording a censored
option-model completion. This repairs episodic/autoreset semantics and
integrates a bounded development ensemble, but the variance output is still an
explicitly uncalibrated residual proxy rather than certified aleatoric
uncertainty. There is no state/action-region calibration, recurrent latent
dynamics, or validated multi-step dream consumer yet.

This is a useful smoke model, not yet a reliable continual world model or a
learned search-control process.

#### 5. STOMP option-model planning is bounded but not validated

[`STOMPAgent`](alberta_framework/core/options.py) updates external return,
duration, discounted baseline mass, discount, and next-state outcome models
when an option terminates, then consumes those models in an explicitly bounded
number of differential semi-MDP backups. This closes the mechanism-level
`model → planning` edge. It does not establish that learned option models
improve held-out lifetime control over matched model-free option execution.

#### 6. OaK curation is manual and narrow

OaK's utility is an EMA of pseudo-reward while an option executes. `curate()`
runs at Python level, finds the worst option, swaps its hand-specified feature
index, and resets selected arrays. That is a useful testable mechanism, but it
does not yet measure the counterfactual contribution of a feature, subtask,
option, or model to lifetime reward or planning accuracy. The option keyboard
is callable, but its learned chord does not govern the base policy.

#### 7. Prototype IA outputs do not yet alter Prototype actions

`PrototypeAgent` returns IA augmentation and recommendations from `update()` as
diagnostics, while its next controller action remains OaK's action. The
standalone recommendation protocol can alter a partner's action and has a
historical held-out **valid rejection**: reward uplift and both augmentation
controls passed, but action-changing intervention prevalence missed its frozen
threshold. The archived v1 rejection remains historical evidence; its prior
exact replay on the already-consumed schedule is nonpromoting, and subsequent
`average_reward.py` drift makes current-source compatibility invalid. The
p=0.75/seeds-60–89 v2 lifecycle is an unissued, permanently development-only
contract: its self-issued plan has no trusted external pre-run chronology, and
`internally_accepted` is hard-coded false. Any future acceptance claim requires
a new schema, untouched seeds, complete shards, and an external chronology
anchor. Until the composed
Prototype consumes partner information through an
auditable fusion rule, that composition demonstrates companion learning rather
than intelligence amplification.

#### 8. “Every component updates every step” is false as written

Optional components do not update when absent. OaK curation occurs
periodically outside JAX. Option models update only at option termination.
Dreaming is disabled by default and gated after warm-up. Temporal uniformity
means bounded, continuing operation with a defined allocation policy; it need
not be marketed as a literal gradient update to every parameter on every
sample.

### Continual evaluation mechanism

The working tree now contains a strict reconstructing v2 continual-evaluation
report plus a bounded learner-neutral scalar streaming executor. Together they
cover prequential performance, post-change adaptation AUC, recovery,
backward/forward transfer, forgetting, immediate change-point stability, tail
latency, resource accounting, explicit safety availability, and final
component/plasticity diagnostics. The executor owns regime scheduling and
held-out probes, enforces predict-before-update ordering, detects
serializer-visible learner mutation during predictions/probes and source-state
mutation during updates, and requires canonical deterministic checkpoints. A
candidate plus at least two exact-budget baselines is still required and regime
metadata stays evaluator-only. Its portable bound artifact hashes the exact
evaluator configuration (including stream/probe digests and learner snapshots)
and the reconstructing metric core, then cross-validates protocol, budget,
condition order, trace length, and latency semantics.

The working tree also has a strict continuing-control evaluator with an exact
`PrototypeAgent` adapter. It gives every condition an independent functional
environment, binds each action to its dispatched observation and decision ID,
and reconstructs direction-aware longitudinal return, adaptation, recovery,
stability, final held-out action score, forgetting, transfer, and worst-window
metrics from raw traces. Protocol-owned exposure rows, thresholds, references,
and applicability make sparse comparisons explicitly unavailable instead of
silently zero. Its v2 report and checkpoints bind canonical environment,
probe, learner, budget, and metric-core identities.

Every constructed report remains `not-assessed`. There is no completed
Prototype candidate/baseline report, only simple scalar/control baselines, no
factorial runner, no longitudinal component traces, and no accelerator-memory,
energy, or safety backend. A strict paired multi-seed campaign runner now binds
seeded evaluator identities, embeds every raw report, preserves unavailable
pairs, and reconstructs deterministic stratified-bootstrap intervals; no
Prototype campaign has been executed through it. The held-out control probes
score one action rather than a frozen-policy rollout, and fresh-per-regime,
oracle-data, stationary-multitask, and realized-resource-matched references are
still missing. These mechanisms become evidence only after those gaps are
closed in a preregistered protocol.

The concurrent [research-status matrix](RESEARCH_STATUS.md) is also a strong
starting point: it separates mechanism, learning, comparison, and integration
evidence, marks every Plan step partial, and proposes a fail-closed recurring
two-agent gate. Its numeric thresholds are explicitly local acceptance
criteria, not claims from the Alberta Plan. They should be preregistered,
sensitivity-tested, and connected to immutable artifacts before any status is
promoted.

## UPGD: canonical method, local divergence, and implications

### Canonical UPGD

The definitive source is Elsayed and Mahmood,
[“Addressing Loss of Plasticity and Catastrophic Forgetting in Continual
Learning”](https://arxiv.org/abs/2404.00781), accepted at
[ICLR 2024](https://openreview.net/forum?id=sKPzAXoylB). The
[official implementation](https://github.com/mohmdelsayed/upgd) is MIT
licensed.

For a parameter \(w_i\), gradient \(g_i\), and optional diagonal Hessian
approximation \(h_{ii}\), the immediate utility approximations are:

\[
u_i^{(1)} = -g_i w_i
\]

\[
u_i^{(2)} = -g_i w_i + \frac{1}{2}h_{ii}w_i^2.
\]

After an EMA, bias correction, normalization, and sigmoid scaling to
\(\bar u_i\), protecting UPGD applies:

\[
w_i \leftarrow w_i -
\alpha\,(g_i+\xi_i)\,(1-\bar u_i),
\qquad \xi_i \sim \mathcal N(0,\sigma^2).
\]

High-utility weights are protected from both the ordinary task gradient and
the perturbation. The paper also studies a **non-protecting** variant that
gates only noise, UPGD-W with decoupled weight decay, AdaUPGD, first- and
second-order utilities, weight- and feature-wise utility, and local/global
normalization.

The pinned source audit found three distinctions that a parity implementation
must not blur. Algorithm 1 uses the bias-corrected global maximum and a
one-\(\alpha\) direction. The repository README instead divides corrected
utilities by the maximum uncorrected EMA and applies \(2\alpha\) to the gated
gradient/noise direction while leaving decoupled decay at \(\alpha\). The
experiment code uses the uncorrected global maximum but returns to a
one-\(\alpha\) direction. Local normalization also differs: literal Algorithm
2 divides the corrected numerator by a row norm of the uncorrected EMA, whereas
the released experiment code normalizes the corrected utility. These are
source discrepancies, not interchangeable definitions. The JAX reference
therefore exposes named equation profiles plus a numerically guarded production
profile; fixed-noise tests pin each profile separately.

Its evidence is particularly relevant to Alberta: strict online updates,
prequential evaluation, no replay, and no task boundary supplied to the
learner. It covers long permuted/label-permuted image streams and PPO control.
It does not establish a universal optimizer: utility ignores parameter
interactions, tuning is dataset-specific, synthetic task changes dominate the
study, and world models, recurrent state, replay combinations, and
transformers are outside its core evidence. One original best configuration
uses \(\sigma=0\), an important warning against treating perturbation as
intrinsically beneficial.

### What the local learner does

The local [`upgd.py`](alberta_framework/core/upgd.py):

- attributes UPGD to the continual-backpropagation line rather than Elsayed and
  Mahmood;
- tracks \(|w_i g_i|\), discarding the sign needed by first-order Taylor
  utility;
- normalizes utility per layer by its maximum and applies a power schedule;
- takes an ordinary SGD step first;
- adds utility-scaled noise after that step, so the task gradient is never
  protected; and
- perturbs hidden trunk weights, while biases and readout heads receive
  ordinary updates.

It also includes useful original machinery—ObGD, warm-up/ramp/interval
schedules, Rademacher noise, meta-plasticity, readout variants, and unit
recycling. Those additions make it more important, not less, to give the
algorithm an honest identity. Existing results should not be invalidated by a
silent semantic rewrite.

### Required resolution

1. Preserve the current implementation under a name such as
   `UtilityPerturbedSGD` or a documented `nonprotecting_power` mode.
2. Correct its literature attribution.
3. Add a small faithful JAX implementation of first-order protecting and
   non-protecting UPGD, UPGD-W, and optionally AdaUPGD.
4. Establish tiny-tree numerical parity against a hand implementation and the
   official PyTorch update.
5. Compare signed versus absolute utility, global sigmoid versus local power
   scaling, protecting versus non-protecting, Gaussian/Rademacher/no noise, and
   trunk-only versus all eligible parameters.
6. Report plasticity and forgetting separately. A single final accuracy number
   cannot show which problem was solved.

### Completed Input-permuted MNIST development diagnostic

The published-configuration runner completed 10 matched one-million-step
UPGD-W/AdamW seeds. UPGD-W mean online accuracy was
`0.7791470803916454` (SE `0.000055690729820870456`) and AdamW was
`0.7190002817213534` (SE `0.0005943125024635892`). Their paired descriptive
mean difference was `0.06014679867029188` (sample SD
`0.0018825070977402044`, SE `0.000595301014029226`), positive for 10/10
seeds. UPGD-W differed by about `-0.000853` from the paper's approximate
figure read-off; AdamW differed by about `+0.039`, an explicit reproduction
gap. Exact early/late-task summaries are recorded in
[CONTINUAL_LEARNING_EVIDENCE.md](CONTINUAL_LEARNING_EVIDENCE.md).

The canonical artifact
`outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json` passes strict
structural validation and is bound by
`outputs/upgd_ipmnist/nonpromoting_receipt.v2.json`; its byte-preserved
predecessor is `nonpromoting_receipt.v1.json`. The original
`outputs/upgd_ipmnist/results.v1.json` is preserved; its validator failure is
limited to a note that did not preserve the exact 10-vs-20-seed limitation.
Neither artifact is scientific evidence. The run used half the publication's
20 seeds, changed stream seeding, task-boundary logging, and numeric details,
and did not bind worker source, the full import closure, commands, environment,
or dataset bytes at execution time. A post-hoc receipt cannot supply that
attestation. The run therefore supports no inferential, SOTA, or Alberta Plan
claim and cannot settle the optimizer questions above. A promotable
replication requires a fresh source-bound full-20-seed execution, not appended
seeds.

The active future execution surface is now the namespaced v3 contract in
`alberta_framework/benchmarks/upgd_ipmnist_v3.py`: issue one immutable plan,
run exactly one learner/seed per shard, then merge only the complete planned
Cartesian product while binding every shard's bytes. The plan closes the
configuration, selected hyperparameters, deviations, exactly 20 fresh
operator-reserved seed IDs, data identity and locators, runtime, semantic commands, and static
transitive local import closure. No v3 plan has been issued, no v3 shard or
artifact exists, and no fresh v3 seed has been consumed. Because seed
reservation and execution lack independent attestation, v3 is permanently
nonpromoting. See [UPGD_IPMNIST_V3_RUNBOOK.md](UPGD_IPMNIST_V3_RUNBOOK.md).

### Follow-ons and adjacent perturbation methods

| Work | Evidence | Mechanism and finding | Recommendation |
|---|---:|---|---|
| [ReCL](https://arxiv.org/abs/2411.06916) | C | Reconstructs samples from converged weights; reports that vanilla UPGD can be weak in task-boundary multi-epoch settings | Use as evidence that UPGD is protocol-dependent; do not adopt model inversion yet |
| [FOGO](https://arxiv.org/abs/2606.10406) | C | Stores compressed gradient-direction history and orthogonalizes conflicting updates; outperforms UPGD in reported boundary-based Class-IL tests | Watch and reimplement only after stronger evidence/licensing |
| [Loss of plasticity / Continual Backprop](https://www.nature.com/articles/s41586-024-07711-7) | A | Replaces low-utility mature units and zeroes outgoing weights; strong long-stream plasticity evidence | Required baseline; already partly present |
| [Self-Normalized Resets](https://openreview.net/forum?id=G82uQztzxl) | A/B | Resets only when inactivity is statistically surprising relative to a unit's history | Add as a low-knob reset baseline |
| [L2-to-init / regenerative regularization](https://arxiv.org/abs/2308.11958) | B | Keeps parameters near initialization; often strong in RL and Forager | Required cheap baseline |
| [Weight Clipping](https://arxiv.org/abs/2407.01704) | B | Bounds parameter growth after each update | Required cheap, noise-free baseline |
| [Spectral regularization](https://arxiv.org/abs/2406.06811) | A/B | Controls layer spectral norms and gradient conditioning | Add as a mechanistically distinct baseline |
| [C-CHAIN](https://arxiv.org/abs/2506.00592) | B/C | Links NTK-rank loss to output churn and penalizes out-of-batch churn | Benchmark after core baselines; canonical code lacks a reusable license |
| [FIRE](https://arxiv.org/abs/2602.08040) | A/C | Boundary-time projection toward isometry | Useful diagnostic comparator; boundary dependence conflicts with the primary setting |
| [Dynamical Isometry / AdamO](https://arxiv.org/abs/2606.09762) | C | Regularizes Jacobian singular values and proposes an Adam-style optimizer | High-priority watch/reimplementation study |
| [Calibrated Partial Resets](https://arxiv.org/abs/2607.24996) | C | Pulls low-utility neurons partway toward initialization; strong very-long-run claims | Experimental lane only; the preprint was three days old at this snapshot |

“Universal perturbation” usually refers to
[universal adversarial perturbations](https://arxiv.org/abs/2005.08087): a
single input-space attack that fools many examples. That is not a continuation
of UPGD's beneficial parameter perturbation. It belongs in robustness testing,
not in the continual optimizer merely because the terms overlap.

## What the broader literature contributes

### 0. Continual learning is a lifetime protocol, not a task-list score

[A Definition of Continual Reinforcement
Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html)
defines the problem through an agent's indefinitely extended learning/search
process rather than a finite list of labeled tasks.
[Rethinking the Foundations for Continual Reinforcement
Learning](https://openreview.net/forum?id=cNRVG7y63A) further motivates
history-process and lifetime-regret semantics for continuing agents. These
works support the Plan's task-free framing: task identities are evaluator
metadata, not learner inputs.

The most useful protocol and benchmark references form a ladder:

| Work | What it establishes | Limitation and use here |
|---|---|---|
| [Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606) | Sparse initialization, ObGD, traces, parameter-free normalization, and online scaling can make strict single-pass deep RL competitive in reported domains | Preprint and not a full retention study; reproduce exact algorithmic parity as an early baseline |
| [Deep RL with Gradient Eligibility Traces](https://rlj.cs.umass.edu/2025/papers/Paper302.html) | Forward-replay and backward-streaming trace variants can improve online control | Compare with Alberta's forward/backward views; traces alone do not solve forgetting |
| [Continual World](https://arxiv.org/abs/2105.10919) | Standardized robotic task sequences and transfer/forgetting measures | Segmented simulator tasks; an intermediate gate, not the final task-free test |
| [Where is the Truth?](https://proceedings.mlr.press/v267/busch25a.html) | Sequentially changing confounders can defeat methods that handle jointly observed nuisance factors | Add sensor/nuisance correlation shifts to prevent shortcut success |
| [AgarCL](https://arxiv.org/abs/2505.18347) | A non-episodic, stochastic, partially observed, evolving control stream; reported neuron-maintenance methods add little | One fresh benchmark, but a valuable naturalistic falsifier |
| [Forgetting, Ignorance or Myopia](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6b44ee74539ea77d6a0d50d468724371-Abstract-Conference.html) | A computationally heavy learner may look stable while missing incoming stream items | Always report observations processed/dropped and update latency |

### 1. Plasticity is necessary but not retention

The Nature
[loss-of-plasticity study](https://www.nature.com/articles/s41586-024-07711-7)
distinguishes inability to relearn from catastrophic forgetting and shows
dormancy, rank loss, and parameter growth over long streams. UPGD, continual
backpropagation, L2-to-init, clipping, resets, churn penalties, and isometry
regularization all address aspects of trainability.

None is a complete memory mechanism. A method can keep gradients healthy while
overwriting an old policy. Conversely, a large replay store can preserve old
samples while the actor loses its ability to turn them into competent
behavior. Alberta should therefore track at least:

- current-task adaptation and recovery time;
- retained old-task competence;
- forward and backward transfer;
- dormant units and activation diversity;
- effective/stable rank and sampled empirical-NTK rank;
- gradient norms and gradient alignment;
- parameter distance from initialization;
- policy/value churn on a fixed probe set; and
- fresh-head or reset-probe learning speed.

No one metric is an adequate proxy.

### 2. Partial observability makes state construction central

[Forager](https://arxiv.org/abs/2605.01131) (**C**, RLC 2026-era work) is
exceptionally well aligned with the Alberta problem: continuing average reward,
limited field of view, hidden nonstationarity, constant environment memory, and
an unending task generator. Its key empirical message is that common
plasticity mitigations help less than useful state construction. Simple traces
and recurrent state can outperform feed-forward agents that remain plastic.

This directly challenges the current fixed random GRU. Alberta needs a
state-builder benchmark in which:

- recurrence is trained from prediction and control signals;
- the state is evaluated for sufficiency, not merely dimensionality;
- hidden changes can be anticipated rather than only tracked after reward
  errors; and
- state utility is attributed causally to predictions, control, and model
  accuracy.

The cheap Forager/Foragax environment should become an early mandatory gate
before expensive robotics.

The in-tree stationary causal-map planner is a useful L0 comparator, not
closure of that gate. It learns a relative map, reward means, and respawn
timings from ordinary observations while using the task's public 15×15
toroidal movement structure; it is neither a learned recurrent state builder
nor a nonstationary-retention result. The field-of-view tuning stage has
selected `step3e3`, but the 30-seed evaluation lane remains incomplete at this
snapshot: the Alberta worker produced no batch or report and is no longer
active, while the matched official-DQN and relearning directories are
quarantined under superseded execution contracts. The separate four-seed
RTU-RTRL development receipt and its post-output, resource-unmatched DQN
comparison are useful development diagnostics, not substitutes for the
missing paired protocol. Execution manifests and unsealed development
receipts without completed matched reports are not performance evidence.

Two subsequent open CPU screens completed all candidates on the same two
consumed development seeds. Under their fail-closed v4 aggregates,
`DQN_LN-common-control` ranked first in the feed-forward set (mean FOV
tail-EMA AUC `1.49084`) and `PPO-RTU_LN_128_1_relu` ranked first in the
stateful set (`1.78110`). Candidate budgets are not necessarily matched, the
rankings cannot be compared across screens, and the stateful RTU-PPO path
retains its documented action/environment RNG reuse confound. A content-level
fixed-action direct/wrapper parity trace also matched exactly, but the external
executor receipt is explicitly unverified and nonpromoting. These results are
useful implementation diagnostics only; they do not replace the missing
matched held-out Forager protocol or support a learned-state, superiority, or
SOTA claim.

Two accepted adjacent controls make the state claim falsifiable. A carefully
implemented [recurrent model-free
baseline](https://proceedings.mlr.press/v162/ni22a.html) beats specialized
POMDP methods on most of its tested suite, so a conventional trainable GRU
cannot be omitted. [Real-Time Recurrent Learning Using Trace
Units](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)
provides an online-credit-assignment alternative with favorable reported
compute. Both should be matched against identity, hand-designed traces, and the
echo-state baseline.

### 3. Continual world models need bounded, distribution-aware memory

Several complementary results matter:

- [Continual-Dreamer](https://arxiv.org/abs/2211.15944) (**A/B**, CoLLAs
  2023) shows task-agnostic selective replay and world-model exploration can
  reduce forgetting in MiniGrid and MiniHack.
- [Continual RL by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html)
  (**A**, ICML 2025) shows that a shallow Follow-the-Leader model plus MPC has a
  regret guarantee under its assumptions and can outperform deeper continual
  world-model baselines. This is a strong reason to keep a simple online model
  as a reference rather than jumping directly to a large latent model.
- [ARROW](https://arxiv.org/abs/2603.11395) (**C**, TMLR-era 2026) gives
  DreamerV3 a fixed-budget short-term FIFO plus long-term
  distribution-matching buffer. It reports less forgetting on Atari and
  comparable forward transfer on Procgen under matched memory.
- [The World Model Remembers, the Actor Forgets](https://arxiv.org/abs/2607.19749)
  (**C**, July 2026) reports component-level probes on three-seed MiniGrid
  chains: replay preserved measurable world-model knowledge while the actor
  forgot. Graded dream self-imitation restored and rehearsed behavior where
  policy-gradient learning in the same imagination did not. The code and
  preregistration trail are public, but the result is new, small-scale, and
  should be treated as a hypothesis to reproduce.
- [The Effectiveness of World Models for Pseudo-Rehearsal](https://arxiv.org/abs/1903.02647)
  (**B**) provides an older task-label-free generative-replay precedent.

Together these works argue for three explicit probes: world-model retention,
critic retention, and actor retention. Return alone cannot localize failure.
They also suggest a bounded dual-memory design and an experimental supervised
actor-rehearsal path. They do **not** justify uncritically replaying model
fantasies: uncertainty, termination, reward calibration, and selection quality
must be measured before imagined data can train behavior.

Replay itself requires controls:

| Work | Mechanism | Alberta implication |
|---|---|---|
| [CLEAR](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) | Bounded replay with policy/value cloning and off-policy correction | Store old action probabilities and values with real trajectory fragments; rehearse the actor explicitly |
| [Maximally Interfered Retrieval](https://papers.nips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html) | Select candidates predicted to be harmed by the incoming update | Compare interference priority with raw TD/model-error priority |
| [Dark Experience Replay](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | Reservoir replay plus old outputs | Preserve behavioral targets, not observations alone |
| [t-DGR](https://proceedings.mlr.press/v274/yue25a.html) | Generates complete trajectory samples rather than recursive one-step fragments | Add a trajectory-structured comparator for Continual World |
| [Replay Can Provably Increase Forgetting](https://proceedings.mlr.press/v330/mahdaviyeh26a.html) | Replay count can have a non-monotonic relationship with forgetting | Sweep replay rate and measure realized benefit; “more dreams” is not a default |
| [RECALL](https://arxiv.org/abs/2311.11557) | Reward-scale mismatch, plasticity loss, and offline shift can break replay | Normalize rewards and test actor distillation and distribution shift separately |

### 4. Surprise should estimate learnable information, not raw error

Prediction error is attractive because it is local and cheap, but it combines:

- **epistemic uncertainty:** reducible uncertainty caused by limited data;
- **aleatoric uncertainty:** irreducible stochasticity;
- model misspecification and representation failure; and
- genuine environment change.

Classic and modern results provide a useful progression:

| Work | Signal | Lesson for Alberta |
|---|---|---|
| [Intrinsic Curiosity Module](https://proceedings.mlr.press/v70/pathak17a.html) | Feature-space forward error | Useful baseline, but raw error can reward stochastic distractions |
| [Self-supervised exploration via disagreement](https://proceedings.mlr.press/v97/pathak19a.html) | Ensemble dynamics disagreement | Better approximation to epistemic uncertainty |
| [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html) | Planned future ensemble disagreement | Connect uncertainty to the world model and prospective search |
| [Progress Curiosity](https://proceedings.mlr.press/v119/kim20e.html) | Improvement in model learning | Prefer learnable regions, not permanently unpredictable ones |
| [Avoiding Noisy TVs with Aleatoric Uncertainty](https://arxiv.org/abs/2102.04399) | Mean/variance prediction | Explicitly discount irreducible stochasticity |
| [Learning Progress Monitoring](https://arxiv.org/abs/2509.25438) | Predicted reduction in error | Promising newer noise-robust exploration signal |
| [Surprise as a Signal for Plasticity and Metacognition](https://arxiv.org/abs/2606.31495) | Latent prediction error gates memory writes and confidence | Interesting proof of concept; frozen encoders, offline consolidation, and single-seed retention results limit confidence |

The practical conclusion is to maintain separate calibrated estimates for
epistemic disagreement, aleatoric variance, change-point likelihood, and
learning progress. Raw squared error should remain a diagnostic, not directly
become reward, replay priority, or perturbation magnitude.

### 5. “Does this gradient spark joy?” means auditing the candidate update

The user-requested criterion is introspective and gradient-level: given a
candidate update \(u_c\), does caller-supplied first-order evidence under an
explicit, caller-attested independence contract predict useful learning without
retention or safety harm, and is the update small enough to trust? The
implementation cannot infer independence from arrays: the caller must attest
that the objective comes from a separate validation/probe objective, retention
from a protected old-skill loss, and safety from a separately trained cost. For
raw-gradient mode \(u_c=-\alpha g\) is only a plain-gradient proposal; momentum,
preconditioning, clipping, and other optimizer transforms must pass their
already formed update. The caller sets `probe_independence_attested`; the
bounded implementation records:

\[
\Delta_o=\langle g_o,u_c\rangle,\quad
\Delta_r=\langle g_r,u_c\rangle,\quad
\Delta_s=\langle g_s,u_c\rangle,\qquad
a_k=-\frac{\langle g_k,u_c\rangle}
{\lVert g_k\rVert\lVert u_c\rVert}.
\]

Alignment uses scale-safe norms and normalized-coordinate dots accumulated by
a fixed balanced reduction. Conservative float32 roundoff intervals cover the
norms, normalization, products, and reduction. Non-finite derived dots/norms,
unresolved nonzero-input norms, raw/normalized sign disagreement, and
cancellation intervals that cross zero fail the evidence gate. The trust bound
uses the certified norm upper endpoint and the nonzero floor uses its lower
endpoint. Positive magnitude gates use the conservative dot-interval edge. For
exact zero thresholds, both the raw dot and certified normalized direction must
permit the verdict; a raw dot underflowed to exact zero can still use a resolved
normalized sign. Candidate cosines are clipped to \([-1,1]\); alignment gates
and factors use their certified lower endpoints. An exact \(1.0\) threshold
applies a four-machine-epsilon float32 tolerance to that lower endpoint, while
other thresholds remain exact. Candidate factors produce a tentative scale \(s\) and update
\(\tilde u=s u_c\). It accepts only when `probe_independence_attested` is true,
all three probes and all
eight named learning-value channels have explicit valid availability, evidence
is finite, both \(u_c\) and \(\tilde u\) meet objective, retention, and safety
magnitude gates, candidate directions meet their alignment thresholds,
\(u_c\) is inside the trust bound, and \(\tilde u\) clears the update-norm
floor. The returned scale is \(s\) on acceptance and zero otherwise. This
two-stage audit prevents soft scaling from violating a required
objective-improvement floor or rescuing a raw harmful candidate. The
elementwise float32 \(\tilde u\) receives fresh norm and dot certificates, so
rounding in \(s u_c\) cannot hide behind scalar-scaled candidate diagnostics. The factors
are separately reported and are not a reward/UX score or a sum over
learning-value channels. Inputs and outputs are detached; no meta-gradient
claim is made. Numeric controls must survive as finite normal float32 values.
This is a local linear audit, not proof that the realized nonlinear update will
improve the system.

The corresponding `apply_gradient_joy_update` boundary reassesses the
candidate internally, derives the effective stored delta after dtype cast and
parameter addition by promoting both stored endpoints to at least float32
before subtraction, and conservatively re-audits that delta under the same
evidence and thresholds. This prevents low-precision subtraction rounding from
understating an out-of-bound stored change. It atomically commits the exact shape-matched PyTree
only when both audits accept, all application values are finite, and at least
one stored parameter value actually changes. Its typed result keeps the
formed-candidate assessment, effective-delta assessment, and parameter change
`applied` distinct; overflow, non-finite parameters, updates lost entirely to
finite-precision addition, and quantization-altered probe verdicts remain
explicit no-ops. This closes the implementation boundary without upgrading the
local first-order audit into realized-improvement evidence.

That meaning is distinct from the related 2026 paper family's use of the word
“delight.” In [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608),
action
surprisal and delight are:

\[
\ell_t=-\log \pi_\theta(a_t\mid h_t), \qquad
\chi_t=A_t\ell_t.
\]

The actor term is weighted by:

\[
w_t=\sigma(\chi_t/\eta), \qquad
\Delta\theta \propto
\sum_t \operatorname{stopgrad}(w_t) A_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t).
\]

Positive advantage on an improbable action is a rare success and receives a
large gate. Negative advantage on an improbable action is a rare failure and
is suppressed. In general—and specifically across heterogeneous contexts—this
changes the expected policy-gradient direction, so it is not merely a
variance-reduction trick. In the paper's symmetric single-context bandit,
however, DG preserves the expected direction and reduces perpendicular
variance. The stop-gradient is part of the algorithm: differentiating through
\(w_t\) defines a different update.

[Does This Gradient Spark Joy?](https://arxiv.org/abs/2603.20526) then uses the
same \(\chi_t\) to decide whether a sample deserves a backward pass. A Kondo
gate compares delight with compute price \(c\):

\[
p(\text{backward}_t)=\sigma((\chi_t-c)/\tau).
\]

The paper reports that a target-rate implementation retains much of DG's
quality with far fewer backwards on MNIST contextual bandits and transformer
token reversal. Its own analysis identifies a **gambling pathology**: a rare
suboptimal action with high reward variance can produce a lucky outcome with
large positive delight, and the surprisal multiplier makes the false signal
stronger.

That compute result needs a strict qualifier. The paper reports logical
selected-backward counts and plots an analytical compute model under assumed
forward/backward cost ratios; it does not report a measured three-percent
kernel cost or wall-clock speedup. Separately, inspection of the released small
experiment at this snapshot found loss masks inside a full-batch autodiff
graph, which still computes every logit and constructs the full backward graph;
that is a source-inspection observation, not a paper result. A real sparse
implementation must screen with a detached forward pass, gather a static set of
survivors, recompute them, and differentiate only that subset. This is unlikely
to help Alberta's normal batch-size-one update.

The surrounding family adds:

- [Delightful Distributed Policy Gradient](https://arxiv.org/abs/2603.20521):
  suppress rare failures but retain rare successes from stale or faulty actors
  without requiring behavior probabilities. It is a biased robustness
  heuristic, not importance sampling; retain clipped ratios or V-trace when
  behavior probabilities exist;
- [Delightful Exploration](https://arxiv.org/abs/2605.13287): gate exploratory
  overrides using prospective expected improvement times host-policy
  surprisal. Its Pandora equivalence is exact only for revealed-value search;
  in noisy independent-arm bandits expected improvement is a
  value-of-perfect-information proxy that upper-bounds the one-step knowledge
  gradient; and
- [Delightful Gradients Accelerate Corner Escape](https://arxiv.org/abs/2605.11908):
  tabular convergence/corner-escape results plus an exact shared-function-
  approximation counterexample in which parameter coupling admits a suboptimal
  interior fixed point.

All are **C** at this snapshot. DPG reports 100-seed contextual-bandit MNIST,
30-seed token reversal, and 28 DeepMind Control tasks with only three seeds per
task. Kondo reports 30-seed MNIST, 10-seed token reversal, and a 30-seed
gambling study. The exploration and corner papers rely mainly on stationary
bandits, DeepSea, tabular analysis, and small MNIST recovery studies. None
demonstrates retention, backward transfer, reduced forgetting, world-model
learning, long-horizon continual control, robotics, or safety.

The official Apache-2.0
[`google-deepmind/egg`](https://github.com/google-deepmind/egg) repository
covers small token experiments for the first papers, not their full MNIST or
continuous-control evidence. Author-owned
[`iosband/trl-dg`](https://github.com/iosband/trl-dg) has no declared software
license. The later papers publish equations and embedded listings but no
separate reusable reference implementation.

The right Alberta use is therefore a bounded experiment:

1. implement DG only for the stochastic actor, with a detached gate;
2. compare against the identical actor-critic with ordinary policy gradient;
3. keep the critic, world model, safety-cost learner, and rare-failure memory
   ungated;
4. start with actor traces disabled: multiplying today's gate over an
   accumulated eligibility trace incorrectly reweights historical score
   gradients;
5. defer continuous action until the policy exposes the exact transformed
   action log-probability—differential surprisal depends on action units, and
   clipping a Gaussian action breaks the density contract;
6. add an aleatoric-variance veto for the gambling pathology;
7. test the Kondo gate as compute allocation only after DG reproduces;
8. measure quality per environment step, forward pass, backward pass, and wall
   clock; and
9. reject the method if it worsens safety, old-skill retention, calibration, or
   effective sample size.

If a later trace experiment gates each newly added score term,
\(e_t=\gamma\lambda e_{t-1}+\operatorname{stopgrad}(w_t)\nabla\log\pi_t\),
label it an Alberta extension rather than paper-equivalent DG. A minimum
catastrophe-update floor and change-triggered gate reset are likewise sensible
safety/nonstationarity extensions, not claims from the papers.

Neither paper-specific delight nor gradient joy should be folded into UPGD
utility. UPGD asks whether a **parameter** is currently useful. Paper-specific
delight asks whether an **actor sample** should be emphasized. The gradient-joy
audit asks whether a **candidate parameter update** passes objective,
retention, safety, and trust checks. Epistemic surprise asks whether an
**environment transition** can teach the model. These are different causal
objects.

### 6. Experience memory moves the stability problem; it does not remove it

[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
is a useful **V**-level complement to the Alberta Plan: agents should learn from
long, grounded streams; act and explore; use world models and temporal
abstraction; and obtain rewards from grounded consequences rather than only
static human data.

LLM-agent work provides useful memory and skill-library patterns, but most of
it is episodic, text-mediated, and non-parametric. It should not be mistaken
for solving online continual RL. Still, several patterns are directly useful
for a bounded experiential layer:

- [Voyager](https://arxiv.org/abs/2305.16291) uses an executable, composable
  skill library and automatic curriculum; options should likewise carry
  competence and failure metadata, not only feature prototypes.
- [Reflexion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)
  and [ExpeL](https://ojs.aaai.org/index.php/AAAI/article/view/29936) show how
  episodic outcomes can be compressed into reusable semantic procedures.
  Their language feedback can be wrong, so provenance and delayed outcome
  revision are mandatory.
- [How Memory Management Impacts LLM
  Agents](https://aclanthology.org/2026.acl-long.27/) finds that retrieved
  experiences can propagate both good and erroneous behavior. Retrieval
  precision, stale-memory detection, and negative transfer are therefore
  first-class metrics.

Two newer end-to-end studies sharpen the same point:

- [Experience-driven Lifelong Learning and StuLife](https://arxiv.org/abs/2508.19005)
  (**C**) separates exploration, long-term memory, skill abstraction, and
  internalization in a long-form agent benchmark.
- [When Continual Learning Moves to Memory](https://arxiv.org/abs/2604.27003)
  (**C**) finds that bounded-context retrieval recreates interference:
  procedural abstractions can transfer better than detailed trajectories, and
  designs with strong forward transfer can still forget.

For an Eliza agent, external memory is nevertheless the safest first route to
long-lived semantic and procedural experience. It can be inspected, bounded,
versioned, forgotten deliberately, and kept across model upgrades. Online
foundation-model weight updates should remain a separate, later research lane.
The low-level Alberta learner should export grounded experience summaries and
skill outcomes through the existing bridge boundary; the elizaOS runtime
should not import the research internals.

## Literature-to-gap decision map

| Repository gap | Best-supported sources | Action |
|---|---|---|
| UPGD provenance and semantics | UPGD ICLR 2024, official MIT code | **Adopt faithful baseline; preserve local variant** |
| Plasticity diagnostics | Nature loss-of-plasticity, C-CHAIN, spectral/isometry work | **Adopt diagnostics before choosing a method** |
| Long-stream capacity renewal | Continual Backprop, SNR, L2-init, clipping | **Benchmark as required baselines** |
| Learned state under partial observability | Forager, recurrent generate-and-test/trace literature | **Highest-priority capability work** |
| Continual world model | ICML 2025 online shallow model, Continual-Dreamer | **Adopt shallow reference and bounded replay** |
| Memory-efficient model replay | ARROW | **Reproduce dual-buffer idea, not results by assertion** |
| Actor forgets despite model memory | Dream Rehearsal preprint | **Component probes immediately; rehearsal experimentally** |
| Noisy surprise | Disagreement, Plan2Explore, aleatoric mapping, learning progress | **Separate uncertainty channels** |
| Delightful actor updates | DG and corner-escape papers | **Small isolated experiment** |
| Compute-budgeted learning | Kondo gate | **Experiment after DG; retain minimum update floor** |
| Search control | Plan2Explore, Delightful Exploration, prioritized sweeping | **Learn prospective priority; compare to random dreams** |
| External experiential memory | Era of Experience, memory-reuse study | **Use bounded procedural memory with retrieval evaluation** |
| Complete OaK lifecycle | Alberta Plan, reward-respecting subtasks, option keyboard | **Connect lifecycle to measured control/planning utility** |

## Alberta Plan citation and GitHub trail

Direct citation counts for the Plan are surprisingly sparse and index
dependent. At this snapshot:

- OpenAlex returned seven works in its `cites` relation.
- OpenCitations COCI returned two DOI-indexed citing works.
- A public GitHub repository search for the exact phrase in README files
  returned fifteen repositories, but most were notes, mirrors, or unrelated
  collections rather than implementations.

These are search snapshots, not exhaustive impact measures. arXiv papers,
theses, talks, and recent workshop work are unevenly indexed.

The clearest explicit trails are:

- [Multi-timescale reinforcement learning in the brain](https://doi.org/10.1101/2023.11.12.566754),
  relevant to a Horde of predictions at multiple timescales;
- [Scaling Goal-based Exploration via Pruning Proto-goals](https://doi.org/10.24963/ijcai.2023/384),
  relevant to discovery and curation of candidate goals;
- [Harms from Increasingly Agentic Algorithmic Systems](https://doi.org/10.1145/3593013.3594033),
  relevant to governance rather than implementation;
- the 2025 RLC OaK keynote and associated architecture discussions, which
  elaborate the vision but do not provide a canonical complete implementation;
- [`j-klawson/alberta-framework`](https://github.com/j-klawson/alberta-framework),
  the public framework lineage;
- [`lalalune/alberta`](https://github.com/lalalune/alberta), the direct source
  of this vendor;
- [`epicgamer17/modular-rl`](https://github.com/epicgamer17/modular-rl), a
  separate implementation/ablation effort with useful tests and candid
  negative results, but no declared license; and
- [`sharifnassab/SALT-Project`](https://github.com/sharifnassab/SALT-Project),
  associated with continual meta-learning research.

The sparse independent implementation trail is another reason to require
reference parity and external benchmark protocols rather than accepting class
names as confirmation.

The GitHub API snapshot also suggests a reference-quality rather than
production-quality UPGD ecosystem. The official UPGD repository had 28 stars
and seven forks, old pinned PyTorch dependencies, no releases, and no visible
CI/test suite; the broader Continual Backprop/loss-of-plasticity repository had
388 stars and 85 forks. These counts are neither scientific evidence nor
quality scores. They simply reinforce the recommendation to port the equations
into small local parity tests instead of adopting an aging training stack.

## Reuse and licensing notes

| Repository | License status at snapshot | Appropriate use |
|---|---|---|
| [`mohmdelsayed/upgd`](https://github.com/mohmdelsayed/upgd) | MIT | Equation and tiny-update reference; port with new JAX tests rather than depend on old PyTorch pins |
| [`shibhansh/loss-of-plasticity`](https://github.com/shibhansh/loss-of-plasticity) | MIT | Continual Backprop experiment/reference code |
| [`ajozefiak/SelfNormalizedResets`](https://github.com/ajozefiak/SelfNormalizedResets) | Apache-2.0 | Reset baseline reference |
| [`mohmdelsayed/weight-clipping`](https://github.com/mohmdelsayed/weight-clipping) | MIT | Simple baseline reference |
| [`Cerenaut/ARROW`](https://github.com/Cerenaut/ARROW) | MIT | Algorithmic reference; do not import its Dreamer stack into Alberta |
| [`steventango/continual-foragax-agents`](https://github.com/steventango/continual-foragax-agents) | MIT | Benchmark-agent reference |
| [`gurpnijjer/dream-rehearsal`](https://github.com/gurpnijjer/dream-rehearsal) | Apache-2.0 | Reproduction target for the new actor-rehearsal hypothesis |
| [`ramanans1/plan2explore`](https://github.com/ramanans1/plan2explore) | Apache-2.0 | Conceptual/reference implementation for disagreement planning |
| `C-CHAIN`, `FIRE`, and several newer research repos | No verified reusable license | Read and independently implement equations only, or obtain permission |
| [`google-deepmind/egg`](https://github.com/google-deepmind/egg) | Apache-2.0 | Small official DG/Kondo token reference; does not cover the full paper evidence |
| [`iosband/trl-dg`](https://github.com/iosband/trl-dg) | No declared software license | Inspect behavior; independently implement equations |
| [`lalalune/kondo-gate`](https://github.com/lalalune/kondo-gate) | MIT; community implementation | Useful API comparison, but its masked full backward does not demonstrate compute savings |
| Later delight paper family | No separate verified reusable implementation | Implement from equations behind an experimental API |

A repository being public does not make its code reusable. Every imported
artifact needs a license check, a pinned source revision, attribution, and
local behavioral tests.

## Operational definition of “complete” for this program

General continual intelligence is not a finite software ticket. The program
needs a falsifiable engineering target. A **complete continual experiential
prototype** should mean an agent that:

1. consumes one continuing stream of observations, actions, grounded rewards,
   discounts, and partner signals;
2. receives no task ID, boundary, replay epoch, or privileged reset signal;
3. has fixed declared memory and per-step compute budgets;
4. learns a recurrent state that improves prediction and control under partial
   observability;
5. maintains current-task plasticity and old-skill retention over repeated and
   novel changes;
6. learns calibrated predictions and a world model, and uses them in bounded
   search;
7. distinguishes reducible uncertainty, irreducible noise, utility, and
   advantage;
8. discovers, evaluates, composes, and retires bounded features, predictions,
   subtasks, options, and models;
9. exposes component-level health, provenance, and safety diagnostics;
10. checkpoints and resumes the entire learner, including optimizer, memory,
    RNG, lifecycle, and calibration state;
11. benefits from partner information through an auditable action path; and
12. beats matched non-continual and continual baselines on preregistered online
    metrics across synthetic, continual-control, partially observable, and
    embodied streams.

The companion [implementation plan](CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md)
turns this definition into staged work packages and exit gates.
