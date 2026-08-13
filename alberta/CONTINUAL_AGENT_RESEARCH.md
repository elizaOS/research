# Continual Experiential Agents: Research Synthesis and Alberta Framework Audit

- **Research snapshot:** 2026-07-30
- **Repository snapshot:** elizaOS research commit `c28144b`; Alberta Framework
  forked from `lalalune/alberta` commit
  `2ac35333efae45cf969ce02ec1f2703476fed6c2`
- **Working-tree status refresh:** 2026-08-03 — the dated status passages
  below reflect the current tree, not the committed snapshot
- **Scope:** continual learning, continual reinforcement learning, world
  models, state construction, plasticity, forgetting, surprise, curiosity,
  paper-defined delight/Kondo gating, candidate-update auditing, experience reuse, temporal
  abstraction, and intelligence
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
   by default. One mutually restricted opt-in pair-feature lifecycle now reaches
   linear OaK and an exact linear Horde. Its lifecycle learner uses balanced
   shadow-prediction utility; a separate opt-in auditor measures only one-step
   frozen-consumer deletion loss and a matched shadow-candidate insertion
   cohort. An additional stateless adapter may use those post-observation
   utilities only to rank active deletion within the active cohort and
   candidate insertion within the candidate cohort. It has no promotion or
   go/no-go authority and does not discover cumulants, subtasks, or options.
   An opt-in v18 Prototype coordinator now gives one pair lifecycle/router
   authority to linear OaK, an ordered linear Horde, a generated-input/fixed-
   physical-output world model, and exact feature-bound memory. Descriptor
   adoption is conjunctive across every consumer, while a veto retains each
   valid ordinary old-bank update; planning remains disabled by default. The
   older strict development-only A/B/A harness still uses its separate stable-
   base model lane and scripted partner.
   A separate standalone WP7.2 v1 mechanism now selects fixed-budget
   cumulant/subtask proposals from four source families under strict gates,
   but it has no consumer, option-lifecycle, curation, or promotion authority.
   The legacy IA recommendation remains
   diagnostic, while separate partner fusion can affect action selection.
   Bounded option-model planning now exists, but lacks a promoted
   matched-resource benefit result.
3. **The historical `UPGDLearner` is not canonical protecting UPGD.** It uses
   `|w g|`, takes an ordinary SGD update, then adds utility-scaled perturbation
   to trunk weights. Canonical UPGD uses signed Taylor utility and gates both
   gradient and noise. The working tree has since resolved this honestly:
   `upgd.py` keeps the historical semantics, now with corrected attribution
   and its deviations documented, and
   [`canonical_upgd.py`](alberta_framework/core/canonical_upgd.py) provides a
   separate source-profiled faithful implementation (details below).
4. **Plasticity and forgetting are different failures.** Resetting or
   perturbing unused capacity can keep an agent trainable without preserving
   old behavior. Replay or parameter protection can preserve old behavior
   without keeping the network trainable. Both must be measured and addressed.
5. **State construction is at least as important as plasticity repair.**
   Forager results show recurrent or trace-based state can matter more than
   several plasticity mitigations in a continuing, partially observable world.
   Alberta's current fixed-weight GRU does not learn what history to retain.
6. **Prediction error, epistemic surprise, learning progress, and delight are
   not interchangeable.** Raw error is vulnerable to stochastic “noisy TV.”
   Paper-defined delight is exact float32 advantage multiplied by action
   surprisal. In *Does This Gradient Spark Joy?*, the Kondo gate uses that
   signal to form detached forward admission intent; a sample sparks joy iff
   the actor consumer actually includes its exact contribution in a backward
   pass that executes.
   It is not a generic replacement for
   novelty, model uncertainty, parameter utility, or the separate
   multi-objective candidate-update safety audit.
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
- a separate consumed-root hidden-rule dyad now binds context identity to exact
  semantic births under three-slot/four-rule capacity pressure. A matched
  controller-row scrub removes stale Q/trace transfer across recycled births,
  while an executable prefix twin proves that no past-only deterministic
  policy can guarantee whether B or D should be forgotten when their observed
  histories are identical and only the future recurrence differs. Selective
  forgetting therefore requires an explicit prior, learned recurrence model,
  randomized/minimax objective, or bounded archive; it cannot be defined as
  hindsight-perfect eviction. A first defaults-off current-birth recurrence
  score changes two later evictions at `epsilon=.05`, avoids eight completed
  recurrence intervals, and yields `+.0024999976` reward; the other three
  epsilon arms are exact nulls and the prefix-limited first choice is unchanged,
  so the remaining cross-birth future-value gap stays explicit. A matched
  one-record victim cache then required a strict observed prediction win over
  both a fresh prior and every live source model before transferring lineage.
  It produced zero matches or behavioral effects at every epsilon; at `.05`,
  all four cache-valid tests failed the fresh comparison and contained an exact
  tie. One outcome is therefore insufficient on this dyad, and the next
  mechanism must accumulate bounded causal evidence rather than relax the gate.
  A standalone fixed-`H=2` successor now does so mechanically: it freezes the
  archive and all live pre-update comparators at a full-bank birth, waits for a
  second completed transition, and transfers exact lineage only when the same
  birth survives pairwise never-worse evidence with strict support against
  every comparator. Its 563-byte-per-agent state has separate configuration
  and complete-content SHA-256 tokens and no parameter-transplant path. This
  closes the bounded sidecar mechanism. A matched dyad wrapper now snapshots
  rescue before the outcome, routes exact zero or that past-only value through
  the prioritized context update, stages both sidecars/scrubs/controllers, and
  commits the whole 2,088-byte state atomically with matched calls and RNG.
  Ten focused wrapper tests pass. This still does not close the dyad result:
  host-transition provenance remains caller-owned and the full 4,000-step
  comparison has not run;
- the current generated-AST lane now closes task-blind finite product
  reachability but falsifies two simple retention repairs on one consumed
  8,998-step life. Topological headroom admits recurring A/B/C and improves
  greedy reward to `0.12247` while losing every target and retaining obsolete
  `p12`; left-packing removes `p12` but never co-retains two targets, ends with
  none, and falls to `0.03979`. Slot capacity and placement are therefore not a
  learned future-recurrence value. A frozen three-arm contribution-future-
  utility comparison then passed every structural contract but rejected both
  enabled endpoints: the disabled internal comparator retained A and scored
  `0.274283`, while mix-one/decay-zero and mix-one/decay-0.95 retained no A/B/C
  and scored `-0.003112` and `-0.020449`. No endpoint was selected or retuned.
  A five-arm v2 on a new development root froze current, future, half-mix,
  normalized, and longer-horizon mechanisms with exact
  active/candidate f32 ranks and honest intervention-specific work. Its public
  validator/serializer cannot consume the one-shot panel, private report/arm
  helpers require the latch's live capability, and a pure-stdlib external
  declaration bound the source/protocol/key/stream and summary-first one-shot
  chronology. Its sole attempt completed the first arm's scan, then failed
  before returning any arm endpoint because an all-step margin diagnostic was
  incorrectly asserted to be cadence-only. No report or result exists, the
  root cannot be retried, and the current evaluator refuses reentry. Eighteen
  evaluator, six decommissioned-declaration, and three outcome contracts pass.
  A failing-test-first synthetic trace repairs the diagnostic/endpoint split
  for a future root only; it does not revive or reinterpret v2;
  a frozen-theta standalone adapter supplies
  exact base/tail and full slot/birth identity plus recomputing deferred commit.
  An isolated transaction routes changed births through exact linear OaK and
  optional ordered Horde axes, with caller-derived safe-boundary curation
  suppression. A two-transition development lane now performs real old-bank
  OaK/Horde updates, consumes an unsafe due opportunity, then routes one birth
  at a safe primitive boundary with exact survivor/scrub/cache checks. It is not
  a matched outcome, and this standalone adapter is not itself a Prototype,
  model, or memory composition;
- an isolated model edge now lets the live generated tail condition a linear
  world model without turning mutable feature identities into output targets.
  Physical base-delta/reward/discount heads stay fixed; source learning precedes
  exact input-column routing, and a fully source-bound planning transaction
  re-augments predicted physical successors before one OaK base backup. Tests
  show a surviving generated column causally changes both the prediction and
  backup while newborn columns start at zero. This remains L0 reachability with
  defaults-off planning, no partner model, and no calibration, retention, or
  benefit result. A separate opt-in v18 Prototype coordinator now composes the
  same generated-input/fixed-output model with one pair lifecycle/router,
  linear OaK, ordered linear Horde, exact feature-bound memory, and exact
  checkpoints. It evaluates feature and world learning once, admits a
  destination only when every consumer is ready, and otherwise preserves all
  valid ordinary old-bank successors. That is L0 composition, not an outcome;
- the existing additive fast/slow learner now has exact two-word lifetime,
  fail-stop, atomic rollback, strict records, and resource contracts. Its
  consumed A/B/A diagnostic is negative for retention: the slow A-probe MSE
  rises from `0.0188931` after A1 to `4.01919` after B (`212.73x`), so its low
  A2 tail after further updates is reacquisition rather than permanent memory;
- an explicitly non-source-faithful Alberta-derived Permanent/Transient sibling
  uses independent tanh representations, fixed 788-byte state, no replay or
  task boundary, and a same-work no-consolidation ablation. Its permanent A
  probe rises from `0.0434621` after A1 to `3.92572` immediately after B, while
  the combined probe rises from `0.0345716` to `4.63162`; A2 improvement follows
  more A updates. Separating always-active representations therefore does not
  supply selective dormancy or protect contradictory recurring knowledge;
- a generic fixed-bank regression integration of the existing
  `ContextInference` active-only-freeze law now binds ownership before each
  outcome and computes every expert candidate with matched state/work. On the
  same consumed A/B/A source it reactivates the learned A expert after one A2
  outcome, but per-sample selection fragments identity: A switches 10 times in
  A1 and three times in B, including one B update that changes the learned-A
  subtree. The primitive preserves nonselected experts per transaction, but
  does not establish phase-long dormancy or pre-outcome context inference;
- a fixed two-event pairwise-dominance quarantine now repairs that specific
  overwrite failure without tuning a margin or dwell threshold. Its opening
  event nominates a unique no-worse dormant challenger but commits no expert
  update; a source-bound second event must add strict support for the same
  candidate. On the consumed 512/512/512 A/B/A source, learned A stays bit-
  exact through B with zero B updates and returns after two observed A2
  outcomes. Enabled A1/B/A2 prequential MSE is
  `0.0156662/0.0145220/0.0174704`, compared with
  `0.0156662/0.0994502/0.0754818` and 498 B-to-A updates when routing is
  disabled at the same logical work. This is a clean bounded dormant-expert
  retention result for one consumed synthetic life, but it still learns a
  switch only after outcomes, has exactly two supplied experts, and supplies
  no multi-seed, control, generated-feature, artifact, or promoted result;
- the smallest hidden-context full-agent composition now runs two fully
  instantiated `PrototypeAgent` objects through one uninterrupted 512/512/512
  A/B/A life. Oracle
  rule coordinates are destroyed, both actions precede outcome-based context
  updates, and one outer transaction couples the shared environment to both
  pair-feature/OaK/Horde/world-model/memory agents. Routing the existing two-
  slot inferred state gives A1/B/A2 mean-agent reward
  `0.991401/0.977935/0.964493`, versus
  `0.984638/0.0149503/0.994810` with identical inference work left unrouted;
  routed A2 recovers from `0.781546` early to `0.997949` in the tail. This is
  one consumed U0 causal-development life with 41,718-byte fixed state, not a
  held-out result. It immediately predates the final validation-only contract
  gates. A first exact replay lost its constructed comparison to a caller-side
  formatter error; the sole declared recovery reproduced every declared report
  field but failed closed because `prototype_agent.py` changed between clean
  preflight and comparison. The recovery budget is consumed, no report was
  retained, current-source compatibility is not established, and full-report
  identity is not claimed. Capacity equals the two-rule support, the initial
  slot-zero prior is part of the intervention, and no checkpoint, threshold,
  artifact, feature-generality, selective-forgetting, or promotion claim
  follows. A factorized U1 successor is also
  implemented but unexecuted: learned or uniform simultaneous partner beliefs
  marginalize a grounded four-cell joint-action model and may replace the
  post-memory primitive before one outer U0/planner commit. Its planner-only
  rows are `PP`, `M0P1`, `P0M1`, and `MM`, where `M` is post-memory fallback;
  memory query/write/eviction diagnostics remain available, but same-event
  memory reward effects are not attributed. It adds 3,758 persistent bytes,
  zero post-init planner RNG or replay, and retains four environment proposals
  per event. On one stable dependency snapshot, 17
  focused core/wrapper cases passed; the remaining case exposed an over-strong
  bit-exact float assertion. A 589-leaf comparison found only 17 float leaves
  with 1--2 ULP drift and exact discrete/key leaves, so the wrapper now uses
  U0's declared `rtol=1e-6, atol=1e-7; discrete exact` contract. The corrected
  parity case passed separately, but `prototype_agent.py` changed during that
  run. No current-source 18/18 claim exists, and a stable-source verification
  must pass before the full runner is considered;
- the first HCCL causal-core transaction rung is now implemented but remains
  L0 and nonpromoting. A fixed development world owns the physics, hidden-sign
  and observation state, named Threefry streams, immutable source-bound event
  receipts, typed task/net/safety/message signals, pure proposals, and atomic
  rollback. A separate adapter owns exactly one such world state and one
  adjacent-cube attribution-kernel state. It consumes an already-prepared event
  plus exact `B/M/P` action receipts, stages
  `MM/B0M1/M0B1/BB/PP/M0P1/P0M1/MM`, requires bit-exact duplicate `MM`, and can
  adopt only the `PP` world successor once. Source clocks, event identity, and
  all action-receipt identities fail closed on stale, tampered, cross-world, or
  cross-event use; rejection preserves the complete source for same-receipt
  retry. Composite orchestration and its prebound scan are explicitly
  host/eager-only after a full compiled scan approached the operational memory
  cliff; the smaller world-proposal and attribution donors retain their JIT
  boundaries. That first adapter has no learning stacks or fast/slow context
  and feature-lineage integration. A separate newer primitive-only factory and
  runner can construct a fixed 420- or 8,998-event integrated-dyad life, but
  neither schedule is an executed research result and partial resume/checkpoint
  is absent. Issued runbook, protocol seeds, result, threshold, artifact/writer,
  evidence, and promotion authority remain absent. Communication is still a
  later HCCL-v1 extension;
- a separate v1 `HCCLExternalCoordinatorBaseBridge` now supplies one bounded
  base-only agent-substrate rung. It owns one HCCL world/attribution state and
  two independently initialized external learned-state/router/audit
  coordinator states, starts them from their own exact raw 16-channel world
  observations, and binds each exact cached primitive as `B=M=P`. Six
  deterministic action-receipt identities include the corresponding full
  decision and lifecycle identities; one common hard-mask matrix governs all
  three layers, and excluding a cached action rejects before any fallback can
  be invented. The PP proposal supplies each coordinator exactly its own
  executed action, net reward, and next observation. HCCL plus both
  coordinators adopt all-or-none, with bit-exact rollback and same-event retry.
  Exact zero memory/planner contrasts describe only this ablation; they are not
  “no joy” conclusions and no actor backward pass occurs. Host-only staging,
  in-memory checkpointing, and fixed resources are L0 contracts, not a life,
  run, seed, artifact, threshold, evidence, benefit, or promotion claim;
- a separate v1 `HCCLTwoLiveMemoryBridge` now replaces that base-only ablation
  with exactly two live learned-memory adapters while retaining one HCCL
  owner. Pending receipts bind each agent's `B` and `M`; absent receipts use
  `B=M`, and `P=M` remains an explicit no-planner rung. Agent 0 settles only
  `M0B1-BB` and agent 1 only `B0M1-BB`; memory interaction is audit-only.
  Each child receives its own executed `M` action and `PP` outcome, next masks
  install only on the all-owner commit, and every rejection is atomic. This is
  host/eager L0 memory-utility integration, not delight, actor backward,
  evidence, benefit, or promotion;
- the additive v1 `HCCLTwoLiveMemoryPrepareAdoptBridge` keeps that exact
  persistent owner tree but exposes one transient downstream transaction.
  Preparation evaluates each donor once and binds both agents' candidate live
  states, raw/final STOMP owners, complete finalization traces, and independent
  extended masks. Adoption performs integrity checks and child adoption only,
  with no donor reevaluation; veto, replay, tamper, or a foreign binding returns
  the complete source and outer-gates child-applied facts. Partner fusion is
  admitted only over an immutable feature bank. This is still unauthenticated
  host/eager `P=M` L0 plumbing with no Kondo actor backward, so it establishes
  neither delight nor that any gradient sparks joy;
- an additive `HCCLTwoLiveMemoryFactorizedPlannerBridge` now supplies a
  snapshot-free `P` rung. Its complete persistent owner tree is one HCCL
  state, the same two live post-memory `M` states, and one paired factorized
  planner state/cache. It completes the paired behavior/joint-world models
  once and reconstructs transient `P` states through the public Prototype
  cached-action replacement. The planner is grounded on the external GRU
  builder's 17-wide constructed state while the 16-channel raw observation
  remains separately bound to the HCCL plant and `PP` commit. Hard-mask
  fallback is `M`, the effective `P` pair alone is consumed, and adoption fully
  validates both current and candidate planner states/caches with no donor
  reevaluation. Seven focused current-source cases pass individually. This is
  still host/eager unauthenticated L0 plumbing without external or physical
  dispatch, safety, Kondo actor backward, delight, matched benefit, evidence,
  or promotion;
- the additive no-planner
  `HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge` retains those two sole
  live STOMP owners and adds only detached repeated-option metadata. It
  consumes each raw STOMP result once, projects a cached memory-action change
  through the full coordinator owner, and adopts every owner together or not
  at all. At a real started-but-quiescent boundary, one selected agent may
  consume the exact fresh-cold atomic swap while the HCCL world, other agent,
  learned memory, pending feedback, and primitive masks remain exact and no
  cold slot persists. This remains `P=M`, host/eager, unkeyed L0 integrity
  plumbing with no planner, physical dispatch, authentication, safety, Kondo
  backward, delight, benefit, evidence, or promotion;
- `HCCLContinualDyadTransaction` is the first atomic integrated-owner rung.
  It owns one HCCL world/attribution state, two live post-memory action stacks,
  two slow-context states, and one paired factorized planner. A strict split
  stops after both memory phases and completes with either the planner or the
  constructible disabled-planner form without donor reevaluation; all owners
  adopt together or return the complete source. Its ordinary `step` now owns
  event, B/M/P binding, canonical memory provenance, preparation, receipt, and
  adoption from only a source state and hard masks. The production fixed-life
  executor delegates to that method, while the separate factory/runner requires
  a fresh complete 420- or 8,998-event primitive-only schedule and returns only
  an in-memory trace. Neither schedule is an executed research result; partial
  resume/checkpoint, physical dispatch, matched result, evidence, promotion,
  and causal-core completion remain absent;
- `HCCLKondoContinualDyadRoute` v3 uses that disabled-planner form for recurring
  actor-owned `P`. Genesis installs the first proposal/compact certificate
  without actor backward; every successor consumes the prior compact lineage
  through one Kondo actor transaction before sampling and atomically installing
  the next pair. The planner remains a learning-only shadow, both action stacks
  remain the sole Prototype owners, and the route derives both learned-memory
  inputs from its own causal-core event and canonical agent rows rather than
  accepting caller provenance. Each successor also derives the exact pending-
  `P`/current-`PP` protected batch and updates separate two-row linear reward-
  value and cost-value heads with detached pre-update bootstraps before Kondo;
  cost is `safety_cost + message_charge`, currently exact zero. Outer rollback
  cannot erase an actor backward that already executed, but only the nested
  actor result can say its gradient contribution sparked joy. Scheduling,
  actor keys, and all-true masks remain caller-driven host/eager L0 machinery.
  The protected-only checkpoint is not route recovery, and there is no
  autonomous life, route checkpoint/resource closure, authentication,
  dispatch, physical safety or critic-efficacy result, evaluator, benefit,
  evidence, or promotion;
- STOMP now keeps environment return separate from pseudo-reward, uses a
  coherent discounted differential semi-MDP target, and consumes option models
  in bounded planning backups;
- OaK transition ownership, option-start accounting, active-option curation,
  Prototype primitive-action credit, and dream isolation have focused
  regressions;
- an isolated nonlinear discrete actor/critic now binds the actually executed
  action, target and caller-declared behavior likelihoods/revisions, and exact
  identity before clipped per-decision action-importance correction. Separate
  actor/critic head and trunk traces, component plastic/frozen policies, typed
  RNG, exact clocks, checkpoint construction, rollback, and persistent-byte
  accounting are L0-tested; it remains discounted scalar-V machinery without
  state-visitation correction, average reward, learned component utility, or
  matched outcome evidence;
- a separate nonlinear discrete differential actor/critic now owns independent
  `tanh` actor and critic networks, eligibility traces, momentum, fixed
  head/trunk plasticity policies, bounded utility telemetry, and a learned
  reward-rate baseline. Exact cached target and caller-owned behavior
  distributions, revisions, action identity, owner digest, pure proposal, and
  recomputing atomic commit contracts cover both ordinary epsilon-mixture
  behavior scores and clipped per-decision action importance. This closes an
  average-reward mechanism gap at L0 only: action ratios do not correct state
  visitation, utility does not control plasticity, and no matched control,
  convergence, retention, or safety result exists;
- one restricted Prototype feature bank now updates linear OaK and an ordered
  linear Horde under old-bank semantics before atomically routing both
  post-update consumers, with scale/group-balanced proxy utility and a
  schema-bound v4 checkpoint;
- a separate fixed-universe WP7.2 v1 lane now emits only complete,
  fixed-quota cumulant/subtask proposal cohorts from controllable events,
  feature changes, reward-transition atoms, and typed prediction bottlenecks,
  alongside frozen random and exact hand-authored comparators under the same
  budget;
- a pinned Foragax runner and protocol importer are present, and a
  stage-conformant five-seed field-of-view tuning stage selected `step3e3`;
  the 30-seed evaluation lane remains incomplete: the Alberta worker has no
  completed batch or report and is no longer active, while its matched
  official-DQN and relearning companions are quarantined and there is no
  completed comparison report;
- a separate four-seed RTU-RTRL GPU development run completed 500,000 steps
  with FOV tail-EMA AUC 1.550 mean and 0.324 sample SD, but its exact receipt
  is explicitly nonpromoting because selection was not preregistered and
  source closure is incomplete. A reconciled unsealed DQN receipt gives a
  descriptive +0.331 matched-seed mean difference, but it was configured after
  RTU output and has unmatched runtime, representation, resources, and update
  work, so no admissible paired baseline exists; and
- a matched-current Forager campaign contract now exists through its
  qualification stage only: `outputs/forager/` holds a completed executor
  qualification and a prepared open-tuning campaign with published manifests
  but zero executed tuning cells, the sealed held-out evaluation stage
  (`alberta_framework/benchmarks/forager_matched_sealed_evaluation_campaign.py`)
  has no console script and has never run, and every
  authority-bearing path terminates at an
  external trust resolver that does not exist in-tree, so the shipped parity
  receipt remains unverified with `promotion_authorized: false`.

These are narrow advances, not an integrated completion result. The broader
legacy/default pairwise discovery path still fails a 10× scale-shock gate. An
opt-in scale-robust v2 package passed its immutable frozen narrow comparison,
but registered source drift now makes that artifact invalid for the current
learner. It also retains visible context, an exhaustive finite pair archive,
and one fixed learner initialization; it does not close general feature
discovery or control. State construction in the most integrated v18
composition is still fixed rather than learned, the
representation/model/planning lifecycle is not closed, and no single bounded
agent life exercises and ablates all required links.

As of 2026-08-01 the fail-closed evidence registry
(`alberta-evidence-status`) reports overall `invalid` (exit 2) with all five
registered claims invalid: registered source files were edited after the
artifacts were pinned, so no persisted claim is currently supported against
the working tree. That is designed behavior, not a defect to silence; the
frozen outcomes recorded inside the pinned artifacts are unchanged, and
renewal requires frozen-protocol reruns to new artifact paths and schema
versions. [RESEARCH_STATUS.md](RESEARCH_STATUS.md) is the live evidence
matrix.

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

[VENDORING.md](VENDORING.md) records that the upstream root-level
`benchmarks/` scripts tree, `docs/`, `examples/`, and `scripts/` were not
carried over (the local `outputs/` tree is this fork's own pinned evidence,
not the upstream one). Consequently, [tests/conftest.py](tests/conftest.py)
collect-ignores 39 test modules in this checkout because their import-time
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

At the committed snapshot, the complete test collection available during the
review finished successfully on Python 3.12.3: **1,519 passed and 55 skipped
in 1,302.94 seconds**. Ruff also reported clean. This is strong
implementation-health evidence. It is not the missing scientific evidence:
all 51 canonical Step 1/2 evidence tests skipped because their required
output artifacts were absent, and the collection hook excluded the 39
benchmark/example-dependent modules described above. The working tree has
since grown far past this figure (6,958 tests collected on 2026-08-01); the
snapshot numbers above are historical, and the Step 1/2 canonical-artifact
skips still apply in this checkout.

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

The current working tree also adds a public, non-learning
`EmbodiedSafetyEnvelope` L0 kernel. It validates current and proposed joint,
workspace, collision, timing, bridge, identity, and version constraints and can
return only the proposal, a statically configured fallback, or no action.
Emergency stop latches in an independent checksummed transition even when the
ordinary command transaction is rejected; reset requires a strictly newer
stationary-safe sample, a monotonic nonce, and an authority-bound token after
external caller authentication. Rollback preserves diagnostics while suspending
deployment, and checkpoint restore requires an exact revision plus SHA-256
anchor retained outside the payload. A pure fixed-ring shadow readout combines
zero hard violations with Wilson, calibration, and latency gates but has zero
dispatch, deployment, or promotion authority. These are mechanism contracts,
not a geometry proof, physical-safety result, robot-simulation result, or
deployment authorization.

A strict nonpromoting synthetic robot fault audit now exercises that envelope
over one frozen 30-event continuing schedule covering telemetry/wear drift,
timing and delayed-reward metadata faults, sensor corruption/failure, bridge
loss/recovery, unsafe candidates, emergency stop, reset, rollback, and exact
checkpoint recovery. Only envelope-available commands are marked executed in
simulation accounting, and physical dispatch remains zero. The complete trace,
shadow facts (with success explicitly proxied by action availability), hard
interventions, fallback identity, action-availability
recovery delays, eager/JIT/scan parity, externally anchored checkpoint resume,
and exact causal replay are retained.

The schedule is synthetic, not a dynamics simulator or geometry proof. Its
learner/controller is only an unchanged opaque witness, so no learner reset or
adaptation latency is tested. The pure kernel does not authenticate callers;
external authentication is still required before using its non-secret
authority tokens. The held-out family is declared but unexecuted, and there is
no matched no-candidate comparator, evidence seed, threshold, artifact,
physical-safety result, or deployment authority. The audit is therefore L0
fault-path inspection and remains `not_assessed`, not a robot-simulation or
WP9 exit result.

A separate minimal dynamics/adaptation evaluator now closes only that
mechanism-wiring omission. It runs adaptive and zero-learning `PrototypeAgent`
arms through owned bounded two-joint plants over A/B/A and one separately
declared consumed change family. Typed common randomness covers only exogenous
dynamics, sensor, fault, and latency inputs; policy randomness and trajectories
remain independent. Every primitive becomes a robot-like command and crosses
the hard envelope. Certified fallback changes are rebound to Prototype's real
cached credit owner, and blocked actions advance neither plant nor learner.
The full command/version/transition/revision trace, fixed resources, pure-
dynamics parity, composite checkpoint resume, and exact replay are retained.
This is still `not_assessed`: it has no untouched held-out family, threshold,
artifact, physical dispatch, safety certificate, adaptation-efficacy result,
or WP9 exit authority.

These strengths should be preserved. The roadmap should connect and validate
them, not replace them wholesale.

### Step-by-step status

| Step | Kernel status | Integration status | Evidence in this tree | Assessment |
|---|---|---|---|---|
| 1 | Strong linear/adaptive optimizer coverage | Used by later learners | Unit and smoke coverage; full replication artifacts omitted | **Implemented; re-establish benchmark evidence** |
| 2 | Extensive nonlinear and feature-lifecycle machinery | Several alternative paths, no single accepted lifecycle | Critical benchmark and theory gates omitted; upstream evidence mixed | **Active research, not complete** |
| 3 | Horde, mixed and independent demons, linear off-policy TD | Optional Horde in full agent | Nonlinear/off-policy/trace combinations remain open | **Substantial partial implementation** |
| 4 | SARSA, discrete actor-critic, exact-density bounded continuous average-reward actor-critic, isolated nonlinear shared-trunk action-importance correction, and a separate nonlinear discrete differential actor/critic | Step 1–4 pipeline exists; continuous and nonlinear cores remain isolated | Actor/critic is provisional versus SARSA; one off-policy lane is discounted scalar-V/action-ratio-only, the differential lane still has action-only correction, and all isolated lanes have L0 contracts without matched outcome evidence | **Implemented surfaces; weak closure evidence** |
| 5 | Average-reward prediction machinery | Available to later components | Small gates | **Kernel implemented** |
| 6 | Differential continuing control | Used by STOMP/OaK | Small/surrogate environments | **Kernel implemented; benchmark breadth missing** |
| 7 | Legacy one-step planning, real-anchor ensemble Dyna, and guarded short rollouts | Optional guarded legacy dreams and fixed-budget option-model backups can update control; a strict local planner→gauge→actor/critic composition closes rollout-tensor substitution, but ensemble lanes remain outside Prototype/dispatch and external grounding facts remain caller attestations | Small gates; no calibrated or matched-budget ensemble planning gain | **Mechanisms and a bounded local consumer exist; grounding and planning evidence remain open** |
| 8 | Trainable state builders, separate balanced comprehensive auxiliary objectives, and bounded feed-forward/recurrent world-model lanes | Opt-in Prototype paths return committed real model gradients to learned state; online-gated, dense full-GRU, and compressed-RTU builders expose RTRL proposal/commit contracts. Transactional adapters route the smaller GVF/inverse and caller-targeted comprehensive objective sets; a separate learner-owned target producer/Prototype owner derives the same comprehensive families from the accepted transition. Strict-linear consumers for both comprehensive target authorities perform causal RTU whole-unit replacement before next-action selection. The v18 atomic lane gives one pair lifecycle/router to linear OaK, ordered linear Horde, a generated-input/fixed-physical-output model, and exact feature-bound memory, with conjunctive destination adoption and ordinary-update retention on veto | Narrow historical FTL result plus nonpromoting L0 diagnostics | **GRU/RTU comprehensive mechanics and one atomic generated-feature/model/memory composition exist; calibration, selective-retention/control outcomes, and Forager gates remain open** |
| 9 | Dream guards, typed priorities, replay rehearsal, candidate scoring, prospective fixed-budget exploration, and calibrated four-mode search control | A raw-representation Prototype/STOMP sidecar snapshots live learned primitive/option models under one shared budget; a separate default-off v2 can consume candidate-specific evidence to replace the cached primitive and arm its actual owner under a caller mask; a consumed synthetic lane derives all six exploration arms' scores from independent executed histories | L0 contracts and development integration only; no safety/physical-dispatch authority or matched held-out exploration/search-control outcome | **Causal stochastic-trap mechanics and an owner-correct live policy edge exist; policy benefit and evidence remain open** |
| 10 | Cumulant/subtask discovery, semantic-bound option lifecycle audit, options, outcome models, and extended actions | A bounded scheduler observes discovery continuously, requests fresh four-family proposals at an exact cadence/retry, and receipt-gates their installation into preallocated live STOMP slots at quiescent boundaries; cumulants are rematerialized and cold slots masked throughout behavior/learning/planning. One controller can execute an exact caller-authorized two-rebind retirement; a separate one-canonical-state controller can later install and reactivate one externally authorized replacement in the resulting cold slot. An opt-in one-owner bridge composes detached authority metadata with the sole live Prototype→OaK→STOMP owner, reuses the exact raw trace, propagates cold masks through search/Dyna/audit, and metadata-finalizes the sole post-control owner | L0 proposal/scheduler/installation/audit/authorized-retirement/replacement/bridge contracts; go/no-go remains caller-owned and there is no promoted matched-budget planning or repeated-lifecycle benefit | **A bounded live-owner composition exists; autonomous authority, repeated lifecycle policy, and outcome evidence remain open** |
| 11 | Utility EMA, bounded curation, keyboard proposal/owner-correct dispatch, lifecycle diagnostics, and calibrated option-model search | Explicit consumers can change cached policy; discovery can reach live STOMP, exact authorized retirement can leave cold vacancies, and a default-off calibrated consumer can form a proposal-only chord without starting an option, but automatic chord discovery/replacement remains absent | L0 mechanism gates; no physical dispatch or matched automated lifecycle benefit | **Live narrow edges exist; autonomous curation evidence is open** |
| 12 | Exo-cerebellum/exo-cortex APIs plus bounded partner-policy fusion | Opt-in Prototype can change the real primitive under hard safety and exact feedback ownership; a matched-initialization three-arm v2 evaluator now isolates the wrapper intervention | Historical IA valid rejection; the consumed 12-event matched lane gives learned versus fixed-zero internal separation but no behavioral separation | **Action mechanism and matched instrumentation exist; efficacy gate open** |

The continuous Step 4 instrumentation is a strict fixed 12-event,
one-dimensional A/B/A evaluator starting from an immutable source-bound
snapshot. Phase/case identities, preferred action centers, rewards, and
reference values remain evaluator-owned. Its report reconstructs cached
pre-tanh latent and transformed-action ownership, transformed densities, the
exact latent action ratio, same-state gauge-centered critic error, actor
error/churn, plasticity/activity, successor ownership, counters, resources,
final state, and exact live replay. Only transformed diagnostic densities have
an explicit symmetric eight-float32-ULP backend allowance; the policy-defining
ratio remains bit-exact. No-overwrite reports/checkpoints have absolute
byte/scalar limits. This is `not-assessed` development instrumentation, not a
continuous retention or control result.

### Ranked integration gaps

#### 1. The full-agent label is misleading

At the audited snapshot,
[`PrototypeAgent`](alberta_framework/core/prototype_agent.py) described itself
as integrating all twelve steps; its docstring now scopes it as an
experimental integration of mechanisms mapped to the Plan. Either way, its
default configuration instantiates only a small OaK/STOMP path. The world
model, dreaming, Horde, IA, and GRU are all optional and disabled by default.
The separate
[`AlbertaPipeline`](alberta_framework/pipeline.py) explicitly covers Steps 1–4
only. Neither object is the complete feedback system described by the Plan.

#### 2. State construction has a first causal world/control-gradient lane

The legacy `GRUPerceptionConfig` remains an honestly named fixed-weight
echo-state baseline. `PrototypeAgent` now also accepts the common
`StateBuilder` contract with identity, fixed-trace, online-gated recurrent,
conventional dense full-GRU, and diagonal complex compressed-RTU
implementations. The full GRU has learned
input and recurrent update/reset/candidate maps plus exact fixed-parameter
RTRL sensitivities under the same proposal/commit, resource, reset, and
checkpoint contracts. Its sensitivity storage is `O(H * P)`, and sensitivity
carry after parameter changes is explicitly approximate. `start()` advances
recurrence once and caches the exact raw
observation, representation, primitive action, and lifecycle/generation token;
`act()` only returns that cached decision and cannot advance recurrence.
Explicit transitions advance a builder once per bootstrap observation, reset
episode-local recurrence at a boundary, and consume the post-reset decision
observation once. Eager, JIT, scan, and checkpoint tests cover those counts.

The RTU builder persists the recurrence's unit-diagonal sensitivity structure
instead of a dense hidden-by-parameter Jacobian. It is exact for fixed
parameters and shares the source-bound advanced-destination commit contract.
Default carry after parameter updates is approximate; the optional diagonal
Taylor path owns its exact source vector, accumulated actual delta, and source
update words but still omits mixed-parameter Hessian terms.

A standalone `RTUGenerateAndTest` L0 lifecycle now closes the mechanism-only
recycling gap for that builder. Before the downstream gradient changes the
RTU, it observes per-unit real/imaginary activation-gradient products and
updates fixed-capacity diagnostic utility, age, and support. In the strict live
composition, the owning adapter also jointly deletes each unit's real and
imaginary current-representation channels against frozen pre-update
comprehensive heads and records a separate positive bounded causal-utility EMA
and evidence count. Exact periodic replacement requires this causal rank—there
is no contribution-rank fallback—alongside the existing maturity, protection,
cadence, quota, and active-option gates. Missing or immature causal evidence
defers recycling without dropping ordinary learning; an attempted invalid or
non-finite internal counterfactual rejects the whole outer transaction. The stable
lowest-causal-utility mature units outside a protected set are replaced
under a fixed quota. Each replacement redraws the unit's polar recurrence and
two input rows and scrubs its activation, compressed RTRL sensitivity, and
optional Taylor trace/source/delta slices. If ordinary RTU learning is also
proposed, the lifecycle recomputes that exact source-bound commit and accepts
only its bit-identical live destination before recycling; stale or invented
destinations fail atomically. The public finalization receipt reconstructs the
advance receipt, independently reruns RTU commit, and exact-matches its
destination and selected mask. That proves derivation from supplied values but
does not authenticate caller-owned lifecycle, objective-gradient, or ordinary-
proposal authority. A strict comprehensive-objective composition now
prepares recurrence without learning or action RNG, applies the current
transition's combined objective update to the old representation, performs at
most one atomic whole-unit replacement, scrubs every recycled comprehensive-
head, linear STOMP/OaK base, intra-option, trace, and option-model axis, and
then selects the
ordinary next action from the recycled representation. An executing option
defers replacement while still committing recurrence, objective learning, and
the real transition. Content-bound prepare/finalize receipts and separate
replacement-unit/replacement-event clocks bind the builder revision to accepted
transitions plus nonempty replacement events. The composition owns the
lifecycle source and internally constructs the gradient and source-bound
proposal, closing the three lower-level authority boundaries. Its declared
source work is four builder-commit and two RTU-commit evaluations plus one
frozen-head counterfactual per unit for preflight and independent derivation;
persistent state receives one logical ordinary
update and at most one replacement event. The RTU-enabled composition declares
the stricter per-unit uint32 lifetime bound of `2**32 - 1` accepted transitions.
The envelope deliberately rejects
nonlinear STOMP, planning, world/model/replay/dreaming, Horde, IA,
partner/memory, GRU, historical candidate-update audit, and feature-lifecycle
sidecars. This prequential frozen-head causal-deletion mechanism is not an
independently held-out efficacy result, and neither it nor the contribution
diagnostic is the paper-defined delight signal. It has no matched recurrent
comparison, control outcome, or evidence claim.

The online-gated builder now exposes a source-bound proposal and destination
commit boundary. An opt-in Prototype lane combines the bounded ensemble's real
predict-before-update representation gradient with one current control-loss
semi-gradient. Idle primitive transitions use the base-Q objective; executing
options use the current intra-option objective. Delayed semi-MDP option-start
credit and replay gradients are excluded from the current representation. The
stateless mixer records source norms, weights, clipping, cosine/conflict, and
failure provenance, then proposes the exact mixed candidate against the builder
state that emitted the modeled representation. Only the parameter delta is
committed into the already-advanced builder state, preserving hidden state and
sensitivity. A second opt-in boundary accepts decision-bound, independently
attested objective, retention, and safety probes and stores the delta only when
the complete candidate-update audit passes and its finite-precision
application succeeds. Missing, stale, incomplete, or
non-finite sidecars veto only builder learning, not the real control/model
transition.

The stateless `ExternalBuilderCandidateEvidenceProducer` now makes the
external full-GRU variant of that probe boundary executable without accepting
caller-shaped parameter gradients. It binds three caller-owned
representation-space probes to exact coordinator event/builder/Prototype/
feature-generation/decision identity and pulls their hidden components through
the cached source RTRL sensitivity. Identity drift or non-finite values yield
unavailable exact-zero evidence, while independence remains an explicit
caller attestation that arrays cannot prove. The real coordinator accepts a
valid produced bundle under a permissive mechanism configuration and changes
only its external builder. These three analytic pullbacks add no model forward
and no actor backward; they are candidate-audit plumbing, not a “sparks joy”
fact, calibrated evidence, or an outcome result.

This is causal mechanism integration, not a learned-state result. It is only a
two-source grounded-model/current-control path, not an empirically balanced
objective set. A separate `BalancedStateObjectives` L0 kernel now supplies
linear GVF heads at multiple strictly ordered discounts plus a
consecutive-pair inverse-action head. The head families update separately, the
GVF family is averaged before fixed positive group masses are applied, and
clipped current/successor representation gradients remain bound to the exact
executed-action receipt and caller-owned revisions. Strict checkpoints,
resources, fail-stop clocks, retry-safe rejection, JIT/scan, and a real
`OnlineGatedStateBuilder` proposal/destination commit are tested. A separate
opt-in `PrototypeBalancedStateObjectives` transaction now authenticates the
exact dispatched decision/action and bitwise representation, scores the
final/bootstrap observation before any autoreset decision, pulls back current
and bootstrap gradients through their respective recurrent sensitivities, and
commits one clipped builder update. Its cached next representation retains the
pre-auxiliary decision-time builder owner while the live builder is exactly one
accepted auxiliary update newer; failures roll back every component, RNG,
clock, and cache. The ordinary Prototype path remains unchanged. Weights are
still declared rather than empirically calibrated. A separate
`ComprehensiveStateObjectives` L0 kernel now supplies action-conditional
next-observation/latent, reward, Bernoulli termination, multi-timescale GVF,
state-value, selected-action-advantage, and inverse-action heads. Prediction
and control subheads are averaged inside fixed family masses; all heads retain
independent parameters, step sizes, and exact revision rows. Stable BCE,
finite-difference representation gradients, bit-exact action ownership,
non-earlier successor revisions, numerical rollback, strict resources and
checkpoints, and eager/JIT/scan parity are tested. A separate opt-in
`PrototypeComprehensiveStateObjectives` transaction binds caller target bits,
source revision, and provenance to the exact Prototype decision/action,
observation event, final/bootstrap observation, and builder owner. It supports
the online-gated, full-GRU, and compressed-RTU builders, sums current and successor RTRL
pullbacks, commits one clipped logical update, and atomically restores every
component on rejection. The strict-linear RTU composition above consumes that
update through the independently rederived recurrent destination and scrubs
recycled consumers
before selection. A separate `CausalStateObjectiveTargetProducer` derives the
ordinary target families from one accepted transition, and its versioned
`PrototypeCausalStateObjectiveTargets` owner now supports the exact RTU builder
only with the matching strict lifecycle. Its deletion scorer holds both the
pre-update objective parameters and the one factual learner-owned target bundle
fixed. Whole-unit replacement atomically scrubs the objective/pending cache,
RTU sensitivity/Taylor ownership, and supported linear STOMP consumers to
canonical `+0.0`; invalid scoring, consumer corruption, or a late successor
cache refusal restores the complete source including lifecycle RNG. Typed
metadata/sentinel checkpoints resume exactly, and the RTU lane fail-stops after
`2**32 - 1` accepted transitions. Calibrated target semantics/masses, general
lifecycle compatibility,
independently held-out feature-utility efficacy, broader lifecycle curation,
matched ablations, and the Forager gate are absent. The builder therefore must
not be called a learned-state
success. The consumed, resource-unmatched write/hold probe gave four-seed mean
frozen-suffix accuracy `0.5158` observation-only, `0.5292` fixed trace,
`0.5258` online-gated, `0.5067` full GRU, and `0.5617` RTU. The RTU used
`1,324` total persistent bytes versus `12,204` for the full GRU. This is a
descriptive supervised-development signal, not a matched control result or
causal efficacy inference.

#### 3. Prototype has a narrow WP7.1b auditor and WP7.1c rank adapter, not the required loop

An opt-in Prototype lane augments a fixed-width builder base with a bounded
pair-product bank. Its original mode trains from one owner-bound behavior TD
target: base-Q while idle or the executing option's intra-option target. Its
shared mode places that control target in channel zero and follows it with
`HordeUpdateResult.td_targets` in the declared demon order. Linear OaK and the
linear Horde update under the old descriptor bank first. The feature learner
then observes the ordered target vector and, at a safe idle/cached-base
boundary, routes every post-update consumer feature axis atomically in exactly
two router calls. Unsafe curation is rolled back and deferred rather than
queued.

Scale normalization applies within the proxy, while explicit task weights give
control `0.5` and each of `D` Horde demons `0.5/D`. This prevents target scale
or demon count alone from owning the bank, but it remains shadow prediction
utility rather than a causal downstream deletion test. The augmented gradient
is pulled back against the exact pre-route generation and full descriptor bank.
One enabled-only bundle binds linear OaK, Horde, the consumer binding, and an
ordered-schema digest; the shared composition therefore requires a v4
Prototype checkpoint. An exhausted lifecycle is an audited no-op that leaves
the already-advanced, step-aligned OaK/Horde consumers untouched. Exact static
checks reject nonlinear OaK or Horde, a non-LMS Horde optimizer, and any Horde
normalizer.

An additional opt-in auditor forms scores from the old descriptor bank's
frozen, predict-before-update consumer targets, predictions, and linear tail
weights. Active utility is the exact normalized one-step half-squared-loss
increase under deletion. Candidate insertion is evaluated in a separate
matched shadow-candidate cohort with private normalized-LMS contributions; it is scored
before shadow weights, utility EMAs, or scale moments update and is not compared
against active utility to route features. Task mass remains fixed at `0.5` for
control and `0.5/D` for each ordered demon, without redistributing unavailable
mass.

If the lifecycle commits its two-call consumer route, the auditor explicitly
rebinds private state by descriptor identity without another router call. New
or colliding candidate identities are zeroed. Enabling this instrumentation
nests audit state and its digest around the existing consumer bundle in a
single atomic state bundle and requires the v5 checkpoint schema; disabling it
leaves the v4 checkpoint and behavior unchanged.

WP7.1c adds a stateless ranking adapter to this exact lane. It reads the
post-observation audit EMAs as feature deletion/insertion sensitivity, not
paper-defined actor-sample delight: it neither scores actor samples nor
selects backward passes. Lower deletion utility ranks only among active slots, while higher
insertion utility ranks only among candidates; an active score is never
compared with a candidate score. Every configured task must independently
meet the evidence floor for a slot before that slot is rank-eligible. The
fixed control/Horde task mass is neither redistributed nor renormalized.

This adapter influences rank, not the curation decision. Existing active and
candidate ages, maintenance cadence, candidate-confirmation rules, internal
proxy promotion floor and margin, and the safe routing boundary retain all
promotion and go/no-go authority. The adapter owns no persistent state and
adds no RNG draw, backward pass, consumer update, or router call. Its exact v6
Prototype checkpoint shell binds the ranking configuration and digest around
the v5 bundle; disabling the adapter leaves v5 behavior unchanged.

This utility-auditor/curation configuration still excludes world models,
replay, dreaming, IA, partner fusion, experiential memory, and GRU perception.
Separate exact-Identity Prototype lanes admit feature-bound experiential memory
and a stable-base action-conditioned model, and a development-only A/B/A
harness executes those lanes together with OaK and managed Horde. The isolated
routed linear model instead consumes the generated tail and re-augments its
fixed physical successor for one OaK base backup.

An opt-in v18 coordinator now composes one exact-Identity pair lifecycle/router
with linear OaK, ordered linear Horde, that generated-input/fixed-physical-
output model and anchor buffer, and exact feature-bound experiential memory.
Feature and world learning are prepared once; lifecycle, world, memory, and
all consumer readiness are conjunctive for destination adoption, while veto
retains all valid ordinary old-bank updates. At the lifetime observation cap,
the lifecycle remains an exact rejected no-op and only the authenticated
current augmented encoding is locally derived for cache consistency. Mirrored
bindings are caches, one state owns the resources, checkpoints bind the v18
configuration, and planning defaults off. The v18 lane does not include the
utility auditor or feed model/planning outcomes into feature ranking. Thus it
closes the missing atomic ownership edge but still does not close Step 8's
required learning cycle:

`features → world-model quality → planning/control utility → feature ranking → features`.

The pair lifecycle (including its ordered-Horde form) and that routed world
model now share an external-readiness transaction shape. Preparation binds the
authoritative source and produces both an ordinary old-bank successor and a
routed destination with exactly one learner evaluation. Adoption performs no
learner or router work: a ready receipt selects the routed candidate, while a
veto preserves the already-computed ordinary update and records rollback rather
than deferral. The world primitive deliberately separates ordinary validity
from destination validity, so a bad route cannot erase a valid physical-model
update when the coordinator vetoes it. Exact-content receipts reject stale or
tampered trees but are unkeyed integrity declarations, not authentication.
Their byte records describe serialized logical leaf occurrences, not allocator
peaks. These primitives remain independently usable; the v18 coordinator is
the narrow higher-level consumer and supplies no caller authentication.

The auditor and adapter close only an instantaneous frozen-consumer loss
diagnostic and within-cohort ranking surface. There is no adapted-consumer
causal deletion result, realized return, planning, control, safety, or
empirical benefit, automatic cumulant/subtask or option discovery, curation,
promotion, or go/no-go authority, scientific promotion, WP7 completion,
Alberta Plan completion, or L3 claim. They renew no registered evidence
artifact; any artifact whose registered source hashes differ remains invalid.

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
  default uncertainty to zero; dreaming is deliberately disabled for all
  three modern model lanes — ensemble, replay-rehearsal, and recurrent
  latent — until uncertainty and rollout-validity gates are calibrated, so
  guarded dreaming runs only on the legacy single-model lane;
- random anchors and uniformly random actions drive imagined updates;
- accepted dream backups consume the predicted discount; and
- aggregate error EMA is thresholded without calibration by state/action
  region.

The replay composition commits the real ensemble update, causal typed-signal
record, fixed-quota dual-memory sample, and model-member rehearsal as one
transaction. Replay has separate mask RNG, masks, update counters, and event
counters; it cannot update causal signal calibration, residual variance, the
real RNG/counters, actor, critic, or state builder. In Prototype, only the
commit-gated real representation gradient can reach the builder or candidate-update
audit. Stored transitions retain the final/bootstrap observation—not the
post-reset decision observation—and representation versions make stale/future
samples explicit. This is bounded L0 composition; there is no replay-retention
or control comparison and no actor rehearsal.

A separate `RealStateOneStepDyna` kernel now covers WP4.6's first ordered item
without changing that Prototype restriction. It records exact real
representation/action anchors before the real update, then permits planning
only from exact current ensemble and Q states whose revisions advanced
monotonically. Each accepted backup forms
`reward_hat + continuation_hat * max Q(next)` before updating one primitive
head. Support, residual readiness, epistemic/residual magnitude, finite-value,
and termination-agreement checks can veto a backup; model state is read-only,
synthetic traces start at zero, hidden utility/lifecycle state is restored, and
planning RNG/clocks/checkpoints/resources are disjoint. This is still L0:
thresholds are supplied rather than calibrated, there is no Prototype adapter,
actor update, matched planning benefit, or evidence artifact.

`EnsembleShortRolloutPlanner` now covers WP4.6's policy/uncertainty-directed
short-rollout and terminal-correct multi-step-return items. One exact real
anchor plus immutable linear policy/value and full ensemble-content receipts
produce fixed-shape proposal batches only. Every imagined transition must pass
action support, residual readiness/magnitude, epistemic, finite-value, and
member termination-agreement guards. Learned termination stops padding and
forbids value bootstrap; only a valid horizon truncation may bootstrap. Exact
revision/content aliases, RNG, clocks, checkpoints, and resource ceilings fail
closed while the model, policy, and value owners remain read-only. The lane is
not a Prototype, STOMP, or model update consumer, and its global support proxy
and thresholds are uncalibrated.

The separate `ImaginedRolloutSelectionGauge` supplies the grounded
authorization boundary before an actor consumes those proposals. One frozen
source/model generation feeds a bounded causal audit partitioned by primitive
action and caller-declared region. Independent evidence, realized-validity,
reward-error, next-observation-error, termination, success lower-bound,
top-quantile-purity, and caller-owned safety/protected checks cannot compensate
for one another. Audit records cannot authorize their own proposal, and each
receipt binds every candidate transition plus the exact calibration
revision/content. Every later transition is rejected when any valid path
predecessor fails a gate. `AuthorizedImaginedRolloutActorCritic` forms
autodiff-free authorization metadata, revalidates the source and receipt at
commit, then performs exactly one fixed-shape actor/critic backward pass for a
current receipt—or zero for stale, replayed, or tampered preflight. It uses exact
terminal rewards instead of bootstrapping terminal targets, and weights dream
imitation by bounded positive advantage. A competent-real episode-cloning mode
has the same transition/update ceilings and prefix-closed admission. Its
caller-declared competence is not authenticated, and the deterministic tags
provide post-mint integrity rather than proof that the planner issued the
original tensors. A strict `GroundedImaginationComposition` now derives the
planner policy/value authority directly from the live actor/critic, obtains the
rollout batch internally, passes that exact value into the gauge, and adopts
planner, authorization, learner, dream, and composition clocks atomically
around the only possible backward pass. This removes caller rollout-tensor
substitution at the composed boundary. Model support, the real anchor, region
assignments, safety/protection masks, and environmental truth remain caller
attestations. This is an L0 mechanism comparison surface: thresholds are
supplied, and no matched real-return, retention, Prototype, dispatch, safety,
or promotion result exists.

The explicit transition boundary now distinguishes the final/bootstrap
observation from the next post-reset decision observation. World-model, Horde,
and Bellman targets consume the former; OaK/STOMP selection, IA recommendation,
and the cached next action consume the latter. Positive-discount truncation
interrupts an active option after its final update without recording a censored
option-model completion. This repairs episodic/autoreset semantics and
integrates a bounded development ensemble, but the variance output is still an
explicitly uncalibrated residual proxy rather than certified aleatoric
uncertainty. A separate recurrent latent ensemble and source-bound prequential
development evaluator now expose real heteroscedastic heads plus descriptive
ID/OOD and evaluator-owned state/action-region diagnostics. The recurrent model
is now an opt-in fourth Prototype lane whose exact dispatched cache, real NLL
gradient, and causal signal state commit transactionally. A companion
recurrent-reset A/B/A-style diagnostic reconstructs phase and recurrence-entry
errors from exact reused cases. None of these mechanisms establishes
calibration, retention, or control benefit. The new authorization-gated
multi-step actor/critic consumer is L0 only and has no matched real-return or
retention validation.

This is a useful smoke model, not yet a reliable continual world model or a
learned search-control process.

#### 5. WP7 proposal, audit, and planning pieces are not yet one lifecycle

`CumulantSubtaskDiscovery` v1 is a standalone, fixed-universe proposal
mechanism over controllable events, feature changes, reward-relevant transition
atoms, and typed prediction bottlenecks. Its two-phase boundary arms after the
behavior action and freezes predict-before-update probes and insertion
predictions. Observation is forward-shifted: successor semantics become the
next arm's candidate values, and a reward-transition atom born on the current
outcome cannot collect learnability or contribution evidence from that same
outcome.

Every candidate must independently pass learnability, randomized-propensity
controllability, novelty, and frozen reward/model insertion-contribution
gates. Missing evidence in one gate cannot be offset by a high score in
another. Prediction-bottleneck candidates also require typed epistemic and
progress evidence and pass a persistent running-mean aleatoric veto. Novelty
is checked against incumbents and earlier selected proposals. Contribution
retains the frozen task weights and requires every task channel rather than
renormalizing around missing mass.

Each of the four families has a fixed positive quota, quotas are never
reassigned, and an incomplete family suppresses the entire discovered bundle.
The discovered cohort is matched to a cohort backed by a random projection
bank sampled once at initialization and an identity-bound, exactly `B`-entry
hand-authored cohort. All three have budget `B` and materialize into the same
compact appended tail slots; candidate IDs are not option feature indices.
Exact v1 configuration/checkpoint schemas, source/semantic/transition/revision
bindings, tamper checks, static ceilings, and resource counts bound this path.

This mechanism is deliberately separate from WP7.1c: it invokes neither Kondo
nor delight and performs no backward pass. It also cannot mutate OaK, STOMP,
Prototype, or Horde and has no curation, promotion, go/no-go, or scientific
authority. The consumer integration test constructs fresh, identically
configured STOMP agents and performs one finite update per proposal cohort;
the discovery object itself never installs or trains those consumers.

[`STOMPAgent`](alberta_framework/core/options.py) updates external return,
duration, discounted baseline mass, discount, and next-state outcome models
when an option terminates, then consumes those models in an explicitly bounded
number of differential semi-MDP backups. This closes the mechanism-level
`model → planning` edge. It does not establish that learned option models
improve held-out lifetime control over matched model-free option execution.
An opt-in Prototype composition now provides a stricter value-only search
slice: it recomputes every supported option's differential semi-MDP Bellman
residual after each accepted backup, stably selects the largest magnitude, and
commits only the base learner under a fixed resource budget. It uses the exact
next decision representation and fresh post-transition option models/reward
rate, but intentionally preserves the action that OaK already cached for that
representation. The changed values can influence only a later extended-action
selection boundary. Completion count is support, not calibration; primitive
dreams still have a separate random budget, so this is neither combined search
nor a matched-benefit result.

The standalone `OptionLifecycleAudit` adds the missing semantic-bound audit
surface around this boundary. Its exact arm/observe transaction freezes the
full option semantic/generation set, source and representation revisions,
state revision/checksum, initiation context, caller-randomized primitive
assignment, and pre-update option-model signature. Per-context initiation,
completion reasons, censoring, discounted external return, pseudo-return,
model error, planning usage, redundancy, and resource costs remain separate.
Primitive comparisons require both treatment and primitive evidence in every
configured context and use fixed equal context mass. Semantic replacement
resets all slot-local history and advances generation unless the descriptor is
bit-identical; maintenance is bounded and proposal-only. The standalone audit
object by itself has no OaK/Prototype wiring or replacement authority. An opt-in persistent
`STOMPOptionLifecycle` wrapper now supplies the live STOMP edge: actual option
ownership, starts, natural/censored endings, frozen pre-update model signatures,
return inputs, outcome deltas, planning use, and costs drive the audit. The
observer has zero control authority. Exhaustion or rejected external
attribution freezes the audit with a terminal reason while valid STOMP updates
continue; persistent composed-state corruption alone fails closed for
checkpoint recovery. Disabled auditing preserves raw STOMP state and RNG, and
explicit shape-compatible semantic rebind resets changed option-local learner,
model, trace, optimizer, and base-head state. Its signed-int32
`max_observations` cap remains a finite attribution horizon, not a control veto.

`CalibratedExtendedSearchControl` supplies a separate strict four-mode tabular
core for model-free extended-Q replay, primitive-model search, option-model
search, and their combined candidate union. Every mode shares one real-anchor
bank and the same exact backup budget; combined search never receives a second
family budget. Correct primitive and differential semi-MDP option targets are
ranked by a noncompensating product of value-change, future-anchor
reachability, model reliability, and support. Natural completion versus
censoring, pending mid-option checkpoint resume, semantic-generation resets,
stable ties, and exact resources are explicit. A strict development
evaluator runs all four modes from one immutable source/runtime-bound model
and calibration snapshot over the same evaluator-owned Threefry trace and exact
budget `B`. It preserves raw diagnostics, checkpoint/resume, exact causal
replay verification, and tamper rejection, but remains `not-assessed`: its model is frozen,
experience is action independent, and its contrasts have no thresholds or
verdicts. Calibration/support/revision state also ends at a configured
signed-int32 observation cap, which must disable secondary search without
stopping real control. Separately, a raw-representation
`PrototypeSTOMPCalibratedSearchAgent` snapshots the actual learned Prototype
primitive and STOMP option models, settles their live outcomes under one shared
budget, and preserves exact ownership. Its sidecar values have no policy,
dispatch, or keyboard authority and cannot block valid real learning. A
separate default-off `PrototypeSTOMPCalibratedDispatchAgent` v2 can consume
candidate-specific evidence at one exact anchor, form a primitive or proposal-
only option-keyboard command, intersect a caller-owned hard mask, replace the
real cached primitive once, and arm the base or already-active option owner that
will receive credit. A planned option never starts or switches an option.
Proposal unavailability may retain only an independently safe current-owner
command; withholding exposes `-1` and accepts no transition until a zero-
learning retry succeeds. This closes a mechanical policy edge, not safety,
physical dispatch, calibration benefit, or search-control evidence.

`CumulantOptionInstallation` now joins the proposal and consumer sides through
one strict L0 edge. A complete fresh proposal binds external descriptor
semantics to a fixed bank of STOMP slots; every live observation rematerializes
the four selected cumulant families, and an extended-action eligibility mask
keeps cold slots out of behavior, real bootstraps, skip diagnostics, planning
selection, and planning bootstraps. Bit-identical slots are preserved and
changed slots are wholly reset at the lifecycle's public quiescent rebind
boundary. An active option or comparator produces a no-op, not a hidden queue;
the caller must supply a later fresh proposal. Installer exhaustion freezes
replacement only, while installed control continues.

The installation's empty STOMP template may additionally reserve a fixed
suffix beyond the raw discovery prefix. Option-cumulant indices remain
immediately after the raw prefix, standalone tokens fill the reserved cells
with exact zero, and nonzero suffix tamper is rejected. This leaves the
historical zero-suffix layout unchanged while allowing a separately bound
external owner to supply its own exact hidden/generated-feature tail.

`CumulantOptionScheduler` closes the mechanical timing edge. Every accepted
scheduler transition arms and observes discovery; exact periodic cadence or a
bounded zero-payload retry requests a newly observed bundle. Installation
requires a source/canonical/lifetime-bound caller receipt with a strictly newer
authority revision and a quiescent live lifecycle. Applied installs always
advance the scheduler-owned Threefry key, while exhausted attempt capacity and
terminal scheduling clocks fail explicitly without vetoing installed control.
Maintenance emits only a bounded semantic-generation-bound retirement handoff;
it cannot retire anything.

`AuthorizedOptionRetirementController` supplies the separate execution edge.
It accepts only a live handoff plus a distinct caller receipt binding the exact
slots, source/representation, descriptors, generations, lifecycle/audit/
controller revisions, validity window, and two independent reset keys. The
controller recomputes a fixed noncompensating retirement policy, then uses two
public quiescent lifecycle rebinds to scrub approved slots and restore their
installer semantic identity before retaining an authoritative cold mask.
Selection, bootstrap, planning, and attribution all consume that mask. No
retirement or replacement payload is queued, and either rebind failure rolls
back the complete composition.

`AuthorizedOptionReplacementController` owns the next single-cold-slot
transaction without duplicating the installation subtree. It projects
retirement control from one canonical scheduler state, stages one ordinary
scheduler observation with install authority denied, and keeps the newly
discovered replacement transient. Host commit rederives and exact-compares the
preparation before a separate caller receipt may install and reactivate exactly
that slot. Decline persists only ordinary discovery/incumbent materialization
and a bounded retry; candidate materialization, its RNG successor, and its
semantic identity do not enter state. Active-option, stale/replay, freshness,
capacity, forged-preparation, and partial-commit cases fail closed. This v1
wrapper permits exactly one retirement and one replacement. Its unkeyed
checksums and receipts detect accidental drift and declare authority facts;
they do not authenticate a caller or prove cryptographic lineage.

A separate stateless v2 `FreshColdSlotCumulantCohortFilter` can prepare, but
not itself install, one same-family fresh candidate for an exact cold slot. It
revalidates family quotas and uniqueness independently of the v1 bundle
checksum, preserves all live descriptors and semantics, and rejects stale,
tampered, or cross-family material. The original candidate universe still has
no alternate; a versioned fixture with one additional eligible feature
descriptor yields the expected one-slot cohort. A separate opt-in v2
`AuthorizedFreshColdSlotAtomicSwapController` now consumes only that exact
prepared cohort. It rederives the transient authorized retirement, ordinary v1
preparation, filter source/output, and public lower scheduler/replacement
adoption preparations, then selects only an all-installed successor with one
exact reset target and all live slots preserved. No-fresh, decline, outer veto,
stale/replay, foreign identity/config, key substitution, or checksum-valid
bundle/target tamper returns the exact all-installed outer source. Preparation
uses three scheduler observations and one installation-candidate evaluation;
full commit rederivation uses six and three respectively, while adopting at
most one installation. The wrapper derives no RNG root or split beyond the
four caller keys. Receipts/checksums are unkeyed integrity declarations, not
authentication; this does not rerun or repair the consumed repeated-lifecycle
negative.

The separate opt-in `PrototypeOptionAuthorityBridge` closes the bounded live
owner edge without duplicating STOMP. One persistent `PrototypeAgentState`
contains the sole nested Prototype→OaK→STOMP owner; authority and lifecycle
state remain detached metadata with one borrowed binding. Unequal pristine
owners require an explicit directional receipt bound to both exact source
states and typed owner digests; unchanged-source reevaluation is idempotent.
The receipt, checksums, and checkpoint hashes are unkeyed integrity, not caller
authentication. Every optional Prototype sidecar crosses without
reinterpretation, and one installed-slot mask reaches real behavior/bootstrap,
internal planning, option search, guarded Dyna, and lifecycle attribution. The
lifecycle consumes the exact raw `STOMPUpdateResult` without reevaluation; a
transient five-stage trace classifies option-search, feature-route, Dyna,
memory-dispatch, and partner-dispatch mutations before metadata-only
finalization binds the sole final owner. Invalid sources cannot commit and use
a primitives-only transient mask. Dynamic audit refusal preserves valid
Prototype control while retaining authority metadata and latching
desynchronization. Diagnostics split real, imagined, total, search-update, and
internal-planning work.

Together these are L0 proposal, scheduling, installation, audit,
authorized-retirement/replacement, one-owner bridge, option-consumer, and search-control
mechanisms, not empirical evidence that discovered subtasks help. Automatic
timing and
externally authorized retirement/replacement are present, but autonomous
go/no-go authority, repeated lifecycle policy, caller authentication, matched
control benefit, physical-dispatch and safety authority, WP7 exit, evidence
promotion, SOTA, Alberta Plan completion, and a successful uninterrupted
repeated-lifecycle outcome remain open.

#### 6. OaK curation is manual and narrow

OaK's utility is an EMA of pseudo-reward while an option executes. `curate()`
runs at Python level, finds the worst option, swaps its hand-specified feature
index, and resets selected arrays. That is a useful testable mechanism, but it
does not yet measure the counterfactual contribution of a feature, subtask,
option, or model to lifetime reward or planning accuracy. The option keyboard
now has a strict deterministic proposal and an opt-in real dispatch boundary.
The replacement rewrites the precise base or intra-option action cache that
will receive the next transition's credit, requires exact decision-observation
identity and a hard safety mask, preserves RNG, and fails closed on corrupt
state or an unsafe base. Thus an explicit consumer can make a keyboard chord
govern the next primitive action without miscrediting its counterfactual. The
default OaK loop still does not autonomously select or learn which chord to
dispatch, and no lifetime-control or planning benefit has been measured.

For higher-level option coordination, OaK can now adopt one already-evaluated,
caller-authoritative STOMP result. The seam validates complete source and
transition identity, both clock layers, endpoints, and success diagnostics,
then performs OaK accounting without a second STOMP update. A separate
quiescent rebind changes only one declared reset option slot and its extended-
action head, preserving global/primitive state, RNG, and clocks while clearing
only the corresponding OaK statistics. Optional extended-action masks keep
cold slots out of behavior selection, real bootstraps, and planning. These
unkeyed trusted-caller seams prove consistency, not caller authentication or
autonomous retirement/replacement authority.

#### 7. Prototype partner fusion is opt-in and action-changing

The legacy IA companion still exposes augmentation and a recommendation as
diagnostics. Separately, `PrototypeAgent` can now opt into the bounded
`PartnerPolicyFusion` L0 core as an actual action consumer. On each accepted
transition it resolves feedback for the exact prior four-word Prototype
lifecycle ID, derives the real OaK counterfactual score, optionally supplies the
current OaK keyboard proposal, and chooses among
ignore/query/accept/discrete-blend/clarification routes under a caller-owned
hard mask. A changed primitive rewrites the exact base-or-option cache that
will receive the next credit, and the recurrent model cache receives the same
effective action. Unsafe base dispatch or a corrupt resulting state rolls back
the whole Prototype transition; stale, duplicate, or misattributed feedback is
an atomic no-op. Start remains base-only, and cold-start accepts remain
explicitly uncalibrated development exploration. A new strict stress evaluator
supplies the same frozen 96-event multi-partner stream to learned,
outcome-blinded, and base-only conditions while holding tensor/state shapes and
decision/feedback call counts fixed. Observable contexts recur across a hidden
mid-life reliability reversal, with raw cost spikes, partner and total
disconnects, hard-mask exclusions, actions, routes, feedback targets, and
recovery descriptions retained. Its exact causal replay and prefix checkpoint
are source/runtime bound, but the lane is threshold-free and `not_assessed`;
one evaluator-owned schedule is neither confidence calibration nor a
partner-benefit result. A separate consumed 12-execution lane now closes the
mechanism-only gap between that stress trace and the real agent loop: three
independently owned `PrototypeAgent`/fusion/environment arms receive only the
same frozen exogenous context, noise, availability, cost, and mask schedule,
then generate their own observations, rewards, messages, realized-assistance
feedback, and actions. It exercises an uncued reliability reversal, total and
partial disconnects, cost spikes, caller-owned hard masks, real action-changing
updates, exact causal replay, and in-memory resume. This is still L0
`not_assessed` instrumentation, not a matched causal comparison: the learner
states and RNG streams are independent, the life is only 12 executions, and
there are no thresholds, trials, calibration result, or promotion authority.
A separate v2 evaluator removes that initialization confound: learned-feedback,
fixed-zero outcome-blind, and empty-message base-only wrappers begin from
bit-identical typed RNG, Prototype, fusion, and environment state. The
context/noise/drift/availability/cost/mask schedule is paired, but each arm owns
its later causal trajectory after action divergence. Exact raw hash chains,
prefix reconstruction, replay, checkpoint/resume, source/runtime/config
binding, eager/JIT Prototype parity, and matched logical work are covered. On
the consumed 12-event run, learned and fixed-zero each changed the action three
times with equal task/net return; base-only changed none. Learned and
fixed-zero final Prototype states differ without realized behavioral
separation. This is a null descriptive L0 `not_assessed` result with no
threshold, winner, artifact, efficacy, or promotion authority.
The standalone recommendation protocol can alter a partner's action and has a
historical held-out **valid rejection**: reward uplift and both augmentation
controls passed, but action-changing intervention prevalence missed its frozen
threshold. The archived v1 rejection remains historical evidence; its prior
exact replay on the already-consumed schedule is nonpromoting, and subsequent
`average_reward.py` drift makes current-source compatibility invalid. The
p=0.75/seeds-60–89 v2 lifecycle is an unissued, permanently development-only
contract: its self-issued plan has no trusted external pre-run chronology, and
`internally_accepted` is hard-coded false. Any future acceptance claim requires
a new schema, untouched seeds, complete shards, and an external chronology
anchor. The new composition closes the action-consumption mechanism gap only:
without a calibrated reliability result or a matched closed-loop partner-uplift
result, it is not evidence of intelligence amplification.

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

Every constructed report remains `not-assessed`. A fixed v1 in-memory
development report now runs `PrototypeAgent`, a running-reward bandit, and a
frozen-action baseline over consumed seeds 1701/1702 with independent
functional environment states. It embeds reconstructing evaluator reports,
raw action/decision ownership, exact opportunities and logical persistent
bytes, deterministic logical latency, available parameter/policy/value churn,
and explicit available/inapplicable/unavailable entries for every WP1
diagnostic. Source/runtime replay and checkpoint continuation are validated.
This is the literal versioned report-construction witness, not an outcome:
there is no artifact writer, promotable seed panel, factorial runner, realized-
compute match, accelerator-memory/energy backend, internal gradient/NTK
measurement, or enabled dynamic-component/world-model diagnostic. A strict
paired multi-seed campaign runner now binds
seeded evaluator identities, embeds every raw report, preserves unavailable
pairs, and reconstructs deterministic stratified-bootstrap intervals; no
promotable Prototype campaign has been executed through it. The held-out control probes
score one action rather than a frozen-policy rollout, and fresh-per-regime,
oracle-data, stationary-multitask, and realized-resource-matched references are
still missing. These mechanisms become evidence only after those gaps are
closed in a preregistered protocol.

The fail-closed `complete_prototype_manifest` provides the distinct final
accounting boundary. It enumerates all 18 acceptance properties and accepts a
row only through immutable artifact bytes plus a trusted validator receipt for
the same pinned prototype configuration and exact role, frozen L3 evidence,
untouched held-out seeds, protocol/scientific-outcome digests, and complete
source-hash closure. Its row statuses, aggregate flags, overall status, and
self-digest are reconstructed before a command-style status code is returned.
Enabling optional paper delight or Kondo adds actor-learning and measured-
compute guardrail roles. With no default evidence bindings, it correctly
reports that the current mechanism inventory is not a complete prototype.

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

The local [`upgd.py`](alberta_framework/core/upgd.py) deliberately extends the
paper algorithm rather than reproducing it. It:

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
algorithm an honest identity, and existing results should not be invalidated
by a silent semantic rewrite. At the audited snapshot the module also
misattributed UPGD to the continual-backpropagation (Dohare et al.) line; its
docstring now attributes Elsayed and Mahmood correctly, documents each
deviation, and points to the canonical reference below.

### Resolution status

1. **Done — preserved.** The historical learner keeps its name and exact
   update; its docstring labels it a deliberate extension (absolute
   \(|wg|\) utility, power gate, unprotected task gradient) rather than the
   paper algorithm.
2. **Done — attribution corrected** in the `upgd.py` module docstring.
3. **Done —**
   [`canonical_upgd.py`](alberta_framework/core/canonical_upgd.py) implements
   first-order protecting and non-protecting UPGD and UPGD-W as a small JAX
   PyTree transform with named source profiles (`paper_global`,
   `official_readme_global`, `official_experiment_global`,
   `official_experiment_local`, `paper_local_literal`, `safe_extended`) that
   keep the published paper/README/experiment-code discrepancies visible.
   It also keeps two adaptive identities separate: `OfficialAdaUPGD` is an
   equation-level port bound to the released RL `AdaptiveUPGD` at commit
   `b75e90ad4b09c28971ac9dbb902a8fd86709b28c` and preserves that source's
   first/second moments, raw-utility maximum, noise placement, two-alpha
   direction, one-alpha decay, and numeric quirks; `AlbertaAdaUPGD` is a
   guarded derived first-order extension. Neither is an efficacy result or a
   selected default. Second-order utility remains future work.
4. **Partially done.** Fixed-perturbation tests in
   `tests/test_canonical_upgd.py` and `tests/test_canonical_adaupgd.py` pin each
   profile's update numerically. Official adaptive parity is deliberately
   bounded to the finite, all-active, single-group fixed-noise equation; the
   aged PyTorch stack and its implicit RNG are not claimed bit-identical.
5. **Open.** Compare signed versus absolute utility, global sigmoid versus
   local power scaling, protecting versus non-protecting,
   Gaussian/Rademacher/no noise, and trunk-only versus all eligible
   parameters.
6. **Open.** Report plasticity and forgetting separately. A single final
   accuracy number cannot show which problem was solved.

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
| [Self-Normalized Resets](https://openreview.net/forum?id=G82uQztzxl) | A/B | Resets only when inactivity is statistically surprising relative to a unit's history | Bounded L0 baseline now exists; run it in the matched matrix |
| [L2-to-init / regenerative regularization](https://arxiv.org/abs/2308.11958) | B | Keeps parameters near initialization; often strong in RL and Forager | Required cheap baseline |
| [Weight Clipping](https://arxiv.org/abs/2407.01704) | B | Bounds parameter growth after each update | Required cheap, noise-free baseline |
| [Spectral regularization](https://arxiv.org/abs/2406.06811) | A/B | Controls layer spectral norms and gradient conditioning | A source-profiled single-dense-layer L0 mechanism now exists; it still needs a generic network arm and matched matrix |
| [C-CHAIN](https://proceedings.mlr.press/v267/tang25g.html) | B/C | Links NTK-rank loss to output churn and penalizes out-of-batch churn | An independent Equation 8/NTK L0 comparator now exists; benchmark only after implementing the full sequential arm, because the public code has no verified reusable license |
| [FIRE](https://arxiv.org/abs/2602.08040) | A/C | Boundary-time projection toward isometry | Useful diagnostic comparator; boundary dependence conflicts with the primary setting |
| [Dynamical Isometry / AdamO](https://arxiv.org/abs/2606.09762) | C | Regularizes Jacobian singular values and proposes an Adam-style optimizer | A clean-room single-matrix Equation 16/19/20 L0 mechanism now exists; benchmark before any optimizer/default claim |
| [Optimization-Centric Plasticity](https://arxiv.org/abs/2603.21173) | C | Argues that task-specific zero-gradient/local-optimum trapping, rather than capacity loss alone, drives apparent dormancy; parameter constraints reduce entrenchment | A bounded A/B/A gradient-flow/L2 diagnostic now exists; run it in the matched matrix and do not infer plasticity from dormant fraction alone |
| [Calibrated Partial Resets](https://arxiv.org/abs/2607.24996) | C | Pulls low-utility neurons partway toward initialization; strong very-long-run claims | A pinned-reference single-layer L0 mechanism now exists; the week-old v1 preprint still warrants an experimental matched lane only |

The new `SelfNormalizedResets` core makes that recommendation executable for
one fixed-width dense ReLU layer. It uses exact per-unit ages and a bounded
completed-gap ring, estimates a positive-support geometric law, evaluates the
observed silent-run survival in stable log space, and applies deterministic
post-optimizer reset slices to incoming weights/biases, outgoing weights, and
supported optimizer state. Its explicit mapping is
`P(A >= age + 1) = (1 - p)^age`; this is a documented Alberta indexing/window
convention rather than a claim of bit-equivalence with the authors' released
silent-age histogram. Long-clock, cap, corruption, reset-persistence,
checkpoint, and eager/JIT/scan tests establish L0 mechanism integrity only.
No plasticity/retention comparison or scientific result has selected it.

The new clean-room `CChain` surface independently implements the paper's
Equation 8 term, `0.5 * mean((f_current - stop_gradient(f_lag))**2)`, on a
declared reference batch disjoint from the base-loss batch. It requires one
scalar model output per reference sample and rejects vector-valued per-sample
extensions. It forms one
combined gradient only after exact state, parameter, lifetime, and declared
sample-identity preflight; commit uses no autodiff and shifts the accepted
source into a one-step-lag reference slot. The appendix running absolute-loss
ratio is present with explicit Alberta window, warmup, epsilon, and clamp
controls. Diagnostic empirical-NTK rank follows the paper's minimum
singular-value-prefix definition and is kept separate from update authority.

This closes an equation/instrumentation gap, not the C-CHAIN outcome gap. The
surface does not reproduce the full sequential PPO/DQN algorithm, does not
authenticate model/loss callables, dataset provenance, or caller-applied
optimizer state, and is not composed into Prototype. It has no matched
plasticity/retention/control result, default status, artifact, evidence, or
promotion authority.

The development-only
`OptimizationCentricPlasticityDevelopmentReport` now crosses an ordinary
nonlinear SGD learner and an initialization-centred L2-constrained twin on one
frozen evaluator-owned A/B/A stream. Both start from the same immutable
bit-exact parameters and receive the same samples and update opportunities.
At each switch, raw old/incoming gradient vectors, alignment, fixed-radius
local loss probes, and parameter displacement/churn are retained; dormancy is
measured alongside them but is not an input to the fixed descriptive
zero-gradient/local-neighbourhood flag. Source/runtime/config/protocol identity,
full causal reconstruction, and zero output/evidence authority are explicit.
The single 12-update schedule has no calibrated threshold or efficacy verdict,
so it is a diagnostic mechanism rather than support for OCP, L2 constraints,
or a default plasticity method.

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

The current working tree now has the bounded routing boundary needed to keep
that separation explicit. `LearningValueRouter` validates the eight named
channels independently, records their producer/object/units/domain metadata,
and computes only causal pre-update Welford normalizations. Its named actor,
exploration, memory, adaptation, safety, and full candidate-audit evidence routes
mask unavailable fields to exact zero and never create a universal scalar.
Safety remains independently routable, and the `delight` field
is accepted only when it exactly matches the float32 advantage-surprisal
identity. The candidate-update audit independently enforces the same evidence
identity before treating its eight-channel bundle as complete; it does not
perform the paper's Kondo selection. An opt-in Prototype owner now advances one
persistent router state only with an accepted real outer transition and feeds
the audit the raw candidate-evidence route. Representation-candidate validity
gates only the candidate/probe facts, not producer availability; normalized
values never enter the audit. The enabled composition is checkpointed as v19,
while disabled configs and state PyTrees retain their historical bytes. This is
L0 mechanism coverage: producer-declared uncertainty is not thereby calibrated,
no consumer benefit has been demonstrated, and the router does not decide
whether a sample receives a backward pass.

### 5. Candidate-update safety audit (separate from paper delight)

The repository also contains an introspective, gradient-level mechanism: given a
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

The canonical `apply_candidate_update` boundary (historically exported as the
compatibility-only `apply_gradient_joy_update` alias) reassesses the candidate
internally, derives the effective stored delta after dtype cast and
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

The canonical public surface is `CandidateUpdateAudit*`,
`assess_candidate_update`, `apply_candidate_update`,
`PrototypeCandidateUpdateAuditEvidence`, and
`candidate_update_audit_evidence`. Historical `GradientJoy*`, function, result,
and Prototype keyword spellings remain exact compatibility aliases only.

This candidate-update audit is distinct from the user's paper-defined meaning
of “delight” and “sparks joy.” In
[Delightful Policy Gradient](https://arxiv.org/abs/2603.14608), action
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
same \(\chi_t\) to decide whether a sample deserves a backward pass. Delight is
the exact float32 statistic \(A[-\log\pi(a\mid s)]\). The Kondo gate forms
detached forward admission intent; “sparks joy” is true iff the actor consumer
actually includes that exact contribution in a backward pass it executes. The
fact is independent of gradient finiteness, parameter-update acceptance, and
later outer-transaction acceptance. The forward gate compares delight with
compute price \(c\):

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

The current working tree now contains that bounded L0 boundary in
`KondoGate`. It derives float32 delight internally from advantage and the
selected-action log probability and implements finite-temperature
Bernoulli-price and deterministic fixed-rate top-k modes. The latter follows
the released reference's valid-count/ties-to-even target rule but uses a stable
lowest-index tie break rather than its threshold-based over-selection. It
preserves caller-declared forced samples and reports a flag requiring caller-managed full-shape fallback instead
of silently truncating an over-capacity selection, and exposes a config-bound
fixed-capacity host gather for downstream autodiff. When capacity is below
batch size, tests inspect the smaller backward JAXPR. The screen separately
has eager/JIT/scan parity; invalid transactions, checkpoints, and accounting
are covered.

`KondoSparseActor` is now the first concrete actor consumer of that boundary.
Its nonlinear categorical actor performs a full forward screen, binds exact
action identity, policy revision, and bitwise behavior log probability, gathers
the fixed-capacity actor-only batch, and invokes `jax.value_and_grad` only
afterward. A capacity-3 versus batch-6 JAXPR witnesses the smaller backward
shape. Sparse and full-shape execution tests perturb rejected features,
actions, and detached advantages and require the selected mask, actor loss,
and actor gradient to remain bit-identical. That makes “sparks joy” an
observed fact that the contribution entered an executed actor backward,
not merely the standalone gate's forward verdict. Forced guardrail rows and
Bernoulli overflows take an explicitly labeled
full-shape masked fallback that retains every selected sample. Full-batch
protected arrays pass through with a canonical digest. Returns and baseline
predictions enter the actor only through detached advantage; critic and safety
features stay outside its loss, and the protected learners remain full-batch
and ungated. Corrupt transactions roll back, and resources and source-bound
checkpoint integrity are explicit.

`KondoExecutedActionLineageBridge` now binds that actor execution to the
action-stack decision that actually became `P`. Each unmasked proposal row
commits to the full post-memory source, preparation, decision, candidate
binding, actor snapshot, typed key, revision, sampled action, and bitwise
behavior log probability. The bridge reconstructs public adoption and admits
the row only when the proposal digest is the consumed planner candidate and
the same action is planner-before-mask, final `P`, and the action named by the
following real transition. A memory-selected or overridden row is rejected.
Invalid rows are sanitized before one actor step, protected critic/baseline/
return/safety arrays stay full-batch, and an all-invalid batch performs no
backward and preserves the actor exactly. The nested `KondoSparseActorResult`
is still the only canonical execution-level joy surface. These bindings are
unkeyed host L0 integrity, not caller or physical-execution authentication,
dispatch, safety/critic execution, efficacy, evidence, or promotion.

`HCCLKondoContinualDyadRoute` v3 composes that lineage with the atomic HCCL donor.
Its `event0` installs the first actor-owned `P` pair and compact certificate
without actor backward. Each generic successor `event` consumes the prior
certificate through one actor transaction before sampling and atomically
installing the next pair. Actor input is the 23-wide post-memory base; the
factorized planner is a disabled-dispatch learning shadow, masks must be all
true, and two live action stacks remain the sole Prototype owners. The route
prepares its causal-core event and then derives exactly one learned-memory input
per agent: uncertainty unavailable, safety cost available exact `+0.0`,
reliability one, row source identity, and provenance `2 * source_step + row`.
No caller memory-metadata surface remains. A later child veto returns
persistent state bit-exactly while the nested actor result still records any
actor backward that already executed; the route exposes no joy alias. V3 owns
the protected payload: each successor updates zero-initialized linear reward-
value and cost-value heads over both exact transition rows using detached
pre-update next-value targets, then passes the resulting baseline/return target
directly to Kondo. Cost is current `PP safety_cost + message_charge`; both are
zero in this world. Scheduling, keys, and masks remain external. The learner-
only checkpoint is not composite route recovery. This is a working-tree L0
mechanism refresh, not a revision of the frozen research conclusions or a
claim of autonomous execution, authenticated dispatch, physical safety or
critic efficacy, matched benefit, evidence, promotion, or Alberta Plan
completion.

A strict four-arm development evaluator now compares ordinary full backward,
capacity-matched uniform sparse backward, Kondo top-k sparse backward, and a
separate diagnostic overflow fallback from one immutable parameter snapshot
and source trace. It discloses unequal selected samples, exact compiled
backward shapes/invocations, logical multiplication proxies, held-out-within-
development diagnostics, and parameter changes. Its separately provenance-
bound timing section compiles, warms, blocks, and evaluator-interleaves the
fixed kernels before retaining raw `perf_counter_ns` samples and nearest-rank
p50/p95. Timing excludes host screening/gathering and measures neither memory,
energy, nor end-to-end latency. Wall-clock bytes are outside exact deterministic
replay, and every outcome remains threshold-free and `not_assessed`; no
speedup, efficacy, safety, or promotion conclusion is formed.

All three development evaluators now serialize v2 contracts. Cross-arm
outcomes describe executed actor-backward inclusion neutrally; replay and
on-policy records use `executed_actor_backward_mask` with canonical meaning
`gradient-contribution-entered-executed-actor-backward`. The canonical
execution-level use of `sparks_joy` is an actual `KondoSparseActorResult`:
ordinary-full and uniform-sparse use manual backward kernels rather than Kondo
transactions, and ordinary-full makes no delight-selection claim.

A separate strict replay lane now composes the same actor consumer with
full-batch baseline, critic, representation, world-model, and
safety/guardrail learners over one uninterrupted A1/B/A2 contextual-gambling
trace. Ordinary-full, capacity-matched uniform, paper top-k Kondo, and a
fixed-capacity top-k-plus-minimum-random-reserve extension each receive one
actor and one protected update opportunity per source batch. The protected
gradients, results, predictions, and final states are independently
bit-identical across arms, including the rare-failure stratum. It records
current-policy delight, executed actor-backward inclusion masks, gather indices
and shapes, full protected coverage, logical row-slot proxies, descriptive
recurrence/recovery/retention readouts, and exact checkpoint-prefix replay.

This replay is evaluator-supplied and action-fixed. It has no source behavior
policy and applies no importance correction, so its actor updates are
off-policy surrogates rather than valid policy-gradient or DG-efficacy
estimates. The selected-action surprisal is the current actor's probability of
the recorded action. The reserve is an Alberta extension, not paper Kondo, and
the logical proxies are not measured FLOPs or runtime. The lane has no wall-
clock, memory, energy, policy, safety, output, evidence, or promotion verdict;
all status remains `not_assessed`.

The adjacent closed-loop development lane samples each arm's own actions under
one immutable actor revision per batch, then performs exactly one actor and one
full protected update at the boundary. Evaluator-owned typed Threefry uniforms
pair only exogenous randomness; trajectories are independent and never assumed
equal. Exact actions, behavior log probabilities, revisions, environment
parents, rare-failure coverage, source/runtime bindings, checkpoints, and
causal replay are retained. Host Kondo screen/gather orchestration remains
non-JIT while collection and fixed backward kernels are JIT-compatible. This
is still L0 mechanism shape: there is no dream integration, measured compute
saving, DG reproduction, learning benefit, safety result, evidence, or
promotion. Unkeyed checkpoint digests are integrity checks, not authenticity.

The surrounding family adds:

- [Delightful Distributed Policy Gradient](https://arxiv.org/abs/2603.20521):
  suppress rare failures but retain rare successes from stale or faulty actors
  without requiring behavior probabilities. It is a biased robustness
  heuristic, not importance sampling; retain clipped ratios or V-trace when
  behavior probabilities exist;
- [Delightful Exploration](https://arxiv.org/abs/2605.13287) (paper title;
  repository code calls its non-gradient quantity an exploration score): gate
  exploratory overrides using prospective expected improvement times host-policy
  surprisal. Its Pandora equivalence is exact only for revealed-value search;
  in noisy independent-arm bandits expected improvement is a
  value-of-perfect-information proxy that upper-bounds the one-step knowledge
  gradient; and
- [Delightful Gradients Accelerate Corner Escape](https://arxiv.org/abs/2605.11908):
  tabular convergence/corner-escape results plus an exact shared-function-
  approximation counterexample in which parameter coupling admits a suboptimal
  interior fixed point.

The tree now contains a strict `ProspectiveExploration` L0 implementation of
the separate exploration equation. One fixed candidate batch is scored by
expected improvement times capped host-relative surprisal or by random,
epsilon-greedy, ensemble-disagreement, information-gain, and learning-progress
comparators under the same budget and logical RNG schedule. Event, owner,
producer-revision, and pre-decision attestation receipts fail closed. Ranking
is completed without reading the caller-owned hard shield; afterward the
shield can admit the candidate or a separately permitted host fallback. The
module estimates none of its supplied scores, and its Boolean shield is not
physical-safety evidence. A separate consumed eight-event development lane
closes that score-production loop only for a tiny synthetic setting: each of
the six arms owns its environment, linear-TD ensemble, selector, and shield,
and derives expected improvement, disagreement, information gain, and positive
learning progress only from its own executed transitions. Pairing stops at
exogenous noise. The world contains a progress-resetting noisy-TV action and a
delayed invest/collect opportunity; the caller hard mask remains the sole
admissibility owner. Exact causal replay, raw hash chains, in-memory resume,
and matched logical budgets validate mechanics only. The report is
`not_assessed`, consumed, threshold-free, artifact-free, and winner-free; the
linear estimator is not exact sequential value of information. Neither lane
establishes environmental efficacy or paper reproduction. The canonical v2
API calls its non-gradient quantity an expected-improvement--surprisal score:
it is not DG/Kondo delight and the selector executes no actor backward.
Historical v1 `DelightfulExploration` import/config spellings are
compatibility-only; v1 checkpoints remain fail-closed.

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
7. integrate and evaluate the Kondo gate as compute allocation only after DG
   reproduces;
8. measure quality per environment step, forward pass, backward pass, and wall
   clock; and
9. reject the method if it worsens safety, old-skill retention, calibration, or
   effective sample size.

If a later trace experiment gates each newly added score term,
\(e_t=\gamma\lambda e_{t-1}+\operatorname{stopgrad}(w_t)\nabla\log\pi_t\),
label it an Alberta extension rather than paper-equivalent DG. A minimum
catastrophe-update floor and change-triggered gate reset are likewise sensible
safety/nonstationarity extensions, not claims from the papers.

Neither paper-defined delight nor the candidate-update audit should be folded
into UPGD utility. UPGD asks whether a **parameter** is currently useful.
Delight is the scalar attached to an **actor sample**: DG uses it as a detached
weight, while Kondo uses it to form forward admission intent and the actor
consumer determines whether the sample actually enters a backward pass. The
separate candidate-update audit asks whether a
**candidate parameter update** passes objective,
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

The repository now has a bounded typed `ExperientialMemory` and a strict
development-only evaluator for those failure modes. The evaluator owns one
fixed recurring A/B/A schedule, queries an immutable empty-snapshot copy before
each write, and retains the complete neighbor, gate, prediction, fallback,
harmful-recall, and eviction trace. It compares with a stateless no-memory
fallback on the same event opportunities and reconstructs first/return
descriptions without exposing regime or expected-outcome annotations to the
memory. A separate stateless `ExperientialMemoryPolicy` turns only retrieved
categorical action mass into a safe lowest-index argmax, with raw mass,
normalized mass, reliability, provenance, and hard safety kept separate. The
opt-in Prototype composition now queries this boundary on the next decision
representation before writing a grounded exemplar of the action that actually
ran, bootstrap representation, and reward. Memory dispatch precedes partner
fusion, and required unsafe/corrupt memory transactions roll the whole
Prototype event back. Full decision identities, no-memory shape compatibility,
curation, checkpoints, eager/JIT/scan parity, and the two deterministic
pre-state queries are explicit. This is instrumentation and L0 integration,
not a transfer result: no threshold is applied and the fallback has no matched
storage allocation.

A separate `LearnedExperientialMemoryController` now makes the narrow learned
retrieval/eviction question executable. It retains the fixed store as the sole
memory owner, learns only an additional admission veto over already-safe
retrievals, and learns the nonnegative per-row utility values that the real
eviction rule consumes. An admitted read owns one pending receipt; matching
feedback updates only when the caller declares actual use and a bounded
same-decision counterfactual delta. Insertion-clock, provenance, and source
checks prevent feedback from following a slot across replacement. This is a
bounded L0 mechanism with zero RNG and strict replay/checkpoint contracts, not
an authenticated causal estimator, learned representation, or memory-benefit
result.

A separate v1 `ExternalLearnedStateLiveMemoryAdapter` now composes that
controller as the sole memory owner around one external full-GRU/router/audit
coordinator, with the inner Prototype historical memory disabled. It settles
an exact prior receipt first, updates the coordinator exactly once, queries the
next raw decision observation before writing the actually executed current
transition, and permits only admitted exact one-hot retrievals to use
Prototype's public cached-action replacement under the caller's hard mask.
Soft retrievals and safe-base fallbacks have no action or learning authority.
The pending binding preserves the memory transaction, Prototype decision,
pre-retrieval base action, effective action, retrieval action, and exact mask;
coordinator, memory, binding, and replacement adopt together or return the
complete source for retry. This is host-orchestrated L0 ownership and causal
ordering, not authenticated feedback, a transfer result, or promotion.

A separate v1 `HCCLLearnedMemoryFeedbackBridge` now replaces the caller-shaped
counterfactual input for one constrained HCCL path. It owns exactly one HCCL
world/attribution adapter state, one learned-memory controller state, and one
fixed pending binding. Preparation binds a categorical admitted retrieval to
the exact controller transaction, HCCL source/decision and event, B/M receipt
identities and contents, common hard mask, selected agent, retrieved action,
and routing outcome. The exact eight-proposal event then supplies only that
agent's immediate `memory_total.net_reward`; the other agent's effective B and
M actions must match. Masked or unrouted retrievals settle through the
controller's no-learning path, while any stale, tampered, cross-event,
cross-slot, bound, or downstream failure preserves the whole composite for
retry. This is memory-utility feedback: it computes no exact actor-sample
delight statistic and executes no actor backward. Host/eager orchestration,
strict in-memory checkpointing, and bounded resources are L0 contracts only;
no agent, life, run, artifact, threshold, benefit, evidence, or promotion claim
follows.

The standalone `ConsolidatedMemory` adds fixed-capacity semantic and procedural
stores. Semantic GVF, fact, and affordance records, plus procedural skill
records, carry canonical identity and provenance hashes, generations,
confidence, source/representation revisions, evidence moments, clocks,
staleness, and invalidation. Procedural records also retain success/failure and
outcome moments with an exact option-lifecycle link. Query always precedes a
compatible same-generation merge, next-generation reset, or deterministic
bounded replacement; stale and invalid records cannot be retrieved. Strict
checkpoints and exact resources are L0 storage contracts. Two opt-in wrappers
now consume the same store from live Prototype control. The procedural wrapper
settles exact pending outcome feedback, preserves actual-action learning, then
applies the conservative readout to the next cached primitive after
experiential memory and partner fusion under the intersection of all hard-
safety masks. A separately versioned semantic wrapper settles that procedural
feedback first, performs a semantic pre-write query/current-record write on the
shared store, and supplies the accepted prior payload—or an exact zero tail—to
the ordinary Prototype representation for its next decision. Exact decision,
request/record, provenance, generation, kind, lifecycle, and source/
representation revision bindings fail closed. Terminal memory becomes a no-op
while valid Prototype control continues. The same wrappers now persist a
checksum-bound decision, selected primitive, and exact mask for post-envelope
settlement. An admitted unchanged action is a no-op; a changed admitted
fallback atomically rebinds Prototype's actual credit owner and cancels only
matching procedural and partner recommendation owners. No-action, stale,
disallowed, corrupt, or partially satisfiable settlements preserve the whole
state for retry and write no learner, memory, reliability, or physical action.
The caller settlement is not authentication that a command executed.

`PrototypeEmbodiedCommandAdapter` now replaces that caller-shaped last edge
with a narrow real-envelope identity transaction. A fixed bank gives each
primitive exactly one float32 `EmbodiedCommand`; preparation binds the live
semantic dispatch owner and hard mask to complete telemetry, control-clock,
version, and envelope-source identity. Settlement recomputes
`EmbodiedSafetyEnvelope`, bit-compares the whole result, and reverse-maps only
a unique selected command or certified mask-admitted fallback before changing
the semantic owner. A real no-action rejection and a stop-only capacity path
still adopt the envelope ledger or latch, close the attempt, and retain the
same semantic owner for a fresh-identity retry. This is neither a command-
geometry proof nor physical dispatch, caller authentication, learned safety,
deployment, evidence, or promotion authority.

`PrototypeEmbodiedDevelopmentHarness` composes exactly one such adapter with
a bounded deterministic plant and a grounded shadow state. For an accepted
proposal or certified fallback, it obtains the actual plant reward and
successor observation, settles the semantic owner, applies exactly one real
Prototype transition, and atomically adopts the live rearmed owner with the
transient shadow result. Two accepted cycles demonstrate reset-free alignment
of plant observations and Prototype/OaK clocks. No-action remains an envelope-
only retry; stop, fallback, shadow mismatch, replay, tamper, and checkpoint
paths fail closed. Plant-capacity exhaustion stops future scheduling without
inventing termination or truncation. This is L0 whole-agent reachability, not
physical dispatch, a safety certificate, delight/KondoGate-intent/
KondoSparseActor-backward assessment, efficacy, evidence, promotion, or a WP9
exit.

A paired slow development benchmark then gives adaptive STOMP and a
`zero_stomp_step_size_control` separately owned copies of that harness. Exactly
five STOMP optimizer step sizes may differ. After normalizing the materialized
base-LMS step-size leaves, every other initial semantic array, RNG, cache,
observation, trace, and clock must be dtype/shape/typed-key-implementation/
host-byte exact. Starts require empty harness pending/last-commit records, an
unset adapter settlement ledger, zero Prototype/OaK/adapter/plant clocks, and sufficient
remaining capacity. V1 fixes four continuing attempts and the bridge
disconnect at attempt 1, producing three real plant/Prototype commits plus one
exact unavailable action/reward record per arm. Reports retain raw
availability, fallbacks, plant state, clocks, rearming, adopted updates, shadow
work, and exact logical/resource budgets. Both lifetime AUCs are normalized
over named committed-transition or attempt indices, not adaptation or post-
change AUC. Sixteen fast synthetic pytest contracts cover typed-key identity,
exact sentinels, signed-zero identity, live selected-source/runtime drift,
replay, externally supplied prefix reconstruction, and content-plus-resealed
tampering. The real lane runs only through
`alberta-prototype-embodied-paired-development`, emits JSON to stdout, and
writes no artifact. The zero-step invariant covers only step-size-governed
real-owner STOMP parameters, not decay-only option-model EMAs, semantic memory,
shadow learning, traces, or caches. It remains `not_assessed`, with no winner,
threshold, physical dispatch, efficacy/safety, delight/KondoGate-intent/
KondoSparseActor-backward assessment, semantic use of historical `GradientJoy`
compatibility names, evidence, deployment, promotion, or WP9-exit claim.

A frozen 17-event development stress evaluator now compares this store with a
same-kernel masked-readout ablation and a zero-storage/no-kernel comparator from
one empty source-bound snapshot. Raw traces reconstruct query-before-write,
precision, abstention, harm, recurrence/recovery, retained semantic utility,
stale-skill harm, eviction/provenance, counters, and exact resources; strict
source/runtime/protocol bindings, checkpoints, eager/compiled parity, and
integrity-bound causal replay fail closed. This is `not-assessed` instrumentation. The
no-memory arm is experience/opportunity matched but not storage/compute
matched, the schedule is evaluator-owned and finite, and no threshold or
efficacy result exists.

`ConsolidatedProceduralMemoryPolicy` remains a separate stateless proposal
boundary over an already-produced procedural retrieval. Exact lifecycle
compatibility, evidence consistency, a Wilson success lower bound, finite
outcome moments, derived uncertainty, nonnegative categorical score mass, and
a caller hard-safety mask are all noncompensating gates. The policy itself
makes no memory query or write, uses no RNG, and has no dispatch or promotion
authority; the live wrapper, not this readout, owns the explicit cached-action
composition. Neither wrapper establishes procedural/semantic transfer,
negative-transfer, stale-skill efficacy, safety, or physical-dispatch benefit.

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
| Learned state under partial observability | Forager, recurrent generate-and-test/trace literature | **Highest-priority outcome/comparison work; strict live RTU replacement remains L0 only** |
| Task-free context detection and expert reuse | [SWOKS](https://proceedings.mlr.press/v274/dick25a.html), [task-agnostic online GP mixtures](https://arxiv.org/abs/2006.11441) | **Compare after the fixed two-event quarantine; retain predict-before-outcome ownership and disclose their buffers, thresholds, growth/merge rules, and rollback delay** |
| Continual world model | ICML 2025 online shallow model, Continual-Dreamer | **Adopt shallow reference and bounded replay** |
| Memory-efficient model replay | ARROW | **Reproduce dual-buffer idea, not results by assertion** |
| Actor forgets despite model memory | Dream Rehearsal preprint | **Component probes immediately; rehearsal experimentally** |
| Noisy surprise | Disagreement, Plan2Explore, aleatoric mapping, learning progress | **Separate uncertainty channels** |
| Delightful actor updates | DG and corner-escape papers | **Small isolated experiment** |
| Compute-budgeted learning | Kondo gate | **L0 nonlinear actor consumer gathers before a real sparse backward; integrate with replay/actor-critic and measure after DG; retain minimum update floor** |
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
- [`lalalune/alberta`](https://github.com/lalalune/alberta), the upstream this
  tree forked from;
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
| `C-CHAIN`, `FIRE`, and several newer research repos | No verified reusable license | Keep implementation independent: C-CHAIN's Equation 8/NTK comparator is now clean-room and L0; obtain permission before using repository code |
| [`google-deepmind/egg@d005eac`](https://github.com/google-deepmind/egg/tree/d005eac307a0cb6ccb9c63ad03aee39a3e3c30d4) | Apache-2.0 | Pinned initial small DG/Kondo token reference; does not cover the full paper evidence, and its thresholded masked loss is not a sparse-backward implementation |
| [`iosband/trl-dg`](https://github.com/iosband/trl-dg) | No declared software license | Inspect behavior; independently implement equations |
| [`lalalune/kondo-gate`](https://github.com/lalalune/kondo-gate) | MIT; community implementation | Useful API comparison, but its masked full backward does not demonstrate compute savings |
| Later DG/delight paper family | No separate verified reusable implementation | Implement from equations behind an experimental API |

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
