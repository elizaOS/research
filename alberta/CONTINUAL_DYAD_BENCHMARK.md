# Continual Dyad Benchmark — staged HCCL successor design

## Status and authority

This document describes a staged successor to the current hidden-Prototype dyad.
Its eventual full target remains **Hidden-Cue Cooperative Corridor Life v1
(HCCL-v1)**. HCCL-v1 is an unimplemented, unexecuted design, not an issued
protocol, benchmark result, evidence artifact, default configuration, or Alberta
Plan completion claim.

The smaller **HCCL causal core** is still the next integrated-life target. Its
first L0 subset is now code-complete: a fixed world/event-receipt owner, the
standalone adjacent-cube attribution kernel, and an atomic adapter between
them. The adapter consumes caller-prepared `B/M/P` receipts; it does not provide
the two learning agents or execute a life. Additional L0 rungs now atomically
compose two live action stacks, slow contexts, HCCL, and a paired planner, and
a recurring Kondo route can own `P`; they remain caller-driven mechanism
contracts rather than the integrated life. This document itself does not
authorize execution. It does not issue a runbook; reserve, derive, consume, or
hold out a seed; create a writer, artifact schema, threshold, validator, or
promotion path; select a future-utility setting; or grant evidence authority.

The stages are:

| Stage | Purpose | Deliberately absent |
|---|---|---|
| Existing U0/U1 | Mechanism donors and causal reference | HCCL geometry and integrated feature/context lineage |
| HCCL causal core | Smallest property-bearing continual dyad with exact two-layer action attribution | Autonomous option lifecycle, multistep rollout, ensemble uncertainty, exact communication attribution, physical-safety claims |
| Implemented L0 composition ladder | Atomic live-stack/context/planner ownership and recurring actor-owned `P` | Autonomous event generation, complete target transaction, benchmark life, authentication, physical safety, efficacy evidence |
| HCCL-v1 | Later all-enabled successor | Nothing may be claimed complete until its missing mechanisms and evaluators exist |
| Scale/external ladder | Longevity, more agents, Forager, and robot simulation | Promotion remains a separate fail-closed campaign process |

The causal core reuses rather than replaces:

- two independently learning `PrototypeAgent`s, prior-decision action timing,
  experiential memory, immutable same-prestate proposals, and the outer atomic
  transaction from the U0 hidden-Prototype life;
- factorized partner behavior and grounded joint-world modeling from U1;
- the hidden Markov sign, ordinary noisy cues, and oracle-free transition
  boundary from the hidden-partner lives;
- descriptor-aware consumer routing patterns from `IntegratedHiddenPartnerAgent`;
- the pre-outcome rescue and post-outcome `SequentialLineageCache` ordering from
  the sequential hidden-rule development life; and
- the bounded dynamics and pure `step_result` interface of
  `RecurringTwoAgentWorld`.

These donors do not already form the causal core. In particular, current U1
uses its four U0 proposal rows for planner-only attribution, consumes a stable
raw representation in its present wrapper, and performs one-step joint-cell
evaluation rather than four real backups.

## Stage 1: the HCCL causal core

### One environment and one uninterrupted life

The causal core is a cooperative dyad with two complete, independently
initialized learning agents. It does not concatenate unrelated tasks or reset a
learner between regimes. There is no scripted partner, terminal transition,
boundary callback, or learner-visible phase identity.

The physical substrate retains these `RecurringTwoAgentWorld` settings:

| Field | Value |
|---|---:|
| world limit | `1.0` |
| damping | `0.75` |
| acceleration | `0.15` |
| time delta | `1.0` |
| maximum speed | `0.25` |
| initial positions | `[-0.5, 0.5]` |

The visible meet/avoid context bits are destroyed. A fast hidden Markov sign
`z` and two ordinary noisy cues reuse the hidden-partner world-feedback values:

- `P(z flips) = 0.03` per committed transition;
- cue flip probabilities are `0.25` and `0.35`; and
- contextual outcome noise is `0.15`.

**Alberta development resolution (L0 world/event-receipt rung, 2026-08-03).**
`alberta_framework.streams.hccl_causal_core.HCCLCausalCoreWorld` now pins a
fair initial `z`, typed Threefry named streams, and fixed action-independent
draw counts. Each event draws one world flip, two next-cue flips, one outcome
flip, ten standard-normal nuisance values, and two standard-normal partner-
velocity observation errors. The world flip occurs after current scoring; the
cues target the next sign; and the outcome flip affects only convention factor
`P`, whose clean and noisy values remain separate. Nuisance index 11 has
exactly ten times the variance (a `sqrt(10)` standard-deviation multiplier)
only for an agent at `xi < -0.8L`. Noisy partner velocity uses a conservative
fixed standard deviation of `0.01` in physical-velocity units before
normalization. These are Alberta development choices, not source-paper claims
or an issued stochastic protocol.

The same module pins the physics, evaluator-only schedule mapping, observation
layout, immutable source-bound event receipt, pure proposal, and atomic
commit/rollback mechanics. The separate transaction adapter composes this
world with exactly one attribution-kernel state, binds the world/source clock,
event identity, and exact `B/M/P` receipt identities, evaluates the fixed eight
same-source proposals, and atomically selects only `PP`. Its composite stage
and prebound scan are host/eager-only; JIT remains supported at the smaller
world-proposal and attribution-kernel boundaries. Neither module implements a
learning agent or the integrated HCCL causal-core life, executes the 8,998-step
schedule, reserves or consumes a protocol seed, creates an artifact or writer,
sets a threshold, or grants evidence/promotion authority. This document still
does not authorize benchmark execution. The prebound scan only replays and
verifies a complete caller-supplied receipt trace; it does not generate events,
actions, or decisions or orchestrate an online life. An issued runbook must
explicitly adopt or version these development resolutions before any run
exists.

### Reward factors and typed signals

Let `L = 1`, `Vmax = 0.25`, positions be `x0, x1`, velocities be `v0, v1`,
and primitive actions be `a0, a1` in `{0, 1}`. Define:

```text
d   = |x0 - x1| / (2L)
b   = 0.6Lz
gi  = 0.2(1 - |xi - b| / 1.6L) + 0.8 1[|xi - b| <= 0.1L]
G   = 0.5(1 - d) + 0.25(g0 + g1)
V   = clip(1 - (|v0 - 0.8Vmax z| + |v1 - 0.8Vmax z|) / (3.6Vmax), 0, 1)
P   = 1[(2a0 - 1)(2a1 - 1) = z]
```

The evaluator-hidden slow regime selects one task score over the same dynamics:

| Regime | Task score | Role |
|---|---|---|
| A | `G` | recurring hidden-target rendezvous |
| B | `V` | recurring coordinated-velocity control |
| C | `0.5(G + V)` | new factor recombination, then recurrence |
| D | `P` | single-exposure convention; nonrecurring in this evaluator schedule |

The transaction never passes an ambiguous scalar called only `reward`. It
binds four typed signals:

- `task_score`: common environmental objective;
- `message_charge_i`: price of the message delivered to agent `i`, independent
  of the counterfactual action after the delivery receipt is fixed;
- `safety_cost_i`: soft, proposal-specific cost computed from the proposed
  effective action and same-prestate dynamics; and
- `net_reward_i = task_score - message_charge_i - safety_cost_i`.

The slow context bank and its lineage cache learn from `task_score`, so message
price or safety-cost drift cannot masquerade as a task regime. Control,
experiential memory, and the planner optimize `net_reward_i`. The eight Horde
heads are typed independently: task reward at discounts `0.5`, `0.9`, and
`0.99`; partner action; safety cost; TV occupancy; target-zone occupancy; and
neutral/unavailable option success. The hard action mask is a caller-owned
pre-action input, never a learned reward or a relaxable safety cost.

In the minimal causal core, communication is unavailable with zero price and
the soft safety cost is identically zero, so `net_reward_i = task_score` while
all four typed channels remain present. Nonzero message and safety channels
belong to later HCCL-v1 stages and require pinned functions before execution.

### Exact life geometry

The property-bearing life has 8,998 committed transitions. Entry and tail
metric windows are 64 steps. Segment lengths are deliberately nonperiodic.

| Occurrence | Regime | Half-open range | Steps | Evaluator role |
|---:|---|---:|---:|---|
| 0 | A | `[0, 769)` | 769 | initial acquisition |
| 1 | B | `[769, 1566)` | 797 | interference and fresh adaptation |
| 2 | A | `[1566, 2395)` | 829 | first recurrence |
| 3 | D | `[2395, 3252)` | 857 | single exposure; no recurrence in this evaluator schedule |
| 4 | A | `[3252, 4135)` | 883 | retention after D |
| 5 | C | `[4135, 5046)` | 911 | first factor-recombination transfer |
| 6 | A | `[5046, 5987)` | 941 | recurrence after C |
| 7 | B | `[5987, 6958)` | 971 | late plasticity and recurrence |
| 8 | C | `[6958, 7967)` | 1009 | transferred-rule recurrence |
| 9 | A | `[7967, 8998)` | 1031 | final retention |

Labels, boundaries, lengths, occurrence numbers, future schedule, and steps to
a boundary are evaluator-only. D never returns in this finite life, but the
agent cannot know that from the past. Metrics may report retrospective D
eviction regret; they must not say the learner knew that D was obsolete.

### Observation width and pair grammar

The external raw observation is exactly 16 scalars, in this order:

| Indices | Values |
|---|---|
| `0:4` | normalized own position, relative position, own velocity, noisy partner velocity |
| `4:6` | previous own and observed partner primitive actions |
| `6:8` | previous task score and previous own net reward |
| `8` | history-available bit |
| `9:11` | two noisy `z` cues |
| `11:16` | five nuisance channels; index 11 is the TV-sensitive nuisance |

At genesis, previous-action and previous-reward fields use neutral values and
the history bit is zero. The L0 world/event-receipt rung maps primitive ids
`0/1` to signed dynamics/history values `-1/+1`, uses the normalizations above,
and rejects nonfinite candidates. Those are development mechanics only; an
eventual runbook must explicitly adopt them and close full-run finite-value and
runtime bounds.

The causal core separates three learner-produced state surfaces:

- raw observation: 16 scalars;
- fast recurrent hidden-sign state: 4 scalars; and
- slow context responsibilities: 3 scalars for a three-slot context bank.

The fixed pair grammar is deliberately restricted to the 16 raw coordinates.
It therefore contains exactly `C(16, 2) = 120` unordered raw-pair candidates
and 12 active generated slots. Fast state and slow context are fixed-width
typed consumer inputs, not extra members of the pair universe. The complete
consumer representation is consequently `16 + 4 + 3 + 12 = 35` scalars. A
future design that admits state/context coordinates to the pair grammar must
change the declared width and candidate count; appending only the three context
coordinates would make them 19 and 171, not 16 and 120.

When an agent occupies the action-reachable TV region `xi < -0.8L`, the
variance of nuisance index 11 is multiplied by ten. This changes neither
dynamics nor any task score. It tests whether feature selection follows
grounded predictive/control utility rather than raw surprise.

Separate typed pre-action inputs carry the hard action mask. Communication is
not a third action replacement in the causal core; the later HCCL-v1 rules are
specified below.

After the action, each learner receives its own executed primitive action, the
completed partner action, task score, own net reward, safety cost, message
charge, scalar discount, next raw observation, and fixed Horde cumulants. Slow
context inference consumes only completed `(partner action, own action,
task_score)` and can affect only the next decision.

The learner never receives regime, phase, boundary, clock, future schedule,
hidden sign, noiseless cues, reliability mode, corruption bit, evaluator
counterfactuals, feature semantic labels, TV irrelevance, fault identity, or
fresh-reference starts.

### Fixed causal-core capacities

| Resource | Causal-core value |
|---|---:|
| slow context slots | 3 |
| sequential-lineage archive capacity | 1 |
| fast recurrent state | 4 scalars |
| raw observation / full consumer representation | 16 / 35 scalars |
| active generated features / raw-pair candidates | 12 / 120 |
| feature replacement cadence | global post-steps 64, 128, ... (140 opportunities) |
| experiential-memory rows | 64 per agent |
| Horde heads | 8 |
| partner behavior models | 1 per agent |
| grounded joint-world models | 1 per agent |
| planner joint cells | 4 per agent/event (`A^2`, with `A=2`) |
| multistep planner backups | 0 in the causal core |
| autonomous installed/candidate options | 0 in the causal core |
| uncertainty ensemble members | 1 in the causal core |

The four joint cells are one-step evaluations, not four real backups. Four
backups, 1/4/16-step rollout endpoints, two installed/four candidate options,
and a three-member uncertainty ensemble remain HCCL-v1 extensions.
The eighth, option-success Horde head is allocated with a neutral cumulant and
an unavailable bit in the causal core; it becomes live only with an issued
option-lifecycle stage.

## Exact action timing and attribution

### Persisted decision receipts

At the start of committed event `t`, state already contains three effective
primitive-action receipts for each agent, all prepared after transition `t-1`
for decision `t`:

- `B_t`: the discarded no-memory Prototype preview;
- `M_t`: the post-memory action; and
- `P_t`: the one-step planned action prepared from `M_t`.

Each receipt binds at least agent identity, decision clock, source-transition
clock, raw-observation identity, fast-state clock, slow-context birth,
per-feature birth ledger, memory generation, planner-model clock, hard-mask
generation, action before/after the hard mask, RNG receipt, and exogenous-event
receipt identity. Slot equality is never semantic identity.

Genesis prepares `B_0`, `M_0`, and `P_0` from the initial observation with the
history bit zero; it must not synthesize a fake completed transition.

### Immutable exogenous event receipt

Preparation for decision `t` also creates one immutable event receipt `X_t`
from a dedicated source-clock binding. It contains every ordinary stochastic
choice used by that decision and transition: message delivery/corruption when
enabled, contextual outcome noise, hidden-sign transition, next cues, and next
nuisance draws. All same-prestate proposals consume `X_t` without mutating it.
There are no action-dependent draw counts or rejection-driven key advances.

If any validation or child update fails, the complete source state, including
`X_t` and all B/M/P receipts, is returned bit-exactly. A retry evaluates the
same event receipt.

### Two adjacent dyad cubes

For agent actions written as joint pairs, every committed event stages:

```text
memory cube:   MM, B0M1, M0B1, BB
planner cube:  PP, M0P1, P0M1, MM
```

The two `MM` calls are deliberately duplicated and must return bit-exactly
equal transitions and typed signals. The transaction designates seven
counterfactual slots and can commit only `PP`; a successful event therefore
discards seven of eight proposals, while a rejected transaction discards all
eight. At most four effective joint actions and seven action-receipt vertices
occur in one transaction; action coincidences can reduce the former further.

The exact adjacent-layer contrasts are:

```text
memory total       = signal(MM) - signal(BB)
memory interaction = signal(MM) - signal(B0M1) - signal(M0B1) + signal(BB)

planner total       = signal(PP) - signal(MM)
planner interaction = signal(PP) - signal(M0P1) - signal(P0M1) + signal(MM)

total action-stack effect = signal(PP) - signal(BB)
                          = memory total + planner total
```

Each formula is evaluated separately for task score, net reward, and safety
cost. Action coincidences may make a contrast zero but do not remove calls.
These are immediate realized same-prestate effects under `X_t`, not long-run
off-policy returns.

Implementation note (L0 transaction only):
`alberta_framework.core.hccl_causal_attribution.HCCLCausalAttributionKernel`
now implements this isolated eight-proposal attribution transaction over
caller-supplied, source-bound exogenous and B/M/P receipts. It enforces the
fixed proposal order, duplicate-`MM` equality, typed task/net/safety/message
contrasts, `PP - BB` telescoping, PP-only commit, eager host preflight, traced
preflight limitations, and atomic rollback. The separate
`HCCLWorldAttributionAdapter` now feeds it the exact eight pure world proposals
from one immutable causal-core event and can persist only the PP world/kernel
pair. It owns no duplicate world state. Stale, tampered, cross-world,
cross-event, proposal-failed, or downstream-rejected transactions preserve the
bit-exact composite source for same-receipt retry. The adapter is explicitly a
host/eager orchestration boundary; compiling the full prebound scan approached
the operational memory cliff, so only the smaller donors claim JIT support.
These modules still do not implement either learning agent, execute a life,
issue a runbook, reserve seeds, write an artifact, set a threshold, or carry
evidence or promotion authority.

`HCCLExternalCoordinatorBaseBridge` now adds a separate base-only integration
rung without changing those donors. It owns exactly one adapter state and two
independently initialized external learned-state/router/audit coordinator
states, started from the world's two exact raw 16-channel observation rows.
Each exact cached primitive is bound as `B=M=P`; six deterministic receipt
identities cover the matching coordinator decision, lifecycle, and clocks, and
all layers share the same hard masks. A mask excluding a cached action rejects
before receipt construction rather than granting fallback authority. The PP
proposal gives each coordinator exactly one transition with its own executed
action, net reward, and next observation. HCCL and both coordinators adopt
all-or-none, and rejection returns the three complete sources bit-exactly.
Zero adjacent-layer contrasts are facts of this ablation only, not “no joy” or
actor-gradient conclusions. Host-only orchestration and strict in-memory
checkpoint/resources create no memory/planner, life, run, seed, output,
artifact, threshold, evidence, or promotion authority.

A separate v1 `HCCLLearnedMemoryFeedbackBridge` now consumes this transaction
for one bounded learned-memory feedback path. It owns exactly one adapter
state, one `LearnedExperientialMemoryControllerState`, and one fixed pending
binding. Preparation binds a controller-admitted categorical retrieval to the
exact HCCL source/decision and event, B/M receipt identities and contents,
common hard mask, selected agent, retrieved action, and routing result. The
unbound agent must have the same effective B and M action, so the selected
agent's immediate `memory_total.net_reward` is a legitimate unilateral
retrieval contrast. Masked or unrouted retrievals clear only through matching
controller no-learning settlement. World/attribution, controller, and binding
adopt atomically or roll back bit-exactly for retry. The orchestration and its
bounded scan are host/eager-only; this is memory utility. It computes no exact
actor-sample delight statistic and executes no actor backward. It runs no life
and creates no seed, output, artifact, threshold, evidence, or promotion
authority. Its
bounded prebound scan likewise only replays and verifies an already complete
caller-supplied receipt trace.

`HCCLTwoLiveMemoryBridge` adds the distinct two-live-adapter transaction. One
HCCL owner and exactly two live learned-memory owners bind existing pending
receipts as per-agent `B/M`; absent receipts use `B=M`, and `P=M` is an
explicit no-planner rung. The settled causal effects are exactly agent-0
`M0B1-BB` and agent-1 `B0M1-BB`; the memory interaction remains an audit fact
and is not broadcast as controller feedback. Each child advances from its own
executed `M` action and `PP` net reward/next observation, with next-event masks
installed only on atomic adoption. This host/eager L0 mechanism computes no
actor-sample delight statistic and performs no actor backward pass; it creates
no run, evidence, benefit, or promotion authority.

`HCCLContinualDyadTransaction` is the first all-at-once owner of the HCCL
world/attribution state, both live post-memory action stacks, both slow-context
states, and one paired factorized planner. Its split boundary completes both
agents through memory once and can then bind either the planner or an explicit
disabled-planner form without donor reevaluation. Every persistent owner adopts
together or the complete source returns. This is host/eager L0 machinery, not
an autonomous causal-core life, physical dispatch, benchmark result, evidence,
or promotion.

`HCCLKondoContinualDyadRoute` uses that disabled planner as a learning-only
shadow and makes the Kondo actor the recurring owner of `P`. `event0` installs
the first proposal and compact certificate without actor backward. Every
successor consumes the prior compact lineage through one actor transaction
before sampling and atomically installing the next pair. Delight is the exact
float32 statistic `advantage * selected-action surprisal`; a row sparks joy iff
its exact contribution enters an actor backward that actually executes. That
fact remains only in the nested actor result and survives a later outer veto,
even though persistent state rolls back. Actor input is the 23-wide post-memory
base, masks must be all true, and the live action stacks remain the only
Prototype owners. The route has no autonomous event source, checkpoint/resource
closure, authentication, dispatch, physical safety, evaluator, matched benefit,
evidence, or promotion authority.

The memory contrast is the effect of the prior decision's memory dispatch on
the current effective action. A memory query/update performed after outcome
`t` prepares `M_(t+1)` and cannot be credited for action `t`.

### Target complete event transaction order

A future completed causal-core event must preserve this causal order. The
current L0 route implements only a bounded subset of it:

1. Validate the complete source state, all B/M/P receipts, the immutable
   exogenous receipt, hard-mask generations, semantic births, and clocks.
2. Before proposing any outcome, snapshot the slow-context reward models and
   the H=2 rescue already present in source state.
3. Stage both four-call cubes from the same source and exogenous receipt,
   require the duplicate `MM` results to agree bit-exactly, and select the valid
   `PP` transition without committing it yet.
4. Settle proposal-specific task, net, safety, and message signals.
5. Update each three-slot context bank from completed actions and task score,
   using only that pre-outcome rescue; derive the exact context-birth event.
6. Propose each `SequentialLineageCache` update from the pre-update reward
   models and the post-context event. A new rescue may affect the next
   allocation at the earliest, never the allocation that created it.
7. Advance the fast recurrent state exactly once from learner-visible
   observation/action/outcome fields.
8. Update the source-bound Prototype/feature learners and produce two
   same-source candidates: the discarded no-memory preview `B_(t+1)` and the
   memory-sidecar candidate/action `M_(t+1)`.
9. Update partner behavior and grounded joint-world models on the executed
   transition under the source feature bank.
10. Route every feature consumer from the old bank to the applied post-update
   bank, then construct the next 35-scalar representation.
11. Evaluate all four one-step joint cells per agent under the same applied
   partner belief and prepare `P_(t+1)` from `M_(t+1)`.
12. Commit environment, agents, contexts, H=2 caches, birth ledgers, memory,
    behavior/world/planner models, receipts, exogenous state, and clocks in one
    outer transaction, or return the exact source.

The partner belief used across one focal agent's candidate own actions is
identical. Conditioning the partner distribution on an action that has not yet
been revealed would violate simultaneous-action causality.

## Semantic birth and feature-consumer routing

Each live feature slot has a fixed-width birth record containing at least its
descriptor, per-slot birth counter, source clock, parents, and lifecycle event.
The same descriptor reacquired under a later birth is a newborn, not a
survivor. A global generation count or descriptor equality alone is
insufficient.

The source representation updates the executed transition first. One atomic
route then maps the applied post-update bank into every generated-feature
consumer:

- OaK and all Horde prediction heads;
- the grounded world model, whose output coordinates remain fixed physical and
  typed signal coordinates;
- experiential-memory keys/values and exact feature re-encoding;
- partner `BehaviorModel` input columns;
- `GroundedJointWorldModel` input columns; and
- the planner/control readout.

An exact survivor carries or moves its learned column. A newborn destination
is zero initialized. A retired identity is scrubbed. Inactive destinations
remain neutral. The route and all consumer audits are inside the same outer
transaction as feature birth. No consumer may observe a new descriptor with a
stale prior birth's parameters.

If “authenticated birth” is meant literally as security authentication, an
unkeyed content hash is only an integrity binding and the future protocol must
add a keyed/signature authority. Otherwise the runbook should use the narrower
term “source-bound semantic birth.”

The H=2 cache transfers lineage/rescue information only. It does not transplant
an evicted context model's parameters and must not be reported as dormant model
storage.

## One-factor causal-core panel

The minimum diagnostic panel is one-factor-at-a-time, not a factorial search:

| Arm | Sole routed intervention |
|---|---|
| `full` | all causal-core mechanisms routed |
| `fast_state_unrouted` | recurrent state learns/advances; its consumer coordinates are neutral |
| `slow_context_unrouted` | context learns and births occur; fixed neutral responsibilities reach consumers |
| `lineage_rescue_unrouted` | H=2 sidecar/audits run; rescue dispatch is neutral |
| `feature_random_rank` | learned and random ranks are both computed; random rank selects curation transactions |
| `feature_consumers_unrouted` | lifecycle/route audits run; generated columns are neutral at all consumers |
| `memory_dispatch_unrouted` | memory query/write/rebind runs; `B` rather than `M` is dispatched |
| `uniform_partner_belief` | behavior prediction/learning runs; a uniform belief reaches planning |
| `planner_dispatch_unrouted` | cells/models update; `M` rather than `P` is dispatched |

All arms allocate the same declared persistent shapes, use paired exogenous key
roles, stage eight environment calls, preserve scheduled candidate/curation
opportunities, and compute the named routed/unrouted alternatives before
selection. Mechanism-owned RNGs advance under the same static rule.

This is **not** a claim of matched total work. Once actions diverge, endogenous
states, traces, values, branch-effective operations, compilation behavior,
latency, and trajectories can differ. Reports may claim matched shapes,
exogenous roles, proposal counts, scheduled opportunities, and explicitly
audited base logical calls only. They must report intervention-specific work
and must not relabel opportunity matching as equal FLOPs or equal wall time.

### Future-utility v2 gate

`compositional_future_utility_calibration_v2_development.py` does not authorize
a future-utility default for HCCL. Its sole attempt failed after the first arm's
compiled scan because an all-step margin diagnostic was incorrectly asserted
to be cadence-only; no arm record, endpoint, report, winner, or default exists,
and the root cannot be retried. Its learner and raw-6/depth-3/candidate-8
geometry also differ from the causal core, and its phase lengths rotate the
HCCL lengths even though the A/B/A/D/A/C/A/B/C/A order and 8,998 total are the
same. The unobserved immediate contextual-bandit structural endpoints would
not have been TD control, world modeling, or dyadic planning evidence anyway.

The causal core therefore uses current utility as a conservative pre-v2
reference, not as selection of the v2 current arm or of any v2 winner/default,
unless a later document prospectively names one development hypothesis. If a
future-utility contrast is added, every compared arm must either compute
current, future, normalization, age, uncertainty, and random-score paths and
route only the declared result, or disclose their intervention-specific work.
The failed existing v2 declaration cannot retroactively select the HCCL full
arm.

## Communication boundary

Communication is not an independent action-replacement layer in the causal
core. Learned partner behavior is sufficient to test hidden-partner inference
and multi-agent planning there.

HCCL-v1 may add one noniterative exchange after both agents prepare `M_t` and
before either prepares `P_t`: each agent exposes its `M_t` proposal and a
bounded confidence, the immutable receipt applies availability/corruption once,
and the delivered typed message becomes an input to the recipient's planner.
Both final actions are still fixed simultaneously. Message price is charged on
delivery in every matched message arm, including an input-unrouted arm.

Under that design, `PP-MM` is the effect of the entire planning stage,
including any delivered message. An arm comparison may estimate the long-run
value of routing the message, but the eight-call audit cannot separate the
same-event message contribution from other planner inputs.

If exact same-event communication attribution is required, HCCL-v1 must define
a third cached action level and a third adjacent dyad cube. Three layers require
12 proposal calls and at most 10 distinct action-receipt vertices per event;
the two-action dyad still has at most four effective joint action vectors. It
is invalid to claim separate memory, planner, and communication
effects from the two-layer eight-call budget.

Before communication becomes executable, its proposal-confidence function,
fusion rule, availability/corruption draw semantics, delivery timing, price
settlement, and neutral unavailable encoding must be pinned. Reliability mode,
the uncorrupted proposal, and the corruption bit remain evaluator-only.

## Per-event work and resource accounting

For the 8,998-step causal core, one arm has these selected named calls exactly:

| Item | Per event | Per life |
|---|---:|---:|
| environment proposal calls | 8 | 71,984 |
| designated counterfactual slots | 7 | 62,986 |
| discarded proposals after a successful PP commit | 7 | 62,986 |
| duplicate-`MM` equality checks | 1 | 8,998 |
| context updates | 2 | 17,996 |
| fast-state updates | 2 | 17,996 |
| sequential-lineage sidecar proposals | 2 | 17,996 |
| generated-feature/lifecycle routes | 2 | 17,996 |
| discarded no-memory Prototype candidates | 2 | 17,996 |
| memory-sidecar Prototype candidates | 2 | 17,996 |
| memory query/write transactions | 2 | 17,996 |
| behavior-model candidates | 2 | 17,996 |
| grounded-world-model candidates | 2 | 17,996 |
| planner decisions | 2 | 17,996 |
| one-step joint planner cells | 8 | 71,984 |
| real multistep backups | 0 | 0 |

“At most 7” is the distinct action-receipt-vertex bound, not permission to omit
the duplicate call or equality check. With two binary-action agents there are
at most four effective joint actions; coincident B/M/P actions can reduce that
number without changing the eight calls. A rejected transaction discards all
eight proposals rather than relabeling the seven designated counterfactual
slots.

Instantiating the existing H=2 cache formula at HCCL context geometry gives:

```text
34K + 141 + 8AD + 4KAD bytes per agent
```

For `K=3` context slots, `A=2` own/partner actions, and `D=2` partner one-hot
coordinates, that is 323 bytes per agent and 646 bytes per dyad. This is the
cache state only, not the context models or a whole-agent memory total.

An exact pair-memory re-encode over `F` generated coordinates and `M` memory
rows requires `2FM` pair products per agent when the current U0 adapter is
retained; `F=12, M=64` gives 1,536. Pair candidate enumeration is
`B(B-1)/2`, giving 120 cells at `B=16`.

Every future report must measure initialized, boundary, peak, and final
persistent bytes from the concrete state tree; transient trace bytes;
environment/model/route/sidecar calls; compilation/workspace exclusions; and
block-until-ready p50/p95/p99/max latency. Speculative component sums are not a
substitute for measured whole-state accounting.

Evaluator references require separate accounting. A fresh learner consuming
the acting trajectory is an off-policy diagnostic and cannot be called fresh
control return. A reference that acts needs a shadow environment, paired
exogenous receipts, and its own proposal and resource counts; it never affects
the primary learner.

## Causal-core endpoints

These are endpoint families, not thresholds or acceptance rules.

### Feature discovery

- per-slot descriptor and birth acquisition/admission lag;
- active/candidate occupancy, promotions, refreshes, cascade losses, and churn;
- exact survival versus same-descriptor new-birth reacquisition;
- evaluator-only A/B/C target-signature coexistence and D eviction;
- grounded feature contribution and action change versus random ranking; and
- feature-bank reachability plus exact consumer route/scrub audits.

The evaluator may label a target signature; the learner never receives that
label. A finite pair grammar is not evidence of general feature discovery.

### Remembering

- A/B/C context lineage, feature birth, memory provenance, and downstream
  consumer-column survival across intervening regimes;
- recurrence entry/tail task and prediction performance against the prior
  occurrence tail;
- H=2 opening, confirmation, rejection, transfer, and later target-change
  diagnostics; and
- helpful/harmful retrieval plus exact `MM-BB` action-layer effects.

### Forgetting and capacity reclamation

- D context/feature/memory occupancy and eviction latency;
- stale D occupancy, reclaimed capacity, and negative transfer after D;
- whether retiring D avoids later eviction of recurring A/B/C identities; and
- inactive/retired consumer scrubbing and absence of stale-birth reuse.

Because D's nonrecurrence is evaluator hindsight, these are retrospective
schedule-regret endpoints, not proof that the learner knows the future.

### Catastrophic forgetting and transfer

- phase-performance matrix, per-regime peak-to-later forgetting, and worst
  recurring-regime entry/tail gap;
- recovery length, recovery area, recurrence slope, and backward transfer;
- semantic survival separated from reacquisition; and
- first-C transfer versus an explicitly classified fresh/shallow reference,
  followed by C recurrence.

Lifetime averages must not hide failure on one recurring regime.

### Hidden state, partner inference, world model, and planning

- fast hidden-sign calibration against the evaluator-only Bayes reference,
  without feeding that reference to the learner;
- slow-context occupancy/birth accuracy as an evaluator diagnostic;
- partner-action NLL, Brier score, calibration, assigned probability on the
  completed action, and drift recovery versus uniform belief;
- prequential raw/reward/discount errors by recurrence and joint-action cell;
- planner action-change rate and exact `PP-MM`, `MM-BB`, and telescoping
  `PP-BB` benefit/harm/interactions; and
- long-run full versus uniform-belief, memory-unrouted, and planner-unrouted
  return, clearly separated from immediate cube effects.

The causal core supports one-step model/planning endpoints only. It cannot
report 4/16-step rollout, ensemble coverage, or option-planning results.

### Continuing operation and scale

- committed transitions, zero resets/boundary callbacks, exact nested clocks,
  rejection/retry counts, and retry parity;
- initial/peak/final/boundary bytes and named calls;
- proposals per commit, candidate cells, route/re-encode work, H=2 work, and
  latency quantiles; and
- reward, retention, and recovery per byte, call, and measured time.

## HCCL-v1 extensions

HCCL-v1 remains the later all-enabled successor. It may add, one bounded stage
at a time:

1. planner-input communication and its input-unrouted price-matched arm;
2. autonomous proposal/install/use/retire option lifecycle with primitive
   fallback;
3. three-member uncertainty models and genuine fixed-budget multistep backups;
4. explicit exploration and optional candidate-update interventions;
5. composite checkpoint/suffix parity; and
6. synthetic envelope faults and rollback mechanics, still without claiming
   physical safety.

Only after those mechanisms exist may HCCL-v1 restore endpoints for skill
lifecycle, 1/4/16-step rollout, ensemble coverage, exact message value,
checkpointing, and synthetic safety faults. An all-true action mask is not a
safety certificate.

Its target deltas from the causal core remain:

| Resource | Causal core | Later HCCL-v1 target |
|---|---:|---:|
| installed / modeled option candidates | `0 / 0` | `2 / 4` |
| uncertainty ensemble members | `1` | `3` |
| genuine planner backups | `0` | `4` per agent/event |
| planner rollout endpoints | one-step only | `1 / 4 / 16` steps |
| communication | unavailable, neutral typed input | planner input, or an explicitly budgeted third action layer |
| checkpoint/fault campaign | absent | composite suffix parity and synthetic rollback audits |

The full 18-row ambition is staged as follows. “Core” means only that the
causal core can produce a bounded development diagnostic, not that a property
is accepted.

| Row | Property | Earliest meaningful stage |
|---:|---|---|
| 1 | continuing operation | Core |
| 2 | temporal/resource bounds | Core for measured software bounds; external for hardware deadlines |
| 3 | plasticity | Core |
| 4 | retention/catastrophic forgetting | Core |
| 5 | within-family transfer | Core; external for broad transfer |
| 6 | state construction | Core for hidden-sign/context separation; Forager/robot for grounding |
| 7 | typed continual prediction | Core |
| 8 | world model | Core one-step; HCCL-v1 for 4/16-step and ensemble coverage |
| 9 | planning | Core one-step primitives; HCCL-v1 for backups/options |
| 10 | exploration | HCCL-v1 |
| 11 | feature lifecycle | Core raw-pair grammar only |
| 12 | skill lifecycle | HCCL-v1 |
| 13 | optional candidate-update audit | HCCL-v1 |
| 14 | experiential memory | Core |
| 15 | communication/IA | HCCL-v1 |
| 16 | checkpointing | HCCL-v1 |
| 17 | synthetic safety/fault mechanics | HCCL-v1; external for physical safety |
| 18 | reproducibility | issued campaign and independent reproduction, not an environment outcome |

If the later checkpoint/fault stage retains the earlier design, composite
snapshots occur after committed steps 2048, 4777, and 8192; caller-owned
fallback attempts occur at 1024, 4096, and 7168; and stale/NaN rollback attempts
occur before committed steps 3072 and 6144. A failed attempt must advance no
state and retry the same event receipt. These positions remain inactive design
values until an issued stage pins the exact fault payloads and snapshot schema.

The earlier communication-drift schedule remains a possible HCCL-v1 design,
not part of the causal core:

| Half-open range | Flip probability | Availability | Price |
|---:|---:|---:|---:|
| `[0, 1103)` | `0.05` | `1.00` | `0.00` |
| `[1103, 2249)` | `0.30` | `1.00` | `0.01` |
| `[2249, 3473)` | `0.90` | `1.00` | `0.01` |
| `[3473, 4651)` | `0.05` | `0.25` | `0.02` |
| `[4651, 5879)` | `0.05` | `1.00` | `0.02` |
| `[5879, 7069)` | `0.05` | `1.00` | `0.20` |
| `[7069, 8219)` | `0.90` | `1.00` | `0.20` |
| `[8219, 8998)` | `0.05` | `1.00` | `0.05` |

No part of this table defines a runnable communication protocol until the
missing semantics listed above are pinned.

## Mechanics, longevity, and external ladder

| Rung | Geometry | Purpose |
|---|---|---|
| Smoke | ten lengths `(33,35,37,39,41,43,45,47,49,51)` | CI mechanics only; too short for retention or scientific claims |
| Core-L1 | canonical 8,998-step dyad | first property-bearing causal-core development life |
| Core-L2 | 8 cycles, 71,984 steps | longevity; D is replaced by A after cycle 0 |
| Core-L3 | 112 cycles, 1,007,776 steps | million-step bounded-state stability; D remains nonrecurring |
| Scale-L4 | Core-L1 geometry with 4 learning agents on a fixed-degree ring | multi-agent scaling without changing the task family |
| HCCL-v1 | staged extensions on validated core mechanics | all proposed synthetic mechanisms, still development-only |
| External | unchanged policy interface in Forager, then robot simulation | grounding required beyond the toy |

Smoke success cannot select a scientific setting. Longevity runs cannot reuse
development or consumed evidence roots for promotion. This document reserves
none of those roots.

## Scaling and impossibility bounds

For `N` agents, exact all-agent interaction attribution for one before/after
action layer requires `2^N` proposals. For `L` adjacent action layers it
requires `L * 2^N` calls and has at most `1 + L(2^N - 1)` distinct
action-receipt vertices, while effective joint action vectors remain bounded by
the action domain. The dyad/two-layer case is 8 calls, at most 7 receipt
vertices, and at most 4 effective joint actions; a dyad with three independently
attributed layers uses 12 calls and at most 10 receipt vertices.

At larger `N`, a runbook must predeclare one bounded alternative:

- `N+1` actual-plus-unilateral direct-effect calls per layer;
- a bounded `O(N^2)` pairwise audit; or
- a fixed randomized-coalition budget.

None recovers all higher-order interactions, and reports must say which were
omitted.

A factorized planner with action count `A` and learned neighbor degree `d`
requires `A^(d+1)` joint cells per agent. A fixed-degree ring can keep planner
work linear in `N`; a fully connected partner set is exponential. Raw-pair
feature enumeration is quadratic in base width. Higher-order exhaustive
feature grammars are combinatorial and require a fixed sparse/dovetail proposal
budget.

No finite fixed-capacity agent can losslessly retain arbitrarily many
independent tasks. Three live context models plus a one-entry lineage archive
cannot preserve four arbitrary reward models because the archive does not copy
parameters. Likewise, finite feature and memory banks imply a retention/regret
tradeoff rather than a universal no-forgetting guarantee.

There is also an identical-history limit: after the same observed past, one
possible future may recur B while another recurs D. No past-only deterministic
eviction rule can always choose correctly in both futures. A future protocol
must declare a recurrence prior/hazard, randomized or minimax policy, or regret
objective. H=2 lineage evidence supports only a local predictive rescue claim,
not guaranteed long-horizon recurrence value.

Same-prestate proposal cubes identify immediate action effects under one shared
exogenous receipt. They do not identify the long-run return of an unexecuted
policy. One-factor life arms are still required for trajectory-level effects.

## Honesty boundary

A completed causal core could mechanically test continuing operation, bounded
resources, plasticity, recurrence retention, within-family transfer, raw-pair
feature selection, memory dispatch, hidden-partner prediction, and one-step
planning. The current L0 route does not yet establish that suite. Even a
completed core could not by itself establish:

- general feature discovery outside a finite product grammar;
- absolute absence of catastrophic forgetting;
- correct eviction under every possible future;
- multistep/option planning or ensemble uncertainty;
- broad held-out task-family transfer;
- real-time control-hardware deadlines;
- physical geometry, dynamics, or safety;
- population/statistical reliability from one deterministic life;
- clean-checkout independent reproduction; or
- Alberta Plan completion.

Any future result remains development evidence until an issued frozen
protocol, untouched held-out roots, versioned artifacts, a strict validator,
complete source/runtime closure, external benchmark ladder, and the
repository's fail-closed promotion process establish a narrower claim.
