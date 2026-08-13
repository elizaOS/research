# Alberta Plan research status

This document separates implemented mechanisms from scientific evidence. The
Alberta Plan is a research programme, not a conformance specification, so the
thresholds below are project criteria chosen to make completion falsifiable.

The current verdict is **in progress**. This repository contains substantial
online-learning machinery and broad unit coverage, but it does not yet contain
a fail-closed, end-to-end demonstration of one bounded agent learning through
an uninterrupted nonstationary life while retaining recurring critical
knowledge.

Registry snapshot (2026-08-02): `alberta-evidence-status` reports **all five
registered claims `invalid`** (overall `invalid`, exit `2`). The recurring
pair-feature, scale-robust pair-feature, and two-agent coadaptation claims
fail current-registered-source validation, and the FTL decision-fidelity and
intelligence-amplification historical compatibility chains fail, because
registered source files were edited after the artifacts were pinned. The
pinned artifacts remain immutable historical records; none certifies this
working tree. The strict current validators also expose narrower reconstruction
failures rather than stopping at the first error: the FTL chain now observes
drift in its fidelity and CLI sources in addition to the artifact builder, and
the IA chain reports current canonical controller-budget mismatches for every
stored condition. Those are additional invalidity diagnostics, not new
scientific outcomes. This is the fail-closed design working as intended:
renewal requires rerunning each frozen protocol to a new artifact path and
schema version with untouched preregistered seeds, never a consumed-seed
replay.

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
| 2 | Generate nonlinear features, estimate future utility, and replace them under a fixed budget without losing recurring useful structure | Two immutable narrow L2 pair-feature artifacts, now source-invalidated against the evolved current learner, plus current-source generated-AST schedule/twin/scrub/expanded-lineage infrastructure, an exact 64-bit compositional lifetime clock, defaults-off novelty/ancestor credit, and a bounded product dovetail. In the 8,998-step silent-task life, all raw pairs and A/B/C/D triples are structurally reachable. Headroom makes A/B/C admissible and raises consumed-root greedy reward from `-0.01045` to `0.12247`; exact base/tail birth identity now crosses real old-bank linear OaK/Horde updates, routed physical-model learning, and exact feature-bound memory in one opt-in v18 Prototype transaction. A frozen contribution-future-utility comparison passes its exact contracts but the two enabled endpoints lose every A/B/C target and trail its disabled internal comparator by `-0.277395` and `-0.294732` executed reward | Partial and currently falsified at selective retention. Headroom still loses every A/B/C lifetime and retains obsolete `p12`; left-packing removes `p12` but never co-retains two A/B/C features, ends with none, and drops reward to `0.03979`; the tested one-step and decay-0.95 contribution utilities are rejected without selection or retuning. The atomic Prototype route is deterministic, linear, L0, and planning-default-off; it establishes composition rather than held-out selective retention or an end-to-end bounded-agent benefit |
| 3 | Learn many continuing, possibly off-policy GVFs with history and feature finding | Horde, TD/GTD, traces, history, and working-memory component tests | Partial |
| 4 | Progress from bandits through contextual and sequential actor-critic control with feature finding | SARSA, discrete/continuous average-reward actor-critic, a nonlinear shared-trunk discounted action-importance core, and a separate nonlinear discrete differential core have L0 mechanisms plus small control/diagnostic tests. A strict development lane now gives the nonlinear differential core and DifferentialSARSA full prequential traces, checkpoint/replay, and descriptive RiverSwim A/B/A diagnostics under common key roles | Partial; successor timing, parameterization, bytes, and realized scalar work are explicitly unmatched, nonlinear importance correction remains action-only, and no matched-resource retention/control exit result exists |
| 5 | Learn both differential average-reward GVFs and conventional value-plus-expected-duration predictions | Differential TD/GTD/Horde tests plus a deterministic online two-head option return/duration diagnostic | Partial; the defining mechanism reaches L1, but no promoted comparison or integrated option-control result exists |
| 6 | Reproducible continuing control suite, including RiverSwim, access control, Jellybean, GARNET, and continuing conversions | Closed-loop micro-MDPs, a pinned Foragax protocol runner, a completed five-seed field-of-view tuning stage, an unsealed four-seed RTU-RTRL/DQN development comparison, and a qualified matched-current campaign whose open-tuning cells are entirely unexecuted; the old selected 30-seed evaluation produced no batch or report | Partial; the named suite and a completed paper-length admissible paired Alberta comparison are not present |
| 7 | Validate incremental average-reward planning, then function approximation and adaptive features | Legacy bounded Dyna, an isolated real-anchor ensemble one-step kernel, proposal-only policy/uncertainty short rollouts with terminal-correct returns, a strict local planner→grounded-gauge→actor/critic composition, and an eight-seed development RiverSwim diagnostic | Partial; the new composition removes caller rollout-tensor substitution but its model support, real anchors, regions, and safety facts remain unauthenticated attestations. It is not Prototype- or dispatch-integrated, is not externally calibrated, and has no frozen held-out artifact or planning-benefit result |
| 8 | Close the perception → model → feature ranking → feature replacement → model feedback loop | Historical held-out decision-fidelity acceptance for a lifetime-statistics transition model (immutable artifact; its compatibility chain no longer validates against the current tree), online-gated, dense full-GRU, and compressed-RTU RTRL builders, a standalone causal whole-unit RTU generate-and-test lifecycle, transactional Prototype adapters for the smaller GVF/inverse and caller-targeted comprehensive objective sets, a learner-owned causal target producer/Prototype transaction, strict-linear RTU replacement consumers for both comprehensive target authorities, and one opt-in v18 Prototype composition sharing a pair bank across linear OaK, an ordered linear Horde, a generated-input/fixed-physical-output world model, and exact feature-bound memory. A fixed-budget one-step frozen-consumer deletion/separate-shadow-insertion auditor remains a mutually exclusive ranking lane | Partial; the narrow FTL comparison reached L2 under its frozen source only, while the current learned-state objectives/builders/lifecycle, atomic pair/model/memory composition, auditor, and rank adapter are L0. Objective masses, learner-derived target quality, and optional-cumulant semantics are uncalibrated, recurrent changing-parameter sensitivity is approximate, live RTU replacement excludes the general nonlinear envelope and has no matched outcome, atomic planning is default-off, and partner modeling, utility-curation feedback, selective-retention benefit, and outcome evidence remain open |
| 9 | Improve exploration and planning order under matched real-transition and backup budgets | Prioritized, surprise, utility, guarded-dream mechanisms, a prospective-improvement selector, a consumed causal six-arm noisy-TV/delayed-opportunity development lane with independent per-arm histories and matched logical budgets, planning diagnostics, a four-mode calibrated primitive/option controller, a raw-representation Prototype/STOMP sidecar, and a separate default-off v2 policy consumer that can replace the cached primitive under a caller mask and arm the actual credit owner | Partial; causal score production, online model/calibration integration, and an owner-correct live action edge exist, but the exploration lane is eight-event, synthetic, consumed, and `not_assessed`; the v2 policy has no safety or physical-dispatch authority, no matched benefit result, and no held-out or promoted search-control evidence |
| 10 | Discover reward-respecting subtasks, learn options and option models, and consume those models in planning | A fixed-universe v1 gate emits exact-budget four-family proposals plus frozen-random and exact-hand comparators. A bounded scheduler observes discovery every accepted transition, requests a fresh bundle at exact cadence/retry, receipt-gates quiescent installation into preallocated STOMP slots, reevaluates cumulants, and emits authority-free maintenance handoffs. One controller executes an exactly authorized two-rebind retirement; a separate one-canonical-state transaction can prepare and, under new exact caller authority, replace the resulting cold slot once. An opt-in one-owner bridge composes that metadata with the sole live Prototype→OaK→STOMP controller, reuses the exact raw control trace, carries cold masks through search/Dyna/audit, and finalizes the sole post-control owner without reevaluating STOMP | Partial; all mechanics and the bridge remain L0, host-orchestrated, and `not_assessed`. Go/no-go authority remains external, no proposal is queued, only one retirement/replacement is supported, and there is no repeated-lifecycle, matched-benefit, held-out, or defining outcome evidence |
| 11 | Track causal utility and safely replace features, subtasks, options, and models; compose an option keyboard | OaK transition ownership, active-option-safe curation, keyboard mechanics, a pair-feature auditor/rank adapter, and a bounded option-lifecycle maintenance report whose live STOMP observer has explicit zero control/curation authority | Partial; exhaustion or rejected attribution correctly freezes only the observer, but maintenance remains proposal-only, existing ages/cadence/confirmation/proxy/safety retain go/no-go authority, automatic keyboard consumption is absent, and causal lifecycle outcome evidence is missing |
| 12 | Measurably increase another learning agent's capability in a closed interaction loop | The frozen v1 IA run is a historical valid rejection: reward uplift and both augmentation controls passed, while action-changing intervention prevalence missed its threshold; a separate exact-321-byte development kernel composes two learning roles with online behavior/world prediction, one-step planning, causal channel/freeze controls, and trace-reconstructed phase diagnostics. Its nonexecuting 11-condition paired-scan declaration has a matched runner and authenticated replay. A separate 12-event Prototype/fusion v2 lane starts all three interventions bit-identically and pairs exogenous schedules while preserving arm-owned causal trajectories | Partial; the historical v1 threshold still fails and current-source compatibility is invalid. The development runner remains `not_assessed` and unexecuted under its strict permit. In the consumed v2 lane, learned feedback and fixed-zero produce different internal state but no behavioral separation. There is no calibration, held-out partner-benefit result, artifact, threshold, evidence, or promotion authority |

No row currently satisfies the completion rule.

The new `complete_prototype_manifest` contract separately enumerates all 18
final acceptance properties and their exact evidence roles. Artifact bytes,
the common prototype-configuration digest, exact evidence role, frozen
protocol/scientific-outcome digests, and validator source closures are pinned;
validators must return accepted
scientific L3 receipts with untouched held-out seeds.
Optional paper delight and Kondo paths add mandatory actor-learning or measured
compute guardrail roles. The contract distinguishes `not-ready` from
`invalid`, rejects path/source/artifact/relabel tampering, and has no default
evidence bindings. It also reconstructs every row, aggregate flag, overall
status, and its own digest before returning an exit code. It therefore makes
the current absence of a completion
certificate executable without expanding or weakening the narrow five-claim
registry.

### Embodied hard-envelope status

`EmbodiedSafetyEnvelope` now provides a public, deterministic L0 command
filter. It independently gates measured and proposed joint position, velocity,
torque, workspace, and collision clearance; bridge and emergency-stop state;
telemetry identity/freshness; control deadlines; and exact model, optimizer,
lifecycle, and configuration bindings. A safe proposal passes, an unsafe
proposal can use only the statically configured in-envelope fallback, and an
unsafe current state or failed transaction yields no available action.
Untrusted reward, partner metadata, and learned cost have no override path.

Emergency stop assertion has an independent checksummed transition and cannot
be suppressed by command replay, decision-capacity exhaustion, or invalid
optional metadata. Reset requires a fresh nonce, the declared non-secret
authority token after external caller authentication, and a stationary-safe
sample whose telemetry ID and sample tick are both strictly newer than the stop
sample. Authority-bound rollback preserves counters and the latest exact
action/version ledger while suspending deployment. Restore requires an exact
revision and SHA-256 state digest retained outside the checkpoint payload,
which rejects an older snapshot that would erase a latch or consumed nonce.

Pure shadow evaluation writes to a fixed recent ring; its proposal-only
readout requires zero retained hard violations plus Wilson success-LCB,
calibration, latency, and sample-count gates. The kernel dispatches nothing and
has no learning, deployment, promotion, or physical-safety authority. Its
fallback is not a geometry/dynamics proof, token equality is not
authentication, trusted-anchor storage and caller authentication are external,
and no physical-canary outcome exists.

`PrototypeEmbodiedCommandAdapter` now closes the next identity and settlement
edge from the consolidated semantic-memory Prototype owner into that exact
envelope. A fixed finite bank maps each primitive to one unique float32 command;
the pending receipt binds the complete Prototype decision/mask, envelope
source, telemetry, clocks, versions, and metadata. Settlement recomputes and
bit-compares the envelope result before reverse-mapping an accepted or
certified-fallback command to exactly one mask-admitted primitive and atomically
settling the semantic credit owner. An exact no-action or stop-only outcome
instead adopts the envelope rejection ledger or fresh emergency-stop latch,
closes only the attempt receipt, and preserves the same semantic owner for a
fresh-identity retry. The bank is not a geometry certificate, and the adapter
has no hardware-dispatch, caller-authentication, safety, learning, evidence,
deployment, or promotion authority. It is L0 mechanism coverage, not a WP9
exit result.

`PrototypeEmbodiedDevelopmentHarness` now adds the bounded plant/successor
edge around that adapter. It owns exactly one semantic adapter, one
deterministic fixed-capacity plant, and one grounded shadow state. Settlement
recomputes the envelope, proposes the uniquely mapped actual plant action,
settles the dispatch owner, and applies exactly one real semantic Prototype
transition from the plant reward and post-observation before adopting all
three descendants atomically. Two accepted cycles run without reinitializing;
Prototype and OaK clocks advance `0 → 1 → 2` and the armed raw observation
remains equal to the plant. Certified fallback credits the executed primitive;
no-action is envelope-only and retryable; stop-only preserves semantic, plant,
and shadow state; shadow mismatch records proposal without commit and rolls
the whole harness back. Exhausting plant capacity is a scheduling halt after
an ordinary learned transition, not a fabricated termination or truncation.
Strict checkpoints and fixed call/state accounting are tested. Shadow tags are
unkeyed integrity sentinels, however, and the harness has no physical dispatch,
geometry/safety proof, caller authentication, delight/KondoGate-intent/
KondoSparseActor-backward assessment, efficacy, evidence, deployment, or
promotion authority. It
remains L0 `not_assessed`.

A strict paired development benchmark now wraps two independently owned copies
of that harness. The `adaptive_stomp` and
`zero_stomp_step_size_control` configurations may differ only in the five
declared STOMP optimizer step sizes. Every other initial semantic array, RNG,
cache, observation, trace, and clock must be dtype/shape/typed-key-
implementation/host-byte exact after normalizing the materialized base-LMS
step-size leaves. Starts require empty harness pending/last-commit records, an
unset adapter settlement ledger, zero Prototype/OaK/adapter/plant clocks, and
sufficient remaining capacity. V1 fixes four continuing attempts without reset or task ID and the
bridge disconnect at attempt 1, so three attempts commit real plant/Prototype
transitions and one emits exact unavailable action/reward sentinels per arm.
The selected-source/runtime-bound report records raw availability, fallbacks,
plant state, clocks, rearming, adopted updates, shadow work, and exact logical/
resource budgets. Its named normalized lifetime AUCs use committed-transition
or attempt indices and are descriptive, not adaptation or post-change AUC.
Sixteen fast synthetic pytest contracts cover typed-key identity, exact
sentinels, signed-zero identity, live drift, exact replay, externally supplied
prefix reconstruction, and content-plus-resealed tampering. The real slow lane
runs outside pytest through
`alberta-prototype-embodied-paired-development`, emits JSON to stdout, and
writes no artifact. The scoped zero-step witness excludes decay-only option-
model EMAs, semantic memory, shadow learning, traces, and caches; no whole-
agent freeze is claimed. Every result is L0 and `not_assessed`, with no winner,
threshold, physical dispatch, efficacy/safety result, delight/KondoGate-intent/
KondoSparseActor-backward assessment, semantic use of historical `GradientJoy`
compatibility names, evidence, deployment, or promotion authority.

A strict development-only synthetic fault audit now runs one fixed 30-event
continuing schedule through the envelope. It covers observation/wear drift,
stale/deadline faults, delayed untrusted reward metadata, NaN/out-of-bounds/
failed sensors, bridge loss/recovery, unsafe commands, emergency stop, reset,
rollback, and checkpoint recovery. Only envelope-available commands count as
executed in simulation accounting; physical dispatch is zero. Raw causal
records, hard interventions, exact fallbacks, shadow/readiness facts,
whose success input is only an action-availability proxy, action-availability
recovery delays, resource accounting, eager/JIT/scan
parity, externally anchored checkpoint resume, and exact replay are retained.

This is a synthetic telemetry/command schedule rather than a learned dynamics
simulator or geometry proof. The learner/controller is an unchanged opaque
witness, so learner adaptation and reset-free recovery are not measured.
Caller authentication remains external, the declared held-out family is not
executed, the unmatched no-candidate arm is omitted, and there are no seeds,
thresholds, artifacts, efficacy/safety verdicts, or deployment authority. The
lane stays `not_assessed`; WP9's dynamics-simulation, held-out adaptation, and
physical exit gates remain open.

A second strict development lane now supplies the previously missing minimal
dynamics/learner edge without closing those gates. Adaptive and zero-learning
`PrototypeAgent` arms own independent policies and bounded two-joint plants;
only typed exogenous dynamics/sensor/fault/latency inputs are paired. The
frozen arm has equal state capacity and all 12 envelope/update opportunities;
this trace has seven available update calls and five exact no-command/no-
transition skips in each arm. Every fallback that changes the primitive action
is rebound through the public Prototype credit-owner API before simulation.
The trace shows changed adaptive parameters, unchanged frozen action/model/
utility parameters, and independent trajectories, but defines no benefit or
recovery threshold. Its separately declared change family is already consumed
development data and is never held out or promotable. Full composite resume,
source/runtime-bound replay, and dynamics eager/JIT/scan parity are diagnostic
contracts only; physical dispatch, artifacts, efficacy/safety assessment, and
deployment/promotion authority remain zero.

The lifetime audit now extends beyond the original compositional learner.
Exact big-endian `uint32[2]` words identify up to `2^64 - 1` committed events
(about 584.6 million years at 1 kHz), while legacy int32 fields remain
saturating telemetry. The main Step 1--4 production pipeline, normalizers,
actor-critic and off-policy learners, multi-head learner, dense and latent
world-model paths, rollout-level dreaming, `FastSlowLearner`, the foundational
`LinearLearner`, `MLPLearner`, `TDLinearLearner`, and `TrueOnlineTDLearner` families,
Prototype's outer transaction, and several integrated state/feature/option/
memory/Horde/partner paths now use exact clocks. So do `LearningPartnerWorld`,
`PartialObservationWrapper`, both adaptive-opponent streams, and hardened
matrix, recurring-multi-agent, gauntlet, feature-discovery,
hidden-partner-mapping, and hidden-partner-feedback paths.
The adversarial-pursuit stream additionally owner-authenticates its two-phase
emit/resolve transaction. These paths fail closed on invalid source/candidate
state, preserve atomic rollback, reject ambiguous legacy counters, derive
schedules without saturated telemetry, and treat an all-ones clock as
terminal rather than wrapping.

This is a finite lifetime contract, not mathematical infinity and not yet a
whole-repository guarantee. Early gauntlet and out-of-class compatibility
surfaces, auxiliary resource managers, learner variants outside the four
hardened foundational families, and some diagnostic/query counters still have
narrower integer or float horizons. Fixed buffers and estimator counters also
impose explicit capacity limits, and legacy host birth/uptime timing floats are
outside the bit-exact JAX learning-state contract. Those surfaces remain audit
items before an indefinite-agent claim is possible.

### Current compositional-control falsification

The current-source development lane now separates proposal reachability,
topological capacity, and selective retention instead of treating feature
presence as one property. One exact 8,998-step silent-task contextual-bandit
life supplies only six raw Rademacher channels and realized action reward. Its
task-agnostic product dovetail reaches all 15 raw pairs and naturally forms the
A/B/C/D depth-three targets without task IDs, boundaries, or injected
expressions.

On consumed root 0, novelty plus ancestor credit alone still admits only D:
greedy reward is `-0.0104468`, useful intermediate `p45` is lost nine times,
and obsolete `p12` occupies the bank for 95.733% of the life. Reserving
topological headroom fixes that specific admission obstruction: A/B/C acquire
6/10/7 lifetimes, `p45` presence rises to 59.029%, and greedy reward rises to
`0.122472`. It does not solve memory: every A/B/C lifetime and all five `p45`
lifetimes are lost, all three targets are absent at the end, and the obsolete
`p12` born at step 192 is never retired.

A defaults-off left-pack sibling changes only destination placement. It moves
intermediates to slot 6 and targets to slot 7, but A/B/C simultaneous-presence
counts are `[8166, 832, 0, 0]` for zero/one/two/three targets. A/B/C suffer
4/9/6 acquisitions and losses, `p45` and `p12` suffer 12/12 and 13/13, every
target is absent at every phase exit and life end, and greedy reward falls to
`0.0397866`. Removing the obsolete feature is therefore not evidence of
remembering the recurring ones. Headroom and left-packing expose opposite
failure modes and falsify slot availability or placement as a sufficient
selective-retention policy. This one consumed development root has no threshold,
held-out inference, artifact, promotion authority, or general feature-finding
claim.

The first frozen three-arm contribution-future-utility comparison then
rejected both enabled endpoints on this consumed life: its disabled internal
comparator retained A and scored `0.274283`, while mix-one/decay-zero and
mix-one/decay-0.95 retained no A/B/C and scored `-0.003112` and `-0.020449`.
No endpoint was selected or retuned. A separate v2 froze a new development
root and five-arm mechanism panel spanning current utility, future utility, a
half mix, uncertainty/age normalization, and a longer float32 horizon. Its
pre-run evaluator/declaration contracts passed, but the sole authorized
attempt failed after the first arm's compiled scan and before any arm record
returned. `decision_margin_passed` is an all-step diagnostic; the endpoint
extractor incorrectly required every raw pass to coincide with a 32-step
curation opportunity and raised `a strict-margin pass occurred outside a due
curation event`. The selected sources remained valid, but no report, hash,
arm endpoint, winner/default, threshold, artifact, evidence, or promotion
result exists. The root and retry budget are consumed. The current evaluator
now rejects before source-array construction, and its historical declaration
is intentionally source-invalid. Eighteen evaluator, six decommissioned-
declaration, and three immutable-outcome contracts pass. A failing-test-first
synthetic trace now separates all-step margin diagnostics from due-opportunity
endpoint counts, but that mechanism repair cannot revive or reinterpret the
consumed v2 root. A future newly issued root must also close the declaration
self-hash/import-path/integrity gaps before execution; it cannot reconstruct or
retry v2.

The independently namespaced v3 successor closed those execution-governance
gaps with a fresh root, exact 19-module import closure, durable one-shot ledger,
and strict summary observability. Its sole root `0x12EFD48B` completed all five
8,998-step scans, then failed closed at report serialization. The gate required
direct selected-candidate admissions to be at least all structural acquisition
episodes, but cascade refills can create additional absent-to-present episodes;
the producer's actual invariant is direct admissions no greater than structural
episodes. The durable terminal records `panel_completed=true`,
`report_sha256=null`, and failure receipt
`5150a4aa08ba3d17b644ae1ed0357d1c6359123e96298ac3ca2cd0d03bff894d`.
No validated arm record, endpoint, report, winner, default, threshold, evidence,
or promotion result exists, and no retry or recovery is authorized. V3 source
bytes and ledger remain the immutable failed-attempt record. A future v4 must
use a new schema, namespace, and root with an explicit direct/cascade acquisition
episode partition; it cannot pass the v3 in-memory result through a repaired
gate or otherwise reconstruct it.

The frozen-theta `CompositionalFeatureAdapter` separately closes a deployment
mechanic. It emits `[exact stable base | generated tail]`, binds the full bank
and per-slot birth words, and bit-authenticates bounded row re-encoding. Its new
source-bound prepare/commit boundary defers adoption until consumers report
ready; commit recomputes the proposal, rejects stale/tampered payloads, and
advances persistent state/RNG/clocks at most once while disclosing two logical
learner-update evaluations. Every JAX-authoritative learning/binding leaf is
bit-authenticated; legacy host birth/uptime floats retain the project-wide
non-bit-exact outer-JIT boundary. Eager, outer-JIT, scan, tamper, stale-retry,
exact-byte, and structural-row tests pass. This standalone frozen-theta adapter
itself has no Prototype consumer; the separate v18 owner below consumes related
seams. Theta revision identity remains frozen, and this mechanism changes no
outcome.

An isolated `CompositionalConsumerRouter` now carries that identity into one
exact linear OaK state and an optional ordered linear Horde state. It keeps the
valid stable source separate from a caller-attested one-step post-update
candidate, requires exact adapter/OaK/STOMP/base/Horde clock parity, checks the
source cache, recomputes the candidate cache, and routes by slot-birth words
rather than equal descriptors. Changed births scrub every affected base/Horde
weight and trace, option-policy column, option-model input/output axis, and
option-start cache while all survivor bits and LMS scalar optimizer state are
preserved. Commit recomputes both proposal and route and rejects the joint pair
atomically.

The learner/adapter also accepts one dynamic `curation_allowed` bit. False
consumes a due fixed or learned-resource cadence opportunity without any
proposal, promotion, refresh, cascade, rebound, or regeneration, while ordinary
learning, candidate aging, RNG, clocks, phase/accumulator, and the learned
resource manager advance once. This prevents an unsafe option boundary from
replaying the same structural proposal forever, provided the live caller derives
the safety bit before preparing the update. The router does not derive it.
A separate development integration now uses actual public `OaKAgent` and Horde
updates under the old descriptor bank. Its first option transition advances
both consumers while false permission consumes the due event without a birth;
the next option termination returns to a primitive boundary and commits one
birth with mask `[F,F,T,F]`. All five adapter/OaK/STOMP/base/Horde clocks reach
`[0,2]`; routed survivor bits equal the real post-update states, changed axes
are zeroed, scalar optimizer state is preserved, and the next cache equals the
candidate-bank representation. This closes the synthetic-consumer caveat only
for a deterministic two-transition linear transaction. It remains outside
Prototype and establishes no feature, prediction, or control benefit.

## Pinned narrow results

Each result below was produced once under a frozen protocol and is pinned
immutably. As of 2026-08-01, none of them validates against the current
working tree (see the registry snapshot above); they are historical evidence
for their frozen sources.

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

### Narrow Prototype pair-feature lifecycle, WP7.1b auditor, and WP7.1c ranking

An opt-in `PrototypeFeatureLifecycle` supplies a deliberately restricted L0
path from a fixed-width base representation through bounded pair products.
The original mode trains the pair bank from one exact owner-bound behavior TD
target: base-Q while idle or the current intra-option target while an option
executes. The shared mode adds an ordered linear Horde. Its task vector places
that control target first and then uses the Horde update's TD targets in the
declared demon order. Linear OaK and Horde both update under the old descriptor
bank before the lifecycle observes those targets. At a safe idle/cached-base
boundary, one accepted descriptor mutation routes all post-update OaK/Horde
feature axes atomically in exactly two router calls; otherwise curation is
deferred by restoring the pre-curation proposal state rather than queuing the
mutation.

The proxy is scale-normalized and group-balanced: control receives weight
`0.5`, while each of `D` demons receives `0.5/D`, so duplicating Horde heads
does not increase the group's aggregate weight. This is shadow prediction
utility, not a causal intervention on downstream return, planning, or model
quality. Any builder gradient through the augmented representation is pulled
back against the exact pre-route generation and full descriptor bank.
Feature identity, linear OaK, ordered Horde semantics, and the consumer binding
are coupled by a schema digest in one enabled-only state bundle, which requires
the v4 Prototype checkpoint. Once the lifecycle observation ceiling is
exhausted, its audited transaction is a no-op while the already-updated,
step-aligned consumers remain untouched.

A separate opt-in `PrototypeFeatureUtilityAuditor` observes the same ordered
control-plus-Horde event without controlling curation. It forms active scores
from the old descriptor bank's frozen, predict-before-update consumer targets,
predictions, and tail weights. Each intervention is the exact normalized
one-step half-squared-loss increase from deleting that active contribution. A
separate matched shadow-candidate cohort uses private normalized-LMS
contributions to score insertion loss reduction before updating its shadow
weights, utility EMAs, or scale moments; the two cohorts are not compared for
routing. Control retains fixed mass `0.5`, each of `D` ordered demons retains
`0.5/D`, and unavailable task mass is not reassigned.

After a committed two-call consumer route, the auditor explicitly rebinds its
private state by descriptor identity without invoking the router again. New or
colliding candidate identities start at zero. The fixed-size state caps
observations and fails invalid transactions atomically under eager, JIT, and
scan. Enabling the auditor nests it with the existing OaK/Horde/binding bundle
under an ordered digest in one atomic state bundle and requires a v5 Prototype
checkpoint. Disabling it leaves the v4 bundle and behavior unchanged.

WP7.1c adds an opt-in stateless ranking adapter over the post-observation audit
EMAs. Its feature-gradient utility is deletion/insertion sensitivity, not
paper-defined actor-sample delight: it neither scores actor samples nor selects
backward passes. The adapter gives lower deletion
utility priority only within active slots and higher insertion utility priority
only within candidates. It never compares an active score with a candidate
score. Each rank-eligible slot must meet the exact evidence floor on every
configured task; fixed control/Horde task mass is not reassigned or
renormalized.

The rank adapter has influence, not curation authority. Active and candidate
ages, maintenance cadence, candidate confirmation, the learner's internal
proxy promotion floor and margin, and the safe routing boundary retain all
promotion and go/no-go authority. The adapter owns no persistent state and
adds no RNG draw, backward pass, consumer update, or router call. Its exact v6
Prototype checkpoint shell binds the ranking configuration and digest around
the v5 bundle, while disabling it leaves v5 behavior unchanged.

The lifecycle has explicit allocation ceilings, exact accounting for its owned
state and routed consumer axes, and versioned config/checkpoint contracts; that
declaration is not a total Prototype/OaK/Horde footprint. A standalone lifecycle checkpoint owns
neither consumer: callers must persist the returned binding atomically with
their OaK and, when managed, Horde checkpoints. The shared Prototype mode
requires exact linear OaK, an exact linear Horde with scalar LMS optimizer
state and no normalizer, and rejects the nonlinear/adaptive variants. Its
generated pair tail remains outside every world-model input in that v4-v6
utility/auditor configuration. One narrow
development-only model lane now composes an exact
`PrototypeFeatureLifecycleConfig`, exact `IdentityStateBuilderConfig`, and the
legacy `ActionConditionedWorldModel`, whose configuration must be the exact
`ActionConditionedWorldModelConfig` type. The model, recent-observation buffer,
and action-interaction columns consume only the stable base prefix, so an
accepted descriptor curation does not route or reset model coordinates. The
v17 wrapper binds the model state to a SHA-256 digest of the complete serialized
Prototype configuration; state validation also binds every LMS optimizer
scalar to the configured step size and derives exact buffer occupancy, ring
index, and unused-row zeroing from the observation lifetime. A rejected feature
transaction or refused model update rolls back the model, buffer, and complete
feature-lane Prototype event atomically. The historical direct-world path
without feature lifecycle retains its best-effort model semantics. The focused
nine-test mechanism suite passed in 42.36 seconds, and the broader Prototype
horizon suite passed 17/17 in 25.72 seconds. These are L0, nonpromoting
mechanism checks. Dreaming, replay, ensemble and recurrent world models, IA,
partner fusion, and GRU perception remain unsupported; no model quality,
planning benefit, retention, artifact, evidence, or promotion follows.

An isolated successor now crosses the previously missing generated-input edge
without making generated identities into output targets. It learns a linear
model from `[stable base | live pair products | primitive-action onehot]` to
fixed normalized base-delta/reward/discount heads. A pre-outcome snapshot binds
the complete source world/router, base/action, augmented input, and prediction;
consume revalidates the authoritative current world/router, learns under the
old bank, stores the physical successor, and then routes input weights and
traces. Stable/action columns and descriptor survivors stay bit-exact, while
newborn and inactive columns are positive zero.

The same full-snapshot rule covers a defaults-off one-backup planner: complete
world, router, OaK, anchor, action, and proposal content is revalidated before
the physical predicted successor is re-augmented under the live bank and only
OaK's base learner is carried. A learned survivor-column ablation changes both
the physical proposal and actual OaK backup. Same-clock alternate histories,
stale caches, cold generations, invalid routes/candidates, and disabled
planning roll back. The mechanism has fixed state and history-independent work:
for `N` generated slots and `H=B+2` physical heads, dynamic inputs add `8HN`
bytes; real prepare/consume uses `2N` pair products, three forwards (prepare,
consume authentication, and learner update), one backward, and at most one
route, while planning uses `4N` pair products, two forwards, and at most one
OaK update/base backup. It remains outside Prototype, memory,
checkpoints, partner modeling, and utility curation when used through its
standalone API. The opt-in v18 `PrototypeAgent` mode now coordinates that same
routed world state with the one authoritative pair lifecycle, linear OaK,
ordered linear Horde, and exact feature-bound memory. Lifecycle, world, and
memory readiness are conjunctive; an external veto retains all valid ordinary
old-bank updates, while the lifecycle-cap case locally derives only the exact
current encoding and leaves lifecycle/adoption state unchanged. The v18
checkpoint and resource record contain one lifecycle/router, one OaK, one
Horde, one world/buffer, and one memory owner; mirrored bindings are caches,
not authorities. Planning remains default-off. There is still no calibrated
error/uncertainty, utility-curation or partner-model composition, retention or
control benefit, artifact, evidence, promotion, or default authority.

Both that routed world model and the pair lifecycle now expose a common
external-readiness shape without introducing a second consumer router. A
source-bound preparation computes the ordinary old-bank successor and routed
destination from exactly one learner evaluation. A content-bound readiness
receipt then selects the routed destination or retains the ordinary successor;
adoption evaluates neither learner nor router. External veto is recorded as a
rollback rather than deferral, and the world-model ordinary update remains
available even when its destination route is internally invalid. Source,
consumer, receipt, or preparation mismatch is an exact state no-op. The
receipt is an unkeyed integrity binding whose logical transient byte record
counts serialized PyTree leaf occurrences, not physical allocator residency.
These common seams remain independently usable coordination mechanics. The
v18 atomic Prototype/model/memory/checkpoint owner described above consumes
them; utility-auditor/curation composition is still open.

A separate narrow development-only exception composes `ExperientialMemory`
when the base state builder is the exact `IdentityStateBuilderConfig`.
`PrototypeFeatureMemory` binds the memory rows to the live feature descriptor/
generation identity and atomically re-encodes every valid observation, key,
and outcome row before the same transition's query and write when a descriptor-
generation successor commits. Failed or corrupt rebinding rolls back the whole
Prototype transition. The adapter has fixed persistent bytes and a history-
independent worst-case rebind bound of `capacity` rows and
`2 * active_pair_slots * capacity` pair products, with zero memory-clock
advance and zero RNG draws; the v16 Prototype checkpoint binds its exact
composition digest. Learned base builders remain unsupported because their
stored prefixes cannot yet be reconstructed safely across feature-bank drift.
The diagnostic auditor and adapters close bounded mechanics, not the utility-
to-curation-to-outcome loop required by the Plan. The tests establish no
retention benefit, adapted-consumer deletion, realized return, planning,
safety, control, or empirical benefit, automatic cumulant/subtask or option
discovery, scientific evidence or promotion, WP7 completion, Alberta Plan
completion, or L3 integration claim. No registered artifact is renewed; any
artifact with differing registered source hashes remains invalid.

### Standalone WP7.2 v1 proposal-only cumulant/subtask discovery

`CumulantSubtaskDiscovery` owns a configuration-fixed universe across four
source families: controllable events, feature changes, reward-relevant
transition atoms, and typed prediction bottlenecks. The two-phase transaction
arms after action selection and freezes every predict-before-update probe and
reward/model insertion prediction. `observe` accepts only the exact transition,
semantic generation, source, canonical universe, and state revision. Its
successor semantics are cached for the next arm. A reward-transition atom born
on outcome `t -> t+1` therefore cannot collect evidence from that outcome; its
earliest evidence is on `t+1 -> t+2`.

Learnability, controllability, novelty, and contribution are independent,
noncompensating gates. Learnability compares the probe with a running baseline
instead of accepting raw surprise. Controllability uses only declared
randomized, valid-propensity action evidence and requires each action's floor.
Novelty is required against every incumbent and each earlier selected
proposal. Contribution is a frozen pre-update insertion audit over reward and
model tasks; every channel retains its configured mass and evidence floor, so
missing mass is never reassigned. A bottleneck proposal additionally requires
typed epistemic and progress floors and passes a persistent running-mean
aleatoric veto.

Every family owns a fixed positive quota and the quotas sum exactly to the
option budget `B`. They are never reassigned; failure to fill one family
suppresses the complete discovered bundle. The comparison cohorts are a
random projection bank sampled once at initialization and an identity-bound
hand-authored list of exactly `B` descriptors. All three cohorts therefore use
the same exact budget and compact appended tail slots
`raw_feature_dim ... raw_feature_dim + B - 1`; candidate identifiers are not
STOMP feature indices.

Strict v1 config/checkpoint schemas, source/semantic/transition/revision
bindings, projection and payload tamper checks, static allocation ceilings,
and exact persistent-resource accounting bound the mechanism. This is separate
from WP7.1c: it invokes neither Kondo nor delight and declares no backward
passes. It also performs zero OaK, STOMP, Prototype, or Horde mutation and has
no curation, promotion, go/no-go, or scientific-promotion authority. The
integration test supplies the three matched bundles to fresh, identically
configured STOMP agents for one finite update each; discovery itself does not
install, train, or own those consumers. A separate opt-in
`CumulantOptionInstallation` composition now accepts a complete fresh bundle,
binds its descriptors into preallocated live STOMP slots, rematerializes their
cumulants on each observation, and masks every cold slot across behavior,
learning, skip diagnostics, and planning. Its public quiescent lifecycle
rebind preserves identical slots and fully resets changed ones. Deferred work
is not queued and requires later fresh re-proposal; installation capacity does
not freeze a valid installed controller. Its empty STOMP template may opt into
a reserved observation suffix without moving the compact option-cumulant
indices. Standalone materializations fill those later cells with exact zero
and reject nonzero tamper; the zero-suffix path is unchanged, while an external
owner may bind the reserved cells to its own exact semantics.

A separate bounded scheduler now drives discovery observation and fresh
proposal cadence, requires strictly increasing caller authority receipts at
quiescent install boundaries, advances its key on every applied install, and
emits only authority-free retirement handoffs.

`AuthorizedOptionRetirementController` now turns one exact handoff into a
possible retirement only when a distinct caller receipt authorizes the exact
slots, owners, revisions, generations, validity window, and two independent
reset keys. It recomputes noncompensating support, randomized primitive-margin,
reliability, model-error, planning-use, and redundancy facts from the live
audit. Two public lifecycle rebinds scrub the complete approved slot and
restore its installer semantic identity before an authoritative persistent
mask leaves it cold across selection, bootstrap, planning, and attribution.
Active options, stale authority, either reset failure, and capacity exhaustion
are atomic no-ops; no replacement is queued or installed.

`AuthorizedOptionReplacementController` now supplies the next narrow edge
without giving the scheduler or audit autonomous authority. It persists one
canonical scheduler/installer subtree while projecting the retirement
controller's mask and control updates from that owner. After exactly one cold
slot exists, `prepare` observes the scheduler with installation authority
denied. The ordinary discovery/incumbent successor is independently usable;
the fresh one-slot bundle remains only in a transient preparation. Host
`commit` reruns that preparation from its source and exact inputs, then either
uses a new caller receipt to install/reactivate the one slot or commits only
the ordinary successor and retry marker. Active-option changes, stale/replayed
receipts, freshness and capacity exhaustion, forged prepared payloads, and
partial replacement all fail closed. The wrapper supports one retirement and
one replacement; its checksums and receipts are unkeyed integrity declarations,
not caller authentication or cryptographic lineage.

The separate opt-in `PrototypeOptionAuthorityBridge` closes the bounded live
owner edge without duplicating STOMP. Its persistent state contains one
`PrototypeAgentState`; nested Prototype→OaK→STOMP is the sole `STOMPState`
owner, while authority and lifecycle state are detached metadata with one
borrowed binding and no persistent preparation. Unequal pristine owners can be
reconciled only by an explicit directional receipt bound to both exact source
states and typed owner digests. Re-evaluation against the same unchanged pair
is idempotent. The receipt, state checksums, and checkpoint hashes are unkeyed
integrity bindings and do not authenticate a caller.

Each ordinary bridge transition forwards the historical candidate-audit
compatibility argument, experiential-memory input, and partner-policy input
and feedback without reinterpretation. One installed-slot mask reaches
Prototype start/update, real OaK/STOMP behavior and bootstrap, internal option
planning, option search, guarded Dyna, and lifecycle planning-use attribution.
The lifecycle consumes the exact raw `STOMPUpdateResult` without reevaluation;
a transient five-stage trace classifies option search, feature-axis routing,
Dyna, memory dispatch, and partner dispatch before metadata-only finalization
binds the sole final owner. An invalid or coherently resealed source cannot
commit and its transient candidate is primitives-only. Well-typed dynamic
audit refusal never rolls back valid Prototype control: authority metadata is
retained and desynchronization is latched. Diagnostics separately count real,
imagined, total, option-search-update, and internal-planning work; adoption,
retirement, and replacement evaluate STOMP zero additional times.

A separate versioned repeated bridge preserves that same sole nested
Prototype→OaK→STOMP owner while keeping only detached cycle metadata. Its
ordinary transition delegates once and consumes the already-evaluated raw
STOMP result; caller-authorized retirement/replacement atomically rebinds the
exact OaK owner. Array validation has eager/JIT/scan coverage, while full
retirement and replacement provenance replay is host-only.

The calibration-consumed nonwriting v1 development schedule is blocked rather
than successful. Cycle 0 completes in two replacement attempts, but cycle 1
exhausts the scheduler-derived cap of eight candidate-refresh attempts. Every
proposal is due and ready but reselects the same `(1, 3, 4, 5)` cohort;
`changed_slots` is all false, so the exact-one-cold semantic-change gate rejects
the repeated incumbent before any new scheduler installation attempt. The typed
fail-closed regression binds those facts, the valid unchanged source state,
completed cycle 0, no report/winner/benefit, and checkpoint suffix
`not_assessed`. The routing-disabled arm and suffix replay are not reached, so
there is no opportunity-matched comparator or parity result. The consumed
failure is recorded in `NEGATIVE_RESULTS_LEDGER.md`; changing the v1 schedule,
rotating seeds, injecting state, or raising the bound is not authorized.
The opt-in `AuthorizedOptionAtomicSwapController` v2 adapter prevents a
transient retirement from persisting its cold slot unless the same transaction
also adopts a fresh exact replacement. No-fresh, decline, tamper, and replay
are exact all-installed source no-ops. This narrows the persistence failure
surface but does not change discovery or create, rotate, or widen the candidate
universe; the repeated-incumbent/no-fresh-semantic cause remains. The consumed
v1 harness has not been rerun or fixed.

The stateless v2 `FreshColdSlotCumulantCohortFilter` now demonstrates the
missing candidate-selection step without owning adoption. It independently
revalidates the incumbent family/quota/uniqueness layout, holds every live slot
fixed, and chooses only a locally eligible, pair-novel, semantically fresh
candidate from the cold slot's family. The original six-candidate universe is
still unavailable; an explicit seven-candidate fixture with one additional
feature descriptor yields the exact one-slot replacement. Tamper and a
checksum-valid cross-family splice fail closed. The filter itself owns no
state, RNG, install, authority receipt, or commit path. A separate opt-in v2
`AuthorizedFreshColdSlotAtomicSwapController` now consumes its exact prepared
type: outer commit rederives retirement, the ordinary v1 preparation, the
filter source/output, and additive public scheduler/replacement adoption
preparations. Only an exact one-slot replacement can move the persistent state
from all-installed to all-installed; the one-cold retirement is transient, and
no-fresh, decline, outer veto, replay, stale identity/revision/key, or
checksum-valid payload/target tamper is the exact outer-source no-op. A full
commit rederivation executes six scheduler observations, two filter
derivations, two retirement rebind evaluations, and three deterministic
installation-candidate evaluations while adopting at most one installation.
The wrapper creates no RNG roots or splits; child work uses only four supplied
caller keys. All receipts/checksums remain unkeyed integrity declarations, not
authentication. This is mechanism coverage only, and the v1 negative remains
neither rerun nor fixed.

This is scheduled L0 mechanism coverage with externally authorized retirement
and replacement plus a bounded L0 live-owner composition. It establishes no
empirical benefit, caller authentication, autonomous go/no-go or repeated
lifecycle policy, physical-dispatch or safety authority, WP7 exit, evidence
promotion, SOTA, Alberta Plan completion, or L3 result. It renews no registered
artifact; all five live registry claims remain source-invalid.

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
artifact-builder hash. A consumed-seed replay on seeds 30–59 once validated
and matched the historical artifact exactly after excluding only operational
metadata, that builder hash, and its digest derivative; the replay was
nonpromoting, and subsequent registered-source drift means the live registry
now reports this compatibility chain `invalid` for the current tree. Because
the exact historical artifact-builder source was not archived, even a passing
replay establishes deterministic scientific compatibility, not complete
historical source recoverability.

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
The live registry now marks this claim `invalid` because registered source
hashes have drifted from the frozen run; the pinned artifact remains the
immutable historical record.
The Wilson lower bound for 30/30 observed recoveries is only 0.886, so the run
does not establish a population recovery probability of 0.95. More
fundamentally, the controller is a tiny contextual bandit with visible regime
cues and separate value cells. This is narrow coadaptation and retention
evidence, not autonomous feature discovery, task inference, IA, or an L3
continual agent.

A separate in-memory development probe removes the visible rule from both
learning agents. Each independent SARSA controller acts first; only afterward
does its own two-slot `ContextInference` bank consume own action, observed
partner action, and common reward for the next decision. The complete fixed
two-seed development panel now reports recurrent-early improvements of
`+0.0546875` and `+0.1171875` over a state- and update-matched unrouted
inference control. On both seeds, both learners formed distinct rule slots,
reused them with `0.99` recurrent agreement, and switched four steps after
each hidden rule change. Recurrent-tail rewards were `0.94140625` and
`0.953125`; overall rewards were `0.9300` and `0.94208336`. The unmatched
visible-rule ceiling retained recurrent-early advantages of `0.1015625` and
`0.05859375`. The joint environment, two controllers, and two context banks
use 408 persistent JAX-array bytes, no replay, and exact two-word clocks; the
deterministic schedule consumes no environment randomness. This is still a
development feasibility result only: capacity exactly equals the two-rule
support, tail reward does not uniformly beat the unrouted arm, and there is no
noise, eviction pressure, threshold, artifact, causal population estimate, or
promotion authority.

A U0 successor now composes that hidden inference timing with two complete
`PrototypeAgent`s. It destroys the visible rule coordinates, fixes both actions
before either context update, and permits the post-outcome inferred one-hot to
enter only the next decision. Each event stages four same-prestate joint-action
environment proposals, two context candidates, two discarded no-memory
Prototype previews, and two memory-sidecar candidates from one source, then
uses one outer all-or-none transaction for the environment and both learners.
Prototype, OaK, STOMP, Horde, stable-base world-model, pair-lifecycle, and
memory clocks stay aligned.

On the consumed 512/512/512 root, inferred routing records A1/B/A2 mean-agent
reward `0.991401/0.977935/0.964493`, compared with
`0.984638/0.0149503/0.994810` when the same context banks update but their
outputs remain unrouted. The routed A2 early window is `0.781546` and its tail
is `0.997949`; both banks switch twice and end in slot zero in both arms. The
routed memory sidecar records 800 beneficial, 61 harmful, and 2,211 neutral
current-event unilateral counterfactuals for the prior decision's memory
dispatch, versus 186/365/2,521 unrouted. They are not effects of the
query/write performed after that outcome and do not establish retained-memory
benefit. Both arms commit all 1,536 events with identical 41,718-byte
persistent state and logical work. This is a
large B-phase effect in one consumed U0 life, not a selected or population
result. The recorded run immediately predates the final validation-only
contract gates. A first exact replay completed both arms and constructed its
partial comparison, but a caller-side formatter error discarded the in-memory
result. The sole predeclared recovery again reproduced every declared report
field (`mismatches=[]`), yet `prototype_agent.py` changed after its clean
preflight and before comparison. Its fail-closed conclusion is therefore
`source-manifest-mismatch`, not current-source compatibility; the recovery
budget is consumed, no report or artifact was retained, and full-report
identity is not claimed. The initial slot-zero prior is routed, the two-slot
capacity exactly matches the two rules, causal trajectories diverge, partner
action is absent from each world model, and there is no composite checkpoint,
strict report validator, writer, threshold, artifact, evidence, promotion,
general feature-finding, selective-forgetting, or Alberta Plan completion
authority.

A second consumed-root development lane introduces actual capacity pressure:
two independent differential-SARSA controllers and two three-slot context
banks act for 4,000 uninterrupted steps across four hidden conventions. Exact
per-agent `(namespace, birth_words)` identities prevent recyclable slot
indices—or coincident birth words across agents—from masquerading as semantic
continuity. The predeclared root-0 epsilon grid remains unselected. At
`epsilon=.2`, both agents ended with distinct recurring A/B/C births and phase-
tail rewards from `0.65625` to `0.8125`; `.4` kept that identity structure but
lowered overall reward from `.70625` to `.516`. Lower exploration produced
semantic reacquisition or aliasing. A post-audit matched intervention now
authenticates every newly allocated birth and zeros exactly its stale SARSA Q
and trace column before next-action scoring. It removed every observed cross-
birth consumption, with descriptive root-0 reward changes `+.01025`,
`+.01200`, `+.00100`, and `+.00425` across the four epsilons. No epsilon or
intervention was selected as a default.

The same evaluator makes the eviction limit executable rather than rhetorical.
The actual and counterfactual schedules have a byte-identical history through
first C admission; only their future B-versus-D recurrence differs, which
flips the unique zero-recurrence-loss eviction. A deterministic policy using
only that common past therefore cannot guarantee the correct first eviction
on both futures. “Knowing what to forget” must mean minimizing expected or
worst-case future regret under an explicit prior, learned recurrence signal,
randomized/minimax objective, or bounded retained archive—not hindsight
omniscience. The scrub fixes semantic parameter contamination but supplies no
such prediction. Both results are consumed, deterministic, threshold-free L0
development diagnostics with no untouched-root panel, artifact, population
estimate, or promotion authority.

The first strictly past-only selective-retention intervention now makes that
boundary operational. Each live context birth owns an authenticated exact
entry count; `entries - 1` protects births that have actually completed prior
recurrence intervals. The API is defaults-off and affects only a valid
full-bank fresh allocation, while stored reuse, free-slot allocation, and the
matched controller scrub remain unchanged. Its no-signal arm is bit-exact with
the scrub baseline. On consumed root 0, `epsilon=.05` changed two agent
evictions at step 3216, avoided eight completed recurrence intervals, reduced
per-agent churn from 25 to 24, and changed overall reward from `.86700004` to
`.86950004`; `.1`, `.2`, and `.4` were exact nulls. The earliest capacity
decision is unchanged at every epsilon because both prefix-twin alternatives
have zero prior recurrence. This is evidence that authenticated past recurrence
can sometimes support selective retention, not a future-value solution: the
score resets on semantic rebirth, has no realized loss target, and cannot solve
cold-start indistinguishability.

A matched one-record cross-birth predictive-rescue sidecar tests the smallest
archive repair without reading future recurrence. Each agent retains one
source-victim reward model and exact lineage/rescue identity; a full-bank birth
may inherit it only when the just-observed transition makes the cache strictly
better than a fresh prior and every live source model. Live priorities are
fixed before that outcome, ties/non-finite values abstain, and both conditions
perform identical cache, scrub, work, and RNG transactions. The result is a
clean null across `.05/.1/.2/.4`: zero strict matches, rescue increments,
eviction changes, or reward deltas. At `.05`, four cache-valid tests all failed
against the fresh prior and all had an exact tie; the other epsilons produced no
cache-valid test. Thus one transition did not disambiguate a prior convention
on this dyad. The next repair must prespecify bounded sequential evidence or an
explicit prior, not weaken this gate after observing the null.

That bounded repair now exists as a fixed-`H=2` `SequentialLineageCache` and a
matched hidden-rule dyad wrapper, but its 4,000-step outcome has not been run.
At every host event the core evaluates one archived victim, a fixed fresh
prior, and every live source reward model from the exact pre-update bank. A
full-bank birth can only open quarantine; the next completed transition must
preserve pairwise never-worse support and provide strict support at least once
against every eligible comparator before the exact surviving birth inherits
lineage and one rescue count. The wrapper snapshots the reward banks and live
rescue before proposing the outcome, dispatches either exact zero or that
past-only rescue into the prioritized context update, then proposes both
sidecars, both authenticated controller scrubs, both controller updates, and
one whole-dyad all-or-none commit. Both conditions execute the same calls and
RNG advance. The opening archive and complete comparator bank are frozen, all
parameters remain untouched, and any confirmation can affect only later
eviction protection. A 32-byte configuration token plus a 32-byte canonical
state-content SHA-256 token fail closed on stale mutation. For the hidden-rule
geometry (`K=3`, four actions, four observation coordinates), the fixed
sidecar is 563 bytes per agent and 1,126 bytes for the pair; the measured base
carry is 962 bytes and the composite is 2,088 bytes. Forty-nine core tests and
10 focused wrapper tests cover quarantine, causal event binding, overlap,
saturation, tamper, digest parity, scrub preparation, whole-dyad rollback,
resources, and work. This is L0 mechanism/integration completion only: the
policy-agnostic core cannot authenticate its host transition, its unkeyed
digest is integrity rather than external provenance, and the full matched
no-signal-versus-`H=2` panel remains unexecuted.

The first HCCL causal-core transaction subset is implemented and integration-
tested, but it is not a completed causal core or a result assembled by naming
the U0, U1, and `H=2` donors together. `HCCLCausalCoreWorld` owns one fixed
development world state, immutable named-stream event receipts, pure proposals,
and atomic rollback. `HCCLWorldAttributionAdapter` owns exactly one such world
state and one attribution-kernel state. Given an already-prepared event and
exact source-bound `B/M/P` action receipts, it evaluates
`MM/B0M1/M0B1/BB/PP/M0P1/P0M1/MM`, requires bit-exact duplicate `MM`, preserves
typed task/net/safety/message contrasts, and can adopt only `PP` once. Clock,
event, and action-receipt identity mismatches, proposal failures, or downstream
rejection preserve the full source for same-receipt retry. Composite stage and
prebound scan execution are deliberately host/eager-only after the compiled
scan approached the operational memory cliff; the smaller donor kernels retain
their JIT boundaries. The prebound scan only replays and verifies a complete
caller-supplied receipt trace; it does not orchestrate an online life or
generate events, actions, or decisions. That first world/attribution adapter
contains neither learning stack nor fast/slow context and generated-feature
lineage integration. A separate newer primitive-only factory/runner can
construct a fixed 420- or 8,998-event life over the integrated dyad, but neither
life has been executed here as a research result and the runner has no partial
resume or checkpoint. Communication remains neutral/unavailable and belongs to
later HCCL-v1. No issued runbook, protocol seed, result, threshold, artifact/
writer, evidence, or promotion authority exists.

`HCCLExternalCoordinatorBaseBridge` now implements a separate base-only
three-owner rung: one world/attribution state and two independently keyed
`ExternalLearnedStateRouterAuditCoordinatorState` values. Each coordinator is
started from its own exact raw 16-channel world row. Its cached primitive,
complete decision/lifecycle identity, and clocks deterministically bind six
distinct B/M/P receipt identities under one common hard-mask matrix; a mask
that excludes either cached action is rejected without fallback authority.
Because `B=M=P`, the exact zero memory/planner contrasts are explicitly scoped
to this ablation. The PP proposal updates each coordinator exactly once from
its own action, net reward, and next observation, and all three owners adopt
only together. Public child-applied facts are outer-commit gated while nested
donor results retain attempted-stage diagnostics. This is not a delight or
“no joy” judgment and performs no actor backward pass. The composite is
host/eager-only with strict in-memory checkpoint/resources and no schedule,
seed, output, artifact, threshold, evidence, or promotion authority.

A separate v1 `HCCLLearnedMemoryFeedbackBridge` now connects this exact HCCL
transaction to one `LearnedExperientialMemoryController` without modifying
either donor. It owns exactly those two states plus one fixed pending binding.
The binding covers the controller transaction, HCCL source/decision and event,
B/M receipt identities and contents, common hard mask, selected agent,
categorical action, and routing result. Settlement derives only the selected
agent's immediate `memory_total.net_reward`, with the unbound agent required
to have equal effective B and M actions. Masked or unrouted retrievals use the
controller's matching no-learning settlement; every composite failure rolls
all owners back bit-exactly for retry. The boundary and bounded prebound scan
are host/eager-only, and the scan is only a replay/verifier for an already
complete caller-supplied receipt trace. This is a memory-utility credit
mechanism, not delight, an actor backward pass, or a learning-benefit result,
and it adds no agent, schedule, seed, run, artifact, threshold, evidence, or
promotion authority.

`HCCLTwoLiveMemoryBridge` now supplies the separate two-live-agent L0 rung. It
owns one HCCL state and exactly two live learned-memory adapter states. Existing
pending receipts bind each agent's `B` and `M`; absent receipts use `B=M`, and
`P=M` is explicitly no planner. Agent 0 receives only `M0B1-BB`, agent 1 only
`B0M1-BB`, and memory interaction remains audit-only. The PP outcome advances
each child from its own executed `M` action, while next masks install only on
the atomic all-owner commit. The host/eager bridge has strict checkpoints and
resources, but no delight or actor backward, run, evidence, benefit, or
promotion claim.

The additive v1 `HCCLTwoLiveMemoryPrepareAdoptBridge` leaves that owner tree
and v1 API unchanged while exposing a transient downstream boundary. One
preparation evaluates the HCCL and both live adapters once and retains exact
candidate, raw/final STOMP, owner-finalization-trace, and per-agent
extended-mask facts. Adoption runs integrity checks and the two live child
adoptions only, with zero world or learner reevaluation. A veto, replay,
tamper, or foreign binding returns the full source and outer-gates public
child-applied facts. Partner fusion is allowed only when its shared feature
axis cannot be replaced. The receipts are unkeyed integrity bindings rather
than caller authentication, preparation is not persisted, and `P=M` remains
the explicit no-planner rung. No Kondo actor backward executes, so this seam
cannot establish delight or that a gradient sparks joy.

The additive `HCCLTwoLiveMemoryFactorizedPlannerBridge` supplies the next
planner rung without persisting per-agent `P` Prototype snapshots. One HCCL
state, the same two live post-memory `M` states, and one paired factorized
planner state/cache are the complete owner tree. A preparation evaluates HCCL
and both live adapters once, completes the paired behavior/joint-world models
once, and reconstructs each transient `P` through the public cached-action
replacement. Planner grounding is the external GRU builder's 17-wide
constructed state; the 16-channel physical raw observation remains separately
bound to the HCCL plant transition and `PP` commit. A hard mask can replace a
raw proposal with `M`, but the effective `P` pair is what HCCL consumes and
`PP` is the sole world successor. Adoption fully validates current and
candidate planner states/caches and performs no donor, model, or world
reevaluation. All seven focused current-source cases pass individually. This
is still unauthenticated host/eager L0 mechanism integration with no external
or physical dispatch, safety, Kondo actor backward, delight, matched planning
benefit, evidence, or promotion claim.

The additive `HCCLTwoLiveMemoryRepeatedOptionPrepareAdoptBridge` composes the
no-planner rung with two coordinator-free repeated-option metadata bundles;
the only persistent STOMP owners remain inside the two live coordinators.
Ordinary preparation consumes each raw STOMP result once and projects a real
memory action change through the complete cached-action/coordinator owner.
Adoption is all-or-none. At an exact started-but-quiescent boundary, the
per-agent atomic path consumes the fresh-cold retirement/filter/replacement
seam while preserving the HCCL world, the other agent, learned memory, pending
feedback, and primitive masks; an intermediate cold slot is never persistent.
Focused contracts cover full-owner projection, tamper/foreign/veto/replay
rollback, exact lower diagnostics, and resources. This is still `P=M`,
host/eager L0, unkeyed integrity plumbing with no planner, physical dispatch,
caller authentication, safety authority, actor backward, delight, empirical
benefit, evidence, or promotion claim.

`HCCLContinualDyadTransaction` is the first atomic integrated-owner rung. One
transaction owns the HCCL world/attribution state, two live post-memory action
stacks, two slow-context states, and one paired factorized planner. Its split
boundary completes both agents through memory once, then either binds the
planner or supports the explicitly disabled-planner form without donor
reevaluation; all owners adopt or the complete source returns. The additive
ordinary `step(state, next_hard_action_masks)` path owns event preparation,
current B/M/P binding, both exact causal-core memory inputs, transaction
preparation, integrity receipt, and adoption. The caller can supply none of
those intermediate identities. The fixed-life executor now delegates to that
same path; its explicit runner accepts a fresh typed key and requires every
event of exactly the 420- or 8,998-event primitive-only schedule to commit with
continuous clocks before returning a complete in-memory trace. This remains
host/eager L0 machinery: no partial resume, checkpoint, reserved protocol seed,
physical dispatch, matched result, evidence, promotion, or completion of the
causal-core target follows.

The separate `KondoExecutedActionLineageBridge` binds a fixed unmasked actor
proposal batch to the exact post-memory action-stack source, preparation,
decision, and candidate binding. It rederives the actor sample and bit-exact
behavior log probability, reconstructs public action-stack adoption, and
admits a row only when proposal digest, consumed planner candidate, planner
action before masking, final `P`, and the following real transition action all
agree. Memory-selected or otherwise overridden paths are actor-ineligible.
Invalid rows are sanitized before exactly one `KondoSparseActor.step`; with no
eligible row that call executes no backward and preserves actor state exactly.
Protected critic/baseline/return/safety arrays remain full-batch, and only the
nested `KondoSparseActorResult` carries the canonical execution-level joy fact.
The v1 bridge accepts only all-true masks and is host-only, unkeyed L0 integrity
rather than caller or physical-execution authentication, dispatch/safety/critic
authority, efficacy, evidence, or promotion.

`HCCLKondoContinualDyadRoute` v3 makes Kondo the recurring owner of `P` over that
atomic donor. Genesis `event0` installs a proposal and compact adoption
certificate without actor backward. Each successor `event` first consumes the
prior compact lineage through one actor transaction and only then samples and
atomically installs the next pair. The paired planner is a learning-only
shadow with `planning_enabled=False`; actor features are the 23-wide
post-memory base, and the two live action stacks remain the only Prototype
owners. Immediately after preparing its own causal-core event, the route
derives exactly two learned-memory inputs from that event and the two canonical
agent rows. Their uncertainty is unavailable, safety cost is available exact
`+0.0`, reliability is one, source identity is the row, and provenance is
`2 * source_step + row`; none of those fields is a caller surface. An outer
adoption veto restores the complete persistent source but does not rewrite
whether the nested actor backward already executed. Before every recurring
actor step, the route derives a two-row protected batch from the exact pending
`P` and current `PP` transitions and updates separate linear reward-value and
cost-value heads. Both heads use the same pre-update snapshot and detached
one-step bootstraps; cost is `safety_cost + message_charge`. Their full-batch
features, reward baseline, and return target pass directly to Kondo. Event 0
does no protected or actor update, and route, actor, and protected clocks remain
bound. The route supports only all-true masks; scheduling, actor keys, and masks
remain caller-driven host/eager L0 machinery. Current HCCL costs are zero, the
protected-only checkpoint is not route recovery, and there is no autonomous
life, route checkpoint/resource closure, authentication, dispatch, physical
safety or critic-efficacy result, evaluator, benefit, evidence, or promotion.

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
exact no-ops. OaK now also has a source-bound adoption seam for one
caller-authoritative `STOMPUpdateResult`: it validates bit-exact source state,
outer/nested clocks, endpoints, and success diagnostics, then performs OaK
accounting without a second STOMP evaluation. A separate quiescent rebind seam
admits only reset-slot STOMP policy/model/base-head changes, preserves global,
primitive, unchanged-slot, clock, and RNG leaves, and zeros only those slots'
three OaK statistics. Optional extended-action masks exclude cold options from
selection, real bootstrap, and planning. These trusted-caller seams provide
unkeyed integrity rather than derivation authentication. This is ownership-
level mechanism coverage, not automatic chord
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

The standalone rollout layer now owns an exact big-endian `uint32[2]` event
identity while its int32 step count is saturating telemetry only. A terminal
all-ones identity, corrupt source state, or invalid behavior/world prediction
returns the original state and RNG atomically. `DreamRolloutStepResult`
exposes the pre/post identities and validation/commit diagnostics. Strict v2
config, state, result, and resource schemas reject extra or mismatched fields;
legacy config/state migration is explicit and accepts only unambiguous
representable counters. Fixed accounting covers concrete persistent bytes and
the configured maximum steps, behavior/world calls, and key splits, with zero
persistent capacity growth; the exact clock adds 8 bytes over the old rollout
state. Eager, JIT, and JIT-disabled mechanism tests cover these contracts.
They do not establish dream quality, planning benefit, an all-enabled
whole-Prototype composition, artifacts, evidence, promotion, or Alberta Plan
completion.

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

### Self-Normalized Resets baseline

`SelfNormalizedResets` now provides the missing low-knob WP2 reset mechanism
for one fixed-width dense ReLU layer. Each unit owns exact two-word age and
epoch clocks plus a fixed trailing ring of completed positive-support
inter-firing intervals. The estimated geometric tail uses stable direct
`log1p(-p)` arithmetic and the explicit observed-run mapping
`P(A >= age + 1) = (1 - p)^age`; eligibility additionally requires exact
history and post-reset warmup floors. The ordinary caller update occurs first,
then selected incoming columns/biases are freshly initialized, outgoing rows
are zeroed, and exact Adam slices are cleared. Deterministic caps, typed
Threefry ownership, atomic invalid-input no-ops, source/representation binding,
strict checkpoints, real reset persistence, and eager/JIT/scan parity are
tested.

This is an L0 standalone baseline, not a result. The positive-support
fixed-window resolution is serialized and is not claimed bit-equivalent to the
authors' released silent-age histogram implementation. It has not been wired
into a matched whole-learner campaign or shown to improve plasticity,
retention, safety, or control.

### Spectral regularization, AdamO, and Calibrated Partial Resets

Three further WP2 arms now exist as independent, defaults-off dense-layer L0
mechanisms. `SpectralRegularizer` evaluates
`(sigma_max(W)^k - 1)^2 + ||b||_2^(2k)` with a persisted power-iteration probe.
`AdamO` updates Adam moments from the task gradient alone and adds the
rectangular Gram-isometry gradient as a separately scaled delta.
`CalibratedPartialResets` maintains normalized per-unit incoming-gradient
utility, schedules partial pulls toward fresh He-uniform incoming weights, and
decays the corresponding outgoing rows. Its RNG advances only on reset events,
and it does not mutate caller-owned base-optimizer state. Biases are excluded
because the paper appendix and released v1 JAX implementation disagree.

Equation tests, exact clocks, finite atomic rejection, fixed resources, strict
checkpoints, and eager/JIT/scan behavior establish mechanism integrity only.
These surfaces are not multi-layer or convolution wrappers, Prototype arms,
matched comparisons, efficacy evidence, default selections, or a WP2 exit.

### Optimization-centric plasticity diagnostic

`OptimizationCentricPlasticityDevelopmentReport` now provides a bounded
task-crossed diagnostic rather than treating dormant fraction as plasticity.
One ordinary nonlinear SGD learner and one initialization-centred
L2-constrained twin start from the same immutable parameters and process the
same evaluator-owned A/B/A stream with equal update opportunities. At each
switch the in-memory report retains bit-exact old-task and incoming-task raw
gradients, dot/cosine alignment, fixed-radius two-sided local probes, and phase
parameter displacement, churn, and sign changes. Hidden-activation dormancy is
measured separately and is excluded from the fixed descriptive
zero-gradient/local-neighbourhood rule. Config, protocol, executed source,
runtime, snapshots, logical resources, and full causal replay are fail-closed.

This is one 12-update L0 `not_assessed` schedule. It writes no artifact, owns no
threshold/evidence/promotion authority, and establishes neither the OCP
hypothesis nor a benefit for L2 constraints. It remains an input to the open
matched WP2 plasticity/retention matrix.

### Independent C-CHAIN equation comparator

`CChain` is now a clean-room, paper-equation L0 comparator rather than a
borrowed implementation. Its exact Equation 8 term is
`0.5 * mean((f_current(reference) - stop_gradient(f_lag(reference)))**2)`.
The comparator requires exactly one scalar model output per reference sample;
it rejects vector-valued per-sample extensions rather than widening that
exactness claim.
Train and reference rows carry separately declared, nonzero, unique, disjoint
`uint32[2]` identities. A valid proposal computes the base objective and churn
term in one combined `jax.value_and_grad`; a rejected runtime preflight and the
commit boundary perform zero autodiff. A successful commit advances an exact
two-word lifetime, moves the proposal source into the one-step-lag slot, and
binds the next expected current tree to caller-supplied applied parameters.

The appendix coefficient ratio uses running mean absolute base and churn loss.
Its finite denominator, warmup, bounded trailing ring, and coefficient clamps
are explicit Alberta controls, not Equation 8. Ring indexing is exact across a
32-bit clock rollover. A separate empirical-NTK helper reports the paper's
minimum singular-value-prefix rank for `1 - delta` mass and descriptive
diagonal/off-diagonal statistics; it cannot gate an update or claim evidence.

This surface does **not** reproduce the paper's full sequential PPO/DQN
algorithm. Model/loss binding words, sample identities, and content tags are
caller-owned unkeyed integrity declarations rather than authenticated callable
or data provenance; optimizer application is external and unauthenticated.
There is no Prototype/default-agent composition, matched plasticity or control
comparison, efficacy result, artifact, evidence, promotion, or SOTA claim.

### Fast/slow recurrence is adaptation, not retained permanent memory

The existing additive gated `FastSlowLearner` now has an authoritative
big-endian `uint32[2]` lifetime, saturating int32 telemetry, an all-ones
fail-stop, atomic rejection of corrupt/non-finite source, input, gradient, or
candidate state, and strict v2 config/state/result/resource records. Its valid
three-step predictions, errors, and metrics retain their historical float32
values; full eager/JIT state comparison uses the declared `rtol=1e-6`,
`atol=1e-7` backend allowance rather than claiming bit equality. The exact
clock adds eight persistent bytes (`1296 -> 1304`) and no replay capacity.

A separate consumed 512/512/512 A/B/A scalar regression life compares ordinary
fast/slow learning with an equal-shape slow-only ablation. Ordinary fast/slow
slightly improves some early and tail adaptation errors, but its slow-component
A probe changes from MSE `0.0188931` after A1 to `4.01919` after B, a
`4.00029` increase and `212.73x` ratio. Its A2 tail MSE `0.0269165` appears only
after further A updates. That is reacquisition, not preservation: the current
slow path overwrites A and does not implement a Permanent/Transient retention
boundary. This is one synthetic consumed-root L0 `not_assessed` negative
diagnostic with no threshold, held-out panel, artifact, evidence, promotion, or
default-selection authority.

An explicitly Alberta-derived online Permanent/Transient sibling now tests
whether merely separating the two representations repairs that failure. It
owns independent tanh encoders and heads, trains the transient residual,
distils the post-transient current-sample prediction into the permanent path,
and decays the transient head. Both ordinary consolidation and a
no-consolidation ablation use the same 788-byte state, two gradients and three
in-update forwards per step, with no replay, task identity, boundary, reset, or
online random draw. The design record marks the learner `source_faithful=False`
and enumerates seven departures from the cited small-task and Craftax methods.

On the same already-consumed 512/512/512 source, ordinary permanent A-probe MSE
changes from `0.0434621` after A1 to `3.92572` immediately after B; the combined
probe changes from `0.0345716` to `4.63162`. The permanent path returns to
`0.0503954` only after A2 updates. A readout-only removal shows that the
permanent contribution helps after A1 and A2 but harms the A probe after B.
Thus this always-active permanent subsystem is another overwrite channel, not
a dormant retained memory. The result is an in-memory consumed-root L0
`not_assessed` falsification, not a source reproduction, thresholded verdict,
artifact, comparison winner, evidence claim, or default selection.

A separate generic latent-context expert bank tests actual active-only
dormancy. It credits the existing `ContextInference` freeze law, fixes two
linear experts from birth, and binds the complete owner/parameter/observation/
prediction cache before accepting each target. Both selective and no-selection
arms use the same 32-byte state and compute both expert predictions, losses,
and analytic candidate gradients; exactly one subtree commits per event. On
the consumed A/B/A source, selective routing learns A as runtime expert 0 and
reactivates it after the first observed A2 outcome. Its phase prequential MSEs
are `0.02007`, `0.01496`, and `0.000722`, versus `0.01158`, `0.04369`, and
`0.04535` for no selection.

That apparent recurrence benefit is not clean retained memory. Per-sample
routing switches 10 times in A1 and three times in B; one B sample selects and
updates the learned-A expert, changing its A-probe MSE from `3.73e-20` to
`1.90e-05` and changing its subtree hash. The first B and A2 predictions occur
before their revealing outcomes, and selection/training only afterward is not
pre-outcome context identification. The core proves bit-exact nonselected
subtrees per transaction and one-outcome A2 reactivation, while the consumed
diagnostic falsifies phase-long dormancy under one-sample routing. It has no
tuning, threshold, writer, artifact, evidence, promotion, or default authority.

A fixed-horizon successor now tests whether two authenticated outcomes are
enough to quarantine the switch rather than merely delay the same overwrite.
On an opening event, one unique dormant challenger may be globally no worse
than the current owner, but neither expert updates. The source-bound second
event must preserve that no-worse relation and add strict support for the same
candidate; a remaining tie, ambiguity, substitution, or non-finite candidate
rejects atomically. This is a fixed `H=2` rule with no margin, dwell search,
replay, or online RNG.

On the already-consumed 512/512/512 A/B/A source, the enabled arm's learned-A
subtree is bit-exact before and after B, receives zero B updates, and is
reactivated after two observed A2 outcomes. Its A1/B/A2 prequential MSEs are
`0.0156662`, `0.0145220`, and `0.0174704`; the same-work routing-disabled arm
records `0.0156662`, `0.0994502`, and `0.0754818` and sends 498 B updates to A.
Four openings yield two confirmations and two rejections, hence four deliberate
zero-commit quarantine events. This is the first clean bounded dormant-expert
retention result in this consumed synthetic lane. It remains L0 development
work with supplied two-expert capacity and horizon, post-outcome switch
inference, one consumed root, and no control benefit, generated-feature link,
writer, artifact, calibrated threshold, scientific evidence, promotion, or
default authority.

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

The separate `NonlinearOffPolicyActorCritic` closes a narrower mechanism gap
without changing that evidence verdict. It uses a shared tanh trunk with
separate actor/critic heads, distinct actor/critic contributions to trunk
eligibility, and clipped per-decision `pi(a|s) / mu(a|s)` correction. Learning
can consume only the exact cached executed-action receipt: observation and
action bits, target and caller-declared behavior log probabilities, target and
behavior revision words, and the action identity are all bound before the
transition. Actor, critic, and trunk have fixed independent plastic/frozen
policies and optimizer state; target-policy revisions, typed Threefry
sampling, exact fail-stop clocks, atomic invalid-input rollback, strict
checkpoints, and complete persistent-state byte accounting are contract-tested,
including a hand-derived two-step trace and eager/JIT/scan parity.

This remains an L0 kernel, not off-policy control evidence. It is a discounted
scalar-V method with one tanh trunk, clips ratios by construction, does not
correct initial-state or state-visitation mismatch, and treats the external
behavior revision as a bound caller declaration rather than independently
authenticating a behavior-policy owner. It has no learned component-utility
policy, average-reward baseline, policy-churn evaluator, matched SARSA arm,
retention result, convergence guarantee, benchmark artifact, or promotion
authority.

### Paper delight/Kondo, candidate-update auditing, and typed signals

`LearningValueRouter` now gives the eight named learning-value channels a
fixed-state causal boundary. It records producer, causal object, units, domain,
bounds, and normalization scale for each channel; validates them independently;
and normalizes only from pre-update Welford statistics. Its paper-DG actor,
exploration, model-memory/replay, adaptation/change, safety, and full candidate-
audit evidence routes expose exact-zero unavailable fields rather than a
universal sum. Delight must exactly match the float32 advantage-times-surprisal
identity, while unrelated failures cannot suppress valid safety cost. The
router performs neither Kondo selection nor a candidate-update verdict. The
opt-in Prototype v19 composition advances one owner-bound router state only on
accepted real transitions and gives the state-builder audit only its raw
candidate-evidence route. Representation-candidate validity gates the candidate
and probes but not producer availability; normalized values are diagnostic
routes only. Router-disabled config/state/checkpoint shapes remain historical.
This is bounded L0 routing with checkpoint/JIT/scan contracts, not evidence
that its inputs are calibrated or that any consumer improves.

`assess_candidate_update` is the canonical API for a separate optimizer-level
safety audit: does a
candidate gradient or formed update predict improvement on caller-supplied
objective, retention, and safety probes under a caller-attested independence
contract while respecting an update-norm bound? It fails closed unless
`probe_independence_attested` is true and all eight typed learning-value
channels have explicit valid availability. Its detached assessment exposes
`candidate_update_audit_passed` (the canonical alias of its single stored
`accepted` verdict), together with a weakest-link weight and named
raw-candidate plus tentative-update diagnostics.
`CandidateUpdateAudit*`, `apply_candidate_update`,
`PrototypeCandidateUpdateAuditEvidence`, and
`candidate_update_audit_evidence` are the canonical public names;
`GradientJoy*`, `assess_gradient_joy`, `apply_gradient_joy_update`, and the old
Prototype keyword remain exact compatibility aliases only.
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
`apply_candidate_update` is the canonical application boundary; the historical
`apply_gradient_joy_update` spelling is a compatibility-only alias. It reassesses
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
transition. The standalone audit marks the paper-defined `LearningValue.delight`
channel unavailable unless it is the bit-exact finite float32 product of valid
declared advantage and action surprisal. The Prototype derives delight
internally rather than accepting a caller-supplied replacement for the audit
probe gradients. Neither mechanism is a generic reward score, and
neither gates safety or model learning. The gradient audit remains a local
first-order L0 mechanism, not realized-improvement evidence.

`ExternalBuilderCandidateEvidenceProducer` now supplies the stateless
full-GRU form of that probe boundary. It binds caller-owned objective,
retention, and safety representation probes to one exact external-coordinator
identity and analytically pulls their hidden components through the cached
source RTRL sensitivity. Stale or non-finite probes become unavailable exact
zeros and independence remains caller-attested. A permissive mechanism test
reaches the real coordinator's builder-only accepted-update path with three
pullbacks, zero extra model forwards, and zero actor backwards. This closes
plumbing only: it neither establishes that a gradient “sparks joy” nor
provides calibrated probes, realized benefit, safety authority, or promotable
evidence.

In the paper's terminology, delight is the exact float32 advantage times
selected-action surprisal, and a sample “sparks joy” iff that exact contribution
enters an actor backward pass that actually executes. `KondoGate` supplies an
L0 detached forward admission plan and fixed-capacity sparse gather; the
`KondoSparseActor` consumer establishes execution. That fact is independent of
gradient finiteness, parameter-update acceptance, and later outer-transaction
acceptance. The gate
supports finite-temperature Bernoulli-price
and deterministic fixed-rate top-k modes, preserves caller-declared forced
samples, and reports a flag requiring caller-managed full-shape fallback on
overflow. When capacity is below batch size, tests verify that the downstream
backward JAXPR receives the smaller gathered shape. The screen has
eager/JIT/scan coverage; the config-bound gather is intentionally host-only,
and checkpoint, resource, and invalid-input contracts are also tested.
Canonical gate config/checkpoint payloads are v2 and expose only detached
backward-admission-intent semantics; exact v1 payloads remain a strict
import-only compatibility path.
`KondoSparseActor` now performs a real nonlinear categorical backward only
after that gather; tests witness a capacity-3 JAXPR rather than the full batch
of 6, while forced/overflow survivors take an explicit full-shape fallback.
In sparse and full-shape fallback paths, changing rejected features, actions,
and detached advantages leaves the actor loss and complete gradient bit-
identical, directly pinning “sparks joy” to a contribution entering an executed
actor backward rather than the standalone screen's forward intent.
Exact action/revision/behavior-probability contracts are tested. Returns and
baseline predictions enter the actor only through detached advantage; critic
and safety features stay outside its loss, while protected learners remain
full-batch and ungated. The host orchestration boundary is not JIT/scan
compatible. A strict four-arm development evaluator now starts
ordinary-full, capacity-matched uniform-sparse, Kondo-top-k, and diagnostic-
overflow arms from the same immutable parameters and source trace. It retains
selected indices, compiled backward shapes/invocations, logical proxies,
held-out-within-development diagnostics, and update-free interleaved
`perf_counter_ns` samples with raw/nearest-rank p50/p95 timing. Timing excludes
host screen/gather and wall-clock bytes are outside exact deterministic replay.
All results remain `not_assessed`; this first evaluator has no protected-
learner replay composition, demonstrated compute saving, DG reproduction in
this consumer, or learning/safety benefit.

All three evaluator contracts are now v2. Cross-arm outcomes describe executed
actor-backward inclusion neutrally; replay and on-policy records use
`executed_actor_backward_mask` with canonical meaning
`gradient-contribution-entered-executed-actor-backward`. The canonical
execution-level use of `sparks_joy` is an actual `KondoSparseActorResult`:
ordinary-full and uniform-sparse use manual backward kernels rather than Kondo
transactions, and ordinary-full makes no delight-selection claim.

A second strict replay lane now runs ordinary-full, capacity-matched uniform,
paper top-k Kondo, and fixed-capacity Kondo-plus-minimum-random-reserve actor
updates beside independent full-batch baseline, critic, representation,
world-model, and safety/guardrail learners on one uninterrupted A1/B/A2
contextual-gambling trace. Every arm receives exactly one actor and one
protected update opportunity per source batch. The protected gradients,
results, predictions, rare-failure coverage, and final states are bit-identical
across arms; current-policy delight, executed actor-backward inclusion masks,
selected indices,
backward shapes, logical row-slot proxies, descriptive recurrence diagnostics,
and exact causal checkpoint replay are retained.

The replay source fixes actions and exposes no behavior policy, and the actor
updates apply no importance correction. They are therefore off-policy
surrogates, not policy-gradient or DG-efficacy estimates. The minimum random
reserve is an Alberta extension, logical proxies are not FLOPs, and the lane
measures no wall clock, memory, energy, end-to-end latency, policy outcome, or
safety benefit. It writes no artifact and remains `not_assessed`. Thus a
bounded actor-critic-adjacent replay composition now exists.

A separate closed-loop development evaluator now samples each of the same four
arms from its own immutable actor revision, using evaluator-owned typed
Threefry common uniforms for exogenous randomness only. Actor/environment
trajectories may diverge; each arm still receives exactly one boundary actor
update and one full protected update per batch, including forced rare failures.
Exact actions, behavior log probabilities, revisions, causal chains,
checkpoint resume, and source/runtime-bound replay are retained in memory.
This mechanism remains `not_assessed`: it establishes no learning efficacy,
compute saving, safety benefit, evidence, promotion, or WP5 completion.

`ProspectiveExploration` now implements the separate prospective-exploration
extension as a fixed-budget L0 selector. It computes expected improvement
times capped host-relative surprisal and exposes random, epsilon-greedy,
ensemble-disagreement, information-gain, and learning-progress comparators
under the same candidate and logical RNG schedule. Exact source-event,
producer-owner/revision, and pre-decision attestation receipts fail closed.
Candidate ranking never consumes the caller-owned hard shield; only after
selection can that shield admit the candidate or a separately permitted host
fallback. Expected improvement and all comparator metrics are supplied rather
than calibrated, causal attestation is metadata rather than proof, and a
Boolean shield receipt is not physical-safety evidence. The synthetic
stochastic-trap and long-horizon diagnostics are threshold-free mechanism
checks, not exploration efficacy or an exit-gate result. The canonical v2
score is `expected_improvement_surprisal_score`; it is not DG/Kondo delight and
no actor backward executes. Historical v1 `DelightfulExploration` import and
config spellings remain compatibility-only, while v1 checkpoints fail closed.

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

`RealStateOneStepDyna` now supplies an isolated WP4.6 item-1 kernel for the
ensemble lane. It records bounded real representation/action anchors against
exact decision-time ownership, then accepts only exact current model/control
states whose revisions advanced monotonically through the caller's real
updates. Its one-step target is formed before each synthetic Q update as
`reward_hat + continuation_hat * max Q(next)`. Observed action support,
residual readiness, epistemic and residual limits, finite values, and ensemble
termination agreement veto unsafe proposals; synthetic traces start at zero
and hidden utility/lifecycle state is restored. Planning RNG, clocks,
checkpoints, and resource scopes are separate, and the model is read-only. The
kernel is not integrated into `PrototypeAgent`, the veto thresholds are not
externally calibrated, and no retained-model or control-benefit result exists.

`EnsembleShortRolloutPlanner` now covers WP4.6 items 2 and 3 as a separate L0
proposal lane. It starts from exact real decision anchors and binds immutable
linear policy/value arrays plus the full ensemble-state content to monotonic
revision receipts. Fixed-shape policy-directed or max-epistemic paths require
support, residual readiness/magnitude, epistemic, finite-value, and termination
agreement at every transition. Learned terminal steps stop padding and cannot
bootstrap; valid horizon truncations may bootstrap before a reverse multi-step
return. Model, policy, and value state is read-only, and stale/content-alias/
clock/capacity failures are atomic. No Prototype, STOMP, actor, critic, or model
consumer is implicit in the planner. A separate
`ImaginedRolloutSelectionGauge` freezes one generation and builds a bounded,
causal primitive-action × region audit before issuing full-content candidate
receipts. Evidence, realized validity, reward/next-observation error,
termination, success lower-bound, top-quantile purity, and caller-owned
safety/protected masks are noncompensating, and a proposal cannot audit itself.
Admission is path-prefix closed. `AuthorizedImaginedRolloutActorCritic` makes
an autodiff-free authorization proposal, revalidates it at commit, and performs
exactly one real fixed-shape actor/critic backward pass only after a fresh
preflight; stale, replayed, or tampered preflights perform zero. Critic targets
are terminal-correct and dream imitation uses graded positive advantage; a
competent-real cloning mode uses the same prefix-closed transition/update
budget. Tags cover post-mint integrity but do not authenticate planner issuance
or the caller's competence assertion. A strict
`GroundedImaginationComposition` now derives the planner policy/value snapshot
from the live actor/critic, obtains the rollout batch locally, passes that exact
batch directly into the gauge, and atomically couples planner, authorization,
learner, dream, and composition clocks around the sole possible backward pass.
This closes the caller tensor-substitution gap at the composed boundary. Model
support, the real anchor, region assignments, safety/protection masks, and
environmental truth remain unauthenticated caller attestations. The audit
floors remain supplied, there is no Prototype or dispatch integration, and no
matched planning, return, or retention benefit has been assessed.

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
baselines. This remains L0 development infrastructure: there is no rollout
probe, realized-resource matching, energy measurement, or scientific protocol.
A fixed versioned in-memory development report now runs `PrototypeAgent`, a
running-reward bandit, and a frozen-action baseline over two consumed A/B/A
seeds. It embeds reconstructing evaluator reports, raw action/decision
ownership, exact opportunity/logical-byte accounting, deterministic logical
latency, parameter/policy/value churn, and explicit applicability for every
other WP1 field. Source/runtime-bound replay and checkpoint continuation are
validated. This satisfies report construction only: it is always
`not_assessed`, writes no artifact, has unavailable hardware/internal-gradient
measurements and inapplicable disabled-component diagnostics, and establishes
no efficacy or promotion result. A strict paired campaign companion can
run the ordinary evaluator over declared seeded factories and reconstruct raw
per-seed reports, direction-normalized differences, and deterministic
stratified bootstrap intervals. It remains `not-assessed`; no promotable
Prototype campaign has been run and the descriptive intervals are not a
promotion decision.

### Learnable state and bounded experiential memory

Identity, fixed-trace, online trainable gated, conventional dense full-GRU,
and diagonal complex compressed-RTU state builders now share a causal
fixed-budget contract. The recurrent
builders expose pure proposals from the source state and an atomic commit into
the already-advanced destination
state, preserving recurrent carry and sensitivity. The full GRU uses exact
fixed-parameter RTRL for its dense input/recurrent gates, with explicit
`O(H * P)` sensitivity storage and approximate carry after parameter updates.
The RTU persists only unit-diagonal sensitivities, exact for fixed recurrent
parameters; its default moving-parameter carry and optional source/delta-owned
diagonal Taylor correction are explicitly approximate.
The separate `RTUGenerateAndTest` lifecycle observes the pre-update downstream
representation gradient and maintains fixed per-unit effective-contribution
telemetry/age/support. Both strict live Prototype objective adapters now score
joint deletion of each complex unit's real and imaginary representation
channels against the frozen pre-update balanced heads. Positive bounded
deletion-loss changes have an independent EMA and evidence floor; live
replacement ranks only this causal utility, never the contribution proxy.
Missing or immature evidence defers recycling without dropping the real
transition or ordinary builder learning; attempted invalid/non-finite internal
scoring rejects the entire outer transaction. At eligible boundaries it selects a
stable fixed quota of mature, low-causal-utility, unprotected complex units. Whole-unit
replacement redraws polar recurrence and both input rows and scrubs activation,
compressed sensitivities, and optional Taylor trace/source/delta slices. An
optional ordinary builder update is recomputed from its exact source, and only
that bit-identical live destination can commit before replacement; stale or
invented destinations roll both states back. Typed Threefry ownership, exact
clocks, resources, checkpoints, and eager/JIT/scan parity are covered. The
lower-level finalization receipt reconstructs the exact advance receipt,
independently reruns the RTU commit, and exact-matches the destination and mask,
so it proves deterministic derivation from supplied values. It does not
authenticate the caller's lifecycle source, downstream objective/gradient, or
ordinary learning proposal. Strict
comprehensive-objective adapters now consume this lifecycle for linear
STOMP/OaK only: they prepare recurrence without learning/action RNG, learn the
current transition under the old representation, apply at most one atomic
whole-unit replacement, scrub the selected comprehensive-head plus
base/intra-option/option-model axes, then perform the ordinary next-action
selection under the recycled
representation. Active options defer replacement without discarding the real
transition. Distinct unit/event clocks make the builder revision equal accepted
transitions plus nonempty replacement events. Each adapter owns the lifecycle
source and constructs the objective gradient and source-bound proposal, closing
the three lower-level authority boundaries. The caller-targeted lane preserves
its bound target receipt; `PrototypeCausalStateObjectiveTargets` instead derives
one learner-owned factual target bundle from the accepted transition and reuses
it unchanged for every frozen-head deletion. Four source-level builder-commit
evaluations, two RTU-commit evaluations, and one frozen-head deletion
counterfactual per complex unit cover preflight and independent derivation
checks, but only one logical ordinary update and at most one logical replacement
event persist. Each RTU-enabled adapter's lifetime is bounded by the per-unit uint32
age/support/evidence counters (`2**32 - 1` accepted transitions). Nonlinear STOMP, planning,
world/model/replay/dreaming, Horde, IA, partner/memory, GRU, historical
candidate-update audit, and feature lifecycle remain excluded. This is L0
prequential causal-deletion/recycling machinery, not paper-defined delight, an
independently held-out outcome probe, autonomous objective discovery, or
learned-state benefit.
The opt-in `PrototypeAgent`
path can pass the real grounded-model gradient and a causal current-control
semi-gradient through a stateless successor mixer. Idle transitions use the
base-Q objective; executing options use the current intra-option objective;
delayed option-start and replay gradients are excluded. Source norms, weights,
clipping, cosine/conflict, and failures are explicit, and the exact mixed
candidate feeds both the builder proposal and the candidate-update audit. Its
small write/hold POMDP is explicitly development-only and resource-unmatched.
Its deterministic consumed four-seed default run changed all trainable
recurrent parameter vectors and produced mean frozen-suffix accuracy `0.5158`
observation-only, `0.5292` fixed trace, `0.5258` online-gated, and `0.5067`
full GRU; the RTU reached `0.5617` with `1,324` total persistent bytes versus
`12,204` for the full GRU. That is a descriptive, supervised,
resource-unmatched development signal, not an exit-gate success.
A separate `BalancedStateObjectives` L0 kernel now owns separately updated
linear GVF heads at multiple strictly ordered discounts and a consecutive-pair
inverse-action head. It averages the GVF family before fixed positive group
masses are applied, returns clipped current/successor gradients bound to one
exact executed-action receipt and caller representation revisions, and has
strict resources, checkpoints, fail-stop clocks, retry-safe rejection, and an
`OnlineGatedStateBuilder` commit witness. An opt-in
`PrototypeBalancedStateObjectives` adapter now authenticates the exact
Prototype decision/action, bitwise representation, observation event, and
decision-time builder revision; it scores terminal/bootstrap state before
autoreset, commits the two recurrent-sensitivity gradients as one clipped
builder update, and rolls every component back atomically on rejection. The
ordinary Prototype path remains unchanged. The weights are declared rather
than empirically calibrated. A separate standalone
`ComprehensiveStateObjectives` L0 kernel now owns action-conditional
next-observation/latent, reward, stable Bernoulli termination,
multiple-timescale GVF, distinct state-value and selected-action-advantage,
and inverse-action heads. Prediction and control subheads are averaged inside
six fixed positive family masses; independent head parameters, step sizes,
revision rows, exact receipts, numerical rollback, finite-difference
gradients, strict checkpoints/resources, and eager/JIT/scan parity are tested.
An opt-in `PrototypeComprehensiveStateObjectives` transaction now composes it
with online-gated, full-GRU, and compressed-RTU builders. Caller target bits,
source/provenance, dispatched decision/action, observation event,
final/bootstrap observation, and builder owner are bound before current and
successor pullbacks are summed into one clipped logical update; failure restores
the whole composition. The strict-linear RTU lane now consumes that update via
content-bound prepare/finalize receipts, independently recomputes its derived
destination, and scrubs recycled consumer axes before selection. The
caller-targeted lane's targets and masses remain uncalibrated. A separate
`PrototypeCausalStateObjectiveTargets` owner removes ordinary target selection
from the caller and now accepts the exact RTU builder only together with the
matching strict lifecycle. Its deletion scorer freezes the pre-update heads
and reuses the single factual learner-owned target bundle; no target is
regenerated under deletion. Objective/pending, RTU sensitivity/Taylor, and
supported linear STOMP recycled axes are all persisted as canonical `+0.0`.
Late cache refusal, invalid scoring, or finite selected-axis corruption rolls
Prototype, target owner, lifecycle/RNG, and every consumer back bit-for-bit.
Exact typed checkpoint metadata, canonical empty-array sentinels, continuation
after restore, causal resource counters, and the isolated uint32 fail-stop are
covered. The learner-owned lane is likewise L0 `not_assessed`; its objective
masses, target quality, and optional-cumulant semantics remain uncalibrated.
General consumer compatibility,
independently held-out feature-utility efficacy, broader lifecycle curation,
and matched Forager outcomes remain open, so the Step 8 exit gate is not met.

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
confidence claim. A defaults-off `ExperientialMemoryAdvantageGate` now adds a
stricter stateless dispatch boundary: both the base and proposed categorical
actions must have exact one-hot neighbor support and minimum similarity-weight
mass, and the proposed action's weighted immediate-reward mean must strictly
exceed the base action's by the configured margin. The default requires one
supporting neighbor, mass `0.1`, and strict positive advantage. It adds zero
persistent state and zero RNG, and disabling it preserves the historical
Prototype branch. This is an associational immediate-outcome check, not an
effective-sample-size/confidence interval, delayed-credit method, causal
intervention estimate, or solution to hidden-context aliasing. An opt-in
Prototype path now queries the next decision
representation before writing the grounded current exemplar, stores a one-hot
of the primitive action that actually ran with bootstrap representation and
reward, composes memory before partner fusion, and rolls back the entire event
when a required memory transaction is unsafe or corrupt. Full lifecycle IDs,
no-memory shape compatibility, checkpoint, curation, eager/JIT/scan, and exact
resource contracts are tested; the latter discloses two deterministic
pre-state queries and zero RNG per required transaction. Retrieval therefore
has not established held-out forward transfer under matched capacity; this
remains L0 integration rather than the WP8 memory exit gate.

`LearnedExperientialMemoryController` now adds a separate bounded learning
owner around that unchanged store. Its seven-feature linear gate can only
reject a retrieval already accepted by the fixed safety/freshness contract.
The controller also owns the nested store's utility channel as a learned
retention estimate, so matching feedback changes real future eviction rather
than only reporting a score. Query, admission, access accounting, and write
are atomic and query-before-write; one pending receipt binds neighbor slots to
insertion clock, provenance, and source so a reused slot cannot receive stale
credit. Learning requires caller-declared use plus a bounded same-decision
counterfactual utility delta. That declaration is integrity-bound but not
authenticated. Exact resources, strict checkpoints, eager/JIT/scan behavior,
negative/positive retention witnesses, and rollback are mechanism-tested.
There is no learned embedding, held-out benefit, or promotion authority.

The separate v1 `ExternalLearnedStateLiveMemoryAdapter` now makes that
controller the sole memory owner around one external full-GRU/router/audit
coordinator whose inner Prototype memory is disabled. Exact prior feedback
settles first; the coordinator updates once; the next raw decision observation
queries the pre-write store; and the current exemplar records the primitive
that actually executed. Only an admitted exact one-hot retrieval may use
Prototype's public cached-action replacement under the caller mask. Its pending
binding preserves the memory transaction, Prototype decision, pre-retrieval
base action, effective action, retrieval action, and exact mask. Soft/fallback
retrievals have no learning authority, and all owners adopt atomically or
return the complete source. The composition is host-orchestrated L0 mechanism
coverage without authentication, dispatch/safety authority, benefit, or
promotion.

For one exact HCCL path, the separate
`HCCLLearnedMemoryFeedbackBridge` now replaces that caller-shaped utility delta
with source-bound eight-proposal attribution. Its pending binding is
specific to one controller transaction, agent, event, B/M receipts, mask, and
categorical retrieval. The learned quantity remains retrieval/retention
utility: it is deliberately distinct from delight or a “gradient sparks joy”
signal and never runs an actor backward pass. This is still L0 integration,
not an integrated live-agent feedback loop or held-out benefit claim.

For the bounded pair-feature lifecycle only, the development-only
`PrototypeFeatureMemory` adapter extends that integration under the exact
`IdentityStateBuilderConfig`. A descriptor-generation change atomically
rebinds all valid stored representations from reconstructable base prefixes
before retrieval or insertion; adapter failure rejects the outer transition.
Its fixed resource declaration and v16 composition-digest checkpoint are
mechanical contracts, not evidence that rebinding improves retention or
prevents catastrophic forgetting. Learned base builders and generated-pair-
tail modeling are unsupported; only the separate stable-base legacy world
model is admitted, while dreaming, replay/ensemble/recurrent models, IA, and
partner-fusion consumers remain disabled.

A separate bounded, development-only recurrence harness declares one
`3 x 512` visible-cue meet/avoid/meet life through a single static Prototype
configuration with linear OaK, a managed linear Horde, pair-feature lifecycle,
feature-bound experiential memory, and a linear
`ActionConditionedWorldModel`. The model consumes only the stable base prefix,
uses action interactions, and owns a capacity-one recent-observation buffer;
generated pair tails are not modeled. Its world-model `gamma=1.0` matches the
continuing environment discount. Its full arm is accompanied by matched
memory-readout, feature-promotion, joint, and cue-masked controls, plus exact-
default visible and cue-masked outcome-gated arms. Every arm
pays for one discarded no-memory preview update and one committed update per
real event, so both paths invoke the model while preview state is never carried
into the life. The strict in-memory report reconstructs exact event/Prototype
clocks, per-event world predictions and A → B → A phase prequential model error,
model recurrence and reacquisition summaries, fixed model/buffer and whole-state
resources (using the ownership key `world_model_bundle_nbytes`), matched logical
work, transaction diagnostics, and causal replay. On consumed seed 0, the
visible gate allowed 59 of 1,536 proposals, with 52 helpful and 7 harmful
action changes and cumulative dispatch delta `+5.0654`. The cue-masked gate
allowed 182 proposals, abstained on 1,354, produced 142 helpful and 40 harmful
changes, and changed the corresponding ungated arm's net-harm diagnosis into
positive cumulative delta `+8.1594`. Each gated arm used 20,733 persistent
bytes, 3,073 exact preview/commit/stale assessments, and no gate-owned random
draws. Focused eager/compiled checks isolate only Horde-derived float32 fusion
drift (maximum trace difference `1.49e-8`) under a declared `1e-7` tolerance;
all dispatch, reward, work, resource, and authority fields remain exact, and
the report makes no full-state digest-equality claim. This exercises the
reporting and negative-transfer guard contracts, not a promoted default-life
result. The partner is scripted, the primary arms see the task cue, and the
gate uses same-context immediate outcomes rather than causal or delayed
credit. Realized compute and allocator residency are not matched, and no
dreaming or planning is performed. There is no artifact writer, threshold,
held-out seed panel, confidence interval, scientific evidence, or promotion
path. The harness remains L0, `not-assessed`, and nonpromoting; it does not
establish general model quality, transfer, catastrophic-forgetting resistance,
planning benefit, or Alberta Plan completion.

`ConsolidatedMemory` now supplies a separate fixed-capacity L0 semantic and
procedural storage core. Semantic records represent typed GVF, fact, or
affordance payloads; procedural records retain skill payloads,
success/failure and outcome moments, and an exact option-lifecycle link. Both
carry SHA-256 semantic/provenance identity, generation, confidence,
source/representation revisions, evidence moments, use clocks, staleness, and
invalidation. Query-before-write is causal, compatible observations merge,
next-generation or changed semantics reset evidence, and deterministic
bounded replacement plus strict source/namespace checkpointing is tested.
One frozen 17-event development companion now runs full memory, an exact
same-kernel retrieval-ablation, and a zero-storage/no-kernel arm from the same
empty source-bound snapshot. It reconstructs retrieval precision, abstention,
harm, recurrence/recovery, retained utility, stale-skill harm,
eviction/provenance, counters/resources, checkpoints, eager/compiled parity,
and integrity-bound causal replay. The report is explicitly `not-assessed`: its exact-
match finite schedule has no threshold, and the no-memory arm is not
storage/compute matched. A separate stateless
`ConsolidatedProceduralMemoryPolicy` accepts an already-produced procedural
retrieval only after exact lifecycle, evidence/count, Wilson success-bound,
outcome-uncertainty, nonnegative score-mass, and caller hard-safety gates. It
can propose only the lowest-index safe positive-mass action and owns no query,
write, RNG, dispatch, mutation, checkpoint, or promotion authority.
`PrototypeConsolidatedMemoryAgent` now supplies the explicit live composition:
pending procedural feedback settles first, Prototype learns the actual action,
and only then does the shared store propose a next cached primitive after
experiential memory and partner fusion under intersected safety masks. A
separately versioned semantic wrapper uses the same controller state, queries
before writing, and feeds an accepted prior payload—or an exact zero tail—into
the ordinary next Prototype context before that procedural query. A separate
post-envelope settlement binds the exposed decision, selected primitive, and
exact mask. An admitted unchanged action is a state no-op; a changed admitted
fallback atomically rebinds Prototype's actual credit owner and cancels only
matching procedural and partner recommendation owners. No-action, stale,
disallowed, corrupt, or partially satisfiable settlements preserve the whole
state for retry and write no learner, memory, or reliability evidence.
Missing, stale, rejected, exhausted, or serialized memory never freezes valid
base control; persistent corruption fails closed. The separate learned
experiential controller is not yet composed here and its caller-supplied
counterfactual feedback is unauthenticated. These mechanisms still have no
transfer, negative-transfer, stale-skill, semantic-utility, safety, or
physical-dispatch benefit result, so the WP8 exit gate remains open.

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
tested. A separate strict `not-assessed` stress lane now supplies learned,
outcome-blinded, and base-only conditions the same frozen 96-event
multi-partner stream, fixed state/message shapes, and exact per-event
decision/feedback call counts. It retains the complete causal trace across an
observable-context reliability reversal, communication costs and spikes,
partner-specific and total disconnects, and hard-mask exclusions; exact replay
and prefix checkpoint/resume fail closed. This is a development description,
not confidence calibration, a matched closed-loop Prototype benefit, a
multi-seed result, or the WP8 exit gate. A separate consumed 12-execution
evaluator now runs learned, outcome-blind, and base-only conditions through
three independently owned real `PrototypeAgent`, fusion, environment,
authority-receipt, and trace-chain lifecycles. Only frozen exogenous context,
noise, drift, availability, costs, and hard-mask candidates are paired; each
arm's actions generate its own later observations, rewards, messages, and
feedback across the hidden reliability reversal. The evaluator validates real
action-changing updates, realized action-relative assistance, caller-owned
mask enforcement, exact causal-prefix replay, in-memory resume, update-boundary
eager/JIT parity, and matched logical call/shape budgets. It writes no output
and remains consumed L0 `not_assessed` instrumentation. Because learner states
and RNG streams are independent and the life has only 12 executions, its
between-arm summaries are descriptive rather than a causal partner-benefit or
calibration result. A separate v2 evaluator removes that initialization
confound: learned-feedback, fixed-zero outcome-blind, and empty-message
base-only wrappers begin with bit-identical typed RNG, Prototype, fusion, and
environment state, receive one paired exogenous schedule, and own their later
causal trajectories after actions diverge. Exact prefix reconstruction, raw
hash chains, replay, checkpoint/resume, source/runtime/config binding,
eager/JIT parity, and matched logical work are validated. In the consumed
12-event trace, learned and fixed-zero each changed action three times and had
the same task/net return, while base-only changed none; learned and fixed-zero
internal states differed without realized behavioral separation. This is a
null descriptive L0 result with no threshold, winner, artifact, efficacy, or
promotion authority.

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

Three later, in-memory-only target-only development falsifications refine that
failure without altering or promoting the pinned artifact. First, a fixed
`9 x 256` microcycle made target-only relevance evidence causal to the learned
target, but remained a valid rejection: C had only 191 post-acquisition
observations and failed the unchanged learning/use gates. Second, an exact
31-transition stale-retirement cadence moved D's retirement from step 1,791
to 1,673 while preserving at least 11 of 12 live slots, yet both matched arms
already met final absence and D failed the unchanged learning gate; cadence
alone was therefore rejected as the explanation. Third, one fresh derived
development seed doubled every segment to 512 and geometrically doubled the
time-based retention grace from 640 to 1,280. That run was structurally valid
and passed every unchanged lifecycle-v2 gate: C was acquired at step 2,689,
had 383 remaining first-exposure observations, reached first-late reward
`0.8671875`, survived continuously, and returned at early reward `0.828125`;
D was acquired at 1,601, retired with an exact linked reset at 3,391, and
remained absent. Mean reward was `0.8220486`.

The doubled-horizon result supports an observation-horizon hypothesis on one
consumed development seed; it is not a paired causal estimate, robustness
result, or scientific evidence. All three successor modules forbid output
writes and expose no promotion or seed-search path.

The artifact is explicitly `development_only` with
`scientific_promotion_allowed: false`. Its partner is scripted and
nonlearning, its schedule family is fixed, and discovery searches a closed
exhaustive 66-pair archive. It therefore establishes an L0 integrated
development and falsification rung, not learning-partner coadaptation, general
feature finding, an L2 comparison, or L3 Alberta integration.

### Hidden-regime and hidden-partner protocol machinery (design-only)

Several evaluation lanes are complete as fail-closed validators but have
never executed an outcome. The hidden-regime factorial track
(`evaluation/hidden_regime_factorial_*` plus its governance, readiness,
checkpoint, trace-audit, and oracle modules) defines a 240-case calibration
design, threshold freeze, and protected evaluation plan with
`PROTOCOL_STATUS = "calibration_design_frozen_outcomes_unexecuted"`; no
hidden-regime artifact exists under `outputs/`, the protected plan is
`preregistered_unexecuted`, and no execution issuer is available. The older
six-condition signaling evidence draft is execution-disabled
(`draft_execution_disabled_pending_factorial_boundary_protocol`): its CLI can
validate pre-existing files but its plan and run commands always fail. The
three protected structural-generalization manifests in
`streams/hidden_regime_signaling.py` record that no protected-candidate
learner outcome has ever been executed, and the signaling development runner
hard-codes `SCIENTIFIC_PROMOTION_ALLOWED = False`.

On the hidden-partner side, both reserved lifecycle namespaces — v4 lease
tuning and the frozen v5 confirmation candidate — are `FORBIDDEN/UNEXECUTED`;
their run and verify tooling fails closed on every path, so that grid and
confirmation plan are retired protocol machinery, not pending work. The v6
lifecycle-world lane has runner, runtime, validator, matched-suite, and
source-replay modules plus 18 bound control arms, but its certification gate
lists every open prerequisite as `NOT_CERTIFIED` and no v6 life, calibration,
threshold, or artifact has been produced. All of this is pre-evidence
infrastructure, not results.

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

Two sibling development lanes are likewise nonpromoting by construction. The
IPMNIST mechanism-combination record (`benchmarks/ipmnist_screening.py`,
`outputs/ipmnist_screening/`) contains 144 screening shards (48 arms × 3
seeds) and 69 full-horizon confirmation shards, but a 2026-08-02 read-only
audit rejected its campaign provenance. Its own proxy receipt has
`proxy_validated: false`: all three UPGD controls miss their claimed
full-horizon prefixes by maximum per-task differences of `0.0084`–`0.0096`.
The aggregate summary is stale at 132 shards, the round-2 driver fails on a
nonexistent aggregate field and an empty confirmation invocation, and its
later result file cannot be emitted by that driver. Stored per-task curves do
reconstruct the reported means (including the round-2 `0.98` vs `0.99` delta
of `+0.0008356` and the noisy-transplant delta of `-0.0018589`), but v1 shards
bind no source, command, dataset, or exact manifest. These are historical
hypothesis-generating measurements, not an authenticated current-source
campaign and not a validated 60→200-task result.
The Label-permuted EMNIST diagnostic (`benchmarks/upgd_label_emnist.py`,
`outputs/upgd_label_emnist/results.v1.json`, 3 seeds versus the paper's 20)
reproduces the published qualitative separation — UPGD-W online accuracy
rises to a `0.728` last-quarter mean while AdamW collapses to a `0.201`
whole-run mean — with both whole-run gaps against approximate figure
read-offs explicitly flagged. Both lanes declare
`scientific_promotion_allowed: false`.

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

The current matched-source campaign toward a paired Alberta-versus-baseline
comparison uses candidate-universe schema v2 (SHA-256 `6a9315cb…`) but has no
renewed qualification/open artifact; a fresh qualification and new output
namespace are required before execution. The immutable historical v1
candidate universe (SHA-256 `2c3b214c…`, 23 registered arms: 14 Alberta
candidates, 7 external inferential comparators, and 2 descriptive references)
passed executor qualification
(`outputs/forager/matched_current_qualification_2c3b214c_v1`), and the
open-tuning stage (`matched_current_open_tuning_2c3b214c_v1`) publishes its
complete immutable manifests — but its `runs/` and `completions/` directories
are empty: zero tuning cells have executed. The sealed held-out stage (seal,
6×30 evaluation schedule, final analysis, and statistics modules) is
implemented and contract-tested but has never run and has produced no artifact.
The evaluation runner is exposed as `alberta-forager-matched-sealed-evaluation`;
seal and final analysis remain module-only. Every authority-bearing path
terminates at an external trust resolver that does not exist in-tree, so in-tree
code alone cannot produce promoted evidence. The 2026-07-31 internal candidate audit
(`FORAGER_ALBERTA_CANDIDATE_AUDIT.md`) recorded implementation GO but
campaign authority NOT CLEARED, and its recommended seed-0 q-grid divergence
probe (`forager_causal_grid_divergence_probe.py`) is built but unrun.
`FORAGER_COMPARATOR_AUDIT.md` records that the panel is a matched-panel
comparison — several comparator arms are explicit task adaptations, not exact
upstream replays — and that historical v1 cannot support its descriptor's
"best among the panel" wording. Candidate-universe v2 removes that wording and
explicitly limits interpretation to the three preregistered contrasts; neither
version licenses a panel-wide winner claim.

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
validates against the current checkout: registered-source drift (first
observed in `alberta_framework/core/average_reward.py`) makes the live
evidence-registry compatibility chain `invalid`. This does not alter the
archived historical v1 rejection, and v1 does not certify the current
implementation.

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

## Scaling contract

Scaling must be stated in terms of explicit capacity knobs, not inferred from
the absence of Python-side growth in a short run. The new
`measure_prototype_agent_state_resources` API measures every persistent JAX
array in a concrete Prototype state and partitions the bytes by top-level
owner without double-counting shared feature/Horde or interaction/memory
bundles. It includes legacy timing scalars when Prototype initialization
materializes them as JAX leaves, while their values remain outside
learning-state identity semantics. Compiler workspaces and transient arrays
are excluded; latency and peak device memory therefore remain separate
measurements.

The dominant configured terms are:

- dense learner/state-builder parameters and traces scale with adjacent layer
  products; adding optimizer statistics multiplies those parameter-shaped
  terms by a method-specific constant;
- a linear Horde scales as `O(number_of_demons * representation_width)`, while
  shared nonlinear trunks add their layer products once plus one head per
  demon;
- ensemble world models scale linearly in member count and per-member model
  size; explicit joint-action tables scale with the product of agents' action
  counts and therefore require factorization before many-agent use;
- fixed-capacity experiential/replay memory scales linearly in capacity and
  stored representation/transition width;
- the birth-bound recurrence history used by the selective-retention probe is
  exactly `32K` persistent bytes per agent for `K` context slots and requires
  `O(K)` scoring at each step. The current `K=3` dyad adds 96 bytes per agent
  plus 48 logical transient bytes for both raw and dispatched score vectors;
  it stores no replay, but it also forgets recurrence value when a semantic
  birth is evicted;
- an explicitly enumerated pair-feature universe is quadratic in base width,
  even when active and candidate banks are capped; higher-order exhaustive
  enumeration is not a credible scaling route, so proposal generation must be
  sparse, sampled, or compositional;
- the compositional adapter's identity binding is exactly
  `32 * bank_width + 44` bytes. Re-encoding `R` stored rows evaluates exactly
  `2 * R * bank_width`
  feature slots, and authenticated prepare/commit evaluates the complete learner
  twice while advancing it once. The current silent-task audit also materializes
  candidate-by-destination diagnostics and multi-megabyte host traces, so its
  2,072-byte persistent learner state must not be confused with realized peak
  evaluator or compiler memory;
- for `N` generated slots and `M` managed linear-consumer scalars, the isolated
  consumer route checks `3M` scalar positions and performs two representation
  calls (`2N` feature-slot evaluations); prepare plus recomputing commit doubles
  those figures to `6M`, four calls, and `4N`. The captured curation permission
  adds one transient boolean byte and no persistent state;
- the generated-input/fixed-output world-model fixture (`B=3`, `N=2`, two
  actions, `H=5`, three physical anchors, one OaK option) owns 529 persistent
  bytes: a 388-byte model core containing the 340-byte learner, a 44-byte
  anchor buffer, and 97 wrapper/binding/digest bytes. Its source OaK snapshot
  is 496 bytes, prepared real-transition cache 674 bytes, and prepared plan
  cache 1,258 bytes. The 80-byte generated-input increment is `8HN` for weights
  plus traces. These full-snapshot cache sizes are transient transaction costs,
  not persistent capacity, and their linear pair-feature geometry does not
  cover nonlinear, ensemble, partner, or multi-step models;
- the Alberta-derived permanent/transient diagnostic is exactly 788 persistent
  bytes for its one-input, one-output, 32+32-hidden configuration: 388 bytes
  per parameter subtree plus a 12-byte exact-clock/telemetry state. It performs
  two gradients and three internal forwards per update with zero replay and
  fixed capacity; this accounting does not make it work-matched to FastSlow;
- the two-expert latent-context diagnostic has 32 persistent bytes (16
  parameter bytes plus 16 owner/clock bytes) and a 45-byte pre-outcome cache.
  It evaluates four expert predictions including cache authentication, two
  losses, and two analytic candidate gradients per event, with one subtree
  commit, zero replay/RNG, and no capacity growth;
- its fixed two-event pairwise-dominance successor has 53 persistent bytes and
  a 70-byte pending cache. The 1,536-event consumed life performs 6,144 expert
  predictions including authentication, 3,072 losses, and 3,072 candidate
  calculations with zero replay/RNG; four quarantine openings deliberately
  make no parameter commit. Fixed `H=2` is bounded work, not evidence that this
  delay or two-expert capacity scales to open-ended contexts;
- the one-record cross-birth rescue sidecar is 161 bytes per agent (322 bytes
  joint) and raises the hidden-rule scan carry from 978 to 1,300 bytes. Its
  fixed 4,000-step protocol projects bounded exact rescue words to float32
  priorities without loss; a longer-life/general implementation must compare
  exact words directly rather than assume that projection remains ordered;
- the standalone sequential-lineage successor is 563 bytes per agent and
  1,126 bytes joint at the same `K=3`, four-action, four-observation geometry,
  raising the measured 962-byte base carry to 2,088 bytes once composed. Its fixed
  `H=2` bank has five predictions per agent/event and freezes one opening
  candidate plus every live comparator; the core's SHA-256 integrity work is
  explicit. The wrapper matches outer calls and RNG across its two conditions,
  while its full 4,000-step behavioral comparison is not yet run or claimed;
- the hidden-context two-Prototype-agent U0 state is exactly 41,718 persistent
  JAX-array bytes in both arms. Staging four environment successors, two
  context successors, two no-memory previews, and two memory-sidecar Prototype
  candidates has an 83,328-byte full-state-copy lower bound before diagnostics,
  compiler workspaces, or allocator residency. Its 1,536 events call the
  environment 6,144 times, context inference 3,072 times, and Prototype update
  6,144 times, while only two Prototype and two context successors persist per
  event; this is bounded fixed geometry, not a FLOP, latency, or many-agent
  scaling result;
- the factorized U1 planner adds 3,758 persistent bytes to the canonical
  two-agent Prototype life and retains four environment proposals per event.
  Its planner-only rows are `PP`, `M0P1`, `P0M1`, and `MM`, with post-memory
  `M` as the fallback; memory query/write/eviction diagnostics remain present,
  but no same-event memory reward attribution is reported. It evaluates all
  four ordered own/partner action cells per agent, updates two behavior and two
  grounded models, and adds zero post-initialization planner RNG draws or
  replay. These are one-step joint-cell evaluations, not multistep planning
  backups. The 1,536-event three-arm runner is frozen but
  unexecuted. On one stable dependency snapshot, 17 focused core/wrapper cases
  passed and the sole failure was an over-strong whole-composite bit-exact
  float assertion. A 589-leaf audit found 17 float leaves with 1--2 ULP drift
  and exact discrete/key leaves; the corrected tolerance-based parity case
  passed separately, but `prototype_agent.py` changed during that run. No
  current-source 18/18 claim exists, and stable-source verification remains
  required;
- planning work scales with the fixed backup/candidate budget per decision,
  and option work with the number of live options and their bounded model
  updates; and
- every exact two-word lifetime identity is `O(1)` state. It removes premature
  32-bit wrap but does not remove configured estimator, archive, replay, or
  model-capacity limits.

Every integrated run must publish both the measured ownership partition and
the configuration-derived upper bound, then show that initial, peak, final,
and checkpoint sizes agree within the declared transient-memory semantics.
Scaling a toy dyad to many agents also requires replacing full joint-action
enumeration, measuring communication and inference cost, and testing partner
identity turnover rather than merely lengthening the same two-agent schedule.
The new U0 dyad makes the first cost explicit: its four actual/base joint
proposals are the complete `2^2` factorial needed for exact own, partner, and
interaction effects. Extending that exact decomposition to `N` agents costs
`2^N` environment proposals per real transition. A scalable successor must
predeclare either `N+1` actual-plus-unilateral proposals for direct own-action
effects, an `O(N^2)` bounded pairwise-interaction budget, or a fixed-size
randomized coalition/Shapley estimator; it cannot report full interactions
while silently paying exponential work. Independent per-agent Prototype,
context, and memory state otherwise scale linearly only while models and
policies remain factorized and partner identity is not copied into a joint
table. The implemented dyad attribution kernel realizes the two-layer bound;
any scaled HCCL successor would generalize it to
`L * 2^N` calls across `L` adjacent action layers, with at most
`1 + L(2^N - 1)` action-receipt vertices. Its dyad/two-layer core is eight
calls, seven designated counterfactual slots, at most seven receipt vertices,
and at most four effective joint actions; separately attributing a third
communication layer would require 12 calls and at most 10 receipt vertices.

## Live 18-row completion audit

This table separates implementation progress from scientific completion. A
mechanism or development diagnostic is not an accepted scorecard row. The
strict `complete_prototype_manifest` still has no registered evidence index,
so all 18 L3 rows remain open.

| Property | Current mechanism/development state | L3 blocker |
|---|---|---|
| Continuing operation | Exact-clock/atomic contracts cover the main pipeline, Prototype, and several recurring environments | One all-enabled uninterrupted lifetime plus the remaining legacy-counter audit |
| Temporal/resource bounds | Component budgets and concrete whole-Prototype state-byte partition exist | Whole-run peak memory and p50/p95/p99 deadline measurements |
| Plasticity | Source-profiled UPGD, separate official-source and guarded-derived first-order AdaUPGD APIs, continual backprop, Self-Normalized Resets, isolated spectral-regularization/AdamO/Calibrated-Partial-Reset dense-layer arms, Autostep/IDBD variants, OCP task-crossed probes, an isolated C-CHAIN Equation 8/NTK comparator, exact-horizon FastSlow and Alberta-derived Permanent/Transient negative recurrence controls, a one-sample active-only expert partial/null comparator, and a fixed `H=2` quarantine with clean consumed-root dormant-expert retention exist | Held-out later-regime trend against fresh/shallow matched baselines; the three newest optimizer mechanisms and C-CHAIN are not generic agent-integrated matched arms, both timescale splits overwrite A, and the successful quarantine is one synthetic root with two supplied experts and post-outcome switching |
| Retention | Selective forgetting, recurrence, lifecycle, memory, and actor/world probes exist; the fixed two-event quarantine preserves one learned expert bit-exact through the interfering phase on a consumed A/B/A life, a birth-bound completed-recurrence score changes two later evictions with a small consumed-root benefit, while a strict one-transition cross-birth archive is null and prefix twins plus compositional headroom/left-pack siblings expose unresolved cold-start and prospective-value failures | Prespecified whole-agent forgetting and worst-task lower bounds with bounded sequential evidence, a causal future-recurrence/value estimator across semantic rebirths, or a declared prior; the clean expert result is not generated-feature or control retention |
| Transfer | Experiential/consolidated-memory and feature-transfer development probes exist | Paired held-out forward-transfer interval |
| State construction | Fixed trace, working memory, temporal context, online-gated, dense full-GRU, and compressed-RTU RTRL builders, a causal whole-unit RTU generate-and-test lifecycle, live smaller, caller-targeted comprehensive, and learner-owned causal-target Prototype transactions, strict-linear whole-unit replacement consumers, and POMDP diagnostics exist | Calibrated objective masses, target quality, optional-cumulant semantics, broader lifecycle compatibility, and matched recurrent comparison, plus partially observable Forager and robot-simulation outcomes |
| Prediction | Exact independent/mixed/stacked Horde and multi-timescale machinery exist | Calibrated and decision-useful held-out GVF result |
| World model | Dense, latent, recurrent, grounded, FTL, ensemble, replay, recurrence diagnostics, an exact stable-base Prototype/feature-lifecycle lane, and a v18 atomic Prototype route for the generated-input/fixed-physical-output linear model and memory exist | Retention, uncertainty calibration, partner-aware dynamics, utility-curation feedback, and real rollout/decision validation together |
| Planning | STOMP/OaK, option search, legacy dreaming, one-step partner planning, isolated ensemble Dyna, guarded short rollouts, a direct planner→gauge→actor/critic L0 transaction, and one v18 Prototype-integrated generated-input model→OaK base backup exist | Externally calibrated grounding and matched primitive/option planner benefit under a fixed backup budget; the new backup remains default-off |
| Exploration | Epsilon, uncertainty/surprise, option-search, prospective-improvement selectors, and a consumed causal six-arm stochastic-trap development lane exist | Held-out coverage/return uplift plus noisy-TV resistance under matched physical resources |
| Feature lifecycle | Bounded generated/pair/deep mechanisms exist; one opt-in v18 Prototype route now gives the pair bank atomic old-bank OaK/Horde learning, fixed-output world learning, exact memory rebinding/use, all-consumer curation adoption, checkpoints, and an optional one-backup planning consumer | Utility-auditor/curation coexistence, selective-retention benefit, and held-out random-curation comparison under the same resources |
| Skill lifecycle | Bounded unlabeled proposal scheduling reaches live STOMP installation; modeling, search, exact caller-authorized retirement/replacement, and a one-owner Prototype→OaK→STOMP bridge exist | Autonomous go/no-go and repeated lifecycle policy, caller authentication, held-out control benefit, and a successful uninterrupted repeated-lifecycle outcome |
| Candidate-update audit | Candidate-gradient mixing/auditing, paper delight/Kondo, and causal utility auditors exist | Realized-outcome and measured-compute guardrails for every enabled optional path |
| Experiential memory | Fixed-capacity memory, conservative policy, Prototype integration, exact-identity feature-bank rebinding, negative-transfer diagnostics, a caller-feedback-bound learned admission/retention controller, and a separate one-owner live external-coordinator adapter exist | Integrating source-bound causal feedback into that live adapter, caller authentication, held-out transfer benefit, and a negative-transfer bound |
| IA | Behavior/partner/world models, fusion, and Prototype IA paths exist | Causal benefit under drift/cost; the historical frozen gate was a valid rejection and is now source-invalid |
| Checkpointing | Versioned migrations and parity tests cover many components and Prototype schemas | One all-enabled exact/tolerance-defined resume contract |
| Safety | Fail-stop atomic learners, a synthetic embodied safety envelope/fault audit, a source-bound semantic Prototype command adapter, and a bounded plant/shadow harness with real multi-step Prototype successor learning exist | Dynamics/geometry validation, held-out adaptation, caller authentication, and physical exit gates |
| Reproducibility | Strict registries, validators, manifests, hashes, runbooks, and negative-result records exist | Clean-checkout reproduction of a frozen whole-agent campaign; all five current narrow claims are source-invalid |

The shortest critical path is therefore not another isolated component: it is
an all-enabled development harness with matched ablations, followed by a
separately frozen and externally anchored campaign only after that harness has
stopped changing. The current mechanism work is necessary preparation for that
run, not a substitute for it.

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
completion certificate. At this snapshot it exits `2` with all five claims
`invalid`, per the registry snapshot at the top of this document.

## Relation to LeWorldModel

[LeWorldModel](https://arxiv.org/abs/2603.19312) is an
action-conditioned joint-embedding predictive world model. Its learned visual
encoder, predictor, and anti-collapse regularization are relevant candidates
for Steps 8–9. It is not itself a continual learner or an agent/partner model:
the current v3 protocol trains for repeated epochs over fixed offline datasets,
its reported controller performs short-horizon CEM/MPC only after training, and
it does not test recurring-task retention or online representation drift. The
paper's two-term next-embedding/SIGReg objective is therefore an anti-collapse
comparator, not evidence for continual learning. Adoption requires online
bounded updates, latent-drift/retention probes, matched replay-free and
fixed-encoder ablations, and a separate behavior model for other agents.

A development-only, nonpromoting
`latent_world_model_recurrence_development` evaluator now supplies the first
small online bridge. It feeds fixed-encoder, prediction-trained-encoder, and
collapse-gated-encoder arms one uninterrupted hidden `A -> B -> A` stream on a
persistently excited unit-circle state manifold. The regimes apply opposing
action-conditioned rotations without resets or learner-visible phase IDs, so
the recurrence window measures genuine interference rather than decay toward
an easy zero signal. The evaluator records prequential latent/reward/discount
errors, entry forgetting, recovery and residual forgetting, exact common-input
and initialization bindings, matched persistent resources, update/gate rates,
phase SIGReg diagnostics, and matched physical-versus-nuisance surprise. It
has no thresholds, seed search, artifact writer, or promotion route. SIGReg is
diagnosed but not optimized, the default run is one low-dimensional
initialization, and the negative physical-versus-nuisance separation observed
in current development runs is a counterexample to treating raw latent
surprise as grounded physical understanding.

The separate generated-input linear lane closes a different, more mechanical
bridge: a changing discovered representation can condition fixed physical
next-state/reward/discount heads and an OaK backup without routing mutable
feature identities as prediction targets. It is online and bounded, but it is
not a LeWorldModel reproduction: there is no visual encoder, joint-embedding
objective, SIGReg training, stochastic uncertainty, multi-step MPC, partner
behavior model, or matched outcome. The two lanes should meet only after the
fixed-output route has a calibrated nonlinear/latent comparator and the
factorized world-and-agent model has explicit recurrence retention tests.

## External baselines to reproduce

- [Toward a New Approach to Model-based Reinforcement Learning](https://www.incompleteideas.net/papers/MBRL2.pdf):
  test the proposed distinction between feature utility for performance and
  utility for learning, including slow descendant-to-ancestor utility
  propagation. The current novelty-admission and ancestor-max-backup paths
  implement only a small fixed-grammar fragment; they do not yet discover
  recurrent state features, GVF questions, or option models.
- [Auxiliary task discovery through generate-and-test](https://proceedings.mlr.press/v232/rafiee23a.html):
  compare learned auxiliary-question utility against the present hand-fixed
  action-reward heads. A generated algebraic feature is not equivalent to an
  autonomously selected predictive question.
- [What Should I Know?](https://proceedings.mlr.press/v199/kearney22a.html):
  add a meta-gradient predictive-feature arm that learns which GVFs support
  control from one stream, especially in the hidden-context and partial-
  observation lanes.
- [Permanent and Transient Representations for Continual Reinforcement Learning](https://openreview.net/forum?id=5XfxEQ2SCt):
  the new Alberta-derived online regression baseline tests separate
  fast/transient and slow/permanent representations but deliberately departs
  from the paper's task-buffer and Craftax mechanisms. A source-faithful
  reproduction and a learned dormant/reactivation comparison remain open; the
  current always-active permanent path is overwritten on the consumed A/B/A
  diagnostic and is not a settled default.
- [Statistical Context Detection for Deep Lifelong Reinforcement Learning](https://proceedings.mlr.press/v274/dick25a.html):
  compare its sliced-Wasserstein/KS action-reward context detector and delayed
  policy rollback against Alberta's fixed-bank hidden-context lanes. SWOKS is
  online and label-free at inference, but uses sliding data/distance windows,
  a significance threshold, a stable-phase rule, multiple policies, and
  periodic rollback; it is therefore a larger calibrated comparator rather
  than justification for relabeling a current prediction after its outcome.
- [Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes](https://arxiv.org/abs/2006.11441):
  compare its sticky sequential variational assignment and expert reuse on a
  bounded changing-dynamics lane. The published method supplies the relevant
  transition prior, but dynamically grows/merges/prunes GP experts, retains
  representative data, and exposes concentration, stickiness, merge,
  distillation, and inducing-point parameters. A fixed-resource Alberta arm
  must predeclare how those capacities are capped and report assignment delay.
- [Continual Reinforcement Learning by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html):
  fixed sparse random features plus online Follow-The-Leader ridge statistics
  are a replay-free dynamics baseline. Its shared-stationary-dynamics
  assumption must be tested separately from genuinely changing physics.
- [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7):
  continual backpropagation, dormant-unit fraction, weight magnitude, and
  effective rank belong in every deep learner sweep.
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://proceedings.mlr.press/v267/tang25g.html):
  the exact Equation 8 churn objective, appendix coefficient ratio, and
  empirical-NTK rank diagnostics now have an isolated clean-room L0 surface.
  A full sequential algorithm and matched outcome comparison remain missing.
- [CORA](https://proceedings.mlr.press/v199/powers22b.html): continual
  evaluation, isolated forgetting, and zero-shot forward transfer complement
  the task-matrix metrics implemented here.
- [MEAL](https://openreview.net/forum?id=Mxg6mo1Xzj): after the tiny two-agent
  gate is reliable, the scaling lane should include long cooperative
  multi-agent task sequences rather than treating a three-segment toy as the
  final benchmark.
- [Nevo-CRL](https://openreview.net/forum?id=Hv0jK8xYcT): compare its
  fixed-capacity task-specific connectivity masks and neuro-evolutionary
  population against Alberta's bounded expert/feature lifecycle only in a
  task-aware comparator. Its ICML 2026 result does not establish a label-free,
  single-life Alberta default.
- [HTAC](https://openreview.net/forum?id=akfJfpUEBj): hierarchical task-aware
  composition is a current modular-transfer comparator for offline continual
  RL, but its task encoding and offline data boundary do not answer Alberta's
  online hidden-regime problem.
- [Newt](https://openreview.net/forum?id=MPabX9LEds&noteId=W8pvGCVoVm): its
  200-task language-conditioned world-model benchmark is a useful future scale
  reference for cross-embodiment transfer. Demonstration pretraining, language
  task cues, and multitask online training make it neither a matched baseline
  nor evidence for the present reset-free continual agent.
- [CRL-VLA](https://arxiv.org/abs/2602.03445): its frozen/training dual-critic
  comparison is a relevant 2026 stability--plasticity arm for embodied
  policy post-training. Goal conditioning, a pretrained vision-language-action
  policy, and task-family evaluation make it an external comparator rather
  than support for Alberta's label-free, from-stream control claims.
- [CRoSS](https://arxiv.org/abs/2602.04868): use its high-diversity robotic
  simulation suite only after the current bounded software composition has a
  stable policy interface. A scalable simulator can expose longer-sequence
  failures and embodiment transfer, but it does not replace the repository's
  fixed-memory, no-boundary, safety-envelope, or physical exit gates.
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
- [Rethinking Plasticity in Deep Reinforcement Learning](https://arxiv.org/abs/2603.21173):
  cross task switches with gradient-flow and constrained-parameter diagnostics
  so dormant fraction is not mistaken for task-independent capacity loss. This
  is a March 2026 single-preprint hypothesis, not a validated default.
- [ProgAgent](https://arxiv.org/abs/2603.07784): compare its privileged visual
  progress reward, adversarial non-expert regularization, coreset replay, and
  synaptic-intelligence PPO on the authors' stated ContinualBench/Meta-World
  setting before treating its reported robot results as relevant to an
  ordinary-reward Alberta agent. Expert-video access makes it a separate
  comparator, not a matched baseline.
- [The World Model Remembers, the Actor
  Forgets](https://arxiv.org/abs/2607.19749): separately probe world-model,
  value, and actor retention. Its new three-seed result suggests graded dream
  self-imitation as an experimental actor-rehearsal condition, but requires a
  larger preregistered reproduction before adoption.

## Immediate closure order

1. Treat the one-seed `9 x 512` hidden-partner pass as a mechanism result only.
   Predeclare a disjoint paired development panel that varies observation
   horizon while preserving the grace geometry and unchanged lifecycle gates;
   quantify seed variability before freezing any new untouched held-out
   protocol. Keep this scripted-partner rung distinct from learning-partner
   coadaptation.
2. Keep the p=0.75/seeds-60–89 IA v2 development contract unissued until its
   source/runtime envelope is settled and execution is explicitly authorized.
   V2 cannot promote; a future evidence protocol requires a new schema,
   untouched seeds, and an externally anchored pre-run plan. V1 consumed-seed
   selection and replay records remain nonpromoting.
3. Promote planning, option, and prototype integration diagnostics only after
   disjoint seeds, paired intervals, matched budgets, and versioned artifacts.
4. Continue lifting or deliberately replace the current Prototype
   compatibility boundary. The shared feature-lifecycle lane now supports
   experiential memory with atomic row rebinding and the legacy world model on
   stable base coordinates, both only with the exact identity builder. The v18
   lane additionally routes the generated tail through a fixed-physical-output
   linear model and the same feature-bound memory under one owner. It still
   rejects learned base builders and general dreaming, replay, ensemble/
   recurrent models, IA, and partner fusion because those representations are
   not yet reconstructable or routed atomically. Then run
   one resource-matched, uninterrupted whole-agent harness with adaptive
   optimization, learned state, autonomous feature targets, bounded feature/
   skill/memory lifecycles, Horde prediction, world/partner models, and
   planning feedback together. The visible-cue feature-memory recurrence
   harness is a useful partial integration, but no configuration or joint
   report currently exercises every path under one clock, checkpoint, latency
   budget, and recurrence schedule.
5. Resolve the matched-current campaign's external authority blockers (trust
   resolver and executor-receipt verification) and run the seed-0 q-grid
   divergence probe before spending the open-tuning budget; only then execute
   open tuning and the sealed held-out stage under the frozen manifests, with
   the matched official baseline under the same accepted runtime contract.
   Never resume the quarantined directories or promote from execution
   manifests, development receipts, or incomplete reports.
