# Continual-learning evidence map

What this repository can currently exercise, organized by Alberta Plan
property. Mechanism tests, development diagnostics, and promoted held-out
evidence are labeled separately; a passing calibrated test is not
automatically a scientific replication. Run a listed test with

```bash
.venv/bin/python -m pytest tests/<file> -q -o addopts=""
```

To inspect every registered promoted artifact without rerunning its scientific
protocol, run `alberta-evidence-status`. It returns `0` only when all
registered claims are accepted, `1` for valid rejection or missing evidence,
and `2` for invalid evidence. The resulting manifest is an index of narrow
claims, not a completion certificate.

Claims are scoped honestly: **mechanism diagnostic** = a controlled component
demonstration (possibly with an oracle representation); **integrated** = the
property shown in a closed-loop or multi-component agent; **autonomous** = no
oracle machinery, discovered by the learner itself.  Known gaps are listed at
the end — this is an evidence map, not a completion certificate.

At this snapshot (2026-08-02) the live registry exits `2` with **all five
registered claims `invalid`**: the recurring pair-feature, scale-robust
pair-feature, and two-agent coadaptation claims fail
current-registered-source validation, and the FTL decision-fidelity and
intelligence-amplification historical compatibility chains fail, because
registered source files were edited after the artifacts were pinned. The
immutable artifacts retain their historical outcomes; none certifies this
working tree. The FTL validator additionally finds drift outside its permitted
artifact-builder-only historical reconstruction, while the IA validator finds
current canonical controller-budget mismatches for all stored conditions.
Those are invalidity diagnostics, not rerun outcomes. Renewal requires
rerunning each frozen protocol to a new artifact path and schema version with
untouched preregistered seeds.

The separate `complete_prototype_manifest` contract enumerates all 18 final
scorecard properties and fails closed unless every required role points to
immutable artifact bytes plus a trusted validator receipt for frozen L3
scientific evidence, the same pinned prototype configuration, untouched
held-out seeds, the exact role and protocol/scientific-outcome digests, and a
source-hash closure. Row and aggregate statuses plus the manifest self-digest
are reconstructed before its exit code is trusted.
Optional paper delight or Kondo use adds mandatory actor-learning and measured-
compute guardrail roles. The repository intentionally supplies no default
binding, so this accounting contract cannot relabel the mechanisms below as a
complete prototype.

## Property → evidence

| Property | Evidence (measured, median unless noted) | Scope | Test |
|---|---|---|---|
| Tracking beats any fixed step-size | IDBD drift-segment MSE 0.044 vs best swept fixed-α 0.155 (3.5x) | mechanism | `test_gauntlet_certification.py` (P1) |
| Step-size relevance | every relevant dim's learned α above every irrelevant dim's | mechanism | `test_gauntlet_certification.py` (P2) |
| Plasticity (event recovery) | abrupt-switch recovery in ~620 steps; 2nd event not slower | mechanism | `test_gauntlet_certification.py` (P3) |
| Plasticity (no decay over 60 blocks) | adaptation ratio: SGD 0.30 vs UPGD 1.16 / CBP 1.05 | mechanism | `test_plasticity_gate.py` |
| Source-profiled first-order UPGD/AdaUPGD equations | Named paper/README/experiment UPGD profiles remain separate. `OfficialAdaUPGD` is bound to the released RL source commit/path and preserves its corrected first/second moments, raw-utility maximum, outside-denominator noise, two-alpha direction, one-alpha decay, and zero-division/non-finite propagation; `AlbertaAdaUPGD` is an explicitly derived guarded extension | L0 equation/contract coverage only; fixed-noise finite/all-active/single-group parity does not establish PyTorch RNG identity, plasticity, retention, optimizer superiority, default selection, or SOTA | `test_canonical_upgd.py`, `test_canonical_adaupgd.py` |
| Self-Normalized Resets implementation | A fixed-width dense-ReLU consumer keeps exact per-unit ages and a bounded completed-gap ring, evaluates the positive-support geometric observed-run tail `P(A >= age + 1) = (1 - p)^age` in stable log space, and after the caller optimizer update refreshes selected incoming columns/biases, zeros outgoing rows, and clears supported Adam moments. Exact warmup/history/equality boundaries, long clocks, caps, corruption rollback, source/representation-bound checkpoint resume, reset persistence, and eager/JIT/scan parity are tested | L0 standalone mechanism only; the fixed-window positive-support mapping is an explicit Alberta convention rather than released-histogram bit parity, and there is no matched learner, plasticity, retention, control, safety, default-selection, promotion, or scientific-evidence result | `test_self_normalized_resets.py`, `test_self_normalized_resets_integration.py` |
| Optimization-centric plasticity diagnostic | One frozen evaluator-owned A/B/A nonlinear regression stream is applied to exact shared-initialization ordinary-SGD and initialization-centred L2 arms. Switch reports retain raw bit-exact old/incoming gradients, alignment, fixed-radius local probes, phase displacement/churn/sign changes, and separately measured dormancy; deterministic replay binds config, protocol, source, runtime, snapshots, and logical resources | L0 `not_assessed` diagnostic only; fixed descriptive floors, two local directions, one 12-update schedule, no artifact writes, no matched broad matrix, and no OCP/L2/plasticity/retention/default-selection or promotion result | `test_optimization_centric_plasticity_development.py` |
| Independent C-CHAIN equation comparator and empirical-NTK diagnostics | On declared nonzero, unique, disjoint train/reference sample identities, computes the exact Equation 8 half mean-squared current-versus-detached-one-step-lag output churn for exactly one scalar model output per reference sample; vector-valued per-sample extensions fail closed. One combined base-plus-churn `value_and_grad` runs after fail-closed preflight. Commit performs no autodiff, advances an exact two-word clock, rolls the source into the lag slot, and binds the caller-applied current tree. The appendix absolute-loss ratio has explicit bounded-window, warmup, epsilon, and clamp controls; ring indexing is rollover-safe. Checkpoints, resources, tamper/stale rollback, eager/JIT behavior, and the paper's singular-value-prefix NTK rank plus diagonal/off-diagonal diagnostics are covered | L0 `not_assessed` isolated comparator only. It is not the full sequential C-CHAIN PPO/DQN algorithm; callable identity, dataset provenance, and external optimizer application are unauthenticated. There is no Prototype/default-agent integration, matched plasticity/retention/control comparison, efficacy, artifact, evidence, promotion, or SOTA result | `test_cchain.py` |
| Fast/slow recurrence and exact-horizon diagnostic | The existing additive gated fast/slow learner now has an authoritative `uint32[2]` lifetime, saturating telemetry, terminal fail-stop, atomic invalid-input/candidate rollback, strict v2 config/state/result/resource records, exact bytes, and eager/JIT/scan contracts. On its consumed 512/512/512 A/B/A scalar life, the ordinary arm's slow A-probe MSE changed from `0.0188931` after A1 to `4.01919` after B (`212.73x`); its A2 tail MSE `0.0269165` is recovery by further updates | L0 `not_assessed` negative diagnostic. The fast path slightly accelerates adaptation but the slow path overwrites A, so the result is relearning rather than permanent-memory retention. One synthetic consumed root, no thresholds, artifact, evidence, promotion, or default claim | `test_fast_slow.py`, `test_fast_slow_horizon.py`, `test_fast_slow_recurrence_development.py`, `test_fast_slow_recurrence_development_integration.py` |
| Alberta-derived permanent/transient recurrence | A fixed 788-byte learner separates independently trained tanh permanent and transient encoders/heads, learns the transient residual, distils its post-update current-sample prediction into the permanent path, then decays the transient head. It has exact lifetime/rollback/resource contracts, no replay, task ID, boundary, reset, or online RNG, and an equal-state/equal-work no-consolidation arm. On the already-consumed FastSlow A/B/A source, the ordinary permanent A-probe MSE changed from `0.0434621` after A1 to `3.92572` immediately after B; its A2-tail `0.0503954` follows additional A updates. The combined A probe similarly changed from `0.0345716` to `4.63162` | L0 `not_assessed` negative diagnostic. This is explicitly an Alberta-derived supervised `k=1` baseline with seven machine-readable departures, not a source-faithful reproduction of the cited Permanent/Transient algorithms. The permanent path is always active and is overwritten by B; A2 is reacquisition. One consumed root, no sweep, thresholds, writer, artifact, evidence, promotion, source-parity, or default claim | `test_permanent_transient.py`, `test_permanent_transient_recurrence_development.py`, `test_permanent_transient_recurrence_development_integration.py` |
| Leakage-safe latent-context regression experts | A defaults-off fixed two-expert bank applies the existing `ContextInference` active-only-freeze law to generic regression with a complete predict-before-outcome ownership cache. Every update evaluates both expert predictions, losses, and analytic candidate gradients, but commits exactly one subtree; exact ties retain the current owner, nonselected subtrees are bit-exact per transaction, and the matched ablation keeps the current owner with identical 32-byte state and work. On the consumed FastSlow A/B/A source, selective routing learned runtime identity A=expert 0 and reactivated it after one observed A2 outcome. Its prequential phase MSEs were `0.02007/0.01496/0.000722` versus `0.01158/0.04369/0.04535` without selection | L0 `not_assessed` partial/null diagnostic. Per-sample routing fragmented identity: 10 A1 switches and 3 B switches included one B update to the learned A expert, changing its direct A-probe MSE from `3.73e-20` to `1.90e-05` and its subtree hash. The first B and A2 predictions necessarily precede their outcomes; selection/training afterward is not pre-outcome context identification. Clean phase-long dormancy and retention are false. One consumed root, no tuning, threshold, writer, artifact, evidence, promotion, or default claim | `test_latent_context_experts.py`, `test_latent_context_expert_recurrence_development.py`, `test_latent_context_expert_recurrence_development_integration.py` |
| Two-event pairwise-dominance quarantine | A fixed `H=2` router binds the source expert bank, owner, candidate, predictions, losses, and first-event no-worse/strict masks. Opening on one unique globally no-worse dormant challenger makes no parameter commit; the independently authenticated second event confirms only the same candidate with preserved no-worse and accumulated strict support, while ambiguity, ties without strict support, tampering, numeric failure, and stale state reject atomically. On the consumed FastSlow A/B/A source, the enabled arm kept learned A bit-exact through B, made zero B updates to it, and reactivated it after two observed A2 outcomes. Its prequential phase MSEs were `0.0156662/0.0145220/0.0174704`, versus `0.0156662/0.0994502/0.0754818` and 498 B-to-A updates in a routing-disabled same-work arm. Four openings yielded two confirmations and two rejections | L0/development `not_assessed` clean retention on one consumed synthetic root only. The fixed two-event delay intentionally skips four parameter commits; context remains outcome-inferred, expert count and horizon are supplied, and there is no task-free growth/merge policy, multi-seed control result, writer, artifact, threshold, evidence, promotion, or default claim | `test_pairwise_dominance_quarantine.py`, `test_two_event_latent_context_experts.py`, `test_two_event_latent_context_expert_recurrence_development.py`, `test_two_event_latent_context_expert_recurrence_development_integration.py` |
| Reduced optimizer-level forgetting | UPGD retains more task-1 accuracy than its σ=0 twin over 24 paired seeds and shows savings on revisit; forgetting remains measurable | mechanism, not a no-catastrophic-forgetting result | `test_forgetting_gate.py` |
| Remember vs forget = representation | context-gated savings 9.8–26x vs reinit-twin 1.0 and raw 1.2; retention through nonlinear interference 36x | mechanism (oracle features) | `test_gauntlet_certification.py` (P4/P5) |
| Pair-feature finding → memory | exhaustive degree-two learner on raw observations plus a visible context cue reconstructs context gating (≥15/24 slots are ctx×x products) and finds all four supplied-family nonlinear pairs on 8/8 calibration seeds | development mechanism (finite all-pairs archive; not general/autonomous) | `test_gauntlet_discovery.py` |
| Recurring pair-bank held-out result | the immutable v1 artifact retained A/B/C in 30/30 held-out lives and evicted obsolete D from the active bank, but current registered source hashes no longer match it | historical narrow L2 result, currently source-invalidated; supplied heads and a counted exhaustive 15-pair archive | `outputs/recurring_feature/evidence.v1.json` |
| Pair features under a scale shock | the immutable v2 artifact passed on 30 namespace-derived fresh seeds, but current registered source hashes no longer match it | historical narrow L2 package result, currently source-invalidated; visible context, finite exhaustive pair archive, one fixed learner initialization, and primary vs legacy changes normalization/ObGD while adding 464 frozen-artifact bytes | `outputs/scale_robust_feature/evidence.v2.json` |
| Generated compositional recurrence lifecycle substrate | a finite canonical depth-three AST grammar, exact 4,559-step recurrence schedule, five declared controls, a D-mapping twin, a fixed-shape scrub that resets slot-local state, and an expanded-tree compiler that closes target, descendant, and candidate lineage and validates the exact scrub transaction. Its production learner carries an exact big-endian `uint32[2]` lifetime clock, bounded replacement, saturating age/telemetry counters, strict legacy migration, and a bit-exact terminal no-op. A fixed seed-101 in-memory probe source-replayed all 7,856 paired transitions through post-step 3,928 without feature/lineage injection, slot selection, search, threshold, output, or promotion authority: both arms naturally generated candidate exact-D at post-step 800; the reference promoted and first used D at 1,728, while the D-mapping-never-seen twin did so at 3,584 during its first genuine D phase | L0/development feasibility only. The observed run proves one natural generated-D birth/utility path is reachable, not its probability or generality. The result is one consumed development seed; targets A–D and the finite grammar are supplied; exact outcome assertions were recorded only as development regression; structural presence is not behavioral retention; and the probe stops before the complete scrub/rollover/reacquisition schedule. The negative candidate-only/no-cascade ceiling does not characterize the production promotion/cascade lifecycle. There is no held-out artifact, threshold, scientific promotion authority, general feature-finding result, or complete indefinite composed agent | `test_compositional_long_horizon_counters.py`, `test_generated_class_recurrence.py`, `test_generated_class_d_mapping_twin.py`, `test_generated_class_lifecycle_scrub.py`, `test_generated_expression_lineage.py`, `test_generated_class_reachability.py`, `test_generated_reacquisition_epoch.py`, `test_generated_natural_d_birth_probe.py` |
| Scaffold-free compositional control life | one exact 8,998-step silent-task contextual-bandit life gives the learner only six raw Rademacher channels, its selected action reward, and fixed state. Defaults-off novelty admission and transitive ancestor utility backup are causally inspectable. A task-agnostic product dovetail closes the production proposal gap: all 15 raw pairs enter both candidate and active banks, and candidate trajectories contain A/B/C/D triples without target expressions, task IDs, boundaries, or injected features. The base consumed-root arm admits only D (`-0.01045` greedy reward). Topology headroom makes A/B/C admissible and raises reward to `0.12247`, but every A/B/C lifetime is later lost and obsolete `p12` remains. A left-pack sibling removes `p12` at end yet retains at most one of A/B/C at a time, ends with none, and lowers reward to `0.03979` | Strict in-memory L0 development falsification, always `not-assessed`. Proposal reachability and structural headroom are repaired, while the opposed headroom/left-pack failures show that slot availability and placement are not a recurrence-value retention policy. The product grammar is finite; the cursor has the explicit ceiling `replacement_interval * cycle <= 65,536` (1,920 here); bank-level algebraic recurrence is not authenticated birth identity; and there are no thresholds, held-out seeds, artifact, promotion path, or general feature-finding result | `test_compositional_ancestor_feature_credit.py`, `test_compositional_dovetail_product_coverage.py`, `test_compositional_topology_headroom.py`, `test_compositional_control_life_development.py`, `test_compositional_control_life_development_integration.py` |
| Generated-feature deployed representation and consumer identity | a frozen-theta standalone adapter deploys `[exact stable base | generated tail]`, binds full topology plus global/per-slot birth words, authenticates its curation trace, and atomically re-encodes valid rows. Source-bound prepare/commit recomputes every JAX-authoritative proposal leaf. A separate router now binds one exact linear OaK state and an optional ordered linear Horde state, separates the stable source from the caller-attested one-step post-update state, authenticates clocks/bindings/caches, preserves survivor bits, and scrubs every changed-birth axis in value weights/traces, option policies/models, caches, and Horde heads. One captured boolean curation permission lets a caller consume an unsafe due opportunity without forming a birth while ordinary learning/RNG/clocks/cadence advance; later safe curation remains possible. A two-transition development lane now uses real public OaK/Horde updates under the old bank: the unsafe option step advances both learners and consumes the due opportunity without a birth, while the next safe primitive step commits birth mask `[F,F,T,F]`, preserves survivor bits, scrubs changed axes, and rebinds the actual consumer cache. Exact persistent/transient bytes, bounded work, stale/tamper rollback, terminal counters, same-bank no-op, and eager/JIT/scan execution are tested | L0 narrow transaction/integration mechanism only. The caller must derive the OaK safe boundary before proposal formation; the router does not do so. The real loop is deterministic, two transitions, linear OaK/Horde only, and has no matched policy benefit. The adapter/router are not wired into Prototype; theta and parameter-revision identity remain frozen; model, memory, partner, and checkpoint consumers are absent; and no outcome, evidence, promotion, or default claim follows | `test_compositional_feature_adapter.py`, `test_compositional_curation_permission.py`, `test_compositional_consumer_router.py`, `test_compositional_consumer_router_real_update_integration.py` |
| Generated-input fixed-physical-output world model and one-backup planner | A linear model consumes `[stable base | live pair products | primitive-action onehot]` while keeping exactly `B+2` physical heads for normalized stable-base delta, reward, and discount; generated output heads are forbidden. A causal prepare/consume boundary binds the complete source world/router, base/action, augmented input, and pre-outcome prediction, then learns under the source bank before exact input-weight/trace routing. Base/action columns and descriptor survivors are bit-exact; newborn/inactive columns are positive zero. A second full-snapshot boundary binds source world, router, OaK, anchor, action, and prediction; it re-augments the predicted physical successor under the live bank and carries only one OaK base-learner backup. Same-clock alternate content, cache/request tampering, invalid candidates, cold generations, and disabled planning are atomic no-ops. A focused witness shows a learned survivor column changes both the physical proposal and real OaK backup | L0 isolated mechanism only. Planning defaults off; error/warmup ceilings are uncalibrated. There is no uncertainty or partner-action model, general generated AST integration, Prototype/memory/checkpoint/utility-curation composition, matched planning/control/retention benefit, artifact, evidence, promotion, or default claim | `test_prototype_routed_linear_world_model.py` |
| Long-horizon single life (64k steps, no resets) | final-cycle savings 19x (highest of the whole life — memory consolidates, never erodes); fresh-task adaptation flat with age; raw twin stuck at 0.2–0.5 savings forever; 0 NaN | mechanism (oracle features) | `test_lifetime_demonstration.py` |
| Stability under 10x input-scale shift | Autostep 0.014 MSE, SwiftTD 0.011, both 0 NaN; IDBD's divergence *detected* by the same stream | mechanism | `test_gauntlet_certification.py` (P6 + SwiftTD) |
| Off-policy soundness | Baird star: semi-gradient diverges, GTD/ETD converge (linear + horde level); per-decision IS trace verified to closed form | mechanism | `test_baird.py`, `test_baird_horde.py` |
| Forward/backward-view equivalence | independent-demon TD(λ) matches offline λ-returns to 1.4e-7; forbidden trunk-trace scheme mismatches by 299% | mechanism | `test_forward_view_equivalence.py` |
| Classical conditioning | Kamin blocking (w_blocked 7e-6 vs control 0.5); reacquisition savings 0.52x steps with IDBD step-size as the memory | mechanism | `test_pavlovian_learning.py` |
| Closed-loop control learns | SARSA + DifferentialSARSA reward rises vs random baseline on 2-state MDP and RiverSwim (analytic optima) | integrated | `test_control_learning_gates.py` |
| Continuous average-reward actor/critic and recurrence instrumentation | Direct affine-`tanh` Gaussian decisions retain exact pre-tanh ownership, stable transformed target/behavior densities, and the exact latent action ratio. A strict fixed 12-event A/B/A evaluator reconstructs densities, rewards, same-state gauge-centered critic error, actor error/churn, plasticity/activity, successor ownership, state, resources, checkpoints, and exact live replay; transformed diagnostic densities alone permit a symmetric eight-float32-ULP backend bound while the policy ratio remains bit-exact | L0/development `not-assessed` mechanism; no state-distribution correction, convergence, retention, efficacy, calibration, or promotion claim | `test_continuous_average_reward_actor_critic.py`, `test_continuous_actor_critic_retention.py` |
| Nonlinear discrete action-importance-corrected actor/critic | One shared tanh trunk, categorical actor, and scalar critic consume an exact cached executed-action receipt with target and caller-declared behavior log probabilities/revisions. Clipped per-decision action ratios advance separate actor/critic head and trunk traces; component plastic/frozen policies, target revisions, typed Threefry sampling, exact clocks, atomic rollback, checkpoint construction, complete persistent-byte accounting, a hand-derived two-step trace, and eager/JIT/scan parity are tested | L0 `not_assessed` mechanism only; discounted scalar V, one trunk, clipped action-ratio correction only, no authenticated external behavior owner, state-visitation correction, average-reward baseline, learned utility policy, comparator, efficacy, retention, convergence, artifact, or promotion claim | `test_nonlinear_off_policy_actor_critic.py`, `test_nonlinear_off_policy_actor_critic_integration.py` |
| Nonlinear discrete average-reward actor/critic | Separate one-hidden-layer `tanh` actor and critic networks own independent head/trunk traces, momentum, fixed plasticity policies, and bounded utility telemetry around a differential TD error and learned reward-rate baseline. Exact target and caller-owned behavior distributions, revisions, action identity, and owner digest are bound through a pure proposal and recomputing atomic commit. Ordinary epsilon-mixture behavior scores and clipped per-decision target-action importance are separately tested with fail-stop clocks, raw numeric gates, rollback, checkpoints, resources, and eager/JIT/scan parity | L0 `not_assessed` mechanism only; clipped importance does not correct state visitation, utility has no learned plasticity authority, and there is no comparator, efficacy, retention, convergence, safety, artifact, or promotion claim | `test_nonlinear_average_reward_actor_critic.py`, `test_nonlinear_average_reward_actor_critic_integration.py` |
| Non-learning embodied command envelope and fault-path audit | `EmbodiedSafetyEnvelope` independently checks telemetry freshness/identity, deadlines, bridge state, exact model/optimizer/lifecycle/config bindings, joint position/velocity/torque, workspace, collision clearance, and emergency-stop state before returning a proposed command, a fixed in-envelope fallback, or no action. Emergency-stop/reset and authority-bound rollback/checkpoint transitions fail closed. A frozen 30-event synthetic audit reconstructs interventions, fallbacks, disconnect/reconnect, stop/reset/rollback, shadow-readiness facts, resources, checkpoint resume, and exact causal replay while recording zero physical dispatches | L0/development `not_assessed` mechanism only; neither the static fallback nor the synthetic telemetry schedule is a geometry/dynamics proof, physical-safety certificate, caller-authentication system, learner-adaptation result, held-out trial, or deployment/promotion authority | `test_embodied_safety_envelope.py`, `test_embodied_safety_envelope_integration.py`, `test_embodied_robot_fault_injection_development.py`, `test_embodied_robot_fault_injection_development_integration.py` |
| Reset-free embodied dynamics/adaptation diagnostic | Adaptive and zero-learning `PrototypeAgent` arms own independent bounded two-joint plants across 12 A/B/A-plus-change-family events, with identical initial learned parameters, matched capacity and event/update opportunities, and typed Threefry pairing restricted to exogenous dynamics, sensor, latency, and fault inputs. Every primitive command crosses the hard envelope; a changed fallback rebinds the public cached action before credit, and an unavailable action produces no plant step or transition. The trace witnesses changed adaptive parameters, unchanged frozen policy/model/utility surfaces, seven accepted updates and five skips per arm, full composite checkpoint resume, fixed resources, pure-kernel eager/JIT/scan parity, and source/runtime-bound causal replay | Finite consumed L0 development diagnostic, always `not_assessed`; policy trajectories diverge, the frozen arm is not experience matched, the declared change family is consumed rather than held out, the selected source manifest is not a transitive dependency lock, and there is no adaptation efficacy, safety, geometry, physical dispatch, deployment, evidence, or promotion claim | `test_embodied_dynamics_adaptation_development.py`, `test_embodied_dynamics_adaptation_development_integration.py` |
| Option value plus duration (Step 5) | online TD heads recover returns `(6, 4)` and durations `(10, 2)` in a deterministic renewal problem; value-only picks the slow option while predicted return/duration picks the optimal fast option | L1 development mechanism; supplied options/features, same successor state, no held-out control comparison | `test_option_value_duration.py` |
| Support-aware option-model value search | An opt-in stateless controller recomputes completion-supported differential semi-MDP option targets and Bellman residuals after every accepted backup, stably prioritizes the largest magnitude under a fixed budget, and commits only the base learner while preserving traces, action/lifecycle ownership, OaK counters, models, and RNG | L0 mechanism only; completion count is not calibration, the already cached action is not refreshed, effects are deferred to a later extended-action selection, and there is no shared primitive/option budget or control-benefit result | `test_option_search_control.py` |
| Causal state construction | identity, fixed-trace, online-gated, dense full-GRU, and diagonal complex compressed-RTU builders satisfy fixed-budget/JIT/checkpoint contracts. The RTU retains unit-diagonal sensitivities exact for fixed parameters, with approximate moving-parameter carry. A deterministic write/hold POMDP confirms all trainable paths change parameters. Its consumed four-seed default run gave mean frozen-suffix accuracy 0.5158 observation-only, 0.5292 fixed trace, 0.5258 online-gated, 0.5067 full GRU, and 0.5617 RTU; RTU total persistent state was 1,324 bytes versus 12,204 for the full GRU | L0/development mechanism and descriptive signal only; arms are resource-unmatched, the task is a supervised fixed-delay probe rather than control, seeds are consumed, and there is no artifact, inference, or promotion | `test_state_builder.py`, `test_learnable_gru_state_builder.py`, `test_recurrent_trace_unit_state_builder.py`, `test_recurrent_trace_unit_state_builder_integration.py`, `test_state_builder_pomdp.py` |
| Causal RTU generate-and-test lifecycle | Per-unit utility is an EMA of the causal pre-update absolute real/imaginary activation-gradient contribution. Exact period/warmup/age/support/protection gates select a stable fixed quota of the lowest-utility eligible units. Whole-unit replacement redraws polar recurrence and both input rows; activation, compressed RTRL sensitivity, and optional Taylor trace/source/delta slices are scrubbed while nonselected bits are preserved. Optional ordinary RTU learning is recomputed from the exact source and binds the sole admissible advanced destination. A content-bound finalization receipt reconstructs the advance receipt, independently reruns RTU commit, and exact-matches destination/mask, proving derivation but not authenticating caller-supplied lifecycle, objective-gradient, or ordinary-proposal authority. Stale/tampered proposals, invented destinations, numeric failure, and clock exhaustion roll lifecycle and builder back; typed Threefry ownership, exact resources/checkpoints, and eager/JIT/scan parity hold | L0 `not_assessed` mechanism only; effective contribution is not paper-defined delight, the objective is externally supplied, moving-parameter RTRL/Taylor limitations remain, and the live Prototype consumer is restricted to the strict-linear comprehensive lane recorded below. There is no matched recurrent comparison, retention/control outcome, Forager result, evidence, or promotion authority | `test_rtu_generate_and_test.py` |
| Prototype shared pair-feature lifecycle and WP7.1b utility audit | An opt-in fixed-width pair bank has a legacy control-only mode and a shared linear OaK/Horde mode. The shared task vector is control-first followed by `HordeUpdateResult.td_targets` in declared demon order; both consumers update under the old bank before an accepted change atomically routes their post-update feature axes in exactly two calls. Scale-normalized lifecycle proxy utility weights control `0.5` and each of `D` demons `0.5/D`; an ordered schema digest couples the consumers and binding in a v4 checkpoint. A separate fixed-budget auditor uses the old-bank, predict-before-update target/prediction/tail-weight snapshot to score the exact normalized one-step half-squared-loss increase from deleting each active contribution. A matched shadow-candidate insertion cohort is scored separately before normalized-LMS shadow, EMA, and moment updates. Auditor task mass stays fixed at `0.5` for control and `0.5/D` for each ordered demon; after a committed two-call route it explicitly rebinds by descriptor identity without another router call. Enabling it nests audit state with the consumer bundle in one atomic state bundle and requires the v5 checkpoint schema; disabling it leaves v4 unchanged. Strict validation admits only linear OaK plus exact linear LMS/no-normalizer Horde state | L0 diagnostic mechanism/integration only. The raw cohorts are not themselves a curation ranking and the auditor alone has no curation authority. It establishes no adapted-consumer deletion effect, empirical return or benefit, planning, control, safety, automatic subtask/option discovery, promotion, WP7 completion, or L3 evidence, and renews no registered evidence artifact. World models and memory remain excluded from this auditor/curation lane; separate exact-Identity exceptions below do not consume the generated tail or share its ranking authority. Standalone callers must checkpoint all consumers with the binding, and registered artifacts whose source hashes differ remain invalid | `test_interaction_task_utility_weights.py`, `test_prototype_feature_lifecycle.py`, `test_prototype_feature_lifecycle_horde.py`, `test_prototype_feature_lifecycle_integration.py`, `test_prototype_shared_feature_horde.py`, `test_prototype_feature_utility.py`, `test_prototype_feature_utility_integration.py` |
| Prototype WP7.1c audit-ranked pair-feature curation | An opt-in stateless policy converts post-observation audit EMAs into transient rank overrides. This feature deletion/insertion sensitivity is not paper-defined actor-sample delight: it neither scores actor samples nor selects backward passes. Lower deletion utility ranks active slots only against active slots; higher insertion utility ranks candidates only against candidates, with no cross-cohort comparison. Every configured task must meet the exact evidence floor for each eligible slot, and fixed control/Horde task mass is never redistributed or renormalized. Existing ages, maintenance cadence, candidate confirmation, proxy promotion floor and margin, and safe routing retain all go/no-go authority. An exact v6 checkpoint shell binds the ranking config/digest around the v5 utility/consumer bundle, while disabling it leaves v5 behavior unchanged. The adapter has zero persistent state and adds zero RNG draws, backward passes, consumer updates, or router calls | L0 ranking mechanism/integration only. Audit utility has within-cohort ranking influence, not curation, promotion, or go/no-go authority. No empirical return or benefit, adapted-consumer deletion effect, planning, control, safety, automatic subtask/option discovery, evidence renewal, scientific promotion, WP7 completion, Alberta Plan completion, or L3 evidence is established | `test_prototype_feature_utility_curation.py`, `test_prototype_feature_utility_curation_integration.py`, `test_prototype_feature_lifecycle_horde.py`, `test_feature_discovery.py` |
| WP7.2/7.3 cumulant/subtask proposals, bounded scheduler, live STOMP installation, and authorized retirement | A fixed universe covers controllable events, feature changes, reward-relevant transition atoms, and typed prediction bottlenecks under noncompensating learnability, controllability, novelty, contribution, and aleatoric-veto gates. Fixed family quotas, frozen random projections, and exactly `B` hand descriptors provide equal-budget cohorts. A strict adapter installs only a complete fresh source/canonical/transition-bound bundle, rematerializes its cumulants, masks cold slots, and preserves or resets slot state at a quiescent semantic boundary. A bounded scheduler observes discovery every accepted transition, requests a newly observed bundle at exact cadence/retry without storing proposal payloads, and emits authority-free maintenance handoffs. A separate controller binds one handoff to exact caller authority, recomputes noncompensating live-audit retirement facts, performs two independently keyed public rebinds to scrub approved slots, and persists the cold mask through selection, bootstrap, planning, and attribution | L0 proposal/scheduler/installation/authorized-retirement mechanism only. Orchestration and go/no-go receipts remain caller-owned; no replacement is queued, and there is no autonomous authority policy, OaK/Prototype composition, empirical benefit, WP7 exit, evidence renewal/promotion, Alberta Plan completion, or L3 evidence | `test_cumulant_subtask_discovery.py`, `test_cumulant_subtask_discovery_integration.py`, `test_cumulant_option_installation.py`, `test_cumulant_option_installation_public_exports.py`, `test_cumulant_option_scheduler.py`, `test_authorized_option_retirement.py`, `test_stomp_extended_action_mask.py` |
| WP7.3 v1 option lifecycle audit and live STOMP observer | Exact two-phase transactions bind source/representation, transition, the full semantic/generation universe, state revision/checksum, context, comparator assignment/propensity, and frozen pre-update model signature. Per-context initiation, completion reasons, censoring, STOMP-compatible discounted external return, pseudo-return, model error, planning use, redundancy, and resource cost remain separate. Randomized treatment/primitive evidence must meet both floors in every context and aggregates use fixed equal context mass. An opt-in persistent wrapper derives real ownership, starts/endings, frozen model signatures, return inputs, outcome deltas, planning, and costs from STOMP. Valid STOMP commits even when audit capacity is terminal or external attribution is rejected; the audit freezes, persistent composed corruption fails closed, and changed semantics explicitly reset affected option-local learner/model state | L0 audit/live-observer mechanism only; randomized assignments are observational, audit failure is terminal for that lifecycle, OaK/Prototype and automatic replacement are absent, and the signed-int32 observation cap limits attribution. There is no dispatch, curation, transfer, promotion, empirical-benefit, WP7-exit, or L3 authority | `test_option_lifecycle_audit.py`, `test_option_lifecycle_audit_integration.py`, `test_stomp_option_lifecycle.py`, `test_stomp_option_lifecycle_integration.py` |
| WP7.4 v1 calibrated extended search and live Prototype/STOMP sidecar | Model-free extended-Q, primitive-model, option-model, and combined modes share one real-anchor bank, state shape, and exact secondary-update budget; combined mode ranks one primitive/option union. Correct primitive and differential semi-MDP option targets feed a noncompensating value-change × future-reachability × model-reliability × support priority. A strict development evaluator gives all four arms one immutable source/runtime-bound model/calibration snapshot, the same evaluator-owned Threefry trace, and exactly `B` attempts, with raw accounting, checkpoint/resume, integrity-bound replay, and corruption rejection. A separate opt-in raw-representation adapter snapshots the actual learned Prototype primitive model and STOMP option models at live decisions; primitive arms settle on the next accepted transition, option arms retain exact ownership through natural completion/censoring, and both families still share exactly `B`. The sidecar draws no planner RNG, never rewrites Prototype's action, and cannot block valid real learning when optional search exhausts | L0/development mechanics and live online model/calibration integration only. The evaluator remains `not-assessed`, frozen-model, action independent, and consumes one nonpromoting seed. The live sidecar Q surface has no policy, dispatch, or keyboard authority; it excludes learned/recurrent representation lanes and establishes no policy benefit, WP7 exit, promotion, or L3 evidence | `test_calibrated_extended_search_control.py`, `test_calibrated_extended_search_control_integration.py`, `test_calibrated_extended_search_development.py`, `test_calibrated_extended_search_development_integration.py`, `test_prototype_stomp_calibrated_search.py`, `test_prototype_stomp_calibrated_search_integration.py` |
| WP7.4 v2 calibrated search-to-live-policy composition | A separate default-off wrapper settles the prior arm, admits only candidate-specific calibrated evidence at one exact real anchor, forms a primitive or proposal-only option-keyboard command, intersects a caller-owned hard primitive mask, uses Prototype's public cached-action replacement once, and arms the actual base or active-option owner that will receive the next transition. Proposal availability is distinct from dispatch authorization; an unavailable proposal may retain only an independently safe current-owner command. Withheld decisions expose `-1`, reject forged inbound learning, and may be retried with a fresh mask without learning. Config/proposal/state/checkpoint bindings, exact clocks, active-option preservation, arm-owner preconditions, rollback, resources, and eager/JIT/scan behavior are tested | L0 `not_assessed` policy-mechanism integration only. The wrapper is not a safety authority or physical dispatcher, starts no option, adds no planner RNG/model update/backward pass, has no matched control-benefit or calibration result, and supplies no WP7 exit, evidence, promotion, or L3 claim | `test_prototype_stomp_calibrated_dispatch.py` |
| Consolidated procedural-memory readout and live Prototype consumer | An already-produced procedural retrieval must pass exact compatibility/freshness and option-lifecycle identity, count consistency, minimum evidence, Wilson success lower bound, finite bounded outcome moments, derived standard-error, nonnegative categorical score-mass, and mandatory hard-safety gates. The stateless result is the lowest-index safe positive-mass argmax or an explicit abstention. An opt-in wrapper resolves exact pending feedback, retains Prototype learning for the actual executed primitive, queries the shared store only for the next decision, intersects caller/experiential/partner masks, and uses the public cached-action replacement boundary with persistent decision/request/mask ownership. Memory exhaustion freezes only memory; corruption fails closed | L0 readout/live-consumer mechanism only. The readout itself still has zero query/write/RNG/dispatch/mutation/state work; the wrapper has no autonomous skill creation, physical dispatch, efficacy, safety, WP8, promotion, or L3 result | `test_consolidated_memory_policy.py`, `test_consolidated_memory_policy_integration.py`, `test_consolidated_memory_controller.py`, `test_consolidated_memory_controller_integration.py`, `test_prototype_consolidated_memory.py`, `test_prototype_consolidated_memory_integration.py` |
| Consolidated-memory post-envelope dispatch settlement | The live procedural/semantic compositions persist a checksum-bound owner for the exposed Prototype decision, selected primitive, and exact hard mask. An unchanged admitted action is an exact no-op. A changed admitted fallback uses the public cached-action replacement, cancels only exactly matching procedural and partner feedback owners, and adopts the rebound action and owner atomically. No-action remains retryable; stale identity/action, disallowed fallback, owner corruption, partner mismatch, partial cancellation, and checkpoint tamper preserve the complete state. Cancellation writes no memory/reliability evidence or counters, and settlement applies no learner update, RNG draw, or physical command | L0 cross-layer mechanism only. The settlement input is caller supplied, does not authenticate an envelope or hardware executor, and grants no safety, physical-dispatch, efficacy, evidence, promotion, WP8-exit, or WP9-exit authority | `test_consolidated_memory_controller.py`, `test_partner_policy_fusion.py`, `test_prototype_consolidated_memory.py`, `test_prototype_consolidated_memory_integration.py`, `test_prototype_consolidated_semantic_memory.py`, `test_prototype_consolidated_embodied_dispatch_integration.py` |
| Hidden-partner integration kernel | focused tests cover the causal composed transition; eight uninterrupted development lives add matched state, memory, lifecycle, carry, retention, planning, partner-belief, and curation ablations, with all full lives finite and mean reward 0.89335 | L0/development integration only; artifact is structurally nonpromoting and fails its obsolete-D forgetting check because D is absent at life end in 0/8 full lives; partner is scripted and the candidate archive is closed | `test_integrated_hidden_partner.py`, `outputs/hidden_partner_development/robustness.v1.json` |
| Hidden-partner v6 execution contract | fixed-shape runner, 15 primary plus 3 diagnostic controls, exact source/runtime payload closure, intervention witnesses, stream/RNG reconstruction, deterministic initial state, strict per-run structural validation, matched-suite validation, and a separate canonical source-replay verifier | development infrastructure only; every open certification prerequisite remains `NOT_CERTIFIED`, and no complete v6 life, source replay, 18-arm outcome suite, calibration, threshold, artifact, efficacy result, or promotion authority has been produced | `test_hidden_partner_lifecycle_world_v6_runner.py`, `test_hidden_partner_lifecycle_world_v6_runtime.py`, `test_hidden_partner_lifecycle_world_v6_intervention_audit.py`, `test_hidden_partner_lifecycle_world_v6_validator.py`, `test_hidden_partner_lifecycle_world_v6_matched_suite.py`, `test_hidden_partner_lifecycle_world_v6_source_replay.py` |
| Hidden-regime signaling and factorial calibration | a 240-case factorial calibration design, threshold-freeze and protected-plan validators, execution governance, readiness receipts, trace audit, and summary/lineage oracles are implemented fail-closed; the six-condition evidence draft is execution-disabled and the three protected structural-generalization worlds record zero executed learner outcomes | design-only; `outputs/` contains no hidden-regime artifact, the protected plan is `preregistered_unexecuted` with no execution issuer, and the development runner hard-codes promotion disallowed | `test_hidden_regime_factorial_calibration.py`, `test_hidden_regime_factorial_thresholds.py`, `test_hidden_regime_signaling_evidence.py`, `test_hidden_regime_signaling.py` |
| Hidden co-learning world-and-agent planning kernel | two online signaling roles learn on one shared physical row while a behavior model predicts the beneficiary and a grounded joint model predicts next cue, reward, and discount; a one-step helper planner marginalizes both models under randomized consumption and causal channel/freeze controls. Primitive traces reconstruct reward, model error, switch cost, and same-context recurrence diagnostics; persistent state is exactly 321 bytes with zero replay. A separate nonexecuting declaration and in-memory runner bind all 11 bridge conditions to four paired exploratory seeds, common-random-number reconstruction, raw outputs, phase diagnostics, and update-opportunity-versus-committed-write accounting. Execution requires an exact full-source/runtime-bound request, a one-campaign/one-replay process-local permit, and a fresh strict live host-quiescence check; authenticated replay bit-compares the complete nested run | L0/development infrastructure only; oracle context and target stay evaluator-only. The matched runner, public cross-arm RNG audit, execution gate, and authenticated replay mechanism are implemented, but no permit has authorized the default campaign, live quiescence has not been verified for it, and no default 3,072-step outcome, efficacy, retention, calibration, threshold, artifact, held-out result, or promotion claim exists | `test_hidden_learning_partner_planning_development.py`, `test_hidden_learning_partner_planning_scan_plan.py`, `test_hidden_learning_partner_planning_runner.py` |
| Stationary Forager causal map | observation/action/reward history builds a relative toroidal map and learns channel rewards and respawn timings; privilege isolation, serialization, JIT, chunk, batch, and short installed-environment mechanics are tested | L0 task-specific mechanism; uses public 15×15 movement structure, has no performance artifact, and is not learned recurrent state, nonstationary retention, or a SOTA result | `test_causal_map_forager.py` |
| Forager field-of-view protocol | a stage-conformant 10k-step tuning matrix on five disjoint seeds selected `step3e3`; the old 30×500k evaluation produced no batch or report and is inactive, while its official-DQN and relearning companions are explicitly quarantined | benchmark execution only; no registered performance claim, paired comparison, or promotion | `outputs/forager/fov_tuning_10k_seeds1000000_1000004/report.json` |
| Forager RTU-RTRL development run | four open development seeds completed 500k GPU steps with FOV tail-EMA AUC mean 1.550, sample SD 0.324, and range 1.167–1.936; an unsealed DQN comparator gives a descriptive RTU-minus-DQN mean of +0.331 on the same four seeds | nonpromoting development receipts; no preregistered selection, held-out seeds, admissible paired baseline, matched runtime/resources, complete source closure, or SOTA claim | `outputs/forager/rtu_rtrl_500k_dev4/receipt.v1.json`, `outputs/forager/dqn_fov_500k_dev_seeds2000001_2000004_reconciled/receipt.v1.json`, `test_forager_development_receipts.py` |
| Forager open candidate screens and RNG parity | complete two-seed CPU screens rank DQN-LN first within the feed-forward set (mean 1.49084) and PPO-RTU-LN first within the stateful set (mean 1.78110); a fixed-action direct/wrapper content trace matches exactly | nonpromoting development diagnostics; consumed seeds, candidate budgets not necessarily matched, retained RTU-PPO RNG confound, and an externally unverified parity receipt with promotion explicitly unauthorized | `outputs/forager/fov_baseline_screening_cpu_v3_execution/aggregate.json`, `outputs/forager/fov_stateful_baseline_screening_cpu_v3_corrected_v4_execution/aggregate.json`, `outputs/forager/rng_parity_live_qualification_v1_execution/receipt.json` |
| WP1 Prototype continual-control report construction | One versioned in-memory report runs `PrototypeAgent`, a running-reward bandit, and a frozen-action baseline on consumed seeds 1701/1702 with independent environment states, raw action/decision ownership, reconstructing evaluator reports, exact opportunities/logical bytes, deterministic logical latency, explicit diagnostic applicability, source/runtime replay, and checkpoint continuation | L0 `not_assessed` report witness only; no output artifact, promotable seeds, realized-compute matching, hardware/energy/internal-gradient measurements, or efficacy inference | `test_prototype_continual_control_development.py`, `test_prototype_continual_control_development_integration.py` |
| Balanced/comprehensive state objectives, full-GRU/RTU RTRL, and opt-in Prototype adapters | Separate linear multiple-timescale GVF and consecutive-pair inverse-action heads update from an exact executed-action receipt; the comprehensive kernel adds next-observation/latent, reward, stable Bernoulli termination, value, and selected-action-advantage heads with width/head-count-invariant family masses and separate revisions. Dense full-GRU and compressed RTU builders supply exact fixed-parameter RTRL under the shared proposal/commit/checkpoint/resource contract. The comprehensive transaction binds caller target bits/provenance, both recurrent pullbacks, final/bootstrap observation, and builder owner to the exact Prototype decision/action, then makes one clipped logical builder update or rolls every component back. A strict-linear lane owns the RTU lifecycle source, internally constructs its gradient and ordinary proposal, independently rederives the committed destination, atomically replaces whole units, scrubs recycled base/intra-option/option-model axes, and performs ordinary next-action selection; an active option defers replacement. Four source-level builder-commit evaluations and two RTU-commit evaluations implement preflight/derivation checks, while only one logical ordinary update and at most one replacement event persist | L0 mechanism/integration only; masses/targets are uncalibrated, GRU storage is `O(H * P)`, both moving-parameter carries are approximate, live replacement excludes nonlinear/model/replay/dreaming/Horde/IA/partner/memory/GRU/feature-lifecycle configurations, and no retention/control/Forager outcome exists | `test_balanced_state_objectives.py`, `test_balanced_state_objectives_integration.py`, `test_comprehensive_state_objectives.py`, `test_comprehensive_state_objectives_integration.py`, `test_learnable_gru_state_builder.py`, `test_recurrent_trace_unit_state_builder.py`, `test_rtu_generate_and_test.py`, `test_prototype_balanced_state_objectives.py`, `test_prototype_balanced_state_objectives_integration.py`, `test_prototype_comprehensive_state_objectives.py`, `test_prototype_comprehensive_state_objectives_integration.py` |
| Real-anchor ensemble one-step Dyna | Bounded anchors and exact monotonically advancing model/control revisions form guarded one-step `reward + continuation * max Q` synthetic updates while the ensemble remains read-only and planning RNG/traces/resources stay isolated | L0 mechanism only; supplied uncalibrated gates, no Prototype integration, actor update, matched planning gain, or held-out artifact | `test_one_step_dyna.py`, `test_one_step_dyna_integration.py` |
| Guarded ensemble short-rollout proposals | Exact real anchors and full source/model/policy/value revision-content receipts produce fixed-shape policy-directed or max-epistemic paths. Support, residual, epistemic, finite-value, and termination-agreement gates apply per step; learned termination stops padding and cannot bootstrap, while valid horizon truncation uses the immutable value snapshot for terminal-correct reverse returns. Same-revision content aliases, stale decisions, and capacity failure are atomic no-ops | L0 proposal mechanism only; immutable linear policy/value authority, global support proxy, supplied uncalibrated gates, no Prototype/STOMP/actor/critic/model consumer, and no matched control or retention outcome | `test_ensemble_short_rollouts.py`, `test_ensemble_short_rollouts_integration.py` |
| Grounded imagined-rollout authorization and bounded actor/critic consumer | One frozen source/model generation accumulates a causal fixed-capacity audit by primitive action × caller region. Evidence, realized validity, reward and next-observation error, termination, success LCB, top-quantile purity, safety admission, and protected masks are separate gates; audit/candidate transition identities must be disjoint, and a failed valid predecessor rejects every later path transition. Full-content receipts bind the candidate, masks, generation, and calibration content. The learner proposal is autodiff-free; commit revalidates the source, receipt, freshness, and budget before exactly one guarded fixed-shape `value_and_grad`, while a stale/tampered/replayed preflight performs zero. Terminal critic targets use terminal reward, dream imitation is graded by bounded positive advantage, and competent-real cloning has the same prefix-closed transition/update ceilings. A strict direct composition derives policy/value authority from the live actor/critic, passes the exact locally produced planner batch directly into the gauge, and atomically couples all planner/authorization/learner/dream clocks, removing public rollout-tensor substitution | L0 `not_assessed` mechanism only; deterministic tags provide post-mint integrity but neither the standalone gauge nor the direct composition authenticates the external model/support facts, real environment/anchor, regions, safety masks, or competent-real truth. Floors are supplied rather than empirically calibrated, and there is no Prototype/dispatch integration, matched outcome, safety certification, evidence, promotion, or SOTA claim | `test_imagined_rollout_selection_gauge.py`, `test_grounded_imagination_composition.py` |
| Prospective fixed-budget exploration and causal synthetic score-production lane | The selector computes expected improvement times capped host-relative surprisal and five comparator priorities under one candidate/RNG budget before the caller-owned hard shield. A consumed eight-event evaluator gives all six arms independent environment/estimator/selector/shield state, pairs exogenous noise only, and derives every score from that arm's executed action-conditioned linear-TD history in a world with progress-resetting noisy TV and delayed invest/collect behavior. Exact raw hash-chain replay, causal-prefix reconstruction, in-memory resume, and matched logical opportunities/resources are validated | L0 `not_assessed` mechanism/development instrumentation; the standalone API still accepts supplied scores, the bounded linear ensemble is not exact sequential VOI, the shield is not physical-safety evidence, and there is no threshold, winner, artifact, held-out environmental efficacy, physical-resource match, or promotion result | `test_delightful_exploration.py`, `test_delightful_exploration_integration.py`, `test_causal_exploration_estimator.py`, `test_prospective_exploration_development.py` |
| Typed surprise/progress/change | hand-checked ensemble disagreement and uncertainty stay separate; high modeled aleatoric noise does not create epistemic surprise/change, while a sustained calibrated residual shift exceeds an isolated outlier | L0/development mechanism; internal frozen calibration, no routing-consumer benefit | `test_learning_signals.py` |
| Candidate-update safety audit and application (historical gradient-joy API) | `candidate_update_audit_passed` reports the formed-candidate verdict after complete finite objective/retention/safety evidence, fresh roundoff-resolved dot/norm certificates for both the candidate and actual elementwise-rounded tentative tree, conservative magnitude/alignment endpoints, and a certified trust-bound pass; the effective stored delta is re-audited after dtype cast/addition, while `applied`/`audited_candidate_update_applied` separately reports that a finite-precision parameter change was actually committed; `sparks_joy` and `joyful_gradient_applied` remain compatibility aliases, not paper terminology | L0 mechanism; detached local first-order probes and fail-closed application boundary, not realized improvement or a DG/Kondo reproduction | `test_delight.py`, `test_prototype_gradient_joy.py` |
| Paper-defined Kondo screen/gather and nonlinear actor consumer | Computes delight exactly as advantage times selected-action surprisal in a detached forward screen; “sparks joy” means selected for an actor backward pass. Finite-temperature Bernoulli-price and deterministic fixed-rate top-k modes preserve forced samples. `KondoSparseActor` binds exact action/revision/behavior-log-probability identity, gathers a fixed actor-only capacity before `jax.value_and_grad`, and uses an explicit full-shape fallback without dropping forced/overflow survivors. A capacity-3 versus batch-6 JAXPR witnesses the smaller backward shape; critic, baseline, return, and safety inputs remain full-batch pass-through data. One action-fixed replay diagnostic gives four arms one actor update plus full protected learning per batch but is explicitly off-policy. A separate closed-loop diagnostic freezes each arm's own actor only during collection, samples that arm's actions from its own policy with shared exogenous uniforms, and updates once at the batch boundary; actions and trajectories may diverge, while rare failures remain in both actor and full protected learning | L0 mechanism/integration and development diagnostics only. Gate/gather orchestration is host-side, only collection/fixed-shape backward kernels are JIT-compatible, the replay lane has no source behavior policy or importance correction, and the closed-loop lane establishes neither measured compute savings nor learning, safety, DG-reproduction, evidence, or promotion benefit; every evaluator status is `not_assessed` | `test_kondo_gate.py`, `test_kondo_sparse_actor.py`, `test_kondo_sparse_actor_integration.py`, `test_kondo_actor_critic_replay_development.py`, `test_kondo_actor_critic_replay_development_integration.py`, `test_kondo_actor_critic_on_policy_development.py`, `test_kondo_actor_critic_on_policy_development_integration.py` |
| Real Prototype partner-fusion closed-loop instrumentation | A consumed 12-execution evaluator runs learned-feedback, fixed-zero outcome-blind, and empty-message base-only conditions through independently owned real `PrototypeAgent`, fusion, environment, hard-mask-authority, receipt, and raw hash-chain state. Only a frozen exogenous context/noise/drift/availability/cost/mask schedule is paired; executed actions causally generate each arm's later observations, rewards, messages, and feedback across an uncued reliability reversal. Real action-changing updates, realized action-relative assistance, caller-owned mask enforcement, exact causal-prefix replay, in-memory resume, update-boundary eager/JIT parity, and matched logical call/shape budgets are checked. Legacy host-only learner birth/uptime leaves are canonicalized to exact float32 zero before hashing, and resealed noncanonical timing state is rejected | Consumed L0 `not_assessed` mechanism/development instrumentation only; independent learner initializations/RNG prevent a matched causal comparison, the life is short, and there is no artifact writer, threshold, multi-seed trial, confidence-calibration result, efficacy claim, WP8 exit, or promotion authority | `test_prototype_partner_fusion_closed_loop_development.py` |
| Matched-initialization v2 Prototype partner-fusion instrumentation | A separate consumed 12-event evaluator starts learned-feedback, fixed-zero outcome-blind, and empty-message base-only wrappers from bit-identical typed RNG, Prototype, fusion, and environment state. Only the wrapper intervention differs. Exogenous context/noise/drift/availability/cost/mask inputs are paired, while each arm owns its later causal trajectory after action divergence. Raw hash chains, exact prefix reconstruction, replay, checkpoint/resume, fresh source/runtime/config binding, eager/JIT Prototype parity, and matched logical work are checked. Learned and fixed-zero each changed action three times and had equal task/net return; their final Prototype states differ without realized behavioral separation. Base-only changed action zero times | Permanently L0 `not_assessed`, consumed, threshold-free, winner-free, artifact-free, and nonpromoting. The horizon is 12 events, every toy action is declared safe, there is no shielding-efficacy test, behavioral separation, calibrated reliability, multi-seed estimate, partner-benefit result, WP8 exit, evidence, or promotion authority | `test_prototype_partner_fusion_matched_v2_development.py` |
| Bounded experiential retrieval and policy integration | Fixed-capacity query-before-write memory rejects stale/wrong-version, unsafe, uncertain, distant, and non-finite entries; deterministic eviction and exact byte/checkpoint parity hold. A stateless categorical policy boundary and opt-in Prototype composition query before write, bind full lifecycle IDs, store the primitive action actually executed with its grounded outcome, compose memory before partner fusion, preserve no-memory shapes, and roll a required unsafe/corrupt event back atomically. A defaults-off, zero-state conservative gate now permits an action-changing retrieval only when exact one-hot neighbors give both actions sufficient support/weight mass and the proposed action's weighted immediate-reward mean is strictly greater than the base action's | L0 mechanism/integration; the outcome gate is associational, uses immediate rewards, and supplies neither delayed causal credit nor context-aliasing, confidence-interval, nonstationarity, held-out transfer, or control-benefit evidence | `test_experiential_memory.py`, `test_experiential_memory_policy.py`, `test_experiential_memory_advantage_gate.py`, `test_prototype_experiential_memory.py` |
| Pair-lifecycle memory and stable-base whole-agent recurrence | An exact-Identity adapter binds every fixed-capacity memory row to the live pair descriptor/generation and atomically re-encodes valid rows before query/write after curation. A distinct v17 composition binds a linear action-conditioned world model to the complete Prototype config, configured LMS scalars, exact lifetime-derived ring buffer, and stable base prefix. One strict development-only A/B/A runner composes those mechanisms with linear OaK and managed Horde, matched preview/readout/promotion work, stale-replay rejection, per-phase world errors, exact resources, and report reconstruction. On consumed seed 0, the exact-default outcome gate allowed `59/1536` visible-cue and `182/1536` cue-masked proposals; the masked arm recorded 142 helpful versus 40 harmful changes and cumulative dispatch delta `+8.1594`, reversing the corresponding ungated arm's net-harm diagnosis | L0/development `not-assessed` mechanism result only; one scripted-partner life, immediate-reward association rather than causal memory credit, generated pair tail excluded from the model, no planning/model-to-curation feedback, no threshold, held-out panel, confidence interval, artifact, promotion, or Alberta Plan completion | `test_prototype_feature_memory_integration.py`, `test_prototype_feature_world_model_integration.py`, `test_prototype_feature_memory_recurrence_development.py`, `test_prototype_feature_memory_recurrence_development_integration.py` |
| Bounded semantic/procedural consolidation and live semantic context | Fixed semantic GVF/fact/affordance and procedural skill stores retain identity/provenance hashes, generations, confidence, source/representation revisions, evidence moments, clocks, staleness, invalidation, success/failure/outcomes, and option-lifecycle links. Every transaction queries pre-write state before compatible merge, next-generation reset, or deterministic replacement; strict checkpoints and exact resources fail closed. A separately versioned wrapper shares the procedural controller state, serializes behind exact pending feedback, performs a semantic pre-write query/current-record write, and supplies `[raw observation, accepted prior payload or exact zero]` to the ordinary next Prototype decision. Exact request/record and both decision identities gate the context; a deterministic weight witness confirms accepted context can change the real next action without direct dispatch authority | L0 storage/live-context mechanism only; the signed-int32 operation cap becomes a continuing-control no-op, but there is no learned retrieval/eviction, fixed-capacity transfer, negative-transfer, stale-skill/semantic-utility efficacy, safety, physical-dispatch, WP8-exit, promotion, or L3 result | `test_consolidated_memory.py`, `test_consolidated_memory_integration.py`, `test_prototype_consolidated_semantic_memory.py`, `test_prototype_consolidated_semantic_memory_integration.py` |
| Semantic/procedural memory stress instrumentation | One frozen 17-event schedule runs full memory, an identical-kernel masked-readout ablation, and a zero-storage/no-kernel arm from the same empty source-bound snapshot. It reconstructs query-before-write, precision/abstention/harm, recurrence/recovery, retained utility, stale-skill harm, eviction/provenance, exact counters/resources, eager/compiled parity, checkpoint resume, and integrity-bound replay | Development `not-assessed` instrumentation only; finite exact-match schedule, no thresholds, the no-memory arm is not storage/compute matched, and no transfer/negative-transfer efficacy, WP8 exit, promotion, SOTA, or L3 claim | `test_consolidated_memory_transfer.py`, `test_consolidated_memory_transfer_integration.py` |
| Sparse-world-model decision fidelity | historical held-out normalized six-step regret 0.000877 vs 0.294800 untrained and 0.049557 raw ridge | historical narrow L2 acceptance; its consumed-seed compatibility chain no longer validates against the current tree (registry `invalid`), replay is nonpromoting, and the unarchived old builder source prevents a full source-recoverability claim | `test_ftl_decision_artifact.py`, `test_historical_ftl_evidence_chain.py` |
| Planning diagnostic (Steps 7/9) | Dyna planning=4: +965 cumulative reward vs planning=0 on eight calibration seeds; guarded dreaming: +798; corruption gate closes ≤4 steps | development integration diagnostic; no held-out artifact or matched update-count claim | `test_planning_benefit.py` |
| Prototype composition diagnostic | selected PrototypeAgent composition improves post-switch reward by +0.05–0.12 over one flat configuration on eight calibration seeds | development integration diagnostic; supplied options and unmatched compute | `test_prototype_nonsaturating.py` |
| Intelligence amplification (Step 12) | historical v1 uplift was `0.267 [0.255, 0.278]` and both augmentation controls passed, but intervention prevalence was 8.73% vs the frozen 10% minimum; the prior consumed-seed compatibility record is nonpromoting and current-source validation is now invalid after registered-source drift (first observed in `average_reward.py`); p=0.75/seeds-60–89 v2 is unissued and permanently development-only because its self-issued plan lacks external chronology | historical valid scientific rejection; no current-source compatibility claim and no v2 plan, reservation, run, shard, or artifact; v2 cannot yield an acceptance result | `test_historical_ia_evidence_chain.py`, `test_continual_ia_v2_contract.py` |
| Learner-generated non-stationarity | learning-opponent stream (drift = another learner's learning curve; IDBD's α tracks the opponent's learning phase); adversarial pursuit drives a frozen predictor to 1771 MSE (48x growth) while continual learners hold ≤0.12 | development mechanism; not a result from two simultaneously acting, co-learning agents | `test_multiagent_sim.py` |
| Multi-agent: convention memory | two learning agents recall joint conventions instantly on rule recurrence with context (t=20 floor, reward 0.94) vs relearning forever without (t≈50, 0.78); conventions are emergent (≥3 distinct across seeds) | development toy dyad with visible context and separate table cells; not hidden-context or general forgetting evidence | `test_multiagent_sim.py` |
| Hidden-context two-learner feasibility | on the complete fixed two-seed development panel, two independent SARSA controllers and two independent two-slot context banks acted before either learner observed the partner action/reward. Routing inferred context improved recurrent early reward over a state/update-matched unrouted inference control by `+0.0546875` and `+0.1171875`. On both seeds, both learners formed distinct rule slots, reused them with `0.99` recurrent agreement, and switched four steps after every rule change; recurrent-tail reward was `0.94140625` and `0.953125`. The joint environment, controllers, and context banks use 408 persistent JAX-array bytes, no replay, and exact two-word clocks; the deterministic environment consumes no randomness | L0 two-seed development feasibility only. Capacity equals the two-rule support, the schedule is deterministic, and there is no noise, eviction pressure, threshold, artifact, causal population estimate, autonomous feature finding, or promotion authority. Tail reward does not uniformly beat the unrouted control, so the observed benefit is faster recurrence rather than settled superiority | `test_hidden_context_coadaptation_development.py`, `test_matrix_game_horizon.py` |
| Hidden-context two-Prototype-agent continual life | Two full independently learning `PrototypeAgent`s share one hidden-rule environment while retaining separate context, pair-feature/OaK/Horde/stable-base-world-model, and experiential-memory state. Visible context coordinates are destroyed; both actions are fixed before two completed-experience context updates, whose inferred one-hots can affect only the next decision. Each of 1,536 events stages four same-prestate environment proposals, two discarded no-memory Prototype previews, and two memory-sidecar candidates, then carries all children through one outer transaction. In the consumed run immediately before the final validation-only gates, routed A1/B/A2 mean-agent reward was `0.991401/0.977935/0.964493`, versus `0.984638/0.0149503/0.994810` for identical but unrouted inference work; routed A2 early/tail was `0.781546/0.997949`. Both arms committed every event with identical 41,718-byte state, 6,144 environment proposals, 3,072 context updates, 6,144 Prototype calls, and 2,944 memory evictions | L0/U0 consumed-root composition and descriptive causal intervention only. The short current-source gate suite passes, but one exact default replay remains pending under a predeclared partial comparator and full-report identity is not claimed. That comparator binds selected direct files rather than a transitive source closure and does not bind runtime identity. The initial slot-zero context prior is routed too; each arm owns its diverging trajectory, capacity exactly equals two rules, world models omit partner action, memory effects are immediate same-event counterfactuals, and no atomic composite checkpoint, strict report validator, writer, threshold, held-out panel, artifact, evidence, promotion, general feature-finding, selective-forgetting, or Alberta Plan completion claim exists | `test_hidden_prototype_two_agent_continual_life_development.py`, `test_hidden_prototype_two_agent_continual_life_replay_declaration.py` |
| Hidden-rule capacity pressure and semantic controller routing | One consumed 4,000-step root runs two independent differential-SARSA controllers and two three-slot context banks across four hidden conventions, with exact semantic birth identities rather than recyclable slot aliases. The fixed epsilon grid has no selected winner: at `epsilon=.2`, both agents end with distinct recurring A/B/C births while phase-tail reward remains `0.65625–0.8125`; `.4` preserves the identity pattern but lowers overall reward to `.516`. A post-audit matched controller scrub authenticates every new birth, zeros only its stale Q/trace column before next-action scoring, and removes all observed cross-birth consumption. A separate defaults-off paired intervention protects a live birth by its authenticated completed recurrence count, affects only full-bank fresh allocation, and leaves stored reuse/free allocation unchanged. Its no-signal arm is bit-exact with the scrub baseline. At `epsilon=.05` it changed two eviction targets, avoided eight completed recurrence intervals, reduced per-agent churn from 25 to 24, and changed reward from `.86700004` to `.86950004`; `.1/.2/.4` were exact nulls. A one-record cross-birth sidecar then tested an archived victim only after full-bank births and required its observed error to strictly beat a fresh prior and every live source model before transferring exact lineage/rescue value. It used 161 bytes per agent with matched work/RNG but was an exact null at every epsilon: zero strict matches, rescue increments, target changes, or reward deltas. At `.05`, four cache-valid tests all failed the fresh-prior gate and all contained an exact tie; the other epsilons had no cache-valid test. An executable prefix twin shares all history through first C admission but makes future B versus D recurrence flip the zero-loss eviction; both have zero recurrence evidence at that first choice, which the intervention leaves unchanged | L0 consumed-root calibration and post-audit mechanisms only; no epsilon/default was selected, the environment is deterministic, the current-birth count forgets value across semantic rebirths, and a single outcome cannot disambiguate the archived convention on this root. The prefix theorem still requires a declared prior, sequential predictive signal, randomized/minimax objective, or retained archive rather than licensing a hindsight eviction. No untouched-root panel, threshold, artifact, population estimate, or promotion authority exists | `test_hidden_rule_capacity_pressure_development.py`, `test_hidden_rule_capacity_pressure_development_integration.py` |
| Recurring two-agent coadaptation (held-out) | the immutable artifact's held-out joint-minus-frozen paired uplift was `0.3358 [0.2991, 0.3738]`, all 30 lives recovered the recurring regime (mean 13.4 steps), and measured interference forgetting was zero, but current registered source hashes no longer match | historical narrow L2 result, currently source-invalidated; tiny contextual bandit with visible regime cues and separate value cells, and 30/30 recoveries bind population recovery probability only at 0.886 | `outputs/continual_multiagent/evidence.json`, `test_continual_multiagent_benchmark.py` |
| Statistics machinery is itself validated | CI coverage 0.944–0.957 empirical vs 95% nominal; Holm ⊇ Bonferroni on 2000/2000 draws; a real rank-biserial sign bug found and fixed | meta | `test_statistics_validation.py` |

## Recurring design law (found independently three times)

**Any always-on shared parameter is a forgetting channel.**  A shared bias, an
un-gated base feature block, or any weight that stays active across tasks
splits credit with task-specific features and is overwritten by whichever task
is current.  Exclusive gating (feature = 0 when its context is inactive ⇒ zero
gradient ⇒ untouched weights *and* preserved per-weight step-sizes) is the
mechanical form of "knowing what to remember" — and the discovery rung shows a
generate-and-test learner selects context × observation products from a
supplied exhaustive finite pair archive. This motivated
`DifferentialSARSAConfig.use_bias=False` for continual-memory experiments.

## Bugs fixed this campaign (all failing-test-first)

SARSA dead-λ for control heads · STOMP env-reward grounding + idle-update leak
· OaK curation zeros-instead-of-init + step-0 eviction · IA exo-cortex
crediting its own actions instead of the partner's · Step-7 pre-warmup RNG
freeze · off-policy horde ρ² trace composition + GTD correction missing ρ ·
`reset_dormant_neurons` optimizer-pytree corruption · baseline optimizers
missing from the config registry · prototype-basis recycled slots inheriting
stale readouts · compositional raw-index aliasing · UPGD-memory blend-logit
gradient bias · Mann-Whitney rank-biserial sign inversion · Prototype
double-advanced recurrent action queries + observation/action ABA transition
ownership + lossy uint64 action aliases + truncation selecting on the wrong
observation · online-gated state learning committing non-finite gradients.

## New capabilities

`core/swift_td.py` (SwiftTD, float32-exact vs the authors' C++ reference) ·
`core/stacked_horde.py` (stacked linear Horde: the demon axis as one batched
array axis — exact TD(λ) semantics, per-decision IS, NaN masking, nexting
helper; 1024 demons × 2000 steps in ~0.2 s steady / 0.3 s compile vs ~140 s
run + 144 s compile for the loop-unrolled path; an unversioned local engineering
observation measured 65,536 demons at 4.0e7 demon-updates/s on CPU, while
`tests/test_stacked_horde.py` asserts exact semantics, analytic fixed points,
all-1024-demons-learn, and generous time bounds only at the tested scale) ·
`core/option_value_duration.py` (separate conventional option-return and
remaining-duration TD heads) ·
`core/state_builder.py` (causal identity/fixed-trace/trainable state contract,
episode-local reset with lifetime counters, and fail-closed scale-safe online
learning) ·
`core/learning_signals.py` (typed ensemble uncertainty/progress/change producer) ·
`core/experiential_memory.py` (fixed-capacity versioned episodic retrieval) ·
`core/consolidated_memory.py` (fixed-capacity semantic/procedural records with
generation, provenance, staleness, and option-lifecycle identity) ·
`core/option_lifecycle_audit.py` (semantic-bound per-context option lifecycle
and randomized primitive-comparator accounting) ·
`core/calibrated_extended_search_control.py` (one-budget four-mode primitive/
option search-control contract) ·
`core/behavior_model.py` (bounded online partner-action prediction and
input-loss gradient) ·
`core/joint_partner_world.py` (bounded joint-outcome table with external
partner-belief marginalization) ·
`core/feature_bank_router.py` (fail-closed atomic descriptor-identity routing
for every downstream feature consumer) ·
`core/prototype_feature_lifecycle.py` (fixed-resource pair discovery plus
generation-and-full-descriptor-bound gradient pullback, bound OaK consumer,
and atomic linear-OaK routing) ·
`core/integrated_hidden_partner.py` (bounded composed hidden-partner
development kernel) ·
`benchmarks/causal_map_forager.py` (stationary observation-causal relative-map
planner) ·
`benchmarks/forager_matrix.py` (frozen tuning/evaluation matrix execution and
source snapshots) ·
`streams/gauntlet.py` (the Alberta Gauntlet + lifetime stream + scorecards) ·
`streams/hidden_partner_mapping.py` (uncued recurring scripted-partner
development world) ·
`streams/opponent.py` (learning-opponent + adversarial-pursuit streams) ·
`streams/matrix_game.py` (recurring convention game) ·
`streams/closed_loop.py` (2-state switching MDP + RiverSwim with analytic
optima).

## Published-protocol replication lanes and current execution status (2026-08-01)

The repository carries runners designed around published configurations. A
runner, or even a completed development diagnostic, is not by itself a
protocol-exact replication: source/execution binding, seed counts, paired
completion, and held-out gates remain part of the promotion boundary.

- **Input-permuted MNIST, UPGD protocol** (Elsayed & Mahmood ICLR 2024:
  1M examples one-per-step, permutation every 5,000 steps = 200 tasks,
  300×150 ReLU MLP, softmax CE, average online accuracy) —
  `benchmarks/upgd_ipmnist.py`, artifacts under `outputs/upgd_ipmnist/`.
  The network, task shape, and horizon match the selected publication
  configuration; the complete protocol does not. The 10 matched development
  seeds are complete. UPGD-W mean online accuracy is `0.7791470803916454` (SE
  `0.000055690729820870456`), with first/last 50
  tasks `0.7774191806316375`/`0.779775180220604` (drift
  `+0.002355999588966484`) and first/last 20 tasks
  `0.7742789795994759`/`0.7796929806470871`. AdamW is
  `0.7190002817213534` (SE `0.0005943125024635892`), with first/last 50 tasks
  `0.7597851804494857`/`0.6918167824745177` (drift
  `-0.06796839797496801`) and first/last 20 tasks
  `0.7735259813070297`/`0.6909119826555253`.

  The paired UPGD-W-minus-AdamW mean is `0.06014679867029188` (sample SD
  `0.0018825070977402044`, SE `0.000595301014029226`), positive in 10/10
  pairs. A post-inspection descriptive t interval is
  `[0.05880013421741869, 0.06149346312316507]`; neither that interval nor the
  sign count is admissible as inferential evidence. Relative to approximate
  publication figure read-offs, the UPGD-W gap is about `-0.000853` and the
  AdamW gap is about `+0.039`; the AdamW difference is an explicit
  reproduction gap.

  The original `outputs/upgd_ipmnist/results.v1.json` is preserved and its
  strict validator rejects it only because the note does not preserve the
  exact 10-vs-20-seed limitation. The canonical reconciled artifact is
  `outputs/upgd_ipmnist/results.reconciled_nonpromoting.v2.json`; it passes
  strict structural validation and is bound by
  `outputs/upgd_ipmnist/nonpromoting_receipt.v2.json`, whose immutable
  predecessor is `nonpromoting_receipt.v1.json`. Both are permanently
  nonpromoting. The run used 10 rather than the publication's 20 seeds,
  seeded streams that are unseeded upstream, exact task-block boundaries
  rather than the upstream shifted logging convention, and float32 bias
  correction rather than upstream mixed float64 scalars. Its shards did not
  bind worker source, the full import closure, commands, environment, or
  dataset bytes at execution time. Post-hoc reconstruction cannot repair that
  execution-attestation gap. A scientific replication requires a fresh,
  source-bound full-20-seed run, not appended seeds. These values support
  no inferential, SOTA, or Alberta Plan claim. See
  `outputs/upgd_ipmnist/RUNBOOK.md` for the sealed historical procedure.
  The active future execution contract is the namespaced v3
  plan/single-learner-single-seed-shard/exact-merge lifecycle. It binds the
  exact config and selected hyperparameters, closed deviations, exactly 20
  fresh operator-reserved seed IDs, dataset bytes, runtime, full static local import
  closure, semantic commands, exact Cartesian coverage, and each shard's raw
  bytes. No v3 plan has been issued, no v3 result exists, and no fresh v3 seed
  has been consumed. V3 is permanently nonpromoting without independent
  execution attestation; see `UPGD_IPMNIST_V3_RUNBOOK.md`.
- **IPMNIST mechanism-combination screening** (development, never promotable)
  — `benchmarks/ipmnist_screening.py`, `outputs/ipmnist_screening/`. The
  stored record currently contains 144 screening shards (48 arms × 3 seeds)
  and 69 full-horizon confirmation shards. A 2026-08-02 read-only audit
  reconstructed every table mean below directly from the stored per-task
  curves, but rejected the campaign provenance. The checked-in
  `proxy_validation.json` says `proxy_validated: false`: AdamW controls match,
  while UPGD controls differ from their claimed full-horizon prefixes by up to
  `0.0084`–`0.0096`. `summary.json` covers only 132/144 current shards. The
  round-2 driver reads a nonexistent top-level `average_online_accuracy`, its
  logs contain the resulting `KeyError` and an empty-argument confirmation
  invocation, and the later `frontier2_results.json` has a shape the declared
  driver cannot emit. Its means are numerically recoverable by averaging
  `per_task_accuracy`, but its artifact provenance is not reconstructable.
  Finally, v1 shards bind no source closure, command, dataset bytes, or shard
  manifest. The table is therefore a set of historical development
  observations, not a validated proxy campaign or a result attributable to
  the current source. Mean online accuracy is shown against the repo's own
  10-seed published-config UPGD-W reproduction (`0.7791`):

  | Arm | Seeds | Mean ± SE | Scoping |
  |---|---:|---:|---|
  | `adamw_cbp_r3e4` | 3 | `0.80126 ± 0.00022` | protocol-pure (star, best) |
  | `adamw_cbp` | 10 | `0.79876 ± 0.00009` | protocol-pure |
  | `upgd_w_wd0005` | 10 | `0.78431 ± 0.00014` | protocol-pure (tuned) |
  | `upgd_l2init` | 3 | `0.78042 ± 0.00030` | protocol-pure |
  | `upgd_idbd` | 3 | `0.77895 ± 0.00020` | tie with control |
  | `sigma0_ndecay099` | 3 | `0.86245 ± 0.00034` | protocol-extended, campaign best |
  | `upgd_ema_norm` | 10 | `0.8536 ± 0.0001` | protocol-extended |
  | `upgd_ema_norm_sigma0` | 3 | `0.85051 ± 0.00025` | protocol-extended |
  | `upgd_ema_norm_wd0005` | 3 | `0.84745 ± 0.00008` | protocol-extended |
  | `sgd_ema_norm` | 3 | `0.83991 ± 0.00007` | protocol-extended |
  | `adamw_cbp_ema_norm` | 3 | `0.76895 ± 0.00226` | composition, erodes below control |

  The stored `adamw_cbp` means remain above the stored pool64 control at 200
  tasks; this is not an authenticated 60→200 trajectory. `upgd_w_wd0005`
  runs pool64 noise (measured pool64+harness-vs-batched-exact control
  delta at 200 tasks, seeds 0-2: `−0.0003`; the `−0.00012` previously
  quoted is not reproducible from the shipped artifacts — see
  `outputs/ipmnist_screening/AUDIT.md`). Scoping: protocol-pure arms keep
  the published raw input encoding (`adamw_cbp` is AdamW +
  continual-backprop unit recycling); the `ema_norm` family prepends online
  EMA input normalization and is therefore protocol-extended (an
  input-encoding change the published architecture does not include),
  reported on its own rows, never as the headline. `upgd_w_wd0005` shows the
  published UPGD-W hyperparameters are suboptimal on their own benchmark.
  Descriptive dissection suggested by the stored means (not authenticated as
  stable across a common 60→200-task source-bound trajectory):
  input conditioning contributes `+0.061` (`sgd_ema_norm` vs the `0.7791`
  reproduction), the utility gate `+0.011` (`upgd_ema_norm_sigma0` vs
  `sgd_ema_norm`), and the perturbation `+0.003` when normalization is
  present (`upgd_ema_norm` vs `sigma0`). The raw-input contrast inverts the
  last term: disabling the perturbation in the published raw-input
  configuration (`upgd_w_sigma0`, 60-task proxy) *costs* `−0.035` — the
  noise is load-bearing without normalization, and input conditioning
  substitutes for it. None of these is promotable without a fresh
  source-bound preregistered run per the v3 contract. Screening results
  support hypothesis generation only; the runbook is
  `outputs/ipmnist_screening/RUNBOOK.md`, the arm-by-arm record is
  `outputs/ipmnist_screening/FINAL_REPORT.md`, and the mechanism analysis is
  `CONTINUAL_LEARNING_THEORY.md` (three-failure-modes theory with
  pre-registered outcome matrix and the dissection-cascade addendum; first
  refutation recorded: utility-gated protection on the AdamW+CBP base
  *costs* `−0.0055` on this no-recurrence protocol, exactly the tie/hurt
  branch the matrix anticipated — protection pays rent only where tasks
  recur).
- **Label-permuted EMNIST, UPGD protocol** (Elsayed & Mahmood ICLR 2024:
  EMNIST balanced 47 classes, labels permuted every 2,500 steps, 400 tasks,
  1M online examples) — `benchmarks/upgd_label_emnist.py`; first artifact
  `outputs/upgd_label_emnist/results.v1.json` (3 seeds versus the paper's 20,
  a documented deviation). It reproduces the published qualitative
  separation: UPGD-W online accuracy rises through the life (last-quarter
  mean `0.728`, whole-run `0.672` versus a ~`0.74` figure read-off) while
  AdamW collapses (whole-run `0.201` versus a ~`0.35` read-off); both
  whole-run gaps are flagged as reproduction gaps. Development-only,
  `scientific_promotion_allowed: false`.
- **Slowly-Changing Regression** (Dohare et al. Nature 2024: m=20 bits,
  f=15 flipping, T=10,000, 100-LTU target net, 3M examples, 100 runs) —
  `benchmarks/slowly_changing_regression.py` and
  `benchmarks/slowly_changing_regression_v2.py` (60+ focused tests). The v2
  lane is a publication-shaped development extension, not an exact
  replication: the selected ordinary-BP arm uses ReLU Kaiming initialization
  and true-MSE gradients, while CBP and UPGD are Alberta-local extensions. Its
  strict plan/shard/artifact schemas reject duplicate/non-finite JSON, bind
  source/run/runtime digests, require one method/seed per immutable shard and
  the exact planned Cartesian product, bind canonical semantic commands, verify
  shared environments, exactly replay every shard at merge and ordinary
  validation, and reconstruct descriptive results without a post-hoc gate.
  Structural-only diagnostics are explicitly nonvalid. Any future
  self-recorded plan is not externally attested, so promotion is permanently
  forbidden. An earlier reduced pilot was not archived; its values remain
  excluded. The historical `replication.v1.json` is absent and is not
  retrofitted. Execution status (2026-08-01): an immutable v2 pre-run plan
  was issued on 2026-07-31 (`outputs/slowly_changing_regression/plan.v2.json`,
  3 methods × 100 seeds = 300 shards) and the sharded run was launched. 49
  `publication_bp_relu_sgd` shards completed; every remaining shard attempt
  then failed closed with `current source bytes differ from the pre-run plan`
  after working-tree edits diverged from the plan's bound source bytes.
  Completing the lane requires issuing a fresh plan against a quiet source
  tree; no merged artifact exists and no result is claimed. The launch
  procedure is in `outputs/slowly_changing_regression/RUNBOOK.md`.
- **Foragax (Forager) field-of-view lane** (Tang et al. 2026 protocol):
  tuning stage complete (winner `step3e3`, CI-based selection, tuning→eval
  hash link). The old 500k×30-seed evaluation produced no batch or report and
  is inactive. Its official-DQN and relearning companion directories are
  explicitly quarantined because they were terminated under superseded
  schemas/source snapshots; they must not be resumed, imported, or compared.
  A separate RTU-RTRL 4×500k open-development GPU run completed with FOV
  tail-EMA AUC 1.550 ± 0.324 sample SD and is preserved in
  `outputs/forager/rtu_rtrl_500k_dev4/receipt.v1.json`. A reconciled unsealed
  DQN receipt on the same four seeds records a descriptive `+0.331`
  RTU-minus-DQN mean, but the comparator was configured post-output and has
  unmatched runtime, representation, resources, and update work. It is not an
  admissible pair and neither receipt has promotion standing.
  Current matched-source uses candidate-universe schema v2 (SHA-256
  `6a9315cb…`) and has no renewed qualification/open artifact; it requires a
  fresh qualification and new namespace before execution. The immutable
  historical v1 campaign (candidate universe SHA-256 `2c3b214c…`, 23
  registered arms) completed executor qualification
  (`outputs/forager/matched_current_qualification_2c3b214c_v1`), but the
  prepared open-tuning stage
  (`outputs/forager/matched_current_open_tuning_2c3b214c_v1`) has executed
  zero cells — its `runs/` and `completions/` are empty — and the sealed
  held-out stage (seal, 6×30 schedule, final analysis, statistics) is
  implemented and tested but has never run. Its evaluation runner is exposed
  as `alberta-forager-matched-sealed-evaluation`; seal and final analysis remain
  module-only. Every authority path requires an external trust resolver that
  does not exist in-tree; the 2026-07-31 audits recorded implementation GO but campaign
  authority NOT CLEARED (`FORAGER_ALBERTA_CANDIDATE_AUDIT.md`,
  `FORAGER_COMPARATOR_AUDIT.md`).
- **OPMNIST 800-task closure** (the ROADMAP's published-scale multi-seed
  boundary): 3 seeds × 48M updates running detached in the upstream tree
  with per-chunk checkpoints and status sidecars. As of 2026-07-31 the
  workers were advancing, but no merged result or gate outcome exists and
  nothing has been ingested here: the fail-closed ingestion/validation
  contract is defined in `OPMNIST_DEVELOPMENT_INGESTION.md` and no bundle
  exists under `outputs/`. Merge and gate commands are recorded in the
  upstream runbook. Neither `multi_seed_full_scale` nor
  `solved_opmnist_step2` is predicted or claimed before the completed
  artifacts are validated.

## Exploratory longevity and scaling observations

The one-million-step longevity extension is now reproducible in-tree:
`tests/test_lifetime_longevity.py` runs the identical oracle-gated lifetime
protocol for 125 cycles (1M steps) x 8 seeds (~45 s including compile) and
asserts: zero non-finite steps over 8M updates, late-life savings ≥ 5x with
no erosion vs early life (measured medians 10–18x across the whole life),
last-20-cycle re-entry error < 0.25x first exposure (measured 0.19 vs 2.00),
and flat unstressed fresh-task adaptation (slope ≈ +0.0002/cycle).  Same
mechanism-diagnostic scope as the 64k version — what it adds is the horizon
(~375 interference events, ~41 scale shocks, 124 recurrences in one life).

Likewise, the following 24-core/JAX observations came from benchmark drivers
in the omitted upstream `benchmarks/` tree. They are engineering hypotheses
until a versioned local runner and artifact reproduce them:

- **Embarrassing axes (vmap/scan)**: seeds (128-seed full-gauntlet run in
  7.9 s — 16x seeds for 5.6x time), independent agents (64 vmapped
  agent+world pairs: 2.27M agent-steps/s; 32x agents for 2.6x time), and time
  itself (`lax.scan`).
- **The wall**: the demon/head axis.  `MultiHeadMLPLearner.update` unrolls a
  Python loop over heads inside jit (`multi_head_learner.py:801`), so horde
  throughput collapses 4,255 → 14 steps/s from 8 → 1024 demons (144 s compile,
  16.4 GB).  A GPU does not fix this — it is program-size growth.
- **Dispatch overhead**: single jitted calls cost ~10x their scanned
  amortized cost (PrototypeAgent 0.81 vs 0.075 ms/step); daemon-style
  deployments should micro-batch steps.
- **Top 3 candidate levers**: (1) vmap the demon axis (stacked head params) in
  `MultiHeadMLPLearner`/`IndependentDemonHorde`;
  (2) batched worlds (vmap agent+environment pairs) for population-scale
  multi-agent studies; (3) shard the seed axis across devices and micro-batch
  daemon dispatch.

## Pinned scale-robust package result

The scale-robust pair-feature v2 protocol was frozen on direct-key development
seeds 8–15, then run once on 30 exact seeds derived from the
``alberta-scale-robust-v2-fresh-evidence-2026-07-30:`` SHA-256 namespace. Its
immutable artifact at ``outputs/scale_robust_feature/evidence.v2.json`` was
accepted L2 under its frozen source; the scientific digest is
``c2fee922c04a59fe26b4b8c9cfa77ddd9198cfa2bc923f54fec14b649bd3bb2c``.

The current registry reports this claim as invalid because the registered
interaction-feature source has evolved. The artifact is preserved byte for
byte and is historical evidence only for that frozen implementation. Reusing
its consumed seeds can establish compatibility but cannot promote; a renewed
claim needs a new path/schema and untouched preregistered seeds.

The primary arm achieved median C/D/final-C savings
``10.968 / 12.491 / 5.933``, final-C tail MSE ``0.038707`` (per-seed maximum
``0.049835``), and scale-shock early/cumulative/tail MSE
``48.270 / 14.123 / 4.690`` (per-seed maxima
``126.982 / 44.035 / 37.326``), with no non-finite step. Paired mean
primary-versus-legacy final-tail error reduction was
``8.725 [8.239, 9.194]`` and final-C savings gain was
``4.869 [4.410, 5.358]``. Before final C reacquisition, the retained arm's
structural C- and D-product advantages over no retention were
``1.300 [1.000, 1.600]`` and ``2.700 [2.267, 3.133]``.

This supports the frozen package comparison, not causal attribution to an
individual mechanism. Primary versus legacy changes scale normalization and
ObGD together and adds 464 persistent bytes. All stream seeds share one fixed
learner initialization; context is visible; every one of the 91 finite
pair descriptors and its weight stays resident. The retained-product result
is structural and does not show those columns causally contributing to an
output. It is not task inference, general feature finding, continual control,
or L3 integration.

## Development advances in wave 3 (not promoted closure)

1. **Corrupt-dream regret — development comparison**
   (`test_planning_benefit.py::test_step9_gate_strictly_reduces_corruption_regret`).
   Three self-neutralization channels traced numerically: (i) dreams updating
   r̄ (now ``dreams_update_average_reward=False`` by default — and measured
   strictly worse for both arms when enabled under discounted updates);
   (ii) the additive linear world model cannot represent XOR payoffs, so
   corrupt dreams only shifted Q *levels* (fixed via
   ``model_include_action_interactions`` passthrough); (iii) post-flip dream
   TD errors ≈ 0 while real unlearning outruns them (mitigated with amplified
   exposure).  With all three handled: permissive-minus-gated post-flip
   regret mean gap 40.4 / median 19.0 over 96 disjoint verification seeds,
   ~8% of permissive seeds lock catastrophically (regret up to 559) while the
   gated agent never exceeds 145. The old non-inferiority test is retained,
   its config now documented as the self-neutralizing regime. This is not a
   frozen matched-budget planning artifact and does not close Steps 7 or 9.
2. **Single-life integration — development diagnostic**
   (`test_integrated_life.py`, 12 tests, 54 s).  ONE agent, ONE uninterrupted
   48,000-step closed-loop life: 120 payoff switches (60 recurrences per
   rule) plus a mid-life all-channel input-noise stressor, 8 seeds, per-seed
   paired no-context ablation.  Gated rung: lifetime reward ≥0.969 (optimum
   1.0), re-coordination flat at ~0.97 from recurrence 1 through 59, paired
   memory gap vs ablation +0.60 per seed, recovery time at the metric floor
   early AND late, zero NaN, bit-stable state shapes, stressor survived with
   ≤0.005 lasting drop.  PrototypeAgent rung on the same life: ≥0.88 lifetime,
   no plasticity trend, bounded state across 11 curations — its *retention*
   is honestly weak (linear Q over [x, ctx] cannot represent both rules), so
   the memory observation rests on rung 1 and its causal ablation. Context is
   visible, the strong representation is hand-built, all eight seeds were
   used for calibration, and there is no immutable held-out artifact or
   matched full-stack ablation. It is therefore not L3 evidence.
3. **Discovery-driven control — development integration**, in its
   reward-prediction-target form only
   `test_discovery_control_life.py`, 10 tests).  Features discovered ONLINE
   (action-conditioned next-reward prediction) feed a
   ``DifferentialSARSAAgent(use_bias=False)`` whose Q-weights carry over by
   feature identity across bank refreshes.  Every seed's bank finds 3-4 of
   the 4 oracle ctx×state products during control; recurrence early-window
   reward 0.939 vs raw twin 0.751; the discovered bank recovers 92-94% of
   the oracle-representation advantage; the no-carry ablation isolates
   weight-carry-over-on-persistent-features as the memory mechanism (-0.095).
   Works both two-timescale (refresh every 50 steps) and fully simultaneous,
   but uses observable context and calibrated tests rather than promoted
   held-out evidence.

## Wave-4 development diagnostics (nonpromoting)

1. **Context inference** (`core/context_inference.py` +
   `test_context_inference.py`, 13 tests). `ContextInference` maintains a
   bounded bank of per-(state, action) reward tables, updates only the active
   slot, and gates control features by the inferred slot. On the tested hidden
   two-rule life, the paired gap over the no-context ablation was +0.519 mean
   / +0.493 minimum, per-era phase agreement was at least 0.9999 over the 24
   inspected seeds, and clean flip-detection lag was four steps. This is
   development evidence: thresholds were calibrated on those same three
   eight-seed batches, the model is specific to the environment's reward
   table and scale, the inferred-slot × observation feature map is
   hand-designed, and the comparison is resource-unmatched. It has no partner,
   learned world model, planning, or feature lifecycle.
2. **TD/GVF-target discovery** (`test_td_target_discovery.py`, 9 tests).
   Candidates predict the controller's differential-SARSA target
   ``r - rbar + Q(s', a')``. The bank found at least three of four
   context-binding products per seed and beat the raw twin by about +0.16 on
   recurrence, while an information-free target stayed near its chance
   feature count. The immediate-reward target retained a small advantage.
   All four eight-seed batches informed the reported robustness and calibrated
   thresholds; context is oracle-visible, the archive exhausts all 15 pairs,
   and the discovery arm is not resource-matched to raw control. There is no
   versioned held-out artifact.
3. **PrototypeAgent-level retention**
   (`test_prototype_discovered_retention.py`, 7 tests). Gated features raised
   settled recurrence from about 0.739 to 0.923; a pair bank learned during
   the first 4,000 steps and then frozen reached about 0.916. The diagnostic
   also exposed a world-model LMS divergence and uses a 0.05 step size for the
   eight-slot discovered arm versus 0.2 for the four-feature arms. Its eight
   development and eight robustness seeds were inspected; context remains
   oracle-visible, the bank is frozen for the final 20,000 steps, discovered
   state has 1,040 elements versus 440 in the plain/gated arms, and
   per-component planning/option ablations are absent. The discovered arm's
   lifetime reward is slightly below the plain arm, so this is neither a
   matched package comparison nor L3 closure.
4. **Historical wave-3 confirmation** (`test_held_out_confirmation.py`,
   11 tests). The precommitted floors passed on their first then-uninspected
   batches: primary-only scale discovery on seeds 30–45, discovery-control on
   eight keys at 1000+, a hand-gated life on eight seeds at 5000+, and
   oracle-feature longevity on one base key split eight ways. This is useful
   development robustness evidence, but it has no common versioned artifact,
   strict source chain, or matched full-stack comparison. The primary-only
   scale run exposed seeds 30–45 and cannot stand in for v2; the pinned v2
   result used the separate namespace-derived schedule above but is currently
   source-invalidated. The handcrafted
   context/gating and small seed batches keep the remaining historical checks
   nonpromoting.

## Hidden-partner development integration

`hidden-partner-mapping-v0`, `BehaviorModel`, the bounded external-belief joint
world model, and `FeatureBankRouter` now compose with learned recurrent state,
bounded pair discovery, descriptor-routed consumers, and explicit-action
differential SARSA in one fixed-shape kernel. The stream hides regime and
boundary channels, but its stochastic partner is scripted and nonlearning.

The versioned development artifact runs eight uninterrupted lives with matched
state, memory, lifecycle, carry, retention, planning, partner-belief, and
curation ablations. All eight full lives are finite and contract-valid, and the
full condition beats lifecycle-frozen and random-curation controls on every
paired seed. Its development gate still fails: obsolete D is absent from the
active bank at life end in `0/8` full lives. The artifact declares
`development_only: true` and `scientific_promotion_allowed: false`; the fixed
schedule family, closed exhaustive 66-pair archive, and scripted partner keep
this at L0/development integration rather than coadaptation, general forgetting
resistance, or L3.

## Remaining open items

1. **Dreams vs one-hot observations**: `PrototypeAgent` has an opt-in,
   fixed-proposal-budget `sample_one_hot` dream-observation mode with isolated
   categorical randomness; the legacy model-prediction path remains the
   default. On eight already-consumed RiverSwim development seeds, sampled
   one-hot dreams raised mean lifetime reward from `0.1930` to `0.2475`
   (paired `+0.0545`; 6/8 seeds positive, range `-0.0769` to `+0.1670`).
   This arm was designed after observing the seeds, so it is an L0
   nonpromoting diagnostic. General and held-out efficacy remain open.
2. **Compositional next steps** (research program, not gaps in the
   demonstrated claims): context inference feeding PrototypeAgent end-to-end
   (both are only separate development diagnostics); TD-target discovery under
   *hidden* context; utility-driven multi-consumer curation with realized
   causal deletion benefit beyond the narrow owner-bound pair bank; persistent
   consumption and lifecycle validation for the standalone cumulant/subtask
   proposals; automatic option discovery and lifecycle; richer regime models
   beyond per-(state, action) reward tables;
   and integration of the existing learning-partner rung into the
   hidden-context/world-model stack. Seeds 46–59 were not
   part of the older confirmation, but v2 excludes the whole 30–59 namespace
   rather than treating a partially exposed range as fresh.
3. External items (rlsecd integration, physical-robot transfer) require
   assets outside this repository.
