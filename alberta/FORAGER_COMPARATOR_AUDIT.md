# Forager Comparator Completeness Audit

Date: 2026-07-31

> **Frozen-v1 status:** `matched_current_open_tuning_2c3b214c_v1` is immutable and
> nonmodifiable. This audit records comparator provenance and limitations. It does not
> authorize edits to v1, evidence promotion, or performance claims. As of 2026-08-01 no
> open-tuning cell and no held-out evaluation has been executed; every limitation below
> is therefore prospective, constraining how a future v1 result may be interpreted.

## Technical summary

The official upstream repository contains one production learning configuration for the
exact stationary task `ForagaxTwoBiomeLarge-v1` with aperture 9, plus one development DQN
sweep configuration. It also contains Random, Search-Nearest, and Search-Oracle baselines
for that environment. Frozen v1 registers the production DQN and Search-Oracle, but not
Random or Search-Nearest.

The registered LayerNorm DQN, CReLU DQN, ReDo, DRQN, PPO, and RTU-PPO arms broaden the
method panel through explicit task adaptations. **None of those arms is an exact upstream
configuration replay on the stationary task.** The inferential PPO and RTU-PPO arms also
use a reviewed RNG-isolation derivative rather than the exact upstream implementation.

This is therefore a frozen matched-panel comparison, not a replay of every upstream
experiment and not an exhaustive literature comparison. Compute, replay, optimizer,
parameter, and state budgets are unequal by design and must remain visible when the panel
is interpreted.

## Scope and source lock

The audit was read-only. It did not execute benchmarks or inspect reward arrays.

| Item | Audit-time record or immutable identity |
|---|---|
| Official repository | `https://github.com/steventango/continual-foragax-agents.git` |
| Audited branch | `main`, verified against the remote on 2026-07-31 |
| Commit | `9710f60fa30da5badc451ad7ce3ff296d5070830` |
| Git tree | `a5ad878ac4be0567c43dfd9177471c4b5a910bfa` |
| Audited local checkout | `/tmp/foragax-agents-official` |
| `continual-foragax` version | `0.55.0` from upstream `uv.lock` |
| Frozen candidate universe | SHA-256 `2c3b214cf29e013e3f8d88b2558bd94f75e92330bf0ddcc6afd7514279a1ee77` |
| Frozen open protocol | SHA-256 `b17da8af19cac570c426c74ff6bbc0e4ee0a4b95a4486c3ad5da19ceb3f8176e` |
| Primary paper | [Forager paper, arXiv:2605.01131v1](https://arxiv.org/html/2605.01131v1) |

The branch name, verification date, and temporary checkout are audit-time context. The
commit, Git tree, version, and frozen descriptor hashes are the immutable identities.

The exact frozen task is:

- environment: `ForagaxTwoBiomeLarge-v1`
- aperture: 9
- observation: color
- horizon: 499,712 environment steps

The frozen descriptors are:

- `outputs/forager/matched_current_open_tuning_2c3b214c_v1/candidate-universe.json`
- `outputs/forager/matched_current_open_tuning_2c3b214c_v1/open-protocol.json`

## Exact-task upstream inventory

A repository-wide scan for `ForagaxTwoBiomeLarge-v1` found only E138 production and
development files. No PPO, RTU-PPO, DRQN, ReDo, mitigation, LayerNorm-DQN, or CReLU-DQN
configuration uses the exact environment upstream.

| Upstream configuration | Exact-task relationship | Frozen-v1 disposition |
|---|---|---|
| `experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json` | Only production learning config at exact aperture 9 | Registered as inferential `external_dqn_plain`, with declared horizon and diagnostic transforms |
| `experiments/E138-two-biome-large/foragax-sweep/ForagaxTwoBiomeLarge-v1/9/DQN.json` | 50k development sweep with hyperparameter arrays and development seeds | Not registered; not an inferential production comparator |
| `experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Random.json` | Same environment; action policy ignores observations, although config uses aperture 1 | Absent; suitable future descriptive sanity floor |
| `experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Search-Nearest.json` | Same environment, but privileged full-world object access | Absent; suitable future descriptive context only |
| `experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Search-Oracle.json` | Same environment, privileged full-world access and known reward ordering | Registered as descriptive-only `search_oracle` |
| E138 DQN apertures 3, 5, 7, 11, 13, and 15 | Different observation tasks | Correctly outside exact-aperture inference |

Pinned upstream links:

- [E138 FOV9 DQN](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json)
- [E138 baselines](https://github.com/steventango/continual-foragax-agents/tree/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines)

## Frozen-v1 external mapping

“Transfer” below means that upstream code or a configuration family was adapted to the
stationary environment and frozen horizon. A transfer is **not an exact upstream replay**.

| Candidate | Role | Configuration provenance | Replay status |
|---|---|---|---|
| `external_dqn_plain` | Inferential | Exact E138 FOV9 JSON plus declared `total_steps`, `ntk_freq`, and `x_ref_steps` transforms | Exact upstream implementation; transformed config, not an exact run replay |
| `external_dqn_ln` | Inferential | `fov_baseline_screening_v1/configs/DQN_LN-common-control.json` | Local stationary-task control using upstream implementation; not an upstream E138 config replay |
| `external_dqn_crelu` | Inferential | `fov_baseline_screening_v1/configs/DQN_CReLU-common-control.json` | Local stationary-task control using upstream implementation; not an upstream E138 config replay |
| `external_dqn_redo` | Inferential | `fov_stateful_baseline_screening_v1/configs/DQN_ReDo_PostLNScore.json` | SquareWave-derived task transfer; not an exact upstream replay |
| `external_drqn_paper` | Inferential | `fov_stateful_baseline_screening_v1/configs/DRQN-paper-v1.json` | Paper-parameter task transfer; not the upstream XFinal DRQN config replay |
| `isolated_ppo` | Inferential | `fov_stateful_baseline_screening_v1/configs/PPO_2048_relu.json` | SquareWave/RGB-to-stationary/color transfer on reviewed RNG-isolated source; not an exact upstream replay |
| `isolated_rtu` | Inferential | `fov_stateful_baseline_screening_v1/configs/PPO-RTU_LN_128_1_relu.json` | SquareWave/RGB-to-stationary/color transfer on reviewed RNG-isolated source; not an exact upstream replay |
| `exact_ppo` | Descriptive only | Same adapted PPO config on exact upstream shared-RNG implementation | Source-orientation diagnostic only; still not an original upstream task/config replay |
| `search_oracle` | Descriptive only | Exact upstream privileged algorithm with horizon transform | Not eligible for learning-comparator inference |

The exact upstream shared-RNG RTU-PPO orientation is bound by the source screen but omitted
from the registered panel. The candidate-universe descriptor records this as an explicit
scope limitation.

## Material mismatches and interpretation limits

### Frozen v1 cannot identify the best arm across the registered universe

The open stage evaluates all 14 Alberta and seven external inferential candidates on ten
499,712-step tuning seeds. It ranks each group by the lower endpoint of a two-sided 95%
candidate-wise bootstrap interval, not by the candidate's tuning mean, then advances one
Alberta candidate and three external candidates. The held-out stage evaluates only those
four selected candidates plus two fixed descriptive references on 30 new seeds.

Consequently, held-out v1 executes three named Alberta-vs-external contrasts among four
tuning-selected inferential arms; the two fixed references are descriptive only. It
supports neither a winner among the six executed arms nor a held-out best among all 21
inferential candidates, and it cannot support the candidate-universe descriptor's broader
“best among the frozen registered matched candidate panel” wording. That wording is an
immutable-v1 claim-scope defect, not permission to reinterpret or edit the frozen protocol.
Any result must remain conditional on the declared selection rule and must not be called a
panel-wide winner or SOTA.

### Frozen secondary sign-flip gates are nonconfirmatory sensitivity analyses

The two secondary hypotheses name a paired mean difference but compute paired sign-flip
p-values followed by Holm adjustment. Pairing alone does not make a sign-flip test exact
for a mean null: it requires a null distribution invariant to the individual sign flips
(for example, symmetric paired differences or randomized exchangeable method labels).
Frozen v1 preregisters no such identifying assumption or randomization mechanism.

The secondary p-values, adjusted p-values, and reject flags may therefore be reproduced as
frozen descriptive sensitivity calculations, but they must not be treated as valid
confirmatory mean-superiority tests. A future protocol must preregister a justified
sampling model and compatible test, or use a calibrated bounded-mean procedure, before
secondary gates carry inferential weight. Holm adjusts valid p-values; it cannot repair
p-values that lack sign-exchangeability.

This defect is separate from the frozen primary calculation, but the ordinary paired
percentile bootstrap is not assumption-free either. The protocol states no seed-
superpopulation, sampling model, or bootstrap regularity conditions. Its nominal 95%
lower endpoint is therefore a frozen resampling summary, not an established population-
level confidence bound.

### Frozen v1 tuning replays neither reconstructed nor current FOV tuning

Alberta's reconstructed historical NumPy FOV preset declares five 10,000-step tuning
runs; its provenance is explicitly not an upstream attestation. The pinned current
Foragax E138 sweep declares five 50,000-step runs. Frozen v1 instead gives every
inferential candidate ten 499,712-step tuning seeds, so it replays neither protocol.
The horizon arithmetic is internally consistent (`244 × 2,048 = 3,904 × 128 =
499,712`); the difference is a reproduction-scope limit, not an execution defect.

### The E138 FOV9 horizon is internally inconsistent upstream

The checked-in FOV9 DQN JSON says `total_steps: 10000`. The other six production
apertures, all three E138 baselines, the E138 plotting/AUC assumptions, and the paper use
500k steps. The production job does not visibly override `total_steps`, so the checked-in
FOV9 file must not be silently described as a 500k exact replay.

Frozen v1 explicitly changes it to 499,712 steps and disables diagnostic-only NTK fields.
The result is a registered matched-horizon transform. It is 288 steps below nominal 500k
and retains a 400k epsilon-decay schedule.

### Replay, exploration, and optimizer controls are unequal

- Plain, LayerNorm, and CReLU DQN use replay capacity 10,000, minimum replay 32,
  batch 32, update frequency 4, and target refresh 128.
- ReDo uses replay 1,000, minimum replay 50, fixed epsilon 0.1, and learning rate 0.003.
- The registered paper-parameter DRQN uses replay 1,000, sequence length 32, burn-in 16,
  batch 4, fixed epsilon 0.1, and learning rate 0.001.
- Upstream XFinal `DRQN.json` instead uses batch 32, epsilon 0.25, and learning rate
  0.0001. That separate current-XFinal arm was screened and excluded.

### PPO and RTU-PPO are task and state-construction transfers

The source R1 configurations use `ForagaxSquareWaveTwoBiome-v11`, generally 10M steps,
and RGB observations. The frozen configurations use the stationary environment, color,
and 499,712 steps.

The upstream PPO route constructs network inputs from the image, previous-action one-hot,
and previous reward; optional phase or trace features may be appended. RTU-PPO additionally
maintains recurrent state. Those inputs were developed for a hidden reward-switching task,
whereas the frozen target is stationary. The transfer changes the scientific setting in
addition to the horizon and observation encoding.

The exact upstream PPO runner also reuses one derived RNG key for policy sampling and
environment stepping in `src/rtu_ppo.py`. Frozen inferential PPO and RTU-PPO use a reviewed
RNG-isolation derivative. Therefore their historical shared-RNG screen values do not
transfer as inferential support.

Relevant upstream configuration families are located under:

- `experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/`
- `experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11/9/`
- `experiments/R3-plasticity-ReLU-LN/foragax/ForagaxSquareWaveTwoBiome-v11/9/`
- `experiments/E139-ppo-plasticity/`

These include L2, L2-to-initialization, reset, reward-trace, shrink-and-perturb,
ReLU/CReLU, LayerNorm, DRQN, PT-DQN, PPO, and RTU-PPO variants. None supplies exact-task
inferential evidence for the frozen stationary protocol.

### Search baselines are not matched-observation learners

`src/algorithms/SearchAgent.py` uses the full object world and graph search.
Search-Nearest assigns equal priority to the two positive object classes; Search-Oracle
encodes their reward ordering. Both are privileged relative to aperture-9 color agents.
Random is nonprivileged in behavior because it ignores the observation, but it remains a
nonlearning reference rather than a learning-comparator candidate.

## Frozen resource accounting

These are protocol-bound counts, not general properties of the methods.

| Candidate | Protocol parameter count | Optimizer updates | Replay capacity | Protocol `recurrent_state_elements` |
|---|---:|---:|---:|---:|
| causal-map q grid (9 arms) | 0 | 0 | 0 | 1,847 |
| `alberta_horde_default` | 41,285 | 499,712 | 0 | 0 |
| `alberta_horde_eps05` | 41,285 | 499,712 | 0 | 0 |
| `alberta_horde_recurrent64` | 49,477 | 499,712 | 0 | 64 |
| `alberta_horde_step3e3` | 41,285 | 499,712 | 0 | 0 |
| `alberta_rtu_h08_taylor` | 4,685 | 499,712 | 0 | 32 |
| `external_dqn_plain` | 87,876 | 124,920 | 10,000 | 0 |
| `external_dqn_ln` | 88,164 | 124,920 | 10,000 | 0 |
| `external_dqn_crelu` | 92,228 | 124,920 | 10,000 | 0 |
| `external_dqn_redo` | 20,612 | 124,915 | 1,000 | 0 |
| `external_drqn_paper` | 49,540 | 124,915 | 1,000 | 64 |
| `isolated_ppo` | 305,381 | 31,232 | 0 | 0 |
| `isolated_rtu` | 452,069 | 15,616 | 0 | 288,768 |

The table covers all 14 Alberta inferential arms (the causal-map row spans its nine
q-grid arms), but its columns are not a complete or cross-method memory/compute census. Parameter counts exclude optimizer state and target
snapshots. PPO's replay value of zero excludes 2,048/128-step rollout storage, and causal
optimizer updates of zero exclude 499,712 non-gradient state updates. Recurrent64 Horde
also has a 61,248-element fixed substrate outside its 49,477 trainable parameters. The
causal-map arms have no trainable parameters but carry a 1,847-element learned finite state.
Local RTU's value 32 counts actor/critic carry only; it excludes 576 RTRL sensitivities,
576 Taylor sensitivities, 4,685 eligibility-tree elements, normalization/history, and RNG.

Alberta Horde and local RTU receive the full flattened image plus engineered channel means,
the previous action/reward, and three reward traces with decays 0.9, 0.99, and 0.999.
Registered DQN/DRQN receive the image plus previous action/reward through `NNAgent`
defaults; PPO/RTU-PPO likewise receive image plus previous action/reward. The unmatched
additions are chiefly Alberta's means/traces and the causal planner's structural priors.
That planner uses public 15×15 toroidal geometry, the four-action convention, and the
public move/reward-before-respawn transition order. It does not consume evaluator state,
global position, the hidden object grid, biome labels, or evaluator `info`.

Consequently, any eventual matched-panel result is conditional on the frozen method and
resource bundle. It does not isolate architecture, memory, replay, or compute as a causal
factor, and it does not establish data-, input-, or prior-knowledge equivalence.

## Screened and omitted arms

The screens below are open-development candidate-generation provenance. Their rankings are
not scientific evidence and do not support claims about the frozen matched candidates.

| Screen | Registered representatives | Screened exclusions |
|---|---|---|
| `dqn_common_control_v3` | LayerNorm DQN, CReLU DQN, plain DQN | reward trace, L2, reset head, causal history, causal history plus reward trace, L2-to-init, shrink-and-perturb, SWR |
| `stateful_corrected_v4` | isolated RTU-PPO, isolated PPO, ReDo, paper-parameter DRQN; exact PPO descriptive only | ReDo architecture control, current-XFinal DRQN, PT-DQN, PT-DQN architecture control |

Other explicit omissions are:

- exact upstream shared-RNG RTU-PPO orientation;
- Search-Nearest descriptive privileged context;
- Random descriptive sanity floor;
- unscreened PPO mitigation and architecture variants from the SquareWave experiments.

These are frozen panel-scope choices. They must not be interpreted as negative findings
about an omitted method.

## Future-cycle recommendations

All recommendations apply only to a separately versioned future protocol. **Do not modify
frozen v1 or its pinned outputs.**

1. Either evaluate every inferential candidate on held-out seeds before making a
   panel-best claim, or define the estimand explicitly as the performance of the frozen
   tuning-selection procedure and limit claims to the selected candidates.
2. Replace raw paired sign flips with a preregistered joint seed-block procedure for mean
   differences, such as a centered studentized multiplier/bootstrap max-T construction
   with simultaneous one-sided lower bounds. State the seed-superpopulation and bootstrap
   regularity assumptions, and freeze any hierarchical primary/secondary gate in advance.
3. Add Search-Nearest and Random as descriptive-only exact-environment references. Bind
   Random to aperture 9 or record and test its observation independence explicitly.
4. Add the exact shared-RNG RTU-PPO orientation if symmetric source-orientation accounting
   with `exact_ppo` is desired.
5. Resolve the E138 FOV9 10k-versus-500k inconsistency through upstream provenance. Keep
   any future horizon transform explicit rather than describing it as an exact replay.
6. Tune transferred PPO, RTU-PPO, ReDo, and DRQN configurations only on new development
   seeds for the exact stationary task. Keep those seeds outside any future held-out set.
7. Include the paper/XFinal generic PPO and Real-Time PPO settings as distinct configs
   rather than treating the current R1 ReLU transfers as their exact representatives.
8. Consider a frozen future screen for PPO L2, L2-to-init, shrink-and-perturb, reset,
   CReLU, and LayerNorm controls.
9. Add explicit resource-control arms for replay 10k versus 1k, fixed versus decaying
   exploration, previous-action/reward inputs, parameter count, and optimizer updates.
10. Continue binding source commits, trees, configuration hashes, and field-level
   transforms. Automate an exact-environment inventory before freezing the next universe.

No recommendation above changes the evidence status of v1 or upgrades any current claim.

## Primary-source literature addendum (2026-07-31)

This addendum distinguishes the exact stationary task from related Forager tasks. It is a
read-only literature and official-source audit; it does not add a candidate to frozen v1,
change a result, or establish a general Forager leaderboard.

### No published exact-task learning SOTA was found in the bounded audited corpus

The bounded corpus comprised the pinned official repository, the primary Forager paper,
its cited experiment/configuration families, and the 2026 streaming RTU-RTRL paper below;
it was not a systematic review of every publication venue. No result found there is
directly comparable to `ForagaxTwoBiomeLarge-v1`, color observation, aperture 9, and a
499,712-step horizon. The [Forager paper](https://arxiv.org/html/2605.01131v1) reports a
field-of-view experiment with DQN and 30 independent trials, but not an exhaustive
learning-method ranking for this exact stationary configuration. Accordingly, v1 cannot
call a learning arm “Forager SOTA.” Any future SOTA claim requires a fresh, documented
exact-task literature search. Frozen v1 can report only its three named held-out
calculations, conditional on the configured policies, resources, task, horizon, seeds,
and metric.

The sole upstream production learning configuration at this exact environment and aperture
is [E138 FOV9 DQN](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json).
It specifies `total_steps: 10000`, even though the other E138 apertures, baselines, and the
paper's FOV discussion use a nominal 500k scale. Its concrete configuration is a
`Forager2Net` with two 64-unit hidden layers, Adam (`alpha=3e-4`, `beta1=0.9`,
`beta2=0.9`, `eps=1e-8`), gamma 0.99,
epsilon 1.0-to-0.05 over 400k steps, replay capacity 10,000, batch/minimum history 32,
update frequency 4, and target refresh 128. The paper independently reports FOV9's
selected step size `3e-4`, update frequency 4, and target update 128
([Tables 4–5](https://arxiv.org/html/2605.01131v1#S14.T4)). A 499,712- or 500k-step
execution is therefore a matched-horizon transform, not an exact replay of that JSON, and
must bind the field-level changes, new tuning data, and a fresh held-out evaluation.

The exact-environment
[Random](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Random.json),
[Search-Nearest](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Search-Nearest.json),
and [Search-Oracle](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Search-Oracle.json)
should be added only as descriptive context in a separately versioned protocol. Random's
checked-in configuration names aperture 1, so a future protocol must either bind aperture
9 or establish and record observation-independence. Search-Nearest and Search-Oracle
request object/full-world access; Oracle also encodes the reward ordering. They are
privileged references, not matched-observation learners.

### Strongest related result: nonstationary RTU-PPO, not stationary evidence

The primary paper's strongest qualitative learning result is RTU-PPO on
`ForagaxSquareWaveTwoBiome-v11`, not on `ForagaxTwoBiomeLarge-v1`: it states that RTU-PPO
outperformed the other learning methods, was close to search with low variance, and that
DRQN was similar to DQN ([Section 9](https://arxiv.org/html/2605.01131v1#S9)). This is
useful transfer provenance but not direct evidence for the stationary target: the task has
hidden reward switching, different biome/wall dynamics, previous-action and previous-reward
inputs, and a 10M-step experimental horizon. The paper used 10% tuning with 10 trials and
30 independent 10M-step evaluations ([experiment details](https://arxiv.org/html/2605.01131v1#S14.SS4)).

The exact official color/FOV9
[RTU-PPO configuration](https://github.com/steventango/continual-foragax-agents/blob/9710f60fa30da5badc451ad7ce3ff296d5070830/experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json)
uses 10M steps, a 2,048-step rollout, four epochs and 32 minibatches, gamma 0.99,
GAE 0.95, clip 0.2, gradient norm 0.5, entropy 0.1, actor Adam `3e-4`, critic scale 10,
and a convolutional encoder with LayerNorm, 512 RTUs, and a 64-unit head. It has a
substantially different state, update, and compute budget from the frozen 499,712-step
stationary transfer (including its 128-RTU configuration). A future stationary comparison
must retune it on fresh development seeds and disclose trainable parameters, recurrent
carry and RTRL-trace state, rollout/replay storage, optimizer updates, and wall-clock use.

The paper identifies a compact next tier for the *switching* task: DQN and PPO with an
exponentially weighted reward trace (selected decay 0.9), DQN+CReLU, and DQN/PPO L2-to-
initialization. Its selected DQN controls are CReLU (`alpha=3e-3`, epsilon 0.1) and L2-init
(`alpha=1e-3`, epsilon 0.1, lambda `1e-5`); PPO L2-init uses actor `1e-3`, critic scale
0.1, entropy 0.01, and lambda `1e-4`
([Tables 6–9](https://arxiv.org/html/2605.01131v1#S14.SS4)). These are roadmap arms for
new target-task development tuning, not rankable support for the stationary protocol.

### Unevaluated streaming RTU-RTRL arm

[Farr et al., “Streaming Reinforcement Learning under Partial Observability with Real-Time
Recurrent Learning,” arXiv:2605.24709v2 (2026-07-07)](https://arxiv.org/html/2605.24709)
introduces RTU-RTRL combinations with streaming QRC and stream AC. It evaluates
MemoryChain, POPGym, and masked MuJoCo—not Forager—and explicitly does not claim
state-of-the-art absolute performance. The method is nevertheless a relevant future family
for a single-stream, constant-memory Forager protocol: it has a diagonal RTU with exact
RTRL for recurrent parameters, but an explicitly one-step approximation for encoder
gradients ([method and limitations](https://arxiv.org/html/2605.24709#S3)).

It must be introduced as a separate implementation and preregistered development/evaluation
cycle, never as a result transfer. Its resource ledger must include both RTU carry and
RTRL sensitivity state, per-step update cost, any eligibility traces, normalization state,
and the absence of replay/rollout storage. This prevents a nominally “streaming” comparison
from concealing a materially different memory or compute budget.
