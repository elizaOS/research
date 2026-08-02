# Continual learning on streaming supervision: a mechanistic theory from the IPMNIST campaign

> **2026-08-02 addendum — the completed dissection cascade.**  The
> follow-up ablations fully decomposed the top result (`upgd_ema_norm`,
> 0.85359 at 200 tasks) into mechanism contributions (60-task proxy):
>
> | component removed | arm | mean | contribution |
> |---|---|---|---|
> | (full method) | `upgd_ema_norm` | 0.8529 | — |
> | − perturbation | `upgd_ema_norm_sigma0` | 0.8520 | noise ≈ **+0.001** (not load-bearing) |
> | − utility gate | `sgd_ema_norm` | 0.8406 | gate ≈ **+0.012** (real, modest) |
> | − normalization (raw UPGD-W) | `upgd_w_control` | 0.7778 | conditioning ≈ **+0.062** (dominant) |
>
> Three conclusions.  (1) **Input conditioning dominates**: bare
> normalize+SGD+decay (0.8406) beats the published SOTA method run as
> published (0.7791) by +0.06 — much of what the plasticity-method
> literature fights on this benchmark dissolves under the Alberta Plan's
> Step-1 normalization tenet, which the published setup omitted.
> (2) The **utility gate survives dissection** as a genuine, orthogonal
> +0.012 protection effect — the UPGD idea is real, but an order of
> magnitude smaller than its enabling condition.  (3) The
> **perturbation contributes nothing** here (no-recurrence protocol) —
> the best form, *normalized utility-gated SGD (σ=0)*, delivers the
> full 0.85-class result at ~1/7th UPGD-W's compute.  The composition
> result (`adamw_cbp_ema_norm` 0.7995 ≈ `adamw_cbp`) closes the loop:
> Adam's second-moment denominator IS input conditioning, so
> normalization is redundant there — and normalized gated-SGD beats
> conditioned Adam+CBP outright.  Full-horizon confirmations of both
> ablation arms run under `outputs/ipmnist_screening/confirm_full/`.
> The hyperparameter star found published UPGD-W values locally optimal
> under normalization (wd0005/lr shifts all hurt), so the wd0005 gain
> does not compose — tuning wins and conditioning wins are alternatives,
> not additive.
>
> **Full-horizon confirmation (200 tasks, 3 seeds each):** the
> decomposition is stable across the 3.3× horizon extension —
> `upgd_ema_norm_sigma0` 0.85051 and `sgd_ema_norm` 0.83991 (seed
> spread ≤ 0.0008 on both), giving conditioning **+0.061**, gate
> **+0.011**, noise **+0.003** against the full method's 0.85359 and
> the published-config baseline's 0.7791.  Every rung of the cascade —
> including bare normalize+SGD+decay — beats the published SOTA method
> at full protocol length.

Status: development analysis (never promotable evidence). Every number below
is from our own protocol-exact runs — the ICLR-2024 online Input-permuted
MNIST protocol (`alberta_framework/benchmarks/upgd_ipmnist.py`), its 60-task
exact-prefix screening proxy
(`alberta_framework/benchmarks/ipmnist_screening.py`,
`outputs/ipmnist_screening/shards/`), the 200-task confirmations
(`outputs/ipmnist_screening/confirm_full/`), and the label-permuted EMNIST
400-task lane (`outputs/upgd_label_emnist/results.v1.json`). Screening arms
use 3 paired seeds (identical schedules and init per seed across arms);
per-seed stderr on the screen is ~0.0004, so paired differences beyond
~0.002 are far outside noise. "Online accuracy" is the protocol's
average-per-task accuracy of single-example predictions made before each
update.

This document does three things:

- **(a)** reads the completed per-task curves and says *why* each mechanism
  worked or failed;
- **(b)** states the three-orthogonal-failure-modes theory precisely and maps
  every method we measured onto it, with coupling costs;
- **(c)** derives the wave-4 novel arms (implemented in
  `ipmnist_screening.py`, queued in
  `outputs/ipmnist_screening/jobs4.txt`), each with mechanism equations, a
  falsifiable prediction, a novelty statement, and expected failure modes.

---

## 1. The measured base

### 1.1 Headline numbers

| lane | arm | result |
|---|---|---|
| IPMNIST 200-task, 10 seeds | UPGD-W (published config) | **0.7791** (clean reproduction of the ICLR-2024 ~0.78 figure) |
| IPMNIST 200-task, 10 seeds (confirm_full) | `upgd_w_wd0005` | **0.78431** — beats the published-config SOTA (per-seed range 0.78356-0.78497; first 3 seeds paired vs control all improve) |
| IPMNIST 200-task, 10 seeds (confirm_full) | `adamw_cbp` | **0.79876** (per-seed range 0.79829-0.79916) — paired seeds 0-2: 0.79892 vs `upgd_w_control` 0.77883, **+0.0201** |
| IPMNIST 60-task screen, 3 paired seeds | `adamw_cbp` | 0.7965, paired **+0.0188** vs `upgd_w_control` 0.7778 |
| Label-permuted EMNIST, 400 tasks | UPGD-W | online accuracy **rises 0.40 → 0.737** across recurrences; AdamW **collapses to ~0.20** |

60-task screen, paired mean difference vs `upgd_w_control` (3 seeds):

| arm | Δ vs control | one-line verdict |
|---|---|---|
| `adamw_cbp` | **+0.0188** | new leader; different base optimizer entirely |
| `upgd_w_wd0005` | +0.0055 | half the published decay is simply better |
| `upgd_l2init` | +0.0014 | decay-toward-init ≈ mild improvement |
| `upgd_idbd` | +0.0001 | per-weight step-sizes *applied as rates*: null |
| `upgd_w_udecay0999`/`099999` | ~-0.001 | utility-EMA timescale is well-tuned already |
| `upgd_cbp` | -0.0012 | CBP redundant under UPGD's own perturbation |
| `upgd_w_wclip_k2` | -0.0059 | mild norm clamp: mild cost |
| `upgd_w_wd002` | -0.0178 | 2x decay forgets too much |
| `adamw_control` | -0.0222 | fast early, decays late (§1.2) |
| `upgd_w_wclip_k1` | -0.0381 | init-bound clip: capacity ceiling (§1.2) |
| `upgd_autostep` | -0.0915 | batch-1 meta-gradient variance: collapse (§1.2) |

### 1.2 What the per-task curves say (the shapes, not just the means)

Seed-mean per-task online accuracy at selected tasks (60-task shards):

| arm | t1 | t2 | t5 | t10 | t20 | t30 | t40 | t50 | t60 | late slope/task |
|---|---|---|---|---|---|---|---|---|---|---|
| `adamw_control` | **0.7694** | **0.7899** | 0.7779 | 0.7738 | 0.7673 | 0.7512 | 0.7521 | 0.7515 | 0.7290 | **-0.00184** |
| `adamw_cbp` | 0.7689 | 0.7922 | 0.7835 | 0.7903 | 0.7982 | 0.7950 | 0.7976 | 0.8025 | **0.8020** | -0.00012 |
| `upgd_w_control` | 0.6928 | 0.7701 | 0.7778 | 0.7757 | 0.7771 | 0.7816 | 0.7759 | 0.7829 | 0.7849 | -0.00023 |
| `upgd_w_wd0005` | 0.7015 | 0.7793 | 0.7837 | 0.7848 | 0.7863 | 0.7817 | 0.7818 | 0.7913 | 0.7904 | ~0 |
| `upgd_autostep` | 0.7267 | 0.7699 | 0.7405 | 0.7423 | 0.7261 | 0.6873 | 0.6665 | 0.6396 | **0.5895** | **-0.00307** |
| `upgd_w_wclip_k1` | 0.6661 | 0.7365 | 0.7377 | 0.7407 | 0.7365 | 0.7411 | 0.7421 | 0.7430 | 0.7405 | -0.00033 |
| `upgd_w_udecay0999` | 0.7116 | 0.7735 | 0.7755 | 0.7781 | 0.7746 | 0.7775 | 0.7753 | 0.7824 | 0.7824 | ~0 |
| `upgd_w_udecay099999` | 0.6603 | 0.7593 | 0.7732 | 0.7779 | 0.7792 | 0.7792 | 0.7763 | 0.7821 | 0.7853 | ~0 |

Five distinct failure/success *shapes*:

1. **AdamW: early dominance, monotone late decay.** AdamW wins task 1 by
   +0.077 over UPGD-W (0.7694 vs 0.6928) and the t1-5 window by +0.022
   (0.7803 vs 0.7583) — per-weight curvature-normalized steps adapt within a
   task far faster than any fixed-scalar-rate method. It then decays
   monotonically (window means 0.7803 → 0.7725 → 0.7548 → 0.7375; late slope
   -0.00184/task) and its plasticity metric falls 0.367 → 0.317. The
   crossover with UPGD-W happens around task ~8-10. This is *accumulating
   damage*, not a capacity ceiling: the curve keeps falling at 60 tasks and
   (published protocol) lands near 0.68 at 200.
2. **CBP repairs exactly the decay, and nothing else.** `adamw_cbp` tracks
   `adamw_control` bit-for-nearly-bit early (t1 0.7689 vs 0.7694 — CBP has
   not yet fired), then diverges upward exactly where the control decays:
   slope -0.00184 → -0.00012, plasticity *rises* 0.373 → 0.428, late window
   0.7999. The 200-task confirmation (10 seeds) shows the same repaired
   shape: 0.8029 at t100, 0.7993 over t181-200. Selective recycling of dead
   units removes the decay term while leaving Adam's within-task speed
   untouched. **Residual signal (10-seed correction):** the early 3-seed
   read of a -0.009 t100→t200 drift was inflated by the noisy single-task
   endpoint; at 10 seeds the robust window-level drift is peak window
   0.8011 (t69-88) → 0.7993 (t181-200), ~**-0.002**. Small, but still
   monotone-negative after Mode-2 repair — this residual (plus the EMNIST
   recurrence result, which carries the strong Mode-3 evidence) is the
   wedge for wave 4 (§3.1).
3. **UPGD-W: slow start, flat forever.** The gate + decay + perturbation
   package costs within-task speed (t1 0.6928; it never wins any early
   window) but the curve then *rises* to a plateau (~0.785 at 60 tasks,
   0.7833 at t100) and holds with slope -0.0002. The utility-decay star
   shows the protection timescale trade directly: a faster utility EMA
   (0.999) starts better (t1 0.7116) and ends worse (late 0.7772 — noisy
   protection); a slower one (0.99999) starts much worse (t1 0.6603 —
   protection and its bias-correction develop too slowly) and ends at
   control level. The published 0.9999 sits at the optimum of that trade —
   the gate is doing real protection work with a tuned memory horizon.
4. **Autostep: systematic step-size collapse, not seed divergence.** All
   three seeds decay in lockstep (t60: 0.563/0.616/0.590; slope
   -0.00307/task) and its plasticity is the lowest measured (0.22 → 0.19).
   With batch-1 meta-gradients, Autostep's per-weight rate adaptation keeps
   absorbing single-example variance into permanent step-size changes; the
   effective rates drift away from the working regime and the network
   gradually stops tracking. Applying per-weight rates online is
   *risk-dominated* at batch size 1 (IDBD at meta 1e-3 stayed neutral only
   because its meta-updates barely move).
5. **Weight clipping at the init bound: a ceiling, not a decay.**
   `wclip_k1` is flat from task 2 onward (~0.740 everywhere, slope
   -0.0003) but permanently ~4 points below control: clamping weights to
   the PyTorch-init interval preserves update capability but removes the
   representational capacity the task needs (ReLU nets need weights beyond
   init scale to fit 10-way MNIST at width 300/150). `k2` costs only
   -0.006 — the bound, not the mechanism, is the problem.

Two more curve facts used below:

- **Label recurrence flips the ranking.** On label-permuted EMNIST
  (400 tasks, 47 classes, recurring permutations) UPGD-W's online accuracy
  *rises* across recurrences 0.40 → 0.737 — protected weights act as
  consolidated memory that later tasks reuse — while AdamW collapses to
  0.20. Protection is worth ~3.7x accuracy under recurrence, and ~+0.02
  under weak recurrence (IPMNIST permutes inputs but keeps the label
  structure and hidden-layer features partially reusable).
- **`upgd_cbp` is null (-0.0012).** Adding an explicit regenerator to
  UPGD-W finds nothing to fix. Either UPGD's perturbation already
  regenerates (then removing it should re-expose decay), or UPGD-SGD
  dynamics never create dead capacity at these widths (then the
  perturbation is dead weight at 85-90% of step cost). §3.3 dissects this.

---

## 2. The three-orthogonal-failure-modes theory

### 2.1 Statement

A single online learner on a nonstationary stream degrades through three
mechanistically independent channels. Each has a distinct curve signature,
a distinct granularity, and a distinct fix; the fixes compose cleanly if
and only if they are driven by separate signals.

- **Mode 1 — slow within-task adaptation** (per-weight, *rate* channel).
  A fixed scalar step size is mismatched to per-parameter gradient scale
  and curvature, so each new task is tracked slowly. Signature: depressed
  task-1/early-window accuracy, no late decay. Measured: UPGD-W loses
  -0.077 on task 1 vs AdamW; every fixed-rate arm shows the same slow
  start.
- **Mode 2 — dead capacity / loss of plasticity** (per-unit, *capacity*
  channel). Units saturate or die (ReLU zero regions, magnitude growth),
  gradient flow through them vanishes, and the effective network shrinks
  monotonically. Signature: monotone late decay of per-task accuracy with
  declining plasticity metric. Measured: `adamw_control` slope -0.00184
  with plasticity 0.367 → 0.317; cured by CBP (slope -0.00012, plasticity
  0.428) — a per-unit intervention with no rate or protection content.
- **Mode 3 — catastrophic overwrite** (per-weight, *memory* channel).
  Weights that encode still-useful structure are updated by the new task's
  gradients through shared, always-on parameters. Signature: under strong
  recurrence, failure to improve across task revisits (AdamW flat at 0.20
  on EMNIST-400 while UPGD climbs to 0.737); under weak recurrence, a slow
  late drift even after Mode 2 is repaired (`adamw_cbp`'s ~-0.002
  peak-to-late window drift at 10 seeds — weak on IPMNIST; the recurrence
  lane is the load-bearing Mode-3 measurement).

**Orthogonality claim (falsifiable):** the three modes have disjoint
mechanisms and granularities, so a method addressing all three with
*uncoupled* signals should strictly dominate any method that addresses a
subset or couples the fixes. Nothing published combines all three cleanly.

### 2.2 Method-to-mode map with coupling costs

| method | Mode 1 (rate) | Mode 2 (capacity) | Mode 3 (memory) | coupling cost |
|---|---|---|---|---|
| SGD, fixed lr | – | – | – | baseline |
| Adam(W) | **yes** (per-weight 1/sqrt(v) scaling) | no — normalized steps keep moving weights at scale-invariant speed, accelerating saturation | no — and worse: scale-invariance overwrites small-but-useful weights as fast as large ones | none (does one thing) |
| CBP (Dohare et al., Nature 2024) | – | **yes** (per-unit utility + recycle) | – | none — touches only units it declares dead; composes with anything (our `adamw_cbp`) |
| UPGD-W (Elsayed & Mahmood, ICLR 2024) | no (fixed scalar lr; slow task-1) | yes, via *gated perturbation* (noise sized by 1-gate regrows low-utility weights) | **yes**, via utility gate (protect high `-w·g` weights) | **high**: one signal (utility EMA) and one knob (the gate) drive both protection and regeneration; the perturbation costs 85-90% of step time; the coupling forbids using an adaptive-rate base (noise would be rescaled by 1/sqrt(v)) |
| L2 / L2-Init / weight decay | – | indirect (contraction limits saturation) | crude, utility-blind drift bound — helps at the right magnitude (wd0005 +0.0055), destroys memory beyond it (wd002 -0.0178) | none, but untargeted |
| Weight clipping (Elsayed et al., RLC 2024) | – | yes (bounded weights can't saturate) | yes (bounded drift) | **capacity clamp**: the same bound that protects also caps expressivity (k1 -0.038 flat-low) |
| IDBD / Autostep (applied as rates) | intended | – | – | **variance coupling**: batch-1 meta-gradients convert noise into permanent rate drift (autostep -0.0915 collapse) |
| Streaming EWC / SI / MAS (published baselines) | – | – | yes (quadratic anchor) | anchors are irreversible → blocks relearning; published IPMNIST places them at 0.70-0.72, below UPGD's 0.78 |

### 2.3 Design laws (established by the campaign, used in §3)

1. **Exclusive gating = memory; always-on shared parameters = forgetting
   channels.** Any weight every task can write to will eventually be
   overwritten; memory requires per-weight write-attenuation.
2. **Protection must be reversible.** The gate must *release* weights whose
   utility signal decays (UPGD does; EWC-style anchors do not; that is the
   0.78-vs-0.72 gap).
3. **Per-weight step-size statistics carry relevance information even when
   they are unsafe to apply as rates.** IDBD's alphas encode "this weight
   has been learning consistently"; applying them (autostep) is
   risk-dominated at batch 1, but *reading* them costs nothing (§3.4).
4. **Regeneration and protection compose only through separate signals at
   separate granularity.** Per-unit death (CBP) and per-weight utility
   (UPGD gate) are different objects; welding them into one mechanism
   (UPGD's gated noise) buys elegance at the price of a mandatory noise
   draw and a mandatory fixed-rate base.
5. **The perturbation is regeneration, priced at 85-90% of the UPGD step.**
   If a dedicated regenerator is present (or none is needed), the noise
   should be removable at near-zero accuracy cost (§3.3 tests this).

The wave-4 arms are the four falsifiable consequences of these laws, plus
one fairness control.

---

## 3. Wave-4 novel mechanisms (implemented; queued in `jobs4.txt`)

All arms are registered in
`alberta_framework/benchmarks/ipmnist_screening.py` with parity/reduction
unit tests in `tests/test_ipmnist_screening.py`; 60-task screen, seeds 0-2,
paired against `upgd_w_control` (and, for the adamw family, re-merged
against `adamw_cbp`). Runs are **queued, not launched** (box contended).

### 3.1 `guarded_cbp_adam` — complete all three modes with zero coupling

**Mechanism.** Base: `adamw_cbp` (Modes 1+2). Added: UPGD's protection
*only*, applied to Adam's delta. Per step, with UPGD's bias-corrected
utility EMA and network-global sigmoid squashing:

```
U_i   <- beta_u * U_i + (1 - beta_u) * (-g_i * w_i)          beta_u = 0.9999
gate_i = sigmoid( (U_i / (1 - beta_u^t)) / max_j U_j )
step_i = Adam_delta_i(m, v, count; g_i)                       (moments see raw g)
w_i   <- w_i - (1 - guard_scale * gate_i) * step_i            guard_scale = 1
```

CBP recycling is unchanged (per-unit `|a·da|` utility, rate 1e-4, maturity
100) and additionally zeroes the recycled unit's guard-utility slices, so
fresh units restart at the neutral gate 0.5. **No perturbation anywhere** —
regeneration is CBP's job (law 4), so the noise channel (85-90% of UPGD's
cost) is simply absent. Reduction: `guard_scale = 0` is bit-exact
`adamw_cbp` (pinned).

**Why novel.** UPGD fuses protection and regeneration in one gated-noise
step on a fixed-rate base; CBP and ReDo-style unit resets regenerate
without any protection; EWC/SI/MAS protect via irreversible anchors;
masking methods (PackNet, WSN, supermasks) need task boundaries. A
*reversible, boundary-free, first-order per-weight write-gate on an
adaptive-rate base with an independent per-unit regenerator* is exactly the
zero-coupling composition of all three modes, and we find no published
instance (closest in spirit: UPGD's own appendix ablations, which never
decouple the gate from the noise or from the SGD base).

**Falsifiable prediction.** Screen: `guarded_cbp_adam` > `adamw_cbp` by
> +0.003 paired (i.e. > 0.7995 vs 0.7965), with the gain concentrated in
the late window; on any 200-task confirmation, the peak-to-late window
drift (10-seed baseline: 0.8011 t69-88 → 0.7993 t181-200, ~-0.002) closes
to ~0. Note the drift headroom is only ~+0.002 at 10 seeds — the screen
paired diff is the primary test, the drift shape is secondary
confirmation. **Refutation:** paired diff ≤ 0 vs `adamw_cbp`
refutes the theory's central claim — either the residual late drift is not
Mode-3 overwrite, or protection cannot be factored out of UPGD and applied
to curvature-normalized deltas (i.e. the modes are *not* independent of the
base optimizer, breaking §2.1's orthogonality).

**Expected failure modes.** (i) The 0.5 neutral gate halves fresh recycled
units' learning rate — an adverse protection/regeneration interaction
(diagnosis: `guarded_cbp_adam` late window good but early-window worse than
`adamw_cbp`); a follow-up would initialize recycled guard utility at the
network minimum instead of 0. (ii) Global-max gate normalization inherits
UPGD's known quirk (a negative global max flips the division); rare in
practice. (iii) Any protection taxes Mode 1 — expect a small task-1 dip vs
`adamw_cbp`.

### 3.2 `adamw_cbp_noreset` — is optimizer-state freshness at recycle load-bearing?

**Correction to the brief.** The proposed "`adamw_cbp_vreset`" (reset m and
v for recycled units) is *already the leader's behavior*: our `adamw_cbp`
zeroes the recycled unit's m/v/count slices (per-element counts restart
Adam's bias correction), per `cbp_maybe_replace_layer` and the module
docstring. The mechanism is therefore dissected in the informative
direction: an **ablation that keeps stale moments** across replacement.
Reduction: `cbp_replacement_rate = 0` reduces both arms to `adamw_control`
(pinned).

**Mechanism.** Identical to `adamw_cbp` except `_cbp_update` receives no
optimizer arrays: a recycled unit's fresh weights inherit the dead unit's
m, v, and count. For a truly dormant unit, incoming-gradient v has decayed
toward 0 while count stayed large, so the first post-recycle step is
`lr * g / (sqrt(v_hat) + eps)` with `v_hat ≈ 0.01 g²` — about **10x the
properly bias-corrected step** (overshoot). For a merely-low-utility (still
active) unit, stale v is *large* and the fresh weights inherit tiny steps
(under-maturation). Either way freshness should win; the direction of the
failure identifies which recycle population dominates.

**Falsifiable prediction.** `adamw_cbp_noreset` < `adamw_cbp` by ≥ 0.002
paired, with the deficit growing with task index (recycles accumulate:
rate 1e-4 x 450 units ⇒ one replacement per ~22 steps, ~13k over 60
tasks). **Refutation:** a tie refutes moment-freshness as a load-bearing
part of the leader (the recycling *architecture* — fresh weights, zeroed
outgoing rows — would then carry the whole +0.0188), which would also
license cheaper CBP implementations that skip optimizer surgery.

**Related work.** Dohare et al. reset optimizer state on reinit; ReDo
(Sokar et al., ICML 2023) resets dormant units' optimizer state in deep
RL; Asadi et al. (2023) study full-optimizer resets. The *per-unit
ablation inside a supervised continual stream* (reset vs carry, everything
else paired) is not published as far as we know.

### 3.3 `upgd_w_sigma0` — is UPGD's perturbation load-bearing without recurrence of dead capacity?

**Mechanism.** Lean UPGD-W with `sigma = 0`, i.e. pure utility-gated SGD
with decoupled decay:

```
w_i <- w_i * (1 - lr * wd) - lr * g_i * (1 - gate_i)
```

The per-step 282,160-element normal draw — measured at 85-90% of UPGD-W
single-core step cost — is skipped entirely (bit-exact vs the control
factory run at `noise_std = 0`; pinned).

**Why this dissection matters.** §1.2's `upgd_cbp` null (-0.0012) is
ambiguous: either the perturbation already does all needed regeneration
(CBP redundant), or gated-SGD-plus-decay never kills units at these widths
(both regenerators idle). `sigma0` separates the hypotheses: remove the
perturbation and see whether Mode-2 decay appears. The paper's own
framing (perturbation as the anti-plasticity ingredient, S&P heritage)
predicts decay; our theory (laws 4-5: on IPMNIST at width 300/150 with
decoupled decay, dead capacity barely accumulates, so the noise is priced
regeneration with nothing to regenerate) predicts a near-tie.

**Falsifiable prediction.** |paired diff vs `upgd_w_control`| < 0.003 and
late slope > -0.001/task. **Refutation:** a late-window decay shape
(slope < -0.001 and paired diff < -0.005) refutes our reading and
establishes the perturbation as the live regeneration channel on this
protocol (and retro-explains the `upgd_cbp` null as noise pre-empting
CBP). **Payoff if it ties:** a leader-class UPGD variant at ~1/7 the step
cost (the noise draw dominates), and the mechanism story "gate + decay is
the whole UPGD effect on IPMNIST" — which the paper's presentation
obscures. (Dead-unit counts would decide this directly, but shards store
no parameters; if the tie is ambiguous, a follow-up instrumented run
should log per-layer dead-ReLU fractions.)

**Related work.** UPGD's ablations never run sigma=0 at the published
operating point on this protocol; Shrink-and-Perturb is perturbation
*without* gating (the converse dissection); Dohare et al. show plain SGD+
decay still loses plasticity at *much* longer horizons — a tie here would
be horizon- and width-qualified, and is claimed only for this protocol.

**Expected failure modes.** If wrong, wrongness should appear as
late-onset decay (Mode-2 signature); a *mid*-run dip instead would suggest
the noise also served as beneficial exploration (a fourth role our theory
does not credit).

### 3.4 `upgd_alpha_utility` — is the protection *signal's identity* what matters?

**Mechanism.** UPGD-W with the identical gate geometry but a different
protection signal: per-weight step-size relevance, maintained passively
(law 3). An IDBD pair (Meyer error-free form, exactly the equations of our
`upgd_idbd` arm) runs on the raw gradient but is **never applied as a
rate**:

```
log_alpha_i <- clip(log_alpha_i + meta * g_i * h_i, [-10, 0])     meta = 1e-2
h_i         <- h_i * max(0, 1 - alpha_i * g_i^2) + alpha_i * g_i
s_i     = log_alpha_i - ln(alpha_init)                (drift from init)
gate_i  = sigmoid( s_i / max_j |s_j| )                (0.5 when all drifts are 0)
w_i    <- w_i * (1 - lr * wd) - lr * (g_i + xi_i) * (1 - gate_i)
```

The gate reads a scale-free (shift-of-init and positive-rescaling
invariant) squashing of the drift — the rank-like content of the alpha
statistic, matching UPGD's own global-max normalization in role. The
applied step keeps the published fixed lr, decay, and perturbation, so the
*only* difference from `upgd_w_control` is which statistic the gate reads.
Reduction: `meta = 0` is bit-exact the closed-form half-gated step
(pinned); the mechanism test pins that a persistent-sign gradient earns
more protection than a sign-alternating one.

**Theory.** UPGD's `-w·g` utility protects *current contributors* — weights
whose removal would hurt the current loss. Alpha-drift protects
*consistent learners* — weights whose gradients have agreed with their own
history. After an input permutation, the input layer's gradients
decorrelate immediately: alternating meta-gradients drive `log_alpha` down
(the same dynamics that make applied-IDBD shrink rates), so stale
input-layer weights shed protection *faster* than a 0.9999-EMA of `-w·g`
can decay, while genuinely reused hidden-layer weights keep it. Predicted
consequence: faster post-switch recovery.

**Falsifiable prediction.** Early-task and post-switch accuracy above
control (t1-5 window > 0.7583) with overall paired diff ≥ +0.002.
**Refutation:** an exact tie (|diff| < 0.001 and no early-window gain)
means protection-signal identity is irrelevant on this protocol — only the
gate geometry matters — deleting law 3's "alpha carries relevance" as an
actionable claim. A clear loss (< -0.005) means `-w·g`'s
current-contribution weighting is load-bearing and consistency is the
wrong thing to protect.

**Related work.** IDBD/TIDBD step-sizes have been *read* as feature
relevance (Sutton 1992; Kearney et al.'s TIDBD work interprets learned
step-sizes as relevance/attention; Mahmood's representation-search work
uses them for feature pruning). Using a passive step-size statistic as the
protection gate of a perturbation-based continual learner — while
explicitly *not* applying it as a rate, dodging the autostep failure we
measured — is, to our knowledge, untried.

**Expected failure modes.** (i) Gate range is compressed to
[sigmoid(-1), sigmoid(1)] ≈ [0.27, 0.73] — max protection is weaker than
UPGD's; could cap the benefit. (ii) meta=1e-2 at batch 1 may develop drift
too slowly for 5,000-step tasks (signal arrives after the switch has
already cost accuracy). (iii) Bias weights with steady gradients may hoard
protection.

### 3.5 `adamw_cbp_{r3e5,r3e4,m50,m200}` — fairness star on the untuned leader

Not a mechanism — a control for a confound. UPGD-W's published config is
the product of the authors' sweep, and we gave it a further neighborhood
star (which found wd0005); `adamw_cbp` won with *zero* tuning at CBP
defaults (rate 1e-4, maturity 100, utility decay 0.99). Axis-aligned star:
replacement rate {3e-5, 3e-4}, maturity {50, 200}, one axis at a time.
**Prediction:** modest headroom (0 to +0.004); more recycling (r3e4) helps
if Mode-2 repair is still rate-limited, hurts if churn destroys maturing
units. If all four land within ±0.002 of the leader, the +0.0188 margin is
hyperparameter-robust — which strengthens, not weakens, the confirmation
case. **Refutation content:** if r3e5 (3x less recycling) ties the leader,
the required recycling volume is tiny and Mode-2 damage on IPMNIST is
concentrated in few units; if m50 wins, maturity 100 is delaying repairs.

---

## 4. Outcome matrix (pre-registered readings)

| observation (60-task screen, paired) | reading |
|---|---|
| `guarded_cbp_adam` > `adamw_cbp` + 0.003 | three-mode theory confirmed end-to-end; promote to 200-task confirmation immediately |
| `guarded_cbp_adam` ≤ `adamw_cbp` | orthogonality broken at the protection/base interface — check early-vs-late window split before abandoning (early-only deficit ⇒ fix the fresh-unit 0.5 gate, retry) |
| `adamw_cbp_noreset` ≥ `adamw_cbp` | moment freshness not load-bearing; leader's edge is architectural recycling alone |
| `upgd_w_sigma0` ties control | perturbation is redundant on IPMNIST ⇒ gate+decay is the mechanism; adopt sigma0 as the cheap UPGD base everywhere |
| `upgd_w_sigma0` decays late | perturbation is the live regenerator; upgd_cbp null was pre-emption; UPGD's coupling is earning its cost |
| `upgd_alpha_utility` > control with early-window gain | protection-signal identity matters; alpha statistics are a usable relevance channel (law 3 upgraded) |
| `upgd_alpha_utility` ties control exactly | only gate geometry matters; drop signal-engineering, keep `-w·g` |
| all `adamw_cbp_*` star within ±0.002 | leader margin is hyperparameter-robust |

Anything crossing paired +0.005 over `upgd_w_control` is a full-protocol
confirmation candidate per the screening runbook; screening results alone
claim nothing.
