# Micro-continual benchmark suite — design, transfer validation, coordination

Status: development infrastructure + development calibration evidence
(`development_screening_diagnostic`, nonpromoting — nothing here is
scientific evidence). Module:
`alberta_framework/benchmarks/micro_continual.py` (canonical Gaussian suite,
`gauss-v1`); tests: `tests/test_micro_continual.py`. This file is the
coordination doc for the discovery lane; it supersedes the provisional spec
previously at this path (that spec's contract is preserved in section 7).

Directive being served: *"numbers instead of MNIST — the smallest
environment that scales up"* — a first-principles micro-benchmark suite for
fast iteration over the IPMNIST campaign lanes, validated as a proxy by
reproducing the campaign's measured full-protocol method ordering.

## 1. Design

**Base distribution (all numbers, no data files).** `n_classes` Gaussian
*mixture* classes in `dim` dimensions. Three structural ingredients mimic
what calibration (section 4) proved load-bearing about MNIST:

1. **Heterogeneous per-dimension marginals** — a log-spaced scale spectrum
   spanning `spectrum_decades` decades plus per-dimension offsets (MNIST
   pixel marginals vary wildly). This gives the conditioning axis its
   teeth: raw-input optimization is ill-conditioned; per-feature statistic
   tracking fixes it.
2. **Within-class multimodality** — `n_components` Gaussian components per
   class (MNIST classes are mixtures of writing styles). This makes hidden
   features genuinely expensive to learn, which is what makes protection
   (the utility gate) worth anything.
3. **Sparse, localized component structure** — each component displaces only
   `component_sparsity` dimensions; class means signal on a `class_sparsity`
   fraction (the stroke-locality mimic). Dense random geometry makes
   features diffuse and cheap to re-learn; protection then never pays.

**Regime axes (M1–M4).** Every `regime_length` steps one transform switches,
each isolating one measured difficulty axis of the campaign
(`CONTINUAL_LEARNING_THEORY.md`):

| code | family | transform | axis distilled |
|---|---|---|---|
| M1 | `input_permutation` | fresh coordinate permutation per regime | input shift (IPMNIST) |
| M2 | `label_permutation` | bijective label remap per regime | label shift (L/P-EMNIST) |
| M3 | `scale_shift` | global `x -> c_r * x`, `c_r` log-uniform | conditioning in isolation |
| M4 | `recurrence` | permutations from a small revisiting pool | memory / context re-use |

**Analytic references, by construction.** All four transforms are bijections
acting covariantly on the generative parameters, so the Bayes-optimal
accuracy is *regime-invariant* and known exactly: the Bayes rule is the
closed-form mixture discriminant
`argmax_c logsumexp_k(-0.5 * mahalanobis²(x, mu_ck))`, evaluated to
arbitrary Monte-Carlo precision by `bayes_reference` (binomial SEM
reported), with the exact closed form `Phi(delta/2)` for the two-class
unimodal case cross-checking the machinery (pinned by tests, as is
transform invariance). `chance = 1/n_classes` is the floor reference.

**Metric.** The protocol's online-accuracy-while-learning: every prediction
scored *before* the update that consumes its example; per-regime means; the
headline number is the mean over all regimes. No train/test split to hide
in.

**Method ladder (campaign equations re-used, not re-implemented).** The six
`LADDER_ARMS` import the campaign's registered factories and hyperparameters
so the proxy measures the same mechanisms (cross-module one-step parity
pinned by tests):

| arm | source | campaign row |
|---|---|---|
| `sgd_raw` | in-module (hand-pinned) | mechanism-free floor |
| `adamw` | `upgd_ipmnist` factory, published config | `adamw_control` |
| `upgd_raw` | `upgd_ipmnist` factory, published config | `upgd_w_control` (ICLR-2024 SOTA form) |
| `sgd_norm` | `ipmnist_screening._make_sgd_ema_norm_learner` | `sgd_ema_norm_d099` (conditioned floor) |
| `gated_norm` | `ipmnist_screening._make_upgd_shiftnorm_learner` | **`sigma0_shiftnorm_d099` champion form** (0.86449, n=20) |
| `naive_bayes` | `ipmnist_screening._make_naive_bayes_learner` | `naive_bayes` (V3); variance floor rescaled to the micro spectrum (1e-4) |

## 2. Frozen transfer-validated M1 operating point (`gauss-v1` defaults)

```
family=input_permutation  n_regimes=100  regime_length=5000
dim=256  n_classes=10  n_components=6
spectrum_decades=2.0  mean_separation=0.4
component_scale=1.2  component_sparsity=10  class_sparsity=0.2
noise_scale=1.0  offset_scale=1.0
MLP 256 -> 75 -> 38 -> 10 (protocol init), streams/init paired across arms
```

These are the `MicroStreamConfig()` defaults. Validation artifacts:
`outputs/micro_continual/ladder_m1/` (immutable shards +
`summary_input_permutation.json` + `transfer_input_permutation.json`).

## 3. Transfer-validity evidence (M1 ladder, seeds 0–2, paired)

**Verdict: `transfer_valid = true` — all six primary checks and the
secondary check pass** (`transfer_input_permutation.json`, exit 0).

Ladder results (mean over 3 seeds ± stderr; Bayes ceiling 0.9843 ± 0.0035
across seed geometries, chance 0.10):

| arm | micro mean | per-seed | shape | protocol-200 reference |
|---|---|---|---|---|
| `gated_norm` | **0.6911** ± 0.015 | 0.673 / 0.722 / 0.679 | slow first regime (0.34), rises above `sgd_norm`, flat late | champion 0.86449, top |
| `sgd_norm` | 0.6813 ± 0.017 | 0.661 / 0.715 / 0.668 | fast from scratch (first regime 0.63), flat | 0.8399–0.85 conditioned floor |
| `naive_bayes` | 0.6052 ± 0.014 | 0.587 / 0.634 / 0.595 | instant (first regime 0.61), flat | 0.7851 (V3) |
| `upgd_raw` | 0.2801 ± 0.017 | 0.248 / 0.284 / 0.308 | slow start (first regime 0.15), rises, holds | 0.7791 published SOTA config |
| `sgd_raw` | 0.2740 ± 0.006 | 0.263 / 0.274 / 0.284 | early 0.34 window, decays to 0.24 | (not a campaign row) |
| `adamw` | 0.2282 ± 0.005 | 0.222 / 0.225 / 0.238 | first regime 0.25, monotone decay | ~0.68, monotone decay |

Pre-registered ordering checks against the campaign's measured facts:

| check | micro measurement | campaign fact | pass |
|---|---|---|---|
| conditioning dominates | `sgd_norm − upgd_raw` = **+0.401** (all seeds; 41x the gate delta) | +0.061 vs +0.011 decomposition | yes |
| gate small-positive | `gated_norm − sgd_norm` = **+0.0098** (+0.0119 / +0.0073 / +0.0103, all seeds positive; 41x smaller than conditioning) | gate = +0.011 | yes |
| Adam decays | first-quarter 0.2376 → last-quarter 0.2202; slope −0.00053/regime | 0.7803 → 0.7375; −0.00184/task | yes |
| Adam below UPGD-W | 0.2282 < 0.2801 | ~0.68 < 0.779 | yes |
| naive Bayes placement | 0.2801 < 0.6052 < 0.6813 | 0.7778 < 0.7851 < 0.8399 | yes |
| champion top | `gated_norm` best arm | champion tops the campaign | yes |
| (secondary) Adam fast early | first regime 0.253 > 0.146 | t1: 0.7694 > 0.6928 | yes |

The curve *shapes* also transfer: UPGD-W's slow-start-then-rise-and-hold,
Adam's early-strength-then-monotone-decay, naive Bayes' instant flat line,
the champion form's slow first task followed by per-regime superiority over
the ungated conditioned arm — each matches its full-protocol signature
(`CONTINUAL_LEARNING_THEORY.md` section 1.2).

Caveats (honest limits of the validation): 3 seeds; seed spread is dominated
by geometry variation (each seed draws a new class geometry), so paired
per-seed deltas are the meaningful statistics; the micro dynamic range is
wider than the protocol's (raw arms sit lower relative to the conditioned
arms); the sgd_raw row has no protocol counterpart to check; M2–M4 orderings
are NOT validated (section 6). Nothing here promotes any scientific claim.

## 4. Calibration history — what was load-bearing (honest record)

Seven calibration rounds (`outputs/micro_continual/calibration/`, probes
1–11) iterated the generator per the pre-registration ("if ordering
disagrees, iterate the generator until it agrees"). The failures are the
useful part:

1. **v0 (dim=32, unimodal, dense, T=1000).** Everything compressed toward
   chance; `gated_norm` collapsed to 0.17 while `sgd_norm` reached 0.44;
   published UPGD-W pinned at chance. Dissection probes (shift detector
   disabled; noise disabled) localized the collapse to the **utility
   gate**, and step-level instrumentation showed no sign pathology — the
   gate simply halves the effective step (mean gate ~0.5), which is fatal
   when the regime is far shorter than convergence.
2. **The `||x||²` effect — input dim sets the optimizer operating point.**
   The input-layer step per unit lr scales with the input norm; at dim=32
   the champion's lr 0.01 operates ~25x below its 784-dim protocol point,
   so the gate's halving is pure tax. The gate deficit shrank monotonically
   with dim (−0.067 @128 → −0.031 @256 → −0.006 @320, dense probes).
   Dimensionality is not a free "smallness" knob.
3. **Regime length is Adam's operating regime.** At the published lr 1e-4,
   AdamW cannot express its fast-early/decay-late protocol shape at T ≤
   2500 on these nets; at T=5000 (protocol-equal) the signature appears.
4. **Unimodal Gaussians make protection worthless.** Single-component
   classes re-learn in a few hundred steps; `sgd_norm` saturates within
   every regime and protected-feature transfer has no headroom — the gate
   stayed negative at every dense operating point probed. Adding
   within-class mixtures (`n_components=6`) moved the gate from −0.09
   toward −0.01.
5. **Sparse localized components flipped the gate positive** (−0.003 dense
   → +0.011 at q=10/dim=256/K=6 — numerically the protocol's own +0.011).
   Diffuse dense features are cheap to re-learn; sparse conjunctions are
   specific, expensive, and reusable — protection finally pays. Sparsity
   also demoted model-matched naive Bayes below the conditioned deep arms
   (diagonal-unimodal NB cannot represent the mixture), fixing the V3
   placement, which model-matched unimodal streams had inverted.
6. **Raw-arm decay appears for free** on hard sparse streams (`sgd_raw`
   0.34 → 0.24 across regimes) — the stale-statistics/damage phenomenology
   the campaign measured, absent on easy dense streams.

Generator-design summary: *the ordering of continual-learning methods is not
scale-free.* Matching a benchmark's mechanism ranking requires matching (a)
the optimizer operating point (input norm / dim), (b) the horizon ratios
(regime length vs from-scratch convergence time), and (c) the feature
economics (how expensive and how reusable features are). Those three,
not MNIST pixels, are what the campaign's ordering was measuring.

## 5. Iteration-speed evidence

Wall clocks measured on this box (24 cores, concurrent campaign load ~7–9;
the protocol references are the pinned shards from the same box):

| lane | protocol wall/seed | micro wall/seed (R=100, T=5000) | speedup |
|---|---|---|---|
| champion form (`sigma0_shiftnorm_d099` 200-task confirm) | 346.9 s | `gated_norm` 16.1 s | **21x** |
| conditioned SGD class (`sgd_ema_norm_d099` 200-task confirm) | 339.5 s | `sgd_norm` 3.2 s | **106x** |
| published UPGD-W (`upgd_w_control` 200-task confirm) | 17,006 s | `upgd_raw` 181.9 s | **93x** |
| naive Bayes (60-task screen shard) | 14.3 s | 2.0 s | 7x |
| champion form vs 60-task screening proxy | 158.4 s | 16.1 s | 10x |

For the discovery lane's inner loop the relevant numbers are the
champion-class and SGD-class rows: a candidate evaluation that costs
~347 s/seed on the full lane (or ~158 s on the 60-task screen) costs
3–16 s here — and fitness runs may further drop `n_regimes` (e.g. 40 →
~6 s champion-class), since the validated structure lives in the geometry,
not the horizon. The stream itself is ~2x fewer steps (500k vs 1M) with a
~9x smaller network; the noise-free champion-class arms gain the most
because the suite keeps the protocol's T=5000 (required for Adam's
operating regime) but shrinks everything else. Full-suite context: one
complete 6-arm × 3-seed validated ladder = ~11 minutes, of which 9 are the
published UPGD-W noise arm.

## 6. Usage

```bash
# one shard (idempotent: existing shards are validated and kept)
.venv/bin/python -m alberta_framework.benchmarks.micro_continual run \
  --family input_permutation --arm gated_norm --seed 0 --out outputs/micro_continual/dev

# full ladder + summary + (M1, full ladder) transfer validation
# exit 0 = ordering reproduced, 2 = not (receipt preserved either way)
.venv/bin/python -m alberta_framework.benchmarks.micro_continual ladder \
  --family input_permutation --seeds 0 1 2 --out <NEW dir>
```

```python
# programmatic (the discovery-lane fitness path)
from alberta_framework.benchmarks.micro_continual import (
    MicroStreamConfig, bayes_reference, run_micro_arm)
result = run_micro_arm(MicroStreamConfig(), "gated_norm", seed=0)
ceiling = bayes_reference(MicroStreamConfig(), seed=0)
```

Guidance for the discovery lane:

- **Fitness runs**: use the frozen M1 geometry; drop `n_regimes` (e.g. 40)
  for screening-speed iteration. Score against `bayes_reference` (known
  ceiling) and the `gated_norm` / `sgd_norm` anchors; micro wins promote
  nothing — route candidates to `ipmnist_screening` arms and the real
  60-task protocol per the screening runbook.
- **Axis dissection**: M2 isolates label shift, M3 isolates conditioning,
  M4 is the recurrence/memory axis (context re-use; the V4 direction).
  Descriptive reference runs (R=40, seed 0, frozen geometry;
  `outputs/micro_continual/families_r40/`):

  | arm | M2 label | M3 scale | M4 recurrence |
  |---|---|---|---|
  | `gated_norm` | 0.788 | **0.866** | 0.715 |
  | `sgd_norm` | **0.824** | 0.849 | **0.717** |
  | `naive_bayes` | 0.795 | 0.740 | 0.621 |
  | `sgd_raw` | 0.641 | 0.710 | 0.569 |
  | `adamw` | 0.243 | **0.785** | 0.464 |
  | `upgd_raw` | 0.302 | 0.501 | 0.326 |

  Axis face-validity worth noting (descriptive, 1 seed): on M3 — the pure
  conditioning axis — AdamW jumps from last place to third (its second
  moment IS input conditioning, the campaign's `adamw_cbp_ema_norm`
  conclusion) and the champion's *shift-triggered* normalizer beats the
  plain EMA normalizer exactly where shift-triggered re-conditioning should
  matter (+0.017); on M2, inputs are stationary so `sgd_raw` recovers and
  the conditioned-SGD floor leads. **Transfer validation is defined on M1
  only** — M2–M4 orderings were not calibrated against protocol facts and
  must not be quoted as validated.
- **Shards are immutable; summaries are derived and replaceable.** New
  configurations go to NEW output directories.

## 7. Reconciliation with the provisional digits suite

`micro_continual.py` also carries the rule-discovery track's provisional
sklearn-digits suite (`MICRO_SUITE`, `build_micro_stream`,
`provisional-v1`), kept verbatim because `rule_discovery` and its tests
import it. Its contract (from the superseded provisional spec) remains in
force for that harness until migration: search fitness reads M1+M2+M3 only
(seeds 0–1); digits-M4 (`permutation_affine`) and M1p (7x7 crop) are
selection-validation holdouts (seeds 101–103); micro results promote
nothing. Reconciliation plan: the discovery harness migrates its fitness
tasks to the canonical Gaussian suite (analytic Bayes ceilings, validated
M1 transfer, campaign-parity arms); the digits suite then becomes a holdout
family or is retired. Until then the two suites coexist in one module; the
digits suite has no transfer validation and no analytic references — treat
its scores as relative only.
