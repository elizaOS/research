# Ceiling analysis: is 0.95 reachable on the ICLR-2024 IPMNIST protocol?

**Question.** The campaign champion `sigma0_ndecay099` scores **0.86245** average
online accuracy (200 tasks x 5000 steps, one example per step, 300x150 ReLU MLP,
seeds 0-2). What is the realistic ceiling of this metric, and what would 0.95
require?

**Verdict (short).** 0.95 is **not architecture-impossible but is outside every
mechanism class currently in the arsenal**. The champion *family* (EMA-input-norm
+ utility-gated constant-step SGD + decay) is hard-capped at **~0.933** — measured
as its own asymptote with the non-stationarity switched off. The architecture and
the one-example online regime cap out at **~0.974** (AdamW on a stationary
stream). The realistic near-term ceiling for protocol-pure continual methods is
**~0.90 (transient improvements) to ~0.93 (transfer breakthrough)**. Every point
beyond 0.93 requires *simultaneously* Adam-class within-task convergence sustained
continually and near-instant cross-permutation transfer — neither exists today.

All numbers here are development-grade diagnostics
(`development_screening_diagnostic`, seeds 0-2, nonpromoting). Raw artifacts:
`outputs/ipmnist_screening/ceiling/` (runner + analyzer scripts included there;
per-step accuracy recorded as uint8 `.npy`). The full-protocol reruns reproduce
the pinned `confirm_full/` per-task curves to max abs diff 6e-8 (JSON rounding),
so the instrumented runs are the same trajectories as the reported 0.86245.

---

## (1) Oracle ceilings, measured

### (1a) Stationary oracle — fresh net, unpermuted MNIST, 5000 online steps

Per-task ceiling for a from-scratch learner (no continual problem, no carry).
3 seeds; "late window" = steps 4000-5000 of the task.

| method | avg online acc (the metric) | late-window acc |
|---|---|---|
| `sigma0_ndecay099` (champion) | 0.7842 (sd 0.0028) | 0.8843 |
| `adamw_control` (protocol AdamW) | 0.7671 (sd 0.0028) | 0.8593 |
| `sgd_ema_norm` | **0.8301** (sd 0.0027) | 0.8857 |

- A from-scratch learner pays ~0.12-0.20 of metric just learning the task: the
  first 250 steps average 0.15-0.51 accuracy for every method.
- The protocol baseline reproduction (0.7791) is essentially the stationary
  from-scratch score — i.e. the baseline transfers almost nothing across tasks.
- The champion transfers substantially: its protocol per-task average (0.8629 on
  tasks 20-199) is **+0.079 above its own from-scratch score** — that is what
  "carrying features across permutations" is currently worth.
- Note `sgd_ema_norm` is *faster from scratch* than the champion (the utility
  gate ~halves the effective step early on a fresh net); the champion wins under
  the protocol because it protects and reuses features across tasks.

### (1b) Carried oracle — champion with the SAME permutation every task

No non-stationarity at all: the identical method, but permutation of task 0 is
reused for 60 tasks (300k stationary steps; fresh example draws per task).
This is the asymptote of the champion family with perfect feature reuse.

| method | t1 | t10 | t20 | t40 | t60 | late (last 10 tasks) | late-task late-window |
|---|---|---|---|---|---|---|---|
| `sigma0_ndecay099` (3 seeds) | 0.786 | 0.926 | 0.932 | 0.927 | 0.934 | **0.9324** (sd 0.0007) | 0.9328 |
| `adamw_control` (1 seed) | 0.769 | 0.942 | 0.962 | 0.971 | 0.976 | **0.9741** | 0.9747 |

Two decisive facts:

1. **The champion family saturates at ~0.933.** Even with unlimited stationary
   experience and zero interference the champion never exceeds ~0.933-0.936 —
   this is the noise floor of constant-step-size one-example SGD with the gate
   and decay, not a horizon effect (the curve is flat from ~task 20 onward).
   **0.95 is unreachable for this method class even if the continual-learning
   problem were removed entirely.**
2. **The architecture + online-single-example regime is NOT the binding
   constraint.** Plain protocol AdamW on the same stationary stream reaches
   0.974 online accuracy — within 0.007 of the converged batch reference. The
   often-suspected "online single-pass tax" is ~0.007, not ~0.05.

### (1c) Batch reference — converged 300x150 MLP on MNIST test

Minibatch Adam(1e-3), batch 128, 30 epochs, protocol init/scaling, 2 seeds:
test accuracy **0.981 / 0.982** (best), 0.979 / 0.981 (final); train ~0.995.
Matches the well-known ~0.98 for this architecture class.

**Ceiling ladder (measured):**

```
0.981   batch-converged 300x150            (architecture cap)
0.974   AdamW, online 1-example, stationary (online tax: -0.007)
0.933   champion family, stationary         (family optimizer floor: -0.041)
0.904   champion within-task plateau under the protocol (5000-step horizon: -0.029)
0.862   champion score                      (re-adaptation transient: -0.041)
0.779   baseline reproduction               (= from-scratch, no transfer)
```

---

## (2) Error budget of the champion at 0.86245

From per-step accuracy of the exact confirm_full trajectories (200 tasks x 5000
steps x 3 seeds). Total error **E = 0.13755**. Within-task plateau = mean
accuracy in steps 4000-5000 of each task = **0.90373**.

| component | size | share of E |
|---|---|---|
| (i) within-task re-adaptation transient (steps 0-4000 below the task's own plateau) | **0.04128** | 30.0% |
| (ii) asymptotic within-task error (1 - 0.90373) | **0.09627** | 70.0% |
| (iii) late-life drift | **~0** (plateau tasks 20-60: 0.9045; tasks 160-200: 0.9052; slope +3e-7/task) | 0% |
| early-life warmup (tasks 0-19) | 0.0004 | 0.3% |

Transient detail (mean within-task curve; excess vs the task's own plateau):

| step bucket | acc | excess err | contribution to E | % of E |
|---|---|---|---|---|
| 0-50 | 0.274 | 0.630 | 0.0063 | 4.6% |
| 50-100 | 0.468 | 0.436 | 0.0044 | 3.2% |
| 100-250 | 0.651 | 0.252 | 0.0076 | 5.5% |
| 250-500 | 0.779 | 0.125 | 0.0062 | 4.5% |
| 500-1000 | 0.839 | 0.064 | 0.0064 | 4.7% |
| 1000-2000 | 0.873 | 0.031 | 0.0062 | 4.5% |
| 2000-3500 | 0.891 | 0.012 | 0.0037 | 2.7% |
| 3500-5000 | 0.902 | 0.002 | 0.0005 | 0.4% |

First-500-step accuracy is 0.659 vs plateau 0.904 (gap 0.245); the 0-500 window
holds 0.0245 of E (17.8%), the 500-4000 tail another 0.0168 (12.2%).

**Which bucket is biggest?** The *asymptotic* bucket (70%) — but it splits into
two different mechanisms when read against the ceilings in (1):

- **0.904 → 0.933 (0.029, 21% of E): within-task convergence speed.** The same
  method reaches 0.933 given ~50-100k stationary steps; 5000 steps/task is simply
  not enough for its tail convergence. This is optimization speed, not capacity.
- **0.933 → 0.974 (0.041, 30% of E): the family's optimizer floor.** Constant-lr
  gated SGD asymptotes 0.041 below what AdamW achieves on the identical stream.
  This is preconditioning/step-size adaptation, sacrificed for continual
  stability (protocol AdamW *under the protocol* collapses to no-transfer 0.779
  behavior, and every Adam+protection arm screened (adamw_cbp* family) topped
  out at 0.801).
- There is **no late-life drift bucket**: the champion is drift-free over 200
  tasks (that battle is won; nothing to gain here).

So the error budget is: **30% re-adaptation transient, 21% within-task horizon
(convergence speed), 30% family optimizer floor, 19% shared architecture/online
floor (1 - 0.974 = 0.026 of irreducible error under any online method)**.

---

## (3) Verdict: the realistic ceiling, and what each point costs

**Hard bounds established empirically:**

- **Metric ceiling of the current champion family: 0.933.** Zero transient +
  full asymptote = the carried-oracle plateau. 0.95 is excluded *within this
  family* by ≥ 1.7 points.
- **Protocol ceiling for any online method: ≈ 0.96-0.97.** Perfect-transfer
  behavior equals the carried-oracle curve; for an Adam-class learner that is
  0.974 sustained, minus an unavoidable new-permutation inference cost (even an
  oracle that knows MNIST must observe some examples to identify the new pixel
  map; at ~50-100 informative steps/task that costs ~0.005-0.01 of the metric).
  0.95 sits *below* this bound — it is not information-theoretically excluded.
- **Capacity does not bind.** The 300x150 net reaches 0.981 batch / 0.974
  online-stationary. A bigger net or replay is *not* required for 0.95 in
  principle; breaking protocol on architecture attacks the wrong constraint.

**Mechanism map — what each remaining point requires:**

| target | gap closed | mechanism class required |
|---|---|---|
| 0.862 → ~0.88 | halve the transient (0.041 budget) | faster re-conditioning at task boundaries — boundary-adaptive normalizer decay, warm-started statistics, transient-aware step sizes. Incremental; same family. The 0-500-step window alone holds 0.0245. |
| ~0.88 → 0.904 | zero the transient | near-instant permutation re-mapping: infer the new pixel map from per-pixel statistics (pixel marginals identify the permutation) and permute the input layer instead of re-learning it. New mechanism, protocol-legal. |
| 0.904 → 0.933 | 5000-step horizon → family asymptote | faster within-task convergence: per-weight preconditioning that stays continually stable (the screened IDBD/Autostep/Adam+CBP arms all failed to combine both). |
| 0.933 → 0.95+ | exceed the family floor | Adam-class asymptote (0.974 measured) *sustained across 200 permutations* — i.e. solve plasticity-stability for adaptive optimizers — AND near-oracle transfer at boundaries. Both at once. Research program, not tuning. |

**Bottom line.** The realistic ceiling is **~0.90** for transient-side
improvements to the current champion, **~0.93** if cross-permutation transfer is
solved within the family, both confirmed against the empirical family cap of
0.933. **0.95 is reachable only by a method that combines Adam-class stationary
convergence with continual stability and fast permutation transfer** — the
protocol itself (one example/step, 5000 steps/task, 300x150) leaves 0.96-0.97 of
headroom, so no protocol constraint needs breaking; what binds is the absence of
any known optimizer in that combined class. If the goal is a headline ≥0.95
*without* new mechanisms, the only routes are protocol-breaking: multiple passes
or replay per task (attacks the 0.904→0.933 horizon gap directly) — architecture
scaling alone would not suffice, since even the batch cap of this net is 0.981
and the transient/optimizer terms dominate the deficit.

---

### Reproduction

- Runner: `outputs/ipmnist_screening/ceiling/ceiling_runs.py`
  (modes: `stationary` / `carried` / `full` / `batch`; exact screening-spec
  seed-derivation and RNG chain; per-step accuracy to `*_per_step.npy`).
- Analyzer: `outputs/ipmnist_screening/ceiling/ceiling_analyze.py` →
  `analysis_summary.json` (all numbers above).
- Cross-check: full-mode per-task means vs `confirm_full/sigma0_ndecay099_seed{0,1,2}.json`
  max abs diff 6e-8.
- Caveats: seeds 0-2 (carried AdamW: seed 0 only); the ~0.96-0.97 any-method
  protocol ceiling combines the measured 0.974 stationary asymptote with an
  *analytic* estimate of the unavoidable permutation-inference cost — the
  transfer-oracle itself was not implemented; carried-oracle horizons are 60
  tasks (curves flat well before the end).

---

## What number do we need

**(2026-08-03 addendum — the honest target ladder, written after the
transient-attack wave. All numbers are development diagnostics; artifacts in
`outputs/ipmnist_screening/nb_ensemble/` and `confirm_full/`.)**

The question section (3) leaves open is not "is 0.95 reachable" (settled: not
in this family) but *what target is worth aiming at and what each rung
actually means*. The record is **0.86449 ± 0.00009** (`sigma0_shiftnorm_d099`,
n=20). Its residual error decomposes, freshly measured on the record champion
itself (seed-0 per-step trace, 60 tasks, shifted tasks only):

- within-task plateau **0.9015** (the shift-adaptive detector's mid-task
  false triggers price ~0.002 of the 0.9037 family plateau — the detector
  buys its boundary speed with a small stationary tax);
- residual re-adaptation transient **0.0366** of the metric, of which
  **0.0223 sits in the first 500 post-shift steps** and 0.0143 in the
  500-4000 tail;
- beyond the transient: the 5000-step convergence shortfall (plateau → 0.933
  family asymptote, ~0.031 from this champion) and the optimizer floor
  (0.933 → 0.974), proven non-importable by seven hybrid negatives.

**The ladder** (each rung = a mechanism class fully cashed in, from 0.86449):

| rung | number | what it means |
|---|---|---|
| record | 0.86449 | current champion, n=20 |
| screen co-best | 0.8657 | `disc_r1_pscale_norms` (structure transfer) |
| **this wave** | **0.86671 / 0.86678** | `nb_ensemble_nbreset` 60-task screen / 200-task confirm (seeds 0-2): +0.0022 paired over the champion on ALL seeds, 59/59 shifted tasks improved, horizon-stable (+0.0018 late-window) |
| early window zeroed | ~0.887 | first-500 transient fully captured (+0.0223) — the ceiling of "fast member" ensembles/switches whose members re-estimate in ≤500 steps |
| transient zeroed | ~0.9015 | the champion's own plateau; nothing left of budget (i) |
| **0.90 (target)** | — | = capturing **97% of the measured residual transient** (0.0355/0.0366), or equivalently ~86% of the family's original 0.0413 transient budget; within 0.001 of the transient-zeroed line |
| 0.8925 (milestone) | — | 76% of the residual transient: the whole 0-500 window plus ~half the tail |
| solved-problem-class | 0.933 | family asymptote: transient AND 5000-step shortfall both zeroed; only the (unimportable) optimizer floor remains |
| any-online-method | 0.974 | protocol AdamW stationary asymptote |
| architecture cap | 0.981 | batch-converged 300x150 |

So **0.90 is not an arbitrary round number: it is the transient-solved line**
— and simultaneously the "within 0.03 of the 0.933 family asymptote" line,
which is the natural *publish-grade* bar: a method scoring ≥0.90 here
demonstrably operates within noise of its own measured mechanism ceiling,
and the demonstration becomes defensible when paired with cross-protocol
generality (the label-permuted EMNIST lane, where the same conditioning
mechanisms must not collapse).

**What this wave measured about reachability.** The adaptive champion/NB
ensemble (`nb_ensemble_champion`: accuracy-weighted probability mixture,
vote weights = online annealed per-member correctness EMAs, decay 0.995 /
softmax temperature 80, no oracle) plus two probes:

1. **The "flat naive Bayes" premise is refuted at step granularity.** The
   standalone NB tracker looks flat per task (~0.785 from t1), but per-step
   it has its own post-shift transient — first-500 shifted-task accuracy
   0.634 vs the champion's 0.678. Task-level flatness was aliasing: NB's
   class statistics are as stale after a permutation as anyone's; they just
   re-estimate on a ~500-step window that a per-task average hides. A vote
   between two similarly-transient members has almost nothing to switch to:
   the bare ensemble gains only +0.00096 (mostly task-0 warmup, where NB
   genuinely is ahead of a from-scratch net by +0.048).
2. **Making the NB member itself shift-robust is what pays**
   (`nb_ensemble_nbreset`): the raw-pixel shift detector (boundary
   shifted-feature fraction 0.034-0.061 vs mid-task p99 0.0077 — a >2x
   separation both ways, trigger frozen at 0.03) resets NB's per-class
   anneal clocks, so its statistics re-estimate at effective decay 1/2.
   Post-shift buckets vs champion: +0.069 (steps 50-100), +0.063 (100-250),
   tie from 250 on; the vote swings to NB in 98% of tasks at mean step 121
   (vs 15%/307 without the reset) and hands back with a measured -0.006
   switch-back lag at 500-1000. Screen 0.86671, confirm 0.86678 ± 0.00037.
3. **A third closed-form member (linear RLS over normalized pixels) adds
   warmup only** (`nb_ensemble_rls3`: t1 +0.056, shifted tasks +0.0001):
   its sufficient statistics are exactly as permutation-stale as NB's.

**The measured wall.** Even a per-example oracle over {champion, NB} — pick
whichever is right at every step — reaches only **0.8975** on shifted tasks
(0.43/0.65/0.76 in the 0-50/50-100/100-250 buckets): in the first ~100
post-shift steps *no member in the pool is right*, because every member's
sufficient statistics were permuted. Ensembling redistributes accuracy that
exists in the pool; it cannot create post-shift accuracy no member has. The
nbreset ensemble captured 5.3% of the residual transient; the remaining 95%
lives mostly below step 250. Reaching 0.90 therefore needs a mechanism whose
*state survives the permutation*: identification is information-bounded at
~2,000 samples for first-order statistics (NEW_DIRECTIONS V1), so the live
candidates are higher-order/model-side identification, recurrence-indexed
context reuse (direction D), or attacking the other budget entirely — the
0.029 convergence shortfall (the `rls_head` lane). The honest reading: 0.90
is reachable only if one of those classes lands; without them, the
transient-attack ceiling for fast-member ensembles is the ~0.887 early-window
line, and the measured state of the art on it is 0.8668.

Reproduction: runner/analyzer + per-step and vote-weight traces in
`outputs/ipmnist_screening/nb_ensemble/` (`nb_ensemble_runs.py`,
`nb_ensemble_analyze.py`, `analysis_nb_ensemble_{champion,nbreset}_seed0.json`);
screen shards `shards/nb_ensemble_*_seed{0,1,2}.json` merged in
`summary_nb_ensemble.json`; 200-task confirmations
`confirm_full/nb_ensemble_nbreset_seed*.json` (seeds 3-19 running at close of
this pass — the 20-seed publication read follows their completion).
