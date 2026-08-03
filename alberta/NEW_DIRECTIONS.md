# Continual learning is hard: what a 60-arm campaign taught us, and where genuinely new architectures could go

Status: research essay + pre-registered minimal validations (development,
nonpromoting). Grounded entirely in this repo's measured campaign:
reproduction 0.7791 → record 0.86449 ± 0.00009 (n=20) on the ICLR-2024
Input-permuted MNIST protocol, ~60 screened arms, 25+ ledgered negatives,
an audited decomposition, and a measured ceiling (family 0.933 /
achievable-class 0.974 / batch 0.981).

## 1. Why this problem is genuinely hard

**The metric charges you for learning.** Online average accuracy counts
every prediction made *while* adapting. There is no train/test split to
hide in: a method that adapts in 500 steps instead of 2000 wins even if
both converge identically. This inverts deep-learning intuition — the
binding constraint is not what you can represent but how fast you can
*re-estimate*.

**Non-stationarity is not one thing.** Our cross-protocol results split it
cleanly: input-distribution shift (IPMNIST) is dissolved almost entirely by
fast input-statistics tracking (+0.061 of our +0.085 came from
conditioning); label-mapping shift (L/P EMNIST) is untouched by
conditioning (bare-conditioned SGD collapses to 0.50 there) and requires
protection/consolidation machinery (the gate). Methods are not "continual
learning methods"; they are *specific-non-stationarity* methods, and the
literature's failure to factor this is why its comparisons confuse.

**Stale state is the universal pathology.** Every mechanism that carries
statistics across a boundary is a liability at that boundary: slow
normalizer means (our decay star), Adam's second moment (every gated-Adam
hybrid lost; the naive composition's v carried stale scale), meta-learned
step-sizes (Autostep collapse at batch 1; SwiftTD-stabilized IDBD −0.175),
and APOLLO's channel statistics (0.8472 despite the best calibration probe
of the sweep). The champion wins because it holds *almost no state that can
go stale*: fast-tracked input statistics (re-estimated in ~100 steps,
shift-triggered per-feature), a utility EMA, and the weights themselves.
**Adaptivity is a bet that the past predicts the future; non-stationarity
is precisely when that bet fails.**

**Always-on shared parameters are forgetting channels.** Found three
independent times (RL biases, un-gated feature blocks, the head under label
permutation). Memory is a *representational* property — exclusive gating
(zero activation ⇒ zero gradient ⇒ perfect retention) — not an optimizer
property. Optimizer-level protection (UPGD's gate) is worth +0.011; its
slot is substitutable (L2-Init ties it at 0.8641).

**Depth's edge is smaller than anyone admits.** A no-backprop random-feature
RLS tracker reaches 0.848 — the deep champion's whole advantage over
closed-form linear tracking is +0.017. Most of what the 300×150 network
"learns" per task, a fixed random basis plus recursive least squares tracks.

## 2. The measured error budget — what is actually left

The champion's remaining error (0.135) decomposes as: **~30% re-adaptation
transient** (the first ~500 steps after each permutation, before statistics
and weights re-align) and **~70% asymptotic** (0.029 within-5000-step
convergence shortfall + 0.041 optimizer floor). We proved the optimizer
floor is *not importable* — seven designed hybrids including APOLLO all
lost; gated SGD at batch 1 beats every adaptive rule under non-stationarity.
So the only attackable budget with current architecture is the transient.
And the transient is not an optimization problem. It is an **identification
problem**.

## 3. The key insight nobody in this literature is using

An input permutation does not destroy information. It *relabels
coordinates*. The task after a shift is not "learn a new function" — it is
"discover which wire moved where." Our own mechanism data says the
identifying information is present and cheap: per-feature running
statistics (mean, variance — exactly what our normalizer tracks per
feature) are permutation-*covariant* fingerprints. MNIST pixels have wildly
heterogeneous marginal statistics (corner pixels ≈ 0 always; center pixels
high-variance), so **matching per-feature statistics across the boundary
recovers the permutation as an assignment problem** — solvable in closed
form (sort/Hungarian on statistic vectors), no gradient descent at all. If
the assignment is recoverable from a few hundred samples, the transient
collapses: apply the inverse permutation and the *old network is already
correct*. Gradient descent is the wrong tool for a combinatorial
re-labeling; we have been paying a 2000-step gradient tax on a problem with
a 200-sample closed-form answer.

This generalizes beyond the benchmark trick: it is the Alberta Plan's state
construction applied to the supervised stream — *treat the non-stationarity
as a latent variable, identify it, and index memory by it*. We already
built the machinery in the RL track (`core/context_inference.py`: K-slot
bank, change-point detection, slot re-use = memory of contexts). The new
direction is unifying them.

## 4. Genuinely different architectures worth building

**(A) Alignment-first architecture (permutation as latent state).** A
frozen-or-slow trunk plus an *input alignment layer* — a permutation/soft
assignment estimated online from feature statistics, re-solved at detected
shifts. Learning splits into: slow weight learning (rare, protected) and
fast combinatorial identification (per shift). Prediction: near-zero
transient on IPMNIST; ceiling jumps toward the 0.933 family asymptote
minus only the identification lag. Novelty: no plasticity-literature
method treats the shift as an inference target; they all re-learn weights.

**(B) Streaming generative classifier (no gradients anywhere).** Online
class-conditional diagonal Gaussians (streaming naive Bayes) with the same
fast-EMA statistics. Permutation just permutes the stored means — which
the aligner can un-permute. Everything is closed-form, per-feature, and
shift-robust by construction. Even standalone (no alignment) this is an
untested baseline class on this protocol; with alignment it becomes a
memory system: statistics are stored *per context* and re-attached on
recurrence.

**(C) Dual-speed fast-weights.** Persistent slow feature bank + cheap
disposable readout (Hebbian/RLS) rebuilt per regime. Our data says
readouts re-learn in hundreds of steps while bodies take thousands; make
that split architectural. The RFF+RLS result (0.848 with a *random* bank)
says the slow bank barely needs training at all.

**(D) Statistics-gated modular experts.** The conditioning statistics that
identify the regime also *route* it: K small experts gated by
context-inference over input statistics, exclusive activation = exclusive
gating = retention by construction (the convention-game mechanism, applied
to supervised streams). This is the architecture our RL-track results have
been pointing at all along.

## 5. Pre-registered minimal validations (cheap, decisive)

- **V1 — Assignment recovery (numpy, minutes):** after a permutation, match
  per-feature running mean/var (optionally + class-conditional means)
  old-vs-new via sort/Hungarian. Measure assignment accuracy vs samples
  {50, 200, 500, 2000}. *Promote if >90% of relevant (non-constant) pixels
  correctly assigned within ≤500 samples.* Refutes direction A if
  statistics are too degenerate (many identical-statistic pixels).
- **V2 — Alignment-composition arm:** champion network, but at each
  detected shift estimate the alignment and apply the inverse permutation
  instead of (or before) re-learning. Screen at 60 tasks. *Promote if
  transient (first-500-step error) halves and screen exceeds 0.870.*
  This would be the largest single-step gain of the campaign if V1 holds.
- **V3 — Streaming naive Bayes baseline:** protocol-exact, no gradients.
  *Promote as a baseline row regardless of score; promote as a method if
  >0.80 (it would beat published SOTA with no network).* 
- **V4 — Dual-speed RFF+RLS with per-context readout cache:** re-use the
  RFF+RLS lane; cache/restore readouts keyed by context-inference. On
  IPMNIST (no recurrence) predict no gain (control); on a recurring
  variant, predict instant recovery. *Promotes direction D's memory claim.*

Failure of V1/V2 is informative: it would mean the identifying statistics
are insufficient, and the honest conclusion becomes "the transient is
irreducible without richer fingerprints (class-conditional, pairwise)" —
itself a measurable next rung.

## 6. The wider opportunity

The field's benchmarks reward exactly what our record exploits: fast
re-estimation. But real continual worlds (the Alberta Plan's target) mix
input drift, label drift, recurrence, and structure that is *identifiable
rather than relearnable*. The deepest lesson of this campaign: **treat
non-stationarity as something to be identified, indexed, and re-attached —
not merely survived.** Conditioning was the first instance (identify the
scale, divide it out). Alignment is the second (identify the relabeling,
invert it). Context-indexed memory is the general case — and it is,
recognizably, the Alberta Plan's own thesis arrived at from the opposite
direction: through 60 arms of supervised-stream evidence.

## Validation results (2026-08-03, development, nonpromoting)

Executed in pre-registered order; artifacts under `outputs/new_directions/`
plus screening shards under `outputs/ipmnist_screening/`. Refutations are
reported at the same rank as promotions.

**V1 — Assignment recovery: REFUTED.** Protocol data + protocol
permutations, seeds 0-2 x boundaries 0-2, champion fast-EMA (decay 0.99)
statistics on both sides, Hungarian + global-greedy assignment
(`outputs/new_directions/V1_assignment.{json,md}`). Relevant-pixel
(reference var > 0.01, ~507/784) assignment accuracy vs the
pre-registered >0.90-within-500-samples bar:

| N | mean/var | + class-conditional means |
|---|---|---|
| 50 | 0.010 | 0.197 |
| 200 | 0.017 | 0.619 |
| 500 | 0.019 | **0.785** |
| 2000 | 0.015 | 0.840 |

The sanity/oracle probes localize the failure precisely: with exact
full-dataset statistics the assignment IS fully recoverable (99.2%
relevant from marginal mean/std; 100% with class means), and the
no-shift control clears 99.6% — the mapping and machinery are correct.
The binding constraint is estimator precision: MNIST marginal statistics
are nearly radially symmetric, with inter-pixel separations below the
~100-sample-window EMA noise, and even a frozen 5,000-sample reference
with plain sample statistics post-shift
(`V1_assignment_exploratory.json`, post-hoc) reaches only 0.80 at N=500
and crosses 0.91 only at N≈2000. **Identifying the permutation from
first-moment fingerprints costs ~2,000 samples — the same order as the
gradient transient it was meant to replace.** Section 3's "200-sample
closed-form answer" does not exist at this protocol's noise level; the
next rung requires richer fingerprints (pairwise/model-side statistics)
and must beat that measured ~2,000-sample information floor to matter.

**V2 — Alignment-composition arm: GATED OUT (not executed).** The
pre-registration makes V2 conditional on V1 promotion; V1 refuted, so no
`align_champion` arm was implemented or screened. Given V1's measured
identification lag (~2,000 samples for ~0.9 relevant accuracy), the arm's
predicted mechanism — solve the assignment inside the first ~500
post-shift steps and skip the transient — is not available: by the time
the assignment is solvable, the champion's own re-adaptation has already
re-converged most of the transient.

**V3 — Streaming naive Bayes: baseline row promoted; method bar NOT
met.** Registered as screening arm `naive_bayes` (class-conditional
diagonal Gaussians, annealed fast-EMA statistics, argmax posterior, no
gradients, no MLP; `nb_decay` 0.98 / var floor 0.1 frozen by a 2-task
seed-0 diagnostic before the screen). 60-task screen, paired seeds 0-2
(`outputs/ipmnist_screening/summary_naive_bayes.json`,
`outputs/new_directions/V3_naive_bayes.json`): **0.78510** per-seed
[0.784617, 0.78512, 0.785567] — below the pre-registered 0.80 method bar
(no 200-task confirmation triggered), but **+0.00734 paired over the
published-configuration UPGD-W control (0.77776), all seeds improve**. A
closed-form statistic tracker with no network beats the published deep
SOTA configuration on its own protocol, while remaining ~0.064 below the
rff_rls tracking control (0.8490) and ~0.079 below the conditioned
champion (0.86396): exactly the ordering the tracking-not-learning thesis
predicts — and standalone direction B is refuted as a record path.

**V4 — Dual-speed RFF+RLS readout cache: not executed** (out of scope of
this validation pass; remains pre-registered).

Net verdict for section 4: direction A's cheap-identification premise is
refuted at the first-moment fingerprint class on this protocol; direction
B survives only as a strong baseline, not a method. The measured
~2,000-sample identification floor is itself the new result: on IPMNIST,
*re-learning the input layer by conditioned gradient descent is
information-theoretically competitive with explicitly identifying the
permutation from first-order statistics* — the transient is not a
combinatorial free lunch. Identification-first architectures need either
higher-order fingerprints or recurrence (context re-use, direction D)
to pay for themselves.
