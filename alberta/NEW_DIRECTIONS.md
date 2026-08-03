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
