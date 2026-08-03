# V1 — Assignment recovery across a permutation boundary

Pre-registered validation V1 of `NEW_DIRECTIONS.md` (development
diagnostic, nonpromoting). Full numbers: `V1_assignment.json`
(pre-registered protocol) and `V1_assignment_exploratory.json` (post-hoc
estimator diagnostic).

## Protocol (as pre-registered)

IPMNIST protocol data and permutation machinery
(`upgd_ipmnist.load_mnist_train`, `build_schedule`); seeds 0-2 x task
boundaries 0-2 (9 cells). Reference statistics: the champion's own
annealed fast-EMA (decay 0.99, `ema_normalize` equations) per-feature
mean/variance over the 5,000-sample pre-shift task, plus class-conditional
mean EMAs (labels observed online post-prediction — legal). Post-shift:
fresh statistics after N in {50, 200, 500, 2000} samples. Assignment by
Hungarian (`scipy.optimize.linear_sum_assignment`) and global-greedy on
the statistic vectors. Relevant pixels: reference variance > 0.01
(~507/784); constant background pixels are mutually interchangeable, so
only relevant-pixel accuracy is functionally meaningful.

**Promotion criterion (pre-registered): >90% of relevant pixels correctly
assigned within <=500 samples.**

## Verdict: REFUTED

| N | mean/var only (Hungarian) | + class-conditional means (Hungarian) |
|---|---|---|
| 50 | 0.010 | 0.197 |
| 200 | 0.017 | 0.619 |
| 500 | 0.019 | 0.785 |
| 2000 | 0.015 | 0.840 |

(mean relevant-pixel accuracy over the 9 cells; greedy is uniformly worse
than Hungarian by ~0.1 at the class-conditional fingerprint.)

No configuration reaches 0.90 within 500 samples. Direction A's premise —
that the champion's own per-feature running statistics identify the
permutation cheaply — is refuted at the pre-registered operating point.

## Why it fails (sanity + oracle probes, exploratory)

1. **The information IS present in the limit.** With exact full-dataset
   statistics on both sides, Hungarian matching recovers 99.2% of
   relevant pixels from marginal mean/std alone, and 100% with
   class-conditional means. The no-shift control (same permutation both
   sides) recovers 99.6%/100% — the pipeline and ground-truth mapping are
   correct.
2. **The estimator, not the fingerprint, is the binding constraint.**
   MNIST marginal statistics are nearly radially symmetric: neighboring
   pixels differ by less than the fast-EMA's standard error (decay 0.99
   = ~100-sample effective window on BOTH sides), so marginal matching
   collapses to ~1-2% — near chance — at every N. Class-conditional
   means break the symmetry (10 extra dimensions) but each class sees
   only ~N/10 samples.
3. **A slow/frozen reference does not rescue it within budget**
   (post-hoc estimator swap, `V1_assignment_exploratory.json`): with a
   perfect 5,000-sample plain-average reference and plain sample
   statistics post-shift, class-conditional matching reaches 0.80 at
   N=500 and only crosses 0.91 at N=2000 — the identification itself
   costs ~2,000 samples, the same order as the gradient re-adaptation
   transient it was meant to replace.

## Reading

The honest conclusion pre-registered in the essay applies: the transient
is not reducible by statistic-matching at this budget — identifying a
784-way relabeling from class-conditional first moments needs roughly as
many samples as gradient descent needs to re-learn the input layer. The
"200-sample closed-form answer" claimed in section 3 of the essay does
not exist at the protocol's noise level; richer fingerprints (pairwise
statistics, model-side probes such as per-unit activation correlations)
are the measurable next rung, and any of them must beat the ~2,000-sample
information floor measured here to matter.

Consequence for the pre-registered chain: **V2 (alignment-composition
arm) is gated out** — its premise requires V1 promotion. Not implemented,
not screened.
