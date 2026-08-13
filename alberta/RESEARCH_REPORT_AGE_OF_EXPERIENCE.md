# The Age of Experience: research report and forward plan

Status: development research synthesis (2026-08-02). Nothing in this document
is promoted evidence; all campaign numbers referenced are development-grade
and nonpromoting under the repository's evidence rules.

Provenance audit: the stored IPMNIST curves reconstruct the numbers cited
below, but they do not constitute a completed current-source campaign. The
checked-in proxy receipt rejects the UPGD prefix comparisons, the summary is
missing 12 current shards, the round-2 driver cannot consume the shard schema,
and v1 shards bind no source, command, or dataset bytes. “Best,” “champion,”
and mechanism language below therefore means only highest stored development
mean or research hypothesis. It is not a SOTA, causal, or cross-horizon
claim; a fresh source-bound lifecycle must attempt to reproduce it.

This report answers, in order: (1) is our result actually SOTA and what is
everyone else doing; (2) what optimizer/mechanism improvements are still on
the table; (3) what completely different approaches deserve arms; (4) what
the Alberta Plan literature and Sutton's recent program say is missing for
agents that learn from experience; (5) where our current approach is weak;
(6) the derived next experiment set.

## 0. Where we stand

On the ICLR-2024 UPGD input-permuted MNIST protocol (1M examples one per
step, permute every 5000, 200 tasks, 300x150 ReLU MLP, average online
accuracy):

| Method | Online acc. | Note |
|---|---:|---|
| Published SOTA (UPGD-W, our 10-seed repro) | 0.7791 | best in print |
| Best protocol-pure improvement (`adamw_cbp_r3e4`) | 0.80126 | no input-encoding change |
| **Highest stored mean (`sigma0_ndecay099`)** | **0.86245** | EMA input norm (decay 0.99) + utility-gated SGD, no noise |

The stored means motivate, but do not authenticate, this mechanism hypothesis:
input-statistics conditioning +0.061 (dominant), utility gate +0.011,
perturbation noise: load-bearing on raw inputs (−0.035 to remove), neutral
under slow conditioning, **harmful under fast conditioning** (−0.002). The
decay star peaks at 0.98–0.99 (~100–200-step effective window against
5000-step tasks); slower or faster tracking loses symmetrically. Design laws
from the campaign: any always-on shared parameter is a forgetting channel;
exclusive gating is memory; protection pays rent only where tasks recur.

## 1. Benchmark landscape — where the result can travel

(Fleet survey, 2026-08-02. Full table with citations in the survey text
below; headline conclusions here.)

- **Continual ImageNet (Dohare et al., Nature 2024)** is the single
  highest-leverage port: canonical loss-of-plasticity benchmark, real
  class-distribution shift rather than pixel permutation, small from-scratch
  networks, published baseline story (BP 89%→77% by task 2000; CBP holds
  ~89%). Matching/beating CBP there converts the claim from "wins a
  permutation benchmark" to "input-conditioning-speed generalizes to real
  distribution shift."
- **Slowly-changing regression (same paper)** is nearly free (~15
  CPU-min/run) and is the sharpest falsification lane: inputs near-
  stationary, target drifts. Prediction: input-norm inert, gate active. A
  predicted dissociation is stronger evidence than another win. (Note: the
  repo already has a pinned `outputs/slowly_changing_regression/` lane with
  immutability rules — any new runs go to new paths.)
- **Forager/Foragax (Tang et al. 2026, arXiv:2605.01131)** — already
  in-repo; extends the claim to partially-observable reward-driven streams,
  same Alberta lineage. (Authorship note: Tang, Xiong, et al., not Bhatt.)
- **Label-permuted mini-ImageNet** completes the UPGD paper's triptych;
  cheap (same harness) and reviewers will expect it.
- **Plasticine (2025)** — third-party unified plasticity harness; entering
  our method there buys external validity.
- CLOC/CGLM (real chronological photo streams) are a poor experimental fit
  (pretrained frozen backbones dominate; they sidestep plasticity) but a
  necessary related-work contrast.
- Prior art to engage explicitly in any paper: Continual Normalization
  (arXiv 2203.16102), CoLLAs 2025 reinit-works-better-with-LN
  (arXiv 2508.00212), BN covariate-shift framing. None studies fast online
  EMA tracking of raw input statistics at one example per step as the
  load-bearing mechanism — that niche is open.

## 2. Optimizer landscape (APOLLO and friends)

The organizing fact: at batch size 1 a dense layer's gradient is exactly
rank-1, `G = δ xᵀ`, so **column-wise (fan-in) gradient scaling is input
normalization at the weight level** — including for hidden layers, where it
becomes hidden-activation conditioning our input-only normalizer does not
provide. This makes APOLLO's channel-wise scaling mathematically continuous
with our winning finding, and makes its random-projection state unnecessary
at our scale: we can keep exact per-column second moments in O(n) per layer.

Survey verdicts (full per-optimizer table in the fleet transcript;
staleness = state invalidated at each permutation boundary):

- **APOLLO** (2412.05270): channel-wise norm-ratio scaling of the raw
  gradient via Adam-on-projected-state. Right structure; at batch 1 the
  distilled exact form ("column-wise RMS-scaled gated SGD") supersedes it.
  Its Norm-Growth Limiter (clip update-norm growth ratio) is a free
  stabilizer worth porting.
- **Muon**: momentum + Newton–Schulz orthogonalization; *no second-moment
  state at all* — best staleness profile in the family; spectral
  equalization is conditioning-by-construction. Genuine new-best candidate.
- **One-sided Shampoo right factor**: at batch 1, `GᵀG = ‖δ‖² x xᵀ` — an EMA
  of the input covariance; full whitening vs our diagonal normalization is
  the natural next mechanism question.
- **Cautious wrapper (C-AdamW mask)**: zero any update coordinate whose sign
  disagrees with the current gradient — precisely a stale-momentum
  suppressor at task boundaries; five lines, stateless.
- **Lion**: sign updates = stateless implicit conditioning; cheap datapoint.
- Predicted to **hurt** (stale state at boundaries): GaLore/Fira (periodic
  SVD projector), SOAP/SPlus (periodic eigenbasis), Sophia (stale Hessian),
  schedule-free (iterate averaging drags in pre-boundary weights), AdEMAMix
  as published (slow direction EMA spans unrelated tasks), MARS
  (variance-reduction assumes smooth drift), Adam-mini (row-wise = wrong
  axis; fan-in is the conditioning axis).
- Literature check: nobody has run APOLLO/Fira/Adam-mini/cautious on a
  streaming plasticity benchmark; "Adam on Local Time" (2412.17113) resets
  Adam bias-correction at nonstationarities (supports the staleness
  framing); lower β2 reducing dead units (Lyle/Dohare) supports the
  tracking-speed framing.

Ranked implementable arms (each <150 lines JAX):
1. **`colnorm_gate`** — column-wise RMS-scaled gated SGD, decay 0.99, both
   layers. Prediction ≥0.862; extends conditioning to hidden activations.
2. **`muon_gate`** — Nesterov momentum 0.95 + 5 NS iterations +
   utility gate + EMA input norm. Prediction 0.85–0.87.
3. **`shampoo1s_gate`** — one-sided input-whitening (EMA covariance,
   inverse-sqrt every 100 steps). Tests off-diagonal correlations beyond
   diagonal scaling. Prediction 0.855–0.868.
4. **cautious mask** on any momentum arm — boundary-localized gain.
5. **`lion_gate`** with EMA norm — stateless-conditioning datapoint.
6. Granularity/staleness controls: tensor-wise scalar (APOLLO-Mini),
   row-wise (Adam-mini axis test), slow-direction EMA (AdEMAMix) — three
   falsifiable predictions (all should underperform #1).

## 3. Plasticity literature and SOTA verification

**Verdict: 0.86245 is above every number published on the exact protocol.**
The on-protocol print ceiling is Weight Clipping (RLC 2024) at ~0.79–0.80;
UPGD itself ~0.75–0.78. No 2025–2026 paper re-ran the exact 1M-example
stream with a better result — the field moved to easier batched
permuted-MNIST variants (SNR 0.88 with batch-16/tiny nets — not
comparable), label-permuted streams (FADE 0.807 on L/P EMNIST 2000-task —
the one adjacent result that materially advanced SOTA), or RL. Caveat
stated honestly: "unbeaten" means nobody published on-protocol, not that
newer mechanisms were tried and lost.

**Novelty check on our conditioning claim — verified with two near-misses**
to cite defensively: stream-x uses running obs-normalization in streaming
RL (never isolated for supervised plasticity); CLeAN (2603.17548) does
EMA min-max input normalization on tabular cybersecurity streams (no
permutation protocol, no plasticity framing). Precise claim available: *no
prior work applies EMA input-statistics normalization as a plasticity
mechanism in the online permuted-input setting, and none reports it on
this protocol.*

Untried mechanisms ranked by expected gain on our protocol:
1. **FADE — meta-learned per-parameter weight decay** (2604.27063): learns
   the retention/forgetting rate per weight online via a forward-mode
   meta-gradient (`jax.jvp` one tangent pass). Utility-flavored like our
   gate but continuous; targets weight-norm drift, which input norm does
   not. Strongest new mechanism; stack on the champion.
2. **NaP-style fixed-norm projection + LayerNorm** (2407.01800): targets
   effective-learning-rate decay; orthogonal to input statistics.
3. **Weight clipping composed with EMA norm** — one line, the published
   on-protocol runner-up; plausibly additive.
4. **Spectral regularization** (2406.06811): top singular value ≈1 via one
   power iteration per step.
5. **C-CHAIN churn regularization** (ICML 2025) — needs a small reference
   buffer (protocol purity caveat).
6. **SNR firing-rate resets** (ICLR 2025): per-neuron silence-duration
   resets; fixes dead units, likely additive with conditioning.
7. Activation-shape swaps (smooth-leaky/randomized-leaky, AdaLin, AID).

## 4. Alberta Plan citations, OaK, and the Era of Experience

~32 works cite the Alberta Plan; almost none report benchmark numbers. The
substantive frame-setters:

- **Rethinking the Foundations for Continual RL** (Elelimy, Szepesvári,
  White, Bowling — 2504.08161, RLC 2025): four pillars of classical RL are
  *antithetical* to continual learning (MDP formalism, "final policy"
  artifacts, expected-sum evaluation, episodic benchmarks); proposes the
  **history process** as the formal object and **deviation regret** as the
  continual metric. Our evaluation lanes still mostly measure windowed
  accuracy/return — adopting deviation regret is a cheap, high-credibility
  upgrade.
- **OaK architecture** (Sutton, RLC/NeurIPS 2025 talks; no paper yet):
  every component learns continually; **every weight has its own
  meta-learned step-size** (online cross-validation, IDBD-descendant);
  open-ended abstraction via FC-STOMP (Feature → Subtask → Option → Model →
  Planning) with generate-and-test curation. Sutton's two named blockers:
  (1) reliable continual deep learning — *exactly the lane our campaign
  attacks*; (2) meta-learned feature generation (the FC front end, no
  accepted algorithm).
- **Era of Experience** (Silver & Sutton 2025): agent streams persisting
  months/years, grounded rewards, experiential planning. Critique line: the
  "missing reward" active-inference commentary (2508.05619) argues for
  intrinsic epistemic signals.
- **Permanence is utility-gated, not designated** — the convergent answer
  across OaK/Nature/SwiftTD to "what do we remember forever": validated
  knowledge freezes itself via tiny meta-learned step-sizes; GVF knowledge
  is *self-verifying* against the stream. This is our design law
  ("protection pays rent") stated as a research program.
- Alberta-native communication line: **Communicative Capital** (Pilarski,
  1711.03676) and **Pavlovian Signalling with GVFs** (2201.03709) — tokens
  derived from one agent's GVF predictions, learned online by a partner.
  **The literature hole: no published work runs a Lewis/signaling game
  between two never-resetting continual learners under drift.** Our
  hidden-partner and signaling streams are ~80% of the apparatus.
- Institutions executing the Plan: Openmind Research Institute (Sutton,
  Modayil, Bowling/Pilarski fellows); Keen Technologies' physical-Atari
  program (real-time, no resets, non-stationary).

Ranked capability gaps for this repo (full experiment sketches in fleet
transcript): (1) closed-loop FC-STOMP — discovered features *earning*
subtasks/options/models automatically, ≥50 unattended promotions; (2)
per-weight meta-learned step-sizes in deep nets (IDBD-through-backprop /
MetaOptimize on our IPMNIST harness — OaK's bet, untested in our stack, and
step-size histograms would verify "validated knowledge self-freezes"); (3)
autonomous GVF discovery/curation (what to predict); (4) deviation-regret
evaluation; (5) GVF-grounded emergent communication between two continual
learners (highest novelty per unit cost — open territory); (6) deep
off-policy stability at Horde scale; (7) grounded/epistemic reward; (8)
explicit fast/slow permanent-transient decomposition; (9) per-step
compute-bounded ranking (big-world discipline); (10) a 10^9-step soak run
(no published Alberta-line artifact exists at that horizon).

## 5. Left-field mechanisms

The survey's organizing observation: **on the standard protocol, permutations
never recur, so online accuracy measures F1 (adaptation speed) + F2 (dead
capacity) almost exclusively — F3 (overwrite) is unobservable by
construction.** Any memory-motivated substrate (MoE, BTSP, SDM, replay) must
be scored on the recurring lane (already in the harness) with a *paired*
prediction: ≈null on standard, positive on recurring. A paired prediction
that holds is stronger evidence for our design laws than any single number.

Two new candidate laws derived a priori:

- **Tracking-budget law**: for segment length T and p tracked conditioning
  parameters, useful adaptation needs N_eff ≳ p while N_eff ≪ T →
  conditioning quality is an inverted-U in p, peaking near p ~ T/10. With
  T=5000: optimum ≈ 500 parameters — more than diagonal (784), far less
  than full covariance (3×10⁵). The right whitening arm is **diagonal +
  rank-k**, not full ZCA.
- **Share what is permutation-invariant; re-estimate only the assignment**:
  under permutation the input-covariance eigen*values* are invariant — only
  eigenvectors permute. Lifetime-slow shared state that is invariant is NOT
  a forgetting channel; this motivates permutation-inference whitening
  (track fast per-index signatures, solve a Sinkhorn assignment to a
  lifetime catalogue, permute the high-quality lifetime whitener).

Top arms by (expected gain × novelty), full designs in the fleet transcript:

1. **W5 — random Fourier features + streaming RLS readout, no backprop**
   (predicted 0.88–0.92): structurally eliminates all three failure modes
   (RLS *is* the exponential-window optimum every step; no plasticity to
   lose). If it wins, the benchmark measures tracking, not learning — the
   logical endpoint of our own finding. Informative either way, cheapest.
2. **W3 — diagonal + rank-k whitening ladder** (k ∈ {0,4,16,64,256,full}):
   tests the tracking-budget law; yields a scaling law regardless of
   outcome. Bonus: eigenvalues tracked slow (invariant), eigenvectors fast.
3. **M1 — exclusive normalizer bank** (K EMA-normalizer slots, top-1 routed
   by match to fast statistics — the proven mechanism as its own router):
   two-sided test of "exclusive gating = memory" and "protection pays rent
   only where tasks recur"; routing-temperature sweep turns the gating law
   into a dose-response curve.
4. **W4 — permutation-inference whitening** (the crazy one; built-in
   assignment-accuracy diagnostic fails loudly).
5. **C3 — TTT self-supervised permutation adapter**: the learned label-free
   adversary to the whitening ladder; decides whether the exploitable input
   structure is second-order (closed-form wins) or nonlinear (TTT wins).

Mandatory adversarial control: **E2 gradient whitening (Muon)** — if
whitening the *gradient* matches whitening the *input*, our central claim
weakens to "conditioning matters, location incidental." Run before any
strong publication claim. (In flight as `muon_gate`.)

Null/law-test controls worth their cost: stream-x ObGD bounding (external
corroboration from RL that batch-1 pathology = input scaling), forward-
forward (predicted strictly worse), predictive coding (harness validation,
predicted ±0.005 of backprop), three-factor eligibility traces (predicted
null — no temporal credit to assign in i.i.d.-within-task streams),
feedback alignment (predicted monotone degradation — a law test),
BTSP one-shot memory head (recurring lane only; recovery within 1–5
samples or it's wrong), KAN splines (predicted no gain — tests "locality ⇒
no interference" against real input dimensionality).

## 6. Weaknesses of the current approach

Honest accounting, synthesized from all five surveys:

1. **The protocol cannot see forgetting.** Standard IPMNIST never repeats a
   permutation; our headline metric is blind to F3. Our design laws about
   memory rest on the gauntlet/recurring lanes, not the headline. Fix: run
   the paired recurring-lane program (M1/BTSP/CLS arms).
2. **The champion may be protocol-shaped.** Fast input-statistics tracking
   wins where nonstationarity is *input-side and abrupt*. Label-permuted
   streams (EMNIST lane, in flight) and slowly-changing regression are the
   dissociation tests; Continual ImageNet is the real-shift test. If the
   mechanism doesn't travel, the claim narrows to permutation-family
   benchmarks.
3. **Gradient-vs-input conditioning is unresolved** until `muon_gate` and
   the whitening ladder run (E2 adversarial control).
4. **W5 (RLS) is an existential control**: if a 1985 adaptive filter over
   random features beats every deep method, the benchmark story must be
   rewritten around tracking-vs-learning — better we discover this than a
   reviewer.
5. **Single-layer conditioning**: our normalizer conditions the input only;
   hidden-activation statistics drift uncorrected (colnorm arm addresses
   exactly this; the failed hidden_norm arm suggests naive hidden
   normalization is not the answer — the colnorm weight-space form may be).
6. **No meta-learned permanence.** Our gate uses a fixed utility decay;
   OaK's bet is per-weight step-sizes meta-learned by online cross-
   validation, and FADE's is per-weight learned decay. Both are one rung
   above our fixed-hyperparameter gate.
7. **Evaluation formalism**: windowed accuracy is the "atemporal artifact"
   Elelimy et al. warn about; deviation regret is the continual-native
   metric and cheap to add.
8. **Scale**: 300×150 MLP, MNIST pixels. Nothing yet says the laws survive
   convnets, transformers, or 10⁹-step horizons.

## 7. Derived experiment plan

Ordered by decisive-information-per-compute; standard lane = 60-task screen
(seeds 0-2, paired vs champion) with 200-task confirmation at >+0.002:

**Wave A — protect and extend the claim (cheap, this week):**
- A1 `muon_gate` (E2 adversarial control) — in flight.
- A2 `colnorm_gate` (conditioning extended to hidden layers) — in flight.
- A3 `lion_gate` (stateless-conditioning datapoint) — in flight.
- A4 stream-x ObGD bounding + sparse init on the champion (~free).
- A5 weight clipping composed with the champion (one line).
- A6 EMNIST transfer verdict + slowly-changing-regression dissociation
  (new artifact paths; pinned lanes stay immutable).

**Wave B — the whitening ladder and the RLS reframing:**
- B1 W5 RFF+RLS (m ∈ {512,1024,2048}, λ ∈ {0.999, 0.9999}).
- B2 W1/W2/W3 as one cascade (full ZCA / streaming decorrelation / rank-k
  ladder); read against the tracking-budget law.
- B3 W4 permutation-inference whitening (after B2 calibrates signatures).

**Wave C — memory lanes (paired standard/recurring predictions):**
- C1 M1 exclusive normalizer bank + routing-temperature dose-response.
- C2 BTSP head, CLS associative head (recurring lane).
- C3 FADE per-parameter meta weight decay stacked on the champion — the
  strongest new optimizer-level mechanism from the literature; also the
  bridge toward OaK meta-learned permanence.

**Wave D — mechanism-frontier and Alberta Plan gaps:**
- D1 IDBD-through-backprop / MetaOptimize per-weight step-sizes on the
  IPMNIST harness (OaK's bet; verify "validated knowledge self-freezes"
  via step-size histograms).
- D2 TTT permutation adapter vs the whitening ladder.
- D3 Deviation-regret metric added to evaluation CLIs; re-score one
  campaign under it (new artifact path).
- D4 Continual ImageNet port (highest-leverage external benchmark);
  label-permuted mini-ImageNet (completes the UPGD triptych).
- D5 GVF-grounded two-continual-learner signaling game (the open
  literature hole; hidden-partner substrates + Pavlovian signalling).
- D6 10⁹-step soak with plasticity/step-size telemetry (no published
  Alberta-line artifact at that horizon).

Null-control battery (run opportunistically, cheap): forward-forward,
predictive coding (harness validation), eligibility traces, feedback
alignment, KAN, APOLLO-Mini/Adam-mini granularity controls, AdEMAMix slow
direction EMA (all with pre-registered predicted-null/negative outcomes —
they convert the theory into evidence).

Success criteria: any arm beating 0.86245 at 200 tasks with all seeds
positive becomes the new champion; any predicted-null that turns positive
triggers a theory revision before further scaling; the paired
recurring-lane predictions are scored as law tests, not leaderboard
entries.
