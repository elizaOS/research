# SOTA Landscape 2026 — online continual supervised learning around IPMNIST

Compiled 2026-08-02 (web survey; arXiv/OpenReview-centric, abstracts + HTML fetches).
Scope: what happened in 2025–2026 *beyond* the papers we already covered
(UPGD/ICLR-24, Weight-Clipping/RLC-24, FADE, SNR, CBP/Nature-24, stream-x, Lyle NaP).
Anchor: the ICLR-2024 input-permuted-MNIST protocol — 1M examples, 200 tasks x 5000
steps, one example per step, 300x150 ReLU MLP, **whole-stream online accuracy**, no
pretraining, no replay (Elsayed & Mahmood 2024, arXiv:2404.00781).

## 0. Our anchor numbers (outputs/ipmnist_screening/confirm_full/, FINAL_REPORT.md)

| Arm | Online acc |
|---|---|
| l2init_ema_norm (comparison wave, 2026-08-02: norm decay 0.99 + SGD + decay toward init, no gate) | 0.86457 |
| sigma0_ndecay099 (champion: norm decay 0.99, sigma=0) | 0.86245 |
| upgd_ema_norm (n=10) | 0.85362 |
| sgd_ema_norm | 0.83991 |
| adamw_cbp_r3e4 (protocol-pure best) | 0.80126 |
| baseline UPGD-W reproduction (published SOTA config; AdamW baseline is 0.7190) | 0.7791 |

Decomposition: conditioning +0.061 (mechanism = input-statistics tracking speed),
gate +0.011, noise +0.003 with norm / −0.035 without (CONTINUAL_LEARNING_THEORY.md).

**Headline of this survey:** we found **no 2025–2026 published number on the exact
ICLR-24 IPMNIST protocol (1M examples, 300x150 MLP, whole-stream online accuracy)
that exceeds 0.86245** (the survey-date champion; the campaign best has since
moved to 0.86457/0.86459 — `l2init_ema_norm` / `sigma0_shiftnorm_d099`, both
2026-08-02, still development-grade), and in fact no 2025–2026 paper re-running that exact protocol
with absolute numbers at all — the field has moved to (a) degradation-curve/plasticity
diagnostics on smaller PMNIST variants, (b) RL plasticity, and (c) pretrained-encoder
class-incremental benchmarks. The nearest numerically-larger claim (BiMU, 90.3%) is on
a materially different protocol (see §1.1).

---

## 1. Ledger of 2025–2026 claims (permuted-MNIST-family and plasticity methods)

### 1.1 Numbers on permuted-MNIST-family protocols

| Paper | Protocol | Metric | Number | Comparable to ours? |
|---|---|---|---|---|
| **BiMU** — Active CL with Metaplastic Binary Bayesian NNs (arXiv:2605.30198, May 2026) | 1000-task PMNIST, **binary-weight MLP, 1 hidden layer of 100 units**, 1 epoch/task, batch 11, no replay, no task boundaries | **mean accuracy over the LAST 5 tasks** (late-stream capability, not whole-stream online acc) | **90.30 ± 0.38%** (baselines collapse: BayesBiNN 41.1%, Synaptic Metaplasticity 10.3%, STE 29.4%) | **No.** Different metric (last-5-task vs whole-stream average), different data budget (~60k examples/task vs 5000), batch 11 vs 1, binary weights. Last-task-window accuracy is systematically higher than whole-stream online accuracy (our champion's late-task within-task accuracy also exceeds its 0.862 stream average). Worth citing as the current headline "big number" on a PMNIST stream and why it is not our protocol. |
| **AdaLin** — Adaptive Linearity Injection (arXiv:2505.09486, May 2025) | 400-task PMNIST, 10k images/task, 2-layer MLP of 100 units, 1 epoch/task, batch 16 | average online train accuracy per task (degradation curves; figure-level, no headline scalar) | maintains plasticity where ReLU baseline degrades; beats CReLU, deep Fourier, L2, S&P | **Partially.** Same metric family (online accuracy) but Lewandowski-style 10k/task batch-16 protocol, no UPGD/CBP/LayerNorm comparison. No absolute number >0.86-class. |
| **CCBP** — Continuous Continual Backpropagation (OpenReview UJqXhFFzKu) | long sequences of distribution shifts (supervised + continual RL) | degradation curves | outperforms decay-based (L2, S&P) and reset-based (CBP, ReDO) methods after long shift sequences; uniquely prevents policy collapse | Method-level: **continuous utility-scaled partial resets of all hidden parameters** instead of CBP's periodic full reinit of low-utility units. No IPMNIST absolute number. |
| **Calibrated Partial Resets** (arXiv:2607.24996, Jul 2026) | continual RL focus | — | utility-scaled pull of low-utility neurons toward init; avoids binary-reset brittleness and uniform-decay bluntness | Sibling of CCBP; confirms the field converging on **soft/graded resets** — mechanistically adjacent to our gate (+0.011) and to (shrink-and-)perturb. |
| **Experience Replay Addresses Loss of Plasticity** (Wang, Chandra, Zhang, arXiv:2503.20018, Mar 2025) | regression/classification/policy-eval streams incl. permuted-input settings, replay + Transformer | loss-of-plasticity disappearance | claims LoP vanishes with replay+in-context learning; no protocol-pure MLP numbers | **No** (uses replay + Transformer; our protocol forbids replay). Useful as a contrarian control citation. |
| **Self-Normalized Resets** (SNR) — now **ICLR 2025** (arXiv:2410.20098 v3; OpenReview G82uQztzxl) | PMNIST-family + others | online-style accuracy, degradation | robust to its threshold hyperparameter where competitors are sensitive | Already in our covered set; note final venue = ICLR 2025. |
| **Activation by Interval-wise Dropout** (arXiv:2502.01342, Feb 2025) | PMNIST-family plasticity benchmarks | degradation | dropout-like intervention prevents plasticity loss | Diagnostic-level, no comparable absolute number. |
| **Activation Function Design Sustains Plasticity** (arXiv:2509.22562, Sep 2025) | continual supervised benchmarks | degradation | activation choice alone mitigates LoP | Same family as AdaLin / Deep Fourier Features (arXiv:2410.20634). |

### 1.2 New mechanisms/optimizers (2025–2026), mostly without IPMNIST numbers

| Paper | Mechanism class | One-line claim |
|---|---|---|
| **Spectral Collapse Drives Loss of Plasticity** (arXiv:2509.22335, v. May 2026; reported as ICML 2026) | **curvature/conditioning** | LoP = Hessian spectral collapse at new-task init (meaningful curvature directions vanish); fix = effective-feature-rank regularizer + L2 ("L2-ER"), motivated by K-FAC Hessian analysis; proves loss-weighted Gram matrix spectrally equivalent to GGN. Benchmarks incl. PMNIST (early performance boost + preserved plasticity). |
| **A Unified Noise-Curvature View of Loss of Trainability** (Baveja, Lewandowski, Schmidt, arXiv:2509.19698, Sep 2025) | **curvature-aware step-size** | single metrics (Hessian rank, sharpness, grad norm) don't predict trainability loss; batch-size-aware gradient-noise bound + curvature-volatility bound → **adaptive per-layer effective step-size scheduler** whose auto trajectories "mirror manually engineered step-size decay schedules"; beats CReLU, Wasserstein reg, L2. |
| **Preserving Plasticity via Dynamical Isometry / AdamO** (Rosseau, Müller, Nowé, arXiv:2606.09762, Jun 2026) | **Jacobian conditioning** | plasticity preserved by keeping layer-wise Jacobian singular values ≈ 1 (empirical-NTK argument); isometry regularization **decoupled from gradient updates ("AdamO", analogous to AdamW)**; matches/beats existing approaches on supervised+RL plasticity benchmarks. |
| **Predicting Plasticity: optimization readiness** (arXiv:2605.09044, May 2026) | **diagnostic** | "optimization readiness" (gradient strength x reliability) lower-bounds one-step gain; ranks checkpoint trainability better than prior diagnostics on Slowly-Changing Regression + PMNIST. |
| **Do NNs Lose Plasticity in a Gradually Changing World?** (Liu & Mou, arXiv:2602.09234, Feb 2026) | **problem framing** | LoP severity is tied to *abruptness* of task transitions; gradual input/output interpolation substantially reduces it. Relevant to us: IPMNIST's abrupt permutation switches are the hard case; our input-statistics-tracking-speed mechanism is exactly a fast-adaptation-to-abrupt-shift story. |
| **Barriers for Learning in an Evolving World** (arXiv:2510.00304, Oct 2025) | **theory** | mathematical treatment of LoP; connects to ill-conditioned Hessians from parameter-norm growth. |
| **CBPNet** (arXiv:2509.15785, Sep 2025) | resets on edge | continual-backprop prompt network for edge devices. |
| **Muon-OGD** (arXiv:2605.08949, May 2026) | **Muon/second-order-lite, continual** | Muon-style spectral-norm update geometry + orthogonal projection away from prior-task directions, Newton–Schulz matrix sign; for **LLM continual fine-tuning**, not streams; beats Frobenius-projection OGD baselines. FOGO (arXiv:2606.10406) similar. No PMNIST/streaming numbers — Muon-class optimizers have **not** yet been evaluated on IPMNIST-style online streams (open arm for us). |
| **FLAD** — flatness decomposition for CL (arXiv:2601.07636, Jan 2026) | **sharpness-aware** | decomposes SAM perturbation into gradient-aligned + stochastic-noise components for efficient continual learning; batch CL benchmarks, not online streams. |
| **Can Scale Save Us From Plasticity Loss in LLMs?** (arXiv:2606.24752, Jun 2026) | scale study | plasticity loss persists at LLM scale (per title/abstract); shows the problem is not washed out by scale. |
| **Plasticity Loss in Deep RL: A Survey** (Klein et al., arXiv:2411.04832, **v3 Apr 2026**) | survey | 50+ mitigations taxonomized; "general regularization techniques often outperform domain-specific interventions" — convergent with our conditioning-dominates reading. |
| **Lyle, "The state of plasticity in 2025"** (clarelyle.com blog, Sep 2025) | field summary | "layer normalization and a well-tuned optimizer are almost always enough to avoid catastrophic network pathologies"; neuron-level resets ≈ free extra; **warns naive *input* normalization can erase task-relevant information in robotics**. |

---

## 2. Freshness verdict on our novelty claim (conditioning dominates; mechanism = input-statistics tracking speed)

**Verdict: still fresh as of 2026-08-02.** Nobody we could find has published the
specific finding that, on an online permuted-input stream, (a) the bulk of the
achievable gain (+0.061 of our +0.083) comes from *conditioning*, (b) the operative
mechanism is the *speed of tracking input statistics* after an abrupt shift, and
(c) gate (+0.011) and noise (+0.003/−0.035) are second-order by comparison.

But the field is **converging on conditioning from the weight/Hessian side**, so the
window is narrowing. Nearest neighbors, all 2025–2026, all weight-side:

1. **Spectral collapse (2509.22335, ICML 2026)** — "LoP *is* Hessian conditioning
   collapse." Closest conceptual rival; their evidence is Hessian-spectrum
   diagnostics + rank regularizers, not an input-side normalization decomposition,
   and not whole-stream online accuracy accounting.
2. **Unified noise-curvature (2509.19698)** — trainability = f(gradient noise,
   curvature); an *optimizer-side* conditioning story with an adaptive step-size
   scheduler. Their noise/curvature split is a cousin of our conditioning/noise
   decomposition.
3. **Dynamical isometry / AdamO (2606.09762)** — conditioning of layer Jacobians as
   the preservation mechanism.
4. **NaP (Lyle et al., NeurIPS 2024, arXiv:2407.01800)** — already known to us;
   still the canonical *normalization = effective-learning-rate control* citation;
   no 2025–2026 successor paper from that group supersedes it (blog post only).
5. **Lyle 2025 blog** — asserts normalization + tuned optimizer near-sufficiency
   (consistent with us) but offers no decomposition and explicitly flags input
   normalization as risky in RL/robotics — i.e., the *input-statistics-tracking*
   framing is not claimed there.

**Positioning advice:** frame our result as the *supervised-stream, input-side,
accounting-complete* counterpart of the spectral-collapse and noise-curvature
papers: they infer conditioning from Hessian spectra; we *measure* the online-accuracy
value of conditioning directly and locate it in input-statistics tracking speed.
Cite 2509.22335, 2509.19698, 2606.09762, 2407.01800, and the survey v3 as the
convergent-evidence cluster. Risk to monitor: an ICML/NeurIPS 2026 camera-ready
adding an input-normalization ablation to any of these would erode priority — the
spectral-collapse paper's May 2026 revision is the one to watch.

---

## 3. Leaderboard reality: why nobody serious reports 0.95 on this protocol

- **Pretrained encoders trivialize MNIST-class continual learning.** The dominant
  2024–2026 online-CL literature (RanPAC arXiv:2307.02251/NeurIPS-23; NSCE; RanDumb
  arXiv:2402.08823/NeurIPS-24; PROL arXiv:2507.12305; prompt/PTM families on Mammoth
  and Avalanche) sits on frozen pretrained ViT/foundation features, where a streaming
  linear/prototype head gets 0.95+ on MNIST-family and near-joint accuracy on much
  harder sets. **Our protocol forbids pretraining, so those numbers are
  categorically incomparable** — that is the honest answer to "why not 0.95".
- **RanDumb is the sharpest indictment:** a *fixed random Fourier projection* + a
  streaming Mahalanobis/NCM linear classifier **beats learned online-CL
  representations across standard online-CL benchmarks and nearly matches joint
  training** — i.e., most online-CL leaderboard gains never came from online
  representation learning at all. Corollary for us: the from-scratch,
  representation-learning-under-nonstationarity setting (ICLR-24 IPMNIST) is the
  setting where representation learning is actually exercised; whole-stream online
  accuracy there has no known free lunch.
- **Class-incremental leaderboards (CLEAR, Mammoth, Avalanche stacks) measure a
  different axis** — forgetting/retention with task or class semantics, usually with
  replay buffers and multi-epoch task visits — not single-pass next-example online
  accuracy under input re-permutation. Numbers do not transfer in either direction.
- Within-protocol context: plain AdamW already reaches ~0.77 whole-stream online
  accuracy (reaching ~0.77 within each 5000-step task); the entire contested range
  on this protocol is roughly 0.78–0.87, and our 0.86245 sits at the top of every
  number we can find for it.

---

## 4. Alberta / Mahmood / Sutton adjacent work, 2025–2026

| Work | Venue | Relevance |
|---|---|---|
| **Intentional Updates for Streaming RL** — Sharifnassab, Elsayed, De Asis, Mahmood, Sutton (arXiv:2604.19033) | **ICML 2026** | Step-size control by *specifying intended outcomes* (fixed fractional TD-error reduction; policy-gradient bounds) instead of tuning raw step sizes; claims SOTA streaming performance ≈ batch/replay. SwiftTD-lineage; the supervised analogue (fixed fractional per-example loss reduction) is directly testable on IPMNIST. |
| **Swift-Sarsa** — Javed et al. | RLDM 2025 | Linear control extension of SwiftTD (RLC-24 best paper, already known). |
| **Javed PhD thesis, "Real-time RL for Achieving Goals in Big Worlds"** (Jan 2025, incompleteideas.net) | thesis | Consolidates SwiftTD/Swift-Sarsa + big-world hypothesis; no new supervised-stream numbers. |
| **Multi-stream Sequence Learning** — Elsayed & Mahmood | ICML 2025 Workshop (ES-FoMo) | Preserve natural continuity of streams rather than shuffling — protocol-philosophy support for online-accuracy evaluation. |
| **Extending Differential TD (reward centering) for episodic problems** — De Asis, He, Elsayed | RLC 2026 | Average-reward/centering line; not supervised. |
| **Farrahi & Mahmood, learning without time-based resets in SAC** | CoLLAs 2025 | Continual RL without resets. |
| **The World Is Bigger (computationally-embedded big-world)** (arXiv:2512.23419) | Dec 2025 | Big-world hypothesis formalization. |

None of these publish IPMNIST supervised numbers; UPGD (2404.00781) remains the
group's reference point on this protocol, which our 0.85362 (n=10)
protocol-extended `upgd_ema_norm` and 0.86245 champion both exceed
(development-grade, seeds shared with selection — see AUDIT.md).

---

## 5. Three strongest ideas worth importing

1. **Continuous utility-scaled soft resets (CCBP / Calibrated Partial Resets).**
   Replace our binary/periodic gate with a *continuous* partial pull of every hidden
   parameter toward init, strength ∝ (1 − utility). Two 2026 papers independently
   report it dominates both decay-based (L2/S&P) and hard-reset (CBP/ReDO) methods
   at long horizons. Cheap, protocol-pure, and slots exactly into the +0.011 gate
   slot — plausible upgrade on top of sigma0_ndecay099. (OpenReview UJqXhFFzKu;
   arXiv:2607.24996.)
2. **Noise/curvature-bounded per-layer adaptive step-size scheduling**
   (arXiv:2509.19698). Their auto-derived per-layer effective step-size trajectories
   reproduce hand-tuned decay schedules with no tuning. Our champion *is* a
   hand-tuned norm-decay (0.99) arm, and a sibling session is sweeping fast-decay
   variants (0.9/0.95/0.98) by hand — this is the principled replacement: derive the
   decay from the gradient-noise + curvature-volatility bound instead of sweeping
   it. Also the natural rebuttal-proof story for *why* norm decay 0.99 wins.
3. **Gradient-decoupled isometry/spectral regularization (AdamO-style / L2-ER)**
   (arXiv:2606.09762; arXiv:2509.22335). Weight-side conditioning control that is
   orthogonal to our input-side EMA normalization. One decisive experiment: add
   isometry/effective-rank regularization to (a) baseline AdamW and (b) our
   champion. If (a) improves but (b) doesn't, weight-side and input-side
   conditioning are redundant → strengthens "conditioning dominates, input-statistics
   tracking is the operative lever". If additive, we gain a new arm. Either outcome
   is publishable ammunition.

Honorable mentions: uncertainty-modulated per-parameter step sizes from BiMU
(2605.30198) as a Bayesian gate alternative; Muon-geometry updates (2605.08949) as an
untested optimizer class on IPMNIST streams; replay+Transformer in-context result
(2503.20018) as the contrarian control to cite when defending the no-replay protocol.

---

## Sources (primary)

- Elsayed & Mahmood, UPGD, ICLR 2024 — https://arxiv.org/abs/2404.00781
- BiMU, Metaplastic Binary Bayesian NNs — https://arxiv.org/abs/2605.30198
- AdaLin — https://arxiv.org/abs/2505.09486 (html v1)
- CCBP — https://openreview.net/forum?id=UJqXhFFzKu
- Calibrated Partial Resets — https://arxiv.org/abs/2607.24996
- SNR (ICLR 2025) — https://arxiv.org/abs/2410.20098 ; https://openreview.net/forum?id=G82uQztzxl
- Spectral Collapse Drives Loss of Plasticity — https://arxiv.org/abs/2509.22335
- Unified Noise-Curvature View — https://arxiv.org/abs/2509.19698
- Dynamical Isometry / AdamO — https://arxiv.org/abs/2606.09762
- Optimization readiness — https://arxiv.org/abs/2605.09044
- Gradually Changing World — https://arxiv.org/abs/2602.09234
- Barriers for Learning in an Evolving World — https://arxiv.org/abs/2510.00304
- Experience Replay Addresses LoP — https://arxiv.org/abs/2503.20018
- Interval-wise Dropout — https://arxiv.org/abs/2502.01342
- Activation Function Design — https://arxiv.org/abs/2509.22562
- Deep Fourier Features — https://arxiv.org/abs/2410.20634
- CBPNet — https://arxiv.org/abs/2509.15785
- Muon-OGD — https://arxiv.org/abs/2605.08949 ; FOGO — https://arxiv.org/abs/2606.10406
- FLAD — https://arxiv.org/abs/2601.07636
- Scale vs plasticity in LLMs — https://arxiv.org/abs/2606.24752
- Plasticity-loss survey (v3 2026) — https://arxiv.org/abs/2411.04832
- NaP — https://arxiv.org/abs/2407.01800
- Lyle, state of plasticity 2025 — https://clarelyle.com/posts/2025-09-06-plasticity-survey.html
- RanDumb — https://arxiv.org/abs/2402.08823 ; RanPAC — https://arxiv.org/abs/2307.02251
- PROL — https://arxiv.org/abs/2507.12305
- Intentional Updates for Streaming RL — https://arxiv.org/abs/2604.19033
- Elsayed publications — https://mohmdelsayed.github.io/publications/ ; Mahmood — https://armahmood.github.io/publications/
- Javed thesis — http://incompleteideas.net/papers/javed_khurram_202501_phd.pdf
- Big-world follow-up — https://arxiv.org/abs/2512.23419

Caveats: claims taken from abstracts/HTML fetches, not full-PDF audits; AdaLin and
CCBP numbers are figure-level (no scalar tables); BiMU per-task example count
inferred from "one epoch per task"; venue attributions ("ICML 2026" for 2509.22335,
2604.19033) come from search snippets/author pages, not proceedings pages.
