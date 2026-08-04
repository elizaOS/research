# Negative-results ledger

Purpose: the durable record of concluded negative, bounded, or
tried-and-not-pursued results, so future work does not re-try them and so
the code that produced them can be pruned without losing the conclusion.
Every entry names where the full record lives. Nothing here is promoted
evidence; scoping language follows the repository's evidence rules.

Rule: **before deleting any lane's code, its conclusion must have an entry
here** (or in a doc this file points to).

## Optimizer / update-rule lane (IPMNIST screening)

1. **Raw-gradient learning rates do not transfer to normalized, orthogonalized,
   or sign updates.** All three wave-A update-rule arms (`colnorm_gate`,
   `muon_gate`, `lion_gate`) scored chance at the champion's lr=0.01:
   normalized updates have unit-scale magnitudes, so colnorm random-walked to
   uniform logits and lion diverged by task 2. Calibrated rates (1e-3 / 3e-3 /
   1e-4) recovered all three to above-champion cold-start speed. Record:
   commit `a553ca7`, draft shards in
   `outputs/ipmnist_screening/shards_draft_updrule_lr001/`.
2. **RFF bandwidth mis-calibration is catastrophic, and the calibrated
   no-backprop control is a major result, not a negative.** Draft `rff_rls`
   with `gamma=0.05`, unclipped z-scores scored 0.177 (phase noise — commit
   `02314d1`, drafts in `shards_draft_rff_gamma005/`). Calibrated
   (`gamma=0.001`, clip ±3): **0.849 on the 60-task screen** — above every
   published deep method; the deep champion retains +0.013. The linear floor
   (`lin_rls`) is ~0.70–0.75. The negative that stands: near-zero-variance
   pixels + tiny normalizer epsilon produce extreme z-scores that must be
   clipped before any phase-based feature map.
3. **Perturbation noise is a crude substitute for input conditioning.**
   Load-bearing on raw inputs (−0.035 to remove), neutral under slow
   conditioning (+0.003), harmful under fast conditioning (−0.0019, all
   seeds — `ema_norm_ndecay099`). Record: `frontier2_results.json`,
   CONTINUAL_LEARNING_THEORY.md addendum.
4. **The normalizer-decay star is closed.** Optimum is a plateau at
   0.98–0.99; 0.95 −0.0035, 0.9 −0.0138, 0.999 −0.0119, 0.9999 −0.019.
   Hidden-layer RMS normalization actively hurts (−0.0186); epsilon floors,
   gate temperature, and local gate are flat. Record: `frontier_results.json`,
   `frontier2_results.json`.
5. **`guarded_cbp_adam` refuted its pre-registered prediction** (−0.0055):
   completing all three failure modes with zero coupling did not win —
   protection pays rent only where tasks recur. Record: theory doc outcome
   matrix.
6. **Composition is sub-additive: Adam's second moment IS input
   conditioning.** `adamw_cbp_ema_norm` ≈ `adamw_cbp` (normalization
   redundant under Adam); `upgd_ema_norm_wd0005` shows the tuning win and
   conditioning win are alternatives, not additive. Record: theory doc §1.
7. **IPMNIST ceiling analysis** (untracked
   `outputs/ipmnist_screening/CEILING_ANALYSIS.md`): champion family
   hard-capped ~0.933; architecture+online-regime ceiling ~0.974; realistic
   protocol-pure ceiling 0.90–0.93. 0.95 is outside every mechanism class
   currently in the arsenal.
8. **Red-team audit F1** (untracked `outputs/ipmnist_screening/AUDIT.md`):
   the "bit-for-bit vs full lane" proxy claim is false at protocol scale —
   batched vs unbatched XLA compilations of the identical UPGD-W step
   diverge 1–2 ulp by step 10, which chaos amplifies to ±0.01 per-task
   jitter. Paired within-runner comparisons cancel the artifact; the AdamW
   prefixes match bitwise. Explains shipped `proxy_validated=false`.

20. **Wave-A 60-task verdicts (all three update-rule arms lose to the
    champion; the adversarial control supports the input-statistics
    thesis).** With calibrated learning rates and the champion's
    conditioning held fixed: `muon_gate` (gradient orthogonalization —
    the pre-registered gradient-vs-input whitening adversarial control)
    0.8404, −0.021 vs the 0.8616 champion on every seed ⇒ conditioning
    location matters and the input side wins. `colnorm_gate` 0.7764
    (−0.085): per-column RMS scaling on top of an already-conditioned
    input is harmful at horizon despite winning the 2-task cold-start —
    early-transient speed does not predict 60-task rank. `lion_gate`
    0.6551 (−0.206): sign updates discard magnitude information this
    regime needs. Record: `outputs/ipmnist_screening/waveA_results.json`,
    shards in `shards/`.
26. **RLS forgetting factors (lambda < 1) covariance-wind-up and collapse on
    learned ReLU features — a failure mode the dense-feature `rff_rls`
    control could never show.** In the `rls_head_*` family (champion body +
    streaming-RLS readout on the 150-dim penultimate features), every
    lambda<1 arm without a P reset collapsed to chance mid-screen with the
    onset ordered exactly as the overflow theory predicts: P grows as
    `(1/lambda)^t` along unexcited (dead/quiet ReLU) directions, and float32
    overflows at `e^88.7`; measured onsets task 8–27 at lambda 0.995 and
    task 31–40 at lambda 0.999 (~88k steps), never at lambda 1.0. The
    collapsed shards carry non-finite per-task losses and are rejected by
    the merge validator by design (`rls_head_l0995`, `rls_head_l0999`,
    `rls_head_resid` in `shards/`). Wind-up-immune configurations survive:
    lambda=1 (P nonincreasing PSD) with detector-driven P resets, and the
    trace-cap probe `rls_head_l0999_pcap` (0.86608). Do not re-try plain
    exponential forgetting on sparse learned features.
27. **The residual-driven body is unstable exactly when its head is (and
    2-task diagnostics cannot see it).** `rls_head_resid` (body trained by
    the RLS head's own residual, ridge 1.0, lambda 0.999, no reset)
    collapsed by task 5–9 — before the head's own wind-up horizon: the
    body-chases-head feedback amplifies head error. The same residual
    design on the wind-up-immune head (`rls_head_resid_l1_preset005`,
    lambda 1.0 + P reset 0.05) is the family's best arm (0.86938 screen;
    200-task n=20 confirmation 0.87114 +/- 0.00010, paired +0.00665 vs
    the champion on the same seeds, all improve, drift-free —
    `confirm_rls_head/`, `summary_rls_head_confirm.json`) — the
    error-signal identity was never the problem; head stability was. Its 2-task diagnostic variant at
    lambda 0.995 collapsed to 0.105 on task 2 (fast-forgetting heads are
    unusable as body error signals even short-horizon).
28. **Small RLS ridge (large initial/post-reset gain) wins every short
    diagnostic and loses the horizon — again.** The 2-task ridge star was
    monotone toward small ridge (.8328/.8465/.8530/.8578/.8596 for
    1.0/0.3/0.1/0.03/0.01), but at 60 tasks ridge<=0.1 arms suffer
    seed-level partial collapses (`rls_head_l0999_preset005_r01` seed 1:
    0.711; `_r003` seed 1: 0.809; `_r001` all seeds below champion), while
    ridge 1.0 is stable. Same lesson as the wave-A colnorm entry:
    early-transient speed does not predict 60-task rank. Record:
    `outputs/ipmnist_screening/summary_rls_head.json`, shards in `shards/`.
29. **The convergence-shortfall attack via the readout ALONE is refuted at
    the plateau; the residual-trained body is what moves it.**
    Pre-registered question: does an exactly-optimal least-squares readout
    move the champion's within-task plateau (0.9037) toward the 0.933
    family asymptote? Measured (per-step instrumented 60-task runs, seed
    0): champion plateau 0.90420; the readout-only (parallel-body) arms
    `rls_head_l0999_preset005` 0.90443 and `rls_head_l1` 0.90195 — **no
    plateau movement**: at plateau the gated-SGD readout is already
    effectively optimal given the features, so swapping it for exact LS
    buys nothing. The refutation is scoped to the readout: the
    residual-driven winner `rls_head_resid_l1_preset005` measures plateau
    **0.91490 (+0.0107, 37% of the 0.029 shortfall)** with a uniform
    ~+0.011 lift across the 1000-5000-step buckets and a slightly WORSE
    0-100-step boundary window — training the body on the LS residual
    through the exactly-optimal head is a body-convergence mechanism, not
    a readout mechanism. The remaining shortfall to 0.933 is ~0.018.
30. **"Naive Bayes is transient-free" is task-level aliasing; a vote between
    two similarly-transient members has almost nothing to capture.** The NB
    tracker's flat per-task curve (~0.785 from t1) hides a genuine per-step
    post-shift transient: first-500 shifted-step accuracy 0.634 — WORSE
    than the champion's 0.678 (its class statistics are exactly as
    permutation-stale). Consequently the bare accuracy-EMA-weighted
    champion/NB vote (`nb_ensemble_champion`) gains only +0.00096, almost
    all of it task-0 warmup; and the linear-RLS third member
    (`nb_ensemble_rls3`) adds warmup only (+0.056 t1, +0.0001 shifted).
    What DOES pay is making the NB member shift-robust
    (`nb_ensemble_nbreset` detector-driven anneal-clock reset: 0.86671
    screen / 0.86678 confirm, every seed, every task block). Record:
    `outputs/ipmnist_screening/summary_nb_ensemble.json`,
    `outputs/ipmnist_screening/nb_ensemble/analysis_*.json`,
    CEILING_ANALYSIS.md "What number do we need".
31. **Ensembling cannot create post-shift accuracy no member has: the
    per-example two-member oracle is 0.8975.** Selecting whichever of
    {champion, NB} is right at EVERY step (an oracle no online vote can
    beat) reaches only 0.8975 on shifted tasks — 0.43/0.65/0.76 in the
    0-50/50-100/100-250 buckets — because in the first ~100 post-shift
    steps every member's sufficient statistics are permuted. The realized
    nbreset ensemble captured 5.3% of the champion's 0.0366 residual
    transient; the untouched 95% lives below step 250 and requires state
    that survives the permutation (recurrence/context reuse or
    higher-order identification; first-order identification is bounded at
    ~2,000 samples, NEW_DIRECTIONS V1). Record: per-step traces + vote
    traces in `outputs/ipmnist_screening/nb_ensemble/`.

## Evidence / campaign closures

9. **Continual-IA v1 is a valid rejection** at the frozen 10% gate: reward
   uplift and both augmentation controls passed; action-changing
   intervention prevalence missed. Consumed-seed replays are nonpromoting.
   Record: RESEARCH_STATUS.md Step 12, `outputs/continual_ia/` (pinned).
10. **Kondo compute-savings is a permanently excluded claim**:
    `KONDO_IMPLEMENTED = False` in
    `benchmarks/delightful_policy_gradient_development.py` — post-hoc
    counterfactual accounting cannot demonstrate skipped backward work.
11. **Forager matched v1 is immutable and source-incompatible** with the
    current tree; the v2 digest is offline-compatibility-only and will not
    be newly qualified; the old selected 30-seed evaluation produced no
    batch or report. Record: FORAGER_COMPARATOR_AUDIT.md.
12. **Alberta candidate audit: implementation GO, campaign authority NOT
    CLEARED** (total persistent memory and compute not closed). Record:
    FORAGER_ALBERTA_CANDIDATE_AUDIT.md.
13. **RTU Taylor correction is a derivation, not a port** — parameter-wise
    diagonal only, not exact RTRL under moving parameters; disabled by
    default. Record: RTU_TAYLOR_CORRECTION.md.
14. **OPMNIST published-scale ingestion never received data**: upstream
    3-seed × 48M-update run produced no merged result; the 2,864-line
    ingestion surface is a contract awaiting data that never arrived.
    Record: OPMNIST_DEVELOPMENT_INGESTION.md. (The separate in-repo
    800-task 3-seed closure DID complete — `outputs/step2_canonical/`.)
15. **Evidence registry all-invalid (2026-08-01) is working-as-designed**:
    registered source files were edited after artifacts were pinned; a
    dirty worktree alone forces `invalid`. The pinned artifacts remain
    valid historical records that certify no current tree.
16. **`slowly_changing_regression_v2` is not an exact Dohare et al. (2024)
    replication** — comparator arm selected, extensions local; permanently
    nonpromoting schema.
17. **`forager_rtu_ppo_rng_isolation` concluded**: RNG-coupling isolation
    probe finished its purpose; conclusion absorbed into
    FORAGER_BENCHMARK.md.
18. **`forager_causal_grid_divergence_probe.py` is built but unrun**
    (RESEARCH_STATUS.md) — an unrun harness with a 1,922-line test, kept as
    a contract, not a result.
22. **The first compositional contribution-future-utility panel rejected both
    enabled endpoints.** On its sole consumed development root, the disabled
    internal comparator retained A and reached executed reward `0.274283`;
    mix-one/decay-zero and mix-one/decay-0.95 retained no A/B/C target and
    reached `-0.003112` and `-0.020449`. This closes those two exact settings;
    neither may be relabeled as selective retention or rerun. Record:
    RESEARCH_STATUS.md Step 2,
    `alberta_framework/evaluation/compositional_future_utility_development.py`.
23. **Compositional future-utility calibration v2 produced no learner
    result.** Its sole root was consumed after the first arm's compiled scan,
    when an invalid evaluator assertion treated an all-step margin diagnostic
    as cadence-only. No arm record, report, endpoint, winner, default, or
    artifact exists, and no retry or recovery is authorized. A repaired
    evaluator on a new root cannot reconstruct or reinterpret v2. Record:
    `compositional_future_utility_calibration_v2_execution_outcome.py`,
    CONTINUAL_LEARNING_EVIDENCE.md.
24. **Compositional future-utility calibration v3 produced no learner
    result.** The sole fresh development root `0x12EFD48B` completed all five
    scans, then failed closed while serializing arm 0 because the report gate
    asserted that direct candidate admissions must be at least the number of
    structural acquisition episodes. Those quantities have different
    semantics: direct admissions count selected-candidate promotions, while
    structural episodes also admit absent-to-present cascade refills, so the
    valid relation is direct admissions no greater than total episodes. The
    terminal is permanently `failed` with `panel_completed=true`,
    `report_sha256=null`, failure receipt
    `5150a4aa08ba3d17b644ae1ed0357d1c6359123e96298ac3ca2cd0d03bff894d`,
    and no retry, recovery, endpoint extraction, winner selection, evidence,
    or promotion authority. A corrected schema/validator requires a new
    namespace and root and cannot reconstruct or reinterpret v3. Record:
    `outputs/compositional_future_utility_calibration_v3/one_shot_ledger/`.
25. **The repeated Prototype option-lifecycle v1 development schedule is
    blocked at cycle-1 candidate refresh, not a benefit result.** With control
    caps `(1, 5)`, cycle 0 completed in two replacement attempts. Cycle 1 then
    exhausted both the former harness-local five-attempt bound (890.66
    seconds) and, in the finalized regression, all eight attempts mechanically
    inherited from the pre-existing scheduler `max_install_attempts=8`
    (1,153.13 test seconds; 1,155.72 wall-clock seconds; 5,790,940 KiB peak
    RSS). Eight was declared before candidate outcomes and is not a tuned
    threshold. All eight proposals were due and ready, but each reselected
    candidate indices `(1, 3, 4, 5)` and the same descriptors while
    `changed_slots` stayed all false against installed mask
    `(true, false, true, true)`. Thus the exact-one-cold semantic-change gate
    rejected every candidate and the scheduler installation-attempt counter
    stayed `(0, 2)`: the cause is repeated incumbent/no fresh eligible
    semantic, not live-slot drift or resource exhaustion. The typed failure
    preserves those diagnostics, a valid unchanged source state, completed
    cycle 0, and checkpoint suffix `not_assessed`. No report, endpoint, parity
    result, comparator, winner, option-benefit inference, or promotion claim
    exists. Do not retry by changing this consumed v1 schedule, rotating its
    seeds, injecting state, or raising a bound. The opt-in
    `AuthorizedOptionAtomicSwapController` v2 mechanism prevents a transient
    retirement from persisting a cold slot unless the same transaction has a
    fresh, exact replacement: no-fresh, decline, tamper, and replay are exact
    all-installed source no-ops. It does not create, rotate, or widen the
    candidate universe, so it does not resolve the repeated-incumbent/no-fresh-
    semantic cause. A separate stateless v2
    `FreshColdSlotCumulantCohortFilter` now proves that an explicitly versioned
    universe containing one additional eligible feature descriptor can yield
    a same-family fresh replacement for only the cold slot, while the original
    universe remains unavailable and cross-family/tampered cohorts are
    rejected. The filter has no install, adoption, or authority seam, so this
    does not change the negative outcome. The consumed v1 harness has not been
    rerun or fixed.
    Record:
    `alberta_framework/evaluation/prototype_repeated_option_lifecycle_development.py`
    and `CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md` section 7.3.

## EMNIST transfer lane

19. **Bare conditioning collapses under label permutation**: `sgd_ema_norm`
    0.5037 vs `upgd_w` 0.6715 on L/P EMNIST — the utility gate is
    load-bearing where labels permute; input conditioning does not fix
    output-side non-stationarity. (Direction confirmed as pre-registered;
    magnitude partially missed the 0.2–0.4 collapse band — recorded
    honestly.)
21. **The conditioning-equivalence prediction was REFUTED (v2 merge,
    `results.v2.json`)**: pre-registered |`upgd_ema_norm` − 0.6715| ≤ 0.02
    failed — measured 0.7162, +0.045 above the raw-input baseline under
    label permutation with stationary inputs. EMA input conditioning is a
    *general* stream-optimization conditioner, not only an
    input-nonstationarity fix; the tracking-speed component (decay 0.99
    star) is the part specific to input shift. `upgd_ema_norm_sigma0`
    0.7155 — perturbation again contributes nothing under conditioning.
