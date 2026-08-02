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
