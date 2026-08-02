# Red-team audit of the IPMNIST campaign claims (2026-08-02)

Adversarial self-audit of the screening/confirmation numbers behind the
campaign claims (champion `sigma0_ndecay099` 0.86245, `upgd_ema_norm` 0.85362
n=10, `sgd_ema_norm` 0.83991, `adamw_cbp_r3e4` 0.80126, baseline UPGD-W
reproduction 0.7791). Every check below was re-derived with fresh scripts
against the raw artifacts — no reuse of the runner's own accounting
(`merge_shards`/`validate_proxy` were bypassed). Everything here is
development-grade context; nothing changes any evidence-registry claim.

Verdict up front: **no result-invalidating bug was found.** The headline
numbers re-derive exactly, the proxy prefix property is bitwise clean within
the runner, schedules are correct and paired, and predict-before-update
ordering is correct in both runners. Four real defects were found and fixed
(all documentation/labeling), one measured harness artifact explains the
shipped `proxy_validated=false`, and one protocol-scope property of the EMA
normalizer is now explicitly characterized and disclosed.

---

## Findings (severity-ranked)

### F1 — MEDIUM (measured, explained, documented): the "bit-for-bit vs full
### lane" claim is false at protocol scale; `proxy_validation.json` ships
### `all_prefixes_match=false` for `upgd_w_control` with no explanation

- Shipped fact: `proxy_validation.json` reports the three `upgd_w_control`
  60-task screen shards do NOT prefix-match the 200-task
  `outputs/upgd_ipmnist/partials/` curves (max per-task diff 0.0084–0.0096),
  while all three `adamw_control` shards match to ~5e-9 (= the shards'
  8-decimal rounding). `FINAL_REPORT.md` previously embedded this JSON
  truncated mid-sentence with no diagnosis.
- Root cause, isolated by bisection (fresh harnesses replicating each runner,
  validated bitwise against pinned shards):
  - init params, task schedule, noise key, and the first 282,160-element
    noise draw are **bitwise equal** between `run_ipmnist` (vmap-batched)
    and `run_screening_config` (unbatched);
  - params are **bitwise equal after step 1**;
  - by step 10 the two XLA compilations of the *same* UPGD-W step function
    differ by 1-2 ulp (max 2.2e-8 on `w3`) — batched-vs-unbatched
    compilation reassociates float ops (FMA/fusion differences);
  - chaotic amplification over 5,000-step tasks turns this into ≤0.0096
    per-task accuracy jitter; the 60-task mean effect is ~+0.0004
    (screen 0.77776 vs partials prefix 0.77737).
  - AdamW consumes no noise and its update happens not to hit a divergent
    fusion, so it stays bitwise identical across harnesses over 300k steps —
    which simultaneously *proves* the two runners share data bytes,
    schedules, init, and forward/backward exactly.
- Also verified: today's `run_ipmnist` (step mode) reproduces the historical
  baseline partials **bitwise** (first 5 tasks, seed 0, max diff 0.0), i.e.
  the 0.7791 baseline is an exact-noise batched-harness run, and the
  environment is unchanged.
- Impact on claims: none material. All paired arm-vs-control deltas are
  computed within a single runner, where the artifact cancels. The
  cross-harness mean offset (~0.0004) is 20x smaller than the smallest
  confirmed arm delta cited (+0.0025 `adamw_cbp_r3e4` vs `adamw_cbp`) and
  200x smaller than the champion delta (+0.083). The two harnesses are the
  same protocol in distribution; a 1-ulp reassociation is not a protocol
  difference.
- Within-runner prefix property: **exact** — every 60-task shard checked
  (`sigma0_ndecay099`, `upgd_ema_norm_sigma0`, `upgd_ema_norm`,
  `sgd_ema_norm`, seeds 0-2) equals the first 60 entries of its 200-task
  `confirm_full/` shard with max diff 0.0.
- Fixed: `FINAL_REPORT.md` proxy paragraph rewritten with the diagnosis;
  `RUNBOOK.md` proxy section annotated. **Still to fix (not done here
  because a sibling session is actively editing the module):** the
  overstated docstrings in `alberta_framework/benchmarks/ipmnist_screening.py`
  ("validated bit-for-bit against the completed full-horizon shards",
  `run_screening_config` "reproduce the full-horizon lane bit-for-bit",
  `_wrap_grad_learner` likewise) and the CHANGELOG 0.28.0 phrase "control
  parity pinned bitwise" — the parity/prefix pins hold at the small test
  scale and within-runner, not cross-runner at protocol scale. Scope them to
  "bitwise within the screening runner; distribution-identical (1-2 ulp XLA
  divergence, chaos-amplified) vs the batched full lane".

### F2 — MEDIUM (fixed): `FINAL_REPORT.md` labeled all confirmations
### "pool64" — 14 of 16 arms actually ran exact step mode

- The `worker_confirm.sh` pool64 attempt falls back to `--noise-mode step`
  for any arm that declares no pool `noise_update`. Checking the recorded
  `noise_mode` field in every `confirm_full/` shard: only `upgd_w_control`
  (3 shards) and `upgd_w_wd0005` (10 shards) are pool64; `upgd_ema_norm`
  (n=10), `sigma0_ndecay099`, `sgd_ema_norm`, all `adamw_cbp*`, `upgd_idbd`,
  `upgd_l2init`, `upgd_ema_norm_sigma0`, `upgd_ema_norm_wd0005` are exact
  `step` shards. The mislabel was *against* us (exact mode is strictly
  closer to protocol than the pool approximation; identical for σ=0 arms).
- The champion section header "(pool64, seeds 0-2)" was likewise wrong.
- The "known pool-vs-exact delta -0.00012" constant is not reproducible from
  the shipped artifacts: pool64 confirm controls (0.778831 mean, seeds 0-2)
  vs the batched exact partials (0.779137) give **-0.0003**, and that number
  conflates pool noise with the F1 harness artifact. Corrected in
  `FINAL_REPORT.md` and `CONTINUAL_LEARNING_EVIDENCE.md`.

### F3 — MEDIUM (characterized, must stay disclosed): the EMA input
### normalizer folds the CURRENT example into the statistics used to
### normalize that same example, before the prediction is scored

- `ema_normalize` updates mean/var with `x_t` and then normalizes `x_t`
  with the *updated* statistics; the prediction counted by online accuracy
  is made on that output. Checked directly in the update loop
  (`ipmnist_screening.py::ema_normalize` + all `*_ema_norm*` factories).
- This is **not a label leak and not look-ahead**: the prediction at step t
  is a deterministic function of (x_1..x_t) and parameters trained on
  labels y_1..y_{t-1} only. x_t is legitimately available before
  predicting; "predict-before-update" in the protocol governs the
  label-consuming parameter update, which happens strictly after the
  scored prediction in every arm (verified: accuracy is computed from
  logits produced by `value_and_grad` on the PRE-update params in both
  `run_ipmnist` and `run_screening_config`; same for the ext/champion
  factory).
- Quantification (5 tasks, seed 0, fresh harness that first reproduced the
  pinned shards to 5e-9, then swapped in strictly-prior statistics —
  normalize with stats from steps < t, then update):
  - `sigma0_ndecay099`: as-shipped 0.8478 → strict-prior **0.1359**
    (collapse to chance after ~1 task);
  - `upgd_ema_norm` (decay 0.999): 0.8401 → **0.3522** (collapses one task
    later).
  - Reason: same-step inclusion makes the transform *self-bounding* — for a
    feature whose statistics the boundary shifts by delta, the output is
    capped at `decay*delta / sqrt(decay*(1-decay)*delta^2)` ≈
    `sqrt(decay/(1-decay))` (≈10 for decay 0.99, ≈32 for 0.999), whereas
    strictly-prior stats divide an O(1) numerator by a variance that has
    decayed toward the 1e-8 floor on quiet/dead pixels (MNIST borders),
    producing ~1e4 spikes that destroy the network at task boundaries.
- Consequence for claims: the conditioning mechanism is
  "current-inclusive EMA normalization"; the ordering is load-bearing, not
  cosmetic. Any reimplementation or reviewer re-derivation that uses
  prior-only statistics (or adds clipping) is a *different arm*. This is
  now stated here and referenced from README/EVIDENCE; keep it disclosed
  anywhere the +0.061 conditioning number is claimed.

### F4 — LOW (fixed): stale/wrong numbers in prose

- `CONTINUAL_LEARNING_THEORY.md` cited the full method at "0.85359" twice;
  the n=10 confirm re-derives to 0.853622 → cited value corrected to
  0.85362 (n=10). (0.85357 was the historical n=3 value; 0.85359 matched
  nothing.)
- `outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md` anchor table repeated
  0.85359 and mislabeled the 0.7791 row "baseline AdamW reproduction" —
  0.7791(5) is the **UPGD-W** reproduction; the AdamW baseline is 0.7190.
  Both fixed.
- `RUNBOOK.md` claimed "the screening runner has no pool mode" (stale since
  the confirmation waves) — fixed with the actual per-arm mode behavior.

### F5 — INFO (verified clean): schedule, pairing, and seed hygiene

- Permutations: for seeds 0/1/2 at 200 tasks, all 200 rows are valid
  permutations of 0..783, all 200 distinct within a seed, schedules
  distinct across seeds, derived only from the seed
  (`root=jr.key(seed) → split(3)[1]`), so every arm at the same seed sees
  the identical permutation/example stream by construction; the AdamW
  cross-harness bitwise match (F1) confirms schedule identity empirically.
  Example indices: in-range, no within-task duplicates.
- First task IS permuted (no identity row). Verified against upstream
  `mohmdelsayed/upgd@b75e90ad` `core/task/input_permuted_mnist.py`: its
  `__next__` calls `permute()` when `step % change_freq == 0`, i.e. at
  step 0, before the first sample — upstream's first task is permuted too.
  Our convention matches upstream (same as the label-EMNIST lane's
  documented `change_all_lables`-at-step-0 finding).
- Seed exposure: all screening (60-task) shards in `shards/` are seeds 0-2
  only (zero others on disk; all `jobs*.txt` screen files list 0-2 only).
  Seeds 3-9 exist exclusively as `confirm_full/` shards for three arms
  whose configurations were frozen *before* those runs (chronology from
  shard `created_unix`: `upgd_w_wd0005` 3-9 on 08-01 17:24-41,
  `adamw_cbp` 3-9 on 08-01 18:31-19:23, `upgd_ema_norm` 3-9 on 08-02
  00:53-04:04; the sigma0 frontier arms and their selection all ran on
  seeds 0-2). **No tuning decision consumed seeds 3-9.**
- The flip side, stated plainly: every tuned arm (`sigma0_ndecay099`
  champion, `adamw_cbp_r3e4`, `upgd_w_wd0005`) was selected on seeds 0-2
  and confirmed on the same seeds 0-2 (wd0005 additionally on 3-9). The
  champion's 0.86245 therefore carries selection bias of unquantified
  (likely small — screen-to-confirm deltas replicated within ±0.002)
  size. Seeds 3-9 remain unconsumed for `sigma0_ndecay099` and
  `adamw_cbp_r3e4` and are the correct instrument for a selection-free
  development estimate. Added this caveat to README and `FINAL_REPORT.md`.

### F6 — INFO (verified clean): independent re-derivation of every number

- Fresh accounting over all 69 `confirm_full/` shards (no
  `merge_shards`): per-arm means match `FINAL_REPORT.md` to ≤5e-6
  (report rounding); the three spot-checked per-seed lists
  (`sigma0_ndecay099`, `upgd_ema_norm` n=10, `adamw_cbp_r3e4`) match to
  ≤4e-6.
  - champion `sigma0_ndecay099` = 0.862451 (3 × 200-task step shards);
  - `upgd_ema_norm` = 0.853622 (10 shards); `sgd_ema_norm` = 0.839906;
  - `adamw_cbp_r3e4` = 0.801261; baseline partials = 0.779147 (10 seeds).
- Integrity: every `per_task_accuracy` entry in every shard is an exact
  multiple of 1/5000 (a mean of 5,000 binary outcomes) — no failures.
- Derived deltas re-verified: conditioning +0.061 (0.83991−0.77915),
  gate +0.011 (0.85051−0.83991), noise-with-norm +0.003 (0.85362−0.85051;
  +0.0031 on matched seeds 0-2), champion +0.0088 vs 0.85362, +0.083 vs
  0.7791, +0.0119 vs its σ=0 base.
- Metric semantics: online accuracy is the mean over tasks of per-task
  means over equal-length tasks — identical to the whole-stream mean;
  accuracy is scored on the pre-update prediction in every code path read
  (`_step_metrics` docstring is accurate).

---

## What was changed (this audit)

Doc corrections only (the screening module and tests are being actively
extended by a concurrent session and were deliberately left untouched;
F1 lists the docstring fixes still owed there):

- `outputs/ipmnist_screening/FINAL_REPORT.md` — proxy-validation diagnosis;
  "pool64" section/champion labels corrected to per-shard truth;
  pool-vs-exact delta corrected to the measured −0.0003 (with conflation
  note); champion selection-seed caveat.
- `outputs/ipmnist_screening/RUNBOOK.md` — cross-runner prefix caveat;
  pool-mode staleness fix; n=3→n=10 annotation.
- `CONTINUAL_LEARNING_THEORY.md` — 0.85359 → 0.85362 (n=10), twice.
- `CONTINUAL_LEARNING_EVIDENCE.md` — pool-vs-exact delta correction.
- `README.md` — selection-seed caveat under the confirmation table.
- `outputs/ipmnist_screening/SOTA_LANDSCAPE_2026.md` — 0.85362 fix;
  "baseline AdamW" → "baseline UPGD-W" label fix (AdamW baseline is
  0.7190).

Nothing under any pinned `outputs/` evidence directory was touched; the
evidence registry state (all five claims `invalid`, fail-closed by design)
is unchanged. All numbers in this file are development-grade
(`development_screening_diagnostic`) and nonpromoting.
