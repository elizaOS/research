# IPMNIST mechanism-combination screening — runbook

Development screening diagnostic (never promotable evidence) hunting for
configurations that beat the reproduced UPGD-W SOTA on the ICLR-2024 online
Input-permuted MNIST protocol. Code:
`alberta_framework/benchmarks/ipmnist_screening.py`
(tests: `tests/test_ipmnist_screening.py`).

## Proxy

Same protocol as `alberta_framework/benchmarks/upgd_ipmnist.py` at a reduced
horizon: **60 tasks x 5,000 steps** (vs 200 tasks). Because `build_schedule`
folds the task index into per-seed keys, a 60-task run is an **exact prefix**
of the 200-task run for the same seed. Validation therefore checks
bit-level prefix agreement of the control shards against the completed
full-horizon shards in `outputs/upgd_ipmnist/partials/` (read-only), plus
ordering preservation (UPGD-W > AdamW). From the completed 10-seed full runs,
at 60 tasks: UPGD-W 0.7777 vs AdamW 0.7553 (paired +0.0224, all 10 seeds
ordered; per-seed stderr ~0.0004). Do not shrink task_length below 5000.

## Arms (registry names)

- `upgd_w_control` / `adamw_control` — published configs (control + ordering).
- `upgd_idbd`, `upgd_idbd_meta1e2` — UPGD-W + IDBD per-weight step-sizes
  (meta signal = gated loss gradient; log-alpha clipped to [-10, 0]).
- `upgd_autostep` — UPGD-W + Autostep per-weight step-sizes (global M).
- `upgd_l2init` — UPGD-W decay pulls toward initial weights.
- `upgd_ema_norm` — EMA input normalizer (decay 0.999) in front of the MLP.
- `upgd_cbp`, `adamw_cbp` — CBP-style dormant-unit recycling
  (rho=1e-4, utility decay 0.99, maturity 100; protocol-uniform re-init,
  per-unit optimizer-state reset).
- `upgd_w_{sigma005,sigma02,udecay0999,udecay099999,wd0005,wd002}` —
  hyperparameter-neighborhood star around the published UPGD-W config.
- `upgd_w_wclip_{k1,k2,k1_wd0,k2_wd0}` — UPGD-W + per-layer weight clipping
  to `[-kappa*s_l, +kappa*s_l]`, `s_l = 1/sqrt(fan_in)` init bound (Elsayed,
  Lan, Lyle & Mahmood, RLC 2024); kappa in {1,2} x weight decay in {0.01, 0}.
  Jobs in `jobs2.txt`; kappa=inf reduces bit-exactly to the control (pinned).
- `upgd_w_localgate` — utility gate normalized by the per-tensor max instead
  of the network-global max (zero-max guarded; single-tensor parity pinned).

Seeds 0-2 per arm; identical seeds across arms => paired comparison. All
screening shards (including the wclip/localgate wave) use the exact per-step
noise path — the screening runner has no pool mode; pool64 is reserved for
full-protocol confirmations through the `upgd_ipmnist` lane.

## Execution

One process per (config, seed), `OMP_NUM_THREADS=1`, 15-way parallel
(3+ cores left for the detached OPMNIST processes):

```bash
cd outputs/ipmnist_screening
xargs -P 15 -n 2 bash worker.sh < jobs.txt   # idempotent; skips done shards
```

Measured costs at 1 thread, per 5,000-step task: control UPGD ~30s,
IDBD/Autostep ~34s, CBP ~25-29s, ema_norm ~18s, AdamW ~4-9s. A 60-task seed
run is ~20-35 min.

## Resume

Workers are idempotent per shard (`shards/<config>_seed<seed>.json`); rerun
the xargs line to fill in whatever is missing. Logs per worker under `logs/`.

## Validation + merge (after controls / after all shards)

```bash
.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening validate-proxy \
  --shards outputs/ipmnist_screening/shards/upgd_w_control_seed*.json \
           outputs/ipmnist_screening/shards/adamw_control_seed*.json \
  --partials-dir outputs/upgd_ipmnist/partials \
  --output outputs/ipmnist_screening/proxy_validation.json

.venv/bin/python -m alberta_framework.benchmarks.ipmnist_screening merge \
  --shards outputs/ipmnist_screening/shards/*.json \
  --control-name upgd_w_control \
  --output outputs/ipmnist_screening/summary.json
```

A config with paired mean improvement > 0.005 over `upgd_w_control` is a
full-protocol confirmation candidate (200 tasks, 10+ seeds via the
`upgd_ipmnist` v3 lane) — screening results alone claim nothing.

`smoke/` holds tiny CLI calibration runs only; ignore for analysis.

## Wave 3 (`jobs3.txt`)

Two additional arms, seeds 0-2 each (6 jobs in `jobs3.txt`; `jobs.txt` and
`jobs2.txt` untouched). `upgd_w_fade_head` puts FADE-style meta-learned
per-parameter weight decay (Ramesh, Lewandowski & Schmidhuber, arXiv
2604.27063) on the output layer only: `lambda_i = exp(gamma_i)` replaces the
head's fixed decoupled decay, adapted by `gamma_i += theta_lambda * delta_t *
x_i * g_i` with a forward-mode sensitivity trace `g_i <- g_i * max(0, 1 -
lambda_i - alpha*x_i^2) - lambda_i*w_i` (published alpha=0.005, gamma_0=-6.9,
theta_lambda=0.1; gamma capped at 0; lambda=0 reduces bit-exactly to the
zero-head-decay control — pinned). Hidden layers stay published UPGD-W.
`upgd_w_idbd_swift` is the `upgd_idbd` arm plus SwiftTD's two supervised-mode
stabilizers (`alberta_framework/core/swift_td.py` forms, phi_i = gated
gradient z_i): a network-global overshoot bound scaling each update by
`eta/tau` when `tau = sum_i alpha_i z_i^2 > eta` (eta=0.1), and persistent
step-size decay `log_alpha_i += ln(eps) * z_i^2` plus meta-trace reset on
trigger (eps=0.99); `eta=inf, eps=1` reduces bit-exactly to `upgd_idbd`
(pinned). Both are exact-noise (`noise_mode="step"`) arms with no pool-mode
update. Run once the main sweep frees cores, with the same worker pattern:
`cd outputs/ipmnist_screening && xargs -P 6 -n 2 bash worker.sh < jobs3.txt`
(idempotent; shards land in `shards/` and merge with the existing control
shards through the standard `merge` command).

## Wave 4 (`jobs4.txt`) — theory-driven novel arms (QUEUED, not launched)

Eight arms x seeds 0-2 (24 jobs in `jobs4.txt`; earlier jobs files untouched).
Derivations, falsifiable predictions, and refutation criteria for every arm
live in `CONTINUAL_LEARNING_THEORY.md` (repo root). All arms are exact-noise
(`noise_mode="step"`) with no pool-mode update; all reductions below are
pinned by unit tests in `tests/test_ipmnist_screening.py`.

- `guarded_cbp_adam` — the `adamw_cbp` leader plus UPGD-style utility
  *protection only*: Adam's applied per-weight delta is scaled by
  `1 - guard_scale * gate`, gate = UPGD's sigmoid/global-max squashing of the
  bias-corrected `-w*g` utility EMA (utility_decay 0.9999). No perturbation
  (CBP supplies regeneration); moments see raw gradients; recycled units also
  reset their guard utility (fresh units restart at the neutral 0.5 gate).
  `guard_scale=0` reduces bit-exactly to `adamw_cbp` (pinned).
- `adamw_cbp_noreset` — `adamw_cbp` WITHOUT the per-unit Adam m/v/count reset
  at CBP replacement. NOTE: the reset ("vreset") is already the `adamw_cbp`
  default, so the moment-freshness mechanism is dissected by ablation.
  `cbp_replacement_rate=0` reduces to `adamw_control` (pinned for both arms).
- `upgd_w_sigma0` — lean UPGD-W at sigma=0 (pure gated SGD + decoupled
  decay); the per-step noise draw (~85-90% of UPGD step cost) is skipped
  entirely, bit-exact vs the control factory at noise_std=0 (pinned). Runs at
  near-AdamW cost (~5-9s/task vs ~30s single-thread).
- `upgd_alpha_utility` — UPGD-W whose protection gate reads passive IDBD
  step-size drift (log_alpha kept as a statistic on the raw gradient, never
  applied as a rate; gate = sigmoid(drift / global max |drift|), 0.5 at zero
  drift). meta 1e-2, initial_step_size 0.01. `meta=0` reduces bit-exactly to
  the closed-form half-gated step (pinned). Cost ~ `upgd_idbd` (~34s/task).
- `adamw_cbp_{r3e5,r3e4,m50,m200}` — axis-aligned mini-star on the untuned
  leader: replacement_rate {3e-5, 3e-4} and maturity {50, 200}, one axis at a
  time (defaults 1e-4 / 100 held elsewhere). Cost ~ `adamw_cbp`.

Run (only when the confirmation runs + sweep free cores — the box is
contended; do NOT start these alongside the 200-task lanes):

```bash
cd outputs/ipmnist_screening
xargs -P 8 -n 2 bash worker.sh < jobs4.txt   # idempotent; skips done shards
```

Then re-merge with the standard `merge` command (all shards, control
`upgd_w_control`); for the adamw-family arms the interesting paired contrast
is additionally vs `adamw_cbp` (`merge --control-name adamw_cbp` on the
adamw-family shards into a SEPARATE summary file, e.g.
`summary_wave4_adamcbp.json` — never overwrite `summary.json` with a
different control).

## Wave-4b (2026-08-01, appended late): `adamw_cbp_ema_norm`

- `adamw_cbp_ema_norm` — composition arm: the exact `adamw_cbp` update
  behind the exact `upgd_ema_norm` EMA input normalizer (same
  norm_decay=0.999 / norm_epsilon=1e-8, same per-step state threading;
  normalizer state carried in the arm state). `norm_enabled=0` reduces
  bit-exactly to `adamw_cbp` (pinned); the normalizer path is pinned
  bitwise against `upgd_ema_norm`'s on a shared stream. Cost ~ `adamw_cbp`.
- Ops: the three seeds (0-2) were APPENDED to `jobs4.txt` so endgame3's
  per-shard wait loop (which re-reads jobs4.txt each cycle) counts them and
  the final merge includes them. Because endgame3's first blocking
  `xargs < jobs4.txt` had already buffered the original 24 lines and its 6h
  deadline (started 17:18) will lapse before that xargs drains, the three
  workers were ALSO launched directly, detached, at `-P 3` via `worker.sh`
  (idempotent; endgame3's restart branch cannot double-run them — it only
  fires when no `ipmnist_screening run` processes are alive).
- Interesting paired contrasts once merged: vs `upgd_w_control` (standard),
  and vs `adamw_cbp` / `upgd_ema_norm` on the adamw-family/normalized side
  (separate summary files as above; never overwrite `summary.json`).
