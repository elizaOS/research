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
