# IPMNIST 20-seed publication runs (development-grade, nonpromoting)

Scope: `development_screening_diagnostic` — these numbers are the durable
descriptive claim artifact for the screening campaign at the published seed
count (n=20), and are **permanently nonpromoting** under the repository's
evidence rules (no preregistered frozen protocol; the screening source is not
registry-bound). Protocol: ICLR-2024 input-permuted MNIST — 200 tasks x 5000
steps, one example per step, 300x150 ReLU MLP, average online accuracy,
exact per-step noise mode (both arms are noise-free, sigma=0). Runner:
`worker_pub.sh` -> `alberta_framework.benchmarks.ipmnist_screening run`.

## Headline (seeds 0-19)

| Arm | n | Mean online acc +/- stderr |
|---|---:|---|
| `sigma0_shiftnorm_d099` (final best) | 20 | **0.86449 +/- 0.00009** |
| `sigma0_ndecay099` (prior champion) | 20 | 0.86242 +/- 0.00010 |

Cited references (previously stored artifacts, not rerun here):

| Reference | n | Mean online acc +/- stderr | Source |
|---|---:|---|---|
| `upgd_ema_norm` (UPGD-W + EMA input norm) | 10 | 0.85362 +/- 0.00007 | `confirm_full/upgd_ema_norm_seed*.json` |
| `upgd_w` published-config reproduction (baseline) | 10 | 0.77915 +/- 0.00006 | `outputs/upgd_ipmnist/partials/upgd_w_seed*.json` |
| `adamw` published-config reproduction | 10 | 0.71900 +/- 0.00059 | `outputs/upgd_ipmnist/partials/adamw_seed*.json` |

## Selection-bias caveat

Seeds 0-2 of both headline arms were consumed by screening/selection; seeds
3-19 are selection-untouched. Held-out-only means (seeds 3-19, n=17):

- `sigma0_shiftnorm_d099`: 0.86447 +/- 0.00009
- `sigma0_ndecay099`: 0.86241 +/- 0.00011

## Final-best hyperparameters

`sigma0_shiftnorm_d099`: {"fast_decay": 0.9, "gate_beta": 1.0, "hidden_rms": 0.0, "local_gate": 0.0, "noise_std": 0.0, "norm_decay": 0.99, "norm_epsilon": 1e-08, "shift_delta": 0.02, "shift_k": 1.0, "shift_refractory": 0.0, "step_size": 0.01, "utility_decay": 0.9999, "weight_decay": 0.01}

## Per-seed means

`sigma0_shiftnorm_d099`: {"0": 0.864213, "1": 0.864415, "2": 0.865129, "3": 0.864954, "4": 0.864528, "5": 0.864769, "6": 0.863772, "7": 0.864269, "8": 0.86399, "9": 0.86473, "10": 0.864118, "11": 0.86434, "12": 0.864506, "13": 0.864372, "14": 0.864288, "15": 0.865437, "16": 0.864736, "17": 0.864589, "18": 0.864328, "19": 0.864326}

`sigma0_ndecay099`: {"0": 0.862292, "1": 0.861956, "2": 0.863106, "3": 0.86287, "4": 0.862766, "5": 0.862194, "6": 0.862067, "7": 0.861923, "8": 0.862155, "9": 0.862822, "10": 0.861944, "11": 0.861941, "12": 0.862401, "13": 0.862434, "14": 0.862066, "15": 0.863555, "16": 0.862816, "17": 0.862327, "18": 0.862359, "19": 0.862397}

## Mini-star context

Screen-and-confirm chain for the shift-detector mini-star (60-task screen,
paired vs `sigma0_shiftnorm_d099`, +0.002 auto-confirm bar; 200-task
confirmation for candidates): `../shiftstar_results.json`. The 200-task
3-seed confirmation that made `sigma0_shiftnorm_d099` the record holder:
mean 0.86459 seeds [0.864213, 0.864415, 0.865129]
(`../confirm_full/sigma0_shiftnorm_d099_seed*.json`).
