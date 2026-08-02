# Beat-SOTA screening: final report (all arms incl. wave-4)

Proxy validation: {"all_prefixes_match": false, "atol": 1e-06, "checks": [{"config_name": "upgd_w_control", "max_abs_per_task_diff": 0.009199980408630348, "prefix_match": false, "reference_partial": "outputs/upgd_ipmnist/partials/upgd_w_seed0.json", "seed": 0}, {"config_name": "upgd_w_control", "max_abs_per_task_diff": 0.00959998573562626, "prefix_match": false, "reference_partial": "outputs/upgd_ipmnist/partials/u

## Full-protocol pool64 confirmations (200 tasks, seeds 0-2)

Pool64 upgd_w_control: {1: 0.7787729776000001, 2: 0.77910898075, 0: 0.7786109820999999} (exact controls {0: 0.77906, 1: 0.77903, 2: 0.77932}, known pool-vs-exact delta -0.00012)

- adamw_cbp: mean 0.79876 seeds [0.7986989801, 0.7991469797, 0.7989199793999999, 0.7988729805999999, 0.79886197865, 0.7983859769, 0.7982899786500001, 0.79915698075, 0.798724981, 0.7985119803499999] -> BEATS-SOTA
- adamw_cbp_ema_norm: mean 0.76895 seeds [0.7690859817500001, 0.7649679803000001, 0.7727819801499999] -> BELOW
- adamw_cbp_m200: mean 0.79899 seeds [0.7994669811999999, 0.7989409802999999, 0.79855198015] -> BEATS-SOTA
- adamw_cbp_m50: mean 0.79887 seeds [0.79878298145, 0.7988639799499999, 0.7989629795000001] -> BEATS-SOTA
- adamw_cbp_noreset: mean 0.79815 seeds [0.7981189816500001, 0.7978509797500002, 0.7984729809499999] -> BEATS-SOTA
- adamw_cbp_r3e4: mean 0.80126 seeds [0.80084197825, 0.8013829826, 0.8015579771] -> BEATS-SOTA
- adamw_cbp_r3e5: mean 0.79248 seeds [0.7923509806999999, 0.7925649773000001, 0.7925309789499999] -> BEATS-SOTA
- sgd_ema_norm: mean 0.83991 seeds [0.8399779789, 0.8399759790499999, 0.8397639792999999] -> BEATS-SOTA
- upgd_ema_norm: mean 0.85362 seeds [0.8532449779999999, 0.8535339758500001, 0.8539249773500001, 0.8534429807000001, 0.8537239767, 0.85365698035, 0.8537729774999999, 0.8533049790000001, 0.8537279788000001, 0.8538869819999999] -> BEATS-SOTA
- upgd_ema_norm_sigma0: mean 0.85051 seeds [0.8507669786, 0.8500219771499999, 0.8507559773499999] -> BEATS-SOTA
- upgd_ema_norm_wd0005: mean 0.84745 seeds [0.84757997795, 0.8473079781499999, 0.8474579769] -> BEATS-SOTA
- upgd_idbd: mean 0.77895 seeds [0.7788619779999999, 0.77865597975, 0.7793299802999999] -> TIES
- upgd_l2init: mean 0.78042 seeds [0.7801939792, 0.7800519805499999, 0.78101797995] -> BEATS-SOTA
- upgd_w_wd0005: mean 0.78431 seeds [0.78418497905, 0.7842689812499999, 0.7844669820500001, 0.7839469789, 0.7847349790500001, 0.7835629804500001, 0.7840389796000001, 0.7841219829499999, 0.7848029797000001, 0.7849659790000001] -> BEATS-SOTA

## Frontier wave (sigma0 extensions) — NEW CHAMPION

Round-1 screen (60 tasks, seeds 0-2, paired vs `upgd_ema_norm_sigma0`):
`sigma0_ndecay099` (normalizer decay 0.999 → 0.99) +0.00962 paired delta,
all seeds positive; every other extension (epsilon floors, gate temperature,
local gate, hidden-RMS norm, slower decay 0.9999) flat or negative —
`sigma0_hidden_norm` −0.0186 (hidden-layer normalization actively hurts),
`sigma0_ndecay09999` −0.0073 (slower conditioning hurts symmetrically).

Full-protocol 200-task confirmation (pool64, seeds 0-2):

- **sigma0_ndecay099: mean 0.86245 seeds [0.86229, 0.86196, 0.86311] -> NEW BEST**
  (+0.0119 vs its 0.85051 base, +0.0088 vs the 10-seed 0.85362 champion,
  +0.083 vs the 0.7791 SOTA reproduction — with **zero perturbation noise**,
  ~1/7 the champion's compute)

Interpretation: the permutation boundary shifts input statistics
instantaneously; a 100-step EMA (decay 0.99) re-conditions ~10x faster after
each switch than the 1000-step default, and the symmetric loss from slower
decay confirms the mechanism is tracking speed, not smoothing. Round-2
(decay 0.9/0.95/0.98 neighborhood + `ema_norm_ndecay099` noisy transplant)
runs under `frontier2_pipeline.sh` → `frontier2_results.json`.

All frontier numbers are development-grade (`development_screening_diagnostic`),
seeds 0-2, nonpromoting.

## Screening ranked table (proxy, 60 tasks, all arms)

```json
{
 "confirmation_threshold": 0.005,
 "control_name": "upgd_w_control",
 "created_unix": 1785662512.5236602,
 "evidence_policy": {
  "development_only": true,
  "evidence_class": "development_screening_diagnostic",
  "scientific_promotion_allowed": false
 },
 "n_shards": 105,
 "noise_mode": "step",
 "protocol_config": {
  "hidden1": 300,
  "hidden2": 150,
  "input_dim": 784,
  "n_classes": 10,
  "n_tasks": 60,
  "task_length": 5000
 },
 "results": [
  {
   "average_online_accuracy_mean": 0.8528599768888888,
   "average_online_accuracy_stderr": 0.00046428305163735635,
   "average_plasticity_mean": 0.3832127976111111,
   "base_learner": "upgd_w",
   "config_name": "upgd_ema_norm",
   "hyperparameters": {
    "noise_std": 0.1,
    "norm_decay": 0.999,
    "norm_epsilon": 1e-08,
    "step_size": 0.01,
    "utility_decay": 0.9999,
    "weight_decay": 0.01
   },
   "late_window_slope_mean": -0.00020142914285714176,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.07510110966666657,
    "per_seed_diff": [
     0.074467,
     0.07463,
     0.076207
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.0005547853158951784
   },
   "per_seed_average_online_accuracy": [
    0.851943,
    0.85319,
    0.853447
   ],
   "per_seed_late_window_slope": [
    -3.286e-05,
    -0.00021714,
    -0.00035429
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 8007.99
  },
  {
   "average_online_accuracy_mean": 0.8519722013333334,
   "average_online_accuracy_stderr": 8.402754877330571e-05,
   "average_plasticity_mean": 0.32709838050000006,
   "base_learner": "upgd_w",
   "config_name": "upgd_ema_norm_sigma0",
   "hyperparameters": {
    "noise_std": 0.0,
    "norm_decay": 0.999,
    "norm_epsilon": 1e-08,
    "step_size": 0.01,
    "utility_decay": 0.9999,
    "weight_decay": 0.01
   },
   "late_window_slope_mean": -0.00022404717857142813,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.07421333411111097,
    "per_seed_diff": [
     0.07466,
     0.07336,
     0.07462
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.00042682323386305875
   },
   "per_seed_average_online_accuracy": [
    0.852137,
    0.85192,
    0.85186
   ],
   "per_seed_late_window_slope": [
    0.00031,
    -0.000615,
    -0.00036714
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 9487.55
  },
  {
   "average_online_accuracy_mean": 0.8468710877777778,
   "average_online_accuracy_stderr": 7.397019254071422e-05,
   "average_plasticity_mean": 0.39587734249999995,
   "base_learner": "upgd_w",
   "config_name": "upgd_ema_norm_wd0005",
   "hyperparameters": {
    "noise_std": 0.1,
    "norm_decay": 0.999,
    "norm_epsilon": 1e-08,
    "step_size": 0.01,
    "utility_decay": 0.9999,
    "weight_decay": 0.005
   },
   "late_window_slope_mean": -0.0004902371190476181,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.06911222055555548,
    "per_seed_diff": [
     0.069257,
     0.068427,
     0.069653
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.00036139819340986123
   },
   "per_seed_average_online_accuracy": [
    0.846733,
    0.846987,
    0.846893
   ],
   "per_seed_late_window_slope": [
    -0.00045857,
    -0.00052928,
    -0.00048286
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 8804.62
  },
  {
   "average_online_accuracy_mean": 0.842128869388889,
   "average_online_accuracy_stderr": 0.0003142188703131245,
   "average_plasticity_mean": 0.21502089561111107,
   "base_learner": "upgd_w",
   "config_name": "upgd_ema_norm_lr0003",
   "hyperparameters": {
    "noise_std": 0.1,
    "norm_decay": 0.999,
    "norm_epsilon": 1e-08,
    "step_size": 0.003,
    "utility_decay": 0.9999,
    "weight_decay": 0.01
   },
   "late_window_slope_mean": -6.761820238095126e-05,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.06437000216666659,
    "per_seed_diff": [
     0.064363,
     0.06323,
     0.065517
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.0006601121142203985
   },
   "per_seed_average_online_accuracy": [
    0.84184,
    0.84179,
    0.842757
   ],
   "per_seed_late_window_slope": [
    0.00016929,
    -0.00029428,
    -7.786e-05
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 9201.12
  },
  {
   "average_online_accuracy_mean": 0.7994544242777777,
   "average_online_accuracy_stderr": 0.0006451718943619212,
   "average_plasticity_mean": 0.34612086038888884,
   "base_learner": "adamw",
   "config_name": "adamw_cbp_ema_norm",
   "hyperparameters": {
    "beta1": 0.0,
    "beta2": 0.99,
    "cbp_decay_rate": 0.99,
    "cbp_maturity_threshold": 100.0,
    "cbp_replacement_rate": 0.0001,
    "eps": 1e-08,
    "norm_decay": 0.999,
    "norm_enabled": 1.0,
    "norm_epsilon": 1e-08,
    "step_size": 0.0001,
    "weight_decay": 0.0
   },
   "late_window_slope_mean": -0.00012452354761905015,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.021695557055555443,
    "per_seed_diff": [
     0.022987,
     0.019693,
     0.022407
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.0010150174728200006
   },
   "per_seed_average_online_accuracy": [
    0.800463,
    0.798253,
    0.799647
   ],
   "per_seed_late_window_slope": [
    -1.714e-05,
    -3.643e-05,
    -0.00032
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 8217.19
  },
  {
   "average_online_accuracy_mean": 0.7982755348333334,
   "average_online_accuracy_stderr": 0.00018623443527240156,
   "average_plasticity_mean": 0.41667774511111116,
   "base_learner": "adamw",
   "config_name": "adamw_cbp_r3e4",
   "hyperparameters": {
    "beta1": 0.0,
    "beta2": 0.99,
    "cbp_decay_rate": 0.99,
    "cbp_maturity_threshold": 100.0,
    "cbp_replacement_rate": 0.0003,
    "eps": 1e-08,
    "step_size": 0.0001,
    "weight_decay": 0.0
   },
   "late_window_slope_mean": 5.9523642857142464e-05,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.02051666761111104,
    "per_seed_diff": [
     0.020463,
     0.020023,
     0.021063
    ],
    "seeds": [
     0,
     1,
     2
    ],
    "stderr_diff": 0.00030140294673027706
   },
   "per_seed_average_online_accuracy": [
    0.79794,
    0.798583,
    0.798303
   ],
   "per_seed_late_window_slope": [
    0.00065072,
    -0.00013714,
    -0.000335
   ],
   "seeds": [
    0,
    1,
    2
   ],
   "wall_clock_seconds_total": 7855.03
  },
  {
   "average_online_accuracy_mean": 0.7965133136666666,
   "average_online_accuracy_stderr": 0.00031850977600019766,
   "average_plasticity_mean": 0.41223621083333334,
   "base_learner": "adamw",
   "config_name": "adamw_cbp",
   "hyperparameters": {
    "beta1": 0.0,
    "beta2": 0.99,
    "cbp_decay_rate": 0.99,
    "cbp_maturity_threshold": 100.0,
    "cbp_replacement_rate": 0.0001,
    "eps": 1e-08,
    "step_size": 0.0001,
    "weight_decay": 0.0
   },
   "late_window_slope_mean": -0.00011904728571428772,
   "n_seeds": 3,
   "paired_vs_control": {
    "all_seeds_improve": true,
    "beats_control": true,
    "confirmation_candidate": true,
    "control": "upgd_w_control",
    "mean_diff": 0.01875444644444439,
```