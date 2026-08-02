# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `48000000`
- Seeds: `3`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `mlp_h128` | 0.024900 +/- 0.000647 | 0.898933 +/- 0.002885 | 0.106051 +/- 0.002260 | 0.128303 +/- 0.002249 |
| `mlp_h128_sharp` | 0.017447 +/- 0.000988 | 0.897867 +/- 0.006559 | 0.134250 +/- 0.002697 | 0.125874 +/- 0.001929 |
| `mlp_h64` | 0.029295 +/- 0.001542 | 0.880000 +/- 0.004692 | 0.113706 +/- 0.001498 | 0.120393 +/- 0.001348 |
| `mlp_h64_sharp` | 0.021871 +/- 0.000872 | 0.869000 +/- 0.002845 | 0.140429 +/- 0.006819 | 0.124278 +/- 0.007592 |
| `step2_hybrid_memory_trace` | 0.015085 +/- 0.000373 | 0.903400 +/- 0.002600 | 0.137202 +/- 0.000957 | 0.120915 +/- 0.000721 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.014688 +/- 0.000166 | 0.905933 +/- 0.000437 | 0.136956 +/- 0.001237 | 0.122177 +/- 0.001167 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128_sharp`: +0.003817 +/- 0.000093; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h128_sharp`: +0.012397 +/- 0.000587; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h128_sharp`: +0.002362 +/- 0.000675; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.004467 +/- 0.003291; wins/losses/ties 2/1/0.
- `test_mse` vs `mlp_h128`: -0.031151 +/- 0.003126; wins/losses/ties 0/3/0.
- `test_accuracy` vs `mlp_h128`: -0.007387 +/- 0.001528; wins/losses/ties 0/3/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_mse` vs `mlp_h128_sharp`: +0.003765 +/- 0.000102; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `online_mean_accuracy` vs `mlp_h128_sharp`: +0.012049 +/- 0.000629; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_mse` vs `mlp_h128_sharp`: +0.002759 +/- 0.000849; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `final_window_accuracy` vs `mlp_h128`: +0.007000 +/- 0.002458; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_mse` vs `mlp_h128`: -0.030905 +/- 0.003356; wins/losses/ties 0/3/0.
- `step2_hybrid_memory_trace_adaptive_sharp` `test_accuracy` vs `mlp_h128`: -0.006126 +/- 0.002376; wins/losses/ties 0/3/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
