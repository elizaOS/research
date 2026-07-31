# Current-Foragax FOV baseline screening

This is an open-development screening protocol for
`continual-foragax==0.55.0`, `ForagaxTwoBiomeLarge-v1`, aperture 9. It is not a
sealed evaluation and cannot support an official or SOTA claim.

Every configuration is a 100,000-step prefix of the 500,000-step stationary
FOV protocol and uses open development seeds 2,000,001 and 2,000,002. The
common-control variants retain the tuned E138 DQN exploration schedule,
optimizer, replay, target-network, and update settings. They change only the
declared intervention. This first stage is an ablation screen, not final
hyperparameter selection. Promising families must subsequently be tuned on
the full declared tuning set and evaluated from frozen configurations on
disjoint sealed seeds.

Diagnostics are disabled (`ntk_freq=0`, `x_ref_steps=0`) because they are not
part of the reward-learning algorithm and the audited upstream terminal NTK
path fails after an otherwise complete run.

The tested families are:

- CReLU and LayerNorm representation variants.
- L2-to-zero and L2-to-initialization regularization.
- periodic head reset, shrink-and-perturb, and selective weight
  reinitialization (SWR).
- a reward-trace memory baseline with an otherwise equivalent
  convolution-plus-two-hidden-layer network.
- explicit causal action/reward history, with and without the reward trace.

Recurrent DRQN/PT-DQN and PPO/RTU-PPO use materially different compute and
state contracts. They enter a separate second screening stage rather than
being silently represented by these DQN ablations.
