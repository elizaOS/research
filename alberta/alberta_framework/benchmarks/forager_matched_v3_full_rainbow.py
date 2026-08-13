# Copyright 2021 The Dopamine Authors.
# Modifications Copyright 2026 elizaOS contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified Dopamine Full Rainbow core for the matched-v3 Forager task.

This is a source-attributed derivative of Dopamine's compact JAX Full Rainbow
agent, changed for a 9x9 three-channel Forager aperture and four actions.  It
retains the algorithmic components under comparison: three-step returns,
proportional prioritized replay, inverse-square-root importance weights,
C51, Double-Q action selection, factorized noisy layers, and dueling heads.

The implementation deliberately stops at an executable, deterministic update
core.  It does not provide a full Foragax environment runner, result writer,
qualification receipt, or authority mechanism.  Its descriptor therefore says
``implemented_unqualified`` while execution readiness remains false.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax import linen as nn
from jax import Array

FULL_RAINBOW_CONFIG_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_config.v1"
)
FULL_RAINBOW_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_adapter.v1"
)
FULL_RAINBOW_ADAPTER_STATUS: Final = "implemented_unqualified"

_MAX_CANONICAL_BYTES: Final = 2 * 1024 * 1024
_UINT31_MAX: Final = (1 << 31) - 1
_AGENT_NAMESPACE: Final = 0x4147454E
_PRIORITY_EPSILON: Final = 1e-10

EXPECTED_ONLINE_PARAMETER_SCALARS: Final = 908_798
EXPECTED_ADAM_MOMENT_SCALARS: Final = 2 * EXPECTED_ONLINE_PARAMETER_SCALARS
EXPECTED_PARAMETER_TARGET_OPTIMIZER_BYTES: Final = 14_540_772


class FullRainbowContractError(ValueError):
    """An exact-task configuration, input, or descriptor violated its contract."""


class FullRainbowExecutionBlockedError(RuntimeError):
    """The unqualified core was mistaken for a complete authorized runner."""


@dataclass(frozen=True, slots=True)
class FullRainbowForagerConfig:
    """Exact, frozen matched-v3 Full Rainbow configuration."""

    environment_id: str = "ForagaxTwoBiomeLarge-v1"
    observation_type: str = "color"
    observation_shape: tuple[int, int, int] = (9, 9, 3)
    num_actions: int = 4
    horizon: int = 499_712
    raw_reward_values: tuple[int, int, int, int] = (-1, 0, 1, 30)
    gamma: float = 0.99
    update_horizon: int = 3
    stack_size: int = 1
    reward_divisor: float = 30.0
    support_minimum: float = -4.0
    support_maximum: float = 100.0
    num_atoms: int = 51
    prioritized_replay: bool = True
    replay_scheme: str = "proportional"
    replay_capacity: int = 1_000_000
    batch_size: int = 32
    minimum_replay_history: int = 20_000
    update_period: int = 4
    target_update_period: int = 8_000
    importance_weight_exponent: float = 0.5
    priority_update_exponent: float = 0.5
    distributional_c51: bool = True
    double_q: bool = True
    factorized_noisy: bool = True
    dueling: bool = True
    training_epsilon: float = 0.0
    evaluation_epsilon: float = 0.001
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    conv_kernel_sizes: tuple[int, int, int] = (3, 3, 3)
    conv_strides: tuple[int, int, int] = (1, 1, 1)
    conv_padding: str = "VALID"
    hidden_units: int = 512
    noisy_sigma_zero: float = 0.1
    optimizer: str = "adam"
    learning_rate: float = 0.0000625
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 0.00015

    def __post_init__(self) -> None:
        expected = _expected_config_values()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = expected[field.name]
            if type(actual) is not type(wanted) or actual != wanted:
                raise FullRainbowContractError(
                    f"{field.name} must equal the exact Full Rainbow Forager binding"
                )
        if self.raw_return_minimum / self.reward_divisor < self.support_minimum:
            raise FullRainbowContractError("C51 support does not cover the minimum return")
        if self.raw_return_maximum / self.reward_divisor > self.support_maximum:
            raise FullRainbowContractError("C51 support does not cover the maximum return")

    @property
    def raw_return_minimum(self) -> float:
        """The theoretical infinite-horizon raw discounted lower bound."""

        gamma = Fraction(str(self.gamma))
        return float(Fraction(min(self.raw_reward_values), 1) / (1 - gamma))

    @property
    def raw_return_maximum(self) -> float:
        """The theoretical infinite-horizon raw discounted upper bound."""

        gamma = Fraction(str(self.gamma))
        return float(Fraction(max(self.raw_reward_values), 1) / (1 - gamma))


def _expected_config_values() -> dict[str, object]:
    return {
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "observation_type": "color",
        "observation_shape": (9, 9, 3),
        "num_actions": 4,
        "horizon": 499_712,
        "raw_reward_values": (-1, 0, 1, 30),
        "gamma": 0.99,
        "update_horizon": 3,
        "stack_size": 1,
        "reward_divisor": 30.0,
        "support_minimum": -4.0,
        "support_maximum": 100.0,
        "num_atoms": 51,
        "prioritized_replay": True,
        "replay_scheme": "proportional",
        "replay_capacity": 1_000_000,
        "batch_size": 32,
        "minimum_replay_history": 20_000,
        "update_period": 4,
        "target_update_period": 8_000,
        "importance_weight_exponent": 0.5,
        "priority_update_exponent": 0.5,
        "distributional_c51": True,
        "double_q": True,
        "factorized_noisy": True,
        "dueling": True,
        "training_epsilon": 0.0,
        "evaluation_epsilon": 0.001,
        "conv_channels": (32, 64, 64),
        "conv_kernel_sizes": (3, 3, 3),
        "conv_strides": (1, 1, 1),
        "conv_padding": "VALID",
        "hidden_units": 512,
        "noisy_sigma_zero": 0.1,
        "optimizer": "adam",
        "learning_rate": 0.0000625,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 0.00015,
    }


@dataclass(frozen=True, slots=True)
class FullRainbowSeedRoots:
    """Independent explicit environment and candidate-private agent roots."""

    environment: Array
    agent: Array


@dataclass(frozen=True, slots=True)
class ThreeStepReturn:
    """One scaled three-step return and its bootstrap multiplier."""

    scaled_return: float
    bootstrap_discount: float
    terminal: bool


@dataclass(frozen=True, slots=True)
class FactorizedGaussianNoise:
    """One rank-one weight perturbation and matching bias perturbation."""

    weight: Array
    bias: Array


class FullRainbowNetworkOutput(NamedTuple):
    """Distributional network outputs for all four actions."""

    q_values: Array
    logits: Array
    probabilities: Array


@dataclass(frozen=True)
class FullRainbowCoreState:
    """Frozen learning-core state; no environment state or result sink is included."""

    online_params: Any
    target_params: Any
    optimizer_state: Any
    environment_rng: Array
    agent_rng: Array
    optimizer_updates: int


@dataclass(frozen=True)
class FullRainbowReplayBatch:
    """Validated already-accumulated replay sample for one optimizer update."""

    states: Array
    actions: Array
    next_states: Array
    scaled_n_step_rewards: Array
    bootstrap_discounts: Array
    sampling_probabilities: Array


@dataclass(frozen=True)
class FullRainbowTrainMetrics:
    """Unweighted losses drive priorities; weighted loss drives gradients."""

    mean_weighted_loss: Array
    per_example_loss: Array
    importance_weights: Array
    updated_priorities: Array
    double_q_actions: Array


@dataclass(frozen=True, slots=True)
class ProportionalReplaySample:
    """Indices and correction terms from one with-replacement priority draw."""

    indices: Array
    sampling_probabilities: Array
    importance_weights: Array


def _require_uint31(value: object, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise FullRainbowContractError(f"{name} must be an exact uint31 integer")
    return value


def full_rainbow_seed_roots(
    *, environment_seed: int, agent_seed: int
) -> FullRainbowSeedRoots:
    """Derive disjoint roots without letting either seed consume the other."""

    environment = _require_uint31(environment_seed, name="environment_seed")
    agent = _require_uint31(agent_seed, name="agent_seed")
    return FullRainbowSeedRoots(
        environment=jr.key(environment, impl="threefry2x32"),
        agent=jr.fold_in(jr.key(agent, impl="threefry2x32"), _AGENT_NAMESPACE),
    )


def frozen_support(config: FullRainbowForagerConfig) -> Array:
    """Return the immutable 51-atom C51 support used by every network."""

    _require_exact_config(config)
    return jnp.linspace(
        config.support_minimum,
        config.support_maximum,
        config.num_atoms,
        dtype=jnp.float32,
    )


def three_step_return(
    config: FullRainbowForagerConfig,
    *,
    raw_rewards: Sequence[int],
    terminals: Sequence[bool],
) -> ThreeStepReturn:
    """Accumulate exactly three raw rewards, truncating after a terminal."""

    _require_exact_config(config)
    if len(raw_rewards) != config.update_horizon or len(terminals) != config.update_horizon:
        raise FullRainbowContractError("three-step inputs must contain exactly three entries")
    if any(
        type(value) is not int or value not in config.raw_reward_values
        for value in raw_rewards
    ):
        raise FullRainbowContractError("raw reward is outside the exact task reward set")
    if any(type(value) is not bool for value in terminals):
        raise FullRainbowContractError("terminal flags must be exact booleans")

    total = 0.0
    terminal = False
    for offset, (reward, stops) in enumerate(zip(raw_rewards, terminals, strict=True)):
        total += (config.gamma**offset) * reward / config.reward_divisor
        if stops:
            terminal = True
            break
    return ThreeStepReturn(
        scaled_return=total,
        bootstrap_discount=0.0 if terminal else config.gamma**config.update_horizon,
        terminal=terminal,
    )


def _array(value: object, *, name: str, rank: int = 1) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.ndim != rank or array.size == 0:
        raise FullRainbowContractError(f"{name} must be a nonempty rank-{rank} array")
    is_real = np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.floating
    )
    if not is_real or np.issubdtype(array.dtype, np.bool_):
        raise FullRainbowContractError(f"{name} must contain numeric non-boolean values")
    if not bool(np.all(np.isfinite(array))):
        raise FullRainbowContractError(f"{name} must be finite")
    return array


def proportional_sampling_probabilities(priorities: Array) -> Array:
    """Normalize stored priorities, with Dopamine's all-zero uniform fallback."""

    values = _array(priorities, name="priorities")
    if bool(np.any(values < 0.0)):
        raise FullRainbowContractError("priorities must be nonnegative")
    total = float(np.sum(values, dtype=np.float64))
    if total == 0.0:
        return jnp.full(values.shape, 1.0 / values.size, dtype=jnp.float32)
    return jnp.asarray(values / total, dtype=jnp.float32)


def importance_sampling_weights(probabilities: Array) -> Array:
    """Return max-normalized inverse-square-root replay correction weights."""

    values = _array(probabilities, name="sampling probabilities")
    if bool(np.any(values <= 0.0)) or bool(np.any(values > 1.0)):
        raise FullRainbowContractError("sampling probabilities must lie in (0, 1]")
    weights = 1.0 / jnp.sqrt(jnp.asarray(values, dtype=jnp.float32) + _PRIORITY_EPSILON)
    return weights / jnp.max(weights)


def sample_proportional_replay(
    config: FullRainbowForagerConfig,
    key: Array,
    priorities: Array,
) -> ProportionalReplaySample:
    """Draw one exact 32-transition batch with replacement by priority."""

    _require_exact_config(config)
    probabilities = proportional_sampling_probabilities(priorities)
    indices = jr.choice(
        key,
        probabilities.shape[0],
        shape=(config.batch_size,),
        replace=True,
        p=probabilities,
    )
    selected_probabilities = probabilities[indices]
    return ProportionalReplaySample(
        indices=indices,
        sampling_probabilities=selected_probabilities,
        importance_weights=importance_sampling_weights(selected_probabilities),
    )


def priority_updates(per_example_loss: Array) -> Array:
    """Map unweighted cross-entropy losses to stored replay priorities."""

    values = _array(per_example_loss, name="per-example losses")
    if bool(np.any(values < 0.0)):
        raise FullRainbowContractError("per-example losses must be nonnegative")
    return jnp.sqrt(jnp.asarray(values, dtype=jnp.float32) + _PRIORITY_EPSILON)


def _factorized_noise(key: Array, input_features: int, output_features: int) -> tuple[Array, Array]:
    input_key, output_key = jr.split(key)
    input_noise = jr.normal(input_key, (input_features,), dtype=jnp.float32)
    output_noise = jr.normal(output_key, (output_features,), dtype=jnp.float32)
    transformed_input = jnp.sign(input_noise) * jnp.sqrt(jnp.abs(input_noise))
    transformed_output = jnp.sign(output_noise) * jnp.sqrt(jnp.abs(output_noise))
    return jnp.outer(transformed_input, transformed_output), transformed_output


def factorized_gaussian_noise(
    key: Array,
    *,
    input_features: int,
    output_features: int,
    eval_mode: bool,
) -> FactorizedGaussianNoise:
    """Expose the factored-Gaussian perturbation used by every noisy layer."""

    if type(input_features) is not int or input_features <= 0:
        raise FullRainbowContractError("input_features must be a positive integer")
    if type(output_features) is not int or output_features <= 0:
        raise FullRainbowContractError("output_features must be a positive integer")
    if type(eval_mode) is not bool:
        raise FullRainbowContractError("eval_mode must be an exact boolean")
    if eval_mode:
        return FactorizedGaussianNoise(
            weight=jnp.zeros((input_features, output_features), dtype=jnp.float32),
            bias=jnp.zeros((output_features,), dtype=jnp.float32),
        )
    weight, bias = _factorized_noise(key, input_features, output_features)
    return FactorizedGaussianNoise(weight=weight, bias=bias)


class _FactorizedNoisyDense(nn.Module):
    """Dopamine/Fortunato-style factorized noisy affine layer."""

    features: int
    sigma_zero: float

    @nn.compact
    def __call__(self, inputs: Array, *, key: Array, eval_mode: bool) -> Array:
        input_features = inputs.shape[-1]
        limit = 1.0 / math.sqrt(input_features)
        sigma = self.sigma_zero / math.sqrt(input_features)

        def mu_init(rng: Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> Array:
            return jr.uniform(
                rng,
                shape,
                dtype,
                minval=-limit,
                maxval=limit,
            )

        def sigma_init(
            _rng: Array, shape: tuple[int, ...], dtype: Any = jnp.float32
        ) -> Array:
            return jnp.full(shape, sigma, dtype=dtype)

        kernel_mu = self.param("kernel_mu", mu_init, (input_features, self.features))
        kernel_sigma = self.param(
            "kernel_sigma", sigma_init, (input_features, self.features)
        )
        bias_mu = self.param("bias_mu", mu_init, (self.features,))
        bias_sigma = self.param("bias_sigma", sigma_init, (self.features,))
        noise = factorized_gaussian_noise(
            key,
            input_features=input_features,
            output_features=self.features,
            eval_mode=eval_mode,
        )
        kernel = kernel_mu + kernel_sigma * noise.weight
        bias = bias_mu + bias_sigma * noise.bias
        return jnp.matmul(inputs, kernel) + bias


class _ForagerFullRainbowNetwork(nn.Module):
    """Three small valid convolutions followed by noisy dueling C51 heads."""

    config: FullRainbowForagerConfig

    @nn.compact
    def __call__(
        self,
        observation: Array,
        *,
        support: Array,
        key: Array,
        eval_mode: bool,
    ) -> FullRainbowNetworkOutput:
        if observation.shape != self.config.observation_shape:
            raise FullRainbowContractError(
                f"observation shape must be {self.config.observation_shape!r}"
            )
        if support.shape != (self.config.num_atoms,):
            raise FullRainbowContractError("support shape does not match num_atoms")
        value = observation.astype(jnp.float32)
        initializer = nn.initializers.xavier_uniform()
        for index, (channels, kernel_size, stride) in enumerate(
            zip(
                self.config.conv_channels,
                self.config.conv_kernel_sizes,
                self.config.conv_strides,
                strict=True,
            )
        ):
            value = nn.Conv(
                features=channels,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=self.config.conv_padding,
                kernel_init=initializer,
                name=f"conv_{index}",
            )(value)
            value = nn.relu(value)
        value = value.reshape((-1,))
        hidden_key, advantage_key, value_key = jr.split(key, 3)
        value = _FactorizedNoisyDense(
            self.config.hidden_units,
            self.config.noisy_sigma_zero,
            name="hidden",
        )(value, key=hidden_key, eval_mode=eval_mode)
        value = nn.relu(value)
        advantage = _FactorizedNoisyDense(
            self.config.num_actions * self.config.num_atoms,
            self.config.noisy_sigma_zero,
            name="advantage",
        )(value, key=advantage_key, eval_mode=eval_mode)
        atom_value = _FactorizedNoisyDense(
            self.config.num_atoms,
            self.config.noisy_sigma_zero,
            name="value",
        )(value, key=value_key, eval_mode=eval_mode)
        advantage = advantage.reshape((self.config.num_actions, self.config.num_atoms))
        atom_value = atom_value.reshape((1, self.config.num_atoms))
        logits = atom_value + advantage - jnp.mean(advantage, axis=0, keepdims=True)
        probabilities = jax.nn.softmax(logits, axis=-1)
        q_values = jnp.sum(support[None, :] * probabilities, axis=-1)
        return FullRainbowNetworkOutput(q_values, logits, probabilities)


def _network(config: FullRainbowForagerConfig) -> _ForagerFullRainbowNetwork:
    _require_exact_config(config)
    return _ForagerFullRainbowNetwork(config)


def apply_full_rainbow_network(
    config: FullRainbowForagerConfig,
    params: Any,
    observation: Array,
    key: Array,
    *,
    eval_mode: bool,
) -> FullRainbowNetworkOutput:
    """Apply the exact network with explicit train/evaluation noise semantics."""

    if type(eval_mode) is not bool:
        raise FullRainbowContractError("eval_mode must be an exact boolean")
    return cast(
        FullRainbowNetworkOutput,
        _network(config).apply(
            {"params": params},
            observation,
            support=frozen_support(config),
            key=key,
            eval_mode=eval_mode,
        ),
    )


def _optimizer(config: FullRainbowForagerConfig) -> optax.GradientTransformation:
    return optax.adam(
        config.learning_rate,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon,
    )


def initialize_full_rainbow_core(
    config: FullRainbowForagerConfig,
    *,
    environment_seed: int,
    agent_seed: int,
) -> FullRainbowCoreState:
    """Initialize params from only the private root and retain the env root untouched."""

    _require_exact_config(config)
    roots = full_rainbow_seed_roots(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
    )
    next_agent_rng, init_key, noise_key = jr.split(roots.agent, 3)
    variables = _network(config).init(
        init_key,
        jnp.zeros(config.observation_shape, dtype=jnp.float32),
        support=frozen_support(config),
        key=noise_key,
        eval_mode=False,
    )
    params = variables["params"]
    count = parameter_scalar_count(params)
    if count != EXPECTED_ONLINE_PARAMETER_SCALARS:
        raise AssertionError(
            f"Full Rainbow parameter count drifted: {count} != "
            f"{EXPECTED_ONLINE_PARAMETER_SCALARS}"
        )
    return FullRainbowCoreState(
        online_params=params,
        target_params=params,
        optimizer_state=_optimizer(config).init(params),
        environment_rng=roots.environment,
        agent_rng=next_agent_rng,
        optimizer_updates=0,
    )


def parameter_scalar_count(params: Any) -> int:
    """Count every parameter scalar in an arbitrary Flax parameter tree."""

    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params))


def _project_c51(target_support: Array, probabilities: Array, support: Array) -> Array:
    minimum = support[0]
    maximum = support[-1]
    spacing = (maximum - minimum) / (support.shape[0] - 1)
    clipped = jnp.clip(target_support, minimum, maximum)
    positions = (clipped - minimum) / spacing
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.ceil(positions).astype(jnp.int32)
    lower_weight = jnp.where(lower == upper, 1.0, upper - positions)
    upper_weight = jnp.where(lower == upper, 0.0, positions - lower)
    projection = jnp.zeros_like(support)
    projection = projection.at[lower].add(probabilities * lower_weight)
    return projection.at[upper].add(probabilities * upper_weight)


def project_c51_distribution(
    *, target_support: Array, probabilities: Array, support: Array
) -> Array:
    """Project a categorical Bellman target onto an equally spaced frozen support."""

    target_values = _array(target_support, name="target support")
    probability_values = _array(probabilities, name="probabilities")
    support_values = _array(support, name="frozen support")
    if (
        target_values.shape != probability_values.shape
        or target_values.shape != support_values.shape
    ):
        raise FullRainbowContractError("C51 arrays must have identical one-dimensional shapes")
    if bool(np.any(probability_values < 0.0)) or not math.isclose(
        float(np.sum(probability_values, dtype=np.float64)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise FullRainbowContractError("C51 probabilities must be nonnegative and sum to one")
    differences = np.diff(support_values)
    if bool(np.any(differences <= 0.0)) or not bool(
        np.allclose(differences, differences[0], rtol=1e-5, atol=1e-6)
    ):
        raise FullRainbowContractError("C51 support must be increasing and equally spaced")
    return _project_c51(
        jnp.asarray(target_values, dtype=jnp.float32),
        jnp.asarray(probability_values, dtype=jnp.float32),
        jnp.asarray(support_values, dtype=jnp.float32),
    )


def double_q_c51_target(
    *,
    online_next_q_values: Array,
    target_next_probabilities: Array,
    scaled_n_step_reward: Array,
    bootstrap_discount: Array,
    support: Array,
) -> tuple[Array, Array]:
    """Select with online Q-values and evaluate/project the target distribution."""

    online = _array(online_next_q_values, name="online next Q-values")
    target = _array(target_next_probabilities, name="target probabilities", rank=2)
    support_values = _array(support, name="support")
    if target.shape != (online.size, support_values.size):
        raise FullRainbowContractError("target distribution shape does not match actions/support")
    if bool(np.any(target < 0.0)) or not bool(
        np.allclose(np.sum(target, axis=1), np.ones(online.size), atol=1e-6, rtol=0.0)
    ):
        raise FullRainbowContractError("each target action distribution must sum to one")
    reward = np.asarray(scaled_n_step_reward)
    discount = np.asarray(bootstrap_discount)
    if reward.shape != () or discount.shape != ():
        raise FullRainbowContractError("reward and bootstrap discount must be scalars")
    if not bool(np.isfinite(reward)) or not bool(np.isfinite(discount)):
        raise FullRainbowContractError("reward and bootstrap discount must be finite")
    action = jnp.argmax(jnp.asarray(online, dtype=jnp.float32))
    chosen = jnp.asarray(target, dtype=jnp.float32)[action]
    target_atoms = jnp.asarray(reward, dtype=jnp.float32) + jnp.asarray(
        discount, dtype=jnp.float32
    ) * jnp.asarray(support_values, dtype=jnp.float32)
    return action, _project_c51(
        target_atoms,
        chosen,
        jnp.asarray(support_values, dtype=jnp.float32),
    )


def _valid_scaled_returns(config: FullRainbowForagerConfig) -> np.ndarray[Any, Any]:
    values: set[float] = set()
    for first in config.raw_reward_values:
        values.add(first / config.reward_divisor)
        for second in config.raw_reward_values:
            values.add((first + config.gamma * second) / config.reward_divisor)
            for third in config.raw_reward_values:
                values.add(
                    (first + config.gamma * second + config.gamma**2 * third)
                    / config.reward_divisor
                )
    return np.asarray(sorted(values), dtype=np.float32)


def validate_replay_batch(
    config: FullRainbowForagerConfig, batch: FullRainbowReplayBatch
) -> None:
    """Reject malformed, unscaled, or task-incompatible update batches."""

    _require_exact_config(config)
    if type(batch) is not FullRainbowReplayBatch:
        raise FullRainbowContractError("batch must be a FullRainbowReplayBatch")
    states = np.asarray(batch.states)
    next_states = np.asarray(batch.next_states)
    expected_tail = config.observation_shape
    if (
        states.ndim != 4
        or states.shape[1:] != expected_tail
        or states.shape[0] != config.batch_size
    ):
        raise FullRainbowContractError("states do not match the exact observation batch shape")
    if next_states.shape != states.shape:
        raise FullRainbowContractError("next_states must exactly match states shape")
    if not bool(np.all(np.isfinite(states))) or not bool(np.all(np.isfinite(next_states))):
        raise FullRainbowContractError("observations must be finite")
    if bool(np.any(states < 0.0)) or bool(np.any(states > 1.0)):
        raise FullRainbowContractError("states must lie in the color observation range [0, 1]")
    if bool(np.any(next_states < 0.0)) or bool(np.any(next_states > 1.0)):
        raise FullRainbowContractError("next_states must lie in the color observation range [0, 1]")
    for name, observations in (("states", states), ("next_states", next_states)):
        if not bool(np.all((observations == 0.0) | (observations == 1.0))):
            raise FullRainbowContractError(
                f"{name} must contain exact one-hot color indicators"
            )
        if bool(np.any(np.sum(observations, axis=-1) > 1.0)):
            raise FullRainbowContractError(
                f"{name} cannot activate more than one color per aperture cell"
            )
    batch_size = states.shape[0]
    actions = np.asarray(batch.actions)
    if (
        actions.shape != (batch_size,)
        or not np.issubdtype(actions.dtype, np.integer)
        or np.issubdtype(actions.dtype, np.bool_)
        or bool(np.any(actions < 0))
        or bool(np.any(actions >= config.num_actions))
    ):
        raise FullRainbowContractError("actions must be exact integer indices in [0, 4)")
    rewards = _array(batch.scaled_n_step_rewards, name="scaled n-step rewards")
    if rewards.shape != (batch_size,):
        raise FullRainbowContractError("scaled n-step rewards must match batch size")
    valid_returns = _valid_scaled_returns(config)
    if any(
        not bool(np.any(np.isclose(value, valid_returns, rtol=0.0, atol=2e-6)))
        for value in rewards
    ):
        raise FullRainbowContractError("scaled n-step reward is not reachable from task rewards")
    discounts = _array(batch.bootstrap_discounts, name="bootstrap discounts")
    if discounts.shape != (batch_size,):
        raise FullRainbowContractError("bootstrap discounts must match batch size")
    continuing_discount = np.float32(config.gamma**config.update_horizon)
    if any(
        not (
            float(value) == 0.0
            or math.isclose(
                float(value), float(continuing_discount), rel_tol=0.0, abs_tol=1e-7
            )
        )
        for value in discounts
    ):
        raise FullRainbowContractError("bootstrap discounts must be zero or gamma cubed")
    probabilities = _array(batch.sampling_probabilities, name="sampling probabilities")
    if probabilities.shape != (batch_size,):
        raise FullRainbowContractError("sampling probabilities must match batch size")
    if bool(np.any(probabilities <= 0.0)) or bool(np.any(probabilities > 1.0)):
        raise FullRainbowContractError("sampling probabilities must lie in (0, 1]")


def _batched_apply(
    config: FullRainbowForagerConfig,
    params: Any,
    observations: Array,
    key: Array,
    *,
    eval_mode: bool,
) -> FullRainbowNetworkOutput:
    model = _network(config)
    support = frozen_support(config)

    def apply_one(observation: Array) -> FullRainbowNetworkOutput:
        return cast(
            FullRainbowNetworkOutput,
            model.apply(
                {"params": params},
                observation,
                support=support,
                key=key,
                eval_mode=eval_mode,
            ),
        )

    return jax.vmap(apply_one)(observations)


def train_full_rainbow_step(
    config: FullRainbowForagerConfig,
    state: FullRainbowCoreState,
    batch: FullRainbowReplayBatch,
) -> tuple[FullRainbowCoreState, FullRainbowTrainMetrics]:
    """Run one deterministic noisy Double-Q/C51 optimizer and priority update."""

    _require_exact_config(config)
    if type(state) is not FullRainbowCoreState:
        raise FullRainbowContractError("state must be a FullRainbowCoreState")
    validate_replay_batch(config, batch)
    # Preserve Dopamine's assignment order exactly: the first split output
    # drives current-state logits, the second drives target/next-state logits,
    # and the third becomes the next persistent agent key.
    current_key, next_state_key, next_agent_rng = jr.split(state.agent_rng, 3)
    target_output = _batched_apply(
        config,
        state.target_params,
        batch.next_states,
        next_state_key,
        eval_mode=False,
    )
    online_next_output = _batched_apply(
        config,
        state.online_params,
        batch.next_states,
        next_state_key,
        eval_mode=False,
    )
    double_q_actions = jnp.argmax(online_next_output.q_values, axis=1)
    chosen_target_probabilities = jnp.take_along_axis(
        target_output.probabilities,
        double_q_actions[:, None, None],
        axis=1,
    )[:, 0, :]
    support = frozen_support(config)
    target_atoms = (
        batch.scaled_n_step_rewards[:, None]
        + batch.bootstrap_discounts[:, None] * support[None, :]
    )
    targets = jax.vmap(_project_c51, in_axes=(0, 0, None))(
        target_atoms,
        chosen_target_probabilities,
        support,
    )
    targets = jax.lax.stop_gradient(targets)
    weights = importance_sampling_weights(batch.sampling_probabilities)

    def loss_function(params: Any) -> tuple[Array, Array]:
        current_output = _batched_apply(
            config,
            params,
            batch.states,
            current_key,
            eval_mode=False,
        )
        chosen_logits = jnp.take_along_axis(
            current_output.logits,
            batch.actions[:, None, None],
            axis=1,
        )[:, 0, :]
        losses = -jnp.sum(targets * jax.nn.log_softmax(chosen_logits), axis=1)
        return jnp.mean(weights * losses), losses

    (mean_loss, losses), gradients = jax.value_and_grad(
        loss_function, has_aux=True
    )(state.online_params)
    optimizer = _optimizer(config)
    updates, optimizer_state = optimizer.update(
        gradients,
        state.optimizer_state,
        params=state.online_params,
    )
    online_params = optax.apply_updates(state.online_params, updates)
    updated = FullRainbowCoreState(
        online_params=online_params,
        target_params=state.target_params,
        optimizer_state=optimizer_state,
        environment_rng=state.environment_rng,
        agent_rng=next_agent_rng,
        optimizer_updates=state.optimizer_updates + 1,
    )
    return updated, FullRainbowTrainMetrics(
        mean_weighted_loss=mean_loss,
        per_example_loss=losses,
        importance_weights=weights,
        updated_priorities=jnp.sqrt(losses + _PRIORITY_EPSILON),
        double_q_actions=double_q_actions,
    )


def sync_full_rainbow_target(state: FullRainbowCoreState) -> FullRainbowCoreState:
    """Return a frozen state with an exact online-to-target parameter snapshot."""

    if type(state) is not FullRainbowCoreState:
        raise FullRainbowContractError("state must be a FullRainbowCoreState")
    return FullRainbowCoreState(
        online_params=state.online_params,
        target_params=state.online_params,
        optimizer_state=state.optimizer_state,
        environment_rng=state.environment_rng,
        agent_rng=state.agent_rng,
        optimizer_updates=state.optimizer_updates,
    )


def _require_exact_config(config: object) -> FullRainbowForagerConfig:
    if type(config) is not FullRainbowForagerConfig:
        raise FullRainbowContractError("config must be an exact FullRainbowForagerConfig")
    return config


def _canonical_json(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise FullRainbowContractError("value is not finite canonical JSON") from exc
    if len(payload) > _MAX_CANONICAL_BYTES:
        raise FullRainbowContractError("canonical payload exceeds its byte limit")
    return payload


def _full_rainbow_config_payload() -> dict[str, Any]:
    """Construct the exact-task configuration before its bytes are frozen."""

    config = FullRainbowForagerConfig()
    return {
        "schema_version": FULL_RAINBOW_CONFIG_SCHEMA_VERSION,
        "candidate_id": "adapted_full_rainbow",
        "status": FULL_RAINBOW_ADAPTER_STATUS,
        "task": {
            "environment_id": config.environment_id,
            "observation_type": config.observation_type,
            "observation_shape": list(config.observation_shape),
            "num_actions": config.num_actions,
            "horizon": config.horizon,
            "raw_reward_values": list(config.raw_reward_values),
        },
        "algorithm": {
            "gamma": config.gamma,
            "update_horizon": config.update_horizon,
            "stack_size": config.stack_size,
            "prioritized_replay": config.prioritized_replay,
            "replay_scheme": config.replay_scheme,
            "replay_capacity": config.replay_capacity,
            "batch_size": config.batch_size,
            "minimum_replay_history": config.minimum_replay_history,
            "update_period": config.update_period,
            "target_update_period": config.target_update_period,
            "importance_weight_exponent": config.importance_weight_exponent,
            "priority_update": "sqrt(unweighted_cross_entropy + 1e-10)",
            "distributional_c51": config.distributional_c51,
            "double_q": config.double_q,
            "training_epsilon": config.training_epsilon,
            "evaluation_epsilon": config.evaluation_epsilon,
        },
        "reward_and_support": {
            "gamma_exact_fraction": {"numerator": 99, "denominator": 100},
            "reward_divisor": config.reward_divisor,
            "scaled_raw_reward_minimum": min(config.raw_reward_values)
            / config.reward_divisor,
            "scaled_raw_reward_maximum": max(config.raw_reward_values)
            / config.reward_divisor,
            "theoretical_raw_discounted_return_minimum": config.raw_return_minimum,
            "theoretical_raw_discounted_return_maximum": config.raw_return_maximum,
            "theoretical_scaled_discounted_return_minimum": {
                "numerator": -10,
                "denominator": 3,
            },
            "theoretical_scaled_discounted_return_maximum": (
                config.raw_return_maximum / config.reward_divisor
            ),
            "support_minimum": config.support_minimum,
            "support_maximum": config.support_maximum,
            "num_atoms": config.num_atoms,
            "coverage": "covers_theoretical_infinite_horizon_discounted_return",
            "rationale": (
                "Divide by the largest reward (30) to keep the positive return bound at "
                "100; [-4, 100] also covers the -10/3 negative bound."
            ),
        },
        "network": {
            "factorized_noisy": config.factorized_noisy,
            "dueling": config.dueling,
            "conv_channels": list(config.conv_channels),
            "conv_kernel_sizes": list(config.conv_kernel_sizes),
            "conv_strides": list(config.conv_strides),
            "conv_padding": config.conv_padding,
            "conv_derivation": (
                "three 3x3 valid convolutions replace Atari 8x8/4x4/3x3 kernels "
                "that do not fit a 9x9 aperture"
            ),
            "flattened_features": 576,
            "hidden_units": config.hidden_units,
            "noisy_sigma_zero": config.noisy_sigma_zero,
            "independent_layer_noise_subkeys": True,
            "evaluation_noise_disabled": True,
        },
        "optimizer": {
            "name": config.optimizer,
            "learning_rate": config.learning_rate,
            "beta1": config.adam_beta1,
            "beta2": config.adam_beta2,
            "epsilon": config.adam_epsilon,
        },
        "rng": {
            "required_inputs": ["environment_seed", "agent_seed"],
            "seed_domain": "uint31",
            "jax_prng_implementation": "threefry2x32",
            "environment_root": (
                "jax.random.key(environment_seed,impl=threefry2x32)"
            ),
            "environment_schedule": (
                "shared_bridge_direct_root_reset_then_transition_splits"
            ),
            "agent_namespace": "agent/adapted_full_rainbow",
            "agent_namespace_tag_uint32": _AGENT_NAMESPACE,
            "environment_root_consumed_by_core": False,
            "agent_root_initializes_and_drives_network": True,
            "equal_input_seed_values_allowed": True,
            "namespaced_agent_root_remains_disjoint_from_direct_environment_root": True,
        },
    }


_CANONICAL_CONFIG_BYTES: Final = _canonical_json(_full_rainbow_config_payload())
FULL_RAINBOW_CONFIG_SHA256: Final = (
    "835f02bdcf6844b7cd8c5e9fe33230a2a94f3a9c288c812cbfddf473c28b7e3f"
)
if not hmac.compare_digest(
    hashlib.sha256(_CANONICAL_CONFIG_BYTES).hexdigest(),
    FULL_RAINBOW_CONFIG_SHA256,
):
    raise AssertionError("canonical matched-v3 Full Rainbow configuration drifted")


def canonical_full_rainbow_config() -> dict[str, Any]:
    """Decode a detached snapshot from the authenticated configuration bytes."""

    try:
        decoded = json.loads(_CANONICAL_CONFIG_BYTES.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise FullRainbowContractError(
            "frozen Full Rainbow configuration bytes could not be decoded"
        ) from exc
    if type(decoded) is not dict:  # pragma: no cover
        raise FullRainbowContractError(
            "frozen Full Rainbow configuration must encode a plain object"
        )
    return cast(dict[str, Any], decoded)


def canonical_full_rainbow_config_bytes() -> bytes:
    """Return the exact authenticated configuration bytes."""

    return _CANONICAL_CONFIG_BYTES


_UPSTREAM_FILES: Final = (
    {
        "path": "LICENSE",
        "sha256": "e47b2783cb7131207707c35d0aea22277aa1beded6bf9d7c2436cd7de9462323",
    },
    {
        "path": "dopamine/jax/agents/full_rainbow/full_rainbow_agent.py",
        "sha256": "cc85222d9b60b6f05cbb8e6af170a57a3f74c20c9dd72067b70d8daf4cf50595",
    },
    {
        "path": "dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin",
        "sha256": "f926614f7c99ec248f3bafdbb920a7d8497476c0a27d5aad9ca8c69ca9ebc130",
    },
    {
        "path": "dopamine/jax/losses.py",
        "sha256": "42c10699bebf5b41b7bcd5cbeb18693c0f606f3bc427b988426368741e3cbd39",
    },
    {
        "path": "dopamine/jax/agents/dqn/dqn_agent.py",
        "sha256": "53a37912775c1fcce84f3c158c29fb9d63094ba8dc9f8a0c9c627e0f8c519dca",
    },
    {
        "path": "dopamine/jax/networks.py",
        "sha256": "fac813138454e2c947aca78a284b0e79b8f021beaf27b5f99981177ec8ca3bb9",
    },
    {
        "path": "dopamine/jax/agents/rainbow/rainbow_agent.py",
        "sha256": "02c90de41f68c18e66938bc9c5664a5e6154b8c67571114c8955d04a9e67cef8",
    },
    {
        "path": "dopamine/jax/replay_memory/accumulator.py",
        "sha256": "cfe4c849b2121f259fce5cd23e0a349f6ffba45f3c5c167dd63f36da2fc9cd25",
    },
    {
        "path": "dopamine/jax/replay_memory/samplers.py",
        "sha256": "de33adddd80fa4194e5eda14182f1eee50c65492c575e16e5c45630b9c75bb0b",
    },
)


def _operation_accounting(config: FullRainbowForagerConfig) -> dict[str, Any]:
    eligible_replay = config.horizon - config.update_horizon
    first_update_transition = config.minimum_replay_history + config.update_horizon + 1
    optimizer_updates = len(
        range(first_update_transition, config.horizon + 1, config.update_period)
    )
    first_target_refresh = (
        (first_update_transition + config.target_update_period - 1)
        // config.target_update_period
        * config.target_update_period
    )
    target_refreshes = len(
        range(first_target_refresh, config.horizon + 1, config.target_update_period)
    )
    return {
        "schedule_convention": (
            "one-based environment transition; replay has H-3 elements; update after "
            "replay_count>20000 when transition mod 4 is zero"
        ),
        "environment_interactions": config.horizon,
        "eligible_replay_transitions": eligible_replay,
        "first_optimizer_update_transition": first_update_transition,
        "optimizer_updates": optimizer_updates,
        "target_network_refreshes": target_refreshes,
        "replay_samples": optimizer_updates * config.batch_size,
        "c51_projections": optimizer_updates * config.batch_size,
        "priority_updates": optimizer_updates * config.batch_size,
    }


def _descriptor() -> dict[str, Any]:
    config = FullRainbowForagerConfig()
    return {
        "schema_version": FULL_RAINBOW_DESCRIPTOR_SCHEMA_VERSION,
        "candidate_id": "adapted_full_rainbow",
        "status": FULL_RAINBOW_ADAPTER_STATUS,
        "classification": "derived_exact_task_core_non_authorizing",
        "configuration": {
            "schema_version": FULL_RAINBOW_CONFIG_SCHEMA_VERSION,
            "sha256": FULL_RAINBOW_CONFIG_SHA256,
            "configuration_complete": True,
        },
        "source": {
            "repository_id": "dopamine",
            "canonical_url": "https://github.com/google/dopamine",
            "commit_git_sha1": "5873f5494ee0c2d7c016d0ab2ad530354fec59d0",
            "tree_git_sha1": "578408662e298d00e4e855f13f67dc08bd784e7c",
            "archive_sha256": (
                "bea46f755c86725d7ca90c531a08aad86cab62201ac2b9224c82f66dfada7456"
            ),
            "archive_size_bytes": 82_933_760,
            "license": "Apache-2.0",
            "relationship": "modified_derivative",
            "attribution_preserved_in_source_header": True,
            "upstream_review_anchors_bound": True,
            "source_closure_bound": False,
            "files": [dict(item) for item in _UPSTREAM_FILES],
        },
        "preserved_components": [
            "three_step_returns",
            "proportional_prioritized_replay",
            "inverse_square_root_importance_weights",
            "loss_derived_priority_updates",
            "c51_frozen_support_projection",
            "double_q_online_selection_target_evaluation",
            "factorized_noisy_train_eval_semantics",
            "dueling_value_advantage_heads",
        ],
        "exact_operation_accounting": _operation_accounting(config),
        "exact_resource_accounting": {
            "online_parameter_scalars": EXPECTED_ONLINE_PARAMETER_SCALARS,
            "target_parameter_scalars": EXPECTED_ONLINE_PARAMETER_SCALARS,
            "adam_moment_scalars": EXPECTED_ADAM_MOMENT_SCALARS,
            "adam_count_scalars": 1,
            "parameter_dtype": "float32",
            "adam_count_dtype": "int32",
            "parameter_target_optimizer_bytes": (
                EXPECTED_PARAMETER_TARGET_OPTIMIZER_BYTES
            ),
            "configured_replay_capacity_transitions": config.replay_capacity,
            "maximum_resident_transitions_at_bound_horizon": (
                config.horizon - config.update_horizon
            ),
            "replay_storage_bytes": None,
            "replay_storage_bytes_reason": (
                "full runner serialization/dtype layout is not implemented"
            ),
        },
        "runner": {
            "core_update_primitive_implemented": True,
            "full_horizon_runner_implemented": False,
            "result_writer_implemented": False,
            "qualification_receipt_implemented": False,
            "blocker": "full Foragax execution and result transport remain unimplemented",
        },
        "claims": {
            "configuration_complete": True,
            "core_implementation_complete": True,
            "execution_ready": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "authority_granted": False,
        },
        "limitations": [
            "No full Foragax runner, checkpointing, or result writer is included.",
            "No runtime equivalence or environment-RNG qualification receipt exists.",
            (
                "The 9x9 convolutional encoder is a task-derived modification, not "
                "upstream Atari code."
            ),
            "Core unit tests are engineering checks and never scientific evidence.",
        ],
    }


_CANONICAL_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
FULL_RAINBOW_DESCRIPTOR_SHA256: Final = (
    "5436200c47e1b003b0371c30606b52163b4c42427fa84e2fe2f4b2b2273ccae2"
)
if not hmac.compare_digest(
    hashlib.sha256(_CANONICAL_DESCRIPTOR_BYTES).hexdigest(),
    FULL_RAINBOW_DESCRIPTOR_SHA256,
):
    raise AssertionError("canonical matched-v3 Full Rainbow descriptor drifted")


def matched_v3_full_rainbow_descriptor() -> dict[str, Any]:
    """Return a detached, canonical non-authorizing adapter descriptor."""

    return cast(
        dict[str, Any],
        json.loads(_CANONICAL_DESCRIPTOR_BYTES.decode("ascii")),
    )


def canonical_matched_v3_full_rainbow_descriptor_bytes() -> bytes:
    """Return exact canonical bytes for the frozen source/config descriptor."""

    return bytes(_CANONICAL_DESCRIPTOR_BYTES)


def _assert_plain_unaliased_json(value: object) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise FullRainbowContractError("descriptor contains aliased containers")
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise FullRainbowContractError("descriptor contains a non-string key")
            pending.extend(mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise FullRainbowContractError("descriptor contains aliased containers")
            seen.add(identity)
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise FullRainbowContractError("descriptor contains a non-finite float")
        elif type(item) not in {str, int, bool, type(None)}:
            raise FullRainbowContractError("descriptor contains a non-plain JSON value")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullRainbowContractError("descriptor contains duplicate JSON keys")
        result[key] = value
    return result


def validate_matched_v3_full_rainbow_descriptor(
    value: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Accept only the exact canonical descriptor; no readiness inference occurs."""

    if type(value) is bytes:
        raw = value
        if len(raw) > _MAX_CANONICAL_BYTES:
            raise FullRainbowContractError("descriptor exceeds its byte limit")
        try:
            decoded = json.loads(
                raw.decode("ascii"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    FullRainbowContractError(
                        f"descriptor contains forbidden JSON constant {token}"
                    )
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullRainbowContractError("descriptor is not strict JSON") from exc
        if not hmac.compare_digest(raw, _CANONICAL_DESCRIPTOR_BYTES):
            raise FullRainbowContractError("descriptor bytes are not exact canonical bytes")
    elif type(value) is dict:
        decoded = value
    else:
        raise FullRainbowContractError("descriptor must be exact bytes or a plain object")
    _assert_plain_unaliased_json(decoded)
    raw = _canonical_json(decoded)
    if not hmac.compare_digest(raw, _CANONICAL_DESCRIPTOR_BYTES):
        raise FullRainbowContractError("descriptor identity does not match the frozen contract")
    return cast(dict[str, Any], copy.deepcopy(decoded))


def assert_full_rainbow_execution_ready() -> None:
    """Fail closed until a separate qualified full-horizon runner is bound."""

    raise FullRainbowExecutionBlockedError(
        "Full Rainbow core is implemented, but the full Foragax runner is unimplemented "
        "and execution is neither ready nor authorized"
    )


__all__ = [
    "EXPECTED_ADAM_MOMENT_SCALARS",
    "EXPECTED_ONLINE_PARAMETER_SCALARS",
    "EXPECTED_PARAMETER_TARGET_OPTIMIZER_BYTES",
    "FULL_RAINBOW_ADAPTER_STATUS",
    "FULL_RAINBOW_CONFIG_SCHEMA_VERSION",
    "FULL_RAINBOW_CONFIG_SHA256",
    "FULL_RAINBOW_DESCRIPTOR_SCHEMA_VERSION",
    "FULL_RAINBOW_DESCRIPTOR_SHA256",
    "FactorizedGaussianNoise",
    "FullRainbowContractError",
    "FullRainbowCoreState",
    "FullRainbowExecutionBlockedError",
    "FullRainbowForagerConfig",
    "FullRainbowNetworkOutput",
    "FullRainbowReplayBatch",
    "FullRainbowSeedRoots",
    "FullRainbowTrainMetrics",
    "ProportionalReplaySample",
    "ThreeStepReturn",
    "apply_full_rainbow_network",
    "assert_full_rainbow_execution_ready",
    "canonical_full_rainbow_config",
    "canonical_full_rainbow_config_bytes",
    "canonical_matched_v3_full_rainbow_descriptor_bytes",
    "double_q_c51_target",
    "factorized_gaussian_noise",
    "frozen_support",
    "full_rainbow_seed_roots",
    "importance_sampling_weights",
    "initialize_full_rainbow_core",
    "matched_v3_full_rainbow_descriptor",
    "parameter_scalar_count",
    "priority_updates",
    "project_c51_distribution",
    "proportional_sampling_probabilities",
    "sample_proportional_replay",
    "sync_full_rainbow_target",
    "three_step_return",
    "train_full_rainbow_step",
    "validate_matched_v3_full_rainbow_descriptor",
    "validate_replay_batch",
]
