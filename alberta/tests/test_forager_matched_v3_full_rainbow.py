"""Cheap contract tests for the derived matched-v3 Full Rainbow core."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_v3_full_rainbow as rainbow


def _tree_arrays(value: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(value)]


def _assert_trees_equal(left: Any, right: Any) -> None:
    left_leaves = _tree_arrays(left)
    right_leaves = _tree_arrays(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


@pytest.mark.unit
def test_exact_config_support_and_source_faithful_features_are_bound() -> None:
    config = rainbow.FullRainbowForagerConfig()
    payload = rainbow.canonical_full_rainbow_config()

    assert payload["task"] == {
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "observation_type": "color",
        "observation_shape": [9, 9, 3],
        "num_actions": 4,
        "horizon": 499_712,
        "raw_reward_values": [-1, 0, 1, 30],
    }
    assert payload["algorithm"]["update_horizon"] == 3
    assert payload["algorithm"]["prioritized_replay"] is True
    assert payload["algorithm"]["distributional_c51"] is True
    assert payload["algorithm"]["double_q"] is True
    assert payload["network"]["factorized_noisy"] is True
    assert payload["network"]["dueling"] is True
    assert config.raw_return_minimum / config.reward_divisor >= config.support_minimum
    assert config.raw_return_maximum / config.reward_divisor <= config.support_maximum
    support = rainbow.frozen_support(config)
    assert support.shape == (51,)
    assert float(support[0]) == pytest.approx(-4.0)
    assert float(support[-1]) == pytest.approx(100.0)
    assert hashlib.sha256(rainbow.canonical_full_rainbow_config_bytes()).hexdigest() == (
        rainbow.FULL_RAINBOW_CONFIG_SHA256
    )
    assert rainbow.FULL_RAINBOW_CONFIG_SHA256 == (
        "835f02bdcf6844b7cd8c5e9fe33230a2a94f3a9c288c812cbfddf473c28b7e3f"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_actions", 5),
        ("update_horizon", 1),
        ("prioritized_replay", False),
        ("distributional_c51", False),
        ("double_q", False),
        ("factorized_noisy", False),
        ("dueling", False),
        ("num_atoms", 50),
        ("gamma", 0.9),
        ("reward_divisor", 1.0),
        ("support_minimum", -3.0),
        ("support_maximum", 99.0),
        ("observation_shape", (9, 9, 4)),
        ("replay_scheme", "uniform"),
    ],
)
def test_config_rejects_feature_removal_or_exact_task_drift(field: str, value: object) -> None:
    with pytest.raises(rainbow.FullRainbowContractError):
        dataclasses.replace(
            rainbow.FullRainbowForagerConfig(), **{field: value}  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_explicit_environment_and_agent_roots_are_independent_and_strict() -> None:
    first = rainbow.full_rainbow_seed_roots(environment_seed=17, agent_seed=23)
    agent_changed = rainbow.full_rainbow_seed_roots(environment_seed=17, agent_seed=29)
    environment_changed = rainbow.full_rainbow_seed_roots(
        environment_seed=19, agent_seed=23
    )

    np.testing.assert_array_equal(
        jax.random.key_data(first.environment),
        jax.random.key_data(agent_changed.environment),
    )
    assert not np.array_equal(
        jax.random.key_data(first.agent), jax.random.key_data(agent_changed.agent)
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first.agent),
        jax.random.key_data(environment_changed.agent),
    )
    assert not np.array_equal(
        jax.random.key_data(first.environment),
        jax.random.key_data(environment_changed.environment),
    )
    assert not np.array_equal(
        jax.random.key_data(first.environment), jax.random.key_data(first.agent)
    )
    np.testing.assert_array_equal(
        jax.random.key_data(first.environment),
        jax.random.key_data(jax.random.key(17, impl="threefry2x32")),
    )
    for bad in (-1, 1 << 31, True, 1.0):
        with pytest.raises(rainbow.FullRainbowContractError, match="uint31"):
            rainbow.full_rainbow_seed_roots(environment_seed=bad, agent_seed=1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_three_step_return_scales_raw_rewards_and_stops_at_terminal() -> None:
    config = rainbow.FullRainbowForagerConfig()
    continuing = rainbow.three_step_return(
        config,
        raw_rewards=(30, 1, -1),
        terminals=(False, False, False),
    )
    expected = (30.0 + 0.99 * 1.0 - 0.99**2) / 30.0
    assert continuing.scaled_return == pytest.approx(expected)
    assert continuing.bootstrap_discount == pytest.approx(0.99**3)
    assert continuing.terminal is False

    terminal = rainbow.three_step_return(
        config,
        raw_rewards=(1, 30, -1),
        terminals=(False, True, False),
    )
    assert terminal.scaled_return == pytest.approx((1.0 + 0.99 * 30.0) / 30.0)
    assert terminal.bootstrap_discount == 0.0
    assert terminal.terminal is True
    with pytest.raises(rainbow.FullRainbowContractError, match="raw reward"):
        rainbow.three_step_return(
            config,
            raw_rewards=(2, 0, 0),
            terminals=(False, False, False),
        )


@pytest.mark.unit
def test_proportional_replay_weights_and_priority_updates_match_dopamine_semantics() -> None:
    probabilities = rainbow.proportional_sampling_probabilities(
        jnp.asarray([1.0, 2.0, 7.0], dtype=jnp.float32)
    )
    np.testing.assert_allclose(probabilities, [0.1, 0.2, 0.7], rtol=1e-6)
    uniform = rainbow.proportional_sampling_probabilities(jnp.zeros((3,)))
    np.testing.assert_allclose(uniform, [1 / 3, 1 / 3, 1 / 3], rtol=1e-6)

    weights = rainbow.importance_sampling_weights(probabilities)
    expected = 1.0 / np.sqrt(np.asarray([0.1, 0.2, 0.7]) + 1e-10)
    expected /= expected.max()
    np.testing.assert_allclose(weights, expected, rtol=1e-6)
    losses = jnp.asarray([0.0, 0.25, 4.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        rainbow.priority_updates(losses),
        np.sqrt(np.asarray(losses) + 1e-10),
        rtol=1e-6,
    )
    with pytest.raises(rainbow.FullRainbowContractError):
        rainbow.proportional_sampling_probabilities(jnp.asarray([1.0, -1.0]))
    with pytest.raises(rainbow.FullRainbowContractError):
        rainbow.importance_sampling_weights(jnp.asarray([0.5, 0.0]))


@pytest.mark.unit
def test_proportional_replay_draw_is_with_replacement_and_key_deterministic() -> None:
    priorities = jnp.asarray([0.0, 0.0, 7.0], dtype=jnp.float32)
    first = rainbow.sample_proportional_replay(
        rainbow.FullRainbowForagerConfig(), jax.random.key(3), priorities
    )
    repeat = rainbow.sample_proportional_replay(
        rainbow.FullRainbowForagerConfig(), jax.random.key(3), priorities
    )
    np.testing.assert_array_equal(first.indices, np.full((32,), 2))
    np.testing.assert_array_equal(first.indices, repeat.indices)
    np.testing.assert_array_equal(first.sampling_probabilities, np.ones((32,)))
    np.testing.assert_array_equal(first.importance_weights, np.ones((32,)))


@pytest.mark.unit
def test_c51_projection_preserves_mass_and_clips_to_frozen_support() -> None:
    support = jnp.linspace(-4.0, 100.0, 51)
    source = jnp.zeros((51,), dtype=jnp.float32).at[25].set(1.0)
    projected = rainbow.project_c51_distribution(
        target_support=jnp.asarray([1_000.0] * 51),
        probabilities=source,
        support=support,
    )
    assert float(jnp.sum(projected)) == pytest.approx(1.0)
    assert float(projected[-1]) == pytest.approx(1.0)
    np.testing.assert_allclose(
        rainbow.project_c51_distribution(
            target_support=support,
            probabilities=jnp.ones((51,)) / 51,
            support=support,
        ),
        np.ones((51,)) / 51,
        atol=2e-6,
    )


@pytest.mark.unit
def test_double_q_selects_online_action_but_projects_target_distribution() -> None:
    support = jnp.asarray([-1.0, 0.0, 1.0])
    online_q = jnp.asarray([0.0, 5.0, 1.0, 2.0])
    target_probabilities = jnp.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    action, target = rainbow.double_q_c51_target(
        online_next_q_values=online_q,
        target_next_probabilities=target_probabilities,
        scaled_n_step_reward=jnp.asarray(0.0),
        bootstrap_discount=jnp.asarray(1.0),
        support=support,
    )
    assert int(action) == 1
    np.testing.assert_allclose(target, [1.0, 0.0, 0.0])


@pytest.mark.unit
def test_factorized_noise_is_rank_one_and_eval_mode_is_noise_free() -> None:
    training = rainbow.factorized_gaussian_noise(
        jax.random.key(9), input_features=8, output_features=5, eval_mode=False
    )
    evaluation = rainbow.factorized_gaussian_noise(
        jax.random.key(10), input_features=8, output_features=5, eval_mode=True
    )
    assert training.weight.shape == (8, 5)
    assert training.bias.shape == (5,)
    assert np.linalg.matrix_rank(np.asarray(training.weight), tol=1e-6) == 1
    np.testing.assert_array_equal(evaluation.weight, np.zeros((8, 5)))
    np.testing.assert_array_equal(evaluation.bias, np.zeros((5,)))


@pytest.mark.unit
def test_network_is_dueling_noisy_and_initialization_uses_only_agent_root() -> None:
    config = rainbow.FullRainbowForagerConfig()
    first = rainbow.initialize_full_rainbow_core(
        config, environment_seed=17, agent_seed=23
    )
    repeat = rainbow.initialize_full_rainbow_core(
        config, environment_seed=17, agent_seed=23
    )
    environment_changed = rainbow.initialize_full_rainbow_core(
        config, environment_seed=19, agent_seed=23
    )
    agent_changed = rainbow.initialize_full_rainbow_core(
        config, environment_seed=17, agent_seed=29
    )
    _assert_trees_equal(first.online_params, repeat.online_params)
    _assert_trees_equal(first.online_params, environment_changed.online_params)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            _tree_arrays(first.online_params),
            _tree_arrays(agent_changed.online_params),
            strict=True,
        )
    )
    assert rainbow.parameter_scalar_count(first.online_params) == 908_798
    assert rainbow.parameter_scalar_count(first.target_params) == 908_798
    optimizer_leaves = jax.tree_util.tree_leaves(first.optimizer_state)
    assert optimizer_leaves[0].shape == ()
    assert optimizer_leaves[0].dtype == jnp.int32
    assert sum(int(leaf.size) for leaf in optimizer_leaves[1:]) == 1_817_596
    resident_bytes = sum(
        int(leaf.size * leaf.dtype.itemsize)
        for tree in (first.online_params, first.target_params, first.optimizer_state)
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    assert resident_bytes == 14_540_772
    assert first.online_params["advantage"]["kernel_mu"].shape == (512, 4 * 51)
    assert first.online_params["value"]["kernel_mu"].shape == (512, 51)

    observation = jnp.zeros(config.observation_shape, dtype=jnp.float32)
    eval_a = rainbow.apply_full_rainbow_network(
        config, first.online_params, observation, jax.random.key(1), eval_mode=True
    )
    eval_b = rainbow.apply_full_rainbow_network(
        config, first.online_params, observation, jax.random.key(2), eval_mode=True
    )
    train_a = rainbow.apply_full_rainbow_network(
        config, first.online_params, observation, jax.random.key(1), eval_mode=False
    )
    train_b = rainbow.apply_full_rainbow_network(
        config, first.online_params, observation, jax.random.key(2), eval_mode=False
    )
    np.testing.assert_array_equal(eval_a.logits, eval_b.logits)
    assert not np.array_equal(train_a.logits, train_b.logits)
    assert eval_a.logits.shape == (4, 51)
    assert eval_a.probabilities.shape == (4, 51)
    np.testing.assert_allclose(eval_a.probabilities.sum(axis=-1), np.ones(4), atol=1e-6)
    assert "advantage" in first.online_params
    assert "value" in first.online_params


@pytest.mark.unit
@pytest.mark.slow
def test_tiny_distributional_update_is_deterministic_and_updates_priorities() -> None:
    config = rainbow.FullRainbowForagerConfig()
    state = rainbow.initialize_full_rainbow_core(
        config, environment_seed=17, agent_seed=23
    )
    states = jnp.zeros((config.batch_size, *config.observation_shape), dtype=jnp.float32)
    next_states = states.at[1, 4, 4, 1].set(1.0)
    batch = rainbow.FullRainbowReplayBatch(
        states=states,
        actions=jnp.arange(config.batch_size, dtype=jnp.int32) % config.num_actions,
        next_states=next_states,
        scaled_n_step_rewards=jnp.zeros((config.batch_size,), dtype=jnp.float32),
        bootstrap_discounts=jnp.full(
            (config.batch_size,), config.gamma**3, dtype=jnp.float32
        ),
        sampling_probabilities=jnp.full(
            (config.batch_size,), 1.0 / config.batch_size, dtype=jnp.float32
        ),
    )
    updated, metrics = rainbow.train_full_rainbow_step(config, state, batch)
    repeated, repeated_metrics = rainbow.train_full_rainbow_step(config, state, batch)

    assert updated.optimizer_updates == 1
    np.testing.assert_array_equal(
        jax.random.key_data(updated.agent_rng),
        jax.random.key_data(jax.random.split(state.agent_rng, 3)[2]),
    )
    assert metrics.per_example_loss.shape == (config.batch_size,)
    assert bool(jnp.all(jnp.isfinite(metrics.per_example_loss)))
    assert bool(jnp.all(metrics.updated_priorities > 0.0))
    _assert_trees_equal(updated.online_params, repeated.online_params)
    _assert_trees_equal(updated.target_params, state.target_params)
    np.testing.assert_array_equal(metrics.per_example_loss, repeated_metrics.per_example_loss)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            _tree_arrays(state.online_params),
            _tree_arrays(updated.online_params),
            strict=True,
        )
    )


@pytest.mark.unit
def test_replay_batch_validation_is_fail_closed() -> None:
    config = rainbow.FullRainbowForagerConfig()
    state = rainbow.initialize_full_rainbow_core(
        config, environment_seed=1, agent_seed=2
    )
    valid = rainbow.FullRainbowReplayBatch(
        states=jnp.zeros((config.batch_size, 9, 9, 3)),
        actions=jnp.zeros((config.batch_size,), dtype=jnp.int32),
        next_states=jnp.zeros((config.batch_size, 9, 9, 3)),
        scaled_n_step_rewards=jnp.zeros((config.batch_size,)),
        bootstrap_discounts=jnp.full((config.batch_size,), config.gamma**3),
        sampling_probabilities=jnp.full(
            (config.batch_size,), 1.0 / config.batch_size
        ),
    )
    rainbow.validate_replay_batch(config, valid)
    for mutation in (
        dataclasses.replace(valid, states=jnp.zeros((1, 9, 9, 3))),
        dataclasses.replace(valid, actions=jnp.full((config.batch_size,), 4)),
        dataclasses.replace(valid, states=jnp.zeros((config.batch_size, 9, 9, 4))),
        dataclasses.replace(
            valid, states=jnp.full((config.batch_size, 9, 9, 3), 0.5)
        ),
        dataclasses.replace(
            valid,
            states=jnp.zeros((config.batch_size, 9, 9, 3))
            .at[0, 4, 4, :2]
            .set(1.0),
        ),
        dataclasses.replace(
            valid, sampling_probabilities=jnp.zeros((config.batch_size,))
        ),
        dataclasses.replace(
            valid,
            scaled_n_step_rewards=jnp.full((config.batch_size,), jnp.nan),
        ),
        dataclasses.replace(
            valid, bootstrap_discounts=jnp.full((config.batch_size,), 0.5)
        ),
    ):
        with pytest.raises(rainbow.FullRainbowContractError):
            rainbow.train_full_rainbow_step(config, state, mutation)


@pytest.mark.unit
def test_descriptor_is_content_addressed_non_authorizing_and_exactly_accounted() -> None:
    descriptor = rainbow.matched_v3_full_rainbow_descriptor()
    assert descriptor["status"] == "implemented_unqualified"
    assert descriptor["candidate_id"] == "adapted_full_rainbow"
    assert descriptor["source"]["relationship"] == "modified_derivative"
    assert descriptor["source"]["license"] == "Apache-2.0"
    assert descriptor["source"]["upstream_review_anchors_bound"] is True
    assert descriptor["source"]["source_closure_bound"] is False
    pins = {item["path"]: item["sha256"] for item in descriptor["source"]["files"]}
    assert pins["dopamine/jax/agents/full_rainbow/full_rainbow_agent.py"] == (
        "cc85222d9b60b6f05cbb8e6af170a57a3f74c20c9dd72067b70d8daf4cf50595"
    )
    assert pins["dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin"] == (
        "f926614f7c99ec248f3bafdbb920a7d8497476c0a27d5aad9ca8c69ca9ebc130"
    )
    assert pins["dopamine/jax/losses.py"] == (
        "42c10699bebf5b41b7bcd5cbeb18693c0f606f3bc427b988426368741e3cbd39"
    )
    assert pins["dopamine/jax/agents/dqn/dqn_agent.py"] == (
        "53a37912775c1fcce84f3c158c29fb9d63094ba8dc9f8a0c9c627e0f8c519dca"
    )
    accounting = descriptor["exact_operation_accounting"]
    assert accounting["environment_interactions"] == 499_712
    assert accounting["eligible_replay_transitions"] == 499_709
    assert accounting["optimizer_updates"] == 119_928
    assert accounting["target_network_refreshes"] == 60
    assert accounting["replay_samples"] == 3_837_696
    resources = descriptor["exact_resource_accounting"]
    assert resources["online_parameter_scalars"] == 908_798
    assert resources["target_parameter_scalars"] == 908_798
    assert resources["adam_moment_scalars"] == 1_817_596
    assert resources["parameter_target_optimizer_bytes"] == 14_540_772
    assert descriptor["runner"]["full_horizon_runner_implemented"] is False
    assert descriptor["claims"] == {
        "configuration_complete": True,
        "core_implementation_complete": True,
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
    canonical = rainbow.canonical_matched_v3_full_rainbow_descriptor_bytes()
    assert hashlib.sha256(canonical).hexdigest() == rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256
    assert rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256 == (
        "5436200c47e1b003b0371c30606b52163b4c42427fa84e2fe2f4b2b2273ccae2"
    )
    assert rainbow.validate_matched_v3_full_rainbow_descriptor(canonical) == descriptor


@pytest.mark.unit
def test_descriptor_getter_ignores_mutated_private_upstream_pin_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = rainbow.canonical_matched_v3_full_rainbow_descriptor_bytes()
    monkeypatch.setitem(rainbow._UPSTREAM_FILES[0], "sha256", "0" * 64)

    descriptor = rainbow.matched_v3_full_rainbow_descriptor()
    assert descriptor == json.loads(raw)
    assert rainbow.validate_matched_v3_full_rainbow_descriptor(descriptor) == descriptor


@pytest.mark.unit
def test_config_getter_ignores_mutated_private_construction_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = rainbow.canonical_full_rainbow_config_bytes()
    expected = json.loads(raw)
    monkeypatch.setattr(rainbow, "_AGENT_NAMESPACE", 0)

    assert rainbow.canonical_full_rainbow_config() == expected
    assert rainbow.canonical_full_rainbow_config_bytes() == raw


@pytest.mark.unit
def test_descriptor_validator_rejects_mutation_noncanonical_json_and_aliases() -> None:
    descriptor = rainbow.matched_v3_full_rainbow_descriptor()
    mutation = copy.deepcopy(descriptor)
    mutation["claims"]["execution_ready"] = 0
    with pytest.raises(rainbow.FullRainbowContractError):
        rainbow.validate_matched_v3_full_rainbow_descriptor(mutation)

    canonical = rainbow.canonical_matched_v3_full_rainbow_descriptor_bytes()
    noncanonical = json.dumps(descriptor, indent=2, sort_keys=True).encode()
    assert noncanonical != canonical
    with pytest.raises(rainbow.FullRainbowContractError, match="canonical"):
        rainbow.validate_matched_v3_full_rainbow_descriptor(noncanonical)
    duplicate = canonical[:-1] + b',"status":"implemented_unqualified"}'
    with pytest.raises(rainbow.FullRainbowContractError):
        rainbow.validate_matched_v3_full_rainbow_descriptor(duplicate)

    shared: list[object] = []
    aliased = copy.deepcopy(descriptor)
    aliased["alias_a"] = shared
    aliased["alias_b"] = shared
    with pytest.raises(rainbow.FullRainbowContractError):
        rainbow.validate_matched_v3_full_rainbow_descriptor(aliased)


@pytest.mark.unit
def test_execution_guard_never_treats_core_completion_as_authority() -> None:
    with pytest.raises(rainbow.FullRainbowExecutionBlockedError, match="runner"):
        rainbow.assert_full_rainbow_execution_ready()
