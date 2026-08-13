"""Synthetic contracts for the non-authorizing matched-v3 Full Rainbow runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_full_rainbow as core
from alberta_framework.benchmarks import forager_matched_v3_full_rainbow_runner as runner

pytestmark = pytest.mark.unit


def _observation(index: int) -> jnp.ndarray:
    value = np.zeros((9, 9, 3), dtype=np.float32)
    flat = index % (9 * 9 * 3)
    row, remainder = divmod(flat, 9 * 3)
    column, channel = divmod(remainder, 3)
    value[row, column, channel] = 1.0
    return jnp.asarray(value)


def _tiny_schedule() -> runner.FullRainbowRunnerSchedule:
    return runner.FullRainbowRunnerSchedule(
        horizon=12,
        replay_capacity=5,
        batch_size=2,
        minimum_replay_history=2,
        update_period=2,
        target_update_period=4,
        page_size=2,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeEnvironmentState:
    environment_seed: int
    observation: jax.Array
    reset_count: int
    step_count: int
    environment_key: jax.Array

    @property
    def environment_key_use_count(self) -> int:
        return self.reset_count + self.step_count


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeTransition:
    state: _FakeEnvironmentState
    action: int
    reward: int
    done: bool = False
    truncated: bool = False
    info_validated: bool = True

    @property
    def observation(self) -> jax.Array:
        return self.state.observation


class _FakeEnvironmentRuntime:
    def __init__(self, rewards: tuple[int, ...]) -> None:
        self.rewards = rewards
        self.reset_keys: list[tuple[int, int]] = []
        self.step_keys: list[tuple[int, int]] = []
        self.actions: list[int] = []
        self.used_states: list[_FakeEnvironmentState] = []

    def initialize(self, environment_seed: object) -> _FakeEnvironmentState:
        assert type(environment_seed) is int
        carry, reset_key = jr.split(
            jr.key(environment_seed, impl="threefry2x32")
        )
        self.reset_keys.append(
            cast(
                tuple[int, int],
                tuple(int(value) for value in np.asarray(jr.key_data(reset_key))),
            )
        )
        return _FakeEnvironmentState(
            environment_seed=environment_seed,
            observation=_observation(0),
            reset_count=1,
            step_count=0,
            environment_key=carry,
        )

    def step(self, state: object, action: object) -> _FakeTransition:
        assert type(state) is _FakeEnvironmentState
        assert all(state is not used_state for used_state in self.used_states)
        self.used_states.append(state)
        assert type(action) is int
        carry, step_key = jr.split(state.environment_key)
        self.step_keys.append(
            cast(
                tuple[int, int],
                tuple(int(value) for value in np.asarray(jr.key_data(step_key))),
            )
        )
        self.actions.append(action)
        next_count = state.step_count + 1
        return _FakeTransition(
            state=_FakeEnvironmentState(
                environment_seed=state.environment_seed,
                observation=_observation(next_count),
                reset_count=1,
                step_count=next_count,
                environment_key=carry,
            ),
            action=action,
            reward=self.rewards[(next_count - 1) % len(self.rewards)],
        )


class _ReusingEnvironmentRuntime(_FakeEnvironmentRuntime):
    def step(self, state: object, action: object) -> _FakeTransition:
        transition = super().step(state, action)
        assert type(state) is _FakeEnvironmentState
        return dataclasses.replace(transition, state=state)


@dataclasses.dataclass(slots=True)
class _FakeCoreLog:
    action_noise_keys: list[tuple[int, int]]
    update_input_keys: list[tuple[int, int]]
    sync_updates: list[int]


def _key_words(key: Any) -> tuple[int, int]:
    words = np.asarray(jr.key_data(key))
    return int(words[0]), int(words[1])


def _fake_dependencies(
    runtime: _FakeEnvironmentRuntime,
) -> tuple[runner.FullRainbowRunnerDependencies, _FakeCoreLog]:
    log = _FakeCoreLog([], [], [])

    def initialize_core(
        config: core.FullRainbowForagerConfig,
        environment_seed: int,
        agent_seed: int,
    ) -> core.FullRainbowCoreState:
        roots = core.full_rainbow_seed_roots(
            environment_seed=environment_seed,
            agent_seed=agent_seed,
        )
        next_agent, _, _ = jr.split(roots.agent, 3)
        value = jnp.asarray(float(agent_seed), dtype=jnp.float32)
        return core.FullRainbowCoreState(
            online_params=value,
            target_params=value,
            optimizer_state=jnp.asarray(0, dtype=jnp.int32),
            environment_rng=roots.environment,
            agent_rng=next_agent,
            optimizer_updates=0,
        )

    def action_q_values(
        config: core.FullRainbowForagerConfig,
        state: core.FullRainbowCoreState,
        observation: jax.Array,
        noise_key: jax.Array,
    ) -> jax.Array:
        del config, state, observation
        log.action_noise_keys.append(_key_words(noise_key))
        action = int(np.asarray(jr.key_data(noise_key))[0]) % 4
        return jnp.zeros((4,), dtype=jnp.float32).at[action].set(1.0)

    def update_core(
        config: core.FullRainbowForagerConfig,
        state: core.FullRainbowCoreState,
        batch: core.FullRainbowReplayBatch,
    ) -> tuple[core.FullRainbowCoreState, core.FullRainbowTrainMetrics]:
        del config
        log.update_input_keys.append(_key_words(state.agent_rng))
        _, _, next_agent = jr.split(state.agent_rng, 3)
        batch_size = int(batch.actions.shape[0])
        losses = jnp.arange(1, batch_size + 1, dtype=jnp.float32)
        priorities = jnp.sqrt(losses + 1e-10)
        weights = core.importance_sampling_weights(batch.sampling_probabilities)
        updated = dataclasses.replace(
            state,
            online_params=state.online_params + jnp.float32(1.0),
            optimizer_state=state.optimizer_state + jnp.int32(1),
            agent_rng=next_agent,
            optimizer_updates=state.optimizer_updates + 1,
        )
        return updated, core.FullRainbowTrainMetrics(
            mean_weighted_loss=jnp.mean(weights * losses),
            per_example_loss=losses,
            importance_weights=weights,
            updated_priorities=priorities,
            double_q_actions=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def sync_target(state: core.FullRainbowCoreState) -> core.FullRainbowCoreState:
        log.sync_updates.append(state.optimizer_updates)
        return dataclasses.replace(state, target_params=state.online_params)

    dependencies = runner.FullRainbowRunnerDependencies(
        dependency_identity="synthetic_full_rainbow_dependencies_v1",
        environment_runtime=runtime,
        step_environment=runtime.step,
        initialize_core=initialize_core,
        action_q_values=action_q_values,
        update_core=update_core,
        sync_target=sync_target,
        runtime_identity={
            "backend": "synthetic",
            "runtime_qualified": False,
            "foragax_runtime_parity_executed": False,
        },
        compiled_action_kernel=False,
        compiled_update_kernel=False,
    )
    return dependencies, log


def _replace_dependencies(
    dependencies: runner.FullRainbowRunnerDependencies, **changes: Any
) -> runner.FullRainbowRunnerDependencies:
    return dataclasses.replace(
        dependencies,
        runtime_identity=dict(dependencies.runtime_identity),
        **changes,
    )


def _rehash_receipt(receipt: dict[str, Any]) -> bytes:
    receipt.pop("receipt_body_sha256", None)
    receipt["receipt_body_sha256"] = hashlib.sha256(
        runner.canonical_full_rainbow_receipt_body_bytes(receipt)
    ).hexdigest()
    return json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()


def _accumulated(index: int, *, reward: int = 0) -> runner.FullRainbowAccumulatedTransition:
    return runner.FullRainbowAccumulatedTransition(
        state=_observation(index),
        action=index % 4,
        next_state=_observation(index + 3),
        scaled_n_step_reward=float(reward) / 30.0,
        bootstrap_discount=0.99**3,
        source_transition=index + 1,
        available_after_transition=index + 4,
    )


def test_production_schedule_has_exact_source_accounting_without_execution() -> None:
    schedule = runner.production_full_rainbow_schedule()
    accounting = runner.full_rainbow_schedule_accounting(schedule)

    assert dataclasses.asdict(schedule) == {
        "horizon": 499_712,
        "update_horizon": 3,
        "replay_capacity": 1_000_000,
        "batch_size": 32,
        "minimum_replay_history": 20_000,
        "update_period": 4,
        "target_update_period": 8_000,
        "page_size": 4_096,
    }
    assert accounting == {
        "environment_interactions": 499_712,
        "first_replay_insertion_transition": 4,
        "replay_insertions": 499_709,
        "maximum_replay_residency": 499_709,
        "replay_evictions": 0,
        "first_optimizer_update_transition": 20_004,
        "optimizer_updates": 119_928,
        "first_target_sync_transition": 24_000,
        "target_syncs": 60,
        "replay_samples": 3_837_696,
        "priority_update_values": 3_837_696,
    }


def test_accumulator_deliberately_waits_until_transition_four() -> None:
    accumulator = runner.FullRainbowThreeStepAccumulator(
        core.FullRainbowForagerConfig()
    )
    rewards = (30, 1, -1, 30, 0)
    outputs: list[runner.FullRainbowAccumulatedTransition | None] = []
    for index, reward in enumerate(rewards):
        outputs.append(
            accumulator.accumulate(
                observation=_observation(index),
                action=index % 4,
                raw_reward=reward,
                done=False,
                truncated=False,
            )
        )

    assert outputs[:3] == [None, None, None]
    first = outputs[3]
    second = outputs[4]
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.state, _observation(0))
    np.testing.assert_array_equal(first.next_state, _observation(3))
    assert first.action == 0
    assert first.source_transition == 1
    assert first.available_after_transition == 4
    assert first.scaled_n_step_reward == pytest.approx(
        (30 + 0.99 * 1 - 0.99**2) / 30
    )
    assert first.bootstrap_discount == pytest.approx(0.99**3)
    assert second.source_transition == 2
    assert second.available_after_transition == 5
    assert accumulator.raw_transition_count == 5
    assert accumulator.emitted_transition_count == 2


@pytest.mark.parametrize(("done", "truncated"), [(True, False), (False, True)])
def test_accumulator_rejects_terminal_or_reset_semantics(
    done: bool, truncated: bool
) -> None:
    accumulator = runner.FullRainbowThreeStepAccumulator(
        core.FullRainbowForagerConfig()
    )
    with pytest.raises(runner.FullRainbowRunnerContractError, match="continuing"):
        accumulator.accumulate(
            observation=_observation(0),
            action=0,
            raw_reward=0,
            done=done,
            truncated=truncated,
        )


def test_compact_replay_is_lazy_wraps_exactly_and_uses_monotonic_max_priority() -> None:
    replay = runner.CompactPrioritizedReplay(
        capacity=3,
        batch_size=2,
        observation_shape=(9, 9, 3),
        page_size=2,
    )
    assert replay.size == replay.page_count == replay.insertions == 0
    assert replay.max_recorded_priority == 1.0

    assert replay.add(_accumulated(0)) == 0
    replay.update_priorities(
        jnp.asarray([0], dtype=jnp.int32), jnp.asarray([4.0], dtype=jnp.float32)
    )
    assert replay.add(_accumulated(1)) == 1
    assert replay.priority_at(1) == 4.0
    replay.update_priorities(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0.5, 0.25], dtype=jnp.float32),
    )
    assert replay.add(_accumulated(2)) == 2
    assert replay.priority_at(2) == 4.0
    assert replay.add(_accumulated(3)) == 0
    assert replay.add(_accumulated(4)) == 1

    assert replay.size == 3
    assert replay.insertions == 5
    assert replay.evictions == 2
    assert replay.next_index == 2
    assert replay.page_count == 2
    assert replay.transition_at(0).source_transition == 4
    assert replay.transition_at(1).source_transition == 5
    assert replay.transition_at(2).source_transition == 3


def test_duplicate_priority_updates_keep_first_occurrence_but_record_all_value_max() -> None:
    replay = runner.CompactPrioritizedReplay(
        capacity=3,
        batch_size=3,
        observation_shape=(9, 9, 3),
        page_size=2,
    )
    for index in range(3):
        replay.add(_accumulated(index))
    replay.update_priorities(
        jnp.asarray([1, 1, 0], dtype=jnp.int32),
        jnp.asarray([0.25, 9.0, 2.0], dtype=jnp.float32),
    )

    assert replay.priority_at(1) == 0.25
    assert replay.priority_at(0) == 2.0
    assert replay.max_recorded_priority == 9.0
    assert replay.add(_accumulated(3)) == 0
    assert replay.priority_at(0) == 9.0
    assert replay.duplicate_priority_resolution == "first_occurrence_wins"


def test_proportional_sample_is_with_replacement_and_builds_exact_core_batch() -> None:
    replay = runner.CompactPrioritizedReplay(
        capacity=3,
        batch_size=4,
        observation_shape=(9, 9, 3),
        page_size=2,
    )
    for index in range(3):
        replay.add(_accumulated(index))
    replay.update_priorities(
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jnp.asarray([0.0, 0.0, 7.0], dtype=jnp.float32),
    )
    draw = replay.sample(jr.key(3, impl="threefry2x32"))

    np.testing.assert_array_equal(draw.indices, np.full((4,), 2))
    np.testing.assert_array_equal(draw.sampling_probabilities, np.ones((4,)))
    np.testing.assert_array_equal(draw.importance_weights, np.ones((4,)))
    assert type(draw.batch) is core.FullRainbowReplayBatch
    assert draw.batch.states.shape == (4, 9, 9, 3)
    assert draw.batch.states.dtype == jnp.float32
    assert draw.batch.actions.dtype == jnp.int32
    assert draw.batch.scaled_n_step_rewards.dtype == jnp.float32
    assert draw.batch.bootstrap_discounts.dtype == jnp.float32


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_replay_priority_validation_is_fail_closed(value: float) -> None:
    replay = runner.CompactPrioritizedReplay(
        capacity=2,
        batch_size=1,
        observation_shape=(9, 9, 3),
        page_size=1,
    )
    replay.add(_accumulated(0))
    with pytest.raises(runner.FullRainbowRunnerContractError, match="priorit"):
        replay.update_priorities(
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([value], dtype=jnp.float32),
        )


def test_replay_rejects_unreachable_return_and_oversized_capacity_before_allocation() -> None:
    with pytest.raises(runner.FullRainbowRunnerContractError, match="capacity"):
        runner.CompactPrioritizedReplay(
            capacity=1_000_001,
            batch_size=1,
            observation_shape=(9, 9, 3),
            page_size=1,
        )

    replay = runner.CompactPrioritizedReplay(
        capacity=1,
        batch_size=1,
        observation_shape=(9, 9, 3),
        page_size=1,
    )
    with pytest.raises(runner.FullRainbowRunnerContractError, match="unreachable"):
        replay.add(
            dataclasses.replace(_accumulated(0), scaled_n_step_reward=123.0)
        )


def test_zero_epsilon_action_preserves_uniform_less_equal_zero_edge() -> None:
    q_values = jnp.asarray([1.0, 5.0, 2.0, 3.0], dtype=jnp.float32)
    assert runner.resolve_zero_epsilon_action(
        q_values, uniform_draw=0.0, random_action=3
    ) == 3
    assert runner.resolve_zero_epsilon_action(
        q_values, uniform_draw=np.nextafter(np.float32(0), np.float32(1)), random_action=3
    ) == 1


def test_tiny_driver_hits_every_boundary_and_keeps_raw_score_unscaled() -> None:
    rewards = (-1, 0, 1, 30)
    environment = _FakeEnvironmentRuntime(rewards)
    dependencies, log = _fake_dependencies(environment)
    result = runner.run_full_rainbow_engineering(
        environment_seed=17,
        agent_seed=23,
        schedule=_tiny_schedule(),
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    receipt = runner.parse_full_rainbow_engineering_receipt(result.receipt_bytes)

    assert result.interactions == 12
    assert result.cumulative_raw_score == sum(rewards) * 3
    assert list(np.frombuffer(result.raw_reward_trace, dtype=np.int8)) == list(rewards) * 3
    assert receipt["accounting"]["replay_insertions"] == 9
    assert receipt["accounting"]["optimizer_updates"] == 4
    assert receipt["accounting"]["target_syncs"] == 2
    assert receipt["accounting"]["replay_evictions"] == 4
    assert receipt["accounting"]["update_transitions"] == [6, 8, 10, 12]
    assert receipt["accounting"]["target_sync_transitions"] == [8, 12]
    assert receipt["score"]["raw_cumulative_score"] == sum(rewards) * 3
    assert receipt["score"]["reward_scaling_applied_to_score"] is False
    assert receipt["completion"] == {
        "content_and_accounting_independently_prove_execution": False,
        "engineering_schedule_complete": True,
        "exact_matched_v3_horizon_complete": False,
        "interactions_completed": 12,
        "production_runtime_complete": False,
        "requested_horizon": 12,
        "schedule_is_exact_production": False,
    }
    assert receipt["classification"] == "synthetic_engineering_non_authorizing"
    assert receipt["seeds"] == {
        "agent_seed": 23,
        "domain": "uint31",
        "environment_seed": 17,
        "protected_seed_status": "unknown_not_asserted",
        "seed_provenance_classification": "caller_supplied_unverified",
        "seed_provenance_verified": False,
        "source": "caller_supplied",
    }
    assert receipt["limitations"][0] == (
        "Receipt content and accounting do not independently prove execution."
    )
    assert result.production_runtime is False
    assert len(log.action_noise_keys) == 12
    assert len(log.update_input_keys) == 4
    assert len(log.sync_updates) == 2
    assert environment.actions.__len__() == 12


def test_update_eligibility_uses_total_insertions_after_replay_wraparound() -> None:
    environment = _FakeEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    result = runner.run_full_rainbow_engineering(
        environment_seed=17,
        agent_seed=23,
        schedule=runner.FullRainbowRunnerSchedule(
            horizon=10,
            replay_capacity=2,
            batch_size=1,
            minimum_replay_history=3,
            update_period=1,
            target_update_period=2,
            page_size=1,
        ),
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    receipt = runner.parse_full_rainbow_engineering_receipt(result.receipt_bytes)

    assert receipt["accounting"]["maximum_replay_residency"] == 2
    assert receipt["accounting"]["replay_insertions"] == 7
    assert receipt["accounting"]["update_transitions"] == [7, 8, 9, 10]
    assert receipt["accounting"]["target_sync_transitions"] == [8, 10]


def test_driver_rejects_nonfinite_update_and_equal_key_environment_substitution() -> None:
    schedule = runner.FullRainbowRunnerSchedule(
        horizon=6,
        replay_capacity=3,
        batch_size=1,
        minimum_replay_history=2,
        update_period=2,
        target_update_period=4,
        page_size=1,
    )

    for mutation in ("nonfinite", "environment_key_substitution"):
        environment = _FakeEnvironmentRuntime((0,))
        dependencies, _ = _fake_dependencies(environment)
        original_update = dependencies.update_core

        def bad_update(
            config: core.FullRainbowForagerConfig,
            state: core.FullRainbowCoreState,
            batch: core.FullRainbowReplayBatch,
            *,
            selected: str = mutation,
        ) -> tuple[core.FullRainbowCoreState, core.FullRainbowTrainMetrics]:
            updated, metrics = original_update(config, state, batch)
            if selected == "nonfinite":
                updated = dataclasses.replace(
                    updated, online_params=jnp.asarray(np.nan, dtype=jnp.float32)
                )
            else:
                updated = dataclasses.replace(
                    updated,
                    environment_rng=jr.key(17, impl="threefry2x32"),
                )
            return updated, metrics

        bad_dependencies = _replace_dependencies(
            dependencies, update_core=bad_update
        )
        with pytest.raises(runner.FullRainbowRunnerContractError, match="update_core"):
            runner.run_full_rainbow_engineering(
                environment_seed=17,
                agent_seed=23,
                schedule=schedule,
                dependencies=bad_dependencies,
                unqualified_engineering=True,
            )


def test_driver_rejects_nonaliased_initial_target_parameters() -> None:
    environment = _FakeEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    original_initialize = dependencies.initialize_core

    def bad_initialize(
        config: core.FullRainbowForagerConfig,
        environment_seed: int,
        agent_seed: int,
    ) -> core.FullRainbowCoreState:
        state = original_initialize(config, environment_seed, agent_seed)
        return dataclasses.replace(
            state,
            target_params=jnp.asarray(
                np.asarray(state.target_params).copy(), dtype=jnp.float32
            ),
        )

    with pytest.raises(runner.FullRainbowRunnerContractError, match="alias"):
        runner.run_full_rainbow_engineering(
            environment_seed=1,
            agent_seed=2,
            schedule=runner.FullRainbowRunnerSchedule(
                horizon=1,
                replay_capacity=1,
                batch_size=1,
                minimum_replay_history=99,
                update_period=2,
                target_update_period=4,
                page_size=1,
            ),
            dependencies=_replace_dependencies(
                dependencies, initialize_core=bad_initialize
            ),
            unqualified_engineering=True,
        )


def test_agent_chain_uses_four_way_action_split_and_never_consumes_core_env_root() -> None:
    environment = _FakeEnvironmentRuntime((0,))
    dependencies, log = _fake_dependencies(environment)
    result = runner.run_full_rainbow_engineering(
        environment_seed=17,
        agent_seed=23,
        schedule=runner.FullRainbowRunnerSchedule(
            horizon=4,
            replay_capacity=2,
            batch_size=1,
            minimum_replay_history=99,
            update_period=2,
            target_update_period=4,
            page_size=1,
        ),
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    receipt = runner.parse_full_rainbow_engineering_receipt(result.receipt_bytes)

    roots = core.full_rainbow_seed_roots(environment_seed=17, agent_seed=23)
    agent_key, _, _ = jr.split(roots.agent, 3)
    expected_noise: list[tuple[int, int]] = []
    for _ in range(4):
        agent_key, _, noise_key, _ = jr.split(agent_key, 4)
        expected_noise.append(_key_words(noise_key))
    assert log.action_noise_keys == expected_noise
    assert receipt["rng_accounting"] == {
        "agent_continuation_split_calls": 5,
        "agent_parameter_initialization_subkeys": 2,
        "agent_action_split_calls": 4,
        "agent_action_subkeys_produced_per_call": 3,
        "agent_replay_sampling_split_calls": 0,
        "agent_update_split_calls": 0,
        "agent_update_subkeys_produced_per_call": 2,
        "core_environment_key_consumptions": 0,
        "bridge_environment_key_consumptions": 5,
    }


def test_agent_seed_perturbation_cannot_change_bridge_environment_keys() -> None:
    first_environment = _FakeEnvironmentRuntime((0,))
    first_dependencies, _ = _fake_dependencies(first_environment)
    second_environment = _FakeEnvironmentRuntime((0,))
    second_dependencies, _ = _fake_dependencies(second_environment)
    schedule = runner.FullRainbowRunnerSchedule(
        horizon=5,
        replay_capacity=3,
        batch_size=1,
        minimum_replay_history=99,
        update_period=2,
        target_update_period=4,
        page_size=1,
    )

    runner.run_full_rainbow_engineering(
        environment_seed=31,
        agent_seed=7,
        schedule=schedule,
        dependencies=first_dependencies,
        unqualified_engineering=True,
    )
    runner.run_full_rainbow_engineering(
        environment_seed=31,
        agent_seed=99,
        schedule=schedule,
        dependencies=second_dependencies,
        unqualified_engineering=True,
    )

    assert first_environment.reset_keys == second_environment.reset_keys
    assert first_environment.step_keys == second_environment.step_keys


def test_driver_rejects_environment_state_reuse() -> None:
    environment = _ReusingEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    with pytest.raises(runner.FullRainbowRunnerContractError, match="state"):
        runner.run_full_rainbow_engineering(
            environment_seed=1,
            agent_seed=2,
            schedule=runner.FullRainbowRunnerSchedule(
                horizon=1,
                replay_capacity=1,
                batch_size=1,
                minimum_replay_history=99,
                update_period=2,
                target_update_period=4,
                page_size=1,
            ),
            dependencies=dependencies,
            unqualified_engineering=True,
        )


def test_engineering_and_production_execution_require_explicit_unqualified_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    with pytest.raises(runner.FullRainbowRunnerExecutionBlockedError, match="unqualified"):
        runner.run_full_rainbow_engineering(
            environment_seed=1,
            agent_seed=2,
            schedule=_tiny_schedule(),
            dependencies=dependencies,
        )

    monkeypatch.setattr(
        runner,
        "_production_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("dependencies constructed")),
    )
    with pytest.raises(runner.FullRainbowRunnerExecutionBlockedError, match="unqualified"):
        runner.run_matched_v3_full_rainbow(environment_seed=1, agent_seed=2)

    monkeypatch.setattr(runner, "_production_dependencies", lambda: dependencies)
    with pytest.raises(runner.FullRainbowRunnerContractError, match="capability"):
        runner.run_matched_v3_full_rainbow(
            environment_seed=1,
            agent_seed=2,
            unqualified_engineering=True,
        )


def test_runner_descriptor_is_frozen_source_bound_and_non_authorizing() -> None:
    raw = runner.canonical_full_rainbow_runner_descriptor_bytes()
    descriptor = runner.full_rainbow_runner_descriptor()

    assert json.loads(raw) == descriptor
    assert hashlib.sha256(raw).hexdigest() == runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
    assert runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256 == (
        "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
    )
    assert descriptor["bindings"] == {
        "bridge_descriptor_sha256": (
            "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
        ),
        "bridge_implementation_path": (
            "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
        ),
        "bridge_implementation_sha256": (
            "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
        ),
        "core_configuration_sha256": (
            "835f02bdcf6844b7cd8c5e9fe33230a2a94f3a9c288c812cbfddf473c28b7e3f"
        ),
        "core_descriptor_sha256": (
            "5436200c47e1b003b0371c30606b52163b4c42427fa84e2fe2f4b2b2273ccae2"
        ),
        "core_implementation_path": (
            "alberta_framework/benchmarks/forager_matched_v3_full_rainbow.py"
        ),
        "core_implementation_sha256": (
            "7f75a0862ddc21160cea9c0a9faca221a0d757985fc90e5ef02b4673e3c14f5a"
        ),
    }
    assert descriptor["replay"]["duplicate_priority_resolution"] == (
        "first_occurrence_wins"
    )
    assert descriptor["claims"] == {
        "execution_ready": False,
        "execution_authorized": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
    descriptor["claims"]["authority_granted"] = True
    assert runner.full_rainbow_runner_descriptor()["claims"][
        "authority_granted"
    ] is False


def test_injected_dependencies_cannot_obtain_production_classification() -> None:
    environment = _FakeEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    schedule = runner.FullRainbowRunnerSchedule(
        horizon=1,
        replay_capacity=1,
        batch_size=1,
        minimum_replay_history=99,
        update_period=2,
        target_update_period=4,
        page_size=1,
    )

    with pytest.raises(runner.FullRainbowRunnerContractError, match="capability"):
        runner._run_full_rainbow(
            environment_seed=1,
            agent_seed=2,
            schedule=schedule,
            dependencies=dependencies,
            production_runtime=True,
            unqualified_engineering=True,
        )

    result = runner.run_full_rainbow_engineering(
        environment_seed=1,
        agent_seed=2,
        schedule=schedule,
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    with pytest.raises(runner.FullRainbowRunnerContractError, match="capability"):
        runner.validate_full_rainbow_runner_result(
            dataclasses.replace(result, production_runtime=True)
        )
    with pytest.raises(runner.FullRainbowRunnerContractError):
        runner.parse_full_rainbow_result_receipt(result.receipt_bytes)


def test_production_dependency_capability_rejects_in_place_field_mutation() -> None:
    synthetic_dependencies, _ = _fake_dependencies(_FakeEnvironmentRuntime((0,)))
    runtime = object.__new__(bridge.MatchedV3ForagaxRuntime)
    dependencies = runner.FullRainbowRunnerDependencies(
        dependency_identity="production_bridge_and_compiled_full_rainbow_v1",
        environment_runtime=runtime,
        step_environment=bridge.step_matched_v3_foragax_bridge,
        initialize_core=synthetic_dependencies.initialize_core,
        action_q_values=synthetic_dependencies.action_q_values,
        update_core=synthetic_dependencies.update_core,
        sync_target=core.sync_full_rainbow_target,
        runtime_identity={
            "backend": "synthetic_capability_test",
            "runtime_qualified": False,
            "foragax_runtime_parity_executed": False,
        },
        compiled_action_kernel=True,
        compiled_update_kernel=True,
    )
    binding = runner._ProductionDependencyBinding(
        dependency_identity=dependencies.dependency_identity,
        environment_runtime=dependencies.environment_runtime,
        step_environment=dependencies.step_environment,
        initialize_core=dependencies.initialize_core,
        action_q_values=dependencies.action_q_values,
        update_core=dependencies.update_core,
        sync_target=dependencies.sync_target,
        runtime_identity=dependencies.runtime_identity,
        runtime_identity_sha256=runner._canonical_sha256(
            dict(dependencies.runtime_identity)
        ),
        compiled_action_kernel=dependencies.compiled_action_kernel,
        compiled_update_kernel=dependencies.compiled_update_kernel,
    )
    with runner._PRODUCTION_REGISTRY_LOCK:
        runner._PRODUCTION_DEPENDENCY_REGISTRY[dependencies] = binding
    replacements: dict[str, object] = {
        "dependency_identity": "mutated",
        "environment_runtime": object(),
        "step_environment": lambda *_: None,
        "initialize_core": lambda *_: None,
        "action_q_values": lambda *_: None,
        "update_core": lambda *_: None,
        "sync_target": lambda *_: None,
        "runtime_identity": dict(dependencies.runtime_identity),
        "compiled_action_kernel": False,
        "compiled_update_kernel": False,
    }
    try:
        runner._validate_production_dependencies(dependencies)
        for field, replacement in replacements.items():
            original = getattr(dependencies, field)
            object.__setattr__(dependencies, field, replacement)
            with pytest.raises(runner.FullRainbowRunnerContractError, match="binding"):
                runner._validate_production_dependencies(dependencies)
            object.__setattr__(dependencies, field, original)
            runner._validate_production_dependencies(dependencies)
    finally:
        with runner._PRODUCTION_REGISTRY_LOCK:
            runner._PRODUCTION_DEPENDENCY_REGISTRY.pop(dependencies, None)


def test_production_result_capability_rejects_in_place_field_mutation() -> None:
    result = runner.FullRainbowRunnerResult(
        raw_reward_trace=b"\x00",
        cumulative_raw_score=0,
        interactions=1,
        receipt_bytes=b"{}",
        production_runtime=True,
    )
    binding = runner._ProductionResultBinding(
        raw_reward_trace=result.raw_reward_trace,
        raw_reward_trace_sha256=hashlib.sha256(result.raw_reward_trace).hexdigest(),
        cumulative_raw_score=result.cumulative_raw_score,
        interactions=result.interactions,
        receipt_bytes=result.receipt_bytes,
        receipt_sha256=hashlib.sha256(result.receipt_bytes).hexdigest(),
        production_runtime=True,
    )
    with runner._PRODUCTION_REGISTRY_LOCK:
        runner._PRODUCTION_RESULT_REGISTRY[result] = binding
    replacements: dict[str, object] = {
        "raw_reward_trace": bytes(bytearray(b"\x00")),
        "cumulative_raw_score": 1,
        "interactions": 2,
        "receipt_bytes": bytes(bytearray(b"{}")),
        "production_runtime": False,
    }
    try:
        runner._validate_production_result_capability(result)
        for field, replacement in replacements.items():
            original = getattr(result, field)
            object.__setattr__(result, field, replacement)
            with pytest.raises(runner.FullRainbowRunnerContractError, match="capability"):
                runner._validate_production_result_capability(result)
            object.__setattr__(result, field, original)
            runner._validate_production_result_capability(result)
    finally:
        with runner._PRODUCTION_REGISTRY_LOCK:
            runner._PRODUCTION_RESULT_REGISTRY.pop(result, None)


def test_descriptor_and_receipt_parsers_reject_mutation_duplicates_and_authority() -> None:
    descriptor_raw = runner.canonical_full_rainbow_runner_descriptor_bytes()
    descriptor = runner.full_rainbow_runner_descriptor()
    descriptor["claims"]["authority_granted"] = True
    with pytest.raises(runner.FullRainbowRunnerContractError):
        runner.parse_full_rainbow_runner_descriptor(descriptor)
    with pytest.raises(runner.FullRainbowRunnerContractError):
        runner.parse_full_rainbow_runner_descriptor(b" " + descriptor_raw)
    with pytest.raises(runner.FullRainbowRunnerContractError):
        runner.parse_full_rainbow_runner_descriptor(
            descriptor_raw[:-1] + b',"status":"implemented_unqualified"}'
        )

    environment = _FakeEnvironmentRuntime((0,))
    dependencies, _ = _fake_dependencies(environment)
    result = runner.run_full_rainbow_engineering(
        environment_seed=1,
        agent_seed=2,
        schedule=runner.FullRainbowRunnerSchedule(
            horizon=1,
            replay_capacity=1,
            batch_size=1,
            minimum_replay_history=99,
            update_period=2,
            target_update_period=4,
            page_size=1,
        ),
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    receipt = runner.parse_full_rainbow_engineering_receipt(result.receipt_bytes)
    receipt["claims"]["authority_granted"] = True
    changed = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    for invalid in (
        changed,
        b" " + result.receipt_bytes,
        result.receipt_bytes[:-1]
        + b',"status":"completed_engineering_unqualified"}',
    ):
        with pytest.raises(runner.FullRainbowRunnerContractError):
            runner.parse_full_rainbow_engineering_receipt(invalid)

    nested_authority = runner.parse_full_rainbow_engineering_receipt(
        result.receipt_bytes
    )
    nested_authority["runtime_identity"]["execution_authorized"] = True
    with pytest.raises(runner.FullRainbowRunnerContractError, match="authority"):
        runner.parse_full_rainbow_engineering_receipt(
            _rehash_receipt(nested_authority)
        )

    extra_accounting = runner.parse_full_rainbow_engineering_receipt(
        result.receipt_bytes
    )
    extra_accounting["accounting"]["forged_extra"] = 0
    with pytest.raises(runner.FullRainbowRunnerContractError, match="accounting"):
        runner.parse_full_rainbow_engineering_receipt(
            _rehash_receipt(extra_accounting)
        )


def test_partial_horizon_cannot_be_mislabeled_complete_and_trace_is_bound() -> None:
    environment = _FakeEnvironmentRuntime((1,))
    dependencies, _ = _fake_dependencies(environment)
    result = runner.run_full_rainbow_engineering(
        environment_seed=1,
        agent_seed=2,
        schedule=runner.FullRainbowRunnerSchedule(
            horizon=2,
            replay_capacity=1,
            batch_size=1,
            minimum_replay_history=99,
            update_period=2,
            target_update_period=4,
            page_size=1,
        ),
        dependencies=dependencies,
        unqualified_engineering=True,
    )
    receipt = runner.parse_full_rainbow_engineering_receipt(result.receipt_bytes)
    receipt["completion"]["exact_matched_v3_horizon_complete"] = True
    receipt.pop("receipt_body_sha256")
    receipt["receipt_body_sha256"] = hashlib.sha256(
        runner.canonical_full_rainbow_receipt_body_bytes(receipt)
    ).hexdigest()
    forged = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(runner.FullRainbowRunnerContractError, match="completion"):
        runner.parse_full_rainbow_engineering_receipt(forged)

    with pytest.raises(runner.FullRainbowRunnerContractError, match="trace"):
        runner.validate_full_rainbow_runner_result(
            dataclasses.replace(result, raw_reward_trace=b"\x00\x00")
        )
