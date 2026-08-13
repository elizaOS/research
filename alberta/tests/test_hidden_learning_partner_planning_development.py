"""Contracts for the development-only hidden co-learning planning bridge."""

from __future__ import annotations

import dataclasses
import gc

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    BEHAVIOR_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_MODELS_FROZEN,
    BOTH_ROLES_FROZEN,
    CONSTANT_ONE_DELIVERY,
    CONSTANT_ZERO_DELIVERY,
    GROUNDED_FROZEN,
    HELPER_FROZEN,
    HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA,
    JOINT_ADAPTIVE,
    PLANNER_NEVER_CONSUMED,
    SHUFFLED_DELIVERY,
    HiddenDyadFeedback,
    HiddenDyadPreObservation,
    HiddenLearningPartnerPlanningBridge,
    HiddenLearningPartnerPlanningConfig,
    condition_spec,
    run_hidden_learning_partner_planning,
    strip_hidden_learning_partner_oracle,
    validate_hidden_learning_partner_planning_run,
)

pytestmark = pytest.mark.development


@pytest.fixture(autouse=True)
def _release_jax_compilation_cache():
    """Keep this many-condition module within the constrained CI host budget."""

    yield
    jax.clear_caches()
    gc.collect()


def _tiny_config(*, steps_per_phase: int = 2) -> HiddenLearningPartnerPlanningConfig:
    return HiddenLearningPartnerPlanningConfig(
        phase_length=steps_per_phase,
        n_phases=4,
        learning_rate=0.2,
        epsilon=0.2,
        behavior_step_size=0.1,
        grounded_step_size=0.1,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jax.dtypes.issubdtype(left_leaf.dtype, jax.dtypes.prng_key):
            left_leaf = jr.key_data(left_leaf)
            right_leaf = jr.key_data(right_leaf)
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_execution_equivalent(
    left: object,
    right: object,
    *,
    prefix: str = "",
) -> dict[str, int]:
    """Require exact discrete leaves and at most one ULP on execution floats."""

    differences: dict[str, int] = {}
    if dataclasses.is_dataclass(left):
        assert type(left) is type(right)
        for field in dataclasses.fields(left):
            child = field.name if not prefix else f"{prefix}.{field.name}"
            differences.update(
                _assert_execution_equivalent(
                    getattr(left, field.name),
                    getattr(right, field.name),
                    prefix=child,
                )
            )
        return differences
    left_array = left
    right_array = right
    if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
        left_array = jr.key_data(left_array)
        right_array = jr.key_data(right_array)
    left_host = np.asarray(left_array)
    right_host = np.asarray(right_array)
    if np.issubdtype(left_host.dtype, np.floating):
        np.testing.assert_array_max_ulp(left_host, right_host, maxulp=1)
        if not np.array_equal(left_host, right_host):
            differences[prefix] = 1
    else:
        np.testing.assert_array_equal(left_host, right_host)
    return differences


def _key_words(key: jax.Array) -> np.ndarray:
    return np.asarray(jr.key_data(key), dtype=np.uint32)


def test_config_projection_and_condition_surface_are_strict_and_nonpromoting() -> None:
    assert HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA == (
        "alberta.hidden-learning-partner-planning.development.v1"
    )
    default = HiddenLearningPartnerPlanningConfig()
    assert default.phase_length == 512
    assert default.n_phases == 6
    assert default.num_steps == 3_072
    assert default.phase_diagnostic_window_steps == 128
    payload = default.to_dict()
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["claim_thresholds_frozen"] is False

    for kwargs in (
        {"phase_length": True},
        {"phase_length": 0},
        {"n_phases": 3},
        {"n_phases": 5},
        {"learning_rate": float("nan")},
        {"behavior_step_size": -0.1},
        {"grounded_step_size": 0.0},
    ):
        with pytest.raises(ValueError):
            HiddenLearningPartnerPlanningConfig(**kwargs)  # type: ignore[arg-type]

    assert {field.name for field in dataclasses.fields(HiddenDyadPreObservation)} == {
        "helper_cue"
    }
    feedback_fields = {field.name for field in dataclasses.fields(HiddenDyadFeedback)}
    assert feedback_fields == {
        "helper_cue",
        "helper_message",
        "delivered_message",
        "beneficiary_action",
        "reward",
        "next_helper_cue",
        "terminated",
        "discount",
    }
    assert not feedback_fields & {
        "context",
        "public_context",
        "phase_index",
        "step_count",
        "target",
        "oracle",
    }
    assert "target" not in strip_hidden_learning_partner_oracle.__annotations__

    spec = condition_spec(JOINT_ADAPTIVE)
    assert {field.name for field in dataclasses.fields(spec)} == {
        "channel",
        "helper_write",
        "beneficiary_write",
        "behavior_write",
        "grounded_write",
        "planner_consumption",
    }


def test_exact_resource_budget_and_shared_row_interference() -> None:
    config = _tiny_config(steps_per_phase=4)
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=11,
        config=config,
    )
    assert run.resource.signaling_state_nbytes == 80
    assert run.resource.behavior_state_nbytes == 48
    assert run.resource.grounded_state_nbytes == 108
    assert run.resource.learner_model_state_nbytes == 236
    assert run.resource.world_state_nbytes == 32
    assert run.resource.total_state_nbytes == 321
    assert run.resource.replay_capacity == 0
    assert run.resource.exact_tree_match
    np.testing.assert_array_equal(
        run.final_state.behavior.step_words,
        np.asarray((0, config.num_steps), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        run.final_state.grounded.update_words,
        np.asarray((0, config.num_steps), dtype=np.uint32),
    )

    np.testing.assert_array_equal(run.trace.helper_context, np.zeros(config.num_steps))
    np.testing.assert_array_equal(run.trace.beneficiary_context, np.zeros(config.num_steps))
    assert bool(jnp.all(run.trace.shared_inactive_rows_unchanged))
    np.testing.assert_array_equal(
        run.final_state.learner.helper.values[1],
        np.zeros((2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        run.final_state.learner.beneficiary.values[1],
        np.zeros((2, 2), dtype=np.float32),
    )
    assert np.count_nonzero(np.asarray(run.final_state.learner.helper.values[0])) > 0
    assert np.count_nonzero(np.asarray(run.final_state.learner.beneficiary.values[0])) > 0
    assert validate_hidden_learning_partner_planning_run(run) == ()

    bridge = HiddenLearningPartnerPlanningBridge(config, JOINT_ADAPTIVE)
    wider_behavior = run.initial_state.behavior.replace(
        weights=jnp.zeros((2, 2), dtype=jnp.float32)
    )
    shape_tampered = run.initial_state.replace(behavior=wider_behavior)
    tampered_budget = bridge.resource_budget(shape_tampered)
    assert tampered_budget.behavior_state_nbytes == 56
    assert tampered_budget.total_state_nbytes == 329
    assert not tampered_budget.exact_tree_match


def test_cached_predictions_equal_prequential_update_predictions() -> None:
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=3,
        config=_tiny_config(steps_per_phase=3),
    )
    np.testing.assert_array_equal(
        run.trace.behavior_probabilities_pre,
        run.trace.behavior_probabilities_update,
    )
    np.testing.assert_array_equal(
        run.trace.grounded_raw_prediction_pre,
        run.trace.grounded_raw_prediction_update,
    )
    assert bool(jnp.all(run.trace.behavior_prediction_bound))
    assert bool(jnp.all(run.trace.grounded_prediction_bound))
    assert np.all(np.isfinite(np.asarray(run.trace.behavior_nll)))
    assert np.all(np.isfinite(np.asarray(run.trace.grounded_reward_error)))


@pytest.mark.parametrize(
    ("condition", "frozen_components"),
    (
        (HELPER_FROZEN, ("helper",)),
        (BENEFICIARY_FROZEN, ("beneficiary",)),
        (BOTH_ROLES_FROZEN, ("helper", "beneficiary")),
        (BEHAVIOR_FROZEN, ("behavior",)),
        (GROUNDED_FROZEN, ("grounded",)),
        (BOTH_MODELS_FROZEN, ("behavior", "grounded")),
    ),
)
def test_outer_freeze_masks_preserve_exact_state_bits(
    condition: str,
    frozen_components: tuple[str, ...],
) -> None:
    run = run_hidden_learning_partner_planning(
        condition,
        seed=29,
        config=_tiny_config(steps_per_phase=2),
        jit_compile=False,
    )
    for component in frozen_components:
        if component == "helper":
            np.testing.assert_array_equal(
                run.final_state.learner.helper.values,
                run.initial_state.learner.helper.values,
            )
            assert not np.array_equal(
                _key_words(run.final_state.learner.helper.key),
                _key_words(run.initial_state.learner.helper.key),
            )
        elif component == "beneficiary":
            np.testing.assert_array_equal(
                run.final_state.learner.beneficiary.values,
                run.initial_state.learner.beneficiary.values,
            )
            assert not np.array_equal(
                _key_words(run.final_state.learner.beneficiary.key),
                _key_words(run.initial_state.learner.beneficiary.key),
            )
        else:
            _assert_tree_equal(
                getattr(run.final_state, component),
                getattr(run.initial_state, component),
            )
    if "behavior" in frozen_components:
        assert bool(jnp.all(run.trace.behavior_proposal_applied))
        assert not bool(jnp.any(run.trace.behavior_committed_write))
    if "grounded" in frozen_components:
        assert bool(jnp.all(run.trace.grounded_proposal_applied))
        assert not bool(jnp.any(run.trace.grounded_committed_write))
    spec = condition_spec(condition)
    expected_behavior_writes = run.config.num_steps if spec.behavior_write else 0
    expected_grounded_writes = run.config.num_steps if spec.grounded_write else 0
    assert int(run.final_state.behavior.step_count) == expected_behavior_writes
    np.testing.assert_array_equal(
        run.final_state.behavior.step_words,
        np.asarray((0, expected_behavior_writes), dtype=np.uint32),
    )
    assert int(run.final_state.grounded.update_count) == expected_grounded_writes
    np.testing.assert_array_equal(
        run.final_state.grounded.update_words,
        np.asarray((0, expected_grounded_writes), dtype=np.uint32),
    )
    assert validate_hidden_learning_partner_planning_run(run) == ()


def test_named_rng_streams_remain_paired_across_static_interventions() -> None:
    config = _tiny_config(steps_per_phase=2)
    conditions = (
        JOINT_ADAPTIVE,
        HELPER_FROZEN,
        BENEFICIARY_FROZEN,
        BEHAVIOR_FROZEN,
        GROUNDED_FROZEN,
        PLANNER_NEVER_CONSUMED,
    )
    runs = tuple(
        run_hidden_learning_partner_planning(
            condition,
            seed=41,
            config=config,
            jit_compile=False,
        )
        for condition in conditions
    )
    reference = runs[0].final_state
    for run in runs[1:]:
        state = run.final_state
        for left, right in (
            (state.world.cue_key, reference.world.cue_key),
            (state.world.channel_key, reference.world.channel_key),
            (state.learner.helper.key, reference.learner.helper.key),
            (state.learner.beneficiary.key, reference.learner.beneficiary.key),
            (state.planner_key, reference.planner_key),
            (state.intervention_key, reference.intervention_key),
        ):
            np.testing.assert_array_equal(_key_words(left), _key_words(right))


def test_randomized_intervention_and_potential_outcomes_bind_executed_reward() -> None:
    config = _tiny_config(steps_per_phase=8)
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=7,
        config=config,
    )
    trace = run.trace
    expected_message = np.where(
        np.asarray(trace.planner_consumed),
        np.asarray(trace.planner_message),
        np.asarray(trace.ordinary_message),
    )
    np.testing.assert_array_equal(trace.helper_message, expected_message)
    expected_reward = np.take_along_axis(
        np.asarray(trace.delivered_potential_rewards),
        np.asarray(trace.delivered_message, dtype=np.int64)[:, None],
        axis=1,
    )[:, 0]
    np.testing.assert_array_equal(trace.reward, expected_reward)
    assert bool(jnp.all(trace.potential_outcome_bound))
    assert bool(jnp.all(trace.intervention_bound))
    assert np.any(np.asarray(trace.planner_consumed))
    assert np.any(~np.asarray(trace.planner_consumed))
    assert run.metrics.num_steps == config.num_steps
    assert 0.0 <= run.metrics.planner_consumption_rate <= 1.0
    assert 0.0 <= run.metrics.action_change_rate <= 1.0

    never = run_hidden_learning_partner_planning(
        PLANNER_NEVER_CONSUMED,
        seed=7,
        config=config,
        jit_compile=False,
    )
    assert not bool(jnp.any(never.trace.planner_consumed))
    np.testing.assert_array_equal(never.trace.helper_message, never.trace.ordinary_message)


def test_eager_jit_and_short_scan_have_portable_execution_equivalence() -> None:
    config = _tiny_config(steps_per_phase=1)
    bridge = HiddenLearningPartnerPlanningBridge(config, JOINT_ADAPTIVE)
    initial = bridge.initialize(jr.key(5))
    eager = bridge.step(initial)
    compiled = jax.jit(bridge.step)(initial)
    _assert_tree_equal(eager, compiled)

    eager_run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=5,
        config=config,
        jit_compile=False,
    )
    compiled_run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=5,
        config=config,
        jit_compile=True,
    )
    trace_differences = _assert_execution_equivalent(
        eager_run.trace,
        compiled_run.trace,
    )
    state_differences = _assert_execution_equivalent(
        eager_run.final_state,
        compiled_run.final_state,
    )
    differences = trace_differences | state_differences
    assert all(
        path.startswith("behavior") or path.startswith("grounded")
        for path in differences
    )
    # Some XLA/CPU combinations contract the Grounded-model arithmetic and
    # differ from eager execution by one ULP; others remain bit-exact.  Both
    # outcomes satisfy the portable contract.
    assert max(differences.values(), default=0) <= 1


def test_invalid_config_token_latches_and_atomically_preserves_state() -> None:
    config = _tiny_config(steps_per_phase=1)
    bridge = HiddenLearningPartnerPlanningBridge(config, JOINT_ADAPTIVE)
    initial = bridge.initialize(jr.key(17))
    corrupted = initial.replace(config_token=initial.config_token.at[0].add(1))
    result = jax.jit(bridge.step)(corrupted)
    assert not bool(result.trace.accepted)
    assert not bool(result.state.valid)
    preserved = result.state.replace(valid=corrupted.valid)
    _assert_tree_equal(preserved, corrupted)

    blocked = jax.jit(bridge.step)(result.state)
    _assert_tree_equal(blocked.state, result.state)
    assert not bool(blocked.trace.active)
    assert not bool(blocked.trace.accepted)
    assert not bool(blocked.trace.all_finite)
    np.testing.assert_array_equal(blocked.trace.planner_scores, np.zeros((2,)))
    np.testing.assert_array_equal(blocked.trace.helper_key_before, np.zeros((2,)))


@pytest.mark.parametrize("corruption", ("behavior-words", "grounded-telemetry"))
def test_child_clock_mismatch_rejects_the_outer_transition_atomically(
    corruption: str,
) -> None:
    config = _tiny_config(steps_per_phase=1)
    bridge = HiddenLearningPartnerPlanningBridge(config, JOINT_ADAPTIVE)
    initial = bridge.initialize(jr.key(71))
    if corruption == "behavior-words":
        corrupted = initial.replace(
            behavior=initial.behavior.replace(
                step_words=jnp.asarray((0, 1), dtype=jnp.uint32)
            )
        )
    else:
        corrupted = initial.replace(
            grounded=initial.grounded.replace(
                update_count=jnp.asarray(1, dtype=jnp.int32)
            )
        )

    with jax.disable_jit():
        result = bridge.step(corrupted)
    assert not bool(result.trace.accepted)
    assert not bool(result.state.valid)
    if corruption == "behavior-words":
        assert not bool(result.trace.behavior_proposal_applied)
    else:
        assert not bool(result.trace.grounded_proposal_applied)
    _assert_tree_equal(result.state.replace(valid=corrupted.valid), corrupted)


def test_validator_rejects_final_child_word_or_telemetry_corruption() -> None:
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=73,
        config=_tiny_config(steps_per_phase=1),
        jit_compile=False,
    )
    bad_behavior = run.final_state.behavior.replace(
        step_words=run.final_state.behavior.step_words.at[1].add(
            jnp.asarray(1, dtype=jnp.uint32)
        )
    )
    behavior_run = dataclasses.replace(
        run,
        final_state=run.final_state.replace(behavior=bad_behavior),
    )
    assert (
        "final behavior telemetry/exact words differ from committed writes"
        in validate_hidden_learning_partner_planning_run(behavior_run)
    )

    bad_grounded = run.final_state.grounded.replace(
        update_count=run.final_state.grounded.update_count - jnp.asarray(1, dtype=jnp.int32)
    )
    grounded_run = dataclasses.replace(
        run,
        final_state=run.final_state.replace(grounded=bad_grounded),
    )
    assert (
        "final grounded telemetry/exact words differ from committed writes"
        in validate_hidden_learning_partner_planning_run(grounded_run)
    )


def test_constant_channels_and_shuffled_planning_do_not_peek_pending_draw() -> None:
    config = _tiny_config(steps_per_phase=1)
    constant_zero = run_hidden_learning_partner_planning(
        CONSTANT_ZERO_DELIVERY,
        seed=23,
        config=config,
        jit_compile=False,
    )
    constant_one = run_hidden_learning_partner_planning(
        CONSTANT_ONE_DELIVERY,
        seed=23,
        config=config,
        jit_compile=False,
    )
    np.testing.assert_array_equal(constant_zero.trace.delivered_message, np.zeros(4))
    np.testing.assert_array_equal(constant_one.trace.delivered_message, np.ones(4))

    bridge = HiddenLearningPartnerPlanningBridge(config, SHUFFLED_DELIVERY)
    initial = bridge.initialize(jr.key(23))
    alternate_world = initial.world.replace(channel_key=jr.key(999_983))
    alternate = initial.replace(world=alternate_world)
    with jax.disable_jit():
        first = bridge.step(initial).trace
        second = bridge.step(alternate).trace
    np.testing.assert_array_equal(first.planner_scores, second.planner_scores)
    assert int(first.planner_message) == int(second.planner_message)
    assert int(first.ordinary_message) == int(second.ordinary_message)
    assert bool(first.planner_gate_draw) == bool(second.planner_gate_draw)


@pytest.mark.parametrize("n_phases", (4, 6))
def test_threshold_free_phase_diagnostics_align_and_reconstruct(n_phases: int) -> None:
    config = HiddenLearningPartnerPlanningConfig(
        phase_length=1,
        n_phases=n_phases,
        learning_rate=0.2,
        epsilon=0.2,
        behavior_step_size=0.1,
        grounded_step_size=0.1,
    )
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=101 + n_phases,
        config=config,
        jit_compile=False,
    )
    phases = run.metrics.phase_diagnostics
    assert phases.n_phases == n_phases
    assert phases.window_steps == 1
    assert phases.phase_index == tuple(range(n_phases))
    assert phases.hidden_context == tuple(index % 2 for index in range(n_phases))
    assert phases.phase_counts == (1,) * n_phases
    assert phases.leading_counts == (1,) * n_phases
    assert phases.trailing_counts == (1,) * n_phases
    assert phases.phase_valid == (True,) * n_phases

    rewards = tuple(float(value) for value in np.asarray(run.trace.reward))
    nll = tuple(float(value) for value in np.asarray(run.trace.behavior_nll))
    grounded_mse = tuple(
        float(value * value)
        for value in np.asarray(run.trace.grounded_reward_error, dtype=np.float64)
    )
    assert phases.mean_reward == rewards
    assert phases.leading_reward == rewards
    assert phases.trailing_reward == rewards
    assert phases.behavior_mean_nll == nll
    assert phases.grounded_reward_mse == grounded_mse
    assert phases.switch_cost_valid == (False,) + (True,) * (n_phases - 1)
    assert phases.switch_cost_counts == (0,) + (1,) * (n_phases - 1)
    assert phases.switch_cost == (0.0,) + tuple(
        rewards[index - 1] - rewards[index] for index in range(1, n_phases)
    )
    assert phases.recurrence_savings_valid == (False, False) + (True,) * (
        n_phases - 2
    )
    assert phases.recurrence_reference_phase == (-1, -1) + tuple(
        range(n_phases - 2)
    )
    assert phases.recurrence_counts == (0, 0) + (1,) * (n_phases - 2)
    assert phases.recurrence_savings == (0.0, 0.0) + tuple(
        rewards[index] - rewards[index - 2] for index in range(2, n_phases)
    )
    assert validate_hidden_learning_partner_planning_run(run) == ()

    tampered_phase = dataclasses.replace(
        phases,
        leading_reward=(phases.leading_reward[0] + 0.25,) + phases.leading_reward[1:],
    )
    tampered_metrics = dataclasses.replace(run.metrics, phase_diagnostics=tampered_phase)
    tampered = dataclasses.replace(run, metrics=tampered_metrics)
    assert "phase diagnostics differ from primitive trace reconstruction" in (
        validate_hidden_learning_partner_planning_run(tampered)
    )


def test_phase_diagnostics_use_distinct_multi_step_windows() -> None:
    config = HiddenLearningPartnerPlanningConfig(
        phase_length=8,
        n_phases=4,
        learning_rate=0.2,
        epsilon=0.2,
        behavior_step_size=0.1,
        grounded_step_size=0.1,
    )
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=131,
        config=config,
        jit_compile=False,
    )
    phases = run.metrics.phase_diagnostics
    rewards = np.asarray(run.trace.reward, dtype=np.float64).reshape(4, 8)
    behavior_nll = np.asarray(run.trace.behavior_nll, dtype=np.float64).reshape(4, 8)
    grounded_mse = np.square(
        np.asarray(run.trace.grounded_reward_error, dtype=np.float64).reshape(4, 8)
    )

    assert phases.window_steps == 2
    assert phases.phase_counts == (8,) * 4
    assert phases.leading_counts == (2,) * 4
    assert phases.trailing_counts == (2,) * 4
    assert phases.phase_valid == (True,) * 4
    assert phases.mean_reward == tuple(float(np.mean(row)) for row in rewards)
    assert phases.leading_reward == tuple(float(np.mean(row[:2])) for row in rewards)
    assert phases.trailing_reward == tuple(float(np.mean(row[-2:])) for row in rewards)
    assert phases.behavior_mean_nll == tuple(
        float(np.mean(row)) for row in behavior_nll
    )
    assert phases.grounded_reward_mse == tuple(
        float(np.mean(row)) for row in grounded_mse
    )
    assert phases.switch_cost == (0.0,) + tuple(
        float(np.mean(rewards[index - 1, -2:]) - np.mean(rewards[index, :2]))
        for index in range(1, 4)
    )
    assert phases.recurrence_savings == (0.0, 0.0) + tuple(
        float(np.mean(rewards[index, :2]) - np.mean(rewards[index - 2, :2]))
        for index in range(2, 4)
    )
    assert validate_hidden_learning_partner_planning_run(run) == ()


def test_validator_fails_closed_on_intervention_tampering() -> None:
    run = run_hidden_learning_partner_planning(
        JOINT_ADAPTIVE,
        seed=19,
        config=_tiny_config(steps_per_phase=2),
    )
    tampered_trace = run.trace.replace(
        potential_outcome_bound=run.trace.potential_outcome_bound.at[0].set(False)
    )
    tampered = dataclasses.replace(run, trace=tampered_trace)
    errors = validate_hidden_learning_partner_planning_run(tampered)
    assert "potential-outcome binding failed" in errors

    derived_trace = run.trace.replace(
        behavior_action_probability=run.trace.behavior_action_probability.at[0].add(
            jnp.float32(0.25)
        ),
        behavior_nll=run.trace.behavior_nll.at[1].add(jnp.float32(0.25)),
        grounded_reward_error=run.trace.grounded_reward_error.at[2].add(
            jnp.float32(0.25)
        ),
    )
    derived_tamper = dataclasses.replace(run, trace=derived_trace)
    derived_errors = validate_hidden_learning_partner_planning_run(derived_tamper)
    assert (
        "behavior action probability is not bound to the cached prediction"
        in derived_errors
    )
    assert "behavior NLL is not bound to the cached action probability" in derived_errors
    assert "grounded reward error is not bound to prediction and reward" in derived_errors

    bad_metrics = dataclasses.replace(
        run,
        metrics=dataclasses.replace(
            run.metrics,
            mean_reward=run.metrics.mean_reward + 0.125,
        ),
    )
    assert "raw metrics differ from primitive trace reconstruction" in (
        validate_hidden_learning_partner_planning_run(bad_metrics)
    )

    helper_before: list[np.ndarray] = []
    helper_after: list[np.ndarray] = []
    key = run.initial_state.learner.helper.key
    for _ in range(run.config.num_steps):
        helper_before.append(_key_words(key))
        key = jr.split(key, 4)[1]
        helper_after.append(_key_words(key))
    continuous_trace = run.trace.replace(
        helper_key_before=jnp.asarray(np.stack(helper_before), dtype=jnp.uint32),
        helper_key_after=jnp.asarray(np.stack(helper_after), dtype=jnp.uint32),
    )
    continuous_helper = run.final_state.learner.helper.replace(key=key)
    continuous_learner = run.final_state.learner.replace(helper=continuous_helper)
    continuous_final = run.final_state.replace(learner=continuous_learner)
    continuous_tamper = dataclasses.replace(
        run,
        trace=continuous_trace,
        final_state=continuous_final,
    )
    split_errors = validate_hidden_learning_partner_planning_run(continuous_tamper)
    assert "helper RNG split transition failed" in split_errors
    assert "helper RNG continuity failed" not in split_errors
    assert "helper RNG initial binding failed" not in split_errors
    assert "helper RNG final binding failed" not in split_errors
