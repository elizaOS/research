# mypy: disable-error-code="attr-defined,call-arg"
"""L0 contracts for bounded, real-state-anchored one-step Dyna."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.one_step_dyna import (
    ONE_STEP_DYNA_CONFIG_SCHEMA,
    ONE_STEP_DYNA_EVIDENCE_LEVEL,
    ONE_STEP_DYNA_SCIENTIFIC_PROMOTION_ALLOWED,
    OneStepDynaAuthority,
    OneStepDynaConfig,
    RealStateDynaAnchor,
    RealStateOneStepDyna,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.unit

REPRESENTATION_REVISION = jnp.asarray([0, 7], dtype=jnp.uint32)
ANCHOR = jnp.asarray([1.0, 0.0], dtype=jnp.float32)


@pytest.fixture(autouse=True)
def _run_contract_tests_eagerly() -> None:
    """Leave explicit compilation to the integration parity tests."""

    with jax.disable_jit():
        yield
    jax.clear_caches()


def _ensemble() -> WorldModelEnsemble:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        max_input_magnitude=1_000.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )


def _control() -> MultiHeadMLPLearner:
    return MultiHeadMLPLearner(
        n_heads=2,
        hidden_sizes=(),
        step_size=0.25,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _set_model_outputs(
    ensemble: WorldModelEnsemble,
    state: WorldModelEnsembleState,
    member_outputs: tuple[tuple[float, float, float, float], ...],
) -> WorldModelEnsembleState:
    """Install exact affine-intercept predictions without changing clocks."""

    assert len(member_outputs) == ensemble.config.ensemble_size
    members = []
    for member, outputs in zip(state.member_states, member_outputs, strict=True):
        learner = member.learner_state
        heads = learner.head_params.replace(
            weights=tuple(
                jnp.zeros_like(weight) for weight in learner.head_params.weights
            ),
            biases=tuple(
                jnp.asarray([value], dtype=jnp.float32) for value in outputs
            ),
        )
        members.append(
            member.replace(learner_state=learner.replace(head_params=heads))
        )
    result = cast(WorldModelEnsembleState, state.replace(member_states=tuple(members)))
    assert bool(ensemble.state_valid(result))
    return result


def _set_control_values(
    state: MultiHeadMLPState,
    *,
    action_zero: float = 0.0,
    action_one: float = 4.0,
) -> MultiHeadMLPState:
    heads = state.head_params.replace(
        weights=(
            jnp.asarray([[action_zero, 0.0]], dtype=jnp.float32),
            jnp.asarray([[action_one, 0.0]], dtype=jnp.float32),
        ),
        biases=(
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        ),
    )
    return cast(MultiHeadMLPState, state.replace(head_params=heads))


def _planner_config(**overrides: object) -> OneStepDynaConfig:
    defaults: dict[str, object] = {
        "anchor_capacity": 4,
        "backup_budget": 1,
        "min_action_support": 1,
        "max_epistemic_disagreement": 100.0,
        "max_residual_variance": 100.0,
        "require_residual_proxy_ready": False,
        "terminal_discount_threshold": 0.0,
        "max_anchor_records": 20,
        "max_planning_calls": 10,
        "max_planned_backups": 10,
    }
    defaults.update(overrides)
    return OneStepDynaConfig(**defaults)  # type: ignore[arg-type]


def _system(
    *,
    outputs: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 2.0, 0.5),
        (0.0, 0.0, 2.0, 0.5),
    ),
    config: OneStepDynaConfig | None = None,
) -> tuple[
    RealStateOneStepDyna,
    WorldModelEnsembleState,
    MultiHeadMLPState,
]:
    ensemble = _ensemble()
    model_state = _set_model_outputs(
        ensemble,
        ensemble.init(jr.key(1, impl="threefry2x32")),
        outputs,
    )
    control = _control()
    control_state = _set_control_values(
        control.init(2, jr.key(2, impl="threefry2x32"))
    )
    planner = RealStateOneStepDyna(
        ensemble,
        control,
        config or _planner_config(),
    )
    return planner, model_state, control_state


def _authority(
    model_state: WorldModelEnsembleState,
    control_state: MultiHeadMLPState,
    *,
    representation: jax.Array = REPRESENTATION_REVISION,
) -> OneStepDynaAuthority:
    return OneStepDynaAuthority(
        representation_revision_words=representation,
        model_revision_words=model_state.event_count_words,
        control_revision_words=control_state.step_words,
    )


def _record(
    planner: RealStateOneStepDyna,
    model_state: WorldModelEnsembleState,
    control_state: MultiHeadMLPState,
    *,
    decision: int = 1,
    action: int = 0,
    state: object | None = None,
):
    planner_state = (
        planner.init(
            jr.key(3, impl="threefry2x32"),
            REPRESENTATION_REVISION,
            model_state,
            control_state,
        )
        if state is None
        else state
    )
    authority = _authority(model_state, control_state)
    anchor = RealStateDynaAnchor(
        observation=ANCHOR,
        primitive_action=jnp.asarray(action, dtype=jnp.int32),
        decision_id_words=jnp.asarray([0, decision], dtype=jnp.uint32),
        authority=authority,
    )
    return planner.record_real_anchor(
        planner_state,
        model_state,
        control_state,
        anchor,
    )


def _materialize_keys(tree: object) -> object:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(materialize, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(_materialize_keys(left), _materialize_keys(right))


def test_config_construction_and_resources_are_strict_l0_contracts() -> None:
    config = _planner_config(backup_budget=3, max_planned_backups=6)
    payload = config.to_config()
    assert payload["schema"] == ONE_STEP_DYNA_CONFIG_SCHEMA
    assert payload["evidence_level"] == ONE_STEP_DYNA_EVIDENCE_LEVEL == "L0"
    assert payload["scientific_promotion_allowed"] is False
    assert ONE_STEP_DYNA_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert OneStepDynaConfig.from_config(payload) == config
    with pytest.raises(ValueError, match="fields"):
        OneStepDynaConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="max_planned_backups"):
        _planner_config(
            backup_budget=1,
            max_planning_calls=2,
            max_planned_backups=3,
        )

    planner, model_state, control_state = _system(config=config)
    restored = RealStateOneStepDyna.from_config(planner.to_config())
    assert restored.to_config() == planner.to_config()
    state = planner.init(
        jr.key(5, impl="threefry2x32"),
        REPRESENTATION_REVISION,
        model_state,
        control_state,
    )
    budget = planner.resource_budget
    assert budget.persistent_bytes_scope.startswith(
        "planner-owned-persistent-array-leaves-only"
    )
    assert budget.diagnostic_bytes_scope.endswith("not-a-measured-device-peak")
    assert budget.temporary_bytes_scope.startswith("not-measured")
    leaves = jax.tree.leaves(_materialize_keys(state))
    assert sum(np.asarray(leaf).nbytes for leaf in leaves) == budget.persistent_state_bytes
    assert budget.backup_budget == 3
    assert budget.max_ensemble_prediction_calls_per_call == 3
    assert budget.max_member_model_predictions_per_call == 6
    assert budget.max_control_target_forward_calls_per_call == 6
    assert budget.max_control_update_forward_calls_per_call == 3
    assert budget.max_control_forward_calls_per_call == 9
    assert budget.max_control_updates_per_call == 3
    assert budget.max_planning_rng_draws_per_call == 3
    assert budget.model_state_owned == 0
    assert budget.control_state_owned == 0
    assert budget.replay_capacity == config.anchor_capacity
    with pytest.raises(TypeError, match="typed scalar threefry"):
        planner.init(
            jr.key_data(jr.key(5)),
            REPRESENTATION_REVISION,
            model_state,
            control_state,
        )


def test_exact_target_uses_preupdate_reward_continuation_and_successor_value() -> None:
    planner, model_state, control_state = _system()
    recorded = _record(planner, model_state, control_state)
    assert bool(recorded.diagnostics.applied)
    model_before = _materialize_keys(model_state)
    trace_before = _materialize_keys(
        (
            control_state.trunk_traces,
            control_state.head_traces,
            control_state.hidden_unit_utilities,
        )
    )

    result = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.diagnostics.applied[0])
    assert int(result.diagnostics.selected_actions[0]) == 0
    assert float(result.diagnostics.predicted_rewards[0]) == pytest.approx(2.0)
    assert float(result.diagnostics.predicted_continuations[0]) == pytest.approx(0.5)
    assert float(result.diagnostics.successor_values[0]) == pytest.approx(4.0)
    assert float(result.diagnostics.control_targets[0]) == pytest.approx(4.0)
    assert float(result.diagnostics.td_errors[0]) == pytest.approx(4.0)
    chex.assert_trees_all_equal(
        result.diagnostics.pre_model_revision_words,
        result.diagnostics.post_model_revision_words,
    )
    _assert_tree_equal(model_state, model_before)
    _assert_tree_equal(
        (
            result.control_state.trunk_traces,
            result.control_state.head_traces,
            result.control_state.hidden_unit_utilities,
        ),
        trace_before,
    )
    assert int(result.control_state.step_words[1]) == 1
    assert int(result.state.planning_attempt_count_words[1]) == 1
    assert int(result.state.planned_backup_count_words[1]) == 1
    assert int(result.state.rejected_backup_count_words[1]) == 0
    assert bool(planner.state_valid(result.state))


def test_zero_continuation_removes_successor_value_from_target() -> None:
    planner, model_state, control_state = _system(
        outputs=(
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0, 0.0),
        )
    )
    recorded = _record(planner, model_state, control_state)
    result = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert bool(result.diagnostics.applied[0])
    assert float(result.diagnostics.successor_values[0]) == pytest.approx(4.0)
    assert float(result.diagnostics.predicted_continuations[0]) == 0.0
    assert float(result.diagnostics.control_targets[0]) == pytest.approx(2.0)


def test_post_real_update_snapshot_advances_monotonically_and_rollback_is_rejected() -> None:
    planner, model_state, control_state = _system()
    recorded = _record(planner, model_state, control_state)
    model_update = planner.ensemble.update(
        model_state,
        ANCHOR,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(3.0, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        ANCHOR,
    )
    assert bool(model_update.diagnostics.applied)
    control_update = planner.control_learner.update(
        control_state,
        ANCHOR,
        jnp.asarray([1.0, jnp.nan], dtype=jnp.float32),
    )
    assert bool(control_update.update_applied)
    assert not bool(
        jnp.array_equal(
            model_update.state.event_count_words,
            model_state.event_count_words,
        )
    )

    causal = planner.plan(
        recorded.state,
        model_update.state,
        control_update.state,
        _authority(model_update.state, control_update.state),
    )
    assert bool(causal.diagnostics.authority_valid)
    assert bool(causal.diagnostics.transaction_applied)
    chex.assert_trees_all_equal(
        causal.state.bound_model_revision_words,
        model_update.state.event_count_words,
    )

    rolled_back = planner.plan(
        causal.state,
        model_state,
        causal.control_state,
        _authority(model_state, causal.control_state),
    )
    assert not bool(rolled_back.diagnostics.authority_valid)
    assert not bool(rolled_back.diagnostics.transaction_applied)
    _assert_tree_equal(rolled_back.state, causal.state)
    _assert_tree_equal(rolled_back.control_state, causal.control_state)

    stale_control = planner.plan(
        causal.state,
        model_update.state,
        control_state,
        _authority(model_update.state, control_state),
    )
    assert not bool(stale_control.diagnostics.authority_valid)
    assert not bool(stale_control.diagnostics.transaction_applied)
    _assert_tree_equal(stale_control.state, causal.state)


def test_epistemically_uncertain_dream_is_rejected_before_control_update() -> None:
    planner, model_state, control_state = _system(
        outputs=(
            (0.0, 0.0, -10.0, 0.5),
            (0.0, 0.0, 10.0, 0.5),
        ),
        config=_planner_config(max_epistemic_disagreement=0.01),
    )
    recorded = _record(planner, model_state, control_state)
    result = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.epistemic_valid[0])
    assert not bool(result.diagnostics.guard_passed[0])
    assert not bool(result.diagnostics.child_update_applied[0])
    assert not bool(result.diagnostics.applied[0])
    _assert_tree_equal(result.control_state, control_state)
    assert int(result.state.planning_attempt_count_words[1]) == 1
    assert int(result.state.rejected_backup_count_words[1]) == 1


def test_support_and_residual_readiness_are_independent_fail_closed_guards() -> None:
    support_planner, model_state, control_state = _system(
        config=_planner_config(min_action_support=2)
    )
    support_record = _record(support_planner, model_state, control_state)
    unsupported = support_planner.plan(
        support_record.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert int(unsupported.diagnostics.action_support_counts[0]) == 1
    assert not bool(unsupported.diagnostics.support_valid[0])
    assert not bool(unsupported.diagnostics.applied[0])

    residual_planner, residual_model, residual_control = _system(
        config=_planner_config(require_residual_proxy_ready=True)
    )
    residual_record = _record(
        residual_planner,
        residual_model,
        residual_control,
    )
    cold = residual_planner.plan(
        residual_record.state,
        residual_model,
        residual_control,
        _authority(residual_model, residual_control),
    )
    assert not bool(cold.diagnostics.residual_proxy_ready[0])
    assert not bool(cold.diagnostics.applied[0])


def test_member_termination_disagreement_vetoes_the_backup() -> None:
    planner, model_state, control_state = _system(
        outputs=(
            (0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0, 0.5),
        ),
        config=_planner_config(max_epistemic_disagreement=100.0),
    )
    recorded = _record(planner, model_state, control_state)
    result = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert not bool(result.diagnostics.termination_agreement[0])
    assert not bool(result.diagnostics.guard_passed[0])
    assert not bool(result.diagnostics.applied[0])
    _assert_tree_equal(result.control_state, control_state)


def test_invalid_model_snapshot_is_an_atomic_noop_including_planning_rng() -> None:
    planner, model_state, control_state = _system()
    recorded = _record(planner, model_state, control_state)
    invalid_model = model_state.replace(
        residual_variances=model_state.residual_variances.at[0, 0].set(jnp.nan)
    )
    result = planner.plan(
        recorded.state,
        invalid_model,
        control_state,
        _authority(invalid_model, control_state),
    )
    assert not bool(result.diagnostics.model_state_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_equal(result.state, recorded.state)
    _assert_tree_equal(result.control_state, control_state)


def test_anchor_identity_and_revision_tamper_fail_closed() -> None:
    planner, model_state, control_state = _system()
    recorded = _record(planner, model_state, control_state)
    tampered_state = recorded.state.replace(
        anchor_actions=recorded.state.anchor_actions.at[0].set(1)
    )
    assert not bool(planner.state_valid(tampered_state))
    blocked = planner.plan(
        tampered_state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert not bool(blocked.diagnostics.state_valid)
    _assert_tree_equal(blocked.state, tampered_state)
    _assert_tree_equal(blocked.control_state, control_state)

    wrong_revision = _authority(model_state, control_state).replace(
        model_revision_words=jnp.asarray([0, 1], dtype=jnp.uint32)
    )
    revision_blocked = planner.plan(
        recorded.state,
        model_state,
        control_state,
        wrong_revision,
    )
    assert not bool(revision_blocked.diagnostics.authority_valid)
    _assert_tree_equal(revision_blocked.state, recorded.state)

    duplicate = _record(
        planner,
        model_state,
        control_state,
        decision=1,
        state=recorded.state,
    )
    assert not bool(duplicate.diagnostics.decision_identity_valid)
    assert not bool(duplicate.diagnostics.applied)
    _assert_tree_equal(duplicate.state, recorded.state)


def test_exact_lifetime_backup_cap_applies_partial_final_budget_then_stops() -> None:
    config = _planner_config(
        backup_budget=3,
        max_planning_calls=3,
        max_planned_backups=4,
    )
    planner, model_state, control_state = _system(config=config)
    recorded = _record(planner, model_state, control_state)
    first = planner.plan(
        recorded.state,
        model_state,
        control_state,
        _authority(model_state, control_state),
    )
    assert int(first.diagnostics.applied_count) == 3
    second = planner.plan(
        first.state,
        model_state,
        first.control_state,
        _authority(model_state, first.control_state),
    )
    assert int(second.diagnostics.applied_count) == 1
    assert int(second.state.planned_backup_count_words[1]) == 4
    assert int(second.state.planning_attempt_count_words[1]) == 6
    assert int(second.state.rejected_backup_count_words[1]) == 2

    exhausted = planner.plan(
        second.state,
        model_state,
        second.control_state,
        _authority(model_state, second.control_state),
    )
    assert not bool(exhausted.diagnostics.planned_backup_capacity_available)
    assert not bool(exhausted.diagnostics.transaction_applied)
    _assert_tree_equal(exhausted.state, second.state)
    _assert_tree_equal(exhausted.control_state, second.control_state)


def test_ring_overwrite_recomputes_support_and_preserves_ordered_identity() -> None:
    planner, model_state, control_state = _system(
        config=_planner_config(anchor_capacity=2, max_anchor_records=4)
    )
    state = None
    for decision, action in ((1, 0), (2, 1), (3, 1)):
        result = _record(
            planner,
            model_state,
            control_state,
            decision=decision,
            action=action,
            state=state,
        )
        assert bool(result.diagnostics.applied)
        state = result.state
    assert state is not None
    assert int(state.size) == 2
    chex.assert_trees_all_equal(
        state.action_support_counts,
        jnp.asarray([0, 2], dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        state.last_decision_id_words,
        jnp.asarray([0, 3], dtype=jnp.uint32),
    )
    assert bool(planner.state_valid(state))


def test_config_rejects_ambiguous_types_and_child_compatibility() -> None:
    with pytest.raises(ValueError, match="backup_budget"):
        OneStepDynaConfig(backup_budget=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strict boolean"):
        OneStepDynaConfig(require_termination_agreement=1)  # type: ignore[arg-type]
    ensemble = _ensemble()
    with pytest.raises(ValueError, match="heads"):
        RealStateOneStepDyna(
            ensemble,
            MultiHeadMLPLearner(n_heads=3, hidden_sizes=()),
        )
