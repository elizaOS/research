# mypy: disable-error-code="attr-defined,call-arg"
"""Owner-bound LearningValueRouter integration in PrototypeAgent."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.balanced_state_objectives import (
    BalancedStateObjectives,
    BalancedStateObjectivesConfig,
)
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectives,
    ComprehensiveStateObjectivesConfig,
)
from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.learning_value_router import (
    LearningValueRouterConfig,
    LearningValueRouterResult,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeCandidateUpdateAuditEvidence,
    PrototypeLearningValueRouterState,
    PrototypeTransition,
    load_prototype_checkpoint,
    measure_prototype_agent_state_resources,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_balanced_state_objectives import (
    PrototypeBalancedStateObjectives,
)
from alberta_framework.core.prototype_comprehensive_state_objectives import (
    PrototypeComprehensiveStateObjectives,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = pytest.mark.unit

RAW_DIM = 1
FEATURE_DIM = 2
N_ACTIONS = 2
PARAMETER_COUNT = 12


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_enabled_transition_has_eager_jit_parity":
        yield
    else:
        with jax.disable_jit():
            yield


def _oak_config() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=FEATURE_DIM,
            n_primitive_actions=N_ACTIONS,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _builder_config() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=1,
        include_raw_observation=True,
        step_size=0.1,
        gradient_clip=10.0,
    )


def _ensemble_config() -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=ActionConditionedWorldModelConfig(
            observation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.95,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.8,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=FEATURE_DIM + 2,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            fast_loss_decay=0.5,
            slow_loss_decay=0.9,
            max_input_magnitude=100.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=2,
        bootstrap_probability=0.8,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
    )


def _config(*, router: bool, discard_gradient: bool = False) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=_oak_config(),
        state_builder=_builder_config(),
        world_model_ensemble=_ensemble_config(),
        learn_state_builder_from_world_model=not discard_gradient,
        representation_gradient_mixer=(
            RepresentationGradientMixerConfig(
                representation_dim=FEATURE_DIM,
                mode="discard",
            )
            if discard_gradient
            else None
        ),
        gradient_joy=CandidateUpdateAuditConfig(
            candidate_semantics="update",
            max_update_norm=10.0,
        ),
        learning_value_router=(
            LearningValueRouterConfig(
                normalization_min_count=2,
                max_steps=10,
            )
            if router
            else None
        ),
    )


def _transition(state: PrototypeAgentState, value: float) -> PrototypeTransition:
    next_observation = jnp.asarray([value], dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(0.25 + 0.1 * value, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _sidecar(
    state: PrototypeAgentState,
    *,
    advantage: float,
) -> PrototypeCandidateUpdateAuditEvidence:
    available = jnp.asarray(True, dtype=jnp.bool_)
    return PrototypeCandidateUpdateAuditEvidence(
        decision_id=state.current_decision_id,
        objective_probe_gradient=jnp.ones((PARAMETER_COUNT,), dtype=jnp.float32),
        retention_probe_gradient=jnp.ones((PARAMETER_COUNT,), dtype=jnp.float32),
        safety_cost_gradient=jnp.ones((PARAMETER_COUNT,), dtype=jnp.float32),
        objective_probe_available=available,
        retention_probe_available=available,
        safety_probe_available=available,
        probe_independence_attested=available,
        advantage=jnp.asarray(advantage, dtype=jnp.float32),
        action_surprisal=jnp.asarray(0.5, dtype=jnp.float32),
        safety_cost=jnp.asarray(0.25, dtype=jnp.float32),
        advantage_available=available,
        action_surprisal_available=available,
        safety_cost_available=available,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    def materialize_key(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    left_leaves, left_structure = jax.tree.flatten(
        jax.tree.map(materialize_key, left)
    )
    right_leaves, right_structure = jax.tree.flatten(
        jax.tree.map(materialize_key, right)
    )
    assert cast(Any, left_structure) == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_router_config_is_strict_round_trips_and_disabled_bytes_stay_legacy() -> None:
    assert alberta.PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA == (
        PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA
    )
    assert core.PrototypeLearningValueRouterState is (
        PrototypeLearningValueRouterState
    )
    enabled = _config(router=True)
    payload = enabled.to_config()
    assert payload["learning_value_router"] == cast(
        LearningValueRouterConfig,
        enabled.learning_value_router,
    ).to_config()
    assert PrototypeAgentConfig.from_config(copy.deepcopy(payload)) == enabled

    disabled = _config(router=False)
    assert "learning_value_router" not in disabled.to_config()
    disabled_state = PrototypeAgent(disabled).init(jr.key(1))
    assert not isinstance(
        disabled_state.state_builder_state,
        PrototypeLearningValueRouterState,
    )

    with pytest.raises(ValueError, match="candidate-update audit"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=_builder_config(),
            world_model_ensemble=_ensemble_config(),
            learn_state_builder_from_world_model=True,
            learning_value_router=LearningValueRouterConfig(),
        )
    with pytest.raises(ValueError, match="LearningValueRouterConfig"):
        PrototypeAgentConfig(
            oak=_oak_config(),
            state_builder=_builder_config(),
            world_model_ensemble=_ensemble_config(),
            learn_state_builder_from_world_model=True,
            gradient_joy=CandidateUpdateAuditConfig(candidate_semantics="update"),
            learning_value_router=cast(LearningValueRouterConfig, object()),
        )


def test_outer_objective_adapters_reject_router_state_wrapper_explicitly() -> None:
    prototype = PrototypeAgent(_config(router=True))
    balanced = BalancedStateObjectives(
        BalancedStateObjectivesConfig(
            representation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
        )
    )
    with pytest.raises(ValueError, match="does not compose with learning_value_router"):
        PrototypeBalancedStateObjectives(prototype, balanced)

    comprehensive = ComprehensiveStateObjectives(
        ComprehensiveStateObjectivesConfig(
            representation_dim=FEATURE_DIM,
            observation_target_dim=RAW_DIM,
            n_actions=N_ACTIONS,
        )
    )
    with pytest.raises(ValueError, match="does not compose with learning_value_router"):
        PrototypeComprehensiveStateObjectives(prototype, comprehensive)


def test_router_state_is_single_owner_bound_resource_and_validated() -> None:
    enabled_agent = PrototypeAgent(_config(router=True))
    enabled_state = enabled_agent.init(jr.key(2))
    wrapper = cast(
        PrototypeLearningValueRouterState,
        enabled_state.state_builder_state,
    )
    assert isinstance(wrapper, PrototypeLearningValueRouterState)
    assert int(wrapper.learning_value_router_state.step_count) == 0
    budget = enabled_agent.learning_value_router_resource_budget
    assert budget is not None

    disabled_agent = PrototypeAgent(_config(router=False))
    disabled_state = disabled_agent.init(jr.key(2))
    enabled_resources = measure_prototype_agent_state_resources(enabled_state)
    disabled_resources = measure_prototype_agent_state_resources(disabled_state)
    assert enabled_resources.total_nbytes - disabled_resources.total_nbytes == (
        budget.persistent_state_bytes
    )

    corrupt_router = wrapper.learning_value_router_state.replace(
        step_count=jnp.asarray(1, dtype=jnp.int32),
    )
    corrupt = enabled_state.replace(
        state_builder_state=wrapper.replace(
            learning_value_router_state=corrupt_router,
        )
    )
    assert not bool(enabled_agent.validate_state(corrupt))


def test_route_uses_raw_channels_once_and_producer_availability_ignores_candidate() -> None:
    agent = PrototypeAgent(_config(router=True, discard_gradient=True))
    state = agent.start(agent.init(jr.key(3)), jnp.asarray([0.1], dtype=jnp.float32))
    result = None
    for index, value in enumerate((0.2, -0.3, 0.7), start=1):
        result = agent.update_transition(
            state,
            _transition(state, value),
            candidate_update_audit_evidence=_sidecar(
                state,
                advantage=float(index),
            ),
        )
        assert bool(result.transition_diagnostics.valid)
        state = result.state

    assert result is not None
    routed = result.learning_value_router_result
    assert isinstance(routed, LearningValueRouterResult)
    wrapper = cast(PrototypeLearningValueRouterState, state.state_builder_state)
    assert int(wrapper.learning_value_router_state.step_count) == 3
    assert not bool(result.mixed_representation_gradient_valid)
    assert bool(routed.candidate_update_audit_evidence.ready)
    raw = routed.candidate_update_audit_evidence.values
    normalized = routed.candidate_update_audit_evidence.normalized_values
    assert np.asarray(raw.advantage).view(np.uint32) == np.asarray(
        jnp.asarray(3.0, dtype=jnp.float32)
    ).view(np.uint32)
    expected_delight = jnp.asarray(3.0, dtype=jnp.float32) * jnp.asarray(
        0.5,
        dtype=jnp.float32,
    )
    assert np.asarray(raw.delight).view(np.uint32) == np.asarray(
        expected_delight
    ).view(np.uint32)
    assert float(normalized.advantage) != float(raw.advantage)
    assert bool(routed.candidate_update_audit_evidence.availability.advantage)
    assert bool(routed.candidate_update_audit_evidence.availability.safety_cost)

    application = result.candidate_update_audit_application
    assert application is not None
    assert not bool(application.assessment.diagnostics.objective_probe_available)
    assert bool(application.assessment.channel_availability.advantage)
    assert bool(application.assessment.channel_availability.safety_cost)
    assert float(application.assessment.learning_value.advantage) == float(
        raw.advantage
    )
    assert float(application.assessment.learning_value.advantage) != float(
        normalized.advantage
    )


def test_rejected_outer_transition_preserves_router_and_reports_no_router_work() -> None:
    agent = PrototypeAgent(_config(router=True))
    state = agent.start(agent.init(jr.key(4)), jnp.asarray([0.1], dtype=jnp.float32))
    accepted = agent.update_transition(
        state,
        _transition(state, 0.2),
        candidate_update_audit_evidence=_sidecar(state, advantage=1.0),
    )
    before = accepted.state
    stale_transition = _transition(before, 0.4).replace(
        decision_id=before.current_decision_id.at[3].add(
            jnp.asarray(1, dtype=jnp.uint32)
        )
    )
    rejected = agent.update_transition(
        before,
        stale_transition,
        candidate_update_audit_evidence=_sidecar(before, advantage=2.0),
    )

    assert bool(rejected.transition_diagnostics.rejected)
    _assert_tree_equal(rejected.state, before)
    routed = rejected.learning_value_router_result
    assert isinstance(routed, LearningValueRouterResult)
    assert not bool(routed.diagnostics.normalization_state_updated)
    assert not bool(routed.candidate_update_audit_evidence.ready)
    wrapper = cast(
        PrototypeLearningValueRouterState,
        rejected.state.state_builder_state,
    )
    assert int(wrapper.learning_value_router_state.step_count) == 1


def test_router_uses_v19_checkpoint_while_disabled_schema_is_unchanged(
    tmp_path: Path,
) -> None:
    enabled_agent = PrototypeAgent(_config(router=True))
    enabled_state = enabled_agent.start(
        enabled_agent.init(jr.key(5)),
        jnp.asarray([0.1], dtype=jnp.float32),
    )
    path = tmp_path / "router"
    save_prototype_checkpoint(enabled_agent, enabled_state, path)
    assert load_checkpoint_metadata(path)["schema"] == (
        PROTOTYPE_LEARNING_VALUE_ROUTER_CHECKPOINT_SCHEMA
    )
    restored_agent, restored_state = load_prototype_checkpoint(path)
    assert restored_agent.to_config() == enabled_agent.to_config()
    _assert_tree_equal(restored_state, enabled_state)

    disabled_agent = PrototypeAgent(_config(router=False))
    disabled_state = disabled_agent.start(
        disabled_agent.init(jr.key(6)),
        jnp.asarray([0.1], dtype=jnp.float32),
    )
    disabled_path = tmp_path / "disabled"
    save_prototype_checkpoint(disabled_agent, disabled_state, disabled_path)
    assert load_checkpoint_metadata(disabled_path)["schema"] == (
        PROTOTYPE_CHECKPOINT_SCHEMA
    )


def test_enabled_transition_has_eager_jit_parity() -> None:
    agent = PrototypeAgent(_config(router=True))
    state = agent.start(agent.init(jr.key(7)), jnp.asarray([0.1], dtype=jnp.float32))
    transition = _transition(state, 0.35)
    sidecar = _sidecar(state, advantage=1.25)

    eager = agent.update_transition(
        state,
        transition,
        candidate_update_audit_evidence=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(state, transition, sidecar)

    _assert_tree_equal(compiled.state, eager.state)
    _assert_tree_equal(
        compiled.learning_value_router_result,
        eager.learning_value_router_result,
    )
    _assert_tree_equal(
        compiled.candidate_update_audit_application,
        eager.candidate_update_audit_application,
    )
