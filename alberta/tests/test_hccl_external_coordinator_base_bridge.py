# mypy: disable-error-code="attr-defined,call-arg,no-any-return,type-var"
"""Base-only HCCL transactions with two external learned-state coordinators."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.core as core_api
from alberta_framework.core.delight import CandidateUpdateAuditConfig
from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinatorConfig,
)
from alberta_framework.core.feature_bank_router import FeatureBankRouterConfig
from alberta_framework.core.hccl_external_coordinator_base_bridge import (
    HCCL_EXTERNAL_COORDINATOR_BASE_STATUS,
    HCCLExternalCoordinatorBaseBridge,
    HCCLExternalCoordinatorBaseBridgeConfig,
    HCCLExternalCoordinatorBaseBridgeState,
    load_hccl_external_coordinator_base_checkpoint,
    measure_hccl_external_coordinator_base_state_nbytes,
    save_hccl_external_coordinator_base_checkpoint,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterConfig,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.learning_value_router import LearningValueRouterConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_routed_linear_world_model_ensemble_adapter import (
    PrototypeRoutedLinearWorldModelEnsembleAdapterConfig,
)
from alberta_framework.core.routed_linear_world_model_ensemble import (
    RoutedLinearWorldModelEnsembleConfig,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    LearnableGRUStateBuilderConfig,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RAW_DIM = 16
HIDDEN_DIM = 1
BASE_DIM = RAW_DIM + HIDDEN_DIM
PAIR_SLOTS = 1
TOTAL_DIM = BASE_DIM + PAIR_SLOTS
N_ACTIONS = 2
N_DEMONS = 1
TARGET_DIM = BASE_DIM + 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> object:
    with jax.disable_jit():
        yield


def _coordinator_config(*, max_events: int = 8) -> ExternalLearnedStateRouterAuditCoordinatorConfig:
    feature = PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=PAIR_SLOTS,
        candidate_pair_slots=2,
        n_tasks=1 + N_DEMONS,
        n_options=1,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=0,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=8,
        managed_horde_demons=N_DEMONS,
    )
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1_000_000.0,
                    max_option_steps=8,
                ),
            ),
            observation_dim=TOTAL_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            option_step_size=0.01,
            epsilon_base=0.0,
            epsilon_option=0.0,
            option_planning_backups_per_step=0,
        )
    )
    horde = create_horde_spec(
        (
            GVFSpec(
                name="prediction",
                demon_type=DemonType.PREDICTION,
                gamma=0.5,
                lamda=0.0,
                cumulant_index=0,
            ),
        )
    )
    prototype = PrototypeAgentConfig(
        oak=oak,
        horde_spec=horde,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        experiential_memory=None,
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=feature,
    )
    ensemble = RoutedLinearWorldModelEnsembleConfig(
        router=FeatureBankRouterConfig(base_dim=BASE_DIM, active_slots=PAIR_SLOTS),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=BASE_DIM,
            n_actions=N_ACTIONS,
            gamma=0.99,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
            error_decay=0.5,
            include_action_interactions=False,
        ),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=TARGET_DIM,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            max_input_magnitude=1_000.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=2,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-3,
        max_events=8,
        carry_survivors=True,
    )
    return ExternalLearnedStateRouterAuditCoordinatorConfig(
        builder=LearnableGRUStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            step_size=0.01,
            gradient_clip=10.0,
            initialization_scale=0.2,
            include_raw_observation=True,
        ),
        inner=PrototypeRoutedLinearWorldModelEnsembleAdapterConfig(
            prototype=prototype,
            ensemble=ensemble,
        ),
        learning_value_router=LearningValueRouterConfig(max_steps=8),
        candidate_audit=CandidateUpdateAuditConfig(candidate_semantics="update"),
        max_events=max_events,
    )


def _config(
    *,
    agent_0_max_events: int = 8,
    agent_1_max_events: int = 8,
) -> HCCLExternalCoordinatorBaseBridgeConfig:
    return HCCLExternalCoordinatorBaseBridgeConfig(
        hccl=HCCLWorldAttributionAdapterConfig(
            proposal_owner_digest=(
                0x10203040,
                0x50607080,
                0x90A0B0C0,
                0xD0E0F001,
                0x12345678,
                0x9ABCDEF0,
                0x0F1E2D3C,
                0x4B5A6978,
            )
        ),
        agent_0=_coordinator_config(max_events=agent_0_max_events),
        agent_1=_coordinator_config(max_events=agent_1_max_events),
        binding_owner_digest=(
            0xCAFEBABE,
            0x0BADF00D,
            0x13579BDF,
            0x2468ACE1,
            0x31415927,
            0x27182819,
            0x11235813,
            0x21345591,
        ),
    )


def _bridge(
    *,
    agent_0_max_events: int = 8,
    agent_1_max_events: int = 8,
) -> tuple[HCCLExternalCoordinatorBaseBridge, HCCLExternalCoordinatorBaseBridgeState]:
    bridge = HCCLExternalCoordinatorBaseBridge(
        _config(
            agent_0_max_events=agent_0_max_events,
            agent_1_max_events=agent_1_max_events,
        )
    )
    return bridge, bridge.init(jr.key(7))


def _typed_keys(value: object) -> list[np.ndarray[Any, Any]]:
    keys: list[np.ndarray[Any, Any]] = []
    for leaf in jax.tree.leaves(value):
        array = jnp.asarray(leaf)
        if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
            keys.append(np.asarray(jr.key_data(array)))
    return keys


def _pp(result: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf[4], result.hccl_result.world_proposals)


def test_config_exports_exact_three_owner_state_and_independent_start() -> None:
    bridge, state = _bridge()
    payload = bridge.to_config()
    assert payload["mechanism_status"] == HCCL_EXTERNAL_COORDINATOR_BASE_STATUS
    assert payload["mechanism_status"] == "l0-development-hccl-two-coordinator-base-only"
    assert payload["hccl_state_owners"] == 1
    assert payload["external_coordinator_state_owners"] == 2
    assert payload["base_only_ablation"] is True
    assert payload["memory_layer_authority"] is False
    assert payload["planner_layer_authority"] is False
    assert payload["delight_or_actor_backward"] is False
    assert payload["composite_jit_supported"] is False
    for name in (
        "schedule_execution_authorized",
        "seed_authority",
        "output_writes_authorized",
        "artifact_authorized",
        "threshold_authorized",
        "evidence_authorized",
        "promotion_authorized",
    ):
        assert payload[name] is False
    assert HCCLExternalCoordinatorBaseBridge.from_config(payload).to_config() == payload
    assert alberta_framework.HCCLExternalCoordinatorBaseBridge is (
        HCCLExternalCoordinatorBaseBridge
    )
    assert core_api.HCCLExternalCoordinatorBaseBridgeConfig is (
        HCCLExternalCoordinatorBaseBridgeConfig
    )
    assert tuple(state.__dataclass_fields__) == (
        "hccl_state",
        "agent_0_state",
        "agent_1_state",
    )
    assert bool(bridge.state_valid(state))
    observations = bridge.hccl.world.observe(state.hccl_state.world_state)
    chex.assert_trees_all_equal(state.agent_0_state.current_raw_observation, observations[0])
    chex.assert_trees_all_equal(state.agent_1_state.current_raw_observation, observations[1])
    left_keys = _typed_keys(state.agent_0_state)
    right_keys = _typed_keys(state.agent_1_state)
    assert len(left_keys) == len(right_keys) > 0
    assert any(not np.array_equal(left, right) for left, right in zip(left_keys, right_keys))


def test_base_binding_stages_pp_and_updates_each_coordinator_once() -> None:
    bridge, state = _bridge()
    event = bridge.prepare_event(state)
    binding = bridge.bind_base_actions(state, event)
    cached = jnp.stack((state.agent_0_state.current_action, state.agent_1_state.current_action))
    decisions = jnp.stack(
        (
            state.agent_0_state.current_decision_id,
            state.agent_1_state.current_decision_id,
        )
    )
    chex.assert_trees_all_equal(binding.coordinator_decision_words, decisions)
    chex.assert_trees_all_equal(binding.coordinator_lifecycle_words, decisions[:, :2])
    chex.assert_trees_all_equal(binding, bridge.bind_base_actions(state, event))
    for receipt in (binding.base, binding.memory, binding.planner):
        chex.assert_trees_all_equal(receipt.actions_before_mask, cached)
        chex.assert_trees_all_equal(receipt.actions_after_mask, cached)
        chex.assert_trees_all_equal(
            receipt.hard_action_masks,
            jnp.ones((2, 2), dtype=jnp.bool_),
        )
    identities = np.concatenate(
        tuple(
            np.asarray(receipt.action_receipt_identity_words)
            for receipt in (binding.base, binding.memory, binding.planner)
        ),
        axis=0,
    )
    assert np.unique(identities, axis=0).shape[0] == 6
    for agent in range(2):
        excluded = (
            jnp.ones((2, 2), dtype=jnp.bool_)
            .at[agent, cached[agent]]
            .set(False)
        )
        with pytest.raises(ValueError, match="cached coordinator action"):
            bridge.bind_base_actions(state, event, hard_action_masks=excluded)

    result = bridge.stage(
        state,
        event,
        binding,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(result.update_applied)
    assert bool(result.hccl_commit_applied)
    assert bool(result.agent_0_update_applied)
    assert bool(result.agent_1_update_applied)
    assert bool(result.base_only_ablation_valid)
    assert bool(result.memory_contrasts_zero)
    assert bool(result.planner_contrasts_zero)
    assert bool(result.total_stack_contrast_zero)
    assert not bool(result.delight_or_actor_backward)
    assert int(result.work.world_proposal_calls) == 8
    assert int(result.work.attribution_proposal_calls) == 8
    chex.assert_trees_all_equal(
        result.work.coordinator_update_calls,
        jnp.ones((2,), dtype=jnp.int32),
    )
    pp = _pp(result)
    for index, transition in enumerate(
        (result.agent_0_transition, result.agent_1_transition)
    ):
        chex.assert_trees_all_equal(transition.action, cached[index])
        chex.assert_trees_all_equal(transition.reward, pp.signals.net_reward[index])
        chex.assert_trees_all_equal(transition.next_observation, pp.next_observation[index])
        chex.assert_trees_all_equal(
            transition.next_decision_observation,
            pp.next_observation[index],
        )
    chex.assert_trees_all_equal(
        result.state.hccl_state.world_state.step_words,
        jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.agent_0_state.event_words,
        result.state.hccl_state.world_state.step_words,
    )
    chex.assert_trees_all_equal(
        result.state.agent_1_state.event_words,
        result.state.hccl_state.world_state.step_words,
    )
    assert bool(bridge.state_valid(result.state))


def test_tamper_downstream_stale_and_retry_are_atomic_and_jit_rejects() -> None:
    bridge, state = _bridge()
    event = bridge.prepare_event(state)
    binding = bridge.bind_base_actions(state, event)
    tampered = dataclasses.replace(
        binding,
        content_tag_words=binding.content_tag_words.at[0].add(jnp.uint32(1)),
    )
    rejected = bridge.stage(
        state,
        event,
        tampered,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(rejected.binding_matches_source)
    assert bool(rejected.hccl_result.update_applied)
    assert bool(rejected.agent_0_result.diagnostics.transaction_applied)
    assert bool(rejected.agent_1_result.diagnostics.transaction_applied)
    assert not bool(rejected.hccl_commit_applied)
    assert not bool(rejected.agent_0_update_applied)
    assert not bool(rejected.agent_1_update_applied)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)

    downstream = bridge.stage(
        state,
        event,
        binding,
        downstream_candidate_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(downstream.hccl_result.update_applied)
    assert bool(downstream.agent_0_result.diagnostics.transaction_applied)
    assert bool(downstream.agent_1_result.diagnostics.transaction_applied)
    assert not bool(downstream.update_applied)
    chex.assert_trees_all_equal(downstream.state, state)

    retry = bridge.stage(
        state,
        event,
        binding,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(retry.update_applied)
    stale = bridge.stage(
        retry.state,
        event,
        binding,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(stale.event_receipt_valid)
    assert not bool(stale.binding_matches_source)
    assert not bool(stale.update_applied)
    chex.assert_trees_all_equal(stale.state, retry.state)

    with jax.disable_jit(False):
        with pytest.raises(TypeError, match="host/eager"):
            jax.jit(
                lambda source: bridge.stage(
                    source,
                    event,
                    binding,
                    downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
                )
            )(state)


def test_one_coordinator_capacity_failure_rolls_all_three_owners_back() -> None:
    bridge, state = _bridge(agent_0_max_events=2, agent_1_max_events=1)
    first_event = bridge.prepare_event(state)
    first = bridge.stage(
        state,
        first_event,
        bridge.bind_base_actions(state, first_event),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(first.update_applied)
    second_event = bridge.prepare_event(first.state)
    rejected = bridge.stage(
        first.state,
        second_event,
        bridge.bind_base_actions(first.state, second_event),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(rejected.hccl_result.update_applied)
    assert bool(rejected.agent_0_result.diagnostics.transaction_applied)
    assert not bool(rejected.agent_1_result.diagnostics.transaction_applied)
    assert not bool(rejected.hccl_commit_applied)
    assert not bool(rejected.agent_0_update_applied)
    assert not bool(rejected.agent_1_update_applied)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, first.state)


def test_resources_and_in_memory_checkpoint_are_strict() -> None:
    bridge, state = _bridge()
    budget = bridge.resource_budget(state)
    measured = measure_hccl_external_coordinator_base_state_nbytes(state)
    assert budget.total_persistent_state_nbytes == measured
    assert budget.hccl_state_owners == 1
    assert budget.external_coordinator_state_owners == 2
    assert budget.max_world_proposal_calls_per_transaction == 8
    assert budget.coordinator_update_calls_per_transaction == 2
    assert budget.output_write_calls == 0
    assert budget.artifact_bytes_written == 0
    checkpoint = save_hccl_external_coordinator_base_checkpoint(bridge, state)
    restored_bridge, restored = load_hccl_external_coordinator_base_checkpoint(checkpoint)
    assert restored_bridge.to_config() == bridge.to_config()
    chex.assert_trees_all_equal(restored, state)
    tampered = dataclasses.replace(
        checkpoint,
        state=cast(Any, checkpoint.state).replace(
            agent_0_state=cast(Any, checkpoint.state.agent_0_state).replace(
                current_action=checkpoint.state.agent_0_state.current_action
                ^ jnp.asarray(1, dtype=jnp.int32)
            )
        ),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_hccl_external_coordinator_base_checkpoint(tampered)


@pytest.mark.parametrize(
    ("field", "alias"),
    (
        ("base_only_ablation", 1),
        ("memory_layer_authority", 0),
        ("hccl_state_owners", True),
    ),
)
def test_config_rejects_bool_integer_canonical_type_aliases(
    field: str,
    alias: object,
) -> None:
    bridge, _ = _bridge()
    payload = bridge.to_config()
    payload[field] = alias

    with pytest.raises(ValueError, match="unsupported"):
        HCCLExternalCoordinatorBaseBridge.from_config(payload)


def test_checkpoint_rejects_resealed_boolean_integer_alias() -> None:
    bridge, state = _bridge()
    checkpoint = save_hccl_external_coordinator_base_checkpoint(bridge, state)
    aliased = dataclasses.replace(
        checkpoint,
        output_writes_authorized=cast(Any, 0),
    )
    from alberta_framework.core import hccl_external_coordinator_base_bridge as module

    resealed = dataclasses.replace(
        aliased,
        checkpoint_sha256=module._checkpoint_digest(aliased),
    )
    with pytest.raises(ValueError, match="output_writes_authorized"):
        load_hccl_external_coordinator_base_checkpoint(resealed)


def test_checkpoint_rejects_resealed_resource_boolean_integer_alias() -> None:
    bridge, state = _bridge()
    checkpoint = save_hccl_external_coordinator_base_checkpoint(bridge, state)
    resource = dict(checkpoint.resource_budget)
    resource["memory_layer_authority"] = False
    aliased = dataclasses.replace(checkpoint, resource_budget=resource)
    from alberta_framework.core import hccl_external_coordinator_base_bridge as module

    resealed = dataclasses.replace(
        aliased,
        checkpoint_sha256=module._checkpoint_digest(aliased),
    )
    with pytest.raises(ValueError, match="resource budget"):
        load_hccl_external_coordinator_base_checkpoint(resealed)
