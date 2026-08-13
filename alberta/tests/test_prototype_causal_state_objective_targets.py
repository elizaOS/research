"""Contracts for learner-owned causal targets in the Prototype transaction."""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.core as alberta_core
import alberta_framework.core.prototype_causal_state_objective_targets as causal_adapter_module
from alberta_framework.core.causal_state_objective_targets import (
    CausalCumulantMode,
    CausalStateObjectiveTargetProducer,
    CausalStateObjectiveTargetProducerConfig,
)
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectivesConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_causal_state_objective_targets import (
    PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL,
    PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS,
    PrototypeCausalStateObjectiveTargets,
    PrototypeCausalStateObjectiveTargetsScanInputs,
    PrototypeCausalStateObjectiveTargetsState,
    load_prototype_causal_state_objective_targets_checkpoint,
    measure_prototype_causal_state_objective_targets_state_nbytes,
    run_prototype_causal_state_objective_targets_scan,
    save_prototype_causal_state_objective_targets_checkpoint,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.rtu_generate_and_test import (
    RTUGenerateAndTest,
    RTUGenerateAndTestConfig,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilderConfig,
    RecurrentTraceUnitStateBuilderConfig,
)

pytestmark = pytest.mark.integration

RAW_DIM = 2
HIDDEN_DIM = 1
FEATURE_DIM = RAW_DIM + HIDDEN_DIM
N_ACTIONS = 2


def _builder_config() -> OnlineGatedStateBuilderConfig:
    return OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        include_raw_observation=True,
        step_size=0.05,
        gradient_clip=5.0,
        initialization_scale=0.1,
    )


def _prototype(
    *,
    gradient_mixer: bool = False,
    rtu: bool = False,
    rtu_hidden_dim: int = HIDDEN_DIM,
    include_options: bool = True,
) -> PrototypeAgent:
    builder: OnlineGatedStateBuilderConfig | RecurrentTraceUnitStateBuilderConfig
    feature_dim = FEATURE_DIM
    if rtu:
        builder = RecurrentTraceUnitStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=rtu_hidden_dim,
            include_raw_observation=True,
            step_size=0.04,
            gradient_clip=8.0,
            r_min=0.2,
            r_max=0.95,
        )
        feature_dim = RAW_DIM + 2 * rtu_hidden_dim
    else:
        builder = _builder_config()
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=((SubtaskSpec(feature_index=0),) if include_options else ()),
                    observation_dim=feature_dim,
                    n_primitive_actions=N_ACTIONS,
                    base_hidden_sizes=(),
                    base_step_size=0.02,
                    option_step_size=0.02,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=builder,
            representation_gradient_mixer=(
                RepresentationGradientMixerConfig(
                    representation_dim=feature_dim,
                    mode="behavior_only",
                )
                if gradient_mixer
                else None
            ),
        )
    )


def _target_producer(
    *,
    cumulant_mode: CausalCumulantMode = "environment_reward",
    representation_dim: int = FEATURE_DIM,
    max_abs_reward_target: float = 100.0,
) -> CausalStateObjectiveTargetProducer:
    return CausalStateObjectiveTargetProducer(
        CausalStateObjectiveTargetProducerConfig(
            objectives_config=ComprehensiveStateObjectivesConfig(
                representation_dim=representation_dim,
                observation_target_dim=RAW_DIM,
                n_actions=N_ACTIONS,
                gvf_discounts=(0.0, 0.5, 0.9),
                initialization_scale=0.08,
                representation_gradient_clip=10.0,
                max_abs_reward_target=max_abs_reward_target,
                max_abs_control_target=100.0,
                max_abs_cumulant=100.0,
            ),
            transition_owner_digest=tuple(range(8)),
            cumulant_mode=cumulant_mode,
            cumulant_owner_digest=tuple(range(8, 16)),
        )
    )


def _adapter(
    *,
    cumulant_mode: CausalCumulantMode = "environment_reward",
    max_abs_reward_target: float = 100.0,
) -> PrototypeCausalStateObjectiveTargets:
    return PrototypeCausalStateObjectiveTargets(
        _prototype(),
        _target_producer(
            cumulant_mode=cumulant_mode,
            max_abs_reward_target=max_abs_reward_target,
        ),
    )


def _rtu_adapter(
    *,
    with_generate_and_test: bool = True,
    replacement_interval: int = 100,
    minimum_causal_evidence: int = 1,
    hidden_dim: int = HIDDEN_DIM,
    utility_decay: float = 0.0,
    include_options: bool = False,
) -> PrototypeCausalStateObjectiveTargets:
    prototype = _prototype(
        rtu=True,
        rtu_hidden_dim=hidden_dim,
        include_options=include_options,
    )
    target_producer = _target_producer(
        representation_dim=RAW_DIM + 2 * hidden_dim,
    )
    if not with_generate_and_test:
        return PrototypeCausalStateObjectiveTargets(prototype, target_producer)
    builder = cast(RecurrentTraceUnitStateBuilderConfig, prototype.config.state_builder)
    lifecycle = RTUGenerateAndTest(
        RTUGenerateAndTestConfig(
            builder=builder,
            utility_decay=utility_decay,
            replacement_interval=replacement_interval,
            replacement_quota=1,
            minimum_age=0,
            minimum_support=0,
            minimum_causal_evidence=minimum_causal_evidence,
        )
    )
    return PrototypeCausalStateObjectiveTargets(  # type: ignore[call-arg]
        prototype,
        target_producer,
        lifecycle,
    )


def _started(
    adapter: PrototypeCausalStateObjectiveTargets,
) -> PrototypeCausalStateObjectiveTargetsState:
    initial = adapter.init(
        jr.key(3),
        lifecycle_id=jnp.asarray([7, 9], dtype=jnp.uint32),
    )
    result = adapter.start(initial, jnp.asarray([0.25, -0.5], dtype=jnp.float32))
    assert bool(result.start_applied)
    return result.state


def _transition(
    state: PrototypeCausalStateObjectiveTargetsState,
    *,
    reward: float = 0.3,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    next_observation: jax.Array | None = None,
    next_decision_observation: jax.Array | None = None,
    decision_id: jax.Array | None = None,
) -> PrototypeTransition:
    prototype = state.prototype_state
    successor = (
        jnp.asarray([-0.1, 0.4], dtype=jnp.float32)
        if next_observation is None
        else next_observation
    )
    decision_successor = (
        successor if next_decision_observation is None else next_decision_observation
    )
    return PrototypeTransition(  # type: ignore[call-arg]
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=(prototype.current_decision_id if decision_id is None else decision_id),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(truncated, dtype=jnp.bool_),
        next_observation=successor,
        next_decision_observation=decision_successor,
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_bit_exact(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        assert left_array.tobytes() == right_array.tobytes()


def _at_exact_transaction(
    state: PrototypeCausalStateObjectiveTargetsState,
    *,
    transaction_words: tuple[int, int],
    observation_words: tuple[int, int],
    pending_builder_update_words: tuple[int, int],
) -> PrototypeCausalStateObjectiveTargetsState:
    """Move every exact owner clock together for a synthetic wrap contract."""

    transaction = jnp.asarray(transaction_words, dtype=jnp.uint32)
    observation = jnp.asarray(observation_words, dtype=jnp.uint32)
    pending_builder_update = jnp.asarray(
        pending_builder_update_words,
        dtype=jnp.uint32,
    )
    telemetry = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    prototype = cast(Any, state.prototype_state)
    base = cast(Any, prototype.oak_state.stomp_state.base_learner_state).replace(
        step_count=telemetry,
        step_words=transaction,
    )
    stomp = cast(Any, prototype.oak_state.stomp_state).replace(
        base_learner_state=base,
        step_count=telemetry,
        step_words=transaction,
    )
    oak = cast(Any, prototype.oak_state).replace(
        stomp_state=stomp,
        step_count=telemetry,
        step_words=transaction,
    )
    builder = cast(Any, prototype.state_builder_state).replace(
        step_count=telemetry,
        step_words=observation,
        update_count=telemetry,
        update_words=transaction,
    )
    prototype = prototype.replace(
        oak_state=oak,
        state_builder_state=builder,
        observation_event_count=telemetry,
        observation_event_words=observation,
        step_count=telemetry,
        step_words=transaction,
    )
    target = cast(Any, state.target_state)
    objectives = cast(Any, target.objectives_state).replace(
        pending_representation_revision_words=observation,
        pending_action_identity_words=observation,
        decision_words=observation,
        update_words=transaction,
        head_revision_words=jnp.broadcast_to(transaction[None, :], (8, 2)),
    )
    target = target.replace(
        objectives_state=objectives,
        pending_objective_action_identity_words=observation,
        decision_words=observation,
        transition_words=transaction,
    )
    return cast(
        PrototypeCausalStateObjectiveTargetsState,
        cast(Any, state).replace(
            prototype_state=prototype,
            target_state=target,
            pending_builder_step_words=observation,
            pending_builder_update_words=pending_builder_update,
            transaction_words=transaction,
        ),
    )


def _at_rtu_fail_stop(
    state: PrototypeCausalStateObjectiveTargetsState,
) -> PrototypeCausalStateObjectiveTargetsState:
    """Move every owner to the last accepted strict-RTU transaction."""

    maximum = 2**32 - 1
    shifted = _at_exact_transaction(
        state,
        transaction_words=(0, maximum),
        observation_words=(1, 0),
        pending_builder_update_words=(0, maximum),
    )
    prototype = cast(Any, shifted.prototype_state)
    builder = cast(Any, prototype.state_builder_state).replace(
        update_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        update_words=jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    prototype = prototype.replace(state_builder_state=builder)
    lifecycle = shifted.rtu_generate_and_test_state
    assert lifecycle is not None
    lifecycle = lifecycle.replace(
        observation_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
        observation_words=jnp.asarray((0, maximum), dtype=jnp.uint32),
        replacement_count=jnp.asarray(1, dtype=jnp.int32),
        replacement_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        replacement_event_count=jnp.asarray(1, dtype=jnp.int32),
        replacement_event_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        age=jnp.ones_like(lifecycle.age),
        causal_evidence_count=jnp.ones_like(lifecycle.causal_evidence_count),
    )
    return cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            shifted,
            prototype_state=prototype,
            rtu_generate_and_test_state=lifecycle,
        ),
    )


def test_config_is_strict_l0_public_and_rejects_conflicting_owners() -> None:
    adapter = _adapter()
    config = adapter.to_config()
    assert config["evidence_level"] == PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_EVIDENCE_LEVEL
    assert config["evidence_level"] == "L0"
    assert config["outcome_status"] == "not_assessed"
    assert config["scientific_promotion_allowed"] is False
    assert config["target_authority"] == "learner-owned-causal-real-transition"
    assert "caller-training-targets-excluded" in config["ownership"]
    assert PrototypeCausalStateObjectiveTargets.from_config(config).to_config() == config
    assert alberta_framework.PrototypeCausalStateObjectiveTargets is (
        PrototypeCausalStateObjectiveTargets
    )

    with pytest.raises(ValueError, match="representation_dim"):
        PrototypeCausalStateObjectiveTargets(
            _prototype(),
            _target_producer(representation_dim=FEATURE_DIM + 1),
        )
    with pytest.raises(ValueError, match="representation gradient mixing disabled"):
        PrototypeCausalStateObjectiveTargets(
            _prototype(gradient_mixer=True),
            _target_producer(),
        )


def test_optional_rtu_config_state_and_mismatch_are_strict() -> None:
    base = _adapter()
    assert base.rtu_generate_and_test is None
    base_config = base.to_config()
    assert "rtu_generate_and_test_config" not in base_config
    assert PrototypeCausalStateObjectiveTargets.from_config(base_config).to_config() == base_config

    with pytest.raises(ValueError, match="RTU builder requires"):
        _rtu_adapter(with_generate_and_test=False)

    integrated = _rtu_adapter(with_generate_and_test=True)
    lifecycle = integrated.rtu_generate_and_test
    assert lifecycle is not None
    config = integrated.to_config()
    assert config["rtu_generate_and_test_config"] == lifecycle.to_config()
    assert config["max_transitions"] == 2**32 - 1
    restored = PrototypeCausalStateObjectiveTargets.from_config(config)
    assert restored.to_config() == config
    assert restored.rtu_generate_and_test is not None
    assert PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS == 2**32 - 1
    assert alberta_core.PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS == (
        PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS
    )
    assert alberta_framework.PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS == (
        PROTOTYPE_CAUSAL_STATE_OBJECTIVE_TARGETS_RTU_MAX_TRANSITIONS
    )
    initial = integrated.init(jr.key(101))
    assert initial.rtu_generate_and_test_state is not None
    assert bool(integrated.state_valid(initial))

    prototype = _prototype(rtu=True, include_options=False)
    mismatched_builder = RecurrentTraceUnitStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM + 1,
        include_raw_observation=True,
    )
    mismatched_lifecycle = RTUGenerateAndTest(
        RTUGenerateAndTestConfig(
            builder=mismatched_builder,
            replacement_quota=1,
        )
    )
    with pytest.raises(ValueError, match="builder config"):
        PrototypeCausalStateObjectiveTargets(
            prototype,
            _target_producer(representation_dim=RAW_DIM + 2 * HIDDEN_DIM),
            mismatched_lifecycle,
        )


def test_start_binds_exact_prototype_decision_and_target_owner() -> None:
    adapter = _adapter()
    state = _started(adapter)
    target = state.target_state
    prototype = state.prototype_state
    assert bool(adapter.state_valid(state))
    assert bool(state.pending_valid)
    chex.assert_trees_all_equal(target.pending_observation, prototype.current_raw_observation)
    chex.assert_trees_all_equal(
        target.objectives_state.pending_representation,
        prototype.current_representation,
    )
    assert int(target.objectives_state.pending_action) == int(prototype.current_action)
    chex.assert_trees_all_equal(
        target.objectives_state.pending_representation_revision_words,
        prototype.observation_event_words,
    )
    chex.assert_trees_all_equal(
        target.pending_decision_identity_words,
        prototype.current_decision_id,
    )
    chex.assert_trees_all_equal(state.pending_prototype_decision_id, prototype.current_decision_id)


def test_rtu_lifecycle_no_replacement_commits_learner_owned_target_transaction() -> None:
    adapter = _rtu_adapter(
        with_generate_and_test=True,
        replacement_interval=100,
    )
    observation = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    state = adapter.start(
        adapter.init(jr.key(102), lifecycle_id=jnp.asarray([7, 9], dtype=jnp.uint32)),
        observation,
    ).state
    lifecycle = adapter.rtu_generate_and_test
    assert lifecycle is not None
    lifecycle_source = state.rtu_generate_and_test_state
    assert lifecycle_source is not None
    assert bool(adapter.state_valid(state))
    transition = _transition(
        state,
        reward=0.35,
        discount=0.9,
        next_observation=jnp.asarray([-0.4, 0.7], dtype=jnp.float32),
    )

    result = adapter.update_transition(state, transition)

    assert bool(result.update_applied)
    assert bool(result.prototype_transaction_applied)
    assert bool(result.target_transaction_applied)
    assert bool(result.builder_transaction_applied)
    assert bool(result.rtu_causal_deletion_evidence_attempted)
    assert bool(result.rtu_causal_deletion_evidence_available)
    assert bool(result.rtu_causal_deletion_evidence_valid)
    assert result.rtu_generate_and_test is not None
    assert result.rtu_advance_receipt is not None
    assert not bool(jnp.any(result.rtu_generate_and_test.diagnostics.selected_mask))
    assert float(result.target_update.targets.reward) == pytest.approx(0.35)
    assert float(result.target_update.targets.discount) == pytest.approx(0.9)
    lifecycle_state = result.state.rtu_generate_and_test_state
    assert lifecycle_state is not None
    np.testing.assert_array_equal(lifecycle_state.observation_words, [0, 1])
    np.testing.assert_array_equal(lifecycle_state.replacement_event_words, [0, 0])
    np.testing.assert_array_equal(
        jr.key_data(lifecycle_state.rng_key),
        jr.key_data(lifecycle_source.rng_key),
    )
    np.testing.assert_array_equal(result.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(result.state.target_state.transition_words, [0, 1])


def test_real_transition_derives_targets_and_commits_each_child_once() -> None:
    adapter = _adapter()
    state = _started(adapter)
    before_builder = cast(Any, state.prototype_state.state_builder_state)
    result = adapter.update_transition(state, _transition(state))
    assert bool(result.update_applied)
    assert bool(result.bootstrap_transition_applied)
    assert bool(result.prototype_transaction_applied)
    assert bool(result.target_transaction_applied)
    assert bool(result.builder_transaction_applied)
    assert bool(result.next_target_cache_valid)
    assert float(result.target_update.targets.reward) == pytest.approx(0.3)
    assert float(result.target_update.targets.discount) == pytest.approx(0.9)
    assert float(result.target_update.targets.gvf_targets[0]) == pytest.approx(0.3)
    chex.assert_trees_all_equal(
        result.target_update.targets.next_observation,
        jnp.asarray([-0.1, 0.4], dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        result.state.prototype_state.step_words,
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        result.state.target_state.transition_words,
        jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    after_builder = cast(Any, result.state.prototype_state.state_builder_state)
    chex.assert_trees_all_equal(
        after_builder.update_words,
        before_builder.update_words + jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    assert int(result.resource_work.prototype_update_evaluations) == 1
    assert int(result.resource_work.target_owner_update_evaluations) == 1
    assert int(result.resource_work.builder_commit_evaluations) == 1


def test_live_rtu_rank_uses_learner_owned_targets_and_scrubs_all_selected_axes(
    tmp_path: Path,
) -> None:
    hidden_dim = 2
    adapter = _rtu_adapter(
        replacement_interval=1,
        minimum_causal_evidence=1,
        hidden_dim=hidden_dim,
        utility_decay=0.0,
    )
    state = adapter.start(
        adapter.init(jr.key(104)),
        jnp.asarray([0.35, -0.2], dtype=jnp.float32),
    ).state
    representation = state.prototype_state.current_representation
    real_axes = jnp.asarray([RAW_DIM, RAW_DIM + 1], dtype=jnp.int32)
    real_values = representation[real_axes]
    assert bool(jnp.all(jnp.abs(real_values) > jnp.float32(1.0e-5)))

    # The factual value is +0.9 and the learner-owned terminated target is
    # -0.1.  Proxy contribution therefore ranks unit 1 lower, while deleting
    # unit 0 removes the entire error and makes unit 0 causally least useful.
    source_objectives = state.target_state.objectives_state
    value_weights = jnp.zeros_like(source_objectives.value_weights)
    value_weights = value_weights.at[real_axes[0]].set(jnp.float32(1.0) / real_values[0])
    value_weights = value_weights.at[real_axes[1]].set(jnp.float32(-0.1) / real_values[1])
    controlled_objectives = source_objectives.replace(
        observation_weights=jnp.zeros_like(source_objectives.observation_weights),
        observation_bias=jnp.zeros_like(source_objectives.observation_bias),
        latent_weights=jnp.zeros_like(source_objectives.latent_weights),
        latent_bias=jnp.zeros_like(source_objectives.latent_bias),
        reward_weights=jnp.zeros_like(source_objectives.reward_weights),
        reward_bias=jnp.zeros_like(source_objectives.reward_bias),
        termination_weights=jnp.zeros_like(source_objectives.termination_weights),
        termination_bias=jnp.zeros_like(source_objectives.termination_bias),
        gvf_weights=jnp.zeros_like(source_objectives.gvf_weights),
        value_weights=value_weights,
        value_bias=jnp.asarray(0.0, dtype=jnp.float32),
        advantage_weights=jnp.zeros_like(source_objectives.advantage_weights),
        advantage_bias=jnp.zeros_like(source_objectives.advantage_bias),
        inverse_current_weights=jnp.zeros_like(source_objectives.inverse_current_weights),
        inverse_next_weights=jnp.zeros_like(source_objectives.inverse_next_weights),
        inverse_bias=jnp.zeros_like(source_objectives.inverse_bias),
    )
    controlled_target = state.target_state.replace(
        objectives_state=controlled_objectives,
    )
    state = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(state, target_state=controlled_target),
    )
    assert bool(adapter.state_valid(state))

    result = adapter.update_transition(
        state,
        _transition(
            state,
            reward=-0.1,
            discount=0.0,
            terminated=True,
            next_observation=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        ),
    )

    assert bool(result.update_applied)
    assert float(result.target_update.targets.control_value_target) == pytest.approx(-0.1)
    assert float(result.derived_target_receipt.targets.control_value_target) == pytest.approx(-0.1)
    assert bool(result.rtu_causal_deletion_evidence_attempted)
    assert bool(result.rtu_causal_deletion_evidence_available)
    assert bool(result.rtu_causal_deletion_evidence_valid)
    assert result.rtu_generate_and_test is not None
    diagnostics = result.rtu_generate_and_test.diagnostics
    assert bool(diagnostics.causal_deletion_evidence_available)
    assert bool(diagnostics.causal_deletion_evidence_valid)
    assert bool(diagnostics.causal_evidence_required)
    assert float(diagnostics.effective_contribution[0]) > float(
        diagnostics.effective_contribution[1]
    )
    assert float(diagnostics.causal_deletion_loss_change[0]) < float(
        diagnostics.causal_deletion_loss_change[1]
    )
    expected_pre_update_change = (
        jnp.float32(adapter.target_producer.objectives.config.control_group_weight)
        * jnp.float32(0.25)
        * jnp.asarray([-1.0, 0.21], dtype=jnp.float32)
    )
    np.testing.assert_allclose(
        diagnostics.causal_deletion_loss_change,
        expected_pre_update_change,
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(diagnostics.selected_mask, [True, False])

    # Re-evaluating after SGD with the same learner-owned target must differ:
    # the authoritative deletion score above is frozen at the source heads.
    objectives = adapter.target_producer.objectives
    post_cache = objectives.cache_action(
        result.target_update.state.objectives_state,
        representation,
        state.prototype_state.current_action,
        state.prototype_state.observation_event_words,
    )
    assert bool(post_cache.cache_applied)
    fixed_targets = result.target_update.targets

    def evaluate_post_update_head(current_representation: jax.Array) -> jax.Array:
        counterfactual_state = post_cache.state.replace(
            pending_representation=current_representation
        )
        counterfactual_receipt = post_cache.receipt.replace(
            representation=current_representation
        )
        update = objectives.update(
            counterfactual_state,
            counterfactual_receipt,
            fixed_targets.next_latent,
            result.accepted_target_transition.next_representation_revision_words,
            fixed_targets.next_observation,
            fixed_targets.reward,
            fixed_targets.terminated,
            fixed_targets.cumulant,
            fixed_targets.effective_continuation,
            fixed_targets.control_value_target,
            fixed_targets.selected_action_advantage_target,
        )
        assert bool(update.update_applied)
        return update.balanced_loss

    post_factual_loss = evaluate_post_update_head(representation)
    post_update_change = jnp.stack(
        tuple(
            evaluate_post_update_head(
                representation.at[RAW_DIM + unit_index]
                .set(jnp.float32(0.0))
                .at[RAW_DIM + hidden_dim + unit_index]
                .set(jnp.float32(0.0))
            )
            - post_factual_loss
            for unit_index in range(hidden_dim)
        )
    )
    assert float(
        jnp.max(jnp.abs(post_update_change - diagnostics.causal_deletion_loss_change))
    ) > 1.0e-5

    selected_axes = jnp.asarray(
        [RAW_DIM, RAW_DIM + hidden_dim],
        dtype=jnp.int32,
    )
    final_objectives = result.state.target_state.objectives_state
    for weights in (
        final_objectives.observation_weights,
        final_objectives.latent_weights,
        final_objectives.reward_weights,
        final_objectives.termination_weights,
        final_objectives.gvf_weights,
        final_objectives.value_weights,
        final_objectives.advantage_weights,
        final_objectives.inverse_current_weights,
        final_objectives.inverse_next_weights,
    ):
        selected = weights[..., selected_axes]
        np.testing.assert_array_equal(selected, jnp.zeros_like(selected))
        np.testing.assert_array_equal(
            np.asarray(selected).view(np.uint32),
            np.zeros(np.asarray(selected).shape, dtype=np.uint32),
        )
    chex.assert_trees_all_equal(
        final_objectives.pending_representation,
        result.state.prototype_state.current_representation,
    )
    assert bool(result.next_target_cache_valid)
    assert int(result.resource_work.prototype_update_evaluations) == 1
    assert int(result.resource_work.target_owner_update_evaluations) == 1
    assert int(result.resource_work.builder_proposal_evaluations) == 2
    assert int(result.resource_work.builder_commit_evaluations) == 4
    assert int(result.resource_work.causal_deletion_units_scored) == hidden_dim
    assert int(result.resource_work.causal_deletion_frozen_head_evaluations) == 8 * hidden_dim
    assert int(result.resource_work.rtu_generate_and_test_proposal_evaluations) == 1
    assert int(result.resource_work.rtu_generate_and_test_commit_evaluations) == 2
    assert int(result.resource_work.next_target_cache_evaluations) == 1

    corrupt_head = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            target_state=result.state.target_state.replace(
                objectives_state=final_objectives.replace(
                    value_weights=final_objectives.value_weights.at[RAW_DIM].set(
                        jnp.float32(1.0)
                    )
                )
            ),
        ),
    )
    corrupt_pending = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            target_state=result.state.target_state.replace(
                objectives_state=final_objectives.replace(
                    pending_representation=(
                        final_objectives.pending_representation.at[RAW_DIM].set(
                            jnp.float32(1.0)
                        )
                    )
                )
            ),
        ),
    )
    corrupt_signed_zero = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            target_state=result.state.target_state.replace(
                objectives_state=final_objectives.replace(
                    value_weights=final_objectives.value_weights.at[RAW_DIM].set(
                        jnp.float32(-0.0)
                    )
                )
            ),
        ),
    )
    final_builder = cast(Any, result.state.prototype_state.state_builder_state)
    corrupt_sensitivities = final_builder.sensitivities._replace(
        b_real=final_builder.sensitivities.b_real.at[0, 0, 0].set(
            jnp.float32(1.0)
        )
    )
    corrupt_builder = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            prototype_state=result.state.prototype_state.replace(
                state_builder_state=final_builder.replace(
                    sensitivities=corrupt_sensitivities
                )
            ),
        ),
    )
    final_stomp = result.state.prototype_state.oak_state.stomp_state
    final_base = final_stomp.base_learner_state
    corrupt_weights = list(final_base.head_params.weights)
    corrupt_weights[0] = corrupt_weights[0].at[0, RAW_DIM].set(jnp.float32(1.0))
    corrupt_stomp_weight = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            prototype_state=result.state.prototype_state.replace(
                oak_state=result.state.prototype_state.oak_state.replace(
                    stomp_state=final_stomp.replace(
                        base_learner_state=final_base.replace(
                            head_params=final_base.head_params.replace(
                                weights=tuple(corrupt_weights)
                            )
                        )
                    )
                )
            ),
        ),
    )
    corrupt_traces = list(final_base.head_traces)
    weight_trace, bias_trace = corrupt_traces[0]
    corrupt_traces[0] = (
        weight_trace.at[0, RAW_DIM].set(jnp.float32(1.0)),
        bias_trace,
    )
    corrupt_stomp_trace = cast(
        PrototypeCausalStateObjectiveTargetsState,
        dataclasses.replace(
            result.state,
            prototype_state=result.state.prototype_state.replace(
                oak_state=result.state.prototype_state.oak_state.replace(
                    stomp_state=final_stomp.replace(
                        base_learner_state=final_base.replace(
                            head_traces=tuple(corrupt_traces)
                        )
                    )
                )
            ),
        ),
    )
    assert not bool(adapter.state_valid(corrupt_head))
    assert not bool(adapter.state_valid(corrupt_pending))
    assert not bool(adapter.state_valid(corrupt_signed_zero))
    assert not bool(adapter.state_valid(corrupt_builder))
    assert not bool(adapter.state_valid(corrupt_stomp_weight))
    assert not bool(adapter.state_valid(corrupt_stomp_trace))
    with pytest.raises(ValueError, match="invalid Prototype causal target state"):
        save_prototype_causal_state_objective_targets_checkpoint(
            adapter,
            corrupt_head,
            tmp_path / "corrupt-prototype-causal-targets-rtu",
        )
    with pytest.raises(ValueError, match="invalid Prototype causal target state"):
        save_prototype_causal_state_objective_targets_checkpoint(
            adapter,
            corrupt_builder,
            tmp_path / "corrupt-builder-prototype-causal-targets-rtu",
        )


def test_truncation_targets_final_observation_before_reset() -> None:
    adapter = _adapter()
    state = _started(adapter)
    final_observation = jnp.asarray([0.6, -0.2], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.7, 0.8], dtype=jnp.float32)
    result = adapter.update_transition(
        state,
        _transition(
            state,
            truncated=True,
            discount=0.8,
            next_observation=final_observation,
            next_decision_observation=reset_observation,
        ),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.target_update.targets.next_observation, final_observation)
    chex.assert_trees_all_equal(
        result.state.prototype_state.current_raw_observation,
        reset_observation,
    )
    assert float(result.target_update.targets.effective_continuation) == pytest.approx(0.8)


def test_stale_target_refusal_and_optional_cumulant_tamper_roll_back_atomically() -> None:
    adapter = _adapter(max_abs_reward_target=1.0)
    state = _started(adapter)
    refused = adapter.update_transition(state, _transition(state, reward=2.0))
    assert bool(refused.prototype_transaction_applied)
    assert not bool(refused.target_transaction_applied)
    assert not bool(refused.update_applied)
    chex.assert_trees_all_equal(_materialize_keys(refused.state), _materialize_keys(state))

    stale_id = state.prototype_state.current_decision_id.at[3].add(jnp.uint32(1))
    stale = adapter.update_transition(state, _transition(state, decision_id=stale_id))
    assert not bool(stale.transition_identity_matches)
    chex.assert_trees_all_equal(_materialize_keys(stale.state), _materialize_keys(state))

    optional = _adapter(cumulant_mode="bound_optional")
    optional_state = _started(optional)
    receipt = optional.bind_optional_cumulant(
        optional_state,
        value=jnp.asarray(0.7, dtype=jnp.float32),
        source_revision_words=jnp.asarray([0, 2], dtype=jnp.uint32),
        provenance_words=jnp.asarray([3, 5, 7, 11], dtype=jnp.uint32),
    )
    tampered = dataclasses.replace(  # type: ignore[type-var]
        receipt,
        value=jnp.asarray(0.9, dtype=jnp.float32),
    )
    rejected = optional.update_transition(
        optional_state,
        _transition(optional_state),
        tampered,
    )
    assert not bool(rejected.target_update.cumulant_valid)
    chex.assert_trees_all_equal(
        _materialize_keys(rejected.state),
        _materialize_keys(optional_state),
    )


def test_invalid_internal_rtu_causal_scoring_rolls_back_the_outer_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(replacement_interval=1)
    state = adapter.start(
        adapter.init(jr.key(105)),
        jnp.asarray([0.3, -0.15], dtype=jnp.float32),
    ).state
    assert float(state.prototype_state.current_representation[RAW_DIM]) != 0.0
    objectives = adapter.target_producer.objectives
    original_update = objectives._update_jit

    def reject_only_deleted_representation(*args: Any) -> Any:
        objective_result = original_update(*args)
        receipt = args[1]
        counterfactual_valid = receipt.representation[RAW_DIM] != jnp.float32(0.0)
        return objective_result.replace(
            update_applied=objective_result.update_applied & counterfactual_valid,
        )

    monkeypatch.setattr(
        objectives,
        "_update_jit",
        reject_only_deleted_representation,
    )
    with jax.disable_jit():
        result = adapter.update_transition(
            state,
            _transition(
                state,
                reward=0.2,
                discount=0.9,
                next_observation=jnp.asarray([0.45, -0.25], dtype=jnp.float32),
            ),
        )

    assert not bool(result.prototype_transaction_applied)
    assert bool(result.target_transaction_applied)
    assert bool(result.rtu_causal_deletion_evidence_attempted)
    assert not bool(result.rtu_causal_deletion_evidence_available)
    assert not bool(result.rtu_causal_deletion_evidence_valid)
    assert result.rtu_generate_and_test is not None
    assert not bool(result.rtu_generate_and_test.diagnostics.applied)
    assert not bool(result.builder_transaction_applied)
    assert not bool(result.derived_target_receipt_committed)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(state),
    )


def test_late_successor_cache_rejection_rolls_back_a_valid_rtu_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(replacement_interval=1)
    state = adapter.start(
        adapter.init(jr.key(106)),
        jnp.asarray([0.3, -0.15], dtype=jnp.float32),
    ).state
    target_producer = adapter.target_producer
    original_cache = target_producer.cache_decision
    genuine_caches: list[Any] = []

    def reject_successor_cache(*args: Any, **kwargs: Any) -> Any:
        cache = original_cache(*args, **kwargs)
        genuine_caches.append(cache)
        return cache.replace(cache_applied=jnp.asarray(False, dtype=jnp.bool_))

    monkeypatch.setattr(target_producer, "cache_decision", reject_successor_cache)
    with jax.disable_jit():
        result = adapter.update_transition(
            state,
            _transition(
                state,
                reward=0.2,
                discount=0.9,
                next_observation=jnp.asarray([0.45, -0.25], dtype=jnp.float32),
            ),
        )

    assert len(genuine_caches) == 1
    genuine_cache = genuine_caches[0]
    for predicate in (
        genuine_cache.state_valid,
        genuine_cache.source_valid,
        genuine_cache.lifetime_capacity_available,
        genuine_cache.objective_cache_applied,
        genuine_cache.candidate_state_valid,
        genuine_cache.cache_applied,
        result.source_state_valid,
        result.transition_identity_matches,
        result.accepted_transition_matches_prototype,
        result.derived_target_receipt_valid,
        result.bootstrap_event_capacity_available,
        result.bootstrap_transition_applied,
        result.prototype_transaction_applied,
        result.target_transaction_applied,
        result.builder_sources_match,
        result.builder_destination_matches,
        result.builder_transaction_applied,
        result.rtu_observation_proposal_valid,
        result.rtu_lifecycle_source_matches,
        result.rtu_causal_deletion_evidence_available,
        result.rtu_causal_deletion_evidence_valid,
        result.rtu_replacement_cache_safe,
        result.next_target_cache_required,
        result.lifetime_capacity_available,
        result.candidate_state_valid,
    ):
        assert bool(predicate)
    assert result.rtu_generate_and_test is not None
    inner = result.rtu_generate_and_test
    diagnostics = inner.diagnostics
    assert bool(diagnostics.applied)
    assert bool(jnp.any(diagnostics.selected_mask))
    assert int(diagnostics.selected_count) == 1
    assert not np.array_equal(
        np.asarray(diagnostics.pre_rng_key_data),
        np.asarray(diagnostics.post_rng_key_data),
    )
    np.testing.assert_array_equal(
        diagnostics.post_rng_key_data,
        jr.key_data(inner.state.rng_key),
    )
    assert not bool(result.next_target_cache.cache_applied)
    assert not bool(result.next_target_cache_valid)
    assert not bool(result.rtu_observation_transaction_applied)
    assert not bool(result.derived_target_receipt_committed)
    assert not bool(result.update_applied)
    np.testing.assert_array_equal(
        result.post_transaction_words,
        result.pre_transaction_words,
    )
    _assert_tree_bit_exact(result.state, state)
    source_lifecycle = state.rtu_generate_and_test_state
    final_lifecycle = result.state.rtu_generate_and_test_state
    assert source_lifecycle is not None
    assert final_lifecycle is not None
    _assert_tree_bit_exact(final_lifecycle, source_lifecycle)
    np.testing.assert_array_equal(
        diagnostics.pre_rng_key_data,
        jr.key_data(source_lifecycle.rng_key),
    )
    np.testing.assert_array_equal(
        jr.key_data(final_lifecycle.rng_key),
        jr.key_data(source_lifecycle.rng_key),
    )


def test_immature_internal_rtu_evidence_defers_only_replacement_until_floor() -> None:
    adapter = _rtu_adapter(
        replacement_interval=1,
        minimum_causal_evidence=2,
    )
    state = adapter.start(
        adapter.init(jr.key(106)),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    ).state
    lifecycle_source = state.rtu_generate_and_test_state
    assert lifecycle_source is not None

    first = adapter.update_transition(
        state,
        _transition(
            state,
            reward=0.2,
            discount=0.9,
            next_observation=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        ),
    )

    assert bool(first.update_applied)
    assert bool(first.target_transaction_applied)
    assert bool(first.builder_transaction_applied)
    assert bool(first.rtu_causal_deletion_evidence_attempted)
    assert bool(first.rtu_causal_deletion_evidence_available)
    assert bool(first.rtu_causal_deletion_evidence_valid)
    assert first.rtu_generate_and_test is not None
    first_diagnostics = first.rtu_generate_and_test.diagnostics
    assert bool(first_diagnostics.causal_deletion_evidence_available)
    assert bool(first_diagnostics.causal_deletion_evidence_valid)
    assert not bool(jnp.any(first_diagnostics.selected_mask))
    first_lifecycle = first.state.rtu_generate_and_test_state
    assert first_lifecycle is not None
    np.testing.assert_array_equal(first_lifecycle.causal_evidence_count, [1])
    np.testing.assert_array_equal(first_lifecycle.observation_words, [0, 1])
    np.testing.assert_array_equal(first_lifecycle.replacement_event_words, [0, 0])
    np.testing.assert_array_equal(
        jr.key_data(first_lifecycle.rng_key),
        jr.key_data(lifecycle_source.rng_key),
    )
    np.testing.assert_array_equal(first.state.transaction_words, [0, 1])
    np.testing.assert_array_equal(first.state.target_state.transition_words, [0, 1])
    np.testing.assert_array_equal(
        cast(Any, first.state.prototype_state.state_builder_state).update_words,
        [0, 1],
    )

    second = adapter.update_transition(
        first.state,
        _transition(
            first.state,
            reward=-0.1,
            discount=0.8,
            next_observation=jnp.asarray([-0.2, 0.5], dtype=jnp.float32),
        ),
    )

    assert bool(second.update_applied)
    assert bool(second.rtu_causal_deletion_evidence_attempted)
    assert bool(second.rtu_causal_deletion_evidence_available)
    assert bool(second.rtu_causal_deletion_evidence_valid)
    assert second.rtu_generate_and_test is not None
    assert bool(second.rtu_generate_and_test.diagnostics.selected_mask[0])
    second_lifecycle = second.state.rtu_generate_and_test_state
    assert second_lifecycle is not None
    np.testing.assert_array_equal(second_lifecycle.causal_evidence_count, [0])
    np.testing.assert_array_equal(second_lifecycle.observation_words, [0, 2])
    np.testing.assert_array_equal(second_lifecycle.replacement_event_words, [0, 1])
    assert not np.array_equal(
        np.asarray(jr.key_data(second_lifecycle.rng_key)),
        np.asarray(jr.key_data(first_lifecycle.rng_key)),
    )
    np.testing.assert_array_equal(second.state.transaction_words, [0, 2])
    np.testing.assert_array_equal(second.state.target_state.transition_words, [0, 2])
    np.testing.assert_array_equal(
        cast(Any, second.state.prototype_state.state_builder_state).update_words,
        [0, 3],
    )


def test_resource_checkpoint_and_exact_uint64_carry_are_exact(tmp_path: Path) -> None:
    adapter = _adapter()
    state = _started(adapter)
    result = adapter.update_transition(state, _transition(state))
    assert bool(result.update_applied)
    budget = adapter.resource_budget(result.state)
    assert budget.total_state_nbytes == (
        measure_prototype_causal_state_objective_targets_state_nbytes(result.state)
    )
    assert budget.max_prototype_updates_per_transition == 1
    assert budget.max_target_owner_updates_per_transition == 1
    assert budget.max_builder_commits_per_transition == 1
    checkpoint = tmp_path / "prototype-causal-targets"
    save_prototype_causal_state_objective_targets_checkpoint(adapter, result.state, checkpoint)
    restored_adapter, restored_state = load_prototype_causal_state_objective_targets_checkpoint(
        checkpoint
    )
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(
        _materialize_keys(restored_state),
        _materialize_keys(result.state),
    )

    near_wrap = _at_exact_transaction(
        state,
        transaction_words=(0, 2**32 - 1),
        observation_words=(1, 0),
        pending_builder_update_words=(0, 2**32 - 2),
    )
    assert bool(adapter.state_valid(near_wrap))
    carried = adapter.update_transition(near_wrap, _transition(near_wrap))
    assert bool(carried.update_applied)
    chex.assert_trees_all_equal(
        carried.state.transaction_words,
        jnp.asarray([1, 0], dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(
        carried.state.target_state.transition_words,
        carried.state.transaction_words,
    )
    chex.assert_trees_all_equal(
        cast(Any, carried.state.prototype_state.state_builder_state).update_words,
        carried.state.transaction_words,
    )


def test_scan_is_eager_jit_deterministic_without_caller_training_targets() -> None:
    adapter = _adapter()
    initial = _started(adapter)
    inputs = PrototypeCausalStateObjectiveTargetsScanInputs(  # type: ignore[call-arg]
        next_observations=jnp.asarray([[-0.1, 0.4], [0.2, 0.1]], dtype=jnp.float32),
        next_decision_observations=jnp.asarray(
            [[-0.1, 0.4], [0.2, 0.1]],
            dtype=jnp.float32,
        ),
        rewards=jnp.asarray([0.3, -0.1], dtype=jnp.float32),
        discounts=jnp.asarray([0.9, 0.0], dtype=jnp.float32),
        terminated=jnp.asarray([False, True], dtype=jnp.bool_),
        truncated=jnp.asarray([False, False], dtype=jnp.bool_),
        optional_cumulants=jnp.zeros((2,), dtype=jnp.float32),
        optional_cumulant_available=jnp.zeros((2,), dtype=jnp.bool_),
        cumulant_source_revision_words=jnp.zeros((2, 2), dtype=jnp.uint32),
        cumulant_provenance_words=jnp.zeros((2, 4), dtype=jnp.uint32),
    )
    eager = run_prototype_causal_state_objective_targets_scan(adapter, initial, inputs)
    compiled = jax.jit(
        lambda source, arrays: run_prototype_causal_state_objective_targets_scan(
            adapter,
            source,
            arrays,
        )
    )(initial, inputs)
    chex.assert_trees_all_close(
        _materialize_keys(eager),
        _materialize_keys(compiled),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool(jnp.all(eager.update_applied))
    chex.assert_trees_all_equal(
        eager.transaction_words,
        jnp.asarray([[0, 1], [0, 2]], dtype=jnp.uint32),
    )


def test_rtu_compiled_scan_resource_and_checkpoint_are_exact(tmp_path: Path) -> None:
    adapter = _rtu_adapter(
        replacement_interval=1,
        minimum_causal_evidence=1,
    )
    initial = _started(adapter)
    inputs = PrototypeCausalStateObjectiveTargetsScanInputs(  # type: ignore[call-arg]
        next_observations=jnp.asarray(
            [[-0.1, 0.4], [0.2, 0.1]],
            dtype=jnp.float32,
        ),
        next_decision_observations=jnp.asarray(
            [[-0.1, 0.4], [0.2, 0.1]],
            dtype=jnp.float32,
        ),
        rewards=jnp.asarray([0.3, -0.1], dtype=jnp.float32),
        discounts=jnp.asarray([0.9, 0.8], dtype=jnp.float32),
        terminated=jnp.asarray([False, False], dtype=jnp.bool_),
        truncated=jnp.asarray([False, False], dtype=jnp.bool_),
        optional_cumulants=jnp.zeros((2,), dtype=jnp.float32),
        optional_cumulant_available=jnp.zeros((2,), dtype=jnp.bool_),
        cumulant_source_revision_words=jnp.zeros((2, 2), dtype=jnp.uint32),
        cumulant_provenance_words=jnp.zeros((2, 4), dtype=jnp.uint32),
    )
    with jax.disable_jit():
        eager = run_prototype_causal_state_objective_targets_scan(
            adapter,
            initial,
            inputs,
        )
    compiled = jax.jit(
        lambda source, arrays: run_prototype_causal_state_objective_targets_scan(
            adapter,
            source,
            arrays,
        )
    )(initial, inputs)

    chex.assert_trees_all_close(
        _materialize_keys(eager),
        _materialize_keys(compiled),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool(jnp.all(compiled.update_applied))
    lifecycle_state = compiled.state.rtu_generate_and_test_state
    assert lifecycle_state is not None
    np.testing.assert_array_equal(lifecycle_state.observation_words, [0, 2])
    np.testing.assert_array_equal(lifecycle_state.replacement_event_words, [0, 2])
    np.testing.assert_array_equal(compiled.state.transaction_words, [0, 2])
    np.testing.assert_array_equal(compiled.state.target_state.transition_words, [0, 2])
    np.testing.assert_array_equal(
        cast(Any, compiled.state.prototype_state.state_builder_state).update_words,
        [0, 4],
    )

    budget = adapter.resource_budget(compiled.state)
    assert budget.total_state_nbytes == (
        measure_prototype_causal_state_objective_targets_state_nbytes(compiled.state)
    )
    assert budget.rtu_generate_and_test_state_nbytes > 0
    assert budget.max_builder_commits_per_transition == 4
    assert budget.max_rtu_generate_and_test_proposals_per_transition == 1
    assert budget.max_rtu_generate_and_test_commits_per_transition == 2
    assert budget.max_causal_deletion_units_scored_per_transition == HIDDEN_DIM
    assert budget.max_causal_deletion_frozen_head_evaluations_per_transition == 8 * HIDDEN_DIM
    assert budget.max_accepted_transitions == 2**32 - 1

    checkpoint = tmp_path / "prototype-causal-targets-rtu"
    save_prototype_causal_state_objective_targets_checkpoint(
        adapter,
        compiled.state,
        checkpoint,
    )
    metadata = load_checkpoint_metadata(checkpoint)
    empty_leaf_indices = metadata["zero_sized_array_leaf_indices"]
    assert type(empty_leaf_indices) is list
    assert empty_leaf_indices
    assert all(type(index) is int for index in empty_leaf_indices)
    restored_adapter, restored_state = load_prototype_causal_state_objective_targets_checkpoint(
        checkpoint
    )
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(
        _materialize_keys(restored_state),
        _materialize_keys(compiled.state),
    )
    assert bool(restored_adapter.state_valid(restored_state))

    original_transition = _transition(
        compiled.state,
        reward=0.25,
        discount=0.85,
        next_observation=jnp.asarray([0.33, -0.27], dtype=jnp.float32),
    )
    restored_transition = _transition(
        restored_state,
        reward=0.25,
        discount=0.85,
        next_observation=jnp.asarray([0.33, -0.27], dtype=jnp.float32),
    )
    with jax.disable_jit():
        original_next = adapter.update_transition(
            compiled.state,
            original_transition,
        )
        restored_next = restored_adapter.update_transition(
            restored_state,
            restored_transition,
        )
    assert bool(original_next.update_applied)
    assert bool(restored_next.update_applied)
    _assert_tree_bit_exact(original_next, restored_next)
    np.testing.assert_array_equal(restored_next.state.transaction_words, [0, 3])
    np.testing.assert_array_equal(
        restored_next.state.target_state.transition_words,
        [0, 3],
    )
    restored_lifecycle = restored_next.state.rtu_generate_and_test_state
    assert restored_lifecycle is not None
    np.testing.assert_array_equal(restored_lifecycle.observation_words, [0, 3])
    np.testing.assert_array_equal(restored_lifecycle.replacement_event_words, [0, 3])
    np.testing.assert_array_equal(
        cast(Any, restored_next.state.prototype_state.state_builder_state).update_words,
        [0, 6],
    )


def test_rtu_checkpoint_rejects_noncanonical_metadata_and_storage_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _rtu_adapter(replacement_interval=1)
    state = _started(adapter)
    checkpoint = tmp_path / "prototype-causal-targets-rtu-strict-sentinels"
    save_prototype_causal_state_objective_targets_checkpoint(
        adapter,
        state,
        checkpoint,
    )
    metadata = load_checkpoint_metadata(checkpoint)
    indices = metadata["zero_sized_array_leaf_indices"]
    assert type(indices) is list
    assert indices

    noncanonical_manifest = deepcopy(metadata)
    noncanonical_manifest["zero_sized_array_leaf_indices"][0] = float(indices[0])
    with monkeypatch.context() as scoped:
        scoped.setattr(
            causal_adapter_module,
            "load_checkpoint_metadata",
            lambda _path: noncanonical_manifest,
        )
        with pytest.raises(ValueError, match="empty-array storage manifest differs"):
            load_prototype_causal_state_objective_targets_checkpoint(checkpoint)

    noncanonical_resource = deepcopy(metadata)
    total_nbytes = noncanonical_resource["resource_budget"]["total_state_nbytes"]
    noncanonical_resource["resource_budget"]["total_state_nbytes"] = float(total_nbytes)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            causal_adapter_module,
            "load_checkpoint_metadata",
            lambda _path: noncanonical_resource,
        )
        with pytest.raises(ValueError, match="checkpoint resource budget differs"):
            load_prototype_causal_state_objective_targets_checkpoint(checkpoint)

    storage = causal_adapter_module._checkpoint_storage_state(state)  # noqa: SLF001
    storage_leaves, storage_structure = jax.tree.flatten(storage)
    sentinel_index = indices[0]
    storage_leaves[sentinel_index] = jnp.ones_like(storage_leaves[sentinel_index])
    noncanonical_storage = jax.tree.unflatten(storage_structure, storage_leaves)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            causal_adapter_module,
            "load_checkpoint",
            lambda _template, _path: (noncanonical_storage, metadata),
        )
        with pytest.raises(ValueError, match="empty-array storage sentinel differs"):
            load_prototype_causal_state_objective_targets_checkpoint(checkpoint)


def test_rtu_global_fail_stop_rolls_back_the_complete_learner_owned_transaction() -> None:
    adapter = _rtu_adapter(
        replacement_interval=100,
        minimum_causal_evidence=1,
    )
    state = _at_rtu_fail_stop(_started(adapter))
    assert bool(adapter.state_valid(state))

    result = adapter.update_transition(
        state,
        _transition(
            state,
            reward=0.2,
            discount=0.9,
            next_observation=jnp.asarray([0.45, -0.25], dtype=jnp.float32),
        ),
    )

    assert bool(result.source_state_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.builder_transaction_applied)
    assert not bool(result.derived_target_receipt_committed)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(
        _materialize_keys(result.state),
        _materialize_keys(state),
    )
