# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Prototype integration contracts for audit-ranked feature curation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.core as alberta_core
from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.interaction_features import (
    CURATION_ACTIVE_INELIGIBLE_RANK,
    CURATION_CANDIDATE_INELIGIBLE_RANK,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
    PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeFeatureOaKHordeUtilityCurationState,
    PrototypeFeatureOaKHordeUtilityState,
    PrototypeFeatureRepresentationState,
    PrototypeFeatureUtilityCurationIntegrationDiagnostics,
    PrototypeFeatureUtilityIntegrationDiagnostics,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleConfig,
)
from alberta_framework.core.prototype_feature_utility import (
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityState,
)
from alberta_framework.core.prototype_feature_utility_curation import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY,
    PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE,
    PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureUtilityCurationConfig,
    PrototypeFeatureUtilityCurationDiagnostics,
)
from alberta_framework.core.state_builder import IdentityStateBuilderConfig
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    HordeSpec,
    create_horde_spec,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASE_DIM = 4
ACTIVE_SLOTS = 2
CANDIDATE_SLOTS = 6
TOTAL_DIM = BASE_DIM + ACTIVE_SLOTS
N_ACTIONS = 2
N_OPTIONS = 1
N_DEMONS = 2


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest):
    if request.node.name == "test_v15_ranked_lane_is_eager_jit_exact":
        yield
    else:
        with jax.disable_jit():
            yield


def _horde_spec() -> HordeSpec:
    return create_horde_spec(
        (
            GVFSpec(
                name="instant",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
            GVFSpec(
                name="temporal",
                demon_type=DemonType.PREDICTION,
                gamma=0.5,
                lamda=0.25,
                cumulant_index=1,
            ),
        )
    )


def _feature_config(
    *,
    replacement_interval: int = 0,
    max_observations: int = 8,
) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=CANDIDATE_SLOTS,
        n_tasks=1 + N_DEMONS,
        n_options=N_OPTIONS,
        n_primitive_actions=N_ACTIONS,
        option_subtask_feature_indices=(0,),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=max_observations,
        managed_horde_demons=N_DEMONS,
    )


def _utility_config(*, max_observations: int = 8) -> PrototypeFeatureUtilityConfig:
    return PrototypeFeatureUtilityConfig(
        base_feature_dim=BASE_DIM,
        active_pair_slots=ACTIVE_SLOTS,
        candidate_pair_slots=CANDIDATE_SLOTS,
        managed_horde_demons=N_DEMONS,
        utility_decay=0.999,
        shadow_step_size=0.2,
        second_moment_decay=0.5,
        scale_epsilon=1.0e-6,
        max_observations=max_observations,
    )


def _oak_config() -> OaKConfig:
    return OaKConfig(
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
        )
    )


def _agent_config(
    *,
    curation: bool,
    utility: bool = True,
    replacement_interval: int = 0,
    max_observations: int = 8,
    minimum_task_evidence: int = 1,
) -> PrototypeAgentConfig:
    kwargs: dict[str, Any] = {
        "oak": _oak_config(),
        "state_builder": IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        "horde_spec": _horde_spec(),
        "horde_hidden_sizes": (),
        "horde_step_size": 0.1,
        "prototype_feature_lifecycle": _feature_config(
            replacement_interval=replacement_interval,
            max_observations=max_observations,
        ),
    }
    if utility:
        kwargs["prototype_feature_utility"] = _utility_config(
            max_observations=max_observations
        )
    if curation:
        kwargs["prototype_feature_utility_curation"] = (
            PrototypeFeatureUtilityCurationConfig(
                minimum_task_evidence=minimum_task_evidence
            )
        )
    return PrototypeAgentConfig(**kwargs)


def _agent(
    *,
    curation: bool,
    replacement_interval: int = 0,
    max_observations: int = 8,
    minimum_task_evidence: int = 1,
) -> PrototypeAgent:
    return PrototypeAgent(
        _agent_config(
            curation=curation,
            replacement_interval=replacement_interval,
            max_observations=max_observations,
            minimum_task_evidence=minimum_task_evidence,
        )
    )


def _feature_state(state: PrototypeAgentState) -> Any:
    wrapper = state.state_builder_state
    assert type(wrapper) is PrototypeFeatureRepresentationState
    return wrapper.feature_lifecycle_state


def _consumer_bundle(agent: PrototypeAgent, state: PrototypeAgentState) -> Any:
    return agent._shared_feature_horde_bundle(state.oak_state)


def _utility_state(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> PrototypeFeatureUtilityState:
    utility = agent._feature_utility_component_state(state.oak_state)
    assert type(utility) is PrototypeFeatureUtilityState
    return utility


def _replace_utility_state(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
    utility: PrototypeFeatureUtilityState,
) -> PrototypeAgentState:
    consumer = _consumer_bundle(agent, state)
    slot = agent._oak_state_slot(
        consumer.oak_state,
        consumer.consumer_binding,
        consumer.horde_state,
        utility,
    )
    return cast(PrototypeAgentState, state.replace(oak_state=slot))


def _curation_diagnostics(
    result: Any,
) -> PrototypeFeatureUtilityCurationIntegrationDiagnostics:
    diagnostics = result.prototype_feature_utility_curation_diagnostics
    assert type(diagnostics) is PrototypeFeatureUtilityCurationIntegrationDiagnostics
    assert type(diagnostics.policy) is PrototypeFeatureUtilityCurationDiagnostics
    return diagnostics


def _utility_diagnostics(
    result: Any,
) -> PrototypeFeatureUtilityIntegrationDiagnostics:
    diagnostics = result.prototype_feature_utility_diagnostics
    assert type(diagnostics) is PrototypeFeatureUtilityIntegrationDiagnostics
    return diagnostics


def _start_idle(
    agent: PrototypeAgent,
    observation: jax.Array,
    *,
    seed: int | None = None,
) -> tuple[PrototypeAgentState, int]:
    seeds = range(32) if seed is None else (seed,)
    for candidate in seeds:
        state = agent.start(agent.init(jr.key(candidate)), observation)
        if int(_consumer_bundle(agent, state).oak_state.stomp_state.executing_option) == -1:
            return state, candidate
    raise AssertionError("could not obtain a deterministic idle initial decision")


def _transition(
    state: PrototypeAgentState,
    next_observation: jax.Array,
    *,
    reward: float = 0.4,
    cumulants: jax.Array | None = None,
) -> PrototypeTransition:
    if cumulants is None:
        cumulants = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=cumulants,
        horde_discounts=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )


def _force_next_primitive(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    """Keep the next decision at a lifecycle-safe primitive boundary."""

    consumer = _consumer_bundle(agent, state)
    stomp = consumer.oak_state.stomp_state
    learner = stomp.base_learner_state
    biases = tuple(
        jnp.full_like(bias, 100.0 if index == 0 else -100.0)
        for index, bias in enumerate(learner.head_params.biases)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(biases=biases)
    )
    new_oak = consumer.oak_state.replace(
        stomp_state=stomp.replace(base_learner_state=learner)
    )
    slot = agent._oak_state_slot(
        new_oak,
        consumer.consumer_binding,
        consumer.horde_state,
        _utility_state(agent, state),
    )
    return cast(PrototypeAgentState, state.replace(oak_state=slot))


def _eligible_candidate_indices(state: PrototypeAgentState) -> list[int]:
    learner = _feature_state(state).learner_state
    active = {
        tuple(row)
        for row in np.stack(
            (np.asarray(learner.feature_left), np.asarray(learner.feature_right)),
            axis=1,
        ).tolist()
    }
    candidates = np.stack(
        (np.asarray(learner.candidate_left), np.asarray(learner.candidate_right)),
        axis=1,
    )
    return [
        index
        for index, row in enumerate(candidates.tolist())
        if tuple(row) not in active
    ]


def _force_proxy_promotion(
    state: PrototypeAgentState,
    *,
    legacy_candidate: int | None = None,
    audit_candidate: int | None = None,
) -> PrototypeAgentState:
    wrapper = cast(PrototypeFeatureRepresentationState, state.state_builder_state)
    feature = wrapper.feature_lifecycle_state
    learner = feature.learner_state
    eligible = _eligible_candidate_indices(state)
    legacy = eligible[0] if legacy_candidate is None else legacy_candidate
    candidate_utilities = jnp.zeros_like(learner.candidate_utilities)
    candidate_utilities = candidate_utilities.at[legacy].set(0.95)
    if audit_candidate is not None:
        candidate_utilities = candidate_utilities.at[audit_candidate].set(0.8)
    learner = learner.replace(
        utilities=jnp.asarray([0.0, 0.4], dtype=jnp.float32),
        candidate_utilities=candidate_utilities,
    )
    forced = cast(
        PrototypeAgentState,
        state.replace(
            state_builder_state=wrapper.replace(
                feature_lifecycle_state=feature.replace(learner_state=learner)
            )
        ),
    )
    return forced


def _seed_opposed_audit_ranks(
    agent: PrototypeAgent,
    state: PrototypeAgentState,
) -> tuple[PrototypeAgentState, int, int]:
    eligible = _eligible_candidate_indices(state)
    legacy_candidate, audit_candidate = eligible[:2]
    state = _force_proxy_promotion(
        state,
        legacy_candidate=legacy_candidate,
        audit_candidate=audit_candidate,
    )
    utility = _utility_state(agent, state)
    collision = jnp.any(
        jnp.all(
            utility.candidate_descriptors[:, None, :]
            == utility.active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    active_utilities = jnp.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        dtype=jnp.float32,
    )
    candidate_utilities = jnp.zeros_like(utility.candidate_task_utilities)
    candidate_utilities = candidate_utilities.at[:, legacy_candidate].set(0.1)
    candidate_utilities = candidate_utilities.at[:, audit_candidate].set(0.9)
    candidate_utilities = jnp.where(
        collision[None, :],
        0.0,
        candidate_utilities,
    )
    utility = utility.replace(
        active_task_utilities=active_utilities,
        candidate_shadow_weights=jnp.zeros_like(utility.candidate_shadow_weights),
        candidate_task_utilities=candidate_utilities,
        candidate_second_moments=jnp.where(
            collision,
            0.0,
            utility.candidate_second_moments,
        ),
    )
    return (
        _replace_utility_state(agent, state, utility),
        legacy_candidate,
        audit_candidate,
    )


def _materialize_keys(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: (
            jr.key_data(value)
            if getattr(value, "dtype", None) is not None
            and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else value
        ),
        tree,
    )


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_horde_exact(left: Any, right: Any) -> None:
    _assert_tree_exact(
        left.replace(birth_timestamp=0.0, uptime_s=0.0),
        right.replace(birth_timestamp=0.0, uptime_s=0.0),
    )


def _tree_nbytes(tree: Any) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(tree))


def _canonical_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_opt_in_requires_the_exact_v14_shared_lane_and_adds_only_v15_shell() -> None:
    curation_config = PrototypeFeatureUtilityCurationConfig(
        minimum_task_evidence=1
    )
    enabled_config = _agent_config(curation=True)
    enabled = PrototypeAgent(enabled_config)

    assert enabled_config.prototype_feature_utility_curation == curation_config
    assert enabled.to_config()["prototype_feature_utility_curation"] == (
        curation_config.to_config()
    )
    assert PrototypeAgentConfig.from_config(enabled.to_config()).to_config() == (
        enabled.to_config()
    )
    assert (
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA
        == "alberta.prototype_agent.v15"
    )
    initial = enabled.init(jr.key(0))
    assert type(initial.oak_state) is PrototypeFeatureOaKHordeUtilityCurationState
    assert type(initial.oak_state.utility_state) is PrototypeFeatureOaKHordeUtilityState
    assert initial.horde_state is None

    with pytest.raises(ValueError, match="prototype_feature_utility_curation"):
        dataclasses.replace(
            _agent_config(curation=False, utility=False),
            prototype_feature_utility_curation=curation_config,
        )
    with pytest.raises(ValueError, match="minimum_task_evidence|exceed"):
        PrototypeAgent(
            _agent_config(
                curation=True,
                max_observations=1,
                minimum_task_evidence=2,
            )
        )
    with pytest.raises(ValueError, match="at least one candidate"):
        dataclasses.replace(
            _agent_config(curation=False),
            prototype_feature_lifecycle=dataclasses.replace(
                _feature_config(),
                candidate_pair_slots=0,
            ),
            prototype_feature_utility=dataclasses.replace(
                _utility_config(),
                candidate_pair_slots=0,
            ),
            prototype_feature_utility_curation=curation_config,
        )


def test_below_floor_is_a_valid_veto_with_no_legacy_rank_fallback() -> None:
    ranked = _agent(
        curation=True,
        replacement_interval=1,
        minimum_task_evidence=2,
    )
    audited = _agent(curation=False, replacement_interval=1)
    observation = jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32)
    ranked_state, seed = _start_idle(ranked, observation)
    audited_state, _ = _start_idle(audited, observation, seed=seed)
    ranked_state = _force_next_primitive(
        ranked,
        _force_proxy_promotion(ranked_state),
    )
    audited_state = _force_next_primitive(
        audited,
        _force_proxy_promotion(audited_state),
    )
    old_ranked_descriptors = _feature_state(
        ranked_state
    ).router_state.descriptors
    assert bool(ranked._checkpoint_state_valid(ranked_state))
    assert bool(audited._checkpoint_state_valid(audited_state))

    next_observation = jnp.asarray([-0.25, 1.25, -2.0, 0.75], dtype=jnp.float32)
    cumulants = jnp.asarray([0.6, -0.4], dtype=jnp.float32)
    ranked_result = ranked.update_transition(
        ranked_state,
        _transition(ranked_state, next_observation, cumulants=cumulants),
    )
    audited_result = audited.update_transition(
        audited_state,
        _transition(audited_state, next_observation, cumulants=cumulants),
    )

    diagnostics = _curation_diagnostics(ranked_result)
    assert bool(ranked_result.transition_diagnostics.valid)
    assert bool(diagnostics.outer_transaction_committed)
    assert bool(diagnostics.observation_applied)
    assert bool(diagnostics.policy.transaction_valid)
    assert bool(diagnostics.policy.override_enabled)
    assert not bool(diagnostics.policy.curation_ready)
    assert not bool(diagnostics.policy.any_active_rank_ready)
    assert not bool(diagnostics.policy.any_candidate_rank_ready)
    assert bool(diagnostics.priority_override_supplied)
    assert bool(diagnostics.priority_override_consulted)
    assert not bool(diagnostics.curation_allowed)
    assert int(diagnostics.selected_active_slot) == -1
    assert int(diagnostics.selected_candidate_slot) == -1
    np.testing.assert_array_equal(
        diagnostics.selected_active_descriptor,
        np.asarray([-1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        diagnostics.selected_candidate_descriptor,
        np.asarray([-1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        diagnostics.policy.emitted_active_ranks,
        jnp.full(
            (ACTIVE_SLOTS,),
            CURATION_ACTIVE_INELIGIBLE_RANK,
            dtype=jnp.float32,
        ),
    )
    np.testing.assert_array_equal(
        diagnostics.policy.emitted_candidate_ranks,
        jnp.full(
            (CANDIDATE_SLOTS,),
            CURATION_CANDIDATE_INELIGIBLE_RANK,
            dtype=jnp.float32,
        ),
    )
    assert not bool(diagnostics.lifecycle_curation_proposed)
    assert not bool(diagnostics.lifecycle_curation_deferred)
    assert not bool(diagnostics.lifecycle_curation_committed)
    assert not bool(diagnostics.lifecycle_curation_rolled_back)
    np.testing.assert_array_equal(
        _feature_state(ranked_result.state).router_state.descriptors,
        old_ranked_descriptors,
    )
    assert int(_feature_state(ranked_result.state).observe_count) == 1

    audited_feature = audited_result.prototype_feature_lifecycle_diagnostics
    assert audited_feature is not None
    assert bool(audited_result.transition_diagnostics.valid)
    assert bool(audited_feature.lifecycle.curation_committed)
    assert not bool(
        jnp.array_equal(
            _feature_state(audited_result.state).router_state.descriptors,
            old_ranked_descriptors,
        )
    )


def test_post_observation_audit_ranks_override_opposed_proxy_selection() -> None:
    agent = _agent(
        curation=True,
        replacement_interval=2,
        minimum_task_evidence=1,
    )
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    state = _force_next_primitive(agent, state)
    warmup = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.25, 1.5, -0.75, 2.0], dtype=jnp.float32),
            cumulants=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        ),
    )
    assert bool(warmup.transition_diagnostics.valid)
    assert int(_utility_state(agent, warmup.state).observation_count) == 1

    state, legacy_candidate, audit_candidate = _seed_opposed_audit_ranks(
        agent,
        _force_next_primitive(agent, warmup.state),
    )
    assert bool(agent._checkpoint_state_valid(state))
    old_feature = _feature_state(state)
    old_active_descriptors = old_feature.router_state.descriptors
    old_candidate_descriptors = jnp.stack(
        (
            old_feature.learner_state.candidate_left,
            old_feature.learner_state.candidate_right,
        ),
        axis=1,
    )
    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
            cumulants=jnp.asarray([0.5, -0.2], dtype=jnp.float32),
        ),
    )

    diagnostics = _curation_diagnostics(result)
    feature_diagnostics = result.prototype_feature_lifecycle_diagnostics
    assert feature_diagnostics is not None
    legacy = agent.prototype_feature_lifecycle.learner.update(
        old_feature.learner_state,
        state.current_representation[:BASE_DIM],
        feature_diagnostics.task_targets,
    )
    assert int(legacy.curation_selected_active_worst_slot) == 0
    assert int(legacy.curation_selected_promotion_candidate) == legacy_candidate
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.outer_transaction_committed)
    assert bool(diagnostics.observation_applied)
    assert bool(diagnostics.policy.curation_ready)
    assert bool(diagnostics.priority_override_supplied)
    assert bool(diagnostics.priority_override_consulted)
    assert bool(diagnostics.curation_allowed)
    assert int(diagnostics.selected_active_slot) == 1
    assert int(diagnostics.selected_candidate_slot) == audit_candidate
    np.testing.assert_array_equal(
        diagnostics.selected_active_descriptor,
        old_active_descriptors[1],
    )
    np.testing.assert_array_equal(
        diagnostics.selected_candidate_descriptor,
        old_candidate_descriptors[audit_candidate],
    )
    assert float(diagnostics.policy.emitted_active_ranks[1]) < float(
        diagnostics.policy.emitted_active_ranks[0]
    )
    assert float(diagnostics.policy.emitted_candidate_ranks[audit_candidate]) > float(
        diagnostics.policy.emitted_candidate_ranks[legacy_candidate]
    )
    assert bool(diagnostics.lifecycle_curation_proposed)
    assert bool(diagnostics.lifecycle_curation_committed)
    assert not bool(diagnostics.lifecycle_curation_deferred)
    assert not bool(diagnostics.lifecycle_curation_rolled_back)
    new_descriptors = _feature_state(result.state).router_state.descriptors
    np.testing.assert_array_equal(new_descriptors[0], old_active_descriptors[0])
    np.testing.assert_array_equal(
        new_descriptors[1],
        old_candidate_descriptors[audit_candidate],
    )


def test_fixed_task_mass_and_collision_exclusion_survive_missing_task() -> None:
    agent = _agent(
        curation=True,
        replacement_interval=3,
        minimum_task_evidence=1,
    )
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    first = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([0.25, 1.5, -0.75, 2.0], dtype=jnp.float32),
            cumulants=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
        ),
    )
    assert bool(first.transition_diagnostics.valid)
    utility = _utility_state(agent, first.state)
    collision = jnp.any(
        jnp.all(
            utility.candidate_descriptors[:, None, :]
            == utility.active_descriptors[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    active_utilities = jnp.asarray(
        [[0.2, 0.8], [0.4, 0.0], [0.6, 0.4]],
        dtype=jnp.float32,
    )
    candidate_utilities = jnp.asarray(
        [
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.6],
        ],
        dtype=jnp.float32,
    )
    candidate_utilities = jnp.where(
        collision[None, :],
        0.0,
        candidate_utilities,
    )
    utility = utility.replace(
        active_task_utilities=active_utilities,
        candidate_task_utilities=candidate_utilities,
        candidate_shadow_weights=jnp.zeros_like(utility.candidate_shadow_weights),
        candidate_second_moments=jnp.where(
            collision,
            0.0,
            utility.candidate_second_moments,
        ),
    )
    state = _replace_utility_state(agent, first.state, utility)
    assert bool(agent._checkpoint_state_valid(state))

    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
            cumulants=jnp.asarray([0.7, jnp.nan], dtype=jnp.float32),
        ),
    )
    diagnostics = _curation_diagnostics(result)
    observation = _utility_diagnostics(result).observation
    weights = jnp.asarray([0.5, 0.25, 0.25], dtype=jnp.float32)
    expected_active = jnp.sum(
        weights[:, None] * observation.active_task_utilities_after,
        axis=0,
    )
    expected_candidates = jnp.sum(
        weights[:, None] * observation.candidate_task_utilities_after,
        axis=0,
    )

    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.observation_applied)
    np.testing.assert_array_equal(diagnostics.policy.task_weights, weights)
    np.testing.assert_allclose(
        diagnostics.policy.raw_active_fixed_mass_utilities,
        expected_active,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics.policy.raw_candidate_fixed_mass_utilities,
        expected_candidates,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        diagnostics.policy.candidate_collision_mask,
        collision,
    )
    np.testing.assert_array_equal(
        diagnostics.policy.candidate_rank_ready_mask,
        ~collision,
    )
    for collision_index in np.flatnonzero(np.asarray(collision)):
        assert float(
            diagnostics.policy.emitted_candidate_ranks[collision_index]
        ) == CURATION_CANDIDATE_INELIGIBLE_RANK
    assert not bool(observation.target_available[2])
    np.testing.assert_array_equal(
        observation.active_task_evidence_counts_after[2],
        observation.active_task_evidence_counts_before[2],
    )
    assert bool(diagnostics.policy.curation_ready)
    assert not bool(diagnostics.priority_override_consulted)
    assert int(diagnostics.selected_active_slot) == -1
    assert int(diagnostics.selected_candidate_slot) == -1


def test_capacity_cap_is_valid_neutral_and_does_not_touch_lifecycle_or_audit() -> None:
    agent = _agent(
        curation=True,
        replacement_interval=0,
        max_observations=1,
        minimum_task_evidence=1,
    )
    state, _ = _start_idle(
        agent,
        jnp.asarray([0.5, -1.0, 2.0, -0.25], dtype=jnp.float32),
    )
    first = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([1.0, 0.25, -0.5, 0.75], dtype=jnp.float32),
        ),
    )
    assert bool(first.transition_diagnostics.valid)
    feature_before = _feature_state(first.state)
    utility_before = _utility_state(agent, first.state)
    consumer_before = _consumer_bundle(agent, first.state)

    second = agent.update_transition(
        first.state,
        _transition(
            first.state,
            jnp.asarray([-0.75, 1.5, 0.1, -1.0], dtype=jnp.float32),
        ),
    )
    diagnostics = _curation_diagnostics(second)
    utility_diagnostics = _utility_diagnostics(second)

    assert bool(second.transition_diagnostics.valid)
    assert bool(diagnostics.outer_transaction_committed)
    assert not bool(diagnostics.observation_applied)
    assert bool(diagnostics.policy.transaction_valid)
    assert bool(diagnostics.policy.observation_capacity_capped)
    assert not bool(diagnostics.policy.observation_capacity_available)
    assert not bool(diagnostics.policy.override_enabled)
    assert not bool(diagnostics.policy.curation_ready)
    # A disabled, finite override is still supplied to make legacy fallback
    # impossible; the cap keeps it unconsulted and lifecycle state neutral.
    assert bool(diagnostics.priority_override_supplied)
    assert not bool(diagnostics.priority_override_consulted)
    assert not bool(diagnostics.curation_allowed)
    assert int(diagnostics.selected_active_slot) == -1
    assert int(diagnostics.selected_candidate_slot) == -1
    assert not bool(diagnostics.lifecycle_curation_proposed)
    assert not bool(diagnostics.lifecycle_curation_committed)
    assert bool(utility_diagnostics.observation.capacity_capped)
    assert not bool(utility_diagnostics.observation.transaction_applied)
    _assert_tree_exact(_feature_state(second.state), feature_before)
    _assert_tree_exact(_utility_state(agent, second.state), utility_before)
    consumer_after = _consumer_bundle(agent, second.state)
    assert int(consumer_after.oak_state.step_count) == int(
        consumer_before.oak_state.step_count
    ) + 1
    assert int(consumer_after.horde_state.step_count) == int(
        consumer_before.horde_state.step_count
    ) + 1


def test_noncadence_v15_is_bit_exact_with_v14_owned_state() -> None:
    ranked = _agent(curation=True, replacement_interval=3)
    audited = _agent(curation=False, replacement_interval=3)
    observation = jnp.asarray([0.25, -0.75, 1.5, 0.5], dtype=jnp.float32)
    ranked_state, seed = _start_idle(ranked, observation)
    audited_state, _ = _start_idle(audited, observation, seed=seed)
    transition_ranked = _transition(
        ranked_state,
        jnp.asarray([-0.5, 2.0, 0.1, -1.0], dtype=jnp.float32),
        cumulants=jnp.asarray([0.4, -0.3], dtype=jnp.float32),
    )
    transition_audited = _transition(
        audited_state,
        transition_ranked.next_observation,
        cumulants=transition_ranked.horde_cumulants,
    )

    ranked_result = ranked.update_transition(ranked_state, transition_ranked)
    audited_result = audited.update_transition(audited_state, transition_audited)
    diagnostics = _curation_diagnostics(ranked_result)

    assert bool(ranked_result.transition_diagnostics.valid)
    assert bool(audited_result.transition_diagnostics.valid)
    np.testing.assert_array_equal(ranked_result.action, audited_result.action)
    np.testing.assert_array_equal(
        ranked_result.state.current_decision_id,
        audited_result.state.current_decision_id,
    )
    ranked_consumer = _consumer_bundle(ranked, ranked_result.state)
    audited_consumer = _consumer_bundle(audited, audited_result.state)
    _assert_tree_exact(ranked_consumer.oak_state, audited_consumer.oak_state)
    _assert_horde_exact(ranked_consumer.horde_state, audited_consumer.horde_state)
    _assert_tree_exact(
        _feature_state(ranked_result.state),
        _feature_state(audited_result.state),
    )
    _assert_tree_exact(
        _utility_state(ranked, ranked_result.state),
        _utility_state(audited, audited_result.state),
    )
    _assert_tree_exact(
        ranked_result.prototype_feature_utility_diagnostics,
        audited_result.prototype_feature_utility_diagnostics,
    )
    assert bool(diagnostics.observation_applied)
    assert bool(diagnostics.priority_override_supplied)
    assert not bool(diagnostics.priority_override_consulted)
    assert not bool(diagnostics.lifecycle_curation_proposed)
    assert bool(diagnostics.outer_transaction_committed)


def test_committed_ranked_route_calls_two_routes_one_rebind_and_one_observe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(curation=True, replacement_interval=1)
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    state, _, _ = _seed_opposed_audit_ranks(
        agent,
        _force_next_primitive(agent, state),
    )
    lifecycle = agent.prototype_feature_lifecycle
    auditor = agent.prototype_feature_utility
    assert lifecycle is not None
    assert auditor is not None
    calls = {"route": 0, "observe": 0, "rebind": 0}
    original_route = lifecycle.router.route
    original_observe = auditor.observe
    original_rebind = auditor.rebind

    def counted_route(*args: Any, **kwargs: Any) -> Any:
        calls["route"] += 1
        return original_route(*args, **kwargs)

    def counted_observe(*args: Any, **kwargs: Any) -> Any:
        calls["observe"] += 1
        return original_observe(*args, **kwargs)

    def counted_rebind(*args: Any, **kwargs: Any) -> Any:
        calls["rebind"] += 1
        return original_rebind(*args, **kwargs)

    monkeypatch.setattr(lifecycle.router, "route", counted_route)
    monkeypatch.setattr(auditor, "observe", counted_observe)
    monkeypatch.setattr(auditor, "rebind", counted_rebind)
    old_count = int(_utility_state(agent, state).observation_count)

    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
        ),
    )
    diagnostics = _curation_diagnostics(result)
    utility_diagnostics = _utility_diagnostics(result)

    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.lifecycle_curation_committed)
    assert bool(utility_diagnostics.observation.transaction_applied)
    assert bool(utility_diagnostics.rebind_required)
    assert bool(utility_diagnostics.rebind.transaction_applied)
    assert calls == {"route": 2, "observe": 1, "rebind": 1}
    assert int(_utility_state(agent, result.state).observation_count) == old_count + 1
    lifecycle_budget = lifecycle.resource_budget(
        _feature_state(result.state),
        _consumer_bundle(agent, result.state).horde_state,
    )
    assert lifecycle_budget.router_calls_per_observe == 2
    assert lifecycle_budget.router_calls_per_committed_curation == 2
    assert agent.prototype_feature_utility_resource_budget.router_calls_per_observe == 0
    assert (
        agent.prototype_feature_utility_curation_resource_budget.router_calls_per_rank
        == 0
    )


def test_corrupt_post_observation_audit_rolls_back_but_keeps_attempt_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(curation=True, replacement_interval=1)
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    state = _force_next_primitive(agent, _force_proxy_promotion(state))
    auditor = agent.prototype_feature_utility
    assert auditor is not None
    original_observe = auditor.observe

    def corrupt_observe(*args: Any, **kwargs: Any) -> Any:
        result = original_observe(*args, **kwargs)
        corrupt = result.state.replace(
            active_task_utilities=(
                result.state.active_task_utilities.at[0, 0].set(jnp.nan)
            )
        )
        return result.replace(state=corrupt)

    monkeypatch.setattr(auditor, "observe", corrupt_observe)
    result = agent.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
        ),
    )
    diagnostics = _curation_diagnostics(result)
    utility_diagnostics = _utility_diagnostics(result)

    assert bool(result.transition_diagnostics.rejected)
    assert not bool(result.transition_diagnostics.valid)
    assert bool(utility_diagnostics.observation.transaction_applied)
    assert not bool(diagnostics.policy.state_valid)
    assert not bool(diagnostics.policy.transaction_valid)
    # Supplying the policy's disabled zero payload is the fail-closed signal;
    # the integration turns it into an enabled all-sentinel safety override so
    # the lifecycle cannot silently restore legacy proxy ranking.
    assert bool(diagnostics.priority_override_supplied)
    assert bool(diagnostics.priority_override_consulted)
    assert not bool(diagnostics.curation_allowed)
    assert not bool(diagnostics.outer_transaction_committed)
    assert int(diagnostics.selected_active_slot) == -1
    assert int(diagnostics.selected_candidate_slot) == -1
    np.testing.assert_array_equal(
        diagnostics.selected_active_descriptor,
        np.asarray([-1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        diagnostics.selected_candidate_descriptor,
        np.asarray([-1, -1], dtype=np.int32),
    )
    assert not bool(diagnostics.lifecycle_curation_committed)
    _assert_tree_exact(result.state, state)


def test_v15_ranked_lane_is_eager_jit_exact() -> None:
    agent = _agent(curation=True, replacement_interval=1)
    state, _ = _start_idle(
        agent,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    state, _, _ = _seed_opposed_audit_ranks(
        agent,
        _force_next_primitive(agent, state),
    )
    transition = _transition(
        state,
        jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
    )

    eager = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)

    _assert_tree_exact(compiled, eager)
    assert bool(compiled.transition_diagnostics.valid)
    diagnostics = _curation_diagnostics(compiled)
    assert bool(diagnostics.priority_override_consulted)
    assert bool(diagnostics.lifecycle_curation_committed)
    assert bool(diagnostics.outer_transaction_committed)


def test_resource_budget_has_zero_policy_state_rng_learning_and_routing() -> None:
    ranked = _agent(curation=True)
    audited = _agent(curation=False)
    budget = ranked.prototype_feature_utility_curation_resource_budget

    assert budget is not None
    assert budget == ranked.prototype_feature_utility_curation.resource_budget()
    assert budget.persistent_logical_scalars == 0
    assert budget.persistent_state_nbytes == 0
    assert budget.rng_draws_per_rank == 0
    assert budget.backward_passes_per_rank == 0
    assert budget.consumer_updates_per_rank == 0
    assert budget.router_calls_per_rank == 0
    assert budget.curation_decisions_per_rank == 0
    assert budget.ranking_influence is True
    assert budget.curation_authority is False
    assert budget.promotion_authority is False
    assert budget.go_no_go_authority is False
    assert budget.scientific_promotion_allowed is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE is True
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED is False

    ranked_state = ranked.init(jr.key(7))
    audited_state = audited.init(jr.key(7))
    assert _tree_nbytes(ranked_state) == _tree_nbytes(audited_state) + 32
    assert type(ranked_state.oak_state) is PrototypeFeatureOaKHordeUtilityCurationState
    np.testing.assert_array_equal(
        ranked_state.oak_state.utility_state.schema_digest,
        audited_state.oak_state.schema_digest,
    )


def test_v15_populated_round_trip_relabels_and_semantic_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    ranked = _agent(curation=True, replacement_interval=1)
    state, _ = _start_idle(
        ranked,
        jnp.asarray([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32),
    )
    state, _, _ = _seed_opposed_audit_ranks(
        ranked,
        _force_next_primitive(ranked, state),
    )
    routed = ranked.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-1.0, 0.75, 1.25, -2.0], dtype=jnp.float32),
        ),
    )
    diagnostics = _curation_diagnostics(routed)
    utility_diagnostics = _utility_diagnostics(routed)
    assert bool(routed.transition_diagnostics.valid)
    assert bool(diagnostics.lifecycle_curation_committed)
    assert bool(utility_diagnostics.rebind_required)
    assert bool(utility_diagnostics.rebind.transaction_applied)
    persisted = _utility_state(ranked, routed.state)
    assert int(persisted.semantic_generation) > 0
    assert int(persisted.observation_count) > 0
    assert np.any(np.asarray(persisted.active_task_evidence_counts) > 0)

    path = tmp_path / "ranked-v15"
    save_prototype_checkpoint(ranked, routed.state, path)
    assert load_checkpoint_metadata(path)["schema"] == (
        PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA
    )
    restored_agent, restored_state = load_prototype_checkpoint(path)
    assert restored_agent.to_config() == ranked.to_config()
    _assert_tree_exact(restored_state, routed.state)

    ranked_config = ranked.to_config()
    relabeled_v14 = tmp_path / "ranked-relabeled-v14"
    save_checkpoint(
        routed.state,
        relabeled_v14,
        metadata={
            "schema": PROTOTYPE_FEATURE_UTILITY_CHECKPOINT_SCHEMA,
            "agent_config": ranked_config,
            "config_sha256": _canonical_digest(ranked_config),
        },
    )
    with pytest.raises(ValueError, match="v15"):
        load_prototype_checkpoint(relabeled_v14)

    audited = _agent(curation=False)
    audited_state = audited.init(jr.key(8))
    audited_config = audited.to_config()
    relabeled_v15 = tmp_path / "audit-relabeled-v15"
    save_checkpoint(
        audited_state,
        relabeled_v15,
        metadata={
            "schema": PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
            "agent_config": audited_config,
            "config_sha256": _canonical_digest(audited_config),
        },
    )
    with pytest.raises(
        ValueError,
        match="curation|ranking|v15",
    ):
        load_prototype_checkpoint(relabeled_v15)

    tampered_config = {
        **ranked_config,
        "prototype_feature_utility_curation": {
            **ranked_config["prototype_feature_utility_curation"],
            "minimum_task_evidence": 2,
        },
    }
    semantic_tamper = tmp_path / "ranked-semantic-tamper-v15"
    save_checkpoint(
        routed.state,
        semantic_tamper,
        metadata={
            "schema": PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA,
            "agent_config": tampered_config,
            "config_sha256": _canonical_digest(tampered_config),
        },
    )
    with pytest.raises(ValueError, match="inconsistent|digest"):
        load_prototype_checkpoint(semantic_tamper)

    shell = cast(
        PrototypeFeatureOaKHordeUtilityCurationState,
        routed.state.oak_state,
    )
    bad_digest = shell.schema_digest.at[0].set(shell.schema_digest[0] ^ 1)
    corrupt_state = cast(
        PrototypeAgentState,
        routed.state.replace(
            oak_state=shell.replace(schema_digest=bad_digest),
        ),
    )
    assert not bool(ranked._checkpoint_state_valid(corrupt_state))
    with pytest.raises(ValueError, match="inconsistent"):
        save_prototype_checkpoint(
            ranked,
            corrupt_state,
            tmp_path / "corrupt-v15-digest",
        )
    rejected = ranked.update_transition(
        corrupt_state,
        _transition(
            corrupt_state,
            jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        ),
    )
    assert bool(rejected.transition_diagnostics.rejected)
    attempted = _curation_diagnostics(rejected)
    assert not bool(attempted.outer_transaction_committed)
    _assert_tree_exact(rejected.state, corrupt_state)


def test_wp71c_public_exports_are_exact_and_mechanism_only() -> None:
    names = (
        "PROTOTYPE_FEATURE_UTILITY_CURATION_CHECKPOINT_SCHEMA",
        "PrototypeFeatureOaKHordeUtilityCurationState",
        "PrototypeFeatureUtilityCurationIntegrationDiagnostics",
        "PrototypeFeatureUtilityCurationConfig",
        "PrototypeFeatureUtilityCurationDiagnostics",
        "PrototypeFeatureUtilityCurationPolicy",
        "PrototypeFeatureUtilityCurationResourceBudget",
        "PrototypeFeatureUtilityCurationResult",
        "PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE",
        "PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY",
        "PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY",
        "PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY",
        "PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED",
    )
    for module in (alberta_core, alberta_framework):
        for name in names:
            assert hasattr(module, name), name
            assert module.__all__.count(name) == 1

    assert PROTOTYPE_FEATURE_UTILITY_CURATION_RANKING_INFLUENCE is True
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_PROMOTION_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_GO_NO_GO_AUTHORITY is False
    assert PROTOTYPE_FEATURE_UTILITY_CURATION_SCIENTIFIC_PROMOTION_ALLOWED is False
