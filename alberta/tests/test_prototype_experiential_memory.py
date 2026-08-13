# mypy: disable-error-code="attr-defined,call-arg,no-any-return,union-attr"
"""Transactional Prototype integration for bounded experiential memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryAdvantageGateConfig,
)
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    replace_dispatched_primitive_action,
)
from alberta_framework.core.partner_policy_fusion import (
    ROUTE_ACCEPT,
    PartnerMessageBatch,
    PartnerPolicyFusionConfig,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeExperientialMemoryInput,
    PrototypeInteractionState,
    PrototypeMemoryInteractionState,
    PrototypePartnerPolicyFusionInput,
    PrototypeRecurrentLatentWorldModelState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsembleConfig,
)

pytestmark = pytest.mark.unit


def _oak() -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=8),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_hidden_sizes=(),
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _memory(**overrides: Any) -> ExperientialMemoryConfig:
    values: dict[str, Any] = {
        "capacity": 3,
        "observation_dim": 2,
        "key_dim": 2,
        "action_dim": 2,
        "outcome_dim": 3,
        "top_k": 2,
        "min_neighbors": 1,
        "distance_scale": 1.0,
        "min_similarity": 0.1,
        "min_effective_reliability": 0.01,
        "max_uncertainty": 1.0,
        "max_safety_cost": 1.0,
        "max_age": 100,
        "staleness_scale": 100.0,
        "utility_decay": 1.0,
    }
    values.update(overrides)
    return ExperientialMemoryConfig(**values)


def _fusion() -> PartnerPolicyFusionConfig:
    return PartnerPolicyFusionConfig(
        max_partners=1,
        context_dim=2,
        n_actions=2,
        max_abs_context=10.0,
        assistance_value_bound=10.0,
    )


def _agent(
    *,
    ia: bool = False,
    partner: bool = False,
    recurrent: bool = False,
    memory_overrides: dict[str, Any] | None = None,
    advantage_gate: ExperientialMemoryAdvantageGateConfig | None = None,
) -> PrototypeAgent:
    ia_config = (
        IAConfig(
            cerebellum=ExoCerebellumConfig(
                n_demons=1,
                obs_dim=2,
                step_size=0.05,
            ),
            cortex=_oak(),
        )
        if ia
        else None
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak(),
            ia=ia_config,
            partner_policy_fusion=_fusion() if partner else None,
            experiential_memory=_memory(
                **({} if memory_overrides is None else memory_overrides)
            ),
            experiential_memory_advantage_gate=advantage_gate,
            recurrent_latent_world_model_ensemble=(
                RecurrentLatentWorldModelEnsembleConfig(
                    observation_dim=2,
                    n_actions=2,
                    latent_dim=2,
                    ensemble_size=2,
                    learning_rate=0.01,
                    bootstrap_probability=0.7,
                    uncertainty_warmup_steps=1,
                    initialization_scale=0.1,
                    max_updates=100,
                )
                if recurrent
                else None
            ),
        )
    )


def _next_id(state: Any) -> jnp.ndarray:
    return state.current_decision_id.at[3].set(
        state.current_decision_id[3] + jnp.asarray(1, dtype=jnp.uint32)
    )


def _sidecar(
    state: Any,
    provenance_id: int,
    *,
    mask: tuple[bool, bool] = (True, True),
) -> PrototypeExperientialMemoryInput:
    return PrototypeExperientialMemoryInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        current_prototype_decision_id=state.current_decision_id,
        next_prototype_decision_id=_next_id(state),
        query_representation_version=jnp.asarray(1, dtype=jnp.int32),
        entry_representation_version=jnp.asarray(1, dtype=jnp.int32),
        query_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        query_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        entry_uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        entry_uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
        next_action_safety_mask=jnp.asarray(mask, dtype=jnp.bool_),
    )


def _transition(
    state: Any,
    *,
    reward: float = 0.0,
    next_observation: tuple[float, float] = (0.0, 0.0),
) -> PrototypeTransition:
    next_obs = jnp.asarray(next_observation, dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_obs,
        next_decision_observation=next_obs,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _partner_input(
    agent: PrototypeAgent,
    state: Any,
    *,
    suggested_action: int,
) -> PrototypePartnerPolicyFusionInput:
    fusion = agent.partner_policy_fusion
    assert fusion is not None
    messages = fusion.empty_messages()
    next_step = int(state.step_count) + 1
    next_event = int(state.observation_event_count) + 1
    next_step_words = _add_identity_words(state.step_words, 1)
    next_event_words = _add_identity_words(
        state.observation_event_words,
        1,
    )
    messages = cast(
        PartnerMessageBatch,
        messages.replace(
            available=messages.available.at[0].set(True),
            partner_id=messages.partner_id.at[0].set(0),
            observation_id=messages.observation_id.at[0].set(11),
            context_id=messages.context_id.at[0].set(12),
            suggested_action=messages.suggested_action.at[0].set(
                suggested_action
            ),
            declared_confidence=messages.declared_confidence.at[0].set(1.0),
            rationale_reference=messages.rationale_reference.at[0].set(13),
            provenance_reference=messages.provenance_reference.at[0].set(14),
            communication_cost=messages.communication_cost.at[0].set(0.0),
            issued_decision_id=messages.issued_decision_id.at[0].set(next_step),
            issued_event_id=messages.issued_event_id.at[0].set(next_event),
            valid_through_event_id=messages.valid_through_event_id.at[0].set(
                next_event + 1
            ),
            issued_decision_words=messages.issued_decision_words.at[0].set(
                next_step_words
            ),
            issued_event_words=messages.issued_event_words.at[0].set(
                next_event_words
            ),
            valid_through_event_words=(
                messages.valid_through_event_words.at[0].set(
                    _add_identity_words(next_event_words, 1)
                )
            ),
        ),
    )
    return PrototypePartnerPolicyFusionInput(
        available=jnp.asarray(True, dtype=jnp.bool_),
        prototype_decision_id=_next_id(state),
        observation_id=jnp.asarray(11, dtype=jnp.int32),
        context_id=jnp.asarray(12, dtype=jnp.int32),
        context_features=jnp.zeros((2,), dtype=jnp.float32),
        safety_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        keyboard_available=jnp.asarray(False, dtype=jnp.bool_),
        keyboard_vector=jnp.zeros((1,), dtype=jnp.float32),
        messages=messages,
    )


def _add_identity_words(words: Any, delta: int) -> Any:
    """Advance one concrete big-endian uint64 test identity."""

    materialized = np.asarray(words, dtype=np.uint32)
    value = (int(materialized[0]) << 32) | int(materialized[1])
    advanced = value + delta
    assert 0 <= advanced <= np.iinfo(np.uint64).max
    return jnp.asarray(
        (advanced >> 32, advanced & np.iinfo(np.uint32).max),
        dtype=jnp.uint32,
    )


def _memory_state(state: Any) -> Any:
    return cast(
        PrototypeMemoryInteractionState,
        state.ia_state,
    ).experiential_memory_state


def _force_action(agent: PrototypeAgent, state: Any, action: int) -> Any:
    replacement = replace_dispatched_primitive_action(
        state.oak_state.stomp_state,
        state.current_representation,
        jnp.asarray(action, dtype=jnp.int32),
    )
    assert not bool(replacement.decision.failed_closed)
    assert int(replacement.decision.effective_action) == action
    world_model_state = state.world_model_state
    if agent.config.recurrent_latent_world_model_ensemble is not None:
        recurrent = cast(
            PrototypeRecurrentLatentWorldModelState,
            world_model_state,
        )
        world_model_state = recurrent.replace(
            decision_cache=agent._recurrent_decision_for_observation(
                recurrent.model_state,
                state.current_representation,
                jnp.asarray(action, dtype=jnp.int32),
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
    forced = state.replace(
        oak_state=state.oak_state.replace(stomp_state=replacement.state),
        world_model_state=world_model_state,
        current_action=jnp.asarray(action, dtype=jnp.int32),
    )
    assert bool(agent._checkpoint_state_valid(forced))
    return forced


def _seed_opposite_memory_action(
    agent: PrototypeAgent,
    *,
    key: int,
    safety_cost: float = 0.0,
) -> tuple[Any, int, int]:
    state = agent.start(
        agent.init(jr.key(key)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    sidecar = _sidecar(state, 900 + key).replace(
        safety_cost=jnp.asarray(safety_cost, dtype=jnp.float32)
    )
    seeded = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=sidecar,
    )
    assert bool(seeded.transition_diagnostics.valid)
    assert bool(seeded.experiential_memory_diagnostics.transaction_applied)
    ordinary = agent.update_transition(
        seeded.state,
        _transition(seeded.state),
    )
    base_action = int(ordinary.action)
    memory_action = 1 - base_action
    memory = _memory_state(seeded.state)
    slot = int(seeded.experiential_memory_diagnostics.slot)
    adjusted_memory = memory.replace(
        entries=memory.entries.replace(
            actions=memory.entries.actions.at[slot].set(
                jax.nn.one_hot(memory_action, 2, dtype=jnp.float32)
            )
        )
    )
    adjusted_state = seeded.state.replace(
        ia_state=seeded.state.ia_state.replace(
            experiential_memory_state=adjusted_memory
        )
    )
    assert bool(agent._checkpoint_state_valid(adjusted_state))
    return seeded.replace(state=adjusted_state), base_action, memory_action


def test_memory_config_is_opt_in_and_uses_an_outer_shape_wrapper() -> None:
    default_agent = PrototypeAgent(PrototypeAgentConfig(oak=_oak()))
    default_state = default_agent.init(jr.key(0))
    assert default_state.ia_state is None
    assert "experiential_memory" not in default_agent.to_config()

    agent = PrototypeAgent(
        PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory())
    )
    state = agent.init(jr.key(1))
    assert isinstance(state.ia_state, PrototypeMemoryInteractionState)
    assert state.ia_state.interaction_state is None
    assert int(state.ia_state.experiential_memory_state.active_count) == 0

    # Importing the fixed sidecar is part of the public typed contract.
    assert PrototypeExperientialMemoryInput is not None
    assert jnp.asarray(state.ia_state.experiential_memory_state.active_count).shape == ()


def test_config_dimensions_roundtrip_exports_and_legacy_shapes() -> None:
    config = PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory())
    assert PrototypeAgentConfig.from_config(config.to_config()) == config
    assert alberta.PrototypeExperientialMemoryInput is PrototypeExperientialMemoryInput
    assert core.PrototypeMemoryInteractionState is PrototypeMemoryInteractionState

    with pytest.raises(ValueError, match="observation_dim"):
        PrototypeAgentConfig(
            oak=_oak(),
            experiential_memory=_memory(observation_dim=3),
        )
    with pytest.raises(ValueError, match="key_dim"):
        PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory(key_dim=3))
    with pytest.raises(ValueError, match="action_dim"):
        PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory(action_dim=3))
    with pytest.raises(ValueError, match="outcome_dim"):
        PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory(outcome_dim=2))
    with pytest.raises(ValueError, match="requires experiential_memory"):
        PrototypeAgentConfig(
            oak=_oak(),
            experiential_memory_advantage_gate=(
                ExperientialMemoryAdvantageGateConfig()
            ),
        )

    gated_config = PrototypeAgentConfig(
        oak=_oak(),
        experiential_memory=_memory(),
        experiential_memory_advantage_gate=(
            ExperientialMemoryAdvantageGateConfig(
                min_action_support=1,
                min_reward_advantage=0.25,
            )
        ),
    )
    assert PrototypeAgentConfig.from_config(gated_config.to_config()) == gated_config
    assert "experiential_memory_advantage_gate" in gated_config.to_config()

    ia_only = PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak(),
            ia=IAConfig(
                cerebellum=ExoCerebellumConfig(
                    n_demons=1,
                    obs_dim=2,
                    step_size=0.05,
                ),
                cortex=_oak(),
            ),
        )
    ).init(jr.key(90))
    assert not isinstance(ia_only.ia_state, PrototypeMemoryInteractionState)
    assert not isinstance(ia_only.ia_state, PrototypeInteractionState)
    partner_only = PrototypeAgent(
        PrototypeAgentConfig(
            oak=_oak(),
            partner_policy_fusion=_fusion(),
        )
    ).init(jr.key(91))
    assert isinstance(partner_only.ia_state, PrototypeInteractionState)
    assert not isinstance(partner_only.ia_state, PrototypeMemoryInteractionState)

    all_lanes = _agent(ia=True, partner=True).init(jr.key(92))
    assert isinstance(all_lanes.ia_state, PrototypeMemoryInteractionState)
    assert isinstance(all_lanes.ia_state.interaction_state, PrototypeInteractionState)
    assert all_lanes.ia_state.interaction_state.ia_state is not None


def test_query_before_write_and_retrieval_change_the_next_real_action() -> None:
    agent = PrototypeAgent(
        PrototypeAgentConfig(oak=_oak(), experiential_memory=_memory())
    )
    state = agent.start(agent.init(jr.key(2)), jnp.zeros((2,), dtype=jnp.float32))
    state = _force_action(agent, state, 1)

    first = agent.update_transition(
        state,
        _transition(state, reward=-1.0),
        experiential_memory_input=_sidecar(state, 101),
    )
    first_diag = first.experiential_memory_diagnostics
    assert first_diag is not None
    assert bool(first.transition_diagnostics.valid)
    assert bool(first_diag.transaction_applied)
    assert not bool(first_diag.proposal.available)
    assert not bool(first_diag.proposal.retrieval.has_neighbors)
    wrapper = cast(PrototypeMemoryInteractionState, first.state.ia_state)
    assert int(wrapper.experiential_memory_state.active_count) == 1
    slot = int(first_diag.slot)
    np.testing.assert_array_equal(
        wrapper.experiential_memory_state.entries.actions[slot],
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    )

    second = agent.update_transition(
        first.state,
        _transition(first.state),
        experiential_memory_input=_sidecar(first.state, 102),
    )
    second_diag = second.experiential_memory_diagnostics
    assert second_diag is not None
    assert bool(second.transition_diagnostics.valid)
    assert bool(second_diag.proposal.available)
    assert int(second_diag.proposal.action) == 1
    assert int(second_diag.counterfactual_base_action) == 0
    assert int(second.action) == 1
    assert int(second.state.current_action) == 1
    assert int(second.state.oak_state.stomp_state.last_primitive_action) == 1
    assert int(second_diag.proposal.retrieval.neighbor_provenance_ids[0]) == 101


def test_autoreset_writes_final_bootstrap_but_queries_reset_decision() -> None:
    agent = _agent()
    initial = agent.start(
        agent.init(jr.key(3)),
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    )
    transition = PrototypeTransition(
        observation=initial.current_raw_observation,
        action=initial.current_action,
        decision_id=initial.current_decision_id,
        reward=jnp.asarray(2.0, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=jnp.asarray([9.0, 8.0], dtype=jnp.float32),
        next_decision_observation=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
    )
    result = agent.update_transition(
        initial,
        transition,
        experiential_memory_input=_sidecar(initial, 201),
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    slot = int(diagnostics.slot)
    memory = _memory_state(result.state)
    np.testing.assert_array_equal(
        memory.entries.observations[slot],
        initial.current_representation,
    )
    np.testing.assert_array_equal(
        memory.entries.keys[slot],
        initial.current_representation,
    )
    np.testing.assert_array_equal(
        memory.entries.outcomes[slot],
        jnp.asarray([9.0, 8.0, 2.0], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.state.current_representation,
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
    )
    assert int(memory.entries.ages[slot]) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "stale-current-id",
        "stale-next-id",
        "nonfinite-reliability",
        "invalid-version",
    ],
)
def test_missing_or_dynamic_invalid_sidecar_is_exact_memory_noop(
    mutation: str,
) -> None:
    agent = _agent()
    initial = agent.start(agent.init(jr.key(4)), jnp.zeros((2,), dtype=jnp.float32))
    sidecar: PrototypeExperientialMemoryInput | None = _sidecar(initial, 301)
    if mutation == "missing":
        sidecar = None
    elif mutation == "stale-current-id":
        sidecar = sidecar.replace(
            current_prototype_decision_id=jnp.zeros((4,), dtype=jnp.uint32)
        )
    elif mutation == "stale-next-id":
        sidecar = sidecar.replace(
            next_prototype_decision_id=initial.current_decision_id
        )
    elif mutation == "nonfinite-reliability":
        sidecar = sidecar.replace(
            reliability=jnp.asarray(jnp.nan, dtype=jnp.float32)
        )
    else:
        sidecar = sidecar.replace(
            query_representation_version=jnp.asarray(-1, dtype=jnp.int32)
        )
    before_memory = _memory_state(initial)
    result = agent.update_transition(
        initial,
        _transition(initial),
        experiential_memory_input=sidecar,
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert not bool(diagnostics.transaction_required)
    assert not bool(diagnostics.transaction_applied)
    assert not bool(diagnostics.wrote)
    _assert_tree_equal(_memory_state(result.state), before_memory)
    assert int(result.action) == int(diagnostics.counterfactual_base_action)


def test_full_lifecycle_replay_is_rejected_without_memory_effect() -> None:
    agent = _agent()
    old = agent.start(
        agent.init(jr.key(5), lifecycle_id=jnp.asarray([1, 2], dtype=jnp.uint32)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    replayed = _sidecar(old, 401)
    fresh = agent.start(
        agent.init(jr.key(6), lifecycle_id=jnp.asarray([3, 4], dtype=jnp.uint32)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    result = agent.update_transition(
        fresh,
        _transition(fresh),
        experiential_memory_input=replayed,
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert not bool(diagnostics.current_prototype_decision_id_matches)
    assert not bool(diagnostics.next_prototype_decision_id_matches)
    assert not bool(diagnostics.transaction_required)
    _assert_tree_equal(_memory_state(result.state), _memory_state(fresh))


def test_malformed_memory_sidecar_shapes_and_dtypes_raise_before_tracing() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(7)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    transition = _transition(state)
    valid = _sidecar(state, 402)
    with pytest.raises(ValueError, match="next_action_safety_mask"):
        agent.update_transition(
            state,
            transition,
            experiential_memory_input=valid.replace(
                next_action_safety_mask=jnp.ones((3,), dtype=jnp.bool_)
            ),
        )
    with pytest.raises(ValueError, match="current_prototype_decision_id"):
        agent.update_transition(
            state,
            transition,
            experiential_memory_input=valid.replace(
                current_prototype_decision_id=jnp.zeros(
                    (3,),
                    dtype=jnp.uint32,
                )
            ),
        )
    with pytest.raises(ValueError, match="query_uncertainty"):
        agent.update_transition(
            state,
            transition,
            experiential_memory_input=valid.replace(
                query_uncertainty=jnp.asarray(0, dtype=jnp.int32)
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "version",
        "query-unavailable",
        "query-uncertain",
        "unsafe-exemplar",
        "stale",
        "masked-action",
    ],
)
def test_retrieval_abstentions_keep_the_ordinary_action_but_still_write(
    case: str,
) -> None:
    agent = _agent()
    seeded, base_action, memory_action = _seed_opposite_memory_action(
        agent,
        key=20,
        safety_cost=2.0 if case == "unsafe-exemplar" else 0.0,
    )
    state = seeded.state
    if case == "stale":
        memory = _memory_state(state)
        current_words = jnp.broadcast_to(
            memory.step_words,
            memory.entries.insertion_step_words.shape,
        )
        stale_entries = memory.entries.replace(
            ages=jnp.where(memory.entries.valid, 101, memory.entries.ages),
            recency_ages=jnp.where(
                memory.entries.valid,
                0,
                memory.entries.recency_ages,
            ),
            insertion_step_words=jnp.where(
                memory.entries.valid[:, None],
                current_words,
                memory.entries.insertion_step_words,
            ),
            last_access_step_words=jnp.where(
                memory.entries.valid[:, None],
                current_words,
                memory.entries.last_access_step_words,
            ),
            insertion_age_offsets=jnp.where(
                memory.entries.valid,
                101,
                memory.entries.insertion_age_offsets,
            ),
            last_access_age_offsets=jnp.where(
                memory.entries.valid,
                0,
                memory.entries.last_access_age_offsets,
            ),
        )
        state = state.replace(
            ia_state=state.ia_state.replace(
                experiential_memory_state=memory.replace(entries=stale_entries)
            )
        )
    mask = [True, True]
    if case == "masked-action":
        mask[memory_action] = False
    sidecar = _sidecar(
        state,
        501,
        mask=cast(tuple[bool, bool], tuple(mask)),
    )
    if case == "version":
        sidecar = sidecar.replace(
            query_representation_version=jnp.asarray(2, dtype=jnp.int32),
            entry_representation_version=jnp.asarray(2, dtype=jnp.int32),
        )
    elif case == "query-unavailable":
        sidecar = sidecar.replace(
            query_uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
            query_uncertainty_available=jnp.asarray(False, dtype=jnp.bool_),
        )
    elif case == "query-uncertain":
        sidecar = sidecar.replace(
            query_uncertainty=jnp.asarray(2.0, dtype=jnp.float32)
        )

    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=sidecar,
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(diagnostics.transaction_required)
    assert bool(diagnostics.transaction_applied)
    assert bool(diagnostics.wrote)
    assert not bool(diagnostics.proposal.available)
    assert int(result.action) == int(diagnostics.counterfactual_base_action)
    assert int(result.action) == base_action
    if case == "version":
        assert not bool(diagnostics.proposal.retrieval.version_compatible)
    elif case == "stale":
        assert not bool(diagnostics.proposal.retrieval.freshness_ok)
    elif case == "unsafe-exemplar":
        assert not bool(diagnostics.proposal.retrieval.safety_ok)
    elif case.startswith("query-"):
        assert not bool(diagnostics.proposal.retrieval.uncertainty_ok)
    else:
        assert bool(diagnostics.proposal.retrieval.accepted)
        assert not bool(diagnostics.proposal.safe_positive_mass_available)


def test_partner_runs_after_memory_and_audits_memory_as_counterfactual() -> None:
    agent = _agent(partner=True)
    seeded, base_action, memory_action = _seed_opposite_memory_action(
        agent,
        key=30,
    )
    state = seeded.state
    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_sidecar(state, 601),
        partner_policy_fusion_input=_partner_input(
            agent,
            state,
            suggested_action=base_action,
        ),
    )
    memory_diagnostics = result.experiential_memory_diagnostics
    partner_diagnostics = result.partner_policy_fusion_diagnostics
    assert memory_diagnostics is not None
    assert partner_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert int(memory_diagnostics.counterfactual_base_action) == base_action
    assert int(memory_diagnostics.effective_action) == memory_action
    assert int(partner_diagnostics.counterfactual_base_action) == memory_action
    assert int(partner_diagnostics.decision.route) == ROUTE_ACCEPT
    assert int(result.action) == base_action
    wrapper = cast(PrototypeMemoryInteractionState, result.state.ia_state)
    assert isinstance(wrapper.interaction_state, PrototypeInteractionState)
    assert int(wrapper.experiential_memory_state.write_count) == 2


def test_partner_dispatch_respects_the_intersection_of_both_safety_masks() -> None:
    agent = _agent(partner=True)
    seeded, base_action, memory_action = _seed_opposite_memory_action(
        agent,
        key=33,
    )
    state = seeded.state
    memory_mask = [True, True]
    memory_mask[memory_action] = False
    result = agent.update_transition(
        state,
        _transition(state),
        experiential_memory_input=_sidecar(
            state,
            602,
            mask=cast(tuple[bool, bool], tuple(memory_mask)),
        ),
        partner_policy_fusion_input=_partner_input(
            agent,
            state,
            suggested_action=memory_action,
        ),
    )
    memory_diagnostics = result.experiential_memory_diagnostics
    partner_diagnostics = result.partner_policy_fusion_diagnostics
    assert memory_diagnostics is not None
    assert partner_diagnostics is not None
    assert bool(result.transition_diagnostics.valid)
    assert bool(memory_diagnostics.proposal.retrieval.accepted)
    assert not bool(memory_diagnostics.proposal.available)
    np.testing.assert_array_equal(
        partner_diagnostics.decision.shield.caller_action_mask,
        jnp.asarray(memory_mask, dtype=jnp.bool_),
    )
    assert not bool(
        partner_diagnostics.decision.shield.caller_action_mask[memory_action]
    )
    assert int(result.action) == base_action


def test_memory_coexists_with_ia_and_recurrent_cache_owns_final_action() -> None:
    ia_agent = _agent(ia=True)
    ia_state = ia_agent.start(
        ia_agent.init(jr.key(31)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    ia_result = ia_agent.update_transition(
        ia_state,
        _transition(ia_state),
        experiential_memory_input=_sidecar(ia_state, 701),
    )
    ia_wrapper = cast(PrototypeMemoryInteractionState, ia_result.state.ia_state)
    assert ia_wrapper.interaction_state is not None
    assert ia_result.ia_augmented_obs is not None
    assert int(ia_wrapper.experiential_memory_state.write_count) == 1

    recurrent_agent = _agent(recurrent=True)
    seeded, _, memory_action = _seed_opposite_memory_action(
        recurrent_agent,
        key=32,
    )
    result = recurrent_agent.update_transition(
        seeded.state,
        _transition(seeded.state),
        experiential_memory_input=_sidecar(seeded.state, 702),
    )
    diagnostics = result.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(diagnostics.proposal.available)
    recurrent = cast(
        PrototypeRecurrentLatentWorldModelState,
        result.state.world_model_state,
    )
    assert int(recurrent.decision_cache.action) == int(result.action)
    assert int(result.action) == memory_action
    chex.assert_trees_all_equal(
        recurrent.decision_cache.observation,
        result.state.current_representation,
    )
    assert bool(recurrent_agent._checkpoint_state_valid(result.state))


def test_eager_jit_scan_checkpoint_and_resource_accounting(tmp_path: Path) -> None:
    agent = _agent()
    initial = agent.start(
        agent.init(jr.key(40)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    transition = _transition(initial)
    sidecar = _sidecar(initial, 801)
    eager = agent.update_transition(
        initial,
        transition,
        experiential_memory_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        initial,
        transition,
        experiential_memory_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))

    transitions = jax.tree.map(lambda value: value[None], transition)
    sidecars = jax.tree.map(lambda value: value[None], sidecar)
    scanned = jax.jit(agent.scan_transitions)(
        initial,
        transitions,
        experiential_memory_input=sidecars,
    )
    _assert_tree_equal(_materialize_keys(scanned.state), _materialize_keys(eager.state))

    checkpoint = tmp_path / "prototype-memory"
    save_prototype_checkpoint(agent, eager.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    _assert_tree_equal(
        _materialize_keys(restored_state),
        _materialize_keys(eager.state),
    )
    curated_agent, curated_state = agent.curate(
        eager.state,
        jr.key(41),
        available_feature_indices=[1],
    )
    assert curated_agent.config.experiential_memory == agent.config.experiential_memory
    _assert_tree_equal(
        _memory_state(curated_state),
        _memory_state(eager.state),
    )

    memory = agent.experiential_memory
    policy = agent.experiential_memory_policy
    assert memory is not None and policy is not None
    wrapper = cast(PrototypeMemoryInteractionState, eager.state.ia_state)
    actual_bytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree.leaves(wrapper)
    )
    assert actual_bytes == memory.persistent_bytes
    assert int(wrapper.experiential_memory_state.query_count) == 1
    assert int(wrapper.experiential_memory_state.write_count) == 1
    resources = policy.resource_declaration()
    assert resources.external_memory_persistent_state_bytes == actual_bytes
    assert resources.owned_persistent_state_bytes == 0
    assert resources.random_draws_per_proposal == 0
    prototype_resources = agent.experiential_memory_resource_declaration
    assert prototype_resources is not None
    assert prototype_resources.persistent_state_bytes == actual_bytes
    assert prototype_resources.categorical_policy_queries == 1
    assert prototype_resources.causal_step_queries == 1
    assert prototype_resources.total_deterministic_prestate_queries == 2
    assert prototype_resources.writes_attempted == 1
    assert prototype_resources.random_draws == 0
    diagnostics = eager.experiential_memory_diagnostics
    assert diagnostics is not None
    assert int(diagnostics.deterministic_prestate_query_count) == 2
    assert bool(diagnostics.query_before_write)


def test_opt_in_advantage_gate_blocks_unsupported_override_and_roundtrips(
    tmp_path: Path,
) -> None:
    gate_config = ExperientialMemoryAdvantageGateConfig(
        min_action_support=1,
        min_reward_advantage=0.0,
    )
    agent = _agent(advantage_gate=gate_config)
    seeded, base_action, memory_action = _seed_opposite_memory_action(
        agent,
        key=42,
    )
    transition = _transition(seeded.state)
    sidecar = _sidecar(seeded.state, 802)

    eager = agent.update_transition(
        seeded.state,
        transition,
        experiential_memory_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        seeded.state,
        transition,
        experiential_memory_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))
    diagnostics = eager.experiential_memory_diagnostics
    assert diagnostics is not None
    advantage = diagnostics.advantage_gate
    assert advantage is not None
    assert bool(diagnostics.proposal.available)
    assert int(diagnostics.proposal.action) == memory_action
    assert int(diagnostics.counterfactual_base_action) == base_action
    assert bool(advantage.evidence_valid)
    assert int(advantage.base_support_count) == 0
    assert not bool(advantage.support_ready)
    assert not bool(advantage.replacement_allowed)
    assert not bool(diagnostics.dispatch_replacement.applied)
    assert int(eager.action) == base_action

    transitions = jax.tree.map(lambda value: value[None], transition)
    sidecars = jax.tree.map(lambda value: value[None], sidecar)
    scanned = jax.jit(agent.scan_transitions)(
        seeded.state,
        transitions,
        experiential_memory_input=sidecars,
    )
    _assert_tree_equal(
        _materialize_keys(scanned.state),
        _materialize_keys(eager.state),
    )

    checkpoint = tmp_path / "prototype-memory-advantage-gate"
    save_prototype_checkpoint(agent, eager.state, checkpoint)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint)
    assert restored_agent.to_config() == agent.to_config()
    assert restored_agent.config.experiential_memory_advantage_gate == gate_config
    assert restored_agent.experiential_memory_advantage_gate is not None
    _assert_tree_equal(
        _materialize_keys(restored_state),
        _materialize_keys(eager.state),
    )


def test_fixed_capacity_eviction_and_counter_exhaustion_are_bounded() -> None:
    agent = _agent(memory_overrides={"capacity": 2, "top_k": 2})
    state = agent.start(
        agent.init(jr.key(50)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    for index in range(3):
        result = agent.update_transition(
            state,
            _transition(state),
            experiential_memory_input=_sidecar(state, 900 + index),
        )
        assert bool(result.transition_diagnostics.valid)
        state = result.state
    memory = _memory_state(state)
    assert int(memory.active_count) == 2
    assert int(memory.write_count) == 3
    assert int(memory.eviction_count) == 1
    assert int(jnp.sum(memory.entries.valid.astype(jnp.int32))) == 2

    maximum = jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32)
    near_maximum_words = jnp.asarray(
        (np.iinfo(np.uint32).max, np.iinfo(np.uint32).max - 3),
        dtype=jnp.uint32,
    )
    near_maximum_observation_words = jnp.asarray(
        (np.iinfo(np.uint32).max, np.iinfo(np.uint32).max - 2),
        dtype=jnp.uint32,
    )
    exhausted = state.replace(
        step_count=maximum,
        step_words=near_maximum_words,
        observation_event_count=maximum,
        observation_event_words=near_maximum_observation_words,
        oak_state=state.oak_state.replace(
            step_count=maximum,
            step_words=near_maximum_words,
            stomp_state=state.oak_state.stomp_state.replace(
                step_count=maximum,
                step_words=near_maximum_words,
            ),
        ),
    )
    before_memory = _memory_state(exhausted)
    disarmed = agent.update_transition(
        exhausted,
        _transition(exhausted),
        experiential_memory_input=_sidecar(exhausted, 999),
    )
    diagnostics = disarmed.experiential_memory_diagnostics
    assert diagnostics is not None
    assert bool(disarmed.transition_diagnostics.valid)
    assert not bool(disarmed.state.started)
    assert int(disarmed.action) == -1
    assert not bool(diagnostics.transaction_required)
    _assert_tree_equal(_memory_state(disarmed.state), before_memory)


def test_unsafe_base_and_corrupt_memory_roll_back_the_whole_transition() -> None:
    agent = _agent()
    initial = agent.start(
        agent.init(jr.key(60)),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    transition = _transition(initial)
    ordinary = agent.update_transition(initial, transition)
    base_action = int(ordinary.action)
    unsafe_mask = [True, True]
    unsafe_mask[base_action] = False
    sidecar = _sidecar(
        initial,
        1001,
        mask=cast(tuple[bool, bool], tuple(unsafe_mask)),
    )
    eager = agent.update_transition(
        initial,
        transition,
        experiential_memory_input=sidecar,
    )
    compiled = jax.jit(agent.update_transition)(
        initial,
        transition,
        experiential_memory_input=sidecar,
    )
    _assert_tree_equal(_materialize_keys(eager), _materialize_keys(compiled))
    diagnostics = eager.experiential_memory_diagnostics
    assert diagnostics is not None
    assert not bool(eager.transition_diagnostics.valid)
    assert bool(diagnostics.transaction_required)
    assert bool(diagnostics.wrote)
    assert bool(diagnostics.dispatch_replacement.failed_closed)
    assert not bool(diagnostics.transaction_applied)
    _assert_tree_equal(_materialize_keys(eager.state), _materialize_keys(initial))

    memory = _memory_state(initial)
    corrupted_memory = memory.replace(
        entries=memory.entries.replace(
            actions=memory.entries.actions.at[0, 0].set(jnp.nan)
        )
    )
    corrupted = initial.replace(
        ia_state=initial.ia_state.replace(
            experiential_memory_state=corrupted_memory
        )
    )
    rejected = agent.update_transition(
        corrupted,
        _transition(corrupted),
        experiential_memory_input=_sidecar(corrupted, 1002),
    )
    assert not bool(rejected.transition_diagnostics.state_consistent)
    assert not bool(rejected.transition_diagnostics.valid)
    _assert_tree_equal(
        _materialize_keys(rejected.state),
        _materialize_keys(corrupted),
    )
