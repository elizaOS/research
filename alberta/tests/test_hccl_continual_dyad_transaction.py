# mypy: disable-error-code="attr-defined,call-arg,no-any-return,arg-type,type-var,union-attr"
"""Behavioral contract for the smallest atomic HCCL continual dyad."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_context_lineage_retention_seam import CONFIG as CONTEXT_CONFIG
from test_external_learned_state_live_memory_action_stack_adapter import (
    _adapter as _action_stack_donor,
)
from test_external_learned_state_live_memory_adapter import (
    _tree_exact,
)
from test_external_learned_state_router_audit_coordinator import (
    _coordinator as _coordinator_donor,
)
from test_hccl_world_attribution_adapter import _adapter as _hccl_donor
from test_prototype_feature_lifecycle import _prefix_pair_config

from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackConfig,
)
from alberta_framework.core.hccl_causal_attribution import HCCLActionLayer
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA,
    HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA,
    HCCL_CONTINUAL_DYAD_STATE_SCHEMA,
    HCCL_CONTINUAL_DYAD_STATUS,
    HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA,
    HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA,
    HCCLContinualDyadPreparationReceipt,
    HCCLContinualDyadPreparedTransaction,
    HCCLContinualDyadResult,
    HCCLContinualDyadState,
    HCCLContinualDyadThroughMemoryAgent,
    HCCLContinualDyadThroughMemoryTransaction,
    HCCLContinualDyadThroughMemoryWork,
    HCCLContinualDyadTransaction,
    HCCLContinualDyadTransactionConfig,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.option_search_control import OptionSearchControlConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerPlannerConfig,
)
from alberta_framework.core.state_builder import (
    IdentityStateBuilderConfig,
    LearnableGRUStateBuilderConfig,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.streams.hccl_causal_core import HCCLCausalCoreProposal
from alberta_framework.streams.hccl_causal_core import _proposal_tag as _world_proposal_tag

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_AGENTS = 2
N_ACTIONS = 2
PHYSICAL_DIM = 16
CONTEXT_DIM = 3
FAST_DIM = 4
RAW_DIM = PHYSICAL_DIM + CONTEXT_DIM
BASE_DIM = RAW_DIM + FAST_DIM
ACTIVE_PAIR_SLOTS = 12
PAIR_CANDIDATES = 120
ROUTED_DIM = BASE_DIM + ACTIVE_PAIR_SLOTS
FEATURE_REPLACEMENT_INTERVAL = 64
MEMORY_CAPACITY = 64
CORE_L1_EVENTS = 8_998
HORDE_NAMES = (
    "task_discount_0p5",
    "task_discount_0p9",
    "task_discount_0p99",
    "partner_action",
    "safety_cost",
    "tv_occupancy",
    "target_zone_occupancy",
    "option_success_unavailable",
)
HORDE_GAMMAS = (0.5, 0.9, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9)
AGENT_1_OWNER = (
    0xC101C101,
    0xC202C202,
    0xC303C303,
    0xC404C404,
    0xC505C505,
    0xC606C606,
    0xC707C707,
    0xC808C808,
)
BINDING_OWNER = (
    0xD101D101,
    0xD202D202,
    0xD303D303,
    0xD404D404,
    0xD505D505,
    0xD606D606,
    0xD707D707,
    0xD808D808,
)
MASKS = jnp.ones((N_AGENTS, N_ACTIONS), dtype=jnp.bool_)


@dataclasses.dataclass(frozen=True, slots=True)
class _Rig:
    dyad: HCCLContinualDyadTransaction
    state: HCCLContinualDyadState
    seed: int


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedRig:
    rig: _Rig
    event: object
    binding: object
    prepared: HCCLContinualDyadPreparedTransaction
    receipt: HCCLContinualDyadPreparationReceipt
    result: HCCLContinualDyadResult
    preparation_planner_pair_authentication_calls: int
    receipt_planner_pair_authentication_calls: int
    adoption_planner_pair_authentication_calls: int


@dataclasses.dataclass(frozen=True, slots=True)
class _SplitRig:
    prepared_rig: _PreparedRig
    through_memory: HCCLContinualDyadThroughMemoryTransaction
    completed: HCCLContinualDyadPreparedTransaction
    through_planner_pair_authentication_calls: int
    completion_planner_pair_authentication_calls: int


@pytest.fixture(autouse=True)
def _bounded_jax_execution() -> Iterator[None]:
    with jax.disable_jit():
        yield


def _dyad(
    *,
    planning_enabled: bool = True,
    binding_owner_digest: tuple[int, ...] = BINDING_OWNER,
) -> HCCLContinualDyadTransaction:
    """Resize the exact donor configurations to the integrated 19/23/35 geometry."""

    coordinator_donor = _coordinator_donor().config
    lifecycle = dataclasses.replace(
        _prefix_pair_config(replacement_interval=FEATURE_REPLACEMENT_INTERVAL),
        n_tasks=1 + len(HORDE_NAMES),
        n_options=0,
        option_subtask_feature_indices=(),
        managed_horde_demons=len(HORDE_NAMES),
        max_observations=CORE_L1_EVENTS,
    )
    horde = create_horde_spec(
        tuple(
            GVFSpec(
                name=name,
                demon_type=DemonType.PREDICTION,
                gamma=gamma,
                lamda=0.0,
                cumulant_index=index,
            )
            for index, (name, gamma) in enumerate(
                zip(HORDE_NAMES, HORDE_GAMMAS, strict=True)
            )
        )
    )
    oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(),
            observation_dim=ROUTED_DIM,
            n_primitive_actions=N_ACTIONS,
            base_hidden_sizes=(),
            base_step_size=0.01,
            epsilon_base=0.0,
            option_planning_backups_per_step=0,
        )
    )
    prototype = dataclasses.replace(
        coordinator_donor.inner.prototype,
        oak=oak,
        horde_spec=horde,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        experiential_memory=None,
        state_builder=IdentityStateBuilderConfig(observation_dim=BASE_DIM),
        prototype_feature_lifecycle=lifecycle,
        option_search_control=None,
        auto_curate_every=0,
    )
    ensemble_donor = coordinator_donor.inner.ensemble
    ensemble = dataclasses.replace(
        ensemble_donor,
        router=dataclasses.replace(
            ensemble_donor.router,
            base_dim=BASE_DIM,
            active_slots=ACTIVE_PAIR_SLOTS,
        ),
        world_model=dataclasses.replace(
            ensemble_donor.world_model,
            observation_dim=BASE_DIM,
        ),
        signal_estimator=dataclasses.replace(
            ensemble_donor.signal_estimator,
            ensemble_size=1,
            target_dim=BASE_DIM + 2,
        ),
        ensemble_size=1,
        max_events=CORE_L1_EVENTS,
    )
    coordinator = dataclasses.replace(
        coordinator_donor,
        builder=LearnableGRUStateBuilderConfig(
            observation_dim=RAW_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=FAST_DIM,
            step_size=0.01,
            gradient_clip=10.0,
            initialization_scale=0.2,
            include_raw_observation=True,
        ),
        inner=dataclasses.replace(
            coordinator_donor.inner,
            prototype=prototype,
            ensemble=ensemble,
        ),
        learning_value_router=dataclasses.replace(
            coordinator_donor.learning_value_router,
            max_steps=CORE_L1_EVENTS,
        ),
        max_events=CORE_L1_EVENTS,
    )

    action_stack_donor = _action_stack_donor().config
    memory = dataclasses.replace(
        action_stack_donor.learned_memory.memory,
        capacity=MEMORY_CAPACITY,
        observation_dim=RAW_DIM,
        key_dim=RAW_DIM,
        action_dim=N_ACTIONS,
        outcome_dim=RAW_DIM,
        max_age=CORE_L1_EVENTS,
        staleness_scale=float(CORE_L1_EVENTS),
    )
    learned_memory = dataclasses.replace(
        action_stack_donor.learned_memory,
        memory=memory,
    )
    agent_0 = ExternalLearnedStateLiveMemoryActionStackConfig(
        coordinator=coordinator,
        learned_memory=learned_memory,
        final_action_owner_digest=action_stack_donor.final_action_owner_digest,
    )
    agent_1 = ExternalLearnedStateLiveMemoryActionStackConfig(
        coordinator=coordinator,
        learned_memory=learned_memory,
        final_action_owner_digest=AGENT_1_OWNER,
    )
    return HCCLContinualDyadTransaction(
        HCCLContinualDyadTransactionConfig(
            hccl=_hccl_donor().config,
            agent_0=agent_0,
            agent_1=agent_1,
            planner=PrototypeFactorizedPartnerPlannerConfig(
                observation_dim=BASE_DIM,
                prototype_representation_dim=ROUTED_DIM,
                n_actions=N_ACTIONS,
                grounded_initialization_scale=0.25,
                planning_enabled=planning_enabled,
            ),
            context=CONTEXT_CONFIG,
            binding_owner_digest=binding_owner_digest,
        )
    )


def _has_bmp_divergence(state: HCCLContinualDyadState) -> bool:
    return any(
        int(agent.action_binding.memory_action) != int(agent.action_binding.base_action)
        or int(agent.action_binding.final_action) != int(agent.action_binding.memory_action)
        for agent in (state.agent_0_state, state.agent_1_state)
    )


def _assert_primitive_only_state(
    dyad: HCCLContinualDyadTransaction,
    state: HCCLContinualDyadState,
) -> None:
    for index, (adapter, agent_state) in enumerate(
        zip(
            (dyad.agent_0, dyad.agent_1),
            (state.agent_0_state, state.agent_1_state),
            strict=True,
        )
    ):
        prototype_impl = adapter.coordinator.inner.prototype
        prototype_state = (
            agent_state.coordinator_state.inner_state.prototype_state
        )
        oak = prototype_impl._oak_component_state(prototype_state.oak_state)
        stomp = oak.stomp_state
        learner = stomp.base_learner_state

        assert oak.execution_counts.shape == (0,), index
        assert oak.cumulative_pseudo_rewards.shape == (0,), index
        assert oak.utility_ema.shape == (0,), index
        assert stomp.option_policies.q_weights.shape[0] == 0, index
        assert stomp.option_policies.traces.shape[0] == 0, index
        assert stomp.option_policies.average_rewards.shape == (0,), index
        assert stomp.option_models.n_completions.shape == (0,), index
        assert int(stomp.executing_option) == -1, index
        assert int(stomp.option_steps) == 0, index
        assert 0 <= int(stomp.base_last_action) < N_ACTIONS, index
        assert 0 <= int(stomp.last_primitive_action) < N_ACTIONS, index
        assert len(learner.head_params.weights) == N_ACTIONS, index
        assert len(learner.head_params.biases) == N_ACTIONS, index
        assert len(learner.head_optimizer_states) == N_ACTIONS, index
        assert len(learner.head_traces) == N_ACTIONS, index


@pytest.fixture(scope="module")
def rig() -> _Rig:
    """Select a bounded deterministic seed that exercises an actual B/M/P split."""

    with jax.disable_jit():
        dyad = _dyad()
        for seed in range(701, 709):
            state = dyad.init(jr.key(seed))
            if _has_bmp_divergence(state):
                return _Rig(dyad=dyad, state=state, seed=seed)
    raise AssertionError("eight deterministic planner initializations produced no B/M/P split")


@pytest.fixture(scope="module")
def prepared_rig(rig: _Rig) -> _PreparedRig:
    with jax.disable_jit():
        event = rig.dyad.prepare_event(rig.state)
        binding = rig.dyad.bind_current_actions(rig.state, event)
        memory_inputs = rig.dyad.causal_core_memory_event_inputs(rig.state, event)
        planner_type = type(rig.dyad.planner)
        original_authenticate = planner_type.authenticate_pair
        pair_calls = 0

        def counted_authenticate(*args: object, **kwargs: object) -> object:
            nonlocal pair_calls
            pair_calls += 1
            return original_authenticate(*args, **kwargs)

        patcher = pytest.MonkeyPatch()
        patcher.setattr(planner_type, "authenticate_pair", counted_authenticate)
        try:
            prepared = rig.dyad.prepare_transaction(
                rig.state,
                event,
                binding,
                memory_inputs[0],
                memory_inputs[1],
                MASKS,
            )
            preparation_calls = pair_calls
            pair_calls = 0
            receipt = rig.dyad.integrity_receipt(prepared)
            receipt_calls = pair_calls
            pair_calls = 0
            result = rig.dyad.adopt_prepared_transaction(rig.state, prepared, receipt)
            adoption_calls = pair_calls
        finally:
            patcher.undo()
    return _PreparedRig(
        rig=rig,
        event=event,
        binding=binding,
        prepared=prepared,
        receipt=receipt,
        result=result,
        preparation_planner_pair_authentication_calls=preparation_calls,
        receipt_planner_pair_authentication_calls=receipt_calls,
        adoption_planner_pair_authentication_calls=adoption_calls,
    )


@pytest.fixture(scope="module")
def split_rig(prepared_rig: _PreparedRig) -> _SplitRig:
    source = prepared_rig.rig.state
    dyad = prepared_rig.rig.dyad
    planner_type = type(dyad.planner)
    original_authenticate = planner_type.authenticate_pair
    pair_calls = 0

    def counted_authenticate(*args: object, **kwargs: object) -> object:
        nonlocal pair_calls
        pair_calls += 1
        return original_authenticate(*args, **kwargs)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(planner_type, "authenticate_pair", counted_authenticate)
    with jax.disable_jit():
        try:
            memory_inputs = dyad.causal_core_memory_event_inputs(
                source,
                prepared_rig.event,
            )
            through_memory = dyad.prepare_through_memory(
                source,
                prepared_rig.event,
                prepared_rig.binding,
                memory_inputs[0],
                memory_inputs[1],
                MASKS,
            )
            through_calls = pair_calls
            pair_calls = 0
            completed = dyad.complete_with_factorized_planner(source, through_memory)
            completion_calls = pair_calls
        finally:
            patcher.undo()
    return _SplitRig(
        prepared_rig=prepared_rig,
        through_memory=through_memory,
        completed=completed,
        through_planner_pair_authentication_calls=through_calls,
        completion_planner_pair_authentication_calls=completion_calls,
    )


def _assert_atomic_rollback(
    result: HCCLContinualDyadResult,
    source: HCCLContinualDyadState,
    *,
    child_adoptions_called: tuple[int, int] = (0, 0),
    child_reconstructions: tuple[int, int] | None = None,
    child_adoptions_valid: tuple[bool, bool] = (False, False),
    outer_child_recomputations: tuple[int, int] = (1, 1),
    planner_pair_authentication_calls: int = 5,
) -> None:
    if child_reconstructions is None:
        child_reconstructions = child_adoptions_called
    _tree_exact(result.state, source)
    assert not bool(result.update_applied)
    assert bool(result.complete_source_returned)
    assert not bool(result.hccl_owner_committed)
    assert not bool(result.planner_owner_committed)
    np.testing.assert_array_equal(result.action_stack_owners_committed, (False, False))
    np.testing.assert_array_equal(result.context_owners_committed, (False, False))
    np.testing.assert_array_equal(result.lineage_owners_committed, (False, False))
    np.testing.assert_array_equal(result.child_adoptions_valid, child_adoptions_valid)
    np.testing.assert_array_equal(
        result.adoption_work.action_stack_integrity_adoptions,
        child_adoptions_called,
    )
    np.testing.assert_array_equal(
        result.adoption_work.child_adoption_structural_recomputations,
        child_reconstructions,
    )
    np.testing.assert_array_equal(
        result.adoption_work.outer_child_finalization_structural_recomputations,
        outer_child_recomputations,
    )
    assert int(result.adoption_work.outer_commit_decisions) == 1
    assert int(result.adoption_work.outer_committed_pp_world_successors) == 0
    assert int(result.adoption_work.outer_discarded_world_proposals) == 8
    assert int(result.adoption_work.world_reevaluations) == 0
    assert int(result.adoption_work.planner_reevaluations) == 0
    assert (
        int(result.adoption_work.planner_validation_pair_authentication_calls)
        == planner_pair_authentication_calls
    )
    assert (
        int(
            result.adoption_work
            .planner_validation_agent_cache_authentication_evaluations
        )
        == 2 * planner_pair_authentication_calls
    )
    assert (
        int(
            result.adoption_work
            .planner_validation_behavior_probability_vector_evaluations
        )
        == 2 * planner_pair_authentication_calls
    )
    assert (
        int(
            result.adoption_work
            .planner_validation_grounded_joint_cell_prediction_equivalents
        )
        == 8 * planner_pair_authentication_calls
    )
    assert (
        int(
            result.adoption_work
            .planner_validation_expected_reward_marginalization_products
        )
        == 8 * planner_pair_authentication_calls
    )
    for name in (
        "context_reevaluations",
        "coordinator_reevaluations",
        "prototype_reevaluations",
        "learned_memory_reevaluations",
    ):
        np.testing.assert_array_equal(getattr(result.adoption_work, name), (0, 0))


def _observe_planner_pair_authentication_calls(
    monkeypatch: pytest.MonkeyPatch,
    dyad: HCCLContinualDyadTransaction,
    operation: Callable[[], HCCLContinualDyadResult],
) -> tuple[HCCLContinualDyadResult, int]:
    planner_type = type(dyad.planner)
    original = planner_type.authenticate_pair
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(planner_type, "authenticate_pair", counted)
        result = operation()
    return result, calls


def test_outer_schema_is_a_new_nonpromoting_host_only_surface(rig: _Rig) -> None:
    payload = rig.dyad.to_config()

    assert HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA == (
        "alberta.hccl-continual-dyad-transaction-config.v2"
    )
    assert HCCL_CONTINUAL_DYAD_STATUS == (
        "l0-development-hccl-continual-dyad-atomic-transaction"
    )
    assert dataclasses.is_dataclass(HCCLContinualDyadTransactionConfig)
    assert payload["schema"] == HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA
    assert payload["state_schema"] == HCCL_CONTINUAL_DYAD_STATE_SCHEMA
    assert (
        payload["through_memory_agent_schema"]
        == HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA
    )
    assert payload["through_memory_schema"] == HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA
    for name in (
        "schema",
        "state_schema",
        "binding_schema",
        "through_memory_agent_schema",
        "through_memory_schema",
        "prepared_schema",
        "receipt_schema",
        "resource_schema",
    ):
        assert str(payload[name]).endswith(".v2")
    assert payload["mechanism_status"] == HCCL_CONTINUAL_DYAD_STATUS
    assert payload["evidence_level"] == "L0"
    assert payload["base_order"] == ["physical16", "context3", "fast4"]
    assert payload["physical_observation_dim"] == PHYSICAL_DIM
    assert payload["context_coordinate_dim"] == CONTEXT_DIM
    assert payload["fast_state_dim"] == FAST_DIM
    assert payload["external_recurrent_input_dim"] == RAW_DIM
    assert payload["stable_base_dim"] == BASE_DIM
    assert payload["active_pair_slots"] == ACTIVE_PAIR_SLOTS
    assert payload["pair_candidate_slots"] == PAIR_CANDIDATES
    assert payload["routed_representation_dim"] == ROUTED_DIM
    assert payload["minimum_supported_core_events"] == CORE_L1_EVENTS
    assert payload["experiential_memory_rows_per_agent"] == MEMORY_CAPACITY
    assert payload["experiential_memory_minimum_age_horizon"] == CORE_L1_EVENTS
    assert (
        payload["experiential_memory_minimum_staleness_scale"]
        == float(CORE_L1_EVENTS)
    )
    assert payload["uncertainty_ensemble_members_per_agent"] == 1
    assert payload["installed_option_slots_per_agent"] == 0
    assert payload["causal_core_target_uncertainty_ensemble_members"] == 1
    assert payload["causal_core_target_installed_option_slots"] == 0
    limitations = payload["limitations"]
    assert type(limitations) is list
    assert "zero-option-causal-core-geometry-not-yet-supported" not in limitations
    for agent in (rig.dyad.config.agent_0, rig.dyad.config.agent_1):
        prototype = agent.coordinator.inner.prototype
        lifecycle = prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        assert lifecycle.n_options == 0
        assert lifecycle.option_subtask_feature_indices == ()
        assert prototype.oak.stomp.subtask_specs == ()
        assert prototype.oak.stomp.n_options == 0
        assert prototype.oak.stomp.n_total_actions == N_ACTIONS
        assert prototype.oak.stomp.option_planning_backups_per_step == 0
        assert prototype.option_search_control is None
        assert prototype.auto_curate_every == 0
    assert payload["horde_head_order"] == list(HORDE_NAMES)
    assert payload["completed_transition_action"] == "P"
    assert payload["memory_feedback"] == "baseline-context-own-direct-M-effect"
    assert payload["memory_query_uncertainty"] == "unavailable-positive-zero"
    assert payload["memory_entry_uncertainty"] == "unavailable-positive-zero"
    assert payload["memory_entry_safety"] == "PP-available-positive-zero"
    assert payload["memory_entry_reliability"] == "one"
    assert payload["memory_provenance_id"] == "2*source-event-index+agent-index"
    assert payload["memory_source_id"] == "agent-index"
    assert payload["preparation_persisted"] is False
    assert payload["composite_jit_supported"] is False
    for name in (
        "scientific_promotion_allowed",
        "caller_authenticated",
        "output_writes_authorized",
        "artifact_authorized",
        "evidence_authorized",
        "promotion_authorized",
    ):
        assert payload[name] is False
    restored = HCCLContinualDyadTransactionConfig.from_config(payload)
    assert restored.to_config() == payload
    legacy = dict(payload)
    legacy["schema"] = "alberta.hccl-continual-dyad-transaction-config.v1"
    with pytest.raises(ValueError, match="noncanonical or unsupported"):
        HCCLContinualDyadTransactionConfig.from_config(legacy)


def test_config_rejects_noncanonical_feature_cadence_and_memory_capacity() -> None:
    config = _dyad().config

    def with_lifecycle_interval(
        agent: ExternalLearnedStateLiveMemoryActionStackConfig,
        interval: int,
    ) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        inner = agent.coordinator.inner
        lifecycle = inner.prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        prototype = dataclasses.replace(
            inner.prototype,
            prototype_feature_lifecycle=dataclasses.replace(
                lifecycle,
                replacement_interval=interval,
            ),
        )
        coordinator = dataclasses.replace(
            agent.coordinator,
            inner=dataclasses.replace(inner, prototype=prototype),
        )
        return dataclasses.replace(agent, coordinator=coordinator)

    bad_interval_agents = tuple(
        with_lifecycle_interval(agent, 0)
        for agent in (config.agent_0, config.agent_1)
    )
    with pytest.raises(ValueError, match="replacement interval must equal 64"):
        dataclasses.replace(
            config,
            agent_0=bad_interval_agents[0],
            agent_1=bad_interval_agents[1],
        )

    def with_memory_capacity(
        agent: ExternalLearnedStateLiveMemoryActionStackConfig,
        capacity: int,
    ) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        learned = agent.learned_memory
        return dataclasses.replace(
            agent,
            learned_memory=dataclasses.replace(
                learned,
                memory=dataclasses.replace(learned.memory, capacity=capacity),
            ),
        )

    bad_capacity_agents = tuple(
        with_memory_capacity(agent, 4)
        for agent in (config.agent_0, config.agent_1)
    )
    with pytest.raises(ValueError, match="memory capacity must equal 64"):
        dataclasses.replace(
            config,
            agent_0=bad_capacity_agents[0],
            agent_1=bad_capacity_agents[1],
        )

    def with_one_coherent_option(
        agent: ExternalLearnedStateLiveMemoryActionStackConfig,
    ) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        inner = agent.coordinator.inner
        prototype = inner.prototype
        lifecycle = prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        positive = dataclasses.replace(
            prototype,
            oak=dataclasses.replace(
                prototype.oak,
                stomp=dataclasses.replace(
                    prototype.oak.stomp,
                    subtask_specs=(
                        SubtaskSpec(
                            feature_index=0,
                            threshold=1_000_000.0,
                            max_option_steps=8,
                        ),
                    ),
                ),
            ),
            prototype_feature_lifecycle=dataclasses.replace(
                lifecycle,
                n_options=1,
                option_subtask_feature_indices=(0,),
            ),
        )
        return dataclasses.replace(
            agent,
            coordinator=dataclasses.replace(
                agent.coordinator,
                inner=dataclasses.replace(inner, prototype=positive),
            ),
        )

    positive_option_agents = tuple(
        with_one_coherent_option(agent)
        for agent in (config.agent_0, config.agent_1)
    )
    with pytest.raises(ValueError, match="primitive-only zero-option geometry"):
        dataclasses.replace(
            config,
            agent_0=positive_option_agents[0],
            agent_1=positive_option_agents[1],
        )

    def with_option_search(
        agent: ExternalLearnedStateLiveMemoryActionStackConfig,
    ) -> ExternalLearnedStateLiveMemoryActionStackConfig:
        inner = agent.coordinator.inner
        prototype = dataclasses.replace(
            inner.prototype,
            option_search_control=OptionSearchControlConfig(),
        )
        return dataclasses.replace(
            agent,
            coordinator=dataclasses.replace(
                agent.coordinator,
                inner=dataclasses.replace(inner, prototype=prototype),
            ),
        )

    option_search_agents = tuple(
        with_option_search(agent)
        for agent in (config.agent_0, config.agent_1)
    )
    with pytest.raises(ValueError, match="must disable option search"):
        dataclasses.replace(
            config,
            agent_0=option_search_agents[0],
            agent_1=option_search_agents[1],
        )


def test_config_closes_every_nested_lifetime_over_the_complete_core_l1_life() -> None:
    config = _dyad().config

    for agent in (config.agent_0, config.agent_1):
        coordinator = agent.coordinator
        lifecycle = coordinator.inner.prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        assert lifecycle.max_observations >= CORE_L1_EVENTS
        assert coordinator.max_events >= CORE_L1_EVENTS
        assert coordinator.learning_value_router.max_steps >= CORE_L1_EVENTS
        assert coordinator.inner.ensemble.max_events >= CORE_L1_EVENTS
        assert coordinator.inner.ensemble.ensemble_size == 1
        assert agent.learned_memory.memory.max_age >= CORE_L1_EVENTS
        assert agent.learned_memory.memory.staleness_scale >= float(CORE_L1_EVENTS)

    agent = config.agent_0
    coordinator = agent.coordinator
    inner = coordinator.inner
    lifecycle = inner.prototype.prototype_feature_lifecycle
    assert lifecycle is not None
    short_lifecycle = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            inner=dataclasses.replace(
                inner,
                prototype=dataclasses.replace(
                    inner.prototype,
                    prototype_feature_lifecycle=dataclasses.replace(
                        lifecycle,
                        max_observations=CORE_L1_EVENTS - 1,
                    ),
                ),
            ),
        ),
    )
    short_router = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            learning_value_router=dataclasses.replace(
                coordinator.learning_value_router,
                max_steps=CORE_L1_EVENTS - 1,
            ),
            max_events=CORE_L1_EVENTS - 1,
        ),
    )
    short_coordinator = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            max_events=CORE_L1_EVENTS - 1,
        ),
    )
    short_ensemble = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            inner=dataclasses.replace(
                inner,
                ensemble=dataclasses.replace(
                    inner.ensemble,
                    max_events=CORE_L1_EVENTS - 1,
                ),
            ),
        ),
    )
    short_memory_age = dataclasses.replace(
        agent,
        learned_memory=dataclasses.replace(
            agent.learned_memory,
            memory=dataclasses.replace(
                agent.learned_memory.memory,
                max_age=CORE_L1_EVENTS - 1,
            ),
        ),
    )
    short_memory_staleness = dataclasses.replace(
        agent,
        learned_memory=dataclasses.replace(
            agent.learned_memory,
            memory=dataclasses.replace(
                agent.learned_memory.memory,
                staleness_scale=float(CORE_L1_EVENTS - 1),
            ),
        ),
    )
    oversized_ensemble = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            inner=dataclasses.replace(
                inner,
                ensemble=dataclasses.replace(
                    inner.ensemble,
                    signal_estimator=dataclasses.replace(
                        inner.ensemble.signal_estimator,
                        ensemble_size=2,
                    ),
                    ensemble_size=2,
                ),
            ),
        ),
    )
    for short, message in (
        (short_lifecycle, "feature lifetime"),
        (short_router, "learning router"),
        (short_coordinator, "coordinator lifetime"),
        (short_ensemble, "routed ensemble"),
        (short_memory_age, "memory age horizon"),
        (short_memory_staleness, "memory staleness horizon"),
        (oversized_ensemble, "uncertainty ensemble must contain exactly one member"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(config, agent_0=short)


def test_genesis_owns_the_exact_topology_dimensions_and_cross_owner_clocks(
    rig: _Rig,
) -> None:
    dyad = rig.dyad
    state = rig.state
    config = dyad.config
    agents = (state.agent_0_state, state.agent_1_state)
    contexts = (state.context_0_state, state.context_1_state)
    planner_agents = (state.planner_state.agent_0, state.planner_state.agent_1)
    physical = dyad.hccl.world.observe(state.hccl_state.world_state)

    assert tuple(field.name for field in dataclasses.fields(state)) == (
        "config_token",
        "content_token",
        "hccl_state",
        "agent_0_state",
        "agent_1_state",
        "planner_state",
        "context_0_state",
        "context_1_state",
    )
    assert bool(dyad.state_valid(state))
    _assert_primitive_only_state(dyad, state)
    assert physical.shape == (N_AGENTS, PHYSICAL_DIM)
    assert physical.dtype == jnp.float32
    assert len(
        {
            config.agent_0.final_action_owner_digest,
            config.agent_1.final_action_owner_digest,
            config.binding_owner_digest,
        }
    ) == 3

    for index, (agent, context, planner_agent) in enumerate(
        zip(agents, contexts, planner_agents, strict=True)
    ):
        coordinator = agent.coordinator_state
        prototype = coordinator.inner_state.prototype_state
        expected_raw = jnp.concatenate(
            (physical[index], dyad.context.context_coordinates(context))
        ).astype(jnp.float32)
        expected_base = jnp.concatenate(
            (expected_raw, coordinator.builder_state.hidden)
        ).astype(jnp.float32)
        binding = agent.action_binding

        assert coordinator.current_raw_observation.shape == (RAW_DIM,)
        assert coordinator.builder_state.hidden.shape == (FAST_DIM,)
        assert coordinator.current_representation.shape == (BASE_DIM,)
        assert prototype.current_raw_observation.shape == (BASE_DIM,)
        assert prototype.current_representation.shape == (ROUTED_DIM,)
        assert planner_agent.cache.world_input.shape == (BASE_DIM,)
        assert planner_agent.cache.prototype_representation.shape == (ROUTED_DIM,)
        np.testing.assert_array_equal(coordinator.current_raw_observation, expected_raw)
        np.testing.assert_array_equal(coordinator.current_representation, expected_base)
        np.testing.assert_array_equal(prototype.current_raw_observation, expected_base)
        np.testing.assert_array_equal(
            coordinator.event_words,
            state.hccl_state.world_state.step_words,
        )
        np.testing.assert_array_equal(
            prototype.step_words,
            state.hccl_state.world_state.step_words,
        )
        np.testing.assert_array_equal(
            planner_agent.behavior.step_words,
            state.hccl_state.world_state.step_words,
        )
        np.testing.assert_array_equal(
            planner_agent.grounded.update_words,
            state.hccl_state.world_state.step_words,
        )
        np.testing.assert_array_equal(
            context.context.step_words,
            state.hccl_state.world_state.step_words,
        )
        assert bool(binding.available)
        assert bool(binding.planner_bound)
        assert int(planner_agent.cache.base_action) == int(binding.memory_action)
        assert int(planner_agent.cache.effective_action) == int(binding.final_action)
        assert bool(planner_agent.cache.planner_consumed) == bool(binding.planner_consumed)
        np.testing.assert_array_equal(
            binding.final_action_owner_words,
            np.asarray(
                (
                    config.agent_0.final_action_owner_digest,
                    config.agent_1.final_action_owner_digest,
                )[index],
                dtype=np.uint32,
            ),
        )
        assert agent.learned_memory_state.memory.entries.observations.shape == (
            config.agent_0.learned_memory.memory.capacity,
            RAW_DIM,
        )

    np.testing.assert_array_equal(
        agents[0].action_binding.planner_candidate_words,
        agents[1].action_binding.planner_candidate_words,
    )
    assert bool(
        jnp.all(
            dyad.planner.authenticate_pair(
                state.planner_state,
                agents[0].coordinator_state.inner_state.prototype_state,
                agents[1].coordinator_state.inner_state.prototype_state,
            )
        )
    )
    token_tamper = state.replace(
        content_token=state.content_token.at[0].set(
            jnp.bitwise_xor(
                state.content_token[0],
                jnp.asarray(1, dtype=jnp.uint8),
            )
        )
    )
    assert not bool(dyad.state_valid(token_tamper))


def test_action_binding_preserves_six_distinct_bmp_identities_and_real_divergence(
    prepared_rig: _PreparedRig,
) -> None:
    rig = prepared_rig.rig
    binding = prepared_rig.binding
    agents = (rig.state.agent_0_state, rig.state.agent_1_state)

    assert rig.seed in range(701, 709)
    assert bool(rig.dyad.binding_valid(rig.state, prepared_rig.event, binding))
    np.testing.assert_array_equal(
        binding.base_actions,
        tuple(int(agent.action_binding.base_action) for agent in agents),
    )
    np.testing.assert_array_equal(
        binding.memory_actions,
        tuple(int(agent.action_binding.memory_action) for agent in agents),
    )
    np.testing.assert_array_equal(
        binding.final_actions,
        tuple(int(agent.action_binding.final_action) for agent in agents),
    )
    assert np.any(
        np.asarray(binding.memory_actions) != np.asarray(binding.base_actions)
    ) or np.any(
        np.asarray(binding.final_actions) != np.asarray(binding.memory_actions)
    )

    assert int(binding.base.layer) == int(HCCLActionLayer.BASE)
    assert int(binding.memory.layer) == int(HCCLActionLayer.MEMORY)
    assert int(binding.planner.layer) == int(HCCLActionLayer.PLANNER)
    np.testing.assert_array_equal(binding.base.actions_before_mask, binding.base_actions)
    np.testing.assert_array_equal(binding.base.actions_after_mask, binding.base_actions)
    np.testing.assert_array_equal(
        binding.memory.actions_before_mask,
        binding.memory_actions_before_mask,
    )
    np.testing.assert_array_equal(binding.memory.actions_after_mask, binding.memory_actions)
    np.testing.assert_array_equal(
        binding.planner.actions_before_mask,
        binding.planner_actions_before_mask,
    )
    np.testing.assert_array_equal(binding.planner.actions_after_mask, binding.final_actions)
    identity_rows = np.reshape(
        np.stack(
            tuple(
                np.asarray(receipt.action_receipt_identity_words)
                for receipt in (binding.base, binding.memory, binding.planner)
            )
        ),
        (6, 4),
    )
    assert len({tuple(int(word) for word in row) for row in identity_rows}) == 6


def test_invalid_binding_fails_before_every_outcome_or_learning_donor(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = rig.dyad
    event = dyad.prepare_event(rig.state)
    binding = dyad.bind_current_actions(rig.state, event)
    memory_inputs = dyad.causal_core_memory_event_inputs(rig.state, event)
    invalid = binding.replace(
        content_tag_words=binding.content_tag_words.at[0].set(
            jnp.bitwise_xor(
                binding.content_tag_words[0],
                jnp.asarray(1, dtype=jnp.uint32),
            )
        )
    )
    calls = {
        "context_prepare": 0,
        "context_step": 0,
        "hccl_stage": 0,
        "agent_0": 0,
        "agent_1": 0,
        "planner": 0,
    }

    def forbidden(name: str) -> object:
        def call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls[name] += 1
            raise AssertionError(f"{name} ran after invalid preflight")

        return call

    monkeypatch.setattr(dyad.context, "prepare", forbidden("context_prepare"))
    monkeypatch.setattr(dyad.context, "step", forbidden("context_step"))
    monkeypatch.setattr(dyad.hccl, "stage", forbidden("hccl_stage"))
    monkeypatch.setattr(
        dyad.agent_0,
        "prepare_memory_transition",
        forbidden("agent_0"),
    )
    monkeypatch.setattr(
        dyad.agent_1,
        "prepare_memory_transition",
        forbidden("agent_1"),
    )
    monkeypatch.setattr(
        dyad.planner,
        "completed_transition",
        forbidden("planner"),
    )

    with pytest.raises(ValueError, match="source, event, or action binding"):
        dyad.prepare_transaction(
            rig.state,
            event,
            invalid,
            memory_inputs[0],
            memory_inputs[1],
            MASKS,
        )
    assert calls == {name: 0 for name in calls}


def test_noncore_memory_metadata_fails_before_every_outcome_or_learning_donor(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = rig.dyad
    event = dyad.prepare_event(rig.state)
    binding = dyad.bind_current_actions(rig.state, event)
    memory_inputs = dyad.causal_core_memory_event_inputs(rig.state, event)
    for index, item in enumerate(memory_inputs):
        assert not bool(item.query_uncertainty_available)
        assert not bool(item.entry_uncertainty_available)
        assert bool(item.entry_safety_cost_available)
        assert int(item.provenance_id) == index
        assert int(item.source_id) == index
        assert int(jax.lax.bitcast_convert_type(item.query_uncertainty, jnp.uint32)) == 0
        assert int(jax.lax.bitcast_convert_type(item.entry_uncertainty, jnp.uint32)) == 0
        assert int(jax.lax.bitcast_convert_type(item.entry_safety_cost, jnp.uint32)) == 0
    invalid = memory_inputs[0].replace(
        entry_safety_cost=jnp.asarray(0.75, dtype=jnp.float32),
    )
    calls = {
        "context_prepare": 0,
        "context_step": 0,
        "hccl_stage": 0,
        "agent_0": 0,
        "agent_1": 0,
        "planner": 0,
    }

    def forbidden(name: str) -> object:
        def call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls[name] += 1
            raise AssertionError(f"{name} ran after invalid memory metadata")

        return call

    monkeypatch.setattr(dyad.context, "prepare", forbidden("context_prepare"))
    monkeypatch.setattr(dyad.context, "step", forbidden("context_step"))
    monkeypatch.setattr(dyad.hccl, "stage", forbidden("hccl_stage"))
    monkeypatch.setattr(
        dyad.agent_0,
        "prepare_memory_transition",
        forbidden("agent_0"),
    )
    monkeypatch.setattr(
        dyad.agent_1,
        "prepare_memory_transition",
        forbidden("agent_1"),
    )
    monkeypatch.setattr(
        dyad.planner,
        "completed_transition",
        forbidden("planner"),
    )

    with pytest.raises(ValueError, match="causal-core memory event input"):
        dyad.prepare_transaction(
            rig.state,
            event,
            binding,
            invalid,
            memory_inputs[1],
            MASKS,
        )
    assert calls == {name: 0 for name in calls}

    with pytest.raises(ValueError, match="sidecars are unsupported"):
        dyad.prepare_transaction(
            rig.state,
            event,
            binding,
            memory_inputs[0],
            memory_inputs[1],
            MASKS,
            extended_action_masks=(
                jnp.ones((N_ACTIONS,), dtype=jnp.bool_),
                None,
            ),
        )
    assert calls == {name: 0 for name in calls}


def test_one_complete_prepare_integrity_and_adoption_advances_every_owner_once(
    prepared_rig: _PreparedRig,
) -> None:
    dyad = prepared_rig.rig.dyad
    source = prepared_rig.rig.state
    binding = prepared_rig.binding
    prepared = prepared_rig.prepared
    receipt = prepared_rig.receipt
    result = prepared_rig.result
    prepared_agents = (prepared.agent_0, prepared.agent_1)

    assert bool(prepared.source_state_valid)
    assert bool(prepared.event_valid)
    assert bool(prepared.binding_valid)
    assert bool(prepared.binding_matches_source)
    assert bool(jnp.all(prepared.pre_outcome_context_bound))
    assert bool(prepared.hccl_staged_once)
    assert bool(prepared.credit_algebra_valid)
    assert bool(jnp.all(prepared.memory_preparations_valid))
    assert bool(jnp.all(prepared.context_candidates_valid))
    assert bool(prepared.planner_transition_valid)
    assert bool(jnp.all(prepared.finalizations_valid))
    assert bool(prepared.shared_planner_binding_valid)
    assert bool(prepared.candidate_state_valid)
    assert bool(prepared.preparation_valid)
    assert bool(receipt.integrity_bound)
    np.testing.assert_array_equal(
        receipt.prepared_content_tag_words,
        prepared.content_tag_words,
    )
    assert bool(dyad.state_valid(prepared.candidate_state))

    for index, agent in enumerate(prepared_agents):
        source_context = (source.context_0_state, source.context_1_state)[index]
        assert int(agent.agent_index) == index
        assert agent.integrity_receipt is not None
        assert bool(agent.context_result.update_applied)
        assert bool(agent.memory_preparation.preparation_valid)
        assert bool(agent.memory_preparation.transition_final_action_exact)
        assert bool(agent.finalization.finalization_valid)
        donor = agent.memory_preparation.donor_prepared
        assert donor is not None
        coordinator_result = donor.coordinator_result
        assert coordinator_result is not None
        stomp_result = (
            coordinator_result.evaluated.prepared.inner_result
            .prototype_result.oak_stomp_update_result
        )
        assert int(stomp_result.planning_backups) == 0
        assert int(stomp_result.nested_updates_required) == 1
        assert int(stomp_result.nested_updates_applied) == 1
        assert int(stomp_result.executing_option) == -1
        assert int(agent.transition.action) == int(binding.final_actions[index])
        assert agent.transition.horde_cumulants.shape == (len(HORDE_NAMES),)
        assert agent.transition.horde_discounts.shape == (len(HORDE_NAMES),)
        np.testing.assert_array_equal(
            agent.context_preparation.source_content_token,
            source_context.content_token,
        )
        assert int(agent.finalization.final_action_binding.memory_action) == int(
            agent.memory_preparation.memory_candidate_state.action_binding.memory_action
        )
        assert int(agent.finalization.final_action_binding.final_action) == int(
            agent.finalization.candidate_state.action_binding.final_action
        )

    _tree_exact(prepared.candidate_state.hccl_state, prepared.hccl_result.state)
    _tree_exact(
        prepared.candidate_state.agent_0_state,
        prepared.agent_0.finalization.candidate_state,
    )
    _tree_exact(
        prepared.candidate_state.agent_1_state,
        prepared.agent_1.finalization.candidate_state,
    )
    _tree_exact(prepared.candidate_state.planner_state, prepared.planner_result.state)
    _tree_exact(
        prepared.candidate_state.context_0_state,
        prepared.agent_0.context_result.state,
    )
    _tree_exact(
        prepared.candidate_state.context_1_state,
        prepared.agent_1.context_result.state,
    )

    assert bool(result.source_state_matches)
    assert bool(result.source_state_valid)
    assert bool(result.prepared_content_valid)
    assert bool(result.receipt_valid)
    assert bool(jnp.all(result.child_adoptions_valid))
    assert bool(result.candidate_state_valid)
    assert bool(result.hccl_owner_committed)
    assert bool(jnp.all(result.action_stack_owners_committed))
    assert bool(result.planner_owner_committed)
    assert bool(jnp.all(result.context_owners_committed))
    assert bool(jnp.all(result.lineage_owners_committed))
    assert bool(result.update_applied)
    assert not bool(result.complete_source_returned)
    assert result.agent_0_adoption is not None
    assert result.agent_1_adoption is not None
    _tree_exact(result.state, prepared.candidate_state)
    assert bool(dyad.state_valid(result.state))
    _assert_primitive_only_state(dyad, result.state)


def test_prepare_and_adoption_work_are_exact_and_adoption_reevaluates_no_donor(
    prepared_rig: _PreparedRig,
) -> None:
    prepared = prepared_rig.prepared
    work = prepared.work
    result = prepared_rig.result
    planner = prepared_rig.rig.dyad.planner.completed_transition_work_budget()

    assert prepared_rig.preparation_planner_pair_authentication_calls == 5
    assert prepared_rig.receipt_planner_pair_authentication_calls == 4
    assert prepared_rig.adoption_planner_pair_authentication_calls == 6
    assert int(work.planner_pair_authentication_calls) == (
        prepared_rig.preparation_planner_pair_authentication_calls
    )
    assert int(result.adoption_work.planner_validation_pair_authentication_calls) == (
        prepared_rig.adoption_planner_pair_authentication_calls
    )

    assert (
        planner.behavior_parameter_update_attempts,
        planner.grounded_parameter_update_attempts,
        planner.cache_authentication_evaluations,
        planner.behavior_probability_vector_evaluations,
        planner.grounded_joint_cell_prediction_equivalents,
        planner.expected_reward_marginalization_products,
        planner.prototype_replacement_candidates,
        planner.atomic_pair_commit_decisions,
        planner.environment_transition_proposals,
        planner.replay_updates,
        planner.post_init_random_draws,
    ) == (2, 2, 2, 8, 18, 16, 2, 2, 0, 0, 0)

    exact_prepare_scalars = {
        "supplied_event_receipts": 1,
        "supplied_action_binding_bundles": 1,
        "event_receipt_preparations": 0,
        "event_random_draws": 0,
        "action_receipt_validation_rebindings": 3,
        "action_identity_validation_recomputations": 6,
        "hccl_stage_calls": 1,
        "world_proposal_calls": 8,
        "attribution_proposal_calls": 8,
        "designated_counterfactual_slots": 7,
        "inner_discarded_world_proposal_calls": 7,
        "inner_selected_pp_world_successors": 1,
        "outer_committed_pp_world_successors": 0,
        "world_duplicate_mm_checks": 1,
        "attribution_duplicate_mm_checks": 1,
        "memory_credit_panel_derivations": 1,
        "planner_completed_transition_calls": 1,
        "behavior_update_attempts": planner.behavior_parameter_update_attempts,
        "grounded_update_attempts": planner.grounded_parameter_update_attempts,
        "planner_pair_authentication_calls": 5,
        "planner_validation_pair_authentication_calls": 4,
        "planner_transition_pair_authentication_calls": 1,
        "planner_cache_authentication_evaluations": 10,
        "planner_behavior_probability_vector_evaluations": 16,
        "planner_grounded_joint_cell_prediction_equivalents": 50,
        "planner_expected_reward_marginalization_products": 48,
        "planner_replacement_candidates": planner.prototype_replacement_candidates,
        "planner_atomic_pair_commit_decisions": planner.atomic_pair_commit_decisions,
        "planner_decision_evaluations": 2,
        "planner_decision_joint_cells": 8,
        "planner_environment_transition_proposals": planner.environment_transition_proposals,
        "planner_replay_updates": 0,
        "planner_post_init_random_draws": 0,
        "prepared_content_digest_evaluations": 1,
    }
    for name, expected in exact_prepare_scalars.items():
        assert int(getattr(work, name)) == expected

    exact_prepare_vectors = {
        "context_preparations": (1, 1),
        "memory_credit_readouts": (1, 1),
        "context_steps": (1, 1),
        "lineage_proposals": (1, 1),
        "action_stack_memory_preparations": (1, 1),
        "feedback_settlement_evaluations": (0, 0),
        "coordinator_update_evaluations": (1, 1),
        "memory_action_replacement_evaluations": (0, 0),
        "fast_state_transition_attempts": (1, 1),
        "prototype_transition_attempts": (1, 1),
        "feature_lifecycle_route_attempts": (0, 0),
        "feature_lifecycle_arithmetic_count_available": (True, True),
        "active_pair_value_materializations": (60, 60),
        "candidate_pair_product_materializations": (
            PAIR_CANDIDATES,
            PAIR_CANDIDATES,
        ),
        "lifecycle_router_candidate_evaluations": (2, 2),
        "active_pair_slot_capacity": (12, 12),
        "pair_candidate_capacity": (PAIR_CANDIDATES, PAIR_CANDIDATES),
        "routed_representation_width": (35, 35),
        "coordinator_base_action_candidates": (1, 1),
        "memory_action_candidates": (0, 0),
        "learned_memory_query_evaluations": (1, 1),
        "learned_memory_write_evaluations": (1, 1),
        "learned_memory_reencode_evaluations": (0, 0),
        "learned_memory_reencode_count_available": (False, False),
        "final_action_bindings": (1, 1),
        "final_binding_donor_reevaluations": (0, 0),
        "child_finalization_structural_recomputations": (2, 2),
        "child_integrity_receipts": (1, 1),
    }
    for name, expected_vector in exact_prepare_vectors.items():
        np.testing.assert_array_equal(
            getattr(work, name),
            expected_vector,
            err_msg=name,
        )

    adoption_work = result.adoption_work
    assert int(adoption_work.source_state_integrity_checks) == 1
    assert int(adoption_work.preparation_integrity_checks) == 1
    assert int(adoption_work.receipt_integrity_checks) == 1
    assert int(adoption_work.outer_commit_decisions) == 1
    assert int(adoption_work.outer_committed_pp_world_successors) == 1
    assert int(adoption_work.outer_discarded_world_proposals) == 7
    np.testing.assert_array_equal(
        adoption_work.outer_child_finalization_structural_recomputations,
        (1, 1),
    )
    np.testing.assert_array_equal(adoption_work.action_stack_integrity_adoptions, (1, 1))
    np.testing.assert_array_equal(
        adoption_work.child_adoption_structural_recomputations,
        (1, 1),
    )
    assert int(adoption_work.world_reevaluations) == 0
    assert int(adoption_work.planner_reevaluations) == 0
    assert int(adoption_work.planner_validation_pair_authentication_calls) == 6
    assert (
        int(adoption_work.planner_validation_agent_cache_authentication_evaluations)
        == 12
    )
    assert (
        int(adoption_work.planner_validation_behavior_probability_vector_evaluations)
        == 12
    )
    assert (
        int(
            adoption_work.planner_validation_grounded_joint_cell_prediction_equivalents
        )
        == 48
    )
    assert (
        int(adoption_work.planner_validation_expected_reward_marginalization_products)
        == 48
    )
    for name in (
        "context_reevaluations",
        "coordinator_reevaluations",
        "prototype_reevaluations",
        "learned_memory_reevaluations",
    ):
        np.testing.assert_array_equal(getattr(adoption_work, name), (0, 0))

    for child in (result.agent_0_adoption, result.agent_1_adoption):
        assert child is not None
        assert bool(child.diagnostics.transaction_applied)
        assert bool(child.diagnostics.transition_final_action_exact)
        assert bool(child.diagnostics.completed_entry_final_action_exact)
        assert int(child.adoption_work.integrity_evaluations) == 1
        assert int(child.adoption_work.final_action_binding_reconstructions) == 1
        for name in (
            "donor_evaluations",
            "coordinator_update_evaluations",
            "prototype_replacement_evaluations",
            "planner_model_evaluations",
            "learned_memory_evaluations",
        ):
            assert int(getattr(child.adoption_work, name)) == 0


def test_through_memory_split_is_exact_and_wrapper_output_is_bit_identical(
    split_rig: _SplitRig,
) -> None:
    dyad = split_rig.prepared_rig.rig.dyad
    through = split_rig.through_memory
    wrapped = split_rig.prepared_rig.prepared

    assert HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_AGENT_SCHEMA == (
        "alberta.hccl-continual-dyad-through-memory-agent.v2"
    )
    assert HCCL_CONTINUAL_DYAD_THROUGH_MEMORY_SCHEMA == (
        "alberta.hccl-continual-dyad-through-memory-transaction.v2"
    )
    assert isinstance(through, HCCLContinualDyadThroughMemoryTransaction)
    assert isinstance(through.work, HCCLContinualDyadThroughMemoryWork)
    assert isinstance(through.agent_0, HCCLContinualDyadThroughMemoryAgent)
    assert isinstance(through.agent_1, HCCLContinualDyadThroughMemoryAgent)
    assert bool(through.through_memory_valid)
    assert dyad._through_memory_content_valid(
        split_rig.prepared_rig.rig.state,
        through,
    )
    np.testing.assert_array_equal(
        through.content_tag_words,
        dyad._through_memory_tag(through),
    )

    through_agents = (through.agent_0, through.agent_1)
    completed_agents = (split_rig.completed.agent_0, split_rig.completed.agent_1)
    for through_agent, completed_agent in zip(
        through_agents,
        completed_agents,
        strict=True,
    ):
        np.testing.assert_array_equal(
            through_agent.content_tag_words,
            dyad._through_memory_agent_tag(through_agent),
        )
        for field in dataclasses.fields(HCCLContinualDyadThroughMemoryAgent):
            if field.name == "content_tag_words":
                continue
            _tree_exact(
                getattr(through_agent, field.name),
                getattr(completed_agent, field.name),
            )

    for field in dataclasses.fields(HCCLContinualDyadThroughMemoryWork):
        if field.name in {
            "agent_content_digest_evaluations",
            "transaction_content_digest_evaluations",
            "planner_validation_pair_authentication_calls",
            "planner_validation_agent_cache_authentication_evaluations",
            "planner_validation_behavior_probability_vector_evaluations",
            "planner_validation_grounded_joint_cell_prediction_equivalents",
            "planner_validation_expected_reward_marginalization_products",
        }:
            continue
        _tree_exact(
            getattr(through.work, field.name),
            getattr(split_rig.completed.work, field.name),
        )
    np.testing.assert_array_equal(
        through.work.agent_content_digest_evaluations,
        (1, 1),
    )
    assert int(through.work.transaction_content_digest_evaluations) == 1
    assert split_rig.through_planner_pair_authentication_calls == 2
    assert int(through.work.planner_validation_pair_authentication_calls) == 2
    assert (
        int(through.work.planner_validation_agent_cache_authentication_evaluations)
        == 4
    )
    assert (
        int(through.work.planner_validation_behavior_probability_vector_evaluations)
        == 4
    )
    assert (
        int(through.work.planner_validation_grounded_joint_cell_prediction_equivalents)
        == 16
    )
    assert (
        int(through.work.planner_validation_expected_reward_marginalization_products)
        == 16
    )
    assert int(split_rig.completed.work.planner_completed_transition_calls) == 1
    assert split_rig.completion_planner_pair_authentication_calls == 3
    assert (
        int(split_rig.completed.work.planner_validation_pair_authentication_calls)
        == 4
    )
    np.testing.assert_array_equal(
        split_rig.completed.work.final_action_bindings,
        (1, 1),
    )
    np.testing.assert_array_equal(
        split_rig.completed.work.final_binding_donor_reevaluations,
        (0, 0),
    )
    _tree_exact(split_rig.completed, wrapped)


def test_split_rejects_retagged_semantic_tamper_foreign_owner_and_replay_before_planner(
    split_rig: _SplitRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_rig = split_rig.prepared_rig
    dyad = prepared_rig.rig.dyad
    through = split_rig.through_memory
    tampered_agent = dyad._seal_through_memory_agent(
        through.agent_0.replace(
            memory_credit=(
                through.agent_0.memory_credit
                + jnp.asarray(1.0, dtype=jnp.float32)
            )
        )
    )
    tampered = dyad._seal_through_memory(
        through.replace(agent_0=tampered_agent)
    )
    np.testing.assert_array_equal(
        tampered_agent.content_tag_words,
        dyad._through_memory_agent_tag(tampered_agent),
    )
    np.testing.assert_array_equal(
        tampered.content_tag_words,
        dyad._through_memory_tag(tampered),
    )

    foreign_owner = BINDING_OWNER[:-1] + (BINDING_OWNER[-1] ^ 1,)
    foreign_dyad = _dyad(binding_owner_digest=foreign_owner)
    planner_calls = 0

    def _forbidden_planner_call(*args: object, **kwargs: object) -> object:
        nonlocal planner_calls
        del args, kwargs
        planner_calls += 1
        raise AssertionError("planner must not run for a refused split")

    monkeypatch.setattr(
        type(dyad.planner),
        "completed_transition",
        _forbidden_planner_call,
    )
    refused = (
        (dyad, prepared_rig.rig.state, tampered),
        (foreign_dyad, prepared_rig.rig.state, through),
        (dyad, prepared_rig.result.state, through),
    )
    for owner, state, candidate in refused:
        with pytest.raises(ValueError, match="tampered, foreign, replayed, or invalid"):
            owner.complete_with_factorized_planner(state, candidate)
    assert planner_calls == 0


def test_planning_disabled_split_is_constructible_with_matched_work_and_no_dispatch(
    split_rig: _SplitRig,
) -> None:
    dyad = _dyad(planning_enabled=False)
    state = dyad.init(jr.key(719))
    event = dyad.prepare_event(state)
    binding = dyad.bind_current_actions(state, event)
    memory_inputs = dyad.causal_core_memory_event_inputs(state, event)
    through = dyad.prepare_through_memory(
        state,
        event,
        binding,
        memory_inputs[0],
        memory_inputs[1],
        MASKS,
    )
    completed = dyad.complete_with_factorized_planner(state, through)

    assert dyad.config.planner.planning_enabled is False
    assert bool(through.through_memory_valid)
    assert bool(completed.preparation_valid)
    for planner_agent, prepared_agent in zip(
        (completed.planner_result.state.agent_0, completed.planner_result.state.agent_1),
        (completed.agent_0, completed.agent_1),
        strict=True,
    ):
        assert not bool(planner_agent.cache.planner_consumed)
        assert int(prepared_agent.finalization.final_action_binding.final_action) == int(
            prepared_agent.memory_preparation.memory_candidate_state
            .action_binding.memory_action
        )
    _tree_exact(completed.work, split_rig.completed.work)
    resources = dyad.resource_record(
        state,
        event=event,
        binding=binding,
        prepared=completed,
    )
    assert resources.output_write_calls == 0
    assert resources.artifact_bytes_written == 0

def test_resource_record_counts_each_owner_once_and_identifies_smallest_nodes(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    planner_type = type(dyad.planner)
    original_authenticate = planner_type.authenticate_pair
    observed_calls = 0

    def counted_authenticate(*args: object, **kwargs: object) -> object:
        nonlocal observed_calls
        observed_calls += 1
        return original_authenticate(*args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(planner_type, "authenticate_pair", counted_authenticate)
        resources = dyad.resource_record(
            prepared_rig.rig.state,
            event=prepared_rig.event,
            binding=prepared_rig.binding,
            prepared=prepared_rig.prepared,
            receipt=prepared_rig.receipt,
        )

    assert resources.schema == HCCL_CONTINUAL_DYAD_RESOURCE_SCHEMA
    assert resources.hccl_state_owners == 1
    assert resources.action_stack_state_owners == 2
    assert resources.planner_pair_state_owners == 1
    assert resources.context_state_owners == 2
    assert resources.lineage_state_owners == 2
    assert resources.outer_integrity_owners == 1
    assert resources.nested_breakdowns_excluded_from_total
    assert (
        resources.total_persistent_state_nbytes
        == resources.measured_total_persistent_state_nbytes
        == resources.hccl_state_nbytes
        + resources.agent_0_action_stack_nbytes
        + resources.agent_1_action_stack_nbytes
        + resources.planner_pair_state_nbytes
        + resources.context_pair_state_nbytes
        + resources.outer_integrity_nbytes
    )
    assert resources.fast_state_pair_nbytes == N_AGENTS * FAST_DIM * 4 == 32
    assert resources.outer_integrity_nbytes == 2 * 32 == 64
    independently_owned = {
        "hccl": resources.hccl_state_nbytes,
        "agent_0": resources.agent_0_action_stack_nbytes,
        "agent_1": resources.agent_1_action_stack_nbytes,
        "planner_pair": resources.planner_pair_state_nbytes,
        "context_pair": resources.context_pair_state_nbytes,
        "outer_integrity": resources.outer_integrity_nbytes,
    }
    assert min(independently_owned, key=independently_owned.__getitem__) == (
        "outer_integrity"
    )
    assert resources.outer_action_binding_measurement_available
    assert resources.prepared_transaction_measurement_available
    assert resources.preparation_receipt_measurement_available
    assert observed_calls == 6
    assert resources.planner_validation_pair_authentication_calls == observed_calls
    assert resources.planner_validation_agent_cache_authentication_evaluations == 12
    assert resources.planner_validation_behavior_probability_vector_evaluations == 12
    assert (
        resources.planner_validation_grounded_joint_cell_prediction_equivalents
        == 48
    )
    assert resources.planner_validation_expected_reward_marginalization_products == 48
    assert resources.child_finalization_structural_recomputations == (1, 1)
    assert resources.event_receipt_nbytes > 0
    assert resources.outer_action_binding_nbytes > 0
    assert resources.prepared_transaction_nbytes > resources.total_persistent_state_nbytes
    assert resources.preparation_receipt_nbytes > 0
    assert resources.preparation_persisted is False
    assert resources.prepared_checkpoint_supported is False
    assert resources.full_generated_feature_consumer_routing is False
    assert resources.planner_generated_feature_tail_consumed is False
    assert resources.learned_memory_rows_feature_generation_bound is False
    assert resources.learned_memory_reencode_count_available is False
    assert resources.composite_jit_supported is False
    assert resources.output_write_calls == 0
    assert resources.artifact_bytes_written == 0

    tampered = prepared_rig.prepared.replace(
        content_tag_words=prepared_rig.prepared.content_tag_words.at[0].set(
            jnp.bitwise_xor(
                prepared_rig.prepared.content_tag_words[0],
                jnp.asarray(1, dtype=jnp.uint32),
            )
        )
    )
    with pytest.raises(ValueError, match="preparation content is invalid"):
        prepared_rig.rig.dyad.resource_record(
            prepared_rig.rig.state,
            event=prepared_rig.event,
            binding=prepared_rig.binding,
            prepared=tampered,
        )


def test_prepared_content_tamper_rolls_back_every_owner_bit_exactly(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_rig.prepared
    tampered = prepared.replace(
        content_tag_words=prepared.content_tag_words.at[0].set(
            jnp.bitwise_xor(
                prepared.content_tag_words[0],
                jnp.asarray(1, dtype=jnp.uint32),
            )
        )
    )
    refused, observed_calls = _observe_planner_pair_authentication_calls(
        monkeypatch,
        prepared_rig.rig.dyad,
        lambda: prepared_rig.rig.dyad.adopt_prepared_transaction(
            prepared_rig.rig.state,
            tampered,
            prepared_rig.receipt,
        ),
    )

    assert bool(refused.source_state_matches)
    assert bool(refused.source_state_valid)
    assert not bool(refused.prepared_content_valid)
    assert not bool(refused.receipt_valid)
    assert refused.agent_0_adoption is None
    assert refused.agent_1_adoption is None
    assert observed_calls == 1
    _assert_atomic_rollback(
        refused,
        prepared_rig.rig.state,
        outer_child_recomputations=(0, 0),
        planner_pair_authentication_calls=1,
    )


def test_retagged_duplicate_mm_world_payload_is_structurally_rejected(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    prepared = prepared_rig.prepared
    proposals = prepared.hccl_result.world_proposals
    altered = proposals.replace(
        evaluator_regime_id=proposals.evaluator_regime_id.at[7].set(
            (proposals.evaluator_regime_id[7] + 1).astype(jnp.int32)
        )
    )
    altered_row = HCCLCausalCoreProposal(
        **{
            field.name: jax.tree.map(
                lambda leaf: leaf[7],
                getattr(altered, field.name),
            )
            for field in dataclasses.fields(HCCLCausalCoreProposal)
        }
    )
    altered = altered.replace(
        content_tag_words=altered.content_tag_words.at[7].set(
            _world_proposal_tag(altered_row)
        )
    )
    tampered = dyad._seal_prepared(
        prepared.replace(
            hccl_result=prepared.hccl_result.replace(world_proposals=altered)
        )
    )

    with pytest.raises(ValueError, match="invalid continual-dyad preparation"):
        dyad.integrity_receipt(tampered)
    refused, observed_calls = _observe_planner_pair_authentication_calls(
        monkeypatch,
        dyad,
        lambda: dyad.adopt_prepared_transaction(
            prepared_rig.rig.state,
            tampered,
            prepared_rig.receipt,
        ),
    )
    assert not bool(refused.prepared_content_valid)
    assert observed_calls == 5
    _assert_atomic_rollback(
        refused,
        prepared_rig.rig.state,
        planner_pair_authentication_calls=observed_calls,
    )


def test_integrity_veto_rolls_back_even_when_its_content_tag_is_recomputed(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    receipt = prepared_rig.receipt
    bare_veto = receipt.replace(
        integrity_bound=jnp.asarray(False, dtype=jnp.bool_),
        content_tag_words=jnp.zeros_like(receipt.content_tag_words),
    )
    veto = bare_veto.replace(content_tag_words=dyad._receipt_tag(bare_veto))
    refused, observed_calls = _observe_planner_pair_authentication_calls(
        monkeypatch,
        dyad,
        lambda: dyad.adopt_prepared_transaction(
            prepared_rig.rig.state,
            prepared_rig.prepared,
            veto,
        ),
    )

    np.testing.assert_array_equal(veto.content_tag_words, dyad._receipt_tag(veto))
    assert bool(refused.source_state_matches)
    assert bool(refused.source_state_valid)
    assert bool(refused.prepared_content_valid)
    assert not bool(refused.receipt_valid)
    assert refused.agent_0_adoption is None
    assert refused.agent_1_adoption is None
    assert observed_calls == 5
    _assert_atomic_rollback(refused, prepared_rig.rig.state)


def test_one_typed_child_rejection_rolls_back_then_untouched_retry_commits(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    original_adoption = dyad.agent_1.adopt_finalized_transition

    def reject_after_real_reconstruction(*args: object, **kwargs: object) -> object:
        child = original_adoption(*args, **kwargs)
        return child.replace(
            state=args[0],
            diagnostics=child.diagnostics.replace(
                transaction_applied=jnp.asarray(False, dtype=jnp.bool_),
                complete_source_returned=jnp.asarray(True, dtype=jnp.bool_),
                rejected=jnp.asarray(True, dtype=jnp.bool_),
            )
        )

    with monkeypatch.context() as local:
        local.setattr(
            dyad.agent_1,
            "adopt_finalized_transition",
            reject_after_real_reconstruction,
        )
        refused, observed_calls = _observe_planner_pair_authentication_calls(
            monkeypatch,
            dyad,
            lambda: dyad.adopt_prepared_transaction(
                prepared_rig.rig.state,
                prepared_rig.prepared,
                prepared_rig.receipt,
            ),
        )

    np.testing.assert_array_equal(refused.child_adoptions_valid, (True, False))
    np.testing.assert_array_equal(
        refused.adoption_work.action_stack_integrity_adoptions,
        (1, 1),
    )
    np.testing.assert_array_equal(
        refused.adoption_work.child_adoption_structural_recomputations,
        (1, 1),
    )
    assert observed_calls == 5
    assert (
        int(refused.adoption_work.planner_validation_pair_authentication_calls)
        == observed_calls
    )
    _tree_exact(refused.state, prepared_rig.rig.state)
    assert not bool(refused.update_applied)
    assert bool(refused.complete_source_returned)
    assert int(refused.adoption_work.outer_committed_pp_world_successors) == 0
    assert int(refused.adoption_work.outer_discarded_world_proposals) == 8

    retry = dyad.adopt_prepared_transaction(
        prepared_rig.rig.state,
        prepared_rig.prepared,
        prepared_rig.receipt,
    )
    assert bool(retry.update_applied)
    _tree_exact(retry.state, prepared_rig.prepared.candidate_state)


def test_child_adoption_failures_are_normalized_with_exact_work(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    original_adoption = dyad.agent_1.adopt_finalized_transition

    def raise_after_call(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("synthetic child failure")

    def wrong_top_level_type(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    def zero_reconstruction(*args: object, **kwargs: object) -> object:
        child = original_adoption(*args, **kwargs)
        return child.replace(
            adoption_work=child.adoption_work.replace(
                final_action_binding_reconstructions=jnp.asarray(
                    0,
                    dtype=jnp.int32,
                )
            )
        )

    def negative_reconstruction(*args: object, **kwargs: object) -> object:
        child = original_adoption(*args, **kwargs)
        return child.replace(
            adoption_work=child.adoption_work.replace(
                final_action_binding_reconstructions=jnp.asarray(
                    -1,
                    dtype=jnp.int32,
                )
            )
        )

    def malformed_reconstruction(*args: object, **kwargs: object) -> object:
        child = original_adoption(*args, **kwargs)
        return child.replace(
            adoption_work=child.adoption_work.replace(
                final_action_binding_reconstructions=jnp.asarray(
                    (1,),
                    dtype=jnp.int32,
                )
            )
        )

    corruptions: tuple[Callable[..., object], ...] = (
        raise_after_call,
        wrong_top_level_type,
        zero_reconstruction,
        negative_reconstruction,
        malformed_reconstruction,
    )
    for corruption in corruptions:
        with monkeypatch.context() as local:
            local.setattr(
                dyad.agent_1,
                "adopt_finalized_transition",
                corruption,
            )
            refused, observed_calls = _observe_planner_pair_authentication_calls(
                monkeypatch,
                dyad,
                lambda: dyad.adopt_prepared_transaction(
                    prepared_rig.rig.state,
                    prepared_rig.prepared,
                    prepared_rig.receipt,
                ),
            )

        assert observed_calls == 5
        _assert_atomic_rollback(
            refused,
            prepared_rig.rig.state,
            child_adoptions_called=(1, 1),
            child_reconstructions=(1, 0),
            child_adoptions_valid=(True, False),
            planner_pair_authentication_calls=observed_calls,
        )


def test_child_finalization_exception_counts_only_the_attempted_child(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dyad = prepared_rig.rig.dyad
    calls = 0

    def fail_first_finalization(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ValueError("synthetic finalization validation failure")

    with monkeypatch.context() as local:
        local.setattr(dyad, "_child_finalization_bound", fail_first_finalization)
        refused, observed_calls = _observe_planner_pair_authentication_calls(
            monkeypatch,
            dyad,
            lambda: dyad.adopt_prepared_transaction(
                prepared_rig.rig.state,
                prepared_rig.prepared,
                prepared_rig.receipt,
            ),
        )

    assert calls == 1
    assert observed_calls == 4
    _assert_atomic_rollback(
        refused,
        prepared_rig.rig.state,
        outer_child_recomputations=(1, 0),
        planner_pair_authentication_calls=observed_calls,
    )


def test_replay_against_the_successor_returns_that_complete_successor_unchanged(
    prepared_rig: _PreparedRig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successor = prepared_rig.result.state
    replay, observed_calls = _observe_planner_pair_authentication_calls(
        monkeypatch,
        prepared_rig.rig.dyad,
        lambda: prepared_rig.rig.dyad.adopt_prepared_transaction(
            successor,
            prepared_rig.prepared,
            prepared_rig.receipt,
        ),
    )

    assert not bool(replay.source_state_matches)
    assert bool(replay.source_state_valid)
    assert bool(replay.prepared_content_valid)
    assert bool(replay.receipt_valid)
    assert replay.agent_0_adoption is None
    assert replay.agent_1_adoption is None
    assert observed_calls == 5
    _assert_atomic_rollback(
        replay,
        successor,
        planner_pair_authentication_calls=observed_calls,
    )


def test_a_consecutive_event_consumes_prior_p_and_advances_the_same_atomic_topology(
    prepared_rig: _PreparedRig,
) -> None:
    dyad = prepared_rig.rig.dyad
    source = prepared_rig.result.state
    event = dyad.prepare_event(source)
    binding = dyad.bind_current_actions(source, event)
    memory_inputs = dyad.causal_core_memory_event_inputs(source, event)
    prepared = dyad.prepare_transaction(
        source,
        event,
        binding,
        memory_inputs[0],
        memory_inputs[1],
        MASKS,
    )
    receipt = dyad.integrity_receipt(prepared)
    result = dyad.adopt_prepared_transaction(source, prepared, receipt)

    source_agents = (source.agent_0_state, source.agent_1_state)
    prepared_agents = (prepared.agent_0, prepared.agent_1)
    for index, (source_agent, prepared_agent) in enumerate(
        zip(source_agents, prepared_agents, strict=True)
    ):
        assert int(binding.final_actions[index]) == int(
            source_agent.action_binding.final_action
        )
        assert int(prepared_agent.transition.action) == int(
            source_agent.action_binding.final_action
        )
        feedback_required = bool(source_agent.action_binding.memory_feedback_required)
        assert (prepared_agent.memory_feedback is not None) == feedback_required
        assert int(
            prepared_agent.memory_preparation.prepare_work
            .feedback_settlement_evaluations
        ) == int(feedback_required)
        if prepared_agent.memory_feedback is not None:
            assert int(prepared_agent.memory_feedback.memory_action) == int(
                source_agent.action_binding.memory_action
            )
            assert int(prepared_agent.memory_feedback.final_action) == int(
                source_agent.action_binding.final_action
            )

    assert bool(prepared.preparation_valid)
    assert bool(result.update_applied)
    assert not bool(result.complete_source_returned)
    assert bool(dyad.state_valid(result.state))
    assert int(result.state.hccl_state.world_state.step_words[0]) == int(
        source.hccl_state.world_state.step_words[0]
    )
    assert int(result.state.hccl_state.world_state.step_words[1]) == (
        int(source.hccl_state.world_state.step_words[1]) + 1
    )
    for agent in (result.state.agent_0_state, result.state.agent_1_state):
        np.testing.assert_array_equal(
            agent.coordinator_state.event_words,
            result.state.hccl_state.world_state.step_words,
        )
        np.testing.assert_array_equal(
            agent.coordinator_state.inner_state.prototype_state.step_words,
            result.state.hccl_state.world_state.step_words,
        )
