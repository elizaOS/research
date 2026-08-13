# mypy: disable-error-code="arg-type,attr-defined,call-arg"
"""Contracts and one real-donor event for the routed R35 HCCL dyad."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.experiential_memory import ExperientialMemoryConfig
from alberta_framework.core.hccl_authenticated_bmp_projection import (
    HCCLAuthenticatedBMPProjectionConfig,
)
from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCLContinualDyadFactoryConfig,
    build_hccl_continual_dyad_config,
)
from alberta_framework.core.hccl_feature_bound_memory import (
    HCCLFeatureBoundMemoryConfig,
)
from alberta_framework.core.hccl_routed_continual_dyad import (
    HCCL_ROUTED_CONTINUAL_DYAD_SCIENTIFIC_PROMOTION_ALLOWED,
    HCCLRoutedContinualDyad,
    HCCLRoutedContinualDyadConfig,
    HCCLRoutedContinualDyadPreparedTransaction,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
)
from alberta_framework.core.options import SubtaskSpec
from alberta_framework.core.prototype_factorized_partner_planner_v2 import (
    PrototypeFactorizedPartnerPlannerV2Config,
)
from alberta_framework.core.types import create_horde_spec
from alberta_framework.streams.hccl_causal_core import HCCLCausalCoreConfig

_OUTER_OWNER = (0xE101, 0xE202, 0xE303, 0xE404, 0xE505, 0xE606, 0xE707, 0xE808)
_BMP_0_OWNER = (0xA101, 0xA202, 0xA303, 0xA404, 0xA505, 0xA606, 0xA707, 0xA808)
_BMP_1_OWNER = (0xB101, 0xB202, 0xB303, 0xB404, 0xB505, 0xB606, 0xB707, 0xB808)


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
            return False
        if not np.array_equal(
            np.asarray(jax.device_get(left_array)),
            np.asarray(jax.device_get(right_array)),
        ):
            return False
    return True


def _memory_config(agent_index: int, maximum_events: int) -> HCCLFeatureBoundMemoryConfig:
    store = ExperientialMemoryConfig(
        capacity=64,
        observation_dim=35,
        key_dim=35,
        action_dim=2,
        outcome_dim=35,
        top_k=1,
        min_neighbors=1,
        distance_scale=1.0,
        min_similarity=0.0,
        min_effective_reliability=0.01,
        max_uncertainty=1.0,
        max_safety_cost=1.0,
        max_age=maximum_events,
        staleness_scale=float(maximum_events),
        utility_decay=1.0,
        eviction_utility_weight=1.0,
        eviction_recency_weight=1.0,
        recency_scale=10.0,
    )
    return HCCLFeatureBoundMemoryConfig(
        agent_index=agent_index,
        controller=LearnedExperientialMemoryControllerConfig(
            memory=store,
            admission_step_size=0.05,
            retention_step_size=0.1,
            admission_threshold=0.0,
            initial_admission_bias=0.0,
            max_abs_admission_weight=8.0,
            max_abs_counterfactual_delta=1.0,
            retention_prior=0.5,
        ),
    )


def _config() -> HCCLRoutedContinualDyadConfig:
    legacy = build_hccl_continual_dyad_config(
        HCCLContinualDyadFactoryConfig.mechanics_smoke()
    )
    maximum = legacy.agent_0.coordinator.max_events
    return HCCLRoutedContinualDyadConfig(
        hccl=legacy.hccl,
        coordinator=legacy.agent_0.coordinator,
        memory_agent_0=_memory_config(0, maximum),
        memory_agent_1=_memory_config(1, maximum),
        planner=PrototypeFactorizedPartnerPlannerV2Config(),
        context=legacy.context,
        bmp_agent_0=HCCLAuthenticatedBMPProjectionConfig(owner_digest=_BMP_0_OWNER),
        bmp_agent_1=HCCLAuthenticatedBMPProjectionConfig(owner_digest=_BMP_1_OWNER),
        binding_owner_digest=_OUTER_OWNER,
        discount=0.99,
    )


@pytest.mark.unit
def test_config_round_trip_is_strict_and_declares_no_authority() -> None:
    config = _config()
    restored = HCCLRoutedContinualDyadConfig.from_config(config.to_config())
    assert restored.to_config() == config.to_config()
    payload = config.to_config()
    assert payload["caller_supplied_pair_admission"] is False
    assert payload["dispatch_authority"] is False
    assert payload["artifact_authority"] is False
    assert payload["evidence_authority"] is False
    assert payload["promotion_authority"] is False
    assert payload["world_lifetime_events"] == 420
    assert payload["agent_lifetime_events"] == 8_998
    assert payload["planner_geometry"] == {
        "stable_base_dim": 23,
        "routed_representation_dim": 35,
        "n_actions": 2,
    }
    assert payload["context_geometry"] == {
        "max_contexts": 3,
        "observation_dim": 2,
        "n_actions": 2,
    }
    assert payload["dynamic_primitive_safety_masks_supported"] is False
    assert payload["primitive_masks_required_all_true"] is True
    assert HCCL_ROUTED_CONTINUAL_DYAD_SCIENTIFIC_PROMOTION_ALLOWED is False
    with pytest.raises(ValueError):
        dataclasses.replace(
            config,
            memory_agent_0=_memory_config(1, config.coordinator.max_events),
        )


def _config_with_prototype(config: HCCLRoutedContinualDyadConfig, prototype: Any) -> Any:
    inner = dataclasses.replace(config.coordinator.inner, prototype=prototype)
    return dataclasses.replace(config.coordinator, inner=inner)


@pytest.mark.unit
def test_config_rejects_zero_discount_option_geometry_and_wrong_horde() -> None:
    config = _config()
    with pytest.raises(ValueError, match="discount"):
        dataclasses.replace(config, discount=0.0)

    prototype = config.coordinator.inner.prototype
    lifecycle = prototype.prototype_feature_lifecycle
    assert lifecycle is not None
    option_prototype = dataclasses.replace(
        prototype,
        oak=dataclasses.replace(
            prototype.oak,
            stomp=dataclasses.replace(
                prototype.oak.stomp,
                subtask_specs=(SubtaskSpec(feature_index=0),),
            ),
        ),
        prototype_feature_lifecycle=dataclasses.replace(
            lifecycle,
            n_options=1,
            option_subtask_feature_indices=(0,),
        ),
    )
    with pytest.raises(ValueError, match="primitive-only"):
        dataclasses.replace(
            config,
            coordinator=_config_with_prototype(config, option_prototype),
        )

    horde = prototype.horde_spec
    assert horde is not None
    seven_head_prototype = dataclasses.replace(
        prototype,
        horde_spec=create_horde_spec(horde.demons[:7]),
        prototype_feature_lifecycle=dataclasses.replace(
            lifecycle,
            n_tasks=8,
            managed_horde_demons=7,
        ),
    )
    with pytest.raises(ValueError, match="eight-head"):
        dataclasses.replace(
            config,
            coordinator=_config_with_prototype(config, seven_head_prototype),
        )


@pytest.mark.unit
def test_masks_and_prng_fail_before_any_donor_execution() -> None:
    owner = HCCLRoutedContinualDyad(_config())
    partial = jnp.asarray(((True, False), (True, True)), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="all true"):
        owner.init(jr.key(7), initial_hard_action_masks=partial)
    with pytest.raises(TypeError, match="Threefry"):
        owner.init(jr.key(7, impl="rbg"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "crossing",
    (
        "world_l2",
        "coordinator",
        "learning_router",
        "ensemble",
        "lifecycle",
        "memory_0_age",
        "memory_0_staleness",
        "memory_1_age",
        "memory_1_staleness",
    ),
)
def test_config_rejects_crossed_or_short_horizons(crossing: str) -> None:
    config = _config()
    replacement: dict[str, object]
    if crossing == "world_l2":
        replacement = {
            "hccl": dataclasses.replace(
                config.hccl,
                world_config=HCCLCausalCoreConfig.core_l2(),
            )
        }
    elif crossing == "coordinator":
        replacement = {
            "coordinator": dataclasses.replace(config.coordinator, max_events=420)
        }
    elif crossing == "learning_router":
        router = dataclasses.replace(
            config.coordinator.learning_value_router,
            max_steps=config.coordinator.max_events + 1,
        )
        replacement = {
            "coordinator": dataclasses.replace(
                config.coordinator,
                learning_value_router=router,
            )
        }
    elif crossing == "ensemble":
        inner = dataclasses.replace(
            config.coordinator.inner,
            ensemble=dataclasses.replace(
                config.coordinator.inner.ensemble,
                max_events=config.coordinator.max_events + 1,
            ),
        )
        replacement = {
            "coordinator": dataclasses.replace(config.coordinator, inner=inner)
        }
    elif crossing == "lifecycle":
        prototype = config.coordinator.inner.prototype
        lifecycle = prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        replaced_prototype = dataclasses.replace(
            prototype,
            prototype_feature_lifecycle=dataclasses.replace(
                lifecycle,
                max_observations=config.coordinator.max_events + 1,
            ),
        )
        inner = dataclasses.replace(
            config.coordinator.inner,
            prototype=replaced_prototype,
        )
        replacement = {
            "coordinator": dataclasses.replace(config.coordinator, inner=inner)
        }
    else:
        agent_index = 0 if crossing.startswith("memory_0") else 1
        memory = (config.memory_agent_0, config.memory_agent_1)[agent_index]
        store_replacement = (
            {"max_age": config.coordinator.max_events + 1}
            if crossing.endswith("age")
            else {"staleness_scale": float(config.coordinator.max_events + 1)}
        )
        controller = dataclasses.replace(
            memory.controller,
            memory=dataclasses.replace(memory.controller.memory, **store_replacement),
        )
        replacement = {
            f"memory_agent_{agent_index}": dataclasses.replace(
                memory,
                controller=controller,
            )
        }
    with pytest.raises(ValueError):
        dataclasses.replace(config, **replacement)


@pytest.mark.unit
def test_public_event_surface_has_no_caller_pair_admission_or_memory_feedback() -> None:
    owner = HCCLRoutedContinualDyad(_config())
    signature = inspect.signature(owner.prepare_transaction)
    assert "pair_admission_mask" not in signature.parameters
    assert "memory_feedback" not in signature.parameters
    assert tuple(signature.parameters) == (
        "state",
        "event",
        "action_bundle",
        "next_hard_action_masks",
    )


@pytest.fixture(scope="module")
def real_event() -> Iterator[tuple[HCCLRoutedContinualDyad, Any, Any, Any]]:
    owner = HCCLRoutedContinualDyad(_config())
    source = owner.init(jr.key(9_901))
    event = owner.prepare_event(source)
    bundle = owner.bind_actions(source, event)
    prepared = owner.prepare_transaction(
        source,
        event,
        bundle,
        next_hard_action_masks=jnp.ones((2, 2), dtype=jnp.bool_),
    )
    receipt = owner.integrity_receipt(prepared)
    result = owner.adopt(source, prepared, receipt)
    yield owner, source, prepared, result


@pytest.mark.integration
@pytest.mark.slow
def test_real_genesis_and_one_event_commit_exact_r35_m_and_p(
    real_event: tuple[HCCLRoutedContinualDyad, Any, Any, Any],
) -> None:
    owner, source, prepared, result = real_event
    assert bool(owner.state_valid(source))
    assert bool(prepared.preparation_valid)
    assert bool(result.update_applied)
    assert bool(owner.state_valid(result.state))
    assert int(prepared.work.hccl_stage_calls) == 1
    assert int(prepared.work.world_proposal_calls) == 8
    assert int(prepared.work.attribution_proposal_calls) == 8
    assert tuple(np.asarray(prepared.work.context_steps)) == (1, 1)
    assert tuple(np.asarray(prepared.work.coordinator_steps)) == (1, 1)
    assert tuple(np.asarray(prepared.work.memory_steps)) == (1, 1)
    assert tuple(np.asarray(prepared.work.memory_rebinds)) == (1, 1)
    assert int(prepared.work.planner_behavior_updates) == 2
    assert int(prepared.work.planner_grounded_updates) == 2
    assert int(prepared.work.planner_joint_cells) == 8
    assert tuple(np.asarray(prepared.work.bmp_memory_replacements)) == (1, 1)
    assert tuple(np.asarray(prepared.work.bmp_planner_replacements)) == (1, 1)
    for coordinator, record in (
        (result.state.coordinator_0_state, result.state.action_record_0),
        (result.state.coordinator_1_state, result.state.action_record_1),
    ):
        assert int(coordinator.current_action) == int(record.bmp_binding.final_action)
        assert int(coordinator.inner_state.prototype_state.current_action) == int(
            record.bmp_binding.final_action
        )
        assert bool(record.bmp_binding.planner_consumed)


def _assert_resealed_rejected(
    owner: HCCLRoutedContinualDyad,
    source: Any,
    prepared: HCCLRoutedContinualDyadPreparedTransaction,
) -> None:
    resealed = owner._seal_prepared(prepared)
    receipt = owner.integrity_receipt(resealed)
    result = owner.adopt(source, resealed, receipt)
    assert not bool(receipt.integrity_bound)
    assert not bool(result.update_applied)
    assert bool(result.complete_source_returned)
    assert _tree_exact_equal(result.state, source)


@pytest.mark.integration
@pytest.mark.slow
def test_forged_curation_admission_and_one_child_veto_return_complete_source(
    real_event: tuple[HCCLRoutedContinualDyad, Any, Any, Any],
) -> None:
    owner, source, prepared, _ = real_event
    admission = prepared.agent_0.lifecycle_proof.pair_admission_mask.at[0].set(
        ~prepared.agent_0.lifecycle_proof.pair_admission_mask[0]
    )
    forged_proof = prepared.agent_0.lifecycle_proof.replace(
        pair_admission_mask=admission,
        proof_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    forged_agent = prepared.agent_0.replace(lifecycle_proof=forged_proof)
    _assert_resealed_rejected(owner, source, prepared.replace(agent_0=forged_agent))

    vetoed_agent = prepared.agent_0.replace(
        child_valid=jnp.asarray(False, dtype=jnp.bool_)
    )
    _assert_resealed_rejected(owner, source, prepared.replace(agent_0=vetoed_agent))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("kind", ("route", "memory", "planner", "bmp"))
def test_foreign_child_receipts_are_rejected(
    real_event: tuple[HCCLRoutedContinualDyad, Any, Any, Any],
    kind: str,
) -> None:
    owner, source, prepared, _ = real_event
    if kind == "route":
        agent = prepared.agent_0.replace(route_result=prepared.agent_1.route_result)
        tampered = prepared.replace(agent_0=agent)
    elif kind == "memory":
        agent = prepared.agent_0.replace(
            memory_step_result=prepared.agent_1.memory_step_result
        )
        tampered = prepared.replace(agent_0=agent)
    elif kind == "planner":
        changed_action = jnp.int32(1) - prepared.planner_result.prepared_actions[0]
        planner = prepared.planner_result.replace(
            prepared_actions=prepared.planner_result.prepared_actions.at[0].set(
                changed_action
            )
        )
        tampered = prepared.replace(planner_result=planner)
    else:
        agent = prepared.agent_0.replace(
            bmp_integrity_receipt=prepared.agent_1.bmp_integrity_receipt
        )
        tampered = prepared.replace(agent_0=agent)
    _assert_resealed_rejected(owner, source, tampered)


@pytest.mark.integration
@pytest.mark.slow
def test_clock_crossing_and_stale_outer_receipt_are_rejected(
    real_event: tuple[HCCLRoutedContinualDyad, Any, Any, Any],
) -> None:
    owner, source, prepared, _ = real_event
    wrong_coordinator = prepared.candidate_state.coordinator_0_state.replace(
        event_words=source.coordinator_0_state.event_words
    )
    wrong_candidate = owner._seal_state(
        prepared.candidate_state.replace(coordinator_0_state=wrong_coordinator)
    )
    _assert_resealed_rejected(
        owner,
        source,
        prepared.replace(candidate_state=wrong_candidate),
    )

    crossed_masks = prepared.next_hard_action_masks.at[0, 0].set(
        ~prepared.next_hard_action_masks[0, 0]
    )
    _assert_resealed_rejected(
        owner,
        source,
        prepared.replace(next_hard_action_masks=crossed_masks),
    )

    valid_receipt = owner.integrity_receipt(prepared)
    stale_receipt = valid_receipt.replace(
        source_state_token=jnp.zeros_like(valid_receipt.source_state_token)
    )
    result = owner.adopt(source, prepared, stale_receipt)
    assert not bool(result.update_applied)
    assert _tree_exact_equal(result.state, source)
