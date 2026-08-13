# mypy: disable-error-code="attr-defined,operator"
"""Production-owned canonical construction for the primitive-only HCCL dyad."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.hccl_continual_dyad_factory import (
    HCCL_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA,
    HCCLContinualDyadFactory,
    HCCLContinualDyadFactoryConfig,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA,
    HCCLContinualDyadTransactionConfig,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCLCausalCoreConfig,
)

pytestmark = pytest.mark.integration

_CANONICAL_EVENTS = 8_998
_SMOKE_EVENTS = 420
_CORE_L2_EVENTS = 71_984
_CORE_L3_EVENTS = 1_007_776
_HORDE_NAMES = (
    "task_discount_0p5",
    "task_discount_0p9",
    "task_discount_0p99",
    "partner_action",
    "safety_cost",
    "tv_occupancy",
    "target_zone_occupancy",
    "option_success_unavailable",
)
_HORDE_GAMMAS = (0.5, 0.9, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9)
_PROPOSAL_OWNER = (
    0x10203040,
    0x50607080,
    0x90A0B0C0,
    0xD0E0F001,
    0x12345678,
    0x9ABCDEF0,
    0x0F1E2D3C,
    0x4B5A6978,
)
_AGENT_0_OWNER = (
    0xA101A101,
    0xA202A202,
    0xA303A303,
    0xA404A404,
    0xA505A505,
    0xA606A606,
    0xA707A707,
    0xA808A808,
)
_AGENT_1_OWNER = (
    0xC101C101,
    0xC202C202,
    0xC303C303,
    0xC404C404,
    0xC505C505,
    0xC606C606,
    0xC707C707,
    0xC808C808,
)
_BINDING_OWNER = (
    0xD101D101,
    0xD202D202,
    0xD303D303,
    0xD404D404,
    0xD505D505,
    0xD606D606,
    0xD707D707,
    0xD808D808,
)


def _profile_cases() -> tuple[
    tuple[HCCLContinualDyadFactoryConfig, str, int, int], ...
]:
    return (
        (
            HCCLContinualDyadFactoryConfig(),
            HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
            _CANONICAL_EVENTS,
            _CANONICAL_EVENTS,
        ),
        (
            HCCLContinualDyadFactoryConfig.mechanics_smoke(),
            HCCL_CAUSAL_CORE_SMOKE_PROFILE,
            _SMOKE_EVENTS,
            _CANONICAL_EVENTS,
        ),
        (
            HCCLContinualDyadFactoryConfig.core_l2(),
            HCCL_CAUSAL_CORE_L2_PROFILE,
            _CORE_L2_EVENTS,
            _CORE_L2_EVENTS,
        ),
        (
            HCCLContinualDyadFactoryConfig.core_l3(),
            HCCL_CAUSAL_CORE_L3_PROFILE,
            _CORE_L3_EVENTS,
            _CORE_L3_EVENTS,
        ),
    )


def _assert_exact_tree(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            left_array = jr.key_data(left_array)
            right_array = jr.key_data(right_array)
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        np.testing.assert_array_equal(left_array, right_array)


def _assert_canonical_geometry(
    config: HCCLContinualDyadTransactionConfig,
    *,
    agent_lifetime_events: int,
) -> None:
    assert config.hccl.proposal_owner_digest == _PROPOSAL_OWNER
    assert config.agent_0.final_action_owner_digest == _AGENT_0_OWNER
    assert config.agent_1.final_action_owner_digest == _AGENT_1_OWNER
    assert config.binding_owner_digest == _BINDING_OWNER
    assert len(
        {
            config.agent_0.final_action_owner_digest,
            config.agent_1.final_action_owner_digest,
            config.binding_owner_digest,
        }
    ) == 3
    assert config.hccl.proposal_owner_digest[:4] != config.hccl.proposal_owner_digest[4:]

    prototypes: list[Mapping[str, object]] = []
    for agent in (config.agent_0, config.agent_1):
        coordinator = agent.coordinator
        builder = coordinator.builder
        prototype = coordinator.inner.prototype
        lifecycle = prototype.prototype_feature_lifecycle
        assert lifecycle is not None
        stomp = prototype.oak.stomp
        ensemble = coordinator.inner.ensemble
        memory = agent.learned_memory.memory

        assert builder.observation_dim == 19
        assert builder.n_actions == 2
        assert builder.hidden_dim == 4
        assert builder.feature_dim() == 23
        assert builder.include_raw_observation is True
        assert lifecycle.base_feature_dim == 23
        assert lifecycle.effective_pair_source_feature_dim == 16
        assert lifecycle.active_pair_slots == 12
        assert lifecycle.candidate_pair_slots == 120
        assert lifecycle.total_feature_dim == 35
        assert lifecycle.replacement_interval == 64
        assert lifecycle.max_observations == agent_lifetime_events
        assert lifecycle.n_tasks == 9
        assert lifecycle.managed_horde_demons == 8
        assert lifecycle.n_options == 0
        assert lifecycle.option_subtask_feature_indices == ()
        assert stomp.observation_dim == 35
        assert stomp.n_primitive_actions == 2
        assert stomp.n_total_actions == 2
        assert stomp.n_options == 0
        assert stomp.subtask_specs == ()
        assert stomp.base_hidden_sizes == ()
        assert stomp.option_planning_backups_per_step == 0
        assert prototype.option_search_control is None
        assert prototype.auto_curate_every == 0
        assert prototype.horde_spec is not None
        assert tuple(demon.name for demon in prototype.horde_spec.demons) == _HORDE_NAMES
        assert tuple(demon.cumulant_index for demon in prototype.horde_spec.demons) == tuple(
            range(8)
        )
        np.testing.assert_array_equal(
            prototype.horde_spec.gammas,
            np.asarray(_HORDE_GAMMAS, dtype=np.float32),
        )
        assert ensemble.router.base_dim == 23
        assert ensemble.router.active_slots == 12
        assert ensemble.world_model.observation_dim == 23
        assert ensemble.world_model.n_actions == 2
        assert ensemble.world_model.hidden_sizes == ()
        assert ensemble.ensemble_size == 1
        assert ensemble.signal_estimator.ensemble_size == 1
        assert ensemble.signal_estimator.target_dim == 25
        assert ensemble.max_events == agent_lifetime_events
        assert coordinator.learning_value_router.max_steps == agent_lifetime_events
        assert coordinator.max_events == agent_lifetime_events
        assert memory.capacity == 64
        assert (memory.observation_dim, memory.key_dim, memory.outcome_dim) == (19, 19, 19)
        assert memory.action_dim == 2
        assert memory.max_age == agent_lifetime_events
        assert memory.staleness_scale == float(agent_lifetime_events)
        prototypes.append(cast(Mapping[str, object], prototype.to_config()))

    assert prototypes[0] == prototypes[1]
    assert config.planner.observation_dim == 23
    assert config.planner.prototype_representation_dim == 35
    assert config.planner.n_actions == 2
    assert config.planner.planning_enabled is True
    assert config.context.context.max_contexts == 3
    assert config.context.context.observation_dim == 2
    assert config.context.context.n_actions == 2


def test_factory_config_is_strict_json_and_grants_no_authority() -> None:
    assert HCCL_CONTINUAL_DYAD_FACTORY_CONFIG_SCHEMA == (
        "alberta.hccl-continual-dyad-factory.config.v1"
    )
    for factory_config, expected_profile, expected_events, agent_events in _profile_cases():
        payload = factory_config.to_config()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        restored_payload = json.loads(encoded)

        assert payload["schedule_profile"] == expected_profile
        assert payload["maximum_committed_transitions"] == expected_events
        assert payload["agent_lifetime_events"] == agent_events
        assert HCCLContinualDyadFactoryConfig.from_config(restored_payload) == factory_config
        assert HCCLContinualDyadFactoryConfig.from_json(factory_config.to_json()) == factory_config
        for field in (
            "schedule_execution_authorized",
            "transaction_execution_authorized",
            "benchmark_execution_authorized",
            "seed_reservation_or_consumption_authorized",
            "artifact_authorized",
            "output_writes_authorized",
            "evidence_authorized",
            "promotion_authorized",
            "scientific_promotion_allowed",
        ):
            assert payload[field] is False
        assert payload["artifact_write_calls"] == 0
        assert payload["artifact_bytes_written"] == 0

        with pytest.raises(ValueError, match="noncanonical or unsupported"):
            HCCLContinualDyadFactoryConfig.from_config({**restored_payload, "extra": 1})

    with pytest.raises(ValueError, match="schedule_profile"):
        HCCLContinualDyadFactoryConfig(schedule_profile="unversioned-smoke")
    with pytest.raises(ValueError, match="JSON|duplicate|strict"):
        HCCLContinualDyadFactoryConfig.from_json('{"schedule_profile":"a","schedule_profile":"b"}')

    crossed_profile = HCCLContinualDyadFactoryConfig.core_l2().to_config()
    crossed_profile["schedule_profile"] = HCCL_CAUSAL_CORE_L3_PROFILE
    with pytest.raises(ValueError, match="noncanonical|unsupported"):
        HCCLContinualDyadFactoryConfig.from_config(crossed_profile)


def test_existing_factory_and_transaction_manifests_remain_byte_identical() -> None:
    expected = {
        "canonical": (
            HCCLContinualDyadFactoryConfig(),
            (875, "b270cda8f46515870e10fa354852675605b2c52220b8c0fe110a760da804e35a"),
            (35_444, "d678124f30be970b8d00d8cb4c5a9e25af1e657c8eb2c1a5ed219a5c7270b46f"),
        ),
        "smoke": (
            HCCLContinualDyadFactoryConfig.mechanics_smoke(),
            (878, "94bd670dad752feb992b07d5305028a6c8c9a6bf288de450d1050e7c2163249e"),
            (35_544, "0c56efe7c86669a948de418a6c7e42944f7a485331cfb43cd8e15247e2ac0743"),
        ),
    }
    for factory_config, factory_expected, transaction_expected in expected.values():
        factory_bytes = json.dumps(
            factory_config.to_config(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        transaction_bytes = json.dumps(
            HCCLContinualDyadFactory(factory_config).transaction_config.to_config(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert (len(factory_bytes), hashlib.sha256(factory_bytes).hexdigest()) == factory_expected
        assert (
            len(transaction_bytes),
            hashlib.sha256(transaction_bytes).hexdigest(),
        ) == transaction_expected


@pytest.mark.parametrize(
    ("factory_config", "expected_profile", "expected_events", "agent_events"),
    _profile_cases(),
)
def test_factory_reconstructs_exact_primitive_only_v2_geometry(
    factory_config: HCCLContinualDyadFactoryConfig,
    expected_profile: str,
    expected_events: int,
    agent_events: int,
) -> None:
    factory = HCCLContinualDyadFactory(factory_config)
    config = factory.transaction_config
    payload = config.to_config()

    assert HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA.endswith(".v2")
    assert payload["schema"] == HCCL_CONTINUAL_DYAD_CONFIG_SCHEMA
    assert config.hccl.world_config.schedule_profile == expected_profile
    assert config.hccl.world_config.maximum_committed_transitions == expected_events
    restored = HCCLContinualDyadTransactionConfig.from_config(
        cast(dict[str, object], json.loads(json.dumps(payload, allow_nan=False)))
    )
    assert restored.to_config() == payload
    assert payload["installed_option_slots_per_agent"] == 0
    assert payload["causal_core_target_installed_option_slots"] == 0
    assert payload["option_success_horde_head_available"] is False
    assert payload["minimum_supported_core_events"] == agent_events
    assert payload["experiential_memory_minimum_age_horizon"] == agent_events
    assert payload["experiential_memory_minimum_staleness_scale"] == float(agent_events)
    for field in (
        "output_writes_authorized",
        "artifact_authorized",
        "evidence_authorized",
        "promotion_authorized",
        "scientific_promotion_allowed",
    ):
        assert payload[field] is False
    _assert_canonical_geometry(config, agent_lifetime_events=agent_events)


def test_long_profile_rejects_every_core_l1_only_nested_lifetime() -> None:
    config = HCCLContinualDyadFactory(
        HCCLContinualDyadFactoryConfig.core_l2()
    ).transaction_config
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
                        max_observations=_CANONICAL_EVENTS,
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
                max_steps=_CANONICAL_EVENTS,
            ),
            max_events=_CANONICAL_EVENTS,
        ),
    )
    short_coordinator = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(coordinator, max_events=_CANONICAL_EVENTS),
    )
    short_ensemble = dataclasses.replace(
        agent,
        coordinator=dataclasses.replace(
            coordinator,
            inner=dataclasses.replace(
                inner,
                ensemble=dataclasses.replace(
                    inner.ensemble,
                    max_events=_CANONICAL_EVENTS,
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
                max_age=_CANONICAL_EVENTS,
            ),
        ),
    )
    short_memory_staleness = dataclasses.replace(
        agent,
        learned_memory=dataclasses.replace(
            agent.learned_memory,
            memory=dataclasses.replace(
                agent.learned_memory.memory,
                staleness_scale=float(_CANONICAL_EVENTS),
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
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(config, agent_0=short)


def test_smoke_profile_is_an_opt_in_world_schedule_only() -> None:
    canonical = HCCLContinualDyadFactory().transaction_config
    smoke = HCCLContinualDyadFactory(
        HCCLContinualDyadFactoryConfig.mechanics_smoke()
    ).transaction_config

    assert canonical.hccl.world_config == HCCLCausalCoreConfig()
    assert smoke.hccl.world_config == HCCLCausalCoreConfig.mechanics_smoke()
    assert canonical.agent_0.to_config() == smoke.agent_0.to_config()
    assert canonical.agent_1.to_config() == smoke.agent_1.to_config()
    assert canonical.planner.to_config() == smoke.planner.to_config()
    assert canonical.context.to_config() == smoke.context.to_config()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("factory_config", "expected_profile", "expected_events"),
    tuple(case[:3] for case in _profile_cases()[:2]),
)
def test_factory_initialization_is_deterministic_and_valid(
    factory_config: HCCLContinualDyadFactoryConfig,
    expected_profile: str,
    expected_events: int,
) -> None:
    factory = HCCLContinualDyadFactory(factory_config)
    with jax.disable_jit():
        first = factory.init(jr.key(1_401))
        second = factory.init(jr.key(1_401))

    assert first.factory_config == factory_config
    assert second.factory_config == factory_config
    assert first.transaction.config.hccl.world_config.schedule_profile == expected_profile
    assert (
        first.transaction.config.hccl.world_config.maximum_committed_transitions
        == expected_events
    )
    assert bool(first.transaction.state_valid(first.state))
    assert bool(second.transaction.state_valid(second.state))
    _assert_exact_tree(first.state, second.state)
    np.testing.assert_array_equal(first.state.hccl_state.world_state.step_words, (0, 0))

    for adapter, agent_state in zip(
        (first.transaction.agent_0, first.transaction.agent_1),
        (first.state.agent_0_state, first.state.agent_1_state),
        strict=True,
    ):
        prototype = adapter.coordinator.inner.prototype
        prototype_state = agent_state.coordinator_state.inner_state.prototype_state
        oak_state = prototype._oak_component_state(prototype_state.oak_state)
        stomp_state = oak_state.stomp_state
        assert stomp_state.option_policies.q_weights.shape == (0, 2, 35)
        assert stomp_state.option_models.next_state_weights.shape == (0, 35, 35)
        assert oak_state.execution_counts.shape == (0,)
        assert oak_state.utility_ema.shape == (0,)
        assert int(stomp_state.executing_option) == -1
        assert len(stomp_state.base_learner_state.head_params.weights) == 2
        assert 0 <= int(stomp_state.last_primitive_action) < 2
