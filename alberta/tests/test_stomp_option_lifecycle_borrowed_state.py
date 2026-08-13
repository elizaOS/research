# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Contracts for detached STOMP lifecycle metadata and transient borrowing."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    option_semantic_digest,
)
from alberta_framework.core.options import STOMPAgent, STOMPConfig, STOMPState, SubtaskSpec
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycle,
    STOMPOptionLifecycleMetadataState,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]

SOURCE = option_semantic_digest({"source": "borrowed-stomp-test"})
REPRESENTATION = option_semantic_digest({"representation": "obs2-v1"})
LIFECYCLE = jnp.asarray((0xB011, 0x0A5E), dtype=jnp.uint32)


def _lifecycle(*, option_planning_backups_per_step: int = 0) -> STOMPOptionLifecycle:
    stomp = STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0, threshold=0.5, max_option_steps=4),
                SubtaskSpec(feature_index=1, threshold=0.5, max_option_steps=4),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_step_size=0.0,
            base_avg_reward_step_size=0.0,
            base_trace_decay=0.0,
            option_step_size=0.0,
            option_avg_reward_step_size=0.0,
            option_trace_decay=0.0,
            option_model_decay=0.0,
            option_model_step_size=0.0,
            epsilon_base=0.0,
            epsilon_option=0.0,
            option_planning_backups_per_step=option_planning_backups_per_step,
        )
    )
    audit = OptionLifecycleAudit(
        OptionLifecycleAuditConfig(
            n_options=2,
            n_contexts=1,
            outcome_dim=2,
            fixed_horizon=1,
            maintenance_budget=1,
            signature_scales=(1.0,) * 7,
            initiation_opportunity_floor=1,
            completion_evidence_floor=1,
            model_error_evidence_floor=1,
            comparison_treatment_evidence_floor=1,
            comparison_primitive_evidence_floor=1,
            signature_evidence_floor_per_context=1,
            redundancy_shared_context_floor=1,
            max_planning_uses_per_observation=max(
                1,
                option_planning_backups_per_step,
            ),
            max_compute_cost_per_observation=1.0,
            max_observations=16,
        )
    )
    return STOMPOptionLifecycle(stomp, audit)


def test_planning_usage_excludes_completed_cold_option_slots() -> None:
    api = _lifecycle(option_planning_backups_per_step=2)
    state = _init(api)
    models = state.stomp_state.option_models.replace(
        n_completions=jnp.asarray((7, 9), dtype=jnp.int32)
    )
    usage = api._planning_usage(
        models,
        jnp.asarray((0, 0), dtype=jnp.uint32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray((True, True, True, False), dtype=jnp.bool_),
    )

    chex.assert_trees_all_equal(
        usage,
        jnp.asarray((2, 0), dtype=jnp.int32),
    )


def test_planning_usage_attributes_nothing_without_a_live_completed_option() -> None:
    api = _lifecycle(option_planning_backups_per_step=2)
    state = _init(api)
    models = state.stomp_state.option_models.replace(
        n_completions=jnp.asarray((0, 0), dtype=jnp.int32)
    )
    usage = api._planning_usage(
        models,
        jnp.asarray((0, 0), dtype=jnp.uint32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray((True, True, True, True), dtype=jnp.bool_),
    )

    chex.assert_trees_all_equal(
        usage,
        jnp.asarray((0, 0), dtype=jnp.int32),
    )


def _init(api: STOMPOptionLifecycle):
    state = api.init(
        jr.key(3),
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        lifecycle_id=LIFECYCLE,
    )
    learner = state.stomp_state.base_learner_state.replace(
        birth_timestamp=jnp.asarray(
            state.stomp_state.base_learner_state.birth_timestamp,
            dtype=jnp.float32,
        ),
        uptime_s=jnp.asarray(
            state.stomp_state.base_learner_state.uptime_s,
            dtype=jnp.float32,
        ),
    )
    return api._with_checksum(
        state.replace(
            stomp_state=state.stomp_state.replace(base_learner_state=learner)
        )
    )


def _contains_stomp_state(value: Any) -> bool:
    if type(value) is STOMPState:
        return True
    if dataclasses.is_dataclass(value):
        return any(
            _contains_stomp_state(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_stomp_state(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_stomp_state(item) for item in value.values())
    return False


def test_detached_metadata_has_zero_stomp_owners_and_reattaches_bit_exactly() -> None:
    api = _lifecycle()
    state = _init(api)

    metadata = api.detach_borrowed_stomp(state)

    assert type(metadata) is STOMPOptionLifecycleMetadataState
    assert not _contains_stomp_state(metadata)
    assert bool(api.metadata_state_valid(metadata))
    attached = api.attach_borrowed_stomp(metadata, state.stomp_state)
    assert bool(attached.metadata_valid)
    assert bool(attached.stomp_state_valid)
    assert bool(attached.binding_matches)
    assert bool(attached.transaction_applied)
    assert not bool(attached.caller_authenticated)
    chex.assert_trees_all_equal(attached.state, state)


def test_attach_is_eager_jit_exact_and_rejects_one_bit_owner_tamper() -> None:
    api = _lifecycle()
    state = _init(api)
    eager_metadata = api.detach_borrowed_stomp(state)
    jit_metadata = jax.jit(api.detach_borrowed_stomp)(state)
    chex.assert_trees_all_equal(jit_metadata, eager_metadata)

    eager = api.attach_borrowed_stomp(eager_metadata, state.stomp_state)
    compiled = jax.jit(api.attach_borrowed_stomp)(eager_metadata, state.stomp_state)
    chex.assert_trees_all_equal(compiled, eager)

    tampered_stomp = state.stomp_state.replace(
        base_average_reward=jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(
                state.stomp_state.base_average_reward,
                jnp.uint32,
            )
            ^ jnp.uint32(1),
            jnp.float32,
        )
    )
    rejected = jax.jit(api.attach_borrowed_stomp)(eager_metadata, tampered_stomp)
    assert bool(rejected.stomp_state_valid)
    assert not bool(rejected.binding_matches)
    assert not bool(rejected.transaction_applied)


def test_detaching_a_corrupt_full_binding_cannot_create_attachable_metadata() -> None:
    api = _lifecycle()
    state = _init(api)
    corrupt = dataclasses.replace(
        state,
        binding_checksum=state.binding_checksum.at[0].add(jnp.uint32(1)),
    )

    metadata = api.detach_borrowed_stomp(corrupt)
    attached = api.attach_borrowed_stomp(metadata, state.stomp_state)

    assert bool(api.metadata_state_valid(metadata))
    assert not bool(attached.binding_matches)
    assert not bool(attached.transaction_applied)


def test_metadata_checksum_tamper_is_fail_closed() -> None:
    api = _lifecycle()
    state = _init(api)
    metadata = api.detach_borrowed_stomp(state)
    tampered = dataclasses.replace(
        metadata,
        stomp_step_words=metadata.stomp_step_words.at[1].add(jnp.uint32(1)),
    )

    attached = jax.jit(api.attach_borrowed_stomp)(tampered, state.stomp_state)

    assert not bool(attached.metadata_valid)
    assert not bool(attached.binding_matches)
    assert not bool(attached.transaction_applied)


def _started(api: STOMPOptionLifecycle):
    state = _init(api)
    result = api.start(state, jnp.asarray((0.0, 0.0), dtype=jnp.float32))
    assert bool(result.applied)
    return result.state


def _external_transition(api: STOMPOptionLifecycle, state):
    reward = jnp.asarray(0.25, dtype=jnp.float32)
    next_observation = jnp.asarray((0.1, 0.2), dtype=jnp.float32)
    raw = api.stomp_agent.update(
        state.stomp_state,
        reward,
        next_observation,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    metadata = api.detach_borrowed_stomp(state)
    declaration = api.declare_external_stomp_transition(
        metadata,
        state.stomp_state,
        raw,
        env_reward=reward,
        next_observation=next_observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        caller_derivation_declared=jnp.asarray(True, dtype=jnp.bool_),
    )
    return metadata, raw, declaration, reward, next_observation


def test_external_adoption_advances_audit_once_without_reevaluating_stomp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _lifecycle()
    source = _started(api)
    metadata, raw, declaration, reward, next_observation = _external_transition(
        api,
        source,
    )

    def forbidden_update(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("external lifecycle adoption must not evaluate STOMP")

    monkeypatch.setattr(api.stomp_agent, "update", forbidden_update)
    adopted = api.adopt_external_stomp_update(
        metadata,
        source.stomp_state,
        raw,
        declaration,
        env_reward=reward,
        next_observation=next_observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )

    assert bool(adopted.source_metadata_valid)
    assert bool(adopted.source_stomp_valid)
    assert bool(adopted.source_binding_matches)
    assert bool(adopted.result_clock_binding_valid)
    assert bool(adopted.result_endpoint_binding_valid)
    assert bool(adopted.termination_binding_valid)
    assert bool(adopted.reward_binding_valid)
    assert bool(adopted.model_signature_binding_valid)
    assert bool(adopted.declaration_binding_valid)
    assert bool(adopted.audit_applied)
    assert bool(adopted.metadata_advanced)
    assert not bool(adopted.control_transition_rolled_back)
    assert not bool(adopted.derivation_recomputed)
    assert bool(adopted.caller_authority_required)
    assert not bool(adopted.caller_authenticated)
    attached = api.attach_borrowed_stomp(adopted.state, raw.state)
    assert bool(attached.transaction_applied)
    assert int(adopted.state.audit_state.observation_count) == (
        int(metadata.audit_state.observation_count) + 1
    )


@pytest.mark.parametrize(
    "tamper",
    ("source_checksum", "post_clock", "signature", "pseudo", "authority"),
)
def test_external_adoption_mismatch_is_complete_metadata_noop(tamper: str) -> None:
    api = _lifecycle()
    source = _started(api)
    metadata, raw, declaration, reward, next_observation = _external_transition(
        api,
        source,
    )
    if tamper == "source_checksum":
        declaration = declaration.replace(
            source_stomp_checksum=declaration.source_stomp_checksum.at[0].add(
                jnp.uint32(1)
            )
        )
    elif tamper == "post_clock":
        declaration = declaration.replace(
            post_step_words=declaration.post_step_words.at[1].add(jnp.uint32(1))
        )
    elif tamper == "signature":
        declaration = declaration.replace(
            frozen_model_signature=declaration.frozen_model_signature.at[0].set(
                jax.lax.bitcast_convert_type(
                    jax.lax.bitcast_convert_type(
                        declaration.frozen_model_signature[0],
                        jnp.uint32,
                    )
                    ^ jnp.uint32(1),
                    jnp.float32,
                )
            )
        )
    elif tamper == "pseudo":
        declaration = declaration.replace(
            pseudo_reward=declaration.pseudo_reward + jnp.float32(1.0)
        )
    else:
        declaration = declaration.replace(
            caller_derivation_declared=jnp.asarray(False, dtype=jnp.bool_)
        )

    adopted = jax.jit(api.adopt_external_stomp_update)(
        metadata,
        source.stomp_state,
        raw,
        declaration,
        env_reward=reward,
        next_observation=next_observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )

    assert not bool(adopted.metadata_advanced)
    assert not bool(adopted.transaction_applied)
    assert not bool(adopted.control_transition_rolled_back)
    chex.assert_trees_all_equal(adopted.state, metadata)


def test_external_adoption_rejects_result_clock_misattribution_under_jit() -> None:
    api = _lifecycle()
    source = _started(api)
    metadata, raw, declaration, reward, next_observation = _external_transition(
        api,
        source,
    )
    misattributed = raw.replace(
        pre_step_words=raw.pre_step_words.at[1].add(jnp.uint32(1))
    )

    adopted = jax.jit(api.adopt_external_stomp_update)(
        metadata,
        source.stomp_state,
        misattributed,
        declaration,
        env_reward=reward,
        next_observation=next_observation,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
    )

    assert not bool(adopted.result_clock_binding_valid)
    assert not bool(adopted.metadata_advanced)
    chex.assert_trees_all_equal(adopted.state, metadata)
