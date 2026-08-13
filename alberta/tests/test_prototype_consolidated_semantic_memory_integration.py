# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,no-untyped-call,type-var"
"""Live, compiled, and causal semantic-context integration for Prototype."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_prototype_consolidated_memory import _tree_equal
from test_prototype_consolidated_semantic_memory import (
    _agent,
    _initial,
    _record,
    _request,
    _semantic_input,
    _started,
    _transition,
)

from alberta_framework.core.prototype_consolidated_semantic_memory import (
    PrototypeConsolidatedSemanticMemoryAgent,
    PrototypeConsolidatedSemanticMemoryConfig,
    PrototypeConsolidatedSemanticMemoryState,
    PrototypeConsolidatedSemanticTransition,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()


@pytest.fixture
def _eager_only() -> Iterator[None]:
    with jax.disable_jit():
        yield


def _action_witness_agent() -> PrototypeConsolidatedSemanticMemoryAgent:
    composition = _agent().config.composition
    stomp = dataclasses.replace(
        composition.prototype.oak.stomp,
        base_step_size=0.0,
        base_avg_reward_step_size=0.0,
        option_step_size=0.0,
        option_avg_reward_step_size=0.0,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    prototype = dataclasses.replace(
        composition.prototype,
        oak=dataclasses.replace(composition.prototype.oak, stomp=stomp),
    )
    return PrototypeConsolidatedSemanticMemoryAgent(
        PrototypeConsolidatedSemanticMemoryConfig(
            composition=dataclasses.replace(composition, prototype=prototype),
            raw_observation_dim=1,
        )
    )


def _install_semantic_sensitive_policy(
    agent: PrototypeConsolidatedSemanticMemoryAgent,
    state: PrototypeConsolidatedSemanticMemoryState,
) -> PrototypeConsolidatedSemanticMemoryState:
    prototype = state.composition.prototype
    stomp = prototype.oak_state.stomp_state
    learner = stomp.base_learner_state
    params = learner.head_params.replace(
        weights=(
            jnp.asarray(((0.0, 0.0),), dtype=jnp.float32),
            jnp.asarray(((0.0, 10.0),), dtype=jnp.float32),
            jnp.asarray(((0.0, 0.0),), dtype=jnp.float32),
        ),
        biases=(
            jnp.asarray((0.0,), dtype=jnp.float32),
            jnp.asarray((-1.0,), dtype=jnp.float32),
            jnp.asarray((-100.0,), dtype=jnp.float32),
        ),
    )
    next_prototype = prototype.replace(
        oak_state=prototype.oak_state.replace(
            stomp_state=stomp.replace(
                base_learner_state=learner.replace(head_params=params)
            )
        )
    )
    result = state.replace(
        composition=state.composition.replace(prototype=next_prototype)
    )
    assert bool(agent.validate_state(result))
    return result


def test_retrieved_tail_causes_a_real_next_policy_action_difference(
    _eager_only: None,
) -> None:
    """The semantic tail affects Prototype policy, never direct dispatch."""

    agent = _action_witness_agent()
    initial = _initial(agent)
    seeded = agent.composition.controller.semantic_step(
        initial.composition.controller,
        request=_request(),
        record=_record(1.0),
    )
    assert bool(seeded.write.wrote)
    initial = initial.replace(
        composition=initial.composition.replace(controller=seeded.state)
    )
    started = agent.start(
        initial,
        jnp.zeros((1,), dtype=jnp.float32),
    ).state
    assert int(started.composition.prototype.oak_state.stomp_state.executing_option) == -1
    common = _install_semantic_sensitive_policy(agent, started)
    transition = _transition(common, next_raw=0.0)

    zero = agent.update_transition(common, transition, semantic_input=None)
    retrieved = agent.update_transition(
        common,
        transition,
        semantic_input=_semantic_input(
            common,
            # A disagreeing current record proves the producing transition
            # cannot flow backward into the pre-write retrieval.
            record=_record(-9.0),
        ),
    )
    assert int(zero.action) == 0
    assert int(retrieved.action) == 1
    np.testing.assert_array_equal(np.asarray(zero.semantic_payload), np.zeros((1,)))
    np.testing.assert_array_equal(
        np.asarray(retrieved.semantic_payload),
        np.asarray((1.0,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(zero.state.composition.prototype.current_raw_observation),
        np.asarray((0.0, 0.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(retrieved.state.composition.prototype.current_raw_observation),
        np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert bool(retrieved.semantic_candidate.retrieval.accepted)
    assert bool(retrieved.diagnostics.semantic_query_before_write_verified)
    assert bool(
        retrieved.diagnostics.semantic_context_consumed_by_next_prototype_decision
    )
    assert bool(retrieved.diagnostics.current_action_unchanged_before_learning)
    assert not bool(retrieved.diagnostics.direct_dispatch_authority)
    assert not bool(retrieved.diagnostics.safety_override_authority)


def test_terminal_autoreset_uses_zero_for_terminal_learning_and_tail_for_reset(
    _eager_only: None,
) -> None:
    agent = _agent()
    state = _started(agent)
    first = agent.update_transition(
        state,
        _transition(state, next_raw=0.1),
        semantic_input=_semantic_input(state, record=_record(2.0)),
    )
    current = first.state
    transition = PrototypeConsolidatedSemanticTransition(
        observation=current.composition.prototype.current_raw_observation[:1],
        action=current.composition.prototype.current_action,
        decision_id=current.composition.prototype.current_decision_id,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=jnp.asarray((9.0,), dtype=jnp.float32),
        next_decision_observation=jnp.asarray((-1.0,), dtype=jnp.float32),
    )
    result = agent.update_transition(
        current,
        transition,
        semantic_input=_semantic_input(current, record=_record(4.0)),
    )
    assert bool(result.semantic_candidate.retrieval.accepted)
    np.testing.assert_array_equal(
        np.asarray(result.state.composition.prototype.current_raw_observation),
        np.asarray((-1.0, 2.0), dtype=np.float32),
    )
    assert bool(
        result.diagnostics.semantic_context_consumed_by_next_prototype_decision
    )


def test_eager_jit_scan_and_fixed_sequence_are_identical() -> None:
    agent = _agent()
    state = _started(agent)
    transition = _transition(state, next_raw=0.1, reward=0.25)
    semantic = _semantic_input(state)
    eager = agent.update_transition(
        state,
        transition,
        semantic_input=semantic,
    )
    compiled = jax.jit(agent.update_transition)(
        state,
        transition,
        semantic_input=semantic,
    )
    assert _tree_equal(eager.state, compiled.state)
    np.testing.assert_array_equal(np.asarray(eager.action), np.asarray(compiled.action))
    np.testing.assert_array_equal(
        np.asarray(eager.semantic_payload), np.asarray(compiled.semantic_payload)
    )

    observations = jnp.asarray((0.1, 0.2, 0.3), dtype=jnp.float32)

    def scan_step(
        carry: PrototypeConsolidatedSemanticMemoryState,
        next_raw: jax.Array,
    ) -> tuple[PrototypeConsolidatedSemanticMemoryState, tuple[jax.Array, jax.Array]]:
        prototype = carry.composition.prototype
        step_transition = PrototypeConsolidatedSemanticTransition(
            observation=prototype.current_raw_observation[:1],
            action=prototype.current_action,
            decision_id=prototype.current_decision_id,
            reward=jnp.asarray(0.25, dtype=jnp.float32),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=next_raw[None],
            next_decision_observation=next_raw[None],
        )
        result = agent.update_transition(
            carry,
            step_transition,
            semantic_input=_semantic_input(carry),
        )
        return result.state, (result.action, result.semantic_payload)

    scan_state, scan_outputs = jax.jit(
        lambda initial, values: jax.lax.scan(scan_step, initial, values)
    )(state, observations)
    sequential_state = state
    sequential_actions: list[jax.Array] = []
    sequential_payloads: list[jax.Array] = []
    for observation in observations:
        sequential_state, output = scan_step(sequential_state, observation)
        action, payload = output
        sequential_actions.append(action)
        sequential_payloads.append(payload)
    assert _tree_equal(scan_state, sequential_state)
    np.testing.assert_array_equal(
        np.asarray(scan_outputs[0]), np.asarray(jnp.stack(sequential_actions))
    )
    np.testing.assert_array_equal(
        np.asarray(scan_outputs[1]), np.asarray(jnp.stack(sequential_payloads))
    )


def test_checkpoint_rejects_nested_tamper_and_restores_fixed_sequence(
    _eager_only: None,
) -> None:
    agent = _agent()
    state = _started(agent)
    first = agent.update_transition(
        state,
        _transition(state, next_raw=0.1),
        semantic_input=_semantic_input(state),
    ).state
    payload = agent.checkpoint_payload(first)
    composition_payload = cast(dict[str, Any], payload["composition"])
    tampered_composition = dict(composition_payload)
    prototype = first.composition.prototype
    tampered_composition["prototype_state"] = prototype.replace(
        current_action=(prototype.current_action + 1)
        % jnp.asarray(2, dtype=jnp.int32)
    )
    tampered = dict(payload)
    tampered["composition"] = tampered_composition
    with pytest.raises(ValueError, match="SHA differs"):
        agent.restore_checkpoint(
            tampered,
            source_digest=first.composition.controller.memory.source_digest,
            semantic_namespace_digest=(
                first.composition.controller.memory.semantic_namespace_digest
            ),
            representation_revision=0,
            source_revision=0,
        )

    restored = agent.restore_checkpoint(
        payload,
        source_digest=first.composition.controller.memory.source_digest,
        semantic_namespace_digest=(
            first.composition.controller.memory.semantic_namespace_digest
        ),
        representation_revision=0,
        source_revision=0,
    )
    next_result = agent.update_transition(
        restored,
        _transition(restored, next_raw=0.2),
        semantic_input=_semantic_input(restored),
    )
    assert bool(next_result.semantic_candidate.retrieval.accepted)
    assert int(next_result.state.composition.controller.memory.operation_count) == 2
