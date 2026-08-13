"""Post-envelope settlement across the semantic Prototype composition."""

from __future__ import annotations

import jax
import pytest
from test_embodied_safety_envelope import (
    _command,
    _envelope,
    _evaluate,
    _telemetry,
)
from test_embodied_safety_envelope import _state as _envelope_state
from test_prototype_consolidated_memory import _settlement, _tree_equal
from test_prototype_consolidated_semantic_memory import _agent, _started

pytestmark = pytest.mark.integration


def test_embodied_envelope_accepted_fallback_and_no_action_settle_atomically() -> None:
    agent = _agent()
    with jax.disable_jit():
        dispatch = _started(agent, decision=True)
    composition = dispatch.composition
    selected = int(composition.prototype.current_action)
    fallback = 1 - selected
    memory_before = composition.controller.memory

    accepted_envelope = _envelope()
    accepted = _evaluate(
        accepted_envelope,
        _envelope_state(accepted_envelope),
        _telemetry(),
        _command(),
    )
    assert bool(accepted.proposed_accepted)
    assert bool(accepted.action_available)
    accepted_settlement = agent.settle_dispatch(
        dispatch,
        _settlement(composition, executed_action=selected),
    )
    assert bool(accepted_settlement.composition.diagnostics.transaction_committed)
    assert not bool(accepted_settlement.composition.diagnostics.state_changed)
    assert _tree_equal(accepted_settlement.state, dispatch)

    fallback_envelope = _envelope()
    certified_fallback = _evaluate(
        fallback_envelope,
        _envelope_state(fallback_envelope),
        _telemetry(),
        _command(position=(2.0, 0.0)),
    )
    assert bool(certified_fallback.fallback_used)
    assert bool(certified_fallback.action_available)
    assert _tree_equal(certified_fallback.command, fallback_envelope.fallback_command)
    fallback_settlement = agent.settle_dispatch(
        dispatch,
        _settlement(composition, executed_action=fallback),
    )
    fallback_audit = fallback_settlement.composition.diagnostics
    assert bool(fallback_audit.transaction_committed)
    assert bool(fallback_audit.state_changed)
    assert int(fallback_settlement.action) == fallback
    assert int(fallback_settlement.state.composition.prototype.current_action) == fallback
    assert int(fallback_settlement.state.composition.dispatch_owner.selected_action) == fallback
    assert _tree_equal(
        fallback_settlement.state.composition.controller.memory,
        memory_before,
    )
    assert int(fallback_settlement.state.composition.prototype.step_count) == int(
        composition.prototype.step_count
    )
    assert not bool(fallback_audit.learner_update_applied)
    assert not bool(fallback_audit.memory_evidence_written)

    unavailable_envelope = _envelope()
    unavailable = _evaluate(
        unavailable_envelope,
        _envelope_state(unavailable_envelope),
        _telemetry(connected=False),
        _command(),
    )
    assert not bool(unavailable.action_available)
    no_action_settlement = agent.settle_dispatch(
        dispatch,
        _settlement(
            composition,
            executed_action=-1,
            action_available=False,
        ),
    )
    assert bool(no_action_settlement.composition.diagnostics.transaction_committed)
    assert int(no_action_settlement.action) == -1
    assert _tree_equal(no_action_settlement.state, dispatch)
