# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Contracts for two-event quarantined latent-context experts."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.two_event_latent_context_experts import (
    TwoEventLatentContextExpertConfig,
    TwoEventLatentContextExpertLearner,
    TwoEventLatentContextExpertState,
    run_two_event_latent_context_expert_arrays,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _telemetry(value: int) -> jax.Array:
    return jnp.asarray(min(value, _INT32_MAX), dtype=jnp.int32)


def _learner(**overrides: Any) -> TwoEventLatentContextExpertLearner:
    values: dict[str, Any] = {
        "input_dim": 1,
        "output_dim": 1,
        "max_experts": 2,
        "step_size": 0.05,
        "grad_clip": 10.0,
    }
    values.update(overrides)
    return TwoEventLatentContextExpertLearner(
        TwoEventLatentContextExpertConfig(**values)
    )


def _separated_state(
    learner: TwoEventLatentContextExpertLearner,
) -> TwoEventLatentContextExpertState:
    initial = learner.init()
    return initial.replace(
        params=initial.params.replace(
            expert_weights=jnp.asarray([[[1.0]], [[-1.0]]], dtype=jnp.float32),
        ),
        active_expert=jnp.asarray(0, dtype=jnp.int32),
    )


def _assert_rollback(result: Any, source: TwoEventLatentContextExpertState) -> None:
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, source)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)
    assert int(result.parameter_subtree_commit_count) == 0
    assert not bool(jnp.any(result.expert_update_mask))


def test_default_is_disabled_and_resource_bytes_include_complete_pending_snapshot() -> None:
    ordinary = _learner(confirmation_routing_enabled=True)
    ablation = _learner(confirmation_routing_enabled=False)

    assert TwoEventLatentContextExpertConfig(input_dim=1).confirmation_routing_enabled is False
    assert ordinary.resource_record() == ablation.resource_record()
    resources = ordinary.resource_record()
    assert resources.state_nbytes == 53
    assert resources.prediction_cache_nbytes == 70
    assert resources.maximum_expert_predictions_per_update == 4
    assert resources.maximum_expert_losses_per_update == 2
    assert resources.maximum_candidate_gradients_per_update == 2
    assert resources.maximum_expert_subtree_commits_per_update == 1


def test_unique_challenger_opens_quarantine_with_clock_advance_and_zero_commit() -> None:
    learner = _learner(confirmation_routing_enabled=True)
    state = _separated_state(learner)
    cache = learner.predict(state, jnp.asarray([1.0], dtype=jnp.float32))
    result = learner.update(state, cache, jnp.asarray([-1.0], dtype=jnp.float32))

    assert bool(result.update_applied)
    assert bool(result.quarantine_opened)
    assert not bool(result.quarantine_second_evidence)
    assert int(result.parameter_subtree_commit_count) == 0
    assert not bool(jnp.any(result.expert_update_mask))
    chex.assert_trees_all_equal(result.state.params, state.params)
    assert int(result.state.active_expert) == 0
    assert int(result.state.step_count) == 1
    assert bool(result.state.pending_valid)
    assert int(result.state.pending_owner) == 0
    assert int(result.state.pending_candidate) == 1
    chex.assert_trees_all_equal(result.state.pending_birth_words, _words(1))
    chex.assert_trees_all_equal(
        result.state.pending_never_worse,
        jnp.asarray([True, False]),
    )
    chex.assert_trees_all_equal(
        result.state.pending_ever_strict,
        jnp.asarray([True, False]),
    )


def test_second_evidence_confirms_and_only_routing_bool_changes_commit() -> None:
    enabled = _learner(confirmation_routing_enabled=True)
    disabled = _learner(confirmation_routing_enabled=False)
    source = _separated_state(enabled)
    first_observation = jnp.asarray([1.0], dtype=jnp.float32)
    first_target = jnp.asarray([-1.0], dtype=jnp.float32)
    enabled_open = enabled.update(source, enabled.predict(source, first_observation), first_target)
    disabled_open = disabled.update(
        source,
        disabled.predict(source, first_observation),
        first_target,
    )
    chex.assert_trees_all_equal(enabled_open.state, disabled_open.state)

    pending = enabled_open.state
    observation = jnp.asarray([0.5], dtype=jnp.float32)
    target = jnp.asarray([-0.5], dtype=jnp.float32)
    enabled_result = enabled.update(pending, enabled.predict(pending, observation), target)
    disabled_result = disabled.update(pending, disabled.predict(pending, observation), target)

    assert bool(enabled_result.quarantine_second_evidence)
    assert bool(enabled_result.quarantine_confirmed)
    assert not bool(enabled_result.quarantine_rejected)
    assert bool(disabled_result.quarantine_confirmed)
    chex.assert_trees_all_equal(
        enabled_result.candidate_gradient_norms,
        disabled_result.candidate_gradient_norms,
    )
    chex.assert_trees_all_equal(
        enabled_result.quarantine_never_worse,
        disabled_result.quarantine_never_worse,
    )
    chex.assert_trees_all_equal(
        enabled_result.quarantine_ever_strict,
        disabled_result.quarantine_ever_strict,
    )
    assert not bool(enabled_result.state.pending_valid)
    assert not bool(disabled_result.state.pending_valid)
    assert int(enabled_result.parameter_subtree_commit_count) == 1
    assert int(disabled_result.parameter_subtree_commit_count) == 1
    assert int(enabled_result.selected_next_expert) == 1
    assert int(disabled_result.selected_next_expert) == 0
    chex.assert_trees_all_equal(
        enabled_result.expert_update_mask,
        jnp.asarray([False, True]),
    )
    chex.assert_trees_all_equal(
        disabled_result.expert_update_mask,
        jnp.asarray([True, False]),
    )


def test_second_tie_confirms_but_worse_rejects_and_clears_pending() -> None:
    learner = _learner(confirmation_routing_enabled=True)
    source = _separated_state(learner)
    opened = learner.update(
        source,
        learner.predict(source, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([-1.0], dtype=jnp.float32),
    ).state

    tie = learner.update(
        opened,
        learner.predict(opened, jnp.asarray([0.0], dtype=jnp.float32)),
        jnp.asarray([0.0], dtype=jnp.float32),
    )
    assert bool(tie.quarantine_confirmed)
    assert int(tie.selected_next_expert) == 1
    assert not bool(tie.state.pending_valid)

    rejected = learner.update(
        opened,
        learner.predict(opened, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([1.0], dtype=jnp.float32),
    )
    assert not bool(rejected.quarantine_confirmed)
    assert bool(rejected.quarantine_rejected)
    assert int(rejected.selected_next_expert) == 0
    assert not bool(rejected.state.pending_valid)
    assert int(rejected.parameter_subtree_commit_count) == 1


def test_tie_first_then_strict_confirms_but_tie_twice_rejects() -> None:
    learner = _learner(confirmation_routing_enabled=True)
    initial = learner.init()
    first_observation = jnp.asarray([1.0], dtype=jnp.float32)
    first_target = jnp.asarray([0.0], dtype=jnp.float32)
    opened_result = learner.update(
        initial,
        learner.predict(initial, first_observation),
        first_target,
    )

    assert bool(opened_result.quarantine_opened)
    assert int(opened_result.state.pending_candidate) == 1
    chex.assert_trees_all_equal(
        opened_result.state.pending_never_worse,
        jnp.asarray([True, False]),
    )
    chex.assert_trees_all_equal(
        opened_result.state.pending_ever_strict,
        jnp.asarray([False, False]),
    )

    strict_source = opened_result.state.replace(
        params=opened_result.state.params.replace(
            expert_weights=jnp.asarray([[[1.0]], [[-1.0]]], dtype=jnp.float32),
        )
    )
    strict = learner.update(
        strict_source,
        learner.predict(strict_source, first_observation),
        jnp.asarray([-1.0], dtype=jnp.float32),
    )
    tied = learner.update(
        opened_result.state,
        learner.predict(opened_result.state, first_observation),
        first_target,
    )

    assert bool(strict.quarantine_confirmed)
    assert not bool(strict.quarantine_rejected)
    assert int(strict.selected_next_expert) == 1
    assert not bool(tied.quarantine_confirmed)
    assert bool(tied.quarantine_rejected)
    assert int(tied.selected_next_expert) == 0


def test_ambiguous_dormant_challengers_abstain_with_clock_advance_and_zero_commit() -> None:
    learner = _learner(max_experts=3, confirmation_routing_enabled=True)
    initial = learner.init()
    state = initial.replace(
        params=initial.params.replace(
            expert_weights=jnp.asarray([[[1.0]], [[-1.0]], [[-1.0]]], dtype=jnp.float32),
        )
    )
    result = learner.update(
        state,
        learner.predict(state, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([-1.0], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert bool(result.ambiguous_challenger_abstention)
    assert int(result.parameter_subtree_commit_count) == 0
    chex.assert_trees_all_equal(result.state.params, state.params)
    assert int(result.state.step_count) == 1
    assert not bool(result.state.pending_valid)


@pytest.mark.parametrize(
    "field",
    [
        "pending_valid",
        "pending_owner",
        "pending_candidate",
        "pending_birth_words",
        "pending_never_worse",
        "pending_ever_strict",
    ],
)
def test_cache_authenticates_every_pending_field_that_can_change_routing(field: str) -> None:
    learner = _learner(confirmation_routing_enabled=True)
    source = _separated_state(learner)
    pending = learner.update(
        source,
        learner.predict(source, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([-1.0], dtype=jnp.float32),
    ).state
    cache = learner.predict(pending, jnp.asarray([0.5], dtype=jnp.float32))
    replacements: dict[str, jax.Array] = {
        "pending_valid": jnp.asarray(False, dtype=jnp.bool_),
        "pending_owner": jnp.asarray(1, dtype=jnp.int32),
        "pending_candidate": jnp.asarray(0, dtype=jnp.int32),
        "pending_birth_words": _words(2),
        "pending_never_worse": jnp.asarray([False, False]),
        "pending_ever_strict": jnp.asarray([True, True]),
    }
    tampered = pending.replace(**{field: replacements[field]})
    result = learner.update(tampered, cache, jnp.asarray([-0.5], dtype=jnp.float32))

    assert not bool(result.cache_owner_valid)
    _assert_rollback(result, tampered)


@pytest.mark.parametrize(
    "replacement",
    [
        {"pending_valid": jnp.asarray(False, dtype=jnp.bool_)},
        {"pending_owner": jnp.asarray(1, dtype=jnp.int32)},
        {"pending_candidate": jnp.asarray(0, dtype=jnp.int32)},
        {"pending_birth_words": _words(2)},
        {"pending_never_worse": jnp.asarray([False, False])},
        {"pending_ever_strict": jnp.asarray([True, True])},
    ],
)
def test_dynamically_invalid_or_stale_pending_state_fails_closed(
    replacement: dict[str, jax.Array],
) -> None:
    learner = _learner(confirmation_routing_enabled=True)
    source = _separated_state(learner)
    pending = learner.update(
        source,
        learner.predict(source, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([-1.0], dtype=jnp.float32),
    ).state.replace(**replacement)
    cache = learner.predict(pending, jnp.asarray([0.5], dtype=jnp.float32))
    result = learner.update(pending, cache, jnp.asarray([-0.5], dtype=jnp.float32))

    assert not bool(result.source_state_valid)
    _assert_rollback(result, pending)


def test_invalid_pending_payload_must_be_exact_zero() -> None:
    learner = _learner()
    state = learner.init().replace(pending_owner=jnp.asarray(1, dtype=jnp.int32))

    assert not bool(learner.state_valid(state))
    result = learner.update(
        state,
        learner.predict(state, jnp.asarray([1.0], dtype=jnp.float32)),
        jnp.asarray([1.0], dtype=jnp.float32),
    )
    _assert_rollback(result, state)


def test_eager_jit_scan_and_terminal_clock_are_exact() -> None:
    learner = _learner(confirmation_routing_enabled=True)
    state = _separated_state(learner)
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    target = jnp.asarray([-1.0], dtype=jnp.float32)
    with jax.disable_jit():
        eager = learner.update(state, learner.predict(state, observation), target)
    compiled = learner.update(state, learner.predict(state, observation), target)
    chex.assert_trees_all_equal(eager, compiled)

    observations = jnp.asarray([[1.0], [0.5], [0.0]], dtype=jnp.float32)
    targets = jnp.asarray([[-1.0], [-0.5], [0.0]], dtype=jnp.float32)
    scanned = run_two_event_latent_context_expert_arrays(
        learner,
        observations,
        targets,
        state=state,
    )
    assert scanned.predictions.shape == (3, 1)
    assert scanned.quarantine_opened.shape == (3,)
    assert scanned.parameter_subtree_commit_count.shape == (3,)
    assert int(scanned.state.step_count) == 3

    terminal = learner.init().replace(
        step_count=_telemetry(_UINT64_MAX),
        step_words=_words(_UINT64_MAX),
    )
    exhausted = learner.update(
        terminal,
        learner.predict(terminal, observation),
        target,
    )
    assert not bool(exhausted.lifetime_capacity_available)
    _assert_rollback(exhausted, terminal)


def test_nonfinite_target_and_candidate_roll_back_atomically() -> None:
    learner = _learner(step_size=np.finfo(np.float32).max)
    source = _separated_state(learner)
    cache = learner.predict(source, jnp.asarray([1.0], dtype=jnp.float32))

    bad_target = learner.update(source, cache, jnp.asarray([jnp.nan], dtype=jnp.float32))
    assert not bool(bad_target.target_valid)
    _assert_rollback(bad_target, source)

    overflow = learner.update(
        source,
        cache,
        jnp.asarray([-np.finfo(np.float32).max], dtype=jnp.float32),
    )
    assert not bool(overflow.candidate_state_valid)
    _assert_rollback(overflow, source)
