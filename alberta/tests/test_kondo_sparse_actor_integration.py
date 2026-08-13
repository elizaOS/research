# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Persistence and boundary integration for the Kondo sparse actor."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.core.kondo_sparse_actor as sparse_actor_module
from alberta_framework.core.kondo_gate import (
    KONDO_GATE_LEGACY_V1_SCHEMA,
    KONDO_GATE_SCHEMA,
    KondoGate,
    KondoGateConfig,
)
from alberta_framework.core.kondo_sparse_actor import (
    KondoActorParameters,
    KondoActorProtectedInputs,
    KondoSparseActor,
    KondoSparseActorBatch,
    KondoSparseActorConfig,
    KondoSparseActorState,
)

pytestmark = pytest.mark.integration


def _actor() -> tuple[KondoSparseActor, KondoSparseActorState]:
    actor = KondoSparseActor(
        KondoSparseActorConfig(
            feature_dim=3,
            hidden_dim=4,
            action_count=3,
            critic_dim=2,
            safety_dim=2,
            learning_rate=0.02,
            gate=KondoGateConfig(
                batch_size=6,
                target_rate=0.5,
                max_screenings=20,
            ),
        )
    )
    parameters = KondoActorParameters(
        hidden_weight=jnp.asarray(
            [
                [0.20, -0.10, 0.05, 0.30],
                [-0.15, 0.25, 0.10, -0.20],
                [0.05, 0.15, -0.30, 0.10],
            ],
            dtype=jnp.float32,
        ),
        hidden_bias=jnp.asarray([0.01, -0.02, 0.03, 0.04], dtype=jnp.float32),
        output_weight=jnp.asarray(
            [
                [0.30, -0.20, 0.10],
                [-0.10, 0.25, 0.15],
                [0.20, 0.05, -0.25],
                [-0.15, 0.10, 0.20],
            ],
            dtype=jnp.float32,
        ),
        output_bias=jnp.asarray([0.02, -0.03, 0.01], dtype=jnp.float32),
    )
    return actor, actor.init(parameters, jr.key(31, impl="threefry2x32"))


def _batch(
    actor: KondoSparseActor,
    state: KondoSparseActorState,
    *,
    offset: float = 0.0,
) -> KondoSparseActorBatch:
    features = jnp.asarray(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, -0.5, 1.0],
            [-1.0, 0.5, 0.25],
            [0.25, 1.0, -1.0],
            [0.75, -0.25, 0.5],
        ],
        dtype=jnp.float32,
    ) + jnp.asarray(offset, dtype=jnp.float32)
    actions = jnp.asarray([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
    behavior = actor.behavior_log_probability(state, features, actions)
    baseline = jnp.asarray([0.1, 0.2, 0.0, -0.1, 0.3, -0.2], dtype=jnp.float32)
    advantages = jnp.asarray([1.0, -0.5, 1.5, 0.25, 0.75, -0.2], dtype=jnp.float32)
    return KondoSparseActorBatch(
        actor_features=features,
        actions=actions,
        action_identity=actions,
        policy_revision=jnp.full((6,), state.policy_revision, dtype=jnp.int32),
        behavior_log_probability=behavior,
        valid_mask=jnp.ones((6,), dtype=jnp.bool_),
        force_keep_mask=jnp.zeros((6,), dtype=jnp.bool_),
        protected=KondoActorProtectedInputs(
            critic_features=jnp.arange(12, dtype=jnp.float32).reshape(6, 2),
            baseline_predictions=baseline,
            return_targets=baseline + advantages,
            safety_features=jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 3.0,
        ),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = (
            np.asarray(jr.key_data(lhs))
            if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key)
            else np.asarray(lhs)
        )
        rhs_array = (
            np.asarray(jr.key_data(rhs))
            if jax.dtypes.issubdtype(rhs.dtype, jax.dtypes.prng_key)
            else np.asarray(rhs)
        )
        np.testing.assert_array_equal(lhs_array, rhs_array)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _as_legacy_v1_gate_config(payload: dict[str, object]) -> None:
    payload["schema"] = KONDO_GATE_LEGACY_V1_SCHEMA
    payload.pop("backward_admission_intent_semantics")
    payload["sparks_joy_semantics"] = "selected-for-backward-pass"


def test_checkpoint_resume_replays_the_next_backward_bit_exactly() -> None:
    actor, initial = _actor()
    first = actor.step(initial, _batch(actor, initial))
    checkpoint = actor.checkpoint_payload(first.state)
    restored_actor, restored_state = KondoSparseActor.from_checkpoint_payload(checkpoint)
    next_batch = _batch(actor, first.state, offset=0.125)

    uninterrupted = actor.step(first.state, next_batch)
    resumed = restored_actor.step(restored_state, next_batch)

    _assert_tree_equal(uninterrupted, resumed)
    assert actor.checkpoint_payload(uninterrupted.state) == restored_actor.checkpoint_payload(
        resumed.state
    )


def test_legacy_v1_embedded_gate_checkpoint_imports_and_normalizes() -> None:
    actor, initial = _actor()
    first = actor.step(initial, _batch(actor, initial))
    legacy = copy.deepcopy(actor.checkpoint_payload(first.state))
    config = cast(dict[str, object], legacy["config"])
    _as_legacy_v1_gate_config(cast(dict[str, object], config["gate"]))
    state = cast(dict[str, object], legacy["state"])
    gate_checkpoint = cast(dict[str, object], state["gate_checkpoint"])
    gate_checkpoint["schema"] = KONDO_GATE_LEGACY_V1_SCHEMA
    _as_legacy_v1_gate_config(cast(dict[str, object], gate_checkpoint["config"]))
    body = {key: value for key, value in legacy.items() if key != "checkpoint_sha256"}
    legacy["checkpoint_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()

    restored_actor, restored_state = KondoSparseActor.from_checkpoint_payload(legacy)

    _assert_tree_equal(restored_state, first.state)
    normalized = restored_actor.checkpoint_payload(restored_state)
    normalized_config = cast(dict[str, object], normalized["config"])
    normalized_gate = cast(dict[str, object], normalized_config["gate"])
    assert normalized_gate["schema"] == KONDO_GATE_SCHEMA
    assert "sparks_joy_semantics" not in normalized_gate
    normalized_state = cast(dict[str, object], normalized["state"])
    normalized_gate_checkpoint = cast(
        dict[str, object], normalized_state["gate_checkpoint"]
    )
    assert normalized_gate_checkpoint["schema"] == KONDO_GATE_SCHEMA


def test_corrupt_live_state_rejects_and_checkpoint_recovery_continues() -> None:
    actor, initial = _actor()
    first = actor.step(initial, _batch(actor, initial))
    checkpoint = actor.checkpoint_payload(first.state)
    corrupt_parameters = dataclasses.replace(
        first.state.parameters,
        hidden_weight=first.state.parameters.hidden_weight.at[0, 0].set(jnp.nan),
    )
    corrupt_state = dataclasses.replace(first.state, parameters=corrupt_parameters)
    healthy_next_batch = _batch(actor, first.state, offset=0.25)

    rejected = actor.step(corrupt_state, healthy_next_batch)

    assert not bool(rejected.transaction_applied)
    assert not bool(rejected.state_valid)
    assert not bool(rejected.screen.transaction_applied)
    _assert_tree_equal(rejected.state, corrupt_state)

    restored_actor, restored_state = KondoSparseActor.from_checkpoint_payload(checkpoint)
    recovered = restored_actor.step(restored_state, healthy_next_batch)
    expected = actor.step(first.state, healthy_next_batch)
    assert bool(recovered.transaction_applied)
    _assert_tree_equal(recovered, expected)


def test_recomputed_outer_digest_cannot_hide_inconsistent_state_accounting() -> None:
    actor, initial = _actor()
    first = actor.step(initial, _batch(actor, initial))
    tampered = copy.deepcopy(actor.checkpoint_payload(first.state))
    tampered_state = cast(dict[str, Any], tampered["state"])
    tampered_state["actor_backward_count"] = 2
    body = {key: value for key, value in tampered.items() if key != "checkpoint_sha256"}
    tampered["checkpoint_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()

    with pytest.raises(ValueError, match="checkpoint state is invalid"):
        KondoSparseActor.from_checkpoint_payload(tampered)


def test_sparse_orchestration_calls_audited_gather_before_value_and_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, initial = _actor()
    batch = _batch(actor, initial)
    events: list[str] = []
    original_gather = KondoGate.gather_sparse
    original_backward = sparse_actor_module.kondo_actor_backward_kernel

    def gather_wrapper(self: KondoGate, data: object, result: object) -> object:
        events.append("gather")
        return original_gather(self, data, result)  # type: ignore[arg-type]

    def backward_wrapper(parameters: object, fixed_batch: object) -> object:
        events.append("value_and_grad")
        return original_backward(parameters, fixed_batch)

    monkeypatch.setattr(KondoGate, "gather_sparse", gather_wrapper)
    monkeypatch.setattr(
        sparse_actor_module,
        "kondo_actor_backward_kernel",
        backward_wrapper,
    )

    result = actor.step(initial, batch)

    assert bool(result.transaction_applied)
    assert events == ["gather", "value_and_grad"]
    assert int(result.backward_batch_size) == 3 < 6


def test_protected_safety_change_is_full_fidelity_but_actor_gradient_is_unchanged() -> None:
    actor, initial = _actor()
    ordinary = _batch(actor, initial)
    changed_protected = dataclasses.replace(
        ordinary.protected,
        safety_features=ordinary.protected.safety_features.at[5, 1].add(7.0),
    )
    changed = dataclasses.replace(ordinary, protected=changed_protected)

    ordinary_result = actor.step(initial, ordinary)
    changed_result = actor.step(initial, changed)

    assert bool(ordinary_result.sparse_backward_used)
    assert bool(changed_result.sparse_backward_used)
    _assert_tree_equal(ordinary_result.gradient, changed_result.gradient)
    np.testing.assert_array_equal(ordinary_result.actor_loss, changed_result.actor_loss)
    assert not np.array_equal(
        np.asarray(ordinary_result.protected_digest),
        np.asarray(changed_result.protected_digest),
    )
    assert changed_result.protected.safety_features.shape == (6, 2)
    np.testing.assert_array_equal(
        changed_result.protected.safety_features,
        changed_protected.safety_features,
    )


def test_top_k_mode_consumes_no_rng_and_owns_no_parameter_initialization_rng() -> None:
    actor, initial = _actor()
    result = actor.step(initial, _batch(actor, initial))

    np.testing.assert_array_equal(
        jr.key_data(result.state.gate_state.rng_key),
        jr.key_data(initial.gate_state.rng_key),
    )
    assert int(result.screen.random_draw_count) == 0
