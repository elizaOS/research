# mypy: disable-error-code="attr-defined,call-arg"
"""JIT, scan, checkpoint, and public-surface integration contracts."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as public_api
import alberta_framework.core as core_api
from alberta_framework.core import prototype_stomp_calibrated_search as adapter_module
from alberta_framework.core.calibrated_extended_search_control import (
    SEARCH_MODE_COMBINED,
    CalibratedExtendedSearchControlConfig,
)
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
)
from alberta_framework.core.prototype_stomp_calibrated_search import (
    PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_HOST_ONLY,
    PROTOTYPE_STOMP_CALIBRATED_SEARCH_REBIND_HOST_ONLY,
    PrototypeSTOMPCalibratedSearchAgent,
    PrototypeSTOMPCalibratedSearchConfig,
    PrototypeSTOMPCalibratedSearchState,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ANCHORS = jnp.asarray(
    ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
    dtype=jnp.float32,
)
ACTIVE = jnp.ones((4,), dtype=jnp.bool_)
SOURCE = jnp.asarray((0xCA11, 0xB1AD), dtype=jnp.uint32)


def _config() -> PrototypeSTOMPCalibratedSearchConfig:
    prototype = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=100.0,
                        pseudo_reward_scale=1.0,
                        max_option_steps=4,
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.01,
                base_avg_reward_step_size=0.01,
                option_step_size=0.01,
                option_avg_reward_step_size=0.01,
                option_model_decay=0.0,
                option_model_step_size=0.2,
                option_planning_backups_per_step=0,
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.02,
            sparsity=0.0,
            use_layer_norm=False,
        ),
        n_dreams_per_step=0,
        auto_curate_every=0,
    )
    search = CalibratedExtendedSearchControlConfig(
        mode=SEARCH_MODE_COMBINED,
        observation_dim=2,
        anchor_capacity=4,
        n_primitive_actions=2,
        n_options=1,
        backup_budget=1,
        calibration_evidence_floor=2,
        model_support_floor=1,
        confidence_scale=1.0,
        support_prior=1.0,
        model_error_scale=10.0,
        backup_step_size=0.1,
        max_observations=16,
    )
    return PrototypeSTOMPCalibratedSearchConfig(
        prototype=prototype,
        search=search,
    )


def _started_primitive() -> tuple[
    PrototypeSTOMPCalibratedSearchAgent,
    PrototypeSTOMPCalibratedSearchState,
]:
    adapter = PrototypeSTOMPCalibratedSearchAgent(_config())
    state = adapter.init(
        jr.key(19),
        anchor_bank=ANCHORS,
        anchor_active=ACTIVE,
        source_digest=SOURCE,
        representation_generation=3,
        lifecycle_id=jnp.asarray((41, 43), dtype=jnp.uint32),
    )
    prototype = state.prototype
    oak = cast(OaKState, prototype.oak_state)
    learner = oak.stomp_state.base_learner_state
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
            biases=tuple(
                jnp.full_like(bias, 20.0 if index == 0 else -20.0)
                for index, bias in enumerate(learner.head_params.biases)
            ),
        )
    )
    oak = cast(
        OaKState,
        oak.replace(
            stomp_state=oak.stomp_state.replace(base_learner_state=learner)
        ),
    )
    prototype = cast(PrototypeAgentState, prototype.replace(oak_state=oak))
    # Rebinding is intentionally a host-only control-plane boundary.
    rebound = adapter.rebind(
        state,
        prototype_state=prototype,
        source_digest=SOURCE,
        representation_generation=3,
    )
    assert bool(rebound.transaction_applied)
    started = adapter.start(rebound.state, ANCHORS[0])
    assert bool(started.diagnostics.arm_applied)
    return adapter, started.state


def _transition(
    state: PrototypeSTOMPCalibratedSearchState,
    future: jax.Array,
) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.prototype.current_raw_observation,
        action=state.prototype.current_action,
        decision_id=state.prototype.current_decision_id,
        reward=jnp.asarray(0.25, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=future,
        next_decision_observation=future,
    )


def _assert_compiled_tree_parity(left: object, right: object) -> None:
    """Compare semantics at 1e-6 while exempting exact-bit float fingerprints."""

    chex.assert_trees_all_equal_structs(left, right)
    left_path_leaves, _ = jax.tree_util.tree_flatten_with_path(left)
    right_leaves = jax.tree.leaves(right)
    for (path, left_leaf), right_leaf in zip(
        left_path_leaves, right_leaves, strict=True
    ):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        path_text = jax.tree_util.keystr(path)
        if "checksum" in path_text or "pending_cache_digest" in path_text:
            assert left_array.shape == right_array.shape
            assert left_array.dtype == right_array.dtype
        elif jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif jnp.issubdtype(left_array.dtype, jnp.inexact):
            np.testing.assert_allclose(
                left_array,
                right_array,
                rtol=1e-6,
                atol=1e-6,
            )
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_eager_jit_update_and_scan_have_primitive_lane_parity() -> None:
    adapter, state = _started_primitive()
    first_transition = _transition(state, ANCHORS[1])

    eager_first = adapter.update_transition(state, first_transition)
    compiled_first = jax.jit(adapter.update_transition)(state, first_transition)
    _assert_compiled_tree_parity(eager_first, compiled_first)
    assert bool(adapter.validate_state(eager_first.state))
    assert bool(adapter.validate_state(compiled_first.state))

    second_transition = _transition(eager_first.state, ANCHORS[2])
    eager_second = adapter.update_transition(eager_first.state, second_transition)
    compiled_second_transition = _transition(compiled_first.state, ANCHORS[2])
    compiled_second = jax.jit(adapter.update_transition)(
        compiled_first.state,
        compiled_second_transition,
    )
    _assert_compiled_tree_parity(eager_second, compiled_second)
    assert bool(compiled_second.diagnostics.transaction_committed)
    assert bool(adapter.validate_state(compiled_second.state))
    transitions = jax.tree.map(
        lambda first, second: jnp.stack((first, second)),
        first_transition,
        second_transition,
    )
    scanned = adapter.scan_transitions(state, transitions)
    compiled_scan = jax.jit(adapter.scan_transitions)(state, transitions)

    _assert_compiled_tree_parity(scanned.state, eager_second.state)
    _assert_compiled_tree_parity(compiled_scan, scanned)
    assert bool(adapter.validate_state(scanned.state))
    assert bool(adapter.validate_state(compiled_scan.state))
    np.testing.assert_array_equal(
        scanned.actions,
        jnp.stack((eager_first.prototype.action, eager_second.prototype.action)),
    )
    assert bool(jnp.all(scanned.prototype_transition_applied))
    assert bool(jnp.all(scanned.natural_resolutions))
    assert bool(jnp.all(scanned.transaction_committed))


def test_host_checkpoint_rejects_sha_config_source_and_generation_tamper() -> None:
    adapter, state = _started_primitive()
    assert PROTOTYPE_STOMP_CALIBRATED_SEARCH_CHECKPOINT_HOST_ONLY
    assert PROTOTYPE_STOMP_CALIBRATED_SEARCH_REBIND_HOST_ONLY
    budget = adapter.resource_budget(state)
    assert budget.checkpoint_host_only
    assert budget.rebind_host_only

    # This unkeyed digest detects accidental corruption; it is not authenticity.
    payload = adapter.checkpoint_payload(state)
    restored = adapter.restore_checkpoint(
        payload,
        source_digest=SOURCE,
        representation_generation=3,
    )
    chex.assert_trees_all_equal(restored, state)

    state_tamper = dict(payload)
    state_tamper["state"] = state.replace(revision=state.revision + jnp.int32(1))
    with pytest.raises(ValueError, match="SHA differs"):
        adapter.restore_checkpoint(
            state_tamper,
            source_digest=SOURCE,
            representation_generation=3,
        )

    config_tamper = dict(payload)
    config_tamper["config"] = dataclasses.replace(
        adapter.config,
        search=dataclasses.replace(adapter.config.search, max_observations=17),
    ).to_config()
    with pytest.raises(ValueError, match="config differs"):
        adapter.restore_checkpoint(
            config_tamper,
            source_digest=SOURCE,
            representation_generation=3,
        )

    with pytest.raises(ValueError, match="stale, or rebound"):
        adapter.restore_checkpoint(
            payload,
            source_digest=SOURCE.at[0].add(jnp.uint32(1)),
            representation_generation=3,
        )
    with pytest.raises(ValueError, match="stale, or rebound"):
        adapter.restore_checkpoint(
            payload,
            source_digest=SOURCE,
            representation_generation=4,
        )


def test_corrupted_persistent_composition_fails_closed() -> None:
    adapter, state = _started_primitive()
    corrupted = state.replace(
        binding_checksum=state.binding_checksum.at[0].add(jnp.uint32(1))
    )
    result = adapter.update_transition(corrupted, _transition(corrupted, ANCHORS[1]))

    assert not bool(result.diagnostics.composed_state_valid_before)
    assert not bool(result.diagnostics.transaction_committed)
    assert int(result.prototype.action) == -1
    assert not bool(result.decision.armed)
    chex.assert_trees_all_equal(result.state, corrupted)
    with pytest.raises(ValueError, match="corrupted composed state"):
        adapter.rebind(
            corrupted,
            prototype_state=corrupted.prototype,
            source_digest=SOURCE,
            representation_generation=3,
        )


def test_adapter_public_exports_are_identical_at_core_and_package_roots() -> None:
    for name in adapter_module.__all__:
        assert getattr(core_api, name) is getattr(adapter_module, name)
        assert getattr(public_api, name) is getattr(adapter_module, name)
