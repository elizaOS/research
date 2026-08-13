# mypy: disable-error-code="arg-type,attr-defined,operator"
"""Adversarial contracts for the generic HCCL B-to-M-to-P projection seam."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
)
from alberta_framework.core.hccl_authenticated_bmp_projection import (
    HCCL_AUTHENTICATED_BMP_PROJECTION_SCIENTIFIC_PROMOTION_ALLOWED,
    HCCLAuthenticatedBMPProjection,
    HCCLAuthenticatedBMPProjectionConfig,
)
from alberta_framework.core.hccl_continual_dyad_factory import (
    build_hccl_continual_dyad_config,
)
from alberta_framework.core.prototype_agent import PrototypeAgent

_SEAM_OWNER = (101, 102, 103, 104, 105, 106, 107, 108)
_MEMORY_OWNER = jnp.arange(1_001, 1_009, dtype=jnp.uint32)
_PLANNER_OWNER = jnp.arange(2_001, 2_009, dtype=jnp.uint32)


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
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


@pytest.fixture(scope="module")
def fixture() -> Iterator[tuple[HCCLAuthenticatedBMPProjection, Any]]:
    coordinator_config = build_hccl_continual_dyad_config().agent_0.coordinator
    coordinator = ExternalLearnedStateRouterAuditCoordinator(coordinator_config)
    source = coordinator.start(
        coordinator.init(jr.key(801)),
        jnp.zeros((19,), dtype=jnp.float32),
    )
    seam = HCCLAuthenticatedBMPProjection(
        coordinator,
        HCCLAuthenticatedBMPProjectionConfig(owner_digest=_SEAM_OWNER),
    )
    assert bool(coordinator.state_valid(source))
    yield seam, source


def _other(action: Any) -> Any:
    return jnp.asarray(1 - int(action), dtype=jnp.int32)


def _valid_prepared(
    seam: HCCLAuthenticatedBMPProjection,
    source: Any,
) -> tuple[Any, Any]:
    memory = seam.prepare_memory(
        source,
        proposed_action=_other(source.current_action),
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    prepared = seam.prepare_planner(
        memory,
        proposed_action=source.current_action,
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_PLANNER_OWNER,
    )
    return memory, prepared


@pytest.mark.unit
def test_config_is_strict_and_declares_no_authority() -> None:
    config = HCCLAuthenticatedBMPProjectionConfig(owner_digest=_SEAM_OWNER)
    assert HCCLAuthenticatedBMPProjectionConfig.from_config(config.to_config()) == config
    payload = config.to_config()
    assert payload["external_owner_words_caller_authenticated"] is False
    assert payload["dispatch_authority"] is False
    assert payload["artifact_authority"] is False
    assert payload["evidence_authority"] is False
    assert payload["promotion_authority"] is False
    assert HCCL_AUTHENTICATED_BMP_PROJECTION_SCIENTIFIC_PROMOTION_ALLOWED is False
    with pytest.raises(ValueError):
        HCCLAuthenticatedBMPProjectionConfig(owner_digest=(0,) * 8)
    with pytest.raises(ValueError):
        HCCLAuthenticatedBMPProjectionConfig.from_config(
            {**payload, "owner_digest": [True, 102, 103, 104, 105, 106, 107, 108]}
        )


@pytest.mark.unit
@pytest.mark.slow
def test_public_chain_makes_memory_and_planner_actions_authoritative_for_next_cache(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
) -> None:
    seam, source = fixture
    memory, prepared = _valid_prepared(seam, source)
    receipt = seam.integrity_receipt(prepared)
    adopted = seam.adopt(source, prepared, receipt)

    base = int(source.current_action)
    assert bool(memory.phase_valid)
    assert int(memory.base_action) == base
    assert int(memory.proposed_action) == 1 - base
    assert int(memory.memory_action) == 1 - base
    assert int(memory.replacement.state.current_action) == 1 - base
    assert bool(prepared.preparation_valid)
    assert int(prepared.planner_replacement.state.current_action) == base
    assert int(prepared.binding.base_action) == base
    assert int(prepared.binding.memory_action) == 1 - base
    assert int(prepared.binding.final_action) == base
    assert bool(prepared.binding.memory_consumed)
    assert bool(prepared.binding.planner_consumed)
    assert bool(adopted.update_applied)
    assert int(adopted.state.current_action) == base
    assert int(
        adopted.state.inner_state.prototype_state.current_action
    ) == int(prepared.binding.final_action)
    assert bool(seam.binding_valid(adopted.state, prepared.binding))
    assert _tree_exact_equal(source, memory.source_coordinator_state)


@pytest.mark.unit
@pytest.mark.slow
def test_consumed_false_is_an_explicit_noop_but_still_evaluates_fixed_attempt(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
) -> None:
    seam, source = fixture
    memory = seam.prepare_memory(
        source,
        proposed_action=source.current_action,
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(False, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    prepared = seam.prepare_planner(
        memory,
        proposed_action=memory.memory_action,
        consumed=jnp.asarray(False, dtype=jnp.bool_),
        external_owner_words=_PLANNER_OWNER,
    )
    adopted = seam.adopt(source, prepared, seam.integrity_receipt(prepared))

    assert bool(memory.replacement.committed)
    assert bool(prepared.planner_replacement.committed)
    assert not bool(memory.consumed)
    assert not bool(prepared.planner_consumed)
    assert int(memory.base_action) == int(memory.memory_action)
    assert int(prepared.binding.memory_action) == int(prepared.binding.final_action)
    assert bool(adopted.update_applied)
    assert _tree_exact_equal(adopted.state, source)

    ignored_memory = seam.prepare_memory(
        source,
        proposed_action=_other(source.current_action),
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(False, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    ignored_planner = seam.prepare_planner(
        ignored_memory,
        proposed_action=_other(ignored_memory.memory_action),
        consumed=jnp.asarray(False, dtype=jnp.bool_),
        external_owner_words=_PLANNER_OWNER,
    )
    ignored_adoption = seam.adopt(
        source,
        ignored_planner,
        seam.integrity_receipt(ignored_planner),
    )
    assert bool(ignored_memory.phase_valid)
    assert bool(ignored_planner.preparation_valid)
    assert bool(ignored_memory.replacement_candidate_committed)
    assert not bool(ignored_memory.replacement_selected)
    assert bool(ignored_planner.planner_replacement_candidate_committed)
    assert not bool(ignored_planner.planner_replacement_selected)
    assert int(ignored_memory.proposed_action) != int(ignored_memory.memory_action)
    assert int(ignored_planner.planner_proposed_action) != int(
        ignored_planner.binding.final_action
    )
    assert int(ignored_memory.memory_action) == int(source.current_action)
    assert int(ignored_planner.binding.final_action) == int(source.current_action)
    assert bool(ignored_adoption.update_applied)
    assert _tree_exact_equal(ignored_adoption.state, source)


@pytest.mark.unit
@pytest.mark.slow
def test_exact_work_has_two_replacements_and_no_hidden_donor_work(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seam, source = fixture
    original = PrototypeAgent.replace_cached_primitive_action
    calls = 0

    def spy(self: PrototypeAgent, *args: object, **kwargs: object) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PrototypeAgent, "replace_cached_primitive_action", spy)
    _, prepared = _valid_prepared(seam, source)
    assert calls == 2
    receipt = seam.integrity_receipt(prepared)
    adopted = seam.adopt(source, prepared, receipt)
    assert calls == 2
    work = prepared.work
    assert int(work.prototype_memory_replacement_calls) == 1
    assert int(work.prototype_planner_replacement_calls) == 1
    assert int(work.learner_updates) == 0
    assert int(work.model_updates) == 0
    assert int(work.environment_proposals) == 0
    assert int(work.replay_updates) == 0
    assert int(work.rng_draws) == 0
    assert int(adopted.adoption_work.prototype_replacement_calls) == 0
    assert int(adopted.adoption_work.learner_updates) == 0
    assert int(adopted.adoption_work.model_updates) == 0
    assert int(adopted.adoption_work.rng_draws) == 0


@pytest.mark.unit
@pytest.mark.slow
def test_foreign_source_and_foreign_receipt_return_complete_source(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
) -> None:
    seam, source = fixture
    _, prepared = _valid_prepared(seam, source)
    receipt = seam.integrity_receipt(prepared)
    foreign = seam.coordinator.start(
        seam.coordinator.init(jr.key(802)),
        jnp.zeros((19,), dtype=jnp.float32),
    )
    foreign_result = seam.adopt(foreign, prepared, receipt)
    assert not bool(foreign_result.update_applied)
    assert bool(foreign_result.complete_source_returned)
    assert _tree_exact_equal(foreign_result.state, foreign)

    memory_2 = seam.prepare_memory(
        source,
        proposed_action=source.current_action,
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    prepared_2 = seam.prepare_planner(
        memory_2,
        proposed_action=memory_2.memory_action,
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_PLANNER_OWNER,
    )
    wrong_receipt = seam.integrity_receipt(prepared_2)
    wrong_result = seam.adopt(source, prepared, wrong_receipt)
    assert not bool(wrong_result.update_applied)
    assert _tree_exact_equal(wrong_result.state, source)


@pytest.mark.unit
@pytest.mark.slow
def test_tampered_or_coherently_resealed_binding_fails_closed(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
) -> None:
    seam, source = fixture
    _, prepared = _valid_prepared(seam, source)
    tampered_binding = prepared.binding.replace(
        memory_action=prepared.binding.base_action,
    )
    tampered_binding = tampered_binding.replace(
        content_token=seam._binding_token(tampered_binding)
    )
    tampered = prepared.replace(binding=tampered_binding)
    tampered = tampered.replace(content_token=seam._prepared_token(tampered))
    receipt = seam.integrity_receipt(tampered)
    result = seam.adopt(source, tampered, receipt)
    assert not bool(receipt.integrity_bound)
    assert not bool(result.update_applied)
    assert _tree_exact_equal(result.state, source)


@pytest.mark.unit
@pytest.mark.parametrize("layer", ["memory", "planner"])
@pytest.mark.slow
def test_invalid_proposal_veto_rolls_back_bit_exact(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
    layer: str,
) -> None:
    seam, source = fixture
    memory = seam.prepare_memory(
        source,
        proposed_action=(
            jnp.asarray(2, dtype=jnp.int32)
            if layer == "memory"
            else source.current_action
        ),
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    prepared = seam.prepare_planner(
        memory,
        proposed_action=(
            jnp.asarray(2, dtype=jnp.int32)
            if layer == "planner"
            else memory.memory_action
        ),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_PLANNER_OWNER,
    )
    result = seam.adopt(source, prepared, seam.integrity_receipt(prepared))
    assert not bool(prepared.preparation_valid)
    assert not bool(result.update_applied)
    assert bool(result.complete_source_returned)
    assert _tree_exact_equal(result.state, source)


@pytest.mark.unit
@pytest.mark.slow
def test_zero_duplicate_or_projection_owner_words_cannot_bind_external_authority(
    fixture: tuple[HCCLAuthenticatedBMPProjection, Any],
) -> None:
    seam, source = fixture
    zero_memory = seam.prepare_memory(
        source,
        proposed_action=source.current_action,
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=jnp.zeros((8,), dtype=jnp.uint32),
    )
    assert not bool(zero_memory.phase_valid)

    memory, _ = _valid_prepared(seam, source)
    duplicate = seam.prepare_planner(
        memory,
        proposed_action=memory.memory_action,
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=_MEMORY_OWNER,
    )
    assert not bool(duplicate.external_owners_distinct)
    assert not bool(duplicate.preparation_valid)

    own_words = jnp.asarray(_SEAM_OWNER, dtype=jnp.uint32)
    own_memory = seam.prepare_memory(
        source,
        proposed_action=source.current_action,
        hard_action_mask=jnp.ones((2,), dtype=jnp.bool_),
        consumed=jnp.asarray(True, dtype=jnp.bool_),
        external_owner_words=own_words,
    )
    assert not bool(own_memory.external_owner_words_valid)
    assert not bool(own_memory.phase_valid)
