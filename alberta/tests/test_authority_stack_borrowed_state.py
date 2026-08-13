# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Single-owner borrowed-STOMP contracts across the option authority stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from test_authorized_option_replacement import _context

from alberta_framework.core.authorized_option_replacement import (
    AuthorizedOptionReplacementMetadataState,
)
from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallationMetadataState,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CumulantOptionSchedulerMetadataState,
)
from alberta_framework.core.options import STOMPState
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycleMetadataState,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


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


def test_complete_detached_authority_stack_has_zero_stomp_owners() -> None:
    context = _context()
    controller = context.controller
    source = context.pre_retirement_state
    metadata = controller.detach_borrowed_stomp(source)

    assert type(metadata) is AuthorizedOptionReplacementMetadataState
    assert type(metadata.scheduler_metadata) is CumulantOptionSchedulerMetadataState
    installation = metadata.scheduler_metadata.installation_metadata
    assert type(installation) is CumulantOptionInstallationMetadataState
    assert type(installation.lifecycle_metadata) is STOMPOptionLifecycleMetadataState
    assert not _contains_stomp_state(metadata)
    assert bool(controller.metadata_state_valid(metadata))

    stomp = source.scheduler_state.installation_state.lifecycle_state.stomp_state
    attached = controller.attach_borrowed_stomp(metadata, stomp)
    assert bool(attached.scheduler.installation.lifecycle.transaction_applied)
    assert bool(attached.scheduler.installation.transaction_applied)
    assert bool(attached.scheduler.transaction_applied)
    assert bool(attached.transaction_applied)
    assert not bool(attached.caller_authenticated)
    chex.assert_trees_all_equal(attached.state, source)


def test_complete_borrowed_attach_is_jit_safe_for_array_state() -> None:
    context = _context()
    controller = context.controller
    source = context.pre_retirement_state
    metadata = controller.detach_borrowed_stomp(source)
    stomp = source.scheduler_state.installation_state.lifecycle_state.stomp_state

    result = jax.jit(controller.attach_borrowed_stomp)(metadata, stomp)

    assert bool(result.metadata_valid)
    assert bool(result.binding_matches)
    assert bool(result.transaction_applied)


def test_coherently_rechecksummed_owner_misattribution_still_fails_attach() -> None:
    context = _context()
    controller = context.controller
    source = context.pre_retirement_state
    metadata = controller.detach_borrowed_stomp(source)
    scheduler_api = controller.scheduler
    installation_api = scheduler_api.installation
    installation = metadata.scheduler_metadata.installation_metadata
    lifecycle_api = installation_api.lifecycle.with_external_semantic_digests(
        installation.installed_semantic_digests
    )
    lifecycle = lifecycle_api._with_metadata_checksum(
        installation.lifecycle_metadata.replace(
            stomp_binding_checksum=(
                installation.lifecycle_metadata.stomp_binding_checksum.at[0].add(
                    jnp.uint32(1)
                )
            )
        )
    )
    installation = installation_api._with_metadata_checksum(
        installation.replace(lifecycle_metadata=lifecycle)
    )
    scheduler = scheduler_api._with_metadata_checksum(
        metadata.scheduler_metadata.replace(installation_metadata=installation)
    )
    tampered = controller._with_metadata_checksum(
        metadata.replace(scheduler_metadata=scheduler)
    )
    stomp = source.scheduler_state.installation_state.lifecycle_state.stomp_state

    attached = jax.jit(controller.attach_borrowed_stomp)(tampered, stomp)

    assert bool(controller.metadata_state_valid(tampered))
    assert not bool(attached.scheduler.installation.lifecycle.binding_matches)
    assert not bool(attached.transaction_applied)


def test_corrupt_full_outer_binding_cannot_be_laundered_by_detach() -> None:
    context = _context()
    controller = context.controller
    source = context.pre_retirement_state
    corrupt = source.replace(
        binding_checksum=source.binding_checksum.at[0].add(jnp.uint32(1))
    )
    metadata = controller.detach_borrowed_stomp(corrupt)
    stomp = source.scheduler_state.installation_state.lifecycle_state.stomp_state

    attached = controller.attach_borrowed_stomp(metadata, stomp)

    assert bool(controller.metadata_state_valid(metadata))
    assert not bool(attached.binding_matches)
    assert not bool(attached.transaction_applied)
