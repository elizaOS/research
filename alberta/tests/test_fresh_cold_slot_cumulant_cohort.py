# mypy: disable-error-code="arg-type,attr-defined,call-arg,type-var"
"""Fast contracts for opt-in v2 cold-slot cohort filtering."""

from __future__ import annotations

import copy
import dataclasses
import importlib
from typing import Any

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from test_cumulant_option_scheduler import _scheduler as _base_scheduler
from test_cumulant_subtask_discovery import (
    GENERATION,
    SOURCE_DIGEST,
    _config,
    _observe_inputs,
    _snapshot,
    _step,
)

from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallation,
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CumulantSubtaskDiscovery,
)
from alberta_framework.core.option_lifecycle_audit import OptionLifecycleAudit

pytestmark = pytest.mark.integration


def _api() -> tuple[type[Any], type[Any], type[Any]]:
    module = importlib.import_module(
        "alberta_framework.core.fresh_cold_slot_cumulant_cohort"
    )
    return (
        module.FreshColdSlotCumulantCohortFilter,
        module.FreshColdSlotCumulantCohortFilterConfig,
        module.FreshColdSlotCumulantCohortSource,
    )


def _installation(discovery: CumulantSubtaskDiscovery) -> CumulantOptionInstallation:
    base = _base_scheduler().installation
    stomp = dataclasses.replace(
        base.stomp_agent.config,
        subtask_specs=(),
        observation_dim=discovery.config.raw_feature_dim,
    )
    audit = OptionLifecycleAudit(
        dataclasses.replace(
            base.lifecycle.audit.config,
            outcome_dim=(
                discovery.config.raw_feature_dim + discovery.config.option_budget
            ),
            signature_scales=(
                (1.0,)
                * (5 + discovery.config.raw_feature_dim + discovery.config.option_budget)
            ),
        )
    )
    return CumulantOptionInstallation(
        discovery,
        stomp,
        audit,
        base.lifecycle.config,
        base.config,
    )


def _fixture(*, extra_feature: bool) -> tuple[Any, Any, Any, Any]:
    filter_type, config_type, source_type = _api()
    feature_descriptors = ((CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 20),)
    if extra_feature:
        feature_descriptors += ((CUMULANT_SOURCE_FEATURE_CHANGE, 0, -1, 21),)
    discovery = CumulantSubtaskDiscovery(
        _config(feature_change_descriptors=feature_descriptors)
    )
    installation = _installation(discovery)
    state = discovery.init(
        jr.key(7),
        semantic_generation=GENERATION,
        source_digest=SOURCE_DIGEST,
    )
    result = None
    for step in range(8):
        result = _step(discovery, state, step)
        state = result.state
    ready_step = 7
    assert result is not None and bool(result.discovered.ready)
    incumbent = result.discovered
    semantics = installation.semantic_digests_for_bundle(incumbent)
    successor = _snapshot(ready_step + 1)
    observed = _observe_inputs(ready_step)
    live = CumulantOptionLiveInputs(
        raw_features=successor["raw"],
        raw_available=jnp.ones_like(successor["raw"], dtype=jnp.bool_),
        controllable_events=successor["event"],
        controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        transition_atoms=successor["atom"],
        transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
        bottleneck_values=successor["bottleneck"],
        bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE_DIGEST,
        canonical_digest=result.state.canonical_digest,
        transition_id=observed["transition_id"],
        state_observation_count=result.state.observation_count,
    )
    source = source_type(
        discovery_result=result,
        installed_bundle=incumbent,
        installed_semantic_digests=semantics,
        installed_slot_mask=jnp.asarray([True, False, True, True], dtype=jnp.bool_),
        previous_raw_features=_snapshot(ready_step)["raw"],
        previous_raw_available=jnp.ones_like(successor["raw"], dtype=jnp.bool_),
        live_inputs=live,
    )
    config = config_type.from_installation(installation)
    return filter_type(installation, config), source, installation, config


def test_original_universe_has_no_same_family_fresh_candidate_and_is_idempotent() -> None:
    cohort_filter, source, _installation_api, config = _fixture(extra_feature=False)
    before = copy.deepcopy(source)
    prepared = cohort_filter.prepare(source)

    assert config.to_config()["schema_version"].endswith(".v2")
    assert config.to_config()["candidate_universe_schema"].endswith(".v2")
    assert type(config).from_config(config.to_config()) == config
    wrong_schema = config.to_config()
    wrong_schema["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        type(config).from_config(wrong_schema)
    assert bool(prepared.diagnostics.source_valid)
    assert bool(prepared.diagnostics.family_quota_layout_valid)
    assert not bool(prepared.diagnostics.alternate_available)
    assert not bool(prepared.diagnostics.candidate_ready)
    assert not bool(prepared.filtered_bundle.ready)
    np.testing.assert_array_equal(prepared.changed_slots, np.zeros((4,), dtype=bool))
    assert bool(cohort_filter.validate(prepared))
    chex.assert_trees_all_equal(source, before)

    replay = cohort_filter.prepare(source)
    chex.assert_trees_all_equal(replay, prepared)
    chex.assert_trees_all_equal(source, before)


def test_one_extra_feature_selects_only_cold_slot_and_preserves_source() -> None:
    cohort_filter, source, installation, _config_v2 = _fixture(extra_feature=True)
    before = copy.deepcopy(source)
    prepared = cohort_filter.prepare(source)

    assert bool(prepared.diagnostics.source_valid)
    assert bool(prepared.diagnostics.alternate_available)
    assert bool(prepared.diagnostics.same_family_selected)
    assert bool(prepared.diagnostics.family_quota_layout_valid)
    assert bool(prepared.diagnostics.live_slots_preserved)
    assert bool(prepared.diagnostics.target_semantic_fresh)
    assert bool(prepared.diagnostics.exact_target_change)
    assert bool(prepared.diagnostics.filtered_bundle_valid)
    assert bool(prepared.diagnostics.candidate_ready)
    assert bool(prepared.filtered_bundle.ready)
    assert int(prepared.target_slot) == 1
    assert int(prepared.selected_candidate_index) != int(
        source.installed_bundle.selected_candidate_indices[1]
    )
    np.testing.assert_array_equal(prepared.changed_slots, [False, True, False, False])
    np.testing.assert_array_equal(
        prepared.filtered_bundle.selected_family_ids,
        [0, 1, 2, 3],
    )
    live = np.asarray([True, False, True, True])
    np.testing.assert_array_equal(
        np.asarray(prepared.filtered_bundle.selected_candidate_indices)[live],
        np.asarray(source.installed_bundle.selected_candidate_indices)[live],
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.filtered_bundle.selected_descriptors)[live],
        np.asarray(source.installed_bundle.selected_descriptors)[live],
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.candidate_semantic_digests)[live],
        np.asarray(source.installed_semantic_digests)[live],
    )
    assert bool(
        installation.discovery.validate_proposal_bundle(
            prepared.filtered_bundle,
            semantic_generation=source.live_inputs.semantic_generation,
            source_digest=source.live_inputs.source_digest,
            canonical_digest=source.live_inputs.canonical_digest,
            transition_id=source.live_inputs.transition_id,
            state_observation_count=source.live_inputs.state_observation_count,
        )
    )
    assert bool(cohort_filter.validate(prepared))
    chex.assert_trees_all_equal(source, before)

    replay = cohort_filter.prepare(source)
    chex.assert_trees_all_equal(replay, prepared)
    chex.assert_trees_all_equal(source, before)


def test_tamper_and_checksum_valid_cross_family_splice_fail_closed() -> None:
    cohort_filter, source, installation, _config_v2 = _fixture(extra_feature=True)
    authentic = cohort_filter.prepare(source)
    tampered = dataclasses.replace(
        authentic,
        prepared_checksum=authentic.prepared_checksum + jnp.uint32(1),
    )
    assert not bool(cohort_filter.validate(tampered))

    event_index = int(source.installed_bundle.selected_candidate_indices[0])
    candidates = installation.discovery.config.candidate_descriptors
    forged_indices = source.installed_bundle.selected_candidate_indices.at[1].set(event_index)
    forged_descriptors = source.installed_bundle.selected_descriptors.at[1].set(
        jnp.asarray(candidates[event_index], dtype=jnp.int32)
    )
    forged_families = source.installed_bundle.selected_family_ids.at[1].set(0)
    forged = dataclasses.replace(
        source.installed_bundle,
        selected_candidate_indices=forged_indices,
        selected_family_ids=forged_families,
        selected_descriptors=forged_descriptors,
        binding_digest=jnp.zeros((2,), dtype=jnp.uint32),
    )
    forged = dataclasses.replace(
        forged,
        binding_digest=installation.discovery._bundle_checksum(forged),
    )
    assert bool(
        installation.discovery.validate_proposal_bundle(
            forged,
            semantic_generation=forged.semantic_generation,
            source_digest=forged.source_digest,
            canonical_digest=forged.canonical_digest,
            transition_id=forged.transition_id,
            state_observation_count=forged.state_observation_count,
        )
    )
    cross_family_source = dataclasses.replace(
        source,
        installed_bundle=forged,
        installed_semantic_digests=installation.semantic_digests_for_bundle(forged),
    )
    before = copy.deepcopy(cross_family_source)
    refused = cohort_filter.prepare(cross_family_source)
    assert not bool(refused.diagnostics.family_quota_layout_valid)
    assert not bool(refused.diagnostics.source_valid)
    assert not bool(refused.diagnostics.candidate_ready)
    assert not bool(refused.filtered_bundle.ready)
    chex.assert_trees_all_equal(cross_family_source, before)
