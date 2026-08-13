"""Static contracts for the predeclared compositional future-utility panel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, cast

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation.compositional_control_life_development import (
    DEFAULT_CONSUMED_SEED,
    DEFAULT_PHASE_LENGTHS,
    PHASE_ORDER,
    build_default_protocol,
    learner_config_for_arm,
)
from alberta_framework.evaluation.compositional_future_utility_development import (
    ARM_NAMES,
    CONSUMED_KEY_MANIFEST,
    CONSUMED_STREAM_SHA256,
    PROTOCOL_SCHEMA,
    RESOURCE_ACCOUNTING_SCOPE,
    RUNTIME_IDENTITY_SCOPE,
    CompositionalFutureUtilityProtocol,
    _arm_learner_config,
    _build_arm_learner,
    _ProcessAttemptLatch,
    _source_arrays_bound,
    _static_preflight,
    logical_work_per_arm,
)

pytestmark = pytest.mark.unit

_LEFT_PACK = "dovetail_coverage_ancestor_headroom_leftpack"


def test_protocol_is_the_exact_consumed_8998_step_control_life() -> None:
    protocol = CompositionalFutureUtilityProtocol()
    payload = protocol.to_config()

    assert protocol.schema_version == PROTOCOL_SCHEMA
    assert protocol.development_seed == DEFAULT_CONSUMED_SEED
    assert protocol.phase_order == PHASE_ORDER
    assert protocol.phase_lengths == DEFAULT_PHASE_LENGTHS
    assert protocol.total_steps == 8_998
    assert protocol.curation_interval == 32
    assert protocol.left_pack_source_arm == _LEFT_PACK
    assert payload["development_root_already_consumed"] is True
    assert payload["new_seed_or_initialization_drawn"] is False
    assert payload["search_performed"] is False
    assert payload["winner_selection_allowed"] is False
    assert payload["writer_available"] is False
    assert payload["learner_observation_fields"] == ["raw_rademacher_values"]
    assert payload["learner_feedback_fields"] == ["selected_action_reward"]
    assert payload["evaluator_only_fields"] == [
        "phase_name",
        "phase_boundary",
        "target_expression",
        "counterfactual_action_reward",
    ]
    assert CompositionalFutureUtilityProtocol.from_config(payload) == protocol

    with pytest.raises(ValueError, match="frozen"):
        CompositionalFutureUtilityProtocol(development_seed=1)
    with pytest.raises(ValueError, match="frozen"):
        CompositionalFutureUtilityProtocol(phase_lengths=(1,) * 10)


def test_only_the_three_predeclared_contribution_configs_exist() -> None:
    assert ARM_NAMES == (
        "contribution_mix0_decay095",
        "contribution_mix1_decay0",
        "contribution_mix1_decay095",
    )
    configs = {name: _arm_learner_config(name) for name in ARM_NAMES}
    expected = {
        ARM_NAMES[0]: (0.0, 0.95),
        ARM_NAMES[1]: (1.0, 0.0),
        ARM_NAMES[2]: (1.0, 0.95),
    }
    base = learner_config_for_arm(_LEFT_PACK)
    common_departures = {
        "candidate_scoring_mode",
        "candidate_novelty_admission_bonus",
        "future_utility_trace_mode",
    }
    intervention_fields = {"future_utility_mix", "future_utility_trace_decay"}

    for name, config in configs.items():
        mix, decay = expected[name]
        assert config["future_utility_mix"] == mix
        assert config["future_utility_trace_decay"] == decay
        assert config["future_utility_trace_mode"] == "contribution"
        assert config["future_utility_normalization"] == "none"
        assert config["future_utility_rare_task_power"] == 0.0
        assert {key for key in config if config[key] != base[key]} <= (
            common_departures | intervention_fields
        )
        assert config["topology_headroom_reserve"] is True
        assert config["topology_left_pack_destinations"] is True
        assert config["candidate_scoring_mode"] == "legacy"
        assert config["candidate_novelty_admission_bonus"] == 0.0

    for left_name, right_name in zip(ARM_NAMES, ARM_NAMES[1:], strict=False):
        left = configs[left_name]
        right = configs[right_name]
        assert {key for key in left if left[key] != right[key]} <= intervention_fields


def test_corrected_static_preflight_closes_the_exact_intervention_before_scan() -> None:
    preflight = _static_preflight()

    assert preflight["static_audit_performed"] is True
    assert preflight["panel_executed_during_preflight"] is False
    assert preflight["core_constraint_changed"] is False
    assert preflight["active_mixed_utility_overwrite_disabled"] is True
    assert preflight["candidate_mixed_utility_overwrite_disabled"] is True
    assert preflight["future_utility_reaches_active_and_candidate_ranking"] is True
    assert preflight[
        "trace_decay_reaches_active_and_candidate_contribution_traces"
    ] is True
    assert preflight["only_varying_config_fields"] == {
        "future_utility_mix": {
            ARM_NAMES[0]: 0.0,
            ARM_NAMES[1]: 1.0,
            ARM_NAMES[2]: 1.0,
        },
        "future_utility_trace_decay": {
            ARM_NAMES[0]: 0.95,
            ARM_NAMES[1]: 0.0,
            ARM_NAMES[2]: 0.95,
        },
    }
    assert preflight["common_base_departures_from_historical_left_pack"] == {
        "candidate_scoring_mode": {
            "historical_left_pack": "energy_novelty",
            "corrected_common_base": "legacy",
        },
        "candidate_novelty_admission_bonus": {
            "historical_left_pack": 1.0,
            "corrected_common_base": 0.0,
        },
        "future_utility_trace_mode": {
            "historical_left_pack": "marginal",
            "corrected_common_base": "contribution",
        },
    }
    assert "not an environment" in RUNTIME_IDENTITY_SCOPE
    assert "excludes source arrays" in RESOURCE_ACCOUNTING_SCOPE


def _two_update_corrected_probe(
    arm_name: str,
) -> tuple[CompositionalFeatureLearner, CompositionalFeatureState]:
    learner = _build_arm_learner(arm_name)
    learner_key = jr.wrap_key_data(
        jnp.asarray(CONSUMED_KEY_MANIFEST["learner_genesis"], dtype=jnp.uint32),
        impl="threefry2x32",
    )
    state = cast(
        CompositionalFeatureState,
        learner.init(6, learner_key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0,
            uptime_s=0.0,
        ),
    )
    observation = jnp.asarray(
        (0.2, -0.35, 0.5, -0.65, 0.8, -0.95), dtype=jnp.float32
    )
    probes = (
        (
            observation,
            jnp.asarray((1.0, jnp.nan), dtype=jnp.float32),
        ),
        (
            observation,
            jnp.asarray((0.75, jnp.nan), dtype=jnp.float32),
        ),
    )
    for observation, targets in probes:
        state = learner.update(state, observation, targets).state
    return learner, state


def test_corrected_legacy_mix_and_decay_reach_utility_and_trace_state() -> None:
    mix0_learner, mix0_state = _two_update_corrected_probe(ARM_NAMES[0])
    mix1_learner, mix1_state = _two_update_corrected_probe(ARM_NAMES[2])

    assert not bool(jnp.array_equal(mix0_state.utilities, mix1_state.utilities))
    assert not bool(
        jnp.array_equal(mix0_state.candidate_utilities, mix1_state.candidate_utilities)
    )
    chex.assert_trees_all_equal(
        mix0_state.utility_contribution_trace,
        mix1_state.utility_contribution_trace,
    )
    mix0_ranking = mix0_learner.ranking_diagnostics(mix0_state, 6)
    mix1_ranking = mix1_learner.ranking_diagnostics(mix1_state, 6)
    assert not bool(
        jnp.array_equal(mix0_ranking.direct_active_scores, mix1_ranking.direct_active_scores)
    )
    assert not bool(
        jnp.array_equal(
            mix0_ranking.direct_candidate_scores,
            mix1_ranking.direct_candidate_scores,
        )
    )

    decay0_learner, decay0_state = _two_update_corrected_probe(ARM_NAMES[1])
    decay95_learner, decay95_state = _two_update_corrected_probe(ARM_NAMES[2])
    assert not bool(
        jnp.array_equal(
            decay0_state.utility_contribution_trace,
            decay95_state.utility_contribution_trace,
        )
    )
    assert not bool(
        jnp.array_equal(
            decay0_state.candidate_utility_contribution_trace,
            decay95_state.candidate_utility_contribution_trace,
        )
    )
    decay0_ranking = decay0_learner.ranking_diagnostics(decay0_state, 6)
    decay95_ranking = decay95_learner.ranking_diagnostics(decay95_state, 6)
    assert not bool(
        jnp.array_equal(
            decay0_ranking.direct_active_scores,
            decay95_ranking.direct_active_scores,
        )
    )


def test_process_attempt_latch_seals_success_failure_and_concurrent_callers() -> None:
    entered = Event()
    release = Event()
    second_started = Event()
    successful_calls: list[int] = []

    def successful_builder() -> str:
        successful_calls.append(1)
        entered.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release the process-attempt builder")
        return "sole-result"

    def concurrent_successful_get(latch: _ProcessAttemptLatch) -> str:
        second_started.set()
        return latch.get()

    latch = _ProcessAttemptLatch(successful_builder)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(latch.get)
        assert entered.wait(timeout=5.0)
        second = executor.submit(concurrent_successful_get, latch)
        assert second_started.wait(timeout=5.0)
        release.set()
        assert first.result(timeout=5.0) == "sole-result"
        assert second.result(timeout=5.0) == "sole-result"
    assert successful_calls == [1]
    assert latch.get() == "sole-result"

    class PanelFailure(BaseException):
        pass

    failure_entered = Event()
    failure_release = Event()
    failure_waiter_started = Event()
    failed_calls: list[int] = []

    def failed_builder() -> str:
        failed_calls.append(1)
        failure_entered.set()
        if not failure_release.wait(timeout=5.0):
            raise TimeoutError("test did not release the failed process-attempt builder")
        raise PanelFailure("sealed failure")

    def concurrent_failed_get(latch: _ProcessAttemptLatch) -> str:
        failure_waiter_started.set()
        return latch.get()

    failed_latch = _ProcessAttemptLatch(failed_builder)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_failure = executor.submit(failed_latch.get)
        assert failure_entered.wait(timeout=5.0)
        second_failure = executor.submit(concurrent_failed_get, failed_latch)
        assert failure_waiter_started.wait(timeout=5.0)
        failure_release.set()
        with pytest.raises(PanelFailure, match="sealed failure"):
            first_failure.result(timeout=5.0)
        with pytest.raises(RuntimeError, match="sealed after failure") as sealed:
            second_failure.result(timeout=5.0)
    assert isinstance(sealed.value.__cause__, PanelFailure)
    with pytest.raises(RuntimeError, match="sealed after failure"):
        failed_latch.get()
    assert failed_calls == [1]


def test_source_binding_reuses_keys_stream_shapes_and_schedule_exactly() -> None:
    protocol = CompositionalFutureUtilityProtocol()
    actual = _source_arrays_bound(protocol)
    expected_protocol = build_default_protocol()

    expected_key_manifest = {
        name: list(words) for name, words in CONSUMED_KEY_MANIFEST.items()
    }
    assert expected_key_manifest == {
        "exploration": [2_227_216_649, 3_977_711_669],
        "learner_genesis": [2_002_082_676, 3_427_004_161],
        "observations": [2_316_273_231, 3_036_545_927],
        "random_actions": [382_045_127, 333_255_797],
        "root": [0, 329_631_721],
    }
    assert actual.key_manifest == expected_key_manifest
    assert actual.key_manifest["root"] == [0, DEFAULT_CONSUMED_SEED]
    assert actual.stream_sha256 == CONSUMED_STREAM_SHA256
    assert actual.stream_sha256 == (
        "02fd5efbbb304b624fcfd29e259c361d5048233817e896300057d8e36f3fc036"
    )
    assert actual.observations.shape == (expected_protocol.total_steps, 6)
    assert actual.phase_indices.shape == (expected_protocol.total_steps,)
    assert actual.exploration_mask.shape == (expected_protocol.total_steps,)
    assert actual.random_actions.shape == (expected_protocol.total_steps,)
    assert actual.observations.dtype == jnp.float32
    assert actual.phase_indices.dtype == jnp.int32
    assert actual.exploration_mask.dtype == jnp.bool_
    assert actual.random_actions.dtype == jnp.int32
    assert [
        int(jnp.sum(actual.phase_indices == index))
        for index in range(len(PHASE_ORDER))
    ] == list(DEFAULT_PHASE_LENGTHS)


def test_all_arms_have_identical_genesis_shapes_capacity_and_curation_cadence() -> None:
    protocol = CompositionalFutureUtilityProtocol()
    source = _source_arrays_bound(protocol)
    states: list[CompositionalFeatureState] = []
    configs: list[dict[str, Any]] = []
    for arm_name in ARM_NAMES:
        learner = _build_arm_learner(arm_name)
        configs.append(learner.to_config())
        states.append(
            cast(
                CompositionalFeatureState,
                learner.init(6, source.learner_key).replace(  # type: ignore[attr-defined]
                    birth_timestamp=0.0,
                    uptime_s=0.0,
                ),
            )
        )

    chex.assert_trees_all_equal(states[0], states[1], states[2])
    assert all(config["replacement_interval"] == 32 for config in configs)
    assert all(config["n_features"] == 11 for config in configs)
    assert all(config["candidate_count"] == 8 for config in configs)
    assert all(config["n_tasks"] == 2 for config in configs)
    work = logical_work_per_arm(protocol)
    assert work == {
        "learner_updates": 8_998,
        "curation_update_opportunities": 281,
        "behavior_active_feature_value_cells": 8_998 * 11,
        "learner_update_active_feature_value_cells": 8_998 * 11,
        "total_active_feature_value_cells": 2 * 8_998 * 11,
        "learner_update_candidate_feature_value_cells": 8_998 * 8,
        "evaluator_full_q_dot_products": 8_998,
        "evaluator_raw_q_dot_products": 8_998,
        "learner_prediction_q_dot_products": 8_998,
        "full_and_raw_q_dot_products": 2 * 8_998,
        "total_q_dot_products": 3 * 8_998,
        "total_q_head_scalar_outputs": 3 * 8_998 * 2,
        "ranking_diagnostic_calls": 8_999,
        "active_future_reduction_cells": 8_998 * 2 * 11,
        "candidate_future_reduction_cells": 8_998 * 2 * 8,
        "future_contribution_trace_cells": 8_998 * 2 * (11 + 8),
        "future_feature_energy_trace_cells": 8_998 * (11 + 8),
        "persistent_candidate_active_correlation_cells": 11 * 8,
        "candidate_active_correlation_statistical_accumulation_cells": 0,
        "candidate_active_correlation_reset_mask_cells": 8_998 * 88,
        "ranking_candidate_active_correlation_cells": 8_999 * 88,
        "persistent_state_nbytes": 2_072,
        "persistent_search_archive_entries": 0,
        "keys_stream_shapes_and_update_opportunities_matched": True,
        "compiled_flop_equivalence_claimed": False,
    }
    assert work["candidate_active_correlation_reset_mask_cells"] == 791_824
    assert work["ranking_candidate_active_correlation_cells"] == 791_912
    assert work["persistent_state_nbytes"] == 2_072


def test_energy_novelty_statically_masks_future_mix_before_any_panel() -> None:
    masked_config = learner_config_for_arm(_LEFT_PACK)
    masked_config.update(
        future_utility_trace_mode="contribution",
        future_utility_trace_decay=0.95,
    )
    mix0 = CompositionalFeatureLearner.from_config(
        masked_config | {"future_utility_mix": 0.0}
    )
    mix1 = CompositionalFeatureLearner.from_config(
        masked_config | {"future_utility_mix": 1.0}
    )
    key = _source_arrays_bound(CompositionalFutureUtilityProtocol()).learner_key
    mix0_state = cast(
        CompositionalFeatureState,
        mix0.init(6, key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0, uptime_s=0.0
        ),
    )
    mix1_state = cast(
        CompositionalFeatureState,
        mix1.init(6, key).replace(  # type: ignore[attr-defined]
            birth_timestamp=0.0, uptime_s=0.0
        ),
    )
    observation = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
    targets = jnp.asarray([1.0, jnp.nan], dtype=jnp.float32)
    mix0_result = mix0.update(mix0_state, observation, targets)
    mix1_result = mix1.update(mix1_state, observation, targets)

    chex.assert_trees_all_equal(mix0_result, mix1_result)


def test_legacy_scoring_with_historical_positive_novelty_bonus_is_rejected() -> None:
    invalid = learner_config_for_arm(_LEFT_PACK)
    invalid["candidate_scoring_mode"] = "legacy"

    with pytest.raises(
        ValueError,
        match="candidate_novelty_admission_bonus requires candidate_scoring_mode",
    ):
        CompositionalFeatureLearner.from_config(invalid)
