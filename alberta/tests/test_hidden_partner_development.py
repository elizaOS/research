"""Development-only tests for the uninterrupted hidden-partner evaluator."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import fields

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.evidence_manifest import EVIDENCE_SPECS
from alberta_framework.evaluation.hidden_partner_development import (
    HIDDEN_PARTNER_CONDITIONS,
    HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION,
    ROBUSTNESS_SEED_NAMESPACE,
    TUNING_SEED_NAMESPACE,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerDevelopmentRunner,
    LearnerHiddenPartnerTransition,
    derive_hidden_partner_seed_pairs,
    hidden_partner_development_record,
    hidden_partner_run_summary_from_dict,
    run_hidden_partner_development_suite,
    strip_hidden_partner_oracle,
)
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingWorld,
)

pytestmark = pytest.mark.development


def _small_protocol(*, jitter_radius: int = 0) -> HiddenPartnerDevelopmentProtocol:
    return HiddenPartnerDevelopmentProtocol(
        environment=HiddenPartnerMappingConfig(
            base_segment_lengths=(16,) * 9,
            jitter_radius=jitter_radius,
            partner_flip_probability=0.0,
        ),
        recovery_window=4,
        early_late_window=4,
        recovery_reward_threshold=0.5,
        retention_ratio_threshold=0.5,
        recurrent_early_reward_threshold=0.25,
    )


def test_protocol_round_trip_is_strict_and_structurally_nonpromoting() -> None:
    protocol = HiddenPartnerDevelopmentProtocol()
    payload = protocol.to_config()
    restored = HiddenPartnerDevelopmentProtocol.from_config(payload)

    assert restored == protocol
    assert payload["schema_version"] == HIDDEN_PARTNER_DEVELOPMENT_SCHEMA_VERSION
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    conditions = payload["conditions"]
    assert isinstance(conditions, list)
    assert [condition["name"] for condition in conditions] == [
        "full",
        "state_frozen",
        "memory_masked",
        "lifecycle_frozen",
        "no_carry",
        "no_retention",
        "no_planning",
        "uniform_partner",
        "random_curation",
    ]
    no_retention = next(
        condition for condition in conditions if condition["name"] == "no_retention"
    )
    assert no_retention["agent_config"]["active_utility_retention_decay"] is None

    promoted = copy.deepcopy(payload)
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="cannot promote"):
        HiddenPartnerDevelopmentProtocol.from_config(promoted)
    changed_claim = copy.deepcopy(payload)
    changed_claim["claim_scope"] = "complete Alberta Plan"
    with pytest.raises(ValueError, match="changed claims"):
        HiddenPartnerDevelopmentProtocol.from_config(changed_claim)
    missing = copy.deepcopy(payload)
    del missing["conditions"]
    with pytest.raises(ValueError, match="fields"):
        HiddenPartnerDevelopmentProtocol.from_config(missing)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recovery_window": True},
        {"recovery_window": 0},
        {"early_late_window": 2_000},
        {"recovery_reward_threshold": float("nan")},
        {"retention_ratio_threshold": 1.1},
        {"recurrent_early_reward_threshold": True},
    ],
)
def test_protocol_rejects_invalid_development_controls(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        HiddenPartnerDevelopmentProtocol(**kwargs)


def test_seed_namespaces_are_stable_separated_and_uint32_safe() -> None:
    tuning = derive_hidden_partner_seed_pairs(TUNING_SEED_NAMESPACE, 8)
    tuning_again = derive_hidden_partner_seed_pairs(TUNING_SEED_NAMESPACE, 8)
    robustness = derive_hidden_partner_seed_pairs(ROBUSTNESS_SEED_NAMESPACE, 8)

    assert tuning == tuning_again
    assert tuning != robustness
    assert [pair.index for pair in tuning] == list(range(8))
    all_role_seeds = {
        seed
        for pair in (*tuning, *robustness)
        for seed in (pair.stream_seed, pair.initialization_seed)
    }
    assert len(all_role_seeds) == 32
    assert all(0 <= seed < 2**32 for seed in all_role_seeds)
    with pytest.raises(ValueError):
        derive_hidden_partner_seed_pairs("", 1)
    with pytest.raises(ValueError):
        derive_hidden_partner_seed_pairs("valid", True)


def test_oracle_is_physically_absent_from_learner_transition() -> None:
    world = HiddenPartnerMappingWorld(
        HiddenPartnerMappingConfig(
            base_segment_lengths=(4,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        )
    )
    state = world.init(jr.key(0))
    transition, _ = world.step(state, jnp.asarray(1, dtype=jnp.int32))
    stripped = strip_hidden_partner_oracle(transition)

    assert tuple(field.name for field in fields(LearnerHiddenPartnerTransition)) == (
        "observation",
        "focal_action",
        "partner_action",
        "reward",
        "outcome",
        "next_observation",
        "terminated",
        "discount",
    )
    assert not hasattr(stripped, "oracle")
    np.testing.assert_array_equal(stripped.observation, transition.observation)
    np.testing.assert_array_equal(
        stripped.next_observation,
        transition.next_observation,
    )


@pytest.mark.development
def test_small_uninterrupted_life_reconstructs_prequential_metrics_and_contracts() -> None:
    protocol = _small_protocol(jitter_radius=3)
    seed = derive_hidden_partner_seed_pairs("hidden-partner-unit-life", 1)[0]
    result = HiddenPartnerDevelopmentRunner("full", protocol).run(seed)
    summary = result.summary
    active = np.asarray(result.trace.active, dtype=np.bool_)

    assert summary.condition == "full"
    assert summary.cycle_steps == sum(summary.segment_lengths)
    assert summary.cycle_steps == int(result.final_environment_state.step_count)
    assert protocol.maximum_cycle_steps == 171
    assert int(np.sum(active)) == summary.cycle_steps
    assert np.all(active[: summary.cycle_steps])
    assert np.all(~active[summary.cycle_steps :])
    trace_steps = active.shape[0]
    assert result.trace.interaction_phi_pre.shape == (trace_steps, 12)
    assert result.trace.interaction_target.shape == (trace_steps, 1)
    assert result.trace.candidate_output_weights_pre.shape == (trace_steps, 1, 66)
    assert result.trace.interaction_relevance_probe_errors.shape == (
        trace_steps,
        1,
        12,
    )
    assert result.trace.interaction_candidate_promotion_signal.shape == (
        trace_steps,
        66,
    )
    assert result.trace.interaction_durable_read_mask.shape == (trace_steps, 12)
    np.testing.assert_array_equal(
        result.trace.interaction_candidate_reacquisition_required_pre[1:summary.cycle_steps],
        result.trace.interaction_candidate_reacquisition_required_post[
            : summary.cycle_steps - 1
        ],
    )
    np.testing.assert_array_equal(
        result.trace.interaction_relevance_probe_weights_pre[1:summary.cycle_steps],
        result.trace.interaction_relevance_probe_weights_post[: summary.cycle_steps - 1],
    )
    assert len(summary.segments) == 9
    assert [segment.regime_name for segment in summary.segments] == [
        "A",
        "B",
        "A",
        "D",
        "A",
        "C",
        "A",
        "B",
        "C",
    ]
    assert summary.mean_reward == pytest.approx(
        float(np.mean(np.asarray(result.trace.reward)[active]))
    )
    assert summary.mean_behavior_nll == pytest.approx(
        float(np.mean(np.asarray(result.trace.behavior_nll)[active]))
    )
    assert summary.mean_behavior_brier == pytest.approx(
        float(np.mean(np.asarray(result.trace.behavior_brier)[active]))
    )
    assert summary.counter_contract_valid
    assert summary.causal_contract_valid
    assert summary.all_finite
    assert summary.resource_shape_matched
    assert summary.initial_state_nbytes == summary.final_state_nbytes
    assert result.initial_resource.replay_capacity == 0
    assert result.final_resource.replay_capacity == 0
    assert summary.compilation_wall_seconds >= 0.0
    assert summary.execution_wall_seconds > 0.0
    assert summary.mean_execution_microseconds_per_step > 0.0

    probabilities = np.asarray(result.trace.behavior_probabilities)[active]
    assert np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


@pytest.mark.development
def test_paired_suite_matches_life_resources_and_disabled_counter_semantics() -> None:
    protocol = _small_protocol()
    seed = derive_hidden_partner_seed_pairs("hidden-partner-paired-unit", 1)[0]
    summaries = run_hidden_partner_development_suite(
        (seed,),
        protocol=protocol,
        condition_names=("full", "state_frozen", "no_retention", "no_planning"),
    )

    assert [summary.condition for summary in summaries] == [
        "full",
        "state_frozen",
        "no_retention",
        "no_planning",
    ]
    assert len({summary.cycle_steps for summary in summaries}) == 1
    assert len({summary.segment_lengths for summary in summaries}) == 1
    assert len({summary.initial_state_nbytes for summary in summaries}) == 1
    assert all(summary.resource_shape_matched for summary in summaries)
    assert all(summary.counter_contract_valid for summary in summaries)
    assert all(summary.causal_contract_valid for summary in summaries)
    assert all(summary.all_finite for summary in summaries)


@pytest.mark.development
def test_consumed_tuning_life_demonstrates_finite_discovery_and_control() -> None:
    """Regression guard calibrated on a consumed development seed, never promotion."""
    protocol = HiddenPartnerDevelopmentProtocol()
    seed = derive_hidden_partner_seed_pairs(TUNING_SEED_NAMESPACE, 1)[0]
    full = HiddenPartnerDevelopmentRunner("full", protocol).run(seed).summary
    frozen = HiddenPartnerDevelopmentRunner("lifecycle_frozen", protocol).run(seed).summary

    assert hidden_partner_run_summary_from_dict(full.to_dict()) == full
    assert full.all_finite
    assert full.counter_contract_valid
    assert full.causal_contract_valid
    assert full.resource_shape_matched
    assert full.mean_reward > 0.85
    assert full.mean_reward - frozen.mean_reward > 0.05
    assert full.features.c_first_active_step is not None
    assert full.features.d_first_active_step is not None
    assert full.features.c_survived_first_to_recurrent_c
    assert not frozen.features.c_active_at_recurrent_c_entry
    assert not frozen.features.d_active_at_end_of_d


def test_record_remains_outside_promoted_evidence_registry() -> None:
    protocol = _small_protocol()
    seed = derive_hidden_partner_seed_pairs("hidden-partner-record-unit", 1)[0]
    summary = HiddenPartnerDevelopmentRunner("full", protocol).run(seed).summary
    record = hidden_partner_development_record(protocol, (summary,))

    assert record["development_only"] is True
    assert record["scientific_promotion_allowed"] is False
    assert record["seed_roles"]["promoted_seed_role"] == "none"
    assert record["runs"][0]["development_only"] is True
    assert record["runs"][0]["scientific_promotion_allowed"] is False
    assert all(
        spec.name != "hidden_partner_development"
        and "hidden_partner" not in spec.relative_path.as_posix()
        for spec in EVIDENCE_SPECS
    )


def test_condition_table_has_exact_shape_matched_ablation_flags() -> None:
    by_name = {condition.name: condition.config for condition in HIDDEN_PARTNER_CONDITIONS}
    full = by_name["full"]

    assert by_name["state_frozen"] == dataclasses.replace(
        full,
        state_learning_enabled=False,
    )
    assert by_name["memory_masked"] == dataclasses.replace(full, memory_masked=True)
    assert by_name["lifecycle_frozen"] == dataclasses.replace(
        full,
        feature_lifecycle_enabled=False,
    )
    assert by_name["no_carry"] == dataclasses.replace(full, carry_survivors=False)
    assert by_name["no_retention"] == dataclasses.replace(
        full,
        active_utility_retention_decay=None,
    )
    assert by_name["no_planning"] == dataclasses.replace(
        full,
        planning_enabled=False,
    )
    assert by_name["uniform_partner"] == dataclasses.replace(
        full,
        uniform_partner_belief=True,
    )
    assert by_name["random_curation"] == dataclasses.replace(
        full,
        random_feature_curation=True,
    )
