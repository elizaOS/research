"""Tests for the negative generated-class D reachability audit."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_RAW,
    CompositionalFeatureLearner,
)
from alberta_framework.evaluation import (
    generated_class_reachability as reachability_module,
)
from alberta_framework.evaluation.generated_class_reachability import (
    GENERATED_CLASS_REACHABILITY_AUDIT_SCHEMA,
    GENERATED_CLASS_REACHABILITY_AUDIT_STATUS,
    GeneratedClassReachabilityAudit,
    build_generated_class_reachability_audit,
    validate_generated_class_reachability_audit,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    FULL_LIFECYCLE,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_learner,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def audit() -> GeneratedClassReachabilityAudit:
    return build_generated_class_reachability_audit()


def test_second_d_candidate_channel_has_exact_optimistic_ceiling(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    start = sum(protocol.phase_lengths[: audit.second_d_phase_index])
    stop = start + protocol.phase_lengths[audit.second_d_phase_index]
    cadence = protocol.curation_opportunity_audit.curation_interval
    events = tuple(
        range(((start // cadence) + 1) * cadence, (stop // cadence) * cadence + 1, cadence)
    )

    assert audit.schema == GENERATED_CLASS_REACHABILITY_AUDIT_SCHEMA
    assert audit.status == GENERATED_CLASS_REACHABILITY_AUDIT_STATUS
    assert audit.second_d_start_step_zero_based == start == 3_539
    assert audit.second_d_stop_step_zero_based_exclusive == stop == 3_928
    assert audit.second_d_curation_step_counts_one_based == events
    assert audit.second_d_curation_opportunities == len(events) == 12

    gate = Fraction(
        audit.gate_mass_upper_bound_numerator,
        audit.gate_mass_upper_bound_denominator,
    )
    right = Fraction(
        audit.required_right_probability_numerator,
        audit.required_right_probability_denominator,
    )
    left = Fraction(
        audit.optimistic_required_left_parent_probability_numerator,
        audit.optimistic_required_left_parent_probability_denominator,
    )
    applied = Fraction(
        audit.optimistic_proposal_application_probability_numerator,
        audit.optimistic_proposal_application_probability_denominator,
    )
    per_proposal = gate * right * left * applied
    any_birth = 1 - (1 - per_proposal) ** len(events)

    assert gate == Fraction(1, 4)
    assert right == Fraction(1, 4)
    assert left == applied == 1
    assert per_proposal == Fraction(1, 16)
    assert (
        audit.per_candidate_proposal_d_birth_upper_bound_numerator,
        audit.per_candidate_proposal_d_birth_upper_bound_denominator,
    ) == (per_proposal.numerator, per_proposal.denominator)
    assert (
        audit.candidate_channel_any_birth_upper_bound_numerator,
        audit.candidate_channel_any_birth_upper_bound_denominator,
    ) == (any_birth.numerator, any_birth.denominator)
    assert any_birth == Fraction(151_728_638_820_031, 281_474_976_710_656)
    assert audit.candidate_channel_any_birth_upper_bound_float64 == float(any_birth)


def test_exact_zero_operation_support_and_four_way_gate_mass_are_bound_to_core(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    logits = learner._op_logits()
    prior = jnp.asarray(audit.operation_prior_float32, dtype=jnp.float32)
    expected_logits = jnp.where(prior > 0.0, jnp.log(prior), -jnp.inf)
    probabilities = jax.nn.softmax(logits)

    assert tuple(int(value) for value in np.asarray(logits).view(np.uint32)) == (
        audit.op_logits_float32_bits
    )
    assert np.array_equal(np.asarray(logits), np.asarray(expected_logits))
    assert tuple(
        int(value) for value in np.asarray(probabilities).view(np.uint32)
    ) == audit.op_probabilities_float32_bits
    assert bool(jnp.all(jnp.isneginf(logits[prior == 0.0])))
    assert float(probabilities[OP_RAW]) == 0.0
    assert float(probabilities[OP_GATED]) == 0.25
    assert audit.zero_prior_operation_logits_are_negative_infinity
    assert audit.raw_operation_probability_is_exact_zero


def test_reviewed_source_closure_is_bound_to_full_module_and_methods(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    assert audit.schema == "alberta.generated-class-reachability-audit.development.v4"
    source_path = inspect.getsourcefile(CompositionalFeatureLearner)
    assert source_path is not None
    assert audit.reviewed_compositional_features_module_byte_sha256 == (
        hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    )
    reviewed_source_hashes = {
        "__init__": audit.reviewed_init_source_sha256,
        "_op_logits": audit.reviewed_op_logits_source_sha256,
        "_generate_one": audit.reviewed_generate_one_source_sha256,
        "_curation_stage_guidance": (
            audit.reviewed_curation_stage_guidance_source_sha256
        ),
        "_cascade_replace": audit.reviewed_cascade_replace_source_sha256,
        "_cascade_replace_with_mask": (
            audit.reviewed_cascade_replace_with_mask_source_sha256
        ),
        "update": audit.reviewed_update_source_sha256,
    }
    for method_name, reviewed_sha256 in reviewed_source_hashes.items():
        method = getattr(CompositionalFeatureLearner, method_name)
        assert reviewed_sha256 == hashlib.sha256(
            inspect.getsource(method).encode("utf-8")
        ).hexdigest()


def test_module_digest_and_guidance_source_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            reachability_module,
            "_compositional_features_module_byte_sha256",
            lambda: "0" * 64,
        )
        with pytest.raises(RuntimeError, match="module bytes changed"):
            build_generated_class_reachability_audit()

    original = CompositionalFeatureLearner._curation_stage_guidance

    def tampered_guidance(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(
            CompositionalFeatureLearner,
            "_curation_stage_guidance",
            tampered_guidance,
        )
        with pytest.raises(RuntimeError, match="_curation_stage_guidance source changed"):
            build_generated_class_reachability_audit()


def test_candidate_ceiling_does_not_claim_actual_lifecycle_probability(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    assert audit.raw_parent_slots == (0, 1, 2, 3)
    assert audit.required_right_raw_index == audit.required_right_raw_slot == 1
    assert audit.initial_target_active_occurrences == 0
    assert audit.initial_target_candidate_occurrences == 0
    assert audit.initial_left_parent_active_occurrences == 1
    assert audit.initial_right_parent_active_occurrences == 1
    assert audit.initialization_structure_key_invariant_observed
    assert audit.split_key_categorical_model_used
    assert audit.conditional_hazard_chain_rule_used
    assert not audit.cross_event_independence_assumed
    assert not audit.candidate_channel_bound_applies_to_current_lifecycle
    assert not audit.protocol_post_scrub_generation_freeze_complete
    assert not audit.protocol_fresh_reacquisition_generation_epoch_complete
    assert not audit.protocol_fresh_reacquisition_generation_key_namespace_complete
    assert not audit.nontrivial_state_independent_lower_bound_derived
    assert not audit.whole_lifecycle_d_birth_probability_identified
    assert not audit.whole_lifecycle_d_reacquisition_probability_identified
    assert not audit.adequate_reacquisition_probability_demonstrated
    assert any("cascade" in blocker for blocker in audit.blockers)
    assert any("overdepth-regeneration" in blocker for blocker in audit.blockers)
    assert any("ledger" in blocker for blocker in audit.blockers)
    assert any("replay" in requirement for requirement in audit.minimum_identification_contract)


def test_resource_and_operation_accounting_exclude_experiment_execution(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    resources = audit.resource_contract
    operations = audit.operation_contract
    protocol = build_generated_class_recurrence_v0_protocol()

    assert resources.deterministic_initialization_probes == 2
    assert resources.initialization_key_impl == "threefry2x32"
    assert resources.one_initialized_state_jax_nbytes == (
        protocol.resource_contract.jax_state_nbytes
    )
    assert resources.peak_probe_state_jax_nbytes == (
        2 * resources.one_initialized_state_jax_nbytes
    )
    assert resources.learner_updates_executed == 0
    assert resources.runtime_generate_one_calls_executed == 0
    assert resources.full_life_steps_executed == 0
    assert resources.monte_carlo_samples == 0
    assert resources.artifact_bytes_written == 0
    assert resources.wall_clock_threshold is None
    assert not resources.flop_or_hlo_equivalence_claimed

    assert operations.phase_lengths_checked == 9
    assert operations.second_d_curation_events_enumerated == 12
    assert operations.candidate_proposal_channels_counted_per_curation == 1
    assert operations.probability_factors_per_candidate_proposal == 4
    assert operations.conditional_miss_product_exponent == 12
    assert operations.active_cascade_refill_channels_in_probability_bound == 0
    assert (
        operations.candidate_overdepth_regeneration_channels_in_probability_bound == 0
    )
    assert operations.max_active_cascade_refill_write_sites_per_curation == 9
    assert (
        operations.max_candidate_overdepth_regeneration_write_sites_per_curation == 8
    )
    assert operations.max_fresh_structural_write_sites_per_curation == 18

    assert not audit.runner_execution_authorized
    assert not audit.threshold_authorized
    assert not audit.evidence_authorized
    assert not audit.scientific_promotion_allowed


def test_strict_validator_rejects_top_level_and_nested_tampering(
    audit: GeneratedClassReachabilityAudit,
) -> None:
    assert validate_generated_class_reachability_audit(audit) is audit

    forged_probability = dataclasses.replace(
        audit,
        candidate_channel_any_birth_upper_bound_numerator=(
            audit.candidate_channel_any_birth_upper_bound_numerator + 1
        ),
    )
    with pytest.raises(ValueError, match="candidate_channel_any_birth_upper_bound_numerator"):
        validate_generated_class_reachability_audit(forged_probability)

    forged_resources = dataclasses.replace(
        audit,
        resource_contract=dataclasses.replace(
            audit.resource_contract,
            monte_carlo_samples=1,
        ),
    )
    with pytest.raises(ValueError, match="resource_contract"):
        validate_generated_class_reachability_audit(forged_resources)

    with pytest.raises(TypeError, match="exact GeneratedClassReachabilityAudit"):
        validate_generated_class_reachability_audit(object())  # type: ignore[arg-type]
