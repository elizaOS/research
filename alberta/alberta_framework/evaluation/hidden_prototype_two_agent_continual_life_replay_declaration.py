"""Pure-stdlib, development-only partial comparator declaration for the U0 replay."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = "alberta.hidden-prototype-two-agent-replay-declaration.v1"
DEVELOPMENT_ONLY: Final = True
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
FULL_REPORT_IDENTITY_CLAIM_ALLOWED: Final = False
SOURCE_MANIFEST_COVERAGE: Final = "selected-direct-files-not-transitive-closure"
RUNTIME_IDENTITY_BOUND: Final = False
ROUTED: Final = "hidden_inferred_full"
UNROUTED: Final = "hidden_inference_unrouted"

# Ordered digest framing: UTF-8 ``path + NUL + lowercase_sha256 + LF``.
SEMANTIC_SOURCE_MANIFEST: Final = (
    ("alberta_framework/evaluation/hidden_prototype_two_agent_continual_life_development.py",
     "0df7eba37a3b8dc0e0eb283a05fb27676c174ed737c9b8e9dfd3d98c33266ecb"),
    ("alberta_framework/core/context_inference.py",
     "47dec376567b2279f4da4d83979f390a1677d3e941a223e8c0d18d1ef9014493"),
    ("alberta_framework/core/horde.py",
     "52baf7f9c5037b3a504b55b232e02933f3d96b110b4cea5e2e18e7022bd9d551"),
    ("alberta_framework/core/prototype_agent.py",
     "1e05b1f8ea935ac454a485b4ebb5dcff1e2676d13574735ed2d277b7a366f25c"),
    ("alberta_framework/core/prototype_feature_memory.py",
     "2aa51c27ca5cef768577d145d946d00d67107f2bf0e74d66e72e94d9689472ab"),
    ("alberta_framework/core/world_model.py",
     "b5e700a8f313f165fee128ff41393f1734b86bdc2172017ec18c7b7dfb952309"),
    ("alberta_framework/evaluation/hidden_context_coadaptation_development.py",
     "138e0fcf9d9c1b844df47593e5f5d7a0b2c84a4b0b5ae51a17171e9374e0f0a4"),
    ("alberta_framework/evaluation/prototype_feature_memory_recurrence_development.py",
     "fb83bb39791cfc3646417e45f8f33b244d525b4625126c4a359dd3d7e8acdac7"),
    ("alberta_framework/evaluation/prototype_two_learning_agent_recurrence_development.py",
     "d506d8e55f9aafc1bd109074608e4915a7c5159d40827c1e38afda7ae4414e74"),
    ("alberta_framework/streams/recurring_multiagent.py",
     "9306bc727fdc37bf2345df6d69ae9541b548b13be9eb75791553004c52f21ec5"),
)
SEMANTIC_SOURCE_MANIFEST_DIGEST: Final = (
    "7cfb95f3a96fcb441f2d8e5e471ccbe3fb3fba260c9e987ff8ffc776eca921da"
)


def semantic_source_manifest_digest(
    manifest: Sequence[tuple[str, str]] = SEMANTIC_SOURCE_MANIFEST,
) -> str:
    payload = "".join(f"{path}\0{digest}\n" for path, digest in manifest).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_semantic_source_manifest(root: Path) -> tuple[str, ...]:
    """Return fail-closed path/hash errors without importing U0 or JAX."""

    errors: list[str] = []
    resolved_root = root.resolve()
    if semantic_source_manifest_digest() != SEMANTIC_SOURCE_MANIFEST_DIGEST:
        errors.append("semantic source manifest digest is internally inconsistent")
    for path, expected in SEMANTIC_SOURCE_MANIFEST:
        candidate = (resolved_root / path).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            errors.append(f"missing semantic source: {path}")
            continue
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != expected:
            errors.append(f"semantic source digest mismatch: {path}: {expected} != {observed}")
    return tuple(errors)


OBSERVED_METRICS_BY_ARM: Final = {
    ROUTED: {
        "phase_mean_agent_reward": {
            "A1": {"mean": 0.9914010049542412, "early_mean": 0.9373603817075491,
                   "tail_mean": 1.0},
            "B": {"mean": 0.9779351669130847, "early_mean": 0.8513473533093929,
                  "tail_mean": 1.0},
            "A2": {"mean": 0.9644929763162509, "early_mean": 0.78154581412673,
                   "tail_mean": 0.9979492193087935},
        },
        "recurrence": {"A2_early_minus_A1_tail": -0.21845418587327003,
                       "A2_tail_minus_A1_tail": -0.0020507806912064552,
                       "A2_reacquisition": 0.21640340518206358},
        "context_switches": [2, 2],
        "mean_horde_squared_error": 0.004624179810620101,
        "mean_world_model_error": 0.30710208359477775,
        "feature_curation_commits": 5,
        "memory_evictions": 2944,
        "memory_evicted_provenance_summary": {
            "count": 2944, "unique_count": 1480, "minimum": 0, "maximum": 1486,
            "first_eight": [2, 1, 1, 2, 4, 0, 3, 3],
            "last_eight": [1473, 1474, 1475, 1473, 1469, 1454, 1460, 1471],
        },
        "memory_retrievals": 3000,
        "memory_next_action_changes": 863,
        "memory_counterfactual_outcome_counts": {"benefit": 800, "harm": 61,
                                                  "neutral": 2211},
        "mean_same_prestate_reward_effects": {
            "agent0_own_action": 0.018990875881475706,
            "agent1_own_action": 0.018823636734547716,
            "agent0_partner_action": 0.018823636734547716,
            "agent1_partner_action": 0.018990875881475706,
            "agent0_interaction": 0.0003173823157946269,
            "agent1_interaction": 0.0003173823157946269,
            "joint_mean": 0.037497129679347076,
        },
    },
    UNROUTED: {
        "phase_mean_agent_reward": {
            "A1": {"mean": 0.9846382247051224, "early_mean": 0.9017151659354568,
                   "tail_mean": 1.0},
            "B": {"mean": 0.014950294571463019, "early_mean": 0.0,
                  "tail_mean": 0.032067867927253246},
            "A2": {"mean": 0.9948097240412608, "early_mean": 0.9856445351615548,
                   "tail_mean": 1.0},
        },
        "recurrence": {"A2_early_minus_A1_tail": -0.014355464838445187,
                       "A2_tail_minus_A1_tail": 0.0,
                       "A2_reacquisition": 0.014355464838445187},
        "context_switches": [2, 2],
        "mean_horde_squared_error": 0.00683825900757884,
        "mean_world_model_error": 0.3056164535245574,
        "feature_curation_commits": 13,
        "memory_evictions": 2944,
        "memory_evicted_provenance_summary": {
            "count": 2944, "unique_count": 1480, "minimum": 0, "maximum": 1490,
            "first_eight": [2, 1, 1, 2, 4, 0, 3, 3],
            "last_eight": [1460, 1459, 1490, 1471, 1482, 1465, 1473, 1466],
        },
        "memory_retrievals": 2934,
        "memory_next_action_changes": 551,
        "memory_counterfactual_outcome_counts": {"benefit": 186, "harm": 365,
                                                  "neutral": 2521},
        "mean_same_prestate_reward_effects": {
            "agent0_own_action": -0.001367067239092042,
            "agent1_own_action": -0.00795974635790723,
            "agent0_partner_action": -0.00795974635790723,
            "agent1_partner_action": -0.001367067239092042,
            "agent0_interaction": -0.007672626835604508,
            "agent1_interaction": -0.007672626835604508,
            "joint_mean": -0.0016541869554203004,
        },
    },
}

OBSERVED_RESOURCES: Final = {
    "initial_persistent_state_nbytes": 41718, "environment_state_nbytes": 52,
    "context_state_nbytes_per_agent": 94,
    "prototype_state_nbytes_per_agent_initial": [20733, 20733],
    "outer_auxiliary_state_nbytes": 12,
    "initial_persistent_state_decomposition_nbytes": 41718,
    "staged_full-state-copy_lower_bound_nbytes": 83328,
}
OBSERVED_WORK: Final = {
    "requested_joint_transitions": 1536, "environment_proposal_calls": 6144,
    "counterfactual_environment_proposal_calls": 4608,
    "committed_environment_transitions": 1536,
    "context_inference_update_calls": 3072, "context_inference_carried_updates": 3072,
    "discarded_preview_update_calls": 3072, "committed_candidate_update_calls": 3072,
    "prototype_update_calls": 6144, "world_model_update_calls": 6144,
    "world_model_carried_updates": 3072, "explicit_world_model_prediction_calls": 3072,
    "managed_horde_update_calls": 6144, "explicit_horde_prediction_calls": 3072,
    "feature_lifecycle_observe_calls": 6144, "memory_sidecars_supplied": 3072,
    "memory_query_before_write_transactions": 3072, "outer_atomic_decisions": 1536,
    "checkpoint_save_calls": 0, "checkpoint_load_calls": 0,
    "resets_after_initialization": 0, "boundary_callbacks": 0,
    "external_partner_policy_calls": 0,
}
OBSERVED_PARITY_WORK: Final = {
    "requested_joint_transitions": 2, "environment_proposal_calls": 8,
    "counterfactual_environment_proposal_calls": 6,
    "committed_environment_transitions": 2, "context_inference_update_calls": 4,
    "context_inference_carried_updates": 4, "discarded_preview_update_calls": 4,
    "committed_candidate_update_calls": 4, "prototype_update_calls": 8,
    "world_model_update_calls": 8, "world_model_carried_updates": 4,
    "explicit_world_model_prediction_calls": 4, "managed_horde_update_calls": 8,
    "explicit_horde_prediction_calls": 4, "feature_lifecycle_observe_calls": 8,
    "memory_sidecars_supplied": 4, "memory_query_before_write_transactions": 4,
    "outer_atomic_decisions": 2, "checkpoint_save_calls": 0,
    "checkpoint_load_calls": 0, "resets_after_initialization": 0,
    "boundary_callbacks": 0, "external_partner_policy_calls": 0,
}
NEW_WORK_UNOBSERVED: Final = {
    "outer_source_state_validations": 1536, "outer_candidate_state_validations": 1536,
    "outer_clock_alignment_checks": 3072, "per_agent_old_bank_contract_checks": 3072,
    "per_agent_memory_contract_checks": 3072, "per_agent_horde_contract_checks": 3072,
    "per_agent_world_model_contract_checks": 3072,
}
NEW_PARITY_WORK_UNOBSERVED: Final = {
    "outer_source_state_validations": 2, "outer_candidate_state_validations": 2,
    "outer_clock_alignment_checks": 4, "per_agent_old_bank_contract_checks": 4,
    "per_agent_memory_contract_checks": 4, "per_agent_horde_contract_checks": 4,
    "per_agent_world_model_contract_checks": 4,
}
STATIC_METRICS: Final = {
    "reward_aggregation": "arithmetic mean of the two receiving-agent rewards",
    "memory_eviction_accounting": {"expected_after_fixed_capacity_fill": 2944,
                                    "observed": 2944, "exact": True},
    "memory_evicted_provenance_summary": {
        "exact_sequence_location": "trace[*].agents[*].memory.evicted_provenance_id"},
    "memory_counterfactual_outcomes_observed": ["benefit", "harm", "neutral"],
}
STATIC_RESOURCES: Final = {
    "logical_fixed_allocation": True, "final_persistent_state_nbytes": 41718,
    "phase_boundary_persistent_state_nbytes": [41718, 41718, 41718, 41718],
    "peak_persistent_state_nbytes": 41718,
    "context_resource_declaration": {
        "allocated_float32_scalars": 10, "allocated_bool_scalars": 2,
        "allocated_int32_scalars": 5, "allocated_uint32_scalars": 8,
        "state_nbytes": 94, "clock_nbytes": 48, "exact_clock_delta_nbytes": 32,
        "max_contexts": 2, "replay_capacity": 0},
    "prototype_state_nbytes_per_agent_final": [20733, 20733],
    "staged_full-state-copy_lower_bound_semantics": (
        "four environment states + two context candidates + two previews + "
        "two Prototype candidates; diagnostics/compiler workspaces excluded"),
    "compiler_allocator_or_device_residency_claimed": False,
}


def _run_subset(arm: str, route: bool, *, static: bool) -> dict[str, Any]:
    if static:
        return {"arm": arm, "route_inference": route, "metrics": STATIC_METRICS,
                "resources": STATIC_RESOURCES, "work": NEW_WORK_UNOBSERVED,
                "execution_parity": {
                    "float_contract": "rtol=1e-6, atol=1e-7; discrete leaves exact",
                    "measurement_logical_work": NEW_PARITY_WORK_UNOBSERVED},
                "checkpoint_resume_claimed": False}
    return {"metrics": OBSERVED_METRICS_BY_ARM[arm], "resources": OBSERVED_RESOURCES,
            "work": OBSERVED_WORK, "execution_parity": {
                "checked": True, "state_and_trace_equivalent": True,
                "measurement_events_not_carried_into_life": 2,
                "measurement_logical_work": OBSERVED_PARITY_WORK},
            "final_context_slots": [0, 0]}


OBSERVED_EXPECTATIONS: Final = {"runs_by_arm": {
    ROUTED: _run_subset(ROUTED, True, static=False),
    UNROUTED: _run_subset(UNROUTED, False, static=False)},
    "matched_comparison": {"persistent_state_shape_matched": True,
                           "logical_work_matched": True, "same_consumed_root": True,
                           "same_context_configuration_and_update_work": True,
                           "same_Prototype_configuration": True,
                           "only_declared_intervention": (
                               "route the same past-only inferred onehot into the two formerly "
                               "visible cue coordinates")}}
STATIC_EXPECTATIONS: Final = {
    "schema_version": "alberta.hidden-prototype-two-agent-continual-life-development.report.v1",
    "development_only": True, "scientific_promotion_allowed": False,
    "accepted_scientific_evidence": False, "acceptance_status": "not-assessed",
    "writer_available": False, "winner_selected": False,
    "context_capacity_pressure_claimed": False, "checkpoint_resume_claimed": False,
    "runs_by_arm": {ROUTED: _run_subset(ROUTED, True, static=True),
                    UNROUTED: _run_subset(UNROUTED, False, static=True)}}

METRIC_COVERAGE: Final = {
    "observed": tuple(OBSERVED_METRICS_BY_ARM[ROUTED]),
    "static": ("reward_aggregation", "memory_eviction_accounting",
               "memory_counterfactual_outcomes_observed"),
    "mixed": ("memory_evicted_provenance_summary",),
    "unknown": ("context_onehots_used",),
}
RESOURCE_COVERAGE: Final = {"observed": tuple(OBSERVED_RESOURCES),
                            "static": tuple(STATIC_RESOURCES),
                            "unknown": ("prototype_state_measurements_initial",)}
WORK_COVERAGE: Final = {"observed": tuple(OBSERVED_WORK),
                        "static-unobserved": tuple(NEW_WORK_UNOBSERVED)}
PARITY_COVERAGE: Final = {
    "observed": ("checked", "state_and_trace_equivalent",
                 "measurement_events_not_carried_into_life"),
    "static": ("float_contract",), "mixed": ("measurement_logical_work",)}
UNKNOWN_REPORT_PATHS: Final = (
    "environment_config", "agent_config", "protocol", "runs_by_arm.*.trace",
    "runs_by_arm.*.metrics.context_onehots_used",
    "runs_by_arm.*.resources.prototype_state_measurements_initial",
)
PRE_RUN_DECLARATION: Final = {
    "schema_version": SCHEMA_VERSION, "development_only": True,
    "scientific_promotion_allowed": False, "output_writes_allowed": False,
    "coverage": "partial", "full_report_identity_claim_allowed": False,
    "permitted_success_wording": "declared-fields-exact-partial-coverage",
    "manifest_digest": SEMANTIC_SOURCE_MANIFEST_DIGEST,
    "source_manifest_coverage": SOURCE_MANIFEST_COVERAGE,
    "runtime_identity_bound": RUNTIME_IDENTITY_BOUND,
    "source_manifest_must_match_before_and_after_replay": True,
    "unknown_report_paths": UNKNOWN_REPORT_PATHS,
    "new_work_counters_prior_observation_status": "never-observed",
}


def _strict_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _strict_equal(actual[key], value)
            for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return (not isinstance(actual, (str, bytes)) and isinstance(actual, (list, tuple))
                and len(actual) == len(expected)
                and all(_strict_equal(a, b) for a, b in zip(actual, expected, strict=True)))
    return type(actual) is type(expected) and actual == expected


def _subset_mismatches(actual: Any, expected: Any, path: str = "") -> list[str]:
    if not isinstance(expected, Mapping):
        return [] if _strict_equal(actual, expected) else [path]
    if not isinstance(actual, Mapping):
        return [path or "<root>"]
    errors: list[str] = []
    for key, value in expected.items():
        child = f"{path}.{key}" if path else str(key)
        errors.extend(_subset_mismatches(actual.get(key, object()), value, child))
    return errors


def compare_declared_replay(report: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Compare declared subsets exactly; never claim full-report identity."""

    source_errors = validate_semantic_source_manifest(root)
    runs = report.get("runs")
    runs_by_arm = ({str(run.get("arm")): run for run in runs if isinstance(run, Mapping)}
                   if isinstance(runs, list) else {})
    comparable = {**report, "runs_by_arm": runs_by_arm}
    mismatches = _subset_mismatches(comparable, OBSERVED_EXPECTATIONS)
    mismatches.extend(_subset_mismatches(comparable, STATIC_EXPECTATIONS))
    exact = not source_errors and not mismatches
    conclusion = ("source-manifest-mismatch" if source_errors else
                  "declared-field-discrepancy" if mismatches else
                  "declared-fields-exact-partial-coverage")
    return {"coverage": "partial", "full_report_identity": False,
            "source_manifest_coverage": SOURCE_MANIFEST_COVERAGE,
            "runtime_identity_bound": RUNTIME_IDENTITY_BOUND,
            "source_manifest_valid": not source_errors,
            "source_manifest_errors": source_errors, "declared_fields_exact": exact,
            "mismatches": tuple(mismatches), "conclusion": conclusion}


__all__ = ["METRIC_COVERAGE", "NEW_PARITY_WORK_UNOBSERVED",
           "NEW_WORK_UNOBSERVED", "OBSERVED_EXPECTATIONS", "PRE_RUN_DECLARATION",
           "RESOURCE_COVERAGE", "SEMANTIC_SOURCE_MANIFEST",
           "SEMANTIC_SOURCE_MANIFEST_DIGEST", "STATIC_EXPECTATIONS", "UNKNOWN_REPORT_PATHS",
           "WORK_COVERAGE", "PARITY_COVERAGE", "compare_declared_replay",
           "semantic_source_manifest_digest", "validate_semantic_source_manifest"]
