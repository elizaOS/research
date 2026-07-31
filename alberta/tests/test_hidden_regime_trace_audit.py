"""Independent integration and tamper tests for primitive trace auditing."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_regime_signaling_development as development_module
import alberta_framework.evaluation.hidden_regime_trace_audit as trace_audit_module
from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    CONSTANT_CHANNEL,
    HELPER_FROZEN,
    SELECTIVE_FULL,
    SHUFFLED_CHANNEL,
    WRITABLE_LRU,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeRunResult,
    HiddenRegimeSeedPair,
    run_hidden_regime_condition,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    EVIDENCE_BOUNDARY,
    HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
    HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
    SUPPORTED_DEVELOPMENT_SCHEMA,
    SUPPORTED_TRACE_SCHEMA,
    UNOBSERVED_TRANSITION_FIELDS,
    HiddenRegimeTraceAuditInput,
    audit_hidden_regime_run_result,
    audit_hidden_regime_trace,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    ROLE_HELPER,
    SlotRoleOracleConfig,
)
from alberta_framework.streams.hidden_regime_signaling import (
    DEFAULT_REGIME_PERMUTATIONS,
    HiddenRegimeWorldConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tiny_config() -> HiddenRegimeDevelopmentConfig:
    return HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=(5, 7, 4),
            segment_regimes=(0, 1, 0),
            regime_permutations=DEFAULT_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.25,
            epsilon=0.1,
            relevance_rate=0.1,
            lease_length=4,
            confirmation_steps=2,
            durable_retrieval_threshold=0.5,
            candidate_confirmation_threshold=0.75,
            candidate_confirmation_leases=2,
            scratch_training_leases_before_retest=2,
        ),
        metric_window=2,
    )


@pytest.fixture(scope="module")
def seed_pair() -> HiddenRegimeSeedPair:
    return HiddenRegimeSeedPair(
        namespace="manual-trace-audit-unit-v1",
        index=0,
        world_seed=123,
        learner_seed=456,
    )


@pytest.fixture(scope="module")
def direct_run(
    tiny_config: HiddenRegimeDevelopmentConfig,
    seed_pair: HiddenRegimeSeedPair,
) -> HiddenRegimeRunResult:
    return run_hidden_regime_condition(
        SELECTIVE_FULL,
        seed_pair=seed_pair,
        config=tiny_config,
    )


@pytest.fixture(scope="module")
def direct_input(direct_run: HiddenRegimeRunResult) -> HiddenRegimeTraceAuditInput:
    return HiddenRegimeTraceAuditInput.from_run_result(direct_run)


def _replace_trace_field(
    audit_input: HiddenRegimeTraceAuditInput,
    field: str,
    mutate: Callable[[np.ndarray[Any, Any]], None],
) -> HiddenRegimeTraceAuditInput:
    array = np.array(np.asarray(getattr(audit_input.trace, field)), copy=True)
    mutate(array)
    trace = dataclasses.replace(audit_input.trace, **{field: array})
    return dataclasses.replace(audit_input, trace=trace)


def _transform_all_rows(
    trace: HiddenRegimePrimitiveTrace,
    transform: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
) -> HiddenRegimePrimitiveTrace:
    fields = {
        field.name: transform(np.array(np.asarray(getattr(trace, field.name)), copy=True))
        for field in dataclasses.fields(trace)
    }
    return HiddenRegimePrimitiveTrace(**fields)


def _swap_roles(trace: HiddenRegimePrimitiveTrace) -> HiddenRegimePrimitiveTrace:
    updates: dict[str, object] = {}
    field_names = {field.name for field in dataclasses.fields(trace)}
    for helper_field in sorted(name for name in field_names if name.startswith("helper_")):
        beneficiary_field = "beneficiary_" + helper_field.removeprefix("helper_")
        if beneficiary_field in field_names:
            updates[helper_field] = np.array(
                np.asarray(getattr(trace, beneficiary_field)), copy=True
            )
            updates[beneficiary_field] = np.array(
                np.asarray(getattr(trace, helper_field)), copy=True
            )
    return dataclasses.replace(trace, **updates)


def test_complete_trace_passes_all_three_independent_transition_counts(
    direct_run: HiddenRegimeRunResult,
) -> None:
    report = audit_hidden_regime_run_result(direct_run)

    assert report.valid, report.mismatches
    assert report.expected_steps == direct_run.config.num_steps
    assert report.rows_checked == direct_run.config.num_steps
    assert report.helper_transitions_checked == direct_run.config.num_steps
    assert report.beneficiary_transitions_checked == direct_run.config.num_steps
    assert report.world_transitions_checked == direct_run.config.num_steps
    assert report.mismatches == ()
    assert report.evidence_boundary == EVIDENCE_BOUNDARY
    payload = report.to_dict()
    assert payload["schema"] == HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA
    assert payload["valid"] is True
    assert "same-backend JAX threefry" in cast(str, payload["evidence_boundary"])
    assert "independently implemented host state machines" in EVIDENCE_BOUNDARY
    assert "every derived summary and resource field" in EVIDENCE_BOUNDARY
    assert report.unobserved_transition_fields == ()
    assert report.unobserved_transition_fields == UNOBSERVED_TRANSITION_FIELDS
    assert payload["unobserved_transition_fields"] == []
    assert "actual runtime terminated and discount leaves" in EVIDENCE_BOUNDARY


@pytest.mark.parametrize(
    "condition",
    (SELECTIVE_FULL, CONSTANT_CHANNEL, SHUFFLED_CHANNEL, HELPER_FROZEN, WRITABLE_LRU),
)
def test_exact_condition_channel_permits_and_factorial_axes_are_audited(
    condition: str,
    tiny_config: HiddenRegimeDevelopmentConfig,
    seed_pair: HiddenRegimeSeedPair,
) -> None:
    run = run_hidden_regime_condition(
        condition,  # type: ignore[arg-type]
        seed_pair=seed_pair,
        config=tiny_config,
    )

    report = audit_hidden_regime_run_result(run)

    assert report.valid, report.mismatches


def test_audit_condition_binding_is_independent_of_producer_helper(
    direct_run: HiddenRegimeRunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_condition_spec(_condition: object) -> object:
        raise AssertionError("producer condition_spec must not be called by the audit")

    monkeypatch.setattr(development_module, "condition_spec", broken_condition_spec)

    report = audit_hidden_regime_run_result(direct_run)

    assert report.valid, report.mismatches


def test_schema_contracts_are_literal_and_fail_closed(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    assert direct_input.schema == HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA
    assert direct_input.development_schema == SUPPORTED_DEVELOPMENT_SCHEMA
    assert direct_input.trace_schema == SUPPORTED_TRACE_SCHEMA

    for field, bad_value in (
        ("schema", "alberta.hidden-regime.trace-audit-input.v1"),
        ("development_schema", "alberta.hidden-regime-signaling.development.v3"),
        ("trace_schema", "alberta.hidden-regime-signaling.primitive-trace.v2"),
    ):
        report = audit_hidden_regime_trace(
            dataclasses.replace(direct_input, **{field: bad_value})
        )
        assert not report.valid
        assert f"input.{field}" in report.mismatches
        assert report.rows_checked == 0


def test_wrong_input_object_is_a_fail_closed_report() -> None:
    report = audit_hidden_regime_trace({"trace": "not-an-envelope"})

    assert not report.valid
    assert report.rows_checked == 0
    assert report.mismatches == ("input: expected HiddenRegimeTraceAuditInput",)


def test_trace_shape_dtype_and_nonfinite_inputs_are_rejected_before_rows(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    wrong_shape = _replace_trace_field(
        direct_input,
        "reward",
        lambda value: value.resize((value.shape[0], 1), refcheck=False),
    )
    wrong_dtype = dataclasses.replace(
        direct_input,
        trace=dataclasses.replace(
            direct_input.trace,
            reward=np.asarray(direct_input.trace.reward, dtype=np.float64),
        ),
    )
    nonfinite = _replace_trace_field(
        direct_input,
        "helper_selected_value",
        lambda value: value.__setitem__(0, np.float32(np.nan)),
    )

    shape_report = audit_hidden_regime_trace(wrong_shape)
    dtype_report = audit_hidden_regime_trace(wrong_dtype)
    finite_report = audit_hidden_regime_trace(nonfinite)

    assert not shape_report.valid
    assert any("trace.reward.shape" in path for path in shape_report.mismatches)
    assert not dtype_report.valid
    assert any("trace.reward.dtype" in path for path in dtype_report.mismatches)
    assert not finite_report.valid
    assert any("contains nonfinite" in path for path in finite_report.mismatches)
    assert shape_report.rows_checked == dtype_report.rows_checked == finite_report.rows_checked == 0


def test_truncated_reordered_and_gapped_traces_fail(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    truncated_trace = _transform_all_rows(direct_input.trace, lambda value: value[:-1])
    permutation = np.arange(direct_input.config.num_steps)
    permutation[0], permutation[1] = permutation[1], permutation[0]
    reordered_trace = _transform_all_rows(
        direct_input.trace,
        lambda value: value[permutation],
    )
    gapped = _replace_trace_field(
        direct_input,
        "step_index",
        lambda value: value.__setitem__(5, np.int32(9)),
    )

    truncated = audit_hidden_regime_trace(
        dataclasses.replace(direct_input, trace=truncated_trace)
    )
    reordered = audit_hidden_regime_trace(
        dataclasses.replace(direct_input, trace=reordered_trace)
    )
    gap = audit_hidden_regime_trace(gapped)

    assert not truncated.valid
    assert truncated.rows_checked == 0
    assert any(".shape" in path for path in truncated.mismatches)
    assert not reordered.valid
    assert "trace.step_index.sequence" in reordered.mismatches
    assert any("initial." in path or "continuity" in path for path in reordered.mismatches)
    assert not gap.valid
    assert "trace.step_index.sequence" in gap.mismatches


@pytest.mark.parametrize(
    ("field", "mutate", "path_fragment"),
    (
        (
            "helper_value_bits_post",
            lambda value: value.__setitem__((0, 0, 0, 0), value[0, 0, 0, 0] ^ np.uint32(1)),
            "helper.oracle",
        ),
        (
            "helper_decision_action",
            lambda value: value.__setitem__(0, np.int32((int(value[0]) + 1) % 3)),
            "helper.oracle.decision.action",
        ),
        (
            "beneficiary_candidate_value",
            lambda value: value.__setitem__(0, np.float32(value[0] + np.float32(0.25))),
            "beneficiary.oracle.diagnostics.candidate_value",
        ),
        (
            "world_cue_key_data_post",
            lambda value: value.__setitem__((0, 0), value[0, 0] ^ np.uint32(1)),
            "world.oracle.new_state.cue_key_data",
        ),
        (
            "oracle_target",
            lambda value: value.__setitem__(0, np.int8((int(value[0]) + 1) % 3)),
            "world.oracle.diagnostics.oracle_target",
        ),
        (
            "world_terminated",
            lambda value: value.__setitem__(0, np.bool_(not bool(value[0]))),
            "world.oracle.diagnostics.terminated",
        ),
        (
            "world_discount",
            lambda value: value.__setitem__(0, np.float32(0.5)),
            "world.oracle.diagnostics.discount",
        ),
        (
            "helper_private_input",
            lambda value: value.__setitem__(0, np.int32((int(value[0]) + 1) % 3)),
            "binding.helper_private_input_to_world_cue",
        ),
        (
            "helper_write_enabled",
            lambda value: value.__setitem__(0, np.bool_(not bool(value[0]))),
            "binding.helper_write_permit",
        ),
        (
            "lifecycle_synchronized",
            lambda value: value.__setitem__(0, np.bool_(not bool(value[0]))),
            "lifecycle_synchronized",
        ),
        (
            "helper_selective_mutation_violation",
            lambda value: value.__setitem__((0, 1), np.bool_(not bool(value[0, 1]))),
            "helper.selective_mutation_violation",
        ),
    ),
)
def test_major_trace_field_classes_are_independently_tamper_detected(
    direct_input: HiddenRegimeTraceAuditInput,
    field: str,
    mutate: Callable[[np.ndarray[Any, Any]], None],
    path_fragment: str,
) -> None:
    report = audit_hidden_regime_trace(_replace_trace_field(direct_input, field, mutate))

    assert not report.valid
    assert any(path_fragment in path for path in report.mismatches), report.mismatches


def test_relevance_contraction_envelope_uses_exact_candidates_not_a_tolerance(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    tampered = _replace_trace_field(
        direct_input,
        "helper_relevance_mean_post",
        lambda value: value.__setitem__(
            (0, 0),
            np.float32(value[0, 0] + np.float32(0.01)),
        ),
    )

    report = audit_hidden_regime_trace(tampered)

    assert not report.valid
    assert any(
        "rows[0].helper.oracle.new_state.relevance_mean[0]" in path
        for path in report.mismatches
    )
    assert not any(
        item.startswith("rows[0].helper.relevance_mean;")
        for item in report.accepted_float32_contractions
    )


def test_every_admitted_relevance_value_is_an_exact_semantic_candidate(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    errors: list[str] = []
    arrays = trace_audit_module._trace_arrays(
        direct_input.trace,
        direct_input.config.num_steps,
        errors,
    )
    assert arrays is not None, errors
    spec = trace_audit_module._independent_condition_spec(direct_input.condition)
    effective = dataclasses.replace(
        direct_input.config.learner,
        writable_lru_ablation=False,
        durable_write_policy=spec.durable_write_policy,
        replacement_target_policy=spec.replacement_target_policy,
    )
    record = trace_audit_module._row_role_record(
        arrays,
        0,
        ROLE_HELPER,
        SlotRoleOracleConfig.from_config(effective),
        True,
    )
    old_value = np.float32(0.456625)
    gain = np.float32(0.1)
    reward = np.float32(1.0)
    old_means = (float(old_value), 0.2, 0.3, 0.4)
    record = dataclasses.replace(
        record,
        old_state=dataclasses.replace(
            record.old_state,
            relevance_mean=old_means,
            relevance_mass=(566.0, 1.0, 1.0, 1.0),
        ),
        decision=dataclasses.replace(record.decision, slot=0),
        reward=float(reward),
        diagnostics=dataclasses.replace(record.diagnostics, committed_slot=-1),
    )

    candidates = trace_audit_module._relevance_contraction_values(
        old_value,
        gain,
        reward,
    )
    source = np.float32(
        old_value + np.float32(gain * np.float32(reward - old_value))
    )
    fma = np.float32(
        np.float64(old_value)
        + np.float64(gain) * (np.float64(reward) - np.float64(old_value))
    )
    reassociated = np.float32(
        np.float32(old_value - np.float32(gain * old_value))
        + np.float32(gain * reward)
    )
    candidate_bits = {int(value.view(np.uint32)) for value in candidates}
    assert int(source.view(np.uint32)) in candidate_bits
    assert int(fma.view(np.uint32)) in candidate_bits
    assert int(reassociated.view(np.uint32)) in candidate_bits

    for candidate in candidates:
        new_means = (float(candidate), *old_means[1:])
        candidate_record = dataclasses.replace(
            record,
            new_state=dataclasses.replace(
                record.new_state,
                relevance_mean=new_means,
            ),
        )
        match = trace_audit_module._portable_relevance_contraction(
            candidate_record,
            ("new_state.relevance_mean[0]",),
        )
        assert match is not None
        candidate_bits = int(candidate.view(np.uint32))
        assert f"active_value_uint32=0x{candidate_bits:08x}" in match
        assert "formula_ids=" in match
        assert "post_vector_uint32=" in match

    outside = np.float32(old_value + np.float32(0.01))
    outside_record = dataclasses.replace(
        record,
        new_state=dataclasses.replace(
            record.new_state,
            relevance_mean=(float(outside), *old_means[1:]),
        ),
    )
    assert trace_audit_module._portable_relevance_contraction(
        outside_record,
        ("new_state.relevance_mean[0]",),
    ) is None
    assert trace_audit_module._portable_relevance_contraction(
        outside_record,
        ("new_state.relevance_mean[0]", "diagnostics.candidate_relevant"),
    ) is None


def test_relevance_contraction_requires_complete_atomic_commit_movement(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    errors: list[str] = []
    arrays = trace_audit_module._trace_arrays(
        direct_input.trace,
        direct_input.config.num_steps,
        errors,
    )
    assert arrays is not None, errors
    spec = trace_audit_module._independent_condition_spec(direct_input.condition)
    effective = dataclasses.replace(
        direct_input.config.learner,
        writable_lru_ablation=False,
        durable_write_policy=spec.durable_write_policy,
        replacement_target_policy=spec.replacement_target_policy,
    )
    record = trace_audit_module._row_role_record(
        arrays,
        0,
        ROLE_HELPER,
        SlotRoleOracleConfig.from_config(effective),
        True,
    )
    old_value = np.float32(0.456625)
    candidate = trace_audit_module._relevance_contraction_values(
        old_value,
        np.float32(0.1),
        np.float32(1.0),
    )[0]
    record = dataclasses.replace(
        record,
        old_state=dataclasses.replace(
            record.old_state,
            relevance_mean=(float(old_value), 0.2, 0.3, 0.4),
            relevance_mass=(566.0, 1.0, 1.0, 1.0),
        ),
        decision=dataclasses.replace(record.decision, slot=0),
        reward=1.0,
        diagnostics=dataclasses.replace(record.diagnostics, committed_slot=2),
        new_state=dataclasses.replace(
            record.new_state,
            relevance_mean=(0.0, 0.2, float(candidate), 0.4),
        ),
    )
    mismatch_paths = (
        "new_state.relevance_mean[0]",
        "new_state.relevance_mean[2]",
    )

    match = trace_audit_module._portable_relevance_contraction(record, mismatch_paths)
    assert match is not None
    assert f"active_value_uint32=0x{int(candidate.view(np.uint32)):08x}" in match
    assert "post_vector_uint32=0x00000000,0x3e4ccccd," in match

    scratch_not_cleared = dataclasses.replace(
        record,
        new_state=dataclasses.replace(
            record.new_state,
            relevance_mean=(float(candidate), 0.2, float(candidate), 0.4),
        ),
    )
    target_not_copied = dataclasses.replace(
        record,
        new_state=dataclasses.replace(
            record.new_state,
            relevance_mean=(0.0, 0.2, 0.3, 0.4),
        ),
    )
    assert trace_audit_module._portable_relevance_contraction(
        scratch_not_cleared,
        mismatch_paths,
    ) is None
    assert trace_audit_module._portable_relevance_contraction(
        target_not_copied,
        mismatch_paths,
    ) is None


def test_initial_root_key_and_explicit_zero_state_provenance_are_bound(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    key_tamper = _replace_trace_field(
        direct_input,
        "beneficiary_policy_key_data_pre",
        lambda value: value.__setitem__((0, 0), value[0, 0] ^ np.uint32(1)),
    )
    zero_state_tamper = _replace_trace_field(
        direct_input,
        "helper_status_pre",
        lambda value: value.__setitem__((0, 1), np.int8(1)),
    )
    wrong_seed = dataclasses.replace(
        direct_input,
        seed_pair=dataclasses.replace(direct_input.seed_pair, learner_seed=457),
    )

    for tampered, fragment in (
        (key_tamper, "initial.beneficiary"),
        (zero_state_tamper, "initial.helper"),
        (wrong_seed, "initial."),
    ):
        report = audit_hidden_regime_trace(tampered)
        assert not report.valid
        assert any(fragment in path for path in report.mismatches), report.mismatches


def test_final_runtime_state_is_bit_exactly_bound_to_last_post_row(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    helper = direct_input.final_learner_state.helper
    changed_values = helper.values.at[0, 0, 0].set(
        helper.values[0, 0, 0] + jnp.float32(0.25)
    )
    changed_helper = dataclasses.replace(helper, values=changed_values)
    changed_final = dataclasses.replace(
        direct_input.final_learner_state,
        helper=changed_helper,
    )

    report = audit_hidden_regime_trace(
        dataclasses.replace(direct_input, final_learner_state=changed_final)
    )

    assert not report.valid
    assert any("final.helper.continuity.values" in path for path in report.mismatches)


def test_role_swap_is_rejected_by_role_oracles_and_information_flow(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    swapped = dataclasses.replace(direct_input, trace=_swap_roles(direct_input.trace))

    report = audit_hidden_regime_trace(swapped)

    assert not report.valid
    assert any("binding." in path for path in report.mismatches)
    # A complete role swap remains a valid *local* transition for each symmetric
    # role oracle. Named root-key provenance and cross-role information flow are
    # what independently prevent that relabeling.
    assert any("initial.helper.continuity.key_data" in path for path in report.mismatches)
    assert any("final.helper.continuity" in path for path in report.mismatches)


def test_declared_channel_swap_is_rejected_instead_of_trusting_delivery(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    falsely_constant = dataclasses.replace(direct_input, condition=CONSTANT_CHANNEL)

    report = audit_hidden_regime_trace(falsely_constant)

    assert not report.valid
    assert any("world.oracle.delivered_message" in path for path in report.mismatches)


def test_world_and_learner_config_tampering_changes_oracle_expectations(
    direct_input: HiddenRegimeTraceAuditInput,
) -> None:
    permutations = list(direct_input.config.world.regime_permutations)
    permutations[0] = (1, 2, 0)
    changed_world = dataclasses.replace(
        direct_input.config.world,
        regime_permutations=tuple(permutations),
    )
    world_input = dataclasses.replace(
        direct_input,
        config=dataclasses.replace(direct_input.config, world=changed_world),
    )
    changed_learner = dataclasses.replace(direct_input.config.learner, epsilon=0.75)
    learner_input = dataclasses.replace(
        direct_input,
        config=dataclasses.replace(direct_input.config, learner=changed_learner),
    )

    world_report = audit_hidden_regime_trace(world_input)
    learner_report = audit_hidden_regime_trace(learner_input)

    assert not world_report.valid
    assert any("world.oracle" in path for path in world_report.mismatches)
    assert not learner_report.valid
    assert any("helper.oracle.decision" in path for path in learner_report.mismatches)


def test_auditor_source_has_no_evaluator_replay_or_production_transition_calls() -> None:
    source = inspect.getsource(trace_audit_module)

    assert "run_hidden_regime_condition(" not in source
    assert "_scan_runner(" not in source
    assert ".update(" not in source
    assert ".step(" not in source
    assert "SlotSignalingAgent" not in source
    assert "HiddenRegimeSignalingWorld" not in source
