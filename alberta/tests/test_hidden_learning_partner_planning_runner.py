"""Contracts for the gated hidden learning-partner matched-suite runner."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_learning_partner_planning_development as bridge_module
import alberta_framework.evaluation.hidden_learning_partner_planning_runner as runner_module
import alberta_framework.evaluation.hidden_learning_partner_planning_scan_plan as scan_plan_module
from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    BOTH_MODELS_FROZEN,
    JOINT_ADAPTIVE,
    SHUFFLED_DELIVERY,
    HiddenLearningPartnerPlanningConfig,
    HiddenLearningPartnerPlanningRun,
    HiddenLearningPartnerPlanningState,
    HiddenLearningPartnerPlanningTrace,
    HiddenPlanningCondition,
    run_hidden_learning_partner_planning,
    validate_hidden_learning_partner_planning_run,
)
from alberta_framework.evaluation.hidden_learning_partner_planning_runner import (
    ASSESSMENT_STATUS,
    DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT,
    DEVELOPMENT_SEED_ROLE,
    HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA,
    HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS,
    HiddenLearningPartnerPlanningMatchedSuite,
    HiddenLearningPartnerPlanningRunnerError,
    HiddenPlanningAuthenticatedReplayValidation,
    HiddenPlanningCommonRandomNumberAudit,
    HiddenPlanningEvaluatorOwnedStream,
    HiddenPlanningHostQuiescenceSnapshot,
    HiddenPlanningMatchedRunRecord,
    audit_hidden_learning_partner_planning_environment,
    audit_hidden_learning_partner_planning_matched_records,
    authenticate_hidden_learning_partner_planning_development_subpanel,
    authenticate_hidden_learning_partner_planning_matched_suite,
    build_hidden_learning_partner_execution_request,
    build_hidden_learning_partner_planning_run_schedule,
    build_hidden_learning_partner_source_runtime_manifest,
    canonical_hidden_learning_partner_planning_record_keys,
    canonicalize_hidden_learning_partner_planning_records,
    issue_hidden_learning_partner_execution_permit,
    reconstruct_hidden_learning_partner_evaluator_stream,
    run_hidden_learning_partner_planning_matched_suite,
    summarize_hidden_learning_partner_proposal_writes,
    validate_hidden_learning_partner_execution_request,
    validate_hidden_learning_partner_planning_matched_suite,
    validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated,
)
from alberta_framework.evaluation.hidden_learning_partner_planning_scan_plan import (
    CANONICAL_CONDITION_ORDER,
    PAIRED_DEVELOPMENT_SEEDS,
    HiddenLearningPartnerPlanningScanPlan,
    HiddenPlanningSeedBinding,
    build_hidden_learning_partner_planning_scan_plan,
)

pytestmark = pytest.mark.development

_TINY_TEST_ONLY_SEED = 0x71A17E57

type TinyRuns = tuple[
    HiddenLearningPartnerPlanningConfig,
    HiddenPlanningSeedBinding,
    HiddenPlanningEvaluatorOwnedStream,
    dict[str, HiddenLearningPartnerPlanningRun],
]


def _quiescent_snapshot() -> HiddenPlanningHostQuiescenceSnapshot:
    """Build a valid quiet snapshot from this host identity for permit unit tests."""

    live = runner_module._capture_host_quiescence()
    provisional = dataclasses.replace(
        live,
        load_1=0.0,
        load_5=0.0,
        load_15=0.0,
        load_1_per_logical_cpu=0.0,
        runnable_processes=0,
        quiescent=True,
        rejection_reasons=(),
        snapshot_sha256="",
    )
    digest = runner_module._sha256_json(
        runner_module._dataclass_payload_without(provisional, "snapshot_sha256")
    )
    return dataclasses.replace(provisional, snapshot_sha256=digest)


def _busy_snapshot() -> HiddenPlanningHostQuiescenceSnapshot:
    """Build a valid deterministic high-load snapshot for rejection tests."""

    quiet = _quiescent_snapshot()
    load_1 = quiet.max_load_1 + 1.0
    load_per_cpu = load_1 / float(quiet.logical_cpu_count)
    reasons = runner_module._host_rejection_reasons(
        load_1=load_1,
        load_5=quiet.load_5,
        load_15=quiet.load_15,
        load_per_cpu=load_per_cpu,
        runnable_processes=quiet.runnable_processes,
    )
    provisional = dataclasses.replace(
        quiet,
        load_1=load_1,
        load_1_per_logical_cpu=load_per_cpu,
        quiescent=False,
        rejection_reasons=reasons,
        snapshot_sha256="",
    )
    digest = runner_module._sha256_json(
        runner_module._dataclass_payload_without(provisional, "snapshot_sha256")
    )
    return dataclasses.replace(provisional, snapshot_sha256=digest)


def _execution_request(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> runner_module.HiddenPlanningExecutionRequest:
    return build_hidden_learning_partner_execution_request(
        scan_plan,
        execution_mode="eager",
        consumption_acknowledgement=DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT,
    )


@pytest.fixture(scope="module")
def scan_plan() -> HiddenLearningPartnerPlanningScanPlan:
    return build_hidden_learning_partner_planning_scan_plan()


@pytest.fixture(autouse=True)
def isolated_permit_registry() -> Iterator[None]:
    """Keep process-local permit state isolated across mechanism tests."""

    with runner_module._PERMIT_REGISTRY_LOCK:
        runner_module._ISSUED_PERMITS.clear()
        runner_module._REQUEST_PERMITS.clear()
    yield
    with runner_module._PERMIT_REGISTRY_LOCK:
        runner_module._ISSUED_PERMITS.clear()
        runner_module._REQUEST_PERMITS.clear()


@pytest.fixture(scope="module")
def tiny_runs() -> Iterator[TinyRuns]:
    """Exercise one four-step life for every arm, never the default campaign."""

    config = HiddenLearningPartnerPlanningConfig(phase_length=1, n_phases=4)
    # Do not expose outcomes from any of the canonical four development roots.
    # The private builder is used only to obtain the same named-key schema for
    # this synthetic contract test; the production runner never calls it.
    binding_builder = cast(
        Callable[[int, int], HiddenPlanningSeedBinding],
        getattr(scan_plan_module, "_seed_binding"),
    )
    binding = binding_builder(0, _TINY_TEST_ONLY_SEED)
    stream = reconstruct_hidden_learning_partner_evaluator_stream(
        binding,
        config=config,
    )
    runs: dict[str, HiddenLearningPartnerPlanningRun] = {
        condition: run_hidden_learning_partner_planning(
            cast(HiddenPlanningCondition, condition),
            seed=binding.seed,
            config=config,
            jit_compile=False,
        )
        for condition in CANONICAL_CONDITION_ORDER
    }
    yield config, binding, stream, runs
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()


def _tiny_record(
    *,
    condition: str,
    binding: HiddenPlanningSeedBinding,
    stream: HiddenPlanningEvaluatorOwnedStream,
    run: HiddenLearningPartnerPlanningRun,
) -> HiddenPlanningMatchedRunRecord:
    arm_index = CANONICAL_CONDITION_ORDER.index(condition)
    strict_errors = validate_hidden_learning_partner_planning_run(run)
    environment_errors = audit_hidden_learning_partner_planning_environment(run, stream)
    return HiddenPlanningMatchedRunRecord(
        record_index=arm_index,
        seed_index=0,
        seed=binding.seed,
        seed_role=DEVELOPMENT_SEED_ROLE,
        canonical_arm_index=arm_index,
        condition=condition,
        assessment_status=ASSESSMENT_STATUS,
        run=run,
        phase_diagnostics=run.metrics.phase_diagnostics,
        proposal_write_accounting=summarize_hidden_learning_partner_proposal_writes(run),
        strict_run_validation_errors=strict_errors,
        environment_stream_errors=environment_errors,
    )


def _invalid_suite_shell(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> HiddenLearningPartnerPlanningMatchedSuite:
    audit = HiddenPlanningCommonRandomNumberAudit(
        paired_seed_count=0,
        arm_count=0,
        record_count=0,
        evaluator_stream_reconstruction_passed=False,
        action_independent_environment_parity_passed=False,
        shared_initial_state_parity_passed=False,
        cross_arm_trace_key_parity_passed=False,
        final_named_key_parity_passed=False,
        shuffled_channel_output_binding_passed=False,
        canonical_record_order_passed=False,
        errors=("intentionally_incomplete_test_shell",),
    )
    return HiddenLearningPartnerPlanningMatchedSuite(
        schema=HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_SCHEMA,
        status=HIDDEN_LEARNING_PARTNER_PLANNING_RUNNER_STATUS,
        assessment_status=ASSESSMENT_STATUS,
        development_only=True,
        seed_role=DEVELOPMENT_SEED_ROLE,
        consumed_development_seeds=PAIRED_DEVELOPMENT_SEEDS,
        held_out_seeds_used=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        artifact_writes_authorized=False,
        execution_acknowledgement=DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT,
        source_plan_sha256=scan_plan.plan_sha256,
        source_runtime_manifest=build_hidden_learning_partner_source_runtime_manifest(
            execution_mode="eager"
        ),
        execution_request_sha256="0" * 64,
        execution_permit_hmac_sha256="0" * 64,
        suite_binding_sha256="0" * 64,
        authenticated_replay_verified=False,
        canonical_condition_order=CANONICAL_CONDITION_ORDER,
        canonical_record_order=True,
        raw_records_present=True,
        evaluator_streams=(),
        records=(),
        common_random_number_audit=audit,
        aggregate_statistics=None,
        thresholds=None,
        artifact_output_path=None,
    )


def test_schedule_is_inert_complete_paired_and_order_independent(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_life(*args: object, **kwargs: object) -> None:
        raise AssertionError("schedule construction must not run a learner life")

    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        forbidden_life,
    )
    monkeypatch.setattr(
        runner_module,
        "reconstruct_hidden_learning_partner_evaluator_stream",
        forbidden_life,
    )
    forward = build_hidden_learning_partner_planning_run_schedule(scan_plan)
    reverse_order = tuple(reversed(CANONICAL_CONDITION_ORDER))
    reverse = build_hidden_learning_partner_planning_run_schedule(
        scan_plan,
        arm_order=reverse_order,
    )
    canonical_keys = canonical_hidden_learning_partner_planning_record_keys(scan_plan)

    assert len(forward) == len(reverse) == 44
    assert tuple(request.execution_index for request in forward) == tuple(range(44))
    assert tuple((request.seed, request.condition) for request in forward) == canonical_keys
    assert {(request.seed, request.condition) for request in reverse} == set(canonical_keys)
    for seed_index, seed in enumerate(PAIRED_DEVELOPMENT_SEEDS):
        seed_slice = reverse[seed_index * 11 : (seed_index + 1) * 11]
        assert {request.seed for request in seed_slice} == {seed}
        assert tuple(request.condition for request in seed_slice) == reverse_order
        assert {request.canonical_arm_index for request in seed_slice} == set(range(11))


def test_runner_preflight_fails_closed_before_any_default_execution(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def forbidden_life(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("invalid runner preflight reached the life kernel")

    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        forbidden_life,
    )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="acknowledgement"):
        build_hidden_learning_partner_execution_request(
            scan_plan,
            execution_mode="eager",
            consumption_acknowledgement="not-the-exact-acknowledgement",
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="execution request"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=None,
            permit=None,
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="exact tuple"):
        build_hidden_learning_partner_planning_run_schedule(
            scan_plan,
            arm_order=list(CANONICAL_CONDITION_ORDER),
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="duplicate"):
        build_hidden_learning_partner_planning_run_schedule(
            scan_plan,
            arm_order=(CANONICAL_CONDITION_ORDER[0],) * 11,
        )
    tampered = dataclasses.replace(scan_plan, plan_sha256="0" * 64)
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="canonical scan-plan"):
        build_hidden_learning_partner_planning_run_schedule(tampered)
    assert calls == []
    assert validate_hidden_learning_partner_planning_matched_suite(None) == (
        "suite must be an exact HiddenLearningPartnerPlanningMatchedSuite",
    )


def test_explicit_runner_dispatches_to_existing_one_life_kernel(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop at the first call; do not execute a default life in this test."""

    class StopAfterDispatchError(RuntimeError):
        pass

    calls: list[tuple[object, ...]] = []

    def fake_stream(binding: object, **kwargs: object) -> object:
        return SimpleNamespace(seed=getattr(binding, "seed"))

    def stop_after_dispatch(
        condition: object,
        *,
        seed: int,
        config: object,
        jit_compile: bool,
    ) -> None:
        calls.append((condition, seed, config, jit_compile))
        raise StopAfterDispatchError

    quiet = _quiescent_snapshot()
    monkeypatch.setattr(runner_module, "_capture_host_quiescence", lambda: quiet)
    request = _execution_request(scan_plan)
    permit = issue_hidden_learning_partner_execution_permit(request, plan=scan_plan)
    monkeypatch.setattr(
        runner_module,
        "reconstruct_hidden_learning_partner_evaluator_stream",
        fake_stream,
    )
    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        stop_after_dispatch,
    )
    reverse_order = tuple(reversed(CANONICAL_CONDITION_ORDER))
    with pytest.raises(StopAfterDispatchError):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=request,
            permit=permit,
            arm_order=reverse_order,
        )
    assert calls == [
        (
            reverse_order[0],
            PAIRED_DEVELOPMENT_SEEDS[0],
            scan_plan.config,
            False,
        )
    ]
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="already been consumed"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=request,
            permit=permit,
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="completed bound"):
        authenticate_hidden_learning_partner_planning_matched_suite(
            _invalid_suite_shell(scan_plan),
            plan=scan_plan,
            request=request,
            permit=permit,
        )
    assert len(calls) == 1


def test_request_manifest_and_one_run_one_replay_permit_lifecycle_fail_closed(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _execution_request(scan_plan)
    manifest = request.source_runtime_manifest
    assert request.seeds == PAIRED_DEVELOPMENT_SEEDS
    assert request.conditions == CANONICAL_CONDITION_ORDER
    assert request.planned_run_count == 44
    assert request.life_steps == 3_072
    assert request.execution_mode == "eager"
    assert request.consumption_acknowledgement == DEVELOPMENT_EXECUTION_ACKNOWLEDGEMENT
    assert not request.artifact_writes_authorized
    assert not request.evidence_authorized
    assert not request.scientific_promotion_allowed
    assert scan_plan.artifact_output_path is None
    assert (
        "output_path"
        not in inspect.signature(run_hidden_learning_partner_planning_matched_suite).parameters
    )
    assert tuple(source.role for source in manifest.source_files) == (
        "runner",
        "development_kernel",
        "scan_plan",
        "world",
        "signaling_learner",
        "behavior_model",
        "grounded_world_model",
    )
    repository_root = Path(__file__).resolve().parents[1]
    for source in manifest.source_files:
        payload = (repository_root / source.repository_path).read_bytes()
        assert source.nbytes == len(payload) > 0
        assert source.sha256 == hashlib.sha256(payload).hexdigest()
    assert manifest.prng_impl == "threefry2x32"
    assert manifest.prng_key_data_shape == (2,)
    assert manifest.prng_key_data_dtype == "uint32"
    assert len(manifest.manifest_sha256) == len(request.request_sha256) == 64
    tampered_request = dataclasses.replace(
        request,
        source_runtime_manifest=dataclasses.replace(
            manifest,
            jax_version=f"{manifest.jax_version}-tampered",
        ),
    )
    assert validate_hidden_learning_partner_execution_request(
        tampered_request,
        plan=scan_plan,
    )

    quiet = _quiescent_snapshot()
    monkeypatch.setattr(runner_module, "_capture_host_quiescence", lambda: quiet)
    permit = issue_hidden_learning_partner_execution_permit(request, plan=scan_plan)
    assert not permit.artifact_writes_authorized
    assert not permit.evidence_authorized
    assert not permit.scientific_promotion_allowed
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="already has"):
        issue_hidden_learning_partner_execution_permit(request, plan=scan_plan)
    forged = dataclasses.replace(permit, nonce="00" * 32)
    calls: list[str] = []

    def forbidden_stream(*args: object, **kwargs: object) -> None:
        calls.append("stream")
        raise AssertionError("forged permit reached stream reconstruction")

    def forbidden_life(*args: object, **kwargs: object) -> None:
        calls.append("life")
        raise AssertionError("forged permit reached learner execution")

    monkeypatch.setattr(
        runner_module,
        "reconstruct_hidden_learning_partner_evaluator_stream",
        forbidden_stream,
    )
    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        forbidden_life,
    )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="request failed"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=tampered_request,
            permit=permit,
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="permit"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=request,
            permit=forged,
        )
    malformed = dataclasses.replace(permit, issued_time_ns=cast(Any, "not-an-integer"))
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="contract is invalid"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=request,
            permit=malformed,
        )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="completed bound"):
        authenticate_hidden_learning_partner_planning_matched_suite(
            _invalid_suite_shell(scan_plan),
            plan=scan_plan,
            request=request,
            permit=permit,
        )
    assert calls == []

    runner_module._consume_permit_for_run(permit)
    suite_binding = "ab" * 32
    runner_module._bind_completed_suite_to_permit(
        permit,
        suite_binding_sha256=suite_binding,
    )
    runner_module._consume_permit_for_replay(
        permit,
        suite_binding_sha256=suite_binding,
    )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="already consumed"):
        runner_module._consume_permit_for_replay(
            permit,
            suite_binding_sha256=suite_binding,
        )


def test_exact_replay_comparator_preserves_shape_dtype_signed_zero_and_nan_payload_bits() -> None:
    assert runner_module._exact_value_errors(
        0.0,
        -0.0,
        path="scalar",
    ) == ("scalar float bits differ",)
    first_nan = np.asarray(np.uint64(0x7FF8_0000_0000_0001)).view(np.float64).item()
    second_nan = np.asarray(np.uint64(0x7FF8_0000_0000_0002)).view(np.float64).item()
    assert runner_module._exact_value_errors(
        first_nan,
        second_nan,
        path="nan",
    ) == ("nan float bits differ",)
    assert runner_module._exact_value_errors(
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([-0.0], dtype=jnp.float32),
        path="array",
    ) == ("array array bits differ",)
    assert runner_module._exact_value_errors(
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([[1.0]], dtype=jnp.float32),
        path="array",
    ) == ("array array shape or dtype differs",)
    assert runner_module._exact_value_errors(
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float16),
        path="array",
    ) == ("array array shape or dtype differs",)


def test_high_load_rejects_permit_issuance_without_execution(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = _busy_snapshot()
    assert not busy.quiescent
    monkeypatch.setattr(runner_module, "_capture_host_quiescence", lambda: busy)
    request = _execution_request(scan_plan)
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="host is not quiescent"):
        issue_hidden_learning_partner_execution_permit(
            request,
            plan=scan_plan,
        )
    quiet = _quiescent_snapshot()
    monkeypatch.setattr(runner_module, "_capture_host_quiescence", lambda: quiet)
    permit = issue_hidden_learning_partner_execution_permit(request, plan=scan_plan)
    monkeypatch.setattr(runner_module, "_capture_host_quiescence", lambda: busy)
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append("execution")
        raise AssertionError("high-load rejection reached reconstruction or a learner life")

    monkeypatch.setattr(
        runner_module,
        "reconstruct_hidden_learning_partner_evaluator_stream",
        forbidden,
    )
    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        forbidden,
    )
    with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="no longer quiescent"):
        run_hidden_learning_partner_planning_matched_suite(
            scan_plan,
            request=request,
            permit=permit,
        )
    assert calls == []


def test_evaluator_owned_stream_binds_actions_channels_and_raw_phase_trace(
    tiny_runs: TinyRuns,
) -> None:
    config, _, stream, runs = tiny_runs
    assert stream.num_steps == config.num_steps == 4
    for name in (
        "helper_cue",
        "next_helper_cue",
        "oracle_phase_index",
        "oracle_context",
        "oracle_target",
        "shuffled_channel_output",
    ):
        value = np.asarray(getattr(stream, name))
        assert value.shape == (4,)
        assert value.dtype == np.int32
    for name in (
        "cue_key_before",
        "cue_key_after",
        "channel_key_before",
        "channel_key_after",
    ):
        value = np.asarray(getattr(stream, name))
        assert value.shape == (4, 2)
        assert value.dtype == np.uint32

    for run in runs.values():
        assert validate_hidden_learning_partner_planning_run(run) == ()
        assert audit_hidden_learning_partner_planning_environment(run, stream) == ()
        np.testing.assert_array_equal(run.trace.helper_cue, stream.helper_cue)
        np.testing.assert_array_equal(run.trace.next_helper_cue, stream.next_helper_cue)
        assert run.metrics.phase_diagnostics.phase_index == (0, 1, 2, 3)
        assert run.metrics.phase_diagnostics.hidden_context == (0, 1, 0, 1)
    np.testing.assert_array_equal(
        runs[SHUFFLED_DELIVERY].trace.delivered_message,
        stream.shuffled_channel_output,
    )

    tampered = dataclasses.replace(
        stream,
        helper_cue=stream.helper_cue.at[0].set(1 - stream.helper_cue[0]),
    )
    assert "trace.helper_cue differs from the evaluator-owned stream" in (
        audit_hidden_learning_partner_planning_environment(
            runs[JOINT_ADAPTIVE],
            tampered,
        )
    )


def test_proposals_are_counted_separately_from_committed_outer_writes(
    tiny_runs: TinyRuns,
) -> None:
    config, _, _, runs = tiny_runs
    joint = summarize_hidden_learning_partner_proposal_writes(runs[JOINT_ADAPTIVE])
    frozen = summarize_hidden_learning_partner_proposal_writes(runs[BOTH_MODELS_FROZEN])

    assert joint.num_steps == config.num_steps == 4
    assert joint.active_transition_proposal_opportunities == 4
    assert joint.accepted_transitions == 4
    assert joint.rejected_active_transition_proposals == 0
    assert joint.helper_update_proposal_opportunities == joint.helper_committed_writes == 4
    assert (
        joint.beneficiary_update_proposal_opportunities == joint.beneficiary_committed_writes == 4
    )
    assert (
        joint.behavior_update_proposal_opportunities == joint.behavior_applied_proposal_markers == 4
    )
    assert joint.behavior_committed_writes == 4
    assert (
        joint.grounded_update_proposal_opportunities == joint.grounded_applied_proposal_markers == 4
    )
    assert joint.grounded_committed_writes == 4
    assert joint.planner_proposal_opportunities == 4

    assert (
        frozen.behavior_update_proposal_opportunities
        == frozen.behavior_applied_proposal_markers
        == 4
    )
    assert (
        frozen.grounded_update_proposal_opportunities
        == frozen.grounded_applied_proposal_markers
        == 4
    )
    assert frozen.behavior_committed_writes == 0
    assert frozen.grounded_committed_writes == 0
    rejected_trace = dataclasses.replace(
        runs[JOINT_ADAPTIVE].trace,
        accepted=runs[JOINT_ADAPTIVE].trace.accepted.at[0].set(False),
    )
    rejected = summarize_hidden_learning_partner_proposal_writes(
        dataclasses.replace(runs[JOINT_ADAPTIVE], trace=rejected_trace)
    )
    assert rejected.active_transition_proposal_opportunities == 4
    assert rejected.accepted_transitions == 3
    assert rejected.rejected_active_transition_proposals == 1
    assert DEVELOPMENT_SEED_ROLE == "development_consumed_nonpromoting"
    assert ASSESSMENT_STATUS == "not_assessed"


def test_runner_independently_binds_plan_child_exact_words(
    tiny_runs: TinyRuns,
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    config, _, _, runs = tiny_runs
    joint_arm = scan_plan.arms[CANONICAL_CONDITION_ORDER.index(JOINT_ADAPTIVE)]
    tiny_joint_clocks = tuple(
        dataclasses.replace(
            clock,
            final_words=(0, config.num_steps),
            final_telemetry=config.num_steps,
        )
        for clock in joint_arm.exact_child_clocks
    )
    tiny_joint_arm = dataclasses.replace(joint_arm, exact_child_clocks=tiny_joint_clocks)
    joint = runs[JOINT_ADAPTIVE]
    assert runner_module._plan_exact_child_clock_errors(joint, tiny_joint_arm) == ()

    corrupted_behavior = joint.final_state.behavior.replace(
        step_words=joint.final_state.behavior.step_words.at[1].add(jnp.uint32(1))
    )
    corrupted_run = dataclasses.replace(
        joint,
        final_state=joint.final_state.replace(behavior=corrupted_behavior),
    )
    assert (
        "run.final_state.behavior.step_words differs from plan exact child-clock words"
        in runner_module._plan_exact_child_clock_errors(corrupted_run, tiny_joint_arm)
    )

    frozen_arm = scan_plan.arms[CANONICAL_CONDITION_ORDER.index(BOTH_MODELS_FROZEN)]
    frozen = runs[BOTH_MODELS_FROZEN]
    assert runner_module._plan_exact_child_clock_errors(frozen, frozen_arm) == ()
    corrupted_grounded = frozen.final_state.grounded.replace(
        update_count=jnp.asarray(1, dtype=jnp.int32)
    )
    corrupted_frozen = dataclasses.replace(
        frozen,
        final_state=frozen.final_state.replace(grounded=corrupted_grounded),
    )
    assert (
        "run.final_state.grounded.update_count differs from plan child-clock telemetry"
        in runner_module._plan_exact_child_clock_errors(corrupted_frozen, frozen_arm)
    )


def test_tiny_all_arm_panel_audits_crn_and_canonicalizes_reversed_dispatch(
    tiny_runs: TinyRuns,
) -> None:
    config, binding, stream, runs = tiny_runs
    forward_execution = tuple(
        _tiny_record(
            condition=condition,
            binding=binding,
            stream=stream,
            run=runs[condition],
        )
        for condition in CANONICAL_CONDITION_ORDER
    )
    reverse_execution = tuple(reversed(forward_execution))
    canonical_from_forward = canonicalize_hidden_learning_partner_planning_records(
        forward_execution,
        seed_order=(binding.seed,),
    )
    canonical_from_reverse = canonicalize_hidden_learning_partner_planning_records(
        reverse_execution,
        seed_order=(binding.seed,),
    )
    assert all(
        forward is reverse
        for forward, reverse in zip(
            canonical_from_forward,
            canonical_from_reverse,
            strict=True,
        )
    )
    assert tuple(record.condition for record in canonical_from_reverse) == (
        CANONICAL_CONDITION_ORDER
    )

    audit = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(stream,),
        records=canonical_from_reverse,
    )
    assert audit.paired_seed_count == 1
    assert audit.arm_count == 11
    assert audit.record_count == 11
    assert audit.evaluator_stream_reconstruction_passed
    assert audit.action_independent_environment_parity_passed
    assert audit.shared_initial_state_parity_passed
    assert audit.cross_arm_trace_key_parity_passed
    assert audit.final_named_key_parity_passed
    assert audit.shuffled_channel_output_binding_passed
    assert audit.canonical_record_order_passed
    assert audit.errors == ()

    changed_index = 1
    changed = canonical_from_reverse[changed_index]
    key_tampered_trace = cast(
        HiddenLearningPartnerPlanningTrace,
        dataclasses.replace(
            cast(Any, changed.run.trace),
            planner_key_before=changed.run.trace.planner_key_before.at[0, 0].add(jnp.uint32(1)),
        ),
    )
    key_tampered_run = dataclasses.replace(changed.run, trace=key_tampered_trace)
    key_tampered_record = dataclasses.replace(changed, run=key_tampered_run)
    key_tampered_records = list(canonical_from_reverse)
    key_tampered_records[changed_index] = key_tampered_record
    key_audit = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(stream,),
        records=tuple(key_tampered_records),
    )
    assert not key_audit.cross_arm_trace_key_parity_passed
    assert any("planner_key_before" in error for error in key_audit.errors)

    final_key = changed.run.final_state.planner_key
    final_tampered_state = cast(
        HiddenLearningPartnerPlanningState,
        dataclasses.replace(
            cast(Any, changed.run.final_state),
            planner_key=jr.fold_in(final_key, 0xBAD5EED),
        ),
    )
    final_tampered_run = dataclasses.replace(changed.run, final_state=final_tampered_state)
    final_tampered_record = dataclasses.replace(changed, run=final_tampered_run)
    final_tampered_records = list(canonical_from_reverse)
    final_tampered_records[changed_index] = final_tampered_record
    final_audit = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(stream,),
        records=tuple(final_tampered_records),
    )
    assert not final_audit.final_named_key_parity_passed
    assert any("final named key differs: planner_key" in error for error in final_audit.errors)


def test_public_crn_audit_reconstructs_every_stream_field_count_order_and_identity(
    tiny_runs: TinyRuns,
) -> None:
    config, binding, stream, runs = tiny_runs
    records = tuple(
        _tiny_record(
            condition=condition,
            binding=binding,
            stream=stream,
            run=runs[condition],
        )
        for condition in CANONICAL_CONDITION_ORDER
    )
    missing = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(),
        records=records,
    )
    assert not missing.evaluator_stream_reconstruction_passed
    assert "evaluator stream count differs from seed-binding count" in missing.errors

    duplicate = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(stream, stream),
        records=records,
    )
    assert not duplicate.evaluator_stream_reconstruction_passed
    assert "evaluator stream identities are not unique" in duplicate.errors

    field_tampered = dataclasses.replace(
        stream,
        oracle_target=stream.oracle_target.at[0].set(1 - stream.oracle_target[0]),
    )
    field_audit = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(field_tampered,),
        records=records,
    )
    assert not field_audit.evaluator_stream_reconstruction_passed
    assert any("independent reconstruction" in error for error in field_audit.errors)

    malformed_stream = dataclasses.replace(
        stream,
        seed=cast(Any, []),
        helper_cue=cast(Any, None),
    )
    malformed_audit = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding,),
        streams=(malformed_stream,),
        records=records,
    )
    assert not malformed_audit.evaluator_stream_reconstruction_passed
    assert "evaluator stream identities have wrong concrete types" in malformed_audit.errors

    binding_builder = cast(
        Callable[[int, int], HiddenPlanningSeedBinding],
        getattr(scan_plan_module, "_seed_binding"),
    )
    second_binding = binding_builder(1, _TINY_TEST_ONLY_SEED + 1)
    second_stream = reconstruct_hidden_learning_partner_evaluator_stream(
        second_binding,
        config=config,
    )
    reordered = audit_hidden_learning_partner_planning_matched_records(
        config=config,
        bindings=(binding, second_binding),
        streams=(second_stream, stream),
        records=(),
    )
    assert not reordered.evaluator_stream_reconstruction_passed
    assert "evaluator stream order/identities differ from bindings" in reordered.errors


def test_authenticated_replay_bit_compares_complete_tiny_all_arm_panel(
    tiny_runs: TinyRuns,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, binding, stream, runs = tiny_runs
    records = tuple(
        _tiny_record(
            condition=condition,
            binding=binding,
            stream=stream,
            run=runs[condition],
        )
        for condition in CANONICAL_CONDITION_ORDER
    )
    manifest = build_hidden_learning_partner_source_runtime_manifest(execution_mode="eager")
    validation = authenticate_hidden_learning_partner_planning_development_subpanel(
        config=config,
        bindings=(binding,),
        records=records,
        expected_manifest=manifest,
    )
    assert isinstance(validation, HiddenPlanningAuthenticatedReplayValidation)
    assert validation.authenticated_replay_verified
    assert validation.rerun_count == 11
    assert validation.errors == ()

    stale_manifest = dataclasses.replace(manifest, manifest_sha256="0" * 64)
    stale_validation = authenticate_hidden_learning_partner_planning_development_subpanel(
        config=config,
        bindings=(binding,),
        records=records,
        expected_manifest=stale_manifest,
    )
    assert not stale_validation.authenticated_replay_verified
    assert stale_validation.rerun_count == 0
    assert any("current full source/runtime bytes" in error for error in stale_validation.errors)

    empty = authenticate_hidden_learning_partner_planning_development_subpanel(
        config=config,
        bindings=(),
        records=(),
        expected_manifest=manifest,
    )
    assert not empty.authenticated_replay_verified
    assert empty.rerun_count == 0
    assert any("nonempty explicitly supplied" in error for error in empty.errors)

    def cached_run(
        condition: HiddenPlanningCondition,
        *,
        seed: int,
        config: HiddenLearningPartnerPlanningConfig,
        jit_compile: bool,
    ) -> HiddenLearningPartnerPlanningRun:
        assert seed == binding.seed
        assert config == records[0].run.config
        assert not jit_compile
        return runs[condition]

    monkeypatch.setattr(
        bridge_module,
        "run_hidden_learning_partner_planning",
        cached_run,
    )
    changed = records[0]
    changed_world = dataclasses.replace(
        changed.run.final_state.world,
        cue=1 - changed.run.final_state.world.cue,
    )
    changed_final = dataclasses.replace(
        changed.run.final_state,
        world=changed_world,
    )
    changed_record = dataclasses.replace(
        changed,
        run=dataclasses.replace(changed.run, final_state=changed_final),
    )
    tampered_records = (changed_record, *records[1:])
    tampered = authenticate_hidden_learning_partner_planning_development_subpanel(
        config=config,
        bindings=(binding,),
        records=tampered_records,
        expected_manifest=manifest,
    )
    assert not tampered.authenticated_replay_verified
    assert tampered.rerun_count == 11
    assert any(
        "record[0].run.final_state.world.cue array bits differ" in error
        for error in tampered.errors
    )


def test_nested_config_trace_initial_and_final_malformations_reject_without_crashing(
    tiny_runs: TinyRuns,
) -> None:
    _, binding, stream, runs = tiny_runs
    run = runs[JOINT_ADAPTIVE]
    malformed_runs = (
        dataclasses.replace(run, config=cast(Any, None)),
        dataclasses.replace(run, trace=cast(Any, None)),
        dataclasses.replace(run, initial_state=cast(Any, None)),
        dataclasses.replace(
            run,
            final_state=dataclasses.replace(
                run.final_state,
                world=dataclasses.replace(
                    run.final_state.world,
                    cue=jnp.zeros((2,), dtype=jnp.int32),
                ),
            ),
        ),
    )
    for malformed in malformed_runs:
        environment_errors = audit_hidden_learning_partner_planning_environment(
            malformed,
            stream,
        )
        assert environment_errors
        assert environment_errors[0].startswith("nested contract:")
        with pytest.raises(HiddenLearningPartnerPlanningRunnerError, match="nested contract"):
            summarize_hidden_learning_partner_proposal_writes(malformed)

    malformed_record = _tiny_record(
        condition=JOINT_ADAPTIVE,
        binding=binding,
        stream=stream,
        run=run,
    )
    malformed_record = dataclasses.replace(malformed_record, run=malformed_runs[1])
    audit = audit_hidden_learning_partner_planning_matched_records(
        config=run.config,
        bindings=(binding,),
        streams=(stream,),
        records=(malformed_record,),
    )
    assert any("nested contract" in error for error in audit.errors)


def test_suite_validator_rejects_status_authority_and_record_tampering_cheaply(
    scan_plan: HiddenLearningPartnerPlanningScanPlan,
) -> None:
    shell = _invalid_suite_shell(scan_plan)
    status_tampered = dataclasses.replace(shell, assessment_status="accepted")
    assert validate_hidden_learning_partner_planning_matched_suite(status_tampered) == (
        "suite assessment status must remain not_assessed",
    )
    assert validate_hidden_learning_partner_planning_matched_suite_structural_unauthenticated(
        status_tampered
    ) == ("suite assessment status must remain not_assessed",)

    authority_tampered = dataclasses.replace(
        shell,
        evidence_authorized=True,
        scientific_promotion_allowed=True,
    )
    assert validate_hidden_learning_partner_planning_matched_suite(authority_tampered) == (
        "suite carries artifact, evidence, or promotion authority",
    )

    wrong_records = cast(
        Any,
        tuple(None for _ in range(scan_plan.counts.planned_run_count)),
    )
    record_tampered = dataclasses.replace(shell, records=wrong_records)
    record_errors = validate_hidden_learning_partner_planning_matched_suite(record_tampered)
    assert len(record_errors) == scan_plan.counts.planned_run_count
    assert record_errors[0] == "suite record 0 has the wrong concrete type"
    assert record_errors[-1] == "suite record 43 has the wrong concrete type"
