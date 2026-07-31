"""Development-only tests for the minimal genuine learning-partner rung."""

import dataclasses
import inspect
import json
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.signaling_bandit import SignalingBanditAgent
from alberta_framework.evaluation.learning_partner_development import (
    BENEFICIARY_ALONE,
    BENEFICIARY_FROZEN,
    BOTH_FROZEN,
    CONSTANT_ONE_DELIVERY,
    CONSTANT_ZERO_DELIVERY,
    HELPER_FROZEN,
    JOINT_ADAPTIVE,
    MATCHED_CONDITIONS,
    ORACLE_HELPER,
    SHUFFLED_DELIVERY,
    LearningPartnerDevelopmentConfig,
    condition_spec,
    final_cross_play,
    run_beneficiary_alone,
    run_learning_partner,
    run_learning_partner_development,
    validate_learning_partner_development_payload,
)

pytestmark = pytest.mark.development


@pytest.fixture(scope="module")
def learning_config() -> LearningPartnerDevelopmentConfig:
    return LearningPartnerDevelopmentConfig(
        phase_length=128,
        n_phases=4,
        learning_rate=0.1,
        epsilon=0.1,
    )


@pytest.fixture(scope="module")
def joint_runs(learning_config: LearningPartnerDevelopmentConfig):
    return tuple(
        run_learning_partner(JOINT_ADAPTIVE, seed=seed, config=learning_config) for seed in (0, 1)
    )


@pytest.fixture(scope="module")
def control_runs():
    config = LearningPartnerDevelopmentConfig(phase_length=8, n_phases=4)
    return {
        BOTH_FROZEN: run_learning_partner(BOTH_FROZEN, seed=7, config=config),
        SHUFFLED_DELIVERY: run_learning_partner(
            SHUFFLED_DELIVERY,
            seed=7,
            config=config,
        ),
        ORACLE_HELPER: run_learning_partner(ORACLE_HELPER, seed=7, config=config),
    }


@pytest.fixture(scope="module")
def beneficiary_alone_run(learning_config: LearningPartnerDevelopmentConfig):
    return run_beneficiary_alone(seed=7, config=learning_config)


@pytest.fixture(scope="module")
def serialized_report():
    """Tiny truthful run of every condition for serialization audit only."""

    return run_learning_partner_development(
        seeds=(101, 102),
        config=LearningPartnerDevelopmentConfig(phase_length=1, n_phases=4),
    )


@pytest.fixture(scope="module")
def contraction_rounding_report():
    """Short lives containing cells where fused and unfused updates differ."""

    return run_learning_partner_development(
        seeds=(284505771, 3833422168),
        config=LearningPartnerDevelopmentConfig(phase_length=32, n_phases=4),
    )


@pytest.fixture(scope="module")
def default_length_serialized_report():
    """Minimum full report at the default 4,096-step life length."""

    return run_learning_partner_development(
        seeds=(284505771, 3833422168),
        config=LearningPartnerDevelopmentConfig(),
    )


def _float32_bits(value: float | np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).reshape(()).view(np.uint32))


def _portable_update_candidates(
    pre: float,
    reward: float,
    learning_rate: float,
) -> tuple[np.float32, np.float32]:
    pre32 = np.float32(pre)
    reward32 = np.float32(reward)
    rate32 = np.float32(learning_rate)
    delta = np.float32(reward32 - pre32)
    unfused = np.float32(pre32 + np.float32(rate32 * delta))
    fused = np.float32(np.float64(pre32) + np.float64(rate32) * np.float64(delta))
    return unfused, fused


def _first_split_update_with_revisit(
    trace: dict[str, list[object]],
    *,
    role: str,
    learning_rate: float,
) -> tuple[int, int, np.float32]:
    input_name = "cue" if role == "helper" else "delivered_message"
    action_name = "helper_message" if role == "helper" else "beneficiary_action"
    pre_name = f"{role}_value_pre"
    post_name = f"{role}_value_post"
    for step, (pre, reward, actual_post) in enumerate(
        zip(trace[pre_name], trace["reward"], trace[post_name], strict=True)
    ):
        unfused, fused = _portable_update_candidates(
            float(pre),
            float(reward),
            learning_rate,
        )
        if _float32_bits(unfused) == _float32_bits(fused):
            continue
        actual_bits = _float32_bits(float(actual_post))
        if actual_bits == _float32_bits(unfused):
            alternate = fused
        elif actual_bits == _float32_bits(fused):
            alternate = unfused
        else:
            continue
        selected_cell = (
            trace["context"][step],
            trace[input_name][step],
            trace[action_name][step],
        )
        for revisit in range(step + 1, len(trace[pre_name])):
            revisit_cell = (
                trace["context"][revisit],
                trace[input_name][revisit],
                trace[action_name][revisit],
            )
            if revisit_cell == selected_cell:
                return step, revisit, alternate
    raise AssertionError(f"no split {role} update with a later cell revisit")


def test_config_is_strict_and_explicitly_nonpromoting() -> None:
    for kwargs in (
        {"phase_length": True},
        {"phase_length": 0},
        {"n_phases": False},
        {"n_phases": 1},
        {"n_phases": 3},
        {"learning_rate": float("nan")},
        {"epsilon": float("inf")},
    ):
        with pytest.raises(ValueError):
            LearningPartnerDevelopmentConfig(**kwargs)  # type: ignore[arg-type]
    payload = LearningPartnerDevelopmentConfig().to_dict()
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["claim_thresholds_frozen"] is False


def test_learner_public_api_has_no_oracle_target_input() -> None:
    helper_parameters = inspect.signature(SignalingBanditAgent.select_helper).parameters
    beneficiary_parameters = inspect.signature(SignalingBanditAgent.select_beneficiary).parameters
    update_parameters = inspect.signature(SignalingBanditAgent.update).parameters
    assert "target" not in helper_parameters
    assert "target" not in beneficiary_parameters
    assert "target" not in update_parameters
    assert set(helper_parameters) == {"self", "state", "public_context", "private_cue"}
    assert set(beneficiary_parameters) == {
        "self",
        "state",
        "public_context",
        "delivered_message",
    }


def test_all_condition_interventions_are_exact_and_resource_shape_neutral() -> None:
    specs = {condition: condition_spec(condition) for condition in MATCHED_CONDITIONS}
    assert set(specs) == set(MATCHED_CONDITIONS)
    assert specs[JOINT_ADAPTIVE].helper_write
    assert specs[JOINT_ADAPTIVE].beneficiary_write
    assert not specs[HELPER_FROZEN].helper_write
    assert specs[HELPER_FROZEN].beneficiary_write
    assert specs[BENEFICIARY_FROZEN].helper_write
    assert not specs[BENEFICIARY_FROZEN].beneficiary_write
    assert not specs[BOTH_FROZEN].helper_write
    assert not specs[BOTH_FROZEN].beneficiary_write
    assert specs[CONSTANT_ZERO_DELIVERY].channel == "constant_0"
    assert specs[CONSTANT_ONE_DELIVERY].channel == "constant_1"
    assert specs[SHUFFLED_DELIVERY].channel == "shuffled"
    assert specs[ORACLE_HELPER].oracle_helper
    # Specs contain interventions only; none can add or remove a table.
    assert {field.name for field in dataclasses.fields(next(iter(specs.values())))} == {
        "channel",
        "helper_write",
        "beneficiary_write",
        "oracle_helper",
    }


def test_joint_roles_actually_learn_on_multiple_development_seeds(joint_runs) -> None:
    for run in joint_runs:
        assert run.final_helper_state is not None
        assert np.count_nonzero(np.asarray(run.final_helper_state.values)) >= 4
        assert np.count_nonzero(np.asarray(run.final_beneficiary_state.values)) >= 4
        np.testing.assert_array_equal(
            np.asarray(run.probes.joint_accuracy)[-1],
            np.ones((2,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(run.probes.helper_cue_dependence)[-1],
            np.ones((2,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(run.probes.beneficiary_message_dependence)[-1],
            np.ones((2,), dtype=np.float32),
        )


def test_prequential_trace_records_exact_old_state_updates(joint_runs) -> None:
    trace = joint_runs[0].trace
    assert trace.context.dtype == jnp.int8
    assert trace.cue.dtype == jnp.int8
    assert trace.helper_message.dtype == jnp.int8
    assert trace.reward.dtype == jnp.float32
    for field in dataclasses.fields(trace):
        assert bool(jnp.all(jnp.isfinite(getattr(trace, field.name))))
    expected_helper_post = np.asarray(trace.helper_value_pre) + 0.1 * (
        np.asarray(trace.reward) - np.asarray(trace.helper_value_pre)
    )
    expected_beneficiary_post = np.asarray(trace.beneficiary_value_pre) + 0.1 * (
        np.asarray(trace.reward) - np.asarray(trace.beneficiary_value_pre)
    )
    np.testing.assert_allclose(trace.helper_value_post, expected_helper_post, atol=1e-7)
    np.testing.assert_allclose(
        trace.beneficiary_value_post,
        expected_beneficiary_post,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        trace.oracle_target,
        np.bitwise_xor(np.asarray(trace.cue), np.asarray(trace.context)),
    )


def test_phase_boundary_probes_cover_recurrence_without_replay(joint_runs) -> None:
    for run in joint_runs:
        probes = run.probes
        assert np.asarray(probes.helper_messages).shape == (4, 2, 2)
        assert np.asarray(probes.beneficiary_actions).shape == (4, 2, 2)
        assert np.asarray(probes.joint_accuracy).shape == (4, 2)
        np.testing.assert_array_equal(probes.phase_context, np.asarray((0, 1, 0, 1)))
        np.testing.assert_array_equal(
            probes.context_recovery_valid,
            np.asarray(
                (
                    (False, False),
                    (False, False),
                    (True, False),
                    (False, True),
                )
            ),
        )
        assert bool(jnp.all(jnp.isfinite(probes.context_forgetting)))
        assert bool(jnp.all(jnp.isfinite(probes.context_recovery)))
        np.testing.assert_allclose(
            probes.best_constant_accuracy,
            np.full((4, 2), 0.5, dtype=np.float32),
        )
        np.testing.assert_allclose(
            probes.gain_over_best_constant,
            np.asarray(probes.joint_accuracy) - 0.5,
        )
        np.testing.assert_allclose(
            probes.message_flip_effect[-1],
            np.ones((2,), dtype=np.float32),
        )


def test_frozen_shuffled_and_oracle_controls_are_isolated_and_matched(
    joint_runs,
    control_runs,
) -> None:
    matched_bytes = joint_runs[0].resource.total_state_bytes
    matched_scalars = joint_runs[0].resource.total_state_scalars
    for run in (*joint_runs, *control_runs.values()):
        assert run.resource.resource_matched is True
        assert run.resource.total_state_bytes == matched_bytes == 80
        assert run.resource.total_state_scalars == matched_scalars == 20

    frozen = control_runs[BOTH_FROZEN]
    assert not bool(jnp.any(frozen.trace.helper_write))
    assert not bool(jnp.any(frozen.trace.beneficiary_write))
    assert frozen.final_helper_state is not None
    np.testing.assert_array_equal(frozen.final_helper_state.values, np.zeros((2, 2, 2)))
    np.testing.assert_array_equal(
        frozen.final_beneficiary_state.values,
        np.zeros((2, 2, 2)),
    )

    shuffled = control_runs[SHUFFLED_DELIVERY]
    assert np.any(
        np.asarray(shuffled.trace.delivered_message) != np.asarray(shuffled.trace.helper_message)
    )

    oracle = control_runs[ORACLE_HELPER]
    assert oracle.final_helper_state is not None
    np.testing.assert_array_equal(oracle.trace.helper_message, oracle.trace.oracle_target)
    np.testing.assert_array_equal(oracle.final_helper_state.values, np.zeros((2, 2, 2)))
    assert not bool(jnp.any(oracle.trace.helper_write))
    assert np.count_nonzero(np.asarray(oracle.final_beneficiary_state.values)) > 0


def test_beneficiary_alone_is_separate_and_resource_mismatch_is_disclosed(
    joint_runs,
    beneficiary_alone_run,
) -> None:
    alone = beneficiary_alone_run
    assert alone.condition == BENEFICIARY_ALONE
    assert alone.final_helper_state is None
    assert alone.resource.resource_matched is False
    assert alone.resource.resource_mismatch_disclosed is True
    assert alone.resource.helper_state_bytes == 0
    assert alone.resource.total_state_bytes == 40
    assert alone.resource.total_state_bytes < joint_runs[0].resource.total_state_bytes


def test_final_cross_play_is_read_only_full_matrix_with_fixed_derangement(joint_runs) -> None:
    helper_before = [
        np.asarray(run.final_helper_state.values).copy()  # type: ignore[union-attr]
        for run in joint_runs
    ]
    beneficiary_before = [
        np.asarray(run.final_beneficiary_state.values).copy() for run in joint_runs
    ]
    cross_play = final_cross_play(joint_runs)
    assert cross_play.score_matrix.shape == (2, 2)
    np.testing.assert_array_equal(cross_play.derangement_columns, np.asarray((1, 0)))
    np.testing.assert_array_equal(
        cross_play.within_dyad_scores,
        np.diag(cross_play.score_matrix),
    )
    assert np.all((cross_play.score_matrix >= 0.0) & (cross_play.score_matrix <= 1.0))
    for index, run in enumerate(joint_runs):
        np.testing.assert_array_equal(run.final_helper_state.values, helper_before[index])
        np.testing.assert_array_equal(
            run.final_beneficiary_state.values,
            beneficiary_before[index],
        )


def test_serialization_validator_is_fail_closed_and_never_promotes(
    serialized_report,
) -> None:
    payload = serialized_report.to_dict(include_traces=True)
    json.dumps(payload, allow_nan=False)
    assert validate_learning_partner_development_payload(payload) == ()
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["claim_thresholds_frozen"] is False
    assert payload["acceptance_status"] == "descriptive_only_no_acceptance_gate"
    assert payload["traces_included"] is True
    assert "disjoint table rows" in payload["retention_scope"]
    assert "not catastrophic-forgetting evidence" in payload["retention_scope"]
    assert "RNG key" in payload["frozen_role_semantics"]

    promoted = deepcopy(payload)
    promoted["scientific_promotion_allowed"] = True
    assert "scientific_promotion_allowed must be false" in (
        validate_learning_partner_development_payload(promoted)
    )
    bad_cross_play = deepcopy(payload)
    bad_cross_play["cross_play"]["derangement_columns"] = [0, 1]
    assert "cross_play primary assignment must be a derangement" in (
        validate_learning_partner_development_payload(bad_cross_play)
    )


def test_default_4096_step_report_validates_both_float32_orders(
    default_length_serialized_report,
) -> None:
    payload = default_length_serialized_report.to_dict(include_traces=True)
    assert payload["config"]["num_steps"] == 4096
    assert validate_learning_partner_development_payload(payload) == ()


def test_validator_rejects_two_ulp_and_closed_write_tampering(
    contraction_rounding_report,
) -> None:
    two_ulp = contraction_rounding_report.to_dict(include_traces=True)
    post = np.float32(two_ulp["matched_runs"][0]["trace"]["helper_value_post"][0])
    direction = np.float32(np.inf if post < 1.0 else -np.inf)
    tampered = np.nextafter(
        np.nextafter(post, direction, dtype=np.float32),
        direction,
        dtype=np.float32,
    )
    two_ulp["matched_runs"][0]["trace"]["helper_value_post"][0] = float(tampered)
    assert any(
        "helper_value_post portable float32 update mismatch at step 0" in error
        for error in validate_learning_partner_development_payload(two_ulp)
    )

    closed = contraction_rounding_report.to_dict(include_traces=True)
    helper_frozen = closed["matched_runs"][1]
    frozen_post = np.float32(helper_frozen["trace"]["helper_value_post"][0])
    helper_frozen["trace"]["helper_value_post"][0] = float(
        np.nextafter(frozen_post, np.float32(np.inf), dtype=np.float32)
    )
    assert any(
        "helper_value_post closed-write bit mismatch at step 0" in error
        for error in validate_learning_partner_development_payload(closed)
    )


def test_validator_commits_allowed_one_ulp_posts_before_continuity_checks(
    contraction_rounding_report,
) -> None:
    for role in ("helper", "beneficiary"):
        payload = contraction_rounding_report.to_dict(include_traces=True)
        joint = payload["matched_runs"][0]
        trace = joint["trace"]
        step, revisit, alternate = _first_split_update_with_revisit(
            trace,
            role=role,
            learning_rate=float(payload["config"]["learning_rate"]),
        )
        original = np.float32(trace[f"{role}_value_post"][step])
        assert abs(_float32_bits(original) - _float32_bits(alternate)) == 1
        # This alternate post is individually permitted.  The validator must
        # commit it, making the untouched pre at the next cell visit fail
        # bit-exact continuity instead of silently recomputing a host result.
        trace[f"{role}_value_post"][step] = float(alternate)
        errors = validate_learning_partner_development_payload(payload)
        assert any(
            f"{role}_value_pre routing mismatch (bit-continuity) at step {revisit}" in error
            for error in errors
        )


def test_serialization_validator_rejects_nested_and_reconstructed_tampering(
    serialized_report,
) -> None:
    payload = serialized_report.to_dict(include_traces=True)

    nested_promotion = deepcopy(payload)
    nested_promotion["matched_runs"][0]["scientific_promotion_allowed"] = True
    assert any(
        "matched_runs[0].scientific_promotion_allowed" in error
        for error in validate_learning_partner_development_payload(nested_promotion)
    )

    nested_scope = deepcopy(payload)
    del nested_scope["matched_runs"][0]["retention_scope"]
    assert any(
        "matched_runs[0].retention_scope mismatch" in error
        for error in validate_learning_partner_development_payload(nested_scope)
    )

    wrong_namespace = deepcopy(payload)
    wrong_namespace["seed_namespace"] = "held_out"
    assert "seed_namespace mismatch" in validate_learning_partner_development_payload(
        wrong_namespace
    )

    wrong_condition = deepcopy(payload)
    wrong_condition["matched_runs"][0]["condition"] = "not_a_condition"
    errors = validate_learning_partner_development_payload(wrong_condition)
    assert any("condition/order mismatch" in error for error in errors)
    assert any("condition is not recognized" in error for error in errors)

    bad_resource = deepcopy(payload)
    bad_resource["matched_runs"][0]["resource"]["total_state_bytes"] = 79
    assert any(
        "resource.total_state_bytes mismatch" in error
        for error in validate_learning_partner_development_payload(bad_resource)
    )

    bad_trace = deepcopy(payload)
    bad_trace["matched_runs"][0]["trace"]["helper_value_pre"][0] = 0.25
    assert any(
        "helper_value_pre routing mismatch" in error
        for error in validate_learning_partner_development_payload(bad_trace)
    )

    bad_probe = deepcopy(payload)
    bad_probe["matched_runs"][0]["probes"]["joint_accuracy"][0][0] = 0.25
    assert any(
        "probes.joint_accuracy trace reconstruction mismatch" in error
        for error in validate_learning_partner_development_payload(bad_probe)
    )

    bad_final = deepcopy(payload)
    bad_final["matched_runs"][0]["final_helper_values"][0][0][0] = 0.25
    assert any(
        "final_helper_values trace reconstruction mismatch" in error
        for error in validate_learning_partner_development_payload(bad_final)
    )

    bad_primary = deepcopy(payload)
    bad_primary["cross_play"]["primary_comparison"] = "post_hoc_best_pairing"
    assert "cross_play.primary_comparison mismatch" in (
        validate_learning_partner_development_payload(bad_primary)
    )

    fractional_columns = deepcopy(payload)
    fractional_columns["cross_play"]["derangement_columns"] = [1.9, 0.1]
    assert "cross_play.derangement_columns shape mismatch" in (
        validate_learning_partner_development_payload(fractional_columns)
    )

    bad_matrix = deepcopy(payload)
    bad_matrix["cross_play"]["score_matrix"][0][0] = 0.25
    assert any(
        "score_matrix final-table reconstruction mismatch" in error
        for error in validate_learning_partner_development_payload(bad_matrix)
    )

    bad_mean = deepcopy(payload)
    bad_mean["cross_play"]["within_dyad_mean"] = 0.123
    assert "cross_play.within_dyad_mean reconstruction mismatch" in (
        validate_learning_partner_development_payload(bad_mean)
    )

    omitted = serialized_report.to_dict(include_traces=False)
    assert "full validation requires traces_included=true" in (
        validate_learning_partner_development_payload(omitted)
    )
