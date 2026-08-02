"""Adversarial unit contracts for the scan-free v6 matched-suite validator."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation import (
    hidden_partner_lifecycle_world_v6_matched_suite as suite_module,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    build_v6_diagnostic_controls,
    build_v6_primary_controls,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_matched_suite import (
    STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE,
    STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE,
    V6MatchedDevelopmentSuite,
    build_v6_matched_development_suite,
    validate_v6_matched_development_suite,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    V6_INITIAL_STREAM_BIT_ORDER,
    V6_TRANSITION_STREAM_BIT_ORDER,
    HiddenPartnerLifecycleWorldV6Runner,
    V6DevelopmentRun,
    V6ResourceRecord,
    V6RngRecord,
    V6SourceClosureHash,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runtime import V6RuntimeRecord
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    BASE_SEGMENT_LENGTHS,
    MAX_SCAN_STEPS,
    HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    build_hidden_partner_lifecycle_world_v6_scan_plan,
    require_v6_control_suite_ready,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
    STRUCTURALLY_INVALID_DEVELOPMENT_RUN,
    STRUCTURALLY_VALID_DEVELOPMENT_RUN,
    V6DevelopmentRunValidation,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    validate_hidden_partner_lifecycle_world_v6_development_run as real_member_validator,
)
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldOnlineResourceBudget,
)

pytestmark = pytest.mark.unit


def _canonical_member_validation() -> V6DevelopmentRunValidation:
    return V6DevelopmentRunValidation(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_VALIDATOR_SCHEMA,
        status=STRUCTURALLY_VALID_DEVELOPMENT_RUN,
        development_only=True,
        structural_only=True,
        replay_verified=False,
        execution_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        errors=(),
        lifecycle=cast(Any, None),
        coverage=cast(Any, None),
        quality=cast(Any, None),
    )


@pytest.fixture(autouse=True)
def _structurally_valid_members(monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = _canonical_member_validation()
    monkeypatch.setattr(
        suite_module,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        lambda _run: verdict,
    )


@pytest.fixture(scope="module")
def readiness() -> HiddenPartnerLifecycleWorldV6ControlSuiteReadiness:
    return require_v6_control_suite_ready()


@functools.lru_cache(maxsize=4)
def _real_control_initial_states(seed: int = 2) -> tuple[object, ...]:
    controls = (*build_v6_primary_controls(), *build_v6_diagnostic_controls())
    world_key = jr.key(2_000 + seed)
    agent_key = jr.key(4_000 + seed)
    with jax.disable_jit():
        return tuple(
            HiddenPartnerLifecycleWorldV6Runner(control).initialize(world_key, agent_key)
            for control in controls
        )


def _resources() -> V6ResourceRecord:
    budget = HiddenPartnerWorldOnlineResourceBudget(
        world_state_nbytes=100,
        agent_state_nbytes=200,
        filter_state_nbytes=20,
        component_state_nbytes=320,
        config_token_nbytes=32,
        action_nbytes=4,
        valid_nbytes=1,
        step_count_nbytes=4,
        bridge_metadata_nbytes=41,
        total_state_nbytes=361,
        world_replay_capacity=0,
        agent_replay_capacity=0,
        replay_capacity=0,
    )
    signature = (((2,), "float32"), ((2,), "int32"))
    return V6ResourceRecord(
        initial=budget,
        final=budget,
        peak_total_state_nbytes=budget.total_state_nbytes,
        static_total_state_nbytes=True,
        zero_replay=True,
        initial_tree_signature=signature,
        final_tree_signature=signature,
        tree_structure_equal=True,
        tree_signature_equal=True,
    )


def _rng(keys: jnp.ndarray) -> V6RngRecord:
    return V6RngRecord(  # type: ignore[call-arg]
        supplied_key_data=keys,
        initial_world_key_data=jnp.arange(10, dtype=jnp.uint32).reshape((5, 2)),
        final_world_key_data=jnp.arange(10, 20, dtype=jnp.uint32).reshape((5, 2)),
        initial_policy_key_data=jnp.arange(4, dtype=jnp.uint32).reshape((2, 2)),
        final_policy_key_data=jnp.arange(4, 8, dtype=jnp.uint32).reshape((2, 2)),
        initial_interaction_key_data=jnp.asarray((31, 32), dtype=jnp.uint32),
        final_interaction_key_data=jnp.asarray((41, 42), dtype=jnp.uint32),
        initial_stream_bits=jnp.asarray(0, dtype=jnp.uint8),
        world_draw_counts=jnp.full((5,), 30_000, dtype=jnp.int32),
        interaction_key_advance_count=jnp.asarray(30_000, dtype=jnp.int32),
        policy_decision_count=jnp.asarray(30_001, dtype=jnp.int32),
    )


def _build_suite(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> V6MatchedDevelopmentSuite:
    keys = jnp.stack((jr.key_data(jr.key(2_002)), jr.key_data(jr.key(4_002))))
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    source = (V6SourceClosureHash(relative_path="shared.py", sha256="a" * 64),)
    runtime = cast(V6RuntimeRecord, object())
    initial_states = _real_control_initial_states()
    assert len(initial_states) == len(readiness.bindings)
    resources = _resources()
    stream = jnp.zeros((MAX_SCAN_STEPS,), dtype=jnp.uint8)
    runs = tuple(
        V6DevelopmentRun(
            control_name=binding.name,
            primary=binding.family == "primary",
            plan=plan,
            control_config_sha256=binding.control_config_sha256,
            control_matrix_sha256=readiness.control_matrix_sha256,
            bridge_config_sha256=binding.bridge_config_sha256,
            runner_config_sha256="b" * 64,
            source_closure_hashes=source,
            runtime=runtime,
            initial_state=cast(Any, initial_state),
            final_state=cast(Any, None),
            windows=cast(Any, None),
            row_heads=cast(Any, None),
            filter_totals=cast(Any, None),
            action_totals=cast(Any, None),
            audits=cast(Any, None),
            ledger=cast(Any, None),
            lifecycle=cast(Any, None),
            rng=_rng(keys),
            resources=resources,
            stream_code=stream,
        )
        for binding, initial_state in zip(readiness.bindings, initial_states, strict=True)
    )
    return build_v6_matched_development_suite(keys, runs)


def _codes(result: object) -> set[str]:
    return {error.code for error in result.errors}  # type: ignore[attr-defined]


class _EqualitySpoof:
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False


def _replace_run(
    suite: V6MatchedDevelopmentSuite,
    index: int,
    **changes: object,
) -> V6MatchedDevelopmentSuite:
    runs = list(suite.runs)
    runs[index] = dataclasses.replace(runs[index], **changes)  # type: ignore[arg-type]
    return dataclasses.replace(suite, runs=tuple(runs))


def _index(suite: V6MatchedDevelopmentSuite, name: str) -> int:
    return next(index for index, run in enumerate(suite.runs) if run.control_name == name)


@pytest.mark.parametrize("field", ("schema", "status"))
def test_envelope_text_fields_require_exact_builtin_str(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    field: str,
) -> None:
    suite = _build_suite(readiness)
    result = validate_v6_matched_development_suite(
        dataclasses.replace(suite, **{field: _EqualitySpoof()})
    )

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "TYPE" in _codes(result)


def test_complete_canonical_panel_is_structurally_valid_only(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    result = validate_v6_matched_development_suite(suite)

    assert result.status == STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE
    assert result.validated_member_count == 18
    assert result.structural_only
    assert not result.replay_verified
    assert not result.execution_authorized
    assert not result.evidence_authorized
    assert not result.scientific_promotion_allowed
    assert result.errors == ()


def test_real_18_control_initial_states_exhaust_the_reviewed_allowlist(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    baseline = suite.runs[0].initial_state
    nodes, baseline_leaves = suite_module._initial_state_schema_and_leaves(  # noqa: SLF001
        baseline
    )
    assert len(nodes) == suite_module.V6_MATCHED_INITIAL_STATE_NODE_COUNT
    assert len(baseline_leaves) == suite_module.V6_MATCHED_INITIAL_STATE_LEAF_COUNT

    witnessed_by_control: dict[tuple[str, bool], set[str]] = {
        (run.control_name, run.primary): set() for run in suite.runs
    }
    for seed in (0, 2, 4, 6):
        initial_states = _real_control_initial_states(seed)
        seed_baseline = initial_states[0]
        for run, initial_state in zip(suite.runs, initial_states, strict=True):
            seeded_run = dataclasses.replace(run, initial_state=initial_state)
            failures = suite_module._initial_state_comparison_failures(  # noqa: SLF001
                seed_baseline,
                seeded_run,
            )
            assert failures == (), (seed, run.control_name, failures)
            differing = suite_module._initial_state_differing_leaf_paths(  # noqa: SLF001
                seed_baseline,
                initial_state,
            )
            witnessed_by_control[(run.control_name, run.primary)].update(differing)

    for (
        control_key,
        allowed_paths,
    ) in suite_module.V6_MATCHED_INITIAL_CONTROL_ALLOWED_DIFFERENCE_PATHS:
        witnessed = witnessed_by_control[control_key]
        for allowed_path in allowed_paths:
            assert allowed_path in witnessed, (control_key, allowed_path, sorted(witnessed))

    always_allowed_witnesses = {
        "config_token": baseline.replace(
            config_token=baseline.config_token.at[0].add(jnp.uint8(1))
        ),
    }
    assert set(always_allowed_witnesses) == set(
        suite_module.V6_MATCHED_INITIAL_ALWAYS_ALLOWED_DIFFERENCE_PATHS
    )
    for allowed_path, initial_state in always_allowed_witnesses.items():
        run = dataclasses.replace(suite.runs[0], initial_state=initial_state)
        assert suite_module._initial_state_differing_leaf_paths(  # noqa: SLF001
            baseline,
            initial_state,
        ) == (allowed_path,)
        assert (
            suite_module._initial_state_comparison_failures(  # noqa: SLF001
                baseline,
                run,
            )
            == ()
        )


def test_exported_stream_bit_orders_match_the_reviewed_pack_layout() -> None:
    assert V6_TRANSITION_STREAM_BIT_ORDER == (
        "signal_0_positive",
        "signal_1_positive",
        "signal_2_positive",
        "partner_flipped",
        "world_flipped",
        "cue_0_flipped",
        "cue_1_flipped",
        "outcome_flipped",
    )
    assert V6_INITIAL_STREAM_BIT_ORDER == (
        "signal_0_positive",
        "signal_1_positive",
        "signal_2_positive",
        "world_sign_positive",
        "cue_0_positive",
        "cue_1_positive",
        "previous_outcome_positive",
        "has_partner_history",
    )


@pytest.mark.parametrize("mutation", ("reordered", "missing", "duplicate"))
def test_reordered_missing_and_duplicate_arms_fail_closed(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    mutation: str,
) -> None:
    suite = _build_suite(readiness)
    runs = list(suite.runs)
    if mutation == "reordered":
        runs[0], runs[1] = runs[1], runs[0]
    elif mutation == "missing":
        runs.pop()
    else:
        runs[-1] = runs[0]
    result = validate_v6_matched_development_suite(dataclasses.replace(suite, runs=tuple(runs)))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert _codes(result) & {"MEMBER_COUNT", "MEMBER_ORDER"}


def test_member_key_drift_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "world_credit_off")
    rng = cast(Any, suite.runs[index].rng).replace(
        supplied_key_data=suite.runs[index].rng.supplied_key_data.at[0, 0].add(jnp.uint32(1))
    )
    result = validate_v6_matched_development_suite(_replace_run(suite, index, rng=rng))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "KEY_DRIFT" in _codes(result)


@pytest.mark.parametrize("field", suite_module.V6_MATCHED_RNG_SHARED_FIELD_ORDER)
def test_every_shared_rng_endpoint_and_count_drift_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    field: str,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "state_frozen")
    value = getattr(suite.runs[index].rng, field)
    increment = jnp.asarray(1, dtype=value.dtype)
    drifted = value + increment if value.shape == () else value.at[(0,) * value.ndim].add(increment)
    rng = cast(Any, suite.runs[index].rng).replace(**{field: drifted})
    result = validate_v6_matched_development_suite(_replace_run(suite, index, rng=rng))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "RNG_IDENTITY" in _codes(result)


def test_rng_partition_covers_every_record_field_exactly_once() -> None:
    partitioned = tuple(
        field for _, fields in suite_module.V6_MATCHED_RNG_FIELD_PARTITION for field in fields
    )
    runtime = tuple(field.name for field in dataclasses.fields(V6RngRecord))

    assert len(partitioned) == len(set(partitioned))
    assert set(partitioned) == set(runtime)


def test_shared_initial_parameter_drift_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "state_frozen")
    initial = suite.runs[index].initial_state
    builder = initial.agent.state_builder.replace(
        parameters=initial.agent.state_builder.parameters.at[0].add(jnp.float32(0.125))
    )
    drifted_agent = initial.agent.replace(state_builder=builder)
    drifted_initial = initial.replace(agent=drifted_agent)
    result = validate_v6_matched_development_suite(
        _replace_run(suite, index, initial_state=drifted_initial)
    )

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "INITIAL_SHARED_STATE" in _codes(result)


def test_shared_initial_learning_history_drift_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "lifecycle_frozen")
    initial = suite.runs[index].initial_state
    interaction = initial.agent.interaction.replace(
        utilities=initial.agent.interaction.utilities.at[0].add(jnp.float32(0.125))
    )
    drifted_initial = initial.replace(agent=initial.agent.replace(interaction=interaction))

    result = validate_v6_matched_development_suite(
        _replace_run(suite, index, initial_state=drifted_initial)
    )

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "INITIAL_SHARED_STATE" in _codes(result)


@pytest.mark.parametrize("component", ("interaction", "control"))
@pytest.mark.parametrize("field", ("birth_timestamp", "uptime_s"))
def test_initial_canonical_lifecycle_clocks_remain_bit_exact_shared_state(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    component: str,
    field: str,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "lifecycle_frozen")
    initial = suite.runs[index].initial_state
    state = getattr(initial.agent, component)
    value = getattr(state, field)
    drifted_state = state.replace(**{field: value + jnp.float32(0.125)})
    drifted_initial = initial.replace(agent=initial.agent.replace(**{component: drifted_state}))

    result = validate_v6_matched_development_suite(
        _replace_run(suite, index, initial_state=drifted_initial)
    )

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "INITIAL_SHARED_STATE" in _codes(result)


def test_equal_cue_may_change_only_transition_cue_bits(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "equal_cue")
    stream = suite.runs[index].stream_code.at[11].set(jnp.uint8(0x20))
    stream = stream.at[12].set(jnp.uint8(0x40))
    rng = cast(Any, suite.runs[index].rng).replace(
        initial_stream_bits=jnp.asarray(0x30, dtype=jnp.uint8)
    )
    result = validate_v6_matched_development_suite(
        _replace_run(suite, index, stream_code=stream, rng=rng)
    )

    assert result.status == STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE
    assert result.errors == ()


def test_equal_cue_noncue_stream_drift_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "equal_cue")
    stream = suite.runs[index].stream_code.at[9].set(jnp.uint8(0x80))
    result = validate_v6_matched_development_suite(_replace_run(suite, index, stream_code=stream))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "STREAM_NONCUE_DRIFT" in _codes(result)


def test_ordinary_arm_cue_bit_drift_is_forbidden(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "grounded_model_frozen")
    stream = suite.runs[index].stream_code.at[7].set(jnp.uint8(0x20))
    result = validate_v6_matched_development_suite(_replace_run(suite, index, stream_code=stream))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "STREAM_DRIFT" in _codes(result)


@pytest.mark.parametrize("bit_index", range(8))
def test_equal_cue_transition_mask_reviews_every_bit(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    bit_index: int,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "equal_cue")
    bit = jnp.asarray(1 << bit_index, dtype=jnp.uint8)
    stream = (
        suite.runs[index]
        .stream_code.at[17]
        .set(jnp.bitwise_xor(suite.runs[index].stream_code[17], bit))
    )

    result = validate_v6_matched_development_suite(_replace_run(suite, index, stream_code=stream))

    cue_bit = V6_TRANSITION_STREAM_BIT_ORDER[bit_index] in {
        "cue_0_flipped",
        "cue_1_flipped",
    }
    expected_status = (
        STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE
        if cue_bit
        else STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    )
    assert result.status == expected_status
    assert ("STREAM_NONCUE_DRIFT" in _codes(result)) is (not cue_bit)


@pytest.mark.parametrize("bit_index", range(8))
def test_equal_cue_initial_mask_reviews_every_bit(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    bit_index: int,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "equal_cue")
    bit = jnp.asarray(1 << bit_index, dtype=jnp.uint8)
    rng = cast(Any, suite.runs[index].rng).replace(
        initial_stream_bits=jnp.bitwise_xor(
            suite.runs[index].rng.initial_stream_bits,
            bit,
        )
    )

    result = validate_v6_matched_development_suite(_replace_run(suite, index, rng=rng))

    cue_bit = V6_INITIAL_STREAM_BIT_ORDER[bit_index] in {
        "cue_0_positive",
        "cue_1_positive",
    }
    expected_status = (
        STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE
        if cue_bit
        else STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    )
    assert result.status == expected_status
    assert ("INITIAL_STREAM_NONCUE_DRIFT" in _codes(result)) is (not cue_bit)


@pytest.mark.parametrize("bit_index", range(8))
def test_ordinary_transition_stream_forbids_every_bit(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    bit_index: int,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "grounded_model_frozen")
    bit = jnp.asarray(1 << bit_index, dtype=jnp.uint8)
    stream = (
        suite.runs[index]
        .stream_code.at[19]
        .set(jnp.bitwise_xor(suite.runs[index].stream_code[19], bit))
    )

    result = validate_v6_matched_development_suite(_replace_run(suite, index, stream_code=stream))

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "STREAM_DRIFT" in _codes(result)


def test_source_runtime_plan_and_resource_drift_are_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "behavior_credit_off")
    lengths = np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32).copy()
    lengths[0] += 1
    lengths[1] -= 1
    alternate_plan = build_hidden_partner_lifecycle_world_v6_scan_plan(lengths)
    cases = (
        (
            _replace_run(
                suite,
                index,
                source_closure_hashes=(
                    V6SourceClosureHash(relative_path="shared.py", sha256="c" * 64),
                ),
            ),
            "SOURCE_IDENTITY",
        ),
        (_replace_run(suite, index, runtime=cast(V6RuntimeRecord, object())), "RUNTIME_IDENTITY"),
        (_replace_run(suite, index, plan=alternate_plan), "PLAN_GEOMETRY"),
        (
            _replace_run(
                suite,
                index,
                resources=dataclasses.replace(
                    suite.runs[index].resources,
                    peak_total_state_nbytes=suite.runs[index].resources.peak_total_state_nbytes + 1,
                ),
            ),
            "RESOURCE_IDENTITY",
        ),
    )
    for mutated, expected_code in cases:
        result = validate_v6_matched_development_suite(mutated)
        assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
        assert expected_code in _codes(result)


def test_noncanonical_control_digest_is_rejected(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
) -> None:
    suite = _build_suite(readiness)
    index = _index(suite, "random_curation")
    result = validate_v6_matched_development_suite(
        _replace_run(suite, index, control_config_sha256="f" * 64)
    )

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert "CONTROL_BINDING" in _codes(result)


def test_one_failed_per_run_structural_verdict_rejects_suite(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_suite(readiness)
    valid = _canonical_member_validation()
    invalid = dataclasses.replace(valid, status=STRUCTURALLY_INVALID_DEVELOPMENT_RUN)
    monkeypatch.setattr(
        suite_module,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        lambda run: invalid if run.control_name == "state_frozen" else valid,
    )

    result = validate_v6_matched_development_suite(suite)

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert result.validated_member_count == 17
    assert "MEMBER_INVALID" in _codes(result)


def test_real_member_validator_negative_composition_is_scan_free(
    readiness: HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_suite(readiness)
    monkeypatch.setattr(
        suite_module,
        "validate_hidden_partner_lifecycle_world_v6_development_run",
        real_member_validator,
    )
    one_member = dataclasses.replace(suite, runs=(suite.runs[0],))

    result = validate_v6_matched_development_suite(one_member)

    assert result.status == STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    assert result.validated_member_count == 0
    assert _codes(result) & {"MEMBER_INVALID", "MEMBER_VALIDATOR"}


def test_positive_real_validator_integration_blocker_is_explicit() -> None:
    assert suite_module.V6_MATCHED_REAL_VALIDATOR_POSITIVE_INTEGRATION_BLOCKER == (
        "positive unmocked composition requires 18 pre-executed structurally valid runs"
    )
