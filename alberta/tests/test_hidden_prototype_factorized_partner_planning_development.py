# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-any-return"
"""Focused contracts for the development-only U1 factorized-planning wrapper."""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation import (
    hidden_prototype_factorized_partner_planning_development as u1_module,
)
from alberta_framework.evaluation.hidden_prototype_factorized_partner_planning_development import (
    ADDITIONAL_WRAPPER_PLANNER_POST_INIT_RANDOM_DRAWS_PER_EVENT,
    ADDITIONAL_WRAPPER_PLANNER_REPLAY_UPDATES_PER_EVENT,
    ALL_TRUE_CALLER_MASK_IS_SAFETY_CERTIFICATION,
    EAGER_JIT_DISCRETE_LEAVES_EXACT,
    EAGER_JIT_FLOAT_ATOL,
    EAGER_JIT_FLOAT_RTOL,
    FACTORIZED_PLANNING_ARMS,
    INHERITED_U0_POLICY_POST_INIT_RNG_PRESENT,
    LEARNED_PLANNING_DISABLED,
    LEARNED_PLANNING_ENABLED,
    UNIFORM_PLANNING_ENABLED,
    HiddenPrototypeFactorizedPartnerPlanningEvaluator,
    HiddenPrototypeFactorizedPartnerPlanningProtocol,
    HiddenPrototypeFactorizedPartnerPlanningState,
    validate_static_contract,
)
from alberta_framework.evaluation.hidden_prototype_two_agent_continual_life_development import (
    CONSUMED_DEVELOPMENT_ROOT,
    HIDDEN_INFERENCE_UNROUTED,
    HIDDEN_INFERRED_FULL,
    HiddenPrototypeTwoAgentEvaluator,
    HiddenPrototypeTwoAgentProtocol,
)
from alberta_framework.evaluation.prototype_two_learning_agent_recurrence_development import (
    PrototypeTwoLearningAgentRecurrenceProtocol,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _bounded_eager_jax(request: pytest.FixtureRequest) -> Iterator[None]:
    """Keep short cases eager except for the explicit compiled-step parity case."""

    if request.node.name == "test_one_event_compiled_step_matches_eager_composite_contract":
        yield
    else:
        with jax.disable_jit():
            yield


def _short_prototype_protocol() -> PrototypeTwoLearningAgentRecurrenceProtocol:
    return PrototypeTwoLearningAgentRecurrenceProtocol(
        segment_length=1,
        nuisance_dim=0,
        active_pair_slots=1,
        memory_capacity=1,
        replacement_interval=1,
        metric_window=1,
        arm_names=("joint_full",),
    )


def _short_u0_evaluator(*, singular_arm: bool = True) -> HiddenPrototypeTwoAgentEvaluator:
    protocol = HiddenPrototypeTwoAgentProtocol(
        prototype_protocol=_short_prototype_protocol(),
        arm_names=(HIDDEN_INFERRED_FULL,)
        if singular_arm
        else (
            HIDDEN_INFERRED_FULL,
            HIDDEN_INFERENCE_UNROUTED,
        ),
    )
    return HiddenPrototypeTwoAgentEvaluator(protocol)


def _evaluator(arm_name: Any) -> HiddenPrototypeFactorizedPartnerPlanningEvaluator:
    return HiddenPrototypeFactorizedPartnerPlanningEvaluator(
        arm_name,
        u0_evaluator=_short_u0_evaluator(),
    )


def _materialize_keys(tree: object) -> object:
    def convert(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(leaf)
        return leaf

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: object, right: object) -> None:
    def exact_bits(leaf: Any) -> Any:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            if dtype == jnp.float32:
                return jax.lax.bitcast_convert_type(leaf, jnp.uint32)
            if dtype == jnp.float16:
                return jax.lax.bitcast_convert_type(leaf, jnp.uint16)
        return leaf

    chex.assert_trees_all_equal(
        jax.tree.map(exact_bits, _materialize_keys(left)),
        jax.tree.map(exact_bits, _materialize_keys(right)),
    )


def _assert_tree_engine_equivalent(left: object, right: object) -> None:
    """Apply U0/U1's declared close-float, exact-discrete engine contract."""

    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if left_array.dtype.kind in "fc":
            np.testing.assert_allclose(
                left_array,
                right_array,
                rtol=EAGER_JIT_FLOAT_RTOL,
                atol=EAGER_JIT_FLOAT_ATOL,
            )
        else:
            np.testing.assert_array_equal(left_array, right_array)


def _planner_models(state: HiddenPrototypeFactorizedPartnerPlanningState) -> object:
    return (
        state.planner.agent_0.behavior,
        state.planner.agent_0.grounded,
        state.planner.agent_1.behavior,
        state.planner.agent_1.grounded,
    )


def _assert_auxiliary_matches_planner(
    evaluator: HiddenPrototypeFactorizedPartnerPlanningEvaluator,
    state: HiddenPrototypeFactorizedPartnerPlanningState,
) -> None:
    base = evaluator._cache_base_actions(state.planner)
    effective = evaluator._cache_effective_actions(state.planner)
    current = jnp.stack((state.u0.agent_0.current_action, state.u0.agent_1.current_action))
    changed = effective != base
    np.testing.assert_array_equal(current, effective)
    np.testing.assert_array_equal(state.u0.counterfactual_base_actions, base)
    np.testing.assert_array_equal(state.u0.prior_memory_action_changed, changed)
    np.testing.assert_array_equal(state.u0.prior_memory_retrieval_available, changed)


def test_frozen_config_arms_nonclaims_and_strict_guards() -> None:
    assert validate_static_contract() == ()
    protocol = HiddenPrototypeFactorizedPartnerPlanningProtocol()
    payload = protocol.to_config()
    assert protocol.segment_length == 512
    assert protocol.total_steps == 1_536
    assert payload["schedule"] == ["A1", "B", "A2"]
    assert payload["u0_arm"] == HIDDEN_INFERRED_FULL
    assert payload["arm_order"] == [
        LEARNED_PLANNING_ENABLED,
        UNIFORM_PLANNING_ENABLED,
        LEARNED_PLANNING_DISABLED,
    ]
    assert payload["consumed_development_root"] == {
        "namespace": CONSUMED_DEVELOPMENT_ROOT.namespace,
        "index": CONSUMED_DEVELOPMENT_ROOT.index,
        "environment_seed": CONSUMED_DEVELOPMENT_ROOT.environment_seed,
        "initialization_seed": CONSUMED_DEVELOPMENT_ROOT.initialization_seed,
    }
    transaction = cast(Mapping[str, object], payload["transaction"])
    assert transaction["environment_proposals_per_event"] == 4
    assert transaction["environment_proposals_not_eight"] is True
    assert transaction["event_index_bound_to_source_clock"] is True
    assert transaction["post_memory_candidates_constructed_locally"] is True
    assert transaction["complete_composite_all_or_none"] is True
    safety = cast(Mapping[str, object], payload["safety"])
    assert safety == {"caller_mask": "all_true", "physical_safety_certification": False}
    planner_initialization = cast(Mapping[str, object], payload["planner_initialization"])
    assert planner_initialization["same_key_for_every_arm"] is True
    assert planner_initialization["behavior_and_grounded_genesis_bit_identical"] is True
    assert planner_initialization["canonical_observation_dim"] == 8
    assert planner_initialization["canonical_prototype_representation_dim"] == 12
    execution_parity = cast(Mapping[str, object], payload["execution_parity"])
    assert execution_parity == {
        "float_rtol": EAGER_JIT_FLOAT_RTOL,
        "float_atol": EAGER_JIT_FLOAT_ATOL,
        "discrete_leaves_exact": EAGER_JIT_DISCRETE_LEAVES_EXACT,
        "floating_leaves_bit_identical_claimed": False,
    }
    nonclaims = cast(Mapping[str, object], payload["nonclaims"])
    assert nonclaims["underlying_post_memory_transition_binding_claimed"] is False
    assert nonclaims["underlying_base_fallback_source_binding_claimed"] is False
    assert nonclaims["same_event_memory_reward_effect_reported"] is False
    assert nonclaims["safety_certification"] is False
    assert ADDITIONAL_WRAPPER_PLANNER_POST_INIT_RANDOM_DRAWS_PER_EVENT == 0
    assert ADDITIONAL_WRAPPER_PLANNER_REPLAY_UPDATES_PER_EVENT == 0
    assert INHERITED_U0_POLICY_POST_INIT_RNG_PRESENT
    assert not ALL_TRUE_CALLER_MASK_IS_SAFETY_CERTIFICATION

    assert HiddenPrototypeFactorizedPartnerPlanningProtocol.from_config(payload) == protocol
    mutated = copy.deepcopy(payload)
    cast(dict[str, object], mutated["transaction"])["environment_proposals_per_event"] = 8
    with pytest.raises(ValueError, match="frozen declaration"):
        HiddenPrototypeFactorizedPartnerPlanningProtocol.from_config(mutated)
    with pytest.raises(ValueError, match="unsupported"):
        _evaluator("not_an_arm")
    with pytest.raises(ValueError, match="singular routed"):
        HiddenPrototypeFactorizedPartnerPlanningEvaluator(
            LEARNED_PLANNING_ENABLED,
            u0_evaluator=_short_u0_evaluator(singular_arm=False),
        )

    assert tuple(arm.name for arm in FACTORIZED_PLANNING_ARMS) == (
        LEARNED_PLANNING_ENABLED,
        UNIFORM_PLANNING_ENABLED,
        LEARNED_PLANNING_DISABLED,
    )
    assert tuple(
        (arm.planning_enabled, arm.uniform_partner_belief) for arm in FACTORIZED_PLANNING_ARMS
    ) == ((True, False), (True, True), (False, False))


def test_report_builder_rejects_static_drift_before_source_capture_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        u1_module,
        "validate_static_contract",
        lambda: ("sentinel declaration drift",),
    )

    def unexpected_source_capture(*, stage: str) -> Mapping[str, object]:
        pytest.fail(f"source capture reached during static-contract failure: {stage}")

    monkeypatch.setattr(u1_module, "_bound_source_manifest", unexpected_source_capture)
    monkeypatch.setattr(
        u1_module,
        "_run_arm",
        lambda *_args, **_kwargs: pytest.fail("panel execution reached during static failure"),
    )

    with pytest.raises(RuntimeError, match="U1 static contract.*sentinel declaration drift"):
        u1_module._build_report()


def test_report_builder_assembles_mocked_panel_without_executing_learners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_stages: list[str] = []
    arm_names: list[str] = []

    def source_manifest(*, stage: str) -> dict[str, str]:
        source_stages.append(stage)
        return {"manifest_sha256": "0" * 64}

    def run_arm(
        protocol: HiddenPrototypeFactorizedPartnerPlanningProtocol,
        arm: Any,
    ) -> dict[str, object]:
        assert protocol.total_steps == 1_536
        arm_names.append(arm.name)
        return {
            "arm": arm.name,
            "u0_preplanner_genesis_sha256": "1" * 64,
            "planner_model_genesis_sha256": "2" * 64,
            "work": {"events": protocol.total_steps},
            "resources": {
                "initial": {"composite_persistent_state_nbytes": 123},
            },
        }

    monkeypatch.setattr(u1_module, "validate_static_contract", lambda: ())
    monkeypatch.setattr(u1_module, "_bound_source_manifest", source_manifest)
    monkeypatch.setattr(u1_module, "_runtime_identity", lambda: {"runtime": "mocked"})
    monkeypatch.setattr(u1_module, "_run_arm", run_arm)

    report = u1_module._build_report()

    assert source_stages == ["pre-run", "post-run"]
    assert arm_names == [arm.name for arm in FACTORIZED_PLANNING_ARMS]
    assert report["runtime_identity"] == {"runtime": "mocked"}
    assert u1_module._report_hash_reconstructs(report)


def test_selected_source_runtime_and_report_hash_bind_without_full_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre = u1_module._bound_source_manifest(stage="test-pre")
    post = u1_module._bound_source_manifest(stage="test-post")
    assert pre == post
    expected_labels = {
        "evaluation_module_sha256",
        "u0_evaluation_module_sha256",
        "factorized_planner_core_sha256",
        "prototype_agent_core_sha256",
        "behavior_model_core_sha256",
        "grounded_joint_world_model_core_sha256",
        "context_inference_core_sha256",
        "horde_core_sha256",
        "prototype_feature_memory_core_sha256",
        "world_model_core_sha256",
        "hidden_context_evaluation_module_sha256",
        "prototype_feature_memory_recurrence_evaluation_module_sha256",
        "prototype_two_agent_recurrence_evaluation_module_sha256",
        "recurring_multiagent_stream_sha256",
    }
    assert set(pre) == expected_labels | {"manifest_sha256"}
    source_body = {name: value for name, value in pre.items() if name != "manifest_sha256"}
    assert pre["manifest_sha256"] == u1_module._json_sha256(source_body)
    assert u1_module._runtime_identity() == u1_module._runtime_identity()

    body: dict[str, object] = {"schema": "local-test", "nested": {"value": 3}}
    report = u1_module._attach_report_hash(body)
    assert u1_module._report_hash_reconstructs(report)
    tampered = copy.deepcopy(report)
    cast(dict[str, object], tampered["nested"])["value"] = 4
    assert not u1_module._report_hash_reconstructs(tampered)

    monkeypatch.setattr(
        u1_module,
        "_IMPORT_TIME_SELECTED_SOURCE_HASHES",
        (("evaluation_module_sha256", "0" * 64),),
    )
    with pytest.raises(RuntimeError, match="import-time bytes"):
        u1_module._bound_source_manifest(stage="tamper-test")


def test_local_attempt_latch_is_concurrent_once_and_seals_baseexception() -> None:
    callers = 8
    barrier = threading.Barrier(callers)
    attempts: list[int] = []
    builder_entered = threading.Event()
    release_builder = threading.Event()

    def build() -> str:
        attempts.append(1)
        builder_entered.set()
        if not release_builder.wait(timeout=2.0):
            raise RuntimeError("local concurrency test did not release the builder")
        return "compact-report"

    latch = u1_module._ProcessAttemptLatch(build)

    def get_concurrently() -> str:
        barrier.wait()
        return latch.get()

    with ThreadPoolExecutor(max_workers=callers) as pool:
        futures = [pool.submit(get_concurrently) for _ in range(callers)]
        assert builder_entered.wait(timeout=2.0)
        release_builder.set()
        values = [future.result(timeout=2.0) for future in futures]
    assert values == ["compact-report"] * callers
    assert attempts == [1]
    assert latch.get() == "compact-report"
    assert attempts == [1]

    class LocalFatal(BaseException):
        pass

    fatal = LocalFatal("stop")
    failed_attempts: list[int] = []

    def fail() -> str:
        failed_attempts.append(1)
        raise fatal

    failed = u1_module._ProcessAttemptLatch(fail)
    with pytest.raises(LocalFatal, match="stop"):
        failed.get()
    with pytest.raises(RuntimeError, match="sealed after failure") as sealed:
        failed.get()
    assert sealed.value.__cause__ is fatal
    assert failed_attempts == [1]


@pytest.mark.slow
def test_initial_models_resources_and_work_are_bit_identical_and_exact() -> None:
    evaluators = [_evaluator(arm.name) for arm in FACTORIZED_PLANNING_ARMS]
    raw_u0 = [evaluator._initialize_u0_source() for evaluator in evaluators]
    for source in raw_u0[1:]:
        _assert_tree_exact(source, raw_u0[0])

    initialization_keys = [evaluator._planner_initialization_key() for evaluator in evaluators]
    for key in initialization_keys[1:]:
        _assert_tree_exact(key, initialization_keys[0])
    states = [
        evaluator._compose_initial(source)
        for evaluator, source in zip(evaluators, raw_u0, strict=True)
    ]
    for state in states[1:]:
        _assert_tree_exact(_planner_models(state), _planner_models(states[0]))

    composite_sizes: list[int] = []
    work_rows: list[dict[str, object]] = []
    for evaluator, state, arm in zip(evaluators, states, FACTORIZED_PLANNING_ARMS, strict=True):
        assert bool(evaluator.state_is_valid(state))
        assert bool(jnp.all(evaluator.safety_action_masks))
        assert evaluator.arm == arm
        assert evaluator.planner.config.planning_enabled is arm.planning_enabled
        assert evaluator.planner.config.uniform_partner_belief is arm.uniform_partner_belief
        _assert_auxiliary_matches_planner(evaluator, state)

        resources = evaluator.resource_budget(state)
        planner = cast(Mapping[str, object], resources["planner_resource_budget"])
        assert resources["exact_decomposition"] is True
        assert resources["wrapper_extra_persistent_state_nbytes"] == 0
        assert planner["exact_tree_match"] is True
        assert planner["replay_capacity"] == 0
        assert planner["post_init_random_draws_per_event"] == 0
        composite_sizes.append(cast(int, resources["composite_persistent_state_nbytes"]))

        work = evaluator.work_budget(3)
        assert work["environment_proposal_calls"] == 12
        assert work["environment_proposals_per_event"] == 4
        assert work["environment_proposals_not_eight"] is True
        assert work["u0_initialization_calls"] == 1
        assert work["u0_initialization_random_draw_internals_counted_in_u1_scope"] is False
        assert work["planner_initialization_calls"] == 1
        assert work["planner_initialization_key_split_calls"] == 1
        assert work["planner_initial_grounded_uniform_draw_calls"] == 2
        assert work["planner_initial_behavior_keys_stored"] == 2
        assert work["wrapper_event_index_source_clock_bindings"] == 3
        assert work["additional_wrapper_planner_environment_proposals"] == 0
        assert work["additional_wrapper_planner_post_init_random_draws"] == 0
        assert work["additional_wrapper_planner_replay_updates"] == 0
        assert work["threshold_evaluations"] == 0
        assert work["winner_selection_calls"] == 0
        assert work["writer_calls"] == 0
        assert work["checkpoint_save_calls"] == 0
        assert work["checkpoint_load_calls"] == 0
        assert cast(Mapping[str, object], work["u0"])["environment_proposal_calls"] == 12
        work_rows.append(work)
    assert len(set(composite_sizes)) == 1
    assert work_rows[1:] == [work_rows[0], work_rows[0]]


@pytest.mark.slow
@pytest.mark.parametrize(
    "arm_name",
    [
        LEARNED_PLANNING_ENABLED,
        UNIFORM_PLANNING_ENABLED,
        LEARNED_PLANNING_DISABLED,
    ],
)
def test_short_aba_life_has_exact_cube_bindings_and_reachable_arms(arm_name: Any) -> None:
    evaluator = _evaluator(arm_name)
    state = evaluator.initialize()

    for event_index in range(3):
        source = state
        result = evaluator.step(source, jnp.asarray(event_index, dtype=jnp.int32))
        trace = result.trace
        trace_field_names = set(trace.__class__.__dataclass_fields__)
        assert all("same_event" not in name for name in trace_field_names)
        assert bool(trace.event_index_matches_source_clock)
        event_payload = u1_module._selected_event_payload(
            trace,
            event_index=event_index,
            segment_length=1,
        )
        assert event_payload["event_index_matches_source_clock"] is True
        assert cast(Mapping[str, bool], event_payload["bindings"])[
            "event_index_source_clock"
        ] is True
        actual = np.asarray(trace.actual_actions)
        base = np.asarray(trace.base_actions)
        np.testing.assert_array_equal(
            actual,
            evaluator._cache_effective_actions(source.planner),
        )
        np.testing.assert_array_equal(
            base,
            evaluator._cache_base_actions(source.planner),
        )
        expected_cube = np.asarray(
            (
                actual,
                (base[0], actual[1]),
                (actual[0], base[1]),
                base,
            ),
            dtype=np.int32,
        )
        np.testing.assert_array_equal(trace.joint_primitive_actions, expected_cube)
        np.testing.assert_array_equal(trace.joint_primitive_actions[0], actual)
        assert trace.joint_rewards.shape == (4, 2)
        post_memory_observations = np.stack(
            (
                np.asarray(result.state.u0.agent_0.current_raw_observation),
                np.asarray(result.state.u0.agent_1.current_raw_observation),
            )
        ).astype(np.float32)
        row_zero_rewards = np.asarray(trace.joint_rewards[0], dtype=np.float32)
        expected_partner_actions = np.asarray((actual[1], actual[0]), dtype=np.int32)
        expected_grounded_targets = np.concatenate(
            (
                post_memory_observations,
                row_zero_rewards[:, None],
                np.ones((2, 1), dtype=np.float32),
            ),
            axis=1,
        )
        expected_joint_action_indices = np.asarray(
            (
                2 * actual[0] + actual[1],
                2 * actual[1] + actual[0],
            ),
            dtype=np.int32,
        )
        np.testing.assert_array_equal(trace.planner_executed_actions, actual)
        np.testing.assert_array_equal(
            trace.planner_observed_partner_actions,
            expected_partner_actions,
        )
        np.testing.assert_array_equal(
            trace.planner_post_memory_observations,
            post_memory_observations,
        )
        np.testing.assert_array_equal(trace.planner_input_rewards, row_zero_rewards)
        assert float(trace.planner_input_continuation) == 1.0
        np.testing.assert_array_equal(
            trace.planner_grounded_targets,
            expected_grounded_targets,
        )
        np.testing.assert_array_equal(
            trace.planner_grounded_joint_action_indices,
            expected_joint_action_indices,
        )
        assert bool(trace.source_u0_valid)
        assert bool(jnp.all(trace.source_planner_cache_valid))
        assert bool(trace.source_aux_cache_binding_valid)
        assert bool(trace.source_composite_valid)
        assert bool(trace.u0_candidate_committed)
        assert bool(trace.planner_candidate_committed)
        assert bool(trace.local_post_memory_binding_valid)
        assert bool(trace.planner_cube_binding_valid)
        assert bool(trace.candidate_u0_valid)
        assert bool(jnp.all(trace.candidate_planner_cache_valid))
        assert bool(trace.candidate_aux_cache_binding_valid)
        assert bool(trace.candidate_composite_valid)
        assert not bool(trace.no_oracle_channel_consumed)
        assert bool(trace.all_true_caller_mask_used)
        assert not bool(trace.all_true_caller_mask_is_safety_certification)
        assert not bool(trace.forced_outer_rejection)
        assert bool(trace.outer_transaction_committed)
        assert bool(jnp.all(trace.behavior_update_applied))
        assert bool(jnp.all(trace.grounded_update_applied))
        assert trace.memory_query_before_write.shape == (2,)
        assert trace.memory_wrote.shape == (2,)
        assert trace.memory_evicted.shape == (2,)
        assert trace.memory_retrieval_available.shape == (2,)
        assert trace.memory_action_changed.shape == (2,)

        changed = np.asarray(trace.planner_action_changed)
        effects = np.asarray(trace.planner_reward_effects)
        expected_outcomes = np.where(
            changed & (effects > 0.0),
            1,
            np.where(changed & (effects < 0.0), -1, 0),
        )
        np.testing.assert_array_equal(trace.planner_outcome_codes, expected_outcomes)
        if arm_name == LEARNED_PLANNING_ENABLED:
            np.testing.assert_array_equal(
                trace.next_applied_partner_probabilities,
                trace.next_learned_partner_probabilities,
            )
        elif arm_name == UNIFORM_PLANNING_ENABLED:
            np.testing.assert_array_equal(
                trace.next_applied_partner_probabilities,
                np.full((2, 2), 0.5, dtype=np.float32),
            )
        else:
            np.testing.assert_array_equal(trace.actual_actions, trace.base_actions)
            np.testing.assert_array_equal(
                trace.next_effective_actions,
                trace.next_base_actions,
            )
            assert not bool(jnp.any(trace.planner_action_changed))
            assert not bool(jnp.any(trace.next_planner_action_changed))
        state = result.state
        _assert_auxiliary_matches_planner(evaluator, state)
        assert bool(evaluator.state_is_valid(state))


@pytest.mark.slow
def test_one_event_compiled_step_matches_eager_composite_contract() -> None:
    assert not bool(jax.config.jax_disable_jit)
    evaluator = _evaluator(LEARNED_PLANNING_ENABLED)
    with jax.disable_jit():
        source = evaluator.initialize()
        eager = evaluator.step(source, jnp.asarray(0, dtype=jnp.int32))

    compiled = evaluator.compiled_step(source, jnp.asarray(0, dtype=jnp.int32))

    assert bool(eager.trace.outer_transaction_committed)
    assert bool(compiled.trace.outer_transaction_committed)
    _assert_tree_engine_equivalent(compiled, eager)


@pytest.mark.slow
def test_event_index_mismatch_rolls_back_negative_duplicate_and_out_of_order() -> None:
    evaluator = _evaluator(LEARNED_PLANNING_ENABLED)
    source = evaluator.initialize()

    for invalid_index in (-1, 1):
        rejected = evaluator.step(source, jnp.asarray(invalid_index, dtype=jnp.int32))
        assert not bool(rejected.trace.event_index_matches_source_clock)
        assert not bool(rejected.trace.outer_transaction_committed)
        _assert_tree_exact(rejected.state, source)
        event_payload = u1_module._selected_event_payload(
            rejected.trace,
            event_index=invalid_index,
            segment_length=1,
        )
        assert event_payload["event_index_matches_source_clock"] is False
        assert cast(Mapping[str, bool], event_payload["bindings"])[
            "event_index_source_clock"
        ] is False

    with pytest.raises(TypeError, match="exact scalar int32"):
        evaluator.step(source, jnp.asarray(0, dtype=jnp.uint32))

    accepted = evaluator.step(source, jnp.asarray(0, dtype=jnp.int32))
    assert bool(accepted.trace.event_index_matches_source_clock)
    assert bool(accepted.trace.outer_transaction_committed)
    clock_one = accepted.state

    for invalid_index in (0, 2):
        rejected = evaluator.step(clock_one, jnp.asarray(invalid_index, dtype=jnp.int32))
        assert not bool(rejected.trace.event_index_matches_source_clock)
        assert not bool(rejected.trace.outer_transaction_committed)
        _assert_tree_exact(rejected.state, clock_one)


@pytest.mark.slow
def test_outer_transaction_rolls_back_forced_and_auxiliary_tampering() -> None:
    evaluator = _evaluator(LEARNED_PLANNING_ENABLED)
    source = evaluator.initialize()
    forced = evaluator.step(
        source,
        jnp.asarray(0, dtype=jnp.int32),
        force_outer_rejection=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(forced.trace.u0_candidate_committed)
    assert bool(forced.trace.planner_candidate_committed)
    assert bool(forced.trace.candidate_composite_valid)
    assert not bool(forced.trace.outer_transaction_committed)
    _assert_tree_exact(forced.state, source)

    current = jnp.stack((source.u0.agent_0.current_action, source.u0.agent_1.current_action))
    wrong_base = 1 - source.u0.counterfactual_base_actions
    wrong_changed = current != wrong_base
    tampered_u0 = source.u0.replace(
        counterfactual_base_actions=wrong_base,
        prior_memory_retrieval_available=jnp.ones((2,), dtype=jnp.bool_),
        prior_memory_action_changed=wrong_changed,
    )
    tampered = source.replace(u0=tampered_u0)
    assert bool(evaluator.u0._state_valid(tampered.u0))
    assert not bool(evaluator.state_is_valid(tampered))
    rejected = evaluator.step(tampered, jnp.asarray(0, dtype=jnp.int32))
    assert bool(rejected.trace.source_u0_valid)
    assert bool(jnp.all(rejected.trace.source_planner_cache_valid))
    assert not bool(rejected.trace.source_aux_cache_binding_valid)
    assert not bool(rejected.trace.source_composite_valid)
    assert not bool(rejected.trace.planner_cube_binding_valid)
    assert not bool(rejected.trace.outer_transaction_committed)
    _assert_tree_exact(rejected.state, tampered)

    bad_token = source.planner.config_token.at[0].add(jnp.asarray(1, dtype=jnp.uint8))
    guarded = source.replace(planner=source.planner.replace(config_token=bad_token))
    assert not bool(
        jnp.all(
            evaluator.planner.authenticate_pair(
                guarded.planner,
                guarded.u0.agent_0,
                guarded.u0.agent_1,
            )
        )
    )
    assert not bool(evaluator.state_is_valid(guarded))
