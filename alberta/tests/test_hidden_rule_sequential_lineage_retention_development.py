# mypy: disable-error-code="attr-defined,call-arg"
"""Cheap contracts for the development-only sequential-lineage composition.

No test in this module invokes the 4,000-step panel.  Dynamic coverage is
limited to the first two eager events of the already-consumed root-zero life.
"""

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
import pytest

from alberta_framework.core.sequential_lineage_cache import (
    SequentialLineageCache,
    SequentialLineageCacheConfig,
)
from alberta_framework.evaluation import (
    hidden_rule_capacity_pressure_development as capacity_pressure,
)
from alberta_framework.evaluation import (
    hidden_rule_sequential_lineage_retention_development as retention,
)
from alberta_framework.evaluation.hidden_rule_sequential_lineage_retention_development import (
    CONDITIONS,
    H2_PREDICTIVE_RESCUE,
    NO_SIGNAL,
    PROTOCOL,
    SEQUENTIAL_LINEAGE_CONFIG,
    HiddenRuleSequentialLineageRetentionEvaluator,
    HiddenRuleSequentialLineageRetentionProtocol,
)

pytestmark = [pytest.mark.development, pytest.mark.integration]


@pytest.fixture
def eager_jax() -> Iterator[None]:
    """Bound dynamic tests to a handful of noncompiled initial events."""

    with jax.disable_jit():
        yield


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
        if dtype == jnp.dtype(jnp.float32):
            return jax.lax.bitcast_convert_type(leaf, jnp.uint32)
        if dtype == jnp.dtype(jnp.float16):
            return jax.lax.bitcast_convert_type(leaf, jnp.uint16)
        return leaf

    chex.assert_trees_all_equal(
        jax.tree.map(exact_bits, _materialize_keys(left)),
        jax.tree.map(exact_bits, _materialize_keys(right)),
    )


def test_frozen_protocol_geometry_nonclaims_and_strict_config() -> None:
    assert retention.validate_static_contract() == ()
    assert CONDITIONS == (NO_SIGNAL, H2_PREDICTIVE_RESCUE)
    assert retention.DEVELOPMENT_ONLY
    assert retention.CALIBRATION_ROOT_CONSUMED
    assert retention.CONSUMED_ROOT_INDEX == 0
    assert not retention.SCIENTIFIC_PROMOTION_ALLOWED
    assert not retention.EVIDENCE_AUTHORIZED
    assert not retention.OUTPUT_WRITES_ALLOWED
    assert not retention.WRITER_AVAILABLE
    assert retention.ARTIFACT_BYTES_WRITTEN == 0
    assert not retention.THRESHOLDS_USED
    assert not retention.WINNER_SELECTION_ALLOWED
    assert not retention.DEFAULT_CONDITION_AVAILABLE
    assert not retention.ARBITRARY_ROOT_EXECUTION_ALLOWED

    payload = PROTOCOL.to_config()
    assert payload["conditions"] == [NO_SIGNAL, H2_PREDICTIVE_RESCUE]
    assert payload["epsilon_grid"] == list(capacity_pressure.EPSILON_GRID)
    geometry = cast(Mapping[str, object], payload["geometry"])
    assert geometry["max_contexts"] == 3
    assert geometry["n_actions"] == 4
    assert geometry["observation_dim"] == 4
    assert geometry["initial_reward_estimate"] == (
        capacity_pressure.CONTEXT_CONFIG.initial_reward_estimate
    )
    assert geometry["confirmation_horizon"] == 2
    matching = cast(Mapping[str, object], payload["matching"])
    assert matching["evaluator_owns_matched_outer_work_claim"] is True
    assert matching["core_matched_outer_work_claimed"] is False
    nonclaims = cast(Mapping[str, object], payload["nonclaims"])
    assert nonclaims["parameter_transplant"] is False
    assert nonclaims["host_transition_binding_claimed_by_core"] is False
    assert nonclaims["external_sidecar_state_provenance_claimed"] is False

    assert HiddenRuleSequentialLineageRetentionProtocol.from_config(payload) == PROTOCOL
    tampered = copy.deepcopy(payload)
    cast(dict[str, object], tampered["geometry"])["observation_dim"] = 5
    with pytest.raises(ValueError, match="frozen declaration"):
        HiddenRuleSequentialLineageRetentionProtocol.from_config(tampered)
    with pytest.raises(ValueError, match="unknown"):
        HiddenRuleSequentialLineageRetentionEvaluator(0.2, "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumed-root grid"):
        HiddenRuleSequentialLineageRetentionEvaluator(0.3, NO_SIGNAL)


def test_report_builder_rejects_static_drift_before_source_capture_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retention,
        "validate_static_contract",
        lambda: ("sentinel declaration drift",),
    )

    def unexpected_source_capture(*, stage: str) -> dict[str, str]:
        pytest.fail(f"source capture reached during static-contract failure: {stage}")

    monkeypatch.setattr(retention, "_bound_source_manifest", unexpected_source_capture)
    monkeypatch.setattr(
        retention,
        "_run_condition",
        lambda *_args, **_kwargs: pytest.fail("panel execution reached during static failure"),
    )

    with pytest.raises(
        RuntimeError,
        match="sequential-lineage static contract.*sentinel declaration drift",
    ):
        retention._build_report()


def test_report_builder_assembles_mocked_grid_without_executing_learners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_stages: list[str] = []
    run_keys: list[tuple[float, str]] = []

    def source_manifest(*, stage: str) -> dict[str, str]:
        source_stages.append(stage)
        return {"manifest_sha256": "0" * 64}

    def run_condition(epsilon: float, condition: str) -> dict[str, object]:
        run_keys.append((epsilon, condition))
        epsilon_index = capacity_pressure.EPSILON_GRID.index(epsilon)
        return {
            "epsilon": epsilon,
            "condition": condition,
            "initial_state_sha256": f"state-{epsilon_index}",
            "initial_base_sha256": f"base-{epsilon_index}",
            "initial_sidecar_pair_sha256": f"sidecar-{epsilon_index}",
            "resource_budget": {"bytes": 2_088},
            "work_budget": {"steps": 4_000},
            "controller_rng_trace_sha256": f"rng-{epsilon_index}",
        }

    monkeypatch.setattr(retention, "validate_static_contract", lambda: ())
    monkeypatch.setattr(retention, "_bound_source_manifest", source_manifest)
    monkeypatch.setattr(retention, "_runtime_identity", lambda: {"runtime": "mocked"})
    monkeypatch.setattr(retention, "_run_condition", run_condition)

    report = retention._build_report()

    assert source_stages == ["pre-run", "post-run"]
    assert run_keys == [
        (epsilon, condition)
        for epsilon in capacity_pressure.EPSILON_GRID
        for condition in CONDITIONS
    ]
    assert report["runtime_identity"] == {"runtime": "mocked"}
    assert retention._report_hash_reconstructs(report)


def test_selected_sources_runtime_report_hash_and_tamper_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre = retention._bound_source_manifest(stage="test-pre")
    post = retention._bound_source_manifest(stage="test-post")
    assert pre == post
    expected_labels = {
        "evaluation_module_sha256",
        "sequential_lineage_cache_core_sha256",
        "capacity_pressure_evaluator_sha256",
        "context_inference_core_sha256",
        "average_reward_core_sha256",
        "matrix_game_stream_sha256",
        "pairwise_dominance_core_sha256",
    }
    assert set(pre) == expected_labels | {"manifest_sha256"}
    source_body = {name: value for name, value in pre.items() if name != "manifest_sha256"}
    assert pre["manifest_sha256"] == retention._json_sha256(source_body)
    assert (
        pre["sequential_lineage_cache_core_sha256"]
        == retention.EXPECTED_SEQUENTIAL_LINEAGE_CORE_SHA256
    )
    assert retention._runtime_identity() == retention._runtime_identity()

    body: dict[str, object] = {"schema": "local-test", "nested": {"value": 3}}
    report = retention._attach_report_hash(body)
    assert retention._report_hash_reconstructs(report)
    modified = copy.deepcopy(report)
    cast(dict[str, object], modified["nested"])["value"] = 4
    assert not retention._report_hash_reconstructs(modified)

    monkeypatch.setattr(
        retention,
        "_IMPORT_TIME_SELECTED_SOURCE_HASHES",
        (("evaluation_module_sha256", "0" * 64),),
    )
    with pytest.raises(RuntimeError, match="import-time bytes"):
        retention._bound_source_manifest(stage="tamper-test")


def test_process_latch_is_concurrent_once_and_seals_baseexception() -> None:
    callers = 6
    barrier = threading.Barrier(callers)
    entered = threading.Event()
    release = threading.Event()
    attempts: list[int] = []

    def build() -> str:
        attempts.append(1)
        entered.set()
        if not release.wait(timeout=2.0):
            raise RuntimeError("local latch test did not release its builder")
        return "compact-report"

    latch = retention._ProcessAttemptLatch(build)

    def concurrent_get() -> str:
        barrier.wait()
        return latch.get()

    with ThreadPoolExecutor(max_workers=callers) as pool:
        futures = [pool.submit(concurrent_get) for _ in range(callers)]
        assert entered.wait(timeout=2.0)
        release.set()
        assert [future.result(timeout=2.0) for future in futures] == ["compact-report"] * callers
    assert attempts == [1]
    assert latch.get() == "compact-report"

    class LocalFatal(BaseException):
        pass

    fatal = LocalFatal("stop")
    failed_attempts: list[int] = []

    def fail() -> str:
        failed_attempts.append(1)
        raise fatal

    failed = retention._ProcessAttemptLatch(fail)
    with pytest.raises(LocalFatal, match="stop"):
        failed.get()
    with pytest.raises(RuntimeError, match="sealed after failure") as sealed:
        failed.get()
    assert sealed.value.__cause__ is fatal
    assert failed_attempts == [1]


@pytest.mark.slow
def test_condition_genesis_work_and_resources_are_exact_and_equal(eager_jax: None) -> None:
    evaluators = [
        HiddenRuleSequentialLineageRetentionEvaluator(0.2, condition) for condition in CONDITIONS
    ]
    states = [evaluator.initialize() for evaluator in evaluators]
    _assert_tree_exact(states[0], states[1])
    assert states[0].sidecar_0 is not states[0].sidecar_1
    _assert_tree_exact(states[0].sidecar_0, states[0].sidecar_1)
    assert all(
        bool(evaluator.state_is_valid(state)) for evaluator, state in zip(evaluators, states)
    )

    resources = [evaluator.resource_budget(state) for evaluator, state in zip(evaluators, states)]
    assert resources[0].to_dict() == resources[1].to_dict()
    for resource in resources:
        assert resource.measured_base_scan_carry_nbytes == 962
        assert resource.measured_sidecar_0_nbytes == 563
        assert resource.measured_sidecar_1_nbytes == 563
        assert resource.measured_sidecar_pair_nbytes == 1_126
        assert resource.measured_composite_scan_carry_nbytes == 2_088
        assert resource.total_persistent_nbytes == 2_088
        assert resource.sidecar.per_agent_state_nbytes == 563
        assert resource.sidecar.joint_state_nbytes == 1_126
        assert resource.sidecar.base_scan_carry_nbytes == 962
        assert resource.sidecar.total_scan_carry_nbytes == 2_088
        assert resource.sidecar.parameter_transplant_allowed is False
        assert resource.exact_base_match
        assert resource.exact_sidecar_formula_match
        assert resource.exact_composite_match

    work = [evaluator.work_budget() for evaluator in evaluators]
    assert work[0].to_dict() == work[1].to_dict()
    for record in work:
        assert record.total_steps == 4_000
        assert record.sequential_lineage_proposals == 8_000
        assert record.prioritized_context_update_proposals == 8_000
        assert record.pre_outcome_reward_bank_snapshots == 8_000
        assert record.pre_outcome_rescue_score_snapshots == 8_000
        assert record.controller_scrub_preparations == 8_000
        assert record.controller_update_proposals == 8_000
        assert record.outer_all_or_none_commit_decisions == 4_000
        assert record.evaluator_matched_outer_work_claimed
        assert not record.core_matched_outer_work_claimed
        assert not record.core.matched_outer_work_claimed
        assert record.core.prediction_bank_calls == 8_000
        assert record.core.scalar_predictions == 40_000
        assert record.core.random_draws == 0
        assert record.replay_updates == record.reset_callbacks == 0


@pytest.mark.slow
def test_first_two_eager_events_bind_exact_preweights_and_match_conditions(
    eager_jax: None,
) -> None:
    evaluators = {
        condition: HiddenRuleSequentialLineageRetentionEvaluator(0.2, condition)
        for condition in CONDITIONS
    }
    states = {condition: evaluator.initialize() for condition, evaluator in evaluators.items()}

    for _ in range(2):
        results = {}
        for condition in CONDITIONS:
            evaluator = evaluators[condition]
            source = states[condition]
            source_weights = (
                source.base.context_0.reward_weights,
                source.base.context_1.reward_weights,
            )
            result = evaluator.step(source)
            results[condition] = result
            assert bool(result.trace.outer_update_applied)
            assert bool(result.trace.outer_candidate_valid)
            assert bool(result.trace.all_or_none_commit_valid)
            assert bool(result.trace.committed_candidate_exact)
            assert not bool(result.trace.rollback_exact)
            assert bool(result.trace.source_scores_snapshotted_before_outcome)
            assert not bool(result.trace.outcome_routed_to_current_protection)
            assert bool(jnp.all(result.trace.source_rescue_scores_valid))
            assert bool(jnp.all(result.trace.dispatch_binding_valid))
            assert bool(jnp.all(result.trace.context_protection_input_bound))
            assert bool(jnp.all(result.trace.pre_update_weight_binding_valid))
            assert bool(jnp.all(result.trace.event_binding_valid))
            assert bool(jnp.all(result.trace.proposal_source_state_valid))
            assert bool(jnp.all(result.trace.proposal_event_valid))
            assert bool(jnp.all(result.trace.proposal_fields_valid))
            assert bool(jnp.all(result.trace.proposal_update_applied))
            assert bool(jnp.all(result.trace.source_sidecar_valid))
            assert bool(jnp.all(result.trace.candidate_sidecar_valid))
            assert bool(jnp.all(result.trace.committed_sidecar_valid))
            assert bool(jnp.all(result.trace.scrub_preparation_valid))
            assert bool(jnp.all(result.trace.event_clocks_bound))
            assert bool(result.trace.source_clocks_aligned)
            assert bool(result.trace.candidate_clocks_aligned)
            assert bool(result.trace.committed_clocks_aligned)
            assert bool(result.trace.parameter_transplant_absent)
            assert not bool(jnp.any(result.trace.proposal_parameter_transplanted))

            actions = result.trace.capacity.actions
            for agent_index, event in enumerate(result.events):
                _assert_tree_exact(event.source_reward_weights, source_weights[agent_index])
                _assert_tree_exact(
                    event.source_reward_weights,
                    result.trace.pre_update_reward_weights[agent_index],
                )
                own_action = actions[agent_index]
                partner_action = actions[1 - agent_index]
                _assert_tree_exact(event.action, own_action)
                _assert_tree_exact(
                    event.observation,
                    jax.nn.one_hot(partner_action, 4, dtype=jnp.float32),
                )
                _assert_tree_exact(event.reward, result.trace.capacity.reward)
                _assert_tree_exact(
                    event.source_step_words,
                    source.base.context_0.step_words
                    if agent_index == 0
                    else source.base.context_1.step_words,
                )
                _assert_tree_exact(
                    event.post_step_words,
                    result.context_results[agent_index].post_step_words,
                )
                _assert_tree_exact(
                    event.source_birth_words,
                    source.base.ledger_0.slot_birth_words
                    if agent_index == 0
                    else source.base.ledger_1.slot_birth_words,
                )
                assert bool(result.proposals[agent_index].update_applied)
                assert not bool(result.proposals[agent_index].parameter_transplanted)
            assert bool(evaluator.state_is_valid(result.state))
            states[condition] = result.state

        # Rescue counts are still zero in the eager genesis prefix, so the two
        # explicit dispatches and every resulting state bit are identical.
        _assert_tree_exact(
            results[NO_SIGNAL].trace.dispatched_eviction_protection,
            jnp.zeros((2, 3), dtype=jnp.float32),
        )
        _assert_tree_exact(
            results[H2_PREDICTIVE_RESCUE].trace.dispatched_eviction_protection,
            jnp.zeros((2, 3), dtype=jnp.float32),
        )
        _assert_tree_exact(results[NO_SIGNAL].state, results[H2_PREDICTIVE_RESCUE].state)
        _assert_tree_exact(
            results[NO_SIGNAL].trace.capacity.controller_rng_key_words,
            results[H2_PREDICTIVE_RESCUE].trace.capacity.controller_rng_key_words,
        )


@pytest.mark.slow
def test_forced_outer_rejection_rolls_every_child_back_exactly(eager_jax: None) -> None:
    evaluator = HiddenRuleSequentialLineageRetentionEvaluator(0.2, H2_PREDICTIVE_RESCUE)
    state = evaluator.initialize()
    result = evaluator.step(
        state,
        force_outer_rejection=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(result.trace.forced_outer_rejection)
    assert bool(result.trace.outer_candidate_valid)
    assert not bool(result.trace.outer_update_applied)
    assert bool(result.trace.rollback_exact)
    assert not bool(result.trace.committed_candidate_exact)
    assert bool(result.trace.all_or_none_commit_valid)
    assert bool(jnp.all(result.trace.proposal_update_applied))
    assert bool(jnp.all(result.trace.scrub_preparation_valid))
    assert bool(jnp.all(result.trace.controller_updates_proposed))
    assert not bool(result.trace.capacity.update_applied)
    _assert_tree_exact(result.trace.capacity.pre_step_words, result.trace.capacity.post_step_words)
    _assert_tree_exact(result.state, state)


@pytest.mark.slow
@pytest.mark.parametrize("tamper", ["payload", "config", "content"])
def test_sidecar_config_and_content_tamper_fail_outer_transaction(
    eager_jax: None,
    tamper: str,
) -> None:
    evaluator = HiddenRuleSequentialLineageRetentionEvaluator(0.2, NO_SIGNAL)
    state = evaluator.initialize()
    if tamper == "payload":
        sidecar_0 = state.sidecar_0.replace(
            bound_birth_words=state.sidecar_0.bound_birth_words.at[1, 1].set(
                jnp.asarray(1, dtype=jnp.uint32)
            )
        )
    elif tamper == "config":
        different = SequentialLineageCache(
            SequentialLineageCacheConfig(
                max_contexts=3,
                n_actions=4,
                observation_dim=4,
                initial_reward_estimate=0.25,
            )
        )
        sidecar_0 = different.init()
    else:
        sidecar_0 = state.sidecar_0.replace(
            content_token=state.sidecar_0.content_token.at[0].set(
                jnp.bitwise_xor(
                    state.sidecar_0.content_token[0],
                    jnp.asarray(1, dtype=jnp.uint8),
                )
            )
        )
    corrupted = state.replace(sidecar_0=sidecar_0)
    result = evaluator.step(corrupted)
    assert not bool(result.trace.source_sidecar_valid[0])
    assert bool(result.trace.source_sidecar_valid[1])
    assert not bool(result.trace.proposal_update_applied[0])
    assert bool(result.trace.proposal_update_applied[1])
    assert not bool(result.trace.outer_candidate_valid)
    assert not bool(result.trace.outer_update_applied)
    assert bool(result.trace.rollback_exact)
    assert bool(result.trace.all_or_none_commit_valid)
    _assert_tree_exact(result.state, corrupted)


def test_sidecar_configuration_is_exact_context_configuration() -> None:
    assert SEQUENTIAL_LINEAGE_CONFIG == SequentialLineageCacheConfig(
        max_contexts=3,
        n_actions=4,
        observation_dim=4,
        initial_reward_estimate=capacity_pressure.CONTEXT_CONFIG.initial_reward_estimate,
    )
