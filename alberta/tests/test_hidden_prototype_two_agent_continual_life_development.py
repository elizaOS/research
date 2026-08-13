# mypy: disable-error-code="index,no-any-return"
"""Contracts for the hidden two-Prototype-agent development life."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_context_coadaptation_development import (
    _CONTEXT_CONFIG,
    DEVELOPMENT_SEEDS,
)
from alberta_framework.evaluation.hidden_prototype_two_agent_continual_life_development import (
    HIDDEN_INFERENCE_UNROUTED,
    HIDDEN_INFERRED_FULL,
    HIDDEN_PROTOTYPE_TWO_AGENT_ARMS,
    INHERITED_CONTEXT_INFERENCE_CONFIG,
    HiddenPrototypeTwoAgentEvaluator,
    HiddenPrototypeTwoAgentProtocol,
    _hidden_learner_observation,
    _joint_reward_effects,
    run_hidden_prototype_two_agent_continual_life_development,
    validate_static_contract,
)
from alberta_framework.evaluation.prototype_two_learning_agent_recurrence_development import (
    PrototypeTwoLearningAgentRecurrenceProtocol,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _short_protocol() -> HiddenPrototypeTwoAgentProtocol:
    return HiddenPrototypeTwoAgentProtocol(
        prototype_protocol=PrototypeTwoLearningAgentRecurrenceProtocol(
            segment_length=1,
            nuisance_dim=0,
            active_pair_slots=1,
            memory_capacity=1,
            replacement_interval=1,
            metric_window=1,
            arm_names=("joint_full",),
        )
    )


def _tree_equal(left: object, right: object) -> bool:
    leaves_left, structure_left = jax.tree.flatten(left)
    leaves_right, structure_right = jax.tree.flatten(right)
    if structure_left != structure_right or len(leaves_left) != len(leaves_right):
        return False
    for a, b in zip(leaves_left, leaves_right, strict=True):
        dtype_a = getattr(a, "dtype", None)
        dtype_b = getattr(b, "dtype", None)
        if dtype_a is not None and jax.dtypes.issubdtype(dtype_a, jax.dtypes.prng_key):
            a = jax.random.key_data(a)
        if dtype_b is not None and jax.dtypes.issubdtype(dtype_b, jax.dtypes.prng_key):
            b = jax.random.key_data(b)
        array_a = np.asarray(jax.device_get(a))
        array_b = np.asarray(jax.device_get(b))
        if array_a.dtype.kind in "fc":
            if not np.allclose(array_a, array_b, rtol=1.0e-6, atol=1.0e-7):
                return False
        elif not np.array_equal(array_a, array_b):
            return False
    return True


def test_static_contract_reuses_consumed_root_and_context_config_without_claims() -> None:
    assert validate_static_contract() == ()
    assert INHERITED_CONTEXT_INFERENCE_CONFIG is _CONTEXT_CONFIG
    assert tuple(arm.name for arm in HIDDEN_PROTOTYPE_TWO_AGENT_ARMS) == (
        HIDDEN_INFERRED_FULL,
        HIDDEN_INFERENCE_UNROUTED,
    )
    protocol = HiddenPrototypeTwoAgentProtocol()
    payload = protocol.to_config()
    assert payload["consumed_development_root"] == {
        "namespace": DEVELOPMENT_SEEDS[0].namespace,
        "index": DEVELOPMENT_SEEDS[0].index,
        "environment_seed": DEVELOPMENT_SEEDS[0].environment_seed,
        "initialization_seed": DEVELOPMENT_SEEDS[0].initialization_seed,
    }
    assert payload["context_inference"] == _CONTEXT_CONFIG.to_config()
    assert payload["schedule"] == ["A1", "B", "A2"]
    assert payload["development_only"] is True
    assert payload["scientific_promotion_allowed"] is False
    assert payload["output_writes_allowed"] is False
    assert payload["thresholds_or_winner_selection"] is False
    assert payload["context_capacity_pressure_claimed"] is False


def test_oracle_cues_are_destroyed_before_inferred_context_is_routed() -> None:
    first = jnp.asarray((0.2, -0.4, 0.1, -0.1, 1.0, 0.0), dtype=jnp.float32)
    second = first.at[4:6].set(jnp.asarray((0.0, 1.0), dtype=jnp.float32))
    inferred = jnp.asarray((0.0, 1.0), dtype=jnp.float32)

    routed_first = _hidden_learner_observation(first, inferred, jnp.asarray(True))
    routed_second = _hidden_learner_observation(second, inferred, jnp.asarray(True))
    unrouted = _hidden_learner_observation(first, inferred, jnp.asarray(False))

    np.testing.assert_array_equal(routed_first, routed_second)
    np.testing.assert_array_equal(routed_first[4:6], inferred)
    np.testing.assert_array_equal(unrouted[4:6], jnp.zeros((2,), dtype=jnp.float32))
    np.testing.assert_array_equal(unrouted[:4], first[:4])


def test_asymmetric_reward_effect_axes_have_exact_causal_ownership() -> None:
    rewards = jnp.asarray(
        (
            (10.0, 100.0),  # actual0, actual1
            (7.0, 70.0),  # base0, actual1
            (5.0, 50.0),  # actual0, base1
            (1.0, 10.0),  # base0, base1
        ),
        dtype=jnp.float32,
    )
    means, own, partner, interaction, joint_mean = _joint_reward_effects(rewards)

    np.testing.assert_array_equal(means, (55.0, 38.5, 27.5, 5.5))
    np.testing.assert_array_equal(own, (3.0, 50.0))
    np.testing.assert_array_equal(partner, (5.0, 30.0))
    np.testing.assert_array_equal(interaction, (-1.0, -10.0))
    assert float(joint_mean) == 49.5


def test_one_event_is_outer_atomic_and_eager_compiled_equivalent() -> None:
    evaluator = HiddenPrototypeTwoAgentEvaluator(_short_protocol())
    state = evaluator.initialize(HIDDEN_INFERRED_FULL)

    eager = evaluator.step(
        state,
        jnp.asarray(0, dtype=jnp.int32),
        route_inference=jnp.asarray(True),
    )
    compiled = evaluator.compiled_step(
        state,
        jnp.asarray(0, dtype=jnp.int32),
        route_inference=jnp.asarray(True),
    )

    assert bool(eager.trace.outer_transaction_committed)
    assert _tree_equal(eager, compiled)
    np.testing.assert_array_equal(eager.trace.environment_pre_words, (0, 0))
    np.testing.assert_array_equal(eager.trace.environment_post_words, (0, 1))
    np.testing.assert_array_equal(eager.trace.context_pre_words, np.zeros((2, 2)))
    np.testing.assert_array_equal(
        eager.trace.context_post_words,
        np.asarray(((0, 1), (0, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(eager.trace.nested_pre_words, np.zeros((2, 7, 2)))
    np.testing.assert_array_equal(
        eager.trace.nested_post_words,
        np.tile(np.asarray((0, 1), dtype=np.uint32), (2, 7, 1)),
    )
    assert bool(jnp.all(eager.trace.environment_proposals_applied))
    assert bool(jnp.all(eager.trace.context_candidate_updates_applied))
    assert bool(jnp.all(eager.trace.preview_updates_valid))
    assert bool(jnp.all(eager.trace.candidate_updates_valid))
    assert bool(jnp.all(eager.trace.old_bank_routing_valid))
    assert bool(jnp.all(eager.trace.memory_contract_valid))
    assert bool(jnp.all(eager.trace.horde_contract_valid))
    assert bool(jnp.all(eager.trace.world_model_contract_valid))
    assert bool(jnp.all(eager.trace.no_oracle_cue_consumed == jnp.asarray(False)))
    assert eager.trace.joint_rewards.shape == (4, 2)

    rejected = evaluator.step(
        state,
        jnp.asarray(0, dtype=jnp.int32),
        route_inference=jnp.asarray(True),
        force_outer_rejection=jnp.asarray(True),
    )
    assert not bool(rejected.trace.outer_transaction_committed)
    assert _tree_equal(rejected.state, state)
    np.testing.assert_array_equal(rejected.trace.environment_post_words, (0, 0))
    np.testing.assert_array_equal(rejected.trace.context_post_words, np.zeros((2, 2)))
    np.testing.assert_array_equal(rejected.trace.nested_post_words, np.zeros((2, 7, 2)))
    np.testing.assert_array_equal(
        rejected.trace.context_post_onehot,
        np.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rejected.trace.learner_context_channels_next,
        rejected.trace.learner_context_channels_pre,
    )


def test_short_panel_is_matched_bounded_and_descriptive_only() -> None:
    report = run_hidden_prototype_two_agent_continual_life_development(
        _short_protocol(),
        verify_engine_parity=True,
    )
    assert report["development_only"] is True
    assert report["scientific_promotion_allowed"] is False
    assert report["accepted_scientific_evidence"] is False
    assert report["acceptance_status"] == "not-assessed"
    assert report["writer_available"] is False
    assert report["winner_selected"] is False
    assert report["context_capacity_pressure_claimed"] is False
    assert report["checkpoint_resume_claimed"] is False

    runs = cast(list[dict[str, Any]], report["runs"])
    assert [run["arm"] for run in runs] == [
        HIDDEN_INFERRED_FULL,
        HIDDEN_INFERENCE_UNROUTED,
    ]
    assert all(len(run["trace"]) == 3 for run in runs)
    assert all(run["execution_parity"]["state_and_trace_equivalent"] for run in runs)
    assert all(run["metrics"]["memory_evictions"] >= 2 for run in runs)
    assert all(
        run["metrics"]["memory_evicted_provenance_summary"]["count"]
        == run["metrics"]["memory_evictions"]
        for run in runs
    )
    assert all(
        run["metrics"]["memory_eviction_accounting"]
        == {
            "expected_after_fixed_capacity_fill": 4,
            "observed": 4,
            "exact": True,
        }
        for run in runs
    )
    assert all(
        set(run["metrics"]["memory_counterfactual_outcomes_observed"])
        <= {"benefit", "harm", "neutral"}
        for run in runs
    )
    for run in runs:
        work = run["work"]
        assert work["environment_proposal_calls"] == 12
        assert work["context_inference_update_calls"] == 6
        assert work["prototype_update_calls"] == 12
        assert work["discarded_preview_update_calls"] == 6
        assert work["committed_candidate_update_calls"] == 6
        assert work["outer_atomic_decisions"] == 3
        assert work["checkpoint_save_calls"] == 0
        assert work["checkpoint_load_calls"] == 0
        assert run["resources"]["logical_fixed_allocation"] is True
        assert run["resources"]["initial_persistent_state_nbytes"] == (
            run["resources"]["final_persistent_state_nbytes"]
        )
        resources = run["resources"]
        assert resources["initial_persistent_state_nbytes"] == (
            resources["environment_state_nbytes"]
            + 2 * resources["context_state_nbytes_per_agent"]
            + sum(resources["prototype_state_nbytes_per_agent_initial"])
            + resources["outer_auxiliary_state_nbytes"]
        )
        assert resources["initial_persistent_state_decomposition_nbytes"] == (
            resources["initial_persistent_state_nbytes"]
        )
        assert resources["staged_full-state-copy_lower_bound_nbytes"] == (
            4 * resources["environment_state_nbytes"]
            + 2 * resources["context_state_nbytes_per_agent"]
            + 2 * sum(resources["prototype_state_nbytes_per_agent_initial"])
        )
        for index, event in enumerate(run["trace"]):
            assert event["event_index"] == index
            assert event["outer_transaction_committed"] is True
            assert event["no_oracle"]["current_rule_consumed"] is False
            assert event["no_oracle"]["phase_or_boundary_consumed"] is False
            assert event["no_oracle"]["visible_world_cue_consumed"] is False
            dispatch = event["joint_dispatch"]
            assert dispatch["reward_aggregation"] == (
                "arithmetic mean of the two receiving-agent rewards"
            )
            assert all(len(row) == 2 for row in dispatch["rewards"].values())
            for proposal, reward_vector in dispatch["rewards"].items():
                assert dispatch["mean_agent_reward"][proposal] == pytest.approx(
                    sum(reward_vector) / 2.0
                )
            rewards = dispatch["rewards"]
            per_agent_effects = dispatch["effects"]["per_agent"]
            assert per_agent_effects[0]["own_action"] == pytest.approx(
                rewards["actual_actual"][0] - rewards["base0_actual1"][0]
            )
            assert per_agent_effects[1]["own_action"] == pytest.approx(
                rewards["actual_actual"][1] - rewards["actual0_base1"][1]
            )
            assert per_agent_effects[0]["partner_action"] == pytest.approx(
                rewards["actual_actual"][0] - rewards["actual0_base1"][0]
            )
            assert per_agent_effects[1]["partner_action"] == pytest.approx(
                rewards["actual_actual"][1] - rewards["base0_actual1"][1]
            )
            for agent_index in range(2):
                assert per_agent_effects[agent_index]["interaction"] == pytest.approx(
                    rewards["actual_actual"][agent_index]
                    - rewards["base0_actual1"][agent_index]
                    - rewards["actual0_base1"][agent_index]
                    + rewards["base_base"][agent_index]
                )
            assert event["context"]["pre_words"] == [[0, index], [0, index]]
            assert event["context"]["post_words"] == [
                [0, index + 1],
                [0, index + 1],
            ]
            for agent in event["agents"]:
                assert set(agent["nested_clocks"]) == {
                    "prototype",
                    "oak",
                    "stomp",
                    "horde",
                    "world_model",
                    "feature_observe",
                    "memory",
                }
                assert all(
                    clock["pre"] == [0, index]
                    and clock["post"] == [0, index + 1]
                    for clock in agent["nested_clocks"].values()
                )
                assert agent["old_bank_routing"]["valid"] is True
                assert agent["memory"]["query_before_write"] is True
                assert agent["memory"]["wrote"] is True
                assert agent["memory"]["candidate_contract_valid"] is True
                assert agent["memory"]["carried_by_outer_transaction"] is True
                assert agent["horde"][
                    "explicit_prediction_matches_managed_update"
                ] is True
                assert agent["world_model"]["prediction_error_contract_valid"] is True

    comparison = cast(dict[str, object], report["matched_comparison"])
    assert comparison["persistent_state_shape_matched"] is True
    assert comparison["logical_work_matched"] is True
    assert comparison["only_declared_intervention"] == (
        "route the same past-only inferred onehot into the two formerly visible cue coordinates"
    )
    assert "winner" not in comparison
