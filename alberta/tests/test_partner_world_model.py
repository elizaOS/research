"""Development tests for bounded joint partner/world prediction."""

from __future__ import annotations

import dataclasses
import json

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.partner_world_model import (
    BoundedPartnerWorldModel,
    BoundedPartnerWorldModelConfig,
)
from alberta_framework.evaluation.partner_world_diagnostic import (
    ChangingPartnerDiagnosticConfig,
    run_changing_partner_diagnostic,
    summarize_changing_partner_diagnostic,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_public_signals", 0),
        ("n_actions", 1),
        ("outcome_dim", 0),
        ("partner_context_mode", "oracle"),
        ("behavior_step_size", 0.0),
        ("behavior_step_size", 1.01),
        ("world_step_size", 0.0),
        ("reward_bound", 0.0),
        ("reward_bound", float("nan")),
        ("reward_bound", float("inf")),
        ("outcome_bound", 0.0),
        ("outcome_bound", float("nan")),
        ("outcome_bound", float("inf")),
        ("min_probability", 0.0),
        ("min_probability", 1.0),
    ],
)
def test_config_rejects_invalid_parameters(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "n_public_signals": 2,
        "n_actions": 2,
        "outcome_dim": 2,
        "partner_context_mode": "observable_history",
        "behavior_step_size": 0.25,
        "world_step_size": 0.25,
        "reward_bound": 1.0,
        "outcome_bound": 1.0,
        "min_probability": 1e-6,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        BoundedPartnerWorldModelConfig(**kwargs)  # type: ignore[arg-type]


def _tree_nbytes(tree: object) -> int:
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(tree))


def test_matched_modes_have_exact_fixed_resource_budget() -> None:
    aware = BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            partner_context_mode="observable_history",
        )
    )
    stationary = BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            partner_context_mode="stationary",
        )
    )

    assert aware.resource_budget == stationary.resource_budget
    assert aware.resource_budget.to_dict() == {
        "partner_context_rows": 10,
        "allocated_float32_scalars": 32,
        "allocated_int32_scalars": 4,
        "state_nbytes": 144,
        "learned_float32_scalars_touched_per_update": 5,
        "administrative_int32_scalars_touched_per_update": 4,
        "replay_capacity": 0,
    }
    initial = aware.init()
    state = initial
    for step in range(17):
        partner_action = step % 2
        coordinated = float((step // 2) % 2 == partner_action)
        state = aware.update(
            state,
            jnp.asarray((step // 3) % 2, dtype=jnp.int32),
            jnp.asarray((step // 2) % 2, dtype=jnp.int32),
            jnp.asarray(partner_action, dtype=jnp.int32),
            jnp.asarray(2.0 * coordinated - 1.0, dtype=jnp.float32),
            jnp.asarray(
                [coordinated, 0.5 * ((step // 2) % 2 + partner_action)],
                dtype=jnp.float32,
            ),
        ).state

    assert _tree_nbytes(initial) == aware.resource_budget.state_nbytes
    assert _tree_nbytes(state) == aware.resource_budget.state_nbytes
    assert int(state.step_count) == 17


def test_update_is_strictly_predict_before_observe() -> None:
    model = BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            behavior_step_size=0.25,
            world_step_size=0.25,
        )
    )
    state = model.init()
    signal = jnp.asarray(0, dtype=jnp.int32)
    own_action = jnp.asarray(1, dtype=jnp.int32)
    partner_action = jnp.asarray(1, dtype=jnp.int32)
    before_partner = model.predict_partner(state, signal)
    before_decision = model.decide(state, signal)
    before_world = model.predict_world(state, own_action, partner_action)

    update = model.update(
        state,
        signal,
        own_action,
        partner_action,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        update.decision.partner.probabilities,
        before_partner.probabilities,
    )
    chex.assert_trees_all_close(
        update.decision.expected_rewards,
        before_decision.expected_rewards,
    )
    chex.assert_trees_all_close(
        update.observed_joint_prediction.reward,
        before_world.reward,
    )
    chex.assert_trees_all_close(
        update.observed_joint_prediction.outcome,
        before_world.outcome,
    )
    chex.assert_trees_all_close(
        update.partner_action_probability,
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        update.observed_reward_error,
        jnp.asarray(-1.0, dtype=jnp.float32),
    )

    learned_behavior_row = update.state.partner_probabilities[before_partner.context_index]
    after_partner = model.predict_partner(update.state, signal)
    after_world = model.predict_world(update.state, own_action, partner_action)
    chex.assert_trees_all_close(
        learned_behavior_row,
        jnp.asarray([0.375, 0.625], dtype=jnp.float32),
    )
    # The next prediction conditions on the newly observed history and
    # therefore routes to a distinct, still-untrained row.
    chex.assert_trees_all_close(
        after_partner.probabilities,
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        after_world.reward,
        jnp.asarray(0.25, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        after_world.outcome,
        jnp.asarray([0.25, 0.25], dtype=jnp.float32),
    )


def test_decision_marginalizes_joint_world_predictions_over_partner() -> None:
    model = BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            partner_context_mode="stationary",
            behavior_step_size=1.0,
            world_step_size=1.0,
        )
    )
    state = model.init()
    signal = jnp.asarray(0, dtype=jnp.int32)

    for own_action in range(2):
        for partner_action in range(2):
            coordinated = float(own_action == partner_action)
            state = model.update(
                state,
                signal,
                jnp.asarray(own_action, dtype=jnp.int32),
                jnp.asarray(partner_action, dtype=jnp.int32),
                jnp.asarray(2.0 * coordinated - 1.0, dtype=jnp.float32),
                jnp.asarray(
                    [coordinated, 0.5 * (own_action + partner_action)],
                    dtype=jnp.float32,
                ),
            ).state

    context = int(model.predict_partner(state, signal).context_index)
    partner_probabilities = state.partner_probabilities.at[context].set(
        jnp.asarray([0.1, 0.9], dtype=jnp.float32)
    )
    state = dataclasses.replace(
        state,
        partner_probabilities=partner_probabilities,
    )
    decision = model.decide(state, signal)

    chex.assert_trees_all_close(
        decision.expected_rewards,
        jnp.asarray([-0.8, 0.8], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        decision.expected_outcomes,
        jnp.asarray([[0.1, 0.45], [0.9, 0.95]], dtype=jnp.float32),
    )
    assert int(decision.partner.predicted_action) == 1
    assert int(decision.greedy_action) == 1


def test_config_and_state_checkpoint_roundtrip_is_json_exact() -> None:
    model = BoundedPartnerWorldModel(
        BoundedPartnerWorldModelConfig(
            n_public_signals=3,
            n_actions=3,
            outcome_dim=2,
            behavior_step_size=0.15,
            world_step_size=0.4,
        )
    )
    state = model.init()
    state = model.update(
        state,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(-0.75, dtype=jnp.float32),
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    ).state
    payload = json.loads(json.dumps(model.checkpoint_payload(state)))

    restored_model, restored_state = BoundedPartnerWorldModel.from_checkpoint_payload(payload)

    assert restored_model.to_config() == model.to_config()
    assert restored_model.resource_budget == model.resource_budget
    chex.assert_trees_all_equal(restored_state, state)
    assert restored_model.checkpoint_payload(restored_state) == payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("has_history", 2),
        ("previous_public_signal", 99),
        ("step_count", -1),
    ],
)
def test_checkpoint_rejects_invalid_administrative_state(
    field: str,
    value: int,
) -> None:
    model = BoundedPartnerWorldModel(BoundedPartnerWorldModelConfig())
    payload = model.checkpoint_payload(model.init())
    payload["state"][field] = value

    with pytest.raises(ValueError):
        BoundedPartnerWorldModel.from_checkpoint_payload(payload)


def test_checkpoint_rejects_non_simplex_or_extra_fields() -> None:
    model = BoundedPartnerWorldModel(BoundedPartnerWorldModelConfig())
    non_simplex = model.checkpoint_payload(model.init())
    non_simplex["state"]["partner_probabilities"][0] = [0.9, 0.9]
    with pytest.raises(ValueError, match="simplex"):
        BoundedPartnerWorldModel.from_checkpoint_payload(non_simplex)

    extra_field = model.checkpoint_payload(model.init())
    extra_field["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        BoundedPartnerWorldModel.from_checkpoint_payload(extra_field)

    omitted_config = model.checkpoint_payload(model.init())
    del omitted_config["model"]["config"]["outcome_bound"]
    with pytest.raises(ValueError, match="config fields"):
        BoundedPartnerWorldModel.from_checkpoint_payload(omitted_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("has_history", True),
        ("step_count", "1"),
        ("previous_partner_action", 0.0),
    ],
)
def test_checkpoint_rejects_coercive_administrative_values(
    field: str,
    value: object,
) -> None:
    model = BoundedPartnerWorldModel(BoundedPartnerWorldModelConfig())
    payload = model.checkpoint_payload(model.init())
    payload["state"][field] = value
    with pytest.raises(ValueError, match="must be an integer"):
        BoundedPartnerWorldModel.from_checkpoint_payload(payload)


def test_checkpoint_rejects_incoherent_history_state() -> None:
    model = BoundedPartnerWorldModel(BoundedPartnerWorldModelConfig())
    payload = model.checkpoint_payload(model.init())
    payload["state"]["step_count"] = 1
    with pytest.raises(ValueError, match="history flag"):
        BoundedPartnerWorldModel.from_checkpoint_payload(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_length": 0},
        {"phase_offsets": (0, 1)},
        {"phase_offsets": (0, 1, 1)},
        {"phase_offsets": (0, 0, 0)},
        {"n_actions": 1},
        {"exploration_probability": -0.1},
        {"early_window": 513},
    ],
)
def test_development_diagnostic_config_rejects_invalid_stream(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ChangingPartnerDiagnosticConfig(**kwargs)  # type: ignore[arg-type]


def test_development_aba_diagnostic_separates_partner_aware_from_stationary() -> None:
    config = ChangingPartnerDiagnosticConfig(
        seed=0,
        behavior_step_size=0.05,
    )
    result = run_changing_partner_diagnostic(config)
    summary = summarize_changing_partner_diagnostic(result)

    assert summary["claim_scope"] == "development_only"
    assert summary["held_out"] is False
    assert result.history_aware_budget == result.stationary_budget
    assert int(result.history_aware.final_state.step_count) == config.num_steps
    assert int(result.stationary.final_state.step_count) == config.num_steps
    assert summary["config"]["phase_offsets"] == [0, 1, 0]
    assert "phase" not in result.history_aware.final_state.__dataclass_fields__
    assert "phase" not in result.stationary.final_state.__dataclass_fields__

    aware = summary["history_aware"]
    comparisons = summary["comparisons"]
    assert "history context" in summary["comparison_scope"]
    assert min(aware["tail_partner_accuracy"]) >= 0.98
    assert min(aware["tail_greedy_decision_fidelity"]) >= 0.98
    assert comparisons["b_phase_early_history_context_fidelity_gain"] >= 0.20
    assert comparisons["return_a_early_history_context_fidelity_gain"] >= 0.20
    assert comparisons["history_context_partner_accuracy_gain_over_stationary"] >= 0.02
    assert comparisons["history_context_decision_fidelity_gain_over_stationary"] >= 0.02
    assert aware["overall"]["observed_joint_reward_mse"] <= 0.10
    assert aware["overall"]["observed_joint_outcome_mse"] <= 0.05
    assert aware["overall"]["partner_marginal_outcome_mse"] >= 0.0
    assert np.asarray(aware["joint_action_visit_counts"]).sum() == config.num_steps

    # The stationary arm is genuinely restricted to each signal's unknown-
    # history row, while the aware arm uses ordinary prior interaction rows.
    stationary_contexts = result.stationary.context_indices
    stationary_buckets = stationary_contexts % (1 + config.n_actions * config.n_actions)
    assert bool(jnp.all(stationary_buckets == 0))
    assert bool(
        jnp.any(
            result.history_aware.context_indices[1:] % (1 + config.n_actions * config.n_actions) > 0
        )
    )
