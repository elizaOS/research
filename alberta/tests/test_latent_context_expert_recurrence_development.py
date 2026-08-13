"""Static contracts for the consumed-root latent-context expert diagnostic."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.latent_context_experts import LatentContextExpertLearner
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    FastSlowRecurrenceProtocol,
    _source_arrays,
)
from alberta_framework.evaluation.latent_context_expert_recurrence_development import (
    ARM_NAMES,
    LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA,
    LatentContextExpertRecurrenceProtocol,
    _arm_config,
    _source_arrays_bound,
    _state_sha256,
)

pytestmark = pytest.mark.unit


def test_protocol_is_frozen_to_consumed_metadata_free_source() -> None:
    protocol = LatentContextExpertRecurrenceProtocol()
    payload = protocol.to_config()

    assert protocol.schema_version == LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA
    assert protocol.development_root_seed == 20_260_802
    assert protocol.phase_steps == 512
    assert protocol.total_steps == 1536
    assert protocol.max_experts == 2
    assert protocol.step_size == 0.05
    assert protocol.grad_clip == 10.0
    assert payload["development_root_already_consumed"] is True
    assert payload["new_seed_or_initialization_drawn"] is False
    assert payload["seed_or_hyperparameter_search_performed"] is False
    assert payload["learner_metadata_exposed"] == []
    assert payload["first_switched_regime_prediction_precedes_outcome"] is True
    assert LatentContextExpertRecurrenceProtocol.from_config(payload) == protocol

    with pytest.raises(ValueError, match="frozen"):
        LatentContextExpertRecurrenceProtocol(phase_steps=513)
    with pytest.raises(ValueError, match="frozen"):
        LatentContextExpertRecurrenceProtocol(step_size=0.1)


def test_source_binding_reuses_fast_slow_arrays_manifest_and_key_exactly() -> None:
    protocol = LatentContextExpertRecurrenceProtocol()
    actual = _source_arrays_bound(protocol)
    expected = _source_arrays(FastSlowRecurrenceProtocol())

    chex.assert_trees_all_equal(actual[0], expected[0])
    chex.assert_trees_all_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    chex.assert_trees_all_equal(actual[3], expected[3])
    assert actual[0].shape == actual[1].shape == (1536, 1)
    assert actual[0].dtype == actual[1].dtype == jnp.float32


def test_arms_differ_only_in_selection_bool_with_equal_state_and_work() -> None:
    protocol = LatentContextExpertRecurrenceProtocol()
    ordinary_config = _arm_config(protocol, ARM_NAMES[0])
    ablation_config = _arm_config(protocol, ARM_NAMES[1])
    ordinary_payload = ordinary_config.to_config()
    ablation_payload = ablation_config.to_config()
    differences = {
        name for name in ordinary_payload if ordinary_payload[name] != ablation_payload[name]
    }

    assert differences == {"selective_gating"}
    assert ordinary_config.selective_gating is True
    assert ablation_config.selective_gating is False
    ordinary = LatentContextExpertLearner(ordinary_config)
    ablation = LatentContextExpertLearner(ablation_config)
    ordinary_state = ordinary.init()
    ablation_state = ablation.init()
    assert _state_sha256(ordinary_state) == _state_sha256(ablation_state)
    assert ordinary.resource_record() == ablation.resource_record()
    resources = ordinary.resource_record()
    assert resources.state_nbytes == 32
    assert resources.prediction_cache_nbytes == 45
    assert resources.maximum_expert_predictions_per_update == 4
    assert resources.maximum_expert_losses_per_update == 2
    assert resources.maximum_candidate_gradients_per_update == 2
    assert resources.maximum_expert_subtree_commits_per_update == 1


def test_design_boundary_credits_existing_active_only_freeze_law() -> None:
    learner = LatentContextExpertLearner(
        _arm_config(LatentContextExpertRecurrenceProtocol(), ARM_NAMES[0])
    )
    record = learner.design_record

    assert record.conceptual_novelty_claimed is False
    assert record.prior_module == "alberta_framework.core.context_inference"
    assert record.prior_mechanism == "ContextInference active-only-freeze law"
