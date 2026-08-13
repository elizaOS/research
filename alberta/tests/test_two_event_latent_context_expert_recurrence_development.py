"""Static contracts for the consumed-root two-event latent-expert diagnostic."""

from __future__ import annotations

import chex
import pytest

from alberta_framework.core.two_event_latent_context_experts import (
    TwoEventLatentContextExpertLearner,
)
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    FastSlowRecurrenceProtocol,
    _source_arrays,
)
from alberta_framework.evaluation.two_event_latent_context_expert_recurrence_development import (
    ARM_NAMES,
    TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA,
    TwoEventLatentContextExpertRecurrenceProtocol,
    _arm_config,
    _source_arrays_bound,
    _state_sha256,
)

pytestmark = pytest.mark.unit


def test_protocol_freezes_h_two_and_the_consumed_metadata_free_source() -> None:
    protocol = TwoEventLatentContextExpertRecurrenceProtocol()
    payload = protocol.to_config()

    assert (
        protocol.schema_version
        == TWO_EVENT_LATENT_CONTEXT_EXPERT_RECURRENCE_PROTOCOL_SCHEMA
    )
    assert protocol.development_root_seed == 20_260_802
    assert protocol.phase_steps == 512
    assert protocol.total_steps == 1536
    assert protocol.confirmation_horizon == 2
    assert protocol.max_experts == 2
    assert protocol.step_size == 0.05
    assert protocol.grad_clip == 10.0
    assert payload["development_root_already_consumed"] is True
    assert payload["new_seed_or_initialization_drawn"] is False
    assert payload["seed_or_hyperparameter_search_performed"] is False
    assert payload["margin_or_dwell_parameter_present"] is False
    assert payload["learner_metadata_exposed"] == []
    assert TwoEventLatentContextExpertRecurrenceProtocol.from_config(payload) == protocol

    with pytest.raises(ValueError, match="frozen"):
        TwoEventLatentContextExpertRecurrenceProtocol(confirmation_horizon=3)
    with pytest.raises(ValueError, match="frozen"):
        TwoEventLatentContextExpertRecurrenceProtocol(step_size=0.1)


def test_source_binding_reuses_fast_slow_arrays_manifest_and_key_exactly() -> None:
    protocol = TwoEventLatentContextExpertRecurrenceProtocol()
    actual = _source_arrays_bound(protocol)
    expected = _source_arrays(FastSlowRecurrenceProtocol())

    chex.assert_trees_all_equal(actual[0], expected[0])
    chex.assert_trees_all_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    chex.assert_trees_all_equal(actual[3], expected[3])


def test_arms_differ_only_in_confirmation_routing_and_have_recomputed_capacity() -> None:
    protocol = TwoEventLatentContextExpertRecurrenceProtocol()
    enabled_config = _arm_config(protocol, ARM_NAMES[0])
    disabled_config = _arm_config(protocol, ARM_NAMES[1])
    enabled_payload = enabled_config.to_config()
    disabled_payload = disabled_config.to_config()
    differences = {
        name for name in enabled_payload if enabled_payload[name] != disabled_payload[name]
    }

    assert differences == {"confirmation_routing_enabled"}
    assert enabled_config.confirmation_routing_enabled is True
    assert disabled_config.confirmation_routing_enabled is False
    enabled = TwoEventLatentContextExpertLearner(enabled_config)
    disabled = TwoEventLatentContextExpertLearner(disabled_config)
    enabled_state = enabled.init()
    disabled_state = disabled.init()
    assert _state_sha256(enabled_state) == _state_sha256(disabled_state)
    assert enabled.resource_record() == disabled.resource_record()
    resources = enabled.resource_record()
    assert resources.confirmation_horizon == 2
    assert resources.state_nbytes == 53
    assert resources.prediction_cache_nbytes == 70
    assert resources.maximum_expert_predictions_per_update == 4
    assert resources.maximum_expert_losses_per_update == 2
    assert resources.maximum_candidate_gradients_per_update == 2
    assert resources.maximum_expert_subtree_commits_per_update == 1


def test_design_boundary_preserves_the_existing_one_sample_mechanism() -> None:
    learner = TwoEventLatentContextExpertLearner(
        _arm_config(TwoEventLatentContextExpertRecurrenceProtocol(), ARM_NAMES[0])
    )
    record = learner.design_record

    assert record.conceptual_novelty_claimed is False
    assert any("latent-context" in item for item in record.prior_mechanisms)
    assert any("fixed H=2" in item for item in record.integration_scope)
    assert any("zero-parameter-commit" in item for item in record.integration_scope)
