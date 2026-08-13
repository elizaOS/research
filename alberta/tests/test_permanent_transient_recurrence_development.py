"""Static contracts for the consumed-root permanent/transient diagnostic."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.permanent_transient import AlbertaPermanentTransientLearner
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    FastSlowRecurrenceProtocol,
    _source_arrays,
)
from alberta_framework.evaluation.permanent_transient_recurrence_development import (
    ARM_NAMES,
    PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA,
    PermanentTransientRecurrenceProtocol,
    _arm_config,
    _source_arrays_bound,
    _state_sha256,
)

pytestmark = pytest.mark.unit


def test_protocol_is_frozen_to_the_already_consumed_metadata_free_root() -> None:
    protocol = PermanentTransientRecurrenceProtocol()
    payload = protocol.to_config()

    assert protocol.schema_version == PERMANENT_TRANSIENT_RECURRENCE_PROTOCOL_SCHEMA
    assert protocol.development_root_seed == 20_260_802
    assert protocol.phase_steps == 512
    assert protocol.total_steps == 1536
    assert protocol.summary_window == 64
    assert payload["development_root_already_consumed_by_fast_slow"] is True
    assert payload["new_seed_drawn"] is False
    assert payload["seed_or_hyperparameter_search_performed"] is False
    assert payload["target_mapping"] == {"A1": "x", "B": "-x", "A2": "x"}
    assert payload["learner_inputs"] == ["observation", "target"]
    assert payload["learner_metadata_exposed"] == []
    assert payload["pre_post_b_permanent_probe_has_no_a2_updates"] is True
    assert PermanentTransientRecurrenceProtocol.from_config(payload) == protocol

    with pytest.raises(ValueError, match="frozen"):
        PermanentTransientRecurrenceProtocol(phase_steps=513)
    with pytest.raises(ValueError, match="frozen"):
        PermanentTransientRecurrenceProtocol(development_root_seed=0)


def test_source_binding_reuses_fast_slow_arrays_manifest_and_key_exactly() -> None:
    protocol = PermanentTransientRecurrenceProtocol()
    actual = _source_arrays_bound(protocol)
    expected = _source_arrays(FastSlowRecurrenceProtocol())

    chex.assert_trees_all_equal(actual[0], expected[0])
    chex.assert_trees_all_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    chex.assert_trees_all_equal(actual[3], expected[3])
    assert actual[0].shape == actual[1].shape == (1536, 1)
    assert actual[0].dtype == actual[1].dtype == jnp.float32


def test_equal_shape_ablation_changes_only_consolidation_and_preserves_work() -> None:
    protocol = PermanentTransientRecurrenceProtocol()
    ordinary_config = _arm_config(protocol, ARM_NAMES[0])
    ablation_config = _arm_config(protocol, ARM_NAMES[1])
    ordinary_payload = ordinary_config.to_config()
    ablation_payload = ablation_config.to_config()
    differences = {
        name for name in ordinary_payload if ordinary_payload[name] != ablation_payload[name]
    }

    assert differences == {
        "permanent_encoder_step_size",
        "permanent_head_step_size",
    }
    assert ablation_config.permanent_encoder_step_size == 0.0
    assert ablation_config.permanent_head_step_size == 0.0

    initialization_key = _source_arrays_bound(protocol)[3]
    ordinary = AlbertaPermanentTransientLearner(ordinary_config)
    ablation = AlbertaPermanentTransientLearner(ablation_config)
    ordinary_state = ordinary.init(initialization_key)
    ablation_state = ablation.init(initialization_key)
    assert _state_sha256(ordinary_state) == _state_sha256(ablation_state)
    assert ordinary.resource_record().to_dict() == ablation.resource_record().to_dict()
    resources = ordinary.resource_record()
    assert resources.state_nbytes == 788
    assert resources.total_hidden_features == 64
    assert resources.maximum_gradient_evaluations_per_update == 2
    assert resources.replay_capacity == 0
    assert resources.persistent_capacity_growth == 0


def test_design_boundary_is_explicitly_alberta_derived_not_source_faithful() -> None:
    learner = AlbertaPermanentTransientLearner(
        _arm_config(PermanentTransientRecurrenceProtocol(), ARM_NAMES[0])
    )
    record = learner.design_record

    assert record.source_faithful is False
    assert record.public_2026_source_located is False
    assert record.primary_paper_license == "CC BY 4.0"
    assert record.reference_code_license == "MIT"
    assert len(record.departures) == 7
    assert "Alberta-derived" in record.method_name
