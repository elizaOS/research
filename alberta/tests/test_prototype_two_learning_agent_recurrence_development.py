"""Static contracts for the two-learning-agent Prototype recurrence rung."""

from __future__ import annotations

from typing import cast

import pytest

from alberta_framework.evaluation.prototype_two_learning_agent_recurrence_development import (
    ACCEPTANCE_STATUS,
    ACCEPTED_SCIENTIFIC_EVIDENCE,
    DEVELOPMENT_ONLY,
    PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA,
    PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA,
    SCIENTIFIC_PROMOTION_ALLOWED,
    TWO_LEARNING_AGENT_RECURRENCE_ARMS,
    PrototypeTwoLearningAgentRecurrenceProtocol,
    _agent_config,
)

pytestmark = pytest.mark.unit


def test_default_protocol_is_distinct_symmetric_nonpromoting_v1() -> None:
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol()
    payload = protocol.to_config()

    assert protocol.segment_length == 512
    assert protocol.total_steps == 1536
    assert payload["schedule"] == ["A1", "B", "A2"]
    assert payload["agent_count"] == 2
    assert payload["schema_version"] == (
        PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_PROTOCOL_SCHEMA
    )
    assert PROTOTYPE_TWO_LEARNING_AGENT_RECURRENCE_REPORT_SCHEMA.endswith(".v1")
    assert DEVELOPMENT_ONLY
    assert not SCIENTIFIC_PROMOTION_ALLOWED
    assert ACCEPTANCE_STATUS == "not-assessed"
    assert not ACCEPTED_SCIENTIFIC_EVIDENCE
    assert PrototypeTwoLearningAgentRecurrenceProtocol.from_config(payload) == protocol


def test_short_protocol_preserves_atomic_work_and_checkpoint_contracts() -> None:
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=2,
        replacement_interval=1,
        metric_window=1,
        arm_names=("joint_full",),
    )
    payload = protocol.to_config()
    checkpoint_contract = cast(dict[str, object], payload["checkpoint_contract"])
    world_model_contract = cast(dict[str, object], payload["world_model_contract"])

    assert protocol.total_steps == 3
    assert protocol.base_observation_dim == 8
    assert protocol.candidate_pair_slots == 28
    assert payload["transaction_contract"] == {
        "joint_environment_proposals_per_event": 4,
        "discarded_no_memory_previews_per_event": 2,
        "committed_agent_candidates_per_event": 2,
        "carry_only_if_every_proposal_and_candidate_accepts": True,
        "simultaneous_immutable_prestates": True,
    }
    assert checkpoint_contract["labels"] == ["initial", "A1", "B", "A2"]
    assert world_model_contract["gamma"] == 1.0
    assert world_model_contract["partner_action_observed"] is False


def test_all_in_one_agent_config_is_stable_base_and_excludes_fusion_and_ia() -> None:
    protocol = PrototypeTwoLearningAgentRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=2,
        replacement_interval=1,
        metric_window=1,
        arm_names=("joint_full",),
    )
    config = _agent_config(protocol, feature_promotion_enabled=True)

    assert config.world_model is not None
    assert config.world_model.observation_dim == protocol.base_observation_dim
    assert config.world_model.gamma == 1.0
    assert config.horde_spec is not None
    assert config.prototype_feature_lifecycle is not None
    assert config.prototype_feature_lifecycle.managed_horde_demons == 2
    assert config.experiential_memory is not None
    assert config.partner_policy_fusion is None
    assert config.ia is None


def test_five_arms_are_symmetric_matched_ablations_and_counterexample() -> None:
    arms = {arm.name: arm for arm in TWO_LEARNING_AGENT_RECURRENCE_ARMS}

    assert tuple(arms) == (
        "joint_full",
        "memory_readout_blocked",
        "feature_promotion_blocked",
        "dual_blocked",
        "cue_masked_counterexample",
    )
    assert arms["joint_full"].memory_readout_enabled
    assert arms["joint_full"].feature_promotion_enabled
    assert not arms["memory_readout_blocked"].memory_readout_enabled
    assert not arms["feature_promotion_blocked"].feature_promotion_enabled
    assert not arms["dual_blocked"].memory_readout_enabled
    assert not arms["dual_blocked"].feature_promotion_enabled
    assert not arms["cue_masked_counterexample"].cue_visible
    assert all("symmetric" in arm.role for arm in arms.values())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"segment_length": 0}, "segment_length"),
        ({"metric_window": 513}, "metric_window"),
        ({"arm_names": ("memory_readout_blocked", "joint_full")}, "canonical-order"),
        ({"arm_names": ("joint_full", "joint_full")}, "canonical-order"),
        ({"arm_names": ("unknown",)}, "unsupported"),
    ],
)
def test_protocol_rejects_noncanonical_variants(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PrototypeTwoLearningAgentRecurrenceProtocol(**kwargs)  # type: ignore[arg-type]


def test_protocol_roundtrip_rejects_relabeling_and_derived_tampering() -> None:
    payload = PrototypeTwoLearningAgentRecurrenceProtocol().to_config()
    with pytest.raises(ValueError, match="fields"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
            {**payload, "scientific_promotion_allowed": True}
        )
    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
            {**payload, "total_steps": 1535}
        )


@pytest.mark.parametrize(
    ("section", "field", "integer_alias"),
    [
        (None, "development_only", 1),
        ("transaction_contract", "simultaneous_immutable_prestates", 1),
        ("checkpoint_contract", "ephemeral_shadow_only", 1),
        ("world_model_contract", "partner_action_observed", 0),
    ],
)
def test_protocol_roundtrip_rejects_integer_aliases_for_booleans(
    section: str | None,
    field: str,
    integer_alias: int,
) -> None:
    payload = PrototypeTwoLearningAgentRecurrenceProtocol().to_config()
    if section is None:
        payload[field] = integer_alias
    else:
        cast(dict[str, object], payload[section])[field] = integer_alias

    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(payload)
    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
            {**payload, "development_only": 1}
        )
    transaction = cast(dict[str, object], payload["transaction_contract"])
    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
            {
                **payload,
                "transaction_contract": {
                    **transaction,
                    "carry_only_if_every_proposal_and_candidate_accepts": 1,
                },
            }
        )
    world_model = cast(dict[str, object], payload["world_model_contract"])
    with pytest.raises(ValueError, match="canonical"):
        PrototypeTwoLearningAgentRecurrenceProtocol.from_config(
            {
                **payload,
                "world_model_contract": {
                    **world_model,
                    "partner_action_observed": 0,
                },
            }
        )
