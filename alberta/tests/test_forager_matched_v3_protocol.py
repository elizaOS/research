"""Strict static contracts for the unissued matched-v3 Forager protocol."""

from __future__ import annotations

import copy
import hashlib

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_protocol as protocol,
)

pytestmark = pytest.mark.unit


def test_cumulative_reward_metric_is_exact_and_detached() -> None:
    descriptor = protocol.cumulative_reward_metric_descriptor()
    assert descriptor == {
        "schema_version": "alberta.forager_cumulative_reward_metric.v1",
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "observation_type": "color",
        "aperture_size": 9,
        "horizon": 499_712,
        "raw_reward_values": [-1, 0, 1, 30],
        "accumulation": "ordered_exact_integer_sum",
        "score_bounds": {"minimum": -499_712, "maximum": 14_991_360},
        "ordered_difference_bounds": {
            "minimum": -15_491_072,
            "maximum": 15_491_072,
            "range_width": 30_982_144,
        },
        "trace_completeness_required": True,
        "out_of_set_reward_rejected": True,
        "tail_or_ema_metric": False,
    }
    canonical = protocol.canonical_cumulative_reward_metric_bytes()
    assert hashlib.sha256(canonical).hexdigest() == protocol.CUMULATIVE_REWARD_METRIC_SHA256
    assert protocol.CUMULATIVE_REWARD_METRIC_SHA256 == (
        "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
    )
    descriptor["raw_reward_values"].append(99)
    assert protocol.cumulative_reward_metric_descriptor()["raw_reward_values"] == [-1, 0, 1, 30]


@pytest.mark.parametrize("value", [-499_712, 0, 14_991_360])
def test_validate_cumulative_reward_score_accepts_exact_integer_bounds(value: int) -> None:
    assert protocol.validate_cumulative_reward_score(value) == value


@pytest.mark.parametrize(
    "value",
    [True, False, -499_713, 14_991_361, 0.0, float("nan"), "0", None],
)
def test_validate_cumulative_reward_score_rejects_aliases_and_out_of_bounds(
    value: object,
) -> None:
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.validate_cumulative_reward_score(value)


def test_trial_block_generator_plan_is_uninstantiated_and_domain_separated() -> None:
    descriptor = protocol.trial_block_generator_plan_descriptor()
    assert descriptor == {
        "schema_version": "alberta.forager_trial_block_generator_plan.v1",
        "status": "uninstantiated_future_randomness_required",
        "sampling_model": "iid_with_replacement",
        "root_token_bits": 256,
        "derivation": (
            "uint32be_length_prefixed_domain_root_namespace_shake256_uint31be_v1"
        ),
        "derivation_domain": "alberta.forager.matched_v3.trial_block.seed.v1",
        "framing": "each_component_prefixed_by_uint32_big_endian_byte_length",
        "shake256_output_bytes_per_seed": 4,
        "seed_conversion": "big_endian_uint32_mask_most_significant_bit_to_uint31",
        "seed_minimum": 0,
        "seed_maximum": 2_147_483_647,
        "draw_index_minimum": 0,
        "draw_index_maximum": 9_223_372_036_854_775_807,
        "block_identity": "block_<draw_index_uint64_hex16>_<root_token_sha256>",
        "draw_index_affects_seed_derivation": False,
        "future_randomness_receipt_required": True,
        "environment_namespace": "environment",
        "agent_namespace_template": "agent/<candidate_id>",
        "collision_policy": "retain_draws_without_deduplication",
        "qualification_or_probe_access_allowed": False,
        "outcome_informed_extension_allowed": False,
        "available_case_analysis_allowed": False,
    }
    assert (
        hashlib.sha256(protocol.canonical_trial_block_generator_plan_bytes()).hexdigest()
        == protocol.TRIAL_BLOCK_GENERATOR_PLAN_SHA256
    )
    assert protocol.TRIAL_BLOCK_GENERATOR_PLAN_SHA256 == (
        "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
    )
    descriptor["status"] = "instantiated"
    assert (
        protocol.trial_block_generator_plan_descriptor()["status"]
        == "uninstantiated_future_randomness_required"
    )


def test_protocol_descriptors_ignore_mutated_private_construction_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_bytes = protocol.canonical_cumulative_reward_metric_bytes()
    generator_bytes = protocol.canonical_trial_block_generator_plan_bytes()
    monkeypatch.setitem(protocol._CUMULATIVE_REWARD_METRIC, "horizon", 1)
    monkeypatch.setitem(protocol._TRIAL_BLOCK_GENERATOR_PLAN, "status", "mutated")

    assert protocol.canonical_cumulative_reward_metric_bytes() == metric_bytes
    assert protocol.canonical_trial_block_generator_plan_bytes() == generator_bytes
    assert protocol.cumulative_reward_metric_descriptor()["horizon"] == 499_712
    assert protocol.trial_block_generator_plan_descriptor()["status"] == (
        "uninstantiated_future_randomness_required"
    )


def test_protocol_constants_replay_metric_arithmetic() -> None:
    assert protocol.MATCHED_V3_HORIZON * 30 == protocol.MATCHED_V3_SCORE_MAXIMUM
    assert -protocol.MATCHED_V3_HORIZON == protocol.MATCHED_V3_SCORE_MINIMUM
    assert protocol.MATCHED_V3_DIFFERENCE_MAXIMUM == 31 * protocol.MATCHED_V3_HORIZON
    assert protocol.MATCHED_V3_DIFFERENCE_MINIMUM == -31 * protocol.MATCHED_V3_HORIZON
    assert protocol.MATCHED_V3_DIFFERENCE_RANGE_WIDTH == 62 * protocol.MATCHED_V3_HORIZON


def test_trial_block_derivation_has_frozen_domain_separated_test_vector() -> None:
    root_token = bytes(range(32))
    derivation = protocol.derive_trial_block_seeds(
        root_token,
        ("candidate_b", "candidate_a"),
        draw_index=7,
    )

    assert derivation.block_id == (
        "block_0000000000000007_630dcd2966c4336691125448bbb25b4ff412a49c732db2c8ab"
        "c1b8581bd710dd"
    )
    assert derivation.draw_index == 7
    assert derivation.root_token_sha256 == (
        "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
    )
    assert derivation.environment_seed == 79_469_618
    assert tuple((item.candidate_id, item.seed) for item in derivation.agent_seeds) == (
        ("candidate_b", 1_331_302_420),
        ("candidate_a", 584_050_687),
    )
    assert derivation.payload_sha256 == (
        "38fdcb396cc0ad3f271c4aefda29924136eaa0913137164b5761a7215512d6c1"
    )


def test_trial_block_derivation_is_replayable_detached_and_root_opaque() -> None:
    root_token = bytes(reversed(range(32)))
    candidate_ids = ("alberta", "external.dqn", "adapted-ppo-gru")
    derivation = protocol.derive_trial_block_seeds(
        root_token,
        candidate_ids,
        draw_index=19,
    )
    payload = derivation.to_payload()

    assert payload["root_token_embedded"] is False
    assert root_token.hex() not in derivation.canonical_json().decode("ascii")
    assert protocol.validate_trial_block_derivation(
        copy.deepcopy(payload),
        root_token=root_token,
        expected_draw_index=19,
        expected_candidate_ids=candidate_ids,
    ) == derivation

    payload["candidate_ids"].append("mutation")
    assert derivation.to_payload()["candidate_ids"] == list(candidate_ids)


def test_environment_seed_is_panel_independent_and_agent_seed_is_id_stable() -> None:
    root_token = b"\xa5" * 32
    small = protocol.derive_trial_block_seeds(
        root_token,
        ("candidate_a", "candidate_b"),
        draw_index=0,
    )
    reordered = protocol.derive_trial_block_seeds(
        root_token,
        ("candidate_b", "candidate_c", "candidate_a"),
        draw_index=0,
    )

    assert small.environment_seed == reordered.environment_seed
    assert small.agent_seed("candidate_a") == reordered.agent_seed("candidate_a")
    assert small.agent_seed("candidate_b") == reordered.agent_seed("candidate_b")
    assert small.agent_seed("candidate_a") != small.agent_seed("candidate_b")
    assert tuple(item.candidate_id for item in reordered.agent_seeds) == (
        "candidate_b",
        "candidate_c",
        "candidate_a",
    )


@pytest.mark.parametrize(
    ("root_token", "candidate_ids"),
    [
        (b"short", ("candidate",)),
        (b"x" * 33, ("candidate",)),
        ("x" * 32, ("candidate",)),
        (b"x" * 32, ()),
        (b"x" * 32, ("duplicate", "duplicate")),
        (b"x" * 32, ("bad/id",)),
        (b"x" * 32, ("",)),
    ],
)
def test_trial_block_derivation_rejects_malformed_roots_and_candidate_panels(
    root_token: object,
    candidate_ids: tuple[str, ...],
) -> None:
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.derive_trial_block_seeds(
            root_token,  # type: ignore[arg-type]
            candidate_ids,
            draw_index=0,
        )


@pytest.mark.parametrize("draw_index", [True, False, -1, 2**63])
def test_trial_block_derivation_rejects_invalid_draw_indices(draw_index: object) -> None:
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.derive_trial_block_seeds(
            b"x" * 32,
            ("candidate",),
            draw_index=draw_index,  # type: ignore[arg-type]
        )


def test_repeated_root_draws_keep_seeds_but_have_unique_block_identity() -> None:
    root_token = b"\x5a" * 32
    first = protocol.derive_trial_block_seeds(
        root_token,
        ("candidate_a", "candidate_b"),
        draw_index=3,
    )
    repeated = protocol.derive_trial_block_seeds(
        root_token,
        ("candidate_a", "candidate_b"),
        draw_index=11,
    )

    assert first.block_id != repeated.block_id
    assert first.environment_seed == repeated.environment_seed
    assert first.agent_seeds == repeated.agent_seeds


def test_trial_block_replay_rejects_tampering_wrong_root_and_reordered_panel() -> None:
    root_token = b"\x11" * 32
    candidate_ids = ("candidate_a", "candidate_b")
    payload = protocol.derive_trial_block_seeds(
        root_token,
        candidate_ids,
        draw_index=5,
    ).to_payload()

    tampered = copy.deepcopy(payload)
    tampered["environment"]["seed"] += 1
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.validate_trial_block_derivation(
            tampered,
            root_token=root_token,
            expected_draw_index=5,
            expected_candidate_ids=candidate_ids,
        )
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.validate_trial_block_derivation(
            payload,
            root_token=b"\x12" * 32,
            expected_draw_index=5,
            expected_candidate_ids=candidate_ids,
        )
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.validate_trial_block_derivation(
            payload,
            root_token=root_token,
            expected_draw_index=5,
            expected_candidate_ids=tuple(reversed(candidate_ids)),
        )
    with pytest.raises(protocol.ForagerMatchedV3ProtocolError):
        protocol.validate_trial_block_derivation(
            payload,
            root_token=root_token,
            expected_draw_index=6,
            expected_candidate_ids=candidate_ids,
        )
