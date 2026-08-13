"""Static, reward-opaque contracts for the unissued matched-v3 Forager lane.

This module deliberately contains no campaign runner, held-out seed, result loader, or
authority mechanism.  It binds the proposed task/metric arithmetic and the requirements for
a future-randomness trial-block generator so later v3 layers cannot silently inherit v2's
tail metric or hand-picked sequential evaluation seeds.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

MATCHED_V3_ENVIRONMENT_ID: Final = "ForagaxTwoBiomeLarge-v1"
MATCHED_V3_OBSERVATION_TYPE: Final = "color"
MATCHED_V3_APERTURE_SIZE: Final = 9
MATCHED_V3_HORIZON: Final = 499_712
MATCHED_V3_RAW_REWARD_VALUES: Final = (-1, 0, 1, 30)
MATCHED_V3_SCORE_MINIMUM: Final = -MATCHED_V3_HORIZON
MATCHED_V3_SCORE_MAXIMUM: Final = 30 * MATCHED_V3_HORIZON
MATCHED_V3_DIFFERENCE_MINIMUM: Final = -31 * MATCHED_V3_HORIZON
MATCHED_V3_DIFFERENCE_MAXIMUM: Final = 31 * MATCHED_V3_HORIZON
MATCHED_V3_DIFFERENCE_RANGE_WIDTH: Final = 62 * MATCHED_V3_HORIZON

CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION: Final = (
    "alberta.forager_cumulative_reward_metric.v1"
)
TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_trial_block_generator_plan.v1"
)
TRIAL_BLOCK_DERIVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_trial_block_derivation.v1"
)

TRIAL_BLOCK_ROOT_TOKEN_BYTES: Final = 32
TRIAL_BLOCK_DRAW_INDEX_MAXIMUM: Final = (1 << 63) - 1
TRIAL_BLOCK_SEED_BITS: Final = 31
TRIAL_BLOCK_SEED_MAXIMUM: Final = (1 << TRIAL_BLOCK_SEED_BITS) - 1
TRIAL_BLOCK_DERIVATION_DOMAIN: Final = (
    "alberta.forager.matched_v3.trial_block.seed.v1"
)
TRIAL_BLOCK_DERIVATION_ENCODING: Final = (
    "uint32be_length_prefixed_domain_root_namespace_shake256_uint31be_v1"
)

_PORTABLE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TRIAL_BLOCK_CANDIDATES: Final = 256


class ForagerMatchedV3ProtocolError(ValueError):
    """A matched-v3 static protocol value violated its exact contract."""


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _PORTABLE_IDENTIFIER.fullmatch(value) is None:
        raise ForagerMatchedV3ProtocolError(f"{name} must be a portable identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ForagerMatchedV3ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_seed(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= TRIAL_BLOCK_SEED_MAXIMUM:
        raise ForagerMatchedV3ProtocolError(
            f"{name} must be an exact unsigned {TRIAL_BLOCK_SEED_BITS}-bit integer"
        )
    return value


def _require_draw_index(value: object) -> int:
    if type(value) is not int or not 0 <= value <= TRIAL_BLOCK_DRAW_INDEX_MAXIMUM:
        raise ForagerMatchedV3ProtocolError(
            "draw_index must be an exact integer between 0 and 2^63-1"
        )
    return value


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3ProtocolError("value is not canonical JSON") from exc


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


_CUMULATIVE_REWARD_METRIC: Final[dict[str, Any]] = {
    "schema_version": CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION,
    "environment_id": MATCHED_V3_ENVIRONMENT_ID,
    "observation_type": MATCHED_V3_OBSERVATION_TYPE,
    "aperture_size": MATCHED_V3_APERTURE_SIZE,
    "horizon": MATCHED_V3_HORIZON,
    "raw_reward_values": list(MATCHED_V3_RAW_REWARD_VALUES),
    "accumulation": "ordered_exact_integer_sum",
    "score_bounds": {
        "minimum": MATCHED_V3_SCORE_MINIMUM,
        "maximum": MATCHED_V3_SCORE_MAXIMUM,
    },
    "ordered_difference_bounds": {
        "minimum": MATCHED_V3_DIFFERENCE_MINIMUM,
        "maximum": MATCHED_V3_DIFFERENCE_MAXIMUM,
        "range_width": MATCHED_V3_DIFFERENCE_RANGE_WIDTH,
    },
    "trace_completeness_required": True,
    "out_of_set_reward_rejected": True,
    "tail_or_ema_metric": False,
}
_CUMULATIVE_REWARD_METRIC_BYTES: Final = _canonical_bytes(_CUMULATIVE_REWARD_METRIC)
CUMULATIVE_REWARD_METRIC_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
if hashlib.sha256(_CUMULATIVE_REWARD_METRIC_BYTES).hexdigest() != (
    CUMULATIVE_REWARD_METRIC_SHA256
):
    raise AssertionError("canonical matched-v3 cumulative-reward metric drifted")


_TRIAL_BLOCK_GENERATOR_PLAN: Final[dict[str, Any]] = {
    "schema_version": TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION,
    "status": "uninstantiated_future_randomness_required",
    "sampling_model": "iid_with_replacement",
    "root_token_bits": 8 * TRIAL_BLOCK_ROOT_TOKEN_BYTES,
    "derivation": TRIAL_BLOCK_DERIVATION_ENCODING,
    "derivation_domain": TRIAL_BLOCK_DERIVATION_DOMAIN,
    "framing": "each_component_prefixed_by_uint32_big_endian_byte_length",
    "shake256_output_bytes_per_seed": 4,
    "seed_conversion": "big_endian_uint32_mask_most_significant_bit_to_uint31",
    "seed_minimum": 0,
    "seed_maximum": TRIAL_BLOCK_SEED_MAXIMUM,
    "draw_index_minimum": 0,
    "draw_index_maximum": TRIAL_BLOCK_DRAW_INDEX_MAXIMUM,
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
_TRIAL_BLOCK_GENERATOR_PLAN_BYTES: Final = _canonical_bytes(_TRIAL_BLOCK_GENERATOR_PLAN)
TRIAL_BLOCK_GENERATOR_PLAN_SHA256: Final = (
    "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
)
if hashlib.sha256(_TRIAL_BLOCK_GENERATOR_PLAN_BYTES).hexdigest() != (
    TRIAL_BLOCK_GENERATOR_PLAN_SHA256
):
    raise AssertionError("canonical matched-v3 trial-block generator plan drifted")


@dataclass(frozen=True, slots=True)
class CandidateAgentSeed:
    """One candidate-private seed derived from a trial block's root token."""

    candidate_id: str
    namespace: str
    seed: int

    def __post_init__(self) -> None:
        candidate_id = _require_identifier(self.candidate_id, "candidate_id")
        if self.namespace != f"agent/{candidate_id}":
            raise ForagerMatchedV3ProtocolError(
                "candidate agent namespace does not match its candidate ID"
            )
        _require_seed(self.seed, "candidate agent seed")

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "namespace": self.namespace,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class TrialBlockSeedDerivation:
    """Replayable scalar seed derivation; the secret/root token stays external."""

    block_id: str
    draw_index: int
    root_token_sha256: str
    candidate_ids: tuple[str, ...]
    environment_seed: int
    agent_seeds: tuple[CandidateAgentSeed, ...]

    def __post_init__(self) -> None:
        digest = _require_sha256(self.root_token_sha256, "root_token_sha256")
        draw_index = _require_draw_index(self.draw_index)
        if self.block_id != f"block_{draw_index:016x}_{digest}":
            raise ForagerMatchedV3ProtocolError(
                "block_id must bind the exact draw index and root-token SHA-256"
            )
        candidate_ids = _validated_candidate_ids(self.candidate_ids)
        _require_seed(self.environment_seed, "environment seed")
        if type(self.agent_seeds) is not tuple or len(self.agent_seeds) != len(
            candidate_ids
        ):
            raise ForagerMatchedV3ProtocolError(
                "agent_seeds must be a tuple matching the candidate panel"
            )
        if any(type(item) is not CandidateAgentSeed for item in self.agent_seeds):
            raise ForagerMatchedV3ProtocolError(
                "agent_seeds contains an invalid seed record"
            )
        if tuple(item.candidate_id for item in self.agent_seeds) != candidate_ids:
            raise ForagerMatchedV3ProtocolError(
                "agent_seeds must preserve the exact candidate-panel order"
            )

    def agent_seed(self, candidate_id: str) -> int:
        """Return the private seed for one candidate in this block."""

        requested = _require_identifier(candidate_id, "candidate_id")
        for item in self.agent_seeds:
            if item.candidate_id == requested:
                return item.seed
        raise ForagerMatchedV3ProtocolError("candidate is not in this trial block")

    def to_body(self) -> dict[str, Any]:
        return {
            "schema_version": TRIAL_BLOCK_DERIVATION_SCHEMA_VERSION,
            "generator_plan": {
                "schema_version": TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION,
                "sha256": TRIAL_BLOCK_GENERATOR_PLAN_SHA256,
            },
            "derivation": TRIAL_BLOCK_DERIVATION_ENCODING,
            "derivation_domain": TRIAL_BLOCK_DERIVATION_DOMAIN,
            "block_id": self.block_id,
            "draw_index": self.draw_index,
            "root_token_sha256": self.root_token_sha256,
            "root_token_embedded": False,
            "candidate_ids": list(self.candidate_ids),
            "environment": {
                "namespace": "environment",
                "seed": self.environment_seed,
            },
            "agents": [item.to_payload() for item in self.agent_seeds],
            "seed_bits": TRIAL_BLOCK_SEED_BITS,
            "seed_minimum": 0,
            "seed_maximum": TRIAL_BLOCK_SEED_MAXIMUM,
            "collision_policy": "retain_derived_seed_collisions_without_redraw",
            "future_randomness_provenance_verified": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
        }

    @property
    def payload_sha256(self) -> str:
        return _canonical_sha256(self.to_body())

    def to_payload(self) -> dict[str, Any]:
        payload = self.to_body()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    def canonical_json(self) -> bytes:
        return _canonical_bytes(self.to_payload())


def _validated_candidate_ids(candidate_ids: object) -> tuple[str, ...]:
    if type(candidate_ids) is not tuple:
        raise ForagerMatchedV3ProtocolError("candidate_ids must be a tuple")
    values = cast(tuple[object, ...], candidate_ids)
    if not 1 <= len(values) <= _MAX_TRIAL_BLOCK_CANDIDATES:
        raise ForagerMatchedV3ProtocolError(
            "candidate_ids must contain between 1 and 256 candidates"
        )
    result = tuple(
        _require_identifier(value, f"candidate_ids[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(result)) != len(result):
        raise ForagerMatchedV3ProtocolError("candidate_ids must be unique")
    return result


def _length_prefix(value: bytes) -> bytes:
    if len(value) >= 1 << 32:
        raise ForagerMatchedV3ProtocolError("derivation component is too long")
    return len(value).to_bytes(4, "big") + value


def _derive_uint31(root_token: bytes, namespace: str) -> int:
    namespace_bytes = namespace.encode("ascii")
    preimage = b"".join(
        (
            _length_prefix(TRIAL_BLOCK_DERIVATION_DOMAIN.encode("ascii")),
            _length_prefix(root_token),
            _length_prefix(namespace_bytes),
        )
    )
    raw = hashlib.shake_256(preimage).digest(4)
    return int.from_bytes(raw, "big") & TRIAL_BLOCK_SEED_MAXIMUM


def derive_trial_block_seeds(
    root_token: bytes,
    candidate_ids: tuple[str, ...],
    *,
    draw_index: int,
) -> TrialBlockSeedDerivation:
    """Derive a common environment seed and private candidate seeds.

    This function does not obtain or endorse future randomness.  Its caller must
    separately authenticate a preregistered 32-byte root token and preserve that
    provenance for replay.
    """

    if type(root_token) is not bytes or len(root_token) != TRIAL_BLOCK_ROOT_TOKEN_BYTES:
        raise ForagerMatchedV3ProtocolError(
            f"root_token must be exactly {TRIAL_BLOCK_ROOT_TOKEN_BYTES} bytes"
        )
    panel = _validated_candidate_ids(candidate_ids)
    exact_draw_index = _require_draw_index(draw_index)
    root_digest = hashlib.sha256(root_token).hexdigest()
    return TrialBlockSeedDerivation(
        block_id=f"block_{exact_draw_index:016x}_{root_digest}",
        draw_index=exact_draw_index,
        root_token_sha256=root_digest,
        candidate_ids=panel,
        environment_seed=_derive_uint31(root_token, "environment"),
        agent_seeds=tuple(
            CandidateAgentSeed(
                candidate_id=candidate_id,
                namespace=f"agent/{candidate_id}",
                seed=_derive_uint31(root_token, f"agent/{candidate_id}"),
            )
            for candidate_id in panel
        ),
    )


def validate_trial_block_derivation(
    payload: Mapping[str, object],
    *,
    root_token: bytes,
    expected_draw_index: int,
    expected_candidate_ids: tuple[str, ...],
) -> TrialBlockSeedDerivation:
    """Replay an exact detached derivation against its external root token."""

    if type(payload) is not dict:
        raise ForagerMatchedV3ProtocolError(
            "trial-block derivation payload must be a plain object"
        )
    expected = derive_trial_block_seeds(
        root_token,
        expected_candidate_ids,
        draw_index=expected_draw_index,
    )
    supplied = cast(dict[str, Any], payload)
    if _canonical_bytes(supplied) != expected.canonical_json():
        raise ForagerMatchedV3ProtocolError(
            "trial-block derivation does not replay from the expected root and panel"
        )
    return expected


def cumulative_reward_metric_descriptor() -> dict[str, Any]:
    """Return a detached copy of the exact proposed v3 primary metric."""
    return cast(
        dict[str, Any],
        json.loads(_CUMULATIVE_REWARD_METRIC_BYTES.decode("utf-8")),
    )


def canonical_cumulative_reward_metric_bytes() -> bytes:
    """Return the metric's canonical UTF-8 JSON bytes."""
    return _CUMULATIVE_REWARD_METRIC_BYTES


def validate_cumulative_reward_score(value: object) -> int:
    """Validate one exact integer score without accepting bool/float aliases."""
    if type(value) is not int:
        raise ForagerMatchedV3ProtocolError("cumulative reward score must be an exact integer")
    score = value
    if score < MATCHED_V3_SCORE_MINIMUM or score > MATCHED_V3_SCORE_MAXIMUM:
        raise ForagerMatchedV3ProtocolError("cumulative reward score is outside task bounds")
    return score


def trial_block_generator_plan_descriptor() -> dict[str, Any]:
    """Return the uninstantiated, detached v3 held-out generator requirements."""
    return cast(
        dict[str, Any],
        json.loads(_TRIAL_BLOCK_GENERATOR_PLAN_BYTES.decode("utf-8")),
    )


def canonical_trial_block_generator_plan_bytes() -> bytes:
    """Return canonical JSON bytes for the uninstantiated generator plan."""
    return _TRIAL_BLOCK_GENERATOR_PLAN_BYTES


__all__ = [
    "CandidateAgentSeed",
    "CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION",
    "CUMULATIVE_REWARD_METRIC_SHA256",
    "ForagerMatchedV3ProtocolError",
    "MATCHED_V3_APERTURE_SIZE",
    "MATCHED_V3_DIFFERENCE_MAXIMUM",
    "MATCHED_V3_DIFFERENCE_MINIMUM",
    "MATCHED_V3_DIFFERENCE_RANGE_WIDTH",
    "MATCHED_V3_ENVIRONMENT_ID",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_OBSERVATION_TYPE",
    "MATCHED_V3_RAW_REWARD_VALUES",
    "MATCHED_V3_SCORE_MAXIMUM",
    "MATCHED_V3_SCORE_MINIMUM",
    "TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION",
    "TRIAL_BLOCK_GENERATOR_PLAN_SHA256",
    "TRIAL_BLOCK_DERIVATION_DOMAIN",
    "TRIAL_BLOCK_DERIVATION_ENCODING",
    "TRIAL_BLOCK_DERIVATION_SCHEMA_VERSION",
    "TRIAL_BLOCK_DRAW_INDEX_MAXIMUM",
    "TRIAL_BLOCK_ROOT_TOKEN_BYTES",
    "TRIAL_BLOCK_SEED_BITS",
    "TRIAL_BLOCK_SEED_MAXIMUM",
    "TrialBlockSeedDerivation",
    "canonical_cumulative_reward_metric_bytes",
    "canonical_trial_block_generator_plan_bytes",
    "cumulative_reward_metric_descriptor",
    "derive_trial_block_seeds",
    "trial_block_generator_plan_descriptor",
    "validate_trial_block_derivation",
    "validate_cumulative_reward_score",
]
