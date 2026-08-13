"""Non-authorizing full driver contract for the matched-v3 PPO-GRU adapter.

The production surface in this module closes the deterministic driver schedule:
one continuing Foragax trajectory, 499,712 interactions, 976 512-step rollouts,
four contiguous 128-step recurrent segments per rollout, four PPO epochs, and
15,616 optimizer transactions.  Environment randomness is owned exclusively by
the shared Foragax bridge.  The PPO core RNG state is the exclusive owner of the
agent key chain and its environment key is cross-checked but never consumed.

The implementation remains runtime-unqualified and non-authorizing.  It embeds
no protected seed, writes no file, and emits only canonical in-memory engineering
or result receipt bytes whose claims are all false.  Small dependency-injected
traces are explicitly engineering-only and cannot be relabelled as a completed
production result.  Outcomes retain the complete ordered raw reward sequence as
immutable signed-int8 two's-complement bytes for the common artifact sink.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata as importlib_metadata
import json
import math
import platform
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NoReturn, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru

PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_runner.v1"
)
PPO_GRU_ENGINEERING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_engineering_receipt.v1"
)
PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_result_receipt.v1"
)
PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_runtime_identity.v1"
)

PPO_GRU_CORE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py"
)
PPO_GRU_CORE_IMPLEMENTATION_SHA256: Final = (
    "58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e"
)
FORAGAX_BRIDGE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
)
PPO_GRU_RUNNER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
)

_MAX_JSON_BYTES: Final = 4 * 1024 * 1024
_SHA256_HEX_LENGTH: Final = 64
_RAW_REWARD_SUPPORT: Final = frozenset({-1, 0, 1, 30})
_TRACE_CHAIN_DOMAIN: Final = b"alberta.forager_matched_v3.ppo_gru.trace_chain.v1\x00"
_OUTCOME_IDENTITY_DOMAIN: Final = (
    b"alberta.forager_matched_v3.ppo_gru.outcome_identity.v1\x00"
)
_RAW_REWARD_TRACE_ENCODING: Final = "signed_int8_twos_complement"
_RAW_SCORE_REDUCTION: Final = "exact_int64_sum"
_RAW_SCORE_SCALING: Final = "none"


class ForagerMatchedV3PPOGRURunnerError(ValueError):
    """A driver, trace, counter, runtime binding, or receipt violated its contract."""


class _ProductionRuntimeCapability:
    __slots__ = ("__weakref__",)


class _ProductionOutcomeCapability:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class _ProductionRuntimeIdentityParser:
    expected_bytes: bytes

    def __call__(self, raw: bytes) -> Mapping[str, Any]:
        parsed = _strict_json_object(raw, "production runtime identity")
        if not hmac.compare_digest(raw, self.expected_bytes):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime identity differs from the opened bridge runtime"
            )
        return parsed


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()


_LIVE_CORE_SOURCE_SHA256: Final = _source_sha256(
    ppo_gru.__file__, PPO_GRU_CORE_IMPLEMENTATION_PATH
)
if not hmac.compare_digest(
    _LIVE_CORE_SOURCE_SHA256, PPO_GRU_CORE_IMPLEMENTATION_SHA256
):
    raise RuntimeError("matched-v3 PPO-GRU core source identity drifted")

# Bind both the bridge's own frozen descriptor and its reviewed implementation
# bytes.  Any later bridge edit fails import until this runner is reviewed again.
FORAGAX_BRIDGE_IMPLEMENTATION_SHA256: Final = (
    "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
)
_LIVE_BRIDGE_SOURCE_SHA256: Final = _source_sha256(
    bridge.__file__, FORAGAX_BRIDGE_IMPLEMENTATION_PATH
)
if not hmac.compare_digest(
    _LIVE_BRIDGE_SOURCE_SHA256, FORAGAX_BRIDGE_IMPLEMENTATION_SHA256
):
    raise RuntimeError("matched-v3 Foragax bridge source identity drifted")

def _non_authorizing_claims() -> dict[str, bool]:
    return {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }


def _receipt_limitations() -> tuple[str, ...]:
    return (
        "Receipt bytes grant no execution, ingestion, promotion, or performance authority.",
        "Runtime and full-horizon scientific qualification remain outside this receipt.",
        "Canonical hashes validate structure and integrity only; they do not independently "
        "attest execution.",
        "Seed provenance is caller-supplied and unverified unless a separate upstream receipt "
        "is bound.",
        "Receipt reward-trace metadata does not contain or independently validate the trace "
        "bytes.",
    )


def _require_exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def _assert_plain_unaliased_json(value: object, label: str) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3PPOGRURunnerError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3PPOGRURunnerError(
                    f"{label} contains a non-string object key"
                )
            pending.extend(mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3PPOGRURunnerError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3PPOGRURunnerError(
                    f"{label} contains a non-finite number"
                )
        elif item is not None and type(item) not in {str, int, bool}:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"{label} contains non-plain JSON type {type(item).__name__}"
            )


def _canonical_json(value: object, label: str) -> bytes:
    _assert_plain_unaliased_json(value, label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} exceeds its byte limit")
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"duplicate JSON key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> NoReturn:
    raise ForagerMatchedV3PPOGRURunnerError(
        f"non-finite JSON number {token!r} is forbidden"
    )


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} must be exact bytes")
    if len(raw) > _MAX_JSON_BYTES:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} exceeds its byte limit")
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3PPOGRURunnerError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(decoded) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} must be a JSON object")
    canonical = _canonical_json(decoded, label)
    if not hmac.compare_digest(raw, canonical):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], decoded)


def _reject_authority_anywhere(value: object, label: str) -> None:
    denied_true_keys = {
        "execution_ready",
        "execution_authorized",
        "scientific_promotion_allowed",
        "performance_claim_allowed",
        "universal_sota_claim_allowed",
        "authority_granted",
        "runtime_qualified",
    }
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            mapping = cast(dict[str, Any], item)
            for key, nested in mapping.items():
                if key in denied_true_keys and nested is not False:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        f"{label} cannot grant {key}"
                    )
                if key in {"protected_seed", "protected_seeds"} and nested is not False:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        f"{label} cannot carry protected seed material"
                    )
                pending.append(nested)
        elif type(item) is list:
            pending.extend(item)


@dataclass(frozen=True, slots=True)
class PPOGRURunnerGeometry:
    """Rollout/update geometry; only one value is a production geometry."""

    horizon: int
    rollout_steps: int
    segment_steps: int
    update_epochs: int

    def __post_init__(self) -> None:
        for name in ("horizon", "rollout_steps", "segment_steps", "update_epochs"):
            _require_exact_int(getattr(self, name), name, minimum=1)
        if self.horizon % self.rollout_steps != 0:
            raise ForagerMatchedV3PPOGRURunnerError(
                "rollout_steps must divide the driver horizon"
            )
        if self.rollout_steps % self.segment_steps != 0:
            raise ForagerMatchedV3PPOGRURunnerError(
                "segment_steps must divide every rollout"
            )
        if self.segments_per_rollout != 4:
            raise ForagerMatchedV3PPOGRURunnerError(
                "every rollout must contain exactly four contiguous segments"
            )
        if self.update_epochs != 4:
            raise ForagerMatchedV3PPOGRURunnerError(
                "every rollout must retain exactly four PPO epochs"
            )

    @property
    def rollout_count(self) -> int:
        return self.horizon // self.rollout_steps

    @property
    def segments_per_rollout(self) -> int:
        return self.rollout_steps // self.segment_steps

    @property
    def optimizer_updates_per_rollout(self) -> int:
        return self.segments_per_rollout * self.update_epochs

    @property
    def optimizer_update_count(self) -> int:
        return self.rollout_count * self.optimizer_updates_per_rollout

    @property
    def action_draw_count(self) -> int:
        return self.horizon

    @property
    def permutation_draw_count(self) -> int:
        return self.rollout_count * self.update_epochs

    @property
    def total_agent_draw_count(self) -> int:
        return 1 + self.action_draw_count + self.permutation_draw_count

    def to_dict(self) -> dict[str, int]:
        return {
            "horizon": self.horizon,
            "rollout_steps": self.rollout_steps,
            "segment_steps": self.segment_steps,
            "segments_per_rollout": self.segments_per_rollout,
            "update_epochs": self.update_epochs,
            "rollout_count": self.rollout_count,
            "optimizer_updates_per_rollout": self.optimizer_updates_per_rollout,
            "optimizer_update_count": self.optimizer_update_count,
        }


MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY: Final = PPOGRURunnerGeometry(
    horizon=499_712,
    rollout_steps=512,
    segment_steps=128,
    update_epochs=4,
)


def _dependency_binding() -> dict[str, Any]:
    return {
        "ppo_gru_core": {
            "configuration_schema_version": ppo_gru.PPO_GRU_CONFIGURATION_SCHEMA_VERSION,
            "configuration_sha256": ppo_gru.PPO_GRU_CONFIGURATION_SHA256,
            "source_descriptor_schema_version": (
                ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "source_descriptor_sha256": ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256,
            "implementation_path": PPO_GRU_CORE_IMPLEMENTATION_PATH,
            "implementation_source_sha256": PPO_GRU_CORE_IMPLEMENTATION_SHA256,
        },
        "foragax_bridge": {
            "descriptor_schema_version": bridge.FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_sha256": bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
            "implementation_path": FORAGAX_BRIDGE_IMPLEMENTATION_PATH,
            "implementation_source_sha256": FORAGAX_BRIDGE_IMPLEMENTATION_SHA256,
            "runtime_identity_schema_version": (
                PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION
            ),
            "runtime_binding_consumption": (
                "registered_identity_reobserved_at_driver_entry_and_exit_"
                "no_execution_authority"
            ),
        },
    }


def _runner_descriptor() -> dict[str, Any]:
    geometry = MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY
    return {
        "schema_version": PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        "candidate_id": "adapted_ppo_gru",
        "status": "implemented_runtime_unqualified",
        "classification": "full_driver_contract_non_authorizing",
        "implementation": {
            "module": "alberta_framework.benchmarks.forager_matched_v3_ppo_gru_runner",
            "path": PPO_GRU_RUNNER_IMPLEMENTATION_PATH,
            "source_self_hash_bound": False,
        },
        "dependencies": _dependency_binding(),
        "production_geometry": geometry.to_dict(),
        "trajectory": {
            "count": 1,
            "continuing": True,
            "automatic_resets": 0,
            "raw_cumulative_score": True,
            "reward_preprocessing": "identity",
            "raw_reward_trace_retained": True,
            "raw_reward_trace_encoding": _RAW_REWARD_TRACE_ENCODING,
            "score_reduction": _RAW_SCORE_REDUCTION,
            "score_scaling": _RAW_SCORE_SCALING,
            "partial_horizon_can_be_complete_result": False,
        },
        "rng_ownership": {
            "implementation": ppo_gru.PPO_GRU_PRNG_IMPLEMENTATION,
            "environment_owner": "shared_foragax_bridge",
            "ppo_environment_key_consumption": 0,
            "agent_owner": "ppo_gru_core_rng_state",
            "parameter_initialization_draws": 1,
            "action_draws": geometry.action_draw_count,
            "action_draws_per_interaction": 1,
            "permutation_draws": geometry.permutation_draw_count,
            "permutation_draws_per_epoch": 1,
            "total_agent_draws": geometry.total_agent_draw_count,
        },
        "seed_provenance": {
            "classification": "caller_supplied_unverified",
            "upstream_receipt_bound": False,
            "protected_seed_status": "unverified",
        },
        "rollout_validation": {
            "behavior_replay_before_update": True,
            "actual_incoming_gru_carries_bound": True,
            "sampled_action_key_replay": True,
            "behavior_log_probability_replay": True,
            "bootstrap_observation_value_and_reset_bound": True,
            "segmented_gae": "four_independent_contiguous_128_step_recursions",
            "segment_time_order": "strictly_increasing",
            "segment_permutation": "exactly_one_agent_key_per_epoch",
            "learning_rate": "linear_by_rollout_from_0.00025",
        },
        "receipts": {
            "engineering_schema_version": PPO_GRU_ENGINEERING_RECEIPT_SCHEMA_VERSION,
            "result_schema_version": PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
            "canonical_in_memory_bytes_only": True,
            "filesystem_writes": False,
            "partial_production_receipt_forbidden": True,
            "production_emission_requires_process_local_completion_capability": True,
            "persisted_parser_is_structural_not_execution_attestation": True,
        },
        "claims": _non_authorizing_claims(),
        "blockers": [
            "bridge_runtime_binding_and_full_trace_parity_unqualified",
            "full_horizon_compilation_wall_clock_and_memory_unqualified",
            "runner_source_closure_and_reproducible_runtime_image_unqualified",
            "result_ingestion_and_scientific_qualification_unimplemented",
        ],
        "limitations": [
            "No protected seed is embedded, generated, requested, or authorized.",
            "Synthetic dependency traces are engineering checks, not scientific evidence.",
            "Only an in-process capability-backed completed result records execution; its "
            "persisted parser remains structural and grants no ingestion authority.",
            "The runner implementation cannot canonically bind its own source without a "
            "separate source-closure artifact.",
            "Persisted receipt hashes prove canonical structure and byte integrity, not "
            "execution provenance.",
            "Seed provenance remains unverified without a separately bound upstream receipt.",
            "Reward trace bytes remain in the in-memory outcome until a common canonical "
            "artifact sink binds and persists them.",
        ],
    }


_RUNNER_DESCRIPTOR: Final = _runner_descriptor()
_RUNNER_DESCRIPTOR_BYTES: Final = _canonical_json(
    _RUNNER_DESCRIPTOR, "PPO-GRU runner descriptor"
)
PPO_GRU_RUNNER_DESCRIPTOR_SHA256: Final = (
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
)
if not hmac.compare_digest(
    hashlib.sha256(_RUNNER_DESCRIPTOR_BYTES).hexdigest(),
    PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
):
    raise RuntimeError("matched-v3 PPO-GRU runner descriptor identity drifted")


def _frozen_dependency_binding() -> dict[str, Any]:
    decoded = _strict_json_object(
        _RUNNER_DESCRIPTOR_BYTES, "frozen PPO-GRU runner descriptor"
    )
    dependencies = decoded.get("dependencies")
    if type(dependencies) is not dict:
        raise RuntimeError("frozen runner dependency binding is invalid")
    return cast(dict[str, Any], dependencies)


def _frozen_runner_binding() -> dict[str, str]:
    return {
        "descriptor_schema_version": "alberta.forager_matched_v3_ppo_gru_runner.v1",
        "descriptor_sha256": hashlib.sha256(_RUNNER_DESCRIPTOR_BYTES).hexdigest(),
    }


def matched_v3_ppo_gru_runner_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the runtime-unqualified runner contract."""

    return _strict_json_object(_RUNNER_DESCRIPTOR_BYTES, "PPO-GRU runner descriptor")


def canonical_matched_v3_ppo_gru_runner_descriptor_bytes() -> bytes:
    """Return the exact canonical non-authorizing runner descriptor bytes."""

    return bytes(_RUNNER_DESCRIPTOR_BYTES)


def parse_matched_v3_ppo_gru_runner_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen runner descriptor."""

    parsed = _strict_json_object(raw, "PPO-GRU runner descriptor")
    if not hmac.compare_digest(raw, _RUNNER_DESCRIPTOR_BYTES):
        raise ForagerMatchedV3PPOGRURunnerError(
            "PPO-GRU runner descriptor differs from the frozen contract"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class PPOGRUStepEvaluation:
    """One policy evaluation before action sampling or environment stepping."""

    outgoing_carry: Array
    logits: Array
    value: Array


@dataclass(frozen=True, slots=True)
class PPOGRURolloutStep:
    """One immutable adapter-visible behavior transition."""

    step_index: int
    observation: Array
    incoming_carry: Array
    reset_before: bool
    action_key: Array
    logits: Array
    action: int
    old_log_prob: Array
    old_value: Array
    outgoing_carry: Array
    reward: int
    transition_done: bool
    next_observation: Array


@dataclass(frozen=True, slots=True)
class PPOGRURolloutTrace:
    """One contiguous behavior rollout and its terminal bootstrap evaluation."""

    rollout_index: int
    initial_carry: Array
    initial_reset: bool
    steps: tuple[PPOGRURolloutStep, ...]
    bootstrap_observation: Array
    bootstrap_carry: Array
    bootstrap_reset: bool
    bootstrap_value: Array


@dataclass(frozen=True, slots=True)
class PPOGRURunnerSegment:
    """One update payload with runner-visible segment identity and time indices."""

    segment_id: int
    time_indices: tuple[int, ...]
    payload: Any


@dataclass(frozen=True, slots=True)
class PPOGRUTrainingHandle:
    """Opaque model/state pair owned by one dependency implementation."""

    model: Any
    state: Any


_InitializeBridge = Callable[[int], Any]
_StepBridge = Callable[[Any, int], Any]
_ParseRuntimeIdentity = Callable[[bytes], Mapping[str, Any]]
_InitializeTraining = Callable[
    [ppo_gru.PPOGRUConfig, ppo_gru.PPOGRURNGState],
    tuple[PPOGRUTrainingHandle, ppo_gru.PPOGRURNGState],
]
_EvaluateStep = Callable[
    [PPOGRUTrainingHandle, Array, Array, bool], PPOGRUStepEvaluation
]
_ValidateRollout = Callable[
    [PPOGRUTrainingHandle, PPOGRURolloutTrace, ppo_gru.PPOGRUConfig], Any
]
_BuildSegments = Callable[
    [Any, tuple[int, ...], PPOGRURunnerGeometry], tuple[PPOGRURunnerSegment, ...]
]
_UpdateSegment = Callable[
    [PPOGRUTrainingHandle, Any, int, float], PPOGRUTrainingHandle
]
_OptimizerUpdateCount = Callable[[PPOGRUTrainingHandle], int]


@dataclass(frozen=True, slots=True)
class PPOGRURunnerDependencies:
    """Dependency seams; synthetic implementations are engineering-only."""

    classification: str
    initialize_bridge: _InitializeBridge
    step_bridge: _StepBridge
    parse_runtime_identity: _ParseRuntimeIdentity
    initialize_training: _InitializeTraining
    evaluate_step: _EvaluateStep
    validate_rollout: _ValidateRollout
    build_segments: _BuildSegments
    update_segment: _UpdateSegment
    optimizer_update_count: _OptimizerUpdateCount

    def __post_init__(self) -> None:
        if self.classification not in {
            "production_adapter_runtime_unqualified",
            "synthetic_engineering_only",
        }:
            raise ForagerMatchedV3PPOGRURunnerError(
                "dependency classification is invalid"
            )
        for name in (
            "initialize_bridge",
            "step_bridge",
            "parse_runtime_identity",
            "initialize_training",
            "evaluate_step",
            "validate_rollout",
            "build_segments",
            "update_segment",
            "optimizer_update_count",
        ):
            if not callable(getattr(self, name)):
                raise ForagerMatchedV3PPOGRURunnerError(
                    f"dependency {name} must be callable"
                )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PPOGRURunnerOutcome:
    """Closed in-memory accounting from one complete driver invocation."""

    classification: str
    geometry: PPOGRURunnerGeometry
    environment_seed: int
    agent_seed: int
    runtime_identity_bytes: bytes
    runtime_identity_sha256: str
    environment_interactions: int
    rollout_count: int
    optimizer_update_count: int
    parameter_initialization_draw_count: int
    action_draw_count: int
    permutation_draw_count: int
    total_agent_draw_count: int
    ppo_environment_draw_count: int
    bridge_reset_count: int
    bridge_step_count: int
    bridge_environment_key_use_count: int
    raw_reward_trace: bytes
    raw_reward_trace_sha256: str
    raw_cumulative_score: int
    trace_chain_sha256: str
    production_horizon_complete: bool
    _production_capability: _ProductionOutcomeCapability | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PPOGRUProductionRuntime:
    """One validated reusable bridge runtime and its exact runner dependencies."""

    bridge_runtime: bridge.MatchedV3ForagaxRuntime
    runtime_identity_bytes: bytes
    dependencies: PPOGRURunnerDependencies
    _capability: _ProductionRuntimeCapability = field(repr=False, compare=False)


_DEPENDENCY_CALLABLE_FIELDS: Final = (
    "initialize_bridge",
    "step_bridge",
    "parse_runtime_identity",
    "initialize_training",
    "evaluate_step",
    "validate_rollout",
    "build_segments",
    "update_segment",
    "optimizer_update_count",
)


@dataclass(frozen=True, slots=True)
class _ProductionRuntimeBinding:
    runtime_ref: weakref.ReferenceType[PPOGRUProductionRuntime]
    bridge_runtime: bridge.MatchedV3ForagaxRuntime
    runtime_identity_bytes: bytes
    dependencies: PPOGRURunnerDependencies
    dependency_callables: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _ProductionOutcomeBinding:
    outcome_ref: weakref.ReferenceType[PPOGRURunnerOutcome]
    runtime_capability: _ProductionRuntimeCapability
    outcome_sha256: str


_PRODUCTION_REGISTRY_LOCK: Final = threading.RLock()
_PRODUCTION_RUNTIME_REGISTRY: Final = weakref.WeakKeyDictionary[
    _ProductionRuntimeCapability, _ProductionRuntimeBinding
]()
_PRODUCTION_OUTCOME_REGISTRY: Final = weakref.WeakKeyDictionary[
    _ProductionOutcomeCapability, _ProductionOutcomeBinding
]()


def _dependency_callables(
    dependencies: PPOGRURunnerDependencies,
) -> tuple[object, ...]:
    return tuple(getattr(dependencies, name) for name in _DEPENDENCY_CALLABLE_FIELDS)


def _host_array(value: object, label: str) -> np.ndarray[Any, Any]:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must be host-validatable"
        ) from exc


def _validate_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray[Any, Any]:
    host = _host_array(value, label)
    if host.shape != shape or host.dtype != dtype:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must have shape {shape!r} and dtype {dtype}"
        )
    if np.issubdtype(dtype, np.floating) and not bool(np.all(np.isfinite(host))):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} must be finite")
    return host


def _validate_observation(value: object, label: str) -> np.ndarray[Any, Any]:
    host = _validate_array(
        value,
        shape=(9, 9, 3),
        dtype=np.dtype(np.float32),
        label=label,
    )
    if not bool(np.all((host == 0.0) | (host == 1.0))):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must contain exact zero/one channels"
        )
    sums = np.sum(host, axis=-1)
    if not bool(np.all((sums == 0.0) | (sums == 1.0))):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must be zero-or-one-hot per cell"
        )
    return host


def _validate_carry(value: object, hidden_size: int, label: str) -> np.ndarray[Any, Any]:
    return _validate_array(
        value,
        shape=(hidden_size,),
        dtype=np.dtype(np.float32),
        label=label,
    )


def _validate_scalar_float32(value: object, label: str) -> np.ndarray[Any, Any]:
    return _validate_array(
        value,
        shape=(),
        dtype=np.dtype(np.float32),
        label=label,
    )


def _exact_array_equal(actual: object, expected: object, label: str) -> None:
    actual_host = _host_array(actual, label)
    expected_host = _host_array(expected, f"expected {label}")
    if actual_host.shape != expected_host.shape or not np.array_equal(
        actual_host, expected_host
    ):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} differs from its bound value")


def _validate_action_key(value: object, label: str) -> Array:
    try:
        implementation = str(jr.key_impl(cast(Array, value)))
        data = np.asarray(jr.key_data(cast(Array, value)))
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} is not a typed JAX key") from exc
    if implementation != ppo_gru.PPO_GRU_PRNG_IMPLEMENTATION:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must use {ppo_gru.PPO_GRU_PRNG_IMPLEMENTATION}"
        )
    if data.shape != (2,) or data.dtype != np.dtype(np.uint32):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} key data is invalid")
    return cast(Array, value)


def _exact_key_equal(actual: Array, expected: Array, label: str) -> None:
    actual_key = _validate_action_key(actual, label)
    expected_key = _validate_action_key(expected, f"expected {label}")
    if not np.array_equal(
        np.asarray(jr.key_data(actual_key)), np.asarray(jr.key_data(expected_key))
    ):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} differs from its bound value")


def validate_ppo_gru_runner_rollout_trace(
    trace: PPOGRURolloutTrace,
    geometry: PPOGRURunnerGeometry,
    *,
    hidden_size: int,
    expected_rollout_index: int,
    expected_initial_carry: Array,
    expected_initial_observation: Array,
    expected_action_keys: Sequence[Array],
    expected_bootstrap_value: Array,
) -> None:
    """Independently replay action, log-probability, carry, reset, and link bindings."""

    if type(trace) is not PPOGRURolloutTrace:
        raise ForagerMatchedV3PPOGRURunnerError(
            "rollout trace must be an exact PPOGRURolloutTrace"
        )
    if trace.rollout_index != expected_rollout_index:
        raise ForagerMatchedV3PPOGRURunnerError("rollout index is stale or out of order")
    if trace.initial_reset is not False:
        raise ForagerMatchedV3PPOGRURunnerError(
            "the continuing trajectory cannot reset at a rollout boundary"
        )
    if len(trace.steps) != geometry.rollout_steps:
        raise ForagerMatchedV3PPOGRURunnerError(
            "rollout trace length differs from the driver geometry"
        )
    if len(expected_action_keys) != geometry.rollout_steps:
        raise ForagerMatchedV3PPOGRURunnerError(
            "expected action-key count differs from the rollout length"
        )
    _validate_carry(trace.initial_carry, hidden_size, "trace initial carry")
    _exact_array_equal(
        trace.initial_carry, expected_initial_carry, "trace initial carry"
    )
    _validate_observation(expected_initial_observation, "expected initial observation")
    current_observation: object = expected_initial_observation
    current_carry: object = expected_initial_carry
    previous_done = False
    for position, step in enumerate(trace.steps):
        if type(step) is not PPOGRURolloutStep:
            raise ForagerMatchedV3PPOGRURunnerError(
                "trace steps must be exact PPOGRURolloutStep values"
            )
        if step.step_index != position:
            raise ForagerMatchedV3PPOGRURunnerError(
                "trace step indices must be zero-based and contiguous"
            )
        _validate_observation(step.observation, f"step {position} observation")
        _exact_array_equal(
            step.observation, current_observation, f"step {position} observation"
        )
        _validate_carry(step.incoming_carry, hidden_size, f"step {position} incoming carry")
        _exact_array_equal(
            step.incoming_carry, current_carry, f"step {position} incoming carry"
        )
        if type(step.reset_before) is not bool or step.reset_before is not previous_done:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"step {position} reset boundary is invalid"
        )
        key = _validate_action_key(step.action_key, f"step {position} action key")
        _exact_key_equal(key, expected_action_keys[position], f"step {position} action key")
        _validate_array(
            step.logits,
            shape=(4,),
            dtype=np.dtype(np.float32),
            label=f"step {position} logits",
        )
        if type(step.action) is not int or not 0 <= step.action < 4:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"step {position} action must be an exact integer in 0..3"
            )
        replayed_action = int(ppo_gru.sample_categorical_action(key, step.logits))
        if step.action != replayed_action:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"step {position} action differs from its sampled agent key"
            )
        _validate_scalar_float32(step.old_log_prob, f"step {position} old log probability")
        replayed_log_prob = ppo_gru.categorical_log_prob(
            step.logits, jnp.asarray(step.action, dtype=jnp.int32)
        )
        _exact_array_equal(
            step.old_log_prob,
            replayed_log_prob,
            f"step {position} old log probability",
        )
        _validate_scalar_float32(step.old_value, f"step {position} old value")
        _validate_carry(step.outgoing_carry, hidden_size, f"step {position} outgoing carry")
        if type(step.reward) is not int or step.reward not in _RAW_REWARD_SUPPORT:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"step {position} reward is outside exact raw Forager support"
            )
        if type(step.transition_done) is not bool or step.transition_done:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"step {position} must remain a continuing nonterminal transition"
            )
        _validate_observation(step.next_observation, f"step {position} next observation")
        current_observation = step.next_observation
        current_carry = step.outgoing_carry
        previous_done = step.transition_done

    _validate_observation(trace.bootstrap_observation, "bootstrap observation")
    _exact_array_equal(
        trace.bootstrap_observation, current_observation, "bootstrap observation"
    )
    _validate_carry(trace.bootstrap_carry, hidden_size, "bootstrap carry")
    _exact_array_equal(trace.bootstrap_carry, current_carry, "bootstrap carry")
    if type(trace.bootstrap_reset) is not bool or trace.bootstrap_reset is not previous_done:
        raise ForagerMatchedV3PPOGRURunnerError("bootstrap reset boundary is invalid")
    _validate_scalar_float32(trace.bootstrap_value, "bootstrap value")
    _validate_scalar_float32(expected_bootstrap_value, "expected bootstrap value")
    _exact_array_equal(
        trace.bootstrap_value, expected_bootstrap_value, "bootstrap value"
    )


def _trace_sha256(trace: PPOGRURolloutTrace) -> str:
    digest = hashlib.sha256()
    digest.update(b"alberta.forager_matched_v3.ppo_gru.rollout_trace.v1\x00")

    def add_int(label: str, value: int) -> None:
        encoded = f"{label}:{value};".encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)

    def add_array(label: str, value: object, dtype: str) -> None:
        array = np.asarray(value).astype(np.dtype(dtype), copy=False)
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "big"))
        digest.update(encoded_label)
        digest.update(array.ndim.to_bytes(1, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    add_int("rollout", trace.rollout_index)
    add_array("initial_carry", trace.initial_carry, "<f4")
    for step in trace.steps:
        add_int("step", step.step_index)
        add_array("observation", step.observation, "<f4")
        add_array("incoming_carry", step.incoming_carry, "<f4")
        add_array("action_key", jr.key_data(step.action_key), "<u4")
        add_array("logits", step.logits, "<f4")
        add_int("action", step.action)
        add_array("old_log_prob", step.old_log_prob, "<f4")
        add_array("old_value", step.old_value, "<f4")
        add_array("outgoing_carry", step.outgoing_carry, "<f4")
        add_int("reward", step.reward)
        add_array("next_observation", step.next_observation, "<f4")
    add_array("bootstrap_observation", trace.bootstrap_observation, "<f4")
    add_array("bootstrap_carry", trace.bootstrap_carry, "<f4")
    add_array("bootstrap_value", trace.bootstrap_value, "<f4")
    return digest.hexdigest()


def _strict_runtime_identity(
    raw: bytes, parser: _ParseRuntimeIdentity
) -> tuple[bytes, str]:
    identity = _strict_json_object(raw, "runtime identity")
    if type(identity.get("schema_version")) is not str or not identity["schema_version"]:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runtime identity must carry a nonempty schema_version"
        )
    _reject_authority_anywhere(identity, "runtime identity")
    try:
        validated = parser(raw)
    except ForagerMatchedV3PPOGRURunnerError:
        raise
    except Exception as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runtime identity dependency validation failed"
        ) from exc
    if type(validated) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runtime identity parser must return a plain dictionary"
        )
    validated_raw = _canonical_json(validated, "validated runtime identity")
    if not hmac.compare_digest(raw, validated_raw):
        raise ForagerMatchedV3PPOGRURunnerError(
            "runtime identity parser changed or detached from the supplied bytes"
        )
    return bytes(raw), hashlib.sha256(raw).hexdigest()


def _bridge_state_accounting(
    state: object, *, environment_seed: int, expected_steps: int
) -> tuple[int, int, int]:
    try:
        seed = getattr(state, "environment_seed")
        reset_count = getattr(state, "reset_count")
        step_count = getattr(state, "step_count")
        key_uses = getattr(state, "environment_key_use_count")
    except (AttributeError, TypeError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge state does not expose exact runner accounting"
        ) from exc
    if seed != environment_seed or type(seed) is not int:
        raise ForagerMatchedV3PPOGRURunnerError("bridge environment seed drifted")
    if type(reset_count) is not int or reset_count != 1:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge must perform exactly one trajectory reset"
        )
    if type(step_count) is not int or step_count != expected_steps:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge step counter is stale or off by one"
        )
    if type(key_uses) is not int or key_uses != 1 + expected_steps:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge environment-key use counter is stale or off by one"
        )
    return reset_count, step_count, key_uses


def _bridge_observation(state: object, label: str) -> Array:
    try:
        observation = getattr(state, "observation")
    except AttributeError as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} does not expose an observation"
        ) from exc
    _validate_observation(observation, f"{label} observation")
    return cast(Array, observation)


def _validate_transition(
    transition: object,
    *,
    prior_state: object,
    expected_action: int,
    environment_seed: int,
    expected_steps: int,
) -> tuple[object, int, Array]:
    try:
        next_state = getattr(transition, "state")
        action = getattr(transition, "action")
        reward = getattr(transition, "reward")
        done = getattr(transition, "done")
        truncated = getattr(transition, "truncated")
        info_validated = getattr(transition, "info_validated")
    except (AttributeError, TypeError) as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge transition does not expose the exact adapter surface"
        ) from exc
    if next_state is prior_state:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge returned a stale/reused state instead of a linear successor"
        )
    if type(action) is not int or action != expected_action:
        raise ForagerMatchedV3PPOGRURunnerError("bridge transition action drifted")
    if type(reward) is not int or reward not in _RAW_REWARD_SUPPORT:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge transition reward is outside exact raw support"
        )
    if type(done) is not bool or done:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge transition violated the continuing trajectory"
        )
    if type(truncated) is not bool or truncated:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge transition cannot truncate the exact trajectory"
        )
    if type(info_validated) is not bool or not info_validated:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge transition must attest internal info validation"
        )
    _bridge_state_accounting(
        next_state,
        environment_seed=environment_seed,
        expected_steps=expected_steps,
    )
    observation = _bridge_observation(next_state, "next bridge state")
    return next_state, reward, observation


def _rng_snapshot(state: ppo_gru.PPOGRURNGState) -> tuple[int, int, np.ndarray[Any, Any]]:
    environment_count, agent_count = ppo_gru.validate_ppo_gru_rng_state(state)
    environment_key_data = np.asarray(jr.key_data(state.environment_key)).copy()
    return environment_count, agent_count, environment_key_data


def _require_rng_delta(
    before: ppo_gru.PPOGRURNGState,
    after: ppo_gru.PPOGRURNGState,
    *,
    expected_agent_delta: int,
    label: str,
) -> None:
    before_environment, before_agent, before_environment_key = _rng_snapshot(before)
    after_environment, after_agent, after_environment_key = _rng_snapshot(after)
    if before_environment != 0 or after_environment != 0:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} consumed the PPO environment key owned by the bridge"
        )
    if not np.array_equal(before_environment_key, after_environment_key):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} changed the PPO environment key owned by the bridge"
        )
    if after_agent != before_agent + expected_agent_delta:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} agent draw count is off by one or double-consumed"
        )


def _validate_step_evaluation(
    evaluation: PPOGRUStepEvaluation,
    *,
    hidden_size: int,
    label: str,
) -> None:
    if type(evaluation) is not PPOGRUStepEvaluation:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must return an exact PPOGRUStepEvaluation"
        )
    _validate_carry(evaluation.outgoing_carry, hidden_size, f"{label} outgoing carry")
    _validate_array(
        evaluation.logits,
        shape=(4,),
        dtype=np.dtype(np.float32),
        label=f"{label} logits",
    )
    _validate_scalar_float32(evaluation.value, f"{label} value")


def _validate_segment_schedule(
    segments: object,
    order: tuple[int, ...],
    geometry: PPOGRURunnerGeometry,
) -> tuple[PPOGRURunnerSegment, ...]:
    if type(segments) is not tuple or len(segments) != geometry.segments_per_rollout:
        raise ForagerMatchedV3PPOGRURunnerError(
            "epoch must return exactly four segment update payloads"
        )
    exact = cast(tuple[PPOGRURunnerSegment, ...], segments)
    for position, segment in enumerate(exact):
        if type(segment) is not PPOGRURunnerSegment:
            raise ForagerMatchedV3PPOGRURunnerError(
                "epoch segments must be exact PPOGRURunnerSegment values"
            )
        expected_id = order[position]
        if type(segment.segment_id) is not int or segment.segment_id != expected_id:
            raise ForagerMatchedV3PPOGRURunnerError(
                "segment payload order differs from the agent permutation"
            )
        expected_indices = tuple(
            range(
                expected_id * geometry.segment_steps,
                (expected_id + 1) * geometry.segment_steps,
            )
        )
        if segment.time_indices != expected_indices:
            raise ForagerMatchedV3PPOGRURunnerError(
                "segment timesteps must remain contiguous and strictly increasing"
            )
    return exact


def _linear_learning_rate(
    update_index: int,
    geometry: PPOGRURunnerGeometry,
    base_learning_rate: float,
) -> float:
    rollout_index = update_index // geometry.optimizer_updates_per_rollout
    return base_learning_rate * max(1.0 - rollout_index / geometry.rollout_count, 0.0)


def _validate_optimizer_counter(
    dependencies: PPOGRURunnerDependencies,
    training: PPOGRUTrainingHandle,
    expected: int,
    label: str,
) -> None:
    try:
        actual = dependencies.optimizer_update_count(training)
    except Exception as exc:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} optimizer counter could not be read"
        ) from exc
    if type(actual) is not int or actual != expected:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} optimizer update count is stale or off by one"
        )


def _validate_production_runtime_identity(identity: object, label: str) -> None:
    if type(identity) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} must be an exact production runtime identity object"
        )
    exact = cast(dict[str, Any], identity)
    expected_keys = {
        "schema_version",
        "classification",
        "bridge_descriptor_sha256",
        "bridge_implementation_source_sha256",
        "runtime",
        "claims",
    }
    if set(exact) != expected_keys:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} production field membership drifted"
        )
    if exact["schema_version"] != PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} production schema drifted")
    if exact["classification"] != "observed_runtime_unqualified_non_authorizing":
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} production classification drifted"
        )
    dependencies = _frozen_dependency_binding()["foragax_bridge"]
    if exact["bridge_descriptor_sha256"] != dependencies["descriptor_sha256"]:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} bridge descriptor binding drifted"
        )
    if (
        exact["bridge_implementation_source_sha256"]
        != FORAGAX_BRIDGE_IMPLEMENTATION_SHA256
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} bridge source binding drifted"
        )
    if exact["claims"] != _non_authorizing_claims():
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} authority claims drifted")
    runtime = exact["runtime"]
    expected_runtime_keys = {
        "python_implementation",
        "python_version",
        "numpy_version",
        "flax_version",
        "optax_version",
        "jax_version",
        "jaxlib_version",
        "default_prng_impl",
        "threefry_partitionable",
        "jax_enable_x64",
        "backend",
        "foragax_version",
        "foragax_install_tree_sha256",
        "foragax_package_root",
        "runtime_qualified",
    }
    if type(runtime) is not dict or set(runtime) != expected_runtime_keys:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} production runtime field membership drifted"
        )
    runtime = cast(dict[str, Any], runtime)
    for name in (
        "python_implementation",
        "python_version",
        "numpy_version",
        "flax_version",
        "optax_version",
        "backend",
        "foragax_package_root",
    ):
        if type(runtime[name]) is not str or not runtime[name]:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"{label} production runtime {name} must be a nonempty string"
            )
    expected_runtime_values: dict[str, object] = {
        "jax_version": bridge.JAX_REQUIRED_VERSION,
        "jaxlib_version": bridge.JAXLIB_REQUIRED_VERSION,
        "default_prng_impl": ppo_gru.PPO_GRU_PRNG_IMPLEMENTATION,
        "threefry_partitionable": True,
        "jax_enable_x64": False,
        "foragax_version": bridge.FORAGAX_REQUIRED_VERSION,
        "foragax_install_tree_sha256": bridge.FORAGAX_INSTALL_TREE_SHA256,
        "runtime_qualified": False,
    }
    for name, expected in expected_runtime_values.items():
        if type(runtime[name]) is not type(expected) or runtime[name] != expected:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"{label} production runtime {name} drifted"
            )
    _reject_authority_anywhere(exact, label)


def _outcome_identity_sha256(outcome: PPOGRURunnerOutcome) -> str:
    body = {
        "classification": outcome.classification,
        "geometry": outcome.geometry.to_dict(),
        "environment_seed": outcome.environment_seed,
        "agent_seed": outcome.agent_seed,
        "runtime_identity_sha256": outcome.runtime_identity_sha256,
        "environment_interactions": outcome.environment_interactions,
        "rollout_count": outcome.rollout_count,
        "optimizer_update_count": outcome.optimizer_update_count,
        "parameter_initialization_draw_count": (
            outcome.parameter_initialization_draw_count
        ),
        "action_draw_count": outcome.action_draw_count,
        "permutation_draw_count": outcome.permutation_draw_count,
        "total_agent_draw_count": outcome.total_agent_draw_count,
        "ppo_environment_draw_count": outcome.ppo_environment_draw_count,
        "bridge_reset_count": outcome.bridge_reset_count,
        "bridge_step_count": outcome.bridge_step_count,
        "bridge_environment_key_use_count": outcome.bridge_environment_key_use_count,
        "raw_reward_trace_encoding": _RAW_REWARD_TRACE_ENCODING,
        "raw_reward_trace_length": len(outcome.raw_reward_trace),
        "raw_reward_trace_sha256": outcome.raw_reward_trace_sha256,
        "score_reduction": _RAW_SCORE_REDUCTION,
        "score_scaling": _RAW_SCORE_SCALING,
        "raw_cumulative_score": outcome.raw_cumulative_score,
        "trace_chain_sha256": outcome.trace_chain_sha256,
        "production_horizon_complete": outcome.production_horizon_complete,
    }
    metadata = _canonical_json(body, "PPO-GRU outcome identity")
    digest = hashlib.sha256()
    digest.update(_OUTCOME_IDENTITY_DOMAIN)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(outcome.raw_reward_trace).to_bytes(8, "big"))
    digest.update(outcome.raw_reward_trace)
    return digest.hexdigest()


def _validate_registered_production_outcome(outcome: PPOGRURunnerOutcome) -> None:
    capability = outcome._production_capability
    if type(capability) is not _ProductionOutcomeCapability:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production outcome lacks a registered completion capability"
        )
    with _PRODUCTION_REGISTRY_LOCK:
        binding = _PRODUCTION_OUTCOME_REGISTRY.get(capability)
        if binding is None or binding.outcome_ref() is not outcome:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production outcome is not the exact registered completed-run object"
            )
        runtime_binding = _PRODUCTION_RUNTIME_REGISTRY.get(binding.runtime_capability)
        if runtime_binding is None:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production outcome runtime capability is no longer registered"
            )
        expected_sha256 = binding.outcome_sha256
    if not hmac.compare_digest(_outcome_identity_sha256(outcome), expected_sha256):
        raise ForagerMatchedV3PPOGRURunnerError(
            "registered production outcome identity drifted"
        )


def _validate_outcome(
    outcome: PPOGRURunnerOutcome,
    *,
    require_registered_production: bool = True,
) -> None:
    if type(outcome) is not PPOGRURunnerOutcome:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runner outcome must be an exact PPOGRURunnerOutcome"
        )
    geometry = outcome.geometry
    if type(geometry) is not PPOGRURunnerGeometry:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runner outcome geometry must be an exact PPOGRURunnerGeometry"
        )
    if outcome.classification not in {
        "synthetic_engineering_complete",
        "production_runtime_unqualified_complete",
    }:
        raise ForagerMatchedV3PPOGRURunnerError("runner outcome classification is invalid")
    ppo_gru.validate_ppo_gru_seed_pair(outcome.environment_seed, outcome.agent_seed)
    _require_sha256(outcome.runtime_identity_sha256, "runtime_identity_sha256")
    if not hmac.compare_digest(
        hashlib.sha256(outcome.runtime_identity_bytes).hexdigest(),
        outcome.runtime_identity_sha256,
    ):
        raise ForagerMatchedV3PPOGRURunnerError("runtime identity digest drifted")
    runtime_identity = _strict_json_object(
        outcome.runtime_identity_bytes, "runtime identity"
    )
    _reject_authority_anywhere(runtime_identity, "runtime identity")
    expected = {
        "environment_interactions": geometry.horizon,
        "rollout_count": geometry.rollout_count,
        "optimizer_update_count": geometry.optimizer_update_count,
        "parameter_initialization_draw_count": 1,
        "action_draw_count": geometry.action_draw_count,
        "permutation_draw_count": geometry.permutation_draw_count,
        "total_agent_draw_count": geometry.total_agent_draw_count,
        "ppo_environment_draw_count": 0,
        "bridge_reset_count": 1,
        "bridge_step_count": geometry.horizon,
        "bridge_environment_key_use_count": 1 + geometry.horizon,
    }
    for name, value in expected.items():
        if getattr(outcome, name) != value or type(getattr(outcome, name)) is not int:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"runner outcome {name} differs from exact accounting"
            )
    if type(outcome.raw_reward_trace) is not bytes:
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw reward trace must be immutable exact bytes"
        )
    if len(outcome.raw_reward_trace) != geometry.horizon:
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw reward trace length differs from the exact interaction horizon"
        )
    reward_trace_sha256 = _require_sha256(
        outcome.raw_reward_trace_sha256,
        "raw_reward_trace_sha256",
    )
    if not hmac.compare_digest(
        hashlib.sha256(outcome.raw_reward_trace).hexdigest(),
        reward_trace_sha256,
    ):
        raise ForagerMatchedV3PPOGRURunnerError("raw reward trace digest drifted")
    rewards = np.frombuffer(outcome.raw_reward_trace, dtype=np.int8)
    if not bool(np.all(np.isin(rewards, tuple(sorted(_RAW_REWARD_SUPPORT))))):
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw reward trace contains a value outside exact Forager support"
        )
    if type(outcome.raw_cumulative_score) is not int:
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw cumulative score must remain an exact integer"
        )
    exact_score = int(np.sum(rewards, dtype=np.int64))
    if outcome.raw_cumulative_score != exact_score:
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw cumulative score differs from the exact int64 reward-trace sum"
        )
    if not -geometry.horizon <= outcome.raw_cumulative_score <= 30 * geometry.horizon:
        raise ForagerMatchedV3PPOGRURunnerError(
            "raw cumulative score is outside the exact reward-support bounds"
        )
    _require_sha256(outcome.trace_chain_sha256, "trace_chain_sha256")
    is_production = geometry == MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY
    if outcome.production_horizon_complete is not is_production:
        raise ForagerMatchedV3PPOGRURunnerError(
            "partial horizon is mislabeled as a complete production horizon"
        )
    expected_classification = (
        "production_runtime_unqualified_complete"
        if is_production
        else "synthetic_engineering_complete"
    )
    if outcome.classification != expected_classification:
        raise ForagerMatchedV3PPOGRURunnerError(
            "outcome classification differs from its geometry"
        )
    if is_production:
        _validate_production_runtime_identity(runtime_identity, "runtime identity")
        if require_registered_production:
            _validate_registered_production_outcome(outcome)
        elif type(outcome._production_capability) is not _ProductionOutcomeCapability:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production outcome lacks an exact completion capability"
            )
    elif outcome._production_capability is not None:
        raise ForagerMatchedV3PPOGRURunnerError(
            "engineering outcome cannot carry a production completion capability"
        )


def _register_completed_production_outcome(
    outcome: PPOGRURunnerOutcome,
    *,
    runtime_capability: _ProductionRuntimeCapability,
) -> None:
    _validate_outcome(outcome, require_registered_production=False)
    capability = outcome._production_capability
    if type(capability) is not _ProductionOutcomeCapability:
        raise ForagerMatchedV3PPOGRURunnerError(
            "completed production outcome capability type drifted"
        )
    with _PRODUCTION_REGISTRY_LOCK:
        runtime_binding = _PRODUCTION_RUNTIME_REGISTRY.get(runtime_capability)
        if runtime_binding is None or runtime_binding.runtime_ref() is None:
            raise ForagerMatchedV3PPOGRURunnerError(
                "completed production outcome lacks a live registered runtime"
            )
        if capability in _PRODUCTION_OUTCOME_REGISTRY:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production outcome capability is already registered"
            )
        _PRODUCTION_OUTCOME_REGISTRY[capability] = _ProductionOutcomeBinding(
            outcome_ref=weakref.ref(outcome),
            runtime_capability=runtime_capability,
            outcome_sha256=_outcome_identity_sha256(outcome),
        )


def _run_driver(
    *,
    environment_seed: object,
    agent_seed: object,
    runtime_identity_bytes: bytes,
    geometry: PPOGRURunnerGeometry,
    dependencies: PPOGRURunnerDependencies,
    production_runtime: PPOGRUProductionRuntime | None,
) -> PPOGRURunnerOutcome:
    is_production = geometry == MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY
    production_binding: _ProductionRuntimeBinding | None = None
    if is_production:
        if production_runtime is None:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production geometry requires a registered production runtime"
            )
        production_binding = _validated_production_runtime_binding(production_runtime)
        if dependencies is not production_binding.dependencies:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production dependencies differ from their runtime registry binding"
            )
        if not hmac.compare_digest(
            runtime_identity_bytes, production_binding.runtime_identity_bytes
        ):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime identity differs from its registry binding"
            )
    else:
        if production_runtime is not None:
            raise ForagerMatchedV3PPOGRURunnerError(
                "engineering geometry cannot carry a production runtime"
            )
        if dependencies.classification != "synthetic_engineering_only":
            raise ForagerMatchedV3PPOGRURunnerError(
                "engineering geometry requires explicitly synthetic dependencies"
            )
    seeds = ppo_gru.validate_ppo_gru_seed_pair(environment_seed, agent_seed)
    runtime_raw, runtime_sha256 = _strict_runtime_identity(
        runtime_identity_bytes, dependencies.parse_runtime_identity
    )
    config = ppo_gru.matched_v3_ppo_gru_configuration()
    rng_state = ppo_gru.initialize_ppo_gru_rng_state(
        seeds.environment_seed, seeds.agent_seed
    )
    initial_rng_state = rng_state
    try:
        training, initialized_rng_state = dependencies.initialize_training(config, rng_state)
    except Exception as exc:
        raise ForagerMatchedV3PPOGRURunnerError("training initialization failed") from exc
    if type(training) is not PPOGRUTrainingHandle:
        raise ForagerMatchedV3PPOGRURunnerError(
            "training initialization must return PPOGRUTrainingHandle"
        )
    if type(initialized_rng_state) is not ppo_gru.PPOGRURNGState:
        raise ForagerMatchedV3PPOGRURunnerError(
            "training initialization must return PPOGRURNGState"
        )
    _require_rng_delta(
        rng_state,
        initialized_rng_state,
        expected_agent_delta=1,
        label="parameter initialization",
    )
    rng_state = initialized_rng_state
    _validate_optimizer_counter(dependencies, training, 0, "initial")

    try:
        bridge_state = dependencies.initialize_bridge(seeds.environment_seed)
    except Exception as exc:
        raise ForagerMatchedV3PPOGRURunnerError("bridge initialization failed") from exc
    _bridge_state_accounting(
        bridge_state, environment_seed=seeds.environment_seed, expected_steps=0
    )
    observation = _bridge_observation(bridge_state, "initial bridge state")
    carry = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
    reset_before = False
    raw_score = 0
    raw_rewards = bytearray()
    global_interactions = 0
    expected_optimizer_updates = 0
    trace_chain = hashlib.sha256(_TRACE_CHAIN_DOMAIN).digest()

    for rollout_index in range(geometry.rollout_count):
        rollout_initial_carry = carry
        rollout_initial_observation = observation
        steps: list[PPOGRURolloutStep] = []
        action_keys: list[Array] = []
        for step_index in range(geometry.rollout_steps):
            try:
                evaluation = dependencies.evaluate_step(
                    training, carry, observation, reset_before
                )
            except Exception as exc:
                raise ForagerMatchedV3PPOGRURunnerError(
                    "policy evaluation failed without retry"
                ) from exc
            _validate_step_evaluation(
                evaluation,
                hidden_size=config.hidden_size,
                label=f"rollout {rollout_index} step {step_index}",
            )
            before_action_rng = rng_state
            rng_state, action_key = ppo_gru.next_ppo_gru_agent_key(rng_state)
            _require_rng_delta(
                before_action_rng,
                rng_state,
                expected_agent_delta=1,
                label="action sampling",
            )
            action_key = _validate_action_key(action_key, "action sampling key")
            action = int(ppo_gru.sample_categorical_action(action_key, evaluation.logits))
            old_log_prob = ppo_gru.categorical_log_prob(
                evaluation.logits, jnp.asarray(action, dtype=jnp.int32)
            )
            prior_state = bridge_state
            try:
                transition = dependencies.step_bridge(prior_state, action)
            except Exception as exc:
                raise ForagerMatchedV3PPOGRURunnerError(
                    "bridge step failed without retry"
                ) from exc
            global_interactions += 1
            bridge_state, reward, next_observation = _validate_transition(
                transition,
                prior_state=prior_state,
                expected_action=action,
                environment_seed=seeds.environment_seed,
                expected_steps=global_interactions,
            )
            steps.append(
                PPOGRURolloutStep(
                    step_index=step_index,
                    observation=observation,
                    incoming_carry=carry,
                    reset_before=reset_before,
                    action_key=action_key,
                    logits=evaluation.logits,
                    action=action,
                    old_log_prob=old_log_prob,
                    old_value=evaluation.value,
                    outgoing_carry=evaluation.outgoing_carry,
                    reward=reward,
                    transition_done=False,
                    next_observation=next_observation,
                )
            )
            action_keys.append(action_key)
            raw_score += reward
            raw_rewards.append(reward & 0xFF)
            carry = evaluation.outgoing_carry
            observation = next_observation
            reset_before = False

        try:
            bootstrap = dependencies.evaluate_step(
                training, carry, observation, reset_before
            )
        except Exception as exc:
            raise ForagerMatchedV3PPOGRURunnerError(
                "bootstrap evaluation failed without retry"
            ) from exc
        _validate_step_evaluation(
            bootstrap,
            hidden_size=config.hidden_size,
            label=f"rollout {rollout_index} bootstrap",
        )
        trace = PPOGRURolloutTrace(
            rollout_index=rollout_index,
            initial_carry=rollout_initial_carry,
            initial_reset=False,
            steps=tuple(steps),
            bootstrap_observation=observation,
            bootstrap_carry=carry,
            bootstrap_reset=False,
            bootstrap_value=bootstrap.value,
        )
        validate_ppo_gru_runner_rollout_trace(
            trace,
            geometry,
            hidden_size=config.hidden_size,
            expected_rollout_index=rollout_index,
            expected_initial_carry=rollout_initial_carry,
            expected_initial_observation=rollout_initial_observation,
            expected_action_keys=tuple(action_keys),
            expected_bootstrap_value=bootstrap.value,
        )
        try:
            validated_rollout = dependencies.validate_rollout(training, trace, config)
        except Exception as exc:
            raise ForagerMatchedV3PPOGRURunnerError(
                "strict core rollout validation failed"
            ) from exc
        rollout_digest = bytes.fromhex(_trace_sha256(trace))
        trace_chain = hashlib.sha256(trace_chain + rollout_digest).digest()

        for _epoch_index in range(geometry.update_epochs):
            before_permutation_rng = rng_state
            rng_state, order_array = ppo_gru.next_ppo_gru_segment_order(rng_state)
            _require_rng_delta(
                before_permutation_rng,
                rng_state,
                expected_agent_delta=1,
                label="epoch segment permutation",
            )
            try:
                order = tuple(int(value) for value in np.asarray(order_array).tolist())
            except (TypeError, ValueError, OverflowError) as exc:
                raise ForagerMatchedV3PPOGRURunnerError(
                    "segment permutation is not host-validatable"
                ) from exc
            if len(order) != 4 or sorted(order) != [0, 1, 2, 3]:
                raise ForagerMatchedV3PPOGRURunnerError(
                    "agent permutation must contain every segment exactly once"
                )
            try:
                built_segments = dependencies.build_segments(
                    validated_rollout, order, geometry
                )
            except Exception as exc:
                raise ForagerMatchedV3PPOGRURunnerError(
                    "segment construction failed"
                ) from exc
            segments = _validate_segment_schedule(built_segments, order, geometry)
            for segment in segments:
                _validate_optimizer_counter(
                    dependencies,
                    training,
                    expected_optimizer_updates,
                    "pre-update",
                )
                learning_rate = _linear_learning_rate(
                    expected_optimizer_updates,
                    geometry,
                    config.learning_rate,
                )
                try:
                    next_training = dependencies.update_segment(
                        training,
                        segment.payload,
                        expected_optimizer_updates,
                        learning_rate,
                    )
                except Exception as exc:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        "optimizer transaction failed without retry"
                    ) from exc
                if type(next_training) is not PPOGRUTrainingHandle:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        "optimizer transaction must return PPOGRUTrainingHandle"
                    )
                if next_training is training:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        "optimizer transaction returned a stale/reused training handle"
                    )
                if next_training.model is not training.model:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        "optimizer transaction replaced the bound model"
                    )
                if next_training.state is training.state:
                    raise ForagerMatchedV3PPOGRURunnerError(
                        "optimizer transaction reused the stale train state"
                    )
                expected_optimizer_updates += 1
                _validate_optimizer_counter(
                    dependencies,
                    next_training,
                    expected_optimizer_updates,
                    "post-update",
                )
                training = next_training

    _validate_optimizer_counter(
        dependencies,
        training,
        geometry.optimizer_update_count,
        "final",
    )
    final_environment_draws, final_agent_draws, final_environment_key = _rng_snapshot(
        rng_state
    )
    initial_environment_draws, _, initial_environment_key = _rng_snapshot(
        initial_rng_state
    )
    if initial_environment_draws != 0 or final_environment_draws != 0:
        raise ForagerMatchedV3PPOGRURunnerError(
            "runner consumed the PPO environment chain owned by the bridge"
        )
    if not np.array_equal(initial_environment_key, final_environment_key):
        raise ForagerMatchedV3PPOGRURunnerError(
            "runner changed the PPO environment root owned by the bridge"
        )
    if final_agent_draws != geometry.total_agent_draw_count:
        raise ForagerMatchedV3PPOGRURunnerError(
            "final agent draw accounting is off by one"
        )
    reset_count, bridge_steps, bridge_key_uses = _bridge_state_accounting(
        bridge_state,
        environment_seed=seeds.environment_seed,
        expected_steps=geometry.horizon,
    )
    final_runtime_raw, final_runtime_sha256 = _strict_runtime_identity(
        runtime_identity_bytes, dependencies.parse_runtime_identity
    )
    if (
        not hmac.compare_digest(runtime_raw, final_runtime_raw)
        or not hmac.compare_digest(runtime_sha256, final_runtime_sha256)
    ):
        raise ForagerMatchedV3PPOGRURunnerError("runtime identity drifted during execution")

    if is_production:
        if production_runtime is None or production_binding is None:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime binding disappeared before completion"
            )
        final_binding = _validated_production_runtime_binding(production_runtime)
        if final_binding is not production_binding:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime registry binding changed during execution"
            )
    completion_capability = _ProductionOutcomeCapability() if is_production else None
    raw_reward_trace = bytes(raw_rewards)
    outcome = PPOGRURunnerOutcome(
        classification=(
            "production_runtime_unqualified_complete"
            if is_production
            else "synthetic_engineering_complete"
        ),
        geometry=geometry,
        environment_seed=seeds.environment_seed,
        agent_seed=seeds.agent_seed,
        runtime_identity_bytes=runtime_raw,
        runtime_identity_sha256=runtime_sha256,
        environment_interactions=global_interactions,
        rollout_count=geometry.rollout_count,
        optimizer_update_count=expected_optimizer_updates,
        parameter_initialization_draw_count=1,
        action_draw_count=geometry.action_draw_count,
        permutation_draw_count=geometry.permutation_draw_count,
        total_agent_draw_count=final_agent_draws,
        ppo_environment_draw_count=final_environment_draws,
        bridge_reset_count=reset_count,
        bridge_step_count=bridge_steps,
        bridge_environment_key_use_count=bridge_key_uses,
        raw_reward_trace=raw_reward_trace,
        raw_reward_trace_sha256=hashlib.sha256(raw_reward_trace).hexdigest(),
        raw_cumulative_score=raw_score,
        trace_chain_sha256=trace_chain.hex(),
        production_horizon_complete=is_production,
        _production_capability=completion_capability,
    )
    if is_production:
        assert production_runtime is not None
        _register_completed_production_outcome(
            outcome,
            runtime_capability=production_runtime._capability,
        )
    _validate_outcome(outcome)
    return outcome


@dataclass(frozen=True, slots=True)
class _CoreValidatedRollout:
    rollout: ppo_gru.PPOGRURollout
    advantages: Array
    targets: Array


def _bridge_runtime_identity_dict(
    identity: bridge.MatchedV3ForagaxRuntimeIdentity,
) -> dict[str, Any]:
    if type(identity) is not bridge.MatchedV3ForagaxRuntimeIdentity:
        raise ForagerMatchedV3PPOGRURunnerError(
            "bridge runtime identity type drifted"
        )
    return {
        "schema_version": PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "classification": "observed_runtime_unqualified_non_authorizing",
        "bridge_descriptor_sha256": _frozen_dependency_binding()["foragax_bridge"][
            "descriptor_sha256"
        ],
        "bridge_implementation_source_sha256": (
            FORAGAX_BRIDGE_IMPLEMENTATION_SHA256
        ),
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": importlib_metadata.version("numpy"),
            "flax_version": importlib_metadata.version("flax"),
            "optax_version": importlib_metadata.version("optax"),
            "jax_version": identity.jax_version,
            "jaxlib_version": identity.jaxlib_version,
            "default_prng_impl": identity.default_prng_impl,
            "threefry_partitionable": identity.threefry_partitionable,
            "jax_enable_x64": identity.jax_enable_x64,
            "backend": identity.backend,
            "foragax_version": identity.foragax_version,
            "foragax_install_tree_sha256": identity.foragax_install_tree_sha256,
            "foragax_package_root": identity.foragax_package_root,
            "runtime_qualified": identity.runtime_qualified,
        },
        "claims": _non_authorizing_claims(),
    }


def _production_runtime_identity_bytes(
    runtime: bridge.MatchedV3ForagaxRuntime,
) -> bytes:
    if type(runtime) is not bridge.MatchedV3ForagaxRuntime:
        raise ForagerMatchedV3PPOGRURunnerError("bridge runtime handle type drifted")
    return _canonical_json(
        _bridge_runtime_identity_dict(runtime.runtime_identity),
        "PPO-GRU production runtime identity",
    )


def _validate_exact_production_dependencies(
    runtime: PPOGRUProductionRuntime,
) -> None:
    dependencies = runtime.dependencies
    if type(dependencies) is not PPOGRURunnerDependencies:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production dependencies must be an exact PPOGRURunnerDependencies"
        )
    if dependencies.classification != "production_adapter_runtime_unqualified":
        raise ForagerMatchedV3PPOGRURunnerError(
            "registered runtime requires exact production dependencies"
        )
    initialize_bridge = dependencies.initialize_bridge
    if (
        getattr(initialize_bridge, "__self__", None) is not runtime.bridge_runtime
        or getattr(initialize_bridge, "__func__", None)
        is not bridge.MatchedV3ForagaxRuntime.initialize
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            "production bridge initializer callable drifted"
        )
    parser = dependencies.parse_runtime_identity
    if (
        type(parser) is not _ProductionRuntimeIdentityParser
        or not hmac.compare_digest(parser.expected_bytes, runtime.runtime_identity_bytes)
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            "production runtime identity parser callable drifted"
        )
    expected_functions = {
        "step_bridge": bridge.step_matched_v3_foragax_bridge,
        "initialize_training": _production_initialize_training,
        "evaluate_step": _production_evaluate_step,
        "validate_rollout": _production_validate_rollout,
        "build_segments": _production_build_segments,
        "update_segment": _production_update_segment,
        "optimizer_update_count": _production_optimizer_update_count,
    }
    for name, expected in expected_functions.items():
        if getattr(dependencies, name) is not expected:
            raise ForagerMatchedV3PPOGRURunnerError(
                f"production dependency callable {name} drifted"
            )


def _register_production_runtime(runtime: PPOGRUProductionRuntime) -> None:
    if type(runtime) is not PPOGRUProductionRuntime:
        raise ForagerMatchedV3PPOGRURunnerError(
            "only an exact PPOGRUProductionRuntime can be registered"
        )
    if type(runtime._capability) is not _ProductionRuntimeCapability:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production runtime capability type drifted"
        )
    _validate_exact_production_dependencies(runtime)
    expected_identity = _production_runtime_identity_bytes(runtime.bridge_runtime)
    if not hmac.compare_digest(runtime.runtime_identity_bytes, expected_identity):
        raise ForagerMatchedV3PPOGRURunnerError(
            "production runtime identity differs before registration"
        )
    binding = _ProductionRuntimeBinding(
        runtime_ref=weakref.ref(runtime),
        bridge_runtime=runtime.bridge_runtime,
        runtime_identity_bytes=bytes(runtime.runtime_identity_bytes),
        dependencies=runtime.dependencies,
        dependency_callables=_dependency_callables(runtime.dependencies),
    )
    with _PRODUCTION_REGISTRY_LOCK:
        if runtime._capability in _PRODUCTION_RUNTIME_REGISTRY:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime capability is already registered"
            )
        _PRODUCTION_RUNTIME_REGISTRY[runtime._capability] = binding


def _validated_production_runtime_binding(
    runtime: PPOGRUProductionRuntime,
) -> _ProductionRuntimeBinding:
    if type(runtime) is not PPOGRUProductionRuntime:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production runtime must be PPOGRUProductionRuntime"
        )
    capability = runtime._capability
    if type(capability) is not _ProductionRuntimeCapability:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production runtime lacks an exact process-local capability"
        )
    with _PRODUCTION_REGISTRY_LOCK:
        binding = _PRODUCTION_RUNTIME_REGISTRY.get(capability)
        if binding is None or binding.runtime_ref() is not runtime:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime is not the exact registered runtime object"
            )
        if (
            runtime.bridge_runtime is not binding.bridge_runtime
            or runtime.dependencies is not binding.dependencies
        ):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime object binding drifted"
            )
        if runtime.dependencies.classification != "production_adapter_runtime_unqualified":
            raise ForagerMatchedV3PPOGRURunnerError(
                "production dependency classification drifted"
            )
        _validate_exact_production_dependencies(runtime)
        bound_callables = binding.dependency_callables
        current_callables = _dependency_callables(runtime.dependencies)
        if len(current_callables) != len(bound_callables) or any(
            current is not bound
            for current, bound in zip(current_callables, bound_callables, strict=True)
        ):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production dependency callable identity drifted"
            )
        if not hmac.compare_digest(
            runtime.runtime_identity_bytes, binding.runtime_identity_bytes
        ):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime identity bytes drifted from their registry binding"
            )
        reobserved = _production_runtime_identity_bytes(runtime.bridge_runtime)
        if not hmac.compare_digest(runtime.runtime_identity_bytes, reobserved):
            raise ForagerMatchedV3PPOGRURunnerError(
                "production runtime identity drifted during live reobservation"
            )
        return binding


def _production_initialize_training(
    config: ppo_gru.PPOGRUConfig,
    rng_state: ppo_gru.PPOGRURNGState,
) -> tuple[PPOGRUTrainingHandle, ppo_gru.PPOGRURNGState]:
    state, next_rng = ppo_gru.initialize_ppo_gru_train_state(config, rng_state=rng_state)
    model = ppo_gru.PPOGRUActorCritic(
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    return PPOGRUTrainingHandle(model=model, state=state), next_rng


def _production_evaluate_step(
    training: PPOGRUTrainingHandle,
    carry: Array,
    observation: Array,
    reset_before: bool,
) -> PPOGRUStepEvaluation:
    if type(training.model) is not ppo_gru.PPOGRUActorCritic:
        raise ForagerMatchedV3PPOGRURunnerError("production model type drifted")
    if type(training.state) is not ppo_gru.PPOGRUTrainState:
        raise ForagerMatchedV3PPOGRURunnerError("production train-state type drifted")
    applied = cast(
        tuple[Array, Array, Array],
        training.model.apply(
            training.state.variables,
            carry,
            observation,
            jnp.asarray(reset_before, dtype=jnp.bool_),
        ),
    )
    outgoing, logits, value = applied
    return PPOGRUStepEvaluation(outgoing_carry=outgoing, logits=logits, value=value)


def _trace_to_core_rollout(trace: PPOGRURolloutTrace) -> ppo_gru.PPOGRURollout:
    return ppo_gru.PPOGRURollout(
        initial_carry=trace.initial_carry,
        observations=jnp.stack([step.observation for step in trace.steps]),
        reset_before=jnp.asarray(
            [step.reset_before for step in trace.steps], dtype=jnp.bool_
        ),
        actions=jnp.asarray([step.action for step in trace.steps], dtype=jnp.int32),
        rewards=jnp.asarray([step.reward for step in trace.steps], dtype=jnp.float32),
        transition_dones=jnp.asarray(
            [step.transition_done for step in trace.steps], dtype=jnp.bool_
        ),
        old_log_probs=jnp.stack([step.old_log_prob for step in trace.steps]),
        old_values=jnp.stack([step.old_value for step in trace.steps]),
        incoming_carries=jnp.stack([step.incoming_carry for step in trace.steps]),
        bootstrap_observation=trace.bootstrap_observation,
        bootstrap_value=trace.bootstrap_value,
    )


def _production_validate_rollout(
    training: PPOGRUTrainingHandle,
    trace: PPOGRURolloutTrace,
    config: ppo_gru.PPOGRUConfig,
) -> _CoreValidatedRollout:
    if type(training.model) is not ppo_gru.PPOGRUActorCritic:
        raise ForagerMatchedV3PPOGRURunnerError("production model type drifted")
    if type(training.state) is not ppo_gru.PPOGRUTrainState:
        raise ForagerMatchedV3PPOGRURunnerError("production train-state type drifted")
    rollout = _trace_to_core_rollout(trace)
    replayed = ppo_gru.evaluate_ppo_gru_sequence(
        training.model,
        training.state.variables,
        trace.initial_carry,
        rollout.observations,
        rollout.reset_before,
    )
    _exact_array_equal(
        jnp.stack([step.logits for step in trace.steps]),
        replayed.logits,
        "production replay logits",
    )
    _exact_array_equal(
        jnp.stack([step.incoming_carry for step in trace.steps]),
        replayed.incoming_carries,
        "production replay incoming carries",
    )
    _exact_array_equal(
        jnp.stack([step.outgoing_carry for step in trace.steps]),
        replayed.outgoing_carries,
        "production replay outgoing carries",
    )
    _exact_array_equal(
        trace.bootstrap_carry,
        replayed.final_carry,
        "production replay final/bootstrap carry",
    )
    advantages, targets = ppo_gru.validate_ppo_gru_rollout(
        training.model,
        training.state.variables,
        rollout,
        config,
        expected_initial_carry=trace.initial_carry,
        expected_initial_reset=trace.initial_reset,
    )
    return _CoreValidatedRollout(
        rollout=rollout,
        advantages=advantages,
        targets=targets,
    )


def _production_build_segments(
    validated: Any,
    order: tuple[int, ...],
    geometry: PPOGRURunnerGeometry,
) -> tuple[PPOGRURunnerSegment, ...]:
    if type(validated) is not _CoreValidatedRollout:
        raise ForagerMatchedV3PPOGRURunnerError("validated rollout type drifted")
    exact = validated
    segments = ppo_gru.build_ppo_gru_sequence_segments(
        exact.rollout,
        exact.advantages,
        exact.targets,
        segment_steps=geometry.segment_steps,
        segment_order=order,
    )
    return tuple(
        PPOGRURunnerSegment(
            segment_id=segment_id,
            time_indices=tuple(
                int(value) for value in np.asarray(segments.time_indices[position]).tolist()
            ),
            payload=ppo_gru.ppo_gru_loss_batch_from_segment(
                segments,
                position,
                ppo_gru.matched_v3_ppo_gru_configuration(),
            ),
        )
        for position, segment_id in enumerate(order)
    )


def _production_update_segment(
    training: PPOGRUTrainingHandle,
    payload: Any,
    expected_update_index: int,
    expected_learning_rate: float,
) -> PPOGRUTrainingHandle:
    del expected_learning_rate
    if type(training.model) is not ppo_gru.PPOGRUActorCritic:
        raise ForagerMatchedV3PPOGRURunnerError("production model type drifted")
    if type(training.state) is not ppo_gru.PPOGRUTrainState:
        raise ForagerMatchedV3PPOGRURunnerError("production train-state type drifted")
    if type(payload) is not ppo_gru.PPOGRULossBatch:
        raise ForagerMatchedV3PPOGRURunnerError("production loss-batch type drifted")
    current = _production_optimizer_update_count(training)
    if current != expected_update_index:
        raise ForagerMatchedV3PPOGRURunnerError("production update index drifted")
    result = ppo_gru.ppo_gru_update(
        training.model,
        training.state,
        payload,
        ppo_gru.matched_v3_ppo_gru_configuration(),
    )
    return PPOGRUTrainingHandle(model=training.model, state=result.state)


def _production_optimizer_update_count(training: PPOGRUTrainingHandle) -> int:
    if type(training.state) is not ppo_gru.PPOGRUTrainState:
        raise ForagerMatchedV3PPOGRURunnerError("production train-state type drifted")
    counter = training.state.optimizer_updates
    if tuple(counter.shape) != () or counter.dtype != jnp.int32:
        raise ForagerMatchedV3PPOGRURunnerError(
            "production optimizer counter must be scalar int32"
        )
    return int(counter)


def open_matched_v3_ppo_gru_runner_runtime() -> PPOGRUProductionRuntime:
    """Open one reusable exact bridge runtime without qualifying execution."""

    runtime = bridge.open_matched_v3_foragax_runtime()
    runtime_identity_bytes = _production_runtime_identity_bytes(runtime)

    dependencies = PPOGRURunnerDependencies(
        classification="production_adapter_runtime_unqualified",
        initialize_bridge=runtime.initialize,
        step_bridge=bridge.step_matched_v3_foragax_bridge,
        parse_runtime_identity=_ProductionRuntimeIdentityParser(runtime_identity_bytes),
        initialize_training=_production_initialize_training,
        evaluate_step=_production_evaluate_step,
        validate_rollout=_production_validate_rollout,
        build_segments=_production_build_segments,
        update_segment=_production_update_segment,
        optimizer_update_count=_production_optimizer_update_count,
    )
    opened = PPOGRUProductionRuntime(
        bridge_runtime=runtime,
        runtime_identity_bytes=runtime_identity_bytes,
        dependencies=dependencies,
        _capability=_ProductionRuntimeCapability(),
    )
    _register_production_runtime(opened)
    return opened


def production_ppo_gru_runner_dependencies(
    runtime: PPOGRUProductionRuntime,
) -> PPOGRURunnerDependencies:
    """Return exact dependencies from an already-open non-qualified runtime."""

    return _validated_production_runtime_binding(runtime).dependencies


def run_matched_v3_ppo_gru_production(
    *,
    environment_seed: object,
    agent_seed: object,
    runtime: PPOGRUProductionRuntime,
) -> PPOGRURunnerOutcome:
    """Run the exact public-seed horizon and return a non-authorizing in-memory outcome."""

    dependencies = production_ppo_gru_runner_dependencies(runtime)
    return _run_driver(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        runtime_identity_bytes=runtime.runtime_identity_bytes,
        geometry=MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY,
        dependencies=dependencies,
        production_runtime=runtime,
    )


def run_ppo_gru_engineering_driver(
    *,
    environment_seed: object,
    agent_seed: object,
    runtime_identity_bytes: bytes,
    geometry: PPOGRURunnerGeometry,
    dependencies: PPOGRURunnerDependencies,
) -> PPOGRURunnerOutcome:
    """Run a tiny synthetic trace that can never produce a production result receipt."""

    if geometry == MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY:
        raise ForagerMatchedV3PPOGRURunnerError(
            "engineering driver cannot execute or impersonate the production geometry"
        )
    if dependencies.classification != "synthetic_engineering_only":
        raise ForagerMatchedV3PPOGRURunnerError(
            "engineering driver requires explicitly synthetic dependencies"
        )
    return _run_driver(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        runtime_identity_bytes=runtime_identity_bytes,
        geometry=geometry,
        dependencies=dependencies,
        production_runtime=None,
    )


def _receipt_body(outcome: PPOGRURunnerOutcome, *, engineering: bool) -> dict[str, Any]:
    _validate_outcome(outcome)
    if engineering:
        if outcome.production_horizon_complete:
            raise ForagerMatchedV3PPOGRURunnerError(
                "production outcome cannot be emitted as an engineering receipt"
            )
        schema = PPO_GRU_ENGINEERING_RECEIPT_SCHEMA_VERSION
        classification = "synthetic_engineering_non_authorizing"
    else:
        if not outcome.production_horizon_complete:
            raise ForagerMatchedV3PPOGRURunnerError(
                "partial horizon cannot be emitted as a completed result receipt"
            )
        schema = PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
        classification = "production_runtime_unqualified_non_authorizing"
    runtime_identity = _strict_json_object(
        outcome.runtime_identity_bytes, "runtime identity"
    )
    return {
        "schema_version": schema,
        "candidate_id": "adapted_ppo_gru",
        "classification": classification,
        "runner": _frozen_runner_binding(),
        "dependencies": _frozen_dependency_binding(),
        "seeds": {
            "environment_seed": outcome.environment_seed,
            "agent_seed": outcome.agent_seed,
            "provenance": "caller_supplied_unverified",
            "upstream_receipt_bound": False,
            "protected_seed_status": "unverified",
        },
        "geometry": outcome.geometry.to_dict(),
        "accounting": {
            "environment_interactions": outcome.environment_interactions,
            "rollout_count": outcome.rollout_count,
            "optimizer_update_count": outcome.optimizer_update_count,
            "parameter_initialization_draw_count": (
                outcome.parameter_initialization_draw_count
            ),
            "action_draw_count": outcome.action_draw_count,
            "permutation_draw_count": outcome.permutation_draw_count,
            "total_agent_draw_count": outcome.total_agent_draw_count,
            "ppo_environment_draw_count": outcome.ppo_environment_draw_count,
            "bridge_reset_count": outcome.bridge_reset_count,
            "bridge_step_count": outcome.bridge_step_count,
            "bridge_environment_key_use_count": (
                outcome.bridge_environment_key_use_count
            ),
        },
        "raw_reward_trace": {
            "encoding": _RAW_REWARD_TRACE_ENCODING,
            "length": len(outcome.raw_reward_trace),
            "sha256": outcome.raw_reward_trace_sha256,
            "score_reduction": _RAW_SCORE_REDUCTION,
            "score_scaling": _RAW_SCORE_SCALING,
        },
        "raw_cumulative_score": outcome.raw_cumulative_score,
        "trace_chain_sha256": outcome.trace_chain_sha256,
        "production_horizon_complete": outcome.production_horizon_complete,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": outcome.runtime_identity_sha256,
        "claims": _non_authorizing_claims(),
        "limitations": [
            *_receipt_limitations(),
        ],
    }


def _receipt_bytes(outcome: PPOGRURunnerOutcome, *, engineering: bool) -> bytes:
    body = _receipt_body(outcome, engineering=engineering)
    _reject_authority_anywhere(body, "PPO-GRU receipt body")
    receipt = dict(body)
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json(body, "PPO-GRU receipt body")
    ).hexdigest()
    return _canonical_json(receipt, "PPO-GRU receipt")


def canonical_ppo_gru_engineering_receipt_bytes(
    outcome: PPOGRURunnerOutcome,
) -> bytes:
    """Return canonical bytes for a completed tiny synthetic trace."""

    return _receipt_bytes(outcome, engineering=True)


def canonical_ppo_gru_result_receipt_bytes(outcome: PPOGRURunnerOutcome) -> bytes:
    """Serialize only the exact registered in-process full-horizon outcome."""

    return _receipt_bytes(outcome, engineering=False)


def _parse_receipt(
    raw: bytes,
    *,
    engineering: bool,
    expected_receipt_sha256: str | None,
) -> dict[str, Any]:
    label = "PPO-GRU engineering receipt" if engineering else "PPO-GRU result receipt"
    receipt = _strict_json_object(raw, label)
    expected_keys = {
        "schema_version",
        "candidate_id",
        "classification",
        "runner",
        "dependencies",
        "seeds",
        "geometry",
        "accounting",
        "raw_reward_trace",
        "raw_cumulative_score",
        "trace_chain_sha256",
        "production_horizon_complete",
        "runtime_identity",
        "runtime_identity_sha256",
        "claims",
        "limitations",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} field membership drifted")
    supplied_receipt_sha256 = _require_sha256(
        receipt["receipt_sha256"], "receipt_sha256"
    )
    body = dict(receipt)
    del body["receipt_sha256"]
    calculated_receipt_sha256 = hashlib.sha256(
        _canonical_json(body, f"{label} body")
    ).hexdigest()
    if not hmac.compare_digest(supplied_receipt_sha256, calculated_receipt_sha256):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} body digest drifted")
    if expected_receipt_sha256 is not None:
        expected = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
            raise ForagerMatchedV3PPOGRURunnerError(f"{label} artifact digest drifted")
    expected_schema = (
        PPO_GRU_ENGINEERING_RECEIPT_SCHEMA_VERSION
        if engineering
        else PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    )
    expected_classification = (
        "synthetic_engineering_non_authorizing"
        if engineering
        else "production_runtime_unqualified_non_authorizing"
    )
    if receipt["schema_version"] != expected_schema:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} schema drifted")
    if receipt["candidate_id"] != "adapted_ppo_gru":
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} candidate drifted")
    if receipt["classification"] != expected_classification:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} classification drifted")
    expected_runner = _frozen_runner_binding()
    if receipt["runner"] != expected_runner:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} runner binding drifted")
    if receipt["dependencies"] != _frozen_dependency_binding():
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} dependency binding drifted")
    if receipt["claims"] != _non_authorizing_claims():
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} authority claims drifted")
    if receipt["limitations"] != list(_receipt_limitations()):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} limitations drifted")
    _reject_authority_anywhere(receipt, label)
    seeds = receipt["seeds"]
    if type(seeds) is not dict or set(seeds) != {
        "environment_seed",
        "agent_seed",
        "provenance",
        "upstream_receipt_bound",
        "protected_seed_status",
    }:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} seed contract drifted")
    if (
        seeds["provenance"] != "caller_supplied_unverified"
        or seeds["upstream_receipt_bound"] is not False
        or seeds["protected_seed_status"] != "unverified"
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} cannot claim verified seed provenance"
        )
    ppo_gru.validate_ppo_gru_seed_pair(
        seeds["environment_seed"], seeds["agent_seed"]
    )
    geometry_value = receipt["geometry"]
    if type(geometry_value) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} geometry must be an object")
    required_geometry_keys = set(MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY.to_dict())
    if set(geometry_value) != required_geometry_keys:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} geometry fields drifted")
    if any(type(value) is not int for value in geometry_value.values()):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} geometry values must be exact integers"
        )
    geometry = PPOGRURunnerGeometry(
        horizon=_require_exact_int(geometry_value["horizon"], "horizon", minimum=1),
        rollout_steps=_require_exact_int(
            geometry_value["rollout_steps"], "rollout_steps", minimum=1
        ),
        segment_steps=_require_exact_int(
            geometry_value["segment_steps"], "segment_steps", minimum=1
        ),
        update_epochs=_require_exact_int(
            geometry_value["update_epochs"], "update_epochs", minimum=1
        ),
    )
    if geometry_value != geometry.to_dict():
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} derived geometry drifted")
    production_complete = receipt["production_horizon_complete"]
    if type(production_complete) is not bool:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} production completion flag must be boolean"
        )
    if engineering:
        if production_complete or geometry == MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY:
            raise ForagerMatchedV3PPOGRURunnerError(
                "engineering receipt cannot impersonate production completion"
            )
    elif not production_complete or geometry != MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY:
        raise ForagerMatchedV3PPOGRURunnerError(
            "result receipt requires the exact complete production horizon"
        )
    accounting = receipt["accounting"]
    if type(accounting) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} accounting must be an object")
    expected_accounting = {
        "environment_interactions": geometry.horizon,
        "rollout_count": geometry.rollout_count,
        "optimizer_update_count": geometry.optimizer_update_count,
        "parameter_initialization_draw_count": 1,
        "action_draw_count": geometry.action_draw_count,
        "permutation_draw_count": geometry.permutation_draw_count,
        "total_agent_draw_count": geometry.total_agent_draw_count,
        "ppo_environment_draw_count": 0,
        "bridge_reset_count": 1,
        "bridge_step_count": geometry.horizon,
        "bridge_environment_key_use_count": 1 + geometry.horizon,
    }
    if (
        set(accounting) != set(expected_accounting)
        or any(type(value) is not int for value in accounting.values())
        or accounting != expected_accounting
    ):
        raise ForagerMatchedV3PPOGRURunnerError(f"{label} exact accounting drifted")
    reward_trace = receipt["raw_reward_trace"]
    expected_reward_trace = {
        "encoding": _RAW_REWARD_TRACE_ENCODING,
        "length": geometry.horizon,
        "sha256": reward_trace.get("sha256") if type(reward_trace) is dict else None,
        "score_reduction": _RAW_SCORE_REDUCTION,
        "score_scaling": _RAW_SCORE_SCALING,
    }
    if (
        type(reward_trace) is not dict
        or set(reward_trace) != set(expected_reward_trace)
        or reward_trace != expected_reward_trace
    ):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} raw reward trace metadata drifted"
        )
    _require_sha256(reward_trace["sha256"], "raw reward trace sha256")
    if type(receipt["raw_cumulative_score"]) is not int:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} raw cumulative score must be an exact integer"
        )
    if not -geometry.horizon <= receipt["raw_cumulative_score"] <= 30 * geometry.horizon:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} raw cumulative score is outside exact support bounds"
        )
    _require_sha256(receipt["trace_chain_sha256"], "trace_chain_sha256")
    runtime_identity = receipt["runtime_identity"]
    if type(runtime_identity) is not dict:
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} runtime identity must be an object"
        )
    runtime_raw = _canonical_json(runtime_identity, "receipt runtime identity")
    runtime_sha256 = _require_sha256(
        receipt["runtime_identity_sha256"], "runtime_identity_sha256"
    )
    if not hmac.compare_digest(hashlib.sha256(runtime_raw).hexdigest(), runtime_sha256):
        raise ForagerMatchedV3PPOGRURunnerError(
            f"{label} runtime identity digest drifted"
        )
    if not engineering:
        _validate_production_runtime_identity(runtime_identity, f"{label} runtime identity")
    return receipt


def parse_ppo_gru_engineering_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly parse a canonical engineering receipt."""

    return _parse_receipt(
        raw,
        engineering=True,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def parse_ppo_gru_result_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Structurally parse canonical result bytes without attesting execution."""

    return _parse_receipt(
        raw,
        engineering=False,
        expected_receipt_sha256=expected_receipt_sha256,
    )


__all__ = [
    "FORAGAX_BRIDGE_IMPLEMENTATION_PATH",
    "FORAGAX_BRIDGE_IMPLEMENTATION_SHA256",
    "MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY",
    "PPO_GRU_CORE_IMPLEMENTATION_PATH",
    "PPO_GRU_CORE_IMPLEMENTATION_SHA256",
    "PPO_GRU_ENGINEERING_RECEIPT_SCHEMA_VERSION",
    "PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION",
    "PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION",
    "PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION",
    "PPO_GRU_RUNNER_DESCRIPTOR_SHA256",
    "PPO_GRU_RUNNER_IMPLEMENTATION_PATH",
    "ForagerMatchedV3PPOGRURunnerError",
    "PPOGRURolloutStep",
    "PPOGRURolloutTrace",
    "PPOGRURunnerDependencies",
    "PPOGRURunnerGeometry",
    "PPOGRURunnerOutcome",
    "PPOGRUProductionRuntime",
    "PPOGRURunnerSegment",
    "PPOGRUStepEvaluation",
    "PPOGRUTrainingHandle",
    "canonical_matched_v3_ppo_gru_runner_descriptor_bytes",
    "canonical_ppo_gru_engineering_receipt_bytes",
    "canonical_ppo_gru_result_receipt_bytes",
    "matched_v3_ppo_gru_runner_descriptor",
    "open_matched_v3_ppo_gru_runner_runtime",
    "parse_matched_v3_ppo_gru_runner_descriptor",
    "parse_ppo_gru_engineering_receipt",
    "parse_ppo_gru_result_receipt",
    "production_ppo_gru_runner_dependencies",
    "run_matched_v3_ppo_gru_production",
    "run_ppo_gru_engineering_driver",
    "validate_ppo_gru_runner_rollout_trace",
]
