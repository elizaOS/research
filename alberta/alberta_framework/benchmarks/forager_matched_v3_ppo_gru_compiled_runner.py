"""Additive compiled PPO-GRU runner for the matched-v3 Forager task.

This module deliberately does not replace the source-bound v1 runner.  It binds
that runner as a reviewed semantic reference and introduces a distinct v2 result
receipt for an exact 512-transition ``jax.lax.scan`` rollout kernel.  One live
runtime is a single-use, process-local capability.  Completed outcomes are also
process-local capabilities; persisted receipt bytes remain structural and grant
no execution, ingestion, qualification, promotion, or performance authority.

The implementation is present but unexecuted at the full horizon.  Its public
execution surface requires an explicit ``unqualified_engineering=True`` opt-in.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final, NamedTuple, NoReturn, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru_runner as v1_runner

PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_runner_descriptor.v1"
)
PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_result_receipt.v2"
)
PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_compiled_runtime_identity.v1"
)
PPO_GRU_COMPILED_RUNNER_STATUS: Final = "implemented_unexecuted"

PPO_GRU_COMPILED_RUNNER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_compiled_runner.py"
)
BOUND_BRIDGE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
)
BOUND_BRIDGE_IMPLEMENTATION_SHA256: Final = (
    "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
)
BOUND_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
BOUND_FORAGAX_INSTALL_TREE_SHA256: Final = (
    "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
)
BOUND_CORE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py"
)
BOUND_CORE_IMPLEMENTATION_SHA256: Final = (
    "58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e"
)
BOUND_CORE_CONFIGURATION_SHA256: Final = (
    "07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70"
)
BOUND_CORE_SOURCE_DESCRIPTOR_SHA256: Final = (
    "64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca"
)
BOUND_V1_RUNNER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
)
BOUND_V1_RUNNER_IMPLEMENTATION_SHA256: Final = (
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
)
BOUND_V1_RUNNER_DESCRIPTOR_SHA256: Final = (
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
)

MATCHED_V3_HORIZON: Final = 499_712
PPO_GRU_COMPILED_CHUNK_STEPS: Final = 512
PPO_GRU_COMPILED_CHUNK_COUNT: Final = 976
PPO_GRU_SEGMENT_STEPS: Final = 128
PPO_GRU_SEGMENTS_PER_ROLLOUT: Final = 4
PPO_GRU_UPDATE_EPOCHS: Final = 4
PPO_GRU_OPTIMIZER_UPDATES: Final = 15_616
PPO_GRU_PARAMETER_INITIALIZATION_DRAWS: Final = 1
PPO_GRU_ACTION_DRAWS: Final = MATCHED_V3_HORIZON
PPO_GRU_PERMUTATION_DRAWS: Final = PPO_GRU_COMPILED_CHUNK_COUNT * PPO_GRU_UPDATE_EPOCHS
PPO_GRU_TOTAL_AGENT_DRAWS: Final = (
    PPO_GRU_PARAMETER_INITIALIZATION_DRAWS
    + PPO_GRU_ACTION_DRAWS
    + PPO_GRU_PERMUTATION_DRAWS
)
PPO_GRU_BRIDGE_KEY_USES: Final = MATCHED_V3_HORIZON + 1

VIOLATION_OBSERVATION: Final = 1 << 0
VIOLATION_REWARD: Final = 1 << 1
VIOLATION_DONE: Final = 1 << 2
VIOLATION_INFO: Final = 1 << 3
VIOLATION_ENVIRONMENT_TIME: Final = 1 << 4
VIOLATION_POLICY: Final = 1 << 5
VIOLATION_ACTION: Final = 1 << 6

_MAX_DESCRIPTOR_BYTES: Final = 256 * 1024
_MAX_RECEIPT_BYTES: Final = 1024 * 1024
_MAX_JSON_DEPTH: Final = 128
_MAX_JSON_NODES: Final = 100_000
_UINT31_MAXIMUM: Final = (1 << 31) - 1
_TRACE_DOMAIN: Final = b"alberta.forager_matched_v3.ppo_gru.compiled_trace.v1\x00"
_OUTCOME_DOMAIN: Final = b"alberta.forager_matched_v3.ppo_gru.compiled_outcome.v1\x00"
_EXPECTED_INFO_KEYS: Final = frozenset(
    {
        "discount",
        "temperatures",
        "biome_id",
        "object_collected_id",
        "current_biome_mean",
        "max_biome_mean",
        "biome_regret",
        "biome_rank",
        "rewards",
    }
)


class ForagerMatchedV3PPOGRUCompiledRunnerError(ValueError):
    """A compiled-runner descriptor, state, trace, or receipt is invalid."""


class PPOGRUCompiledExecutionBlockedError(RuntimeError):
    """The explicit unqualified-engineering execution acknowledgement is absent."""


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()


def _check_bound_sources() -> None:
    checks = (
        (
            bridge.__file__,
            BOUND_BRIDGE_IMPLEMENTATION_PATH,
            BOUND_BRIDGE_IMPLEMENTATION_SHA256,
        ),
        (
            ppo_gru.__file__,
            BOUND_CORE_IMPLEMENTATION_PATH,
            BOUND_CORE_IMPLEMENTATION_SHA256,
        ),
        (
            v1_runner.__file__,
            BOUND_V1_RUNNER_IMPLEMENTATION_PATH,
            BOUND_V1_RUNNER_IMPLEMENTATION_SHA256,
        ),
    )
    for module_file, path, expected in checks:
        if not hmac.compare_digest(_source_sha256(module_file, path), expected):
            raise RuntimeError(f"compiled PPO-GRU source binding drifted for {path}")
    if (
        bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256 != BOUND_BRIDGE_DESCRIPTOR_SHA256
        or bridge.FORAGAX_INSTALL_TREE_SHA256 != BOUND_FORAGAX_INSTALL_TREE_SHA256
        or ppo_gru.PPO_GRU_CONFIGURATION_SHA256 != BOUND_CORE_CONFIGURATION_SHA256
        or ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
        != BOUND_CORE_SOURCE_DESCRIPTOR_SHA256
        or v1_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
        != BOUND_V1_RUNNER_DESCRIPTOR_SHA256
    ):
        raise RuntimeError("compiled PPO-GRU descriptor binding drifted")


_check_bound_sources()


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "compilation_qualified": False,
        "execution_authorized": False,
        "execution_ready": False,
        "ingestion_authorized": False,
        "performance_claim_allowed": False,
        "promotion_authorized": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


_AUTHORITY_FIELDS: Final = frozenset(_claims())
_DESCRIPTOR_LIMITATIONS: Final = (
    "No real compiled chunk or full-horizon workload has been accepted.",
    "The frozen v1 runner is a semantic reference, not execution provenance.",
    "Receipt bytes are structural and do not independently prove execution.",
    "Seed provenance is caller-supplied and unverified.",
    "No source closure, backend qualification, checkpoint, or writer exists here.",
    (
        "This additive runner is not bound by qualification-plan v1 or the base "
        "configuration plan."
    ),
    (
        "Qualification or scheduled execution requires a separately versioned "
        "pre-observation addendum or plan."
    ),
    "Passing tests cannot authorize ingestion, promotion, or a performance claim.",
)
_RECEIPT_LIMITATIONS: Final = (
    "Receipt bytes do not independently attest execution.",
    "Runtime, source closure, resource use, and scientific qualification remain external.",
    "Seed provenance is caller-supplied and unverified.",
    (
        "This additive runner is not bound by qualification-plan v1 or the base "
        "configuration plan."
    ),
    (
        "Qualification or scheduled execution requires a separately versioned "
        "pre-observation addendum or plan."
    ),
    "No ingestion, promotion, performance, or SOTA authority is granted.",
)


def _reject_authority_anywhere(value: object, *, label: str) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            for key, child in cast(dict[str, object], item).items():
                if key in _AUTHORITY_FIELDS and child is not False:
                    raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                        f"{label} contains non-false authority field {key}"
                    )
                pending.append(child)
        elif type(item) is list:
            pending.extend(cast(list[object], item))


def _assert_plain_unaliased_json(value: object, *, label: str) -> None:
    pending = [(value, 1)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"{label} exceeds the JSON node limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"{label} exceeds the JSON nesting-depth limit"
            )
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    f"{label} contains a non-string key"
                )
            pending.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    f"{label} contains aliased or cyclic containers"
                )
            seen.add(identity)
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    f"{label} contains a non-finite float"
                )
        elif item is not None and type(item) not in {str, int, bool}:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"{label} contains non-plain type {type(item).__name__}"
            )


def _validate_json_lexical_bounds(text: str, *, label: str) -> None:
    """Reject excessive JSON structure before the recursive stdlib decoder runs."""

    depth = 0
    node_tokens = 0
    in_string = False
    escaped = False
    in_primitive = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            in_primitive = False
            node_tokens += 1
        elif character in "[{":
            depth += 1
            node_tokens += 1
            in_primitive = False
            if depth > _MAX_JSON_DEPTH:
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    f"{label} exceeds the JSON nesting-depth limit"
                )
        elif character in "]}":
            depth -= 1
            in_primitive = False
        elif character in ",:":
            in_primitive = False
        elif character in " \t\r\n":
            in_primitive = False
        elif not in_primitive:
            node_tokens += 1
            in_primitive = True
        if node_tokens > _MAX_JSON_NODES:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"{label} exceeds the JSON node limit"
            )


def _canonical_json(value: object, *, label: str, maximum_bytes: int) -> bytes:
    _assert_plain_unaliased_json(value, label=label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is not finite canonical JSON"
        ) from exc
    if len(raw) > maximum_bytes:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} exceeds its byte limit"
        )
    return raw


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> NoReturn:
    raise ForagerMatchedV3PPOGRUCompiledRunnerError(
        f"non-finite JSON constant {token!r}"
    )


def _strict_json_object(raw: object, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(f"{label} must be exact bytes")
    exact = raw
    if not exact or len(exact) > maximum_bytes:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is empty or exceeds its byte limit"
        )
    try:
        text = exact.decode("ascii")
        _validate_json_lexical_bounds(text, label=label)
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3PPOGRUCompiledRunnerError:
        raise
    except (RecursionError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(parsed) is not dict:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} must encode one plain object"
        )
    result = cast(dict[str, Any], parsed)
    _assert_plain_unaliased_json(result, label=label)
    if not hmac.compare_digest(
        exact,
        _canonical_json(result, label=label, maximum_bytes=maximum_bytes),
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is not in exact canonical form"
        )
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} must be an exact lowercase SHA-256"
        )
    return value


def matched_v3_ppo_gru_compiled_accounting() -> dict[str, int]:
    """Return detached exact workload arithmetic without executing JAX."""

    return {
        "action_draws": PPO_GRU_ACTION_DRAWS,
        "automatic_resets": 0,
        "bridge_environment_key_uses": PPO_GRU_BRIDGE_KEY_USES,
        "bridge_resets": 1,
        "compiled_chunk_count": PPO_GRU_COMPILED_CHUNK_COUNT,
        "compiled_chunk_steps": PPO_GRU_COMPILED_CHUNK_STEPS,
        "environment_interactions": MATCHED_V3_HORIZON,
        "optimizer_updates": PPO_GRU_OPTIMIZER_UPDATES,
        "parameter_initialization_draws": PPO_GRU_PARAMETER_INITIALIZATION_DRAWS,
        "permutation_draws": PPO_GRU_PERMUTATION_DRAWS,
        "segment_steps": PPO_GRU_SEGMENT_STEPS,
        "segments_per_rollout": PPO_GRU_SEGMENTS_PER_ROLLOUT,
        "total_agent_draws": PPO_GRU_TOTAL_AGENT_DRAWS,
        "update_epochs": PPO_GRU_UPDATE_EPOCHS,
    }


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        "candidate_id": "adapted_ppo_gru",
        "status": PPO_GRU_COMPILED_RUNNER_STATUS,
        "classification": "additive_compiled_runner_unexecuted_non_authorizing",
        "implementation": {
            "module": (
                "alberta_framework.benchmarks."
                "forager_matched_v3_ppo_gru_compiled_runner"
            ),
            "path": PPO_GRU_COMPILED_RUNNER_IMPLEMENTATION_PATH,
            "source_self_hash_bound": False,
            "source_digest_requires_external_binding": True,
        },
        "bindings": {
            "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_path": BOUND_BRIDGE_IMPLEMENTATION_PATH,
            "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
            "foragax_install_tree_sha256": BOUND_FORAGAX_INSTALL_TREE_SHA256,
            "core_configuration_sha256": BOUND_CORE_CONFIGURATION_SHA256,
            "core_source_descriptor_sha256": BOUND_CORE_SOURCE_DESCRIPTOR_SHA256,
            "core_implementation_path": BOUND_CORE_IMPLEMENTATION_PATH,
            "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
            "v1_semantic_reference_runner_descriptor_sha256": (
                BOUND_V1_RUNNER_DESCRIPTOR_SHA256
            ),
            "v1_semantic_reference_runner_implementation_path": (
                BOUND_V1_RUNNER_IMPLEMENTATION_PATH
            ),
            "v1_semantic_reference_runner_implementation_sha256": (
                BOUND_V1_RUNNER_IMPLEMENTATION_SHA256
            ),
        },
        "geometry": matched_v3_ppo_gru_compiled_accounting(),
        "kernel": {
            "api": "jax.jit_of_jax.lax.scan",
            "rollout_steps": PPO_GRU_COMPILED_CHUNK_STEPS,
            "policy_parameters_fixed_within_rollout": True,
            "environment_and_policy_fused_per_transition": True,
            "bootstrap_evaluation_consumes_agent_key": False,
            "host_step_api_used_after_reset": False,
            "host_validation_at_rollout_boundary": True,
            "host_action_key_chain_replayed": True,
            "host_environment_key_endpoint_replayed": True,
            "host_categorical_action_and_log_prob_replayed": True,
            "invalid_policy_poisoning": (
                "pre_key_split_pre_environment_step_zero_transition_consumption"
            ),
            "invalid_environment_step_poisoning": (
                "first_invalid_environment_transition_then_no_more_transition_calls"
            ),
            "violation_bits": {
                "action": VIOLATION_ACTION,
                "done": VIOLATION_DONE,
                "environment_time": VIOLATION_ENVIRONMENT_TIME,
                "info": VIOLATION_INFO,
                "observation": VIOLATION_OBSERVATION,
                "policy": VIOLATION_POLICY,
                "reward": VIOLATION_REWARD,
            },
        },
        "rng": {
            "implementation": "threefry2x32",
            "categorical_mode": ppo_gru.PPO_GRU_CATEGORICAL_MODE,
            "environment_root": "jax.random.key(environment_seed)",
            "environment_reset": "continuation,reset_key=split(root)",
            "environment_transition": "continuation,step_key=split(continuation)",
            "agent_initialization": "continuation,init_key=split(agent_root)",
            "agent_action": "continuation,action_key=split(continuation)",
            "agent_permutation": "one continuation split per epoch",
            "automatic_resets": 0,
            "ppo_environment_chain_consumed": False,
        },
        "state_lifecycle": {
            "runtime_capability": "weak_process_local_single_use_pid_bound",
            "outcome_capability": "weak_process_local_pid_and_runtime_bound",
            "copy_and_pickle_rejected": True,
            "external_checkpoint_accepted": False,
            "fork_reuse_rejected": True,
            "completion_registration": "run_closure_after_exact_horizon_checks_only",
        },
        "receipt": {
            "schema_version": PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION,
            "full_horizon_only": True,
            "expected_full_file_sha256_required_by_parser": True,
            "persisted_receipt_is_execution_attestation": False,
            "filesystem_writes": False,
            "parser_maximum_nesting_depth": _MAX_JSON_DEPTH,
            "parser_maximum_nodes": _MAX_JSON_NODES,
        },
        "claims": _claims(),
        "limitations": list(_DESCRIPTOR_LIMITATIONS),
    }


_DESCRIPTOR: Final = _descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(
    _DESCRIPTOR,
    label="compiled PPO-GRU runner descriptor",
    maximum_bytes=_MAX_DESCRIPTOR_BYTES,
)
PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256: Final = (
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256,
):
    raise RuntimeError("compiled PPO-GRU descriptor digest drifted")


def matched_v3_ppo_gru_compiled_runner_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the non-authorizing descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_matched_v3_ppo_gru_compiled_runner_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def parse_matched_v3_ppo_gru_compiled_runner_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen descriptor identity."""

    parsed = _strict_json_object(
        raw,
        label="compiled PPO-GRU runner descriptor",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    if not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled PPO-GRU runner descriptor identity drifted"
        )
    _reject_authority_anywhere(parsed, label="compiled PPO-GRU runner descriptor")
    return parsed


class _KernelCarry(NamedTuple):
    environment_state: Any
    observation: Array
    environment_key: Array
    gru_carry: Array
    agent_key: Array
    absolute_step: Array
    violation_mask: Array
    first_invalid_offset: Array


class _KernelStepOutput(NamedTuple):
    observation: Array
    incoming_carry: Array
    outgoing_carry: Array
    action_key_words: Array
    logits: Array
    action: Array
    old_log_prob: Array
    old_value: Array
    raw_reward: Array
    next_observation: Array


class _CompiledChunkResult(NamedTuple):
    environment_state: Any
    observation: Array
    environment_key: Array
    gru_carry: Array
    agent_key: Array
    absolute_step: Array
    violation_mask: Array
    first_invalid_offset: Array
    observations: Array
    incoming_carries: Array
    outgoing_carries: Array
    action_key_words: Array
    logits: Array
    actions: Array
    old_log_probs: Array
    old_values: Array
    raw_rewards: Array
    next_observations: Array
    bootstrap_value: Array


class _ChunkKeyReplay(NamedTuple):
    final_environment_key: Array
    final_agent_key: Array
    action_key_words: Array
    actions: Array
    old_log_probs: Array


type _EnvironmentStep = Callable[[Array, Any, Array, Any], tuple[Any, ...]]
type _PolicyApply = Callable[[Any, Array, Array, Array], tuple[Array, Array, Array]]
type _CompiledKernel = Callable[
    [Any, Any, Array, Array, Array, Array, Array], _CompiledChunkResult
]


def _require_traced_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    label: str,
) -> Array:
    if (
        not hasattr(value, "shape")
        or not hasattr(value, "dtype")
        or tuple(cast(Any, value).shape) != shape
        or cast(Any, value).dtype != dtype
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} must have exact shape {shape!r} and dtype {dtype}"
        )
    return cast(Array, value)


def _state_time(value: object) -> Array:
    candidate: object
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if "time" not in mapping:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled environment state lacks time"
            )
        candidate = mapping["time"]
    elif hasattr(value, "time"):
        candidate = getattr(value, "time")
    else:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled environment state lacks time"
        )
    return _require_traced_array(
        candidate,
        shape=(),
        dtype=jnp.int32,
        label="compiled environment time",
    )


def _support_member(value: Array, support: tuple[int, ...]) -> Array:
    result = jnp.asarray(False)
    for member in support:
        result = result | (value == jnp.asarray(member, dtype=value.dtype))
    return result


def _dynamic_violation_mask(
    *,
    observation: Array,
    reward: Array,
    done: Array,
    info: dict[str, Any],
    environment_state: Any,
    expected_time: Array,
    outgoing_carry: Array,
    logits: Array,
    value: Array,
    old_log_prob: Array,
    action: Array,
) -> Array:
    mask = jnp.asarray(0, dtype=jnp.uint32)
    observation_valid = (
        jnp.all(jnp.isfinite(observation))
        & jnp.all((observation == 0.0) | (observation == 1.0))
        & jnp.all(jnp.sum(observation, axis=-1) <= 1.0)
    )
    mask = mask | jnp.where(
        observation_valid,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_OBSERVATION),
    )
    reward_valid = jnp.isfinite(reward) & _support_member(reward, (-1, 0, 1, 30))
    mask = mask | jnp.where(
        reward_valid,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_REWARD),
    )
    mask = mask | jnp.where(
        ~done,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_DONE),
    )

    discount = cast(Array, info["discount"])
    temperatures = cast(Array, info["temperatures"])
    biome_id = cast(Array, info["biome_id"])
    collected = cast(Array, info["object_collected_id"])
    current = cast(Array, info["current_biome_mean"])
    maximum = cast(Array, info["max_biome_mean"])
    regret = cast(Array, info["biome_regret"])
    rank = cast(Array, info["biome_rank"])
    reward_grid = cast(Array, info["rewards"])
    info_valid = (
        jnp.isfinite(discount)
        & (discount == jnp.float32(1.0))
        & jnp.all(jnp.isfinite(temperatures))
        & jnp.all(temperatures == jnp.float32(0.0))
        & _support_member(biome_id, (-1, 0, 1))
        & _support_member(collected, (-1, 1, 2, 3))
        & jnp.isfinite(current)
        & jnp.isfinite(maximum)
        & jnp.isfinite(regret)
        & (current <= maximum)
        & (regret >= jnp.float32(0.0))
        & ((maximum - current).astype(jnp.float32) == regret.astype(jnp.float32))
        & (rank >= jnp.int32(1))
        & (rank <= jnp.int32(3))
        & jnp.all(jnp.isfinite(reward_grid))
        & jnp.all(
            (reward_grid == jnp.float16(-1.0))
            | (reward_grid == jnp.float16(0.0))
            | (reward_grid == jnp.float16(1.0))
            | (reward_grid == jnp.float16(30.0))
        )
    )
    mask = mask | jnp.where(
        info_valid,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_INFO),
    )
    mask = mask | jnp.where(
        _state_time(environment_state) == expected_time,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_ENVIRONMENT_TIME),
    )
    policy_valid = (
        jnp.all(jnp.isfinite(outgoing_carry))
        & jnp.all(jnp.isfinite(logits))
        & jnp.isfinite(value)
        & jnp.isfinite(old_log_prob)
    )
    mask = mask | jnp.where(
        policy_valid,
        jnp.uint32(0),
        jnp.uint32(VIOLATION_POLICY),
    )
    mask = mask | jnp.where(
        (action >= jnp.int32(0)) & (action < jnp.int32(4)),
        jnp.uint32(0),
        jnp.uint32(VIOLATION_ACTION),
    )
    return mask


def _validate_static_step_result(
    step_result: object,
) -> tuple[Array, Any, Array, Array, dict[str, Any]]:
    if type(step_result) is not tuple or len(step_result) != 5:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled environment step must return exactly five values"
        )
    raw_observation, environment_state, raw_reward, raw_done, raw_info = cast(
        tuple[object, object, object, object, object], step_result
    )
    observation = _require_traced_array(
        raw_observation,
        shape=(9, 9, 3),
        dtype=jnp.float32,
        label="compiled observation",
    )
    reward = _require_traced_array(
        raw_reward,
        shape=(),
        dtype=jnp.float32,
        label="compiled reward",
    )
    done = _require_traced_array(
        raw_done,
        shape=(),
        dtype=jnp.bool_,
        label="compiled done",
    )
    if type(raw_info) is not dict or frozenset(cast(dict[object, object], raw_info)) != (
        _EXPECTED_INFO_KEYS
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled info keys differ from the exact Foragax contract"
        )
    info = cast(dict[str, Any], raw_info)
    specifications = {
        "discount": ((), jnp.float32),
        "temperatures": ((4,), jnp.float32),
        "biome_id": ((), jnp.int16),
        "object_collected_id": ((), jnp.int32),
        "current_biome_mean": ((), jnp.float32),
        "max_biome_mean": ((), jnp.float32),
        "biome_regret": ((), jnp.float32),
        "biome_rank": ((), jnp.int32),
        "rewards": ((9, 9), jnp.float16),
    }
    for name, (shape, dtype) in specifications.items():
        _require_traced_array(
            info[name],
            shape=shape,
            dtype=dtype,
            label=f"compiled info {name}",
        )
    _state_time(environment_state)
    return observation, environment_state, reward, done, info


def _build_chunk_kernel(
    *,
    environment_step: _EnvironmentStep,
    environment_params: Any,
    policy_apply: _PolicyApply,
    hidden_size: int,
    chunk_steps: int,
) -> _CompiledKernel:
    """Build a private pure scan kernel; production fixes ``chunk_steps`` to 512."""

    if not callable(environment_step) or not callable(policy_apply):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled kernel dependencies must be callable"
        )
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled hidden_size must be a positive exact integer"
        )
    if type(chunk_steps) is not int or not 1 <= chunk_steps <= PPO_GRU_COMPILED_CHUNK_STEPS:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk_steps must lie in [1, 512]"
        )

    @jax.jit
    def kernel(
        variables: Any,
        environment_state: Any,
        observation: Array,
        environment_key: Array,
        gru_carry: Array,
        agent_key: Array,
        start_step: Array,
    ) -> _CompiledChunkResult:
        _require_traced_array(
            observation,
            shape=(9, 9, 3),
            dtype=jnp.float32,
            label="initial compiled observation",
        )
        _require_traced_array(
            gru_carry,
            shape=(hidden_size,),
            dtype=jnp.float32,
            label="initial compiled GRU carry",
        )
        _require_traced_array(
            start_step,
            shape=(),
            dtype=jnp.int32,
            label="compiled start step",
        )

        initial = _KernelCarry(
            environment_state=environment_state,
            observation=observation,
            environment_key=environment_key,
            gru_carry=gru_carry,
            agent_key=agent_key,
            absolute_step=start_step,
            violation_mask=jnp.asarray(0, dtype=jnp.uint32),
            first_invalid_offset=jnp.asarray(-1, dtype=jnp.int32),
        )

        def body(
            current: _KernelCarry, offset: Array
        ) -> tuple[_KernelCarry, _KernelStepOutput]:
            def poison_output(state: _KernelCarry) -> _KernelStepOutput:
                return _KernelStepOutput(
                    observation=state.observation,
                    incoming_carry=state.gru_carry,
                    outgoing_carry=state.gru_carry,
                    action_key_words=jnp.zeros((2,), dtype=jnp.uint32),
                    logits=jnp.zeros((4,), dtype=jnp.float32),
                    action=jnp.asarray(0, dtype=jnp.int32),
                    old_log_prob=jnp.asarray(0.0, dtype=jnp.float32),
                    old_value=jnp.asarray(0.0, dtype=jnp.float32),
                    raw_reward=jnp.asarray(0.0, dtype=jnp.float32),
                    next_observation=state.observation,
                )

            def live_step(state: _KernelCarry) -> tuple[_KernelCarry, _KernelStepOutput]:
                applied = policy_apply(
                    variables,
                    state.gru_carry,
                    state.observation,
                    jnp.asarray(False, dtype=jnp.bool_),
                )
                if type(applied) is not tuple or len(applied) != 3:
                    raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                        "compiled policy must return carry, logits, value"
                    )
                outgoing, logits, value = applied
                outgoing = _require_traced_array(
                    outgoing,
                    shape=(hidden_size,),
                    dtype=jnp.float32,
                    label="compiled outgoing carry",
                )
                logits = _require_traced_array(
                    logits,
                    shape=(4,),
                    dtype=jnp.float32,
                    label="compiled logits",
                )
                value = _require_traced_array(
                    value,
                    shape=(),
                    dtype=jnp.float32,
                    label="compiled value",
                )
                policy_valid = (
                    jnp.all(jnp.isfinite(outgoing))
                    & jnp.all(jnp.isfinite(logits))
                    & jnp.isfinite(value)
                )

                def valid_policy_step(
                    valid_state: _KernelCarry,
                ) -> tuple[_KernelCarry, _KernelStepOutput]:
                    next_agent_key, action_key = jr.split(valid_state.agent_key)
                    action = jr.categorical(
                        action_key,
                        logits,
                        axis=-1,
                        mode=ppo_gru.PPO_GRU_CATEGORICAL_MODE,
                    ).astype(jnp.int32)
                    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
                    old_log_prob = jnp.sum(
                        log_probabilities
                        * jax.nn.one_hot(action, 4, dtype=jnp.float32),
                        axis=-1,
                    )
                    next_environment_key, step_key = jr.split(
                        valid_state.environment_key
                    )
                    raw_step = environment_step(
                        step_key,
                        valid_state.environment_state,
                        action,
                        environment_params,
                    )
                    (
                        next_observation,
                        next_environment_state,
                        reward,
                        done,
                        info,
                    ) = _validate_static_step_result(raw_step)
                    expected_time = valid_state.absolute_step + jnp.int32(1)
                    step_mask = _dynamic_violation_mask(
                        observation=next_observation,
                        reward=reward,
                        done=done,
                        info=info,
                        environment_state=next_environment_state,
                        expected_time=expected_time,
                        outgoing_carry=outgoing,
                        logits=logits,
                        value=value,
                        old_log_prob=old_log_prob,
                        action=action,
                    )
                    first_invalid = jnp.where(
                        step_mask != jnp.uint32(0),
                        offset.astype(jnp.int32),
                        valid_state.first_invalid_offset,
                    )
                    next_state = _KernelCarry(
                        environment_state=next_environment_state,
                        observation=next_observation,
                        environment_key=next_environment_key,
                        gru_carry=outgoing,
                        agent_key=next_agent_key,
                        absolute_step=expected_time,
                        violation_mask=step_mask,
                        first_invalid_offset=first_invalid,
                    )
                    output = _KernelStepOutput(
                        observation=valid_state.observation,
                        incoming_carry=valid_state.gru_carry,
                        outgoing_carry=outgoing,
                        action_key_words=jr.key_data(action_key),
                        logits=logits,
                        action=action,
                        old_log_prob=old_log_prob,
                        old_value=value,
                        raw_reward=reward,
                        next_observation=next_observation,
                    )
                    return next_state, output

                def invalid_policy_step(
                    invalid_state: _KernelCarry,
                ) -> tuple[_KernelCarry, _KernelStepOutput]:
                    poisoned = invalid_state._replace(
                        violation_mask=jnp.uint32(VIOLATION_POLICY),
                        first_invalid_offset=offset.astype(jnp.int32),
                    )
                    return poisoned, poison_output(invalid_state)

                return cast(
                    tuple[_KernelCarry, _KernelStepOutput],
                    jax.lax.cond(
                        policy_valid,
                        valid_policy_step,
                        invalid_policy_step,
                        state,
                    ),
                )

            def poisoned_step(
                state: _KernelCarry,
            ) -> tuple[_KernelCarry, _KernelStepOutput]:
                return state, poison_output(state)

            return cast(
                tuple[_KernelCarry, _KernelStepOutput],
                jax.lax.cond(
                    current.violation_mask == jnp.uint32(0),
                    live_step,
                    poisoned_step,
                    current,
                ),
            )

        final, outputs = jax.lax.scan(
            body,
            initial,
            jnp.arange(chunk_steps, dtype=jnp.int32),
        )
        bootstrap = policy_apply(
            variables,
            final.gru_carry,
            final.observation,
            jnp.asarray(False, dtype=jnp.bool_),
        )
        if type(bootstrap) is not tuple or len(bootstrap) != 3:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled bootstrap policy result drifted"
            )
        bootstrap_carry, bootstrap_logits, bootstrap_value = bootstrap
        _require_traced_array(
            bootstrap_carry,
            shape=(hidden_size,),
            dtype=jnp.float32,
            label="compiled bootstrap carry",
        )
        _require_traced_array(
            bootstrap_logits,
            shape=(4,),
            dtype=jnp.float32,
            label="compiled bootstrap logits",
        )
        bootstrap_value = _require_traced_array(
            bootstrap_value,
            shape=(),
            dtype=jnp.float32,
            label="compiled bootstrap value",
        )
        bootstrap_valid = (
            jnp.all(jnp.isfinite(bootstrap_carry))
            & jnp.all(jnp.isfinite(bootstrap_logits))
            & jnp.isfinite(bootstrap_value)
        )
        final_mask = final.violation_mask | jnp.where(
            bootstrap_valid,
            jnp.uint32(0),
            jnp.uint32(VIOLATION_POLICY),
        )
        first_invalid = jnp.where(
            (final.violation_mask == jnp.uint32(0)) & (~bootstrap_valid),
            jnp.int32(chunk_steps),
            final.first_invalid_offset,
        )
        return _CompiledChunkResult(
            environment_state=final.environment_state,
            observation=final.observation,
            environment_key=final.environment_key,
            gru_carry=final.gru_carry,
            agent_key=final.agent_key,
            absolute_step=final.absolute_step,
            violation_mask=final_mask,
            first_invalid_offset=first_invalid,
            observations=outputs.observation,
            incoming_carries=outputs.incoming_carry,
            outgoing_carries=outputs.outgoing_carry,
            action_key_words=outputs.action_key_words,
            logits=outputs.logits,
            actions=outputs.action,
            old_log_probs=outputs.old_log_prob,
            old_values=outputs.old_value,
            raw_rewards=outputs.raw_reward,
            next_observations=outputs.next_observation,
            bootstrap_value=bootstrap_value,
        )

    return cast(_CompiledKernel, kernel)


@jax.jit
def _replay_chunk_key_schedule(
    initial_environment_key: Array,
    initial_agent_key: Array,
    logits: Array,
) -> _ChunkKeyReplay:
    """Replay both split chains and categorical behavior from recorded logits."""

    def step(
        keys: tuple[Array, Array], step_logits: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        environment_key, agent_key = keys
        next_agent_key, action_key = jr.split(agent_key)
        next_environment_key, _step_key = jr.split(environment_key)
        action = jr.categorical(
            action_key,
            step_logits,
            axis=-1,
            mode=ppo_gru.PPO_GRU_CATEGORICAL_MODE,
        ).astype(jnp.int32)
        log_probabilities = jax.nn.log_softmax(step_logits, axis=-1)
        old_log_prob = jnp.sum(
            log_probabilities * jax.nn.one_hot(action, 4, dtype=jnp.float32),
            axis=-1,
        )
        return (
            (next_environment_key, next_agent_key),
            (jr.key_data(action_key), action, old_log_prob),
        )

    (final_environment_key, final_agent_key), outputs = jax.lax.scan(
        step,
        (initial_environment_key, initial_agent_key),
        logits,
    )
    action_key_words, actions, old_log_probs = outputs
    return _ChunkKeyReplay(
        final_environment_key=final_environment_key,
        final_agent_key=final_agent_key,
        action_key_words=action_key_words,
        actions=actions,
        old_log_probs=old_log_probs,
    )


def _host_array(value: object, *, label: str) -> np.ndarray[Any, Any]:
    try:
        return cast(np.ndarray[Any, Any], np.asarray(value))
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is not host-validatable"
        ) from exc


def _host_key_words(value: object, *, label: str) -> tuple[int, int]:
    try:
        key = cast(Any, value)
        implementation = str(jr.key_impl(key))
        words = np.asarray(jr.key_data(key))
        typed = jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} is not a typed JAX key"
        ) from exc
    if (
        not typed
        or implementation != ppo_gru.PPO_GRU_PRNG_IMPLEMENTATION
        or words.shape != (2,)
        or words.dtype != np.dtype(np.uint32)
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} must be exact Threefry2x32"
        )
    return int(words[0]), int(words[1])


def _require_clean_chunk(
    result: object,
    *,
    chunk_steps: int,
    hidden_size: int,
    expected_final_step: int,
    initial_environment_key: Array,
    initial_agent_key: Array,
) -> _CompiledChunkResult:
    if type(result) is not _CompiledChunkResult:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled kernel returned the wrong result type"
        )
    exact = result
    specifications = {
        "observations": ((chunk_steps, 9, 9, 3), np.dtype(np.float32)),
        "incoming_carries": ((chunk_steps, hidden_size), np.dtype(np.float32)),
        "outgoing_carries": ((chunk_steps, hidden_size), np.dtype(np.float32)),
        "action_key_words": ((chunk_steps, 2), np.dtype(np.uint32)),
        "logits": ((chunk_steps, 4), np.dtype(np.float32)),
        "actions": ((chunk_steps,), np.dtype(np.int32)),
        "old_log_probs": ((chunk_steps,), np.dtype(np.float32)),
        "old_values": ((chunk_steps,), np.dtype(np.float32)),
        "raw_rewards": ((chunk_steps,), np.dtype(np.float32)),
        "next_observations": ((chunk_steps, 9, 9, 3), np.dtype(np.float32)),
        "observation": ((9, 9, 3), np.dtype(np.float32)),
        "gru_carry": ((hidden_size,), np.dtype(np.float32)),
        "bootstrap_value": ((), np.dtype(np.float32)),
    }
    host: dict[str, np.ndarray[Any, Any]] = {}
    for name, (shape, dtype) in specifications.items():
        array = _host_array(getattr(exact, name), label=f"compiled chunk {name}")
        if array.shape != shape or array.dtype != dtype:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                f"compiled chunk {name} shape or dtype drifted"
            )
        host[name] = array
    mask = _host_array(exact.violation_mask, label="compiled violation mask")
    invalid = _host_array(
        exact.first_invalid_offset, label="compiled first-invalid offset"
    )
    final_step = _host_array(exact.absolute_step, label="compiled final step")
    if (
        mask.shape != ()
        or mask.dtype != np.dtype(np.uint32)
        or int(mask) != 0
        or invalid.shape != ()
        or invalid.dtype != np.dtype(np.int32)
        or int(invalid) != -1
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk reported a poisoned or invalid transition"
        )
    if (
        final_step.shape != ()
        or final_step.dtype != np.dtype(np.int32)
        or int(final_step) != expected_final_step
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk final step accounting drifted"
        )
    if int(_host_array(_state_time(exact.environment_state), label="environment time")) != (
        expected_final_step
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled environment time accounting drifted"
        )
    _host_key_words(initial_environment_key, label="initial compiled environment key")
    _host_key_words(initial_agent_key, label="initial compiled agent key")
    final_environment_words = _host_key_words(
        exact.environment_key, label="compiled environment key"
    )
    final_agent_words = _host_key_words(exact.agent_key, label="compiled agent key")
    finite_names = (
        "observations",
        "incoming_carries",
        "outgoing_carries",
        "logits",
        "old_log_probs",
        "old_values",
        "raw_rewards",
        "next_observations",
        "observation",
        "gru_carry",
        "bootstrap_value",
    )
    if any(not bool(np.all(np.isfinite(host[name]))) for name in finite_names):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk contains non-finite outputs"
        )
    if (
        not bool(np.all((host["observations"] == 0.0) | (host["observations"] == 1.0)))
        or not bool(
            np.all(
                (host["next_observations"] == 0.0)
                | (host["next_observations"] == 1.0)
            )
        )
        or not bool(np.all(np.isin(host["raw_rewards"], (-1.0, 0.0, 1.0, 30.0))))
        or not bool(np.all((host["actions"] >= 0) & (host["actions"] < 4)))
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk values lie outside the exact task support"
        )
    if not np.array_equal(host["observation"], host["next_observations"][-1]):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk final observation drifted"
        )
    if not np.array_equal(host["gru_carry"], host["outgoing_carries"][-1]):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled chunk final carry drifted"
        )
    replay = _replay_chunk_key_schedule(
        initial_environment_key,
        initial_agent_key,
        exact.logits,
    )
    expected_action_key_words = _host_array(
        replay.action_key_words,
        label="replayed compiled action keys",
    )
    expected_actions = _host_array(
        replay.actions,
        label="replayed compiled actions",
    )
    expected_old_log_probs = _host_array(
        replay.old_log_probs,
        label="replayed compiled old log probabilities",
    )
    if not np.array_equal(host["action_key_words"], expected_action_key_words):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled action-key split chain drifted"
        )
    if not np.array_equal(host["actions"], expected_actions):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled categorical action replay drifted"
        )
    if not np.array_equal(host["old_log_probs"], expected_old_log_probs):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled categorical log-probability replay drifted"
        )
    if final_environment_words != _host_key_words(
        replay.final_environment_key,
        label="replayed final compiled environment key",
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled environment-key split-chain endpoint drifted"
        )
    if final_agent_words != _host_key_words(
        replay.final_agent_key,
        label="replayed final compiled agent key",
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled agent-key split-chain endpoint drifted"
        )
    return exact


def _canonical_array_bytes(value: object, *, label: str) -> bytes:
    array = _host_array(value, label=label)
    if array.dtype.hasobject:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} cannot contain objects"
        )
    little = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    shape = b"".join(int(dimension).to_bytes(8, "big") for dimension in little.shape)
    dtype = little.dtype.str.encode("ascii")
    return bytes(
        len(dtype).to_bytes(4, "big")
        + dtype
        + little.ndim.to_bytes(4, "big")
        + shape
        + little.nbytes.to_bytes(8, "big")
        + little.tobytes(order="C")
    )


def _chunk_trace_digest(result: _CompiledChunkResult, *, rollout_index: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(_TRACE_DOMAIN)
    digest.update(rollout_index.to_bytes(8, "big"))
    for name in (
        "observations",
        "incoming_carries",
        "outgoing_carries",
        "action_key_words",
        "logits",
        "actions",
        "old_log_probs",
        "old_values",
        "raw_rewards",
        "next_observations",
        "bootstrap_value",
    ):
        label = name.encode("ascii")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(_canonical_array_bytes(getattr(result, name), label=name))
    return digest.digest()


class _RuntimeCapability:
    __slots__ = ("__weakref__",)


class _OutcomeCapability:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PPOGRUCompiledRuntime:
    """Single-use, PID-bound runtime for one exact compiled trajectory."""

    bridge_runtime: bridge.MatchedV3ForagaxRuntime
    runtime_identity_bytes: bytes
    _kernel: _CompiledKernel = field(repr=False, compare=False)
    _capability: _RuntimeCapability = field(repr=False, compare=False)
    _pid: int = field(repr=False, compare=False)

    def __copy__(self) -> NoReturn:
        raise TypeError("PPOGRUCompiledRuntime cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("PPOGRUCompiledRuntime cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("PPOGRUCompiledRuntime cannot be pickled")


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PPOGRUCompiledOutcome:
    """Capability-backed result of exactly one completed full-horizon invocation."""

    raw_reward_trace: bytes
    raw_cumulative_score: int
    interactions: int
    rollout_count: int
    optimizer_update_count: int
    total_agent_draw_count: int
    bridge_environment_key_use_count: int
    trace_chain_sha256: str
    runtime_identity_bytes: bytes
    receipt_bytes: bytes
    production_runtime: bool
    _capability: _OutcomeCapability = field(repr=False, compare=False)
    _pid: int = field(repr=False, compare=False)

    def __copy__(self) -> NoReturn:
        raise TypeError("PPOGRUCompiledOutcome cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("PPOGRUCompiledOutcome cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("PPOGRUCompiledOutcome cannot be pickled")


@dataclass(slots=True)
class _RuntimeBinding:
    runtime_ref: weakref.ReferenceType[PPOGRUCompiledRuntime]
    bridge_runtime: bridge.MatchedV3ForagaxRuntime
    runtime_identity_bytes: bytes
    runtime_identity_sha256: str
    kernel: _CompiledKernel
    pid: int
    claimed: bool = False
    in_flight: bool = False
    poisoned: bool = False
    completed: bool = False


@dataclass(frozen=True, slots=True)
class _OutcomeBinding:
    outcome_ref: weakref.ReferenceType[PPOGRUCompiledOutcome]
    runtime_capability: _RuntimeCapability
    outcome_identity_sha256: str
    receipt_sha256: str
    pid: int


_REGISTRY_LOCK: Final = threading.RLock()
_RUNTIME_REGISTRY: Final = weakref.WeakKeyDictionary[_RuntimeCapability, _RuntimeBinding]()
_OUTCOME_REGISTRY: Final = weakref.WeakKeyDictionary[_OutcomeCapability, _OutcomeBinding]()


def _runtime_identity(runtime: bridge.MatchedV3ForagaxRuntime) -> dict[str, Any]:
    if type(runtime) is not bridge.MatchedV3ForagaxRuntime:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "bridge runtime must have the exact bound type"
        )
    return {
        "schema_version": PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "classification": "observed_compiled_runtime_unqualified_non_authorizing",
        "bindings": {
            "compiled_runner_descriptor_sha256": (
                PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
            ),
            "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
            "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
            "core_configuration_sha256": BOUND_CORE_CONFIGURATION_SHA256,
            "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
            "foragax_install_tree_sha256": BOUND_FORAGAX_INSTALL_TREE_SHA256,
        },
        "runtime": asdict(runtime.runtime_identity),
        "kernel": {
            "chunk_steps": PPO_GRU_COMPILED_CHUNK_STEPS,
            "constructed": True,
            "full_horizon_executed": False,
            "runtime_qualified": False,
        },
        "claims": _claims(),
    }


def _build_exact_production_kernel(
    runtime: bridge.MatchedV3ForagaxRuntime,
) -> _CompiledKernel:
    binding = bridge._validate_runtime_handle(runtime)
    config = ppo_gru.matched_v3_ppo_gru_configuration()
    model = ppo_gru.PPOGRUActorCritic(
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )

    def policy_apply(
        variables: Any, carry: Array, observation: Array, reset_before: Array
    ) -> tuple[Array, Array, Array]:
        return cast(
            tuple[Array, Array, Array],
            model.apply(variables, carry, observation, reset_before),
        )

    return _build_chunk_kernel(
        environment_step=binding.environment.step,
        environment_params=binding.params,
        policy_apply=policy_apply,
        hidden_size=config.hidden_size,
        chunk_steps=PPO_GRU_COMPILED_CHUNK_STEPS,
    )


def _register_runtime(
    *,
    bridge_runtime: bridge.MatchedV3ForagaxRuntime,
    kernel: _CompiledKernel,
    runtime_identity_bytes: bytes,
) -> PPOGRUCompiledRuntime:
    capability = _RuntimeCapability()
    pid = os.getpid()
    runtime = PPOGRUCompiledRuntime(
        bridge_runtime=bridge_runtime,
        runtime_identity_bytes=bytes(runtime_identity_bytes),
        _kernel=kernel,
        _capability=capability,
        _pid=pid,
    )
    binding = _RuntimeBinding(
        runtime_ref=weakref.ref(runtime),
        bridge_runtime=bridge_runtime,
        runtime_identity_bytes=bytes(runtime_identity_bytes),
        runtime_identity_sha256=hashlib.sha256(runtime_identity_bytes).hexdigest(),
        kernel=kernel,
        pid=pid,
    )
    with _REGISTRY_LOCK:
        _RUNTIME_REGISTRY[capability] = binding
    return runtime


def open_matched_v3_ppo_gru_compiled_runtime() -> PPOGRUCompiledRuntime:
    """Open one unqualified single-use runtime without compiling or executing a chunk."""

    bridge_runtime = bridge.open_matched_v3_foragax_runtime()
    kernel = _build_exact_production_kernel(bridge_runtime)
    identity_bytes = _canonical_json(
        _runtime_identity(bridge_runtime),
        label="compiled PPO-GRU runtime identity",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    return _register_runtime(
        bridge_runtime=bridge_runtime,
        kernel=kernel,
        runtime_identity_bytes=identity_bytes,
    )


def _validated_runtime_binding(runtime: object) -> _RuntimeBinding:
    if type(runtime) is not PPOGRUCompiledRuntime:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "runtime must be an exact PPOGRUCompiledRuntime"
        )
    exact = runtime
    pid = os.getpid()
    with _REGISTRY_LOCK:
        binding = _RUNTIME_REGISTRY.get(exact._capability)
        if (
            binding is None
            or binding.runtime_ref() is not exact
            or exact._pid != pid
            or binding.pid != pid
            or exact.bridge_runtime is not binding.bridge_runtime
            or exact._kernel is not binding.kernel
            or exact.runtime_identity_bytes != binding.runtime_identity_bytes
            or not hmac.compare_digest(
                hashlib.sha256(exact.runtime_identity_bytes).hexdigest(),
                binding.runtime_identity_sha256,
            )
        ):
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled runtime capability is stale, forked, copied, or forged"
            )
        if binding.poisoned:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled runtime capability is poisoned"
            )
        return binding


def _claim_runtime(runtime: PPOGRUCompiledRuntime) -> _RuntimeBinding:
    binding = _validated_runtime_binding(runtime)
    with _REGISTRY_LOCK:
        if binding.claimed or binding.in_flight or binding.completed:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled runtime is single-use and has already been claimed"
            )
        binding.claimed = True
        binding.in_flight = True
    return binding


def _poison_runtime(binding: _RuntimeBinding) -> None:
    with _REGISTRY_LOCK:
        binding.poisoned = True
        binding.in_flight = False


def _validate_uint31(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAXIMUM:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            f"{label} must be an exact uint31"
        )
    return value


def _require_execution_flag(value: object) -> None:
    if type(value) is not bool or not value:
        raise PPOGRUCompiledExecutionBlockedError(
            "compiled PPO-GRU execution requires explicit "
            "unqualified_engineering=True; it grants no authority"
        )


def _reward_trace_bytes(rewards: object) -> bytes:
    values = _host_array(rewards, label="compiled raw rewards")
    if values.ndim != 1 or values.dtype != np.dtype(np.float32):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled raw rewards must be one float32 vector"
        )
    if not bool(np.all(np.isin(values, (-1.0, 0.0, 1.0, 30.0)))):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled raw reward lies outside exact support"
        )
    return values.astype(np.int8, copy=False).tobytes(order="C")


def _runtime_identity_from_bytes(raw: bytes) -> dict[str, Any]:
    identity = _strict_json_object(
        raw,
        label="compiled PPO-GRU runtime identity",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    expected_keys = {
        "schema_version",
        "classification",
        "bindings",
        "runtime",
        "kernel",
        "claims",
    }
    if set(identity) != expected_keys:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled runtime identity schema drifted"
        )
    expected_bindings = {
        "compiled_runner_descriptor_sha256": PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256,
        "bridge_descriptor_sha256": BOUND_BRIDGE_DESCRIPTOR_SHA256,
        "bridge_implementation_sha256": BOUND_BRIDGE_IMPLEMENTATION_SHA256,
        "core_configuration_sha256": BOUND_CORE_CONFIGURATION_SHA256,
        "core_implementation_sha256": BOUND_CORE_IMPLEMENTATION_SHA256,
        "foragax_install_tree_sha256": BOUND_FORAGAX_INSTALL_TREE_SHA256,
    }
    if (
        identity["schema_version"] != PPO_GRU_COMPILED_RUNTIME_IDENTITY_SCHEMA_VERSION
        or identity["classification"]
        != "observed_compiled_runtime_unqualified_non_authorizing"
        or identity["bindings"] != expected_bindings
        or identity["claims"] != _claims()
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled runtime identity binding drifted"
        )
    runtime = identity["runtime"]
    kernel = identity["kernel"]
    expected_runtime_keys = {
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
    if (
        type(runtime) is not dict
        or set(runtime) != expected_runtime_keys
        or runtime["jax_version"] != bridge.JAX_REQUIRED_VERSION
        or runtime["jaxlib_version"] != bridge.JAXLIB_REQUIRED_VERSION
        or runtime["default_prng_impl"] != bridge.THREEFRY_IMPLEMENTATION
        or runtime["threefry_partitionable"] is not True
        or runtime["jax_enable_x64"] is not False
        or type(runtime["backend"]) is not str
        or not runtime["backend"]
        or runtime["foragax_version"] != bridge.FORAGAX_REQUIRED_VERSION
        or runtime["foragax_install_tree_sha256"]
        != BOUND_FORAGAX_INSTALL_TREE_SHA256
        or type(runtime["foragax_package_root"]) is not str
        or not runtime["foragax_package_root"]
        or runtime.get("runtime_qualified") is not False
        or type(kernel) is not dict
        or kernel
        != {
            "chunk_steps": PPO_GRU_COMPILED_CHUNK_STEPS,
            "constructed": True,
            "full_horizon_executed": False,
            "runtime_qualified": False,
        }
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled runtime remains unqualified by contract"
        )
    _reject_authority_anywhere(identity, label="compiled runtime identity")
    return identity


def _receipt_body(
    *,
    environment_seed: int,
    agent_seed: int,
    runtime_identity_bytes: bytes,
    raw_reward_trace: bytes,
    raw_cumulative_score: int,
    trace_chain_sha256: str,
) -> dict[str, Any]:
    environment_seed = _validate_uint31(environment_seed, label="environment_seed")
    agent_seed = _validate_uint31(agent_seed, label="agent_seed")
    if type(raw_reward_trace) is not bytes or len(raw_reward_trace) != MATCHED_V3_HORIZON:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "raw_reward_trace must be exact full-horizon bytes"
        )
    reward_values = np.frombuffer(raw_reward_trace, dtype=np.int8)
    if not bool(np.all(np.isin(reward_values, (-1, 0, 1, 30)))):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "raw_reward_trace contains a value outside the task support"
        )
    expected_score = int(reward_values.sum(dtype=np.int64))
    if type(raw_cumulative_score) is not int or raw_cumulative_score != expected_score:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "raw_cumulative_score disagrees with the exact reward trace"
        )
    runtime_identity = _runtime_identity_from_bytes(runtime_identity_bytes)
    return {
        "schema_version": PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION,
        "candidate_id": "adapted_ppo_gru",
        "status": "completed_runtime_unqualified",
        "classification": "compiled_full_horizon_non_authorizing",
        "compiled_runner_descriptor_sha256": PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256,
        "implementation": {
            "path": PPO_GRU_COMPILED_RUNNER_IMPLEMENTATION_PATH,
            "source_self_hash_bound": False,
            "source_digest_requires_external_binding": True,
        },
        "bindings": cast(dict[str, Any], copy.deepcopy(_DESCRIPTOR["bindings"])),
        "seeds": {
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
            "domain": "uint31",
            "source": "caller_supplied_unverified",
            "protected_seed_status": "unknown_not_asserted",
        },
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": hashlib.sha256(runtime_identity_bytes).hexdigest(),
        "accounting": matched_v3_ppo_gru_compiled_accounting(),
        "completion": {
            "exact_horizon_complete": True,
            "full_horizon_compiled_execution": True,
            "production_runtime_complete": True,
            "violation_mask": 0,
            "first_invalid_offset": None,
            "content_independently_proves_execution": False,
        },
        "score": {
            "raw_reward_trace_encoding": "signed_int8_twos_complement",
            "raw_reward_trace_length": len(raw_reward_trace),
            "raw_reward_trace_sha256": hashlib.sha256(raw_reward_trace).hexdigest(),
            "raw_cumulative_score": raw_cumulative_score,
            "reward_scaling_applied": False,
        },
        "trace_chain_sha256": _require_sha256(
            trace_chain_sha256, label="trace_chain_sha256"
        ),
        "claims": _claims(),
        "limitations": list(_RECEIPT_LIMITATIONS),
    }


def _receipt_bytes_from_fields(
    *,
    environment_seed: int,
    agent_seed: int,
    runtime_identity_bytes: bytes,
    raw_reward_trace: bytes,
    raw_cumulative_score: int,
    trace_chain_sha256: str,
) -> bytes:
    body = _receipt_body(
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        runtime_identity_bytes=runtime_identity_bytes,
        raw_reward_trace=raw_reward_trace,
        raw_cumulative_score=raw_cumulative_score,
        trace_chain_sha256=trace_chain_sha256,
    )
    _reject_authority_anywhere(body, label="compiled result receipt")
    receipt = dict(body)
    receipt["receipt_body_sha256"] = hashlib.sha256(
        _canonical_json(
            body,
            label="compiled result receipt body",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
    ).hexdigest()
    return _canonical_json(
        receipt,
        label="compiled result receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )


def _outcome_identity(outcome: PPOGRUCompiledOutcome) -> str:
    digest = hashlib.sha256()
    digest.update(_OUTCOME_DOMAIN)
    for value in (
        outcome.raw_reward_trace,
        outcome.raw_cumulative_score.to_bytes(8, "big", signed=True),
        outcome.interactions.to_bytes(8, "big"),
        outcome.rollout_count.to_bytes(8, "big"),
        outcome.optimizer_update_count.to_bytes(8, "big"),
        outcome.total_agent_draw_count.to_bytes(8, "big"),
        outcome.bridge_environment_key_use_count.to_bytes(8, "big"),
        outcome.trace_chain_sha256.encode("ascii"),
        outcome.runtime_identity_bytes,
        outcome.receipt_bytes,
        b"\x01" if outcome.production_runtime else b"\x00",
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _validate_outcome_capability(outcome: object) -> PPOGRUCompiledOutcome:
    if type(outcome) is not PPOGRUCompiledOutcome:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "outcome must be an exact PPOGRUCompiledOutcome"
        )
    exact = outcome
    pid = os.getpid()
    with _REGISTRY_LOCK:
        binding = _OUTCOME_REGISTRY.get(exact._capability)
        runtime = _RUNTIME_REGISTRY.get(binding.runtime_capability) if binding else None
    if (
        binding is None
        or binding.outcome_ref() is not exact
        or binding.pid != pid
        or exact._pid != pid
        or runtime is None
        or not runtime.claimed
        or runtime.in_flight
        or not runtime.completed
        or runtime.poisoned
        or not hmac.compare_digest(
            exact.runtime_identity_bytes,
            runtime.runtime_identity_bytes,
        )
        or not hmac.compare_digest(binding.outcome_identity_sha256, _outcome_identity(exact))
        or not hmac.compare_digest(
            binding.receipt_sha256, hashlib.sha256(exact.receipt_bytes).hexdigest()
        )
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled outcome capability is stale, forked, copied, or forged"
        )
    if (
        len(exact.raw_reward_trace) != MATCHED_V3_HORIZON
        or exact.raw_cumulative_score
        != int(np.frombuffer(exact.raw_reward_trace, dtype=np.int8).sum(dtype=np.int64))
        or exact.interactions != MATCHED_V3_HORIZON
        or exact.rollout_count != PPO_GRU_COMPILED_CHUNK_COUNT
        or exact.optimizer_update_count != PPO_GRU_OPTIMIZER_UPDATES
        or exact.total_agent_draw_count != PPO_GRU_TOTAL_AGENT_DRAWS
        or exact.bridge_environment_key_use_count != PPO_GRU_BRIDGE_KEY_USES
        or exact.production_runtime is not True
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled outcome accounting drifted"
        )
    receipt = parse_ppo_gru_compiled_result_receipt(
        exact.receipt_bytes,
        expected_receipt_sha256=binding.receipt_sha256,
    )
    receipt_runtime = _canonical_json(
        receipt["runtime_identity"],
        label="compiled outcome runtime identity",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    score = cast(dict[str, Any], receipt["score"])
    if (
        not hmac.compare_digest(receipt_runtime, exact.runtime_identity_bytes)
        or score["raw_cumulative_score"] != exact.raw_cumulative_score
        or not hmac.compare_digest(
            score["raw_reward_trace_sha256"],
            hashlib.sha256(exact.raw_reward_trace).hexdigest(),
        )
        or not hmac.compare_digest(
            receipt["trace_chain_sha256"], exact.trace_chain_sha256
        )
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled outcome and receipt identities disagree"
        )
    return exact


def run_matched_v3_ppo_gru_compiled(
    *,
    environment_seed: object,
    agent_seed: object,
    runtime: PPOGRUCompiledRuntime,
    unqualified_engineering: bool = False,
) -> PPOGRUCompiledOutcome:
    """Run the exact horizon only after explicit non-authorizing acknowledgement."""

    _require_execution_flag(unqualified_engineering)
    exact_environment_seed = _validate_uint31(environment_seed, label="environment_seed")
    exact_agent_seed = _validate_uint31(agent_seed, label="agent_seed")
    binding = _claim_runtime(runtime)
    try:
        _runtime_identity_from_bytes(binding.runtime_identity_bytes)
        config = ppo_gru.matched_v3_ppo_gru_configuration()
        model = ppo_gru.PPOGRUActorCritic(
            hidden_size=config.hidden_size,
            num_actions=config.num_actions,
        )
        rng_state = ppo_gru.initialize_ppo_gru_rng_state(
            exact_environment_seed, exact_agent_seed
        )
        initial_environment_key_words = _host_key_words(
            rng_state.environment_key,
            label="initial PPO environment key",
        )
        train_state, rng_state = ppo_gru.initialize_ppo_gru_train_state(
            config,
            rng_state=rng_state,
        )
        environment_draws, agent_draws = ppo_gru.validate_ppo_gru_rng_state(rng_state)
        if environment_draws != 0 or agent_draws != 1:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "parameter initialization RNG accounting drifted"
            )
        initial_bridge_state = binding.bridge_runtime.initialize(exact_environment_seed)
        if type(initial_bridge_state) is not bridge.MatchedV3ForagaxBridgeState:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "bridge reset returned the wrong exact state type"
            )
        environment_state = initial_bridge_state._environment_state
        observation = initial_bridge_state.observation
        environment_key = initial_bridge_state._environment_key
        if initial_bridge_state.reset_count != 1 or initial_bridge_state.step_count != 0:
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "bridge reset accounting drifted"
            )
        carry = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
        raw_trace = bytearray()
        raw_score = 0
        trace_chain = hashlib.sha256(_TRACE_DOMAIN).digest()
        completed_steps = 0
        expected_updates = 0

        for rollout_index in range(PPO_GRU_COMPILED_CHUNK_COUNT):
            _validated_runtime_binding(runtime)
            initial_carry = carry
            initial_chunk_environment_key = environment_key
            initial_chunk_agent_key = rng_state.agent_key
            chunk = binding.kernel(
                train_state.variables,
                environment_state,
                observation,
                environment_key,
                carry,
                rng_state.agent_key,
                jnp.asarray(completed_steps, dtype=jnp.int32),
            )
            completed_steps += PPO_GRU_COMPILED_CHUNK_STEPS
            chunk = _require_clean_chunk(
                chunk,
                chunk_steps=PPO_GRU_COMPILED_CHUNK_STEPS,
                hidden_size=config.hidden_size,
                expected_final_step=completed_steps,
                initial_environment_key=initial_chunk_environment_key,
                initial_agent_key=initial_chunk_agent_key,
            )
            chunk_trace = _reward_trace_bytes(chunk.raw_rewards)
            raw_trace.extend(chunk_trace)
            raw_score += int(np.frombuffer(chunk_trace, dtype=np.int8).sum(dtype=np.int64))
            trace_chain = hashlib.sha256(
                trace_chain + _chunk_trace_digest(chunk, rollout_index=rollout_index)
            ).digest()
            rng_state = replace(
                rng_state,
                agent_key=chunk.agent_key,
                agent_draw_count=(
                    rng_state.agent_draw_count
                    + jnp.uint32(PPO_GRU_COMPILED_CHUNK_STEPS)
                ),
            )
            rollout = ppo_gru.PPOGRURollout(
                initial_carry=initial_carry,
                observations=chunk.observations,
                reset_before=jnp.zeros(
                    (PPO_GRU_COMPILED_CHUNK_STEPS,), dtype=jnp.bool_
                ),
                actions=chunk.actions,
                rewards=chunk.raw_rewards,
                transition_dones=jnp.zeros(
                    (PPO_GRU_COMPILED_CHUNK_STEPS,), dtype=jnp.bool_
                ),
                old_log_probs=chunk.old_log_probs,
                old_values=chunk.old_values,
                incoming_carries=chunk.incoming_carries,
                bootstrap_observation=chunk.observation,
                bootstrap_value=chunk.bootstrap_value,
            )
            advantages, targets = ppo_gru.validate_ppo_gru_rollout(
                model,
                train_state.variables,
                rollout,
                config,
                expected_initial_carry=initial_carry,
                expected_initial_reset=False,
            )
            for _epoch in range(PPO_GRU_UPDATE_EPOCHS):
                rng_state, order = ppo_gru.next_ppo_gru_segment_order(rng_state)
                segments = ppo_gru.build_ppo_gru_sequence_segments(
                    rollout,
                    advantages,
                    targets,
                    segment_steps=PPO_GRU_SEGMENT_STEPS,
                    segment_order=order,
                )
                for position in range(PPO_GRU_SEGMENTS_PER_ROLLOUT):
                    batch = ppo_gru.ppo_gru_loss_batch_from_segment(
                        segments, position, config
                    )
                    result = ppo_gru.ppo_gru_update(model, train_state, batch, config)
                    train_state = result.state
                    expected_updates += 1
                    if int(train_state.optimizer_updates) != expected_updates:
                        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                            "compiled optimizer update accounting drifted"
                        )
            environment_state = chunk.environment_state
            observation = chunk.observation
            environment_key = chunk.environment_key
            carry = chunk.gru_carry

        environment_draws, agent_draws = ppo_gru.validate_ppo_gru_rng_state(rng_state)
        if (
            completed_steps != MATCHED_V3_HORIZON
            or len(raw_trace) != MATCHED_V3_HORIZON
            or expected_updates != PPO_GRU_OPTIMIZER_UPDATES
            or environment_draws != 0
            or agent_draws != PPO_GRU_TOTAL_AGENT_DRAWS
            or _host_key_words(rng_state.environment_key, label="final PPO environment key")
            != initial_environment_key_words
            or int(_host_array(_state_time(environment_state), label="final environment time"))
            != MATCHED_V3_HORIZON
        ):
            raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                "compiled full-horizon closure accounting drifted"
            )
        trace_bytes = bytes(raw_trace)
        receipt_bytes = _receipt_bytes_from_fields(
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            runtime_identity_bytes=binding.runtime_identity_bytes,
            raw_reward_trace=trace_bytes,
            raw_cumulative_score=raw_score,
            trace_chain_sha256=trace_chain.hex(),
        )
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        parse_ppo_gru_compiled_result_receipt(
            receipt_bytes,
            expected_receipt_sha256=receipt_sha256,
        )
        outcome = PPOGRUCompiledOutcome(
            raw_reward_trace=trace_bytes,
            raw_cumulative_score=raw_score,
            interactions=MATCHED_V3_HORIZON,
            rollout_count=PPO_GRU_COMPILED_CHUNK_COUNT,
            optimizer_update_count=PPO_GRU_OPTIMIZER_UPDATES,
            total_agent_draw_count=PPO_GRU_TOTAL_AGENT_DRAWS,
            bridge_environment_key_use_count=PPO_GRU_BRIDGE_KEY_USES,
            trace_chain_sha256=trace_chain.hex(),
            runtime_identity_bytes=binding.runtime_identity_bytes,
            receipt_bytes=receipt_bytes,
            production_runtime=True,
            _capability=_OutcomeCapability(),
            _pid=os.getpid(),
        )
        outcome_binding = _OutcomeBinding(
            outcome_ref=weakref.ref(outcome),
            runtime_capability=runtime._capability,
            outcome_identity_sha256=_outcome_identity(outcome),
            receipt_sha256=receipt_sha256,
            pid=os.getpid(),
        )
        with _REGISTRY_LOCK:
            final_binding = _validated_runtime_binding(runtime)
            if (
                final_binding is not binding
                or not binding.claimed
                or not binding.in_flight
                or binding.completed
                or outcome._capability in _OUTCOME_REGISTRY
            ):
                raise ForagerMatchedV3PPOGRUCompiledRunnerError(
                    "compiled runtime registry changed before exact-horizon completion"
                )
            binding.in_flight = False
            binding.completed = True
            _OUTCOME_REGISTRY[outcome._capability] = outcome_binding
        _validate_outcome_capability(outcome)
        return outcome
    except BaseException:
        _poison_runtime(binding)
        raise


def canonical_ppo_gru_compiled_result_receipt_bytes(
    outcome: PPOGRUCompiledOutcome,
) -> bytes:
    """Return receipt bytes only for the exact live completed outcome capability."""

    exact = _validate_outcome_capability(outcome)
    return bytes(exact.receipt_bytes)


def parse_ppo_gru_compiled_result_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Strictly parse v2 bytes under a caller-carried full-file digest."""

    expected_digest = _require_sha256(
        expected_receipt_sha256,
        label="expected_receipt_sha256",
    )
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_digest
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result receipt full-file digest disagrees"
        )
    receipt = _strict_json_object(
        raw,
        label="compiled PPO-GRU result receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    expected_keys = {
        "schema_version",
        "candidate_id",
        "status",
        "classification",
        "compiled_runner_descriptor_sha256",
        "implementation",
        "bindings",
        "seeds",
        "runtime_identity",
        "runtime_identity_sha256",
        "accounting",
        "completion",
        "score",
        "trace_chain_sha256",
        "claims",
        "limitations",
        "receipt_body_sha256",
    }
    if set(receipt) != expected_keys:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result receipt schema keys drifted"
        )
    supplied_body_digest = _require_sha256(
        receipt["receipt_body_sha256"], label="receipt_body_sha256"
    )
    body = dict(receipt)
    del body["receipt_body_sha256"]
    calculated_body_digest = hashlib.sha256(
        _canonical_json(
            body,
            label="compiled result receipt body",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
    ).hexdigest()
    if not hmac.compare_digest(supplied_body_digest, calculated_body_digest):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result receipt body digest drifted"
        )
    if (
        receipt["schema_version"] != PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION
        or receipt["candidate_id"] != "adapted_ppo_gru"
        or receipt["status"] != "completed_runtime_unqualified"
        or receipt["classification"] != "compiled_full_horizon_non_authorizing"
        or receipt["compiled_runner_descriptor_sha256"]
        != PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256
        or receipt["implementation"]
        != {
            "path": PPO_GRU_COMPILED_RUNNER_IMPLEMENTATION_PATH,
            "source_self_hash_bound": False,
            "source_digest_requires_external_binding": True,
        }
        or receipt["bindings"] != _DESCRIPTOR["bindings"]
        or receipt["accounting"] != matched_v3_ppo_gru_compiled_accounting()
        or receipt["claims"] != _claims()
        or receipt["limitations"] != list(_RECEIPT_LIMITATIONS)
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result receipt identity drifted"
        )
    seeds = receipt["seeds"]
    if (
        type(seeds) is not dict
        or set(seeds)
        != {
            "environment_seed",
            "agent_seed",
            "domain",
            "source",
            "protected_seed_status",
        }
        or type(seeds["environment_seed"]) is not int
        or type(seeds["agent_seed"]) is not int
        or not 0 <= seeds["environment_seed"] <= _UINT31_MAXIMUM
        or not 0 <= seeds["agent_seed"] <= _UINT31_MAXIMUM
        or seeds["domain"] != "uint31"
        or seeds["source"] != "caller_supplied_unverified"
        or seeds["protected_seed_status"] != "unknown_not_asserted"
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result receipt seed fields drifted"
        )
    runtime_identity = receipt["runtime_identity"]
    if type(runtime_identity) is not dict:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result runtime identity must be an object"
        )
    runtime_raw = _canonical_json(
        runtime_identity,
        label="compiled receipt runtime identity",
        maximum_bytes=_MAX_DESCRIPTOR_BYTES,
    )
    _runtime_identity_from_bytes(runtime_raw)
    runtime_digest = _require_sha256(
        receipt["runtime_identity_sha256"], label="runtime_identity_sha256"
    )
    if not hmac.compare_digest(runtime_digest, hashlib.sha256(runtime_raw).hexdigest()):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled runtime identity digest drifted"
        )
    completion = receipt["completion"]
    if completion != {
        "exact_horizon_complete": True,
        "full_horizon_compiled_execution": True,
        "production_runtime_complete": True,
        "violation_mask": 0,
        "first_invalid_offset": None,
        "content_independently_proves_execution": False,
    }:
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result completion fields drifted"
        )
    score = receipt["score"]
    if (
        type(score) is not dict
        or set(score)
        != {
            "raw_reward_trace_encoding",
            "raw_reward_trace_length",
            "raw_reward_trace_sha256",
            "raw_cumulative_score",
            "reward_scaling_applied",
        }
        or score["raw_reward_trace_encoding"] != "signed_int8_twos_complement"
        or score["raw_reward_trace_length"] != MATCHED_V3_HORIZON
        or type(score["raw_cumulative_score"]) is not int
        or not -MATCHED_V3_HORIZON
        <= score["raw_cumulative_score"]
        <= 30 * MATCHED_V3_HORIZON
        or score["reward_scaling_applied"] is not False
    ):
        raise ForagerMatchedV3PPOGRUCompiledRunnerError(
            "compiled result score fields drifted"
        )
    _require_sha256(score["raw_reward_trace_sha256"], label="raw_reward_trace_sha256")
    _require_sha256(receipt["trace_chain_sha256"], label="trace_chain_sha256")
    _reject_authority_anywhere(receipt, label="compiled result receipt")
    return receipt


__all__ = [
    "BOUND_BRIDGE_DESCRIPTOR_SHA256",
    "BOUND_BRIDGE_IMPLEMENTATION_PATH",
    "BOUND_BRIDGE_IMPLEMENTATION_SHA256",
    "BOUND_CORE_CONFIGURATION_SHA256",
    "BOUND_CORE_IMPLEMENTATION_PATH",
    "BOUND_CORE_IMPLEMENTATION_SHA256",
    "BOUND_CORE_SOURCE_DESCRIPTOR_SHA256",
    "BOUND_FORAGAX_INSTALL_TREE_SHA256",
    "BOUND_V1_RUNNER_DESCRIPTOR_SHA256",
    "BOUND_V1_RUNNER_IMPLEMENTATION_PATH",
    "BOUND_V1_RUNNER_IMPLEMENTATION_SHA256",
    "ForagerMatchedV3PPOGRUCompiledRunnerError",
    "MATCHED_V3_HORIZON",
    "PPOGRUCompiledExecutionBlockedError",
    "PPOGRUCompiledOutcome",
    "PPOGRUCompiledRuntime",
    "PPO_GRU_COMPILED_CHUNK_COUNT",
    "PPO_GRU_COMPILED_CHUNK_STEPS",
    "PPO_GRU_COMPILED_RESULT_RECEIPT_SCHEMA_VERSION",
    "PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SCHEMA_VERSION",
    "PPO_GRU_COMPILED_RUNNER_DESCRIPTOR_SHA256",
    "PPO_GRU_COMPILED_RUNNER_STATUS",
    "PPO_GRU_OPTIMIZER_UPDATES",
    "PPO_GRU_TOTAL_AGENT_DRAWS",
    "VIOLATION_ACTION",
    "VIOLATION_DONE",
    "VIOLATION_ENVIRONMENT_TIME",
    "VIOLATION_INFO",
    "VIOLATION_OBSERVATION",
    "VIOLATION_POLICY",
    "VIOLATION_REWARD",
    "canonical_matched_v3_ppo_gru_compiled_runner_descriptor_bytes",
    "canonical_ppo_gru_compiled_result_receipt_bytes",
    "matched_v3_ppo_gru_compiled_accounting",
    "matched_v3_ppo_gru_compiled_runner_descriptor",
    "open_matched_v3_ppo_gru_compiled_runtime",
    "parse_matched_v3_ppo_gru_compiled_runner_descriptor",
    "parse_ppo_gru_compiled_result_receipt",
    "run_matched_v3_ppo_gru_compiled",
]
