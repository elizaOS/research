"""POBAX-derived PPO-GRU core for the future matched-v3 Forager panel.

Derived from POBAX commit ``a5e1d62d14e4efe783885b9d4f19cffa2a568eec``
(Apache-2.0).  This file is a changed exact-task adaptation: it binds one
continuing Forager trajectory, explicit logically separate environment and agent
RNG consumption chains, native-JAX categorical operations (no Distrax), and contiguous
time-segment recurrent minibatches with their actual incoming GRU carries.

The implemented surface is deliberately a core/configuration contract, not a
qualified benchmark runner.  It does not construct Foragax, execute the
499,712-transition workload, write an artifact, or grant execution/scientific
authority.  The source descriptor lists those remaining blockers explicitly.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Final, cast

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal

PPO_GRU_CONFIGURATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_configuration.v1"
)
PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_ppo_gru_source.v1"
)

POBAX_CANONICAL_URL: Final = "https://github.com/taodav/pobax"
POBAX_COMMIT_GIT_SHA1: Final = "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
POBAX_TREE_GIT_SHA1: Final = "d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6"
POBAX_ARCHIVE_SHA256: Final = (
    "f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9"
)
POBAX_ARCHIVE_SIZE_BYTES: Final = 1_699_840
POBAX_LICENSE: Final = "Apache-2.0"
PPO_GRU_PRNG_IMPLEMENTATION: Final = "threefry2x32"
PPO_GRU_CATEGORICAL_MODE: Final = "low"

REQUIRED_POBAX_SOURCE_SHA256_BY_PATH: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "pobax/algos/ppo.py": (
            "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153"
        ),
        "pobax/config.py": (
            "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c"
        ),
        "pobax/models/actor_critic.py": (
            "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f"
        ),
        "pobax/models/discrete.py": (
            "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c"
        ),
        "pobax/models/__init__.py": (
            "c4434b0b1eba13c227cdf479380f5347aa57aba4d2f78a12112c056cdada323a"
        ),
        "pobax/models/network.py": (
            "b3ea151f6a7f9000dd1b529cbcc262c150b767c66664399008aa89283a2e520a"
        ),
        "pobax/models/value.py": (
            "e875e7ef951aba37ea4648328442aaece0fc3415de580c6b5115843eb32366bd"
        ),
        "pyproject.toml": (
            "4f02e96a5d8471f9637ec36dc9536398183f49fb28fa07c5b7f371ffcdbe81d5"
        ),
        "requirements.txt": (
            "8d8a36a4428d481b15c47b9ed1aec573c3dc2472af746be611e9a17dae40a17c"
        ),
    }
)
REQUIRED_POBAX_SOURCE_SIZE_BYTES_BY_PATH: Final[Mapping[str, int]] = MappingProxyType(
    {
        "LICENSE": 11_357,
        "pobax/algos/ppo.py": 19_864,
        "pobax/config.py": 7_047,
        "pobax/models/actor_critic.py": 2_374,
        "pobax/models/discrete.py": 11_026,
        "pobax/models/__init__.py": 2_321,
        "pobax/models/network.py": 7_414,
        "pobax/models/value.py": 671,
        "pyproject.toml": 1_262,
        "requirements.txt": 165,
    }
)

_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024
_UINT31_MAX: Final = (1 << 31) - 1
_OBSERVATION_SHAPE: Final = (9, 9, 3)
_RUNNER_BLOCKERS: Final = (
    "qualified_foragax_environment_bridge_missing",
    "full_horizon_compilation_and_memory_profile_unqualified",
    "environment_trace_and_rng_parity_unqualified",
    "validated_epoch_driver_unimplemented",
    "artifact_writer_and_execution_receipt_unimplemented",
)


class ForagerMatchedV3PPOGRUError(ValueError):
    """The PPO-GRU config, source, recurrent batch, or readiness state is invalid."""


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ForagerMatchedV3PPOGRUError(
            f"{name} must be an exact integer >= {minimum}"
        )
    return value


def _require_finite_float(
    value: object,
    name: str,
    *,
    lower_exclusive: float | None = None,
    upper_inclusive: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ForagerMatchedV3PPOGRUError(f"{name} must be an exact finite float")
    if lower_exclusive is not None and value <= lower_exclusive:
        raise ForagerMatchedV3PPOGRUError(f"{name} must be > {lower_exclusive}")
    if upper_inclusive is not None and value > upper_inclusive:
        raise ForagerMatchedV3PPOGRUError(f"{name} must be <= {upper_inclusive}")
    return value


@dataclass(frozen=True)
class PPOGRUConfig:
    """Exact-task PPO-GRU hyperparameters and closed workload accounting."""

    horizon: int = 499_712
    observation_shape: tuple[int, int, int] = _OBSERVATION_SHAPE
    num_actions: int = 4
    num_envs: int = 1
    rollout_steps: int = 512
    segment_steps: int = 128
    update_epochs: int = 4
    hidden_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 0.00025
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    adam_epsilon: float = 0.00001
    anneal_learning_rate: bool = True

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "num_actions",
            "num_envs",
            "rollout_steps",
            "segment_steps",
            "update_epochs",
            "hidden_size",
        ):
            _require_exact_int(getattr(self, name), name, minimum=1)
        if self.horizon != 499_712:
            raise ForagerMatchedV3PPOGRUError("horizon must remain exactly 499712")
        if type(self.observation_shape) is not tuple or self.observation_shape != (
            _OBSERVATION_SHAPE
        ):
            raise ForagerMatchedV3PPOGRUError(
                "observation_shape must remain the exact 9x9x3 categorical color aperture"
            )
        if self.num_actions != 4:
            raise ForagerMatchedV3PPOGRUError("num_actions must remain exactly four")
        if self.num_envs != 1:
            raise ForagerMatchedV3PPOGRUError(
                "parallel environments and reward aggregation are forbidden"
            )
        if self.rollout_steps != 512 or self.segment_steps != 128:
            raise ForagerMatchedV3PPOGRUError(
                "rollout/segment geometry must remain 512/128"
            )
        if self.update_epochs != 4:
            raise ForagerMatchedV3PPOGRUError("update_epochs must remain exactly four")
        if self.horizon % self.rollout_steps != 0:
            raise ForagerMatchedV3PPOGRUError("rollout_steps must divide the horizon")
        if self.rollout_steps % self.segment_steps != 0:
            raise ForagerMatchedV3PPOGRUError("segment_steps must divide each rollout")
        _require_finite_float(
            self.gamma, "gamma", lower_exclusive=0.0, upper_inclusive=1.0
        )
        _require_finite_float(
            self.gae_lambda,
            "gae_lambda",
            lower_exclusive=0.0,
            upper_inclusive=1.0,
        )
        for name in (
            "learning_rate",
            "clip_epsilon",
            "value_coefficient",
            "entropy_coefficient",
            "max_grad_norm",
            "adam_epsilon",
        ):
            _require_finite_float(getattr(self, name), name, lower_exclusive=0.0)
        if type(self.anneal_learning_rate) is not bool:
            raise ForagerMatchedV3PPOGRUError(
                "anneal_learning_rate must be an exact boolean"
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
    def loss_transition_evaluations(self) -> int:
        return self.horizon * self.update_epochs

    @property
    def segment_permutation_draw_count(self) -> int:
        return self.rollout_count * self.update_epochs

    @property
    def total_agent_subkey_draw_count(self) -> int:
        return 1 + self.horizon + self.segment_permutation_draw_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PPO_GRU_CONFIGURATION_SCHEMA_VERSION,
            "candidate_id": "adapted_ppo_gru",
            "status": "implemented_unqualified",
            "relationship": "derived_exact_task_adapter",
            "task": {
                "environment_id": "ForagaxTwoBiomeLarge-v1",
                "continuing": True,
                "observation_type": "color",
                "observation_shape": list(self.observation_shape),
                "aperture_size": 9,
                "num_actions": self.num_actions,
                "action_distribution": "categorical",
                "reward_range": [-1.0, 30.0],
                "reward_preprocessing": "identity_no_scaling",
                "public_trajectory_count": 1,
                "parallel_environments": self.num_envs,
                "parallel_reward_aggregation": False,
            },
            "architecture": {
                "observation_preprocessing": (
                    "identity_zero_or_one_hot_color_channels_then_flatten"
                ),
                "embedding": "orthogonal_dense_relu",
                "recurrent_core": "flax_linen_gru",
                "hidden_size": self.hidden_size,
                "done_boundary_reset": "zero_incoming_carry_before_current_observation",
                "actor_head": "dense_relu_dense_four_logits",
                "critic_head": "dense_relu_dense_scalar",
                "distribution_dependency": "native_jax_no_distrax",
                "categorical_sampling_mode": PPO_GRU_CATEGORICAL_MODE,
            },
            "optimization": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "gae_recursion": (
                    "independent_128_step_segments_bootstrapped_at_next_behavior_value"
                ),
                "learning_rate": self.learning_rate,
                "anneal_learning_rate_by_rollout": self.anneal_learning_rate,
                "clip_epsilon": self.clip_epsilon,
                "value_coefficient": self.value_coefficient,
                "entropy_coefficient": self.entropy_coefficient,
                "max_global_gradient_norm": self.max_grad_norm,
                "adam_epsilon": self.adam_epsilon,
                "value_loss": "maximum_of_unclipped_and_clipped_squared_error",
                "policy_loss": "negative_minimum_of_unclipped_and_clipped_surrogate",
            },
            "rollout_and_segmentation": {
                "rollout_steps": self.rollout_steps,
                "segment_steps": self.segment_steps,
                "segments_per_rollout": self.segments_per_rollout,
                "segment_order": "required_agent_rng_permutation_once_per_epoch",
                "segment_permutation_rng_draws_per_epoch": 1,
                "within_segment_timestep_order": "strictly_increasing_never_shuffled",
                "segment_initial_carry": "bound_to_actual_rollout_incoming_carry",
                "gae_recursion_steps": self.segment_steps,
                "gae_final_segment_bootstrap": "rollout_terminal_bootstrap_value",
                "update_epochs": self.update_epochs,
            },
            "accounting": {
                "environment_interactions": self.horizon,
                "rollout_count": self.rollout_count,
                "transitions_per_rollout": self.rollout_steps,
                "segments_per_rollout": self.segments_per_rollout,
                "transitions_per_segment": self.segment_steps,
                "update_epochs": self.update_epochs,
                "optimizer_updates_per_rollout": self.optimizer_updates_per_rollout,
                "optimizer_update_count": self.optimizer_update_count,
                "loss_transition_evaluations": self.loss_transition_evaluations,
                "agent_parameter_initialization_draws": 1,
                "agent_action_sampling_draws": self.horizon,
                "agent_segment_permutation_draws": self.segment_permutation_draw_count,
                "total_agent_subkey_draws": self.total_agent_subkey_draw_count,
                "reward_aggregation": "none_single_continuing_trajectory",
            },
            "seed_contract": {
                "environment_seed": "required_exact_uint31_root",
                "agent_seed": "required_exact_uint31_root",
                "prng_implementation": PPO_GRU_PRNG_IMPLEMENTATION,
                "roots_are_logically_separate_consumption_chains": True,
                "equal_numeric_values_allowed": True,
                "equal_numeric_values_correlate_key_streams": True,
                "statistical_independence_claimed": False,
                "environment_draws_from_agent_root": False,
                "agent_draws_from_environment_root": False,
            },
            "claims": {
                "configuration_complete": True,
                "core_implementation_complete": True,
                "validated_epoch_driver_complete": False,
                "full_forager_runner_complete": False,
                "execution_ready": False,
                "execution_authorized": False,
                "scientific_promotion_allowed": False,
            },
        }


def _assert_plain_unaliased_json(value: object, label: str) -> None:
    pending = [value]
    containers: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if id(item) in containers:
                raise ForagerMatchedV3PPOGRUError(
                    f"{label} contains aliased or cyclic containers"
                )
            containers.add(id(item))
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3PPOGRUError(
                    f"{label} contains a non-string object key"
                )
            pending.extend(mapping.values())
        elif type(item) is list:
            if id(item) in containers:
                raise ForagerMatchedV3PPOGRUError(
                    f"{label} contains aliased or cyclic containers"
                )
            containers.add(id(item))
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3PPOGRUError(
                    f"{label} contains a non-finite number"
                )
        elif item is not None and type(item) not in {str, int, bool}:
            raise ForagerMatchedV3PPOGRUError(
                f"{label} contains non-JSON type {type(item).__name__}"
            )


def _canonical_json(value: object, label: str) -> bytes:
    _assert_plain_unaliased_json(value, label)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ForagerMatchedV3PPOGRUError(f"{label} is not canonical JSON") from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3PPOGRUError(f"{label} exceeds the artifact byte limit")
    return raw


_CONFIGURATION: Final = PPOGRUConfig()
_CONFIGURATION_BYTES: Final = _canonical_json(
    _CONFIGURATION.to_dict(), "PPO-GRU configuration"
)
PPO_GRU_CONFIGURATION_SHA256: Final = (
    "07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70"
)
if not hmac.compare_digest(
    hashlib.sha256(_CONFIGURATION_BYTES).hexdigest(),
    PPO_GRU_CONFIGURATION_SHA256,
):
    raise RuntimeError("PPO-GRU canonical configuration digest drift")


def _source_descriptor() -> dict[str, Any]:
    return {
        "schema_version": PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "candidate_id": "adapted_ppo_gru",
        "status": "implemented_unqualified",
        "relationship": "derived_exact_task_adapter",
        "license": {
            "spdx_identifier": POBAX_LICENSE,
            "upstream_license_path": "LICENSE",
            "upstream_notice_file_present": False,
            "modification_notice": (
                "Exact-task derivative with logically separate RNG consumption chains, "
                "native-JAX categorical math, and recurrent time-segment minibatches."
            ),
        },
        "upstream": {
            "upstream_review_anchors_bound": True,
            "source_closure_bound": False,
            "canonical_url": POBAX_CANONICAL_URL,
            "commit_git_sha1": POBAX_COMMIT_GIT_SHA1,
            "tree_git_sha1": POBAX_TREE_GIT_SHA1,
            "archive_sha256": POBAX_ARCHIVE_SHA256,
            "archive_size_bytes": POBAX_ARCHIVE_SIZE_BYTES,
            "relevant_files": [
                {
                    "path": path,
                    "size_bytes": REQUIRED_POBAX_SOURCE_SIZE_BYTES_BY_PATH[path],
                    "sha256": digest,
                }
                for path, digest in REQUIRED_POBAX_SOURCE_SHA256_BY_PATH.items()
            ],
        },
        "derived_implementation": {
            "module": "alberta_framework.benchmarks.forager_matched_v3_ppo_gru",
            "path": "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py",
            "configuration_schema_version": PPO_GRU_CONFIGURATION_SCHEMA_VERSION,
            "configuration_sha256": PPO_GRU_CONFIGURATION_SHA256,
            "source_snapshot_status": "unqualified_current_checkout",
            "preserved_mechanisms": [
                "GRU carry with done-boundary reset",
                "generalized advantage estimation",
                "clipped policy surrogate",
                "clipped value loss",
                "categorical action sampling and log probability",
                "entropy regularization",
                "global gradient clipping",
                "Adam optimizer with rollout-linear learning-rate annealing",
            ],
            "deliberate_changes": [
                "Bind ForagaxTwoBiomeLarge-v1 color aperture 9 and four actions.",
                "Use one continuing public trajectory with no parallel reward aggregation.",
                "Require explicit uint31 environment and agent consumption chains.",
                "Pin JAX threefry2x32 keys and low-mode categorical sampling.",
                "Replace Distrax categorical operations with native JAX operations.",
                "Fold four parallel 128-step lanes into one 512-step trajectory.",
                "Replace environment-axis minibatching with four contiguous 128-step segments.",
                "Bind every segment to its recorded incoming GRU carry.",
                "Truncate and bootstrap GAE at each 128-step segment boundary.",
                "Use 976 exact 512-step rollouts to close the 499712-step horizon.",
            ],
        },
        "claims": {
            "configuration_complete": True,
            "core_implementation_complete": True,
            "validated_epoch_driver_complete": False,
            "full_forager_runner_complete": False,
            "execution_ready": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "authority_granted": False,
        },
        "runner_blockers": list(_RUNNER_BLOCKERS),
        "limitations": [
            "No Foragax environment bridge or full-horizon runner is implemented here.",
            "The compiled update kernel accepts trusted rows from the validated segment builder.",
            "No runtime, memory, RNG-parity, environment-trace, or artifact qualification ran.",
            "Equal numeric environment and agent roots retain correlated key streams.",
            "Passing unit tests cannot authorize execution or support a performance claim.",
        ],
    }


_SOURCE_DESCRIPTOR: Final = _source_descriptor()
_SOURCE_DESCRIPTOR_BYTES: Final = _canonical_json(
    _SOURCE_DESCRIPTOR, "PPO-GRU source descriptor"
)
PPO_GRU_SOURCE_DESCRIPTOR_SHA256: Final = (
    "64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca"
)
if not hmac.compare_digest(
    hashlib.sha256(_SOURCE_DESCRIPTOR_BYTES).hexdigest(),
    PPO_GRU_SOURCE_DESCRIPTOR_SHA256,
):
    raise RuntimeError("PPO-GRU canonical source descriptor digest drift")


def verify_pobax_source_files(sources: object) -> dict[str, Any]:
    """Verify the exact audited POBAX source subset without importing it."""
    if type(sources) is not dict:
        raise ForagerMatchedV3PPOGRUError("POBAX source set must be a plain dictionary")
    source_map = cast(dict[object, object], sources)
    if any(type(path) is not str or type(raw) is not bytes for path, raw in source_map.items()):
        raise ForagerMatchedV3PPOGRUError(
            "POBAX source paths must be exact strings and source values exact bytes"
        )
    expected = set(REQUIRED_POBAX_SOURCE_SHA256_BY_PATH)
    actual = set(cast(dict[str, bytes], sources))
    if actual != expected:
        raise ForagerMatchedV3PPOGRUError(
            "POBAX source membership drift: "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )
    for path in REQUIRED_POBAX_SOURCE_SHA256_BY_PATH:
        raw = cast(dict[str, bytes], sources)[path]
        if len(raw) != REQUIRED_POBAX_SOURCE_SIZE_BYTES_BY_PATH[path]:
            raise ForagerMatchedV3PPOGRUError(f"POBAX source size drift for {path}")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(
            actual_sha256, REQUIRED_POBAX_SOURCE_SHA256_BY_PATH[path]
        ):
            raise ForagerMatchedV3PPOGRUError(f"POBAX source digest drift for {path}")
    return {
        "status": "verified_pinned_source_files",
        "canonical_url": POBAX_CANONICAL_URL,
        "commit_git_sha1": POBAX_COMMIT_GIT_SHA1,
        "tree_git_sha1": POBAX_TREE_GIT_SHA1,
        "archive_sha256": POBAX_ARCHIVE_SHA256,
        "license": POBAX_LICENSE,
        "source_sha256_by_path": dict(REQUIRED_POBAX_SOURCE_SHA256_BY_PATH),
    }


def matched_v3_ppo_gru_configuration() -> PPOGRUConfig:
    """Return the frozen exact-task configuration."""
    return _CONFIGURATION


def canonical_matched_v3_ppo_gru_configuration_bytes() -> bytes:
    """Return canonical exact-task configuration bytes."""
    return _CONFIGURATION_BYTES


def matched_v3_ppo_gru_source_descriptor() -> dict[str, Any]:
    """Decode a fresh non-authorizing descriptor from immutable canonical bytes."""
    decoded = json.loads(_SOURCE_DESCRIPTOR_BYTES.decode("utf-8"))
    return cast(dict[str, Any], decoded)


def canonical_matched_v3_ppo_gru_source_descriptor_bytes() -> bytes:
    """Return canonical provenance descriptor bytes."""
    return _SOURCE_DESCRIPTOR_BYTES


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3PPOGRUError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise ForagerMatchedV3PPOGRUError(f"non-finite JSON number {token!r} is forbidden")


def _parse_exact_artifact(raw: bytes, expected: bytes, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ForagerMatchedV3PPOGRUError(f"{label} must be exact bytes")
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3PPOGRUError(f"{label} exceeds the artifact byte limit")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3PPOGRUError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ForagerMatchedV3PPOGRUError(f"{label} is not strict UTF-8 JSON") from exc
    if type(decoded) is not dict:
        raise ForagerMatchedV3PPOGRUError(f"{label} must be a JSON object")
    canonical = _canonical_json(decoded, label)
    if not hmac.compare_digest(raw, canonical):
        raise ForagerMatchedV3PPOGRUError(f"{label} is not canonical JSON")
    if not hmac.compare_digest(raw, expected):
        raise ForagerMatchedV3PPOGRUError(f"{label} content differs from the frozen contract")
    return cast(dict[str, Any], decoded)


def parse_matched_v3_ppo_gru_configuration(raw: bytes) -> PPOGRUConfig:
    """Accept only the exact canonical configuration artifact."""
    _parse_exact_artifact(raw, _CONFIGURATION_BYTES, "PPO-GRU configuration")
    return _CONFIGURATION


def parse_matched_v3_ppo_gru_source_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical source descriptor artifact."""
    return _parse_exact_artifact(
        raw, _SOURCE_DESCRIPTOR_BYTES, "PPO-GRU source descriptor"
    )


@dataclass(frozen=True)
class PPOGRUSeedPair:
    """Exact uint31 roots for logically separate, potentially correlated chains."""

    environment_seed: int
    agent_seed: int


def _require_uint31(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3PPOGRUError(f"{name} must be an exact uint31")
    return value


def validate_ppo_gru_seed_pair(
    environment_seed: object, agent_seed: object
) -> PPOGRUSeedPair:
    """Validate explicit matched-v3 environment and agent RNG roots."""
    return PPOGRUSeedPair(
        _require_uint31(environment_seed, "environment_seed"),
        _require_uint31(agent_seed, "agent_seed"),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRURNGState:
    """Logically separate environment and agent split chains with exact counters."""

    environment_key: chex.PRNGKey
    agent_key: chex.PRNGKey
    environment_draw_count: chex.Array
    agent_draw_count: chex.Array


def initialize_ppo_gru_rng_state(
    environment_seed: object, agent_seed: object
) -> PPOGRURNGState:
    """Create separate pinned-Threefry typed-key roots from explicit uint31 seeds."""
    seeds = validate_ppo_gru_seed_pair(environment_seed, agent_seed)
    return PPOGRURNGState(
        environment_key=jr.key(
            seeds.environment_seed, impl=PPO_GRU_PRNG_IMPLEMENTATION
        ),
        agent_key=jr.key(seeds.agent_seed, impl=PPO_GRU_PRNG_IMPLEMENTATION),
        environment_draw_count=jnp.asarray(0, dtype=jnp.uint32),
        agent_draw_count=jnp.asarray(0, dtype=jnp.uint32),
    )


def validate_ppo_gru_rng_state(state: PPOGRURNGState) -> tuple[int, int]:
    """Host-validate key implementation, scalar counters, and counter domains."""
    if type(state) is not PPOGRURNGState:
        raise ForagerMatchedV3PPOGRUError("RNG state must be PPOGRURNGState")
    for name in ("environment_key", "agent_key"):
        try:
            implementation = str(jr.key_impl(getattr(state, name)))
        except (TypeError, ValueError) as exc:
            raise ForagerMatchedV3PPOGRUError(f"{name} is not a typed JAX key") from exc
        if implementation != PPO_GRU_PRNG_IMPLEMENTATION:
            raise ForagerMatchedV3PPOGRUError(
                f"{name} must use {PPO_GRU_PRNG_IMPLEMENTATION}"
            )
    counts: list[int] = []
    for name in ("environment_draw_count", "agent_draw_count"):
        value = getattr(state, name)
        if tuple(value.shape) != () or value.dtype != jnp.uint32:
            raise ForagerMatchedV3PPOGRUError(f"{name} must be a scalar uint32")
        try:
            counts.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ForagerMatchedV3PPOGRUError(
                f"{name} must be host-validatable"
            ) from exc
    return counts[0], counts[1]


def next_ppo_gru_environment_key(
    state: PPOGRURNGState,
) -> tuple[PPOGRURNGState, chex.PRNGKey]:
    """Consume one environment-only subkey without touching the agent root."""
    next_root, subkey = jr.split(state.environment_key)
    return (
        replace(
            state,
            environment_key=next_root,
            environment_draw_count=state.environment_draw_count + jnp.uint32(1),
        ),
        subkey,
    )


def next_ppo_gru_agent_key(
    state: PPOGRURNGState,
) -> tuple[PPOGRURNGState, chex.PRNGKey]:
    """Consume one agent-only subkey without touching the environment root."""
    next_root, subkey = jr.split(state.agent_key)
    return (
        replace(
            state,
            agent_key=next_root,
            agent_draw_count=state.agent_draw_count + jnp.uint32(1),
        ),
        subkey,
    )


def next_ppo_gru_segment_order(
    state: PPOGRURNGState,
) -> tuple[PPOGRURNGState, chex.Array]:
    """Consume exactly one agent subkey for the required four-segment permutation."""
    next_state, key = next_ppo_gru_agent_key(state)
    order = jr.permutation(key, jnp.arange(4, dtype=jnp.int32))
    return next_state, order


class PPOGRUActorCritic(nn.Module):
    """POBAX-style Dense/GRU actor-critic specialized to the color aperture."""

    hidden_size: int = 128
    num_actions: int = 4

    @nn.compact
    def __call__(
        self,
        carry: chex.Array,
        observation: chex.Array,
        reset_before: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        if tuple(observation.shape) != _OBSERVATION_SHAPE:
            raise ForagerMatchedV3PPOGRUError(
                "observation must be the exact 9x9x3 color aperture"
            )
        if tuple(carry.shape) != (self.hidden_size,):
            raise ForagerMatchedV3PPOGRUError("GRU carry shape differs from hidden_size")
        if tuple(reset_before.shape) != () or reset_before.dtype != jnp.bool_:
            raise ForagerMatchedV3PPOGRUError("reset_before must be a boolean scalar")
        x = jnp.asarray(observation, dtype=jnp.float32).reshape((-1,))
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="embedding",
        )(x)
        x = nn.relu(x)
        carry = jnp.where(jnp.asarray(reset_before, dtype=jnp.bool_), jnp.zeros_like(carry), carry)
        carry, encoded = nn.GRUCell(features=self.hidden_size, name="gru")(carry, x)

        actor = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="actor_hidden",
        )(encoded)
        actor = nn.relu(actor)
        logits = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_logits",
        )(actor)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(encoded)
        critic = nn.relu(critic)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_value",
        )(critic)
        return carry, logits, jnp.squeeze(value, axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRUSequenceEvaluation:
    """Time-major recurrent predictions and both sides of every carry edge."""

    final_carry: chex.Array
    incoming_carries: chex.Array
    outgoing_carries: chex.Array
    logits: chex.Array
    values: chex.Array


def _require_array_shape(value: Any, shape: tuple[int, ...], name: str) -> None:
    if not hasattr(value, "shape") or tuple(value.shape) != shape:
        raise ForagerMatchedV3PPOGRUError(
            f"{name} shape must be {shape!r}, got {getattr(value, 'shape', None)!r}"
        )


def evaluate_ppo_gru_sequence(
    model: PPOGRUActorCritic,
    variables: Any,
    initial_carry: chex.Array,
    observations: chex.Array,
    reset_before: chex.Array,
) -> PPOGRUSequenceEvaluation:
    """Evaluate a contiguous time-major sequence without timestep shuffling."""
    if not hasattr(observations, "shape") or len(observations.shape) < 2:
        raise ForagerMatchedV3PPOGRUError("observations must have a time dimension")
    steps = int(observations.shape[0])
    if steps < 1:
        raise ForagerMatchedV3PPOGRUError("sequence must contain at least one timestep")
    _require_array_shape(initial_carry, (model.hidden_size,), "initial_carry")
    _require_array_shape(reset_before, (steps,), "reset_before")
    if tuple(observations.shape[1:]) != _OBSERVATION_SHAPE:
        raise ForagerMatchedV3PPOGRUError(
            "sequence observations must have exact trailing shape (9, 9, 3)"
        )
    if observations.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("sequence observations must be float32")
    if reset_before.dtype != jnp.bool_:
        raise ForagerMatchedV3PPOGRUError("sequence reset_before must be boolean")

    def step(
        carry: chex.Array, inputs: tuple[chex.Array, chex.Array]
    ) -> tuple[chex.Array, tuple[chex.Array, chex.Array, chex.Array, chex.Array]]:
        observation, reset = inputs
        incoming = carry
        applied = cast(
            tuple[chex.Array, chex.Array, chex.Array],
            model.apply(variables, carry, observation, reset),
        )
        outgoing, logits, value = applied
        return outgoing, (incoming, outgoing, logits, value)

    final_carry, outputs = jax.lax.scan(step, initial_carry, (observations, reset_before))
    incoming, outgoing, logits, values = outputs
    return PPOGRUSequenceEvaluation(
        final_carry=final_carry,
        incoming_carries=incoming,
        outgoing_carries=outgoing,
        logits=logits,
        values=values,
    )


def _categorical_log_prob_unchecked(
    logits: chex.Array, actions: chex.Array
) -> chex.Array:
    if not hasattr(logits, "shape") or logits.ndim < 1 or logits.shape[-1] != 4:
        raise ForagerMatchedV3PPOGRUError("categorical logits must end in four actions")
    if logits.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("categorical logits must be float32")
    if tuple(actions.shape) != tuple(logits.shape[:-1]):
        raise ForagerMatchedV3PPOGRUError("categorical action shape differs from logits")
    if actions.dtype != jnp.int32:
        raise ForagerMatchedV3PPOGRUError("categorical actions must be int32")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    action_mask = jax.nn.one_hot(actions, 4, dtype=log_probs.dtype)
    return jnp.sum(log_probs * action_mask, axis=-1)


def _validate_categorical_action_values(actions: chex.Array, label: str) -> None:
    try:
        host_actions = np.asarray(actions)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUError(
            f"{label} must be host-validatable before the compiled core"
        ) from exc
    if host_actions.dtype != np.dtype(np.int32):
        raise ForagerMatchedV3PPOGRUError(f"{label} must be int32")
    if np.any((host_actions < 0) | (host_actions >= 4)):
        raise ForagerMatchedV3PPOGRUError(f"{label} must contain only actions 0..3")


def categorical_log_prob(logits: chex.Array, actions: chex.Array) -> chex.Array:
    """Strict host-checked native-JAX categorical log probability."""
    _validate_categorical_action_values(actions, "categorical actions")
    return _categorical_log_prob_unchecked(logits, actions)


def categorical_entropy(logits: chex.Array) -> chex.Array:
    """Native-JAX categorical entropy."""
    if not hasattr(logits, "shape") or logits.ndim < 1 or logits.shape[-1] != 4:
        raise ForagerMatchedV3PPOGRUError("categorical logits must end in four actions")
    if logits.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("categorical logits must be float32")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def sample_categorical_action(key: chex.PRNGKey, logits: chex.Array) -> chex.Array:
    """Sample four-way actions using only the caller-supplied agent key."""
    if not hasattr(logits, "shape") or logits.ndim < 1 or logits.shape[-1] != 4:
        raise ForagerMatchedV3PPOGRUError("categorical logits must end in four actions")
    if logits.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("categorical logits must be float32")
    if str(jr.key_impl(key)) != PPO_GRU_PRNG_IMPLEMENTATION:
        raise ForagerMatchedV3PPOGRUError(
            f"categorical key must use {PPO_GRU_PRNG_IMPLEMENTATION}"
        )
    return jr.categorical(
        key,
        logits,
        axis=-1,
        mode=PPO_GRU_CATEGORICAL_MODE,
    ).astype(jnp.int32)


def _calculate_gae_unchecked(
    *,
    rewards: chex.Array,
    values: chex.Array,
    transition_dones: chex.Array,
    bootstrap_value: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    def reverse_step(
        carry: tuple[chex.Array, chex.Array],
        transition: tuple[chex.Array, chex.Array, chex.Array],
    ) -> tuple[tuple[chex.Array, chex.Array], tuple[chex.Array, chex.Array]]:
        next_value, next_advantage = carry
        reward, value, done = transition
        nonterminal = 1.0 - done.astype(value.dtype)
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        return (value, advantage), (advantage, advantage + value)

    (_, _), (advantages, targets) = jax.lax.scan(
        reverse_step,
        (bootstrap_value, jnp.zeros_like(bootstrap_value)),
        (rewards, values, transition_dones),
        reverse=True,
    )
    return advantages, targets


def calculate_gae(
    *,
    rewards: chex.Array,
    values: chex.Array,
    transition_dones: chex.Array,
    bootstrap_value: chex.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Host-validate and compute time-major GAE at transition done boundaries."""
    if not hasattr(rewards, "shape") or rewards.ndim != 1 or rewards.shape[0] < 1:
        raise ForagerMatchedV3PPOGRUError("GAE rewards must be a nonempty vector")
    steps = int(rewards.shape[0])
    _require_array_shape(values, (steps,), "GAE values")
    _require_array_shape(transition_dones, (steps,), "GAE transition_dones")
    _require_array_shape(bootstrap_value, (), "GAE bootstrap_value")
    if transition_dones.dtype != jnp.bool_:
        raise ForagerMatchedV3PPOGRUError("GAE transition_dones must be boolean")
    for name, value in (
        ("GAE rewards", rewards),
        ("GAE values", values),
        ("GAE bootstrap_value", bootstrap_value),
    ):
        if value.dtype != jnp.float32:
            raise ForagerMatchedV3PPOGRUError(f"{name} must be float32")
        _require_finite_host(value, name)
    gamma_value = _require_finite_float(
        gamma, "gamma", lower_exclusive=0.0, upper_inclusive=1.0
    )
    lambda_value = _require_finite_float(
        gae_lambda, "gae_lambda", lower_exclusive=0.0, upper_inclusive=1.0
    )
    return _calculate_gae_unchecked(
        rewards=rewards,
        values=values,
        transition_dones=transition_dones,
        bootstrap_value=bootstrap_value,
        gamma=gamma_value,
        gae_lambda=lambda_value,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRURollout:
    """One contiguous rollout, including every incoming recurrent carry."""

    initial_carry: chex.Array
    observations: chex.Array
    reset_before: chex.Array
    actions: chex.Array
    rewards: chex.Array
    transition_dones: chex.Array
    old_log_probs: chex.Array
    old_values: chex.Array
    incoming_carries: chex.Array
    bootstrap_observation: chex.Array
    bootstrap_value: chex.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRUSegmentBatch:
    """A permutation of intact contiguous segments; timesteps remain ordered."""

    segment_ids: chex.Array
    time_indices: chex.Array
    initial_carries: chex.Array
    observations: chex.Array
    reset_before: chex.Array
    actions: chex.Array
    rewards: chex.Array
    transition_dones: chex.Array
    old_log_probs: chex.Array
    old_values: chex.Array
    advantages: chex.Array
    targets: chex.Array


def _rollout_steps(rollout: PPOGRURollout) -> int:
    if not hasattr(rollout.observations, "shape") or rollout.observations.ndim < 2:
        raise ForagerMatchedV3PPOGRUError("rollout observations must be time-major")
    steps = int(rollout.observations.shape[0])
    if steps < 1:
        raise ForagerMatchedV3PPOGRUError("rollout must not be empty")
    if tuple(rollout.observations.shape[1:]) != _OBSERVATION_SHAPE:
        raise ForagerMatchedV3PPOGRUError(
            "rollout observations must have exact trailing shape (9, 9, 3)"
        )
    if rollout.observations.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("rollout observations must be float32")
    vector_fields = (
        "reset_before",
        "actions",
        "rewards",
        "transition_dones",
        "old_log_probs",
        "old_values",
    )
    for name in vector_fields:
        _require_array_shape(getattr(rollout, name), (steps,), f"rollout {name}")
    if (
        not hasattr(rollout.incoming_carries, "shape")
        or rollout.incoming_carries.ndim != 2
        or rollout.incoming_carries.shape[0] != steps
    ):
        raise ForagerMatchedV3PPOGRUError(
            "rollout incoming_carries must have shape [time, hidden]"
        )
    hidden_size = int(rollout.incoming_carries.shape[1])
    _require_array_shape(rollout.initial_carry, (hidden_size,), "rollout initial_carry")
    _require_array_shape(
        rollout.bootstrap_observation,
        _OBSERVATION_SHAPE,
        "rollout bootstrap_observation",
    )
    _require_array_shape(rollout.bootstrap_value, (), "rollout bootstrap_value")
    if rollout.bootstrap_observation.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("rollout bootstrap_observation must be float32")
    if rollout.reset_before.dtype != jnp.bool_:
        raise ForagerMatchedV3PPOGRUError("rollout reset_before must be boolean")
    if rollout.transition_dones.dtype != jnp.bool_:
        raise ForagerMatchedV3PPOGRUError("rollout transition_dones must be boolean")
    if rollout.actions.dtype != jnp.int32:
        raise ForagerMatchedV3PPOGRUError("rollout actions must be int32")
    for name in (
        "rewards",
        "old_log_probs",
        "old_values",
        "incoming_carries",
        "initial_carry",
        "bootstrap_value",
    ):
        if getattr(rollout, name).dtype != jnp.float32:
            raise ForagerMatchedV3PPOGRUError(f"rollout {name} must be float32")
    return steps


def calculate_segmented_gae_core(
    rollout: PPOGRURollout,
    *,
    segment_steps: object,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Pure JAX segmented-GAE kernel preserving POBAX's per-lane horizon.

    Each intact segment is bootstrapped from the behavior value of the next
    observation.  The final segment instead uses the rollout's explicit final
    bootstrap value.  Advantage recursion never crosses a segment boundary.
    """
    steps = _rollout_steps(rollout)
    segment_length = _require_exact_int(segment_steps, "segment_steps", minimum=1)
    if steps % segment_length != 0:
        raise ForagerMatchedV3PPOGRUError("segment_steps must divide rollout time")

    advantage_segments: list[chex.Array] = []
    target_segments: list[chex.Array] = []
    for start in range(0, steps, segment_length):
        end = start + segment_length
        bootstrap = (
            rollout.bootstrap_value if end == steps else rollout.old_values[end]
        )
        advantages, targets = _calculate_gae_unchecked(
            rewards=rollout.rewards[start:end],
            values=rollout.old_values[start:end],
            transition_dones=rollout.transition_dones[start:end],
            bootstrap_value=bootstrap,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantage_segments.append(advantages)
        target_segments.append(targets)
    return jnp.concatenate(advantage_segments), jnp.concatenate(target_segments)


def calculate_segmented_gae(
    rollout: PPOGRURollout,
    *,
    segment_steps: object,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Host-validate then execute the pure segmented-GAE kernel."""
    _rollout_steps(rollout)
    for name in ("rewards", "old_values", "bootstrap_value"):
        _require_finite_host(getattr(rollout, name), f"rollout {name}")
    gamma_value = _require_finite_float(
        gamma, "gamma", lower_exclusive=0.0, upper_inclusive=1.0
    )
    lambda_value = _require_finite_float(
        gae_lambda, "gae_lambda", lower_exclusive=0.0, upper_inclusive=1.0
    )
    return calculate_segmented_gae_core(
        rollout,
        segment_steps=segment_steps,
        gamma=gamma_value,
        gae_lambda=lambda_value,
    )


def _host_array(value: chex.Array, label: str) -> np.ndarray[Any, Any]:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUError(
            f"{label} must be host-validatable before the compiled core"
        ) from exc


def _require_finite_host(value: chex.Array, label: str) -> np.ndarray[Any, Any]:
    host = _host_array(value, label)
    if not np.all(np.isfinite(host)):
        raise ForagerMatchedV3PPOGRUError(f"{label} must contain only finite values")
    return host


def _require_zero_or_one_hot_color(value: chex.Array, label: str) -> None:
    host = _require_finite_host(value, label)
    if not np.all((host == 0.0) | (host == 1.0)):
        raise ForagerMatchedV3PPOGRUError(f"{label} must contain only exact zero/one values")
    channel_sums = np.sum(host, axis=-1)
    if not np.all((channel_sums == 0.0) | (channel_sums == 1.0)):
        raise ForagerMatchedV3PPOGRUError(
            f"{label} must be zero-or-one-hot along the color-channel axis"
        )


def _require_exact_array_equal(
    actual: chex.Array, expected: chex.Array, label: str
) -> None:
    actual_host = _host_array(actual, label)
    expected_host = _host_array(expected, f"expected {label}")
    if actual_host.shape != expected_host.shape or not np.array_equal(
        actual_host, expected_host
    ):
        raise ForagerMatchedV3PPOGRUError(f"{label} differs from replayed behavior data")


def validate_ppo_gru_rollout(
    model: PPOGRUActorCritic,
    variables: Any,
    rollout: PPOGRURollout,
    config: PPOGRUConfig,
    *,
    expected_initial_carry: chex.Array,
    expected_initial_reset: object,
) -> tuple[chex.Array, chex.Array]:
    """Replay and bind an exact rollout, then return its only valid segmented GAE."""
    steps = _rollout_steps(rollout)
    if steps != config.rollout_steps:
        raise ForagerMatchedV3PPOGRUError(
            "rollout length differs from the exact configuration"
        )
    if model.hidden_size != config.hidden_size or model.num_actions != config.num_actions:
        raise ForagerMatchedV3PPOGRUError("model differs from the exact configuration")
    _require_array_shape(
        expected_initial_carry,
        (config.hidden_size,),
        "expected_initial_carry",
    )
    if expected_initial_carry.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("expected_initial_carry must be float32")
    _require_exact_array_equal(
        rollout.initial_carry,
        expected_initial_carry,
        "rollout initial_carry",
    )
    if type(expected_initial_reset) is not bool:
        raise ForagerMatchedV3PPOGRUError("expected_initial_reset must be an exact boolean")
    _require_zero_or_one_hot_color(rollout.observations, "rollout observations")
    _require_zero_or_one_hot_color(
        rollout.bootstrap_observation, "rollout bootstrap_observation"
    )
    _validate_categorical_action_values(rollout.actions, "rollout actions")
    rewards = _require_finite_host(rollout.rewards, "rollout rewards")
    if np.any((rewards < -1.0) | (rewards > 30.0)):
        raise ForagerMatchedV3PPOGRUError(
            "rollout rewards must remain within the exact Forager range [-1, 30]"
        )
    for name in (
        "old_log_probs",
        "old_values",
        "incoming_carries",
        "initial_carry",
        "bootstrap_value",
    ):
        _require_finite_host(getattr(rollout, name), f"rollout {name}")
    reset_before = _host_array(rollout.reset_before, "rollout reset_before")
    transition_dones = _host_array(
        rollout.transition_dones, "rollout transition_dones"
    )
    if bool(reset_before[0]) is not expected_initial_reset:
        raise ForagerMatchedV3PPOGRUError(
            "rollout reset_before[0] differs from the runner-carried reset"
        )
    if not np.array_equal(reset_before[1:], transition_dones[:-1]):
        raise ForagerMatchedV3PPOGRUError(
            "rollout reset_before[1:] must equal transition_dones[:-1]"
        )

    evaluation = evaluate_ppo_gru_sequence(
        model,
        variables,
        rollout.initial_carry,
        rollout.observations,
        rollout.reset_before,
    )
    _require_exact_array_equal(
        rollout.incoming_carries,
        evaluation.incoming_carries,
        "rollout incoming_carries",
    )
    _require_exact_array_equal(
        rollout.old_values,
        evaluation.values,
        "rollout old_values",
    )
    replayed_log_probs = _categorical_log_prob_unchecked(
        evaluation.logits, rollout.actions
    )
    _require_exact_array_equal(
        rollout.old_log_probs,
        replayed_log_probs,
        "rollout old_log_probs",
    )
    _, _, replayed_bootstrap_value = cast(
        tuple[chex.Array, chex.Array, chex.Array],
        model.apply(
            variables,
            evaluation.final_carry,
            rollout.bootstrap_observation,
            rollout.transition_dones[-1],
        ),
    )
    _require_exact_array_equal(
        rollout.bootstrap_value,
        replayed_bootstrap_value,
        "rollout bootstrap_value",
    )
    advantages, targets = calculate_segmented_gae(
        rollout,
        segment_steps=config.segment_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    _require_finite_host(advantages, "rollout advantages")
    _require_finite_host(targets, "rollout targets")
    return advantages, targets


def _require_segment_order(
    order: Sequence[int] | chex.Array | None,
    segment_count: int,
) -> chex.Array:
    if order is None:
        return jnp.arange(segment_count, dtype=jnp.int32)
    if type(order) in {tuple, list}:
        result = tuple(cast(Sequence[int], order))
        if any(type(value) is not int for value in result) or sorted(result) != list(
            range(segment_count)
        ):
            raise ForagerMatchedV3PPOGRUError(
                "segment order must be an exact permutation of segment IDs"
            )
        return jnp.asarray(result, dtype=jnp.int32)
    if not hasattr(order, "shape") or tuple(order.shape) != (segment_count,):
        raise ForagerMatchedV3PPOGRUError(
            "compiled segment order must have shape [segments]"
        )
    array_order = cast(chex.Array, order)
    if array_order.dtype != jnp.int32:
        raise ForagerMatchedV3PPOGRUError("compiled segment order must be int32")
    return array_order


def _build_ppo_gru_sequence_segments_unchecked(
    rollout: PPOGRURollout,
    advantages: chex.Array,
    targets: chex.Array,
    *,
    segment_steps: object,
    segment_order: Sequence[int] | chex.Array | None = None,
) -> PPOGRUSegmentBatch:
    """JAX core for a trusted segment order supplied by the validated driver."""
    steps = _rollout_steps(rollout)
    segment_length = _require_exact_int(segment_steps, "segment_steps", minimum=1)
    if steps % segment_length != 0:
        raise ForagerMatchedV3PPOGRUError("segment_steps must divide rollout time")
    _require_array_shape(advantages, (steps,), "advantages")
    _require_array_shape(targets, (steps,), "targets")
    segment_count = steps // segment_length
    order_array = _require_segment_order(segment_order, segment_count)

    def segment(value: chex.Array) -> chex.Array:
        reshaped = value.reshape((segment_count, segment_length, *value.shape[1:]))
        return jnp.take(reshaped, order_array, axis=0)

    carry_segments = rollout.incoming_carries.reshape(
        (segment_count, segment_length, rollout.incoming_carries.shape[1])
    )
    return PPOGRUSegmentBatch(
        segment_ids=order_array,
        time_indices=jnp.take(
            jnp.arange(steps, dtype=jnp.int32).reshape(segment_count, segment_length),
            order_array,
            axis=0,
        ),
        initial_carries=jnp.take(carry_segments[:, 0], order_array, axis=0),
        observations=segment(rollout.observations),
        reset_before=segment(rollout.reset_before),
        actions=segment(rollout.actions),
        rewards=segment(rollout.rewards),
        transition_dones=segment(rollout.transition_dones),
        old_log_probs=segment(rollout.old_log_probs),
        old_values=segment(rollout.old_values),
        advantages=segment(advantages),
        targets=segment(targets),
    )


def build_ppo_gru_sequence_segments(
    rollout: PPOGRURollout,
    advantages: chex.Array,
    targets: chex.Array,
    *,
    segment_steps: object,
    segment_order: Sequence[int] | chex.Array | None = None,
) -> PPOGRUSegmentBatch:
    """Host-validate a whole-segment permutation before executing the JAX core."""
    steps = _rollout_steps(rollout)
    segment_length = _require_exact_int(segment_steps, "segment_steps", minimum=1)
    if steps % segment_length != 0:
        raise ForagerMatchedV3PPOGRUError("segment_steps must divide rollout time")
    segment_count = steps // segment_length
    if segment_order is None:
        host_order = tuple(range(segment_count))
    else:
        try:
            host_order = tuple(int(value) for value in np.asarray(segment_order).tolist())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ForagerMatchedV3PPOGRUError(
                "segment order must be host-validatable before segmentation"
            ) from exc
    checked_order = _require_segment_order(host_order, segment_count)
    return _build_ppo_gru_sequence_segments_unchecked(
        rollout,
        advantages,
        targets,
        segment_steps=segment_length,
        segment_order=checked_order,
    )


def build_validated_ppo_gru_sequence_segments(
    model: PPOGRUActorCritic,
    variables: Any,
    rollout: PPOGRURollout,
    config: PPOGRUConfig,
    *,
    expected_initial_carry: chex.Array,
    expected_initial_reset: object,
    segment_order: Sequence[int] | chex.Array,
) -> PPOGRUSegmentBatch:
    """Build the exact segment batch only after behavior replay and GAE binding."""
    advantages, targets = validate_ppo_gru_rollout(
        model,
        variables,
        rollout,
        config,
        expected_initial_carry=expected_initial_carry,
        expected_initial_reset=expected_initial_reset,
    )
    try:
        host_order = tuple(int(value) for value in np.asarray(segment_order).tolist())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForagerMatchedV3PPOGRUError("segment order is not host-validatable") from exc
    _require_segment_order(host_order, config.segments_per_rollout)
    return build_ppo_gru_sequence_segments(
        rollout,
        advantages,
        targets,
        segment_steps=config.segment_steps,
        segment_order=host_order,
    )


def validate_ppo_gru_sequence_segments(
    segments: PPOGRUSegmentBatch,
    model: PPOGRUActorCritic,
    variables: Any,
    rollout: PPOGRURollout,
    config: PPOGRUConfig,
    *,
    expected_initial_carry: chex.Array,
    expected_initial_reset: object,
) -> None:
    """Replay behavior, GAE, segment content, order, and carry bindings."""
    try:
        order = tuple(int(value) for value in np.asarray(segments.segment_ids).tolist())
        _require_segment_order(order, config.segments_per_rollout)
        expected = build_validated_ppo_gru_sequence_segments(
            model,
            variables,
            rollout,
            config,
            expected_initial_carry=expected_initial_carry,
            expected_initial_reset=expected_initial_reset,
            segment_order=order,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForagerMatchedV3PPOGRUError("segment IDs are invalid") from exc
    for field in fields(PPOGRUSegmentBatch):
        actual_value = np.asarray(getattr(segments, field.name))
        expected_value = np.asarray(getattr(expected, field.name))
        if actual_value.shape != expected_value.shape or not np.array_equal(
            actual_value, expected_value
        ):
            raise ForagerMatchedV3PPOGRUError(
                f"segment content or carry binding drift for {field.name}"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOLossResult:
    """Scalar PPO loss and diagnostics for one intact recurrent segment."""

    total_loss: chex.Array
    policy_loss: chex.Array
    value_loss: chex.Array
    entropy: chex.Array
    approximate_kl: chex.Array
    clip_fraction: chex.Array


def normalize_advantages(advantages: chex.Array) -> chex.Array:
    """Apply POBAX-style zero-mean unit-scale advantage normalization."""
    if not hasattr(advantages, "shape") or advantages.ndim != 1 or advantages.shape[0] < 1:
        raise ForagerMatchedV3PPOGRUError("advantages must be a nonempty vector")
    return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)


def _ppo_clipped_loss_from_predictions_unchecked(
    *,
    logits: chex.Array,
    values: chex.Array,
    actions: chex.Array,
    old_log_probs: chex.Array,
    old_values: chex.Array,
    normalized_advantages: chex.Array,
    targets: chex.Array,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> PPOLossResult:
    if logits.ndim != 2 or logits.shape[-1] != 4:
        raise ForagerMatchedV3PPOGRUError("PPO logits must have shape [time, 4]")
    steps = int(logits.shape[0])
    for name, value in (
        ("values", values),
        ("actions", actions),
        ("old_log_probs", old_log_probs),
        ("old_values", old_values),
        ("normalized_advantages", normalized_advantages),
        ("targets", targets),
    ):
        _require_array_shape(value, (steps,), name)
    epsilon = _require_finite_float(clip_epsilon, "clip_epsilon", lower_exclusive=0.0)
    value_coeff = _require_finite_float(
        value_coefficient, "value_coefficient", lower_exclusive=0.0
    )
    entropy_coeff = _require_finite_float(
        entropy_coefficient, "entropy_coefficient", lower_exclusive=0.0
    )

    new_log_probs = _categorical_log_prob_unchecked(logits, actions)
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    unclipped_surrogate = ratio * normalized_advantages
    clipped_surrogate = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * (
        normalized_advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(unclipped_surrogate, clipped_surrogate))

    clipped_values = old_values + jnp.clip(values - old_values, -epsilon, epsilon)
    value_loss = jnp.mean(
        jnp.maximum(jnp.square(values - targets), jnp.square(clipped_values - targets))
    )
    entropy = jnp.mean(categorical_entropy(logits))
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
    approximate_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > epsilon).astype(jnp.float32))
    return PPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


def ppo_clipped_loss_from_predictions(
    *,
    logits: chex.Array,
    values: chex.Array,
    actions: chex.Array,
    old_log_probs: chex.Array,
    old_values: chex.Array,
    normalized_advantages: chex.Array,
    targets: chex.Array,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> PPOLossResult:
    """Host-validate and compute the clipped policy/value PPO objective."""
    _validate_categorical_action_values(actions, "PPO actions")
    for name, value in (
        ("PPO logits", logits),
        ("PPO values", values),
        ("PPO old_log_probs", old_log_probs),
        ("PPO old_values", old_values),
        ("PPO normalized_advantages", normalized_advantages),
        ("PPO targets", targets),
    ):
        if value.dtype != jnp.float32:
            raise ForagerMatchedV3PPOGRUError(f"{name} must be float32")
        _require_finite_host(value, name)
    return _ppo_clipped_loss_from_predictions_unchecked(
        logits=logits,
        values=values,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        normalized_advantages=normalized_advantages,
        targets=targets,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRULossBatch:
    """One contiguous recurrent segment and its fixed PPO targets."""

    initial_carry: chex.Array
    observations: chex.Array
    reset_before: chex.Array
    actions: chex.Array
    old_log_probs: chex.Array
    old_values: chex.Array
    advantages: chex.Array
    targets: chex.Array


def validate_ppo_gru_loss_batch(
    batch: PPOGRULossBatch,
    config: PPOGRUConfig,
) -> None:
    """Host-validate an exact recurrent segment before entering compiled updates."""
    if tuple(batch.observations.shape) != (
        config.segment_steps,
        *config.observation_shape,
    ):
        raise ForagerMatchedV3PPOGRUError(
            "loss batch observations differ from the exact segment shape"
        )
    _require_array_shape(
        batch.initial_carry,
        (config.hidden_size,),
        "loss batch initial_carry",
    )
    if batch.initial_carry.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("loss batch initial_carry must be float32")
    if batch.observations.dtype != jnp.float32:
        raise ForagerMatchedV3PPOGRUError("loss batch observations must be float32")
    _require_zero_or_one_hot_color(batch.observations, "loss batch observations")
    _require_finite_host(batch.initial_carry, "loss batch initial_carry")
    _require_array_shape(
        batch.reset_before,
        (config.segment_steps,),
        "loss batch reset_before",
    )
    if batch.reset_before.dtype != jnp.bool_:
        raise ForagerMatchedV3PPOGRUError("loss batch reset_before must be boolean")
    for name in ("actions", "old_log_probs", "old_values", "advantages", "targets"):
        value = getattr(batch, name)
        _require_array_shape(value, (config.segment_steps,), f"loss batch {name}")
    _validate_categorical_action_values(batch.actions, "loss batch actions")
    for name in ("old_log_probs", "old_values", "advantages", "targets"):
        value = getattr(batch, name)
        if value.dtype != jnp.float32:
            raise ForagerMatchedV3PPOGRUError(f"loss batch {name} must be float32")
        _require_finite_host(value, f"loss batch {name}")


def ppo_gru_loss_batch_from_segment(
    segments: PPOGRUSegmentBatch,
    segment_position: object,
    config: PPOGRUConfig,
) -> PPOGRULossBatch:
    """Extract one trusted row after the caller validates the complete segment batch."""
    position = _require_exact_int(
        segment_position, "segment_position", minimum=0
    )
    if position >= config.segments_per_rollout:
        raise ForagerMatchedV3PPOGRUError("segment_position is outside the rollout")
    batch = PPOGRULossBatch(
        initial_carry=segments.initial_carries[position],
        observations=segments.observations[position],
        reset_before=segments.reset_before[position],
        actions=segments.actions[position],
        old_log_probs=segments.old_log_probs[position],
        old_values=segments.old_values[position],
        advantages=segments.advantages[position],
        targets=segments.targets[position],
    )
    validate_ppo_gru_loss_batch(batch, config)
    return batch


def _ppo_gru_loss_unchecked(
    model: PPOGRUActorCritic,
    variables: Any,
    batch: PPOGRULossBatch,
    config: PPOGRUConfig,
) -> PPOLossResult:
    evaluation = evaluate_ppo_gru_sequence(
        model,
        variables,
        batch.initial_carry,
        batch.observations,
        batch.reset_before,
    )
    steps = int(batch.observations.shape[0])
    for name in ("actions", "old_log_probs", "old_values", "advantages", "targets"):
        _require_array_shape(getattr(batch, name), (steps,), f"loss batch {name}")
    return _ppo_clipped_loss_from_predictions_unchecked(
        logits=evaluation.logits,
        values=evaluation.values,
        actions=batch.actions,
        old_log_probs=batch.old_log_probs,
        old_values=batch.old_values,
        normalized_advantages=normalize_advantages(batch.advantages),
        targets=batch.targets,
        clip_epsilon=config.clip_epsilon,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
    )


def ppo_gru_loss(
    model: PPOGRUActorCritic,
    variables: Any,
    batch: PPOGRULossBatch,
    config: PPOGRUConfig,
) -> PPOLossResult:
    """Host-validate, rerun an intact GRU segment, and calculate PPO loss."""
    validate_ppo_gru_loss_batch(batch, config)
    if model.hidden_size != config.hidden_size or model.num_actions != config.num_actions:
        raise ForagerMatchedV3PPOGRUError("model differs from the exact configuration")
    return _ppo_gru_loss_unchecked(model, variables, batch, config)


def clip_ppo_gru_gradients(
    gradients: optax.Updates,
    max_global_norm: object,
) -> tuple[optax.Updates, chex.Array, chex.Array]:
    """Clip the complete gradient tree by one shared global norm."""
    limit = _require_finite_float(
        max_global_norm, "max_global_norm", lower_exclusive=0.0
    )
    before = optax.tree.norm(gradients)
    transform = optax.clip_by_global_norm(limit)
    clipped, _ = transform.update(gradients, transform.init(gradients))
    after = optax.tree.norm(clipped)
    return clipped, before, after


def _learning_rate_schedule(config: PPOGRUConfig) -> optax.Schedule:
    def schedule(update_index: chex.Array) -> chex.Array:
        rollout_index = update_index // config.optimizer_updates_per_rollout
        fraction = 1.0 - rollout_index.astype(jnp.float32) / float(config.rollout_count)
        return jnp.asarray(config.learning_rate, dtype=jnp.float32) * jnp.maximum(
            fraction, 0.0
        )

    return schedule


def _optimizer(config: PPOGRUConfig) -> optax.GradientTransformation:
    learning_rate: float | optax.Schedule
    if config.anneal_learning_rate:
        learning_rate = _learning_rate_schedule(config)
    else:
        learning_rate = config.learning_rate
    return optax.adam(learning_rate=learning_rate, eps=config.adam_epsilon)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRUTrainState:
    """Frozen variables, Adam state, and exact update counter."""

    variables: Any
    optimizer_state: optax.OptState
    optimizer_updates: chex.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPOGRUUpdateResult:
    """One deterministic clipped-gradient optimizer transaction."""

    state: PPOGRUTrainState
    loss: PPOLossResult
    gradient_norm_before_clip: chex.Array
    gradient_norm_after_clip: chex.Array


def initialize_ppo_gru_train_state(
    config: PPOGRUConfig,
    *,
    rng_state: PPOGRURNGState,
) -> tuple[PPOGRUTrainState, PPOGRURNGState]:
    """Consume one key from the single owner agent chain for parameter initialization."""
    _, agent_draw_count = validate_ppo_gru_rng_state(rng_state)
    if agent_draw_count != 0:
        raise ForagerMatchedV3PPOGRUError(
            "parameter initialization must be the first agent-key draw"
        )
    model = PPOGRUActorCritic(
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    next_rng_state, initialization_key = next_ppo_gru_agent_key(rng_state)
    variables = model.init(
        initialization_key,
        jnp.zeros((config.hidden_size,), dtype=jnp.float32),
        jnp.zeros(config.observation_shape, dtype=jnp.float32),
        jnp.asarray(False),
    )
    optimizer = _optimizer(config)
    return (
        PPOGRUTrainState(
            variables=variables,
            optimizer_state=optimizer.init(variables),
            optimizer_updates=jnp.asarray(0, dtype=jnp.int32),
        ),
        next_rng_state,
    )


def reset_ppo_gru_optimizer_for_variables(
    state: PPOGRUTrainState,
    config: PPOGRUConfig,
) -> PPOGRUTrainState:
    """Reset Adam after a caller deliberately replaces variables in a test/probe."""
    return replace(
        state,
        optimizer_state=_optimizer(config).init(state.variables),
        optimizer_updates=jnp.asarray(0, dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("model", "config"))
def ppo_gru_update_core(
    model: PPOGRUActorCritic,
    state: PPOGRUTrainState,
    batch: PPOGRULossBatch,
    config: PPOGRUConfig,
) -> PPOGRUUpdateResult:
    """Pure compiled update kernel; callers must host-validate the fixed batch first."""
    def objective(variables: Any) -> tuple[chex.Array, PPOLossResult]:
        result = _ppo_gru_loss_unchecked(model, variables, batch, config)
        return result.total_loss, result

    (_, loss), gradients = jax.value_and_grad(objective, has_aux=True)(state.variables)
    clipped, norm_before, norm_after = clip_ppo_gru_gradients(
        gradients, config.max_grad_norm
    )
    optimizer = _optimizer(config)
    updates, optimizer_state = optimizer.update(
        clipped,
        state.optimizer_state,
        state.variables,
    )
    variables = optax.apply_updates(state.variables, updates)
    next_state = replace(
        state,
        variables=variables,
        optimizer_state=optimizer_state,
        optimizer_updates=state.optimizer_updates + jnp.int32(1),
    )
    return PPOGRUUpdateResult(
        state=next_state,
        loss=loss,
        gradient_norm_before_clip=norm_before,
        gradient_norm_after_clip=norm_after,
    )


def validate_ppo_gru_update_counter(
    state: PPOGRUTrainState,
    config: PPOGRUConfig,
) -> int:
    """Host-check that one more optimizer update fits the exact accounting plan."""
    if (
        tuple(state.optimizer_updates.shape) != ()
        or state.optimizer_updates.dtype != jnp.int32
    ):
        raise ForagerMatchedV3PPOGRUError(
            "optimizer update counter must be a scalar int32"
        )
    try:
        update_index = int(state.optimizer_updates)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3PPOGRUError(
            "optimizer update counter must be host-validatable"
        ) from exc
    if update_index < 0 or update_index >= config.optimizer_update_count:
        raise ForagerMatchedV3PPOGRUError("optimizer update counter is outside the plan")
    return update_index


def ppo_gru_update(
    model: PPOGRUActorCritic,
    state: PPOGRUTrainState,
    batch: PPOGRULossBatch,
    config: PPOGRUConfig,
) -> PPOGRUUpdateResult:
    """Host-validate then apply one compiled PPO segment optimizer transaction."""
    validate_ppo_gru_update_counter(state, config)
    validate_ppo_gru_loss_batch(batch, config)
    if model.hidden_size != config.hidden_size or model.num_actions != config.num_actions:
        raise ForagerMatchedV3PPOGRUError("model differs from the exact configuration")
    return cast(PPOGRUUpdateResult, ppo_gru_update_core(model, state, batch, config))


def assert_matched_v3_ppo_gru_runner_ready() -> None:
    """Fail closed until the exact-task core is embedded in a qualified runner."""
    raise ForagerMatchedV3PPOGRUError(
        "matched-v3 PPO-GRU runner is not ready; blockers=" + ",".join(_RUNNER_BLOCKERS)
    )


__all__ = [
    "POBAX_ARCHIVE_SHA256",
    "POBAX_ARCHIVE_SIZE_BYTES",
    "POBAX_CANONICAL_URL",
    "POBAX_COMMIT_GIT_SHA1",
    "POBAX_LICENSE",
    "POBAX_TREE_GIT_SHA1",
    "PPO_GRU_CONFIGURATION_SCHEMA_VERSION",
    "PPO_GRU_CONFIGURATION_SHA256",
    "PPO_GRU_CATEGORICAL_MODE",
    "PPO_GRU_PRNG_IMPLEMENTATION",
    "PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION",
    "PPO_GRU_SOURCE_DESCRIPTOR_SHA256",
    "REQUIRED_POBAX_SOURCE_SHA256_BY_PATH",
    "REQUIRED_POBAX_SOURCE_SIZE_BYTES_BY_PATH",
    "ForagerMatchedV3PPOGRUError",
    "PPOGRUActorCritic",
    "PPOGRUConfig",
    "PPOGRULossBatch",
    "PPOGRURNGState",
    "PPOGRURollout",
    "PPOGRUSeedPair",
    "PPOGRUSegmentBatch",
    "PPOGRUSequenceEvaluation",
    "PPOGRUTrainState",
    "PPOGRUUpdateResult",
    "PPOLossResult",
    "assert_matched_v3_ppo_gru_runner_ready",
    "build_ppo_gru_sequence_segments",
    "build_validated_ppo_gru_sequence_segments",
    "calculate_gae",
    "calculate_segmented_gae",
    "calculate_segmented_gae_core",
    "canonical_matched_v3_ppo_gru_configuration_bytes",
    "canonical_matched_v3_ppo_gru_source_descriptor_bytes",
    "categorical_entropy",
    "categorical_log_prob",
    "clip_ppo_gru_gradients",
    "evaluate_ppo_gru_sequence",
    "initialize_ppo_gru_rng_state",
    "initialize_ppo_gru_train_state",
    "matched_v3_ppo_gru_configuration",
    "matched_v3_ppo_gru_source_descriptor",
    "next_ppo_gru_agent_key",
    "next_ppo_gru_environment_key",
    "next_ppo_gru_segment_order",
    "normalize_advantages",
    "parse_matched_v3_ppo_gru_configuration",
    "parse_matched_v3_ppo_gru_source_descriptor",
    "ppo_clipped_loss_from_predictions",
    "ppo_gru_loss",
    "ppo_gru_loss_batch_from_segment",
    "ppo_gru_update",
    "ppo_gru_update_core",
    "reset_ppo_gru_optimizer_for_variables",
    "sample_categorical_action",
    "validate_ppo_gru_seed_pair",
    "validate_ppo_gru_loss_batch",
    "validate_ppo_gru_rng_state",
    "validate_ppo_gru_rollout",
    "validate_ppo_gru_sequence_segments",
    "validate_ppo_gru_update_counter",
    "verify_pobax_source_files",
]
