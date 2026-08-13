# mypy: disable-error-code="arg-type,attr-defined,call-arg,operator"
"""Matched nonpromoting development evaluator for calibrated extended search.

The evaluator exercises all four static modes of
``CalibratedExtendedSearchControl`` from one immutable, source-bound initial
snapshot.  A single evaluator-owned continuing trace supplies every arm with
the exact same decision identities, real anchors, executed extended actions,
outcomes, option completions, and pre-outcome primitive/option model snapshot.
The controller receives one shared backup budget ``B`` in every arm; combined
search never receives ``B+B``.

This is intentionally an exact *model-snapshot* boundary, not a live STOMP
adapter.  The frozen snapshot contains primitive reward/discount/outcome
predictions and option return/baseline-mass/discount/outcome predictions.  No
model is updated during a run, behavior is evaluator-owned and action
independent, and planning cannot change the real trace.  The boundary can test
causal search mechanics and accounting, but cannot establish control benefit,
online model adaptation, keyboard-policy benefit, or the WP7 exit gate.

All outcomes remain ``not_assessed``.  Thresholds and aggregate verdicts are
absent, all seeds are consumed nonpromoting development roots, and the module
has no artifact, evidence, scientific-promotion, or policy authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.extend import backend as jax_backend
from jaxtyping import Bool, Float, Int, UInt

import alberta_framework.core.calibrated_extended_search_control as search_module
from alberta_framework.core.calibrated_extended_search_control import (
    CANDIDATE_KIND_OPTION,
    CANDIDATE_KIND_PRIMITIVE,
    SEARCH_MODE_COMBINED,
    SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
    SEARCH_MODE_OPTION_MODEL,
    SEARCH_MODE_PRIMITIVE_MODEL,
    CalibratedExtendedSearchControl,
    CalibratedExtendedSearchControlConfig,
    CalibratedExtendedSearchControlResourceBudget,
    CalibratedExtendedSearchControlState,
)

CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CONFIG_SCHEMA: Final = (
    "alberta.calibrated-extended-search.matched-development.config.v1"
)
CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_SUITE_SCHEMA: Final = (
    "alberta.calibrated-extended-search.matched-development.suite.v1"
)
CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA: Final = (
    "alberta.calibrated-extended-search.matched-development.checkpoint.v1"
)
CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_REPLAY_SCHEMA: Final = (
    "alberta.calibrated-extended-search.matched-development.replay.v1"
)
CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_MANIFEST_SCHEMA: Final = (
    "alberta.calibrated-extended-search.source-runtime-manifest.v1"
)
CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS: Final = (
    "DEVELOPMENT_RAW_MATCHED_FOUR_ARM_NOT_ASSESSED"
)
ASSESSMENT_STATUS: Final = "not_assessed"
DEVELOPMENT_SEED_ROLE: Final = "consumed_nonpromoting_development"
MODEL_SNAPSHOT_BOUNDARY: Final = (
    "frozen-pre-outcome-model-and-calibration-snapshot-no-live-STOMP-updates"
)
SOURCE_CALIBRATION_BOUNDARY: Final = (
    "shared-source-calibration-moments-fixed-before-evaluator-outcomes"
)
TRACE_SEMANTICS: Final = (
    "continuing-evaluator-owned-action-independent-threefry-common-random-numbers"
)
RUNTIME_IDENTITY_SCOPE: Final = (
    "observable-nonsecret-python-jax-xla-device-and-config-fields; "
    "unobservable-compiler-and-host-determinants-require-exact-replay"
)

DEVELOPMENT_ONLY: Final = True
HELD_OUT_SEEDS_USED: Final = False
THRESHOLDS_FROZEN: Final = False
ARTIFACT_WRITES_AUTHORIZED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
POLICY_AUTHORITY: Final = False

CANONICAL_ARM_ORDER: Final = (
    SEARCH_MODE_MODEL_FREE_EXTENDED_Q,
    SEARCH_MODE_PRIMITIVE_MODEL,
    SEARCH_MODE_OPTION_MODEL,
    SEARCH_MODE_COMBINED,
)
MODEL_FREE_VS_OPTION_MODEL: Final = "model_free_extended_q_vs_option_model"
PRIMITIVE_VS_COMBINED: Final = "primitive_model_vs_combined"

_ANCHOR_CAPACITY = 2
_OBSERVATION_DIM = 2
_N_PRIMITIVE_ACTIONS = 2
_N_OPTIONS = 1
_N_EXTENDED_ACTIONS = _N_PRIMITIVE_ACTIONS + _N_OPTIONS
_CANDIDATE_CAPACITY = _ANCHOR_CAPACITY * _N_EXTENDED_ACTIONS
_OPTION_DESCRIPTOR_WIDTH = 4
_MAX_STEPS = 100_000
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_MODULES = (
    ("development_evaluator", __file__),
    ("calibrated_search_core", search_module.__file__),
)


class CalibratedExtendedSearchDevelopmentError(RuntimeError):
    """Raised when a development protocol or replay fails closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> object:
        """Add bytes to an incremental digest."""


def _hash_part(digest: _HashWriter, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _update_exact_hash(digest: _HashWriter, value: object) -> None:
    """Hash supported values with exact concrete types, shapes, and bits."""

    _hash_part(
        digest,
        f"{type(value).__module__}.{type(value).__qualname__}".encode(),
    )
    if isinstance(value, jax.Array):
        if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
            _hash_part(digest, str(jr.key_impl(value)).encode("ascii"))
            array = np.asarray(jax.device_get(jr.key_data(value)))
        else:
            array = np.asarray(jax.device_get(value))
        _hash_part(digest, array.dtype.str.encode("ascii"))
        _hash_part(digest, _canonical_json_bytes(tuple(int(size) for size in array.shape)))
        _hash_part(digest, array.tobytes(order="C"))
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _hash_part(digest, field.name.encode("utf-8"))
            _update_exact_hash(digest, getattr(value, field.name))
        return
    if type(value) is tuple:
        items = cast(tuple[object, ...], value)
        _hash_part(digest, len(items).to_bytes(8, "big", signed=False))
        for item in items:
            _update_exact_hash(digest, item)
        return
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        keys = tuple(sorted(mapping, key=lambda item: str(item)))
        _hash_part(digest, len(keys).to_bytes(8, "big", signed=False))
        for key in keys:
            _update_exact_hash(digest, key)
            _update_exact_hash(digest, mapping[key])
        return
    if value is None:
        _hash_part(digest, b"none")
        return
    if type(value) is bool:
        _hash_part(digest, b"1" if value else b"0")
        return
    if type(value) is int:
        _hash_part(digest, str(value).encode("ascii"))
        return
    if type(value) is float:
        _hash_part(
            digest,
            np.asarray((value,), dtype=np.float64).tobytes(order="C"),
        )
        return
    if type(value) is str:
        _hash_part(digest, value.encode("utf-8"))
        return
    raise CalibratedExtendedSearchDevelopmentError(
        f"unsupported exact-hash type: {type(value).__qualname__}"
    )


def _exact_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_exact_hash(digest, value)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_words(*values: str) -> Array:
    digest = hashlib.sha256("\x00".join(values).encode("ascii")).digest()
    return jnp.asarray(
        (
            int.from_bytes(digest[:4], "little"),
            int.from_bytes(digest[4:8], "little"),
        ),
        dtype=jnp.uint32,
    )


def _host_tree_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree or len(left_leaves) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(jax.device_get(left_leaf))
        right_array = np.asarray(jax.device_get(right_leaf))
        if (
            left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape
            or left_array.tobytes(order="C") != right_array.tobytes(order="C")
        ):
            return False
    return True


def _host_array_prefix_bits_equal(left: Array, right: Array, step: int) -> bool:
    left_prefix = np.asarray(jax.device_get(left[:, :step]))
    right_prefix = np.asarray(jax.device_get(right[:, :step]))
    return (
        left_prefix.dtype == right_prefix.dtype
        and left_prefix.shape == right_prefix.shape
        and left_prefix.tobytes(order="C") == right_prefix.tobytes(order="C")
    )


def _tree_array_nbytes(value: object) -> int:
    if isinstance(value, jax.Array):
        return int(np.asarray(jax.device_get(value)).nbytes)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tree_array_nbytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if type(value) is tuple:
        return sum(_tree_array_nbytes(item) for item in cast(tuple[object, ...], value))
    return 0


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedExtendedSearchDevelopmentConfig:
    """Exact consumed-development protocol; it contains no thresholds."""

    seed: int = 20_260_802
    num_steps: int = 6
    backup_budget: int = 2
    calibration_evidence_floor: int = 2
    model_support_floor: int = 2
    confidence_scale: float = 0.5
    support_prior: float = 2.0
    model_error_scale: float = 2.0
    backup_step_size: float = 0.2
    max_observations: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "num_steps",
            "backup_budget",
            "calibration_evidence_floor",
            "model_support_floor",
            "max_observations",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an exact Python int")
        if not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("seed must be uint32-compatible")
        if not 1 <= self.num_steps <= _MAX_STEPS:
            raise ValueError(f"num_steps must be in [1, {_MAX_STEPS}]")
        if self.num_steps % _CANDIDATE_CAPACITY:
            raise ValueError("num_steps must cover whole fixed candidate cycles")
        if self.num_steps < _CANDIDATE_CAPACITY:
            raise ValueError("num_steps must expose every fixed candidate at least once")
        if not 1 <= self.backup_budget <= _CANDIDATE_CAPACITY:
            raise ValueError("backup_budget must fit the fixed candidate capacity")
        if self.calibration_evidence_floor < 2 or self.model_support_floor < 1:
            raise ValueError("calibration/model support floors are invalid")
        if self.max_observations < self.required_max_observations:
            raise ValueError(
                "max_observations leaves insufficient preloaded-counter headroom; "
                f"requires at least {self.required_max_observations}"
            )
        for name in (
            "confidence_scale",
            "support_prior",
            "model_error_scale",
            "backup_step_size",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive exact float")
        if self.backup_step_size > 1.0:
            raise ValueError("backup_step_size must be at most one")

    @property
    def maximum_anchor_trial_increments(self) -> int:
        """Every natural record contributes one Bernoulli trial per active anchor."""

        return self.num_steps

    @property
    def maximum_candidate_observation_increments(self) -> int:
        """Whole candidate cycles visit every candidate this many times."""

        return self.num_steps // _CANDIDATE_CAPACITY

    @property
    def required_max_observations(self) -> int:
        """Exact largest version-1 observation counter reached by this protocol."""

        candidate_increments = self.maximum_candidate_observation_increments
        return max(
            self.num_steps,  # frozen primitive/option model support
            self.calibration_evidence_floor + self.maximum_anchor_trial_increments,
            self.calibration_evidence_floor + candidate_increments,
            self.model_support_floor + candidate_increments,
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CONFIG_SCHEMA,
            "status": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS,
            "assessment_status": ASSESSMENT_STATUS,
            "development_only": True,
            "seed_role": DEVELOPMENT_SEED_ROLE,
            "held_out_seeds_used": False,
            "thresholds_frozen": False,
            "artifact_writes_authorized": False,
            "evidence_authorized": False,
            "scientific_promotion_allowed": False,
            "policy_authority": False,
            "model_snapshot_boundary": MODEL_SNAPSHOT_BOUNDARY,
            "source_calibration_boundary": SOURCE_CALIBRATION_BOUNDARY,
            "trace_semantics": TRACE_SEMANTICS,
            "canonical_arm_order": CANONICAL_ARM_ORDER,
            "anchor_capacity": _ANCHOR_CAPACITY,
            "observation_dim": _OBSERVATION_DIM,
            "n_primitive_actions": _N_PRIMITIVE_ACTIONS,
            "n_options": _N_OPTIONS,
            "candidate_capacity": _CANDIDATE_CAPACITY,
            **dataclasses.asdict(self),
            "thresholds": None,
        }

    @classmethod
    def from_config(
        cls, value: object
    ) -> CalibratedExtendedSearchDevelopmentConfig:
        if type(value) is not dict:
            raise ValueError("development config must be an exact dict")
        raw = cast(dict[object, object], value)
        expected = set(cls().to_config())
        if set(raw) != expected:
            raise ValueError("development config fields differ from v1")
        fixed = {
            "schema": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CONFIG_SCHEMA,
            "status": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS,
            "assessment_status": ASSESSMENT_STATUS,
            "development_only": True,
            "seed_role": DEVELOPMENT_SEED_ROLE,
            "held_out_seeds_used": False,
            "thresholds_frozen": False,
            "artifact_writes_authorized": False,
            "evidence_authorized": False,
            "scientific_promotion_allowed": False,
            "policy_authority": False,
            "model_snapshot_boundary": MODEL_SNAPSHOT_BOUNDARY,
            "source_calibration_boundary": SOURCE_CALIBRATION_BOUNDARY,
            "trace_semantics": TRACE_SEMANTICS,
            "canonical_arm_order": CANONICAL_ARM_ORDER,
            "anchor_capacity": _ANCHOR_CAPACITY,
            "observation_dim": _OBSERVATION_DIM,
            "n_primitive_actions": _N_PRIMITIVE_ACTIONS,
            "n_options": _N_OPTIONS,
            "candidate_capacity": _CANDIDATE_CAPACITY,
            "thresholds": None,
        }
        for name, expected_value in fixed.items():
            observed = raw[name]
            if type(observed) is not type(expected_value) or observed != expected_value:
                raise ValueError(f"development config fixed field {name} differs")
        kwargs = {
            field.name: raw[field.name]
            for field in dataclasses.fields(cls)
        }
        integer_fields = {
            "seed",
            "num_steps",
            "backup_budget",
            "calibration_evidence_floor",
            "model_support_floor",
            "max_observations",
        }
        for name, item in kwargs.items():
            expected_type = int if name in integer_fields else float
            if type(item) is not expected_type:
                raise ValueError(f"serialized {name} has the wrong concrete type")
        return cls(**cast(dict[str, Any], kwargs))


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchSourceFileHash:
    role: str
    repository_path: str
    nbytes: int
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchSourceRuntimeManifest:
    """Full source bytes and explicit observable, non-secret runtime identity."""

    schema: str
    source_files: tuple[CalibratedSearchSourceFileHash, ...]
    runtime_identity_scope: str
    python_implementation: str
    python_version: str
    python_compiler: str
    chex_version: str
    jax_version: str
    jaxlib_version: str
    numpy_version: str
    operating_system: str
    operating_system_release: str
    machine: str
    processor: str
    backend: str
    backend_platform_version: str
    device_count: int
    local_device_count: int
    device_platforms: tuple[str, ...]
    device_kinds: tuple[str, ...]
    x64_enabled: bool
    default_matmul_precision: str
    numpy_dtype_promotion: str
    numpy_rank_promotion: str
    threefry_partitionable: bool
    default_prng_impl: str
    jit_disabled: bool
    runtime_checks_enabled: bool
    prng_impl: str
    prng_key_dtype: str
    prng_key_data_shape: tuple[int, ...]
    prng_key_data_dtype: str
    manifest_sha256: str


def _source_file_hash(role: str, file_name: object) -> CalibratedSearchSourceFileHash:
    if type(file_name) is not str:
        raise CalibratedExtendedSearchDevelopmentError(
            f"load-bearing source {role!r} has no concrete file"
        )
    path = Path(file_name).resolve(strict=True)
    try:
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise CalibratedExtendedSearchDevelopmentError(
            f"load-bearing source {role!r} is outside the repository"
        ) from exc
    payload = path.read_bytes()
    return CalibratedSearchSourceFileHash(
        role=role,
        repository_path=relative,
        nbytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_calibrated_search_source_runtime_manifest() -> (
    CalibratedSearchSourceRuntimeManifest
):
    """Bind source and observable runtime fields; exact replay remains authoritative.

    Host firmware, opaque compiler decisions, and environment settings not exposed
    through stable public APIs are deliberately not collected.  In particular, no
    environment-variable contents or paths are recorded because they may contain
    secrets.  Bit-exact reconstruction is therefore the final runtime check.
    """

    key = jr.key(0, impl="threefry2x32")
    key_data = jr.key_data(key)
    devices = tuple(jax.devices())
    backend = jax_backend.get_backend()
    provisional = CalibratedSearchSourceRuntimeManifest(
        schema=CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_MANIFEST_SCHEMA,
        source_files=tuple(
            _source_file_hash(role, path) for role, path in _SOURCE_MODULES
        ),
        runtime_identity_scope=RUNTIME_IDENTITY_SCOPE,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_compiler=platform.python_compiler(),
        chex_version=version("chex"),
        jax_version=str(jax.__version__),
        jaxlib_version=version("jaxlib"),
        numpy_version=str(np.__version__),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        backend=str(backend.platform),
        backend_platform_version=str(backend.platform_version),
        device_count=len(devices),
        local_device_count=int(jax.local_device_count()),
        device_platforms=tuple(str(device.platform) for device in devices),
        device_kinds=tuple(str(device.device_kind) for device in devices),
        x64_enabled=bool(jax.config.x64_enabled),
        default_matmul_precision=str(jax.config.jax_default_matmul_precision),
        numpy_dtype_promotion=str(jax.config.jax_numpy_dtype_promotion),
        numpy_rank_promotion=str(jax.config.jax_numpy_rank_promotion),
        threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        default_prng_impl=str(jax.config.jax_default_prng_impl),
        jit_disabled=bool(jax.config.jax_disable_jit),
        runtime_checks_enabled=bool(jax.config.jax_enable_checks),
        prng_impl=str(jr.key_impl(key)),
        prng_key_dtype=str(key.dtype),
        prng_key_data_shape=tuple(int(size) for size in key_data.shape),
        prng_key_data_dtype=str(key_data.dtype),
        manifest_sha256="",
    )
    return dataclasses.replace(
        provisional,
        manifest_sha256=_exact_sha256(provisional),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchModelSnapshot:
    """One immutable pre-outcome model/anchor/value source for all arms."""

    schema: str
    source_manifest_sha256: str
    config_sha256: str
    source_digest: UInt[Array, " 2"]
    representation_generation: Int[Array, ""]
    anchor_bank: Float[Array, "anchor_capacity observation_dim"]
    anchor_active: Bool[Array, " anchor_capacity"]
    initial_q_values: Float[Array, "anchor_capacity n_extended_actions"]
    option_descriptors: Int[Array, "n_options 4"]
    option_generations: Int[Array, " n_options"]
    average_reward: Float[Array, ""]
    primitive_reward_predictions: Float[Array, "anchor_capacity n_primitive_actions"]
    primitive_discount_predictions: Float[Array, "anchor_capacity n_primitive_actions"]
    primitive_next_anchor_probabilities: Float[
        Array, "anchor_capacity n_primitive_actions anchor_capacity"
    ]
    primitive_model_available: Bool[Array, "anchor_capacity n_primitive_actions"]
    primitive_model_support: Int[Array, "anchor_capacity n_primitive_actions"]
    option_return_predictions: Float[Array, "anchor_capacity n_options"]
    option_baseline_mass_predictions: Float[Array, "anchor_capacity n_options"]
    option_discount_predictions: Float[Array, "anchor_capacity n_options"]
    option_next_anchor_probabilities: Float[
        Array, "anchor_capacity n_options anchor_capacity"
    ]
    option_model_available: Bool[Array, "anchor_capacity n_options"]
    option_model_support: Int[Array, "anchor_capacity n_options"]
    option_initiation_available: Bool[Array, "anchor_capacity n_options"]
    learner_revision: Int[Array, ""]
    primitive_model_revision: Int[Array, ""]
    option_model_revision: Int[Array, ""]
    initial_last_realized_targets: Float[Array, " candidate_capacity"]
    initial_last_target_available: Bool[Array, " candidate_capacity"]
    initial_value_change_counts: Int[Array, " candidate_capacity"]
    initial_value_change_means: Float[Array, " candidate_capacity"]
    initial_value_change_m2: Float[Array, " candidate_capacity"]
    initial_model_error_counts: Int[Array, " candidate_capacity"]
    initial_model_error_means: Float[Array, " candidate_capacity"]
    initial_model_error_m2: Float[Array, " candidate_capacity"]
    initial_support_counts: Int[Array, " candidate_capacity"]
    initial_anchor_revisit_trials: Int[Array, " anchor_capacity"]
    initial_anchor_revisit_successes: Int[Array, " anchor_capacity"]
    snapshot_sha256: str


def build_calibrated_search_model_snapshot(
    config: CalibratedExtendedSearchDevelopmentConfig,
    manifest: CalibratedSearchSourceRuntimeManifest,
) -> CalibratedSearchModelSnapshot:
    """Construct the exact frozen model boundary; no live STOMP state is read."""

    config_sha256 = _sha256_json(config.to_config())
    source = _source_words(config_sha256, manifest.manifest_sha256, MODEL_SNAPSHOT_BOUNDARY)
    support = jnp.full(
        (_ANCHOR_CAPACITY, _N_PRIMITIVE_ACTIONS),
        config.num_steps,
        dtype=jnp.int32,
    )
    option_support = jnp.full(
        (_ANCHOR_CAPACITY, _N_OPTIONS), config.num_steps, dtype=jnp.int32
    )
    provisional = CalibratedSearchModelSnapshot(
        schema="alberta.calibrated-extended-search.model-snapshot.v1",
        source_manifest_sha256=manifest.manifest_sha256,
        config_sha256=config_sha256,
        source_digest=source,
        representation_generation=jnp.asarray(1, dtype=jnp.int32),
        anchor_bank=jnp.eye(_ANCHOR_CAPACITY, dtype=jnp.float32),
        anchor_active=jnp.ones((_ANCHOR_CAPACITY,), dtype=jnp.bool_),
        initial_q_values=jnp.zeros(
            (_ANCHOR_CAPACITY, _N_EXTENDED_ACTIONS), dtype=jnp.float32
        ),
        option_descriptors=jnp.asarray(((3, 0, 1, 74),), dtype=jnp.int32),
        option_generations=jnp.asarray((0,), dtype=jnp.int32),
        average_reward=jnp.asarray(0.2, dtype=jnp.float32),
        primitive_reward_predictions=jnp.asarray(
            ((0.8, 0.3), (0.4, 0.9)), dtype=jnp.float32
        ),
        primitive_discount_predictions=jnp.full(
            (_ANCHOR_CAPACITY, _N_PRIMITIVE_ACTIONS), 0.6, dtype=jnp.float32
        ),
        primitive_next_anchor_probabilities=jnp.asarray(
            (
                ((0.8, 0.2), (0.3, 0.7)),
                ((0.4, 0.6), (0.7, 0.3)),
            ),
            dtype=jnp.float32,
        ),
        primitive_model_available=jnp.ones(
            (_ANCHOR_CAPACITY, _N_PRIMITIVE_ACTIONS), dtype=jnp.bool_
        ),
        primitive_model_support=support,
        option_return_predictions=jnp.asarray(((1.2,), (1.4,)), dtype=jnp.float32),
        option_baseline_mass_predictions=jnp.asarray(
            ((1.6,), (1.7,)), dtype=jnp.float32
        ),
        option_discount_predictions=jnp.full(
            (_ANCHOR_CAPACITY, _N_OPTIONS), 0.36, dtype=jnp.float32
        ),
        option_next_anchor_probabilities=jnp.asarray(
            (((0.2, 0.8),), ((0.75, 0.25),)), dtype=jnp.float32
        ),
        option_model_available=jnp.ones(
            (_ANCHOR_CAPACITY, _N_OPTIONS), dtype=jnp.bool_
        ),
        option_model_support=option_support,
        option_initiation_available=jnp.ones(
            (_ANCHOR_CAPACITY, _N_OPTIONS), dtype=jnp.bool_
        ),
        learner_revision=jnp.asarray(0, dtype=jnp.int32),
        primitive_model_revision=jnp.asarray(0, dtype=jnp.int32),
        option_model_revision=jnp.asarray(0, dtype=jnp.int32),
        initial_last_realized_targets=jnp.asarray(
            (0.6, 0.2, 0.1, 0.7, 0.88, 1.06), dtype=jnp.float32
        ),
        initial_last_target_available=jnp.ones(
            (_CANDIDATE_CAPACITY,), dtype=jnp.bool_
        ),
        initial_value_change_counts=jnp.full(
            (_CANDIDATE_CAPACITY,),
            config.calibration_evidence_floor,
            dtype=jnp.int32,
        ),
        initial_value_change_means=jnp.full(
            (_CANDIDATE_CAPACITY,), 0.5, dtype=jnp.float32
        ),
        initial_value_change_m2=jnp.zeros(
            (_CANDIDATE_CAPACITY,), dtype=jnp.float32
        ),
        initial_model_error_counts=jnp.full(
            (_CANDIDATE_CAPACITY,),
            config.calibration_evidence_floor,
            dtype=jnp.int32,
        ),
        initial_model_error_means=jnp.full(
            (_CANDIDATE_CAPACITY,), 0.1, dtype=jnp.float32
        ),
        initial_model_error_m2=jnp.zeros(
            (_CANDIDATE_CAPACITY,), dtype=jnp.float32
        ),
        initial_support_counts=jnp.full(
            (_CANDIDATE_CAPACITY,), config.model_support_floor, dtype=jnp.int32
        ),
        initial_anchor_revisit_trials=jnp.full(
            (_ANCHOR_CAPACITY,),
            config.calibration_evidence_floor,
            dtype=jnp.int32,
        ),
        initial_anchor_revisit_successes=jnp.full(
            (_ANCHOR_CAPACITY,),
            config.calibration_evidence_floor,
            dtype=jnp.int32,
        ),
        snapshot_sha256="",
    )
    return dataclasses.replace(
        provisional,
        snapshot_sha256=_exact_sha256(provisional),
    )


@chex.dataclass(frozen=True)
class CalibratedSearchEvaluatorOwnedTrace:
    """Raw source-reconstructible continuing experience shared by all arms."""

    seed: UInt[Array, ""]
    source_digest: UInt[Array, " 2"]
    decision_ids: UInt[Array, "num_steps 4"]
    decision_anchor_indices: Int[Array, " num_steps"]
    executed_kinds: Int[Array, " num_steps"]
    executed_semantic_indices: Int[Array, " num_steps"]
    future_anchor_indices: Int[Array, " num_steps"]
    external_returns: Float[Array, " num_steps"]
    baseline_masses: Float[Array, " num_steps"]
    terminal_discounts: Float[Array, " num_steps"]
    elapsed_primitive_steps: Int[Array, " num_steps"]
    natural_completions: Bool[Array, " num_steps"]
    censored: Bool[Array, " num_steps"]
    evaluator_key_before: UInt[Array, "num_steps 2"]
    evaluator_key_after: UInt[Array, "num_steps 2"]
    config_sha256_bytes: UInt[Array, " 32"]
    snapshot_sha256_bytes: UInt[Array, " 32"]
    trace_sha256_bytes: UInt[Array, " 32"]


def _hex_bytes(value: str) -> Array:
    return jnp.asarray(tuple(bytes.fromhex(value)), dtype=jnp.uint8)


def _bytes_hex(value: Array) -> str:
    return bytes(np.asarray(jax.device_get(value), dtype=np.uint8)).hex()


def reconstruct_calibrated_search_evaluator_trace(
    config: CalibratedExtendedSearchDevelopmentConfig,
    snapshot: CalibratedSearchModelSnapshot,
) -> CalibratedSearchEvaluatorOwnedTrace:
    """Reconstruct the sole evaluator RNG stream and all real outcomes."""

    t = config.num_steps
    decision_ids = np.zeros((t, 4), dtype=np.uint32)
    anchors = np.zeros((t,), dtype=np.int32)
    kinds = np.zeros((t,), dtype=np.int32)
    semantic_indices = np.zeros((t,), dtype=np.int32)
    future_anchors = np.zeros((t,), dtype=np.int32)
    returns = np.zeros((t,), dtype=np.float32)
    masses = np.ones((t,), dtype=np.float32)
    discounts = np.zeros((t,), dtype=np.float32)
    elapsed = np.ones((t,), dtype=np.int32)
    natural = np.ones((t,), dtype=np.bool_)
    censored = np.zeros((t,), dtype=np.bool_)
    key_before = np.zeros((t, 2), dtype=np.uint32)
    key_after = np.zeros((t, 2), dtype=np.uint32)
    source_words = np.asarray(jax.device_get(snapshot.source_digest), dtype=np.uint32)
    key = jr.key(config.seed, impl="threefry2x32")
    for step in range(t):
        flat_candidate = step % _CANDIDATE_CAPACITY
        extended_index = flat_candidate // _ANCHOR_CAPACITY
        anchor = flat_candidate % _ANCHOR_CAPACITY
        kind = (
            CANDIDATE_KIND_PRIMITIVE
            if extended_index < _N_PRIMITIVE_ACTIONS
            else CANDIDATE_KIND_OPTION
        )
        semantic = (
            extended_index
            if kind == CANDIDATE_KIND_PRIMITIVE
            else extended_index - _N_PRIMITIVE_ACTIONS
        )
        key_before[step] = np.asarray(jax.device_get(jr.key_data(key)))
        key, outcome_key = jr.split(key)
        key_after[step] = np.asarray(jax.device_get(jr.key_data(key)))
        draws = np.asarray(
            jax.device_get(jr.uniform(outcome_key, (2,), dtype=jnp.float32))
        )
        if kind == CANDIDATE_KIND_PRIMITIVE:
            probabilities = np.asarray(
                snapshot.primitive_next_anchor_probabilities[anchor, semantic]
            )
            predicted_return = float(
                snapshot.primitive_reward_predictions[anchor, semantic]
            )
            mass = 1.0
            discount = float(snapshot.primitive_discount_predictions[anchor, semantic])
            duration = 1
        else:
            probabilities = np.asarray(
                snapshot.option_next_anchor_probabilities[anchor, semantic]
            )
            predicted_return = float(snapshot.option_return_predictions[anchor, semantic])
            mass = float(snapshot.option_baseline_mass_predictions[anchor, semantic])
            discount = float(snapshot.option_discount_predictions[anchor, semantic])
            duration = 2 + (step % 2)
        future_anchor = 0 if draws[0] < probabilities[0] else 1
        noise = -0.05 if draws[1] < 0.5 else 0.05
        decision_ids[step] = np.asarray(
            (source_words[0], source_words[1], np.uint32(config.seed), np.uint32(step + 1)),
            dtype=np.uint32,
        )
        anchors[step] = anchor
        kinds[step] = kind
        semantic_indices[step] = semantic
        future_anchors[step] = future_anchor
        returns[step] = np.float32(predicted_return + noise)
        masses[step] = np.float32(mass)
        discounts[step] = np.float32(discount)
        elapsed[step] = duration
    config_hash = _sha256_json(config.to_config())
    provisional = CalibratedSearchEvaluatorOwnedTrace(
        seed=jnp.asarray(config.seed, dtype=jnp.uint32),
        source_digest=snapshot.source_digest,
        decision_ids=jnp.asarray(decision_ids, dtype=jnp.uint32),
        decision_anchor_indices=jnp.asarray(anchors, dtype=jnp.int32),
        executed_kinds=jnp.asarray(kinds, dtype=jnp.int32),
        executed_semantic_indices=jnp.asarray(semantic_indices, dtype=jnp.int32),
        future_anchor_indices=jnp.asarray(future_anchors, dtype=jnp.int32),
        external_returns=jnp.asarray(returns, dtype=jnp.float32),
        baseline_masses=jnp.asarray(masses, dtype=jnp.float32),
        terminal_discounts=jnp.asarray(discounts, dtype=jnp.float32),
        elapsed_primitive_steps=jnp.asarray(elapsed, dtype=jnp.int32),
        natural_completions=jnp.asarray(natural, dtype=jnp.bool_),
        censored=jnp.asarray(censored, dtype=jnp.bool_),
        evaluator_key_before=jnp.asarray(key_before, dtype=jnp.uint32),
        evaluator_key_after=jnp.asarray(key_after, dtype=jnp.uint32),
        config_sha256_bytes=_hex_bytes(config_hash),
        snapshot_sha256_bytes=_hex_bytes(snapshot.snapshot_sha256),
        trace_sha256_bytes=jnp.zeros((32,), dtype=jnp.uint8),
    )
    digest = _exact_sha256(provisional)
    return cast(
        CalibratedSearchEvaluatorOwnedTrace,
        provisional.replace(trace_sha256_bytes=_hex_bytes(digest)),
    )


@chex.dataclass(frozen=True)
class CalibratedSearchArmTrace:
    """Complete causal controller diagnostics for one static arm."""

    arm_transaction_valid: Bool[Array, " num_steps"]
    observe_transaction_valid: Bool[Array, " num_steps"]
    backup_attempts: Int[Array, " num_steps"]
    learner_updates: Int[Array, " num_steps"]
    target_available_count: Int[Array, " num_steps"]
    candidate_eligible_count: Int[Array, " num_steps"]
    value_calibration_ready_count: Int[Array, " num_steps"]
    error_calibration_ready_count: Int[Array, " num_steps"]
    reachability_ready_count: Int[Array, " num_steps"]
    support_ready_count: Int[Array, " num_steps"]
    selected_primitive_count: Int[Array, " num_steps"]
    selected_option_count: Int[Array, " num_steps"]
    natural_resolution: Bool[Array, " num_steps"]
    censored_resolution: Bool[Array, " num_steps"]
    resolved_candidate_index: Int[Array, " num_steps"]
    realized_differential_target: Float[Array, " num_steps"]
    normalized_model_error: Float[Array, " num_steps"]
    priority_sum: Float[Array, " num_steps"]
    priority_max: Float[Array, " num_steps"]
    q_l1_after: Float[Array, " num_steps"]
    state_revision_after: Int[Array, " num_steps"]
    learner_revision_after: Int[Array, " num_steps"]


@chex.dataclass(frozen=True)
class CalibratedSearchDevelopmentState:
    """Fixed-buffer resumable state for all four matched arms."""

    binding_digest: UInt[Array, " 32"]
    step_index: Int[Array, ""]
    controller_states: tuple[
        CalibratedExtendedSearchControlState,
        CalibratedExtendedSearchControlState,
        CalibratedExtendedSearchControlState,
        CalibratedExtendedSearchControlState,
    ]
    arm_transaction_valid: Bool[Array, "4 num_steps"]
    observe_transaction_valid: Bool[Array, "4 num_steps"]
    backup_attempts: Int[Array, "4 num_steps"]
    learner_updates: Int[Array, "4 num_steps"]
    target_available_count: Int[Array, "4 num_steps"]
    candidate_eligible_count: Int[Array, "4 num_steps"]
    value_calibration_ready_count: Int[Array, "4 num_steps"]
    error_calibration_ready_count: Int[Array, "4 num_steps"]
    reachability_ready_count: Int[Array, "4 num_steps"]
    support_ready_count: Int[Array, "4 num_steps"]
    selected_primitive_count: Int[Array, "4 num_steps"]
    selected_option_count: Int[Array, "4 num_steps"]
    natural_resolution: Bool[Array, "4 num_steps"]
    censored_resolution: Bool[Array, "4 num_steps"]
    resolved_candidate_index: Int[Array, "4 num_steps"]
    realized_differential_target: Float[Array, "4 num_steps"]
    normalized_model_error: Float[Array, "4 num_steps"]
    priority_sum: Float[Array, "4 num_steps"]
    priority_max: Float[Array, "4 num_steps"]
    q_l1_after: Float[Array, "4 num_steps"]
    state_revision_after: Int[Array, "4 num_steps"]
    learner_revision_after: Int[Array, "4 num_steps"]


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchArmAccounting:
    mode: str
    real_experience_records: int
    candidate_capacity: int
    shared_backup_budget: int
    expected_backup_attempts: int
    actual_backup_attempts: int
    maximum_learner_updates: int
    actual_learner_updates: int
    evaluator_rng_draws_shared_across_arms: int
    planner_rng_draws: int
    persistent_state_growth_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchArmSummary:
    """Threshold-free raw support/calibration/reachability summary."""

    assessment_status: str
    external_return_sum: float
    natural_resolution_count: int
    censored_resolution_count: int
    eligible_candidate_opportunities: int
    selected_primitive_backups: int
    selected_option_backups: int
    mean_priority: float
    final_q_l1: float
    final_support_counts: tuple[int, ...]
    final_value_change_counts: tuple[int, ...]
    final_model_error_counts: tuple[int, ...]
    final_anchor_revisit_trials: tuple[int, ...]
    final_anchor_revisit_successes: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchArmRecord:
    arm_index: int
    mode: str
    assessment_status: str
    config_sha256: str
    protocol_sha256: str
    source_manifest_sha256: str
    model_snapshot_sha256: str
    evaluator_trace_sha256: str
    initial_snapshot_shared: bool
    resource_budget: CalibratedExtendedSearchControlResourceBudget
    accounting: CalibratedSearchArmAccounting
    summary: CalibratedSearchArmSummary
    trace: CalibratedSearchArmTrace
    final_controller_state: CalibratedExtendedSearchControlState
    final_controller_state_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchDevelopmentContrast:
    """One required descriptive contrast with no decision threshold."""

    name: str
    left_mode: str
    right_mode: str
    assessment_status: str
    learner_update_difference_right_minus_left: int
    selected_option_difference_right_minus_left: int
    mean_priority_difference_right_minus_left: float
    final_q_l1_distance: float
    threshold: None
    verdict: None


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchMatchedAudit:
    arm_count: int
    trace_reconstruction_passed: bool
    immutable_initial_snapshot_passed: bool
    common_random_number_trace_passed: bool
    identical_anchor_bank_passed: bool
    identical_candidate_opportunities_passed: bool
    equal_real_experience_passed: bool
    one_shared_backup_budget_passed: bool
    source_binding_passed: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchEvaluatorAccounting:
    """Exact evaluator allocations, work, and finite terminal ceilings."""

    num_steps: int
    arm_count: int
    candidate_capacity: int
    shared_backup_budget: int
    unique_real_experience_records: int
    total_matched_real_experience_deliveries: int
    total_backup_attempts: int
    maximum_total_learner_updates: int
    actual_total_learner_updates: int
    evaluator_generator_split_calls: int
    evaluator_uniform_generator_calls: int
    evaluator_scalar_random_draws: int
    planner_random_draws: int
    source_snapshot_array_nbytes: int
    evaluator_trace_array_nbytes: int
    resumable_state_array_nbytes: int
    per_observation_state_growth_bytes: int
    load_bearing_source_nbytes: int
    evaluator_num_steps_terminal_cap: int
    controller_signed_int32_counter_terminal_cap: int
    evaluator_uint32_seed_terminal_cap: int


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchDevelopmentSuite:
    """Complete four-arm raw suite; never an evidence artifact."""

    schema: str
    status: str
    assessment_status: str
    development_only: bool
    seed_role: str
    consumed_development_seed: int
    held_out_seeds_used: bool
    thresholds_frozen: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    policy_authority: bool
    model_snapshot_boundary: str
    source_calibration_boundary: str
    config: CalibratedExtendedSearchDevelopmentConfig
    config_sha256: str
    protocol_sha256: str
    source_runtime_manifest: CalibratedSearchSourceRuntimeManifest
    model_snapshot: CalibratedSearchModelSnapshot
    evaluator_trace: CalibratedSearchEvaluatorOwnedTrace
    canonical_arm_order: tuple[str, str, str, str]
    arm_records: tuple[
        CalibratedSearchArmRecord,
        CalibratedSearchArmRecord,
        CalibratedSearchArmRecord,
        CalibratedSearchArmRecord,
    ]
    contrasts: tuple[
        CalibratedSearchDevelopmentContrast,
        CalibratedSearchDevelopmentContrast,
    ]
    matched_audit: CalibratedSearchMatchedAudit
    evaluator_accounting: CalibratedSearchEvaluatorAccounting
    suite_binding_sha256: str
    replay_authenticator_sha256: str
    authenticated_replay_verified: bool
    thresholds: None
    aggregate_verdict: None
    artifact_output_path: None


@dataclasses.dataclass(frozen=True, slots=True)
class CalibratedSearchAuthenticatedReplayValidation:
    schema: str
    assessment_status: str
    source_runtime_verified: bool
    structural_validation_passed: bool
    authenticated_replay_verified: bool
    replay_suite_binding_sha256: str | None
    errors: tuple[str, ...]


def _controller_config(
    config: CalibratedExtendedSearchDevelopmentConfig, mode: str
) -> CalibratedExtendedSearchControlConfig:
    return CalibratedExtendedSearchControlConfig(
        mode=mode,
        observation_dim=_OBSERVATION_DIM,
        anchor_capacity=_ANCHOR_CAPACITY,
        n_primitive_actions=_N_PRIMITIVE_ACTIONS,
        n_options=_N_OPTIONS,
        backup_budget=config.backup_budget,
        calibration_evidence_floor=config.calibration_evidence_floor,
        model_support_floor=config.model_support_floor,
        confidence_scale=config.confidence_scale,
        support_prior=config.support_prior,
        model_error_scale=config.model_error_scale,
        backup_step_size=config.backup_step_size,
        max_observations=config.max_observations,
    )


def _trace_sha256(trace: CalibratedSearchEvaluatorOwnedTrace) -> str:
    return _bytes_hex(trace.trace_sha256_bytes)


def _state_binding_digest(
    config_sha256: str,
    protocol_sha256: str,
    manifest_sha256: str,
    snapshot_sha256: str,
    trace_sha256: str,
) -> Array:
    digest = hashlib.sha256(
        "\x00".join(
            (
                config_sha256,
                protocol_sha256,
                manifest_sha256,
                snapshot_sha256,
                trace_sha256,
            )
        ).encode("ascii")
    ).digest()
    return jnp.asarray(tuple(digest), dtype=jnp.uint8)


class CalibratedExtendedSearchDevelopmentEvaluator:
    """Host orchestrator for one exact matched four-arm development life."""

    def __init__(self, config: CalibratedExtendedSearchDevelopmentConfig) -> None:
        if type(config) is not CalibratedExtendedSearchDevelopmentConfig:
            raise TypeError(
                "config must be an exact CalibratedExtendedSearchDevelopmentConfig"
            )
        self.config = config
        self.config_sha256 = _sha256_json(config.to_config())
        self.source_runtime_manifest = build_calibrated_search_source_runtime_manifest()
        self.controllers = tuple(
            CalibratedExtendedSearchControl(_controller_config(config, mode))
            for mode in CANONICAL_ARM_ORDER
        )
        self.model_snapshot = build_calibrated_search_model_snapshot(
            config, self.source_runtime_manifest
        )
        self.evaluator_trace = reconstruct_calibrated_search_evaluator_trace(
            config, self.model_snapshot
        )
        self.protocol_sha256 = _sha256_json(
            {
                "schema": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_SUITE_SCHEMA,
                "config_sha256": self.config_sha256,
                "source_manifest_sha256": (
                    self.source_runtime_manifest.manifest_sha256
                ),
                "model_snapshot_sha256": self.model_snapshot.snapshot_sha256,
                "evaluator_trace_sha256": _trace_sha256(self.evaluator_trace),
                "canonical_arm_order": CANONICAL_ARM_ORDER,
                "controller_configs": tuple(
                    controller.to_config() for controller in self.controllers
                ),
                "model_snapshot_boundary": MODEL_SNAPSHOT_BOUNDARY,
                "source_calibration_boundary": SOURCE_CALIBRATION_BOUNDARY,
                "trace_semantics": TRACE_SEMANTICS,
                "shared_backup_budget": config.backup_budget,
                "thresholds": None,
                "scientific_promotion_allowed": False,
            }
        )
        self.binding_digest = _state_binding_digest(
            self.config_sha256,
            self.protocol_sha256,
            self.source_runtime_manifest.manifest_sha256,
            self.model_snapshot.snapshot_sha256,
            _trace_sha256(self.evaluator_trace),
        )

    def _assert_current_source_runtime(self) -> None:
        current = build_calibrated_search_source_runtime_manifest()
        if current != self.source_runtime_manifest:
            raise CalibratedExtendedSearchDevelopmentError(
                "source/runtime manifest changed after evaluator construction"
            )

    def _initial_controller_state(
        self, controller: CalibratedExtendedSearchControl
    ) -> CalibratedExtendedSearchControlState:
        snapshot = self.model_snapshot
        initial = controller.init(
            anchor_bank=snapshot.anchor_bank,
            anchor_active=snapshot.anchor_active,
            q_values=snapshot.initial_q_values,
            option_descriptors=snapshot.option_descriptors,
            option_generations=snapshot.option_generations,
            representation_generation=snapshot.representation_generation,
            source_digest=snapshot.source_digest,
            learner_revision=snapshot.learner_revision,
            primitive_model_revision=snapshot.primitive_model_revision,
            option_model_revision=snapshot.option_model_revision,
        )
        return cast(
            CalibratedExtendedSearchControlState,
            initial.replace(
                last_realized_targets=snapshot.initial_last_realized_targets,
                last_target_available=snapshot.initial_last_target_available,
                value_change_counts=snapshot.initial_value_change_counts,
                value_change_means=snapshot.initial_value_change_means,
                value_change_m2=snapshot.initial_value_change_m2,
                model_error_counts=snapshot.initial_model_error_counts,
                model_error_means=snapshot.initial_model_error_means,
                model_error_m2=snapshot.initial_model_error_m2,
                support_counts=snapshot.initial_support_counts,
                anchor_revisit_trials=snapshot.initial_anchor_revisit_trials,
                anchor_revisit_successes=snapshot.initial_anchor_revisit_successes,
            ),
        )

    def init(self) -> CalibratedSearchDevelopmentState:
        t = self.config.num_steps
        shape = (len(CANONICAL_ARM_ORDER), t)
        zeros_i = jnp.zeros(shape, dtype=jnp.int32)
        zeros_f = jnp.zeros(shape, dtype=jnp.float32)
        zeros_b = jnp.zeros(shape, dtype=jnp.bool_)
        negative_i = jnp.full(shape, -1, dtype=jnp.int32)
        states = tuple(
            self._initial_controller_state(controller)
            for controller in self.controllers
        )
        return CalibratedSearchDevelopmentState(
            binding_digest=self.binding_digest,
            step_index=jnp.asarray(0, dtype=jnp.int32),
            controller_states=cast(
                tuple[
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                ],
                states,
            ),
            arm_transaction_valid=zeros_b,
            observe_transaction_valid=zeros_b,
            backup_attempts=zeros_i,
            learner_updates=zeros_i,
            target_available_count=zeros_i,
            candidate_eligible_count=zeros_i,
            value_calibration_ready_count=zeros_i,
            error_calibration_ready_count=zeros_i,
            reachability_ready_count=zeros_i,
            support_ready_count=zeros_i,
            selected_primitive_count=zeros_i,
            selected_option_count=zeros_i,
            natural_resolution=zeros_b,
            censored_resolution=zeros_b,
            resolved_candidate_index=negative_i,
            realized_differential_target=zeros_f,
            normalized_model_error=zeros_f,
            priority_sum=zeros_f,
            priority_max=zeros_f,
            q_l1_after=zeros_f,
            state_revision_after=zeros_i,
            learner_revision_after=zeros_i,
        )

    def _state_errors(self, state: object) -> tuple[str, ...]:
        if type(state) is not CalibratedSearchDevelopmentState:
            return ("state must be an exact CalibratedSearchDevelopmentState",)
        checked = state
        errors: list[str] = []
        if not np.array_equal(
            np.asarray(checked.binding_digest), np.asarray(self.binding_digest)
        ):
            errors.append("state binding digest differs")
        step = int(jax.device_get(checked.step_index))
        if not 0 <= step <= self.config.num_steps:
            errors.append("state step index is out of bounds")
            return tuple(errors)
        if len(checked.controller_states) != len(CANONICAL_ARM_ORDER):
            errors.append("state does not contain exactly four controller states")
            return tuple(errors)
        snapshot = self.model_snapshot
        for index, (controller, controller_state) in enumerate(
            zip(self.controllers, checked.controller_states, strict=True)
        ):
            try:
                valid = controller.validate_state(
                    controller_state,
                    representation_generation=snapshot.representation_generation,
                    source_digest=snapshot.source_digest,
                    option_descriptors=snapshot.option_descriptors,
                    option_generations=snapshot.option_generations,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"controller state {index} contract failed: {exc}")
                continue
            if not bool(jax.device_get(valid)):
                errors.append(f"controller state {index} is invalid")
            if bool(jax.device_get(controller_state.pending)):
                errors.append(f"controller state {index} is pending at checkpoint boundary")
            if int(jax.device_get(controller_state.state_revision)) != step:
                errors.append(f"controller state {index} revision differs from evaluator step")
        shape = (len(CANONICAL_ARM_ORDER), self.config.num_steps)
        arrays = (
            (checked.arm_transaction_valid, "arm_transaction_valid", jnp.bool_),
            (checked.observe_transaction_valid, "observe_transaction_valid", jnp.bool_),
            (checked.backup_attempts, "backup_attempts", jnp.int32),
            (checked.learner_updates, "learner_updates", jnp.int32),
            (checked.target_available_count, "target_available_count", jnp.int32),
            (checked.candidate_eligible_count, "candidate_eligible_count", jnp.int32),
            (
                checked.value_calibration_ready_count,
                "value_calibration_ready_count",
                jnp.int32,
            ),
            (
                checked.error_calibration_ready_count,
                "error_calibration_ready_count",
                jnp.int32,
            ),
            (checked.reachability_ready_count, "reachability_ready_count", jnp.int32),
            (checked.support_ready_count, "support_ready_count", jnp.int32),
            (checked.selected_primitive_count, "selected_primitive_count", jnp.int32),
            (checked.selected_option_count, "selected_option_count", jnp.int32),
            (checked.natural_resolution, "natural_resolution", jnp.bool_),
            (checked.censored_resolution, "censored_resolution", jnp.bool_),
            (checked.resolved_candidate_index, "resolved_candidate_index", jnp.int32),
            (
                checked.realized_differential_target,
                "realized_differential_target",
                jnp.float32,
            ),
            (checked.normalized_model_error, "normalized_model_error", jnp.float32),
            (checked.priority_sum, "priority_sum", jnp.float32),
            (checked.priority_max, "priority_max", jnp.float32),
            (checked.q_l1_after, "q_l1_after", jnp.float32),
            (checked.state_revision_after, "state_revision_after", jnp.int32),
            (checked.learner_revision_after, "learner_revision_after", jnp.int32),
        )
        for array, name, dtype in arrays:
            if array.shape != shape or array.dtype != dtype:
                errors.append(f"state {name} has the wrong static contract")
        float_arrays = tuple(array for array, _, dtype in arrays if dtype == jnp.float32)
        if any(not bool(jnp.all(jnp.isfinite(array))) for array in float_arrays):
            errors.append("state contains non-finite diagnostic values")
        prefix = np.s_[:, :step]
        tail = np.s_[:, step:]
        if step and (
            not bool(jnp.all(checked.arm_transaction_valid[prefix]))
            or not bool(jnp.all(checked.observe_transaction_valid[prefix]))
        ):
            errors.append("completed prefix contains an invalid transaction")
        if step and not bool(
            jnp.all(checked.backup_attempts[prefix] == self.config.backup_budget)
        ):
            errors.append("completed prefix violates the exact shared backup budget")
        if bool(jnp.any(checked.learner_updates[prefix] > self.config.backup_budget)):
            errors.append("completed prefix exceeds the learner-update budget")
        zero_tail_arrays = (
            checked.arm_transaction_valid,
            checked.observe_transaction_valid,
            checked.backup_attempts,
            checked.learner_updates,
            checked.target_available_count,
            checked.candidate_eligible_count,
            checked.value_calibration_ready_count,
            checked.error_calibration_ready_count,
            checked.reachability_ready_count,
            checked.support_ready_count,
            checked.selected_primitive_count,
            checked.selected_option_count,
            checked.natural_resolution,
            checked.censored_resolution,
            checked.realized_differential_target,
            checked.normalized_model_error,
            checked.priority_sum,
            checked.priority_max,
            checked.q_l1_after,
            checked.state_revision_after,
            checked.learner_revision_after,
        )
        if any(bool(jnp.any(array[tail] != 0)) for array in zero_tail_arrays):
            errors.append("unexecuted diagnostic tail is nonzero")
        if bool(jnp.any(checked.resolved_candidate_index[tail] != -1)):
            errors.append("unexecuted resolved-candidate tail is not -1")
        return tuple(dict.fromkeys(errors))

    def validate_state(self, state: object) -> tuple[str, ...]:
        return self._state_errors(state)

    def _canonical_prefix_errors(
        self, state: CalibratedSearchDevelopmentState
    ) -> tuple[str, ...]:
        """Replay and bit-compare every causal field through ``step_index``."""

        step = int(jax.device_get(state.step_index))
        expected = self.advance(self.init(), steps=step)
        errors: list[str] = []
        for index, (observed_state, expected_state) in enumerate(
            zip(state.controller_states, expected.controller_states, strict=True)
        ):
            if not _host_tree_equal(observed_state, expected_state):
                errors.append(f"controller state {index} differs from canonical prefix replay")
        for field in dataclasses.fields(CalibratedSearchDevelopmentState):
            if field.name in {"binding_digest", "step_index", "controller_states"}:
                continue
            observed_array = cast(Array, getattr(state, field.name))
            expected_array = cast(Array, getattr(expected, field.name))
            if not _host_array_prefix_bits_equal(observed_array, expected_array, step):
                errors.append(
                    f"diagnostic {field.name} differs from canonical prefix replay"
                )
        return tuple(errors)

    def _arm_kwargs(
        self,
        controller_state: CalibratedExtendedSearchControlState,
        step: int,
    ) -> dict[str, Array]:
        snapshot = self.model_snapshot
        trace = self.evaluator_trace
        anchor = trace.decision_anchor_indices[step]
        return {
            "decision_id": trace.decision_ids[step],
            "decision_observation": snapshot.anchor_bank[anchor],
            "decision_anchor_index": anchor,
            "executed_kind": trace.executed_kinds[step],
            "executed_index": trace.executed_semantic_indices[step],
            "average_reward": snapshot.average_reward,
            "primitive_reward_predictions": snapshot.primitive_reward_predictions,
            "primitive_discount_predictions": snapshot.primitive_discount_predictions,
            "primitive_next_anchor_probabilities": (
                snapshot.primitive_next_anchor_probabilities
            ),
            "primitive_model_available": snapshot.primitive_model_available,
            "primitive_model_support": snapshot.primitive_model_support,
            "option_return_predictions": snapshot.option_return_predictions,
            "option_baseline_mass_predictions": (
                snapshot.option_baseline_mass_predictions
            ),
            "option_discount_predictions": snapshot.option_discount_predictions,
            "option_next_anchor_probabilities": snapshot.option_next_anchor_probabilities,
            "option_model_available": snapshot.option_model_available,
            "option_model_support": snapshot.option_model_support,
            "option_initiation_available": snapshot.option_initiation_available,
            "representation_generation": snapshot.representation_generation,
            "source_digest": snapshot.source_digest,
            "option_descriptors": snapshot.option_descriptors,
            "option_generations": snapshot.option_generations,
            "learner_revision": controller_state.learner_revision,
            "primitive_model_revision": snapshot.primitive_model_revision,
            "option_model_revision": snapshot.option_model_revision,
        }

    def _observe_kwargs(
        self,
        armed: CalibratedExtendedSearchControlState,
        step: int,
    ) -> dict[str, Array]:
        snapshot = self.model_snapshot
        trace = self.evaluator_trace
        future_anchor = trace.future_anchor_indices[step]
        observed_mask = jnp.where(
            trace.censored[step],
            jnp.zeros((_ANCHOR_CAPACITY,), dtype=jnp.bool_),
            jnp.arange(_ANCHOR_CAPACITY, dtype=jnp.int32) == future_anchor,
        )
        return {
            "decision_id": trace.decision_ids[step],
            "future_observation": snapshot.anchor_bank[future_anchor],
            "observed_future_anchor_mask": observed_mask,
            "external_return": trace.external_returns[step],
            "baseline_mass": trace.baseline_masses[step],
            "terminal_discount": trace.terminal_discounts[step],
            "elapsed_primitive_steps": trace.elapsed_primitive_steps[step],
            "natural_completion": trace.natural_completions[step],
            "censored": trace.censored[step],
            "representation_generation": snapshot.representation_generation,
            "source_digest": snapshot.source_digest,
            "option_descriptors": snapshot.option_descriptors,
            "option_generations": snapshot.option_generations,
            "learner_revision": armed.learner_revision,
            "primitive_model_revision": snapshot.primitive_model_revision,
            "option_model_revision": snapshot.option_model_revision,
        }

    def advance(
        self,
        state: CalibratedSearchDevelopmentState,
        *,
        steps: int,
    ) -> CalibratedSearchDevelopmentState:
        """Advance exactly ``steps`` complete real transactions for every arm."""

        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a non-negative exact Python int")
        self._assert_current_source_runtime()
        errors = self._state_errors(state)
        if errors:
            raise CalibratedExtendedSearchDevelopmentError(
                "cannot advance invalid evaluator state: " + "; ".join(errors)
            )
        start = int(jax.device_get(state.step_index))
        if start + steps > self.config.num_steps:
            raise ValueError("advance would exceed the fixed development trace")
        current = state
        for step in range(start, start + steps):
            controller_states = list(current.controller_states)
            arrays = {
                field.name: getattr(current, field.name)
                for field in dataclasses.fields(CalibratedSearchDevelopmentState)
                if field.name not in {"binding_digest", "step_index", "controller_states"}
            }
            for arm_index, controller in enumerate(self.controllers):
                pre = controller_states[arm_index]
                arm = controller.arm(pre, **self._arm_kwargs(pre, step))
                observed = controller.observe(
                    arm.state, **self._observe_kwargs(arm.state, step)
                )
                if not bool(jax.device_get(arm.diagnostics.transaction_valid)):
                    raise CalibratedExtendedSearchDevelopmentError(
                        f"arm {arm_index} step {step} failed the arm transaction"
                    )
                if not bool(jax.device_get(observed.diagnostics.transaction_valid)):
                    raise CalibratedExtendedSearchDevelopmentError(
                        f"arm {arm_index} step {step} failed observation"
                    )
                controller_states[arm_index] = observed.state
                selected_valid = arm.diagnostics.selected_valid
                selected_kinds = arm.diagnostics.selected_kinds
                value_ready = (
                    pre.value_change_counts >= self.config.calibration_evidence_floor
                )
                error_ready = (
                    pre.model_error_counts >= self.config.calibration_evidence_floor
                )
                reach_ready = (
                    pre.anchor_revisit_trials[
                        controller._candidate_anchor_indices  # noqa: SLF001
                    ]
                    >= self.config.calibration_evidence_floor
                )
                support_ready = pre.support_counts >= self.config.model_support_floor
                values: dict[str, Array] = {
                    "arm_transaction_valid": arm.diagnostics.transaction_valid,
                    "observe_transaction_valid": (
                        observed.diagnostics.transaction_valid
                    ),
                    "backup_attempts": arm.diagnostics.backup_attempt_count,
                    "learner_updates": observed.diagnostics.learner_update_count,
                    "target_available_count": jnp.sum(
                        arm.diagnostics.target_available.astype(jnp.int32)
                    ),
                    "candidate_eligible_count": jnp.sum(
                        arm.diagnostics.candidate_eligible.astype(jnp.int32)
                    ),
                    "value_calibration_ready_count": jnp.sum(
                        value_ready.astype(jnp.int32)
                    ),
                    "error_calibration_ready_count": jnp.sum(
                        error_ready.astype(jnp.int32)
                    ),
                    "reachability_ready_count": jnp.sum(
                        reach_ready.astype(jnp.int32)
                    ),
                    "support_ready_count": jnp.sum(
                        support_ready.astype(jnp.int32)
                    ),
                    "selected_primitive_count": jnp.sum(
                        (
                            selected_valid
                            & (selected_kinds == CANDIDATE_KIND_PRIMITIVE)
                        ).astype(jnp.int32)
                    ),
                    "selected_option_count": jnp.sum(
                        (
                            selected_valid
                            & (selected_kinds == CANDIDATE_KIND_OPTION)
                        ).astype(jnp.int32)
                    ),
                    "natural_resolution": observed.diagnostics.natural_resolution,
                    "censored_resolution": observed.diagnostics.censored_resolution,
                    "resolved_candidate_index": (
                        observed.diagnostics.resolved_candidate_index
                    ),
                    "realized_differential_target": (
                        observed.diagnostics.realized_differential_target
                    ),
                    "normalized_model_error": (
                        observed.diagnostics.normalized_model_error
                    ),
                    "priority_sum": jnp.sum(arm.diagnostics.priorities),
                    "priority_max": jnp.max(arm.diagnostics.priorities),
                    "q_l1_after": jnp.sum(jnp.abs(observed.state.q_values)),
                    "state_revision_after": observed.state.state_revision,
                    "learner_revision_after": observed.state.learner_revision,
                }
                for name, value in values.items():
                    arrays[name] = arrays[name].at[arm_index, step].set(value)
            current = cast(
                CalibratedSearchDevelopmentState,
                current.replace(
                    step_index=jnp.asarray(step + 1, dtype=jnp.int32),
                    controller_states=tuple(controller_states),
                    **arrays,
                ),
            )
        errors = self._state_errors(current)
        if errors:
            raise CalibratedExtendedSearchDevelopmentError(
                "advanced evaluator state failed validation: " + "; ".join(errors)
            )
        return current

    def _arm_trace(
        self, state: CalibratedSearchDevelopmentState, arm_index: int
    ) -> CalibratedSearchArmTrace:
        return CalibratedSearchArmTrace(
            arm_transaction_valid=state.arm_transaction_valid[arm_index],
            observe_transaction_valid=state.observe_transaction_valid[arm_index],
            backup_attempts=state.backup_attempts[arm_index],
            learner_updates=state.learner_updates[arm_index],
            target_available_count=state.target_available_count[arm_index],
            candidate_eligible_count=state.candidate_eligible_count[arm_index],
            value_calibration_ready_count=(
                state.value_calibration_ready_count[arm_index]
            ),
            error_calibration_ready_count=(
                state.error_calibration_ready_count[arm_index]
            ),
            reachability_ready_count=state.reachability_ready_count[arm_index],
            support_ready_count=state.support_ready_count[arm_index],
            selected_primitive_count=state.selected_primitive_count[arm_index],
            selected_option_count=state.selected_option_count[arm_index],
            natural_resolution=state.natural_resolution[arm_index],
            censored_resolution=state.censored_resolution[arm_index],
            resolved_candidate_index=state.resolved_candidate_index[arm_index],
            realized_differential_target=(
                state.realized_differential_target[arm_index]
            ),
            normalized_model_error=state.normalized_model_error[arm_index],
            priority_sum=state.priority_sum[arm_index],
            priority_max=state.priority_max[arm_index],
            q_l1_after=state.q_l1_after[arm_index],
            state_revision_after=state.state_revision_after[arm_index],
            learner_revision_after=state.learner_revision_after[arm_index],
        )

    def _arm_summary(
        self,
        state: CalibratedSearchDevelopmentState,
        arm_index: int,
    ) -> CalibratedSearchArmSummary:
        final = state.controller_states[arm_index]
        trace = self._arm_trace(state, arm_index)
        return CalibratedSearchArmSummary(
            assessment_status=ASSESSMENT_STATUS,
            external_return_sum=float(
                np.sum(np.asarray(self.evaluator_trace.external_returns), dtype=np.float64)
            ),
            natural_resolution_count=int(jnp.sum(trace.natural_resolution)),
            censored_resolution_count=int(jnp.sum(trace.censored_resolution)),
            eligible_candidate_opportunities=int(
                jnp.sum(trace.candidate_eligible_count)
            ),
            selected_primitive_backups=int(jnp.sum(trace.selected_primitive_count)),
            selected_option_backups=int(jnp.sum(trace.selected_option_count)),
            mean_priority=float(jnp.mean(trace.priority_sum)),
            final_q_l1=float(jnp.sum(jnp.abs(final.q_values))),
            final_support_counts=tuple(
                int(value) for value in np.asarray(final.support_counts)
            ),
            final_value_change_counts=tuple(
                int(value) for value in np.asarray(final.value_change_counts)
            ),
            final_model_error_counts=tuple(
                int(value) for value in np.asarray(final.model_error_counts)
            ),
            final_anchor_revisit_trials=tuple(
                int(value) for value in np.asarray(final.anchor_revisit_trials)
            ),
            final_anchor_revisit_successes=tuple(
                int(value) for value in np.asarray(final.anchor_revisit_successes)
            ),
        )

    def _arm_record(
        self, state: CalibratedSearchDevelopmentState, arm_index: int
    ) -> CalibratedSearchArmRecord:
        mode = CANONICAL_ARM_ORDER[arm_index]
        trace = self._arm_trace(state, arm_index)
        final = state.controller_states[arm_index]
        attempts = int(jnp.sum(trace.backup_attempts))
        updates = int(jnp.sum(trace.learner_updates))
        accounting = CalibratedSearchArmAccounting(
            mode=mode,
            real_experience_records=self.config.num_steps,
            candidate_capacity=_CANDIDATE_CAPACITY,
            shared_backup_budget=self.config.backup_budget,
            expected_backup_attempts=self.config.num_steps * self.config.backup_budget,
            actual_backup_attempts=attempts,
            maximum_learner_updates=self.config.num_steps * self.config.backup_budget,
            actual_learner_updates=updates,
            evaluator_rng_draws_shared_across_arms=2 * self.config.num_steps,
            planner_rng_draws=0,
            persistent_state_growth_bytes=0,
        )
        return CalibratedSearchArmRecord(
            arm_index=arm_index,
            mode=mode,
            assessment_status=ASSESSMENT_STATUS,
            config_sha256=self.config_sha256,
            protocol_sha256=self.protocol_sha256,
            source_manifest_sha256=self.source_runtime_manifest.manifest_sha256,
            model_snapshot_sha256=self.model_snapshot.snapshot_sha256,
            evaluator_trace_sha256=_trace_sha256(self.evaluator_trace),
            initial_snapshot_shared=True,
            resource_budget=self.controllers[arm_index].resource_budget,
            accounting=accounting,
            summary=self._arm_summary(state, arm_index),
            trace=trace,
            final_controller_state=final,
            final_controller_state_sha256=_exact_sha256(final),
        )

    @staticmethod
    def _contrast(
        name: str,
        left: CalibratedSearchArmRecord,
        right: CalibratedSearchArmRecord,
    ) -> CalibratedSearchDevelopmentContrast:
        left_q = np.asarray(left.final_controller_state.q_values)
        right_q = np.asarray(right.final_controller_state.q_values)
        return CalibratedSearchDevelopmentContrast(
            name=name,
            left_mode=left.mode,
            right_mode=right.mode,
            assessment_status=ASSESSMENT_STATUS,
            learner_update_difference_right_minus_left=(
                right.accounting.actual_learner_updates
                - left.accounting.actual_learner_updates
            ),
            selected_option_difference_right_minus_left=(
                right.summary.selected_option_backups
                - left.summary.selected_option_backups
            ),
            mean_priority_difference_right_minus_left=(
                right.summary.mean_priority - left.summary.mean_priority
            ),
            final_q_l1_distance=float(np.sum(np.abs(right_q - left_q), dtype=np.float64)),
            threshold=None,
            verdict=None,
        )

    def _matched_audit(
        self, records: tuple[CalibratedSearchArmRecord, ...]
    ) -> CalibratedSearchMatchedAudit:
        errors: list[str] = []
        reconstructed = reconstruct_calibrated_search_evaluator_trace(
            self.config, self.model_snapshot
        )

        trace_passed = _host_tree_equal(reconstructed, self.evaluator_trace)
        if not trace_passed:
            errors.append("evaluator trace reconstruction differs")
        snapshot_hashes = {record.model_snapshot_sha256 for record in records}
        snapshot_passed = snapshot_hashes == {self.model_snapshot.snapshot_sha256}
        if not snapshot_passed:
            errors.append("arms do not share one immutable initial snapshot")
        trace_hashes = {record.evaluator_trace_sha256 for record in records}
        crn_passed = trace_hashes == {_trace_sha256(self.evaluator_trace)}
        if not crn_passed:
            errors.append("arms do not share one evaluator-owned trace")
        anchor_passed = all(
            np.array_equal(
                np.asarray(record.final_controller_state.anchor_bank),
                np.asarray(self.model_snapshot.anchor_bank),
            )
            for record in records
        )
        if not anchor_passed:
            errors.append("an arm changed the fixed real-anchor bank")
        opportunity_passed = all(
            record.accounting.candidate_capacity == _CANDIDATE_CAPACITY
            and record.resource_budget.candidate_capacity == _CANDIDATE_CAPACITY
            for record in records
        )
        if not opportunity_passed:
            errors.append("candidate capacities differ across arms")
        experience_passed = all(
            record.accounting.real_experience_records == self.config.num_steps
            for record in records
        )
        if not experience_passed:
            errors.append("real experience differs across arms")
        budget_passed = all(
            record.accounting.shared_backup_budget == self.config.backup_budget
            and record.accounting.actual_backup_attempts
            == self.config.num_steps * self.config.backup_budget
            and bool(jnp.all(record.trace.backup_attempts == self.config.backup_budget))
            for record in records
        )
        if not budget_passed:
            errors.append("one shared B-attempt budget is not exact")
        source_passed = all(
            record.source_manifest_sha256
            == self.source_runtime_manifest.manifest_sha256
            and record.protocol_sha256 == self.protocol_sha256
            for record in records
        )
        if not source_passed:
            errors.append("an arm differs from the source/protocol binding")
        return CalibratedSearchMatchedAudit(
            arm_count=len(records),
            trace_reconstruction_passed=trace_passed,
            immutable_initial_snapshot_passed=snapshot_passed,
            common_random_number_trace_passed=crn_passed,
            identical_anchor_bank_passed=anchor_passed,
            identical_candidate_opportunities_passed=opportunity_passed,
            equal_real_experience_passed=experience_passed,
            one_shared_backup_budget_passed=budget_passed,
            source_binding_passed=source_passed,
            errors=tuple(errors),
        )

    def _evaluator_accounting(
        self,
        state: CalibratedSearchDevelopmentState,
        records: tuple[CalibratedSearchArmRecord, ...],
    ) -> CalibratedSearchEvaluatorAccounting:
        t = self.config.num_steps
        arms = len(CANONICAL_ARM_ORDER)
        return CalibratedSearchEvaluatorAccounting(
            num_steps=t,
            arm_count=arms,
            candidate_capacity=_CANDIDATE_CAPACITY,
            shared_backup_budget=self.config.backup_budget,
            unique_real_experience_records=t,
            total_matched_real_experience_deliveries=arms * t,
            total_backup_attempts=arms * t * self.config.backup_budget,
            maximum_total_learner_updates=arms * t * self.config.backup_budget,
            actual_total_learner_updates=sum(
                record.accounting.actual_learner_updates for record in records
            ),
            evaluator_generator_split_calls=t,
            evaluator_uniform_generator_calls=t,
            evaluator_scalar_random_draws=2 * t,
            planner_random_draws=0,
            source_snapshot_array_nbytes=_tree_array_nbytes(self.model_snapshot),
            evaluator_trace_array_nbytes=_tree_array_nbytes(self.evaluator_trace),
            resumable_state_array_nbytes=_tree_array_nbytes(state),
            per_observation_state_growth_bytes=0,
            load_bearing_source_nbytes=sum(
                item.nbytes for item in self.source_runtime_manifest.source_files
            ),
            evaluator_num_steps_terminal_cap=_MAX_STEPS,
            controller_signed_int32_counter_terminal_cap=2**31 - 1,
            evaluator_uint32_seed_terminal_cap=2**32 - 1,
        )

    @staticmethod
    def _suite_binding(suite: CalibratedSearchDevelopmentSuite) -> str:
        return _exact_sha256(
            dataclasses.replace(
                suite,
                suite_binding_sha256="",
                replay_authenticator_sha256="",
            )
        )

    @staticmethod
    def _replay_authenticator(suite: CalibratedSearchDevelopmentSuite) -> str:
        return hashlib.sha256(
            "\x00".join(
                (
                    CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_REPLAY_SCHEMA,
                    suite.source_runtime_manifest.manifest_sha256,
                    suite.protocol_sha256,
                    suite.suite_binding_sha256,
                )
            ).encode("ascii")
        ).hexdigest()

    def finalize(
        self, state: CalibratedSearchDevelopmentState
    ) -> CalibratedSearchDevelopmentSuite:
        """Build the threshold-free suite only after the complete fixed trace."""

        self._assert_current_source_runtime()
        errors = self._state_errors(state)
        if errors:
            raise CalibratedExtendedSearchDevelopmentError(
                "cannot finalize invalid evaluator state: " + "; ".join(errors)
            )
        if int(jax.device_get(state.step_index)) != self.config.num_steps:
            raise CalibratedExtendedSearchDevelopmentError(
                "cannot finalize before all fixed real transitions are consumed"
            )
        records_tuple = tuple(
            self._arm_record(state, index) for index in range(len(CANONICAL_ARM_ORDER))
        )
        records = cast(
            tuple[
                CalibratedSearchArmRecord,
                CalibratedSearchArmRecord,
                CalibratedSearchArmRecord,
                CalibratedSearchArmRecord,
            ],
            records_tuple,
        )
        contrasts = (
            self._contrast(MODEL_FREE_VS_OPTION_MODEL, records[0], records[2]),
            self._contrast(PRIMITIVE_VS_COMBINED, records[1], records[3]),
        )
        audit = self._matched_audit(records)
        if audit.errors:
            raise CalibratedExtendedSearchDevelopmentError(
                "matched audit failed: " + "; ".join(audit.errors)
            )
        provisional = CalibratedSearchDevelopmentSuite(
            schema=CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_SUITE_SCHEMA,
            status=CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS,
            assessment_status=ASSESSMENT_STATUS,
            development_only=True,
            seed_role=DEVELOPMENT_SEED_ROLE,
            consumed_development_seed=self.config.seed,
            held_out_seeds_used=False,
            thresholds_frozen=False,
            artifact_writes_authorized=False,
            evidence_authorized=False,
            scientific_promotion_allowed=False,
            policy_authority=False,
            model_snapshot_boundary=MODEL_SNAPSHOT_BOUNDARY,
            source_calibration_boundary=SOURCE_CALIBRATION_BOUNDARY,
            config=self.config,
            config_sha256=self.config_sha256,
            protocol_sha256=self.protocol_sha256,
            source_runtime_manifest=self.source_runtime_manifest,
            model_snapshot=self.model_snapshot,
            evaluator_trace=self.evaluator_trace,
            canonical_arm_order=CANONICAL_ARM_ORDER,
            arm_records=records,
            contrasts=contrasts,
            matched_audit=audit,
            evaluator_accounting=self._evaluator_accounting(state, records),
            suite_binding_sha256="",
            replay_authenticator_sha256="",
            authenticated_replay_verified=False,
            thresholds=None,
            aggregate_verdict=None,
            artifact_output_path=None,
        )
        bound = dataclasses.replace(
            provisional,
            suite_binding_sha256=self._suite_binding(provisional),
        )
        bound = dataclasses.replace(
            bound,
            replay_authenticator_sha256=self._replay_authenticator(bound),
        )
        structural = validate_calibrated_search_development_suite(bound)
        if structural:
            raise CalibratedExtendedSearchDevelopmentError(
                "completed suite failed structural validation: " + "; ".join(structural)
            )
        return bound

    def run(self) -> CalibratedSearchDevelopmentSuite:
        return self.finalize(self.advance(self.init(), steps=self.config.num_steps))

    def checkpoint_payload(
        self, state: CalibratedSearchDevelopmentState
    ) -> dict[str, object]:
        """Return one exact source/protocol/trace-bound resumable payload."""

        self._assert_current_source_runtime()
        errors = self._state_errors(state)
        if errors:
            raise CalibratedExtendedSearchDevelopmentError(
                "cannot checkpoint invalid evaluator state: " + "; ".join(errors)
            )
        payload: dict[str, object] = {
            "schema": CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA,
            "config": self.config.to_config(),
            "config_sha256": self.config_sha256,
            "protocol_sha256": self.protocol_sha256,
            "source_runtime_manifest": self.source_runtime_manifest,
            "model_snapshot_sha256": self.model_snapshot.snapshot_sha256,
            "evaluator_trace_sha256": _trace_sha256(self.evaluator_trace),
            "state": state,
            "state_sha256": _exact_sha256(state),
        }
        payload["checkpoint_sha256"] = _exact_sha256(payload)
        return payload

    def restore_checkpoint(self, payload: object) -> CalibratedSearchDevelopmentState:
        """Restore only an exact current-source payload for this evaluator."""

        self._assert_current_source_runtime()
        if type(payload) is not dict:
            raise ValueError("development checkpoint must be an exact dict")
        raw = cast(dict[object, object], payload)
        expected = {
            "schema",
            "config",
            "config_sha256",
            "protocol_sha256",
            "source_runtime_manifest",
            "model_snapshot_sha256",
            "evaluator_trace_sha256",
            "state",
            "state_sha256",
            "checkpoint_sha256",
        }
        if set(raw) != expected:
            raise ValueError("development checkpoint fields differ from v1")
        checkpoint_digest = raw["checkpoint_sha256"]
        if not _is_sha256(checkpoint_digest):
            raise ValueError("development checkpoint digest is malformed")
        unbound = dict(raw)
        unbound.pop("checkpoint_sha256")
        if checkpoint_digest != _exact_sha256(unbound):
            raise ValueError("development checkpoint digest differs")
        exact_bindings = (
            raw["schema"] == CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA
            and raw["config"] == self.config.to_config()
            and raw["config_sha256"] == self.config_sha256
            and raw["protocol_sha256"] == self.protocol_sha256
            and raw["source_runtime_manifest"] == self.source_runtime_manifest
            and raw["model_snapshot_sha256"] == self.model_snapshot.snapshot_sha256
            and raw["evaluator_trace_sha256"] == _trace_sha256(self.evaluator_trace)
        )
        if not exact_bindings:
            raise ValueError("development checkpoint binding differs")
        state = raw["state"]
        if type(state) is not CalibratedSearchDevelopmentState:
            raise ValueError("development checkpoint state type differs")
        restored = state
        if raw["state_sha256"] != _exact_sha256(restored):
            raise ValueError("development checkpoint state digest differs")
        errors = self._state_errors(restored)
        if errors:
            raise ValueError("development checkpoint state is invalid: " + "; ".join(errors))
        prefix_errors = self._canonical_prefix_errors(restored)
        if prefix_errors:
            raise ValueError(
                "development checkpoint canonical prefix differs: "
                + "; ".join(prefix_errors)
            )
        return restored


def _record_structural_errors(
    evaluator: CalibratedExtendedSearchDevelopmentEvaluator,
    record: CalibratedSearchArmRecord,
    index: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    if record.arm_index != index or record.mode != CANONICAL_ARM_ORDER[index]:
        errors.append(f"arm record {index} identity/order differs")
    if record.assessment_status != ASSESSMENT_STATUS:
        errors.append(f"arm record {index} is assessed")
    if (
        record.config_sha256 != evaluator.config_sha256
        or record.protocol_sha256 != evaluator.protocol_sha256
        or record.source_manifest_sha256
        != evaluator.source_runtime_manifest.manifest_sha256
        or record.model_snapshot_sha256 != evaluator.model_snapshot.snapshot_sha256
        or record.evaluator_trace_sha256 != _trace_sha256(evaluator.evaluator_trace)
    ):
        errors.append(f"arm record {index} source/protocol binding differs")
    if record.initial_snapshot_shared is not True:
        errors.append(f"arm record {index} denies the shared snapshot")
    if record.resource_budget != evaluator.controllers[index].resource_budget:
        errors.append(f"arm record {index} resource declaration differs")
    accounting = record.accounting
    expected_attempts = evaluator.config.num_steps * evaluator.config.backup_budget
    if (
        accounting.mode != record.mode
        or accounting.real_experience_records != evaluator.config.num_steps
        or accounting.candidate_capacity != _CANDIDATE_CAPACITY
        or accounting.shared_backup_budget != evaluator.config.backup_budget
        or accounting.expected_backup_attempts != expected_attempts
        or accounting.actual_backup_attempts != expected_attempts
        or accounting.maximum_learner_updates != expected_attempts
        or not 0 <= accounting.actual_learner_updates <= expected_attempts
        or accounting.evaluator_rng_draws_shared_across_arms
        != 2 * evaluator.config.num_steps
        or accounting.planner_rng_draws != 0
        or accounting.persistent_state_growth_bytes != 0
    ):
        errors.append(f"arm record {index} resource/update accounting differs")
    trace = record.trace
    t = evaluator.config.num_steps
    for field in dataclasses.fields(CalibratedSearchArmTrace):
        array = getattr(trace, field.name)
        if not isinstance(array, jax.Array) or array.shape != (t,):
            errors.append(f"arm record {index} trace {field.name} shape differs")
    if not errors:
        if not bool(jnp.all(trace.arm_transaction_valid)) or not bool(
            jnp.all(trace.observe_transaction_valid)
        ):
            errors.append(f"arm record {index} contains an invalid transaction")
        if not bool(jnp.all(trace.backup_attempts == evaluator.config.backup_budget)):
            errors.append(f"arm record {index} attempt budget differs")
        if bool(jnp.any(trace.learner_updates > evaluator.config.backup_budget)):
            errors.append(f"arm record {index} learner updates exceed B")
        if int(jnp.sum(trace.backup_attempts)) != accounting.actual_backup_attempts:
            errors.append(f"arm record {index} attempt accounting is inconsistent")
        if int(jnp.sum(trace.learner_updates)) != accounting.actual_learner_updates:
            errors.append(f"arm record {index} update accounting is inconsistent")
        if not bool(jnp.all(trace.natural_resolution)) or bool(
            jnp.any(trace.censored_resolution)
        ):
            errors.append(f"arm record {index} real resolution trace differs")
    final = record.final_controller_state
    controller = evaluator.controllers[index]
    snapshot = evaluator.model_snapshot
    try:
        valid = controller.validate_state(
            final,
            representation_generation=snapshot.representation_generation,
            source_digest=snapshot.source_digest,
            option_descriptors=snapshot.option_descriptors,
            option_generations=snapshot.option_generations,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"arm record {index} final state contract failed: {exc}")
    else:
        if not bool(jax.device_get(valid)) or bool(jax.device_get(final.pending)):
            errors.append(f"arm record {index} final controller state is invalid")
        if int(final.state_revision) != evaluator.config.num_steps:
            errors.append(f"arm record {index} final state revision differs")
    if record.final_controller_state_sha256 != _exact_sha256(final):
        errors.append(f"arm record {index} final state digest differs")
    summary = record.summary
    if summary.assessment_status != ASSESSMENT_STATUS:
        errors.append(f"arm record {index} summary is assessed")
    if summary.final_support_counts != tuple(int(v) for v in np.asarray(final.support_counts)):
        errors.append(f"arm record {index} support summary differs")
    if summary.final_value_change_counts != tuple(
        int(v) for v in np.asarray(final.value_change_counts)
    ):
        errors.append(f"arm record {index} value calibration summary differs")
    if summary.final_model_error_counts != tuple(
        int(v) for v in np.asarray(final.model_error_counts)
    ):
        errors.append(f"arm record {index} error calibration summary differs")
    if summary.final_anchor_revisit_trials != tuple(
        int(v) for v in np.asarray(final.anchor_revisit_trials)
    ):
        errors.append(f"arm record {index} reachability trial summary differs")
    if summary.final_anchor_revisit_successes != tuple(
        int(v) for v in np.asarray(final.anchor_revisit_successes)
    ):
        errors.append(f"arm record {index} reachability success summary differs")
    return tuple(errors)


def validate_calibrated_search_development_suite(suite: object) -> tuple[str, ...]:
    """Strict current-source structural validation; it does not authenticate replay."""

    if type(suite) is not CalibratedSearchDevelopmentSuite:
        return ("suite must be an exact CalibratedSearchDevelopmentSuite",)
    checked = suite
    errors: list[str] = []
    forbidden_authority = (
        checked.held_out_seeds_used,
        checked.thresholds_frozen,
        checked.artifact_writes_authorized,
        checked.evidence_authorized,
        checked.scientific_promotion_allowed,
        checked.policy_authority,
        checked.authenticated_replay_verified,
    )
    if (
        checked.schema != CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_SUITE_SCHEMA
        or checked.status != CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS
        or checked.assessment_status != ASSESSMENT_STATUS
        or checked.development_only is not True
        or checked.seed_role != DEVELOPMENT_SEED_ROLE
        or any(value is not False for value in forbidden_authority)
        or checked.model_snapshot_boundary != MODEL_SNAPSHOT_BOUNDARY
        or checked.source_calibration_boundary != SOURCE_CALIBRATION_BOUNDARY
        or checked.thresholds is not None
        or checked.aggregate_verdict is not None
        or checked.artifact_output_path is not None
    ):
        errors.append("suite authority/status boundary differs")
    if type(checked.config) is not CalibratedExtendedSearchDevelopmentConfig:
        errors.append("suite config has the wrong concrete type")
        return tuple(errors)
    try:
        evaluator = CalibratedExtendedSearchDevelopmentEvaluator(checked.config)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return (f"current evaluator reconstruction failed closed: {exc}",)
    if checked.consumed_development_seed != evaluator.config.seed:
        errors.append("suite consumed seed differs")
    if checked.config_sha256 != evaluator.config_sha256:
        errors.append("suite config hash differs")
    if checked.protocol_sha256 != evaluator.protocol_sha256:
        errors.append("suite protocol hash differs")
    if checked.source_runtime_manifest != evaluator.source_runtime_manifest:
        errors.append("suite source/runtime manifest differs from current bytes")
    if _exact_sha256(checked.model_snapshot) != _exact_sha256(evaluator.model_snapshot):
        errors.append("suite immutable model snapshot differs from reconstruction")
    if not _host_tree_equal(checked.evaluator_trace, evaluator.evaluator_trace):
        errors.append("suite evaluator-owned raw trace differs from reconstruction")
    if checked.canonical_arm_order != CANONICAL_ARM_ORDER:
        errors.append("suite canonical arm order differs")
    if len(checked.arm_records) != len(CANONICAL_ARM_ORDER):
        errors.append("suite must contain exactly four arm records")
    else:
        for index, record in enumerate(checked.arm_records):
            if type(record) is not CalibratedSearchArmRecord:
                errors.append(f"arm record {index} has the wrong concrete type")
                continue
            errors.extend(_record_structural_errors(evaluator, record, index))
    if len(checked.contrasts) != 2:
        errors.append("suite must contain exactly two declared contrasts")
    elif len(checked.arm_records) == 4:
        expected_contrasts = (
            evaluator._contrast(
                MODEL_FREE_VS_OPTION_MODEL,
                checked.arm_records[0],
                checked.arm_records[2],
            ),
            evaluator._contrast(
                PRIMITIVE_VS_COMBINED,
                checked.arm_records[1],
                checked.arm_records[3],
            ),
        )
        if checked.contrasts != expected_contrasts:
            errors.append("suite descriptive contrasts differ")
        if any(
            contrast.assessment_status != ASSESSMENT_STATUS
            or contrast.threshold is not None
            or contrast.verdict is not None
            for contrast in checked.contrasts
        ):
            errors.append("suite contrast carries an assessment gate")
    if len(checked.arm_records) == 4:
        expected_audit = evaluator._matched_audit(checked.arm_records)
        if checked.matched_audit != expected_audit or expected_audit.errors:
            errors.append("suite matched audit differs or fails")
        final_state = CalibratedSearchDevelopmentState(
            binding_digest=evaluator.binding_digest,
            step_index=jnp.asarray(evaluator.config.num_steps, dtype=jnp.int32),
            controller_states=cast(
                tuple[
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                    CalibratedExtendedSearchControlState,
                ],
                tuple(record.final_controller_state for record in checked.arm_records),
            ),
            arm_transaction_valid=jnp.stack(
                tuple(record.trace.arm_transaction_valid for record in checked.arm_records)
            ),
            observe_transaction_valid=jnp.stack(
                tuple(record.trace.observe_transaction_valid for record in checked.arm_records)
            ),
            backup_attempts=jnp.stack(
                tuple(record.trace.backup_attempts for record in checked.arm_records)
            ),
            learner_updates=jnp.stack(
                tuple(record.trace.learner_updates for record in checked.arm_records)
            ),
            target_available_count=jnp.stack(
                tuple(record.trace.target_available_count for record in checked.arm_records)
            ),
            candidate_eligible_count=jnp.stack(
                tuple(record.trace.candidate_eligible_count for record in checked.arm_records)
            ),
            value_calibration_ready_count=jnp.stack(
                tuple(
                    record.trace.value_calibration_ready_count
                    for record in checked.arm_records
                )
            ),
            error_calibration_ready_count=jnp.stack(
                tuple(
                    record.trace.error_calibration_ready_count
                    for record in checked.arm_records
                )
            ),
            reachability_ready_count=jnp.stack(
                tuple(
                    record.trace.reachability_ready_count
                    for record in checked.arm_records
                )
            ),
            support_ready_count=jnp.stack(
                tuple(record.trace.support_ready_count for record in checked.arm_records)
            ),
            selected_primitive_count=jnp.stack(
                tuple(
                    record.trace.selected_primitive_count
                    for record in checked.arm_records
                )
            ),
            selected_option_count=jnp.stack(
                tuple(record.trace.selected_option_count for record in checked.arm_records)
            ),
            natural_resolution=jnp.stack(
                tuple(record.trace.natural_resolution for record in checked.arm_records)
            ),
            censored_resolution=jnp.stack(
                tuple(record.trace.censored_resolution for record in checked.arm_records)
            ),
            resolved_candidate_index=jnp.stack(
                tuple(
                    record.trace.resolved_candidate_index for record in checked.arm_records
                )
            ),
            realized_differential_target=jnp.stack(
                tuple(
                    record.trace.realized_differential_target
                    for record in checked.arm_records
                )
            ),
            normalized_model_error=jnp.stack(
                tuple(record.trace.normalized_model_error for record in checked.arm_records)
            ),
            priority_sum=jnp.stack(
                tuple(record.trace.priority_sum for record in checked.arm_records)
            ),
            priority_max=jnp.stack(
                tuple(record.trace.priority_max for record in checked.arm_records)
            ),
            q_l1_after=jnp.stack(
                tuple(record.trace.q_l1_after for record in checked.arm_records)
            ),
            state_revision_after=jnp.stack(
                tuple(record.trace.state_revision_after for record in checked.arm_records)
            ),
            learner_revision_after=jnp.stack(
                tuple(record.trace.learner_revision_after for record in checked.arm_records)
            ),
        )
        expected_accounting = evaluator._evaluator_accounting(
            final_state, checked.arm_records
        )
        if checked.evaluator_accounting != expected_accounting:
            errors.append("suite evaluator resource/update accounting differs")
    if not _is_sha256(checked.suite_binding_sha256):
        errors.append("suite binding digest is malformed")
    elif checked.suite_binding_sha256 != evaluator._suite_binding(checked):
        errors.append("suite binding digest differs")
    if not _is_sha256(checked.replay_authenticator_sha256):
        errors.append("suite replay authenticator is malformed")
    elif checked.replay_authenticator_sha256 != evaluator._replay_authenticator(checked):
        errors.append("suite replay authenticator differs")
    return tuple(dict.fromkeys(errors))


def authenticate_calibrated_search_development_replay(
    suite: object,
) -> CalibratedSearchAuthenticatedReplayValidation:
    """Reconstruct and bit-compare the complete source-bound four-arm suite."""

    structural_errors = validate_calibrated_search_development_suite(suite)
    if type(suite) is not CalibratedSearchDevelopmentSuite:
        return CalibratedSearchAuthenticatedReplayValidation(
            schema=CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_REPLAY_SCHEMA,
            assessment_status=ASSESSMENT_STATUS,
            source_runtime_verified=False,
            structural_validation_passed=False,
            authenticated_replay_verified=False,
            replay_suite_binding_sha256=None,
            errors=structural_errors,
        )
    checked = suite
    errors = list(structural_errors)
    source_verified = not any("source/runtime" in error for error in errors)
    replay_binding: str | None = None
    if not errors:
        try:
            replayed = CalibratedExtendedSearchDevelopmentEvaluator(
                checked.config
            ).run()
            replay_binding = replayed.suite_binding_sha256
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"authenticated replay execution failed closed: {exc}")
        else:
            if _exact_sha256(replayed) != _exact_sha256(checked):
                errors.append("authenticated replay differs from the supplied raw suite")
            if replayed.replay_authenticator_sha256 != checked.replay_authenticator_sha256:
                errors.append("authenticated replay token differs")
    verified = not errors
    return CalibratedSearchAuthenticatedReplayValidation(
        schema=CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_REPLAY_SCHEMA,
        assessment_status=ASSESSMENT_STATUS,
        source_runtime_verified=source_verified,
        structural_validation_passed=not structural_errors,
        authenticated_replay_verified=verified,
        replay_suite_binding_sha256=replay_binding,
        errors=tuple(errors),
    )


__all__ = [
    "ARTIFACT_WRITES_AUTHORIZED",
    "ASSESSMENT_STATUS",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CHECKPOINT_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_CONFIG_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_MANIFEST_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_REPLAY_SCHEMA",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_STATUS",
    "CALIBRATED_EXTENDED_SEARCH_DEVELOPMENT_SUITE_SCHEMA",
    "CANONICAL_ARM_ORDER",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SEED_ROLE",
    "EVIDENCE_AUTHORIZED",
    "HELD_OUT_SEEDS_USED",
    "MODEL_FREE_VS_OPTION_MODEL",
    "MODEL_SNAPSHOT_BOUNDARY",
    "POLICY_AUTHORITY",
    "PRIMITIVE_VS_COMBINED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_CALIBRATION_BOUNDARY",
    "THRESHOLDS_FROZEN",
    "TRACE_SEMANTICS",
    "CalibratedExtendedSearchDevelopmentConfig",
    "CalibratedExtendedSearchDevelopmentError",
    "CalibratedExtendedSearchDevelopmentEvaluator",
    "CalibratedSearchArmAccounting",
    "CalibratedSearchArmRecord",
    "CalibratedSearchArmSummary",
    "CalibratedSearchArmTrace",
    "CalibratedSearchAuthenticatedReplayValidation",
    "CalibratedSearchDevelopmentContrast",
    "CalibratedSearchDevelopmentState",
    "CalibratedSearchDevelopmentSuite",
    "CalibratedSearchEvaluatorOwnedTrace",
    "CalibratedSearchEvaluatorAccounting",
    "CalibratedSearchMatchedAudit",
    "CalibratedSearchModelSnapshot",
    "CalibratedSearchSourceFileHash",
    "CalibratedSearchSourceRuntimeManifest",
    "authenticate_calibrated_search_development_replay",
    "build_calibrated_search_model_snapshot",
    "build_calibrated_search_source_runtime_manifest",
    "reconstruct_calibrated_search_evaluator_trace",
    "validate_calibrated_search_development_suite",
]
