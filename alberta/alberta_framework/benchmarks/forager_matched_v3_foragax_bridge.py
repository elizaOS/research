"""Fail-closed shared Foragax bridge for the matched-v3 adapter runners.

The bridge owns the canonical untagged environment split chain.  A reusable
runtime handle validates the installed Foragax/JAX surface once, while every
trajectory state is a linear, process-local capability: a state can advance
exactly once, cannot be forked or resumed, and poisons its trajectory after a
post-call failure.

This remains a host-side, per-step, unqualified engineering bridge.  It has no
compiled chunk kernel, protected seeds, full-horizon runner, result writer, or
execution-authority mechanism.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import importlib.machinery
import importlib.util
import json
import math
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Final, NoReturn, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_foragax_bridge.v2"
)
FORAGAX_DISTRIBUTION: Final = "continual-foragax"
FORAGAX_REQUIRED_VERSION: Final = "0.55.0"
FORAGAX_REGISTRY_MODULE: Final = "foragax.registry"
FORAGAX_REGISTRY_CALLABLE: Final = "make"
FORAGAX_INSTALL_TREE_HASH_SCHEME: Final = "relative-path+size+bytes-v1"
FORAGAX_INSTALL_TREE_SHA256: Final = (
    "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
)

JAX_REQUIRED_VERSION: Final = "0.11.0"
JAXLIB_REQUIRED_VERSION: Final = "0.11.0"
THREEFRY_IMPLEMENTATION: Final = "threefry2x32"
MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE: Final = "dedicated_environment_split_chain_v1"

MATCHED_V3_ENVIRONMENT_ID: Final = protocol.MATCHED_V3_ENVIRONMENT_ID
MATCHED_V3_OBSERVATION_TYPE: Final = protocol.MATCHED_V3_OBSERVATION_TYPE
MATCHED_V3_APERTURE_SIZE: Final = protocol.MATCHED_V3_APERTURE_SIZE
MATCHED_V3_OBSERVATION_SHAPE: Final = (9, 9, 3)
MATCHED_V3_NUM_ACTIONS: Final = 4
MATCHED_V3_HORIZON: Final = protocol.MATCHED_V3_HORIZON
MATCHED_V3_RAW_REWARD_VALUES: Final = protocol.MATCHED_V3_RAW_REWARD_VALUES
MATCHED_V3_REWARD_DELAY: Final = 0
MATCHED_V3_RANDOM_SHIFT_MAX_STEPS: Final = 0
UINT31_MAXIMUM: Final = (1 << 31) - 1

_MAX_DESCRIPTOR_BYTES: Final = 256 * 1024
_MAX_INSTALL_TREE_FILES: Final = 10_000
_MAX_INSTALL_TREE_BYTES: Final = 256 * 1024 * 1024
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
_SOURCE_FILE_PINS: Final = (
    (
        "foragax/registry.py",
        8_872,
        "eb7ec7e40e99b417422ad46e7965ffad0342006c20fe3623fd14ff438ba048d5",
    ),
    (
        "foragax/env.py",
        94_019,
        "c76d07c9c6dad04e49be00f76f788dfdbb0047cce22448b2faa574ee78a9ef3e",
    ),
)


class ForagerMatchedV3ForagaxBridgeError(ValueError):
    """A bridge input, runtime capability, or transition violated the contract."""


@dataclass(frozen=True, slots=True)
class MatchedV3ForagaxRuntimeIdentity:
    """Observed development-runtime facts; never a qualification receipt."""

    jax_version: str
    jaxlib_version: str
    default_prng_impl: str
    threefry_partitionable: bool
    jax_enable_x64: bool
    backend: str
    foragax_version: str
    foragax_install_tree_sha256: str
    foragax_package_root: str
    runtime_qualified: bool


class _RuntimeCapability:
    __slots__ = ("__weakref__",)


class _StateCapability:
    __slots__ = ("__weakref__",)


@dataclass(slots=True)
class _RuntimeBinding:
    environment: Any
    params: Any
    identity: MatchedV3ForagaxRuntimeIdentity


@dataclass(slots=True)
class _TrajectoryBinding:
    poisoned: bool = False
    in_flight: bool = False


@dataclass(slots=True)
class _StateBinding:
    trajectory: _TrajectoryBinding
    environment_seed: int
    reset_count: int
    step_count: int
    environment_key_words: tuple[int, int]
    environment: Any
    params: Any
    environment_state: Any
    environment_time: int
    observation: Array
    runtime_capability: _RuntimeCapability
    consumed: bool = False


_REGISTRY_LOCK: Final = threading.RLock()
_RUNTIME_REGISTRY: Final = weakref.WeakKeyDictionary[_RuntimeCapability, _RuntimeBinding]()
_STATE_REGISTRY: Final = weakref.WeakKeyDictionary[_StateCapability, _StateBinding]()


@dataclass(frozen=True, slots=True)
class MatchedV3ForagaxRuntime:
    """One once-validated reusable environment object and immutable parameters."""

    runtime_identity: MatchedV3ForagaxRuntimeIdentity
    _environment: Any = field(repr=False, compare=False)
    _params: Any = field(repr=False, compare=False)
    _capability: _RuntimeCapability = field(repr=False, compare=False)

    def initialize(self, environment_seed: object) -> MatchedV3ForagaxBridgeState:
        """Reset one independent trajectory without reconstructing the environment."""

        return _initialize_with_runtime(self, environment_seed)


@dataclass(frozen=True, slots=True)
class MatchedV3ForagaxBridgeState:
    """One immutable, single-use handle in a continuing trajectory."""

    environment_seed: int
    observation: Array
    reset_count: int
    step_count: int
    _runtime: MatchedV3ForagaxRuntime = field(repr=False, compare=False)
    _environment: Any = field(repr=False, compare=False)
    _params: Any = field(repr=False, compare=False)
    _environment_state: Any = field(repr=False, compare=False)
    _environment_key: Array = field(repr=False, compare=False)
    _capability: _StateCapability = field(repr=False, compare=False)

    @property
    def environment_key_use_count(self) -> int:
        """Return one reset-key use plus one use for each completed step."""

        return self.reset_count + self.step_count


@dataclass(frozen=True, slots=True)
class MatchedV3ForagaxTransition:
    """One validated adapter-visible transition with evaluator info removed."""

    state: MatchedV3ForagaxBridgeState
    action: int
    reward: int
    done: bool = False
    truncated: bool = False
    info_validated: bool = True

    @property
    def observation(self) -> Array:
        """Return the validated post-transition observation."""

        return self.state.observation


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        raw = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax bridge descriptor is not finite canonical JSON"
        ) from exc
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax bridge descriptor exceeds its byte limit"
        )
    return raw


def _assert_plain_unaliased_json(value: object) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3ForagaxBridgeError(
                    "descriptor contains aliased or cyclic containers"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ForagerMatchedV3ForagaxBridgeError(
                    "descriptor contains a non-string key"
                )
            pending.extend(mapping.values())
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3ForagaxBridgeError(
                    "descriptor contains aliased or cyclic containers"
                )
            seen.add(identity)
            pending.extend(cast(list[object], item))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ForagerMatchedV3ForagaxBridgeError(
                    "descriptor contains a non-finite number"
                )
        elif item is not None and type(item) not in {str, int, bool}:
            raise ForagerMatchedV3ForagaxBridgeError(
                f"descriptor contains non-plain JSON type {type(item).__name__}"
            )


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
        "status": "implemented_unqualified",
        "classification": "shared_local_adapter_environment_bridge_non_authorizing",
        "adapter_consumers": ["adapted_full_rainbow", "adapted_ppo_gru"],
        "source": {
            "distribution": FORAGAX_DISTRIBUTION,
            "required_version": FORAGAX_REQUIRED_VERSION,
            "canonical_url": "https://github.com/steventango/continual-foragax",
            "release_wheel_sha256": (
                "79b20f234d651feed2736873192fa6e3b224bce9bf6e9674f1ed52a227b073d2"
            ),
            "install_tree_hash_scheme": FORAGAX_INSTALL_TREE_HASH_SCHEME,
            "expected_install_tree_sha256": FORAGAX_INSTALL_TREE_SHA256,
            "installed_api_surface_inspected": True,
            "inspected_file_pins": [
                {"path": path, "size_bytes": size, "sha256": sha256}
                for path, size, sha256 in _SOURCE_FILE_PINS
            ],
            "source_review_complete": False,
            "source_closure_bound": False,
        },
        "task": {
            "environment_id": MATCHED_V3_ENVIRONMENT_ID,
            "observation_type": MATCHED_V3_OBSERVATION_TYPE,
            "aperture_size": MATCHED_V3_APERTURE_SIZE,
            "observation_shape": list(MATCHED_V3_OBSERVATION_SHAPE),
            "num_actions": MATCHED_V3_NUM_ACTIONS,
            "reward_delay": MATCHED_V3_REWARD_DELAY,
            "random_shift_max_steps": MATCHED_V3_RANDOM_SHIFT_MAX_STEPS,
            "trajectory_count_per_state_chain": 1,
            "continuing": True,
            "horizon": MATCHED_V3_HORIZON,
            "raw_reward_values": list(MATCHED_V3_RAW_REWARD_VALUES),
        },
        "runtime": {
            "jax_required_version": JAX_REQUIRED_VERSION,
            "jaxlib_required_version": JAXLIB_REQUIRED_VERSION,
            "default_prng_impl": THREEFRY_IMPLEMENTATION,
            "jax_threefry_partitionable": True,
            "jax_enable_x64": False,
            "foragax_install_tree_checked_at_open": True,
            "environment_capabilities_checked_once_at_open": True,
            "reusable_runtime_handle": True,
            "convenience_api_opens_one_runtime": True,
            "backend_observed_at_open": True,
            "backend_qualified": False,
            "per_step_host_api_jitted": False,
            "real_foragax_api_inspected": True,
            "runtime_parity_executed": False,
            "runtime_qualified": False,
        },
        "rng": {
            "identity": MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE,
            "implementation": THREEFRY_IMPLEMENTATION,
            "environment_seed_type": "exact_uint31",
            "root": "jax.random.key(environment_seed,impl=threefry2x32)",
            "public_contract_root": "jax.random.key(seed)",
            "reset": "environment_key,reset_key=jax.random.split(environment_key)",
            "transition": (
                "environment_key,step_key=jax.random.split(environment_key) exactly once"
            ),
            "reset_during_trajectory": False,
            "agent_key_accepted": False,
            "agent_draws_affect_environment_schedule": False,
        },
        "validation": {
            "observation": "float32_9x9x3_binary_zero_or_one_hot_per_cell",
            "empty_cell_encoding": "all_three_channels_zero",
            "action": "python_int_or_int32_scalar_in_closed_interval_0_3",
            "reward": "float32_scalar_exact_member_of_-1_0_1_30",
            "done": "boolean_scalar_must_be_false",
            "truncation": "not_returned_by_real_api_and_never_synthesized_true",
            "info": "exact_real_v0_55_key_dtype_shape_and_continuing_discount_contract",
            "environment_state_time": "int32_scalar_reset_zero_then_exact_plus_one",
            "adapter_info_access": False,
        },
        "state_lifecycle": {
            "opaque_capabilities": "weak_process_local_per_state",
            "registry_locking": True,
            "registry_binds_seed_counts_key_words_and_object_identities": True,
            "state_advance_semantics": "linear_single_use",
            "stale_fork_and_concurrent_double_step_rejected": True,
            "post_call_failure_poisons_trajectory": True,
            "checkpoint_or_external_state_accepted": False,
            "cryptographically_authenticated_resume": False,
        },
        "accounting": {
            "reset_calls_per_trajectory": 1,
            "step_calls_per_transition": 1,
            "reset_key_uses": 1,
            "step_key_uses_per_transition": 1,
            "automatic_resets": 0,
            "maximum_steps": MATCHED_V3_HORIZON,
        },
        "claims": {
            "execution_ready": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "authority_granted": False,
        },
        "limitations": [
            "Installed API inspection is not runtime parity or qualification.",
            "The inspected files are review anchors, not complete dependency closure.",
            "Observed backend identity is recorded but no backend is qualified here.",
            "The per-step host API is non-JIT and a compiled chunk kernel remains a blocker.",
            "The bridge supplies no full-horizon runner, checkpoint, or result writer.",
            "No protected seed is embedded, requested, generated, or authorized.",
            "Synthetic bridge tests are engineering checks and not scientific evidence.",
            "Private state internals are trusted in-process and are not resumable artifacts.",
        ],
    }


_DESCRIPTOR: Final = _descriptor()
_assert_plain_unaliased_json(_DESCRIPTOR)
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR)
FORAGAX_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 Foragax bridge descriptor identity drifted")


def matched_v3_foragax_bridge_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the frozen non-authorizing descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_matched_v3_foragax_bridge_descriptor_bytes() -> bytes:
    """Return the exact canonical bridge descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3ForagaxBridgeError(
                f"duplicate descriptor key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> NoReturn:
    raise ForagerMatchedV3ForagaxBridgeError(
        f"non-finite descriptor number {token!r}"
    )


def parse_matched_v3_foragax_bridge_descriptor(
    value: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only the exact canonical descriptor, detaching mapping inputs."""

    if isinstance(value, Mapping):
        _assert_plain_unaliased_json(value)
        raw = _canonical_json(value)
    elif type(value) is bytes:
        raw = value
    else:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor must be exact bytes or a plain mapping"
        )
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor exceeds its byte limit"
        )
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ForagerMatchedV3ForagaxBridgeError:
        raise
    except (RecursionError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor is not strict ASCII JSON"
        ) from exc
    if type(decoded) is not dict:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor must encode a plain object"
        )
    if _canonical_json(cast(dict[str, Any], decoded)) != raw:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor is not in exact canonical form"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), FORAGAX_BRIDGE_DESCRIPTOR_SHA256
    ) or raw != _DESCRIPTOR_BYTES:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge descriptor does not match the frozen identity"
        )
    return matched_v3_foragax_bridge_descriptor()


def _foragax_install_tree_identity() -> tuple[str, str]:
    spec = importlib.util.find_spec("foragax")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if (
        spec is None
        or len(locations) != 1
        or spec.origin is None
        or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
    ):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax runtime package origin or loader is ambiguous"
        )
    root = Path(locations[0]).resolve()
    if Path(spec.origin).resolve() != root / "__init__.py" or not root.is_dir():
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax runtime package origin does not match its package root"
        )
    files: list[tuple[str, Path]] = []
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax install tree contains a symbolic-link file"
            )
        size = path.stat().st_size
        total_bytes += size
        files.append((f"foragax/{path.relative_to(root).as_posix()}", path))
        if len(files) > _MAX_INSTALL_TREE_FILES or total_bytes > _MAX_INSTALL_TREE_BYTES:
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax install tree exceeds its verification bounds"
            )
    if not files:
        raise ForagerMatchedV3ForagaxBridgeError("Foragax install tree is empty")
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        contents = path.read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return str(root), digest.hexdigest()


def _observe_runtime_identity() -> dict[str, object]:
    try:
        jaxlib_version = importlib_metadata.version("jaxlib")
        foragax_version = importlib_metadata.version(FORAGAX_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "required JAX/Foragax runtime distributions are missing"
        ) from exc
    package_root, install_tree = _foragax_install_tree_identity()
    try:
        backend = str(jax.default_backend())
    except RuntimeError as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "JAX backend identity could not be observed"
        ) from exc
    return {
        "jax_version": str(jax.__version__),
        "jaxlib_version": jaxlib_version,
        "default_prng_impl": str(jax.config.jax_default_prng_impl),
        "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "backend": backend,
        "foragax_version": foragax_version,
        "foragax_install_tree_sha256": install_tree,
        "foragax_package_root": package_root,
        "runtime_qualified": False,
    }


def _validated_runtime_identity() -> MatchedV3ForagaxRuntimeIdentity:
    observed = _observe_runtime_identity()
    expected: dict[str, object] = {
        "jax_version": JAX_REQUIRED_VERSION,
        "jaxlib_version": JAXLIB_REQUIRED_VERSION,
        "default_prng_impl": THREEFRY_IMPLEMENTATION,
        "threefry_partitionable": True,
        "jax_enable_x64": False,
        "foragax_version": FORAGAX_REQUIRED_VERSION,
        "foragax_install_tree_sha256": FORAGAX_INSTALL_TREE_SHA256,
        "runtime_qualified": False,
    }
    if set(observed) != {
        *expected,
        "backend",
        "foragax_package_root",
    }:
        raise ForagerMatchedV3ForagaxBridgeError(
            "observed runtime identity has an unexpected schema"
        )
    drift = [key for key, value in expected.items() if observed.get(key) != value]
    backend = observed.get("backend")
    package_root = observed.get("foragax_package_root")
    if drift or type(backend) is not str or not backend or type(package_root) is not str:
        detail = ", ".join(drift) if drift else "backend/package root"
        raise ForagerMatchedV3ForagaxBridgeError(
            f"observed runtime differs from the bridge baseline: {detail}"
        )
    return MatchedV3ForagaxRuntimeIdentity(
        jax_version=cast(str, observed["jax_version"]),
        jaxlib_version=cast(str, observed["jaxlib_version"]),
        default_prng_impl=cast(str, observed["default_prng_impl"]),
        threefry_partitionable=cast(bool, observed["threefry_partitionable"]),
        jax_enable_x64=cast(bool, observed["jax_enable_x64"]),
        backend=backend,
        foragax_version=cast(str, observed["foragax_version"]),
        foragax_install_tree_sha256=cast(
            str, observed["foragax_install_tree_sha256"]
        ),
        foragax_package_root=package_root,
        runtime_qualified=False,
    )


def _validate_live_jax_semantics(identity: MatchedV3ForagaxRuntimeIdentity) -> None:
    try:
        jaxlib_version = importlib_metadata.version("jaxlib")
    except importlib_metadata.PackageNotFoundError as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "JAX runtime disappeared after bridge open"
        ) from exc
    current = (
        str(jax.__version__),
        jaxlib_version,
        str(jax.config.jax_default_prng_impl),
        bool(jax.config.jax_threefry_partitionable),
        bool(jax.config.jax_enable_x64),
    )
    expected = (
        identity.jax_version,
        identity.jaxlib_version,
        identity.default_prng_impl,
        identity.threefry_partitionable,
        identity.jax_enable_x64,
    )
    if current != expected:
        raise ForagerMatchedV3ForagaxBridgeError(
            "live JAX runtime semantics drifted after bridge open"
        )


def _load_registry_make(
    identity: MatchedV3ForagaxRuntimeIdentity,
) -> Callable[..., Any]:
    """Load the exact Foragax factory lazily after install-tree verification."""

    try:
        installed = importlib_metadata.version(FORAGAX_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"{FORAGAX_DISTRIBUTION}=={FORAGAX_REQUIRED_VERSION} is not installed"
        ) from exc
    if installed != FORAGAX_REQUIRED_VERSION or installed != identity.foragax_version:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"bridge requires {FORAGAX_DISTRIBUTION}=={FORAGAX_REQUIRED_VERSION}; "
            f"found {installed!r}"
        )
    try:
        registry = importlib.import_module(FORAGAX_REGISTRY_MODULE)
    except (ImportError, RuntimeError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "foragax.registry could not be imported lazily"
        ) from exc
    origin = getattr(registry, "__file__", None)
    if type(origin) is not str or Path(origin).resolve() != (
        Path(identity.foragax_package_root) / "registry.py"
    ).resolve():
        raise ForagerMatchedV3ForagaxBridgeError(
            "foragax.registry resolved outside the verified install tree"
        )
    make = getattr(registry, FORAGAX_REGISTRY_CALLABLE, None)
    if not callable(make):
        raise ForagerMatchedV3ForagaxBridgeError(
            "foragax.registry.make must be callable"
        )
    return cast(Callable[..., Any], make)


def _require_exact_capability(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax capability {label} does not match the exact bridge contract"
        )


def _validate_environment_capabilities(environment: Any, params: Any) -> None:
    _require_exact_capability(getattr(environment, "name", None), MATCHED_V3_ENVIRONMENT_ID, "name")
    _require_exact_capability(
        getattr(environment, "observation_type", None),
        MATCHED_V3_OBSERVATION_TYPE,
        "observation_type",
    )
    _require_exact_capability(
        getattr(environment, "aperture_size", None),
        (MATCHED_V3_APERTURE_SIZE, MATCHED_V3_APERTURE_SIZE),
        "aperture_size",
    )
    _require_exact_capability(
        getattr(environment, "num_actions", None), MATCHED_V3_NUM_ACTIONS, "num_actions"
    )
    for name in ("reset", "step", "action_space", "observation_space"):
        if not callable(getattr(environment, name, None)):
            raise ForagerMatchedV3ForagaxBridgeError(
                f"Foragax capability {name} must be callable"
            )
    if not hasattr(params, "max_steps_in_episode"):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax continuing params lack max_steps_in_episode"
        )
    if params.max_steps_in_episode is not None:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax continuing params must disable episode truncation"
        )
    try:
        action_space = environment.action_space(params)
        observation_space = environment.observation_space(params)
    except Exception as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax capability spaces could not be inspected"
        ) from exc
    _require_exact_capability(
        getattr(action_space, "n", None), MATCHED_V3_NUM_ACTIONS, "action_space.n"
    )
    _require_exact_capability(
        getattr(observation_space, "shape", None),
        MATCHED_V3_OBSERVATION_SHAPE,
        "observation_space.shape",
    )


def open_matched_v3_foragax_runtime() -> MatchedV3ForagaxRuntime:
    """Validate and construct one reusable, explicitly unqualified runtime."""

    identity = _validated_runtime_identity()
    make = _load_registry_make(identity)
    try:
        environment = make(
            MATCHED_V3_ENVIRONMENT_ID,
            aperture_size=MATCHED_V3_APERTURE_SIZE,
            observation_type=MATCHED_V3_OBSERVATION_TYPE,
            random_shift_max_steps=MATCHED_V3_RANDOM_SHIFT_MAX_STEPS,
            reward_delay=MATCHED_V3_REWARD_DELAY,
        )
        params = environment.default_params
    except Exception as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "exact foragax.registry.make construction failed"
        ) from exc
    _validate_environment_capabilities(environment, params)
    capability = _RuntimeCapability()
    runtime = MatchedV3ForagaxRuntime(
        runtime_identity=identity,
        _environment=environment,
        _params=params,
        _capability=capability,
    )
    with _REGISTRY_LOCK:
        _RUNTIME_REGISTRY[capability] = _RuntimeBinding(environment, params, identity)
    return runtime


def _validate_runtime_handle(runtime: object) -> _RuntimeBinding:
    if type(runtime) is not MatchedV3ForagaxRuntime:
        raise ForagerMatchedV3ForagaxBridgeError(
            "runtime must be an exact MatchedV3ForagaxRuntime"
        )
    _validate_live_jax_semantics(runtime.runtime_identity)
    with _REGISTRY_LOCK:
        binding = _RUNTIME_REGISTRY.get(runtime._capability)
        if (
            binding is None
            or runtime._environment is not binding.environment
            or runtime._params is not binding.params
            or runtime.runtime_identity != binding.identity
        ):
            raise ForagerMatchedV3ForagaxBridgeError(
                "runtime handle disagrees with its locked registry binding"
            )
        return binding


def _require_uint31(value: object) -> int:
    if type(value) is not int or not 0 <= value <= UINT31_MAXIMUM:
        raise ForagerMatchedV3ForagaxBridgeError(
            "environment_seed must be an exact uint31"
        )
    return value


def _key_words(value: object) -> tuple[int, int]:
    try:
        key = cast(Any, value)
        implementation = str(jr.key_impl(key))
        words = np.asarray(jr.key_data(key))
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "environment key must be a typed JAX key"
        ) from exc
    if (
        implementation != THREEFRY_IMPLEMENTATION
        or words.shape != (2,)
        or words.dtype != np.dtype(np.uint32)
    ):
        raise ForagerMatchedV3ForagaxBridgeError(
            "environment key must be exact Threefry2x32"
        )
    return int(words[0]), int(words[1])


def _validate_observation(value: object) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation must be an array"
        )
    array = np.asarray(value)
    if array.shape != MATCHED_V3_OBSERVATION_SHAPE:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation must have exact shape (9, 9, 3)"
        )
    if array.dtype != np.dtype(np.float32):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation must originate as exact float32"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation must be finite"
        )
    if not bool(np.all((array == 0.0) | (array == 1.0))):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation channels must be binary"
        )
    if bool(np.any(np.sum(array, axis=-1) > 1.0)):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax observation cells must be zero-hot or one-hot"
        )
    return jnp.asarray(value)


def _validate_action(value: object) -> int:
    if type(value) is int:
        action = value
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ForagerMatchedV3ForagaxBridgeError(
                "action must be a Python int or exact int32 scalar"
            ) from exc
        if array.shape != () or array.dtype != np.dtype(np.int32):
            raise ForagerMatchedV3ForagaxBridgeError(
                "action must be a Python int or exact int32 scalar"
            )
        action = int(array)
    if not 0 <= action < MATCHED_V3_NUM_ACTIONS:
        raise ForagerMatchedV3ForagaxBridgeError(
            "action must be one of the exact four values 0, 1, 2, or 3"
        )
    return action


def _validate_reward(value: object) -> int:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reward must be an exact float32 scalar"
        ) from exc
    if array.shape != () or array.dtype != np.dtype(np.float32):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reward must be an exact float32 scalar"
        )
    reward_float = float(array)
    if not math.isfinite(reward_float):
        raise ForagerMatchedV3ForagaxBridgeError("Foragax reward must be finite")
    reward = int(reward_float)
    if reward_float != reward or reward not in MATCHED_V3_RAW_REWARD_VALUES:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reward must be an exact member of {-1, 0, 1, 30}"
        )
    return reward


def _validate_done(value: object) -> None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax done must be an exact false boolean scalar"
        ) from exc
    if array.shape != () or array.dtype != np.dtype(np.bool_) or bool(array):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax done must be an exact false boolean scalar"
        )


def _require_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax info {label} must be an array"
        ) from exc
    if array.shape != shape or array.dtype != dtype:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax info {label} has the wrong shape or dtype"
        )
    return array


def _require_finite_float32_scalar(value: object, label: str) -> float:
    array = _require_array(value, shape=(), dtype=np.dtype(np.float32), label=label)
    result = float(array)
    if not math.isfinite(result):
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax info {label} must be finite"
        )
    return result


def _require_integer_scalar(value: object, *, dtype: np.dtype[Any], label: str) -> int:
    return int(_require_array(value, shape=(), dtype=dtype, label=label))


def _validate_info(value: object) -> None:
    if type(value) is not dict:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info must be one plain dictionary"
        )
    info = cast(dict[str, object], value)
    if frozenset(info) != _EXPECTED_INFO_KEYS:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info keys do not match the exact 0.55 contract"
        )
    if _require_finite_float32_scalar(info["discount"], "discount") != 1.0:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info discount must remain exactly one"
        )
    temperatures = _require_array(
        info["temperatures"], shape=(4,), dtype=np.dtype(np.float32), label="temperatures"
    )
    if not bool(np.all(np.isfinite(temperatures))) or bool(np.any(temperatures != 0.0)):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info temperatures must remain finite zeros for this task"
        )
    biome_id = _require_integer_scalar(
        info["biome_id"], dtype=np.dtype(np.int16), label="biome_id"
    )
    if biome_id not in {-1, 0, 1}:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info biome_id is outside the exact task range"
        )
    collected = _require_integer_scalar(
        info["object_collected_id"],
        dtype=np.dtype(np.int32),
        label="object_collected_id",
    )
    if collected not in {-1, 1, 2, 3}:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info object_collected_id is outside the exact task support"
        )
    current = _require_finite_float32_scalar(
        info["current_biome_mean"], "current_biome_mean"
    )
    maximum = _require_finite_float32_scalar(info["max_biome_mean"], "max_biome_mean")
    regret = _require_finite_float32_scalar(info["biome_regret"], "biome_regret")
    if current > maximum or regret < 0.0 or np.float32(maximum - current) != np.float32(
        regret
    ):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info biome regret arithmetic is inconsistent"
        )
    rank = _require_integer_scalar(
        info["biome_rank"], dtype=np.dtype(np.int32), label="biome_rank"
    )
    if not 1 <= rank <= 3:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info biome_rank is outside the exact task range"
        )
    reward_grid = _require_array(
        info["rewards"], shape=(9, 9), dtype=np.dtype(np.float16), label="rewards"
    )
    if not bool(np.all(np.isfinite(reward_grid))) or not bool(
        np.all(np.isin(reward_grid, MATCHED_V3_RAW_REWARD_VALUES))
    ):
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax info rewards grid is outside the exact raw reward support"
        )


def _environment_state_time(value: object, *, label: str) -> int:
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if "time" not in mapping:
            raise ForagerMatchedV3ForagaxBridgeError(
                f"Foragax {label} lacks an environment time field"
            )
        candidate = mapping["time"]
    elif hasattr(value, "time"):
        candidate = getattr(value, "time")
    else:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax {label} lacks an environment time field"
        )
    try:
        array = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax {label} time must be an exact int32 scalar"
        ) from exc
    if array.shape != () or array.dtype != np.dtype(np.int32):
        raise ForagerMatchedV3ForagaxBridgeError(
            f"Foragax {label} time must be an exact int32 scalar"
        )
    return int(array)


def _register_initial_state(
    runtime: MatchedV3ForagaxRuntime,
    *,
    environment_seed: int,
    observation: Array,
    environment_state: Any,
    environment_key: Array,
) -> MatchedV3ForagaxBridgeState:
    capability = _StateCapability()
    state = MatchedV3ForagaxBridgeState(
        environment_seed=environment_seed,
        observation=observation,
        reset_count=1,
        step_count=0,
        _runtime=runtime,
        _environment=runtime._environment,
        _params=runtime._params,
        _environment_state=environment_state,
        _environment_key=environment_key,
        _capability=capability,
    )
    binding = _StateBinding(
        trajectory=_TrajectoryBinding(),
        environment_seed=environment_seed,
        reset_count=1,
        step_count=0,
        environment_key_words=_key_words(environment_key),
        environment=runtime._environment,
        params=runtime._params,
        environment_state=environment_state,
        environment_time=0,
        observation=observation,
        runtime_capability=runtime._capability,
    )
    with _REGISTRY_LOCK:
        if runtime._capability not in _RUNTIME_REGISTRY:
            raise ForagerMatchedV3ForagaxBridgeError(
                "runtime disappeared before trajectory registration"
            )
        _STATE_REGISTRY[capability] = binding
    return state


def _initialize_with_runtime(
    runtime: MatchedV3ForagaxRuntime,
    environment_seed: object,
) -> MatchedV3ForagaxBridgeState:
    binding = _validate_runtime_handle(runtime)
    seed = _require_uint31(environment_seed)
    environment_key = jr.key(seed, impl=THREEFRY_IMPLEMENTATION)
    environment_key, reset_key = jr.split(environment_key)
    try:
        reset_result = binding.environment.reset(reset_key, binding.params)
    except Exception as exc:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reset failed before a trajectory was opened"
        ) from exc
    if type(reset_result) is not tuple or len(reset_result) != 2:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reset must return exactly (observation, state)"
        )
    raw_observation, environment_state = reset_result
    observation = _validate_observation(raw_observation)
    if _environment_state_time(environment_state, label="reset state") != 0:
        raise ForagerMatchedV3ForagaxBridgeError(
            "Foragax reset state time must be exactly zero"
        )
    return _register_initial_state(
        runtime,
        environment_seed=seed,
        observation=observation,
        environment_state=environment_state,
        environment_key=environment_key,
    )


def initialize_matched_v3_foragax_bridge(
    environment_seed: object,
    *,
    runtime: MatchedV3ForagaxRuntime | None = None,
) -> MatchedV3ForagaxBridgeState:
    """Reset one trajectory, opening a runtime only for the convenience form."""

    selected = open_matched_v3_foragax_runtime() if runtime is None else runtime
    return _initialize_with_runtime(selected, environment_seed)


def _validate_state_binding_locked(
    state: MatchedV3ForagaxBridgeState,
) -> _StateBinding:
    binding = _STATE_REGISTRY.get(state._capability)
    if binding is None:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state is absent from the process-local registry"
        )
    if binding.trajectory.poisoned:
        raise ForagerMatchedV3ForagaxBridgeError("bridge trajectory is poisoned")
    if binding.consumed or binding.trajectory.in_flight:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state is stale or already consumed"
        )
    exact_objects = (
        state._runtime._capability is binding.runtime_capability
        and state._environment is binding.environment
        and state._params is binding.params
        and state._environment_state is binding.environment_state
        and state.observation is binding.observation
    )
    exact_scalars = (
        type(state.environment_seed) is int
        and state.environment_seed == binding.environment_seed
        and type(state.reset_count) is int
        and state.reset_count == binding.reset_count == 1
        and type(state.step_count) is int
        and state.step_count == binding.step_count
        and 0 <= state.step_count <= MATCHED_V3_HORIZON
    )
    if not exact_objects or not exact_scalars:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state disagrees with its locked registry binding"
        )
    if _key_words(state._environment_key) != binding.environment_key_words:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state key disagrees with its locked registry binding"
        )
    if _environment_state_time(state._environment_state, label="state") != (
        binding.environment_time
    ):
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state time disagrees with its locked registry binding"
        )
    _validate_observation(state.observation)
    if state.step_count >= MATCHED_V3_HORIZON:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge cannot step beyond the exact matched-v3 horizon"
        )
    return binding


def _consume_state(state: object) -> tuple[MatchedV3ForagaxBridgeState, _StateBinding]:
    if type(state) is not MatchedV3ForagaxBridgeState:
        raise ForagerMatchedV3ForagaxBridgeError(
            "bridge state must be an exact MatchedV3ForagaxBridgeState"
        )
    _validate_runtime_handle(state._runtime)
    with _REGISTRY_LOCK:
        binding = _validate_state_binding_locked(state)
        binding.consumed = True
        binding.trajectory.in_flight = True
        return state, binding


def _poison_trajectory(binding: _StateBinding) -> None:
    with _REGISTRY_LOCK:
        binding.trajectory.poisoned = True
        binding.trajectory.in_flight = False


def _complete_step(
    current: MatchedV3ForagaxBridgeState,
    binding: _StateBinding,
    *,
    observation: Array,
    environment_state: Any,
    environment_time: int,
    environment_key: Array,
) -> MatchedV3ForagaxBridgeState:
    capability = _StateCapability()
    state = MatchedV3ForagaxBridgeState(
        environment_seed=current.environment_seed,
        observation=observation,
        reset_count=1,
        step_count=current.step_count + 1,
        _runtime=current._runtime,
        _environment=current._environment,
        _params=current._params,
        _environment_state=environment_state,
        _environment_key=environment_key,
        _capability=capability,
    )
    next_binding = _StateBinding(
        trajectory=binding.trajectory,
        environment_seed=current.environment_seed,
        reset_count=1,
        step_count=current.step_count + 1,
        environment_key_words=_key_words(environment_key),
        environment=current._environment,
        params=current._params,
        environment_state=environment_state,
        environment_time=environment_time,
        observation=observation,
        runtime_capability=current._runtime._capability,
    )
    with _REGISTRY_LOCK:
        if binding.trajectory.poisoned or not binding.trajectory.in_flight:
            raise ForagerMatchedV3ForagaxBridgeError(
                "bridge trajectory registry changed during its environment step"
            )
        _STATE_REGISTRY[capability] = next_binding
        binding.trajectory.in_flight = False
    return state


def step_matched_v3_foragax_bridge(
    state: MatchedV3ForagaxBridgeState,
    action: object,
) -> MatchedV3ForagaxTransition:
    """Atomically consume and advance one exact continuing transition."""

    exact_action = _validate_action(action)
    current, binding = _consume_state(state)
    try:
        environment_key, step_key = jr.split(current._environment_key)
        step_result = current._environment.step(
            step_key,
            current._environment_state,
            jnp.asarray(exact_action, dtype=jnp.int32),
            current._params,
        )
        if type(step_result) is not tuple or len(step_result) != 5:
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax step must return exactly (observation, state, reward, done, info)"
            )
        raw_observation, environment_state, raw_reward, done, info = step_result
        observation = _validate_observation(raw_observation)
        reward = _validate_reward(raw_reward)
        _validate_done(done)
        _validate_info(info)
        environment_time = _environment_state_time(
            environment_state, label="post-step state"
        )
        if environment_time != binding.environment_time + 1:
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax environment state time did not advance by exactly one"
            )
        if environment_state is current._environment_state:
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax environment state identity did not advance"
            )
        next_state = _complete_step(
            current,
            binding,
            observation=observation,
            environment_state=environment_state,
            environment_time=environment_time,
            environment_key=environment_key,
        )
    except BaseException as exc:
        _poison_trajectory(binding)
        if isinstance(exc, ForagerMatchedV3ForagaxBridgeError):
            raise
        if isinstance(exc, Exception):
            raise ForagerMatchedV3ForagaxBridgeError(
                "Foragax step failed after the bridge state was consumed"
            ) from exc
        raise
    return MatchedV3ForagaxTransition(
        state=next_state,
        action=exact_action,
        reward=reward,
    )


__all__ = [
    "FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION",
    "FORAGAX_BRIDGE_DESCRIPTOR_SHA256",
    "FORAGAX_DISTRIBUTION",
    "FORAGAX_INSTALL_TREE_HASH_SCHEME",
    "FORAGAX_INSTALL_TREE_SHA256",
    "FORAGAX_REQUIRED_VERSION",
    "ForagerMatchedV3ForagaxBridgeError",
    "JAX_REQUIRED_VERSION",
    "JAXLIB_REQUIRED_VERSION",
    "MATCHED_V3_APERTURE_SIZE",
    "MATCHED_V3_ENVIRONMENT_ID",
    "MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_NUM_ACTIONS",
    "MATCHED_V3_OBSERVATION_SHAPE",
    "MATCHED_V3_OBSERVATION_TYPE",
    "MATCHED_V3_RAW_REWARD_VALUES",
    "MatchedV3ForagaxBridgeState",
    "MatchedV3ForagaxRuntime",
    "MatchedV3ForagaxRuntimeIdentity",
    "MatchedV3ForagaxTransition",
    "THREEFRY_IMPLEMENTATION",
    "UINT31_MAXIMUM",
    "canonical_matched_v3_foragax_bridge_descriptor_bytes",
    "initialize_matched_v3_foragax_bridge",
    "matched_v3_foragax_bridge_descriptor",
    "open_matched_v3_foragax_runtime",
    "parse_matched_v3_foragax_bridge_descriptor",
    "step_matched_v3_foragax_bridge",
]
