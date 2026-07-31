"""Exact-source adapter for an RNG-isolated upstream PPO/RTU-PPO runner.

The audited upstream ``rtu_ppo.py`` uses the same random key for policy action
sampling and for the environment transition.  That makes its reward trace
ineligible for paired comparisons with runners whose environment key stream is
independent of agent-side random-number consumption.

This module does not silently monkeypatch an import.  It accepts only the
audited upstream source bytes, applies a fixed set of single-occurrence byte
replacements, validates the resulting Python AST, and returns both the derived
source and a canonical provenance descriptor.  A matched-suite builder can
then bake both source trees into one content-addressed OCI image.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

UPSTREAM_RTU_PPO_SOURCE_SHA256: Final = (
    "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
)
UPSTREAM_SOURCE_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
UPSTREAM_SOURCE_PATH: Final = "src/rtu_ppo.py"
ISOLATED_AGENT_RNG_NAMESPACE: Final = 0xA63E7C11
PATCH_SCHEMA_VERSION: Final = (
    "alberta.forager_rtu_ppo_rng_isolation_patch.v1"
)
PATCH_MODE: Final = "exact_single_occurrence_source_replacements_v1"
ENVIRONMENT_RNG_SCHEDULE: Final = "dedicated_environment_split_chain_v1"
EXPECTED_PATCHED_RTU_PPO_SHA256: Final = (
    "c47f3e087cb01722e824efc1d62c2e5880e75a2d937ae8fc122af24ce8967f2d"
)


class RTUPPORngIsolationError(ValueError):
    """Raised when source identity or derived-runner structure is not exact."""


@dataclasses.dataclass(frozen=True)
class SourceReplacement:
    """One exact, single-occurrence source transformation."""

    replacement_id: str
    before: bytes
    after: bytes


@dataclasses.dataclass(frozen=True)
class IsolatedRTUPPOSource:
    """Derived source bytes and their immutable provenance descriptor."""

    source: bytes
    source_sha256: str
    descriptor: Mapping[str, Any]


_REPLACEMENTS: Final = (
    SourceReplacement(
        replacement_id="declare_isolated_agent_namespace",
        before=b"PERIOD = 182500\n",
        after=(
            b"PERIOD = 182500\n"
            b"ISOLATED_AGENT_RNG_NAMESPACE = 0xA63E7C11\n"
        ),
    ),
    SourceReplacement(
        replacement_id="carry_environment_rng_in_gymnax_state",
        before=b"    env_state: Any = struct.field(pytree_node=True)\n",
        after=(
            b"    env_state: Any = struct.field(pytree_node=True)\n"
            b"    environment_rng: Any = struct.field(pytree_node=True)\n"
        ),
    ),
    SourceReplacement(
        replacement_id="split_environment_rng_before_transition",
        before=(
            b"    # STEP ENV\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        _rng, gymnax_state.env_state, action.squeeze(), "
            b"gymnax_state.env_params\n"
            b"    )\n"
        ),
        after=(
            b"    # STEP ENV\n"
            b"    environment_rng, environment_step_rng = jax.random.split(\n"
            b"        gymnax_state.environment_rng\n"
            b"    )\n"
            b"    obs, env_state, reward, done, info = gymnax_state.env_step(\n"
            b"        environment_step_rng,\n"
            b"        gymnax_state.env_state,\n"
            b"        action.squeeze(),\n"
            b"        gymnax_state.env_params,\n"
            b"    )\n"
        ),
    ),
    SourceReplacement(
        replacement_id="persist_environment_rng_after_transition",
        before=(
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
        after=(
            b"        env_params=gymnax_state.env_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    runner_state = (\n"
        ),
    ),
    SourceReplacement(
        replacement_id="root_environment_and_agent_streams",
        before=(
            b"    rng, reset_rng = jax.random.split(rng)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
        ),
        after=(
            b"    environment_rng = rng\n"
            b"    environment_rng, reset_rng = jax.random.split(environment_rng)\n"
            b"    rng = jax.random.fold_in(rng, ISOLATED_AGENT_RNG_NAMESPACE)\n"
            b"    obs, env_state = env.reset(reset_rng, env.default_params)\n"
        ),
    ),
    SourceReplacement(
        replacement_id="initialize_environment_rng_carry",
        before=(
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
        after=(
            b"        env_params=env.default_params,\n"
            b"        env_state=env_state,\n"
            b"        environment_rng=environment_rng,\n"
            b"    )\n"
            b"    action_dim = 4\n"
        ),
    ),
    SourceReplacement(
        replacement_id="preserve_environment_rng_across_update_boundary",
        before=(
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
        after=(
            b"            env_params=gymnax_state.env_params,\n"
            b"            env_state=gymnax_state.env_state,\n"
            b"            environment_rng=gymnax_state.environment_rng,\n"
            b"        )\n"
            b"\n"
            b"        env_step_state = (\n"
        ),
    ),
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain_json(value: Any, *, path: str = "descriptor") -> Any:
    """Return a detached JSON value from mutable or frozen containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RTUPPORngIsolationError(
                    f"{path} contains a non-string object key"
                )
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _plain_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RTUPPORngIsolationError(
        f"{path} contains a non-JSON value of type {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a previously validated JSON value."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RTUPPORngIsolationError(
            "patch descriptor is not canonical JSON"
        ) from exc


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RTUPPORngIsolationError(
            f"derived source must define exactly one {name}()"
        )
    return matches[0]


def _call_paths(function: ast.FunctionDef) -> list[tuple[str, ast.Call]]:
    result: list[tuple[str, ast.Call]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            path = _attribute_path(node.func)
            if path is not None:
                result.append((path, node))
    return result


def _assigned_calls(
    function: ast.FunctionDef,
    path: str,
) -> list[tuple[ast.Assign, ast.Call]]:
    result: list[tuple[ast.Assign, ast.Call]] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _attribute_path(node.value.func) == path
        ):
            result.append((node, node.value))
    return result


def _single_tuple_target_names(assignment: ast.Assign) -> tuple[str, ...] | None:
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], (ast.Tuple, ast.List))
    ):
        return None
    names: list[str] = []
    for element in assignment.targets[0].elts:
        if not isinstance(element, ast.Name):
            return None
        names.append(element.id)
    return tuple(names)


def _single_name_target(assignment: ast.Assign) -> str | None:
    if len(assignment.targets) != 1 or not isinstance(
        assignment.targets[0],
        ast.Name,
    ):
        return None
    return assignment.targets[0].id


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    values = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(values) != 1:
        return None
    return values[0]


def _validate_derived_ast(source: bytes) -> None:
    try:
        decoded = source.decode("utf-8")
        module = ast.parse(decoded, filename=UPSTREAM_SOURCE_PATH)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise RTUPPORngIsolationError(
            "derived source is not valid UTF-8 Python"
        ) from exc

    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GymnaxEnvState"
    ]
    if len(classes) != 1:
        raise RTUPPORngIsolationError(
            "derived source must define exactly one GymnaxEnvState"
        )
    annotated_fields = {
        node.target.id
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    if "environment_rng" not in annotated_fields:
        raise RTUPPORngIsolationError(
            "GymnaxEnvState does not carry the environment RNG"
        )

    env_step = _function(module, "env_step")
    env_calls = _call_paths(env_step)
    split_assignments = [
        (assignment, call)
        for assignment, call in _assigned_calls(env_step, "jax.random.split")
        if len(call.args) == 1
        and _attribute_path(call.args[0])
        == "gymnax_state.environment_rng"
    ]
    if (
        len(split_assignments) != 1
        or _single_tuple_target_names(split_assignments[0][0])
        != ("environment_rng", "environment_step_rng")
    ):
        raise RTUPPORngIsolationError(
            "env_step must split and retain exactly one carried environment RNG"
        )
    transition_calls = [
        call
        for path, call in env_calls
        if path == "gymnax_state.env_step"
    ]
    if (
        len(transition_calls) != 1
        or not transition_calls[0].args
        or not isinstance(transition_calls[0].args[0], ast.Name)
        or transition_calls[0].args[0].id != "environment_step_rng"
    ):
        raise RTUPPORngIsolationError(
            "environment transition does not consume environment_step_rng"
        )
    action_calls = [
        call for path, call in env_calls if path == "agent_step"
    ]
    if (
        len(action_calls) != 1
        or len(action_calls[0].args) < 3
        or not isinstance(action_calls[0].args[2], ast.Name)
        or action_calls[0].args[2].id != "_rng"
    ):
        raise RTUPPORngIsolationError(
            "policy action sampling no longer consumes the upstream agent key"
        )

    experiment = _function(module, "experiment")
    experiment_calls = _call_paths(experiment)
    reset_splits = [
        (assignment, call)
        for assignment, call in _assigned_calls(experiment, "jax.random.split")
        if len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "environment_rng"
    ]
    agent_folds = [
        (assignment, call)
        for assignment, call in _assigned_calls(experiment, "jax.random.fold_in")
        if len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "rng"
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "ISOLATED_AGENT_RNG_NAMESPACE"
    ]
    roots = [
        node
        for node in experiment.body
        if isinstance(node, ast.Assign)
        and _single_name_target(node) == "environment_rng"
        and isinstance(node.value, ast.Name)
        and node.value.id == "rng"
    ]
    if (
        len(roots) != 1
        or len(reset_splits) != 1
        or _single_tuple_target_names(reset_splits[0][0])
        != ("environment_rng", "reset_rng")
        or len(agent_folds) != 1
        or _single_name_target(agent_folds[0][0]) != "rng"
    ):
        raise RTUPPORngIsolationError(
            "experiment must root one environment chain and one namespaced "
            "agent chain"
        )

    gymnax_creates: list[ast.Call] = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and _attribute_path(node.func) == "GymnaxEnvState.create"
        ):
            gymnax_creates.append(node)
    if len(gymnax_creates) != 3:
        raise RTUPPORngIsolationError(
            "derived source has an unexpected GymnaxEnvState.create count"
        )
    for call in gymnax_creates:
        names = [keyword.arg for keyword in call.keywords]
        if names.count("environment_rng") != 1:
            raise RTUPPORngIsolationError(
                "every GymnaxEnvState.create must bind environment_rng once"
            )
    env_step_creates = [
        call
        for path, call in env_calls
        if path == "GymnaxEnvState.create"
    ]
    if len(env_step_creates) != 1:
        raise RTUPPORngIsolationError(
            "env_step must rebuild exactly one GymnaxEnvState"
        )
    env_step_carry = _keyword_value(env_step_creates[0], "environment_rng")
    if not (
        isinstance(env_step_carry, ast.Name)
        and env_step_carry.id == "environment_rng"
    ):
        raise RTUPPORngIsolationError(
            "env_step must persist the post-split environment RNG"
        )
    experiment_creates = [
        call
        for path, call in experiment_calls
        if path == "GymnaxEnvState.create"
    ]
    experiment_carries = [
        _attribute_path(value)
        if isinstance(value, ast.Attribute)
        else value.id
        if isinstance(value, ast.Name)
        else None
        for call in experiment_creates
        if (value := _keyword_value(call, "environment_rng")) is not None
    ]
    if sorted(experiment_carries, key=lambda value: "" if value is None else value) != [
        "environment_rng",
        "gymnax_state.environment_rng",
    ]:
        raise RTUPPORngIsolationError(
            "experiment must initialize and preserve the environment RNG carry"
        )


def derive_isolated_rtu_ppo_source(source: bytes) -> IsolatedRTUPPOSource:
    """Return the one audited RNG-isolated derivation of upstream source."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    observed_source_sha256 = _sha256(source)
    if observed_source_sha256 != UPSTREAM_RTU_PPO_SOURCE_SHA256:
        raise RTUPPORngIsolationError(
            "upstream rtu_ppo.py SHA-256 differs from the audited source"
        )

    derived = source
    replacement_records: list[dict[str, Any]] = []
    for replacement in _REPLACEMENTS:
        count = derived.count(replacement.before)
        if count != 1:
            raise RTUPPORngIsolationError(
                f"replacement {replacement.replacement_id!r} matched "
                f"{count} source locations instead of one"
            )
        derived = derived.replace(
            replacement.before,
            replacement.after,
            1,
        )
        replacement_records.append(
            {
                "replacement_id": replacement.replacement_id,
                "before_sha256": _sha256(replacement.before),
                "after_sha256": _sha256(replacement.after),
            }
        )

    _validate_derived_ast(derived)
    derived_sha256 = _sha256(derived)
    if derived_sha256 != EXPECTED_PATCHED_RTU_PPO_SHA256:
        raise RTUPPORngIsolationError(
            "derived rtu_ppo.py SHA-256 differs from the frozen adapter output"
        )
    descriptor: dict[str, Any] = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "patch_mode": PATCH_MODE,
        "upstream": {
            "repository": "https://github.com/steventango/continual-foragax-agents",
            "commit": UPSTREAM_SOURCE_COMMIT,
            "path": UPSTREAM_SOURCE_PATH,
            "sha256": observed_source_sha256,
        },
        "derived_source_sha256": derived_sha256,
        "environment_rng": {
            "schedule": ENVIRONMENT_RNG_SCHEDULE,
            "root": "jax.random.PRNGKey(seed)",
            "reset": (
                "environment_rng, reset_rng = "
                "jax.random.split(environment_rng)"
            ),
            "transition": (
                "environment_rng, environment_step_rng = "
                "jax.random.split(environment_rng)"
            ),
            "action_key_shared_with_environment": False,
        },
        "agent_rng": {
            "root": (
                "jax.random.fold_in(jax.random.PRNGKey(seed), "
                "ISOLATED_AGENT_RNG_NAMESPACE)"
            ),
            "namespace": ISOLATED_AGENT_RNG_NAMESPACE,
        },
        "replacement_records": replacement_records,
    }
    canonical = _canonical_json(descriptor)
    descriptor["payload_sha256"] = _sha256(canonical)
    return IsolatedRTUPPOSource(
        source=derived,
        source_sha256=derived_sha256,
        descriptor=_freeze_json(descriptor),
    )


def validate_isolated_rtu_ppo_source(
    upstream_source: bytes,
    derived_source: bytes,
    descriptor: Mapping[str, Any],
) -> None:
    """Re-derive and compare a candidate source/descriptor byte-for-byte."""
    if not isinstance(upstream_source, bytes):
        raise TypeError("upstream_source must be bytes")
    if not isinstance(derived_source, bytes):
        raise TypeError("derived_source must be bytes")
    if not isinstance(descriptor, Mapping):
        raise TypeError("descriptor must be a mapping")
    expected = derive_isolated_rtu_ppo_source(upstream_source)
    if not isinstance(descriptor, dict):
        raise RTUPPORngIsolationError(
            "descriptor must be a detached canonical object"
        )
    if derived_source != expected.source:
        raise RTUPPORngIsolationError(
            "derived source bytes differ from the exact re-derivation"
        )
    try:
        actual_descriptor = _canonical_json(descriptor)
        expected_descriptor = _canonical_json(dict(expected.descriptor))
    except RTUPPORngIsolationError:
        raise
    if actual_descriptor != expected_descriptor:
        raise RTUPPORngIsolationError(
            "patch descriptor differs from the exact re-derivation"
        )
