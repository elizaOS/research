"""Explicit source-replay verification for one v6 development run.

This verifier intentionally reuses :class:`HiddenPartnerLifecycleWorldV6Runner`.
It proves only that a candidate run bit-matches a fresh execution of the
currently bound source and runtime.  It is not an independent learner oracle,
an independent aggregate implementation, scientific evidence, or a promotion
decision.  Calling :func:`verify_v6_development_run_source_replay` executes the
full fixed scan and can require substantial compilation memory.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    HiddenPartnerLifecycleWorldV6Control,
    build_v6_diagnostic_controls,
    build_v6_primary_controls,
    validate_v6_control,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    HiddenPartnerLifecycleWorldV6Runner,
    V6DevelopmentRun,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    STRUCTURALLY_VALID_DEVELOPMENT_RUN,
    V6DevelopmentRunValidation,
    validate_hidden_partner_lifecycle_world_v6_development_run,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SOURCE_REPLAY_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.source-replay-development.v1"
)
SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN: Literal["SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN"] = (
    "SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN"
)
SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN: Literal["SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN"] = (
    "SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN"
)
SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN: Literal["SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN"] = (
    "SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN"
)

DEVELOPMENT_ONLY = True
STRUCTURAL_ONLY = False
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False
INDEPENDENT_LEARNER_OR_ACCUMULATOR_ORACLE = False

SourceReplayStatus = Literal[
    "SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN",
    "SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN",
    "SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN",
]

_MAX_DIFFERENCES = 4096


@dataclasses.dataclass(frozen=True, slots=True)
class V6SourceReplayDifference:
    """One deterministic path-addressed replay mismatch or verifier error."""

    code: str
    path: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class V6SourceReplayVerification:
    """Authority-free result of an explicitly requested canonical source replay.

    ``replay_attempted`` means ``runner.run`` was invoked. ``replay_executed``
    means that invocation returned a complete run.  Only ``replay_verified``
    denotes a successful exact comparison.
    """

    schema: str
    status: SourceReplayStatus
    development_only: bool
    structural_only: bool
    replay_verified: bool
    replay_attempted: bool
    replay_executed: bool
    candidate_structurally_valid: bool
    fresh_structurally_valid: bool
    independent_learner_or_accumulator_oracle: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    differences: tuple[V6SourceReplayDifference, ...]


@dataclasses.dataclass
class _ComparisonContext:
    differences: list[V6SourceReplayDifference] = dataclasses.field(default_factory=list)

    def add(self, code: str, path: str, message: str) -> None:
        if len(self.differences) < _MAX_DIFFERENCES:
            self.differences.append(V6SourceReplayDifference(code=code, path=path, message=message))
        elif len(self.differences) == _MAX_DIFFERENCES:
            self.differences.append(
                V6SourceReplayDifference(
                    code="DIFFERENCE_LIMIT",
                    path=path,
                    message=f"comparison stopped after {_MAX_DIFFERENCES} differences",
                )
            )


def _result(
    status: SourceReplayStatus,
    differences: list[V6SourceReplayDifference] | tuple[V6SourceReplayDifference, ...],
    *,
    replay_attempted: bool,
    replay_executed: bool,
    candidate_structurally_valid: bool,
    fresh_structurally_valid: bool,
) -> V6SourceReplayVerification:
    verified = status == SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN
    return V6SourceReplayVerification(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SOURCE_REPLAY_SCHEMA,
        status=status,
        development_only=DEVELOPMENT_ONLY,
        structural_only=STRUCTURAL_ONLY,
        replay_verified=verified,
        replay_attempted=replay_attempted,
        replay_executed=replay_executed,
        candidate_structurally_valid=candidate_structurally_valid,
        fresh_structurally_valid=fresh_structurally_valid,
        independent_learner_or_accumulator_oracle=INDEPENDENT_LEARNER_OR_ACCUMULATOR_ORACLE,
        execution_authorized=EXECUTION_AUTHORIZED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        differences=tuple(differences),
    )


def _array_bytes(value: jax.Array) -> tuple[str, tuple[int, ...], str, bytes]:
    """Return an implementation-aware bit payload for one concrete JAX leaf."""

    if isinstance(value, jax.core.Tracer):
        raise TypeError("JAX tracers are not concrete replay values")
    is_key = jnp.issubdtype(value.dtype, jax.dtypes.prng_key)
    if is_key:
        key_impl = str(jr.key_impl(value))
        host = np.asarray(jax.device_get(jr.key_data(value)))
        return key_impl, tuple(host.shape), host.dtype.str, host.tobytes(order="C")
    host = np.asarray(jax.device_get(value))
    return "ordinary-array", tuple(host.shape), host.dtype.str, host.tobytes(order="C")


def _compare_value(
    ctx: _ComparisonContext,
    candidate: object,
    fresh: object,
    *,
    path: str,
) -> None:
    if len(ctx.differences) > _MAX_DIFFERENCES:
        return

    # Arrays precede class identity because backend-specific JAX concrete array
    # classes are not part of the run schema.  Their shape, dtype, key
    # implementation, and payload bits are part of it.
    if isinstance(candidate, jax.Array) or isinstance(fresh, jax.Array):
        if not isinstance(candidate, jax.Array) or not isinstance(fresh, jax.Array):
            ctx.add("TYPE_MISMATCH", path, "one value is a JAX array and the other is not")
            return
        if candidate.shape != fresh.shape:
            ctx.add("ARRAY_SHAPE", path, "JAX array shapes differ")
            return
        if candidate.dtype != fresh.dtype:
            ctx.add("ARRAY_DTYPE", path, "JAX array dtypes differ")
            return
        try:
            candidate_payload = _array_bytes(candidate)
            fresh_payload = _array_bytes(fresh)
        except (TypeError, ValueError, RuntimeError) as exc:
            ctx.add("ARRAY_CONCRETE", path, f"cannot materialize JAX array bits: {exc}")
            return
        if candidate_payload != fresh_payload:
            code = (
                "PRNG_KEY_BITS"
                if jnp.issubdtype(candidate.dtype, jax.dtypes.prng_key)
                else "ARRAY_BITS"
            )
            ctx.add(code, path, "JAX array payload differs bit-exactly")
        return

    # NumPy arrays are outside the strict in-memory v6 run contract even when
    # both sides happen to contain identical bytes.
    if isinstance(candidate, np.ndarray) or isinstance(fresh, np.ndarray):
        ctx.add("NUMPY_ARRAY", path, "NumPy arrays are not accepted replay leaves")
        return

    # Dataclasses precede mappings: chex dataclasses can expose mapping-like
    # behavior, but their exact class and declared field order are load-bearing.
    candidate_dataclass = dataclasses.is_dataclass(candidate) and not isinstance(candidate, type)
    fresh_dataclass = dataclasses.is_dataclass(fresh) and not isinstance(fresh, type)
    if candidate_dataclass or fresh_dataclass:
        if not candidate_dataclass or not fresh_dataclass:
            ctx.add("TYPE_MISMATCH", path, "one value is a dataclass and the other is not")
            return
        if type(candidate) is not type(fresh):
            ctx.add(
                "CLASS_MISMATCH",
                path,
                f"dataclass types differ: {type(candidate).__name__} != {type(fresh).__name__}",
            )
            return
        candidate_fields = tuple(field.name for field in dataclasses.fields(cast(Any, candidate)))
        fresh_fields = tuple(field.name for field in dataclasses.fields(cast(Any, fresh)))
        if candidate_fields != fresh_fields:
            ctx.add("FIELD_ORDER", path, "dataclass field order differs")
            return
        for field_name in candidate_fields:
            _compare_value(
                ctx,
                getattr(candidate, field_name),
                getattr(fresh, field_name),
                path=f"{path}.{field_name}",
            )
        return

    if type(candidate) is not type(fresh):
        ctx.add(
            "TYPE_MISMATCH",
            path,
            f"exact Python types differ: {type(candidate).__name__} != {type(fresh).__name__}",
        )
        return

    if type(candidate) is tuple:
        candidate_tuple = cast(tuple[object, ...], candidate)
        fresh_tuple = cast(tuple[object, ...], fresh)
        if len(candidate_tuple) != len(fresh_tuple):
            ctx.add("TUPLE_LENGTH", path, "tuple lengths differ")
            return
        for index, (candidate_item, fresh_item) in enumerate(
            zip(candidate_tuple, fresh_tuple, strict=True)
        ):
            _compare_value(ctx, candidate_item, fresh_item, path=f"{path}[{index}]")
        return

    if candidate is None:
        return
    if type(candidate) is float:
        if struct.pack("!d", candidate) != struct.pack("!d", cast(float, fresh)):
            ctx.add("HOST_FLOAT_BITS", path, "host float payload differs bit-exactly")
        return
    if type(candidate) is complex:
        fresh_complex = cast(complex, fresh)
        candidate_bits = struct.pack("!dd", candidate.real, candidate.imag)
        fresh_bits = struct.pack("!dd", fresh_complex.real, fresh_complex.imag)
        if candidate_bits != fresh_bits:
            ctx.add("HOST_COMPLEX_BITS", path, "host complex payload differs bit-exactly")
        return
    if isinstance(candidate, np.generic):
        candidate_scalar = candidate
        fresh_scalar = cast(np.generic, fresh)
        if candidate_scalar.dtype != fresh_scalar.dtype or (
            candidate_scalar.tobytes() != fresh_scalar.tobytes()
        ):
            ctx.add("NUMPY_SCALAR_BITS", path, "NumPy scalar payload differs bit-exactly")
        return
    if type(candidate) in (bool, int, str, bytes):
        if candidate != fresh:
            ctx.add("HOST_VALUE", path, "exact host scalar value differs")
        return
    ctx.add(
        "UNSUPPORTED_TYPE",
        path,
        (
            "replay comparator does not accept "
            f"{type(candidate).__module__}.{type(candidate).__qualname__}"
        ),
    )


def compare_v6_development_runs_bit_exact(
    candidate: object,
    fresh: object,
) -> tuple[V6SourceReplayDifference, ...]:
    """Return every bounded bit-level difference between two exact run records."""

    ctx = _ComparisonContext()
    if type(candidate) is not V6DevelopmentRun:
        ctx.add("TYPE", "run", "candidate must be an exact V6DevelopmentRun")
        return tuple(ctx.differences)
    if type(fresh) is not V6DevelopmentRun:
        ctx.add("TYPE", "fresh", "fresh value must be an exact V6DevelopmentRun")
        return tuple(ctx.differences)
    _compare_value(ctx, candidate, fresh, path="run")
    return tuple(ctx.differences)


def _structural_success(result: object) -> bool:
    return bool(
        type(result) is V6DevelopmentRunValidation
        and result.status == STRUCTURALLY_VALID_DEVELOPMENT_RUN
        and result.development_only is True
        and result.structural_only is True
        and result.replay_verified is False
        and result.execution_authorized is False
        and result.evidence_authorized is False
        and result.scientific_promotion_allowed is False
        and not result.errors
    )


def _canonical_control(name: object, primary: object) -> HiddenPartnerLifecycleWorldV6Control:
    if type(name) is not str:
        raise TypeError("candidate control_name must be an exact built-in str")
    if type(primary) is not bool:
        raise TypeError("candidate primary must be an exact built-in bool")
    controls = build_v6_primary_controls() if primary else build_v6_diagnostic_controls()
    matches = tuple(control for control in controls if control.name == name)
    if len(matches) != 1:
        raise ValueError("candidate control identity does not resolve to one canonical control")
    return validate_v6_control(matches[0])


def _source_keys(candidate: V6DevelopmentRun) -> tuple[jax.Array, jax.Array]:
    key_data = candidate.rng.supplied_key_data
    if isinstance(key_data, jax.core.Tracer):
        raise TypeError("candidate supplied key data must not be a JAX tracer")
    if isinstance(key_data, np.ndarray) or not isinstance(key_data, jax.Array):
        raise TypeError("candidate supplied key data must be a concrete JAX array")
    if key_data.shape != (2, 2) or key_data.dtype != jnp.uint32:
        raise TypeError("candidate supplied key data must be uint32[2,2]")
    host = np.asarray(jax.device_get(key_data))
    keys = tuple(
        jr.wrap_key_data(
            jnp.asarray(host[index], dtype=jnp.uint32),
            impl="threefry2x32",
        )
        for index in range(2)
    )
    for index, key in enumerate(keys):
        if (
            key.shape != ()
            or not jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
            or str(jr.key_impl(key)) != "threefry2x32"
            or not np.array_equal(
                np.asarray(jax.device_get(jr.key_data(key))),
                host[index],
            )
        ):
            raise RuntimeError("typed threefry key reconstruction failed closed")
    return keys[0], keys[1]


def verify_v6_development_run_source_replay(
    candidate: object,
) -> V6SourceReplayVerification:
    """Execute and bit-compare one canonical source replay after strict preflight.

    Candidate plan, state, aggregates, stream, and configuration digests are
    never execution inputs.  Only the structurally validated control identity
    and strict supplied key data select the canonical runner invocation.
    """

    try:
        candidate_validation = validate_hidden_partner_lifecycle_world_v6_development_run(candidate)
    except Exception as exc:  # noqa: BLE001 - public fail-closed boundary
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="CANDIDATE_PREFLIGHT_EXCEPTION",
                    path="run",
                    message=f"candidate structural preflight raised {type(exc).__name__}: {exc}",
                )
            ],
            replay_attempted=False,
            replay_executed=False,
            candidate_structurally_valid=False,
            fresh_structurally_valid=False,
        )
    if not _structural_success(candidate_validation):
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="CANDIDATE_PREFLIGHT",
                    path="run",
                    message="candidate did not pass strict structural validation",
                )
            ],
            replay_attempted=False,
            replay_executed=False,
            candidate_structurally_valid=False,
            fresh_structurally_valid=False,
        )
    if type(candidate) is not V6DevelopmentRun:
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="CANDIDATE_TYPE",
                    path="run",
                    message="structural validator accepted a noncanonical candidate type",
                )
            ],
            replay_attempted=False,
            replay_executed=False,
            candidate_structurally_valid=True,
            fresh_structurally_valid=False,
        )

    try:
        control = _canonical_control(candidate.control_name, candidate.primary)
        world_key, agent_key = _source_keys(candidate)
        runner = HiddenPartnerLifecycleWorldV6Runner(control)
    except Exception as exc:  # noqa: BLE001 - public fail-closed boundary
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="REPLAY_SETUP",
                    path="run",
                    message=f"canonical replay setup raised {type(exc).__name__}: {exc}",
                )
            ],
            replay_attempted=False,
            replay_executed=False,
            candidate_structurally_valid=True,
            fresh_structurally_valid=False,
        )

    try:
        fresh = runner.run(world_key, agent_key)
    except Exception as exc:  # noqa: BLE001 - public fail-closed boundary
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="REPLAY_EXECUTION",
                    path="fresh",
                    message=f"canonical runner raised {type(exc).__name__}: {exc}",
                )
            ],
            replay_attempted=True,
            replay_executed=False,
            candidate_structurally_valid=True,
            fresh_structurally_valid=False,
        )

    try:
        fresh_validation = validate_hidden_partner_lifecycle_world_v6_development_run(fresh)
    except Exception as exc:  # noqa: BLE001 - public fail-closed boundary
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="FRESH_PREFLIGHT_EXCEPTION",
                    path="fresh",
                    message=f"fresh structural validation raised {type(exc).__name__}: {exc}",
                )
            ],
            replay_attempted=True,
            replay_executed=True,
            candidate_structurally_valid=True,
            fresh_structurally_valid=False,
        )
    if not _structural_success(fresh_validation):
        return _result(
            SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN,
            [
                V6SourceReplayDifference(
                    code="FRESH_PREFLIGHT",
                    path="fresh",
                    message="fresh canonical run failed strict structural validation",
                )
            ],
            replay_attempted=True,
            replay_executed=True,
            candidate_structurally_valid=True,
            fresh_structurally_valid=False,
        )

    differences = compare_v6_development_runs_bit_exact(candidate, fresh)
    status: SourceReplayStatus = (
        SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN
        if not differences
        else SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN
    )
    return _result(
        status,
        differences,
        replay_attempted=True,
        replay_executed=True,
        candidate_structurally_valid=True,
        fresh_structurally_valid=True,
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SOURCE_REPLAY_SCHEMA",
    "INDEPENDENT_LEARNER_OR_ACCUMULATOR_ORACLE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_REPLAY_ERROR_DEVELOPMENT_RUN",
    "SOURCE_REPLAY_MISMATCH_DEVELOPMENT_RUN",
    "SOURCE_REPLAY_VERIFIED_DEVELOPMENT_RUN",
    "STRUCTURAL_ONLY",
    "V6SourceReplayDifference",
    "V6SourceReplayVerification",
    "compare_v6_development_runs_bit_exact",
    "verify_v6_development_run_source_replay",
]
