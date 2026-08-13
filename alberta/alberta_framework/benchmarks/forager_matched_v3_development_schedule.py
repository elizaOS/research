"""Pure, content-addressed development scheduling for matched Forager v3.

This module accepts an externally issued *open development* seed registry and
builds the exact block-major Cartesian schedule over the 25 inferential v3
candidates.  It is deliberately a data-contract layer only: it has no seed
issuer, filesystem surface, runner, result loader, authority token, or
promotion path.

Every accepted document is bounded canonical ASCII JSON terminated by one
newline.  Registry and schedule body hashes are self-authenticating only; a
caller must additionally carry the full-file SHA-256.  Content bindings name
bytes but do not attest their provenance or meaning.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import forager_matched_v3_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_v3_configuration_plan as plan
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_seed_registry.v1"
)
DEVELOPMENT_CELL_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_cell.v1"
)
DEVELOPMENT_SCHEDULE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_schedule.v1"
)
DEVELOPMENT_RETRY_POLICY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_retry_policy.v1"
)
DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_seed_derivation.v1"
)
DEVELOPMENT_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.development_seed_provider_receipt.v1"
)
DEVELOPMENT_SEED_DERIVATION_DOMAIN: Final = (
    "alberta.forager.matched_v3.open_development_selection.seed.v1"
)

DEVELOPMENT_CLASSIFICATION: Final = "development_selection_nonpromoting"
DEVELOPMENT_SEED_CLASSIFICATION: Final = (
    "provisioned_open_development_seeds_nonpromoting"
)
DEVELOPMENT_STAGE: Final = "development_selection_v3"
DEVELOPMENT_ORDERING: Final = (
    "block_major_then_frozen_inferential_candidate_order"
)

_MAX_REGISTRY_BYTES: Final = 8 * 1024 * 1024
_MAX_SCHEDULE_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_NODES: Final = 1_000_000
_MAX_JSON_INTEGER_DIGITS: Final = 20
_MAX_BLOCKS: Final = 512
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_RE: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")
_PORTABLE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_CELL_ID_RE: Final = re.compile(r"cell_([0-9a-f]{16})_([0-9a-f]{64})\Z")

_CLAIMS: Final[dict[str, bool]] = {
    "confirmatory_or_held_out": False,
    "execution_authorized": False,
    "scientific_evidence": False,
    "scientific_promotion_allowed": False,
    "universal_sota_claim": False,
}
_REGISTRY_LIMITATIONS: Final[tuple[str, ...]] = (
    "The registry parser does not issue seeds or authenticate the provider receipt bytes.",
    "The registry is open development material and is never a held-out seed source.",
    "Content and body hashes establish byte identity, not randomness provenance.",
)
_SCHEDULE_LIMITATIONS: Final[tuple[str, ...]] = (
    "The schedule is structural data and does not execute a candidate.",
    "Qualification and execution bindings establish content identity only.",
    "Development outcomes cannot be promoted as confirmatory or scientific evidence.",
)


class ForagerMatchedV3DevelopmentScheduleError(ValueError):
    """A development registry, binding, cell, or schedule failed closed."""


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _require_schema(value: object, label: str) -> str:
    if type(value) is not str or _SCHEMA_RE.fullmatch(value) is None:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be a bounded lowercase schema identifier"
        )
    return value


def _require_portable_id(value: object, label: str) -> str:
    if type(value) is not str or _PORTABLE_ID_RE.fullmatch(value) is None:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be a bounded portable identifier"
        )
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} fields differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _require_object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be a plain JSON object"
        )
    return cast(dict[str, Any], value)


def _require_list(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be a plain JSON array"
        )
    return value


def _validate_plain_unaliased_tree(value: object, *, label: str) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"{label} exceeds the JSON node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"{label} exceeds the JSON depth bound"
            )
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3DevelopmentScheduleError(
                    f"{label} must be an unaliased acyclic JSON tree"
                )
            seen.add(identity)
            mapping = cast(dict[object, object], item)
            for key, child in mapping.items():
                if type(key) is not str:
                    raise ForagerMatchedV3DevelopmentScheduleError(
                        f"{label} object keys must be exact strings"
                    )
                pending.append((child, depth + 1))
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3DevelopmentScheduleError(
                    f"{label} must be an unaliased acyclic JSON tree"
                )
            seen.add(identity)
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        elif type(item) not in {str, int, bool, type(None)}:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"{label} contains forbidden JSON value type {type(item).__name__}"
            )


def _canonical_json_bytes(value: object, *, label: str, maximum: int) -> bytes:
    _validate_plain_unaliased_tree(value, label=label)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} is not finite canonical ASCII JSON"
        ) from exc
    if len(raw) > maximum:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} exceeds its canonical byte bound"
        )
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"duplicate JSON key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ForagerMatchedV3DevelopmentScheduleError(
        f"non-finite JSON number {value!r} is forbidden"
    )


def _reject_float(value: str) -> NoReturn:
    raise ForagerMatchedV3DevelopmentScheduleError(
        f"floating-point JSON number {value!r} is forbidden"
    )


def _bounded_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if not digits or len(digits) > _MAX_JSON_INTEGER_DIGITS:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "JSON integer exceeds the digit bound"
        )
    return int(value)


def _decode_canonical_json(raw: bytes, *, label: str, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} violates its JSON byte bound"
        )
    try:
        decoded = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float,
            parse_int=_bounded_json_integer,
        )
    except ForagerMatchedV3DevelopmentScheduleError:
        raise
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    payload = _require_object(decoded, label)
    if _canonical_json_bytes(payload, label=label, maximum=maximum) != raw:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} is not the exact canonical newline-terminated encoding"
        )
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(body: dict[str, Any], *, label: str, maximum: int) -> str:
    return _sha256(_canonical_json_bytes(body, label=label, maximum=maximum))


@dataclass(frozen=True, slots=True)
class ContentBinding:
    """A generic immutable schema/full-file/body content identity."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, "content binding schema_version")
        _require_sha256(self.file_sha256, "content binding file_sha256")
        _require_sha256(self.body_sha256, "content binding body_sha256")

    def to_payload(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicyBinding:
    """Content identity for a separately frozen development retry policy."""

    schema_version: str
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != DEVELOPMENT_RETRY_POLICY_SCHEMA_VERSION:
            raise ForagerMatchedV3DevelopmentScheduleError(
                "retry policy uses the wrong development schema"
            )
        _require_sha256(self.sha256, "retry policy sha256")

    def to_payload(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DescriptorBinding:
    """A generic immutable schema/descriptor content identity."""

    schema_version: str
    sha256: str

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, "descriptor binding schema_version")
        _require_sha256(self.sha256, "descriptor binding sha256")

    def to_payload(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentAgentSeed:
    """One candidate-private seed in an externally issued development block."""

    candidate_id: str
    namespace: str
    seed: int

    def __post_init__(self) -> None:
        if self.candidate_id not in (
            universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development agent seed candidate is not in the inferential universe"
            )
        if self.namespace != f"agent/{self.candidate_id}":
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development agent seed namespace does not bind its candidate"
            )
        _require_exact_int(
            self.seed,
            "development agent seed",
            minimum=0,
            maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "namespace": self.namespace,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentSeedBlock:
    """One ordered development block; numeric seed collisions remain valid."""

    block_ordinal: int
    block_id: str
    derivation_payload_sha256: str
    environment_seed: int
    agent_seeds: tuple[DevelopmentAgentSeed, ...]

    def __post_init__(self) -> None:
        _require_exact_int(
            self.block_ordinal,
            "development seed block ordinal",
            minimum=0,
            maximum=_MAX_BLOCKS - 1,
        )
        _require_portable_id(self.block_id, "development seed block ID")
        _require_sha256(
            self.derivation_payload_sha256,
            "development seed block derivation digest",
        )
        _require_exact_int(
            self.environment_seed,
            "development seed block environment seed",
            minimum=0,
            maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
        )
        if type(self.agent_seeds) is not tuple or any(
            type(item) is not DevelopmentAgentSeed for item in self.agent_seeds
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development seed block agent_seeds must be an exact immutable tuple"
            )
        if tuple(item.candidate_id for item in self.agent_seeds) != (
            universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development seed block agent order is not the exact inferential order"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "block_ordinal": self.block_ordinal,
            "block_id": self.block_id,
            "derivation_payload_sha256": self.derivation_payload_sha256,
            "environment_seed": self.environment_seed,
            "agent_seeds": [item.to_payload() for item in self.agent_seeds],
        }


@dataclass(frozen=True, slots=True)
class DevelopmentSeedRegistry:
    """Detached immutable value accepted from an external seed provider."""

    candidate_order: tuple[str, ...]
    provider_receipt: ContentBinding
    blocks: tuple[DevelopmentSeedBlock, ...]
    registry_body_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        if self.candidate_order != (
            universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry value has the wrong candidate order"
            )
        if (
            type(self.provider_receipt) is not ContentBinding
            or self.provider_receipt.schema_version
            != DEVELOPMENT_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry value has the wrong provider receipt"
            )
        if (
            type(self.blocks) is not tuple
            or not 1 <= len(self.blocks) <= _MAX_BLOCKS
            or any(type(block) is not DevelopmentSeedBlock for block in self.blocks)
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry blocks must be a bounded immutable tuple"
            )
        if tuple(block.block_ordinal for block in self.blocks) != tuple(
            range(len(self.blocks))
        ) or len({block.block_id for block in self.blocks}) != len(self.blocks):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry block identity or order drifted"
            )
        body_digest = _body_sha256(
            self.body_payload(),
            label="development seed registry body",
            maximum=_MAX_REGISTRY_BYTES,
        )
        supplied_body = _require_sha256(
            self.registry_body_sha256, "development registry value body digest"
        )
        if not hmac.compare_digest(body_digest, supplied_body):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry value body digest does not replay"
            )
        raw = _canonical_json_bytes(
            self.to_payload(),
            label="development seed registry",
            maximum=_MAX_REGISTRY_BYTES,
        )
        supplied_file = _require_sha256(
            self.file_sha256, "development registry value file digest"
        )
        if not hmac.compare_digest(_sha256(raw), supplied_file):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development registry value file digest does not replay"
            )

    def body_payload(self) -> dict[str, Any]:
        return {
            "schema_version": DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION,
            "classification": DEVELOPMENT_SEED_CLASSIFICATION,
            "stage": DEVELOPMENT_STAGE,
            "candidate_universe_sha256": (
                universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
            ),
            "candidate_order": list(self.candidate_order),
            "derivation_schema_version": (
                DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION
            ),
            "derivation_domain": DEVELOPMENT_SEED_DERIVATION_DOMAIN,
            "provider_receipt": self.provider_receipt.to_payload(),
            "blocks": [block.to_payload() for block in self.blocks],
            "claims": dict(_CLAIMS),
            "limitations": list(_REGISTRY_LIMITATIONS),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.body_payload(), "registry_body_sha256": self.registry_body_sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentCell:
    """One immutable block/candidate cell in the development schedule."""

    cell_id: str
    ordinal: int
    block_ordinal: int
    candidate_ordinal: int
    block_id: str
    candidate_id: str
    analysis_role: str
    development_selection_group: str
    derivation_payload_sha256: str
    environment_seed: int
    agent_seed_namespace: str
    agent_seed: int
    configuration_record_sha256: str
    candidate_execution_binding_sha256: str
    cell_body_sha256: str

    def __post_init__(self) -> None:
        candidates = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        _require_exact_int(
            self.ordinal,
            "development cell ordinal",
            minimum=0,
            maximum=_MAX_BLOCKS * len(candidates) - 1,
        )
        _require_exact_int(
            self.block_ordinal,
            "development cell block ordinal",
            minimum=0,
            maximum=_MAX_BLOCKS - 1,
        )
        _require_exact_int(
            self.candidate_ordinal,
            "development cell candidate ordinal",
            minimum=0,
            maximum=len(candidates) - 1,
        )
        if (
            self.ordinal != self.block_ordinal * len(candidates) + self.candidate_ordinal
            or self.candidate_id != candidates[self.candidate_ordinal]
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development cell ordinal/candidate coordinates drifted"
            )
        _require_portable_id(self.block_id, "development cell block ID")
        metadata = _candidate_metadata()[self.candidate_id]
        if (
            self.analysis_role,
            self.development_selection_group,
        ) != metadata:
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development cell candidate metadata drifted"
            )
        _require_sha256(
            self.derivation_payload_sha256,
            "development cell derivation payload digest",
        )
        _require_exact_int(
            self.environment_seed,
            "development cell environment seed",
            minimum=0,
            maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
        )
        if self.agent_seed_namespace != f"agent/{self.candidate_id}":
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development cell agent namespace drifted"
            )
        _require_exact_int(
            self.agent_seed,
            "development cell agent seed",
            minimum=0,
            maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
        )
        _require_sha256(
            self.configuration_record_sha256,
            "development cell configuration record digest",
        )
        _require_sha256(
            self.candidate_execution_binding_sha256,
            "development cell execution binding digest",
        )
        replayed = _body_sha256(
            self.body_payload(),
            label=f"development cell {self.ordinal} body",
            maximum=64 * 1024,
        )
        supplied = _require_sha256(
            self.cell_body_sha256, "development cell body digest"
        )
        if (
            not hmac.compare_digest(replayed, supplied)
            or self.cell_id != f"cell_{self.ordinal:016x}_{supplied}"
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development cell body or identity does not replay"
            )

    def body_payload(self) -> dict[str, object]:
        return {
            "schema_version": DEVELOPMENT_CELL_SCHEMA_VERSION,
            "ordinal": self.ordinal,
            "block_ordinal": self.block_ordinal,
            "candidate_ordinal": self.candidate_ordinal,
            "block_id": self.block_id,
            "candidate_id": self.candidate_id,
            "analysis_role": self.analysis_role,
            "development_selection_group": self.development_selection_group,
            "derivation_payload_sha256": self.derivation_payload_sha256,
            "environment_seed": self.environment_seed,
            "agent_seed_namespace": self.agent_seed_namespace,
            "agent_seed": self.agent_seed,
            "configuration_record_sha256": self.configuration_record_sha256,
            "candidate_execution_binding_sha256": (
                self.candidate_execution_binding_sha256
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self.body_payload(),
            "cell_id": self.cell_id,
            "cell_body_sha256": self.cell_body_sha256,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentSchedule:
    """Detached immutable exact development schedule."""

    candidate_universe: DescriptorBinding
    configuration_plan: DescriptorBinding
    cumulative_reward_metric: DescriptorBinding
    seed_registry: ContentBinding
    qualification_manifest: ContentBinding
    candidate_execution_bindings_sha256: str
    retry_policy: RetryPolicyBinding
    candidate_order: tuple[str, ...]
    block_order: tuple[str, ...]
    cells: tuple[DevelopmentCell, ...]
    schedule_body_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        descriptor_bindings = (
            (
                self.candidate_universe,
                universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
                universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
            ),
            (
                self.configuration_plan,
                plan.CONFIGURATION_PLAN_SCHEMA_VERSION,
                plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256,
            ),
            (
                self.cumulative_reward_metric,
                protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION,
                protocol.CUMULATIVE_REWARD_METRIC_SHA256,
            ),
        )
        if any(
            type(binding) is not DescriptorBinding
            or binding.schema_version != schema
            or binding.sha256 != digest
            for binding, schema, digest in descriptor_bindings
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule descriptor binding drifted"
            )
        if (
            type(self.seed_registry) is not ContentBinding
            or self.seed_registry.schema_version != DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION
            or type(self.qualification_manifest) is not ContentBinding
            or type(self.retry_policy) is not RetryPolicyBinding
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule content binding type drifted"
            )
        _require_sha256(
            self.candidate_execution_bindings_sha256,
            "development schedule execution binding-set digest",
        )
        candidates = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
        if self.candidate_order != candidates:
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule candidate order drifted"
            )
        if (
            type(self.block_order) is not tuple
            or not 1 <= len(self.block_order) <= _MAX_BLOCKS
            or len(set(self.block_order)) != len(self.block_order)
            or any(_PORTABLE_ID_RE.fullmatch(item) is None for item in self.block_order)
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule block order drifted"
            )
        if (
            type(self.cells) is not tuple
            or len(self.cells) != len(self.block_order) * len(candidates)
            or any(type(cell) is not DevelopmentCell for cell in self.cells)
        ):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule cells are not the exact Cartesian shape"
            )
        for ordinal, cell in enumerate(self.cells):
            block_ordinal, candidate_ordinal = divmod(ordinal, len(candidates))
            if (
                cell.ordinal != ordinal
                or cell.block_ordinal != block_ordinal
                or cell.candidate_ordinal != candidate_ordinal
                or cell.block_id != self.block_order[block_ordinal]
                or cell.candidate_id != candidates[candidate_ordinal]
                or cell.configuration_record_sha256
                != _configuration_record_sha256(cell.candidate_id)
            ):
                raise ForagerMatchedV3DevelopmentScheduleError(
                    "development schedule cell order or configuration binding drifted"
                )
        body_digest = _body_sha256(
            self.body_payload(),
            label="development schedule body",
            maximum=_MAX_SCHEDULE_BYTES,
        )
        supplied_body = _require_sha256(
            self.schedule_body_sha256, "development schedule value body digest"
        )
        if not hmac.compare_digest(body_digest, supplied_body):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule value body digest does not replay"
            )
        raw = _canonical_json_bytes(
            self.to_payload(),
            label="development schedule",
            maximum=_MAX_SCHEDULE_BYTES,
        )
        supplied_file = _require_sha256(
            self.file_sha256, "development schedule value file digest"
        )
        if not hmac.compare_digest(_sha256(raw), supplied_file):
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development schedule value file digest does not replay"
            )

    def body_payload(self) -> dict[str, Any]:
        return {
            "schema_version": DEVELOPMENT_SCHEDULE_SCHEMA_VERSION,
            "classification": DEVELOPMENT_CLASSIFICATION,
            "stage": DEVELOPMENT_STAGE,
            "candidate_universe": self.candidate_universe.to_payload(),
            "configuration_plan": self.configuration_plan.to_payload(),
            "cumulative_reward_metric": self.cumulative_reward_metric.to_payload(),
            "seed_registry": self.seed_registry.to_payload(),
            "qualification_manifest": self.qualification_manifest.to_payload(),
            "candidate_execution_bindings_sha256": (
                self.candidate_execution_bindings_sha256
            ),
            "retry_policy": self.retry_policy.to_payload(),
            "ordering": DEVELOPMENT_ORDERING,
            "candidate_order": list(self.candidate_order),
            "block_order": list(self.block_order),
            "cells": [cell.to_payload() for cell in self.cells],
            "claims": dict(_CLAIMS),
            "limitations": list(_SCHEDULE_LIMITATIONS),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.body_payload(), "schedule_body_sha256": self.schedule_body_sha256}


def _content_binding(value: object, *, label: str) -> ContentBinding:
    payload = _require_object(value, label)
    _require_exact_keys(
        payload,
        frozenset({"schema_version", "file_sha256", "body_sha256"}),
        label,
    )
    return ContentBinding(
        schema_version=_require_schema(payload["schema_version"], f"{label} schema"),
        file_sha256=_require_sha256(payload["file_sha256"], f"{label} file digest"),
        body_sha256=_require_sha256(payload["body_sha256"], f"{label} body digest"),
    )


def _retry_binding(value: object, *, label: str) -> RetryPolicyBinding:
    payload = _require_object(value, label)
    _require_exact_keys(payload, frozenset({"schema_version", "sha256"}), label)
    return RetryPolicyBinding(
        schema_version=cast(str, payload["schema_version"]),
        sha256=_require_sha256(payload["sha256"], f"{label} digest"),
    )


def _require_claims(value: object, *, label: str) -> None:
    payload = _require_object(value, label)
    _require_exact_keys(payload, frozenset(_CLAIMS), label)
    if payload != _CLAIMS or any(type(item) is not bool or item for item in payload.values()):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must contain the exact all-false development claims"
        )


def _require_limitations(value: object, expected: tuple[str, ...], *, label: str) -> None:
    items = _require_list(value, label)
    if tuple(items) != expected or any(type(item) is not str for item in items):
        raise ForagerMatchedV3DevelopmentScheduleError(f"{label} drifted")


def _candidate_order(value: object, *, label: str) -> tuple[str, ...]:
    items = _require_list(value, label)
    result = tuple(items)
    expected = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    if result != expected or any(type(item) is not str for item in result):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} must be the exact frozen 25-candidate inferential order"
        )
    if len(result) != 25 or len(set(result)) != 25:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"{label} membership or uniqueness drifted"
        )
    return cast(tuple[str, ...], result)


def _parse_agent_seeds(value: object, *, block_ordinal: int) -> tuple[DevelopmentAgentSeed, ...]:
    items = _require_list(value, f"block {block_ordinal} agent_seeds")
    expected = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    if len(items) != len(expected):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"block {block_ordinal} must contain exactly 25 candidate seed records"
        )
    parsed: list[DevelopmentAgentSeed] = []
    for candidate_ordinal, (raw_item, candidate_id) in enumerate(zip(items, expected)):
        item = _require_object(
            raw_item,
            f"block {block_ordinal} agent seed {candidate_ordinal}",
        )
        _require_exact_keys(
            item,
            frozenset({"candidate_id", "namespace", "seed"}),
            f"block {block_ordinal} agent seed {candidate_ordinal}",
        )
        namespace = f"agent/{candidate_id}"
        if item["candidate_id"] != candidate_id or item["namespace"] != namespace:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"block {block_ordinal} agent seed order or namespace drifted"
            )
        parsed.append(
            DevelopmentAgentSeed(
                candidate_id=candidate_id,
                namespace=namespace,
                seed=_require_exact_int(
                    item["seed"],
                    f"block {block_ordinal} agent seed {candidate_id}",
                    minimum=0,
                    maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
                ),
            )
        )
    return tuple(parsed)


def _parse_blocks(value: object) -> tuple[DevelopmentSeedBlock, ...]:
    items = _require_list(value, "development seed blocks")
    if not 1 <= len(items) <= _MAX_BLOCKS:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"development registry must contain between 1 and {_MAX_BLOCKS} blocks"
        )
    parsed: list[DevelopmentSeedBlock] = []
    block_ids: set[str] = set()
    for ordinal, raw_item in enumerate(items):
        item = _require_object(raw_item, f"development seed block {ordinal}")
        _require_exact_keys(
            item,
            frozenset(
                {
                    "block_ordinal",
                    "block_id",
                    "derivation_payload_sha256",
                    "environment_seed",
                    "agent_seeds",
                }
            ),
            f"development seed block {ordinal}",
        )
        supplied_ordinal = _require_exact_int(
            item["block_ordinal"],
            f"development seed block {ordinal} ordinal",
            minimum=0,
            maximum=_MAX_BLOCKS - 1,
        )
        if supplied_ordinal != ordinal:
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development seed block ordinals must be exact contiguous array indices"
            )
        block_id = _require_portable_id(
            item["block_id"], f"development seed block {ordinal} ID"
        )
        if block_id in block_ids:
            raise ForagerMatchedV3DevelopmentScheduleError(
                "development seed block IDs must be unique"
            )
        block_ids.add(block_id)
        parsed.append(
            DevelopmentSeedBlock(
                block_ordinal=ordinal,
                block_id=block_id,
                derivation_payload_sha256=_require_sha256(
                    item["derivation_payload_sha256"],
                    f"development seed block {ordinal} derivation digest",
                ),
                environment_seed=_require_exact_int(
                    item["environment_seed"],
                    f"development seed block {ordinal} environment seed",
                    minimum=0,
                    maximum=protocol.TRIAL_BLOCK_SEED_MAXIMUM,
                ),
                agent_seeds=_parse_agent_seeds(
                    item["agent_seeds"], block_ordinal=ordinal
                ),
            )
        )
    return tuple(parsed)


_REGISTRY_KEYS: Final = frozenset(
    {
        "schema_version",
        "classification",
        "stage",
        "candidate_universe_sha256",
        "candidate_order",
        "derivation_schema_version",
        "derivation_domain",
        "provider_receipt",
        "blocks",
        "claims",
        "limitations",
        "registry_body_sha256",
    }
)


def parse_development_seed_registry(
    raw: bytes,
    *,
    expected_registry_file_sha256: str,
) -> DevelopmentSeedRegistry:
    """Parse externally issued development seeds with a caller-carried file hash.

    There is intentionally no corresponding registry builder or seed-issuance
    function in this module.
    """

    if type(raw) is not bytes:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry must be exact bytes"
        )
    expected_file = _require_sha256(
        expected_registry_file_sha256, "expected development registry file digest"
    )
    if not hmac.compare_digest(_sha256(raw), expected_file):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry differs from the caller-carried full-file digest"
        )
    payload = _decode_canonical_json(
        raw,
        label="development seed registry",
        maximum=_MAX_REGISTRY_BYTES,
    )
    _require_exact_keys(payload, _REGISTRY_KEYS, "development seed registry")
    if payload["schema_version"] != DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry schema drifted or is not a development schema"
        )
    if payload["classification"] != DEVELOPMENT_SEED_CLASSIFICATION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry classification drifted"
        )
    if payload["stage"] != DEVELOPMENT_STAGE:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry stage drifted"
        )
    if (
        payload["candidate_universe_sha256"]
        != universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry candidate-universe binding drifted"
        )
    candidate_order = _candidate_order(
        payload["candidate_order"], label="development registry candidate order"
    )
    if payload["derivation_schema_version"] != DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed derivation schema drifted or is held-out"
        )
    if payload["derivation_domain"] != DEVELOPMENT_SEED_DERIVATION_DOMAIN:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed derivation domain drifted or is held-out"
        )
    provider_receipt = _content_binding(
        payload["provider_receipt"], label="development seed provider receipt"
    )
    if provider_receipt.schema_version != DEVELOPMENT_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed provider receipt schema drifted or is held-out"
        )
    blocks = _parse_blocks(payload["blocks"])
    _require_claims(payload["claims"], label="development seed registry claims")
    _require_limitations(
        payload["limitations"],
        _REGISTRY_LIMITATIONS,
        label="development seed registry limitations",
    )
    supplied_body = _require_sha256(
        payload["registry_body_sha256"], "development seed registry body digest"
    )
    body = dict(payload)
    del body["registry_body_sha256"]
    replayed_body = _body_sha256(
        body,
        label="development seed registry body",
        maximum=_MAX_REGISTRY_BYTES,
    )
    if not hmac.compare_digest(replayed_body, supplied_body):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry body digest does not replay"
        )
    registry = DevelopmentSeedRegistry(
        candidate_order=candidate_order,
        provider_receipt=provider_receipt,
        blocks=blocks,
        registry_body_sha256=supplied_body,
        file_sha256=expected_file,
    )
    if canonical_development_seed_registry_bytes(registry) != raw:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry detached replay differs"
        )
    return registry


def canonical_development_seed_registry_bytes(registry: DevelopmentSeedRegistry) -> bytes:
    """Replay exact bytes from an already parsed registry without issuing one."""

    if type(registry) is not DevelopmentSeedRegistry:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "registry must be an exact parsed DevelopmentSeedRegistry"
        )
    raw = _canonical_json_bytes(
        registry.to_payload(),
        label="development seed registry",
        maximum=_MAX_REGISTRY_BYTES,
    )
    if (
        not hmac.compare_digest(_sha256(raw), registry.file_sha256)
        or not hmac.compare_digest(
            _body_sha256(
                registry.body_payload(),
                label="development seed registry body",
                maximum=_MAX_REGISTRY_BYTES,
            ),
            registry.registry_body_sha256,
        )
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development seed registry immutable identity drifted"
        )
    return raw


def _descriptor_binding(
    schema_version: str,
    sha256: str,
) -> DescriptorBinding:
    return DescriptorBinding(schema_version=schema_version, sha256=sha256)


def _binding_digest(binding: ContentBinding) -> str:
    return _sha256(
        _canonical_json_bytes(
            binding.to_payload(),
            label="candidate execution content binding",
            maximum=4096,
        )
    )


def _validated_execution_bindings(
    value: Mapping[str, ContentBinding],
) -> tuple[tuple[str, ContentBinding], ...]:
    if type(value) is not dict:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "candidate execution bindings must be a plain dict"
        )
    expected = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    if set(value) != set(expected) or any(type(key) is not str for key in value):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "candidate execution bindings must cover exactly the 25 inferential candidates"
        )
    result: list[tuple[str, ContentBinding]] = []
    for candidate_id in expected:
        binding = value[candidate_id]
        if type(binding) is not ContentBinding:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"execution binding for {candidate_id} must be an exact ContentBinding"
            )
        result.append((candidate_id, binding))
    return tuple(result)


def _execution_bindings_sha256(
    bindings: tuple[tuple[str, ContentBinding], ...],
) -> str:
    payload = [
        {
            "candidate_id": candidate_id,
            "binding": binding.to_payload(),
            "binding_sha256": _binding_digest(binding),
        }
        for candidate_id, binding in bindings
    ]
    return _sha256(
        _canonical_json_bytes(
            payload,
            label="candidate execution binding set",
            maximum=256 * 1024,
        )
    )


def _configuration_record_sha256(candidate_id: str) -> str:
    try:
        record = plan.configuration_record(candidate_id)
        raw = plan._canonical_bytes(record)
    except (RuntimeError, ValueError) as exc:
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"configuration-plan record replay failed for {candidate_id}"
        ) from exc
    if type(raw) is not bytes or not raw or raw.endswith(b"\n"):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"configuration-plan canonical record bytes drifted for {candidate_id}"
        )
    return _sha256(raw)


def _candidate_metadata() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {
        candidate.candidate_id: (
            candidate.analysis_role,
            candidate.development_selection_group,
        )
        for candidate in universe.matched_v3_development_candidates()
        if candidate.analysis_role == "inferential"
    }
    if tuple(result) != universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "v3 inferential candidate metadata order drifted"
        )
    return result


def _build_cell(
    *,
    block: DevelopmentSeedBlock,
    candidate_ordinal: int,
    agent_seed: DevelopmentAgentSeed,
    metadata: tuple[str, str],
    configuration_record_sha256: str,
    execution_binding_sha256: str,
) -> DevelopmentCell:
    ordinal = block.block_ordinal * 25 + candidate_ordinal
    body: dict[str, object] = {
        "schema_version": DEVELOPMENT_CELL_SCHEMA_VERSION,
        "ordinal": ordinal,
        "block_ordinal": block.block_ordinal,
        "candidate_ordinal": candidate_ordinal,
        "block_id": block.block_id,
        "candidate_id": agent_seed.candidate_id,
        "analysis_role": metadata[0],
        "development_selection_group": metadata[1],
        "derivation_payload_sha256": block.derivation_payload_sha256,
        "environment_seed": block.environment_seed,
        "agent_seed_namespace": agent_seed.namespace,
        "agent_seed": agent_seed.seed,
        "configuration_record_sha256": configuration_record_sha256,
        "candidate_execution_binding_sha256": execution_binding_sha256,
    }
    digest = _body_sha256(
        body,
        label=f"development cell {ordinal} body",
        maximum=64 * 1024,
    )
    return DevelopmentCell(
        cell_id=f"cell_{ordinal:016x}_{digest}",
        ordinal=ordinal,
        block_ordinal=block.block_ordinal,
        candidate_ordinal=candidate_ordinal,
        block_id=block.block_id,
        candidate_id=agent_seed.candidate_id,
        analysis_role=metadata[0],
        development_selection_group=metadata[1],
        derivation_payload_sha256=block.derivation_payload_sha256,
        environment_seed=block.environment_seed,
        agent_seed_namespace=agent_seed.namespace,
        agent_seed=agent_seed.seed,
        configuration_record_sha256=configuration_record_sha256,
        candidate_execution_binding_sha256=execution_binding_sha256,
        cell_body_sha256=digest,
    )


def build_development_schedule(
    *,
    seed_registry: DevelopmentSeedRegistry,
    qualification_manifest: ContentBinding,
    candidate_execution_bindings: Mapping[str, ContentBinding],
    retry_policy: RetryPolicyBinding,
) -> DevelopmentSchedule:
    """Build the exact nonpromoting block-major development schedule."""

    if type(seed_registry) is not DevelopmentSeedRegistry:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "seed_registry must be an exact parsed DevelopmentSeedRegistry"
        )
    canonical_development_seed_registry_bytes(seed_registry)
    if type(qualification_manifest) is not ContentBinding:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "qualification_manifest must be an exact ContentBinding"
        )
    if type(retry_policy) is not RetryPolicyBinding:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "retry_policy must be an exact RetryPolicyBinding"
        )
    bindings = _validated_execution_bindings(candidate_execution_bindings)
    execution_digest_by_id = {
        candidate_id: _binding_digest(binding) for candidate_id, binding in bindings
    }
    configuration_digest_by_id = {
        candidate_id: _configuration_record_sha256(candidate_id)
        for candidate_id in seed_registry.candidate_order
    }
    metadata_by_id = _candidate_metadata()
    cells = tuple(
        _build_cell(
            block=block,
            candidate_ordinal=candidate_ordinal,
            agent_seed=agent_seed,
            metadata=metadata_by_id[agent_seed.candidate_id],
            configuration_record_sha256=configuration_digest_by_id[
                agent_seed.candidate_id
            ],
            execution_binding_sha256=execution_digest_by_id[agent_seed.candidate_id],
        )
        for block in seed_registry.blocks
        for candidate_ordinal, agent_seed in enumerate(block.agent_seeds)
    )
    universe_binding = _descriptor_binding(
        universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
        universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
    )
    plan_binding = _descriptor_binding(
        plan.CONFIGURATION_PLAN_SCHEMA_VERSION,
        plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256,
    )
    metric_binding = _descriptor_binding(
        protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION,
        protocol.CUMULATIVE_REWARD_METRIC_SHA256,
    )
    registry_binding = ContentBinding(
        schema_version=DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION,
        file_sha256=seed_registry.file_sha256,
        body_sha256=seed_registry.registry_body_sha256,
    )
    body: dict[str, Any] = {
        "schema_version": DEVELOPMENT_SCHEDULE_SCHEMA_VERSION,
        "classification": DEVELOPMENT_CLASSIFICATION,
        "stage": DEVELOPMENT_STAGE,
        "candidate_universe": universe_binding.to_payload(),
        "configuration_plan": plan_binding.to_payload(),
        "cumulative_reward_metric": metric_binding.to_payload(),
        "seed_registry": registry_binding.to_payload(),
        "qualification_manifest": qualification_manifest.to_payload(),
        "candidate_execution_bindings_sha256": _execution_bindings_sha256(bindings),
        "retry_policy": retry_policy.to_payload(),
        "ordering": DEVELOPMENT_ORDERING,
        "candidate_order": list(seed_registry.candidate_order),
        "block_order": [block.block_id for block in seed_registry.blocks],
        "cells": [cell.to_payload() for cell in cells],
        "claims": dict(_CLAIMS),
        "limitations": list(_SCHEDULE_LIMITATIONS),
    }
    schedule_body_sha256 = _body_sha256(
        body,
        label="development schedule body",
        maximum=_MAX_SCHEDULE_BYTES,
    )
    payload = {**body, "schedule_body_sha256": schedule_body_sha256}
    raw = _canonical_json_bytes(
        payload,
        label="development schedule",
        maximum=_MAX_SCHEDULE_BYTES,
    )
    return DevelopmentSchedule(
        candidate_universe=universe_binding,
        configuration_plan=plan_binding,
        cumulative_reward_metric=metric_binding,
        seed_registry=registry_binding,
        qualification_manifest=qualification_manifest,
        candidate_execution_bindings_sha256=cast(
            str, body["candidate_execution_bindings_sha256"]
        ),
        retry_policy=retry_policy,
        candidate_order=seed_registry.candidate_order,
        block_order=tuple(block.block_id for block in seed_registry.blocks),
        cells=cells,
        schedule_body_sha256=schedule_body_sha256,
        file_sha256=_sha256(raw),
    )


_CELL_KEYS: Final = frozenset(
    {
        "schema_version",
        "cell_id",
        "ordinal",
        "block_ordinal",
        "candidate_ordinal",
        "block_id",
        "candidate_id",
        "analysis_role",
        "development_selection_group",
        "derivation_payload_sha256",
        "environment_seed",
        "agent_seed_namespace",
        "agent_seed",
        "configuration_record_sha256",
        "candidate_execution_binding_sha256",
        "cell_body_sha256",
    }
)


def _parse_cell(value: object, *, expected: DevelopmentCell) -> DevelopmentCell:
    payload = _require_object(value, f"development cell {expected.ordinal}")
    _require_exact_keys(payload, _CELL_KEYS, f"development cell {expected.ordinal}")
    supplied_body = _require_sha256(
        payload["cell_body_sha256"], f"development cell {expected.ordinal} body digest"
    )
    body = dict(payload)
    del body["cell_body_sha256"]
    cell_id = body.pop("cell_id")
    replayed = _body_sha256(
        body,
        label=f"development cell {expected.ordinal} body",
        maximum=64 * 1024,
    )
    if (
        not hmac.compare_digest(replayed, supplied_body)
        or cell_id != f"cell_{expected.ordinal:016x}_{supplied_body}"
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"development cell {expected.ordinal} identity does not replay"
        )
    if payload != expected.to_payload():
        raise ForagerMatchedV3DevelopmentScheduleError(
            f"development cell {expected.ordinal} differs from exact schedule replay"
        )
    return expected


_SCHEDULE_KEYS: Final = frozenset(
    {
        "schema_version",
        "classification",
        "stage",
        "candidate_universe",
        "configuration_plan",
        "cumulative_reward_metric",
        "seed_registry",
        "qualification_manifest",
        "candidate_execution_bindings_sha256",
        "retry_policy",
        "ordering",
        "candidate_order",
        "block_order",
        "cells",
        "claims",
        "limitations",
        "schedule_body_sha256",
    }
)


def parse_development_schedule(
    raw: bytes,
    *,
    expected_schedule_file_sha256: str,
    seed_registry: DevelopmentSeedRegistry,
    qualification_manifest: ContentBinding,
    candidate_execution_bindings: Mapping[str, ContentBinding],
    retry_policy: RetryPolicyBinding,
) -> DevelopmentSchedule:
    """Parse and exactly replay a schedule against every caller-carried binding."""

    if type(raw) is not bytes:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule must be exact bytes"
        )
    expected_file = _require_sha256(
        expected_schedule_file_sha256, "expected development schedule file digest"
    )
    if not hmac.compare_digest(_sha256(raw), expected_file):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule differs from the caller-carried full-file digest"
        )
    payload = _decode_canonical_json(
        raw,
        label="development schedule",
        maximum=_MAX_SCHEDULE_BYTES,
    )
    _require_exact_keys(payload, _SCHEDULE_KEYS, "development schedule")
    if payload["schema_version"] != DEVELOPMENT_SCHEDULE_SCHEMA_VERSION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule schema drifted"
        )
    if payload["classification"] != DEVELOPMENT_CLASSIFICATION:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule classification drifted"
        )
    if payload["stage"] != DEVELOPMENT_STAGE:
        raise ForagerMatchedV3DevelopmentScheduleError("development schedule stage drifted")
    if payload["ordering"] != DEVELOPMENT_ORDERING:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule ordering drifted"
        )
    _candidate_order(payload["candidate_order"], label="development schedule candidate order")
    _require_claims(payload["claims"], label="development schedule claims")
    _require_limitations(
        payload["limitations"],
        _SCHEDULE_LIMITATIONS,
        label="development schedule limitations",
    )
    supplied_body = _require_sha256(
        payload["schedule_body_sha256"], "development schedule body digest"
    )
    body = dict(payload)
    del body["schedule_body_sha256"]
    if not hmac.compare_digest(
        _body_sha256(
            body,
            label="development schedule body",
            maximum=_MAX_SCHEDULE_BYTES,
        ),
        supplied_body,
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule body digest does not replay"
        )

    expected = build_development_schedule(
        seed_registry=seed_registry,
        qualification_manifest=qualification_manifest,
        candidate_execution_bindings=candidate_execution_bindings,
        retry_policy=retry_policy,
    )
    if (
        expected.file_sha256 != expected_file
        or expected.schedule_body_sha256 != supplied_body
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule binding set differs from exact replay"
        )
    # Validate nested types before the final exact payload comparison so boolean
    # integer substitutions and malformed cell self-identities fail explicitly.
    for key, expected_binding in (
        ("candidate_universe", expected.candidate_universe.to_payload()),
        ("configuration_plan", expected.configuration_plan.to_payload()),
        ("cumulative_reward_metric", expected.cumulative_reward_metric.to_payload()),
    ):
        supplied_binding = _require_object(payload[key], f"schedule {key} binding")
        _require_exact_keys(
            supplied_binding, frozenset({"schema_version", "sha256"}), f"schedule {key} binding"
        )
        if supplied_binding != expected_binding:
            raise ForagerMatchedV3DevelopmentScheduleError(
                f"development schedule {key} binding drifted"
            )
    supplied_registry = _content_binding(
        payload["seed_registry"], label="schedule seed registry binding"
    )
    supplied_qualification = _content_binding(
        payload["qualification_manifest"], label="schedule qualification binding"
    )
    supplied_retry = _retry_binding(payload["retry_policy"], label="schedule retry policy")
    if (
        supplied_registry != expected.seed_registry
        or supplied_qualification != expected.qualification_manifest
        or supplied_retry != expected.retry_policy
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule content binding drifted"
        )
    _require_sha256(
        payload["candidate_execution_bindings_sha256"],
        "candidate execution binding-set digest",
    )
    block_order_raw = _require_list(payload["block_order"], "schedule block order")
    if tuple(block_order_raw) != expected.block_order or any(
        type(item) is not str for item in block_order_raw
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule block order drifted"
        )
    raw_cells = _require_list(payload["cells"], "development schedule cells")
    if len(raw_cells) != len(expected.cells):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule is not the exact block/candidate Cartesian product"
        )
    for raw_cell, expected_cell in zip(raw_cells, expected.cells):
        _parse_cell(raw_cell, expected=expected_cell)
    if payload != expected.to_payload() or canonical_development_schedule_bytes(expected) != raw:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule differs from exact detached replay"
        )
    return expected


def canonical_development_schedule_bytes(schedule: DevelopmentSchedule) -> bytes:
    """Return exact canonical bytes after checking immutable schedule identities."""

    if type(schedule) is not DevelopmentSchedule:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "schedule must be an exact DevelopmentSchedule"
        )
    raw = _canonical_json_bytes(
        schedule.to_payload(),
        label="development schedule",
        maximum=_MAX_SCHEDULE_BYTES,
    )
    if (
        not hmac.compare_digest(_sha256(raw), schedule.file_sha256)
        or not hmac.compare_digest(
            _body_sha256(
                schedule.body_payload(),
                label="development schedule body",
                maximum=_MAX_SCHEDULE_BYTES,
            ),
            schedule.schedule_body_sha256,
        )
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "development schedule immutable identity drifted"
        )
    return raw


def scheduled_cell(
    schedule: DevelopmentSchedule,
    cell_id: str,
) -> DevelopmentCell:
    """Return one cell by its exact durable identity after replaying that identity."""

    if type(schedule) is not DevelopmentSchedule:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "schedule must be an exact DevelopmentSchedule"
        )
    if type(cell_id) is not str:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "scheduled cell ID must be an exact string"
        )
    match = _CELL_ID_RE.fullmatch(cell_id)
    if match is None:
        raise ForagerMatchedV3DevelopmentScheduleError(
            "scheduled cell ID does not have the exact canonical form"
        )
    ordinal = int(match.group(1), 16)
    if not 0 <= ordinal < len(schedule.cells):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "scheduled cell ID names an ordinal outside the exact Cartesian product"
        )
    cell = schedule.cells[ordinal]
    replayed_body_sha256 = _body_sha256(
        cell.body_payload(),
        label=f"scheduled cell {ordinal} body",
        maximum=64 * 1024,
    )
    if (
        cell.ordinal != ordinal
        or cell.cell_id != cell_id
        or cell.cell_body_sha256 != match.group(2)
        or not hmac.compare_digest(replayed_body_sha256, cell.cell_body_sha256)
        or cell.cell_id != f"cell_{ordinal:016x}_{replayed_body_sha256}"
    ):
        raise ForagerMatchedV3DevelopmentScheduleError(
            "scheduled cell ID, body digest, or ordinal does not replay"
        )
    return cell


__all__ = [
    "ContentBinding",
    "DEVELOPMENT_CELL_SCHEMA_VERSION",
    "DEVELOPMENT_RETRY_POLICY_SCHEMA_VERSION",
    "DEVELOPMENT_SCHEDULE_SCHEMA_VERSION",
    "DEVELOPMENT_SEED_DERIVATION_DOMAIN",
    "DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION",
    "DEVELOPMENT_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION",
    "DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION",
    "DescriptorBinding",
    "DevelopmentAgentSeed",
    "DevelopmentCell",
    "DevelopmentSchedule",
    "DevelopmentSeedBlock",
    "DevelopmentSeedRegistry",
    "ForagerMatchedV3DevelopmentScheduleError",
    "RetryPolicyBinding",
    "build_development_schedule",
    "canonical_development_schedule_bytes",
    "canonical_development_seed_registry_bytes",
    "parse_development_schedule",
    "parse_development_seed_registry",
    "scheduled_cell",
]
