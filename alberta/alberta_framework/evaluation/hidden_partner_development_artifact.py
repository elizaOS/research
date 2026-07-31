"""Strict nonpromoting artifact for hidden-partner development runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import cast

from alberta_framework.evaluation.hidden_partner_development import (
    HIDDEN_PARTNER_CONDITIONS,
    ROBUSTNESS_SEED_NAMESPACE,
    TUNING_SEED_NAMESPACE,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerRunSummary,
    HiddenPartnerSeedPair,
    aggregate_hidden_partner_development,
    derive_hidden_partner_seed_pairs,
    hidden_partner_run_summary_from_dict,
)

HIDDEN_PARTNER_DEVELOPMENT_ARTIFACT_SCHEMA = "alberta.hidden-partner-development-artifact.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATHS = (
    Path("alberta_framework/core/average_reward.py"),
    Path("alberta_framework/core/behavior_model.py"),
    Path("alberta_framework/core/feature_bank_router.py"),
    Path("alberta_framework/core/integrated_hidden_partner.py"),
    Path("alberta_framework/core/interaction_features.py"),
    Path("alberta_framework/core/joint_partner_world.py"),
    Path("alberta_framework/core/state_builder.py"),
    Path("alberta_framework/evaluation/hidden_partner_development.py"),
    Path("alberta_framework/evaluation/hidden_partner_development_artifact.py"),
    Path("alberta_framework/evaluation/hidden_partner_development_cli.py"),
    Path("alberta_framework/streams/hidden_partner_mapping.py"),
)

RunRole = str
_INTERPRETATION = (
    "development robustness and causal-ablation evidence only; this "
    "artifact is structurally unable to promote a scientific claim"
)


@dataclass(frozen=True)
class HiddenPartnerDevelopmentArtifactValidation:
    """Structural validity and separately reported development-check outcome."""

    valid: bool
    development_checks_passed: bool
    errors: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash every source file that can affect a stored development result."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


def build_hidden_partner_development_artifact(
    protocol: HiddenPartnerDevelopmentProtocol,
    role: RunRole,
    summaries: Sequence[HiddenPartnerRunSummary],
    *,
    root: Path = REPO_ROOT,
    operational_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build a digested development-only record from completed paired lives."""
    if role not in {"tuning-replay", "robustness"}:
        raise ValueError("role must be 'tuning-replay' or 'robustness'")
    if not summaries:
        raise ValueError("at least one run summary is required")
    namespace = TUNING_SEED_NAMESPACE if role == "tuning-replay" else ROBUSTNESS_SEED_NAMESPACE
    seed_pairs = tuple(
        sorted(
            {summary.seed_pair for summary in summaries if summary.condition == "full"},
            key=lambda pair: pair.index,
        )
    )
    if not seed_pairs or any(pair.namespace != namespace for pair in seed_pairs):
        raise ValueError("summary seed namespace does not match the declared role")
    expected_pairs = derive_hidden_partner_seed_pairs(namespace, len(seed_pairs))
    if seed_pairs != expected_pairs:
        raise ValueError("summary seeds do not match the exact namespace derivation")
    aggregate = aggregate_hidden_partner_development(summaries)
    scientific_payload: dict[str, object] = {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "run_role": role,
        "seed_namespace": namespace,
        "seed_pairs": [pair.to_dict() for pair in seed_pairs],
        "protocol": protocol.to_config(),
        "condition_order": [condition.name for condition in HIDDEN_PARTNER_CONDITIONS],
        "source_sha256": source_snapshot(root),
        "aggregate": aggregate,
        "runs": [summary.to_dict() for summary in summaries],
        "interpretation": _INTERPRETATION,
    }
    return {
        "schema_version": HIDDEN_PARTNER_DEVELOPMENT_ARTIFACT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "scientific_payload": scientific_payload,
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": _canonical_sha256(scientific_payload),
        },
        "operational_metadata": dict(operational_metadata),
    }


def hidden_partner_development_artifact_json(
    artifact: Mapping[str, object],
) -> str:
    """Serialize strict pretty JSON with no non-finite numeric extension."""
    return (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_hidden_partner_development_artifact(path: Path) -> dict[str, object]:
    """Load a strict JSON object while rejecting duplicate keys and constants."""

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("development artifact must contain one JSON object")
    return parsed


def validate_hidden_partner_development_artifact(
    artifact: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> HiddenPartnerDevelopmentArtifactValidation:
    """Fail closed on schema, role, seed, digest, source, or shape drift."""
    errors: list[str] = []
    expected_top = {
        "schema_version",
        "development_only",
        "scientific_promotion_allowed",
        "scientific_payload",
        "scientific_digest",
        "operational_metadata",
    }
    if set(artifact) != expected_top:
        errors.append("top-level artifact fields do not match the v1 schema")
    if artifact.get("schema_version") != HIDDEN_PARTNER_DEVELOPMENT_ARTIFACT_SCHEMA:
        errors.append("artifact schema version is unsupported")
    if artifact.get("development_only") is not True:
        errors.append("artifact must remain development-only")
    if artifact.get("scientific_promotion_allowed") is not False:
        errors.append("artifact must forbid scientific promotion")

    payload = artifact.get("scientific_payload")
    if not isinstance(payload, Mapping):
        errors.append("scientific_payload must be an object")
        payload = {}
    expected_payload = {
        "development_only",
        "scientific_promotion_allowed",
        "run_role",
        "seed_namespace",
        "seed_pairs",
        "protocol",
        "condition_order",
        "source_sha256",
        "aggregate",
        "runs",
        "interpretation",
    }
    if set(payload) != expected_payload:
        errors.append("scientific_payload fields do not match the v1 schema")
    if payload.get("development_only") is not True:
        errors.append("scientific payload must remain development-only")
    if payload.get("scientific_promotion_allowed") is not False:
        errors.append("scientific payload must forbid promotion")
    if payload.get("interpretation") != _INTERPRETATION:
        errors.append("scientific payload interpretation is not the exact v1 text")

    operational = artifact.get("operational_metadata")
    expected_operational = {
        "argv",
        "generated_at_utc",
        "jax_backend",
        "jax_device_count",
        "jax_version",
        "platform",
        "python_version",
        "wall_seconds",
    }
    if not isinstance(operational, Mapping) or set(operational) != expected_operational:
        errors.append("operational_metadata fields do not match the v1 schema")
    else:
        argv = operational.get("argv")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            errors.append("operational_metadata.argv must be an array of strings")
        for field in ("jax_backend", "jax_version", "platform", "python_version"):
            value = operational.get(field)
            if not isinstance(value, str) or not value:
                errors.append(f"operational_metadata.{field} must be a non-empty string")
        device_count = operational.get("jax_device_count")
        if type(device_count) is not int or device_count <= 0:
            errors.append("operational_metadata.jax_device_count must be positive")
        wall_seconds = operational.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, Real)
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            errors.append("operational_metadata.wall_seconds must be finite and non-negative")
        generated = operational.get("generated_at_utc")
        if not isinstance(generated, str):
            errors.append("operational_metadata.generated_at_utc must be an ISO timestamp")
        else:
            try:
                timestamp = datetime.fromisoformat(generated)
            except ValueError:
                errors.append("operational_metadata.generated_at_utc must be an ISO timestamp")
            else:
                utc_offset = timestamp.utcoffset()
                if utc_offset is None or utc_offset.total_seconds() != 0:
                    errors.append("operational_metadata.generated_at_utc must be UTC")

    digest = artifact.get("scientific_digest")
    if not isinstance(digest, Mapping) or set(digest) != {
        "algorithm",
        "scope",
        "sha256",
    }:
        errors.append("scientific_digest fields are invalid")
    else:
        if digest.get("algorithm") != "sha256":
            errors.append("scientific digest algorithm must be sha256")
        if digest.get("scope") != "$.scientific_payload":
            errors.append("scientific digest scope is invalid")
        if digest.get("sha256") != _canonical_sha256(payload):
            errors.append("scientific payload digest mismatch")

    protocol_payload = payload.get("protocol")
    reconstructed_protocol: HiddenPartnerDevelopmentProtocol | None = None
    if not isinstance(protocol_payload, Mapping):
        errors.append("protocol must be an object")
    else:
        try:
            reconstructed_protocol = HiddenPartnerDevelopmentProtocol.from_config(protocol_payload)
        except (TypeError, ValueError) as error:
            errors.append(f"protocol is invalid: {error}")
        else:
            if reconstructed_protocol.to_config() != (
                HiddenPartnerDevelopmentProtocol().to_config()
            ):
                errors.append("protocol must be the exact default v1 development protocol")

    role = payload.get("run_role")
    expected_namespace: str | None = None
    if role == "tuning-replay":
        expected_namespace = TUNING_SEED_NAMESPACE
    elif role == "robustness":
        expected_namespace = ROBUSTNESS_SEED_NAMESPACE
    else:
        errors.append("run_role must be tuning-replay or robustness")
    if expected_namespace is not None and payload.get("seed_namespace") != expected_namespace:
        errors.append("seed namespace does not match the declared run role")

    seed_pairs = payload.get("seed_pairs")
    expected_seed_objects: tuple[HiddenPartnerSeedPair, ...] = ()
    if not isinstance(seed_pairs, list) or len(seed_pairs) < 2:
        errors.append("seed_pairs must contain at least two development pairs")
        seed_pairs = []
    elif expected_namespace is not None:
        expected_seed_objects = derive_hidden_partner_seed_pairs(
            expected_namespace,
            len(seed_pairs),
        )
        expected_seeds = [pair.to_dict() for pair in expected_seed_objects]
        if seed_pairs != expected_seeds:
            errors.append("seed pairs do not match exact namespace derivation")

    condition_order = [condition.name for condition in HIDDEN_PARTNER_CONDITIONS]
    if payload.get("condition_order") != condition_order:
        errors.append("condition order does not match the exact v1 table")
    runs = payload.get("runs")
    expected_run_count = len(seed_pairs) * len(condition_order)
    reconstructed_summaries: list[HiddenPartnerRunSummary] = []
    if not isinstance(runs, list) or len(runs) != expected_run_count:
        errors.append(f"runs must contain exactly {expected_run_count} paired condition records")
    else:
        for index, run in enumerate(runs):
            if not isinstance(run, Mapping):
                errors.append(f"runs[{index}] must be an object")
                continue
            try:
                summary = hidden_partner_run_summary_from_dict(run)
            except (TypeError, ValueError) as error:
                errors.append(f"runs[{index}] is invalid: {error}")
                continue
            reconstructed_summaries.append(summary)
            seed_index = index // len(condition_order)
            condition_index = index % len(condition_order)
            if summary.condition != condition_order[condition_index]:
                errors.append(f"runs[{index}] condition ordering is invalid")
            if (
                seed_index >= len(expected_seed_objects)
                or summary.seed_pair != expected_seed_objects[seed_index]
            ):
                errors.append(f"runs[{index}] seed pairing is invalid")

        if len(reconstructed_summaries) == expected_run_count:
            for seed_index in range(len(seed_pairs)):
                start = seed_index * len(condition_order)
                seed_summaries = reconstructed_summaries[start : start + len(condition_order)]
                if len({summary.cycle_steps for summary in seed_summaries}) != 1:
                    errors.append(f"seed {seed_index} conditions have unequal cycle lengths")
                if len({summary.segment_lengths for summary in seed_summaries}) != 1:
                    errors.append(f"seed {seed_index} conditions have unequal segment schedules")
                if len({summary.initial_state_nbytes for summary in seed_summaries}) != 1:
                    errors.append(f"seed {seed_index} conditions have unequal state bytes")
                if reconstructed_protocol is not None:
                    for summary in seed_summaries:
                        for segment_index, length in enumerate(summary.segment_lengths):
                            base = reconstructed_protocol.environment.base_segment_lengths[
                                segment_index
                            ]
                            jitter = reconstructed_protocol.environment.jitter_radius
                            if not base - jitter <= length <= base + jitter:
                                errors.append(
                                    f"runs for seed {seed_index} contain an impossible "
                                    f"segment {segment_index} length"
                                )

    expected_sources: Mapping[str, str] | None = None
    try:
        expected_sources = source_snapshot(root)
    except OSError as error:
        errors.append(f"cannot hash current source snapshot: {error}")
    if expected_sources is not None and payload.get("source_sha256") != expected_sources:
        errors.append("source hashes do not match the current pinned development sources")

    aggregate = payload.get("aggregate")
    development_checks_passed = False
    if not isinstance(aggregate, Mapping):
        errors.append("aggregate must be an object")
    else:
        checks = aggregate.get("development_checks")
        if not isinstance(checks, Mapping) or not isinstance(
            checks.get("all_passed"),
            bool,
        ):
            errors.append("aggregate development checks are invalid")
        else:
            development_checks_passed = cast(bool, checks["all_passed"])
        if aggregate.get("scientific_promotion_allowed") is not False:
            errors.append("aggregate must forbid scientific promotion")
        if len(reconstructed_summaries) == expected_run_count:
            try:
                reconstructed_aggregate = aggregate_hidden_partner_development(
                    reconstructed_summaries
                )
            except (TypeError, ValueError) as error:
                errors.append(f"run aggregate cannot be reconstructed: {error}")
            else:
                if _canonical_json_bytes(aggregate) != _canonical_json_bytes(
                    reconstructed_aggregate
                ):
                    errors.append("aggregate does not exactly reconstruct from run summaries")

    return HiddenPartnerDevelopmentArtifactValidation(
        valid=not errors,
        development_checks_passed=development_checks_passed,
        errors=tuple(errors),
    )


__all__ = [
    "HIDDEN_PARTNER_DEVELOPMENT_ARTIFACT_SCHEMA",
    "HiddenPartnerDevelopmentArtifactValidation",
    "REPO_ROOT",
    "SOURCE_PATHS",
    "build_hidden_partner_development_artifact",
    "hidden_partner_development_artifact_json",
    "load_hidden_partner_development_artifact",
    "source_snapshot",
    "validate_hidden_partner_development_artifact",
]
