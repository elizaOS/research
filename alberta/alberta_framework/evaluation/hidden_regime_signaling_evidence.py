# mypy: disable-error-code="call-arg"
"""DRAFT fail-closed machinery for bounded hidden-regime signaling evidence.

This module is deliberately additive to the development evaluator.  It fixes a
candidate, schedule, condition order, seed-derivation rule, preliminary gates,
and compact primitive record before any protected held-out seed is derived or
executed.  Merely importing this module never derives a seed and never runs a
learner.

An accepted artifact would support only a narrow claim: in this finite
three-convention-plus-scratch world, two separately parameterized online
learners retain recurring conventions, replace one stale convention, and
outperform matched interventions under a fixed 552-byte dyad state budget.  It
would not establish unbounded retention, cross-task generality, an independent
replication, or completion of the Alberta Plan.

Scientific execution is disabled while a resource-matched factorial and
misaligned-boundary/symbol protocol replace the obsolete six-condition draft.
The protected namespace cannot be derived through this module in this state.

The SHA-256 digests provide integrity, not authorship or trusted chronology.
The preregistration timestamp is operator supplied and has no external time
anchor.  Exact rerun is intentionally scoped to the same JAX/XLA backend and
runtime; source and dependency hashes make that limitation explicit rather
than pretending that floating-point traces are cross-backend portable.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_signaling_development import (
    BENEFICIARY_FROZEN,
    CONSTANT_CHANNEL,
    HELPER_FROZEN,
    SELECTIVE_FULL,
    SHUFFLED_CHANNEL,
    WRITABLE_LRU,
    HiddenRegimeDevelopmentConfig,
    HiddenRegimePrimitiveTrace,
    HiddenRegimeRunResult,
    HiddenRegimeSeedPair,
    run_hidden_regime_condition,
    validate_hidden_regime_run_result,
)
from alberta_framework.streams.hidden_regime_signaling import HiddenRegimeWorldConfig

PREREGISTRATION_SCHEMA = "alberta.hidden-regime-signaling.preregistration.v1"
EVIDENCE_ARTIFACT_SCHEMA = "alberta.hidden-regime-signaling.evidence.v1"
COMPACT_PRIMITIVES_SCHEMA = "alberta.hidden-regime-signaling.compact-primitives.v1"
SEED_DERIVATION_SCHEMA = "alberta.hidden-regime-signaling.sha256-seeds.v1"

# This is a declaration only.  No module-level expression derives this namespace.
RESERVED_HELDOUT_NAMESPACE = "hidden-regime-signaling-v1-heldout-a-v1"
RESERVED_HELDOUT_NAMESPACE_EXECUTED = False
HELDOUT_SEED_COUNT = 30
MANUAL_TEST_NAMESPACE_PREFIX = "manual-test-hidden-regime-"
PROTOCOL_STATUS = "draft_execution_disabled_pending_factorial_boundary_protocol"
PROTECTED_EXECUTION_ENABLED = False

FROZEN_CONDITION_ORDER: tuple[str, ...] = (
    SELECTIVE_FULL,
    WRITABLE_LRU,
    HELPER_FROZEN,
    BENEFICIARY_FROZEN,
    CONSTANT_CHANNEL,
    SHUFFLED_CHANNEL,
)

# The frozen schedule is the development schedule expanded by the 16-step slot
# lease (64/72/56 -> 1024/1152/896): regimes 0 (A) and 1 (B) recur throughout
# to probe retention under interleaving, regime 2 (C-old) is permanently
# superseded by the incompatible regime 3 (C-new) to probe stale-convention
# replacement, and regime 4 (D-transient) appears exactly once and stays at 16
# steps -- one lease -- as a distraction that must not evict a durable
# convention.  See ``streams/hidden_regime_signaling.py`` for the base tables.
FROZEN_SEGMENT_LENGTHS: tuple[int, ...] = (
    1024,
    1152,
    896,
    1024,
    1152,
    1024,
    896,
    1152,
    1024,
    1024,
    896,
    1152,
    1024,
    16,
    1152,
    896,
    1024,
)
FROZEN_SEGMENT_REGIMES: tuple[int, ...] = (
    0,
    1,
    0,
    2,
    1,
    0,
    2,
    0,
    1,
    3,
    0,
    1,
    3,
    4,
    0,
    1,
    3,
)
FROZEN_REGIME_PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 2, 0),
    (1, 0, 2),
    (2, 1, 0),
    (0, 2, 1),
)
FROZEN_METRIC_WINDOW = 128
FROZEN_NUM_STEPS = 16_528
EXPECTED_DYAD_STATE_BYTES = 552

# These values were frozen from the disclosed consumed calibration audit before
# any protected seed was derived.  Changing any value invalidates a subsequently
# emitted plan through both exact-value and source-hash checks.
FROZEN_THRESHOLDS: dict[str, int | float] = {
    "required_seed_count": 30,
    "required_strict_run_validation_count": 180,
    "minimum_selective_mean_reward": 0.80,
    "minimum_selective_per_seed_reward": 0.78,
    "minimum_selective_per_seed_reward_pass_count": 27,
    "minimum_selective_recurrence_entry_reward": 0.70,
    "minimum_selective_per_seed_recurrence_entry_reward": 0.65,
    "minimum_selective_per_seed_recurrence_pass_count": 27,
    "minimum_mean_reward_delta_vs_writable_lru": 0.005,
    "minimum_recurrence_delta_vs_writable_lru": 0.04,
    "paired_one_sided_confidence": 0.95,
    "paired_t_degrees_freedom": 29,
    "paired_t_critical": 1.6991270265334972,
    "minimum_reward_pairwise_win_count": 21,
    "minimum_recurrence_pairwise_win_count": 24,
    "maximum_disabled_control_mean_reward": 0.40,
    "maximum_disabled_control_recurrence_entry_reward": 0.40,
    "required_disabled_control_durable_commit_count": 0,
    "minimum_strict_lifecycle_pass_count": 27,
    "required_dyad_state_bytes": EXPECTED_DYAD_STATE_BYTES,
}

CONSUMED_CALIBRATION_DISCLOSURE = (
    "candidate and gates use only 30 consumed manual development pairs: selective "
    "mean reward 0.845515, standard deviation 0.012124, minimum 0.808507, and "
    "recurrence entry 0.754796, with 29/30 exact four-commit/one-replacement lives; "
    "writable/LRU mean reward 0.835431, standard deviation 0.012670, minimum "
    "0.807297, and recurrence entry 0.672114, with 6/30 exact lifecycle lives; "
    "helper-frozen, beneficiary-frozen, constant-channel, and shuffled-channel "
    "controls were 0.332--0.336 with zero commits. These post-selection data are "
    "nonpromoting and are never reused as held-out evidence"
)
CLAIM_SCOPE = (
    "bounded fixed-schedule hidden-regime co-learning with three durable conventions, "
    "one scratch slot, two physically separate learners, ordinary local experience, "
    "and a fixed 552-byte joint persistent state"
)
RETENTION_LIMITATION = (
    "three durable slots cannot remember an unbounded number of incompatible "
    "conventions; acceptance is not general continual-learning or Alberta Plan completion"
)
TIME_ATTESTATION_LIMITATION = (
    "the operator-supplied preregistration timestamp is not externally time-stamped; "
    "the digest proves byte identity after creation but not trusted chronology or authorship"
)
REPLAY_SCOPE = (
    "deterministic rerun is source-bound and exact only on the same JAX/XLA backend, "
    "jaxlib/runtime, device family, and floating-point implementation"
)
PROMOTION_POLICY = (
    "a passing artifact never auto-promotes a repository claim; independent human review "
    "must verify untouched seeds, chronology, artifact placement, and claim wording"
)
COMPACT_RECORD_TRUST_BOUNDARY = (
    "compact records are gate-sufficient and retain generator validation sentinels plus "
    "exact trace/final-state hashes, but they do not contain full traces or an external "
    "signature; structural validation proves internal consistency, while only the explicit "
    "same-backend deterministic rerun independently verifies run provenance"
)
SOURCE_CLOSURE_POLICY = (
    "the sorted, deduplicated static transitive local Python import closure rooted at this "
    "protocol module is resolved from import specifications and parsed source, including "
    "parent package __init__ modules; pyproject.toml and uv.lock bind the external dependency "
    "graph; unrelated modules loaded earlier in the process are excluded"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 2_000_000


_SOURCE_ROOT_MODULE = "alberta_framework.evaluation.hidden_regime_signaling_evidence"


def _parent_packages(module: str) -> set[str]:
    parts = module.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts))}


def _local_module_path(module: str, root: Path) -> Path | None:
    """Resolve one Alberta module from its import specification without importing it."""

    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    try:
        spec = importlib.util.find_spec(module)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin in {None, "built-in", "frozen"}:
        return None
    path = Path(spec.origin)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return None
    candidate = root.resolve() / relative
    if relative.suffix != ".py" or not candidate.is_file():
        return None
    return candidate


def _resolve_local_imports(module: str, path: Path, root: Path) -> set[str]:
    """Return statically referenced local modules and their parent packages."""

    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"cannot parse source closure member {path}: {exc}") from exc
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for candidate in candidates:
            parts = candidate.split(".")
            while parts:
                possible = ".".join(parts)
                if _local_module_path(possible, root) is not None:
                    found.add(possible)
                    found.update(_parent_packages(possible))
                    break
                parts.pop()
    return found


def _algorithm_source_paths(root: Path = REPO_ROOT) -> tuple[Path, ...]:
    """Build a deterministic static local import closure for the protocol."""

    resolved_root = root.resolve()
    pending = {_SOURCE_ROOT_MODULE, *_parent_packages(_SOURCE_ROOT_MODULE)}
    visited: set[str] = set()
    selected = {Path("pyproject.toml"), Path("uv.lock")}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        path = _local_module_path(module, resolved_root)
        if path is None:
            raise ValueError(f"source closure module is missing: {module}")
        visited.add(module)
        selected.add(path.relative_to(resolved_root))
        pending.update(_resolve_local_imports(module, path, resolved_root) - visited)
    return tuple(sorted(selected))


SOURCE_PATHS = _algorithm_source_paths()


@dataclass(frozen=True)
class PreregistrationValidation:
    """Result of fail-closed preregistration validation."""

    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceValidation:
    """Structural validity and separately reported frozen-gate outcome."""

    valid: bool
    accepted: bool
    same_backend_rerun_verified: bool
    errors: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_equal(left: object, right: object) -> bool:
    try:
        return _canonical_json_bytes(left) == _canonical_json_bytes(right)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_finite_tree(value: object, errors: list[str], path: str = "$") -> None:
    """Reject non-JSON types and non-finite numbers supplied as Python objects."""

    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            errors.append(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                errors.append(f"{path} contains a non-string mapping key")
                continue
            _validate_finite_tree(child, errors, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, errors, f"{path}[{index}]")
        return
    errors.append(f"{path} contains a non-JSON value of type {type(value).__name__}")


def strict_json_load(path: Path) -> dict[str, object]:
    """Load bounded JSON while rejecting duplicate keys and non-finite values."""

    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON document exceeds the protocol byte limit")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def parse_integer(value: str) -> int:
        if len(value.lstrip("-")) > 20:
            raise ValueError("JSON integer exceeds the protocol digit limit")
        return int(value)

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON float is forbidden")
        return parsed

    parsed = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_int=parse_integer,
        parse_float=parse_float,
    )
    if not isinstance(parsed, dict):
        raise ValueError("protocol JSON must contain one top-level object")

    nodes = 0
    stack: list[tuple[object, int]] = [(parsed, 1)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError("JSON document exceeds the protocol node limit")
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON document exceeds the protocol nesting limit")
        if isinstance(current, Mapping):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return parsed


def strict_json_text(value: Mapping[str, object]) -> str:
    """Serialize deterministic, finite, human-readable protocol JSON."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def source_snapshot(root: Path = REPO_ROOT) -> dict[str, str]:
    """Hash the exact declared conservative source/import closure."""

    return {
        relative.as_posix(): hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


def frozen_protocol_config() -> dict[str, object]:
    """Return a fresh JSON-compatible copy of the exact candidate."""

    return {
        "world": {
            "segment_lengths": list(FROZEN_SEGMENT_LENGTHS),
            "segment_regimes": list(FROZEN_SEGMENT_REGIMES),
            "regime_permutations": [list(row) for row in FROZEN_REGIME_PERMUTATIONS],
            "repeat_schedule": False,
            "total_schedule_steps": FROZEN_NUM_STEPS,
        },
        "learner": {
            "learning_rate": 0.25,
            "epsilon": 0.1,
            "relevance_rate": 0.1,
            "lease_length": 16,
            "confirmation_steps": 8,
            "durable_retrieval_threshold": 0.5,
            "candidate_confirmation_threshold": 0.75,
            "candidate_confirmation_leases": 3,
            "scratch_training_leases_before_retest": 16,
            "writable_lru_ablation": False,
        },
        "metric_window": FROZEN_METRIC_WINDOW,
        "num_steps": FROZEN_NUM_STEPS,
    }


def _frozen_evaluator_config() -> HiddenRegimeDevelopmentConfig:
    """Reconstruct the development evaluator only from frozen evidence values."""

    return HiddenRegimeDevelopmentConfig(
        world=HiddenRegimeWorldConfig(
            segment_lengths=FROZEN_SEGMENT_LENGTHS,
            segment_regimes=FROZEN_SEGMENT_REGIMES,
            regime_permutations=FROZEN_REGIME_PERMUTATIONS,
            repeat_schedule=False,
        ),
        learner=SlotSignalingConfig(
            learning_rate=0.25,
            epsilon=0.1,
            relevance_rate=0.1,
            lease_length=16,
            confirmation_steps=8,
            durable_retrieval_threshold=0.5,
            candidate_confirmation_threshold=0.75,
            candidate_confirmation_leases=3,
            scratch_training_leases_before_retest=16,
            writable_lru_ablation=False,
        ),
        metric_window=FROZEN_METRIC_WINDOW,
    )


def _derive_evidence_seed_pairs(namespace: str, count: int) -> tuple[HiddenRegimeSeedPair, ...]:
    """Derive SHA-separated world/learner seeds without running the protocol."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    pairs: list[HiddenRegimeSeedPair] = []
    used: set[int] = set()
    for index in range(count):
        derived: list[int] = []
        for owner in ("world", "learner"):
            material = (f"{SEED_DERIVATION_SCHEMA}\0{namespace}\0{index}\0{owner}").encode()
            seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
            if seed in used:
                raise ValueError("SHA-derived seed collision; protocol must be versioned")
            used.add(seed)
            derived.append(seed)
        pairs.append(HiddenRegimeSeedPair(namespace, index, derived[0], derived[1]))
    return tuple(pairs)


def _protocol_kind(namespace: str, seed_count: int) -> str:
    if namespace == RESERVED_HELDOUT_NAMESPACE:
        if seed_count != HELDOUT_SEED_COUNT:
            raise ValueError("the protected held-out namespace requires exactly 30 seed pairs")
        return "protected_heldout"
    if namespace.startswith(MANUAL_TEST_NAMESPACE_PREFIX):
        if not 1 <= seed_count <= HELDOUT_SEED_COUNT:
            raise ValueError("manual test plans are limited to one through thirty seed pairs")
        return "manual_nonpromoting"
    raise ValueError("namespace is neither the protected held-out namespace nor a manual test")


def _validate_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset.total_seconds() == 0.0


def build_preregistration_plan(
    *,
    namespace: str,
    seed_count: int,
    preregistered_at_utc: str,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build an unexecuted plan; this is the only operation that derives its seeds."""

    if namespace == RESERVED_HELDOUT_NAMESPACE and not PROTECTED_EXECUTION_ENABLED:
        raise ValueError(
            "protected plan construction is disabled while the factorial/boundary "
            "protocol is unsettled"
        )
    kind = _protocol_kind(namespace, seed_count)
    if not _validate_utc_timestamp(preregistered_at_utc):
        raise ValueError("preregistered_at_utc must be an ISO-8601 UTC timestamp")
    seed_pairs = _derive_evidence_seed_pairs(namespace, seed_count)
    scientific_payload: dict[str, object] = {
        "protocol_kind": kind,
        "protocol_status": PROTOCOL_STATUS,
        "namespace": namespace,
        "seed_derivation_schema": SEED_DERIVATION_SCHEMA,
        "seed_count": seed_count,
        "seed_pairs": [pair.to_dict() for pair in seed_pairs],
        "condition_order": list(FROZEN_CONDITION_ORDER),
        "config": frozen_protocol_config(),
        "thresholds": copy.deepcopy(FROZEN_THRESHOLDS),
        "threshold_status": "frozen_by_this_plan_before_execution",
        "executed": False,
        "reserved_heldout_namespace": RESERVED_HELDOUT_NAMESPACE,
        "reserved_heldout_namespace_executed_at_module_release": (
            RESERVED_HELDOUT_NAMESPACE_EXECUTED
        ),
        "preregistered_at_utc": preregistered_at_utc,
        "source_closure_policy": SOURCE_CLOSURE_POLICY,
        "source_sha256": source_snapshot(root),
        "consumed_calibration_disclosure": CONSUMED_CALIBRATION_DISCLOSURE,
        "claim_scope": CLAIM_SCOPE,
        "retention_limitation": RETENTION_LIMITATION,
        "time_attestation_limitation": TIME_ATTESTATION_LIMITATION,
        "replay_scope": REPLAY_SCOPE,
        "promotion_policy": PROMOTION_POLICY,
        "compact_record_trust_boundary": COMPACT_RECORD_TRUST_BOUNDARY,
        "automatic_promotion_allowed": False,
    }
    digest = _canonical_sha256(scientific_payload)
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "plan_status": "preregistered_unexecuted",
        "scientific_payload": scientific_payload,
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": digest,
        },
    }


def _validate_digest(
    digest: object,
    payload: object,
    *,
    scope: str,
    label: str,
    errors: list[str],
) -> None:
    expected_fields = {"algorithm", "scope", "sha256"}
    if not isinstance(digest, Mapping) or set(digest) != expected_fields:
        errors.append(f"{label} scientific_digest fields do not match the schema")
        return
    if digest.get("algorithm") != "sha256" or digest.get("scope") != scope:
        errors.append(f"{label} scientific_digest metadata mismatch")
    recorded = digest.get("sha256")
    if not _valid_sha256(recorded):
        errors.append(f"{label} scientific_digest sha256 is malformed")
        return
    try:
        expected = _canonical_sha256(payload)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{label} scientific payload is not strict finite JSON")
        return
    if recorded != expected:
        errors.append(f"{label} scientific_digest mismatch")


def validate_preregistration_plan(
    plan: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    check_current_source: bool = True,
) -> PreregistrationValidation:
    """Validate an exact unexecuted plan and optionally bind it to current source."""

    errors: list[str] = []
    _validate_finite_tree(plan, errors)
    if set(plan) != {
        "schema_version",
        "plan_status",
        "scientific_payload",
        "scientific_digest",
    }:
        errors.append("preregistration top-level fields do not match strict v1")
    if plan.get("schema_version") != PREREGISTRATION_SCHEMA:
        errors.append("preregistration schema_version mismatch")
    if plan.get("plan_status") != "preregistered_unexecuted":
        errors.append("plan_status must remain preregistered_unexecuted")
    payload = plan.get("scientific_payload")
    _validate_digest(
        plan.get("scientific_digest"),
        payload,
        scope="$.scientific_payload",
        label="plan",
        errors=errors,
    )
    expected_payload_fields = {
        "protocol_kind",
        "protocol_status",
        "namespace",
        "seed_derivation_schema",
        "seed_count",
        "seed_pairs",
        "condition_order",
        "config",
        "thresholds",
        "threshold_status",
        "executed",
        "reserved_heldout_namespace",
        "reserved_heldout_namespace_executed_at_module_release",
        "preregistered_at_utc",
        "source_closure_policy",
        "source_sha256",
        "consumed_calibration_disclosure",
        "claim_scope",
        "retention_limitation",
        "time_attestation_limitation",
        "replay_scope",
        "promotion_policy",
        "compact_record_trust_boundary",
        "automatic_promotion_allowed",
    }
    if not isinstance(payload, Mapping):
        errors.append("scientific_payload must be a mapping")
        return PreregistrationValidation(False, tuple(errors))
    if set(payload) != expected_payload_fields:
        errors.append("preregistration scientific_payload fields do not match strict v1")

    namespace = payload.get("namespace")
    seed_count = payload.get("seed_count")
    expected_kind: str | None = None
    if not isinstance(namespace, str) or type(seed_count) is not int:
        errors.append("namespace and seed_count have invalid types")
    else:
        try:
            expected_kind = _protocol_kind(namespace, seed_count)
        except ValueError as exc:
            errors.append(str(exc))
    if expected_kind is not None and payload.get("protocol_kind") != expected_kind:
        errors.append("protocol_kind does not match namespace and seed count")
    if payload.get("protocol_status") != PROTOCOL_STATUS:
        errors.append("protocol_status must identify this execution-disabled draft")
    if payload.get("seed_derivation_schema") != SEED_DERIVATION_SCHEMA:
        errors.append("seed_derivation_schema mismatch")
    if payload.get("condition_order") != list(FROZEN_CONDITION_ORDER):
        errors.append("condition_order does not match the frozen order")
    if not _strict_json_equal(payload.get("config"), frozen_protocol_config()):
        errors.append("config does not match the exact frozen candidate")
    if not _strict_json_equal(payload.get("thresholds"), FROZEN_THRESHOLDS):
        errors.append("thresholds do not match the frozen gates")
    if payload.get("threshold_status") != "frozen_by_this_plan_before_execution":
        errors.append("threshold_status mismatch")
    if payload.get("executed") is not False:
        errors.append("preregistration plan must remain unexecuted")
    if payload.get("reserved_heldout_namespace") != RESERVED_HELDOUT_NAMESPACE:
        errors.append("reserved held-out namespace declaration mismatch")
    if payload.get("reserved_heldout_namespace_executed_at_module_release") is not False:
        errors.append("module-release protected namespace sentinel must remain false")
    if not _validate_utc_timestamp(payload.get("preregistered_at_utc")):
        errors.append("preregistered_at_utc must be an ISO-8601 UTC timestamp")
    expected_literals = {
        "source_closure_policy": SOURCE_CLOSURE_POLICY,
        "consumed_calibration_disclosure": CONSUMED_CALIBRATION_DISCLOSURE,
        "claim_scope": CLAIM_SCOPE,
        "retention_limitation": RETENTION_LIMITATION,
        "time_attestation_limitation": TIME_ATTESTATION_LIMITATION,
        "replay_scope": REPLAY_SCOPE,
        "promotion_policy": PROMOTION_POLICY,
        "compact_record_trust_boundary": COMPACT_RECORD_TRUST_BOUNDARY,
        "automatic_promotion_allowed": False,
    }
    for field, expected in expected_literals.items():
        if payload.get(field) != expected:
            errors.append(f"{field} mismatch")

    seeds = payload.get("seed_pairs")
    if not isinstance(seeds, list) or type(seed_count) is not int or len(seeds) != seed_count:
        errors.append("seed_pairs must align exactly with seed_count")
    elif isinstance(namespace, str):
        try:
            expected_seeds = [
                pair.to_dict() for pair in _derive_evidence_seed_pairs(namespace, seed_count)
            ]
        except ValueError as exc:
            errors.append(f"seed derivation failed: {exc}")
        else:
            if seeds != expected_seeds:
                errors.append("seed_pairs do not match SHA-separated derivation")

    recorded_source = payload.get("source_sha256")
    expected_source_keys = {path.as_posix() for path in SOURCE_PATHS}
    if not isinstance(recorded_source, Mapping):
        errors.append("source_sha256 must be a mapping")
    else:
        if set(recorded_source) != expected_source_keys:
            errors.append("source_sha256 paths do not match the declared closure")
        if any(not _valid_sha256(value) for value in recorded_source.values()):
            errors.append("source_sha256 contains a malformed digest")
        if check_current_source:
            try:
                current_source = source_snapshot(root)
            except OSError as exc:
                errors.append(f"current source snapshot failed: {exc}")
            else:
                if dict(recorded_source) != current_source:
                    errors.append("current source differs from the preregistered closure")
    return PreregistrationValidation(not errors, tuple(errors))


def _same_backend_trace_sha256(trace: HiddenRegimePrimitiveTrace) -> str:
    """Hash exact trace array metadata and bytes for optional same-backend rerun."""

    digest = hashlib.sha256()
    for field in dataclasses.fields(trace):
        array = np.ascontiguousarray(np.asarray(getattr(trace, field.name)))
        digest.update(field.name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)) + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _same_backend_state_sha256(state: object) -> str:
    """Hash JAX pytree leaves in stable traversal order for exact rerun."""

    digest = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(state)
    digest.update(str(len(leaves)).encode("ascii") + b"\0")
    for index, leaf in enumerate(leaves):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(index).encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)) + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _segment_bounds() -> tuple[tuple[int, int], ...]:
    starts = np.cumsum((0, *FROZEN_SEGMENT_LENGTHS), dtype=np.int64)
    return tuple(
        (int(starts[index]), int(starts[index + 1])) for index in range(len(FROZEN_SEGMENT_LENGTHS))
    )


FROZEN_SEGMENT_BOUNDS = _segment_bounds()


def _step_segment(step: int) -> int:
    return int(np.searchsorted(np.asarray(FROZEN_SEGMENT_LENGTHS).cumsum(), step, side="right"))


def _integer_coordinates(values: object) -> list[list[int]]:
    coordinates = np.argwhere(np.asarray(values, dtype=np.bool_))
    return [[int(value) for value in row] for row in coordinates]


def _event_records(trace: HiddenRegimePrimitiveTrace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    regimes = np.asarray(trace.regime_id, dtype=np.int32)
    segments = np.asarray(trace.segment_index, dtype=np.int32)
    for step in range(FROZEN_NUM_STEPS):
        for role in ("helper", "beneficiary"):
            committed_slot = int(np.asarray(getattr(trace, f"{role}_committed_slot"))[step])
            retired_slot = int(np.asarray(getattr(trace, f"{role}_retired_slot"))[step])
            if committed_slot < 0 and retired_slot < 0:
                continue
            records.append(
                {
                    "step": step,
                    "segment_index": int(segments[step]),
                    "regime_id": int(regimes[step]),
                    "role": role,
                    "committed_slot": committed_slot,
                    "committed_generation": int(
                        np.asarray(getattr(trace, f"{role}_committed_generation"))[step]
                    ),
                    "retired_slot": retired_slot,
                    "retired_generation": int(
                        np.asarray(getattr(trace, f"{role}_retired_generation"))[step]
                    ),
                }
            )
    return records


def _d_checkpoint_records(trace: HiddenRegimePrimitiveTrace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for segment_index, regime in enumerate(FROZEN_SEGMENT_REGIMES):
        if regime != 4:
            continue
        start, end = FROZEN_SEGMENT_BOUNDS[segment_index]
        record: dict[str, object] = {"segment_index": segment_index}
        for role in ("helper", "beneficiary"):
            record[f"{role}_status_pre"] = (
                np.asarray(getattr(trace, f"{role}_status_pre"))[start].astype(np.int64).tolist()
            )
            record[f"{role}_status_post"] = (
                np.asarray(getattr(trace, f"{role}_status_post"))[end - 1].astype(np.int64).tolist()
            )
            record[f"{role}_generation_pre"] = (
                np.asarray(getattr(trace, f"{role}_generation_pre"))[start]
                .astype(np.int64)
                .tolist()
            )
            record[f"{role}_generation_post"] = (
                np.asarray(getattr(trace, f"{role}_generation_post"))[end - 1]
                .astype(np.int64)
                .tolist()
            )
        records.append(record)
    return records


def _compact_run_result(run: HiddenRegimeRunResult) -> dict[str, object]:
    """Reduce a replay-validated trace to gate-sufficient integer primitives."""

    replay_errors = validate_hidden_regime_run_result(run)
    if replay_errors:
        raise ValueError("development run failed strict replay: " + "; ".join(replay_errors))
    if run.condition not in FROZEN_CONDITION_ORDER:
        raise ValueError("run condition is outside the frozen order")
    if not _strict_json_equal(
        {
            "world": {
                "segment_lengths": list(run.config.world.segment_lengths),
                "segment_regimes": list(run.config.world.segment_regimes),
                "regime_permutations": [list(row) for row in run.config.world.regime_permutations],
                "repeat_schedule": run.config.world.repeat_schedule,
                "total_schedule_steps": run.config.world.total_schedule_steps,
            },
            "learner": {
                key: value
                for key, value in run.config.learner.to_dict().items()
                if key not in {"development_only", "scientific_promotion_allowed"}
            },
            "metric_window": run.config.metric_window,
            "num_steps": run.config.num_steps,
        },
        frozen_protocol_config(),
    ):
        raise ValueError("run config differs from the frozen evidence candidate")

    rewards = np.asarray(run.trace.reward, dtype=np.float32)
    if rewards.shape != (FROZEN_NUM_STEPS,) or not np.all(
        np.logical_or(rewards == np.float32(0.0), rewards == np.float32(1.0))
    ):
        raise ValueError("trace rewards must be exact finite binary primitives")
    segments: list[dict[str, object]] = []
    for segment_index, (start, end) in enumerate(FROZEN_SEGMENT_BOUNDS):
        window = min(FROZEN_METRIC_WINDOW, end - start)
        segment_reward = rewards[start:end]
        segments.append(
            {
                "segment_index": segment_index,
                "regime_id": FROZEN_SEGMENT_REGIMES[segment_index],
                "steps": end - start,
                "reward_count": int(np.sum(segment_reward, dtype=np.int64)),
                "early_steps": window,
                "early_reward_count": int(np.sum(segment_reward[:window], dtype=np.int64)),
                "late_steps": window,
                "late_reward_count": int(np.sum(segment_reward[-window:], dtype=np.int64)),
            }
        )

    diagnostics: dict[str, object] = {}
    for role in ("helper", "beneficiary"):
        writes = np.asarray(getattr(run.trace, f"{role}_value_write"), dtype=np.bool_)
        value_pre = np.asarray(getattr(run.trace, f"{role}_value_pre"), dtype=np.float32)
        candidate = np.asarray(getattr(run.trace, f"{role}_candidate_value"), dtype=np.float32)
        effective = np.logical_and(writes, value_pre.view(np.uint32) != candidate.view(np.uint32))
        candidate_success = np.asarray(
            getattr(run.trace, f"{role}_candidate_lease_success"), dtype=np.bool_
        )
        confirmations_post = np.asarray(
            getattr(run.trace, f"{role}_candidate_confirmations_post"), dtype=np.int32
        )
        diagnostics[f"{role}_value_write_count"] = int(np.count_nonzero(writes))
        diagnostics[f"{role}_effective_learning_update_count"] = int(np.count_nonzero(effective))
        diagnostics[f"{role}_candidate_confirmation_steps"] = [
            int(step)
            for step in np.flatnonzero(np.logical_and(candidate_success, confirmations_post == 0))
        ]
        diagnostics[f"{role}_scratch_retest_steps"] = [
            int(step)
            for step in np.flatnonzero(
                np.asarray(getattr(run.trace, f"{role}_scratch_retest_started"))
            )
        ]
        diagnostics[f"{role}_selective_mutation_coordinates"] = _integer_coordinates(
            getattr(run.trace, f"{role}_selective_mutation_violation")
        )
    diagnostics["lifecycle_desynchronization_steps"] = [
        int(step)
        for step in np.flatnonzero(
            np.logical_not(np.asarray(run.trace.lifecycle_synchronized, dtype=np.bool_))
        )
    ]

    payload: dict[str, object] = {
        "schema_version": COMPACT_PRIMITIVES_SCHEMA,
        "condition": run.condition,
        "strict_run_validation_passed": True,
        "segments": segments,
        "lifecycle_events": _event_records(run.trace),
        "diagnostics": diagnostics,
        "d_checkpoints": _d_checkpoint_records(run.trace),
        "resource": run.resource.to_dict(),
        "same_backend_trace_sha256": _same_backend_trace_sha256(run.trace),
        "same_backend_final_state_sha256": _same_backend_state_sha256(run.final_state),
    }
    payload["compact_digest_sha256"] = _canonical_sha256(payload)
    return payload


@dataclass(frozen=True)
class _RunMetrics:
    condition: str
    mean_reward: float
    recurrence_entry_reward: float
    helper_updates: int
    beneficiary_updates: int
    helper_writes: int
    beneficiary_writes: int
    helper_commits: int
    beneficiary_commits: int
    helper_replacements: int
    beneficiary_replacements: int
    helper_c_replacements: int
    beneficiary_c_replacements: int
    d_non_displacement: bool
    immutable: bool
    synchronized: bool
    resource_valid: bool
    strict_run_validation_passed: bool

    @property
    def strict_selective_lifecycle(self) -> bool:
        return (
            self.strict_run_validation_passed
            and self.helper_updates > 0
            and self.beneficiary_updates > 0
            and self.helper_commits == 4
            and self.beneficiary_commits == 4
            and self.helper_replacements == 1
            and self.beneficiary_replacements == 1
            and self.helper_c_replacements == 1
            and self.beneficiary_c_replacements == 1
            and self.d_non_displacement
            and self.immutable
            and self.synchronized
            and self.resource_valid
        )


def _is_exact_int(value: object, *, minimum: int = 0, maximum: int | None = None) -> bool:
    if type(value) is not int or value < minimum:
        return False
    return maximum is None or value <= maximum


def _validate_sorted_steps(
    value: object,
    label: str,
    errors: list[str],
) -> list[int]:
    if not isinstance(value, list) or any(
        not _is_exact_int(step, maximum=FROZEN_NUM_STEPS - 1) for step in value
    ):
        errors.append(f"{label} must be a list of in-range integer steps")
        return []
    steps = cast(list[int], value)
    if steps != sorted(set(steps)):
        errors.append(f"{label} must be strictly increasing and unique")
    return steps


def _validate_mutation_coordinates(
    value: object,
    label: str,
    errors: list[str],
) -> list[list[int]]:
    if not isinstance(value, list):
        errors.append(f"{label} must be a coordinate list")
        return []
    coordinates: list[list[int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not _is_exact_int(item[0], maximum=FROZEN_NUM_STEPS - 1)
            or not _is_exact_int(item[1], maximum=3)
        ):
            errors.append(f"{label} contains an invalid [step, slot] coordinate")
            return []
        coordinates.append(cast(list[int], item))
    expected = [list(item) for item in sorted({tuple(item) for item in coordinates})]
    if coordinates != expected:
        errors.append(f"{label} must be lexicographically sorted and unique")
    return coordinates


def _validate_segments(value: object, label: str, errors: list[str]) -> tuple[float, float]:
    expected_fields = {
        "segment_index",
        "regime_id",
        "steps",
        "reward_count",
        "early_steps",
        "early_reward_count",
        "late_steps",
        "late_reward_count",
    }
    if not isinstance(value, list) or len(value) != len(FROZEN_SEGMENT_LENGTHS):
        errors.append(f"{label} segments do not match the frozen schedule")
        return 0.0, 0.0
    reward_total = 0
    recurrence_entries: list[float] = []
    previous_regimes: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            errors.append(f"{label} segment {index} fields do not match strict v1")
            continue
        steps = FROZEN_SEGMENT_LENGTHS[index]
        window = min(FROZEN_METRIC_WINDOW, steps)
        expected_fixed = {
            "segment_index": index,
            "regime_id": FROZEN_SEGMENT_REGIMES[index],
            "steps": steps,
            "early_steps": window,
            "late_steps": window,
        }
        for field, expected in expected_fixed.items():
            if item.get(field) != expected:
                errors.append(f"{label} segment {index} {field} mismatch")
        for field, maximum in (
            ("reward_count", steps),
            ("early_reward_count", window),
            ("late_reward_count", window),
        ):
            count = item.get(field)
            if not _is_exact_int(count, maximum=maximum):
                errors.append(f"{label} segment {index} {field} is out of range")
        reward_count = item.get("reward_count")
        early_count = item.get("early_reward_count")
        if type(reward_count) is int:
            reward_total += reward_count
        regime = FROZEN_SEGMENT_REGIMES[index]
        if regime in previous_regimes and regime in (0, 1, 2, 3) and type(early_count) is int:
            recurrence_entries.append(early_count / window)
        previous_regimes.add(regime)
    recurrence = math.fsum(recurrence_entries) / len(recurrence_entries)
    return reward_total / FROZEN_NUM_STEPS, recurrence


def _validate_lifecycle_events(
    value: object,
    label: str,
    errors: list[str],
) -> tuple[int, int, int, int, int, int]:
    expected_fields = {
        "step",
        "segment_index",
        "regime_id",
        "role",
        "committed_slot",
        "committed_generation",
        "retired_slot",
        "retired_generation",
    }
    if not isinstance(value, list):
        errors.append(f"{label} lifecycle_events must be a list")
        return (0, 0, 0, 0, 0, 0)
    previous_order: tuple[int, int] | None = None
    generation_regime: dict[str, dict[int, int]] = {"helper": {}, "beneficiary": {}}
    commits = {"helper": 0, "beneficiary": 0}
    replacements = {"helper": 0, "beneficiary": 0}
    c_replacements = {"helper": 0, "beneficiary": 0}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            errors.append(f"{label} lifecycle event {index} fields mismatch")
            continue
        step = item.get("step")
        role = item.get("role")
        if not _is_exact_int(step, maximum=FROZEN_NUM_STEPS - 1):
            errors.append(f"{label} lifecycle event {index} has invalid step")
            continue
        if role not in ("helper", "beneficiary"):
            errors.append(f"{label} lifecycle event {index} has invalid role")
            continue
        typed_step = cast(int, step)
        typed_role = cast(str, role)
        order = (typed_step, 0 if typed_role == "helper" else 1)
        if previous_order is not None and order <= previous_order:
            errors.append(f"{label} lifecycle events are not strictly ordered")
        previous_order = order
        segment_index = _step_segment(typed_step)
        regime = FROZEN_SEGMENT_REGIMES[segment_index]
        if item.get("segment_index") != segment_index or item.get("regime_id") != regime:
            errors.append(f"{label} lifecycle event {index} step provenance mismatch")
        committed_slot = item.get("committed_slot")
        committed_generation = item.get("committed_generation")
        retired_slot = item.get("retired_slot")
        retired_generation = item.get("retired_generation")
        if not _is_exact_int(committed_slot, minimum=1, maximum=3):
            errors.append(f"{label} lifecycle event {index} committed_slot invalid")
            continue
        if not _is_exact_int(committed_generation, minimum=1, maximum=2_147_483_647):
            errors.append(f"{label} lifecycle event {index} committed_generation invalid")
            continue
        no_retirement = retired_slot == -1 and retired_generation == -1
        has_retirement = retired_slot == committed_slot and _is_exact_int(
            retired_generation, minimum=1, maximum=2_147_483_647
        )
        if not (no_retirement or has_retirement):
            errors.append(f"{label} lifecycle event {index} retirement is not atomic")
            continue
        new_generation = cast(int, committed_generation)
        if new_generation in generation_regime[typed_role]:
            errors.append(f"{label} lifecycle event {index} reuses a generation")
        if has_retirement and cast(int, retired_generation) not in generation_regime[typed_role]:
            errors.append(f"{label} lifecycle event {index} retires an unknown generation")
        if (
            has_retirement
            and generation_regime[typed_role].get(cast(int, retired_generation)) == 2
            and regime == 3
        ):
            c_replacements[typed_role] += 1
        generation_regime[typed_role][new_generation] = regime
        commits[typed_role] += 1
        replacements[typed_role] += int(has_retirement)
    return (
        commits["helper"],
        commits["beneficiary"],
        replacements["helper"],
        replacements["beneficiary"],
        c_replacements["helper"],
        c_replacements["beneficiary"],
    )


def _validate_d_checkpoints(value: object, label: str, errors: list[str]) -> bool:
    expected_fields = {
        "segment_index",
        "helper_status_pre",
        "helper_status_post",
        "helper_generation_pre",
        "helper_generation_post",
        "beneficiary_status_pre",
        "beneficiary_status_post",
        "beneficiary_generation_pre",
        "beneficiary_generation_post",
    }
    expected_segments = [
        index for index, regime in enumerate(FROZEN_SEGMENT_REGIMES) if regime == 4
    ]
    if not isinstance(value, list) or len(value) != len(expected_segments):
        errors.append(f"{label} D checkpoints do not cover the frozen D segments")
        return False
    unchanged = True
    for record_index, (item, segment_index) in enumerate(
        zip(value, expected_segments, strict=True)
    ):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            errors.append(f"{label} D checkpoint {record_index} fields mismatch")
            unchanged = False
            continue
        if item.get("segment_index") != segment_index:
            errors.append(f"{label} D checkpoint {record_index} segment mismatch")
        for role in ("helper", "beneficiary"):
            for kind in ("status", "generation"):
                pre = item.get(f"{role}_{kind}_pre")
                post = item.get(f"{role}_{kind}_post")
                upper = 2 if kind == "status" else 2_147_483_647
                if (
                    not isinstance(pre, list)
                    or not isinstance(post, list)
                    or len(pre) != 4
                    or len(post) != 4
                    or any(not _is_exact_int(entry, maximum=upper) for entry in (*pre, *post))
                ):
                    errors.append(f"{label} D checkpoint {record_index} {role} {kind} is invalid")
                    unchanged = False
                elif pre != post:
                    unchanged = False
    return unchanged


def _validate_resource(value: object, label: str, errors: list[str]) -> bool:
    expected_fields = {
        "initial_state_scalars",
        "final_state_scalars",
        "initial_state_bytes",
        "final_state_bytes",
        "expected_state_bytes",
        "resource_constant",
        "resource_matched",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        errors.append(f"{label} resource fields mismatch")
        return False
    for field in ("initial_state_scalars", "final_state_scalars"):
        if not _is_exact_int(value.get(field), minimum=1):
            errors.append(f"{label} resource {field} must be positive")
    valid = (
        value.get("initial_state_scalars") == value.get("final_state_scalars")
        and value.get("initial_state_bytes") == EXPECTED_DYAD_STATE_BYTES
        and value.get("final_state_bytes") == EXPECTED_DYAD_STATE_BYTES
        and value.get("expected_state_bytes") == EXPECTED_DYAD_STATE_BYTES
        and value.get("resource_constant") is True
        and value.get("resource_matched") is True
    )
    if not valid:
        errors.append(f"{label} resource is not fixed and matched at 552 bytes")
    return valid


def _validate_compact_run(
    value: object,
    expected_condition: str,
    label: str,
    errors: list[str],
) -> _RunMetrics | None:
    expected_fields = {
        "schema_version",
        "condition",
        "strict_run_validation_passed",
        "segments",
        "lifecycle_events",
        "diagnostics",
        "d_checkpoints",
        "resource",
        "same_backend_trace_sha256",
        "same_backend_final_state_sha256",
        "compact_digest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        errors.append(f"{label} fields do not match compact primitives v1")
        return None
    if value.get("schema_version") != COMPACT_PRIMITIVES_SCHEMA:
        errors.append(f"{label} compact schema mismatch")
    if value.get("condition") != expected_condition:
        errors.append(f"{label} condition order mismatch")
    strict_run_validation_passed = value.get("strict_run_validation_passed") is True
    if not strict_run_validation_passed:
        errors.append(f"{label} lacks a successful strict source-bound run validation")
    digest_payload = {key: child for key, child in value.items() if key != "compact_digest_sha256"}
    if not _valid_sha256(value.get("compact_digest_sha256")) or value.get(
        "compact_digest_sha256"
    ) != _canonical_sha256(digest_payload):
        errors.append(f"{label} compact digest mismatch")
    for field in ("same_backend_trace_sha256", "same_backend_final_state_sha256"):
        if not _valid_sha256(value.get(field)):
            errors.append(f"{label} {field} is malformed")

    mean_reward, recurrence = _validate_segments(value.get("segments"), label, errors)
    (
        helper_commits,
        beneficiary_commits,
        helper_replacements,
        beneficiary_replacements,
        helper_c_replacements,
        beneficiary_c_replacements,
    ) = _validate_lifecycle_events(value.get("lifecycle_events"), label, errors)
    d_non_displacement = _validate_d_checkpoints(value.get("d_checkpoints"), label, errors)
    resource_valid = _validate_resource(value.get("resource"), label, errors)

    expected_diagnostic_fields = {
        "helper_value_write_count",
        "beneficiary_value_write_count",
        "helper_effective_learning_update_count",
        "beneficiary_effective_learning_update_count",
        "helper_candidate_confirmation_steps",
        "beneficiary_candidate_confirmation_steps",
        "helper_scratch_retest_steps",
        "beneficiary_scratch_retest_steps",
        "helper_selective_mutation_coordinates",
        "beneficiary_selective_mutation_coordinates",
        "lifecycle_desynchronization_steps",
    }
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_diagnostic_fields:
        errors.append(f"{label} diagnostics fields mismatch")
        return None
    counts: dict[str, int] = {}
    for field in (
        "helper_value_write_count",
        "beneficiary_value_write_count",
        "helper_effective_learning_update_count",
        "beneficiary_effective_learning_update_count",
    ):
        item = diagnostics.get(field)
        if not _is_exact_int(item, maximum=FROZEN_NUM_STEPS):
            errors.append(f"{label} diagnostics {field} is invalid")
            counts[field] = 0
        else:
            counts[field] = cast(int, item)
    if counts["helper_effective_learning_update_count"] > counts["helper_value_write_count"]:
        errors.append(f"{label} helper effective updates exceed writes")
    if (
        counts["beneficiary_effective_learning_update_count"]
        > counts["beneficiary_value_write_count"]
    ):
        errors.append(f"{label} beneficiary effective updates exceed writes")
    for role in ("helper", "beneficiary"):
        _validate_sorted_steps(
            diagnostics.get(f"{role}_candidate_confirmation_steps"),
            f"{label} {role} candidate confirmations",
            errors,
        )
        _validate_sorted_steps(
            diagnostics.get(f"{role}_scratch_retest_steps"),
            f"{label} {role} scratch retests",
            errors,
        )
    helper_mutations = _validate_mutation_coordinates(
        diagnostics.get("helper_selective_mutation_coordinates"),
        f"{label} helper mutations",
        errors,
    )
    beneficiary_mutations = _validate_mutation_coordinates(
        diagnostics.get("beneficiary_selective_mutation_coordinates"),
        f"{label} beneficiary mutations",
        errors,
    )
    desync = _validate_sorted_steps(
        diagnostics.get("lifecycle_desynchronization_steps"),
        f"{label} lifecycle desynchronization",
        errors,
    )
    return _RunMetrics(
        condition=expected_condition,
        mean_reward=mean_reward,
        recurrence_entry_reward=recurrence,
        helper_updates=counts["helper_effective_learning_update_count"],
        beneficiary_updates=counts["beneficiary_effective_learning_update_count"],
        helper_writes=counts["helper_value_write_count"],
        beneficiary_writes=counts["beneficiary_value_write_count"],
        helper_commits=helper_commits,
        beneficiary_commits=beneficiary_commits,
        helper_replacements=helper_replacements,
        beneficiary_replacements=beneficiary_replacements,
        helper_c_replacements=helper_c_replacements,
        beneficiary_c_replacements=beneficiary_c_replacements,
        d_non_displacement=d_non_displacement,
        immutable=not helper_mutations and not beneficiary_mutations,
        synchronized=not desync,
        resource_valid=resource_valid,
        strict_run_validation_passed=strict_run_validation_passed,
    )


def _compact_seed_record(
    seed_pair: HiddenRegimeSeedPair,
    runs: Sequence[HiddenRegimeRunResult],
) -> dict[str, object]:
    if tuple(run.condition for run in runs) != FROZEN_CONDITION_ORDER:
        raise ValueError("executed runs do not follow the frozen condition order")
    if any(run.seed_pair != seed_pair for run in runs):
        raise ValueError("executed run seed pairs are not paired exactly")
    payload: dict[str, object] = {
        "schema_version": COMPACT_PRIMITIVES_SCHEMA,
        "seed_pair": seed_pair.to_dict(),
        "condition_order": list(FROZEN_CONDITION_ORDER),
        "runs": [_compact_run_result(run) for run in runs],
    }
    payload["seed_record_digest_sha256"] = _canonical_sha256(payload)
    return payload


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty evidence sequence")
    return math.fsum(values) / len(values)


def _gate(
    name: str,
    value: int | float | bool,
    operator: str,
    threshold: int | float | bool,
    passed: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _paired_t_summary(deltas: Sequence[float]) -> dict[str, object]:
    """Return the preregistered one-sided paired-t primitives and lower bound."""

    count = len(deltas)
    if count != HELDOUT_SEED_COUNT:
        raise ValueError("paired inference requires exactly 30 paired lives")
    mean_delta = _mean(deltas)
    squared_deviations = math.fsum((value - mean_delta) ** 2 for value in deltas)
    sample_variance = squared_deviations / (count - 1)
    sample_standard_deviation = math.sqrt(sample_variance)
    standard_error = sample_standard_deviation / math.sqrt(count)
    t_critical = cast(float, FROZEN_THRESHOLDS["paired_t_critical"])
    lower_bound = mean_delta - t_critical * standard_error
    return {
        "n": count,
        "degrees_freedom": count - 1,
        "one_sided_confidence": FROZEN_THRESHOLDS["paired_one_sided_confidence"],
        "t_critical": t_critical,
        "sample_mean_delta": mean_delta,
        "sample_variance_delta": sample_variance,
        "sample_standard_deviation_delta": sample_standard_deviation,
        "standard_error_delta": standard_error,
        "one_sided_lower_bound": lower_bound,
        "sample_variance_formula": "sum((delta_i - mean_delta)^2) / (n - 1)",
        "lower_bound_formula": "mean_delta - t_critical(df=29,p=0.95) * sample_sd / sqrt(n)",
    }


def _aggregate_and_gates(
    metrics_by_seed: Sequence[Mapping[str, _RunMetrics]],
) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    if len(metrics_by_seed) != HELDOUT_SEED_COUNT:
        raise ValueError("frozen evidence aggregation requires exactly 30 paired lives")
    condition_metrics: list[dict[str, object]] = []
    for condition in FROZEN_CONDITION_ORDER:
        rewards = [metrics[condition].mean_reward for metrics in metrics_by_seed]
        recurrence = [metrics[condition].recurrence_entry_reward for metrics in metrics_by_seed]
        condition_metrics.append(
            {
                "condition": condition,
                "mean_reward": _mean(rewards),
                "minimum_seed_reward": min(rewards),
                "mean_recurrence_entry_reward": _mean(recurrence),
            }
        )
    metric_lookup = {cast(str, item["condition"]): item for item in condition_metrics}
    selective_reward_deltas = [
        metrics[SELECTIVE_FULL].mean_reward - metrics[WRITABLE_LRU].mean_reward
        for metrics in metrics_by_seed
    ]
    selective_recurrence_deltas = [
        metrics[SELECTIVE_FULL].recurrence_entry_reward
        - metrics[WRITABLE_LRU].recurrence_entry_reward
        for metrics in metrics_by_seed
    ]
    strict_lifecycle_pass_count = sum(
        metrics[SELECTIVE_FULL].strict_selective_lifecycle for metrics in metrics_by_seed
    )
    d_non_displacement_count = sum(
        metrics[SELECTIVE_FULL].d_non_displacement for metrics in metrics_by_seed
    )
    selective_reward_pass_count = sum(
        metrics[SELECTIVE_FULL].mean_reward
        >= cast(float, FROZEN_THRESHOLDS["minimum_selective_per_seed_reward"])
        for metrics in metrics_by_seed
    )
    selective_recurrence_pass_count = sum(
        metrics[SELECTIVE_FULL].recurrence_entry_reward
        >= cast(
            float,
            FROZEN_THRESHOLDS["minimum_selective_per_seed_recurrence_entry_reward"],
        )
        for metrics in metrics_by_seed
    )
    reward_pairwise_win_count = sum(delta > 0.0 for delta in selective_reward_deltas)
    recurrence_pairwise_win_count = sum(delta > 0.0 for delta in selective_recurrence_deltas)
    strict_run_validation_count = sum(
        run.strict_run_validation_passed for metrics in metrics_by_seed for run in metrics.values()
    )
    all_resources = all(
        run.resource_valid for metrics in metrics_by_seed for run in metrics.values()
    )
    all_synchronized = all(
        run.synchronized for metrics in metrics_by_seed for run in metrics.values()
    )
    all_selective_immutable = all(
        run.immutable
        for metrics in metrics_by_seed
        for condition, run in metrics.items()
        if condition != WRITABLE_LRU
    )
    both_learning_roles_update = all(
        metrics[condition].helper_updates > 0 and metrics[condition].beneficiary_updates > 0
        for metrics in metrics_by_seed
        for condition in (SELECTIVE_FULL, WRITABLE_LRU)
    )
    frozen_interventions_honored = all(
        metrics[HELPER_FROZEN].helper_writes == 0
        and metrics[BENEFICIARY_FROZEN].beneficiary_writes == 0
        for metrics in metrics_by_seed
    )
    disabled_controls = (
        HELPER_FROZEN,
        BENEFICIARY_FROZEN,
        CONSTANT_CHANNEL,
        SHUFFLED_CHANNEL,
    )
    disabled_control_commit_counts = {
        condition: sum(
            metrics[condition].helper_commits + metrics[condition].beneficiary_commits
            for metrics in metrics_by_seed
        )
        for condition in disabled_controls
    }
    reward_t = _paired_t_summary(selective_reward_deltas)
    recurrence_t = _paired_t_summary(selective_recurrence_deltas)
    aggregate: dict[str, object] = {
        "seed_count": len(metrics_by_seed),
        "strict_run_validation_count": strict_run_validation_count,
        "condition_metrics": condition_metrics,
        "mean_selective_reward_delta_vs_writable_lru": _mean(selective_reward_deltas),
        "mean_selective_recurrence_delta_vs_writable_lru": _mean(selective_recurrence_deltas),
        "paired_t_inference": {
            "reward_delta": reward_t,
            "recurrence_delta": recurrence_t,
        },
        "selective_reward_pass_count": selective_reward_pass_count,
        "selective_recurrence_pass_count": selective_recurrence_pass_count,
        "reward_pairwise_win_count": reward_pairwise_win_count,
        "recurrence_pairwise_win_count": recurrence_pairwise_win_count,
        "selective_strict_lifecycle_pass_count": strict_lifecycle_pass_count,
        "selective_d_non_displacement_count": d_non_displacement_count,
        "disabled_control_durable_commit_counts": disabled_control_commit_counts,
        "all_conditions_resource_constant_and_matched": all_resources,
        "all_conditions_lifecycle_synchronized": all_synchronized,
        "all_selective_conditions_durable_immutable": all_selective_immutable,
        "selective_and_writable_both_roles_learned": both_learning_roles_update,
        "frozen_role_interventions_honored": frozen_interventions_honored,
    }

    thresholds = FROZEN_THRESHOLDS
    selective = metric_lookup[SELECTIVE_FULL]
    gates = [
        _gate(
            "paired_life_count",
            len(metrics_by_seed),
            "==",
            thresholds["required_seed_count"],
            len(metrics_by_seed) == thresholds["required_seed_count"],
        ),
        _gate(
            "strict_run_validation_count",
            strict_run_validation_count,
            "==",
            thresholds["required_strict_run_validation_count"],
            strict_run_validation_count == thresholds["required_strict_run_validation_count"],
        ),
        _gate(
            "selective_mean_reward",
            cast(float, selective["mean_reward"]),
            ">=",
            thresholds["minimum_selective_mean_reward"],
            cast(float, selective["mean_reward"])
            >= cast(float, thresholds["minimum_selective_mean_reward"]),
        ),
        _gate(
            "selective_per_life_reward_pass_count",
            selective_reward_pass_count,
            ">=",
            thresholds["minimum_selective_per_seed_reward_pass_count"],
            selective_reward_pass_count
            >= thresholds["minimum_selective_per_seed_reward_pass_count"],
        ),
        _gate(
            "selective_recurrence_entry_reward",
            cast(float, selective["mean_recurrence_entry_reward"]),
            ">=",
            thresholds["minimum_selective_recurrence_entry_reward"],
            cast(float, selective["mean_recurrence_entry_reward"])
            >= cast(float, thresholds["minimum_selective_recurrence_entry_reward"]),
        ),
        _gate(
            "selective_per_life_recurrence_pass_count",
            selective_recurrence_pass_count,
            ">=",
            thresholds["minimum_selective_per_seed_recurrence_pass_count"],
            selective_recurrence_pass_count
            >= thresholds["minimum_selective_per_seed_recurrence_pass_count"],
        ),
        _gate(
            "selective_reward_delta_vs_writable_lru",
            cast(float, aggregate["mean_selective_reward_delta_vs_writable_lru"]),
            ">=",
            thresholds["minimum_mean_reward_delta_vs_writable_lru"],
            cast(float, aggregate["mean_selective_reward_delta_vs_writable_lru"])
            >= cast(float, thresholds["minimum_mean_reward_delta_vs_writable_lru"]),
        ),
        _gate(
            "selective_recurrence_delta_vs_writable_lru",
            cast(float, aggregate["mean_selective_recurrence_delta_vs_writable_lru"]),
            ">=",
            thresholds["minimum_recurrence_delta_vs_writable_lru"],
            cast(float, aggregate["mean_selective_recurrence_delta_vs_writable_lru"])
            >= cast(float, thresholds["minimum_recurrence_delta_vs_writable_lru"]),
        ),
        _gate(
            "reward_delta_one_sided_paired_t_lower_bound",
            cast(float, reward_t["one_sided_lower_bound"]),
            ">",
            0.0,
            cast(float, reward_t["one_sided_lower_bound"]) > 0.0,
        ),
        _gate(
            "recurrence_delta_one_sided_paired_t_lower_bound",
            cast(float, recurrence_t["one_sided_lower_bound"]),
            ">",
            0.0,
            cast(float, recurrence_t["one_sided_lower_bound"]) > 0.0,
        ),
        _gate(
            "reward_pairwise_win_count",
            reward_pairwise_win_count,
            ">=",
            thresholds["minimum_reward_pairwise_win_count"],
            reward_pairwise_win_count >= thresholds["minimum_reward_pairwise_win_count"],
        ),
        _gate(
            "recurrence_pairwise_win_count",
            recurrence_pairwise_win_count,
            ">=",
            thresholds["minimum_recurrence_pairwise_win_count"],
            recurrence_pairwise_win_count >= thresholds["minimum_recurrence_pairwise_win_count"],
        ),
    ]
    for condition in disabled_controls:
        mean_reward = cast(float, metric_lookup[condition]["mean_reward"])
        mean_recurrence = cast(
            float,
            metric_lookup[condition]["mean_recurrence_entry_reward"],
        )
        reward_maximum = cast(float, thresholds["maximum_disabled_control_mean_reward"])
        recurrence_maximum = cast(
            float,
            thresholds["maximum_disabled_control_recurrence_entry_reward"],
        )
        gates.extend(
            (
                _gate(
                    f"{condition}_mean_reward_ceiling",
                    mean_reward,
                    "<=",
                    reward_maximum,
                    mean_reward <= reward_maximum,
                ),
                _gate(
                    f"{condition}_mean_recurrence_ceiling",
                    mean_recurrence,
                    "<=",
                    recurrence_maximum,
                    mean_recurrence <= recurrence_maximum,
                ),
                _gate(
                    f"{condition}_durable_commit_count",
                    disabled_control_commit_counts[condition],
                    "==",
                    thresholds["required_disabled_control_durable_commit_count"],
                    disabled_control_commit_counts[condition]
                    == thresholds["required_disabled_control_durable_commit_count"],
                ),
            )
        )
    gates.extend(
        (
            _gate(
                "selective_strict_lifecycle_pass_count",
                strict_lifecycle_pass_count,
                ">=",
                thresholds["minimum_strict_lifecycle_pass_count"],
                strict_lifecycle_pass_count >= thresholds["minimum_strict_lifecycle_pass_count"],
            ),
            _gate(
                "selective_d_non_displacement_count",
                d_non_displacement_count,
                "==",
                thresholds["required_seed_count"],
                d_non_displacement_count == thresholds["required_seed_count"],
            ),
            _gate(
                "all_conditions_resource_constant_and_matched",
                all_resources,
                "is",
                True,
                all_resources,
            ),
            _gate(
                "all_conditions_lifecycle_synchronized",
                all_synchronized,
                "is",
                True,
                all_synchronized,
            ),
            _gate(
                "all_selective_conditions_durable_immutable",
                all_selective_immutable,
                "is",
                True,
                all_selective_immutable,
            ),
            _gate(
                "selective_and_writable_both_roles_learned",
                both_learning_roles_update,
                "is",
                True,
                both_learning_roles_update,
            ),
            _gate(
                "frozen_role_interventions_honored",
                frozen_interventions_honored,
                "is",
                True,
                frozen_interventions_honored,
            ),
        )
    )
    return aggregate, gates, all(cast(bool, gate["passed"]) for gate in gates)


def _validate_seed_records(
    records: object,
    plan_payload: Mapping[str, object],
    errors: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], bool] | None:
    seeds = plan_payload.get("seed_pairs")
    if not isinstance(records, list) or not isinstance(seeds, list) or len(records) != len(seeds):
        errors.append("seed_records must align exactly with the preregistered seeds")
        return None
    if len(records) != HELDOUT_SEED_COUNT:
        errors.append("evidence artifacts require exactly 30 paired seed records")
        return None
    metrics_by_seed: list[dict[str, _RunMetrics]] = []
    expected_seed_fields = {
        "schema_version",
        "seed_pair",
        "condition_order",
        "runs",
        "seed_record_digest_sha256",
    }
    for seed_index, (record_object, expected_seed) in enumerate(zip(records, seeds, strict=True)):
        label = f"seed record {seed_index}"
        if not isinstance(record_object, Mapping) or set(record_object) != expected_seed_fields:
            errors.append(f"{label} fields do not match compact primitives v1")
            continue
        record = record_object
        if record.get("schema_version") != COMPACT_PRIMITIVES_SCHEMA:
            errors.append(f"{label} schema mismatch")
        if record.get("seed_pair") != expected_seed:
            errors.append(f"{label} does not match its preregistered seed pair")
        if record.get("condition_order") != list(FROZEN_CONDITION_ORDER):
            errors.append(f"{label} condition order mismatch")
        digest_payload = {
            key: child for key, child in record.items() if key != "seed_record_digest_sha256"
        }
        if not _valid_sha256(record.get("seed_record_digest_sha256")) or record.get(
            "seed_record_digest_sha256"
        ) != _canonical_sha256(digest_payload):
            errors.append(f"{label} digest mismatch")
        runs = record.get("runs")
        if not isinstance(runs, list) or len(runs) != len(FROZEN_CONDITION_ORDER):
            errors.append(f"{label} runs do not match the frozen condition order")
            continue
        seed_metrics: dict[str, _RunMetrics] = {}
        for condition, run in zip(FROZEN_CONDITION_ORDER, runs, strict=True):
            parsed = _validate_compact_run(
                run,
                condition,
                f"{label} condition {condition}",
                errors,
            )
            if parsed is not None:
                seed_metrics[condition] = parsed
        if set(seed_metrics) == set(FROZEN_CONDITION_ORDER):
            metrics_by_seed.append(seed_metrics)
    if len(metrics_by_seed) != len(records):
        return None
    return _aggregate_and_gates(metrics_by_seed)


def _validate_operational_metadata(value: object, errors: list[str]) -> None:
    expected_fields = {
        "generated_at_utc",
        "wall_seconds",
        "python_version",
        "platform",
        "jax_version",
        "jaxlib_version",
        "numpy_version",
        "jax_backend",
        "jax_devices",
        "argv",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        errors.append("operational_metadata fields do not match strict v1")
        return
    if not _validate_utc_timestamp(value.get("generated_at_utc")):
        errors.append("operational generated_at_utc must be ISO-8601 UTC")
    wall = value.get("wall_seconds")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
    ):
        errors.append("operational wall_seconds must be finite and non-negative")
    for field in (
        "python_version",
        "platform",
        "jax_version",
        "jaxlib_version",
        "numpy_version",
        "jax_backend",
    ):
        if not isinstance(value.get(field), str) or not value.get(field):
            errors.append(f"operational {field} must be a non-empty string")
    for field in ("jax_devices", "argv"):
        items = value.get(field)
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item for item in items
        ):
            errors.append(f"operational {field} must be an array of non-empty strings")


def _operational_metadata(started: float) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - started,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "numpy_version": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "argv": list(sys.argv),
    }


def build_evidence_artifact_from_records(
    plan: Mapping[str, object],
    seed_records: Sequence[Mapping[str, object]],
    *,
    operational_metadata: Mapping[str, object],
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Build a valid acceptance or valid rejection from gate-sufficient records."""

    plan_validation = validate_preregistration_plan(plan, root=root, check_current_source=True)
    if not plan_validation.valid:
        raise ValueError(
            "exact validated preregistration required: " + "; ".join(plan_validation.errors)
        )
    payload = cast(Mapping[str, object], plan["scientific_payload"])
    errors: list[str] = []
    evaluation = _validate_seed_records(list(seed_records), payload, errors)
    _validate_operational_metadata(operational_metadata, errors)
    if errors or evaluation is None:
        raise ValueError("invalid compact evidence records: " + "; ".join(errors))
    completion_source = source_snapshot(root)
    if not _strict_json_equal(completion_source, payload["source_sha256"]):
        raise ValueError("source closure changed between preregistration and artifact completion")
    aggregate, gates, frozen_gates_passed = evaluation
    scientifically_eligible = payload["protocol_kind"] == "protected_heldout"
    accepted = scientifically_eligible and frozen_gates_passed
    outcome = (
        "accepted"
        if accepted
        else "valid_rejection"
        if scientifically_eligible
        else "manual_nonpromoting"
    )
    plan_digest = cast(Mapping[str, object], plan["scientific_digest"])["sha256"]
    scientific_payload: dict[str, object] = {
        "plan_schema_version": PREREGISTRATION_SCHEMA,
        "plan_scientific_sha256": plan_digest,
        "protocol_kind": payload["protocol_kind"],
        "protocol_status": PROTOCOL_STATUS,
        "namespace": payload["namespace"],
        "executed": True,
        "condition_order": list(FROZEN_CONDITION_ORDER),
        "config": frozen_protocol_config(),
        "thresholds": copy.deepcopy(FROZEN_THRESHOLDS),
        "source_sha256": copy.deepcopy(payload["source_sha256"]),
        "completion_source_sha256": completion_source,
        "seed_records": copy.deepcopy(list(seed_records)),
        "aggregate_metrics": aggregate,
        "acceptance_checks": gates,
        "frozen_gates_passed": frozen_gates_passed,
        "eligible_for_scientific_review": scientifically_eligible,
        "accepted": accepted,
        "outcome": outcome,
        "claim_scope": CLAIM_SCOPE,
        "retention_limitation": RETENTION_LIMITATION,
        "time_attestation_limitation": TIME_ATTESTATION_LIMITATION,
        "replay_scope": REPLAY_SCOPE,
        "promotion_policy": PROMOTION_POLICY,
        "compact_record_trust_boundary": COMPACT_RECORD_TRUST_BOUNDARY,
        "automatic_promotion_allowed": False,
    }
    return {
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA,
        "scientific_payload": scientific_payload,
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": _canonical_sha256(scientific_payload),
        },
        "operational_metadata": dict(operational_metadata),
    }


def execute_validated_plan(
    plan: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Execute only an exact, source-current, explicitly supplied plan."""

    if not PROTECTED_EXECUTION_ENABLED:
        raise ValueError(
            "execution refused; draft protocol is disabled pending the settled "
            "factorial/boundary design"
        )
    started = time.monotonic()
    validation = validate_preregistration_plan(plan, root=root, check_current_source=True)
    if not validation.valid:
        raise ValueError(
            "execution refused; exact validated plan required: " + "; ".join(validation.errors)
        )
    payload = cast(Mapping[str, object], plan["scientific_payload"])
    if payload.get("protocol_kind") != "protected_heldout":
        raise ValueError(
            "execution refused; manual plans are schema tests only and cannot emit evidence"
        )
    seeds = cast(list[Mapping[str, object]], payload["seed_pairs"])
    config = _frozen_evaluator_config()
    records: list[dict[str, object]] = []
    for seed_payload in seeds:
        seed_pair = HiddenRegimeSeedPair(
            namespace=cast(str, seed_payload["namespace"]),
            index=cast(int, seed_payload["index"]),
            world_seed=cast(int, seed_payload["world_seed"]),
            learner_seed=cast(int, seed_payload["learner_seed"]),
        )
        runs = tuple(
            run_hidden_regime_condition(
                cast(Any, condition),
                seed_pair=seed_pair,
                config=config,
            )
            for condition in FROZEN_CONDITION_ORDER
        )
        records.append(_compact_seed_record(seed_pair, runs))
    return build_evidence_artifact_from_records(
        plan,
        records,
        operational_metadata=_operational_metadata(started),
        root=root,
    )


def validate_evidence_artifact(
    artifact: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    deterministic_rerun: bool = False,
) -> EvidenceValidation:
    """Validate compact primitives, every gate, digests, source, and optional rerun."""

    errors: list[str] = []
    _validate_finite_tree(artifact, errors)
    plan_validation = validate_preregistration_plan(plan, root=root, check_current_source=True)
    if not plan_validation.valid:
        errors.extend(f"plan: {error}" for error in plan_validation.errors)
    if set(artifact) != {
        "schema_version",
        "scientific_payload",
        "scientific_digest",
        "operational_metadata",
    }:
        errors.append("artifact top-level fields do not match strict v1")
    if artifact.get("schema_version") != EVIDENCE_ARTIFACT_SCHEMA:
        errors.append("artifact schema_version mismatch")
    payload = artifact.get("scientific_payload")
    _validate_digest(
        artifact.get("scientific_digest"),
        payload,
        scope="$.scientific_payload",
        label="artifact",
        errors=errors,
    )
    _validate_operational_metadata(artifact.get("operational_metadata"), errors)
    expected_payload_fields = {
        "plan_schema_version",
        "plan_scientific_sha256",
        "protocol_kind",
        "protocol_status",
        "namespace",
        "executed",
        "condition_order",
        "config",
        "thresholds",
        "source_sha256",
        "completion_source_sha256",
        "seed_records",
        "aggregate_metrics",
        "acceptance_checks",
        "frozen_gates_passed",
        "eligible_for_scientific_review",
        "accepted",
        "outcome",
        "claim_scope",
        "retention_limitation",
        "time_attestation_limitation",
        "replay_scope",
        "promotion_policy",
        "compact_record_trust_boundary",
        "automatic_promotion_allowed",
    }
    if not isinstance(payload, Mapping):
        errors.append("artifact scientific_payload must be a mapping")
        return EvidenceValidation(False, False, False, tuple(errors))
    if set(payload) != expected_payload_fields:
        errors.append("artifact scientific_payload fields do not match strict v1")
    plan_payload_object = plan.get("scientific_payload")
    plan_payload = (
        cast(Mapping[str, object], plan_payload_object)
        if isinstance(plan_payload_object, Mapping)
        else {}
    )
    plan_digest_object = plan.get("scientific_digest")
    expected_plan_digest = (
        plan_digest_object.get("sha256") if isinstance(plan_digest_object, Mapping) else None
    )
    exact_fields = {
        "plan_schema_version": PREREGISTRATION_SCHEMA,
        "plan_scientific_sha256": expected_plan_digest,
        "protocol_kind": plan_payload.get("protocol_kind"),
        "protocol_status": PROTOCOL_STATUS,
        "namespace": plan_payload.get("namespace"),
        "executed": True,
        "condition_order": list(FROZEN_CONDITION_ORDER),
        "config": frozen_protocol_config(),
        "thresholds": FROZEN_THRESHOLDS,
        "source_sha256": plan_payload.get("source_sha256"),
        "completion_source_sha256": plan_payload.get("source_sha256"),
        "claim_scope": CLAIM_SCOPE,
        "retention_limitation": RETENTION_LIMITATION,
        "time_attestation_limitation": TIME_ATTESTATION_LIMITATION,
        "replay_scope": REPLAY_SCOPE,
        "promotion_policy": PROMOTION_POLICY,
        "compact_record_trust_boundary": COMPACT_RECORD_TRUST_BOUNDARY,
        "automatic_promotion_allowed": False,
    }
    for field, expected in exact_fields.items():
        if not _strict_json_equal(payload.get(field), expected):
            errors.append(f"artifact {field} mismatch")

    evaluation = _validate_seed_records(payload.get("seed_records"), plan_payload, errors)
    recomputed_gates_passed = False
    scientifically_eligible = plan_payload.get("protocol_kind") == "protected_heldout"
    recomputed_accepted = False
    if evaluation is not None:
        aggregate, gates, recomputed_gates_passed = evaluation
        recomputed_accepted = scientifically_eligible and recomputed_gates_passed
        if not _strict_json_equal(payload.get("aggregate_metrics"), aggregate):
            errors.append("aggregate_metrics do not recompute from compact primitives")
        if not _strict_json_equal(payload.get("acceptance_checks"), gates):
            errors.append("acceptance_checks do not recompute from compact primitives")
        if payload.get("frozen_gates_passed") is not recomputed_gates_passed:
            errors.append("frozen_gates_passed does not equal the conjunction of gates")
        if payload.get("eligible_for_scientific_review") is not scientifically_eligible:
            errors.append("scientific-review eligibility does not match the plan kind")
        if payload.get("accepted") is not recomputed_accepted:
            errors.append("accepted does not combine eligibility with the frozen gates")
        expected_outcome = (
            "accepted"
            if recomputed_accepted
            else "valid_rejection"
            if scientifically_eligible
            else "manual_nonpromoting"
        )
        if payload.get("outcome") != expected_outcome:
            errors.append("outcome does not match recomputed gate result")

    rerun_verified = False
    if deterministic_rerun and not errors:
        if not scientifically_eligible:
            errors.append("deterministic evidence rerun refuses manual nonpromoting plans")
            return EvidenceValidation(False, False, False, tuple(errors))
        rerun = execute_validated_plan(plan, root=root)
        rerun_payload = cast(Mapping[str, object], rerun["scientific_payload"])
        for field in (
            "seed_records",
            "aggregate_metrics",
            "acceptance_checks",
            "accepted",
            "outcome",
        ):
            if not _strict_json_equal(payload.get(field), rerun_payload.get(field)):
                errors.append(f"same-backend deterministic rerun differs at {field}")
        rerun_verified = not errors
    return EvidenceValidation(
        not errors,
        recomputed_accepted if not errors else False,
        rerun_verified,
        tuple(errors),
    )


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one new protocol document without ever overwriting an artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(strict_json_text(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregister, run, or validate hidden-regime signaling evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create-plan",
        help="derive and write the protected held-out plan without executing it",
    )
    create.add_argument("--preregistered-at-utc", required=True)
    create.add_argument("--output", type=Path, required=True)

    validate_plan_parser = subparsers.add_parser("validate-plan")
    validate_plan_parser.add_argument("--plan", type=Path, required=True)

    run = subparsers.add_parser(
        "run",
        help="execute only the exact source-current plan supplied by --plan",
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    validate_artifact_parser = subparsers.add_parser("validate-artifact")
    validate_artifact_parser.add_argument("--plan", type=Path, required=True)
    validate_artifact_parser.add_argument("--artifact", type=Path, required=True)
    validate_artifact_parser.add_argument(
        "--deterministic-rerun",
        action="store_true",
        help="repeat all runs on the current backend and compare compact primitives",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI with no implicit execution path and exclusive-create output semantics."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "create-plan":
            plan = build_preregistration_plan(
                namespace=RESERVED_HELDOUT_NAMESPACE,
                seed_count=HELDOUT_SEED_COUNT,
                preregistered_at_utc=cast(str, args.preregistered_at_utc),
            )
            _write_new_json(cast(Path, args.output), plan)
            sys.stdout.write("wrote an unexecuted preregistration plan\n")
            return 0
        if args.command == "validate-plan":
            plan = strict_json_load(cast(Path, args.plan))
            preregistration_validation = validate_preregistration_plan(plan)
            if preregistration_validation.valid:
                sys.stdout.write("valid unexecuted preregistration\n")
                return 0
            sys.stderr.write(
                "invalid preregistration: " + "; ".join(preregistration_validation.errors) + "\n"
            )
            return 2
        if args.command == "run":
            output = cast(Path, args.output)
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {output}")
            plan = strict_json_load(cast(Path, args.plan))
            artifact = execute_validated_plan(plan)
            _write_new_json(output, artifact)
            outcome = cast(Mapping[str, object], artifact["scientific_payload"])["outcome"]
            sys.stdout.write(f"wrote hidden-regime evidence outcome={outcome}\n")
            return 0 if outcome == "accepted" else 1
        if args.command == "validate-artifact":
            plan = strict_json_load(cast(Path, args.plan))
            artifact = strict_json_load(cast(Path, args.artifact))
            evidence_validation = validate_evidence_artifact(
                artifact,
                plan,
                deterministic_rerun=cast(bool, args.deterministic_rerun),
            )
            if not evidence_validation.valid:
                sys.stderr.write(
                    "invalid artifact: " + "; ".join(evidence_validation.errors) + "\n"
                )
                return 2
            sys.stdout.write(
                "valid artifact outcome="
                + ("accepted" if evidence_validation.accepted else "valid_rejection")
                + "\n"
            )
            return 0 if evidence_validation.accepted else 1
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"hidden-regime evidence command refused: {exc}\n")
        return 2
    raise AssertionError("argparse returned an unknown command")


if __name__ == "__main__":  # pragma: no cover - exercised as a module entry point
    raise SystemExit(main())
