"""Fail-closed manifest for the complete continual-prototype scorecard.

The ordinary evidence registry intentionally covers only a small set of narrow
claims.  This module provides the separate, stricter contract needed by the
18-row final scorecard in ``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``.  It does
not infer completion from tests, filenames, or booleans stored in artifacts.
Every required evidence role must point to immutable bytes and a source-pinned
validator, and that validator must reconstruct the exact role for the same
prototype configuration as an accepted frozen L3 result.

No default evidence index is supplied.  Until independently versioned
artifacts and validators are registered for every role, the complete-prototype
claim is unavailable by construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

COMPLETE_PROTOTYPE_INDEX_SCHEMA = "alberta.complete-prototype-evidence-index.v1"
COMPLETE_PROTOTYPE_MANIFEST_SCHEMA = "alberta.complete-prototype-manifest.v1"
COMPLETE_PROTOTYPE_CONTRACT_VERSION = "alberta.complete-prototype-scorecard.v1"

EvidenceStatus = Literal["accepted", "valid-rejection", "not-run", "invalid"]
RowStatus = Literal["accepted", "valid-rejection", "not-run", "invalid"]
OverallStatus = Literal["accepted", "not-ready", "invalid"]

_HEX_DIGITS = frozenset("0123456789abcdef")
_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "prototype_configuration_sha256",
        "component_policy",
        "rows",
    }
)
_COMPONENT_POLICY_FIELDS = frozenset(
    {"paper_delight_enabled", "kondo_enabled"}
)
_ROW_FIELDS = frozenset({"property_id", "evidence"})
_REFERENCE_FIELDS = frozenset(
    {
        "role",
        "relative_path",
        "artifact_sha256",
        "expected_schema",
        "expected_protocol_sha256",
        "expected_scientific_digest_sha256",
        "validator_id",
        "validator_source_sha256",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "prototype_configuration_sha256",
        "component_policy",
        "overall_status",
        "all_18_rows_registered",
        "all_18_rows_accepted",
        "rows",
        "interpretation",
        "manifest_sha256",
    }
)
_MANIFEST_ROW_FIELDS = frozenset(
    {"property_id", "label", "required_roles", "status", "evidence", "errors"}
)
_INTERPRETATION = (
    "accepted means every exact scorecard role was reconstructed by a "
    "source-pinned strict validator as accepted frozen L3 evidence; it "
    "establishes a strong prototype, not general intelligence"
)


@dataclass(frozen=True)
class ScorecardRowContract:
    """One exact final-scorecard property and its base evidence roles."""

    property_id: str
    label: str
    required_roles: tuple[str, ...]


SCORECARD_ROWS: tuple[ScorecardRowContract, ...] = (
    ScorecardRowContract(
        "continuing_operation",
        "Continuing operation",
        ("held_out_uninterrupted_lifetime",),
    ),
    ScorecardRowContract(
        "temporal_resource_bounds",
        "Temporal/resource bounds",
        ("whole_agent_fixed_resources", "control_deadline_latency"),
    ),
    ScorecardRowContract(
        "plasticity",
        "Plasticity",
        ("late_regime_adaptation",),
    ),
    ScorecardRowContract(
        "retention",
        "Retention",
        ("prespecified_forgetting_and_worst_task",),
    ),
    ScorecardRowContract(
        "transfer",
        "Transfer",
        ("paired_held_out_forward_transfer",),
    ),
    ScorecardRowContract(
        "state_construction",
        "State construction",
        ("partially_observable_forager", "robot_simulation_state"),
    ),
    ScorecardRowContract(
        "prediction",
        "Prediction",
        ("multi_timescale_gvf_calibration", "multi_timescale_gvf_usefulness"),
    ),
    ScorecardRowContract(
        "world_model",
        "World model",
        ("model_retention", "uncertainty_calibration", "rollout_validation"),
    ),
    ScorecardRowContract(
        "planning",
        "Planning",
        ("matched_primitive_option_search",),
    ),
    ScorecardRowContract(
        "exploration",
        "Exploration",
        ("coverage_and_return", "noisy_tv_resistance"),
    ),
    ScorecardRowContract(
        "feature_lifecycle",
        "Feature lifecycle",
        ("bounded_discovery_vs_random_replacement",),
    ),
    ScorecardRowContract(
        "skill_lifecycle",
        "Skill lifecycle",
        ("discovered_option_control", "unlabeled_composition_and_retirement"),
    ),
    ScorecardRowContract(
        "candidate_update_audit",
        "Candidate-update audit / optional paper DG and Kondo",
        ("audited_update_realized_outcomes",),
    ),
    ScorecardRowContract(
        "experiential_memory",
        "Experiential memory",
        ("fixed_capacity_transfer", "negative_transfer_resistance"),
    ),
    ScorecardRowContract(
        "intelligence_amplification",
        "IA",
        ("causal_partner_benefit_under_drift_and_cost",),
    ),
    ScorecardRowContract(
        "checkpointing",
        "Checkpointing",
        ("all_enabled_resume_parity",),
    ),
    ScorecardRowContract(
        "safety",
        "Safety",
        ("zero_hard_envelope_violations", "fallback_and_rollback_drills"),
    ),
    ScorecardRowContract(
        "reproducibility",
        "Reproducibility",
        (
            "clean_checkout_reproduction",
            "raw_artifacts_hashes_seeds_intervals",
            "negative_results_publication",
        ),
    ),
)

if len(SCORECARD_ROWS) != 18 or len(
    {row.property_id for row in SCORECARD_ROWS}
) != 18:
    raise RuntimeError("the complete-prototype scorecard must contain exactly 18 rows")


@dataclass(frozen=True)
class CompletePrototypeValidationReceipt:
    """Normalized result returned by a trusted artifact validator."""

    valid: bool
    accepted: bool
    evidence_role: str
    prototype_configuration_sha256: str
    schema_version: str
    protocol_sha256: str
    scientific_digest_sha256: str
    evidence_class: Literal["scientific", "development", "unit", "smoke"]
    evidence_level: Literal["L0", "L1", "L2", "L3"]
    frozen_protocol_valid: bool
    untouched_held_out_seeds_valid: bool
    source_closure_valid: bool
    errors: tuple[str, ...] = ()


ArtifactValidator = Callable[[Path], CompletePrototypeValidationReceipt]


@dataclass(frozen=True)
class RegisteredCompletePrototypeValidator:
    """Trusted validator implementation and its complete source closure."""

    validator_id: str
    source_paths: tuple[Path, ...]
    validate: ArtifactValidator

    def __post_init__(self) -> None:
        if type(self.validator_id) is not str or not self.validator_id:
            raise ValueError("validator_id must be a nonempty exact string")
        if type(self.source_paths) is not tuple or not self.source_paths:
            raise ValueError("validator source_paths must be a nonempty tuple")
        normalized = tuple(
            _strict_relative_path(path, name="validator source")
            for path in self.source_paths
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("validator source_paths must be unique")
        if not callable(self.validate):
            raise TypeError("validator validate must be callable")


def _strict_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _strict_relative_path(value: object, *, name: str) -> Path:
    if isinstance(value, Path):
        text = value.as_posix()
    elif type(value) is str:
        text = value
    else:
        raise TypeError(f"{name} must be an exact string or Path")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or text != pure.as_posix()
        or "\\" in text
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{name} must be a canonical repository-relative path")
    return Path(*pure.parts)


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{name} keys must be exact strings")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    payload: Mapping[str, object],
    fields: frozenset[str],
    *,
    name: str,
) -> None:
    actual = frozenset(payload)
    if actual != fields:
        raise ValueError(
            f"{name} fields are not exact; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_under_root(repo_root: Path, relative_path: Path, *, name: str) -> Path:
    resolved = (repo_root / relative_path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"{name} resolves outside repo_root")
    return resolved


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _required_roles(
    row: ScorecardRowContract,
    *,
    paper_delight_enabled: bool,
    kondo_enabled: bool,
) -> tuple[str, ...]:
    roles = row.required_roles
    if row.property_id == "candidate_update_audit":
        if paper_delight_enabled:
            roles += ("paper_delight_actor_learning_and_guardrails",)
        if kondo_enabled:
            roles += ("kondo_measured_compute_and_guardrails",)
    return roles


def empty_complete_prototype_evidence_index(
    *,
    prototype_configuration_sha256: str,
    paper_delight_enabled: bool,
    kondo_enabled: bool,
) -> dict[str, object]:
    """Return the exact empty index shape; it is necessarily not ready."""

    digest = _strict_sha256(
        prototype_configuration_sha256,
        name="prototype_configuration_sha256",
    )
    if type(paper_delight_enabled) is not bool or type(kondo_enabled) is not bool:
        raise TypeError("component-policy flags must be exact booleans")
    return {
        "schema_version": COMPLETE_PROTOTYPE_INDEX_SCHEMA,
        "contract_version": COMPLETE_PROTOTYPE_CONTRACT_VERSION,
        "prototype_configuration_sha256": digest,
        "component_policy": {
            "paper_delight_enabled": paper_delight_enabled,
            "kondo_enabled": kondo_enabled,
        },
        "rows": [
            {"property_id": row.property_id, "evidence": []}
            for row in SCORECARD_ROWS
        ],
    }


def _parse_index(
    index: Mapping[str, object],
) -> tuple[str, bool, bool, Sequence[Mapping[str, object]]]:
    _require_exact_fields(index, _INDEX_FIELDS, name="evidence index")
    if index["schema_version"] != COMPLETE_PROTOTYPE_INDEX_SCHEMA:
        raise ValueError("unknown complete-prototype evidence-index schema")
    if index["contract_version"] != COMPLETE_PROTOTYPE_CONTRACT_VERSION:
        raise ValueError("unknown complete-prototype scorecard contract")
    configuration_sha256 = _strict_sha256(
        index["prototype_configuration_sha256"],
        name="prototype_configuration_sha256",
    )
    policy = _strict_mapping(index["component_policy"], name="component_policy")
    _require_exact_fields(policy, _COMPONENT_POLICY_FIELDS, name="component_policy")
    paper_delight_enabled = policy["paper_delight_enabled"]
    kondo_enabled = policy["kondo_enabled"]
    if type(paper_delight_enabled) is not bool or type(kondo_enabled) is not bool:
        raise TypeError("component-policy flags must be exact booleans")
    rows = index["rows"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence")
    if len(rows) != len(SCORECARD_ROWS):
        raise ValueError("the evidence index must contain exactly 18 scorecard rows")
    parsed_rows: list[Mapping[str, object]] = []
    for ordinal, (raw_row, contract) in enumerate(zip(rows, SCORECARD_ROWS, strict=True)):
        row = _strict_mapping(raw_row, name=f"rows[{ordinal}]")
        _require_exact_fields(row, _ROW_FIELDS, name=f"rows[{ordinal}]")
        if row["property_id"] != contract.property_id:
            raise ValueError("scorecard rows must use the canonical order and identities")
        parsed_rows.append(row)
    return (
        configuration_sha256,
        paper_delight_enabled,
        kondo_enabled,
        tuple(parsed_rows),
    )


def _validator_source_status(
    repo_root: Path,
    registered: RegisteredCompletePrototypeValidator,
    pinned: Mapping[str, object],
) -> tuple[bool, dict[str, str], tuple[str, ...]]:
    expected_paths = tuple(path.as_posix() for path in registered.source_paths)
    if tuple(sorted(pinned)) != tuple(sorted(expected_paths)):
        return False, {}, ("validator source-path closure does not match registry",)
    current: dict[str, str] = {}
    errors: list[str] = []
    for relative_path in registered.source_paths:
        key = relative_path.as_posix()
        try:
            expected = _strict_sha256(
                pinned[key],
                name=f"validator_source_sha256[{key!r}]",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        try:
            absolute = _resolve_under_root(
                repo_root,
                relative_path,
                name=f"validator source {key}",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not absolute.is_file():
            errors.append(f"validator source is missing: {key}")
            continue
        actual = _sha256_file(absolute)
        current[key] = actual
        if actual != expected:
            errors.append(f"validator source hash mismatch: {key}")
    return not errors, current, tuple(errors)


def _evaluate_reference(
    *,
    repo_root: Path,
    role: str,
    prototype_configuration_sha256: str,
    raw_reference: object,
    validators: Mapping[str, RegisteredCompletePrototypeValidator],
) -> dict[str, object]:
    reference = _strict_mapping(raw_reference, name=f"evidence[{role!r}]")
    _require_exact_fields(reference, _REFERENCE_FIELDS, name=f"evidence[{role!r}]")
    if reference["role"] != role:
        raise ValueError("evidence roles must be canonical and ordered")
    relative_path = _strict_relative_path(
        reference["relative_path"],
        name=f"evidence[{role!r}].relative_path",
    )
    artifact_sha256 = _strict_sha256(
        reference["artifact_sha256"],
        name=f"evidence[{role!r}].artifact_sha256",
    )
    expected_protocol_sha256 = _strict_sha256(
        reference["expected_protocol_sha256"],
        name=f"evidence[{role!r}].expected_protocol_sha256",
    )
    expected_scientific_digest_sha256 = _strict_sha256(
        reference["expected_scientific_digest_sha256"],
        name=f"evidence[{role!r}].expected_scientific_digest_sha256",
    )
    expected_schema = reference["expected_schema"]
    validator_id = reference["validator_id"]
    if type(expected_schema) is not str or not expected_schema:
        raise ValueError("expected_schema must be a nonempty exact string")
    if type(validator_id) is not str or not validator_id:
        raise ValueError("validator_id must be a nonempty exact string")
    registered = validators.get(validator_id)
    base: dict[str, object] = {
        "role": role,
        "relative_path": relative_path.as_posix(),
        "expected_artifact_sha256": artifact_sha256,
        "expected_schema": expected_schema,
        "expected_protocol_sha256": expected_protocol_sha256,
        "expected_scientific_digest_sha256": expected_scientific_digest_sha256,
        "validator_id": validator_id,
    }
    if registered is None:
        return {
            **base,
            "status": "invalid",
            "errors": ["validator_id is not present in the trusted registry"],
        }
    pinned_sources = _strict_mapping(
        reference["validator_source_sha256"],
        name=f"evidence[{role!r}].validator_source_sha256",
    )
    source_valid, current_sources, source_errors = _validator_source_status(
        repo_root,
        registered,
        pinned_sources,
    )
    base["validator_source_sha256"] = current_sources
    if not source_valid:
        return {**base, "status": "invalid", "errors": list(source_errors)}
    try:
        artifact_path = _resolve_under_root(
            repo_root,
            relative_path,
            name=f"artifact {relative_path.as_posix()}",
        )
    except ValueError as exc:
        return {**base, "status": "invalid", "errors": [str(exc)]}
    if not artifact_path.is_file():
        return {**base, "status": "not-run", "errors": ["artifact is missing"]}
    current_artifact_sha256 = _sha256_file(artifact_path)
    base["artifact_sha256"] = current_artifact_sha256
    if current_artifact_sha256 != artifact_sha256:
        return {
            **base,
            "status": "invalid",
            "errors": ["artifact bytes do not match the pinned SHA-256 digest"],
        }
    try:
        receipt = registered.validate(artifact_path)
    except Exception as exc:  # validators are a deliberate fail-closed boundary
        return {
            **base,
            "status": "invalid",
            "errors": [f"validator raised {type(exc).__name__}: {exc}"],
        }
    if type(receipt) is not CompletePrototypeValidationReceipt:
        return {
            **base,
            "status": "invalid",
            "errors": ["validator returned the wrong receipt type"],
        }
    boolean_fields = (
        "valid",
        "accepted",
        "frozen_protocol_valid",
        "untouched_held_out_seeds_valid",
        "source_closure_valid",
    )
    if any(type(getattr(receipt, field)) is not bool for field in boolean_fields):
        return {
            **base,
            "status": "invalid",
            "errors": ["validator receipt boolean fields must be exact booleans"],
        }
    if (
        type(receipt.schema_version) is not str
        or type(receipt.evidence_role) is not str
        or type(receipt.prototype_configuration_sha256) is not str
        or type(receipt.protocol_sha256) is not str
        or type(receipt.scientific_digest_sha256) is not str
        or type(receipt.evidence_class) is not str
        or type(receipt.evidence_level) is not str
        or type(receipt.errors) is not tuple
        or any(type(error) is not str for error in receipt.errors)
    ):
        return {
            **base,
            "status": "invalid",
            "errors": ["validator receipt field types are not exact"],
        }
    receipt_errors: list[str] = list(receipt.errors)
    if receipt.evidence_role != role:
        receipt_errors.append("validated artifact evidence role does not match the index")
    try:
        receipt_configuration_digest = _strict_sha256(
            receipt.prototype_configuration_sha256,
            name="receipt prototype_configuration_sha256",
        )
        scientific_digest = _strict_sha256(
            receipt.scientific_digest_sha256,
            name="receipt scientific_digest_sha256",
        )
        protocol_digest = _strict_sha256(
            receipt.protocol_sha256,
            name="receipt protocol_sha256",
        )
    except (TypeError, ValueError) as exc:
        receipt_errors.append(str(exc))
        receipt_configuration_digest = ""
        scientific_digest = ""
        protocol_digest = ""
    if receipt_configuration_digest != prototype_configuration_sha256:
        receipt_errors.append(
            "validated prototype configuration does not match the evidence index"
        )
    if receipt.schema_version != expected_schema:
        receipt_errors.append("validated artifact schema does not match the index")
    if protocol_digest != expected_protocol_sha256:
        receipt_errors.append("validated frozen-protocol digest does not match the index")
    if scientific_digest != expected_scientific_digest_sha256:
        receipt_errors.append("validated scientific digest does not match the index")
    if not receipt.valid:
        receipt_errors.append("strict artifact validator rejected the artifact")
    structural_valid = (
        receipt.valid
        and receipt.frozen_protocol_valid
        and receipt.untouched_held_out_seeds_valid
        and receipt.source_closure_valid
        and receipt.evidence_class == "scientific"
        and receipt.evidence_level == "L3"
        and not receipt_errors
    )
    if structural_valid and receipt.accepted:
        status: EvidenceStatus = "accepted"
    elif structural_valid:
        status = "valid-rejection"
    else:
        status = "invalid"
    return {
        **base,
        "status": status,
        "evidence_role": receipt.evidence_role,
        "prototype_configuration_sha256": receipt_configuration_digest,
        "schema_version": receipt.schema_version,
        "protocol_sha256": protocol_digest,
        "scientific_digest_sha256": scientific_digest,
        "evidence_class": receipt.evidence_class,
        "evidence_level": receipt.evidence_level,
        "frozen_protocol_valid": receipt.frozen_protocol_valid,
        "untouched_held_out_seeds_valid": receipt.untouched_held_out_seeds_valid,
        "source_closure_valid": receipt.source_closure_valid,
        "accepted": receipt.accepted,
        "errors": receipt_errors,
    }


def build_complete_prototype_manifest(
    repo_root: str | Path,
    index: Mapping[str, object],
    *,
    validators: Mapping[str, RegisteredCompletePrototypeValidator],
) -> dict[str, object]:
    """Validate an exact evidence index and return the 18-row manifest."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    configuration_sha256, delight_enabled, kondo_enabled, indexed_rows = _parse_index(index)
    if not isinstance(validators, Mapping):
        raise TypeError("validators must be a mapping")
    for key, validator in validators.items():
        if type(key) is not str or type(validator) is not RegisteredCompletePrototypeValidator:
            raise TypeError("validator registry entries have invalid types")
        if key != validator.validator_id:
            raise ValueError("validator registry key does not match validator_id")

    row_records: list[dict[str, object]] = []
    for contract, indexed_row in zip(SCORECARD_ROWS, indexed_rows, strict=True):
        required_roles = _required_roles(
            contract,
            paper_delight_enabled=delight_enabled,
            kondo_enabled=kondo_enabled,
        )
        raw_evidence = indexed_row["evidence"]
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            raise TypeError(f"{contract.property_id} evidence must be a sequence")
        supplied_roles: list[object] = []
        for item in raw_evidence:
            mapping = _strict_mapping(item, name=f"{contract.property_id} evidence item")
            supplied_roles.append(mapping.get("role"))
        if tuple(supplied_roles) != required_roles:
            if len(raw_evidence) == 0:
                row_records.append(
                    {
                        "property_id": contract.property_id,
                        "label": contract.label,
                        "required_roles": list(required_roles),
                        "status": "not-run",
                        "evidence": [],
                        "errors": ["required evidence roles are not registered"],
                    }
                )
                continue
            raise ValueError(
                f"{contract.property_id} evidence roles must be exact and ordered"
            )
        evidence_records = [
            _evaluate_reference(
                repo_root=root,
                role=role,
                prototype_configuration_sha256=configuration_sha256,
                raw_reference=raw_reference,
                validators=validators,
            )
            for role, raw_reference in zip(required_roles, raw_evidence, strict=True)
        ]
        statuses = cast(list[EvidenceStatus], [record["status"] for record in evidence_records])
        if any(status == "invalid" for status in statuses):
            row_status: RowStatus = "invalid"
        elif any(status == "not-run" for status in statuses):
            row_status = "not-run"
        elif any(status == "valid-rejection" for status in statuses):
            row_status = "valid-rejection"
        else:
            row_status = "accepted"
        row_records.append(
            {
                "property_id": contract.property_id,
                "label": contract.label,
                "required_roles": list(required_roles),
                "status": row_status,
                "evidence": evidence_records,
                "errors": [],
            }
        )

    row_statuses = cast(list[RowStatus], [row["status"] for row in row_records])
    if any(status == "invalid" for status in row_statuses):
        overall_status: OverallStatus = "invalid"
    elif all(status == "accepted" for status in row_statuses):
        overall_status = "accepted"
    else:
        overall_status = "not-ready"
    payload: dict[str, object] = {
        "schema_version": COMPLETE_PROTOTYPE_MANIFEST_SCHEMA,
        "contract_version": COMPLETE_PROTOTYPE_CONTRACT_VERSION,
        "prototype_configuration_sha256": configuration_sha256,
        "component_policy": {
            "paper_delight_enabled": delight_enabled,
            "kondo_enabled": kondo_enabled,
        },
        "overall_status": overall_status,
        "all_18_rows_registered": all(
            len(cast(Sequence[object], row["evidence"])) > 0 for row in row_records
        ),
        "all_18_rows_accepted": all(
            status == "accepted" for status in row_statuses
        ),
        "rows": row_records,
        "interpretation": _INTERPRETATION,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_complete_prototype_manifest(
    manifest: Mapping[str, object],
) -> tuple[str, ...]:
    """Reconstruct one built manifest's structure and self-digest.

    This validates a manifest produced by :func:`build_complete_prototype_manifest`.
    It does not replace rerunning the trusted validators against live artifact
    and source bytes.
    """

    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ("manifest must be a mapping",)
    if frozenset(manifest) != _MANIFEST_FIELDS:
        return ("manifest fields are not exact",)
    if manifest.get("schema_version") != COMPLETE_PROTOTYPE_MANIFEST_SCHEMA:
        errors.append("manifest schema is invalid")
    if manifest.get("contract_version") != COMPLETE_PROTOTYPE_CONTRACT_VERSION:
        errors.append("manifest contract is invalid")
    try:
        _strict_sha256(
            manifest.get("prototype_configuration_sha256"),
            name="manifest prototype_configuration_sha256",
        )
        expected_manifest_sha256 = _strict_sha256(
            manifest.get("manifest_sha256"),
            name="manifest_sha256",
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        expected_manifest_sha256 = ""
    try:
        policy = _strict_mapping(
            manifest.get("component_policy"),
            name="manifest component_policy",
        )
        _require_exact_fields(
            policy,
            _COMPONENT_POLICY_FIELDS,
            name="manifest component_policy",
        )
        delight_enabled = policy["paper_delight_enabled"]
        kondo_enabled = policy["kondo_enabled"]
        if type(delight_enabled) is not bool or type(kondo_enabled) is not bool:
            raise TypeError("manifest component-policy flags must be exact booleans")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        delight_enabled = False
        kondo_enabled = False

    rows = manifest.get("rows")
    row_statuses: list[RowStatus] = []
    all_registered = True
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        errors.append("manifest rows must be a sequence")
        rows = ()
    if len(rows) != len(SCORECARD_ROWS):
        errors.append("manifest must contain exactly 18 rows")
    for ordinal, contract in enumerate(SCORECARD_ROWS):
        if ordinal >= len(rows):
            break
        try:
            row = _strict_mapping(rows[ordinal], name=f"manifest rows[{ordinal}]")
            _require_exact_fields(
                row,
                _MANIFEST_ROW_FIELDS,
                name=f"manifest rows[{ordinal}]",
            )
            if row["property_id"] != contract.property_id or row["label"] != contract.label:
                raise ValueError("manifest row identity or label is noncanonical")
            required_roles = _required_roles(
                contract,
                paper_delight_enabled=delight_enabled,
                kondo_enabled=kondo_enabled,
            )
            if row["required_roles"] != list(required_roles):
                raise ValueError("manifest row required roles are noncanonical")
            raw_evidence = row["evidence"]
            if not isinstance(raw_evidence, Sequence) or isinstance(
                raw_evidence, (str, bytes)
            ):
                raise TypeError("manifest row evidence must be a sequence")
            row_errors = row["errors"]
            if not isinstance(row_errors, list) or any(
                type(error) is not str for error in row_errors
            ):
                raise TypeError("manifest row errors must be a list of strings")
            all_registered = all_registered and len(raw_evidence) > 0
            evidence_statuses: list[EvidenceStatus] = []
            supplied_roles: list[object] = []
            for raw_reference in raw_evidence:
                reference = _strict_mapping(
                    raw_reference,
                    name=f"manifest rows[{ordinal}] evidence",
                )
                supplied_roles.append(reference.get("role"))
                status = reference.get("status")
                if status not in ("accepted", "valid-rejection", "not-run", "invalid"):
                    raise ValueError("manifest evidence status is invalid")
                evidence_statuses.append(status)
            if raw_evidence and tuple(supplied_roles) != required_roles:
                raise ValueError("manifest evidence roles are noncanonical")
            if any(status == "invalid" for status in evidence_statuses):
                reconstructed_status: RowStatus = "invalid"
            elif not evidence_statuses or any(
                status == "not-run" for status in evidence_statuses
            ):
                reconstructed_status = "not-run"
            elif any(status == "valid-rejection" for status in evidence_statuses):
                reconstructed_status = "valid-rejection"
            else:
                reconstructed_status = "accepted"
            if row["status"] != reconstructed_status:
                raise ValueError("manifest row status does not reconstruct")
            row_statuses.append(reconstructed_status)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {ordinal}: {exc}")

    if len(row_statuses) == len(SCORECARD_ROWS):
        all_accepted = all(status == "accepted" for status in row_statuses)
        if any(status == "invalid" for status in row_statuses):
            reconstructed_overall: OverallStatus = "invalid"
        elif all_accepted:
            reconstructed_overall = "accepted"
        else:
            reconstructed_overall = "not-ready"
        if manifest.get("overall_status") != reconstructed_overall:
            errors.append("manifest overall status does not reconstruct")
        if type(manifest.get("all_18_rows_registered")) is not bool or (
            manifest.get("all_18_rows_registered") != all_registered
        ):
            errors.append("manifest all_18_rows_registered does not reconstruct")
        if type(manifest.get("all_18_rows_accepted")) is not bool or (
            manifest.get("all_18_rows_accepted") != all_accepted
        ):
            errors.append("manifest all_18_rows_accepted does not reconstruct")
    if manifest.get("interpretation") != _INTERPRETATION:
        errors.append("manifest interpretation is noncanonical")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    try:
        actual_manifest_sha256 = _canonical_sha256(unsigned)
    except (TypeError, ValueError) as exc:
        errors.append(f"manifest canonicalization failed: {exc}")
    else:
        if actual_manifest_sha256 != expected_manifest_sha256:
            errors.append("manifest self-digest does not match its payload")
    return tuple(errors)


def complete_prototype_manifest_exit_code(
    manifest: Mapping[str, object],
) -> Literal[0, 1, 2]:
    """Map a structurally reconstructed manifest to stable command-style codes."""

    if validate_complete_prototype_manifest(manifest):
        return 2
    status = manifest.get("overall_status")
    if status == "accepted":
        return 0
    if status == "not-ready":
        return 1
    return 2


def complete_prototype_manifest_json(manifest: Mapping[str, object]) -> str:
    """Serialize one manifest canonically without accepting NaN values."""

    return json.dumps(
        manifest,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


__all__ = [
    "COMPLETE_PROTOTYPE_CONTRACT_VERSION",
    "COMPLETE_PROTOTYPE_INDEX_SCHEMA",
    "COMPLETE_PROTOTYPE_MANIFEST_SCHEMA",
    "SCORECARD_ROWS",
    "CompletePrototypeValidationReceipt",
    "RegisteredCompletePrototypeValidator",
    "ScorecardRowContract",
    "build_complete_prototype_manifest",
    "complete_prototype_manifest_exit_code",
    "complete_prototype_manifest_json",
    "empty_complete_prototype_evidence_index",
    "validate_complete_prototype_manifest",
]
