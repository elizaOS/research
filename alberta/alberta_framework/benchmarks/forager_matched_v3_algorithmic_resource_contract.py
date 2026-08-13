"""Source-only algorithmic-resource metadata contract for matched Forager v3.

The contract freezes the first twenty algorithmic resource fields shared by the
local, external, and adapter candidate families.  It only constructs and parses
canonical metadata.  It does not inspect a filesystem, run a producer, invoke a
candidate, read a clock, issue a case, evaluate a ceiling, or grant authority.

The dependency direction is deliberately acyclic: a pre-GO measurement intent
binds the case, configuration, producer, and exact field policy; a later family
runner execution record and the measurements bind into an algorithmic resource
receipt.  A future terminal-v2 record may bind that receipt.  Publication,
terminal, host-success, and merger identities therefore cannot appear in either
artifact defined here.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal, NoReturn, cast

ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
)
ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_measurement_intent.v1"
)
LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_algorithmic_resource_receipt.v1"
)
EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_algorithmic_resource_receipt.v1"
)
ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_algorithmic_resource_receipt.v1"
)

LOCAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_algorithmic_resource_producer_descriptor.v1"
)
EXTERNAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_algorithmic_resource_producer_descriptor.v1"
)
ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.adapter_algorithmic_resource_producer_descriptor.v1"
)

LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_runner_completion.v1"
)
EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_receipt.v1"
)
FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_rainbow_result_receipt.v1"
)
PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.ppo_gru_result_receipt.v1"
)

# One shared union is intentional: none of these historical or unqualified
# adapter identities may be relabelled into either member of a future strict
# algorithmic-resource producer descriptor/source pair.
ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYLIST: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905",
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5",
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6",
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2",
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc",
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c",
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2",
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47",
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565",
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f",
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08",
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e",
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500",
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71",
)
_ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYSET: Final = frozenset(
    ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYLIST
)

ALGORITHMIC_RESOURCE_CONTRACT_STATUS: Final = (
    "implemented_source_only_contract_uninvoked_no_production_receipt"
)
ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION: Final = (
    "score_blind_metadata_only_algorithmic_resource_contract_non_authorizing"
)
ALGORITHMIC_RESOURCE_INTENT_STATUS: Final = "pre_go_measurement_intent_content_only_non_authorizing"
ALGORITHMIC_RESOURCE_RECEIPT_STATUS: Final = (
    "structural_algorithmic_resource_measurements_unqualified_non_authorizing"
)

MATCHED_V3_HORIZON: Final = 499_712

MATCHED_V3_LOCAL_CANDIDATE_IDS: Final = (
    "causal_e025_q050",
    "causal_e025_q075",
    "causal_e025_q090",
    "causal_e050_q050",
    "causal_e050_q075",
    "causal_e050_q090",
    "causal_e100_q050",
    "causal_e100_q075",
    "causal_e100_q090",
    "alberta_horde_default",
    "alberta_horde_eps05",
    "alberta_horde_recurrent64",
    "alberta_horde_step3e3",
    "alberta_rtu_h08_taylor",
)
MATCHED_V3_EXTERNAL_CANDIDATE_IDS: Final = (
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "random_policy",
    "search_nearest",
    "search_oracle",
)
MATCHED_V3_ADAPTER_CANDIDATE_IDS: Final = (
    "adapted_full_rainbow",
    "adapted_ppo_gru",
)
MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
)

ALGORITHMIC_RESOURCE_FIELDS: Final = (
    "max_environment_interactions",
    "max_optimizer_updates",
    "max_gradient_updates",
    "max_sample_updates",
    "max_trainable_parameters",
    "max_frozen_parameters",
    "max_optimizer_state_elements",
    "max_optimizer_state_bytes",
    "max_target_copy_elements",
    "max_target_copy_bytes",
    "max_replay_capacity_transitions",
    "max_replay_peak_bytes",
    "max_rollout_storage_elements",
    "max_rollout_peak_bytes",
    "max_recurrent_carry_elements",
    "max_recurrent_carry_bytes",
    "max_rtrl_sensitivity_elements",
    "max_rtrl_sensitivity_bytes",
    "max_eligibility_elements",
    "max_eligibility_bytes",
)

EXACT_RUNTIME_COUNTER: Final = "exact_runtime_counter"
EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER: Final = "exact_continuous_live_tree_high_water"
EXACT_CONFIGURATION_AND_RUNTIME_CAPACITY: Final = "exact_configuration_and_runtime_capacity"
STRUCTURAL_ABSENCE: Final = "structural_absence"
NOT_ABSENT: Final = "not_absent"
ZERO_FORBIDDEN: Final = "zero_forbidden"

_FIELD_POLICY_ROWS: Final = (
    (
        "max_environment_interactions",
        EXACT_RUNTIME_COUNTER,
        "whole_case_environment_transition_commits",
        ZERO_FORBIDDEN,
    ),
    (
        "max_optimizer_updates",
        EXACT_RUNTIME_COUNTER,
        "whole_case_committed_optimizer_update_transactions",
        "optimizer_subsystem_absent",
    ),
    (
        "max_gradient_updates",
        EXACT_RUNTIME_COUNTER,
        "whole_case_committed_gradient_application_transactions",
        "gradient_update_path_absent",
    ),
    (
        "max_sample_updates",
        EXACT_RUNTIME_COUNTER,
        "whole_case_learning_sample_contributions_consumed",
        "sample_update_path_absent",
    ),
    (
        "max_trainable_parameters",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_trainable_parameter_scalars",
        "trainable_parameter_tree_absent",
    ),
    (
        "max_frozen_parameters",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_frozen_parameter_scalars",
        "frozen_parameter_tree_absent",
    ),
    (
        "max_optimizer_state_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_optimizer_state_scalars",
        "optimizer_state_tree_absent",
    ),
    (
        "max_optimizer_state_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_optimizer_state_array_bytes",
        "optimizer_state_tree_absent",
    ),
    (
        "max_target_copy_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_target_copy_scalars",
        "target_copy_tree_absent",
    ),
    (
        "max_target_copy_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_target_copy_array_bytes",
        "target_copy_tree_absent",
    ),
    (
        "max_replay_capacity_transitions",
        EXACT_CONFIGURATION_AND_RUNTIME_CAPACITY,
        "whole_case_maximum_addressable_replay_transitions",
        "replay_subsystem_absent",
    ),
    (
        "max_replay_peak_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_replay_array_and_metadata_bytes",
        "replay_subsystem_absent",
    ),
    (
        "max_rollout_storage_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_rollout_segment_and_gae_scalars",
        "rollout_storage_absent",
    ),
    (
        "max_rollout_peak_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_rollout_segment_and_gae_array_bytes",
        "rollout_storage_absent",
    ),
    (
        "max_recurrent_carry_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_cross_step_carry_scalars",
        "recurrent_carry_absent",
    ),
    (
        "max_recurrent_carry_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_cross_step_carry_array_bytes",
        "recurrent_carry_absent",
    ),
    (
        "max_rtrl_sensitivity_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_rtrl_sensitivity_scalars",
        "rtrl_sensitivity_absent",
    ),
    (
        "max_rtrl_sensitivity_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_rtrl_sensitivity_array_bytes",
        "rtrl_sensitivity_absent",
    ),
    (
        "max_eligibility_elements",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_eligibility_trace_scalars",
        "eligibility_trace_absent",
    ),
    (
        "max_eligibility_bytes",
        EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER,
        "whole_case_simultaneously_live_eligibility_trace_array_bytes",
        "eligibility_trace_absent",
    ),
)

FIELD_ALLOWED_ZERO_ABSENCE: Final[Mapping[str, str]] = MappingProxyType(
    {field: absence for field, _kind, _scope, absence in _FIELD_POLICY_ROWS}
)

COUPLED_RESOURCE_FIELD_PAIRS: Final = (
    ("max_optimizer_state_elements", "max_optimizer_state_bytes"),
    ("max_target_copy_elements", "max_target_copy_bytes"),
    ("max_replay_capacity_transitions", "max_replay_peak_bytes"),
    ("max_rollout_storage_elements", "max_rollout_peak_bytes"),
    ("max_recurrent_carry_elements", "max_recurrent_carry_bytes"),
    ("max_rtrl_sensitivity_elements", "max_rtrl_sensitivity_bytes"),
    ("max_eligibility_elements", "max_eligibility_bytes"),
)

_MAX_ARTIFACT_BYTES: Final = 1024 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_NODES: Final = 20_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


class ForagerMatchedV3AlgorithmicResourceContractError(ValueError):
    """A source-only algorithmic resource artifact failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3AlgorithmicResourceContractError(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _require_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        _fail(f"{label} must be bounded nonempty printable ASCII")
    return value


def _require_identifier(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail(f"{label} must be one portable identifier")
    return text


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        _fail(f"{label} keys differ")
    return cast(dict[str, Any], value)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"algorithmic-resource JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"algorithmic-resource JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("algorithmic-resource JSON integer exceeds its lexical bound")
    parsed = int(value, 10)
    if not -_MAX_INTEGER <= parsed <= _MAX_INTEGER:
        _fail("algorithmic-resource JSON integer exceeds its value bound")
    return parsed


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("algorithmic-resource JSON contains a duplicate or non-text key")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("algorithmic-resource JSON structure exceeds its bound")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            _require_int(item, "algorithmic-resource JSON integer", minimum=-_MAX_INTEGER)
            return
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("algorithmic-resource JSON strings must be bounded printable ASCII")
            return
        if type(item) not in {dict, list}:
            _fail("algorithmic-resource JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            _fail("algorithmic-resource JSON contains an alias or cycle")
        seen.add(identity)
        if type(item) is list:
            for child in cast(list[object], item):
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str:
                    _fail("algorithmic-resource JSON keys must be exact strings")
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return frozenset(left_map) == frozenset(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(a, b) for a, b in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    _assert_plain_unaliased_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ForagerMatchedV3AlgorithmicResourceContractError(
            "algorithmic-resource value is not canonical JSON"
        ) from exc
    if newline:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("algorithmic-resource canonical JSON exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("algorithmic-resource bytes violate their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3AlgorithmicResourceContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3AlgorithmicResourceContractError(
            "algorithmic-resource bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("algorithmic-resource JSON root must be one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(exact)
    if not hmac.compare_digest(_canonical_json(exact), raw):
        _fail("algorithmic-resource bytes are not canonical")
    return exact


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _body_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(dict(value), newline=False))


def _with_body_sha256(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = _body_sha256(body)
    return result


def _validate_body_sha256(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = _require_sha256(value.get(field), f"{label} body")
    body = dict(value)
    body.pop(field, None)
    expected = _body_sha256(body)
    if not hmac.compare_digest(supplied, expected):
        _fail(f"{label} body SHA-256 differs")
    return supplied


def _validate_caller_file_pin(raw: bytes, expected_file_sha256: object, label: str) -> str:
    expected = _require_sha256(expected_file_sha256, f"{label} caller file pin")
    observed = _sha256(raw)
    if not hmac.compare_digest(observed, expected):
        _fail(f"{label} full-file SHA-256 differs from its caller pin")
    return observed


def _ordered_values_sha256(values: tuple[str, ...]) -> str:
    return _sha256(_canonical_json(list(values), newline=False))


def _capabilities() -> dict[str, bool]:
    return {
        "artifact_authentication": False,
        "clock": False,
        "default_inputs": False,
        "evaluator": False,
        "executor": False,
        "filesystem": False,
        "issuer": False,
        "network": False,
        "process": False,
        "producer_execution": False,
    }


def _readiness() -> dict[str, bool]:
    return {
        "evaluation_ready": False,
        "execution_ready": False,
        "issuance_ready": False,
        "producer_available": False,
        "production_receipt_available": False,
        "qualification_ready": False,
    }


def _authority() -> dict[str, bool]:
    return {
        "evaluation_performed": False,
        "execution_authorized": False,
        "issuance_performed": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "performance_claim_allowed": False,
        "promotion_allowed": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "source_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This contract validates canonical structural metadata only.",
        "No family algorithmic resource producer is implemented or invoked here.",
        "Historical or unqualified adapter identities are rejected as one cross-kind union.",
        "A future full merger must pin one exact production adapter producer pair.",
        "Runner execution receipts are identity bindings, not complete resource records.",
        "Referenced configuration, intent, producer, runner, or basis bytes are not read here.",
        "Publication, terminal, host-success, storage, and merger identities are absent.",
        "A future terminal-v2 record must bind a receipt one-way after it exists.",
        "No ceiling comparison, qualification, issuance, evidence, or claim is produced.",
    ]


def _family_for_candidate(
    candidate_id: str,
) -> Literal["local", "external", "adapter"]:
    if candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
        return "local"
    if candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
        return "external"
    if candidate_id in MATCHED_V3_ADAPTER_CANDIDATE_IDS:
        return "adapter"
    _fail("candidate ID is outside the frozen matched-v3 universe")


def algorithmic_resource_receipt_schema_for_family(family: str) -> str:
    """Return the exact family-specific resource-receipt schema."""

    if family == "local":
        return LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    if family == "adapter":
        return ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION
    _fail("algorithmic resource receipt family differs")


def algorithmic_resource_producer_schema_for_family(family: str) -> str:
    """Return the exact family-specific producer descriptor schema."""

    if family == "local":
        return LOCAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
    if family == "adapter":
        return ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION
    _fail("algorithmic resource producer family differs")


def runner_execution_receipt_schema_for_candidate(candidate_id: str) -> str:
    """Return the frozen runner-level execution schema bound by one receipt."""

    family = _family_for_candidate(candidate_id)
    if family == "local":
        return LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION
    if family == "external":
        return EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION
    if candidate_id == "adapted_full_rainbow":
        return FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION
    if candidate_id == "adapted_ppo_gru":
        return PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    _fail("adapter runner execution schema differs")


def _require_case_projection(
    *,
    case_ordinal: object,
    candidate_id: object,
    candidate_family: object,
    qualification_case_id: object,
) -> tuple[int, str, Literal["local", "external", "adapter"], str]:
    ordinal = _require_int(
        case_ordinal,
        "case ordinal",
        maximum=len(MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS) - 1,
    )
    expected_candidate = MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS[ordinal]
    if type(candidate_id) is not str or candidate_id != expected_candidate:
        _fail("case ordinal and candidate ID differ from the frozen order")
    exact_candidate = expected_candidate
    expected_family = _family_for_candidate(exact_candidate)
    if type(candidate_family) is not str or candidate_family != expected_family:
        _fail("candidate family differs from the frozen candidate projection")
    expected_case_id = f"qualification_{ordinal:02d}_{exact_candidate}"
    if type(qualification_case_id) is not str or qualification_case_id != expected_case_id:
        _fail("qualification case ID differs from the frozen case projection")
    return ordinal, exact_candidate, expected_family, expected_case_id


@dataclass(frozen=True, slots=True)
class ArtifactIdentityV1:
    """One caller-carried canonical artifact identity; no bytes are loaded."""

    schema_version: str
    file_sha256: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "artifact schema")
        _require_sha256(self.file_sha256, "artifact file")
        _require_sha256(self.body_sha256, "artifact body")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "file_sha256": self.file_sha256,
            "body_sha256": self.body_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProducerIdentityV1:
    """One independently pinned descriptor/source producer identity."""

    descriptor_schema_version: str
    descriptor_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.descriptor_schema_version, "producer descriptor schema")
        _require_sha256(self.descriptor_sha256, "producer descriptor")
        _require_sha256(self.source_sha256, "producer source")

    def to_dict(self) -> dict[str, str]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "source_sha256": self.source_sha256,
        }


def _validate_producer_for_family(
    producer: object,
    family: Literal["local", "external", "adapter"],
    label: str,
) -> ProducerIdentityV1:
    if type(producer) is not ProducerIdentityV1:
        _fail(f"{label} producer identity type differs")
    exact = producer
    if exact.descriptor_schema_version != algorithmic_resource_producer_schema_for_family(family):
        _fail(f"{label} producer descriptor schema differs from its family")
    if family == "adapter" and any(
        digest in _ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYSET
        for digest in (exact.descriptor_sha256, exact.source_sha256)
    ):
        _fail(
            "historical or unqualified adapter algorithmic-resource producer identity "
            "cannot fill either strict producer slot"
        )
    return exact


@dataclass(frozen=True, slots=True)
class AlgorithmicResourceFieldPolicyV1:
    """Frozen positive measurement and typed-zero policy for one field."""

    field_name: str
    positive_measurement_kind: str
    measurement_scope: str
    zero_structural_absence_kind: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.field_name, "field-policy field name"),
            (self.positive_measurement_kind, "field-policy positive measurement kind"),
            (self.measurement_scope, "field-policy measurement scope"),
            (self.zero_structural_absence_kind, "field-policy zero structural absence kind"),
        ):
            _require_text(value, label)
        exact = (
            self.field_name,
            self.positive_measurement_kind,
            self.measurement_scope,
            self.zero_structural_absence_kind,
        )
        if exact not in _FIELD_POLICY_ROWS:
            _fail("algorithmic resource field policy differs from the frozen policy")

    def to_dict(self) -> dict[str, str]:
        return {
            "field_name": self.field_name,
            "positive_measurement_kind": self.positive_measurement_kind,
            "measurement_scope": self.measurement_scope,
            "zero_structural_absence_kind": self.zero_structural_absence_kind,
        }


def matched_v3_algorithmic_resource_field_policy() -> tuple[AlgorithmicResourceFieldPolicyV1, ...]:
    """Return the exact immutable first-20 field policy."""

    return tuple(AlgorithmicResourceFieldPolicyV1(*row) for row in _FIELD_POLICY_ROWS)


def algorithmic_resource_field_policy_inventory_sha256(
    field_policy: tuple[AlgorithmicResourceFieldPolicyV1, ...],
) -> str:
    """Return the detached digest of one exact ordered field policy."""

    exact = _validate_field_policy(field_policy)
    return _sha256(
        _canonical_json(
            {"field_policy": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


def _validate_field_policy(
    value: object,
) -> tuple[AlgorithmicResourceFieldPolicyV1, ...]:
    if type(value) is not tuple or any(
        type(item) is not AlgorithmicResourceFieldPolicyV1 for item in value
    ):
        _fail("algorithmic resource field policy must use one exact tuple")
    exact = cast(tuple[AlgorithmicResourceFieldPolicyV1, ...], value)
    if exact != matched_v3_algorithmic_resource_field_policy():
        _fail("algorithmic resource field policy order or content differs")
    return exact


@dataclass(frozen=True, slots=True)
class AlgorithmicResourceMeasurementV1:
    """One exact measurement or one typed proof of structural absence."""

    field_name: str
    observed_value: int
    measurement_kind: str
    measurement_scope: str
    measurement_basis_body_sha256: str
    structural_absence_kind: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.field_name, "measurement field name"),
            (self.measurement_kind, "measurement kind"),
            (self.measurement_scope, "measurement scope"),
            (self.structural_absence_kind, "measurement structural absence kind"),
        ):
            _require_text(value, label)
        if self.field_name not in ALGORITHMIC_RESOURCE_FIELDS:
            _fail("algorithmic resource measurement field differs")
        observed_value = _require_int(
            self.observed_value,
            f"measurement {self.field_name}",
        )
        _require_sha256(
            self.measurement_basis_body_sha256,
            "measurement-basis body",
        )
        policy = matched_v3_algorithmic_resource_field_policy()[
            ALGORITHMIC_RESOURCE_FIELDS.index(self.field_name)
        ]
        if self.measurement_scope != policy.measurement_scope:
            _fail(f"measurement scope differs for {self.field_name}")
        if observed_value > 0:
            if (
                self.measurement_kind != policy.positive_measurement_kind
                or self.structural_absence_kind != NOT_ABSENT
            ):
                _fail(f"positive measurement contract differs for {self.field_name}")
        elif policy.zero_structural_absence_kind == ZERO_FORBIDDEN:
            _fail(f"zero is forbidden for {self.field_name}")
        elif (
            self.measurement_kind != STRUCTURAL_ABSENCE
            or self.structural_absence_kind != policy.zero_structural_absence_kind
        ):
            _fail(f"zero measurement lacks its exact typed absence for {self.field_name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "observed_value": self.observed_value,
            "measurement_kind": self.measurement_kind,
            "measurement_scope": self.measurement_scope,
            "measurement_basis_body_sha256": self.measurement_basis_body_sha256,
            "structural_absence_kind": self.structural_absence_kind,
        }


def _validate_measurements(
    value: object,
) -> tuple[AlgorithmicResourceMeasurementV1, ...]:
    if type(value) is not tuple or any(
        type(item) is not AlgorithmicResourceMeasurementV1 for item in value
    ):
        _fail("algorithmic resource measurements must use one exact tuple")
    exact = cast(tuple[AlgorithmicResourceMeasurementV1, ...], value)
    if tuple(item.field_name for item in exact) != ALGORITHMIC_RESOURCE_FIELDS:
        _fail("algorithmic resource measurements must use the exact first-20 order")
    by_name = {item.field_name: item for item in exact}
    if by_name["max_environment_interactions"].observed_value != MATCHED_V3_HORIZON:
        _fail("environment interactions must equal the exact matched-v3 horizon")
    for left_name, right_name in COUPLED_RESOURCE_FIELD_PAIRS:
        left = by_name[left_name]
        right = by_name[right_name]
        left_zero = left.observed_value == 0
        right_zero = right.observed_value == 0
        if left_zero != right_zero:
            _fail(f"coupled measurement zero/nonzero state differs for {left_name}")
        if left_zero and (
            left.structural_absence_kind != right.structural_absence_kind
            or left.measurement_basis_body_sha256 != right.measurement_basis_body_sha256
        ):
            _fail(f"coupled structural absence proof differs for {left_name}")
    optimizer_updates = by_name["max_optimizer_updates"]
    gradient_updates = by_name["max_gradient_updates"]
    sample_updates = by_name["max_sample_updates"]
    trainable_parameters = by_name["max_trainable_parameters"]
    if optimizer_updates.observed_value == 0 and any(
        by_name[name].observed_value != 0
        for name in ("max_optimizer_state_elements", "max_optimizer_state_bytes")
    ):
        _fail("an absent optimizer subsystem cannot retain optimizer state")
    if optimizer_updates.observed_value > 0 and (
        sample_updates.observed_value == 0 or trainable_parameters.observed_value == 0
    ):
        _fail("positive optimizer updates require samples and trainable parameters")
    if gradient_updates.observed_value > 0 and (
        optimizer_updates.observed_value == 0
        or sample_updates.observed_value == 0
        or trainable_parameters.observed_value == 0
    ):
        _fail(
            "positive gradient updates require optimizer updates, samples, and trainable parameters"
        )
    return exact


def algorithmic_resource_measurement_inventory_sha256(
    fields: tuple[AlgorithmicResourceMeasurementV1, ...],
) -> str:
    """Return the detached digest of one exact ordered measurement inventory."""

    exact = _validate_measurements(fields)
    return _sha256(
        _canonical_json(
            {"fields": [item.to_dict() for item in exact]},
            newline=False,
        )
    )


@dataclass(frozen=True, slots=True)
class AlgorithmicResourceMeasurementIntentV1:
    """Pre-GO case/configuration/policy commitment; never execution authority."""

    schema_version: str
    campaign_spine_sha256: str
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    resource_requirement_body_sha256: str
    configuration_sha256: str
    producer: ProducerIdentityV1
    field_policy_inventory_sha256: str
    field_policy: tuple[AlgorithmicResourceFieldPolicyV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION
        ):
            _fail("algorithmic resource measurement-intent schema differs")
        _require_sha256(self.campaign_spine_sha256, "intent campaign spine")
        _require_sha256(self.case_spine_sha256, "intent case spine")
        _require_case_projection(
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            candidate_family=self.candidate_family,
            qualification_case_id=self.qualification_case_id,
        )
        _require_sha256(
            self.resource_requirement_body_sha256,
            "intent resource requirement body",
        )
        _require_sha256(self.configuration_sha256, "intent configuration")
        _validate_producer_for_family(self.producer, self.candidate_family, "intent")
        _require_sha256(
            self.field_policy_inventory_sha256,
            "intent field-policy inventory",
        )
        policy = _validate_field_policy(self.field_policy)
        if self.field_policy_inventory_sha256 != algorithmic_resource_field_policy_inventory_sha256(
            policy
        ):
            _fail("intent field-policy inventory digest does not replay")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": ALGORITHMIC_RESOURCE_INTENT_STATUS,
            "classification": ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION,
            "campaign_spine_sha256": self.campaign_spine_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "configuration_sha256": self.configuration_sha256,
            "producer": self.producer.to_dict(),
            "field_policy_inventory_sha256": self.field_policy_inventory_sha256,
            "field_policy": [item.to_dict() for item in self.field_policy],
            "capabilities": _capabilities(),
            "readiness": _readiness(),
            "authority": _authority(),
            "claims": _claims(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "intent_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


@dataclass(frozen=True, slots=True)
class AlgorithmicResourceReceiptV1:
    """Family-specific exact first-20 measurements; still non-authorizing."""

    schema_version: str
    campaign_spine_sha256: str
    case_spine_sha256: str
    case_ordinal: int
    candidate_id: str
    candidate_family: Literal["local", "external", "adapter"]
    qualification_case_id: str
    resource_requirement_body_sha256: str
    configuration_sha256: str
    producer: ProducerIdentityV1
    measurement_intent: ArtifactIdentityV1
    runner_execution_receipt: ArtifactIdentityV1
    field_inventory_sha256: str
    fields: tuple[AlgorithmicResourceMeasurementV1, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str:
            _fail("algorithmic resource receipt schema differs from its family")
        _require_case_projection(
            case_ordinal=self.case_ordinal,
            candidate_id=self.candidate_id,
            candidate_family=self.candidate_family,
            qualification_case_id=self.qualification_case_id,
        )
        expected_receipt_schema = algorithmic_resource_receipt_schema_for_family(
            self.candidate_family
        )
        if self.schema_version != expected_receipt_schema:
            _fail("algorithmic resource receipt schema differs from its family")
        _require_sha256(self.campaign_spine_sha256, "receipt campaign spine")
        _require_sha256(self.case_spine_sha256, "receipt case spine")
        _require_sha256(
            self.resource_requirement_body_sha256,
            "receipt resource requirement body",
        )
        _require_sha256(self.configuration_sha256, "receipt configuration")
        _validate_producer_for_family(self.producer, self.candidate_family, "receipt")
        if type(self.measurement_intent) is not ArtifactIdentityV1:
            _fail("measurement-intent identity type differs")
        if (
            self.measurement_intent.schema_version
            != ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION
        ):
            _fail("measurement-intent artifact schema differs")
        if type(self.runner_execution_receipt) is not ArtifactIdentityV1:
            _fail("runner execution-receipt identity type differs")
        if (
            self.runner_execution_receipt.schema_version
            != runner_execution_receipt_schema_for_candidate(self.candidate_id)
        ):
            _fail("runner execution-receipt schema differs from the candidate")
        if self.runner_execution_receipt in {self.measurement_intent}:
            _fail("runner execution receipt cannot alias the measurement intent")
        _require_sha256(self.field_inventory_sha256, "receipt field inventory")
        fields = _validate_measurements(self.fields)
        if self.field_inventory_sha256 != algorithmic_resource_measurement_inventory_sha256(fields):
            _fail("receipt field inventory digest does not replay")

    def to_body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": ALGORITHMIC_RESOURCE_RECEIPT_STATUS,
            "classification": ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION,
            "campaign_spine_sha256": self.campaign_spine_sha256,
            "case_spine_sha256": self.case_spine_sha256,
            "case_ordinal": self.case_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "qualification_case_id": self.qualification_case_id,
            "resource_requirement_body_sha256": self.resource_requirement_body_sha256,
            "configuration_sha256": self.configuration_sha256,
            "producer": self.producer.to_dict(),
            "measurement_intent": self.measurement_intent.to_dict(),
            "runner_execution_receipt": self.runner_execution_receipt.to_dict(),
            "field_inventory_sha256": self.field_inventory_sha256,
            "fields": [item.to_dict() for item in self.fields],
            "capabilities": _capabilities(),
            "readiness": _readiness(),
            "authority": _authority(),
            "claims": _claims(),
            "limitations": _limitations(),
        }

    def to_dict(self) -> dict[str, Any]:
        return _with_body_sha256(self.to_body_dict(), "receipt_body_sha256")

    @property
    def body_sha256(self) -> str:
        return _body_sha256(self.to_body_dict())


def canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(
    intent: AlgorithmicResourceMeasurementIntentV1,
) -> bytes:
    """Serialize one intent as canonical ASCII JSON with one trailing LF."""

    if type(intent) is not AlgorithmicResourceMeasurementIntentV1:
        raise TypeError("measurement intent must use the exact intent type")
    return _canonical_json(intent.to_dict())


def canonical_matched_v3_algorithmic_resource_receipt_bytes(
    receipt: AlgorithmicResourceReceiptV1,
) -> bytes:
    """Serialize one family receipt as canonical ASCII JSON with one trailing LF."""

    if type(receipt) is not AlgorithmicResourceReceiptV1:
        raise TypeError("algorithmic resource receipt must use the exact receipt type")
    return _canonical_json(receipt.to_dict())


def _artifact_identity_from_dict(value: object, label: str) -> ArtifactIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ArtifactIdentityV1.__dataclass_fields__),
        label,
    )
    return ArtifactIdentityV1(**item)


def _producer_identity_from_dict(value: object) -> ProducerIdentityV1:
    item = _require_exact_keys(
        value,
        frozenset(ProducerIdentityV1.__dataclass_fields__),
        "producer identity",
    )
    return ProducerIdentityV1(**item)


def _field_policy_from_dict(value: object) -> AlgorithmicResourceFieldPolicyV1:
    item = _require_exact_keys(
        value,
        frozenset(AlgorithmicResourceFieldPolicyV1.__dataclass_fields__),
        "field policy",
    )
    return AlgorithmicResourceFieldPolicyV1(**item)


def _measurement_from_dict(value: object) -> AlgorithmicResourceMeasurementV1:
    item = _require_exact_keys(
        value,
        frozenset(AlgorithmicResourceMeasurementV1.__dataclass_fields__),
        "resource measurement",
    )
    return AlgorithmicResourceMeasurementV1(**item)


_COMMON_ENVELOPE_KEYS: Final = frozenset(
    {
        "status",
        "classification",
        "capabilities",
        "readiness",
        "authority",
        "claims",
        "limitations",
    }
)


def _validate_envelope(item: dict[str, Any], *, status: str, label: str) -> None:
    if item.pop("status") != status:
        _fail(f"{label} status differs")
    if item.pop("classification") != ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION:
        _fail(f"{label} classification differs")
    if item.pop("capabilities") != _capabilities():
        _fail(f"{label} capabilities differ")
    if item.pop("readiness") != _readiness():
        _fail(f"{label} readiness differs")
    if item.pop("authority") != _authority():
        _fail(f"{label} authority differs")
    if item.pop("claims") != _claims():
        _fail(f"{label} claims differ")
    if item.pop("limitations") != _limitations():
        _fail(f"{label} limitations differ")


def parse_matched_v3_algorithmic_resource_measurement_intent(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> AlgorithmicResourceMeasurementIntentV1:
    """Parse an exact intent after checking its caller-supplied full-file pin."""

    _validate_caller_file_pin(raw, expected_file_sha256, "measurement intent")
    value = _strict_json_load(raw)
    _validate_body_sha256(value, "intent_body_sha256", "measurement intent")
    expected_keys = frozenset(
        {
            *AlgorithmicResourceMeasurementIntentV1.__dataclass_fields__,
            *_COMMON_ENVELOPE_KEYS,
            "intent_body_sha256",
        }
    )
    item = _require_exact_keys(value, expected_keys, "measurement intent")
    item = dict(item)
    item.pop("intent_body_sha256")
    _validate_envelope(item, status=ALGORITHMIC_RESOURCE_INTENT_STATUS, label="intent")
    producer = _producer_identity_from_dict(item.pop("producer"))
    field_policy = item.pop("field_policy")
    if type(field_policy) is not list:
        _fail("measurement intent field policy must be one list")
    intent = AlgorithmicResourceMeasurementIntentV1(
        **item,
        producer=producer,
        field_policy=tuple(_field_policy_from_dict(child) for child in field_policy),
    )
    if raw != canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(intent):
        _fail("measurement intent canonical replay differs")
    return intent


def parse_matched_v3_algorithmic_resource_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> AlgorithmicResourceReceiptV1:
    """Parse an exact family receipt after checking its caller full-file pin."""

    _validate_caller_file_pin(raw, expected_file_sha256, "algorithmic resource receipt")
    value = _strict_json_load(raw)
    _validate_body_sha256(value, "receipt_body_sha256", "algorithmic resource receipt")
    expected_keys = frozenset(
        {
            *AlgorithmicResourceReceiptV1.__dataclass_fields__,
            *_COMMON_ENVELOPE_KEYS,
            "receipt_body_sha256",
        }
    )
    item = _require_exact_keys(value, expected_keys, "algorithmic resource receipt")
    item = dict(item)
    item.pop("receipt_body_sha256")
    _validate_envelope(
        item,
        status=ALGORITHMIC_RESOURCE_RECEIPT_STATUS,
        label="algorithmic resource receipt",
    )
    producer = _producer_identity_from_dict(item.pop("producer"))
    measurement_intent = _artifact_identity_from_dict(
        item.pop("measurement_intent"),
        "measurement-intent identity",
    )
    runner_execution_receipt = _artifact_identity_from_dict(
        item.pop("runner_execution_receipt"),
        "runner execution-receipt identity",
    )
    fields = item.pop("fields")
    if type(fields) is not list:
        _fail("algorithmic resource fields must be one list")
    receipt = AlgorithmicResourceReceiptV1(
        **item,
        producer=producer,
        measurement_intent=measurement_intent,
        runner_execution_receipt=runner_execution_receipt,
        fields=tuple(_measurement_from_dict(child) for child in fields),
    )
    if raw != canonical_matched_v3_algorithmic_resource_receipt_bytes(receipt):
        _fail("algorithmic resource receipt canonical replay differs")
    return receipt


def validate_matched_v3_algorithmic_resource_receipt_chain(
    intent: AlgorithmicResourceMeasurementIntentV1,
    receipt: AlgorithmicResourceReceiptV1,
) -> None:
    """Validate the acyclic intent-to-receipt projection without reading artifacts."""

    if type(intent) is not AlgorithmicResourceMeasurementIntentV1:
        raise TypeError("measurement intent must use the exact intent type")
    if type(receipt) is not AlgorithmicResourceReceiptV1:
        raise TypeError("algorithmic resource receipt must use the exact receipt type")
    intent_raw = canonical_matched_v3_algorithmic_resource_measurement_intent_bytes(intent)
    expected_intent = ArtifactIdentityV1(
        schema_version=ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
        file_sha256=_sha256(intent_raw),
        body_sha256=intent.body_sha256,
    )
    if receipt.measurement_intent != expected_intent:
        _fail("receipt measurement-intent identity differs")
    for field in (
        "campaign_spine_sha256",
        "case_spine_sha256",
        "case_ordinal",
        "candidate_id",
        "candidate_family",
        "qualification_case_id",
        "resource_requirement_body_sha256",
        "configuration_sha256",
        "producer",
    ):
        if getattr(receipt, field) != getattr(intent, field):
            _fail(f"receipt intent projection differs for {field}")


def _contract_descriptor() -> dict[str, Any]:
    field_policy = matched_v3_algorithmic_resource_field_policy()
    body: dict[str, Any] = {
        "schema_version": ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": ALGORITHMIC_RESOURCE_CONTRACT_STATUS,
        "classification": ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION,
        "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
        "full_file_caller_pin_required": True,
        "receipt_schemas": {
            "local": LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            "external": EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            "adapter": ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
        },
        "producer_descriptor_schemas": {
            "local": LOCAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
            "external": EXTERNAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
            "adapter": ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION,
        },
        "runner_execution_receipt_schemas": {
            "local": LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "external": EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "adapted_full_rainbow": (FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION),
            "adapted_ppo_gru": PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
        },
        "adapter_algorithmic_resource_producer_identity_policy": {
            "historical_or_unqualified_identity_sha256_union_denylist": list(
                ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYLIST
            ),
            "historical_or_unqualified_pair_count": 7,
            "denylist_value_count": 14,
            "cross_kind_union_rejection": True,
            "applies_to_families": ["adapter"],
            "applies_to_artifacts": [
                ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION,
                ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION,
            ],
            "applies_to_producer_fields": [
                "descriptor_sha256",
                "source_sha256",
            ],
            "future_full_resource_merger_must_pin_exact_production_producer_pair": True,
            "exact_production_producer_pair_pinned_here": False,
            "production_adapter_algorithmic_resource_producer_implemented_here": False,
            "full_resource_merger_implemented_here": False,
            "producer_descriptor_source_or_runner_bytes_read_here": False,
        },
        "adapted_ppo_gru_runner_schema_policy": {
            "accepted_schema_version": PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION,
            "compiled_v2_admitted": False,
        },
        "measurement_intent_schema_version": (
            ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION
        ),
        "candidate_order": list(MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS),
        "candidate_order_sha256": _ordered_values_sha256(
            MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS
        ),
        "candidate_families": {
            "local": list(MATCHED_V3_LOCAL_CANDIDATE_IDS),
            "external": list(MATCHED_V3_EXTERNAL_CANDIDATE_IDS),
            "adapter": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
        },
        "algorithmic_resource_fields": list(ALGORITHMIC_RESOURCE_FIELDS),
        "algorithmic_resource_field_order_sha256": _ordered_values_sha256(
            ALGORITHMIC_RESOURCE_FIELDS
        ),
        "field_policy_inventory_sha256": (
            algorithmic_resource_field_policy_inventory_sha256(field_policy)
        ),
        "field_policy": [item.to_dict() for item in field_policy],
        "measurement_basis_contract": {
            "digest_kind": "canonical_producer_specific_measurement_basis_body_sha256",
            "schema_validated_by_future_pinned_family_producer": True,
            "basis_bytes_read_or_authenticated_here": False,
            "basis_schema_inferred_here": False,
        },
        "coupled_field_pairs": [list(pair) for pair in COUPLED_RESOURCE_FIELD_PAIRS],
        "measurement_chain": [
            "pre_go_measurement_intent",
            "family_runner_execution_receipt",
            "algorithmic_resource_receipt",
            "future_terminal_v2",
            "future_host_success_v2",
            "future_full_resource_merger",
        ],
        "reverse_receipt_pins_forbidden": [
            "publication",
            "terminal",
            "host_success",
            "storage",
            "full_resource_merger",
        ],
        "runner_execution_receipts_are_complete_resource_records": False,
        "candidate_values_supplied_or_inferred": False,
        "ceiling_comparison_performed": False,
        "capabilities": _capabilities(),
        "readiness": _readiness(),
        "authority": _authority(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return _with_body_sha256(body, "descriptor_body_sha256")


_DESCRIPTOR: Final = _contract_descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR)

# Independently replayed from the canonical one-LF descriptor after schema audit.
PINNED_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)


def matched_v3_algorithmic_resource_contract_descriptor() -> dict[str, Any]:
    """Return the source-only, non-authorizing contract descriptor."""

    return copy.deepcopy(_DESCRIPTOR)


def canonical_matched_v3_algorithmic_resource_contract_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes without bypassing the literal pin."""

    return bytes(_DESCRIPTOR_BYTES)


def matched_v3_algorithmic_resource_contract_descriptor_sha256() -> str:
    """Return the descriptor identity after enforcing its literal audit pin."""

    observed = _sha256(_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(
        observed,
        PINNED_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256,
    ):
        _fail("algorithmic resource contract descriptor drifted from its literal pin")
    return observed


def parse_matched_v3_algorithmic_resource_contract_descriptor(
    raw: bytes,
) -> dict[str, Any]:
    """Parse only the exact frozen descriptor after its pin is audited."""

    value = _strict_json_load(raw)
    _validate_body_sha256(value, "descriptor_body_sha256", "contract descriptor")
    if not _exact_json_equal(value, _DESCRIPTOR):
        _fail("algorithmic resource contract descriptor content differs")
    if raw != _DESCRIPTOR_BYTES:
        _fail("algorithmic resource contract descriptor canonical replay differs")
    matched_v3_algorithmic_resource_contract_descriptor_sha256()
    return copy.deepcopy(value)


__all__ = [
    "ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_IDENTITY_SHA256_DENYLIST",
    "ADAPTER_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "ALGORITHMIC_RESOURCE_CONTRACT_CLASSIFICATION",
    "ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "ALGORITHMIC_RESOURCE_CONTRACT_STATUS",
    "ALGORITHMIC_RESOURCE_FIELDS",
    "ALGORITHMIC_RESOURCE_MEASUREMENT_INTENT_SCHEMA_VERSION",
    "ALGORITHMIC_RESOURCE_RECEIPT_STATUS",
    "AlgorithmicResourceFieldPolicyV1",
    "AlgorithmicResourceMeasurementIntentV1",
    "AlgorithmicResourceMeasurementV1",
    "AlgorithmicResourceReceiptV1",
    "ArtifactIdentityV1",
    "COUPLED_RESOURCE_FIELD_PAIRS",
    "EXACT_CONFIGURATION_AND_RUNTIME_CAPACITY",
    "EXACT_CONTINUOUS_LIVE_TREE_HIGH_WATER",
    "EXACT_RUNTIME_COUNTER",
    "EXTERNAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "FIELD_ALLOWED_ZERO_ABSENCE",
    "ForagerMatchedV3AlgorithmicResourceContractError",
    "FULL_RAINBOW_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "LOCAL_ALGORITHMIC_RESOURCE_PRODUCER_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_ALGORITHMIC_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "LOCAL_RUNNER_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_ALGORITHMIC_RESOURCE_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "NOT_ABSENT",
    "PINNED_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256",
    "PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION",
    "ProducerIdentityV1",
    "STRUCTURAL_ABSENCE",
    "ZERO_FORBIDDEN",
    "algorithmic_resource_field_policy_inventory_sha256",
    "algorithmic_resource_measurement_inventory_sha256",
    "algorithmic_resource_producer_schema_for_family",
    "algorithmic_resource_receipt_schema_for_family",
    "canonical_matched_v3_algorithmic_resource_contract_descriptor_bytes",
    "canonical_matched_v3_algorithmic_resource_measurement_intent_bytes",
    "canonical_matched_v3_algorithmic_resource_receipt_bytes",
    "matched_v3_algorithmic_resource_contract_descriptor",
    "matched_v3_algorithmic_resource_contract_descriptor_sha256",
    "matched_v3_algorithmic_resource_field_policy",
    "parse_matched_v3_algorithmic_resource_contract_descriptor",
    "parse_matched_v3_algorithmic_resource_measurement_intent",
    "parse_matched_v3_algorithmic_resource_receipt",
    "runner_execution_receipt_schema_for_candidate",
    "validate_matched_v3_algorithmic_resource_receipt_chain",
]
