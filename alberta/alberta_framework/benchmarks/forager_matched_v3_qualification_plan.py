"""Pre-observation, score-blind qualification-plan contract for Forager v3.

This module only constructs and validates canonical JSON content.  It does not
materialize source trees, inspect a runtime, run a probe, open a result, publish
an artifact, or grant execution authority.  A caller must supply all two source
closures, the complete runtime identity, one permanently consumed public case
per candidate, an independently pinned pre-observation trust-root receipt,
candidate-specific resource ceilings, and all 28 result-publisher bindings.
There is intentionally no default or production plan.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, cast

QUALIFICATION_PLAN_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_qualification_plan_descriptor.v1"
)
QUALIFICATION_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_qualification_plan.v1"
QUALIFICATION_PLAN_STATUS: Final = "contract_implemented_no_production_plan"
QUALIFICATION_PLAN_CLASSIFICATION: Final = "content_only_unexecuted_non_authorizing"

_MAX_ARTIFACT_BYTES: Final = 2 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_TEXT_LENGTH: Final = 512
_MAX_RESOURCE_VALUE: Final = 2**63 - 1
_MAX_SOURCE_ENTRIES: Final = 2_000_000
_UINT31_MAX: Final = 2**31 - 1
_HORIZON: Final = 499_712

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_VERSION_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+:-]{0,127}\Z")
_CPYTHON_312_VERSION_RE: Final = re.compile(r"3\.12\.(?:0|[1-9][0-9]{0,2})\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
_FORBIDDEN_CASE_MATERIAL_RE: Final = re.compile(
    r"(?:trial[ _.-]*block|held[ _.-]*out|future|protected|confirmatory)",
    re.IGNORECASE,
)

_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
_CANDIDATE_UNIVERSE_SHA256: Final = (
    "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
)
_CUMULATIVE_REWARD_METRIC_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
_TRIAL_BLOCK_GENERATOR_PLAN_SHA256: Final = (
    "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
)
_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256: Final = (
    "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
)
_EXTERNAL_MATERIALIZATION_SOURCE_SHA256: Final = (
    "5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"
)
_ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
_ADAPTER_REWARD_PUBLICATION_SOURCE_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)
_ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "1699a253b45a1ef3e5d23c46639d38167dd04b667d4aa1242c9f4d1571c4f2e5"
)
_ADAPTER_REWARD_BUNDLE_SOURCE_SHA256: Final = (
    "22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469"
)
_SCORER_SOURCE_SHA256: Final = "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
_FORAGAX_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
_FORAGAX_BRIDGE_SOURCE_SHA256: Final = (
    "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
)
_FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256: Final = (
    "546009c19454a7839876df6e758b984db931db5eb234ac23833a232c387aa3bc"
)
_FULL_RAINBOW_RUNNER_SOURCE_SHA256: Final = (
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
)
_PPO_GRU_RUNNER_DESCRIPTOR_SHA256: Final = (
    "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
)
_PPO_GRU_RUNNER_SOURCE_SHA256: Final = (
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
)
_FORAGAX_INSTALL_TREE_SHA256: Final = (
    "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
)

QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_registry.v1"
)
QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_derivation.v1"
)
QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_trust_root_receipt.v1"
)
QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_drand_pulse_record.v1"
)
QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_seed_offline_verifier.v1"
)
QUALIFICATION_SEED_DERIVATION_DOMAIN: Final = (
    "alberta.forager.matched_v3.public_qualification.seed.v1"
)
_DRAND_QUICKNET_PROVIDER_ID: Final = "drand_quicknet"
_DRAND_QUICKNET_CHAIN_HASH: Final = (
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
_DRAND_QUICKNET_SIGNATURE_SCHEME: Final = "bls-unchained-g1-rfc9380"
_DRAND_QUICKNET_TIMESTAMP_SOURCE: Final = "drand_quicknet_round_time"
_DRAND_QUICKNET_PUBLIC_KEY_HEX: Final = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183"
    "c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4"
    "bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
_DRAND_QUICKNET_PUBLIC_KEY_RAW_SHA256: Final = (
    "96e74fcdd3a118406d3800a4e4935e67450a6befde915d47a0d6a13519cee134"
)
_DRAND_QUICKNET_GENESIS_TIME_UNIX: Final = 1_692_803_367
_DRAND_QUICKNET_PERIOD_SECONDS: Final = 3
_DRAND_QUICKNET_BLS_MESSAGE_SCOPE: Final = "unchained_round_only"
_DRAND_QUICKNET_RANDOMNESS_DERIVATION: Final = "sha256_raw_signature_bytes"
_SEED_PAIR_DERIVATION_ALGORITHM: Final = "sha256_canonical_json_high31_v1"
_OFFLINE_VERIFIER_IMPLEMENTATION_STATUS: Final = (
    "required_external_preaccepted_not_implemented_here"
)
_REQUIRED_RUNTIME_HELPER_IDS: Final = ("drand_verify", "oci_runtime", "resource_observer")
_ADAPTER_PUBLICATION_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
)

if (
    len(_DRAND_QUICKNET_PUBLIC_KEY_HEX) != 192
    or hashlib.sha256(bytes.fromhex(_DRAND_QUICKNET_PUBLIC_KEY_HEX)).hexdigest()
    != _DRAND_QUICKNET_PUBLIC_KEY_RAW_SHA256
):
    raise AssertionError("frozen Quicknet public-key identity drifted")

_REQUIRED_SOURCE_IDS: Final = ("external_foragax_agents", "local_alberta")
_SOURCE_MANIFEST_SCHEMAS: Final = {
    "external_foragax_agents": ("alberta.forager_matched_v3_external_materialization.v1"),
    "local_alberta": "alberta.forager_matched_v3.local_source_snapshot.v1",
}
_EXTERNAL_CANDIDATE_IDS: Final = (
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
_LOCAL_CANDIDATE_IDS: Final = (
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
_ADAPTER_CANDIDATE_IDS: Final = ("adapted_full_rainbow", "adapted_ppo_gru")
MATCHED_V3_QUALIFICATION_CANDIDATE_IDS: Final = (
    _LOCAL_CANDIDATE_IDS
    + _EXTERNAL_CANDIDATE_IDS[:9]
    + _ADAPTER_CANDIDATE_IDS
    + _EXTERNAL_CANDIDATE_IDS[9:]
)

_CONFIGURATION_RECORD_SHA256: Final = {
    "causal_e025_q050": "d780067bf7fc6582b7c30a4f7bcb14672ceb15d201fc76e0c7d6e233d0f0660c",
    "causal_e025_q075": "8c4488a4ca6a513731c5671cacdc55397aa7faaa07052ddb86000a53787aae8a",
    "causal_e025_q090": "130d243b230e8a9427f2f60b317eae463993f722ca45a275cb2d8398cff24afa",
    "causal_e050_q050": "373fb27a1566c280047b619c1c18f7065d4e11038c1b71939e1afa7d99ca1dda",
    "causal_e050_q075": "4e2ed83e2f40d6440e9b21f74c69f277abf23bd14e9383b5253e985dfcba731f",
    "causal_e050_q090": "deda929f9d606d08ed9c85c461eaeb8a7bc13c44e536d83eb861c94cbe2417dd",
    "causal_e100_q050": "7b6b85ec68afa398077170ac7fd90bb0256e4b687f4e2156dfc9e89f554aefca",
    "causal_e100_q075": "4b2b287c40d9a97d903e8150add4fd0190557befe9ab8a95cd2daa2c2d289afb",
    "causal_e100_q090": "9d4311599ba6eb46ad8098df0e57ca1fe2c1878cb0c62ea32830a4e3321652ff",
    "alberta_horde_default": "7dbd4f63c60484ffaadbd587c502de6d1079713cc4f044e54deadb6557b6a382",
    "alberta_horde_eps05": "73d818ae3ffaaedf5bbb40df5ff83d703fcd60c7a07a192436f3ba078d27e4b4",
    "alberta_horde_recurrent64": (
        "ac3fe6295280202a8a316d0a07e136cda3db806f9031163e3f767ae0bc0f30ea"
    ),
    "alberta_horde_step3e3": "cd11bac7f31e9a1a32c4a6c8c4706eea258bb040fdd21d220ed785a23a7ff014",
    "alberta_rtu_h08_taylor": "d804c8b79f29da16f085c7f1b4621ae479d780c3b23d30799367982353eb69df",
    "external_dqn_plain": "36af195ff30176f0b1d826fad4f2a7e3a820a5e378536c6635be9bb124caab5a",
    "external_dqn_crelu": "bed3acff23f2684a37e7d57f68f73c53f90c5037b0e5743bd6e39c0c2c420362",
    "external_dqn_redo": "6c65f8f7bfbcb92ef5b2bfd145ab0aa611bbae258b125032f7269af4eb0fd390",
    "external_dqn_reward_trace": (
        "9b78fdadd71f56c593eec14e8bd6d54e1e819053d06ee7d4f12c604d1eafd666"
    ),
    "external_dqn_l2_init": "a7b75eaee1ad9d62ff052373f8ba35b6ad5515af06e30baaa527a5c3cdcb5999",
    "external_pt_dqn_xfinal": ("79afe7a23bb01cbe7f4130bfe831e9c7c65be5f4bd833cfee9d307b942df6d8a"),
    "external_drqn_xfinal": "37bb4c7c17d9933a6ac54b064b81139a572ac43bc09163d62d14beb1ac6db387",
    "isolated_ppo_generic": "d88c5e7359085d362a235365e994117e46543e26e074491d2379e75efb499c47",
    "isolated_rtu_paper_scale": (
        "c01bfeb5af6af6c79b444214d09064fb23601d7a03d669ee9662d3181d139210"
    ),
    "adapted_full_rainbow": "4863d7d569def90d20f89a1aafa2e1984df93be1368070dffe655c7b2699d0b9",
    "adapted_ppo_gru": "4f8b429ff968213d0c05de87553456be7f2c1a67a806944357543025d725d7ca",
    "random_policy": "3646c050470e6ddbd817bae5512096c1225561367f486f1c2a5964e0848b2515",
    "search_nearest": "caf65fa4215b1c0d6a08b8ebbf6ffb481034eeb9f81d7ab0385d5181d45b685d",
    "search_oracle": "9214959547664a9d3d37e32ca472abcadb4b8e14d2e3d739b75b2d3721dbd5a8",
}

_PROBE_PROFILE_IDS: Final = (
    "qualification_seed_provenance_v1",
    "content_import_v1",
    "environment_rng_replay_v1",
    "candidate_seed_transport_v1",
    "full_horizon_resource_v1",
    "result_publication_roundtrip_v1",
)
_RECEIPT_SCHEMAS: Final = {
    "source_observation": "alberta.forager_matched_v3.source_observation.v1",
    "runtime_observation": "alberta.forager_matched_v3.runtime_observation.v1",
    "qualification_seed_observation": (
        "alberta.forager_matched_v3.qualification_seed_observation.v1"
    ),
    "candidate_observation": "alberta.forager_matched_v3.candidate_observation.v1",
    "resource_observation": "alberta.forager_matched_v3.resource_observation.v1",
    "publication_observation": ("alberta.forager_matched_v3.result_publication_observation.v1"),
    "fresh_replay_observation": ("alberta.forager_matched_v3.fresh_replay_observation.v1"),
    "qualification_bundle": "alberta.forager_matched_v3.qualification_bundle.v1",
}

_RESOURCE_FIELDS: Final = (
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
    "max_peak_rss_bytes",
    "max_cpu_time_ns",
    "max_wall_time_ns",
    "max_temporary_peak_bytes",
    "max_disk_peak_bytes",
    "max_thread_count",
    "max_attempt_count",
    "max_failure_count",
)

SourceId = Literal["external_foragax_agents", "local_alberta"]
SourceClosureKind = Literal["derived_checkout_manifest_tree", "normalized_local_source_snapshot"]
QualificationCaseClass = Literal["public_nonbenchmark_permanently_consumed"]


class ForagerMatchedV3QualificationPlanError(ValueError):
    """A qualification descriptor, typed input, or plan failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3QualificationPlanError(
        f"qualification JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3QualificationPlanError(
                f"qualification JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification JSON exceeds its node limit"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification JSON exceeds its depth limit"
            )
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(char) < 0x20 or ord(char) > 0x7E for char in item
            ):
                raise ForagerMatchedV3QualificationPlanError(
                    "qualification JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification JSON must contain only exact JSON scalar/container types"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3QualificationPlanError(
                        "qualification JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare already-bounded JSON values without bool/int aliasing."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_dict = cast(dict[str, Any], left)
        right_dict = cast(dict[str, Any], right)
        return left_dict.keys() == right_dict.keys() and all(
            _exact_json_equal(left_dict[key], right_dict[key]) for key in left_dict
        )
    if type(left) is list:
        left_list = left
        right_list = cast(list[Any], right)
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification canonical root must be a plain object"
        )
    _assert_plain_unaliased_json(value)
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
        raise ForagerMatchedV3QualificationPlanError(
            "qualification value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact exceeds its canonical byte limit"
        )
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact input must be exact bytes"
        )
    if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact violates its byte bound"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact must have one canonical trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3QualificationPlanError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact is not bounded strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification artifact is not in exact canonical form"
        )
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3QualificationPlanError(f"{label} keys are not exact")


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must be a bounded lowercase identifier"
        )
    return value


def _require_version(value: Any, label: str) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must be a bounded printable version identity"
        )
    return value


def _require_relative_path(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) > _MAX_TEXT_LENGTH
        or _RELATIVE_PATH_RE.fullmatch(value) is None
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must be a bounded normalized relative POSIX path"
        )
    return value


def _require_bounded_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_RESOURCE_VALUE,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


@dataclass(frozen=True, slots=True)
class SourceRequirement:
    """Content identity for one already-produced, caller-observed source closure."""

    source_id: SourceId
    closure_kind: SourceClosureKind
    manifest_schema_version: str
    manifest_file_sha256: str
    manifest_body_sha256: str
    source_tree_sha256: str
    inventory_sha256: str
    file_count: int
    directory_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        expected_kind = {
            "external_foragax_agents": "derived_checkout_manifest_tree",
            "local_alberta": "normalized_local_source_snapshot",
        }
        if (
            self.source_id not in expected_kind
            or self.closure_kind != expected_kind[self.source_id]
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "source closure kind does not match its ID"
            )
        if self.manifest_schema_version != _SOURCE_MANIFEST_SCHEMAS[self.source_id]:
            raise ForagerMatchedV3QualificationPlanError("source manifest schema drifted")
        _require_sha256(self.manifest_file_sha256, "source manifest file")
        _require_sha256(self.manifest_body_sha256, "source manifest body")
        _require_sha256(self.source_tree_sha256, "source tree")
        _require_sha256(self.inventory_sha256, "source inventory")
        _require_bounded_int(
            self.file_count, "source file count", minimum=1, maximum=_MAX_SOURCE_ENTRIES
        )
        _require_bounded_int(
            self.directory_count,
            "source directory count",
            maximum=_MAX_SOURCE_ENTRIES,
        )
        if self.file_count + self.directory_count > _MAX_SOURCE_ENTRIES:
            raise ForagerMatchedV3QualificationPlanError("source entry count exceeds its bound")
        _require_bounded_int(self.total_bytes, "source total bytes", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "closure_kind": self.closure_kind,
            "manifest_schema_version": self.manifest_schema_version,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_body_sha256": self.manifest_body_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "inventory_sha256": self.inventory_sha256,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True, slots=True)
class QualificationSeedTrustRootReceiptBinding:
    """Independent content identity needed to authenticate one seed registry.

    The receipt itself is produced and preaccepted outside this module.  This
    class freezes the exact Quicknet chain, signature scheme, pulse proof
    record, beacon round/time/key/signature/randomness identities, derived
    registry, and offline verifier source that an eventual authority must
    verify.  Quicknet signs its unchained round protocol message, not this
    receipt or the Alberta seed registry.  Constructing this binding does not
    perform or claim signature verification or preacceptance chronology.
    """

    receipt_schema_version: str
    receipt_file_sha256: str
    receipt_body_sha256: str
    provider_id: str
    provider_chain_hash: str
    signature_scheme: str
    provider_public_key_sha256: str
    beacon_round: int
    beacon_time_unix: int
    observation_cutoff_unix: int
    beacon_signature_sha256: str
    beacon_randomness_hex: str
    pulse_record_schema_version: str
    pulse_record_file_sha256: str
    pulse_record_body_sha256: str
    seed_registry_schema_version: str
    seed_registry_file_sha256: str
    seed_registry_body_sha256: str
    seed_derivation_algorithm: str
    timestamp_source: str
    offline_verifier_schema_version: str
    offline_verifier_descriptor_sha256: str
    offline_verifier_source_id: SourceId
    offline_verifier_source_tree_sha256: str
    offline_verifier_implementation_path: str
    offline_verifier_source_sha256: str
    offline_verifier_runtime_helper_id: str
    offline_verifier_executable_sha256: str
    offline_verifier_version_output_sha256: str
    offline_verifier_implementation_status: str
    offline_signature_verification_required: bool
    external_preacceptance_required: bool

    def __post_init__(self) -> None:
        if self.receipt_schema_version != QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed trust-root receipt schema drifted"
            )
        _require_sha256(self.receipt_file_sha256, "qualification seed trust-root receipt file")
        _require_sha256(self.receipt_body_sha256, "qualification seed trust-root receipt body")
        if self.provider_id != _DRAND_QUICKNET_PROVIDER_ID:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed provider is not the frozen Quicknet provider"
            )
        if self.provider_chain_hash != _DRAND_QUICKNET_CHAIN_HASH:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed provider chain hash drifted"
            )
        if self.signature_scheme != _DRAND_QUICKNET_SIGNATURE_SCHEME:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed signature scheme drifted"
            )
        if self.provider_public_key_sha256 != _DRAND_QUICKNET_PUBLIC_KEY_RAW_SHA256:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed provider public key is not the frozen Quicknet key"
            )
        _require_bounded_int(
            self.beacon_round,
            "qualification seed beacon round",
            minimum=1,
        )
        _require_bounded_int(
            self.beacon_time_unix,
            "qualification seed beacon time",
            minimum=1,
        )
        _require_bounded_int(
            self.observation_cutoff_unix,
            "qualification seed observation cutoff",
            minimum=1,
        )
        expected_beacon_time_unix = (
            _DRAND_QUICKNET_GENESIS_TIME_UNIX
            + (self.beacon_round - 1) * _DRAND_QUICKNET_PERIOD_SECONDS
        )
        if self.beacon_time_unix != expected_beacon_time_unix:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed beacon time is not the exact Quicknet round time"
            )
        if self.beacon_time_unix >= self.observation_cutoff_unix:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed beacon time must precede the observation cutoff"
            )
        _require_sha256(
            self.beacon_signature_sha256,
            "qualification seed beacon signature",
        )
        _require_sha256(self.beacon_randomness_hex, "qualification seed beacon randomness")
        if not hmac.compare_digest(
            self.beacon_signature_sha256,
            self.beacon_randomness_hex,
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed randomness must equal SHA-256 of the raw signature"
            )
        if self.pulse_record_schema_version != QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed pulse-record schema drifted"
            )
        _require_sha256(
            self.pulse_record_file_sha256,
            "qualification seed pulse-record file",
        )
        _require_sha256(
            self.pulse_record_body_sha256,
            "qualification seed pulse-record body",
        )
        if self.seed_registry_schema_version != QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed trust-root receipt registry schema drifted"
            )
        _require_sha256(
            self.seed_registry_file_sha256,
            "qualification seed trust-root receipt registry file",
        )
        _require_sha256(
            self.seed_registry_body_sha256,
            "qualification seed trust-root receipt registry body",
        )
        if self.seed_derivation_algorithm != _SEED_PAIR_DERIVATION_ALGORITHM:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed pair-derivation algorithm drifted"
            )
        if self.timestamp_source != _DRAND_QUICKNET_TIMESTAMP_SOURCE:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed timestamp source drifted"
            )
        if (
            self.offline_verifier_schema_version
            != QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed offline-verifier schema drifted"
            )
        _require_sha256(
            self.offline_verifier_descriptor_sha256,
            "qualification seed offline-verifier descriptor",
        )
        if self.offline_verifier_source_id != "local_alberta":
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed offline verifier must live in the local harness closure"
            )
        _require_sha256(
            self.offline_verifier_source_tree_sha256,
            "qualification seed offline-verifier source tree",
        )
        _require_relative_path(
            self.offline_verifier_implementation_path,
            "qualification seed offline-verifier implementation path",
        )
        _require_sha256(
            self.offline_verifier_source_sha256,
            "qualification seed offline-verifier source",
        )
        if self.offline_verifier_runtime_helper_id != "drand_verify":
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed offline verifier runtime helper drifted"
            )
        _require_sha256(
            self.offline_verifier_executable_sha256,
            "qualification seed offline-verifier executable",
        )
        _require_sha256(
            self.offline_verifier_version_output_sha256,
            "qualification seed offline-verifier version output",
        )
        if self.offline_verifier_implementation_status != _OFFLINE_VERIFIER_IMPLEMENTATION_STATUS:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed offline-verifier status became authoritative"
            )
        if (
            type(self.offline_signature_verification_required) is not bool
            or self.offline_signature_verification_required is not True
            or type(self.external_preacceptance_required) is not bool
            or self.external_preacceptance_required is not True
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed authentication requires offline verification and preacceptance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_schema_version": self.receipt_schema_version,
            "receipt_file_sha256": self.receipt_file_sha256,
            "receipt_body_sha256": self.receipt_body_sha256,
            "provider_id": self.provider_id,
            "provider_chain_hash": self.provider_chain_hash,
            "signature_scheme": self.signature_scheme,
            "provider_public_key_sha256": self.provider_public_key_sha256,
            "beacon_round": self.beacon_round,
            "beacon_time_unix": self.beacon_time_unix,
            "observation_cutoff_unix": self.observation_cutoff_unix,
            "beacon_signature_sha256": self.beacon_signature_sha256,
            "beacon_randomness_hex": self.beacon_randomness_hex,
            "pulse_record_schema_version": self.pulse_record_schema_version,
            "pulse_record_file_sha256": self.pulse_record_file_sha256,
            "pulse_record_body_sha256": self.pulse_record_body_sha256,
            "seed_registry_schema_version": self.seed_registry_schema_version,
            "seed_registry_file_sha256": self.seed_registry_file_sha256,
            "seed_registry_body_sha256": self.seed_registry_body_sha256,
            "seed_derivation_algorithm": self.seed_derivation_algorithm,
            "timestamp_source": self.timestamp_source,
            "offline_verifier_schema_version": self.offline_verifier_schema_version,
            "offline_verifier_descriptor_sha256": self.offline_verifier_descriptor_sha256,
            "offline_verifier_source_id": self.offline_verifier_source_id,
            "offline_verifier_source_tree_sha256": self.offline_verifier_source_tree_sha256,
            "offline_verifier_implementation_path": (self.offline_verifier_implementation_path),
            "offline_verifier_source_sha256": self.offline_verifier_source_sha256,
            "offline_verifier_runtime_helper_id": self.offline_verifier_runtime_helper_id,
            "offline_verifier_executable_sha256": self.offline_verifier_executable_sha256,
            "offline_verifier_version_output_sha256": (self.offline_verifier_version_output_sha256),
            "offline_verifier_implementation_status": (self.offline_verifier_implementation_status),
            "offline_signature_verification_required": (
                self.offline_signature_verification_required
            ),
            "external_preacceptance_required": self.external_preacceptance_required,
        }


@dataclass(frozen=True, slots=True)
class QualificationSeedRegistryBinding:
    """Identity of a public seed registry that still requires external authentication."""

    registry_schema_version: str
    registry_file_sha256: str
    registry_body_sha256: str
    derivation_schema_version: str
    derivation_domain: str
    provider_id: str
    provider_identity_sha256: str
    provider_receipt_schema_version: str
    provider_receipt_file_sha256: str
    provider_receipt_body_sha256: str
    trust_root_receipt_binding_sha256: str
    external_authentication_required: bool
    issued_before_observation_required: bool

    def __post_init__(self) -> None:
        if self.registry_schema_version != QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed registry schema drifted"
            )
        _require_sha256(self.registry_file_sha256, "qualification seed registry file")
        _require_sha256(self.registry_body_sha256, "qualification seed registry body")
        if self.derivation_schema_version != QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed derivation schema drifted"
            )
        if self.derivation_domain != QUALIFICATION_SEED_DERIVATION_DOMAIN:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed derivation domain drifted"
            )
        if self.provider_id != _DRAND_QUICKNET_PROVIDER_ID:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed registry provider is not frozen"
            )
        if self.provider_identity_sha256 != _DRAND_QUICKNET_CHAIN_HASH:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed registry provider identity drifted"
            )
        if (
            self.provider_receipt_schema_version
            != QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seed provider receipt schema drifted"
            )
        _require_sha256(
            self.provider_receipt_file_sha256,
            "qualification seed provider receipt file",
        )
        _require_sha256(
            self.provider_receipt_body_sha256,
            "qualification seed provider receipt body",
        )
        _require_sha256(
            self.trust_root_receipt_binding_sha256,
            "qualification seed trust-root receipt binding",
        )
        if (
            type(self.external_authentication_required) is not bool
            or self.external_authentication_required is not True
            or type(self.issued_before_observation_required) is not bool
            or self.issued_before_observation_required is not True
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification seeds require external authentication before observation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_schema_version": self.registry_schema_version,
            "registry_file_sha256": self.registry_file_sha256,
            "registry_body_sha256": self.registry_body_sha256,
            "derivation_schema_version": self.derivation_schema_version,
            "derivation_domain": self.derivation_domain,
            "provider_id": self.provider_id,
            "provider_identity_sha256": self.provider_identity_sha256,
            "provider_receipt_schema_version": self.provider_receipt_schema_version,
            "provider_receipt_file_sha256": self.provider_receipt_file_sha256,
            "provider_receipt_body_sha256": self.provider_receipt_body_sha256,
            "trust_root_receipt_binding_sha256": self.trust_root_receipt_binding_sha256,
            "external_authentication_required": self.external_authentication_required,
            "issued_before_observation_required": (self.issued_before_observation_required),
        }


@dataclass(frozen=True, slots=True)
class RuntimeHelperBinding:
    """Content identity for one runtime helper executable."""

    helper_id: str
    executable_sha256: str
    version_output_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.helper_id, "runtime helper ID")
        _require_sha256(self.executable_sha256, "runtime helper executable")
        _require_sha256(self.version_output_sha256, "runtime helper version output")

    def to_dict(self) -> dict[str, str]:
        return {
            "helper_id": self.helper_id,
            "executable_sha256": self.executable_sha256,
            "version_output_sha256": self.version_output_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    """Complete, non-optional identity required before any qualification probe."""

    executor_kind: str
    runtime_executable_sha256: str
    runtime_version_output_sha256: str
    image_digest: str
    image_config_sha256: str
    runtime_profile_sha256: str
    python_implementation: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    foragax_version: str
    foragax_install_tree_sha256: str
    platform: str
    default_prng_impl: str
    jax_enable_x64: bool
    threefry_partitionable: bool
    sandbox_descriptor_sha256: str
    helper_bindings: tuple[RuntimeHelperBinding, ...]

    def __post_init__(self) -> None:
        if self.executor_kind != "networkless_oci_cpu":
            raise ForagerMatchedV3QualificationPlanError("runtime executor kind is not frozen")
        _require_sha256(self.runtime_executable_sha256, "runtime executable")
        _require_sha256(self.runtime_version_output_sha256, "runtime version output")
        if type(self.image_digest) is not str or not self.image_digest.startswith("sha256:"):
            raise ForagerMatchedV3QualificationPlanError("runtime image digest is invalid")
        _require_sha256(
            self.image_digest.removeprefix("sha256:"),
            "runtime image digest",
        )
        _require_sha256(self.image_config_sha256, "runtime image config")
        _require_sha256(self.runtime_profile_sha256, "runtime profile")
        if self.python_implementation != "CPython":
            raise ForagerMatchedV3QualificationPlanError(
                "runtime Python implementation must be exact CPython"
            )
        if (
            type(self.python_version) is not str
            or _CPYTHON_312_VERSION_RE.fullmatch(self.python_version) is None
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "runtime Python version must be an exact CPython 3.12.x release"
            )
        if self.jax_version != "0.11.0" or self.jaxlib_version != "0.11.0":
            raise ForagerMatchedV3QualificationPlanError("JAX/JAXlib version is not bridge-bound")
        if self.foragax_version != "0.55.0":
            raise ForagerMatchedV3QualificationPlanError("Foragax version is not bridge-bound")
        if self.foragax_install_tree_sha256 != _FORAGAX_INSTALL_TREE_SHA256:
            raise ForagerMatchedV3QualificationPlanError("Foragax install tree is not bridge-bound")
        if self.platform != "linux/amd64":
            raise ForagerMatchedV3QualificationPlanError("runtime platform is not frozen")
        if self.default_prng_impl != "threefry2x32":
            raise ForagerMatchedV3QualificationPlanError("runtime PRNG implementation drifted")
        if type(self.jax_enable_x64) is not bool or self.jax_enable_x64 is not False:
            raise ForagerMatchedV3QualificationPlanError("JAX x64 mode must be exact false")
        if type(self.threefry_partitionable) is not bool or self.threefry_partitionable is not True:
            raise ForagerMatchedV3QualificationPlanError(
                "partitionable Threefry mode must be exact true"
            )
        _require_sha256(self.sandbox_descriptor_sha256, "runtime sandbox descriptor")
        if type(self.helper_bindings) is not tuple or not self.helper_bindings:
            raise ForagerMatchedV3QualificationPlanError(
                "runtime helper bindings must be a nonempty exact tuple"
            )
        if any(type(item) is not RuntimeHelperBinding for item in self.helper_bindings):
            raise ForagerMatchedV3QualificationPlanError("runtime helper binding type is invalid")
        helper_ids = tuple(item.helper_id for item in self.helper_bindings)
        if helper_ids != _REQUIRED_RUNTIME_HELPER_IDS:
            raise ForagerMatchedV3QualificationPlanError(
                "runtime helper bindings must be the exact sorted required helper set"
            )
        if len({id(item) for item in self.helper_bindings}) != len(self.helper_bindings):
            raise ForagerMatchedV3QualificationPlanError("runtime helper bindings are aliased")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor_kind": self.executor_kind,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "runtime_version_output_sha256": self.runtime_version_output_sha256,
            "image_digest": self.image_digest,
            "image_config_sha256": self.image_config_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "foragax_version": self.foragax_version,
            "foragax_install_tree_sha256": self.foragax_install_tree_sha256,
            "platform": self.platform,
            "default_prng_impl": self.default_prng_impl,
            "jax_enable_x64": self.jax_enable_x64,
            "threefry_partitionable": self.threefry_partitionable,
            "sandbox_descriptor_sha256": self.sandbox_descriptor_sha256,
            "helper_bindings": [item.to_dict() for item in self.helper_bindings],
        }


@dataclass(frozen=True, slots=True)
class QualificationCase:
    """One public, nonbenchmark case permanently consumed by qualification."""

    case_id: str
    candidate_id: str
    material_class: QualificationCaseClass
    registry_case_ordinal: int
    seed_registry_binding_sha256: str
    seed_registry_file_sha256: str
    seed_registry_body_sha256: str
    provider_identity_sha256: str
    provider_receipt_file_sha256: str
    provider_receipt_body_sha256: str
    derivation_schema_version: str
    derivation_domain: str
    derivation_payload_sha256: str
    environment_seed: int
    agent_seed: int
    environment_seed_derivation_sha256: str
    agent_seed_derivation_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, "qualification case ID")
        if _FORBIDDEN_CASE_MATERIAL_RE.search(self.case_id) is not None:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case references forbidden seed material"
            )
        if self.candidate_id not in MATCHED_V3_QUALIFICATION_CANDIDATE_IDS:
            raise ForagerMatchedV3QualificationPlanError("qualification candidate is unknown")
        if self.material_class != "public_nonbenchmark_permanently_consumed":
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case material class is not permanently consumed public material"
            )
        _require_bounded_int(
            self.registry_case_ordinal,
            "qualification registry case ordinal",
            maximum=len(MATCHED_V3_QUALIFICATION_CANDIDATE_IDS) - 1,
        )
        _require_sha256(
            self.seed_registry_binding_sha256,
            "qualification case seed-registry binding",
        )
        _require_sha256(
            self.seed_registry_file_sha256,
            "qualification case seed-registry file",
        )
        _require_sha256(
            self.seed_registry_body_sha256,
            "qualification case seed-registry body",
        )
        _require_sha256(
            self.provider_identity_sha256,
            "qualification case provider identity",
        )
        _require_sha256(
            self.provider_receipt_file_sha256,
            "qualification case provider receipt file",
        )
        _require_sha256(
            self.provider_receipt_body_sha256,
            "qualification case provider receipt body",
        )
        if self.derivation_schema_version != QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case derivation schema drifted"
            )
        if self.derivation_domain != QUALIFICATION_SEED_DERIVATION_DOMAIN:
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case derivation domain drifted"
            )
        _require_sha256(
            self.derivation_payload_sha256,
            "qualification case derivation payload",
        )
        _require_bounded_int(
            self.environment_seed, "qualification environment seed", maximum=_UINT31_MAX
        )
        _require_bounded_int(self.agent_seed, "qualification agent seed", maximum=_UINT31_MAX)
        _require_sha256(
            self.environment_seed_derivation_sha256,
            "qualification environment-seed derivation",
        )
        _require_sha256(
            self.agent_seed_derivation_sha256,
            "qualification agent-seed derivation",
        )
        if hmac.compare_digest(
            self.environment_seed_derivation_sha256,
            self.agent_seed_derivation_sha256,
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "environment and agent seed derivation identities must be distinct"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "material_class": self.material_class,
            "registry_case_ordinal": self.registry_case_ordinal,
            "seed_registry_binding_sha256": self.seed_registry_binding_sha256,
            "seed_registry_file_sha256": self.seed_registry_file_sha256,
            "seed_registry_body_sha256": self.seed_registry_body_sha256,
            "provider_identity_sha256": self.provider_identity_sha256,
            "provider_receipt_file_sha256": self.provider_receipt_file_sha256,
            "provider_receipt_body_sha256": self.provider_receipt_body_sha256,
            "derivation_schema_version": self.derivation_schema_version,
            "derivation_domain": self.derivation_domain,
            "derivation_payload_sha256": self.derivation_payload_sha256,
            "environment_seed": self.environment_seed,
            "agent_seed": self.agent_seed,
            "environment_seed_derivation_sha256": (self.environment_seed_derivation_sha256),
            "agent_seed_derivation_sha256": self.agent_seed_derivation_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateResourceRequirement:
    """Pre-observation integer ceilings for one candidate qualification case."""

    candidate_id: str
    max_environment_interactions: int
    max_optimizer_updates: int
    max_gradient_updates: int
    max_sample_updates: int
    max_trainable_parameters: int
    max_frozen_parameters: int
    max_optimizer_state_elements: int
    max_optimizer_state_bytes: int
    max_target_copy_elements: int
    max_target_copy_bytes: int
    max_replay_capacity_transitions: int
    max_replay_peak_bytes: int
    max_rollout_storage_elements: int
    max_rollout_peak_bytes: int
    max_recurrent_carry_elements: int
    max_recurrent_carry_bytes: int
    max_rtrl_sensitivity_elements: int
    max_rtrl_sensitivity_bytes: int
    max_eligibility_elements: int
    max_eligibility_bytes: int
    max_peak_rss_bytes: int
    max_cpu_time_ns: int
    max_wall_time_ns: int
    max_temporary_peak_bytes: int
    max_disk_peak_bytes: int
    max_thread_count: int
    max_attempt_count: int
    max_failure_count: int

    def __post_init__(self) -> None:
        if self.candidate_id not in MATCHED_V3_QUALIFICATION_CANDIDATE_IDS:
            raise ForagerMatchedV3QualificationPlanError("resource candidate is unknown")
        for field_name in _RESOURCE_FIELDS:
            _require_bounded_int(getattr(self, field_name), f"resource field {field_name}")
        if self.max_environment_interactions < _HORIZON:
            raise ForagerMatchedV3QualificationPlanError(
                "resource ceiling cannot cover one exact qualification horizon"
            )
        if self.max_thread_count < 1 or self.max_attempt_count < 1:
            raise ForagerMatchedV3QualificationPlanError(
                "resource thread and attempt ceilings must be positive"
            )
        if self.max_failure_count >= self.max_attempt_count:
            raise ForagerMatchedV3QualificationPlanError(
                "resource failure ceiling must be smaller than attempt ceiling"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            **{field_name: getattr(self, field_name) for field_name in _RESOURCE_FIELDS},
        }


@dataclass(frozen=True, slots=True)
class CandidatePublicationBinding:
    """Required content binding for one candidate's eventual result publisher."""

    candidate_id: str
    publisher_kind: str
    descriptor_schema_version: str
    descriptor_sha256: str
    publication_schema_version: str
    source_id: SourceId
    source_tree_sha256: str
    implementation_path: str
    implementation_source_sha256: str
    reload_validator_schema_version: str
    reload_validator_descriptor_sha256: str
    reload_validator_implementation_path: str
    reload_validator_source_sha256: str

    def __post_init__(self) -> None:
        if self.candidate_id not in MATCHED_V3_QUALIFICATION_CANDIDATE_IDS:
            raise ForagerMatchedV3QualificationPlanError("publication candidate is unknown")
        _require_identifier(self.publisher_kind, "publication kind")
        _require_version(self.descriptor_schema_version, "publication descriptor schema")
        _require_sha256(self.descriptor_sha256, "publication descriptor")
        _require_version(self.publication_schema_version, "publication schema")
        if self.source_id != "local_alberta":
            raise ForagerMatchedV3QualificationPlanError(
                "result publishers must live in the local harness source closure"
            )
        _require_sha256(self.source_tree_sha256, "publication source tree")
        _require_relative_path(self.implementation_path, "publication implementation path")
        _require_sha256(self.implementation_source_sha256, "publication source")
        _require_version(
            self.reload_validator_schema_version,
            "publication reload-validator schema",
        )
        _require_sha256(
            self.reload_validator_descriptor_sha256,
            "publication reload-validator descriptor",
        )
        _require_relative_path(
            self.reload_validator_implementation_path,
            "publication reload-validator implementation path",
        )
        _require_sha256(
            self.reload_validator_source_sha256,
            "publication reload-validator source",
        )
        if self.candidate_id in _ADAPTER_CANDIDATE_IDS and (
            self.publisher_kind != "adapter_reward_publication_v1"
            or self.descriptor_schema_version
            != "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            or self.descriptor_sha256 != _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
            or self.publication_schema_version
            != "alberta.forager_matched_v3.adapter_reward_publication.v1"
            or self.source_id != "local_alberta"
            or self.implementation_path != _ADAPTER_PUBLICATION_IMPLEMENTATION_PATH
            or self.implementation_source_sha256 != _ADAPTER_REWARD_PUBLICATION_SOURCE_SHA256
            or self.reload_validator_schema_version
            != "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            or self.reload_validator_descriptor_sha256
            != _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
            or self.reload_validator_implementation_path != _ADAPTER_PUBLICATION_IMPLEMENTATION_PATH
            or self.reload_validator_source_sha256 != _ADAPTER_REWARD_PUBLICATION_SOURCE_SHA256
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "adapter result publisher differs from its implemented content binding"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "publisher_kind": self.publisher_kind,
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "publication_schema_version": self.publication_schema_version,
            "source_id": self.source_id,
            "source_tree_sha256": self.source_tree_sha256,
            "implementation_path": self.implementation_path,
            "implementation_source_sha256": self.implementation_source_sha256,
            "reload_validator_schema_version": self.reload_validator_schema_version,
            "reload_validator_descriptor_sha256": (self.reload_validator_descriptor_sha256),
            "reload_validator_implementation_path": (self.reload_validator_implementation_path),
            "reload_validator_source_sha256": self.reload_validator_source_sha256,
        }


def _dependencies() -> dict[str, dict[str, str]]:
    return {
        "configuration_plan": {
            "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
        "candidate_universe": {
            "schema_version": "alberta.forager_matched_v3_development_universe.v1",
            "sha256": _CANDIDATE_UNIVERSE_SHA256,
        },
        "cumulative_reward_metric": {
            "schema_version": "alberta.forager_cumulative_reward_metric.v1",
            "sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
        },
        "trial_block_generator_plan": {
            "schema_version": "alberta.forager_trial_block_generator_plan.v1",
            "sha256": _TRIAL_BLOCK_GENERATOR_PLAN_SHA256,
        },
        "external_materializer": {
            "schema_version": ("alberta.forager_matched_v3_external_materialization_identity.v1"),
            "sha256": _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256,
            "source_sha256": _EXTERNAL_MATERIALIZATION_SOURCE_SHA256,
        },
        "adapter_reward_publication": {
            "schema_version": (
                "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
            ),
            "descriptor_sha256": _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
            "source_sha256": _ADAPTER_REWARD_PUBLICATION_SOURCE_SHA256,
        },
        "adapter_reward_bundle": {
            "schema_version": ("alberta.forager_matched_v3.adapter_reward_bundle_descriptor.v1"),
            "descriptor_sha256": _ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256,
            "source_sha256": _ADAPTER_REWARD_BUNDLE_SOURCE_SHA256,
        },
        "reward_scorer": {"source_sha256": _SCORER_SOURCE_SHA256},
        "foragax_bridge": {
            "schema_version": "alberta.forager_matched_v3_foragax_bridge.v2",
            "descriptor_sha256": _FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
            "source_sha256": _FORAGAX_BRIDGE_SOURCE_SHA256,
        },
        "full_rainbow_runner": {
            "schema_version": "alberta.forager_matched_v3.full_rainbow_runner.v1",
            "descriptor_sha256": _FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": _FULL_RAINBOW_RUNNER_SOURCE_SHA256,
        },
        "ppo_gru_runner": {
            "schema_version": "alberta.forager_matched_v3_ppo_gru_runner.v1",
            "descriptor_sha256": _PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
            "source_sha256": _PPO_GRU_RUNNER_SOURCE_SHA256,
        },
    }


def _canonicalization_policy() -> dict[str, Any]:
    return {
        "format": "json",
        "encoding": "ascii",
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
        "separators": [",", ":"],
        "trailing_newline": True,
        "duplicate_keys_rejected": True,
        "container_aliases_rejected": True,
        "maximum_bytes": _MAX_ARTIFACT_BYTES,
        "maximum_depth": _MAX_JSON_DEPTH,
        "maximum_nodes": _MAX_JSON_NODES,
    }


def _authentication_policy() -> dict[str, bool]:
    return {
        "development_external_authentication_required": False,
        "qualification_case_external_authentication_required": True,
        "qualification_seed_registry_issued_before_observation_required": True,
        "qualification_seed_trust_root_receipt_external_pin_required": True,
        "qualification_seed_offline_signature_verification_required": True,
        "qualification_seed_offline_signature_verification_implemented_here": False,
        "qualification_seed_preacceptance_chronology_verified_here": False,
        "confirmatory_external_authentication_required": True,
        "execution_authority_separate_from_content_validation": True,
        "serialized_artifact_grants_execution_capability": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "production_plan_issued": False,
        "source_closure_accepted": False,
        "runtime_qualified": False,
        "candidate_qualified": False,
        "qualification_executed": False,
        "benchmark_executed": False,
        "result_observed": False,
        "execution_ready": False,
        "execution_authorized": False,
        "ingestion_authorized": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This artifact is an unexecuted content contract, not a qualification receipt.",
        "No source closure, runtime, candidate, resource observation, or replay is accepted.",
        "Qualification seeds require an independently pinned pre-observation trust-root receipt.",
        "The mandatory offline signature verifier is not implemented or executed by this module.",
        "Quicknet signs only its unchained round message, not this registry or receipt.",
        "A valid or historical Quicknet pulse does not prove receipt preacceptance timing.",
        "Preacceptance chronology remains an external unimplemented requirement.",
        "Reward magnitude, scores, ranks, and observed performance cannot affect acceptance.",
        "External confirmatory authentication and separate execution authority remain required.",
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_PLAN_DESCRIPTOR_SCHEMA_VERSION,
        "status": QUALIFICATION_PLAN_STATUS,
        "classification": QUALIFICATION_PLAN_CLASSIFICATION,
        "dependencies": _dependencies(),
        "required_source_ids": list(_REQUIRED_SOURCE_IDS),
        "candidate_order": list(MATCHED_V3_QUALIFICATION_CANDIDATE_IDS),
        "receipt_schemas": dict(_RECEIPT_SCHEMAS),
        "probe_profile_ids": list(_PROBE_PROFILE_IDS),
        "runtime_contract": {
            "executor_kind": "networkless_oci_cpu",
            "python_implementation": "CPython",
            "python_version_series": "3.12.x",
            "jax_version": "0.11.0",
            "jaxlib_version": "0.11.0",
            "foragax_version": "0.55.0",
            "required_helper_ids": list(_REQUIRED_RUNTIME_HELPER_IDS),
            "networkless_oci_workflow": [
                "verify_image_runtime_and_helper_content_identities",
                "disable_network_before_candidate_import",
                "record_exact_version_outputs_and_runtime_profile",
                "run_only_score_blind_qualification_probes",
            ],
        },
        "qualification_seed_registry_contract": {
            "registry_schema_version": QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
            "derivation_schema_version": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
            "derivation_domain": QUALIFICATION_SEED_DERIVATION_DOMAIN,
            "provider_receipt_schema_version": (QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION),
            "pulse_record_schema_version": QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION,
            "offline_verifier_schema_version": (QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION),
            "provider_id": _DRAND_QUICKNET_PROVIDER_ID,
            "provider_chain_hash": _DRAND_QUICKNET_CHAIN_HASH,
            "signature_scheme": _DRAND_QUICKNET_SIGNATURE_SCHEME,
            "provider_public_key_hex": _DRAND_QUICKNET_PUBLIC_KEY_HEX,
            "provider_public_key_raw_sha256": _DRAND_QUICKNET_PUBLIC_KEY_RAW_SHA256,
            "provider_public_key_hash_input": "hex_decoded_96_bytes",
            "genesis_time_unix": _DRAND_QUICKNET_GENESIS_TIME_UNIX,
            "period_seconds": _DRAND_QUICKNET_PERIOD_SECONDS,
            "bls_message_scope": _DRAND_QUICKNET_BLS_MESSAGE_SCOPE,
            "randomness_derivation": _DRAND_QUICKNET_RANDOMNESS_DERIVATION,
            "timestamp_source": _DRAND_QUICKNET_TIMESTAMP_SOURCE,
            "seed_pair_derivation_algorithm": _SEED_PAIR_DERIVATION_ALGORITHM,
            "offline_verifier_runtime_helper_id": "drand_verify",
            "offline_verifier_runtime_helper_content_bound": True,
            "offline_verifier_implementation_status": (_OFFLINE_VERIFIER_IMPLEMENTATION_STATUS),
            "full_file_and_body_digests_required": True,
            "every_case_binds_registry_receipt_and_derivation_payload": True,
            "receipt_binds_pulse_record_key_round_time_signature_randomness": True,
            "pulse_record_requires_raw_public_key_and_signature": True,
            "quicknet_signature_authenticates_seed_registry": False,
            "quicknet_signature_authenticates_trust_root_receipt": False,
            "registry_seed_pairs_deterministically_derived_from_pulse": True,
            "deterministic_seed_derivation_implemented_here": True,
            "canonical_registry_file_and_body_derivation_implemented_here": True,
            "independent_trust_root_receipt_file_pin_required": True,
            "independent_trust_root_receipt_binding_pin_required": True,
            "offline_signature_verification_required": True,
            "offline_signature_verification_implemented_here": False,
            "pulse_time_alone_proves_preobservation": False,
            "external_preacceptance_chronology_required": True,
            "external_preacceptance_chronology_implemented_here": False,
            "external_authentication_required": True,
            "issued_before_observation_required": True,
            "issuer_api_exposed": False,
        },
        "publication_roundtrip_contract": {
            "all_candidates_require_source_closure_membership": True,
            "reload_validator_binding_required": True,
            "atomic_publication_required": True,
            "strict_reload_required": True,
            "full_digest_equivalence_required": True,
            "score_or_reward_magnitude_observed": False,
        },
        "canonicalization": _canonicalization_policy(),
        "authentication_policy": _authentication_policy(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
QUALIFICATION_PLAN_DESCRIPTOR_SHA256: Final = (
    "258b9e376b82127f912bf2828a6d4e5c7a257ed2a990cd15bf4c9cbd81c17788"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    QUALIFICATION_PLAN_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 qualification-plan descriptor identity drifted")


def _probe_profiles() -> list[dict[str, Any]]:
    return [
        {
            "profile_id": "qualification_seed_provenance_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["qualification_seed_observation"],
            "acceptance_fields": [
                "registry_full_file_and_body_digests_exact",
                "independent_trust_root_receipt_file_pin_exact",
                "independent_trust_root_receipt_binding_pin_exact",
                "provider_chain_public_key_and_signature_scheme_exact",
                "pulse_record_exact",
                "beacon_round_time_signature_and_randomness_exact",
                "offline_verifier_source_closure_membership_exact",
                "offline_signature_verification_exact",
                "deterministic_28_case_seed_pair_derivation_exact",
                "deterministic_registry_file_and_body_digests_exact",
                "derivation_schema_and_domain_exact",
                "case_derivation_payload_membership_exact",
                "beacon_time_precedes_observation_cutoff_exact",
                "external_receipt_preacceptance_chronology_exact",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
        {
            "profile_id": "content_import_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["candidate_observation"],
            "acceptance_fields": [
                "source_membership_exact",
                "configuration_membership_exact",
                "entrypoint_import_exact",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
        {
            "profile_id": "environment_rng_replay_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["fresh_replay_observation"],
            "acceptance_fields": [
                "environment_seed_transport_exact",
                "reset_step_key_schedule_exact",
                "structural_replay_exact",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
        {
            "profile_id": "candidate_seed_transport_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["candidate_observation"],
            "acceptance_fields": [
                "agent_seed_transport_exact",
                "environment_agent_derivations_distinct",
                "candidate_rng_membership_exact",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
        {
            "profile_id": "full_horizon_resource_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["resource_observation"],
            "acceptance_fields": [
                "horizon_accounting_exact",
                "reward_membership_structural_only",
                "all_resource_observations_within_predeclared_integer_ceilings",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
        {
            "profile_id": "result_publication_roundtrip_v1",
            "required_observation_schema": _RECEIPT_SCHEMAS["publication_observation"],
            "acceptance_fields": [
                "publisher_descriptor_membership_exact",
                "publisher_source_closure_membership_exact",
                "reload_validator_membership_exact",
                "atomic_publication_exact",
                "strict_reload_exact",
                "full_file_digest_equivalence_exact",
                "score_and_reward_magnitude_not_decoded",
            ],
            "reward_magnitude_is_acceptance_input": False,
            "score_is_acceptance_input": False,
        },
    ]


def _failure_policy() -> dict[str, Any]:
    return {
        "fixed_before_observation_required": True,
        "fixed_before_observation_verified_here": False,
        "fail_closed": True,
        "missing_required_observation": "reject_case",
        "schema_or_digest_mismatch": "reject_bundle",
        "resource_ceiling_exceeded": "reject_case_no_post_observation_retuning",
        "retry_ceiling_exceeded": "reject_case",
        "partial_candidate_coverage": "reject_plan",
        "reward_magnitude_is_failure_input": False,
        "score_is_failure_input": False,
        "ranking_is_failure_input": False,
    }


def _acceptance_contract() -> dict[str, Any]:
    return {
        "required_observation_schemas": dict(_RECEIPT_SCHEMAS),
        "configuration_membership_exact": True,
        "source_membership_exact": True,
        "runtime_membership_exact": True,
        "seed_transport_replay_exact": True,
        "horizon_accounting_exact": True,
        "resource_accounting_complete": True,
        "qualification_seed_provenance_authentication_required": True,
        "qualification_seed_provenance_authenticated_here": False,
        "qualification_seed_trust_root_receipt_external_pin_exact": True,
        "qualification_seed_offline_signature_verification_exact": True,
        "qualification_seed_registry_issued_before_observation_required": True,
        "result_publication_roundtrip_exact": True,
        "publication_full_file_digest_equivalence_exact": True,
        "reward_membership_may_be_checked_structurally": True,
        "reward_magnitude_is_acceptance_input": False,
        "score_is_acceptance_input": False,
        "ranking_is_acceptance_input": False,
    }


def _source_for_candidate(candidate_id: str) -> str:
    return "external_foragax_agents" if candidate_id in _EXTERNAL_CANDIDATE_IDS else "local_alberta"


def _require_exact_order(items: Sequence[Any], expected_ids: tuple[str, ...], label: str) -> None:
    if type(items) not in {tuple, list}:
        raise ForagerMatchedV3QualificationPlanError(f"{label} must be an exact tuple or list")
    if len({id(item) for item in items}) != len(items):
        raise ForagerMatchedV3QualificationPlanError(f"{label} contains aliased entries")
    ids = tuple(getattr(item, "candidate_id", None) for item in items)
    if ids != expected_ids:
        raise ForagerMatchedV3QualificationPlanError(
            f"{label} must cover the exact ordered 28-candidate universe"
        )


def _validate_source_requirements(items: Sequence[SourceRequirement]) -> None:
    if type(items) not in {tuple, list} or len(items) != len(_REQUIRED_SOURCE_IDS):
        raise ForagerMatchedV3QualificationPlanError(
            "source requirements must be the exact two-entry sequence"
        )
    if len({id(item) for item in items}) != len(items):
        raise ForagerMatchedV3QualificationPlanError("source requirements are aliased")
    if any(type(item) is not SourceRequirement for item in items):
        raise ForagerMatchedV3QualificationPlanError("source requirement type is invalid")
    if tuple(item.source_id for item in items) != _REQUIRED_SOURCE_IDS:
        raise ForagerMatchedV3QualificationPlanError(
            "source requirements are missing, duplicated, unknown, or reordered"
        )
    if len({item.source_tree_sha256 for item in items}) != len(items):
        raise ForagerMatchedV3QualificationPlanError(
            "external candidate and local harness source trees must be distinct"
        )


def _expected_qualification_case_derivation(
    receipt: QualificationSeedTrustRootReceiptBinding,
    candidate_id: str,
    ordinal: int,
) -> tuple[str, int, int, str, str]:
    payload = {
        "schema_version": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
        "domain": QUALIFICATION_SEED_DERIVATION_DOMAIN,
        "algorithm": _SEED_PAIR_DERIVATION_ALGORITHM,
        "provider_chain_hash": receipt.provider_chain_hash,
        "beacon_round": receipt.beacon_round,
        "beacon_randomness_hex": receipt.beacon_randomness_hex,
        "candidate_id": candidate_id,
        "registry_case_ordinal": ordinal,
    }
    payload_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    derivation_digests: list[str] = []
    seeds: list[int] = []
    for lane in ("environment", "agent"):
        lane_payload = {**payload, "lane": lane}
        digest = hashlib.sha256(_canonical_json(lane_payload)).digest()
        derivation_digests.append(digest.hex())
        seeds.append(int.from_bytes(digest[:4], "big") & _UINT31_MAX)
    return (
        payload_sha256,
        seeds[0],
        seeds[1],
        derivation_digests[0],
        derivation_digests[1],
    )


def _expected_qualification_case_id(candidate_id: str, ordinal: int) -> str:
    return f"qualification_{ordinal:02d}_{candidate_id}"


def _expected_qualification_seed_registry_digests(
    receipt: QualificationSeedTrustRootReceiptBinding,
) -> tuple[str, str]:
    cases: list[dict[str, Any]] = []
    for ordinal, candidate_id in enumerate(MATCHED_V3_QUALIFICATION_CANDIDATE_IDS):
        (
            derivation_payload_sha256,
            environment_seed,
            agent_seed,
            environment_seed_derivation_sha256,
            agent_seed_derivation_sha256,
        ) = _expected_qualification_case_derivation(receipt, candidate_id, ordinal)
        cases.append(
            {
                "case_id": _expected_qualification_case_id(candidate_id, ordinal),
                "candidate_id": candidate_id,
                "material_class": "public_nonbenchmark_permanently_consumed",
                "registry_case_ordinal": ordinal,
                "derivation_payload_sha256": derivation_payload_sha256,
                "environment_seed": environment_seed,
                "agent_seed": agent_seed,
                "environment_seed_derivation_sha256": (environment_seed_derivation_sha256),
                "agent_seed_derivation_sha256": agent_seed_derivation_sha256,
            }
        )
    body = {
        "schema_version": QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION,
        "material_class": "public_nonbenchmark_permanently_consumed",
        "provider": {
            "provider_id": receipt.provider_id,
            "provider_chain_hash": receipt.provider_chain_hash,
            "signature_scheme": receipt.signature_scheme,
            "provider_public_key_sha256": receipt.provider_public_key_sha256,
            "beacon_round": receipt.beacon_round,
            "beacon_time_unix": receipt.beacon_time_unix,
            "observation_cutoff_unix": receipt.observation_cutoff_unix,
            "beacon_signature_sha256": receipt.beacon_signature_sha256,
            "beacon_randomness_hex": receipt.beacon_randomness_hex,
            "pulse_record_schema_version": receipt.pulse_record_schema_version,
            "pulse_record_file_sha256": receipt.pulse_record_file_sha256,
            "pulse_record_body_sha256": receipt.pulse_record_body_sha256,
        },
        "derivation": {
            "schema_version": QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION,
            "domain": QUALIFICATION_SEED_DERIVATION_DOMAIN,
            "algorithm": _SEED_PAIR_DERIVATION_ALGORITHM,
        },
        "candidate_order": list(MATCHED_V3_QUALIFICATION_CANDIDATE_IDS),
        "cases": cases,
    }
    body_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    file_sha256 = hashlib.sha256(
        _canonical_json({**body, "registry_body_sha256": body_sha256})
    ).hexdigest()
    return file_sha256, body_sha256


def _validate_qualification_cases(
    items: Sequence[QualificationCase],
    seed_registry: QualificationSeedRegistryBinding,
    receipt: QualificationSeedTrustRootReceiptBinding,
) -> None:
    _require_exact_order(items, MATCHED_V3_QUALIFICATION_CANDIDATE_IDS, "qualification cases")
    if any(type(item) is not QualificationCase for item in items):
        raise ForagerMatchedV3QualificationPlanError("qualification case type is invalid")
    expected_registry_fields = (
        seed_registry.registry_file_sha256,
        hashlib.sha256(_canonical_json(seed_registry.to_dict())).hexdigest(),
        seed_registry.registry_body_sha256,
        seed_registry.provider_identity_sha256,
        seed_registry.provider_receipt_file_sha256,
        seed_registry.provider_receipt_body_sha256,
        seed_registry.derivation_schema_version,
        seed_registry.derivation_domain,
    )
    for ordinal, item in enumerate(items):
        if (
            item.case_id != _expected_qualification_case_id(item.candidate_id, ordinal)
            or item.registry_case_ordinal != ordinal
            or (
                item.seed_registry_file_sha256,
                item.seed_registry_binding_sha256,
                item.seed_registry_body_sha256,
                item.provider_identity_sha256,
                item.provider_receipt_file_sha256,
                item.provider_receipt_body_sha256,
                item.derivation_schema_version,
                item.derivation_domain,
            )
            != expected_registry_fields
        ):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case is cross-wired from its authenticated seed registry"
            )
        if (
            item.derivation_payload_sha256,
            item.environment_seed,
            item.agent_seed,
            item.environment_seed_derivation_sha256,
            item.agent_seed_derivation_sha256,
        ) != _expected_qualification_case_derivation(receipt, item.candidate_id, ordinal):
            raise ForagerMatchedV3QualificationPlanError(
                "qualification case seed pair is not the exact deterministic pulse derivation"
            )
    case_ids = tuple(item.case_id for item in items)
    if len(case_ids) != len(set(case_ids)):
        raise ForagerMatchedV3QualificationPlanError("qualification case IDs are duplicated")
    derivations = tuple(
        digest
        for item in items
        for digest in (
            item.derivation_payload_sha256,
            item.environment_seed_derivation_sha256,
            item.agent_seed_derivation_sha256,
        )
    )
    if len(derivations) != len(set(derivations)):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed derivation identities are not globally distinct"
        )


def _validate_seed_authentication_binding(
    receipt: QualificationSeedTrustRootReceiptBinding,
    seed_registry: QualificationSeedRegistryBinding,
    source_requirements: Sequence[SourceRequirement],
    runtime_requirement: RuntimeRequirement,
    expected_receipt_file_sha256: str,
    expected_receipt_binding_sha256: str,
) -> None:
    if type(receipt) is not QualificationSeedTrustRootReceiptBinding:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed trust-root receipt binding type is invalid"
        )
    expected_receipt_file_sha256 = _require_sha256(
        expected_receipt_file_sha256,
        "independently expected qualification seed trust-root receipt file",
    )
    if not hmac.compare_digest(receipt.receipt_file_sha256, expected_receipt_file_sha256):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed trust-root receipt differs from its independent pin"
        )
    receipt_binding_sha256 = hashlib.sha256(_canonical_json(receipt.to_dict())).hexdigest()
    expected_receipt_binding_sha256 = _require_sha256(
        expected_receipt_binding_sha256,
        "independently expected qualification seed trust-root receipt binding",
    )
    if not hmac.compare_digest(receipt_binding_sha256, expected_receipt_binding_sha256):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed trust-root receipt binding differs from its independent pin"
        )
    if (
        seed_registry.provider_id != receipt.provider_id
        or seed_registry.provider_identity_sha256 != receipt.provider_chain_hash
        or seed_registry.provider_receipt_schema_version != receipt.receipt_schema_version
        or seed_registry.provider_receipt_file_sha256 != receipt.receipt_file_sha256
        or seed_registry.provider_receipt_body_sha256 != receipt.receipt_body_sha256
        or seed_registry.trust_root_receipt_binding_sha256 != receipt_binding_sha256
        or seed_registry.registry_schema_version != receipt.seed_registry_schema_version
        or seed_registry.registry_file_sha256 != receipt.seed_registry_file_sha256
        or seed_registry.registry_body_sha256 != receipt.seed_registry_body_sha256
    ):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed registry is not bound to its preaccepted trust-root receipt"
        )
    expected_registry_file_sha256, expected_registry_body_sha256 = (
        _expected_qualification_seed_registry_digests(receipt)
    )
    if (
        seed_registry.registry_file_sha256 != expected_registry_file_sha256
        or seed_registry.registry_body_sha256 != expected_registry_body_sha256
    ):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed registry is not the exact deterministic pulse-derived registry"
        )
    local_source_tree_sha256 = next(
        item.source_tree_sha256 for item in source_requirements if item.source_id == "local_alberta"
    )
    if receipt.offline_verifier_source_tree_sha256 != local_source_tree_sha256:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed offline verifier is not attached to the local source closure"
        )
    runtime_helper = next(
        item
        for item in runtime_requirement.helper_bindings
        if item.helper_id == receipt.offline_verifier_runtime_helper_id
    )
    if (
        receipt.offline_verifier_executable_sha256 != runtime_helper.executable_sha256
        or receipt.offline_verifier_version_output_sha256 != runtime_helper.version_output_sha256
    ):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed offline verifier differs from its exact runtime helper"
        )


def _validate_resources(items: Sequence[CandidateResourceRequirement]) -> None:
    _require_exact_order(items, MATCHED_V3_QUALIFICATION_CANDIDATE_IDS, "resource requirements")
    if any(type(item) is not CandidateResourceRequirement for item in items):
        raise ForagerMatchedV3QualificationPlanError("resource requirement type is invalid")


def _validate_publication_bindings(
    items: Sequence[CandidatePublicationBinding],
    source_requirements: Sequence[SourceRequirement],
) -> None:
    _require_exact_order(
        items,
        MATCHED_V3_QUALIFICATION_CANDIDATE_IDS,
        "result publication bindings",
    )
    if any(type(item) is not CandidatePublicationBinding for item in items):
        raise ForagerMatchedV3QualificationPlanError("publication binding type is invalid")
    source_tree_by_id = {item.source_id: item.source_tree_sha256 for item in source_requirements}
    for item in items:
        if item.source_tree_sha256 != source_tree_by_id[item.source_id]:
            raise ForagerMatchedV3QualificationPlanError(
                "publication binding is not attached to its exact source closure"
            )


def _bindings() -> dict[str, Any]:
    return {
        "qualification_plan_descriptor": {
            "schema_version": QUALIFICATION_PLAN_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": QUALIFICATION_PLAN_DESCRIPTOR_SHA256,
        },
        "dependencies": _dependencies(),
    }


def build_matched_v3_qualification_plan(
    *,
    source_requirements: Sequence[SourceRequirement],
    runtime_requirement: RuntimeRequirement,
    qualification_seed_registry: QualificationSeedRegistryBinding,
    qualification_seed_trust_root_receipt: QualificationSeedTrustRootReceiptBinding,
    expected_qualification_seed_trust_root_receipt_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_binding_sha256: str,
    qualification_cases: Sequence[QualificationCase],
    resource_requirements: Sequence[CandidateResourceRequirement],
    result_publication_bindings: Sequence[CandidatePublicationBinding],
) -> dict[str, Any]:
    """Build one detached, unexecuted plan from complete caller-supplied identities."""

    _validate_source_requirements(source_requirements)
    if type(runtime_requirement) is not RuntimeRequirement:
        raise ForagerMatchedV3QualificationPlanError("runtime requirement type is invalid")
    if type(qualification_seed_registry) is not QualificationSeedRegistryBinding:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification seed registry binding type is invalid"
        )
    _validate_seed_authentication_binding(
        qualification_seed_trust_root_receipt,
        qualification_seed_registry,
        source_requirements,
        runtime_requirement,
        expected_qualification_seed_trust_root_receipt_file_sha256,
        expected_qualification_seed_trust_root_receipt_binding_sha256,
    )
    _validate_qualification_cases(
        qualification_cases,
        qualification_seed_registry,
        qualification_seed_trust_root_receipt,
    )
    _validate_resources(resource_requirements)
    _validate_publication_bindings(result_publication_bindings, source_requirements)

    resources_by_id = {item.candidate_id: item for item in resource_requirements}
    publications_by_id = {item.candidate_id: item for item in result_publication_bindings}
    candidate_requirements = [
        {
            "candidate_id": candidate_id,
            "configuration_record_sha256": _CONFIGURATION_RECORD_SHA256[candidate_id],
            "source_id": _source_for_candidate(candidate_id),
            "probe_profile_ids": list(_PROBE_PROFILE_IDS),
            "result_publication_binding": publications_by_id[candidate_id].to_dict(),
            "resource_scope": {
                "candidate_id": candidate_id,
                "integer_ceiling_fields": list(_RESOURCE_FIELDS),
                "requirement_body_sha256": hashlib.sha256(
                    _canonical_json(resources_by_id[candidate_id].to_dict())
                ).hexdigest(),
            },
            "acceptance": _acceptance_contract(),
        }
        for candidate_id in MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    ]
    body: dict[str, Any] = {
        "schema_version": QUALIFICATION_PLAN_SCHEMA_VERSION,
        "status": QUALIFICATION_PLAN_STATUS,
        "classification": QUALIFICATION_PLAN_CLASSIFICATION,
        "bindings": _bindings(),
        "source_requirements": [item.to_dict() for item in source_requirements],
        "runtime_requirement": runtime_requirement.to_dict(),
        "qualification_seed_trust_root_receipt": (qualification_seed_trust_root_receipt.to_dict()),
        "qualification_seed_registry": qualification_seed_registry.to_dict(),
        "probe_profiles": _probe_profiles(),
        "candidate_requirements": candidate_requirements,
        "resource_contract": {
            "scope": "per_candidate_public_qualification_case_integer_ceilings_v1",
            "integer_ceiling_fields": list(_RESOURCE_FIELDS),
            "requirements": [item.to_dict() for item in resource_requirements],
            "compute_efficiency_claimed": False,
            "resource_matched_claimed": False,
        },
        "seed_boundary": {
            "material_class": "public_nonbenchmark_permanently_consumed",
            "registry_binding_sha256": hashlib.sha256(
                _canonical_json(qualification_seed_registry.to_dict())
            ).hexdigest(),
            "trust_root_receipt_binding_sha256": hashlib.sha256(
                _canonical_json(qualification_seed_trust_root_receipt.to_dict())
            ).hexdigest(),
            "trust_root_receipt_external_pin_required": True,
            "offline_signature_verification_required": True,
            "offline_signature_verification_implemented_here": False,
            "preacceptance_chronology_implemented_here": False,
            "cases": [item.to_dict() for item in qualification_cases],
            "external_authentication_required": True,
            "issued_before_observation_required": True,
            "trial_block_material_allowed": False,
            "held_out_material_allowed": False,
            "future_randomness_allowed": False,
            "scientific_seed_reuse_allowed": False,
        },
        "failure_policy": _failure_policy(),
        "authentication_policy": _authentication_policy(),
        "claims": _claims(),
        "limitations": _limitations(),
    }
    plan = {**body, "plan_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest()}
    _validate_plan(
        plan,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            expected_qualification_seed_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            expected_qualification_seed_trust_root_receipt_binding_sha256
        ),
    )
    return _strict_json_load(_canonical_json(plan))


def _validate_source_dict(value: Mapping[str, Any], expected_source_id: str) -> None:
    _require_exact_keys(
        value,
        frozenset(
            {
                "source_id",
                "closure_kind",
                "manifest_schema_version",
                "manifest_file_sha256",
                "manifest_body_sha256",
                "source_tree_sha256",
                "inventory_sha256",
                "file_count",
                "directory_count",
                "total_bytes",
            }
        ),
        "source requirement",
    )
    if value["source_id"] != expected_source_id:
        raise ForagerMatchedV3QualificationPlanError("source requirement order drifted")
    expected_kind = {
        "external_foragax_agents": "derived_checkout_manifest_tree",
        "local_alberta": "normalized_local_source_snapshot",
    }[expected_source_id]
    if value["closure_kind"] != expected_kind:
        raise ForagerMatchedV3QualificationPlanError("source closure kind drifted")
    if value["manifest_schema_version"] != _SOURCE_MANIFEST_SCHEMAS[expected_source_id]:
        raise ForagerMatchedV3QualificationPlanError("source manifest schema drifted")
    for key in (
        "manifest_file_sha256",
        "manifest_body_sha256",
        "source_tree_sha256",
        "inventory_sha256",
    ):
        _require_sha256(value[key], f"source {key}")
    files = _require_bounded_int(
        value["file_count"], "source file count", minimum=1, maximum=_MAX_SOURCE_ENTRIES
    )
    directories = _require_bounded_int(
        value["directory_count"],
        "source directory count",
        maximum=_MAX_SOURCE_ENTRIES,
    )
    if files + directories > _MAX_SOURCE_ENTRIES:
        raise ForagerMatchedV3QualificationPlanError("source entry count exceeds its bound")
    _require_bounded_int(value["total_bytes"], "source total bytes", minimum=1)


def _validate_seed_trust_root_receipt_dict(
    value: Mapping[str, Any],
) -> QualificationSeedTrustRootReceiptBinding:
    _require_exact_keys(
        value,
        frozenset(
            {
                "receipt_schema_version",
                "receipt_file_sha256",
                "receipt_body_sha256",
                "provider_id",
                "provider_chain_hash",
                "signature_scheme",
                "provider_public_key_sha256",
                "beacon_round",
                "beacon_time_unix",
                "observation_cutoff_unix",
                "beacon_signature_sha256",
                "beacon_randomness_hex",
                "pulse_record_schema_version",
                "pulse_record_file_sha256",
                "pulse_record_body_sha256",
                "seed_registry_schema_version",
                "seed_registry_file_sha256",
                "seed_registry_body_sha256",
                "seed_derivation_algorithm",
                "timestamp_source",
                "offline_verifier_schema_version",
                "offline_verifier_descriptor_sha256",
                "offline_verifier_source_id",
                "offline_verifier_source_tree_sha256",
                "offline_verifier_implementation_path",
                "offline_verifier_source_sha256",
                "offline_verifier_runtime_helper_id",
                "offline_verifier_executable_sha256",
                "offline_verifier_version_output_sha256",
                "offline_verifier_implementation_status",
                "offline_signature_verification_required",
                "external_preacceptance_required",
            }
        ),
        "qualification seed trust-root receipt binding",
    )
    return QualificationSeedTrustRootReceiptBinding(
        receipt_schema_version=value["receipt_schema_version"],
        receipt_file_sha256=value["receipt_file_sha256"],
        receipt_body_sha256=value["receipt_body_sha256"],
        provider_id=value["provider_id"],
        provider_chain_hash=value["provider_chain_hash"],
        signature_scheme=value["signature_scheme"],
        provider_public_key_sha256=value["provider_public_key_sha256"],
        beacon_round=value["beacon_round"],
        beacon_time_unix=value["beacon_time_unix"],
        observation_cutoff_unix=value["observation_cutoff_unix"],
        beacon_signature_sha256=value["beacon_signature_sha256"],
        beacon_randomness_hex=value["beacon_randomness_hex"],
        pulse_record_schema_version=value["pulse_record_schema_version"],
        pulse_record_file_sha256=value["pulse_record_file_sha256"],
        pulse_record_body_sha256=value["pulse_record_body_sha256"],
        seed_registry_schema_version=value["seed_registry_schema_version"],
        seed_registry_file_sha256=value["seed_registry_file_sha256"],
        seed_registry_body_sha256=value["seed_registry_body_sha256"],
        seed_derivation_algorithm=value["seed_derivation_algorithm"],
        timestamp_source=value["timestamp_source"],
        offline_verifier_schema_version=value["offline_verifier_schema_version"],
        offline_verifier_descriptor_sha256=value["offline_verifier_descriptor_sha256"],
        offline_verifier_source_id=value["offline_verifier_source_id"],
        offline_verifier_source_tree_sha256=value["offline_verifier_source_tree_sha256"],
        offline_verifier_implementation_path=value["offline_verifier_implementation_path"],
        offline_verifier_source_sha256=value["offline_verifier_source_sha256"],
        offline_verifier_runtime_helper_id=value["offline_verifier_runtime_helper_id"],
        offline_verifier_executable_sha256=value["offline_verifier_executable_sha256"],
        offline_verifier_version_output_sha256=value["offline_verifier_version_output_sha256"],
        offline_verifier_implementation_status=value["offline_verifier_implementation_status"],
        offline_signature_verification_required=value["offline_signature_verification_required"],
        external_preacceptance_required=value["external_preacceptance_required"],
    )


def _validate_seed_registry_dict(
    value: Mapping[str, Any],
) -> QualificationSeedRegistryBinding:
    _require_exact_keys(
        value,
        frozenset(
            {
                "registry_schema_version",
                "registry_file_sha256",
                "registry_body_sha256",
                "derivation_schema_version",
                "derivation_domain",
                "provider_id",
                "provider_identity_sha256",
                "provider_receipt_schema_version",
                "provider_receipt_file_sha256",
                "provider_receipt_body_sha256",
                "trust_root_receipt_binding_sha256",
                "external_authentication_required",
                "issued_before_observation_required",
            }
        ),
        "qualification seed registry binding",
    )
    return QualificationSeedRegistryBinding(
        registry_schema_version=value["registry_schema_version"],
        registry_file_sha256=value["registry_file_sha256"],
        registry_body_sha256=value["registry_body_sha256"],
        derivation_schema_version=value["derivation_schema_version"],
        derivation_domain=value["derivation_domain"],
        provider_id=value["provider_id"],
        provider_identity_sha256=value["provider_identity_sha256"],
        provider_receipt_schema_version=value["provider_receipt_schema_version"],
        provider_receipt_file_sha256=value["provider_receipt_file_sha256"],
        provider_receipt_body_sha256=value["provider_receipt_body_sha256"],
        trust_root_receipt_binding_sha256=value["trust_root_receipt_binding_sha256"],
        external_authentication_required=value["external_authentication_required"],
        issued_before_observation_required=value["issued_before_observation_required"],
    )


def _validate_runtime_dict(value: Mapping[str, Any]) -> RuntimeRequirement:
    expected_keys = frozenset(
        {
            "executor_kind",
            "runtime_executable_sha256",
            "runtime_version_output_sha256",
            "image_digest",
            "image_config_sha256",
            "runtime_profile_sha256",
            "python_implementation",
            "python_version",
            "jax_version",
            "jaxlib_version",
            "foragax_version",
            "foragax_install_tree_sha256",
            "platform",
            "default_prng_impl",
            "jax_enable_x64",
            "threefry_partitionable",
            "sandbox_descriptor_sha256",
            "helper_bindings",
        }
    )
    _require_exact_keys(value, expected_keys, "runtime requirement")
    helpers = value["helper_bindings"]
    if type(helpers) is not list or not helpers:
        raise ForagerMatchedV3QualificationPlanError("runtime helper bindings are incomplete")
    helper_objects: list[RuntimeHelperBinding] = []
    for helper in helpers:
        _require_exact_keys(
            cast(dict[str, Any], helper),
            frozenset({"helper_id", "executable_sha256", "version_output_sha256"}),
            "runtime helper binding",
        )
        helper_objects.append(
            RuntimeHelperBinding(
                helper_id=helper["helper_id"],
                executable_sha256=helper["executable_sha256"],
                version_output_sha256=helper["version_output_sha256"],
            )
        )
    return RuntimeRequirement(
        executor_kind=value["executor_kind"],
        runtime_executable_sha256=value["runtime_executable_sha256"],
        runtime_version_output_sha256=value["runtime_version_output_sha256"],
        image_digest=value["image_digest"],
        image_config_sha256=value["image_config_sha256"],
        runtime_profile_sha256=value["runtime_profile_sha256"],
        python_implementation=value["python_implementation"],
        python_version=value["python_version"],
        jax_version=value["jax_version"],
        jaxlib_version=value["jaxlib_version"],
        foragax_version=value["foragax_version"],
        foragax_install_tree_sha256=value["foragax_install_tree_sha256"],
        platform=value["platform"],
        default_prng_impl=value["default_prng_impl"],
        jax_enable_x64=value["jax_enable_x64"],
        threefry_partitionable=value["threefry_partitionable"],
        sandbox_descriptor_sha256=value["sandbox_descriptor_sha256"],
        helper_bindings=tuple(helper_objects),
    )


def _validate_resource_dict(value: Mapping[str, Any], expected_candidate_id: str) -> None:
    _require_exact_keys(
        value,
        frozenset({"candidate_id", *_RESOURCE_FIELDS}),
        "candidate resource requirement",
    )
    if value["candidate_id"] != expected_candidate_id:
        raise ForagerMatchedV3QualificationPlanError("resource candidate order drifted")
    for field_name in _RESOURCE_FIELDS:
        _require_bounded_int(value[field_name], f"resource field {field_name}")
    if value["max_environment_interactions"] < _HORIZON:
        raise ForagerMatchedV3QualificationPlanError("resource horizon coverage is incomplete")
    if value["max_thread_count"] < 1 or value["max_attempt_count"] < 1:
        raise ForagerMatchedV3QualificationPlanError("resource positive ceiling is invalid")
    if value["max_failure_count"] >= value["max_attempt_count"]:
        raise ForagerMatchedV3QualificationPlanError("resource failure ceiling is invalid")


def _validate_publication_dict(
    value: Mapping[str, Any],
    expected_candidate_id: str,
    source_tree_by_id: Mapping[str, str],
) -> None:
    _require_exact_keys(
        value,
        frozenset(
            {
                "candidate_id",
                "publisher_kind",
                "descriptor_schema_version",
                "descriptor_sha256",
                "publication_schema_version",
                "source_id",
                "source_tree_sha256",
                "implementation_path",
                "implementation_source_sha256",
                "reload_validator_schema_version",
                "reload_validator_descriptor_sha256",
                "reload_validator_implementation_path",
                "reload_validator_source_sha256",
            }
        ),
        "candidate publication binding",
    )
    binding = CandidatePublicationBinding(
        candidate_id=value["candidate_id"],
        publisher_kind=value["publisher_kind"],
        descriptor_schema_version=value["descriptor_schema_version"],
        descriptor_sha256=value["descriptor_sha256"],
        publication_schema_version=value["publication_schema_version"],
        source_id=value["source_id"],
        source_tree_sha256=value["source_tree_sha256"],
        implementation_path=value["implementation_path"],
        implementation_source_sha256=value["implementation_source_sha256"],
        reload_validator_schema_version=value["reload_validator_schema_version"],
        reload_validator_descriptor_sha256=value["reload_validator_descriptor_sha256"],
        reload_validator_implementation_path=value["reload_validator_implementation_path"],
        reload_validator_source_sha256=value["reload_validator_source_sha256"],
    )
    if binding.candidate_id != expected_candidate_id:
        raise ForagerMatchedV3QualificationPlanError("publication candidate order drifted")
    if binding.source_tree_sha256 != source_tree_by_id[binding.source_id]:
        raise ForagerMatchedV3QualificationPlanError(
            "publication source-tree membership binding drifted"
        )


def _validate_case_dict(
    value: Mapping[str, Any],
    expected_candidate_id: str,
    expected_ordinal: int,
    seed_registry: QualificationSeedRegistryBinding,
    receipt: QualificationSeedTrustRootReceiptBinding,
) -> QualificationCase:
    _require_exact_keys(
        value,
        frozenset(
            {
                "case_id",
                "candidate_id",
                "material_class",
                "registry_case_ordinal",
                "seed_registry_binding_sha256",
                "seed_registry_file_sha256",
                "seed_registry_body_sha256",
                "provider_identity_sha256",
                "provider_receipt_file_sha256",
                "provider_receipt_body_sha256",
                "derivation_schema_version",
                "derivation_domain",
                "derivation_payload_sha256",
                "environment_seed",
                "agent_seed",
                "environment_seed_derivation_sha256",
                "agent_seed_derivation_sha256",
            }
        ),
        "qualification case",
    )
    case = QualificationCase(
        case_id=value["case_id"],
        candidate_id=value["candidate_id"],
        material_class=value["material_class"],
        registry_case_ordinal=value["registry_case_ordinal"],
        seed_registry_binding_sha256=value["seed_registry_binding_sha256"],
        seed_registry_file_sha256=value["seed_registry_file_sha256"],
        seed_registry_body_sha256=value["seed_registry_body_sha256"],
        provider_identity_sha256=value["provider_identity_sha256"],
        provider_receipt_file_sha256=value["provider_receipt_file_sha256"],
        provider_receipt_body_sha256=value["provider_receipt_body_sha256"],
        derivation_schema_version=value["derivation_schema_version"],
        derivation_domain=value["derivation_domain"],
        derivation_payload_sha256=value["derivation_payload_sha256"],
        environment_seed=value["environment_seed"],
        agent_seed=value["agent_seed"],
        environment_seed_derivation_sha256=value["environment_seed_derivation_sha256"],
        agent_seed_derivation_sha256=value["agent_seed_derivation_sha256"],
    )
    if (
        case.candidate_id != expected_candidate_id
        or case.case_id != _expected_qualification_case_id(expected_candidate_id, expected_ordinal)
        or case.registry_case_ordinal != expected_ordinal
    ):
        raise ForagerMatchedV3QualificationPlanError("qualification case order drifted")
    if (
        case.seed_registry_binding_sha256
        != hashlib.sha256(_canonical_json(seed_registry.to_dict())).hexdigest()
        or case.seed_registry_file_sha256 != seed_registry.registry_file_sha256
        or case.seed_registry_body_sha256 != seed_registry.registry_body_sha256
        or case.provider_identity_sha256 != seed_registry.provider_identity_sha256
        or case.provider_receipt_file_sha256 != seed_registry.provider_receipt_file_sha256
        or case.provider_receipt_body_sha256 != seed_registry.provider_receipt_body_sha256
        or case.derivation_schema_version != seed_registry.derivation_schema_version
        or case.derivation_domain != seed_registry.derivation_domain
    ):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification case registry or provider receipt is cross-wired"
        )
    if (
        case.derivation_payload_sha256,
        case.environment_seed,
        case.agent_seed,
        case.environment_seed_derivation_sha256,
        case.agent_seed_derivation_sha256,
    ) != _expected_qualification_case_derivation(receipt, expected_candidate_id, expected_ordinal):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification case seed pair is not the exact deterministic pulse derivation"
        )
    return case


def _validate_plan(
    value: Mapping[str, Any],
    *,
    expected_qualification_seed_trust_root_receipt_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_binding_sha256: str,
) -> None:
    _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "bindings",
                "source_requirements",
                "runtime_requirement",
                "qualification_seed_trust_root_receipt",
                "qualification_seed_registry",
                "probe_profiles",
                "candidate_requirements",
                "resource_contract",
                "seed_boundary",
                "failure_policy",
                "authentication_policy",
                "claims",
                "limitations",
                "plan_body_sha256",
            }
        ),
        "qualification plan",
    )
    if (
        value["schema_version"] != QUALIFICATION_PLAN_SCHEMA_VERSION
        or value["status"] != QUALIFICATION_PLAN_STATUS
        or value["classification"] != QUALIFICATION_PLAN_CLASSIFICATION
    ):
        raise ForagerMatchedV3QualificationPlanError("qualification plan identity drifted")
    if not _exact_json_equal(value["bindings"], _bindings()):
        raise ForagerMatchedV3QualificationPlanError("qualification dependency binding drifted")
    sources = value["source_requirements"]
    if type(sources) is not list or len(sources) != 2:
        raise ForagerMatchedV3QualificationPlanError("qualification source coverage is incomplete")
    for source, source_id in zip(sources, _REQUIRED_SOURCE_IDS, strict=True):
        _validate_source_dict(cast(dict[str, Any], source), source_id)
    if len({source["source_tree_sha256"] for source in sources}) != len(sources):
        raise ForagerMatchedV3QualificationPlanError(
            "external candidate and local harness source trees must be distinct"
        )
    source_tree_by_id = {
        cast(str, source["source_id"]): cast(str, source["source_tree_sha256"])
        for source in cast(list[dict[str, Any]], sources)
    }
    runtime_requirement = _validate_runtime_dict(cast(dict[str, Any], value["runtime_requirement"]))
    trust_root_receipt = _validate_seed_trust_root_receipt_dict(
        cast(dict[str, Any], value["qualification_seed_trust_root_receipt"])
    )
    seed_registry = _validate_seed_registry_dict(
        cast(dict[str, Any], value["qualification_seed_registry"])
    )
    source_objects = tuple(
        SourceRequirement(
            source_id=source["source_id"],
            closure_kind=source["closure_kind"],
            manifest_schema_version=source["manifest_schema_version"],
            manifest_file_sha256=source["manifest_file_sha256"],
            manifest_body_sha256=source["manifest_body_sha256"],
            source_tree_sha256=source["source_tree_sha256"],
            inventory_sha256=source["inventory_sha256"],
            file_count=source["file_count"],
            directory_count=source["directory_count"],
            total_bytes=source["total_bytes"],
        )
        for source in cast(list[dict[str, Any]], sources)
    )
    _validate_seed_authentication_binding(
        trust_root_receipt,
        seed_registry,
        source_objects,
        runtime_requirement,
        expected_qualification_seed_trust_root_receipt_file_sha256,
        expected_qualification_seed_trust_root_receipt_binding_sha256,
    )
    if not _exact_json_equal(value["probe_profiles"], _probe_profiles()):
        raise ForagerMatchedV3QualificationPlanError("qualification probe profiles drifted")

    resources = cast(dict[str, Any], value["resource_contract"])
    _require_exact_keys(
        resources,
        frozenset(
            {
                "scope",
                "integer_ceiling_fields",
                "requirements",
                "compute_efficiency_claimed",
                "resource_matched_claimed",
            }
        ),
        "qualification resource contract",
    )
    if (
        resources["scope"] != "per_candidate_public_qualification_case_integer_ceilings_v1"
        or resources["integer_ceiling_fields"] != list(_RESOURCE_FIELDS)
        or resources["compute_efficiency_claimed"] is not False
        or resources["resource_matched_claimed"] is not False
    ):
        raise ForagerMatchedV3QualificationPlanError("qualification resource scope drifted")
    resource_items = resources["requirements"]
    if type(resource_items) is not list or len(resource_items) != len(
        MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    ):
        raise ForagerMatchedV3QualificationPlanError("resource candidate coverage is incomplete")
    for resource, candidate_id in zip(
        resource_items, MATCHED_V3_QUALIFICATION_CANDIDATE_IDS, strict=True
    ):
        _validate_resource_dict(cast(dict[str, Any], resource), candidate_id)

    candidate_items = value["candidate_requirements"]
    if type(candidate_items) is not list or len(candidate_items) != len(
        MATCHED_V3_QUALIFICATION_CANDIDATE_IDS
    ):
        raise ForagerMatchedV3QualificationPlanError("candidate coverage is incomplete")
    for candidate, resource, candidate_id in zip(
        candidate_items,
        resource_items,
        MATCHED_V3_QUALIFICATION_CANDIDATE_IDS,
        strict=True,
    ):
        candidate_map = cast(dict[str, Any], candidate)
        _require_exact_keys(
            candidate_map,
            frozenset(
                {
                    "candidate_id",
                    "configuration_record_sha256",
                    "source_id",
                    "probe_profile_ids",
                    "result_publication_binding",
                    "resource_scope",
                    "acceptance",
                }
            ),
            "candidate qualification requirement",
        )
        if (
            candidate_map["candidate_id"] != candidate_id
            or candidate_map["configuration_record_sha256"]
            != _CONFIGURATION_RECORD_SHA256[candidate_id]
            or candidate_map["source_id"] != _source_for_candidate(candidate_id)
            or candidate_map["probe_profile_ids"] != list(_PROBE_PROFILE_IDS)
            or not _exact_json_equal(candidate_map["acceptance"], _acceptance_contract())
        ):
            raise ForagerMatchedV3QualificationPlanError("candidate qualification binding drifted")
        _validate_publication_dict(
            cast(dict[str, Any], candidate_map["result_publication_binding"]),
            candidate_id,
            source_tree_by_id,
        )
        expected_scope = {
            "candidate_id": candidate_id,
            "integer_ceiling_fields": list(_RESOURCE_FIELDS),
            "requirement_body_sha256": hashlib.sha256(
                _canonical_json(cast(dict[str, Any], resource))
            ).hexdigest(),
        }
        if candidate_map["resource_scope"] != expected_scope:
            raise ForagerMatchedV3QualificationPlanError("candidate resource binding drifted")

    seed_boundary = cast(dict[str, Any], value["seed_boundary"])
    _require_exact_keys(
        seed_boundary,
        frozenset(
            {
                "material_class",
                "registry_binding_sha256",
                "trust_root_receipt_binding_sha256",
                "trust_root_receipt_external_pin_required",
                "offline_signature_verification_required",
                "offline_signature_verification_implemented_here",
                "preacceptance_chronology_implemented_here",
                "cases",
                "external_authentication_required",
                "issued_before_observation_required",
                "trial_block_material_allowed",
                "held_out_material_allowed",
                "future_randomness_allowed",
                "scientific_seed_reuse_allowed",
            }
        ),
        "qualification seed boundary",
    )
    if (
        seed_boundary["material_class"] != "public_nonbenchmark_permanently_consumed"
        or seed_boundary["registry_binding_sha256"]
        != hashlib.sha256(_canonical_json(seed_registry.to_dict())).hexdigest()
        or seed_boundary["trust_root_receipt_binding_sha256"]
        != hashlib.sha256(_canonical_json(trust_root_receipt.to_dict())).hexdigest()
        or seed_boundary["trust_root_receipt_external_pin_required"] is not True
        or seed_boundary["offline_signature_verification_required"] is not True
        or seed_boundary["offline_signature_verification_implemented_here"] is not False
        or seed_boundary["preacceptance_chronology_implemented_here"] is not False
        or seed_boundary["external_authentication_required"] is not True
        or seed_boundary["issued_before_observation_required"] is not True
        or any(
            seed_boundary[field] is not False
            for field in (
                "trial_block_material_allowed",
                "held_out_material_allowed",
                "future_randomness_allowed",
                "scientific_seed_reuse_allowed",
            )
        )
    ):
        raise ForagerMatchedV3QualificationPlanError("qualification seed boundary drifted")
    cases = seed_boundary["cases"]
    if type(cases) is not list or len(cases) != len(MATCHED_V3_QUALIFICATION_CANDIDATE_IDS):
        raise ForagerMatchedV3QualificationPlanError("qualification case coverage is incomplete")
    case_ids: set[str] = set()
    derivations: set[str] = set()
    for ordinal, (case, candidate_id) in enumerate(
        zip(cases, MATCHED_V3_QUALIFICATION_CANDIDATE_IDS, strict=True)
    ):
        case_map = cast(dict[str, Any], case)
        validated_case = _validate_case_dict(
            case_map,
            candidate_id,
            ordinal,
            seed_registry,
            trust_root_receipt,
        )
        if case_map["case_id"] in case_ids:
            raise ForagerMatchedV3QualificationPlanError("qualification case ID duplicated")
        case_ids.add(case_map["case_id"])
        for digest in (
            validated_case.derivation_payload_sha256,
            validated_case.environment_seed_derivation_sha256,
            validated_case.agent_seed_derivation_sha256,
        ):
            if digest in derivations:
                raise ForagerMatchedV3QualificationPlanError(
                    "qualification derivation identity duplicated"
                )
            derivations.add(digest)

    if not _exact_json_equal(value["failure_policy"], _failure_policy()):
        raise ForagerMatchedV3QualificationPlanError("qualification failure policy drifted")
    if not _exact_json_equal(value["authentication_policy"], _authentication_policy()):
        raise ForagerMatchedV3QualificationPlanError("qualification authentication drifted")
    if not _exact_json_equal(value["claims"], _claims()) or any(
        item is not False for item in value["claims"].values()
    ):
        raise ForagerMatchedV3QualificationPlanError("qualification claim became true")
    if not _exact_json_equal(value["limitations"], _limitations()):
        raise ForagerMatchedV3QualificationPlanError("qualification limitations drifted")
    expected_body = dict(value)
    supplied_body_digest = expected_body.pop("plan_body_sha256")
    _require_sha256(supplied_body_digest, "qualification plan body")
    calculated_body_digest = hashlib.sha256(_canonical_json(expected_body)).hexdigest()
    if not hmac.compare_digest(supplied_body_digest, calculated_body_digest):
        raise ForagerMatchedV3QualificationPlanError("qualification plan body digest disagrees")
    _assert_plain_unaliased_json(value)
    _canonical_json(value)


def matched_v3_qualification_plan_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the fixed descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_qualification_plan_descriptor_bytes() -> bytes:
    """Return exact ASCII canonical descriptor bytes, including the newline."""

    return _DESCRIPTOR_BYTES


def matched_v3_qualification_plan_descriptor_sha256() -> str:
    """Return the frozen descriptor digest."""

    return QUALIFICATION_PLAN_DESCRIPTOR_SHA256


def parse_matched_v3_qualification_plan_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact pinned qualification descriptor."""

    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), QUALIFICATION_PLAN_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification descriptor differs from its frozen identity"
        )
    return value


def canonical_matched_v3_qualification_plan_bytes(
    plan: Mapping[str, Any],
    *,
    expected_qualification_seed_trust_root_receipt_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_binding_sha256: str,
) -> bytes:
    """Validate and encode one caller-built qualification plan."""

    _validate_plan(
        plan,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            expected_qualification_seed_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            expected_qualification_seed_trust_root_receipt_binding_sha256
        ),
    )
    return _canonical_json(plan)


def matched_v3_qualification_plan_sha256(
    plan: Mapping[str, Any],
    *,
    expected_qualification_seed_trust_root_receipt_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_binding_sha256: str,
) -> str:
    """Return the full-file digest for one validated plan."""

    return hashlib.sha256(
        canonical_matched_v3_qualification_plan_bytes(
            plan,
            expected_qualification_seed_trust_root_receipt_file_sha256=(
                expected_qualification_seed_trust_root_receipt_file_sha256
            ),
            expected_qualification_seed_trust_root_receipt_binding_sha256=(
                expected_qualification_seed_trust_root_receipt_binding_sha256
            ),
        )
    ).hexdigest()


def parse_matched_v3_qualification_plan_artifact(
    raw: bytes,
    *,
    expected_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_file_sha256: str,
    expected_qualification_seed_trust_root_receipt_binding_sha256: str,
) -> dict[str, Any]:
    """Parse only with independent plan-file and seed trust-root receipt pins."""

    _require_sha256(expected_file_sha256, "expected qualification-plan file")
    if type(raw) is not bytes:
        raise ForagerMatchedV3QualificationPlanError(
            "qualification-plan artifact input must be exact bytes"
        )
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_file_sha256):
        raise ForagerMatchedV3QualificationPlanError(
            "qualification-plan full-file digest disagrees"
        )
    value = _strict_json_load(raw)
    _validate_plan(
        value,
        expected_qualification_seed_trust_root_receipt_file_sha256=(
            expected_qualification_seed_trust_root_receipt_file_sha256
        ),
        expected_qualification_seed_trust_root_receipt_binding_sha256=(
            expected_qualification_seed_trust_root_receipt_binding_sha256
        ),
    )
    return value


__all__ = [
    "CandidatePublicationBinding",
    "CandidateResourceRequirement",
    "ForagerMatchedV3QualificationPlanError",
    "MATCHED_V3_QUALIFICATION_CANDIDATE_IDS",
    "QUALIFICATION_PLAN_CLASSIFICATION",
    "QUALIFICATION_PLAN_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_DESCRIPTOR_SHA256",
    "QUALIFICATION_PLAN_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_STATUS",
    "QUALIFICATION_SEED_DERIVATION_DOMAIN",
    "QUALIFICATION_SEED_DERIVATION_SCHEMA_VERSION",
    "QUALIFICATION_SEED_OFFLINE_VERIFIER_SCHEMA_VERSION",
    "QUALIFICATION_SEED_PULSE_RECORD_SCHEMA_VERSION",
    "QUALIFICATION_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION",
    "QUALIFICATION_SEED_REGISTRY_SCHEMA_VERSION",
    "QualificationCase",
    "QualificationSeedRegistryBinding",
    "QualificationSeedTrustRootReceiptBinding",
    "RuntimeHelperBinding",
    "RuntimeRequirement",
    "SourceRequirement",
    "build_matched_v3_qualification_plan",
    "canonical_matched_v3_qualification_plan_bytes",
    "canonical_matched_v3_qualification_plan_descriptor_bytes",
    "matched_v3_qualification_plan_descriptor",
    "matched_v3_qualification_plan_descriptor_sha256",
    "matched_v3_qualification_plan_sha256",
    "parse_matched_v3_qualification_plan_artifact",
    "parse_matched_v3_qualification_plan_descriptor",
]
