"""Additive, fail-closed matched-v3 qualification-plan v3 gap ledger.

This standard-library-only module preserves the frozen 28-candidate order while
separating three notions that older plans conflated: a publisher implementation
can be recognized by exact content identity, still be nonqualifying, and still
leave its qualification slot unavailable.  The current local and external
publication closures can therefore account for 26 implementation entries without
making the registry complete.  Both adapter slots remain strict gaps.

The artifact is deliberately unable to accept self-reported structural presence
as evidence.  Source closure, sealed staging, a fresh build, runtime Quicknet
verification, seed chronology, a production host executor, the full resource
merger, observation registry v2, a separate issuer, and a separate evaluator all
remain unsatisfied even when a caller reports every structural input present.

No function here issues a case, seed, execution capability, observation, decision,
or authority.  The two historical images and every component of the prior one-shot
build lineage are permanently rejected.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan_descriptor.v3"
)
QUALIFICATION_PLAN_V3_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_plan.v3"
)
QUALIFICATION_PLAN_V3_STATUS: Final = (
    "contract_implemented_all_qualification_gaps_unsatisfied"
)
QUALIFICATION_PLAN_V3_CLASSIFICATION: Final = (
    "content_only_conditional_publisher_identity_gap_ledger_non_authorizing"
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
MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS: Final = (
    MATCHED_V3_LOCAL_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[:9]
    + MATCHED_V3_ADAPTER_CANDIDATE_IDS
    + MATCHED_V3_EXTERNAL_CANDIDATE_IDS[9:]
)

STRUCTURAL_GAP_IDS: Final = (
    "fresh_source_closure",
    "sealed_staging",
    "fresh_cpu_oci_build",
    "runtime_quicknet_verifier",
    "external_preacceptance_seed_chronology",
    "production_host_executor",
    "full_resource_merger",
    "observation_registry_v2",
    "separate_plan_issuer",
    "separate_acceptance_evaluator",
)

HISTORICAL_IMAGE_IDS: Final = (
    "sha256:a1f491fc786a788b2629e0670ee52ad84138057e58dd795703a830ea2e42c269",
    "sha256:5ecaabefce6439a8731c19e7a55fedb666788242baf035e6ffca86eb31299768",
)
HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS: Final = (
    (
        "context_receipt_sha256",
        "ccacc85f9adf6d81368050be37c67cbd38bb2423cc147deea580a152acf2b330",
    ),
    (
        "execution_receipt_sha256",
        "38cab52b6d247bf045405bd9de9d63b36f00d4e2f79bbb7a154d663ee24b8e9d",
    ),
    (
        "publication_receipt_sha256",
        "28892dd3be5c29df122a94a4feb35045fd17f95475e5e7237c0a04b4b15cbd88",
    ),
)

# Local closure pins are intentionally isolated here so a future independently
# audited source ripple is one compact literal update before descriptor refreezing.
PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "fbc914f1dae39588cb49c76c372db358233302d7a955d9669121e94b08934a6f"
)
PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256: Final = (
    "48640a7e352383eac58fed24c8c36c77fcf3bbed8baf78ce663394d1f7e90200"
)
PINNED_LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "f1fb7d28f0508c38b0d53173707ea5cb006b669793d3401091a942874ee3b878"
)
PINNED_LOCAL_REWARD_BUNDLE_SOURCE_SHA256: Final = (
    "93e824e2518ce405f457329d7c2aa77ddc0fd140d157d155f04f4a9342e0eb9f"
)

# The external closure was independently validated after its strict-type ripple.
PINNED_EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
)
PINNED_EXTERNAL_EXECUTION_CONTRACT_SOURCE_SHA256: Final = (
    "b53381a21f47fd488e79f97630211c2e90ab43faf7775fb8d8ed5cbebcff76d2"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "59d470d6c31e1d3dce8eded401e6331994ca007b94524d8e00714c1f2c66f30b"
)
PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256: Final = (
    "645d232134b220f57b466d3f9c3e140ace8bad3835d9ed290fc066a3c257a80c"
)
PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256: Final = (
    "7c7d007f29b55d6e4a72467d72c4b793568847930d7eb0c17cc276b027e74ceb"
)
PINNED_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256: Final = (
    "f3fba30c37500b73250992cdcef459fb9814aafce056af301e31a2f066a1ab3a"
)
PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256: Final = (
    "0f0c12a93f458ded1188185fed8c0c97e5763f5efa5151f84b70f28b2c945636"
)
PINNED_EXTERNAL_EXECUTION_RUNNER_SOURCE_SHA256: Final = (
    "7ae6a28674076e3e8c0d862d13fc900e7a7c868ef1fa4cb3da333cca35dcc0d7"
)

_HISTORICAL_ADAPTER_PUBLICATION_V1_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)
_HISTORICAL_ADAPTER_PUBLICATION_V1_SOURCE_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)
_UNQUALIFIED_ADAPTER_ATOMIC_V2_DESCRIPTOR_SHA256: Final = (
    "679ea0f6b5d572ec7777d45f4bc115c8d6bcf7df3f3155bd3a784fa59c48dfc6"
)
_UNQUALIFIED_ADAPTER_ATOMIC_V2_SOURCE_SHA256: Final = (
    "bae29ef65246c7beabe34a134a755c18e10a1467dd9914b65be1f05a760bb6f2"
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
_COMPILED_PPO_GRU_RUNNER_DESCRIPTOR_SHA256: Final = (
    "3d95ed7f550cdbd946934e02f452f072bf2a0397a39dfb712be9782d2d6e2565"
)
_COMPILED_PPO_GRU_RUNNER_SOURCE_SHA256: Final = (
    "08dc9c8d36fb98661ec4a8922973dc25df78d881807651f873843e7ddf64a27f"
)
_COMPILED_PPO_GRU_BUNDLE_DESCRIPTOR_SHA256: Final = (
    "cc9e2ad605496682ff2870bb6db312f56ad4926f4805a4a90fbacac4f648cf08"
)
_COMPILED_PPO_GRU_BUNDLE_SOURCE_SHA256: Final = (
    "e50466c185d66334f629915944407d72cb4aff4aa611dffbbe20de8aa8146f6e"
)
_COMPILED_PPO_GRU_SIX_FILE_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "a7827fd32b526c1ad3f9d22549a66fa054c3785c75891560356db82791a3f500"
)
_COMPILED_PPO_GRU_SIX_FILE_PUBLICATION_SOURCE_SHA256: Final = (
    "42ea4bbf5f01818b1f1f44c9410eeaa0a1fe51326a29399c175e1e859e6b8a71"
)

_OBSERVATION_REGISTRY_V1_DESCRIPTOR_SHA256: Final = (
    "f28d01ae9750ee5989f613dbdc64b91f8a8a500faa460b9b5a8c89aa59b31c09"
)
_QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256: Final = (
    "4d2241ebf8e4e431e33addf317c116531a6605a391906f6bddf18491e0764fdd"
)
_SEED_REGISTRY_DESCRIPTOR_SHA256: Final = (
    "fba1ab637f72de87c926169f2e0df5e66a8a2c7dcf855f00442a33dbe42fbef2"
)
_ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256: Final = (
    "e424201576200d05f5da31822cb59a5a61ef06ee29ec267cb20727e8e2e6bfb7"
)
_ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256: Final = (
    "4d34951ccb4b265caa29794457cdd8a5dd837ecf4b73b7a44e4f849bf8c8106e"
)
_HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256: Final = (
    "da7692691aee585b774a2d4a31ba7243d2f5ce005b9b31fe8ceb4a1993653bb8"
)
_HOST_QUALIFICATION_EXECUTOR_SOURCE_SHA256: Final = (
    "d8bbc666a49e252662807f256c7f212c9a7c8c3be279b928a6a93ed77532a2e1"
)

# These three contracts have completed their independent descriptor/source
# audits.  Recognizing their exact identities records source-only validation
# dependencies; it does not imply that a production producer, receipt, reload,
# host, observation issuer, or resource merger exists.
_FINAL_ALGORITHMIC_RESOURCE_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.algorithmic_resource_contract_descriptor.v1"
)
_FINAL_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9eb50aa96169dc9cb38745d729e0b429b01781b32435c86a54cee99b6590321d"
)
_FINAL_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256: Final = (
    "c0df02b504d3d5695782f0b68b1518ae4b549a5e13074c7a5ce6dd39313abef3"
)
_FINAL_STORAGE_BOUNDARY_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_storage_boundary_contract_descriptor.v1"
)
_FINAL_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "d294de196f3b96192e3810571ddbe5b39fdf4615efec9d4460cf4e4d5f6c6a4c"
)
_FINAL_STORAGE_BOUNDARY_CONTRACT_SOURCE_SHA256: Final = (
    "9ae173c4ddbecac1ea64777d6227db6f07b78db97c8485175e7cf4954b645dcf"
)
_FINAL_NORMALIZED_PUBLICATION_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_publication_commitment_contract_descriptor.v1"
)
_FINAL_NORMALIZED_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "e2b2c556bba5ee4eb168a1d990eb73b6b273a6685c7e86818ed5bee142191420"
)
_FINAL_NORMALIZED_PUBLICATION_CONTRACT_SOURCE_SHA256: Final = (
    "7737ff1b12dab2fc569cda241821a37fee47c6038dcadf1c3578f79fccf82c80"
)

# The first algorithmic-resource validator was retired by the finalized
# contract above.  Descriptor and source identities share one deny-union on
# purpose: neither identity kind may be relabelled as a full-merger producer.
_RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256: Final = (
    "12e6b772ac8930b83752446b5754b7a76709c491b5ed54eb242422f73d3d5733"
)
_RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256: Final = (
    "e6b9a736fdaff1bcf1b6467eadbd8441fc7f1d0be45bc419fe6385f36b241bf8"
)
_ALGORITHMIC_RESOURCE_VALIDATOR_CROSS_KIND_MERGER_EXCLUSION_SHA256S: Final = (
    _FINAL_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256,
    _FINAL_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256,
    _RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256,
    _RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
)

_HOST_EXECUTOR_V2_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.host_qualification_executor_descriptor.v2"
)
_OBSERVATION_REGISTRY_V2_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v2"
)
_OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_observation_candidate_batch.v2"
)
_FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_resource_merger_descriptor.v1"
)
_FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.full_resource_merger_receipt.v1"
)
_RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.qualification_resource_merger_candidate.v2"
)

_MAX_ARTIFACT_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 150_000
_MAX_TEXT_LENGTH: Final = 16_384
_MAX_INTEGER: Final = 2**63 - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PORTABLE_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\Z")


class ForagerMatchedV3QualificationPlanV3Error(ValueError):
    """A v3 gap-ledger identity, lineage, or canonical artifact failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3QualificationPlanV3Error(message)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_image_id(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_ID_RE.fullmatch(value) is None
        or value == "sha256:" + "0" * 64
    ):
        _fail(f"{label} must be one sha256: image ID")
    return value


def _require_relative_path(value: object, label: str) -> str:
    if type(value) is not str or _RELATIVE_PATH_RE.fullmatch(value) is None:
        _fail(f"{label} must use canonical relative slash syntax")
    components = value.split("/")
    if (
        not components
        or any(component in {"", ".", ".."} for component in components)
        or "/".join(components) != value
    ):
        _fail(f"{label} contains a noncanonical path component")
    return value


def _reject_constant(value: str) -> NoReturn:
    _fail(f"qualification-plan v3 JSON contains non-finite constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"qualification-plan v3 JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("qualification-plan v3 JSON integer exceeds its lexical bound")
    result = int(value)
    if not -_MAX_INTEGER <= result <= _MAX_INTEGER:
        _fail("qualification-plan v3 JSON integer exceeds its value bound")
    return result


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"qualification-plan v3 JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("qualification-plan v3 JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("qualification-plan v3 JSON exceeds its depth bound")
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail("qualification-plan v3 JSON integer exceeds its value bound")
            continue
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("qualification-plan v3 JSON strings must be bounded printable ASCII")
            continue
        if type(item) not in {dict, list}:
            _fail("qualification-plan v3 JSON contains an inexact non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("qualification-plan v3 JSON containers must be unaliased")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in cast(list[object], item))
        else:
            for key, child in cast(dict[object, object], item).items():
                if type(key) is not str or len(key) > _MAX_TEXT_LENGTH:
                    _fail("qualification-plan v3 JSON object key is invalid")
                pending.append((child, depth + 1))


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
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ForagerMatchedV3QualificationPlanV3Error(
            "qualification-plan v3 JSON cannot be canonically encoded"
        ) from exc
    if newline:
        raw += b"\n"
    if not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("qualification-plan v3 JSON exceeds its byte bound")
    return raw


def _strict_json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= _MAX_ARTIFACT_BYTES:
        _fail("qualification-plan v3 bytes are invalid or exceed their bound")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3QualificationPlanV3Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ForagerMatchedV3QualificationPlanV3Error(
            "qualification-plan v3 bytes are not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("qualification-plan v3 JSON root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if raw != _canonical_json(result):
        _fail("qualification-plan v3 bytes are not exact canonical JSON")
    return result


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return set(left_map) == set(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        return len(left_items) == len(right_items) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    return bool(left == right)


@dataclass(frozen=True, slots=True)
class PublisherComponentIdentityV3:
    """One exact descriptor/source pair in a conditionally recognized closure."""

    role: str
    descriptor_schema_version: str
    descriptor_sha256: str
    implementation_path: str
    implementation_source_sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not str or _PORTABLE_ID_RE.fullmatch(self.role) is None:
            _fail("publisher component role is invalid")
        if (
            type(self.descriptor_schema_version) is not str
            or not self.descriptor_schema_version.startswith("alberta.forager_matched_v3.")
        ):
            _fail("publisher component descriptor schema is invalid")
        _require_relative_path(
            self.implementation_path,
            "publisher component implementation path",
        )
        _require_sha256(self.descriptor_sha256, "publisher component descriptor")
        _require_sha256(self.implementation_source_sha256, "publisher component source")

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_sha256": self.descriptor_sha256,
            "implementation_path": self.implementation_path,
            "implementation_source_sha256": self.implementation_source_sha256,
        }


@dataclass(frozen=True, slots=True)
class PublisherClosureIdentityV3:
    """Exact family closure whose recognition grants no qualification status."""

    family: str
    candidate_ids: tuple[str, ...]
    trust_direction: str
    components: tuple[PublisherComponentIdentityV3, ...]

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in {"local", "external"}:
            _fail("publisher closure family must be local or external")
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            _fail("publisher closure candidate IDs must be one nonempty exact tuple")
        if any(type(item) is not str for item in self.candidate_ids):
            _fail("publisher closure candidate IDs must be exact strings")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            _fail("publisher closure candidate IDs repeat")
        if (
            type(self.trust_direction) is not str
            or _PORTABLE_ID_RE.fullmatch(self.trust_direction) is None
        ):
            _fail("publisher closure trust direction is invalid")
        if type(self.components) is not tuple or not self.components:
            _fail("publisher closure components must be one nonempty exact tuple")
        if any(type(item) is not PublisherComponentIdentityV3 for item in self.components):
            _fail("publisher closure contains an inexact component")
        if len({id(item) for item in self.components}) != len(self.components):
            _fail("publisher closure components are aliased")
        roles = tuple(item.role for item in self.components)
        if len(set(roles)) != len(roles):
            _fail("publisher closure component roles repeat")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "candidate_ids": list(self.candidate_ids),
            "trust_direction": self.trust_direction,
            "components": [item.to_dict() for item in self.components],
            "recognition_disposition": "implementation_identity_only_nonqualifying",
            "qualification_ready": False,
        }


@dataclass(frozen=True, slots=True)
class QualificationPlanV3StructuralInputs:
    """Caller-reported structural presence; never evidence or gap satisfaction."""

    signals: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if type(self.signals) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in self.signals
        ):
            _fail("structural inputs must be exact pairs")
        if tuple(item[0] for item in self.signals) != STRUCTURAL_GAP_IDS:
            _fail("structural inputs must use the exact gap order")
        if any(type(name) is not str for name, _value in self.signals):
            _fail("structural input names must be exact strings")
        if any(type(value) is not bool for _name, value in self.signals):
            _fail("structural input values must be exact booleans")

    @classmethod
    def from_mapping(cls, values: Mapping[str, bool]) -> QualificationPlanV3StructuralInputs:
        if (
            type(values) is not dict
            or any(type(key) is not str for key in values)
            or set(values) != set(STRUCTURAL_GAP_IDS)
        ):
            _fail("structural input mapping keys differ")
        return cls(tuple((name, values[name]) for name in STRUCTURAL_GAP_IDS))

    def to_dict(self) -> dict[str, bool]:
        return dict(self.signals)


def matched_v3_local_publisher_closure_v3() -> PublisherClosureIdentityV3:
    """Return the independently audited local bundle/publisher identity pair."""

    return PublisherClosureIdentityV3(
        family="local",
        candidate_ids=MATCHED_V3_LOCAL_CANDIDATE_IDS,
        trust_direction="bundle_to_publisher_to_atomic_with_live_reverse_replay_no_cycle",
        components=(
            PublisherComponentIdentityV3(
                role="local_reward_bundle",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.local_reward_bundle_descriptor.v1"
                ),
                descriptor_sha256=PINNED_LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_local_reward_bundle.py"
                ),
                implementation_source_sha256=PINNED_LOCAL_REWARD_BUNDLE_SOURCE_SHA256,
            ),
            PublisherComponentIdentityV3(
                role="local_reward_publisher",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.local_reward_publication_descriptor.v1"
                ),
                descriptor_sha256=PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_local_reward_publication.py"
                ),
                implementation_source_sha256=(
                    PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256
                ),
            ),
        ),
    )


def matched_v3_external_publisher_closure_v3() -> PublisherClosureIdentityV3:
    """Return the validated external contract/runner/consumer/publisher closure."""

    return PublisherClosureIdentityV3(
        family="external",
        candidate_ids=MATCHED_V3_EXTERNAL_CANDIDATE_IDS,
        trust_direction=(
            "contract_to_runner_to_consumer_to_publisher_to_atomic_"
            "no_reverse_source_cycle"
        ),
        components=(
            PublisherComponentIdentityV3(
                role="external_execution_contract",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.external_execution_contract_descriptor.v1"
                ),
                descriptor_sha256=PINNED_EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_external_execution_contract.py"
                ),
                implementation_source_sha256=(
                    PINNED_EXTERNAL_EXECUTION_CONTRACT_SOURCE_SHA256
                ),
            ),
            PublisherComponentIdentityV3(
                role="external_execution_runner",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.external_execution_runner_descriptor.v1"
                ),
                descriptor_sha256=PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_external_execution_runner.py"
                ),
                implementation_source_sha256=(
                    PINNED_EXTERNAL_EXECUTION_RUNNER_SOURCE_SHA256
                ),
            ),
            PublisherComponentIdentityV3(
                role="external_outcome_consumer",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.external_outcome_consumer_descriptor.v1"
                ),
                descriptor_sha256=PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_external_outcome_consumer.py"
                ),
                implementation_source_sha256=(
                    PINNED_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256
                ),
            ),
            PublisherComponentIdentityV3(
                role="external_reward_publisher",
                descriptor_schema_version=(
                    "alberta.forager_matched_v3.external_reward_publication_descriptor.v1"
                ),
                descriptor_sha256=PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
                implementation_path=(
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_external_reward_publication.py"
                ),
                implementation_source_sha256=(
                    PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256
                ),
            ),
        ),
    )


def synthetically_complete_matched_v3_structural_inputs_v3() -> (
    QualificationPlanV3StructuralInputs
):
    """Return all-true caller signals; they deliberately satisfy no gap."""

    return QualificationPlanV3StructuralInputs.from_mapping(
        {name: True for name in STRUCTURAL_GAP_IDS}
    )


def _claims() -> dict[str, bool]:
    return {
        "authority_granted": False,
        "build_qualified": False,
        "evidence_authority": False,
        "executed_bytecode_attested": False,
        "execution_authorized": False,
        "performance_claim_allowed": False,
        "production_plan_issued": False,
        "promotion_allowed": False,
        "publication_authority_granted": False,
        "publisher_registry_complete": False,
        "qualification_granted": False,
        "resource_matched": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "seed_chronology_accepted": False,
        "source_qualified": False,
        "staging_qualified": False,
        "universal_sota_claim_allowed": False,
    }


def _readiness() -> dict[str, bool]:
    return {
        "acceptance_evaluator_available": False,
        "execution_ready": False,
        "fresh_build_available": False,
        "full_resource_merger_available": False,
        "observation_registry_v2_available": False,
        "plan_issuer_available": False,
        "production_host_executor_available": False,
        "publisher_registry_ready": False,
        "qualification_ready": False,
        "runtime_quicknet_verifier_available": False,
        "seed_chronology_acceptor_available": False,
        "structural_inputs_sufficient": False,
    }


_GAP_REQUIREMENTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "fresh_source_closure": "new jointly bound local_external_adapter source closure",
        "sealed_staging": "new closure-bound sealed staging artifact",
        "fresh_cpu_oci_build": (
            "new build receipts and image excluding every historical lineage"
        ),
        "runtime_quicknet_verifier": (
            "runtime verifier with authenticated Quicknet vectors and key"
        ),
        "external_preacceptance_seed_chronology": (
            "independent acceptance proving seed receipt chronology before observation"
        ),
        "production_host_executor": "separately authorized implemented host OCI executor",
        "full_resource_merger": "complete 28-field all-descendant resource merger",
        "observation_registry_v2": "v3-bound registry with exact v2 observation contracts",
        "separate_plan_issuer": "future issuer outside this content descriptor",
        "separate_acceptance_evaluator": "future evaluator outside issuer and executor",
    }
)


def _gap_ledger() -> list[dict[str, Any]]:
    return [
        {
            "gap_id": gap_id,
            "required_future_artifact": _GAP_REQUIREMENTS[gap_id],
            "satisfied": False,
            "accepted_artifact_sha256": None,
            "caller_structural_presence_can_satisfy": False,
        }
        for gap_id in STRUCTURAL_GAP_IDS
    ]


def _adapter_exclusions() -> dict[str, Any]:
    return {
        "strict_candidate_ids": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
        "strict_slot_count": 2,
        "recognized_strict_slot_count": 0,
        "missing_strict_candidate_ids": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
        "historical_adapter_publication_v1": {
            "descriptor_sha256": _HISTORICAL_ADAPTER_PUBLICATION_V1_DESCRIPTOR_SHA256,
            "source_sha256": _HISTORICAL_ADAPTER_PUBLICATION_V1_SOURCE_SHA256,
            "counted_toward_strict_slots": False,
            "disposition": "historical_v1_excluded_not_reinterpreted",
        },
        "adapter_atomic_publication_v2": {
            "descriptor_sha256": _UNQUALIFIED_ADAPTER_ATOMIC_V2_DESCRIPTOR_SHA256,
            "source_sha256": _UNQUALIFIED_ADAPTER_ATOMIC_V2_SOURCE_SHA256,
            "status": (
                "implemented_unexecuted_unqualified_surfaces_host_isolation_unproven"
            ),
            "counted_toward_strict_slots": False,
            "disposition": "recognized_unqualified_content_not_strict_publisher",
        },
        "historical_compiled_ppo_gru_addendum": {
            "candidate_id": "adapted_ppo_gru",
            "status": "implemented_compiled_path_unqualified_historical_addendum",
            "components": [
                {
                    "role": "compiled_ppo_gru_runner",
                    "descriptor_sha256": (
                        _COMPILED_PPO_GRU_RUNNER_DESCRIPTOR_SHA256
                    ),
                    "source_sha256": _COMPILED_PPO_GRU_RUNNER_SOURCE_SHA256,
                },
                {
                    "role": "compiled_ppo_gru_bundle",
                    "descriptor_sha256": (
                        _COMPILED_PPO_GRU_BUNDLE_DESCRIPTOR_SHA256
                    ),
                    "source_sha256": _COMPILED_PPO_GRU_BUNDLE_SOURCE_SHA256,
                },
                {
                    "role": "compiled_ppo_gru_six_file_publication",
                    "descriptor_sha256": (
                        _COMPILED_PPO_GRU_SIX_FILE_PUBLICATION_DESCRIPTOR_SHA256
                    ),
                    "source_sha256": (
                        _COMPILED_PPO_GRU_SIX_FILE_PUBLICATION_SOURCE_SHA256
                    ),
                },
            ],
            "fills_adapted_ppo_gru_slot": False,
            "counted_toward_strict_slots": False,
            "qualification_ready": False,
            "relabel_as_qualified_allowed": False,
            "escape_via_historical_addendum_allowed": False,
        },
        "runner_dispositions": {
            "adapted_full_rainbow": {
                "descriptor_sha256": _FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256,
                "source_sha256": _FULL_RAINBOW_RUNNER_SOURCE_SHA256,
                "status": "implemented_unqualified",
                "qualification_ready": False,
                "relabel_as_qualified_allowed": False,
            },
            "adapted_ppo_gru": {
                "descriptor_sha256": _PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
                "source_sha256": _PPO_GRU_RUNNER_SOURCE_SHA256,
                "status": "implemented_runtime_unqualified",
                "qualification_ready": False,
                "relabel_as_qualified_allowed": False,
            },
        },
    }


def _historical_exclusions() -> dict[str, Any]:
    return {
        "image_ids": list(HISTORICAL_IMAGE_IDS),
        "one_shot_build_lineage_components": [
            {"role": role, "sha256": digest}
            for role, digest in HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS
        ],
        "partial_component_reuse_allowed": False,
        "historical_image_relabel_allowed": False,
        "automatic_rebuild_or_reuse_allowed": False,
    }


def _publisher_policy() -> dict[str, Any]:
    return {
        "required_candidate_count": 28,
        "conditional_recognition_is_qualification": False,
        "recognized_implementation_can_issue_or_execute": False,
        "local": {
            "maximum_candidate_count": 14,
            "expected_closure": matched_v3_local_publisher_closure_v3().to_dict(),
            "recognition": "only_if_every_literal_identity_and_order_match",
            "qualification_ready": False,
        },
        "external": {
            "maximum_candidate_count": 12,
            "expected_closure": matched_v3_external_publisher_closure_v3().to_dict(),
            "recognition": "only_if_every_literal_identity_and_order_match",
            "qualification_ready": False,
        },
        "adapter": _adapter_exclusions(),
        "maximum_recognizable_nonqualifying_count": 26,
        "strict_qualifying_count": 0,
        "complete": False,
    }


def _finalized_source_only_dependencies() -> dict[str, Any]:
    common = {
        "recognition": "exact_finalized_source_only_identity",
        "source_contract_implemented": True,
        "production_implementation_available": False,
        "production_receipt_available": False,
        "qualification_ready": False,
        "non_authorizing": True,
        "fills_any_structural_gap": False,
    }
    return {
        "algorithmic_resource_contract": {
            "schema_version": _FINAL_ALGORITHMIC_RESOURCE_CONTRACT_SCHEMA_VERSION,
            "status": "implemented_source_only_contract_uninvoked_no_production_receipt",
            "classification": (
                "score_blind_metadata_only_algorithmic_resource_contract_non_authorizing"
            ),
            "implementation_path": (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_algorithmic_resource_contract.py"
            ),
            "descriptor_sha256": (
                _FINAL_ALGORITHMIC_RESOURCE_CONTRACT_DESCRIPTOR_SHA256
            ),
            "implementation_source_sha256": (
                _FINAL_ALGORITHMIC_RESOURCE_CONTRACT_SOURCE_SHA256
            ),
            **common,
        },
        "storage_boundary_contract": {
            "schema_version": _FINAL_STORAGE_BOUNDARY_CONTRACT_SCHEMA_VERSION,
            "status": "implemented_source_only_contract_uninvoked_no_production_receipt",
            "classification": (
                "score_blind_metadata_only_storage_boundary_contract_non_authorizing"
            ),
            "implementation_path": (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_qualification_storage_boundary.py"
            ),
            "descriptor_sha256": (
                _FINAL_STORAGE_BOUNDARY_CONTRACT_DESCRIPTOR_SHA256
            ),
            "implementation_source_sha256": (
                _FINAL_STORAGE_BOUNDARY_CONTRACT_SOURCE_SHA256
            ),
            **common,
        },
        "normalized_publication_contract": {
            "schema_version": _FINAL_NORMALIZED_PUBLICATION_CONTRACT_SCHEMA_VERSION,
            "status": (
                "implemented_source_only_expected_reload_commitment_non_authorizing"
            ),
            "classification": (
                "score_blind_metadata_only_normalized_commitment_non_authorizing"
            ),
            "implementation_path": (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_qualification_publication_commitment.py"
            ),
            "descriptor_sha256": (
                _FINAL_NORMALIZED_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
            ),
            "implementation_source_sha256": (
                _FINAL_NORMALIZED_PUBLICATION_CONTRACT_SOURCE_SHA256
            ),
            **common,
        },
    }


def _cross_kind_identity_exclusions() -> dict[str, Any]:
    return {
        "algorithmic_resource_validator_identity_sha256_union": list(
            _ALGORITHMIC_RESOURCE_VALIDATOR_CROSS_KIND_MERGER_EXCLUSION_SHA256S
        ),
        "retired_algorithmic_resource_validator": {
            "descriptor_sha256": (
                _RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_DESCRIPTOR_SHA256
            ),
            "source_sha256": _RETIRED_ALGORITHMIC_RESOURCE_VALIDATOR_SOURCE_SHA256,
            "retired": True,
            "relabel_allowed": False,
        },
        "descriptor_source_cross_kind_substitution_allowed": False,
        "algorithmic_validator_can_substitute_for_full_resource_merger": False,
        "storage_validator_can_substitute_for_full_resource_merger": False,
        "endpoint_observer_can_substitute_for_full_resource_merger": False,
    }


def _observation_and_execution_gaps() -> dict[str, Any]:
    return {
        "observation_registry_v1": {
            "schema_version": (
                "alberta.forager_matched_v3.qualification_observation_registry_descriptor.v1"
            ),
            "descriptor_sha256": _OBSERVATION_REGISTRY_V1_DESCRIPTOR_SHA256,
            "status": "implemented_structural_validators_no_observation_issuer",
            "v2_compatible": False,
            "v3_compatible": False,
            "explicit_blocker": True,
            "may_fill_observation_registry_v2_gap": False,
        },
        "observation_registry_v2": {
            "schema_version": _OBSERVATION_REGISTRY_V2_DESCRIPTOR_SCHEMA_VERSION,
            "candidate_batch_schema_version": (
                _OBSERVATION_CANDIDATE_BATCH_V2_SCHEMA_VERSION
            ),
            "implementation_path": (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_qualification_observations_v2.py"
            ),
            "descriptor_body_sha256": None,
            "descriptor_file_sha256": None,
            "implementation_source_sha256": None,
            "structural_candidate_validator_implemented": True,
            "descriptor_finalized": False,
            "source_identity_finalized": False,
            "observation_issuer_available": False,
            "acceptance_evaluator_available": False,
            "production_registry_available": False,
            "gap_satisfied": False,
        },
        "host_executor": {
            "incompatible_v1": {
                "schema_version": (
                    "alberta.forager_matched_v3.host_qualification_executor_descriptor.v1"
                ),
                "implementation_path": (
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_host_qualification_executor.py"
                ),
                "descriptor_sha256": _HOST_QUALIFICATION_EXECUTOR_DESCRIPTOR_SHA256,
                "implementation_source_sha256": (
                    _HOST_QUALIFICATION_EXECUTOR_SOURCE_SHA256
                ),
                "metadata_only_structural_content_recognized": True,
                "v2_compatible": False,
                "permanently_excluded_from_v2_slot": True,
                "may_fill_production_executor_gap": False,
            },
            "source_only_v2": {
                "schema_version": _HOST_EXECUTOR_V2_DESCRIPTOR_SCHEMA_VERSION,
                "implementation_path": (
                    "alberta_framework/benchmarks/"
                    "forager_matched_v3_host_qualification_executor_v2.py"
                ),
                "descriptor_body_sha256": None,
                "descriptor_file_sha256": None,
                "implementation_source_sha256": None,
                "source_contract_implemented": True,
                "descriptor_finalized": False,
                "source_identity_finalized": False,
                "production_backend_available": False,
                "production_executor_available": False,
                "may_self_issue": False,
                "may_self_evaluate": False,
                "gap_satisfied": False,
            },
        },
        "quicknet": {
            "source_descriptor_sha256": _QUICKNET_VERIFIER_SOURCE_DESCRIPTOR_SHA256,
            "source_only_content_recognized": True,
            "runtime_verifier_available": False,
            "runtime_verification_accepted": False,
        },
        "seed_registry": {
            "descriptor_sha256": _SEED_REGISTRY_DESCRIPTOR_SHA256,
            "content_only_derivation_recognized": True,
            "external_preacceptance_chronology_accepted": False,
            "seed_issuer_available": False,
        },
        "resource_observation": {
            "endpoint_observer_descriptor_sha256": (
                _ENDPOINT_RESOURCE_OBSERVER_DESCRIPTOR_SHA256
            ),
            "endpoint_observer_source_sha256": _ENDPOINT_RESOURCE_OBSERVER_SOURCE_SHA256,
            "endpoint_observer_counts_as_full_28_field_merger": False,
            "full_resource_merger_available": False,
        },
        "full_resource_merger": {
            "descriptor_schema_version": _FULL_RESOURCE_MERGER_DESCRIPTOR_SCHEMA_VERSION,
            "receipt_schema_version": _FULL_RESOURCE_MERGER_RECEIPT_SCHEMA_VERSION,
            "structural_candidate_schema_version": (
                _RESOURCE_MERGER_CANDIDATE_V2_SCHEMA_VERSION
            ),
            "structural_candidate_validator_path": (
                "alberta_framework/benchmarks/"
                "forager_matched_v3_qualification_observations_v2.py"
            ),
            "implementation_path": None,
            "descriptor_body_sha256": None,
            "descriptor_file_sha256": None,
            "implementation_source_sha256": None,
            "structural_candidate_validator_implemented": True,
            "descriptor_finalized": False,
            "source_identity_finalized": False,
            "production_merger_implemented": False,
            "production_receipt_available": False,
            "endpoint_observer_can_substitute": False,
            "finalized_dependency_identity_can_substitute": False,
            "gap_satisfied": False,
        },
        "finalized_source_only_dependencies": _finalized_source_only_dependencies(),
        "cross_kind_identity_exclusions": _cross_kind_identity_exclusions(),
        "future_separation": {
            "plan_issuer_must_be_separate": True,
            "acceptance_evaluator_must_be_separate": True,
            "host_executor_may_not_self_issue_or_self_evaluate": True,
            "issuer_available": False,
            "evaluator_available": False,
        },
    }


def _limitations() -> list[str]:
    return [
        "Publisher closure recognition is content identity only, not qualification.",
        "All ten production gaps remain unsatisfied under all-true structural signals.",
        "Observation registry v1 is incompatible and has no observation issuer.",
        "Host v1 is incompatible; source-only host v2 has no production backend or final pins.",
        "Observation registry v2 is source-only, unpinned, and has no issuer or evaluator.",
        "No full resource-merger producer, final identity, or production receipt exists.",
        "Final algorithmic, storage, and publication contracts remain nonauthorizing.",
        "Neither historical adapter publication implementation fills a strict adapter slot.",
        "Full Rainbow and PPO-GRU runners remain explicitly unqualified.",
        "A new source closure, sealed staging, fresh build, and new image are required.",
        "A future separate issuer and a future separate evaluator remain mandatory.",
        "No plan or assessment permits performance, comparative, promotion, or SOTA claims.",
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
        "plan_schema_version": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "status": QUALIFICATION_PLAN_V3_STATUS,
        "classification": QUALIFICATION_PLAN_V3_CLASSIFICATION,
        "versioning": {
            "additive": True,
            "plan_v2_mutated": False,
            "plan_v2_accepted_as_v3": False,
            "frozen_outputs_modified": False,
        },
        "universe": {
            "horizon": MATCHED_V3_HORIZON,
            "candidate_count": 28,
            "candidate_order": list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
            "partitions": {
                "local": list(MATCHED_V3_LOCAL_CANDIDATE_IDS),
                "external": list(MATCHED_V3_EXTERNAL_CANDIDATE_IDS),
                "adapter": list(MATCHED_V3_ADAPTER_CANDIDATE_IDS),
            },
            "partition_counts": {"local": 14, "external": 12, "adapter": 2},
        },
        "publisher_registry": _publisher_policy(),
        "historical_exclusions": _historical_exclusions(),
        "required_gap_ledger": _gap_ledger(),
        "observation_and_execution_gaps": _observation_and_execution_gaps(),
        "readiness": _readiness(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BODY_BYTES: Final = _canonical_json(_descriptor(), newline=False)
_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256: Final = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)
# This existing public name is the descriptor FILE identity: compact sorted
# ASCII plus exactly one trailing LF.  BODY and FILE pins remain independent.
QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256: Final = (
    "0000000000000000000000000000000000000000000000000000000000000000"
)


def _require_descriptor_pins() -> tuple[str, str]:
    body_pin = QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256
    file_pin = QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256
    if body_pin == "0" * 64 or file_pin == "0" * 64:
        _fail("qualification-plan v3 descriptor BODY/FILE pins are not finalized")
    _require_sha256(body_pin, "qualification-plan v3 descriptor BODY")
    _require_sha256(file_pin, "qualification-plan v3 descriptor FILE")
    observed_body = hashlib.sha256(_DESCRIPTOR_BODY_BYTES).hexdigest()
    observed_file = hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest()
    if (
        not hmac.compare_digest(observed_body, body_pin)
        or not hmac.compare_digest(observed_file, file_pin)
        or _DESCRIPTOR_BYTES != _DESCRIPTOR_BODY_BYTES + b"\n"
    ):
        _fail("qualification-plan v3 descriptor BODY/FILE identity drifted")
    return body_pin, file_pin


def _validated_publisher_closures(
    closures: Sequence[PublisherClosureIdentityV3],
) -> dict[str, PublisherClosureIdentityV3]:
    if type(closures) not in {tuple, list}:
        _fail("publisher closures must be one exact tuple or list")
    if len(closures) > 2 or len({id(item) for item in closures}) != len(closures):
        _fail("publisher closures exceed their family bound or are aliased")
    if any(type(item) is not PublisherClosureIdentityV3 for item in closures):
        _fail("publisher closures contain an inexact value")
    by_family = {item.family: item for item in closures}
    if len(by_family) != len(closures):
        _fail("publisher closures repeat a family")
    expected = {
        "local": matched_v3_local_publisher_closure_v3(),
        "external": matched_v3_external_publisher_closure_v3(),
    }
    for family, closure in by_family.items():
        if closure != expected[family]:
            _fail(f"{family} publisher closure identity or order differs")
    return by_family


def _validated_proposed_lineage(
    *,
    proposed_image_id: str | None,
    proposed_build_lineage_components: Sequence[str],
) -> dict[str, Any]:
    image: str | None = None
    if proposed_image_id is not None:
        image = _require_image_id(proposed_image_id, "proposed v3 image")
        if image in HISTORICAL_IMAGE_IDS:
            _fail("historical CPU OCI image is permanently forbidden in plan v3")
    if type(proposed_build_lineage_components) not in {tuple, list}:
        _fail("proposed build lineage components must be one exact tuple or list")
    components = tuple(
        _require_sha256(item, "proposed build lineage component")
        for item in proposed_build_lineage_components
    )
    if len(set(components)) != len(components):
        _fail("proposed build lineage components repeat")
    historical = {digest for _role, digest in HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS}
    if any(item in historical for item in components):
        _fail("prior one-shot build lineage component is permanently forbidden")
    return {
        "image_id": image,
        "build_lineage_components": list(components),
        "image_present": image is not None,
        "component_count": len(components),
        "accepted_as_fresh_build": False,
        "qualifies_runtime": False,
    }


def _candidate_publisher_entries(
    closures: Mapping[str, PublisherClosureIdentityV3],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ordinal, candidate_id in enumerate(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS):
        if candidate_id in MATCHED_V3_LOCAL_CANDIDATE_IDS:
            family = "local"
        elif candidate_id in MATCHED_V3_EXTERNAL_CANDIDATE_IDS:
            family = "external"
        else:
            family = "adapter"
        if family == "adapter":
            status = "strict_adapter_publisher_required_not_implemented"
        elif family in closures:
            status = "recognized_nonqualifying_publisher_implementation"
        else:
            status = "required_family_closure_not_recognized"
        result.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                "family": family,
                "publisher_status": status,
                "qualification_ready": False,
            }
        )
    return result


def _plan_body(
    *,
    structural_inputs: QualificationPlanV3StructuralInputs,
    publisher_closures: Sequence[PublisherClosureIdentityV3],
    proposed_image_id: str | None,
    proposed_build_lineage_components: Sequence[str],
) -> dict[str, Any]:
    descriptor_body_sha256, descriptor_file_sha256 = _require_descriptor_pins()
    if type(structural_inputs) is not QualificationPlanV3StructuralInputs:
        _fail("structural inputs have an inexact type")
    closures = _validated_publisher_closures(publisher_closures)
    lineage = _validated_proposed_lineage(
        proposed_image_id=proposed_image_id,
        proposed_build_lineage_components=proposed_build_lineage_components,
    )
    signals = structural_inputs.to_dict()
    gaps = [
        {
            **gap,
            "caller_reports_structural_presence": signals[cast(str, gap["gap_id"])],
        }
        for gap in _gap_ledger()
    ]
    candidate_entries = _candidate_publisher_entries(closures)
    recognized = sum(
        item["publisher_status"]
        == "recognized_nonqualifying_publisher_implementation"
        for item in candidate_entries
    )
    return {
        "schema_version": QUALIFICATION_PLAN_V3_SCHEMA_VERSION,
        "status": QUALIFICATION_PLAN_V3_STATUS,
        "classification": QUALIFICATION_PLAN_V3_CLASSIFICATION,
        "descriptor_binding": {
            "schema_version": QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION,
            "body_sha256": descriptor_body_sha256,
            "file_sha256": descriptor_file_sha256,
        },
        "candidate_order": list(MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS),
        "partition_counts": {"local": 14, "external": 12, "adapter": 2},
        "structural_inputs": structural_inputs.to_dict(),
        "blocker_ledger": gaps,
        "publisher_registry": {
            "recognized_closures": [
                closures[family].to_dict()
                for family in ("local", "external")
                if family in closures
            ],
            "candidate_entries": candidate_entries,
            "recognized_nonqualifying_count": recognized,
            "missing_strict_adapter_count": 2,
            "missing_strict_adapter_candidate_ids": list(
                MATCHED_V3_ADAPTER_CANDIDATE_IDS
            ),
            "qualifying_count": 0,
            "complete": False,
        },
        "proposed_lineage": lineage,
        "historical_exclusions": _historical_exclusions(),
        "observation_and_execution_gaps": _observation_and_execution_gaps(),
        "future_authority_boundary": {
            "separate_plan_issuer_required": True,
            "separate_acceptance_evaluator_required": True,
            "implemented_here": False,
        },
        "readiness": _readiness(),
        "claims": _claims(),
        "limitations": _limitations(),
    }


def build_matched_v3_qualification_plan_v3(
    *,
    structural_inputs: QualificationPlanV3StructuralInputs,
    publisher_closures: Sequence[PublisherClosureIdentityV3],
    proposed_image_id: str | None,
    proposed_build_lineage_components: Sequence[str],
) -> dict[str, Any]:
    """Build one nonissuing v3 gap ledger under explicit caller inputs."""

    _require_descriptor_pins()
    body = _plan_body(
        structural_inputs=structural_inputs,
        publisher_closures=publisher_closures,
        proposed_image_id=proposed_image_id,
        proposed_build_lineage_components=proposed_build_lineage_components,
    )
    result = {
        **body,
        "plan_body_sha256": hashlib.sha256(_canonical_json(body, newline=False)).hexdigest(),
    }
    return copy.deepcopy(result)


def _component_from_dict(value: object) -> PublisherComponentIdentityV3:
    if type(value) is not dict or set(value) != {
        "role",
        "descriptor_schema_version",
        "descriptor_sha256",
        "implementation_path",
        "implementation_source_sha256",
    }:
        _fail("publisher component artifact keys differ")
    return PublisherComponentIdentityV3(**cast(dict[str, Any], value))


def _closure_from_dict(value: object) -> PublisherClosureIdentityV3:
    if type(value) is not dict or set(value) != {
        "family",
        "candidate_ids",
        "trust_direction",
        "components",
        "recognition_disposition",
        "qualification_ready",
    }:
        _fail("publisher closure artifact keys differ")
    item = dict(cast(dict[str, Any], value))
    if (
        item.pop("recognition_disposition")
        != "implementation_identity_only_nonqualifying"
        or item.pop("qualification_ready") is not False
        or type(item["candidate_ids"]) is not list
        or type(item["components"]) is not list
    ):
        _fail("publisher closure artifact disposition or collections differ")
    item["candidate_ids"] = tuple(item["candidate_ids"])
    item["components"] = tuple(_component_from_dict(value) for value in item["components"])
    return PublisherClosureIdentityV3(**item)


def _validate_plan(value: Mapping[str, Any]) -> None:
    _assert_plain_unaliased_json(value)
    if type(value) is not dict or set(value) != {
        "schema_version",
        "status",
        "classification",
        "descriptor_binding",
        "candidate_order",
        "partition_counts",
        "structural_inputs",
        "blocker_ledger",
        "publisher_registry",
        "proposed_lineage",
        "historical_exclusions",
        "observation_and_execution_gaps",
        "future_authority_boundary",
        "readiness",
        "claims",
        "limitations",
        "plan_body_sha256",
    }:
        _fail("qualification-plan v3 artifact keys differ")
    plan = cast(dict[str, Any], value)
    structural = QualificationPlanV3StructuralInputs.from_mapping(
        cast(dict[str, bool], plan["structural_inputs"])
    )
    registry = plan["publisher_registry"]
    if type(registry) is not dict or type(registry.get("recognized_closures")) is not list:
        _fail("qualification-plan v3 publisher registry shape differs")
    closures = tuple(_closure_from_dict(item) for item in registry["recognized_closures"])
    lineage = plan["proposed_lineage"]
    if type(lineage) is not dict or type(lineage.get("build_lineage_components")) is not list:
        _fail("qualification-plan v3 proposed lineage shape differs")
    expected_body = _plan_body(
        structural_inputs=structural,
        publisher_closures=closures,
        proposed_image_id=lineage.get("image_id"),
        proposed_build_lineage_components=lineage["build_lineage_components"],
    )
    body = copy.deepcopy(plan)
    supplied_body_sha256 = _require_sha256(body.pop("plan_body_sha256"), "plan body")
    if not _exact_json_equal(body, expected_body):
        _fail("qualification-plan v3 content differs from its fail-closed reconstruction")
    observed_body_sha256 = hashlib.sha256(_canonical_json(body, newline=False)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, observed_body_sha256):
        _fail("qualification-plan v3 body digest differs")
    if any(value is not False for value in cast(dict[str, Any], plan["claims"]).values()):
        _fail("qualification-plan v3 authority claim became true")
    if any(value is not False for value in cast(dict[str, Any], plan["readiness"]).values()):
        _fail("qualification-plan v3 readiness became true")


def canonical_matched_v3_qualification_plan_v3_bytes(plan: Mapping[str, Any]) -> bytes:
    """Validate and canonically encode one nonauthorizing plan artifact."""

    _require_descriptor_pins()
    _validate_plan(plan)
    return _canonical_json(plan)


def replay_matched_v3_qualification_plan_v3(
    raw: bytes,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Replay canonical plan bytes under one exact full-file caller pin."""

    _require_descriptor_pins()
    expected = _require_sha256(expected_plan_sha256, "expected qualification plan")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected
    ):
        _fail("qualification-plan v3 full-file SHA-256 differs")
    plan = _strict_json(raw)
    _validate_plan(plan)
    return copy.deepcopy(plan)


def parse_matched_v3_qualification_plan_v3(
    raw: bytes,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Alias strict replay under the conventional parser name."""

    return replay_matched_v3_qualification_plan_v3(
        raw,
        expected_plan_sha256=expected_plan_sha256,
    )


def matched_v3_qualification_plan_v3_descriptor() -> dict[str, Any]:
    """Return detached descriptor content even while its pins are unfinished."""

    return copy.deepcopy(_descriptor())


def canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes() -> bytes:
    """Return compact sorted descriptor BODY bytes without a trailing LF."""

    return bytes(_DESCRIPTOR_BODY_BYTES)


def canonical_matched_v3_qualification_plan_v3_descriptor_bytes() -> bytes:
    """Return compact sorted descriptor FILE bytes with exactly one LF."""

    return bytes(_DESCRIPTOR_BYTES)


def matched_v3_qualification_plan_v3_descriptor_body_sha256() -> str:
    """Return the finalized descriptor BODY identity, failing while zero-pinned."""

    body_pin, _file_pin = _require_descriptor_pins()
    return body_pin


def matched_v3_qualification_plan_v3_descriptor_sha256() -> str:
    """Return the finalized descriptor FILE identity, failing while zero-pinned."""

    _body_pin, file_pin = _require_descriptor_pins()
    return file_pin


def parse_matched_v3_qualification_plan_v3_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen v3 descriptor."""

    body_pin, file_pin = _require_descriptor_pins()
    value = _strict_json(raw)
    if (
        not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            file_pin,
        )
        or not hmac.compare_digest(
            hashlib.sha256(raw[:-1]).hexdigest(),
            body_pin,
        )
        or not _exact_json_equal(value, _descriptor())
        or raw != _DESCRIPTOR_BYTES
    ):
        _fail("qualification-plan v3 descriptor differs")
    return copy.deepcopy(value)


__all__ = [
    "ForagerMatchedV3QualificationPlanV3Error",
    "HISTORICAL_IMAGE_IDS",
    "HISTORICAL_ONE_SHOT_BUILD_LINEAGE_COMPONENTS",
    "MATCHED_V3_ADAPTER_CANDIDATE_IDS",
    "MATCHED_V3_EXTERNAL_CANDIDATE_IDS",
    "MATCHED_V3_HORIZON",
    "MATCHED_V3_LOCAL_CANDIDATE_IDS",
    "MATCHED_V3_QUALIFICATION_V3_CANDIDATE_IDS",
    "PINNED_EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256",
    "PINNED_EXTERNAL_EXECUTION_CONTRACT_SOURCE_SHA256",
    "PINNED_EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256",
    "PINNED_EXTERNAL_EXECUTION_RUNNER_SOURCE_SHA256",
    "PINNED_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256",
    "PINNED_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256",
    "PINNED_EXTERNAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "PINNED_EXTERNAL_REWARD_PUBLICATION_SOURCE_SHA256",
    "PINNED_LOCAL_REWARD_BUNDLE_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_REWARD_BUNDLE_SOURCE_SHA256",
    "PINNED_LOCAL_REWARD_PUBLICATION_DESCRIPTOR_SHA256",
    "PINNED_LOCAL_REWARD_PUBLICATION_SOURCE_SHA256",
    "PublisherClosureIdentityV3",
    "PublisherComponentIdentityV3",
    "QUALIFICATION_PLAN_V3_CLASSIFICATION",
    "QUALIFICATION_PLAN_V3_DESCRIPTOR_BODY_SHA256",
    "QUALIFICATION_PLAN_V3_DESCRIPTOR_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V3_DESCRIPTOR_SHA256",
    "QUALIFICATION_PLAN_V3_SCHEMA_VERSION",
    "QUALIFICATION_PLAN_V3_STATUS",
    "QualificationPlanV3StructuralInputs",
    "STRUCTURAL_GAP_IDS",
    "build_matched_v3_qualification_plan_v3",
    "canonical_matched_v3_qualification_plan_v3_bytes",
    "canonical_matched_v3_qualification_plan_v3_descriptor_body_bytes",
    "canonical_matched_v3_qualification_plan_v3_descriptor_bytes",
    "matched_v3_external_publisher_closure_v3",
    "matched_v3_local_publisher_closure_v3",
    "matched_v3_qualification_plan_v3_descriptor",
    "matched_v3_qualification_plan_v3_descriptor_body_sha256",
    "matched_v3_qualification_plan_v3_descriptor_sha256",
    "parse_matched_v3_qualification_plan_v3",
    "parse_matched_v3_qualification_plan_v3_descriptor",
    "replay_matched_v3_qualification_plan_v3",
    "synthetically_complete_matched_v3_structural_inputs_v3",
]
