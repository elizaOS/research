"""Descriptor-only external execution contract for matched Forager v3.

This module freezes the twelve external candidate command and artifact layouts
without executing them.  It does not import the historical configuration-plan
module, reinterpret that plan's superseded v1 materializer binding, issue seed
material, open a workload, expose a filesystem capability, inspect a result, or
accept any scientific claim.  The current materializer-v2, seed-transport, and
result-bridge identities are an additive content overlay only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import (
    _forager_matched_v3_external_result_bridge as _result_bridge,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as _materializer,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_seed_transport as _seed_transport,
)

EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_contract_descriptor.v1"
)
EXTERNAL_EXECUTION_CONTRACT_STATUS: Final = (
    "implemented_descriptor_only_unexecuted_unqualified_non_authorizing"
)
EXTERNAL_EXECUTION_CONTRACT_CLASSIFICATION: Final = (
    "historical_plan_v1_with_additive_v2_materialization_overlay"
)

_MAX_DESCRIPTOR_BYTES: Final = 2 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_TEXT: Final = 16 * 1024
_MAX_JSON_INTEGER_DIGITS: Final = 19

_CONFIGURATION_PLAN_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_configuration_plan.v1"
_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
_CONFIGURATION_PLAN_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_configuration_plan.py"
)
_CONFIGURATION_PLAN_SOURCE_SHA256: Final = (
    "ad711eaa61511c6b1d43b86b867e09ba70f7124d5d67966b22d1f7ef3a556a84"
)
_HISTORICAL_MATERIALIZER_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization.v1"
)
_HISTORICAL_MATERIALIZER_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization_identity.v1"
)
_HISTORICAL_MATERIALIZER_IDENTITY_SHA256: Final = (
    "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
)
_HISTORICAL_MATERIALIZER_SOURCE_SHA256: Final = (
    "5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"
)

_MATERIALIZER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_external_materialization.v2"
_MATERIALIZER_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization_identity.v2"
)
_MATERIALIZER_IDENTITY_SHA256: Final = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
_MATERIALIZER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py"
)
_MATERIALIZER_SOURCE_SHA256: Final = (
    "3ff59a9f88d79b122fa66a1cdca009a68ff524806a7a7c58e5d565cd30ecaafe"
)

_SEED_TRANSPORT_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_external_seed_transport.v1"
_SEED_TRANSPORT_DESCRIPTOR_SHA256: Final = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)
_SEED_TRANSPORT_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_seed_transport.py"
)
_SEED_TRANSPORT_SOURCE_SHA256: Final = (
    "18f24a5116ae927c903b23a5cc64b1628aa135c808ccc985bbf2060e831d66f0"
)
_DERIVED_SOURCE_SHA256_BY_PATH: Final = {
    "src/continuing_main.py": ("ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"),
    "src/problems/BaseProblem.py": (
        "a4ab77408c1bb38dd3f4e72d830765176c38bba4b73b69fe296765a0272d87dc"
    ),
    "src/problems/Foragax.py": ("ff6e875511fcc574bafde7f114382dccf5303dba96f4154d5abbc16744d8e7c9"),
    "src/rtu_ppo.py": ("1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"),
}

_RESULT_BRIDGE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_result_bridge_descriptor.v1"
)
_RESULT_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
)
_RESULT_BRIDGE_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_external_result_bridge.py"
)
_RESULT_BRIDGE_SOURCE_SHA256: Final = (
    "c1859f0cfb7862e22c470f89ad9d3298a76b1fb419bf1431069f286f593e22f7"
)

_HORIZON: Final = 499_712
_PPO_ROLLOUT_STEPS: Final = 2_048
_PPO_ROLLOUT_COUNT: Final = 244
_CONTINUING_ENTRYPOINT: Final = "src/continuing_main.py"
_PPO_ENTRYPOINT: Final = "src/rtu_ppo.py"
_ENVIRONMENT_SEED_PLACEHOLDER: Final = "<environment_seed_uint31>"
_AGENT_SEED_PLACEHOLDER: Final = "<candidate_private_agent_seed_uint31>"
_SAVE_BASE_PLACEHOLDER: Final = "<fresh_candidate_private_save_base>"
_CHECKPOINT_ROOT_PLACEHOLDER: Final = "<new_empty_candidate_private_checkpoint_root>"
_STAGED_CHECKOUT_ROOT_PLACEHOLDER: Final = "<staged_materialized_checkout_root>"
_PPO_VIDEO_RELATIVE_PATH: Final = "videos/0/497664_499712-episode-0.mp4"


class ForagerMatchedV3ExternalExecutionContractError(ValueError):
    """The frozen descriptor, record identity, or live dependency binding drifted."""


# Tuple fields: candidate ID, original path, original SHA-256, derived SHA-256,
# top-level agent/output stem, family, and external reward NPY descriptor.
_CANDIDATE_SPECS: Final = (
    (
        "external_dqn_plain",
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json",
        "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
        "1d8a711ee1e4db575cb0edcacbaf38f97bd06cddc24019eb64b8c410e84b4e85",
        "DQN",
        "continuing",
        "<f2",
    ),
    (
        "external_dqn_crelu",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_CReLU.json",
        "d433b87789e180df3f153cebdafa53f3b6278325fcd32889c8959552cecfeda0",
        "ef92352b97d92e7d40458db48157f589b0d0984f2f4286947c9a1f28bd522892",
        "DQN_CReLU",
        "continuing",
        "<f2",
    ),
    (
        "external_dqn_redo",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_ReDo_PostLNScore.json"),
        "61fa39de8426e2fb78305846b26f6c7a977c72b9cc8a61fc70419f8c15afc8ab",
        "c38288f2ddb6a5dd8892954b499370d04399ec41e966fe790643c9d64b5ffc54",
        "DQN_ReDo_PostLNScore",
        "continuing",
        "<f2",
    ),
    (
        "external_dqn_reward_trace",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_reward_trace.json"),
        "3d14f03bc22eec14e4abcc32e635c1dbfa83d4149ef2eaca3609ddba3281ffcb",
        "8641a3b4673940f5519f074b617ccc58a6c14b61a8b448df434cebb3d5f4c974",
        "DQN_reward_trace",
        "continuing",
        "<f2",
    ),
    (
        "external_dqn_l2_init",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_L2_Init.json"),
        "6a90d4e970c66d0cc968c9988e0e91a3341fdcb2126954a1b7314f7154b53934",
        "2a2a1dc503b0617c35c202027a646db32186e2668d4b8988215f516a036b9107",
        "DQN_L2_Init",
        "continuing",
        "<f2",
    ),
    (
        "external_pt_dqn_xfinal",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json"),
        "4f2ff117d4b82458e3a4bb373d54d03d5b1fedeb4d0b25214235facb5ff2b690",
        "05eaad6da93d8c42d8bd60da3d6c3728bca5c653608eb98210a48a76bedce2e2",
        "PT_DQN_64",
        "continuing",
        "<f2",
    ),
    (
        "external_drqn_xfinal",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "70a5ee902aa6128ec65c6d4fd33e27da0e3eaa02bd4ea8b776baf3fa158c27de",
        "2b0e177420a9f9a4c8a7bd7aede9c7d2c5add3da4c8b3e301f32bb2588637047",
        "DRQN",
        "continuing",
        "<f2",
    ),
    (
        "isolated_ppo_generic",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/ActorCriticMLP.json"),
        "c8915481c67045339de4b013372d2538eafa91b21c639d2fb0e08d0c60865228",
        "27ffdffcf3ff3e722be5cdfe58d6bc07348ebe5380478032eedfaf435b754c71",
        "ActorCriticMLP",
        "ppo",
        "<f4",
    ),
    (
        "isolated_rtu_paper_scale",
        (
            "experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json"
        ),
        "b9e7bf1bfa307239df848677b6ad4e7c76ef316567b11f75e9455625efc20e65",
        "c32e240bf8c78cf2c7d1ad958bbfc8975b55160fb09490401763a346c2a21090",
        "PPO-RTU_LN_2048",
        "ppo",
        "<f4",
    ),
    (
        "random_policy",
        ("experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Random.json"),
        "24b9d17d2fa4d5da0dc9afd24bbd605fdd4e7574a70f13dc9648e6e6412f6a9a",
        "d20dc9294baab331c4658e4c682d5e1eee3c6f7cc6baf5d17586f48362e8936d",
        "Random",
        "continuing",
        "<f2",
    ),
    (
        "search_nearest",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Nearest.json"
        ),
        "2c2f67b13f818c7a639411e491095f04dbf3e789a1197c40a6a659ef26e0238d",
        "97b644c4c625155ae16fa7b69432ea0774f767142cc0e28b3d6fcec18c17d2ab",
        "Search-Nearest",
        "continuing",
        "<f2",
    ),
    (
        "search_oracle",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Oracle.json"
        ),
        "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89",
        "Search-Oracle",
        "continuing",
        "<f2",
    ),
)

EXTERNAL_EXECUTION_CANDIDATE_IDS: Final = tuple(spec[0] for spec in _CANDIDATE_SPECS)


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3ExternalExecutionContractError(
        f"descriptor contains forbidden constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3ExternalExecutionContractError(
        f"descriptor contains forbidden float {value!r}"
    )


def _parse_bounded_json_int(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_JSON_INTEGER_DIGITS:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3ExternalExecutionContractError(
                f"descriptor contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3ExternalExecutionContractError(
                "descriptor exceeds its JSON node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3ExternalExecutionContractError(
                "descriptor exceeds its JSON depth bound"
            )
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3ExternalExecutionContractError(
                    "descriptor must be an unaliased acyclic JSON tree"
                )
            seen.add(identity)
            mapping = item
            for key, child in mapping.items():
                if type(key) is not str or len(key) > _MAX_JSON_TEXT:
                    raise ForagerMatchedV3ExternalExecutionContractError(
                        "descriptor keys must be bounded exact strings"
                    )
                pending.append((child, depth + 1))
        elif type(item) is list:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3ExternalExecutionContractError(
                    "descriptor must be an unaliased acyclic JSON tree"
                )
            seen.add(identity)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > _MAX_JSON_TEXT:
                raise ForagerMatchedV3ExternalExecutionContractError(
                    "descriptor string exceeds its bound"
                )
        elif item is not None and type(item) not in {bool, int}:
            raise ForagerMatchedV3ExternalExecutionContractError(
                "descriptor contains a non-plain JSON scalar"
            )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor canonical root must be a plain object"
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
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor is not finite canonical ASCII JSON"
        ) from exc
    if len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor exceeds its canonical byte bound"
        )
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_DESCRIPTOR_BYTES:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor input must be bounded exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor must have one canonical trailing newline"
        )
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_bounded_json_int,
        )
    except ForagerMatchedV3ExternalExecutionContractError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor is not bounded strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "descriptor is not in exact canonical form"
        )
    return result


def _bounded_source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise ForagerMatchedV3ExternalExecutionContractError(
            f"dependency source path differs from {expected_suffix}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "dependency source verification requires O_NOFOLLOW"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionContractError(
            f"dependency source cannot be opened safely: {expected_suffix}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_SOURCE_BYTES
        ):
            raise ForagerMatchedV3ExternalExecutionContractError(
                f"dependency source is not one bounded single-link file: {expected_suffix}"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3ExternalExecutionContractError(
                    f"dependency source was truncated while reading: {expected_suffix}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3ExternalExecutionContractError(
                f"dependency source grew while reading: {expected_suffix}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_before != stable_after:
        raise ForagerMatchedV3ExternalExecutionContractError(
            f"dependency source changed while reading: {expected_suffix}"
        )
    return digest.hexdigest()


def _configuration_plan_source_path() -> str:
    materializer_file = _materializer.__file__
    if type(materializer_file) is not str:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "materializer source path is unavailable"
        )
    return os.path.join(
        os.path.dirname(materializer_file),
        "forager_matched_v3_configuration_plan.py",
    )


def _verify_live_dependency_bindings() -> None:
    source_bindings = (
        (
            "historical configuration plan",
            _configuration_plan_source_path(),
            _CONFIGURATION_PLAN_SOURCE_PATH,
            _CONFIGURATION_PLAN_SOURCE_SHA256,
        ),
        (
            "materializer v2",
            _materializer.__file__,
            _MATERIALIZER_SOURCE_PATH,
            _MATERIALIZER_SOURCE_SHA256,
        ),
        (
            "external seed transport",
            _seed_transport.__file__,
            _SEED_TRANSPORT_SOURCE_PATH,
            _SEED_TRANSPORT_SOURCE_SHA256,
        ),
        (
            "external result bridge",
            _result_bridge.__file__,
            _RESULT_BRIDGE_SOURCE_PATH,
            _RESULT_BRIDGE_SOURCE_SHA256,
        ),
    )
    for label, module_file, source_path, expected_sha256 in source_bindings:
        actual_sha256 = _bounded_source_sha256(module_file, source_path)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise ForagerMatchedV3ExternalExecutionContractError(f"{label} source binding drifted")

    if (
        _materializer.EXTERNAL_MATERIALIZATION_SCHEMA_VERSION != _MATERIALIZER_SCHEMA_VERSION
        or _materializer.EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION
        != _MATERIALIZER_IDENTITY_SCHEMA_VERSION
        or _materializer.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
        != _MATERIALIZER_IDENTITY_SHA256
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "materializer-v2 identity binding drifted"
        )
    try:
        materializer_identity = _materializer.canonical_pinned_external_checkout_identity_bytes()
    except (AssertionError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "materializer-v2 identity binding drifted"
        ) from exc
    if not hmac.compare_digest(
        hashlib.sha256(materializer_identity).hexdigest(),
        _MATERIALIZER_IDENTITY_SHA256,
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "materializer-v2 identity binding drifted"
        )

    if (
        _seed_transport.SCHEMA_VERSION != _SEED_TRANSPORT_SCHEMA_VERSION
        or _seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
        != _SEED_TRANSPORT_DESCRIPTOR_SHA256
        or tuple(_seed_transport.SOURCE_PATHS) != tuple(_DERIVED_SOURCE_SHA256_BY_PATH)
        or dict(_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH)
        != _DERIVED_SOURCE_SHA256_BY_PATH
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external seed-transport binding drifted"
        )
    try:
        seed_descriptor = (
            _seed_transport.canonical_matched_v3_external_seed_transport_descriptor_bytes()
        )
        _seed_transport.parse_matched_v3_external_seed_transport_descriptor(seed_descriptor)
    except (AssertionError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external seed-transport binding drifted"
        ) from exc
    if not hmac.compare_digest(
        hashlib.sha256(seed_descriptor).hexdigest(),
        _SEED_TRANSPORT_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external seed-transport binding drifted"
        )

    expected_formats = {
        candidate_id: (family, npy_descr)
        for candidate_id, _path, _original, _derived, _stem, family, npy_descr in (_CANDIDATE_SPECS)
    }
    if (
        _result_bridge.EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION
        != _RESULT_BRIDGE_SCHEMA_VERSION
        or _result_bridge.EXTERNAL_RESULT_BRIDGE_DESCRIPTOR_SHA256
        != _RESULT_BRIDGE_DESCRIPTOR_SHA256
        or tuple(_result_bridge.EXTERNAL_RESULT_CANDIDATE_IDS)
        != EXTERNAL_EXECUTION_CANDIDATE_IDS
        or dict(_result_bridge.EXTERNAL_RESULT_CANDIDATE_FORMATS) != expected_formats
        or _result_bridge.MATCHED_V3_REWARD_HORIZON != _HORIZON
        or _result_bridge.CANONICAL_SCORER_NPZ_SIZE_BYTES != 499_980
        or _result_bridge.MAX_EXTERNAL_NPZ_BYTES != 64 * 1024 * 1024
        or _result_bridge.MAX_ZIP_MEMBER_COUNT != 128
        or _result_bridge.MAX_ZIP_TOTAL_COMPRESSED_BYTES != 64 * 1024 * 1024
        or _result_bridge.MAX_ZIP_TOTAL_EXPANDED_BYTES != 64 * 1024 * 1024
        or _result_bridge.MAX_NPY_HEADER_BYTES != 4 * 1024
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external result-bridge binding drifted"
        )
    try:
        bridge_descriptor = _result_bridge.canonical_external_result_bridge_descriptor_bytes()
        _result_bridge.parse_external_result_bridge_descriptor(bridge_descriptor)
    except (AssertionError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external result-bridge binding drifted"
        ) from exc
    if not hmac.compare_digest(
        hashlib.sha256(bridge_descriptor).hexdigest(),
        _RESULT_BRIDGE_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external result-bridge binding drifted"
        )


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority": False,
        "artifact_set_accepted": False,
        "candidate_qualified": False,
        "execution_authorized": False,
        "execution_ready": False,
        "live_execution_completed": False,
        "materialization_accepted": False,
        "performance_claim_allowed": False,
        "qualification_authority": False,
        "result_accepted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "source_closure_qualified": False,
        "universal_sota_claim_allowed": False,
        "workload_executed": False,
    }


def _apis() -> dict[str, bool]:
    return {
        "capability_acceptance_exposed": False,
        "capability_issuance_exposed": False,
        "executor_exposed": False,
        "filesystem_capability_exposed": False,
        "filesystem_mutation_exposed": False,
        "result_acceptance_exposed": False,
        "result_loader_exposed": False,
        "seed_acceptance_exposed": False,
        "seed_issuer_exposed": False,
        "subprocess_exposed": False,
        "workload_exposed": False,
    }


def _candidate_claims() -> dict[str, bool]:
    return {
        "candidate_qualified": False,
        "execution_authorized": False,
        "execution_ready": False,
        "result_accepted": False,
        "runtime_qualified": False,
        "scientific_promotion_allowed": False,
    }


def _candidate_record(spec: tuple[str, str, str, str, str, str, str]) -> dict[str, Any]:
    (
        candidate_id,
        original_path,
        original_sha256,
        derived_sha256,
        output_stem,
        family,
        npy_descr,
    ) = spec
    is_ppo = family == "ppo"
    entrypoint = _PPO_ENTRYPOINT if is_ppo else _CONTINUING_ENTRYPOINT
    max_steps = _PPO_ROLLOUT_COUNT if is_ppo else _HORIZON
    configuration_suffix = original_path.removeprefix("experiments/")
    configuration_directory = configuration_suffix.rsplit("/", 1)[0]
    result_directory = f"results/{configuration_directory}/{output_stem}"
    result_npz = f"{result_directory}/data/0.npz"
    results_database = f"{result_directory}/results.db"
    video = f"{result_directory}/{_PPO_VIDEO_RELATIVE_PATH}" if is_ppo else None
    artifacts = [
        {"artifact_kind": "external_reward_npz", "path": result_npz},
        {"artifact_kind": "sibling_results_database", "path": results_database},
    ]
    if video is not None:
        artifacts.append({"artifact_kind": "ppo_video", "path": video})
    return {
        "candidate_id": candidate_id,
        "configuration": {
            "original_relative_path": original_path,
            "original_sha256": original_sha256,
            "derived_sha256": derived_sha256,
            "output_stem": output_stem,
            "derived_configuration_staging_relative_path": original_path,
            "staging_preserves_original_path_below_experiments": True,
        },
        "execution": {
            "family": family,
            "npy_descr": npy_descr,
            "entrypoint_path": entrypoint,
            "entrypoint_sha256": _DERIVED_SOURCE_SHA256_BY_PATH[entrypoint],
            "working_directory_placeholder": _STAGED_CHECKOUT_ROOT_PLACEHOLDER,
            "working_directory_is_staged_materialized_checkout_root": True,
            "index": 0,
            "environment_seed_placeholder": _ENVIRONMENT_SEED_PLACEHOLDER,
            "agent_seed_placeholder": _AGENT_SEED_PLACEHOLDER,
            "max_steps": max_steps,
            "interaction_horizon": _HORIZON,
            "ppo_rollout_steps": _PPO_ROLLOUT_STEPS if is_ppo else None,
            "ppo_rollout_count": _PPO_ROLLOUT_COUNT if is_ppo else None,
            "argv": [
                "--exp",
                original_path,
                "--idxs",
                "0",
                "--environment_seed",
                _ENVIRONMENT_SEED_PLACEHOLDER,
                "--agent_seed",
                _AGENT_SEED_PLACEHOLDER,
                "--max_steps",
                str(max_steps),
                "--save_path",
                _SAVE_BASE_PLACEHOLDER,
                "--checkpoint_path",
                _CHECKPOINT_ROOT_PLACEHOLDER,
                "--silent",
            ],
        },
        "root_contract": {
            "save_base_placeholder": _SAVE_BASE_PLACEHOLDER,
            "save_base_candidate_private": True,
            "save_base_fresh_empty_before_execution_required": True,
            "save_base_is_distinct_from_derived_result_directory": True,
            "derived_result_directory_relative_to_save_base": result_directory,
            "doubled_results_component_forbidden": True,
            "checkpoint_root_placeholder": _CHECKPOINT_ROOT_PLACEHOLDER,
            "checkpoint_root_candidate_private": True,
            "checkpoint_root_fresh_empty_before_execution_required": True,
            "checkpoint_root_empty_after_execution_required": True,
            "checkpoint_root_exact_final_entries": [],
        },
        "artifact_contract": {
            "result_directory": result_directory,
            "paths_are_relative_to_candidate_private_save_base": True,
            "result_npz_path": result_npz,
            "results_database_path": results_database,
            "results_database_is_sibling_of_data_directory": True,
            "ppo_video_relative_to_result_directory": (
                _PPO_VIDEO_RELATIVE_PATH if is_ppo else None
            ),
            "ppo_video_path": video,
            "exact_files": artifacts,
            "missing_files_allowed": False,
            "extra_files_allowed": False,
        },
        "claims": _candidate_claims(),
    }


def _dependencies() -> dict[str, Any]:
    return {
        "historical_configuration_plan_v1": {
            "schema_version": _CONFIGURATION_PLAN_SCHEMA_VERSION,
            "descriptor_sha256": _CONFIGURATION_PLAN_SHA256,
            "current_source_path": _CONFIGURATION_PLAN_SOURCE_PATH,
            "current_source_sha256": _CONFIGURATION_PLAN_SOURCE_SHA256,
            "status_for_materialization": "historical_superseded",
            "imported_or_reconstructed_here": False,
            "silently_upgraded_to_materializer_v2": False,
            "historical_materializer_binding": {
                "manifest_schema_version": _HISTORICAL_MATERIALIZER_SCHEMA_VERSION,
                "identity_schema_version": (_HISTORICAL_MATERIALIZER_IDENTITY_SCHEMA_VERSION),
                "identity_sha256": _HISTORICAL_MATERIALIZER_IDENTITY_SHA256,
                "source_sha256": _HISTORICAL_MATERIALIZER_SOURCE_SHA256,
                "selected_for_materialization": False,
                "superseded": True,
            },
        },
        "materializer_v2_overlay": {
            "manifest_schema_version": _MATERIALIZER_SCHEMA_VERSION,
            "identity_schema_version": _MATERIALIZER_IDENTITY_SCHEMA_VERSION,
            "identity_sha256": _MATERIALIZER_IDENTITY_SHA256,
            "source_path": _MATERIALIZER_SOURCE_PATH,
            "source_sha256": _MATERIALIZER_SOURCE_SHA256,
            "relationship_to_configuration_plan_v1": "separate_additive_overlay",
            "materialization_performed": False,
            "production_manifest_accepted": False,
        },
        "external_seed_transport": {
            "schema_version": _SEED_TRANSPORT_SCHEMA_VERSION,
            "descriptor_sha256": _SEED_TRANSPORT_DESCRIPTOR_SHA256,
            "source_path": _SEED_TRANSPORT_SOURCE_PATH,
            "source_sha256": _SEED_TRANSPORT_SOURCE_SHA256,
            "derived_entrypoint_sha256_by_path": {
                _CONTINUING_ENTRYPOINT: _DERIVED_SOURCE_SHA256_BY_PATH[_CONTINUING_ENTRYPOINT],
                _PPO_ENTRYPOINT: _DERIVED_SOURCE_SHA256_BY_PATH[_PPO_ENTRYPOINT],
            },
        },
        "external_result_bridge": {
            "schema_version": _RESULT_BRIDGE_SCHEMA_VERSION,
            "descriptor_sha256": _RESULT_BRIDGE_DESCRIPTOR_SHA256,
            "source_path": _RESULT_BRIDGE_SOURCE_PATH,
            "source_sha256": _RESULT_BRIDGE_SOURCE_SHA256,
            "bounds": {
                "reward_horizon": _HORIZON,
                "canonical_scorer_npz_size_bytes": 499_980,
                "maximum_external_npz_bytes": 64 * 1024 * 1024,
                "maximum_zip_member_count": 128,
                "maximum_zip_total_compressed_bytes": 64 * 1024 * 1024,
                "maximum_zip_total_expanded_bytes": 64 * 1024 * 1024,
                "maximum_npy_header_bytes": 4 * 1024,
            },
        },
    }


def _descriptor() -> dict[str, Any]:
    candidates = [_candidate_record(spec) for spec in _CANDIDATE_SPECS]
    return {
        "schema_version": EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": EXTERNAL_EXECUTION_CONTRACT_STATUS,
        "classification": EXTERNAL_EXECUTION_CONTRACT_CLASSIFICATION,
        "dependencies": _dependencies(),
        "candidate_count": len(candidates),
        "candidate_order": list(EXTERNAL_EXECUTION_CANDIDATE_IDS),
        "candidates": candidates,
        "workload_contract": {
            "interaction_horizon": _HORIZON,
            "continuing_max_steps": _HORIZON,
            "ppo_max_steps": _PPO_ROLLOUT_COUNT,
            "ppo_rollout_steps": _PPO_ROLLOUT_STEPS,
            "ppo_rollout_count": _PPO_ROLLOUT_COUNT,
            "ppo_interaction_count": _PPO_ROLLOUT_STEPS * _PPO_ROLLOUT_COUNT,
            "exactly_one_index": 0,
            "all_commands_end_with_silent": True,
            "all_save_bases_fresh_and_candidate_private": True,
            "save_path_is_base_not_derived_result_directory": True,
            "doubled_results_component_forbidden": True,
            "all_checkpoint_roots_fresh_candidate_private_and_finally_empty": True,
        },
        "artifact_inventory_policy": {
            "path_rule": (
                "<candidate-private save base>/results/"
                "<configuration directory after experiments/>/"
                "<top-level agent output stem>"
            ),
            "exact_results_prefix_count": 1,
            "doubled_results_component_forbidden": True,
            "derived_configuration_must_retain_original_relative_path": True,
            "reward_path_relative_to_result_directory": "data/0.npz",
            "database_path_relative_to_result_directory": "results.db",
            "ppo_only_video_path_relative_to_result_directory": (_PPO_VIDEO_RELATIVE_PATH),
            "ppo_video_basis": (
                "single_index_forces_allocate_frames_and_final_full_2048_step_rollout"
            ),
            "missing_candidates_allowed": False,
            "extra_candidates_allowed": False,
            "missing_artifacts_allowed": False,
            "extra_artifacts_allowed": False,
        },
        "canonicalization": {
            "format": "sorted_compact_ascii_json_with_one_newline",
            "allow_nan": False,
            "floats_allowed": False,
            "duplicate_keys_rejected": True,
            "exact_scalar_types_required": True,
            "container_aliases_rejected": True,
            "candidate_order_is_load_bearing": True,
            "maximum_bytes": _MAX_DESCRIPTOR_BYTES,
            "maximum_depth": _MAX_JSON_DEPTH,
            "maximum_nodes": _MAX_JSON_NODES,
        },
        "apis": _apis(),
        "claims": _claims(),
        "limitations": [
            "This descriptor executes no workload and materializes no checkout.",
            "It supplies and accepts no seed, result, capability, or filesystem authority.",
            (
                "Configuration-plan v1 is historical and superseded for materialization; "
                "its frozen content identity is not rewritten or silently upgraded."
            ),
            (
                "Materializer v2, seed transport, and result bridge are content bindings "
                "only; none establishes runtime readiness or qualification."
            ),
            "All listed output paths are expected future artifacts, not observed files.",
            "A future independently authorized executor and validator remain required.",
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256,
):
    raise AssertionError("external execution-contract descriptor identity drifted")


def external_execution_contract_descriptor() -> dict[str, Any]:
    """Return a detached descriptor after validating every live dependency binding."""

    _verify_live_dependency_bindings()
    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_external_execution_contract_descriptor_bytes() -> bytes:
    """Return the exact frozen descriptor bytes after live dependency validation."""

    _verify_live_dependency_bindings()
    return _DESCRIPTOR_BYTES


def external_execution_contract_descriptor_sha256() -> str:
    """Return the exact frozen descriptor digest after live dependency validation."""

    _verify_live_dependency_bindings()
    return EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256


def parse_external_execution_contract_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical descriptor under the live dependency bindings."""

    _verify_live_dependency_bindings()
    value = _strict_json_load(raw)
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256,
    ):
        raise ForagerMatchedV3ExternalExecutionContractError(
            "external execution-contract descriptor identity drifted"
        )
    return value


def external_execution_candidate_record(candidate_id: str) -> dict[str, Any]:
    """Return one detached exact record; no execution or result authority is granted."""

    _verify_live_dependency_bindings()
    if type(candidate_id) is not str:
        raise ForagerMatchedV3ExternalExecutionContractError("candidate_id must be an exact string")
    snapshot = _strict_json_load(_DESCRIPTOR_BYTES)
    records = cast(list[dict[str, Any]], snapshot["candidates"])
    for record in records:
        record_candidate_id = record["candidate_id"]
        if type(record_candidate_id) is str and hmac.compare_digest(
            candidate_id, record_candidate_id
        ):
            return _strict_json_load(_canonical_json(record))
    raise ForagerMatchedV3ExternalExecutionContractError(
        f"unknown external candidate {candidate_id!r}"
    )


__all__ = [
    "EXTERNAL_EXECUTION_CANDIDATE_IDS",
    "EXTERNAL_EXECUTION_CONTRACT_CLASSIFICATION",
    "EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256",
    "EXTERNAL_EXECUTION_CONTRACT_STATUS",
    "ForagerMatchedV3ExternalExecutionContractError",
    "canonical_external_execution_contract_descriptor_bytes",
    "external_execution_candidate_record",
    "external_execution_contract_descriptor",
    "external_execution_contract_descriptor_sha256",
    "parse_external_execution_contract_descriptor",
]
