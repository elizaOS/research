"""Sealed, nonexecuting source staging for matched-v3 external candidates.

The public entrypoint consumes a live retained materializer-v2 directory
capability.  It never accepts a checkout path, seed, result, execution token,
or caller-authored overlay.  The parent verifies the frozen external execution
descriptor, derives all twelve configurations in memory from the exact retained
source bytes, and sends a bounded request to a fresh ``-I -S -B`` stdlib child.
The child independently rereads every manifested source member through the
inherited directory descriptor, verifies it, overlays the twelve derived
configurations, and writes one canonical POSIX USTAR stream.

The materializer manifest is absent from the workload root and retained byte for
byte below a dedicated attestation namespace.  A final staging manifest is also
an archive member.  To avoid a recursive hash identity, that manifest inventories
every *other* member and binds its own body digest; the retained bundle capability
binds the exact full-manifest and whole-archive digests out of band.

Nothing in this module imports or executes staged code, launches Docker, issues
randomness, reads a result, publishes an artifact, or grants execution,
acceptance, qualification, evidence, or promotion authority.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import selectors
import signal
import stat
import sys
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final, Never, NoReturn, SupportsIndex, cast

EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_contract_descriptor.v1"
)
EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_manifest.v1"
)
EXTERNAL_STAGING_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_private_request.v1"
)
EXTERNAL_STAGING_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_private_receipt.v1"
)
EXTERNAL_STAGING_STATUS: Final = (
    "implemented_sealed_source_bundle_unexecuted_unqualified_non_authorizing"
)

_EXECUTION_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_contract_descriptor.v1"
)
_EXECUTION_CONTRACT_SHA256: Final = (
    "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
)
_MATERIALIZER_SCHEMA_VERSION: Final = "alberta.forager_matched_v3_external_materialization.v2"
_MATERIALIZER_IDENTITY_SHA256: Final = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
_MATERIALIZER_MANIFEST_FILENAME: Final = (
    ".alberta-forager-matched-v3-external-materialization.v2.json"
)
_CONFIGURATION_TRANSFORM_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_configuration_transform.v1"
)
_CONFIGURATION_SOURCE_SUFFIX: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_configuration.py"
)
_CONFIGURATION_SOURCE_SHA256: Final = (
    "59aa86ea35a1f5065274329ae8ff54c5997e7642693d623701a7b269fb956904"
)

EXTERNAL_STAGING_ATTESTATION_NAMESPACE: Final = ".alberta-forager-matched-v3-attestations"
EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH: Final = (
    f"{EXTERNAL_STAGING_ATTESTATION_NAMESPACE}/external-materialization.v2.json"
)
EXTERNAL_STAGING_FINAL_MANIFEST_PATH: Final = (
    f"{EXTERNAL_STAGING_ATTESTATION_NAMESPACE}/external-staging-manifest.v1.json"
)

_MAX_SOURCE_MEMBERS: Final = 20_000
_MAX_ARCHIVE_MEMBERS: Final = _MAX_SOURCE_MEMBERS + 2
_MAX_MEMBER_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_PAYLOAD_BYTES: Final = 1024 * 1024 * 1024
_MAX_REQUEST_BYTES: Final = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 64 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_STDERR_BYTES: Final = 1024 * 1024
_MAX_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_TEXT_BYTES: Final = 64 * 1024 * 1024
_MAX_PATH_BYTES: Final = 255
_MAX_COMPONENT_BYTES: Final = 255
_MAX_PATH_COMPONENTS: Final = 256
_READ_CHUNK_BYTES: Final = 1024 * 1024
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10 * 1024
_WORKER_TIMEOUT_SECONDS: Final = 180.0
_WORKER_CLEANUP_GRACE_SECONDS: Final = 1.0

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_OCTAL_RE: Final = re.compile(rb"[0-7]+\Z")
_WORKER_ARGUMENT_RE: Final = re.compile(r"[1-9][0-9]*\Z")


class ForagerMatchedV3ExternalStagingError(RuntimeError):
    """The staging descriptor, retained source, overlay, or archive failed closed."""


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "artifact_accepted": False,
        "candidate_qualified": False,
        "execution_authority_granted": False,
        "execution_capability_granted": False,
        "execution_ready": False,
        "filesystem_publication_authority_granted": False,
        "materialization_accepted": False,
        "performance_claim_allowed": False,
        "qualification_authority_granted": False,
        "result_acceptance_authority_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "seed_authority_granted": False,
        "source_closure_qualified": False,
        "universal_sota_claim_allowed": False,
        "workload_executed": False,
    }


def _limitations() -> list[str]:
    return [
        "The bundle is a caller-retained source staging capability, not an execution capability.",
        "The isolated helper reads and archives bytes but never imports or executes staged code.",
        "No seed, result, runtime, image, dependency environment, or hardware is qualified here.",
        "The materializer archive identity remains provenance rather than verified archive input.",
        (
            "The final manifest inventories every non-self archive member; its exact own bytes "
            "and the complete USTAR identity are bound by the retained capability to avoid a "
            "recursive self-hash."
        ),
        "A valid descriptor, manifest, or bundle grants no execution or acceptance authority.",
    ]


_CANDIDATE_IDS: Final = (
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
_FROZEN_EXECUTION_RECORDS: Final = (
    (
        "external_dqn_plain",
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json",
        "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
        "1d8a711ee1e4db575cb0edcacbaf38f97bd06cddc24019eb64b8c410e84b4e85",
    ),
    (
        "external_dqn_crelu",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_CReLU.json",
        "d433b87789e180df3f153cebdafa53f3b6278325fcd32889c8959552cecfeda0",
        "ef92352b97d92e7d40458db48157f589b0d0984f2f4286947c9a1f28bd522892",
    ),
    (
        "external_dqn_redo",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_ReDo_PostLNScore.json"),
        "61fa39de8426e2fb78305846b26f6c7a977c72b9cc8a61fc70419f8c15afc8ab",
        "c38288f2ddb6a5dd8892954b499370d04399ec41e966fe790643c9d64b5ffc54",
    ),
    (
        "external_dqn_reward_trace",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_reward_trace.json"),
        "3d14f03bc22eec14e4abcc32e635c1dbfa83d4149ef2eaca3609ddba3281ffcb",
        "8641a3b4673940f5519f074b617ccc58a6c14b61a8b448df434cebb3d5f4c974",
    ),
    (
        "external_dqn_l2_init",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_L2_Init.json",
        "6a90d4e970c66d0cc968c9988e0e91a3341fdcb2126954a1b7314f7154b53934",
        "2a2a1dc503b0617c35c202027a646db32186e2668d4b8988215f516a036b9107",
    ),
    (
        "external_pt_dqn_xfinal",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json",
        "4f2ff117d4b82458e3a4bb373d54d03d5b1fedeb4d0b25214235facb5ff2b690",
        "05eaad6da93d8c42d8bd60da3d6c3728bca5c653608eb98210a48a76bedce2e2",
    ),
    (
        "external_drqn_xfinal",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "70a5ee902aa6128ec65c6d4fd33e27da0e3eaa02bd4ea8b776baf3fa158c27de",
        "2b0e177420a9f9a4c8a7bd7aede9c7d2c5add3da4c8b3e301f32bb2588637047",
    ),
    (
        "isolated_ppo_generic",
        ("experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/ActorCriticMLP.json"),
        "c8915481c67045339de4b013372d2538eafa91b21c639d2fb0e08d0c60865228",
        "27ffdffcf3ff3e722be5cdfe58d6bc07348ebe5380478032eedfaf435b754c71",
    ),
    (
        "isolated_rtu_paper_scale",
        (
            "experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json"
        ),
        "b9e7bf1bfa307239df848677b6ad4e7c76ef316567b11f75e9455625efc20e65",
        "c32e240bf8c78cf2c7d1ad958bbfc8975b55160fb09490401763a346c2a21090",
    ),
    (
        "random_policy",
        ("experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/Baselines/Random.json"),
        "24b9d17d2fa4d5da0dc9afd24bbd605fdd4e7574a70f13dc9648e6e6412f6a9a",
        "d20dc9294baab331c4658e4c682d5e1eee3c6f7cc6baf5d17586f48362e8936d",
    ),
    (
        "search_nearest",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Nearest.json"
        ),
        "2c2f67b13f818c7a639411e491095f04dbf3e789a1197c40a6a659ef26e0238d",
        "97b644c4c625155ae16fa7b69432ea0774f767142cc0e28b3d6fcec18c17d2ab",
    ),
    (
        "search_oracle",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Oracle.json"
        ),
        "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89",
    ),
)
_PINNED_PORTABLE_ALIAS_GROUPS: Final = (
    (
        (
            "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
            "metrics/NTKRank_LOP_vs_NoLOP.png"
        ),
        (
            "experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/"
            "metrics/ntkrank_LOP_vs_NoLOP.png"
        ),
    ),
)


def _transform(
    pointer: str,
    value_type: str,
    expected_original: str | int,
    replacement: str | int,
) -> dict[str, object]:
    return {
        "pointer": pointer,
        "value_type": value_type,
        "expected_original": expected_original,
        "replacement": replacement,
    }


def _transform_descriptor(*transforms: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": _CONFIGURATION_TRANSFORM_SCHEMA_VERSION,
        "transforms": list(transforms),
    }


_DQN_PLAIN_TRANSFORM: Final = _transform_descriptor(
    _transform("/metaParameters/experiment/ntk_freq", "integer", 2_500, 0),
    _transform("/metaParameters/experiment/x_ref_steps", "integer", 100, 0),
    _transform("/total_steps", "integer", 10_000, 499_712),
)
_XFINAL_TRANSFORM: Final = _transform_descriptor(
    _transform(
        "/metaParameters/environment/env_id",
        "string",
        "ForagaxSquareWaveTwoBiome-v11",
        "ForagaxTwoBiomeLarge-v1",
    ),
    _transform("/total_steps", "integer", 10_000_000, 499_712),
)
_RTU_PAPER_TRANSFORM: Final = _transform_descriptor(
    _transform(
        "/metaParameters/environment/env_id",
        "string",
        "ForagaxSquareWaveTwoBiome-v11",
        "ForagaxTwoBiomeLarge-v1",
    ),
    _transform("/metaParameters/experiment/ntk_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/weight_drift_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/weight_norm_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/x_ref_steps", "integer", 1_000, 0),
    _transform("/total_steps", "integer", 10_000_000, 499_712),
)
_RANDOM_TRANSFORM: Final = _transform_descriptor(
    _transform("/metaParameters/environment/aperture_size", "integer", 1, 9),
    _transform("/total_steps", "integer", 500_000, 499_712),
)
_DESCRIPTIVE_TRANSFORM: Final = _transform_descriptor(
    _transform("/total_steps", "integer", 500_000, 499_712),
)

_TRANSFORM_BY_CANDIDATE: Final = {
    "external_dqn_plain": _DQN_PLAIN_TRANSFORM,
    **{
        candidate_id: _XFINAL_TRANSFORM
        for candidate_id in (
            "external_dqn_crelu",
            "external_dqn_redo",
            "external_dqn_reward_trace",
            "external_dqn_l2_init",
            "external_pt_dqn_xfinal",
            "external_drqn_xfinal",
            "isolated_ppo_generic",
        )
    },
    "isolated_rtu_paper_scale": _RTU_PAPER_TRANSFORM,
    "random_policy": _RANDOM_TRANSFORM,
    "search_nearest": _DESCRIPTIVE_TRANSFORM,
    "search_oracle": _DESCRIPTIVE_TRANSFORM,
}

_TRANSFORM_SHA256_BY_KIND: Final = {
    "external_dqn_plain": "d85d2fec4fa18d3ab749f57a0a0b240daf57e05c3cd329bb08d17aac48b5ffeb",
    "xfinal": "fd20ddfef5fc160f14a0c47d2acd74335a361b061067fada88dd0ef1b42d1497",
    "isolated_rtu_paper_scale": (
        "68b904bed65ab157edbd323725126810d9fd72d7ccc69685a45eaa2aaba48f3b"
    ),
    "random_policy": "fcadac34348354a318950ab1761312064e12af56e8a7f51f2191fcd79e6890e4",
    "descriptive": "d9e02ef47a882a9769792a0367ac309d6f6ab43a6e077521a12c1e0fe098cb0e",
}


def _expected_transform_sha256(candidate_id: str) -> str:
    if candidate_id == "external_dqn_plain":
        return _TRANSFORM_SHA256_BY_KIND["external_dqn_plain"]
    if candidate_id in {
        "external_dqn_crelu",
        "external_dqn_redo",
        "external_dqn_reward_trace",
        "external_dqn_l2_init",
        "external_pt_dqn_xfinal",
        "external_drqn_xfinal",
        "isolated_ppo_generic",
    }:
        return _TRANSFORM_SHA256_BY_KIND["xfinal"]
    if candidate_id == "isolated_rtu_paper_scale":
        return _TRANSFORM_SHA256_BY_KIND["isolated_rtu_paper_scale"]
    if candidate_id == "random_policy":
        return _TRANSFORM_SHA256_BY_KIND["random_policy"]
    if candidate_id in {"search_nearest", "search_oracle"}:
        return _TRANSFORM_SHA256_BY_KIND["descriptive"]
    raise ForagerMatchedV3ExternalStagingError(
        f"unknown external staging candidate {candidate_id!r}"
    )


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3ExternalStagingError(
        f"staging JSON contains forbidden constant {value!r}"
    )


def _raise_json_float(value: str) -> NoReturn:
    raise ForagerMatchedV3ExternalStagingError(f"staging JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3ExternalStagingError("staging JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3ExternalStagingError(
                f"staging JSON contains duplicate key {key!r}"
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
            raise ForagerMatchedV3ExternalStagingError("staging JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3ExternalStagingError("staging JSON exceeds its depth bound")
        if type(item) in {dict, list}:
            identity = id(item)
            if identity in seen:
                raise ForagerMatchedV3ExternalStagingError(
                    "staging JSON must be an unaliased acyclic tree"
                )
            seen.add(identity)
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ForagerMatchedV3ExternalStagingError(
                            "staging JSON keys must be exact strings"
                        )
                    pending.append((child, depth + 1))
            else:
                pending.extend((child, depth + 1) for child in cast(list[Any], item))
        elif type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3ExternalStagingError(
                    "staging JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_TEXT_BYTES:
                raise ForagerMatchedV3ExternalStagingError(
                    "staging JSON string exceeds its byte bound"
                )
        elif item is not None and type(item) not in {bool, int}:
            raise ForagerMatchedV3ExternalStagingError("staging JSON contains a non-plain scalar")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3ExternalStagingError(
            "staging canonical JSON root must be a plain object"
        )
    _assert_plain_unaliased_json(value)
    try:
        return (
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
        raise ForagerMatchedV3ExternalStagingError(
            "staging value is not canonical ASCII JSON"
        ) from exc


def _canonical_base_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalStagingError(
            "base manifest is not canonical ASCII JSON"
        ) from exc


def _strict_json_load(
    raw: bytes,
    *,
    maximum_bytes: int,
    trailing_newline: bool,
) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3ExternalStagingError("staging JSON input must be bounded exact bytes")
    if trailing_newline and (not raw.endswith(b"\n") or raw.endswith(b"\n\n")):
        raise ForagerMatchedV3ExternalStagingError(
            "staging JSON must have one canonical trailing newline"
        )
    if not trailing_newline and raw.endswith(b"\n"):
        raise ForagerMatchedV3ExternalStagingError("base manifest must not have a trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMatchedV3ExternalStagingError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalStagingError("staging JSON is not strict ASCII JSON") from exc
    if type(value) is not dict:
        raise ForagerMatchedV3ExternalStagingError("staging JSON root must be a plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    canonical = _canonical_json(result) if trailing_newline else _canonical_base_json(result)
    if not hmac.compare_digest(canonical, raw):
        raise ForagerMatchedV3ExternalStagingError("staging JSON is not in exact canonical form")
    return result


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ForagerMatchedV3ExternalStagingError(f"{label} must be one lowercase SHA-256")
    return value


def _require_exact_int(value: Any, label: str, *, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ForagerMatchedV3ExternalStagingError(
            f"{label} must be a bounded exact nonnegative integer"
        )
    return value


def _portable_path_key(path: str) -> str:
    return "/".join(
        unicodedata.normalize("NFKC", part).casefold() for part in PurePosixPath(path).parts
    )


def _validate_relative_path(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ForagerMatchedV3ExternalStagingError(f"{label} must be a nonempty exact string")
    path = value
    if unicodedata.normalize("NFKC", path) != path:
        raise ForagerMatchedV3ExternalStagingError(f"{label} must use NFKC Unicode")
    if "\\" in path or any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ForagerMatchedV3ExternalStagingError(f"{label} contains a forbidden character")
    if any(character in '<>:"|?*' for character in path):
        raise ForagerMatchedV3ExternalStagingError(
            f"{label} contains a reserved portable character"
        )
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not a canonical relative path")
    try:
        encoded = path.encode("utf-8", "strict")
        encoded_parts = tuple(part.encode("utf-8", "strict") for part in pure.parts)
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not strict UTF-8") from exc
    if (
        len(encoded) > _MAX_PATH_BYTES
        or len(pure.parts) > _MAX_PATH_COMPONENTS
        or any(len(part) > _MAX_COMPONENT_BYTES for part in encoded_parts)
    ):
        raise ForagerMatchedV3ExternalStagingError(
            f"{label} is not representable by the bounded staging path contract"
        )
    windows_devices = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    for part in pure.parts:
        portable = unicodedata.normalize("NFKC", part).casefold()
        if part.endswith((".", " ")):
            raise ForagerMatchedV3ExternalStagingError(
                f"{label} has a component ending in dot or space"
            )
        if portable == ".git" or re.fullmatch(r"\.?git~[0-9]{1,6}", portable):
            raise ForagerMatchedV3ExternalStagingError(f"{label} aliases reserved Git metadata")
        if portable.split(".", 1)[0] in windows_devices:
            raise ForagerMatchedV3ExternalStagingError(f"{label} aliases a Windows device")
    _split_ustar_path(path)
    return path


def _split_ustar_path(path: str) -> tuple[bytes, bytes]:
    encoded = path.encode("utf-8", "strict")
    if len(encoded) <= 100:
        return b"", encoded
    slash_positions = [index for index, byte in enumerate(encoded) if byte == ord("/")]
    for slash in reversed(slash_positions):
        prefix = encoded[:slash]
        name = encoded[slash + 1 :]
        if prefix and name and len(prefix) <= 155 and len(name) <= 100:
            return prefix, name
    raise ForagerMatchedV3ExternalStagingError(
        f"path is not exactly representable in POSIX USTAR: {path}"
    )


def _validate_path_set(
    paths: Sequence[str],
    *,
    allowed_alias_groups: Sequence[Sequence[str]],
) -> None:
    if type(paths) not in {list, tuple}:
        raise ForagerMatchedV3ExternalStagingError("path inventory must be one exact sequence")
    exact: set[str] = set()
    nodes: dict[str, set[tuple[str, str]]] = {}
    leaves_by_key: dict[str, set[str]] = {}
    for index, raw_path in enumerate(paths):
        path = _validate_relative_path(raw_path, f"path inventory[{index}]")
        if path in exact:
            raise ForagerMatchedV3ExternalStagingError("path inventory contains duplicates")
        exact.add(path)
        parts = PurePosixPath(path).parts
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            nodes.setdefault(_portable_path_key(ancestor), set()).add(("directory", ancestor))
            if ancestor in exact:
                raise ForagerMatchedV3ExternalStagingError(
                    "path inventory contains a file/ancestor collision"
                )
        key = _portable_path_key(path)
        nodes.setdefault(key, set()).add(("regular", path))
        leaves_by_key.setdefault(key, set()).add(path)
    sorted_exact = sorted(exact)
    for left, right in zip(sorted_exact, sorted_exact[1:], strict=False):
        if right.startswith(left + "/"):
            raise ForagerMatchedV3ExternalStagingError(
                "path inventory contains a file/descendant collision"
            )
    expected_aliases = {frozenset(group) for group in allowed_alias_groups}
    if any(len(group) < 2 for group in expected_aliases):
        raise ForagerMatchedV3ExternalStagingError(
            "portable alias exceptions must contain at least two paths"
        )
    observed_aliases: set[frozenset[str]] = set()
    for collision_nodes in nodes.values():
        if len(collision_nodes) < 2:
            continue
        if any(kind != "regular" for kind, _path in collision_nodes):
            raise ForagerMatchedV3ExternalStagingError(
                "path inventory contains a portable ancestor collision"
            )
        observed_aliases.add(frozenset(path for _kind, path in collision_nodes))
    for paths_with_same_key in leaves_by_key.values():
        if len(paths_with_same_key) > 1:
            observed_aliases.add(frozenset(paths_with_same_key))
    if observed_aliases != expected_aliases:
        raise ForagerMatchedV3ExternalStagingError(
            "path inventory portable aliases differ from the exact frozen exception"
        )


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    if type(value) is not int or value < 0:
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not nonnegative")
    token = format(value, "o").encode("ascii")
    if len(token) > width - 1:
        raise ForagerMatchedV3ExternalStagingError(f"{label} exceeds its USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _canonical_ustar_header(path: str, size: int, mode: int) -> bytes:
    if mode not in {0o444, 0o555}:
        raise ForagerMatchedV3ExternalStagingError("USTAR member mode is not frozen")
    prefix, name = _split_ustar_path(path)
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _ustar_octal(mode, 8, "member mode")
    header[108:116] = _ustar_octal(0, 8, "member uid")
    header[116:124] = _ustar_octal(0, 8, "member gid")
    header[124:136] = _ustar_octal(size, 12, "member size")
    header[136:148] = _ustar_octal(0, 12, "member mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = sum(header)
    checksum_token = format(checksum, "06o").encode("ascii")
    if len(checksum_token) != 6:
        raise ForagerMatchedV3ExternalStagingError("USTAR checksum overflowed")
    header[148:156] = checksum_token + b"\0 "
    return bytes(header)


@dataclass(frozen=True)
class _ArchiveMember:
    path: str
    size_bytes: int
    sha256: str
    mode: int
    raw: bytes


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise ForagerMatchedV3ExternalStagingError("descriptor write made no progress")
        view = view[written:]


class _HashingWriter:
    def __init__(self, descriptor: int, maximum_bytes: int) -> None:
        self.descriptor = descriptor
        self.maximum_bytes = maximum_bytes
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, raw: bytes) -> None:
        if type(raw) is not bytes or self.size + len(raw) > self.maximum_bytes:
            raise ForagerMatchedV3ExternalStagingError("canonical USTAR exceeds its byte bound")
        _write_all(self.descriptor, raw)
        self.size += len(raw)
        self.digest.update(raw)


def _write_canonical_ustar(
    descriptor: int,
    members: Sequence[_ArchiveMember],
    *,
    allowed_alias_groups: Sequence[Sequence[str]] = (),
) -> tuple[int, str]:
    if type(members) not in {list, tuple} or not 0 < len(members) <= _MAX_ARCHIVE_MEMBERS:
        raise ForagerMatchedV3ExternalStagingError("USTAR member count is invalid")
    ordered = sorted(members, key=lambda item: item.path.encode("utf-8"))
    if list(members) != ordered:
        raise ForagerMatchedV3ExternalStagingError(
            "USTAR members must be supplied in exact UTF-8 path order"
        )
    _validate_path_set(
        [item.path for item in ordered],
        allowed_alias_groups=allowed_alias_groups,
    )
    writer = _HashingWriter(descriptor, _MAX_ARCHIVE_BYTES)
    for member in ordered:
        if (
            type(member) is not _ArchiveMember
            or type(member.raw) is not bytes
            or len(member.raw) != member.size_bytes
            or member.size_bytes > _MAX_MEMBER_BYTES
            or not hmac.compare_digest(_sha256(member.raw), member.sha256)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"USTAR member identity is invalid: {member.path}"
            )
        writer.write(_canonical_ustar_header(member.path, member.size_bytes, member.mode))
        writer.write(member.raw)
        padding = (-member.size_bytes) % _USTAR_BLOCK_BYTES
        if padding:
            writer.write(bytes(padding))
    writer.write(bytes(2 * _USTAR_BLOCK_BYTES))
    record_padding = (-writer.size) % _USTAR_RECORD_BYTES
    if record_padding:
        writer.write(bytes(record_padding))
    return writer.size, writer.digest.hexdigest()


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": EXTERNAL_STAGING_STATUS,
        "classification": "sealed_external_source_staging_non_authorizing",
        "dependencies": {
            "materializer_v2": {
                "manifest_schema_version": _MATERIALIZER_SCHEMA_VERSION,
                "identity_sha256": _MATERIALIZER_IDENTITY_SHA256,
                "retained_capability_required": True,
                "named_path_accepted": False,
            },
            "external_execution_contract": {
                "schema_version": _EXECUTION_CONTRACT_SCHEMA_VERSION,
                "descriptor_sha256": _EXECUTION_CONTRACT_SHA256,
                "live_descriptor_parsed_on_every_stage": True,
                "historical_configuration_plan_imported": False,
            },
            "configuration_transform": {
                "schema_version": _CONFIGURATION_TRANSFORM_SCHEMA_VERSION,
                "source_path": _CONFIGURATION_SOURCE_SUFFIX,
                "source_sha256": _CONFIGURATION_SOURCE_SHA256,
                "frozen_descriptor_sha256_by_kind": dict(_TRANSFORM_SHA256_BY_KIND),
            },
        },
        "configuration_overlays": {
            "candidate_count": len(_CANDIDATE_IDS),
            "candidate_order": list(_CANDIDATE_IDS),
            "caller_supplied_overlay_allowed": False,
            "original_read_from_retained_tree": True,
            "parent_derivation_replayed": True,
            "isolated_child_original_reread_required": True,
            "execution_contract_original_and_derived_hashes_required": True,
            "staging_path_equals_original_path_required": True,
        },
        "isolated_builder": {
            "interpreter_flags": ["-I", "-S", "-B"],
            "stdlib_only": True,
            "source_directory_descriptor_inherited": True,
            "bounded_sealed_request_descriptor_inherited": True,
            "exact_staging_implementation_source_descriptor_inherited": True,
            "implementation_source_sealed_snapshot_inherited": True,
            "implementation_source_named_path_executed": False,
            "final_manifest_binds_import_time_implementation_sha256": True,
            "directory_descriptor_relative_reads": True,
            "nofollow_nonblocking_regular_single_link_reads": True,
            "pre_and_post_name_descriptor_identity_checks": True,
            "workload_imported": False,
            "workload_executed": False,
        },
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "member_order": "ascending_utf8_path_bytes",
            "member_types": ["regular"],
            "source_mode_mapping": {"100644": "0444", "100755": "0555"},
            "attestation_mode": "0444",
            "uid": 0,
            "gid": 0,
            "mtime": 0,
            "uname": "",
            "gname": "",
            "path_encoding": "strict_utf8_posix_ustar_name_prefix",
            "payload_padding": "zero_to_512_byte_block",
            "end_blocks": 2,
            "record_padding": "zero_to_exact_10240_byte_multiple",
            "materializer_root_manifest_included": False,
            "materializer_manifest_attestation_path": (EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH),
            "final_manifest_path": EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            "final_manifest_self_excluded_from_its_payload_inventory": True,
            "output_storage": "sealed_unlinked_private_regular_file_descriptor",
            "output_descriptor_access_mode": "read_only",
        },
        "verification": {
            "retained_source_reverify_before": True,
            "retained_source_reverify_after_archive_replay": True,
            "parent_raw_ustar_replay_required": True,
            "returned_capability_reverify_checks_raw_ustar": True,
            "descriptor_and_inode_drift_rejected": True,
            "portable_alias_policy": (
                "only_exact_materializer_identity_exception_preserved_as_distinct_members"
            ),
        },
        "limits": {
            "maximum_source_members": _MAX_SOURCE_MEMBERS,
            "maximum_archive_members": _MAX_ARCHIVE_MEMBERS,
            "maximum_member_bytes": _MAX_MEMBER_BYTES,
            "maximum_source_payload_bytes": _MAX_SOURCE_PAYLOAD_BYTES,
            "maximum_request_bytes": _MAX_REQUEST_BYTES,
            "maximum_manifest_bytes": _MAX_MANIFEST_BYTES,
            "maximum_receipt_bytes": _MAX_RECEIPT_BYTES,
            "maximum_archive_bytes": _MAX_ARCHIVE_BYTES,
            "maximum_path_bytes": _MAX_PATH_BYTES,
            "worker_timeout_seconds": int(_WORKER_TIMEOUT_SECONDS),
        },
        "apis": {
            "path_input_exposed": False,
            "caller_overlay_input_exposed": False,
            "seed_input_exposed": False,
            "result_input_exposed": False,
            "extractor_exposed": False,
            "publisher_exposed": False,
            "runner_exposed": False,
            "workload_subprocess_exposed": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "ceea86b38822f3add0465788003d349dd221a49fba5f3fa069bfec985537caea"
)
if not hmac.compare_digest(_sha256(_DESCRIPTOR_BYTES), EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256):
    raise AssertionError("external staging-contract descriptor identity drifted")


def external_staging_contract_descriptor() -> dict[str, Any]:
    """Return a detached, nonauthorizing staging-contract descriptor."""

    return _strict_json_load(
        _DESCRIPTOR_BYTES,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=True,
    )


def canonical_external_staging_contract_descriptor_bytes() -> bytes:
    """Return the exact canonical staging-contract descriptor bytes."""

    return _DESCRIPTOR_BYTES


def external_staging_contract_descriptor_sha256() -> str:
    """Return the exact staging-contract descriptor SHA-256."""

    return EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256


def parse_external_staging_contract_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact pinned staging-contract descriptor."""

    value = _strict_json_load(
        raw,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=True,
    )
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        _sha256(raw), EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging-contract descriptor identity drifted"
        )
    return value


def _require_plain_object(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ForagerMatchedV3ExternalStagingError(f"{label} fields do not match")
    return cast(dict[str, Any], value)


def _source_mode(git_mode: Any, label: str) -> tuple[str, int]:
    if git_mode == "100644":
        return "0444", 0o444
    if git_mode == "100755":
        return "0555", 0o555
    raise ForagerMatchedV3ExternalStagingError(f"{label} has an unsupported Git mode")


def _base_file_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_tree = manifest.get("source_tree")
    if type(source_tree) is not dict:
        raise ForagerMatchedV3ExternalStagingError("base manifest source_tree is invalid")
    raw_files = source_tree.get("files")
    count = source_tree.get("materialized_regular_file_count")
    total = source_tree.get("materialized_total_size_bytes")
    if (
        type(raw_files) is not list
        or not raw_files
        or len(raw_files) > _MAX_SOURCE_MEMBERS
        or type(count) is not int
        or count != len(raw_files)
        or type(total) is not int
        or total < 0
        or total > _MAX_SOURCE_PAYLOAD_BYTES
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "base manifest source inventory exceeds staging bounds"
        )
    records: list[dict[str, Any]] = []
    observed_total = 0
    previous_path: str | None = None
    for index, item in enumerate(raw_files):
        if type(item) is not dict:
            raise ForagerMatchedV3ExternalStagingError(
                f"base manifest source file {index} is not a plain object"
            )
        record = cast(dict[str, Any], item)
        required = {
            "path",
            "git_mode",
            "upstream_blob_git_sha1",
            "upstream_size_bytes",
            "upstream_sha256",
            "materialized_size_bytes",
            "materialized_sha256",
            "transformed",
        }
        if set(record) != required:
            raise ForagerMatchedV3ExternalStagingError(
                f"base manifest source file {index} fields do not match"
            )
        path = _validate_relative_path(record["path"], f"base source file {index} path")
        if previous_path is not None and path <= previous_path:
            raise ForagerMatchedV3ExternalStagingError(
                "base source records must be path-sorted and unique"
            )
        _source_mode(record["git_mode"], f"base source file {path}")
        size = _require_exact_int(
            record["materialized_size_bytes"],
            f"base source file {path} size",
            maximum=_MAX_MEMBER_BYTES,
        )
        _require_sha256(record["materialized_sha256"], f"base source file {path} digest")
        _require_sha256(record["upstream_sha256"], f"base source file {path} upstream digest")
        if type(record["upstream_size_bytes"]) is not int or record["upstream_size_bytes"] < 0:
            raise ForagerMatchedV3ExternalStagingError(
                f"base source file {path} upstream size is invalid"
            )
        if type(record["transformed"]) is not bool:
            raise ForagerMatchedV3ExternalStagingError(
                f"base source file {path} transformed flag is invalid"
            )
        observed_total += size
        if observed_total > _MAX_SOURCE_PAYLOAD_BYTES:
            raise ForagerMatchedV3ExternalStagingError("base source payload exceeds staging bounds")
        records.append(record)
        previous_path = path
    if observed_total != total:
        raise ForagerMatchedV3ExternalStagingError(
            "base source payload total differs from its manifest"
        )
    return records


def _base_alias_groups(manifest: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    identity = manifest.get("identity")
    if type(identity) is not dict:
        raise ForagerMatchedV3ExternalStagingError("base identity is invalid")
    raw_aliases = identity.get("portable_path_aliases")
    if type(raw_aliases) is not list:
        raise ForagerMatchedV3ExternalStagingError("base portable aliases are invalid")
    groups: dict[str, list[str]] = {}
    previous: str | None = None
    for index, item in enumerate(raw_aliases):
        if type(item) is not dict or set(item) != {"path", "blob_git_sha1"}:
            raise ForagerMatchedV3ExternalStagingError(f"base portable alias {index} is invalid")
        path = _validate_relative_path(item["path"], f"base portable alias {index} path")
        if previous is not None and path <= previous:
            raise ForagerMatchedV3ExternalStagingError(
                "base portable aliases must be path-sorted and unique"
            )
        groups.setdefault(_portable_path_key(path), []).append(path)
        previous = path
    result = tuple(tuple(paths) for _key, paths in sorted(groups.items()))
    if any(len(group) < 2 for group in result):
        raise ForagerMatchedV3ExternalStagingError("base portable alias exception is incomplete")
    if result != _PINNED_PORTABLE_ALIAS_GROUPS:
        raise ForagerMatchedV3ExternalStagingError(
            "base portable alias exception differs from the exact frozen pair"
        )
    return result


@dataclass(frozen=True)
class _DerivedOverlay:
    candidate_id: str
    path: str
    original_size_bytes: int
    original_sha256: str
    derived_size_bytes: int
    derived_sha256: str
    transform_descriptor_sha256: str
    raw: bytes

    def manifest_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "path": self.path,
            "original_size_bytes": self.original_size_bytes,
            "original_sha256": self.original_sha256,
            "derived_size_bytes": self.derived_size_bytes,
            "derived_sha256": self.derived_sha256,
            "transform_descriptor_sha256": self.transform_descriptor_sha256,
            "archive_mode": "0444",
        }


def _inventory_record(
    *,
    path: str,
    size_bytes: int,
    sha256: str,
    mode: str,
    provenance: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "mode": mode,
        "provenance": provenance,
    }


def _build_stage_manifest(
    base_manifest_raw: bytes,
    base_manifest: dict[str, Any],
    overlays: Sequence[_DerivedOverlay],
) -> tuple[bytes, str, list[dict[str, Any]], tuple[tuple[str, ...], ...]]:
    if type(overlays) not in {list, tuple} or tuple(item.candidate_id for item in overlays) != (
        _CANDIDATE_IDS
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "derived overlays do not cover the exact candidate order"
        )
    base_records = _base_file_records(base_manifest)
    base_by_path = {cast(str, record["path"]): record for record in base_records}
    overlay_by_path: dict[str, _DerivedOverlay] = {}
    for overlay in overlays:
        if type(overlay) is not _DerivedOverlay or overlay.path in overlay_by_path:
            raise ForagerMatchedV3ExternalStagingError(
                "derived overlay paths must be exact and unique"
            )
        base_record = base_by_path.get(overlay.path)
        if base_record is None:
            raise ForagerMatchedV3ExternalStagingError(
                f"derived overlay path is absent from base manifest: {overlay.path}"
            )
        if (
            base_record["git_mode"] != "100644"
            or base_record["materialized_size_bytes"] != overlay.original_size_bytes
            or not hmac.compare_digest(
                cast(str, base_record["materialized_sha256"]), overlay.original_sha256
            )
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"derived overlay original binding differs: {overlay.candidate_id}"
            )
        if (
            type(overlay.raw) is not bytes
            or len(overlay.raw) != overlay.derived_size_bytes
            or not hmac.compare_digest(_sha256(overlay.raw), overlay.derived_sha256)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"derived overlay bytes differ: {overlay.candidate_id}"
            )
        overlay_by_path[overlay.path] = overlay

    inventory: list[dict[str, Any]] = []
    for record in base_records:
        path = cast(str, record["path"])
        selected_overlay = overlay_by_path.get(path)
        mode, _numeric_mode = _source_mode(record["git_mode"], f"base source file {path}")
        if selected_overlay is None:
            inventory.append(
                _inventory_record(
                    path=path,
                    size_bytes=cast(int, record["materialized_size_bytes"]),
                    sha256=cast(str, record["materialized_sha256"]),
                    mode=mode,
                    provenance="materializer_v2_regular_file",
                )
            )
        else:
            inventory.append(
                _inventory_record(
                    path=path,
                    size_bytes=selected_overlay.derived_size_bytes,
                    sha256=selected_overlay.derived_sha256,
                    mode="0444",
                    provenance="derived_configuration_overlay",
                )
            )
    inventory.append(
        _inventory_record(
            path=EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
            size_bytes=len(base_manifest_raw),
            sha256=_sha256(base_manifest_raw),
            mode="0444",
            provenance="relocated_exact_materializer_v2_manifest",
        )
    )
    inventory.sort(key=lambda item: cast(str, item["path"]).encode("utf-8"))
    aliases = _base_alias_groups(base_manifest)
    _validate_path_set(
        [cast(str, item["path"]) for item in inventory] + [EXTERNAL_STAGING_FINAL_MANIFEST_PATH],
        allowed_alias_groups=aliases,
    )
    total_bytes = sum(cast(int, item["size_bytes"]) for item in inventory)
    if total_bytes > _MAX_SOURCE_PAYLOAD_BYTES + _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3ExternalStagingError("final payload inventory exceeds staging bounds")
    identity_sha256 = _require_sha256(
        base_manifest.get("identity_sha256"), "base materializer identity digest"
    )
    if not hmac.compare_digest(identity_sha256, _MATERIALIZER_IDENTITY_SHA256):
        raise ForagerMatchedV3ExternalStagingError(
            "base materializer identity differs from the pinned v2 identity"
        )
    body: dict[str, Any] = {
        "schema_version": EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        "status": EXTERNAL_STAGING_STATUS,
        "classification": "sealed_external_source_staging_non_authorizing",
        "staging_contract_descriptor_sha256": (EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256),
        "implementation_source_sha256": _IMPORTED_IMPLEMENTATION_SOURCE_SHA256,
        "execution_contract": {
            "schema_version": _EXECUTION_CONTRACT_SCHEMA_VERSION,
            "descriptor_sha256": _EXECUTION_CONTRACT_SHA256,
            "candidate_count": len(_CANDIDATE_IDS),
            "candidate_order": list(_CANDIDATE_IDS),
        },
        "base_materialization": {
            "manifest_schema_version": _MATERIALIZER_SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "manifest_root_path_removed": _MATERIALIZER_MANIFEST_FILENAME,
            "manifest_attestation_path": EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
            "manifest_size_bytes": len(base_manifest_raw),
            "manifest_sha256": _sha256(base_manifest_raw),
            "source_regular_file_count": len(base_records),
            "source_materialized_total_size_bytes": sum(
                cast(int, record["materialized_size_bytes"]) for record in base_records
            ),
        },
        "configuration_overlays": [item.manifest_record() for item in overlays],
        "payload_inventory": inventory,
        "archive_layout": {
            "format": "canonical_posix_ustar_uncompressed",
            "nonself_member_count": len(inventory),
            "nonself_payload_bytes": total_bytes,
            "final_manifest_path": EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            "final_manifest_mode": "0444",
            "final_manifest_self_excluded_from_payload_inventory": True,
            "complete_member_count": len(inventory) + 1,
            "member_order": "ascending_utf8_path_bytes",
            "record_size_bytes": _USTAR_RECORD_BYTES,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }
    body["manifest_body_sha256"] = _sha256(_canonical_json(body))
    raw = _canonical_json(body)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest exceeds its byte bound"
        )
    parsed = parse_external_staging_manifest(raw, expected_manifest_sha256=_sha256(raw))
    if parsed != body:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest detached replay differs"
        )
    return raw, _sha256(raw), inventory, aliases


def _validate_inventory(value: Any) -> tuple[list[dict[str, Any]], int]:
    if type(value) is not list or not value or len(value) > _MAX_ARCHIVE_MEMBERS:
        raise ForagerMatchedV3ExternalStagingError("staging payload inventory is invalid")
    inventory: list[dict[str, Any]] = []
    previous: bytes | None = None
    total = 0
    for index, item in enumerate(value):
        record = _require_plain_object(
            item,
            frozenset({"path", "size_bytes", "sha256", "mode", "provenance"}),
            f"staging payload inventory {index}",
        )
        path = _validate_relative_path(record["path"], f"staging payload inventory {index} path")
        encoded = path.encode("utf-8")
        if previous is not None and encoded <= previous:
            raise ForagerMatchedV3ExternalStagingError(
                "staging payload inventory must be UTF-8-path-sorted and unique"
            )
        size = _require_exact_int(
            record["size_bytes"],
            f"staging payload inventory {path} size",
            maximum=_MAX_MEMBER_BYTES,
        )
        _require_sha256(record["sha256"], f"staging payload inventory {path} digest")
        if record["mode"] not in {"0444", "0555"}:
            raise ForagerMatchedV3ExternalStagingError(
                f"staging payload inventory {path} mode is invalid"
            )
        if record["provenance"] not in {
            "materializer_v2_regular_file",
            "derived_configuration_overlay",
            "relocated_exact_materializer_v2_manifest",
        }:
            raise ForagerMatchedV3ExternalStagingError(
                f"staging payload inventory {path} provenance is invalid"
            )
        total += size
        if total > _MAX_SOURCE_PAYLOAD_BYTES + _MAX_MANIFEST_BYTES:
            raise ForagerMatchedV3ExternalStagingError(
                "staging payload inventory total exceeds its bound"
            )
        inventory.append(record)
        previous = encoded
    return inventory, total


def parse_external_staging_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Parse one exact digest-bound, authority-denying final staging manifest."""

    expected_digest = _require_sha256(
        expected_manifest_sha256, "expected external staging manifest digest"
    )
    if type(raw) is not bytes or not hmac.compare_digest(_sha256(raw), expected_digest):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest digest does not match"
        )
    manifest = _strict_json_load(
        raw,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=True,
    )
    root_keys = frozenset(
        {
            "schema_version",
            "status",
            "classification",
            "staging_contract_descriptor_sha256",
            "implementation_source_sha256",
            "execution_contract",
            "base_materialization",
            "configuration_overlays",
            "payload_inventory",
            "archive_layout",
            "claims",
            "limitations",
            "manifest_body_sha256",
        }
    )
    if set(manifest) != root_keys:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest root fields do not match"
        )
    if (
        manifest["schema_version"] != EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != EXTERNAL_STAGING_STATUS
        or manifest["classification"] != "sealed_external_source_staging_non_authorizing"
        or manifest["staging_contract_descriptor_sha256"]
        != EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256
        or manifest["implementation_source_sha256"] != _IMPORTED_IMPLEMENTATION_SOURCE_SHA256
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest identity fields do not match"
        )
    execution = _require_plain_object(
        manifest["execution_contract"],
        frozenset({"schema_version", "descriptor_sha256", "candidate_count", "candidate_order"}),
        "external staging execution contract",
    )
    if (
        execution["schema_version"] != _EXECUTION_CONTRACT_SCHEMA_VERSION
        or execution["descriptor_sha256"] != _EXECUTION_CONTRACT_SHA256
        or execution["candidate_count"] != len(_CANDIDATE_IDS)
        or execution["candidate_order"] != list(_CANDIDATE_IDS)
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging execution-contract binding does not match"
        )
    base = _require_plain_object(
        manifest["base_materialization"],
        frozenset(
            {
                "manifest_schema_version",
                "identity_sha256",
                "manifest_root_path_removed",
                "manifest_attestation_path",
                "manifest_size_bytes",
                "manifest_sha256",
                "source_regular_file_count",
                "source_materialized_total_size_bytes",
            }
        ),
        "external staging base materialization",
    )
    if (
        base["manifest_schema_version"] != _MATERIALIZER_SCHEMA_VERSION
        or base["identity_sha256"] != _MATERIALIZER_IDENTITY_SHA256
        or base["manifest_root_path_removed"] != _MATERIALIZER_MANIFEST_FILENAME
        or base["manifest_attestation_path"] != EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging base-materialization identity does not match"
        )
    _require_sha256(base["manifest_sha256"], "base materialization manifest digest")
    _require_exact_int(
        base["manifest_size_bytes"],
        "base materialization manifest size",
        maximum=_MAX_MANIFEST_BYTES,
    )
    source_count = _require_exact_int(
        base["source_regular_file_count"],
        "base source file count",
        maximum=_MAX_SOURCE_MEMBERS,
    )
    _require_exact_int(
        base["source_materialized_total_size_bytes"],
        "base source payload size",
        maximum=_MAX_SOURCE_PAYLOAD_BYTES,
    )

    raw_overlays = manifest["configuration_overlays"]
    if type(raw_overlays) is not list or len(raw_overlays) != len(_CANDIDATE_IDS):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging configuration overlays do not cover all candidates"
        )
    overlay_paths: set[str] = set()
    overlay_by_path: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_overlays):
        record = _require_plain_object(
            item,
            frozenset(
                {
                    "candidate_id",
                    "path",
                    "original_size_bytes",
                    "original_sha256",
                    "derived_size_bytes",
                    "derived_sha256",
                    "transform_descriptor_sha256",
                    "archive_mode",
                }
            ),
            f"external staging overlay {index}",
        )
        (
            expected_candidate_id,
            expected_path,
            expected_original_sha256,
            expected_derived_sha256,
        ) = _FROZEN_EXECUTION_RECORDS[index]
        if (
            record["candidate_id"] != expected_candidate_id
            or record["path"] != expected_path
            or record["original_sha256"] != expected_original_sha256
            or record["derived_sha256"] != expected_derived_sha256
            or record["transform_descriptor_sha256"]
            != _expected_transform_sha256(expected_candidate_id)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging overlay execution semantics differ"
            )
        path = _validate_relative_path(record["path"], f"external staging overlay {index} path")
        if path in overlay_paths or record["archive_mode"] != "0444":
            raise ForagerMatchedV3ExternalStagingError(
                "external staging overlay paths or modes differ"
            )
        overlay_paths.add(path)
        for field in ("original_sha256", "derived_sha256", "transform_descriptor_sha256"):
            _require_sha256(record[field], f"external staging overlay {index} {field}")
        _require_exact_int(
            record["original_size_bytes"],
            f"external staging overlay {index} original size",
            maximum=_MAX_MEMBER_BYTES,
        )
        _require_exact_int(
            record["derived_size_bytes"],
            f"external staging overlay {index} derived size",
            maximum=_MAX_MEMBER_BYTES,
        )
        overlay_by_path[path] = record

    inventory, inventory_total = _validate_inventory(manifest["payload_inventory"])
    inventory_by_path = {cast(str, record["path"]): record for record in inventory}
    for path, overlay in overlay_by_path.items():
        item = inventory_by_path.get(path)
        if (
            item is None
            or item["provenance"] != "derived_configuration_overlay"
            or item["mode"] != "0444"
            or item["size_bytes"] != overlay["derived_size_bytes"]
            or item["sha256"] != overlay["derived_sha256"]
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging overlay differs from payload inventory"
            )
    attestation = inventory_by_path.get(EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH)
    if (
        attestation is None
        or attestation["provenance"] != "relocated_exact_materializer_v2_manifest"
        or attestation["mode"] != "0444"
        or attestation["size_bytes"] != base["manifest_size_bytes"]
        or attestation["sha256"] != base["manifest_sha256"]
        or _MATERIALIZER_MANIFEST_FILENAME in inventory_by_path
        or EXTERNAL_STAGING_FINAL_MANIFEST_PATH in inventory_by_path
        or len(inventory) != source_count + 1
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging materializer-manifest relocation differs"
        )
    layout = _require_plain_object(
        manifest["archive_layout"],
        frozenset(
            {
                "format",
                "nonself_member_count",
                "nonself_payload_bytes",
                "final_manifest_path",
                "final_manifest_mode",
                "final_manifest_self_excluded_from_payload_inventory",
                "complete_member_count",
                "member_order",
                "record_size_bytes",
            }
        ),
        "external staging archive layout",
    )
    if layout != {
        "format": "canonical_posix_ustar_uncompressed",
        "nonself_member_count": len(inventory),
        "nonself_payload_bytes": inventory_total,
        "final_manifest_path": EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
        "final_manifest_mode": "0444",
        "final_manifest_self_excluded_from_payload_inventory": True,
        "complete_member_count": len(inventory) + 1,
        "member_order": "ascending_utf8_path_bytes",
        "record_size_bytes": _USTAR_RECORD_BYTES,
    }:
        raise ForagerMatchedV3ExternalStagingError("external staging archive layout does not match")
    claims = manifest["claims"]
    expected_claims = _claims()
    if (
        type(claims) is not dict
        or set(claims) != set(expected_claims)
        or any(claims[key] is not False for key in expected_claims)
        or manifest["limitations"] != _limitations()
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest authority denial does not match"
        )
    body_digest = _require_sha256(
        manifest["manifest_body_sha256"], "external staging manifest body digest"
    )
    body = dict(manifest)
    body.pop("manifest_body_sha256")
    if not hmac.compare_digest(_sha256(_canonical_json(body)), body_digest):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging manifest body digest does not match"
        )
    return _strict_json_load(
        raw,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=True,
    )


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging requires O_NOFOLLOW and O_DIRECTORY"
        )
    flags = os.O_RDONLY | os.O_NONBLOCK | nofollow | directory
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    return flags


def _file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise ForagerMatchedV3ExternalStagingError("external staging requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | nofollow
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    return flags


def _duplicate_directory_descriptor(descriptor: int, label: str) -> int:
    if type(descriptor) is not int or descriptor < 0:
        raise ForagerMatchedV3ExternalStagingError(f"{label} descriptor is invalid")
    duplicate = -1
    try:
        before = os.fstat(descriptor)
        duplicate = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
        after = os.fstat(descriptor)
        duplicate_stat = os.fstat(duplicate)
        duplicate_inheritable = os.get_inheritable(duplicate)
    except OSError as exc:
        if duplicate >= 0:
            try:
                os.close(duplicate)
            except OSError:
                pass
        raise ForagerMatchedV3ExternalStagingError(
            f"{label} descriptor cannot be retained"
        ) from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or _directory_identity(before) != _directory_identity(after)
        or _directory_identity(before) != _directory_identity(duplicate_stat)
        or duplicate_inheritable
    ):
        os.close(duplicate)
        raise ForagerMatchedV3ExternalStagingError(f"{label} descriptor identity changed")
    return duplicate


def _open_relative_parent(root_descriptor: int, path: str) -> tuple[int, str]:
    parts = PurePosixPath(path).parts
    current = _duplicate_directory_descriptor(root_descriptor, "source root")
    try:
        for part in parts[:-1]:
            child = -1
            try:
                before = os.stat(part, dir_fd=current, follow_symlinks=False)
                child = os.open(part, _directory_flags(), dir_fd=current)
                opened = os.fstat(child)
                after = os.stat(part, dir_fd=current, follow_symlinks=False)
            except OSError as exc:
                if child >= 0:
                    try:
                        os.close(child)
                    except OSError:
                        pass
                raise ForagerMatchedV3ExternalStagingError(
                    f"source ancestor is inaccessible: {path}"
                ) from exc
            if (
                not stat.S_ISDIR(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or _stat_signature(before) != _stat_signature(opened)
                or _stat_signature(before) != _stat_signature(after)
                or stat.S_IMODE(opened.st_mode) != 0o755
            ):
                try:
                    os.close(child)
                except OSError:
                    pass
                raise ForagerMatchedV3ExternalStagingError(
                    f"source ancestor changed or is not canonical: {path}"
                )
            previous = current
            current = child
            try:
                os.close(previous)
            except OSError as exc:
                try:
                    os.close(current)
                except OSError:
                    pass
                current = previous
                raise ForagerMatchedV3ExternalStagingError(
                    f"source ancestor descriptor cleanup failed: {path}"
                ) from exc
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _read_relative_regular_file(
    root_descriptor: int,
    path: str,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    expected_mode: int,
    maximum_bytes: int,
) -> bytes:
    path = _validate_relative_path(path, "source member path")
    if expected_mode not in {0o644, 0o755}:
        raise ForagerMatchedV3ExternalStagingError("source member expected mode is invalid")
    if expected_size is not None:
        _require_exact_int(expected_size, "source member expected size", maximum=maximum_bytes)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "source member expected digest")
    root_before = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ForagerMatchedV3ExternalStagingError("source root descriptor is not a directory")
    parent, name = _open_relative_parent(root_descriptor, path)
    descriptor = -1
    try:
        try:
            name_before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            descriptor = os.open(name, _file_flags(), dir_fd=parent)
            before = os.fstat(descriptor)
        except OSError as exc:
            raise ForagerMatchedV3ExternalStagingError(
                f"source member is inaccessible: {path}"
            ) from exc
        if (
            not stat.S_ISREG(name_before.st_mode)
            or stat.S_ISLNK(name_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_signature(name_before) != _stat_signature(before)
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_size < 0
            or before.st_size > maximum_bytes
            or (expected_size is not None and before.st_size != expected_size)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"source member is not one canonical single-link regular file: {path}"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            except InterruptedError:
                continue
            if not chunk:
                raise ForagerMatchedV3ExternalStagingError(
                    f"source member was truncated while reading: {path}"
                )
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3ExternalStagingError(f"source member grew while reading: {path}")
        after = os.fstat(descriptor)
        name_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        root_after = os.fstat(root_descriptor)
        if (
            _stat_signature(before) != _stat_signature(after)
            or _stat_signature(before) != _stat_signature(name_after)
            or _directory_identity(root_before) != _directory_identity(root_after)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"source member or root changed while reading: {path}"
            )
        actual_digest = digest.hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(actual_digest, expected_sha256):
            raise ForagerMatchedV3ExternalStagingError(
                f"source member bytes differ from their manifest: {path}"
            )
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _bounded_source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise ForagerMatchedV3ExternalStagingError(
            f"dependency source path differs from {expected_suffix}"
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        raise ForagerMatchedV3ExternalStagingError(
            "dependency source verification requires O_NOFOLLOW"
        )
    flags = os.O_RDONLY | os.O_NONBLOCK | nofollow
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if type(cloexec) is int:
        flags |= cloexec
    try:
        descriptor = os.open(module_file, flags)
    except OSError as exc:
        raise ForagerMatchedV3ExternalStagingError(
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
            raise ForagerMatchedV3ExternalStagingError(
                f"dependency source is not one bounded single-link file: {expected_suffix}"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise ForagerMatchedV3ExternalStagingError(
                    f"dependency source was truncated: {expected_suffix}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3ExternalStagingError(f"dependency source grew: {expected_suffix}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_signature(before) != _stat_signature(after):
        raise ForagerMatchedV3ExternalStagingError(f"dependency source changed: {expected_suffix}")
    return digest.hexdigest()


@dataclass(frozen=True)
class _ExecutionRecord:
    candidate_id: str
    path: str
    original_sha256: str
    derived_sha256: str


def _validated_execution_records() -> tuple[bytes, tuple[_ExecutionRecord, ...]]:
    from alberta_framework.benchmarks import (
        forager_matched_v3_external_execution_contract as execution_contract,
    )

    try:
        raw = execution_contract.canonical_external_execution_contract_descriptor_bytes()
        parsed = execution_contract.parse_external_execution_contract_descriptor(raw)
    except (AssertionError, ValueError) as exc:
        raise ForagerMatchedV3ExternalStagingError(
            "live external execution-contract binding failed"
        ) from exc
    if (
        execution_contract.EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256
        != _EXECUTION_CONTRACT_SHA256
        or not hmac.compare_digest(_sha256(raw), _EXECUTION_CONTRACT_SHA256)
        or parsed.get("schema_version") != _EXECUTION_CONTRACT_SCHEMA_VERSION
        or parsed.get("candidate_count") != len(_CANDIDATE_IDS)
        or parsed.get("candidate_order") != list(_CANDIDATE_IDS)
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external execution-contract identity or order drifted"
        )
    candidates = parsed.get("candidates")
    if type(candidates) is not list or len(candidates) != len(_CANDIDATE_IDS):
        raise ForagerMatchedV3ExternalStagingError("external execution-contract candidates drifted")
    records: list[_ExecutionRecord] = []
    paths: list[str] = []
    for index, candidate in enumerate(candidates):
        if type(candidate) is not dict or candidate.get("candidate_id") != _CANDIDATE_IDS[index]:
            raise ForagerMatchedV3ExternalStagingError(
                "external execution-contract candidate order drifted"
            )
        configuration = candidate.get("configuration")
        if type(configuration) is not dict:
            raise ForagerMatchedV3ExternalStagingError(
                "external execution-contract configuration is invalid"
            )
        path = _validate_relative_path(
            configuration.get("original_relative_path"),
            f"external execution candidate {index} configuration path",
        )
        if (
            configuration.get("derived_configuration_staging_relative_path") != path
            or configuration.get("staging_preserves_original_path_below_experiments") is not True
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external execution-contract staging path relation drifted"
            )
        original_sha256 = _require_sha256(
            configuration.get("original_sha256"),
            f"external execution candidate {index} original digest",
        )
        derived_sha256 = _require_sha256(
            configuration.get("derived_sha256"),
            f"external execution candidate {index} derived digest",
        )
        records.append(
            _ExecutionRecord(
                candidate_id=_CANDIDATE_IDS[index],
                path=path,
                original_sha256=original_sha256,
                derived_sha256=derived_sha256,
            )
        )
        paths.append(path)
    _validate_path_set(paths, allowed_alias_groups=())
    result = tuple(records)
    if (
        tuple(
            (item.candidate_id, item.path, item.original_sha256, item.derived_sha256)
            for item in result
        )
        != _FROZEN_EXECUTION_RECORDS
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external execution-contract exact configuration records drifted"
        )
    return raw, result


def _derive_overlays(
    root_descriptor: int,
    base_manifest: dict[str, Any],
    records: Sequence[_ExecutionRecord],
) -> tuple[_DerivedOverlay, ...]:
    from alberta_framework.benchmarks import forager_matched_v3_configuration as configuration

    actual_source_sha256 = _bounded_source_sha256(
        configuration.__file__, _CONFIGURATION_SOURCE_SUFFIX
    )
    if not hmac.compare_digest(actual_source_sha256, _CONFIGURATION_SOURCE_SHA256):
        raise ForagerMatchedV3ExternalStagingError(
            "configuration transform implementation binding drifted"
        )
    if (
        configuration.DESCRIPTOR_SCHEMA_VERSION != _CONFIGURATION_TRANSFORM_SCHEMA_VERSION
        or tuple(record.candidate_id for record in records) != _CANDIDATE_IDS
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "configuration transform schema or execution records drifted"
        )
    base_by_path = {cast(str, item["path"]): item for item in _base_file_records(base_manifest)}
    overlays: list[_DerivedOverlay] = []
    for record in records:
        base = base_by_path.get(record.path)
        if base is None:
            raise ForagerMatchedV3ExternalStagingError(
                f"external configuration is absent from base manifest: {record.path}"
            )
        if base["git_mode"] != "100644" or base["materialized_sha256"] != record.original_sha256:
            raise ForagerMatchedV3ExternalStagingError(
                f"external configuration original binding drifted: {record.candidate_id}"
            )
        original_size = cast(int, base["materialized_size_bytes"])
        original_raw = _read_relative_regular_file(
            root_descriptor,
            record.path,
            expected_size=original_size,
            expected_sha256=record.original_sha256,
            expected_mode=0o644,
            maximum_bytes=_MAX_MEMBER_BYTES,
        )
        descriptor = cast(
            dict[str, Any], json.loads(json.dumps(_TRANSFORM_BY_CANDIDATE[record.candidate_id]))
        )
        descriptor_sha256 = configuration.canonical_descriptor_sha256(descriptor)
        expected_descriptor_sha256 = _expected_transform_sha256(record.candidate_id)
        if not hmac.compare_digest(descriptor_sha256, expected_descriptor_sha256):
            raise ForagerMatchedV3ExternalStagingError(
                f"external configuration descriptor drifted: {record.candidate_id}"
            )
        try:
            derived = configuration.derive_configuration(original_raw, descriptor)
        except configuration.ForagerMatchedV3ConfigurationError as exc:
            raise ForagerMatchedV3ExternalStagingError(
                f"external configuration derivation failed: {record.candidate_id}"
            ) from exc
        if (
            derived.original_sha256 != record.original_sha256
            or derived.descriptor_sha256 != expected_descriptor_sha256
            or derived.derived_sha256 != record.derived_sha256
            or len(derived.derived_canonical_bytes) > _MAX_MEMBER_BYTES
        ):
            raise ForagerMatchedV3ExternalStagingError(
                f"external configuration derivation identity drifted: {record.candidate_id}"
            )
        overlays.append(
            _DerivedOverlay(
                candidate_id=record.candidate_id,
                path=record.path,
                original_size_bytes=original_size,
                original_sha256=record.original_sha256,
                derived_size_bytes=len(derived.derived_canonical_bytes),
                derived_sha256=derived.derived_sha256,
                transform_descriptor_sha256=derived.descriptor_sha256,
                raw=derived.derived_canonical_bytes,
            )
        )
    return tuple(overlays)


def _pread_exact(descriptor: int, size: int, offset: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        try:
            chunk = os.pread(descriptor, min(remaining, _READ_CHUNK_BYTES), position)
        except InterruptedError:
            continue
        if not chunk:
            raise ForagerMatchedV3ExternalStagingError(f"{label} ended early")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _expected_complete_inventory(
    manifest_raw: bytes,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[tuple[str, ...], ...]]:
    inventory = [dict(item) for item in cast(list[dict[str, Any]], manifest["payload_inventory"])]
    inventory.append(
        _inventory_record(
            path=EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            size_bytes=len(manifest_raw),
            sha256=_sha256(manifest_raw),
            mode="0444",
            provenance="final_staging_manifest_self",
        )
    )
    inventory.sort(key=lambda item: cast(str, item["path"]).encode("utf-8"))
    _validate_path_set(
        [cast(str, item["path"]) for item in inventory],
        allowed_alias_groups=_PINNED_PORTABLE_ALIAS_GROUPS,
    )
    return inventory, _PINNED_PORTABLE_ALIAS_GROUPS


def _hash_fd(descriptor: int, size: int, label: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = _pread_exact(
            descriptor,
            min(_READ_CHUNK_BYTES, size - offset),
            offset,
            label,
        )
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _verify_canonical_ustar_fd(
    descriptor: int,
    *,
    expected_size: int,
    expected_sha256: str,
    manifest_raw: bytes,
    manifest_sha256: str,
) -> dict[str, Any]:
    _require_exact_int(expected_size, "expected USTAR size", maximum=_MAX_ARCHIVE_BYTES)
    _require_sha256(expected_sha256, "expected USTAR digest")
    manifest = parse_external_staging_manifest(
        manifest_raw,
        expected_manifest_sha256=manifest_sha256,
    )
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3ExternalStagingError(
            "retained USTAR descriptor is inaccessible"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_size != expected_size
        or expected_size <= 0
        or expected_size % _USTAR_RECORD_BYTES != 0
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "retained USTAR descriptor metadata does not match"
        )
    inventory, aliases = _expected_complete_inventory(manifest_raw, manifest)
    if len(inventory) != cast(dict[str, Any], manifest["archive_layout"])["complete_member_count"]:
        raise ForagerMatchedV3ExternalStagingError(
            "complete USTAR inventory count differs from final manifest"
        )
    offset = 0
    observed_paths: list[str] = []
    for record in inventory:
        path = cast(str, record["path"])
        size = cast(int, record["size_bytes"])
        mode_text = cast(str, record["mode"])
        mode = 0o444 if mode_text == "0444" else 0o555
        header = _pread_exact(descriptor, _USTAR_BLOCK_BYTES, offset, f"USTAR header {path}")
        expected_header = _canonical_ustar_header(path, size, mode)
        if not hmac.compare_digest(header, expected_header):
            raise ForagerMatchedV3ExternalStagingError(f"USTAR header is not canonical: {path}")
        offset += _USTAR_BLOCK_BYTES
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = _pread_exact(
                descriptor,
                min(remaining, _READ_CHUNK_BYTES),
                offset,
                f"USTAR payload {path}",
            )
            digest.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        if not hmac.compare_digest(digest.hexdigest(), cast(str, record["sha256"])):
            raise ForagerMatchedV3ExternalStagingError(f"USTAR payload digest differs: {path}")
        padding = (-size) % _USTAR_BLOCK_BYTES
        if padding and any(_pread_exact(descriptor, padding, offset, f"USTAR padding {path}")):
            raise ForagerMatchedV3ExternalStagingError(f"USTAR payload padding is nonzero: {path}")
        offset += padding
        observed_paths.append(path)
    if any(
        _pread_exact(
            descriptor,
            2 * _USTAR_BLOCK_BYTES,
            offset,
            "USTAR end blocks",
        )
    ):
        raise ForagerMatchedV3ExternalStagingError("USTAR end blocks are nonzero")
    offset += 2 * _USTAR_BLOCK_BYTES
    expected_final_size = offset + ((-offset) % _USTAR_RECORD_BYTES)
    if expected_final_size != expected_size:
        raise ForagerMatchedV3ExternalStagingError("USTAR record padding length is not canonical")
    tail = expected_size - offset
    if tail and any(_pread_exact(descriptor, tail, offset, "USTAR record padding")):
        raise ForagerMatchedV3ExternalStagingError("USTAR record padding is nonzero")
    _validate_path_set(observed_paths, allowed_alias_groups=aliases)
    actual_sha256 = _hash_fd(descriptor, expected_size, "retained USTAR")
    after = os.fstat(descriptor)
    if not hmac.compare_digest(actual_sha256, expected_sha256) or _stat_signature(
        before
    ) != _stat_signature(after):
        raise ForagerMatchedV3ExternalStagingError(
            "retained USTAR digest or descriptor stability differs"
        )
    return manifest


def _memfd_flags() -> int:
    cloexec = getattr(os, "MFD_CLOEXEC", None)
    sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    if type(cloexec) is not int or type(sealing) is not int:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging requires sealed anonymous memfd support"
        )
    return cloexec | sealing


def _create_private_memfd(label: str) -> int:
    creator = getattr(os, "memfd_create", None)
    if creator is None:
        raise ForagerMatchedV3ExternalStagingError("external staging requires os.memfd_create")
    descriptor = -1
    try:
        descriptor = creator(label, _memfd_flags())
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ForagerMatchedV3ExternalStagingError(
            "private anonymous staging descriptor cannot be created"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or os.get_inheritable(descriptor)
    ):
        os.close(descriptor)
        raise ForagerMatchedV3ExternalStagingError(
            "private anonymous staging descriptor metadata is invalid"
        )
    return int(descriptor)


def _required_seals() -> int:
    names = ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    values = [getattr(fcntl, name, None) for name in names]
    if any(type(value) is not int for value in values):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging requires full memfd sealing support"
        )
    return sum(cast(int, value) for value in values)


def _seal_and_reopen_readonly(descriptor: int, *, expected_size: int) -> int:
    readonly = -1
    try:
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _required_seals())
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        before = os.fstat(descriptor)
        readonly = os.open(
            f"/proc/self/fd/{descriptor}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        after = os.fstat(readonly)
        readonly_flags = fcntl.fcntl(readonly, fcntl.F_GETFL)
        readonly_inheritable = os.get_inheritable(readonly)
    except OSError as exc:
        if readonly >= 0:
            try:
                os.close(readonly)
            except OSError:
                pass
        raise ForagerMatchedV3ExternalStagingError(
            "private staging descriptor cannot be sealed and reopened read-only"
        ) from exc
    if (
        seals & _required_seals() != _required_seals()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_size != expected_size
        or stat.S_IMODE(before.st_mode) != 0o400
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or readonly_flags & os.O_ACCMODE != os.O_RDONLY
        or readonly_inheritable
    ):
        os.close(readonly)
        raise ForagerMatchedV3ExternalStagingError(
            "sealed read-only staging descriptor metadata differs"
        )
    return readonly


def _sealed_bytes_fd(raw: bytes, label: str, maximum_bytes: int) -> int:
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        raise ForagerMatchedV3ExternalStagingError(f"{label} bytes exceed their exact bound")
    writable = _create_private_memfd(label)
    readonly = -1
    try:
        _write_all(writable, raw)
        readonly = _seal_and_reopen_readonly(writable, expected_size=len(raw))
        result = readonly
        try:
            os.close(writable)
        finally:
            writable = -1
        readonly = -1
        return result
    finally:
        if readonly >= 0:
            os.close(readonly)
        if writable >= 0:
            os.close(writable)


def _build_worker_request(
    *,
    execution_descriptor_raw: bytes,
    base_manifest_sha256: str,
    overlays: Sequence[_DerivedOverlay],
    stage_manifest_raw: bytes,
    stage_manifest_sha256: str,
) -> bytes:
    if tuple(item.candidate_id for item in overlays) != _CANDIDATE_IDS:
        raise ForagerMatchedV3ExternalStagingError(
            "worker request overlays do not cover the frozen order"
        )
    request = {
        "schema_version": EXTERNAL_STAGING_REQUEST_SCHEMA_VERSION,
        "execution_contract_descriptor_sha256": _EXECUTION_CONTRACT_SHA256,
        "execution_contract_descriptor_base64": base64.b64encode(execution_descriptor_raw).decode(
            "ascii"
        ),
        "base_manifest_sha256": base_manifest_sha256,
        "overlays": [
            {
                **item.manifest_record(),
                "derived_bytes_base64": base64.b64encode(item.raw).decode("ascii"),
            }
            for item in overlays
        ],
        "stage_manifest_sha256": stage_manifest_sha256,
        "stage_manifest_base64": base64.b64encode(stage_manifest_raw).decode("ascii"),
    }
    raw = _canonical_json(request)
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ForagerMatchedV3ExternalStagingError(
            "isolated staging request exceeds its byte bound"
        )
    return raw


def _strict_base64(value: Any, label: str, maximum_bytes: int) -> bytes:
    if type(value) is not str or len(value) > 4 * ((maximum_bytes + 2) // 3):
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not bounded base64")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not strict base64") from exc
    if len(raw) > maximum_bytes or base64.b64encode(raw).decode("ascii") != value:
        raise ForagerMatchedV3ExternalStagingError(f"{label} is not canonical base64")
    return raw


def _execution_records_from_parsed_descriptor(
    parsed: Mapping[str, Any],
) -> tuple[_ExecutionRecord, ...]:
    if (
        parsed.get("schema_version") != _EXECUTION_CONTRACT_SCHEMA_VERSION
        or parsed.get("candidate_count") != len(_CANDIDATE_IDS)
        or parsed.get("candidate_order") != list(_CANDIDATE_IDS)
    ):
        raise ForagerMatchedV3ExternalStagingError("worker execution descriptor identity differs")
    candidates = parsed.get("candidates")
    if type(candidates) is not list or len(candidates) != len(_CANDIDATE_IDS):
        raise ForagerMatchedV3ExternalStagingError("worker execution descriptor candidates differ")
    records: list[_ExecutionRecord] = []
    paths: list[str] = []
    for index, raw_candidate in enumerate(candidates):
        if (
            type(raw_candidate) is not dict
            or raw_candidate.get("candidate_id") != _CANDIDATE_IDS[index]
            or type(raw_candidate.get("configuration")) is not dict
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "worker execution descriptor candidate order differs"
            )
        configuration = cast(dict[str, Any], raw_candidate["configuration"])
        path = _validate_relative_path(
            configuration.get("original_relative_path"),
            f"worker execution candidate {index} path",
        )
        if (
            configuration.get("derived_configuration_staging_relative_path") != path
            or configuration.get("staging_preserves_original_path_below_experiments") is not True
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "worker execution descriptor staging relation differs"
            )
        records.append(
            _ExecutionRecord(
                candidate_id=_CANDIDATE_IDS[index],
                path=path,
                original_sha256=_require_sha256(
                    configuration.get("original_sha256"),
                    f"worker execution candidate {index} original digest",
                ),
                derived_sha256=_require_sha256(
                    configuration.get("derived_sha256"),
                    f"worker execution candidate {index} derived digest",
                ),
            )
        )
        paths.append(path)
    _validate_path_set(paths, allowed_alias_groups=())
    result = tuple(records)
    if (
        tuple(
            (item.candidate_id, item.path, item.original_sha256, item.derived_sha256)
            for item in result
        )
        != _FROZEN_EXECUTION_RECORDS
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "worker execution descriptor exact configuration records differ"
        )
    return result


def _read_bounded_fd(descriptor: int, maximum_bytes: int, label: str) -> bytes:
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ForagerMatchedV3ExternalStagingError(f"{label} descriptor is invalid") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
        or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _required_seals() != _required_seals()
    ):
        raise ForagerMatchedV3ExternalStagingError(f"{label} descriptor metadata differs")
    raw = _pread_exact(descriptor, metadata.st_size, 0, label)
    if _stat_signature(metadata) != _stat_signature(os.fstat(descriptor)):
        raise ForagerMatchedV3ExternalStagingError(f"{label} descriptor changed")
    return raw


@dataclass(frozen=True)
class _WorkerRequest:
    execution_records: tuple[_ExecutionRecord, ...]
    base_manifest_sha256: str
    overlay_by_path: Mapping[str, _DerivedOverlay]
    stage_manifest_raw: bytes
    stage_manifest_sha256: str


def _parse_worker_request(raw: bytes) -> _WorkerRequest:
    request = _strict_json_load(
        raw,
        maximum_bytes=_MAX_REQUEST_BYTES,
        trailing_newline=True,
    )
    if (
        set(request)
        != {
            "schema_version",
            "execution_contract_descriptor_sha256",
            "execution_contract_descriptor_base64",
            "base_manifest_sha256",
            "overlays",
            "stage_manifest_sha256",
            "stage_manifest_base64",
        }
        or request["schema_version"] != EXTERNAL_STAGING_REQUEST_SCHEMA_VERSION
    ):
        raise ForagerMatchedV3ExternalStagingError("isolated staging request fields differ")
    if request["execution_contract_descriptor_sha256"] != _EXECUTION_CONTRACT_SHA256:
        raise ForagerMatchedV3ExternalStagingError(
            "isolated staging request execution digest differs"
        )
    execution_raw = _strict_base64(
        request["execution_contract_descriptor_base64"],
        "worker execution descriptor",
        _MAX_MANIFEST_BYTES,
    )
    if not hmac.compare_digest(_sha256(execution_raw), _EXECUTION_CONTRACT_SHA256):
        raise ForagerMatchedV3ExternalStagingError("worker execution descriptor digest differs")
    execution_parsed = _strict_json_load(
        execution_raw,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=True,
    )
    execution_records = _execution_records_from_parsed_descriptor(execution_parsed)
    base_manifest_sha256 = _require_sha256(
        request["base_manifest_sha256"], "worker base manifest digest"
    )
    stage_manifest_sha256 = _require_sha256(
        request["stage_manifest_sha256"], "worker stage manifest digest"
    )
    stage_manifest_raw = _strict_base64(
        request["stage_manifest_base64"],
        "worker stage manifest",
        _MAX_MANIFEST_BYTES,
    )
    stage_manifest = parse_external_staging_manifest(
        stage_manifest_raw,
        expected_manifest_sha256=stage_manifest_sha256,
    )
    if (
        cast(dict[str, Any], stage_manifest["base_materialization"])["manifest_sha256"]
        != base_manifest_sha256
    ):
        raise ForagerMatchedV3ExternalStagingError("worker stage manifest base digest differs")
    raw_overlays = request["overlays"]
    if type(raw_overlays) is not list or len(raw_overlays) != len(_CANDIDATE_IDS):
        raise ForagerMatchedV3ExternalStagingError("worker overlay count differs")
    manifest_overlays = cast(list[dict[str, Any]], stage_manifest["configuration_overlays"])
    overlay_by_path: dict[str, _DerivedOverlay] = {}
    for index, raw_overlay in enumerate(raw_overlays):
        if type(raw_overlay) is not dict:
            raise ForagerMatchedV3ExternalStagingError("worker overlay is not an object")
        overlay = dict(raw_overlay)
        encoded = overlay.pop("derived_bytes_base64", None)
        if overlay != manifest_overlays[index]:
            raise ForagerMatchedV3ExternalStagingError("worker overlay differs from final manifest")
        execution = execution_records[index]
        if (
            overlay.get("candidate_id") != execution.candidate_id
            or overlay.get("path") != execution.path
            or overlay.get("original_sha256") != execution.original_sha256
            or overlay.get("derived_sha256") != execution.derived_sha256
            or overlay.get("transform_descriptor_sha256")
            != _expected_transform_sha256(execution.candidate_id)
            or overlay.get("archive_mode") != "0444"
        ):
            raise ForagerMatchedV3ExternalStagingError("worker overlay execution binding differs")
        derived_size = _require_exact_int(
            overlay.get("derived_size_bytes"),
            "worker overlay derived size",
            maximum=_MAX_MEMBER_BYTES,
        )
        raw_derived = _strict_base64(encoded, "worker overlay derived bytes", _MAX_MEMBER_BYTES)
        if (
            len(raw_derived) != derived_size
            or not hmac.compare_digest(_sha256(raw_derived), execution.derived_sha256)
            or execution.path in overlay_by_path
        ):
            raise ForagerMatchedV3ExternalStagingError("worker overlay bytes or path differ")
        overlay_by_path[execution.path] = _DerivedOverlay(
            candidate_id=execution.candidate_id,
            path=execution.path,
            original_size_bytes=_require_exact_int(
                overlay.get("original_size_bytes"),
                "worker overlay original size",
                maximum=_MAX_MEMBER_BYTES,
            ),
            original_sha256=execution.original_sha256,
            derived_size_bytes=derived_size,
            derived_sha256=execution.derived_sha256,
            transform_descriptor_sha256=cast(str, overlay["transform_descriptor_sha256"]),
            raw=raw_derived,
        )
    return _WorkerRequest(
        execution_records=execution_records,
        base_manifest_sha256=base_manifest_sha256,
        overlay_by_path=overlay_by_path,
        stage_manifest_raw=stage_manifest_raw,
        stage_manifest_sha256=stage_manifest_sha256,
    )


def _validate_worker_base_manifest(
    raw: bytes,
    expected_sha256: str,
) -> dict[str, Any]:
    if not hmac.compare_digest(_sha256(raw), expected_sha256):
        raise ForagerMatchedV3ExternalStagingError(
            "worker base materializer manifest digest differs"
        )
    manifest = _strict_json_load(
        raw,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        trailing_newline=False,
    )
    if (
        manifest.get("schema_version") != _MATERIALIZER_SCHEMA_VERSION
        or manifest.get("identity_sha256") != _MATERIALIZER_IDENTITY_SHA256
        or type(manifest.get("identity")) is not dict
        or type(manifest.get("source_tree")) is not dict
    ):
        raise ForagerMatchedV3ExternalStagingError("worker base materializer identity differs")
    claims = manifest.get("claims")
    if (
        type(claims) is not dict
        or not claims
        or any(value is not False for value in claims.values())
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "worker base materializer authority denial differs"
        )
    _base_file_records(manifest)
    _base_alias_groups(manifest)
    return manifest


def _worker_member_bytes(
    *,
    source_descriptor: int,
    path: str,
    inventory_record: Mapping[str, Any],
    base_by_path: Mapping[str, dict[str, Any]],
    base_manifest_raw: bytes,
    request: _WorkerRequest,
) -> bytes:
    if path == EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH:
        raw = base_manifest_raw
    elif path == EXTERNAL_STAGING_FINAL_MANIFEST_PATH:
        raw = request.stage_manifest_raw
    else:
        base = base_by_path.get(path)
        if base is None:
            raise ForagerMatchedV3ExternalStagingError(
                f"worker inventory path is absent from base materialization: {path}"
            )
        _output_mode, source_mode = _source_mode(base["git_mode"], f"worker source member {path}")
        original = _read_relative_regular_file(
            source_descriptor,
            path,
            expected_size=cast(int, base["materialized_size_bytes"]),
            expected_sha256=cast(str, base["materialized_sha256"]),
            expected_mode=0o644 if source_mode == 0o444 else 0o755,
            maximum_bytes=_MAX_MEMBER_BYTES,
        )
        overlay = request.overlay_by_path.get(path)
        if overlay is None:
            if inventory_record.get("provenance") != "materializer_v2_regular_file":
                raise ForagerMatchedV3ExternalStagingError(
                    f"worker source provenance differs: {path}"
                )
            raw = original
        else:
            if (
                base["git_mode"] != "100644"
                or len(original) != overlay.original_size_bytes
                or not hmac.compare_digest(_sha256(original), overlay.original_sha256)
                or inventory_record.get("provenance") != "derived_configuration_overlay"
            ):
                raise ForagerMatchedV3ExternalStagingError(
                    f"worker overlay original binding differs: {path}"
                )
            raw = overlay.raw
    if len(raw) != inventory_record.get("size_bytes") or not hmac.compare_digest(
        _sha256(raw), cast(str, inventory_record.get("sha256"))
    ):
        raise ForagerMatchedV3ExternalStagingError(
            f"worker member bytes differ from final inventory: {path}"
        )
    return raw


def _write_worker_archive(
    source_descriptor: int,
    base_manifest_raw: bytes,
    base_manifest: dict[str, Any],
    request: _WorkerRequest,
    output_descriptor: int,
) -> tuple[int, str, int]:
    stage_manifest = parse_external_staging_manifest(
        request.stage_manifest_raw,
        expected_manifest_sha256=request.stage_manifest_sha256,
    )
    base_records = _base_file_records(base_manifest)
    base_by_path = {cast(str, item["path"]): item for item in base_records}
    inventory, aliases = _expected_complete_inventory(request.stage_manifest_raw, stage_manifest)
    inventory_paths = {cast(str, item["path"]) for item in inventory}
    expected_paths = set(base_by_path) | {
        EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
        EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
    }
    if inventory_paths != expected_paths or set(request.overlay_by_path) != {
        record.path for record in request.execution_records
    }:
        raise ForagerMatchedV3ExternalStagingError(
            "worker final inventory does not preserve the exact source paths"
        )
    _validate_path_set(
        [cast(str, item["path"]) for item in inventory],
        allowed_alias_groups=aliases,
    )
    writer = _HashingWriter(output_descriptor, _MAX_ARCHIVE_BYTES)
    for record in inventory:
        path = cast(str, record["path"])
        mode_text = cast(str, record["mode"])
        mode = 0o444 if mode_text == "0444" else 0o555
        raw = _worker_member_bytes(
            source_descriptor=source_descriptor,
            path=path,
            inventory_record=record,
            base_by_path=base_by_path,
            base_manifest_raw=base_manifest_raw,
            request=request,
        )
        writer.write(_canonical_ustar_header(path, len(raw), mode))
        writer.write(raw)
        padding = (-len(raw)) % _USTAR_BLOCK_BYTES
        if padding:
            writer.write(bytes(padding))
    writer.write(bytes(2 * _USTAR_BLOCK_BYTES))
    record_padding = (-writer.size) % _USTAR_RECORD_BYTES
    if record_padding:
        writer.write(bytes(record_padding))
    return writer.size, writer.digest.hexdigest(), len(inventory)


def _worker_stage(
    source_descriptor: int,
    request_descriptor: int,
    receipt_descriptor: int,
) -> None:
    if len({source_descriptor, request_descriptor, receipt_descriptor}) != 3:
        raise ForagerMatchedV3ExternalStagingError("worker descriptors must be exact and distinct")
    source_before = os.fstat(source_descriptor)
    if not stat.S_ISDIR(source_before.st_mode):
        raise ForagerMatchedV3ExternalStagingError(
            "worker source capability is not a retained directory"
        )
    request_raw = _read_bounded_fd(
        request_descriptor, _MAX_REQUEST_BYTES, "isolated staging request"
    )
    request = _parse_worker_request(request_raw)
    base_manifest_raw = _read_relative_regular_file(
        source_descriptor,
        _MATERIALIZER_MANIFEST_FILENAME,
        expected_size=None,
        expected_sha256=request.base_manifest_sha256,
        expected_mode=0o644,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    base_manifest = _validate_worker_base_manifest(base_manifest_raw, request.base_manifest_sha256)
    stage_manifest = parse_external_staging_manifest(
        request.stage_manifest_raw,
        expected_manifest_sha256=request.stage_manifest_sha256,
    )
    base_binding = cast(dict[str, Any], stage_manifest["base_materialization"])
    if (
        base_binding["manifest_size_bytes"] != len(base_manifest_raw)
        or base_binding["manifest_sha256"] != request.base_manifest_sha256
        or base_binding["source_regular_file_count"] != len(_base_file_records(base_manifest))
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "worker base manifest differs from final staging manifest"
        )
    archive_size, archive_sha256, member_count = _write_worker_archive(
        source_descriptor,
        base_manifest_raw,
        base_manifest,
        request,
        1,
    )
    source_after = os.fstat(source_descriptor)
    if _directory_identity(source_before) != _directory_identity(source_after):
        raise ForagerMatchedV3ExternalStagingError("worker source capability identity changed")
    receipt = _canonical_json(
        {
            "schema_version": EXTERNAL_STAGING_RECEIPT_SCHEMA_VERSION,
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
            "member_count": member_count,
            "base_manifest_sha256": request.base_manifest_sha256,
            "stage_manifest_sha256": request.stage_manifest_sha256,
        }
    )
    if len(receipt) > _MAX_RECEIPT_BYTES:
        raise ForagerMatchedV3ExternalStagingError(
            "isolated staging receipt exceeds its byte bound"
        )
    _write_all(receipt_descriptor, receipt)


def _worker_entrypoint(argv: Sequence[str]) -> int:
    if (
        type(argv) is not list
        or len(argv) != 4
        or argv[0] != "--isolated-stage-worker"
        or any(_WORKER_ARGUMENT_RE.fullmatch(value) is None for value in argv[1:])
    ):
        return 64
    descriptors = tuple(int(value) for value in argv[1:])
    try:
        _worker_stage(*descriptors)
    except BaseException as exc:
        message = f"external staging worker failed: {type(exc).__name__}: {exc}\n"
        raw = message.encode("ascii", "backslashreplace")[:_MAX_STDERR_BYTES]
        try:
            _write_all(2, raw)
        except BaseException:
            pass
        return 2
    return 0


def _terminate_process_group(process: Any) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    deadline = time.monotonic() + _WORKER_CLEANUP_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


@dataclass(frozen=True)
class _WorkerResult:
    archive_size_bytes: int
    archive_sha256: str
    member_count: int


@dataclass(frozen=True)
class _BoundWorkerSource:
    descriptor: int
    signature: tuple[int, ...]
    sha256: str


def _implementation_source_path() -> str:
    module_file = __file__
    expected_named = type(module_file) is str and module_file.endswith(
        "alberta_framework/benchmarks/forager_matched_v3_external_staging.py"
    )
    expected_worker_fd = (
        type(module_file) is str
        and re.fullmatch(r"/proc/self/fd/[1-9][0-9]*", module_file) is not None
        and len(sys.argv) >= 2
        and sys.argv[1] == "--isolated-stage-worker"
    )
    if not expected_named and not expected_worker_fd:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging implementation source path is unavailable"
        )
    return module_file


def _open_bound_worker_source() -> _BoundWorkerSource:
    module_file = _implementation_source_path()
    named_descriptor = -1
    snapshot_descriptor = -1
    try:
        named_descriptor = os.open(module_file, _file_flags())
        metadata = os.fstat(named_descriptor)
    except OSError as exc:
        if named_descriptor >= 0:
            try:
                os.close(named_descriptor)
            except OSError:
                pass
        raise ForagerMatchedV3ExternalStagingError(
            "external staging implementation source cannot be retained"
        ) from exc
    try:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > _MAX_SOURCE_BYTES
            or os.get_inheritable(named_descriptor)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging implementation source metadata is invalid"
            )
        raw = _pread_exact(
            named_descriptor,
            metadata.st_size,
            0,
            "staging implementation source",
        )
        digest = _sha256(raw)
        after = os.fstat(named_descriptor)
        if (
            _stat_signature(metadata) != _stat_signature(after)
            or _stat_signature(metadata) != _IMPORTED_IMPLEMENTATION_SOURCE_SIGNATURE
            or not hmac.compare_digest(digest, _IMPORTED_IMPLEMENTATION_SOURCE_SHA256)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging implementation changed since module import"
            )
        snapshot_descriptor = _sealed_bytes_fd(
            raw,
            "alberta-external-stage-worker-source",
            _MAX_SOURCE_BYTES,
        )
        snapshot_metadata = os.fstat(snapshot_descriptor)
        snapshot_flags = fcntl.fcntl(snapshot_descriptor, fcntl.F_GETFL)
        snapshot_descriptor_flags = fcntl.fcntl(snapshot_descriptor, fcntl.F_GETFD)
        snapshot_seals = fcntl.fcntl(snapshot_descriptor, fcntl.F_GET_SEALS)
        if (
            not stat.S_ISREG(snapshot_metadata.st_mode)
            or snapshot_metadata.st_nlink != 0
            or snapshot_metadata.st_size != len(raw)
            or stat.S_IMODE(snapshot_metadata.st_mode) != 0o400
            or snapshot_flags & os.O_ACCMODE != os.O_RDONLY
            or snapshot_descriptor_flags & fcntl.FD_CLOEXEC == 0
            or snapshot_seals & _required_seals() != _required_seals()
            or os.get_inheritable(snapshot_descriptor)
            or not hmac.compare_digest(
                _hash_fd(
                    snapshot_descriptor,
                    snapshot_metadata.st_size,
                    "sealed staging implementation source",
                ),
                digest,
            )
            or _stat_signature(os.fstat(snapshot_descriptor)) != _stat_signature(snapshot_metadata)
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "sealed external staging implementation source differs"
            )
        result = _BoundWorkerSource(
            descriptor=snapshot_descriptor,
            signature=_stat_signature(snapshot_metadata),
            sha256=digest,
        )
        try:
            os.close(named_descriptor)
        finally:
            named_descriptor = -1
        snapshot_descriptor = -1
        return result
    finally:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        if named_descriptor >= 0:
            os.close(named_descriptor)


def _reverify_bound_worker_source(source: _BoundWorkerSource) -> None:
    try:
        metadata = os.fstat(source.descriptor)
        flags = fcntl.fcntl(source.descriptor, fcntl.F_GETFL)
        descriptor_flags = fcntl.fcntl(source.descriptor, fcntl.F_GETFD)
        seals = fcntl.fcntl(source.descriptor, fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(source.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging implementation source descriptor became inaccessible"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 0
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or flags & os.O_ACCMODE != os.O_RDONLY
        or descriptor_flags & fcntl.FD_CLOEXEC == 0
        or seals & _required_seals() != _required_seals()
        or inheritable
        or _stat_signature(metadata) != source.signature
        or not hmac.compare_digest(
            _hash_fd(
                source.descriptor,
                metadata.st_size,
                "staging implementation source",
            ),
            source.sha256,
        )
        or _stat_signature(os.fstat(source.descriptor)) != source.signature
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "external staging implementation source descriptor drifted"
        )


def _open_imported_implementation_source(
    module_file: str,
) -> tuple[int, os.stat_result]:
    """Open named source safely or duplicate the exact inherited worker FD."""

    worker_match = re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", module_file)
    descriptor = -1
    if worker_match is None:
        try:
            descriptor = os.open(module_file, _file_flags())
            return descriptor, os.fstat(descriptor)
        except OSError as exc:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise ForagerMatchedV3ExternalStagingError(
                "external staging implementation source cannot be bound at import"
            ) from exc

    inherited = int(worker_match.group(1))
    if inherited <= 2:
        raise ForagerMatchedV3ExternalStagingError(
            "external staging worker source descriptor is not private"
        )
    try:
        before = os.fstat(inherited)
        inherited_flags = fcntl.fcntl(inherited, fcntl.F_GETFL)
        inherited_seals = fcntl.fcntl(inherited, fcntl.F_GET_SEALS)
        descriptor = fcntl.fcntl(inherited, fcntl.F_DUPFD_CLOEXEC, 3)
        after = os.fstat(inherited)
        duplicate = os.fstat(descriptor)
        duplicate_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        duplicate_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ForagerMatchedV3ExternalStagingError(
            "external staging inherited implementation source cannot be retained"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_size < 1
        or before.st_size > _MAX_SOURCE_BYTES
        or stat.S_IMODE(before.st_mode) != 0o400
        or inherited_flags & os.O_ACCMODE != os.O_RDONLY
        or inherited_seals & _required_seals() != _required_seals()
        or _stat_signature(before) != _stat_signature(after)
        or _stat_signature(before) != _stat_signature(duplicate)
        or duplicate_flags & os.O_ACCMODE != os.O_RDONLY
        or duplicate_seals & _required_seals() != _required_seals()
        or inheritable
    ):
        os.close(descriptor)
        raise ForagerMatchedV3ExternalStagingError(
            "external staging inherited implementation source identity drifted"
        )
    return descriptor, duplicate


def _capture_imported_implementation_source_identity() -> tuple[tuple[int, ...], str]:
    module_file = _implementation_source_path()
    descriptor, metadata = _open_imported_implementation_source(module_file)
    try:
        expected_link_count = (
            0 if re.fullmatch(r"/proc/self/fd/[1-9][0-9]*", module_file) is not None else 1
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != expected_link_count
            or metadata.st_size < 1
            or metadata.st_size > _MAX_SOURCE_BYTES
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging implementation import identity is invalid"
            )
        digest = _hash_fd(descriptor, metadata.st_size, "staging implementation source")
        after = os.fstat(descriptor)
        if _stat_signature(metadata) != _stat_signature(after):
            raise ForagerMatchedV3ExternalStagingError(
                "external staging implementation changed during module import"
            )
        return _stat_signature(metadata), digest
    finally:
        os.close(descriptor)


(
    _IMPORTED_IMPLEMENTATION_SOURCE_SIGNATURE,
    _IMPORTED_IMPLEMENTATION_SOURCE_SHA256,
) = _capture_imported_implementation_source_identity()


def _run_isolated_worker(
    *,
    source_descriptor: int,
    request_descriptor: int,
    output_descriptor: int,
    base_manifest_sha256: str,
    stage_manifest_sha256: str,
    expected_member_count: int,
) -> _WorkerResult:
    import subprocess

    if (
        any(
            type(value) is not int or value <= 2
            for value in (
                source_descriptor,
                request_descriptor,
                output_descriptor,
            )
        )
        or len({source_descriptor, request_descriptor, output_descriptor}) != 3
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "isolated staging descriptors must be distinct private descriptors"
        )
    receipt_read, receipt_write = os.pipe2(getattr(os, "O_CLOEXEC", 0))
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stderr = bytearray()
    receipt = bytearray()
    archive_digest = hashlib.sha256()
    archive_size = 0
    surfaced_error: BaseException | None = None
    worker_source: _BoundWorkerSource | None = None
    try:
        worker_source = _open_bound_worker_source()
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{worker_source.descriptor}",
            "--isolated-stage-worker",
            str(source_descriptor),
            str(request_descriptor),
            str(receipt_write),
        ]
        environment = {
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=(
                    source_descriptor,
                    request_descriptor,
                    receipt_write,
                    worker_source.descriptor,
                ),
                cwd="/",
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            raise ForagerMatchedV3ExternalStagingError(
                "isolated staging child could not start"
            ) from exc
        os.close(receipt_write)
        receipt_write = -1
        if process.stdout is None or process.stderr is None:
            raise ForagerMatchedV3ExternalStagingError(
                "isolated staging child pipes are unavailable"
            )
        streams = {
            process.stdout.fileno(): "archive",
            process.stderr.fileno(): "stderr",
            receipt_read: "receipt",
        }
        for descriptor, label in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, label)
        deadline = time.monotonic() + _WORKER_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ForagerMatchedV3ExternalStagingError(
                    "isolated staging child exceeded its wall-clock bound"
                )
            events = selector.select(min(remaining, 0.1))
            if not events and process.poll() is not None:
                _terminate_process_group(process)
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _mask in events:
                descriptor = key.fd
                try:
                    chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                except InterruptedError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                label = key.data
                if label == "archive":
                    archive_size += len(chunk)
                    if archive_size > _MAX_ARCHIVE_BYTES:
                        raise ForagerMatchedV3ExternalStagingError(
                            "isolated staging archive exceeds its byte bound"
                        )
                    _write_all(output_descriptor, chunk)
                    archive_digest.update(chunk)
                elif label == "stderr":
                    stderr.extend(chunk)
                    if len(stderr) > _MAX_STDERR_BYTES:
                        raise ForagerMatchedV3ExternalStagingError(
                            "isolated staging stderr exceeds its byte bound"
                        )
                else:
                    receipt.extend(chunk)
                    if len(receipt) > _MAX_RECEIPT_BYTES:
                        raise ForagerMatchedV3ExternalStagingError(
                            "isolated staging receipt exceeds its byte bound"
                        )
        returncode = process.wait(timeout=_WORKER_CLEANUP_GRACE_SECONDS)
        _reverify_bound_worker_source(worker_source)
        if returncode != 0 or stderr:
            detail = bytes(stderr).decode("ascii", "replace").strip()
            raise ForagerMatchedV3ExternalStagingError(
                "isolated staging child failed"
                + (f": {detail}" if detail else f" with status {returncode}")
            )
        receipt_value = _strict_json_load(
            bytes(receipt),
            maximum_bytes=_MAX_RECEIPT_BYTES,
            trailing_newline=True,
        )
        expected_receipt_keys = {
            "schema_version",
            "archive_size_bytes",
            "archive_sha256",
            "member_count",
            "base_manifest_sha256",
            "stage_manifest_sha256",
        }
        actual_archive_sha256 = archive_digest.hexdigest()
        if (
            set(receipt_value) != expected_receipt_keys
            or receipt_value["schema_version"] != EXTERNAL_STAGING_RECEIPT_SCHEMA_VERSION
            or receipt_value["archive_size_bytes"] != archive_size
            or receipt_value["archive_sha256"] != actual_archive_sha256
            or receipt_value["member_count"] != expected_member_count
            or receipt_value["base_manifest_sha256"] != base_manifest_sha256
            or receipt_value["stage_manifest_sha256"] != stage_manifest_sha256
            or archive_size <= 0
            or archive_size % _USTAR_RECORD_BYTES != 0
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "isolated staging receipt differs from the drained archive"
            )
        return _WorkerResult(
            archive_size_bytes=archive_size,
            archive_sha256=actual_archive_sha256,
            member_count=expected_member_count,
        )
    except BaseException as exc:
        surfaced_error = exc
        if process is not None:
            _terminate_process_group(process)
        raise
    finally:
        selector.close()
        if receipt_write >= 0:
            os.close(receipt_write)
        try:
            os.close(receipt_read)
        except OSError as cleanup_error:
            if surfaced_error is None:
                raise ForagerMatchedV3ExternalStagingError(
                    "isolated staging receipt cleanup failed"
                ) from cleanup_error
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            if process.poll() is None:
                _terminate_process_group(process)
            try:
                process.wait(timeout=_WORKER_CLEANUP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
        if worker_source is not None:
            os.close(worker_source.descriptor)


_RETAINED_BUNDLE_CREATION_TOKEN: Final = object()


class RetainedExternalStagingBundle:
    """PID-bound, sealed, read-only descriptor capability for one staged USTAR.

    This content capability is not execution authority.  It has no extraction,
    publication, runner, seed, result, or acceptance method.
    """

    __slots__ = (
        "_archive_sha256",
        "_archive_size_bytes",
        "_descriptor",
        "_device",
        "_inode",
        "_manifest_raw",
        "_manifest_sha256",
        "_owner_pid",
    )

    def __init__(
        self,
        creation_token: object,
        descriptor: int,
        device: int,
        inode: int,
        archive_size_bytes: int,
        archive_sha256: str,
        manifest_raw: bytes,
        manifest_sha256: str,
    ) -> None:
        if creation_token is not _RETAINED_BUNDLE_CREATION_TOKEN:
            raise TypeError("retained staging bundles require the staging context")
        self._descriptor = descriptor
        self._device = device
        self._inode = inode
        self._archive_size_bytes = archive_size_bytes
        self._archive_sha256 = archive_sha256
        self._manifest_raw = manifest_raw
        self._manifest_sha256 = manifest_sha256
        self._owner_pid = os.getpid()

    def __reduce__(self) -> Never:
        raise TypeError("retained staging bundles cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("retained staging bundles cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("retained staging bundles cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("retained staging bundles cannot be copied")

    def _invalidate(self, *, close_if_owned: bool) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        if descriptor < 0 or not close_if_owned:
            return
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            return
        if (metadata.st_dev, metadata.st_ino) != (self._device, self._inode):
            return
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _require_active(self) -> int:
        if os.getpid() != self._owner_pid:
            self._invalidate(close_if_owned=False)
            raise ForagerMatchedV3ExternalStagingError(
                "retained staging bundle is invalid after a PID change"
            )
        descriptor = self._descriptor
        if descriptor < 0:
            raise ForagerMatchedV3ExternalStagingError("retained staging bundle is closed")
        try:
            metadata = os.fstat(descriptor)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        except OSError as exc:
            self._invalidate(close_if_owned=False)
            raise ForagerMatchedV3ExternalStagingError(
                "retained staging bundle descriptor became inaccessible"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != self._archive_size_bytes
            or (metadata.st_dev, metadata.st_ino) != (self._device, self._inode)
            or flags & os.O_ACCMODE != os.O_RDONLY
            or descriptor_flags & fcntl.FD_CLOEXEC == 0
            or seals & _required_seals() != _required_seals()
            or os.get_inheritable(descriptor)
        ):
            same_identity = (metadata.st_dev, metadata.st_ino) == (self._device, self._inode)
            self._invalidate(close_if_owned=same_identity)
            raise ForagerMatchedV3ExternalStagingError(
                "retained staging bundle descriptor identity drifted"
            )
        return descriptor

    @property
    def closed(self) -> bool:
        """Return whether this retained context has closed or invalidated the bundle."""

        if self._descriptor >= 0 and os.getpid() != self._owner_pid:
            self._invalidate(close_if_owned=False)
        return self._descriptor < 0

    @property
    def proc_fd_path(self) -> str:
        """Return the procfs descriptor path while this context is active."""

        return f"/proc/self/fd/{self._require_active()}"

    @property
    def subprocess_pass_fds(self) -> tuple[int, ...]:
        """Return the sole descriptor fact a separately authorized caller may pass."""

        return (self._require_active(),)

    @property
    def owner_pid(self) -> int:
        """Return the sole process ID in which this capability is valid."""

        self._require_active()
        return self._owner_pid

    @property
    def archive_size_bytes(self) -> int:
        """Return the exact canonical USTAR byte length."""

        self._require_active()
        return self._archive_size_bytes

    @property
    def archive_sha256(self) -> str:
        """Return the exact canonical USTAR SHA-256."""

        self._require_active()
        return self._archive_sha256

    @property
    def manifest_bytes(self) -> bytes:
        """Return the exact embedded final staging-manifest bytes."""

        self._require_active()
        return self._manifest_raw

    @property
    def manifest_sha256(self) -> str:
        """Return the exact embedded final staging-manifest SHA-256."""

        self._require_active()
        return self._manifest_sha256

    def manifest(self) -> dict[str, Any]:
        """Return a detached strict replay of the nonauthorizing final manifest."""

        self._require_active()
        return parse_external_staging_manifest(
            self._manifest_raw,
            expected_manifest_sha256=self._manifest_sha256,
        )

    def reverify(self) -> dict[str, Any]:
        """Reverify descriptor identity, full USTAR bytes, and the final manifest."""

        descriptor = self._require_active()
        try:
            manifest = _verify_canonical_ustar_fd(
                descriptor,
                expected_size=self._archive_size_bytes,
                expected_sha256=self._archive_sha256,
                manifest_raw=self._manifest_raw,
                manifest_sha256=self._manifest_sha256,
            )
            self._require_active()
            return manifest
        except BaseException:
            self._invalidate(close_if_owned=True)
            raise

    def close(self) -> None:
        """Close and permanently invalidate this retained bundle."""

        self._invalidate(close_if_owned=True)


def _capability_source_descriptor(capability: Any, materializer: Any) -> int:
    if type(capability) is not materializer.RetainedExternalMaterializationTree:
        raise ForagerMatchedV3ExternalStagingError(
            "staging requires one exact live retained materializer-v2 capability"
        )
    try:
        pass_fds = capability.subprocess_pass_fds
    except (AttributeError, materializer.ExternalMaterializationError) as exc:
        raise ForagerMatchedV3ExternalStagingError(
            "retained materializer-v2 capability is not active"
        ) from exc
    if (
        type(pass_fds) is not tuple
        or len(pass_fds) != 1
        or type(pass_fds[0]) is not int
        or pass_fds[0] <= 2
    ):
        raise ForagerMatchedV3ExternalStagingError(
            "retained materializer-v2 descriptor fact is invalid"
        )
    return pass_fds[0]


@contextmanager
def _stage_matched_v3_external_workload(
    capability: Any,
) -> Iterator[RetainedExternalStagingBundle]:
    from alberta_framework.benchmarks import (
        forager_matched_v3_external_materialization as materializer,
    )

    execution_raw, execution_records = _validated_execution_records()
    source_descriptor = _capability_source_descriptor(capability, materializer)
    root_duplicate = -1
    request_descriptor = -1
    output_writable = -1
    output_readonly = -1
    bundle: RetainedExternalStagingBundle | None = None
    try:
        try:
            pre_manifest = capability.reverify()
        except (AssertionError, materializer.ExternalMaterializationError) as exc:
            raise ForagerMatchedV3ExternalStagingError(
                "retained materializer-v2 capability failed pre-staging verification"
            ) from exc
        if source_descriptor != _capability_source_descriptor(capability, materializer):
            raise ForagerMatchedV3ExternalStagingError(
                "retained materializer-v2 descriptor drifted before staging"
            )
        root_duplicate = _duplicate_directory_descriptor(
            source_descriptor, "retained materializer-v2 source"
        )
        base_manifest_raw = _read_relative_regular_file(
            root_duplicate,
            _MATERIALIZER_MANIFEST_FILENAME,
            expected_size=None,
            expected_sha256=None,
            expected_mode=0o644,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
        base_manifest_sha256 = _sha256(base_manifest_raw)
        try:
            base_manifest = materializer.parse_matched_v3_external_materialization_manifest(
                base_manifest_raw,
                expected_manifest_sha256=base_manifest_sha256,
            )
        except (AssertionError, materializer.ExternalMaterializationError) as exc:
            raise ForagerMatchedV3ExternalStagingError(
                "retained source does not bind the exact materializer-v2 production identity"
            ) from exc
        if _canonical_base_json(pre_manifest) != base_manifest_raw or pre_manifest != base_manifest:
            raise ForagerMatchedV3ExternalStagingError(
                "retained capability replay differs from its exact manifest file"
            )
        overlays = _derive_overlays(root_duplicate, base_manifest, execution_records)
        (
            stage_manifest_raw,
            stage_manifest_sha256,
            inventory,
            _aliases,
        ) = _build_stage_manifest(base_manifest_raw, base_manifest, overlays)
        request_raw = _build_worker_request(
            execution_descriptor_raw=execution_raw,
            base_manifest_sha256=base_manifest_sha256,
            overlays=overlays,
            stage_manifest_raw=stage_manifest_raw,
            stage_manifest_sha256=stage_manifest_sha256,
        )
        request_descriptor = _sealed_bytes_fd(
            request_raw, "alberta-external-stage-request", _MAX_REQUEST_BYTES
        )
        output_writable = _create_private_memfd("alberta-external-stage-bundle")
        result = _run_isolated_worker(
            source_descriptor=source_descriptor,
            request_descriptor=request_descriptor,
            output_descriptor=output_writable,
            base_manifest_sha256=base_manifest_sha256,
            stage_manifest_sha256=stage_manifest_sha256,
            expected_member_count=len(inventory) + 1,
        )
        _verify_canonical_ustar_fd(
            output_writable,
            expected_size=result.archive_size_bytes,
            expected_sha256=result.archive_sha256,
            manifest_raw=stage_manifest_raw,
            manifest_sha256=stage_manifest_sha256,
        )
        try:
            post_manifest = capability.reverify()
        except (AssertionError, materializer.ExternalMaterializationError) as exc:
            raise ForagerMatchedV3ExternalStagingError(
                "retained materializer-v2 capability failed post-staging verification"
            ) from exc
        if (
            post_manifest != pre_manifest
            or source_descriptor != _capability_source_descriptor(capability, materializer)
            or _read_relative_regular_file(
                root_duplicate,
                _MATERIALIZER_MANIFEST_FILENAME,
                expected_size=len(base_manifest_raw),
                expected_sha256=base_manifest_sha256,
                expected_mode=0o644,
                maximum_bytes=_MAX_MANIFEST_BYTES,
            )
            != base_manifest_raw
        ):
            raise ForagerMatchedV3ExternalStagingError(
                "retained materializer-v2 source changed across staging"
            )
        output_readonly = _seal_and_reopen_readonly(
            output_writable, expected_size=result.archive_size_bytes
        )
        os.close(output_writable)
        output_writable = -1
        metadata = os.fstat(output_readonly)
        bundle = RetainedExternalStagingBundle(
            _RETAINED_BUNDLE_CREATION_TOKEN,
            output_readonly,
            metadata.st_dev,
            metadata.st_ino,
            result.archive_size_bytes,
            result.archive_sha256,
            stage_manifest_raw,
            stage_manifest_sha256,
        )
        output_readonly = -1
        bundle.reverify()
        yield bundle
    finally:
        if bundle is not None:
            bundle.close()
        if output_readonly >= 0:
            os.close(output_readonly)
        if output_writable >= 0:
            os.close(output_writable)
        if request_descriptor >= 0:
            os.close(request_descriptor)
        if root_duplicate >= 0:
            os.close(root_duplicate)


def stage_matched_v3_external_workload(
    retained_materialization: Any,
) -> AbstractContextManager[RetainedExternalStagingBundle]:
    """Stage the exact retained production source into one sealed USTAR context.

    The argument must be the live capability yielded by
    ``retain_verified_matched_v3_external_materialization_tree``.  Paths,
    serialized manifests, descriptor integers, and caller-authored overlays are
    deliberately not accepted.
    """

    return _stage_matched_v3_external_workload(retained_materialization)


__all__ = [
    "EXTERNAL_STAGING_ATTESTATION_NAMESPACE",
    "EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256",
    "EXTERNAL_STAGING_FINAL_MANIFEST_PATH",
    "EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION",
    "EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH",
    "EXTERNAL_STAGING_STATUS",
    "ForagerMatchedV3ExternalStagingError",
    "RetainedExternalStagingBundle",
    "canonical_external_staging_contract_descriptor_bytes",
    "external_staging_contract_descriptor",
    "external_staging_contract_descriptor_sha256",
    "parse_external_staging_contract_descriptor",
    "parse_external_staging_manifest",
    "stage_matched_v3_external_workload",
]


if __name__ == "__main__":
    raise SystemExit(_worker_entrypoint(sys.argv[1:]))
