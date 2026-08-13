"""Frozen source pins for future, nonpromoting Forager memory comparators.

This module is a descriptor-only research registry.  It records exact upstream
source identities and adaptation concepts that may be considered in a future
open-development lane.  It neither extends the frozen matched-v3 28-candidate
universe nor authorizes source download, import, execution, qualification,
promotion, acceptance, or performance claims.

The inventory verifier accepts only canonical digest-and-size records for the
declared relevant source subset.  A separate read-only verifier can hash those
exact declared files below a caller-supplied checkout.  Neither verifier fetches
source, authenticates an archive or full tree, or executes an upstream
implementation.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, NoReturn, cast

MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_memory_comparator_development_sources.v2"
)
SOURCE_SUBSET_INVENTORY_SCHEMA_VERSION: Final = (
    "alberta.forager_memory_comparator_source_subset_inventory.v1"
)
MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_STATUS: Final = (
    "source_pins_registered_descriptor_only_unexecuted_unqualified_nonpromoting"
)

_MAX_DESCRIPTOR_BYTES: Final = 2 * 1024 * 1024
_MAX_INVENTORY_BYTES: Final = 512 * 1024
_MAX_SOURCE_FILE_BYTES: Final = 64 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_SOURCE_FILES: Final = 256
_MAX_PATH_BYTES: Final = 512
_MAX_PATH_COMPONENT_BYTES: Final = 255
_MAX_SOURCE_ROOT_PATH_BYTES: Final = 4096
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_BYTES: Final = 64 * 1024
_MAX_JSON_INTEGER_DIGITS: Final = 19
_READ_CHUNK_BYTES: Final = 1024 * 1024

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[a-z][a-z0-9_]*\Z")


class ForagerMemoryComparatorDevelopmentSourcesError(ValueError):
    """The descriptor or a declared source-subset inventory failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMemoryComparatorDevelopmentSourcesError(message)


def _require_exact_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a nonempty exact string")
    return value


def _require_identifier(value: object, label: str) -> str:
    text = _require_exact_string(value, label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        _fail(f"{label} must be a canonical lowercase identifier")
    return text


def _require_sha256(value: object, label: str) -> str:
    text = _require_exact_string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return text


def _require_git_sha1(value: object, label: str) -> str:
    text = _require_exact_string(value, label)
    if _GIT_SHA1_RE.fullmatch(text) is None:
        _fail(f"{label} must be a lowercase Git SHA-1")
    return text


def _require_positive_int(value: object, label: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(f"{label} must be a positive bounded exact integer")
    return value


def _validate_source_path(value: object) -> str:
    path = _require_exact_string(value, "source path")
    if (
        not path.isascii()
        or "\x00" in path
        or "\\" in path
        or path.startswith("/")
        or path.endswith("/")
        or len(path.encode("utf-8")) > _MAX_PATH_BYTES
        or unicodedata.normalize("NFC", path) != path
    ):
        _fail("source path must be bounded canonical ASCII POSIX text")
    pure = PurePosixPath(path)
    if str(pure) != path or any(
        component in {"", ".", ".."} or len(component.encode("utf-8")) > _MAX_PATH_COMPONENT_BYTES
        for component in pure.parts
    ):
        _fail("source path contains a noncanonical component or alias")
    return path


@dataclass(frozen=True, slots=True)
class SourceFilePin:
    """Exact digest and byte length for one relevant upstream source file."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_source_path(self.path)
        _require_sha256(self.sha256, f"source SHA-256 for {self.path}")
        _require_positive_int(
            self.size_bytes,
            f"source size for {self.path}",
            maximum=_MAX_SOURCE_FILE_BYTES,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class SourceFamilyPin:
    """One immutable upstream archive identity and its declared source subset."""

    family_id: str
    url: str
    commit_git_sha1: str
    tree_git_sha1: str
    archive_sha256: str
    archive_size_bytes: int
    license_spdx: str
    source_files: tuple[SourceFilePin, ...]
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "source family ID")
        url = _require_exact_string(self.url, "upstream URL")
        if (
            not url.isascii()
            or not url.startswith("https://github.com/")
            or url.endswith("/")
            or "?" in url
            or "#" in url
        ):
            _fail("upstream URL must be an exact canonical HTTPS GitHub repository URL")
        _require_git_sha1(self.commit_git_sha1, "upstream commit")
        _require_git_sha1(self.tree_git_sha1, "upstream tree")
        _require_sha256(self.archive_sha256, "upstream archive SHA-256")
        _require_positive_int(
            self.archive_size_bytes,
            "upstream archive size",
            maximum=_MAX_ARCHIVE_BYTES,
        )
        if self.license_spdx not in {"Apache-2.0", "MIT"}:
            _fail("source-family SPDX license is not in the frozen registry")
        if (
            type(self.source_files) is not tuple
            or not self.source_files
            or len(self.source_files) > _MAX_SOURCE_FILES
            or any(type(item) is not SourceFilePin for item in self.source_files)
        ):
            _fail("source-family subset must be a bounded tuple of exact file pins")
        paths = tuple(item.path for item in self.source_files)
        if paths != tuple(sorted(paths, key=lambda path: path.encode("utf-8"))):
            _fail("source-family paths must use ascending UTF-8 byte order")
        _reject_path_aliases(paths)
        if (
            type(self.candidate_ids) is not tuple
            or not self.candidate_ids
            or any(_IDENTIFIER_RE.fullmatch(item) is None for item in self.candidate_ids)
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
        ):
            _fail("source-family candidate IDs must be unique canonical tuples")

    def upstream_dict(self) -> dict[str, Any]:
        return {
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "commit_git_sha1": self.commit_git_sha1,
            "license_spdx": self.license_spdx,
            "tree_git_sha1": self.tree_git_sha1,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentCandidateConcept:
    """A source-derived concept with no run, qualification, or claim authority."""

    candidate_id: str
    family_id: str
    design_note: str

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "development candidate ID")
        _require_identifier(self.family_id, "development source family ID")
        note = _require_exact_string(self.design_note, "development design note")
        if len(note.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
            _fail("development design note is too large")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "classification": "open_development_nonpromoting",
            "design_note": self.design_note,
            "execution_authority_granted": False,
            "family_id": self.family_id,
            "observation_access": "forager_image_only",
            "promotion_allowed": False,
            "qualification_granted": False,
            "sota_claim_allowed": False,
            "source_relationship": "source_derived_not_exact_execution",
        }


@dataclass(frozen=True, slots=True)
class SourceSubsetVerification:
    """Identity of one accepted canonical digest-and-size subset inventory."""

    family_id: str
    inventory_sha256: str
    source_file_count: int
    source_total_size_bytes: int
    source_bytes_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "inventory_sha256": self.inventory_sha256,
            "source_file_count": self.source_file_count,
            "source_total_size_bytes": self.source_total_size_bytes,
            "archive_authenticated": False,
            "execution_authority_granted": False,
            "source_bytes_verified": self.source_bytes_verified,
        }


def _reject_path_aliases(paths: Sequence[str]) -> None:
    seen_exact: set[str] = set()
    seen_portable: set[str] = set()
    for path in paths:
        canonical = _validate_source_path(path)
        portable = unicodedata.normalize("NFC", canonical).casefold()
        if canonical in seen_exact or portable in seen_portable:
            _fail("source subset contains a duplicate or portable path alias")
        seen_exact.add(canonical)
        seen_portable.add(portable)


FROZEN_MATCHED_V3_CANDIDATE_IDS: Final = (
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
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
    "random_policy",
    "search_nearest",
    "search_oracle",
)


def _pin(path: str, sha256: str, size_bytes: int) -> SourceFilePin:
    return SourceFilePin(path=path, sha256=sha256, size_bytes=size_bytes)


_SOURCE_FAMILIES: Final = (
    SourceFamilyPin(
        family_id="pobax_ld_gtrxl",
        url="https://github.com/taodav/pobax",
        commit_git_sha1="a5e1d62d14e4efe783885b9d4f19cffa2a568eec",
        tree_git_sha1="d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6",
        archive_sha256="f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9",
        archive_size_bytes=1_699_840,
        license_spdx="Apache-2.0",
        source_files=(
            _pin(
                "LICENSE",
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
                11_357,
            ),
            _pin(
                "pobax/algos/ppo.py",
                "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153",
                19_864,
            ),
            _pin(
                "pobax/algos/transformer_xl.py",
                "e51c3c9530963e902bbab4f23683c6e4c9a9f0a6399ada624cab5da6c7e462bf",
                22_308,
            ),
            _pin(
                "pobax/config.py",
                "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c",
                7_047,
            ),
            _pin(
                "pobax/models/actor_critic.py",
                "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f",
                2_374,
            ),
            _pin(
                "pobax/models/discrete.py",
                "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c",
                11_026,
            ),
            _pin(
                "pobax/models/rel_multi_head.py",
                "354e8ad9a0e7efdc8eb04ee1104c694173bf7594d4a76631fe384f144ad2c333",
                23_241,
            ),
            _pin(
                "pobax/models/transformerXL.py",
                "bfc1b5d734be61e8ca3ac4bfb0da0d992a6fd131982a3b42c7870102c5e762cd",
                6_562,
            ),
        ),
        candidate_ids=("adapted_ppo_gru_ld", "adapted_ppo_gtrxl"),
    ),
    SourceFamilyPin(
        family_id="agalite",
        url="https://github.com/subho406/agalite",
        commit_git_sha1="101acbecc121a258ad8f7e58e2f782f546674979",
        tree_git_sha1="c76616f5ac4fba0bfd700095f1c174f11144471e",
        archive_sha256="2784a2491b0844cf902f3f6e9896b18730a01ca7ea72ebea490ace780f914ecb",
        archive_size_bytes=56_142,
        license_spdx="Apache-2.0",
        source_files=(
            _pin(
                "LICENSE",
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
                11_357,
            ),
            _pin(
                "config_pure/craftax/arelit.yaml",
                "949f146735e0ef3090a184f125b1823f4a7af1ba6a4ee28f2bb01074a02ced79",
                533,
            ),
            _pin(
                "src/models/agalite/agalite.py",
                "00d921f46740e43aed9e444c51852b0e7fdb80cd489550e1c815d2c70e00a89b",
                12_103,
            ),
            _pin(
                "src/models/agalite/kernels.py",
                "27fc4e6f1747558b13c0e9e29996e7b22004e44c58404124750deaef4769bb8d",
                1_337,
            ),
            _pin(
                "src/models/agalite/layers.py",
                "610a703e3da2736fef14a2d72545a04143fcac4a9823ce707c7301fecd2e8978",
                5_915,
            ),
            _pin(
                "src_pure/models/agalite.py",
                "2bce6fe8dc417dfffbc9644011ea3efe9a172da2ab7bbdcd8ac357929ea00fc6",
                22_854,
            ),
            _pin(
                "src_pure/purejaxrl/ppo_rnn.py",
                "85e5ad42ea93311fc81c3a5ef8ce4559f709f46941e08155894f32b95fc1b6c9",
                12_016,
            ),
        ),
        candidate_ids=("adapted_ppo_agalite",),
    ),
    SourceFamilyPin(
        family_id="memory_traces",
        url="https://github.com/onnoeberhard/memory-traces",
        commit_git_sha1="fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd",
        tree_git_sha1="af6f2cdfd2dcabd079a030cc1e2357f09886fd27",
        archive_sha256="55701c411d293f63d6570563b53ec6b0bc84ae380ecb95eea42ea41928c1a4f9",
        archive_size_bytes=13_733,
        license_spdx="MIT",
        source_files=(
            _pin(
                "LICENSE",
                "6c9d35f885f47922acb8c77681b1dbd4b16f186bcb0ba5de3948e0efbad0a5f9",
                1_070,
            ),
            _pin(
                "examples/ppo_tmaze.py",
                "841bdd3d62ce4143f149b5a5fe1c18ec37c1a96def1594ba2faa04d466bae88f",
                3_157,
            ),
            _pin(
                "pyproject.toml",
                "ff0d1c3c3917520e7272d4425a3ce7f43167f5654accf8f792fd00ff2882fb38",
                412,
            ),
            _pin(
                "traces/main.py",
                "180697f158e173dc2d51ff11013090fd711c3f41e213972527e2f2a7a3ddbe3b",
                22_110,
            ),
            _pin(
                "traces/ppo.py",
                "e01aa53aa5e72e6890c2942ae77892786afe3d1f9256d5c378da1076fdfee541",
                15_180,
            ),
        ),
        candidate_ids=(
            "adapted_ppo_memory_trace_control_l0",
            "adapted_ppo_memory_trace_l0_l090",
            "adapted_ppo_memory_trace_l0_l099",
            "adapted_ppo_memory_trace_l0_l0999",
            "adapted_ppo_memory_trace_multiscale_l0_l090_l099_l0999",
        ),
    ),
    SourceFamilyPin(
        family_id="shm",
        url="https://github.com/thaihungle/SHM",
        commit_git_sha1="651f9e27e0fd3a3ec46a0f45b84e0128c5f8a312",
        tree_git_sha1="22fc6aaa216e3aa8032b31d56e51c31c6ea9c1b4",
        archive_sha256="4c12d7b5a5ca1356b99d31721e1cca3eb5137d4ae37afc34ad9d28b5d1e62193",
        archive_size_bytes=537_371,
        license_spdx="Apache-2.0",
        source_files=(
            _pin(
                "LICENSE",
                "26e45f86ea13d4ae20136b9c1d693149acd385dbec88a439b317fe8f2da0d55a",
                11_337,
            ),
            _pin(
                "README.md",
                "eaaeb9c20a30f36798fb347034be11a62416079f700ebcb079184f01c18ef3ae",
                10_814,
            ),
            _pin(
                "pomdp-baselines/torchkit/shm.py",
                "7df5a127d286434a52a8294f68b6e86ac297d010c06c6f774d4404b7b617965b",
                2_519,
            ),
            _pin(
                "popgym/baselines/ray_models/ray_shm.py",
                "d45e97cd9fc606372c44c885892614c79d7f422c550277a0b7d0935807f475e4",
                3_471,
            ),
            _pin(
                "shm.py",
                "7ba92a52e7ec4d75f2b8aab09ea463324b3b550dff597747df1e524aa75c0146",
                2_538,
            ),
            _pin(
                "train_popgym.py",
                "e0acff8bff5aa42cab7c096f264bebae2b1c1bb1bf4212b4c733159064b87aa3",
                7_288,
            ),
        ),
        candidate_ids=(
            "adapted_ppo_shm_popgym_code_faithful",
            "adapted_ppo_shm_intended_random",
        ),
    ),
    SourceFamilyPin(
        family_id="ffm",
        url="https://github.com/proroklab/ffm",
        commit_git_sha1="b3f94d2a0f35ba05089faf19ab1df846057cf8b6",
        tree_git_sha1="7684b03c81dc9fc16f7ac973c8c6425dd279e6f4",
        archive_sha256="b9497c94255a4d0e2d32666fea52595439b72cbc1d291ed50b528b7d62c4c69d",
        archive_size_bytes=286_966,
        license_spdx="MIT",
        source_files=(
            _pin(
                "LICENSE",
                "db7b51734c0b098407c121530d810026743145f11333bcae9b58b442197f9695",
                1_056,
            ),
            _pin(
                "README.md",
                "1e65c7d16b1f8e773aec6f8b71161435a0e09d963d72edb856b7ec216e7dfbed",
                4_291,
            ),
            _pin(
                "aggregations.py",
                "ce7f28de73acf6e663dd779b0b46b549307b5a1a9dd12a365adfb9911b63e099",
                19_380,
            ),
            _pin(
                "models/ffm_outer.py",
                "65a2356a16fa4aa188d4aa78fd4d464cc3a6f84c7500b72f09ff9f31a8762d0a",
                2_947,
            ),
            _pin(
                "models/ray_ffm.py",
                "f80e107310f27044ae050a29e89569bb424ed8e25f08d98039ef55f2919b0875",
                8_166,
            ),
            _pin(
                "ppo.py",
                "718c3eae1406b8f9485dbebed307c9f01210c501db7aaac1106f134a76dbced0",
                7_850,
            ),
            _pin(
                "standalone_jax/ffm/__init__.py",
                "39da83dd9994b5f4745a59a0bf79cf26abbd209a670dae5ce9100048dec2076a",
                70,
            ),
            _pin(
                "standalone_jax/ffm/ffa.py",
                "dffcc578c91f3baa8ca37d3fbc05b73031e3a2a06dd6d9be5c8d4cbe899945d2",
                2_672,
            ),
            _pin(
                "standalone_jax/ffm/ffm.py",
                "00832954cc87ce4b7a25a2b23104e649bc77f8f538f2751f3f90a9a9b368275a",
                2_817,
            ),
            _pin(
                "standalone_jax/setup.py",
                "71b89ef1823a3c25a390aecd26452e8ab1f72f625e88f3e7dce8836c9992c401",
                206,
            ),
        ),
        candidate_ids=(
            "adapted_ppo_ffm_jax_source",
            "adapted_ppo_ffm_paper_m32_c4",
        ),
    ),
)

SOURCE_FAMILY_BY_ID: Final[Mapping[str, SourceFamilyPin]] = MappingProxyType(
    {family.family_id: family for family in _SOURCE_FAMILIES}
)

_DEVELOPMENT_CANDIDATES: Final = (
    DevelopmentCandidateConcept(
        "adapted_ppo_gru_ld",
        "pobax_ld_gtrxl",
        "PPO-GRU with a positive-weight two-critic learning-dynamics objective.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_gtrxl",
        "pobax_ld_gtrxl",
        "PPO with a source-derived GTrXL memory after state and global-argument review.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_agalite",
        "agalite",
        "PPO with a source-derived AGaLiTe carry; this is not exact upstream execution.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_memory_trace_control_l0",
        "memory_traces",
        "Image-only memoryless lambda-zero PPO control for the trace family.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_memory_trace_l0_l090",
        "memory_traces",
        "Image-only PPO with observation traces at lambdas zero and 0.9.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_memory_trace_l0_l099",
        "memory_traces",
        "Image-only PPO with observation traces at lambdas zero and 0.99.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_memory_trace_l0_l0999",
        "memory_traces",
        "Image-only PPO with observation traces at lambdas zero and 0.999.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_memory_trace_multiscale_l0_l090_l099_l0999",
        "memory_traces",
        "Image-only PPO with the frozen four-scale observation-trace concept.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_shm_popgym_code_faithful",
        "shm",
        "Code-faithful POPGym/root SHM whose uniform_(0,1).long path selects row zero.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_shm_intended_random",
        "shm",
        "Distinct intended-random SHM using 128-row random selection and state clamping.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_ffm_jax_source",
        "ffm",
        "Image-only PPO with the audited standalone-JAX FFM initialization and reset contract.",
    ),
    DevelopmentCandidateConcept(
        "adapted_ppo_ffm_paper_m32_c4",
        "ffm",
        "Image-only PPO with the paper-scale m=32, c=4 FFM shape and frozen numerics.",
    ),
)

DEVELOPMENT_MEMORY_CANDIDATE_IDS: Final = tuple(
    candidate.candidate_id for candidate in _DEVELOPMENT_CANDIDATES
)
DEVELOPMENT_CANDIDATE_BY_ID: Final[Mapping[str, DevelopmentCandidateConcept]] = MappingProxyType(
    {candidate.candidate_id: candidate for candidate in _DEVELOPMENT_CANDIDATES}
)


def _assert_static_registry() -> None:
    if (
        len(FROZEN_MATCHED_V3_CANDIDATE_IDS) != 28
        or len(set(FROZEN_MATCHED_V3_CANDIDATE_IDS)) != 28
    ):
        raise RuntimeError("frozen matched-v3 candidate universe drifted")
    if len(SOURCE_FAMILY_BY_ID) != len(_SOURCE_FAMILIES):
        raise RuntimeError("duplicate source-family ID in static registry")
    if len(DEVELOPMENT_CANDIDATE_BY_ID) != len(_DEVELOPMENT_CANDIDATES):
        raise RuntimeError("duplicate development candidate ID in static registry")
    if set(FROZEN_MATCHED_V3_CANDIDATE_IDS) & set(DEVELOPMENT_MEMORY_CANDIDATE_IDS):
        raise RuntimeError("development memory concepts overlap the frozen matched-v3 universe")
    flattened = tuple(
        candidate for family in _SOURCE_FAMILIES for candidate in family.candidate_ids
    )
    if flattened != DEVELOPMENT_MEMORY_CANDIDATE_IDS:
        raise RuntimeError("source-family candidate order and development order disagree")
    if any(
        candidate.family_id != family.family_id
        for family in _SOURCE_FAMILIES
        for candidate_id in family.candidate_ids
        for candidate in (DEVELOPMENT_CANDIDATE_BY_ID[candidate_id],)
    ):
        raise RuntimeError("development candidate refers to the wrong source family")


_assert_static_registry()


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "artifact_accepted": False,
        "candidate_qualified": False,
        "evidence_authority_granted": False,
        "execution_authority_granted": False,
        "execution_ready": False,
        "matched_v3_run_authorized": False,
        "performance_claim_allowed": False,
        "production_plan_issued": False,
        "promotion_allowed": False,
        "publication_authority_granted": False,
        "qualification_authority_granted": False,
        "qualification_granted": False,
        "result_acceptance_authority_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "seed_authority_granted": False,
        "sota_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "workload_executed": False,
    }


def _semantic_contract(family_id: str) -> dict[str, Any]:
    if family_id == "pobax_ld_gtrxl":
        return {
            "adapted_ppo_gru_ld": {
                "actor_advantage_source": "critic_0",
                "clipped_value_loss_required": True,
                "critic_count": 2,
                "disagreement_term_required": True,
                "ld_weight_must_be_strictly_positive": True,
                "zero_ld_weight_is_treatment": False,
            },
            "adapted_ppo_gtrxl": {
                "evaluation_state_threading_review_required": True,
                "global_args_dependency_review_required": True,
                "source_state_semantics_accepted_without_review": False,
            },
            "shared_adapter_policy": {
                "forager_image_encoder_required": True,
                "previous_action_input_allowed": False,
                "upstream_exact_execution_asserted": False,
            },
        }
    if family_id == "agalite":
        return {
            "adapted_ppo_agalite": {
                "forager_image_encoder_required": True,
                "previous_action_input_allowed": False,
                "source_derived_only": True,
                "upstream_exact_execution_asserted": False,
            }
        }
    if family_id == "memory_traces":
        return {
            "candidate_grid": [
                ["0", "0.9"],
                ["0", "0.99"],
                ["0", "0.999"],
                ["0", "0.9", "0.99", "0.999"],
            ],
            "memoryless_control": {
                "candidate_id": "adapted_ppo_memory_trace_control_l0",
                "lambda_decimal_strings": ["0"],
                "observation_access": "forager_image_only",
            },
            "previous_action_trace_allowed": False,
            "qualification_or_held_out_seed_selection_allowed": False,
            "selection_seed_class": "development_only",
            "selected_variant_promotable": False,
        }
    if family_id == "shm":
        return {
            "candidate_concepts_may_be_conflated": False,
            "pomdp_torchkit_semantics": {
                "candidate_id": "adapted_ppo_shm_intended_random",
                "row_selection": "randint_over_128_then_state_clamp",
                "source_path": "pomdp-baselines/torchkit/shm.py",
            },
            "popgym_root_semantics": {
                "candidate_id": "adapted_ppo_shm_popgym_code_faithful",
                "row_selection": "uniform_(0,1).long_selects_row_0",
                "source_paths": [
                    "shm.py",
                    "popgym/baselines/ray_models/ray_shm.py",
                ],
            },
            "semantics_are_distinct": True,
        }
    if family_id == "ffm":
        return {
            "candidate_concepts_may_be_conflated": False,
            "jax_source_semantics": {
                "candidate_id": "adapted_ppo_ffm_jax_source",
                "context_parameter_initialization": (
                    "two_pi_over_linspace_1_to_1024_by_context_size"
                ),
                "decay_clamp": "maximum_negative_1e_6_only",
                "decay_parameter_initialization": (
                    "linspace_negative_e_to_negative_1e_6_by_memory_size"
                ),
                "done_mask_application": (
                    "incoming_done_masks_predecessor_before_incoming_input_addition"
                ),
                "hidden_size_affects_recurrence": False,
                "initial_state_shape": [1, "memory_size", "context_size"],
                "paper_configuration_exact": False,
                "source_paths": [
                    "standalone_jax/ffm/ffa.py",
                    "standalone_jax/ffm/ffm.py",
                ],
            },
            "paper_scale_semantics": {
                "candidate_id": "adapted_ppo_ffm_paper_m32_c4",
                "context_size": 4,
                "decay_retention_beta_decimal": "0.01",
                "initialization_horizon_steps": 1_024,
                "maximum_forward_sequence_steps_tested": 1_024,
                "memory_size": 32,
                "paper_arxiv_id": "2310.04128v1",
                "paper_recurrent_complex_elements": 128,
                "paper_recurrent_real_equivalent_dimensions": 256,
                "recommended_gamma_multiplication_precision": "float64",
                "source_jax_implementation_asserted_equivalent": False,
            },
            "shared_adapter_policy": {
                "episode_boundary_alignment_must_be_frozen_and_tested": True,
                "forager_image_encoder_required": True,
                "matched_ppo_backbone_required": True,
                "previous_action_input_allowed": False,
                "upstream_exact_execution_asserted": False,
            },
        }
    raise RuntimeError("unknown frozen source family")


def _adaptation_blockers(family_id: str) -> list[str]:
    if family_id == "pobax_ld_gtrxl":
        return [
            (
                "The LD weight must be positive and selected only on development seeds; "
                "zero is not the treatment."
            ),
            (
                "The two-critic loss and critic-zero actor advantage require a new "
                "matched adapter review."
            ),
            (
                "GTrXL evaluation-state threading and upstream global-argument "
                "dependencies remain unclosed."
            ),
            (
                "No image-only Forager adapter, resource envelope, publisher, or "
                "execution plan is issued."
            ),
        ]
    if family_id == "agalite":
        return [
            (
                "The future adapter is source-derived and cannot be described as exact "
                "upstream execution."
            ),
            (
                "No image-only Forager adapter, carry audit, resource envelope, or "
                "publisher is issued."
            ),
        ]
    if family_id == "memory_traces":
        return [
            (
                "All lambda selection is restricted to development seeds and cannot "
                "consume qualification seeds."
            ),
            (
                "The lambda-zero control and every trace arm require one matched "
                "image-only PPO backbone."
            ),
            "No candidate selection, resource envelope, publisher, or execution plan is issued.",
        ]
    if family_id == "shm":
        return [
            (
                "The POPGym/root row-zero behavior and POMDP-toolkit random-row "
                "behavior are distinct semantics."
            ),
            (
                "A future development protocol must name one semantic contract and "
                "cannot merge their results."
            ),
            (
                "No image-only Forager adapter, resource envelope, publisher, or "
                "execution plan is issued."
            ),
        ]
    if family_id == "ffm":
        return [
            (
                "The standalone-JAX initialization and the paper-scale configuration are "
                "distinct contracts and their results cannot be merged."
            ),
            (
                "The source done mask applies to the incoming element; a future adapter must "
                "freeze and test alignment with the Forager episode-boundary convention."
            ),
            (
                "The paper recommends float64 gamma multiplication and reports at most 1024 "
                "steps per forward pass; rollout chunking and CPU resources remain unfrozen."
            ),
            (
                "The future image-only adapter omits the paper's action-observation encoding "
                "and therefore cannot be called exact upstream execution."
            ),
            "No matched adapter, resource envelope, publisher, or execution plan is issued.",
        ]
    raise RuntimeError("unknown frozen source family")


def _paper_only_unregistered_gaps() -> list[dict[str, Any]]:
    return [
        {
            "audited_commit_git_sha1": "78709d2b5f99d40f10c8f5f4047c15f3dbb023b9",
            "audited_tree_git_sha1": "213ecff0e04cdc989087cb7f7a27b718fa3839f8",
            "candidate_registered": False,
            "clean_room_paper_specification_required": True,
            "code_adaptation_allowed_by_this_registry": False,
            "family_id": "memoroids_s5_tbb",
            "license_observation": (
                "no_repository_license_or_license_file_declared_in_audited_snapshot"
            ),
            "official_repository_url": "https://github.com/proroklab/memoroids",
            "paper_arxiv_id": "2402.09900v3",
            "paper_venue": "NeurIPS_2024",
            "performance_claim_reproduced": False,
            "source_family_registered": False,
            "source_imported_or_executed_here": False,
            "tbb_matched_update_budget_frozen": False,
        }
    ]


def _source_family_dict(family: SourceFamilyPin) -> dict[str, Any]:
    return {
        "adaptation_blockers": _adaptation_blockers(family.family_id),
        "archive_authenticated_here": False,
        "archive_identity_scope": (
            "audit_time_byte_receipt_not_reproduced_or_authenticated_here"
        ),
        "candidate_concepts": list(family.candidate_ids),
        "classification": "open_development_source_reference_nonpromoting",
        "family_id": family.family_id,
        "relevant_source_subset": [item.to_dict() for item in family.source_files],
        "semantic_contract": _semantic_contract(family.family_id),
        "source_imported_or_executed_here": False,
        "source_relationship": "source_derived_not_exact_execution",
        "subset_scope": "declared_relevant_digest_and_size_records_only",
        "subset_verifier_authenticates_archive": False,
        "upstream": family.upstream_dict(),
    }


def _descriptor() -> dict[str, Any]:
    return {
        "claims": _claims(),
        "classification": "descriptor_only_open_development_nonpromoting",
        "development_candidate_concepts": [
            candidate.to_dict() for candidate in _DEVELOPMENT_CANDIDATES
        ],
        "development_candidate_order": list(DEVELOPMENT_MEMORY_CANDIDATE_IDS),
        "frozen_matched_v3_universe": {
            "candidate_count": 28,
            "candidate_order": list(FROZEN_MATCHED_V3_CANDIDATE_IDS),
            "registry_authorizes_matched_v3_runs": False,
            "registry_extends_universe": False,
            "universe_is_immutable": True,
        },
        "limitations": [
            "This registry records source pins and future development concepts only.",
            "It cannot extend or reinterpret the frozen matched-v3 28-candidate universe.",
            "It cannot authorize a download, build, import, benchmark run, seed, or result.",
            (
                "The canonical inventory verifier checks only declared records; the "
                "optional read-only file verifier hashes exactly those declared bytes, "
                "but neither authenticates the archive or full tree."
            ),
            (
                "Audit-time archive byte pins require a newly reproduced deterministic "
                "materialization receipt before any future plan."
            ),
            "No source family is qualified, executed, promotable, or evidence-bearing here.",
            "No superiority, Forager SOTA, or universal SOTA claim may use this descriptor.",
            (
                "Paper-only gaps are exclusion records; they register neither source code nor "
                "a runnable candidate."
            ),
            "Repository-license observations are snapshot facts, not legal advice.",
        ],
        "paper_only_unregistered_gaps": _paper_only_unregistered_gaps(),
        "schema_version": MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SCHEMA_VERSION,
        "source_families": [_source_family_dict(family) for family in _SOURCE_FAMILIES],
        "status": MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_STATUS,
    }


def _canonical_json(value: object, *, maximum_bytes: int) -> bytes:
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
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ForagerMemoryComparatorDevelopmentSourcesError(
            "value cannot be encoded as strict canonical JSON"
        ) from exc
    if not 1 <= len(raw) <= maximum_bytes:
        _fail("canonical JSON exceeds its byte bound")
    return raw


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
        _fail("JSON integer exceeds the digit bound")
    return int(value)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail("non-finite JSON number is forbidden")
    return parsed


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_json_shape(value: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("JSON exceeds the shape bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                _fail("non-finite JSON number is forbidden")
            continue
        if type(current) is str:
            if len(current.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                _fail("JSON string exceeds the byte bound")
            continue
        if type(current) is list:
            stack.extend((item, depth + 1) for item in cast(list[object], current))
            continue
        if type(current) is dict:
            mapping = cast(dict[str, object], current)
            for key, item in mapping.items():
                if type(key) is not str:
                    _fail("JSON object keys must be exact strings")
                stack.append((key, depth + 1))
                stack.append((item, depth + 1))
            continue
        _fail("JSON contains an unsupported value type")


def _strict_json(raw: bytes, *, maximum_bytes: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= maximum_bytes:
        _fail("JSON input must be bounded exact bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMemoryComparatorDevelopmentSourcesError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ForagerMemoryComparatorDevelopmentSourcesError("invalid strict JSON") from exc
    if type(parsed) is not dict:
        _fail("JSON root must be an exact object")
    _validate_json_shape(parsed)
    result = cast(dict[str, Any], parsed)
    canonical = _canonical_json(result, maximum_bytes=maximum_bytes)
    if not hmac.compare_digest(raw, canonical):
        _fail("JSON input is not the exact canonical encoding")
    return result


_DESCRIPTOR: Final = _descriptor()
_DESCRIPTOR_BYTES: Final = _canonical_json(_DESCRIPTOR, maximum_bytes=_MAX_DESCRIPTOR_BYTES)

MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SHA256: Final = (
    "a98f78d5e5483c8dfbd821b953793e61ae820c1f1b3906a18b886836da7e116c"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SHA256,
):
    raise RuntimeError("memory-comparator development source descriptor drifted")


def memory_comparator_development_sources_descriptor_bytes() -> bytes:
    """Return the exact frozen descriptor bytes; this grants no authority."""

    return _DESCRIPTOR_BYTES


def memory_comparator_development_sources_descriptor() -> dict[str, Any]:
    """Return an isolated copy of the non-authorizing development descriptor."""

    return copy.deepcopy(_DESCRIPTOR)


def parse_memory_comparator_development_sources_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen canonical descriptor bytes."""

    parsed = _strict_json(raw, maximum_bytes=_MAX_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SHA256,
    ):
        _fail("memory-comparator development descriptor SHA-256 drifted")
    if parsed != _DESCRIPTOR:
        _fail("memory-comparator development descriptor content drifted")
    return copy.deepcopy(parsed)


def _require_family_id(family_id: object) -> SourceFamilyPin:
    requested = _require_identifier(family_id, "expected source family ID")
    try:
        return SOURCE_FAMILY_BY_ID[requested]
    except KeyError as exc:
        raise ForagerMemoryComparatorDevelopmentSourcesError(
            f"unknown source family {requested!r}"
        ) from exc


def _source_subset_inventory(family: SourceFamilyPin) -> dict[str, Any]:
    return {
        "family_id": family.family_id,
        "schema_version": SOURCE_SUBSET_INVENTORY_SCHEMA_VERSION,
        "source_files": [item.to_dict() for item in family.source_files],
        "source_subset_scope": "declared_relevant_digest_and_size_records_only",
    }


def expected_source_subset_inventory_bytes(family_id: str) -> bytes:
    """Return a canonical expected subset inventory without reading source."""

    family = _require_family_id(family_id)
    return _canonical_json(
        _source_subset_inventory(family),
        maximum_bytes=_MAX_INVENTORY_BYTES,
    )


def _validate_inventory_records(value: object) -> tuple[SourceFilePin, ...]:
    if type(value) is not list:
        _fail("source_files must be an exact JSON array")
    records = cast(list[object], value)
    if not 1 <= len(records) <= _MAX_SOURCE_FILES:
        _fail("source_files count is outside the bound")
    pins: list[SourceFilePin] = []
    for record in records:
        if type(record) is not dict:
            _fail("source_files entries must be exact JSON objects")
        item = cast(dict[str, object], record)
        if set(item) != {"path", "sha256", "size_bytes"}:
            _fail("source_files entry has missing or extra fields")
        pins.append(
            SourceFilePin(
                path=_require_exact_string(item["path"], "inventory source path"),
                sha256=_require_exact_string(item["sha256"], "inventory source SHA-256"),
                size_bytes=_require_positive_int(
                    item["size_bytes"],
                    "inventory source size",
                    maximum=_MAX_SOURCE_FILE_BYTES,
                ),
            )
        )
    paths = tuple(pin.path for pin in pins)
    if paths != tuple(sorted(paths, key=lambda path: path.encode("utf-8"))):
        _fail("source_files must use exact ascending UTF-8 path order")
    _reject_path_aliases(paths)
    return tuple(pins)


def verify_source_subset_inventory(
    raw: bytes,
    *,
    expected_family_id: str | None = None,
) -> SourceSubsetVerification:
    """Verify an exact canonical relevant-source digest-and-size inventory.

    Acceptance binds only the named records.  It does not prove that source
    bytes were read, authenticate the pinned archive, or authorize execution.
    """

    parsed = _strict_json(raw, maximum_bytes=_MAX_INVENTORY_BYTES)
    if set(parsed) != {
        "family_id",
        "schema_version",
        "source_files",
        "source_subset_scope",
    }:
        _fail("source-subset inventory has missing or extra fields")
    if parsed["schema_version"] != SOURCE_SUBSET_INVENTORY_SCHEMA_VERSION:
        _fail("source-subset inventory schema version drifted")
    if parsed["source_subset_scope"] != "declared_relevant_digest_and_size_records_only":
        _fail("source-subset inventory scope drifted")
    family = _require_family_id(parsed["family_id"])
    if expected_family_id is not None:
        expected = _require_family_id(expected_family_id)
        if family.family_id != expected.family_id:
            _fail("source-subset inventory family does not match the expected family")
    pins = _validate_inventory_records(parsed["source_files"])
    if pins != family.source_files:
        _fail("source-subset membership, ordering, size, or SHA-256 drifted")
    expected_raw = expected_source_subset_inventory_bytes(family.family_id)
    if not hmac.compare_digest(raw, expected_raw):
        _fail("source-subset inventory is not the exact frozen canonical inventory")
    return SourceSubsetVerification(
        family_id=family.family_id,
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
        source_file_count=len(pins),
        source_total_size_bytes=sum(pin.size_bytes for pin in pins),
        source_bytes_verified=False,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _source_directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        _fail("source subset file verification requires Linux no-follow directories")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_source_root(root: Path) -> tuple[int, tuple[int, ...]]:
    if type(root) is not type(Path()) or not root.is_absolute() or root == Path("/"):
        _fail("source checkout root must be one exact non-root absolute Path")
    raw = os.fspath(root)
    if "\x00" in raw or len(os.fsencode(raw)) > _MAX_SOURCE_ROOT_PATH_BYTES:
        _fail("source checkout root exceeds its path bound")
    components = root.parts[1:]
    if not components or any(
        component in {"", ".", ".."}
        or len(os.fsencode(component)) > _MAX_PATH_COMPONENT_BYTES
        for component in components
    ):
        _fail("source checkout root has a noncanonical component")
    flags = _source_directory_flags()
    try:
        current = os.open("/", flags)
    except OSError as exc:
        raise ForagerMemoryComparatorDevelopmentSourcesError(
            "cannot anchor the source checkout filesystem root"
        ) from exc
    try:
        for component in components:
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        opened = os.fstat(current)
        named = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(opened) != _directory_identity(named)
        ):
            _fail("source checkout root is not one stable link-free directory")
        return current, _directory_identity(opened)
    except ForagerMemoryComparatorDevelopmentSourcesError:
        os.close(current)
        raise
    except OSError as exc:
        os.close(current)
        raise ForagerMemoryComparatorDevelopmentSourcesError(
            "cannot open source checkout root through a link-free directory chain"
        ) from exc


def _open_source_parent(root_descriptor: int, path: str) -> tuple[int, str]:
    parts = PurePosixPath(_validate_source_path(path)).parts
    current = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            following = os.open(
                component,
                _source_directory_flags(),
                dir_fd=current,
            )
            os.close(current)
            current = following
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _verify_one_source_file(
    root_descriptor: int,
    pin: SourceFilePin,
    seen_identities: set[tuple[int, int]],
) -> None:
    parent_descriptor = -1
    descriptor = -1
    try:
        try:
            parent_descriptor, name = _open_source_parent(root_descriptor, pin.path)
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except ForagerMemoryComparatorDevelopmentSourcesError:
            raise
        except OSError as exc:
            raise ForagerMemoryComparatorDevelopmentSourcesError(
                f"cannot open exact source subset member {pin.path!r}"
            ) from exc
        try:
            identity = _file_identity(opened)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != pin.size_bytes
                or _file_identity(named) != identity
            ):
                _fail(f"unsafe or wrong-size exact source subset member {pin.path!r}")
            inode_key = (opened.st_dev, opened.st_ino)
            if inode_key in seen_identities:
                _fail("source subset contains a hard-link identity alias")
            seen_identities.add(inode_key)
            digest = hashlib.sha256()
            remaining = pin.size_bytes
            while remaining:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
                if not chunk:
                    _fail(f"exact source subset member ended early for {pin.path!r}")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                _fail(f"exact source subset member grew while read for {pin.path!r}")
            after = os.fstat(descriptor)
            located = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if identity != _file_identity(after) or identity != _file_identity(located):
                _fail(f"exact source subset member changed while read for {pin.path!r}")
            if not hmac.compare_digest(digest.hexdigest(), pin.sha256):
                _fail(f"exact source subset SHA-256 drifted for {pin.path!r}")
        except OSError as exc:
            raise ForagerMemoryComparatorDevelopmentSourcesError(
                f"cannot read exact source subset member {pin.path!r}"
            ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def verify_source_subset_files(
    family_id: str,
    root: Path,
) -> SourceSubsetVerification:
    """Hash the declared source subset below one caller-supplied checkout root.

    Files outside the declared subset are deliberately out of scope.  Every
    declared member must be a stable, single-link regular file at its canonical
    path with the exact frozen length and digest.  This read-only check still
    grants no archive authentication, execution, qualification, or claim
    authority.
    """

    family = _require_family_id(family_id)
    root_descriptor, root_identity = _open_source_root(root)
    seen_identities: set[tuple[int, int]] = set()
    try:
        for pin in family.source_files:
            _verify_one_source_file(root_descriptor, pin, seen_identities)
        opened_after = os.fstat(root_descriptor)
        named_after = os.stat(root, follow_symlinks=False)
        if (
            _directory_identity(opened_after) != root_identity
            or _directory_identity(named_after) != root_identity
        ):
            _fail("source checkout root changed during subset verification")
    except OSError as exc:
        raise ForagerMemoryComparatorDevelopmentSourcesError(
            "cannot revalidate source checkout root"
        ) from exc
    finally:
        os.close(root_descriptor)
    inventory_raw = expected_source_subset_inventory_bytes(family.family_id)
    return SourceSubsetVerification(
        family_id=family.family_id,
        inventory_sha256=hashlib.sha256(inventory_raw).hexdigest(),
        source_file_count=len(family.source_files),
        source_total_size_bytes=sum(pin.size_bytes for pin in family.source_files),
        source_bytes_verified=True,
    )


__all__ = [
    "DEVELOPMENT_CANDIDATE_BY_ID",
    "DEVELOPMENT_MEMORY_CANDIDATE_IDS",
    "FROZEN_MATCHED_V3_CANDIDATE_IDS",
    "ForagerMemoryComparatorDevelopmentSourcesError",
    "MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SCHEMA_VERSION",
    "MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SHA256",
    "MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_STATUS",
    "SOURCE_FAMILY_BY_ID",
    "SOURCE_SUBSET_INVENTORY_SCHEMA_VERSION",
    "DevelopmentCandidateConcept",
    "SourceFamilyPin",
    "SourceFilePin",
    "SourceSubsetVerification",
    "expected_source_subset_inventory_bytes",
    "memory_comparator_development_sources_descriptor",
    "memory_comparator_development_sources_descriptor_bytes",
    "parse_memory_comparator_development_sources_descriptor",
    "verify_source_subset_files",
    "verify_source_subset_inventory",
]
