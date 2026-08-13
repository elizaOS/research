"""Score/reward-value-undecoded in-container worker for matched Forager v3.

The worker launches exactly one of the twelve frozen external candidates from
the staged source tree embedded in the matched-v3 CPU OCI image.  It never
imports the result bridge or scorer and never opens the reward member inside
the upstream NPZ.  Success is represented by an opaque, PID-bound, single-use
capability; process completion, paths, serialized receipts, and copied bytes do
not recreate that capability.

This is deliberately *not* a host OCI executor.  A later qualification
component must create and inspect a networkless, read-only container from the
fresh qualified image, bind this worker source, issue seeds, observe resources,
and invoke the exact preloaded in-container consumer that claims the opaque outcome
before score-bearing conversion and atomic publication on a dedicated output boundary.
Running this module directly on a host cannot qualify a runtime or candidate.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal, Never, NoReturn, Protocol, cast

EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_runner_descriptor.v1"
)
EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_receipt.v1"
)
EXTERNAL_EXECUTION_RUNNER_STATUS: Final = (
    "implemented_in_container_worker_unexecuted_unqualified_non_authorizing"
)

EXPLICIT_EXTERNAL_EXECUTION_OPT_IN: Final = (
    "AUTHORIZE ONE NONQUALIFYING MATCHED-V3 EXTERNAL IN-CONTAINER WORKER EXECUTION"
)
_PRODUCTION_RUNNER_EXACT_SCOPE: Final = (
    "captured_process_runner_and_loaded_semantic_surface_only_not_runtime_qualification"
)
_INJECTED_TEST_RUNNER_SCOPE: Final = "injected_test_runner_permanently_nonproduction"

_EXECUTION_CONTRACT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_execution_contract_descriptor.v1"
)
_EXECUTION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
)
_EXECUTION_CONTRACT_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_execution_contract.py"
)
_EXECUTION_CONTRACT_SOURCE_SHA256: Final = (
    "b53381a21f47fd488e79f97630211c2e90ab43faf7775fb8d8ed5cbebcff76d2"
)
_EXTERNAL_STAGING_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_contract_descriptor.v1"
)
_EXTERNAL_STAGING_DESCRIPTOR_SHA256: Final = (
    "ceea86b38822f3add0465788003d349dd221a49fba5f3fa069bfec985537caea"
)
_EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_staging_manifest.v1"
)
_SEED_TRANSPORT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_seed_transport.v1"
)
_SEED_TRANSPORT_DESCRIPTOR_SHA256: Final = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)
_RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_result_bridge_descriptor.v1"
)
_RESULT_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
)
_CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256: Final = (
    "31d4c5a101f441bc082bdaf9250050f7950440271e6360854d5faa9fcd7ff34a"
)
_ATOMIC_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "b224fe9fdc438ccab0df5bfd3199e1d264feacbb99147970cc68a9c703b9e98e"
)
_EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_external_outcome_consumer_isolated_v1"
)
_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_outcome_consumer_descriptor.v1"
)
_EXTERNAL_OUTCOME_CONSUMER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_outcome_consumer.py"
)
_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256"
)
_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256"
)

_WORKLOAD_ROOT: Final = Path("/opt/elizaos/src/external-foragax")
_PRIVATE_RUNTIME_PARENT: Final = Path("/run/alberta")
_PYTHON_EXECUTABLE: Final = Path("/usr/local/bin/python3.12")
_PYTHON_ARGV0: Final = "/usr/local/bin/python"
_EXPECTED_PYTHON_VERSION: Final = "3.12.3"
_EXPECTED_UID: Final = 65_532
_EXPECTED_GID: Final = 65_532

_HORIZON: Final = 499_712
_PPO_ROLLOUT_STEPS: Final = 2_048
_PPO_ROLLOUT_COUNT: Final = 244
_UINT31_MAX: Final = (1 << 31) - 1
_MAX_TIMEOUT_SECONDS: Final = 86_400
_MAX_STDOUT_BYTES: Final = 16 * 1024 * 1024
_MAX_STDERR_BYTES: Final = 16 * 1024 * 1024
_MAX_EXTERNAL_NPZ_BYTES: Final = 64 * 1024 * 1024
_MAX_RESULTS_DATABASE_BYTES: Final = 512 * 1024 * 1024
_MAX_PPO_VIDEO_BYTES: Final = 512 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 4 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES: Final = 4 * 1024 * 1024
_MAX_EXECUTABLE_BYTES: Final = 512 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_TEXT_BYTES: Final = 1024 * 1024
_MAX_PATH_BYTES: Final = 4096
_MAX_PATH_COMPONENT_BYTES: Final = 255
_PROCESS_CLEANUP_SECONDS: Final = 2.0
_READ_BLOCK_BYTES: Final = 1024 * 1024
_MAX_INVENTORY_ENTRIES: Final = 100_000
_MAX_INVENTORY_DEPTH: Final = 64
_MAX_CLEANUP_ENTRIES: Final = 100_000
_MAX_CLEANUP_DEPTH: Final = 64
_CLEANUP_BATCH_ENTRIES: Final = 1024
_MIN_PROCESS_RETURNCODE: Final = -(signal.NSIG - 1)
_MAX_PROCESS_RETURNCODE: Final = 255

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_NAME_RE: Final = re.compile(r"external-[0-9a-f]{32}\Z")
_FORBIDDEN_SCORE_MODULE_PREFIXES: Final = (
    "alberta_framework.benchmarks._forager_matched_v3_external_result_bridge",
    "alberta_framework.benchmarks._forager_matched_v3_scorer",
    "alberta_framework.benchmarks.forager_matched_v3_protocol",
)

_CONTINUING_ENTRYPOINT: Final = "src/continuing_main.py"
_PPO_ENTRYPOINT: Final = "src/rtu_ppo.py"
_ENTRYPOINT_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        _CONTINUING_ENTRYPOINT: (
            "ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"
        ),
        _PPO_ENTRYPOINT: (
            "1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"
        ),
    }
)
_PPO_VIDEO_RELATIVE_PATH: Final = "videos/0/497664_499712-episode-0.mp4"


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate_id: str
    configuration_path: str
    configuration_sha256: str
    output_stem: str
    family: Literal["continuing", "ppo"]


_CANDIDATE_SPECS: Final = (
    _CandidateSpec(
        "external_dqn_plain",
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json",
        "1d8a711ee1e4db575cb0edcacbaf38f97bd06cddc24019eb64b8c410e84b4e85",
        "DQN",
        "continuing",
    ),
    _CandidateSpec(
        "external_dqn_crelu",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_CReLU.json",
        "ef92352b97d92e7d40458db48157f589b0d0984f2f4286947c9a1f28bd522892",
        "DQN_CReLU",
        "continuing",
    ),
    _CandidateSpec(
        "external_dqn_redo",
        (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "DQN_ReDo_PostLNScore.json"
        ),
        "c38288f2ddb6a5dd8892954b499370d04399ec41e966fe790643c9d64b5ffc54",
        "DQN_ReDo_PostLNScore",
        "continuing",
    ),
    _CandidateSpec(
        "external_dqn_reward_trace",
        (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "DQN_reward_trace.json"
        ),
        "8641a3b4673940f5519f074b617ccc58a6c14b61a8b448df434cebb3d5f4c974",
        "DQN_reward_trace",
        "continuing",
    ),
    _CandidateSpec(
        "external_dqn_l2_init",
        (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "DQN_L2_Init.json"
        ),
        "2a2a1dc503b0617c35c202027a646db32186e2668d4b8988215f516a036b9107",
        "DQN_L2_Init",
        "continuing",
    ),
    _CandidateSpec(
        "external_pt_dqn_xfinal",
        (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "PT_DQN_64.json"
        ),
        "05eaad6da93d8c42d8bd60da3d6c3728bca5c653608eb98210a48a76bedce2e2",
        "PT_DQN_64",
        "continuing",
    ),
    _CandidateSpec(
        "external_drqn_xfinal",
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "2b0e177420a9f9a4c8a7bd7aede9c7d2c5add3da4c8b3e301f32bb2588637047",
        "DRQN",
        "continuing",
    ),
    _CandidateSpec(
        "isolated_ppo_generic",
        (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "ActorCriticMLP.json"
        ),
        "27ffdffcf3ff3e722be5cdfe58d6bc07348ebe5380478032eedfaf435b754c71",
        "ActorCriticMLP",
        "ppo",
    ),
    _CandidateSpec(
        "isolated_rtu_paper_scale",
        (
            "experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json"
        ),
        "c32e240bf8c78cf2c7d1ad958bbfc8975b55160fb09490401763a346c2a21090",
        "PPO-RTU_LN_2048",
        "ppo",
    ),
    _CandidateSpec(
        "random_policy",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Random.json"
        ),
        "d20dc9294baab331c4658e4c682d5e1eee3c6f7cc6baf5d17586f48362e8936d",
        "Random",
        "continuing",
    ),
    _CandidateSpec(
        "search_nearest",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Nearest.json"
        ),
        "97b644c4c625155ae16fa7b69432ea0774f767142cc0e28b3d6fcec18c17d2ab",
        "Search-Nearest",
        "continuing",
    ),
    _CandidateSpec(
        "search_oracle",
        (
            "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
            "Baselines/Search-Oracle.json"
        ),
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89",
        "Search-Oracle",
        "continuing",
    ),
)

EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS: Final = tuple(
    spec.candidate_id for spec in _CANDIDATE_SPECS
)
_CANDIDATE_BY_ID: Final[Mapping[str, _CandidateSpec]] = MappingProxyType(
    {spec.candidate_id: spec for spec in _CANDIDATE_SPECS}
)

_EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD: Final = sys.modules.get(
    _EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME
)
_EXTERNAL_OUTCOME_CONSUMER_SINK_AT_LOAD: Final = getattr(
    _EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD,
    "_consume_claimed_matched_v3_external_execution_payload",
    None,
)
_EXTERNAL_OUTCOME_CONSUMER_GUARD_AT_LOAD: Final = getattr(
    _EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD, "_CONSUMER_GUARD_AT_LOAD", None
)
_EXTERNAL_OUTCOME_CONSUMER_REPLAY_AT_LOAD: Final = getattr(
    _EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD,
    "_replay_external_outcome_consumer_guard",
    None,
)


class ForagerMatchedV3ExternalExecutionRunnerError(RuntimeError):
    """The worker boundary, workload, artifact, receipt, or cleanup failed closed."""

    process_state_uncertain: bool
    filesystem_state_uncertain: bool

    def __init__(
        self,
        message: str,
        *,
        process_state_uncertain: bool = False,
        filesystem_state_uncertain: bool = False,
    ) -> None:
        self.process_state_uncertain = process_state_uncertain
        self.filesystem_state_uncertain = filesystem_state_uncertain
        super().__init__(message)


def _fail(
    message: str,
    *,
    process_state_uncertain: bool = False,
    filesystem_state_uncertain: bool = False,
) -> NoReturn:
    raise ForagerMatchedV3ExternalExecutionRunnerError(
        message,
        process_state_uncertain=process_state_uncertain,
        filesystem_state_uncertain=filesystem_state_uncertain,
    )


@dataclass(frozen=True, slots=True)
class BoundedExternalProcessResult:
    """Exact result of the private bounded process boundary."""

    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limit_exceeded: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.returncode) is not int
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
            or type(self.timed_out) is not bool
            or type(self.output_limit_exceeded) is not bool
        ):
            raise TypeError("bounded external process results require exact field types")
        if not _MIN_PROCESS_RETURNCODE <= self.returncode <= _MAX_PROCESS_RETURNCODE:
            raise ValueError("bounded external process returncode is outside the OS process range")


class ExternalProcessRunner(Protocol):
    """Private process seam used by focused tests and the default live runner."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        executable_descriptor: int,
        inherited_descriptors: tuple[int, ...],
        working_directory: str,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> BoundedExternalProcessResult: ...


@dataclass(frozen=True, slots=True)
class MatchedV3ExternalExecutionCompletion:
    """Score-bearing bytes released without value interpretation by this runner."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    execution_receipt_bytes: bytes
    execution_receipt_sha256: str
    upstream_reward_npz: bytes
    upstream_results_database: bytes
    upstream_video: bytes | None
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or type(self.environment_seed) is not int
            or type(self.agent_seed) is not int
            or type(self.execution_receipt_bytes) is not bytes
            or type(self.execution_receipt_sha256) is not str
            or type(self.upstream_reward_npz) is not bytes
            or type(self.upstream_results_database) is not bytes
            or (
                self.upstream_video is not None
                and type(self.upstream_video) is not bytes
            )
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
        ):
            raise TypeError("external execution completion fields require exact types")


@dataclass(frozen=True, slots=True)
class _SealedExternalExecutionPayload:
    """Private score-bearing bytes not decoded here, held until one path wins."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    execution_receipt_bytes: bytes
    execution_receipt_sha256: str
    upstream_reward_npz: bytes
    upstream_results_database: bytes
    upstream_video: bytes | None
    stdout: bytes
    stderr: bytes
    production_runner_exact: bool

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or type(self.environment_seed) is not int
            or type(self.agent_seed) is not int
            or type(self.execution_receipt_bytes) is not bytes
            or type(self.execution_receipt_sha256) is not str
            or type(self.upstream_reward_npz) is not bytes
            or type(self.upstream_results_database) is not bytes
            or (
                self.upstream_video is not None
                and type(self.upstream_video) is not bytes
            )
            or type(self.stdout) is not bytes
            or type(self.stderr) is not bytes
            or type(self.production_runner_exact) is not bool
        ):
            raise TypeError("sealed external execution payload fields require exact types")


class _ExecutionCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 external worker execution capability>"

    def __copy__(self) -> Never:
        raise TypeError("external execution capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> Never:
        raise TypeError("external execution capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("external execution capabilities cannot be serialized")


@dataclass(slots=True)
class _ExecutionState:
    pid: int
    status: Literal["issued", "consumed"]
    candidate_id: str
    environment_seed: int
    agent_seed: int
    source_sha256: str
    test_only_injected: bool


class _OutcomeCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 external worker outcome capability>"

    def __copy__(self) -> Never:
        raise TypeError("external outcome capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> Never:
        raise TypeError("external outcome capabilities cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("external outcome capabilities cannot be serialized")


@dataclass(slots=True)
class _OutcomeState:
    pid: int
    status: Literal["live", "consumed"]
    execution_capability: _ExecutionCapability
    execution_identity: int
    source_sha256: str
    content_sha256: tuple[str, ...]
    sealed_payload: _SealedExternalExecutionPayload
    production_runner_exact: bool


class _InjectedTestOnlyMarker:
    __slots__ = ()


_INJECTED_TEST_ONLY_MARKER: Final = _InjectedTestOnlyMarker()


_CAPABILITY_LOCK: Final = threading.Lock()
_EXECUTION_CAPABILITIES: Final[
    weakref.WeakKeyDictionary[_ExecutionCapability, _ExecutionState]
] = weakref.WeakKeyDictionary()
_OUTCOME_CAPABILITIES: Final[weakref.WeakKeyDictionary[_OutcomeCapability, _OutcomeState]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(slots=True)
class _DirectoryAnchor:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    owner: tuple[int, int]
    mode: int

    def verify(self, *, path_required: bool = True) -> None:
        if self.descriptor < 0:
            _fail("directory anchor is closed")
        try:
            observed = os.fstat(self.descriptor)
        except OSError as exc:
            raise ForagerMatchedV3ExternalExecutionRunnerError(
                "directory anchor became inaccessible"
            ) from exc
        if (
            _directory_identity(observed) != self.identity
            or not stat.S_ISDIR(observed.st_mode)
        ):
            _fail("directory anchor identity changed")
        if path_required:
            try:
                named = os.stat(self.path, follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3ExternalExecutionRunnerError(
                    "directory anchor path became inaccessible"
                ) from exc
            if _directory_identity(named) != self.identity:
                _fail("directory anchor path was substituted")

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(slots=True)
class _ExecutableAnchor:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    size_bytes: int
    sha256: str

    def verify(self) -> None:
        if self.descriptor < 0:
            _fail("executable anchor is closed")
        try:
            observed = os.fstat(self.descriptor)
            named = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3ExternalExecutionRunnerError(
                "worker executable anchor became inaccessible"
            ) from exc
        if (
            _stat_identity(observed) != self.identity
            or _stat_identity(named) != self.identity
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_size != self.size_bytes
            or not hmac.compare_digest(
                _hash_open_descriptor(self.descriptor, self.size_bytes), self.sha256
            )
        ):
            _fail("worker executable identity changed")

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


class _SourceMemberIdentity(Protocol):
    def __call__(
        self,
        root: _DirectoryAnchor,
        path: str,
        expected_sha256: str,
    ) -> dict[str, Any]: ...


class _CleanupExecutionRoot(Protocol):
    def __call__(
        self,
        parent: _DirectoryAnchor,
        name: str,
        root: _DirectoryAnchor,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    workload_root: Path
    private_runtime_parent: Path
    python_executable: Path
    python_argv0: str
    process_runner: ExternalProcessRunner
    source_member_identity: _SourceMemberIdentity
    cleanup_execution_root: _CleanupExecutionRoot
    production_runner_exact: bool
    test_only_process_runner_injected: bool
    closure_integrity_checked: bool


_PRODUCTION_PROCESS_RUNNER: ExternalProcessRunner
_PRODUCTION_SOURCE_MEMBER_IDENTITY: _SourceMemberIdentity
_PRODUCTION_CLEANUP_EXECUTION_ROOT: _CleanupExecutionRoot


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_uint31(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        _fail(f"{label} must be one exact uint31")
    return value


def _require_ceiling(value: object, label: str, *, maximum: int, zero: bool) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one exact integer in [{minimum}, {maximum}]")
    return value


def _require_exact_object(
    value: object,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be one exact nonnegative integer")
    return value


def _require_process_returncode(value: object, label: str) -> int:
    if (
        type(value) is not int
        or value < _MIN_PROCESS_RETURNCODE
        or value > _MAX_PROCESS_RETURNCODE
    ):
        _fail(f"{label} must be one exact plausible OS process returncode")
    return value


def _assert_plain_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("external execution JSON exceeds its structural bound")
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_TEXT_BYTES:
                _fail("external execution JSON text exceeds its bound")
            continue
        if type(item) is list:
            identity = id(item)
            if identity in seen:
                _fail("external execution JSON must be unaliased and acyclic")
            seen.add(identity)
            pending.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in seen:
                _fail("external execution JSON must be unaliased and acyclic")
            seen.add(identity)
            for key, child in item.items():
                if type(key) is not str:
                    _fail("external execution JSON object key is not exact text")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))
            continue
        _fail("external execution JSON contains a non-plain value")


def _canonical_json(value: Mapping[str, Any], *, maximum: int = _MAX_RECEIPT_BYTES) -> bytes:
    if type(value) is not dict:
        _fail("external execution canonical JSON root must be a plain object")
    _assert_plain_json(value)
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
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external execution value is not canonical ASCII JSON"
        ) from exc
    if not raw or len(raw) > maximum:
        _fail("external execution canonical JSON exceeds its byte bound")
    return raw


def _reject_constant(value: str) -> NoReturn:
    _fail(f"external execution JSON contains forbidden constant {value!r}")


def _reject_float(value: str) -> NoReturn:
    _fail(f"external execution JSON contains forbidden float {value!r}")


def _parse_int(value: str) -> int:
    if len(value.removeprefix("-")) > 19:
        _fail("external execution JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"external execution JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail("external execution JSON input must be bounded exact bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("external execution JSON must have one canonical trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3ExternalExecutionRunnerError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external execution JSON is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail("external execution JSON root must be a plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if not hmac.compare_digest(_canonical_json(result, maximum=maximum), raw):
        _fail("external execution JSON is not in exact canonical form")
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_map = left
        right_map = right
        return set(left_map) == set(right_map) and all(
            _exact_json_equal(left_map[key], right_map[key]) for key in left_map
        )
    if type(left) is list:
        left_list = left
        right_list = right
        return len(left_list) == len(right_list) and all(
            _exact_json_equal(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    return bool(left == right)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        _fail("external execution requires Linux O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        _fail("external execution requires Linux O_NOFOLLOW")
    return os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)


def _validate_absolute_path(path: Path, label: str) -> tuple[str, ...]:
    if type(path) is not type(Path()) or not path.is_absolute() or path == Path("/"):
        _fail(f"{label} must be one exact non-root absolute Path")
    raw = os.fspath(path)
    if "\x00" in raw or len(raw.encode("utf-8")) > _MAX_PATH_BYTES:
        _fail(f"{label} exceeds its path bound")
    parts = path.parts[1:]
    if not parts:
        _fail(f"{label} cannot be the filesystem root")
    for part in parts:
        if (
            part in {"", ".", ".."}
            or "/" in part
            or "\\" in part
            or len(part.encode("utf-8")) > _MAX_PATH_COMPONENT_BYTES
        ):
            _fail(f"{label} contains an unsafe component")
    return parts


def _open_absolute_directory(path: Path, label: str) -> _DirectoryAnchor:
    parts = _validate_absolute_path(path, label)
    current = os.open("/", _directory_flags())
    try:
        for part in parts:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = next_descriptor
        metadata = os.fstat(current)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != _directory_identity(named)
        ):
            _fail(f"{label} is not one stable link-free directory")
        return _DirectoryAnchor(
            path=path,
            descriptor=current,
            identity=_directory_identity(metadata),
            owner=(metadata.st_uid, metadata.st_gid),
            mode=stat.S_IMODE(metadata.st_mode),
        )
    except OSError as exc:
        os.close(current)
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            f"{label} cannot be opened through a link-free directory chain"
        ) from exc
    except BaseException:
        os.close(current)
        raise


def _hash_open_descriptor(descriptor: int, expected_size: int) -> str:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "bounded regular file cannot be rewound"
        ) from exc
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
        if not block:
            _fail("bounded regular file was truncated while hashing")
        digest.update(block)
        remaining -= len(block)
    if os.read(descriptor, 1):
        _fail("bounded regular file grew while hashing")
    return digest.hexdigest()


def _open_executable(path: Path) -> _ExecutableAnchor:
    _validate_absolute_path(path, "worker executable")
    try:
        descriptor = os.open(path, _file_flags())
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "worker executable cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > _MAX_EXECUTABLE_BYTES
            or not metadata.st_mode & stat.S_IXUSR
            or _stat_identity(metadata) != _stat_identity(named)
        ):
            _fail("worker executable is not one stable bounded executable")
        digest = _hash_open_descriptor(descriptor, metadata.st_size)
        return _ExecutableAnchor(
            path=path,
            descriptor=descriptor,
            identity=_stat_identity(metadata),
            size_bytes=metadata.st_size,
            sha256=digest,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _validate_relative_path(path: str, label: str) -> tuple[str, ...]:
    if type(path) is not str or not path or path.startswith("/") or "\x00" in path:
        _fail(f"{label} is not one exact relative path")
    pure = PurePosixPath(path)
    parts = pure.parts
    if (
        pure.as_posix() != path
        or not parts
        or any(
            part in {"", ".", ".."}
            or "\\" in part
            or len(part.encode("utf-8")) > _MAX_PATH_COMPONENT_BYTES
            for part in parts
        )
    ):
        _fail(f"{label} is not one canonical bounded relative path")
    return parts


def _open_relative_parent(root_descriptor: int, path: str) -> tuple[int, str]:
    parts = _validate_relative_path(path, "relative file")
    current = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _read_relative_regular(
    root_descriptor: int,
    path: str,
    *,
    maximum_bytes: int,
    expected_sha256: str | None = None,
    require_nonempty: bool = True,
    require_current_owner: bool = True,
) -> bytes:
    parent, name = _open_relative_parent(root_descriptor, path)
    descriptor = -1
    try:
        descriptor = os.open(name, _file_flags(), dir_fd=parent)
        before = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                require_current_owner
                and (before.st_uid != os.getuid() or before.st_gid != os.getgid())
            )
            or before.st_mode & 0o022
            or before.st_size > maximum_bytes
            or (require_nonempty and before.st_size < 1)
            or _stat_identity(before) != _stat_identity(named)
        ):
            _fail(f"artifact is not one stable private bounded regular file: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            block = os.read(descriptor, min(_READ_BLOCK_BYTES, remaining))
            if not block:
                _fail(f"artifact was truncated while reading: {path}")
            chunks.append(block)
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            _fail(f"artifact grew while reading: {path}")
        after = os.fstat(descriptor)
        renamed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            before
        ) != _stat_identity(renamed):
            _fail(f"artifact identity changed while reading: {path}")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(
            actual_sha256, expected_sha256
        ):
            _fail(f"source member digest differs: {path}")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _source_member_identity(
    root: _DirectoryAnchor,
    path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    raw = _read_relative_regular(
        root.descriptor,
        path,
        maximum_bytes=_MAX_SOURCE_FILE_BYTES,
        expected_sha256=expected_sha256,
        require_current_owner=False,
    )
    return {"path": path, "sha256": _sha256(raw), "size_bytes": len(raw)}


def _module_source_sha256(path: object, *, expected_suffix: str, label: str) -> str:
    if type(path) is not str or not path.endswith(expected_suffix):
        _fail(f"{label} source path is unavailable")
    try:
        descriptor = os.open(path, _file_flags())
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            f"{label} source cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > 16 * 1024 * 1024
        ):
            _fail(f"{label} source is not one bounded regular file")
        digest = _hash_open_descriptor(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            _fail(f"{label} source changed while hashing")
        return digest
    finally:
        os.close(descriptor)


def _current_module_source_sha256() -> str:
    return _module_source_sha256(
        globals().get("__file__"),
        expected_suffix=(
            "alberta_framework/benchmarks/forager_matched_v3_external_execution_runner.py"
        ),
        label="external execution worker",
    )


_MODULE_SOURCE_SHA256_AT_IMPORT: Final = _current_module_source_sha256()


def _require_current_module_source(expected: str | None = None) -> str:
    current = _current_module_source_sha256()
    if not hmac.compare_digest(current, _MODULE_SOURCE_SHA256_AT_IMPORT):
        _fail("external execution worker source changed after module import")
    if expected is not None and not hmac.compare_digest(current, expected):
        _fail("external execution capability worker source is stale")
    return current


def _require_in_container_runtime() -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() not in {"x86_64", "amd64"}
        or platform.python_version() != _EXPECTED_PYTHON_VERSION
        or os.getuid() != _EXPECTED_UID
        or os.geteuid() != _EXPECTED_UID
        or os.getgid() != _EXPECTED_GID
        or os.getegid() != _EXPECTED_GID
        or os.getuid() != os.geteuid()
        or os.getgid() != os.getegid()
    ):
        _fail("external execution worker is outside the exact in-container runtime profile")
    try:
        executable = Path(os.path.realpath(sys.executable))
    except (OSError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external execution worker interpreter identity is unavailable"
        ) from exc
    if executable != _PYTHON_EXECUTABLE:
        _fail("external execution worker interpreter path differs")
    forbidden_environment = tuple(
        sorted(
            key
            for key in os.environ
            if key
            in {
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PYTHONHOME",
                "PYTHONINSPECT",
                "PYTHONPATH",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
            }
            or key.startswith("LD_AUDIT")
        )
    )
    if forbidden_environment:
        _fail("external execution worker rejects ambient interpreter injection variables")


def _require_score_decoding_modules_absent() -> None:
    try:
        names = tuple(sys.modules)
    except RuntimeError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external execution module registry changed during inspection"
        ) from exc
    if any(type(name) is not str for name in names):
        _fail("external execution module registry contains a non-string key")
    forbidden = tuple(
        sorted(
            name
            for name in names
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_SCORE_MODULE_PREFIXES
            )
        )
    )
    if forbidden:
        _fail(
            "external execution non-decoding boundary rejects preloaded modules: "
            + ", ".join(forbidden)
        )


def _require_captured_external_outcome_consumer() -> types.FunctionType:
    module = _EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD
    if type(module) is not types.ModuleType:
        _fail("exact external outcome consumer was not loaded before the runner")
    if sys.modules.get(_EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME) is not module:
        _fail("exact external outcome consumer module identity changed")
    source_sha256 = _require_sha256(
        _EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256_INPUT,
        "external outcome consumer source",
    )
    descriptor_sha256 = _require_sha256(
        _EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256_INPUT,
        "external outcome consumer descriptor",
    )
    if not hmac.compare_digest(
        _module_source_sha256(
            getattr(module, "__file__", None),
            expected_suffix=_EXTERNAL_OUTCOME_CONSUMER_SOURCE_PATH,
            label="external outcome consumer",
        ),
        source_sha256,
    ):
        _fail("external outcome consumer source identity differs")
    descriptor = getattr(
        module, "canonical_external_outcome_consumer_descriptor_bytes", None
    )
    sink = getattr(
        module,
        "_consume_claimed_matched_v3_external_execution_payload",
        None,
    )
    replay = getattr(module, "_replay_external_outcome_consumer_guard", None)
    if (
        type(descriptor) is not types.FunctionType
        or type(sink) is not types.FunctionType
        or type(replay) is not types.FunctionType
        or sink is not _EXTERNAL_OUTCOME_CONSUMER_SINK_AT_LOAD
        or replay is not _EXTERNAL_OUTCOME_CONSUMER_REPLAY_AT_LOAD
        or getattr(module, "_CONSUMER_GUARD_AT_LOAD", None)
        is not _EXTERNAL_OUTCOME_CONSUMER_GUARD_AT_LOAD
        or getattr(module, "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION", None)
        != _EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION
        or getattr(module, "EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256", None)
        != descriptor_sha256
    ):
        _fail("external outcome consumer captured API identity differs")
    exact_replay = cast(types.FunctionType, replay)
    exact_sink = cast(types.FunctionType, sink)
    raw = descriptor()
    if type(raw) is not bytes or not hmac.compare_digest(
        _sha256(raw), descriptor_sha256
    ):
        _fail("external outcome consumer descriptor bytes differ")
    guarded = exact_replay(_EXTERNAL_OUTCOME_CONSUMER_GUARD_AT_LOAD)
    if (
        type(guarded) is not tuple
        or len(guarded) != 2
        or guarded[1] is not exact_sink
    ):
        _fail("external outcome consumer guarded surface replay differs")
    return cast(types.FunctionType, exact_sink)


def _candidate(candidate_id: object) -> _CandidateSpec:
    if type(candidate_id) is not str:
        _fail("candidate_id must be one exact string")
    spec = _CANDIDATE_BY_ID.get(candidate_id)
    if spec is None:
        _fail("candidate_id is not one frozen external candidate")
    return spec


def _entrypoint(spec: _CandidateSpec) -> str:
    return _PPO_ENTRYPOINT if spec.family == "ppo" else _CONTINUING_ENTRYPOINT


def _max_steps(spec: _CandidateSpec) -> int:
    return _PPO_ROLLOUT_COUNT if spec.family == "ppo" else _HORIZON


def _result_directory(spec: _CandidateSpec) -> str:
    configuration_suffix = spec.configuration_path.removeprefix("experiments/")
    if configuration_suffix == spec.configuration_path:
        _fail("candidate configuration is outside the experiments namespace")
    directory = configuration_suffix.rsplit("/", 1)[0]
    return f"results/{directory}/{spec.output_stem}"


def _artifact_paths(spec: _CandidateSpec) -> tuple[tuple[str, str], ...]:
    root = _result_directory(spec)
    values = [
        ("upstream_reward_npz", f"{root}/data/0.npz"),
        ("upstream_results_database", f"{root}/results.db"),
    ]
    if spec.family == "ppo":
        values.append(("upstream_video", f"{root}/{_PPO_VIDEO_RELATIVE_PATH}"))
    return tuple(values)


def _expected_directories(spec: _CandidateSpec) -> frozenset[str]:
    result: set[str] = set()
    for _kind, path in _artifact_paths(spec):
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            result.add("/".join(parts[:index]))
    return frozenset(result)


_PRODUCTION_FUNCTION_SNAPSHOT: Mapping[str, tuple[object, object]]
_PRODUCTION_CALLABLE_DEFAULT_SNAPSHOT: Mapping[
    str,
    tuple[object, object, tuple[Any, ...]],
]
_TRANSITIVE_BEHAVIOR_SNAPSHOT: Mapping[
    str,
    tuple[object, object, object, object, tuple[Any, ...]],
]
_INVARIANT_OBJECT_SNAPSHOT: Mapping[str, object]
_INVARIANT_VALUE_SNAPSHOT: tuple[Any, ...]

_TRANSITIVE_BEHAVIOR_PATHS: Final = (
    "copy",
    "copy.deepcopy",
    "hashlib",
    "hashlib.sha256",
    "hmac",
    "hmac.compare_digest",
    "json",
    "json.dumps",
    "json.loads",
    "os",
    "os.O_CLOEXEC",
    "os.O_DIRECTORY",
    "os.O_NOFOLLOW",
    "os.O_RDONLY",
    "os.P_PID",
    "os.SEEK_SET",
    "os.WEXITED",
    "os.WNOHANG",
    "os.WNOWAIT",
    "os.close",
    "os.dup",
    "os.environ",
    "os.fspath",
    "os.fstat",
    "os.fsync",
    "os.getegid",
    "os.geteuid",
    "os.getgid",
    "os.getpid",
    "os.getuid",
    "os.killpg",
    "os.lseek",
    "os.mkdir",
    "os.open",
    "os.path",
    "os.path.realpath",
    "os.read",
    "os.rmdir",
    "os.scandir",
    "os.set_blocking",
    "os.stat",
    "os.unlink",
    "os.waitid",
    "platform",
    "platform.machine",
    "platform.python_version",
    "platform.system",
    "re",
    "re.compile",
    "re.fullmatch",
    "secrets",
    "secrets.token_hex",
    "selectors",
    "selectors.BaseSelector",
    "selectors.DefaultSelector",
    "selectors.DefaultSelector.__init__",
    "selectors.DefaultSelector.close",
    "selectors.DefaultSelector.get_map",
    "selectors.DefaultSelector.register",
    "selectors.DefaultSelector.select",
    "selectors.DefaultSelector.unregister",
    "selectors.EVENT_READ",
    "signal",
    "signal.NSIG",
    "signal.SIGKILL",
    "signal.SIGTERM",
    "stat",
    "stat.S_IMODE",
    "stat.S_ISDIR",
    "stat.S_ISLNK",
    "stat.S_ISREG",
    "stat.S_IXUSR",
    "subprocess",
    "subprocess.DEVNULL",
    "subprocess.PIPE",
    "subprocess.Popen",
    "subprocess.Popen.__init__",
    "subprocess.Popen.wait",
    "subprocess.TimeoutExpired",
    "sys",
    "sys.executable",
    "types",
    "sys.modules",
    "threading",
    "threading.Lock",
    "time",
    "time.monotonic",
    "time.sleep",
    "weakref",
    "weakref.WeakKeyDictionary",
    "weakref.WeakKeyDictionary.__setitem__",
    "weakref.WeakKeyDictionary.get",
    "Path",
    "Path.__new__",
    "Path.is_absolute",
    "Path.parts",
    "Path.read_text",
    "PurePosixPath",
    "PurePosixPath.__new__",
    "PurePosixPath.parts",
    "Mapping",
    "MappingProxyType",
    "cast",
)


def _default_value_fingerprint(value: object) -> tuple[object, object]:
    if type(value) in {type(None), bool, int, float, str, bytes}:
        return type(value), value
    return type(value), id(value)


def _callable_defaults_fingerprint(value: object) -> tuple[Any, ...]:
    defaults = getattr(value, "__defaults__", None)
    kwdefaults = getattr(value, "__kwdefaults__", None)
    frozen_defaults = (
        None
        if defaults is None
        else tuple(_default_value_fingerprint(item) for item in defaults)
    )
    frozen_kwdefaults = (
        None
        if kwdefaults is None
        else tuple(
            sorted(
                (key, *_default_value_fingerprint(item))
                for key, item in kwdefaults.items()
            )
        )
    )
    return frozen_defaults, frozen_kwdefaults


def _resolve_transitive_behavior_path(label: str) -> object:
    parts = label.split(".")
    current = globals().get(parts[0])
    for attribute in parts[1:]:
        current = getattr(current, attribute, None)
    return current


def _transitive_behavior_state(
    value: object,
) -> tuple[object, object, object, object, tuple[Any, ...]]:
    return (
        value,
        getattr(value, "__code__", None),
        getattr(value, "__defaults__", None),
        getattr(value, "__kwdefaults__", None),
        _callable_defaults_fingerprint(value),
    )


def _constant_invariant_payload() -> tuple[Any, ...]:
    values = (
        EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
        EXTERNAL_EXECUTION_RUNNER_STATUS,
        EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
        _PRODUCTION_RUNNER_EXACT_SCOPE,
        _INJECTED_TEST_RUNNER_SCOPE,
        _EXECUTION_CONTRACT_SCHEMA_VERSION,
        _EXECUTION_CONTRACT_DESCRIPTOR_SHA256,
        _EXECUTION_CONTRACT_SOURCE_PATH,
        _EXECUTION_CONTRACT_SOURCE_SHA256,
        _EXTERNAL_STAGING_DESCRIPTOR_SCHEMA_VERSION,
        _EXTERNAL_STAGING_DESCRIPTOR_SHA256,
        _EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
        _SEED_TRANSPORT_SCHEMA_VERSION,
        _SEED_TRANSPORT_DESCRIPTOR_SHA256,
        _RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
        _RESULT_BRIDGE_DESCRIPTOR_SHA256,
        _CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256,
        _ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
        _EXTERNAL_OUTCOME_CONSUMER_ISOLATED_MODULE_NAME,
        _EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION,
        _EXTERNAL_OUTCOME_CONSUMER_SOURCE_PATH,
        _EXTERNAL_OUTCOME_CONSUMER_SOURCE_SHA256_INPUT,
        _EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SHA256_INPUT,
        str(_WORKLOAD_ROOT),
        str(_PRIVATE_RUNTIME_PARENT),
        str(_PYTHON_EXECUTABLE),
        _PYTHON_ARGV0,
        _EXPECTED_PYTHON_VERSION,
        _EXPECTED_UID,
        _EXPECTED_GID,
        _HORIZON,
        _PPO_ROLLOUT_STEPS,
        _PPO_ROLLOUT_COUNT,
        _UINT31_MAX,
        _MAX_TIMEOUT_SECONDS,
        _MAX_STDOUT_BYTES,
        _MAX_STDERR_BYTES,
        _MAX_EXTERNAL_NPZ_BYTES,
        _MAX_RESULTS_DATABASE_BYTES,
        _MAX_PPO_VIDEO_BYTES,
        _MAX_RECEIPT_BYTES,
        _MAX_SOURCE_FILE_BYTES,
        _MAX_EXECUTABLE_BYTES,
        _MAX_JSON_DEPTH,
        _MAX_JSON_NODES,
        _MAX_JSON_TEXT_BYTES,
        _MAX_PATH_BYTES,
        _MAX_PATH_COMPONENT_BYTES,
        _PROCESS_CLEANUP_SECONDS,
        _READ_BLOCK_BYTES,
        _MAX_INVENTORY_ENTRIES,
        _MAX_INVENTORY_DEPTH,
        _MAX_CLEANUP_ENTRIES,
        _MAX_CLEANUP_DEPTH,
        _CLEANUP_BATCH_ENTRIES,
        _MIN_PROCESS_RETURNCODE,
        _MAX_PROCESS_RETURNCODE,
        (_SHA256_RE.pattern, _SHA256_RE.flags),
        (_PRIVATE_NAME_RE.pattern, _PRIVATE_NAME_RE.flags),
        _FORBIDDEN_SCORE_MODULE_PREFIXES,
        _TRANSITIVE_BEHAVIOR_PATHS,
        _CONTINUING_ENTRYPOINT,
        _PPO_ENTRYPOINT,
        _PPO_VIDEO_RELATIVE_PATH,
        tuple(_ENTRYPOINT_SHA256.items()),
        tuple(
            tuple(
                (type(value), value)
                for value in (
                    spec.candidate_id,
                    spec.configuration_path,
                    spec.configuration_sha256,
                    spec.output_stem,
                    spec.family,
                )
            )
            for spec in _CANDIDATE_SPECS
        ),
        EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS,
        tuple((key, id(value)) for key, value in _CANDIDATE_BY_ID.items()),
        _MODULE_SOURCE_SHA256_AT_IMPORT,
        EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256,
        _DESCRIPTOR_BYTES,
    )
    return tuple((type(value), value) for value in values)


def _require_production_closure_integrity() -> None:
    try:
        function_snapshot = _PRODUCTION_FUNCTION_SNAPSHOT
        callable_default_snapshot = _PRODUCTION_CALLABLE_DEFAULT_SNAPSHOT
        transitive_snapshot = _TRANSITIVE_BEHAVIOR_SNAPSHOT
        object_snapshot = _INVARIANT_OBJECT_SNAPSHOT
        value_snapshot = _INVARIANT_VALUE_SNAPSHOT
    except NameError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external execution production closure snapshot is unavailable"
        ) from exc
    for label, (expected_function, expected_code) in function_snapshot.items():
        owner_name, separator, attribute = label.partition(".")
        owner = globals().get(owner_name)
        current = getattr(owner, attribute, None) if separator else owner
        if (
            current is not expected_function
            or getattr(current, "__code__", None) is not expected_code
        ):
            _fail(f"external execution production closure integrity differs: {label}")
    for label, (
        expected_defaults,
        expected_kwdefaults,
        expected_fingerprint,
    ) in callable_default_snapshot.items():
        owner_name, separator, attribute = label.partition(".")
        owner = globals().get(owner_name)
        current = getattr(owner, attribute, None) if separator else owner
        if (
            getattr(current, "__defaults__", None) is not expected_defaults
            or getattr(current, "__kwdefaults__", None) is not expected_kwdefaults
            or _callable_defaults_fingerprint(current) != expected_fingerprint
        ):
            _fail(f"external execution callable defaults differ: {label}")
    for label, expected_state in transitive_snapshot.items():
        current = _resolve_transitive_behavior_path(label)
        observed_state = _transitive_behavior_state(current)
        if (
            observed_state[0] is not expected_state[0]
            or observed_state[1] is not expected_state[1]
            or observed_state[2] is not expected_state[2]
            or observed_state[3] is not expected_state[3]
            or observed_state[4] != expected_state[4]
        ):
            _fail(f"external execution transitive behavior differs: {label}")
    for name, expected_object in object_snapshot.items():
        if globals().get(name) is not expected_object:
            _fail(f"external execution invariant object identity differs: {name}")
    if _constant_invariant_payload() != value_snapshot:
        _fail("external execution invariant candidate or constant content differs")


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority": False,
        "all_descendant_cleanup_proven": False,
        "candidate_qualified": False,
        "cgroup_or_container_empty_proven": False,
        "fresh_isolated_worker_process_proven": False,
        "host_oci_execution_implemented": False,
        "performance_claim_allowed": False,
        "parent_process_startup_closure_proven": False,
        "publication_authorized": False,
        "qualification_authority": False,
        "resource_observation_accepted": False,
        "result_accepted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _descriptor() -> dict[str, Any]:
    candidates = []
    for spec in _CANDIDATE_SPECS:
        entrypoint = _entrypoint(spec)
        candidates.append(
            {
                "candidate_id": spec.candidate_id,
                "family": spec.family,
                "configuration_path": spec.configuration_path,
                "configuration_sha256": spec.configuration_sha256,
                "entrypoint_path": entrypoint,
                "entrypoint_sha256": _ENTRYPOINT_SHA256[entrypoint],
                "max_steps": _max_steps(spec),
                "artifact_paths": [
                    {"kind": kind, "path": path} for kind, path in _artifact_paths(spec)
                ],
            }
        )
    return {
        "schema_version": EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        "receipt_schema_version": EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "status": EXTERNAL_EXECUTION_RUNNER_STATUS,
        "classification": (
            "score_reward_fields_not_decoded_in_container_worker_non_authorizing"
        ),
        "candidate_count": len(candidates),
        "candidate_order": list(EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS),
        "candidates": candidates,
        "bindings": {
            "execution_contract": {
                "schema_version": _EXECUTION_CONTRACT_SCHEMA_VERSION,
                "descriptor_sha256": _EXECUTION_CONTRACT_DESCRIPTOR_SHA256,
                "source_path": _EXECUTION_CONTRACT_SOURCE_PATH,
                "source_sha256": _EXECUTION_CONTRACT_SOURCE_SHA256,
                "imported": False,
            },
            "external_staging": {
                "descriptor_schema_version": _EXTERNAL_STAGING_DESCRIPTOR_SCHEMA_VERSION,
                "descriptor_sha256": _EXTERNAL_STAGING_DESCRIPTOR_SHA256,
                "manifest_schema_version": _EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION,
            },
            "seed_transport": {
                "schema_version": _SEED_TRANSPORT_SCHEMA_VERSION,
                "descriptor_sha256": _SEED_TRANSPORT_DESCRIPTOR_SHA256,
            },
            "result_bridge": {
                "schema_version": _RESULT_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
                "descriptor_sha256": _RESULT_BRIDGE_DESCRIPTOR_SHA256,
                "imported": False,
                "invoked": False,
            },
            "external_outcome_consumer": {
                "descriptor_schema_version": (
                    _EXTERNAL_OUTCOME_CONSUMER_DESCRIPTOR_SCHEMA_VERSION
                ),
                "loaded_before_runner_required_for_publication": True,
                "source_and_descriptor_caller_injected": True,
                "captured_guard_and_function_surface_replayed": True,
                "statically_imported": False,
            },
            "cpu_runtime_lock_descriptor_sha256": _CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256,
            "future_atomic_publication_descriptor_sha256": (
                _ATOMIC_PUBLICATION_DESCRIPTOR_SHA256
            ),
        },
        "runtime_role": {
            "role": "in_container_worker_only",
            "workload_root": str(_WORKLOAD_ROOT),
            "private_runtime_parent": str(_PRIVATE_RUNTIME_PARENT),
            "python_executable": str(_PYTHON_EXECUTABLE),
            "python_version": _EXPECTED_PYTHON_VERSION,
            "uid": _EXPECTED_UID,
            "gid": _EXPECTED_GID,
            "host_oci_executor_required": True,
            "host_oci_executor_implemented_here": False,
            "host_cgroup_or_container_empty_proof_required": True,
            "fresh_isolated_worker_process_required": True,
            "fresh_isolated_worker_process_proven_here": False,
            "pinned_in_container_outcome_consumer_driver_required": True,
            "pinned_in_container_outcome_consumer_driver_implemented_here": True,
            "network_or_container_api_used": False,
            "already_started_parent_loader_closure_proven": False,
        },
        "capabilities": {
            "execution": {
                "explicit_exact_opt_in": EXPLICIT_EXTERNAL_EXECUTION_OPT_IN,
                "opaque": True,
                "pid_bound": True,
                "single_use": True,
                "serializable": False,
            },
            "outcome": {
                "second_explicit_opt_in": True,
                "opaque": True,
                "pid_bound": True,
                "single_use": True,
                "serializable": False,
                "private_sealed_payload_until_claim": True,
                "public_completion_or_private_publication_exclusive": True,
                "claim_precedes_consumer_callback": True,
                "failed_private_path_retriable": False,
            },
            "completion_paths_or_bytes_are_authority": False,
        },
        "process_contract": {
            "exactly_one_candidate_per_capability": True,
            "python_flags": ["-B"],
            "isolated_mode": False,
            "exact_environment_and_private_home_replace_isolated_mode": True,
            "ambient_loader_and_python_injection_variables_rejected": True,
            "real_and_effective_uid_gid_match_required": True,
            "already_started_parent_loader_closure_proven": False,
            "pythonhashseed_environment_honored": True,
            "new_session": True,
            "stdin": "devnull",
            "bounded_stdout_stderr": True,
            "timeout_required": True,
            "leader_waited_and_reaped": True,
            "process_group_termination_and_absence_check": True,
            "all_descendant_cleanup_proven": False,
            "cgroup_or_container_empty_proven": False,
            "future_host_cgroup_or_container_empty_proof_required": True,
            "future_host_fresh_isolated_worker_process_required": True,
            "cleanup_uncertainty_fails": True,
            "working_directory": str(_WORKLOAD_ROOT),
            "fresh_private_save_checkpoint_runtime_roots": True,
        },
        "artifact_contract": {
            "opaque_npz_not_opened": True,
            "database_not_decoded": True,
            "video_not_decoded": True,
            "score_or_reward_magnitude_decoded": False,
            "exact_inventory_required": True,
            "checkpoint_root_empty_after_execution_required": True,
            "caller_declared_wall_output_artifact_bounds": {
                "maximum_stdout_bytes": _MAX_STDOUT_BYTES,
                "maximum_stderr_bytes": _MAX_STDERR_BYTES,
                "maximum_external_npz_bytes": _MAX_EXTERNAL_NPZ_BYTES,
                "maximum_results_database_bytes": _MAX_RESULTS_DATABASE_BYTES,
                "maximum_ppo_video_bytes": _MAX_PPO_VIDEO_BYTES,
                "maximum_timeout_seconds": _MAX_TIMEOUT_SECONDS,
            },
            "cpu_memory_or_cgroup_observation_performed": False,
            "resource_qualification_performed": False,
            "inventory_maximum_entries": _MAX_INVENTORY_ENTRIES,
            "inventory_maximum_depth": _MAX_INVENTORY_DEPTH,
            "cleanup_maximum_entries": _MAX_CLEANUP_ENTRIES,
            "cleanup_maximum_depth": _MAX_CLEANUP_DEPTH,
            "cleanup_batch_entries": _CLEANUP_BATCH_ENTRIES,
        },
        "test_seam": {
            "injected_process_runner_supported_internally": True,
            "injected_receipts_permanently_nonproduction": True,
            "injected_path_can_claim_production_runner_exact": False,
        },
        "score_decoding_parent_boundary": {
            "external_result_bridge_preloaded_allowed": False,
            "scorer_preloaded_allowed": False,
            "checked_at_issue_execute_and_consume": True,
            "future_direct_byte_load_required": True,
        },
        "source_identity": {
            "self_observed_at_import_and_rechecked": True,
            "self_pinned_by_future_plan_required": True,
            "loaded_module_globals_transitive_callables_and_defaults_checked": True,
            "production_runner_exact_scope": _PRODUCTION_RUNNER_EXACT_SCOPE,
            "fresh_isolated_worker_process_required": True,
            "fresh_isolated_worker_process_proven_here": False,
            "same_process_monkeypatch_resistance_claimed": False,
        },
        "claims": _claims(),
        "limitations": [
            "This module is an in-container worker, not a host OCI executor.",
            (
                "Opaque capabilities cannot cross a PID boundary; a future pinned "
                "in-container driver must consume the outcome before host transport."
            ),
            (
                "A later host executor must bind the exact image, networkless/read-only "
                "sandbox, runtime helpers, source archive, and worker source before use."
            ),
            (
                "The worker checks the selected entrypoint and configuration before and "
                "after execution; full image source closure remains a host-executor duty."
            ),
            (
                "The returned receipt and artifact bytes are nonauthorizing and cannot "
                "recreate either opaque capability."
            ),
            (
                "Artifact bytes, sizes, and digests are score/reward-bearing or may be "
                "content side channels; this runner claims only that it does not decode "
                "or branch on score/reward values."
            ),
            (
                "Only caller-declared wall-time, output-byte, and artifact-byte bounds are "
                "enforced here; CPU, memory, cgroup, and other resource observations are not."
            ),
            (
                "Process-group cleanup covers only the original process group; escaped "
                "descendants and a cgroup-or-container-empty state are not proven here."
            ),
            (
                "The future host must prove the candidate cgroup or container is empty "
                "before accepting any execution content."
            ),
            (
                "Ordinary production helper and constant replacement is checked, but "
                "same-process authority can replace exported entry points; qualification "
                "must direct-load a separately pinned worker."
            ),
            (
                "Ambient injection-variable rejection is redundant and does not prove "
                "how the already-running parent process was started."
            ),
            (
                "The loaded semantic surface is checked for ordinary rebinding, callable "
                "code, and default drift, but production still requires a fresh isolated "
                "worker process proven by the future host."
            ),
            "No reward magnitude, score, ranking, or scientific claim is decoded here.",
        ],
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
# Provisional until the pending execution-contract/staging hash ripple is complete.
EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256: Final = (
    "0f0c12a93f458ded1188185fed8c0c97e5763f5efa5151f84b70f28b2c945636"
)
if not hmac.compare_digest(
    _sha256(_DESCRIPTOR_BYTES), EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256
):
    raise AssertionError("external execution runner descriptor identity drifted")


def external_execution_runner_descriptor() -> dict[str, Any]:
    """Return detached nonauthorizing worker contract content."""

    _require_current_module_source()
    return _strict_json(_DESCRIPTOR_BYTES, maximum=_MAX_RECEIPT_BYTES)


def canonical_external_execution_runner_descriptor_bytes() -> bytes:
    """Return exact canonical worker descriptor bytes."""

    _require_current_module_source()
    return _DESCRIPTOR_BYTES


def external_execution_runner_descriptor_sha256() -> str:
    """Return the frozen descriptor digest."""

    _require_current_module_source()
    return EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256


def parse_external_execution_runner_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical worker descriptor."""

    _require_current_module_source()
    value = _strict_json(raw, maximum=_MAX_RECEIPT_BYTES)
    if raw != _DESCRIPTOR_BYTES or not hmac.compare_digest(
        _sha256(raw), EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256
    ):
        _fail("external execution runner descriptor identity drifted")
    return value


def issue_matched_v3_external_execution_capability(
    *,
    explicit_execution_opt_in: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
) -> object:
    """Issue one PID-bound single-use capability after exact explicit opt-in."""

    _require_production_closure_integrity()
    return _issue_matched_v3_external_execution_capability(
        explicit_execution_opt_in=explicit_execution_opt_in,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        test_only_injected=False,
    )


def _issue_matched_v3_external_execution_capability_for_test(
    *,
    test_only_marker: object,
    explicit_execution_opt_in: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
) -> object:
    """Issue a permanently nonproduction capability for an injected focused test."""

    _require_production_closure_integrity()
    if test_only_marker is not _INJECTED_TEST_ONLY_MARKER:
        _fail("injected external execution test issuance requires its exact marker")
    return _issue_matched_v3_external_execution_capability(
        explicit_execution_opt_in=explicit_execution_opt_in,
        candidate_id=candidate_id,
        environment_seed=environment_seed,
        agent_seed=agent_seed,
        test_only_injected=True,
    )


def _issue_matched_v3_external_execution_capability(
    *,
    explicit_execution_opt_in: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    test_only_injected: bool,
) -> object:
    if type(test_only_injected) is not bool:
        _fail("external execution capability mode must be exact")

    if (
        type(explicit_execution_opt_in) is not str
        or not hmac.compare_digest(
            explicit_execution_opt_in, EXPLICIT_EXTERNAL_EXECUTION_OPT_IN
        )
    ):
        _fail("external execution capability issuance requires exact explicit opt-in")
    spec = _candidate(candidate_id)
    environment = _require_uint31(environment_seed, "environment_seed")
    agent = _require_uint31(agent_seed, "agent_seed")
    _require_score_decoding_modules_absent()
    if not test_only_injected:
        _require_in_container_runtime()
    source_sha256 = _require_current_module_source()
    capability = _ExecutionCapability()
    with _CAPABILITY_LOCK:
        _EXECUTION_CAPABILITIES[capability] = _ExecutionState(
            pid=os.getpid(),
            status="issued",
            candidate_id=spec.candidate_id,
            environment_seed=environment,
            agent_seed=agent,
            source_sha256=source_sha256,
            test_only_injected=test_only_injected,
        )
    return capability


def _consume_execution_capability(
    capability: object,
) -> tuple[_ExecutionCapability, _ExecutionState]:
    if type(capability) is not _ExecutionCapability:
        _fail("external execution requires one authentic opaque capability")
    exact = capability
    with _CAPABILITY_LOCK:
        state = _EXECUTION_CAPABILITIES.get(exact)
        if state is None or state.status != "issued":
            _fail("external execution capability is unknown or already consumed")
        if state.pid != os.getpid():
            state.status = "consumed"
            _fail("external execution capability cannot cross a PID boundary")
        state.status = "consumed"
    _require_current_module_source(state.source_sha256)
    return exact, state


def _child_environment(runtime_proc_path: str) -> dict[str, str]:
    if re.fullmatch(r"/proc/self/fd/[0-9]+", runtime_proc_path) is None:
        _fail("private runtime procfs path is invalid")
    return {
        "ALL_PROXY": "",
        "HOME": runtime_proc_path,
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_PLATFORMS": "cpu",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": "",
        "LD_PRELOAD": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "NO_PROXY": "",
        "PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": runtime_proc_path,
        "TZ": "UTC",
        "XDG_CACHE_HOME": runtime_proc_path,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        "all_proxy": "",
        "http_proxy": "",
        "https_proxy": "",
        "no_proxy": "",
    }


def _create_child_directory(parent: int, name: str) -> _DirectoryAnchor:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        _fail("private child directory name is invalid")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "private child directory creation failed",
            filesystem_state_uncertain=True,
        ) from exc
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        removed = False
        try:
            os.rmdir(name, dir_fd=parent)
            removed = True
        except OSError:
            pass
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "private child directory cannot be retained",
            filesystem_state_uncertain=not removed,
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        os.close(descriptor)
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "private child directory identity cannot be established",
            filesystem_state_uncertain=True,
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or _directory_identity(metadata) != _directory_identity(named)
    ):
        os.close(descriptor)
        _fail(
            "private child directory identity differs",
            filesystem_state_uncertain=True,
        )
    return _DirectoryAnchor(
        path=Path(f"/proc/self/fd/{parent}/{name}"),
        descriptor=descriptor,
        identity=_directory_identity(metadata),
        owner=(metadata.st_uid, metadata.st_gid),
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _create_execution_root(parent: _DirectoryAnchor) -> tuple[str, _DirectoryAnchor]:
    if parent.mode != 0o700 or parent.owner != (os.getuid(), os.getgid()):
        _fail("private runtime parent must be owned mode 0700")
    for _attempt in range(64):
        name = f"external-{secrets.token_hex(16)}"
        if _PRIVATE_NAME_RE.fullmatch(name) is None:
            _fail("generated private execution name differs")
        try:
            root = _create_child_directory(parent.descriptor, name)
        except FileExistsError:
            continue
        named = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if _directory_identity(named) != root.identity:
            root.close()
            _fail(
                "private execution root name was substituted",
                filesystem_state_uncertain=True,
            )
        return name, root
    _fail("cannot allocate one fresh private execution root")


def _inventory_tree(root_descriptor: int) -> tuple[frozenset[str], frozenset[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    identities: set[tuple[int, int]] = set()
    entry_count = 0

    def visit(descriptor: int, prefix: tuple[str, ...], depth: int) -> None:
        nonlocal entry_count
        if depth > _MAX_INVENTORY_DEPTH:
            _fail("private artifact tree exceeds the bounded inventory depth")
        try:
            iterator = os.scandir(descriptor)
        except OSError as exc:
            raise ForagerMatchedV3ExternalExecutionRunnerError(
                "private artifact tree cannot be enumerated"
            ) from exc
        with iterator:
            for entry in iterator:
                entry_count += 1
                if entry_count > _MAX_INVENTORY_ENTRIES:
                    _fail("private artifact tree exceeds the bounded inventory entry count")
                if type(entry.name) is not str:
                    _fail("private artifact entry name is not exact text")
                path = "/".join((*prefix, entry.name))
                _validate_relative_path(path, "private artifact path")
                metadata = entry.stat(follow_symlinks=False)
                inode = (metadata.st_dev, metadata.st_ino)
                if inode in identities:
                    _fail("private artifact tree contains an inode alias")
                identities.add(inode)
                if (
                    metadata.st_uid != os.getuid()
                    or metadata.st_gid != os.getgid()
                    or metadata.st_mode & 0o022
                ):
                    _fail(f"private artifact ownership or mode differs: {path}")
                if stat.S_ISDIR(metadata.st_mode):
                    directories.add(path)
                    child = os.open(entry.name, _directory_flags(), dir_fd=descriptor)
                    try:
                        if _stat_identity(os.fstat(child)) != _stat_identity(metadata):
                            _fail(f"private artifact directory was substituted: {path}")
                        visit(child, (*prefix, entry.name), depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    files.add(path)
                else:
                    _fail(f"private artifact entry is a link or special file: {path}")

    visit(root_descriptor, (), 0)
    return frozenset(files), frozenset(directories)


def _remove_tree_contents(
    descriptor: int,
    *,
    entry_budget: list[int] | None = None,
    depth: int = 0,
) -> None:
    if entry_budget is None:
        entry_budget = [_MAX_CLEANUP_ENTRIES]
    if depth > _MAX_CLEANUP_DEPTH:
        _fail(
            "private cleanup tree exceeds the bounded cleanup depth",
            filesystem_state_uncertain=True,
        )
    while True:
        names: list[str] = []
        with os.scandir(descriptor) as iterator:
            for entry in iterator:
                if type(entry.name) is not str:
                    _fail(
                        "private cleanup entry name is not exact text",
                        filesystem_state_uncertain=True,
                    )
                names.append(entry.name)
                if len(names) >= _CLEANUP_BATCH_ENTRIES:
                    break
        if not names:
            return
        for name in names:
            entry_budget[0] -= 1
            if entry_budget[0] < 0:
                _fail(
                    "private cleanup tree exceeds the bounded cleanup entry count",
                    filesystem_state_uncertain=True,
                )
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
                try:
                    if _stat_identity(os.fstat(child)) != _stat_identity(metadata):
                        _fail(
                            "private cleanup directory was substituted",
                            filesystem_state_uncertain=True,
                        )
                    _remove_tree_contents(
                        child,
                        entry_budget=entry_budget,
                        depth=depth + 1,
                    )
                finally:
                    os.close(child)
                os.rmdir(name, dir_fd=descriptor)
            else:
                os.unlink(name, dir_fd=descriptor)


def _cleanup_execution_root(
    parent: _DirectoryAnchor,
    name: str,
    root: _DirectoryAnchor,
) -> None:
    try:
        root.verify(path_required=False)
        named = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if _directory_identity(named) != root.identity:
            _fail(
                "private execution root name changed before cleanup",
                filesystem_state_uncertain=True,
        )
        _remove_tree_contents(root.descriptor)
        with os.scandir(root.descriptor) as iterator:
            retained_entry = next(iterator, None)
        if retained_entry is not None:
            _fail(
                "private execution root retained entries after cleanup",
                filesystem_state_uncertain=True,
            )
        named_after = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if _directory_identity(named_after) != root.identity:
            _fail(
                "private execution root name changed during cleanup",
                filesystem_state_uncertain=True,
            )
        os.rmdir(name, dir_fd=parent.descriptor)
        unlinked = os.fstat(root.descriptor)
        if unlinked.st_nlink != 0:
            _fail(
                "private execution root unlink did not detach the retained inode",
                filesystem_state_uncertain=True,
            )
        root.close()
        os.fsync(parent.descriptor)
        try:
            os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        _fail(
            "private execution root remains after cleanup",
            filesystem_state_uncertain=True,
        )
    except ForagerMatchedV3ExternalExecutionRunnerError as exc:
        exc.filesystem_state_uncertain = True
        raise
    except BaseException as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "private execution root cleanup failed",
            filesystem_state_uncertain=True,
        ) from exc


def _live_process_group_member_pids(process_group: int, leader_pid: int) -> tuple[int, ...]:
    members: list[int] = []
    try:
        iterator = os.scandir("/proc")
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "process-group membership cannot be inspected",
            process_state_uncertain=True,
        ) from exc
    with iterator:
        for entry in iterator:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            if pid == leader_pid:
                continue
            try:
                raw = Path(entry.path, "stat").read_text(encoding="ascii")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError) as exc:
                raise ForagerMatchedV3ExternalExecutionRunnerError(
                    "process-group member metadata cannot be read",
                    process_state_uncertain=True,
                ) from exc
            _prefix, separator, suffix = raw.rpartition(") ")
            fields = suffix.split()
            if separator != ") " or len(fields) < 4:
                _fail("process-group member metadata is malformed", process_state_uncertain=True)
            try:
                group = int(fields[2])
            except ValueError as exc:
                raise ForagerMatchedV3ExternalExecutionRunnerError(
                    "process-group member identity is malformed",
                    process_state_uncertain=True,
                ) from exc
            if group == process_group and fields[0] not in {"X", "Z", "x"}:
                members.append(pid)
    return tuple(sorted(members))


def _wait_leader_unreaped(process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            result = os.waitid(os.P_PID, process.pid, options)
        except (ChildProcessError, OSError) as exc:
            raise ForagerMatchedV3ExternalExecutionRunnerError(
                "process leader cannot be observed before cleanup",
                process_state_uncertain=True,
            ) from exc
        if result is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.01, remaining))


def _terminate_anchored_process_group(process: subprocess.Popen[bytes]) -> None:
    group = process.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "process group cannot receive SIGTERM",
            process_state_uncertain=True,
        ) from exc
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if not _live_process_group_member_pids(group, process.pid):
            return
        time.sleep(0.01)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "process group cannot receive SIGKILL",
            process_state_uncertain=True,
        ) from exc
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if not _live_process_group_member_pids(group, process.pid):
            return
        time.sleep(0.01)
    if _live_process_group_member_pids(group, process.pid):
        _fail("process group retained descendants after SIGKILL", process_state_uncertain=True)


def _default_process_runner(
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    executable_descriptor: int,
    inherited_descriptors: tuple[int, ...],
    working_directory: str,
    timeout_seconds: int,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> BoundedExternalProcessResult:
    """Run one process with bounded pipes and anchored process-group cleanup."""

    if (
        type(argv) is not tuple
        or not argv
        or any(type(value) is not str for value in argv)
        or type(environment) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in environment.items())
        or type(executable_descriptor) is not int
        or executable_descriptor < 0
        or type(inherited_descriptors) is not tuple
        or any(type(value) is not int or value < 0 for value in inherited_descriptors)
        or type(working_directory) is not str
        or not working_directory.startswith("/proc/self/fd/")
    ):
        _fail("bounded external process request fields are invalid")
    passed = tuple(sorted({executable_descriptor, *inherited_descriptors}))
    for descriptor in passed:
        try:
            os.fstat(descriptor)
        except OSError as exc:
            raise ForagerMatchedV3ExternalExecutionRunnerError(
                "bounded external process inherited descriptor is not open"
            ) from exc
    try:
        process = subprocess.Popen(
            list(argv),
            executable=f"/proc/self/fd/{executable_descriptor}",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            cwd=working_directory,
            pass_fds=passed,
            start_new_session=True,
            env=dict(environment),
            umask=0o077,
        )
    except (OSError, ValueError) as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external workload process could not start"
        ) from exc
    stdout = bytearray()
    stderr = bytearray()
    selector: selectors.BaseSelector | None = None
    streams: dict[int, tuple[Any, bytearray, int]] = {}
    timed_out = False
    output_limit_exceeded = False
    leader_reaped = False
    returncode: int | None = None
    failure: BaseException | None = None
    deadline = time.monotonic() + timeout_seconds

    def finish(*, terminate_first: bool) -> int:
        nonlocal leader_reaped, timed_out
        if terminate_first:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise ForagerMatchedV3ExternalExecutionRunnerError(
                    "external workload process group cannot be killed",
                    process_state_uncertain=True,
                ) from exc
        wait_timeout = _PROCESS_CLEANUP_SECONDS
        if not terminate_first and not timed_out and not output_limit_exceeded:
            wait_timeout = max(0.0, deadline - time.monotonic())
        try:
            _wait_leader_unreaped(process, wait_timeout)
        except subprocess.TimeoutExpired:
            if not terminate_first and not output_limit_exceeded:
                timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _wait_leader_unreaped(process, _PROCESS_CLEANUP_SECONDS)
        _terminate_anchored_process_group(process)
        result = process.wait(timeout=_PROCESS_CLEANUP_SECONDS)
        leader_reaped = True
        return result

    try:
        if process.stdout is None or process.stderr is None:
            _fail(
                "external workload process pipes are unavailable",
                process_state_uncertain=True,
            )
        selector = selectors.DefaultSelector()
        for stream, destination, maximum in (
            (process.stdout, stdout, stdout_limit_bytes),
            (process.stderr, stderr, stderr_limit_bytes),
        ):
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(stream, selectors.EVENT_READ, descriptor)
            streams[descriptor] = (stream, destination, maximum)
        forced_deadline: float | None = None
        while selector.get_map():
            now = time.monotonic()
            if not timed_out and not output_limit_exceeded and now >= deadline:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                forced_deadline = time.monotonic() + _PROCESS_CLEANUP_SECONDS
            if forced_deadline is not None and now >= forced_deadline:
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                    cast(Any, key.fileobj).close()
                break
            timeout = 0.1
            if not timed_out and not output_limit_exceeded:
                timeout = max(0.0, min(timeout, deadline - now))
            for key, _mask in selector.select(timeout):
                descriptor = cast(int, key.data)
                stream, destination, maximum = streams[descriptor]
                try:
                    block = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(stream)
                    stream.close()
                    continue
                available = max(0, maximum - len(destination))
                if available:
                    destination.extend(block[:available])
                if len(block) > available and not output_limit_exceeded:
                    output_limit_exceeded = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    forced_deadline = time.monotonic() + _PROCESS_CLEANUP_SECONDS
        returncode = finish(terminate_first=False)
    except BaseException as exc:
        failure = exc
    finally:
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"selector cleanup also failed: {type(exc).__name__}: {exc}"
                    )
        for stream, _destination, _maximum in streams.values():
            try:
                if not stream.closed:
                    stream.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"pipe cleanup also failed: {type(exc).__name__}: {exc}"
                    )
        if not leader_reaped:
            try:
                returncode = finish(terminate_first=True)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"process cleanup also failed: {type(exc).__name__}: {exc}"
                    )
    if failure is not None:
        if isinstance(failure, ForagerMatchedV3ExternalExecutionRunnerError):
            raise failure
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external workload monitoring or cleanup failed",
            process_state_uncertain=not leader_reaped,
        ) from failure
    if returncode is None:
        _fail("external workload return status is absent", process_state_uncertain=True)
    return BoundedExternalProcessResult(
        returncode=returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        timed_out=timed_out,
        output_limit_exceeded=output_limit_exceeded,
    )


def _run_process(
    runner: ExternalProcessRunner,
    argv: tuple[str, ...],
    *,
    environment: dict[str, str],
    executable_descriptor: int,
    inherited_descriptors: tuple[int, ...],
    working_directory: str,
    timeout_seconds: int,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> BoundedExternalProcessResult:
    try:
        result = runner(
            argv,
            environment=environment,
            executable_descriptor=executable_descriptor,
            inherited_descriptors=inherited_descriptors,
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
            stdout_limit_bytes=stdout_limit_bytes,
            stderr_limit_bytes=stderr_limit_bytes,
        )
    except ForagerMatchedV3ExternalExecutionRunnerError:
        raise
    except BaseException as exc:
        raise ForagerMatchedV3ExternalExecutionRunnerError(
            "external workload process runner failed",
            process_state_uncertain=True,
        ) from exc
    if type(result) is not BoundedExternalProcessResult:
        _fail("external workload process runner returned a noncanonical result")
    returncode = _require_process_returncode(
        result.returncode, "external workload process returncode"
    )
    if len(result.stdout) > stdout_limit_bytes or len(result.stderr) > stderr_limit_bytes:
        _fail("external workload process runner exceeded a declared output ceiling")
    if result.timed_out:
        _fail("external workload timed out after bounded process cleanup")
    if result.output_limit_exceeded:
        _fail("external workload exceeded a declared output ceiling")
    if returncode != 0:
        _fail(f"external workload exited nonzero: {returncode}")
    return result


def _artifact_record(kind: str, path: str, raw: bytes) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path,
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _receipt_body(
    *,
    context: _ExecutionContext,
    spec: _CandidateSpec,
    environment_seed: int,
    agent_seed: int,
    worker_source_sha256: str,
    source_preflight: tuple[dict[str, Any], dict[str, Any]],
    source_postflight: tuple[dict[str, Any], dict[str, Any]],
    source_root: _DirectoryAnchor,
    executable: _ExecutableAnchor,
    execution_name: str,
    argv: tuple[str, ...],
    environment: Mapping[str, str],
    process_result: BoundedExternalProcessResult,
    artifacts: tuple[tuple[str, str, bytes], ...],
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    maximum_external_npz_bytes: int,
    maximum_results_database_bytes: int,
    maximum_ppo_video_bytes: int,
) -> dict[str, Any]:
    entrypoint = _entrypoint(spec)
    return {
        "schema_version": EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "status": (
            "external_in_container_worker_completed_unqualified_non_authorizing"
            if context.production_runner_exact
            else "external_injected_test_seam_completed_unqualified_non_authorizing"
        ),
        "classification": (
            "score_reward_bearing_content_uninterpreted_by_runner_non_authorizing"
        ),
        "descriptor_binding": {
            "schema_version": EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256,
        },
        "worker_source_sha256": worker_source_sha256,
        "candidate": {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "configuration_path": spec.configuration_path,
            "configuration_sha256": spec.configuration_sha256,
            "entrypoint_path": entrypoint,
            "entrypoint_sha256": _ENTRYPOINT_SHA256[entrypoint],
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
            "max_steps": _max_steps(spec),
            "interaction_horizon": _HORIZON,
            "ppo_rollout_steps": _PPO_ROLLOUT_STEPS if spec.family == "ppo" else None,
            "ppo_rollout_count": _PPO_ROLLOUT_COUNT if spec.family == "ppo" else None,
        },
        "source": {
            "root_path": str(context.workload_root),
            "root_device": source_root.identity[0],
            "root_inode": source_root.identity[1],
            "selected_preflight": list(source_preflight),
            "selected_postflight": list(source_postflight),
            "selected_members_equal": source_preflight == source_postflight,
            "full_image_source_closure_reverified_here": False,
        },
        "runtime": {
            "role": (
                "in_container_worker_only"
                if context.production_runner_exact
                else "injected_test_seam_only"
            ),
            "python_version": _EXPECTED_PYTHON_VERSION,
            "executable_path": str(context.python_executable),
            "executable_sha256": executable.sha256,
            "executable_size_bytes": executable.size_bytes,
            "uid": _EXPECTED_UID,
            "gid": _EXPECTED_GID,
            "execution_root_name": execution_name,
            "working_directory": f"/proc/self/fd/{source_root.descriptor}",
            "argv": list(argv),
            "environment": dict(environment),
            "stdin": "devnull",
            "new_session": True,
            "production_runner_exact": context.production_runner_exact,
            "production_runner_exact_scope": (
                _PRODUCTION_RUNNER_EXACT_SCOPE
                if context.production_runner_exact
                else _INJECTED_TEST_RUNNER_SCOPE
            ),
            "closure_integrity_checked": context.closure_integrity_checked,
            "loaded_transitive_behavior_and_defaults_checked": (
                context.closure_integrity_checked
            ),
            "test_only_process_runner_injected": context.test_only_process_runner_injected,
            "fresh_isolated_worker_process_proven_here": False,
            "future_host_fresh_isolated_worker_process_required": True,
            "parent_runtime_profile_checked": context.production_runner_exact,
            "parent_ambient_injection_variables_absent": context.production_runner_exact,
            "real_effective_uid_gid_match_checked": context.production_runner_exact,
            "process_group_cleanup_completed": context.production_runner_exact,
            "all_descendant_cleanup_proven": False,
            "cgroup_or_container_empty_proven": False,
            "future_host_cgroup_or_container_empty_proof_required": True,
        },
        "process": {
            "returncode": process_result.returncode,
            "timed_out": process_result.timed_out,
            "output_limit_exceeded": process_result.output_limit_exceeded,
            "stdout": {
                "sha256": _sha256(process_result.stdout),
                "size_bytes": len(process_result.stdout),
            },
            "stderr": {
                "sha256": _sha256(process_result.stderr),
                "size_bytes": len(process_result.stderr),
            },
        },
        "artifacts": [
            _artifact_record(kind, path, raw) for kind, path, raw in artifacts
        ],
        "inventory": {
            "exact_files_verified": True,
            "exact_directories_verified": True,
            "missing_or_extra_entries": False,
            "checkpoint_root_empty_after_execution": True,
            "save_and_checkpoint_roots_fresh_private_and_distinct": True,
        },
        "ceilings": {
            "timeout_seconds": timeout_seconds,
            "maximum_stdout_bytes": maximum_stdout_bytes,
            "maximum_stderr_bytes": maximum_stderr_bytes,
            "maximum_external_npz_bytes": maximum_external_npz_bytes,
            "maximum_results_database_bytes": maximum_results_database_bytes,
            "maximum_ppo_video_bytes": maximum_ppo_video_bytes,
        },
        "runner_content_handling": {
            "npz_container_opened": False,
            "database_decoded": False,
            "video_decoded": False,
            "reward_magnitudes_decoded": False,
            "score_computed": False,
            "ranking_computed": False,
        },
        "claims": _claims(),
        "limitations": [
            (
                "This receipt is an unqualified in-container worker record."
                if context.production_runner_exact
                else "This receipt is an injected-test-only nonproduction record."
            ),
            "A separately authenticated host OCI executor and publisher remain required.",
            (
                "Original-process-group cleanup does not prove escaped descendants absent; "
                "the future host must prove the cgroup or container empty."
            ),
            (
                "A future host must prove this worker ran in a fresh isolated process; "
                "the receipt cannot establish that property itself."
            ),
            "Artifact digests do not accept or score a result.",
        ],
    }


def _receipt_payload(**kwargs: Any) -> dict[str, Any]:
    body = _receipt_body(**kwargs)
    return {**body, "receipt_body_sha256": _sha256(_canonical_json(body))}


def _validate_artifact_inputs(
    receipt: Mapping[str, Any],
    *,
    spec: _CandidateSpec,
    upstream_reward_npz: bytes,
    upstream_results_database: bytes,
    upstream_video: bytes | None,
    stdout: bytes,
    stderr: bytes,
) -> None:
    if (
        type(upstream_reward_npz) is not bytes
        or type(upstream_results_database) is not bytes
        or (upstream_video is not None and type(upstream_video) is not bytes)
        or type(stdout) is not bytes
        or type(stderr) is not bytes
    ):
        _fail("external execution replay content must use exact immutable bytes")
    if (
        not upstream_reward_npz
        or not upstream_results_database
        or (upstream_video is not None and not upstream_video)
    ):
        _fail("external execution replay artifacts must be nonempty")
    process = _require_exact_object(
        receipt.get("process"),
        frozenset(
            {
                "returncode",
                "timed_out",
                "output_limit_exceeded",
                "stdout",
                "stderr",
            }
        ),
        "external execution receipt process",
    )
    returncode = _require_process_returncode(
        process["returncode"], "external execution receipt process returncode"
    )
    if (
        returncode != 0
        or process["timed_out"] is not False
        or process["output_limit_exceeded"] is not False
    ):
        _fail("external execution receipt process outcome differs")
    expected_streams = {"stdout": stdout, "stderr": stderr}
    for name, raw in expected_streams.items():
        record = _require_exact_object(
            process.get(name),
            frozenset({"sha256", "size_bytes"}),
            f"external execution receipt {name}",
        )
        if (
            _require_nonnegative_int(
                record["size_bytes"], f"external execution receipt {name} size"
            )
            != len(raw)
            or _require_sha256(
                record["sha256"], f"external execution receipt {name} digest"
            )
            != _sha256(raw)
        ):
            _fail(f"external execution receipt {name} binding differs")
    records = receipt.get("artifacts")
    if type(records) is not list:
        _fail("external execution receipt artifact records are invalid")
    content = [upstream_reward_npz, upstream_results_database]
    if upstream_video is not None:
        content.append(upstream_video)
    if len(records) != len(content):
        _fail("external execution receipt artifact cardinality differs")
    expected_paths = _artifact_paths(spec)
    for record, (kind, path), raw in zip(records, expected_paths, content, strict=True):
        exact_record = _require_exact_object(
            record,
            frozenset({"kind", "path", "sha256", "size_bytes"}),
            f"external execution receipt artifact {kind}",
        )
        if (
            exact_record["kind"] != kind
            or exact_record["path"] != path
            or _require_nonnegative_int(
                exact_record["size_bytes"], f"external execution artifact {kind} size"
            )
            != len(raw)
            or _require_sha256(
                exact_record["sha256"], f"external execution artifact {kind} digest"
            )
            != _sha256(raw)
        ):
            _fail(f"external execution receipt artifact binding differs: {kind}")


def parse_matched_v3_external_execution_receipt(
    raw: bytes,
    *,
    expected_receipt_sha256: str,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    upstream_reward_npz: bytes,
    upstream_results_database: bytes,
    upstream_video: bytes | None,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    """Strictly replay one nonauthorizing receipt without decoding artifact content."""

    _require_production_closure_integrity()
    _require_current_module_source()
    expected = _require_sha256(expected_receipt_sha256, "expected receipt")
    if type(raw) is not bytes or not hmac.compare_digest(_sha256(raw), expected):
        _fail("external execution receipt full-file digest differs")
    receipt = _strict_json(raw, maximum=_MAX_RECEIPT_BYTES)
    expected_keys = {
        "schema_version",
        "status",
        "classification",
        "descriptor_binding",
        "worker_source_sha256",
        "candidate",
        "source",
        "runtime",
        "process",
        "artifacts",
        "inventory",
        "ceilings",
        "runner_content_handling",
        "claims",
        "limitations",
        "receipt_body_sha256",
    }
    if set(receipt) != expected_keys:
        _fail("external execution receipt root fields differ")
    if (
        receipt["schema_version"] != EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION
        or receipt["classification"]
        != "score_reward_bearing_content_uninterpreted_by_runner_non_authorizing"
        or not _exact_json_equal(
            receipt["descriptor_binding"],
            {
                "schema_version": EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
                "sha256": EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256,
            },
        )
    ):
        _fail("external execution receipt identity fields differ")
    worker_source = _require_sha256(receipt["worker_source_sha256"], "worker source")
    _require_current_module_source(worker_source)
    candidate = receipt["candidate"]
    spec = _candidate(candidate_id)
    environment = _require_uint31(environment_seed, "environment_seed")
    agent = _require_uint31(agent_seed, "agent_seed")
    entrypoint = _entrypoint(spec)
    expected_candidate = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "configuration_path": spec.configuration_path,
        "configuration_sha256": spec.configuration_sha256,
        "entrypoint_path": entrypoint,
        "entrypoint_sha256": _ENTRYPOINT_SHA256[entrypoint],
        "environment_seed": environment,
        "agent_seed": agent,
        "max_steps": _max_steps(spec),
        "interaction_horizon": _HORIZON,
        "ppo_rollout_steps": _PPO_ROLLOUT_STEPS if spec.family == "ppo" else None,
        "ppo_rollout_count": _PPO_ROLLOUT_COUNT if spec.family == "ppo" else None,
    }
    if not _exact_json_equal(candidate, expected_candidate):
        _fail("external execution receipt candidate binding differs")

    source = _require_exact_object(
        receipt["source"],
        frozenset(
            {
                "root_path",
                "root_device",
                "root_inode",
                "selected_preflight",
                "selected_postflight",
                "selected_members_equal",
                "full_image_source_closure_reverified_here",
            }
        ),
        "external execution receipt source",
    )
    if (
        type(source["root_path"]) is not str
        or source["selected_members_equal"] is not True
        or source["full_image_source_closure_reverified_here"] is not False
        or _require_nonnegative_int(source["root_device"], "source root device") < 0
        or _require_nonnegative_int(source["root_inode"], "source root inode") < 1
    ):
        _fail("external execution receipt source identity differs")
    expected_source_paths = (
        (entrypoint, _ENTRYPOINT_SHA256[entrypoint]),
        (spec.configuration_path, spec.configuration_sha256),
    )
    preflight = source["selected_preflight"]
    postflight = source["selected_postflight"]
    if (
        type(preflight) is not list
        or type(postflight) is not list
        or len(preflight) != 2
        or not _exact_json_equal(preflight, postflight)
    ):
        _fail("external execution receipt selected-source replay differs")
    for record, (path, digest) in zip(
        preflight, expected_source_paths, strict=True
    ):
        exact_record = _require_exact_object(
            record,
            frozenset({"path", "sha256", "size_bytes"}),
            "external execution receipt selected source member",
        )
        if (
            exact_record["path"] != path
            or exact_record["sha256"] != digest
            or not 1
            <= _require_nonnegative_int(
                exact_record["size_bytes"], "selected source size"
            )
            <= _MAX_SOURCE_FILE_BYTES
        ):
            _fail("external execution receipt selected source member differs")

    runtime = _require_exact_object(
        receipt["runtime"],
        frozenset(
            {
                "role",
                "python_version",
                "executable_path",
                "executable_sha256",
                "executable_size_bytes",
                "uid",
                "gid",
                "execution_root_name",
                "working_directory",
                "argv",
                "environment",
                "stdin",
                "new_session",
                "production_runner_exact",
                "production_runner_exact_scope",
                "closure_integrity_checked",
                "loaded_transitive_behavior_and_defaults_checked",
                "test_only_process_runner_injected",
                "fresh_isolated_worker_process_proven_here",
                "future_host_fresh_isolated_worker_process_required",
                "parent_runtime_profile_checked",
                "parent_ambient_injection_variables_absent",
                "real_effective_uid_gid_match_checked",
                "process_group_cleanup_completed",
                "all_descendant_cleanup_proven",
                "cgroup_or_container_empty_proven",
                "future_host_cgroup_or_container_empty_proof_required",
            }
        ),
        "external execution receipt runtime",
    )
    working_directory = runtime["working_directory"]
    execution_root_name = runtime["execution_root_name"]
    argv_value = runtime["argv"]
    child_environment = runtime["environment"]
    _require_sha256(runtime["executable_sha256"], "worker executable digest")
    production_runner_exact = runtime["production_runner_exact"]
    test_only_process_runner_injected = runtime["test_only_process_runner_injected"]
    if (
        type(production_runner_exact) is not bool
        or runtime["closure_integrity_checked"] is not True
        or type(test_only_process_runner_injected) is not bool
    ):
        _fail("external execution receipt closure or process-runner marker differs")
    expected_status = (
        "external_in_container_worker_completed_unqualified_non_authorizing"
        if production_runner_exact
        else "external_injected_test_seam_completed_unqualified_non_authorizing"
    )
    expected_role = (
        "in_container_worker_only"
        if production_runner_exact
        else "injected_test_seam_only"
    )
    expected_process_group_cleanup = production_runner_exact
    expected_test_injection = not production_runner_exact
    expected_runner_scope = (
        _PRODUCTION_RUNNER_EXACT_SCOPE
        if production_runner_exact
        else _INJECTED_TEST_RUNNER_SCOPE
    )
    source_root_path = source["root_path"]
    executable_path = runtime["executable_path"]
    if type(executable_path) is not str:
        _fail("external execution receipt executable path is invalid")
    _validate_absolute_path(Path(source_root_path), "receipt source root")
    _validate_absolute_path(Path(executable_path), "receipt executable")
    if (
        receipt["status"] != expected_status
        or runtime["role"] != expected_role
        or runtime["python_version"] != _EXPECTED_PYTHON_VERSION
        or not 1
        <= _require_nonnegative_int(
            runtime["executable_size_bytes"], "worker executable size"
        )
        <= _MAX_EXECUTABLE_BYTES
        or runtime["uid"] != _EXPECTED_UID
        or runtime["gid"] != _EXPECTED_GID
        or runtime["production_runner_exact_scope"] != expected_runner_scope
        or runtime["loaded_transitive_behavior_and_defaults_checked"] is not True
        or test_only_process_runner_injected is not expected_test_injection
        or runtime["fresh_isolated_worker_process_proven_here"] is not False
        or runtime["future_host_fresh_isolated_worker_process_required"] is not True
        or runtime["parent_runtime_profile_checked"] is not production_runner_exact
        or runtime["parent_ambient_injection_variables_absent"]
        is not production_runner_exact
        or runtime["real_effective_uid_gid_match_checked"]
        is not production_runner_exact
        or type(execution_root_name) is not str
        or _PRIVATE_NAME_RE.fullmatch(execution_root_name) is None
        or type(working_directory) is not str
        or re.fullmatch(r"/proc/self/fd/[0-9]+", working_directory) is None
        or runtime["stdin"] != "devnull"
        or runtime["new_session"] is not True
        or runtime["process_group_cleanup_completed"]
        is not expected_process_group_cleanup
        or runtime["all_descendant_cleanup_proven"] is not False
        or runtime["cgroup_or_container_empty_proven"] is not False
        or runtime["future_host_cgroup_or_container_empty_proof_required"] is not True
        or type(argv_value) is not list
        or any(type(value) is not str for value in argv_value)
        or type(child_environment) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in child_environment.items()
        )
    ):
        _fail("external execution receipt runtime identity differs")
    if production_runner_exact and (
        source_root_path != str(_WORKLOAD_ROOT)
        or executable_path != str(_PYTHON_EXECUTABLE)
    ):
        _fail("external execution receipt production runtime path differs")
    argv = cast(list[str], argv_value)
    if len(argv) != 18:
        _fail("external execution receipt argv cardinality differs")
    save_proc = argv[14]
    checkpoint_proc = argv[16]
    runtime_proc = child_environment.get("TMPDIR")
    proc_paths = (working_directory, save_proc, checkpoint_proc, runtime_proc)
    if (
        type(runtime_proc) is not str
        or any(
            type(path) is not str
            or re.fullmatch(r"/proc/self/fd/[0-9]+", path) is None
            for path in proc_paths
        )
        or len(set(proc_paths)) != 4
    ):
        _fail("external execution receipt inherited descriptor paths differ")
    expected_argv0 = _PYTHON_ARGV0 if production_runner_exact else argv[0]
    if not production_runner_exact:
        _validate_absolute_path(Path(expected_argv0), "receipt injected-test argv0")
    expected_argv = [
        expected_argv0,
        "-B",
        entrypoint,
        "--exp",
        spec.configuration_path,
        "--idxs",
        "0",
        "--environment_seed",
        str(environment),
        "--agent_seed",
        str(agent),
        "--max_steps",
        str(_max_steps(spec)),
        "--save_path",
        save_proc,
        "--checkpoint_path",
        checkpoint_proc,
        "--silent",
    ]
    if argv != expected_argv or child_environment != _child_environment(runtime_proc):
        _fail("external execution receipt argv or environment differs")

    ceilings = _require_exact_object(
        receipt["ceilings"],
        frozenset(
            {
                "timeout_seconds",
                "maximum_stdout_bytes",
                "maximum_stderr_bytes",
                "maximum_external_npz_bytes",
                "maximum_results_database_bytes",
                "maximum_ppo_video_bytes",
            }
        ),
        "external execution receipt ceilings",
    )
    timeout = _require_ceiling(
        ceilings["timeout_seconds"],
        "receipt timeout_seconds",
        maximum=_MAX_TIMEOUT_SECONDS,
        zero=False,
    )
    del timeout
    stdout_ceiling = _require_ceiling(
        ceilings["maximum_stdout_bytes"],
        "receipt maximum_stdout_bytes",
        maximum=_MAX_STDOUT_BYTES,
        zero=True,
    )
    stderr_ceiling = _require_ceiling(
        ceilings["maximum_stderr_bytes"],
        "receipt maximum_stderr_bytes",
        maximum=_MAX_STDERR_BYTES,
        zero=True,
    )
    npz_ceiling = _require_ceiling(
        ceilings["maximum_external_npz_bytes"],
        "receipt maximum_external_npz_bytes",
        maximum=_MAX_EXTERNAL_NPZ_BYTES,
        zero=False,
    )
    database_ceiling = _require_ceiling(
        ceilings["maximum_results_database_bytes"],
        "receipt maximum_results_database_bytes",
        maximum=_MAX_RESULTS_DATABASE_BYTES,
        zero=False,
    )
    video_ceiling = _require_ceiling(
        ceilings["maximum_ppo_video_bytes"],
        "receipt maximum_ppo_video_bytes",
        maximum=_MAX_PPO_VIDEO_BYTES,
        zero=spec.family != "ppo",
    )
    if (
        len(stdout) > stdout_ceiling
        or len(stderr) > stderr_ceiling
        or len(upstream_reward_npz) > npz_ceiling
        or len(upstream_results_database) > database_ceiling
        or (upstream_video is not None and len(upstream_video) > video_ceiling)
        or (spec.family == "continuing" and video_ceiling != 0)
        or (spec.family == "ppo" and video_ceiling == 0)
    ):
        _fail("external execution receipt content exceeds its declared ceiling")
    claims = receipt["claims"]
    if not _exact_json_equal(claims, _claims()):
        _fail("external execution receipt authority denial differs")
    handling = receipt["runner_content_handling"]
    if (
        type(handling) is not dict
        or set(handling)
        != {
            "npz_container_opened",
            "database_decoded",
            "video_decoded",
            "reward_magnitudes_decoded",
            "score_computed",
            "ranking_computed",
        }
        or any(value is not False for value in handling.values())
    ):
        _fail("external execution receipt non-decoding record differs")
    inventory = receipt["inventory"]
    expected_inventory = {
        "exact_files_verified": True,
        "exact_directories_verified": True,
        "missing_or_extra_entries": False,
        "checkpoint_root_empty_after_execution": True,
        "save_and_checkpoint_roots_fresh_private_and_distinct": True,
    }
    if not _exact_json_equal(inventory, expected_inventory):
        _fail("external execution receipt inventory attestation differs")
    expected_limitations = [
        (
            "This receipt is an unqualified in-container worker record."
            if production_runner_exact
            else "This receipt is an injected-test-only nonproduction record."
        ),
        "A separately authenticated host OCI executor and publisher remain required.",
        (
            "Original-process-group cleanup does not prove escaped descendants absent; "
            "the future host must prove the cgroup or container empty."
        ),
        (
            "A future host must prove this worker ran in a fresh isolated process; "
            "the receipt cannot establish that property itself."
        ),
        "Artifact digests do not accept or score a result.",
    ]
    if not _exact_json_equal(receipt["limitations"], expected_limitations):
        _fail("external execution receipt limitations differ")
    _validate_artifact_inputs(
        receipt,
        spec=spec,
        upstream_reward_npz=upstream_reward_npz,
        upstream_results_database=upstream_results_database,
        upstream_video=upstream_video,
        stdout=stdout,
        stderr=stderr,
    )
    supplied_body_sha256 = _require_sha256(
        receipt["receipt_body_sha256"], "receipt body"
    )
    body = dict(receipt)
    body.pop("receipt_body_sha256")
    if not hmac.compare_digest(_sha256(_canonical_json(body)), supplied_body_sha256):
        _fail("external execution receipt body digest differs")
    return copy.deepcopy(receipt)


def execute_matched_v3_external_candidate(
    *,
    execution_capability: object,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    maximum_external_npz_bytes: int,
    maximum_results_database_bytes: int,
    maximum_ppo_video_bytes: int,
) -> object:
    """Execute one exact candidate and return only an opaque outcome capability."""

    _require_production_closure_integrity()
    exact_capability, state = _consume_execution_capability(execution_capability)
    if state.test_only_injected:
        _fail("an injected-test capability cannot enter the production runner")
    _require_in_container_runtime()
    context = _ExecutionContext(
        workload_root=_WORKLOAD_ROOT,
        private_runtime_parent=_PRIVATE_RUNTIME_PARENT,
        python_executable=_PYTHON_EXECUTABLE,
        python_argv0=_PYTHON_ARGV0,
        process_runner=_PRODUCTION_PROCESS_RUNNER,
        source_member_identity=_PRODUCTION_SOURCE_MEMBER_IDENTITY,
        cleanup_execution_root=_PRODUCTION_CLEANUP_EXECUTION_ROOT,
        production_runner_exact=True,
        test_only_process_runner_injected=False,
        closure_integrity_checked=True,
    )
    return _execute_matched_v3_external_candidate(
        exact_capability=exact_capability,
        state=state,
        context=context,
        timeout_seconds=timeout_seconds,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
        maximum_external_npz_bytes=maximum_external_npz_bytes,
        maximum_results_database_bytes=maximum_results_database_bytes,
        maximum_ppo_video_bytes=maximum_ppo_video_bytes,
    )


def _execute_matched_v3_external_candidate_for_test(
    *,
    test_only_marker: object,
    execution_capability: object,
    workload_root: Path,
    private_runtime_parent: Path,
    python_executable: Path,
    python_argv0: str,
    process_runner: ExternalProcessRunner,
    source_member_identity: _SourceMemberIdentity,
    cleanup_execution_root: _CleanupExecutionRoot | None = None,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    maximum_external_npz_bytes: int,
    maximum_results_database_bytes: int,
    maximum_ppo_video_bytes: int,
) -> object:
    """Run the injected test seam, which can never claim the production runner."""

    _require_production_closure_integrity()
    if test_only_marker is not _INJECTED_TEST_ONLY_MARKER:
        _fail("injected external execution test requires its exact marker")
    exact_capability, state = _consume_execution_capability(execution_capability)
    if not state.test_only_injected:
        _fail("a production capability cannot enter the injected test runner")
    _validate_absolute_path(workload_root, "injected-test workload root")
    _validate_absolute_path(private_runtime_parent, "injected-test runtime parent")
    _validate_absolute_path(python_executable, "injected-test Python executable")
    if type(python_argv0) is not str:
        _fail("injected-test Python argv0 must be exact text")
    _validate_absolute_path(Path(python_argv0), "injected-test Python argv0")
    if not callable(process_runner) or not callable(source_member_identity):
        _fail("injected-test process or source seam is not callable")
    cleanup = cleanup_execution_root or _PRODUCTION_CLEANUP_EXECUTION_ROOT
    if not callable(cleanup):
        _fail("injected-test cleanup seam is not callable")
    context = _ExecutionContext(
        workload_root=workload_root,
        private_runtime_parent=private_runtime_parent,
        python_executable=python_executable,
        python_argv0=python_argv0,
        process_runner=process_runner,
        source_member_identity=source_member_identity,
        cleanup_execution_root=cleanup,
        production_runner_exact=False,
        test_only_process_runner_injected=True,
        closure_integrity_checked=True,
    )
    return _execute_matched_v3_external_candidate(
        exact_capability=exact_capability,
        state=state,
        context=context,
        timeout_seconds=timeout_seconds,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
        maximum_external_npz_bytes=maximum_external_npz_bytes,
        maximum_results_database_bytes=maximum_results_database_bytes,
        maximum_ppo_video_bytes=maximum_ppo_video_bytes,
    )


def _execute_matched_v3_external_candidate(
    *,
    exact_capability: _ExecutionCapability,
    state: _ExecutionState,
    context: _ExecutionContext,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    maximum_external_npz_bytes: int,
    maximum_results_database_bytes: int,
    maximum_ppo_video_bytes: int,
) -> object:
    _require_production_closure_integrity()
    if context.production_runner_exact:
        if (
            state.test_only_injected
            or context.test_only_process_runner_injected
            or context.closure_integrity_checked is not True
            or context.workload_root != _WORKLOAD_ROOT
            or context.private_runtime_parent != _PRIVATE_RUNTIME_PARENT
            or context.python_executable != _PYTHON_EXECUTABLE
            or context.python_argv0 != _PYTHON_ARGV0
            or context.process_runner is not _PRODUCTION_PROCESS_RUNNER
            or context.source_member_identity is not _PRODUCTION_SOURCE_MEMBER_IDENTITY
            or context.cleanup_execution_root is not _PRODUCTION_CLEANUP_EXECUTION_ROOT
        ):
            _fail("external execution production context lost its captured closure binding")
    elif (
        not state.test_only_injected
        or context.test_only_process_runner_injected is not True
        or context.closure_integrity_checked is not True
    ):
        _fail("external execution injected context attempted a production binding")
    _require_score_decoding_modules_absent()
    spec = _candidate(state.candidate_id)
    timeout = _require_ceiling(
        timeout_seconds, "timeout_seconds", maximum=_MAX_TIMEOUT_SECONDS, zero=False
    )
    stdout_ceiling = _require_ceiling(
        maximum_stdout_bytes,
        "maximum_stdout_bytes",
        maximum=_MAX_STDOUT_BYTES,
        zero=True,
    )
    stderr_ceiling = _require_ceiling(
        maximum_stderr_bytes,
        "maximum_stderr_bytes",
        maximum=_MAX_STDERR_BYTES,
        zero=True,
    )
    npz_ceiling = _require_ceiling(
        maximum_external_npz_bytes,
        "maximum_external_npz_bytes",
        maximum=_MAX_EXTERNAL_NPZ_BYTES,
        zero=False,
    )
    database_ceiling = _require_ceiling(
        maximum_results_database_bytes,
        "maximum_results_database_bytes",
        maximum=_MAX_RESULTS_DATABASE_BYTES,
        zero=False,
    )
    video_ceiling = _require_ceiling(
        maximum_ppo_video_bytes,
        "maximum_ppo_video_bytes",
        maximum=_MAX_PPO_VIDEO_BYTES,
        zero=spec.family != "ppo",
    )
    if spec.family == "continuing" and video_ceiling != 0:
        _fail("continuing candidates require maximum_ppo_video_bytes exactly zero")
    if spec.family == "ppo" and video_ceiling == 0:
        _fail("PPO candidates require a positive video ceiling")
    worker_source_sha256 = _require_current_module_source(state.source_sha256)

    source_root: _DirectoryAnchor | None = None
    private_parent: _DirectoryAnchor | None = None
    execution_root: _DirectoryAnchor | None = None
    save_root: _DirectoryAnchor | None = None
    checkpoint_root: _DirectoryAnchor | None = None
    runtime_root: _DirectoryAnchor | None = None
    executable: _ExecutableAnchor | None = None
    execution_name = ""
    sealed_payload: _SealedExternalExecutionPayload | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        source_root = _open_absolute_directory(context.workload_root, "staged workload root")
        executable = _open_executable(context.python_executable)
        entrypoint = _entrypoint(spec)
        source_preflight = (
            context.source_member_identity(
                source_root, entrypoint, _ENTRYPOINT_SHA256[entrypoint]
            ),
            context.source_member_identity(
                source_root, spec.configuration_path, spec.configuration_sha256
            ),
        )
        private_parent = _open_absolute_directory(
            context.private_runtime_parent, "private runtime parent"
        )
        if private_parent.owner != (os.getuid(), os.getgid()) or private_parent.mode != 0o700:
            _fail("private runtime parent ownership or mode differs")
        execution_name, execution_root = _create_execution_root(private_parent)
        save_root = _create_child_directory(execution_root.descriptor, "save")
        checkpoint_root = _create_child_directory(execution_root.descriptor, "checkpoints")
        runtime_root = _create_child_directory(execution_root.descriptor, "runtime")
        if len(
            {
                source_root.descriptor,
                executable.descriptor,
                execution_root.descriptor,
                save_root.descriptor,
                checkpoint_root.descriptor,
                runtime_root.descriptor,
            }
        ) != 6:
            _fail("external execution inherited descriptors contain an alias")
        save_proc = f"/proc/self/fd/{save_root.descriptor}"
        checkpoint_proc = f"/proc/self/fd/{checkpoint_root.descriptor}"
        runtime_proc = f"/proc/self/fd/{runtime_root.descriptor}"
        working_directory = f"/proc/self/fd/{source_root.descriptor}"
        argv = (
            context.python_argv0,
            "-B",
            entrypoint,
            "--exp",
            spec.configuration_path,
            "--idxs",
            "0",
            "--environment_seed",
            str(state.environment_seed),
            "--agent_seed",
            str(state.agent_seed),
            "--max_steps",
            str(_max_steps(spec)),
            "--save_path",
            save_proc,
            "--checkpoint_path",
            checkpoint_proc,
            "--silent",
        )
        environment = _child_environment(runtime_proc)
        process_result = _run_process(
            context.process_runner,
            argv,
            environment=environment,
            executable_descriptor=executable.descriptor,
            inherited_descriptors=(
                source_root.descriptor,
                execution_root.descriptor,
                save_root.descriptor,
                checkpoint_root.descriptor,
                runtime_root.descriptor,
            ),
            working_directory=working_directory,
            timeout_seconds=timeout,
            stdout_limit_bytes=stdout_ceiling,
            stderr_limit_bytes=stderr_ceiling,
        )
        source_root.verify()
        executable.verify()
        execution_root.verify()
        save_root.verify()
        checkpoint_root.verify()
        runtime_root.verify()
        source_postflight = (
            context.source_member_identity(
                source_root, entrypoint, _ENTRYPOINT_SHA256[entrypoint]
            ),
            context.source_member_identity(
                source_root, spec.configuration_path, spec.configuration_sha256
            ),
        )
        if source_preflight != source_postflight:
            _fail("selected external source members changed across execution")
        save_files, save_directories = _inventory_tree(save_root.descriptor)
        expected_paths = frozenset(path for _kind, path in _artifact_paths(spec))
        if save_files != expected_paths or save_directories != _expected_directories(spec):
            _fail("external execution save-root inventory differs")
        checkpoint_files, checkpoint_directories = _inventory_tree(
            checkpoint_root.descriptor
        )
        if checkpoint_files or checkpoint_directories:
            _fail("external execution checkpoint root is not empty")
        payloads: list[tuple[str, str, bytes]] = []
        for kind, path in _artifact_paths(spec):
            ceiling = {
                "upstream_reward_npz": npz_ceiling,
                "upstream_results_database": database_ceiling,
                "upstream_video": video_ceiling,
            }[kind]
            payloads.append(
                (
                    kind,
                    path,
                    _read_relative_regular(
                        save_root.descriptor,
                        path,
                        maximum_bytes=ceiling,
                    ),
                )
            )
        artifacts = tuple(payloads)
        payload = _receipt_payload(
            context=context,
            spec=spec,
            environment_seed=state.environment_seed,
            agent_seed=state.agent_seed,
            worker_source_sha256=worker_source_sha256,
            source_preflight=source_preflight,
            source_postflight=source_postflight,
            source_root=source_root,
            executable=executable,
            execution_name=execution_name,
            argv=argv,
            environment=environment,
            process_result=process_result,
            artifacts=artifacts,
            timeout_seconds=timeout,
            maximum_stdout_bytes=stdout_ceiling,
            maximum_stderr_bytes=stderr_ceiling,
            maximum_external_npz_bytes=npz_ceiling,
            maximum_results_database_bytes=database_ceiling,
            maximum_ppo_video_bytes=video_ceiling,
        )
        receipt_raw = _canonical_json(payload)
        receipt_sha256 = _sha256(receipt_raw)
        by_kind = {kind: raw for kind, _path, raw in artifacts}
        sealed_payload = _SealedExternalExecutionPayload(
            candidate_id=spec.candidate_id,
            environment_seed=state.environment_seed,
            agent_seed=state.agent_seed,
            execution_receipt_bytes=receipt_raw,
            execution_receipt_sha256=receipt_sha256,
            upstream_reward_npz=by_kind["upstream_reward_npz"],
            upstream_results_database=by_kind["upstream_results_database"],
            upstream_video=by_kind.get("upstream_video"),
            stdout=process_result.stdout,
            stderr=process_result.stderr,
            production_runner_exact=context.production_runner_exact,
        )
        parse_matched_v3_external_execution_receipt(
            sealed_payload.execution_receipt_bytes,
            expected_receipt_sha256=sealed_payload.execution_receipt_sha256,
            candidate_id=sealed_payload.candidate_id,
            environment_seed=sealed_payload.environment_seed,
            agent_seed=sealed_payload.agent_seed,
            upstream_reward_npz=sealed_payload.upstream_reward_npz,
            upstream_results_database=sealed_payload.upstream_results_database,
            upstream_video=sealed_payload.upstream_video,
            stdout=sealed_payload.stdout,
            stderr=sealed_payload.stderr,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        for anchor in (runtime_root, checkpoint_root, save_root):
            if anchor is not None:
                try:
                    anchor.close()
                except OSError as exc:
                    cleanup_errors.append(exc)
        if execution_root is not None and private_parent is not None and execution_name:
            try:
                context.cleanup_execution_root(
                    private_parent,
                    execution_name,
                    execution_root,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        if execution_root is not None and execution_root.descriptor >= 0:
            try:
                execution_root.close()
            except OSError as exc:
                cleanup_errors.append(exc)
        for anchor in (private_parent, source_root):
            if anchor is not None:
                try:
                    anchor.close()
                except OSError as exc:
                    cleanup_errors.append(exc)
        if executable is not None:
            try:
                executable.close()
            except OSError as exc:
                cleanup_errors.append(exc)
    if primary_error is not None:
        if cleanup_errors:
            for index, cleanup_error in enumerate(cleanup_errors, start=1):
                primary_error.add_note(
                    f"external execution cleanup failure {index}/{len(cleanup_errors)}: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            if isinstance(primary_error, ForagerMatchedV3ExternalExecutionRunnerError):
                primary_error.filesystem_state_uncertain = True
            else:
                combined = ForagerMatchedV3ExternalExecutionRunnerError(
                    "external execution failed and private filesystem cleanup is uncertain",
                    process_state_uncertain=bool(
                        getattr(primary_error, "process_state_uncertain", False)
                    ),
                    filesystem_state_uncertain=True,
                )
                combined.add_note(
                    f"primary failure: {type(primary_error).__name__}: {primary_error}"
                )
                for note in getattr(primary_error, "__notes__", ()):
                    combined.add_note(note)
                raise combined from primary_error
        raise primary_error
    if cleanup_errors:
        combined_cleanup = ForagerMatchedV3ExternalExecutionRunnerError(
            f"external execution cleanup failed in {len(cleanup_errors)} operation(s)",
            filesystem_state_uncertain=True,
        )
        for index, cleanup_error in enumerate(cleanup_errors, start=1):
            combined_cleanup.add_note(
                f"cleanup failure {index}/{len(cleanup_errors)}: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise combined_cleanup from cleanup_errors[0]
    if sealed_payload is None:
        _fail("sealed external execution payload disappeared before outcome issuance")
    current_source = _require_current_module_source(worker_source_sha256)
    outcome = _OutcomeCapability()
    content_sha256 = (
        sealed_payload.execution_receipt_sha256,
        _sha256(sealed_payload.upstream_reward_npz),
        _sha256(sealed_payload.upstream_results_database),
        _sha256(sealed_payload.upstream_video or b""),
        _sha256(sealed_payload.stdout),
        _sha256(sealed_payload.stderr),
    )
    with _CAPABILITY_LOCK:
        execution_state = _EXECUTION_CAPABILITIES.get(exact_capability)
        if (
            execution_state is None
            or execution_state.pid != os.getpid()
            or execution_state.status != "consumed"
        ):
            _fail("consumed external execution capability disappeared")
        _OUTCOME_CAPABILITIES[outcome] = _OutcomeState(
            pid=os.getpid(),
            status="live",
            execution_capability=exact_capability,
            execution_identity=id(exact_capability),
            source_sha256=current_source,
            content_sha256=content_sha256,
            sealed_payload=sealed_payload,
            production_runner_exact=context.production_runner_exact,
        )
    return outcome


def _claim_external_execution_outcome(outcome_capability: object) -> _OutcomeState:
    if type(outcome_capability) is not _OutcomeCapability:
        _fail("external execution outcome access requires one authentic capability")
    exact = outcome_capability
    with _CAPABILITY_LOCK:
        state = _OUTCOME_CAPABILITIES.get(exact)
        if state is None or state.status != "live":
            _fail("external execution outcome is unknown, stale, or already consumed")
        if state.pid != os.getpid():
            state.status = "consumed"
            _fail("external execution outcome cannot cross a PID boundary")
        execution = _EXECUTION_CAPABILITIES.get(state.execution_capability)
        if (
            execution is None
            or execution.pid != os.getpid()
            or execution.status != "consumed"
            or state.execution_identity != id(state.execution_capability)
        ):
            state.status = "consumed"
            _fail("external outcome lost its execution-capability binding")
        state.status = "consumed"
    return state


def _validated_sealed_external_payload(
    state: _OutcomeState,
) -> _SealedExternalExecutionPayload:
    _require_current_module_source(state.source_sha256)
    _require_score_decoding_modules_absent()
    sealed = state.sealed_payload
    if type(sealed) is not _SealedExternalExecutionPayload:
        _fail("external execution outcome sealed payload type differs")
    observed = (
        _sha256(sealed.execution_receipt_bytes),
        _sha256(sealed.upstream_reward_npz),
        _sha256(sealed.upstream_results_database),
        _sha256(sealed.upstream_video or b""),
        _sha256(sealed.stdout),
        _sha256(sealed.stderr),
    )
    if observed != state.content_sha256 or not hmac.compare_digest(
        sealed.execution_receipt_sha256, observed[0]
    ):
        _fail("external execution outcome content identity differs")
    parsed = parse_matched_v3_external_execution_receipt(
        sealed.execution_receipt_bytes,
        expected_receipt_sha256=sealed.execution_receipt_sha256,
        candidate_id=sealed.candidate_id,
        environment_seed=sealed.environment_seed,
        agent_seed=sealed.agent_seed,
        upstream_reward_npz=sealed.upstream_reward_npz,
        upstream_results_database=sealed.upstream_results_database,
        upstream_video=sealed.upstream_video,
        stdout=sealed.stdout,
        stderr=sealed.stderr,
    )
    runtime = cast(dict[str, Any], parsed["runtime"])
    if (
        runtime["production_runner_exact"] is not state.production_runner_exact
        or sealed.production_runner_exact is not state.production_runner_exact
    ):
        _fail("external outcome lost its production-runner binding")
    return sealed


def consume_matched_v3_external_execution_outcome(
    *,
    outcome_capability: object,
    explicit_content_access_opt_in: bool,
) -> MatchedV3ExternalExecutionCompletion:
    """Irrevocably choose public nonauthorizing byte completion for one outcome."""

    _require_production_closure_integrity()
    if (
        type(explicit_content_access_opt_in) is not bool
        or explicit_content_access_opt_in is not True
    ):
        _fail("external execution outcome access requires exact explicit opt-in")
    state = _claim_external_execution_outcome(outcome_capability)
    sealed = _validated_sealed_external_payload(state)
    return MatchedV3ExternalExecutionCompletion(
        candidate_id=sealed.candidate_id,
        environment_seed=sealed.environment_seed,
        agent_seed=sealed.agent_seed,
        execution_receipt_bytes=sealed.execution_receipt_bytes,
        execution_receipt_sha256=sealed.execution_receipt_sha256,
        upstream_reward_npz=sealed.upstream_reward_npz,
        upstream_results_database=sealed.upstream_results_database,
        upstream_video=sealed.upstream_video,
        stdout=sealed.stdout,
        stderr=sealed.stderr,
    )


def _consume_outcome_for_captured_external_consumer(
    *,
    outcome_capability: object,
    publication_parent: Path,
    expected_candidate_id: str,
    expected_environment_seed: int,
    expected_agent_seed: int,
    expected_environment_seed_commitment_sha256: str,
    expected_agent_seed_commitment_sha256: str,
    expected_qualification_plan_sha256: str,
    expected_qualification_case_manifest_sha256: str,
    expected_publisher_source_tree_sha256: str,
    expected_workload_source_tree_sha256: str,
    expected_staging_manifest_sha256: str,
    maximum_publication_total_bytes: int,
    explicit_publication_opt_in: bool,
) -> object:
    """Irrevocably choose the exact captured private publication path."""

    _require_production_closure_integrity()
    if explicit_publication_opt_in is not True:
        _fail("external outcome publication requires exact explicit opt-in")
    state = _claim_external_execution_outcome(outcome_capability)
    sealed = _validated_sealed_external_payload(state)
    sink = _require_captured_external_outcome_consumer()
    return sink(
        sealed_payload=sealed,
        publication_parent=publication_parent,
        expected_candidate_id=expected_candidate_id,
        expected_environment_seed=expected_environment_seed,
        expected_agent_seed=expected_agent_seed,
        expected_environment_seed_commitment_sha256=(
            expected_environment_seed_commitment_sha256
        ),
        expected_agent_seed_commitment_sha256=expected_agent_seed_commitment_sha256,
        expected_qualification_plan_sha256=expected_qualification_plan_sha256,
        expected_qualification_case_manifest_sha256=(
            expected_qualification_case_manifest_sha256
        ),
        expected_publisher_source_tree_sha256=expected_publisher_source_tree_sha256,
        expected_workload_source_tree_sha256=expected_workload_source_tree_sha256,
        expected_staging_manifest_sha256=expected_staging_manifest_sha256,
        maximum_publication_total_bytes=maximum_publication_total_bytes,
        explicit_publication_opt_in=True,
    )


_PRODUCTION_PROCESS_RUNNER = _default_process_runner
_PRODUCTION_SOURCE_MEMBER_IDENTITY = _source_member_identity
_PRODUCTION_CLEANUP_EXECUTION_ROOT = _cleanup_execution_root
_PRODUCTION_FUNCTION_SNAPSHOT = MappingProxyType(
    {
        "_fail": (_fail, _fail.__code__),
        "_default_value_fingerprint": (
            _default_value_fingerprint,
            _default_value_fingerprint.__code__,
        ),
        "_callable_defaults_fingerprint": (
            _callable_defaults_fingerprint,
            _callable_defaults_fingerprint.__code__,
        ),
        "_resolve_transitive_behavior_path": (
            _resolve_transitive_behavior_path,
            _resolve_transitive_behavior_path.__code__,
        ),
        "_transitive_behavior_state": (
            _transitive_behavior_state,
            _transitive_behavior_state.__code__,
        ),
        "BoundedExternalProcessResult.__post_init__": (
            BoundedExternalProcessResult.__post_init__,
            BoundedExternalProcessResult.__post_init__.__code__,
        ),
        "MatchedV3ExternalExecutionCompletion.__post_init__": (
            MatchedV3ExternalExecutionCompletion.__post_init__,
            MatchedV3ExternalExecutionCompletion.__post_init__.__code__,
        ),
        "_SealedExternalExecutionPayload.__post_init__": (
            _SealedExternalExecutionPayload.__post_init__,
            _SealedExternalExecutionPayload.__post_init__.__code__,
        ),
        "_DirectoryAnchor.verify": (
            _DirectoryAnchor.verify,
            _DirectoryAnchor.verify.__code__,
        ),
        "_DirectoryAnchor.close": (
            _DirectoryAnchor.close,
            _DirectoryAnchor.close.__code__,
        ),
        "_ExecutableAnchor.verify": (
            _ExecutableAnchor.verify,
            _ExecutableAnchor.verify.__code__,
        ),
        "_ExecutableAnchor.close": (
            _ExecutableAnchor.close,
            _ExecutableAnchor.close.__code__,
        ),
        "_sha256": (_sha256, _sha256.__code__),
        "_require_sha256": (_require_sha256, _require_sha256.__code__),
        "_require_uint31": (_require_uint31, _require_uint31.__code__),
        "_require_ceiling": (_require_ceiling, _require_ceiling.__code__),
        "_require_exact_object": (
            _require_exact_object,
            _require_exact_object.__code__,
        ),
        "_require_nonnegative_int": (
            _require_nonnegative_int,
            _require_nonnegative_int.__code__,
        ),
        "_require_process_returncode": (
            _require_process_returncode,
            _require_process_returncode.__code__,
        ),
        "_assert_plain_json": (_assert_plain_json, _assert_plain_json.__code__),
        "_canonical_json": (_canonical_json, _canonical_json.__code__),
        "_reject_constant": (_reject_constant, _reject_constant.__code__),
        "_reject_float": (_reject_float, _reject_float.__code__),
        "_parse_int": (_parse_int, _parse_int.__code__),
        "_without_duplicates": (_without_duplicates, _without_duplicates.__code__),
        "_strict_json": (_strict_json, _strict_json.__code__),
        "_exact_json_equal": (_exact_json_equal, _exact_json_equal.__code__),
        "_stat_identity": (_stat_identity, _stat_identity.__code__),
        "_directory_identity": (_directory_identity, _directory_identity.__code__),
        "_directory_flags": (_directory_flags, _directory_flags.__code__),
        "_file_flags": (_file_flags, _file_flags.__code__),
        "_validate_absolute_path": (
            _validate_absolute_path,
            _validate_absolute_path.__code__,
        ),
        "_open_absolute_directory": (
            _open_absolute_directory,
            _open_absolute_directory.__code__,
        ),
        "_hash_open_descriptor": (
            _hash_open_descriptor,
            _hash_open_descriptor.__code__,
        ),
        "_open_executable": (_open_executable, _open_executable.__code__),
        "_validate_relative_path": (
            _validate_relative_path,
            _validate_relative_path.__code__,
        ),
        "_open_relative_parent": (
            _open_relative_parent,
            _open_relative_parent.__code__,
        ),
        "_read_relative_regular": (
            _read_relative_regular,
            _read_relative_regular.__code__,
        ),
        "_source_member_identity": (
            _source_member_identity,
            _source_member_identity.__code__,
        ),
        "_module_source_sha256": (
            _module_source_sha256,
            _module_source_sha256.__code__,
        ),
        "_current_module_source_sha256": (
            _current_module_source_sha256,
            _current_module_source_sha256.__code__,
        ),
        "_require_current_module_source": (
            _require_current_module_source,
            _require_current_module_source.__code__,
        ),
        "_require_in_container_runtime": (
            _require_in_container_runtime,
            _require_in_container_runtime.__code__,
        ),
        "_require_score_decoding_modules_absent": (
            _require_score_decoding_modules_absent,
            _require_score_decoding_modules_absent.__code__,
        ),
        "_require_captured_external_outcome_consumer": (
            _require_captured_external_outcome_consumer,
            _require_captured_external_outcome_consumer.__code__,
        ),
        "_candidate": (_candidate, _candidate.__code__),
        "_entrypoint": (_entrypoint, _entrypoint.__code__),
        "_max_steps": (_max_steps, _max_steps.__code__),
        "_result_directory": (_result_directory, _result_directory.__code__),
        "_artifact_paths": (_artifact_paths, _artifact_paths.__code__),
        "_expected_directories": (
            _expected_directories,
            _expected_directories.__code__,
        ),
        "_constant_invariant_payload": (
            _constant_invariant_payload,
            _constant_invariant_payload.__code__,
        ),
        "_require_production_closure_integrity": (
            _require_production_closure_integrity,
            _require_production_closure_integrity.__code__,
        ),
        "_claims": (_claims, _claims.__code__),
        "issue_matched_v3_external_execution_capability": (
            issue_matched_v3_external_execution_capability,
            issue_matched_v3_external_execution_capability.__code__,
        ),
        "_issue_matched_v3_external_execution_capability_for_test": (
            _issue_matched_v3_external_execution_capability_for_test,
            _issue_matched_v3_external_execution_capability_for_test.__code__,
        ),
        "_issue_matched_v3_external_execution_capability": (
            _issue_matched_v3_external_execution_capability,
            _issue_matched_v3_external_execution_capability.__code__,
        ),
        "_consume_execution_capability": (
            _consume_execution_capability,
            _consume_execution_capability.__code__,
        ),
        "_child_environment": (_child_environment, _child_environment.__code__),
        "_create_child_directory": (
            _create_child_directory,
            _create_child_directory.__code__,
        ),
        "_create_execution_root": (
            _create_execution_root,
            _create_execution_root.__code__,
        ),
        "_inventory_tree": (_inventory_tree, _inventory_tree.__code__),
        "_remove_tree_contents": (
            _remove_tree_contents,
            _remove_tree_contents.__code__,
        ),
        "_cleanup_execution_root": (
            _cleanup_execution_root,
            _cleanup_execution_root.__code__,
        ),
        "_live_process_group_member_pids": (
            _live_process_group_member_pids,
            _live_process_group_member_pids.__code__,
        ),
        "_wait_leader_unreaped": (
            _wait_leader_unreaped,
            _wait_leader_unreaped.__code__,
        ),
        "_terminate_anchored_process_group": (
            _terminate_anchored_process_group,
            _terminate_anchored_process_group.__code__,
        ),
        "_default_process_runner": (
            _default_process_runner,
            _default_process_runner.__code__,
        ),
        "_run_process": (_run_process, _run_process.__code__),
        "_artifact_record": (_artifact_record, _artifact_record.__code__),
        "_receipt_body": (_receipt_body, _receipt_body.__code__),
        "_receipt_payload": (_receipt_payload, _receipt_payload.__code__),
        "_validate_artifact_inputs": (
            _validate_artifact_inputs,
            _validate_artifact_inputs.__code__,
        ),
        "parse_matched_v3_external_execution_receipt": (
            parse_matched_v3_external_execution_receipt,
            parse_matched_v3_external_execution_receipt.__code__,
        ),
        "execute_matched_v3_external_candidate": (
            execute_matched_v3_external_candidate,
            execute_matched_v3_external_candidate.__code__,
        ),
        "_execute_matched_v3_external_candidate_for_test": (
            _execute_matched_v3_external_candidate_for_test,
            _execute_matched_v3_external_candidate_for_test.__code__,
        ),
        "_execute_matched_v3_external_candidate": (
            _execute_matched_v3_external_candidate,
            _execute_matched_v3_external_candidate.__code__,
        ),
        "_claim_external_execution_outcome": (
            _claim_external_execution_outcome,
            _claim_external_execution_outcome.__code__,
        ),
        "_validated_sealed_external_payload": (
            _validated_sealed_external_payload,
            _validated_sealed_external_payload.__code__,
        ),
        "consume_matched_v3_external_execution_outcome": (
            consume_matched_v3_external_execution_outcome,
            consume_matched_v3_external_execution_outcome.__code__,
        ),
        "_consume_outcome_for_captured_external_consumer": (
            _consume_outcome_for_captured_external_consumer,
            _consume_outcome_for_captured_external_consumer.__code__,
        ),
    }
)
_PRODUCTION_CALLABLE_DEFAULT_SNAPSHOT = MappingProxyType(
    {
        label: (
            getattr(function, "__defaults__", None),
            getattr(function, "__kwdefaults__", None),
            _callable_defaults_fingerprint(function),
        )
        for label, (function, _code) in _PRODUCTION_FUNCTION_SNAPSHOT.items()
    }
)
_TRANSITIVE_BEHAVIOR_SNAPSHOT = MappingProxyType(
    {
        label: _transitive_behavior_state(_resolve_transitive_behavior_path(label))
        for label in _TRANSITIVE_BEHAVIOR_PATHS
    }
)
_INVARIANT_OBJECT_SNAPSHOT = MappingProxyType(
    {
        "_PRODUCTION_FUNCTION_SNAPSHOT": _PRODUCTION_FUNCTION_SNAPSHOT,
        "_PRODUCTION_CALLABLE_DEFAULT_SNAPSHOT": _PRODUCTION_CALLABLE_DEFAULT_SNAPSHOT,
        "_TRANSITIVE_BEHAVIOR_SNAPSHOT": _TRANSITIVE_BEHAVIOR_SNAPSHOT,
        "_TRANSITIVE_BEHAVIOR_PATHS": _TRANSITIVE_BEHAVIOR_PATHS,
        "_CandidateSpec": _CandidateSpec,
        "_CANDIDATE_SPECS": _CANDIDATE_SPECS,
        "EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS": EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS,
        "_CANDIDATE_BY_ID": _CANDIDATE_BY_ID,
        "_ENTRYPOINT_SHA256": _ENTRYPOINT_SHA256,
        "_SHA256_RE": _SHA256_RE,
        "_PRIVATE_NAME_RE": _PRIVATE_NAME_RE,
        "_WORKLOAD_ROOT": _WORKLOAD_ROOT,
        "_PRIVATE_RUNTIME_PARENT": _PRIVATE_RUNTIME_PARENT,
        "_PYTHON_EXECUTABLE": _PYTHON_EXECUTABLE,
        "_ExecutionCapability": _ExecutionCapability,
        "_OutcomeCapability": _OutcomeCapability,
        "_SealedExternalExecutionPayload": _SealedExternalExecutionPayload,
        "_EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD": (
            _EXTERNAL_OUTCOME_CONSUMER_MODULE_AT_LOAD
        ),
        "_EXTERNAL_OUTCOME_CONSUMER_SINK_AT_LOAD": (
            _EXTERNAL_OUTCOME_CONSUMER_SINK_AT_LOAD
        ),
        "_EXTERNAL_OUTCOME_CONSUMER_GUARD_AT_LOAD": (
            _EXTERNAL_OUTCOME_CONSUMER_GUARD_AT_LOAD
        ),
        "_EXTERNAL_OUTCOME_CONSUMER_REPLAY_AT_LOAD": (
            _EXTERNAL_OUTCOME_CONSUMER_REPLAY_AT_LOAD
        ),
        "_INJECTED_TEST_ONLY_MARKER": _INJECTED_TEST_ONLY_MARKER,
        "_CAPABILITY_LOCK": _CAPABILITY_LOCK,
        "_EXECUTION_CAPABILITIES": _EXECUTION_CAPABILITIES,
        "_OUTCOME_CAPABILITIES": _OUTCOME_CAPABILITIES,
        "_DESCRIPTOR_BYTES": _DESCRIPTOR_BYTES,
        "_PRODUCTION_PROCESS_RUNNER": _PRODUCTION_PROCESS_RUNNER,
        "_PRODUCTION_SOURCE_MEMBER_IDENTITY": _PRODUCTION_SOURCE_MEMBER_IDENTITY,
        "_PRODUCTION_CLEANUP_EXECUTION_ROOT": _PRODUCTION_CLEANUP_EXECUTION_ROOT,
    }
)
_INVARIANT_VALUE_SNAPSHOT = _constant_invariant_payload()


__all__ = [
    "BoundedExternalProcessResult",
    "EXPLICIT_EXTERNAL_EXECUTION_OPT_IN",
    "EXTERNAL_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_EXECUTION_RUNNER_CANDIDATE_IDS",
    "EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_EXECUTION_RUNNER_DESCRIPTOR_SHA256",
    "EXTERNAL_EXECUTION_RUNNER_STATUS",
    "ExternalProcessRunner",
    "ForagerMatchedV3ExternalExecutionRunnerError",
    "MatchedV3ExternalExecutionCompletion",
    "canonical_external_execution_runner_descriptor_bytes",
    "consume_matched_v3_external_execution_outcome",
    "execute_matched_v3_external_candidate",
    "external_execution_runner_descriptor",
    "external_execution_runner_descriptor_sha256",
    "issue_matched_v3_external_execution_capability",
    "parse_external_execution_runner_descriptor",
    "parse_matched_v3_external_execution_receipt",
]
