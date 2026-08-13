"""Capability-gated, bounded-memory execution for the 14 local Forager v3 arms.

An exact isolated top-level direct-file load of this module, with no JAX, NumPy,
Foragax, or Alberta modules already loaded, constructs content descriptors and an
empty weak capability registry only.  The forbidden prefixes are checked again at
capability issuance and, after capability consumption, immediately before the
runner-owned dependency-import transition.  Execution additionally requires a
bootstrap to read the exact source bytes, inject their SHA-256, and compile/execute
those same bytes.  A plain loader without that binding and a normal package import
are both descriptor/parser-only and cannot issue or spend execution capability.

The scoped module load itself does not import JAX, NumPy, a Forager runner, or the
historical worker; inspect source files; create a runtime; run a subprocess; or
execute a workload.  Normal package initializers may already have imported heavy
runtime code, which is why that route is never execution-eligible.

After an exact explicit opt-in and a PID-bound opaque single-use capability are
consumed, the loader checks the seven declared relevant source pins before importing
the runtime modules.  Those pins are intentionally only a relevant subset, not a
complete local or distribution source closure; complete closure remains external and
unqualified.

A successful execution returns only a weak process-local outcome capability.
Consuming that capability after a second exact opt-in exposes a non-authorizing
completion receipt plus exactly one signed-int8 byte per environment interaction.
No transition, observation, action, optimizer state, or full runner result is retained.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import os
import re
import sys
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, NoReturn, cast

LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_runner_descriptor.v1"
)
LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_runner_completion.v1"
)
LOCAL_RUNNER_STATUS: Final = "implemented_unexecuted"
LOCAL_RUNNER_COMPLETION_STATUS: Final = "completed_unqualified_non_authorizing"

MATCHED_V3_LOCAL_RUNNER_ISOLATED_MODULE_NAME: Final = (
    "_alberta_forager_matched_v3_local_runner_isolated_v1"
)
_BOOTSTRAP_SOURCE_SHA256_INPUT: Final = globals().get(
    "_MATCHED_V3_LOCAL_RUNNER_BOOTSTRAP_SOURCE_SHA256"
)
_MODULE_NAME_INPUT: Final = globals().get("__name__")
_MODULE_PACKAGE_INPUT: Final = globals().get("__package__")
_EXECUTION_PRELOAD_PREFIXES: Final = (
    "alberta_framework",
    "chex",
    "foragax",
    "jax",
    "jaxlib",
    "ml_dtypes",
    "numpy",
    "scipy",
)
_NONEXACT_MODULE_KEYS_AT_LOAD: Final = tuple(
    type(module_name).__name__ for module_name in sys.modules if type(module_name) is not str
)
_PRELOADED_EXECUTION_MODULES: Final = tuple(
    sorted(
        module_name
        for module_name in sys.modules
        if type(module_name) is str
        and any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in _EXECUTION_PRELOAD_PREFIXES
        )
    )
)
_ISOLATED_TOP_LEVEL_NAME_BOUNDARY: Final = (
    type(_MODULE_NAME_INPUT) is str
    and _MODULE_NAME_INPUT == MATCHED_V3_LOCAL_RUNNER_ISOLATED_MODULE_NAME
    and (
        _MODULE_PACKAGE_INPUT is None
        or (type(_MODULE_PACKAGE_INPUT) is str and _MODULE_PACKAGE_INPUT == "")
    )
    and not _NONEXACT_MODULE_KEYS_AT_LOAD
    and not _PRELOADED_EXECUTION_MODULES
)

MATCHED_V3_LOCAL_RUNNER_HORIZON: Final = 499_712
MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID: Final = "ForagaxTwoBiomeLarge-v1"
MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE: Final = "color"
MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE: Final = 9

_MAX_CANONICAL_BYTES: Final = 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 100_000
_MAX_SOURCE_BYTES: Final = 8 * 1024 * 1024
_UINT31_MAX: Final = 2**31 - 1
_REWARD_BYTES: Final = frozenset({0, 1, 30, 255})
_REWARD_VALUES: Final = (-1, 0, 1, 30)
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_BOOTSTRAP_SOURCE_SHA256: Final = (
    _BOOTSTRAP_SOURCE_SHA256_INPUT
    if type(_BOOTSTRAP_SOURCE_SHA256_INPUT) is str
    and _SHA256_RE.fullmatch(_BOOTSTRAP_SOURCE_SHA256_INPUT) is not None
    else None
)
_ISOLATED_TOP_LEVEL_LOAD: Final = (
    _ISOLATED_TOP_LEVEL_NAME_BOUNDARY and _BOOTSTRAP_SOURCE_SHA256 is not None
)

_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
_CUMULATIVE_REWARD_METRIC_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256: Final = (
    "d15d70b55d965b2c135f1dcaa36a74173e4023e4fdc9430c43660df54f1bb38c"
)
_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256: Final = (
    "1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b"
)

MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS: Final = (
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

_IMPLEMENTATION_KIND_BY_CANDIDATE: Final = {
    **{
        candidate_id: "alberta_causal_map"
        for candidate_id in MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS[:9]
    },
    **{
        candidate_id: "alberta_horde_actor_critic"
        for candidate_id in MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS[9:13]
    },
    "alberta_rtu_h08_taylor": "alberta_rtu_rtrl",
}

_WORKER_ENVELOPE_SHA256_BY_CANDIDATE: Final = {
    "causal_e025_q050": "1290335563481b7ac2fd3eda91ef9c63216684fd096f3ab5b16591de0870c736",
    "causal_e025_q075": "69a5df44db99866a0ee3967677fad66ea94c60b1bfa8317936e2c142fac34ed1",
    "causal_e025_q090": "e21692571fc751bdf2c4fa0e89ad43b12dbd51c72a0821d5839fc82f1031f8f4",
    "causal_e050_q050": "916bd37e04c39dc16c19153032fc1c3baf12a941efb3df95860ee9f03c1ef331",
    "causal_e050_q075": "afaa3ea47cd410a43541c85976fa6f718c5f70504494f70496385ec37ea84a63",
    "causal_e050_q090": "ab555510e08a98e733d01a9b145d19073bb17ba31681a459a55a978d5a4faf33",
    "causal_e100_q050": "00390162a1950e976a7b3e216b8c6d94a76427c38c8e30bbdc25fa583bf018a8",
    "causal_e100_q075": "8d7a8afdb204c1837834ef633e2524bf569180c763a34a96c883c6e2cd33fb48",
    "causal_e100_q090": "899658dff1eeaadf59de8dc437d1324429306b8a427a4ed67ccf54437931955c",
    "alberta_horde_default": "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a",
    "alberta_horde_eps05": "ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f",
    "alberta_horde_recurrent64": (
        "870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef"
    ),
    "alberta_horde_step3e3": "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5",
    "alberta_rtu_h08_taylor": ("07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"),
}

_CONFIGURATION_RECORD_SHA256_BY_CANDIDATE: Final = {
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
    "alberta_rtu_h08_taylor": ("d804c8b79f29da16f085c7f1b4621ae479d780c3b23d30799367982353eb69df"),
}

_PINNED_SOURCE_SHA256: Final = {
    "local_configuration": ("6ccf0fb75d2fa30c8c66788468cbd366a1ef3da36b61dbc2e6dc99d38aeda7b7"),
    "forager": "e76305bd8cf933e80c3c67ffb0bfedc7f98be5874ef8be5f468954832d90ae43",
    "causal_map_forager": ("0d05e3265c69f3d1cbfbbc47f7994010316f2925065f705ab523cb3b6897aa1b"),
    "matched_alberta_worker": ("ce81f7f474b5abf6619f13449f7defa200fd6601cb1910d061ea27b1a9a0c8e7"),
    "matched_open_protocol": ("8b01d287c6dad6db2938fde9a7ce703cb07c6b863e73e52e90ce6c82010e2cfb"),
    "recurrent_trace_actor_critic": (
        "35d476a945fa84d1aafdb0f1d2ab0b9c42f6ca0e3f47ebc82ce7f020d0bca287"
    ),
    "strict_reward_scorer": ("eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"),
}

_SOURCE_SUFFIX_BY_ID: Final = {
    "local_configuration": "/forager_matched_v3_local_configuration.py",
    "forager": "/forager.py",
    "causal_map_forager": "/causal_map_forager.py",
    "matched_alberta_worker": "/_forager_matched_alberta_worker.py",
    "matched_open_protocol": "/forager_matched_open_protocol.py",
    "recurrent_trace_actor_critic": "/recurrent_trace_actor_critic.py",
    "strict_reward_scorer": "/_forager_matched_v3_scorer.py",
}


class ForagerMatchedV3LocalRunnerError(RuntimeError):
    """A local runner capability, dependency, reward trace, or receipt failed closed."""


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalRunnerError(
        f"local-runner JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ForagerMatchedV3LocalRunnerError(
            "local-runner JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalRunnerError(
                f"local-runner JSON contains duplicate key {key!r}"
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
            raise ForagerMatchedV3LocalRunnerError("local-runner JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalRunnerError("local-runner JSON exceeds its depth bound")
        if type(item) is str:
            if len(item) > 1024 or any(ord(char) < 0x20 or ord(char) > 0x7E for char in item):
                raise ForagerMatchedV3LocalRunnerError(
                    "local-runner JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalRunnerError("local-runner JSON contains a non-plain value")
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalRunnerError("local-runner JSON contains a container alias")
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalRunnerError(
                        "local-runner JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalRunnerError("local-runner canonical root must be a plain object")
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
        raise ForagerMatchedV3LocalRunnerError(
            "local-runner value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_CANONICAL_BYTES:
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact exceeds its byte bound")
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_CANONICAL_BYTES:
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact must be bounded exact bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalRunnerError(
            "local-runner artifact must have one trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalRunnerError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact is not strict JSON") from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact root must be a plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3LocalRunnerError("local-runner artifact is not exactly canonical")
    return result


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3LocalRunnerError(f"{label} keys are not exact")
    return cast(dict[str, Any], value)


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        exact_right = cast(list[Any], right)
        return len(left) == len(exact_right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, exact_right, strict=True)
        )
    if type(left) is dict:
        exact_left = cast(dict[str, Any], left)
        exact_right_map = cast(dict[str, Any], right)
        return exact_left.keys() == exact_right_map.keys() and all(
            _exact_json_equal(exact_left[key], exact_right_map[key]) for key in exact_left
        )
    return False


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3LocalRunnerError(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_uint31(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT31_MAX:
        raise ForagerMatchedV3LocalRunnerError(f"{label} must be one exact uint31 integer")
    return value


def _claims() -> dict[str, bool]:
    return {
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "execution_authority_granted": False,
        "serialized_content_grants_capability": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "A completion receipt records one unqualified local execution only.",
        "The in-memory trace retains rewards but no observations, actions, or transitions.",
        (
            "Only seven declared relevant source pins are checked; complete source "
            "closure is external."
        ),
        "Execution requires an exact-byte bootstrap source binding.",
        "Normal package import is descriptor/parser-only and grants no execution capability.",
        "A serialized descriptor or completion receipt grants no execution capability.",
        "No completion is scientific evidence or authorizes promotion or a SOTA claim.",
    ]


def _candidate_bindings() -> list[dict[str, str]]:
    return [
        {
            "candidate_id": candidate_id,
            "implementation_kind": _IMPLEMENTATION_KIND_BY_CANDIDATE[candidate_id],
            "configuration_record_sha256": (
                _CONFIGURATION_RECORD_SHA256_BY_CANDIDATE[candidate_id]
            ),
            "worker_envelope_sha256": (_WORKER_ENVELOPE_SHA256_BY_CANDIDATE[candidate_id]),
        }
        for candidate_id in MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_RUNNER_STATUS,
        "classification": "local_execution_implementation_non_authorizing",
        "configuration_plan": {
            "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
        "metric": {
            "schema_version": "alberta.forager_cumulative_reward_metric.v1",
            "sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
            "input": "exact_ordered_signed_int8_reward_trace",
            "strict_scorer_source_sha256": _PINNED_SOURCE_SHA256["strict_reward_scorer"],
        },
        "local_configuration": {
            "source_descriptor_sha256": (_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256),
            "builder_descriptor_sha256": (_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256),
        },
        "relevant_source_sha256": dict(_PINNED_SOURCE_SHA256),
        "source_closure": {
            "checked_relevant_pin_count": 7,
            "checked_scope": "declared_relevant_source_subset_only",
            "complete_local_source_closure": False,
            "complete_distribution_source_closure": False,
            "complete_closure_required_externally": True,
            "qualified": False,
        },
        "candidate_bindings": _candidate_bindings(),
        "task": {
            "preset": "field_of_view",
            "environment_id": MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID,
            "observation_type": MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE,
            "aperture_size": MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE,
            "horizon": MATCHED_V3_LOCAL_RUNNER_HORIZON,
            "reward_values": list(_REWARD_VALUES),
        },
        "seed_transport": {
            "environment_seed": "explicit_uint31",
            "agent_seed": "explicit_uint31_lane_paired",
            "environment_agent_seed_collisions_allowed": True,
            "collision_rejection": False,
        },
        "reward_sink": {
            "storage": "bounded_in_memory_bytearray",
            "encoding": "signed_int8_twos_complement_v1",
            "maximum_payload_bytes": MATCHED_V3_LOCAL_RUNNER_HORIZON,
            "full_transition_retention": False,
            "filesystem_artifact_written": False,
        },
        "execution_capability": {
            "isolated_top_level_direct_file_load_required": True,
            "isolated_top_level_module_name": (MATCHED_V3_LOCAL_RUNNER_ISOLATED_MODULE_NAME),
            "bootstrap_injected_source_sha256_required": True,
            "bootstrap_loader_contract": "read_hash_compile_exec_exact_bytes_v1",
            "plain_spec_loader_grants_capability": False,
            "normal_package_import_grants_capability": False,
            "preloaded_module_prefixes_rejected": list(_EXECUTION_PRELOAD_PREFIXES),
            "forbidden_prefixes_rechecked_at_capability_issue": True,
            "forbidden_prefixes_rechecked_at_pre_import_run_boundary": True,
            "pre_import_boundary_failure_consumes_capability": True,
            "runner_owned_import_transition_required": True,
            "post_import_exact_module_identity_validation_required": True,
            "explicit_opt_in_required_at_issue": True,
            "explicit_opt_in_required_at_run": True,
            "opaque": True,
            "weakly_registered": True,
            "pid_bound": True,
            "single_use": True,
            "consumed_before_lazy_import_or_workload": True,
            "serializable": False,
            "serialized_content_grants_capability": False,
        },
        "outcome_capability": {
            "opaque": True,
            "weakly_registered": True,
            "pid_bound": True,
            "bound_to_consumed_execution_capability_identity": True,
            "bound_to_trace_and_receipt_sha256": True,
            "bound_to_observed_runner_source_sha256": True,
            "single_use_content_access": True,
            "serializable": False,
            "structural_content_is_non_authorizing": True,
        },
        "import_contract": {
            "scope": "exact_isolated_top_level_direct_file_load_only",
            "bootstrap_source_identity_required": True,
            "plain_spec_loader_execution_eligible": False,
            "normal_package_import_execution_eligible": False,
            "normal_package_import_may_preload_parent_dependencies": True,
            "preloaded_at_scoped_load_allowed": False,
            "preloaded_at_capability_issue_allowed": False,
            "preloaded_at_pre_import_run_boundary_allowed": False,
            "post_import_dependencies_allowed_only_after_runner_owned_transition": True,
            "jax_imported_by_scoped_load": False,
            "numpy_imported_by_scoped_load": False,
            "forager_runner_imported_by_scoped_load": False,
            "filesystem_inspection_by_scoped_load": False,
            "subprocess_by_scoped_load": False,
            "workload_by_scoped_load": False,
        },
        "completion_schema_version": LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION,
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
LOCAL_RUNNER_DESCRIPTOR_SHA256: Final = (
    "2237914749f353d2700bbb0f33a66d8789268a5e156f2961be2e626f42efd2a1"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(), LOCAL_RUNNER_DESCRIPTOR_SHA256
):
    raise AssertionError("matched-v3 local-runner descriptor identity drifted")


class _LocalExecutionCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local execution capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("local execution capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("local execution capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("local execution capabilities cannot be serialized")


@dataclass(slots=True)
class _CapabilityState:
    pid: int
    status: Literal["issued", "consumed"]


_CAPABILITY_LOCK: Final = threading.Lock()
_CAPABILITIES: Final[weakref.WeakKeyDictionary[_LocalExecutionCapability, _CapabilityState]] = (
    weakref.WeakKeyDictionary()
)


class _LocalOutcomeCapability:
    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<matched-v3 local outcome capability>"

    def __copy__(self) -> NoReturn:
        raise TypeError("local outcome capabilities cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("local outcome capabilities cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("local outcome capabilities cannot be serialized")


@dataclass(slots=True)
class _OutcomeState:
    pid: int
    status: Literal["live", "consumed"]
    execution_capability: _LocalExecutionCapability
    execution_capability_identity: int
    trace_sha256: str
    receipt_sha256: str
    runner_source_sha256: str
    completion: MatchedV3LocalRunCompletion


_OUTCOMES: Final[weakref.WeakKeyDictionary[_LocalOutcomeCapability, _OutcomeState]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(slots=True)
class _RuntimeImportTransition:
    pid: int
    status: Literal["loading", "loaded", "validated", "poisoned"]
    module_bindings: tuple[tuple[str, object], ...]


_RUNTIME_IMPORT_LOCK: Final = threading.Lock()
_RUNTIME_IMPORT_TRANSITION: _RuntimeImportTransition | None = None


def _current_execution_module_bindings() -> tuple[tuple[str, object], ...]:
    try:
        module_items = tuple(sys.modules.items())
    except RuntimeError as exc:
        raise ForagerMatchedV3LocalRunnerError(
            "runtime module registry changed during forbidden-prefix observation"
        ) from exc
    nonexact_keys = tuple(
        type(module_name).__name__
        for module_name, _module in module_items
        if type(module_name) is not str
    )
    if nonexact_keys:
        raise ForagerMatchedV3LocalRunnerError(
            "runtime module registry contains a non-exact-string key"
        )
    return tuple(
        sorted(
            (
                (module_name, module)
                for module_name, module in module_items
                if any(
                    module_name == prefix or module_name.startswith(f"{prefix}.")
                    for prefix in _EXECUTION_PRELOAD_PREFIXES
                )
            ),
            key=lambda item: item[0],
        )
    )


def _require_clean_runtime_import_boundary(stage: str) -> None:
    with _RUNTIME_IMPORT_LOCK:
        transition = _RUNTIME_IMPORT_TRANSITION
        bindings = _current_execution_module_bindings()
        if transition is not None or bindings:
            names = ", ".join(name for name, _module in bindings[:8])
            detail = names if names else cast(_RuntimeImportTransition, transition).status
            raise ForagerMatchedV3LocalRunnerError(
                f"{stage} rejects preloaded runtime dependencies or a prior runtime "
                f"import transition: {detail}"
            )


def _begin_runtime_import_transition() -> _RuntimeImportTransition:
    global _RUNTIME_IMPORT_TRANSITION

    with _RUNTIME_IMPORT_LOCK:
        if _RUNTIME_IMPORT_TRANSITION is not None:
            raise ForagerMatchedV3LocalRunnerError(
                "runtime dependency import transition is already active or spent"
            )
        bindings = _current_execution_module_bindings()
        if bindings:
            names = ", ".join(name for name, _module in bindings[:8])
            raise ForagerMatchedV3LocalRunnerError(
                f"runtime dependency loader rejects preloaded runtime dependencies: {names}"
            )
        transition = _RuntimeImportTransition(
            pid=os.getpid(),
            status="loading",
            module_bindings=(),
        )
        _RUNTIME_IMPORT_TRANSITION = transition
        return transition


def _finish_runtime_import_transition(
    transition: _RuntimeImportTransition,
    required_modules: Mapping[str, object],
) -> None:
    with _RUNTIME_IMPORT_LOCK:
        if (
            _RUNTIME_IMPORT_TRANSITION is not transition
            or transition.pid != os.getpid()
            or transition.status != "loading"
        ):
            transition.status = "poisoned"
            raise ForagerMatchedV3LocalRunnerError(
                "runtime dependency import transition identity drifted"
            )
        bindings = _current_execution_module_bindings()
        binding_by_name = dict(bindings)
        if not bindings or any(
            binding_by_name.get(module_name) is not module
            for module_name, module in required_modules.items()
        ):
            transition.status = "poisoned"
            raise ForagerMatchedV3LocalRunnerError(
                "runtime dependency import did not produce the exact required modules"
            )
        transition.module_bindings = bindings
        transition.status = "loaded"


def _poison_runtime_import_transition(transition: _RuntimeImportTransition) -> None:
    with _RUNTIME_IMPORT_LOCK:
        transition.status = "poisoned"


def _validate_runtime_import_transition() -> None:
    with _RUNTIME_IMPORT_LOCK:
        transition = _RUNTIME_IMPORT_TRANSITION
        if transition is None or transition.pid != os.getpid() or transition.status != "loaded":
            if transition is not None:
                transition.status = "poisoned"
            raise ForagerMatchedV3LocalRunnerError(
                "post-import validation requires the exact runner-owned import transition"
            )
        try:
            current = _current_execution_module_bindings()
        except ForagerMatchedV3LocalRunnerError:
            transition.status = "poisoned"
            raise
        if len(current) != len(transition.module_bindings) or any(
            current_name != expected_name or current_module is not expected_module
            for (current_name, current_module), (expected_name, expected_module) in zip(
                current, transition.module_bindings, strict=True
            )
        ):
            transition.status = "poisoned"
            raise ForagerMatchedV3LocalRunnerError(
                "runtime dependency module identities drifted before post-import validation"
            )
        transition.status = "validated"


def _require_bootstrap_runner_source_current(expected_sha256: str) -> None:
    exact_expected = _require_sha256(expected_sha256, "bootstrap local runner source")
    current = _require_sha256(_current_runner_source_sha256(), "current local runner source")
    if not hmac.compare_digest(current, exact_expected):
        raise ForagerMatchedV3LocalRunnerError(
            "bootstrap local runner source identity is stale or forged"
        )


def _require_isolated_top_level_execution_boundary(*, require_current_source: bool) -> str:
    if not _ISOLATED_TOP_LEVEL_NAME_BOUNDARY:
        raise ForagerMatchedV3LocalRunnerError(
            "local execution capability requires the exact isolated top-level "
            "direct-file module load without preloaded runtime dependencies; "
            "normal package import is descriptor/parser-only"
        )
    if _BOOTSTRAP_SOURCE_SHA256 is None:
        raise ForagerMatchedV3LocalRunnerError(
            "local execution capability requires a bootstrap-injected SHA-256 "
            "of the exact source bytes compiled and executed"
        )
    expected = _require_sha256(_BOOTSTRAP_SOURCE_SHA256, "bootstrap local runner source")
    if require_current_source:
        _require_bootstrap_runner_source_current(expected)
    return expected


def issue_matched_v3_local_execution_capability(*, explicit_execution_opt_in: bool) -> object:
    """Issue one opaque weak capability after an exact explicit opt-in."""

    if type(explicit_execution_opt_in) is not bool or explicit_execution_opt_in is not True:
        raise ForagerMatchedV3LocalRunnerError(
            "issuing a local execution capability requires exact explicit opt-in"
        )
    _require_isolated_top_level_execution_boundary(require_current_source=True)
    _require_clean_runtime_import_boundary("capability issuance")
    capability = _LocalExecutionCapability()
    state = _CapabilityState(pid=os.getpid(), status="issued")
    with _CAPABILITY_LOCK:
        _CAPABILITIES[capability] = state
    return capability


def _consume_capability(capability: object) -> None:
    if type(capability) is not _LocalExecutionCapability:
        raise ForagerMatchedV3LocalRunnerError(
            "local execution requires an authentic opaque capability"
        )
    with _CAPABILITY_LOCK:
        state = _CAPABILITIES.get(capability)
        if state is None or state.status != "issued":
            raise ForagerMatchedV3LocalRunnerError(
                "local execution capability is unknown or already consumed"
            )
        if state.pid != os.getpid():
            state.status = "consumed"
            raise ForagerMatchedV3LocalRunnerError(
                "local execution capability cannot cross a PID boundary"
            )
        state.status = "consumed"


@dataclass(frozen=True, slots=True)
class _RuntimeDependencies:
    numpy: Any
    local_configuration: Any
    worker: Any
    forager: Any
    causal_map_forager: Any
    source_sha256_by_id: Mapping[str, str]


def _bounded_source_sha256(path: Any) -> str:
    import stat
    from pathlib import Path

    if type(path) is not type(Path()):
        raise ForagerMatchedV3LocalRunnerError("source path must be one exact Path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedV3LocalRunnerError("cannot open an exact source dependency") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > _MAX_SOURCE_BYTES
        ):
            raise ForagerMatchedV3LocalRunnerError(
                "source dependency is not one bounded single-link regular file"
            )
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForagerMatchedV3LocalRunnerError("source dependency ended while being read")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3LocalRunnerError("source dependency grew while being read")
        after = os.fstat(descriptor)

        def identity(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise ForagerMatchedV3LocalRunnerError("source dependency changed while being read")
        return digest.hexdigest()
    except OSError as exc:
        raise ForagerMatchedV3LocalRunnerError(
            "source dependency could not be read exactly"
        ) from exc
    finally:
        os.close(descriptor)


def _runtime_source_paths() -> dict[str, Any]:
    from pathlib import Path

    runner_path = Path(__file__).absolute()
    benchmark_root = runner_path.parent
    framework_root = benchmark_root.parent
    return {
        "local_configuration": (benchmark_root / "forager_matched_v3_local_configuration.py"),
        "forager": benchmark_root / "forager.py",
        "causal_map_forager": benchmark_root / "causal_map_forager.py",
        "matched_alberta_worker": benchmark_root / "_forager_matched_alberta_worker.py",
        "matched_open_protocol": benchmark_root / "forager_matched_open_protocol.py",
        "recurrent_trace_actor_critic": (
            framework_root / "core" / "recurrent_trace_actor_critic.py"
        ),
        "strict_reward_scorer": benchmark_root / "_forager_matched_v3_scorer.py",
        "local_runner_observed": runner_path,
    }


def _verify_imported_module_path(module: Any, expected_path: Any, source_id: str) -> None:
    from pathlib import Path

    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise ForagerMatchedV3LocalRunnerError(
            f"imported source dependency has no exact path: {source_id}"
        )
    try:
        actual = Path(module_file).resolve(strict=True)
        expected = expected_path.resolve(strict=True)
    except OSError as exc:
        raise ForagerMatchedV3LocalRunnerError(
            f"imported source dependency path cannot be resolved: {source_id}"
        ) from exc
    if actual != expected or not module_file.endswith(_SOURCE_SUFFIX_BY_ID[source_id]):
        raise ForagerMatchedV3LocalRunnerError(
            f"imported source dependency path drifted: {source_id}"
        )


def _load_runtime_dependencies() -> _RuntimeDependencies:
    bootstrap_source_sha256 = _require_isolated_top_level_execution_boundary(
        require_current_source=True
    )
    transition = _begin_runtime_import_transition()
    try:
        source_paths = _runtime_source_paths()
        source_sha256_by_id = {
            source_id: _bounded_source_sha256(source_paths[source_id])
            for source_id in _PINNED_SOURCE_SHA256
        }
        for source_id, expected in _PINNED_SOURCE_SHA256.items():
            if not hmac.compare_digest(source_sha256_by_id[source_id], expected):
                raise ForagerMatchedV3LocalRunnerError(
                    f"local runner source dependency drifted before import: {source_id}"
                )
        module_names = {
            "local_configuration": (
                "alberta_framework.benchmarks.forager_matched_v3_local_configuration"
            ),
            "forager": "alberta_framework.benchmarks.forager",
            "causal_map_forager": "alberta_framework.benchmarks.causal_map_forager",
            "matched_alberta_worker": (
                "alberta_framework.benchmarks._forager_matched_alberta_worker"
            ),
            "matched_open_protocol": ("alberta_framework.benchmarks.forager_matched_open_protocol"),
            "recurrent_trace_actor_critic": ("alberta_framework.core.recurrent_trace_actor_critic"),
        }
        modules = {
            source_id: importlib.import_module(module_name)
            for source_id, module_name in module_names.items()
        }
        for source_id, expected in _PINNED_SOURCE_SHA256.items():
            if source_id in modules:
                _verify_imported_module_path(modules[source_id], source_paths[source_id], source_id)
            after_import = _bounded_source_sha256(source_paths[source_id])
            if not hmac.compare_digest(after_import, expected):
                raise ForagerMatchedV3LocalRunnerError(
                    f"local runner source dependency drifted during import: {source_id}"
                )
        post_import_runner_sha256 = _bounded_source_sha256(source_paths["local_runner_observed"])
        if not hmac.compare_digest(post_import_runner_sha256, bootstrap_source_sha256):
            raise ForagerMatchedV3LocalRunnerError(
                "bootstrap local runner source drifted during dependency import"
            )
        source_sha256_by_id["local_runner_observed"] = bootstrap_source_sha256
        numpy = importlib.import_module("numpy")
        _finish_runtime_import_transition(
            transition,
            {
                **{
                    module_name: modules[source_id]
                    for source_id, module_name in module_names.items()
                },
                "numpy": numpy,
            },
        )
        return _RuntimeDependencies(
            numpy=numpy,
            local_configuration=modules["local_configuration"],
            worker=modules["matched_alberta_worker"],
            forager=modules["forager"],
            causal_map_forager=modules["causal_map_forager"],
            source_sha256_by_id=source_sha256_by_id,
        )
    finally:
        if transition.status == "loading":
            _poison_runtime_import_transition(transition)


def _validate_runtime_source_bindings(dependencies: _RuntimeDependencies) -> dict[str, str]:
    _validate_runtime_import_transition()
    source_hashes = dict(dependencies.source_sha256_by_id)
    if set(source_hashes) != {*_PINNED_SOURCE_SHA256, "local_runner_observed"}:
        raise ForagerMatchedV3LocalRunnerError("runtime dependency source coverage is not exact")
    for source_id, expected in _PINNED_SOURCE_SHA256.items():
        actual = _require_sha256(source_hashes[source_id], f"runtime source {source_id}")
        if not hmac.compare_digest(actual, expected):
            raise ForagerMatchedV3LocalRunnerError(
                f"runtime dependency source drifted: {source_id}"
            )
    observed_runner = _require_sha256(
        source_hashes["local_runner_observed"], "observed local runner source"
    )
    expected_runner = _require_isolated_top_level_execution_boundary(require_current_source=True)
    if not hmac.compare_digest(observed_runner, expected_runner):
        raise ForagerMatchedV3LocalRunnerError(
            "runtime dependency runner source disagrees with bootstrap identity"
        )
    return source_hashes


def _current_runner_source_sha256() -> str:
    return _bounded_source_sha256(_runtime_source_paths()["local_runner_observed"])


class _InMemoryInt8RewardSink:
    __slots__ = (
        "_aborted",
        "_buffer",
        "_count",
        "_environment_seed",
        "_finalized",
        "_horizon",
        "_numpy",
        "_sha256",
        "_total_reward",
    )

    def __init__(self, numpy_module: Any, environment_seed: int, horizon: int) -> None:
        if horizon != MATCHED_V3_LOCAL_RUNNER_HORIZON:
            raise ForagerMatchedV3LocalRunnerError("reward sink horizon drifted")
        self._numpy = numpy_module
        self._environment_seed = environment_seed
        self._horizon = horizon
        self._buffer = bytearray(horizon)
        self._count = 0
        self._total_reward = 0
        self._sha256 = hashlib.sha256()
        self._finalized = False
        self._aborted = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def total_reward(self) -> int:
        return self._total_reward

    @property
    def finalized(self) -> bool:
        return self._finalized

    def append(self, rewards: Any, biome_regrets: Any) -> None:
        if self._finalized or self._aborted:
            raise ForagerMatchedV3LocalRunnerError("reward sink is already closed")
        np = self._numpy
        reward_array = np.asarray(rewards)
        regret_array = np.asarray(biome_regrets)
        if (
            reward_array.ndim != 1
            or reward_array.dtype != np.dtype(np.float32)
            or regret_array.shape != reward_array.shape
            or regret_array.dtype != np.dtype(np.float32)
            or not bool(np.all(np.isfinite(reward_array)))
            or not bool(np.all(np.isfinite(regret_array)))
        ):
            raise ForagerMatchedV3LocalRunnerError(
                "reward sink chunks must be same-shape finite float32 arrays"
            )
        if reward_array.size == 0:
            raise ForagerMatchedV3LocalRunnerError("reward sink chunks must be nonempty")
        if not bool(np.all(np.isin(reward_array, np.asarray(_REWARD_VALUES, dtype=np.float32)))):
            raise ForagerMatchedV3LocalRunnerError("reward sink observed an out-of-domain reward")
        end = self._count + int(reward_array.size)
        if end > self._horizon:
            raise ForagerMatchedV3LocalRunnerError("reward sink exceeded the exact horizon")
        encoded = reward_array.astype(np.int8, copy=False).tobytes(order="C")
        if len(encoded) != reward_array.size:
            raise ForagerMatchedV3LocalRunnerError("reward int8 encoding length drifted")
        self._buffer[self._count : end] = encoded
        self._sha256.update(encoded)
        self._total_reward += int(np.sum(reward_array, dtype=np.int64))
        self._count = end

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized or self._aborted:
            raise ForagerMatchedV3LocalRunnerError("reward sink was finalized after closure")
        if self._count != self._horizon:
            raise ForagerMatchedV3LocalRunnerError("reward sink does not cover the exact horizon")
        self._finalized = True
        return {
            "schema_version": "alberta.forager_matched_v3.in_memory_reward_trace.v1",
            "environment_seed": self._environment_seed,
            "count": self._count,
            "dtype": "int8",
            "encoding": "signed_int8_twos_complement_v1",
            "sha256": self._sha256.hexdigest(),
            "filesystem_path": None,
        }

    def abort(self) -> None:
        if self._aborted:
            return
        self._buffer[:] = b""
        self._aborted = True
        self._finalized = False
        self._count = 0
        self._total_reward = 0

    def trace_bytes(self) -> bytes:
        if not self._finalized or self._aborted or self._count != self._horizon:
            raise ForagerMatchedV3LocalRunnerError("reward trace is not finalized")
        trace = bytes(self._buffer)
        if len(trace) != self._horizon or not hmac.compare_digest(
            hashlib.sha256(trace).hexdigest(), self._sha256.hexdigest()
        ):
            raise ForagerMatchedV3LocalRunnerError("final reward trace identity drifted")
        return trace


def _validate_candidate(candidate_id: Any) -> str:
    if type(candidate_id) is not str or candidate_id not in MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS:
        raise ForagerMatchedV3LocalRunnerError("local runner candidate is unknown")
    return candidate_id


def _validated_worker_payload(built: Any, candidate_id: str) -> tuple[str, Mapping[str, Any]]:
    if (
        getattr(built, "candidate_id", None) != candidate_id
        or getattr(built, "configuration_sha256", None)
        != _WORKER_ENVELOPE_SHA256_BY_CANDIDATE[candidate_id]
        or getattr(built, "source_descriptor_sha256", None)
        != _LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        or getattr(built, "builder_descriptor_sha256", None)
        != _LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3LocalRunnerError("local configuration binding drifted")
    payload_method = getattr(built, "payload", None)
    if not callable(payload_method):
        raise ForagerMatchedV3LocalRunnerError("local configuration has no payload method")
    payload = payload_method()
    payload_map = _require_exact_keys(
        payload,
        frozenset({"schema_version", "implementation_kind", "configuration"}),
        "local worker envelope",
    )
    if payload_map["schema_version"] != "alberta.forager_matched_worker_configuration.v1":
        raise ForagerMatchedV3LocalRunnerError("local worker schema drifted")
    implementation_kind = payload_map["implementation_kind"]
    if implementation_kind != _IMPLEMENTATION_KIND_BY_CANDIDATE[candidate_id]:
        raise ForagerMatchedV3LocalRunnerError("local implementation kind drifted")
    if type(payload_map["configuration"]) is not dict:
        raise ForagerMatchedV3LocalRunnerError("local agent configuration is not an object")
    return cast(str, implementation_kind), cast(Mapping[str, Any], payload_map["configuration"])


def _receipt_bindings(
    candidate_id: str,
    source_sha256_by_id: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "local_runner_descriptor": {
            "schema_version": LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_RUNNER_DESCRIPTOR_SHA256,
        },
        "configuration_plan": {
            "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
        "cumulative_reward_metric": {
            "schema_version": "alberta.forager_cumulative_reward_metric.v1",
            "sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
            "strict_scorer_source_sha256": _PINNED_SOURCE_SHA256["strict_reward_scorer"],
        },
        "candidate_configuration": {
            "candidate_id": candidate_id,
            "configuration_record_sha256": (
                _CONFIGURATION_RECORD_SHA256_BY_CANDIDATE[candidate_id]
            ),
            "worker_envelope_sha256": (_WORKER_ENVELOPE_SHA256_BY_CANDIDATE[candidate_id]),
            "source_descriptor_sha256": (_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256),
            "builder_descriptor_sha256": (_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256),
        },
        "relevant_source_sha256": dict(source_sha256_by_id),
    }


def _completion_receipt(
    *,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    implementation_kind: str,
    underlying_agent_name: str,
    trace: bytes,
    cumulative_reward: int,
    source_sha256_by_id: Mapping[str, str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION,
        "status": LOCAL_RUNNER_COMPLETION_STATUS,
        "classification": "content_only_unqualified_execution_completion",
        "bindings": _receipt_bindings(candidate_id, source_sha256_by_id),
        "task": {
            "preset": "field_of_view",
            "environment_id": MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID,
            "observation_type": MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE,
            "aperture_size": MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE,
            "horizon": MATCHED_V3_LOCAL_RUNNER_HORIZON,
        },
        "candidate": {
            "candidate_id": candidate_id,
            "implementation_kind": implementation_kind,
        },
        "seed_transport": {
            "environment_seed": environment_seed,
            "agent_seed": agent_seed,
            "environment_transport": "runner_seeds_single_lane_uint31",
            "agent_transport": "runner_agent_seeds_single_lane_uint31",
            "environment_agent_seed_collision": environment_seed == agent_seed,
            "environment_agent_seed_collisions_allowed": True,
        },
        "reward_trace": {
            "encoding": "signed_int8_twos_complement_v1",
            "dtype": "int8",
            "count": len(trace),
            "size_bytes": len(trace),
            "sha256": hashlib.sha256(trace).hexdigest(),
            "allowed_values": list(_REWARD_VALUES),
            "cumulative_reward": cumulative_reward,
            "full_transition_retention": False,
        },
        "underlying_result": {
            "agent_name": underlying_agent_name,
            "environment_seed": environment_seed,
            "steps": MATCHED_V3_LOCAL_RUNNER_HORIZON,
            "mode": "strict",
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return {
        **body,
        "receipt_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def _signed_reward_total(trace: bytes) -> int:
    return sum(value if value < 128 else value - 256 for value in trace)


def _validate_completion_receipt(value: Mapping[str, Any], trace: bytes) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "bindings",
                "task",
                "candidate",
                "seed_transport",
                "reward_trace",
                "underlying_result",
                "claims",
                "limitations",
                "receipt_body_sha256",
            }
        ),
        "local runner completion receipt",
    )
    if (
        receipt["schema_version"] != LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION
        or receipt["status"] != LOCAL_RUNNER_COMPLETION_STATUS
        or receipt["classification"] != "content_only_unqualified_execution_completion"
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion identity drifted")
    candidate = _require_exact_keys(
        receipt["candidate"],
        frozenset({"candidate_id", "implementation_kind"}),
        "local completion candidate",
    )
    candidate_id = _validate_candidate(candidate["candidate_id"])
    if candidate["implementation_kind"] != _IMPLEMENTATION_KIND_BY_CANDIDATE[candidate_id]:
        raise ForagerMatchedV3LocalRunnerError("local completion implementation drifted")
    task = _require_exact_keys(
        receipt["task"],
        frozenset({"preset", "environment_id", "observation_type", "aperture_size", "horizon"}),
        "local completion task",
    )
    if not _exact_json_equal(
        task,
        {
            "preset": "field_of_view",
            "environment_id": MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID,
            "observation_type": MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE,
            "aperture_size": MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE,
            "horizon": MATCHED_V3_LOCAL_RUNNER_HORIZON,
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion task drifted")

    seeds = _require_exact_keys(
        receipt["seed_transport"],
        frozenset(
            {
                "environment_seed",
                "agent_seed",
                "environment_transport",
                "agent_transport",
                "environment_agent_seed_collision",
                "environment_agent_seed_collisions_allowed",
            }
        ),
        "local completion seed transport",
    )
    environment_seed = _require_uint31(seeds["environment_seed"], "receipt environment seed")
    agent_seed = _require_uint31(seeds["agent_seed"], "receipt agent seed")
    if (
        seeds["environment_transport"] != "runner_seeds_single_lane_uint31"
        or seeds["agent_transport"] != "runner_agent_seeds_single_lane_uint31"
        or seeds["environment_agent_seed_collision"] is not (environment_seed == agent_seed)
        or seeds["environment_agent_seed_collisions_allowed"] is not True
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion seed transport drifted")

    bindings = _require_exact_keys(
        receipt["bindings"],
        frozenset(
            {
                "local_runner_descriptor",
                "configuration_plan",
                "cumulative_reward_metric",
                "candidate_configuration",
                "relevant_source_sha256",
            }
        ),
        "local completion bindings",
    )
    expected_sources = dict(_PINNED_SOURCE_SHA256)
    sources = _require_exact_keys(
        bindings["relevant_source_sha256"],
        frozenset({*_PINNED_SOURCE_SHA256, "local_runner_observed"}),
        "local completion source bindings",
    )
    for source_id, expected in expected_sources.items():
        if sources[source_id] != expected:
            raise ForagerMatchedV3LocalRunnerError("local completion source binding drifted")
    _require_sha256(sources["local_runner_observed"], "observed local runner source")
    if not _exact_json_equal(
        bindings["local_runner_descriptor"],
        {
            "schema_version": LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_RUNNER_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion descriptor binding drifted")
    if not _exact_json_equal(
        bindings["configuration_plan"],
        {
            "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
            "sha256": _CONFIGURATION_PLAN_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion plan binding drifted")
    if not _exact_json_equal(
        bindings["cumulative_reward_metric"],
        {
            "schema_version": "alberta.forager_cumulative_reward_metric.v1",
            "sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
            "strict_scorer_source_sha256": _PINNED_SOURCE_SHA256["strict_reward_scorer"],
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion metric binding drifted")
    if not _exact_json_equal(
        bindings["candidate_configuration"],
        {
            "candidate_id": candidate_id,
            "configuration_record_sha256": _CONFIGURATION_RECORD_SHA256_BY_CANDIDATE[candidate_id],
            "worker_envelope_sha256": _WORKER_ENVELOPE_SHA256_BY_CANDIDATE[candidate_id],
            "source_descriptor_sha256": _LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256,
            "builder_descriptor_sha256": _LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion configuration drifted")

    if type(trace) is not bytes or len(trace) != MATCHED_V3_LOCAL_RUNNER_HORIZON:
        raise ForagerMatchedV3LocalRunnerError("local completion trace length drifted")
    if any(value not in _REWARD_BYTES for value in trace):
        raise ForagerMatchedV3LocalRunnerError("local completion trace reward domain drifted")
    trace_record = _require_exact_keys(
        receipt["reward_trace"],
        frozenset(
            {
                "encoding",
                "dtype",
                "count",
                "size_bytes",
                "sha256",
                "allowed_values",
                "cumulative_reward",
                "full_transition_retention",
            }
        ),
        "local completion reward trace",
    )
    trace_sha256 = hashlib.sha256(trace).hexdigest()
    if not _exact_json_equal(
        trace_record,
        {
            "encoding": "signed_int8_twos_complement_v1",
            "dtype": "int8",
            "count": MATCHED_V3_LOCAL_RUNNER_HORIZON,
            "size_bytes": MATCHED_V3_LOCAL_RUNNER_HORIZON,
            "sha256": trace_sha256,
            "allowed_values": list(_REWARD_VALUES),
            "cumulative_reward": _signed_reward_total(trace),
            "full_transition_retention": False,
        },
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion trace binding drifted")
    underlying = _require_exact_keys(
        receipt["underlying_result"],
        frozenset({"agent_name", "environment_seed", "steps", "mode"}),
        "local completion underlying result",
    )
    if (
        type(underlying["agent_name"]) is not str
        or _IDENTIFIER_RE.fullmatch(underlying["agent_name"]) is None
        or type(underlying["environment_seed"]) is not int
        or underlying["environment_seed"] != environment_seed
        or type(underlying["steps"]) is not int
        or underlying["steps"] != MATCHED_V3_LOCAL_RUNNER_HORIZON
        or underlying["mode"] != "strict"
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion result identity drifted")
    claims = _require_exact_keys(receipt["claims"], frozenset(_claims()), "local completion claims")
    if not _exact_json_equal(claims, _claims()) or any(
        claim is not False for claim in claims.values()
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion claim became true")
    if not _exact_json_equal(receipt["limitations"], _limitations()):
        raise ForagerMatchedV3LocalRunnerError("local completion limitations drifted")
    body = dict(receipt)
    supplied_body_sha256 = body.pop("receipt_body_sha256")
    _require_sha256(supplied_body_sha256, "local completion body")
    if not hmac.compare_digest(
        supplied_body_sha256, hashlib.sha256(_canonical_json(body)).hexdigest()
    ):
        raise ForagerMatchedV3LocalRunnerError("local completion body digest drifted")
    _assert_plain_unaliased_json(receipt)
    _canonical_json(receipt)


@dataclass(frozen=True, slots=True)
class MatchedV3LocalRunCompletion:
    """One bounded reward trace and its strict non-authorizing completion receipt."""

    candidate_id: str
    environment_seed: int
    agent_seed: int
    reward_trace: bytes
    canonical_receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_candidate(self.candidate_id)
        _require_uint31(self.environment_seed, "completion environment seed")
        _require_uint31(self.agent_seed, "completion agent seed")
        _require_sha256(self.receipt_sha256, "completion receipt")
        if type(self.reward_trace) is not bytes or type(self.canonical_receipt_bytes) is not bytes:
            raise ForagerMatchedV3LocalRunnerError(
                "completion content must be exact immutable bytes"
            )
        if not hmac.compare_digest(
            hashlib.sha256(self.canonical_receipt_bytes).hexdigest(), self.receipt_sha256
        ):
            raise ForagerMatchedV3LocalRunnerError("completion receipt digest disagrees")
        receipt = _strict_json_load(self.canonical_receipt_bytes)
        _validate_completion_receipt(receipt, self.reward_trace)
        if (
            receipt["candidate"]["candidate_id"] != self.candidate_id
            or receipt["seed_transport"]["environment_seed"] != self.environment_seed
            or receipt["seed_transport"]["agent_seed"] != self.agent_seed
        ):
            raise ForagerMatchedV3LocalRunnerError("completion dataclass identity drifted")

    def receipt(self) -> dict[str, Any]:
        """Replay structural content; this object itself grants no authority."""

        return parse_matched_v3_local_completion_receipt(
            self.canonical_receipt_bytes,
            reward_trace=self.reward_trace,
            expected_receipt_sha256=self.receipt_sha256,
        )


def consume_matched_v3_local_outcome(
    *,
    outcome_capability: object,
    explicit_content_access_opt_in: bool,
) -> MatchedV3LocalRunCompletion:
    """Consume one live outcome handle and expose its structural publication content."""

    if (
        type(explicit_content_access_opt_in) is not bool
        or explicit_content_access_opt_in is not True
    ):
        raise ForagerMatchedV3LocalRunnerError(
            "local outcome access requires exact explicit opt-in"
        )
    if type(outcome_capability) is not _LocalOutcomeCapability:
        raise ForagerMatchedV3LocalRunnerError(
            "local outcome access requires an authentic opaque capability"
        )
    with _CAPABILITY_LOCK:
        state = _OUTCOMES.get(outcome_capability)
        if state is None or state.status != "live":
            raise ForagerMatchedV3LocalRunnerError(
                "local outcome capability is unknown, stale, or already consumed"
            )
        current_pid = os.getpid()
        if state.pid != current_pid:
            state.status = "consumed"
            raise ForagerMatchedV3LocalRunnerError(
                "local outcome capability cannot cross a PID boundary"
            )
        execution_state = _CAPABILITIES.get(state.execution_capability)
        if (
            execution_state is None
            or execution_state.pid != current_pid
            or execution_state.status != "consumed"
            or id(state.execution_capability) != state.execution_capability_identity
        ):
            state.status = "consumed"
            raise ForagerMatchedV3LocalRunnerError(
                "local outcome lost its consumed execution-capability binding"
            )
        state.status = "consumed"
    current_runner_source_sha256 = _require_sha256(
        _current_runner_source_sha256(), "current local runner source"
    )
    registered_runner_source_sha256 = _require_sha256(
        state.runner_source_sha256, "registered local runner source"
    )
    if not hmac.compare_digest(current_runner_source_sha256, registered_runner_source_sha256):
        raise ForagerMatchedV3LocalRunnerError("local outcome runner source is stale")
    completion = state.completion
    if type(completion) is not MatchedV3LocalRunCompletion:
        raise ForagerMatchedV3LocalRunnerError("local outcome structural content is stale")
    registered_trace_sha256 = _require_sha256(state.trace_sha256, "registered local outcome trace")
    registered_receipt_sha256 = _require_sha256(
        state.receipt_sha256, "registered local outcome receipt"
    )
    if (
        not hmac.compare_digest(
            hashlib.sha256(completion.reward_trace).hexdigest(), registered_trace_sha256
        )
        or not hmac.compare_digest(completion.receipt_sha256, registered_receipt_sha256)
        or not hmac.compare_digest(
            hashlib.sha256(completion.canonical_receipt_bytes).hexdigest(),
            registered_receipt_sha256,
        )
    ):
        raise ForagerMatchedV3LocalRunnerError("local outcome content identity is stale")
    parse_matched_v3_local_completion_receipt(
        completion.canonical_receipt_bytes,
        reward_trace=completion.reward_trace,
        expected_receipt_sha256=completion.receipt_sha256,
    )
    return completion


def run_matched_v3_local_candidate(
    *,
    candidate_id: str,
    environment_seed: int,
    agent_seed: int,
    explicit_execution_opt_in: bool,
    execution_capability: object,
) -> object:
    """Execute one exact local arm after consuming an explicit single-use capability."""

    if type(explicit_execution_opt_in) is not bool or explicit_execution_opt_in is not True:
        raise ForagerMatchedV3LocalRunnerError(
            "local execution requires exact explicit run-time opt-in"
        )
    bootstrap_source_sha256 = _require_isolated_top_level_execution_boundary(
        require_current_source=False
    )
    _consume_capability(execution_capability)
    _require_clean_runtime_import_boundary("pre-import run boundary")
    _require_bootstrap_runner_source_current(bootstrap_source_sha256)
    exact_candidate_id = _validate_candidate(candidate_id)
    exact_environment_seed = _require_uint31(environment_seed, "environment seed")
    exact_agent_seed = _require_uint31(agent_seed, "agent seed")

    sink: _InMemoryInt8RewardSink | None = None
    try:
        dependencies = _load_runtime_dependencies()
        source_sha256_by_id = _validate_runtime_source_bindings(dependencies)
        builder = getattr(
            dependencies.local_configuration,
            "build_matched_v3_local_configuration",
            None,
        )
        if not callable(builder):
            raise ForagerMatchedV3LocalRunnerError("local configuration builder is unavailable")
        built = builder(exact_candidate_id)
        implementation_kind, configuration_payload = _validated_worker_payload(
            built, exact_candidate_id
        )
        parser = getattr(dependencies.worker, "_parse_agent_configuration", None)
        if not callable(parser):
            raise ForagerMatchedV3LocalRunnerError(
                "local typed configuration parser is unavailable"
            )
        agent_configuration = parser(implementation_kind, configuration_payload)

        env_type = getattr(dependencies.forager, "ForagerEnvConfig", None)
        benchmark_type = getattr(dependencies.forager, "ForagerBenchmarkConfig", None)
        if env_type is None or benchmark_type is None:
            raise ForagerMatchedV3LocalRunnerError("local Forager config types are unavailable")
        environment = env_type.paper_field_of_view(
            aperture_size=MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE
        )
        if (
            getattr(environment, "resolved_env_id", None) != MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID
            or getattr(environment, "resolved_observation_type", None)
            != MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE
            or getattr(environment, "aperture_size", None) != MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE
        ):
            raise ForagerMatchedV3LocalRunnerError("local field-of-view task drifted")
        benchmark = benchmark_type(
            environment=environment,
            steps=MATCHED_V3_LOCAL_RUNNER_HORIZON,
            seed=exact_environment_seed,
            ewm_decay=0.999,
            record_every=100,
            final_window=100_000,
            jax_chunk_size=10_000,
        )
        if (
            getattr(benchmark, "steps", None) != MATCHED_V3_LOCAL_RUNNER_HORIZON
            or getattr(benchmark, "seed", None) != exact_environment_seed
            or getattr(benchmark, "environment", None) is not environment
        ):
            raise ForagerMatchedV3LocalRunnerError("local benchmark task drifted")

        def sink_factory(seed: int, steps: int) -> _InMemoryInt8RewardSink:
            nonlocal sink
            if (
                type(seed) is not int
                or seed != exact_environment_seed
                or type(steps) is not int
                or steps != MATCHED_V3_LOCAL_RUNNER_HORIZON
                or sink is not None
            ):
                raise ForagerMatchedV3LocalRunnerError(
                    "local runner requested an unexpected reward sink"
                )
            sink = _InMemoryInt8RewardSink(
                dependencies.numpy,
                exact_environment_seed,
                MATCHED_V3_LOCAL_RUNNER_HORIZON,
            )
            return sink

        runner_kwargs = {
            "agent_seeds": (exact_agent_seed,),
            "mode": "strict",
            "reward_trace_sink_factory": sink_factory,
        }
        if implementation_kind == "alberta_causal_map":
            runner = getattr(
                dependencies.causal_map_forager,
                "run_causal_map_forager_seeds",
                None,
            )
        elif implementation_kind == "alberta_horde_actor_critic":
            runner = getattr(dependencies.forager, "run_alberta_forager_seeds", None)
        else:
            runner = getattr(dependencies.forager, "run_rtu_rtrl_forager_seeds", None)
        if not callable(runner):
            raise ForagerMatchedV3LocalRunnerError("local candidate runner is unavailable")
        results = runner(
            agent_configuration,
            benchmark,
            (exact_environment_seed,),
            **runner_kwargs,
        )
        if type(results) is not tuple or len(results) != 1:
            raise ForagerMatchedV3LocalRunnerError("local runner result cardinality drifted")
        result = results[0]
        if sink is None or not sink.finalized:
            raise ForagerMatchedV3LocalRunnerError("local runner did not finalize its reward sink")
        if (
            getattr(result, "seed", None) != exact_environment_seed
            or getattr(result, "steps", None) != MATCHED_V3_LOCAL_RUNNER_HORIZON
        ):
            raise ForagerMatchedV3LocalRunnerError("local runner result identity drifted")
        result_total = getattr(result, "total_reward", None)
        if type(result_total) not in {int, float}:
            raise ForagerMatchedV3LocalRunnerError("local runner reward total disagrees")
        numeric_result_total = cast(int | float, result_total)
        if not math.isfinite(numeric_result_total) or numeric_result_total != sink.total_reward:
            raise ForagerMatchedV3LocalRunnerError("local runner reward total disagrees")
        agent_name = getattr(result, "agent", None)
        if type(agent_name) is not str or _IDENTIFIER_RE.fullmatch(agent_name) is None:
            raise ForagerMatchedV3LocalRunnerError("local runner agent identity is invalid")
        trace = sink.trace_bytes()
        current_runner_source_sha256 = _require_sha256(
            _current_runner_source_sha256(), "post-run local runner source"
        )
        if not hmac.compare_digest(
            current_runner_source_sha256,
            source_sha256_by_id["local_runner_observed"],
        ):
            raise ForagerMatchedV3LocalRunnerError(
                "local runner source drifted during candidate execution"
            )
        receipt = _completion_receipt(
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            implementation_kind=implementation_kind,
            underlying_agent_name=agent_name,
            trace=trace,
            cumulative_reward=sink.total_reward,
            source_sha256_by_id=source_sha256_by_id,
        )
        receipt_bytes = _canonical_json(receipt)
        completion = MatchedV3LocalRunCompletion(
            candidate_id=exact_candidate_id,
            environment_seed=exact_environment_seed,
            agent_seed=exact_agent_seed,
            reward_trace=trace,
            canonical_receipt_bytes=receipt_bytes,
            receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        )
        exact_execution_capability = cast(_LocalExecutionCapability, execution_capability)
        outcome_capability = _LocalOutcomeCapability()
        with _CAPABILITY_LOCK:
            execution_state = _CAPABILITIES.get(exact_execution_capability)
            if (
                execution_state is None
                or execution_state.pid != os.getpid()
                or execution_state.status != "consumed"
            ):
                raise ForagerMatchedV3LocalRunnerError(
                    "consumed execution capability disappeared before outcome binding"
                )
            _OUTCOMES[outcome_capability] = _OutcomeState(
                pid=execution_state.pid,
                status="live",
                execution_capability=exact_execution_capability,
                execution_capability_identity=id(exact_execution_capability),
                trace_sha256=hashlib.sha256(trace).hexdigest(),
                receipt_sha256=completion.receipt_sha256,
                runner_source_sha256=current_runner_source_sha256,
                completion=completion,
            )
        return outcome_capability
    except ForagerMatchedV3LocalRunnerError:
        if sink is not None:
            sink.abort()
        raise
    except Exception as exc:
        if sink is not None:
            sink.abort()
        raise ForagerMatchedV3LocalRunnerError(
            "local candidate execution failed after capability consumption"
        ) from exc
    except BaseException:
        if sink is not None:
            sink.abort()
        raise


def matched_v3_local_runner_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the frozen local-runner descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_local_runner_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_local_runner_descriptor_sha256() -> str:
    """Return the exact local-runner descriptor digest."""

    return LOCAL_RUNNER_DESCRIPTOR_SHA256


def parse_matched_v3_local_runner_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen local-runner descriptor."""

    value = _strict_json_load(raw)
    if value != _descriptor() or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), LOCAL_RUNNER_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3LocalRunnerError(
            "local-runner descriptor differs from its frozen identity"
        )
    return value


def parse_matched_v3_local_completion_receipt(
    raw: bytes,
    *,
    reward_trace: bytes,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Replay a completion only with its exact trace and external receipt digest."""

    _require_sha256(expected_receipt_sha256, "expected local completion receipt")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_receipt_sha256
    ):
        raise ForagerMatchedV3LocalRunnerError(
            "local completion receipt full-file digest disagrees"
        )
    value = _strict_json_load(raw)
    _validate_completion_receipt(value, reward_trace)
    return value


__all__ = [
    "ForagerMatchedV3LocalRunnerError",
    "LOCAL_RUNNER_COMPLETION_SCHEMA_VERSION",
    "LOCAL_RUNNER_COMPLETION_STATUS",
    "LOCAL_RUNNER_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_RUNNER_DESCRIPTOR_SHA256",
    "LOCAL_RUNNER_STATUS",
    "MATCHED_V3_LOCAL_RUNNER_APERTURE_SIZE",
    "MATCHED_V3_LOCAL_RUNNER_CANDIDATE_IDS",
    "MATCHED_V3_LOCAL_RUNNER_ENVIRONMENT_ID",
    "MATCHED_V3_LOCAL_RUNNER_HORIZON",
    "MATCHED_V3_LOCAL_RUNNER_ISOLATED_MODULE_NAME",
    "MATCHED_V3_LOCAL_RUNNER_OBSERVATION_TYPE",
    "MatchedV3LocalRunCompletion",
    "canonical_matched_v3_local_runner_descriptor_bytes",
    "consume_matched_v3_local_outcome",
    "issue_matched_v3_local_execution_capability",
    "matched_v3_local_runner_descriptor",
    "matched_v3_local_runner_descriptor_sha256",
    "parse_matched_v3_local_completion_receipt",
    "parse_matched_v3_local_runner_descriptor",
    "run_matched_v3_local_candidate",
]
