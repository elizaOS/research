"""Pure-stdlib file-path bootstrap for the sole v3 development attempt.

The production command has only two modes: describe the byte-bound ledger
candidate, or consume an already-issued ledger and run the operational
evaluator once.  It never issues a root, creates or repairs genesis, writes an
experiment artifact, retries an attempt, selects a result, or promotes a
scientific claim.

This file deliberately has no first-party or JAX imports.  The ledger and
declared-source loader are loaded anonymously from their exact source paths;
the canonical Alberta namespace is installed only after ``started.json`` has
been durably created and validated.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final, cast

DEVELOPMENT_ONLY: Final = True
ROOT_ISSUANCE_AUTHORIZED: Final = False
GENESIS_CREATION_AUTHORIZED: Final = False
EXPERIMENT_OUTPUT_WRITES_ALLOWED: Final = False
LEDGER_WRITES_ONLY: Final = True
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
RETRY_OR_RECOVERY_AUTHORIZED: Final = False

PRODUCTION_CAMPAIGN: Final = "compositional-future-utility-calibration-v3-cadence-separated"
PRODUCTION_DEVELOPMENT_ROOT: Final = 317_707_403
PRODUCTION_DEVELOPMENT_ROOT_HEX: Final = "0x12EFD48B"
PRODUCTION_PROTOCOL_CONFIG_SHA256: Final = (
    "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"
)
PRODUCTION_CONTROL_PROTOCOL_CONFIG_SHA256: Final = (
    "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
)
PRODUCTION_RUNTIME_CONFIG_SHA256: Final = (
    "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6"
)
PRODUCTION_CONSUMED_HISTORY_SHA256: Final = (
    "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
)
PRODUCTION_KEY_MANIFEST_SHA256: Final = (
    "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"
)
PRODUCTION_STREAM_SHA256: Final = "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758"
PRODUCTION_CADENCE_BOUND_STREAM_SHA256: Final = (
    "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
)
PRODUCTION_SOURCE_ENVELOPE_SHA256: Final = (
    "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
)
PRODUCTION_ACKNOWLEDGEMENT: Final = (
    "I acknowledge that starting the sole nonpromoting v3 development attempt "
    "irreversibly consumes root 0x12EFD48B; no retry, recovery, winner selection, "
    "evidence promotion, or artifact output is authorized."
)

PRODUCTION_LEDGER_RELATIVE_PATH: Final = (
    "outputs/compositional_future_utility_calibration_v3/one_shot_ledger"
)
PRODUCTION_LEDGER_PRIMITIVE_RELATIVE_PATH: Final = (
    "alberta_framework/evaluation/_one_shot_development_ledger.py"
)
PRODUCTION_DECLARED_LOADER_RELATIVE_PATH: Final = (
    "alberta_framework/evaluation/_compositional_future_utility_declared_loader.py"
)
PRODUCTION_EVALUATOR_MODULE: Final = (
    "alberta_framework.evaluation._compositional_future_utility_v3_evaluator"
)
PRODUCTION_REPORT_GATE_MODULE: Final = (
    "alberta_framework.evaluation._compositional_future_utility_v3_report_gate"
)

# Keep both tuples literal: the production closure is meant to be auditable
# without importing this module or deriving a path from a module name.
PRODUCTION_CLOSURE_NAMES: Final = (
    "alberta_framework.core.compositional_features",
    "alberta_framework.core.future_utility",
    "alberta_framework.core.resource_manager",
    "alberta_framework.evaluation._compositional_future_utility_calibration_engine",
    "alberta_framework.evaluation._compositional_future_utility_state_gate",
    "alberta_framework.evaluation._compositional_future_utility_v3_evaluator",
    "alberta_framework.evaluation._compositional_future_utility_v3_report_gate",
    "alberta_framework.evaluation._compositional_future_utility_v3_reward_counts",
    "alberta_framework.evaluation.compositional_control_life_development",
    "alberta_framework.evaluation.compositional_discovery_development",
    "alberta_framework.evaluation.compositional_future_utility_calibration_v3_protocol",
    "alberta_framework.evaluation.compositional_future_utility_calibration_v3_source",
    "alberta_framework.evaluation.compositional_future_utility_panel_core",
    "alberta_framework.evaluation.generated_birth_identity_ledger",
    "alberta_framework.evaluation.generated_birth_identity_scrub_epoch",
    "alberta_framework.evaluation.generated_class_lifecycle_scrub",
    "alberta_framework.evaluation.generated_class_recurrence",
    "alberta_framework.evaluation.generated_expression_lineage",
    "alberta_framework.evaluation.generated_reacquisition_epoch",
)
PRODUCTION_CLOSURE_PATHS: Final = (
    "alberta_framework/core/compositional_features.py",
    "alberta_framework/core/future_utility.py",
    "alberta_framework/core/resource_manager.py",
    "alberta_framework/evaluation/_compositional_future_utility_calibration_engine.py",
    "alberta_framework/evaluation/_compositional_future_utility_state_gate.py",
    "alberta_framework/evaluation/_compositional_future_utility_v3_evaluator.py",
    "alberta_framework/evaluation/_compositional_future_utility_v3_report_gate.py",
    "alberta_framework/evaluation/_compositional_future_utility_v3_reward_counts.py",
    "alberta_framework/evaluation/compositional_control_life_development.py",
    "alberta_framework/evaluation/compositional_discovery_development.py",
    "alberta_framework/evaluation/compositional_future_utility_calibration_v3_protocol.py",
    "alberta_framework/evaluation/compositional_future_utility_calibration_v3_source.py",
    "alberta_framework/evaluation/compositional_future_utility_panel_core.py",
    "alberta_framework/evaluation/generated_birth_identity_ledger.py",
    "alberta_framework/evaluation/generated_birth_identity_scrub_epoch.py",
    "alberta_framework/evaluation/generated_class_lifecycle_scrub.py",
    "alberta_framework/evaluation/generated_class_recurrence.py",
    "alberta_framework/evaluation/generated_expression_lineage.py",
    "alberta_framework/evaluation/generated_reacquisition_epoch.py",
)

PRODUCTION_ARM_NAMES: Final = (
    "current_mix0_decay095_none",
    "future_mix1_decay095_none",
    "calibrated_mix05_decay095_none",
    "normalized_mix1_decay095_uncertainty_age",
    "horizon_mix1_decay883_uncertainty_age",
)
PRODUCTION_AUTHORIZATION_STAGES: Final = (
    "entry-preflight",
    "before-scan:current_mix0_decay095_none",
    "after-scan:current_mix0_decay095_none",
    "before-scan:future_mix1_decay095_none",
    "after-scan:future_mix1_decay095_none",
    "before-scan:calibrated_mix05_decay095_none",
    "after-scan:calibrated_mix05_decay095_none",
    "before-scan:normalized_mix1_decay095_uncertainty_age",
    "after-scan:normalized_mix1_decay095_uncertainty_age",
    "before-scan:horizon_mix1_decay883_uncertainty_age",
    "after-scan:horizon_mix1_decay883_uncertainty_age",
    "closure-postflight",
    "completion-postflight",
)

DESCRIPTION_SCHEMA: Final = "alberta.one-shot-v3-bootstrap.description.v1"
OUTCOME_SCHEMA: Final = "alberta.one-shot-v3-bootstrap.outcome.v2"
FAILURE_SCHEMA: Final = "alberta.one-shot-v3-bootstrap.failure.v1"


def _exact_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _exact_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_relative_path(value: object, name: str) -> str:
    text = _exact_string(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise ValueError(f"{name} must be a canonical repository-relative POSIX path")
    return text


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapCampaign:
    """Complete generic campaign definition used by the internal orchestrator."""

    campaign: str
    development_root: int
    development_root_hex: str
    protocol_config_sha256: str
    control_protocol_config_sha256: str
    runtime_config_sha256: str
    consumed_history_sha256: str
    key_manifest_sha256: str
    stream_sha256: str
    cadence_bound_stream_sha256: str
    source_envelope_sha256: str
    acknowledgement: str
    repo_root: Path
    ledger_relative_path: str
    ledger_primitive_relative_path: str
    declared_loader_relative_path: str
    closure: tuple[tuple[str, str], ...]
    evaluator_module: str
    report_gate_module: str
    authorization_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_string(self.campaign, "campaign")
        if type(self.development_root) is not int or not 0 <= self.development_root <= 0xFFFFFFFF:
            raise ValueError("development_root must be an exact unsigned 32-bit integer")
        if type(self.development_root_hex) is not str or self.development_root_hex != (
            f"0x{self.development_root:08X}"
        ):
            raise ValueError("development_root_hex does not encode development_root")
        for field in (
            "protocol_config_sha256",
            "control_protocol_config_sha256",
            "runtime_config_sha256",
            "consumed_history_sha256",
            "key_manifest_sha256",
            "stream_sha256",
            "cadence_bound_stream_sha256",
            "source_envelope_sha256",
        ):
            _sha256(getattr(self, field), field)
        _exact_string(self.acknowledgement, "acknowledgement")
        if not isinstance(self.repo_root, Path):
            raise TypeError("repo_root must be a pathlib.Path")
        for field in (
            "ledger_relative_path",
            "ledger_primitive_relative_path",
            "declared_loader_relative_path",
        ):
            _canonical_relative_path(getattr(self, field), field)
        if type(self.closure) is not tuple or not self.closure:
            raise TypeError("closure must be a non-empty exact tuple")
        names: list[str] = []
        paths: list[str] = []
        for index, item in enumerate(self.closure):
            if type(item) is not tuple or len(item) != 2:
                raise TypeError(f"closure[{index}] must be an exact pair")
            canonical_name = _exact_string(item[0], f"closure[{index}].canonical_name")
            relative_path = _canonical_relative_path(item[1], f"closure[{index}].relative_path")
            if canonical_name.replace(".", "/") + ".py" != relative_path:
                raise ValueError(f"closure[{index}] name and path differ")
            names.append(canonical_name)
            paths.append(relative_path)
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("closure names must be unique and sorted")
        if len(paths) != len(set(paths)):
            raise ValueError("closure paths must be unique")
        if self.evaluator_module not in names or self.report_gate_module not in names:
            raise ValueError("evaluator and report gate must both be declared in the closure")
        if type(self.authorization_stages) is not tuple or not self.authorization_stages:
            raise TypeError("authorization_stages must be a non-empty exact tuple")
        for index, stage in enumerate(self.authorization_stages):
            _exact_string(stage, f"authorization_stages[{index}]")
        if len(set(self.authorization_stages)) != len(self.authorization_stages):
            raise ValueError("authorization stages must be unique")
        if "closure-postflight" not in self.authorization_stages:
            raise ValueError("authorization stages must include closure-postflight")


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapOutcome:
    """Validated in-memory result with one fresh descriptive-report view."""

    campaign: str
    panel_completed: bool
    canonical_report_json: str
    report_sha256: str
    terminal_sha256: str
    execution_source_closure_sha256: str

    def __post_init__(self) -> None:
        _exact_string(self.campaign, "campaign")
        if type(self.panel_completed) is not bool or not self.panel_completed:
            raise ValueError("successful bootstrap outcome requires a completed panel")
        for field in (
            "report_sha256",
            "terminal_sha256",
            "execution_source_closure_sha256",
        ):
            _sha256(getattr(self, field), field)
        canonical_report_json = _exact_string(
            self.canonical_report_json,
            "canonical_report_json",
        )
        try:
            parsed = json.loads(canonical_report_json)
        except (TypeError, ValueError) as error:
            raise ValueError("canonical_report_json cannot be decoded") from error
        if type(parsed) is not dict:
            raise TypeError("canonical_report_json must encode an exact JSON object")
        try:
            observed_canonical = _canonical_json(parsed)
        except (TypeError, ValueError) as error:
            raise ValueError("canonical_report_json contains a noncanonical value") from error
        if observed_canonical != canonical_report_json:
            raise ValueError("canonical_report_json is not canonical JSON")
        report = cast(dict[str, object], parsed)
        if report.get("report_sha256") != self.report_sha256:
            raise ValueError("canonical report hash does not match report_sha256")
        body = {key: value for key, value in report.items() if key != "report_sha256"}
        body_sha256 = hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()
        if body_sha256 != self.report_sha256:
            raise ValueError("canonical report SHA-256 does not reconstruct")

    @property
    def descriptive_report(self) -> dict[str, object]:
        """Return a fresh copy of the validated descriptive report."""

        value = json.loads(self.canonical_report_json)
        if type(value) is not dict:
            raise RuntimeError("validated descriptive report is no longer a JSON object")
        return cast(dict[str, object], value)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": OUTCOME_SCHEMA,
            "campaign": self.campaign,
            "status": "completed",
            "panel_completed": self.panel_completed,
            "report_sha256": self.report_sha256,
            "terminal_sha256": self.terminal_sha256,
            "execution_source_closure_sha256": self.execution_source_closure_sha256,
            "descriptive_report": self.descriptive_report,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
            "retry_or_recovery_authorized": False,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedCampaign:
    definition: BootstrapCampaign
    ledger: ModuleType
    declared_loader: ModuleType
    closure_records: tuple[dict[str, object], ...]
    execution_source_closure_sha256: str
    bootstrap_sha256: str
    ledger_primitive_sha256: str
    declared_loader_sha256: str
    ledger_spec: Any
    genesis: dict[str, object]


def _reject_preloaded_runtime() -> None:
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "alberta_framework"
        or name.startswith("alberta_framework.")
        or name.startswith("jax")
    )
    if forbidden:
        raise RuntimeError(
            "bootstrap requires a clean process; forbidden modules are loaded: "
            + ", ".join(forbidden)
        )


def _validated_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("repo_root must be an absolute pathlib.Path")
    if root.is_symlink():
        raise ValueError("repo_root must not be a symlink")
    resolved = root.resolve(strict=True)
    if resolved != root or not resolved.is_dir():
        raise ValueError("repo_root must be an exact resolved directory")
    return resolved


def _resolved_relative(root: Path, relative_path: str, *, kind: str) -> Path:
    relative = PurePosixPath(_canonical_relative_path(relative_path, "relative_path"))
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{kind} path contains a symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or not resolved.is_relative_to(root):
        raise ValueError(f"{kind} path is not exact and inside repo_root: {relative_path}")
    if kind == "directory":
        if not resolved.is_dir():
            raise ValueError(f"path is not a directory: {relative_path}")
    elif not resolved.is_file():
        raise ValueError(f"path is not a regular file: {relative_path}")
    return resolved


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anonymous_load(path: Path, *, label: str, expected_sha256: str) -> ModuleType:
    _sha256(expected_sha256, "expected_sha256")
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} source changed before anonymous load")
    private_name = (
        f"_one_shot_v3_{label}_"
        f"{hashlib.sha256((str(path) + str(id(object()))).encode('utf-8')).hexdigest()}"
    )
    spec = importlib.util.spec_from_file_location(private_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not construct anonymous {label} module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(cast(str, module.__file__)).resolve(strict=True) != path:
            raise RuntimeError(f"anonymous {label} module origin changed")
        if _sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{label} source changed during anonymous load")
    except BaseException:
        if sys.modules.get(private_name) is module:
            del sys.modules[private_name]
        raise
    if sys.modules.get(private_name) is not module:
        raise RuntimeError(f"anonymous {label} module registration changed during load")
    del sys.modules[private_name]
    return module


def _bootstrap_source_path() -> Path:
    path = Path(__file__)
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("bootstrap source path must be absolute and nonsymlinked")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise RuntimeError("bootstrap source path must be exact and resolved")
    return resolved


def _source_records(
    root: Path, closure: Sequence[tuple[str, str]]
) -> tuple[dict[str, object], ...]:
    records: tuple[dict[str, object], ...] = tuple(
        cast(
            dict[str, object],
            {
                "canonical_name": canonical_name,
                "relative_path": relative_path,
                "sha256": _sha256_file(_resolved_relative(root, relative_path, kind="source")),
            },
        )
        for canonical_name, relative_path in closure
    )
    if list(records) != sorted(records, key=lambda record: cast(str, record["canonical_name"])):
        raise RuntimeError("source records are not in canonical-name order")
    return records


def _prepare_campaign(definition: BootstrapCampaign) -> _PreparedCampaign:
    if type(definition) is not BootstrapCampaign:
        raise TypeError("definition must be an exact BootstrapCampaign")
    root = _validated_root(definition.repo_root)
    bootstrap_path = _bootstrap_source_path()
    ledger_path = _resolved_relative(
        root, definition.ledger_primitive_relative_path, kind="ledger primitive"
    )
    loader_path = _resolved_relative(
        root, definition.declared_loader_relative_path, kind="declared loader"
    )
    bootstrap_sha256 = _sha256_file(bootstrap_path)
    ledger_sha256 = _sha256_file(ledger_path)
    loader_sha256 = _sha256_file(loader_path)
    ledger = _anonymous_load(ledger_path, label="ledger", expected_sha256=ledger_sha256)
    declared_loader = _anonymous_load(
        loader_path, label="declared_loader", expected_sha256=loader_sha256
    )
    closure_records = _source_records(root, definition.closure)
    closure_sha256 = cast(str, ledger.canonical_json_sha256(list(closure_records)))
    spec = ledger.OneShotDevelopmentLedgerSpec(
        campaign=definition.campaign,
        development_root=definition.development_root,
        development_root_hex=definition.development_root_hex,
        protocol_config_sha256=definition.protocol_config_sha256,
        control_protocol_config_sha256=definition.control_protocol_config_sha256,
        runtime_config_sha256=definition.runtime_config_sha256,
        consumed_history_sha256=definition.consumed_history_sha256,
        key_manifest_sha256=definition.key_manifest_sha256,
        stream_sha256=definition.stream_sha256,
        cadence_bound_stream_sha256=definition.cadence_bound_stream_sha256,
        source_envelope_sha256=definition.source_envelope_sha256,
        execution_source_closure_sha256=closure_sha256,
        bootstrap_sha256=bootstrap_sha256,
        ledger_primitive_sha256=ledger_sha256,
        declared_loader_sha256=loader_sha256,
        acknowledgement=definition.acknowledgement,
    )
    genesis = cast(dict[str, object], ledger.genesis_record(spec))
    return _PreparedCampaign(
        definition=definition,
        ledger=ledger,
        declared_loader=declared_loader,
        closure_records=closure_records,
        execution_source_closure_sha256=closure_sha256,
        bootstrap_sha256=bootstrap_sha256,
        ledger_primitive_sha256=ledger_sha256,
        declared_loader_sha256=loader_sha256,
        ledger_spec=spec,
        genesis=genesis,
    )


def describe_campaign(definition: BootstrapCampaign) -> dict[str, object]:
    """Describe an exact candidate without creating a directory or ledger."""

    _reject_preloaded_runtime()
    prepared = _prepare_campaign(definition)
    spec_config = dataclasses.asdict(cast(Any, prepared.ledger_spec))
    description: dict[str, object] = {
        "schema": DESCRIPTION_SCHEMA,
        "campaign": definition.campaign,
        "ledger_relative_path": definition.ledger_relative_path,
        "ledger_spec": spec_config,
        "execution_source_closure": [dict(record) for record in prepared.closure_records],
        "genesis_candidate": dict(prepared.genesis),
        "authority": {
            "development_only": True,
            "root_issuance_authorized": False,
            "genesis_creation_authorized": False,
            "experiment_output_writes_allowed": False,
            "ledger_writes_only": True,
            "retry_or_recovery_authorized": False,
            "evidence_authorized": False,
            "scientific_promotion_allowed": False,
        },
    }
    # Canonicalization is also a strict-tree check.  The return remains a fresh
    # object so callers cannot mutate the prepared bindings.
    prepared.ledger.canonical_json(description)
    return description


def _validated_ledger_directory(definition: BootstrapCampaign) -> Path:
    root = _validated_root(definition.repo_root)
    return _resolved_relative(root, definition.ledger_relative_path, kind="directory")


def _bindings_from_prepared(
    prepared: _PreparedCampaign,
    report_gate: ModuleType,
    started: Mapping[str, object],
) -> object:
    genesis_sha256 = _sha256(prepared.genesis.get("genesis_sha256"), "genesis_sha256")
    started_sha256 = _sha256(started.get("started_sha256"), "started_sha256")
    bindings = report_gate.ExpectedExecutionBindings(
        execution_source_closure_sha256=prepared.execution_source_closure_sha256,
        bootstrap_sha256=prepared.bootstrap_sha256,
        ledger_primitive_sha256=prepared.ledger_primitive_sha256,
        declared_loader_sha256=prepared.declared_loader_sha256,
        genesis_sha256=genesis_sha256,
        started_sha256=started_sha256,
    )
    return bindings


def _canonicalize_loaded_closure(
    prepared: _PreparedCampaign,
    loaded: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, object], ...]:
    root = prepared.definition.repo_root
    relative_records: list[dict[str, object]] = []
    for canonical_name, absolute_path, digest in loaded:
        path = Path(absolute_path)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise RuntimeError("declared loader returned a nonexact absolute source path")
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("declared loader returned a source outside repo_root") from error
        relative_records.append(
            {
                "canonical_name": canonical_name,
                "relative_path": relative_path,
                "sha256": digest,
            }
        )
    records = tuple(relative_records)
    if records != prepared.closure_records:
        raise RuntimeError("canonical relative execution closure differs from its preflight pin")
    observed_sha256 = cast(str, prepared.ledger.canonical_json_sha256(list(records)))
    if observed_sha256 != prepared.execution_source_closure_sha256:
        raise RuntimeError("canonical relative execution closure digest changed")
    return records


def _validate_integrity(prepared: _PreparedCampaign, finder: object) -> None:
    definition = prepared.definition
    if _sha256_file(_bootstrap_source_path()) != prepared.bootstrap_sha256:
        raise RuntimeError("bootstrap source changed during the attempt")
    ledger_path = _resolved_relative(
        definition.repo_root,
        definition.ledger_primitive_relative_path,
        kind="ledger primitive",
    )
    loader_path = _resolved_relative(
        definition.repo_root,
        definition.declared_loader_relative_path,
        kind="declared loader",
    )
    if _sha256_file(ledger_path) != prepared.ledger_primitive_sha256:
        raise RuntimeError("ledger primitive changed during the attempt")
    if _sha256_file(loader_path) != prepared.declared_loader_sha256:
        raise RuntimeError("declared loader changed during the attempt")
    if _source_records(definition.repo_root, definition.closure) != prepared.closure_records:
        raise RuntimeError("declared source bytes changed during the attempt")
    loaded = prepared.declared_loader.validate_loaded_closure(finder)
    _canonicalize_loaded_closure(prepared, loaded)


def _remove_declared_namespace(
    definition: BootstrapCampaign, namespace_modules: Sequence[ModuleType]
) -> None:
    for canonical_name, _ in reversed(definition.closure):
        sys.modules.pop(canonical_name, None)
    for module in reversed(tuple(namespace_modules)):
        if sys.modules.get(module.__name__) is module:
            del sys.modules[module.__name__]


def _progress_snapshot_config(progress: Any | None) -> dict[str, object]:
    if progress is None:
        return {"available": False, "panel_completed": False}
    try:
        snapshot = progress.snapshot()
        if callable(getattr(snapshot, "to_config", None)):
            raw = snapshot.to_config()
        elif dataclasses.is_dataclass(snapshot):
            raw = dataclasses.asdict(cast(Any, snapshot))
        else:
            raise TypeError("progress snapshot has no strict configuration surface")
        if type(raw) is not dict or any(type(key) is not str for key in raw):
            raise TypeError("progress snapshot configuration must be an exact string-keyed dict")
        config = cast(dict[str, object], raw)
        if type(config.get("panel_completed")) is not bool:
            raise TypeError("progress snapshot lacks an exact panel_completed flag")
        return {"available": True, **config}
    except BaseException as snapshot_error:
        return {
            "available": False,
            "panel_completed": False,
            "snapshot_failure_type": type(snapshot_error).__name__,
            "snapshot_failure_message": str(snapshot_error),
        }


def _failure_digest(
    prepared: _PreparedCampaign,
    error: BaseException,
    *,
    stage: str,
    progress_snapshot: Mapping[str, object],
) -> str:
    panel_completed = progress_snapshot.get("panel_completed")
    if type(panel_completed) is not bool:
        raise TypeError("failure progress snapshot lacks exact panel completion")
    body: dict[str, object] = {
        "schema": FAILURE_SCHEMA,
        "campaign": prepared.definition.campaign,
        "development_root": prepared.definition.development_root,
        "development_root_hex": prepared.definition.development_root_hex,
        "bootstrap_stage": stage,
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "panel_completed": panel_completed,
        "progress": dict(progress_snapshot),
        "attempts_consumed": 1,
        "retry_or_recovery_authorized": False,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
        "experiment_output_writes_allowed": False,
    }
    return cast(str, prepared.ledger.canonical_json_sha256(body))


def _record_failed_terminal(
    prepared: _PreparedCampaign,
    ledger_directory: Path,
    error: BaseException,
    *,
    stage: str,
    progress: Any | None,
) -> None:
    progress_snapshot = _progress_snapshot_config(progress)
    panel_completed = progress_snapshot["panel_completed"]
    if type(panel_completed) is not bool:
        raise TypeError("failure progress snapshot lacks exact panel completion")
    failure_sha256 = _failure_digest(
        prepared,
        error,
        stage=stage,
        progress_snapshot=progress_snapshot,
    )
    prepared.ledger.record_terminal(
        ledger_directory,
        prepared.ledger_spec,
        status="failed",
        panel_completed=panel_completed,
        failure_sha256=failure_sha256,
    )
    prepared.ledger.validate_terminal(
        ledger_directory,
        prepared.ledger_spec,
        status="failed",
        panel_completed=panel_completed,
        failure_sha256=failure_sha256,
    )


class CampaignBootstrap:
    """Single in-process owner of the loader's sealed ``ProcessAttempt``."""

    def __init__(self, definition: BootstrapCampaign, *, acknowledgement: str) -> None:
        if type(definition) is not BootstrapCampaign:
            raise TypeError("definition must be an exact BootstrapCampaign")
        if type(acknowledgement) is not str:
            raise TypeError("acknowledgement must be an exact string")
        _reject_preloaded_runtime()
        self._prepared = _prepare_campaign(definition)
        self._acknowledgement = acknowledgement
        self._attempt: Any | None = None

        def builder(capability: object) -> BootstrapOutcome:
            return self._execute(capability)

        self._attempt = self._prepared.declared_loader.ProcessAttempt(builder)

    def run(self) -> BootstrapOutcome:
        """Run once or return the one cached in-process success."""

        attempt = self._attempt
        if attempt is None:
            raise RuntimeError("bootstrap process attempt is unavailable")
        return cast(BootstrapOutcome, attempt.get())

    def completed_value(self) -> BootstrapOutcome | None:
        attempt = self._attempt
        if attempt is None:
            return None
        return cast(BootstrapOutcome | None, attempt.completed_value())

    def _execute(self, capability: object) -> BootstrapOutcome:
        prepared = self._prepared
        definition = prepared.definition
        attempt = self._attempt
        if attempt is None or not attempt.authorizes(capability):
            raise PermissionError("bootstrap does not own the active process capability")
        ledger_directory = _validated_ledger_directory(definition)
        stage = "genesis-preflight"
        progress: Any | None = None
        consumption_invoked = False
        consumed = False
        namespace_modules: tuple[ModuleType, ...] = ()
        finder: Any | None = None
        try:
            if self._acknowledgement != definition.acknowledgement:
                raise PermissionError("execution acknowledgement is not exact")
            prepared.ledger.validate_genesis(ledger_directory, prepared.ledger_spec)
            if prepared.ledger.inspect_state(ledger_directory) != "issued-unused":
                raise RuntimeError("the one-shot ledger is not issued-unused")

            stage = "consume-attempt"
            consumption_invoked = True
            started = cast(
                dict[str, object],
                prepared.ledger.consume_attempt(
                    ledger_directory,
                    prepared.ledger_spec,
                    acknowledgement=self._acknowledgement,
                ),
            )
            consumed = True
            prepared.ledger.validate_started(ledger_directory, prepared.ledger_spec)

            stage = "install-declared-namespace"
            namespace_modules = cast(
                tuple[ModuleType, ...],
                prepared.declared_loader.install_namespace_stubs(definition.repo_root),
            )
            bindings = tuple(
                prepared.declared_loader.DeclaredModuleBinding(
                    canonical_name=cast(str, record["canonical_name"]),
                    relative_path=cast(str, record["relative_path"]),
                    sha256=cast(str, record["sha256"]),
                )
                for record in prepared.closure_records
            )
            finder = prepared.declared_loader.DeclaredSourceFinder(definition.repo_root, bindings)
            stage = "load-declared-evaluator"
            with finder:
                evaluator = finder.load(definition.evaluator_module)
                report_gate = sys.modules[definition.report_gate_module]
                expected_bindings = _bindings_from_prepared(prepared, report_gate, started)
                progress = evaluator.V3EvaluatorProgress()
                authorization_index = 0

                def authorizer(
                    supplied_capability: object,
                    supplied_stage: str,
                    supplied_bindings: object,
                ) -> bool:
                    nonlocal authorization_index
                    if (
                        not attempt.authorizes(supplied_capability)
                        or supplied_capability is not capability
                        or supplied_bindings is not expected_bindings
                        or not sys.meta_path
                        or sys.meta_path[0] is not finder
                        or authorization_index >= len(definition.authorization_stages)
                        or supplied_stage != definition.authorization_stages[authorization_index]
                    ):
                        return False
                    if supplied_stage == "closure-postflight":
                        _validate_integrity(prepared, finder)
                    authorization_index += 1
                    return True

                stage = "evaluate-operational-panel"
                result = evaluator.evaluate_v3_operational_panel(
                    attempt_capability=capability,
                    attempt_authorizer=authorizer,
                    expected_bindings=expected_bindings,
                    progress=progress,
                )
                if authorization_index != len(definition.authorization_stages):
                    raise RuntimeError("evaluator did not consume the exact authorization sequence")
                report_sha256 = _sha256(result.report_sha256, "report_sha256")
                panel_completed = result.progress.panel_completed
                if type(panel_completed) is not bool or not panel_completed:
                    raise RuntimeError("evaluator returned without a completed panel")

                # Validate and freeze the only descriptive result before the
                # irreversible completed terminal is allowed to exist.  The
                # terminal prediction is pure and contains only the report
                # digest, so the ledger remains a hash-only authority record.
                stage = "result-observability-preflight"
                expected_terminal = cast(
                    dict[str, object],
                    prepared.ledger.terminal_record(
                        prepared.ledger_spec,
                        status="completed",
                        panel_completed=True,
                        report_sha256=report_sha256,
                        failure_sha256=None,
                    ),
                )
                outcome = BootstrapOutcome(
                    campaign=definition.campaign,
                    panel_completed=True,
                    canonical_report_json=result.canonical_report_json,
                    report_sha256=report_sha256,
                    terminal_sha256=_sha256(
                        expected_terminal.get("terminal_sha256"),
                        "terminal_sha256",
                    ),
                    execution_source_closure_sha256=(
                        prepared.execution_source_closure_sha256
                    ),
                )
                _canonical_json(outcome.to_config())

                # This is intentionally the final first-party integrity
                # operation.  Only cleanup and anonymous ledger operations
                # follow it.
                stage = "final-execution-closure"
                _validate_integrity(prepared, finder)

            _remove_declared_namespace(definition, namespace_modules)
            namespace_modules = ()
            stage = "record-completed-terminal"
            terminal = cast(
                dict[str, object],
                prepared.ledger.record_terminal(
                    ledger_directory,
                    prepared.ledger_spec,
                    status="completed",
                    panel_completed=True,
                    report_sha256=report_sha256,
                ),
            )
            if terminal != expected_terminal:
                raise RuntimeError("completed terminal differs from its prevalidated record")
            prepared.ledger.validate_terminal(
                ledger_directory,
                prepared.ledger_spec,
                status="completed",
                panel_completed=True,
                report_sha256=report_sha256,
            )
            return outcome
        except BaseException as error:
            if finder is not None:
                try:
                    finder.uninstall()
                except BaseException as cleanup_error:
                    error.add_note(f"declared finder cleanup failed: {cleanup_error}")
            if namespace_modules:
                try:
                    _remove_declared_namespace(definition, namespace_modules)
                except BaseException as cleanup_error:
                    error.add_note(f"declared namespace cleanup failed: {cleanup_error}")
            if consumption_invoked and not consumed:
                try:
                    prepared.ledger.validate_started(ledger_directory, prepared.ledger_spec)
                except BaseException:
                    pass
                else:
                    consumed = True
            if consumed:
                try:
                    _record_failed_terminal(
                        prepared,
                        ledger_directory,
                        error,
                        stage=stage,
                        progress=progress,
                    )
                except BaseException as terminal_error:
                    error.add_note(f"failed terminal sealing also failed: {terminal_error}")
            raise


def execute_campaign(definition: BootstrapCampaign, *, acknowledgement: str) -> BootstrapOutcome:
    """Construct the one bootstrap owner and enter its sealed attempt once."""

    return CampaignBootstrap(definition, acknowledgement=acknowledgement).run()


def production_campaign() -> BootstrapCampaign:
    """Build the exact production definition from the executing file path."""

    bootstrap_path = _bootstrap_source_path()
    repo_root = bootstrap_path.parents[2]
    root = _validated_root(repo_root)
    closure = tuple(zip(PRODUCTION_CLOSURE_NAMES, PRODUCTION_CLOSURE_PATHS, strict=True))
    if len(closure) != 19:
        raise RuntimeError("production closure must contain exactly 19 modules")
    return BootstrapCampaign(
        campaign=PRODUCTION_CAMPAIGN,
        development_root=PRODUCTION_DEVELOPMENT_ROOT,
        development_root_hex=PRODUCTION_DEVELOPMENT_ROOT_HEX,
        protocol_config_sha256=PRODUCTION_PROTOCOL_CONFIG_SHA256,
        control_protocol_config_sha256=PRODUCTION_CONTROL_PROTOCOL_CONFIG_SHA256,
        runtime_config_sha256=PRODUCTION_RUNTIME_CONFIG_SHA256,
        consumed_history_sha256=PRODUCTION_CONSUMED_HISTORY_SHA256,
        key_manifest_sha256=PRODUCTION_KEY_MANIFEST_SHA256,
        stream_sha256=PRODUCTION_STREAM_SHA256,
        cadence_bound_stream_sha256=PRODUCTION_CADENCE_BOUND_STREAM_SHA256,
        source_envelope_sha256=PRODUCTION_SOURCE_ENVELOPE_SHA256,
        acknowledgement=PRODUCTION_ACKNOWLEDGEMENT,
        repo_root=root,
        ledger_relative_path=PRODUCTION_LEDGER_RELATIVE_PATH,
        ledger_primitive_relative_path=PRODUCTION_LEDGER_PRIMITIVE_RELATIVE_PATH,
        declared_loader_relative_path=PRODUCTION_DECLARED_LOADER_RELATIVE_PATH,
        closure=closure,
        evaluator_module=PRODUCTION_EVALUATOR_MODULE,
        report_gate_module=PRODUCTION_REPORT_GATE_MODULE,
        authorization_stages=PRODUCTION_AUTHORIZATION_STAGES,
    )


def _require_exact_file_path_cli() -> None:
    if __spec__ is not None or __package__ not in {None, ""}:
        raise RuntimeError("bootstrap CLI must be invoked by its exact file path")
    frame = sys._getframe()
    try:
        while frame is not None:
            if frame.f_globals.get("__name__") == "runpy":
                raise RuntimeError("bootstrap CLI cannot be entered through runpy")
            frame = frame.f_back
    finally:
        del frame
    main_module = sys.modules.get("__main__")
    if main_module is None or vars(main_module) is not globals():
        raise RuntimeError("bootstrap CLI cannot be entered through runpy or exec")
    argv_path = Path(sys.argv[0])
    source_path = _bootstrap_source_path()
    if (
        not argv_path.is_absolute()
        or argv_path.is_symlink()
        or argv_path.resolve(strict=True) != source_path
        or argv_path != source_path
    ):
        raise RuntimeError("bootstrap CLI must be invoked by its exact absolute file path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe or consume the sole nonpromoting v3 development attempt."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe-spec", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledgement")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact production CLI; generic campaigns are never CLI-selectable."""

    _require_exact_file_path_cli()
    arguments = _parser().parse_args(argv)
    definition = production_campaign()
    if arguments.describe_spec:
        if arguments.acknowledgement is not None:
            raise SystemExit("--describe-spec does not accept --acknowledgement")
        description = describe_campaign(definition)
        print(json.dumps(description, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        return 0
    if arguments.acknowledgement != PRODUCTION_ACKNOWLEDGEMENT:
        raise SystemExit("--execute requires the exact production acknowledgement")
    outcome = execute_campaign(definition, acknowledgement=cast(str, arguments.acknowledgement))
    print(json.dumps(outcome.to_config(), sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapCampaign",
    "BootstrapOutcome",
    "CampaignBootstrap",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXPERIMENT_OUTPUT_WRITES_ALLOWED",
    "GENESIS_CREATION_AUTHORIZED",
    "LEDGER_WRITES_ONLY",
    "PRODUCTION_ACKNOWLEDGEMENT",
    "PRODUCTION_CAMPAIGN",
    "PRODUCTION_CLOSURE_NAMES",
    "PRODUCTION_CLOSURE_PATHS",
    "PRODUCTION_DEVELOPMENT_ROOT",
    "PRODUCTION_DEVELOPMENT_ROOT_HEX",
    "PRODUCTION_LEDGER_RELATIVE_PATH",
    "RETRY_OR_RECOVERY_AUTHORIZED",
    "ROOT_ISSUANCE_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "describe_campaign",
    "execute_campaign",
    "production_campaign",
]
