"""Contracts for the inert cadence-separated future-utility protocol."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "alberta_framework/evaluation/compositional_future_utility_calibration_v3_protocol.py"
)
HISTORICAL_ARM_SOURCE = (
    ROOT / "alberta_framework/evaluation/compositional_future_utility_calibration_v2_development.py"
)


def _load_protocol() -> ModuleType:
    name = "_compositional_future_utility_calibration_v3_protocol_test"
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("protocol module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assigned_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _assigned_value(node: ast.stmt) -> ast.expr | None:
    if isinstance(node, ast.AnnAssign):
        return node.value
    if isinstance(node, ast.Assign):
        return node.value
    return None


def _resolve_literal(node: ast.expr, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_resolve_literal(item, values) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _resolve_literal(key, values): _resolve_literal(value, values)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Subscript):
        collection = _resolve_literal(node.value, values)
        index = _resolve_literal(node.slice, values)
        return collection[index]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MappingProxyType"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _resolve_literal(node.args[0], values)
    raise ValueError(f"nonliteral historical expression: {ast.dump(node)}")


def _historical_arm_literals() -> dict[str, Any]:
    tree = ast.parse(HISTORICAL_ARM_SOURCE.read_text(encoding="utf-8"))
    wanted = {"LONG_TRACE_DECAY", "LONG_TRACE_DECAY_F32_BITS", "ARM_NAMES"}
    wanted |= {"_ARM_PARAMETERS", "_ARM_ROLES"}
    values: dict[str, Any] = {}
    remaining = set(wanted)
    while remaining:
        progressed = False
        for node in tree.body:
            name = _assigned_name(node)
            value = _assigned_value(node)
            if name not in remaining or value is None:
                continue
            try:
                values[name] = _resolve_literal(value, values)
            except (KeyError, ValueError):
                continue
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise AssertionError(f"could not resolve historical literals: {sorted(remaining)}")
    return values


def _nested_dicts(value: object) -> Iterator[dict[str, object]]:
    if type(value) is dict:
        mapping = value
        yield mapping
        for nested in mapping.values():
            yield from _nested_dicts(nested)
    elif type(value) is list:
        for nested in value:
            yield from _nested_dicts(nested)


def test_identity_derives_digest_root_and_hex_without_external_state() -> None:
    protocol = _load_protocol()

    assert protocol.PROTOCOL_NAMESPACE == (
        "alberta.compositional-future-utility-calibration-v3-cadence-separated"
    )
    assert protocol.derive_namespace_sha256(protocol.PROTOCOL_NAMESPACE) == (
        "12efd48b6159117b40a887ccbc2fad0a37a72b045746198999942321242766a2"
    )
    assert hashlib.sha256(protocol.PROTOCOL_NAMESPACE.encode("ascii")).hexdigest() == (
        protocol.PROTOCOL_NAMESPACE_SHA256
    )
    assert protocol.derive_root_from_namespace_sha256(protocol.PROTOCOL_NAMESPACE_SHA256) == (
        317_707_403
    )
    assert protocol.DEVELOPMENT_ROOT == 0x12EFD48B == 317_707_403
    assert protocol.format_root_hex(protocol.DEVELOPMENT_ROOT) == "0x12EFD48B"
    assert protocol.DEVELOPMENT_ROOT_HEX == "0x12EFD48B"


def test_schedule_geometry_residues_and_opportunities_reconstruct_exactly() -> None:
    protocol = _load_protocol()

    assert protocol.PHASE_ORDER == ("A", "B", "A", "D", "A", "C", "A", "B", "C", "A")
    assert protocol.PHASE_LENGTHS == (773, 811, 839, 877, 907, 937, 967, 999, 1020, 868)
    assert (
        protocol.reconstruct_phase_boundaries(protocol.PHASE_LENGTHS)
        == (
            0,
            773,
            1584,
            2423,
            3300,
            4207,
            5144,
            6111,
            7110,
            8130,
            8998,
        )
        == protocol.PHASE_BOUNDARIES
    )
    assert protocol.reconstruct_boundary_residues(
        protocol.PHASE_BOUNDARIES, protocol.CURATION_INTERVAL
    ) == (0, 5, 16, 23, 4, 15, 24, 31, 6, 2, 6)
    assert protocol.reconstruct_curation_opportunities(
        protocol.PHASE_BOUNDARIES, protocol.CURATION_INTERVAL
    ) == (24, 25, 26, 28, 28, 29, 30, 32, 32, 27)
    assert protocol.CURATION_OPPORTUNITIES_PER_PHASE == (
        24,
        25,
        26,
        28,
        28,
        29,
        30,
        32,
        32,
        27,
    )
    assert protocol.TOTAL_STEPS == sum(protocol.PHASE_LENGTHS) == 8_998
    assert protocol.TOTAL_CURATION_OPPORTUNITIES == 281


def test_new_root_does_not_collide_with_explicitly_declared_consumed_roots() -> None:
    protocol = _load_protocol()

    assert protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS == (
        329_631_721,
        1_924_178_934,
    )
    assert protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES == (
        "0x13A5C7E9",
        "0x72B0A3F6",
    )
    assert protocol.DEVELOPMENT_ROOT not in protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS
    assert len(set(protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS)) == 2


@pytest.mark.parametrize(
    ("constant_name", "index", "replacement"),
    (
        ("DECLARED_CONSUMED_DEVELOPMENT_ROOTS", 0, 329_631_722),
        ("DECLARED_CONSUMED_DEVELOPMENT_ROOTS", 1, 1_924_178_935),
        ("DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES", 0, "0x13A5C7EA"),
        ("DECLARED_CONSUMED_DEVELOPMENT_ROOT_HEXES", 1, "0x72B0A3F7"),
    ),
)
def test_consumed_root_constants_reject_each_historical_identity_mutation(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    index: int,
    replacement: int | str,
) -> None:
    protocol = _load_protocol()
    values = list(getattr(protocol, constant_name))
    values[index] = replacement
    monkeypatch.setattr(protocol, constant_name, tuple(values))

    with pytest.raises(ValueError, match="frozen"):
        protocol.CompositionalFutureUtilityCalibrationV3Protocol()


def test_arm_names_roles_parameters_and_long_decay_match_historical_literals() -> None:
    protocol = _load_protocol()
    historical = _historical_arm_literals()

    assert protocol.ARM_NAMES == historical["ARM_NAMES"]
    assert protocol.LONG_TRACE_DECAY == historical["LONG_TRACE_DECAY"]
    assert protocol.LONG_TRACE_DECAY_F32_BITS == historical["LONG_TRACE_DECAY_F32_BITS"]
    assert struct.pack(">f", protocol.LONG_TRACE_DECAY).hex() == "3f7fcc93"
    assert {
        arm.name: (
            arm.future_utility_mix,
            arm.future_utility_trace_decay,
            arm.future_utility_normalization,
        )
        for arm in protocol.ARMS
    } == historical["_ARM_PARAMETERS"]
    assert {arm.name: arm.role for arm in protocol.ARMS} == historical["_ARM_ROLES"]


def test_arm_matrix_varies_only_the_three_declared_intervention_fields() -> None:
    protocol = _load_protocol()
    expected_fields = (
        "future_utility_mix",
        "future_utility_trace_decay",
        "future_utility_normalization",
    )

    assert protocol.INTERVENTION_FIELDS == expected_fields
    arm_configs = [arm.to_config() for arm in protocol.ARMS]
    for config in arm_configs:
        assert tuple(config) == ("name", "role", "interventions")
        interventions = config["interventions"]
        assert type(interventions) is dict
        assert tuple(interventions) == expected_fields
    assert protocol.reconstruct_varying_intervention_fields(protocol.ARMS) == expected_fields


def test_source_geometry_and_corrected_common_base_are_frozen_before_keys() -> None:
    protocol = _load_protocol()
    config = protocol.canonical_protocol_config()

    assert config["source_geometry"] == {
        "left_pack_source_arm": "dovetail_coverage_ancestor_headroom_leftpack",
        "epsilon": 0.1,
        "entry_window": 64,
        "tail_window": 64,
        "raw_dim": 6,
        "active_slots": 11,
        "candidate_slots": 8,
        "action_heads": 2,
        "allocated_max_depth": 3,
        "target_names": ["A", "B", "C"],
        "learner_observation_fields": ["raw_rademacher_values"],
        "learner_feedback_fields": ["selected_action_reward"],
        "resets_allowed": False,
        "source_arm_config": {
            "name": "dovetail_coverage_ancestor_headroom_leftpack",
            "role": (
                "matched headroom arm with lowest-index margin-eligible "
                "destination placement"
            ),
            "composed_readout_enabled": True,
            "effective_max_depth": 3,
            "generation_strategy": "dovetail_product_coverage",
            "retention_slow_utility_decay": 0.999,
            "ancestor_utility_backup_decay": 0.95,
            "candidate_novelty_admission_bonus": 1.0,
            "topology_headroom_reserve": True,
            "topology_left_pack_destinations": True,
        },
    }
    common_base = config["corrected_common_base"]
    assert set(common_base) == {
        "invariant_field_count",
        "invariant_fields_sha256",
        "invariant_fields",
    }
    assert common_base["invariant_field_count"] == 56
    invariant_fields = common_base["invariant_fields"]
    assert len(invariant_fields) == 56
    assert not set(protocol.INTERVENTION_FIELDS) & set(invariant_fields)
    assert invariant_fields["type"] == "CompositionalFeatureLearner"
    assert invariant_fields["step_size_output"] == 0.01
    assert invariant_fields["step_size_theta"] == 0.001
    assert invariant_fields["candidate_scoring_mode"] == "legacy"
    assert invariant_fields["candidate_novelty_admission_bonus"] == 0.0
    assert invariant_fields["future_utility_trace_mode"] == "contribution"
    assert invariant_fields["future_utility_normalization_decay"] == 0.99
    assert invariant_fields["future_utility_rare_task_power"] == 0.0
    assert protocol.canonical_json_sha256(invariant_fields) == common_base[
        "invariant_fields_sha256"
    ]

    for arm in protocol.ARMS:
        learner_config = protocol.reconstruct_arm_learner_config(arm)
        assert len(learner_config) == 59
        assert learner_config["future_utility_mix"] == arm.future_utility_mix
        assert learner_config["future_utility_trace_decay"] == (
            arm.future_utility_trace_decay
        )
        assert learner_config["future_utility_normalization"] == (
            arm.future_utility_normalization
        )


def test_task_reward_and_forgetting_control_semantics_are_machine_readable() -> None:
    protocol = _load_protocol()
    task = protocol.canonical_protocol_config()["task_semantics"]

    assert task == {
        "signature_names": ["A", "B", "C", "D", "shared_p45", "obsolete_p12"],
        "signature_raw_indices": [
            [1, 4, 5],
            [2, 4, 5],
            [3, 4, 5],
            [1, 2, 3],
            [4, 5],
            [1, 2],
        ],
        "signature_roles": [
            "recurring_root",
            "recurring_root",
            "recurring_root",
            "one_exposure_obsolete_root",
            "shared_recurring_intermediate",
            "one_exposure_obsolete_intermediate",
        ],
        "phase_target_raw_indices": [
            [1, 4, 5],
            [2, 4, 5],
            [1, 4, 5],
            [1, 2, 3],
            [1, 4, 5],
            [3, 4, 5],
            [1, 4, 5],
            [2, 4, 5],
            [3, 4, 5],
            [1, 4, 5],
        ],
        "observation_values": [-1.0, 1.0],
        "observation_probabilities": [0.5, 0.5],
        "observation_coordinates_independent": True,
        "target_value_operation": "product",
        "action_values": [0, 1],
        "action_reward_multipliers": [-1.0, 1.0],
        "action_sign_equation": "2 * action - 1",
        "executed_reward_equation": "action_reward_multiplier * target_value",
        "greedy_action_rule": "first_argmax_of_composed_full_q",
        "exploration_rule": "epsilon_mask_selects_pinned_uniform_random_action",
        "learner_target_rule": "selected_head_reward_other_head_nan",
        "counterfactual_action_reward_is_learner_visible": False,
        "phase_identity_is_learner_visible": False,
    }


def test_protocol_and_nested_records_are_frozen_and_exact() -> None:
    protocol = _load_protocol()
    declaration = protocol.CompositionalFutureUtilityCalibrationV3Protocol()

    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.development_root = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.arms[0].role = "tampered"
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.lifecycle.entries_consumed = 1
    with pytest.raises(ValueError, match="frozen"):
        protocol.CompositionalFutureUtilityCalibrationV3Protocol(development_root=True)
    with pytest.raises(ValueError, match="strict"):
        protocol.FutureUtilityArm(
            name="x",
            role="x",
            future_utility_mix=1,
            future_utility_trace_decay=0.95,
            future_utility_normalization="none",
        )


def test_canonical_config_roundtrip_and_hash_are_strict_and_stable() -> None:
    protocol = _load_protocol()
    declaration = protocol.CompositionalFutureUtilityCalibrationV3Protocol()
    config = declaration.to_config()

    assert protocol.reconstruct_protocol(config) == declaration
    assert protocol.canonical_protocol_config() == config
    encoded = protocol.canonical_json(config)
    assert json.loads(encoded) == config
    assert protocol.PROTOCOL_CONFIG_SHA256 == (
        "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"
    )
    assert protocol.canonical_json_sha256(config) == protocol.PROTOCOL_CONFIG_SHA256
    assert protocol.protocol_config_sha256(declaration) == protocol.PROTOCOL_CONFIG_SHA256
    assert protocol.reconstruct_protocol(json.loads(encoded)).to_config() == config


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("identity", "development_root"), 317_707_404),
        (
            ("identity", "declared_consumed_development_roots"),
            [329_631_722, 1_924_178_934],
        ),
        (
            ("identity", "declared_consumed_development_root_hexes"),
            ["0x13A5C7EA", "0x72B0A3F6"],
        ),
        (("schedule", "phase_lengths"), [774, 810, 839, 877, 907, 937, 967, 999, 1020, 868]),
        (("arms", 0, "interventions", "future_utility_mix"), 1.0),
        (("corrected_common_base", "invariant_fields", "step_size_output"), 0.02),
        (("task_semantics", "action_reward_multipliers"), [1.0, -1.0]),
        (("lifecycle", "entries_consumed"), 1),
        (("authority", "evidence_authorized"), True),
    ),
)
def test_reconstruction_rejects_any_tamper(
    path: tuple[str | int, ...], replacement: object
) -> None:
    protocol = _load_protocol()
    config = json.loads(protocol.canonical_json(protocol.canonical_protocol_config()))
    target: Any = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement

    with pytest.raises(ValueError, match="canonical frozen protocol"):
        protocol.reconstruct_protocol(config)


def test_reconstruction_rejects_non_json_shape_and_extra_fields() -> None:
    protocol = _load_protocol()
    config = protocol.canonical_protocol_config()
    config["extra"] = False
    with pytest.raises(ValueError, match="canonical frozen protocol"):
        protocol.reconstruct_protocol(config)

    config = protocol.canonical_protocol_config()
    config["schedule"]["phase_order"] = tuple(config["schedule"]["phase_order"])
    with pytest.raises(ValueError, match="strict JSON"):
        protocol.reconstruct_protocol(config)


def test_module_is_pure_stdlib_and_has_no_operational_surface() -> None:
    source = PROTOCOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots |= {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    definitions = {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned = {name for node in tree.body if (name := _assigned_name(node)) is not None}

    assert roots <= set(sys.stdlib_module_names) | {"__future__"}
    assert not ({"jax", "jaxlib", "numpy"} & roots)
    assert "v2" not in source.casefold()
    assert "calibration_v2" not in source
    assert "calibration-v2" not in source
    assert not any("runner" in name or "latch" in name for name in definitions)
    assert not any(
        name.startswith(("run", "execute", "issue", "consume", "enter")) for name in definitions
    )
    assert not any(name.startswith(("KEY", "STREAM")) for name in assigned)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"open", "compile", "exec", "eval"}
        for node in ast.walk(tree)
    )


def test_issued_unused_lifecycle_and_no_authority_flags_are_fail_closed() -> None:
    protocol = _load_protocol()
    declaration = protocol.CompositionalFutureUtilityCalibrationV3Protocol()
    lifecycle = declaration.lifecycle
    authority = declaration.authority

    assert lifecycle.state == "issued-unused"
    assert lifecycle.maximum_operational_entries == 1
    assert lifecycle.entries_consumed == 0
    assert lifecycle.entries_remaining == 1
    assert not lifecycle.entry_capability_provided
    assert authority.development_only
    assert not authority.panel_executed
    assert not authority.result_available
    assert not authority.execution_authorized_by_protocol
    assert not authority.output_writes_allowed
    assert not authority.artifact_authorized
    assert not authority.threshold_authorized
    assert not authority.winner_or_default_selection_allowed
    assert not authority.search_or_tuning_allowed
    assert not authority.retry_allowed
    assert not authority.recovery_allowed
    assert not authority.evidence_authorized
    assert not authority.scientific_promotion_allowed
    assert not protocol.PANEL_EXECUTED
    assert not protocol.RESULT_AVAILABLE
    assert not any(
        callable(value)
        and any(token in name.casefold() for token in ("runner", "execute", "entry_capability"))
        for name, value in vars(protocol).items()
    )


def test_every_nested_config_mapping_has_only_declared_keys() -> None:
    protocol = _load_protocol()
    config = protocol.canonical_protocol_config()
    mappings = tuple(_nested_dicts(config))

    assert mappings
    assert set(config) == {
        "schema",
        "identity",
        "schedule",
        "source_geometry",
        "corrected_common_base",
        "task_semantics",
        "arms",
        "lifecycle",
        "authority",
    }
    assert all(type(mapping) is dict for mapping in mappings)
