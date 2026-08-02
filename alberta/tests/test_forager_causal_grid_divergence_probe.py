from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import os
import secrets
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import forager_causal_grid_divergence_probe as probe

pytestmark = pytest.mark.unit


def _actions_by_candidate() -> dict[str, bytes]:
    first = bytes(probe.FIXED_STEPS)
    return {
        "causal_e050_q050": first,
        "causal_e050_q075": first,
        "causal_e050_q090": bytes(2_000) + bytes([1]) * 8_000,
    }


def _candidate_record(
    candidate_id: str,
    quantile: float,
    configuration_sha256: str,
    actions: bytes,
    *,
    delay: int,
) -> dict[str, Any]:
    action_tree = probe._build_action_merkle_tree(actions)
    return {
        "action_count": probe.FIXED_STEPS,
        "action_trace_encoding": probe.ACTION_MERKLE_ENCODING,
        "action_trace_sha256": probe._action_merkle_root_sha256(action_tree),
        "candidate_id": candidate_id,
        "configuration_sha256": configuration_sha256,
        "per_channel_diagnostics": [
            {
                "channel_index": channel,
                "exact_count_final": channel,
                "estimated_delay": {
                    "change_count": 0,
                    "final": delay,
                    "maximum": delay,
                    "mean_hex": float(delay).hex(),
                    "minimum": delay,
                    "sample_count": probe.FIXED_STEPS,
                    "sum": delay * probe.FIXED_STEPS,
                },
            }
            for channel in range(3)
        ],
        "respawn_safety_quantile": quantile,
    }


def _pair_record(
    left_id: str,
    right_id: str,
    left_actions: bytes,
    right_actions: bytes,
) -> dict[str, Any]:
    divergence_index = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(left_actions, right_actions, strict=True)
            )
            if left != right
        ),
        None,
    )
    prefix_count = (
        probe.FIXED_STEPS if divergence_index is None else divergence_index
    )
    assert left_actions[:prefix_count] == right_actions[:prefix_count]
    return {
        "first_action_divergence_step": (
            None if divergence_index is None else divergence_index + 1
        ),
        "first_divergence_merkle_proof": (
            None
            if divergence_index is None
            else probe._first_divergence_merkle_proof(
                probe._build_action_merkle_tree(left_actions),
                probe._build_action_merkle_tree(right_actions),
                divergence_index=divergence_index,
                left_action=left_actions[divergence_index],
                right_action=right_actions[divergence_index],
            )
        ),
        "left_candidate_id": left_id,
        "right_candidate_id": right_id,
    }


def _child_payload(*, divergent: bool = True) -> dict[str, Any]:
    actions = _actions_by_candidate()
    if not divergent:
        actions = {candidate_id: bytes(probe.FIXED_STEPS) for candidate_id in actions}
    candidates = [
        _candidate_record(
            candidate_id,
            quantile,
            configuration_sha256,
            actions[candidate_id],
            delay=10 + index,
        )
        for index, (candidate_id, quantile, configuration_sha256) in enumerate(
            probe._CANDIDATE_CONFIGURATIONS
        )
    ]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(probe._CANDIDATE_CONFIGURATIONS):
        for right in probe._CANDIDATE_CONFIGURATIONS[left_index + 1 :]:
            pairs.append(
                _pair_record(
                    left[0],
                    right[0],
                    actions[left[0]],
                    actions[right[0]],
                )
            )
    return {
        "schema_version": probe.CHILD_SCHEMA_VERSION,
        "candidates": candidates,
        "pairwise_action_divergence": pairs,
        "runtime_observation": {
            "device_platforms": ["cpu"],
            "jax_backend": "cpu",
            "jax_enable_x64": False,
            "jax_version": "0.9.0.1",
            "jaxlib_version": "0.9.0.1",
            "numpy_version": "2.4.1",
            "python_version": "3.12.3",
            "threefry_partitionable": True,
        },
    }


def _execution_envelope() -> dict[str, Any]:
    return {
        "image_sha256": probe.QUALIFIED_IMAGE_SHA256,
        "kind": "sha256_pinned_qualified_oci",
        "network": "none",
        "oci_runtime_executable_sha256": "d" * 64,
        "probe_mount": "private_exact_readonly_snapshot_v1",
        "qualification_mount": "minimal_exact_readable_snapshot_v2",
        "qualified_image_executed": True,
        "root_filesystem": "read_only",
        "source_mount": "pinned_archive_extracted_in_private_tmpfs_v1",
    }


def _private_probe_mount(tmp_path: Path) -> Path:
    path = tmp_path / "private-probe.py"
    path.write_bytes(Path(probe.__file__).read_bytes())
    path.chmod(0o444)
    return path


def _all_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_mapping_keys(child))
    return keys


def test_contract_is_fixed_to_public_seed_zero_and_ten_thousand_steps() -> None:
    assert probe.SCHEMA_VERSION == "alberta.forager_causal_grid_divergence_probe.v2"
    assert probe.DEFAULT_OUTPUT_ROOT.name == "causal_q_grid_divergence_seed0_v2"
    assert probe.FIXED_SEED == 0
    assert probe.FIXED_EXPLORATION_PROBABILITY == 0.05
    assert probe.FIXED_STEPS == 10_000
    assert probe.ACTION_MERKLE_TREE_LEAF_COUNT == 16_384
    assert probe.ACTION_MERKLE_ROOT_LEVEL == 14
    assert [item[:2] for item in probe._CANDIDATE_CONFIGURATIONS] == [
        ("causal_e050_q050", 0.5),
        ("causal_e050_q075", 0.75),
        ("causal_e050_q090", 0.9),
    ]
    run_probe_parameters = {
        parameter.name
        for parameter in inspect.signature(probe.run_probe).parameters.values()
    }
    assert run_probe_parameters == {
        "qualification_root",
        "source_root",
        "output_root",
        "oci_runtime",
    }
    parser_destinations = {
        action.dest for action in probe._public_parser()._actions  # noqa: SLF001
    }
    assert parser_destinations == {
        "help",
        "output_root",
        "qualification_root",
        "source_root",
    }
    for forbidden_arguments in (("--steps", "2000"), ("--seed", "1")):
        with pytest.raises(SystemExit):
            probe._public_parser().parse_args(forbidden_arguments)


def test_harness_is_outside_registered_alberta_source_tree() -> None:
    harness_path = Path(probe.__file__).resolve()
    repository_root = Path(__file__).resolve().parents[1]

    assert harness_path.parent == repository_root
    assert not (repository_root / "alberta_framework" / "benchmarks" / harness_path.name).exists()


def test_frozen_source_configuration_runtime_and_task_bindings_are_exact() -> None:
    configurations, manifest = probe._load_bound_inputs(
        probe.DEFAULT_QUALIFICATION_ROOT,
        probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "source",
    )

    assert manifest["open_protocol_sha256"] == probe.OPEN_PROTOCOL_SHA256
    assert manifest["sources"]["alberta"]["archive"]["sha256"] == (
        probe.SOURCE_ARCHIVE_SHA256
    )
    assert manifest["runtime_qualification"]["image_sha256"] == (
        probe.QUALIFIED_IMAGE_SHA256
    )
    bodies = [configuration["configuration"] for configuration in configurations]
    assert [body["respawn_safety_quantile"] for body in bodies] == [0.5, 0.75, 0.9]
    without_q = []
    for body in bodies:
        comparison = dict(body)
        comparison.pop("respawn_safety_quantile")
        comparison.pop("respawn_quantile_z")
        without_q.append(comparison)
    assert without_q[0] == without_q[1] == without_q[2]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("schema_version", "wrong.schema.v1"),
        lambda value: value["sources"]["alberta"].__setitem__(
            "root", "sources/alberta/elsewhere"
        ),
        lambda value: value["sources"]["alberta"].__setitem__(
            "snapshot_descriptor_path", "sources/alberta/wrong.json"
        ),
        lambda value: value["sources"]["alberta"].__setitem__(
            "patch_path", "sources/alberta/unexpected.patch"
        ),
        lambda value: value["sources"]["alberta"].__setitem__(
            "unexpected", False
        ),
        lambda value: value.__setitem__("unexpected", False),
    ),
)
def test_qualification_manifest_contract_rejects_schema_path_and_field_drift(
    mutate: Any,
) -> None:
    manifest = json.loads(
        (probe.DEFAULT_QUALIFICATION_ROOT / "manifest.json").read_bytes()
    )
    mutate(manifest)
    with pytest.raises(probe.CausalGridDivergenceProbeError):
        probe._qualification_manifest_bindings(manifest)  # noqa: SLF001


def test_manifest_bound_source_root_rejects_siblings_symlinks_and_relative_paths(
    tmp_path: Path,
) -> None:
    expected = probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "source"
    probe._require_manifest_source_root(  # noqa: SLF001
        probe.DEFAULT_QUALIFICATION_ROOT,
        expected,
    )
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    symlink = tmp_path / "source-link"
    symlink.symlink_to(expected, target_is_directory=True)
    relative = Path(os.path.relpath(expected, Path.cwd()))
    for changed in (sibling, symlink, relative):
        with pytest.raises(
            probe.CausalGridDivergenceProbeError,
            match="source root|absolute canonical",
        ):
            probe._require_manifest_source_root(  # noqa: SLF001
                probe.DEFAULT_QUALIFICATION_ROOT,
                changed,
            )

    qualification_root = tmp_path / "qualification"
    escaped_root = tmp_path / "escaped"
    escaped_source = escaped_root / "alberta" / "source"
    qualification_root.mkdir()
    escaped_source.mkdir(parents=True)
    (qualification_root / "sources").symlink_to(
        escaped_root,
        target_is_directory=True,
    )
    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="source root|canonical|qualification",
    ):
        probe._require_manifest_source_root(  # noqa: SLF001
            qualification_root,
            escaped_source,
        )


@pytest.mark.parametrize(
    "relative_text",
    probe._qualification_mount_relative_paths(),  # noqa: SLF001
)
def test_every_qualification_input_rejects_symlinked_path_components(
    tmp_path: Path,
    relative_text: str,
) -> None:
    qualification_root = tmp_path / "qualification"
    outside_root = tmp_path / "outside"
    qualification_root.mkdir()
    relative = Path(relative_text)
    target = outside_root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"{}")
    if len(relative.parts) == 1:
        (qualification_root / relative.name).symlink_to(target)
    else:
        (qualification_root / relative.parts[0]).symlink_to(
            outside_root / relative.parts[0],
            target_is_directory=True,
        )

    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="canonical|qualification|symlink",
    ):
        probe._qualification_input_path(  # noqa: SLF001
            qualification_root,
            relative_text,
            label="test qualification input",
        )


def test_bound_input_loader_rejects_a_bad_manifest_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read = probe._read_stable_regular_file  # noqa: SLF001

    def read(path: Path, *, label: str, maximum: int) -> bytes:
        if label == "matched qualification manifest digest sidecar":
            return b"0" * 64 + b"\n"
        return real_read(path, label=label, maximum=maximum)

    monkeypatch.setattr(probe, "_read_stable_regular_file", read)
    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="manifest sidecar",
    ):
        probe._load_bound_inputs(  # noqa: SLF001
            probe.DEFAULT_QUALIFICATION_ROOT,
            probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "source",
        )


@pytest.mark.parametrize(
    "case",
    (
        "traversal_path",
        "absolute_path",
        "bad_digest",
        "wrong_schema",
        "extra_field",
        "source_drift",
        "entrypoint_drift",
    ),
)
def test_capability_receipt_contract_rejects_path_schema_and_binding_drift(
    case: str,
) -> None:
    root = probe.DEFAULT_QUALIFICATION_ROOT
    manifest = json.loads((root / "manifest.json").read_bytes())
    candidate_id, _quantile, configuration_sha256 = probe._CANDIDATE_CONFIGURATIONS[0]
    candidate = copy.deepcopy(manifest["candidates"][candidate_id])
    binding = copy.deepcopy(candidate["capability_receipt"])
    payload = json.loads((root / binding["path"]).read_bytes())
    source_binding = manifest["sources"]["alberta"]["binding"]
    if case == "traversal_path":
        binding["path"] = "receipts/../manifest.json"
    elif case == "absolute_path":
        binding["path"] = "/tmp/receipt.json"
    elif case == "bad_digest":
        binding["sha256"] = "not-a-digest"
    elif case == "wrong_schema":
        payload["schema_version"] = "wrong.schema.v1"
    elif case == "extra_field":
        payload["unexpected"] = False
    elif case == "source_drift":
        payload["source"]["archive_sha256"] = "f" * 64
    else:
        payload["entrypoint_path"] = "alberta_framework/wrong.py"
    with pytest.raises(probe.CausalGridDivergenceProbeError):
        probe._validate_capability_receipt(  # noqa: SLF001
            candidate_id=candidate_id,
            expected_configuration_sha256=configuration_sha256,
            capability_binding=binding,
            capability_payload=payload,
            candidate_manifest=candidate,
            source_binding=source_binding,
        )


def test_source_identity_rejects_byte_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "module.py"
    source_file.write_bytes(b"value = 1\n")
    inventory = {
        "schema_version": "alberta.forager_source_inventory.v1",
        "files": probe._source_inventory_records(source_root),
    }
    inventory_raw = probe._canonical_json_bytes(inventory)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(inventory_raw)

    probe._verify_source_identity(
        source_root,
        inventory_path,
        expected_inventory_sha256=hashlib.sha256(inventory_raw).hexdigest(),
    )
    source_file.write_bytes(b"value = 2\n")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="source bytes"):
        probe._verify_source_identity(
            source_root,
            inventory_path,
            expected_inventory_sha256=hashlib.sha256(inventory_raw).hexdigest(),
        )


def test_pinned_source_archive_reconstructs_exact_private_inventory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "private-source"

    probe._extract_pinned_source_archive(
        probe.DEFAULT_QUALIFICATION_ROOT,
        destination,
    )

    expected = probe._load_source_inventory_records(
        probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "inventory.json"
    )
    assert probe._source_inventory_records(destination) == list(expected)
    assert destination != (
        probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "source"
    )
    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="unexpectedly exists",
    ):
        probe._extract_pinned_source_archive(
            probe.DEFAULT_QUALIFICATION_ROOT,
            destination,
        )


def test_qualification_mount_mirror_is_minimal_exact_and_world_readable(
    tmp_path: Path,
) -> None:
    destination_root = tmp_path / "qualification-mount"
    destination_root.mkdir(mode=0o700)

    digests = probe._materialize_readable_qualification_mount(
        probe.DEFAULT_QUALIFICATION_ROOT,
        destination_root,
    )

    expected_paths = probe._qualification_mount_relative_paths()
    assert "manifest.json.sha256" in expected_paths
    assert tuple(digests) == expected_paths
    observed_files = sorted(
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
    )
    assert observed_files == sorted(expected_paths)
    for relative in expected_paths:
        source_raw = (probe.DEFAULT_QUALIFICATION_ROOT / relative).read_bytes()
        copied = destination_root / relative
        assert copied.read_bytes() == source_raw
        assert digests[relative] == hashlib.sha256(source_raw).hexdigest()
        assert copied.stat().st_mode & 0o004
    for directory in (
        destination_root,
        *(path for path in destination_root.rglob("*") if path.is_dir()),
    ):
        assert directory.stat().st_mode & 0o005 == 0o005


def test_run_child_uses_distinct_minimal_mirror_and_removes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    expected_payload = _child_payload()
    expected_envelope = _execution_envelope()

    def fake_run_child_with_mount(
        qualification_root: Path,
        probe_path: Path,
        *,
        oci_runtime: Path,
        expected_probe_source_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observed["qualification_root"] = qualification_root
        observed["probe_path"] = probe_path
        observed["oci_runtime"] = oci_runtime
        observed["expected_probe_source_sha256"] = expected_probe_source_sha256
        observed["files"] = sorted(
            path.relative_to(qualification_root).as_posix()
            for path in qualification_root.rglob("*")
            if path.is_file()
        )
        observed["root_mode"] = qualification_root.stat().st_mode
        observed["probe_mode"] = probe_path.stat().st_mode
        observed["probe_sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        return expected_payload, expected_envelope

    monkeypatch.setattr(
        probe,
        "_run_child_with_qualification_mount",
        fake_run_child_with_mount,
    )
    source_root = probe.DEFAULT_QUALIFICATION_ROOT / "sources" / "alberta" / "source"
    runtime = tmp_path / "docker"
    source_digest = hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()

    returned = probe._run_child(
        source_root,
        probe.DEFAULT_QUALIFICATION_ROOT,
        oci_runtime=runtime,
        expected_probe_source_sha256=source_digest,
    )

    mirror = observed["qualification_root"]
    probe_mirror = observed["probe_path"]
    assert returned == (expected_payload, expected_envelope)
    assert mirror != probe.DEFAULT_QUALIFICATION_ROOT
    assert observed["oci_runtime"] == runtime
    assert observed["expected_probe_source_sha256"] == source_digest
    assert observed["files"] == sorted(probe._qualification_mount_relative_paths())
    assert observed["root_mode"] & 0o005 == 0o005
    assert observed["probe_mode"] & 0o222 == 0
    assert observed["probe_sha256"] == source_digest
    assert not mirror.exists()
    assert not probe_mirror.exists()


def test_shared_environment_key_helper_preserves_exact_object_identity() -> None:
    sentinel = object()

    shared = probe._shared_step_keys(sentinel, 3)

    assert len(shared) == 3
    assert all(item is sentinel for item in shared)
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="positive"):
        probe._shared_step_keys(sentinel, 0)


def test_coupled_lane_helpers_share_reset_and_step_keys_and_emit_pre_step_actions() -> None:
    reset_key = object()
    step_key = object()
    reset_calls: list[Any] = []
    step_calls: list[tuple[Any, int]] = []

    def reset(key: Any, _parameters: Any) -> tuple[str, str]:
        reset_calls.append(key)
        lane = len(reset_calls) - 1
        return f"observation-{lane}", f"environment-{lane}"

    observations, environment_states = probe._coupled_lane_resets(
        reset,
        reset_key,
        "parameters",
        3,
    )

    def environment_step(
        key: Any,
        environment_state: str,
        action: int,
        _parameters: Any,
    ) -> tuple[str, str, int, bool, dict[str, Any]]:
        step_calls.append((key, action))
        return (
            f"next-{environment_state}",
            f"next-{environment_state}",
            action,
            False,
            {"must_not_escape": True},
        )

    def agent_step(
        agent_state: str,
        reward: int,
        _observation: str,
        _configuration: str,
    ) -> tuple[str, int, dict[str, Any]]:
        return f"next-{agent_state}", reward + 100, {"ignored": True}

    result = probe._coupled_lane_step(
        environment_step=environment_step,
        agent_step=agent_step,
        delay_for=lambda _state, configuration: f"delay-{configuration}",
        step_key=step_key,
        environment_parameters="parameters",
        environment_states=environment_states,
        agent_states=("agent-0", "agent-1", "agent-2"),
        actions=(10, 20, 30),
        configurations=("q050", "q075", "q090"),
    )
    next_actions = result[2]
    executed_actions = result[5]

    assert observations == ("observation-0", "observation-1", "observation-2")
    assert all(key is reset_key for key in reset_calls)
    assert all(key is step_key for key, _action in step_calls)
    assert [action for _key, action in step_calls] == [10, 20, 30]
    assert executed_actions == (10, 20, 30)
    assert next_actions == (110, 120, 130)
    assert not set(next_actions) & set(executed_actions)


def test_action_merkle_root_is_indexed_fixed_length_and_action_bound() -> None:
    actions = bytes(probe.FIXED_STEPS)
    changed = bytearray(actions)
    changed[-1] = 3

    root = probe._action_merkle_root_sha256(probe._build_action_merkle_tree(actions))
    assert root == probe._action_merkle_root_sha256(
        probe._build_action_merkle_tree(actions)
    )
    assert root != probe._action_merkle_root_sha256(
        probe._build_action_merkle_tree(bytes(changed))
    )
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="fixed-length"):
        probe._build_action_merkle_tree(actions[:-1])
    invalid = bytearray(actions)
    invalid[0] = 4
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="invalid action"):
        probe._build_action_merkle_tree(bytes(invalid))


def test_divergence_payload_has_one_based_witnesses_and_passes() -> None:
    payload = _child_payload(divergent=True)

    assert probe._validate_child_payload(payload) is True
    assert [
        pair["first_action_divergence_step"]
        for pair in payload["pairwise_action_divergence"]
    ] == [None, 2_001, 2_001]
    receipt, passed = probe._assemble_receipt(
        payload,
        probe_source_sha256="a" * 64,
        execution_envelope=_execution_envelope(),
    )

    assert passed is True
    assert receipt["schema_version"] == probe.SCHEMA_VERSION
    assert receipt["status"] == "action_divergence_observed"
    assert receipt["divergence_summary"] == {
        "all_pairs_diverged": False,
        "any_pair_diverged": True,
        "divergent_pair_count": 2,
        "gate": "at_least_one_pair_must_diverge",
        "pair_count": 3,
    }
    assert receipt["promotion_authorized"] is False
    assert receipt["protocol_relationship"]["retroactive_evidence_gate"] is False
    assert receipt["execution_envelope"]["qualification_mount"] == (
        "minimal_exact_readable_snapshot_v2"
    )
    assert receipt["frozen_inputs"]["qualification_manifest_path"] == "manifest.json"
    assert receipt["frozen_inputs"]["qualification_manifest_sidecar_path"] == (
        "manifest.json.sha256"
    )
    assert receipt["frozen_inputs"]["qualification_manifest_schema_version"] == (
        probe.QUALIFICATION_MANIFEST_SCHEMA_VERSION
    )
    assert receipt["frozen_inputs"]["alberta_source_root_path"] == (
        "sources/alberta/source"
    )
    assert receipt["frozen_inputs"]["capability_receipt_schema_version"] == (
        probe.CAPABILITY_RECEIPT_SCHEMA_VERSION
    )
    disclosed_actions = [
        action
        for pair in receipt["pairwise_action_divergence"]
        if pair["first_divergence_merkle_proof"] is not None
        for action in (
            pair["first_divergence_merkle_proof"]["left_action"],
            pair["first_divergence_merkle_proof"]["right_action"],
        )
    ]
    assert len(disclosed_actions) == 2 * receipt["divergence_summary"][
        "divergent_pair_count"
    ]
    assert all(action in {0, 1, 2, 3} for action in disclosed_actions)
    assert receipt["action_boundary"] == {
        "action_arrays_emitted": False,
        "action_arrays_persisted": False,
        "commitment": probe.ACTION_MERKLE_ENCODING,
        "first_divergence_proof": "canonical_paired_merkle_descent_v1",
        "maximum_disclosed_action_scalars": len(disclosed_actions),
        "scalar_domain": [0, 1, 2, 3],
    }
    protected = {"actions", "observations", "states", "rewards", "regrets", "returns", "scores"}
    assert not (_all_mapping_keys(receipt) & protected)


def test_all_equal_is_valid_rejection_and_main_uses_exit_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _child_payload(divergent=False)
    receipt, passed = probe._assemble_receipt(
        payload,
        probe_source_sha256="a" * 64,
        execution_envelope=_execution_envelope(),
    )

    assert passed is False
    assert receipt["status"] == "rejected_no_action_divergence"
    assert receipt["divergence_summary"]["divergent_pair_count"] == 0
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda **_kwargs: (Path("receipt.json"), receipt, False),
    )
    assert probe.main([]) == 1


def test_malformed_or_unqualified_result_is_exit_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs: Any) -> Any:
        raise probe.CausalGridDivergenceProbeError("injected integrity fault")

    monkeypatch.setattr(probe, "run_probe", fail)

    assert probe.main([]) == 2
    bad_envelope = _execution_envelope()
    bad_envelope["qualified_image_executed"] = False
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="qualified OCI"):
        probe._assemble_receipt(
            _child_payload(),
            probe_source_sha256="a" * 64,
            execution_envelope=bad_envelope,
        )


def test_strict_json_rejects_numeric_overflow_as_a_probe_error() -> None:
    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="non-finite JSON number",
    ):
        probe._decode_strict_json(b'{"value":1e999}', label="overflow fixture")


def test_child_schema_rejects_channel_shape_and_impossible_delay_aggregates() -> None:
    wrong_shape = _child_payload()
    wrong_shape["candidates"][0]["per_channel_diagnostics"].pop()
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="channel count"):
        probe._validate_child_payload(wrong_shape)

    impossible = _child_payload()
    aggregate = impossible["candidates"][0]["per_channel_diagnostics"][0][
        "estimated_delay"
    ]
    aggregate.update(
        {
            "final": 10,
            "maximum": 10,
            "mean_hex": 1.0.hex(),
            "minimum": 10,
            "sum": probe.FIXED_STEPS,
        }
    )
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="inconsistent"):
        probe._validate_child_payload(impossible)

    boolean_index = _child_payload()
    boolean_index["candidates"][0]["per_channel_diagnostics"][0][
        "channel_index"
    ] = False
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="order drifted"):
        probe._validate_child_payload(boolean_index)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["candidates"][0].__setitem__(
            "action_count",
            float(probe.FIXED_STEPS),
        ),
        lambda value: value["candidates"][0]["per_channel_diagnostics"][0][
            "estimated_delay"
        ].__setitem__("sample_count", float(probe.FIXED_STEPS)),
        lambda value: value["pairwise_action_divergence"][1][
            "first_divergence_merkle_proof"
        ].__setitem__(
            "tree_leaf_count",
            float(probe.ACTION_MERKLE_TREE_LEAF_COUNT),
        ),
    ),
)
def test_child_schema_rejects_float_aliases_for_exact_integer_fields(
    mutate: Any,
) -> None:
    payload = _child_payload()
    mutate(payload)

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="integer"):
        probe._validate_child_payload(payload)


def test_child_schema_rejects_protected_or_unknown_array_fields() -> None:
    payload = _child_payload()
    payload["candidates"][0]["actions"] = [0] * probe.FIXED_STEPS

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="fields drifted"):
        probe._validate_child_payload(payload)


def test_output_root_is_canonical_disjoint_and_write_once(tmp_path: Path) -> None:
    qualification_root = tmp_path / "qualification"
    source_root = qualification_root / "source"
    output_root = tmp_path / "development" / "probe-v2"
    qualification_root.mkdir()
    source_root.mkdir()
    receipt = {"schema_version": probe.SCHEMA_VERSION, "status": "test"}

    probe._validate_output_root(
        output_root,
        qualification_root=qualification_root,
        source_root=source_root,
    )
    receipt_path = probe._write_receipt(output_root, receipt)
    raw = probe._canonical_json_bytes(receipt)
    assert receipt_path.read_bytes() == raw
    assert (output_root / "receipt.json.sha256").read_text(encoding="ascii") == (
        f"{hashlib.sha256(raw).hexdigest()}\n"
    )
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="write-once"):
        probe._write_receipt(output_root, receipt)
    assert list(output_root.parent.glob(f".{output_root.name}.staging-*")) == []
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="overlaps"):
        probe._validate_output_root(
            source_root / "unsafe",
            qualification_root=qualification_root,
            source_root=source_root,
        )
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="canonical absolute"):
        probe._validate_output_root(
            Path("relative-output"),
            qualification_root=qualification_root,
            source_root=source_root,
        )


def test_oversized_receipt_is_rejected_before_staging_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "oversized"
    monkeypatch.setattr(probe, "MAXIMUM_JSON_BYTES", 1)

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="before publication"):
        probe._write_receipt(output_root, {"payload": "too-large"})

    assert not output_root.exists()
    assert list(tmp_path.glob(".oversized.staging-*")) == []


def test_oci_child_command_is_digest_pinned_read_only_and_has_no_scientific_knobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    source_root.mkdir()
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)
    observed: dict[str, Any] = {}
    payload = _child_payload()

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> Any:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stderr=b"",
            stdout=probe._canonical_json_bytes(payload),
        )

    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)

    returned, envelope = probe._run_child_with_qualification_mount(
        qualification_root,
        probe_mount,
        oci_runtime=runtime,
        expected_probe_source_sha256=hashlib.sha256(
            Path(probe.__file__).read_bytes()
        ).hexdigest(),
    )

    assert returned == payload
    assert envelope["qualified_image_executed"] is True
    command = observed["command"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--pull=never" in command
    for proxy_variable in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        assert f"--env={proxy_variable}=" in command
    assert f"sha256:{probe.QUALIFIED_IMAGE_SHA256}" in command
    assert not any("destination=/inputs/source" in argument for argument in command)
    assert "/run/alberta/source" in command
    assert "--steps" not in command
    assert "--seed" not in command
    assert observed["timeout_seconds"] == 60 * 60
    assert observed["maximum_stdout_bytes"] == probe.MAXIMUM_JSON_BYTES
    assert observed["maximum_stderr_bytes"] == probe.MAXIMUM_STDERR_BYTES


def test_named_container_cleanup_requires_bounded_exact_name_absence_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "docker"
    environment = {"PATH": "/usr/bin"}
    container_name = "alberta-causal-q-probe-" + "a" * 24
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> Any:
        commands.append(command)
        if len(commands) == 1:
            return subprocess.CompletedProcess(
                command,
                returncode=1,
                stdout=b"",
                stderr=b"Error: No such container",
            )
        assert kwargs["timeout_seconds"] == 120
        assert (
            kwargs["maximum_stdout_bytes"]
            == probe.MAXIMUM_CLEANUP_INSPECTION_BYTES
        )
        assert (
            kwargs["maximum_stderr_bytes"]
            == probe.MAXIMUM_CLEANUP_INSPECTION_BYTES
        )
        return subprocess.CompletedProcess(command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)
    assert probe._cleanup_named_container(runtime, container_name, environment)  # noqa: SLF001
    assert commands == [
        (runtime.as_posix(), "rm", "--force", container_name),
        (
            runtime.as_posix(),
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            f"--filter=name=^/{container_name}$",
        ),
    ]


def test_named_container_cleanup_rejects_a_still_present_exact_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "docker"
    container_name = "alberta-causal-q-probe-" + "b" * 24
    calls = 0

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, b"", b"missing")
        return subprocess.CompletedProcess(command, 0, b"c" * 64 + b"\n", b"")

    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)
    assert not probe._cleanup_named_container(  # noqa: SLF001
        runtime,
        container_name,
        {"PATH": "/usr/bin"},
    )
    assert calls == 2


def test_named_container_cleanup_accepts_one_successful_exact_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-causal-q-probe-" + "c" * 24
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> Any:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)
    assert probe._cleanup_named_container(  # noqa: SLF001
        tmp_path / "docker",
        container_name,
        {"PATH": "/usr/bin"},
    )
    assert len(commands) == 1


def test_named_container_cleanup_fails_closed_if_absence_query_overflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "alberta-causal-q-probe-" + "d" * 24
    calls = 0

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, b"", b"missing")
        raise probe.CausalGridDivergenceProbeError("injected inspection overflow")

    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)
    assert not probe._cleanup_named_container(  # noqa: SLF001
        tmp_path / "docker",
        container_name,
        {"PATH": "/usr/bin"},
    )
    assert calls == 2


@pytest.mark.parametrize(
    "container_name",
    (
        "--force",
        "alberta-causal-q-probe-" + "a" * 23,
        "alberta-causal-q-probe-" + "g" * 24,
    ),
)
def test_named_container_cleanup_rejects_non_internal_names_without_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container_name: str,
) -> None:
    called = False

    def run(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("invalid cleanup names must not reach the OCI runtime")

    monkeypatch.setattr(probe, "_run_bounded_command", run)
    assert not probe._cleanup_named_container(  # noqa: SLF001
        tmp_path / "docker",
        container_name,
        {"PATH": "/usr/bin"},
    )
    assert called is False


def test_oci_timeout_attempts_cleanup_for_the_exact_validated_container_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    source_root.mkdir()
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)
    commands: list[tuple[str, ...]] = []
    cleanup_calls: list[tuple[Path, str]] = []

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> Any:
        commands.append(command)
        raise subprocess.TimeoutExpired(command, 60 * 60)

    def fake_cleanup(
        runtime_path: Path,
        container_name: str,
        _environment: Any,
    ) -> bool:
        cleanup_calls.append((runtime_path, container_name))
        return True

    monkeypatch.setattr(secrets, "token_hex", lambda _count: "b" * 24)
    monkeypatch.setattr(probe, "_run_bounded_command", fake_run)
    monkeypatch.setattr(probe, "_cleanup_named_container", fake_cleanup)

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="cleanup completed"):
        probe._run_child_with_qualification_mount(
            qualification_root,
            probe_mount,
            oci_runtime=runtime,
            expected_probe_source_sha256=hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        )

    container_name = "alberta-causal-q-probe-" + "b" * 24
    assert f"--name={container_name}" in commands[0]
    assert cleanup_calls == [(runtime.resolve(), container_name)]


@pytest.mark.parametrize("failure_kind", ("oserror", "subprocess"))
def test_oci_runner_failures_are_public_after_confirmed_named_container_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)
    cleanup_names: list[str] = []
    failure: BaseException = (
        OSError("injected private runner failure")
        if failure_kind == "oserror"
        else subprocess.SubprocessError("injected private runner failure")
    )

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    def clean(
        _runtime_path: Path,
        container_name: str,
        _environment: Any,
    ) -> bool:
        cleanup_names.append(container_name)
        return True

    monkeypatch.setattr(secrets, "token_hex", lambda _count: "a" * 24)
    monkeypatch.setattr(probe, "_run_bounded_command", fail)
    monkeypatch.setattr(probe, "_cleanup_named_container", clean)

    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="runner failed; named-container cleanup completed",
    ) as caught:
        probe._run_child_with_qualification_mount(
            qualification_root,
            probe_mount,
            oci_runtime=runtime,
            expected_probe_source_sha256=hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        )

    assert caught.value.__cause__ is failure
    assert cleanup_names == ["alberta-causal-q-probe-" + "a" * 24]


def test_completed_oci_failure_confirms_named_container_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)
    cleanup_names: list[str] = []

    def fail(command: tuple[str, ...], **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(command, 17, b"", b"child failed")

    def clean(
        _runtime_path: Path,
        container_name: str,
        _environment: Any,
    ) -> bool:
        cleanup_names.append(container_name)
        return True

    monkeypatch.setattr(secrets, "token_hex", lambda _count: "c" * 24)
    monkeypatch.setattr(probe, "_run_bounded_command", fail)
    monkeypatch.setattr(probe, "_cleanup_named_container", clean)

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="cleanup completed"):
        probe._run_child_with_qualification_mount(
            qualification_root,
            probe_mount,
            oci_runtime=runtime,
            expected_probe_source_sha256=hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        )
    assert cleanup_names == ["alberta-causal-q-probe-" + "c" * 24]


def test_probe_source_mutation_across_oci_execution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    source_root.mkdir()
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)

    def mutate_private_probe(command: tuple[str, ...], **_kwargs: Any) -> Any:
        original_size = probe_mount.stat().st_size
        probe_mount.chmod(0o600)
        probe_mount.write_bytes(b"x" * original_size)
        return subprocess.CompletedProcess(
            command,
            0,
            probe._canonical_json_bytes(_child_payload()),
            b"",
        )

    monkeypatch.setattr(probe, "_run_bounded_command", mutate_private_probe)

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="bytes changed"):
        probe._run_child_with_qualification_mount(
            qualification_root,
            probe_mount,
            oci_runtime=runtime,
            expected_probe_source_sha256=hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        )


@pytest.mark.parametrize("fault", [KeyboardInterrupt(), SystemExit(9)])
def test_oci_interrupts_always_attempt_named_container_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: BaseException,
) -> None:
    source_root = tmp_path / "source"
    qualification_root = tmp_path / "qualification"
    runtime = tmp_path / "docker"
    source_root.mkdir()
    qualification_root.mkdir()
    runtime.write_bytes(b"fake docker executable")
    runtime.chmod(0o755)
    probe_mount = _private_probe_mount(tmp_path)
    cleanup_names: list[str] = []

    def interrupt(*_args: Any, **_kwargs: Any) -> Any:
        raise fault

    def clean(
        _runtime_path: Path,
        container_name: str,
        _environment: Any,
    ) -> bool:
        cleanup_names.append(container_name)
        return True

    monkeypatch.setattr(secrets, "token_hex", lambda _count: "d" * 24)
    monkeypatch.setattr(probe, "_run_bounded_command", interrupt)
    monkeypatch.setattr(probe, "_cleanup_named_container", clean)

    with pytest.raises(type(fault)):
        probe._run_child_with_qualification_mount(
            qualification_root,
            probe_mount,
            oci_runtime=runtime,
            expected_probe_source_sha256=hashlib.sha256(
                Path(probe.__file__).read_bytes()
            ).hexdigest(),
        )
    assert cleanup_names == ["alberta-causal-q-probe-" + "d" * 24]


def test_bounded_command_never_accumulates_beyond_the_stdout_limit() -> None:
    command = (
        sys.executable,
        "-c",
        "import os; os.write(1, b'x' * 4096)",
    )

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="hard byte bound"):
        probe._run_bounded_command(
            command,
            environment=os.environ,
            timeout_seconds=10,
            maximum_stdout_bytes=64,
            maximum_stderr_bytes=64,
        )


def test_bounded_command_cleans_process_and_pipes_when_selector_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.killed = False
            self.waited = False

        def poll(self) -> int | None:
            return -9 if self.killed else None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            return -9

    class FailingSelector:
        def __init__(self) -> None:
            self.closed = False

        def register(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected selector registration failure")

        def close(self) -> None:
            self.closed = True

    process = FakeProcess()
    selector = FailingSelector()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(selectors, "DefaultSelector", lambda: selector)

    with pytest.raises(OSError, match="selector registration"):
        probe._run_bounded_command(
            ("unused",),
            environment={},
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )

    assert process.killed is True
    assert process.waited is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


@pytest.mark.parametrize(
    ("failure_phase", "cleanup_error", "expected_message"),
    (
        (
            "kill",
            OSError("injected kill failure"),
            "bounded child could not be terminated cleanly",
        ),
        (
            "kill",
            ProcessLookupError("injected exited-child race"),
            "hard byte bound",
        ),
        (
            "wait",
            subprocess.TimeoutExpired(("unused",), 10),
            "bounded child could not be reaped after termination",
        ),
        (
            "wait",
            OSError("injected wait failure"),
            "bounded child could not be inspected after termination",
        ),
        (
            "selector_close",
            OSError("injected selector close failure"),
            "bounded child resources could not be closed cleanly",
        ),
        (
            "stdout_close",
            OSError("injected stdout close failure"),
            "bounded child resources could not be closed cleanly",
        ),
        (
            "stderr_close",
            OSError("injected stderr close failure"),
            "bounded child resources could not be closed cleanly",
        ),
    ),
    ids=(
        "kill-error",
        "kill-process-gone",
        "reap-timeout",
        "reap-error",
        "selector-close-error",
        "stdout-close-error",
        "stderr-close-error",
    ),
)
def test_bounded_command_normalizes_termination_and_reap_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    cleanup_error: BaseException,
    expected_message: str,
) -> None:
    class CloseBuffer(io.BytesIO):
        def __init__(self, failure_name: str) -> None:
            super().__init__()
            self.failure_name = failure_name

        def close(self) -> None:
            super().close()
            if failure_phase == self.failure_name:
                raise cleanup_error

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = CloseBuffer("stdout_close")
            self.stderr = CloseBuffer("stderr_close")
            self.kill_called = False
            self.waited = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_called = True
            if failure_phase == "kill":
                raise cleanup_error

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            if failure_phase == "wait":
                raise cleanup_error
            return -9

    class SelectorKey:
        data = "stdout"
        fd = 101

    class OverflowSelector:
        def __init__(self) -> None:
            self.closed = False

        def register(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def get_map(self) -> dict[str, bool]:
            return {"active": True}

        def select(self, _timeout: float) -> list[tuple[SelectorKey, int]]:
            return [(SelectorKey(), selectors.EVENT_READ)]

        def close(self) -> None:
            self.closed = True
            if failure_phase == "selector_close":
                raise cleanup_error

    process = FakeProcess()
    selector = OverflowSelector()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"xx")

    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match=expected_message,
    ) as caught:
        probe._run_bounded_command(  # noqa: SLF001
            ("unused",),
            environment={},
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )

    assert process.kill_called is True
    assert process.waited is True
    if isinstance(cleanup_error, ProcessLookupError):
        assert caught.value.__cause__ is None
    else:
        assert caught.value.__cause__ is cleanup_error
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_publication_after_rename_has_distinct_uncertain_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published-uncertain"
    checks = iter((True, True, False))
    monkeypatch.setattr(
        probe,
        "_directory_descriptor_matches_path",
        lambda _descriptor, _path: next(checks),
    )

    with pytest.raises(probe.ReceiptPublishedButUncertainError, match="do not reuse"):
        probe._write_receipt(output_root, {"status": "test"})
    assert output_root.is_dir()
    assert (output_root / "receipt.json").is_file()

    def uncertain(**_kwargs: Any) -> Any:
        raise probe.ReceiptPublishedButUncertainError("injected")

    monkeypatch.setattr(probe, "run_probe", uncertain)
    assert probe.main([]) == 3


def test_published_descriptor_close_failure_is_uncertain_and_closes_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "published-close-failure"
    real_match = probe._directory_entry_matches_descriptor  # noqa: SLF001
    matched_descriptors: list[tuple[int, int]] = []

    def track_match(parent: int, name: str, held: int) -> bool:
        matched_descriptors.append((parent, held))
        return real_match(parent, name, held)

    monkeypatch.setattr(probe, "_directory_entry_matches_descriptor", track_match)
    real_close = os.close
    attempted_closes: list[int] = []
    failure_index: int | None = None

    def fail_published_close(descriptor: int) -> None:
        nonlocal failure_index
        attempted_closes.append(descriptor)
        published_descriptor = (
            matched_descriptors[1][1] if len(matched_descriptors) >= 2 else None
        )
        if descriptor == published_descriptor and failure_index is None:
            real_close(descriptor)
            failure_index = len(attempted_closes) - 1
            raise OSError("injected published-descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", fail_published_close)

    with pytest.raises(
        probe.ReceiptPublishedButUncertainError,
        match="do not reuse",
    ):
        probe._write_receipt(output_root, {"status": "test"})

    assert output_root.is_dir()
    assert failure_index is not None
    parent_descriptor, staging_descriptor = matched_descriptors[0]
    assert staging_descriptor in attempted_closes[failure_index + 1 :]
    assert parent_descriptor in attempted_closes[failure_index + 1 :]


def test_unpublished_descriptor_close_failure_is_public_and_closes_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "unpublished-close-failure"
    checks = iter((True, False))
    monkeypatch.setattr(
        probe,
        "_directory_descriptor_matches_path",
        lambda _descriptor, _path: next(checks),
    )
    real_create = probe._create_staging_directory  # noqa: SLF001
    held_descriptors: dict[str, int] = {}

    def track_create(parent: int, *, output_name: str) -> tuple[str, int]:
        staging_name, staging_descriptor = real_create(
            parent,
            output_name=output_name,
        )
        held_descriptors.update(parent=parent, staging=staging_descriptor)
        return staging_name, staging_descriptor

    monkeypatch.setattr(probe, "_create_staging_directory", track_create)
    real_close = os.close
    attempted_closes: list[int] = []
    failed = False

    def fail_staging_close(descriptor: int) -> None:
        nonlocal failed
        attempted_closes.append(descriptor)
        if descriptor == held_descriptors.get("staging") and not failed:
            real_close(descriptor)
            failed = True
            raise OSError("injected staging-descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", fail_staging_close)

    with pytest.raises(probe.CausalGridDivergenceProbeError) as caught:
        probe._write_receipt(output_root, {"status": "test"})

    assert not isinstance(caught.value, probe.ReceiptPublishedButUncertainError)
    assert not output_root.exists()
    assert failed is True
    staging_index = attempted_closes.index(held_descriptors["staging"])
    assert held_descriptors["parent"] in attempted_closes[staging_index + 1 :]


def test_parent_path_change_before_rename_leaves_no_published_or_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "not-published"
    checks = iter((True, False))
    monkeypatch.setattr(
        probe,
        "_directory_descriptor_matches_path",
        lambda _descriptor, _path: next(checks),
    )

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="before publication"):
        probe._write_receipt(output_root, {"status": "test"})
    assert not output_root.exists()
    assert list(tmp_path.glob(".not-published.staging-*")) == []


def test_destination_inode_mismatch_after_rename_is_published_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "inode-mismatch"
    monkeypatch.setattr(
        probe,
        "_directory_entry_matches_descriptor",
        lambda _parent, _name, _held: False,
    )

    with pytest.raises(
        probe.ReceiptPublishedButUncertainError,
        match="do not reuse",
    ):
        probe._write_receipt(output_root, {"status": "test"})
    assert output_root.is_dir()


def test_strict_receipt_loader_replays_schema_digest_and_exact_inventory(
    tmp_path: Path,
) -> None:
    receipt, _passed = probe._assemble_receipt(
        _child_payload(),
        probe_source_sha256="a" * 64,
        execution_envelope=_execution_envelope(),
    )
    output_root = tmp_path / "valid-receipt"
    probe._write_receipt(output_root, receipt)

    assert probe.load_receipt(output_root) == receipt
    (output_root / "unexpected.txt").write_bytes(b"unexpected")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="exactly"):
        probe.load_receipt(output_root)


def test_strict_receipt_loader_rejects_sidecar_canonical_and_schema_tampering(
    tmp_path: Path,
) -> None:
    receipt, _passed = probe._assemble_receipt(
        _child_payload(),
        probe_source_sha256="a" * 64,
        execution_envelope=_execution_envelope(),
    )

    sidecar_root = tmp_path / "sidecar-tamper"
    probe._write_receipt(sidecar_root, receipt)
    sidecar_path = sidecar_root / "receipt.json.sha256"
    sidecar_path.chmod(0o600)
    sidecar_path.write_bytes(b"0" * 64 + b"\n")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="sidecar"):
        probe.load_receipt(sidecar_root)

    canonical_root = tmp_path / "canonical-tamper"
    probe._write_receipt(canonical_root, receipt)
    receipt_path = canonical_root / "receipt.json"
    canonical_sidecar = canonical_root / "receipt.json.sha256"
    receipt_path.chmod(0o600)
    canonical_sidecar.chmod(0o600)
    noncanonical = receipt_path.read_bytes() + b"\n"
    receipt_path.write_bytes(noncanonical)
    canonical_sidecar.write_bytes(hashlib.sha256(noncanonical).hexdigest().encode() + b"\n")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="canonical"):
        probe.load_receipt(canonical_root)

    schema_root = tmp_path / "schema-tamper"
    probe._write_receipt(schema_root, receipt)
    schema_receipt_path = schema_root / "receipt.json"
    schema_sidecar = schema_root / "receipt.json.sha256"
    schema_receipt_path.chmod(0o600)
    schema_sidecar.chmod(0o600)
    changed = copy.deepcopy(receipt)
    changed["classification"] = "tampered"
    changed_raw = probe._canonical_json_bytes(changed)
    schema_receipt_path.write_bytes(changed_raw)
    schema_sidecar.write_bytes(hashlib.sha256(changed_raw).hexdigest().encode() + b"\n")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="schema replay"):
        probe.load_receipt(schema_root)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("seed", False),
        lambda value: value.__setitem__("promotion_authorized", 0),
        lambda value: value["action_boundary"]["scalar_domain"].__setitem__(0, False),
        lambda value: value["divergence_summary"].__setitem__(
            "divergent_pair_count", False
        ),
    ),
)
def test_strict_receipt_loader_rejects_boolean_integer_aliases(
    tmp_path: Path,
    mutate: Any,
) -> None:
    receipt, _passed = probe._assemble_receipt(
        _child_payload(divergent=False),
        probe_source_sha256="a" * 64,
        execution_envelope=_execution_envelope(),
    )
    mutate(receipt)
    raw = probe._canonical_json_bytes(receipt)
    output_root = tmp_path / "boolean-alias"
    output_root.mkdir()
    (output_root / "receipt.json").write_bytes(raw)
    (output_root / "receipt.json.sha256").write_bytes(
        hashlib.sha256(raw).hexdigest().encode() + b"\n"
    )

    with pytest.raises(probe.CausalGridDivergenceProbeError, match="schema replay"):
        probe.load_receipt(output_root)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="duplicate"):
        probe._decode_strict_json(b'{"a":1,"a":2}', label="test")
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="non-finite"):
        probe._decode_strict_json(b'{"a":NaN}', label="test")
    with pytest.raises(ValueError):
        json.dumps(float("nan"), allow_nan=False)


def test_pair_commitment_tampering_and_raw_payload_drift_fail_closed() -> None:
    payload = _child_payload(divergent=False)
    tampered = copy.deepcopy(payload)
    tampered["candidates"][1]["action_trace_sha256"] = "f" * 64
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="unequal action"):
        probe._validate_child_payload(tampered)

    divergent = _child_payload(divergent=True)
    proof = divergent["pairwise_action_divergence"][1][
        "first_divergence_merkle_proof"
    ]
    proof["levels"][0]["right_arm_left_child_sha256"] = "f" * 64
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="candidate roots"):
        probe._validate_child_payload(divergent)


@pytest.mark.parametrize("divergence_index", [0, 1, 1_999, 2_000, 9_999])
def test_merkle_proof_binds_exact_first_divergence_at_boundary_indices(
    divergence_index: int,
) -> None:
    left_actions = bytes(probe.FIXED_STEPS)
    right_values = bytearray(left_actions)
    right_values[divergence_index] = 3
    right_actions = bytes(right_values)
    left_tree = probe._build_action_merkle_tree(left_actions)
    right_tree = probe._build_action_merkle_tree(right_actions)
    proof = probe._first_divergence_merkle_proof(
        left_tree,
        right_tree,
        divergence_index=divergence_index,
        left_action=0,
        right_action=3,
    )

    probe._validate_first_divergence_merkle_proof(
        proof,
        divergence_index=divergence_index,
        left_root_sha256=probe._action_merkle_root_sha256(left_tree),
        right_root_sha256=probe._action_merkle_root_sha256(right_tree),
    )


def test_merkle_proof_rejects_noncanonical_path_equal_actions_and_hidden_earlier_change() -> None:
    divergence_index = 2_000
    left_actions = bytes(probe.FIXED_STEPS)
    right_values = bytearray(left_actions)
    right_values[divergence_index] = 1
    right_actions = bytes(right_values)
    left_tree = probe._build_action_merkle_tree(left_actions)
    right_tree = probe._build_action_merkle_tree(right_actions)
    roots = {
        "left_root_sha256": probe._action_merkle_root_sha256(left_tree),
        "right_root_sha256": probe._action_merkle_root_sha256(right_tree),
    }
    proof = probe._first_divergence_merkle_proof(
        left_tree,
        right_tree,
        divergence_index=divergence_index,
        left_action=0,
        right_action=1,
    )

    wrong_path = copy.deepcopy(proof)
    wrong_path["levels"][0]["followed_child"] = (
        "right" if wrong_path["levels"][0]["followed_child"] == "left" else "left"
    )
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="noncanonical"):
        probe._validate_first_divergence_merkle_proof(
            wrong_path,
            divergence_index=divergence_index,
            **roots,
        )

    equal_actions = copy.deepcopy(proof)
    equal_actions["right_action"] = equal_actions["left_action"]
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="actions are equal"):
        probe._validate_first_divergence_merkle_proof(
            equal_actions,
            divergence_index=divergence_index,
            **roots,
        )

    earlier_values = bytearray(right_actions)
    earlier_values[divergence_index - 1] = 2
    earlier_tree = probe._build_action_merkle_tree(bytes(earlier_values))
    hidden_earlier = probe._first_divergence_merkle_proof(
        left_tree,
        earlier_tree,
        divergence_index=divergence_index,
        left_action=0,
        right_action=1,
    )
    with pytest.raises(
        probe.CausalGridDivergenceProbeError,
        match="unequal earlier subtree|becomes equal",
    ):
        probe._validate_first_divergence_merkle_proof(
            hidden_earlier,
            divergence_index=divergence_index,
            left_root_sha256=roots["left_root_sha256"],
            right_root_sha256=probe._action_merkle_root_sha256(earlier_tree),
        )


def test_merkle_proof_rejects_boolean_integer_aliases() -> None:
    left_actions = bytes(probe.FIXED_STEPS)
    right_values = bytearray(left_actions)
    right_values[0] = 1
    right_actions = bytes(right_values)
    left_tree = probe._build_action_merkle_tree(left_actions)
    right_tree = probe._build_action_merkle_tree(right_actions)
    proof = probe._first_divergence_merkle_proof(
        left_tree,
        right_tree,
        divergence_index=0,
        left_action=0,
        right_action=1,
    )
    roots = {
        "left_root_sha256": probe._action_merkle_root_sha256(left_tree),
        "right_root_sha256": probe._action_merkle_root_sha256(right_tree),
    }

    boolean_index = copy.deepcopy(proof)
    boolean_index["divergence_index"] = False
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="identity drifted"):
        probe._validate_first_divergence_merkle_proof(
            boolean_index,
            divergence_index=0,
            **roots,
        )

    boolean_level = copy.deepcopy(proof)
    boolean_level["levels"][-1]["level"] = True
    with pytest.raises(probe.CausalGridDivergenceProbeError, match="noncanonical"):
        probe._validate_first_divergence_merkle_proof(
            boolean_level,
            divergence_index=0,
            **roots,
        )
