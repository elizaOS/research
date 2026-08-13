"""Tests for the pure, non-authorizing matched-v3 local configuration builder."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_open_protocol as matched_current,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_configuration as local_configuration,
)

_EXPECTED_IDS = (
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

_EXPECTED_SHA256 = {
    "causal_e025_q050": "1290335563481b7ac2fd3eda91ef9c63216684fd096f3ab5b16591de0870c736",
    "causal_e025_q075": "69a5df44db99866a0ee3967677fad66ea94c60b1bfa8317936e2c142fac34ed1",
    "causal_e025_q090": "e21692571fc751bdf2c4fa0e89ad43b12dbd51c72a0821d5839fc82f1031f8f4",
    "causal_e050_q050": "916bd37e04c39dc16c19153032fc1c3baf12a941efb3df95860ee9f03c1ef331",
    "causal_e050_q075": "afaa3ea47cd410a43541c85976fa6f718c5f70504494f70496385ec37ea84a63",
    "causal_e050_q090": "ab555510e08a98e733d01a9b145d19073bb17ba31681a459a55a978d5a4faf33",
    "causal_e100_q050": "00390162a1950e976a7b3e216b8c6d94a76427c38c8e30bbdc25fa583bf018a8",
    "causal_e100_q075": "8d7a8afdb204c1837834ef633e2524bf569180c763a34a96c883c6e2cd33fb48",
    "causal_e100_q090": "899658dff1eeaadf59de8dc437d1324429306b8a427a4ed67ccf54437931955c",
    "alberta_horde_default": (
        "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a"
    ),
    "alberta_horde_eps05": (
        "ab402dd011e2d97df423ffa2f0203ea9fe3c01dcfc89db66d2f2fdf404b7204f"
    ),
    "alberta_horde_recurrent64": (
        "870e805b046f1751cac48368b07827e3c27059d849f2a84b1c2e499e75e0f6ef"
    ),
    "alberta_horde_step3e3": (
        "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5"
    ),
    "alberta_rtu_h08_taylor": (
        "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
    ),
}
_EXPECTED_SOURCE_DESCRIPTOR_SHA256 = (
    "d15d70b55d965b2c135f1dcaa36a74173e4023e4fdc9430c43660df54f1bb38c"
)
_EXPECTED_BUILDER_DESCRIPTOR_SHA256 = (
    "1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@pytest.mark.unit
def test_exact_native_candidate_set_replays_matched_current_payloads_and_hashes() -> None:
    source = matched_current.matched_current_alberta_configurations()
    built = local_configuration.build_all_matched_v3_local_configurations()

    assert local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS == _EXPECTED_IDS
    assert tuple(item.candidate_id for item in built) == _EXPECTED_IDS
    assert tuple(source) == _EXPECTED_IDS
    assert dict(local_configuration.EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE) == (
        _EXPECTED_SHA256
    )
    for item in built:
        expected_bytes = _canonical_bytes(source[item.candidate_id])
        assert item.canonical_json_bytes == expected_bytes
        assert item.configuration_sha256 == _EXPECTED_SHA256[item.candidate_id]
        assert hashlib.sha256(item.canonical_json_bytes).hexdigest() == (
            item.configuration_sha256
        )
        assert item.payload() == source[item.candidate_id]
        assert item.status == "implemented_unqualified"
        assert item.configuration_complete is True
        assert item.execution_ready is False
        assert item.execution_authorized is False
        assert item.scientific_promotion_allowed is False
        assert item.universal_sota_claim_allowed is False


@pytest.mark.unit
def test_exact_group_specific_builder_bindings_are_stable() -> None:
    expected = {
        **{
            candidate_id: (
                "alberta.forager_matched_v3.generated_local.causal_map_grid.v1"
            )
            for candidate_id in _EXPECTED_IDS[:9]
        },
        **{
            candidate_id: (
                "alberta.forager_matched_v3.generated_local.horde_actor_critic.v1"
            )
            for candidate_id in _EXPECTED_IDS[9:13]
        },
        _EXPECTED_IDS[13]: (
            "alberta.forager_matched_v3.generated_local.rtu_h08_taylor.v1"
        ),
    }
    assert dict(local_configuration.BUILDER_ID_BY_CANDIDATE) == expected
    assert {
        item.candidate_id: item.builder_id
        for item in local_configuration.build_all_matched_v3_local_configurations()
    } == expected


@pytest.mark.unit
def test_builder_and_source_descriptors_are_canonical_detached_and_non_authorizing() -> None:
    source = local_configuration.matched_v3_local_configuration_source_descriptor()
    builder = local_configuration.matched_v3_local_configuration_builder_descriptor()

    assert source["schema_version"] == (
        local_configuration.LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SCHEMA_VERSION
    )
    assert source["status"] == "implemented_unqualified"
    assert source["source_snapshot_status"] == "unqualified_current_checkout"
    assert source["claims"] == {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
    assert builder["schema_version"] == (
        local_configuration.LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SCHEMA_VERSION
    )
    assert builder["status"] == "implemented_unqualified"
    assert builder["source_descriptor_sha256"] == (
        local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
    )
    assert builder["claims"] == source["claims"]

    source_bytes = (
        local_configuration.canonical_matched_v3_local_configuration_source_descriptor_bytes()
    )
    builder_bytes = (
        local_configuration.canonical_matched_v3_local_configuration_builder_descriptor_bytes()
    )
    assert source_bytes == _canonical_bytes(source)
    assert builder_bytes == _canonical_bytes(builder)
    assert hashlib.sha256(source_bytes).hexdigest() == (
        local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
    )
    assert hashlib.sha256(builder_bytes).hexdigest() == (
        local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
    )
    assert local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256 == (
        _EXPECTED_SOURCE_DESCRIPTOR_SHA256
    )
    assert local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256 == (
        _EXPECTED_BUILDER_DESCRIPTOR_SHA256
    )

    source["claims"]["execution_authorized"] = True
    builder["candidate_bindings"][0]["candidate_id"] = "mutated"
    assert local_configuration.matched_v3_local_configuration_source_descriptor()[
        "claims"
    ]["execution_authorized"] is False
    assert local_configuration.matched_v3_local_configuration_builder_descriptor()[
        "candidate_bindings"
    ][0]["candidate_id"] == _EXPECTED_IDS[0]


@pytest.mark.unit
def test_descriptor_reads_ignore_mutated_private_construction_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bytes = (
        local_configuration.canonical_matched_v3_local_configuration_source_descriptor_bytes()
    )
    builder_bytes = (
        local_configuration.canonical_matched_v3_local_configuration_builder_descriptor_bytes()
    )
    monkeypatch.setitem(local_configuration._SOURCE_DESCRIPTOR, "status", "mutated")
    monkeypatch.setitem(local_configuration._BUILDER_DESCRIPTOR, "status", "mutated")

    assert (
        local_configuration.canonical_matched_v3_local_configuration_source_descriptor_bytes()
        == source_bytes
    )
    assert (
        local_configuration.canonical_matched_v3_local_configuration_builder_descriptor_bytes()
        == builder_bytes
    )
    assert local_configuration.matched_v3_local_configuration_source_descriptor()[
        "status"
    ] == "implemented_unqualified"
    assert local_configuration.matched_v3_local_configuration_builder_descriptor()[
        "status"
    ] == "implemented_unqualified"


@pytest.mark.unit
def test_builder_module_has_no_eager_jax_or_matched_current_source_import() -> None:
    module_path = Path(local_configuration.__file__)
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    eager_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_modules = {
        alias.name
        for node in eager_imports
        for alias in node.names
    } | {
        node.module
        for node in eager_imports
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(name == "jax" or name.startswith("jax.") for name in imported_modules)
    assert not any("forager_matched_open_protocol" in name for name in imported_modules)

    probe = (
        "import runpy,sys; "
        "assert 'jax' not in sys.modules; "
        "runpy.run_path(sys.argv[1], run_name='__local_configuration_purity_probe__'); "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", probe, str(module_path)],
        check=True,
        capture_output=True,
        text=True,
    )


class _StringAlias(str):
    pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "candidate_id",
    [
        True,
        1,
        None,
        _StringAlias("causal_e025_q050"),
        "CAUSAL_E025_Q050",
        " causal_e025_q050",
        "causal_e025_q050 ",
        "not_a_candidate",
    ],
)
def test_candidate_identifier_aliases_and_unknown_values_fail_closed(
    candidate_id: object,
) -> None:
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="candidate_id|unknown matched-v3 local candidate",
    ):
        local_configuration.build_matched_v3_local_configuration(candidate_id)


@pytest.mark.unit
def test_payloads_and_aggregate_results_are_detached_and_immutable() -> None:
    first = local_configuration.build_matched_v3_local_configuration(_EXPECTED_IDS[0])
    payload = first.payload()
    payload["configuration"]["exploration_probability"] = 1.0

    second = local_configuration.build_matched_v3_local_configuration(_EXPECTED_IDS[0])
    assert second.payload()["configuration"]["exploration_probability"] == 0.025
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.candidate_id = "mutated"  # type: ignore[misc]
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="authority",
    ):
        dataclasses.replace(first, execution_authorized=True)

    all_first = local_configuration.build_all_matched_v3_local_configurations()
    all_second = local_configuration.build_all_matched_v3_local_configurations()
    assert all_first is not all_second
    assert all_first[0] is not all_second[0]
    assert all_first[0].canonical_json_bytes == all_second[0].canonical_json_bytes


@pytest.mark.unit
def test_parser_accepts_only_exact_candidate_bound_canonical_payload() -> None:
    built = local_configuration.build_matched_v3_local_configuration(_EXPECTED_IDS[0])
    parsed = local_configuration.parse_matched_v3_local_configuration_payload(
        _EXPECTED_IDS[0], built.canonical_json_bytes
    )
    assert parsed == built

    noncanonical = json.dumps(built.payload(), indent=2).encode("utf-8")
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="canonical|does not match",
    ):
        local_configuration.parse_matched_v3_local_configuration_payload(
            _EXPECTED_IDS[0], noncanonical
        )
    other = local_configuration.build_matched_v3_local_configuration(_EXPECTED_IDS[1])
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="does not match",
    ):
        local_configuration.parse_matched_v3_local_configuration_payload(
            _EXPECTED_IDS[0], other.canonical_json_bytes
        )
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="exact bytes",
    ):
        local_configuration.parse_matched_v3_local_configuration_payload(
            _EXPECTED_IDS[0], built.canonical_json_bytes.decode("utf-8")  # type: ignore[arg-type]
        )


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["missing", "extra", "payload", "alias"])
def test_source_configuration_set_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = matched_current.matched_current_alberta_configurations()
    forged: dict[str, dict[str, Any]] = copy.deepcopy(source)
    if mutation == "missing":
        forged.pop(_EXPECTED_IDS[-1])
    elif mutation == "extra":
        forged["alias"] = copy.deepcopy(forged[_EXPECTED_IDS[0]])
    elif mutation == "payload":
        forged[_EXPECTED_IDS[0]]["configuration"]["exploration_probability"] = 0.5
    else:
        forged[_EXPECTED_IDS[1]] = forged[_EXPECTED_IDS[0]]

    monkeypatch.setattr(
        matched_current,
        "matched_current_alberta_configurations",
        lambda: forged,
    )
    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match="candidate|drift|aliased",
    ):
        local_configuration.build_all_matched_v3_local_configurations()


@pytest.mark.unit
def test_single_candidate_build_validates_the_atomic_14_arm_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = matched_current.matched_current_alberta_configurations()
    source[_EXPECTED_IDS[-1]]["configuration"]["freeze_after_steps"] = 1
    monkeypatch.setattr(
        matched_current,
        "matched_current_alberta_configurations",
        lambda: source,
    )

    with pytest.raises(
        local_configuration.ForagerMatchedV3LocalConfigurationError,
        match=f"local worker envelope drift for {_EXPECTED_IDS[-1]}",
    ):
        local_configuration.build_matched_v3_local_configuration(_EXPECTED_IDS[0])
