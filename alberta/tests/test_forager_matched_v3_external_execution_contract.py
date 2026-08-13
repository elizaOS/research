"""Cheap exhaustive tests for the descriptor-only external execution contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_external_execution_contract as contract,
)

pytestmark = pytest.mark.unit

_DESCRIPTOR_SHA256 = "9e1a8d73ec14de554b3fdb3e5457f0448ca91adc46bf9f53988e7538bbc0eca4"
_CONFIGURATION_PLAN_SHA256 = "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
_CONFIGURATION_PLAN_SOURCE_SHA256 = (
    "ad711eaa61511c6b1d43b86b867e09ba70f7124d5d67966b22d1f7ef3a556a84"
)
_MATERIALIZER_V2_IDENTITY_SHA256 = (
    "74cf45b9d09b06c17dd38c8713940f32a04e887259bb027c75bfa680e7b43192"
)
_MATERIALIZER_V2_SOURCE_SHA256 = "3ff59a9f88d79b122fa66a1cdca009a68ff524806a7a7c58e5d565cd30ecaafe"
_SEED_TRANSPORT_DESCRIPTOR_SHA256 = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)
_SEED_TRANSPORT_SOURCE_SHA256 = "18f24a5116ae927c903b23a5cc64b1628aa135c808ccc985bbf2060e831d66f0"
_RESULT_BRIDGE_DESCRIPTOR_SHA256 = (
    "19c784eeb709b44f2729ba4a6cf9af35a563995f51d1af91b1674af8523a90dd"
)
_RESULT_BRIDGE_SOURCE_SHA256 = "c1859f0cfb7862e22c470f89ad9d3298a76b1fb419bf1431069f286f593e22f7"
_CONTINUING_ENTRYPOINT_SHA256 = "ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"
_PPO_ENTRYPOINT_SHA256 = "1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"
_PPO_VIDEO = "videos/0/497664_499712-episode-0.mp4"

# This order is the external-spec insertion order in configuration-plan v1.
_EXPECTED = (
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


def _canonical(value: object) -> bytes:
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


def _leaf_paths(value: Any, prefix: tuple[str | int, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if type(value) is dict:
        for key, child in value.items():
            yield from _leaf_paths(child, prefix + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _leaf_paths(child, prefix + (index,))
    else:
        yield prefix


def _get(value: Any, path: tuple[Any, ...]) -> Any:
    current = value
    for component in path:
        current = current[component]
    return current


def _set(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    parent = _get(value, path[:-1])
    parent[path[-1]] = replacement


def _mutated_scalar(value: Any) -> Any:
    if value is None:
        return False
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "-mutated"
    raise AssertionError(f"unexpected leaf type: {type(value).__name__}")


def test_descriptor_is_self_pinned_canonical_detached_and_exactly_parseable() -> None:
    raw = contract.canonical_external_execution_contract_descriptor_bytes()
    descriptor = contract.external_execution_contract_descriptor()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert len(raw) == 45_121
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert contract.EXTERNAL_EXECUTION_CONTRACT_DESCRIPTOR_SHA256 == _DESCRIPTOR_SHA256
    assert contract.external_execution_contract_descriptor_sha256() == _DESCRIPTOR_SHA256
    assert contract.parse_external_execution_contract_descriptor(raw) == descriptor
    descriptor["claims"]["execution_ready"] = True
    assert contract.external_execution_contract_descriptor()["claims"]["execution_ready"] is False


def test_historical_plan_is_bound_but_explicitly_not_upgraded_for_materialization() -> None:
    dependencies = contract.external_execution_contract_descriptor()["dependencies"]
    historical = dependencies["historical_configuration_plan_v1"]
    assert historical == {
        "schema_version": "alberta.forager_matched_v3_configuration_plan.v1",
        "descriptor_sha256": _CONFIGURATION_PLAN_SHA256,
        "current_source_path": (
            "alberta_framework/benchmarks/forager_matched_v3_configuration_plan.py"
        ),
        "current_source_sha256": _CONFIGURATION_PLAN_SOURCE_SHA256,
        "status_for_materialization": "historical_superseded",
        "imported_or_reconstructed_here": False,
        "silently_upgraded_to_materializer_v2": False,
        "historical_materializer_binding": {
            "manifest_schema_version": ("alberta.forager_matched_v3_external_materialization.v1"),
            "identity_schema_version": (
                "alberta.forager_matched_v3_external_materialization_identity.v1"
            ),
            "identity_sha256": ("5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"),
            "source_sha256": ("5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"),
            "selected_for_materialization": False,
            "superseded": True,
        },
    }
    assert dependencies["materializer_v2_overlay"] == {
        "manifest_schema_version": ("alberta.forager_matched_v3_external_materialization.v2"),
        "identity_schema_version": (
            "alberta.forager_matched_v3_external_materialization_identity.v2"
        ),
        "identity_sha256": _MATERIALIZER_V2_IDENTITY_SHA256,
        "source_path": (
            "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py"
        ),
        "source_sha256": _MATERIALIZER_V2_SOURCE_SHA256,
        "relationship_to_configuration_plan_v1": "separate_additive_overlay",
        "materialization_performed": False,
        "production_manifest_accepted": False,
    }


def test_seed_transport_result_bridge_and_all_bounds_are_exact() -> None:
    dependencies = contract.external_execution_contract_descriptor()["dependencies"]
    assert dependencies["external_seed_transport"] == {
        "schema_version": "alberta.forager_matched_v3_external_seed_transport.v1",
        "descriptor_sha256": _SEED_TRANSPORT_DESCRIPTOR_SHA256,
        "source_path": (
            "alberta_framework/benchmarks/forager_matched_v3_external_seed_transport.py"
        ),
        "source_sha256": _SEED_TRANSPORT_SOURCE_SHA256,
        "derived_entrypoint_sha256_by_path": {
            "src/continuing_main.py": _CONTINUING_ENTRYPOINT_SHA256,
            "src/rtu_ppo.py": _PPO_ENTRYPOINT_SHA256,
        },
    }
    assert dependencies["external_result_bridge"] == {
        "schema_version": ("alberta.forager_matched_v3.external_result_bridge_descriptor.v1"),
        "descriptor_sha256": _RESULT_BRIDGE_DESCRIPTOR_SHA256,
        "source_path": (
            "alberta_framework/benchmarks/_forager_matched_v3_external_result_bridge.py"
        ),
        "source_sha256": _RESULT_BRIDGE_SOURCE_SHA256,
        "bounds": {
            "reward_horizon": 499_712,
            "canonical_scorer_npz_size_bytes": 499_980,
            "maximum_external_npz_bytes": 64 * 1024 * 1024,
            "maximum_zip_member_count": 128,
            "maximum_zip_total_compressed_bytes": 64 * 1024 * 1024,
            "maximum_zip_total_expanded_bytes": 64 * 1024 * 1024,
            "maximum_npy_header_bytes": 4 * 1024,
        },
    }


@pytest.mark.parametrize(
    ("candidate_id", "path", "original", "derived", "stem", "family", "npy_descr"),
    _EXPECTED,
)
def test_every_candidate_binds_exact_configuration_command_and_output(
    candidate_id: str,
    path: str,
    original: str,
    derived: str,
    stem: str,
    family: str,
    npy_descr: str,
) -> None:
    record = contract.external_execution_candidate_record(candidate_id)
    configuration = record["configuration"]
    assert configuration == {
        "original_relative_path": path,
        "original_sha256": original,
        "derived_sha256": derived,
        "output_stem": stem,
        "derived_configuration_staging_relative_path": path,
        "staging_preserves_original_path_below_experiments": True,
    }

    is_ppo = family == "ppo"
    entrypoint = "src/rtu_ppo.py" if is_ppo else "src/continuing_main.py"
    entrypoint_sha256 = _PPO_ENTRYPOINT_SHA256 if is_ppo else _CONTINUING_ENTRYPOINT_SHA256
    max_steps = 244 if is_ppo else 499_712
    execution = record["execution"]
    assert execution == {
        "family": family,
        "npy_descr": npy_descr,
        "entrypoint_path": entrypoint,
        "entrypoint_sha256": entrypoint_sha256,
        "working_directory_placeholder": "<staged_materialized_checkout_root>",
        "working_directory_is_staged_materialized_checkout_root": True,
        "index": 0,
        "environment_seed_placeholder": "<environment_seed_uint31>",
        "agent_seed_placeholder": "<candidate_private_agent_seed_uint31>",
        "max_steps": max_steps,
        "interaction_horizon": 499_712,
        "ppo_rollout_steps": 2_048 if is_ppo else None,
        "ppo_rollout_count": 244 if is_ppo else None,
        "argv": [
            "--exp",
            path,
            "--idxs",
            "0",
            "--environment_seed",
            "<environment_seed_uint31>",
            "--agent_seed",
            "<candidate_private_agent_seed_uint31>",
            "--max_steps",
            str(max_steps),
            "--save_path",
            "<fresh_candidate_private_save_base>",
            "--checkpoint_path",
            "<new_empty_candidate_private_checkpoint_root>",
            "--silent",
        ],
    }
    assert execution["argv"][1] == configuration["derived_configuration_staging_relative_path"]

    root = record["root_contract"]
    config_directory = path.removeprefix("experiments/").rsplit("/", 1)[0]
    result_directory = f"results/{config_directory}/{stem}"
    assert root == {
        "save_base_placeholder": "<fresh_candidate_private_save_base>",
        "save_base_candidate_private": True,
        "save_base_fresh_empty_before_execution_required": True,
        "save_base_is_distinct_from_derived_result_directory": True,
        "derived_result_directory_relative_to_save_base": result_directory,
        "doubled_results_component_forbidden": True,
        "checkpoint_root_placeholder": "<new_empty_candidate_private_checkpoint_root>",
        "checkpoint_root_candidate_private": True,
        "checkpoint_root_fresh_empty_before_execution_required": True,
        "checkpoint_root_empty_after_execution_required": True,
        "checkpoint_root_exact_final_entries": [],
    }

    result_npz = f"{result_directory}/data/0.npz"
    database = f"{result_directory}/results.db"
    video = f"{result_directory}/{_PPO_VIDEO}" if is_ppo else None
    exact_files = [
        {"artifact_kind": "external_reward_npz", "path": result_npz},
        {"artifact_kind": "sibling_results_database", "path": database},
    ]
    if video is not None:
        exact_files.append({"artifact_kind": "ppo_video", "path": video})
    assert record["artifact_contract"] == {
        "result_directory": result_directory,
        "paths_are_relative_to_candidate_private_save_base": True,
        "result_npz_path": result_npz,
        "results_database_path": database,
        "results_database_is_sibling_of_data_directory": True,
        "ppo_video_relative_to_result_directory": _PPO_VIDEO if is_ppo else None,
        "ppo_video_path": video,
        "exact_files": exact_files,
        "missing_files_allowed": False,
        "extra_files_allowed": False,
    }
    assert not result_directory.startswith("results/results/")
    assert all(value is False for value in record["claims"].values())


def test_candidate_order_is_exact_plan_order_with_no_missing_or_extra_records() -> None:
    descriptor = contract.external_execution_contract_descriptor()
    expected_ids = tuple(item[0] for item in _EXPECTED)
    assert contract.EXTERNAL_EXECUTION_CANDIDATE_IDS == expected_ids
    assert tuple(descriptor["candidate_order"]) == expected_ids
    assert tuple(record["candidate_id"] for record in descriptor["candidates"]) == expected_ids
    assert descriptor["candidate_count"] == 12
    assert len(set(expected_ids)) == 12
    assert descriptor["artifact_inventory_policy"]["missing_candidates_allowed"] is False
    assert descriptor["artifact_inventory_policy"]["extra_candidates_allowed"] is False


def test_horizon_ppo_arithmetic_save_base_and_video_policy_are_exact() -> None:
    descriptor = contract.external_execution_contract_descriptor()
    workload = descriptor["workload_contract"]
    assert workload["interaction_horizon"] == 499_712
    assert workload["continuing_max_steps"] == 499_712
    assert workload["ppo_max_steps"] == 244
    assert workload["ppo_rollout_steps"] == 2_048
    assert workload["ppo_rollout_count"] == 244
    assert workload["ppo_interaction_count"] == 499_712 == 244 * 2_048
    assert workload["exactly_one_index"] == 0
    assert workload["save_path_is_base_not_derived_result_directory"] is True
    assert workload["doubled_results_component_forbidden"] is True
    policy = descriptor["artifact_inventory_policy"]
    assert policy["exact_results_prefix_count"] == 1
    assert policy["doubled_results_component_forbidden"] is True
    assert policy["derived_configuration_must_retain_original_relative_path"] is True
    assert policy["ppo_only_video_path_relative_to_result_directory"] == _PPO_VIDEO
    assert policy["missing_artifacts_allowed"] is False
    assert policy["extra_artifacts_allowed"] is False


def test_every_api_and_authority_readiness_qualification_acceptance_claim_is_false() -> None:
    descriptor = contract.external_execution_contract_descriptor()
    assert descriptor["status"] == (
        "implemented_descriptor_only_unexecuted_unqualified_non_authorizing"
    )
    assert all(value is False for value in descriptor["apis"].values())
    assert all(value is False for value in descriptor["claims"].values())
    assert descriptor["claims"] == {
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


def test_record_lookup_is_detached_exact_and_rejects_nonexact_ids() -> None:
    first = contract.external_execution_candidate_record("external_dqn_plain")
    first["execution"]["argv"][3] = "1"
    second = contract.external_execution_candidate_record("external_dqn_plain")
    assert second["execution"]["argv"][3] == "0"
    with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
        contract.external_execution_candidate_record("unknown")
    for invalid in (None, True, 1, b"external_dqn_plain"):
        with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
            contract.external_execution_candidate_record(invalid)  # type: ignore[arg-type]


def test_every_descriptor_leaf_is_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "_verify_live_dependency_bindings", lambda: None)
    descriptor = contract.external_execution_contract_descriptor()
    paths = list(_leaf_paths(descriptor))
    assert len(paths) > 500
    for path in paths:
        changed = copy.deepcopy(descriptor)
        original = _get(changed, path)
        _set(changed, path, _mutated_scalar(original))
        with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
            contract.parse_external_execution_contract_descriptor(_canonical(changed))


def test_bool_aliases_order_changes_cross_version_and_inventory_changes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "_verify_live_dependency_bindings", lambda: None)
    descriptor = contract.external_execution_contract_descriptor()
    mutations: list[dict[str, Any]] = []

    changed = copy.deepcopy(descriptor)
    changed["candidate_count"] = True
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["workload_contract"]["exactly_one_index"] = False
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["claims"]["execution_ready"] = 0
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["schema_version"] = (
        "alberta.forager_matched_v3.external_execution_contract_descriptor.v2"
    )
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["dependencies"]["historical_configuration_plan_v1"]["schema_version"] = (
        "alberta.forager_matched_v3_configuration_plan.v2"
    )
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    historical = changed["dependencies"]["historical_configuration_plan_v1"]
    historical["descriptor_sha256"] = historical["current_source_sha256"]
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidate_order"][0], changed["candidate_order"][1] = (
        changed["candidate_order"][1],
        changed["candidate_order"][0],
    )
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidates"][0], changed["candidates"][1] = (
        changed["candidates"][1],
        changed["candidates"][0],
    )
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidates"].pop()
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidates"].append(copy.deepcopy(changed["candidates"][0]))
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidates"][0]["artifact_contract"]["exact_files"].pop()
    mutations.append(changed)
    changed = copy.deepcopy(descriptor)
    changed["candidates"][0]["artifact_contract"]["exact_files"].append(
        {"artifact_kind": "extra", "path": "results/extra"}
    )
    mutations.append(changed)

    for malformed in mutations:
        with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
            contract.parse_external_execution_contract_descriptor(_canonical(malformed))


def test_parser_rejects_duplicate_noncanonical_nonascii_float_huge_and_wrong_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "_verify_live_dependency_bindings", lambda: None)
    raw = contract.canonical_external_execution_contract_descriptor_bytes()
    descriptor = contract.external_execution_contract_descriptor()
    malformed = (
        raw.replace(b'"status":', b'"status":"forged","status":', 1),
        json.dumps(descriptor, indent=2, sort_keys=True).encode("ascii") + b"\n",
        raw.removesuffix(b"\n"),
        raw + b"\n",
        b'{"value":"\xc3\xa9"}\n',
        b'{"value":1.0}\n',
        b'{"value":100000000000000000000}\n',
        b"[]\n",
        b"",
        b"{}\n" * (2 * 1024 * 1024),
    )
    for value in malformed:
        with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
            contract.parse_external_execution_contract_descriptor(value)
    for invalid_input in (None, bytearray(raw), memoryview(raw), raw.decode("ascii")):
        with pytest.raises(contract.ForagerMatchedV3ExternalExecutionContractError):
            contract.parse_external_execution_contract_descriptor(
                invalid_input  # type: ignore[arg-type]
            )


def test_all_public_accessors_fail_on_live_source_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = contract._bounded_source_sha256

    def drift(module_file: object, expected_suffix: str) -> str:
        if expected_suffix.endswith("forager_matched_v3_external_materialization.py"):
            return "0" * 64
        return original(module_file, expected_suffix)

    monkeypatch.setattr(contract, "_bounded_source_sha256", drift)
    raw = contract._DESCRIPTOR_BYTES
    accessors = (
        contract.external_execution_contract_descriptor,
        contract.canonical_external_execution_contract_descriptor_bytes,
        contract.external_execution_contract_descriptor_sha256,
        lambda: contract.parse_external_execution_contract_descriptor(raw),
        lambda: contract.external_execution_candidate_record("external_dqn_plain"),
    )
    for accessor in accessors:
        with pytest.raises(
            contract.ForagerMatchedV3ExternalExecutionContractError,
            match="materializer v2 source binding drifted",
        ):
            accessor()


def test_live_dependency_identity_and_constant_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            getattr(contract, "_materializer"),
            "PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256",
            "0" * 64,
            "materializer-v2 identity binding drifted",
        ),
        (
            getattr(contract, "_seed_transport"),
            "EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256",
            "0" * 64,
            "external seed-transport binding drifted",
        ),
        (
            getattr(contract, "_result_bridge"),
            "MAX_ZIP_MEMBER_COUNT",
            127,
            "external result-bridge binding drifted",
        ),
    )
    for module, name, value, message in cases:
        with monkeypatch.context() as scoped:
            scoped.setattr(module, name, value)
            with pytest.raises(
                contract.ForagerMatchedV3ExternalExecutionContractError,
                match=message,
            ):
                contract.external_execution_contract_descriptor()


def test_live_result_bridge_candidate_reordering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = getattr(contract, "_result_bridge")
    original = tuple(bridge.EXTERNAL_RESULT_CANDIDATE_IDS)
    reordered = (original[1], original[0], *original[2:])
    assert set(reordered) == set(original)
    assert reordered != original

    monkeypatch.setattr(bridge, "EXTERNAL_RESULT_CANDIDATE_IDS", reordered)
    with pytest.raises(
        contract.ForagerMatchedV3ExternalExecutionContractError,
        match="external result-bridge binding drifted",
    ):
        contract.external_execution_contract_descriptor()


def test_live_dependency_sources_have_the_exact_bound_hashes() -> None:
    root = Path(contract.__file__).resolve().parents[2]
    expected = {
        "alberta_framework/benchmarks/forager_matched_v3_configuration_plan.py": (
            _CONFIGURATION_PLAN_SOURCE_SHA256
        ),
        "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py": (
            _MATERIALIZER_V2_SOURCE_SHA256
        ),
        "alberta_framework/benchmarks/forager_matched_v3_external_seed_transport.py": (
            _SEED_TRANSPORT_SOURCE_SHA256
        ),
        "alberta_framework/benchmarks/_forager_matched_v3_external_result_bridge.py": (
            _RESULT_BRIDGE_SOURCE_SHA256
        ),
    }
    for relative, expected_sha256 in expected.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected_sha256


def test_static_surface_has_no_runner_workload_dependency_import_or_authorizing_api() -> None:
    source = inspect.getsource(contract)
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    forbidden_import_fragments = (
        "subprocess",
        "configuration_plan",
    )
    assert not any(
        fragment in imported for imported in imports for fragment in forbidden_import_fragments
    )
    public_names = set(contract.__all__)
    assert not any(
        fragment in name.lower()
        for name in public_names
        for fragment in (
            "run_",
            "workload",
            "publish",
            "accept_result",
            "issue_seed",
            "issue_capability",
            "open_path",
            "materialize",
        )
    )
    for name in (
        "run",
        "execute",
        "open_workload",
        "publish",
        "accept_result",
        "accept_seed",
        "issue_seed",
        "issue_capability",
        "from_path",
        "DEFAULT_INPUT",
    ):
        assert not hasattr(contract, name)
