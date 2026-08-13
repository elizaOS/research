"""Tests for the unexecuted matched-v3 configuration plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as adapter_reward_bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_publication as adapter_reward_publication,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_candidate_universe as universe,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_configuration_plan as plan,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as external_materialization,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_seed_transport as external_seed_transport,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_foragax_bridge as foragax_bridge,
)
from alberta_framework.benchmarks import forager_matched_v3_full_rainbow as full_rainbow
from alberta_framework.benchmarks import (
    forager_matched_v3_full_rainbow_runner as full_rainbow_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_configuration as local_configuration,
)
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru
from alberta_framework.benchmarks import (
    forager_matched_v3_ppo_gru_runner as ppo_gru_runner,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_protocol as protocol,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_EXTERNAL_IDENTITIES = {
    "external_dqn_plain": (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json",
        "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
        "1d8a711ee1e4db575cb0edcacbaf38f97bd06cddc24019eb64b8c410e84b4e85",
    ),
    "external_dqn_crelu": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_CReLU.json",
        "d433b87789e180df3f153cebdafa53f3b6278325fcd32889c8959552cecfeda0",
        "ef92352b97d92e7d40458db48157f589b0d0984f2f4286947c9a1f28bd522892",
    ),
    "external_dqn_redo": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
        "DQN_ReDo_PostLNScore.json",
        "61fa39de8426e2fb78305846b26f6c7a977c72b9cc8a61fc70419f8c15afc8ab",
        "c38288f2ddb6a5dd8892954b499370d04399ec41e966fe790643c9d64b5ffc54",
    ),
    "external_dqn_reward_trace": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
        "DQN_reward_trace.json",
        "3d14f03bc22eec14e4abcc32e635c1dbfa83d4149ef2eaca3609ddba3281ffcb",
        "8641a3b4673940f5519f074b617ccc58a6c14b61a8b448df434cebb3d5f4c974",
    ),
    "external_dqn_l2_init": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_L2_Init.json",
        "6a90d4e970c66d0cc968c9988e0e91a3341fdcb2126954a1b7314f7154b53934",
        "2a2a1dc503b0617c35c202027a646db32186e2668d4b8988215f516a036b9107",
    ),
    "external_pt_dqn_xfinal": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json",
        "4f2ff117d4b82458e3a4bb373d54d03d5b1fedeb4d0b25214235facb5ff2b690",
        "05eaad6da93d8c42d8bd60da3d6c3728bca5c653608eb98210a48a76bedce2e2",
    ),
    "external_drqn_xfinal": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "70a5ee902aa6128ec65c6d4fd33e27da0e3eaa02bd4ea8b776baf3fa158c27de",
        "2b0e177420a9f9a4c8a7bd7aede9c7d2c5add3da4c8b3e301f32bb2588637047",
    ),
    "isolated_ppo_generic": (
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/ActorCriticMLP.json",
        "c8915481c67045339de4b013372d2538eafa91b21c639d2fb0e08d0c60865228",
        "27ffdffcf3ff3e722be5cdfe58d6bc07348ebe5380478032eedfaf435b754c71",
    ),
    "isolated_rtu_paper_scale": (
        "experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/"
        "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json",
        "b9e7bf1bfa307239df848677b6ad4e7c76ef316567b11f75e9455625efc20e65",
        "c32e240bf8c78cf2c7d1ad958bbfc8975b55160fb09490401763a346c2a21090",
    ),
    "random_policy": (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Random.json",
        "24b9d17d2fa4d5da0dc9afd24bbd605fdd4e7574a70f13dc9648e6e6412f6a9a",
        "d20dc9294baab331c4658e4c682d5e1eee3c6f7cc6baf5d17586f48362e8936d",
    ),
    "search_nearest": (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Nearest.json",
        "2c2f67b13f818c7a639411e491095f04dbf3e789a1197c40a6a659ef26e0238d",
        "97b644c4c625155ae16fa7b69432ea0774f767142cc0e28b3d6fcec18c17d2ab",
    ),
    "search_oracle": (
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Oracle.json",
        "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89",
    ),
}

_LOCAL_HASHES = {
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
    "alberta_rtu_h08_taylor": "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792",
}


@pytest.mark.unit
def test_plan_binds_exact_candidate_order_and_static_protocols() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()

    assert descriptor["schema_version"] == plan.CONFIGURATION_PLAN_SCHEMA_VERSION
    assert tuple(record["candidate_id"] for record in descriptor["candidates"]) == (
        universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
    )
    assert descriptor["bindings"]["candidate_universe_sha256"] == (
        universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
    )
    assert descriptor["bindings"]["candidate_universe_sha256"] == (
        "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
    )
    assert descriptor["bindings"]["cumulative_reward_metric_sha256"] == (
        protocol.CUMULATIVE_REWARD_METRIC_SHA256
    )
    assert descriptor["bindings"]["trial_block_generator_plan_sha256"] == (
        protocol.TRIAL_BLOCK_GENERATOR_PLAN_SHA256
    )
    assert descriptor["task"] == {
        "environment_id": protocol.MATCHED_V3_ENVIRONMENT_ID,
        "observation_type": protocol.MATCHED_V3_OBSERVATION_TYPE,
        "aperture_size": protocol.MATCHED_V3_APERTURE_SIZE,
        "horizon": protocol.MATCHED_V3_HORIZON,
    }
    assert descriptor["execution_ready"] is False
    assert descriptor["execution_authorized"] is False
    assert descriptor["scientific_promotion_allowed"] is False
    assert all(record["execution_ready"] is False for record in descriptor["candidates"])
    dopamine = next(
        item
        for item in descriptor["source_repositories"]
        if item["repository_id"] == "dopamine"
    )
    paths = {item["path"] for item in dopamine["relevant_files"]}
    assert "dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin" in paths
    assert "dopamine/jax/agents/full_rainbow/full_rainbow.gin" not in paths


@pytest.mark.unit
def test_canonical_plan_is_content_addressed_and_snapshots_are_detached() -> None:
    canonical = plan.canonical_matched_v3_configuration_plan_bytes()
    assert hashlib.sha256(canonical).hexdigest() == plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256
    assert plan.matched_v3_configuration_plan_sha256() == (
        plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256
    )

    first = plan.matched_v3_configuration_plan_descriptor()
    first["bindings"]["candidate_universe_sha256"] = "0" * 64
    first["candidates"][0]["execution_shape"]["horizon"] = 1
    first["candidates"].append({"candidate_id": "injected"})
    second = plan.matched_v3_configuration_plan_descriptor()

    assert second["bindings"]["candidate_universe_sha256"] == (
        universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
    )
    assert second["candidates"][0]["execution_shape"]["horizon"] == 499_712
    assert len(second["candidates"]) == 28


@pytest.mark.unit
def test_artifact_parser_rejects_any_readiness_or_authority_mutation() -> None:
    canonical = plan.canonical_matched_v3_configuration_plan_bytes()
    assert plan.parse_matched_v3_configuration_plan_artifact(canonical) == (
        plan.matched_v3_configuration_plan_descriptor()
    )

    mutated = json.loads(canonical)
    mutated["execution_ready"] = True
    mutated["execution_authorized"] = True
    raw = json.dumps(mutated, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="frozen digest"):
        plan.parse_matched_v3_configuration_plan_artifact(raw)
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="must be bytes"):
        plan.parse_matched_v3_configuration_plan_artifact("{}")  # type: ignore[arg-type]


@pytest.mark.unit
def test_per_candidate_records_are_detached_and_reject_unknown_or_type_aliases() -> None:
    record = plan.configuration_record("external_dqn_plain")
    record["configuration"]["derived_sha256"] = "0" * 64

    assert plan.configuration_record("external_dqn_plain")["configuration"][
        "derived_sha256"
    ] == _EXTERNAL_IDENTITIES["external_dqn_plain"][2]
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="unknown candidate"):
        plan.configuration_record("not-a-candidate")
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="candidate_id"):
        plan.configuration_record(True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_all_local_envelopes_have_exact_builder_and_digest_bindings() -> None:
    built_by_id = {
        candidate_id: plan.build_and_verify_local_configuration(candidate_id)
        for candidate_id in _LOCAL_HASHES
    }
    for candidate_id, expected_sha256 in _LOCAL_HASHES.items():
        configuration = plan.configuration_record(candidate_id)["configuration"]
        assert configuration["kind"] == "generated_local"
        assert configuration["builder_id"].startswith(
            "alberta.forager_matched_v3.generated_local."
        )
        assert configuration["builder_id"] == built_by_id[candidate_id].builder_id
        assert configuration["worker_envelope_sha256"] == expected_sha256
        assert configuration["worker_envelope_sha256"] == (
            built_by_id[candidate_id].configuration_sha256
        )
        assert configuration["configuration_complete"] is True
        assert configuration["source_snapshot_status"] == "unqualified_current_checkout"
        assert configuration["builder_status"] == "implemented_unqualified"
        assert configuration["source_descriptor_sha256"] == (
            local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        )
        assert configuration["builder_descriptor_sha256"] == (
            local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
        )
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="no local configuration builder",
    ):
        plan.build_and_verify_local_configuration("external_dqn_plain")
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="candidate_id"):
        plan.build_and_verify_local_configuration(True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_all_external_records_bind_exact_paths_raw_and_derived_hashes() -> None:
    assert plan.EXTERNAL_CONFIGURATION_CANDIDATE_IDS == tuple(_EXTERNAL_IDENTITIES)
    for candidate_id, expected in _EXTERNAL_IDENTITIES.items():
        configuration = plan.configuration_record(candidate_id)["configuration"]
        assert configuration["kind"] == "derived_external"
        assert (
            configuration["original_path"],
            configuration["original_sha256"],
            configuration["derived_sha256"],
        ) == expected
        assert configuration["configuration_complete"] is True
        assert configuration["transform_descriptor"]["schema_version"] == (
            "alberta.forager_matched_v3_configuration_transform.v1"
        )


@pytest.mark.unit
def test_transform_bundles_bind_types_expected_values_order_and_digests() -> None:
    plain = plan.configuration_record("external_dqn_plain")["configuration"]
    assert plain["transform_descriptor_sha256"] == (
        "d85d2fec4fa18d3ab749f57a0a0b240daf57e05c3cd329bb08d17aac48b5ffeb"
    )
    assert plain["transform_descriptor"]["transforms"] == [
        {
            "pointer": "/metaParameters/experiment/ntk_freq",
            "value_type": "integer",
            "expected_original": 2500,
            "replacement": 0,
        },
        {
            "pointer": "/metaParameters/experiment/x_ref_steps",
            "value_type": "integer",
            "expected_original": 100,
            "replacement": 0,
        },
        {
            "pointer": "/total_steps",
            "value_type": "integer",
            "expected_original": 10000,
            "replacement": 499712,
        },
    ]

    xfinal = plan.configuration_record("external_pt_dqn_xfinal")["configuration"]
    assert xfinal["transform_descriptor_sha256"] == (
        "fd20ddfef5fc160f14a0c47d2acd74335a361b061067fada88dd0ef1b42d1497"
    )
    assert [item["pointer"] for item in xfinal["transform_descriptor"]["transforms"]] == [
        "/metaParameters/environment/env_id",
        "/total_steps",
    ]

    rtu = plan.configuration_record("isolated_rtu_paper_scale")["configuration"]
    assert rtu["transform_descriptor_sha256"] == (
        "68b904bed65ab157edbd323725126810d9fd72d7ccc69685a45eaa2aaba48f3b"
    )
    assert [item["pointer"] for item in rtu["transform_descriptor"]["transforms"]] == [
        "/metaParameters/environment/env_id",
        "/metaParameters/experiment/ntk_freq",
        "/metaParameters/experiment/weight_drift_freq",
        "/metaParameters/experiment/weight_norm_freq",
        "/metaParameters/experiment/x_ref_steps",
        "/total_steps",
    ]


@pytest.mark.unit
def test_adapter_cores_have_exact_configuration_and_descriptor_bindings() -> None:
    expected = {
        "adapted_full_rainbow": {
            "repository_id": "dopamine",
            "configuration_schema_version": (
                full_rainbow.FULL_RAINBOW_CONFIG_SCHEMA_VERSION
            ),
            "configuration_sha256": full_rainbow.FULL_RAINBOW_CONFIG_SHA256,
            "adapter_descriptor_schema_version": (
                full_rainbow.FULL_RAINBOW_DESCRIPTOR_SCHEMA_VERSION
            ),
            "adapter_descriptor_sha256": (
                full_rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256
            ),
            "implementation_path": (
                "alberta_framework/benchmarks/forager_matched_v3_full_rainbow.py"
            ),
            "implementation_source_sha256": (
                "7f75a0862ddc21160cea9c0a9faca221a0d757985fc90e5ef02b4673e3c14f5a"
            ),
            "runner_descriptor_schema_version": (
                full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "runner_descriptor_sha256": (
                full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
            ),
            "runner_implementation_path": (
                "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
            ),
            "runner_implementation_source_sha256": (
                "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
            ),
            "runner_result_receipt_schema_version": (
                full_rainbow_runner.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
            ),
        },
        "adapted_ppo_gru": {
            "repository_id": "pobax",
            "configuration_schema_version": (
                ppo_gru.PPO_GRU_CONFIGURATION_SCHEMA_VERSION
            ),
            "configuration_sha256": ppo_gru.PPO_GRU_CONFIGURATION_SHA256,
            "adapter_descriptor_schema_version": (
                ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "adapter_descriptor_sha256": ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256,
            "implementation_path": (
                "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py"
            ),
            "implementation_source_sha256": (
                "58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e"
            ),
            "runner_descriptor_schema_version": (
                ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION
            ),
            "runner_descriptor_sha256": (
                ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
            ),
            "runner_implementation_path": (
                "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
            ),
            "runner_implementation_source_sha256": (
                "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
            ),
            "runner_result_receipt_schema_version": (
                ppo_gru_runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
            ),
        },
    }
    for candidate_id, identities in expected.items():
        configuration = plan.configuration_record(candidate_id)["configuration"]
        assert configuration == {
            "kind": "derived_local_adapter",
            **identities,
            "status": "implemented_unqualified_core_runner_and_result_conversion",
            "configuration_complete": True,
            "core_implementation_complete": True,
            "full_runner_complete": True,
            "in_memory_result_conversion_complete": True,
            "runtime_qualified": False,
            "durable_result_publication_complete": False,
            "upstream_review_anchors_bound": True,
            "source_closure_bound": False,
            "source_snapshot_status": "core_and_runner_sources_bound_unqualified",
        }
        source = _REPOSITORY_ROOT / identities["implementation_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == identities[
            "implementation_source_sha256"
        ]
        runner_source = _REPOSITORY_ROOT / identities["runner_implementation_path"]
        assert hashlib.sha256(runner_source.read_bytes()).hexdigest() == identities[
            "runner_implementation_source_sha256"
        ]

    plan.assert_configuration_complete()


@pytest.mark.unit
def test_adapter_core_artifact_verifier_replays_both_frozen_contracts() -> None:
    for candidate_id in ("adapted_full_rainbow", "adapted_ppo_gru"):
        assert plan.verify_adapter_core_artifacts(candidate_id) == (
            plan.configuration_record(candidate_id)["configuration"]
        )
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="no adapter-core artifacts",
    ):
        plan.verify_adapter_core_artifacts("external_dqn_plain")
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="candidate_id"):
        plan.verify_adapter_core_artifacts(True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_adapter_runner_verifier_replays_both_full_horizon_contracts() -> None:
    for candidate_id in ("adapted_full_rainbow", "adapted_ppo_gru"):
        configuration = plan.configuration_record(candidate_id)["configuration"]
        assert plan.verify_adapter_runner_artifact(candidate_id) == {
            key: configuration[key]
            for key in (
                "runner_descriptor_schema_version",
                "runner_descriptor_sha256",
                "runner_implementation_path",
                "runner_implementation_source_sha256",
                "runner_result_receipt_schema_version",
                "full_runner_complete",
                "runtime_qualified",
                "durable_result_publication_complete",
            )
        }
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="no adapter runner artifact",
    ):
        plan.verify_adapter_runner_artifact("external_dqn_plain")
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="candidate_id"):
        plan.verify_adapter_runner_artifact(True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_external_materializer_identity_is_bound_but_unexecuted() -> None:
    binding = plan.matched_v3_configuration_plan_descriptor()["external_materializer"]
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="historical v1 external materializer is superseded and unavailable",
    ):
        plan.verify_external_materialization_identity_artifact()
    assert binding == {
        "manifest_schema_version": "alberta.forager_matched_v3_external_materialization.v1",
        "identity_schema_version": (
            "alberta.forager_matched_v3_external_materialization_identity.v1"
        ),
        "identity_sha256": (
            "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
        ),
        "implementation_path": (
            "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py"
        ),
        "implementation_source_sha256": (
            "5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"
        ),
        "status": "implemented_unexecuted",
        "production_materialization_accepted": False,
        "production_manifest_sha256": None,
        "materialized_source_closure_bound": False,
        "archive_bytes_verified": False,
        "runtime_dependencies_qualified": False,
        "execution_authorized": False,
    }
    assert external_materialization.EXTERNAL_MATERIALIZATION_SCHEMA_VERSION.endswith(".v2")
    assert (
        external_materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
        != binding["identity_sha256"]
    )


@pytest.mark.unit
def test_adapter_result_conversion_is_source_bound_but_not_published() -> None:
    binding = plan.matched_v3_configuration_plan_descriptor()[
        "adapter_result_conversion"
    ]
    assert plan.verify_adapter_reward_bundle_descriptor() == binding
    assert binding["descriptor_schema_version"] == (
        adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
    )
    assert binding["descriptor_sha256"] == (
        adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
    )
    assert binding["scorer_source_sha256"] == (
        "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
    )
    assert binding["canonical_reward_artifact_size_bytes"] == (
        scorer.CANONICAL_NPZ_SIZE_BYTES
    )
    assert binding["in_memory_conversion_complete"] is True
    assert binding["filesystem_publication_complete"] is False
    assert binding["campaign_ingestion_complete"] is False
    assert binding["ingestion_authorized"] is False


@pytest.mark.unit
def test_adapter_result_publication_is_implemented_but_not_accepted() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()
    binding = descriptor["adapter_result_publication"]
    assert plan.verify_adapter_reward_publication_descriptor() == binding
    assert binding == {
        "descriptor_schema_version": (
            adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        ),
        "descriptor_sha256": (
            "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
        ),
        "publication_schema_version": (
            adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
        ),
        "implementation_path": (
            "alberta_framework/benchmarks/"
            "forager_matched_v3_adapter_reward_publication.py"
        ),
        "implementation_source_sha256": (
            "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
        ),
        "status": "implemented_unexecuted",
        "candidate_consumers": ["adapted_full_rainbow", "adapted_ppo_gru"],
        "adapter_reward_bundle_descriptor_schema_version": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "adapter_reward_bundle_descriptor_sha256": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
        ),
        "adapter_reward_bundle_manifest_schema_version": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        ),
        "implementation_complete": True,
        "production_publication_accepted": False,
        "production_publication_file_sha256": None,
        "campaign_ingestion_complete": False,
        "ingestion_authorized": False,
        "runtime_qualified": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
    }
    source = _REPOSITORY_ROOT / binding["implementation_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == binding[
        "implementation_source_sha256"
    ]
    assert binding["descriptor_sha256"] == (
        adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
    )
    assert binding["production_publication_accepted"] is False
    assert binding["production_publication_file_sha256"] is None
    assert descriptor["adapter_result_conversion"][
        "filesystem_publication_complete"
    ] is False
    for candidate_id in binding["candidate_consumers"]:
        assert plan.configuration_record(candidate_id)["configuration"][
            "durable_result_publication_complete"
        ] is False


@pytest.mark.unit
def test_publication_verifier_replays_dependency_and_ignores_mutable_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutable = plan._MATCHED_V3_CONFIGURATION_PLAN["adapter_result_publication"]
    monkeypatch.setitem(mutable, "production_publication_accepted", True)
    verified = plan.verify_adapter_reward_publication_descriptor()
    assert verified["production_publication_accepted"] is False

    original_parser = (
        adapter_reward_publication.parse_adapter_reward_publication_descriptor
    )

    def parse_with_dependency_drift(raw: bytes) -> dict[str, Any]:
        descriptor = original_parser(raw)
        descriptor["dependency"]["descriptor_sha256"] = "0" * 64
        return descriptor

    monkeypatch.setattr(
        adapter_reward_publication,
        "parse_adapter_reward_publication_descriptor",
        parse_with_dependency_drift,
    )
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="publication descriptor binding drift",
    ):
        plan.verify_adapter_reward_publication_descriptor()


@pytest.mark.unit
def test_shared_environment_bridge_is_content_bound_but_unqualified() -> None:
    binding = plan.matched_v3_configuration_plan_descriptor()[
        "shared_environment_bridge"
    ]
    assert plan.verify_shared_environment_bridge_artifact() == binding
    assert binding == {
        "schema_version": foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_sha256": foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
        "implementation_path": (
            "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
        ),
        "implementation_source_sha256": (
            "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
        ),
        "status": "implemented_unqualified",
        "adapter_consumers": ["adapted_full_rainbow", "adapted_ppo_gru"],
        "environment_rng_schedule": "dedicated_environment_split_chain_v1",
        "runtime_parity_executed": False,
        "runtime_qualified": False,
        "compiled_chunk_kernel_complete": False,
        "source_closure_bound": False,
    }
    source = _REPOSITORY_ROOT / binding["implementation_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == binding[
        "implementation_source_sha256"
    ]


@pytest.mark.unit
def test_exact_update_and_non_gradient_operation_accounting_is_bound() -> None:
    plain = plan.configuration_record("external_dqn_plain")["execution_shape"]
    assert plain["optimizer_update_count"] == 124_920
    assert plain["non_gradient_operations"]["target_snapshot_refreshes"] == 975

    redo = plan.configuration_record("external_dqn_redo")["execution_shape"]
    assert redo["optimizer_update_count"] == 124_915
    assert redo["non_gradient_operations"]["redo_recycles"] == 49

    pt = plan.configuration_record("external_pt_dqn_xfinal")["execution_shape"]
    assert pt["optimizer_update_count"] == 140_154
    assert pt["optimizer_update_subcounts"] == {
        "transient": 124_915,
        "permanent": 15_239,
    }
    assert pt["non_gradient_operations"]["permanent_update_events"] == 49
    assert pt["non_gradient_operations"]["transient_parameter_decays"] == 49
    assert pt["replay_capacity_transitions"] == {
        "main": 1000,
        "permanent": 10000,
    }

    causal = plan.configuration_record("causal_e025_q050")["execution_shape"]
    assert causal["optimizer_update_count"] == 0
    assert causal["non_gradient_operations"]["causal_transition_updates"] == 499_712

    horde = plan.configuration_record("alberta_horde_default")["execution_shape"]
    assert horde["optimizer_update_count"] == 999_424
    assert horde["optimizer_update_subcounts"] == {
        "actor": 499_712,
        "critic": 499_712,
    }

    local_rtu = plan.configuration_record("alberta_rtu_h08_taylor")["execution_shape"]
    assert local_rtu["optimizer_update_count"] == 999_424
    assert local_rtu["optimizer_update_subcounts"] == {
        "actor": 499_712,
        "critic": 499_712,
    }

    rainbow = plan.configuration_record("adapted_full_rainbow")["execution_shape"]
    assert rainbow["entrypoint_path"] == (
        "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
    )
    assert rainbow["entrypoint_sha256"] == (
        "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
    )
    assert rainbow["interaction_count"] == 499_712
    assert rainbow["optimizer_update_count"] == 119_928
    assert rainbow["optimizer_update_subcounts"] == {
        "joint_distributional_q": 119_928
    }
    assert rainbow["replay_capacity_transitions"] == {
        "main": 1_000_000,
        "permanent": 0,
    }
    assert rainbow["non_gradient_operations"]["target_snapshot_refreshes"] == 60
    assert rainbow["exact_workload_argv"] is None

    rainbow_rng = plan.configuration_record("adapted_full_rainbow")["rng_contract"]
    assert rainbow_rng["environment_seed_transport"] == (
        "shared_foragax_bridge_uint31_direct_threefry2x32_split_chain"
    )
    assert rainbow_rng["agent_seed_transport"] == (
        "full_rainbow_agent_uint31_threefry2x32_folded_namespace"
    )
    assert rainbow_rng["statistical_independence_claimed"] is False

    recurrent_ppo = plan.configuration_record("adapted_ppo_gru")["execution_shape"]
    assert recurrent_ppo["entrypoint_path"] == (
        "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
    )
    assert recurrent_ppo["entrypoint_sha256"] == (
        "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
    )
    assert recurrent_ppo["interaction_count"] == 499_712
    assert recurrent_ppo["rollout_steps"] == 512
    assert recurrent_ppo["rollout_count"] == 976
    assert recurrent_ppo["epochs"] == 4
    assert recurrent_ppo["minibatches_per_epoch"] == 4
    assert recurrent_ppo["optimizer_update_count"] == 15_616
    assert recurrent_ppo["optimizer_update_subcounts"] == {
        "joint_policy_value_recurrent_segment": 15_616
    }
    assert recurrent_ppo["exact_workload_argv"] is None

    rng_contract = plan.configuration_record("adapted_ppo_gru")["rng_contract"]
    assert rng_contract["environment_seed_transport"] == (
        "shared_foragax_bridge_uint31_direct_threefry2x32_split_chain"
    )
    assert rng_contract["agent_seed_transport"] == (
        "ppo_gru_agent_uint31_threefry2x32_split_chain"
    )
    assert rng_contract["statistical_independence_claimed"] is False


@pytest.mark.unit
@pytest.mark.parametrize("candidate_id", ["isolated_ppo_generic", "isolated_rtu_paper_scale"])
def test_ppo_exact_244_update_invocation_rejects_upstream_245_default(
    candidate_id: str,
) -> None:
    shape = plan.configuration_record(candidate_id)["execution_shape"]
    assert shape["rollout_steps"] == 2048
    assert shape["rollout_count"] == 244
    assert shape["optimizer_update_count"] == 31_232
    assert shape["exact_workload_argv"] == [
        "--exp",
        "<derived_configuration_path>",
        "--idxs",
        "<exactly_one_index>",
        "--environment_seed",
        "<environment_seed_uint31>",
        "--agent_seed",
        "<candidate_private_agent_seed_uint31>",
        "--max_steps",
        "244",
        "--save_path",
        "<candidate_private_results_root>",
        "--checkpoint_path",
        "<new_empty_candidate_private_checkpoint_root>",
        "--silent",
    ]
    assert shape["default_without_override"] == {
        "rollout_count": 245,
        "interaction_count": 501_760,
        "accepted": False,
    }
    assert shape["interaction_count"] == 499_712


@pytest.mark.unit
def test_diagnostic_and_save_policy_is_fail_closed_and_records_video_caveat() -> None:
    plain = plan.configuration_record("external_dqn_plain")["diagnostic_policy"]
    assert plain["ntk_enabled"] is False
    assert plain["video_length_steps"] == 0
    assert plain["checkpoint_milestones_within_horizon"] == 0

    redo = plan.configuration_record("external_dqn_redo")["diagnostic_policy"]
    assert redo["save_every_steps"] == 1_000_000
    assert redo["checkpoint_milestones_within_horizon"] == 0

    rtu = plan.configuration_record("isolated_rtu_paper_scale")["diagnostic_policy"]
    assert rtu["ntk_enabled"] is False
    assert rtu["configured_reference_steps"] == 0
    assert rtu["chunked_ref_source_value"] == 256
    assert rtu["chunked_ref_effective_value"] == 1
    assert rtu["chunked_ref_inert"] is True
    assert rtu["weight_norm_enabled"] is False
    assert rtu["weight_drift_enabled"] is False
    assert rtu["video_length_steps"] == 1_000
    assert rtu["single_index_video_capture_unavoidable_without_source_derivation"] is True


@pytest.mark.unit
def test_rng_and_source_snapshot_blockers_keep_every_candidate_unready() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()
    blocker_ids = {item["blocker_id"] for item in descriptor["readiness_blockers"]}
    assert {
        "environment_rng_independence_unqualified",
        "accepted_external_materialization_manifest_missing",
        "accepted_external_materialized_source_closure_missing",
        "external_seed_transport_runtime_unqualified",
        "local_v3_source_snapshot_missing",
        "adapter_source_closure_and_qualification_missing",
        "adapter_full_horizon_resource_profiles_unqualified",
        "adapter_environment_bridge_runtime_parity_and_compiled_kernel_unqualified",
        "adapter_production_publication_acceptance_and_campaign_ingestion_missing",
    } <= blocker_ids
    for record in descriptor["candidates"]:
        assert record["rng_contract"]["environment_seed_namespace"] == "environment"
        assert record["rng_contract"]["agent_seed_namespace"] == (
            f"agent/{record['candidate_id']}"
        )
        if record["candidate_id"] in _LOCAL_HASHES:
            expected_status = "implemented_unqualified_local_api"
        elif record["candidate_id"] in _EXTERNAL_IDENTITIES:
            expected_status = "implemented_unqualified_external_patch_set"
        elif record["candidate_id"] in {
            "adapted_full_rainbow",
            "adapted_ppo_gru",
        }:
            expected_status = "implemented_unqualified_core_bridge_and_full_runner_apis"
        else:
            expected_status = "unimplemented"
        assert record["rng_contract"]["transport_status"] == expected_status
        if record["candidate_id"] == "adapted_full_rainbow":
            assert record["rng_contract"]["environment_transport_descriptor_sha256"] == (
                foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
            )
            assert record["rng_contract"]["agent_transport_descriptor_sha256"] == (
                full_rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256
            )
            assert record["rng_contract"]["runner_transport_descriptor_sha256"] == (
                full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
            )
        if record["candidate_id"] == "adapted_ppo_gru":
            assert record["rng_contract"]["environment_transport_descriptor_sha256"] == (
                foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
            )
            assert record["rng_contract"]["agent_transport_descriptor_sha256"] == (
                ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
            )
            assert record["rng_contract"]["runner_transport_descriptor_sha256"] == (
                ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
            )
        assert record["rng_contract"]["runtime_trace_verified"] is False
        assert record["rng_contract"]["source_closure_bound"] is False
        assert record["rng_contract"]["statistical_independence_claimed"] is False
        assert record["execution_ready"] is False

    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="not execution-ready"):
        plan.assert_execution_ready()


@pytest.mark.unit
def test_external_execution_shapes_bind_derived_two_seed_patch_set() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()
    source = descriptor["upstream_execution_source"]

    assert source["seed_transport_patch_set_descriptor_sha256"] == (
        external_seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
    )
    assert source["seed_transport_patch_set_scope"] == "derived_patch_set_only"
    assert source["full_dependency_inventory_bound"] is False
    assert source["materialization_status"] == (
        "implementation_available_no_accepted_production_manifest"
    )
    assert source["runtime_trace_verified"] is False
    assert source["derived_source_files"] == [
        {
            "path": path,
            "sha256": external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[
                path
            ],
        }
        for path in external_seed_transport.SOURCE_PATHS
    ]

    continuing_ids = set(_EXTERNAL_IDENTITIES) - {
        "isolated_ppo_generic",
        "isolated_rtu_paper_scale",
    }
    for candidate_id in _EXTERNAL_IDENTITIES:
        record = plan.configuration_record(candidate_id)
        shape = record["execution_shape"]
        expected_path = (
            "src/continuing_main.py"
            if candidate_id in continuing_ids
            else "src/rtu_ppo.py"
        )
        assert shape["entrypoint_path"] == expected_path
        assert shape["entrypoint_sha256"] == (
            external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[
                expected_path
            ]
        )
        argv = shape["exact_workload_argv"]
        assert argv[argv.index("--environment_seed") + 1] == (
            "<environment_seed_uint31>"
        )
        assert argv[argv.index("--agent_seed") + 1] == (
            "<candidate_private_agent_seed_uint31>"
        )
        assert record["rng_contract"]["environment_transport_descriptor_sha256"] == (
            external_seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
        )
        assert record["rng_contract"]["agent_transport_descriptor_sha256"] == (
            external_seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
        )


@pytest.mark.unit
def test_external_derivation_rejects_wrong_bytes_and_v2_substitutes() -> None:
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="raw SHA-256"):
        plan.derive_and_verify_external_configuration(
            "external_pt_dqn_xfinal",
            b'{"agent":"PT_DQN","total_steps":499712}',
        )
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="external"):
        plan.derive_and_verify_external_configuration("causal_e025_q050", b"{}")
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="bytes"):
        plan.derive_and_verify_external_configuration(
            "external_dqn_plain",
            '{"agent":"DQN"}',  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_external_seed_transport_wrapper_rejects_an_incomplete_source_set() -> None:
    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="two-seed patch-set derivation failed",
    ):
        plan.derive_and_verify_external_seed_transport({})


@pytest.mark.unit
def test_external_set_requires_exact_keys_before_derivation() -> None:
    values = {candidate_id: b"{}" for candidate_id in plan.EXTERNAL_CONFIGURATION_CANDIDATE_IDS}
    del values["external_dqn_plain"]
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="missing"):
        plan.verify_external_configuration_set(values)

    values["external_dqn_plain"] = b"{}"
    values["adapted_full_rainbow"] = b"{}"
    with pytest.raises(plan.ForagerMatchedV3ConfigurationPlanError, match="extra"):
        plan.verify_external_configuration_set(values)


@pytest.mark.unit
def test_plan_snapshot_contains_only_plain_json_values() -> None:
    def visit(value: Any) -> None:
        assert value is None or type(value) in {dict, list, str, int, bool}
        if type(value) is dict:
            assert all(type(key) is str for key in value)
            for child in value.values():
                visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)

    visit(plan.matched_v3_configuration_plan_descriptor())


@pytest.mark.unit
def test_plan_digest_has_an_independent_literal_pin() -> None:
    assert plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256 == (
        "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
    )


@pytest.mark.unit
def test_nested_plan_shape_rejects_extra_fields_and_boolean_aliases() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["candidates"][0]["execution_shape"]["authority_override"] = True
    with pytest.raises(AssertionError, match="key drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["candidates"][0]["configuration"]["configuration_complete"] = 1
    with pytest.raises(AssertionError, match="configuration_complete"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["readiness_blockers"][0]["authority_override"] = True
    with pytest.raises(AssertionError, match="key drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["readiness_blockers"][0]["candidate_ids"] = [
        "adapted_full_rainbow",
        "adapted_full_rainbow",
    ]
    with pytest.raises(AssertionError, match="candidate drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["external_materializer"]["execution_authorized"] = True
    with pytest.raises(AssertionError, match="materializer binding drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["adapter_result_conversion"]["ingestion_authorized"] = True
    with pytest.raises(AssertionError, match="result conversion binding drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["adapter_result_publication"]["production_publication_accepted"] = True
    with pytest.raises(AssertionError, match="result publication binding drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["adapter_result_publication"]["implementation_complete"] = 1
    with pytest.raises(AssertionError, match="result publication binding drift"):
        plan._validate_plan(descriptor)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    descriptor["adapter_result_publication"]["authority_override"] = True
    with pytest.raises(AssertionError, match="key drift"):
        plan._validate_plan(descriptor)


@pytest.mark.unit
def test_readiness_gate_cannot_bypass_false_execution_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(plan._MATCHED_V3_CONFIGURATION_PLAN, "readiness_blockers", [])
    monkeypatch.setitem(plan._MATCHED_V3_CONFIGURATION_PLAN, "execution_ready", True)
    monkeypatch.setitem(plan._MATCHED_V3_CONFIGURATION_PLAN, "execution_authorized", True)
    for record in plan._MATCHED_V3_CONFIGURATION_PLAN["candidates"]:
        monkeypatch.setitem(record, "execution_ready", True)
        monkeypatch.setitem(record, "execution_authorized", True)

    with pytest.raises(
        plan.ForagerMatchedV3ConfigurationPlanError,
        match="blockers=",
    ):
        plan.assert_execution_ready()


@pytest.mark.unit
def test_configuration_gate_uses_frozen_complete_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan.assert_configuration_complete()
    for record in plan._MATCHED_V3_CONFIGURATION_PLAN["candidates"]:
        monkeypatch.setitem(record["configuration"], "configuration_complete", False)
    monkeypatch.setitem(plan._MATCHED_V3_CONFIGURATION_PLAN, "configuration_complete", False)

    plan.assert_configuration_complete()
    assert plan.matched_v3_configuration_plan_descriptor()[
        "configuration_complete"
    ] is True


@pytest.mark.unit
def test_canonical_read_paths_ignore_mutated_private_construction_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = plan.canonical_matched_v3_configuration_plan_bytes()
    candidate_id = universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS[0]
    mutable_record = plan._MATCHED_V3_CONFIGURATION_PLAN["candidates"][0]

    monkeypatch.setitem(plan._MATCHED_V3_CONFIGURATION_PLAN["task"], "horizon", 1)
    monkeypatch.setitem(mutable_record["execution_shape"], "horizon", 1)

    descriptor = plan.matched_v3_configuration_plan_descriptor()
    parsed = plan.parse_matched_v3_configuration_plan_artifact(raw)
    record = plan.configuration_record(candidate_id)
    assert descriptor["task"]["horizon"] == protocol.MATCHED_V3_HORIZON
    assert parsed["task"]["horizon"] == protocol.MATCHED_V3_HORIZON
    assert record["execution_shape"]["horizon"] == protocol.MATCHED_V3_HORIZON


@pytest.mark.unit
def test_local_and_adapter_replay_ignore_mutated_private_identity_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_candidate = "causal_e025_q050"
    monkeypatch.setitem(plan._LOCAL_CONFIGURATION_SHA256, local_candidate, "0" * 64)
    monkeypatch.setitem(plan._LOCAL_BUILDER_BY_ID, local_candidate, "mutated")
    built = plan.build_and_verify_local_configuration(local_candidate)
    assert built.configuration_sha256 == _LOCAL_HASHES[local_candidate]

    adapter_candidate = "adapted_full_rainbow"
    monkeypatch.setitem(
        plan._ADAPTER_IDENTITIES[adapter_candidate],
        "configuration_sha256",
        "0" * 64,
    )
    verified = plan.verify_adapter_core_artifacts(adapter_candidate)
    assert verified["configuration_sha256"] == full_rainbow.FULL_RAINBOW_CONFIG_SHA256
    monkeypatch.setitem(
        plan._ADAPTER_IDENTITIES[adapter_candidate],
        "runner_descriptor_sha256",
        "0" * 64,
    )
    runner = plan.verify_adapter_runner_artifact(adapter_candidate)
    assert runner["runner_descriptor_sha256"] == (
        full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
    )


@pytest.mark.unit
def test_external_record_reads_ignore_mutated_private_transform_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_id = "external_dqn_plain"
    monkeypatch.setitem(
        plan._EXTERNAL_SPECS[candidate_id],
        "original_sha256",
        "0" * 64,
    )
    assert plan.configuration_record(candidate_id)["configuration"][
        "original_sha256"
    ] == _EXTERNAL_IDENTITIES[candidate_id][1]


@pytest.mark.unit
def test_review_anchor_hashes_match_the_pinned_materialization_identity() -> None:
    descriptor = plan.matched_v3_configuration_plan_descriptor()
    identity = external_materialization.pinned_external_checkout_identity()
    source_hashes = {
        transform.path: transform.upstream_sha256
        for transform in identity.source_transforms
    }
    for anchor in descriptor["upstream_execution_source"]["review_anchors"]:
        assert source_hashes[anchor["path"]] == anchor["sha256"]
