"""Content-addressed, fail-closed historical configuration plan for matched Forager v3.

The plan binds configuration, source-materialization, full-runner, in-memory result
conversion, and atomic content-publication identities without authorizing a workload.
Import-time binding checks read the exact implementation source files but never run a
benchmark, materialize a checkout, publish content, or read a result.  The two
third-party exact-task cores, their shared environment bridge, full-horizon runners,
historical v1 external-materializer binding, strict adapter reward conversion, and adapter
publication writer are content-bound.  The v1 materializer implementation was later
superseded because it could not represent the pinned repository's exact Git tree; this
artifact remains byte-for-byte stable and must be combined with a separately versioned v2
overlay rather than reinterpreted in place.  Runtime and resource qualification,
accepted source closure, an accepted production publication, campaign ingestion, the
local source snapshot, and execution authority remain absent.  Consequently every
candidate is explicitly execution-unready.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import _forager_matched_v3_scorer as scorer
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_bundle as adapter_reward_bundle,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_adapter_reward_publication as adapter_reward_publication,
)
from alberta_framework.benchmarks import forager_matched_v3_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_v3_configuration as derivation
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
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol

CONFIGURATION_PLAN_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_configuration_plan.v1"
)

_MAX_PLAN_BYTES: Final = 2 * 1024 * 1024
_CANDIDATE_UNIVERSE_SHA256: Final = (
    "a441b35eed4ec6327bf03463099a46e9c2596f2a169182fd317fe51c98b4c750"
)
_CUMULATIVE_REWARD_METRIC_SHA256: Final = (
    "ee5ec2dfd0a1647b890817590f7293f3740a8e1b34287b69b562cf864013b3cd"
)
_TRIAL_BLOCK_GENERATOR_PLAN_SHA256: Final = (
    "90fadf6bda3e25c3c6078205fc8e7618e31b4539aae78d6c82ec192aa057eace"
)
_ENVIRONMENT_ID: Final = "ForagaxTwoBiomeLarge-v1"
_OBSERVATION_TYPE: Final = "color"
_APERTURE_SIZE: Final = 9
_HORIZON: Final = 499_712
_UPSTREAM_REPOSITORY_ID: Final = "foragax_agents"
_UPSTREAM_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
_UPSTREAM_TREE: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
_UPSTREAM_ARCHIVE_SHA256: Final = (
    "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
)
_UPSTREAM_ARCHIVE_SIZE: Final = 314_961_920

_CONTINUING_MAIN_SOURCE_SHA256: Final = (
    "681c2dae9569a0bbd72c8f47a3a63d51176071308f9762f3d81855da79c3aebf"
)
_RTU_PPO_SOURCE_SHA256: Final = (
    "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
)
_DERIVED_CONTINUING_MAIN_SOURCE_SHA256: Final = (
    "ca9748cf92107b41c1d1e6cd17d4a1a3c517fa5921c55469c1e66a73ef8d2551"
)
_DERIVED_RTU_PPO_SOURCE_SHA256: Final = (
    "1859b4cde5695fcedd5cd21280caa0df029057e1b90e364f3bace225d127f3f1"
)
_EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256: Final = (
    "66be593917a47c8eca4e1a3227407e060ebb52ac835e4207dc32fc81de7d13ad"
)
_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256: Final = (
    "d15d70b55d965b2c135f1dcaa36a74173e4023e4fdc9430c43660df54f1bb38c"
)
_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256: Final = (
    "1368d3a0c96acd83e82cef75c9d014533dd783d0e6af27714ac47e2f1907840b"
)
_FULL_RAINBOW_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_full_rainbow.py"
)
_FULL_RAINBOW_IMPLEMENTATION_SHA256: Final = (
    "7f75a0862ddc21160cea9c0a9faca221a0d757985fc90e5ef02b4673e3c14f5a"
)
_PPO_GRU_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru.py"
)
_PPO_GRU_IMPLEMENTATION_SHA256: Final = (
    "58c3b853bae51b9791c8121b899a259d60b2586e15b5722a84fac78f4d2c5e1e"
)
_FORAGAX_BRIDGE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_foragax_bridge.py"
)
_FORAGAX_BRIDGE_IMPLEMENTATION_SHA256: Final = (
    "5aa304ee2ec185d038038fdd3e5cd093ecda85507ab7ee5e733ff1a47b21e362"
)
_FORAGAX_BRIDGE_DESCRIPTOR_SHA256: Final = (
    "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
)
_EXTERNAL_MATERIALIZER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py"
)
_EXTERNAL_MATERIALIZER_IMPLEMENTATION_SHA256: Final = (
    "5a7b0d41de86952cd393bb53c4ee3eec8006ab3edc2b42a85f688cbf74dbd041"
)
_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256: Final = (
    "5932626998b1fe75a3bf172d03d832b6c2e98b2d29e7d85507fa17665869b90a"
)
_HISTORICAL_EXTERNAL_MATERIALIZATION_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization.v1"
)
_HISTORICAL_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3_external_materialization_identity.v1"
)
_FULL_RAINBOW_RUNNER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_full_rainbow_runner.py"
)
_FULL_RAINBOW_RUNNER_IMPLEMENTATION_SHA256: Final = (
    "5546b8cd6b394857ad96d4e2bdcaf6e3427cdb16057dd8f67e79654dd617146c"
)
_PPO_GRU_RUNNER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_ppo_gru_runner.py"
)
_PPO_GRU_RUNNER_IMPLEMENTATION_SHA256: Final = (
    "afffdbaf46b9af2cfffe131c8a3bb88dee6de257a8b21296068f22ad5aa93d47"
)
_SCORER_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/_forager_matched_v3_scorer.py"
)
_SCORER_IMPLEMENTATION_SHA256: Final = (
    "eaf2467218355bd8643d8e80a49a1411eabfbea9ad35d4d0f561983f3110993e"
)
_ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_bundle.py"
)
_ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_SHA256: Final = (
    "22199838219cfb5610d83fb71cb828f087b1a4754132f1c325388571e8aa2469"
)
_ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_adapter_reward_publication.py"
)
_ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_SHA256: Final = (
    "8c2c42aad0db0a8eeb45ad2d33f3d76046121fe1f74160e8d1a10231dbe545b5"
)
_ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "5ca0f236a7b6ac58a67578282ca2091f1a443a72502c81fe08b2ecf850ec7905"
)

_DQN_FULL_UPDATE_BLOCKS: Final = 124_927
_DQN_MIN32_UPDATES: Final = 124_920
_DQN_MIN50_UPDATES: Final = 124_915
_DQN_TARGET_REFRESHES: Final = 975
_PERIODIC_UPDATE_EVENTS: Final = 49
_PT_PERMANENT_STEPS_PER_EVENT: Final = 311
_PT_PERMANENT_UPDATES: Final = 15_239
_PT_TOTAL_UPDATES: Final = 140_154
_PPO_ROLLOUT_STEPS: Final = 2_048
_PPO_ROLLOUT_COUNT: Final = 244
_PPO_EPOCHS: Final = 4
_PPO_MINIBATCHES: Final = 32
_PPO_OPTIMIZER_UPDATES: Final = 31_232
_FULL_RAINBOW_OPTIMIZER_UPDATES: Final = 119_928
_FULL_RAINBOW_TARGET_REFRESHES: Final = 60
_PPO_GRU_ROLLOUT_STEPS: Final = 512
_PPO_GRU_ROLLOUT_COUNT: Final = 976
_PPO_GRU_SEGMENTS_PER_EPOCH: Final = 4
_PPO_GRU_EPOCHS: Final = 4
_PPO_GRU_OPTIMIZER_UPDATES: Final = 15_616


class ForagerMatchedV3ConfigurationPlanError(ValueError):
    """The frozen configuration plan or supplied source bytes are invalid."""


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if type(module_file) is not str or not module_file.endswith(expected_suffix):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    try:
        raw = Path(module_file).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        raw = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration plan is not finite canonical JSON"
        ) from exc
    if len(raw) > _MAX_PLAN_BYTES:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration plan exceeds its canonical byte limit"
        )
    return raw


def _plain_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _canonical_bytes(value)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration plan could not be detached"
        ) from exc
    if type(decoded) is not dict:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration plan snapshot must be a plain object"
        )
    return cast(dict[str, Any], decoded)


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


def _descriptor(*transforms: dict[str, object]) -> dict[str, Any]:
    return {
        "schema_version": derivation.DESCRIPTOR_SCHEMA_VERSION,
        "transforms": list(transforms),
    }


_DQN_PLAIN_DESCRIPTOR: Final = _descriptor(
    _transform("/metaParameters/experiment/ntk_freq", "integer", 2_500, 0),
    _transform("/metaParameters/experiment/x_ref_steps", "integer", 100, 0),
    _transform("/total_steps", "integer", 10_000, protocol.MATCHED_V3_HORIZON),
)
_DQN_PLAIN_DESCRIPTOR_SHA256: Final = (
    "d85d2fec4fa18d3ab749f57a0a0b240daf57e05c3cd329bb08d17aac48b5ffeb"
)

_XFINAL_DESCRIPTOR: Final = _descriptor(
    _transform(
        "/metaParameters/environment/env_id",
        "string",
        "ForagaxSquareWaveTwoBiome-v11",
        protocol.MATCHED_V3_ENVIRONMENT_ID,
    ),
    _transform("/total_steps", "integer", 10_000_000, protocol.MATCHED_V3_HORIZON),
)
_XFINAL_DESCRIPTOR_SHA256: Final = (
    "fd20ddfef5fc160f14a0c47d2acd74335a361b061067fada88dd0ef1b42d1497"
)

_RTU_PAPER_DESCRIPTOR: Final = _descriptor(
    _transform(
        "/metaParameters/environment/env_id",
        "string",
        "ForagaxSquareWaveTwoBiome-v11",
        protocol.MATCHED_V3_ENVIRONMENT_ID,
    ),
    _transform("/metaParameters/experiment/ntk_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/weight_drift_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/weight_norm_freq", "integer", 100_000, 0),
    _transform("/metaParameters/experiment/x_ref_steps", "integer", 1_000, 0),
    _transform("/total_steps", "integer", 10_000_000, protocol.MATCHED_V3_HORIZON),
)
_RTU_PAPER_DESCRIPTOR_SHA256: Final = (
    "68b904bed65ab157edbd323725126810d9fd72d7ccc69685a45eaa2aaba48f3b"
)

_RANDOM_DESCRIPTOR: Final = _descriptor(
    _transform("/metaParameters/environment/aperture_size", "integer", 1, 9),
    _transform("/total_steps", "integer", 500_000, protocol.MATCHED_V3_HORIZON),
)
_RANDOM_DESCRIPTOR_SHA256: Final = (
    "fcadac34348354a318950ab1761312064e12af56e8a7f51f2191fcd79e6890e4"
)

_DESCRIPTIVE_HORIZON_DESCRIPTOR: Final = _descriptor(
    _transform("/total_steps", "integer", 500_000, protocol.MATCHED_V3_HORIZON),
)
_DESCRIPTIVE_HORIZON_DESCRIPTOR_SHA256: Final = (
    "d9e02ef47a882a9769792a0367ac309d6f6ab43a6e077521a12c1e0fe098cb0e"
)


def _external_spec(
    path: str,
    original_sha256: str,
    descriptor: Mapping[str, Any],
    descriptor_sha256: str,
    derived_sha256: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "original_sha256": original_sha256,
        "descriptor": _plain_snapshot(descriptor),
        "descriptor_sha256": descriptor_sha256,
        "derived_sha256": derived_sha256,
    }


_EXTERNAL_SPECS: Final[dict[str, dict[str, Any]]] = {
    "external_dqn_plain": _external_spec(
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/DQN.json",
        "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
        _DQN_PLAIN_DESCRIPTOR,
        _DQN_PLAIN_DESCRIPTOR_SHA256,
        "1d8a711ee1e4db575cb0edcacbaf38f97bd06cddc24019eb64b8c410e84b4e85",
    ),
    "external_dqn_crelu": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_CReLU.json",
        "d433b87789e180df3f153cebdafa53f3b6278325fcd32889c8959552cecfeda0",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "ef92352b97d92e7d40458db48157f589b0d0984f2f4286947c9a1f28bd522892",
    ),
    "external_dqn_redo": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
        "DQN_ReDo_PostLNScore.json",
        "61fa39de8426e2fb78305846b26f6c7a977c72b9cc8a61fc70419f8c15afc8ab",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "c38288f2ddb6a5dd8892954b499370d04399ec41e966fe790643c9d64b5ffc54",
    ),
    "external_dqn_reward_trace": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
        "DQN_reward_trace.json",
        "3d14f03bc22eec14e4abcc32e635c1dbfa83d4149ef2eaca3609ddba3281ffcb",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "8641a3b4673940f5519f074b617ccc58a6c14b61a8b448df434cebb3d5f4c974",
    ),
    "external_dqn_l2_init": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DQN_L2_Init.json",
        "6a90d4e970c66d0cc968c9988e0e91a3341fdcb2126954a1b7314f7154b53934",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "2a2a1dc503b0617c35c202027a646db32186e2668d4b8988215f516a036b9107",
    ),
    "external_pt_dqn_xfinal": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/PT_DQN_64.json",
        "4f2ff117d4b82458e3a4bb373d54d03d5b1fedeb4d0b25214235facb5ff2b690",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "05eaad6da93d8c42d8bd60da3d6c3728bca5c653608eb98210a48a76bedce2e2",
    ),
    "external_drqn_xfinal": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "70a5ee902aa6128ec65c6d4fd33e27da0e3eaa02bd4ea8b776baf3fa158c27de",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "2b0e177420a9f9a4c8a7bd7aede9c7d2c5add3da4c8b3e301f32bb2588637047",
    ),
    "isolated_ppo_generic": _external_spec(
        "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/ActorCriticMLP.json",
        "c8915481c67045339de4b013372d2538eafa91b21c639d2fb0e08d0c60865228",
        _XFINAL_DESCRIPTOR,
        _XFINAL_DESCRIPTOR_SHA256,
        "27ffdffcf3ff3e722be5cdfe58d6bc07348ebe5380478032eedfaf435b754c71",
    ),
    "isolated_rtu_paper_scale": _external_spec(
        "experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/"
        "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_2048.json",
        "b9e7bf1bfa307239df848677b6ad4e7c76ef316567b11f75e9455625efc20e65",
        _RTU_PAPER_DESCRIPTOR,
        _RTU_PAPER_DESCRIPTOR_SHA256,
        "c32e240bf8c78cf2c7d1ad958bbfc8975b55160fb09490401763a346c2a21090",
    ),
    "random_policy": _external_spec(
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Random.json",
        "24b9d17d2fa4d5da0dc9afd24bbd605fdd4e7574a70f13dc9648e6e6412f6a9a",
        _RANDOM_DESCRIPTOR,
        _RANDOM_DESCRIPTOR_SHA256,
        "d20dc9294baab331c4658e4c682d5e1eee3c6f7cc6baf5d17586f48362e8936d",
    ),
    "search_nearest": _external_spec(
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Nearest.json",
        "2c2f67b13f818c7a639411e491095f04dbf3e789a1197c40a6a659ef26e0238d",
        _DESCRIPTIVE_HORIZON_DESCRIPTOR,
        _DESCRIPTIVE_HORIZON_DESCRIPTOR_SHA256,
        "97b644c4c625155ae16fa7b69432ea0774f767142cc0e28b3d6fcec18c17d2ab",
    ),
    "search_oracle": _external_spec(
        "experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/"
        "Baselines/Search-Oracle.json",
        "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
        _DESCRIPTIVE_HORIZON_DESCRIPTOR,
        _DESCRIPTIVE_HORIZON_DESCRIPTOR_SHA256,
        "426fc604bfbf9c2545a505d9fdf4c2a7a7fdf063ddb3a0fefd22308149c05e89",
    ),
}

EXTERNAL_CONFIGURATION_CANDIDATE_IDS: Final = tuple(_EXTERNAL_SPECS)

_LOCAL_CONFIGURATION_SHA256: Final = {
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
    "alberta_horde_step3e3": (
        "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5"
    ),
    "alberta_rtu_h08_taylor": (
        "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
    ),
}

_LOCAL_BUILDER_BY_ID: Final = {
    **{
        candidate_id: "alberta.forager_matched_v3.generated_local.causal_map_grid.v1"
        for candidate_id in universe.MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS
    },
    **{
        candidate_id: "alberta.forager_matched_v3.generated_local.horde_actor_critic.v1"
        for candidate_id in universe.MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS
    },
    universe.MATCHED_V3_LOCAL_RTU_CANDIDATE_ID: (
        "alberta.forager_matched_v3.generated_local.rtu_h08_taylor.v1"
    ),
}

_ADAPTER_IDENTITIES: Final = {
    "adapted_full_rainbow": {
        "repository_id": "dopamine",
        "configuration_schema_version": full_rainbow.FULL_RAINBOW_CONFIG_SCHEMA_VERSION,
        "configuration_sha256": full_rainbow.FULL_RAINBOW_CONFIG_SHA256,
        "adapter_descriptor_schema_version": (
            full_rainbow.FULL_RAINBOW_DESCRIPTOR_SCHEMA_VERSION
        ),
        "adapter_descriptor_sha256": full_rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256,
        "implementation_path": _FULL_RAINBOW_IMPLEMENTATION_PATH,
        "implementation_source_sha256": _FULL_RAINBOW_IMPLEMENTATION_SHA256,
        "runner_descriptor_schema_version": (
            full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SCHEMA_VERSION
        ),
        "runner_descriptor_sha256": (
            full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
        ),
        "runner_implementation_path": _FULL_RAINBOW_RUNNER_IMPLEMENTATION_PATH,
        "runner_implementation_source_sha256": (
            _FULL_RAINBOW_RUNNER_IMPLEMENTATION_SHA256
        ),
        "runner_result_receipt_schema_version": (
            full_rainbow_runner.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
        ),
    },
    "adapted_ppo_gru": {
        "repository_id": "pobax",
        "configuration_schema_version": ppo_gru.PPO_GRU_CONFIGURATION_SCHEMA_VERSION,
        "configuration_sha256": ppo_gru.PPO_GRU_CONFIGURATION_SHA256,
        "adapter_descriptor_schema_version": (
            ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "adapter_descriptor_sha256": ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256,
        "implementation_path": _PPO_GRU_IMPLEMENTATION_PATH,
        "implementation_source_sha256": _PPO_GRU_IMPLEMENTATION_SHA256,
        "runner_descriptor_schema_version": (
            ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SCHEMA_VERSION
        ),
        "runner_descriptor_sha256": ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256,
        "runner_implementation_path": _PPO_GRU_RUNNER_IMPLEMENTATION_PATH,
        "runner_implementation_source_sha256": _PPO_GRU_RUNNER_IMPLEMENTATION_SHA256,
        "runner_result_receipt_schema_version": (
            ppo_gru_runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
        ),
    },
}

_EXPECTED_CANDIDATE_IDS: Final = (
    tuple(_LOCAL_CONFIGURATION_SHA256)
    + EXTERNAL_CONFIGURATION_CANDIDATE_IDS[:9]
    + tuple(_ADAPTER_IDENTITIES)
    + EXTERNAL_CONFIGURATION_CANDIDATE_IDS[9:]
)


def _empty_operations() -> dict[str, int]:
    return {
        "causal_transition_updates": 0,
        "target_snapshot_refreshes": 0,
        "redo_recycles": 0,
        "permanent_update_events": 0,
        "transient_parameter_decays": 0,
    }


def _continuing_argv() -> list[str]:
    return [
        "--exp",
        "<derived_configuration_path>",
        "--idxs",
        "<exactly_one_index>",
        "--environment_seed",
        "<environment_seed_uint31>",
        "--agent_seed",
        "<candidate_private_agent_seed_uint31>",
        "--max_steps",
        str(protocol.MATCHED_V3_HORIZON),
        "--save_path",
        "<candidate_private_results_root>",
        "--checkpoint_path",
        "<new_empty_candidate_private_checkpoint_root>",
        "--silent",
    ]


def _ppo_argv() -> list[str]:
    return [
        "--exp",
        "<derived_configuration_path>",
        "--idxs",
        "<exactly_one_index>",
        "--environment_seed",
        "<environment_seed_uint31>",
        "--agent_seed",
        "<candidate_private_agent_seed_uint31>",
        "--max_steps",
        str(_PPO_ROLLOUT_COUNT),
        "--save_path",
        "<candidate_private_results_root>",
        "--checkpoint_path",
        "<new_empty_candidate_private_checkpoint_root>",
        "--silent",
    ]


def _execution_shape(candidate_id: str) -> dict[str, Any]:
    operations = _empty_operations()
    shape: dict[str, Any] = {
        "horizon": protocol.MATCHED_V3_HORIZON,
        "interaction_count": protocol.MATCHED_V3_HORIZON,
        "horizon_unit": "environment_transitions",
        "entrypoint_path": None,
        "entrypoint_sha256": None,
        "update_schedule": None,
        "rollout_steps": None,
        "rollout_count": None,
        "epochs": None,
        "minibatches_per_epoch": None,
        "optimizer_update_count": None,
        "optimizer_update_subcounts": None,
        "replay_capacity_transitions": {"main": 0, "permanent": 0},
        "non_gradient_operations": operations,
        "exact_workload_argv": None,
        "default_without_override": None,
    }
    if candidate_id in universe.MATCHED_V3_CAUSAL_SELECTION_CANDIDATE_IDS:
        shape["update_schedule"] = "causal_nonparametric_every_transition"
        shape["optimizer_update_count"] = 0
        shape["optimizer_update_subcounts"] = {}
        operations["causal_transition_updates"] = protocol.MATCHED_V3_HORIZON
        return shape
    if candidate_id in universe.MATCHED_V3_HORDE_SELECTION_CANDIDATE_IDS:
        shape["update_schedule"] = "online_horde_actor_critic_every_transition"
        shape["optimizer_update_count"] = 2 * protocol.MATCHED_V3_HORIZON
        shape["optimizer_update_subcounts"] = {
            "actor": protocol.MATCHED_V3_HORIZON,
            "critic": protocol.MATCHED_V3_HORIZON,
        }
        return shape
    if candidate_id == universe.MATCHED_V3_LOCAL_RTU_CANDIDATE_ID:
        shape["update_schedule"] = "online_rtu_rtrl_every_transition"
        shape["optimizer_update_count"] = 2 * protocol.MATCHED_V3_HORIZON
        shape["optimizer_update_subcounts"] = {
            "actor": protocol.MATCHED_V3_HORIZON,
            "critic": protocol.MATCHED_V3_HORIZON,
        }
        return shape
    if candidate_id == "adapted_full_rainbow":
        shape.update(
            {
                "entrypoint_path": _FULL_RAINBOW_RUNNER_IMPLEMENTATION_PATH,
                "entrypoint_sha256": _FULL_RAINBOW_RUNNER_IMPLEMENTATION_SHA256,
                "update_schedule": "full_rainbow_replay_update_period_4",
                "optimizer_update_count": _FULL_RAINBOW_OPTIMIZER_UPDATES,
                "optimizer_update_subcounts": {
                    "joint_distributional_q": _FULL_RAINBOW_OPTIMIZER_UPDATES
                },
                "replay_capacity_transitions": {
                    "main": 1_000_000,
                    "permanent": 0,
                },
            }
        )
        operations["target_snapshot_refreshes"] = _FULL_RAINBOW_TARGET_REFRESHES
        return shape
    if candidate_id == "adapted_ppo_gru":
        shape.update(
            {
                "entrypoint_path": _PPO_GRU_RUNNER_IMPLEMENTATION_PATH,
                "entrypoint_sha256": _PPO_GRU_RUNNER_IMPLEMENTATION_SHA256,
                "update_schedule": "ppo_gru_contiguous_recurrent_segment_updates",
                "rollout_steps": _PPO_GRU_ROLLOUT_STEPS,
                "rollout_count": _PPO_GRU_ROLLOUT_COUNT,
                "epochs": _PPO_GRU_EPOCHS,
                "minibatches_per_epoch": _PPO_GRU_SEGMENTS_PER_EPOCH,
                "optimizer_update_count": _PPO_GRU_OPTIMIZER_UPDATES,
                "optimizer_update_subcounts": {
                    "joint_policy_value_recurrent_segment": (
                        _PPO_GRU_OPTIMIZER_UPDATES
                    )
                },
            }
        )
        return shape
    if candidate_id in {"isolated_ppo_generic", "isolated_rtu_paper_scale"}:
        shape.update(
            {
                "entrypoint_path": "src/rtu_ppo.py",
                "entrypoint_sha256": _DERIVED_RTU_PPO_SOURCE_SHA256,
                "update_schedule": "ppo_rollout_minibatch_updates",
                "rollout_steps": _PPO_ROLLOUT_STEPS,
                "rollout_count": _PPO_ROLLOUT_COUNT,
                "epochs": _PPO_EPOCHS,
                "minibatches_per_epoch": _PPO_MINIBATCHES,
                "optimizer_update_count": _PPO_OPTIMIZER_UPDATES,
                "optimizer_update_subcounts": {"joint_policy_value": _PPO_OPTIMIZER_UPDATES},
                "exact_workload_argv": _ppo_argv(),
                "default_without_override": {
                    "rollout_count": 245,
                    "interaction_count": 501_760,
                    "accepted": False,
                },
            }
        )
        return shape

    shape["entrypoint_path"] = "src/continuing_main.py"
    shape["entrypoint_sha256"] = _DERIVED_CONTINUING_MAIN_SOURCE_SHA256
    shape["exact_workload_argv"] = _continuing_argv()
    if candidate_id in {"random_policy", "search_nearest", "search_oracle"}:
        shape["update_schedule"] = "environment_step_only_reference"
        shape["optimizer_update_count"] = 0
        shape["optimizer_update_subcounts"] = {}
        return shape

    shape["update_schedule"] = "continuing_main_update_frequency_4"
    shape["replay_capacity_transitions"] = {
        "main": 10_000 if candidate_id == "external_dqn_plain" else 1_000,
        "permanent": 10_000 if candidate_id == "external_pt_dqn_xfinal" else 0,
    }
    updates = (
        _DQN_MIN32_UPDATES if candidate_id == "external_dqn_plain" else _DQN_MIN50_UPDATES
    )
    shape["optimizer_update_count"] = updates
    shape["optimizer_update_subcounts"] = {"main": updates}
    operations["target_snapshot_refreshes"] = _DQN_TARGET_REFRESHES
    if candidate_id == "external_dqn_redo":
        operations["redo_recycles"] = _PERIODIC_UPDATE_EVENTS
    if candidate_id == "external_pt_dqn_xfinal":
        shape["optimizer_update_count"] = _PT_TOTAL_UPDATES
        shape["optimizer_update_subcounts"] = {
            "transient": _DQN_MIN50_UPDATES,
            "permanent": _PT_PERMANENT_UPDATES,
        }
        operations["permanent_update_events"] = _PERIODIC_UPDATE_EVENTS
        operations["transient_parameter_decays"] = _PERIODIC_UPDATE_EVENTS
    return shape


def _diagnostic_policy(candidate_id: str) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "status": "bound_unqualified",
        "ntk_enabled": False,
        "reference_collection_enabled": False,
        "configured_reference_steps": None,
        "effective_reference_steps": None,
        "chunked_ref_source_value": None,
        "chunked_ref_effective_value": None,
        "chunked_ref_inert": True,
        "weight_norm_enabled": False,
        "weight_drift_enabled": False,
        "video_length_steps": 0,
        "save_every_steps": None,
        "video_every_steps": None,
        "checkpoint_milestones_within_horizon": 0,
        "fresh_empty_checkpoint_root_required": True,
        "single_index_video_capture_unavoidable_without_source_derivation": False,
        "single_index_video_window_steps": 0,
    }
    if candidate_id in _ADAPTER_IDENTITIES:
        policy["status"] = "adapter_full_runner_implemented_unqualified"
        return policy
    if candidate_id in _LOCAL_CONFIGURATION_SHA256:
        policy["status"] = "local_worker_unqualified_for_v3"
        return policy
    if candidate_id in {"isolated_ppo_generic", "isolated_rtu_paper_scale"}:
        policy["video_length_steps"] = 1_000
        policy["single_index_video_capture_unavoidable_without_source_derivation"] = True
        policy["single_index_video_window_steps"] = _PPO_ROLLOUT_STEPS
        if candidate_id == "isolated_ppo_generic":
            policy["effective_reference_steps"] = 128
        else:
            policy["configured_reference_steps"] = 0
            policy["effective_reference_steps"] = 0
            policy["chunked_ref_source_value"] = 256
            policy["chunked_ref_effective_value"] = 1
        return policy
    if candidate_id == "external_dqn_plain":
        policy["configured_reference_steps"] = 0
        policy["effective_reference_steps"] = 0
    save_every = (
        1_000_000
        if candidate_id in {"external_dqn_redo", "external_pt_dqn_xfinal"}
        else 10_001_000
    )
    policy["save_every_steps"] = save_every
    policy["video_every_steps"] = save_every
    return policy


def _rng_contract(candidate_id: str) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "environment_seed_namespace": "environment",
        "agent_seed_namespace": f"agent/{candidate_id}",
        "environment_seed_transport": None,
        "agent_seed_transport": None,
        "transport_status": "unimplemented",
        "transport_artifact_scope": None,
        "environment_rng_independence_qualified": False,
        "agent_consumption_invariance_proved": False,
        "runtime_trace_verified": False,
        "source_closure_bound": False,
        "statistical_independence_claimed": False,
        "environment_transport_descriptor_sha256": None,
        "agent_transport_descriptor_sha256": None,
        "runner_transport_descriptor_sha256": None,
    }
    if candidate_id in _LOCAL_CONFIGURATION_SHA256:
        contract.update(
            {
                "environment_seed_transport": "seeds_sequence_lane_index",
                "agent_seed_transport": "agent_seeds_sequence_lane_index",
                "transport_status": "implemented_unqualified_local_api",
                "transport_artifact_scope": "local_api_without_source_snapshot",
            }
        )
    elif candidate_id in _EXTERNAL_SPECS:
        contract.update(
            {
                "environment_seed_transport": "required_cli_environment_seed_uint31",
                "agent_seed_transport": "required_cli_agent_seed_uint31",
                "transport_status": "implemented_unqualified_external_patch_set",
                "transport_artifact_scope": "derived_patch_set_only",
                "environment_transport_descriptor_sha256": (
                    _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
                ),
                "agent_transport_descriptor_sha256": (
                    _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
                ),
            }
        )
    elif candidate_id == "adapted_full_rainbow":
        contract.update(
            {
                "environment_seed_transport": (
                    "shared_foragax_bridge_uint31_direct_threefry2x32_split_chain"
                ),
                "agent_seed_transport": (
                    "full_rainbow_agent_uint31_threefry2x32_folded_namespace"
                ),
                "transport_status": (
                    "implemented_unqualified_core_bridge_and_full_runner_apis"
                ),
                "transport_artifact_scope": (
                    "core_bridge_and_full_runner_without_runtime_qualification"
                ),
                "environment_transport_descriptor_sha256": (
                    _FORAGAX_BRIDGE_DESCRIPTOR_SHA256
                ),
                "agent_transport_descriptor_sha256": (
                    full_rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256
                ),
                "runner_transport_descriptor_sha256": (
                    full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
                ),
            }
        )
    elif candidate_id == "adapted_ppo_gru":
        contract.update(
            {
                "environment_seed_transport": (
                    "shared_foragax_bridge_uint31_direct_threefry2x32_split_chain"
                ),
                "agent_seed_transport": (
                    "ppo_gru_agent_uint31_threefry2x32_split_chain"
                ),
                "transport_status": (
                    "implemented_unqualified_core_bridge_and_full_runner_apis"
                ),
                "transport_artifact_scope": (
                    "core_bridge_and_full_runner_without_runtime_qualification"
                ),
                "environment_transport_descriptor_sha256": (
                    _FORAGAX_BRIDGE_DESCRIPTOR_SHA256
                ),
                "agent_transport_descriptor_sha256": (
                    ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
                ),
                "runner_transport_descriptor_sha256": (
                    ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
                ),
            }
        )
    return contract


def _local_configuration(candidate_id: str) -> dict[str, Any]:
    return {
        "kind": "generated_local",
        "repository_id": "local_alberta",
        "builder_id": _LOCAL_BUILDER_BY_ID[candidate_id],
        "builder_status": "implemented_unqualified",
        "builder_descriptor_sha256": (
            _LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
        ),
        "source_descriptor_sha256": (
            _LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        ),
        "worker_envelope_sha256": _LOCAL_CONFIGURATION_SHA256[candidate_id],
        "configuration_complete": True,
        "source_snapshot_status": "unqualified_current_checkout",
    }


def _external_configuration(candidate_id: str) -> dict[str, Any]:
    spec = _EXTERNAL_SPECS[candidate_id]
    return {
        "kind": "derived_external",
        "repository_id": _UPSTREAM_REPOSITORY_ID,
        "status": "exact_transform_bound_unqualified_for_v3",
        "original_path": spec["path"],
        "original_sha256": spec["original_sha256"],
        "transform_descriptor": _plain_snapshot(spec["descriptor"]),
        "transform_descriptor_sha256": spec["descriptor_sha256"],
        "derived_sha256": spec["derived_sha256"],
        "configuration_complete": True,
        "source_snapshot_status": (
            "upstream_archive_pinned_seed_patch_set_derived_unqualified"
        ),
    }


def _adapter_configuration(candidate_id: str) -> dict[str, Any]:
    identity = _ADAPTER_IDENTITIES[candidate_id]
    return {
        "kind": "derived_local_adapter",
        **identity,
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


def _observation_contract(candidate_id: str) -> dict[str, Any]:
    if candidate_id in {"search_nearest", "search_oracle"}:
        return {
            "access": "privileged_global_object_access",
            "observation_type": "object",
            "aperture_size": -1,
            "inferentially_matched": False,
            "qualification_required": True,
        }
    return {
        "access": (
            "observation_independent_random_action_reference"
            if candidate_id == "random_policy"
            else "matched_partial_color_aperture_9"
        ),
        "observation_type": protocol.MATCHED_V3_OBSERVATION_TYPE,
        "aperture_size": protocol.MATCHED_V3_APERTURE_SIZE,
        "inferentially_matched": candidate_id != "random_policy",
        "qualification_required": True,
    }


def _shared_environment_bridge() -> dict[str, Any]:
    return {
        "schema_version": foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_sha256": _FORAGAX_BRIDGE_DESCRIPTOR_SHA256,
        "implementation_path": _FORAGAX_BRIDGE_IMPLEMENTATION_PATH,
        "implementation_source_sha256": _FORAGAX_BRIDGE_IMPLEMENTATION_SHA256,
        "status": "implemented_unqualified",
        "adapter_consumers": list(_ADAPTER_IDENTITIES),
        "environment_rng_schedule": (
            foragax_bridge.MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE
        ),
        "runtime_parity_executed": False,
        "runtime_qualified": False,
        "compiled_chunk_kernel_complete": False,
        "source_closure_bound": False,
    }


def _external_materializer() -> dict[str, Any]:
    return {
        "manifest_schema_version": _HISTORICAL_EXTERNAL_MATERIALIZATION_SCHEMA_VERSION,
        "identity_schema_version": (
            _HISTORICAL_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION
        ),
        "identity_sha256": _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256,
        "implementation_path": _EXTERNAL_MATERIALIZER_IMPLEMENTATION_PATH,
        "implementation_source_sha256": _EXTERNAL_MATERIALIZER_IMPLEMENTATION_SHA256,
        "status": "implemented_unexecuted",
        "production_materialization_accepted": False,
        "production_manifest_sha256": None,
        "materialized_source_closure_bound": False,
        "archive_bytes_verified": False,
        "runtime_dependencies_qualified": False,
        "execution_authorized": False,
    }


def _adapter_result_conversion() -> dict[str, Any]:
    return {
        "descriptor_schema_version": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        ),
        "descriptor_sha256": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
        ),
        "manifest_schema_version": (
            adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        ),
        "implementation_path": _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_PATH,
        "implementation_source_sha256": (
            _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_SHA256
        ),
        "status": adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_STATUS,
        "candidate_consumers": list(_ADAPTER_IDENTITIES),
        "scorer_source_path": _SCORER_IMPLEMENTATION_PATH,
        "scorer_source_sha256": _SCORER_IMPLEMENTATION_SHA256,
        "score_receipt_schema_version": scorer.SCORE_RECEIPT_SCHEMA_VERSION,
        "reward_container_schema_version": scorer.NPZ_CONTAINER_SCHEMA_VERSION,
        "raw_trace_encoding_schema_version": scorer.RAW_TRACE_ENCODING_SCHEMA_VERSION,
        "canonical_reward_artifact_size_bytes": scorer.CANONICAL_NPZ_SIZE_BYTES,
        "in_memory_conversion_complete": True,
        "filesystem_publication_complete": False,
        "campaign_ingestion_complete": False,
        "ingestion_authorized": False,
    }


def _adapter_result_publication() -> dict[str, Any]:
    return {
        "descriptor_schema_version": (
            adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        ),
        "descriptor_sha256": _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256,
        "publication_schema_version": (
            adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
        ),
        "implementation_path": _ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_PATH,
        "implementation_source_sha256": (
            _ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_SHA256
        ),
        "status": adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_STATUS,
        "candidate_consumers": list(_ADAPTER_IDENTITIES),
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


def _candidate_records() -> list[dict[str, Any]]:
    universe_candidates = universe.matched_v3_development_universe_descriptor()["candidates"]
    universe_by_id = {item["candidate_id"]: item for item in universe_candidates}
    records: list[dict[str, Any]] = []
    for candidate_id in universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS:
        if candidate_id in _LOCAL_CONFIGURATION_SHA256:
            configuration = _local_configuration(candidate_id)
        elif candidate_id in _EXTERNAL_SPECS:
            configuration = _external_configuration(candidate_id)
        else:
            configuration = _adapter_configuration(candidate_id)
        candidate = universe_by_id[candidate_id]
        records.append(
            {
                "candidate_id": candidate_id,
                "analysis_role": candidate["analysis_role"],
                "development_selection_group": candidate["development_selection_group"],
                "configuration": configuration,
                "execution_shape": _execution_shape(candidate_id),
                "diagnostic_policy": _diagnostic_policy(candidate_id),
                "rng_contract": _rng_contract(candidate_id),
                "observation_contract": _observation_contract(candidate_id),
                "execution_ready": False,
                "execution_authorized": False,
                "scientific_promotion_allowed": False,
            }
        )
    return records


def _configuration_plan() -> dict[str, Any]:
    universe_descriptor = universe.matched_v3_development_universe_descriptor()
    return {
        "schema_version": CONFIGURATION_PLAN_SCHEMA_VERSION,
        "status": "unexecuted_configuration_complete_design",
        "classification": "development_only_nonpromoting_configuration_plan",
        "bindings": {
            "candidate_universe_schema_version": (
                universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION
            ),
            "candidate_universe_sha256": _CANDIDATE_UNIVERSE_SHA256,
            "cumulative_reward_metric_schema_version": (
                protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
            ),
            "cumulative_reward_metric_sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
            "trial_block_generator_plan_schema_version": (
                protocol.TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION
            ),
            "trial_block_generator_plan_sha256": (
                _TRIAL_BLOCK_GENERATOR_PLAN_SHA256
            ),
        },
        "task": {
            "environment_id": _ENVIRONMENT_ID,
            "observation_type": _OBSERVATION_TYPE,
            "aperture_size": _APERTURE_SIZE,
            "horizon": _HORIZON,
        },
        "source_repositories": universe_descriptor["source_pins"],
        "upstream_execution_source": {
            "repository_id": _UPSTREAM_REPOSITORY_ID,
            "commit_git_sha1": _UPSTREAM_COMMIT,
            "tree_git_sha1": _UPSTREAM_TREE,
            "archive_sha256": _UPSTREAM_ARCHIVE_SHA256,
            "archive_size_bytes": _UPSTREAM_ARCHIVE_SIZE,
            "review_anchors": [
                {
                    "path": "src/continuing_main.py",
                    "sha256": _CONTINUING_MAIN_SOURCE_SHA256,
                },
                {"path": "src/rtu_ppo.py", "sha256": _RTU_PPO_SOURCE_SHA256},
            ],
            "seed_transport_patch_set_schema_version": (
                external_seed_transport.SCHEMA_VERSION
            ),
            "seed_transport_patch_set_descriptor_sha256": (
                _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
            ),
            "seed_transport_patch_set_scope": "derived_patch_set_only",
            "seed_transport_patch_set_status": (
                "implemented_unqualified_source_and_ast_contract"
            ),
            "derived_source_files": [
                {
                    "path": path,
                    "sha256": external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[
                        path
                    ],
                }
                for path in external_seed_transport.SOURCE_PATHS
            ],
            "full_dependency_inventory_bound": False,
            "materialization_status": (
                "implementation_available_no_accepted_production_manifest"
            ),
            "runtime_trace_verified": False,
        },
        "external_materializer": _external_materializer(),
        "local_execution_source": {
            "repository_id": "local_alberta",
            "snapshot_sha256": None,
            "inventory_sha256": None,
            "configuration_source_descriptor_sha256": (
                _LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
            ),
            "configuration_builder_descriptor_sha256": (
                _LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
            ),
            "status": "configuration_builder_implemented_source_snapshot_missing",
        },
        "shared_environment_bridge": _shared_environment_bridge(),
        "adapter_result_conversion": _adapter_result_conversion(),
        "adapter_result_publication": _adapter_result_publication(),
        "candidate_ids": list(universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS),
        "candidates": _candidate_records(),
        "update_accounting_constants": {
            "continuing_main_initial_agent_step": 1,
            "continuing_main_update_frequency": 4,
            "continuing_main_alignment_prefix_steps": 3,
            "continuing_main_full_update_blocks": _DQN_FULL_UPDATE_BLOCKS,
            "buffer_minimum_32_optimizer_updates": _DQN_MIN32_UPDATES,
            "buffer_minimum_50_optimizer_updates": _DQN_MIN50_UPDATES,
            "target_refresh_period_updates": 128,
            "periodic_frequency_updates": 2_500,
            "periodic_update_events": _PERIODIC_UPDATE_EVENTS,
            "pt_permanent_steps_per_event": _PT_PERMANENT_STEPS_PER_EVENT,
        },
        "readiness_blockers": [
            {
                "blocker_id": "environment_rng_independence_unqualified",
                "candidate_ids": list(universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS),
            },
            {
                "blocker_id": "local_v3_source_snapshot_missing",
                "candidate_ids": list(_LOCAL_CONFIGURATION_SHA256),
            },
            {
                "blocker_id": "accepted_external_materialized_source_closure_missing",
                "candidate_ids": list(_EXTERNAL_SPECS),
            },
            {
                "blocker_id": "accepted_external_materialization_manifest_missing",
                "candidate_ids": list(_EXTERNAL_SPECS),
            },
            {
                "blocker_id": "external_seed_transport_runtime_unqualified",
                "candidate_ids": list(_EXTERNAL_SPECS),
            },
            {
                "blocker_id": "adapter_source_closure_and_qualification_missing",
                "candidate_ids": list(_ADAPTER_IDENTITIES),
            },
            {
                "blocker_id": "adapter_full_horizon_resource_profiles_unqualified",
                "candidate_ids": list(_ADAPTER_IDENTITIES),
            },
            {
                "blocker_id": (
                    "adapter_environment_bridge_runtime_parity_and_compiled_kernel_unqualified"
                ),
                "candidate_ids": list(_ADAPTER_IDENTITIES),
            },
            {
                "blocker_id": (
                    "adapter_production_publication_acceptance_and_campaign_ingestion_missing"
                ),
                "candidate_ids": list(_ADAPTER_IDENTITIES),
            },
        ],
        "configuration_complete": True,
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
        "limitations": [
            "No workload, result, score, or protected seed is supplied or authorized.",
            (
                "Full Rainbow and PPO-GRU exact-task cores and full-horizon runners "
                "are implemented and configuration-bound but remain runtime-unqualified."
            ),
            "External two-seed transport is a source patch set, not a source closure.",
            (
                "The external materializer is implemented, but no production derived "
                "checkout or manifest has been accepted."
            ),
            "External RNG replay and environment traces remain runtime-unqualified.",
            "Local configuration builders are implemented but remain unqualified.",
            "The local Alberta v3 execution source snapshot remains unbound.",
            (
                "The shared adapter environment bridge is implemented, but real "
                "Foragax parity, backend qualification, and a compiled chunk kernel "
                "remain missing."
            ),
            (
                "Strict in-memory adapter reward conversion and an atomic content-only "
                "publication writer are implemented; no production publication has "
                "been accepted and campaign ingestion remains missing."
            ),
            "Adapter source closure, resource profiles, and runtime behavior remain unqualified.",
            "PPO exact horizon requires the bound --max_steps 244 update override.",
        ],
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{label} key drift: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _assert_unaliased_plain_json(value: Any) -> None:
    pending = [value]
    containers: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if id(item) in containers:
                raise AssertionError("configuration plan contains aliased containers")
            containers.add(id(item))
            if any(type(key) is not str for key in item):
                raise AssertionError("configuration plan contains a non-string key")
            pending.extend(item.values())
        elif type(item) is list:
            if id(item) in containers:
                raise AssertionError("configuration plan contains aliased containers")
            containers.add(id(item))
            pending.extend(item)
        elif item is not None and type(item) not in {str, int, bool}:
            raise AssertionError("configuration plan contains a non-plain JSON value")


def _validate_dependency_pins() -> None:
    if universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256 != _CANDIDATE_UNIVERSE_SHA256:
        raise AssertionError("matched-v3 candidate-universe digest drift")
    if protocol.CUMULATIVE_REWARD_METRIC_SHA256 != _CUMULATIVE_REWARD_METRIC_SHA256:
        raise AssertionError("matched-v3 cumulative-reward metric digest drift")
    if (
        protocol.TRIAL_BLOCK_GENERATOR_PLAN_SHA256
        != _TRIAL_BLOCK_GENERATOR_PLAN_SHA256
    ):
        raise AssertionError("matched-v3 trial-block generator-plan digest drift")
    if (
        protocol.MATCHED_V3_ENVIRONMENT_ID,
        protocol.MATCHED_V3_OBSERVATION_TYPE,
        protocol.MATCHED_V3_APERTURE_SIZE,
        protocol.MATCHED_V3_HORIZON,
    ) != (_ENVIRONMENT_ID, _OBSERVATION_TYPE, _APERTURE_SIZE, _HORIZON):
        raise AssertionError("matched-v3 task constants drift")
    if universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS != _EXPECTED_CANDIDATE_IDS:
        raise AssertionError("matched-v3 exact 28-candidate order drift")
    for module_file, expected_path, expected_sha256 in (
        (
            full_rainbow_runner.__file__,
            _FULL_RAINBOW_RUNNER_IMPLEMENTATION_PATH,
            _FULL_RAINBOW_RUNNER_IMPLEMENTATION_SHA256,
        ),
        (
            ppo_gru_runner.__file__,
            _PPO_GRU_RUNNER_IMPLEMENTATION_PATH,
            _PPO_GRU_RUNNER_IMPLEMENTATION_SHA256,
        ),
        (scorer.__file__, _SCORER_IMPLEMENTATION_PATH, _SCORER_IMPLEMENTATION_SHA256),
        (
            adapter_reward_bundle.__file__,
            _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_PATH,
            _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_SHA256,
        ),
        (
            adapter_reward_publication.__file__,
            _ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_PATH,
            _ADAPTER_REWARD_PUBLICATION_IMPLEMENTATION_SHA256,
        ),
    ):
        if not hmac.compare_digest(
            _source_sha256(module_file, expected_path), expected_sha256
        ):
            raise AssertionError(f"matched-v3 implementation source drift: {expected_path}")
    if (
        external_seed_transport.EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
        != _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
    ):
        raise AssertionError("external two-seed patch-set descriptor drift")
    if dict(external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH) != {
        "src/continuing_main.py": _DERIVED_CONTINUING_MAIN_SOURCE_SHA256,
        "src/problems/BaseProblem.py": (
            "a4ab77408c1bb38dd3f4e72d830765176c38bba4b73b69fe296765a0272d87dc"
        ),
        "src/problems/Foragax.py": (
            "ff6e875511fcc574bafde7f114382dccf5303dba96f4154d5abbc16744d8e7c9"
        ),
        "src/rtu_ppo.py": _DERIVED_RTU_PPO_SOURCE_SHA256,
    }:
        raise AssertionError("external two-seed derived-source pin drift")
    if (
        local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_CANDIDATE_IDS
        != tuple(_LOCAL_CONFIGURATION_SHA256)
        or dict(local_configuration.EXPECTED_CONFIGURATION_SHA256_BY_CANDIDATE)
        != _LOCAL_CONFIGURATION_SHA256
        or dict(local_configuration.BUILDER_ID_BY_CANDIDATE)
        != _LOCAL_BUILDER_BY_ID
    ):
        raise AssertionError("local configuration builder identity drift")
    if (
        local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        != _LOCAL_CONFIGURATION_SOURCE_DESCRIPTOR_SHA256
        or local_configuration.MATCHED_V3_LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
        != _LOCAL_CONFIGURATION_BUILDER_DESCRIPTOR_SHA256
    ):
        raise AssertionError("local configuration descriptor drift")
    if (
        foragax_bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
        != _FORAGAX_BRIDGE_DESCRIPTOR_SHA256
        or foragax_bridge.MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE
        != "dedicated_environment_split_chain_v1"
    ):
        raise AssertionError("shared Foragax bridge identity drift")
    bridge_descriptor = foragax_bridge.parse_matched_v3_foragax_bridge_descriptor(
        foragax_bridge.canonical_matched_v3_foragax_bridge_descriptor_bytes()
    )
    bridge_source = cast(dict[str, Any], bridge_descriptor["source"])
    bridge_task = cast(dict[str, Any], bridge_descriptor["task"])
    bridge_runtime = cast(dict[str, Any], bridge_descriptor["runtime"])
    bridge_rng = cast(dict[str, Any], bridge_descriptor["rng"])
    bridge_claims = cast(dict[str, Any], bridge_descriptor["claims"])
    if (
        bridge_descriptor["status"] != "implemented_unqualified"
        or bridge_descriptor["adapter_consumers"]
        != ["adapted_full_rainbow", "adapted_ppo_gru"]
        or bridge_source["source_review_complete"] is not False
        or bridge_source["source_closure_bound"] is not False
        or bridge_task["environment_id"] != _ENVIRONMENT_ID
        or bridge_task["observation_type"] != _OBSERVATION_TYPE
        or bridge_task["aperture_size"] != _APERTURE_SIZE
        or bridge_task["horizon"] != _HORIZON
        or bridge_runtime["runtime_parity_executed"] is not False
        or bridge_runtime["runtime_qualified"] is not False
        or bridge_runtime["per_step_host_api_jitted"] is not False
        or bridge_rng["identity"]
        != foragax_bridge.MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE
        or bridge_rng["root"]
        != "jax.random.key(environment_seed,impl=threefry2x32)"
        or bridge_rng["agent_key_accepted"] is not False
        or any(value is not False for value in bridge_claims.values())
    ):
        raise AssertionError("shared Foragax bridge contract drift")
    rainbow_runner_raw = (
        full_rainbow_runner.canonical_full_rainbow_runner_descriptor_bytes()
    )
    rainbow_runner_descriptor = full_rainbow_runner.parse_full_rainbow_runner_descriptor(
        rainbow_runner_raw
    )
    if (
        hashlib.sha256(rainbow_runner_raw).hexdigest()
        != full_rainbow_runner.FULL_RAINBOW_RUNNER_DESCRIPTOR_SHA256
        or rainbow_runner_descriptor["candidate_id"] != "adapted_full_rainbow"
        or rainbow_runner_descriptor["status"] != "implemented_unqualified"
        or any(
            value is not False
            for value in cast(dict[str, Any], rainbow_runner_descriptor["claims"]).values()
        )
    ):
        raise AssertionError("Full Rainbow runner contract drift")
    ppo_runner_raw = (
        ppo_gru_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
    )
    ppo_runner_descriptor = ppo_gru_runner.parse_matched_v3_ppo_gru_runner_descriptor(
        ppo_runner_raw
    )
    if (
        hashlib.sha256(ppo_runner_raw).hexdigest()
        != ppo_gru_runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
        or ppo_runner_descriptor["candidate_id"] != "adapted_ppo_gru"
        or ppo_runner_descriptor["status"] != "implemented_runtime_unqualified"
        or any(
            value is not False
            for value in cast(dict[str, Any], ppo_runner_descriptor["claims"]).values()
        )
    ):
        raise AssertionError("PPO-GRU runner contract drift")
    reward_bundle_raw = (
        adapter_reward_bundle.canonical_adapter_reward_bundle_descriptor_bytes()
    )
    reward_bundle_descriptor = (
        adapter_reward_bundle.parse_adapter_reward_bundle_descriptor(reward_bundle_raw)
    )
    if (
        hashlib.sha256(reward_bundle_raw).hexdigest()
        != adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
        or reward_bundle_descriptor["status"] != "implemented_unqualified"
        or list(reward_bundle_descriptor["candidate_bindings"])
        != list(_ADAPTER_IDENTITIES)
        or any(
            value is not False
            for value in cast(dict[str, Any], reward_bundle_descriptor["claims"]).values()
        )
    ):
        raise AssertionError("adapter reward bundle contract drift")
    publication_raw = (
        adapter_reward_publication.canonical_adapter_reward_publication_descriptor_bytes()
    )
    publication_descriptor = (
        adapter_reward_publication.parse_adapter_reward_publication_descriptor(
            publication_raw
        )
    )
    publication_dependency = cast(
        dict[str, Any], publication_descriptor["dependency"]
    )
    if (
        adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION
        != "alberta.forager_matched_v3.adapter_reward_publication_descriptor.v1"
        or adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
        != "alberta.forager_matched_v3.adapter_reward_publication.v1"
        or adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_STATUS
        != "implemented_unexecuted"
        or adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        != _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or hashlib.sha256(publication_raw).hexdigest()
        != _ADAPTER_REWARD_PUBLICATION_DESCRIPTOR_SHA256
        or publication_descriptor["status"] != "implemented_unexecuted"
        or publication_descriptor["candidate_consumers"]
        != list(_ADAPTER_IDENTITIES)
        or publication_dependency["source_path"]
        != _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_PATH
        or publication_dependency["source_sha256"]
        != _ADAPTER_REWARD_BUNDLE_IMPLEMENTATION_SHA256
        or publication_dependency["descriptor_schema_version"]
        != adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SCHEMA_VERSION
        or publication_dependency["descriptor_sha256"]
        != adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_DESCRIPTOR_SHA256
        or publication_dependency["manifest_schema_version"]
        != adapter_reward_bundle.ADAPTER_REWARD_BUNDLE_MANIFEST_SCHEMA_VERSION
        or any(
            value is not False
            for value in cast(dict[str, Any], publication_descriptor["claims"]).values()
        )
    ):
        raise AssertionError("adapter reward publication contract drift")
    if (
        full_rainbow.FULL_RAINBOW_ADAPTER_STATUS != "implemented_unqualified"
        or full_rainbow.FULL_RAINBOW_CONFIG_SHA256
        != _ADAPTER_IDENTITIES["adapted_full_rainbow"]["configuration_sha256"]
        or full_rainbow.FULL_RAINBOW_DESCRIPTOR_SHA256
        != _ADAPTER_IDENTITIES["adapted_full_rainbow"]["adapter_descriptor_sha256"]
    ):
        raise AssertionError("Full Rainbow adapter identity drift")
    rainbow_descriptor = full_rainbow.validate_matched_v3_full_rainbow_descriptor(
        full_rainbow.canonical_matched_v3_full_rainbow_descriptor_bytes()
    )
    rainbow_accounting = cast(
        dict[str, Any], rainbow_descriptor["exact_operation_accounting"]
    )
    rainbow_source = cast(dict[str, Any], rainbow_descriptor["source"])
    rainbow_runner = cast(dict[str, Any], rainbow_descriptor["runner"])
    rainbow_rng = cast(
        dict[str, Any], full_rainbow.canonical_full_rainbow_config()["rng"]
    )
    if (
        rainbow_descriptor["claims"]["configuration_complete"] is not True
        or rainbow_descriptor["claims"]["core_implementation_complete"] is not True
        or rainbow_descriptor["claims"]["execution_ready"] is not False
        or rainbow_source["upstream_review_anchors_bound"] is not True
        or rainbow_source["source_closure_bound"] is not False
        or rainbow_runner["core_update_primitive_implemented"] is not True
        or rainbow_runner["full_horizon_runner_implemented"] is not False
        or rainbow_runner["result_writer_implemented"] is not False
        or rainbow_runner["qualification_receipt_implemented"] is not False
        or rainbow_rng["seed_domain"] != "uint31"
        or rainbow_rng["jax_prng_implementation"] != "threefry2x32"
        or rainbow_rng["environment_root_consumed_by_core"] is not False
        or rainbow_rng["agent_root_initializes_and_drives_network"] is not True
        or rainbow_rng[
            "namespaced_agent_root_remains_disjoint_from_direct_environment_root"
        ]
        is not True
        or rainbow_accounting["environment_interactions"] != _HORIZON
        or rainbow_accounting["optimizer_updates"]
        != _FULL_RAINBOW_OPTIMIZER_UPDATES
        or rainbow_accounting["target_network_refreshes"]
        != _FULL_RAINBOW_TARGET_REFRESHES
    ):
        raise AssertionError("Full Rainbow adapter accounting drift")
    ppo_configuration = ppo_gru.matched_v3_ppo_gru_configuration()
    ppo_configuration_payload = ppo_configuration.to_dict()
    ppo_descriptor = ppo_gru.parse_matched_v3_ppo_gru_source_descriptor(
        ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes()
    )
    ppo_upstream = cast(dict[str, Any], ppo_descriptor["upstream"])
    ppo_seed_contract = cast(
        dict[str, Any], ppo_configuration_payload["seed_contract"]
    )
    ppo_claims = cast(dict[str, Any], ppo_descriptor["claims"])
    if (
        ppo_configuration_payload["status"] != "implemented_unqualified"
        or ppo_gru.PPO_GRU_CONFIGURATION_SHA256
        != _ADAPTER_IDENTITIES["adapted_ppo_gru"]["configuration_sha256"]
        or ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
        != _ADAPTER_IDENTITIES["adapted_ppo_gru"]["adapter_descriptor_sha256"]
        or ppo_claims["configuration_complete"] is not True
        or ppo_claims["core_implementation_complete"] is not True
        or ppo_claims["validated_epoch_driver_complete"] is not False
        or ppo_claims["full_forager_runner_complete"] is not False
        or ppo_claims["execution_ready"] is not False
        or ppo_upstream["upstream_review_anchors_bound"] is not True
        or ppo_upstream["source_closure_bound"] is not False
        or ppo_seed_contract["environment_seed"] != "required_exact_uint31_root"
        or ppo_seed_contract["agent_seed"] != "required_exact_uint31_root"
        or ppo_seed_contract["prng_implementation"] != "threefry2x32"
        or ppo_seed_contract["roots_are_logically_separate_consumption_chains"]
        is not True
        or ppo_seed_contract["equal_numeric_values_correlate_key_streams"] is not True
        or ppo_seed_contract["statistical_independence_claimed"] is not False
        or ppo_descriptor["runner_blockers"]
        != [
            "qualified_foragax_environment_bridge_missing",
            "full_horizon_compilation_and_memory_profile_unqualified",
            "environment_trace_and_rng_parity_unqualified",
            "validated_epoch_driver_unimplemented",
            "artifact_writer_and_execution_receipt_unimplemented",
        ]
        or ppo_configuration.horizon != _HORIZON
        or ppo_configuration.rollout_steps != _PPO_GRU_ROLLOUT_STEPS
        or ppo_configuration.rollout_count != _PPO_GRU_ROLLOUT_COUNT
        or ppo_configuration.segments_per_rollout != _PPO_GRU_SEGMENTS_PER_EPOCH
        or ppo_configuration.update_epochs != _PPO_GRU_EPOCHS
        or ppo_configuration.optimizer_update_count != _PPO_GRU_OPTIMIZER_UPDATES
    ):
        raise AssertionError("PPO-GRU adapter identity or accounting drift")
    source_pins = {
        pin["repository_id"]: pin
        for pin in universe.matched_v3_development_universe_descriptor()["source_pins"]
    }
    dopamine_files = {
        item["path"]: item["sha256"]
        for item in source_pins["dopamine"]["relevant_files"]
    }
    rainbow_files = {
        item["path"]: item["sha256"] for item in rainbow_descriptor["source"]["files"]
    }
    pobax_files = {
        item["path"]: item["sha256"]
        for item in source_pins["pobax"]["relevant_files"]
    }
    if dopamine_files != rainbow_files or pobax_files != dict(
        ppo_gru.REQUIRED_POBAX_SOURCE_SHA256_BY_PATH
    ):
        raise AssertionError("adapter upstream relevant-file pin drift")


def _validate_plan(value: Mapping[str, Any]) -> None:
    _validate_dependency_pins()
    _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "bindings",
                "task",
                "source_repositories",
                "upstream_execution_source",
                "external_materializer",
                "local_execution_source",
                "shared_environment_bridge",
                "adapter_result_conversion",
                "adapter_result_publication",
                "candidate_ids",
                "candidates",
                "update_accounting_constants",
                "readiness_blockers",
                "configuration_complete",
                "execution_ready",
                "execution_authorized",
                "scientific_promotion_allowed",
                "universal_sota_claim_allowed",
                "limitations",
            }
        ),
        "configuration plan",
    )
    if value["schema_version"] != CONFIGURATION_PLAN_SCHEMA_VERSION:
        raise AssertionError("configuration-plan schema drift")
    if value["bindings"] != {
        "candidate_universe_schema_version": (
            universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION
        ),
        "candidate_universe_sha256": _CANDIDATE_UNIVERSE_SHA256,
        "cumulative_reward_metric_schema_version": (
            protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION
        ),
        "cumulative_reward_metric_sha256": _CUMULATIVE_REWARD_METRIC_SHA256,
        "trial_block_generator_plan_schema_version": (
            protocol.TRIAL_BLOCK_GENERATOR_PLAN_SCHEMA_VERSION
        ),
        "trial_block_generator_plan_sha256": _TRIAL_BLOCK_GENERATOR_PLAN_SHA256,
    }:
        raise AssertionError("v3 universe/metric/trial-plan binding drift")
    if value["task"] != {
        "environment_id": _ENVIRONMENT_ID,
        "observation_type": _OBSERVATION_TYPE,
        "aperture_size": _APERTURE_SIZE,
        "horizon": _HORIZON,
    }:
        raise AssertionError("matched-v3 task binding drift")
    if tuple(value["candidate_ids"]) != universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS:
        raise AssertionError("candidate ID order drift")
    candidates = cast(list[dict[str, Any]], value["candidates"])
    if tuple(item["candidate_id"] for item in candidates) != (
        universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
    ):
        raise AssertionError("candidate record order drift")
    if len(candidates) != len({item["candidate_id"] for item in candidates}):
        raise AssertionError("candidate IDs are not unique")
    by_id = {record["candidate_id"]: record for record in candidates}
    for record in candidates:
        candidate_id = cast(str, record["candidate_id"])
        _require_exact_keys(
            record,
            frozenset(
                {
                    "candidate_id",
                    "analysis_role",
                    "development_selection_group",
                    "configuration",
                    "execution_shape",
                    "diagnostic_policy",
                    "rng_contract",
                    "observation_contract",
                    "execution_ready",
                    "execution_authorized",
                    "scientific_promotion_allowed",
                }
            ),
            f"candidate record {candidate_id}",
        )
        configuration = cast(dict[str, Any], record["configuration"])
        if candidate_id in _LOCAL_CONFIGURATION_SHA256:
            configuration_keys = frozenset(
                {
                    "kind",
                    "repository_id",
                    "builder_id",
                    "builder_status",
                    "builder_descriptor_sha256",
                    "source_descriptor_sha256",
                    "worker_envelope_sha256",
                    "configuration_complete",
                    "source_snapshot_status",
                }
            )
        elif candidate_id in _ADAPTER_IDENTITIES:
            configuration_keys = frozenset(
                {
                    "kind",
                    "repository_id",
                    "configuration_schema_version",
                    "configuration_sha256",
                    "adapter_descriptor_schema_version",
                    "adapter_descriptor_sha256",
                    "implementation_path",
                    "implementation_source_sha256",
                    "runner_descriptor_schema_version",
                    "runner_descriptor_sha256",
                    "runner_implementation_path",
                    "runner_implementation_source_sha256",
                    "runner_result_receipt_schema_version",
                    "status",
                    "configuration_complete",
                    "core_implementation_complete",
                    "full_runner_complete",
                    "in_memory_result_conversion_complete",
                    "runtime_qualified",
                    "durable_result_publication_complete",
                    "upstream_review_anchors_bound",
                    "source_closure_bound",
                    "source_snapshot_status",
                }
            )
        else:
            configuration_keys = frozenset(
                {
                    "kind",
                    "repository_id",
                    "status",
                    "original_path",
                    "original_sha256",
                    "transform_descriptor",
                    "transform_descriptor_sha256",
                    "derived_sha256",
                    "configuration_complete",
                    "source_snapshot_status",
                }
            )
        _require_exact_keys(
            configuration,
            configuration_keys,
            f"candidate configuration {candidate_id}",
        )
        if type(configuration["configuration_complete"]) is not bool:
            raise AssertionError(
                f"configuration_complete must be an exact boolean for {candidate_id}"
            )
        execution_shape = cast(dict[str, Any], record["execution_shape"])
        _require_exact_keys(
            execution_shape,
            frozenset(
                {
                    "horizon",
                    "interaction_count",
                    "horizon_unit",
                    "entrypoint_path",
                    "entrypoint_sha256",
                    "update_schedule",
                    "rollout_steps",
                    "rollout_count",
                    "epochs",
                    "minibatches_per_epoch",
                    "optimizer_update_count",
                    "optimizer_update_subcounts",
                    "replay_capacity_transitions",
                    "non_gradient_operations",
                    "exact_workload_argv",
                    "default_without_override",
                }
            ),
            f"execution shape {candidate_id}",
        )
        _require_exact_keys(
            cast(dict[str, Any], execution_shape["replay_capacity_transitions"]),
            frozenset({"main", "permanent"}),
            f"replay capacities {candidate_id}",
        )
        _require_exact_keys(
            cast(dict[str, Any], execution_shape["non_gradient_operations"]),
            frozenset(
                {
                    "causal_transition_updates",
                    "target_snapshot_refreshes",
                    "redo_recycles",
                    "permanent_update_events",
                    "transient_parameter_decays",
                }
            ),
            f"non-gradient operations {candidate_id}",
        )
        _require_exact_keys(
            cast(dict[str, Any], record["diagnostic_policy"]),
            frozenset(
                {
                    "status",
                    "ntk_enabled",
                    "reference_collection_enabled",
                    "configured_reference_steps",
                    "effective_reference_steps",
                    "chunked_ref_source_value",
                    "chunked_ref_effective_value",
                    "chunked_ref_inert",
                    "weight_norm_enabled",
                    "weight_drift_enabled",
                    "video_length_steps",
                    "save_every_steps",
                    "video_every_steps",
                    "checkpoint_milestones_within_horizon",
                    "fresh_empty_checkpoint_root_required",
                    "single_index_video_capture_unavoidable_without_source_derivation",
                    "single_index_video_window_steps",
                }
            ),
            f"diagnostic policy {candidate_id}",
        )
        _require_exact_keys(
            cast(dict[str, Any], record["rng_contract"]),
            frozenset(
                {
                    "environment_seed_namespace",
                    "agent_seed_namespace",
                    "environment_seed_transport",
                    "agent_seed_transport",
                    "transport_status",
                    "transport_artifact_scope",
                    "environment_rng_independence_qualified",
                    "agent_consumption_invariance_proved",
                    "runtime_trace_verified",
                    "source_closure_bound",
                    "statistical_independence_claimed",
                    "environment_transport_descriptor_sha256",
                    "agent_transport_descriptor_sha256",
                    "runner_transport_descriptor_sha256",
                }
            ),
            f"RNG contract {candidate_id}",
        )
        _require_exact_keys(
            cast(dict[str, Any], record["observation_contract"]),
            frozenset(
                {
                    "access",
                    "observation_type",
                    "aperture_size",
                    "inferentially_matched",
                    "qualification_required",
                }
            ),
            f"observation contract {candidate_id}",
        )
        if any(
            record[field] is not False
            for field in (
                "execution_ready",
                "execution_authorized",
                "scientific_promotion_allowed",
            )
        ):
            raise AssertionError("candidate readiness or authority became true")
        if record["execution_shape"]["horizon"] != protocol.MATCHED_V3_HORIZON:
            raise AssertionError("candidate horizon drift")
        if execution_shape != _execution_shape(candidate_id):
            raise AssertionError(f"execution-shape content drift for {candidate_id}")
        if record["diagnostic_policy"] != _diagnostic_policy(candidate_id):
            raise AssertionError(f"diagnostic-policy content drift for {candidate_id}")
        if record["observation_contract"] != _observation_contract(candidate_id):
            raise AssertionError(f"observation-contract content drift for {candidate_id}")
        rng = record["rng_contract"]
        if (
            rng != _rng_contract(candidate_id)
            or rng["environment_rng_independence_qualified"] is not False
            or rng["agent_consumption_invariance_proved"] is not False
        ):
            raise AssertionError("candidate RNG readiness drift")
    for candidate_id in _LOCAL_CONFIGURATION_SHA256:
        if by_id[candidate_id]["configuration"] != _local_configuration(candidate_id):
            raise AssertionError(f"generated-local configuration drift for {candidate_id}")
    for candidate_id, spec in _EXTERNAL_SPECS.items():
        if by_id[candidate_id]["configuration"] != _external_configuration(candidate_id):
            raise AssertionError(f"external configuration record drift for {candidate_id}")
        canonical_descriptor_sha = derivation.canonical_descriptor_sha256(spec["descriptor"])
        if canonical_descriptor_sha != spec["descriptor_sha256"]:
            raise AssertionError(f"transform descriptor drift for {candidate_id}")
    for candidate_id in _ADAPTER_IDENTITIES:
        if by_id[candidate_id]["configuration"] != _adapter_configuration(candidate_id):
            raise AssertionError(f"adapter-core configuration drift for {candidate_id}")
    ppo_records = {
        record["candidate_id"]: record
        for record in candidates
        if record["candidate_id"] in {"isolated_ppo_generic", "isolated_rtu_paper_scale"}
    }
    for record in ppo_records.values():
        shape = record["execution_shape"]
        if (
            shape["rollout_count"] != _PPO_ROLLOUT_COUNT
            or shape["rollout_steps"] * shape["rollout_count"]
            != protocol.MATCHED_V3_HORIZON
            or shape["exact_workload_argv"] != _ppo_argv()
            or shape["default_without_override"]["accepted"] is not False
        ):
            raise AssertionError("PPO exact-update override drift")
    upstream_execution = cast(dict[str, Any], value["upstream_execution_source"])
    _require_exact_keys(
        upstream_execution,
        frozenset(
            {
                "repository_id",
                "commit_git_sha1",
                "tree_git_sha1",
                "archive_sha256",
                "archive_size_bytes",
                "review_anchors",
                "seed_transport_patch_set_schema_version",
                "seed_transport_patch_set_descriptor_sha256",
                "seed_transport_patch_set_scope",
                "seed_transport_patch_set_status",
                "derived_source_files",
                "full_dependency_inventory_bound",
                "materialization_status",
                "runtime_trace_verified",
            }
        ),
        "upstream execution source",
    )
    expected_derived_sources = [
        {
            "path": path,
            "sha256": external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH[
                path
            ],
        }
        for path in external_seed_transport.SOURCE_PATHS
    ]
    if (
        upstream_execution["derived_source_files"] != expected_derived_sources
        or upstream_execution["seed_transport_patch_set_descriptor_sha256"]
        != _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256
        or upstream_execution["full_dependency_inventory_bound"] is not False
        or upstream_execution["materialization_status"]
        != "implementation_available_no_accepted_production_manifest"
        or upstream_execution["runtime_trace_verified"] is not False
    ):
        raise AssertionError("external seed-transport source binding drift")
    materializer = cast(dict[str, Any], value["external_materializer"])
    _require_exact_keys(
        materializer,
        frozenset(
            {
                "manifest_schema_version",
                "identity_schema_version",
                "identity_sha256",
                "implementation_path",
                "implementation_source_sha256",
                "status",
                "production_materialization_accepted",
                "production_manifest_sha256",
                "materialized_source_closure_bound",
                "archive_bytes_verified",
                "runtime_dependencies_qualified",
                "execution_authorized",
            }
        ),
        "external materializer",
    )
    if materializer != _external_materializer():
        raise AssertionError("external materializer binding drift")
    local_execution = cast(dict[str, Any], value["local_execution_source"])
    _require_exact_keys(
        local_execution,
        frozenset(
            {
                "repository_id",
                "snapshot_sha256",
                "inventory_sha256",
                "configuration_source_descriptor_sha256",
                "configuration_builder_descriptor_sha256",
                "status",
            }
        ),
        "local execution source",
    )
    shared_bridge = cast(dict[str, Any], value["shared_environment_bridge"])
    _require_exact_keys(
        shared_bridge,
        frozenset(
            {
                "schema_version",
                "descriptor_sha256",
                "implementation_path",
                "implementation_source_sha256",
                "status",
                "adapter_consumers",
                "environment_rng_schedule",
                "runtime_parity_executed",
                "runtime_qualified",
                "compiled_chunk_kernel_complete",
                "source_closure_bound",
            }
        ),
        "shared environment bridge",
    )
    if shared_bridge != _shared_environment_bridge():
        raise AssertionError("shared environment bridge binding drift")
    result_conversion = cast(dict[str, Any], value["adapter_result_conversion"])
    _require_exact_keys(
        result_conversion,
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "manifest_schema_version",
                "implementation_path",
                "implementation_source_sha256",
                "status",
                "candidate_consumers",
                "scorer_source_path",
                "scorer_source_sha256",
                "score_receipt_schema_version",
                "reward_container_schema_version",
                "raw_trace_encoding_schema_version",
                "canonical_reward_artifact_size_bytes",
                "in_memory_conversion_complete",
                "filesystem_publication_complete",
                "campaign_ingestion_complete",
                "ingestion_authorized",
            }
        ),
        "adapter result conversion",
    )
    if result_conversion != _adapter_result_conversion():
        raise AssertionError("adapter result conversion binding drift")
    result_publication = cast(dict[str, Any], value["adapter_result_publication"])
    _require_exact_keys(
        result_publication,
        frozenset(
            {
                "descriptor_schema_version",
                "descriptor_sha256",
                "publication_schema_version",
                "implementation_path",
                "implementation_source_sha256",
                "status",
                "candidate_consumers",
                "adapter_reward_bundle_descriptor_schema_version",
                "adapter_reward_bundle_descriptor_sha256",
                "adapter_reward_bundle_manifest_schema_version",
                "implementation_complete",
                "production_publication_accepted",
                "production_publication_file_sha256",
                "campaign_ingestion_complete",
                "ingestion_authorized",
                "runtime_qualified",
                "execution_authorized",
                "scientific_promotion_allowed",
            }
        ),
        "adapter result publication",
    )
    if (
        result_publication["implementation_complete"] is not True
        or result_publication["production_publication_file_sha256"] is not None
        or any(
            result_publication[field] is not False
            for field in (
                "production_publication_accepted",
                "campaign_ingestion_complete",
                "ingestion_authorized",
                "runtime_qualified",
                "execution_authorized",
                "scientific_promotion_allowed",
            )
        )
        or result_publication != _adapter_result_publication()
    ):
        raise AssertionError("adapter result publication binding drift")
    source_pins = cast(list[dict[str, Any]], value["source_repositories"])
    upstream = next(
        item for item in source_pins if item["repository_id"] == _UPSTREAM_REPOSITORY_ID
    )
    if (
        upstream["commit_git_sha1"],
        upstream["tree_git_sha1"],
        upstream["archive_sha256"],
        upstream["archive_size_bytes"],
    ) != (
        _UPSTREAM_COMMIT,
        _UPSTREAM_TREE,
        _UPSTREAM_ARCHIVE_SHA256,
        _UPSTREAM_ARCHIVE_SIZE,
    ):
        raise AssertionError("Foragax agent source-repository pin drift")
    dopamine = next(item for item in source_pins if item["repository_id"] == "dopamine")
    relevant_paths = {item["path"] for item in dopamine["relevant_files"]}
    corrected_gin = "dopamine/jax/agents/full_rainbow/configs/full_rainbow.gin"
    obsolete_gin = "dopamine/jax/agents/full_rainbow/full_rainbow.gin"
    if corrected_gin not in relevant_paths or obsolete_gin in relevant_paths:
        raise AssertionError("Dopamine Full Rainbow configuration path drift")
    blockers = cast(list[dict[str, Any]], value["readiness_blockers"])
    seen_blocker_ids: set[str] = set()
    known_candidate_ids = set(universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS)
    for blocker in blockers:
        _require_exact_keys(
            blocker,
            frozenset({"blocker_id", "candidate_ids"}),
            "readiness blocker",
        )
        blocker_id = blocker["blocker_id"]
        candidate_ids = blocker["candidate_ids"]
        if (
            type(blocker_id) is not str
            or not blocker_id
            or blocker_id in seen_blocker_ids
        ):
            raise AssertionError("readiness blocker ID drift")
        seen_blocker_ids.add(blocker_id)
        if (
            type(candidate_ids) is not list
            or not candidate_ids
            or any(type(item) is not str for item in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
            or not set(candidate_ids) <= known_candidate_ids
        ):
            raise AssertionError(f"readiness blocker candidate drift for {blocker_id}")
    if value["configuration_complete"] is not True:
        raise AssertionError("configuration plan completeness became false")
    if any(
        value[field] is not False
        for field in (
            "execution_ready",
            "execution_authorized",
            "scientific_promotion_allowed",
            "universal_sota_claim_allowed",
        )
    ):
        raise AssertionError("configuration plan readiness or authority became true")
    expected = _configuration_plan()
    if _canonical_bytes(value) != _canonical_bytes(expected):
        raise AssertionError("configuration plan nested content drift")
    _assert_unaliased_plain_json(value)
    _canonical_bytes(value)


_MATCHED_V3_CONFIGURATION_PLAN: Final = _configuration_plan()
_validate_plan(_MATCHED_V3_CONFIGURATION_PLAN)
_MATCHED_V3_CONFIGURATION_PLAN_BYTES: Final = _canonical_bytes(
    _MATCHED_V3_CONFIGURATION_PLAN
)
MATCHED_V3_CONFIGURATION_PLAN_SHA256: Final = (
    "55680786cf5a76aa2a51de35205a9bb543420c7f27aa41846d40a94dcf965fc7"
)
if not hmac.compare_digest(
    hashlib.sha256(_MATCHED_V3_CONFIGURATION_PLAN_BYTES).hexdigest(),
    MATCHED_V3_CONFIGURATION_PLAN_SHA256,
):
    raise AssertionError("matched-v3 configuration plan digest drift")


def _frozen_plan_snapshot() -> dict[str, Any]:
    """Decode a fresh snapshot from the authenticated construction output."""

    try:
        decoded = json.loads(_MATCHED_V3_CONFIGURATION_PLAN_BYTES.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ForagerMatchedV3ConfigurationPlanError(
            "frozen configuration-plan bytes could not be decoded"
        ) from exc
    if type(decoded) is not dict:  # pragma: no cover
        raise ForagerMatchedV3ConfigurationPlanError(
            "frozen configuration-plan bytes must encode a plain object"
        )
    return cast(dict[str, Any], decoded)


def _frozen_record_index() -> dict[str, dict[str, Any]]:
    """Index candidate records from a fresh canonical-byte snapshot."""

    snapshot = _frozen_plan_snapshot()
    records = cast(list[dict[str, Any]], snapshot["candidates"])
    return {cast(str, record["candidate_id"]): record for record in records}


def matched_v3_configuration_plan_descriptor() -> dict[str, Any]:
    """Return a detached plain-JSON snapshot of the unexecuted plan."""

    return _frozen_plan_snapshot()


def canonical_matched_v3_configuration_plan_bytes() -> bytes:
    """Return exact canonical bytes for the frozen configuration plan."""

    return _MATCHED_V3_CONFIGURATION_PLAN_BYTES


def matched_v3_configuration_plan_sha256() -> str:
    """Return the exact digest of the canonical configuration-plan bytes."""

    return MATCHED_V3_CONFIGURATION_PLAN_SHA256


def parse_matched_v3_configuration_plan_artifact(raw: bytes) -> dict[str, Any]:
    """Accept only the exact canonical, authority-denying plan artifact."""

    if not isinstance(raw, bytes):
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration-plan artifact must be bytes"
        )
    if len(raw) > _MAX_PLAN_BYTES:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration-plan artifact exceeds its byte limit"
        )
    if not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        MATCHED_V3_CONFIGURATION_PLAN_SHA256,
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration-plan artifact does not match the frozen digest"
        )
    if raw != _MATCHED_V3_CONFIGURATION_PLAN_BYTES:
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration-plan artifact is not the exact canonical encoding"
        )
    return _frozen_plan_snapshot()


def configuration_record(candidate_id: str) -> dict[str, Any]:
    """Return one detached candidate record from the exact 28-arm plan."""

    if type(candidate_id) is not str:
        raise ForagerMatchedV3ConfigurationPlanError("candidate_id must be a string")
    record = _frozen_record_index().get(candidate_id)
    if record is None:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"unknown candidate {candidate_id!r}"
        )
    return _plain_snapshot(record)


def derive_and_verify_external_configuration(
    candidate_id: str,
    original_bytes: bytes,
) -> derivation.DerivedConfiguration:
    """Replay one exact external transform after proving its raw byte identity."""

    if type(candidate_id) is not str:
        raise ForagerMatchedV3ConfigurationPlanError("candidate_id must be a string")
    try:
        configuration = configuration_record(candidate_id)["configuration"]
    except ForagerMatchedV3ConfigurationPlanError:
        configuration = None
    if type(configuration) is not dict or configuration.get("kind") != "derived_external":
        raise ForagerMatchedV3ConfigurationPlanError(
            f"candidate {candidate_id!r} has no complete external configuration"
        )
    if not isinstance(original_bytes, bytes):
        raise ForagerMatchedV3ConfigurationPlanError(
            "external original configuration must be bytes"
        )
    raw = bytes(original_bytes)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(
        actual_sha256,
        cast(str, configuration["original_sha256"]),
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"external configuration raw SHA-256 mismatch for {candidate_id}"
        )
    try:
        result = derivation.derive_configuration(
            raw,
            cast(dict[str, Any], configuration["transform_descriptor"]),
        )
    except derivation.ForagerMatchedV3ConfigurationError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"external configuration transform failed for {candidate_id}"
        ) from exc
    if (
        not hmac.compare_digest(
            result.original_sha256,
            cast(str, configuration["original_sha256"]),
        )
        or not hmac.compare_digest(
            result.descriptor_sha256,
            cast(str, configuration["transform_descriptor_sha256"]),
        )
        or not hmac.compare_digest(
            result.derived_sha256,
            cast(str, configuration["derived_sha256"]),
        )
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"external configuration derivation digest drift for {candidate_id}"
        )
    return result


def build_and_verify_local_configuration(
    candidate_id: str,
) -> local_configuration.BuiltMatchedV3LocalConfiguration:
    """Build one local envelope and verify every plan-bound identity."""

    if type(candidate_id) is not str:
        raise ForagerMatchedV3ConfigurationPlanError("candidate_id must be a string")
    try:
        record = configuration_record(candidate_id)["configuration"]
    except ForagerMatchedV3ConfigurationPlanError:
        record = None
    if type(record) is not dict or record.get("kind") != "generated_local":
        raise ForagerMatchedV3ConfigurationPlanError(
            f"candidate {candidate_id!r} has no local configuration builder"
        )
    expected = cast(str, record["worker_envelope_sha256"])
    try:
        built = local_configuration.build_matched_v3_local_configuration(candidate_id)
    except local_configuration.ForagerMatchedV3LocalConfigurationError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"local configuration build failed for {candidate_id}"
        ) from exc
    if (
        not hmac.compare_digest(built.configuration_sha256, expected)
        or built.builder_id != record["builder_id"]
        or built.status != record["builder_status"]
        or built.source_descriptor_sha256 != record["source_descriptor_sha256"]
        or built.builder_descriptor_sha256 != record["builder_descriptor_sha256"]
        or built.configuration_complete is not True
        or built.execution_ready is not False
        or built.execution_authorized is not False
        or built.scientific_promotion_allowed is not False
        or built.universal_sota_claim_allowed is not False
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"local configuration binding drift for {candidate_id}"
        )
    return built


def verify_adapter_core_artifacts(candidate_id: str) -> dict[str, Any]:
    """Replay one adapter's exact core config and descriptor without running it."""

    if type(candidate_id) is not str:
        raise ForagerMatchedV3ConfigurationPlanError("candidate_id must be a string")
    try:
        expected = configuration_record(candidate_id)["configuration"]
    except ForagerMatchedV3ConfigurationPlanError:
        expected = None
    if type(expected) is not dict or expected.get("kind") != "derived_local_adapter":
        raise ForagerMatchedV3ConfigurationPlanError(
            f"candidate {candidate_id!r} has no adapter-core artifacts"
        )
    try:
        if candidate_id == "adapted_full_rainbow":
            configuration_bytes = full_rainbow.canonical_full_rainbow_config_bytes()
            descriptor_bytes = (
                full_rainbow.canonical_matched_v3_full_rainbow_descriptor_bytes()
            )
            descriptor = full_rainbow.validate_matched_v3_full_rainbow_descriptor(
                descriptor_bytes
            )
            config_binding = descriptor["configuration"]
            descriptor_config_schema = config_binding["schema_version"]
            descriptor_config_sha256 = config_binding["sha256"]
        else:
            configuration_bytes = (
                ppo_gru.canonical_matched_v3_ppo_gru_configuration_bytes()
            )
            descriptor_bytes = (
                ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes()
            )
            ppo_gru.parse_matched_v3_ppo_gru_configuration(configuration_bytes)
            descriptor = ppo_gru.parse_matched_v3_ppo_gru_source_descriptor(
                descriptor_bytes
            )
            config_binding = descriptor["derived_implementation"]
            descriptor_config_schema = config_binding["configuration_schema_version"]
            descriptor_config_sha256 = config_binding["configuration_sha256"]
    except (
        full_rainbow.FullRainbowContractError,
        ppo_gru.ForagerMatchedV3PPOGRUError,
    ) as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"adapter-core artifact validation failed for {candidate_id}"
        ) from exc
    if (
        not hmac.compare_digest(
            hashlib.sha256(configuration_bytes).hexdigest(),
            expected["configuration_sha256"],
        )
        or not hmac.compare_digest(
            hashlib.sha256(descriptor_bytes).hexdigest(),
            expected["adapter_descriptor_sha256"],
        )
        or descriptor["candidate_id"] != candidate_id
        or descriptor_config_schema != expected["configuration_schema_version"]
        or descriptor_config_sha256 != expected["configuration_sha256"]
        or descriptor["claims"]["configuration_complete"] is not True
        or descriptor["claims"]["execution_ready"] is not False
        or descriptor["claims"]["execution_authorized"] is not False
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"adapter-core artifact binding drift for {candidate_id}"
        )
    return cast(dict[str, Any], expected)


def verify_adapter_runner_artifact(candidate_id: str) -> dict[str, Any]:
    """Replay one exact full-runner descriptor and source binding without execution."""

    if type(candidate_id) is not str:
        raise ForagerMatchedV3ConfigurationPlanError(
            "candidate_id must be an exact string"
        )
    try:
        configuration = _frozen_record_index()[candidate_id]["configuration"]
    except KeyError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"unknown candidate_id: {candidate_id!r}"
        ) from exc
    if type(configuration) is not dict or configuration.get("kind") != (
        "derived_local_adapter"
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"candidate {candidate_id!r} has no adapter runner artifact"
        )
    expected = cast(dict[str, Any], configuration)
    try:
        if candidate_id == "adapted_full_rainbow":
            raw = full_rainbow_runner.canonical_full_rainbow_runner_descriptor_bytes()
            descriptor = full_rainbow_runner.parse_full_rainbow_runner_descriptor(raw)
            result_schema = (
                full_rainbow_runner.FULL_RAINBOW_RESULT_RECEIPT_SCHEMA_VERSION
            )
            expected_status = "implemented_unqualified"
            module_file = full_rainbow_runner.__file__
        elif candidate_id == "adapted_ppo_gru":
            raw = (
                ppo_gru_runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
            )
            descriptor = ppo_gru_runner.parse_matched_v3_ppo_gru_runner_descriptor(raw)
            result_schema = ppo_gru_runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
            expected_status = "implemented_runtime_unqualified"
            module_file = ppo_gru_runner.__file__
        else:
            raise ForagerMatchedV3ConfigurationPlanError(
                f"candidate {candidate_id!r} has no adapter runner artifact"
            )
    except (
        full_rainbow_runner.FullRainbowRunnerContractError,
        ppo_gru_runner.ForagerMatchedV3PPOGRURunnerError,
    ) as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"adapter runner artifact validation failed for {candidate_id}"
        ) from exc
    claims = cast(dict[str, Any], descriptor["claims"])
    if (
        descriptor["schema_version"] != expected["runner_descriptor_schema_version"]
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), expected["runner_descriptor_sha256"]
        )
        or descriptor["candidate_id"] != candidate_id
        or descriptor["status"] != expected_status
        or result_schema != expected["runner_result_receipt_schema_version"]
        or not hmac.compare_digest(
            _source_sha256(module_file, expected["runner_implementation_path"]),
            expected["runner_implementation_source_sha256"],
        )
        or any(value is not False for value in claims.values())
        or expected["full_runner_complete"] is not True
        or expected["runtime_qualified"] is not False
        or expected["durable_result_publication_complete"] is not False
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            f"adapter runner artifact binding drift for {candidate_id}"
        )
    return {
        key: expected[key]
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


def verify_external_materialization_identity_artifact() -> dict[str, Any]:
    """Fail explicitly because this immutable plan binds the superseded v1 materializer."""

    expected = cast(
        dict[str, Any], _frozen_plan_snapshot()["external_materializer"]
    )
    if (
        expected["manifest_schema_version"]
        != _HISTORICAL_EXTERNAL_MATERIALIZATION_SCHEMA_VERSION
        or expected["identity_schema_version"]
        != _HISTORICAL_EXTERNAL_MATERIALIZATION_IDENTITY_SCHEMA_VERSION
        or expected["identity_sha256"] != _EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
        or expected["implementation_source_sha256"]
        != _EXTERNAL_MATERIALIZER_IMPLEMENTATION_SHA256
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "historical v1 external materializer binding drift"
        )
    raise ForagerMatchedV3ConfigurationPlanError(
        "historical v1 external materializer is superseded and unavailable; "
        "use a separately versioned v2 materialization overlay"
    )


def verify_adapter_reward_bundle_descriptor() -> dict[str, Any]:
    """Replay the strict in-memory adapter result-conversion descriptor."""

    expected = cast(
        dict[str, Any], _frozen_plan_snapshot()["adapter_result_conversion"]
    )
    try:
        raw = adapter_reward_bundle.canonical_adapter_reward_bundle_descriptor_bytes()
        descriptor = adapter_reward_bundle.parse_adapter_reward_bundle_descriptor(raw)
    except adapter_reward_bundle.ForagerMatchedV3AdapterRewardBundleError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "adapter reward bundle descriptor validation failed"
        ) from exc
    if (
        descriptor["schema_version"] != expected["descriptor_schema_version"]
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), expected["descriptor_sha256"]
        )
        or list(descriptor["candidate_bindings"]) != expected["candidate_consumers"]
        or descriptor["scorer"]["source_sha256"] != expected["scorer_source_sha256"]
        or descriptor["scorer"]["canonical_npz_size_bytes"]
        != expected["canonical_reward_artifact_size_bytes"]
        or not hmac.compare_digest(
            _source_sha256(
                adapter_reward_bundle.__file__, expected["implementation_path"]
            ),
            expected["implementation_source_sha256"],
        )
        or any(
            value is not False
            for value in cast(dict[str, Any], descriptor["claims"]).values()
        )
        or expected["filesystem_publication_complete"] is not False
        or expected["campaign_ingestion_complete"] is not False
        or expected["ingestion_authorized"] is not False
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "adapter reward bundle descriptor binding drift"
        )
    return expected


def verify_adapter_reward_publication_descriptor() -> dict[str, Any]:
    """Replay the atomic content-publication descriptor without publishing."""

    snapshot = _frozen_plan_snapshot()
    expected = cast(dict[str, Any], snapshot["adapter_result_publication"])
    conversion = cast(dict[str, Any], snapshot["adapter_result_conversion"])
    try:
        raw = (
            adapter_reward_publication.canonical_adapter_reward_publication_descriptor_bytes()
        )
        descriptor = (
            adapter_reward_publication.parse_adapter_reward_publication_descriptor(raw)
        )
    except (
        adapter_reward_publication.ForagerMatchedV3AdapterRewardPublicationError
    ) as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "adapter reward publication descriptor validation failed"
        ) from exc
    dependency = cast(dict[str, Any], descriptor["dependency"])
    claims = cast(dict[str, Any], descriptor["claims"])
    if (
        descriptor["schema_version"] != expected["descriptor_schema_version"]
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), expected["descriptor_sha256"]
        )
        or descriptor["status"] != expected["status"]
        or descriptor["candidate_consumers"] != expected["candidate_consumers"]
        or adapter_reward_publication.ADAPTER_REWARD_PUBLICATION_SCHEMA_VERSION
        != expected["publication_schema_version"]
        or dependency["source_path"] != conversion["implementation_path"]
        or dependency["source_sha256"]
        != conversion["implementation_source_sha256"]
        or dependency["descriptor_schema_version"]
        != conversion["descriptor_schema_version"]
        or dependency["descriptor_sha256"] != conversion["descriptor_sha256"]
        or dependency["manifest_schema_version"]
        != conversion["manifest_schema_version"]
        or expected["adapter_reward_bundle_descriptor_schema_version"]
        != conversion["descriptor_schema_version"]
        or expected["adapter_reward_bundle_descriptor_sha256"]
        != conversion["descriptor_sha256"]
        or expected["adapter_reward_bundle_manifest_schema_version"]
        != conversion["manifest_schema_version"]
        or not hmac.compare_digest(
            _source_sha256(
                adapter_reward_publication.__file__, expected["implementation_path"]
            ),
            expected["implementation_source_sha256"],
        )
        or any(value is not False for value in claims.values())
        or expected["implementation_complete"] is not True
        or expected["production_publication_file_sha256"] is not None
        or any(
            expected[field] is not False
            for field in (
                "production_publication_accepted",
                "campaign_ingestion_complete",
                "ingestion_authorized",
                "runtime_qualified",
                "execution_authorized",
                "scientific_promotion_allowed",
            )
        )
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "adapter reward publication descriptor binding drift"
        )
    return expected


def verify_shared_environment_bridge_artifact() -> dict[str, Any]:
    """Replay the exact bridge descriptor without granting runtime qualification."""

    expected = cast(
        dict[str, Any], _frozen_plan_snapshot()["shared_environment_bridge"]
    )
    try:
        raw = foragax_bridge.canonical_matched_v3_foragax_bridge_descriptor_bytes()
        descriptor = foragax_bridge.parse_matched_v3_foragax_bridge_descriptor(raw)
    except foragax_bridge.ForagerMatchedV3ForagaxBridgeError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "shared environment bridge artifact validation failed"
        ) from exc
    if (
        not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), expected["descriptor_sha256"]
        )
        or descriptor["schema_version"] != expected["schema_version"]
        or descriptor["adapter_consumers"] != expected["adapter_consumers"]
        or descriptor["rng"]["identity"] != expected["environment_rng_schedule"]
        or descriptor["runtime"]["runtime_parity_executed"] is not False
        or descriptor["runtime"]["runtime_qualified"] is not False
        or descriptor["claims"]["execution_ready"] is not False
        or descriptor["claims"]["execution_authorized"] is not False
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "shared environment bridge artifact binding drift"
        )
    return expected


def derive_and_verify_external_seed_transport(
    sources: dict[str, bytes],
) -> external_seed_transport.DerivedExternalSeedTransport:
    """Derive the exact four-file patch set and verify the plan binding."""

    try:
        derived = external_seed_transport.derive_matched_v3_external_seed_transport(
            sources
        )
    except external_seed_transport.ExternalSeedTransportError as exc:
        raise ForagerMatchedV3ConfigurationPlanError(
            "external two-seed patch-set derivation failed"
        ) from exc
    if (
        not hmac.compare_digest(
            derived.descriptor_sha256,
            _EXTERNAL_SEED_TRANSPORT_DESCRIPTOR_SHA256,
        )
        or dict(derived.source_sha256_by_path)
        != dict(external_seed_transport.EXPECTED_DERIVED_SOURCE_SHA256_BY_PATH)
    ):
        raise ForagerMatchedV3ConfigurationPlanError(
            "external two-seed patch-set binding drift"
        )
    return derived


def verify_external_configuration_set(
    originals: Mapping[str, bytes],
) -> dict[str, derivation.DerivedConfiguration]:
    """Verify the exact 12-record external byte set with no missing or extra IDs."""

    if type(originals) is not dict:
        raise ForagerMatchedV3ConfigurationPlanError(
            "external configuration set must be a plain dictionary"
        )
    actual = set(originals)
    expected = set(EXTERNAL_CONFIGURATION_CANDIDATE_IDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ForagerMatchedV3ConfigurationPlanError(
            f"external configuration IDs differ; missing={missing!r}, extra={extra!r}"
        )
    return {
        candidate_id: derive_and_verify_external_configuration(
            candidate_id,
            originals[candidate_id],
        )
        for candidate_id in EXTERNAL_CONFIGURATION_CANDIDATE_IDS
    }


def _raise_incomplete_configuration(missing: list[str]) -> NoReturn:
    raise ForagerMatchedV3ConfigurationPlanError(
        "configuration plan is incomplete for: " + ", ".join(missing)
    )


def assert_configuration_complete() -> None:
    """Fail while any candidate lacks an exact generated or derived configuration."""

    snapshot = _frozen_plan_snapshot()
    record_by_id = _frozen_record_index()
    missing = []
    if snapshot["configuration_complete"] is not True:
        missing.append("global_configuration_complete")
    missing.extend(
        candidate_id
        for candidate_id in universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
        if record_by_id[candidate_id]["configuration"]["configuration_complete"]
        is not True
    )
    if missing:
        _raise_incomplete_configuration(missing)


def assert_execution_ready() -> None:
    """Fail unless readiness and explicit execution authority are both true."""

    snapshot = _frozen_plan_snapshot()
    record_by_id = _frozen_record_index()
    blockers = cast(list[dict[str, Any]], snapshot["readiness_blockers"])
    unready = [
        candidate_id
        for candidate_id in universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
        if record_by_id[candidate_id]["execution_ready"] is not True
    ]
    unauthorized = [
        candidate_id
        for candidate_id in universe.MATCHED_V3_DEVELOPMENT_CANDIDATE_IDS
        if record_by_id[candidate_id]["execution_authorized"] is not True
    ]
    if (
        blockers
        or unready
        or unauthorized
        or snapshot["execution_ready"] is not True
        or snapshot["execution_authorized"] is not True
    ):
        blocker_ids = ", ".join(item["blocker_id"] for item in blockers)
        raise ForagerMatchedV3ConfigurationPlanError(
            "configuration plan is not execution-ready or execution-authorized; blockers="
            + blocker_ids
        )


__all__ = [
    "CONFIGURATION_PLAN_SCHEMA_VERSION",
    "EXTERNAL_CONFIGURATION_CANDIDATE_IDS",
    "ForagerMatchedV3ConfigurationPlanError",
    "MATCHED_V3_CONFIGURATION_PLAN_SHA256",
    "assert_configuration_complete",
    "assert_execution_ready",
    "build_and_verify_local_configuration",
    "canonical_matched_v3_configuration_plan_bytes",
    "configuration_record",
    "derive_and_verify_external_configuration",
    "derive_and_verify_external_seed_transport",
    "matched_v3_configuration_plan_descriptor",
    "matched_v3_configuration_plan_sha256",
    "parse_matched_v3_configuration_plan_artifact",
    "verify_adapter_core_artifacts",
    "verify_adapter_reward_bundle_descriptor",
    "verify_adapter_reward_publication_descriptor",
    "verify_adapter_runner_artifact",
    "verify_external_configuration_set",
    "verify_external_materialization_identity_artifact",
    "verify_shared_environment_bridge_artifact",
]
