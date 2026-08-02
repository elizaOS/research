"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_open_protocol`.

The builder under test freezes the matched-current open-tuning design — task,
horizon, seed sets, candidate panel, selection rule, analysis plan — while
requiring callers to supply externally qualified, content-addressed
source/configuration/runtime bindings.  The tests check both directions: the
frozen design constants are exact (candidate ids, seeds, implementation
digests), and incomplete, placeholder, or tampered qualification inputs fail
closed.

This module doubles as the shared open-protocol fixture library: five sibling
suites (campaign, executor, evaluation-campaign, sealed-evaluation-campaign,
final-analysis) import it as ``protocol_fixtures``/``open_protocol_fixtures``
and treat :func:`_build` as the canonical parsed open-stage protocol.  Changes
to the ``_runtime``/``_source``/``_configuration``/``_qualifications`` helpers
ripple into those suites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

import pytest

from alberta_framework.benchmarks import forager_matched_open_protocol as builder
from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
from alberta_framework.benchmarks.forager_matched_candidate_universe import (
    MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256,
)
from alberta_framework.benchmarks.forager_matched_evidence import (
    MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256,
    MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256,
)
from alberta_framework.benchmarks.forager_matched_protocol import (
    AllowedTransform,
    ConfigurationBinding,
    ForagerMatchedProtocol,
    ResourceAccounting,
    SourceBinding,
    candidate_capability_descriptor_sha256,
    parse_forager_matched_protocol,
)
from alberta_framework.benchmarks.forager_matched_statistics import (
    PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
)

# candidate_id -> (upstream config path, sha256 of the pinned original bytes,
# sha256 after the allowed byte-preserving literal replacements, transform items).
_UPSTREAM_CONFIGS: Final = {
    "external_dqn_ln": (
        "fov_baseline_screening_v1/configs/DQN_LN-common-control.json",
        "0f25fde6f7d8818d833a529b35f80ebc14f90087ea3d2560c43ff2b417ec09d4",
        "4b4a9c5d355da716500c300fbd0940190f15ca6da0885f5f030a7b76ba02c928",
        (("total_steps", 499_712),),
    ),
    "external_dqn_crelu": (
        "fov_baseline_screening_v1/configs/DQN_CReLU-common-control.json",
        "0f1f76f7097bc00309dc7bba08b0951836a3220c7c1326c61c31e066961af15f",
        "4a969466bd5e7d35b937cbc7fc10354f87dc1e39b42b5ee770cf67a73bb47450",
        (("total_steps", 499_712),),
    ),
    "external_dqn_plain": (
        (
            "experiments/E138-two-biome-large/foragax/"
            "ForagaxTwoBiomeLarge-v1/9/DQN.json"
        ),
        "ee01cb9616d4bf06a4d8f6927a79a510aeeba5f6ca1613c4d4d3eacccdd0ec25",
        "cbcaa3949b3c4e898bc615dc157272bd63603cdf25b579aff3a0dddc2d61c7bc",
        (
            ("total_steps", 499_712),
            ("metaParameters.experiment.ntk_freq", 0),
            ("metaParameters.experiment.x_ref_steps", 0),
        ),
    ),
    "external_dqn_redo": (
        "fov_stateful_baseline_screening_v1/configs/DQN_ReDo_PostLNScore.json",
        "bfa3a27de72c3a02eb2cfe96e71ea442d6d86650d5e48fc80a992eacc5634f7d",
        "b7928ca79bcf689f31a777853438538c2719ecd1ff9f2a9be3689c44df7766ca",
        (("total_steps", 499_712),),
    ),
    "external_drqn_paper": (
        "fov_stateful_baseline_screening_v1/configs/DRQN-paper-v1.json",
        "428cad1dfeb3083fa8e0133fef3b655ab8b8d68cbc3c3852d28fb5cb9750412f",
        "7bcfb9f24d460dcde45a8f71307fdd12507abafa73a210794a83d8a352fcb7d9",
        (("total_steps", 499_712),),
    ),
    "isolated_ppo": (
        "fov_stateful_baseline_screening_v1/configs/PPO_2048_relu.json",
        "71f3be260eb47fce74875720adf75790ff8a8a84734ad3428e60a80edba1c29c",
        "61e3b3f08436c8356f52cd7f34c09493958ceea6c46cceed509cf514b9c98885",
        (("metaParameters.num_updates", 244), ("total_steps", 499_712)),
    ),
    "isolated_rtu": (
        "fov_stateful_baseline_screening_v1/configs/PPO-RTU_LN_128_1_relu.json",
        "a81b81d75cf5cff197fac14e61924707d385d46436e63856f56e083fca59a30e",
        "a8ea338e4fdc14c579a1ae3be00b4acee1c3c0e8042c7fec1d335a52c95844a3",
        (("metaParameters.num_updates", 3_904), ("total_steps", 499_712)),
    ),
    "exact_ppo": (
        "fov_stateful_baseline_screening_v1/configs/PPO_2048_relu.json",
        "71f3be260eb47fce74875720adf75790ff8a8a84734ad3428e60a80edba1c29c",
        "61e3b3f08436c8356f52cd7f34c09493958ceea6c46cceed509cf514b9c98885",
        (("metaParameters.num_updates", 244), ("total_steps", 499_712)),
    ),
    "search_oracle": (
        (
            "experiments/E138-two-biome-large/foragax/"
            "ForagaxTwoBiomeLarge-v1/Baselines/Search-Oracle.json"
        ),
        "86bd5822c3ec03db2a16b4001bccb903df72a27c19078fe13a46f475e851caf1",
        "91cf177ae8aba4ccc70cd5f28e379bea71da444bdd0a84e49bcd8209a93960e2",
        (("total_steps", 499_712),),
    ),
}

_PINNED_CONFIG_ROOT: Final = Path(__file__).resolve().parents[1] / "outputs" / "forager"
_SEARCH_ORACLE_FIXTURE: Final = (
    Path(__file__).resolve().parent / "fixtures" / "forager_matched" / "Search-Oracle.json"
)
_PLAIN_DQN_FIXTURE: Final = (
    Path(__file__).resolve().parent / "fixtures" / "forager_matched" / "DQN.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_rng_isolated_source_family_is_named_by_its_actual_relationship() -> None:
    assert builder.MATCHED_CURRENT_RNG_ISOLATED_SOURCE_CANDIDATE_IDS == (
        "isolated_ppo",
        "isolated_rtu",
    )
    assert not hasattr(builder, "MATCHED_CURRENT_RECURRENT_CANDIDATE_IDS")


@pytest.mark.parametrize(
    ("candidate_id", "replacements"),
    (
        ("external_dqn_ln", ((b'"total_steps": 100000', b'"total_steps": 499712'),)),
        (
            "external_dqn_crelu",
            ((b'"total_steps": 100000', b'"total_steps": 499712'),),
        ),
        (
            "external_dqn_plain",
            (
                (b'"total_steps": 10000', b'"total_steps": 499712'),
                (b'"ntk_freq": 2500', b'"ntk_freq": 0'),
                (b'"x_ref_steps": 100', b'"x_ref_steps": 0'),
            ),
        ),
        ("external_dqn_redo", ((b'"total_steps": 102400', b'"total_steps": 499712'),)),
        ("external_drqn_paper", ((b'"total_steps": 102400', b'"total_steps": 499712'),)),
        (
            "isolated_ppo",
            (
                (b'"num_updates": 50', b'"num_updates": 244'),
                (b'"total_steps": 102400', b'"total_steps": 499712'),
            ),
        ),
        (
            "isolated_rtu",
            (
                (b'"num_updates": 800', b'"num_updates": 3904'),
                (b'"total_steps": 102400', b'"total_steps": 499712'),
            ),
        ),
        ("search_oracle", ((b'"total_steps": 500000', b'"total_steps": 499712'),)),
    ),
)
def test_frozen_config_hashes_are_exact_byte_preserving_derivations(
    candidate_id: str,
    replacements: tuple[tuple[bytes, bytes], ...],
) -> None:
    relative_path, original_sha256, derived_sha256, _ = _UPSTREAM_CONFIGS[candidate_id]
    source_path = {
        "search_oracle": _SEARCH_ORACLE_FIXTURE,
        "external_dqn_plain": _PLAIN_DQN_FIXTURE,
    }.get(candidate_id, _PINNED_CONFIG_ROOT / relative_path)
    raw = source_path.read_bytes()
    if candidate_id == "search_oracle":
        # The exact upstream file has no terminal newline; text fixtures do.
        assert raw.endswith(b"\n")
        raw = raw[:-1]
    assert hashlib.sha256(raw).hexdigest() == original_sha256
    for old, new in replacements:
        assert raw.count(old) == 1
        raw = raw.replace(old, new)
    assert hashlib.sha256(raw).hexdigest() == derived_sha256


def _runtime() -> builder.MatchedCurrentRuntimeQualification:
    return builder.MatchedCurrentRuntimeQualification(
        image_sha256=builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256,
        runtime_profile_sha256=builder.MATCHED_CURRENT_RUNTIME_PROFILE_SHA256,
        executor_qualification_receipt_sha256=(
            builder.MATCHED_CURRENT_EXECUTOR_QUALIFICATION_RECEIPT_SHA256
        ),
        qualification_trust_anchor_identity="test_external_qualification_anchor_v1",
    )


def _transforms(items: tuple[tuple[str, int], ...]) -> tuple[AllowedTransform, ...]:
    return tuple(
        AllowedTransform(
            transform_type="byte_preserving_unique_literal_replacement",
            target=target,
            value_type="integer",
            value=value,
        )
        for target, value in items
    )


def _configuration(candidate_id: str) -> ConfigurationBinding:
    alberta_fingerprints = builder.matched_current_alberta_configuration_fingerprints()
    if candidate_id in alberta_fingerprints:
        digest = alberta_fingerprints[candidate_id]
        return ConfigurationBinding(
            original_path=f"matched_current/alberta_configs/{candidate_id}.json",
            original_sha256=digest,
            derived_sha256=digest,
            allowed_transforms=(),
        )
    path, original, derived, transform_items = _UPSTREAM_CONFIGS[candidate_id]
    return ConfigurationBinding(
        original_path=path,
        original_sha256=original,
        derived_sha256=derived,
        allowed_transforms=_transforms(transform_items),
    )


def _source(candidate_id: str) -> SourceBinding:
    kind: Literal["git_tree", "reviewed_snapshot"]
    if candidate_id in builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS:
        repository = builder.MATCHED_CURRENT_ALBERTA_REPOSITORY
        commit = builder.MATCHED_CURRENT_ALBERTA_BASE_COMMIT
        kind = "reviewed_snapshot"
        source_id = "alberta-matched-source"
    else:
        repository = builder.MATCHED_CURRENT_UPSTREAM_REPOSITORY
        commit = builder.MATCHED_CURRENT_UPSTREAM_BASE_COMMIT
        kind = (
            "reviewed_snapshot"
            if candidate_id in {"isolated_ppo", "isolated_rtu"}
            else "git_tree"
        )
        source_id = "upstream-isolated-source" if kind == "reviewed_snapshot" else "upstream-tree"
    return SourceBinding(
        provenance_kind=kind,
        repository=repository,
        base_commit=commit,
        tree_git_sha1=(
            builder.MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1 if kind == "git_tree" else None
        ),
        archive_sha256=(
            _sha(f"archive:{source_id}")
            if kind == "reviewed_snapshot"
            else builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256
        ),
        inventory_sha256=(
            _sha(f"inventory:{source_id}")
            if kind == "reviewed_snapshot"
            else builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256
        ),
        snapshot_descriptor_sha256=(
            _sha(f"snapshot:{source_id}") if kind == "reviewed_snapshot" else None
        ),
    )


def _qualifications() -> dict[str, builder.MatchedCurrentCandidateQualification]:
    def resources(candidate_id: str, index: int) -> ResourceAccounting:
        if candidate_id in builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS:
            return ResourceAccounting(0, 0, 0, 1_000 + index)
        if candidate_id in (
            builder.MATCHED_CURRENT_HORDE_CANDIDATE_IDS
            + builder.MATCHED_CURRENT_LOCAL_RTU_CANDIDATE_IDS
        ):
            recurrent = (
                64
                if candidate_id
                in {"alberta_horde_recurrent64", "alberta_rtu_h08_taylor"}
                else 0
            )
            return ResourceAccounting(1_000 + index, 999_424, 0, recurrent)
        if candidate_id == "search_oracle":
            return ResourceAccounting(0, 0, 0, 0)
        replay = {
            "external_dqn_ln": 10_000,
            "external_dqn_crelu": 10_000,
            "external_dqn_plain": 10_000,
            "external_dqn_redo": 1_000,
            "external_drqn_paper": 1_000,
            "isolated_ppo": 0,
            "isolated_rtu": 0,
            "exact_ppo": 0,
        }[candidate_id]
        recurrent = 64 if candidate_id in {"external_drqn_paper", "isolated_rtu"} else 0
        return ResourceAccounting(1_000 + index, 124_928, replay, recurrent)

    return {
        candidate_id: builder.MatchedCurrentCandidateQualification(
            source=_source(candidate_id),
            configuration=_configuration(candidate_id),
            effective_seed_proof_sha256=_sha(f"seed-proof:{candidate_id}"),
            capability_qualification_receipt_sha256=_sha(f"capability:{candidate_id}"),
            resources=resources(candidate_id, index),
        )
        for index, candidate_id in enumerate(builder.MATCHED_CURRENT_CANDIDATE_IDS)
    }


def _build() -> ForagerMatchedProtocol:
    """Return the canonical parsed open-stage protocol fixture.

    Built from synthetic-but-well-formed qualification inputs; stage is
    ``open_tuning`` with the exact frozen design (horizon 499_712, tuning
    seeds 2_300_001..2_300_010, evaluation seeds 2_200_001..2_200_030).
    Sibling suites import this as their starting protocol.
    """
    return builder.build_forager_matched_open_protocol(
        runtime=_runtime(),
        candidate_qualifications=_qualifications(),
    )


def test_builder_freezes_exact_matched_current_design() -> None:
    protocol = _build()

    assert protocol.stage == "open_tuning"
    assert protocol.horizon == 499_712
    assert protocol.tuning_seeds == tuple(range(2_300_001, 2_300_011))
    assert protocol.evaluation_seeds == tuple(range(2_200_001, 2_200_031))
    assert protocol.active_seeds == protocol.tuning_seeds
    assert tuple(candidate.candidate_id for candidate in protocol.candidates) == (
        builder.MATCHED_CURRENT_CANDIDATE_IDS
    )

    groups = protocol.selection_plan.groups
    assert protocol.selection_plan.candidate_universe_sha256 == (
        MATCHED_CURRENT_CANDIDATE_UNIVERSE_SHA256
    )
    assert tuple(group.selection_group for group in groups) == ("alberta", "external")
    assert groups[0].candidate_ids == builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS
    assert groups[0].advance_count == 1
    assert groups[1].candidate_ids == builder.MATCHED_CURRENT_EXTERNAL_CANDIDATE_IDS
    assert groups[1].advance_count == 3
    assert protocol.evaluation_panel.selection_slots == protocol.selection_plan.slots
    assert tuple(slot.rank for slot in protocol.selection_plan.slots) == (
        1,
        1,
        2,
        3,
    )

    assert protocol.primary_hypothesis.intervention_slot.selection_group == "alberta"
    assert protocol.primary_hypothesis.comparator_slot.selection_group == "external"
    assert protocol.primary_hypothesis.comparator_slot.rank == 1
    assert tuple(item.comparator_slot.rank for item in protocol.secondary_hypotheses) == (2, 3)
    assert protocol.multiplicity_policy.hypothesis_ids == (
        "alberta_vs_external_rank2",
        "alberta_vs_external_rank3",
    )


def test_causal_panel_is_full_unique_factorial_and_binds_default_center() -> None:
    configurations = builder.matched_current_causal_configurations()
    fingerprints = builder.matched_current_causal_configuration_fingerprints()
    alberta_fingerprints = builder.matched_current_alberta_configuration_fingerprints()

    assert tuple(configurations) == builder.MATCHED_CURRENT_CAUSAL_CANDIDATE_IDS
    assert len(configurations) == len(fingerprints) == len(set(fingerprints.values())) == 9
    assert {
        config["exploration_probability"] for config in configurations.values()
    } == {0.025, 0.05, 0.1}
    assert {config["respawn_safety_quantile"] for config in configurations.values()} == {
        0.5,
        0.75,
        0.9,
    }
    assert fingerprints["causal_e050_q075"] == CausalMapForagerConfig().fingerprint()

    protocol = _build()
    for candidate_id in fingerprints:
        binding = protocol.candidate_index[candidate_id].configuration
        assert binding.original_sha256 == alberta_fingerprints[candidate_id]
        assert binding.derived_sha256 == alberta_fingerprints[candidate_id]
        assert binding.allowed_transforms == ()


def test_alberta_panel_binds_three_local_families_and_exact_worker_envelopes() -> None:
    configurations = builder.matched_current_alberta_configurations()
    fingerprints = builder.matched_current_alberta_configuration_fingerprints()

    assert tuple(configurations) == builder.MATCHED_CURRENT_ALBERTA_CANDIDATE_IDS
    assert len(configurations) == len(fingerprints) == 14
    assert {
        configuration["implementation_kind"] for configuration in configurations.values()
    } == {
        "alberta_causal_map",
        "alberta_horde_actor_critic",
        "alberta_rtu_rtrl",
    }
    assert fingerprints["alberta_horde_default"] == (
        "7e7e681ca3a06e6f5c9bcdf0c4de42a4775439967ac41504c3b9ebd971d0db7a"
    )
    assert fingerprints["alberta_horde_step3e3"] == (
        "feb2cd34628b3d87873163e1c78d8ea0b5aba4e4652dcba67138bd3f6eba6bc5"
    )
    assert fingerprints["alberta_rtu_h08_taylor"] == (
        "07571eeec0e132027c819cc3a0c8d781a0df71ecbd840947d3641e2ea3831792"
    )


def test_metric_rng_and_analysis_implementation_identities_are_frozen() -> None:
    protocol = _build()
    descriptor = builder.matched_current_metric_implementation_descriptor()

    assert descriptor["horizon"] == 499_712
    assert descriptor["sample_count"] == 4_998
    assert descriptor["tail_start_index"] == 4_498
    assert descriptor["tail_sample_count"] == 500
    assert protocol.task.task_identity_sha256 == builder.MATCHED_CURRENT_TASK_IDENTITY_SHA256
    assert protocol.task.environment_rng_schedule_sha256 == (
        builder.MATCHED_CURRENT_ENVIRONMENT_RNG_SCHEDULE_SHA256
    )
    assert builder.MATCHED_CURRENT_ENVIRONMENT_RNG_PARITY_CONTRACT_SHA256 == (
        "0f7b0d52e55523ce81b35ccf85446b32c1baddbc76fee76e7d25489a7274aa27"
    )
    assert protocol.selection_plan.metric_implementation_sha256 == (
        "ea4648d8733af3ab5a05c05543eddcc9a1d0415c6cba3935ed4b3c6d9e2506e4"
    )
    assert protocol.selection_plan.statistic_implementation_sha256 == (
        MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
    )
    assert protocol.selection_plan.bootstrap_rng_implementation_sha256 == (
        MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
    )
    assert protocol.analysis_plan.primary.implementation_sha256 == (
        PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
    )
    assert protocol.analysis_plan.secondary.implementation_sha256 == (
        SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
    )


def test_isolated_policy_gradient_and_descriptive_boundaries_are_exact() -> None:
    protocol = _build()
    isolated_ppo = protocol.candidate_index["isolated_ppo"]
    isolated_rtu = protocol.candidate_index["isolated_rtu"]
    exact_ppo = protocol.candidate_index["exact_ppo"]
    oracle = protocol.candidate_index["search_oracle"]

    assert isolated_ppo.selection_group == isolated_rtu.selection_group == "external"
    assert isolated_ppo.execution_semantics.rollout_steps == 2_048
    assert isolated_ppo.execution_semantics.num_rollouts == 244
    assert isolated_rtu.execution_semantics.rollout_steps == 128
    assert isolated_rtu.execution_semantics.num_rollouts == 3_904
    assert isolated_ppo.entrypoint_family == isolated_rtu.entrypoint_family == (
        "rtu_ppo_rng_isolation_adapter"
    )
    assert isolated_ppo.agent_rng.identity == isolated_rtu.agent_rng.identity == (
        "isolated_agent_rng_v1"
    )
    assert isolated_ppo.pairing.eligible is isolated_rtu.pairing.eligible is True

    assert exact_ppo.implementation_kind == "upstream_ppo"
    assert exact_ppo.agent_rng.environment_key_shared is True
    assert exact_ppo.pairing.exclusion_reasons == ("shared_agent_environment_rng",)
    assert oracle.observation_access.access_mode == "privileged_global_objects"
    assert oracle.observation_access.observation_type == "object"
    assert oracle.observation_access.aperture_size == -1
    assert oracle.observation_access.privileged_fields == (
        "global_objects",
        "known_reward_priority",
    )
    assert oracle.pairing.exclusion_reasons == ("privileged_observation_access",)
    assert protocol.evaluation_panel.fixed_descriptive_candidate_ids == (
        "exact_ppo",
        "search_oracle",
    )


def test_generation_is_canonical_and_independent_of_input_mapping_order() -> None:
    qualifications = _qualifications()
    first = builder.build_forager_matched_open_protocol_bytes(
        runtime=_runtime(), candidate_qualifications=qualifications
    )
    second = builder.build_forager_matched_open_protocol_bytes(
        runtime=_runtime(),
        candidate_qualifications=dict(reversed(tuple(qualifications.items()))),
    )

    assert first == second
    assert parse_forager_matched_protocol(first).canonical_bytes == first
    assert b"timestamp" not in first
    assert b"seed_scores" not in first


def test_metric_descriptor_import_time_digest_self_check() -> None:
    descriptor = builder.matched_current_metric_implementation_descriptor()
    raw = json.dumps(
        descriptor,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == (
        builder.MATCHED_CURRENT_METRIC_SEMANTIC_CONTRACT_SHA256
    )


def test_builder_computes_every_capability_subject_from_complete_semantics() -> None:
    protocol = _build()

    descriptors = []
    for candidate in protocol.candidates:
        expected = candidate_capability_descriptor_sha256(candidate)
        descriptors.append(expected)
        assert candidate.runtime_binding.qualified_capability_descriptor_sha256 == expected
        assert candidate.runtime_binding.image_sha256 == (
            builder.MATCHED_CURRENT_REQUIRED_IMAGE_SHA256
        )
        assert candidate.runtime_binding.runtime_profile_sha256 == (
            protocol.runtime.runtime_profile_sha256
        )
    assert len(descriptors) == len(set(descriptors))


def test_builder_requires_the_complete_exact_qualification_panel() -> None:
    qualifications = _qualifications()
    del qualifications["isolated_ppo"]
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="isolated_ppo"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    qualifications["posthoc_candidate"] = qualifications["external_dqn_ln"]
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="posthoc_candidate"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )


def test_placeholder_or_mismatched_trust_bindings_fail_closed() -> None:
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="placeholder"):
        builder.build_forager_matched_open_protocol(
            runtime=replace(_runtime(), runtime_profile_sha256="0" * 64),
            candidate_qualifications=_qualifications(),
        )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="frozen.*image"):
        builder.build_forager_matched_open_protocol(
            runtime=replace(_runtime(), image_sha256=_sha("different-image")),
            candidate_qualifications=_qualifications(),
        )

    qualifications = _qualifications()
    target = qualifications["isolated_rtu"]
    qualifications["isolated_rtu"] = replace(
        target,
        capability_qualification_receipt_sha256="f" * 64,
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="placeholder"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )


def test_source_and_configuration_derivations_fail_closed() -> None:
    qualifications = _qualifications()
    target = qualifications["isolated_ppo"]
    qualifications["isolated_ppo"] = replace(
        target,
        source=replace(
            target.source,
            provenance_kind="git_tree",
            tree_git_sha1=builder.MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1,
            snapshot_descriptor_sha256=None,
        ),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="source identity"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    target = qualifications["isolated_ppo"]
    bad_transforms = (
        AllowedTransform(
            transform_type="byte_preserving_unique_literal_replacement",
            target="metaParameters.num_updates",
            value_type="integer",
            value=245,
        ),
        target.configuration.allowed_transforms[1],
    )
    qualifications["isolated_ppo"] = replace(
        target,
        configuration=replace(target.configuration, allowed_transforms=bad_transforms),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="frozen transform"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    target = qualifications["causal_e050_q075"]
    qualifications["causal_e050_q075"] = replace(
        target,
        configuration=replace(target.configuration, derived_sha256=_sha("retuned-after-open")),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="Alberta worker"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )


def test_source_cohorts_and_upstream_archive_identity_are_exact() -> None:
    protocol = _build()
    upstream_ids = (
        "external_dqn_ln",
        "external_dqn_crelu",
        "external_dqn_plain",
        "external_dqn_redo",
        "external_drqn_paper",
        "exact_ppo",
        "search_oracle",
    )
    for candidate_id in upstream_ids:
        source = protocol.candidate_index[candidate_id].source
        assert source.tree_git_sha1 == builder.MATCHED_CURRENT_UPSTREAM_TREE_GIT_SHA1
        assert source.archive_sha256 == builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_SHA256
        assert source.inventory_sha256 == (
            builder.MATCHED_CURRENT_UPSTREAM_ARCHIVE_INVENTORY_SHA256
        )
        assert source.snapshot_descriptor_sha256 is None

    qualifications = _qualifications()
    target = qualifications["causal_e100_q090"]
    qualifications["causal_e100_q090"] = replace(
        target,
        source=replace(target.source, archive_sha256=_sha("other-causal-snapshot")),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="source family"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    target = qualifications["isolated_rtu"]
    qualifications["isolated_rtu"] = replace(
        target,
        source=replace(target.source, inventory_sha256=_sha("other-isolation-patch")),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="source family"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )


def test_resource_accounting_semantics_fail_closed() -> None:
    qualifications = _qualifications()
    target = qualifications["external_dqn_ln"]
    qualifications["external_dqn_ln"] = replace(
        target,
        resources=replace(target.resources, replay_capacity_transitions=9_999),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="replay capacity"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    target = qualifications["external_drqn_paper"]
    qualifications["external_drqn_paper"] = replace(
        target,
        resources=replace(target.resources, recurrent_state_elements=0),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="recurrent state"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )

    qualifications = _qualifications()
    target = qualifications["search_oracle"]
    qualifications["search_oracle"] = replace(
        target,
        resources=replace(target.resources, parameter_count=1),
    )
    with pytest.raises(builder.ForagerMatchedOpenProtocolBuildError, match="zero learned"):
        builder.build_forager_matched_open_protocol(
            runtime=_runtime(), candidate_qualifications=qualifications
        )
