"""Tests for the non-authorizing future Forager memory-source registry."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_memory_comparator_development_sources as sources

pytestmark = pytest.mark.unit


_EXPECTED_DESCRIPTOR_SHA256 = "a98f78d5e5483c8dfbd821b953793e61ae820c1f1b3906a18b886836da7e116c"

_FROZEN_MATCHED_V3_CANDIDATE_IDS = (
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
    "external_dqn_plain",
    "external_dqn_crelu",
    "external_dqn_redo",
    "external_dqn_reward_trace",
    "external_dqn_l2_init",
    "external_pt_dqn_xfinal",
    "external_drqn_xfinal",
    "isolated_ppo_generic",
    "isolated_rtu_paper_scale",
    "adapted_full_rainbow",
    "adapted_ppo_gru",
    "random_policy",
    "search_nearest",
    "search_oracle",
)

_EXPECTED_FAMILY_PINS = {
    "pobax_ld_gtrxl": {
        "url": "https://github.com/taodav/pobax",
        "commit_git_sha1": "a5e1d62d14e4efe783885b9d4f19cffa2a568eec",
        "tree_git_sha1": "d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6",
        "archive_sha256": "f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9",
        "archive_size_bytes": 1_699_840,
        "license_spdx": "Apache-2.0",
    },
    "agalite": {
        "url": "https://github.com/subho406/agalite",
        "commit_git_sha1": "101acbecc121a258ad8f7e58e2f782f546674979",
        "tree_git_sha1": "c76616f5ac4fba0bfd700095f1c174f11144471e",
        "archive_sha256": "2784a2491b0844cf902f3f6e9896b18730a01ca7ea72ebea490ace780f914ecb",
        "archive_size_bytes": 56_142,
        "license_spdx": "Apache-2.0",
    },
    "memory_traces": {
        "url": "https://github.com/onnoeberhard/memory-traces",
        "commit_git_sha1": "fcfdacc0b0a06dc181b49b9ef95893dbae7f2bcd",
        "tree_git_sha1": "af6f2cdfd2dcabd079a030cc1e2357f09886fd27",
        "archive_sha256": "55701c411d293f63d6570563b53ec6b0bc84ae380ecb95eea42ea41928c1a4f9",
        "archive_size_bytes": 13_733,
        "license_spdx": "MIT",
    },
    "shm": {
        "url": "https://github.com/thaihungle/SHM",
        "commit_git_sha1": "651f9e27e0fd3a3ec46a0f45b84e0128c5f8a312",
        "tree_git_sha1": "22fc6aaa216e3aa8032b31d56e51c31c6ea9c1b4",
        "archive_sha256": "4c12d7b5a5ca1356b99d31721e1cca3eb5137d4ae37afc34ad9d28b5d1e62193",
        "archive_size_bytes": 537_371,
        "license_spdx": "Apache-2.0",
    },
    "ffm": {
        "url": "https://github.com/proroklab/ffm",
        "commit_git_sha1": "b3f94d2a0f35ba05089faf19ab1df846057cf8b6",
        "tree_git_sha1": "7684b03c81dc9fc16f7ac973c8c6425dd279e6f4",
        "archive_sha256": "b9497c94255a4d0e2d32666fea52595439b72cbc1d291ed50b528b7d62c4c69d",
        "archive_size_bytes": 286_966,
        "license_spdx": "MIT",
    },
}

_EXPECTED_SOURCE_SUBSETS = {
    "pobax_ld_gtrxl": {
        "LICENSE": ("c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11_357),
        "pobax/algos/ppo.py": (
            "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153",
            19_864,
        ),
        "pobax/algos/transformer_xl.py": (
            "e51c3c9530963e902bbab4f23683c6e4c9a9f0a6399ada624cab5da6c7e462bf",
            22_308,
        ),
        "pobax/config.py": (
            "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c",
            7_047,
        ),
        "pobax/models/actor_critic.py": (
            "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f",
            2_374,
        ),
        "pobax/models/discrete.py": (
            "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c",
            11_026,
        ),
        "pobax/models/rel_multi_head.py": (
            "354e8ad9a0e7efdc8eb04ee1104c694173bf7594d4a76631fe384f144ad2c333",
            23_241,
        ),
        "pobax/models/transformerXL.py": (
            "bfc1b5d734be61e8ca3ac4bfb0da0d992a6fd131982a3b42c7870102c5e762cd",
            6_562,
        ),
    },
    "agalite": {
        "LICENSE": ("c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", 11_357),
        "config_pure/craftax/arelit.yaml": (
            "949f146735e0ef3090a184f125b1823f4a7af1ba6a4ee28f2bb01074a02ced79",
            533,
        ),
        "src/models/agalite/agalite.py": (
            "00d921f46740e43aed9e444c51852b0e7fdb80cd489550e1c815d2c70e00a89b",
            12_103,
        ),
        "src/models/agalite/kernels.py": (
            "27fc4e6f1747558b13c0e9e29996e7b22004e44c58404124750deaef4769bb8d",
            1_337,
        ),
        "src/models/agalite/layers.py": (
            "610a703e3da2736fef14a2d72545a04143fcac4a9823ce707c7301fecd2e8978",
            5_915,
        ),
        "src_pure/models/agalite.py": (
            "2bce6fe8dc417dfffbc9644011ea3efe9a172da2ab7bbdcd8ac357929ea00fc6",
            22_854,
        ),
        "src_pure/purejaxrl/ppo_rnn.py": (
            "85e5ad42ea93311fc81c3a5ef8ce4559f709f46941e08155894f32b95fc1b6c9",
            12_016,
        ),
    },
    "memory_traces": {
        "LICENSE": ("6c9d35f885f47922acb8c77681b1dbd4b16f186bcb0ba5de3948e0efbad0a5f9", 1_070),
        "examples/ppo_tmaze.py": (
            "841bdd3d62ce4143f149b5a5fe1c18ec37c1a96def1594ba2faa04d466bae88f",
            3_157,
        ),
        "pyproject.toml": ("ff0d1c3c3917520e7272d4425a3ce7f43167f5654accf8f792fd00ff2882fb38", 412),
        "traces/main.py": (
            "180697f158e173dc2d51ff11013090fd711c3f41e213972527e2f2a7a3ddbe3b",
            22_110,
        ),
        "traces/ppo.py": (
            "e01aa53aa5e72e6890c2942ae77892786afe3d1f9256d5c378da1076fdfee541",
            15_180,
        ),
    },
    "shm": {
        "LICENSE": ("26e45f86ea13d4ae20136b9c1d693149acd385dbec88a439b317fe8f2da0d55a", 11_337),
        "README.md": ("eaaeb9c20a30f36798fb347034be11a62416079f700ebcb079184f01c18ef3ae", 10_814),
        "pomdp-baselines/torchkit/shm.py": (
            "7df5a127d286434a52a8294f68b6e86ac297d010c06c6f774d4404b7b617965b",
            2_519,
        ),
        "popgym/baselines/ray_models/ray_shm.py": (
            "d45e97cd9fc606372c44c885892614c79d7f422c550277a0b7d0935807f475e4",
            3_471,
        ),
        "shm.py": ("7ba92a52e7ec4d75f2b8aab09ea463324b3b550dff597747df1e524aa75c0146", 2_538),
        "train_popgym.py": (
            "e0acff8bff5aa42cab7c096f264bebae2b1c1bb1bf4212b4c733159064b87aa3",
            7_288,
        ),
    },
    "ffm": {
        "LICENSE": ("db7b51734c0b098407c121530d810026743145f11333bcae9b58b442197f9695", 1_056),
        "README.md": (
            "1e65c7d16b1f8e773aec6f8b71161435a0e09d963d72edb856b7ec216e7dfbed",
            4_291,
        ),
        "aggregations.py": (
            "ce7f28de73acf6e663dd779b0b46b549307b5a1a9dd12a365adfb9911b63e099",
            19_380,
        ),
        "models/ffm_outer.py": (
            "65a2356a16fa4aa188d4aa78fd4d464cc3a6f84c7500b72f09ff9f31a8762d0a",
            2_947,
        ),
        "models/ray_ffm.py": (
            "f80e107310f27044ae050a29e89569bb424ed8e25f08d98039ef55f2919b0875",
            8_166,
        ),
        "ppo.py": (
            "718c3eae1406b8f9485dbebed307c9f01210c501db7aaac1106f134a76dbced0",
            7_850,
        ),
        "standalone_jax/ffm/__init__.py": (
            "39da83dd9994b5f4745a59a0bf79cf26abbd209a670dae5ce9100048dec2076a",
            70,
        ),
        "standalone_jax/ffm/ffa.py": (
            "dffcc578c91f3baa8ca37d3fbc05b73031e3a2a06dd6d9be5c8d4cbe899945d2",
            2_672,
        ),
        "standalone_jax/ffm/ffm.py": (
            "00832954cc87ce4b7a25a2b23104e649bc77f8f538f2751f3f90a9a9b368275a",
            2_817,
        ),
        "standalone_jax/setup.py": (
            "71b89ef1823a3c25a390aecd26452e8ab1f72f625e88f3e7dce8836c9992c401",
            206,
        ),
    },
}


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


def _descriptor_family(descriptor: dict[str, Any], family_id: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        next(
            family for family in descriptor["source_families"] if family["family_id"] == family_id
        ),
    )


def _inventory(family_id: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(sources.expected_source_subset_inventory_bytes(family_id)),
    )


def test_descriptor_is_canonical_and_frozen() -> None:
    raw = sources.memory_comparator_development_sources_descriptor_bytes()
    assert raw == _canonical(json.loads(raw))
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_DESCRIPTOR_SHA256
    assert sources.MEMORY_COMPARATOR_DEVELOPMENT_SOURCES_DESCRIPTOR_SHA256 == (
        _EXPECTED_DESCRIPTOR_SHA256
    )
    parsed = sources.parse_memory_comparator_development_sources_descriptor(raw)
    assert parsed == sources.memory_comparator_development_sources_descriptor()
    assert parsed["schema_version"] == (
        "alberta.forager_memory_comparator_development_sources.v2"
    )
    assert parsed["limitations"][3] == (
        "The canonical inventory verifier checks only declared records; the optional "
        "read-only file verifier hashes exactly those declared bytes, but neither "
        "authenticates the archive or full tree."
    )
    assert parsed["limitations"][4] == (
        "Audit-time archive byte pins require a newly reproduced deterministic "
        "materialization receipt before any future plan."
    )


def test_descriptor_parser_rejects_noncanonical_duplicate_nonfinite_and_oversize() -> None:
    raw = sources.memory_comparator_development_sources_descriptor_bytes()
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.parse_memory_comparator_development_sources_descriptor(b" " + raw)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.parse_memory_comparator_development_sources_descriptor(b'{"x":1,"x":2}\n')
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.parse_memory_comparator_development_sources_descriptor(b'{"x":NaN}\n')
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.parse_memory_comparator_development_sources_descriptor(b"{}" + b" " * 2_097_152)


def test_all_authority_qualification_execution_promotion_and_sota_claims_are_false() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    forbidden_true_fragments = (
        "authority",
        "authorized",
        "qualified",
        "qualification_granted",
        "execution_ready",
        "executed",
        "promotion_allowed",
        "sota_claim_allowed",
        "performance_claim_allowed",
        "production_plan_issued",
        "scientific_evidence_created",
        "runtime_qualified",
        "artifact_accepted",
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, bool) and any(
                    fragment in key for fragment in forbidden_true_fragments
                ):
                    assert child is False, key
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(descriptor)
    assert descriptor["claims"]
    assert set(descriptor["claims"].values()) == {False}


def test_frozen_matched_v3_universe_is_exact_explicit_and_disjoint() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    frozen = descriptor["frozen_matched_v3_universe"]
    assert sources.FROZEN_MATCHED_V3_CANDIDATE_IDS == _FROZEN_MATCHED_V3_CANDIDATE_IDS
    assert tuple(frozen["candidate_order"]) == _FROZEN_MATCHED_V3_CANDIDATE_IDS
    assert frozen["candidate_count"] == 28
    assert frozen["registry_extends_universe"] is False
    assert frozen["registry_authorizes_matched_v3_runs"] is False
    assert set(sources.DEVELOPMENT_MEMORY_CANDIDATE_IDS).isdisjoint(
        _FROZEN_MATCHED_V3_CANDIDATE_IDS
    )
    assert set(frozen["candidate_order"]).isdisjoint(descriptor["development_candidate_order"])


def test_source_family_archive_and_exact_subset_pins() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    assert tuple(sources.SOURCE_FAMILY_BY_ID) == tuple(_EXPECTED_FAMILY_PINS)
    assert [item["family_id"] for item in descriptor["source_families"]] == list(
        _EXPECTED_FAMILY_PINS
    )
    for family_id, expected_upstream in _EXPECTED_FAMILY_PINS.items():
        family = _descriptor_family(descriptor, family_id)
        assert family["upstream"] == expected_upstream
        assert family["archive_authenticated_here"] is False
        assert family["archive_identity_scope"] == (
            "audit_time_byte_receipt_not_reproduced_or_authenticated_here"
        )
        observed = {
            item["path"]: (item["sha256"], item["size_bytes"])
            for item in family["relevant_source_subset"]
        }
        assert observed == _EXPECTED_SOURCE_SUBSETS[family_id]
        assert list(observed) == sorted(observed, key=lambda value: value.encode("utf-8"))


def test_public_mappings_and_records_are_immutable_and_outputs_are_copies() -> None:
    with pytest.raises(TypeError):
        sources.SOURCE_FAMILY_BY_ID["new"] = sources.SOURCE_FAMILY_BY_ID["agalite"]  # type: ignore[index]
    with pytest.raises(TypeError):
        sources.DEVELOPMENT_CANDIDATE_BY_ID["new"] = sources.DEVELOPMENT_CANDIDATE_BY_ID[  # type: ignore[index]
            "adapted_ppo_agalite"
        ]
    with pytest.raises(FrozenInstanceError):
        sources.SOURCE_FAMILY_BY_ID["agalite"].archive_size_bytes = 1  # type: ignore[misc]

    first = sources.memory_comparator_development_sources_descriptor()
    first["claims"]["execution_authority_granted"] = True
    first["source_families"].clear()
    second = sources.memory_comparator_development_sources_descriptor()
    assert second["claims"]["execution_authority_granted"] is False
    assert len(second["source_families"]) == 5


@pytest.mark.parametrize("family_id", tuple(_EXPECTED_FAMILY_PINS))
def test_exact_source_subset_inventory_accepts_only_canonical_frozen_records(
    family_id: str,
) -> None:
    raw = sources.expected_source_subset_inventory_bytes(family_id)
    assert raw == _canonical(json.loads(raw))
    verified = sources.verify_source_subset_inventory(raw, expected_family_id=family_id)
    assert verified.family_id == family_id
    assert verified.inventory_sha256 == hashlib.sha256(raw).hexdigest()
    assert verified.source_file_count == len(_EXPECTED_SOURCE_SUBSETS[family_id])
    assert verified.source_total_size_bytes == sum(
        size for _digest, size in _EXPECTED_SOURCE_SUBSETS[family_id].values()
    )
    assert verified.source_bytes_verified is False


def test_source_subset_file_verifier_hashes_bytes_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "checkout"
    root.mkdir(parents=True)
    raw_by_path = {
        "LICENSE": b"alpha\n",
        "src/model.py": b"model-source\n",
    }
    pins = tuple(
        sources.SourceFilePin(
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )
        for path, raw in raw_by_path.items()
    )
    family = sources.SourceFamilyPin(
        family_id="synthetic",
        url="https://github.com/example/synthetic",
        commit_git_sha1="1" * 40,
        tree_git_sha1="2" * 40,
        archive_sha256="3" * 64,
        archive_size_bytes=1,
        license_spdx="MIT",
        source_files=pins,
        candidate_ids=("synthetic_candidate",),
    )

    def require_synthetic(value: object) -> sources.SourceFamilyPin:
        assert value == "synthetic"
        return family

    monkeypatch.setattr(sources, "_require_family_id", require_synthetic)
    for path, raw in raw_by_path.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)

    verified = sources.verify_source_subset_files("synthetic", root)
    assert verified.source_bytes_verified is True
    assert verified.source_file_count == 2
    assert verified.source_total_size_bytes == sum(map(len, raw_by_path.values()))

    before_descriptors = len(tuple(Path("/proc/self/fd").iterdir()))
    (root / "src" / "model.py").unlink()
    for _attempt in range(16):
        with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
            sources.verify_source_subset_files("synthetic", root)
    assert len(tuple(Path("/proc/self/fd").iterdir())) == before_descriptors
    (root / "src" / "model.py").write_bytes(raw_by_path["src/model.py"])

    source_directory = root / "src"
    real_source_directory = root / "src-real"
    source_directory.rename(real_source_directory)
    source_directory.symlink_to(real_source_directory, target_is_directory=True)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", root)
    source_directory.unlink()
    real_source_directory.rename(source_directory)

    real_license = root / "LICENSE.real"
    (root / "LICENSE").rename(real_license)
    (root / "LICENSE").symlink_to(real_license)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", root)
    (root / "LICENSE").unlink()
    real_license.rename(root / "LICENSE")

    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", alias_parent / "checkout")
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", Path("relative-checkout"))
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", cast(Any, str(root)))

    (root / "LICENSE").write_bytes(b"omega\n")
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_files("synthetic", root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_size", "wrong_hash", "order"])
def test_source_subset_inventory_fails_closed_on_membership_or_identity_tamper(
    mutation: str,
) -> None:
    inventory = _inventory("memory_traces")
    records = inventory["source_files"]
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(
            {
                "path": "unregistered.py",
                "sha256": "1" * 64,
                "size_bytes": 1,
            }
        )
    elif mutation == "wrong_size":
        records[0]["size_bytes"] += 1
    elif mutation == "wrong_hash":
        records[0]["sha256"] = "1" * 64
    else:
        records.reverse()
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(_canonical(inventory))


@pytest.mark.parametrize("alias", ["./LICENSE", "x/../LICENSE", "LICENSE/", "license"])
def test_source_subset_inventory_rejects_path_aliases(alias: str) -> None:
    inventory = _inventory("memory_traces")
    inventory["source_files"][0]["path"] = alias
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(_canonical(inventory))


def test_source_subset_inventory_rejects_duplicates_nonfinite_and_wrong_family() -> None:
    inventory = _inventory("memory_traces")
    inventory["source_files"].append(copy.deepcopy(inventory["source_files"][0]))
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(_canonical(inventory))

    raw = sources.expected_source_subset_inventory_bytes("memory_traces")
    duplicate_key = raw.replace(b'"family_id":', b'"family_id":"memory_traces","family_id":', 1)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(duplicate_key)
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(b'{"family_id":NaN}\n')
    with pytest.raises(sources.ForagerMemoryComparatorDevelopmentSourcesError):
        sources.verify_source_subset_inventory(raw, expected_family_id="agalite")


def test_candidate_concepts_are_source_derived_development_only_and_image_only() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    candidates = descriptor["development_candidate_concepts"]
    assert tuple(item["candidate_id"] for item in candidates) == (
        sources.DEVELOPMENT_MEMORY_CANDIDATE_IDS
    )
    for candidate in candidates:
        assert candidate["classification"] == "open_development_nonpromoting"
        assert candidate["observation_access"] == "forager_image_only"
        assert candidate["source_relationship"] == "source_derived_not_exact_execution"
        assert candidate["execution_authority_granted"] is False
        assert candidate["qualification_granted"] is False
        assert candidate["promotion_allowed"] is False
        assert candidate["sota_claim_allowed"] is False


def test_pobax_ld_and_gtrxl_require_the_recorded_semantics_and_reviews() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    pobax = _descriptor_family(descriptor, "pobax_ld_gtrxl")["semantic_contract"]
    assert pobax["adapted_ppo_gru_ld"] == {
        "actor_advantage_source": "critic_0",
        "clipped_value_loss_required": True,
        "critic_count": 2,
        "disagreement_term_required": True,
        "ld_weight_must_be_strictly_positive": True,
        "zero_ld_weight_is_treatment": False,
    }
    assert pobax["adapted_ppo_gtrxl"]["evaluation_state_threading_review_required"] is True
    assert pobax["adapted_ppo_gtrxl"]["global_args_dependency_review_required"] is True


def test_memory_trace_grid_has_image_only_control_and_development_only_selection() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    traces = _descriptor_family(descriptor, "memory_traces")["semantic_contract"]
    assert traces["memoryless_control"] == {
        "candidate_id": "adapted_ppo_memory_trace_control_l0",
        "lambda_decimal_strings": ["0"],
        "observation_access": "forager_image_only",
    }
    assert traces["candidate_grid"] == [
        ["0", "0.9"],
        ["0", "0.99"],
        ["0", "0.999"],
        ["0", "0.9", "0.99", "0.999"],
    ]
    assert traces["selection_seed_class"] == "development_only"
    assert traces["qualification_or_held_out_seed_selection_allowed"] is False


def test_shm_reproducibility_split_is_explicit_and_candidates_cannot_be_conflated() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    shm = _descriptor_family(descriptor, "shm")["semantic_contract"]
    popgym = shm["popgym_root_semantics"]
    intended = shm["pomdp_torchkit_semantics"]
    assert popgym["candidate_id"] == "adapted_ppo_shm_popgym_code_faithful"
    assert popgym["row_selection"] == "uniform_(0,1).long_selects_row_0"
    assert intended["candidate_id"] == "adapted_ppo_shm_intended_random"
    assert intended["row_selection"] == "randint_over_128_then_state_clamp"
    assert shm["semantics_are_distinct"] is True
    assert shm["candidate_concepts_may_be_conflated"] is False
    assert popgym != intended


def test_ffm_source_and_paper_semantics_are_explicitly_distinct() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    ffm = _descriptor_family(descriptor, "ffm")["semantic_contract"]
    source = ffm["jax_source_semantics"]
    paper = ffm["paper_scale_semantics"]

    assert source["candidate_id"] == "adapted_ppo_ffm_jax_source"
    assert source["decay_parameter_initialization"] == (
        "linspace_negative_e_to_negative_1e_6_by_memory_size"
    )
    assert source["done_mask_application"] == (
        "incoming_done_masks_predecessor_before_incoming_input_addition"
    )
    assert source["hidden_size_affects_recurrence"] is False
    assert source["paper_configuration_exact"] is False

    assert paper["candidate_id"] == "adapted_ppo_ffm_paper_m32_c4"
    assert paper["memory_size"] == 32
    assert paper["context_size"] == 4
    assert paper["paper_recurrent_complex_elements"] == 128
    assert paper["paper_recurrent_real_equivalent_dimensions"] == 256
    assert paper["decay_retention_beta_decimal"] == "0.01"
    assert paper["initialization_horizon_steps"] == 1_024
    assert paper["recommended_gamma_multiplication_precision"] == "float64"
    assert paper["source_jax_implementation_asserted_equivalent"] is False

    assert ffm["candidate_concepts_may_be_conflated"] is False
    assert ffm["shared_adapter_policy"] == {
        "episode_boundary_alignment_must_be_frozen_and_tested": True,
        "forager_image_encoder_required": True,
        "matched_ppo_backbone_required": True,
        "previous_action_input_allowed": False,
        "upstream_exact_execution_asserted": False,
    }


def test_memoroids_tbb_is_a_non_authorizing_paper_only_gap() -> None:
    descriptor = sources.memory_comparator_development_sources_descriptor()
    assert descriptor["paper_only_unregistered_gaps"] == [
        {
            "audited_commit_git_sha1": "78709d2b5f99d40f10c8f5f4047c15f3dbb023b9",
            "audited_tree_git_sha1": "213ecff0e04cdc989087cb7f7a27b718fa3839f8",
            "candidate_registered": False,
            "clean_room_paper_specification_required": True,
            "code_adaptation_allowed_by_this_registry": False,
            "family_id": "memoroids_s5_tbb",
            "license_observation": (
                "no_repository_license_or_license_file_declared_in_audited_snapshot"
            ),
            "official_repository_url": "https://github.com/proroklab/memoroids",
            "paper_arxiv_id": "2402.09900v3",
            "paper_venue": "NeurIPS_2024",
            "performance_claim_reproduced": False,
            "source_family_registered": False,
            "source_imported_or_executed_here": False,
            "tbb_matched_update_budget_frozen": False,
        }
    ]
    assert "memoroids_s5_tbb" not in sources.SOURCE_FAMILY_BY_ID
    assert all(
        "memoroid" not in candidate_id
        for candidate_id in sources.DEVELOPMENT_MEMORY_CANDIDATE_IDS
    )
