"""Contracts for the issued-unused v3 exogenous source binding."""

from __future__ import annotations

import ast
import base64
import dataclasses
import hashlib
import itertools
from importlib.metadata import FileHash, PackagePath
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as protocol,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_source as source_binding,
)

pytestmark = pytest.mark.integration

SOURCE_PATH = (
    Path(__file__).parents[1]
    / "alberta_framework/evaluation/compositional_future_utility_calibration_v3_source.py"
)

CONSUMED_HISTORY_KEY_ROLES = (
    "root",
    "observations",
    "exploration",
    "random_actions",
    "learner_genesis",
)


def test_key_manifest_is_protocol_derived_literal_and_distinct() -> None:
    assert source_binding.PROTOCOL_CONFIG_SHA256 == protocol.PROTOCOL_CONFIG_SHA256
    assert source_binding.KEY_MANIFEST == {
        "observations": (1_180_172_486, 737_689_529),
        "exploration": (1_781_034_651, 2_339_008_862),
        "random_actions": (3_049_045_980, 2_471_128_907),
        "learner_genesis": (2_648_309_318, 4_192_599_369),
    }
    assert len(set(source_binding.KEY_MANIFEST.values())) == 4
    assert source_binding.KEY_MANIFEST_SHA256 == (
        "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"
    )
    for role, expected in source_binding.KEY_MANIFEST.items():
        assert source_binding.derive_role_key_words(
            protocol.PROTOCOL_CONFIG_SHA256,
            role,
        ) == expected


def test_bound_source_reconstructs_every_stream_and_cadence_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_init(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("source binding initialized a learner")

    monkeypatch.setattr(CompositionalFeatureLearner, "init", forbidden_init)
    bound = source_binding.build_bound_v3_source()

    assert bound.validated
    assert bound.stream_sha256 == (
        "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758"
    )
    assert bound.cadence_bound_stream_sha256 == (
        "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
    )
    assert bound.stream_envelope_sha256 == (
        "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
    )
    assert bound.consumed_history_sha256 == source_binding.CONSUMED_HISTORY_SHA256
    assert int(jnp.count_nonzero(bound.source.curation_due_mask)) == 281
    assert bound.source.observations.shape == (8_998, 6)
    assert bound.source.phase_indices.shape == (8_998,)
    assert bound.source.stream_sha256 != source_binding.CONSUMED_STREAM_SHA256


def test_public_bound_validator_revalidates_and_returns_the_exact_supplied_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = source_binding.build_bound_v3_source()
    expected_hashes = {
        "protocol": "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c",
        "control_protocol": (
            "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
        ),
        "runtime": "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6",
        "consumed_history": (
            "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
        ),
        "key_manifest": (
            "ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"
        ),
        "stream": "f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758",
        "cadence_stream": (
            "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
        ),
        "stream_envelope": (
            "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
        ),
    }
    observed_revalidations: list[
        tuple[
            control.CompositionalControlLifeProtocol,
            control.BoundCompositionalControlLifeSource,
        ]
    ] = []
    original = source_binding._validate_constructed_source

    def track_revalidation(
        control_protocol: control.CompositionalControlLifeProtocol,
        source: control.BoundCompositionalControlLifeSource,
    ) -> tuple[source_binding.StreamArrayRecord, ...]:
        observed_revalidations.append((control_protocol, source))
        return original(control_protocol, source)

    monkeypatch.setattr(
        source_binding,
        "_validate_constructed_source",
        track_revalidation,
    )

    returned = source_binding.validate_bound_v3_source(bound)

    assert returned is bound
    assert observed_revalidations == [(bound.control_protocol, bound.source)]
    assert {
        "protocol": source_binding.PROTOCOL_CONFIG_SHA256,
        "control_protocol": source_binding.CONTROL_PROTOCOL_CONFIG_SHA256,
        "runtime": source_binding.RUNTIME_CONFIG_SHA256,
        "consumed_history": source_binding.CONSUMED_HISTORY_SHA256,
        "key_manifest": source_binding.KEY_MANIFEST_SHA256,
        "stream": source_binding.STREAM_SHA256,
        "cadence_stream": source_binding.CADENCE_BOUND_STREAM_SHA256,
        "stream_envelope": source_binding.STREAM_ENVELOPE_SHA256,
    } == expected_hashes
    assert "validate_bound_v3_source" in source_binding.__all__


@pytest.mark.parametrize(
    "tamper",
    ("source_array", "stream_receipt", "stream_records", "validated"),
)
def test_public_bound_validator_rejects_dataclasses_replace_derived_tampering(
    tamper: str,
) -> None:
    bound = source_binding.build_bound_v3_source()
    tampered = dataclasses.replace(bound)
    if tamper == "source_array":
        altered_source = dataclasses.replace(
            tampered.source,
            observations=tampered.source.observations.at[0, 0].multiply(-1),
        )
        object.__setattr__(tampered, "source", altered_source)
    elif tamper == "stream_receipt":
        object.__setattr__(tampered, "stream_sha256", "0" * 64)
    elif tamper == "stream_records":
        object.__setattr__(tampered, "stream_records", tampered.stream_records[:-1])
    else:
        object.__setattr__(tampered, "validated", False)

    with pytest.raises((TypeError, ValueError, RuntimeError), match="receipt|stream|array|valid"):
        source_binding.validate_bound_v3_source(tampered)


def test_public_bound_validator_rejects_unbound_type_and_protocol_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = source_binding.build_bound_v3_source()
    with pytest.raises(TypeError, match="BoundV3Source"):
        source_binding.validate_bound_v3_source(cast(Any, bound.source))

    monkeypatch.setattr(control, "ALLOCATED_MAX_DEPTH", control.ALLOCATED_MAX_DEPTH + 1)
    with pytest.raises(ValueError, match="geometry|semantic|control"):
        source_binding.validate_bound_v3_source(bound)


def test_stream_envelope_pins_exact_array_metadata_and_bytes() -> None:
    bound = source_binding.build_bound_v3_source()
    records = {record.name: record for record in bound.stream_records}

    assert tuple(records) == source_binding.STREAM_ARRAY_NAMES
    assert records["observations"].shape == (8_998, 6)
    assert records["observations"].dtype == "<f4"
    assert records["observations"].nbytes == 215_952
    assert records["observations"].bytes_sha256 == (
        "af5d844a6d1fb34846c2546f0c65d40b08ecb6f73c43377462bab6b248eda037"
    )
    assert records["curation_due_mask"].dtype == "|b1"
    assert records["curation_due_mask"].nbytes == 8_998
    assert sum(record.nbytes for record in records.values()) == 305_932
    assert all(len(record.bytes_sha256) == 64 for record in records.values())


def test_protocol_binds_full_production_config_arm_and_task_semantics() -> None:
    historical = control.learner_config_for_arm(protocol.LEFT_PACK_SOURCE_ARM)
    for arm in protocol.ARMS:
        expected = dict(historical)
        expected.update(
            {
                "candidate_scoring_mode": "legacy",
                "candidate_novelty_admission_bonus": 0.0,
                "future_utility_trace_mode": "contribution",
                "future_utility_mix": arm.future_utility_mix,
                "future_utility_trace_decay": arm.future_utility_trace_decay,
                "future_utility_normalization": arm.future_utility_normalization,
                "future_utility_normalization_decay": 0.99,
                "future_utility_rare_task_power": 0.0,
            }
        )
        assert protocol.reconstruct_arm_learner_config(arm) == expected

    source_arms = tuple(
        arm for arm in control.CONTROL_LIFE_ARMS if arm.name == protocol.LEFT_PACK_SOURCE_ARM
    )
    assert len(source_arms) == 1
    assert source_arms[0].to_config() == dict(protocol.SOURCE_ARM_CONFIG)
    assert control.PHASE_ORDER == protocol.PHASE_ORDER
    assert control.SIGNATURE_NAMES == protocol.SIGNATURE_NAMES
    assert control.SIGNATURE_RAW_INDICES == protocol.SIGNATURE_RAW_INDICES

    for values in itertools.product((-1.0, 1.0), repeat=protocol.RAW_DIM):
        observation = jnp.asarray(values, dtype=jnp.float32)
        for phase, raw_indices in enumerate(protocol.PHASE_TARGET_RAW_INDICES):
            expected_target = float(np.prod(np.asarray(values)[list(raw_indices)]))
            assert float(control._phase_target(observation, jnp.asarray(phase))) == (
                expected_target
            )


def test_source_receipt_rejects_altered_arrays_even_with_pinned_strings() -> None:
    bound = source_binding.build_bound_v3_source()
    altered_source = dataclasses.replace(
        bound.source,
        observations=jnp.zeros_like(bound.source.observations),
    )

    with pytest.raises((ValueError, RuntimeError), match="stream|receipt|array"):
        dataclasses.replace(bound, source=altered_source)


@pytest.mark.parametrize(
    "field",
    ("phase_indices", "curation_due_mask", "learner_key"),
)
def test_source_receipt_rejects_schedule_cadence_or_genesis_tamper(field: str) -> None:
    bound = source_binding.build_bound_v3_source()
    if field == "phase_indices":
        altered_source = dataclasses.replace(
            bound.source,
            phase_indices=bound.source.phase_indices.at[0].set(1),
        )
    elif field == "curation_due_mask":
        altered_source = dataclasses.replace(
            bound.source,
            curation_due_mask=bound.source.curation_due_mask.at[0].set(True),
        )
    else:
        altered_source = dataclasses.replace(bound.source, learner_key=jr.key(0))

    with pytest.raises((ValueError, RuntimeError), match="stream|phase|cadence|key|array"):
        dataclasses.replace(bound, source=altered_source)


def test_source_receipt_rejects_same_total_different_control_protocol() -> None:
    bound = source_binding.build_bound_v3_source()
    lengths = list(protocol.PHASE_LENGTHS)
    lengths[0], lengths[1] = lengths[1], lengths[0]
    altered_protocol = control.CompositionalControlLifeProtocol(
        phase_lengths=tuple(lengths),
        epsilon=protocol.EPSILON,
        entry_window=protocol.ENTRY_WINDOW,
        tail_window=protocol.TAIL_WINDOW,
    )

    with pytest.raises(ValueError, match="control protocol"):
        dataclasses.replace(bound, control_protocol=altered_protocol)


def test_source_receipt_rejects_host_array_legacy_key_or_mutable_manifest() -> None:
    bound = source_binding.build_bound_v3_source()
    altered_sources = (
        dataclasses.replace(
            bound.source,
            observations=cast(Any, np.asarray(bound.source.observations)),
        ),
        dataclasses.replace(
            bound.source,
            learner_key=jnp.asarray(
                source_binding.KEY_MANIFEST["learner_genesis"],
                dtype=jnp.uint32,
            ),
        ),
        dataclasses.replace(
            bound.source,
            key_manifest=dict(bound.source.key_manifest),
        ),
    )

    for altered_source in altered_sources:
        with pytest.raises((TypeError, ValueError), match="JAX|typed|manifest"):
            dataclasses.replace(bound, source=altered_source)


def test_runtime_mismatch_is_rejected_before_source_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = dict(source_binding.RUNTIME_CONFIG)
    observed["jax_threefry_partitionable"] = False
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: observed,
        raising=False,
    )

    def forbidden_builder(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("source generation began before runtime rejection")

    monkeypatch.setattr(
        control,
        "build_bound_compositional_control_life_source",
        forbidden_builder,
    )
    with pytest.raises(ValueError, match="runtime"):
        source_binding.build_bound_v3_source()


def test_runtime_binds_exact_installed_dependency_file_manifests() -> None:
    expected = {
        "chex": (
            "0.1.92",
            28,
            "50ada405c5dd57cca4da5add6f4b8f63bb053d5d24407f48ab4f7b86e3015664",
        ),
        "jax": (
            "0.11.0",
            629,
            "83109308c4587705e55cca19f7325b68a0f93542de44891404655ebab43dc52b",
        ),
        "jaxlib": (
            "0.11.0",
            133,
            "08392cf01d90354a176bd0ce41e6602368b49069140639e0231358b8642a798b",
        ),
        "jaxtyping": (
            "0.3.11",
            21,
            "60ec4237d3efc025a5af88eadbd9137ed83aed18130961e0d78b103bc62d301e",
        ),
    }
    assert tuple(expected) == source_binding.BYTE_BOUND_RUNTIME_DISTRIBUTIONS
    for name, (version, file_count, digest) in expected.items():
        manifest = source_binding._installed_distribution_manifest(name)
        files = cast(list[dict[str, object]], manifest["files"])
        assert manifest == {
            "schema": source_binding.INSTALLED_DISTRIBUTION_MANIFEST_SCHEMA,
            "distribution": name,
            "version": version,
            "files": files,
        }
        assert len(files) == file_count
        assert [record["path"] for record in files] == sorted(
            cast(str, record["path"]) for record in files
        )
        assert not any(str(record["path"]).endswith("/RECORD") for record in files)
        assert all(
            type(record["nbytes"]) is int
            and record["nbytes"] >= 0
            and type(record["sha256"]) is str
            and len(record["sha256"]) == 64
            for record in files
        )
        assert source_binding._canonical_sha256(manifest) == digest

    observed = source_binding._observed_runtime_config()
    assert observed == dict(source_binding.RUNTIME_CONFIG)
    assert source_binding._canonical_sha256(observed) == (
        source_binding.RUNTIME_CONFIG_SHA256
    )


def test_installed_distribution_manifest_rejects_record_byte_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_file = tmp_path / "chex/__init__.py"
    package_file.parent.mkdir()
    package_file.write_bytes(b"exact installed package bytes\n")
    record_file = tmp_path / "chex-0.1.92.dist-info/RECORD"
    record_file.parent.mkdir()
    record_file.write_bytes(b"fixture RECORD bytes\n")

    package_path = PackagePath("chex/__init__.py")
    wrong_digest = hashlib.sha256(b"different bytes").digest()
    cast(Any, package_path).hash = FileHash(
        "sha256="
        + base64.urlsafe_b64encode(wrong_digest).rstrip(b"=").decode("ascii")
    )
    cast(Any, package_path).size = package_file.stat().st_size
    record_path = PackagePath("chex-0.1.92.dist-info/RECORD")
    cast(Any, record_path).hash = None
    cast(Any, record_path).size = None

    class FakeDistribution:
        metadata = {"Name": "chex"}
        version = "0.1.92"
        files = [package_path, record_path]

        @staticmethod
        def locate_file(path: object) -> Path:
            return tmp_path / str(path)

    monkeypatch.setattr(
        source_binding,
        "package_distribution",
        lambda _name: FakeDistribution(),
    )
    with pytest.raises(ValueError, match="differs from its RECORD receipt"):
        source_binding._installed_distribution_manifest("chex")


def test_dependency_manifest_drift_is_rejected_in_nonarray_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = source_binding._installed_distribution_manifest

    def altered_manifest(name: str) -> dict[str, object]:
        manifest = original(name)
        if name == "chex":
            files = cast(list[dict[str, object]], manifest["files"])
            files[0]["sha256"] = "0" * 64
        return manifest

    monkeypatch.setattr(
        source_binding,
        "_installed_distribution_manifest",
        altered_manifest,
    )
    with pytest.raises(ValueError, match="runtime configuration"):
        source_binding.validate_protocol_and_source_constants()


def test_source_validator_rejects_control_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "ALLOCATED_MAX_DEPTH", 4)
    with pytest.raises(ValueError, match="geometry|semantic|control"):
        source_binding.validate_protocol_and_source_constants()


def test_bound_phase_runs_and_due_indices_match_the_exact_protocol() -> None:
    bound = source_binding.build_bound_v3_source()
    phases = np.asarray(bound.source.phase_indices)
    change_points = np.flatnonzero(phases[1:] != phases[:-1]) + 1
    runs = np.diff(np.concatenate(([0], change_points, [phases.size])))
    due_indices = np.flatnonzero(np.asarray(bound.source.curation_due_mask)) + 1

    assert tuple(int(value) for value in runs) == protocol.PHASE_LENGTHS
    assert np.array_equal(
        due_indices,
        np.arange(32, protocol.TOTAL_STEPS + 1, 32, dtype=np.int64),
    )
    assert bound.control_protocol_config_sha256 == (
        "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
    )


def test_all_declared_consumed_streams_and_key_manifests_are_excluded() -> None:
    assert set(source_binding.CONSUMED_STREAM_SHA256S) == {
        "02fd5efbbb304b624fcfd29e259c361d5048233817e896300057d8e36f3fc036",
        "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c",
    }
    assert source_binding.STREAM_SHA256 not in source_binding.CONSUMED_STREAM_SHA256S
    consumed_words = {
        words
        for manifest in source_binding.CONSUMED_KEY_MANIFESTS
        for words in manifest.values()
    }
    assert not (set(source_binding.KEY_MANIFEST.values()) & consumed_words)


def test_consumed_history_digest_binds_both_complete_historical_records() -> None:
    expected = {
        "schema": (
            "alberta.compositional-future-utility-calibration-v3-consumed-history.v1"
        ),
        "scope": (
            "development-history reuse exclusion only; no result, evidence, or "
            "promotion authority"
        ),
        "records": [
            {
                "lane": "compositional_future_utility_development_v1",
                "development_root": 329_631_721,
                "development_root_hex": "0x13A5C7E9",
                "stream_sha256": (
                    "02fd5efbbb304b624fcfd29e259c361d5048233817e896300057d8e36f3fc036"
                ),
                "key_manifest": {
                    "root": [0, 329_631_721],
                    "observations": [2_316_273_231, 3_036_545_927],
                    "exploration": [2_227_216_649, 3_977_711_669],
                    "random_actions": [382_045_127, 333_255_797],
                    "learner_genesis": [2_002_082_676, 3_427_004_161],
                },
            },
            {
                "lane": "compositional_future_utility_calibration_v2",
                "development_root": 1_924_178_934,
                "development_root_hex": "0x72B0A3F6",
                "stream_sha256": (
                    "bb741db073a13026425d2cc98cce93a1af1d1b65f2abf24ebc97e43b61abd39c"
                ),
                "key_manifest": {
                    "root": [0, 1_924_178_934],
                    "observations": [1_189_056_302, 2_383_774_845],
                    "exploration": [3_352_410_003, 3_947_271_724],
                    "random_actions": [3_382_640_669, 4_117_898_437],
                    "learner_genesis": [2_592_838_183, 3_227_537_730],
                },
            },
        ],
    }

    assert source_binding.canonical_consumed_history_config() == expected
    assert source_binding.CONSUMED_HISTORY_SHA256 == (
        "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
    )
    assert source_binding.STREAM_ENVELOPE_SCHEMA == (
        "alberta.compositional-future-utility-calibration-v3-source-stream."
        "full-binding.v2"
    )


@pytest.mark.parametrize("history_index", (0, 1))
def test_each_consumed_stream_mutation_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    history_index: int,
) -> None:
    streams = list(source_binding.CONSUMED_STREAM_SHA256S)
    streams[history_index] = str(history_index + 1) * 64
    monkeypatch.setattr(source_binding, "CONSUMED_STREAM_SHA256S", tuple(streams))
    if history_index == 1:
        monkeypatch.setattr(source_binding, "CONSUMED_STREAM_SHA256", streams[-1])
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight began")),
    )

    with pytest.raises(ValueError, match="consumed history|consumed stream"):
        source_binding.validate_protocol_and_source_constants()


@pytest.mark.parametrize("history_index", (0, 1))
def test_each_consumed_protocol_root_mutation_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    history_index: int,
) -> None:
    roots = list(protocol.DECLARED_CONSUMED_DEVELOPMENT_ROOTS)
    roots[history_index] += 1
    monkeypatch.setattr(
        protocol,
        "DECLARED_CONSUMED_DEVELOPMENT_ROOTS",
        tuple(roots),
    )
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight began")),
    )

    with pytest.raises(ValueError, match="frozen|consumed development root"):
        source_binding.validate_protocol_and_source_constants()


@pytest.mark.parametrize(
    ("history_index", "role"),
    tuple(itertools.product((0, 1), CONSUMED_HISTORY_KEY_ROLES)),
)
def test_every_consumed_role_key_mutation_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    history_index: int,
    role: str,
) -> None:
    manifests = [dict(manifest) for manifest in source_binding.CONSUMED_KEY_MANIFESTS]
    old_words = manifests[history_index][role]
    replacement = (old_words[0], old_words[1] ^ 0x80000000)
    assert replacement not in source_binding.KEY_MANIFEST.values()
    manifests[history_index][role] = replacement
    monkeypatch.setattr(
        source_binding,
        "CONSUMED_KEY_MANIFESTS",
        tuple(MappingProxyType(manifest) for manifest in manifests),
    )
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight began")),
    )

    with pytest.raises(ValueError, match="consumed history|consumed root"):
        source_binding.validate_protocol_and_source_constants()


def test_incomplete_consumed_role_manifest_is_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = [dict(manifest) for manifest in source_binding.CONSUMED_KEY_MANIFESTS]
    del manifests[0]["learner_genesis"]
    monkeypatch.setattr(
        source_binding,
        "CONSUMED_KEY_MANIFESTS",
        tuple(MappingProxyType(manifest) for manifest in manifests),
    )
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight began")),
    )

    with pytest.raises(ValueError, match="roles are incomplete"):
        source_binding.validate_protocol_and_source_constants()


@pytest.mark.parametrize(
    ("constant_name", "replacement"),
    (
        ("CONSUMED_STREAM_SHA256", "3" * 64),
        ("CONSUMED_HISTORY_SHA256", "4" * 64),
    ),
)
def test_consumed_history_digest_or_compatibility_alias_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    replacement: str,
) -> None:
    monkeypatch.setattr(source_binding, constant_name, replacement)
    monkeypatch.setattr(
        source_binding,
        "_observed_runtime_config",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight began")),
    )

    with pytest.raises(ValueError, match="consumed history|consumed stream"):
        source_binding.validate_protocol_and_source_constants()


def test_stream_array_record_rejects_invalid_dtype_or_byte_geometry() -> None:
    with pytest.raises(ValueError, match="dtype"):
        source_binding.StreamArrayRecord("observations", "not-a-dtype", (1,), 1, "0" * 64)
    with pytest.raises(ValueError, match="nbytes"):
        source_binding.StreamArrayRecord("observations", "<f4", (2,), 7, "0" * 64)


def test_wrong_protocol_digest_cannot_reproduce_the_frozen_keys() -> None:
    wrong = "0" * 64
    assert {
        role: source_binding.derive_role_key_words(wrong, role)
        for role in source_binding.KEY_ROLES
    } != source_binding.KEY_MANIFEST
    with pytest.raises(ValueError, match="protocol config digest"):
        source_binding.validate_protocol_and_source_constants(
            observed_protocol_config_sha256=wrong
        )


def test_source_binding_has_no_attempt_result_or_scientific_authority() -> None:
    assert source_binding.DEVELOPMENT_ONLY
    assert source_binding.SOURCE_GENERATION_ALLOWED
    assert not source_binding.OPERATIONAL_ENTRY_CONSUMED
    assert not source_binding.PANEL_EXECUTION_AUTHORIZED
    assert not source_binding.RESULT_AVAILABLE
    assert not source_binding.OUTPUT_WRITES_ALLOWED
    assert not source_binding.EVIDENCE_AUTHORIZED
    assert not source_binding.SCIENTIFIC_PROMOTION_ALLOWED

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    names = {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert not any("runner" in name or "latch" in name for name in names)
    assert not any(name.startswith(("execute", "consume", "enter")) for name in names)


def test_source_arrays_are_finite_binary_or_phase_bounded() -> None:
    bound = source_binding.build_bound_v3_source()
    observations = np.asarray(bound.source.observations)
    phases = np.asarray(bound.source.phase_indices)
    exploration = np.asarray(bound.source.exploration_mask)
    actions = np.asarray(bound.source.random_actions)

    assert np.all(np.isfinite(observations))
    assert set(np.unique(observations)) == {-1.0, 1.0}
    assert set(np.unique(phases)) == set(range(10))
    assert exploration.dtype == np.bool_
    assert set(np.unique(actions)) <= {0, 1}
