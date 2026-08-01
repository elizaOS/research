from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_rng_parity as parity
from alberta_framework.benchmarks import forager_rng_parity_qualification as qualification


def _runtime_identity(probe_source_sha256: str) -> parity.VerifiedRuntimeIdentity:
    return parity.VerifiedRuntimeIdentity(
        required_oci_image_id=parity.REQUIRED_OCI_IMAGE_ID,
        build_attestation_sha256=parity.REQUIRED_BUILD_ATTESTATION_SHA256,
        source_repository=parity.REQUIRED_SOURCE_REPOSITORY,
        source_commit=parity.REQUIRED_SOURCE_COMMIT,
        source_tree_git_sha1=parity.REQUIRED_SOURCE_TREE_GIT_SHA1,
        source_archive_sha256=parity.REQUIRED_SOURCE_ARCHIVE_SHA256,
        source_archive_inventory_sha256=(parity.REQUIRED_SOURCE_ARCHIVE_INVENTORY_SHA256),
        dependency_lock_sha256=parity.REQUIRED_DEPENDENCY_LOCK_SHA256,
        wrapper_source_path=parity.REQUIRED_WRAPPER_PATH.as_posix(),
        wrapper_source_sha256=parity.REQUIRED_WRAPPER_SHA256,
        source_mount_mode=parity.REQUIRED_SOURCE_MOUNT_MODE,
        foragax_distribution=parity.REQUIRED_FORAGAX_DISTRIBUTION,
        foragax_version=parity.REQUIRED_FORAGAX_VERSION,
        foragax_wheel_sha256=parity.REQUIRED_FORAGAX_WHEEL_SHA256,
        foragax_install_tree_hash_scheme=(parity.REQUIRED_FORAGAX_INSTALL_TREE_HASH_SCHEME),
        foragax_install_tree_sha256=parity.REQUIRED_FORAGAX_INSTALL_TREE_SHA256,
        python_version=parity.REQUIRED_PYTHON_VERSION,
        python_executable_sha256=parity.REQUIRED_PYTHON_EXECUTABLE_SHA256,
        jax_version=parity.REQUIRED_JAX_VERSION,
        jaxlib_version=parity.REQUIRED_JAXLIB_VERSION,
        backend=parity.REQUIRED_BACKEND,
        cpu_device_count=1,
        prng_impl=parity.REQUIRED_PRNG_IMPL,
        threefry_partitionable=True,
        jax_enable_x64=False,
        probe_module_sha256=probe_source_sha256,
    )


def _trace(config: parity.FixedActionProbeConfig) -> parity.RawEnvironmentTrace:
    reset_keys, transition_keys = parity.expected_key_schedule(config)
    return parity.RawEnvironmentTrace(
        reset=parity.RawResetRecord(
            keys=reset_keys,
            observation=np.asarray([0, 1], dtype=np.int32),
            state={"position": np.asarray([1, 0], dtype=np.int32)},
        ),
        transitions=tuple(
            parity.RawTransitionRecord(
                index=index,
                action=action,
                keys=transition_keys[index],
                observation=np.asarray([index, action], dtype=np.int32),
                reward=np.asarray(index - action, dtype=np.float32),
                done=np.asarray(False, dtype=np.bool_),
                info={"discount": np.asarray(1.0, dtype=np.float32)},
                state={"position": np.asarray([index + 1, action], dtype=np.int32)},
            )
            for index, action in enumerate(config.actions)
        ),
    )


def _collector(
    collector: parity.CollectorKind,
    plan: qualification.HostQualificationPlan,
    trace: parity.RawEnvironmentTrace | None = None,
) -> parity.ParityCollectorResult:
    actual_trace = _trace(plan.config) if trace is None else trace
    digest = parity.digest_environment_trace(
        plan.config,
        actual_trace,
        runner_label=collector,
    )
    draft = parity.ParityCollectorResult(
        collector=collector,
        runtime=_runtime_identity(plan.probe_source_sha256),
        config=plan.config,
        trace=digest,
        payload_sha256="",
    )
    return replace(
        draft,
        payload_sha256=hashlib.sha256(
            parity.canonical_json_bytes(draft.unsigned_dict())
        ).hexdigest(),
    )


def _probe_source() -> Path:
    return Path(parity.__file__).resolve(strict=True)


def test_plan_is_open_nonpromoting_and_never_executes_docker() -> None:
    plan = qualification.build_host_qualification_plan(_probe_source())
    payload = plan.to_dict()

    assert qualification.LIVE_DOCKER_EXECUTION_DEFERRED is True
    assert payload["classification"] == qualification.HOST_QUALIFICATION_CLASSIFICATION
    assert payload["seed_class"] == "public_open_qualification"
    assert payload["promotion_authorized"] is False
    assert payload["live_docker_execution_deferred"] is True
    assert payload["process_model"] == "two_distinct_ephemeral_oci_processes"
    assert payload["probe"] == qualification.OPEN_QUALIFICATION_CONFIG.to_dict()
    assert (
        payload["plan_sha256"]
        == hashlib.sha256(parity.canonical_json_bytes(plan.unsigned_dict())).hexdigest()
    )
    assert plan.canonical_bytes == parity.canonical_json_bytes(payload)


def test_materialized_commands_freeze_two_distinct_sandboxed_processes() -> None:
    probe_source = _probe_source()
    plan = qualification.build_host_qualification_plan(probe_source)
    wrapper, direct = qualification.materialize_collector_commands(plan, probe_source)

    assert wrapper != direct
    for command, collector in ((wrapper, "wrapper"), (direct, "direct")):
        assert command[:2] == (qualification.DOCKER_PATH, "run")
        assert command.count("run") == 1
        assert "--rm" in command
        assert "--pull=never" in command
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert "--security-opt=no-new-privileges" in command
        assert "--user=65532:65532" in command
        assert "--privileged" not in command
        for forbidden_name in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONBREAKPOINT",
            "PYTHONINSPECT",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        ):
            assert f"--env={forbidden_name}=" in command
        image_index = command.index(parity.REQUIRED_OCI_IMAGE_ID)
        assert command[image_index + 1 : image_index + 5] == (
            parity.REQUIRED_PYTHON_EXECUTABLE.as_posix(),
            "-I",
            "-B",
            qualification.CONTAINER_PROBE_PATH,
        )
        assert f"--collector={collector}" == command[-1]
        assert f"--seed={plan.config.seed}" in command
        assert ("--actions=" + ",".join(str(action) for action in plan.config.actions)) in command
        source_mount = next(
            argument for argument in command if argument.startswith("--mount=type=bind,source=")
        )
        assert f"source={probe_source.as_posix()}" in source_mount
        assert f"destination={qualification.CONTAINER_PROBE_PATH}" in source_mount
        assert source_mount.endswith(",readonly")
        assert not any(
            token in argument.upper()
            for argument in command
            for token in ("AWS_", "CREDENTIAL", "SECRET", "TOKEN")
        )
        parsed = parity.build_argument_parser().parse_args(command[image_index + 5 :])
        assert parsed.seed == plan.config.seed
        assert parsed.actions == plan.config.actions
        assert parsed.collector == collector
    assert wrapper[:-1] == direct[:-1]


def test_plan_and_materialization_reject_source_alias_or_drift(tmp_path: Path) -> None:
    source = tmp_path / "probe.py"
    source.write_bytes(b"print('first')\n")
    plan = qualification.build_host_qualification_plan(source)

    source.write_bytes(b"print('changed')\n")
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="differs from the frozen plan",
    ):
        qualification.materialize_collector_commands(plan, source)

    target = tmp_path / "target.py"
    target.write_bytes(b"print('target')\n")
    symbolic = tmp_path / "symbolic.py"
    symbolic.symlink_to(target)
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="symbolic link",
    ):
        qualification.build_host_qualification_plan(symbolic)

    hard_link = tmp_path / "hard.py"
    os.link(target, hard_link)
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="single-link",
    ):
        qualification.build_host_qualification_plan(target)


def test_qualification_receipt_replays_caller_held_inputs() -> None:
    plan = qualification.build_host_qualification_plan(_probe_source())
    wrapper = _collector("wrapper", plan)
    direct = _collector("direct", plan)
    executor_receipt = "e" * 64

    receipt = qualification.qualify_collector_outputs(
        plan,
        wrapper.canonical_bytes,
        direct.to_dict(),
        executor_qualification_receipt_sha256=executor_receipt,
    )
    payload = receipt.to_dict()

    assert payload["status"] == ("content_complete_external_executor_receipt_unverified")
    assert payload["promotion_authorized"] is False
    assert payload["external_executor_receipt_requires_trust_resolver"] is True
    assert payload["plan_sha256"] == plan.plan_sha256
    assert payload["wrapper_collector_sha256"] == wrapper.payload_sha256
    assert payload["direct_collector_sha256"] == direct.payload_sha256
    assert (
        payload["payload_sha256"]
        == hashlib.sha256(parity.canonical_json_bytes(receipt.unsigned_dict())).hexdigest()
    )
    assert (
        qualification.validate_host_qualification_receipt(
            receipt.canonical_bytes,
            plan,
            wrapper.to_dict(),
            direct.canonical_bytes,
            expected_executor_qualification_receipt_sha256=executor_receipt,
        )
        == receipt
    )


def test_qualification_rejects_role_source_trace_and_executor_drift() -> None:
    plan = qualification.build_host_qualification_plan(_probe_source())
    wrapper = _collector("wrapper", plan)
    direct = _collector("direct", plan)

    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="collector roles",
    ):
        qualification.qualify_collector_outputs(
            plan,
            direct.to_dict(),
            wrapper.to_dict(),
            executor_qualification_receipt_sha256="d" * 64,
        )

    wrong_source = replace(
        direct,
        runtime=replace(direct.runtime, probe_module_sha256="0" * 64),
        payload_sha256="",
    )
    wrong_source = replace(
        wrong_source,
        payload_sha256=hashlib.sha256(
            parity.canonical_json_bytes(wrong_source.unsigned_dict())
        ).hexdigest(),
    )
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="probe source",
    ):
        qualification.qualify_collector_outputs(
            plan,
            wrapper.to_dict(),
            wrong_source.to_dict(),
            executor_qualification_receipt_sha256="d" * 64,
        )

    changed_trace = _trace(plan.config)
    first = changed_trace.transitions[0]
    changed_trace = replace(
        changed_trace,
        transitions=(
            replace(first, reward=np.asarray(999.0, dtype=np.float32)),
            *changed_trace.transitions[1:],
        ),
    )
    mismatched = _collector("direct", plan, changed_trace)
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="reward",
    ):
        qualification.qualify_collector_outputs(
            plan,
            wrapper.to_dict(),
            mismatched.to_dict(),
            executor_qualification_receipt_sha256="d" * 64,
        )

    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="lowercase SHA-256",
    ):
        qualification.qualify_collector_outputs(
            plan,
            wrapper.to_dict(),
            direct.to_dict(),
            executor_qualification_receipt_sha256="not-a-digest",
        )


def test_receipt_tampering_or_external_executor_mismatch_fails_replay() -> None:
    plan = qualification.build_host_qualification_plan(_probe_source())
    wrapper = _collector("wrapper", plan)
    direct = _collector("direct", plan)
    receipt = qualification.qualify_collector_outputs(
        plan,
        wrapper.to_dict(),
        direct.to_dict(),
        executor_qualification_receipt_sha256="a" * 64,
    )

    tampered: dict[str, Any] = json.loads(receipt.canonical_bytes)
    tampered["promotion_authorized"] = True
    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="does not replay",
    ):
        qualification.validate_host_qualification_receipt(
            tampered,
            plan,
            wrapper.to_dict(),
            direct.to_dict(),
            expected_executor_qualification_receipt_sha256="a" * 64,
        )

    with pytest.raises(
        qualification.ForagerRngParityQualificationError,
        match="does not replay",
    ):
        qualification.validate_host_qualification_receipt(
            receipt.to_dict(),
            plan,
            wrapper.to_dict(),
            direct.to_dict(),
            expected_executor_qualification_receipt_sha256="b" * 64,
        )
