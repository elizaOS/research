"""Adversarial tests for the pure matched-v3 runtime-storage-escape protocol."""

from alberta_framework.benchmarks import (
    forager_matched_v3_storage_runtime_escape_protocol_v1 as escape,
)


def test_runtime_escape_schema_inventory_is_exact() -> None:
    assert escape.RAW_SCHEMA_INVENTORY == (
        "alberta.forager_matched_v3.raw.docker_implicit_mount_inventory_pre_go.v1",
        "alberta.forager_matched_v3.raw.docker_create_inspect_storage_projection.v1",
        "alberta.forager_matched_v3.raw.final_oci_spec_storage_projection.v1",
        "alberta.forager_matched_v3.raw.console_fifo_stdio_inventory.v1",
        "alberta.forager_matched_v3.raw.docker_storage_api_operation_journal.v1",
        "alberta.forager_matched_v3.raw.rootfs_upperdir_pre_go_baseline.v1",
        "alberta.forager_matched_v3.raw.rootfs_upperdir_go_to_seal_delta.v1",
        "alberta.forager_matched_v3.raw.docker_volume_inventory_delta.v1",
    )
