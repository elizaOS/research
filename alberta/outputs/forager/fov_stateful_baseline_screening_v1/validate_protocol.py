#!/usr/bin/env python3
"""Fail-closed validator for the open-development stateful-baseline screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
COMMIT = "9710f60fa30da5badc451ad7ce3ff296d5070830"
TREE = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
EXPECTED_TASK = {
    "aperture_size": 9,
    "env_id": "ForagaxTwoBiomeLarge-v1",
    "observation_type": "color",
}
EXPECTED_CONFIGS = (
    "configs/DQN-PT-architecture-control.json",
    "configs/DQN-ReDo-architecture-control.json",
    "configs/DQN_ReDo_PostLNScore.json",
    "configs/DRQN-current-XFinal.json",
    "configs/DRQN-paper-v1.json",
    "configs/PPO-RTU_LN_128_1_relu.json",
    "configs/PPO_2048_relu.json",
    "configs/PT_DQN.json",
)
EXPECTED_AGENTS = {
    "configs/DQN-PT-architecture-control.json": "DQN",
    "configs/DQN-ReDo-architecture-control.json": "DQN",
    "configs/DQN_ReDo_PostLNScore.json": "DQN_ReDo_PostLNScore",
    "configs/DRQN-current-XFinal.json": "DRQN",
    "configs/DRQN-paper-v1.json": "DRQN",
    "configs/PPO-RTU_LN_128_1_relu.json": "PPO-RTU_LN_128_1_relu",
    "configs/PPO_2048_relu.json": "PPO_2048_relu",
    "configs/PT_DQN.json": "PT_DQN",
}
EXPECTED_TEMPLATES = {
    "xfinal_redo_fixed": {
        "path": (
            "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/"
            "DQN_ReDo_PostLNScore.json"
        ),
        "sha256": "61fa39de8426e2fb78305846b26f6c7a977c72b9cc8a61fc70419f8c15afc8ab",
        "git_blob_sha1": "628b157a504e2983b23d9a79de2c8b0de5588d72",
        "size_bytes": 1207,
    },
    "xfinal_drqn_current_audit_checkout_fixed": {
        "path": "experiments/XFinal/foragax/ForagaxSquareWaveTwoBiome-v11/9/DRQN.json",
        "sha256": "70a5ee902aa6128ec65c6d4fd33e27da0e3eaa02bd4ea8b776baf3fa158c27de",
        "git_blob_sha1": "c07ce8a880d4ea0de6f9c0997e73cc93c3328df9",
        "size_bytes": 1034,
    },
    "x33_drqn_published_v1_fixed": {
        "path": (
            "experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/DRQN.json"
        ),
        "sha256": "7fd55284b98b19d9068eb1a63206744b71c7371f06f24891f4c78aea71b14635",
        "git_blob_sha1": "76200365ba5c512984609d464a82ce3e61fa1629",
        "size_bytes": 1031,
    },
    "x34_pt_dqn_published_v1_fixed": {
        "path": (
            "experiments/X34-ForagaxSquareWaveTwoBiome-v11/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PT_DQN.json"
        ),
        "sha256": "02daae7d0a14ff735cef658e4d97e8f5664266462f68fe791fbceb6edfc96b3d",
        "git_blob_sha1": "b8b7957fc772b7b5f95ea8fe7d88699e129a1498",
        "size_bytes": 1094,
    },
    "r1_ppo_2048_relu_fixed": {
        "path": (
            "experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PPO_2048_relu.json"
        ),
        "sha256": "0e0837940e02704735647d3583618fab101ee35cb36cd1de0b2d6d9e7d66125a",
        "git_blob_sha1": "85eb61245d1b7438a18cbf9951cd068b212f1695",
        "size_bytes": 1316,
    },
    "r1_rtu_ppo_128_relu_fixed": {
        "path": (
            "experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/"
            "ForagaxSquareWaveTwoBiome-v11/9/PPO-RTU_LN_128_1_relu.json"
        ),
        "sha256": "c2e7fcfee29b922bc526e983fbb15275fde20867841cdb172ad55e125bbdc73a",
        "git_blob_sha1": "c352fba43bb9fb7579ff8d4113fa23feb78e4601",
        "size_bytes": 1319,
    },
}
CONFIG_TO_TEMPLATE = {
    "configs/DQN-PT-architecture-control.json": "x34_pt_dqn_published_v1_fixed",
    "configs/DQN-ReDo-architecture-control.json": "xfinal_redo_fixed",
    "configs/DQN_ReDo_PostLNScore.json": "xfinal_redo_fixed",
    "configs/DRQN-current-XFinal.json": "xfinal_drqn_current_audit_checkout_fixed",
    "configs/DRQN-paper-v1.json": "x33_drqn_published_v1_fixed",
    "configs/PPO-RTU_LN_128_1_relu.json": "r1_rtu_ppo_128_relu_fixed",
    "configs/PPO_2048_relu.json": "r1_ppo_2048_relu_fixed",
    "configs/PT_DQN.json": "x34_pt_dqn_published_v1_fixed",
}
EXPECTED_SOURCE_PATHS = (
    "src/continuing_main.py",
    "src/rtu_ppo.py",
    "src/experiment/ExperimentModel.py",
    "src/problems/BaseProblem.py",
    "src/problems/Foragax.py",
    "src/problems/registry.py",
    "src/environments/Foragax.py",
    "src/algorithms/BaseAgent.py",
    "src/algorithms/registry.py",
    "src/algorithms/PPORegistry.py",
    "src/algorithms/nn/NNAgent.py",
    "src/algorithms/nn/DQN.py",
    "src/algorithms/nn/DQN_ReDo.py",
    "src/algorithms/nn/DRQN.py",
    "src/algorithms/nn/PT_DQN.py",
    "src/algorithms/nn/ACConv.py",
    "src/algorithms/nn/RealTimeACConv.py",
    "src/algorithms/nn/activations.py",
    "src/algorithms/nn/rtus/rtus.py",
    "src/representations/networks.py",
    "src/optimizers.py",
    "src/utils/rlglue/rl_glue.py",
    "pyproject.toml",
    "uv.lock",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        fail(f"non-finite JSON constant in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode()
    return hashlib.sha1(prefix + data).hexdigest()  # noqa: S324 - Git object identity


def require_safe_file(root: Path, relative: str) -> Path:
    require(not Path(relative).is_absolute(), f"absolute declared path: {relative}")
    path = root / relative
    require(path.is_file(), f"missing regular file: {path}")
    require(not path.is_symlink(), f"symlink is not allowed: {path}")
    require(path.resolve().is_relative_to(root.resolve()), f"path escapes root: {path}")
    return path


def assert_no_lists(value: Any, location: str) -> None:
    if isinstance(value, list):
        fail(f"configuration sweep/list is forbidden at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_lists(child, f"{location}.{key}")


def derived_template(template: dict[str, Any], config_path: str) -> dict[str, Any]:
    expected = copy.deepcopy(template)
    expected["total_steps"] = 102_400
    environment = expected["metaParameters"]["environment"]
    environment["env_id"] = "ForagaxTwoBiomeLarge-v1"
    environment["aperture_size"] = 9
    environment["observation_type"] = "color"
    if config_path == "configs/PPO_2048_relu.json":
        expected["metaParameters"]["num_updates"] = 50
    elif config_path == "configs/PPO-RTU_LN_128_1_relu.json":
        expected["metaParameters"]["num_updates"] = 800
    elif config_path == "configs/DQN-PT-architecture-control.json":
        expected["agent"] = "DQN"
        for field in ("pm_buffer_size", "pt_decay", "pt_optimizer", "pt_update_freq"):
            del expected["metaParameters"][field]
    elif config_path == "configs/DQN-ReDo-architecture-control.json":
        expected["agent"] = "DQN"
        for field in (
            "redo_freq",
            "redo_reset_layernorm",
            "redo_score_after_ln",
            "redo_threshold",
        ):
            del expected["metaParameters"][field]
    return expected


def validate_protocol(protocol: dict[str, Any], upstream: Path) -> dict[str, str]:
    require(
        protocol.get("schema_version")
        == "alberta.forager_fov_stateful_baseline_screening.v1",
        "wrong protocol schema",
    )
    require(protocol.get("status") == "configuration_frozen_execution_pending", "wrong status")
    require(protocol.get("stage") == "open_development_second_stage", "wrong stage")
    require(protocol.get("evidence_class") == "open_development", "wrong evidence class")
    require(protocol.get("scientific_promotion_allowed") is False, "promotion must be false")
    require(protocol.get("sota_claim_allowed") is False, "SOTA claim must be false")

    task = protocol["task"]
    require(task["foragax_distribution"] == "continual-foragax", "wrong distribution")
    require(task["foragax_version"] == "0.55.0", "wrong Foragax version")
    require(task["env_id"] == EXPECTED_TASK["env_id"], "wrong environment")
    require(task["aperture_size"] == 9, "wrong aperture")
    require(task["observation_type"] == "color", "wrong observation type")
    require(task["steps_per_seed"] == 102_400, "wrong horizon")
    require(task["seeds"] == [2_000_001, 2_000_002], "wrong seeds")
    transport = task["seed_transport"]
    require(transport["pyexputils_indices"] == "2000001:2000003", "wrong indices")
    require(transport["nested_seed_offset"] == 0, "seed offset is not zero")
    require(transport["expected_stored_seeds"] == task["seeds"], "wrong stored seeds")
    require(transport["expected_effective_seeds"] == task["seeds"], "wrong effective seeds")

    metric = protocol["metric"]
    require(
        metric["source"] == "raw per-transition rewards from each upstream NPZ archive",
        "raw source",
    )
    require(metric["collector_summaries_used"] is False, "collector summaries forbidden")
    require(metric["reward_trace_shape"] == [102_400], "wrong trace shape")
    require(metric["ema_decay"] == 0.999, "wrong EMA decay")
    require(metric["ema_initial_value"] == 0.0, "wrong EMA initialization")
    require(metric["bias_correction"] is False, "bias correction must be false")
    require(metric["subsample_every_steps"] == 100, "wrong sample cadence")
    require(metric["subsample_index_origin"] == 0, "wrong sample origin")
    require(metric["sample_count"] == 1024, "wrong sample count")
    require(metric["tail_start_index"] == 921, "wrong tail start")
    require(metric["tail_sample_count"] == 103, "wrong tail count")
    require(metric["direction"] == "maximize", "wrong metric direction")
    score_path = require_safe_file(BASE, metric["implementation"]["path"])
    require(sha256_file(score_path) == metric["implementation"]["sha256"], "score hash")
    require(
        metric["implementation"]["sha256"]
        == "5ceaf510ae05b8bd34cac147b739032bffe5f536a130405afb178312fc66659a",
        "unexpected score implementation",
    )

    selection = protocol["selection_rule"]
    require(selection["frozen_before_reward_execution"] is True, "selection not frozen")
    require(selection["aggregate_statistic"].startswith("arithmetic mean"), "wrong aggregate")
    require(selection["ranking"] == "descending aggregate statistic", "wrong ranking")
    require(selection["advance_count"] == 3, "wrong advance count")
    require(
        selection["tie_break"] == "configuration path ascending by Unicode code point",
        "wrong tie break",
    )
    require(selection["paired_contrasts_affect_ranking"] is False, "paired ranking leak")

    templates = {
        item["id"]: {key: value for key, value in item.items() if key != "id"}
        for item in protocol["upstream_templates"]
    }
    require(templates == EXPECTED_TEMPLATES, "template declarations differ from frozen values")
    loaded_templates: dict[str, dict[str, Any]] = {}
    for template_id, declaration in EXPECTED_TEMPLATES.items():
        path = require_safe_file(upstream, declaration["path"])
        data = path.read_bytes()
        require(len(data) == declaration["size_bytes"], f"template size: {template_id}")
        require(sha256_bytes(data) == declaration["sha256"], f"template SHA-256: {template_id}")
        require(blob_sha1(data) == declaration["git_blob_sha1"], f"template blob: {template_id}")
        loaded_templates[template_id] = load_json(path)

    configurations = protocol["configurations"]
    declared_paths = tuple(item["path"] for item in configurations)
    require(declared_paths == EXPECTED_CONFIGS, "wrong configuration set or order")
    require(len(set(declared_paths)) == 8, "configuration paths are not unique")
    by_path = {item["path"]: item for item in configurations}
    config_hashes: dict[str, str] = {}
    for relative in EXPECTED_CONFIGS:
        declaration = by_path[relative]
        require(declaration["agent"] == EXPECTED_AGENTS[relative], f"agent: {relative}")
        require(
            declaration["upstream_template_id"] == CONFIG_TO_TEMPLATE[relative],
            f"template binding: {relative}",
        )
        path = require_safe_file(BASE, relative)
        config = load_json(path)
        assert_no_lists(config, relative)
        digest = sha256_file(path)
        require(digest == declaration["sha256"], f"configuration hash: {relative}")
        config_hashes[relative] = digest
        require(config["agent"] == EXPECTED_AGENTS[relative], f"loaded agent: {relative}")
        require(config["problem"] == "Foragax", f"problem: {relative}")
        require(config["total_steps"] == 102_400, f"horizon: {relative}")
        hypers = config["metaParameters"]
        require(hypers["environment"] == EXPECTED_TASK, f"task: {relative}")
        require(hypers["experiment"]["seed_offset"] == 0, f"seed offset: {relative}")
        template = loaded_templates[CONFIG_TO_TEMPLATE[relative]]
        require(
            config == derived_template(template, relative),
            f"non-declared derivation: {relative}",
        )

    ppo = load_json(BASE / "configs/PPO_2048_relu.json")["metaParameters"]
    rtu = load_json(BASE / "configs/PPO-RTU_LN_128_1_relu.json")["metaParameters"]
    require(ppo["rollout_steps"] * ppo["num_updates"] == 102_400, "PPO horizon product")
    require(rtu["rollout_steps"] * rtu["num_updates"] == 102_400, "RTU horizon product")

    paper_drqn = load_json(BASE / "configs/DRQN-paper-v1.json")["metaParameters"]
    current_drqn = load_json(BASE / "configs/DRQN-current-XFinal.json")["metaParameters"]
    require(
        (paper_drqn["optimizer"]["alpha"], paper_drqn["epsilon"], paper_drqn["batch"])
        == (0.001, 0.1, 4),
        "published-v1 DRQN values",
    )
    require(
        (current_drqn["optimizer"]["alpha"], current_drqn["epsilon"], current_drqn["batch"])
        == (0.0001, 0.25, 32),
        "current-XFinal DRQN values",
    )
    pt = load_json(BASE / "configs/PT_DQN.json")["metaParameters"]
    require(pt["optimizer"]["alpha"] == 0.0003, "PT-DQN alpha")
    require(
        pt["representation"]
        == {
            "conv": "None",
            "hidden": 32,
            "layers": 2,
            "type": "ForagerNet",
            "use_layernorm": True,
        },
        "PT-DQN architecture",
    )

    reference = protocol["published_reference"]
    require(reference["arxiv_identifier"] == "2605.01131", "wrong paper")
    require(reference["version"] == "v1", "wrong paper version")
    require(reference["relevant_table"] == 6, "wrong paper table")
    require(
        reference["retrieved_pdf_sha256"]
        == "eb84a712a8171d967b9a7860afa3d6a7df9107eaf4ab587848cdf5c848602ff0",
        "wrong paper PDF hash",
    )

    runtime = protocol["runtime"]
    require(runtime["commit"] == COMMIT, "wrong source commit")
    require(runtime["git_tree_sha1"] == TREE, "wrong source tree")
    require(
        runtime["development_image_id"]
        == "sha256:e8a9789cee5e1e607256a92f035013416479141ee3cd1d489af1b0738cb854c3",
        "wrong image",
    )
    require(runtime["qualified_production_image"] is False, "image qualification misclaimed")
    require(runtime["gpu_execution_performed_for_this_protocol"] is False, "GPU misclaimed")
    require(runtime["full_screen_execution"] == "pending", "full execution misclaimed")
    smoke = runtime["configuration_smoke"]
    smoke_path = require_safe_file(BASE, smoke["script"]["path"])
    require(sha256_file(smoke_path) == smoke["script"]["sha256"], "smoke helper hash")
    require(
        smoke["script"]["sha256"]
        == "cb0cac5deb7b7a7ce648cc57adf572a4244a91f259f4091c6165920ae774b2ac",
        "unexpected smoke helper",
    )
    require(smoke["exact_indices"] == "2000001:2000003", "wrong smoke indices")
    require(smoke["transitions_per_lane"] == 1, "wrong smoke transition count")

    source_declarations = protocol["source_files"]
    source_paths = tuple(item["path"] for item in source_declarations)
    require(source_paths == EXPECTED_SOURCE_PATHS, "wrong source-file selection")
    source_hashes: dict[str, str] = {}
    for declaration in source_declarations:
        path = require_safe_file(upstream, declaration["path"])
        digest = sha256_file(path)
        require(digest == declaration["sha256"], f"source hash: {declaration['path']}")
        source_hashes[declaration["path"]] = digest
    lock_text = (upstream / "uv.lock").read_text(encoding="utf-8")
    require(
        'name = "continual-foragax"\nversion = "0.55.0"' in lock_text,
        "uv.lock does not resolve continual-foragax 0.55.0",
    )

    git_dir = upstream / ".git"
    if git_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", f"{COMMIT}^{{tree}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        require(head == COMMIT, "checkout HEAD is not the frozen commit")
        require(tree == TREE, "checkout tree is not the frozen tree")

    return config_hashes | source_hashes


def validate_receipt(protocol: dict[str, Any], require_receipt: bool) -> bool:
    path = BASE / "CONFIGURATION_SMOKE_RECEIPT.json"
    if not path.exists():
        require(not require_receipt, "smoke receipt is required")
        return False
    receipt = load_json(path)
    require(
        receipt.get("schema_version") == "alberta.forager_stateful_smoke_receipt.v1",
        "wrong receipt schema",
    )
    require(receipt.get("status") == "passed", "receipt status is not passed")
    require(receipt.get("full_reward_execution_performed") is False, "reward execution misclaimed")
    require(receipt.get("save_or_video_stub_used") is False, "save/video stub was used")
    require(receipt["protocol_sha256"] == sha256_file(BASE / "PROTOCOL.json"), "protocol binding")
    require(receipt["image_id"] == protocol["runtime"]["development_image_id"], "receipt image")
    require(receipt["indices"] == [2_000_001, 2_000_002], "receipt indices")
    require(receipt["indices_argument"] == "2000001:2000003", "receipt index argument")

    artifacts = {item["path"]: item["sha256"] for item in receipt["artifact_hashes"]}
    required_artifacts = {
        "PROTOCOL.json",
        "README.md",
        "score_raw_rewards.py",
        "smoke_one_transition.py",
        "validate_protocol.py",
        *EXPECTED_CONFIGS,
    }
    require(set(artifacts) == required_artifacts, "wrong receipt artifact set")
    for relative, digest in artifacts.items():
        require(HEX64.fullmatch(digest) is not None, f"invalid artifact hash: {relative}")
        require(
            sha256_file(require_safe_file(BASE, relative)) == digest,
            f"artifact hash: {relative}",
        )

    records = receipt["smoke_results"]
    require(tuple(item["configuration"] for item in records) == EXPECTED_CONFIGS, "smoke set")
    require(len(records) == 8, "wrong smoke count")
    expected_runtime = {
        "continual_foragax_version": "0.55.0",
        "cuda_visible_devices": "",
        "gid": 65532,
        "jax_default_backend": "cpu",
        "jax_devices": ["TFRT_CPU_0"],
        "jax_platform_name": "cpu",
        "jax_platforms": "cpu",
        "network_interfaces": ["lo"],
        "nonroot": True,
        "nvidia_device_glob": [],
        "nvidia_visible_devices": "void",
        "root_filesystem_read_only": True,
        "uid": 65532,
    }
    protocol_configs = {item["path"]: item for item in protocol["configurations"]}
    for record in records:
        relative = record["configuration"]
        require(record["exit_code"] == 0, f"smoke exit: {relative}")
        require(record["wall_seconds"] > 0, f"smoke wall time: {relative}")
        for stream in ("stdout", "stderr"):
            stream_record = record[stream]
            require(HEX64.fullmatch(stream_record["sha256"]) is not None, f"{stream}: {relative}")
            require(stream_record["size_bytes"] >= 0, f"{stream} size: {relative}")
        result = record["parsed_result"]
        require(result["status"] == "passed", f"parsed smoke status: {relative}")
        require(result["runtime"] == expected_runtime, f"runtime: {relative}")
        loaded = result["loaded_config"]
        require(
            loaded["config_sha256"] == protocol_configs[relative]["sha256"],
            f"hash: {relative}",
        )
        require(loaded["agent"] == EXPECTED_AGENTS[relative], f"loaded agent: {relative}")
        require(loaded["configured_total_steps"] == 102_400, f"loaded horizon: {relative}")
        require(loaded["environment"] == EXPECTED_TASK, f"loaded task: {relative}")
        require(loaded["num_permutations"] == 1, f"permutations: {relative}")
        require(loaded["requested_indices"] == [2_000_001, 2_000_002], f"indices: {relative}")
        require(loaded["stored_seeds"] == [2_000_001, 2_000_002], f"stored seeds: {relative}")
        require(loaded["effective_seeds"] == [2_000_001, 2_000_002], f"seeds: {relative}")
        require(loaded["nested_seed_offset"] == 0, f"offset: {relative}")
        if relative == "configs/PPO_2048_relu.json":
            require((loaded["rollout_steps"], loaded["num_updates"]) == (2048, 50), "PPO schedule")
        elif relative == "configs/PPO-RTU_LN_128_1_relu.json":
            require((loaded["rollout_steps"], loaded["num_updates"]) == (128, 800), "RTU schedule")
        lanes = result["transition"]["lanes"]
        require([lane["seed"] for lane in lanes] == [2_000_001, 2_000_002], f"lanes: {relative}")
        require(all(lane["transition_count"] == 1 for lane in lanes), f"transitions: {relative}")
        require(all(math.isfinite(lane["reward"]) for lane in lanes), f"reward finite: {relative}")
        if relative.startswith("configs/PPO"):
            require(
                result["transition"]["entrypoint"] == "src/rtu_ppo.py",
                f"entrypoint: {relative}",
            )
            overrides = result["transition"]["smoke_scheduler_overrides"]
            require(overrides["rollout_steps"] == 1, f"PPO smoke rollout: {relative}")
            require(overrides["num_updates"] == 1, f"PPO smoke updates: {relative}")
            require(overrides["allocate_frames"] is False, f"PPO frames: {relative}")
            require(overrides["diagnostics"] is False, f"PPO diagnostics: {relative}")
        else:
            require(
                result["transition"]["entrypoint"] == "src/continuing_main.py",
                f"entrypoint: {relative}",
            )
            require(
                result["transition"]["smoke_scheduler_overrides"]
                == {"max_steps_per_lane": 1},
                f"DQN smoke scheduler: {relative}",
            )
    return True


def validate_layout(require_receipt: bool, require_immutable: bool) -> None:
    expected_files = {
        "PROTOCOL.json",
        "README.md",
        "score_raw_rewards.py",
        "smoke_one_transition.py",
        "validate_protocol.py",
        *EXPECTED_CONFIGS,
    }
    if require_receipt or (BASE / "CONFIGURATION_SMOKE_RECEIPT.json").exists():
        expected_files.add("CONFIGURATION_SMOKE_RECEIPT.json")
    actual_files = {path.relative_to(BASE).as_posix() for path in BASE.rglob("*") if path.is_file()}
    actual_dirs = {path.relative_to(BASE).as_posix() for path in BASE.rglob("*") if path.is_dir()}
    require(actual_files == expected_files, "unexpected, missing, or generated protocol files")
    require(actual_dirs == {"configs"}, "unexpected protocol directories")
    require(not any(path.is_symlink() for path in BASE.rglob("*")), "protocol symlink found")
    if require_immutable:
        for path in (BASE, BASE / "configs"):
            require(stat.S_IMODE(path.stat().st_mode) == 0o555, f"directory mode: {path}")
        for relative in expected_files:
            path = BASE / relative
            require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"file mode: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--require-receipt", action="store_true")
    parser.add_argument("--require-immutable", action="store_true")
    args = parser.parse_args()

    upstream = args.upstream_root.resolve(strict=True)
    require(upstream.is_dir(), "upstream root is not a directory")
    protocol = load_json(BASE / "PROTOCOL.json")
    validate_layout(args.require_receipt, args.require_immutable)
    validated_hashes = validate_protocol(protocol, upstream)
    receipt_present = validate_receipt(protocol, args.require_receipt)
    output = {
        "schema_version": "alberta.forager_stateful_protocol_validation.v1",
        "status": "passed",
        "protocol_sha256": sha256_file(BASE / "PROTOCOL.json"),
        "configuration_count": len(EXPECTED_CONFIGS),
        "template_count": len(EXPECTED_TEMPLATES),
        "source_file_count": len(EXPECTED_SOURCE_PATHS),
        "validated_hash_count": len(validated_hashes),
        "receipt_present": receipt_present,
        "immutability_required": args.require_immutable,
        "full_reward_execution_performed": False,
    }
    print(json.dumps(output, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
