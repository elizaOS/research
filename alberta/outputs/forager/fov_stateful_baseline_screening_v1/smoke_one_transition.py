#!/usr/bin/env python3
"""Development-only CPU construction and one-transition smoke probe.

This script is intentionally stored beside the derived configurations.  It
does not patch the audited upstream checkout.  PPO uses a smoke-only scheduler
override (one rollout step, one minibatch, one update) so that exactly one
environment transition is exercised while retaining the configured network,
optimizer, and task dispatch.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np

SOURCE_ROOT = Path("/opt/foragax-agents")
SOURCE_DIR = SOURCE_ROOT / "src"
if not SOURCE_DIR.is_dir():
    raise RuntimeError(f"audited source directory is absent: {SOURCE_DIR}")
sys.path.insert(0, str(SOURCE_DIR))


EXPECTED_TASK = {
    "aperture_size": 9,
    "env_id": "ForagaxTwoBiomeLarge-v1",
    "observation_type": "color",
}
EXPECTED_HORIZON = 102_400
EXPECTED_FORAGAX_VERSION = "0.55.0"


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError("configuration must be a JSON object")
    return payload


def _tree_stats(tree: Any) -> dict[str, int]:
    arrays = 0
    elements = 0
    nbytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        if str(leaf.dtype).startswith("key<"):
            leaf = jax.random.key_data(leaf)
        size = 1
        for dimension in leaf.shape:
            size *= int(dimension)
        arrays += 1
        elements += size
        nbytes += size * np.dtype(leaf.dtype).itemsize
    return {"array_leaves": arrays, "elements": elements, "bytes": nbytes}


def _runtime_probe() -> dict[str, Any]:
    network_interfaces = sorted(path.name for path in Path("/sys/class/net").glob("*"))
    return {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "nonroot": os.getuid() != 0,
        "root_filesystem_read_only": bool(os.statvfs("/").f_flag & os.ST_RDONLY),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "nvidia_device_glob": sorted(glob.glob("/dev/nvidia*")),
        "network_interfaces": network_interfaces,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platform_name": os.environ.get("JAX_PLATFORM_NAME"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "continual_foragax_version": importlib.metadata.version("continual-foragax"),
    }


def _assert_runtime(runtime: dict[str, Any]) -> None:
    if not runtime["nonroot"]:
        raise RuntimeError("smoke container must run as a nonroot user")
    if not runtime["root_filesystem_read_only"]:
        raise RuntimeError("smoke container root filesystem must be read-only")
    if runtime["jax_default_backend"] != "cpu":
        raise RuntimeError("smoke container must use the JAX CPU backend")
    if runtime["nvidia_device_glob"]:
        raise RuntimeError("smoke container exposes NVIDIA device nodes")
    if runtime["nvidia_visible_devices"] != "void":
        raise RuntimeError("NVIDIA_VISIBLE_DEVICES must be exactly 'void'")
    if runtime["continual_foragax_version"] != EXPECTED_FORAGAX_VERSION:
        raise RuntimeError("unexpected continual-foragax distribution version")
    if any(interface != "lo" for interface in runtime["network_interfaces"]):
        raise RuntimeError("networkless smoke container has a non-loopback interface")


def _validate_loaded_config(
    path: Path, indices: tuple[int, int]
) -> tuple[Any, dict[str, Any]]:
    from experiment import ExperimentModel

    raw = path.read_bytes()
    config = _load_json(path)
    experiment = ExperimentModel.load(path)
    if experiment.numPermutations() != 1:
        raise RuntimeError("configuration must resolve to exactly one permutation")
    if config.get("problem") != "Foragax":
        raise RuntimeError("configuration problem must be Foragax")
    if config.get("total_steps") != EXPECTED_HORIZON:
        raise RuntimeError("configuration has the wrong common horizon")
    hypers = experiment.get_hypers(indices[0])
    if hypers.get("environment") != EXPECTED_TASK:
        raise RuntimeError("configuration has the wrong exact task")
    stored_seeds = [experiment.getRun(index) for index in indices]
    if stored_seeds != list(indices):
        raise RuntimeError("one-permutation run indices do not resolve to requested seeds")
    if hypers.get("experiment", {}).get("seed_offset", 0) != 0:
        raise RuntimeError("nested development seed offset must be zero")

    rollout_steps = hypers.get("rollout_steps")
    num_updates = hypers.get("num_updates")
    if str(config["agent"]).startswith("PPO"):
        if not isinstance(rollout_steps, int) or not isinstance(num_updates, int):
            raise RuntimeError("PPO configurations must explicitly fix rollout and update counts")
        if rollout_steps * num_updates != EXPECTED_HORIZON:
            raise RuntimeError("PPO rollout_steps * num_updates must equal the common horizon")

    loaded = {
        "agent": config["agent"],
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "configured_total_steps": config["total_steps"],
        "num_permutations": experiment.numPermutations(),
        "requested_indices": list(indices),
        "stored_seeds": stored_seeds,
        "nested_seed_offset": hypers.get("experiment", {}).get("seed_offset", 0),
        "effective_seeds": [
            seed + hypers.get("experiment", {}).get("seed_offset", 0)
            for seed in stored_seeds
        ],
        "environment": hypers["environment"],
        "rollout_steps": rollout_steps,
        "num_updates": num_updates,
    }
    return experiment, loaded


def _reward_lanes(
    output_root: Path, indices: tuple[int, int], family: str
) -> list[dict[str, Any]]:
    reward_archives: dict[int, tuple[Path, np.ndarray]] = {}
    for archive_path in output_root.rglob("*.npz"):
        with np.load(archive_path, allow_pickle=False) as archive:
            if "rewards" not in archive.files:
                continue
            rewards = np.asarray(archive["rewards"])
        matching = [index for index in indices if archive_path.name == f"{index}.npz"]
        if len(matching) != 1:
            raise RuntimeError(f"cannot bind reward archive to one seed: {archive_path}")
        seed = matching[0]
        if seed in reward_archives:
            raise RuntimeError(f"duplicate reward archive for seed {seed}")
        reward_archives[seed] = (archive_path, rewards)
    if sorted(reward_archives) != list(indices):
        raise RuntimeError(
            f"expected reward archives for {list(indices)}, found {sorted(reward_archives)}"
        )

    lanes = []
    for seed in indices:
        archive_path, rewards = reward_archives[seed]
        if rewards.size != 1:
            raise RuntimeError(
                f"{family} smoke seed {seed} executed {rewards.size} transitions"
            )
        reward = float(rewards.reshape(-1)[0])
        if not np.isfinite(reward):
            raise RuntimeError(f"{family} smoke seed {seed} produced a non-finite reward")
        lanes.append(
            {
                "seed": seed,
                "transition_count": int(rewards.size),
                "reward": reward,
                "reward_archive_relative_to_tmp": archive_path.relative_to(
                    "/tmp"
                ).as_posix(),
            }
        )
    return lanes


def _smoke_dqn(
    config_path: Path, indices: tuple[int, int], agent_name: str
) -> dict[str, Any]:
    from algorithms.registry import getAgent

    output_root = Path("/tmp/stateful-dqn-smoke-output")
    checkpoint_root = Path("/tmp/stateful-dqn-smoke-checkpoints")
    original_argv = sys.argv
    sys.argv = [
        str(SOURCE_DIR / "continuing_main.py"),
        "--exp",
        str(config_path),
        "--idxs",
        f"{indices[0]}:{indices[-1] + 1}",
        "--save_path",
        str(output_root),
        "--checkpoint_path",
        str(checkpoint_root),
        "--max_steps",
        "1",
        "--silent",
    ]
    try:
        runpy.run_path(str(SOURCE_DIR / "continuing_main.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
    agent_class = getAgent(agent_name)
    return {
        "lanes": _reward_lanes(output_root, indices, "DQN-family"),
        "agent_class": f"{agent_class.__module__}.{agent_class.__qualname__}",
        "entrypoint": "src/continuing_main.py",
        "smoke_scheduler_overrides": {"max_steps_per_lane": 1},
    }


def _smoke_ppo(config_path: Path, indices: tuple[int, int]) -> dict[str, Any]:
    import rtu_ppo

    output_root = Path("/tmp/stateful-ppo-smoke-output")
    checkpoint_root = Path("/tmp/stateful-ppo-smoke-checkpoints")
    captured: dict[str, Any] = {}

    original_experiment = rtu_ppo.experiment
    original_train_state = rtu_ppo.TrainState

    class CapturingTrainState:
        @staticmethod
        def create(*args: Any, **kwargs: Any) -> Any:
            state = original_train_state.create(*args, **kwargs)
            captured["primary_parameters"] = _tree_stats(state.params)
            captured["optimizer_state"] = _tree_stats(state.opt_state)
            return state

    def one_transition_experiment(rng: Any, config: Any) -> Any:
        smoke_config = config.replace(
            rollout_steps=1,
            num_mini_batch=1,
            num_updates=1,
            allocate_frames=False,
            video_length=0,
            compute_ntk=False,
            compute_weight_norm=False,
            compute_weight_drift=False,
            compute_plasticity=False,
        )
        return original_experiment(rng, smoke_config)

    rtu_ppo.TrainState = CapturingTrainState
    rtu_ppo.experiment = one_transition_experiment
    original_argv = sys.argv
    sys.argv = [
        str(SOURCE_DIR / "rtu_ppo.py"),
        "--exp",
        str(config_path),
        "--idxs",
        f"{indices[0]}:{indices[-1] + 1}",
        "--save_path",
        str(output_root),
        "--checkpoint_path",
        str(checkpoint_root),
        "--silent",
    ]
    try:
        rtu_ppo.main()
    finally:
        sys.argv = original_argv
        rtu_ppo.experiment = original_experiment
        rtu_ppo.TrainState = original_train_state

    return {
        "lanes": _reward_lanes(output_root, indices, "PPO-family"),
        "agent_class": "algorithms.PPORegistry.getAgent(config.agent)",
        "entrypoint": "src/rtu_ppo.py",
        "primary_parameters": captured["primary_parameters"],
        "optimizer_state": captured["optimizer_state"],
        "smoke_scheduler_overrides": {
            "rollout_steps": 1,
            "num_mini_batch": 1,
            "num_updates": 1,
            "allocate_frames": False,
            "diagnostics": False,
            "reason": (
                "exercise exactly one transition; architecture and optimizer "
                "hyperparameters are unchanged"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--indices", type=int, nargs=2, default=(2_000_001, 2_000_002)
    )
    args = parser.parse_args()

    indices = tuple(args.indices)
    if indices != (2_000_001, 2_000_002):
        raise ValueError("smoke indices must be the protocol's exact ordered seed pair")
    config_path = args.config.resolve(strict=True)
    runtime = _runtime_probe()
    _assert_runtime(runtime)
    _, loaded = _validate_loaded_config(config_path, indices)
    if str(loaded["agent"]).startswith("PPO"):
        transition = _smoke_ppo(config_path, indices)
    else:
        transition = _smoke_dqn(config_path, indices, str(loaded["agent"]))
    payload = {
        "schema_version": "alberta.forager_stateful_one_transition_smoke.v1",
        "status": "passed",
        "runtime": runtime,
        "loaded_config": loaded,
        "transition": transition,
    }
    print("SMOKE_RESULT " + json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
