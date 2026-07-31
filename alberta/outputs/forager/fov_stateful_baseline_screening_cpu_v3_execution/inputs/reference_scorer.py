#!/usr/bin/env python3
"""Recompute the frozen FOV screening metric from raw upstream NPZ rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HORIZON = 102_400
EMA_DECAY = 0.999
EMA_INITIAL_VALUE = 0.0
SUBSAMPLE_EVERY = 100
TAIL_FRACTION = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("reward archive must be a regular non-symlink file")
    with np.load(resolved, allow_pickle=False) as archive:
        if "rewards" not in archive.files:
            raise ValueError("reward archive has no 'rewards' array")
        rewards = np.asarray(archive["rewards"])
    if rewards.shape != (HORIZON,):
        raise ValueError(f"raw reward array must have exact shape ({HORIZON},)")
    if rewards.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("raw rewards must have a numeric dtype")
    if not bool(np.all(np.isfinite(rewards))):
        raise ValueError("raw rewards must all be finite")

    rewards64 = rewards.astype(np.float64, copy=False)
    ema = EMA_INITIAL_VALUE
    samples: list[float] = []
    for index, reward in enumerate(rewards64):
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * float(reward)
        if index % SUBSAMPLE_EVERY == 0:
            samples.append(ema)
    expected_samples = (HORIZON + SUBSAMPLE_EVERY - 1) // SUBSAMPLE_EVERY
    if len(samples) != expected_samples:
        raise RuntimeError("internal EMA sample count mismatch")
    tail_start = int((1.0 - TAIL_FRACTION) * len(samples))
    tail = np.asarray(samples[tail_start:], dtype=np.float64)
    if tail.size == 0:
        raise RuntimeError("frozen EMA tail is empty")

    contiguous = np.ascontiguousarray(rewards)
    return {
        "path": resolved.as_posix(),
        "npz_sha256": _sha256(resolved),
        "reward_trace_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "reward_dtype": rewards.dtype.str,
        "reward_shape": list(rewards.shape),
        "reward_sum_float64": float(np.sum(rewards64, dtype=np.float64)),
        "fov_last_10pct_ema_auc": float(np.mean(tail, dtype=np.float64)),
        "final_unadjusted_ema": float(ema),
        "ema_sample_count": len(samples),
        "ema_tail_start_index": tail_start,
        "ema_tail_sample_count": int(tail.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, nargs="+")
    args = parser.parse_args()
    payload = {
        "schema_version": "alberta.forager_stateful_raw_reward_score.v1",
        "metric_contract": {
            "name": "fov_last_10pct_ema_auc",
            "horizon": HORIZON,
            "ema_decay": EMA_DECAY,
            "ema_initial_value": EMA_INITIAL_VALUE,
            "bias_correction": False,
            "subsample_every_steps": SUBSAMPLE_EVERY,
            "subsample_first_reward": True,
            "tail_fraction_of_sampled_curve": TAIL_FRACTION,
            "expected_sample_count": 1024,
            "expected_tail_start_index": 921,
            "expected_tail_sample_count": 103,
        },
        "results": [score(path) for path in args.npz],
        "collector_summaries_used": False,
    }
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
