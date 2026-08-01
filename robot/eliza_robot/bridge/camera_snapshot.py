"""Bounded, non-blocking camera snapshot capture for the bridge server."""

from __future__ import annotations

import asyncio
import base64
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

try:
    from PIL import Image as _PILImage

    _HAS_PIL = True
except ImportError:
    _PILImage = None
    _HAS_PIL = False


CAMERA_SNAPSHOT_TIMEOUT_SEC = 2.0
CAMERA_SNAPSHOT_MAX_WIDTH = 4_096
CAMERA_SNAPSHOT_MAX_HEIGHT = 4_096
CAMERA_SNAPSHOT_MAX_PIXELS = 1_920 * 1_080
CAMERA_SNAPSHOT_MAX_PNG_BYTES = 8 * 1024 * 1024


class CameraSnapshotError(RuntimeError):
    """Base class for a snapshot that cannot be returned safely."""


class CameraSnapshotBusyError(CameraSnapshotError):
    """A prior snapshot worker is still accessing the camera."""


class CameraSnapshotTimeoutError(CameraSnapshotError):
    """Capture or encoding exceeded its wall-clock deadline."""


class CameraSnapshotUnavailableError(CameraSnapshotError):
    """The selected camera did not return a frame."""


@dataclass(frozen=True)
class CameraSnapshotLimits:
    """Hard resource limits applied before a snapshot reaches the wire."""

    timeout_sec: float = CAMERA_SNAPSHOT_TIMEOUT_SEC
    max_width: int = CAMERA_SNAPSHOT_MAX_WIDTH
    max_height: int = CAMERA_SNAPSHOT_MAX_HEIGHT
    max_pixels: int = CAMERA_SNAPSHOT_MAX_PIXELS
    max_png_bytes: int = CAMERA_SNAPSHOT_MAX_PNG_BYTES

    def __post_init__(self) -> None:
        timeout = self.timeout_sec
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise ValueError("camera snapshot timeout_sec must be finite and in (0, 30]")
        try:
            timeout_value = float(timeout)
        except OverflowError as exc:
            raise ValueError(
                "camera snapshot timeout_sec must be finite and in (0, 30]"
            ) from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0.0 or timeout_value > 30.0:
            raise ValueError("camera snapshot timeout_sec must be finite and in (0, 30]")
        for name, value, upper in (
            ("max_width", self.max_width, CAMERA_SNAPSHOT_MAX_WIDTH),
            ("max_height", self.max_height, CAMERA_SNAPSHOT_MAX_HEIGHT),
            ("max_pixels", self.max_pixels, CAMERA_SNAPSHOT_MAX_PIXELS),
            ("max_png_bytes", self.max_png_bytes, CAMERA_SNAPSHOT_MAX_PNG_BYTES),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > upper:
                raise ValueError(f"camera snapshot {name} must be an integer in 1..{upper}")


@dataclass(frozen=True)
class EncodedCameraSnapshot:
    """Validated PNG payload ready for a response envelope."""

    frame_base64: str
    width: int
    height: int
    png_bytes: int


CameraFrame: TypeAlias = npt.NDArray[np.generic]
CameraCapture: TypeAlias = Callable[[], CameraFrame | None]


def _capture_and_encode(
    capture: CameraCapture,
    limits: CameraSnapshotLimits,
) -> EncodedCameraSnapshot:
    """Run the synchronous camera API and encoder inside a worker thread."""
    frame = capture()
    if frame is None:
        raise CameraSnapshotUnavailableError("camera did not return a frame")
    if not isinstance(frame, np.ndarray):
        raise CameraSnapshotError("snapshot frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise CameraSnapshotError(
            f"snapshot frame must be (H,W,3) uint8 RGB; got {frame.shape}"
        )

    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height < 1 or width < 1:
        raise CameraSnapshotError("snapshot frame dimensions must be positive")
    if width > limits.max_width or height > limits.max_height:
        raise CameraSnapshotError(
            "snapshot frame dimensions exceed limit: "
            f"{width}x{height} > {limits.max_width}x{limits.max_height}"
        )
    pixels = width * height
    if pixels > limits.max_pixels:
        raise CameraSnapshotError(
            f"snapshot frame has {pixels} pixels; limit is {limits.max_pixels}"
        )
    if frame.dtype != np.uint8:
        raise CameraSnapshotError(
            f"snapshot frame dtype must be uint8; got {frame.dtype}"
        )
    if not _HAS_PIL or _PILImage is None:
        raise CameraSnapshotError("camera.snapshot requires Pillow")

    buffer = io.BytesIO()
    _PILImage.fromarray(frame).save(buffer, format="PNG")
    png = buffer.getvalue()
    if len(png) > limits.max_png_bytes:
        raise CameraSnapshotError(
            f"encoded snapshot is {len(png)} bytes; limit is {limits.max_png_bytes}"
        )
    return EncodedCameraSnapshot(
        frame_base64=base64.b64encode(png).decode("ascii"),
        width=width,
        height=height,
        png_bytes=len(png),
    )


class CameraSnapshotCoordinator:
    """Run at most one capture/encode job without blocking the event loop.

    ``asyncio`` cannot terminate a Python thread whose device read has hung.
    Timeout and caller cancellation therefore leave the shielded worker
    registered. Further snapshots fail closed as busy until that exact worker
    exits, preventing an orphan from multiplying camera I/O or encoder memory.
    """

    def __init__(self, limits: CameraSnapshotLimits | None = None) -> None:
        if limits is not None and not isinstance(limits, CameraSnapshotLimits):
            raise TypeError("limits must be CameraSnapshotLimits")
        self._limits = limits or CameraSnapshotLimits()
        self._in_flight: asyncio.Task[EncodedCameraSnapshot] | None = None

    @property
    def busy(self) -> bool:
        task = self._in_flight
        return task is not None and not task.done()

    def _worker_done(self, task: asyncio.Task[EncodedCameraSnapshot]) -> None:
        # Retrieve failures even when the request that started the worker has
        # already timed out or disconnected.
        if not task.cancelled():
            _ = task.exception()
        if self._in_flight is task:
            self._in_flight = None

    async def capture(self, capture: CameraCapture) -> EncodedCameraSnapshot:
        current = self._in_flight
        if current is not None and not current.done():
            raise CameraSnapshotBusyError(
                "camera snapshot worker is still busy after an earlier request"
            )

        worker = asyncio.create_task(asyncio.to_thread(_capture_and_encode, capture, self._limits))
        self._in_flight = worker
        worker.add_done_callback(self._worker_done)
        try:
            return await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=float(self._limits.timeout_sec),
            )
        except TimeoutError as exc:
            raise CameraSnapshotTimeoutError(
                f"camera snapshot exceeded {float(self._limits.timeout_sec):g}s timeout"
            ) from exc
