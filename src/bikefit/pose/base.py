from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self

import numpy as np
from pydantic import BaseModel


class Landmark(BaseModel):
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1] — y=0 en haut, y=1 en bas (convention OpenCV)
    z: float
    visibility: float

    def pixel_coords(self, width: int, height: int) -> np.ndarray:
        return np.array([self.x * width, self.y * height])


class FrameLandmarks(BaseModel):
    timestamp_ms: int
    frame_idx: int
    detected: bool
    landmarks: list[Landmark]  # 33 landmarks MediaPipe si detected, vide sinon


class PoseEstimator(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def process(
        self, frame_bgr: np.ndarray, timestamp_ms: int, frame_idx: int
    ) -> FrameLandmarks: ...
