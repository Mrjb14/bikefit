from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TimedFrame:
    timestamp_ms: int
    frame_idx: int
    frame_bgr: np.ndarray


class VideoReader:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Vidéo introuvable : {self._path}")
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {self._path}")
        self._fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def frames(self) -> Iterator[TimedFrame]:
        cap = cv2.VideoCapture(str(self._path))
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield TimedFrame(
                    timestamp_ms=int(frame_idx * 1000 / self._fps),
                    frame_idx=frame_idx,
                    frame_bgr=frame,
                )
                frame_idx += 1
        finally:
            cap.release()
