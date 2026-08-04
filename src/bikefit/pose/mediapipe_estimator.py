from __future__ import annotations

import urllib.request
from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from bikefit._paths import MODELS_DIR
from bikefit.pose.base import FrameLandmarks, Landmark

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
MODEL_PATH = MODELS_DIR / "pose_landmarker_heavy.task"


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement du modèle MediaPipe → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé.")
    return MODEL_PATH


class MediaPipePoseEstimator:
    def __init__(self) -> None:
        model_path = _ensure_model()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, frame_idx: int) -> FrameLandmarks:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return FrameLandmarks(
                timestamp_ms=timestamp_ms,
                frame_idx=frame_idx,
                detected=False,
                landmarks=[],
            )

        lms = result.pose_landmarks[0]
        landmarks = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in lms
        ]
        return FrameLandmarks(
            timestamp_ms=timestamp_ms,
            frame_idx=frame_idx,
            detected=True,
            landmarks=landmarks,
        )
