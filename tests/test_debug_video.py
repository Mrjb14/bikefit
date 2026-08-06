"""Tests du module debug.video — vidéo de debug annotée (squelette + HUD)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from bikefit.debug.video import (
    DebugVideoWriter,
    _format_hud,
    _row_to_points,
    render_debug_video_from_landmarks,
    render_debug_video_live,
)
from bikefit.ingest.reader import VideoReader
from bikefit.kinematics.angles import compute_angles
from bikefit.pose.base import FrameLandmarks, Landmark

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_synthetic_video(
    path: Path, n_frames: int = 5, width: int = 64, height: int = 48
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 40) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _landmark_row(frame_idx: int, timestamp_ms: int, detected: bool) -> dict:
    row: dict = {"frame_idx": frame_idx, "timestamp_ms": timestamp_ms, "detected": detected}
    for i in range(33):
        if detected:
            row[f"lm_{i}_x"] = 0.1 + 0.02 * i
            row[f"lm_{i}_y"] = 0.1 + 0.017 * i
            row[f"lm_{i}_z"] = 0.0
            row[f"lm_{i}_vis"] = 1.0
        else:
            row[f"lm_{i}_x"] = float("nan")
            row[f"lm_{i}_y"] = float("nan")
            row[f"lm_{i}_z"] = float("nan")
            row[f"lm_{i}_vis"] = float("nan")
    return row


class _FakeEstimator:
    """PoseEstimator minimal (duck-typing) — pas de MediaPipe requis pour les tests."""

    def __init__(self, detected_frames: set[int]) -> None:
        self._detected_frames = detected_frames

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, frame_idx: int) -> FrameLandmarks:
        if frame_idx not in self._detected_frames:
            return FrameLandmarks(
                timestamp_ms=timestamp_ms, frame_idx=frame_idx, detected=False, landmarks=[]
            )
        landmarks = [
            Landmark(x=0.1 + 0.02 * i, y=0.1 + 0.017 * i, z=0.0, visibility=1.0) for i in range(33)
        ]
        return FrameLandmarks(
            timestamp_ms=timestamp_ms, frame_idx=frame_idx, detected=True, landmarks=landmarks
        )


# ── _row_to_points ──────────────────────────────────────────────────────────────


class TestRowToPoints:
    def test_all_nan_gives_all_none(self) -> None:
        row = pd.Series(_landmark_row(0, 0, detected=False))
        points = _row_to_points(row, width=100, height=200)
        assert len(points) == 33
        assert all(p is None for p in points)

    def test_valid_row_gives_pixel_coords(self) -> None:
        row = pd.Series(_landmark_row(0, 0, detected=True))
        points = _row_to_points(row, width=100, height=200)
        assert points[0] == (int(0.1 * 100), int(0.1 * 200))
        assert all(p is not None for p in points)


# ── _format_hud ──────────────────────────────────────────────────────────────────


class TestFormatHud:
    def test_numeric_value_formatted(self) -> None:
        row = pd.Series({"genou": 145.234})
        lines = _format_hud(row, [("genou", "Genou", "°")])
        assert lines == ["Genou : 145.2°"]

    def test_nan_value_shows_placeholder(self) -> None:
        row = pd.Series({"genou": float("nan")})
        lines = _format_hud(row, [("genou", "Genou", "°")])
        assert lines == ["Genou : --"]

    def test_missing_column_shows_placeholder(self) -> None:
        row = pd.Series({"autre": 1.0})
        lines = _format_hud(row, [("genou", "Genou", "°")])
        assert lines == ["Genou : --"]


# ── DebugVideoWriter ─────────────────────────────────────────────────────────────


class TestDebugVideoWriter:
    def test_writes_readable_video(self, tmp_path: Path) -> None:
        out = tmp_path / "debug.mp4"
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        with DebugVideoWriter(out, width=64, height=48, fps=10.0) as writer:
            for i in range(3):
                writer.write(
                    frame,
                    points_px=None,
                    hud_lines=[f"x={i}"],
                    detected=True,
                    frame_idx=i,
                    timestamp_ms=i * 100,
                )

        assert out.exists()
        assert out.stat().st_size > 0
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 3

    def test_draws_skeleton_and_hud_without_crashing(self, tmp_path: Path) -> None:
        out = tmp_path / "debug.mp4"
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        points = _row_to_points(pd.Series(_landmark_row(0, 0, detected=True)), width=64, height=48)
        with DebugVideoWriter(out, width=64, height=48, fps=10.0) as writer:
            writer.write(
                frame,
                points_px=points,
                hud_lines=["Genou : 145.0°"],
                detected=True,
                frame_idx=0,
                timestamp_ms=0,
            )
        assert out.stat().st_size > 0


# ── render_debug_video_from_landmarks (fichier) ─────────────────────────────────


class TestRenderDebugVideoFromLandmarks:
    def test_creates_annotated_video(self, tmp_path: Path) -> None:
        video_path = tmp_path / "source.mp4"
        _write_synthetic_video(video_path, n_frames=4, width=64, height=48)

        landmarks_df = pd.DataFrame(
            [
                _landmark_row(0, 0, detected=True),
                _landmark_row(1, 100, detected=False),
                _landmark_row(2, 200, detected=True),
                _landmark_row(3, 300, detected=True),
            ]
        )
        metrics_df = pd.DataFrame(
            {
                "frame_idx": [0, 2, 3],
                "genou": [140.0, 150.0, float("nan")],
            }
        )

        out = tmp_path / "debug.mp4"
        render_debug_video_from_landmarks(
            video_path,
            landmarks_df,
            metrics_df,
            hud_columns=[("genou", "Genou", "°")],
            output_path=out,
            progress=False,
        )

        assert out.exists()
        cap = cv2.VideoCapture(str(out))
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 4  # une frame de sortie par frame source, même non détectée


# ── render_debug_video_live (flux/webcam) ───────────────────────────────────────


class TestRenderDebugVideoLive:
    def test_writes_video_and_returns_landmarks_df(self, tmp_path: Path) -> None:
        video_path = tmp_path / "source.mp4"
        _write_synthetic_video(video_path, n_frames=5, width=64, height=48)
        reader = VideoReader(video_path)
        estimator = _FakeEstimator(detected_frames={0, 2, 3, 4})

        out = tmp_path / "debug_live.mp4"
        metrics_fn = partial(compute_angles, side="right", width=64, height=48)

        df = render_debug_video_live(
            reader,
            estimator,
            compute_metrics=metrics_fn,
            hud_columns=[("genou", "Genou", "°")],
            output_path=out,
            progress=False,
        )

        assert len(df) == 5
        assert out.exists()
        cap = cv2.VideoCapture(str(out))
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 5

    def test_bounded_by_max_duration(self, tmp_path: Path) -> None:
        video_path = tmp_path / "source.mp4"
        _write_synthetic_video(video_path, n_frames=20, width=64, height=48)
        reader = VideoReader(video_path)
        estimator = _FakeEstimator(detected_frames=set(range(20)))

        out = tmp_path / "debug_live.mp4"
        metrics_fn = partial(compute_angles, side="right", width=64, height=48)

        # fichier local : timestamp_ms = frame_idx * 1000 / fps (10 fps)
        # -> borné à 0.25s = frames 0-2
        df = render_debug_video_live(
            reader,
            estimator,
            compute_metrics=metrics_fn,
            hud_columns=[],
            output_path=out,
            max_duration_s=0.25,
            progress=False,
        )
        assert len(df) < 20


@pytest.mark.parametrize("width,height", [(64, 48)])
def test_row_to_points_and_hud_are_pure(width: int, height: int) -> None:
    """Régression : les helpers ne modifient pas leurs arguments d'entrée."""
    row = pd.Series(_landmark_row(0, 0, detected=True))
    row_copy = row.copy()
    _row_to_points(row, width, height)
    pd.testing.assert_series_equal(row, row_copy)
