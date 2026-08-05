"""Tests de calibration ArUco sur frames synthétiques générées par OpenCV."""

from __future__ import annotations

import numpy as np
import pytest

from bikefit.calibration.aruco import calibrate_from_frames, detect_marker_size_px
from bikefit.calibration.models import CalibrationResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_marker_frame(
    marker_side_px: int = 200,
    canvas_size: int = 640,
    marker_id: int = 0,
) -> np.ndarray:
    """Génère une frame BGR avec un marqueur ArUco centré sur fond blanc."""
    import cv2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_side_px)

    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    offset = (canvas_size - marker_side_px) // 2
    canvas[offset : offset + marker_side_px, offset : offset + marker_side_px] = marker_img

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _make_blank_frame(canvas_size: int = 640) -> np.ndarray:
    return np.full((canvas_size, canvas_size, 3), 200, dtype=np.uint8)


# ── Tests detect_marker_size_px ───────────────────────────────────────────────


class TestDetectMarkerSizePx:
    def test_detects_marker_returns_approx_size(self) -> None:
        frame = _make_marker_frame(marker_side_px=200)
        size = detect_marker_size_px(frame)
        assert size is not None
        # La taille détectée doit être proche de 200 px (± 5 %)
        assert abs(size - 200) < 10

    def test_no_marker_returns_none(self) -> None:
        frame = _make_blank_frame()
        assert detect_marker_size_px(frame) is None

    def test_small_marker_detected(self) -> None:
        frame = _make_marker_frame(marker_side_px=80)
        size = detect_marker_size_px(frame)
        assert size is not None
        assert abs(size - 80) < 6

    def test_different_marker_ids_detected(self) -> None:
        for mid in (0, 5, 10, 20):
            frame = _make_marker_frame(marker_id=mid)
            assert detect_marker_size_px(frame) is not None


# ── Tests calibrate_from_frames ───────────────────────────────────────────────


class TestCalibrateFromFrames:
    def test_basic_calibration(self) -> None:
        # Marqueur de 200 px dans l'image représente 100 mm réels
        # -> mm_per_pixel = 100 / 200 = 0.5
        frames = [_make_marker_frame(marker_side_px=200)] * 5
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        assert pytest.approx(result.mm_per_pixel, rel=0.05) == 0.5

    def test_returns_none_when_no_marker(self) -> None:
        frames = [_make_blank_frame()] * 10
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is None

    def test_partial_detection(self) -> None:
        # 5 frames avec marqueur + 5 sans
        frames = [_make_marker_frame()] * 5 + [_make_blank_frame()] * 5
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        assert result.n_detected == 5
        assert result.n_sampled == 10

    def test_sample_every_reduces_sampled_count(self) -> None:
        frames = [_make_marker_frame()] * 20
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=5)
        assert result is not None
        # 20 frames / 5 = 4 frames échantillonnées
        assert result.n_sampled == 4

    def test_median_robust_to_outlier(self) -> None:
        # 9 frames avec marqueur 200 px + 1 frame avec marqueur 100 px (outlier)
        good = [_make_marker_frame(marker_side_px=200)] * 9
        outlier = [_make_marker_frame(marker_side_px=100)]
        result = calibrate_from_frames(good + outlier, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        # La médiane doit coller au marqueur 200 px, pas à l'outlier
        assert pytest.approx(result.mm_per_pixel, rel=0.05) == 0.5

    def test_detection_rate_property(self) -> None:
        frames = [_make_marker_frame()] * 6 + [_make_blank_frame()] * 4
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        assert pytest.approx(result.detection_rate) == 0.6

    def test_is_reliable_enough_detections(self) -> None:
        frames = [_make_marker_frame()] * 10
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        assert result.is_reliable

    def test_is_not_reliable_too_few(self) -> None:
        # Seulement 2 détections sur 20 frames → < 50 % et < 5
        frames = [_make_marker_frame()] * 2 + [_make_blank_frame()] * 18
        result = calibrate_from_frames(frames, marker_size_mm=100.0, sample_every=1)
        assert result is not None
        assert not result.is_reliable


# ── Tests CalibrationResult ───────────────────────────────────────────────────


class TestCalibrationResult:
    def test_pydantic_roundtrip(self) -> None:
        r = CalibrationResult(
            mm_per_pixel=0.5,
            marker_size_mm=100.0,
            n_detected=8,
            n_sampled=10,
        )
        assert CalibrationResult(**r.model_dump()) == r

    def test_stored_in_fit_metrics(self) -> None:
        from bikefit.kinematics.metrics import AngleStats, FitMetrics

        stats = AngleStats(mean=150.0, std=1.0, min=145.0, max=155.0, n_cycles=10)
        calib = CalibrationResult(
            mm_per_pixel=0.42, marker_size_mm=100.0, n_detected=7, n_sampled=10
        )
        metrics = FitMetrics(
            genou_extension_pmb=stats,
            genou_flexion_pmh=stats,
            hanche_fermee_pmh=stats,
            tronc=stats,
            coude=stats,
            cheville_pmb=stats,
            n_cycles_total=10,
            n_cycles_used=10,
            detection_rate=1.0,
            calibration=calib,
        )
        assert metrics.calibration is not None
        assert metrics.calibration.mm_per_pixel == pytest.approx(0.42)
