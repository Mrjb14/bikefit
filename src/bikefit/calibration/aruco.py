"""Calibration mm/pixel par détection de marqueur ArUco."""

from __future__ import annotations

import numpy as np

from bikefit.calibration.models import CalibrationResult

# Dictionnaire par défaut : DICT_4X4_50 — simple, peu de faux positifs
_DEFAULT_DICT_ID = None  # résolu à l'import pour éviter l'import cv2 au niveau module


def _aruco_dict():
    import cv2

    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _detector():
    import cv2

    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(_aruco_dict(), params)


def detect_marker_size_px(frame_bgr: np.ndarray) -> float | None:
    """Détecte le premier marqueur ArUco dans une frame et retourne sa taille en pixels.

    La taille est définie comme la moyenne des longueurs des 4 côtés du marqueur
    (robuste aux légères perspectives).
    Retourne None si aucun marqueur n'est détecté.
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)

    if ids is None or len(corners) == 0:
        return None

    # Premier marqueur détecté — 4 coins ordonnés (TL, TR, BR, BL)
    pts = corners[0][0].astype(np.float32)  # shape (4, 2)
    sides = [
        np.linalg.norm(pts[1] - pts[0]),  # haut
        np.linalg.norm(pts[2] - pts[1]),  # droite
        np.linalg.norm(pts[3] - pts[2]),  # bas
        np.linalg.norm(pts[0] - pts[3]),  # gauche
    ]
    return float(np.mean(sides))


def calibrate_from_frames(
    frames_bgr: list[np.ndarray],
    marker_size_mm: float,
    sample_every: int = 10,
) -> CalibrationResult | None:
    """Calcule mm/pixel à partir d'une liste de frames (usage tests/spike).

    Pour les vidéos réelles, préférer `calibrate_from_video` qui lit le fichier
    sans charger toutes les frames en RAM.
    """
    sampled = frames_bgr[::sample_every]
    sizes_px: list[float] = []

    for frame in sampled:
        size = detect_marker_size_px(frame)
        if size is not None:
            sizes_px.append(size)

    return _build_result(sizes_px, len(sampled), marker_size_mm)


def calibrate_from_video(
    video_path: str,
    marker_size_mm: float,
    sample_every: int = 10,
) -> CalibrationResult | None:
    """Calcule mm/pixel directement depuis un fichier vidéo.

    Lit une frame toutes les `sample_every` frames sans tout charger en RAM.
    Prend la médiane des tailles détectées pour robustesse aux occlusions.

    Args:
        video_path: chemin vers le fichier vidéo.
        marker_size_mm: taille réelle du côté du marqueur imprimé, en mm.
        sample_every: intervalle d'échantillonnage.

    Returns:
        CalibrationResult ou None si aucun marqueur n'est détecté.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    sizes_px: list[float] = []
    n_sampled = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                n_sampled += 1
                size = detect_marker_size_px(frame)
                if size is not None:
                    sizes_px.append(size)
            frame_idx += 1
    finally:
        cap.release()

    return _build_result(sizes_px, n_sampled, marker_size_mm)


def _build_result(
    sizes_px: list[float],
    n_sampled: int,
    marker_size_mm: float,
) -> CalibrationResult | None:
    if not sizes_px:
        return None
    median_size_px = float(np.median(sizes_px))
    return CalibrationResult(
        mm_per_pixel=marker_size_mm / median_size_px,
        marker_size_mm=marker_size_mm,
        n_detected=len(sizes_px),
        n_sampled=n_sampled,
    )
