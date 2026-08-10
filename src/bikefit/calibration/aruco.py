"""Calibration mm/pixel par détection de marqueur ArUco."""

from __future__ import annotations

import numpy as np

from bikefit.calibration.models import CalibrationResult, MarkerPosition

# Dictionnaire par défaut : DICT_4X4_50 — simple, peu de faux positifs
_DEFAULT_DICT_ID = None  # résolu à l'import pour éviter l'import cv2 au niveau module


def _aruco_dict():
    import cv2

    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _detector():
    import cv2

    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(_aruco_dict(), params)


def _find_corners(frame_bgr: np.ndarray, marker_id: int | None) -> np.ndarray | None:
    """Coins (4, 2) du marqueur demandé, ou du premier détecté si `marker_id` est None.

    Nécessaire dès que plusieurs marqueurs cohabitent dans le champ (ID=0 pour la
    calibration mm/px, ID=1 sur le BB) : sans filtrage par ID, le premier marqueur
    renvoyé par OpenCV est arbitraire (ordre de détection, pas l'ID).
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)

    if ids is None or len(corners) == 0:
        return None

    if marker_id is None:
        return corners[0][0].astype(np.float32)  # type: ignore[no-any-return]

    matches = np.flatnonzero(ids.flatten() == marker_id)
    if matches.size == 0:
        return None
    return corners[matches[0]][0].astype(np.float32)  # type: ignore[no-any-return]


def detect_marker_size_px(frame_bgr: np.ndarray, marker_id: int | None = None) -> float | None:
    """Détecte un marqueur ArUco dans une frame et retourne sa taille en pixels.

    `marker_id=None` (défaut) : premier marqueur détecté, quel que soit son ID.
    `marker_id=<int>` : ignore tout marqueur dont l'ID ne correspond pas.

    La taille est définie comme la moyenne des longueurs des 4 côtés du marqueur
    (robuste aux légères perspectives).
    Retourne None si aucun marqueur (correspondant) n'est détecté.
    """
    pts = _find_corners(frame_bgr, marker_id)
    if pts is None:
        return None
    sides = [
        np.linalg.norm(pts[1] - pts[0]),  # haut
        np.linalg.norm(pts[2] - pts[1]),  # droite
        np.linalg.norm(pts[3] - pts[2]),  # bas
        np.linalg.norm(pts[0] - pts[3]),  # gauche
    ]
    return float(np.mean(sides))


def detect_marker_center_px(frame_bgr: np.ndarray, marker_id: int) -> tuple[float, float] | None:
    """Centre pixel (moyenne des 4 coins) du marqueur `marker_id` dans une frame.

    Retourne None si ce marqueur n'est pas détecté dans la frame.
    """
    pts = _find_corners(frame_bgr, marker_id)
    if pts is None:
        return None
    center = pts.mean(axis=0)
    return float(center[0]), float(center[1])


def calibrate_from_frames(
    frames_bgr: list[np.ndarray],
    marker_size_mm: float,
    sample_every: int = 10,
    marker_id: int | None = 0,
) -> CalibrationResult | None:
    """Calcule mm/pixel à partir d'une liste de frames (usage tests/spike).

    `marker_id=0` par défaut : le marqueur de calibration a toujours l'ID=0
    (convention du projet, cf. CLAUDE.md) — évite de calibrer par erreur sur un
    autre marqueur (ex. ID=1 sur le BB) présent dans le même champ.

    Pour les vidéos réelles, préférer `calibrate_from_video` qui lit le fichier
    sans charger toutes les frames en RAM.
    """
    sampled = frames_bgr[::sample_every]
    sizes_px: list[float] = []

    for frame in sampled:
        size = detect_marker_size_px(frame, marker_id=marker_id)
        if size is not None:
            sizes_px.append(size)

    return _build_result(sizes_px, len(sampled), marker_size_mm)


def calibrate_from_video(
    video_path: str,
    marker_size_mm: float,
    sample_every: int = 10,
    marker_id: int | None = 0,
) -> CalibrationResult | None:
    """Calcule mm/pixel directement depuis un fichier vidéo.

    Lit une frame toutes les `sample_every` frames sans tout charger en RAM.
    Prend la médiane des tailles détectées pour robustesse aux occlusions.

    Args:
        video_path: chemin vers le fichier vidéo.
        marker_size_mm: taille réelle du côté du marqueur imprimé, en mm.
        sample_every: intervalle d'échantillonnage.
        marker_id: ID du marqueur de calibration (0 par défaut, cf. convention
            du projet). Évite de calibrer sur un autre marqueur présent dans le
            champ (ex. ID=1 sur le BB, Phase 3A+).

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
                size = detect_marker_size_px(frame, marker_id=marker_id)
                if size is not None:
                    sizes_px.append(size)
            frame_idx += 1
    finally:
        cap.release()

    return _build_result(sizes_px, n_sampled, marker_size_mm)


def locate_marker_in_video(
    video_path: str,
    marker_id: int,
    sample_every: int = 10,
) -> MarkerPosition | None:
    """Localise un marqueur ArUco fixe (ex. collé sur le BB) dans une vidéo.

    Le vélo et la caméra étant fixes l'un par rapport à l'autre pendant une
    capture home trainer, la position pixel du marqueur est stable d'une frame
    à l'autre — on en prend la médiane par axe pour la robustesse aux occlusions
    (pied/manivelle passant devant), comme `calibrate_from_video` pour la taille.

    Retourne None si le marqueur n'est détecté sur aucune frame échantillonnée.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    xs: list[float] = []
    ys: list[float] = []
    n_sampled = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                n_sampled += 1
                center = detect_marker_center_px(frame, marker_id)
                if center is not None:
                    xs.append(center[0])
                    ys.append(center[1])
            frame_idx += 1
    finally:
        cap.release()

    if not xs:
        return None
    return MarkerPosition(
        x_px=float(np.median(xs)),
        y_px=float(np.median(ys)),
        n_detected=len(xs),
        n_sampled=n_sampled,
    )


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
