"""Vérification des intrinsèques (fx) avec le marqueur ArUco du projet.

La surface frontale varie en 1/(fx·fy) : 2 % d'erreur sur fx = 4 % sur l'aire. Les
intrinsèques Kinect « d'usine » (`KINECT_V1_REGISTERED`) sont une approximation.

Contrôle sans damier : le marqueur ID=0 (100 mm) tenu face à la caméra donne à la
fois sa taille en pixels (image couleur) et sa distance (profondeur alignée). Par le
modèle pinhole, fx ≈ taille_px · z / taille_mm. Une médiane sur plusieurs frames
donne un fx estimé à comparer à celui de l'enregistrement.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from bikefit.aero.recording import DepthRecording

# Au-delà de cet écart relatif sur fx, les intrinsèques sont à corriger.
MAX_FX_ERROR = 0.02


class IntrinsicsCheck(BaseModel):
    fx_recording: float
    fx_estimated: float
    n_frames: int  # frames où marqueur ET profondeur ont été mesurés

    @property
    def relative_error(self) -> float:
        return (self.fx_estimated - self.fx_recording) / self.fx_recording

    @property
    def area_error(self) -> float:
        """Erreur relative de la surface calculée avec fx_recording, vs la vraie surface.

        A_calculée ∝ 1/fx_recording², A_vraie ∝ 1/fx_estimé² : un fx d'enregistrement
        trop grand sous-estime la surface (erreur négative), ≈ 2× l'erreur sur fx.
        """
        return (self.fx_estimated / self.fx_recording) ** 2 - 1

    @property
    def is_ok(self) -> bool:
        return abs(self.relative_error) <= MAX_FX_ERROR


def estimate_fx_from_marker(
    frame_bgr: np.ndarray,
    depth_mm: np.ndarray,
    marker_size_mm: float,
    marker_id: int = 0,
) -> float | None:
    """fx estimé sur une frame : taille_px · z / taille_mm (None si non mesurable).

    z = médiane de la profondeur sur le centre du marqueur (moitié de son côté), pour
    éviter les pixels de bord où la profondeur saute vers le fond.
    """
    from bikefit.calibration.aruco import detect_marker_center_px, detect_marker_size_px

    size_px = detect_marker_size_px(frame_bgr, marker_id=marker_id)
    center = detect_marker_center_px(frame_bgr, marker_id)
    if size_px is None or center is None:
        return None
    half = max(1, int(size_px / 4))
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = depth_mm.shape
    patch = depth_mm[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]
    vals = patch[patch > 0]
    if vals.size < 4:
        return None
    return float(size_px * np.median(vals) / marker_size_mm)


def check_intrinsics_with_marker(
    recording: DepthRecording,
    marker_size_mm: float,
    sample_every: int = 10,
) -> IntrinsicsCheck | None:
    """Compare le fx de l'enregistrement au fx mesuré sur le marqueur (rgb.mp4 requis)."""
    import cv2

    rgb = recording.rgb_path
    if rgb is None:
        return None
    estimates: list[float] = []
    cap = cv2.VideoCapture(str(rgb))
    try:
        idx = 0
        while idx < recording.frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                fx = estimate_fx_from_marker(frame, recording.read(idx), marker_size_mm)
                if fx is not None:
                    estimates.append(fx)
            idx += 1
    finally:
        cap.release()
    if not estimates:
        return None
    return IntrinsicsCheck(
        fx_recording=recording.intrinsics.fx,
        fx_estimated=round(float(np.median(estimates)), 1),
        n_frames=len(estimates),
    )
