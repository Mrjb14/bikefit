"""Vidéo de contrôle de la segmentation : profondeur en fausses couleurs + silhouette.

La segmentation est le maillon faible de la mesure (sol, roue, mur, trous IR) : il faut
pouvoir *voir* ce qui a été compté avant de croire un chiffre. Vert = silhouette
mesurée, rouge = trous bouchés, gris = hors silhouette.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Plage d'affichage de la profondeur (mm) : la Kinect v1 mesure de ~0,8 à ~4 m.
_DISPLAY_RANGE_MM = (500.0, 4500.0)


def render_segmentation_frame(
    depth_mm: np.ndarray,
    silhouette: np.ndarray | None,
    holes: np.ndarray | None,
    area_m2: float,
) -> np.ndarray:
    import cv2

    lo, hi = _DISPLAY_RANGE_MM
    norm = np.clip((depth_mm.astype(np.float64) - lo) / (hi - lo), 0, 1)
    gray = (255 * (1 - norm)).astype(np.uint8)
    gray[depth_mm == 0] = 0
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if silhouette is not None:
        overlay = img.copy()
        overlay[silhouette] = (60, 200, 60)
        if holes is not None:
            overlay[holes] = (40, 40, 220)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    label = f"A = {area_m2:.3f} m2" if not np.isnan(area_m2) else "segmentation echouee"
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


class SegmentationVideoWriter:
    """Hook `on_frame` d'analyze_frontal_area -> MP4 (ouvert à la première frame)."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: object | None = None

    def __call__(
        self,
        idx: int,
        depth_mm: np.ndarray,
        silhouette: np.ndarray | None,
        holes: np.ndarray | None,
        area_m2: float,
    ) -> None:
        import cv2

        frame = render_segmentation_frame(depth_mm, silhouette, holes, area_m2)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (w, h)
            )
        self._writer.write(frame)  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()  # type: ignore[attr-defined]
            self._writer = None
