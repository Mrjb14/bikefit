"""Modèles de calibration ArUco."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationResult(BaseModel):
    mm_per_pixel: float
    marker_size_mm: float
    n_detected: int = Field(description="Frames où le marqueur a été détecté")
    n_sampled: int = Field(description="Frames analysées au total")

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_sampled if self.n_sampled > 0 else 0.0

    @property
    def is_reliable(self) -> bool:
        """Calibration fiable si détecté sur >= 50 % des frames échantillonnées."""
        return self.detection_rate >= 0.5 and self.n_detected >= 5


class MarkerPosition(BaseModel):
    """Position pixel (médiane sur plusieurs frames) d'un marqueur ArUco fixe.

    Utilisé pour localiser un marqueur collé sur un point fixe du vélo (ex. le
    boîtier de pédalier) plutôt qu'une taille — la caméra et le vélo étant fixes
    l'un par rapport à l'autre pendant la capture, sa position pixel est stable.
    """

    x_px: float
    y_px: float
    n_detected: int = Field(description="Frames où le marqueur a été détecté")
    n_sampled: int = Field(description="Frames analysées au total")

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_sampled if self.n_sampled > 0 else 0.0

    @property
    def is_reliable(self) -> bool:
        """Position fiable si détectée sur >= 50 % des frames échantillonnées."""
        return self.detection_rate >= 0.5 and self.n_detected >= 5
