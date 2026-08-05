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
