from __future__ import annotations

from pydantic import BaseModel

from bikefit.calibration.models import CalibrationResult


class AngleStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    n_cycles: int
    # Indice de confiance (principe n°1 : toute métrique porte sa fiabilité).
    # visibility : visibilité moyenne du sommet le moins bien vu de l'angle.
    # frames_kept : part des frames réellement exploitées après filtrage —
    # une valeur basse signale une métrique calculée sur une minorité de frames.
    visibility: float | None = None
    frames_kept: float | None = None

    @property
    def is_reliable(self) -> bool:
        """Faux si la mesure repose majoritairement sur des points inférés."""
        if self.visibility is None or self.frames_kept is None:
            return True
        return self.visibility >= 0.5 and self.frames_kept >= 0.5


class FitMetrics(BaseModel):
    genou_extension_pmb: AngleStats  # max genou par cycle (point mort bas)
    genou_flexion_pmh: AngleStats  # min genou par cycle (point mort haut)
    hanche_fermee_pmh: AngleStats  # min hanche par cycle
    tronc: AngleStats  # moyenne tronc par cycle
    coude: AngleStats  # moyenne coude par cycle
    cheville_pmb: AngleStats  # cheville au PMB
    n_cycles_total: int
    n_cycles_used: int
    detection_rate: float
    calibration: CalibrationResult | None = None
    # Plan frontal (vue de face, Phase 3C) — absents si pas de vidéo face analysée
    genou_deviation_gauche: AngleStats | None = None  # angle signé (degrés) par cycle
    genou_deviation_droit: AngleStats | None = None  # angle signé (degrés) par cycle
    largeur_coudes: AngleStats | None = None  # mm si calibration ArUco dispo, sinon None
