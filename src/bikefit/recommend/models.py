"""Modèles de sortie du moteur de recommandations."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Priorite(str, Enum):
    haute = "haute"
    moyenne = "moyenne"
    faible = "faible"


class Recommendation(BaseModel):
    reglage: str           # ex. "hauteur_selle", "longueur_prolongateurs"
    direction: str         # "monter", "descendre", "avancer", "reculer", "raccourcir", "allonger"
    delta_mm: float | None = None   # amplitude estimée en mm (None si non calculable)
    motif: str             # explication humaine lisible
    priorite: Priorite
    butee_atteinte: bool = False  # True si la plage de réglage est déjà au maximum


class FitReport(BaseModel):
    profil_utilise: str
    recommandations: list[Recommendation]
    angles_hors_cible: list[str]   # noms des angles en dehors de la plage
    angles_ok: list[str]
    resume: str
