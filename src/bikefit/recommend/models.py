"""Modèles de sortie du moteur de recommandations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Priorite(StrEnum):
    haute = "haute"
    moyenne = "moyenne"
    faible = "faible"


class Recommendation(BaseModel):
    reglage: str  # ex. "hauteur_selle", "longueur_prolongateurs"
    # "monter", "descendre", "avancer", "reculer", "raccourcir", "allonger", "ajuster",
    # "resserrer", "elargir" — ou "arbitrer" si deux règles se contredisent sur ce réglage
    direction: str
    delta_mm: float | None = None  # amplitude estimée en mm (None si non calculable)
    motif: str  # explication humaine lisible
    priorite: Priorite
    butee_atteinte: bool = False  # True si la plage de réglage est déjà au maximum


class FitReport(BaseModel):
    profil_utilise: str
    recommandations: list[Recommendation]
    angles_hors_cible: list[str]  # noms des angles en dehors de la plage
    angles_ok: list[str]
    # Métriques sans aucun cycle exploitable (point masqué sur toute la vidéo) :
    # ni OK ni hors cible. Défaut [] pour relire les sessions déjà enregistrées.
    angles_non_mesures: list[str] = []
    resume: str
