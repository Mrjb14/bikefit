"""Modèles Pydantic pour le profil vélo et les réglages."""
from __future__ import annotations

from pydantic import BaseModel


class BikeSetup(BaseModel):
    """Réglages actuels mesurés en mm/degrés."""
    hauteur_selle_bb: float | None = None      # axe pédalier → sommet selle
    recul_selle: float | None = None           # mm positif = reculé
    hauteur_guidon_bb: float | None = None     # axe pédalier → pad repose-mains
    reach_prolongateurs: float | None = None   # centre bb → extrémité prolongateurs


class AdjustmentRange(BaseModel):
    min: float | None = None
    max: float | None = None


class PlagesReglage(BaseModel):
    hauteur_selle: AdjustmentRange = AdjustmentRange()
    recul_selle: AdjustmentRange = AdjustmentRange()
    reach_prolong: AdjustmentRange = AdjustmentRange()


class BikeModel(BaseModel):
    marque: str
    nom: str
    annee: int
    taille: str
    discipline: str = "contre-la-montre"


class BikeProfile(BaseModel):
    modele: BikeModel
    longueur_manivelle_mm: float = 172.5
    setup_actuel: BikeSetup = BikeSetup()
    plages_reglage: PlagesReglage = PlagesReglage()
