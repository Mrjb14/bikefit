"""Modèle Pydantic pour le profil physique du coureur + loader YAML."""

from __future__ import annotations

import contextlib
from datetime import date
from enum import StrEnum

import yaml
from pydantic import BaseModel

from bikefit._paths import CONFIG_DIR


class Flexibilite(StrEnum):
    faible = "faible"
    moyenne = "moyenne"
    bonne = "bonne"
    excellente = "excellente"


class Mensurations(BaseModel):
    taille: int | None = None
    entrejambe: int | None = None
    longueur_femur: int | None = None
    longueur_tibia: int | None = None
    longueur_buste: int | None = None
    longueur_bras: int | None = None
    largeur_epaules: int | None = None
    largeur_hanches: int | None = None
    pointure: int | None = None


class ProfilCoureur(BaseModel):
    poids_kg: float | None = None
    age: int | None = None
    flexibilite: Flexibilite | None = None
    historique_blessure: str | None = None


class ObjectifEpreuve(BaseModel):
    type_epreuve: str | None = None
    distance_km: float | None = None
    duree_effort_min: int | None = None
    echeance: date | None = None


class RiderProfile(BaseModel):
    mensurations: Mensurations = Mensurations()
    profil: ProfilCoureur = ProfilCoureur()
    objectif: ObjectifEpreuve = ObjectifEpreuve()

    def saddle_height_geo_mm(self, longueur_manivelle_mm: float = 172.5) -> float | None:
        """Hauteur selle géométrique (BB → sommet selle) via formule LeMond.

        entrejambe × 0.883, corrigée de l'écart de longueur de manivelle.
        Retourne None si entrejambe non renseigné.
        """
        if self.mensurations.entrejambe is None:
            return None
        base = self.mensurations.entrejambe * 0.883
        # +0.5 mm de hauteur par mm de manivelle plus courte que la référence 172.5
        correction = (172.5 - longueur_manivelle_mm) * 0.5
        return round(base + correction)


def load_rider_profile(name: str = "mensurations") -> RiderProfile | None:
    """Charge config/rider/{name}.yaml. Retourne None si le fichier est absent."""
    path = CONFIG_DIR / "rider" / f"{name}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    with contextlib.suppress(Exception):
        # model_validate : les sous-structures arrivent en dict brut depuis le YAML.
        return RiderProfile.model_validate(
            {
                "mensurations": data.get("mensurations") or {},
                "profil": data.get("profil") or {},
                "objectif": data.get("objectif") or {},
            }
        )
    return None
