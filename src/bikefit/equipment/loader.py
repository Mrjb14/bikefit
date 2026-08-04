"""Chargement d'un profil vélo depuis un fichier YAML."""
from __future__ import annotations

from pathlib import Path

import yaml

from bikefit._paths import CONFIG_DIR
from bikefit.equipment.models import BikeProfile


def load_bike_profile(name: str) -> BikeProfile:
    """Charge config/bikes/{name}.yaml et retourne un BikeProfile."""
    path = CONFIG_DIR / "bikes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Fiche vélo introuvable : {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Extrait longueur manivelle depuis la transmission si présente
    manivelle = 172.5
    try:
        manivelle = float(data["transmission"]["pedalier"]["longueur_manivelles"])
    except (KeyError, TypeError):
        pass

    return BikeProfile(
        modele=data["modele"],
        longueur_manivelle_mm=manivelle,
        setup_actuel=data.get("setup_actuel") or {},
        plages_reglage=data.get("plages_reglage") or {},
    )
