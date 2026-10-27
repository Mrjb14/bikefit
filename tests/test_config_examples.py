"""Les fiches d'exemple versionnées doivent rester chargeables par les modèles."""

from pathlib import Path

import yaml

from bikefit.equipment.models import BikeProfile
from bikefit.equipment.rider import RiderProfile

EXAMPLES = Path(__file__).resolve().parent.parent / "config" / "examples"


def _load(name: str) -> dict:
    return yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))


def test_bike_example_valide() -> None:
    data = _load("giant_trinity_2019.example.yaml")
    bike = BikeProfile.model_validate(
        {
            "modele": data["modele"],
            "setup_actuel": data.get("setup_actuel") or {},
            "plages_reglage": data.get("plages_reglage") or {},
        }
    )
    assert bike.modele.marque == "Giant"
    assert data["transmission"]["pedalier"]["longueur_manivelles"] is None


def test_rider_example_valide() -> None:
    data = _load("mensurations.example.yaml")
    rider = RiderProfile.model_validate(
        {
            "mensurations": data["mensurations"],
            "profil": data["profil"],
            "objectif": data["objectif"],
        }
    )
    assert rider.mensurations.entrejambe is None
