"""Tests de la géométrie d'estimation des mensurations corporelles.

Garde-fou principal : la convention d'axe y. Une inversion de signe ici produit
des longueurs négatives et désactive silencieusement la mise à l'échelle par la
taille — c'est exactement le bug qui a produit un entrejambe de -786 mm.
"""

import pytest

from bikefit.measure.body import _NOSE_ANKLE_RATIO, measurements_from_world_landmarks


class _LM:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _standing_person(
    femur_m: float = 0.40,
    tibia_m: float = 0.40,
    buste_m: float = 0.50,
    epaules_m: float = 0.40,
    hanches_m: float = 0.28,
    nose_above_shoulder_m: float = 0.25,
) -> list[_LM]:
    """Squelette debout, convention y=bas, origine au centre des hanches.

    Sous le bassin y augmente (genou puis cheville), au-dessus y diminue
    (épaules puis nez).
    """
    lm = [_LM(0.0, 0.0) for _ in range(33)]
    hip_y = 0.0
    knee_y = hip_y + femur_m
    ankle_y = knee_y + tibia_m
    shoulder_y = hip_y - buste_m
    nose_y = shoulder_y - nose_above_shoulder_m

    lm[0] = _LM(0.0, nose_y)
    lm[11] = _LM(-epaules_m / 2, shoulder_y)  # épaule gauche
    lm[12] = _LM(+epaules_m / 2, shoulder_y)  # épaule droite
    lm[23] = _LM(-hanches_m / 2, hip_y)  # hanche gauche
    lm[24] = _LM(+hanches_m / 2, hip_y)  # hanche droite
    lm[25] = _LM(-hanches_m / 2, knee_y)  # genou gauche
    lm[26] = _LM(+hanches_m / 2, knee_y)  # genou droit
    lm[27] = _LM(-hanches_m / 2, ankle_y)  # cheville gauche
    lm[28] = _LM(+hanches_m / 2, ankle_y)  # cheville droite
    return lm


class TestSignConvention:
    """Le bug historique : toutes les longueurs verticales doivent être > 0."""

    def test_all_lengths_are_positive(self) -> None:
        out = measurements_from_world_landmarks(_standing_person())
        for key, value in out.items():
            assert value > 0, f"{key} devrait être positif, vaut {value}"

    def test_inseam_is_positive_with_scaling(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(), taille_mm=1790)
        assert out["entrejambe"] > 0


class TestUnscaledMeasurements:
    def test_femur_matches_input(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(femur_m=0.42))
        assert out["longueur_femur"] == pytest.approx(420, abs=1)

    def test_tibia_matches_input(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(tibia_m=0.38))
        assert out["longueur_tibia"] == pytest.approx(380, abs=1)

    def test_inseam_is_femur_plus_tibia(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(femur_m=0.40, tibia_m=0.39))
        assert out["entrejambe"] == pytest.approx(790, abs=1)
        assert out["entrejambe"] == pytest.approx(
            out["longueur_femur"] + out["longueur_tibia"], abs=1
        )

    def test_torso_matches_input(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(buste_m=0.52))
        assert out["longueur_buste"] == pytest.approx(520, abs=1)

    def test_widths_match_input(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(epaules_m=0.41, hanches_m=0.29))
        assert out["largeur_epaules"] == pytest.approx(410, abs=1)
        assert out["largeur_hanches"] == pytest.approx(290, abs=1)


class TestHeightScaling:
    def test_scaling_is_applied(self) -> None:
        """Une personne mesurée 2x trop petite doit être remise à l'échelle."""
        half = _standing_person(
            femur_m=0.20, tibia_m=0.20, buste_m=0.25, nose_above_shoulder_m=0.125
        )
        full = _standing_person(
            femur_m=0.40, tibia_m=0.40, buste_m=0.50, nose_above_shoulder_m=0.25
        )

        out_half = measurements_from_world_landmarks(half, taille_mm=1790)
        out_full = measurements_from_world_landmarks(full, taille_mm=1790)

        # Après mise à l'échelle, les deux doivent converger vers la même personne
        assert out_half["entrejambe"] == pytest.approx(out_full["entrejambe"], rel=0.01)

    def test_scale_factor_uses_nose_ankle_ratio(self) -> None:
        # nez→cheville = 0.25 + 0.50 + 0.40 + 0.40 = 1.55 m
        lm = _standing_person()
        out = measurements_from_world_landmarks(lm, taille_mm=1790)
        expected_scale = (1790 / 1000 * _NOSE_ANKLE_RATIO) / 1.55
        assert out["entrejambe"] == pytest.approx(800 * expected_scale, abs=2)

    def test_no_scaling_without_height(self) -> None:
        lm = _standing_person(femur_m=0.40, tibia_m=0.40)
        out = measurements_from_world_landmarks(lm, taille_mm=None)
        assert out["entrejambe"] == pytest.approx(800, abs=1)

    def test_degenerate_pose_skips_scaling(self) -> None:
        """Nez et cheville quasi confondus (pose aberrante) : pas de division folle."""
        lm = [_LM(0.0, 0.0) for _ in range(33)]
        lm[27] = _LM(0.0, 0.05)
        lm[28] = _LM(0.0, 0.05)
        out = measurements_from_world_landmarks(lm, taille_mm=1790)
        # scale reste à 1.0 → valeurs brutes, pas d'explosion numérique
        assert out["entrejambe"] == pytest.approx(50, abs=1)


class TestPlausibility:
    def test_realistic_person_gives_realistic_ratios(self) -> None:
        out = measurements_from_world_landmarks(_standing_person(), taille_mm=1790)
        ratio = out["entrejambe"] / 1790
        assert 0.40 < ratio < 0.60
