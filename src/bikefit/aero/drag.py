"""De la surface frontale à la traînée : Cd calibré sur route, CdA, puissance.

La Kinect mesure A (surface frontale). La traînée dépend de CdA = Cd·A, et Cd (la
« forme » : tête, épaules, casque, tissu) est invisible pour une caméra. Il se
calibre une fois sur le terrain : on mesure le vrai CdA d'une position (méthode
Chung / élévation virtuelle, capteur de puissance, route plate sans vent), on mesure
A de cette même position à la Kinect, et Cd = CdA_terrain / A_kinect.

Le Cd ainsi obtenu absorbe aussi les biais constants du montage (vélo, home trainer
dans le champ, intrinsèques approximatives) : il n'est valable que pour CE montage
de capture. Il ne transfère pas un CdA « absolu » d'une installation à l'autre.

Même sans Cd, la comparaison de deux positions reste utile : à Cd égal, l'écart
relatif de A est l'écart relatif de traînée. C'est une hypothèse (une tête mieux
rentrée peut gagner en Cd sans réduire A), signalée comme telle.
"""

from __future__ import annotations

# Masse volumique de l'air au niveau de la mer à ~15 °C (kg/m³).
RHO_SEA_LEVEL = 1.225

# Plage de Cd plausible pour un cycliste en position CLM/triathlon (vélo inclus).
# Hors de cette plage, la calibration est probablement fausse (A mal segmentée,
# CdA terrain bruité) : on prévient sans bloquer.
CD_PLAUSIBLE_RANGE = (0.45, 1.10)


def calibrate_cd(cda_terrain_m2: float, area_m2: float) -> float:
    """Cd = CdA mesuré sur route / surface frontale Kinect de la même position."""
    if area_m2 <= 0:
        raise ValueError("Surface frontale nulle : calibration impossible.")
    if cda_terrain_m2 <= 0:
        raise ValueError("CdA terrain doit être strictement positif.")
    return cda_terrain_m2 / area_m2


def cd_is_plausible(cd: float) -> bool:
    lo, hi = CD_PLAUSIBLE_RANGE
    return lo <= cd <= hi


def aero_power_w(cda_m2: float, speed_kmh: float, rho: float = RHO_SEA_LEVEL) -> float:
    """Puissance absorbée par la traînée à vitesse constante, sans vent : ½·ρ·CdA·v³."""
    v = speed_kmh / 3.6
    return 0.5 * rho * cda_m2 * v**3


def time_saved_s(
    cda_ref_m2: float,
    cda_new_m2: float,
    distance_km: float,
    speed_kmh: float,
) -> float:
    """Temps gagné (s, négatif = perdu) à puissance aéro constante sur `distance_km`.

    À puissance aéro égale, v ∝ CdA^(-1/3). Néglige la résistance au roulement, ce qui
    surestime légèrement le gain (de l'ordre de 10-20 % à ces vitesses) : un ordre de
    grandeur, pas une prédiction de chrono.
    """
    if cda_ref_m2 <= 0 or cda_new_m2 <= 0 or speed_kmh <= 0:
        raise ValueError("CdA et vitesse doivent être strictement positifs.")
    v_ref = speed_kmh / 3.6
    v_new = v_ref * (cda_ref_m2 / cda_new_m2) ** (1 / 3)
    d = distance_km * 1000
    return d / v_ref - d / v_new
