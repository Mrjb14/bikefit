"""Comparaison de positions : surface, CdA estimé, watts et temps, avec significativité.

La première position sert de référence. Un écart n'est déclaré significatif que s'il
dépasse deux fois l'erreur standard combinée des deux moyennes (fenêtres de pédalage
indépendantes) : sous ce seuil, on ne sait pas distinguer un vrai gain du bruit de
mesure — principe n°1, pas de chiffre sans sa fiabilité.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from bikefit.aero.area import FrontalAreaStats
from bikefit.aero.drag import aero_power_w, time_saved_s


class PositionResult(BaseModel):
    name: str
    stats: FrontalAreaStats
    cda_m2: float | None = None  # None sans Cd calibré
    power_w: float | None = None  # puissance aéro à la vitesse de référence
    # Écarts vs la position de référence (None pour la référence elle-même)
    delta_area_pct: float | None = None
    delta_power_w: float | None = None  # à vitesse égale ; None sans Cd
    time_saved_s: float | None = None  # à puissance aéro égale ; None sans Cd
    significant: bool | None = None


def _standard_error(stats: FrontalAreaStats) -> float:
    if stats.n_windows < 2:
        return math.inf
    return stats.std_m2 / math.sqrt(stats.n_windows)


def compare_positions(
    named_stats: list[tuple[str, FrontalAreaStats]],
    cd: float | None,
    speed_kmh: float,
    distance_km: float,
) -> list[PositionResult]:
    """Résultats par position ; la première est la référence des écarts."""
    if not named_stats:
        return []
    results: list[PositionResult] = []
    ref_name, ref = named_stats[0]
    del ref_name
    ref_cda = cd * ref.mean_m2 if cd is not None and ref.is_measured else None

    for i, (name, stats) in enumerate(named_stats):
        cda = cd * stats.mean_m2 if cd is not None and stats.is_measured else None
        res = PositionResult(
            name=name,
            stats=stats,
            cda_m2=round(cda, 4) if cda is not None else None,
            power_w=round(aero_power_w(cda, speed_kmh), 1) if cda is not None else None,
        )
        if i > 0 and stats.is_measured and ref.is_measured:
            delta = stats.mean_m2 - ref.mean_m2
            se = math.hypot(_standard_error(stats), _standard_error(ref))
            res.delta_area_pct = round(100 * delta / ref.mean_m2, 2)
            res.significant = bool(abs(delta) > 2 * se)
            if cda is not None and ref_cda is not None:
                res.delta_power_w = round(
                    aero_power_w(cda, speed_kmh) - aero_power_w(ref_cda, speed_kmh), 1
                )
                res.time_saved_s = round(time_saved_s(ref_cda, cda, distance_km, speed_kmh), 1)
        results.append(res)
    return results


def format_duration(seconds: float) -> str:
    """'+2 min 10 s' / '-45 s' (ASCII : la console Windows cp1252 est la plateforme de dev)."""
    sign = "+" if seconds >= 0 else "-"
    s = round(abs(seconds))
    m, s = divmod(s, 60)
    return f"{sign}{m} min {s:02d} s" if m else f"{sign}{s} s"
