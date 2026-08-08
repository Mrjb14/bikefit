"""Fiabilité des landmarks : MediaPipe fournit un score `visibility` par point.

Ce score distingue un point réellement observé d'un point *inféré* par le modèle.
Sur une vue de profil, le côté opposé à la caméra tombe typiquement à 0.2-0.3 : le
squelette reste plausible, mais les coordonnées sont devinées. Les utiliser dans un
angle produit une valeur d'apparence normale, sans le moindre signal d'alerte —
contraire au principe « toute métrique porte un indice de confiance ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Seuil MediaPipe usuel. En dessous, le point est considéré comme inféré et non
# observé : il est exclu des angles plutôt que de les fausser silencieusement.
VISIBILITY_THRESHOLD = 0.5


def landmark_visibility(df: pd.DataFrame, lm_idx: int) -> np.ndarray:
    """Colonne de visibilité d'un landmark.

    Renvoie des 1.0 si la colonne est absente : les DataFrames synthétiques des tests
    et d'éventuelles sources sans score restent traités comme pleinement visibles,
    donc le comportement historique est préservé.
    """
    col = f"lm_{lm_idx}_vis"
    if col not in df.columns:
        return np.ones(len(df), dtype=float)
    return df[col].to_numpy(dtype=float)


def reliability_mask(
    df: pd.DataFrame,
    lm_indices: tuple[int, ...],
    threshold: float = VISIBILITY_THRESHOLD,
) -> np.ndarray:
    """True là où TOUS les landmarks donnés atteignent le seuil.

    Un angle vaut par son point le plus faible : si un seul de ses trois sommets est
    deviné, l'angle entier est douteux. Les NaN comptent comme non fiables.
    """
    vis = np.stack([landmark_visibility(df, i) for i in lm_indices], axis=1)
    with np.errstate(invalid="ignore"):
        return np.all(np.nan_to_num(vis, nan=0.0) >= threshold, axis=1)


def weakest_visibility(df: pd.DataFrame, lm_indices: tuple[int, ...]) -> np.ndarray:
    """Visibilité du sommet le moins bien vu, frame par frame."""
    vis = np.stack([landmark_visibility(df, i) for i in lm_indices], axis=1)
    with np.errstate(invalid="ignore"):
        return np.nanmin(vis, axis=1)


def mean_visibility(
    df: pd.DataFrame,
    lm_indices: tuple[int, ...],
) -> float:
    """Visibilité moyenne du maillon le plus faible, sur l'ensemble des frames.

    C'est le minimum par frame qui est moyenné (et non la moyenne des points) :
    un angle n'est pas plus fiable que son sommet le moins bien vu.
    """
    vis = np.stack([landmark_visibility(df, i) for i in lm_indices], axis=1)
    weakest = np.nanmin(vis, axis=1)
    if len(weakest) == 0 or np.all(np.isnan(weakest)):
        return 0.0
    return float(np.nanmean(weakest))
