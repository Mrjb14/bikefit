"""Estimation du setup vélo depuis les landmarks de pose + calibration ArUco."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import BODY, Side


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit algébrique de cercle (méthode de Kåsa). Retourne (cx, cy, rayon)."""
    mat = np.column_stack([x, y, np.ones_like(x)])  # noqa: N806
    b_vec = x**2 + y**2
    c, *_ = np.linalg.lstsq(mat, b_vec, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = float(np.sqrt(max(0.0, c[2] + cx**2 + cy**2)))
    return float(cx), float(cy), r


def estimate_bike_setup(
    landmarks_df: pd.DataFrame,
    mm_per_pixel: float,
    image_width: int,
    image_height: int,
    side: Side = "right",
    bb_px: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Estime setup_actuel depuis les landmarks de pose et la calibration mm/pixel.

    Principe :
    - BB       : `bb_px` s'il est fourni (marqueur ArUco ID=1 collé sur le BB,
                 Phase 3A+, précision visée ±5 mm) ; sinon centre du cercle tracé
                 par la cheville sur l'ensemble des frames (fit algébrique de
                 Kåsa — approche intermédiaire, ±20 mm).
    - hauteur_selle_bb   : distance verticale BB → hanche au PMB × mm/px
    - hauteur_guidon_bb  : distance verticale BB → poignet médian × mm/px
                           (positif si pads au-dessus du BB)
    - reach_prolongateurs: distance horizontale BB → poignet médian × mm/px

    Les coordonnées image ont y croissant vers le bas (convention OpenCV).
    Clés préfixées par '_' = debug seulement, ne pas écrire dans le YAML.
    """
    lm = BODY[side]
    detected = landmarks_df[landmarks_df["detected"]]

    def _idx(name: str) -> int:
        return lm[name]

    def _cols(name: str) -> tuple[str, str]:
        i = _idx(name)
        return f"lm_{i}_x", f"lm_{i}_y"

    ax_col, ay_col = _cols("ankle")
    hx_col, hy_col = _cols("hip")
    wx_col, wy_col = _cols("wrist")

    # Pixels réels (les landmarks sont normalisés [0,1] dans le DataFrame)
    ankles = detected[[ax_col, ay_col]].dropna().copy()
    ankles[ax_col] *= image_width
    ankles[ay_col] *= image_height

    if len(ankles) < 10:
        raise ValueError(f"Cheville insuffisamment détectée ({len(ankles)} frames < 10).")

    # ── BB : marqueur ArUco s'il est fourni, sinon centre du cercle des chevilles ──
    if bb_px is not None:
        bb_x, bb_y = bb_px
        # Rayon = distance médiane cheville → BB réel (plus fiable que le rayon
        # de Kåsa, qui minimise l'erreur radiale autour d'un centre qui lui est
        # propre, pas forcément le vrai BB).
        radius_px = float(
            np.median(np.hypot(ankles[ax_col].to_numpy() - bb_x, ankles[ay_col].to_numpy() - bb_y))
        )
    else:
        bb_x, bb_y, radius_px = _fit_circle(
            ankles[ax_col].to_numpy(),
            ankles[ay_col].to_numpy(),
        )

    # ── Hauteur selle : BB → hanche au PMB ───────────────────────────────────
    hauteur_selle_bb: float | None = None
    if hy_col in detected.columns:
        tmp = detected[[ay_col, hy_col]].dropna().copy()
        tmp[ay_col] *= image_height
        tmp[hy_col] *= image_height
        # PMB = 10% des frames avec cheville la plus basse (y max dans l'image)
        thresh = tmp[ay_col].quantile(0.90)
        pmb = tmp[tmp[ay_col] >= thresh]
        if len(pmb) > 0:
            hip_pmb = pmb[hy_col].median()
            # En image : bb_y > hip_y (hanche plus haute = y plus petit)
            hauteur_selle_bb = (bb_y - hip_pmb) * mm_per_pixel

    # ── Hauteur guidon et reach : BB → poignet ───────────────────────────────
    hauteur_guidon_bb: float | None = None
    reach_prolongateurs: float | None = None
    if wx_col in detected.columns and wy_col in detected.columns:
        wrists = detected[[wx_col, wy_col]].dropna().copy()
        wrists[wx_col] *= image_width
        wrists[wy_col] *= image_height
        if len(wrists) > 0:
            wx_med = wrists[wx_col].median()
            wy_med = wrists[wy_col].median()
            hauteur_guidon_bb = (bb_y - wy_med) * mm_per_pixel
            reach_prolongateurs = abs(bb_x - wx_med) * mm_per_pixel

    return {
        "_bb_px": (round(bb_x), round(bb_y)),
        "_rayon_cheville_mm": round(radius_px * mm_per_pixel),
        "hauteur_selle_bb": round(hauteur_selle_bb) if hauteur_selle_bb is not None else None,
        "hauteur_guidon_bb": round(hauteur_guidon_bb) if hauteur_guidon_bb is not None else None,
        "reach_prolongateurs": round(reach_prolongateurs)
        if reach_prolongateurs is not None
        else None,
    }
