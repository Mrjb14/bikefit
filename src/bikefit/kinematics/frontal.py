"""Métriques du plan frontal (vue de face, Phase 3C) : alignement genoux, largeur coudes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import BODY


def _px(df: pd.DataFrame, lm_idx: int, width: int, height: int) -> np.ndarray:
    x = df[f"lm_{lm_idx}_x"].to_numpy(dtype=float) * width
    y = df[f"lm_{lm_idx}_y"].to_numpy(dtype=float) * height
    return np.stack([x, y], axis=1)  # (N, 2)


def _knee_deviation_angle(hip: np.ndarray, ankle: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Angle signé (degrés) entre la direction hanche→cheville et hanche→genou.

    0° si le genou est exactement sur la droite hanche-cheville (jambe "droite"
    en plan frontal, pas de déviation). Le signe vient du produit vectoriel
    (hanche→cheville) × (hanche→genou) : indique de quel côté le genou dévie,
    mais l'interprétation valgus/varus dépend du côté (gauche/droit) et du sens
    de la caméra — laissée au module recommend/report, pas calculée ici.
    """
    v_ref = ankle - hip  # (N, 2)
    v_knee = knee - hip
    cross = v_ref[:, 0] * v_knee[:, 1] - v_ref[:, 1] * v_knee[:, 0]
    dot = np.sum(v_ref * v_knee, axis=1)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arctan2(cross, dot))


def compute_frontal_metrics(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Calcule les métriques plan frontal par frame depuis les landmarks vue de face.

    Renvoie un DataFrame avec :
    - genou_deviation_gauche / genou_deviation_droit : angle signé (degrés) de
      déviation du genou par rapport à la droite hanche-cheville.
    - largeur_coudes_px : distance coude gauche ↔ coude droit en pixels
      (conversion en mm via calibration ArUco laissée à l'agrégation, cf. aggregate.py).
    """
    rows: dict[str, np.ndarray] = {
        "frame_idx": df["frame_idx"].to_numpy(),
        "timestamp_ms": df["timestamp_ms"].to_numpy(),
    }

    for side, key in (("left", "genou_deviation_gauche"), ("right", "genou_deviation_droit")):
        hip = _px(df, BODY[side]["hip"], width, height)
        knee = _px(df, BODY[side]["knee"], width, height)
        ankle = _px(df, BODY[side]["ankle"], width, height)
        rows[key] = _knee_deviation_angle(hip, ankle, knee)

    left_elbow = _px(df, BODY["left"]["elbow"], width, height)
    right_elbow = _px(df, BODY["right"]["elbow"], width, height)
    rows["largeur_coudes_px"] = np.linalg.norm(left_elbow - right_elbow, axis=1)

    # Propagation NaN : si frame non détectée, toutes les métriques → NaN
    not_detected = ~df["detected"].to_numpy(dtype=bool)
    for key in ("genou_deviation_gauche", "genou_deviation_droit", "largeur_coudes_px"):
        rows[key] = rows[key].copy()
        rows[key][not_detected] = np.nan

    return pd.DataFrame(rows)
