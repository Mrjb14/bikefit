from __future__ import annotations

import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import ANGLE_JOINTS, BODY, Side
from bikefit.kinematics.visibility import (
    VISIBILITY_THRESHOLD,
    reliability_mask,
    weakest_visibility,
)


def _px(df: pd.DataFrame, lm_idx: int, width: int, height: int) -> np.ndarray:
    x = df[f"lm_{lm_idx}_x"].to_numpy(dtype=float) * width
    y = df[f"lm_{lm_idx}_y"].to_numpy(dtype=float) * height
    return np.stack([x, y], axis=1)  # (N, 2)


def _vec_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle vectorisé (degrés) au sommet b pour N frames. Renvoie NaN si vecteur nul."""
    v1 = a - b  # (N, 2)
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    dot = np.sum(v1 * v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where((n1 > 1e-6) & (n2 > 1e-6), dot / (n1 * n2), np.nan)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> np.ndarray:
    """Angle du tronc par rapport à l'horizontale (degrés). y axe image vers le bas."""
    v = shoulder - hip  # (N, 2)
    return np.degrees(np.arctan2(-v[:, 1], np.abs(v[:, 0])))


def compute_angles(
    df: pd.DataFrame,
    side: Side,
    width: int,
    height: int,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
) -> pd.DataFrame:
    """Calcule les 5 angles métier par frame. Renvoie un DataFrame avec une ligne par frame.

    Un angle dont l'un des sommets est sous le seuil de visibilité est mis à NaN :
    MediaPipe infère les points masqués (côté opposé à la caméra typiquement à 0.2-0.3),
    et les inclure produirait une valeur d'apparence normale calculée sur des positions
    devinées.
    """
    joints = BODY[side]
    rows: dict[str, np.ndarray] = {
        "frame_idx": df["frame_idx"].to_numpy(),
        "timestamp_ms": df["timestamp_ms"].to_numpy(),
    }
    unreliable: dict[str, np.ndarray] = {}

    for angle_name, (pa, pb, pc) in ANGLE_JOINTS.items():
        a = _px(df, joints[pa], width, height)
        b = _px(df, joints[pb], width, height)
        c = _px(df, joints[pc], width, height)
        idx = (joints[pa], joints[pb], joints[pc])
        rows[angle_name] = _vec_angle(a, b, c)
        rows[f"{angle_name}_vis"] = weakest_visibility(df, idx)
        unreliable[angle_name] = ~reliability_mask(df, idx, visibility_threshold)

    shoulder = _px(df, joints["shoulder"], width, height)
    hip = _px(df, joints["hip"], width, height)
    trunk_idx = (joints["shoulder"], joints["hip"])
    rows["tronc"] = _trunk_angle(shoulder, hip)
    rows["tronc_vis"] = weakest_visibility(df, trunk_idx)
    unreliable["tronc"] = ~reliability_mask(df, trunk_idx, visibility_threshold)

    # Propagation NaN : frame non détectée, ou sommet inféré plutôt qu'observé
    not_detected = ~df["detected"].to_numpy(dtype=bool)
    for key in ("genou", "hanche", "coude", "cheville", "tronc"):
        rows[key] = rows[key].copy()
        rows[key][not_detected | unreliable[key]] = np.nan

    return pd.DataFrame(rows)
