from __future__ import annotations

import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import ANGLE_JOINTS, BODY, Side


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
) -> pd.DataFrame:
    """Calcule les 5 angles métier par frame. Renvoie un DataFrame avec une ligne par frame."""
    joints = BODY[side]
    rows: dict[str, np.ndarray] = {
        "frame_idx": df["frame_idx"].to_numpy(),
        "timestamp_ms": df["timestamp_ms"].to_numpy(),
    }

    for angle_name, (pa, pb, pc) in ANGLE_JOINTS.items():
        a = _px(df, joints[pa], width, height)
        b = _px(df, joints[pb], width, height)
        c = _px(df, joints[pc], width, height)
        rows[angle_name] = _vec_angle(a, b, c)

    shoulder = _px(df, joints["shoulder"], width, height)
    hip = _px(df, joints["hip"], width, height)
    rows["tronc"] = _trunk_angle(shoulder, hip)

    # Propagation NaN : si frame non détectée, tous les angles → NaN
    not_detected = ~df["detected"].to_numpy(dtype=bool)
    for key in ("genou", "hanche", "coude", "cheville", "tronc"):
        rows[key] = rows[key].copy()
        rows[key][not_detected] = np.nan

    return pd.DataFrame(rows)
