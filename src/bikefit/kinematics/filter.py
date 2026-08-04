from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _sg_window(fps: float) -> int:
    w = max(5, int(fps / 5))
    return w if w % 2 == 1 else w + 1


def filter_landmarks(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Applique un filtre Savitzky-Golay (~6 Hz) sur les coordonnées des landmarks.

    Les frames non détectées sont interpolées linéairement avant le filtre,
    puis remises à NaN en sortie.
    """
    result = df.copy()
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    window = _sg_window(fps)
    detected = df["detected"].to_numpy(dtype=bool)
    idx = np.arange(len(df))
    detected_idx = idx[detected]

    if len(detected_idx) < window:
        return result

    for col in coord_cols:
        series = df[col].to_numpy(dtype=float)
        # Interpolation linéaire sur les gaps
        series_interp = np.interp(idx, detected_idx, series[detected])
        # Filtre Savitzky-Golay
        series_filt = savgol_filter(series_interp, window_length=window, polyorder=3)
        # Remet NaN sur les frames non détectées
        series_filt[~detected] = np.nan
        result[col] = series_filt

    return result
