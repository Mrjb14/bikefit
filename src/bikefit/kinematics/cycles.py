from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from bikefit.kinematics.landmarks import BODY, Side


def detect_cycles(
    df: pd.DataFrame,
    side: Side,
    fps: float,
) -> list[tuple[int, int]]:
    """Détecte les cycles de pédalage via les extrema de la cheville (axe y image).

    Retourne une liste de (frame_début, frame_fin) pour chaque cycle complet.
    Le point mort bas (PMB) correspond au maximum de y_cheville (cheville en bas de course).
    """
    ankle_idx = BODY[side]["ankle"]
    ankle_y = df[f"lm_{ankle_idx}_y"].to_numpy(dtype=float)

    # Remplace NaN par interpolation pour la détection de pics
    idx = np.arange(len(ankle_y))
    detected = ~np.isnan(ankle_y)
    if detected.sum() < 2:
        return []
    ankle_y_interp = np.interp(idx, idx[detected], ankle_y[detected])

    # distance minimale entre deux PMB : 60 rpm = 1 cycle/s → fps frames
    # distance maximale : 40 rpm = 1.5 s → 1.5 * fps frames
    min_dist = int(fps * 0.4)  # 150 rpm max
    pmb_indices, _ = find_peaks(ankle_y_interp, distance=min_dist)

    if len(pmb_indices) < 2:
        return []

    cycles = [(int(pmb_indices[i]), int(pmb_indices[i + 1])) for i in range(len(pmb_indices) - 1)]
    return cycles
