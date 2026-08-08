from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from bikefit.kinematics.landmarks import BODY, Side

# Rejette un cycle dont la durée s'écarte de plus de 35% de la médiane des cycles
# détectés : un faux PMB (bruit de détection) produit un cycle anormalement court,
# qui fausse l'écart-type des angles agrégés (cf. CLAUDE.md, "Robustesse des cycles").
CYCLE_DURATION_TOLERANCE = 0.35

# Sous ce nombre de cycles, la médiane n'est pas assez fiable pour distinguer
# un vrai outlier d'une variation normale de cadence.
MIN_CYCLES_FOR_PLAUSIBILITY_FILTER = 3


def _filter_implausible_cycles(cycles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Rejette les cycles dont la durée s'écarte trop de la médiane (faux PMB)."""
    if len(cycles) < MIN_CYCLES_FOR_PLAUSIBILITY_FILTER:
        return cycles

    durations = np.array([end - start for start, end in cycles], dtype=float)
    median = float(np.median(durations))
    if median <= 0:
        return cycles

    plausible = np.abs(durations - median) <= CYCLE_DURATION_TOLERANCE * median
    return [cycle for cycle, keep in zip(cycles, plausible, strict=True) if keep]


def detect_cycles(
    df: pd.DataFrame,
    side: Side,
    fps: float,
) -> list[tuple[int, int]]:
    """Détecte les cycles de pédalage via les extrema de la cheville (axe y image).

    Retourne une liste de (frame_début, frame_fin) pour chaque cycle complet.
    Le point mort bas (PMB) correspond au maximum de y_cheville (cheville en bas de course).
    Les cycles dont la durée s'écarte trop de la médiane (faux PMB) sont rejetés,
    cf. `_filter_implausible_cycles`.
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

    # find_peaks renvoie des positions dans le tableau ; on les traduit en frame_idx,
    # car c'est sur cette colonne que aggregate_cycles/aggregate_frontal_cycles
    # découpent ensuite. Identique tant que le DataFrame part de la frame 0 sans trou,
    # mais indispensable dès qu'il est tronqué (options --start/--end).
    frame_ids = df["frame_idx"].to_numpy()
    cycles = [
        (int(frame_ids[pmb_indices[i]]), int(frame_ids[pmb_indices[i + 1]]))
        for i in range(len(pmb_indices) - 1)
    ]
    return _filter_implausible_cycles(cycles)
