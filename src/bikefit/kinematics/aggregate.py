from __future__ import annotations

import pandas as pd

from bikefit.kinematics.metrics import AngleStats, FitMetrics


def _stats(series: pd.Series, n_cycles: int) -> AngleStats:
    return AngleStats(
        mean=round(float(series.mean()), 1),
        std=round(float(series.std(ddof=1)), 1) if len(series) > 1 else 0.0,
        min=round(float(series.min()), 1),
        max=round(float(series.max()), 1),
        n_cycles=n_cycles,
    )


def aggregate_cycles(
    angle_df: pd.DataFrame,
    cycles: list[tuple[int, int]],
) -> FitMetrics:
    """Agrège les angles sur N cycles de pédalage et retourne FitMetrics."""
    per_cycle: list[dict] = []

    for start, end in cycles:
        mask = (angle_df["frame_idx"] >= start) & (angle_df["frame_idx"] <= end)
        cyc = angle_df[mask]

        # Ignore les cycles avec trop de NaN (> 10 %)
        if cyc["genou"].isna().mean() > 0.10:
            continue

        per_cycle.append({
            "genou_ext": cyc["genou"].max(),
            "genou_flex": cyc["genou"].min(),
            "hanche_fermee": cyc["hanche"].min(),
            "tronc": cyc["tronc"].mean(),
            "coude": cyc["coude"].mean(),
            "cheville_pmb": cyc["cheville"].max(),
        })

    if not per_cycle:
        raise ValueError(
            "Aucun cycle valide trouvé. Vérifiez la qualité de détection et le cadrage."
        )

    stats_df = pd.DataFrame(per_cycle)
    n_used = len(stats_df)

    return FitMetrics(
        genou_extension_pmb=_stats(stats_df["genou_ext"], n_used),
        genou_flexion_pmh=_stats(stats_df["genou_flex"], n_used),
        hanche_fermee_pmh=_stats(stats_df["hanche_fermee"], n_used),
        tronc=_stats(stats_df["tronc"], n_used),
        coude=_stats(stats_df["coude"], n_used),
        cheville_pmb=_stats(stats_df["cheville_pmb"], n_used),
        n_cycles_total=len(cycles),
        n_cycles_used=n_used,
        detection_rate=round(float(angle_df["genou"].notna().mean()), 3),
    )
