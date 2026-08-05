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

        per_cycle.append(
            {
                "genou_ext": cyc["genou"].max(),
                "genou_flex": cyc["genou"].min(),
                "hanche_fermee": cyc["hanche"].min(),
                "tronc": cyc["tronc"].mean(),
                "coude": cyc["coude"].mean(),
                "cheville_pmb": cyc["cheville"].max(),
            }
        )

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


def aggregate_frontal_cycles(
    frontal_df: pd.DataFrame,
    cycles: list[tuple[int, int]],
    mm_per_pixel: float | None = None,
) -> dict[str, AngleStats]:
    """Agrège les métriques plan frontal (Phase 3C) sur N cycles de pédalage.

    Renvoie un dict avec les clés 'genou_deviation_gauche', 'genou_deviation_droit'
    et, si mm_per_pixel est fourni (calibration ArUco), 'largeur_coudes' — prêt à
    injecter dans les champs optionnels de FitMetrics.
    """
    per_cycle: list[dict] = []

    for start, end in cycles:
        mask = (frontal_df["frame_idx"] >= start) & (frontal_df["frame_idx"] <= end)
        cyc = frontal_df[mask]

        if cyc["genou_deviation_gauche"].isna().mean() > 0.10:
            continue

        per_cycle.append(
            {
                "genou_deviation_gauche": cyc["genou_deviation_gauche"].mean(),
                "genou_deviation_droit": cyc["genou_deviation_droit"].mean(),
                "largeur_coudes_px": cyc["largeur_coudes_px"].mean(),
            }
        )

    if not per_cycle:
        raise ValueError(
            "Aucun cycle plan frontal valide trouvé."
            " Vérifiez la qualité de détection et le cadrage."
        )

    stats_df = pd.DataFrame(per_cycle)
    n_used = len(stats_df)

    result: dict[str, AngleStats] = {
        "genou_deviation_gauche": _stats(stats_df["genou_deviation_gauche"], n_used),
        "genou_deviation_droit": _stats(stats_df["genou_deviation_droit"], n_used),
    }

    if mm_per_pixel is not None:
        largeur_mm = stats_df["largeur_coudes_px"] * mm_per_pixel
        result["largeur_coudes"] = _stats(largeur_mm, n_used)

    return result
