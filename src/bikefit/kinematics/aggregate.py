from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from bikefit.kinematics.metrics import AngleStats, FitMetrics

# Un cycle n'est retenu pour une métrique que si moins de 10 % de ses frames sont NaN
# pour cette métrique (frame non détectée ou point inféré, cf. visibility.py).
MAX_NAN_RATIO_PER_CYCLE = 0.10

# (clé de sortie, colonne d'angle source, réduction sur le cycle)
_SAGITTAL_REDUCTIONS: list[tuple[str, str, str]] = [
    ("genou_extension_pmb", "genou", "max"),
    ("genou_flexion_pmh", "genou", "min"),
    ("hanche_fermee_pmh", "hanche", "min"),
    ("tronc", "tronc", "mean"),
    ("coude", "coude", "mean"),
    ("cheville_pmb", "cheville", "max"),
]

_FRONTAL_COLUMNS = ("genou_deviation_gauche", "genou_deviation_droit", "largeur_coudes_px")


def _stats(
    series: pd.Series,
    n_cycles: int,
    visibility: float | None = None,
    frames_kept: float | None = None,
) -> AngleStats:
    """Statistiques sur les valeurs par cycle. Série vide -> métrique non mesurée (NaN)."""
    if len(series) == 0:
        nan = math.nan
        return AngleStats(
            mean=nan,
            std=nan,
            min=nan,
            max=nan,
            n_cycles=0,
            visibility=round(visibility, 2) if visibility is not None else None,
            frames_kept=round(frames_kept, 3) if frames_kept is not None else None,
        )
    return AngleStats(
        mean=round(float(series.mean()), 1),
        std=round(float(series.std(ddof=1)), 1) if len(series) > 1 else 0.0,
        min=round(float(series.min()), 1),
        max=round(float(series.max()), 1),
        n_cycles=n_cycles,
        visibility=round(visibility, 2) if visibility is not None else None,
        frames_kept=round(frames_kept, 3) if frames_kept is not None else None,
    )


def _frames_kept(angle_df: pd.DataFrame, column: str) -> float:
    """Part des frames où l'angle a survécu au filtrage (détection + visibilité)."""
    if column not in angle_df.columns or len(angle_df) == 0:
        return 1.0
    return float(angle_df[column].notna().mean())


def _mean_vis(df: pd.DataFrame, column: str) -> float | None:
    col = f"{column}_vis"
    if col not in df.columns:
        return None
    value = df[col].mean()
    return None if pd.isna(value) else float(value)


def _per_cycle_values(
    df: pd.DataFrame,
    cycles: list[tuple[int, int]],
    column: str,
    reduce: Callable[[pd.Series], float],
) -> pd.Series:
    """Une valeur par cycle valide pour `column` (cycles trop lacunaires écartés).

    La validité est jugée métrique par métrique : un coude masqué sur un cycle ne
    doit pas faire perdre le genou de ce même cycle, et inversement. Une valeur
    extraite d'un cycle troué (ex. max du genou alors que le PMB est masqué)
    biaiserait la moyenne sans aucun signal d'alerte.
    """
    values: list[float] = []
    for start, end in cycles:
        mask = (df["frame_idx"] >= start) & (df["frame_idx"] <= end)
        cyc = df.loc[mask, column]
        if cyc.empty or cyc.isna().mean() > MAX_NAN_RATIO_PER_CYCLE:
            continue
        values.append(float(reduce(cyc)))
    return pd.Series(values, dtype=float)


_REDUCERS: dict[str, Callable[[pd.Series], float]] = {
    "max": lambda s: s.max(),
    "min": lambda s: s.min(),
    "mean": lambda s: s.mean(),
}


def aggregate_cycles(
    angle_df: pd.DataFrame,
    cycles: list[tuple[int, int]],
) -> FitMetrics:
    """Agrège les angles sur N cycles de pédalage et retourne FitMetrics.

    Le genou est la métrique de référence : sans aucun cycle exploitable pour lui,
    l'analyse n'a pas de sens et une ValueError est levée. Les autres métriques
    peuvent individuellement être non mesurées (AngleStats à NaN, `is_measured`
    faux) sans faire échouer l'ensemble.
    """
    stats: dict[str, AngleStats] = {}
    for key, column, how in _SAGITTAL_REDUCTIONS:
        values = _per_cycle_values(angle_df, cycles, column, _REDUCERS[how])
        stats[key] = _stats(
            values, len(values), _mean_vis(angle_df, column), _frames_kept(angle_df, column)
        )

    n_used = stats["genou_extension_pmb"].n_cycles
    if n_used == 0:
        raise ValueError(
            "Aucun cycle valide trouvé. Vérifiez la qualité de détection et le cadrage."
        )

    return FitMetrics(
        genou_extension_pmb=stats["genou_extension_pmb"],
        genou_flexion_pmh=stats["genou_flexion_pmh"],
        hanche_fermee_pmh=stats["hanche_fermee_pmh"],
        tronc=stats["tronc"],
        coude=stats["coude"],
        cheville_pmb=stats["cheville_pmb"],
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

    Renvoie un dict prêt à injecter dans les champs optionnels de FitMetrics, avec
    les clés 'genou_deviation_gauche', 'genou_deviation_droit' et, si mm_per_pixel
    est fourni (calibration ArUco **de la caméra de face**), 'largeur_coudes'.
    Une métrique sans aucun cycle exploitable est simplement absente du dict (un
    genou masqué ne fait pas perdre l'autre). ValueError si aucune ne l'est.
    """
    result: dict[str, AngleStats] = {}
    for column in _FRONTAL_COLUMNS:
        values = _per_cycle_values(frontal_df, cycles, column, _REDUCERS["mean"])
        if values.empty:
            continue
        vis = _mean_vis(frontal_df, column)
        kept = _frames_kept(frontal_df, column)
        if column == "largeur_coudes_px":
            if mm_per_pixel is None:
                continue  # sans échelle, une largeur en pixels n'est pas exploitable
            result["largeur_coudes"] = _stats(values * mm_per_pixel, len(values), vis, kept)
        else:
            result[column] = _stats(values, len(values), vis, kept)

    if not result:
        raise ValueError(
            "Aucun cycle plan frontal valide trouvé."
            " Vérifiez la qualité de détection et le cadrage."
        )
    return result
