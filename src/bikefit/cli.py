"""CLI BikeFit — Phase 1 & 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Analyse vidéo de position cycliste (CLM)")


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(help="Vidéo de profil (mp4/mov)")],
    view: Annotated[str, typer.Option(help="side | front")] = "side",
    bike: Annotated[str, typer.Option(help="Fiche vélo dans config/bikes/")] = "giant_trinity_2019",
    side: Annotated[str, typer.Option(help="right | left — côté filmé")] = "right",
    profile: Annotated[
        str, typer.Option(help="Profil cible : triathlon_XL | clm_long | clm_court")
    ] = "triathlon_XL",
    output: Annotated[Path | None, typer.Option(help="Fichier JSON de sortie")] = None,
    report: Annotated[Path | None, typer.Option(help="Rapport HTML de sortie")] = None,
    no_cache: Annotated[bool, typer.Option(help="Force le recalcul des landmarks")] = False,
) -> None:
    """Analyse une vidéo et produit les métriques de fitting + recommandations."""
    from bikefit.equipment import load_bike_profile
    from bikefit.ingest import VideoReader
    from bikefit.kinematics import (
        aggregate_cycles,
        compute_angles,
        detect_cycles,
        filter_landmarks,
    )
    from bikefit.pose import LandmarkCache, MediaPipePoseEstimator
    from bikefit.pose.cache import CACHE_DIR
    from bikefit.recommend import recommend as run_recommend
    from bikefit.report import render_html

    if not video.exists():
        typer.echo(f"Erreur : vidéo introuvable : {video}", err=True)
        raise typer.Exit(1)

    if side not in ("right", "left"):
        typer.echo("Erreur : --side doit être 'right' ou 'left'", err=True)
        raise typer.Exit(1)

    # ── Ingest ───────────────────────────────────────────────────────────────
    reader = VideoReader(video)
    typer.echo(
        f"Vidéo : {reader.width}x{reader.height}"
        f" @ {reader.fps:.1f} fps, {reader.frame_count} frames"
    )

    # ── Pose (avec cache Parquet) ─────────────────────────────────────────────
    cache = LandmarkCache(CACHE_DIR / "landmarks")
    if no_cache:
        from bikefit.pose.cache import _video_key

        key = _video_key(video)
        cache_file = CACHE_DIR / "landmarks" / f"{video.stem}_{key}.parquet"
        if cache_file.exists():
            cache_file.unlink()
            typer.echo("Cache supprimé.")

    with MediaPipePoseEstimator() as estimator:
        df = cache.get_or_compute(video, reader, estimator)

    detection_rate = df["detected"].mean()
    typer.echo(f"Taux de détection : {detection_rate:.1%}")

    # ── Kinematics ───────────────────────────────────────────────────────────
    df_filtered = filter_landmarks(df, reader.fps)
    angle_df = compute_angles(df_filtered, side, reader.width, reader.height)  # type: ignore[arg-type]
    cycles = detect_cycles(df_filtered, side, reader.fps)  # type: ignore[arg-type]
    typer.echo(f"Cycles détectés : {len(cycles)}")

    if len(cycles) < 3:
        typer.echo("Avertissement : moins de 3 cycles — résultats peu fiables.", err=True)

    metrics = aggregate_cycles(angle_df, cycles)

    # ── Équipement ───────────────────────────────────────────────────────────
    try:
        bike_profile = load_bike_profile(bike)
    except FileNotFoundError:
        typer.echo(
            f"Avertissement : fiche vélo '{bike}' introuvable, recommandations sans butées.",
            err=True,
        )
        from bikefit.equipment.models import BikeModel, BikeProfile

        bike_profile = BikeProfile(modele=BikeModel(marque="", nom=bike, annee=0, taille=""))

    # ── Recommandations ──────────────────────────────────────────────────────
    fit_report = run_recommend(metrics, bike_profile, profile=profile)

    # ── Sortie JSON ──────────────────────────────────────────────────────────
    result = {
        "metriques": metrics.model_dump(),
        "rapport": fit_report.model_dump(),
    }
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if output:
        output.write_text(json_str, encoding="utf-8")
        typer.echo(f"Métriques sauvegardées -> {output}")
    else:
        typer.echo("\n" + json_str)

    # ── Rapport HTML ─────────────────────────────────────────────────────────
    if report:
        render_html(metrics, fit_report, angle_df, video, report)
        typer.echo(f"Rapport HTML -> {report}")
    else:
        default_report = video.with_suffix(".html")
        render_html(metrics, fit_report, angle_df, video, default_report)
        typer.echo(f"Rapport HTML -> {default_report}")


if __name__ == "__main__":
    app()
