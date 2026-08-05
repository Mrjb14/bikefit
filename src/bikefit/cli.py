"""CLI BikeFit — Phase 1, 2 & 3B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Analyse vidéo de position cycliste (CLM)")

# ── Chemin de la base par défaut ──────────────────────────────────────────────
_DB_DEFAULT = Path.home() / ".bikefit" / "sessions.db"

# ── Noms des angles pour l'affichage ─────────────────────────────────────────
_ANGLE_LABELS = {
    "genou_extension_pmb": "Genou ext. PMB",
    "genou_flexion_pmh": "Genou flex. PMH",
    "hanche_fermee_pmh": "Hanche PMH",
    "tronc": "Tronc",
    "coude": "Coude",
    "cheville_pmb": "Cheville PMB",
}


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
    save_session: Annotated[bool, typer.Option(help="Sauvegarde la session dans la BDD")] = False,
    db: Annotated[Path, typer.Option(help="Chemin de la base SQLite")] = _DB_DEFAULT,
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

    # ── Session SQLite ────────────────────────────────────────────────────────
    if save_session:
        from bikefit.storage import Session, SessionStore

        store = SessionStore(db)
        session = store.save(
            Session(
                video_path=str(video.resolve()),
                bike_name=bike,
                profile=profile,
                setup=bike_profile.setup_actuel,
                metrics=metrics,
                report=fit_report,
            )
        )
        typer.echo(f"Session #{session.id} sauvegardée dans {db}")


@app.command()
def history(
    db: Annotated[Path, typer.Option(help="Chemin de la base SQLite")] = _DB_DEFAULT,
    limit: Annotated[int, typer.Option(help="Nombre de sessions à afficher")] = 20,
) -> None:
    """Liste les sessions enregistrées (les plus récentes en premier)."""
    from bikefit.storage import SessionStore

    if not db.exists():
        typer.echo("Aucune session enregistrée.")
        raise typer.Exit(0)

    store = SessionStore(db)
    sessions = store.list_all()[:limit]

    if not sessions:
        typer.echo("Aucune session enregistrée.")
        return

    typer.echo(f"{'#':>4}  {'Date':>19}  {'Vélo':<22}  {'Profil':<14}  Angles OK / Hors cible")
    typer.echo("-" * 80)
    for s in sessions:
        ok = len(s.report.angles_ok)
        hors = len(s.report.angles_hors_cible)
        date_str = s.date.strftime("%Y-%m-%d %H:%M")
        typer.echo(
            f"{s.id:>4}  {date_str:>19}  {s.bike_name:<22}"
            f"  {s.profile:<14}  {ok} OK / {hors} hors cible"
        )


@app.command()
def compare(
    id1: Annotated[int, typer.Argument(help="ID de la session de référence (avant)")],
    id2: Annotated[int, typer.Argument(help="ID de la session à comparer (après)")],
    db: Annotated[Path, typer.Option(help="Chemin de la base SQLite")] = _DB_DEFAULT,
) -> None:
    """Compare deux sessions : affiche les deltas d'angles et de recommandations."""
    from bikefit.storage import SessionStore

    if not db.exists():
        typer.echo("Aucune session enregistrée.", err=True)
        raise typer.Exit(1)

    store = SessionStore(db)
    s1 = store.get(id1)
    s2 = store.get(id2)

    for sid, s in [(id1, s1), (id2, s2)]:
        if s is None:
            typer.echo(f"Session #{sid} introuvable.", err=True)
            raise typer.Exit(1)

    assert s1 is not None and s2 is not None

    d1, d2 = s1.date.strftime("%Y-%m-%d"), s2.date.strftime("%Y-%m-%d")
    typer.echo(f"\nComparaison  #{id1} ({d1})  vs  #{id2} ({d2})\n")

    # ── Tableau des angles ────────────────────────────────────────────────────
    m1, m2 = s1.metrics.model_dump(), s2.metrics.model_dump()
    typer.echo(f"  {'Angle':<22}  {'#' + str(id1):>8}  {'#' + str(id2):>8}  {'Delta':>8}")
    typer.echo("  " + "-" * 54)
    for key, label in _ANGLE_LABELS.items():
        v1 = m1.get(key, {})
        v2 = m2.get(key, {})
        if not v1 or not v2:
            continue
        mean1, mean2 = v1["mean"], v2["mean"]
        delta = mean2 - mean1
        sign = "+" if delta >= 0 else ""
        ok2 = key not in s2.report.angles_hors_cible
        marker = "OK" if ok2 else "!!"
        typer.echo(f"  {label:<22}  {mean1:>7.1f}°  {mean2:>7.1f}°  {sign}{delta:>6.1f}°  {marker}")

    # ── Recommandations restantes dans #id2 ──────────────────────────────────
    recs2 = s2.report.recommandations
    if recs2:
        typer.echo(f"\n  Recommandations session #{id2} :")
        for r in recs2:
            delta_str = f"{r.delta_mm:+.0f} mm" if r.delta_mm is not None else "directionnel"
            typer.echo(f"    [{r.priorite.upper():<7}] {r.reglage} {r.direction} ({delta_str})")
    else:
        typer.echo(f"\n  Session #{id2} : aucune recommandation — position optimale !")

    typer.echo("")


if __name__ == "__main__":
    app()
