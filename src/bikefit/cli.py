"""CLI BikeFit — Phase 1, 2 & 3B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from bikefit.kinematics.landmarks import Side

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
    front: Annotated[
        str | None,
        typer.Option(
            help="Vidéo (fichier) ou flux MJPEG (URL http://...) vue de face, optionnel — Phase 3C"
        ),
    ] = None,
    front_duration: Annotated[
        float, typer.Option(help="Durée de capture (s) si --front est un flux réseau")
    ] = 15.0,
    bike: Annotated[str, typer.Option(help="Fiche vélo dans config/bikes/")] = "giant_trinity_2019",
    side: Annotated[str, typer.Option(help="right | left — côté filmé")] = "right",
    profile: Annotated[
        str, typer.Option(help="Profil cible : triathlon_XL | clm_long | clm_court")
    ] = "triathlon_XL",
    output: Annotated[Path | None, typer.Option(help="Fichier JSON de sortie")] = None,
    report: Annotated[Path | None, typer.Option(help="Rapport HTML de sortie")] = None,
    rider: Annotated[
        str, typer.Option(help="Fiche coureur dans config/rider/ (sans .yaml)")
    ] = "mensurations",
    no_cache: Annotated[bool, typer.Option(help="Force le recalcul des landmarks")] = False,
    save_session: Annotated[bool, typer.Option(help="Sauvegarde la session dans la BDD")] = False,
    db: Annotated[Path, typer.Option(help="Chemin de la base SQLite")] = _DB_DEFAULT,
    marker_size: Annotated[
        float | None,
        typer.Option(help="Taille du côté du marqueur ArUco en mm (ex. 100)."),
    ] = None,
) -> None:
    """Analyse une vidéo et produit les métriques de fitting + recommandations."""
    from bikefit.equipment import load_bike_profile, load_rider_profile
    from bikefit.ingest import VideoReader
    from bikefit.kinematics import (
        aggregate_cycles,
        aggregate_frontal_cycles,
        compute_angles,
        compute_frontal_metrics,
        detect_cycles,
        filter_landmarks,
    )
    from bikefit.pose import LandmarkCache, MediaPipePoseEstimator, compute_landmarks
    from bikefit.pose.cache import CACHE_DIR
    from bikefit.recommend import recommend as run_recommend
    from bikefit.report import render_html

    if not video.exists():
        typer.echo(f"Erreur : vidéo introuvable : {video}", err=True)
        raise typer.Exit(1)

    if side not in ("right", "left"):
        typer.echo("Erreur : --side doit être 'right' ou 'left'", err=True)
        raise typer.Exit(1)
    filmed_side: Side = "right" if side == "right" else "left"

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
    angle_df = compute_angles(df_filtered, filmed_side, reader.width, reader.height)
    cycles = detect_cycles(df_filtered, filmed_side, reader.fps)
    typer.echo(f"Cycles détectés : {len(cycles)}")

    if len(cycles) < 3:
        typer.echo("Avertissement : moins de 3 cycles — résultats peu fiables.", err=True)

    metrics = aggregate_cycles(angle_df, cycles)

    # ── Calibration ArUco (optionnelle) ──────────────────────────────────────
    if marker_size is not None:
        from bikefit.calibration import calibrate_from_video

        typer.echo(f"Calibration ArUco (marqueur {marker_size:.0f} mm)...")
        calib = calibrate_from_video(str(video), marker_size_mm=marker_size)
        if calib is None:
            typer.echo("Avertissement : aucun marqueur ArUco détecté.", err=True)
        else:
            metrics = metrics.model_copy(update={"calibration": calib})
            status = "fiable" if calib.is_reliable else "peu fiable"
            typer.echo(
                f"Calibration {status} : {calib.mm_per_pixel:.4f} mm/px"
                f" ({calib.n_detected}/{calib.n_sampled} frames)"
            )

    # ── Vue de face (optionnelle, Phase 3C) ──────────────────────────────────
    if front is not None:
        typer.echo(f"\nVue de face : {front}")
        front_reader = VideoReader(front)
        typer.echo(
            f"  {front_reader.width}x{front_reader.height} @ {front_reader.fps:.1f} fps"
            + (" (flux réseau)" if front_reader.is_stream else "")
        )

        with MediaPipePoseEstimator() as front_estimator:
            if front_reader.is_stream:
                typer.echo(f"  Capture {front_duration:.0f}s...")
                front_df = compute_landmarks(
                    front_reader, front_estimator, max_duration_s=front_duration
                )
            else:
                front_cache = LandmarkCache(CACHE_DIR / "landmarks")
                front_df = front_cache.get_or_compute(Path(front), front_reader, front_estimator)

        front_detection_rate = front_df["detected"].mean()
        typer.echo(f"  Taux de détection : {front_detection_rate:.1%}")

        front_filtered = filter_landmarks(front_df, front_reader.fps)
        frontal_df = compute_frontal_metrics(
            front_filtered, front_reader.width, front_reader.height
        )
        front_cycles = detect_cycles(front_filtered, "right", front_reader.fps)
        typer.echo(f"  Cycles détectés : {len(front_cycles)}")

        if len(front_cycles) < 2:
            typer.echo(
                "  Avertissement : moins de 2 cycles vue de face — résultats peu fiables.",
                err=True,
            )
        else:
            mm_per_px = metrics.calibration.mm_per_pixel if metrics.calibration else None
            frontal_stats = aggregate_frontal_cycles(
                frontal_df, front_cycles, mm_per_pixel=mm_per_px
            )
            metrics = metrics.model_copy(update=frontal_stats)

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

    rider_profile = load_rider_profile(rider)
    if rider_profile is None and rider != "mensurations":
        typer.echo(f"Avertissement : fiche coureur '{rider}' introuvable.", err=True)
    elif rider_profile is not None:
        flex = rider_profile.profil.flexibilite
        epreuve = rider_profile.objectif.type_epreuve
        typer.echo(
            f"Coureur : flexibilité={flex or '?'}, épreuve={epreuve or '?'}"
            + (
                f", entrejambe={rider_profile.mensurations.entrejambe} mm"
                if rider_profile.mensurations.entrejambe
                else ""
            )
        )

    # ── Recommandations ──────────────────────────────────────────────────────
    fit_report = run_recommend(metrics, bike_profile, profile=profile, rider=rider_profile)

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
    report_path = report if report else video.with_suffix(".html")
    render_html(
        metrics,
        fit_report,
        angle_df,
        video,
        report_path,
        rider=rider_profile,
        bike=bike_profile,
    )
    typer.echo(f"Rapport HTML -> {report_path}")

    # ── Session SQLite ────────────────────────────────────────────────────────
    if save_session:
        from bikefit.storage import Session, SessionStore

        store = SessionStore(db)
        session = store.save(
            Session(
                video_path=str(video.resolve()),
                bike_name=bike,
                rider_name=rider if rider_profile is not None else None,
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


@app.command("measure-body")
def measure_body(
    photo: Annotated[Path, typer.Argument(help="Photo debout du coureur (jpg/png)")],
    rider: Annotated[
        str, typer.Option(help="Fiche coureur dans config/rider/ (sans .yaml)")
    ] = "mensurations",
    write: Annotated[bool, typer.Option(help="Met à jour les champs null dans le YAML")] = False,
) -> None:
    """Estime les mensurations corporelles depuis une photo debout (MediaPipe)."""
    from bikefit._paths import CONFIG_DIR
    from bikefit.equipment import load_rider_profile
    from bikefit.measure.body import estimate_body_measurements
    from bikefit.measure.yaml_io import update_null_fields

    if not photo.exists():
        typer.echo(f"Erreur : photo introuvable : {photo}", err=True)
        raise typer.Exit(1)

    rider_profile = load_rider_profile(rider)
    taille_mm = rider_profile.mensurations.taille if rider_profile else None
    if taille_mm:
        typer.echo(f"Taille connue : {taille_mm} mm — utilisée pour l'échelle.")
    else:
        typer.echo("Avertissement : taille non renseignée — estimations non calibrées.", err=True)

    typer.echo("Détection de pose en cours...")
    estimates = estimate_body_measurements(photo, taille_mm=taille_mm)

    typer.echo("\nEstimations (mm) — précision indicative ±5-10% :")
    for key, val in estimates.items():
        typer.echo(f"  {key:<20} {val:>5} mm")

    # ── Annotation visuelle ───────────────────────────────────────────────────
    from bikefit.measure.render import annotate_body_photo

    annotated = photo.with_name(photo.stem + "_annotated.jpg")
    try:
        annotate_body_photo(photo, estimates, annotated)
        typer.echo(f"Photo annotée     -> {annotated}")
    except Exception as exc:
        typer.echo(f"Avertissement : annotation échouée ({exc})", err=True)

    if write:
        yaml_path = CONFIG_DIR / "rider" / f"{rider}.yaml"
        if not yaml_path.exists():
            typer.echo(f"Erreur : {yaml_path} introuvable.", err=True)
            raise typer.Exit(1)
        updated = update_null_fields(yaml_path, estimates)
        if updated:
            typer.echo(f"\nOK: {len(updated)} champ(s) mis à jour dans {yaml_path} : {updated}")
        else:
            typer.echo("\nAucun champ null à mettre à jour (déjà renseignés).")
    else:
        typer.echo(
            "\nAjoutez --write pour mettre à jour automatiquement les champs null"
            f" dans config/rider/{rider}.yaml"
        )


@app.command("measure-setup")
def measure_setup(
    video: Annotated[Path, typer.Argument(help="Vidéo de profil sur le home trainer")],
    marker_size: Annotated[
        float, typer.Option(help="Taille du marqueur ArUco en mm (obligatoire)")
    ],
    side: Annotated[str, typer.Option(help="right | left — côté filmé")] = "right",
    bike: Annotated[str, typer.Option(help="Fiche vélo dans config/bikes/")] = "giant_trinity_2019",
    write: Annotated[bool, typer.Option(help="Met à jour les champs null dans le YAML")] = False,
    no_cache: Annotated[bool, typer.Option(help="Force le recalcul des landmarks")] = False,
) -> None:
    """Estime setup_actuel (hauteur selle, guidon, reach) depuis la vidéo + ArUco."""
    from bikefit._paths import CONFIG_DIR
    from bikefit.calibration import calibrate_from_video
    from bikefit.ingest import VideoReader
    from bikefit.measure.setup import estimate_bike_setup
    from bikefit.measure.yaml_io import update_null_fields
    from bikefit.pose import LandmarkCache, MediaPipePoseEstimator
    from bikefit.pose.cache import CACHE_DIR

    if not video.exists():
        typer.echo(f"Erreur : vidéo introuvable : {video}", err=True)
        raise typer.Exit(1)
    if side not in ("right", "left"):
        typer.echo("Erreur : --side doit être 'right' ou 'left'", err=True)
        raise typer.Exit(1)
    filmed_side: Side = "right" if side == "right" else "left"

    # ── Pose (avec cache) ─────────────────────────────────────────────────────
    reader = VideoReader(video)
    typer.echo(f"Vidéo : {reader.width}×{reader.height} @ {reader.fps:.1f} fps")

    cache = LandmarkCache(CACHE_DIR / "landmarks")
    if no_cache:
        from bikefit.pose.cache import _video_key

        key = _video_key(video)
        cache_file = CACHE_DIR / "landmarks" / f"{video.stem}_{key}.parquet"
        if cache_file.exists():
            cache_file.unlink()

    with MediaPipePoseEstimator() as estimator:
        df = cache.get_or_compute(video, reader, estimator)

    typer.echo(f"Détection pose : {df['detected'].mean():.1%} des frames")

    # ── Calibration ArUco ─────────────────────────────────────────────────────
    typer.echo(f"Calibration ArUco (marqueur {marker_size:.0f} mm)...")
    calib = calibrate_from_video(str(video), marker_size_mm=marker_size)
    if calib is None:
        typer.echo("Erreur : aucun marqueur ArUco détecté. Vérifier --marker-size.", err=True)
        raise typer.Exit(1)
    status = "fiable" if calib.is_reliable else "peu fiable"
    typer.echo(
        f"Calibration {status} : {calib.mm_per_pixel:.4f} mm/px"
        f" ({calib.n_detected}/{calib.n_sampled} frames)"
    )

    # ── Estimation du setup ───────────────────────────────────────────────────
    typer.echo("Estimation du setup vélo (fit de cercle sur trajectoire cheville)...")
    estimates = estimate_bike_setup(
        df,
        mm_per_pixel=calib.mm_per_pixel,
        image_width=reader.width,
        image_height=reader.height,
        side=filmed_side,
    )

    # ── Annotation visuelle (avant de popper les clés de debug) ──────────────
    from bikefit.measure.render import annotate_setup_frame

    annotated = video.with_name(video.stem + "_annotated.jpg")
    try:
        annotate_setup_frame(
            str(video),
            df,
            estimates,
            calib.mm_per_pixel,
            reader.width,
            reader.height,
            filmed_side,
            annotated,
        )
        typer.echo(f"Frame annotée     -> {annotated}")
    except Exception as exc:
        typer.echo(f"Avertissement : annotation échouée ({exc})", err=True)

    # Infos de debug
    bb_px = estimates.pop("_bb_px")
    rayon = estimates.pop("_rayon_cheville_mm")
    typer.echo(f"  BB estimé : pixel ({bb_px[0]}, {bb_px[1]})")
    typer.echo(f"  Rayon cheville ≈ {rayon} mm (manivelle + hauteur chaussure)")

    typer.echo("\nEstimations setup_actuel (mm) — précision indicative ±20 mm :")
    display = {k: v for k, v in estimates.items() if v is not None}
    for key, val in display.items():
        typer.echo(f"  {key:<24} {val:>5} mm")

    typer.echo(
        "\nNote : hauteur_selle_bb = BB → hanche au PMB (hanche ≠ sommet selle,"
        " écart ≈ 30-50 mm selon selle/cuissard)."
    )

    if write:
        yaml_path = CONFIG_DIR / "bikes" / f"{bike}.yaml"
        if not yaml_path.exists():
            typer.echo(f"Erreur : {yaml_path} introuvable.", err=True)
            raise typer.Exit(1)
        updated = update_null_fields(yaml_path, display)
        if updated:
            typer.echo(f"\nOK: {len(updated)} champ(s) mis à jour dans {yaml_path} : {updated}")
        else:
            typer.echo("\nAucun champ null à mettre à jour (déjà renseignés).")
    else:
        typer.echo(
            f"\nAjoutez --write pour mettre à jour automatiquement les champs null"
            f" dans config/bikes/{bike}.yaml"
        )


if __name__ == "__main__":
    app()
