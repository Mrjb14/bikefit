"""CLI BikeFit — Phase 1, 2 & 3B."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    import pandas as pd

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

# ── Colonnes affichées dans le HUD des vidéos de debug (colonne, libellé, unité) ─
_SIDE_HUD_COLUMNS: list[tuple[str, str, str]] = [
    ("genou", "Genou", "°"),
    ("hanche", "Hanche", "°"),
    ("tronc", "Tronc", "°"),
    ("coude", "Coude", "°"),
    ("cheville", "Cheville", "°"),
]
_FRONT_HUD_COLUMNS: list[tuple[str, str, str]] = [
    ("genou_deviation_gauche", "Dev. genou G", "°"),
    ("genou_deviation_droit", "Dev. genou D", "°"),
    ("largeur_coudes_px", "Largeur coudes", "px"),
]


def _trim_landmarks(
    df: pd.DataFrame,
    fps: float,
    start_s: float | None,
    end_s: float | None,
    label: str,
) -> pd.DataFrame:
    """Restreint les landmarks à l'intervalle [start_s, end_s] (secondes).

    Appliqué APRÈS le cache : les landmarks de la vidéo entière restent calculés une
    seule fois, changer les bornes se rejoue instantanément (principe de cache du
    projet). `frame_idx` est conservé tel quel — c'est la clé de découpage utilisée
    en aval par aggregate_cycles.
    """
    if start_s is None and end_s is None:
        return df

    lo = int((start_s or 0.0) * fps)
    hi = int(end_s * fps) if end_s is not None else None

    mask = df["frame_idx"] >= lo
    if hi is not None:
        mask &= df["frame_idx"] < hi
    trimmed = df[mask]

    if trimmed.empty:
        typer.echo(
            f"Erreur : l'intervalle demandé ne contient aucune frame ({label}).",
            err=True,
        )
        raise typer.Exit(1)

    kept_s = len(trimmed) / fps if fps > 0 else 0.0
    typer.echo(
        f"  Découpage {label} : {len(trimmed)}/{len(df)} frames conservées"
        f" (~{kept_s:.0f}s, de {lo / fps:.0f}s)"
    )
    return trimmed


@app.command()
def analyze(
    video: Annotated[
        str,
        typer.Argument(
            help="Vidéo de profil : fichier (mp4/mov), flux MJPEG (http://...) ou webcam (webcam:0)"
        ),
    ],
    duration: Annotated[
        float, typer.Option(help="Durée de capture (s) si `video` est un flux réseau ou une webcam")
    ] = 15.0,
    front: Annotated[
        str | None,
        typer.Option(
            help="Vidéo de face : fichier, flux MJPEG (http://...) ou webcam (webcam:0),"
            " optionnel — Phase 3C"
        ),
    ] = None,
    front_duration: Annotated[
        float,
        typer.Option(help="Durée de capture (s) si --front est un flux réseau ou une webcam"),
    ] = 15.0,
    start: Annotated[
        float | None,
        typer.Option(help="Ignore les N premières secondes de la vidéo de profil (fichier)"),
    ] = None,
    end: Annotated[
        float | None,
        typer.Option(help="Ignore la vidéo de profil au-delà de N secondes (fichier)"),
    ] = None,
    front_start: Annotated[
        float | None,
        typer.Option(help="Ignore les N premières secondes de la vidéo de face (fichier)"),
    ] = None,
    front_end: Annotated[
        float | None,
        typer.Option(help="Ignore la vidéo de face au-delà de N secondes (fichier)"),
    ] = None,
    debug_video: Annotated[
        Path | None,
        typer.Option(help="Vidéo de debug (squelette + angles superposés) pour la vue de profil"),
    ] = None,
    front_debug_video: Annotated[
        Path | None,
        typer.Option(help="Vidéo de debug (squelette + métriques superposés) pour la vue de face"),
    ] = None,
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
    from bikefit._paths import REPORTS_DIR
    from bikefit.equipment import load_bike_profile, load_rider_profile
    from bikefit.ingest import VideoReader
    from bikefit.ingest.reader import _is_stream_url, _is_webcam_source, _same_source
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

    video_is_live = _is_stream_url(video) or _is_webcam_source(video)
    video_path = Path(video)

    if not video_is_live and not video_path.exists():
        typer.echo(f"Erreur : vidéo introuvable : {video}", err=True)
        raise typer.Exit(1)

    if side not in ("right", "left"):
        typer.echo("Erreur : --side doit être 'right' ou 'left'", err=True)
        raise typer.Exit(1)
    filmed_side: Side = "right" if side == "right" else "left"

    if front is not None and _same_source(video, front):
        typer.echo(
            "Erreur : --front pointe vers la même source que la vidéo de profil"
            " (fichier, flux ou webcam) — impossible de lancer les deux vues sur"
            " la même caméra.",
            err=True,
        )
        raise typer.Exit(1)

    # ── Ingest ───────────────────────────────────────────────────────────────
    reader = VideoReader(video)
    kind_note = " (webcam)" if reader.is_webcam else " (flux réseau)" if reader.is_stream else ""
    typer.echo(
        f"Vidéo : {reader.width}x{reader.height}"
        f" @ {reader.fps:.1f} fps, {reader.frame_count} frames{kind_note}"
    )

    # ── Pose (avec cache Parquet, sauf source live) ──────────────────────────
    cache = LandmarkCache(CACHE_DIR / "landmarks")
    with MediaPipePoseEstimator() as estimator:
        if video_is_live:
            typer.echo(f"Capture {duration:.0f}s...")
            if debug_video is not None:
                from functools import partial

                from bikefit.debug import render_debug_video_live

                metrics_fn = partial(
                    compute_angles, side=filmed_side, width=reader.width, height=reader.height
                )
                df = render_debug_video_live(
                    reader,
                    estimator,
                    metrics_fn,
                    _SIDE_HUD_COLUMNS,
                    debug_video,
                    max_duration_s=duration,
                )
                typer.echo(f"Vidéo de debug -> {debug_video}")
            else:
                df = compute_landmarks(reader, estimator, max_duration_s=duration)
        else:
            if no_cache:
                from bikefit.pose.cache import _video_key

                key = _video_key(video_path)
                cache_file = CACHE_DIR / "landmarks" / f"{video_path.stem}_{key}.parquet"
                if cache_file.exists():
                    cache_file.unlink()
                    typer.echo("Cache supprimé.")

            df = cache.get_or_compute(video_path, reader, estimator)

    if not video_is_live:
        df = _trim_landmarks(df, reader.fps, start, end, "profil")

    detection_rate = df["detected"].mean()
    typer.echo(f"Taux de détection : {detection_rate:.1%}")

    # ── Kinematics ───────────────────────────────────────────────────────────
    df_filtered = filter_landmarks(df, reader.fps)
    angle_df = compute_angles(df_filtered, filmed_side, reader.width, reader.height)
    cycles = detect_cycles(df_filtered, filmed_side, reader.fps)
    typer.echo(f"Cycles détectés : {len(cycles)}")

    if len(cycles) < 3:
        typer.echo("Avertissement : moins de 3 cycles — résultats peu fiables.", err=True)

    # ── Vidéo de debug (fichier : rejoue le fichier avec les angles lissés) ──
    if debug_video is not None and not video_is_live:
        from bikefit.debug import render_debug_video_from_landmarks

        typer.echo("Génération de la vidéo de debug...")
        render_debug_video_from_landmarks(
            video_path, df_filtered, angle_df, _SIDE_HUD_COLUMNS, debug_video
        )
        typer.echo(f"Vidéo de debug -> {debug_video}")

    metrics = aggregate_cycles(angle_df, cycles)

    # ── Calibration ArUco (optionnelle) ──────────────────────────────────────
    if marker_size is not None:
        if video_is_live:
            typer.echo(
                "Avertissement : calibration ArUco non supportée pour un flux réseau ou"
                " une webcam (--marker-size ignoré) — utiliser un fichier vidéo.",
                err=True,
            )
        else:
            from bikefit.calibration import calibrate_from_video

            typer.echo(f"Calibration ArUco (marqueur {marker_size:.0f} mm)...")
            calib = calibrate_from_video(str(video_path), marker_size_mm=marker_size)
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
        if front_reader.is_webcam:
            front_kind_note = " (webcam)"
        elif front_reader.is_stream:
            front_kind_note = " (flux réseau)"
        else:
            front_kind_note = ""
        typer.echo(
            f"  {front_reader.width}x{front_reader.height} @ {front_reader.fps:.1f} fps"
            f"{front_kind_note}"
        )

        with MediaPipePoseEstimator() as front_estimator:
            if front_reader.is_live:
                typer.echo(f"  Capture {front_duration:.0f}s...")
                if front_debug_video is not None:
                    from functools import partial

                    from bikefit.debug import render_debug_video_live

                    front_metrics_fn = partial(
                        compute_frontal_metrics,
                        width=front_reader.width,
                        height=front_reader.height,
                    )
                    front_df = render_debug_video_live(
                        front_reader,
                        front_estimator,
                        front_metrics_fn,
                        _FRONT_HUD_COLUMNS,
                        front_debug_video,
                        max_duration_s=front_duration,
                    )
                    typer.echo(f"  Vidéo de debug -> {front_debug_video}")
                else:
                    front_df = compute_landmarks(
                        front_reader, front_estimator, max_duration_s=front_duration
                    )
            else:
                front_cache = LandmarkCache(CACHE_DIR / "landmarks")
                front_df = front_cache.get_or_compute(Path(front), front_reader, front_estimator)

        if not front_reader.is_live:
            front_df = _trim_landmarks(front_df, front_reader.fps, front_start, front_end, "face")

        front_detection_rate = front_df["detected"].mean()
        typer.echo(f"  Taux de détection : {front_detection_rate:.1%}")

        front_filtered = filter_landmarks(front_df, front_reader.fps)
        frontal_df = compute_frontal_metrics(
            front_filtered, front_reader.width, front_reader.height
        )

        if front_debug_video is not None and not front_reader.is_live:
            from bikefit.debug import render_debug_video_from_landmarks

            typer.echo("  Génération de la vidéo de debug (vue de face)...")
            render_debug_video_from_landmarks(
                Path(front), front_filtered, frontal_df, _FRONT_HUD_COLUMNS, front_debug_video
            )
            typer.echo(f"  Vidéo de debug -> {front_debug_video}")

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
    # Par défaut dans REPORTS_DIR, jamais à côté de la vidéo source : une analyse
    # ne doit pas polluer le dossier des entrées (cf. data/sources vs data/outputs).
    if report:
        report_path = report
    else:
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        stem = "live" if video_is_live else video_path.stem
        report_path = REPORTS_DIR / f"{stem}_{stamp}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
    render_html(
        metrics,
        fit_report,
        angle_df,
        video_path,
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
                video_path=video if video_is_live else str(video_path.resolve()),
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
    from bikefit._paths import ANNOTATED_DIR, CONFIG_DIR
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

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    annotated = ANNOTATED_DIR / f"{photo.stem}_annotated.jpg"
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
    from bikefit._paths import ANNOTATED_DIR, CONFIG_DIR
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

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    annotated = ANNOTATED_DIR / f"{video.stem}_annotated.jpg"
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
