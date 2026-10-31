"""UI Streamlit — Phase 4 (MVP) : lance une analyse depuis le navigateur.

Réutilise `bikefit.cli.analyze` directement (même pipeline que le CLI, aucune
logique dupliquée) — measure-body/measure-setup restent CLI-only pour l'instant.

Lancement : uv run --extra ui streamlit run src/bikefit/ui/app.py
"""

from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime

import streamlit as st

from bikefit._paths import CONFIG_DIR, OUTPUTS_DIR, REPORTS_DIR, SOURCES_DIR
from bikefit.cli import analyze as run_analyze
from bikefit.ui.helpers import list_configs, parse_optional_float, save_upload

UPLOADS_DIR = SOURCES_DIR / "uploads"
METRICS_DIR = OUTPUTS_DIR / "metrics"

st.set_page_config(page_title="BikeFit — Analyse", layout="wide")
st.title("BikeFit — Analyse")
st.caption("Upload vidéo(s), lance l'analyse, consulte le rapport — sans passer par le terminal.")

bikes = list_configs(CONFIG_DIR, "bikes") or ["giant_trinity_2019"]
riders = list_configs(CONFIG_DIR, "rider") or ["mensurations"]

with st.form("analyze_form"):
    col_profil, col_face = st.columns(2)
    with col_profil:
        st.subheader("Vue de profil (obligatoire)")
        profile_video = st.file_uploader("Vidéo profil", type=["mp4", "mov", "avi"])
        side = st.selectbox("Côté filmé", ["right", "left"])
        start_raw = st.text_input("Début (s) — ignorer avant", placeholder="ex. 12")
        end_raw = st.text_input("Fin (s) — ignorer après", placeholder="ex. 70")
        marker_size_raw = st.text_input("Marqueur ArUco (mm, optionnel)", placeholder="ex. 100")
    with col_face:
        st.subheader("Vue de face (optionnelle)")
        front_video = st.file_uploader("Vidéo face", type=["mp4", "mov", "avi"])
        front_start_raw = st.text_input("Début face (s)", placeholder="ex. 8")
        front_end_raw = st.text_input("Fin face (s)", placeholder="ex. 65")
        front_marker_raw = st.text_input(
            "Marqueur ArUco face (mm, optionnel)",
            placeholder="ex. 100",
            help="Marqueur visible par la caméra de face, requis pour la largeur des coudes (mm).",
        )

    st.subheader("Configuration")
    col_bike, col_rider, col_profile = st.columns(3)
    with col_bike:
        bike = st.selectbox("Vélo", bikes)
    with col_rider:
        rider = st.selectbox("Coureur", riders)
    with col_profile:
        target_profile = st.selectbox("Profil cible", ["triathlon_XL", "clm_long", "clm_court"])

    save_session = st.checkbox("Sauvegarder la session", value=True)
    submitted = st.form_submit_button("Lancer l'analyse")

if submitted:
    if profile_video is None:
        st.error("La vidéo de profil est obligatoire.")
        st.stop()

    try:
        start_s = parse_optional_float(start_raw)
        end_s = parse_optional_float(end_raw)
        front_start_s = parse_optional_float(front_start_raw)
        front_end_s = parse_optional_float(front_end_raw)
        marker_size_mm = parse_optional_float(marker_size_raw)
        front_marker_mm = parse_optional_float(front_marker_raw)
    except ValueError:
        st.error("Les champs numériques (début/fin/marqueur) doivent être des nombres.")
        st.stop()

    profile_path = save_upload(profile_video, UPLOADS_DIR)
    front_path = save_upload(front_video, UPLOADS_DIR) if front_video is not None else None

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    report_path = REPORTS_DIR / f"{profile_path.stem}_{stamp}.html"
    output_path = METRICS_DIR / f"{profile_path.stem}_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = io.StringIO()
    failed = False
    with st.spinner("Analyse en cours (peut prendre plusieurs minutes)..."):
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_analyze(
                    video=str(profile_path),
                    front=str(front_path) if front_path else None,
                    side=side,
                    start=start_s,
                    end=end_s,
                    front_start=front_start_s,
                    front_end=front_end_s,
                    bike=bike,
                    profile=target_profile,
                    rider=rider,
                    marker_size=marker_size_mm,
                    front_marker_size=front_marker_mm,
                    output=output_path,
                    report=report_path,
                    save_session=save_session,
                )
        except Exception as exc:
            failed = True
            log.write(f"\n{exc.__class__.__name__}: {exc}")

    if failed:
        st.error("Échec de l'analyse — voir le journal ci-dessous.")
        st.text(log.getvalue())
        st.stop()

    st.success("Analyse terminée.")
    with st.expander("Journal"):
        st.text(log.getvalue())

    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        rapport = result["rapport"]
        st.subheader("Recommandations")
        recos = rapport.get("recommandations", [])
        if not recos:
            st.info("Aucune recommandation — position déjà dans les cibles.")
        for r in recos:
            delta = f"{r['delta_mm']:+.0f} mm" if r.get("delta_mm") is not None else "directionnel"
            st.markdown(
                f"**[{r['priorite'].upper()}]** {r['reglage']} — {r['direction']} ({delta})  \n"
                f"{r['motif']}"
            )
        if rapport.get("resume"):
            st.caption(rapport["resume"])

    if report_path.exists():
        st.subheader("Rapport complet")
        html = report_path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=1200, scrolling=True)
        st.download_button(
            "Télécharger le rapport HTML", html, file_name=report_path.name, mime="text/html"
        )
