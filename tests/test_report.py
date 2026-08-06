"""Tests du module report — génération HTML (Jinja2), données synthétiques."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from bikefit.kinematics.metrics import AngleStats, FitMetrics
from bikefit.recommend.models import FitReport, Priorite, Recommendation
from bikefit.report import render_html

# ── Helpers ───────────────────────────────────────────────────────────────────


def _stats(mean: float = 145.0) -> AngleStats:
    return AngleStats(mean=mean, std=1.2, min=mean - 5, max=mean + 5, n_cycles=10)


def _metrics(**overrides) -> FitMetrics:
    base = {
        "genou_extension_pmb": _stats(150.0),
        "genou_flexion_pmh": _stats(71.0),
        "hanche_fermee_pmh": _stats(55.0),
        "tronc": _stats(16.0),
        "coude": _stats(96.0),
        "cheville_pmb": _stats(138.0),
        "n_cycles_total": 10,
        "n_cycles_used": 10,
        "detection_rate": 1.0,
    }
    base.update(overrides)
    return FitMetrics(**base)


def _report(recs: list[Recommendation] | None = None, resume: str = "Résumé de test.") -> FitReport:
    recs = recs or []
    return FitReport(
        profil_utilise="triathlon_XL",
        recommandations=recs,
        angles_hors_cible=[r.reglage for r in recs],
        angles_ok=[],
        resume=resume,
    )


def _angle_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "genou": [140.0, 145.0, 150.0, float("nan")],
            "hanche": [50.0, 52.0, 55.0, 53.0],
            "tronc": [16.0, 16.5, 17.0, 16.8],
            "coude": [95.0, 96.0, 97.0, 96.5],
            "cheville": [130.0, 135.0, 138.0, 132.0],
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRenderHtml:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("video.mp4"), out)
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!")

    def test_video_name_and_profile_present(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("ma_video_test.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "ma_video_test.mp4" in html
        assert "triathlon_XL" in html

    def test_angle_stats_rendered(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        metrics = _metrics(genou_extension_pmb=_stats(152.3))
        render_html(metrics, _report(), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "152.3" in html

    def test_recommendation_motif_rendered(self, tmp_path: Path) -> None:
        rec = Recommendation(
            reglage="hauteur_selle",
            direction="descendre",
            delta_mm=5,
            motif="La selle est probablement trop haute.",
            priorite=Priorite.haute,
        )
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report([rec]), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "La selle est probablement trop haute." in html
        assert "5" in html

    def test_no_recommendation_resume(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        report = _report(resume="Position dans les plages cibles.")
        render_html(_metrics(), report, _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "Position dans les plages cibles." in html

    def test_charts_json_embeds_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        match = re.search(r"const chartsData = (\{.*?\});", html, re.DOTALL)
        assert match is not None
        charts = json.loads(match.group(1))
        assert "genou" in charts
        assert charts["genou"]["traces"][0]["y"][:3] == [140.0, 145.0, 150.0]
        # NaN converti en null JSON, pas en NaN littéral (invalide en JSON strict)
        assert charts["genou"]["traces"][0]["y"][3] is None

    def test_frontal_metrics_absent_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "Déviation genoux" not in html

    def test_frontal_metrics_rendered_when_present(self, tmp_path: Path) -> None:
        metrics = _metrics(
            genou_deviation_gauche=_stats(2.5),
            genou_deviation_droit=_stats(-1.8),
            largeur_coudes=_stats(410.0),
        )
        out = tmp_path / "rapport.html"
        render_html(metrics, _report(), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "Déviation genoux" in html
        assert "410" in html


class TestAutoescape:
    """Régression : autoescape doit être actif (cf. fix XSS report/html.py)."""

    def test_motif_html_is_escaped(self, tmp_path: Path) -> None:
        rec = Recommendation(
            reglage="hauteur_selle",
            direction="descendre",
            delta_mm=5,
            motif='<script>alert(1)</script> & "quotes"',
            priorite=Priorite.haute,
        )
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report([rec]), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_video_filename_is_escaped(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        malicious_name = Path("<img src=x onerror=alert(1)>.mp4")
        render_html(_metrics(), _report(), _angle_df(), malicious_name, out)
        html = out.read_text(encoding="utf-8")
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;img" in html

    def test_charts_json_still_valid_js_despite_autoescape(self, tmp_path: Path) -> None:
        """charts_json contient des guillemets/accolades — doit rester du JSON brut (|safe),
        pas échappé en entités HTML qui casseraient le <script>."""
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("v.mp4"), out)
        html = out.read_text(encoding="utf-8")
        match = re.search(r"const chartsData = (\{.*?\});", html, re.DOTALL)
        assert match is not None
        json.loads(match.group(1))  # ne lève pas — JSON valide, pas d'entités &quot; etc.


class TestRiderContext:
    def test_no_rider_renders_without_crash(self, tmp_path: Path) -> None:
        out = tmp_path / "rapport.html"
        render_html(_metrics(), _report(), _angle_df(), Path("v.mp4"), out, rider=None)
        assert out.exists()
