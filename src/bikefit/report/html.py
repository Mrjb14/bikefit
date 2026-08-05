"""Génération du rapport HTML (Jinja2 + Plotly)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from bikefit.equipment.rider import RiderProfile
from bikefit.kinematics.metrics import FitMetrics
from bikefit.recommend.models import FitReport

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "template.html.jinja2"

_ANGLE_COLS = [
    ("genou", "Genou", "#68d391"),
    ("hanche", "Hanche", "#f6ad55"),
    ("tronc", "Tronc", "#90cdf4"),
    ("coude", "Coude", "#fc8181"),
    ("cheville", "Cheville", "#b794f4"),
]

_ANNOTATION_KEYS = [
    "hauteur_selle",
    "recul_selle",
    "hauteur_prolongateurs",
    "longueur_prolongateurs",
    "cale_chaussure",
]

_PRIORITY_COLORS = {
    "haute": "#fc8181",
    "moyenne": "#fbd38d",
    "faible": "#90cdf4",
}


def _build_charts_json(angle_df: pd.DataFrame) -> str:
    """Retourne un objet JSON {col: {name, color, traces}} — un graphe par angle."""
    frames = angle_df["frame_idx"].tolist()
    charts: dict = {}
    for col, name, color in _ANGLE_COLS:
        if col not in angle_df.columns:
            continue
        vals = angle_df[col].tolist()
        charts[col] = {
            "name": name,
            "color": color,
            "traces": [
                {
                    "type": "scattergl",
                    "mode": "lines",
                    "x": frames,
                    "y": [None if (v != v) else round(v, 2) for v in vals],
                    "line": {"color": color, "width": 1.5},
                    "showlegend": False,
                }
            ],
        }
    return json.dumps(charts)


def _build_annotations(report: FitReport) -> dict:
    """Construit le dict d'annotations pour le schéma SVG du vélo."""
    result: dict = {
        key: {"color": "#4a5568", "delta": None, "direction": None} for key in _ANNOTATION_KEYS
    }
    for rec in report.recommandations:
        if rec.reglage in result:
            result[rec.reglage]["color"] = _PRIORITY_COLORS.get(rec.priorite.value, "#4a5568")
            result[rec.reglage]["delta"] = rec.delta_mm
            result[rec.reglage]["direction"] = rec.direction
    return result


def _build_rider_ctx(rider: RiderProfile | None, longueur_manivelle_mm: float = 172.5) -> dict:
    """Construit le contexte coureur pour le template (valeurs affichables)."""
    if rider is None:
        return {}
    return {
        "taille": rider.mensurations.taille,
        "poids_kg": rider.profil.poids_kg,
        "age": rider.profil.age,
        "flexibilite": rider.profil.flexibilite,
        "type_epreuve": rider.objectif.type_epreuve,
        "echeance": rider.objectif.echeance,
        "entrejambe": rider.mensurations.entrejambe,
        "selle_geo_mm": rider.saddle_height_geo_mm(longueur_manivelle_mm),
    }


def render(
    metrics: FitMetrics,
    report: FitReport,
    angle_df: pd.DataFrame,
    video_path: Path,
    output_path: Path,
    rider: RiderProfile | None = None,
    longueur_manivelle_mm: float = 172.5,
) -> None:
    """Génère le fichier HTML complet."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    tmpl = env.get_template(_TEMPLATE_NAME)

    html = tmpl.render(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        video_name=video_path.name,
        metrics=metrics,
        report=report,
        ann=_build_annotations(report),
        charts_json=_build_charts_json(angle_df),
        rider=_build_rider_ctx(rider, longueur_manivelle_mm),
    )
    output_path.write_text(html, encoding="utf-8")
