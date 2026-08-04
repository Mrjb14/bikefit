"""Génération du rapport HTML (Jinja2 + Plotly)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from bikefit.kinematics.metrics import FitMetrics
from bikefit.recommend.models import FitReport

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "template.html.jinja2"

_ANGLE_COLS = [
    ("genou",   "Genou",   "#68d391"),
    ("hanche",  "Hanche",  "#f6ad55"),
    ("tronc",   "Tronc",   "#90cdf4"),
    ("coude",   "Coude",   "#fc8181"),
    ("cheville","Cheville","#b794f4"),
]


def _build_chart_json(angle_df: pd.DataFrame) -> str:
    """Construit les traces Plotly pour les angles sur l'ensemble des frames."""
    traces = []
    frames = angle_df["frame_idx"].tolist()
    for col, name, color in _ANGLE_COLS:
        if col not in angle_df.columns:
            continue
        vals = angle_df[col].tolist()
        traces.append({
            "type": "scattergl",
            "mode": "lines",
            "name": name,
            "x": frames,
            "y": [None if (v != v) else round(v, 2) for v in vals],  # NaN → None pour JSON
            "line": {"color": color, "width": 1.5},
        })
    return json.dumps(traces)


def render(
    metrics: FitMetrics,
    report: FitReport,
    angle_df: pd.DataFrame,
    video_path: Path,
    output_path: Path,
) -> None:
    """Génère le fichier HTML complet."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    tmpl = env.get_template(_TEMPLATE_NAME)

    html = tmpl.render(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        video_name=video_path.name,
        metrics=metrics,
        report=report,
        chart_json=_build_chart_json(angle_df),
    )
    output_path.write_text(html, encoding="utf-8")
