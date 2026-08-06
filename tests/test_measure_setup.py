"""Tests de l'estimation de setup vélo (fit de cercle de Kåsa + cotes BB).

Données synthétiques : la cheville décrit un cercle parfait autour d'un BB de
position connue, ce qui permet de vérifier les cotes au pixel près.
"""

import numpy as np
import pandas as pd
import pytest

from bikefit.kinematics.landmarks import BODY
from bikefit.measure.setup import _fit_circle, estimate_bike_setup

_W, _H = 1000, 1000
_R = BODY["right"]


def _setup_df(
    n: int = 120,
    bb_px: tuple[float, float] = (500.0, 700.0),
    radius_px: float = 150.0,
    hip_px: tuple[float, float] = (450.0, 300.0),
    wrist_px: tuple[float, float] = (800.0, 350.0),
) -> pd.DataFrame:
    """Cheville sur un cercle autour du BB ; hanche et poignet fixes."""
    theta = np.linspace(0, 4 * np.pi, n, endpoint=False)
    ankle_x = bb_px[0] + radius_px * np.cos(theta)
    ankle_y = bb_px[1] + radius_px * np.sin(theta)

    return pd.DataFrame(
        {
            "frame_idx": np.arange(n),
            "timestamp_ms": np.arange(n) * 33,
            "detected": np.ones(n, dtype=bool),
            f"lm_{_R['ankle']}_x": ankle_x / _W,
            f"lm_{_R['ankle']}_y": ankle_y / _H,
            f"lm_{_R['hip']}_x": np.full(n, hip_px[0] / _W),
            f"lm_{_R['hip']}_y": np.full(n, hip_px[1] / _H),
            f"lm_{_R['wrist']}_x": np.full(n, wrist_px[0] / _W),
            f"lm_{_R['wrist']}_y": np.full(n, wrist_px[1] / _H),
        }
    )


class TestFitCircle:
    def test_recovers_known_circle(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        cx, cy, r = _fit_circle(30 + 12 * np.cos(theta), -7 + 12 * np.sin(theta))
        assert cx == pytest.approx(30.0, abs=1e-6)
        assert cy == pytest.approx(-7.0, abs=1e-6)
        assert r == pytest.approx(12.0, abs=1e-6)

    def test_partial_arc_still_fits(self) -> None:
        # Un demi-tour suffit géométriquement à contraindre le cercle
        theta = np.linspace(0, np.pi, 30)
        cx, cy, r = _fit_circle(5 + 3 * np.cos(theta), 2 + 3 * np.sin(theta))
        assert cx == pytest.approx(5.0, abs=1e-6)
        assert cy == pytest.approx(2.0, abs=1e-6)
        assert r == pytest.approx(3.0, abs=1e-6)

    def test_robust_to_small_noise(self) -> None:
        rng = np.random.default_rng(0)
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        x = 100 + 40 * np.cos(theta) + rng.normal(0, 0.5, 200)
        y = 50 + 40 * np.sin(theta) + rng.normal(0, 0.5, 200)
        cx, cy, r = _fit_circle(x, y)
        assert cx == pytest.approx(100.0, abs=1.0)
        assert cy == pytest.approx(50.0, abs=1.0)
        assert r == pytest.approx(40.0, abs=1.0)


class TestEstimateBikeSetup:
    def test_finds_bb_at_circle_center(self) -> None:
        out = estimate_bike_setup(_setup_df(), mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["_bb_px"] == (500, 700)

    def test_crank_radius_in_mm(self) -> None:
        # rayon 150 px à 0.5 mm/px → 75 mm
        out = estimate_bike_setup(_setup_df(), mm_per_pixel=0.5, image_width=_W, image_height=_H)
        assert out["_rayon_cheville_mm"] == pytest.approx(75, abs=1)

    def test_saddle_height_is_bb_to_hip(self) -> None:
        # BB à y=700, hanche à y=300 → 400 px → 400 mm à 1 mm/px
        out = estimate_bike_setup(_setup_df(), mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["hauteur_selle_bb"] == pytest.approx(400, abs=2)

    def test_handlebar_height_is_bb_to_wrist(self) -> None:
        # BB à y=700, poignet à y=350 → 350 px
        out = estimate_bike_setup(_setup_df(), mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["hauteur_guidon_bb"] == pytest.approx(350, abs=2)

    def test_reach_is_horizontal_bb_to_wrist(self) -> None:
        # BB à x=500, poignet à x=800 → 300 px
        out = estimate_bike_setup(_setup_df(), mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["reach_prolongateurs"] == pytest.approx(300, abs=2)

    def test_reach_is_always_positive(self) -> None:
        # Poignet à gauche du BB (caméra de l'autre côté) : la cote reste une distance
        df = _setup_df(wrist_px=(200.0, 350.0))
        out = estimate_bike_setup(df, mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["reach_prolongateurs"] > 0

    def test_mm_per_pixel_scales_all_measures(self) -> None:
        df = _setup_df()
        a = estimate_bike_setup(df, mm_per_pixel=1.0, image_width=_W, image_height=_H)
        b = estimate_bike_setup(df, mm_per_pixel=2.0, image_width=_W, image_height=_H)
        for key in ("hauteur_selle_bb", "hauteur_guidon_bb", "reach_prolongateurs"):
            assert b[key] == pytest.approx(a[key] * 2, abs=2)

    def test_left_side_landmarks(self) -> None:
        lm = BODY["left"]
        n = 120
        theta = np.linspace(0, 4 * np.pi, n, endpoint=False)
        df = pd.DataFrame(
            {
                "frame_idx": np.arange(n),
                "timestamp_ms": np.arange(n) * 33,
                "detected": np.ones(n, dtype=bool),
                f"lm_{lm['ankle']}_x": (500 + 150 * np.cos(theta)) / _W,
                f"lm_{lm['ankle']}_y": (700 + 150 * np.sin(theta)) / _H,
                f"lm_{lm['hip']}_x": np.full(n, 0.45),
                f"lm_{lm['hip']}_y": np.full(n, 0.30),
                f"lm_{lm['wrist']}_x": np.full(n, 0.80),
                f"lm_{lm['wrist']}_y": np.full(n, 0.35),
            }
        )
        out = estimate_bike_setup(
            df, mm_per_pixel=1.0, image_width=_W, image_height=_H, side="left"
        )
        assert out["_bb_px"] == (500, 700)

    def test_ignores_undetected_frames(self) -> None:
        df = _setup_df()
        df.loc[0:20, "detected"] = False
        out = estimate_bike_setup(df, mm_per_pixel=1.0, image_width=_W, image_height=_H)
        assert out["_bb_px"] == (500, 700)

    def test_too_few_frames_raises(self) -> None:
        df = _setup_df(n=5)
        with pytest.raises(ValueError, match="insuffisamment"):
            estimate_bike_setup(df, mm_per_pixel=1.0, image_width=_W, image_height=_H)
