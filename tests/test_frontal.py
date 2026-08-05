"""Tests unitaires — métriques plan frontal sur squelettes synthétiques aux valeurs connues."""

import numpy as np
import pandas as pd

from bikefit.kinematics.frontal import _knee_deviation_angle, compute_frontal_metrics


class TestKneeDeviationAngle:
    def test_knee_on_line_is_zero(self) -> None:
        hip = np.array([[0.0, 0.0]])
        ankle = np.array([[0.0, 100.0]])
        knee = np.array([[0.0, 50.0]])
        assert abs(_knee_deviation_angle(hip, ankle, knee)[0]) < 0.01

    def test_knee_offset_sign_flips_with_direction(self) -> None:
        hip = np.array([[0.0, 0.0]])
        ankle = np.array([[0.0, 100.0]])
        knee_right = np.array([[10.0, 50.0]])
        knee_left = np.array([[-10.0, 50.0]])
        angle_right = _knee_deviation_angle(hip, ankle, knee_right)[0]
        angle_left = _knee_deviation_angle(hip, ankle, knee_left)[0]
        assert angle_right < 0
        assert angle_left > 0
        assert abs(angle_right + angle_left) < 0.01  # symétrique

    def test_magnitude_matches_arctan(self) -> None:
        # déviation de 10 px à mi-hauteur d'un segment de 100 px → atan(10/50)
        hip = np.array([[0.0, 0.0]])
        ankle = np.array([[0.0, 100.0]])
        knee = np.array([[10.0, 50.0]])
        expected = np.degrees(np.arctan2(10.0, 50.0))
        angle = _knee_deviation_angle(hip, ankle, knee)[0]
        assert abs(abs(angle) - expected) < 0.01

    def test_multiple_frames(self) -> None:
        hip = np.array([[0.0, 0.0], [0.0, 0.0]])
        ankle = np.array([[0.0, 100.0], [0.0, 100.0]])
        knee = np.array([[0.0, 50.0], [10.0, 50.0]])
        angles = _knee_deviation_angle(hip, ankle, knee)
        assert abs(angles[0]) < 0.01
        assert angles[1] < 0


class TestComputeFrontalMetrics:
    def _synthetic_df(self, n: int = 5) -> pd.DataFrame:
        """Squelette vue de face : hanches/genoux/chevilles alignés (0 déviation),
        coudes gauche/droit écartés de 200 px."""
        rows = []
        for i in range(n):
            row: dict = {"frame_idx": i, "timestamp_ms": i * 40, "detected": True}
            for j in range(33):
                row[f"lm_{j}_x"] = 0.5
                row[f"lm_{j}_y"] = 0.5
                row[f"lm_{j}_z"] = 0.0
                row[f"lm_{j}_vis"] = 1.0
            # jambe gauche (23 hanche, 25 genou, 27 cheville) alignée verticalement
            row["lm_23_x"], row["lm_23_y"] = 0.4, 0.4
            row["lm_25_x"], row["lm_25_y"] = 0.4, 0.6
            row["lm_27_x"], row["lm_27_y"] = 0.4, 0.8
            # jambe droite (24, 26, 28) alignée verticalement
            row["lm_24_x"], row["lm_24_y"] = 0.6, 0.4
            row["lm_26_x"], row["lm_26_y"] = 0.6, 0.6
            row["lm_28_x"], row["lm_28_y"] = 0.6, 0.8
            # coudes (13 gauche, 14 droit) écartés de 200 px (image 1000x1000)
            row["lm_13_x"], row["lm_13_y"] = 0.3, 0.5
            row["lm_14_x"], row["lm_14_y"] = 0.5, 0.5
            rows.append(row)
        return pd.DataFrame(rows)

    def test_aligned_legs_zero_deviation(self) -> None:
        df = self._synthetic_df()
        out = compute_frontal_metrics(df, width=1000, height=1000)
        assert abs(out["genou_deviation_gauche"].mean()) < 0.5
        assert abs(out["genou_deviation_droit"].mean()) < 0.5

    def test_elbow_width_px(self) -> None:
        df = self._synthetic_df()
        out = compute_frontal_metrics(df, width=1000, height=1000)
        assert abs(out["largeur_coudes_px"].mean() - 200.0) < 0.5

    def test_undetected_frames_are_nan(self) -> None:
        df = self._synthetic_df(n=3)
        df.loc[1, "detected"] = False
        out = compute_frontal_metrics(df, width=1000, height=1000)
        assert np.isnan(out.loc[1, "genou_deviation_gauche"])
        assert np.isnan(out.loc[1, "largeur_coudes_px"])
        assert not np.isnan(out.loc[0, "genou_deviation_gauche"])
