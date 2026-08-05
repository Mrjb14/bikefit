"""Tests unitaires — angles sur squelettes synthétiques aux valeurs connues."""

import numpy as np
import pandas as pd

from bikefit.kinematics.angles import _trunk_angle, _vec_angle, compute_angles


def _pts(*coords: tuple[float, float]) -> list[np.ndarray]:
    return [np.array([[x, y]]) for x, y in coords]


class TestVecAngle:
    def test_90_degrees(self) -> None:
        # b en origine, a vers le haut, c vers la droite → 90°
        a = np.array([[0.0, -1.0]])
        b = np.array([[0.0, 0.0]])
        c = np.array([[1.0, 0.0]])
        assert abs(_vec_angle(a, b, c)[0] - 90.0) < 0.01

    def test_180_degrees(self) -> None:
        # points colinéaires → 180°
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        c = np.array([[2.0, 0.0]])
        assert abs(_vec_angle(a, b, c)[0] - 180.0) < 0.01

    def test_45_degrees(self) -> None:
        # v1 = [1, 0], v2 = [cos45, sin45] depuis le sommet b → angle = 45°
        b = np.array([[0.0, 0.0]])
        a = np.array([[1.0, 0.0]])
        c = np.array([[np.cos(np.radians(45)), np.sin(np.radians(45))]])
        angle = _vec_angle(a, b, c)[0]
        assert abs(angle - 45.0) < 0.01

    def test_nan_on_zero_vector(self) -> None:
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])  # a == b → vecteur nul
        c = np.array([[2.0, 0.0]])
        assert np.isnan(_vec_angle(a, b, c)[0])

    def test_multiple_frames(self) -> None:
        a = np.array([[0.0, 0.0], [0.0, -1.0]])
        b = np.array([[1.0, 0.0], [0.0, 0.0]])
        c = np.array([[2.0, 0.0], [1.0, 0.0]])
        angles = _vec_angle(a, b, c)
        assert abs(angles[0] - 180.0) < 0.01
        assert abs(angles[1] - 90.0) < 0.01


class TestTrunkAngle:
    def test_horizontal(self) -> None:
        # shoulder à droite de hip, même hauteur → 0°
        shoulder = np.array([[100.0, 50.0]])
        hip = np.array([[0.0, 50.0]])
        assert abs(_trunk_angle(shoulder, hip)[0] - 0.0) < 0.01

    def test_45_degrees(self) -> None:
        # shoulder au-dessus et à droite (y image vers le bas → shoulder.y < hip.y)
        shoulder = np.array([[100.0, 0.0]])
        hip = np.array([[0.0, 100.0]])
        angle = _trunk_angle(shoulder, hip)[0]
        assert abs(angle - 45.0) < 0.5

    def test_clm_position(self) -> None:
        # Position CLM typique : tronc ~15° / horizontale
        import math

        dx = 100.0
        dy = -math.tan(math.radians(15)) * dx  # dy négatif = shoulder plus haut que hip
        shoulder = np.array([[dx, 100.0 + dy]])
        hip = np.array([[0.0, 100.0]])
        angle = _trunk_angle(shoulder, hip)[0]
        assert abs(angle - 15.0) < 1.0


class TestComputeAngles:
    def _synthetic_df(self, n: int = 5) -> pd.DataFrame:
        """DataFrame synthétique avec un squelette à angles connus."""
        # Positions normalisées pour un squelette droit côté right :
        # hanche (24) : (0.5, 0.4), genou (26) : (0.5, 0.6), cheville (28) : (0.5, 0.8)
        # → genou angle = 180° (jambe droite)
        rows = []
        for i in range(n):
            row: dict = {"frame_idx": i, "timestamp_ms": i * 33, "detected": True}
            for j in range(33):
                row[f"lm_{j}_x"] = 0.5
                row[f"lm_{j}_y"] = 0.5
                row[f"lm_{j}_z"] = 0.0
                row[f"lm_{j}_vis"] = 1.0
            # hanche (24), genou (26), cheville (28) alignés verticalement
            row["lm_24_x"] = 0.5
            row["lm_24_y"] = 0.2  # hanche
            row["lm_26_x"] = 0.5
            row["lm_26_y"] = 0.5  # genou
            row["lm_28_x"] = 0.5
            row["lm_28_y"] = 0.8  # cheville
            rows.append(row)
        return pd.DataFrame(rows)

    def test_straight_leg_180(self) -> None:
        df = self._synthetic_df()
        angle_df = compute_angles(df, "right", width=1000, height=1000)
        assert abs(angle_df["genou"].mean() - 180.0) < 1.0

    def test_undetected_frames_are_nan(self) -> None:
        df = self._synthetic_df(n=3)
        df.loc[1, "detected"] = False
        angle_df = compute_angles(df, "right", width=1000, height=1000)
        assert np.isnan(angle_df.loc[1, "genou"])
        assert not np.isnan(angle_df.loc[0, "genou"])
