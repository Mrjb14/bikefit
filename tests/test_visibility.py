"""Tests du filtrage par visibilité des landmarks.

Enjeu : MediaPipe infère les points masqués (côté opposé à la caméra typiquement à
0.2-0.3 de visibilité). Sans filtrage, un angle calculé sur ces positions devinées
ressort avec le même statut qu'une mesure réelle — mesuré sur une vraie capture :
genou gauche à 0.32 de visibilité en vue de face, 94 % des frames sous le seuil.
"""

import numpy as np
import pandas as pd
import pytest

from bikefit.kinematics.angles import compute_angles
from bikefit.kinematics.frontal import compute_frontal_metrics
from bikefit.kinematics.landmarks import BODY
from bikefit.kinematics.visibility import (
    VISIBILITY_THRESHOLD,
    landmark_visibility,
    mean_visibility,
    reliability_mask,
)


def _df(n: int = 4, vis: float = 1.0) -> pd.DataFrame:
    """Squelette droit (genou à 180°), visibilité uniforme paramétrable."""
    rows = []
    for i in range(n):
        row: dict = {"frame_idx": i, "timestamp_ms": i * 33, "detected": True}
        for j in range(33):
            row[f"lm_{j}_x"] = 0.5
            row[f"lm_{j}_y"] = 0.5
            row[f"lm_{j}_z"] = 0.0
            row[f"lm_{j}_vis"] = vis
        # hanche(24)/genou(26)/cheville(28) alignés verticalement -> 180°
        row["lm_24_y"], row["lm_26_y"], row["lm_28_y"] = 0.2, 0.5, 0.8
        rows.append(row)
    return pd.DataFrame(rows)


class TestHelpers:
    def test_missing_column_treated_as_visible(self) -> None:
        # Les DataFrames sans score (tests, sources exotiques) restent exploitables
        df = pd.DataFrame({"frame_idx": [0, 1]})
        assert np.all(landmark_visibility(df, 26) == 1.0)

    def test_mask_requires_all_landmarks(self) -> None:
        df = _df(n=2)
        df.loc[:, "lm_26_vis"] = 0.1  # un seul point faible suffit à disqualifier
        assert not reliability_mask(df, (24, 26, 28)).any()

    def test_mask_true_when_all_above(self) -> None:
        assert reliability_mask(_df(n=2, vis=0.9), (24, 26, 28)).all()

    def test_nan_counts_as_unreliable(self) -> None:
        df = _df(n=2)
        df.loc[:, "lm_26_vis"] = np.nan
        assert not reliability_mask(df, (24, 26, 28)).any()

    def test_mean_visibility_uses_weakest_link(self) -> None:
        # 0.9 / 0.9 / 0.3 -> la moyenne doit refléter le maillon faible (0.3)
        df = _df(n=3, vis=0.9)
        df.loc[:, "lm_28_vis"] = 0.3
        assert mean_visibility(df, (24, 26, 28)) == pytest.approx(0.3, abs=1e-6)


class TestComputeAnglesFiltering:
    def test_visible_angles_are_kept(self) -> None:
        out = compute_angles(_df(vis=0.95), "right", width=1000, height=1000)
        assert out["genou"].notna().all()
        assert abs(out["genou"].mean() - 180.0) < 1.0

    def test_low_visibility_becomes_nan(self) -> None:
        df = _df(vis=0.95)
        df.loc[:, "lm_26_vis"] = 0.2  # genou droit inféré
        out = compute_angles(df, "right", width=1000, height=1000)
        assert out["genou"].isna().all()

    def test_only_affected_angle_is_dropped(self) -> None:
        # Cheville faible -> l'angle cheville tombe, le tronc (épaule+hanche) survit
        df = _df(vis=0.95)
        df.loc[:, "lm_28_vis"] = 0.1
        out = compute_angles(df, "right", width=1000, height=1000)
        assert out["cheville"].isna().all()
        assert out["tronc"].notna().all()

    def test_threshold_is_configurable(self) -> None:
        df = _df(vis=0.4)
        assert compute_angles(df, "right", 1000, 1000)["genou"].isna().all()
        relaxed = compute_angles(df, "right", 1000, 1000, visibility_threshold=0.3)
        assert relaxed["genou"].notna().all()

    def test_hidden_side_does_not_affect_filmed_side(self) -> None:
        # Cas réel : on filme la droite, la gauche est inférée à 0.27
        df = _df(vis=0.95)
        for name in ("knee", "ankle", "elbow", "wrist"):
            df.loc[:, f"lm_{BODY['left'][name]}_vis"] = 0.27
        out = compute_angles(df, "right", width=1000, height=1000)
        assert out["genou"].notna().all(), "le côté filmé reste exploitable"


class TestFrontalFiltering:
    def _front_df(self, n: int = 4, vis: float = 1.0) -> pd.DataFrame:
        df = _df(n=n, vis=vis)
        for side, sign in (("left", -1), ("right", 1)):
            df[f"lm_{BODY[side]['hip']}_x"] = 0.5 + sign * 0.1
            df[f"lm_{BODY[side]['knee']}_x"] = 0.5 + sign * 0.1
            df[f"lm_{BODY[side]['ankle']}_x"] = 0.5 + sign * 0.1
        return df

    def test_kept_when_visible(self) -> None:
        out = compute_frontal_metrics(self._front_df(vis=0.9), width=1000, height=1000)
        assert out["genou_deviation_gauche"].notna().all()

    def test_dropped_when_knee_inferred(self) -> None:
        df = self._front_df(vis=0.9)
        df.loc[:, f"lm_{BODY['left']['knee']}_vis"] = 0.32  # valeur mesurée en réel
        out = compute_frontal_metrics(df, width=1000, height=1000)
        assert out["genou_deviation_gauche"].isna().all()
        assert out["genou_deviation_droit"].notna().all(), "l'autre genou reste valide"

    def test_elbow_width_dropped_when_one_elbow_inferred(self) -> None:
        df = self._front_df(vis=0.9)
        df.loc[:, f"lm_{BODY['left']['elbow']}_vis"] = 0.1
        out = compute_frontal_metrics(df, width=1000, height=1000)
        assert out["largeur_coudes_px"].isna().all()


def test_threshold_default_is_documented_value() -> None:
    assert VISIBILITY_THRESHOLD == 0.5
