"""Tests du filtre Savitzky-Golay sur signaux synthétiques.

Cette brique conditionne le jalon ±2° (cf. CLAUDE.md principe n°1) : elle doit
lisser le bruit sans déformer le signal ni décaler les extrema (le PMB est
détecté par maximum local juste après).
"""

import numpy as np
import pandas as pd

from bikefit.kinematics.filter import _sg_window, filter_landmarks


def _df(y: np.ndarray, detected: np.ndarray | None = None) -> pd.DataFrame:
    """DataFrame minimal : un seul landmark (0), coordonnées x constantes."""
    n = len(y)
    det = np.ones(n, dtype=bool) if detected is None else detected
    return pd.DataFrame(
        {
            "frame_idx": np.arange(n),
            "timestamp_ms": np.arange(n) * 33,
            "detected": det,
            "lm_0_x": np.full(n, 0.5),
            "lm_0_y": y,
        }
    )


class TestSgWindow:
    def test_window_is_always_odd(self) -> None:
        # savgol_filter exige une fenêtre impaire
        for fps in (15.0, 20.0, 24.0, 25.0, 30.0, 50.0, 60.0, 120.0):
            assert _sg_window(fps) % 2 == 1

    def test_window_has_floor(self) -> None:
        # Un flux très lent ne doit pas donner une fenêtre dégénérée (< polyorder)
        assert _sg_window(1.0) >= 5

    def test_window_grows_with_fps(self) -> None:
        assert _sg_window(120.0) > _sg_window(30.0)


class TestFilterLandmarks:
    def test_reduces_noise_on_sine(self) -> None:
        n = 300
        t = np.linspace(0, 6 * np.pi, n)
        clean = 0.5 + 0.2 * np.sin(t)
        rng = np.random.default_rng(42)
        noisy = clean + rng.normal(0, 0.01, n)

        out = filter_landmarks(_df(noisy), fps=30.0)
        filtered = out["lm_0_y"].to_numpy()

        err_before = np.abs(noisy - clean).mean()
        err_after = np.abs(filtered - clean).mean()
        assert err_after < err_before

    def test_preserves_signal_amplitude(self) -> None:
        # Un lissage trop agressif écraserait l'amplitude → fausserait les angles
        n = 300
        t = np.linspace(0, 6 * np.pi, n)
        clean = 0.5 + 0.2 * np.sin(t)

        out = filter_landmarks(_df(clean), fps=30.0)
        filtered = out["lm_0_y"].to_numpy()

        amp_in = clean.max() - clean.min()
        amp_out = filtered.max() - filtered.min()
        assert abs(amp_out - amp_in) / amp_in < 0.05

    def test_undetected_frames_stay_nan(self) -> None:
        n = 100
        y = np.full(n, 0.5)
        detected = np.ones(n, dtype=bool)
        detected[40:45] = False
        y[40:45] = np.nan

        out = filter_landmarks(_df(y, detected), fps=30.0)
        filtered = out["lm_0_y"].to_numpy()

        assert np.isnan(filtered[40:45]).all()
        assert not np.isnan(filtered[:40]).any()

    def test_interpolates_across_gaps_before_filtering(self) -> None:
        # Un trou au milieu ne doit pas contaminer les frames voisines avec des NaN
        n = 100
        y = 0.5 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n))
        detected = np.ones(n, dtype=bool)
        detected[50] = False
        y[50] = np.nan

        out = filter_landmarks(_df(y, detected), fps=30.0)
        filtered = out["lm_0_y"].to_numpy()

        assert np.isnan(filtered[50])
        assert not np.isnan(filtered[49])
        assert not np.isnan(filtered[51])

    def test_too_few_frames_returns_unchanged(self) -> None:
        # Moins de frames que la fenêtre → savgol impossible, on renvoie tel quel
        y = np.array([0.5, 0.6, 0.7])
        df = _df(y)
        out = filter_landmarks(df, fps=30.0)
        np.testing.assert_array_equal(out["lm_0_y"].to_numpy(), y)

    def test_does_not_mutate_input(self) -> None:
        y = 0.5 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, 200))
        df = _df(y.copy())
        original = df["lm_0_y"].to_numpy().copy()
        filter_landmarks(df, fps=30.0)
        np.testing.assert_array_equal(df["lm_0_y"].to_numpy(), original)
