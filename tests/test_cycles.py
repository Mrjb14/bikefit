"""Tests de la détection de cycles de pédalage sur signaux synthétiques.

Convention y-bas (OpenCV) : le PMB est un MAXIMUM local de ankle_y
(cheville en bas de course = y grand). Cf. CLAUDE.md.
"""

import numpy as np
import pandas as pd

from bikefit.kinematics.cycles import detect_cycles
from bikefit.kinematics.landmarks import BODY


def _pedaling_df(
    n_cycles: float,
    fps: float = 30.0,
    rpm: float = 90.0,
    side: str = "right",
    detected: np.ndarray | None = None,
) -> pd.DataFrame:
    """Trajectoire de cheville sinusoïdale à cadence connue.

    rpm tours/min → période = 60/rpm secondes = fps * 60/rpm frames.
    """
    frames_per_cycle = fps * 60.0 / rpm
    n = int(round(n_cycles * frames_per_cycle))
    t = np.arange(n)
    # -cos → démarre en haut de course, premier maximum (PMB) à mi-période
    ankle_y = 0.5 - 0.2 * np.cos(2 * np.pi * t / frames_per_cycle)

    ankle_idx = BODY[side]["ankle"]  # type: ignore[index]
    det = np.ones(n, dtype=bool) if detected is None else detected
    return pd.DataFrame(
        {
            "frame_idx": t,
            "timestamp_ms": (t * 1000 / fps).astype(int),
            "detected": det,
            f"lm_{ankle_idx}_x": np.full(n, 0.5),
            f"lm_{ankle_idx}_y": ankle_y,
        }
    )


class TestDetectCycles:
    def test_counts_expected_cycles(self) -> None:
        # 6 périodes → 6 PMB → 5 intervalles entre PMB consécutifs
        df = _pedaling_df(n_cycles=6, fps=30.0, rpm=90.0)
        cycles = detect_cycles(df, "right", fps=30.0)
        assert len(cycles) == 5

    def test_cycle_length_matches_cadence(self) -> None:
        fps, rpm = 30.0, 90.0
        expected = fps * 60.0 / rpm  # 20 frames à 90 rpm / 30 fps
        df = _pedaling_df(n_cycles=8, fps=fps, rpm=rpm)
        cycles = detect_cycles(df, "right", fps=fps)

        lengths = [end - start for start, end in cycles]
        assert all(abs(length - expected) <= 1 for length in lengths)

    def test_cycles_are_contiguous_and_ordered(self) -> None:
        df = _pedaling_df(n_cycles=6)
        cycles = detect_cycles(df, "right", fps=30.0)
        for start, end in cycles:
            assert start < end
        # la fin d'un cycle est le début du suivant
        for i in range(len(cycles) - 1):
            assert cycles[i][1] == cycles[i + 1][0]

    def test_pmb_is_ankle_low_point(self) -> None:
        # Le début de chaque cycle doit tomber sur un maximum de ankle_y
        fps = 30.0
        df = _pedaling_df(n_cycles=6, fps=fps)
        ankle_y = df[f"lm_{BODY['right']['ankle']}_y"].to_numpy()
        cycles = detect_cycles(df, "right", fps=fps)

        for start, _ in cycles:
            window = ankle_y[max(0, start - 3) : start + 4]
            assert ankle_y[start] >= window.max() - 1e-9

    def test_left_side_uses_left_ankle(self) -> None:
        df = _pedaling_df(n_cycles=6, side="left")
        assert len(detect_cycles(df, "left", fps=30.0)) == 5

    def test_low_cadence_still_detected(self) -> None:
        # 50 rpm : les PMB sont plus espacés, min_dist ne doit pas les fusionner
        df = _pedaling_df(n_cycles=6, fps=30.0, rpm=50.0)
        assert len(detect_cycles(df, "right", fps=30.0)) == 5

    def test_flat_signal_returns_no_cycle(self) -> None:
        n = 200
        ankle_idx = BODY["right"]["ankle"]
        df = pd.DataFrame(
            {
                "frame_idx": np.arange(n),
                "timestamp_ms": np.arange(n) * 33,
                "detected": np.ones(n, dtype=bool),
                f"lm_{ankle_idx}_x": np.full(n, 0.5),
                f"lm_{ankle_idx}_y": np.full(n, 0.5),
            }
        )
        assert detect_cycles(df, "right", fps=30.0) == []

    def test_all_nan_returns_empty(self) -> None:
        df = _pedaling_df(n_cycles=6)
        df[f"lm_{BODY['right']['ankle']}_y"] = np.nan
        assert detect_cycles(df, "right", fps=30.0) == []

    def test_gaps_are_interpolated_not_fatal(self) -> None:
        # Quelques frames non détectées ne doivent pas casser la détection
        df = _pedaling_df(n_cycles=8)
        col = f"lm_{BODY['right']['ankle']}_y"
        df.loc[15:17, col] = np.nan
        cycles = detect_cycles(df, "right", fps=30.0)
        assert len(cycles) >= 5

    def test_too_short_returns_empty(self) -> None:
        # Moins de 2 PMB → aucun cycle complet mesurable
        df = _pedaling_df(n_cycles=0.5)
        assert detect_cycles(df, "right", fps=30.0) == []


class TestFrameIdxAlignment:
    """Les bornes renvoyées sont des frame_idx, pas des positions de tableau.

    aggregate_cycles découpe avec `angle_df["frame_idx"] >= start`. Si detect_cycles
    renvoyait des positions, tout DataFrame tronqué (options --start/--end) produirait
    des fenêtres décalées — silencieusement, car un masque vide donne des NaN que
    pandas ignore ensuite dans les moyennes.
    """

    def test_offset_dataframe_returns_frame_idx(self) -> None:
        df = _pedaling_df(n_cycles=6)
        offset = 558
        df["frame_idx"] = df["frame_idx"] + offset

        cycles = detect_cycles(df, "right", fps=30.0)
        assert cycles, "des cycles doivent être détectés"
        for start, end in cycles:
            assert start >= offset
            assert end <= int(df["frame_idx"].max())

    def test_offset_windows_select_real_rows(self) -> None:
        # Le test qui aurait attrapé le bug : chaque fenêtre doit sélectionner des lignes
        df = _pedaling_df(n_cycles=6)
        df["frame_idx"] = df["frame_idx"] + 558

        for start, end in detect_cycles(df, "right", fps=30.0):
            mask = (df["frame_idx"] >= start) & (df["frame_idx"] <= end)
            assert mask.sum() > 0

    def test_matches_positions_when_starting_at_zero(self) -> None:
        # Cas historique (vidéo entière) : frame_idx == position, résultat inchangé
        df = _pedaling_df(n_cycles=6)
        cycles = detect_cycles(df, "right", fps=30.0)
        frame_ids = df["frame_idx"].to_numpy()
        for start, end in cycles:
            assert frame_ids[start] == start
            assert frame_ids[end] == end
