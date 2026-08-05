"""Tests unitaires — agrégation des métriques plan frontal sur cycles synthétiques."""

import numpy as np
import pandas as pd
import pytest

from bikefit.kinematics.aggregate import aggregate_frontal_cycles


def _synthetic_frontal_df(n: int = 20) -> pd.DataFrame:
    """2 cycles de 10 frames, déviation genou constante (gauche=+3°, droit=-2°),
    largeur coudes constante à 150 px."""
    return pd.DataFrame(
        {
            "frame_idx": np.arange(n),
            "timestamp_ms": np.arange(n) * 40,
            "genou_deviation_gauche": np.full(n, 3.0),
            "genou_deviation_droit": np.full(n, -2.0),
            "largeur_coudes_px": np.full(n, 150.0),
        }
    )


def test_aggregate_basic_values() -> None:
    df = _synthetic_frontal_df()
    cycles = [(0, 9), (10, 19)]
    result = aggregate_frontal_cycles(df, cycles)

    assert result["genou_deviation_gauche"].mean == pytest.approx(3.0, abs=0.01)
    assert result["genou_deviation_droit"].mean == pytest.approx(-2.0, abs=0.01)
    assert result["genou_deviation_gauche"].n_cycles == 2
    assert "largeur_coudes" not in result  # pas de mm_per_pixel fourni


def test_aggregate_with_calibration_converts_to_mm() -> None:
    df = _synthetic_frontal_df()
    cycles = [(0, 9), (10, 19)]
    result = aggregate_frontal_cycles(df, cycles, mm_per_pixel=0.5)

    assert result["largeur_coudes"].mean == pytest.approx(75.0, abs=0.01)  # 150px * 0.5mm/px


def test_cycle_with_too_many_nan_is_skipped() -> None:
    df = _synthetic_frontal_df()
    df.loc[0:8, "genou_deviation_gauche"] = np.nan  # 9/10 frames NaN sur le 1er cycle
    cycles = [(0, 9), (10, 19)]
    result = aggregate_frontal_cycles(df, cycles)
    assert result["genou_deviation_gauche"].n_cycles == 1


def test_no_valid_cycle_raises() -> None:
    df = _synthetic_frontal_df(n=5)
    df["genou_deviation_gauche"] = np.nan
    with pytest.raises(ValueError):
        aggregate_frontal_cycles(df, [(0, 4)])
