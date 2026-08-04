"""Tests géométriques sur configurations synthétiques connues.

Le calcul d'angle 3 points est LA brique critique : on le verrouille ici
avant tout le reste (cf. CLAUDE.md, principe n°1).
"""

import numpy as np


def angle_3pts(a, b, c):
    """Duplique scripts/spike_bikefit.py — sera importé de bikefit.kinematics en Phase 1."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def test_angle_droit():
    assert angle_3pts([0, 1], [0, 0], [1, 0]) == 90.0


def test_angle_plat():
    assert angle_3pts([-1, 0], [0, 0], [1, 0]) == 180.0


def test_angle_45():
    assert abs(angle_3pts([1, 1], [0, 0], [1, 0]) - 45.0) < 1e-9


def test_invariance_echelle():
    a1 = angle_3pts([0, 2], [0, 0], [3, 1])
    a2 = angle_3pts([0, 200], [0, 0], [300, 100])
    assert abs(a1 - a2) < 1e-9
