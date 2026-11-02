"""Aéro (vue de face RGB-D, ex. Kinect) : surface frontale, CdA calibré, puissance.

Indépendant du pipeline MediaPipe : seule la carte de profondeur est utilisée.
Voir docs/kinect-aero.md pour la capture et la calibration terrain du Cd.
"""

from bikefit.aero.area import FrontalAreaStats, analyze_frontal_area
from bikefit.aero.drag import aero_power_w, calibrate_cd, cd_is_plausible, time_saved_s
from bikefit.aero.intrinsics import IntrinsicsCheck, check_intrinsics_with_marker
from bikefit.aero.recording import (
    KINECT_V1_REGISTERED,
    DepthIntrinsics,
    DepthRecording,
    save_depth_recording,
)

__all__ = [
    "KINECT_V1_REGISTERED",
    "DepthIntrinsics",
    "DepthRecording",
    "FrontalAreaStats",
    "IntrinsicsCheck",
    "aero_power_w",
    "analyze_frontal_area",
    "calibrate_cd",
    "cd_is_plausible",
    "check_intrinsics_with_marker",
    "save_depth_recording",
    "time_saved_s",
]
