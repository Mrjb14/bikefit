from typing import Literal

Side = Literal["left", "right"]

# Indices MediaPipe Pose (33 landmarks)
BODY: dict[Side, dict[str, int]] = {
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
        "heel": 30,
        "foot": 32,
    },
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
        "heel": 29,
        "foot": 31,
    },
}

# (point_A, sommet, point_B) pour chaque angle métier
ANGLE_JOINTS: dict[str, tuple[str, str, str]] = {
    "genou": ("hip", "knee", "ankle"),
    "hanche": ("shoulder", "hip", "knee"),
    "coude": ("shoulder", "elbow", "wrist"),
    "cheville": ("knee", "ankle", "foot"),
}

# Connexions squelette MediaPipe (33 landmarks) à tracer — partagé entre measure/render.py
# (annotation photo/frame) et debug/video.py (vidéo de debug).
SKELETON_EDGES: list[tuple[int, int]] = [
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),  # torse
    (11, 13),
    (13, 15),  # bras gauche
    (12, 14),
    (14, 16),  # bras droit
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),  # jambe gauche
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),  # jambe droite
]
