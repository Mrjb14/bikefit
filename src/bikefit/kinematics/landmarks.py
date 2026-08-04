from typing import Literal

Side = Literal["left", "right"]

# Indices MediaPipe Pose (33 landmarks)
BODY: dict[Side, dict[str, int]] = {
    "right": {
        "shoulder": 12, "elbow": 14, "wrist": 16,
        "hip": 24, "knee": 26, "ankle": 28,
        "heel": 30, "foot": 32,
    },
    "left": {
        "shoulder": 11, "elbow": 13, "wrist": 15,
        "hip": 23, "knee": 25, "ankle": 27,
        "heel": 29, "foot": 31,
    },
}

# (point_A, sommet, point_B) pour chaque angle métier
ANGLE_JOINTS: dict[str, tuple[str, str, str]] = {
    "genou":    ("hip", "knee", "ankle"),
    "hanche":   ("shoulder", "hip", "knee"),
    "coude":    ("shoulder", "elbow", "wrist"),
    "cheville": ("knee", "ankle", "foot"),
}
