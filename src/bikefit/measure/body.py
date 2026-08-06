"""Estimation des mensurations corporelles depuis une photo debout."""

from __future__ import annotations

from pathlib import Path

# Proportion nez→cheville / stature (anthropométrie moyenne adulte)
_NOSE_ANKLE_RATIO = 0.935


def estimate_body_measurements(
    photo_path: Path,
    taille_mm: int | None = None,
) -> dict[str, int]:
    """Estime les mensurations depuis une photo debout (profil ou face à face).

    Utilise MediaPipe Pose en mode IMAGE pour extraire les world_landmarks 3D
    (en mètres, origine = centre des hanches, y = haut).

    Si taille_mm est fourni (recommandé), applique un facteur d'échelle basé
    sur la proportion nez→cheville / stature (≈ 0.935).

    Retourne un dict avec les mêmes clés que mensurations.yaml, valeurs en mm.
    """
    import mediapipe as mp

    from bikefit.pose.mediapipe_estimator import _ensure_model

    model_path = _ensure_model()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as det:
        result = det.detect(mp.Image.create_from_file(str(photo_path)))

    if not result.pose_world_landmarks:
        raise RuntimeError("Aucune personne détectée dans la photo.")

    wl = result.pose_world_landmarks[0]

    def _y(i: int) -> float:
        # y=bas dans les world_landmarks MediaPipe (même convention que les frames image)
        return wl[i].y

    def _x(i: int) -> float:
        return wl[i].x

    # Moyennes gauche/droite pour robustesse
    hip_y = (_y(23) + _y(24)) / 2  # ≈ 0.0 (origine)
    knee_y = (_y(25) + _y(26)) / 2
    ankle_y = (_y(27) + _y(28)) / 2
    shoulder_y = (_y(11) + _y(12)) / 2
    nose_y = _y(0)

    raw_m: dict[str, float] = {
        "entrejambe": ankle_y - hip_y,  # hanche → cheville
        "longueur_femur": knee_y - hip_y,  # hanche → genou
        "longueur_tibia": ankle_y - knee_y,  # genou → cheville
        "longueur_buste": hip_y - shoulder_y,  # hanche → épaule (proxy buste)
        "largeur_epaules": abs(_x(11) - _x(12)),
        "largeur_hanches": abs(_x(23) - _x(24)),
    }

    # Facteur d'échelle si taille connue
    scale = 1.0
    if taille_mm is not None:
        nose_ankle_m = ankle_y - nose_y
        if nose_ankle_m > 0.1:
            scale = (taille_mm / 1000 * _NOSE_ANKLE_RATIO) / nose_ankle_m

    return {k: round(v * scale * 1000) for k, v in raw_m.items()}
