"""Annotation visuelle des mesures corps et setup vélo (OpenCV)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import SKELETON_EDGES

# BGR — choisi pour être lisibles sur fond sombre ET clair
_BONE = (160, 160, 160)
_DOT = (50, 220, 80)

_C: dict[str, tuple[int, int, int]] = {
    "inseam": (50, 210, 70),  # vert
    "femur": (0, 165, 255),  # orange
    "tibia": (255, 200, 50),  # bleu clair
    "buste": (100, 80, 255),  # rouge-violet
    "circle": (255, 200, 80),  # cyan-jaune (trajectoire cheville)
    "bb": (30, 170, 255),  # orange vif (BB)
    "selle": (50, 210, 70),  # vert
    "guidon": (100, 80, 255),  # rouge-violet
    "reach": (0, 165, 255),  # orange
}


# ── Helpers ───────────────────────────────────────────────────────────────────


class _LM:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _px(lm: list[_LM], i: int, w: int, h: int) -> tuple[int, int]:
    return int(lm[i].x * w), int(lm[i].y * h)


def _avg(lm: list[_LM], i: int, j: int, w: int, h: int) -> tuple[int, int]:
    return int((lm[i].x + lm[j].x) / 2 * w), int((lm[i].y + lm[j].y) / 2 * h)


def _draw_skeleton(img: np.ndarray, lm: list[_LM], w: int, h: int) -> None:
    for a, b in SKELETON_EDGES:
        cv2.line(img, _px(lm, a, w, h), _px(lm, b, w, h), _BONE, 2, cv2.LINE_AA)
    for i in range(33):
        cv2.circle(img, _px(lm, i, w, h), 5, _DOT, -1, cv2.LINE_AA)


def _bracket(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    x_offset: int,
    w: int,
) -> None:
    """Crochet de mesure vertical entre p1 et p2, décalé de x_offset vers la droite."""
    bx = max(p1[0], p2[0]) + x_offset
    bx = max(10, min(bx, w - 180))
    cv2.line(img, (bx, p1[1]), (bx, p2[1]), color, 2, cv2.LINE_AA)
    cv2.line(img, (bx - 6, p1[1]), (bx + 6, p1[1]), color, 2)
    cv2.line(img, (bx - 6, p2[1]), (bx + 6, p2[1]), color, 2)
    mid_y = (p1[1] + p2[1]) // 2
    cv2.putText(
        img, label, (bx + 9, mid_y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA
    )


def _bracket_left(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    x_offset: int,
) -> None:
    """Crochet de mesure vertical, décalé vers la GAUCHE."""
    bx = min(p1[0], p2[0]) - x_offset
    bx = max(10, bx)
    cv2.line(img, (bx, p1[1]), (bx, p2[1]), color, 2, cv2.LINE_AA)
    cv2.line(img, (bx - 6, p1[1]), (bx + 6, p1[1]), color, 2)
    cv2.line(img, (bx - 6, p2[1]), (bx + 6, p2[1]), color, 2)
    mid_y = (p1[1] + p2[1]) // 2
    # texte à gauche du crochet
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    cv2.putText(
        img, label, (bx - tw - 9, mid_y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA
    )


# ── API publique ──────────────────────────────────────────────────────────────


def annotate_body_photo(
    photo_path: Path,
    measurements: dict[str, int],
    output_path: Path,
) -> None:
    """Dessine squelette + crochets de mesure sur la photo debout.

    Rerun MediaPipe en mode IMAGE pour obtenir les landmarks 2D normalisés.
    Sauvegarde l'image annotée dans output_path (JPEG).
    """
    import mediapipe as mp

    from bikefit.pose.mediapipe_estimator import _ensure_model

    img = cv2.imread(str(photo_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {photo_path}")
    h, w = img.shape[:2]

    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as det:
        result = det.detect(mp.Image.create_from_file(str(photo_path)))

    img_out = img.copy()

    if not result.pose_landmarks:
        cv2.putText(
            img_out, "Aucune pose détectée", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 220), 2
        )
        cv2.imwrite(str(output_path), img_out)
        return

    raw = result.pose_landmarks[0]
    lm = [_LM(p.x, p.y) for p in raw]

    _draw_skeleton(img_out, lm, w, h)

    hip = _avg(lm, 23, 24, w, h)
    knee = _avg(lm, 25, 26, w, h)
    ankle = _avg(lm, 27, 28, w, h)
    shldr = _avg(lm, 11, 12, w, h)

    if "entrejambe" in measurements:
        _bracket(
            img_out, hip, ankle, f"Entrejambe  {measurements['entrejambe']} mm", _C["inseam"], 70, w
        )
    if "longueur_femur" in measurements:
        _bracket(
            img_out, hip, knee, f"Femur  {measurements['longueur_femur']} mm", _C["femur"], 20, w
        )
    if "longueur_tibia" in measurements:
        _bracket(
            img_out, knee, ankle, f"Tibia  {measurements['longueur_tibia']} mm", _C["tibia"], 20, w
        )
    if "longueur_buste" in measurements:
        _bracket_left(
            img_out, hip, shldr, f"Buste  {measurements['longueur_buste']} mm", _C["buste"], 60
        )

    cv2.imwrite(str(output_path), img_out)


def annotate_setup_frame(
    video_path: str,
    landmarks_df: pd.DataFrame,
    estimates: dict,
    mm_per_pixel: float,
    image_width: int,
    image_height: int,
    side: Literal["right", "left"],
    output_path: Path,
) -> None:
    """Dessine squelette, trajectoire cheville, BB et lignes de mesure
    sur la frame au Point Mort Bas (cheville la plus basse).

    estimates doit contenir les clés '_bb_px' et '_rayon_cheville_mm'
    (avant qu'elles ne soient supprimées par le CLI).
    """
    from bikefit.kinematics.landmarks import BODY

    lm_map = BODY[side]
    w, h = image_width, image_height
    detected = landmarks_df[landmarks_df["detected"]].copy()

    if len(detected) < 5:
        return

    # ── Frame au PMB : cheville la plus basse (y max dans l'image) ───────────
    ay_col = f"lm_{lm_map['ankle']}_y"
    pmb_idx = int(detected.loc[detected[ay_col].idxmax(), "frame_idx"])

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pmb_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return

    img_out = frame.copy()

    # ── Squelette sur la frame PMB ────────────────────────────────────────────
    row = landmarks_df[landmarks_df["frame_idx"] == pmb_idx]
    if not row.empty and row.iloc[0]["detected"]:
        r = row.iloc[0]
        lm = [_LM(r.get(f"lm_{i}_x", 0.5), r.get(f"lm_{i}_y", 0.5)) for i in range(33)]
        _draw_skeleton(img_out, lm, w, h)

    # ── Trajectoire cheville (tous les points détectés, sous-échantillonnés) ──
    ax_col = f"lm_{lm_map['ankle']}_x"
    ankles = detected[[ax_col, ay_col]].dropna()
    for _, ar in ankles.iloc[::4].iterrows():
        pt = (int(ar[ax_col] * w), int(ar[ay_col] * h))
        cv2.circle(img_out, pt, 3, _C["circle"], -1, cv2.LINE_AA)

    # ── Cercle ajusté (trajectoire cheville) + BB ─────────────────────────────
    bb_px = estimates.get("_bb_px")
    r_mm = estimates.get("_rayon_cheville_mm", 0)
    r_px = int(r_mm / mm_per_pixel) if mm_per_pixel > 0 else 0

    if bb_px is not None:
        bx, by = int(bb_px[0]), int(bb_px[1])
        if r_px > 0:
            cv2.circle(img_out, (bx, by), r_px, _C["circle"], 1, cv2.LINE_AA)
        cv2.drawMarker(img_out, (bx, by), _C["bb"], cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.putText(
            img_out,
            "BB",
            (bx + 14, by - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            _C["bb"],
            1,
            cv2.LINE_AA,
        )

        # ── Lignes de mesure verticales / horizontales ────────────────────────
        def _vline(val_mm: float | None, x_shift: int, color: tuple, label: str) -> None:
            if val_mm is None:
                return
            top_y = by - int(val_mm / mm_per_pixel)
            x = bx + x_shift
            cv2.line(img_out, (x, by), (x, top_y), color, 2, cv2.LINE_AA)
            cv2.line(img_out, (x - 6, by), (x + 6, by), color, 2)
            cv2.line(img_out, (x - 6, top_y), (x + 6, top_y), color, 2)
            mid_y = (by + top_y) // 2
            cv2.putText(
                img_out,
                f"{label}  {val_mm:.0f} mm",
                (x + 10, mid_y + 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        _vline(estimates.get("hauteur_selle_bb"), 0, _C["selle"], "Selle")
        _vline(estimates.get("hauteur_guidon_bb"), 25, _C["guidon"], "Guidon")

        reach = estimates.get("reach_prolongateurs")
        if reach is not None:
            rx = bx + int(reach / mm_per_pixel)
            cv2.line(img_out, (bx, by + 10), (rx, by + 10), _C["reach"], 2, cv2.LINE_AA)
            cv2.line(img_out, (bx, by + 4), (bx, by + 16), _C["reach"], 2)
            cv2.line(img_out, (rx, by + 4), (rx, by + 16), _C["reach"], 2)
            cv2.putText(
                img_out,
                f"Reach  {reach:.0f} mm",
                (bx + 10, by + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                _C["reach"],
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), img_out)
