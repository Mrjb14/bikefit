"""Vidéo de debug : superpose squelette + métriques calculées sur la vidéo d'origine.

Fonctionne pour toute source (`ingest.VideoReader` : fichier, flux MJPEG réseau,
webcam) via deux stratégies distinctes :

- Fichier : `render_debug_video_from_landmarks` rejoue le fichier une seconde fois
  (décodage seul, pas de MediaPipe) et superpose le squelette + les métriques déjà
  calculées par le pipeline (post-filtre Savitzky-Golay — les valeurs réellement
  utilisées pour les recommandations).
- Flux réseau / webcam : impossible à "rejouer" — `render_debug_video_live` écrit la
  vidéo de debug pendant l'unique passage de capture, avec le squelette et les
  métriques bruts par frame (pas de lissage, qui nécessite l'ensemble de la capture).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from bikefit.kinematics.landmarks import SKELETON_EDGES

if TYPE_CHECKING:
    from bikefit.ingest.reader import TimedFrame, VideoReader
    from bikefit.pose.base import FrameLandmarks, PoseEstimator

_BONE = (170, 170, 170)  # BGR
_DOT = (60, 220, 90)
_STATUS_OK = (80, 220, 80)
_STATUS_MISSING = (60, 60, 230)
_HUD_TEXT = (235, 235, 235)
_HUD_BG = (15, 15, 15)


def _row_to_points(row: pd.Series, width: int, height: int) -> list[tuple[int, int] | None]:
    """Coordonnées pixel des 33 landmarks (None si non détecté/NaN)."""
    points: list[tuple[int, int] | None] = []
    for i in range(33):
        x, y = row.get(f"lm_{i}_x"), row.get(f"lm_{i}_y")
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            points.append(None)
        else:
            points.append((int(x * width), int(y * height)))
    return points


def _format_hud(row: pd.Series, columns: list[tuple[str, str, str]]) -> list[str]:
    """columns : (colonne, libellé, unité) -> lignes de texte "libellé : valeur unité"."""
    lines = []
    for col, label, unit in columns:
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            lines.append(f"{label} : --")
        else:
            lines.append(f"{label} : {val:.1f}{unit}")
    return lines


def _draw_skeleton(img: np.ndarray, points: list[tuple[int, int] | None]) -> None:
    for a, b in SKELETON_EDGES:
        pa, pb = points[a], points[b]
        if pa is not None and pb is not None:
            cv2.line(img, pa, pb, _BONE, 2, cv2.LINE_AA)
    for p in points:
        if p is not None:
            cv2.circle(img, p, 4, _DOT, -1, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    hud_lines: list[str],
    detected: bool,
    frame_idx: int,
    timestamp_ms: int,
) -> None:
    status = "DETECTE" if detected else "NON DETECTE"
    all_lines = [f"frame {frame_idx}  t={timestamp_ms} ms", status, *hud_lines]

    pad, line_h, panel_w = 10, 22, 260
    panel_h = pad * 2 + line_h * len(all_lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _HUD_BG, -1)
    img[:] = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    y = pad + 16
    for i, line in enumerate(all_lines):
        color = (_STATUS_OK if detected else _STATUS_MISSING) if i == 1 else _HUD_TEXT
        cv2.putText(img, line, (pad, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += line_h


# Mesure du fps réel (sources live) : fenêtre d'observation avant d'ouvrir le
# fichier, et plafond mémoire du tampon pour ne pas exploser sur de la haute
# résolution (on ouvre dès que l'une des deux limites est atteinte).
_FPS_MEASURE_SECONDS = 2.0
_FPS_MEASURE_MAX_BYTES = 64 * 1024 * 1024


class DebugVideoWriter:
    """Écrit une vidéo MP4 annotée (squelette + HUD) frame par frame.

    Ne prend en entrée que des frames BGR déjà décodées et les données à superposer —
    jamais la source elle-même, pour rester utilisable quelle que soit son origine
    (fichier, flux réseau, webcam).

    measure_fps : pour une source live (webcam/flux réseau), le fps annoncé par le
    driver est le débit *nominal* du capteur, pas celui réellement obtenu — l'inférence
    de pose tourne en ligne et ralentit la boucle. Écrire avec le fps nominal produit
    une vidéo qui défile trop vite (30 déclarés pour ~20 réels = lecture 1,5×), ce qui
    fausse la lecture de la cadence de pédalage. Dans ce mode, les premières frames
    sont tamponnées le temps de mesurer le débit réel via leurs timestamps (horloge
    murale), puis le fichier est ouvert au bon fps et le tampon vidé.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        measure_fps: bool = False,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self._size = (width, height)
        self._nominal_fps = fps if fps > 0 else 25.0
        self._writer: cv2.VideoWriter | None = None
        self._pending: list[np.ndarray] = []
        self._pending_bytes = 0
        self._first_ts: int | None = None
        self._last_ts: int | None = None

        if not measure_fps:
            self._open(self._nominal_fps)

    def _open(self, fps: float) -> None:
        # VideoWriter.fourcc (et non le legacy cv2.VideoWriter_fourcc, absent des
        # stubs) — valeur identique au runtime, vérifié sur OpenCV 4.x et 5.x.
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._output_path), fourcc, fps, self._size)
        if not writer.isOpened():
            raise RuntimeError(f"Impossible de créer la vidéo de debug : {self._output_path}")
        self._writer = writer

    def _measured_fps(self) -> float:
        """fps réel déduit des timestamps tamponnés, nominal si indéterminable."""
        if self._first_ts is None or self._last_ts is None or len(self._pending) < 2:
            return self._nominal_fps
        elapsed_s = (self._last_ts - self._first_ts) / 1000.0
        if elapsed_s <= 0:
            return self._nominal_fps
        return (len(self._pending) - 1) / elapsed_s

    def _flush_pending(self) -> None:
        self._open(self._measured_fps())
        assert self._writer is not None
        for img in self._pending:
            self._writer.write(img)
        self._pending.clear()
        self._pending_bytes = 0

    def write(
        self,
        frame_bgr: np.ndarray,
        points_px: list[tuple[int, int] | None] | None,
        hud_lines: list[str],
        detected: bool,
        frame_idx: int,
        timestamp_ms: int,
    ) -> None:
        img = frame_bgr.copy()
        if points_px is not None:
            _draw_skeleton(img, points_px)
        _draw_hud(img, hud_lines, detected, frame_idx, timestamp_ms)

        if self._writer is not None:
            self._writer.write(img)
            return

        # Mode mesure : on tamponne jusqu'à disposer d'un débit fiable.
        if self._first_ts is None:
            self._first_ts = timestamp_ms
        self._last_ts = timestamp_ms
        self._pending.append(img)
        self._pending_bytes += img.nbytes

        elapsed_s = (timestamp_ms - self._first_ts) / 1000.0
        if elapsed_s >= _FPS_MEASURE_SECONDS or self._pending_bytes >= _FPS_MEASURE_MAX_BYTES:
            self._flush_pending()

    def close(self) -> None:
        # Capture plus courte que la fenêtre de mesure : on ouvre quand même,
        # avec la meilleure estimation disponible, pour ne pas perdre les frames.
        if self._writer is None:
            self._flush_pending()
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> DebugVideoWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def render_debug_video_from_landmarks(
    video_path: Path,
    landmarks_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    hud_columns: list[tuple[str, str, str]],
    output_path: Path,
    progress: bool = True,
) -> None:
    """Rejoue un fichier vidéo local et superpose squelette + métriques déjà calculées.

    landmarks_df et metrics_df doivent porter une colonne `frame_idx` — la vidéo est
    redécodée (pas de ré-inférence MediaPipe), donc rapide même sur un cache "hit".
    """
    from bikefit.ingest.reader import VideoReader

    reader = VideoReader(video_path)
    landmarks_by_frame = landmarks_df.set_index("frame_idx")
    metrics_by_frame = metrics_df.set_index("frame_idx")

    with DebugVideoWriter(output_path, reader.width, reader.height, reader.fps or 25.0) as writer:
        if progress:
            print(f"Rendu vidéo de debug ({reader.frame_count or '?'} frames) ...")
        for frame in reader.frames():
            detected = False
            points_px = None
            if frame.frame_idx in landmarks_by_frame.index:
                row = landmarks_by_frame.loc[frame.frame_idx]
                detected = bool(row["detected"])
                if detected:
                    points_px = _row_to_points(row, reader.width, reader.height)

            hud_lines: list[str] = []
            if frame.frame_idx in metrics_by_frame.index:
                hud_lines = _format_hud(metrics_by_frame.loc[frame.frame_idx], hud_columns)

            writer.write(
                frame.frame_bgr, points_px, hud_lines, detected, frame.frame_idx, frame.timestamp_ms
            )
            if progress and frame.frame_idx % 100 == 0:
                print(f"  frame {frame.frame_idx}/{reader.frame_count or '?'}")


def render_debug_video_live(
    reader: VideoReader,
    estimator: PoseEstimator,
    compute_metrics: Callable[[pd.DataFrame], pd.DataFrame],
    hud_columns: list[tuple[str, str, str]],
    output_path: Path,
    max_duration_s: float | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Capture en direct (flux réseau/webcam) tout en écrivant la vidéo de debug.

    Un flux live ne se rejoue pas : squelette et métriques sont calculés frame par
    frame à partir des landmarks bruts (pas de lissage Savitzky-Golay, qui nécessite
    l'ensemble de la capture). compute_metrics : fonction vectorisée déjà liée à ses
    paramètres fixes (ex. `functools.partial(compute_angles, side=side, width=w,
    height=h)`), appelée sur un DataFrame d'une seule ligne par frame.

    Retourne le DataFrame de landmarks (même contenu que `compute_landmarks`).
    """
    from bikefit.pose.cache import _fl_to_dict, compute_landmarks

    # measure_fps : le débit réel d'une source live est imposé par la vitesse de
    # l'inférence, pas par le fps nominal du capteur (cf. DebugVideoWriter).
    with DebugVideoWriter(
        output_path, reader.width, reader.height, reader.fps or 25.0, measure_fps=True
    ) as writer:

        def _on_frame(frame: TimedFrame, fl: FrameLandmarks) -> None:
            row_df = pd.DataFrame([_fl_to_dict(fl)])
            points_px = None
            hud_lines: list[str] = []
            if fl.detected:
                points_px = _row_to_points(row_df.iloc[0], reader.width, reader.height)
                metrics_row = compute_metrics(row_df).iloc[0]
                hud_lines = _format_hud(metrics_row, hud_columns)
            writer.write(
                frame.frame_bgr,
                points_px,
                hud_lines,
                fl.detected,
                frame.frame_idx,
                frame.timestamp_ms,
            )

        return compute_landmarks(
            reader, estimator, progress=progress, max_duration_s=max_duration_s, on_frame=_on_frame
        )
