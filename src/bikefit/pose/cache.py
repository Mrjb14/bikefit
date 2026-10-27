from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bikefit._paths import CACHE_DIR
from bikefit.ingest.reader import TimedFrame, VideoReader
from bikefit.pose.base import FrameLandmarks, PoseEstimator


def _video_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _fl_to_dict(fl: FrameLandmarks) -> dict:
    row: dict = {
        "frame_idx": fl.frame_idx,
        "timestamp_ms": fl.timestamp_ms,
        "detected": fl.detected,
    }
    if fl.detected:
        for i, lm in enumerate(fl.landmarks):
            row[f"lm_{i}_x"] = lm.x
            row[f"lm_{i}_y"] = lm.y
            row[f"lm_{i}_z"] = lm.z
            row[f"lm_{i}_vis"] = lm.visibility
    else:
        for i in range(33):
            row[f"lm_{i}_x"] = float("nan")
            row[f"lm_{i}_y"] = float("nan")
            row[f"lm_{i}_z"] = float("nan")
            row[f"lm_{i}_vis"] = float("nan")
    return row


def compute_landmarks(
    reader: VideoReader,
    estimator: PoseEstimator,
    progress: bool = True,
    max_duration_s: float | None = None,
    on_frame: Callable[[TimedFrame, FrameLandmarks], None] | None = None,
) -> pd.DataFrame:
    """Calcule les landmarks pour chaque frame lue par `reader` (fichier ou flux réseau).

    max_duration_s borne la capture dans le temps — indispensable pour un flux
    réseau (ESP32-CAM) qui n'a pas de fin naturelle, ignoré pour un fichier local.

    on_frame, si fourni, est appelé une fois par frame traitée avec la frame brute et
    son résultat de pose — permet de brancher un hook (ex. écriture d'une vidéo de
    debug) sans coupler ce module à un souci d'affichage/rendu (bikefit.debug.video).
    """
    if progress:
        print(f"Calcul des landmarks ({reader.frame_count or '?'} frames) ...")
    rows = []
    for frame in reader.frames():
        if max_duration_s is not None and frame.timestamp_ms > max_duration_s * 1000:
            break
        fl = estimator.process(frame.frame_bgr, frame.timestamp_ms, frame.frame_idx)
        rows.append(_fl_to_dict(fl))
        if on_frame is not None:
            on_frame(frame, fl)
        if progress and frame.frame_idx % 100 == 0:
            print(f"  frame {frame.frame_idx}/{reader.frame_count or '?'}")

    return pd.DataFrame(rows)


class LandmarkCache:
    def __init__(self, cache_dir: Path = CACHE_DIR / "landmarks") -> None:
        self._cache_dir = cache_dir

    def path_for(self, video_path: Path) -> Path:
        """Fichier Parquet associé à une vidéo (clé = chemin résolu + mtime + taille)."""
        return self._cache_dir / f"{video_path.stem}_{_video_key(video_path)}.parquet"

    def invalidate(self, video_path: Path) -> bool:
        """Supprime le cache d'une vidéo (option --no-cache). True s'il existait."""
        cache_path = self.path_for(video_path)
        if not cache_path.exists():
            return False
        cache_path.unlink()
        return True

    def get_or_compute(
        self,
        video_path: Path,
        reader: VideoReader,
        estimator: PoseEstimator,
        progress: bool = True,
    ) -> pd.DataFrame:
        cache_path = self.path_for(video_path)

        if cache_path.exists():
            if progress:
                print(f"Cache trouvé : {cache_path.name}")
            return pd.read_parquet(cache_path)

        df = compute_landmarks(reader, estimator, progress=progress)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if progress:
            print(f"Cache sauvegardé : {cache_path.name}")
        return df
