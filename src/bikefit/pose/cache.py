from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from bikefit._paths import CACHE_DIR
from bikefit.ingest.reader import VideoReader
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


class LandmarkCache:
    def __init__(self, cache_dir: Path = CACHE_DIR / "landmarks") -> None:
        self._cache_dir = cache_dir

    def get_or_compute(
        self,
        video_path: Path,
        reader: VideoReader,
        estimator: PoseEstimator,
        progress: bool = True,
    ) -> pd.DataFrame:
        key = _video_key(video_path)
        cache_path = self._cache_dir / f"{video_path.stem}_{key}.parquet"

        if cache_path.exists():
            if progress:
                print(f"Cache trouvé : {cache_path.name}")
            return pd.read_parquet(cache_path)

        if progress:
            print(f"Calcul des landmarks ({reader.frame_count} frames) ...")
        rows = []
        for frame in reader.frames():
            fl = estimator.process(frame.frame_bgr, frame.timestamp_ms, frame.frame_idx)
            rows.append(_fl_to_dict(fl))
            if progress and frame.frame_idx % 100 == 0:
                print(f"  frame {frame.frame_idx}/{reader.frame_count}")

        df = pd.DataFrame(rows)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if progress:
            print(f"Cache sauvegardé : {cache_path.name}")
        return df
