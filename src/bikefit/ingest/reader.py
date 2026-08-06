from __future__ import annotations

import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_STREAM_PREFIXES = ("http://", "https://", "rtsp://")

# fps indicatif si le flux ne le déclare pas (ESP32-CAM ~15-25 fps, cf. CLAUDE.md).
# N'est jamais utilisé pour l'horodatage (voir frames()) — affichage/info seulement.
_STREAM_FALLBACK_FPS = 20.0

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_CHUNK_SIZE = 4096


def _is_stream_url(source: str) -> bool:
    return source.startswith(_STREAM_PREFIXES)


def _iter_mjpeg_frames(url: str, timeout: float = 10.0) -> Iterator[np.ndarray]:
    """Découpe un flux HTTP multipart/x-mixed-replace (MJPEG) en frames BGR décodées.

    cv2.VideoCapture (backend FFmpeg) ne parse pas de façon fiable le multipart brut
    renvoyé par un petit serveur MJPEG maison (ex: firmware ESP32-CAM) — on lit le flux
    HTTP nous-mêmes et on découpe sur les marqueurs JPEG SOI/EOI plutôt que de compter
    sur un démuxeur générique.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk
            start = buf.find(_JPEG_SOI)
            end = buf.find(_JPEG_EOI, start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = buf[start : end + 2]
                buf = buf[end + 2 :]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame


@dataclass
class TimedFrame:
    timestamp_ms: int
    frame_idx: int
    frame_bgr: np.ndarray


class VideoReader:
    """Lit une vidéo locale (fichier) ou un flux réseau MJPEG (ESP32-CAM, Phase 3C).

    Pour un flux réseau : frame_count est inconnu (0, flux continu, pas de fin),
    et le fps déclaré par la source n'est généralement pas fiable → les timestamps
    sont calculés à partir de l'horloge murale au moment de la capture plutôt que
    d'un fps fixe (frame_idx * 1000 / fps), pour rester exacts malgré la gigue réseau.
    """

    def __init__(self, path: Path | str) -> None:
        self._is_stream = isinstance(path, str) and _is_stream_url(path)

        self._source: str
        if self._is_stream:
            self._path: Path | None = None
            self._source = str(path)
        else:
            self._path = Path(path)
            if not self._path.exists():
                raise FileNotFoundError(f"Vidéo introuvable : {self._path}")
            self._source = str(self._path)

        if self._is_stream:
            try:
                first_frame = next(_iter_mjpeg_frames(self._source))
            except (StopIteration, OSError) as exc:
                raise RuntimeError(f"Impossible d'ouvrir le flux MJPEG : {self._source}") from exc
            self._height, self._width = first_frame.shape[:2]
            self._frame_count = 0
            self._fps = _STREAM_FALLBACK_FPS
        else:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la vidéo : {self._source}")
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_stream(self) -> bool:
        return self._is_stream

    def frames(self) -> Iterator[TimedFrame]:
        if self._is_stream:
            yield from self._frames_stream()
        else:
            yield from self._frames_file()

    def _frames_stream(self) -> Iterator[TimedFrame]:
        start = time.monotonic()
        for frame_idx, frame in enumerate(_iter_mjpeg_frames(self._source)):
            yield TimedFrame(
                timestamp_ms=int((time.monotonic() - start) * 1000),
                frame_idx=frame_idx,
                frame_bgr=frame,
            )

    def _frames_file(self) -> Iterator[TimedFrame]:
        cap = cv2.VideoCapture(self._source)
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield TimedFrame(
                    timestamp_ms=int(frame_idx * 1000 / self._fps),
                    frame_idx=frame_idx,
                    frame_bgr=frame,
                )
                frame_idx += 1
        finally:
            cap.release()
