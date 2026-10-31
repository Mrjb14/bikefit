"""Enregistrement de profondeur sur disque (capteur RGB-D, ex. Kinect Xbox 360).

Format : un dossier contenant
- `meta.json` : capteur, fps, intrinsèques de la caméra à laquelle la profondeur est
  alignée (fx, fy, cx, cy en pixels), taille d'image, horodatages ;
- `depth/000000.png`, `depth/000001.png`… : une carte de profondeur par frame, PNG
  16 bits non signé, valeur = distance en **mm** le long de l'axe optique, 0 = invalide ;
- `rgb.mp4` (optionnel) : flux couleur aligné, pour MediaPipe ou la vérification ArUco.

Pourquoi des PNG et pas un gros .npz : 60 s à 30 fps en 640×480×16 bits pèsent ~1 Go
en mémoire. Les PNG se compressent bien (sans perte) et se lisent frame par frame,
comme `VideoReader` lit une vidéo.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

_META_FILE = "meta.json"
_DEPTH_DIR = "depth"
RGB_FILE = "rgb.mp4"


class DepthIntrinsics(BaseModel):
    """Intrinsèques pinhole (pixels) de la caméra à laquelle la profondeur est alignée."""

    fx: float
    fy: float
    cx: float
    cy: float


# Valeurs usuelles de la caméra couleur Kinect v1 (640×480), à laquelle libfreenect
# aligne la profondeur en mode DEPTH_REGISTERED. Approximation d'usine : l'erreur sur
# fx se retrouve au carré sur la surface (2 % sur fx -> 4 % sur l'aire). À vérifier
# avec le marqueur ArUco (`check_intrinsics_with_marker`) ou un damier.
KINECT_V1_REGISTERED = DepthIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5)


class DepthMeta(BaseModel):
    sensor: str = "kinect_v1"
    fps: float
    width: int
    height: int
    intrinsics: DepthIntrinsics
    timestamps_ms: list[int]

    @property
    def frame_count(self) -> int:
        return len(self.timestamps_ms)


@dataclass(frozen=True)
class DepthFrame:
    frame_idx: int
    timestamp_ms: int
    depth_mm: np.ndarray  # (H, W) uint16, 0 = pas de mesure


class DepthRecording:
    """Lecture paresseuse d'un enregistrement de profondeur (dossier, cf. module)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        meta_path = self.path / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Enregistrement de profondeur invalide : {meta_path} absent")
        self.meta = DepthMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

    @property
    def intrinsics(self) -> DepthIntrinsics:
        return self.meta.intrinsics

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def frame_count(self) -> int:
        return self.meta.frame_count

    @property
    def rgb_path(self) -> Path | None:
        p = self.path / RGB_FILE
        return p if p.exists() else None

    def frame_path(self, idx: int) -> Path:
        return self.path / _DEPTH_DIR / f"{idx:06d}.png"

    def read(self, idx: int) -> np.ndarray:
        import cv2

        img = cv2.imread(str(self.frame_path(idx)), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Frame de profondeur illisible : {self.frame_path(idx)}")
        if img.dtype != np.uint16 or img.ndim != 2:
            raise ValueError(f"Frame {idx} : PNG 16 bits mono-canal attendu, reçu {img.dtype}")
        return img  # type: ignore[no-any-return]

    def frames(self, every: int = 1) -> Iterator[DepthFrame]:
        for idx in range(0, self.frame_count, max(1, every)):
            yield DepthFrame(idx, self.meta.timestamps_ms[idx], self.read(idx))


def save_depth_recording(
    path: Path | str,
    depth_frames: Iterable[np.ndarray | tuple[int, np.ndarray]],
    fps: float | None,
    intrinsics: DepthIntrinsics,
    timestamps_ms: list[int] | None = None,
    sensor: str = "kinect_v1",
) -> DepthRecording:
    """Écrit un enregistrement (utilisé par le script de capture et les tests).

    Les frames sont écrites au fil de l'eau : un générateur de capture n'est jamais
    matérialisé en mémoire. Chaque élément est une carte de profondeur, ou un couple
    (horodatage_ms, carte) pour une capture live. `fps=None` : fps réel déduit des
    horodatages (le débit nominal d'un capteur n'est pas tenu sous charge, cf. debug/).
    """
    import cv2

    root = Path(path)
    (root / _DEPTH_DIR).mkdir(parents=True, exist_ok=True)
    width = height = 0
    stamps: list[int] = []
    for idx, item in enumerate(depth_frames):
        stamp: int | None = None
        if isinstance(item, tuple):
            stamp, depth = item
        else:
            depth = item
        frame = np.asarray(depth)
        if frame.ndim != 2:
            raise ValueError(f"Frame {idx} : carte 2D attendue, reçu {frame.shape}")
        frame = np.clip(frame, 0, 65535).astype(np.uint16)
        height, width = frame.shape
        if not cv2.imwrite(str(root / _DEPTH_DIR / f"{idx:06d}.png"), frame):
            raise OSError(f"Écriture impossible : frame {idx}")
        if stamp is None:
            if timestamps_ms is not None:
                stamp = timestamps_ms[idx]
            elif fps is not None:
                stamp = round(idx * 1000 / fps)
            else:
                raise ValueError("fps ou horodatages requis")
        stamps.append(int(stamp))

    if fps is None:
        span_s = (stamps[-1] - stamps[0]) / 1000 if len(stamps) > 1 else 0.0
        fps = (len(stamps) - 1) / span_s if span_s > 0 else 30.0

    meta = DepthMeta(
        sensor=sensor,
        fps=round(fps, 2),
        width=width,
        height=height,
        intrinsics=intrinsics,
        timestamps_ms=stamps,
    )
    (root / _META_FILE).write_text(json.dumps(meta.model_dump(), indent=2), encoding="utf-8")
    return DepthRecording(root)
