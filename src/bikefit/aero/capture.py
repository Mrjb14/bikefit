"""Capture Kinect Xbox 360 (v1) -> enregistrement de profondeur (sources SDK 1.8 / libfreenect)."""

from __future__ import annotations

import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from bikefit.aero.recording import (
    KINECT_V1_REGISTERED,
    RGB_FILE,
    DepthIntrinsics,
    DepthRecording,
    save_depth_recording,
)

DepthSource = Callable[[], np.ndarray]  # carte (480, 640) uint16 en mm, 0 = invalide
RgbSource = Callable[[], np.ndarray]  # image (480, 640, 3) BGR uint8

# fps nominal écrit dans rgb.mp4 : seul l'ordre des frames compte (le fps réel est dans
# meta.json). Pour convertir un temps réel en temps de rgb.mp4 : t * fps_reel / RGB_NOMINAL_FPS.
RGB_NOMINAL_FPS = 30.0


def capture(
    out_dir: Path,
    read_depth: DepthSource,
    read_rgb: RgbSource | None,
    duration_s: float,
    intrinsics: DepthIntrinsics = KINECT_V1_REGISTERED,
    clock: Callable[[], float] = time.monotonic,
    on_progress: Callable[[int, float], None] | None = None,
) -> DepthRecording:
    """Capture `duration_s` secondes. fps réel déduit des horodatages."""
    import cv2

    writer: cv2.VideoWriter | None = None
    t0 = clock()

    def frames() -> Iterator[tuple[int, np.ndarray]]:
        nonlocal writer
        n = 0
        while True:
            elapsed = clock() - t0
            if elapsed >= duration_s:
                return
            depth = read_depth()
            if read_rgb is not None:
                bgr = read_rgb()
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    h, w = bgr.shape[:2]
                    # fps nominal du conteneur : seul l'ordre des frames compte, chaque
                    # frame RGB i correspond à la frame de profondeur i (meta.json).
                    writer = cv2.VideoWriter(
                        str(out_dir / RGB_FILE),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        RGB_NOMINAL_FPS,
                        (w, h),
                    )
                writer.write(bgr)
            yield round(elapsed * 1000), depth
            n += 1
            if on_progress is not None:
                on_progress(n, elapsed)

    try:
        return save_depth_recording(out_dir, frames(), fps=None, intrinsics=intrinsics)
    finally:
        if writer is not None:
            writer.release()


SDK_EXE = Path(__file__).resolve().parents[3] / "tools" / "kinect_sdk" / "kinect_grab.exe"
_W, _H = 640, 480
_PACKET_MAGIC = b"KFRM"


def read_sdk_packet(stream: BinaryIO) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Lit un paquet de `kinect_grab.exe` : (timestamp_ms, profondeur mm, image BGR).

    Format : "KFRM" | int32 timestamp | profondeur 640*480 uint16 | BGR 640*480*3.
    Renvoie None si le flux est fermé (processus arrêté).
    """

    def read_exact(n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    head = read_exact(8)
    if head is None:
        return None
    if head[:4] != _PACKET_MAGIC:
        raise ValueError("Flux kinect_grab corrompu (en-tête KFRM attendu).")
    (stamp,) = struct.unpack("<i", head[4:])
    depth_raw = read_exact(_W * _H * 2)
    rgb_raw = read_exact(_W * _H * 3)
    if depth_raw is None or rgb_raw is None:
        return None
    depth = np.frombuffer(depth_raw, dtype="<u2").reshape(_H, _W)
    bgr = np.frombuffer(rgb_raw, dtype=np.uint8).reshape(_H, _W, 3)
    return stamp, depth, bgr


TiltSetter = Callable[[float], None]


def sdk_sources() -> tuple[DepthSource, RgbSource, Callable[[], None]]:
    """Kinect SDK 1.8 via `tools/kinect_sdk/kinect_grab.exe` (cf. build.ps1)."""
    read_depth, read_rgb, stop, _ = sdk_sources_with_motor()
    return read_depth, read_rgb, stop


def sdk_sources_with_motor() -> tuple[DepthSource, RgbSource, Callable[[], None], TiltSetter]:
    """Comme `sdk_sources`, plus le moteur d'inclinaison (degrés, + = vers le haut)."""
    if not SDK_EXE.exists():
        sys.exit(f"{SDK_EXE} absent : lancer tools/kinect_sdk/build.ps1 (Kinect SDK 1.8 requis).")
    proc = subprocess.Popen(
        [str(SDK_EXE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None
    )
    assert proc.stdout is not None and proc.stdin is not None
    latest: dict[str, np.ndarray] = {}

    def read_depth() -> np.ndarray:
        pkt = read_sdk_packet(proc.stdout)  # type: ignore[arg-type]
        if pkt is None:
            sys.exit("kinect_grab s'est arrêté : voir le message ci-dessus (alimentation, pilote).")
        _, depth, latest["bgr"] = pkt
        return depth.copy()

    def read_rgb() -> np.ndarray:
        return latest["bgr"].copy()

    def stop() -> None:
        proc.stdin.close()  # type: ignore[union-attr]
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def set_tilt(degrees: float) -> None:
        proc.stdin.write(f"TILT {round(degrees)}\n".encode())  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    return read_depth, read_rgb, stop, set_tilt


def freenect_sources() -> tuple[DepthSource, RgbSource, Callable[[], None]]:
    try:
        import freenect  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            "Module `freenect` introuvable : installer libfreenect et sa liaison Python"
            " (voir docs/kinect-aero.md, section Installation)."
        )

    def read_depth() -> np.ndarray:
        data = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
        if data is None:
            sys.exit("Kinect introuvable : vérifier l'alimentation secteur et le pilote USB.")
        return np.asarray(data[0], dtype=np.uint16)

    def read_rgb() -> np.ndarray:
        import cv2

        data = freenect.sync_get_video()
        if data is None:
            sys.exit("Flux couleur Kinect indisponible.")
        return cv2.cvtColor(np.asarray(data[0]), cv2.COLOR_RGB2BGR)

    return read_depth, read_rgb, freenect.sync_stop
