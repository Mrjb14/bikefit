"""Capture Kinect Xbox 360 (v1) -> enregistrement de profondeur pour `bikefit aero`.

Usage :
    python scripts/kinect_capture.py data/sources/aero/pads_hauts --duration 60
    python -m bikefit aero data/sources/aero/pads_hauts data/sources/aero/pads_bas

Écrit un dossier : `depth/*.png` (16 bits, mm, alignée sur la caméra couleur), `rgb.mp4`
(contrôle visuel + vérification fx au marqueur ArUco), `meta.json` (fps réel mesuré,
horodatages, intrinsèques). Format décrit dans `bikefit.aero.recording`.

Pilote : libfreenect (OpenKinect) et sa liaison Python `freenect`, mode
DEPTH_REGISTERED (profondeur en mm alignée sur l'image couleur). Installation sous
Windows : docs/kinect-aero.md. **Non testé sur matériel réel** à l'écriture de ce
script — la logique de capture (`capture`) est testée avec une fausse source.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

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
                        str(out_dir / RGB_FILE), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (w, h)
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


def _freenect_sources() -> tuple[DepthSource, RgbSource, Callable[[], None]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="Dossier de sortie (créé)")
    parser.add_argument("--duration", type=float, default=60.0, help="Durée (s), défaut 60")
    parser.add_argument(
        "--countdown",
        type=float,
        default=5.0,
        help="Délai avant capture (s) pour se mettre en position",
    )
    parser.add_argument("--no-rgb", action="store_true", help="Ne pas enregistrer rgb.mp4")
    args = parser.parse_args()

    if (args.out_dir / "meta.json").exists():
        sys.exit(f"{args.out_dir} contient déjà un enregistrement : choisir un autre dossier.")

    read_depth, read_rgb, stop = _freenect_sources()
    read_depth()  # réveille le capteur (premières frames souvent vides)
    for remaining in range(int(args.countdown), 0, -1):
        print(f"Capture dans {remaining} s... pédaler en position.", flush=True)
        time.sleep(1)

    def progress(n: int, elapsed: float) -> None:
        if n % 30 == 0:
            print(f"  {elapsed:5.1f} s - {n} frames", flush=True)

    try:
        rec = capture(
            args.out_dir,
            read_depth,
            None if args.no_rgb else read_rgb,
            args.duration,
            on_progress=progress,
        )
    finally:
        stop()
    print(
        f"OK : {rec.frame_count} frames @ {rec.fps:.1f} fps -> {args.out_dir}\n"
        f"Analyse : python -m bikefit aero {args.out_dir} --debug-dir data/outputs/debug"
    )


if __name__ == "__main__":
    main()
