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
from pathlib import Path

from bikefit.aero.capture import (  # noqa: F401  (réexportés : tests/test_aero.py)
    SDK_EXE,
    capture,
    freenect_sources,
    read_sdk_packet,
    sdk_sources,
)


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
    parser.add_argument(
        "--driver",
        choices=["auto", "sdk", "freenect"],
        default="auto",
        help="sdk = Kinect SDK 1.8 (kinect_grab.exe), freenect = libfreenect ; auto : sdk si build",
    )
    parser.add_argument("--no-rgb", action="store_true", help="Ne pas enregistrer rgb.mp4")
    args = parser.parse_args()

    if (args.out_dir / "meta.json").exists():
        sys.exit(f"{args.out_dir} contient déjà un enregistrement : choisir un autre dossier.")

    use_sdk = args.driver == "sdk" or (args.driver == "auto" and SDK_EXE.exists())
    read_depth, read_rgb, stop = sdk_sources() if use_sdk else freenect_sources()
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
