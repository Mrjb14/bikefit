#!/usr/bin/env python3
"""
BikeFit CLM — Spike Phase 0
============================
Objectif : valider que MediaPipe détecte correctement les articulations
dans TES conditions réelles (lumière, tenue, arrière-plan).

Usage :
    python spike_bikefit.py video_profil.mp4 --side right
    python spike_bikefit.py video_profil.mp4 --side left --output-dir resultats

Sorties (dans ./spike_output par défaut) :
    - annotated.mp4   : vidéo avec squelette + angles superposés
    - landmarks.csv   : coordonnées brutes de chaque articulation par frame
    - angles.csv      : angles calculés par frame
    - summary.json    : statistiques (min/max/moyenne des angles, fps, visibilité)

Installation :
    pip install mediapipe opencv-python numpy

Critère de succès du spike :
    - visibilité moyenne > 0.8 sur hanche/genou/cheville
    - le squelette suit la jambe sans "sauter" sur la jambe opposée
    - l'angle du genou oscille de façon régulière (sinusoïde propre)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("MediaPipe manquant : pip install mediapipe")

# ---------------------------------------------------------------------------
# Landmarks MediaPipe Pose (indices officiels)
# ---------------------------------------------------------------------------
LANDMARKS = {
    "right": {
        "shoulder": 12, "elbow": 14, "wrist": 16,
        "hip": 24, "knee": 26, "ankle": 28,
        "heel": 30, "foot": 32,
    },
    "left": {
        "shoulder": 11, "elbow": 13, "wrist": 15,
        "hip": 23, "knee": 25, "ankle": 27,
        "heel": 29, "foot": 31,
    },
}

# Segments à dessiner (paires de noms de points)
SKELETON = [
    ("shoulder", "elbow"), ("elbow", "wrist"),
    ("shoulder", "hip"), ("hip", "knee"),
    ("knee", "ankle"), ("ankle", "heel"), ("heel", "foot"),
]

# Angles à calculer : nom -> (point A, sommet, point B)
ANGLES = {
    "genou": ("hip", "knee", "ankle"),
    "hanche": ("shoulder", "hip", "knee"),
    "coude": ("shoulder", "elbow", "wrist"),
    "cheville": ("knee", "ankle", "foot"),
}

# Plages cibles indicatives en position CLM (pour affichage seulement)
TARGETS = {
    "genou": (70, 150),     # min PMH ~70-75, max PMB ~140-150
    "hanche": (40, 110),
    "coude": (90, 110),
}


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle en degrés au sommet b, formé par les segments b->a et b->c."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
    """Angle du tronc par rapport à l'horizontale, en degrés (0 = à plat)."""
    v = shoulder - hip
    # Attention : axe y de l'image pointe vers le bas
    return float(np.degrees(np.arctan2(-(v[1]), abs(v[0]))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Spike bike fitting MediaPipe")
    parser.add_argument("video", type=Path, help="Vidéo de profil (mp4/mov)")
    parser.add_argument("--side", choices=["right", "left"], default="right",
                        help="Côté du corps face à la caméra (défaut: right)")
    parser.add_argument("--output-dir", type=Path, default=Path("spike_output"))
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=2,
                        help="0=rapide, 2=précis (défaut: 2)")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Vidéo introuvable : {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lm_ids = LANDMARKS[args.side]

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vidéo : {width}x{height} @ {fps:.1f} fps, {n_frames} frames")
    if fps < 50:
        print("⚠️  fps < 50 : acceptable pour le spike, mais vise 60 fps "
              "pour les mesures finales.")

    writer = cv2.VideoWriter(
        str(args.output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )

    pose = mp.solutions.pose.Pose(
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmark_rows: list[dict] = []
    angle_rows: list[dict] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            pts: dict[str, np.ndarray] = {}
            vis: dict[str, float] = {}
            for name, idx in lm_ids.items():
                lm = lms[idx]
                pts[name] = np.array([lm.x * width, lm.y * height])
                vis[name] = lm.visibility

            # --- enregistrement landmarks bruts
            row = {"frame": frame_idx, "t": frame_idx / fps}
            for name in lm_ids:
                row[f"{name}_x"] = round(pts[name][0], 2)
                row[f"{name}_y"] = round(pts[name][1], 2)
                row[f"{name}_vis"] = round(vis[name], 3)
            landmark_rows.append(row)

            # --- angles
            arow = {"frame": frame_idx, "t": frame_idx / fps}
            for aname, (pa, pb, pc) in ANGLES.items():
                arow[aname] = round(angle_3pts(pts[pa], pts[pb], pts[pc]), 1)
            arow["tronc"] = round(trunk_angle(pts["shoulder"], pts["hip"]), 1)
            angle_rows.append(arow)

            # --- dessin squelette
            for p1, p2 in SKELETON:
                c1 = tuple(pts[p1].astype(int))
                c2 = tuple(pts[p2].astype(int))
                cv2.line(frame, c1, c2, (0, 255, 0), 3)
            for name, p in pts.items():
                low_vis = vis[name] < 0.6
                color = (0, 0, 255) if low_vis else (0, 200, 255)
                cv2.circle(frame, tuple(p.astype(int)), 7, color, -1)

            # --- affichage angles à l'écran
            y0 = 40
            for aname in ("genou", "hanche", "coude", "tronc"):
                val = arow[aname]
                txt = f"{aname}: {val:.0f} deg"
                cv2.putText(frame, txt, (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(frame, txt, (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                y0 += 40
        else:
            cv2.putText(frame, "POSE NON DETECTEE", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{n_frames}")

    cap.release()
    writer.release()
    pose.close()

    if not angle_rows:
        sys.exit("Aucune pose détectée sur toute la vidéo. "
                 "Revoir éclairage / cadrage / contraste.")

    # ------------------------------------------------------------------
    # Exports CSV
    # ------------------------------------------------------------------
    def write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")

    write_csv(args.output_dir / "landmarks.csv", landmark_rows)
    write_csv(args.output_dir / "angles.csv", angle_rows)

    # ------------------------------------------------------------------
    # Statistiques de synthèse
    # ------------------------------------------------------------------
    detection_rate = len(angle_rows) / frame_idx
    summary: dict = {
        "video": str(args.video),
        "fps": fps,
        "frames": frame_idx,
        "detection_rate": round(detection_rate, 3),
        "angles": {},
        "visibilite_moyenne": {},
    }

    for aname in list(ANGLES) + ["tronc"]:
        vals = np.array([r[aname] for r in angle_rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            summary["angles"][aname] = {
                "min": round(float(np.min(vals)), 1),
                "max": round(float(np.max(vals)), 1),
                "moyenne": round(float(np.mean(vals)), 1),
                "ecart_type": round(float(np.std(vals)), 1),
            }

    for name in lm_ids:
        v = np.array([r[f"{name}_vis"] for r in landmark_rows])
        summary["visibilite_moyenne"][name] = round(float(np.mean(v)), 3)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Verdict du spike
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSULTATS DU SPIKE")
    print("=" * 60)
    print(f"Taux de détection : {detection_rate:.0%}")
    print("\nVisibilité moyenne par articulation :")
    for name, v in summary["visibilite_moyenne"].items():
        flag = "✅" if v > 0.8 else ("⚠️ " if v > 0.6 else "❌")
        print(f"  {flag} {name:10s} {v:.2f}")
    print("\nAngles (min / max / moyenne ± σ) :")
    for aname, s in summary["angles"].items():
        target = TARGETS.get(aname)
        tstr = f"   [cible CLM ~{target[0]}-{target[1]}°]" if target else ""
        print(f"  {aname:10s} {s['min']:6.1f} / {s['max']:6.1f} / "
              f"{s['moyenne']:6.1f} ± {s['ecart_type']:.1f}{tstr}")

    knee = summary["angles"].get("genou", {})
    print("\nVerdict :")
    hip_knee_ankle_ok = all(
        summary["visibilite_moyenne"].get(n, 0) > 0.8
        for n in ("hip", "knee", "ankle")
    )
    if hip_knee_ankle_ok and detection_rate > 0.95:
        print("  ✅ Conditions de capture exploitables — GO pour la Phase 1.")
    else:
        print("  ⚠️  Détection fragile. Pistes : plus de lumière, fond uni,")
        print("     marqueurs contrastés sur les articulations, caméra plus proche.")
    if knee and knee.get("ecart_type", 0) < 5:
        print("  ⚠️  L'angle du genou varie peu : pédalais-tu pendant la capture ?")

    print(f"\nFichiers écrits dans : {args.output_dir}/")
    print("  → Regarde annotated.mp4 : le squelette doit suivre ta jambe")
    print("    côté caméra sans sauter sur la jambe opposée.")


if __name__ == "__main__":
    main()
