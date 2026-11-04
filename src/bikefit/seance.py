"""Séance d'entraînement : une commande, Kinect + caméra de profil, fenêtre de pédalage auto.

`bikefit seance` enchaîne ce qui demandait une dizaine de commandes : vérification du
cadrage, compte à rebours, capture simultanée (Kinect = vue de face + profondeur, caméra
de profil = angles), détection automatique de la fenêtre de pédalage (plus de
`--start/--end` à la main), analyse, comparaison à la position aéro de référence.

Les briques pures (fenêtre de pédalage, vérification du cadrage, enregistreur live,
référence aéro) sont testées isolément ; `run_seance` ne fait que les orchestrer.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

from bikefit._paths import AERO_REFERENCE_FILE, SEANCES_DIR
from bikefit.aero.area import (
    DEFAULT_BAND_MM,
    MAX_RELIABLE_HOLE_RATIO,
    FrontalAreaStats,
    fit_floor_plane,
    median_depth,
    reference_depth_mm,
    segment_rider,
)
from bikefit.aero.recording import DepthIntrinsics
from bikefit.ingest.reader import TimedFrame

Echo = Callable[[str], None]
TiltSetter = Callable[[float], None]
KinectSources = tuple[Callable[[], np.ndarray], Callable[[], np.ndarray], Callable[[], None]]

# Fenêtre de pédalage : au moins ce nombre de cycles consécutifs, sinon on retombe sur
# des marges fixes (la détection n'est pas assez sûre pour couper automatiquement).
MIN_CYCLES_FOR_WINDOW = 5
FALLBACK_SKIP_START_S = 8.0
FALLBACK_SKIP_END_S = 3.0

# Cadrage Kinect : plage de distance du buste (la Kinect v1 mesure de 0,8 à ~4 m ; on
# vise ~2 m) et décalage latéral toléré de la silhouette, en fraction de la largeur.
MIN_RIDER_DISTANCE_MM = 1200.0
MAX_RIDER_DISTANCE_MM = 3200.0
MAX_OFF_CENTER = 0.20


# ── Fenêtre de pédalage ───────────────────────────────────────────────────────


def pedaling_window(cycles: list[tuple[int, int]], fps: float) -> tuple[float, float, int] | None:
    """(début_s, fin_s, n_cycles) de la plus longue suite de cycles de pédalage.

    `cycles` : sortie de `detect_cycles` (frame_idx début/fin). Un cycle manquant (rejeté
    par le filtre de plausibilité) ne coupe pas la suite ; un trou plus long, si — c'est la
    montée sur le vélo ou la descente. None si aucune suite n'atteint
    `MIN_CYCLES_FOR_WINDOW` cycles.
    """
    if not cycles or fps <= 0:
        return None
    median = float(np.median([end - start for start, end in cycles]))
    runs: list[list[tuple[int, int]]] = [[cycles[0]]]
    for cycle in cycles[1:]:
        if cycle[0] - runs[-1][-1][1] <= median:
            runs[-1].append(cycle)
        else:
            runs.append([cycle])
    best = max(runs, key=lambda r: r[-1][1] - r[0][0])
    if len(best) < MIN_CYCLES_FOR_WINDOW:
        return None
    return best[0][0] / fps, best[-1][1] / fps, len(best)


def detect_window_from_video(video: Path, side: str) -> tuple[float, float, int] | None:
    """Fenêtre de pédalage d'une vidéo (pose via le cache de landmarks, puis cycles)."""
    from bikefit.ingest import VideoReader
    from bikefit.kinematics import detect_cycles, filter_landmarks
    from bikefit.pose import LandmarkCache, MediaPipePoseEstimator
    from bikefit.pose.cache import CACHE_DIR

    reader = VideoReader(video)
    with MediaPipePoseEstimator() as estimator:
        df = LandmarkCache(CACHE_DIR / "landmarks").get_or_compute(video, reader, estimator)
    filtered = filter_landmarks(df, reader.fps)
    cycles = detect_cycles(filtered, "right" if side == "right" else "left", reader.fps)
    return pedaling_window(cycles, reader.fps)


# ── Vérification du cadrage (avant de lancer la capture) ──────────────────────


class Preflight(BaseModel):
    errors: list[str] = []  # bloquant : la capture serait inexploitable
    warnings: list[str] = []  # à corriger si possible
    depth_ref_mm: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight_depth(frames: list[np.ndarray], intr: DepthIntrinsics) -> Preflight:
    """Le cycliste est-il bien cadré ? Distance, centrage, silhouette, sol, trous."""
    background = median_depth(frames)
    ref = reference_depth_mm(background)
    if ref is None:
        return Preflight(
            errors=[
                "Aucune profondeur mesurée au centre : cycliste absent du champ ou à moins"
                " de 0,8 m de la Kinect."
            ]
        )
    result = Preflight(depth_ref_mm=ref)
    if ref < MIN_RIDER_DISTANCE_MM:
        result.errors.append(
            f"Cycliste trop près ({ref / 1000:.1f} m) : reculer la Kinect vers ~2 m."
        )
    elif ref > MAX_RIDER_DISTANCE_MM:
        result.errors.append(
            f"Cycliste trop loin ({ref / 1000:.1f} m) : rapprocher la Kinect vers ~2 m."
        )

    floor = fit_floor_plane(background, intr)
    if floor is None:
        result.warnings.append(
            "Sol non détecté (Kinect de niveau ? sol visible ?) : il peut être compté."
        )
    seg = segment_rider(
        background, intr, max(1.0, ref - DEFAULT_BAND_MM), ref + DEFAULT_BAND_MM, floor
    )
    if seg is None:
        result.errors.append("Silhouette du cycliste introuvable : vérifier le cadrage.")
        return result

    silhouette, holes = seg
    cols = np.flatnonzero(silhouette.any(axis=0))
    width = silhouette.shape[1]
    offset = ((cols[0] + cols[-1]) / 2 - width / 2) / width
    if cols[0] <= 2 or cols[-1] >= width - 3:
        result.errors.append(
            "Silhouette coupée sur le côté (ou mur pris pour le cycliste) : centrer la Kinect"
            " dans l'axe du vélo et vérifier qu'il n'y a rien d'autre dans le champ."
        )
    elif abs(offset) > MAX_OFF_CENTER:
        side = "à droite" if offset > 0 else "à gauche"
        result.errors.append(
            f"Cycliste décentré ({abs(offset):.0%} de l'image, {side}) : recentrer la Kinect"
            " dans l'axe du vélo."
        )
    hole_ratio = float(holes.sum()) / max(1, int(silhouette.sum()))
    if hole_ratio > MAX_RELIABLE_HOLE_RATIO:
        result.warnings.append(
            f"{hole_ratio:.0%} de la silhouette sans mesure (vêtement noir/brillant ? soleil ?)."
        )
    return result


# ── Inclinaison de la Kinect (moteur) ─────────────────────────────────────────

KINECT_VFOV_DEG = 43.0  # champ vertical de la Kinect v1
TILT_LIMIT_DEG = 15.0  # au-delà, la surface frontale est trop biaisée (caméra non frontale)
TILT_TOLERANCE_PX = 40  # décalage vertical toléré de la silhouette (~3,6°)
TILT_SETTLE_S = 3.0  # le moteur est lent ; les images pendant le mouvement sont floues
MAX_TILT_STEPS = 2
REFERENCE_TILT_WARN_DEG = 3.0


def tilt_correction_deg(frames: list[np.ndarray], intr: DepthIntrinsics) -> float | None:
    """Correction d'inclinaison (degrés, + = vers le haut) pour centrer le cycliste.

    La silhouette est trop basse dans l'image -> regarder plus bas (négatif), et inversement.
    None si le cycliste n'est pas trouvé (rien à corriger de façon fiable).
    """
    background = median_depth(frames)
    ref = reference_depth_mm(background)
    if ref is None:
        return None
    floor = fit_floor_plane(background, intr)
    seg = segment_rider(
        background, intr, max(1.0, ref - DEFAULT_BAND_MM), ref + DEFAULT_BAND_MM, floor
    )
    if seg is None:
        return None
    silhouette = seg[0]
    cols = np.flatnonzero(silhouette.any(axis=0))
    if cols[0] <= 2 or cols[-1] >= silhouette.shape[1] - 3:
        return None  # mur pris pour le cycliste
    rows = np.flatnonzero(silhouette.any(axis=1))
    center_px = (rows[0] + rows[-1]) / 2
    offset_px = center_px - silhouette.shape[0] / 2
    return float(-offset_px * KINECT_VFOV_DEG / silhouette.shape[0])


def auto_tilt(
    read_depth: Callable[[], np.ndarray],
    set_tilt: TiltSetter,
    current_deg: float,
    intr: DepthIntrinsics,
    echo: Echo,
) -> float:
    """Affine l'inclinaison jusqu'à ce que le cycliste soit centré. Renvoie l'angle final."""
    for _ in range(MAX_TILT_STEPS):
        frames = [read_depth() for _ in range(10)]
        delta = tilt_correction_deg(frames, intr)
        if delta is None:
            echo("  Inclinaison : cycliste introuvable, angle conservé.")
            break
        if abs(delta) * 480 / KINECT_VFOV_DEG < TILT_TOLERANCE_PX:
            break
        target = max(-TILT_LIMIT_DEG, min(TILT_LIMIT_DEG, current_deg + delta))
        if round(target) == round(current_deg):
            break
        echo(f"  Inclinaison de la Kinect : {current_deg:+.0f}° -> {target:+.0f}°")
        set_tilt(target)
        current_deg = target
        _drain(read_depth, TILT_SETTLE_S)
    return current_deg


def _drain(read_depth: Callable[[], np.ndarray], seconds: float) -> None:
    """Lit (et jette) des frames pendant `seconds` : laisse le moteur finir, vide le tampon."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        read_depth()


# ── Enregistrement de la caméra de profil ─────────────────────────────────────

_FPS_MEASURE_S = 2.0
_MAX_PENDING_BYTES = 300 * 1024 * 1024


PREVIEW_DEPTH_RANGE_MM = (500.0, 4000.0)


def preview_image(depth: np.ndarray | None, bgr: np.ndarray, label: str) -> np.ndarray:
    """Image couleur | profondeur (fausses couleurs, noir = pas de mesure) + une ligne de texte."""
    h, w = bgr.shape[:2]
    if depth is None:
        depth_view = np.zeros_like(bgr)
    else:
        lo, hi = PREVIEW_DEPTH_RANGE_MM
        scaled = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
        depth_view = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
        depth_view[depth == 0] = 0
        if depth_view.shape[:2] != (h, w):
            depth_view = cv2.resize(depth_view, (w, h))
    panel = np.hstack([bgr, depth_view])
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return panel


class LivePreview:
    """Fenêtre de contrôle en direct (cv2.imshow) : se désactive seule si l'affichage échoue."""

    WINDOW = "BikeFit seance (couleur | profondeur)"

    def __init__(self) -> None:
        self.enabled = True

    def show(self, depth: np.ndarray | None, bgr: np.ndarray, label: str) -> None:
        if not self.enabled:
            return
        try:
            cv2.imshow(self.WINDOW, preview_image(depth, bgr, label))
            cv2.waitKey(1)
        except cv2.error:  # OpenCV sans interface graphique : on continue sans aperçu
            self.enabled = False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass


class LiveVideoRecorder:
    """Écrit une source live en MP4 au fps réellement obtenu, pas au fps annoncé.

    Les premières secondes sont tamponnées le temps de mesurer le débit (horloge murale),
    comme `DebugVideoWriter(measure_fps=True)` — sans le HUD qu'il superpose.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._pending: list[np.ndarray] = []
        self._pending_bytes = 0
        self._first_ts: int | None = None
        self._last_ts: int | None = None
        self.fps: float | None = None

    def write(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        if self._writer is not None:
            self._writer.write(frame_bgr)
            return
        if self._first_ts is None:
            self._first_ts = timestamp_ms
            h, w = frame_bgr.shape[:2]
            self._size = (w, h)
        self._last_ts = timestamp_ms
        self._pending.append(frame_bgr.copy())
        self._pending_bytes += frame_bgr.nbytes
        if (timestamp_ms - self._first_ts) / 1000 >= _FPS_MEASURE_S or (
            self._pending_bytes >= _MAX_PENDING_BYTES
        ):
            self._flush()

    def _flush(self) -> None:
        if self._size is None or self._first_ts is None or self._last_ts is None:
            return
        span_s = (self._last_ts - self._first_ts) / 1000
        self.fps = (
            (len(self._pending) - 1) / span_s if span_s > 0 and len(self._pending) > 1 else 30.0
        )
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, self._size
        )
        if not writer.isOpened():
            raise RuntimeError(f"Impossible de créer la vidéo : {self.path}")
        self._writer = writer
        for img in self._pending:
            writer.write(img)
        self._pending.clear()
        self._pending_bytes = 0

    def close(self) -> None:
        if self._writer is None:
            self._flush()
        if self._writer is not None:
            self._writer.release()


class ProfileRecorder:
    """Enregistre la caméra de profil dans un thread pendant que la Kinect capture."""

    def __init__(self, frames: Iterator[TimedFrame], path: Path, duration_s: float) -> None:
        self._frames = frames
        self._duration_ms = duration_s * 1000
        self.recorder = LiveVideoRecorder(path)
        self.first_frame = threading.Event()
        self.error: str | None = None
        self.n_frames = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self._frames:
                if frame.timestamp_ms > self._duration_ms:
                    break
                self.recorder.write(frame.frame_bgr, frame.timestamp_ms)
                self.n_frames += 1
                self.first_frame.set()
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur, jamais avalé
            self.error = f"{exc.__class__.__name__}: {exc}"
        finally:
            try:
                self.recorder.close()
            except Exception as exc:  # noqa: BLE001
                self.error = self.error or f"{exc.__class__.__name__}: {exc}"
            self.first_frame.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)


# ── Position aéro de référence ────────────────────────────────────────────────


class AeroReference(BaseModel):
    name: str
    date: str
    stats: FrontalAreaStats
    tilt_deg: float | None = None  # inclinaison Kinect de la référence (surface = f(inclinaison))


def save_reference(
    path: Path, name: str, stats: FrontalAreaStats, tilt_deg: float | None = None
) -> AeroReference:
    ref = AeroReference(
        name=name,
        date=datetime.now().isoformat(timespec="seconds"),
        stats=stats,
        tilt_deg=tilt_deg,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ref.model_dump_json(indent=2), encoding="utf-8")
    return ref


def load_reference(path: Path) -> AeroReference | None:
    if not path.exists():
        return None
    try:
        return AeroReference.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


# ── Orchestration ─────────────────────────────────────────────────────────────


@dataclass
class SeanceOptions:
    duration_s: float = 60.0
    countdown_s: float = 10.0
    profile_source: str | None = None  # webcam:N ou flux MJPEG ; None = Kinect seule
    side: str = "right"
    bike: str = "giant_trinity_2019"
    rider: str = "mensurations"
    target: str = "triathlon_XL"
    driver: str = "auto"
    set_reference: bool = False
    keep_raw: bool = False
    force: bool = False
    preview: bool = False  # fenêtre de contrôle en direct (activée par le CLI)
    auto_tilt: bool = True  # moteur : horizontale puis centrage du cycliste (SDK uniquement)
    tilt_deg: float = 0.0  # inclinaison de départ
    speed_kmh: float = 36.0
    distance_km: float = 180.0
    out_root: Path = field(default_factory=lambda: SEANCES_DIR)
    reference_file: Path = field(default_factory=lambda: AERO_REFERENCE_FILE)


class SeanceResult(BaseModel):
    folder: str
    window_s: tuple[float, float] | None
    window_detected: bool
    n_cycles: int | None
    aero: FrontalAreaStats | None
    aero_delta_pct: float | None = None
    aero_significant: bool | None = None
    tilt_deg: float | None = None
    reference_saved: bool = False
    cda_m2: float | None = None
    power_w: float | None = None
    session_id: int | None = None
    report: str | None = None
    angles_ok: list[str] = []
    angles_hors_cible: list[str] = []
    top_recommendation: str | None = None


def _countdown_and_preflight(
    read_depth: Callable[[], np.ndarray],
    opts: SeanceOptions,
    intr: DepthIntrinsics,
    echo: Echo,
    clock: Callable[[], float] = time.monotonic,
    refine: Callable[[], None] | None = None,
    on_frame: Callable[[float], None] | None = None,
) -> Preflight:
    """Vide le flux Kinect pendant le compte à rebours, puis juge le cadrage."""
    recent: deque[np.ndarray] = deque(maxlen=15)
    t_end = clock() + opts.countdown_s
    last_shown = -1
    echo("Mets-toi en position et pédale : vérification du cadrage pendant le compte à rebours.")
    while (remaining := t_end - clock()) > 0:
        recent.append(read_depth())
        if on_frame is not None:
            on_frame(remaining)
        if int(remaining) != last_shown:
            last_shown = int(remaining)
            echo(f"  Capture dans {last_shown + 1} s...")
    if refine is not None:
        refine()  # ex. centrage du cycliste par le moteur ; les frames d'avant sont périmées
        recent.clear()
    while len(recent) < 15:
        recent.append(read_depth())
    return preflight_depth(list(recent), intr)


def run_seance(
    opts: SeanceOptions,
    echo: Echo = print,
    kinect: KinectSources | None = None,
    profile_frames: Callable[[], Iterator[TimedFrame]] | None = None,
    window_fn: Callable[[Path, str], tuple[float, float, int] | None] | None = None,
    set_tilt: TiltSetter | None = None,
    preview: LivePreview | None = None,
) -> SeanceResult | None:
    """Une séance complète. None si interrompue (cadrage refusé, caméra muette)."""
    from bikefit.aero import analyze_frontal_area
    from bikefit.aero.capture import (
        RGB_NOMINAL_FPS,
        SDK_EXE,
        capture,
        freenect_sources,
        sdk_sources_with_motor,
    )
    from bikefit.aero.compare import compare_positions
    from bikefit.aero.recording import KINECT_V1_REGISTERED, RGB_FILE
    from bikefit.aero.render import SegmentationVideoWriter
    from bikefit.equipment import load_rider_profile

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = (
        opts.out_root / stamp
    )  # créé à la première écriture : pas de dossier vide si on abandonne

    if kinect is None:
        use_sdk = opts.driver == "sdk" or (opts.driver == "auto" and SDK_EXE.exists())
        if use_sdk:
            depth_src, rgb_src, stop_src, set_tilt = sdk_sources_with_motor()
            kinect = (depth_src, rgb_src, stop_src)
        else:
            kinect = freenect_sources()
            if opts.auto_tilt:
                echo("Moteur d'inclinaison : disponible avec le pilote SDK uniquement.")
    read_depth, read_rgb, stop_kinect = kinect
    if preview is None and opts.preview:
        preview = LivePreview()
    show_label = ["Kinect"]
    if preview is not None:
        raw_depth, raw_rgb, shown = read_depth, read_rgb, preview
        last_depth: list[np.ndarray | None] = [None]

        def read_depth() -> np.ndarray:
            depth = raw_depth()
            last_depth[0] = depth
            return depth

        def read_rgb() -> np.ndarray:
            bgr = raw_rgb()
            shown.show(last_depth[0], bgr, show_label[0])
            return bgr

    tilt = [opts.tilt_deg if set_tilt is not None and opts.auto_tilt else None]

    if opts.profile_source is not None and profile_frames is None:
        from bikefit.ingest import VideoReader

        reader = VideoReader(opts.profile_source)  # échoue tout de suite si caméra absente
        profile_frames = reader.frames

    profile_rec: ProfileRecorder | None = None
    try:
        read_depth()  # réveille le capteur (premières frames souvent vides)
        refine = None
        if set_tilt is not None and opts.auto_tilt:
            echo(f"Inclinaison de la Kinect : {opts.tilt_deg:+.0f}° (horizontale)...")
            set_tilt(opts.tilt_deg)
            _drain(read_depth, TILT_SETTLE_S)

            def refine() -> None:
                assert set_tilt is not None
                tilt[0] = auto_tilt(
                    read_depth, set_tilt, tilt[0] or 0.0, KINECT_V1_REGISTERED, echo
                )

        def on_countdown_frame(remaining: float) -> None:
            show_label[0] = f"Compte a rebours {remaining:.0f} s - cadre-toi, pedale"
            if preview is not None:
                read_rgb()

        pre = _countdown_and_preflight(
            read_depth,
            opts,
            KINECT_V1_REGISTERED,
            echo,
            refine=refine,
            on_frame=on_countdown_frame,
        )
        for w in pre.warnings:
            echo(f"  Attention : {w}")
        if not pre.ok:
            for e in pre.errors:
                echo(f"  Cadrage refusé : {e}")
            if not opts.force:
                echo("Corrige le cadrage et relance (ou --force pour capturer quand même).")
                return None
            echo("  --force : capture lancée malgré le cadrage.")

        if profile_frames is not None:
            profile_rec = ProfileRecorder(
                profile_frames(), folder / "profil.mp4", opts.duration_s + 1
            )
            profile_rec.start()
            if not profile_rec.first_frame.wait(10) or profile_rec.n_frames == 0:
                echo(f"Caméra de profil muette : {profile_rec.error or 'aucune image reçue'}.")
                return None

        echo(f"Capture de {opts.duration_s:.0f} s en cours... pédale normalement.")
        capture_start = time.monotonic()

        def read_rgb_recording() -> np.ndarray:
            elapsed = time.monotonic() - capture_start
            show_label[0] = f"REC {elapsed:.0f} / {opts.duration_s:.0f} s"
            return read_rgb()

        rec = capture(
            folder / "kinect",
            read_depth,
            read_rgb_recording,
            opts.duration_s,
            on_progress=lambda n, t: echo(f"  {t:5.0f} s") if n % 150 == 0 else None,
        )
    finally:
        stop_kinect()
        if preview is not None:
            preview.close()
        if profile_rec is not None:
            profile_rec.join(timeout=15)
    if profile_rec is not None and profile_rec.error:
        echo(f"Attention : caméra de profil interrompue ({profile_rec.error}).")
    echo(f"Capture terminée : {rec.frame_count} frames Kinect @ {rec.fps:.1f} fps.")

    # ── Fenêtre de pédalage (une détection, appliquée aux deux capteurs) ────────
    profile_mp4 = folder / "profil.mp4" if profile_rec is not None else None
    rgb_mp4 = folder / "kinect" / RGB_FILE
    window_video = profile_mp4 if profile_mp4 is not None else rgb_mp4
    echo("Détection de la fenêtre de pédalage...")
    detect = window_fn or detect_window_from_video
    found = detect(window_video, opts.side) if window_video.exists() else None
    detected = found is not None
    if found is not None:
        start_s, end_s, n_cycles = found
        # Temps de rgb.mp4 (fps nominal) -> temps réel Kinect si la fenêtre vient du RGB
        if window_video == rgb_mp4:
            scale = RGB_NOMINAL_FPS / rec.fps
            start_s, end_s = start_s * scale, end_s * scale
        echo(f"  Pédalage de {start_s:.0f} s à {end_s:.0f} s ({n_cycles} cycles).")
    else:
        # Marges fixes, bornées à une fraction de la durée (capture courte : jamais vide)
        skip_start = min(FALLBACK_SKIP_START_S, 0.15 * opts.duration_s)
        skip_end = min(FALLBACK_SKIP_END_S, 0.05 * opts.duration_s)
        start_s, end_s = skip_start, opts.duration_s - skip_end
        n_cycles = None
        echo(
            "  Pédalage non détecté de façon sûre : marges fixes"
            f" ({skip_start:.0f} s au début, {skip_end:.0f} s à la fin)."
        )

    # ── Aéro (Kinect) ──────────────────────────────────────────────────────────
    aero: FrontalAreaStats | None = None
    seg_writer = SegmentationVideoWriter(folder / "segmentation.mp4", rec.fps)
    try:
        aero, _ = analyze_frontal_area(rec, start_s=start_s, end_s=end_s, on_frame=seg_writer)
    except ValueError as exc:
        echo(f"Aéro non mesuré : {exc}")
    finally:
        seg_writer.close()

    result = SeanceResult(
        folder=str(folder),
        window_s=(start_s, end_s),
        window_detected=detected,
        n_cycles=n_cycles,
        aero=aero,
        tilt_deg=tilt[0],
    )

    rider_profile = load_rider_profile(opts.rider)
    cd = rider_profile.aero.cd if rider_profile is not None else None
    if aero is not None and aero.is_measured:
        reference = load_reference(opts.reference_file)
        named = [(reference.name, reference.stats)] if reference is not None else []
        named.append(("séance", aero))
        compared = compare_positions(named, cd, opts.speed_kmh, opts.distance_km)
        today = compared[-1]
        result.cda_m2, result.power_w = today.cda_m2, today.power_w
        if reference is not None:
            result.aero_delta_pct = today.delta_area_pct
            result.aero_significant = today.significant
            if (
                reference.tilt_deg is not None
                and tilt[0] is not None
                and abs(tilt[0] - reference.tilt_deg) > REFERENCE_TILT_WARN_DEG
            ):
                echo(
                    f"  Attention : inclinaison Kinect {tilt[0]:+.0f}° contre"
                    f" {reference.tilt_deg:+.0f}° pour la référence : la surface projetée"
                    " change légèrement avec l'angle, comparaison moins précise."
                )
        if opts.set_reference:
            save_reference(opts.reference_file, f"séance {stamp}", aero, tilt[0])
            result.reference_saved = True
            echo(f"  Position de référence enregistrée -> {opts.reference_file}")

    # ── Angles de profil + vue de face (MediaPipe) ─────────────────────────────
    if profile_mp4 is not None and profile_mp4.exists():
        _analyze_profile(
            opts, result, folder, profile_mp4, rgb_mp4, (start_s, end_s), rec.fps, echo
        )

    if not opts.keep_raw:
        shutil.rmtree(folder / "kinect" / "depth", ignore_errors=True)
    (folder / "seance.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _print_summary(result, opts, cd is not None, echo)
    return result


def run_seance_series(
    opts: SeanceOptions,
    every_min: float,
    times: int,
    echo: Echo = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    **inject: object,
) -> list[SeanceResult | None]:
    """Une capture courte toutes les `every_min` minutes pendant une longue sortie.

    Les cartes de profondeur pèsent ~0,5 Go/min : filmer 1 h 45 en continu est exclu.
    Des fenêtres de 60 s suffisent pour voir la position dériver avec la fatigue. La
    première capture prend `--set-reference` s'il est demandé ; une capture refusée
    (cadrage, caméra) ou en erreur n'arrête pas la série.
    """
    t0 = clock()
    results: list[SeanceResult | None] = []
    marks: list[float] = []
    for i in range(times):
        if i > 0:
            remaining = t0 + i * every_min * 60 - clock()
            if remaining > 0:
                echo(f"\nProchaine capture dans {remaining / 60:.0f} min : pédale tranquille.")
                sleep(remaining)
        minute = (clock() - t0) / 60
        echo(f"\n-- Capture {i + 1}/{times} (minute {minute:.0f}) --")
        marks.append(minute)
        try:
            results.append(
                run_seance(
                    replace(opts, set_reference=opts.set_reference and i == 0),
                    echo=echo,
                    **inject,  # type: ignore[arg-type]
                )
            )
        except Exception as exc:  # noqa: BLE001 — une capture ratée ne perd pas les autres
            echo(f"Capture {i + 1} en erreur : {exc.__class__.__name__}: {exc}")
            results.append(None)
    _print_drift(results, marks, opts, echo)
    return results


def _print_drift(
    results: list[SeanceResult | None], marks: list[float], opts: SeanceOptions, echo: Echo
) -> None:
    from bikefit.aero.compare import compare_positions
    from bikefit.equipment import load_rider_profile

    measured = [
        (f"minute {m:.0f}", r.aero)
        for m, r in zip(marks, results, strict=True)
        if r is not None and r.aero is not None and r.aero.is_measured
    ]
    echo("\n== Dérive de la position pendant la sortie ==")
    if len(measured) < 2:
        echo("Moins de 2 captures exploitables : pas de comparaison possible.")
        return
    rider = load_rider_profile(opts.rider)
    cd = rider.aero.cd if rider is not None else None
    for res in compare_positions(measured, cd, opts.speed_kmh, opts.distance_km):
        s = res.stats
        line = f"{res.name:>10} : A = {s.mean_m2:.3f} m²"
        if res.delta_area_pct is not None:
            verdict = "significatif" if res.significant else "bruit"
            line += f"  ({res.delta_area_pct:+.1f} % vs début, {verdict})"
            if res.delta_power_w is not None:
                line += f", {res.delta_power_w:+.1f} W"
        echo(line)
    echo("Hausse significative de la surface = tu te redresses ou écartes les coudes.")


def _analyze_profile(
    opts: SeanceOptions,
    result: SeanceResult,
    folder: Path,
    profile_mp4: Path,
    rgb_mp4: Path,
    window: tuple[float, float],
    kinect_fps: float,
    echo: Echo,
) -> None:
    from bikefit._paths import REPORTS_DIR
    from bikefit.aero.capture import RGB_NOMINAL_FPS
    from bikefit.cli import analyze

    start_s, end_s = window
    report = REPORTS_DIR / f"seance_{folder.name}.html"
    metrics = folder / "metriques.json"
    # La fenêtre est en secondes réelles ; rgb.mp4 est écrit à fps nominal.
    to_rgb = kinect_fps / RGB_NOMINAL_FPS
    log = io.StringIO()
    echo("Analyse des angles (MediaPipe, quelques instants)...")

    def run(front: Path | None) -> None:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            analyze(
                video=str(profile_mp4),
                front=str(front) if front is not None else None,
                side=opts.side,
                start=start_s,
                end=end_s,
                front_start=start_s * to_rgb,
                front_end=end_s * to_rgb,
                bike=opts.bike,
                profile=opts.target,
                rider=opts.rider,
                output=metrics,
                report=report,
                save_session=True,
            )

    try:
        run(rgb_mp4 if rgb_mp4.exists() else None)
    except Exception as exc:  # noqa: BLE001 — la vue de face ne doit pas faire perdre le profil
        echo(f"  Vue de face inexploitable ({exc.__class__.__name__}) : analyse du profil seul.")
        try:
            run(None)
        except Exception as exc2:  # noqa: BLE001
            echo(f"  Analyse du profil impossible : {exc2.__class__.__name__}: {exc2}")
            (folder / "analyse.log").write_text(log.getvalue(), encoding="utf-8")
            return
    (folder / "analyse.log").write_text(log.getvalue(), encoding="utf-8")

    if metrics.exists():
        rapport = json.loads(metrics.read_text(encoding="utf-8")).get("rapport", {})
        result.angles_ok = rapport.get("angles_ok", [])
        result.angles_hors_cible = rapport.get("angles_hors_cible", [])
        recos = rapport.get("recommandations", [])
        if recos:
            r = recos[0]
            delta = f" ({r['delta_mm']:+.0f} mm)" if r.get("delta_mm") is not None else ""
            result.top_recommendation = f"{r['reglage']} : {r['direction']}{delta}"
    match = re.findall(r"Session #(\d+)", log.getvalue())
    result.session_id = int(match[-1]) if match else None
    result.report = str(report) if report.exists() else None


def _print_summary(result: SeanceResult, opts: SeanceOptions, has_cd: bool, echo: Echo) -> None:
    echo("\n== Séance ==")
    if result.window_s is not None:
        start_s, end_s = result.window_s
        how = f"{result.n_cycles} cycles" if result.window_detected else "marges fixes"
        echo(f"Pédalage analysé : {start_s:.0f}-{end_s:.0f} s ({how}).")
    if result.angles_ok or result.angles_hors_cible:
        echo(
            f"Angles : {len(result.angles_ok)} dans la cible,"
            f" {len(result.angles_hors_cible)} hors cible"
            + (f" ({', '.join(result.angles_hors_cible)})" if result.angles_hors_cible else "")
        )
        if result.top_recommendation:
            echo(f"Priorité : {result.top_recommendation}")
    elif opts.profile_source is not None:
        echo("Angles : non disponibles (voir analyse.log dans le dossier de la séance).")
    a = result.aero
    if a is not None and a.is_measured:
        line = f"Aéro : A = {a.mean_m2:.3f} m² ± {a.std_m2:.3f} ({a.n_windows} fenêtres)"
        if result.cda_m2 is not None and result.power_w is not None:
            line += (
                f", CdA ~ {result.cda_m2:.3f} -> {result.power_w:.0f} W à {opts.speed_kmh:.0f} km/h"
            )
        echo(line)
        if result.tilt_deg is not None:
            echo(f"  Inclinaison Kinect : {result.tilt_deg:+.0f}°")
        if result.aero_delta_pct is not None:
            verdict = "significatif" if result.aero_significant else "dans le bruit de mesure"
            echo(f"  vs référence : {result.aero_delta_pct:+.1f} % ({verdict})")
        elif result.reference_saved:
            echo("  Référence enregistrée : les prochaines séances seront comparées à celle-ci.")
        else:
            echo("  Pas de référence : relancer avec --set-reference sur ta meilleure position.")
        if not has_cd:
            echo("  Cd non calibré : surface seulement (cf. docs/kinect-aero.md).")
        for issue in a.reliability_issues:
            echo(f"  Peu fiable : {issue}")
    if result.report:
        echo(f"Rapport : {result.report}")
    if result.session_id is not None:
        echo(f"Session #{result.session_id} enregistrée (bikefit history / compare).")
    echo(f"Dossier : {result.folder}")
