"""Tests de `bikefit seance` : briques pures + séance complète avec une Kinect factice."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from bikefit.aero import KINECT_V1_REGISTERED, DepthRecording
from bikefit.aero.area import FrontalAreaStats
from bikefit.ingest.reader import TimedFrame
from bikefit.seance import (
    TILT_LIMIT_DEG,
    KinectSources,
    LiveVideoRecorder,
    ProfileRecorder,
    SeanceOptions,
    auto_tilt,
    load_reference,
    pedaling_window,
    preflight_depth,
    run_seance,
    run_seance_series,
    save_reference,
    tilt_correction_deg,
)
from tests.test_aero import _draw_rect, _empty_scene, _rider

INTR = KINECT_V1_REGISTERED


def _cycles(n: int, length: int = 30, start: int = 0) -> list[tuple[int, int]]:
    return [(start + i * length, start + (i + 1) * length) for i in range(n)]


# ── Fenêtre de pédalage ───────────────────────────────────────────────────────


class TestPedalingWindow:
    def test_contiguous_cycles(self) -> None:
        window = pedaling_window(_cycles(10, 30, start=60), fps=30.0)
        assert window == (2.0, 12.0, 10)

    def test_longest_run_wins_over_short_burst(self) -> None:
        # 3 cycles parasites (montée), long silence, puis 12 vrais cycles
        cycles = _cycles(3, 30, 0) + _cycles(12, 30, 600)
        window = pedaling_window(cycles, fps=30.0)
        assert window is not None
        assert window[0] == 20.0 and window[2] == 12

    def test_single_rejected_cycle_does_not_split_the_run(self) -> None:
        cycles = _cycles(6, 30, 0) + _cycles(6, 30, 210)  # un cycle (180-210) manquant
        window = pedaling_window(cycles, fps=30.0)
        assert window is not None
        assert window[2] == 12  # une seule suite, pas deux de 6

    def test_too_few_cycles_returns_none(self) -> None:
        assert pedaling_window(_cycles(4), fps=30.0) is None

    def test_empty_or_bad_fps(self) -> None:
        assert pedaling_window([], fps=30.0) is None
        assert pedaling_window(_cycles(8), fps=0.0) is None


# ── Vérification du cadrage ───────────────────────────────────────────────────


class TestPreflight:
    def test_well_framed_rider_is_ok(self) -> None:
        pf = preflight_depth([_rider(z=2500.0).astype(np.uint16)] * 5, INTR)
        assert pf.ok, pf.errors
        assert pf.depth_ref_mm == pytest.approx(2500.0, abs=50)

    def test_too_close(self) -> None:
        pf = preflight_depth([_rider(z=1000.0).astype(np.uint16)] * 5, INTR)
        assert not pf.ok
        assert any("trop près" in e for e in pf.errors)

    def test_off_center_rider(self) -> None:
        depth = _draw_rect(_empty_scene(True), 900, 1400, -300, 800, 2500.0)
        pf = preflight_depth([depth.astype(np.uint16)] * 5, INTR)
        assert not pf.ok

    def test_empty_room_is_refused(self) -> None:
        pf = preflight_depth([_empty_scene(True).astype(np.uint16)] * 5, INTR)
        assert not pf.ok  # le mur du fond ne doit pas passer pour un cycliste

    def test_no_depth_at_all(self) -> None:
        pf = preflight_depth([np.zeros((480, 640), np.uint16)] * 5, INTR)
        assert not pf.ok
        assert pf.depth_ref_mm is None


# ── Enregistrement de la caméra de profil ─────────────────────────────────────


def _timed(n: int, step_ms: int = 40) -> list[TimedFrame]:
    return [
        TimedFrame(
            timestamp_ms=i * step_ms,
            frame_idx=i,
            frame_bgr=np.full((48, 64, 3), i % 255, np.uint8),
        )
        for i in range(n)
    ]


class TestLiveVideoRecorder:
    def test_writes_at_measured_fps(self, tmp_path: Path) -> None:
        rec = LiveVideoRecorder(tmp_path / "p.mp4")
        for f in _timed(100, step_ms=40):  # 25 fps réels sur 4 s
            rec.write(f.frame_bgr, f.timestamp_ms)
        rec.close()
        assert rec.fps == pytest.approx(25.0, rel=0.02)
        cap = cv2.VideoCapture(str(tmp_path / "p.mp4"))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 100
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, rel=0.02)
        cap.release()

    def test_short_capture_is_still_written(self, tmp_path: Path) -> None:
        rec = LiveVideoRecorder(tmp_path / "short.mp4")
        for f in _timed(10):
            rec.write(f.frame_bgr, f.timestamp_ms)
        rec.close()
        assert (tmp_path / "short.mp4").exists()


class TestProfileRecorder:
    def test_stops_at_duration(self, tmp_path: Path) -> None:
        pr = ProfileRecorder(iter(_timed(200, step_ms=40)), tmp_path / "p.mp4", duration_s=2.0)
        pr.start()
        pr.join(5)
        assert pr.error is None
        assert 45 <= pr.n_frames <= 52  # 2 s à 25 fps
        assert pr.first_frame.is_set()

    def test_error_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        def broken() -> Iterator[TimedFrame]:
            yield _timed(1)[0]
            raise RuntimeError("caméra débranchée")

        pr = ProfileRecorder(broken(), tmp_path / "p.mp4", duration_s=5.0)
        pr.start()
        pr.join(5)
        assert pr.error is not None and "caméra débranchée" in pr.error
        assert pr.n_frames == 1


# ── Référence aéro ────────────────────────────────────────────────────────────


def test_reference_roundtrip_and_missing(tmp_path: Path) -> None:
    stats = FrontalAreaStats(
        mean_m2=0.41,
        std_m2=0.01,
        min_m2=0.39,
        max_m2=0.43,
        n_windows=12,
        window_s=0.7,
        period_detected=True,
        hole_ratio=0.02,
        frames_kept=0.97,
        depth_ref_mm=2400.0,
        near_mm=1500.0,
        far_mm=3300.0,
        floor_removed=True,
    )
    path = tmp_path / "ref.json"
    assert load_reference(path) is None
    save_reference(path, "ma ref", stats)
    ref = load_reference(path)
    assert ref is not None and ref.name == "ma ref" and ref.stats.mean_m2 == 0.41
    path.write_text("pas du json", encoding="utf-8")
    assert load_reference(path) is None


# ── Séance complète (Kinect factice, pas de MediaPipe) ────────────────────────


def _fake_kinect(scene: np.ndarray) -> KinectSources:
    depth = scene.astype(np.uint16)

    def read_depth() -> np.ndarray:
        time.sleep(0.04)  # ~25 fps
        return depth.copy()

    def read_rgb() -> np.ndarray:
        return np.zeros((480, 640, 3), np.uint8)

    return read_depth, read_rgb, lambda: None


def _opts(tmp_path: Path, **kw: bool) -> SeanceOptions:
    return SeanceOptions(
        duration_s=2.2,
        countdown_s=0.3,
        rider="__absent__",
        out_root=tmp_path / "seances",
        reference_file=tmp_path / "ref.json",
        **kw,
    )


def test_full_seance_kinect_only_then_compare_to_reference(tmp_path: Path) -> None:
    logs: list[str] = []
    kinect = _fake_kinect(_rider(z=2500.0))

    first = run_seance(
        _opts(tmp_path, set_reference=True),
        echo=logs.append,
        kinect=kinect,
        window_fn=lambda video, side: (0.5, 2.0, 7),
    )
    assert first is not None and first.aero is not None and first.aero.is_measured
    assert first.window_detected and first.window_s is not None
    # Fenêtre issue de rgb.mp4 (fps nominal 30) -> convertie en temps réel Kinect
    fps = DepthRecording(Path(first.folder) / "kinect").fps
    assert first.window_s == pytest.approx((0.5 * 30 / fps, 2.0 * 30 / fps))
    assert first.aero.mean_m2 == pytest.approx(0.55, rel=0.05)  # 0,5 × 1,1 m
    assert (tmp_path / "ref.json").exists()
    assert first.reference_saved
    assert not any("Pas de référence" in line for line in logs)
    assert any("Référence enregistrée :" in line for line in logs)
    folder = Path(first.folder)
    assert (folder / "seance.json").exists()
    assert (folder / "segmentation.mp4").exists()
    assert not (folder / "kinect" / "depth").exists()  # brut supprimé par défaut

    time.sleep(1.1)  # horodatage de dossier distinct
    second = run_seance(
        _opts(tmp_path),
        echo=logs.append,
        kinect=kinect,
        window_fn=lambda video, side: None,  # pédalage non détecté -> marges fixes
    )
    assert second is not None and not second.window_detected
    assert second.aero_delta_pct is not None
    assert abs(second.aero_delta_pct) < 1.0  # même scène, même surface
    assert any("vs référence" in line for line in logs)
    assert any("marges fixes" in line for line in logs)


def test_seance_refused_when_framing_is_wrong(tmp_path: Path) -> None:
    logs: list[str] = []
    result = run_seance(
        _opts(tmp_path),
        echo=logs.append,
        kinect=_fake_kinect(_empty_scene(True)),  # pièce vide : pas de cycliste
        window_fn=lambda v, s: None,
    )
    assert result is None
    assert any("Cadrage refusé" in line for line in logs)
    assert not list((tmp_path / "seances").glob("*/kinect"))  # rien capturé


def test_force_captures_despite_bad_framing(tmp_path: Path) -> None:
    result = run_seance(
        _opts(tmp_path, force=True),
        echo=lambda _: None,
        kinect=_fake_kinect(_empty_scene(True)),
        window_fn=lambda v, s: None,
    )
    assert result is not None


def test_series_spaces_captures_and_reports_drift(tmp_path: Path) -> None:
    logs: list[str] = []
    waits: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    # Capture 1 : posture de base ; capture 2 : le cycliste s'élargit (jambe latérale)
    base = _rider(z=2500.0).astype(np.uint16)
    wider = _rider(z=2500.0, leg_mm=150.0).astype(np.uint16)
    runs = [0]

    def read_depth() -> np.ndarray:
        time.sleep(0.04)
        return (base if runs[0] == 0 else wider).copy()

    def stop() -> None:
        runs[0] += 1

    out = run_seance_series(
        _opts(tmp_path, set_reference=True),
        every_min=1.0,
        times=2,
        echo=logs.append,
        sleep=sleep,
        clock=lambda: now[0],
        kinect=(read_depth, lambda: np.zeros((480, 640, 3), np.uint8), stop),
        window_fn=lambda v, s: (0.2, 2.0, 7),
    )
    assert len(out) == 2 and all(r is not None for r in out)
    assert waits == [pytest.approx(60.0)]  # 2e capture à t0 + 1 min
    assert any("Dérive de la position" in line for line in logs)
    drift = [line for line in logs if "vs début" in line]
    # 2 s de capture = 1-2 fenêtres : écart chiffré, mais jamais déclaré significatif
    assert drift and "+1" in drift[0] and "bruit" in drift[0]
    first, second = out
    assert first is not None and second is not None
    assert first.aero is not None and second.aero is not None
    assert second.aero.mean_m2 > first.aero.mean_m2 * 1.05


def test_series_survives_a_failed_capture(tmp_path: Path) -> None:
    logs: list[str] = []

    def boom() -> np.ndarray:
        raise RuntimeError("Kinect débranchée")

    out = run_seance_series(
        _opts(tmp_path),
        every_min=1.0,
        times=2,
        echo=logs.append,
        sleep=lambda s: None,
        clock=lambda: 0.0,
        kinect=(boom, lambda: np.zeros((480, 640, 3), np.uint8), lambda: None),
        window_fn=lambda v, s: None,
    )
    assert out == [None, None]
    assert sum("en erreur" in line for line in logs) == 2
    assert any("Moins de 2 captures" in line for line in logs)


# ── Moteur d'inclinaison ──────────────────────────────────────────────────────

PX_PER_DEG = 480 / 43.0


def _scene_shifted(rows: float) -> np.ndarray:
    """Cycliste centré (y de -550 à 550 mm à 2,5 m), décalé de `rows` pixels vers le bas."""
    dy = rows * 2500.0 / INTR.fy
    depth = _draw_rect(_empty_scene(True), -250, 250, -550 + dy, 550 + dy, 2500.0)
    return depth.astype(np.uint16)


class TestTilt:
    def test_centered_rider_needs_no_correction(self) -> None:
        delta = tilt_correction_deg([_scene_shifted(0)] * 5, INTR)
        assert delta is not None and abs(delta) < 1.0

    def test_rider_too_low_tilts_down_and_too_high_tilts_up(self) -> None:
        low = tilt_correction_deg([_scene_shifted(80)] * 5, INTR)
        high = tilt_correction_deg([_scene_shifted(-80)] * 5, INTR)
        assert low == pytest.approx(-80 / PX_PER_DEG, abs=1.0)
        assert high == pytest.approx(80 / PX_PER_DEG, abs=1.0)

    def test_empty_room_gives_no_correction(self) -> None:
        assert tilt_correction_deg([_empty_scene(True).astype(np.uint16)] * 5, INTR) is None

    def _motor(self, initial_rows: float) -> tuple[object, object, list[float]]:
        state = {"tilt": 0.0}
        calls: list[float] = []

        def read_depth() -> np.ndarray:
            time.sleep(0.005)
            return _scene_shifted(initial_rows + state["tilt"] * PX_PER_DEG)

        def set_tilt(deg: float) -> None:
            calls.append(deg)
            state["tilt"] = float(round(deg))

        return read_depth, set_tilt, calls

    def test_auto_tilt_centers_the_rider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("bikefit.seance.TILT_SETTLE_S", 0.05)
        read_depth, set_tilt, calls = self._motor(80)
        final = auto_tilt(read_depth, set_tilt, 0.0, INTR, lambda _: None)  # type: ignore[arg-type]
        assert final == pytest.approx(-7.0, abs=1.0)
        assert len(calls) == 1  # un seul mouvement suffit, pas d'oscillation

    def test_auto_tilt_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("bikefit.seance.TILT_SETTLE_S", 0.05)
        read_depth, set_tilt, calls = self._motor(160)  # il faudrait -14° puis plus
        final = auto_tilt(read_depth, set_tilt, 0.0, INTR, lambda _: None)  # type: ignore[arg-type]
        assert abs(final) <= TILT_LIMIT_DEG
        assert len(calls) <= 2  # MAX_TILT_STEPS

    def test_auto_tilt_keeps_angle_when_rider_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("bikefit.seance.TILT_SETTLE_S", 0.05)
        calls: list[float] = []
        final = auto_tilt(
            lambda: _empty_scene(True).astype(np.uint16),
            calls.append,
            3.0,
            INTR,
            lambda _: None,
        )
        assert final == 3.0 and calls == []


def test_seance_runs_motor_then_captures_centered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bikefit.seance.TILT_SETTLE_S", 0.05)
    state = {"tilt": 0.0}
    sent: list[float] = []

    def read_depth() -> np.ndarray:
        time.sleep(0.02)
        return _scene_shifted(80 + state["tilt"] * PX_PER_DEG)

    def set_tilt(deg: float) -> None:
        sent.append(deg)
        state["tilt"] = float(round(deg))

    logs: list[str] = []
    result = run_seance(
        _opts(tmp_path),
        echo=logs.append,
        kinect=(read_depth, lambda: np.zeros((480, 640, 3), np.uint8), lambda: None),
        set_tilt=set_tilt,
        window_fn=lambda v, s: (0.2, 2.0, 7),
    )
    assert result is not None and result.aero is not None
    assert sent[0] == 0.0  # d'abord horizontale
    assert result.tilt_deg == pytest.approx(-7.0, abs=1.0)
    assert any("Inclinaison de la Kinect" in line for line in logs)
    assert any("Inclinaison Kinect : -" in line for line in logs)  # affichée au résumé


def test_no_auto_tilt_never_touches_the_motor(tmp_path: Path) -> None:
    sent: list[float] = []
    result = run_seance(
        _opts(tmp_path, auto_tilt=False),
        echo=lambda _: None,
        kinect=_fake_kinect(_rider(z=2500.0)),
        set_tilt=sent.append,
        window_fn=lambda v, s: None,
    )
    assert result is not None and sent == [] and result.tilt_deg is None


def test_every_printed_line_is_encodable_on_a_windows_console(tmp_path: Path) -> None:
    """cp1252 (console Windows) : un seul caractère exotique fait planter l'affichage."""
    logs: list[str] = []
    run_seance(
        _opts(tmp_path, set_reference=True),
        echo=logs.append,
        kinect=_fake_kinect(_rider(z=2500.0)),
        window_fn=lambda v, s: (0.2, 2.0, 7),
    )
    run_seance_series(
        _opts(tmp_path),
        every_min=1.0,
        times=2,
        echo=logs.append,
        sleep=lambda s: None,
        clock=lambda: 0.0,
        kinect=_fake_kinect(_rider(z=2500.0)),
        window_fn=lambda v, s: None,
    )
    assert logs
    for line in logs:
        line.encode("cp1252")  # lève UnicodeEncodeError au moindre caractère non supporté
