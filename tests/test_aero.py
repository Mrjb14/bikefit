"""Tests du module aero sur des scènes de profondeur synthétiques à géométrie connue.

Scène type (repère caméra, mm, y vers le bas) : caméra Kinect v1 à 1 m du sol, mur du
fond à 4,5 m, sol horizontal, « cycliste » = rectangle(s) fronto-parallèle(s) de
dimensions connues : la surface attendue est calculable exactement.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bikefit.aero import (
    KINECT_V1_REGISTERED,
    DepthRecording,
    aero_power_w,
    analyze_frontal_area,
    calibrate_cd,
    cd_is_plausible,
    save_depth_recording,
    time_saved_s,
)
from bikefit.aero.area import (
    dominant_period_frames,
    fit_floor_plane,
    frame_area_m2,
    reference_depth_mm,
    segment_rider,
)
from bikefit.aero.intrinsics import estimate_fx_from_marker

INTR = KINECT_V1_REGISTERED
H, W = 480, 640
WALL_MM = 4500.0
CAMERA_HEIGHT_MM = 1000.0


def _build_scene(floor: bool) -> np.ndarray:
    depth = np.full((H, W), WALL_MM)
    if floor:
        v = np.arange(H, dtype=float)
        with np.errstate(divide="ignore"):
            z_floor = np.where(v > INTR.cy, CAMERA_HEIGHT_MM * INTR.fy / (v - INTR.cy), np.inf)
        rows = z_floor < WALL_MM
        depth[rows, :] = z_floor[rows, None]
    return depth


_SCENES = {True: _build_scene(True), False: _build_scene(False)}


def _empty_scene(floor: bool = True) -> np.ndarray:
    return _SCENES[floor].copy()


def _draw_rect(
    depth: np.ndarray, x0: float, x1: float, y0: float, y1: float, z: float
) -> np.ndarray:
    """Rectangle fronto-parallèle [x0,x1]×[y0,y1] (mm) à la profondeur z, avec occlusion."""
    u0 = int(math.ceil(INTR.cx + INTR.fx * x0 / z))
    u1 = int(math.ceil(INTR.cx + INTR.fx * x1 / z))
    v0 = int(math.ceil(INTR.cy + INTR.fy * y0 / z))
    v1 = int(math.ceil(INTR.cy + INTR.fy * y1 / z))
    region = depth[max(0, v0) : min(H, v1), max(0, u0) : min(W, u1)]
    np.minimum(region, z, out=region)
    return depth


def _rider(z: float = 2500.0, floor: bool = True, leg_mm: float = 0.0) -> np.ndarray:
    """Buste+vélo : 500 × 1100 mm (y de -300 à 800), plus une « jambe » latérale
    de largeur leg_mm sur 600 mm de haut, collée au bloc."""
    depth = _draw_rect(_empty_scene(floor), -250, 250, -300, 800, z)
    if leg_mm > 0:
        _draw_rect(depth, 250, 250 + leg_mm, 200, 800, z)
    return depth


BODY_M2 = 0.5 * 1.1


def _area(depth: np.ndarray, near: float = 1600, far: float = 3400) -> tuple[float, float]:
    seg = segment_rider(depth.astype(np.uint16), INTR, near, far)
    assert seg is not None
    return frame_area_m2(depth.astype(np.uint16), seg[0], seg[1], INTR)


# ── Format d'enregistrement ───────────────────────────────────────────────────


class TestRecording:
    def test_roundtrip(self, tmp_path: Path) -> None:
        frames = [_rider().astype(np.uint16) for _ in range(3)]
        rec = save_depth_recording(tmp_path / "rec", frames, fps=30.0, intrinsics=INTR)

        loaded = DepthRecording(tmp_path / "rec")
        assert loaded.frame_count == 3
        assert loaded.meta.width == W and loaded.meta.height == H
        assert loaded.meta.timestamps_ms == [0, 33, 67]
        assert loaded.intrinsics == INTR
        np.testing.assert_array_equal(loaded.read(1), frames[1])
        assert [f.frame_idx for f in rec.frames(every=2)] == [0, 2]
        assert loaded.rgb_path is None

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DepthRecording(tmp_path)


# ── Aire d'une frame ──────────────────────────────────────────────────────────


class TestFrameArea:
    def test_known_rectangle_area(self) -> None:
        area, holes = _area(_rider(floor=False))
        assert area == pytest.approx(BODY_M2, rel=0.02)
        assert holes == 0.0

    def test_area_independent_of_distance(self) -> None:
        """Propriété clé : reculer la caméra ne change pas la surface en m²."""
        near, _ = _area(_rider(z=2000, floor=False), near=1000, far=3000)
        far_, _ = _area(_rider(z=3000, floor=False), near=2000, far=4000)
        assert near == pytest.approx(far_, rel=0.02)

    def test_depth_dropout_inside_silhouette_is_filled(self) -> None:
        """Cuissard noir : la Kinect ne mesure rien sur une zone du buste (profondeur 0).
        Ces pixels font partie du cycliste et doivent compter dans la surface."""
        depth = _rider(floor=False)
        depth[260:300, 300:340] = 0  # trou intérieur
        area, holes = _area(depth)
        assert area == pytest.approx(BODY_M2, rel=0.02)
        assert holes == pytest.approx(40 * 40 / (BODY_M2 * 1e6 / (2500 / INTR.fx) ** 2), rel=0.05)

    def test_see_through_gap_is_excluded(self) -> None:
        """Espace entre bras et buste : on y mesure le mur, c'est de l'air -> exclu."""
        depth = _rider(floor=False)
        depth[260:300, 300:340] = WALL_MM
        area, holes = _area(depth)
        gap_m2 = 40 * 40 * (2500 / INTR.fx) ** 2 / 1e6
        assert area == pytest.approx(BODY_M2 - gap_m2, rel=0.02)
        assert holes == 0.0

    def test_nothing_in_band_returns_none(self) -> None:
        assert segment_rider(_empty_scene(floor=False).astype(np.uint16), INTR, 1600, 3400) is None


# ── Sol et profondeur de référence ────────────────────────────────────────────


class TestFloor:
    def test_floor_plane_recovered(self) -> None:
        plane = fit_floor_plane(_rider().astype(np.uint16), INTR)
        assert plane is not None
        assert plane.ny == pytest.approx(-1.0, abs=0.01)
        assert plane.d == pytest.approx(CAMERA_HEIGHT_MM, abs=15)  # hauteur de la caméra

    def test_no_floor_visible_returns_none(self) -> None:
        assert fit_floor_plane(_rider(floor=False).astype(np.uint16), INTR) is None

    def test_floor_removal_prevents_inflated_area(self) -> None:
        depth = _rider().astype(np.uint16)
        plane = fit_floor_plane(depth, INTR)
        with_floor = segment_rider(depth, INTR, 1600, 3400, floor=None)
        without_floor = segment_rider(depth, INTR, 1600, 3400, floor=plane)
        assert with_floor is not None and without_floor is not None
        inflated, _ = frame_area_m2(depth, *with_floor, INTR)
        clean, _ = frame_area_m2(depth, *without_floor, INTR)
        assert inflated > clean * 1.3  # le sol dans la bande gonfle fortement la surface
        assert clean == pytest.approx(BODY_M2, rel=0.02)

    def test_reference_depth_targets_torso(self) -> None:
        assert reference_depth_mm(_rider(z=2500).astype(np.uint16)) == pytest.approx(2500)


# ── Période de pédalage ───────────────────────────────────────────────────────


class TestPeriod:
    def test_sinusoid_period(self) -> None:
        t = np.arange(300)
        assert dominant_period_frames(np.sin(2 * np.pi * t / 20), fps=30) == 20

    def test_harmonic_not_mistaken_for_multiple(self) -> None:
        t = np.arange(300)
        signal = np.sin(2 * np.pi * t / 15) + 0.3 * np.sin(2 * np.pi * t / 30)
        assert dominant_period_frames(signal, fps=30) in (15, 30)

    def test_constant_signal_has_no_period(self) -> None:
        assert dominant_period_frames(np.full(200, 0.4), fps=30) is None

    def test_noise_has_no_period(self) -> None:
        rng = np.random.default_rng(1)
        assert dominant_period_frames(rng.normal(size=300), fps=30) is None

    def test_tolerates_nan(self) -> None:
        t = np.arange(300)
        signal = np.sin(2 * np.pi * t / 24)
        signal[50:55] = np.nan
        assert dominant_period_frames(signal, fps=30) == 24


# ── Analyse complète d'un enregistrement ──────────────────────────────────────


def _pedaling_recording(
    path: Path, n: int = 200, period: int = 20, dropout: bool = False
) -> DepthRecording:
    def gen():  # type: ignore[no-untyped-def]
        for i in range(n):
            leg = 50 + 40 * math.sin(2 * math.pi * i / period)
            depth = _rider(leg_mm=leg)
            if dropout:
                depth[200:330, 280:360] = 0  # ~30 % de la silhouette sans mesure
            yield depth.astype(np.uint16)

    return save_depth_recording(path, gen(), fps=30.0, intrinsics=INTR)


class TestAnalyzeFrontalArea:
    def test_mean_area_over_pedaling_windows(self, tmp_path: Path) -> None:
        stats, series = analyze_frontal_area(_pedaling_recording(tmp_path / "r"))

        expected = BODY_M2 + 0.6 * 0.050  # jambe moyenne : 50 mm × 600 mm
        assert stats.mean_m2 == pytest.approx(expected, rel=0.02)
        assert stats.period_detected
        assert stats.window_s == pytest.approx(20 / 30, abs=0.01)
        assert stats.n_windows == 10
        assert stats.floor_removed
        assert stats.depth_ref_mm == pytest.approx(2500)
        assert stats.is_reliable, stats.reliability_issues
        assert len(series) == 200 and not np.isnan(series).any()

    def test_start_end_trim(self, tmp_path: Path) -> None:
        stats, series = analyze_frontal_area(
            _pedaling_recording(tmp_path / "r"), start_s=1.0, end_s=5.0
        )
        assert len(series) == 120
        assert stats.n_windows == 6

    def test_heavy_dropout_flagged_unreliable(self, tmp_path: Path) -> None:
        stats, _ = analyze_frontal_area(_pedaling_recording(tmp_path / "r", dropout=True))
        assert not stats.is_reliable
        assert any("sans mesure" in issue for issue in stats.reliability_issues)

    def test_empty_center_raises(self, tmp_path: Path) -> None:
        rec = save_depth_recording(
            tmp_path / "r",
            [np.zeros((H, W), dtype=np.uint16)] * 3,
            fps=30.0,
            intrinsics=INTR,
        )
        with pytest.raises(ValueError, match="centre"):
            analyze_frontal_area(rec)


# ── Traînée ───────────────────────────────────────────────────────────────────


class TestDrag:
    def test_aero_power(self) -> None:
        # ½ · 1.225 · 0.25 · 10³ = 153.1 W à 36 km/h
        assert aero_power_w(0.25, 36.0) == pytest.approx(153.125)

    def test_calibrate_cd(self) -> None:
        assert calibrate_cd(0.245, 0.35) == pytest.approx(0.7)
        assert cd_is_plausible(0.7)
        assert not cd_is_plausible(2.0)
        with pytest.raises(ValueError):
            calibrate_cd(0.245, 0.0)

    def test_time_saved_sign_and_magnitude(self) -> None:
        gain = time_saved_s(0.25, 0.24, distance_km=180, speed_kmh=36)
        assert gain > 0
        # 4 % de CdA en moins -> ~1,37 % de vitesse -> ~4 min sur 5 h
        assert 200 < gain < 260
        assert time_saved_s(0.24, 0.25, 180, 36) < 0


# ── Vérification des intrinsèques au marqueur ArUco ───────────────────────────


def test_estimate_fx_from_marker() -> None:
    import cv2

    side_px = 70
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side_px
    )
    frame = np.full((H, W), 255, dtype=np.uint8)
    frame[200 : 200 + side_px, 300 : 300 + side_px] = marker
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Marqueur de 100 mm vu à 70 px : il est à z = fx·100/70 = 750 mm
    depth = np.full((H, W), 750, dtype=np.uint16)

    fx = estimate_fx_from_marker(frame_bgr, depth, marker_size_mm=100)
    assert fx is not None
    assert fx == pytest.approx(INTR.fx, rel=0.03)


# ── Comparaison de positions ──────────────────────────────────────────────────


def _stats(mean: float, std: float = 0.004, n: int = 12) -> object:
    from bikefit.aero.area import FrontalAreaStats

    return FrontalAreaStats(
        mean_m2=mean,
        std_m2=std,
        min_m2=mean - std,
        max_m2=mean + std,
        n_windows=n,
        window_s=0.68,
        period_detected=True,
        hole_ratio=0.02,
        frames_kept=1.0,
        depth_ref_mm=2400,
        near_mm=1500,
        far_mm=3300,
        floor_removed=True,
    )


class TestComparePositions:
    def test_without_cd_only_relative(self) -> None:
        from bikefit.aero.compare import compare_positions

        res = compare_positions([("a", _stats(0.40)), ("b", _stats(0.38))], None, 36, 180)
        assert res[0].delta_area_pct is None
        assert res[1].delta_area_pct == pytest.approx(-5.0)
        assert res[1].significant is True
        assert res[1].cda_m2 is None and res[1].delta_power_w is None

    def test_with_cd_power_and_time(self) -> None:
        from bikefit.aero.compare import compare_positions

        res = compare_positions([("a", _stats(0.40)), ("b", _stats(0.38))], 0.7, 36, 180)
        assert res[0].cda_m2 == pytest.approx(0.28)
        assert res[1].delta_power_w == pytest.approx(-0.5 * 1.225 * 0.7 * 0.02 * 1000, abs=0.1)
        assert res[1].time_saved_s is not None and res[1].time_saved_s > 0

    def test_difference_within_noise_not_significant(self) -> None:
        from bikefit.aero.compare import compare_positions

        res = compare_positions(
            [("a", _stats(0.400, std=0.02, n=6)), ("b", _stats(0.395, std=0.02, n=6))],
            None,
            36,
            180,
        )
        assert res[1].significant is False

    def test_format_duration(self) -> None:
        from bikefit.aero.compare import format_duration

        assert format_duration(130.4) == "+2 min 10 s"
        assert format_duration(-45) == "-45 s"


# ── Commande CLI `aero` ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def recordings(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("aero_cli")
    a = root / "pads_hauts"
    b = root / "pads_bas"
    _pedaling_recording(a, n=120)

    def lower():  # type: ignore[no-untyped-def]
        for i in range(120):  # position plus compacte : bloc moins haut de 100 mm
            depth = _draw_rect(_empty_scene(), -250, 250, -200, 800, 2500)
            _draw_rect(depth, 250, 300 + 40 * math.sin(2 * math.pi * i / 20), 200, 800, 2500)
            yield depth.astype(np.uint16)

    save_depth_recording(b, lower(), fps=30.0, intrinsics=INTR)
    return a, b


class TestAeroCli:
    def _invoke(self, *args: str):  # type: ignore[no-untyped-def]
        from typer.testing import CliRunner

        from bikefit.cli import app

        return CliRunner().invoke(app, ["aero", *args])

    def test_compare_two_positions_json(
        self, recordings: tuple[Path, Path], tmp_path: Path
    ) -> None:
        import json

        out = tmp_path / "aero.json"
        result = self._invoke(
            str(recordings[0]), str(recordings[1]), "--rider", "absent", "--output", str(out)
        )
        assert result.exit_code == 0, result.output
        assert "Cd non calibré" in result.output

        def _reject(token: str) -> None:
            raise AssertionError(token)

        data = json.loads(out.read_text(encoding="utf-8"), parse_constant=_reject)
        pos = data["positions"]
        assert pos[1]["delta_area_pct"] < -5  # 0,05 m² de moins sur ~0,58 m²
        assert pos[1]["significant"] is True
        assert data["cd"] is None

    def test_cda_terrain_calibrates_and_writes(
        self,
        recordings: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil

        config = tmp_path / "config"
        (config / "rider").mkdir(parents=True)
        shutil.copy(
            Path(__file__).parent.parent / "config/examples/mensurations.example.yaml",
            config / "rider" / "moi.yaml",
        )
        monkeypatch.setattr("bikefit._paths.CONFIG_DIR", config)
        monkeypatch.setattr("bikefit.equipment.rider.CONFIG_DIR", config)

        result = self._invoke(
            str(recordings[0]),
            str(recordings[1]),
            "--rider",
            "moi",
            "--cda-terrain",
            "0.40",
            "--write",
        )
        assert result.exit_code == 0, result.output
        assert "Calibration : Cd" in result.output
        assert " W à vitesse égale" in result.output

        from bikefit.equipment.rider import load_rider_profile

        rider = load_rider_profile("moi")
        assert rider is not None and rider.aero.cd is not None
        assert rider.aero.cda_terrain_m2 == pytest.approx(0.40)

        # Un 2ᵉ passage relit le Cd depuis la fiche, sans --cda-terrain
        again = self._invoke(str(recordings[1]), "--rider", "moi")
        assert "fiche moi" in again.output and "CdA ~" in again.output

    def test_debug_video_written(self, recordings: tuple[Path, Path], tmp_path: Path) -> None:
        result = self._invoke(
            str(recordings[0]), "--rider", "absent", "--debug-dir", str(tmp_path / "dbg")
        )
        assert result.exit_code == 0, result.output
        video = tmp_path / "dbg" / "pads_hauts_segmentation.mp4"
        assert video.exists() and video.stat().st_size > 0

    def test_invalid_recording_exits_1(self, tmp_path: Path) -> None:
        result = self._invoke(str(tmp_path / "nope"))
        assert result.exit_code == 1


def test_rider_profile_reads_aero_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "rider").mkdir()
    (tmp_path / "rider" / "r.yaml").write_text(
        "mensurations: {taille: 1790}\naero: {cd: 0.68, source: test}\n", encoding="utf-8"
    )
    monkeypatch.setattr("bikefit.equipment.rider.CONFIG_DIR", tmp_path)
    from bikefit.equipment.rider import load_rider_profile

    rider = load_rider_profile("r")
    assert rider is not None and rider.aero.cd == pytest.approx(0.68)


# ── Script de capture (fausse source : pas de Kinect en CI) ───────────────────


def test_capture_script_with_fake_kinect(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kinect_capture", Path(__file__).parent.parent / "scripts" / "kinect_capture.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ticks = iter(np.arange(0, 10, 0.04))  # horloge simulée : 25 fps réels
    scene = _rider().astype(np.uint16)
    rec = mod.capture(
        tmp_path / "cap",
        read_depth=lambda: scene,
        read_rgb=lambda: np.zeros((H, W, 3), np.uint8),
        duration_s=2.0,
        clock=lambda: float(next(ticks)),
    )
    assert rec.frame_count == 49  # t = 0,04 … 1,96 s
    assert rec.fps == pytest.approx(25.0, rel=0.01)  # fps réel, pas le nominal 30
    assert rec.rgb_path is not None
    np.testing.assert_array_equal(rec.read(0), scene)


def test_check_intrinsics_end_to_end(tmp_path: Path) -> None:
    """rgb.mp4 avec marqueur + profondeur cohérente -> fx retrouvé ; fx faux -> signalé."""
    import cv2

    from bikefit.aero import DepthIntrinsics, check_intrinsics_with_marker

    side_px = 80
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side_px
    )
    frame = np.full((H, W), 255, dtype=np.uint8)
    frame[200 : 200 + side_px, 280 : 280 + side_px] = marker
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    z = round(INTR.fx * 100 / side_px)  # 656 mm

    def write(path: Path, fx: float) -> DepthRecording:
        rec = save_depth_recording(
            path,
            [np.full((H, W), z, dtype=np.uint16)] * 20,
            fps=30.0,
            intrinsics=DepthIntrinsics(fx=fx, fy=fx, cx=INTR.cx, cy=INTR.cy),
        )
        writer = cv2.VideoWriter(str(path / "rgb.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 30, (W, H))
        for _ in range(20):
            writer.write(bgr)
        writer.release()
        return rec

    good = check_intrinsics_with_marker(write(tmp_path / "ok", INTR.fx), 100)
    assert good is not None and good.is_ok and good.n_frames == 2

    bad = check_intrinsics_with_marker(write(tmp_path / "bad", 580.0), 100)
    assert bad is not None and not bad.is_ok
    assert bad.area_error < -0.15  # fx surestimé de ~10 % -> surface sous-estimée de ~20 %

    assert check_intrinsics_with_marker(DepthRecording(tmp_path / "ok"), 100) is not None
