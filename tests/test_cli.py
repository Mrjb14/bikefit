"""Tests du CLI (Typer) — commandes qui ne nécessitent pas le pipeline vidéo/pose.

`analyze` dépend de MediaPipe + une vidéo réelle et n'est couvert ici au-delà de sa
validation d'arguments précoce qu'en remplaçant MediaPipePoseEstimator par un faux
estimateur (TestAnalyzeDebugVideo) — `measure-body`/`measure-setup` restent hors
scope. `history` et `compare` sont testés de bout en bout (vrai SQLite, sans mock).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from bikefit.cli import app
from bikefit.equipment.models import BikeModel, BikeProfile, BikeSetup
from bikefit.kinematics.metrics import AngleStats, FitMetrics
from bikefit.pose.base import FrameLandmarks, Landmark
from bikefit.recommend import recommend
from bikefit.storage import Session, SessionStore

runner = CliRunner()


class _FakeCliPoseEstimator:
    """Remplace MediaPipePoseEstimator dans les tests CLI — même squelette rigide
    translaté verticalement en sinusoïde (fait osciller ankle_y -> cycles détectables),
    sans dépendance MediaPipe."""

    def __enter__(self) -> _FakeCliPoseEstimator:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int, frame_idx: int) -> FrameLandmarks:
        shift = 0.05 * math.sin(2 * math.pi * frame_idx / 5)
        landmarks = [
            Landmark(x=0.1 + 0.02 * i, y=0.3 + 0.017 * i + shift, z=0.0, visibility=1.0)
            for i in range(33)
        ]
        return FrameLandmarks(
            timestamp_ms=timestamp_ms, frame_idx=frame_idx, detected=True, landmarks=landmarks
        )


def _write_synthetic_video(
    path: Path, n_frames: int = 20, width: int = 64, height: int = 48
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    for _ in range(n_frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


# ── Helpers (mêmes que test_storage.py) ────────────────────────────────────────


def _stats(mean: float) -> AngleStats:
    return AngleStats(mean=mean, std=1.0, min=mean - 5, max=mean + 5, n_cycles=10)


def _metrics(genou: float = 150.0) -> FitMetrics:
    return FitMetrics(
        genou_extension_pmb=_stats(genou),
        genou_flexion_pmh=_stats(71.0),
        hanche_fermee_pmh=_stats(55.0),
        tronc=_stats(16.0),
        coude=_stats(96.0),
        cheville_pmb=_stats(138.0),
        n_cycles_total=10,
        n_cycles_used=10,
        detection_rate=1.0,
    )


def _bike() -> BikeProfile:
    return BikeProfile(modele=BikeModel(marque="Giant", nom="Trinity", annee=2019, taille="M"))


def _session(genou: float = 150.0, video: str = "test.mp4") -> Session:
    metrics = _metrics(genou)
    report = recommend(metrics, _bike(), profile="triathlon_XL")
    return Session(
        date=datetime(2026, 8, 5, 10, 0, 0),
        video_path=video,
        bike_name="giant_trinity_2019",
        profile="triathlon_XL",
        setup=BikeSetup(),
        metrics=metrics,
        report=report,
    )


# ── analyze : validation précoce (avant pipeline pose) ─────────────────────────


class TestAnalyzeValidation:
    def test_missing_video_exits_1(self, tmp_path: Path) -> None:
        # CliRunner appelle l'app Typer directement (pas __main__.py) — le sous-commande
        # "analyze" doit être explicite, contrairement à `python -m bikefit video.mp4`.
        result = runner.invoke(app, ["analyze", str(tmp_path / "introuvable.mp4")])
        assert result.exit_code == 1
        assert "introuvable" in result.output.lower()

    def test_invalid_side_exits_1(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")  # existe, contenu bidon — --side rejeté avant lecture
        result = runner.invoke(app, ["analyze", str(video), "--side", "gauche"])
        assert result.exit_code == 1
        assert "--side" in result.output

    def test_front_same_file_as_video_exits_1(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        result = runner.invoke(app, ["analyze", str(video), "--front", str(video)])
        assert result.exit_code == 1
        assert "même source" in result.output.lower()

    def test_front_same_webcam_index_exits_1(self) -> None:
        # Ni video ni front n'ont besoin d'une vraie webcam : la garde anti-double-usage
        # s'exécute avant toute ouverture de caméra.
        result = runner.invoke(app, ["analyze", "webcam:0", "--front", "webcam:0"])
        assert result.exit_code == 1
        assert "même source" in result.output.lower()

    def test_front_different_file_does_not_trigger_same_source_guard(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        front = tmp_path / "front.mp4"
        video.write_bytes(b"\x00")
        front.write_bytes(b"\x00")
        result = runner.invoke(app, ["analyze", str(video), "--front", str(front)])
        # Fichiers bidon -> échoue plus loin (vidéo illisible), mais pas sur la garde.
        assert "même source" not in result.output.lower()


# ── analyze --debug-video : pipeline complet, MediaPipe remplacé par un faux ───


class TestAnalyzeDebugVideo:
    def test_debug_video_generated_for_file_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video_path = tmp_path / "video.mp4"
        _write_synthetic_video(video_path)

        monkeypatch.setattr("bikefit.pose.MediaPipePoseEstimator", _FakeCliPoseEstimator)
        monkeypatch.setattr("bikefit.pose.cache.CACHE_DIR", tmp_path / "cache")

        debug_out = tmp_path / "debug.mp4"
        report_out = tmp_path / "report.html"
        result = runner.invoke(
            app,
            [
                "analyze",
                str(video_path),
                "--side",
                "right",
                "--debug-video",
                str(debug_out),
                "--report",
                str(report_out),
                "--bike",
                "velo_de_test_inexistant",
            ],
        )

        assert result.exit_code == 0, result.output
        assert debug_out.exists()
        assert debug_out.stat().st_size > 0
        assert report_out.exists()

        cap = cv2.VideoCapture(str(debug_out))
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 20  # une frame de sortie par frame source


# ── history ──────────────────────────────────────────────────────────────────


class TestHistory:
    def test_no_db_file(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        result = runner.invoke(app, ["history", "--db", str(db)])
        assert result.exit_code == 0
        assert "Aucune session" in result.output

    def test_empty_db(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        SessionStore(db)  # crée le fichier + le schéma, sans session
        result = runner.invoke(app, ["history", "--db", str(db)])
        assert "Aucune session" in result.output

    def test_lists_saved_sessions(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store = SessionStore(db)
        store.save(_session(genou=150.0, video="a.mp4"))
        store.save(_session(genou=148.0, video="b.mp4"))
        result = runner.invoke(app, ["history", "--db", str(db)])
        assert result.exit_code == 0
        assert "giant_trinity_2019" in result.output
        assert result.output.count("triathlon_XL") == 2

    def test_limit_option(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store = SessionStore(db)
        for i in range(5):
            store.save(_session(video=f"v{i}.mp4"))
        result = runner.invoke(app, ["history", "--db", str(db), "--limit", "2"])
        lines = result.output.splitlines()
        session_lines = [line for line in lines if "giant_trinity_2019" in line]
        assert len(session_lines) == 2


# ── compare ──────────────────────────────────────────────────────────────────


class TestCompare:
    def test_missing_db_exits_1(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        result = runner.invoke(app, ["compare", "1", "2", "--db", str(db)])
        assert result.exit_code == 1
        assert "Aucune session" in result.output

    def test_unknown_session_id_exits_1(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store = SessionStore(db)
        store.save(_session())
        result = runner.invoke(app, ["compare", "1", "999", "--db", str(db)])
        assert result.exit_code == 1
        assert "#999" in result.output
        assert "introuvable" in result.output.lower()

    def test_compare_two_sessions_shows_delta(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store = SessionStore(db)
        store.save(_session(genou=150.0, video="avant.mp4"))
        store.save(_session(genou=145.0, video="apres.mp4"))
        result = runner.invoke(app, ["compare", "1", "2", "--db", str(db)])
        assert result.exit_code == 0
        assert "#1" in result.output
        assert "#2" in result.output
        assert "-5.0" in result.output  # delta genou : 145 - 150

    def test_compare_no_recommendations_message(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store = SessionStore(db)
        # genou=145 -> dans la cible triathlon_XL (140-150), pas de recommandation
        store.save(_session(genou=145.0))
        store.save(_session(genou=145.0, video="autre.mp4"))
        result = runner.invoke(app, ["compare", "1", "2", "--db", str(db)])
        assert "position optimale" in result.output.lower()
