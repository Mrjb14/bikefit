"""Tests du module storage — SessionStore SQLite (données synthétiques)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from bikefit.equipment.models import BikeModel, BikeProfile, BikeSetup
from bikefit.kinematics.metrics import AngleStats, FitMetrics
from bikefit.recommend import recommend
from bikefit.storage import Session, SessionStore

# ── Helpers ───────────────────────────────────────────────────────────────────


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


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSessionStore:
    def test_save_assigns_id(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        saved = store.save(_session())
        assert saved.id == 1

    def test_save_multiple_increments_id(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        s1 = store.save(_session())
        s2 = store.save(_session(video="autre.mp4"))
        assert s1.id == 1
        assert s2.id == 2

    def test_get_returns_saved_session(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        saved = store.save(_session(genou=148.0))
        retrieved = store.get(saved.id)  # type: ignore[arg-type]
        assert retrieved is not None
        assert retrieved.metrics.genou_extension_pmb.mean == pytest.approx(148.0)

    def test_get_unknown_returns_none(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        assert store.get(999) is None

    def test_list_all_order(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        store.save(_session())
        store.save(_session())
        store.save(_session())
        sessions = store.list_all()
        assert len(sessions) == 3
        # les plus récentes en premier (même date ici — au moins même nombre)
        assert [s.id for s in sessions] == [3, 2, 1]

    def test_latest_returns_last(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        store.save(_session())
        last = store.save(_session(video="last.mp4"))
        assert store.latest().id == last.id  # type: ignore[union-attr]

    def test_latest_empty_db(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        assert store.latest() is None

    def test_delete_existing(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        saved = store.save(_session())
        assert store.delete(saved.id)  # type: ignore[arg-type]
        assert store.get(saved.id) is None  # type: ignore[arg-type]

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        assert not store.delete(42)

    def test_roundtrip_preserves_fields(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "sessions.db")
        orig = _session(genou=152.5, video="profil_right.mp4")
        saved = store.save(orig)
        back = store.get(saved.id)  # type: ignore[arg-type]
        assert back is not None
        assert back.video_path == "profil_right.mp4"
        assert back.bike_name == "giant_trinity_2019"
        assert back.profile == "triathlon_XL"
        assert back.date == orig.date
        assert back.metrics.genou_extension_pmb.mean == pytest.approx(152.5)

    def test_db_persists_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        store1 = SessionStore(db)
        store1.save(_session())
        store2 = SessionStore(db)
        assert len(store2.list_all()) == 1


def test_session_with_unmeasured_metric_roundtrips(tmp_path: Path) -> None:
    """Régression : une métrique NaN était sérialisée en null puis refusée à la relecture,
    rendant la session (et `history`/`compare` sur toute la base) illisible."""
    import math

    from bikefit.kinematics.metrics import AngleStats

    nan = float("nan")
    session = _session()
    session = session.model_copy(
        update={
            "metrics": session.metrics.model_copy(
                update={"coude": AngleStats(mean=nan, std=nan, min=nan, max=nan, n_cycles=0)}
            )
        }
    )
    store = SessionStore(tmp_path / "s.db")
    saved = store.save(session)

    loaded = store.get(saved.id or 0)
    assert loaded is not None
    assert math.isnan(loaded.metrics.coude.mean)
    assert not loaded.metrics.coude.is_measured
    assert len(store.list_all()) == 1
