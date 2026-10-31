"""SessionStore — persistance SQLite des sessions d'analyse."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from bikefit.storage.models import Session

_DB_NAME = "sessions.db"
_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    NOT NULL,
    video_path   TEXT    NOT NULL,
    bike_name    TEXT    NOT NULL,
    rider_name   TEXT,
    profile      TEXT    NOT NULL,
    setup_json   TEXT    NOT NULL,
    metrics_json TEXT    NOT NULL,
    report_json  TEXT    NOT NULL
);
"""

_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN rider_name TEXT",
]


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_DDL)
            # Apply forward migrations, ignoring only "column already exists"
            for sql in _MIGRATIONS:
                try:
                    conn.execute(sql)
                    conn.commit()
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc):
                        raise

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connexion transactionnelle ET fermée en sortie.

        `with sqlite3.connect(...)` seul valide/annule la transaction mais ne ferme
        pas la connexion : sous Windows (plateforme de dev principale), le fichier
        .db reste alors verrouillé tant que le garbage collector ne passe pas.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save(self, session: Session) -> Session:
        """Persiste une session et retourne la session avec son id assigné."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sessions
                    (date, video_path, bike_name, rider_name, profile,
                     setup_json, metrics_json, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.date.isoformat(),
                    session.video_path,
                    session.bike_name,
                    session.rider_name,
                    session.profile,
                    session.setup.model_dump_json(),
                    session.metrics.model_dump_json(),
                    session.report.model_dump_json(),
                ),
            )
            conn.commit()
        return session.model_copy(update={"id": cur.lastrowid})

    def get(self, session_id: int) -> Session | None:
        """Retourne une session par id, ou None si introuvable."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _row_to_session(row) if row else None

    def list_all(self) -> list[Session]:
        """Retourne toutes les sessions, les plus récentes en premier."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY date DESC, id DESC").fetchall()
        return [_row_to_session(r) for r in rows]

    def latest(self) -> Session | None:
        """Retourne la session la plus récente, ou None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions ORDER BY date DESC, id DESC LIMIT 1"
            ).fetchone()
        return _row_to_session(row) if row else None

    def delete(self, session_id: int) -> bool:
        """Supprime une session. Retourne True si elle existait."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        return cur.rowcount > 0


def _row_to_session(row: sqlite3.Row) -> Session:
    from bikefit.equipment.models import BikeSetup
    from bikefit.kinematics.metrics import FitMetrics
    from bikefit.recommend.models import FitReport

    keys = row.keys()
    return Session(
        id=row["id"],
        date=datetime.fromisoformat(row["date"]),
        video_path=row["video_path"],
        bike_name=row["bike_name"],
        rider_name=row["rider_name"] if "rider_name" in keys else None,
        profile=row["profile"],
        setup=BikeSetup(**json.loads(row["setup_json"])),
        metrics=FitMetrics(**json.loads(row["metrics_json"])),
        report=FitReport(**json.loads(row["report_json"])),
    )
