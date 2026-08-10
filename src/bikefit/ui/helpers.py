"""Fonctions pures de l'UI Streamlit — sans dépendance à streamlit, testables directement."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol


class UploadedFileLike(Protocol):
    """Ce que `st.file_uploader` renvoie — juste assez pour l'écrire sur disque."""

    name: str

    def getbuffer(self) -> memoryview: ...


def list_configs(config_dir: Path, subdir: str) -> list[str]:
    """Liste les fiches YAML disponibles (bikes/rider) par nom, sans l'extension."""
    d = config_dir / subdir
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def parse_optional_float(raw: str) -> float | None:
    """Convertit un champ texte en float, ou None si vide. Lève ValueError si invalide."""
    raw = raw.strip()
    return float(raw) if raw else None


def save_upload(uploaded_file: UploadedFileLike, dest_dir: Path) -> Path:
    """Écrit un fichier uploadé sur disque, horodaté pour éviter les collisions.

    Persiste dans dest_dir (convention data/sources/) plutôt qu'un temp dir : les
    vidéos réutilisées entre deux analyses profitent ainsi du cache de landmarks
    (clé = chemin résolu + mtime + size).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"{stamp}_{uploaded_file.name}"
    dest.write_bytes(bytes(uploaded_file.getbuffer()))
    return dest
