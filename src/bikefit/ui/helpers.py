"""Fonctions pures de l'UI Streamlit — sans dépendance à streamlit, testables directement."""

from __future__ import annotations

import hashlib
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
    """Écrit un fichier uploadé sur disque, nommé par hash de contenu.

    Persiste dans dest_dir (convention data/sources/) plutôt qu'un temp dir, avec
    un nom déterministe (hash du contenu, pas un timestamp) : réuploader la même
    vidéo retombe sur le même chemin résolu SANS le réécrire (mtime intact) — le
    cache de landmarks (clé = chemin résolu + mtime + size) est donc bien réutilisé
    d'une soumission du formulaire à l'autre, comme au CLI.

    `uploaded_file.name` ne garde que le nom de base (`Path(...).name`) : un champ
    contrôlé côté client ne doit jamais dicter directement où on écrit sur disque.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    content = bytes(uploaded_file.getbuffer())
    digest = hashlib.sha256(content).hexdigest()[:16]
    safe_name = Path(uploaded_file.name).name
    dest = dest_dir / f"{digest}_{safe_name}"
    if not dest.exists():
        dest.write_bytes(content)
    return dest
