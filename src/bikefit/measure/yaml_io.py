"""Mise à jour des champs null dans les fichiers YAML (commentaires préservés)."""

from __future__ import annotations

import re
from pathlib import Path


def update_null_fields(path: Path, updates: dict[str, int | float]) -> list[str]:
    """Remplace les valeurs `null` par les estimations, commentaires conservés.

    Seuls les champs valant exactement `null` sont remplacés.
    Retourne la liste des clés effectivement mises à jour.
    """
    text = path.read_text(encoding="utf-8")
    updated: list[str] = []
    for key, value in updates.items():
        new_text, n = re.subn(
            rf"^(\s*{re.escape(key)}\s*:\s*)null(\s*#.*)?$",
            lambda m, v=value: f"{m.group(1)}{v}{m.group(2) or ''}",
            text,
            flags=re.MULTILINE,
        )
        if n > 0:
            text = new_text
            updated.append(key)
    if updated:
        path.write_text(text, encoding="utf-8")
    return updated
