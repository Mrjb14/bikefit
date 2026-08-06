"""Tests de update_null_fields — réécriture ciblée des YAML de configuration.

Ce module écrit dans les fichiers de config personnels de l'utilisateur : la
garantie « ne jamais écraser une valeur déjà renseignée » est critique
(cf. CLAUDE.md, module measure/).
"""

from pathlib import Path

import pytest

from bikefit.measure.yaml_io import update_null_fields

_SAMPLE = """\
# Commentaire d'en-tête
mensurations:
  taille:              1790    # mm
  entrejambe:          null    # mm (sol -> pubis)
  longueur_femur:      null    # mm
  largeur_epaules:     null
  pointure:            43      # EU
"""


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    path = tmp_path / "mensurations.yaml"
    path.write_text(_SAMPLE, encoding="utf-8")
    return path


def test_fills_null_field(yaml_file: Path) -> None:
    updated = update_null_fields(yaml_file, {"entrejambe": 786})
    assert updated == ["entrejambe"]
    assert "entrejambe:          786" in yaml_file.read_text(encoding="utf-8")


def test_never_overwrites_existing_value(yaml_file: Path) -> None:
    updated = update_null_fields(yaml_file, {"taille": 9999, "pointure": 45})
    assert updated == []
    text = yaml_file.read_text(encoding="utf-8")
    assert "taille:              1790" in text
    assert "pointure:            43" in text


def test_preserves_inline_comments(yaml_file: Path) -> None:
    update_null_fields(yaml_file, {"entrejambe": 786})
    assert "# mm (sol -> pubis)" in yaml_file.read_text(encoding="utf-8")


def test_preserves_header_comment(yaml_file: Path) -> None:
    update_null_fields(yaml_file, {"entrejambe": 786})
    assert "# Commentaire d'en-tête" in yaml_file.read_text(encoding="utf-8")


def test_handles_field_without_comment(yaml_file: Path) -> None:
    updated = update_null_fields(yaml_file, {"largeur_epaules": 400})
    assert updated == ["largeur_epaules"]
    assert "largeur_epaules:     400" in yaml_file.read_text(encoding="utf-8")


def test_multiple_fields_at_once(yaml_file: Path) -> None:
    updated = update_null_fields(yaml_file, {"entrejambe": 786, "longueur_femur": 392, "taille": 1})
    assert set(updated) == {"entrejambe", "longueur_femur"}  # taille déjà remplie
    text = yaml_file.read_text(encoding="utf-8")
    assert "entrejambe:          786" in text
    assert "longueur_femur:      392" in text


def test_unknown_key_is_ignored(yaml_file: Path) -> None:
    assert update_null_fields(yaml_file, {"cle_inexistante": 123}) == []


def test_file_untouched_when_nothing_updated(yaml_file: Path) -> None:
    before = yaml_file.read_text(encoding="utf-8")
    update_null_fields(yaml_file, {"taille": 1})
    assert yaml_file.read_text(encoding="utf-8") == before


def test_float_value(yaml_file: Path) -> None:
    update_null_fields(yaml_file, {"entrejambe": 786.5})
    assert "entrejambe:          786.5" in yaml_file.read_text(encoding="utf-8")


def test_result_is_still_valid_yaml(yaml_file: Path) -> None:
    import yaml

    update_null_fields(yaml_file, {"entrejambe": 786, "longueur_femur": 392})
    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    assert data["mensurations"]["entrejambe"] == 786
    assert data["mensurations"]["longueur_femur"] == 392
    assert data["mensurations"]["taille"] == 1790
    assert data["mensurations"]["largeur_epaules"] is None
