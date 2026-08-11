from pathlib import Path

import pytest

from bikefit.ui.helpers import list_configs, parse_optional_float, save_upload


class _FakeUploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


def test_list_configs_lists_yaml_stems(tmp_path: Path) -> None:
    bikes_dir = tmp_path / "bikes"
    bikes_dir.mkdir()
    (bikes_dir / "giant_trinity_2019.yaml").write_text("modele: {}\n")
    (bikes_dir / "other.yaml").write_text("modele: {}\n")
    (bikes_dir / "notes.txt").write_text("ignored")

    assert list_configs(tmp_path, "bikes") == ["giant_trinity_2019", "other"]


def test_list_configs_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert list_configs(tmp_path, "does_not_exist") == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", None), ("   ", None), ("12", 12.0), ("12.5", 12.5), ("-3.2", -3.2)],
)
def test_parse_optional_float(raw: str, expected: float | None) -> None:
    assert parse_optional_float(raw) == expected


def test_parse_optional_float_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_optional_float("not-a-number")


def test_save_upload_writes_content_with_hash_prefix(tmp_path: Path) -> None:
    fake = _FakeUploadedFile("video.mp4", b"fake-video-bytes")

    dest = save_upload(fake, tmp_path)

    assert dest.exists()
    assert dest.read_bytes() == b"fake-video-bytes"
    assert dest.name.endswith("_video.mp4")
    assert dest.parent == tmp_path


def test_save_upload_creates_dest_dir_if_missing(tmp_path: Path) -> None:
    dest_dir = tmp_path / "nested" / "uploads"
    fake = _FakeUploadedFile("a.mov", b"x")

    dest = save_upload(fake, dest_dir)

    assert dest.exists()
    assert dest_dir.exists()


def test_save_upload_reuploading_same_content_reuses_path_and_mtime(tmp_path: Path) -> None:
    # Le cache de landmarks est keyé sur chemin résolu + mtime + size : réuploader
    # la même vidéo doit retomber sur le même fichier SANS le réécrire, sinon
    # chaque soumission du formulaire recalculerait toute l'inférence MediaPipe.
    fake = _FakeUploadedFile("video.mp4", b"same-bytes")

    first = save_upload(fake, tmp_path)
    mtime_after_first = first.stat().st_mtime_ns

    second = save_upload(_FakeUploadedFile("video.mp4", b"same-bytes"), tmp_path)

    assert second == first
    assert second.stat().st_mtime_ns == mtime_after_first


def test_save_upload_different_content_same_name_gets_distinct_paths(tmp_path: Path) -> None:
    first = save_upload(_FakeUploadedFile("video.mp4", b"content-a"), tmp_path)
    second = save_upload(_FakeUploadedFile("video.mp4", b"content-b"), tmp_path)

    assert first != second
    assert first.read_bytes() == b"content-a"
    assert second.read_bytes() == b"content-b"


def test_save_upload_strips_path_components_from_name(tmp_path: Path) -> None:
    fake = _FakeUploadedFile("../../etc/passwd", b"x")

    dest = save_upload(fake, tmp_path)

    assert dest.parent == tmp_path
    assert dest.name.endswith("_passwd")
    assert ".." not in dest.parts
