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


def test_save_upload_writes_content_with_timestamp_prefix(tmp_path: Path) -> None:
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
