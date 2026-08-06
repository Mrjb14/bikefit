"""Tests sur la détection fichier local / flux réseau / webcam dans VideoReader."""

from pathlib import Path

import pytest

from bikefit.ingest.reader import (
    VideoReader,
    _is_stream_url,
    _is_webcam_source,
    _same_source,
    _webcam_index,
)


def test_is_stream_url_http():
    assert _is_stream_url("http://192.168.1.42/stream")


def test_is_stream_url_https():
    assert _is_stream_url("https://example.com/stream")


def test_is_stream_url_rtsp():
    assert _is_stream_url("rtsp://192.168.1.42/stream")


def test_is_stream_url_local_path():
    assert not _is_stream_url("data/video_test_right.mp4")
    assert not _is_stream_url(r"C:\videos\test.mp4")


def test_local_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        VideoReader("data/ce_fichier_n_existe_pas.mp4")


# ── Webcam ───────────────────────────────────────────────────────────────────


def test_is_webcam_source():
    assert _is_webcam_source("webcam:0")
    assert _is_webcam_source("webcam:1")
    assert not _is_webcam_source("data/video_test_right.mp4")
    assert not _is_webcam_source("http://192.168.1.42/stream")


def test_webcam_index_parses_digits():
    assert _webcam_index("webcam:0") == 0
    assert _webcam_index("webcam:12") == 12


def test_webcam_index_rejects_non_digit():
    with pytest.raises(ValueError):
        _webcam_index("webcam:abc")
    with pytest.raises(ValueError):
        _webcam_index("webcam:")


# ── _same_source (garde anti-double-usage vue profil / vue face) ───────────────


class TestSameSource:
    def test_same_file_path_identical_string(self):
        assert _same_source("video.mp4", "video.mp4")

    def test_same_file_different_notation(self, tmp_path: Path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        rel = f"{tmp_path}/./v.mp4"
        assert _same_source(str(f), rel)

    def test_different_files(self):
        assert not _same_source("a.mp4", "b.mp4")

    def test_same_webcam_index(self):
        assert _same_source("webcam:0", "webcam:0")

    def test_different_webcam_index(self):
        assert not _same_source("webcam:0", "webcam:1")

    def test_same_stream_url(self):
        url = "http://192.168.1.42/stream"
        assert _same_source(url, url)

    def test_different_stream_url(self):
        assert not _same_source("http://192.168.1.42/stream", "http://192.168.1.43/stream")

    def test_webcam_vs_file_never_same(self):
        assert not _same_source("webcam:0", "video.mp4")

    def test_webcam_vs_stream_never_same(self):
        assert not _same_source("webcam:0", "http://192.168.1.42/stream")

    def test_stream_vs_file_never_same(self):
        assert not _same_source("http://192.168.1.42/stream", "video.mp4")
