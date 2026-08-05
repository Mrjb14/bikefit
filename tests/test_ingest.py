"""Tests sur la détection fichier local / flux réseau dans VideoReader."""

import pytest

from bikefit.ingest.reader import VideoReader, _is_stream_url


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
