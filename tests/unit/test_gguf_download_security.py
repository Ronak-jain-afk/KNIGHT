"""Regression tests for C-07..C-10: GGUF download SSRF, redirect, size, magic."""

import socket
from types import SimpleNamespace

import httpx
import pytest
import respx

from app.config import Settings
from app.llm.ggufbook import (
    DownloadTask,
    DownloadURLError,
    _open_validated_stream,
    _sync_download_worker,
    validate_download_url,
)

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", model_download_max_bytes=1024)
    s.ensure_dirs()
    monkeypatch.setattr(
        "app.security.net_guard.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))
        ],
    )
    monkeypatch.setattr(
        "app.llm.ggufbook.shutil.disk_usage",
        lambda path: SimpleNamespace(total=10**12, used=0, free=10**12),
    )
    return s


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/model.gguf",
        "https://user:pass@huggingface.co/model.gguf",
        "https://huggingface.co:8443/model.gguf",
        "https://127.0.0.1/model.gguf",
        "https://[::1]/model.gguf",
        "https://evil.example/model.gguf",
        "https://huggingface.co.evil.example/model.gguf",
        "ftp://huggingface.co/model.gguf",
        "https://huggingface.co:notaport/model.gguf",
    ],
)
def test_download_url_rejects_unsafe_origins(settings, url):
    """C-07: only https + allowlisted host + default port are accepted."""
    with pytest.raises(DownloadURLError):
        validate_download_url(url, settings)


def test_download_url_resolves_and_blocks_private_dns(settings, monkeypatch):
    """C-07: an allowlisted name that resolves internally is still rejected."""
    monkeypatch.setattr(
        "app.security.net_guard.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(DownloadURLError, match="blocked IP"):
        validate_download_url("https://huggingface.co/model.gguf", settings)


def test_download_url_accepts_allowlisted_public_host(settings):
    url = "https://huggingface.co/org/repo/resolve/main/model.gguf"
    assert validate_download_url(url, settings) == url


def test_redirect_to_loopback_is_rejected(settings):
    """C-08: redirects are validated hop-by-hop, never followed blindly."""
    start = "https://huggingface.co/start.gguf"
    with respx.mock:
        respx.get(start).respond(
            302, headers={"location": "http://127.0.0.1/secret.gguf"}
        )
        with (
            httpx.Client(follow_redirects=False) as client,
            pytest.raises(DownloadURLError, match="https"),
        ):
            _open_validated_stream(client, start, settings)


def test_download_aborts_when_content_length_exceeds_cap(settings):
    """C-09: a huge declared Content-Length is rejected before writing."""
    url = "https://huggingface.co/huge.gguf"
    with respx.mock:
        respx.get(url).respond(
            200,
            headers={"content-length": str(10 * 1024**3)},
            content=b"GGUF" + b"x" * 32,
        )
        task = DownloadTask("t-big", "org/repo", "huge.gguf", url, 0)
        _sync_download_worker(task, settings)

    assert task.status == "failed"
    assert task.error is not None and "limit" in task.error.lower()
    assert not (settings.models_dir / "huge.gguf").exists()
    assert not (settings.models_dir / "huge.gguf.part").exists()


def test_download_rejects_non_gguf_payload(settings):
    """C-10: the payload must start with the GGUF magic; no partial file remains."""
    url = "https://huggingface.co/fake.gguf"
    with respx.mock:
        respx.get(url).respond(200, content=b"NOT-A-GGUF-PAYLOAD" + b"x" * 64)
        task = DownloadTask("t-fake", "org/repo", "fake.gguf", url, 0)
        _sync_download_worker(task, settings)

    assert task.status == "failed"
    assert task.error is not None and "GGUF" in task.error
    assert not (settings.models_dir / "fake.gguf").exists()
    assert not (settings.models_dir / "fake.gguf.part").exists()


def test_download_writes_atomically_on_success(settings):
    """C-10: a valid GGUF is fsynced and atomically moved into place."""
    url = "https://huggingface.co/good.gguf"
    payload = b"GGUF" + b"\x00" * 128
    with respx.mock:
        respx.get(url).respond(200, content=payload)
        task = DownloadTask("t-good", "org/repo", "good.gguf", url, 0)
        _sync_download_worker(task, settings)

    target = settings.models_dir / "good.gguf"
    assert task.status == "completed"
    assert target.read_bytes() == payload
    assert not (settings.models_dir / "good.gguf.part").exists()
