"""Regression tests for M-17..M-23: runner flags, key handling, cleanup,
checksum verification, archive traversal and version parsing."""

import hashlib
import os
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Self

import pytest

from app.config import Settings
from scripts import download_llama, run_all, run_llama, smoke_test


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._sent = False

    def read(self, size: int = -1) -> bytes:
        del size
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# M-17 / M-18: canonical flags and no guessable key fallback
# ---------------------------------------------------------------------------


def test_run_llama_uses_canonical_flags(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    settings = Settings(data_dir=tmp_path / "data", llama_api_key="real-key")
    monkeypatch.setattr(run_llama, "get_settings", lambda: settings)
    monkeypatch.setattr(run_llama.sys, "argv", ["run_llama.py"])
    monkeypatch.setattr(
        run_llama, "find_llama_bin", lambda vendor: tmp_path / "llama-server"
    )
    monkeypatch.setattr(
        run_llama, "find_model_file", lambda models_dir, explicit: tmp_path / "m.gguf"
    )
    monkeypatch.setattr(
        run_llama.os, "execvpe", lambda program, argv, env: captured.update(argv=argv)
    )
    monkeypatch.setattr(run_llama.platform, "system", lambda: "Linux")

    run_llama.main()

    argv = captured["argv"]
    assert "--no-webui" in argv
    assert "--no-ui" not in argv
    assert "--reasoning" not in argv
    assert "--jinja" in argv
    assert "--no-context-shift" in argv
    assert argv[argv.index("--port") + 1] == "8080"
    assert argv[argv.index("-c") + 1] == "16384"


def test_run_llama_aborts_without_api_key(monkeypatch, tmp_path):
    """M-18: no silent `"local"` fallback."""
    settings = Settings(data_dir=tmp_path / "data", llama_api_key="")
    monkeypatch.setattr(run_llama, "get_settings", lambda: settings)
    monkeypatch.setattr(run_llama.sys, "argv", ["run_llama.py"])
    monkeypatch.setattr(
        run_llama, "find_llama_bin", lambda vendor: tmp_path / "llama-server"
    )
    monkeypatch.setattr(
        run_llama, "find_model_file", lambda models_dir, explicit: tmp_path / "m.gguf"
    )

    with pytest.raises(SystemExit) as excinfo:
        run_llama.main()

    assert "API key" in str(excinfo.value)


def test_build_llama_argv_rejects_empty_key():
    from app.llm.llama_args import build_llama_argv

    with pytest.raises(ValueError, match="non-empty API key"):
        build_llama_argv(
            llama_bin=Path("/opt/llama-server"),
            model_path=Path("/models/m.gguf"),
            port=8080,
            api_key="",
            settings=Settings(),
        )


# ---------------------------------------------------------------------------
# M-19: smoke test reuses the runner's key
# ---------------------------------------------------------------------------


def test_resolve_api_key_precedence(monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path / "data", llama_api_key="settings-key")
    monkeypatch.delenv("KNIGHT_LLAMA_API_KEY", raising=False)

    settings.llama_api_key_file.parent.mkdir(parents=True, exist_ok=True)
    settings.llama_api_key_file.write_text("file-key", encoding="utf-8")

    assert smoke_test.resolve_api_key(settings, "explicit-key") == "explicit-key"

    monkeypatch.setenv("KNIGHT_LLAMA_API_KEY", "env-key")
    assert smoke_test.resolve_api_key(settings) == "env-key"

    monkeypatch.delenv("KNIGHT_LLAMA_API_KEY")
    assert smoke_test.resolve_api_key(settings) == "file-key"

    settings.llama_api_key_file.unlink()
    assert smoke_test.resolve_api_key(settings) == "settings-key"


def test_run_all_writes_runner_key_file(tmp_path):
    """M-19: the runner persists its generated key for the smoke test."""
    key_file = tmp_path / "data" / "llama_api_key"
    run_all.write_api_key_file(key_file, "generated-secret")

    assert key_file.read_text(encoding="utf-8") == "generated-secret"
    if os.name == "posix":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# M-20: cleanup must not raise TimeoutExpired
# ---------------------------------------------------------------------------


class HungProcess:
    def __init__(self, terminate_error: bool = False) -> None:
        self.killed = False
        self._terminate_error = terminate_error
        self._waits = 0

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        if self._terminate_error:
            raise OSError("already gone")

    def wait(self, timeout: float | None = None) -> int:
        self._waits += 1
        if self._waits == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


def test_stop_process_escalates_after_timeout():
    proc = HungProcess()

    smoke_test.stop_process(proc, "llama-server", timeout=0.01)

    assert proc.killed is True


def test_stop_process_swallows_oserror():
    smoke_test.stop_process(HungProcess(terminate_error=True), "api", timeout=0.01)


# ---------------------------------------------------------------------------
# M-21: checksum verification
# ---------------------------------------------------------------------------


def test_download_file_rejects_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"not-the-real-binary"
    monkeypatch.setattr(
        download_llama.urllib.request,
        "urlopen",
        lambda url, timeout: FakeResponse(payload),
    )
    dest = tmp_path / "llama.zip"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_llama.download_file("https://example.invalid/llama.zip", dest, "0" * 64)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_file_accepts_matching_checksum(tmp_path, monkeypatch):
    payload = b"verified-binary"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        download_llama.urllib.request,
        "urlopen",
        lambda url, timeout: FakeResponse(payload),
    )
    dest = tmp_path / "llama.zip"

    download_llama.download_file("https://example.invalid/llama.zip", dest, expected)

    assert dest.read_bytes() == payload
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_file_refuses_unpublished_digest(tmp_path):
    with pytest.raises(ValueError, match="No published SHA-256"):
        download_llama.download_file(
            "https://example.invalid/llama.zip", tmp_path / "llama.zip", None
        )


# ---------------------------------------------------------------------------
# M-22: archive traversal / symlink rejection
# ---------------------------------------------------------------------------


def test_zip_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../evil.txt", "owned")

    with pytest.raises(ValueError, match="escapes destination"):
        download_llama.extract_archive(archive, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()


def test_zip_archive_symlink_is_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, "")

    with pytest.raises(ValueError, match="symlink"):
        download_llama.extract_archive(archive, tmp_path / "out")


def test_tar_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("../../evil.txt")
        info.size = 0
        import io

        tf.addfile(info, io.BytesIO(b""))

    with pytest.raises(ValueError, match="escapes destination"):
        download_llama.extract_archive(archive, tmp_path / "out")


def test_safe_archive_extracts_normally(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub/file.txt", "hello")

    out = tmp_path / "out"
    download_llama.extract_archive(archive, out)

    assert (out / "sub" / "file.txt").read_text(encoding="utf-8") == "hello"


# ---------------------------------------------------------------------------
# M-23: CUDA versions compare as versions, not floats
# ---------------------------------------------------------------------------


def test_version_tuple_compares_numerically():
    assert download_llama.version_tuple("12.10") > download_llama.version_tuple("12.9")
    assert download_llama.version_tuple("13.4") == (13, 4)


def test_cuda_13_10_is_treated_as_newer_than_13_4():
    """`float("13.10") == 13.1` used to pick the wrong runtime."""
    assert download_llama.get_default_cuda(is_win=False, detected="13.10") == "13.4"
    assert download_llama.get_default_cuda(is_win=True, detected="13.10") == "13.4"
