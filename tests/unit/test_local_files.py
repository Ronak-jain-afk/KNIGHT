# SPDX-License-Identifier: Apache-2.0
import platform

import pytest

from app.config import get_settings
from app.tools.local_files import list_directory, read_local_file, resolve_system_path


def test_read_local_file_success(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    docs_dir = get_settings().documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    test_file = docs_dir / "sample_code.py"
    test_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    result = read_local_file.invoke({"path": str(test_file)})
    assert "--- FILE:" in result
    assert "def hello():" in result
    assert "sample_code.py" in result


def test_read_local_file_pagination(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    docs_dir = get_settings().documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    test_file = docs_dir / "large.txt"
    test_file.write_text("A" * 1000 + "B" * 1000, encoding="utf-8")

    result = read_local_file.invoke({"path": str(test_file), "offset_chars": 1000, "max_chars": 500})
    assert "--- FILE:" in result
    assert "Offset: 1000..1500" in result
    assert "B" * 500 in result
    assert "A" not in result


def test_read_local_file_not_found():
    result = read_local_file.invoke({"path": "non_existent_file_xyz_123.txt"})
    assert "ERROR: file_not_found" in result


def test_read_local_file_sensitive_name_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    docs_dir = get_settings().documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    env_file = docs_dir / ".env"
    env_file.write_text("SECRET=123", encoding="utf-8")

    result = read_local_file.invoke({"path": str(env_file)})
    assert "ERROR: forbidden" in result


def test_read_local_file_offset_out_of_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("KNIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    docs_dir = get_settings().documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    short_file = docs_dir / "short.txt"
    short_file.write_text("Short text", encoding="utf-8")

    result = read_local_file.invoke({"path": str(short_file), "offset_chars": 500})
    assert "ERROR: offset_out_of_bounds" in result


def test_list_directory_success(tmp_path):
    sub_dir = tmp_path / "my_project"
    sub_dir.mkdir(parents=True, exist_ok=True)

    file_a = sub_dir / "main.py"
    file_a.write_text("print('test')", encoding="utf-8")

    child_folder = sub_dir / "assets"
    child_folder.mkdir(parents=True, exist_ok=True)

    result = list_directory.invoke({"path": str(sub_dir)})
    assert "📁 DIRECTORY:" in result
    assert "📁 assets/" in result
    assert "📄 main.py" in result


def test_read_local_file_on_directory_lists_contents(tmp_path):
    sub_dir = tmp_path / "docs_folder"
    sub_dir.mkdir(parents=True, exist_ok=True)

    file_doc = sub_dir / "readme.txt"
    file_doc.write_text("info", encoding="utf-8")

    result = read_local_file.invoke({"path": str(sub_dir)})
    assert "NOTE: " in result
    assert "is a directory, not a file" in result
    assert "📄 readme.txt" in result


def test_resolve_system_path_cross_platform():
    # Test tilde expansion
    p_home = resolve_system_path("~/projects")
    assert not str(p_home).startswith("~")

    # Test Windows drive translation on Linux/WSL
    if platform.system().lower() != "windows":
        p_win = resolve_system_path(r"C:\dev\walk")
        assert str(p_win).startswith("/mnt/c/dev/walk") or str(p_win).startswith("/c/dev/walk")


def test_read_local_file_string_arguments(tmp_path):
    test_file = tmp_path / "hello.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    # Pass strings for offset_chars and max_chars as local LLMs sometimes do
    result = read_local_file.invoke({"path": str(test_file), "offset_chars": "0", "max_chars": "500"})
    assert "--- FILE:" in result
    assert "print('hello')" in result


def test_read_local_file_binary_rejected(tmp_path):
    bin_file = tmp_path / "compiled.bin"
    bin_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00")

    result = read_local_file.invoke({"path": str(bin_file)})
    assert "ERROR: binary_file" in result


def test_describe_tool_error_accepts_two_arguments():
    from app.agent.middleware import describe_tool_error

    class DummyTool:
        def __init__(self):
            self.name = "test_tool"

    class DummyRequest:
        def __init__(self):
            self.tool = DummyTool()
            self.tool_call = {"name": "test_tool", "id": "123"}

    err = ValueError("Something failed inside tool")
    msg = describe_tool_error(err, DummyRequest())
    assert "ERROR: test_tool_failure:" in msg
    assert "Something failed inside tool" in msg


def test_read_local_file_with_active_source_collector(tmp_path):
    from app.agent.sources import SourceCollector, current_sources

    test_file = tmp_path / "cited.py"
    test_file.write_text("x = 42\n", encoding="utf-8")

    collector = SourceCollector()
    token = current_sources.set(collector)
    try:
        result = read_local_file.invoke({"path": str(test_file)})
        assert "--- FILE:" in result
        items = collector.items()
        assert len(items) == 1
        assert "cited.py" in items[0]["url"]
        assert items[0]["snippet"] is not None
    finally:
        current_sources.reset(token)


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/id_rsa",
        ".ssh/id_ed25519",
        ".aws/credentials",
        ".bash_history",
        ".zsh_history",
        ".config/gh/hosts.yml",
    ],
)
def test_sensitive_home_paths_are_forbidden(tmp_path, monkeypatch, relative):
    """C-26: credential stores and shell history are denied by default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("sensitive", encoding="utf-8")

    result = read_local_file.invoke({"path": "~/" + relative})
    assert result.startswith("ERROR: forbidden:"), result


def test_browser_login_data_is_forbidden(tmp_path):
    """C-26: browser credential databases are denied by filename."""
    target = tmp_path / "Login Data"
    target.write_text("sqlite", encoding="utf-8")

    result = read_local_file.invoke({"path": str(target)})
    assert result.startswith("ERROR: forbidden:"), result


def test_list_directory_never_leaks_denied_entries(tmp_path):
    """C-27: denied entries are filtered before any stat/metadata lookup."""
    directory = tmp_path / "listing"
    directory.mkdir()
    (directory / ".env").write_text("SECRET=1", encoding="utf-8")
    (directory / "id_rsa").write_text("key", encoding="utf-8")
    (directory / "notes.txt").write_text("hello", encoding="utf-8")

    result = list_directory.invoke({"path": str(directory)})

    assert "notes.txt" in result
    assert ".env" not in result
    assert "id_rsa" not in result


def test_allowed_dotfile_can_be_read(tmp_path):
    """C-26: the opt-in allowlist still permits templates like .env.example."""
    target = tmp_path / ".env.example"
    target.write_text("KNIGHT_HOST=127.0.0.1\n", encoding="utf-8")

    result = read_local_file.invoke({"path": str(target)})
    assert "--- FILE:" in result
    assert "KNIGHT_HOST=127.0.0.1" in result


def test_negative_offset_is_rejected(tmp_path):
    """C-28: negative offsets must not use Python's negative-slice semantics."""
    target = tmp_path / "paged.txt"
    target.write_text("A" * 100 + "TAIL", encoding="utf-8")

    result = read_local_file.invoke({"path": str(target), "offset_chars": -100})
    assert "ERROR: invalid_offset" in result
    assert "TAIL" not in result


@pytest.mark.skipif(
    platform.system().lower() == "windows",
    reason="symlink creation requires privileges on Windows",
)
def test_symlink_to_system_file_is_forbidden(tmp_path):
    """C-29: a symlink inside the workspace cannot escape to /etc/passwd."""
    from pathlib import Path

    if not Path("/etc/passwd").exists():
        pytest.skip("/etc/passwd is not available on this platform")

    link = tmp_path / "innocent.txt"
    link.symlink_to("/etc/passwd")

    result = read_local_file.invoke({"path": str(link)})
    assert result.startswith("ERROR: forbidden:"), result


