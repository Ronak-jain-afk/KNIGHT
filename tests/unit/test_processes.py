# SPDX-License-Identifier: Apache-2.0
"""Regression tests for C-11/C-12/C-17/C-18: process handling and key logging."""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.utils import processes

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sleeper():
    procs: list[subprocess.Popen] = []

    def start() -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        procs.append(proc)
        return proc

    yield start

    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_terminate_process_tree_only_kills_the_given_child(sleeper):
    """C-11: terminating the tracked child must not touch siblings or our group."""
    first = sleeper()
    second = sleeper()
    time.sleep(0.1)

    processes.terminate_process_tree(first, timeout=5)
    first.wait(timeout=5)

    assert first.poll() is not None
    assert second.poll() is None


@pytest.mark.skipif(
    shutil.which("lsof") is None and shutil.which("ss") is None,
    reason="no supported port-inspection tool available",
)
def test_free_port_refuses_to_kill_unrelated_listener():
    """C-12: an unrelated listener is reported, never pattern-killed."""
    from scripts import run_all

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    code = (
        "import http.server, sys; "
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), "
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(port)])
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)

        if not processes.find_listener_pids(port):
            pytest.skip("port inspection tools returned no PID")

        assert run_all.free_port(port, "test listener") is False
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_windows_netstat_parsing(monkeypatch):
    """C-18: the Windows branch has a working port -> PID parser."""

    class Result:
        returncode = 0
        stdout = (
            "  TCP    127.0.0.1:8080    0.0.0.0:0    LISTENING    4242\n"
            "  TCP    127.0.0.1:9000    0.0.0.0:0    LISTENING    5555\n"
        )

    monkeypatch.setattr(processes, "IS_WINDOWS", True)
    monkeypatch.setattr(processes, "_run", lambda argv, timeout=5.0: Result())

    assert processes.find_listener_pids(8080) == [4242]


def test_no_pkill_in_launch_paths():
    """C-11/C-12: neither the manager nor the runner may pattern-kill processes."""
    for relative in ("app/llm/manager.py", "scripts/run_all.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "pkill" not in source, relative


def test_api_key_is_logged_as_a_fingerprint_only():
    """C-17: stdout must never contain a prefix/substring of the generated key."""
    from scripts.run_all import key_fingerprint

    key = "not-a-real-key-abcdef"
    fingerprint = key_fingerprint(key)
    assert fingerprint != key[:8]
    assert key not in fingerprint

    source = (REPO_ROOT / "scripts/run_all.py").read_text(encoding="utf-8")
    assert "generated_api_key[:8]" not in source
    assert "key_fingerprint" in source
