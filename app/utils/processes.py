"""Cross-platform process and port helpers.

Replaces ``pkill -9 -f ...`` which matched command lines as substrings and could
kill unrelated processes (editors, CI jobs, other users' servers). Everything here
works by PID and uses only the standard library plus OS utilities.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import time

IS_WINDOWS = platform.system().lower() == "windows"


def _run(argv: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def find_listener_pids(port: int, host: str = "127.0.0.1") -> list[int]:
    """Best-effort PIDs listening on ``port`` (empty when no OS tool is available)."""
    pids: set[int] = set()

    if IS_WINDOWS:
        result = _run(["netstat", "-ano", "-p", "tcp"])
        if result and result.returncode == 0:
            suffix = f":{port}"
            for line in result.stdout.splitlines():
                cols = line.split()
                if (
                    len(cols) >= 5
                    and cols[1].endswith(suffix)
                    and cols[3].upper() == "LISTENING"
                    and cols[4].isdigit()
                ):
                    pids.add(int(cols[4]))
        return sorted(pids)

    result = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    if result and result.returncode == 0:
        for token in result.stdout.split():
            if token.strip().isdigit():
                pids.add(int(token))

    if not pids:
        result = _run(["ss", "-lptnH", f"sport = :{port}"])
        if result and result.returncode == 0:
            for line in result.stdout.splitlines():
                for chunk in line.split("pid=")[1:]:
                    token = chunk.split(",")[0].strip()
                    if token.isdigit():
                        pids.add(int(token))

    return sorted(pids)


def process_name(pid: int) -> str:
    """Best-effort executable name for ``pid`` (empty string when unknown)."""
    if IS_WINDOWS:
        result = _run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
        if result and result.returncode == 0 and result.stdout.strip():
            first = result.stdout.splitlines()[0].split(",")[0].strip('" ')
            return first
        return ""

    proc_comm = f"/proc/{pid}/comm"
    try:
        with open(proc_comm, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        pass

    result = _run(["ps", "-p", str(pid), "-o", "comm="])
    if result and result.returncode == 0:
        return result.stdout.strip()
    return ""


def process_cmdline(pid: int) -> str:
    """Best-effort full command line for ``pid`` (empty string when unknown)."""
    if IS_WINDOWS:
        result = _run(
            ["powershell", "-NoProfile", "-Command", f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"]
        )
        if result and result.returncode == 0:
            return result.stdout.strip()
        return ""

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass

    result = _run(["ps", "-p", str(pid), "-o", "args="])
    if result and result.returncode == 0:
        return result.stdout.strip()
    return ""


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if IS_WINDOWS:
        result = _run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
        return bool(result and result.returncode == 0 and str(pid) in result.stdout)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    # A terminated child of this process stays visible as a zombie until reaped;
    # treat zombies as dead so wait loops do not hang for their full timeout.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            fields = fh.read().split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except OSError:
        pass
    return True


def terminate_pid(pid: int, timeout: float = 5.0) -> bool:
    """Terminate a single PID (and its tree). Returns True when it is gone."""
    if not is_pid_alive(pid):
        return True

    if IS_WINDOWS:
        _run(["taskkill", "/PID", str(pid), "/T", "/F"])
    else:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False

            deadline = time.time() + (timeout if sig == signal.SIGTERM else 2.0)
            while time.time() < deadline:
                if not is_pid_alive(pid):
                    return True
                time.sleep(0.1)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.1)
    return not is_pid_alive(pid)


def _signal_child(proc: subprocess.Popen, sig: int) -> None:
    """Signal a child process, preferring its own process group.

    If the child never called ``setsid()``/``start_new_session``, ``getpgid()``
    returns *our* process group. Signalling that group would kill the parent
    application (or the test runner), so in that case signal the PID only.
    """
    try:
        child_pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    try:
        own_pgid = os.getpgid(0)
    except OSError:  # pragma: no cover - platform specific
        own_pgid = None

    try:
        if child_pgid == own_pgid:
            os.kill(proc.pid, sig)
        else:
            os.killpg(child_pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(proc.pid, sig)
        except ProcessLookupError:
            return


def terminate_process_tree(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    """Terminate a child process started by this application (and its children)."""
    if proc is None or proc.poll() is not None:
        return

    if IS_WINDOWS:
        _run(["taskkill", "/PID", str(proc.pid), "/T", "/F"])
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return

    _signal_child(proc, signal.SIGTERM)

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal_child(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
