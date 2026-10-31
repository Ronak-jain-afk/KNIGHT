import contextlib
import hashlib
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings
from app.utils.processes import find_listener_pids, process_cmdline, process_name


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def stop_process(proc: subprocess.Popen | None, name: str, timeout: int = 5) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def wait_for_llama(
    server_root: str,
    api_key: str,
    proc: subprocess.Popen | None = None,
    timeout: int = 120,
) -> bool:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.time()
    while time.time() - start_time < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{server_root}/health", headers=headers)
                if resp.status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(1.0)
    return False


def free_port(port: int, label: str) -> bool:
    """Report an occupied port; never pattern-kill an unrelated process.

    Identifying the listener and refusing to touch it is intentional: a
    pattern-based kill can hit editors, CI jobs, or another user's server that
    merely mentions "llama-server" on its command line.
    """
    if not is_port_in_use(port):
        return True

    pids = find_listener_pids(port)
    if not pids:
        print(
            f"Port {port} ({label}) is in use by a process that could not be identified. "
            "Stop it manually and retry."
        )
        return False

    for pid in pids:
        name = process_name(pid)
        cmdline = process_cmdline(pid)
        print(
            f"Port {port} ({label}) is already in use by PID {pid} "
            f"({name or 'unknown'})."
        )
        if cmdline:
            print(f"  command: {cmdline}")

    print("Stop the process(es) above manually and retry.")
    return False


def key_fingerprint(api_key: str) -> str:
    """Return a short, non-reversible fingerprint for logging (never a prefix)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]


def write_api_key_file(path: Path, api_key: str) -> None:
    """Persist the runner's key (0600 on POSIX) for smoke_test.py to reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(api_key, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows only supports the read-only bit; the file is still local-only.
        pass


def main() -> None:
    settings = get_settings()
    server_root = settings.llama_base_url.rstrip("/").removesuffix("/v1")
    generated_api_key = secrets.token_urlsafe(32)
    write_api_key_file(settings.llama_api_key_file, generated_api_key)

    # Pre-flight: free ports only when the current listener is ours.
    if not free_port(8080, "llama-server"):
        sys.exit(1)
    if not free_port(settings.port, "Knight API"):
        sys.exit(1)

    # Launch llama-server via run_llama.py
    python_bin = sys.executable
    llama_script = str(Path(__file__).parent / "run_llama.py")

    llama_cmd = [
        python_bin,
        llama_script,
        "--api-key",
        generated_api_key,
        "--port",
        "8080",
    ]

    key_id = key_fingerprint(generated_api_key)
    print(
        f"Launching llama-server process "
        f"(API key fingerprint: sha256:{key_id}, length: {len(generated_api_key)})."
    )
    llama_proc = subprocess.Popen(llama_cmd)
    api_proc = None

    try:
        print("Waiting for llama-server to be ready...")
        if not wait_for_llama(server_root, generated_api_key, proc=llama_proc, timeout=120):
            print("Timed out waiting for llama-server to report healthy (or server exited)!")
            stop_process(llama_proc, "llama-server")
            sys.exit(1)

        print("llama-server is healthy! Starting Knight API server...")

        api_env = os.environ.copy()
        api_env["KNIGHT_LLAMA_API_KEY"] = generated_api_key

        from scripts.run_llama import find_model_file

        with contextlib.suppress(FileNotFoundError):
            loaded_model_file = find_model_file(settings.models_dir)
            api_env["KNIGHT_MODEL_NAME"] = loaded_model_file.name

        api_cmd = [
            python_bin,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            settings.host,
            "--port",
            str(settings.port),
        ]

        api_proc = subprocess.Popen(api_cmd, env=api_env)

        # Handle clean shutdown on Ctrl+C or SIGTERM
        def handle_signal(sig, frame):
            print("\nShutting down Knight processes...")
            stop_process(api_proc, "API server")
            stop_process(llama_proc, "llama-server")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Wait on either process
        while True:
            if llama_proc.poll() is not None:
                print("llama-server process exited unexpectedly.")
                stop_process(api_proc, "API server")
                sys.exit(llama_proc.returncode or 1)
            if api_proc and api_proc.poll() is not None:
                print("Knight API server exited.")
                stop_process(llama_proc, "llama-server")
                sys.exit(api_proc.returncode or 0)
            time.sleep(0.5)

    except (KeyboardInterrupt, SystemExit):
        stop_process(api_proc, "API server")
        stop_process(llama_proc, "llama-server")
        sys.exit(0)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"Error in runner: {exc}")
        stop_process(api_proc, "API server")
        stop_process(llama_proc, "llama-server")
        sys.exit(1)


if __name__ == "__main__":
    main()
