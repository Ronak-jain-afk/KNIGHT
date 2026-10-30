import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import Settings, get_settings


def resolve_api_key(settings: Settings, explicit: str | None = None) -> str:
    """Resolve the runner's key: CLI flag, environment, key file, then settings."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("KNIGHT_LLAMA_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = settings.llama_api_key_file
    try:
        file_key = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        file_key = ""
    if file_key:
        return file_key
    return settings.llama_api_key.strip()


def stop_process(proc: subprocess.Popen | None, name: str, timeout: float = 5.0) -> None:
    """Terminate a spawned child without letting TimeoutExpired mask the result."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  -> {name} did not stop in {timeout}s; killing it.")
        try:
            proc.kill()
        except OSError as exc:  # pragma: no cover - process already gone
            print(f"  -> could not kill {name}: {exc}")
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            print(f"  -> {name} still running after kill.")
    except OSError as exc:
        print(f"  -> could not stop {name}: {exc}")


def check_endpoint(url: str, headers: dict | None = None) -> bool:
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(url, headers=headers or {})
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def smoke_test_llama(server_root: str, api_key: str, model_name: str) -> None:
    print(f"[1/3] Testing llama-server health at {server_root}/health...")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=10.0) as client:
        # Check health
        health_resp = client.get(f"{server_root}/health", headers=headers)
        if health_resp.status_code != 200:
            sys.exit(f"FAIL: llama-server /health returned status {health_resp.status_code}")
        print("  -> /health OK (200)")

        # --no-webui should leave the embedded UI disabled.
        ui_resp = client.get(f"{server_root}/", headers=headers)
        if ui_resp.status_code == 200 and "llama" in ui_resp.text.lower():
            print("  -> WARNING: embedded llama.cpp web UI responded; --no-webui may be missing.")
        else:
            print("  -> Embedded web UI disabled (--no-webui).")

        # Check models
        models_resp = client.get(f"{server_root}/v1/models", headers=headers)
        if models_resp.status_code != 200:
            sys.exit(f"FAIL: llama-server /v1/models returned status {models_resp.status_code}")
        models_data = models_resp.json()
        model_list = models_data.get("data", [])
        if not model_list:
            sys.exit("FAIL: /v1/models returned empty data array")
        loaded_model = model_list[0].get("id", model_name)
        print(f"  -> Model loaded: {loaded_model}")

        # Forced tool call test
        print("[2/3] Testing grammar-constrained tool call...")
        tool_payload = {
            "model": loaded_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Search the web for the latest Python release news right now.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for fresh information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
        }

        resp = client.post(
            f"{server_root}/v1/chat/completions",
            json=tool_payload,
            headers=headers,
            timeout=30.0,
        )
        if resp.status_code != 200:
            sys.exit(f"FAIL: /v1/chat/completions returned status {resp.status_code}: {resp.text}")

        res_json = resp.json()
        choices = res_json.get("choices", [])
        if not choices:
            sys.exit("FAIL: No choices in chat completion response")

        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            print("Response message was:", json.dumps(msg, indent=2))
            sys.exit(
                "FAIL: Expected tool_calls array in message, but got none or prose generation!"
            )

        call_fn = tool_calls[0].get("function", {})
        if call_fn.get("name") != "web_search":
            sys.exit(f"FAIL: Expected tool name 'web_search', got: {call_fn.get('name')}")

        print(f"  -> Tool call verified: {call_fn.get('name')}({call_fn.get('arguments')})")


def smoke_test_api(app_url: str) -> None:
    print(f"[3/3] Checking App API health at {app_url}/api/health...")
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{app_url}/api/health")
        if resp.status_code != 200:
            sys.exit(f"FAIL: App /api/health returned status {resp.status_code}")
        data = resp.json()
        if not data.get("llama", {}).get("reachable"):
            sys.exit(f"FAIL: App reports llama unreachable: {data}")
        print(f"  -> App API health OK: {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Knight's llama-server and API stack.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="llama-server API key (defaults to KNIGHT_LLAMA_API_KEY, then the runner key file)",
    )
    args = parser.parse_args()

    settings = get_settings()
    server_root = settings.llama_base_url.rstrip("/").removesuffix("/v1")
    app_url = f"http://{settings.host}:{settings.port}"
    api_key = resolve_api_key(settings, args.api_key)

    llama_proc = None
    api_proc = None

    try:
        # Check if llama-server is already running
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if not check_endpoint(f"{server_root}/health", headers=headers):
            print("llama-server not detected; spawning for smoke test...")
            llama_script = str(Path(__file__).parent / "run_llama.py")
            llama_env = os.environ.copy()
            if api_key:
                llama_env["KNIGHT_LLAMA_API_KEY"] = api_key
            llama_proc = subprocess.Popen(
                [sys.executable, llama_script],
                env=llama_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(60):
                time.sleep(1.0)
                if check_endpoint(f"{server_root}/health", headers=headers):
                    print("llama-server started successfully.")
                    break
            else:
                sys.exit("FAIL: Timed out waiting for spawned llama-server to start.")

        if not check_endpoint(f"{app_url}/api/health"):
            print("App API not detected; spawning for smoke test...")
            api_env = os.environ.copy()
            if api_key:
                api_env["KNIGHT_LLAMA_API_KEY"] = api_key
            api_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    settings.host,
                    "--port",
                    str(settings.port),
                ],
                env=api_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                time.sleep(0.5)
                if check_endpoint(f"{app_url}/api/health"):
                    print("App API started successfully.")
                    break
            else:
                sys.exit("FAIL: Timed out waiting for spawned App API to start.")

        smoke_test_llama(server_root, api_key, settings.model_name)
        smoke_test_api(app_url)
        print("\nAll smoke tests PASSED successfully!")

    finally:
        if api_proc:
            print("Stopping spawned App API process...")
            stop_process(api_proc, "App API")
        if llama_proc:
            print("Stopping spawned llama-server process...")
            stop_process(llama_proc, "llama-server")


if __name__ == "__main__":
    main()
