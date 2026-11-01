# SPDX-License-Identifier: Apache-2.0
"""Canonical llama-server launch arguments.

Both ``scripts/run_llama.py`` and the in-app model swapper must build the exact
same command line. Keeping the argv in one place prevents a hot swap from silently
dropping security/format flags required by plan §18 (``--jinja``, ``--no-webui``,
``--no-context-shift``, random ``--api-key``, restricted ``--cors-origins``, and an
explicit ``-c`` that is never 0).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from app.config import Settings

IS_WINDOWS = platform.system().lower() == "windows"
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_vendor_dir() -> Path:
    return REPO_ROOT / "vendor" / "llama"


def find_llama_bin(vendor_dir: Path | None = None) -> Path:
    """Locate the llama-server binary inside the vendor directory."""
    vendor = Path(vendor_dir) if vendor_dir else default_vendor_dir()
    bin_name = "llama-server.exe" if IS_WINDOWS else "llama-server"

    direct = vendor / bin_name
    if direct.exists():
        return direct

    matches = sorted(vendor.rglob(bin_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {bin_name} in {vendor}. "
        "Run `uv run python scripts/download_llama.py` first."
    )


def build_llama_argv(
    *,
    llama_bin: Path,
    model_path: Path,
    port: int,
    api_key: str,
    settings: Settings,
    context_tokens: int | None = None,
    ngl: int = 99,
) -> list[str]:
    """Build the canonical llama-server argv (a list, never a shell string)."""
    if not api_key or not api_key.strip():
        raise ValueError("llama-server requires a non-empty API key (refusing to start).")

    ctx = context_tokens if context_tokens is not None else settings.context_tokens
    if ctx <= 0:
        raise ValueError("Context tokens must be > 0 (never 0: 0 means the model's full context).")

    return [
        str(llama_bin),
        "-m",
        str(model_path),
        "-c",
        str(ctx),
        "-np",
        "1",
        "-ngl",
        str(ngl),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--cors-origins",
        f"http://127.0.0.1:{settings.port}",
        "--jinja",
        "--no-webui",
        "--no-context-shift",
        "--chat-template-kwargs",
        '{"enable_thinking": false}',
        "--temp",
        str(settings.temperature),
        "--top-p",
        str(settings.top_p),
        "--top-k",
        str(settings.top_k),
        "--presence-penalty",
        str(settings.presence_penalty),
        "--repeat-penalty",
        "1.0",
    ]


def llama_env(llama_bin: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for llama-server (adds the shared-library path on POSIX)."""
    env = dict(base_env if base_env is not None else os.environ)
    if not IS_WINDOWS:
        lib_dir = str(Path(llama_bin).parent.resolve())
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env
