import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.llm.llama_args import build_llama_argv, default_vendor_dir, find_llama_bin, llama_env


def find_model_file(models_dir: Path, explicit_model: str | None = None) -> Path:
    if explicit_model:
        p = Path(explicit_model)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified model file does not exist: {explicit_model}")

    default_name = "Qwen3.5-4B-UD-Q4_K_XL.gguf"
    default_path = models_dir / default_name
    if default_path.exists():
        return default_path

    # Look for any gguf in models_dir
    ggufs = list(models_dir.glob("*.gguf"))
    if ggufs:
        return ggufs[0]

    raise FileNotFoundError(
        f"No .gguf model found in {models_dir}. "
        "Run `uv run python scripts/download_model.py` first."
    )


def redact_argv(argv: list[str]) -> str:
    """Render argv for logging without leaking the API key."""
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        redacted.append(token)
        if token in ("--api-key",):
            hide_next = True
    return " ".join(redacted)


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run llama-server with verified settings.")
    parser.add_argument("--model", default=None, help="Path to GGUF model file")
    parser.add_argument("--port", default=8080, type=int, help="Server port (default: 8080)")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to KNIGHT_LLAMA_API_KEY; required)",
    )
    parser.add_argument(
        "--context-tokens",
        "-c",
        default=settings.context_tokens,
        type=int,
        help="Context tokens (default: 16384; never 0)",
    )
    parser.add_argument(
        "--ngl",
        default=99,
        type=int,
        help="Number of GPU layers to offload (default: 99)",
    )
    parser.add_argument(
        "--vendor-dir",
        default=None,
        help="Path to llama vendor directory (default: <repo>/vendor/llama)",
    )
    args = parser.parse_args()

    vendor_dir = Path(args.vendor_dir) if args.vendor_dir else default_vendor_dir()
    llama_bin = find_llama_bin(vendor_dir)
    model_path = find_model_file(settings.models_dir, args.model)
    api_key = (args.api_key or settings.llama_api_key or "").strip()
    if not api_key:
        sys.exit(
            "Error: no llama-server API key configured. Set KNIGHT_LLAMA_API_KEY "
            "or pass --api-key; refusing to start with a guessable key."
        )

    cmd = build_llama_argv(
        llama_bin=llama_bin,
        model_path=model_path,
        port=args.port,
        api_key=api_key,
        settings=settings,
        context_tokens=args.context_tokens,
        ngl=args.ngl,
    )

    env = llama_env(llama_bin)

    print("Starting llama-server:")
    print(redact_argv(cmd))
    if platform.system().lower() != "windows":
        os.execvpe(cmd[0], cmd, env)
    else:
        try:
            proc = subprocess.run(cmd, check=False, env=env)
            sys.exit(proc.returncode)
        except KeyboardInterrupt:
            print("\nllama-server stopped by user.")


if __name__ == "__main__":
    main()
