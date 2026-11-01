# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GGUF model from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        default="unsloth/Qwen3.5-4B-GGUF",
        help="Hugging Face repo ID (default: unsloth/Qwen3.5-4B-GGUF)",
    )
    parser.add_argument(
        "--filename",
        default="Qwen3.5-4B-UD-Q4_K_XL.gguf",
        help="Target GGUF file name (default: Qwen3.5-4B-UD-Q4_K_XL.gguf)",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/models",
        help="Destination directory (default: data/models)",
    )
    parser.add_argument(
        "--fastembed",
        action="store_true",
        help="Also download the FastEmbed embedding model (BAAI/bge-small-en-v1.5) for semantic memory search",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.fastembed:
        print("Downloading FastEmbed model 'BAAI/bge-small-en-v1.5'...")
        from fastembed import TextEmbedding

        fastembed_dir = dest_dir / "fastembed"
        fastembed_dir.mkdir(parents=True, exist_ok=True)
        try:
            _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(fastembed_dir))
            print(f"FastEmbed model downloaded and verified in: {fastembed_dir}")
        except (OSError, RuntimeError, ValueError) as exc:
            sys.exit(f"Failed to download FastEmbed model: {exc}")
        return

    target_path = dest_dir / args.filename

    if target_path.exists() and target_path.stat().st_size > 100_000_000:
        print(f"Model already present at {target_path} ({target_path.stat().st_size} bytes).")
        return

    print(f"Downloading {args.filename} from {args.repo_id}...")
    try:
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(dest_dir),
        )
        print(f"Model downloaded successfully to: {downloaded}")
    except (HfHubHTTPError, OSError) as exc:
        sys.exit(f"Failed to download model: {exc}")


if __name__ == "__main__":
    main()
