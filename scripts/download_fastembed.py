import argparse
import sys
from pathlib import Path

from fastembed import TextEmbedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FastEmbed ONNX embedding model.")
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model name (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/models/fastembed",
        help="Destination directory (default: data/models/fastembed)",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FastEmbed model '{args.model_name}' to {dest_dir}...")
    try:
        # Initializing TextEmbedding without local_files_only will download to cache_dir
        _ = TextEmbedding(model_name=args.model_name, cache_dir=str(dest_dir))
        print(f"FastEmbed model downloaded and verified successfully in: {dest_dir}")
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(f"Failed to download FastEmbed model: {exc}")


if __name__ == "__main__":
    main()
