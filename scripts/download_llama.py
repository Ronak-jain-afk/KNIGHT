import argparse
import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

LLAMA_REPO = "ggml-org/llama.cpp"


def version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version without float coercion (``12.10`` > ``12.9``)."""
    parts: list[int] = []
    for piece in version.split("."):
        digits = re.match(r"\d+", piece.strip())
        if digits:
            parts.append(int(digits.group(0)))
    return tuple(parts)


def detect_cuda_version() -> str | None:
    """Check nvidia-smi for installed CUDA version."""
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
        match = re.search(r"CUDA\s+(?:UMD\s+)?Version:\s*(\d+\.\d+)", output)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def get_default_cuda(is_win: bool, detected: str | None) -> str:
    if detected:
        ver = version_tuple(detected)
        if ver:
            threshold = version_tuple("13.4")
            if is_win:
                return "13.4" if ver >= threshold else "12.4"
            return "13.4" if ver >= threshold else "12.8"
    return "13.4" if is_win else "12.8"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> bool:
    return hmac.compare_digest(_sha256_file(path).lower(), expected_sha256.lower())


def fetch_release_digests(tag: str) -> dict[str, str]:
    """Fetch published asset SHA-256 digests from the GitHub release metadata."""
    url = f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{tag}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "knight-setup",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)

    digests: dict[str, str] = {}
    for asset in data.get("assets", []):
        digest = asset.get("digest")
        name = asset.get("name")
        if isinstance(digest, str) and digest.startswith("sha256:") and name:
            digests[name] = digest.split(":", 1)[1].lower()
    return digests


def download_file(url: str, dest_path: Path, expected_sha256: str | None) -> None:
    """Download to a ``.part`` file, verify SHA-256, then atomically rename."""
    if not expected_sha256:
        raise ValueError(
            f"No published SHA-256 digest for {dest_path.name}; refusing to download "
            "an unverified llama.cpp archive."
        )

    print(f"Downloading: {url} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, expected_sha256.lower()):
            raise ValueError(
                f"Checksum mismatch for {dest_path.name}: expected "
                f"{expected_sha256.lower()}, got {actual}"
            )
        part_path.replace(dest_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


def _safe_member_path(dest_dir: Path, member_name: str) -> Path:
    """Reject absolute/traversal archive members and return the resolved target."""
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Archive member escapes destination directory: {member_name!r}")

    target = (dest_dir / normalized).resolve()
    if not target.is_relative_to(dest_dir.resolve()):
        raise ValueError(f"Archive member escapes destination directory: {member_name!r}")
    return target


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {archive_path.name} to {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                _safe_member_path(dest_dir, info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"Archive contains a symlink: {info.filename!r}")
            zf.extractall(dest_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            members = tf.getmembers()
            for member in members:
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Archive contains a link/device: {member.name!r}")
                _safe_member_path(dest_dir, member.name)
            # ``data`` filter is belt-and-braces on top of the explicit checks.
            tf.extractall(dest_dir, members=members, filter="data")
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned llama.cpp release.")
    parser.add_argument("--tag", default="b11980", help="llama.cpp release tag (default: b11980)")
    parser.add_argument("--cuda", default=None, help="CUDA version string (e.g. 13.4, 12.8, 12.4)")
    parser.add_argument(
        "--variant",
        choices=["cuda", "vulkan", "cpu"],
        default="cuda",
        help="Backend variant (cuda, vulkan, cpu)",
    )
    parser.add_argument(
        "--dest",
        default="vendor/llama",
        help="Destination directory (default: vendor/llama)",
    )
    args = parser.parse_args()

    is_win = platform.system().lower() == "windows"
    dest_dir = Path(args.dest)
    cache_dir = Path("data/cache/llama_downloads")
    cache_dir.mkdir(parents=True, exist_ok=True)

    detected_cuda = detect_cuda_version()
    if detected_cuda:
        print(f"Detected CUDA version: {detected_cuda}")

    cuda_ver = args.cuda or get_default_cuda(is_win, detected_cuda)
    tag = args.tag

    try:
        digests = fetch_release_digests(tag)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(f"Error: could not fetch checksum metadata for {tag}: {exc}")

    base_url = f"https://github.com/{LLAMA_REPO}/releases/download/{tag}"
    downloads: list[str] = []

    if is_win:
        if args.variant == "cuda":
            bin_name = f"llama-{tag}-bin-win-cuda-{cuda_ver}-x64.zip"
            cudart_name = f"cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip"
            downloads.extend([bin_name, cudart_name])
        elif args.variant == "vulkan":
            downloads.append(f"llama-{tag}-bin-win-vulkan-x64.zip")
        else:
            downloads.append(f"llama-{tag}-bin-win-cpu-x64.zip")
    else:
        # Linux x86_64
        if args.variant == "cuda":
            bin_name = f"llama-{tag}-bin-ubuntu-cuda-{cuda_ver}-x64.tar.gz"
            cudart_name = f"cudart-llama-{tag}-bin-ubuntu-cuda-{cuda_ver}-x64.tar.gz"
            downloads.extend([bin_name, cudart_name])
        elif args.variant == "vulkan":
            downloads.append(f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz")
        else:
            downloads.append(f"llama-{tag}-bin-ubuntu-x64.tar.gz")

    for file_name in downloads:
        expected = digests.get(file_name)
        if not expected:
            sys.exit(
                f"Error: no published SHA-256 digest for {file_name}. "
                "Refusing to install an unverified binary."
            )
        url = f"{base_url}/{file_name}"
        archive_path = cache_dir / file_name
        if archive_path.exists() and not verify_sha256(archive_path, expected):
            print(f"Cached archive failed checksum verification; re-downloading {file_name}")
            archive_path.unlink(missing_ok=True)
        if not archive_path.exists():
            download_file(url, archive_path, expected)
        if not verify_sha256(archive_path, expected):
            sys.exit(f"Error: checksum verification failed for {archive_path}")
        extract_archive(archive_path, dest_dir)

    # Flatten any files in subdirectories into dest_dir
    for sub in list(dest_dir.iterdir()):
        if sub.is_dir():
            for item in list(sub.iterdir()):
                target = dest_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            try:
                sub.rmdir()
            except OSError:
                pass

    server_bin = dest_dir / ("llama-server.exe" if is_win else "llama-server")
    if not server_bin.exists():
        # Check subdirectories if archive had top-level folder
        found = list(dest_dir.rglob("llama-server.exe" if is_win else "llama-server"))
        if found:
            server_bin = found[0]
            # Copy or move files to top level of dest_dir
            for f in server_bin.parent.iterdir():
                target = dest_dir / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
            server_bin = dest_dir / ("llama-server.exe" if is_win else "llama-server")

    if not server_bin.exists():
        sys.exit(f"Error: {server_bin} was not found after extraction!")

    if not is_win:
        os.chmod(server_bin, 0o755)

    print(f"Successfully installed llama-server to: {server_bin}")


if __name__ == "__main__":
    main()
