import logging
import os
import platform
import re
from pathlib import Path

from langchain_core.tools import tool

from app.agent.sources import current_sources
from app.config import get_settings

logger = logging.getLogger(__name__)

BLOCKED_NAME_PATTERNS = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".kdbx",
    "credentials",
    ".sqlite",
    ".sqlite3",
}

BLOCKED_NAME_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "private_key",
    "privatekey",
    "api_key",
    "apikey",
    "auth_token",
    "access_token",
    "session_key",
)

# Exact filenames that commonly hold credentials.
BLOCKED_FILENAMES = {
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".bash_history",
    ".zsh_history",
    ".fish_history",
    ".python_history",
    ".psql_history",
    ".mysql_history",
    ".wget-hsts",
    "known_hosts",
    "authorized_keys",
    "login data",
    "login data for account",
    "cookies",
    "cookies-journal",
    "key4.db",
    "logins.json",
    "cert9.db",
    "credentials.json",
}

BLOCKED_DIRECTORY_PARTS = {
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".password-store",
    ".mozilla",
    ".thunderbird",
    "google-chrome",
    "brave-browser",
    "chromium",
    "microsoft-edge",
    "opera",
    "vivaldi",
}

BLOCKED_PATH_FRAGMENTS = (
    "/.config/gh",
    "/.config/gcloud",
    "/.config/gh/",
    "/.local/share/keyrings",
    "/.docker/config.json",
)

# Dotfiles we still allow (templates and ordinary project metadata).
ALLOWED_DOTFILES = {
    ".env.example",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".prettierrc",
    ".ruff.toml",
}

BLOCKED_SYSTEM_PREFIXES = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/etc",
    "/boot",
    "/root",
)

BLOCKED_INTERNAL_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
}


def resolve_system_path(raw_path: str) -> Path:
    """Resolve a user-provided path cross-platform.

    Supports:
    - Windows drive paths on Linux/WSL (e.g. C:\\dev\\walk -> /mnt/c/dev/walk)
    - Tilde expansion (e.g. ~/projects -> /home/username/projects)
    - Relative paths against project root and documents directory
    - Standard Windows paths on Windows
    """
    cleaned = raw_path.strip().strip("'\"")

    # 1. Expand ~ (user home)
    if cleaned == "~" or cleaned.startswith(("~/", "~\\")):
        cleaned = str(Path(cleaned).expanduser())

    # 2. Check for Windows drive letter like C:\ or C:/ or c:\
    drive_match = re.match(r"^([a-zA-Z]):[\\/](.*)", cleaned)
    if drive_match:
        drive_letter = drive_match.group(1).lower()
        subpath = drive_match.group(2).replace("\\", "/")
        if platform.system().lower() != "windows":
            # Running on Linux / WSL2: convert drive to /mnt/x/
            wsl_candidate = Path(f"/mnt/{drive_letter}/{subpath}")
            alt_candidate = Path(f"/{drive_letter}/{subpath}")
            if alt_candidate.exists() and not wsl_candidate.exists():
                return alt_candidate.resolve()
            return wsl_candidate
        else:
            # On native Windows
            return Path(f"{drive_match.group(1).upper()}:/{subpath}").resolve()

    # 3. On Unix-like systems, normalize any accidental backslashes
    if platform.system().lower() != "windows" and "\\" in cleaned:
        cleaned = cleaned.replace("\\", "/")

    p = Path(cleaned)
    if not p.is_absolute():
        # Check relative to cwd first
        cwd_cand = (Path.cwd() / p).resolve()
        if cwd_cand.exists():
            return cwd_cand

        # Check relative to documents directory
        settings = get_settings()
        doc_cand = (settings.documents_dir / p).resolve()
        if doc_cand.exists():
            return doc_cand

        return cwd_cand

    return p.resolve()


def is_path_safe(resolved: Path) -> tuple[bool, str]:
    """Check if the path targets safe files, blocking sensitive keys, system files, and .env."""
    posix_str = str(resolved).replace("\\", "/")
    lowered = posix_str.lower()

    # Check blocked system directories (Linux/WSL)
    for sys_prefix in BLOCKED_SYSTEM_PREFIXES:
        if lowered == sys_prefix or lowered.startswith(sys_prefix + "/"):
            return False, f"Access denied: System directory '{sys_prefix}' is restricted."

    for fragment in BLOCKED_PATH_FRAGMENTS:
        if fragment in lowered:
            return False, "Access denied: Credential store directory is restricted."

    # Check internal junk directories and credential directories
    parts = [part for part in resolved.parts if part not in ("/", "\\")]
    for part in parts:
        if part.lower() in BLOCKED_DIRECTORY_PARTS:
            return False, f"Access denied: Credential directory '{part}' is blocked."
        if part in BLOCKED_INTERNAL_PARTS:
            return False, f"Access denied: Internal development directory '{part}' is restricted."

    # Check blocked file names / extensions
    filename = resolved.name.lower()
    if filename in ALLOWED_DOTFILES:
        return True, ""
    if filename in BLOCKED_FILENAMES:
        return False, f"Access denied: Sensitive file '{resolved.name}' is blocked."
    for substring in BLOCKED_NAME_SUBSTRINGS:
        if substring in filename:
            return False, "Access denied: Filename looks like a credential store."

    for pattern in BLOCKED_NAME_PATTERNS:
        if pattern.startswith(".") and filename.endswith(pattern):
            return False, f"Access denied: Sensitive file extension '{pattern}' is blocked."
        if pattern in filename:
            return False, f"Access denied: Sensitive credential pattern '{pattern}' is blocked."

    # Deny dotfiles and dot-directories unless explicitly allowlisted.
    for index, part in enumerate(reversed(parts)):
        if part.startswith(".") and part not in (".", ".."):
            if index == 0 and part.lower() in ALLOWED_DOTFILES:
                continue
            return False, f"Access denied: Hidden file or directory '{part}' is blocked."

    return True, ""


def format_size(num_bytes: int) -> str:
    """Format byte counts into human-readable strings."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


@tool
def list_directory(path: str = ".", max_items: int = 60) -> str:
    """List files and subdirectories in a local directory to explore the file system or project structure.

    Args:
        path: Directory path to explore. Accepts Windows paths (e.g. C:\\dev\\walk),
              WSL paths (/mnt/c/...), user home paths (~/projects), or relative paths (default: ".").
        max_items: Maximum items to display before truncating (default 60).

    Returns:
        Formatted listing of files and directories, or an error starting with 'ERROR:'.
    """
    try:
        resolved = resolve_system_path(path)
    except (ValueError, OSError) as exc:
        return f"ERROR: invalid_path: Could not resolve '{path}': {exc}"

    safe, reason = is_path_safe(resolved)
    if not safe:
        return f"ERROR: forbidden: {reason}"

    if not resolved.exists():
        return f"ERROR: directory_not_found: Path '{path}' (resolved to '{resolved}') does not exist."

    if resolved.is_file():
        return (
            f"NOTE: '{path}' is a file ({format_size(resolved.stat().st_size)}), not a directory. "
            f"Use read_local_file('{path}') to view its contents."
        )

    try:
        entries = list(resolved.iterdir())
    except PermissionError:
        return f"ERROR: permission_denied: Insufficient permissions to read directory '{resolved}'."
    except OSError as exc:
        return f"ERROR: read_error: Could not list directory: {exc}"

    # Separate dirs and files, ignore hidden files by default
    dirs: list[Path] = []
    files: list[Path] = []

    for entry in entries:
        # Filter before touching metadata: denied files must not leak existence/size.
        if entry.is_symlink():
            continue
        if entry.name.startswith(".") and entry.name != ".env.example":
            continue
        entry_safe, _ = is_path_safe(entry)
        if not entry_safe:
            continue
        if entry.is_dir():
            dirs.append(entry)
        else:
            files.append(entry)

    try:
        max_items_val = int(max_items) if max_items is not None else 60
    except (ValueError, TypeError):
        max_items_val = 60

    dirs.sort(key=lambda d: d.name.lower())
    files.sort(key=lambda f: f.name.lower())

    total_count = len(dirs) + len(files)
    lines = [f"📁 DIRECTORY: {resolved} ({len(dirs)} folders, {len(files)} files)"]

    items_added = 0
    for d in dirs:
        if items_added >= max_items_val:
            break
        lines.append(f"  📁 {d.name}/")
        items_added += 1

    for f in files:
        if items_added >= max_items_val:
            break
        try:
            sz = format_size(f.stat().st_size)
        except OSError:
            sz = "unknown size"
        lines.append(f"  📄 {f.name} ({sz})")
        items_added += 1

    if total_count > items_added:
        lines.append(f"  ... (showing {items_added} of {total_count} items. Specify a subfolder to explore deeper)")

    lines.append("\nTip: Use read_local_file('<path>') to view any file above, or list_directory on a subfolder.")
    return "\n".join(lines)


@tool
def read_local_file(path: str, offset_chars: int = 0, max_chars: int = 4000) -> str:
    """Read contents of a local file (code, text, markdown, json, csv, config, or logs).
    If a directory path is given, it will automatically list the directory contents instead.

    Args:
        path: Path to the file. Accepts Windows paths (e.g. C:\\dev\\walk\\file.txt),
              WSL paths (/mnt/c/...), user home paths (~/projects/...), or relative paths.
        offset_chars: Starting character offset for pagination (default 0).
        max_chars: Maximum characters to read (default 4000, capped at 12000).

    Returns:
        The text content of the file or directory listing, or an error string starting with 'ERROR:'.
    """
    try:
        resolved = resolve_system_path(path)
    except (ValueError, OSError) as exc:
        return f"ERROR: invalid_path: Could not resolve '{path}': {exc}"

    safe, reason = is_path_safe(resolved)
    if not safe:
        return f"ERROR: forbidden: {reason}"

    if not resolved.exists():
        return f"ERROR: file_not_found: Path '{path}' (resolved to '{resolved}') does not exist on this system."

    # If user/model passed a directory instead of a file, gracefully list directory!
    if resolved.is_dir():
        return (
            f"NOTE: '{path}' is a directory, not a file. Here are its contents:\n\n"
            + list_directory.invoke({"path": str(resolved)})
        )

    # Safely convert offset and limit arguments
    try:
        offset_val = int(offset_chars) if offset_chars is not None else 0
    except (ValueError, TypeError):
        offset_val = 0

    if offset_val < 0:
        return "ERROR: invalid_offset: offset_chars must be >= 0."

    try:
        max_val = int(max_chars) if max_chars is not None else 4000
    except (ValueError, TypeError):
        max_val = 4000

    # Re-resolve immediately before opening to narrow the symlink-swap window.
    try:
        resolved = resolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return f"ERROR: read_error: Cannot resolve '{resolved}': {exc}"

    safe, reason = is_path_safe(resolved)
    if not safe:
        return f"ERROR: forbidden: {reason}"

    # Open without following a final symlink, then verify the inode we opened is the
    # one we validated (TOCTOU guard).
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        before = resolved.stat()
        fd = os.open(resolved, flags)
    except (OSError, PermissionError) as exc:
        return f"ERROR: read_error: Cannot read '{resolved}': {exc}"

    try:
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            return "ERROR: read_error: File changed while it was being validated."

        if after.st_size > 5_000_000:
            return (
                f"ERROR: file_too_large: File size ({format_size(after.st_size)}) "
                "exceeds the 5 MB limit."
            )

        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership transferred to the file object
            raw = handle.read()
    finally:
        if fd != -1:
            os.close(fd)

    if b"\x00" in raw[:1024]:
        return (
            f"ERROR: binary_file: '{path}' appears to be a compiled or binary file. "
            "Only text, code, config, logs, and document files can be read."
        )

    content = raw.decode("utf-8", errors="replace")

    total_len = len(content)
    if offset_val >= total_len:
        return (
            f"ERROR: offset_out_of_bounds: File length is {total_len} chars, offset {offset_val} is beyond end."
        )

    limit = min(max(max_val, 100), 12000)
    chunk = content[offset_val : offset_val + limit]

    # Record source citation if collector active
    collector = current_sources.get()
    display_name = str(resolved)
    try:
        cwd = Path.cwd().resolve()
        if resolved.is_relative_to(cwd):
            display_name = str(resolved.relative_to(cwd))
    except ValueError:
        pass

    if collector:
        collector.add(
            f"file://{display_name}",
            title=f"File: {display_name}",
            snippet=chunk[:200].replace("\n", " "),
        )

    return (
        f"--- FILE: {display_name} (Total: {total_len} chars | Offset: {offset_val}..{offset_val + len(chunk)}) ---\n"
        f"{chunk}"
    )
