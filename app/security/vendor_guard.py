# SPDX-License-Identifier: Apache-2.0
"""Vendor asset verification: check recorded versions and SHA-256 digests against manifest."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_VENDOR_DIR = Path(__file__).resolve().parents[1] / "web" / "vendor"
MANIFEST_FILE = "manifest.json"


def verify_vendor_manifest(vendor_dir: Path | None = None) -> dict[str, str]:
    """Verify that all vendored web assets match their manifest SHA-256 digests.

    Raises RuntimeError if manifest is missing or if any file fails integrity verification.
    """
    vdir = vendor_dir or DEFAULT_VENDOR_DIR
    manifest_path = vdir / MANIFEST_FILE

    if not manifest_path.exists():
        raise RuntimeError(f"Vendor manifest missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to read vendor manifest {manifest_path}: {exc}") from exc

    verified_assets: dict[str, str] = {}
    for filename, meta in manifest.items():
        asset_path = vdir / filename
        if not asset_path.exists():
            raise RuntimeError(f"Vendored asset missing: {asset_path}")

        expected_hash = meta.get("sha256", "").strip().lower()
        if not expected_hash:
            raise RuntimeError(f"Missing expected sha256 for vendored asset: {filename}")

        actual_hash = hashlib.sha256(asset_path.read_bytes()).hexdigest().lower()
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise RuntimeError(
                f"Vendored asset integrity mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        verified_assets[filename] = meta.get("version", "unknown")

    logger.debug("Verified %d vendored asset(s) against manifest", len(verified_assets))
    return verified_assets
