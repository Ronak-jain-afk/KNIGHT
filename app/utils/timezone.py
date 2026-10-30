"""Configured-timezone helpers shared by the prompt and datetime tool."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

_FALLBACK: tzinfo = UTC


def resolve_timezone(tz_name: str | None = None) -> tzinfo:
    """Return the configured IANA zone, falling back to the host local zone.

    An empty name means "use the host timezone"; an unknown IANA name is logged
    and also falls back to the host zone rather than crashing a tool call.
    """
    name = (tz_name or "").strip()
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            logger.warning("Unknown timezone %r (%s); using host timezone", name, exc)

    try:
        local = datetime.now().astimezone().tzinfo
    except (OSError, ValueError):  # pragma: no cover - extremely defensive
        local = None
    return local or _FALLBACK


def timezone_label(tz: tzinfo) -> str:
    """Best-effort human label for a tzinfo (IANA key when available)."""
    key = getattr(tz, "key", None)
    if key:
        return str(key)
    name = datetime.now(tz).tzname()
    return name or "local"
