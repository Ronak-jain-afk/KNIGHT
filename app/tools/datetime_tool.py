from datetime import datetime

from langchain_core.tools import tool

from app.config import get_settings
from app.utils.timezone import resolve_timezone, timezone_label


@tool
def current_datetime() -> str:
    """Return the current date, time, and timezone information."""
    tz = resolve_timezone(get_settings().tz)
    now = datetime.now(tz)
    return f"Current time: {now.isoformat()} ({timezone_label(tz)})"
