# SPDX-License-Identifier: Apache-2.0
import contextvars
import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from app.config import Settings

current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_run_id", default="")
current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_thread_id", default=""
)

_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+"), r"\1[REDACTED]"),
    (re.compile(r"(sk-[A-Za-z0-9_\-]{8})[A-Za-z0-9_\-]+"), r"\1[REDACTED]"),
    (re.compile(r"(--api-key\s+)[^\s]+"), r"\1[REDACTED]"),
)


def redact_text(text: str) -> str:
    """Apply every secret-redaction pattern to a fully formatted string."""
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id.get() or "-"
        record.thread_id = current_thread_id.get() or "-"

        # Resolve lazy %-args first so secrets passed as arguments are caught
        # (`logger.info("api_key=%s", key)`), then flatten the record.
        try:
            message = record.getMessage()
        except (TypeError, ValueError, KeyError):
            message = str(record.msg)
        record.msg = redact_text(message)
        record.args = ()

        # Tracebacks bypass msg/args entirely; redact the formatted exception too.
        if record.exc_info:
            record.exc_text = redact_text(
                logging.Formatter().formatException(record.exc_info)
            )
        return True


def setup_logging(settings: Settings) -> None:
    settings.ensure_dirs()
    log_file = settings.logs_dir / "knight.log"

    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [run_id=%(run_id)s thread_id=%(thread_id)s] %(message)s"
    )
    redacting_filter = RedactingFilter()

    # Rotating file handler (5 MB x 3)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redacting_filter)
    root_logger.addHandler(file_handler)

    # Console stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(redacting_filter)
    root_logger.addHandler(stdout_handler)
