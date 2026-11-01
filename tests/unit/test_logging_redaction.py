# SPDX-License-Identifier: Apache-2.0
"""Regression tests for M-09/M-10: redaction of lazy args and exception text."""

import logging
import sys

from app.logging_setup import RedactingFilter, redact_text


def _record(
    msg: object,
    args: tuple = (),
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="knight.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


def test_redacts_secret_passed_as_lazy_arg():
    """M-09: `logger.info('key=%s', key)` must not leak the key."""
    secret = "sk-abcdefgh12345678"
    record = _record("api key=%s", (secret,))

    assert RedactingFilter().filter(record) is True

    rendered = record.getMessage()
    assert secret not in rendered
    assert "sk-abcdefgh[REDACTED]" in rendered
    assert record.args == ()


def test_redacts_bearer_and_argv_args():
    """M-09: every pattern applies to formatted %-args, not just `record.msg`."""
    record = _record(
        "auth=%s command=%s",
        ("Bearer supersecrettoken", "--api-key another-secret"),
    )
    RedactingFilter().filter(record)

    rendered = record.getMessage()
    assert "supersecrettoken" not in rendered
    assert "another-secret" not in rendered
    assert rendered.count("[REDACTED]") == 2


def test_redacts_exception_traceback_text():
    """M-10: tracebacks bypass msg/args, so `exc_text` has to be redacted too."""
    try:
        raise RuntimeError("failed with Bearer supersecrettoken at /srv/private")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = _record("request failed", exc_info=exc_info)
    RedactingFilter().filter(record)

    assert record.exc_text is not None
    assert "supersecrettoken" not in record.exc_text
    assert "[REDACTED]" in record.exc_text

    formatted = logging.Formatter("%(message)s").format(record)
    assert "supersecrettoken" not in formatted


def test_redact_text_handles_all_patterns():
    text = (
        "header=Bearer abc.def.ghi "
        "key=sk-12345678deadbeef "
        "cmd=--api-key topsecret"
    )
    redacted = redact_text(text)
    for secret in ("abc.def.ghi", "deadbeef", "topsecret"):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") == 3
