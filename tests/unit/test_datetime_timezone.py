"""M-05/M-06: configured timezone labels and per-run date injection."""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import SystemMessage

from app.agent import middleware as middleware_module
from app.agent import prompts
from app.agent.middleware import MemoryInjection
from app.config import Settings
from app.tools import datetime_tool


def test_datetime_tool_uses_configured_timezone(monkeypatch):
    """M-05: the emitted offset matches the configured zone, not a UTC label."""
    monkeypatch.setattr(
        datetime_tool, "get_settings", lambda: Settings(tz="Asia/Kolkata")
    )

    output = datetime_tool.current_datetime.invoke({})

    expected_offset = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%z")
    assert "+05:30" in output
    assert expected_offset[:3] + ":" + expected_offset[3:] in output
    assert "Asia/Kolkata" in output


def test_document_datetime_reports_configured_zone():
    output = datetime_tool.current_datetime.invoke({})
    assert "Current time:" in output
    assert "(" in output and ")" in output


def test_prompt_date_uses_configured_timezone():
    prompt = prompts.format_system_prompt("Asia/Kolkata")
    assert "timezone Asia/Kolkata" in prompt


def test_graph_prompt_body_has_no_frozen_date():
    """M-06: the startup prompt body must not embed a date."""
    assert "Today is" not in prompts.SYSTEM_PROMPT_BODY


class _FakeRequest:
    def __init__(self, system_message: SystemMessage) -> None:
        self.system_message = system_message
        self.state: dict = {}

    def override(self, **kwargs):
        clone = _FakeRequest(self.system_message)
        clone.state = self.state
        clone.system_message = kwargs.get("system_message", self.system_message)
        return clone


def test_middleware_resolves_date_on_every_call(monkeypatch):
    """M-06: a new call after midnight sees the new date without a restart."""
    request = _FakeRequest(SystemMessage(content="base prompt"))
    injection = MemoryInjection()

    monkeypatch.setattr(middleware_module, "build_date_context", lambda: "Today is 2026-10-30.")
    first = injection._inject(request)
    assert "Today is 2026-10-30." in first.system_message.content

    monkeypatch.setattr(middleware_module, "build_date_context", lambda: "Today is 2026-10-31.")
    second = injection._inject(request)
    assert "Today is 2026-10-31." in second.system_message.content
    assert "2026-10-30" not in second.system_message.content
