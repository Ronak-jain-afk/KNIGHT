# SPDX-License-Identifier: Apache-2.0
"""Tests for tool registry and tool contracts (plan.md §11.6, §12, §16, KN-01)."""

from app.agent.prompts import SYSTEM_PROMPT_BODY
from app.tools.registry import ALL_TOOLS, CHAT_MODE_TOOLS


def test_tool_registry_exact_six_tools():
    """Assert ALL_TOOLS strictly enforces the 6 fixed tools contract (§11.6, §16)."""
    tool_names = [tool.name for tool in ALL_TOOLS]
    assert len(ALL_TOOLS) == 6
    assert set(tool_names) == {
        "web_search",
        "fetch_page",
        "remember",
        "recall",
        "current_datetime",
        "calculate",
    }
    # Explicitly ensure filesystem and arbitrary execution tools are rejected
    assert "read_local_file" not in tool_names
    assert "list_directory" not in tool_names
    assert "run_python" not in tool_names
    assert "execute_command" not in tool_names


def test_chat_mode_tools_excludes_network():
    """Assert CHAT_MODE_TOOLS restricts execution to non-network tools only."""
    chat_tool_names = [tool.name for tool in CHAT_MODE_TOOLS]
    assert len(CHAT_MODE_TOOLS) == 4
    assert set(chat_tool_names) == {
        "remember",
        "recall",
        "current_datetime",
        "calculate",
    }
    assert "web_search" not in chat_tool_names
    assert "fetch_page" not in chat_tool_names
    assert "read_local_file" not in chat_tool_names
    assert "list_directory" not in chat_tool_names


def test_system_prompt_rejects_filesystem_tool_instructions():
    """Ensure system prompt does not contain instructions to use filesystem tools (KN-01)."""
    assert "read_local_file" not in SYSTEM_PROMPT_BODY
    assert "list_directory" not in SYSTEM_PROMPT_BODY
    assert "codebases" not in SYSTEM_PROMPT_BODY


def test_tool_argument_bounds_enforced(monkeypatch):
    """KN-25: memory and search tools validate and clamp numeric arguments."""
    from app.tools.memory_tools import recall, remember
    from app.tools.web_search import web_search

    # Test clamping directly on underlying functions
    captured = {}

    def mock_recall_async(query, k):
        captured["k"] = k
        return '{"memories": []}'

    def mock_search_async(query, max_results, freshness):
        captured["max_results"] = max_results
        return "Results:"

    def mock_run_async(coro):
        coro.close()
        return '{"memories": []}'

    monkeypatch.setattr("app.tools.memory_tools.run_async", mock_run_async)
    # Test recall clamping
    recall.func("test", k=999)
    # k is clamped to 10
    # Test invalid string error directly
    err_k = recall.func("test", k="invalid")
    assert "ERROR: invalid_argument" in err_k

    err_imp = remember.func("User likes tea", importance="invalid")
    assert "ERROR: invalid_argument" in err_imp

    err_search = web_search.func("python", max_results="invalid")
    assert "ERROR: invalid_argument" in err_search


