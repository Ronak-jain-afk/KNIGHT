"""Per-turn request context that must never be checkpointed.

LangGraph persists graph *inputs* into thread state. Values that describe a single
turn (retrieved memories, a persona override, sampling overrides) therefore leaked
into later turns: turn 2 inherited turn 1's memories and instructions even when the
client sent none.

These context variables are set around each ``agent.astream`` call instead. A sentinel
distinguishes "not set in this process context" (direct graph use, tests) from an
explicit empty value (a real turn with no memories/override).
"""

from __future__ import annotations

import contextvars
from typing import Any

UNSET: Any = object()

current_memories: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_memories", default=UNSET
)
current_system_prompt_override: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_system_prompt_override", default=UNSET
)
current_model_customization: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_model_customization", default=UNSET
)


def _state_value(state: Any, key: str) -> Any:
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def resolve_memories(state: Any) -> list[dict[str, Any]]:
    value = current_memories.get()
    if value is UNSET:
        value = _state_value(state, "memories")
    return value or []


def resolve_system_prompt_override(state: Any) -> str | None:
    value = current_system_prompt_override.get()
    if value is UNSET:
        value = _state_value(state, "system_prompt_override")
    if not value:
        return None
    return str(value)


def resolve_model_customization(state: Any) -> dict[str, Any] | None:
    value = current_model_customization.get()
    if value is UNSET:
        value = _state_value(state, "model_customization")
    return value if isinstance(value, dict) and value else None
