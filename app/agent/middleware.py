import contextvars
import json
import math
from collections.abc import Callable, Iterable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import SystemMessage, ToolMessage

from app.agent.prompts import build_date_context, render_memories_block
from app.agent.run_context import (
    resolve_memories,
    resolve_model_customization,
    resolve_system_prompt_override,
)
from app.tools.registry import CHAT_MODE_TOOLS

# Per-turn loop guard tracking context variable
loop_guard_state: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "loop_guard_state", default=None
)


def approx_token_counter(messages: Iterable[Any]) -> int:
    """Fast approximation: ceil(len(text)/3.6) + 4 per message."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text_len = len(content)
        elif isinstance(content, list):
            text_len = sum(len(str(c)) for c in content)
        else:
            text_len = len(str(content))
        total += math.ceil(text_len / 3.6) + 4
    return total


def describe_tool_error(error: Exception, *args: Any, **kwargs: Any) -> str:
    """Convert unexpected tool exceptions into clean error ToolMessages."""
    tool_name = "tool"
    if args:
        request = args[0]
        if hasattr(request, "tool") and request.tool:
            tool_name = getattr(request.tool, "name", "tool")
        elif hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "tool")
    return f"ERROR: {tool_name}_failure: {error}"


class ModelCustomizationMiddleware(AgentMiddleware):
    """Binds dynamic runtime model parameters (temperature, top_p, top_k, max_tokens, penalties, thinking) per request."""

    def _apply_customization(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        state = request.state
        custom = resolve_model_customization(state)
        if not custom:
            return request

        bind_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if k in custom and custom[k] is not None:
                bind_kwargs[k] = custom[k]

        if "max_output_tokens" in custom and custom["max_output_tokens"] is not None:
            bind_kwargs["max_tokens"] = custom["max_output_tokens"]

        if "top_k" in custom and custom["top_k"] is not None:
            extra_body["top_k"] = custom["top_k"]

        if "enable_thinking" in custom and custom["enable_thinking"] is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(custom["enable_thinking"])}

        if extra_body:
            bind_kwargs["extra_body"] = extra_body

        if bind_kwargs and hasattr(request.model, "bind"):
            bound_model = request.model.bind(**bind_kwargs)
            return request.override(model=bound_model)

        return request

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        request = self._apply_customization(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        request = self._apply_customization(request)
        return await handler(request)


class MemoryInjection(AgentMiddleware):
    """Injects the current date and known memories into the system message.

    The date is resolved on every model call so a server that stays up past
    midnight does not keep advertising its launch date.
    """

    def _inject(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        state = request.state
        memories = resolve_memories(state)
        mem_block = render_memories_block(memories)

        sys_msg = request.system_message
        if sys_msg is None:
            return request

        date_block = build_date_context()
        new_content = f"{date_block}\n\n{sys_msg.content}\n\n{mem_block}"
        override = resolve_system_prompt_override(state)
        if override:
            new_content = f"{new_content}\n\nCUSTOM USER INSTRUCTIONS\n{override.strip()}"
        new_sys_msg = SystemMessage(content=new_content)
        return request.override(system_message=new_sys_msg)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._inject(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        return await handler(self._inject(request))


class ToolPolicy(AgentMiddleware):
    """Restricts tools to non-web tools when state.mode == 'chat'."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        state = request.state
        mode = state.get("mode", "agent") if isinstance(state, dict) else getattr(state, "mode", "agent")
        if mode == "chat":
            request = request.override(tools=list(CHAT_MODE_TOOLS))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        state = request.state
        mode = state.get("mode", "agent") if isinstance(state, dict) else getattr(state, "mode", "agent")
        if mode == "chat":
            request = request.override(tools=list(CHAT_MODE_TOOLS))
        return await handler(request)


class TruncateToolOutput(AgentMiddleware):
    """Truncates tool output text exceeding fetch_max_chars."""

    def __init__(self, max_chars: int = 6000) -> None:
        super().__init__()
        self.max_chars = max_chars

    def _truncate(self, message: Any) -> Any:
        if (
            isinstance(message, ToolMessage)
            and isinstance(message.content, str)
            and len(message.content) > self.max_chars
        ):
            message.content = message.content[: self.max_chars] + "\n…[truncated]"
        return message

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        result = handler(request)
        return self._truncate(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        result = await handler(request)
        return self._truncate(result)


class LoopGuard(AgentMiddleware):
    """Prevents repeated identical tool calls and bounds tool calls per turn."""

    def __init__(self, max_calls_per_turn: int = 8) -> None:
        super().__init__()
        self.max_calls_per_turn = max_calls_per_turn

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        state = loop_guard_state.get()
        call_count = state.get("count", 0) if state else 0
        if call_count >= self.max_calls_per_turn:
            request = request.override(tools=[])
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Any],
    ) -> Any:
        state = loop_guard_state.get()
        call_count = state.get("count", 0) if state else 0
        if call_count >= self.max_calls_per_turn:
            request = request.override(tools=[])
        return await handler(request)

    def _check_and_record(self, request: ToolCallRequest) -> ToolMessage | None:
        tc = request.tool_call
        fn_name = tc.get("name", "")
        args_str = json.dumps(tc.get("args", {}), sort_keys=True)
        key = (fn_name, args_str)

        current = loop_guard_state.get()
        if not current:
            current = {"seen": set(), "count": 0}
            loop_guard_state.set(current)

        if key in current["seen"]:
            return ToolMessage(
                content="ERROR: already called with these arguments; use the earlier result",
                tool_call_id=tc.get("id", ""),
                status="error",
            )

        current["seen"].add(key)
        current["count"] += 1
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        early_err = self._check_and_record(request)
        if early_err is not None:
            return early_err
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        early_err = self._check_and_record(request)
        if early_err is not None:
            return early_err
        return await handler(request)
