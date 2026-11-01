# SPDX-License-Identifier: Apache-2.0
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolErrorMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from app.agent.middleware import (
    LoopGuard,
    MemoryInjection,
    ModelCustomizationMiddleware,
    ToolPolicy,
    TruncateToolOutput,
    approx_token_counter,
    describe_tool_error,
)
from app.agent.prompts import SYSTEM_PROMPT_BODY
from app.agent.state import ChatState
from app.config import Settings
from app.tools.registry import ALL_TOOLS


def build_agent(
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None,
    streaming_model: BaseChatModel,
    blocking_model: BaseChatModel,
    tools: list[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble create_agent with custom middleware, checkpointer, and static system prompt."""
    return create_agent(
        model=streaming_model,
        tools=tools if tools is not None else ALL_TOOLS,
        state_schema=ChatState,
        system_prompt=SYSTEM_PROMPT_BODY,
        middleware=[
            ModelCustomizationMiddleware(),
            MemoryInjection(),
            ToolPolicy(),
            TruncateToolOutput(settings.fetch_max_chars),
            LoopGuard(settings.max_tool_calls_per_turn),
            ToolErrorMiddleware(on_error=describe_tool_error),
            SummarizationMiddleware(
                model=blocking_model,
                trigger=("tokens", settings.summary_trigger_tokens),
                keep=("messages", settings.summary_keep_messages),
                token_counter=approx_token_counter,
            ),
        ],
        checkpointer=checkpointer,
    )
