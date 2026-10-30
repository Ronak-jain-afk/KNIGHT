from typing import Any

from langchain.agents import AgentState


class ChatState(AgentState, total=False):
    mode: str  # "agent" | "chat"
    memories: list[dict[str, Any]]  # injected per turn by the API layer
    system_prompt_override: str | None  # optional custom instructions per conversation
    model_customization: dict[str, Any] | None  # runtime model sampling overrides
