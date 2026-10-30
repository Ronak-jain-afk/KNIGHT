import logging
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.client import get_chat_model
from app.memory.repository import get_memory_repository

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "Extract durable facts about the user from this exchange.\n"
    "Return ONLY facts that will still be true in a month.\n"
    "Skip small talk, one-off questions, and anything containing secrets."
)


class ExtractedMemory(BaseModel):
    text: str = Field(
        description="One self-contained sentence, written in third person ('User ...')"
    )
    kind: Literal["preference", "fact", "project", "event"] = Field(
        default="fact", description="Category of the memory"
    )
    importance: int = Field(
        default=3, ge=1, le=5, description="1 (trivial) to 5 (critical)"
    )


class ExtractedMemories(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


async def extract_memories_background(
    messages: list[BaseMessage],
    source_thread_id: str | None = None,
) -> int:
    """Background task to extract durable user facts and store them in MemoryRepository."""
    settings = get_settings()
    if not settings.memory_enabled or not settings.memory_auto_extract:
        return 0

    if not messages:
        return 0

    # Take last 6 messages
    recent = messages[-6:]
    formatted_lines: list[str] = []

    for msg in recent:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            # If tool calls were made, mention tool names or content
            formatted_lines.append(f"Assistant: {msg.content}")

    conversation_text = "\n\n".join(formatted_lines).strip()
    if len(conversation_text) > 4000:
        conversation_text = conversation_text[-4000:]

    if not conversation_text:
        return 0

    try:
        model = get_chat_model(settings, streaming=False)
        structured_model = model.with_structured_output(
            ExtractedMemories,
            method="json_schema",
        )

        response: ExtractedMemories = await structured_model.ainvoke(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": conversation_text},
            ]
        )

        if not response or not response.memories:
            return 0

        repo = get_memory_repository(settings)
        saved_count = 0
        for m in response.memories:
            res = await repo.add_memory(
                text=m.text,
                kind=m.kind,
                importance=m.importance,
                source_thread_id=source_thread_id,
            )
            if res in ("Saved.", "Already known."):
                saved_count += 1

        logger.info(
            "Auto-extracted %d memories from thread %s",
            saved_count,
            source_thread_id,
        )
        return saved_count
    except (RuntimeError, ValueError, OSError, TimeoutError, AttributeError, KeyError) as exc:
        logger.warning("Background memory extraction skipped: %s", exc)
        return 0
