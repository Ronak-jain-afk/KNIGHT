import re
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.streaming import stream_agent_run
from app.api.schemas import ModelCustomization
from app.config import Settings, get_settings
from app.db import conversations

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Existing thread ID or null")
    message: str = Field(..., description="User message content")
    mode: Literal["agent", "chat"] | None = Field(
        default=None,
        description="Execution mode (defaults to the selected profile's mode, then 'agent')",
    )
    profile_id: str | None = Field(
        default=None,
        description="Optional Knight profile / persona preset identifier",
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Optional custom system prompt or persona instructions",
        max_length=4000,
    )
    model_customization: ModelCustomization | None = Field(
        default=None,
        description="Optional runtime model sampling and feature overrides",
    )


def derive_title(text: str) -> str:
    """Derive clean conversation title from first 48 chars of message."""
    clean = re.sub(r"\s+", " ", text.strip())
    title = clean[:48].strip()
    # Trim trailing punctuation
    title = title.rstrip(".,;:!?-")
    return title or "New Conversation"


@router.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )
    if len(user_msg) > settings.max_message_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Message exceeds maximum length of "
                f"{settings.max_message_chars:,} characters."
            ),
        )

    thread_id = req.conversation_id
    if not thread_id:
        thread_id = str(uuid.uuid4())
        title = derive_title(user_msg)
        await conversations.create_conversation(settings.app_db_path, thread_id, title)
    else:
        existing = await conversations.get_conversation(settings.app_db_path, thread_id)
        if not existing:
            title = derive_title(user_msg)
            await conversations.create_conversation(settings.app_db_path, thread_id, title)

    # Increment conversation turn count
    await conversations.increment_turn_count(settings.app_db_path, thread_id)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    agent = getattr(request.app.state, "agent", None)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent engine not initialized.",
        )

    # Long-term memories to be injected
    injected_memories: list[dict] = []
    if settings.memory_enabled:
        from app.memory.repository import get_memory_repository

        repo = get_memory_repository(settings)
        injected_memories = await repo.search(
            query=user_msg,
            top_k=settings.memory_top_k,
            min_score=settings.memory_min_score,
        )

    # Resolve profile persona and mode
    final_system_prompt = req.system_prompt_override
    profile_mode: str | None = None

    if req.profile_id:
        from app.db import profiles

        prof = await profiles.get_profile(settings.app_db_path, req.profile_id)
        if prof:
            profile_mode = prof.get("mode")
            profile_prompt = prof.get("system_prompt", "").strip()
            if profile_prompt:
                if final_system_prompt and final_system_prompt.strip():
                    final_system_prompt = (
                        f"{profile_prompt}\n\n{final_system_prompt.strip()}"
                    )
                else:
                    final_system_prompt = profile_prompt

    # An explicit client mode wins; otherwise honour the profile's preference.
    final_mode = req.mode or profile_mode or "agent"

    custom_model_dict = (
        req.model_customization.model_dump(exclude_none=True)
        if req.model_customization
        else None
    )

    return StreamingResponse(
        stream_agent_run(
            agent=agent,
            settings=settings,
            thread_id=thread_id,
            run_id=run_id,
            user_message=user_msg,
            mode=final_mode,
            memories=injected_memories,
            http_request=request,
            system_prompt_override=final_system_prompt,
            model_customization=custom_model_dict,
            model_name=getattr(request.app.state, "active_model", None),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
