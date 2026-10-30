import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import conversations, sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

MAX_TITLE_LENGTH = 100


class ConversationCreate(BaseModel):
    title: str = Field(
        default="New Conversation",
        min_length=1,
        max_length=MAX_TITLE_LENGTH,
        description="Initial conversation title",
    )


class ConversationUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_TITLE_LENGTH,
        description="Updated title",
    )
    archived: bool | None = Field(
        default=None,
        description="Archive (true) or restore (false) the conversation",
    )


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict] = []


class ConversationForkRequest(BaseModel):
    message_id: str | None = Field(
        default=None,
        description="Message ID up to which to fork. If omitted, forks entire conversation.",
    )
    title: str | None = Field(
        default=None,
        max_length=MAX_TITLE_LENGTH,
        description="Optional custom title for the branch.",
    )


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    archived: bool
    pinned: bool = False
    turn_count: int
    parent_conversation_id: str | None = None
    forked_from_message_id: str | None = None
    messages: list[MessageItem] = []


class PinConversationRequest(BaseModel):
    pinned: bool = True


@router.get("", response_model=list[dict])
async def list_conversations_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    include_archived: bool = Query(default=False),
) -> list[dict]:
    return await conversations.list_conversations(
        settings.app_db_path,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_conversation_endpoint(
    payload: ConversationCreate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    new_id = str(uuid.uuid4())
    return await conversations.create_conversation(settings.app_db_path, new_id, payload.title)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_endpoint(
    conversation_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationDetail:
    meta = await conversations.get_conversation(settings.app_db_path, conversation_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Load thread sources
    all_sources = await sources.get_thread_sources(settings.app_db_path, conversation_id)
    sources_by_msg: dict[str, list[dict]] = {}
    for src in all_sources:
        msg_id = src.get("message_id")
        if msg_id:
            sources_by_msg.setdefault(msg_id, []).append(src)

    # Load messages from checkpoint state
    agent = getattr(request.app.state, "agent", None)
    rendered_messages: list[MessageItem] = []

    if agent:
        try:
            state = await agent.aget_state({"configurable": {"thread_id": conversation_id}})
            messages = state.values.get("messages", []) if state else []
            for msg in messages:
                role = "assistant"
                if hasattr(msg, "type"):
                    role = "user" if msg.type == "human" else ("assistant" if msg.type == "ai" else msg.type)
                content = getattr(msg, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                msg_id = getattr(msg, "id", f"msg_{uuid.uuid4().hex[:8]}")
                rendered_messages.append(
                    MessageItem(
                        id=msg_id,
                        role=role,
                        content=content,
                        sources=sources_by_msg.get(msg_id, []),
                    )
                )
        except (KeyError, ValueError, AttributeError, RuntimeError) as exc:
            logger.debug(
                "Could not load state for conversation %s: %s", conversation_id, exc
            )

    return ConversationDetail(
        id=meta["id"],
        title=meta["title"],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        archived=meta["archived"],
        pinned=meta["pinned"],
        turn_count=meta["turn_count"],
        parent_conversation_id=meta.get("parent_conversation_id"),
        forked_from_message_id=meta.get("forked_from_message_id"),
        messages=rendered_messages,
    )


@router.post(
    "/{conversation_id}/fork",
    response_model=ConversationDetail,
    status_code=status.HTTP_201_CREATED,
)
async def fork_conversation_endpoint(
    conversation_id: str,
    payload: ConversationForkRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationDetail:
    """Fork/branch conversation history up to a specific turn (Time Travel)."""
    meta = await conversations.get_conversation(settings.app_db_path, conversation_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    agent = getattr(request.app.state, "agent", None)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent engine not initialized.",
        )

    try:
        state = await agent.aget_state({"configurable": {"thread_id": conversation_id}})
    except (KeyError, ValueError, AttributeError, RuntimeError) as exc:
        logger.debug(
            "Could not load state for conversation %s: %s", conversation_id, exc
        )
        state = None

    source_messages = state.values.get("messages", []) if state else []

    # Slice messages up to message_id if specified
    if payload.message_id:
        target_idx = None
        for idx, m in enumerate(source_messages):
            if getattr(m, "id", None) == payload.message_id:
                target_idx = idx
                break
        if target_idx is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Message '{payload.message_id}' not found in conversation history.",
            )
        forked_messages = source_messages[: target_idx + 1]
    else:
        forked_messages = list(source_messages)

    # Determine title
    default_title = (
        f"{meta['title']} (Branch)"
        if not meta["title"].endswith("(Branch)")
        else f"{meta['title']} 2"
    )
    new_title = (
        payload.title.strip()
        if payload.title and payload.title.strip()
        else default_title
    )

    # Count turns in forked slice
    turn_count = sum(
        1
        for m in forked_messages
        if getattr(m, "type", "") == "human" or getattr(m, "role", "") == "user"
    )

    new_id = str(uuid.uuid4())
    new_meta = await conversations.create_conversation(
        settings.app_db_path,
        conversation_id=new_id,
        title=new_title,
        parent_conversation_id=conversation_id,
        forked_from_message_id=payload.message_id,
        turn_count=turn_count,
    )

    # Copy thread citations for matching messages
    message_ids = {
        getattr(m, "id", "") for m in forked_messages if getattr(m, "id", "")
    }
    await sources.copy_thread_sources(
        settings.app_db_path,
        source_thread_id=conversation_id,
        target_thread_id=new_id,
        message_ids=message_ids,
    )

    # Seed state in LangGraph checkpointer for the new thread
    if forked_messages:
        await agent.aupdate_state(
            {"configurable": {"thread_id": new_id}},
            {"messages": forked_messages},
        )

    # Load thread sources for rendering
    all_sources = await sources.get_thread_sources(settings.app_db_path, new_id)
    sources_by_msg: dict[str, list[dict]] = {}
    for src in all_sources:
        m_id = src.get("message_id")
        if m_id:
            sources_by_msg.setdefault(m_id, []).append(src)

    rendered_messages: list[MessageItem] = []
    for msg in forked_messages:
        role = "assistant"
        if hasattr(msg, "type"):
            role = (
                "user"
                if msg.type == "human"
                else ("assistant" if msg.type == "ai" else msg.type)
            )
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        msg_id = getattr(msg, "id", f"msg_{uuid.uuid4().hex[:8]}")
        rendered_messages.append(
            MessageItem(
                id=msg_id,
                role=role,
                content=content,
                sources=sources_by_msg.get(msg_id, []),
            )
        )

    return ConversationDetail(
        id=new_meta["id"],
        title=new_meta["title"],
        created_at=new_meta["created_at"],
        updated_at=new_meta["updated_at"],
        archived=new_meta["archived"],
        pinned=new_meta.get("pinned", False),
        turn_count=new_meta["turn_count"],
        parent_conversation_id=new_meta["parent_conversation_id"],
        forked_from_message_id=new_meta["forked_from_message_id"],
        messages=rendered_messages,
    )


@router.patch("/{conversation_id}", response_model=dict)
async def update_conversation_endpoint(
    conversation_id: str,
    payload: ConversationUpdate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    existing = await conversations.get_conversation(settings.app_db_path, conversation_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if payload.title is not None:
        success = await conversations.update_conversation_title(
            settings.app_db_path, conversation_id, payload.title.strip()
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

    if payload.archived is not None:
        success = await conversations.archive_conversation(
            settings.app_db_path, conversation_id, archived=payload.archived
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

    updated = await conversations.get_conversation(settings.app_db_path, conversation_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return {"status": "ok", **updated}


@router.delete("/{conversation_id}", response_model=dict)
async def delete_conversation_endpoint(
    conversation_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    # Delete checkpoints
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer and hasattr(checkpointer, "adelete_thread"):
        try:
            await checkpointer.adelete_thread(conversation_id)
        except (KeyError, ValueError, AttributeError, RuntimeError) as exc:
            logger.debug(
                "Could not load state for conversation %s: %s", conversation_id, exc
            )

    success = await conversations.delete_conversation(settings.app_db_path, conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return {"status": "deleted", "id": conversation_id}


@router.post("/{conversation_id}/pin", response_model=dict)
async def toggle_pin_conversation_endpoint(
    conversation_id: str,
    payload: PinConversationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    existing = await conversations.get_conversation(
        settings.app_db_path, conversation_id
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    success = await conversations.set_conversation_pinned(
        settings.app_db_path, conversation_id, pinned=payload.pinned
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update pinned state",
        )
    updated = await conversations.get_conversation(
        settings.app_db_path, conversation_id
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return updated
