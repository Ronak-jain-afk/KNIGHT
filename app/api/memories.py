from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import memories as db_memories
from app.memory.repository import MemoryRepository, get_memory_repo_dep

router = APIRouter(prefix="/api/memories", tags=["memories"])


class MemoryItemSchema(BaseModel):
    id: str
    text: str
    kind: str
    importance: int
    created_at: str
    updated_at: str
    last_used_at: str | None = None
    use_count: int = 0


class CreateMemoryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    kind: Literal["preference", "fact", "project", "event"] = "fact"
    importance: int = Field(default=3, ge=1, le=5)


class UpdateMemoryRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=1000)
    importance: int | None = Field(default=None, ge=1, le=5)


class ReindexResponse(BaseModel):
    reindexed_count: int


@router.get("", response_model=list[MemoryItemSchema])
async def list_memories(
    settings: Annotated[Settings, Depends(get_settings)],
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[MemoryItemSchema]:
    rows = await db_memories.list_memories(
        settings.app_db_path,
        kind=kind,
        limit=limit,
        offset=offset,
    )
    return [
        MemoryItemSchema(
            id=r["id"],
            text=r["text"],
            kind=r["kind"],
            importance=r["importance"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            last_used_at=r["last_used_at"],
            use_count=r["use_count"],
        )
        for r in rows
    ]


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_memory(
    req: CreateMemoryRequest,
    repo: Annotated[MemoryRepository, Depends(get_memory_repo_dep)],
) -> dict:
    result = await repo.add_memory(
        text=req.text,
        kind=req.kind,
        importance=req.importance,
    )
    if result.startswith("Skipped:"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result,
        )
    return {"status": result}


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_memories(
    repo: Annotated[MemoryRepository, Depends(get_memory_repo_dep)],
) -> ReindexResponse:
    count = await repo.reindex()
    return ReindexResponse(reindexed_count=count)


@router.delete("/{memory_id}", response_model=dict)
async def delete_memory(
    memory_id: str,
    repo: Annotated[MemoryRepository, Depends(get_memory_repo_dep)],
) -> dict:
    success = await repo.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found",
        )
    return {"ok": True}


@router.patch("/{memory_id}", response_model=dict)
async def update_memory(
    memory_id: str,
    req: UpdateMemoryRequest,
    repo: Annotated[MemoryRepository, Depends(get_memory_repo_dep)],
) -> dict:
    if req.text is None and req.importance is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field (text or importance) must be provided for update",
        )
    success = await repo.update_memory(
        memory_id=memory_id,
        text=req.text,
        importance=req.importance,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found or rejected",
        )
    return {"ok": True, "id": memory_id}
