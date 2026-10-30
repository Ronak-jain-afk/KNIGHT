import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm.ggufbook import (
    DownloadManager,
    DownloadURLError,
    get_hf_model_variants,
    search_hf_gguf_models,
    validate_download_url,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ggufbook", tags=["ggufbook"])


class StartDownloadRequest(BaseModel):
    repo_id: str = Field(..., description="Hugging Face repository ID")
    filename: str = Field(..., description="Target .gguf filename")
    download_url: str = Field(..., description="Direct resolve download URL")
    size_bytes: int = Field(default=0, ge=0)


@router.get("/search")
async def search_models(
    q: str = Query(default="", description="Search query string"),
) -> list[dict[str, Any]]:
    """Search Hugging Face models for GGUF repositories."""
    return await search_hf_gguf_models(query=q)


@router.get("/models/{repo_id:path}")
async def get_model_details(
    repo_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Retrieve all GGUF quantization variants and hardware compatibility for a repository."""
    details = await get_hf_model_variants(repo_id=repo_id, settings=settings)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hugging Face repository '{repo_id}' not found or contains no GGUF models.",
        )
    return details


@router.post("/download", status_code=status.HTTP_202_ACCEPTED)
async def start_download(
    req: StartDownloadRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Start or enqueue a background download of a GGUF model."""
    target_name = req.filename.strip()
    if not target_name or not target_name.lower().endswith(".gguf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must end with '.gguf'",
        )

    # Prevent path traversal
    if Path(target_name).name != target_name or any(
        c in target_name for c in ["/", "\\", ".."]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal characters are forbidden.",
        )

    # Check if file already exists
    dest = settings.models_dir / target_name
    if dest.exists() and dest.is_file():
        return {
            "status": "already_exists",
            "message": f"Model '{target_name}' is already downloaded in models directory.",
            "filename": target_name,
        }

    # Validate the download URL before enqueuing: no SSRF, no non-HTTPS origins.
    try:
        cleaned_url = validate_download_url(req.download_url, settings)
    except DownloadURLError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid download_url: {exc}",
        ) from exc

    task = await DownloadManager.start_download(
        repo_id=req.repo_id.strip(),
        filename=target_name,
        download_url=cleaned_url,
        size_bytes=req.size_bytes,
        settings=settings,
    )
    return task.to_dict()


@router.get("/downloads")
async def list_downloads() -> list[dict[str, Any]]:
    """List active, queued, and completed GGUF downloads."""
    return DownloadManager.list_tasks()


@router.post("/downloads/{task_id}/cancel")
async def cancel_download(
    task_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Cancel an active or queued download."""
    cancelled = await DownloadManager.cancel_download(task_id, settings)
    if not cancelled:
        task = DownloadManager.get_task(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Download task '{task_id}' not found.",
            )
        return {"status": task.status, "message": f"Task cannot be cancelled (status: {task.status})"}
    return {"status": "cancelled", "task_id": task_id}
