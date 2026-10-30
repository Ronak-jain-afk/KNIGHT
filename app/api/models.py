import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm.manager import ModelManager, list_gguf_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelInfo(BaseModel):
    name: str
    stem: str
    size_bytes: int
    size_formatted: str
    modified_at: str
    is_active: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    active_model: str


class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., description="GGUF model filename in models directory")


class SwitchModelResponse(BaseModel):
    ok: bool
    model_name: str
    message: str


@router.get("", response_model=ModelsResponse)
async def get_models(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelsResponse:
    """List available GGUF model files in the models directory."""
    active_model = getattr(request.app.state, "active_model", None) or ModelManager.get_active_model(
        settings
    )
    raw_models = list_gguf_models(settings, active_model=active_model)
    models = [ModelInfo(**m) for m in raw_models]
    return ModelsResponse(
        models=models,
        active_model=active_model,
    )


@router.post("/switch", response_model=SwitchModelResponse)
async def switch_model(
    req: SwitchModelRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Any:
    """Hot-swap the active GGUF inference model."""
    target_name = req.model_name.strip()
    if not target_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_name cannot be empty.",
        )

    # Prevent path traversal
    if Path(target_name).name != target_name or any(c in target_name for c in ["/", "\\", ".."]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model filename; path traversal characters are forbidden.",
        )

    res = await ModelManager.switch_model(
        settings=settings,
        model_filename=target_name,
        app=request.app,
    )

    if not res.get("ok"):
        error_code = res.get("error", "")
        status_code = (
            status.HTTP_409_CONFLICT
            if error_code == "model_swapping"
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=status_code,
            detail=res.get("detail") or res.get("error") or "Failed to switch model.",
        )

    return SwitchModelResponse(
        ok=True,
        model_name=res["model_name"],
        message=res.get("message", "Model switched successfully."),
    )
