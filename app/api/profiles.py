# SPDX-License-Identifier: Apache-2.0
import logging
import re
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import profiles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


class ProfileItem(BaseModel):
    id: str
    name: str
    icon: str
    description: str
    system_prompt: str
    mode: Literal["agent", "chat"]
    is_builtin: bool
    created_at: str
    updated_at: str


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=60, description="Display name for profile")
    icon: str = Field(default="✍️", max_length=8, description="Emoji or symbol icon")
    description: str = Field(default="", max_length=200, description="Short summary of profile role")
    system_prompt: str = Field(..., min_length=1, max_length=4000, description="Persona instructions")
    mode: Literal["agent", "chat"] = Field(default="agent", description="Preferred interaction mode")


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    icon: str | None = Field(default=None, max_length=8)
    description: str | None = Field(default=None, max_length=200)
    system_prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    mode: Literal["agent", "chat"] | None = None


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "custom-profile"


@router.get("", response_model=list[ProfileItem])
async def list_profiles_endpoint(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[ProfileItem]:
    raw_profiles = await profiles.list_profiles(settings.app_db_path)
    return [ProfileItem(**p) for p in raw_profiles]


@router.get("/{profile_id}", response_model=ProfileItem)
async def get_profile_endpoint(
    profile_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProfileItem:
    p = await profiles.get_profile(settings.app_db_path, profile_id)
    if not p:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found.",
        )
    return ProfileItem(**p)


@router.post("", response_model=ProfileItem, status_code=status.HTTP_201_CREATED)
async def create_profile_endpoint(
    req: ProfileCreate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProfileItem:
    base_slug = slugify(req.name)
    profile_id = f"custom-{base_slug}-{uuid.uuid4().hex[:6]}"

    created = await profiles.create_profile(
        db_path=settings.app_db_path,
        profile_id=profile_id,
        name=req.name.strip(),
        icon=req.icon.strip() or "✍️",
        description=req.description.strip(),
        system_prompt=req.system_prompt.strip(),
        mode=req.mode,
    )
    return ProfileItem(**created)


@router.patch("/{profile_id}", response_model=ProfileItem)
async def update_profile_endpoint(
    profile_id: str,
    req: ProfileUpdate,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProfileItem:
    existing = await profiles.get_profile(settings.app_db_path, profile_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found.",
        )

    fields_set = req.model_fields_set
    updated = await profiles.update_profile(
        db_path=settings.app_db_path,
        profile_id=profile_id,
        name=req.name.strip() if "name" in fields_set and req.name is not None else None,
        icon=req.icon.strip() if "icon" in fields_set and req.icon is not None else None,
        description=req.description.strip() if "description" in fields_set and req.description is not None else None,
        system_prompt=req.system_prompt.strip() if "system_prompt" in fields_set and req.system_prompt is not None else None,
        mode=req.mode if "mode" in fields_set else None,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update profile.",
        )

    refreshed = await profiles.get_profile(settings.app_db_path, profile_id)
    return ProfileItem(**refreshed)  # type: ignore[arg-type]


@router.delete("/{profile_id}", response_model=dict)
async def delete_profile_endpoint(
    profile_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    existing = await profiles.get_profile(settings.app_db_path, profile_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found.",
        )
    if existing["is_builtin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Built-in system profiles cannot be deleted.",
        )

    deleted = await profiles.delete_profile(settings.app_db_path, profile_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete profile.",
        )
    return {"status": "deleted", "id": profile_id}
