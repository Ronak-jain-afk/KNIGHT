# SPDX-License-Identifier: Apache-2.0
import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Windows device names are reserved regardless of extension (H-21).
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class DocumentItem(BaseModel):
    name: str
    size_bytes: int
    char_count: int
    modified_at: str
    extension: str


class CreateDocumentRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., max_length=2_000_000)
    overwrite: bool = False


def sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters."""
    clean = Path(name).name  # Strips any directory components
    clean = _INVALID_FILENAME_CHARS.sub("_", clean)
    clean = re.sub(r"[^\w\s\.-]", "_", clean).strip()
    clean = clean.rstrip(". ")  # Windows drops trailing dots/spaces
    clean = re.sub(r"^\.+", "", clean)  # No hidden files (.env, etc.)
    if not clean:
        return "document.txt"

    stem, dot, extension = clean.partition(".")
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
        clean = stem + dot + extension
    return clean


async def read_capped_body(request: Request, max_bytes: int) -> bytes | None:
    """Read a request body in chunks, returning ``None`` once over the cap (H-19)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _scan_documents(docs_dir: Path) -> list[DocumentItem]:
    items: list[DocumentItem] = []
    try:
        entries = sorted(docs_dir.iterdir())
    except OSError:
        return []

    for p in entries:
        try:
            if not p.is_file() or p.name.startswith("."):
                continue
            stat = p.stat()
            mod_iso = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
            try:
                char_count = len(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                char_count = 0
            items.append(
                DocumentItem(
                    name=p.name,
                    size_bytes=stat.st_size,
                    char_count=char_count,
                    modified_at=mod_iso,
                    extension=p.suffix.lower(),
                )
            )
        except OSError:
            continue
    return items


@router.get("", response_model=list[DocumentItem])
async def list_documents(
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[DocumentItem]:
    docs_dir = settings.documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    return await asyncio.to_thread(_scan_documents, docs_dir)


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_document(
    req: CreateDocumentRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    safe_name = sanitize_filename(req.filename)
    docs_dir = settings.documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    target = (docs_dir / safe_name).resolve()
    if not target.is_relative_to(docs_dir.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path traversal attempt.",
        )

    if target.exists() and not req.overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{safe_name}' already exists.",
        )

    await asyncio.to_thread(target.write_text, req.content, encoding="utf-8")
    return {
        "status": "created",
        "name": safe_name,
        "path": f"documents/{safe_name}",
        "char_count": len(req.content),
    }


@router.post("/upload", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_document_raw(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    filename: str | None = Query(default=None),
    x_filename: str | None = Header(default=None, alias="X-Filename"),
    overwrite: bool = Query(default=False),
) -> dict:
    name_param = filename or x_filename or "uploaded_file.txt"
    safe_name = sanitize_filename(name_param)

    body = await read_capped_body(request, settings.max_attachment_bytes)
    if body is None:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "File size exceeds the "
                f"{settings.max_attachment_bytes / 1_000_000:g} MB limit."
            ),
        )

    docs_dir = settings.documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    target = (docs_dir / safe_name).resolve()
    if not target.is_relative_to(docs_dir.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path traversal attempt.",
        )

    if target.exists() and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{safe_name}' already exists.",
        )

    await asyncio.to_thread(target.write_bytes, body)
    return {
        "status": "uploaded",
        "name": safe_name,
        "path": f"documents/{safe_name}",
        "size_bytes": len(body),
    }


@router.delete("/{filename}", response_model=dict)
async def delete_document(
    filename: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    safe_name = sanitize_filename(filename)
    target = (settings.documents_dir / safe_name).resolve()

    if not target.is_relative_to(settings.documents_dir.resolve()) or not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found.",
        )

    target.unlink()
    return {"status": "deleted", "name": safe_name}
