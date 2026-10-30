import json
from typing import Literal

from langchain_core.tools import tool

from app.memory.repository import get_memory_repository
from app.utils.async_bridge import run_async


async def _remember_async(
    text: str,
    kind: Literal["preference", "fact", "project", "event"] = "fact",
    importance: int = 3,
) -> str:
    repo = get_memory_repository()
    return await repo.add_memory(text=text, kind=kind, importance=importance)


async def _recall_async(query: str, k: int = 5) -> str:
    repo = get_memory_repository()
    results = await repo.search(query=query, top_k=k)
    clean_results = [
        {"text": r["text"], "kind": r["kind"], "score": r["score"]}
        for r in results
    ]
    return json.dumps({"memories": clean_results}, ensure_ascii=False)


@tool
def remember(
    text: str,
    kind: Literal["preference", "fact", "project", "event"] = "fact",
    importance: int = 3,
) -> str:
    """Save a durable fact or preference about the user for future conversations.

    Args:
        text: One self-contained sentence, written in third person ("User ...").
        kind: Category of the memory (preference, fact, project, event).
        importance: 1 (trivial) to 5 (critical).
    Returns:
        "Saved." | "Already known." | "Skipped: looks like a secret."
    """
    try:
        return run_async(_remember_async(text, kind, importance))
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        return f"ERROR: remember_failed: {exc}"


@tool
def recall(query: str, k: int = 5) -> str:
    """Search long-term memory for facts relevant to a query.

    Args:
        query: Query terms to search stored memories for.
        k: Maximum number of memories to return (1-10).

    Returns:
        JSON: {"memories":[{"text":...,"kind":...,"score":0.82}]} or {"memories":[]}
    """
    try:
        return run_async(_recall_async(query, k))
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        return f"ERROR: recall_failed: {exc}"
