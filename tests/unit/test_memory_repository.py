from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import app.memory.repository as repository_module
from app.config import Settings
from app.db import memories as db_memories
from app.db.sqlite import run_migrations
from app.memory.repository import MemoryRepository, looks_like_secret


def test_looks_like_secret():
    assert looks_like_secret("My key is sk-abcdef1234567890abcdef")
    assert looks_like_secret("Token: ghp_1234567890abcdefghij1234567890abcdef")
    assert looks_like_secret("AWS key: AKIAIOSFODNN7EXAMPLE")
    assert looks_like_secret("-----BEGIN PRIVATE KEY-----")
    assert looks_like_secret("User password is 123")
    assert looks_like_secret("hash 0123456789abcdef0123456789abcdef")
    assert not looks_like_secret("User prefers dark mode.")
    assert not looks_like_secret("User's timezone is America/New_York.")


def test_secret_heuristic_does_not_flag_prose():
    """H-15: anchored token shapes must not reject ordinary hyphenated words."""
    assert not looks_like_secret("The user prefers task-based workflows.")
    assert not looks_like_secret("The user is risk-averse.")
    assert not looks_like_secret("Discuss the disk-based cache.")
    assert looks_like_secret("sk-" + "a" * 40)
    assert looks_like_secret("ghp_" + "a" * 36)
    assert looks_like_secret("AKIA" + "A" * 16)


def test_single_content_hash_implementation():
    """H-18: write and read paths share one normalization + hash function."""
    assert not hasattr(repository_module, "compute_content_hash")
    assert not hasattr(repository_module, "normalize_memory_text")
    assert db_memories.hash_memory_text("  User likes tea!  ") == db_memories.hash_memory_text(
        "user likes tea"
    )


@pytest.mark.asyncio
async def test_memory_secret_rejection(tmp_path):
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    repo = MemoryRepository(settings)
    await repo.initialize()

    res = await repo.add_memory("User secret API key is sk-1234567890abcdef123456")
    assert res == "Skipped: looks like a secret."
    assert len(repo._items) == 0


@pytest.mark.asyncio
async def test_memory_exact_hash_dedup(tmp_path):
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    repo = MemoryRepository(settings)
    await repo.initialize()

    res1 = await repo.add_memory("User lives in Tokyo.", kind="fact", importance=2)
    assert res1 == "Saved."
    assert len(repo._items) == 1

    # Exact duplicate with different casing/punctuation
    res2 = await repo.add_memory("  user lives in tokyo!  ", kind="fact", importance=4)
    assert res2 == "Already known."
    assert len(repo._items) == 1
    # Importance bumped to higher value
    item = next(iter(repo._items.values()))
    assert item.importance == 4


@pytest.mark.asyncio
async def test_memory_semantic_dedup(tmp_path):
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    mock_embeddings = MagicMock()
    mock_embeddings.is_available.return_value = True
    mock_embeddings.initialize = AsyncMock(return_value=True)

    # Return very close vectors (cosine > 0.95)
    vec1 = np.array([1.0, 0.0, 0.0], dtype="<f4")
    vec2 = np.array([0.98, 0.05, 0.0], dtype="<f4")

    mock_embeddings.embed_text = AsyncMock(side_effect=[vec1, vec2])

    repo = MemoryRepository(settings, embedding_service=mock_embeddings)
    await repo.initialize()

    res1 = await repo.add_memory("User is a Python developer.", kind="fact", importance=3)
    assert res1 == "Saved."

    res2 = await repo.add_memory("User develops software with Python.", kind="fact", importance=4)
    assert res2 == "Already known."
    assert len(repo._items) == 1
    # Merged text takes longer string and higher importance
    item = next(iter(repo._items.values()))
    assert item.text == "User develops software with Python."
    assert item.importance == 4


@pytest.mark.asyncio
async def test_memory_cosine_ranking_and_importance_5(tmp_path):
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    mock_embeddings = MagicMock()
    mock_embeddings.is_available.return_value = True
    mock_embeddings.initialize = AsyncMock(return_value=True)

    # Three memories:
    # 1. Close to query (dot ~ 1.0)
    # 2. Somewhat close (dot ~ 0.6)
    # 3. Unrelated (dot ~ 0.1), but importance 5 and old
    q_vec = np.array([1.0, 0.0, 0.0], dtype="<f4")
    vec_close = np.array([0.95, 0.1, 0.0], dtype="<f4")
    vec_mid = np.array([0.6, 0.4, 0.0], dtype="<f4")
    vec_far = np.array([0.1, 0.9, 0.0], dtype="<f4")

    mock_embeddings.embed_text = AsyncMock(
        side_effect=[
            vec_close,
            vec_mid,
            vec_far,
            q_vec,  # query embedding
        ]
    )

    repo = MemoryRepository(settings, embedding_service=mock_embeddings)
    await repo.initialize()

    await repo.add_memory("User lives in Berlin.", kind="fact", importance=3)
    await repo.add_memory("User speaks German.", kind="fact", importance=2)
    await repo.add_memory("User has severe peanut allergy.", kind="fact", importance=5)

    # Manually backdate the allergy memory last_used_at to 10 days ago
    for item in repo._items.values():
        if "peanut" in item.text:
            ten_days_ago = datetime.now(UTC) - timedelta(days=10)
            item.last_used_at = ten_days_ago.isoformat()

    results = await repo.search("Where does user reside?", top_k=2)

    # The close memory must be retrieved
    texts = [r["text"] for r in results]
    assert "User lives in Berlin." in texts
    # Critical importance 5 must also be included even if low similarity
    assert "User has severe peanut allergy." in texts


@pytest.mark.asyncio
async def test_memory_delete(tmp_path):
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    repo = MemoryRepository(settings)
    await repo.initialize()

    await repo.add_memory("User likes tea.", kind="preference", importance=3)
    assert len(repo._items) == 1

    item_id = next(iter(repo._items.keys()))
    deleted = await repo.delete_memory(item_id)
    assert deleted is True
    assert len(repo._items) == 0

    results = await repo.search("tea")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_keyword_fallback_honours_min_score(tmp_path):
    """H-16: weak lexical matches are excluded instead of reported as strong."""
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    repo = MemoryRepository(settings)
    await repo.initialize()
    assert repo.embedding_service.is_available() is False

    await repo.add_memory("User likes tea in the morning.", kind="preference", importance=2)
    await repo.add_memory("User likes coffee after lunch.", kind="preference", importance=2)

    exact = await repo.search("tea")
    assert [r["text"] for r in exact] == ["User likes tea in the morning."]
    assert exact[0]["score"] == 1.0
    assert exact[0]["score_kind"] == "lexical"

    # One of three query words matches, i.e. 0.33 < default min_score 0.35.
    weak = await repo.search("tea coffee unrelated")
    assert weak == []


@pytest.mark.asyncio
async def test_text_update_without_embedding_clears_stale_vector(tmp_path):
    """H-17: an updated text must not keep the old vector for cosine search."""
    settings = Settings(data_dir=tmp_path)
    await run_migrations(settings.app_db_path)

    mock_embeddings = MagicMock()
    mock_embeddings.is_available.return_value = True
    mock_embeddings.initialize = AsyncMock(return_value=True)
    mock_embeddings.embed_text = AsyncMock(
        return_value=np.array([1.0, 0.0, 0.0], dtype="<f4")
    )

    repo = MemoryRepository(settings, embedding_service=mock_embeddings)
    await repo.initialize()

    await repo.add_memory("User is a Python developer.", kind="fact", importance=3)
    memory_id = next(iter(repo._items))

    mock_embeddings.is_available.return_value = False
    updated = await repo.update_memory(memory_id, text="User is a Rust developer.")
    assert updated is True

    item = repo._items[memory_id]
    assert item.text == "User is a Rust developer."
    assert item.embedding is None
    assert memory_id not in repo._matrix_ids

    row = await db_memories.get_memory(settings.app_db_path, memory_id)
    assert row is not None
    assert row["embedding"] is None
    assert row["embedding_model"] is None

