import aiosqlite
import pytest

from app.db import cache, conversations, memories, runs, sources
from app.db.sqlite import get_db_connection, init_checkpoint_db, run_migrations


@pytest.mark.asyncio
async def test_migrations_idempotent(tmp_path):
    db_path = tmp_path / "test_knight.sqlite"

    # First migration run
    await run_migrations(db_path)

    async with get_db_connection(db_path) as db:
        c = await db.execute("PRAGMA user_version;")
        v1 = (await c.fetchone())[0]
        assert v1 == 4

        # Check table creation
        c2 = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations';"
        )
        assert (await c2.fetchone()) is not None

        # Check profiles table
        cp = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='profiles';"
        )
        assert (await cp.fetchone()) is not None

        # Check parent_conversation_id column
        c3 = await db.execute("PRAGMA table_info(conversations);")
        cols = [r["name"] for r in await c3.fetchall()]
        assert "parent_conversation_id" in cols
        assert "forked_from_message_id" in cols
        assert "pinned" in cols

    # Second migration run (idempotency check)
    await run_migrations(db_path)

    async with get_db_connection(db_path) as db:
        c = await db.execute("PRAGMA user_version;")
        v2 = (await c.fetchone())[0]
        assert v2 == 4


@pytest.mark.asyncio
async def test_wal_journal_mode(tmp_path):
    app_db = tmp_path / "knight.sqlite"
    cp_db = tmp_path / "checkpoints.sqlite"

    await run_migrations(app_db)
    await init_checkpoint_db(cp_db)

    # Check app_db
    async with aiosqlite.connect(str(app_db)) as db:
        c = await db.execute("PRAGMA journal_mode;")
        mode = (await c.fetchone())[0]
        assert mode.lower() == "wal"

    # Check cp_db
    async with aiosqlite.connect(str(cp_db)) as db:
        c = await db.execute("PRAGMA journal_mode;")
        mode = (await c.fetchone())[0]
        assert mode.lower() == "wal"


@pytest.mark.asyncio
async def test_conversations_crud(tmp_path):
    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    conv_id = "test-conv-123"
    created = await conversations.create_conversation(db_path, conv_id, "Initial Title")
    assert created["id"] == conv_id
    assert created["title"] == "Initial Title"
    assert created["turn_count"] == 0

    fetched = await conversations.get_conversation(db_path, conv_id)
    assert fetched is not None
    assert fetched["title"] == "Initial Title"

    updated = await conversations.update_conversation_title(db_path, conv_id, "Renamed Title")
    assert updated is True

    inc = await conversations.increment_turn_count(db_path, conv_id)
    assert inc is True

    fetched2 = await conversations.get_conversation(db_path, conv_id)
    assert fetched2["title"] == "Renamed Title"
    assert fetched2["turn_count"] == 1

    archived = await conversations.archive_conversation(db_path, conv_id, True)
    assert archived is True

    active_list = await conversations.list_conversations(db_path, include_archived=False)
    assert not any(c["id"] == conv_id for c in active_list)

    all_list = await conversations.list_conversations(db_path, include_archived=True)
    assert any(c["id"] == conv_id for c in all_list)

    deleted = await conversations.delete_conversation(db_path, conv_id)
    assert deleted is True

    assert await conversations.get_conversation(db_path, conv_id) is None


@pytest.mark.asyncio
async def test_sources_crud(tmp_path):
    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    thread_id = "thread-1"
    msg_id = "msg-1"
    test_sources = [
        {"idx": 1, "url": "https://example.com/1", "title": "First Page"},
        {"idx": 2, "url": "https://example.com/2", "title": "Second Page"},
    ]

    await sources.save_message_sources(db_path, thread_id, msg_id, test_sources)

    msg_srcs = await sources.get_message_sources(db_path, thread_id, msg_id)
    assert len(msg_srcs) == 2
    assert msg_srcs[0]["idx"] == 1
    assert msg_srcs[0]["url"] == "https://example.com/1"
    assert msg_srcs[1]["title"] == "Second Page"

    thread_srcs = await sources.get_thread_sources(db_path, thread_id)
    assert len(thread_srcs) == 2

    # Test copy_thread_sources for branching
    target_thread = "thread-forked-branch"
    await sources.copy_thread_sources(
        db_path,
        source_thread_id=thread_id,
        target_thread_id=target_thread,
        message_ids={msg_id},
    )
    forked_srcs = await sources.get_thread_sources(db_path, target_thread)
    assert len(forked_srcs) == 2
    assert forked_srcs[0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_runs_crud(tmp_path):
    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    run_id = "run-xyz"
    thread_id = "thread-xyz"

    await runs.start_run(db_path, run_id, thread_id, "qwen3.5-4b")
    r1 = await runs.get_run(db_path, run_id)
    assert r1 is not None
    assert r1["status"] == "running"

    finished = await runs.finish_run(
        db_path,
        run_id,
        "ok",
        prompt_tokens=150,
        completion_tokens=42,
        tool_calls=1,
        elapsed_ms=850,
    )
    assert finished is True

    r2 = await runs.get_run(db_path, run_id)
    assert r2["status"] == "ok"
    assert r2["prompt_tokens"] == 150
    assert r2["completion_tokens"] == 42
    assert r2["tool_calls"] == 1
    assert r2["elapsed_ms"] == 850


@pytest.mark.asyncio
async def test_cache_crud(tmp_path):
    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    # Search cache
    s_key = "hash_search_1"
    s_payload = [{"title": "Test", "url": "http://test.com"}]
    await cache.set_search_cache(db_path, s_key, s_payload, ttl_hours=1)
    retrieved_s = await cache.get_search_cache(db_path, s_key)
    assert retrieved_s == s_payload

    # Page cache
    p_url = "https://example.com/article"
    await cache.set_page_cache(
        db_path,
        p_url,
        final_url="https://example.com/article",
        title="Article Title",
        markdown="# Content",
        ttl_hours=1,
    )
    retrieved_p = await cache.get_page_cache(db_path, p_url)
    assert retrieved_p is not None
    assert retrieved_p["title"] == "Article Title"
    assert retrieved_p["markdown"] == "# Content"

    # Purge (no expired items yet)
    del_s, del_p = await cache.purge_expired_cache(db_path)
    assert del_s == 0
    assert del_p == 0


@pytest.mark.asyncio
async def test_memories_crud(tmp_path):
    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    mem_id = "mem-1"
    inserted = await memories.insert_memory(
        db_path,
        mem_id,
        text="User prefers Python over JavaScript",
        kind="preference",
        importance=4,
    )
    assert inserted["id"] == mem_id
    assert inserted["importance"] == 4

    fetched = await memories.get_memory(db_path, mem_id)
    assert fetched is not None
    assert fetched["text"] == "User prefers Python over JavaScript"
    assert fetched["use_count"] == 0

    await memories.record_memories_usage(db_path, [mem_id])

    fetched2 = await memories.get_memory(db_path, mem_id)
    assert fetched2["use_count"] == 1

    listed = await memories.list_memories(db_path, kind="preference")
    assert len(listed) == 1
    assert listed[0]["id"] == mem_id

    empty_listed = await memories.list_memories(db_path, kind="event")
    assert len(empty_listed) == 0

    deleted = await memories.delete_memory(db_path, mem_id)
    assert deleted is True
    assert await memories.get_memory(db_path, mem_id) is None


@pytest.mark.asyncio
async def test_profiles_crud(tmp_path):
    from app.db import profiles

    db_path = tmp_path / "knight.sqlite"
    await run_migrations(db_path)

    # 1. Built-in profiles seeded
    profs = await profiles.list_profiles(db_path)
    assert len(profs) >= 4
    ids = [p["id"] for p in profs]
    assert "general" in ids
    assert "software-engineer" in ids
    assert "devops-terminal" in ids
    assert "research-analyst" in ids

    # Built-in cannot be deleted
    assert await profiles.delete_profile(db_path, "general") is False

    # 2. Create custom profile
    custom = await profiles.create_profile(
        db_path,
        profile_id="custom-test",
        name="Rust Specialist",
        icon="🦀",
        description="Write blazing fast memory safe Rust",
        system_prompt="Always write idiomatic Rust using cargo clippy standards.",
        mode="agent",
    )
    assert custom["id"] == "custom-test"
    assert custom["is_builtin"] is False

    # 3. Read profile
    p = await profiles.get_profile(db_path, "custom-test")
    assert p is not None
    assert p["name"] == "Rust Specialist"
    assert p["icon"] == "🦀"

    # 4. Update profile
    updated = await profiles.update_profile(
        db_path,
        profile_id="custom-test",
        name="Rust & Cargo Specialist",
        mode="chat",
    )
    assert updated is True
    p2 = await profiles.get_profile(db_path, "custom-test")
    assert p2["name"] == "Rust & Cargo Specialist"
    assert p2["mode"] == "chat"

    # 5. Delete custom profile
    deleted = await profiles.delete_profile(db_path, "custom-test")
    assert deleted is True
    assert await profiles.get_profile(db_path, "custom-test") is None
