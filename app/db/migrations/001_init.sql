PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,              -- UUIDv4, doubles as LangGraph thread_id
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,                 -- ISO-8601 UTC
    updated_at  TEXT NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0,
    turn_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);

-- Sources are attached to the assistant message that used them.
CREATE TABLE IF NOT EXISTS message_sources (
    thread_id   TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    idx         INTEGER NOT NULL,              -- 1-based citation number
    url         TEXT NOT NULL,
    title       TEXT,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (thread_id, message_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_sources_thread ON message_sources(thread_id);

CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    text              TEXT NOT NULL,
    kind              TEXT NOT NULL CHECK (kind IN ('preference','fact','project','event')),
    importance        INTEGER NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    embedding         BLOB,                    -- float32 little-endian, length = dims*4
    embedding_model   TEXT,
    content_hash      TEXT NOT NULL UNIQUE,    -- sha256(normalized text)
    source_thread_id  TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_used_at      TEXT,
    use_count         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

CREATE TABLE IF NOT EXISTS search_cache (
    key         TEXT PRIMARY KEY,              -- sha256(provider|query|max_results|freshness)
    payload     TEXT NOT NULL,                 -- JSON array of results
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS page_cache (
    url         TEXT PRIMARY KEY,              -- requested URL
    final_url   TEXT NOT NULL,
    title       TEXT,
    markdown    TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                 TEXT PRIMARY KEY,
    thread_id          TEXT NOT NULL,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL,          -- ok | error | cancelled
    error_code         TEXT,
    model              TEXT,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    tool_calls         INTEGER NOT NULL DEFAULT 0,
    elapsed_ms         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(thread_id, started_at DESC);
