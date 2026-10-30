-- Knight Persona & System Prompt Profiles
CREATE TABLE IF NOT EXISTS profiles (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    icon               TEXT NOT NULL DEFAULT '',
    description        TEXT NOT NULL DEFAULT '',
    system_prompt      TEXT NOT NULL DEFAULT '',
    mode               TEXT NOT NULL DEFAULT 'agent' CHECK (mode IN ('agent', 'chat')),
    is_builtin         INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profiles_updated ON profiles(updated_at DESC);

-- Seed default built-in profiles
INSERT OR IGNORE INTO profiles (id, name, icon, description, system_prompt, mode, is_builtin, created_at, updated_at)
VALUES
(
    'general',
    'General Assistant',
    '',
    'Balanced, helpful, concise, well-reasoned assistant.',
    'You are a helpful, concise, and balanced AI assistant. Answer clearly and directly without unnecessary preamble.',
    'agent',
    1,
    '2026-10-29T00:00:00Z',
    '2026-10-29T00:00:00Z'
),
(
    'software-engineer',
    'Software Engineer',
    '',
    'Rigorous, clean code, type annotations, unit tests, modern best practices.',
    'You are an expert senior software engineer. Write clean, idiomatic, type-annotated code. Prioritize correctness, robustness, and performance. Include unit tests and edge case coverage where relevant. Avoid fluff.',
    'agent',
    1,
    '2026-10-29T00:00:00Z',
    '2026-10-29T00:00:00Z'
),
(
    'devops-terminal',
    'Terminal & DevOps',
    '',
    'Concise shell commands, flags explained, security warnings, cross-platform.',
    'You are a pragmatic Linux and Windows systems administrator and DevOps engineer. Provide copy-paste ready, robust terminal commands and scripts. Concisely explain flag semantics and caution against dangerous operations.',
    'agent',
    1,
    '2026-10-29T00:00:00Z',
    '2026-10-29T00:00:00Z'
),
(
    'research-analyst',
    'Research Analyst',
    '',
    'Exhaustive web search, rigorous source citations, structured synthesis.',
    'You are an objective, thorough research analyst. Prioritize verifiable factual data, synthesize diverse perspectives, and structure findings logically with clear citations and references.',
    'agent',
    1,
    '2026-10-29T00:00:00Z',
    '2026-10-29T00:00:00Z'
);
