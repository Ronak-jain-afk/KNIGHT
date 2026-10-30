# Knight: Local-First AI Agent

> **KNIGHT**: **K**nowledge **N**avigation, **I**ntelligence, **G**uidance & **H**elpful **T**ools

Knight is a single-user, local-first artificial intelligence assistant that runs entirely on your local machine. It combines a FastAPI backend, a zero-build vanilla web interface, and an accelerated llama.cpp server for local open-weights language models.

The system requires no external cloud subscriptions, sends no telemetry, and operates strictly over loopback interfaces.

---

## Architecture Overview

```
+-------------------------------------------------------------------+
|               Browser Web UI (HTML5, CSS, Vanilla JS)             |
|              Zero build step, zero CDNs, vendored assets          |
+---------------------------------+---------------------------------+
                                  | HTTP / Server-Sent Events (SSE)
                                  v
+-------------------------------------------------------------------+
|                    FastAPI Backend (127.0.0.1:8000)               |
|                                                                   |
|  Security Middleware:                                             |
|    - Loopback host validation (127.0.0.1 / ::1)                   |
|    - Content-Security-Policy and HTTP security headers            |
|    - Origin and Sec-Fetch-Site cross-origin protection            |
|    - Optional Bearer token authentication                         |
|    - Sensitive credential log redaction                           |
|                                                                   |
|  Agent Orchestration (LangGraph):                                 |
|    - State management via create_agent                            |
|    - LoopGuard recursion and cycle prevention                     |
|    - Message window summarization middleware                      |
|    - SourceCollector citation tracing [1], [2]                    |
|    - Async bridge for thread-safe synchronous tool calls          |
|                                                                   |
|  Integrated Tools:                                                |
|    - web_search (DuckDuckGo, SearXNG, Brave)                      |
|    - fetch_page (NetGuard SSRF defense, HTML quarantine)          |
|    - calculate (AST-based safe mathematical evaluator)            |
|    - current_datetime (Timezone-aware clock)                      |
|    - remember / recall (Semantic memory store)                    |
|                                                                   |
|  Storage Layer:                                                   |
|    - data/knight.sqlite (conversations, memories, caches, runs)   |
|    - data/checkpoints.sqlite (LangGraph thread checkpoints)       |
+-------------------+-------------------------------+---------------+
                    |                               |
                    v                               v
+-----------------------------------+   +---------------------------+
|    llama-server (127.0.0.1:8080)  |   | Local Audio & Embedding:  |
|    llama.cpp pinned build b11980  |   |  - FastEmbed (ONNX, 384d) |
|    OpenAI-compatible /v1/chat     |   |  - Whistle ASR (16.9 MB)  |
+-----------------------------------+   +---------------------------+
```

---

## Key Features

### 1. Interactive Streaming Chat Interface
- Real-time response streaming via Server-Sent Events (SSE).
- Markdown rendering with vendored Marked.js and strict DOMPurify sanitization.
- Code blocks with syntax badges, one-click copying, and file download support.
- Numbered citation tags [1], [2] linked to verified source records.
- Stop generation control with clean async generator cancellation.
- Configurable audio sound effects for interaction events.

### 2. Conversation Management
- Grouped chronological history (Today, Yesterday, Previous 7 Days, Older).
- Conversation pinning for quick access.
- Inline double-click title editing.
- Fork conversation from any point in history.
- Full export to JSON or Markdown formats.
- Safe conversation deletion with custom confirmation modals.

### 3. Agentic Tool Execution
- Multi-Provider Web Search: Supports DuckDuckGo, SearXNG, and Brave Search with automated fallback, SQLite caching, and deduplication.
- Secure Page Fetching: Extracts clean text and markdown from web pages. Wrapped in untrusted content quarantine tags. Protected by NetGuard SSRF defense.
- AST Calculator: Safe arithmetic evaluation using Python Abstract Syntax Trees. Enforces recursion depth and exponent magnitude limits.
- System Timezone Integration: Dynamically injects the current host date and IANA timezone per execution turn.

### 4. Long-Term Durable Memory
- On-device embedding extraction using FastEmbed (BAAI/bge-small-en-v1.5, 384 dimensions).
- In-memory numpy cosine similarity ranking over SQLite storage.
- Lexical keyword fallback when embedding models are not present.
- Automated background memory extraction following completed turns.
- Secret scrubbing prevents private keys, tokens, or passwords from entering long-term storage.
- Dedicated Memory Drawer to review, search, add, or delete memories manually.

### 5. On-Device Speech-to-Text (ASR)
- Integrated Whistle model by Cactus Compute (16.9 MB lightweight on-device speech-to-text model).
- Audio visualizer tray and microphone recording in the chat composer.
- Client-side 16 kHz mono WAV encoding.
- Optional automatic prompt submission after speech completion.

### 6. GGUFBook Model Explorer
- Search and browse open-weight GGUF models directly from Hugging Face.
- Automated hardware fit evaluation (Full GPU, Tight VRAM, Partial CPU, Insufficient RAM) based on local system telemetry.
- Direct background downloading with SHA-256 integrity verification, safe path handling, and cancellation guards.
- Runtime model switching without restarting backend services.

### 7. Developer and Power-User Utilities
- Slash command menu (/help, /clear, /search, /remember, /model, /status).
- In-chat search to quickly locate terms in active conversations.
- Sampling presets (Creative, Balanced, Precise, Fast, Custom) with real-time sliders for temperature, top_p, top_k, presence penalty, frequency penalty, and reasoning mode toggle.
- Hardware HUD displaying live GPU VRAM, Host RAM, and compute device telemetry.

---

## Security Model

Knight is engineered from the ground up to protect user privacy and local machine integrity:

1. Loopback-Only Binding: The application binds strictly to 127.0.0.1 or ::1. Binding to 0.0.0.0 is explicitly blocked at the configuration level.
2. NetGuard SSRF Defense: Validates URLs across every redirect hop (up to 3 hops). Resolves DNS addresses and blocks private subnets, loopback addresses, link-local addresses, and cloud metadata endpoints.
3. Content Security Policy: Strict CSP headers and HTML meta tags prevent remote script execution (default-src 'self'; script-src 'self'; connect-src 'self').
4. Origin Verification: Rejects cross-origin and cross-site HTTP requests using Origin and Sec-Fetch-Site headers.
5. Bearer Authentication: Supports an optional KNIGHT_AUTH_TOKEN. When configured, all API endpoints require authentication headers, transmitted automatically by the web UI.
6. Zero External CDNs: Marked.js and DOMPurify are vendored locally in app/web/vendor/ with integrity verified against a SHA-256 manifest on startup.
7. Log Redaction: A centralized logging filter scrubs API keys, auth tokens, and sensitive headers from terminal and file logs.

---

## System Requirements

- Operating System: Windows 11 (Primary) or Linux (Ubuntu, Debian, WSL2).
- Python Version: Python >=3.12, <3.14.
- Package Manager: Astral uv.
- GPU (Optional): NVIDIA GPU with CUDA support for accelerated inference, or CPU with AVX2.
- Disk Space: 2 GB for dependencies and binaries, plus storage for downloaded GGUF models.

---

## Installation & Setup

### 1. Clone the Repository

```bash
cd knight
```

### 2. Install Astral uv

If you do not have uv installed:
- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Linux / macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 3. Sync Dependencies

Install Python dependencies with pinned versions using uv:

```bash
uv sync --extra dev
```

### 4. Download llama.cpp Server

Download the pinned binary (build b11980) matching your operating system and hardware:

```bash
uv run python scripts/download_llama.py --tag b11980
```

### 5. Download Language and Embedding Models

Download a recommended GGUF model:

```bash
uv run python scripts/download_model.py
```

Download FastEmbed model weights for semantic memory:

```bash
uv run python scripts/download_fastembed.py
```

---

## Running the Application

### Full Stack Startup (Recommended)

Run the automated orchestrator that launches both the llama.cpp server and the FastAPI backend:

#### Windows 11 (PowerShell)
```powershell
scripts\start.ps1
```
Or directly with Python:
```powershell
uv run python scripts\run_all.py
```

#### Linux / WSL2
```bash
./scripts/start.sh
```
Or directly with Python:
```bash
uv run python scripts/run_all.py
```

Once running, open your web browser at:
**http://127.0.0.1:8000**

### Standalone API Server

If you are running an external llama-server instance:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Testing & Quality Assurance

Knight includes an automated test suite covering unit contracts, integration flows, and security constraints.

### Run Offline Test Suite
Executes all unit and integration tests without requiring an active llama-server:

```bash
uv run pytest -m "not live"
```

Verify that all tests pass without runtime warnings:

```bash
uv run pytest -m "not live" -W error::RuntimeWarning
```

### Run Live Inference Smoke Test
Tests end-to-end tool calling against a running llama-server:

```bash
uv run python scripts/smoke_test.py
```

### Run Code Linter
Verifies compliance with Ruff formatting and code quality rules:

```bash
uv run ruff check .
```

---

## Configuration Reference

All settings are configured through environment variables prefixed with `KNIGHT_` or defined in a local `.env` file.

| Variable | Type | Default | Description |
|---|---|---|---|
| `KNIGHT_HOST` | string | `127.0.0.1` | Server bind host. Must be a loopback address. |
| `KNIGHT_PORT` | integer | `8000` | Port for the FastAPI server. |
| `KNIGHT_DATA_DIR` | path | `./data` | Directory for SQLite databases and models. |
| `KNIGHT_LLAMA_BASE_URL` | string | `http://127.0.0.1:8080/v1` | URL for the OpenAI-compatible llama-server. |
| `KNIGHT_LLAMA_API_KEY` | string | `""` | Bearer key for llama-server. |
| `KNIGHT_MODEL_NAME` | string | `qwen3.5-4b` | Active model identifier. |
| `KNIGHT_CONTEXT_TOKENS` | integer | `16384` | Context window size. Must match server `-c`. |
| `KNIGHT_MAX_OUTPUT_TOKENS` | integer | `1024` | Maximum tokens per completion response. |
| `KNIGHT_TEMPERATURE` | float | `0.7` | Sampling temperature. |
| `KNIGHT_TOP_P` | float | `0.8` | Top-p nucleus sampling. |
| `KNIGHT_TOP_K` | integer | `20` | Top-k sampling limit. |
| `KNIGHT_PRESENCE_PENALTY` | float | `1.5` | Repetition presence penalty. |
| `KNIGHT_ENABLE_THINKING` | boolean | `false` | Enable or disable model reasoning tags. |
| `KNIGHT_TZ` | string | `""` | IANA timezone name. Defaults to host clock. |
| `KNIGHT_SUMMARY_TRIGGER_TOKENS` | integer | `9000` | Token threshold to trigger history compaction. |
| `KNIGHT_SUMMARY_KEEP_MESSAGES` | integer | `8` | Number of recent messages preserved raw. |
| `KNIGHT_MAX_TOOL_CALLS_PER_TURN` | integer | `8` | Maximum tool calls before loop termination. |
| `KNIGHT_SEARCH_PROVIDERS` | string | `ddgs,searxng,brave` | Comma-separated search provider order. |
| `KNIGHT_SEARXNG_URL` | string | `""` | Optional custom SearXNG instance URL. |
| `KNIGHT_BRAVE_API_KEY` | string | `""` | Optional Brave Search API key. |
| `KNIGHT_FETCH_MAX_BYTES` | integer | `2000000` | Maximum raw download bytes per page fetch. |
| `KNIGHT_FETCH_MAX_CHARS` | integer | `6000` | Character limit passed to model after parsing. |
| `KNIGHT_MEMORY_ENABLED` | boolean | `true` | Enable or disable semantic memory store. |
| `KNIGHT_MEMORY_TOP_K` | integer | `6` | Maximum memory facts retrieved per turn. |
| `KNIGHT_MEMORY_MIN_SCORE` | float | `0.35` | Minimum cosine similarity score threshold. |
| `KNIGHT_ASR_ENABLED` | boolean | `true` | Enable or disable speech-to-text recording. |
| `KNIGHT_ASR_LANGUAGE` | string | `""` | Language code for ASR (empty for auto-detect). |
| `KNIGHT_AUTH_TOKEN` | string | `""` | Optional bearer token protecting /api/* routes. |
| `KNIGHT_LOG_LEVEL` | string | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR). |
| `KNIGHT_LOG_PROMPTS` | boolean | `false` | Log prompt payloads with secret redaction. |

---

## Directory Structure

```
knight/
|-- app/
|   |-- agent/             # LangGraph agent, middleware, prompts, and streaming
|   |-- api/               # FastAPI route controllers and schemas
|   |-- asr/               # On-device Whistle speech recognition service
|   |-- db/                # SQLite connection pool and repository helpers
|   |-- llm/               # Model manager, hardware telemetry, and GGUFBook
|   |-- memory/            # FastEmbed ONNX embedding and cosine ranking
|   |-- search/            # DuckDuckGo, SearXNG, and Brave search providers
|   |-- security/          # NetGuard SSRF defense, middleware, and vendor guard
|   |-- tools/             # Agent tool definitions (search, fetch, calc, memory)
|   |-- utils/             # Async bridge and cross-platform process management
|   |-- web/               # Vanilla HTML5, CSS, and JavaScript interface
|   |-- config.py          # Pydantic Settings and environment validation
|   |-- logging_setup.py   # Redacting log filter
|   `-- main.py            # FastAPI application factory and lifespan hooks
|-- data/                  # SQLite storage, checkpoints, and model files
|-- scripts/               # Server runners, downloaders, and smoke test
|-- tests/
|   |-- integration/       # End-to-end API, UI, and streaming tests
|   `-- unit/              # Isolated tool, database, and security tests
|-- pyproject.toml         # Project metadata and dependencies
`-- uv.lock                # Deterministic dependency lockfile
```
