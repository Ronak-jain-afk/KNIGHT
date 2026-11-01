# SPDX-License-Identifier: Apache-2.0
import ipaddress
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host; loopback only")
    port: int = Field(default=8000, description="Server port")
    data_dir: Path = Field(default=Path("./data"), description="Path to data directory")

    # Llama server settings
    llama_base_url: str = Field(
        default="http://127.0.0.1:8080/v1", description="llama-server OpenAI-compatible base URL"
    )
    llama_api_key: str = Field(
        default="",
        description="llama-server bearer API key (must be set explicitly; no guessable fallback)",
    )
    model_name: str = Field(default="qwen3.5-4b", description="Model name label")
    context_tokens: int = Field(default=16384, description="Context tokens (must match server -c)")
    max_output_tokens: int = Field(default=1024, description="Max generation output tokens")
    temperature: float = Field(default=0.7, description="Generation temperature")
    top_p: float = Field(default=0.8, description="Top-p sampling")
    top_k: int = Field(default=20, description="Top-k sampling")
    presence_penalty: float = Field(default=1.5, description="Presence penalty")
    enable_thinking: bool = Field(default=False, description="Enable thinking mode")
    tz: str = Field(
        default="",
        description="IANA timezone name (e.g. Asia/Kolkata). Empty uses the host timezone.",
    )

    # Request size limits shared with the web client via /api/config.
    max_message_chars: int = Field(
        default=32_000, description="Maximum user message / composed prompt characters"
    )
    max_attachment_bytes: int = Field(
        default=2_000_000, description="Maximum size of a single attached file (bytes)"
    )

    # Agent & summarization settings
    summary_trigger_tokens: int = Field(default=9000, description="Summary trigger token threshold")
    summary_keep_messages: int = Field(
        default=8, description="Number of recent messages to keep unsummarized"
    )
    max_tool_calls_per_turn: int = Field(
        default=8, description="Loop guard maximum tool calls per turn"
    )

    # Search & fetch settings
    search_providers: str = Field(
        default="ddgs,searxng,brave", description="Ordered comma-separated search providers"
    )
    searxng_url: str = Field(default="", description="Optional SearXNG base URL")
    brave_api_key: str = Field(default="", description="Optional Brave Search API key")
    search_cache_ttl_hours: int = Field(default=24, description="Search cache TTL in hours")
    page_cache_ttl_hours: int = Field(default=168, description="Page cache TTL in hours")
    cache_purge_interval_seconds: int = Field(
        default=3600, description="How often expired search/page cache rows are purged"
    )
    cache_max_rows: int = Field(
        default=10_000, description="Hard cap on rows per search/page cache table"
    )
    fetch_max_bytes: int = Field(default=2000000, description="Max raw bytes per page fetch")
    fetch_timeout_seconds: float = Field(default=12.0, description="Fetch HTTP timeout seconds")
    fetch_max_chars: int = Field(
        default=6000, description="Max characters of markdown passed to model"
    )

    # Model download hardening
    model_download_max_bytes: int = Field(
        default=20 * 1024**3,
        description="Hard cap for a single GGUF download (bytes)",
    )
    model_download_hosts: str = Field(
        default="huggingface.co",
        description="Comma-separated allowlist of hosts for GGUF downloads "
        "(suffixes starting with '.' are treated as wildcards)",
    )

    # Memory settings
    memory_enabled: bool = Field(default=True, description="Enable long-term memory")
    memory_top_k: int = Field(default=6, description="Top k memories to retrieve")
    memory_min_score: float = Field(default=0.35, description="Cosine similarity score floor")
    memory_auto_extract: bool = Field(
        default=True, description="Background post-turn memory extraction"
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Fastembed model name"
    )
    embedding_dims: int = Field(default=384, description="Embedding dimension")

    # ASR settings - Whistle model (16.9 MB) by Cactus Compute
    asr_enabled: bool = Field(default=True, description="Enable on-device ASR voice prompting")
    asr_model_repo: str = Field(
        default="Cactus-Compute/whistle", description="Hugging Face repo for Whistle ASR"
    )
    asr_language: str = Field(
        default="", description="Default ASR language code or empty for auto-detect"
    )
    asr_max_audio_seconds: int = Field(
        default=120, description="Maximum voice clip length in seconds"
    )

    # Security & operational settings
    auth_token: str = Field(default="", description="Optional bearer token for /api/* calls")
    log_level: str = Field(default="INFO", description="Logging level")
    log_prompts: bool = Field(default=False, description="Whether to log full model prompts")
    dev: bool = Field(default=False, description="Dev mode flag (enables /api/shutdown)")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        value = (v or "").strip()
        if not value:
            raise ValueError(
                "KNIGHT_HOST must be 127.0.0.1 or ::1; Knight is strictly "
                "local-first/loopback only."
            )

        if value.lower() == "localhost":
            return value

        try:
            address = ipaddress.ip_address(value.strip("[]"))
        except ValueError as exc:
            raise ValueError(
                f"KNIGHT_HOST='{value}' is not a valid IP address; Knight is strictly "
                "local-first/loopback only."
            ) from exc

        if not address.is_loopback:
            raise ValueError(
                f"KNIGHT_HOST='{value}' is not a loopback address; Knight is strictly "
                "local-first/loopback only (use 127.0.0.1 or ::1)."
            )
        return value

    @property
    def parsed_search_providers(self) -> list[str]:
        return [p.strip() for p in self.search_providers.split(",") if p.strip()]

    @property
    def parsed_download_hosts(self) -> list[str]:
        return [h.strip().lower() for h in self.model_download_hosts.split(",") if h.strip()]

    @property
    def app_db_path(self) -> Path:
        return self.data_dir / "knight.sqlite"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def llama_api_key_file(self) -> Path:
        """Key file written by the runner so the smoke test can authenticate."""
        return self.data_dir / "llama_api_key"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def asr_dir(self) -> Path:
        return self.models_dir / "whistle"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.asr_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
