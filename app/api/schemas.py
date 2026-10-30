from pydantic import BaseModel, Field


class LlamaHealth(BaseModel):
    reachable: bool = Field(..., description="Whether llama-server is reachable")
    model: str | None = Field(default=None, description="Model identifier if reachable")
    n_ctx: int | None = Field(default=None, description="Context window size if reachable")
    slots_busy: int | None = Field(default=None, description="Count of active busy slots if reachable")


class HealthResponse(BaseModel):
    app: str = Field(default="ok", description="FastAPI app status")
    llama: LlamaHealth = Field(..., description="Llama server health details")
    db: str = Field(default="ok", description="App database status")
    memory: str = Field(default="ok", description="Memory system status (ok, degraded, disabled)")


class ConfigResponse(BaseModel):
    model_name: str
    context_tokens: int
    enable_thinking: bool
    search_providers: list[str]
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_output_tokens: int = 1024
    presence_penalty: float = 1.5
    max_message_chars: int = 32_000
    max_attachment_bytes: int = 2_000_000


class ModelCustomization(BaseModel):
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    top_k: int | None = Field(
        default=None, ge=1, le=200, description="Top-k token candidate cutoff"
    )
    max_output_tokens: int | None = Field(
        default=None, ge=64, le=8192, description="Max generated output tokens"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    enable_thinking: bool | None = Field(
        default=None, description="Enable thinking mode in chat template"
    )
