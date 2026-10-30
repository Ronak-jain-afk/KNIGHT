import logging

import httpx
from langchain_openai import ChatOpenAI

from app.api.schemas import LlamaHealth
from app.config import Settings

logger = logging.getLogger(__name__)


def get_chat_model(settings: Settings, *, streaming: bool = True) -> ChatOpenAI:
    """Build a ChatOpenAI client configured for local llama-server."""
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_base=settings.llama_base_url,
        openai_api_key=settings.llama_api_key,
        streaming=streaming,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_output_tokens,
        extra_body={
            "top_k": settings.top_k,
            "presence_penalty": settings.presence_penalty,
            "chat_template_kwargs": {
                "enable_thinking": settings.enable_thinking,
            },
        },
    )


async def check_llama_health(settings: Settings) -> LlamaHealth:
    """Check if the local llama-server is running and query basic props."""
    server_root = settings.llama_base_url.rstrip("/").removesuffix("/v1")

    headers = {}
    if settings.llama_api_key:
        headers["Authorization"] = f"Bearer {settings.llama_api_key}"

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{server_root}/health", headers=headers)
            if resp.status_code not in (200, 503):
                return LlamaHealth(reachable=False)

            health_data = resp.json() if resp.status_code == 200 else {}
            slots_busy = health_data.get("slots_processing")

            model_name: str | None = None
            n_ctx: int | None = None

            try:
                props_resp = await client.get(f"{server_root}/props", headers=headers)
                if props_resp.status_code == 200:
                    props_data = props_resp.json()
                    gen_settings = props_data.get("default_generation_settings", {})
                    n_ctx = gen_settings.get("n_ctx")
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug("Could not retrieve props from llama-server: %s", exc)

            try:
                models_resp = await client.get(f"{server_root}/v1/models", headers=headers)
                if models_resp.status_code == 200:
                    models_data = models_resp.json()
                    data = models_data.get("data", [])
                    if data:
                        model_name = data[0].get("id")
            except (httpx.HTTPError, ValueError) as exc:
                logger.debug("Could not retrieve models list from llama-server: %s", exc)

            return LlamaHealth(
                reachable=True,
                model=model_name or settings.model_name,
                n_ctx=n_ctx or settings.context_tokens,
                slots_busy=slots_busy or 0,
            )
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return LlamaHealth(reachable=False)
    except httpx.HTTPError as exc:
        logger.debug("HTTP error while checking llama-server health: %s", exc)
        return LlamaHealth(reachable=False)
