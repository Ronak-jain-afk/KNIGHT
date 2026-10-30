import asyncio
import logging
import re
from urllib.parse import urljoin

import httpx
import trafilatura
from langchain_core.tools import tool

from app.agent.sources import current_sources, normalize_url
from app.config import get_settings
from app.db import cache
from app.security.net_guard import (
    BlockedAddressError,
    InvalidURLError,
    NetGuard,
    pinned_transport,
)
from app.security.ratelimit import fetch_limiter, get_outbound_semaphore
from app.utils.async_bridge import run_async

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Knight/0.1 (https://github.com/ronak/knight; contact: knight@local)"

MIN_MAX_CHARS = 500
RATE_LIMIT_MAX_WAIT_SECONDS = 5.0


def extract_title(html: str, fallback_url: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    og_match = re.search(
        r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\'](.*?)["\']',
        html,
        re.IGNORECASE,
    )
    if og_match:
        return og_match.group(1).strip()
    return fallback_url


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clamp_max_chars(value: int, configured_max: int) -> int:
    """Keep the model-supplied ``max_chars`` within the configured output cap (H-05)."""
    return min(max(int(value), MIN_MAX_CHARS), configured_max)


async def read_capped_body(response: httpx.Response, max_bytes: int) -> bytes | None:
    """Consume at most ``max_bytes`` from a streaming response (H-03).

    Returns ``None`` as soon as the limit is exceeded so the caller can abort
    without buffering an unbounded body.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def extract_markdown(html_text: str, final_url: str) -> str:
    extracted = trafilatura.extract(
        html_text,
        url=final_url,
        output_format="markdown",
        include_links=True,
        include_tables=True,
        favor_recall=True,
    )
    if not extracted or len(extracted) < 200:
        fallback = trafilatura.html2txt(html_text)
        extracted = fallback or "(No readable text content found on page)"
    return collapse_blank_lines(extracted)


@tool
def fetch_page(url: str, max_chars: int = 6000) -> str:
    """Download a public web page and return its main article text as Markdown.
    Do NOT use this tool for local files or local filesystem paths; only for public http/https URLs.

    Args:
        url: Absolute http(s) URL.
        max_chars: Maximum characters to return (500-12000).

    Returns:
        "[n] TITLE\\nURL: ...\\n\\n<untrusted_web_content>\\n...markdown...\\n</untrusted_web_content>"
        or "ERROR: <reason>".
    """
    try:
        return run_async(_fetch_page_async(url, max_chars))
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        return f"ERROR: fetch_failed: {exc}"


async def _fetch_page_async(url: str, max_chars: int = 6000) -> str:
    settings = get_settings()
    clean_url = normalize_url(url)
    max_chars = clamp_max_chars(max_chars, settings.fetch_max_chars)

    # 1. Page cache lookup
    cached = await cache.get_page_cache(settings.app_db_path, clean_url)
    if cached:
        collector = current_sources.get()
        idx = collector.add(cached["final_url"], cached["title"]) if collector else 1
        return (
            f"[{idx}] {cached['title']}\nURL: {cached['final_url']}\n\n"
            f"<untrusted_web_content>\n{cached['markdown'][:max_chars]}\n</untrusted_web_content>"
        )

    # Enforce (not just log) the outbound fetch rate limit (H-07).
    if not await fetch_limiter.wait(max_wait_seconds=RATE_LIMIT_MAX_WAIT_SECONDS):
        return "ERROR: rate_limited: too many page fetches, try again shortly"

    # 2. SSRF guard, validated IP pinning, and manual redirect handling
    current_url = clean_url
    client_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain",
    }
    hops = 0
    while hops < 3:
        hops += 1
        try:
            hostname, _port, pinned_ip = NetGuard.resolve_url(current_url)
        except (BlockedAddressError, InvalidURLError) as exc:
            return f"ERROR: blocked_address: {exc}"

        transport = pinned_transport(hostname, pinned_ip)
        try:
            async with (
                get_outbound_semaphore(),
                httpx.AsyncClient(
                    transport=transport,
                    follow_redirects=False,
                    timeout=settings.fetch_timeout_seconds,
                    headers=client_headers,
                    trust_env=False,
                ) as client,
                client.stream("GET", current_url) as resp,
            ):
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        return "ERROR: redirect_missing_location"
                    current_url = urljoin(current_url, location)
                    continue

                if not resp.is_success:
                    return f"ERROR: http_error: {resp.status_code}"

                # 4. Content-Type check
                c_type = resp.headers.get("content-type", "").lower()
                if not any(t in c_type for t in ALLOWED_CONTENT_TYPES):
                    return f"ERROR: unsupported_content_type: {c_type}"

                # 5. Check the declared size, then stream-cap the body.
                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        declared_bytes = int(content_length)
                    except ValueError:
                        declared_bytes = 0
                    if declared_bytes > settings.fetch_max_bytes:
                        return (
                            "ERROR: payload_too_large: exceeds "
                            f"{settings.fetch_max_bytes} bytes"
                        )

                body_bytes = await read_capped_body(resp, settings.fetch_max_bytes)
                if body_bytes is None:
                    return (
                        "ERROR: payload_too_large: exceeds "
                        f"{settings.fetch_max_bytes} bytes"
                    )

                final_url = str(resp.url)

            html_text = body_bytes.decode(resp.encoding or "utf-8", errors="replace")
            title = extract_title(html_text, final_url)

            # 6. Trafilatura extraction runs off the event loop (H-04).
            try:
                cleaned_md = await asyncio.wait_for(
                    asyncio.to_thread(extract_markdown, html_text, final_url),
                    timeout=settings.fetch_timeout_seconds,
                )
            except TimeoutError:
                return f"ERROR: extraction_timeout: exceeded {settings.fetch_timeout_seconds}s"

            # 7. Cache the same capped text that can be returned (H-06).
            cached_md = cleaned_md[: settings.fetch_max_chars]
            await cache.set_page_cache(
                settings.app_db_path,
                clean_url,
                final_url,
                title,
                cached_md,
                ttl_hours=settings.page_cache_ttl_hours,
            )

            # 8. Register in source collector
            collector = current_sources.get()
            idx = collector.add(final_url, title) if collector else 1

            return (
                f"[{idx}] {title}\nURL: {final_url}\n\n"
                f"<untrusted_web_content>\n{cached_md[:max_chars]}\n</untrusted_web_content>"
            )

        except httpx.TimeoutException:
            return f"ERROR: fetch_timeout: request exceeded {settings.fetch_timeout_seconds}s"
        except httpx.HTTPError as exc:
            return f"ERROR: http_error: {exc}"

    return "ERROR: too_many_redirects: exceeded 3 hops"
