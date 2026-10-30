import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_web_ui_serves_html():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Root serves index.html
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "<title>Knight - Local AI Assistant</title>" in resp.text
        assert 'data-theme="dark"' in resp.text
        assert 'src="/theme-bootstrap.js"' in resp.text
        assert 'src="/app.js"' in resp.text
        assert "search-conversations-input" in resp.text
        assert "fork-chat-btn" in resp.text
        assert "export-chat-btn" in resp.text
        assert "settings-dialog" in resp.text
        assert "confirm-dialog" in resp.text
        assert "export-dialog" in resp.text
        assert "presets-chip-bar" in resp.text
        assert "enable-thinking-toggle" in resp.text
        assert "top-p-slider" in resp.text
        assert "top-k-slider" in resp.text
        assert "max-tokens-slider" in resp.text
        assert "presence-penalty-slider" in resp.text
        assert "frequency-penalty-slider" in resp.text
        assert "shortcuts-dialog" in resp.text
        assert "attach-file-btn" in resp.text
        assert "attachments-tray" in resp.text
        assert "model-select" in resp.text
        assert "hardware-hud-btn" in resp.text
        assert "hardware-dialog" in resp.text
        assert "open-profiles-btn" in resp.text
        assert "profile-select" in resp.text
        assert "profiles-dialog" in resp.text
        assert "open-ggufbook-btn" in resp.text
        assert "ggufbook-dialog" in resp.text
        assert "ggufbook-search-input" in resp.text
        assert "ggufbook-repos-container" in resp.text
        assert "ggufbook-variants-view" in resp.text
        assert "ggufbook-downloads-drawer" in resp.text
        assert "mic-btn" in resp.text
        assert "voice-recording-tray" in resp.text
        assert "settings-asr-badge" in resp.text
        assert "settings-asr-download-btn" in resp.text
        assert "inchat-search-bar" in resp.text
        assert "inchat-search-toggle-btn" in resp.text
        assert "slash-commands-menu" in resp.text
        assert "sound-effects-toggle" in resp.text

        # 2. Security headers (CSP, etc.) and stylesheet
        assert "Content-Security-Policy" in resp.headers
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
        resp_css = await client.get("/styles.css")
        assert resp_css.status_code == 200
        assert "--font-sans" in resp_css.text
        assert ".modal-xl" in resp_css.text
        assert ".modal-confirm" in resp_css.text
        assert ".modal-export" in resp_css.text
        assert ".compat-optimal" in resp_css.text
        assert ".voice-recording-tray" in resp_css.text
        assert ".floating-circle-btn.mic-btn" in resp_css.text
        assert ".inchat-search-bar" in resp_css.text
        assert ".slash-commands-menu" in resp_css.text
        assert ".msg-telemetry-badge" in resp_css.text
        assert ".code-card-header" in resp_css.text

        # 3. JavaScript app
        resp_js = await client.get("/app.js")
        assert resp_js.status_code == 200
        assert "Knight Web Client" in resp_js.text
        assert "showConfirmDialog" in resp_js.text
        assert "openExportDialog" in resp_js.text
        assert "searchGGUFBook" in resp_js.text
        assert "dispatchGGUFDownload" in resp_js.text
        assert "startVoiceRecording" in resp_js.text
        assert "encodeWAV" in resp_js.text
        assert "SLASH_COMMANDS" in resp_js.text
        assert "performInChatSearch" in resp_js.text
        assert "SoundEffects" in resp_js.text
        assert "togglePinConversation" in resp_js.text
        assert 'String(val ?? "")' in resp_js.text

        # Frontend contracts (U-01 through U-07)
        assert "t.task_id || t.id" in resp_js.text
        assert "t.progress_pct ?? t.percentage" in resp_js.text
        assert "t.speed_mb_s ?? t.speed_mbps" in resp_js.text
        assert "t.total_bytes ?? t.size_bytes" in resp_js.text
        assert "v.already_downloaded ?? v.is_downloaded" in resp_js.text
        assert "fit.label || fit.badge" in resp_js.text
        assert "Cannot cancel: missing task ID" in resp_js.text
        assert "BEFORE sanitization" in resp_js.text
        assert 'ADD_ATTR: ["target", "rel", "class", "title"]' in resp_js.text

        # 4. Vendored marked.min.js
        resp_marked = await client.get("/vendor/marked.min.js")
        assert resp_marked.status_code == 200
        assert len(resp_marked.text) > 5000

        # 5. Vendored purify.min.js
        resp_purify = await client.get("/vendor/purify.min.js")
        assert resp_purify.status_code == 200
        assert len(resp_purify.text) > 5000

        # 6. Vendored manifest and integrity tags
        assert 'integrity="sha256-k04+Nuni2gr7Gm51B1uw8JrwUpOoROhKdHfvQJEcNJo="' in resp.text
        assert 'integrity="sha256-jrQbZYgx+rF1+tm80A/LLYTg7TolpVBT1OzUREuLQ6A="' in resp.text
        assert 'content="default-src \'self\';' in resp.text
        resp_manifest = await client.get("/vendor/manifest.json")
        assert resp_manifest.status_code == 200
        assert "marked.min.js" in resp_manifest.text


def test_web_ui_sends_authorization_on_every_api_call():
    """C-20: with KNIGHT_AUTH_TOKEN set, the UI must authenticate every /api call."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "app" / "web" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "async function apiFetch" in js
    assert 'headers.set("Authorization", `Bearer ${token}`)' in js
    assert "AUTH_TOKEN_KEY" in js
    assert "await fetch(" not in js
    assert js.count("await apiFetch(") >= 20

