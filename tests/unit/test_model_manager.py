"""Regression tests for C-13/C-14/C-15/C-16: model swap safety and argv."""

import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import manager as manager_module
from app.llm.gate import InferenceGate, inference_gate
from app.llm.llama_args import build_llama_argv
from app.llm.manager import ModelManager, resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path / "data", llama_api_key="test-llama-key")
    settings.ensure_dirs()
    return settings


def test_canonical_argv_contains_all_required_flags():
    """C-14: the single builder emits every plan §18 flag."""
    argv = build_llama_argv(
        llama_bin=Path("/opt/llama/llama-server"),
        model_path=Path("/models/test.gguf"),
        port=8080,
        api_key="secret-key",
        settings=Settings(),
    )

    for flag in (
        "--jinja",
        "--no-webui",
        "--no-context-shift",
        "--api-key",
        "--cors-origins",
        "--chat-template-kwargs",
    ):
        assert flag in argv

    assert argv[argv.index("-c") + 1] == "16384"
    assert argv[argv.index("--port") + 1] == "8080"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--api-key") + 1] == "secret-key"
    assert argv[argv.index("--chat-template-kwargs") + 1] == '{"enable_thinking": false}'


def test_canonical_argv_refuses_zero_context_and_empty_key():
    with pytest.raises(ValueError, match="Context tokens"):
        build_llama_argv(
            llama_bin=Path("/opt/llama/llama-server"),
            model_path=Path("/models/test.gguf"),
            port=8080,
            api_key="secret-key",
            settings=Settings(),
            context_tokens=0,
        )

    with pytest.raises(ValueError, match="API key"):
        build_llama_argv(
            llama_bin=Path("/opt/llama/llama-server"),
            model_path=Path("/models/test.gguf"),
            port=8080,
            api_key="   ",
            settings=Settings(),
        )


def test_runner_and_manager_share_the_same_argv_builder():
    """C-14: no second, drifting copy of the launch flags."""
    for relative in ("scripts/run_llama.py", "app/llm/manager.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "build_llama_argv" in source, relative


@pytest.mark.asyncio
async def test_inference_gate_drains_and_reopens():
    """C-13: swaps can wait for in-flight work instead of killing it."""
    gate = InferenceGate()
    assert await gate.wait_idle(0.01) is True

    gate.enter()
    assert gate.active == 1
    assert await gate.wait_idle(0.01) is False

    gate.leave()
    assert gate.active == 0
    assert await gate.wait_idle(1.0) is True


def test_resolve_model_path_rejects_escapes_and_non_gguf(tmp_path):
    """C-16: only bare .gguf filenames inside models_dir are accepted."""
    settings = make_settings(tmp_path)

    for name in (
        "",
        "   ",
        "../escape.gguf",
        "..\\escape.gguf",
        "/tmp/escape.gguf",
        "sub/escape.gguf",
        "model.bin",
    ):
        with pytest.raises(ValueError):
            resolve_model_path(settings, name)

    valid = settings.models_dir / "valid.gguf"
    valid.write_bytes(b"GGUF")
    assert resolve_model_path(settings, "valid.gguf") == valid.resolve()


@pytest.mark.skipif(
    platform.system().lower() == "windows",
    reason="symlink creation requires privileges on Windows",
)
def test_resolve_model_path_rejects_symlink_escape(tmp_path):
    settings = make_settings(tmp_path)
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF")
    (settings.models_dir / "link.gguf").symlink_to(outside)

    with pytest.raises(ValueError, match="outside the models directory"):
        resolve_model_path(settings, "link.gguf")


@pytest.mark.asyncio
async def test_switch_updates_runtime_state_without_mutating_settings(
    tmp_path, monkeypatch
):
    """C-15: a successful swap records the active model outside Settings."""
    settings = make_settings(tmp_path)
    (settings.models_dir / "new.gguf").write_bytes(b"GGUF")
    fake_app = SimpleNamespace(
        state=SimpleNamespace(checkpointer=object(), active_model=settings.model_name)
    )

    class FakeProc:
        def poll(self):
            return None

    monkeypatch.setattr(
        manager_module, "find_llama_bin", lambda *args, **kwargs: Path("/fake/llama-server")
    )
    monkeypatch.setattr(manager_module, "terminate_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(
        manager_module.subprocess, "Popen", lambda argv, **kwargs: FakeProc()
    )

    async def ready(*args, **kwargs):
        return True

    monkeypatch.setattr(manager_module, "wait_for_server_health", ready)
    monkeypatch.setattr(manager_module, "get_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(manager_module, "build_agent", lambda **kwargs: object())

    try:
        result = await ModelManager.switch_model(settings, "new.gguf", fake_app)
        active_model = ModelManager.get_active_model(settings)
    finally:
        ModelManager._active_model = None
        ModelManager._active_server_proc = None

    assert result["ok"] is True
    assert settings.model_name == "qwen3.5-4b"
    assert fake_app.state.active_model == "new.gguf"
    assert active_model == "new.gguf"


@pytest.mark.asyncio
async def test_switch_refuses_while_inference_is_active(tmp_path, monkeypatch):
    """C-13: an active turn yields a typed model_swapping error, not a killed stream."""
    settings = make_settings(tmp_path)
    (settings.models_dir / "busy.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(manager_module, "DRAIN_TIMEOUT_SECONDS", 0.01)

    baseline = inference_gate.active
    inference_gate.enter()
    try:
        result = await ModelManager.switch_model(
            settings,
            "busy.gguf",
            SimpleNamespace(state=SimpleNamespace()),
        )
    finally:
        inference_gate.leave()

    assert result["ok"] is False
    assert result["error"] == "model_swapping"
    assert inference_gate.active == baseline
